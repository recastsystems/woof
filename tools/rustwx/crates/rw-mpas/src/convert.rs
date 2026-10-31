//! Gather one MPAS history frame onto a render window and write it as a
//! wrfout-shaped classic netCDF frame.
//!
//! The renderer has no unstructured path. Its real named products -- the ones
//! with operational colortables, titles and map furniture -- come from the
//! real-WRF import route, which needs a structured grid, `Times`,
//! `XLAT`/`XLONG` and a rank-4 `T`. This module produces exactly that, and
//! nothing more: no field is invented, and every emitted file carries the
//! source history digest, the mesh digest, the weights digest, the resample
//! method and the engine that wrote it.

use std::path::{Path, PathBuf};

use rw_store::netcdf_classic::{
    NcAttr, NcData, NcDim, NcFormat, NcType, NcVarDef,
};

#[path = "render_netcdf4.rs"]
mod render_netcdf4;
use render_netcdf4::RenderFrameWriter as NcClassicWriter;
pub use render_netcdf4::{render_frame_compression, RenderCompression};

use crate::composite::{COMPOSITE_SCHEMA, CompositePlan, SeamStat};
use crate::error::{MpasError, MpasResult};
use crate::fieldmap::{FIELD_MAP, FieldSource, Rank, field_set_allows, field_set_is_known, remap_class};
use crate::remap::{PRODUCT_REMAP_METHOD, ProductRemap};
use crate::history::{MpasFrame, Timestamp, expected_levels};
use crate::weights::NearestCellWeights;
use crate::window::ProjAttr;

/// The two extra fields a composed frame carries.  Named so they sort
/// beside the WRF furniture and render through the generic `var:` path.
pub const COMPOSITE_SOURCE_VAR: &str = "COMPOSITE_SOURCE";
pub const COMPOSITE_SPACING_VAR: &str = "COMPOSITE_DX_KM";

pub const BRIDGE_SCHEMA: &str = "mpas-port.history-to-wrfout-render-frame/v1";

/// Native mass winds bypass the ordinary WRF face-grid round-trip smoothing.
pub const RADAR_NATIVE_WINDS: &str = "earth-relative-mass-grid/v1";
pub const RADAR_U_EARTH: &str = "RADAR_U_EARTH";
pub const RADAR_V_EARTH: &str = "RADAR_V_EARTH";

/// WRF's `T` is a perturbation about this reference potential temperature,
/// and wrf-core reconstructs full theta as `T + 300`.
pub const WRF_THETA_REFERENCE_K: f32 = 300.0;
/// wrf-core divides full geopotential by exactly this to get height.
pub const WRF_GRAVITY: f32 = 9.80665;

const MMINLU: &str = "MODIFIED_IGBP_MODIS_NOAH";

/// What one conversion produced.
#[derive(Debug, Clone)]
pub struct EmittedFrame {
    pub path: PathBuf,
    pub window: String,
    pub shape: (usize, usize),
    pub levels: usize,
    pub written: Vec<String>,
    pub absent: Vec<String>,
    pub seconds: f64,
    pub bytes: u64,
    pub valid_time: Timestamp,
    pub simulation_start: Timestamp,
    pub lead_seconds: i64,
    /// How far the two runs disagree across the seam, per surface field.
    /// Empty for a single-source conversion, which has no seam.
    pub seam: Vec<SeamStat>,
}

/// Knobs the caller sets; everything else is fixed by the schema.
#[derive(Debug, Clone)]
pub struct ConvertOptions {
    pub simulation_start: Option<Timestamp>,
    pub field_set: String,
    pub grid_id: i32,
    pub title: String,
    /// The producing model's name, written as `GPUWM_MODEL_LABEL` for the
    /// renderer's metadata row.  `None` writes nothing, and the frame's
    /// plots carry the generic wrfout identity (`WRF`) as before.
    pub model_label: Option<String>,
    /// The grid the plots should name in their headline, written as
    /// `GPUWM_GRID_LABEL` (`hex 15 km`).  `None` writes nothing, and the
    /// renderer falls back to the WRF domain token (`d01 15 km`).
    pub grid_label: Option<String>,
    pub format: NcFormat,
    pub clobber: bool,
    /// A fixed geographic terrain plane for independently meshed segments.
    pub terrain_authority: Option<std::sync::Arc<TerrainAuthority>>,
}

#[derive(Debug)]
pub struct TerrainAuthority {
    pub path: PathBuf,
    pub sha256: String,
    values: Vec<f32>,
}

impl TerrainAuthority {
    /// The authority is itself an exported CF lat/lon frame on this exact
    /// target grid. Atmospheric columns retain their absolute heights;
    /// this binds source orography, not a balanced model restart.
    pub fn read(path: &Path, weights: &NearestCellWeights) -> MpasResult<Self> {
        let file = netcrust::File::open(path)?;
        let (ny,nx) = weights.shape();
        let read = |name: &str, units: &str| -> MpasResult<(Vec<f64>,Vec<usize>)> {
            let var = file.variable(name).ok_or_else(||MpasError::Refusal(format!("terrain authority lacks {name}")))?;
            if var.attribute("units").and_then(|a|a.as_string()) != Some(units)
                || var.attribute("scale_factor").is_some() || var.attribute("add_offset").is_some() {
                return Err(MpasError::Refusal(format!("terrain authority {name}: expected unpacked {units}")));
            }
            let array = var.array_f64()?;
            let shape = array.shape().to_vec();
            let values = array.into_values();
            if values.iter().any(|v| !v.is_finite()
                || var.attribute("_FillValue").and_then(|a|a.as_f64()) == Some(*v)
                || var.attribute("missing_value").and_then(|a|a.as_f64()) == Some(*v)) {
                return Err(MpasError::Refusal(format!("terrain authority {name}: missing values")));
            }
            Ok((values,shape))
        };
        let (lat,ls) = read("lat","degrees_north")?;
        let (lon,os) = read("lon","degrees_east")?;
        let (values,shape) = read("HGT","m")?;
        if ls != [ny] || os != [nx] || (shape != [ny,nx] && shape != [1,ny,nx])
            || weights.target_latitude.iter().enumerate().any(|(i,v)| *v != lat[i/nx])
            || weights.target_longitude.iter().enumerate().any(|(i,v)| *v != lon[i%nx]) {
            return Err(MpasError::Refusal("terrain authority must be on the identical CF lat/lon target grid".into()));
        }
        let values: Vec<f32> = values.into_iter().map(|v|v as f32).collect();
        if values.iter().any(|v|!v.is_finite()) {
            return Err(MpasError::Refusal("terrain authority overflows float32".into()));
        }
        Ok(Self{path:std::fs::canonicalize(path)?, sha256:crate::sha256_file(path)?, values})
    }
}

/// The global attribute the renderer reads the model's name from
/// (`rustwx_products::shared_context::MODEL_LABEL_ATTRIBUTE`; this crate
/// does not depend on that one, so the spelling is stated in both and a
/// test in each pins it).  A frame this converter writes is a mesh
/// forecast resampled onto a regular grid, imported as `wrf` like any
/// wrfout, so without it every one of its maps said `WRF`.
pub const MODEL_LABEL_ATTRIBUTE: &str = "GPUWM_MODEL_LABEL";

/// The global attribute the renderer reads a plot headline's grid name
/// from (`rw_wrfbatch`'s `GRID_LABEL_ATTRIBUTE` pins the same literal).
///
/// THE BREAKAGE THIS PREVENTS, MEASURED (2026-10-10): every frame this
/// converter wrote carried only WRF's `GRID_ID = 1` and the render grid's
/// `DX`, so every hex map was headlined `(d01 20 km)` -- a WRF nest number
/// the model does not have and the spacing of a resample grid instead of
/// the model's own mesh. The converter is the one place that has read the
/// mesh, so it says which mesh it was.
pub const GRID_LABEL_ATTRIBUTE: &str = "GPUWM_GRID_LABEL";

/// The global attribute that says what a frame's `REFC` holds beyond the
/// microphysics column maximum (`rw_wrfbatch`'s `REFC_NOTE_ATTRIBUTE` pins
/// the same literal and adds the note to the composite reflectivity map's
/// title).  Written on a frame that carries the cumulus scheme's rain rate
/// (`CUPREC`), whose reflectivity `REFC` then includes
/// (`fieldmap::add_convective_reflectivity`).
///
/// THE BREAKAGE THIS PREVENTS: on a coarse hex row the echoes a viewer sees
/// are mostly the Z-R reflectivity of parameterised rain, not hydrometeors
/// the model carried, and nothing on the map said so (WOOF review,
/// 2026-10-29: run_9a0796b2e9c8e251's 20Z map drew 20-30 dBZ over
/// Minnesota from the cumulus scheme's rain alone).
pub const REFC_NOTE_ATTRIBUTE: &str = "GPUWM_REFC_NOTE";

/// The note itself, as the map title prints it.
pub const REFC_CONVECTIVE_NOTE: &str = "incl. parameterised convection";

/// The longest grid label the renderer's headline takes (the same ceiling
/// as the model label's metadata row).
pub const GRID_LABEL_MAX_CHARS: usize = 32;

/// Kilometres as a mesh is named: whole kilometres from 10 km up, one
/// trimmed decimal below (`15`, `4.5`, `3`).
fn km_token(metres: f64) -> String {
    let km = metres / 1000.0;
    if km >= 10.0 {
        format!("{}", km.round() as i64)
    } else {
        let text = format!("{:.1}", (km * 10.0).round() / 10.0);
        text.trim_end_matches('0').trim_end_matches('.').to_string()
    }
}

/// The spacing a mesh is known by, from its own per-cell spacing:
/// `15 km` for a uniform mesh (its median), `96-3 km` for a graded one
/// (background to refined region).  Uniform is the refined-region rule's
/// own test -- coarsest within [`crate::window::MESH_FOCUS_REFINEMENT_FACTOR`]
/// of the finest -- so a mesh this calls uniform is one `--window mesh`
/// refuses as having no refined region, and the two never disagree.
///
/// A graded mesh is named by its two REGIONS, not its two most distorted
/// cells: the refined end is the median of the cells within the factor of
/// the finest (the very set `--window mesh` frames and takes its `dx`
/// from), the background end the median of the cells within the factor of
/// the coarsest.  THE BREAKAGE THIS PREVENTS, MEASURED (2026-10-28): the
/// 96-to-3 km zoom globe (r700, 286,530 cells, per-cell spacing 2.51 to
/// 100.45 km, refined median 3.04 km) was headlined `100-2.5 km` from its
/// single extreme cells, a mesh nobody built.
pub fn mesh_spacing_label(cell_spacing_m: &[f64]) -> Option<String> {
    let finite: Vec<f64> = cell_spacing_m
        .iter()
        .copied()
        .filter(|s| s.is_finite() && *s > 0.0)
        .collect();
    if finite.is_empty() || finite.len() != cell_spacing_m.len() {
        return None;
    }
    let min = finite.iter().copied().fold(f64::INFINITY, f64::min);
    let max = finite.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let factor = crate::window::MESH_FOCUS_REFINEMENT_FACTOR;
    if max <= factor * min {
        Some(format!("{} km", km_token(median(finite))))
    } else {
        let fine: Vec<f64> = finite.iter().copied().filter(|s| *s <= factor * min).collect();
        let coarse: Vec<f64> = finite.into_iter().filter(|s| *s * factor >= max).collect();
        Some(format!("{}-{} km", km_token(median(coarse)), km_token(median(fine))))
    }
}

/// The upper median of a non-empty set (the element a sort puts at `len / 2`).
fn median(mut values: Vec<f64>) -> f64 {
    let mid = values.len() / 2;
    values.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
    values[mid]
}

/// `MODEL SPACING` (`hex 15 km`), or `mesh SPACING` when no model was
/// named, held to the headline's limits; `None` when the mesh carries no
/// usable spacing.
pub fn grid_label(model_label: Option<&str>, cell_spacing_m: &[f64]) -> Option<String> {
    let spacing = mesh_spacing_label(cell_spacing_m)?;
    let model = model_label
        .map(|m| m.split_whitespace().collect::<Vec<_>>().join(" "))
        .filter(|m| !m.is_empty())
        .unwrap_or_else(|| "mesh".to_string());
    let label = format!("{model} {spacing}");
    (label.chars().count() <= GRID_LABEL_MAX_CHARS
        && !label.chars().any(|ch| ch.is_control() || ch == '|'))
    .then_some(label)
}

/// The token a caller's grid label (`rw_mpas_convert --grid-label`) may
/// carry where the spacing THIS PICTURE shows goes ([`window_spacing_label`]):
/// `Core zoom {spacing}` -> `Core zoom 8-38 km`.
pub const GRID_LABEL_SPACING_TOKEN: &str = "{spacing}";

/// How close to the coarsest sampled cell a cell must be to count as the
/// picture's coarse end: the coarse end is the median of that band, so one
/// distorted cell does not name it (the same reason [`mesh_spacing_label`]
/// names a mesh by its regions).
pub const WINDOW_COARSE_BAND: f64 = 1.1;

/// The spacing a PICTURE shows, from the mesh cells its window samples
/// (`sampled`, indices into `cell_spacing_m`): finest first, as a menu reads
/// a range -- `8-38 km` for a zoom globe's core picture that reaches 38 km
/// cells at its corners, `120 km` on a uniform mesh.
///
/// On a uniform mesh (the whole mesh within
/// [`crate::window::MESH_FOCUS_REFINEMENT_FACTOR`] of its finest, the rule
/// [`mesh_spacing_label`] and `--window mesh` share) it is one value, the
/// median of the sampled cells.  On a variable mesh the fine end is the
/// median of the sampled cells within that factor of the finest sampled
/// (the refined region's own spacing, 3.04 km on a 3 km core, not its 2.5 km
/// extreme cell) and the coarse end the median of the sampled cells within
/// [`WINDOW_COARSE_BAND`] of the coarsest sampled; equal after rounding, one
/// value.  `None` when no sampled cell has a usable spacing.
///
/// THE BREAKAGE THIS PREVENTS (the project owner, 2026-10-30): every view of a zoom globe
/// was headlined by the WHOLE mesh's range, so the core picture of
/// run_713c9e1913164e02 read `(hex 133-6.9 km)` under a site menu entry
/// `Core zoom · 8 → 38 km`, and the view around the core of
/// run_c108e049fecb4bd0 showed `Δx 6.08 km` under `Around the core · 6 → 96 km`.
pub fn window_spacing_label(cell_spacing_m: &[f64], sampled: &[usize]) -> Option<String> {
    let usable = |s: f64| s.is_finite() && s > 0.0;
    let values: Vec<f64> = sampled
        .iter()
        .filter_map(|&i| cell_spacing_m.get(i).copied())
        .filter(|s| usable(*s))
        .collect();
    if values.is_empty() {
        return None;
    }
    let factor = crate::window::MESH_FOCUS_REFINEMENT_FACTOR;
    let (mesh_min, mesh_max) = cell_spacing_m
        .iter()
        .copied()
        .filter(|s| usable(*s))
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), s| (lo.min(s), hi.max(s)));
    if mesh_max <= factor * mesh_min {
        return Some(format!("{} km", km_token(median(values))));
    }
    let min = values.iter().copied().fold(f64::INFINITY, f64::min);
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let fine = km_token(median(values.iter().copied().filter(|s| *s <= factor * min).collect()));
    let coarse = km_token(median(
        values.into_iter().filter(|s| *s * WINDOW_COARSE_BAND >= max).collect(),
    ));
    Some(if fine == coarse { format!("{fine} km") } else { format!("{fine}-{coarse} km") })
}

/// A caller's grid label (`rw_mpas_convert --grid-label`) as the headline
/// prints it: whitespace collapsed, [`GRID_LABEL_SPACING_TOKEN`] replaced by
/// `spacing`, then held to the renderer's headline limits (at most
/// [`GRID_LABEL_MAX_CHARS`] characters, no control character, no `|`).
/// An error names what broke, so a caller's text is never silently dropped.
pub fn caller_grid_label(text: &str, spacing: Option<&str>) -> Result<String, String> {
    let collapsed = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let label = if collapsed.contains(GRID_LABEL_SPACING_TOKEN) {
        let spacing = spacing.ok_or_else(|| {
            format!(
                "--grid-label {collapsed:?} asks for {GRID_LABEL_SPACING_TOKEN}, and no cell the \
                 window samples carries a usable spacing"
            )
        })?;
        collapsed.replace(GRID_LABEL_SPACING_TOKEN, spacing)
    } else {
        collapsed
    };
    if label.is_empty() {
        return Err("--grid-label needs a name".to_string());
    }
    if label.chars().count() > GRID_LABEL_MAX_CHARS {
        return Err(format!(
            "--grid-label {label:?} is longer than {GRID_LABEL_MAX_CHARS} characters, the plot headline's limit"
        ));
    }
    if label.chars().any(|ch| ch.is_control() || ch == '|') {
        return Err(format!(
            "--grid-label {label:?} holds a control character or '|', the plot metadata row's separator"
        ));
    }
    Ok(label)
}

impl Default for ConvertOptions {
    fn default() -> Self {
        ConvertOptions {
            simulation_start: None,
            field_set: "full".to_string(),
            grid_id: 1,
            title: "MPAS-A v8.4.1 CUDA port history resampled for rw_wrfbatch".to_string(),
            model_label: None,
            grid_label: None,
            format: NcFormat::Offset64,
            clobber: false,
            terrain_authority: None,
        }
    }
}

/// Cell-centre values to x-staggered faces, WRF `U` layout.
///
/// The renderer destaggers with a two-point average, so the round trip is a
/// 1/4-1/2-1/4 smoother in x rather than the identity. That is a real and
/// deliberate divergence: WRF's `U` genuinely lives on faces and there is no
/// local face field whose two-point average reproduces an arbitrary
/// cell-centre field. It touches the three-dimensional wind only; the surface
/// wind products read the unstaggered `U10`/`V10` and are untouched.
fn restagger_x(values: &[f32], levels: usize, ny: usize, nx: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; levels * ny * (nx + 1)];
    for level in 0..levels {
        for row in 0..ny {
            let src = &values[(level * ny + row) * nx..(level * ny + row + 1) * nx];
            let dst = &mut out[(level * ny + row) * (nx + 1)..(level * ny + row + 1) * (nx + 1)];
            dst[0] = src[0];
            for i in 1..nx {
                dst[i] = (src[i - 1] + src[i]) * 0.5;
            }
            dst[nx] = src[nx - 1];
        }
    }
    out
}

/// Cell-centre values to y-staggered faces, WRF `V` layout.
fn restagger_y(values: &[f32], levels: usize, ny: usize, nx: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; levels * (ny + 1) * nx];
    for level in 0..levels {
        let src = &values[level * ny * nx..(level + 1) * ny * nx];
        let dst = &mut out[level * (ny + 1) * nx..(level + 1) * (ny + 1) * nx];
        dst[..nx].copy_from_slice(&src[..nx]);
        for row in 1..ny {
            for i in 0..nx {
                dst[row * nx + i] = (src[(row - 1) * nx + i] + src[row * nx + i]) * 0.5;
            }
        }
        dst[ny * nx..].copy_from_slice(&src[(ny - 1) * nx..ny * nx]);
    }
    out
}

/// The per-variable attributes every emitted field carries.
/// The time unit of the CF `Time` coordinate a lat-lon frame carries.
/// Fixed, so every frame of every run states its valid time on one epoch
/// and a reader's mapping can pin the unit text.
pub const CF_TIME_UNITS: &str = "seconds since 1970-01-01 00:00:00";

/// CF 1-D coordinates of a regular lat-lon frame.
///
/// THE BREAKAGE THIS PREVENTS: a frame carried only WRF's 2-D `XLAT`/`XLONG`
/// and dimension names with no coordinate variables, so the engine's mapped
/// NetCDF route ("coordinate must be finite non-empty 1-D") could not read a
/// hex frame at all, and no ArWen domain could take hex as its parent.  A
/// Lambert frame has no 1-D axes and gets none.
struct CfCoordinates {
    latitude: Vec<f64>,
    longitude: Vec<f64>,
    /// Soil layer TOP depths in cm, from the init's `dzs`; `None` when the
    /// init is absent or its layer thicknesses differ between columns.
    soil_tops_cm: Option<Vec<f64>>,
}

fn cf_coordinates(
    weights: &NearestCellWeights,
    frame: &MpasFrame,
    ny: usize,
    nx: usize,
) -> MpasResult<Option<CfCoordinates>> {
    if !matches!(weights.window, crate::window::Window::LatLon(_)) {
        return Ok(None);
    }
    if weights.target_latitude.len() != ny * nx || weights.target_longitude.len() != ny * nx {
        return Err(MpasError::Refusal(
            "lat-lon window coordinates do not cover its grid".to_string(),
        ));
    }
    let latitude: Vec<f64> = (0..ny).map(|j| weights.target_latitude[j * nx]).collect();
    let longitude: Vec<f64> = weights.target_longitude[..nx].to_vec();
    // A regular grid or nothing: a 1-D axis that does not describe every row
    // and column would georeference the frame wrong without a word.
    for j in 0..ny {
        for i in 0..nx {
            if weights.target_latitude[j * nx + i] != latitude[j]
                || weights.target_longitude[j * nx + i] != longitude[i]
            {
                return Err(MpasError::Refusal(format!(
                    "lat-lon window point ({j}, {i}) is off its row and column axes"
                )));
            }
        }
    }
    let soil_tops_cm = match (&frame.init_path, frame.n_soil_levels) {
        (Some(path), n) if n > 0 => soil_layer_tops_cm(path, n)?,
        _ => None,
    };
    Ok(Some(CfCoordinates { latitude, longitude, soil_tops_cm }))
}

/// Soil layer top depths (cm) from the init's `dzs(Time, nCells, nSoilLevels)`,
/// rounded to 1e-4 cm so FP32 thickness sums (0.1 + 0.3 = 0.4000000059 m)
/// land on the depths they state.  `None` unless every column carries the
/// same thicknesses: one coordinate cannot describe two soil columns.
fn soil_layer_tops_cm(init: &Path, n_soil: usize) -> MpasResult<Option<Vec<f64>>> {
    let file = netcrust::File::open(init)?;
    if file.variable("dzs").is_none() {
        return Ok(None);
    }
    let values = file.read_array_f64("dzs")?.into_values();
    if values.len() < n_soil || values.len() % n_soil != 0 {
        return Ok(None);
    }
    let first = values[..n_soil].to_vec();
    if values.chunks(n_soil).any(|column| column != first.as_slice())
        || first.iter().any(|v| !v.is_finite() || *v <= 0.0)
    {
        return Ok(None);
    }
    let mut top_m = 0.0f64;
    let mut tops = Vec::with_capacity(n_soil);
    for thickness in first {
        tops.push((top_m * 100.0 * 1.0e4).round() / 1.0e4);
        top_m += thickness;
    }
    Ok(Some(tops))
}

fn field_attrs(rank_is_3d: bool, description: &str, units: &str, stagger: &str) -> Vec<NcAttr> {
    vec![
        NcAttr::int("FieldType", 104),
        NcAttr::text("MemoryOrder", if rank_is_3d { "XYZ" } else { "XY " }),
        NcAttr::text("description", description),
        NcAttr::text("units", units),
        NcAttr::text("stagger", stagger),
        NcAttr::text("coordinates", "XLONG XLAT XTIME"),
    ]
}

/// Write one wrfout-shaped frame. Returns the emission record.
/// One fine run's frame and the weights that put it on the composite
/// window.  Index 0 of the layer list is the base.
pub struct CompositeLayer<'a> {
    pub label: &'a str,
    pub frame: &'a MpasFrame,
    pub weights: &'a NearestCellWeights,
}

/// Everything the composite gather needs: who owns each point, and the
/// frames to read from.
pub struct CompositeGather<'a> {
    pub plan: &'a CompositePlan,
    pub layers: &'a [CompositeLayer<'a>],
}

impl CompositeGather<'_> {
    /// One field, assembled point by point from whichever source the plan
    /// gave that point.
    ///
    /// A source that does not carry the field cannot answer for its
    /// points, and those points fall back to the base rather than to a
    /// fill value: a hole in one variable of an otherwise composed frame
    /// is a worse artifact than a coarser value, and the fallback is
    /// reported rather than silent.
    fn gather(
        &self,
        wrf_name: &str,
        levels: usize,
        points: usize,
    ) -> MpasResult<(Vec<f32>, Vec<String>)> {
        let mut out = vec![0.0f32; levels * points];
        let mut fell_back: Vec<String> = Vec::new();
        fn field_of<'f>(
            layer: &CompositeLayer<'f>,
            wrf_name: &str,
        ) -> Option<&'f (Vec<f32>, usize)> {
            layer.frame.fields.get(wrf_name)
        }
        let base = self.layers.first().ok_or_else(|| {
            MpasError::Refusal("a composite gather needs at least the base layer".to_string())
        })?;
        let (base_values, base_levels) = field_of(base, wrf_name).ok_or_else(|| {
            MpasError::Refusal(format!(
                "the composite base does not carry {wrf_name}, though it was planned"
            ))
        })?;
        if *base_levels != levels {
            return Err(MpasError::Refusal(format!(
                "{wrf_name} carries {base_levels} level(s) on the base; the frame declares                  {levels}"
            )));
        }
        for slot in 0..points {
            let mut which = self.plan.source_index[slot] as usize;
            if which >= self.layers.len() {
                return Err(MpasError::Refusal(format!(
                    "the composite plan names source {which} at grid point {slot}, but only                      {} layer(s) were handed over",
                    self.layers.len()
                )));
            }
            if which > 0 {
                match field_of(&self.layers[which], wrf_name) {
                    Some((_, layer_levels)) if *layer_levels == levels => {}
                    _ => {
                        let label = self.layers[which].label.to_string();
                        if !fell_back.contains(&label) {
                            fell_back.push(label);
                        }
                        which = 0;
                    }
                }
            }
            let layer = &self.layers[which];
            let values = if which == 0 {
                base_values
            } else {
                &field_of(layer, wrf_name).expect("checked above").0
            };
            let cell = layer.weights.cell_index[slot] as usize;
            for level in 0..levels {
                out[level * points + slot] = values[cell * levels + level];
            }
        }
        Ok((out, fell_back))
    }
}

pub fn convert_frame(
    frame: &MpasFrame,
    weights: &NearestCellWeights,
    destination: &Path,
    options: &ConvertOptions,
) -> MpasResult<EmittedFrame> {
    convert_frame_inner(frame, weights, None, None, destination, options)
}

/// The single-source conversion through the product remap: each field by
/// its [`crate::fieldmap::remap_class`] (linear state, conservative
/// accumulations, nearest categories).  Same schema and variables as
/// [`convert_frame`]; the frame records the method per variable.
pub fn convert_frame_remapped(
    frame: &MpasFrame,
    remap: &ProductRemap<'_>,
    destination: &Path,
    options: &ConvertOptions,
) -> MpasResult<EmittedFrame> {
    convert_frame_inner(frame, remap.nearest, None, Some(remap), destination, options)
}

/// The same conversion, with the base frame's values replaced by a finer
/// source's wherever the plan says a finer source covers that point.
///
/// `composite: None` is byte-for-byte the single-source conversion; the
/// composite branch is additive and touches nothing on that path.
pub fn convert_frame_composite(
    frame: &MpasFrame,
    weights: &NearestCellWeights,
    composite: Option<&CompositeGather<'_>>,
    destination: &Path,
    options: &ConvertOptions,
) -> MpasResult<EmittedFrame> {
    convert_frame_inner(frame, weights, composite, None, destination, options)
}

/// The frame's `REFC_NOTE_ATTRIBUTE`: the convective note when its REFC took
/// the cumulus scheme's reflectivity (it carries both `REFC` and `CUPREC`;
/// the history reader adds the one to the other whenever both are read).
fn refc_note(frame: &MpasFrame) -> Option<&'static str> {
    (frame.fields.contains_key("REFC") && frame.fields.contains_key("CUPREC"))
        .then_some(REFC_CONVECTIVE_NOTE)
}

fn convert_frame_inner(
    frame: &MpasFrame,
    weights: &NearestCellWeights,
    composite: Option<&CompositeGather<'_>>,
    remap: Option<&ProductRemap<'_>>,
    destination: &Path,
    options: &ConvertOptions,
) -> MpasResult<EmittedFrame> {
    if composite.is_some() && remap.is_some() {
        return Err(MpasError::Refusal(
            "a composite frame switches source point by point and has no single mesh to              interpolate across, so it cannot take the product remap"
                .to_string(),
        ));
    }
    let started = std::time::Instant::now();
    if !field_set_is_known(&options.field_set) {
        return Err(MpasError::Refusal(format!(
            "unknown field set '{}' (known: full, surface)",
            options.field_set
        )));
    }
    if destination.exists() && !options.clobber {
        return Err(MpasError::Refusal(format!(
            "{} exists; pass --clobber to replace it",
            destination.display()
        )));
    }
    if let Some(parent) = destination.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let (ny, nx) = weights.shape();
    let nz = frame.n_levels;
    let origin = options.simulation_start.unwrap_or(frame.valid_time);
    let lead_seconds = frame.valid_time.seconds_since(&origin);
    if lead_seconds < 0 {
        return Err(MpasError::Refusal(
            "simulation start follows the frame valid time".to_string(),
        ));
    }

    // --- dimensions ---------------------------------------------------------
    let mut dims = vec![
        NcDim::record("Time"),
        NcDim::fixed("DateStrLen", 19),
        NcDim::fixed("west_east", nx),
        NcDim::fixed("south_north", ny),
        NcDim::fixed("bottom_top", nz),
        NcDim::fixed("west_east_stag", nx + 1),
        NcDim::fixed("south_north_stag", ny + 1),
        NcDim::fixed("bottom_top_stag", nz + 1),
    ];
    if frame.n_soil_levels > 0 {
        dims.push(NcDim::fixed("soil_layers_stag", frame.n_soil_levels));
    }
    let dimid = |name: &str| -> usize {
        dims.iter()
            .position(|d| d.name == name)
            .expect("dimension declared above")
    };

    // --- global attributes --------------------------------------------------
    let projection = weights.window.projection_attributes()?;
    let mut gattrs: Vec<NcAttr> = vec![
        NcAttr::text("TITLE", options.title.clone()),
        NcAttr::text("SIMULATION_START_DATE", origin.label()),
        NcAttr::text("START_DATE", origin.label()),
        NcAttr::int("WEST-EAST_GRID_DIMENSION", nx as i32 + 1),
        NcAttr::int("SOUTH-NORTH_GRID_DIMENSION", ny as i32 + 1),
        NcAttr::int("BOTTOM-TOP_GRID_DIMENSION", nz as i32 + 1),
        NcAttr::text("GRIDTYPE", "C"),
        NcAttr::text("MMINLU", MMINLU),
        NcAttr::int("ISWATER", 17),
        NcAttr::int("ISLAKE", 21),
        NcAttr::int("ISICE", 15),
        NcAttr::int("ISURBAN", 13),
        NcAttr::int("GRID_ID", options.grid_id),
        NcAttr::int("PARENT_ID", 0),
        NcAttr::int("I_PARENT_START", 1),
        NcAttr::int("J_PARENT_START", 1),
        NcAttr::int("PARENT_GRID_RATIO", 1),
    ];
    if let Some(label) = options.model_label.as_deref() {
        gattrs.push(NcAttr::text(MODEL_LABEL_ATTRIBUTE, label));
    }
    if let Some(label) = options.grid_label.as_deref() {
        gattrs.push(NcAttr::text(GRID_LABEL_ATTRIBUTE, label));
    }
    if let Some(note) = refc_note(frame) {
        gattrs.push(NcAttr::text(REFC_NOTE_ATTRIBUTE, note));
    }
    if let Some(terrain) = &options.terrain_authority {
        if terrain.values.len() != ny*nx {
            return Err(MpasError::Refusal("terrain authority shape changed".into()));
        }
        gattrs.push(NcAttr::text("GPUWM_TERRAIN_AUTHORITY",terrain.path.display().to_string()));
        gattrs.push(NcAttr::text("GPUWM_TERRAIN_AUTHORITY_SHA256",&terrain.sha256));
        gattrs.push(NcAttr::text("GPUWM_TERRAIN_POLICY","HGT bound to a common geographic authority; atmospheric columns/absolute heights unchanged; not a balanced restart"));
    }
    // A composite can declare a single physics identity only when every source
    // agrees. Missing metadata remains missing; no scheme is invented here.
    gattrs.extend(
        frame
            .science_attributes
            .iter()
            .filter(|attribute| {
                composite.is_none_or(|gather| {
                    gather
                        .layers
                        .iter()
                        .all(|layer| layer.frame.science_attributes.contains(attribute))
                })
            })
            .cloned(),
    );
    let radar_winds = options.field_set == "full"
        && frame.fields.contains_key("U")
        && frame.fields.contains_key("V");
    if radar_winds {
        gattrs.push(NcAttr::text("RADAR_NATIVE_WINDS", RADAR_NATIVE_WINDS));
    }
    for (name, value) in &projection {
        gattrs.push(match value {
            ProjAttr::Int(v) => NcAttr::int(name.clone(), *v),
            ProjAttr::Float(v) => NcAttr::float(name.clone(), *v as f32),
            ProjAttr::Text(v) => NcAttr::text(name.clone(), v.clone()),
        });
    }
    let file_name = |p: Option<&PathBuf>| -> String {
        p.and_then(|p| p.file_name())
            .and_then(|n| n.to_str())
            .unwrap_or_default()
            .to_string()
    };
    gattrs.extend([
        NcAttr::text("MPAS_BRIDGE_SCHEMA", BRIDGE_SCHEMA),
        NcAttr::text("MPAS_SOURCE", "MPAS-A v8.4.1 full-physics CUDA port"),
        NcAttr::text(
            "MPAS_HISTORY_FILE",
            file_name(Some(&frame.history_path)),
        ),
        NcAttr::text("MPAS_HISTORY_SHA256", frame.history_sha256.clone()),
        NcAttr::text("MPAS_MESH_SHA256", weights.mesh_sha256.clone()),
        NcAttr::text(
            "MPAS_MESH_FILE",
            file_name(Some(&PathBuf::from(&weights.mesh_path))),
        ),
        NcAttr::text(
            "MPAS_INIT_SHA256",
            frame.init_sha256.clone().unwrap_or_default(),
        ),
        NcAttr::text("MPAS_INIT_FILE", file_name(frame.init_path.as_ref())),
        NcAttr::text("MPAS_WEIGHTS_SHA256", weights.weights_sha256.clone()),
        NcAttr::text("MPAS_WEIGHTS_SCHEMA", crate::weights::WEIGHTS_SCHEMA),
        NcAttr::text(
            "MPAS_RESAMPLE_METHOD",
            if remap.is_some() { PRODUCT_REMAP_METHOD } else { crate::weights::RESAMPLE_METHOD },
        ),
        NcAttr::text("MPAS_MESH_ANGLE_UNITS", weights.mesh_angle_units.label()),
        NcAttr::text("MPAS_RENDER_WINDOW", weights.window_name.clone()),
        NcAttr::text("MPAS_RENDER_WINDOW_SPEC", weights.window.spec_json()),
        NcAttr::int("MPAS_NATIVE_NCELLS", frame.n_cells as i32),
        NcAttr::float("MPAS_MAX_NEAREST_CELL_KM", weights.max_distance_km()),
        NcAttr::text("MPAS_FIELD_SET", options.field_set.clone()),
        NcAttr::text(
            "MPAS_WIND_FRAME",
            "U, V, U10 and V10 are earth-relative; SINALPHA=0 and COSALPHA=1 declare no grid rotation so the renderer's grid-to-earth rotation is the identity",
        ),
        NcAttr::text(
            "MPAS_NONCLAIM",
            if remap.is_some() {
                "product remap for display: accumulations are area-conservative to the reported quadrature error; not a forecast-skill claim"
            } else {
                "nearest-cell visualization resample; not a conservative remap and not a forecast-skill claim"
            },
        ),
        NcAttr::text(
            "MPAS_BRIDGE_ENGINE",
            concat!("rw-mpas ", env!("CARGO_PKG_VERSION"), " (rust)"),
        ),
    ]);
    // A split-stream history (RRFS-MPAS) names where its centres and its
    // diag fields came from.  Written only for such a frame, so a history
    // that carries its own centres converts to the same bytes as before.
    if let Some(split) = &frame.split {
        gattrs.push(NcAttr::text(
            "MPAS_CELL_CENTRES_FILE",
            file_name(Some(&split.coordinates_path)),
        ));
        gattrs.push(NcAttr::text(
            "MPAS_CELL_CENTRES_SHA256",
            split.coordinates_sha256.clone(),
        ));
        gattrs.push(NcAttr::text(
            "MPAS_CELL_CENTRES_AGREED_FIELDS",
            split.agreed_static_fields.join(","),
        ));
        gattrs.push(NcAttr::text("MPAS_DIAG_FILE", file_name(split.diag_path.as_ref())));
        gattrs.push(NcAttr::text(
            "MPAS_DIAG_SHA256",
            split.diag_sha256.clone().unwrap_or_default(),
        ));
        gattrs.push(NcAttr::text("MPAS_DIAG_FIELDS", split.diag_fields.join(",")));
    }
    // A history with its own centres that took fields from its diag stream
    // (MPAS-A's `output` leaves refl10cm_max there) names that file the same
    // way; written only for such a frame, so every other frame is unchanged.
    if let Some(diag) = &frame.companion_diag {
        gattrs.push(NcAttr::text("MPAS_DIAG_FILE", file_name(Some(&diag.path))));
        gattrs.push(NcAttr::text("MPAS_DIAG_SHA256", diag.sha256.clone().unwrap_or_default()));
        gattrs.push(NcAttr::text("MPAS_DIAG_FIELDS", diag.fields.join(",")));
    }
    if let Some(r) = remap {
        gattrs.push(NcAttr::text("MPAS_PRODUCT_REMAP", r.identity_json()));
        gattrs.push(NcAttr::text("MPAS_PRODUCT_REMAP_SHA256", r.sha256.clone()));
    }
    if let Some(gather) = composite {
        gattrs.push(NcAttr::text("MPAS_COMPOSITE_SCHEMA", COMPOSITE_SCHEMA));
        gattrs.push(NcAttr::text("MPAS_COMPOSITE_SPEC", gather.plan.spec_json()));
        gattrs.push(NcAttr::text(
            "MPAS_COMPOSITE_NONCLAIM",
            "values at a point come from exactly one model run; the change of source is a              hard switch and nothing is blended across it",
        ));
    }

    // --- variable schedule --------------------------------------------------
    // Resolved once so the writer gets the whole schema up front, then walked
    // a second time to push the data. `plan` holds what each variable is; the
    // values are produced on the second pass so no frame is held twice.
    struct Planned {
        name: &'static str,
        dims: Vec<usize>,
        mapping: Option<usize>,
        stagger: &'static str,
        rank_is_3d: bool,
        units: &'static str,
        description: &'static str,
    }

    let mut plan: Vec<Planned> = Vec::new();
    let mut absent: Vec<String> = frame.absent.clone();

    for (index, mapping) in FIELD_MAP.iter().enumerate() {
        if !field_set_allows(&options.field_set, mapping) {
            continue;
        }
        if mapping.source != FieldSource::Constant && !frame.fields.contains_key(mapping.wrf_name) {
            if !absent.iter().any(|n| n == mapping.wrf_name) {
                absent.push(mapping.wrf_name.to_string());
            }
            continue;
        }
        if mapping.rank == Rank::Soil && frame.n_soil_levels == 0 {
            if !absent.iter().any(|n| n == mapping.wrf_name) {
                absent.push(mapping.wrf_name.to_string());
            }
            continue;
        }
        let (dim_names, stagger, rank_is_3d): (Vec<&str>, &'static str, bool) = match mapping.rank {
            Rank::Mass3d => (
                vec!["Time", "bottom_top", "south_north", "west_east"],
                "",
                true,
            ),
            Rank::W3d => (
                vec!["Time", "bottom_top_stag", "south_north", "west_east"],
                "Z",
                true,
            ),
            Rank::U3d => (
                vec!["Time", "bottom_top", "south_north", "west_east_stag"],
                "X",
                true,
            ),
            Rank::V3d => (
                vec!["Time", "bottom_top", "south_north_stag", "west_east"],
                "Y",
                true,
            ),
            Rank::Soil => (
                vec!["Time", "soil_layers_stag", "south_north", "west_east"],
                "Z",
                true,
            ),
            Rank::Surface => (vec!["Time", "south_north", "west_east"], "", false),
        };
        plan.push(Planned {
            name: mapping.wrf_name,
            dims: dim_names.iter().map(|n| dimid(n)).collect(),
            mapping: Some(index),
            stagger,
            rank_is_3d,
            units: mapping.units,
            description: mapping.description,
        });
    }

    let mut vars: Vec<NcVarDef> = vec![
        NcVarDef::new(
            "Times",
            NcType::Char,
            vec![dimid("Time"), dimid("DateStrLen")],
        ),
        NcVarDef::new(
            "XLAT",
            NcType::Float,
            vec![dimid("Time"), dimid("south_north"), dimid("west_east")],
        )
        .with_attrs(field_attrs(
            false,
            "LATITUDE, SOUTH IS NEGATIVE",
            "degree_north",
            "",
        )),
        NcVarDef::new(
            "XLONG",
            NcType::Float,
            vec![dimid("Time"), dimid("south_north"), dimid("west_east")],
        )
        .with_attrs(field_attrs(
            false,
            "LONGITUDE, WEST IS NEGATIVE",
            "degree_east",
            "",
        )),
    ];
    let cf = cf_coordinates(weights, frame, ny, nx)?;
    if let Some(cf) = &cf {
        let axis = |name: &str, dim: &str, attrs: Vec<NcAttr>| {
            NcVarDef::new(name, NcType::Double, vec![dimid(dim)]).with_attrs(attrs)
        };
        vars.push(axis("lat", "south_north", vec![
            NcAttr::text("units", "degrees_north"),
            NcAttr::text("standard_name", "latitude"),
            NcAttr::text("axis", "Y"),
        ]));
        vars.push(axis("lon", "west_east", vec![
            NcAttr::text("units", "degrees_east"),
            NcAttr::text("standard_name", "longitude"),
            NcAttr::text("axis", "X"),
        ]));
        vars.push(axis("Time", "Time", vec![
            NcAttr::text("units", CF_TIME_UNITS),
            NcAttr::text("calendar", "standard"),
            NcAttr::text("standard_name", "time"),
        ]));
        vars.push(axis("bottom_top", "bottom_top", vec![
            NcAttr::text("units", "1"),
            NcAttr::text("long_name", "model mass level, 1 the lowest"),
            NcAttr::text("positive", "up"),
        ]));
        vars.push(axis("bottom_top_stag", "bottom_top_stag", vec![
            NcAttr::text("units", "1"),
            NcAttr::text("long_name", "model level interface, 1 the surface"),
            NcAttr::text("positive", "up"),
        ]));
        if cf.soil_tops_cm.is_some() {
            vars.push(axis("soil_layers_stag", "soil_layers_stag", vec![
                NcAttr::text("units", "cm"),
                NcAttr::text("long_name", "soil layer top depth"),
                NcAttr::text("positive", "down"),
            ]));
        }
    }
    for planned in &plan {
        let mut attrs = field_attrs(
            planned.rank_is_3d,
            planned.description,
            planned.units,
            planned.stagger,
        );
        if remap.is_some() {
            let class = remap_class(planned.name);
            let constant = planned
                .mapping
                .is_some_and(|i| FIELD_MAP[i].source == FieldSource::Constant);
            if !constant {
                attrs.push(NcAttr::text("MPAS_RESAMPLE", class.label()));
            }
        }
        vars.push(NcVarDef::new(planned.name, NcType::Float, planned.dims.clone()).with_attrs(attrs));
    }
    if radar_winds {
        for (name, description) in [
            (RADAR_U_EARTH, "EARTH-RELATIVE ZONAL WIND ON MASS GRID"),
            (RADAR_V_EARTH, "EARTH-RELATIVE MERIDIONAL WIND ON MASS GRID"),
        ] {
            vars.push(
                NcVarDef::new(
                    name,
                    NcType::Float,
                    vec![
                        dimid("Time"),
                        dimid("bottom_top"),
                        dimid("south_north"),
                        dimid("west_east"),
                    ],
                )
                .with_attrs(field_attrs(true, description, "m s-1", "")),
            );
        }
    }
    // gpuwm addition (VENDOR.md): a composed frame carries the decision it
    // was made by, as data, on the same grid as the fields.  Two reasons
    // it is a variable rather than only an attribute: the seam can be
    // drawn from it, and it renders through the generic `var:` product
    // path, so "which run drew this pixel" is a picture anyone can look at
    // instead of a claim in a receipt.
    if composite.is_some() {
        vars.push(
            NcVarDef::new(
                COMPOSITE_SOURCE_VAR,
                NcType::Float,
                vec![dimid("Time"), dimid("south_north"), dimid("west_east")],
            )
            .with_attrs(field_attrs(
                false,
                "COMPOSITE SOURCE INDEX (0 = BASE)",
                "index",
                "",
            )),
        );
        vars.push(
            NcVarDef::new(
                COMPOSITE_SPACING_VAR,
                NcType::Float,
                vec![dimid("Time"), dimid("south_north"), dimid("west_east")],
            )
            .with_attrs(field_attrs(false, "COMPOSITE SHOWN CELL SPACING", "km", "")),
        );
    }
    let since = format!(
        "minutes since {:04}-{:02}-{:02} {:02}:{:02}:{:02}",
        origin.year, origin.month, origin.day, origin.hour, origin.minute, origin.second
    );
    vars.push(
        NcVarDef::new("XTIME", NcType::Float, vec![dimid("Time")]).with_attrs(vec![
            NcAttr::int("FieldType", 104),
            NcAttr::text("MemoryOrder", "0  "),
            NcAttr::text("description", since.clone()),
            NcAttr::text("units", since),
            NcAttr::text("stagger", ""),
        ]),
    );
    vars.push(
        NcVarDef::new("ITIMESTEP", NcType::Int, vec![dimid("Time")]).with_attrs(vec![
            NcAttr::int("FieldType", 106),
            NcAttr::text("MemoryOrder", "0  "),
            NcAttr::text("description", ""),
            NcAttr::text("units", ""),
            NcAttr::text("stagger", ""),
        ]),
    );

    // --- write --------------------------------------------------------------
    let temporary = destination.with_file_name(format!(
        ".{}.tmp",
        destination
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("frame")
    ));
    if temporary.exists() {
        std::fs::remove_file(&temporary)?;
    }
    let mut written: Vec<String> = Vec::new();
    let mut composite_fallbacks: Vec<String> = Vec::new();
    let mut seam: Vec<SeamStat> = Vec::new();
    let boundary = composite
        .map(|gather| gather.plan.boundary_slots())
        .unwrap_or_default();
    let result = (|| -> MpasResult<()> {
        let mut writer =
            NcClassicWriter::create(&temporary, options.format, dims.clone(), gattrs, vars, 1)?;
        writer.put_record_text("Times", 0, &frame.valid_time.label())?;
        let lat32: Vec<f32> = weights.target_latitude.iter().map(|&v| v as f32).collect();
        let lon32: Vec<f32> = weights.target_longitude.iter().map(|&v| v as f32).collect();
        writer.put_record("XLAT", 0, NcData::Floats(&lat32))?;
        writer.put_record("XLONG", 0, NcData::Floats(&lon32))?;
        written.push("XLAT".into());
        written.push("XLONG".into());
        if let Some(cf) = &cf {
            let levels: Vec<f64> = (1..=nz).map(|k| k as f64).collect();
            let interfaces: Vec<f64> = (1..=nz + 1).map(|k| k as f64).collect();
            let epoch = Timestamp { year: 1970, month: 1, day: 1, hour: 0, minute: 0, second: 0 };
            writer.put("lat", NcData::Doubles(&cf.latitude))?;
            writer.put("lon", NcData::Doubles(&cf.longitude))?;
            writer.put_record(
                "Time",
                0,
                NcData::Doubles(&[frame.valid_time.seconds_since(&epoch) as f64]),
            )?;
            writer.put("bottom_top", NcData::Doubles(&levels))?;
            writer.put("bottom_top_stag", NcData::Doubles(&interfaces))?;
            written.extend(["lat", "lon", "Time", "bottom_top", "bottom_top_stag"].map(String::from));
            if let Some(tops) = &cf.soil_tops_cm {
                writer.put("soil_layers_stag", NcData::Doubles(tops))?;
                written.push("soil_layers_stag".into());
            }
        }

        for planned in &plan {
            let mapping = &FIELD_MAP[planned.mapping.expect("every planned entry maps")];
            let levels = expected_levels(mapping, frame).ok_or_else(|| {
                MpasError::Refusal(format!(
                    "{} has no level count on this frame",
                    mapping.wrf_name
                ))
            })?;
            let mut values: Vec<f32> = match mapping.source {
                FieldSource::Constant => {
                    let points = ny * nx;
                    match mapping.wrf_name {
                        "COSALPHA" => vec![1.0f32; points],
                        "SINALPHA" => vec![0.0f32; points],
                        _ => vec![0.0f32; levels * points],
                    }
                }
                _ => {
                    let (source, source_levels) = frame
                        .fields
                        .get(mapping.wrf_name)
                        .expect("planned entries have their source field");
                    if *source_levels != levels {
                        return Err(MpasError::Refusal(format!(
                            "{} carries {source_levels} level(s); the frame declares {levels}",
                            mapping.wrf_name
                        )));
                    }
                    match composite {
                        Some(gather) => {
                            let (values, fell_back) =
                                gather.gather(mapping.wrf_name, levels, ny * nx)?;
                            for label in fell_back {
                                let note = format!("{}:{}", mapping.wrf_name, label);
                                if !composite_fallbacks.contains(&note) {
                                    composite_fallbacks.push(note);
                                }
                            }
                            values
                        }
                        None => match remap {
                            Some(r) => r.gather(remap_class(mapping.wrf_name), source, levels)?,
                            None => weights.gather(source, levels)?,
                        },
                    }
                }
            };
            match mapping.wrf_name {
                "T" => values.iter_mut().for_each(|v| *v -= WRF_THETA_REFERENCE_K),
                "PHB" => values.iter_mut().for_each(|v| *v *= WRF_GRAVITY),
                "HGT" => if let Some(terrain) = &options.terrain_authority {
                    values.copy_from_slice(&terrain.values);
                } else if let (None, Some((zgrid, zl))) = (composite, frame.fields.get("PHB")) {
                    // MPAS defines the terrain as the ground of its height
                    // coordinate (init_atmosphere: ter = zgrid(1)), so HGT is
                    // taken from zgrid's lowest level, which is the surface PHB
                    // itself carries.  Breakage prevented: the hex RRFS
                    // limited-area history (rrfs/c3-lam, mv3km 2023-03-31)
                    // writes ter = 0 everywhere while zgrid(1) holds the real
                    // terrain, so every product that reduced to sea level or
                    // took heights above ground used flat terrain (MSLP 864-1013
                    // hPa following the mountains, 1000 hPa heights of -1462 m).
                    let ground: Vec<f32> = zgrid.chunks(*zl).map(|c| c[0]).collect();
                    values = match remap {
                        Some(r) => r.gather(remap_class("HGT"), &ground, 1)?,
                        None => weights.gather(&ground, 1)?,
                    };
                },
                _ => {}
            }
            if radar_winds {
                let mass_name = match mapping.wrf_name {
                    "U" => Some(RADAR_U_EARTH),
                    "V" => Some(RADAR_V_EARTH),
                    _ => None,
                };
                if let Some(name) = mass_name {
                    writer.put_record(name, 0, NcData::Floats(&values))?;
                    written.push(name.to_string());
                }
            }
            let staggered = match mapping.rank {
                Rank::U3d => restagger_x(&values, levels, ny, nx),
                Rank::V3d => restagger_y(&values, levels, ny, nx),
                _ => values,
            };
            // The seam is measured on the values that were WRITTEN, against
            // the base's values for the same points -- not on a proxy and
            // not on the source arrays, so the number describes the picture
            // a reader will actually see.
            if let Some(gather) = composite {
                if matches!(mapping.rank, Rank::Surface) && !boundary.is_empty() {
                    if let Some(base_layer) = gather.layers.first() {
                        if let Some((values, 1)) = base_layer
                            .frame
                            .fields
                            .get(mapping.wrf_name)
                            .map(|(v, l)| (v, *l))
                        {
                            let mut base_plane = vec![0.0f32; ny * nx];
                            for (slot, out) in base_plane.iter_mut().enumerate() {
                                *out = values[base_layer.weights.cell_index[slot] as usize];
                            }
                            match mapping.wrf_name {
                                "T" => base_plane
                                    .iter_mut()
                                    .for_each(|v| *v -= WRF_THETA_REFERENCE_K),
                                "PHB" => base_plane.iter_mut().for_each(|v| *v *= WRF_GRAVITY),
                                _ => {}
                            }
                            if let Some(stat) = SeamStat::measure(
                                mapping.wrf_name,
                                &boundary,
                                &gather.plan.source_index,
                                &staggered,
                                &base_plane,
                                ny * nx,
                            ) {
                                seam.push(stat);
                            }
                        }
                    }
                }
            }
            writer.put_record(mapping.wrf_name, 0, NcData::Floats(&staggered))?;
            written.push(mapping.wrf_name.to_string());
        }

        if let Some(gather) = composite {
            let source_index: Vec<f32> = gather
                .plan
                .source_index
                .iter()
                .map(|&v| f32::from(v))
                .collect();
            let spacing_km: Vec<f32> = gather
                .plan
                .shown_spacing_metres
                .iter()
                .map(|&v| v / 1000.0)
                .collect();
            writer.put_record(COMPOSITE_SOURCE_VAR, 0, NcData::Floats(&source_index))?;
            writer.put_record(COMPOSITE_SPACING_VAR, 0, NcData::Floats(&spacing_km))?;
            written.push(COMPOSITE_SOURCE_VAR.to_string());
            written.push(COMPOSITE_SPACING_VAR.to_string());
        }
        writer.put_record(
            "XTIME",
            0,
            NcData::Floats(&[(lead_seconds as f64 / 60.0) as f32]),
        )?;
        writer.put_record("ITIMESTEP", 0, NcData::Ints(&[0]))?;
        writer.finish()?;
        Ok(())
    })();
    if let Err(err) = result {
        let _ = std::fs::remove_file(&temporary);
        return Err(err);
    }
    std::fs::rename(&temporary, destination)?;

    let bytes = std::fs::metadata(destination)?.len();
    written.sort();
    written.dedup();
    absent.sort();
    absent.dedup();
    Ok(EmittedFrame {
        path: destination.to_path_buf(),
        window: weights.window_name.clone(),
        shape: (ny, nx),
        levels: nz,
        written,
        absent,
        seconds: started.elapsed().as_secs_f64(),
        bytes,
        valid_time: frame.valid_time,
        simulation_start: origin,
        lead_seconds,
        seam,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn x_restaggering_copies_the_edges_and_averages_the_interior() {
        let values = vec![1.0f32, 3.0, 7.0];
        let out = restagger_x(&values, 1, 1, 3);
        assert_eq!(out, vec![1.0, 2.0, 5.0, 7.0]);
    }

    #[test]
    fn y_restaggering_copies_the_edges_and_averages_the_interior() {
        // two rows, two columns
        let values = vec![1.0f32, 2.0, 5.0, 8.0];
        let out = restagger_y(&values, 1, 2, 2);
        assert_eq!(out, vec![1.0, 2.0, 3.0, 5.0, 5.0, 8.0]);
    }

    #[test]
    fn restaggering_averages_in_f32_not_f64() {
        // 0.1 + 0.2 differs between an f32 add and an f64 add rounded to f32,
        // which is exactly the kind of drift a port introduces by accident.
        let values = vec![0.1f32, 0.2f32];
        let out = restagger_x(&values, 1, 1, 2);
        assert_eq!(out[1].to_bits(), ((0.1f32 + 0.2f32) * 0.5f32).to_bits());
    }

    #[test]
    fn grid_label_names_uniform_and_variable_meshes() {
        let uniform = [29_000.0, 30_000.0, 31_500.0];
        assert_eq!(grid_label(Some("hex"), &uniform).as_deref(), Some("hex 30 km"));
        let variable = [3_000.0, 12_000.0, 52_000.0];
        assert_eq!(grid_label(Some("hex"), &variable).as_deref(), Some("hex 52-3 km"));
        assert_eq!(grid_label(None, &[937.5, 940.0]).as_deref(), Some("mesh 0.9 km"));
        assert_eq!(grid_label(Some("hex"), &[]), None);
    }

    /// The renderer reads this exact name (`rustwx_products::shared_context`
    /// pins the same literal), and a converter that writes nothing unless
    /// asked keeps every existing frame byte-identical.
    #[test]
    fn the_model_label_attribute_is_the_renderers_and_is_off_by_default() {
        assert_eq!(MODEL_LABEL_ATTRIBUTE, "GPUWM_MODEL_LABEL");
        assert_eq!(ConvertOptions::default().model_label, None);
    }

    /// The renderer reads this exact name (`rw_wrfbatch` pins the same
    /// literal), and only a frame whose REFC took the cumulus scheme's
    /// reflectivity says so: a convection-permitting row (no cumulus scheme,
    /// no `cuprec`) converts to the same attributes as before.
    #[test]
    fn the_refc_note_names_parameterised_convection_only_when_the_frame_carries_cuprec() {
        assert_eq!(REFC_NOTE_ATTRIBUTE, "GPUWM_REFC_NOTE");
        let mut frame = MpasFrame {
            history_path: "unused".into(), history_sha256: String::new(), init_path: None, init_sha256: None,
            valid_time: crate::history::Timestamp::parse("2026-10-28_20:00:00").unwrap(),
            n_cells: 2, n_levels: 1, n_soil_levels: 0, fields: Default::default(),
            latitude_degrees: vec![], longitude_degrees: vec![], absent: vec![], science_attributes: vec![],
            split: None, companion_diag: None,
        };
        frame.fields.insert("REFC".into(), (vec![-35.0, 41.0], 1));
        assert_eq!(refc_note(&frame), None);
        frame.fields.insert("CUPREC".into(), (vec![0.0, 0.0], 1));
        assert_eq!(refc_note(&frame), Some("incl. parameterised convection"));
        frame.fields.remove("REFC");
        assert_eq!(refc_note(&frame), None);
    }

    #[test]
    fn the_grid_label_names_the_mesh_spacing_not_the_render_grid() {
        assert_eq!(GRID_LABEL_ATTRIBUTE, "GPUWM_GRID_LABEL");
        assert_eq!(ConvertOptions::default().grid_label, None);
        let uniform15: Vec<f64> = (0..101).map(|i| 14_400.0 + 12.0 * i as f64).collect();
        assert_eq!(mesh_spacing_label(&uniform15).as_deref(), Some("15 km"));
        assert_eq!(mesh_spacing_label(&[29_870.0; 5]).as_deref(), Some("30 km"));
        assert_eq!(mesh_spacing_label(&[59_800.0, 61_000.0, 58_500.0]).as_deref(), Some("60 km"));
        assert_eq!(mesh_spacing_label(&[4_500.0; 3]).as_deref(), Some("4.5 km"));
        assert_eq!(mesh_spacing_label(&[3_000.0, 12_000.0, 60_000.0]).as_deref(), Some("60-3 km"));
        // A 96-to-3 km zoom globe: a few distorted cells at either end
        // (2.51 km, 100.45 km) must not name the mesh; its regions do.
        let mut zoom: Vec<f64> = vec![2_513.5, 100_451.8];
        zoom.extend((0..400).map(|i| 2_950.0 + 0.5 * i as f64)); // refined region, median ~3.05 km
        zoom.extend((0..60).map(|i| 6_000.0 + 700.0 * i as f64)); // transition 6-47 km
        zoom.extend((0..200).map(|i| 94_000.0 + 20.0 * i as f64)); // background, median ~96 km
        assert_eq!(mesh_spacing_label(&zoom).as_deref(), Some("96-3 km"));
        assert_eq!(mesh_spacing_label(&[]), None);
        assert_eq!(mesh_spacing_label(&[15_000.0, f64::NAN]), None);
        assert_eq!(grid_label(Some("hex"), &uniform15).as_deref(), Some("hex 15 km"));
        assert_eq!(grid_label(None, &uniform15).as_deref(), Some("mesh 15 km"));
        assert_eq!(grid_label(Some("a|b"), &uniform15), None);
    }

    /// Project ruling, 2026-10-30: the site menu read `Core zoom · 8 → 38 km` over a
    /// picture headlined `(hex 133-6.9 km)`, the whole mesh's range. A view's
    /// label names the cells THAT picture samples, finest first.
    #[test]
    fn a_view_label_names_the_cells_its_picture_samples() {
        // An 8 -> 128 km zoom globe: two distorted cells, the refined core,
        // the transition, the background.
        let mut zoom: Vec<f64> = vec![6_900.0, 133_000.0];
        zoom.extend((0..400).map(|i| 7_900.0 + 0.5 * i as f64)); // core, median ~8.0 km
        zoom.extend((0..60).map(|i| 12_000.0 + 1_000.0 * i as f64)); // transition 12-71 km
        zoom.extend((0..200).map(|i| 126_000.0 + 10.0 * i as f64)); // background ~127 km
        let core: Vec<usize> = (2..402).collect();
        let core_to_corner: Vec<usize> = (2..402).chain(402..429).collect(); // reaches 38 km
        let whole: Vec<usize> = (0..zoom.len()).collect();
        assert_eq!(window_spacing_label(&zoom, &core).as_deref(), Some("8 km"));
        assert_eq!(window_spacing_label(&zoom, &core_to_corner).as_deref(), Some("8-37 km"));
        assert_eq!(window_spacing_label(&zoom, &whole).as_deref(), Some("8-127 km"));
        // A uniform mesh is one value wherever the picture is.
        let uniform: Vec<f64> = (0..100).map(|i| 115_000.0 + 100.0 * i as f64).collect();
        assert_eq!(window_spacing_label(&uniform, &[0, 50, 99]).as_deref(), Some("120 km"));
        assert_eq!(window_spacing_label(&uniform, &[]), None);
        assert_eq!(window_spacing_label(&[f64::NAN, 3_000.0], &[0]), None);
    }

    #[test]
    fn a_caller_grid_label_is_written_as_given_or_refused_by_name() {
        assert_eq!(caller_grid_label("Core zoom 8-38 km", None).as_deref(), Ok("Core zoom 8-38 km"));
        assert_eq!(caller_grid_label("  Core   zoom {spacing} ", Some("8-36 km")).as_deref(),
                   Ok("Core zoom 8-36 km"));
        assert!(caller_grid_label("Core zoom {spacing}", None).unwrap_err().contains("{spacing}"));
        assert!(caller_grid_label("   ", None).unwrap_err().contains("needs a name"));
        assert!(caller_grid_label("a|b", None).unwrap_err().contains("'|'"));
        assert!(caller_grid_label(&"x".repeat(33), None).unwrap_err().contains("32 characters"));
        assert_eq!(caller_grid_label(&"x".repeat(32), None).map(|l| l.len()), Ok(32));
    }

    #[test]
    fn staggered_levels_survive_independently() {
        let values = vec![1.0f32, 3.0, 10.0, 30.0];
        let out = restagger_x(&values, 2, 1, 2);
        assert_eq!(out, vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0]);
    }
}
