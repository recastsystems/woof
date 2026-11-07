//! Bounded MPAS static-field builder.
//!
//! Large geography data are processed tile-by-tile and plane-by-plane.
//! The only persistent geography-derived arrays are destination-sized
//! accumulators and the final fields.  No full global WPS source mosaic and
//! no source-pixel x destination-cell table is constructed.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicI32, AtomicI64, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use rayon::prelude::*;

use crate::error::{MpasError, MpasResult};
use crate::staticfile::coordframe::CoordinateRepresentation;
use crate::static_geog::{GeogDataset, SourceGeometry, SourceKind, TileRef};
use crate::static_memory::{
    checked_len, checked_vec, checked_vec_filled, checked_vec_with, plan_tile_parallelism,
    BufferLedger, MemoryReceipt, TileParallelPlan,
};
use crate::static_operators::{build_operator_fields, OperatorFields, OperatorMesh, FIFTEEN};
use rw_store::netcdf_classic::{
    NcAttr, NcClassicWriter, NcData, NcDim, NcFormat, NcType, NcVarDef,
};

pub const MPAS_EARTH_RADIUS_M: f64 = 6_371_229.0;

/// The outermost regional boundary ring, MPAS's `nBdyLayers = 7` (five
/// relaxation plus two specified layers). Only cells, edges and vertices at
/// this mask value may carry absent-neighbour sentinels, and only pixels
/// landing on these cells take the `mpas_in_cell` containment test.
pub const REGIONAL_OUTERMOST_MASK: i32 = 7;

/// The in-memory marker for "that neighbour was culled away". Deliberately
/// `usize::MAX` so an unguarded index panics instead of reading cell 0.
pub const ABSENT_NEIGHBOR: usize = usize::MAX;
pub const MPAS_OMEGA: f64 = 7.29212e-5;

/// The schema tag the unified writer stamps into a legacy-stream file
/// (`rw_mpas_static --legacy-static`, the pre-2.9.0 82-variable stream).
pub const UNIFIED_SCHEMA_TAG: &str = "rw-mpas-static/v3-unified";

/// The schema tag stamped into a native-stream file, the 2.9.0 default.
///
/// THE STATIC CACHE MARKER. A static is cached by its consumers (DA case
/// preparation, the hex point door, mesh rows) as a file that exists, and a
/// pre-2.9.0 one carries the with-lakes land use with every lake stored as
/// land. Moving this tag is what lets a consumer tell such a file from a
/// native one without opening its arrays: `rw_mpas_static_schema` in the
/// file and `schema` in the receipt both carry it.
pub const NATIVE_SCHEMA_TAG: &str = "rw-mpas-static/v4-native";

/// The schema tag for the stream `stages` selects.
pub fn schema_tag(stages: &StaticStageOptions) -> &'static str {
    if stages.native_init { NATIVE_SCHEMA_TAG } else { UNIFIED_SCHEMA_TAG }
}

/// MPAS: core_init_atmosphere/Registry.xml:498-585,882-903.
fn native_static_field_enabled(name: &str, stages: &StaticStageOptions) -> bool {
    match name {
        "defc_a"|"defc_b"|"lu_index"|"soilcat_top" => false,
        "soilcomp"|"soilcl1"|"soilcl2"|"soilcl3"|"soilcl4" => stages.noahmp,
        "var2d"|"con"|"oa1"|"oa2"|"oa3"|"oa4"|"ol1"|"ol2"|"ol3"|"ol4" => stages.gwd,
        _=>true,
    }
}

/// MPAS: src/framework/mpas_stream_manager.F:4898-4900,5037-5162.
/// Native streams discard connectivity outside each entity's active stencil.
fn static_topology_words<'a>(raw: &'a [i32], stride: usize, counts: &[usize], native: bool) -> std::borrow::Cow<'a,[i32]> {
    if !native {return std::borrow::Cow::Borrowed(raw);}
    let mut output=raw.to_vec();
    for (row,&used) in counts.iter().enumerate() {output[row*stride+used..(row+1)*stride].fill(0);}
    std::borrow::Cow::Owned(output)
}

/// Model-side unit conversions, applied on top of each archive's own
/// `scale_factor`.
///
/// A dataset's `index` declares what its numbers mean and this crate's reader
/// honours that declaration. The MODEL separately expects a convention of its
/// own, and the two are not the same statement. Grading a built static against
/// a published one caught both places where they disagree, and neither would
/// have raised anything -- both are finite, both are positive, and both pass
/// every range check a validator is likely to hold:
///
/// * `greenfrac` -- the archive declares `units="fraction"` and, after its own
///   0.01 scale, delivers 0..0.97. The published static carries 0..92.07. The
///   model wants PERCENT, so honouring the archive alone hands the surface
///   scheme a vegetation fraction one hundredth of the real one, everywhere.
/// * `snoalb` -- the archive declares `units="percent"` and, after its own
///   0.01 scale, delivers 0..84. The published static carries 0..0.84. The
///   model wants a FRACTION, so honouring the archive alone hands the
///   radiation scheme a snow albedo of 84, which is not an albedo at all.
///
/// Green fractions retain their raw percent convention through `MeanPolicy::Green`;
/// snow albedo uses the model-side scale below.
pub const SNOALB_MODEL_SCALE: f32 = 0.01;

#[derive(Debug, Clone)]
pub struct StaticGeogPaths {
    pub terrain: PathBuf,
    pub landuse: PathBuf,
    pub soilcat: PathBuf,
    pub greenfrac: PathBuf,
    pub albedo: PathBuf,
    pub snow_albedo: PathBuf,
    pub soil_temperature: PathBuf,
    pub soilcomp: PathBuf,
    pub soilcl1: PathBuf,
    pub soilcl2: PathBuf,
    pub soilcl3: PathBuf,
    pub soilcl4: PathBuf,
}

/// MPAS: mpas_init_atm_static.F:488-1198,1199-1250.
#[derive(Debug, Clone)]
pub struct StaticStageOptions {
    pub vegetation_ncep: bool,
    pub albedo_ncep: bool,
    pub snow_ncep: bool,
    pub noahmp: bool,
    pub gwd: bool,
    /// Native driver owns the classification label, independently of geoindex.
    pub landuse_label: Option<String>,
    /// Emit MPAS init_atmosphere's package-dependent static stream.
    pub native_init: bool,
    /// Optional with-lakes geography, kept separate from native land-use voting.
    pub lake_mask_geog: Option<PathBuf>,
}
impl Default for StaticStageOptions {
    fn default() -> Self {
        Self { vegetation_ncep: false, albedo_ncep: false, snow_ncep: false, noahmp: true, gwd: true, landuse_label: None, native_init: false, lake_mask_geog: None }
    }
}

#[derive(Debug, Clone)]
pub struct StaticBuildConfig {
    pub grid_path: PathBuf,
    pub out_path: PathBuf,
    pub geog: StaticGeogPaths,
    pub stages: StaticStageOptions,
    pub supersample: usize,
    pub supersample_landuse: usize,
    pub supersample_30s: usize,
    pub host_memory_limit_bytes: Option<u64>,
    /// Tiles to process concurrently.  `None` takes the host CPU count,
    /// which the host-memory budget then bounds.  A build is deterministic
    /// at every setting: see [`resolve_tile_workers`].
    pub tile_workers: Option<usize>,
    /// Native MPAS v8.4.2 GWD data path; None retains the current static suite.
    pub native_gwd: Option<crate::native_gwd::NativeGwdConfig>,
    pub receipt_path: Option<PathBuf>,
    pub provenance: String,
    /// The `xtime` stamp the file carries.
    pub valid_time: String,
    /// The nominal spacing to DECLARE, in metres. `None` takes the grid's own.
    ///
    /// A static's `nominalMinDc` is compared BIT FOR BIT, as FP32, against the
    /// nominal spacing a mesh binding declares: it is the scalar the
    /// gravity-wave drag length scale is rebound to, so a value that is merely
    /// close is a wrong answer wearing a right one.
    pub nominal_dx_m: Option<f64>,
    /// Refuse rather than replace an existing `out_path`.
    ///
    /// Overwriting a static silently would leave every init file and run
    /// receipt that pinned its sha256 pointing at different bytes under the
    /// same name.
    pub clobber: bool,
    /// How the fifteen coordinate arrays are stored, and what the file
    /// DECLARES it stored them as.
    ///
    /// [`CoordinateRepresentation::Binary32EarthCentred`] is the default and
    /// is what preserves the dycore byte-identity anchor: it is what native
    /// MPAS-A writes and what both published statics carry.
    /// [`CoordinateRepresentation::Binary64EarthCentred`] is legal only for a
    /// mesh with NO native counterpart, where there is nothing to be
    /// byte-identical to; `rw_mpas_static` takes it from the grid file's own
    /// declaration rather than from a flag, so the choice travels with the
    /// mesh instead of with the command line.
    pub coordinates: CoordinateRepresentation,
}

/// The `xtime` a static carries when a caller did not say.
pub const DEFAULT_VALID_TIME: &str = "2000-01-01_00:00:00";

/// Tolerance the consumer applies to the grid/static cross-check: the
/// unit-sphere angle times the earth radius must land within 0.1 % of the
/// declared physical value.
pub const NOMINAL_DX_CROSS_CHECK_RELATIVE: f64 = 1e-3;

/// Which WPS_GEOG directory serves which static field.
///
/// A TABLE, and the whole extension mechanism: a new source is a row here, not
/// a branch and not a per-dataset module. The names are the archive's own
/// directory names, which is what `gpuwm fetch-geog` stages and what the
/// door's completeness check looks for, so the three cannot drift apart.
/// WHERE under the root a directory sits is [`GEOG_CONTAINER_TABLE`]'s
/// business, resolved by [`resolve_geog_dataset_dir`].
///
/// The soil category and deep-soil temperature rows are the permissive
/// sources (Project ruling, 2026-10-14: no default path into data whose licence limits
/// sale or use; 2026-10-21: non-permissive WPS_GEOG soil sources are held out
/// of the default set): SoilGrids250m's `texture_top` (CC BY 4.0, the same 16
/// categories as STATSGO/FAO) and the ERA5 climatology `gpuwm fetch-geog`
/// builds in `soiltemp_1deg`'s exact layout. MPAS-A's own STATSGO/FAO tiles
/// and ISLSCP-I `soiltemp_1deg` are rows of [`MPAS_STATIC_SOURCE_TABLE`],
/// read only when named.
pub const GEOG_DATASET_TABLE: [(&str, &str); 12] = [
    ("terrain", "topo_gmted2010_30s"),
    ("landuse", "modis_landuse_20class_30s_with_lakes"),
    ("soilcat", "texture_top"),
    ("greenfrac", "greenfrac_fpar_modis"),
    ("albedo", "albedo_modis"),
    ("snow_albedo", "maxsnowalb_modis"),
    ("soil_temperature", "soiltemp_era5_1deg"),
    ("soilcomp", "soilcomp"),
    ("soilcl1", "texture_layer1"),
    ("soilcl2", "texture_layer2"),
    ("soilcl3", "texture_layer3"),
    ("soilcl4", "texture_layer4"),
];

/// MPAS: mpas_init_atm_static.F:369-447,488-1198.
/// Option selection is data; the current door's staged defaults stay in the
/// GEOG_DATASET_TABLE above.
pub const MPAS_STATIC_SOURCE_TABLE: &[(&str,&str,&str)] = &[
    ("topo-data","GTOPO30","topo_30s"),
    ("topo-data","GMTED2010","topo_gmted2010_30s"),
    ("landuse-data","USGS","landuse_30s"),
    ("landuse-data","MODIFIED_IGBP_MODIS_NOAH","modis_landuse_20class_30s"),
    ("landuse-data","MODIFIED_IGBP_MODIS_NOAH_15s","modis_landuse_20class_15s"),
    ("soilcat-data","SOILGRIDS","texture_top"),
    ("soilcat-data","STATSGO","soiltype_top_30s"),
    ("soilcat-data","BNU","bnu_soiltype_top"),
    ("soiltemp-data","ERA5","soiltemp_era5_1deg"),
    ("soiltemp-data","ISLSCP","soiltemp_1deg"),
    ("vegfrac-data","MODIS","greenfrac_fpar_modis"),
    ("vegfrac-data","NCEP","greenfrac"),
    ("albedo-data","MODIS","albedo_modis"),
    ("albedo-data","NCEP","albedo_ncep"),
    ("maxsnowalbedo-data","MODIS","maxsnowalb_modis"),
    ("maxsnowalbedo-data","NCEP","maxsnowalb"),
];

/// The terms of a dataset the default geography holds out, quoted from its
/// own documentation (gpuwm/geog_assets.py carries the same lines), or None.
pub fn held_out_terms(dataset: &str) -> Option<&'static str> {
    match dataset {
        "soiltype_top_30s" | "soiltype_bot_30s" => Some(
            "FAO-UNESCO Digital Soil Map of the World outside CONUS: \"(C) FAO/UNESCO, 1995 All \
             rights reserved worldwide\", \"The user shall not reproduce for the benefit of third \
             parties the data\" (DSMW notes v3.6, https://swat.tamu.edu/media/116411/notes.txt): \
             not permissive"),
        "soiltemp_1deg" => Some(
            "ISLSCP-I 1987-88 ECMWF annual-mean temperature; CEDA's record states Permitted Use \
             \"Unstated\" (https://catalogue.ceda.ac.uk/uuid/b9c3effaab9fe9a2206b58af500f157d): \
             licence unclear"),
        _ => None,
    }
}

/// MPAS: mpas_init_atm_static.F:369-447,488-1198 (dataset selection).
pub fn select_static_source(cfg: &mut StaticBuildConfig, root: &Path, slot: &str, choice: &str) -> MpasResult<()> {
    let dataset=MPAS_STATIC_SOURCE_TABLE.iter().find(|(s,c,_)| *s==slot && *c==choice)
        .ok_or_else(|| MpasError::Refusal(format!("unsupported MPAS static {slot}={choice}")))?.2;
    let path=resolve_geog_dataset_dir(root,dataset);
    if let Some(terms)=held_out_terms(dataset) {
        eprintln!("geography: {dataset} ({slot} {choice}) is held out of the default geography \
                   and read here only because it was named -- {terms}");
    }
    match slot {
        "topo-data" => cfg.geog.terrain=path,
        "landuse-data" => {
            cfg.geog.landuse=path;
            cfg.stages.landuse_label=Some(if choice=="USGS" { "USGS" } else { "MODIFIED_IGBP_MODIS_NOAH" }.into());
        },
        "soilcat-data" => cfg.geog.soilcat=path,
        "soiltemp-data" => cfg.geog.soil_temperature=path,
        "vegfrac-data" => { cfg.geog.greenfrac=path; cfg.stages.vegetation_ncep=choice=="NCEP"; },
        "albedo-data" => { cfg.geog.albedo=path; cfg.stages.albedo_ncep=choice=="NCEP"; },
        "maxsnowalbedo-data" => { cfg.geog.snow_albedo=path; cfg.stages.snow_ncep=choice=="NCEP"; },
        _ => unreachable!("source table slot"),
    }
    Ok(())
}

/// Which dataset directories `gpuwm fetch-geog` stages INSIDE a container
/// directory rather than at the archive root: `(directory, container)`.
///
/// One upstream archive carries the Noah-MP soil group as seven dataset
/// directories under a single top-level directory, and the fetcher unpacks it
/// as it comes: `WPS_GEOG/soilgrids/soilcomp`, `WPS_GEOG/soilgrids/texture_layer1`
/// and so on. Its pin -- the `soilgrids` row of `gpuwm/geog_assets.py`, whose
/// `index_subdirs` declares exactly these children -- is what that layout
/// follows, and this table mirrors the row so the builder looks where the
/// product stages. Every reader once assumed the flat layout alone, so an
/// install that ran the fetch exactly as doctor told it to was refused for
/// five datasets it already had.
///
/// A row here, not a branch: a future container archive is a row in the pin
/// table and a matching row here, and nothing else changes.
pub const GEOG_CONTAINER_TABLE: [(&str, &str); 7] = [
    ("soilcomp", "soilgrids"),
    ("texture_top", "soilgrids"),
    ("texture_bot", "soilgrids"),
    ("texture_layer1", "soilgrids"),
    ("texture_layer2", "soilgrids"),
    ("texture_layer3", "soilgrids"),
    ("texture_layer4", "soilgrids"),
];

/// Where dataset directory `dataset` lives under `root`.
///
/// `root/<dataset>` when that carries a WPS `index`; otherwise
/// `root/<container>/<dataset>` for a container [`GEOG_CONTAINER_TABLE`]
/// names for it; otherwise the flat path, so [`StaticGeogPaths::missing`]
/// still names the dataset by its own directory name and the refusal reads
/// the same in both layouts. The flat path is tried first, matching the
/// door's own lookup, so a hand-built archive keeps winning over a staged
/// one on a box that carries both.
pub fn resolve_geog_dataset_dir(root: &Path, dataset: &str) -> PathBuf {
    let flat = root.join(dataset);
    if flat.join("index").is_file() {
        return flat;
    }
    for (child, container) in GEOG_CONTAINER_TABLE.iter() {
        if *child == dataset {
            let nested = root.join(container).join(dataset);
            if nested.join("index").is_file() {
                return nested;
            }
        }
    }
    flat
}

impl StaticGeogPaths {
    /// Resolve every geography directory from one archive root, in either
    /// layout `gpuwm fetch-geog` stages ([`resolve_geog_dataset_dir`]).
    pub fn under(root: &Path) -> Self {
        let at = |name: &str| -> PathBuf {
            resolve_geog_dataset_dir(
                root,
                GEOG_DATASET_TABLE
                    .iter()
                    .find(|(slot, _)| *slot == name)
                    .expect("every slot is a row of GEOG_DATASET_TABLE")
                    .1,
            )
        };
        Self {
            terrain: at("terrain"),
            landuse: at("landuse"),
            soilcat: at("soilcat"),
            greenfrac: at("greenfrac"),
            albedo: at("albedo"),
            snow_albedo: at("snow_albedo"),
            soil_temperature: at("soil_temperature"),
            soilcomp: at("soilcomp"),
            soilcl1: at("soilcl1"),
            soilcl2: at("soilcl2"),
            soilcl3: at("soilcl3"),
            soilcl4: at("soilcl4"),
        }
    }

    /// Every slot in declaration order, for a completeness check.
    pub fn slots(&self) -> [(&'static str, &Path); 12] {
        [
            ("terrain", &self.terrain),
            ("landuse", &self.landuse),
            ("soilcat", &self.soilcat),
            ("greenfrac", &self.greenfrac),
            ("albedo", &self.albedo),
            ("snow_albedo", &self.snow_albedo),
            ("soil_temperature", &self.soil_temperature),
            ("soilcomp", &self.soilcomp),
            ("soilcl1", &self.soilcl1),
            ("soilcl2", &self.soilcl2),
            ("soilcl3", &self.soilcl3),
            ("soilcl4", &self.soilcl4),
        ]
    }

    /// The datasets that are not there, by directory name.
    ///
    /// Checked before the mesh is read so a missing archive is a five-second
    /// refusal instead of a wasted geography pass.
    pub fn missing(&self) -> Vec<String> {
        self.slots()
            .iter()
            .filter(|(_, path)| !path.join("index").is_file())
            .map(|(_, path)| {
                path.file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_else(|| path.display().to_string())
            })
            .collect()
    }
}

/// The requested worker count before the memory budget bounds it.
///
/// The output is byte-identical at every worker count -- every accumulator in
/// this module is an integer, and integer addition does not care what order
/// it happens in -- so this knob buys throughput and never changes an answer.
/// The determinism gate in this module's tests is what holds that true.
pub fn resolve_tile_workers(explicit: Option<usize>) -> usize {
    if let Some(n) = explicit {
        return n.max(1);
    }
    if let Ok(raw) = std::env::var("RW_MPAS_TILE_WORKERS") {
        if let Ok(v) = raw.trim().parse::<usize>() {
            if v > 0 {
                return v;
            }
        }
    }
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

#[derive(Debug, serde::Serialize)]
pub struct DatasetReceipt {
    pub field: String,
    pub path: String,
    pub tiles: usize,
    pub source_plane_bytes: u64,
    pub supersample: usize,
    pub destination_bytes: u64,
    pub elapsed_seconds: f64,
    /// Legacy receipt slot; native static means use zero propagation rings.
    pub spread_rounds: usize,
    /// Cells no source pixel reached, which took the pixel under their own
    /// centre instead of native's 0/0 (terrain) or all-zero vote (category).
    /// See `centre_pixels`.
    pub centre_seeded_cells: usize,
}

/// What goes INTO the static: identity, never duration and never a PATH.
///
/// Both exclusions exist for the same reason and are the same rule. A consumer
/// registry pins this file by byte count and sha256, so anything in it that
/// varies with WHEN or WHERE the build ran makes two identical builds into two
/// unregisterable files. A duration varies with the run; an absolute path
/// varies with the machine, which is worse, because a registry pin is
/// cross-machine by nature. `grid_sha256` names the grid better than a path
/// does: it survives the file being moved and it catches the file being
/// replaced. The paths themselves stay on [`StaticBuildReceipt`], which goes
/// to the `--receipt` sidecar.
///
/// The band row count is deliberately ABSENT: it is a function of the host's
/// memory budget, and the same mesh built on two boxes must be the same bytes.
#[derive(Debug, Clone, serde::Serialize)]
pub struct StaticProvenance {
    pub engine: String,
    pub schema: String,
    pub grid_sha256: String,
    pub n_cells: usize,
    pub n_edges: usize,
    pub n_vertices: usize,
    pub sphere_radius: f64,
    pub nominal_dx_m: f64,
    pub nominal_dx_f32_bits: u32,
    pub gwd_water_category: i64,
    pub gwd_water_category_source: String,
    pub gwd_variance_smoothed_cells: usize,
    pub mminlu: String,
    /// Source mpas_init_atm_gwd.F:1126-1128: ny<4 reads row zero.
    /// Proposed word set only; exclusion requires the owner's case-specific sign-off.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub native_undefined_ol1_cells: Vec<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub native_undefined_exclusion_status: Option<String>,
}

#[derive(Debug, serde::Serialize)]
pub struct StaticBuildReceipt {
    pub schema: String,
    pub grid_path: String,
    pub output_path: String,
    pub n_cells: usize,
    pub n_edges: usize,
    pub n_vertices: usize,
    pub max_edges: usize,
    pub grid_scaled_from_unit_sphere: bool,
    pub output_bytes: u64,
    pub datasets: Vec<DatasetReceipt>,
    pub parallel: TileParallelPlan,
    pub tile_map_cache: TileMapCacheReceipt,
    pub memory: MemoryReceipt,
    pub generated_fields: Vec<String>,
    pub operator_provenance: BTreeMap<String, String>,
    pub provenance: String,
    /// The document stamped into the file itself.
    pub stamped_provenance: StaticProvenance,
    pub gwd_seconds: f64,
    pub gwd_band_rows: i64,
    pub gwd_bands: usize,
    pub gwd_band_bytes: u64,
    pub gwd_variance_smoothed_cells: usize,
    /// Present when the grid carries a regional boundary zone: the counts
    /// that say exactly where this static CANNOT match a whole-sphere build,
    /// because the data needed is outside the region.
    pub regional: Option<RegionalStaticNotes>,
    /// What FP32 storage did to this file's own edge lengths, and whether the
    /// consumer's recomputation will accept them.
    pub fp32_metric_agreement: crate::staticfile::fp32metrics::Fp32MetricAgreement,
    pub sha256: String,
    pub variables: Vec<String>,
}

/// The measured regional divergences of one static build, stated so a
/// comparison against a culled whole-sphere static reads them as expected
/// rather than as defects.
#[derive(Debug, Clone, serde::Serialize)]
pub struct RegionalStaticNotes {
    /// Cells at each bdyMaskCell value 0..7.
    pub ring_cell_counts: [usize; 8],
    /// Cells whose deriv_two side is 0 because their neighbour ring reaches a
    /// culled cell (native `atm_initialize_advection_rk` regional behaviour).
    pub deriv_two_zero_stencil_cells: usize,
    /// Outermost-ring cells left out of the GWD isolated-point smooth.
    pub gwd_smooth_skipped_boundary_cells: usize,
}

#[derive(Debug)]
struct Mesh {
    n_cells: usize,
    n_edges: usize,
    n_vertices: usize,
    max_edges: usize,
    max_edges2: usize,
    vertex_degree: usize,
    scaled_from_unit_sphere: bool,
    on_a_sphere: bool,
    sphere_radius: f64,

    lat_cell: Vec<f64>,
    lon_cell: Vec<f64>,
    lat_edge: Vec<f64>,
    lon_edge: Vec<f64>,
    lat_vertex: Vec<f64>,
    lon_vertex: Vec<f64>,
    x_cell: Vec<f64>,
    y_cell: Vec<f64>,
    z_cell: Vec<f64>,
    x_edge: Vec<f64>,
    y_edge: Vec<f64>,
    z_edge: Vec<f64>,
    x_vertex: Vec<f64>,
    y_vertex: Vec<f64>,
    z_vertex: Vec<f64>,
    dc_edge: Vec<f64>,
    dv_edge: Vec<f64>,
    area_cell: Vec<f64>,
    area_triangle: Vec<f64>,
    angle_edge: Vec<f64>,
    mesh_density: Vec<f64>,
    nominal_min_dc: f64,

    n_edges_on_cell: Vec<usize>,
    cells_on_cell: Vec<usize>,
    edges_on_cell: Vec<usize>,
    vertices_on_cell: Vec<usize>,
    cells_on_edge: Vec<[usize; 2]>,
    vertices_on_edge: Vec<[usize; 2]>,

    // ---- topology as the grid file stores it, 1-BASED and VERBATIM --------
    //
    // The arrays above are the zero-based, padding-normalised views the
    // geography and operator passes index with. These are the SAME arrays as
    // the grid file wrote them, and they are what the static carries.
    //
    // The distinction is not tidiness. A consumer cross-examines grid and
    // static for bit-identity of every topology array and refuses the PAIR
    // when they differ, so a re-derived padding slot -- writing 0 where the
    // grid wrote a repeat of the last neighbour, say -- makes a static that
    // is arithmetically right and unusable. Copy, do not reconstruct.
    raw_cells_on_cell: Vec<i32>,
    raw_edges_on_cell: Vec<i32>,
    raw_vertices_on_cell: Vec<i32>,
    raw_cells_on_edge: Vec<i32>,
    raw_vertices_on_edge: Vec<i32>,
    raw_edges_on_edge: Vec<i32>,
    raw_cells_on_vertex: Vec<i32>,
    raw_edges_on_vertex: Vec<i32>,

    n_edges_on_edge: Vec<i32>,
    weights_on_edge: Vec<f64>,
    kite_areas_on_vertex: Vec<f64>,

    index_to_cell_id: Vec<i32>,
    index_to_edge_id: Vec<i32>,
    index_to_vertex_id: Vec<i32>,

    bdy_mask_cell: Vec<i32>,
    bdy_mask_edge: Vec<i32>,
    bdy_mask_vertex: Vec<i32>,
}

fn dim(file: &netcrust::File, name: &str) -> MpasResult<usize> {
    file.dimension(name)
        .map(|d| d.len())
        .ok_or_else(|| MpasError::Refusal(format!("grid has no {name} dimension")))
}

fn f64v(file: &netcrust::File, name: &str) -> MpasResult<Vec<f64>> {
    Ok(file
        .read_array_f64(name)
        .map_err(|e| MpasError::Refusal(format!("grid has no readable {name}: {e}")))?
        .into_values())
}

/// MPAS: mpas_init_atm_static.F:292-310: mesh pools are RKIND.
fn rkindv(file: &netcrust::File, name: &str) -> MpasResult<Vec<f64>> {
    Ok(f64v(file, name)?.into_iter().map(|v| (v as f32) as f64).collect())
}

fn i32v(file: &netcrust::File, name: &str) -> MpasResult<Vec<i32>> {
    Ok(f64v(file, name)?
        .into_iter()
        .map(|v| v.round() as i32)
        .collect())
}

fn optional_i32v(file: &netcrust::File, name: &str, len: usize, fill: i32) -> MpasResult<Vec<i32>> {
    if file.variable(name).is_some() {
        let got = i32v(file, name)?;
        if got.len() != len {
            return Err(MpasError::Refusal(format!(
                "grid {name} has {} values, expected {len}",
                got.len()
            )));
        }
        Ok(got)
    } else {
        Ok(vec![fill; len])
    }
}

/// An integer grid array read verbatim, with its shape held to `want`.
fn shaped_i32(file: &netcrust::File, name: &str, want: usize) -> MpasResult<Vec<i32>> {
    let got = i32v(file, name)?;
    if got.len() != want {
        return Err(MpasError::Refusal(format!(
            "grid {name} has {} values, expected {want}; writing it into a static \
             would shift every later element by the difference",
            got.len()
        )));
    }
    Ok(got)
}

/// A floating grid array read verbatim, with its shape held to `want`.
fn shaped_f64(file: &netcrust::File, name: &str, want: usize) -> MpasResult<Vec<f64>> {
    let got = f64v(file, name)?;
    if got.len() != want {
        return Err(MpasError::Refusal(format!(
            "grid {name} has {} values, expected {want}; writing it into a static \
             would shift every later element by the difference",
            got.len()
        )));
    }
    Ok(got)
}

/// A global-ID array, taken from the grid when it has one and generated as
/// the 1-based identity when it does not.
///
/// Generating is safe and copying is safer. A serial mesh numbers its cells
/// 1..nCells and every published grid stores exactly that, so the identity is
/// the right answer; but a decomposed or renumbered grid carries a permutation
/// there, and re-deriving it would hand a consumer a static whose global IDs
/// disagree with the grid it is paired with.
fn identity_ids(file: &netcrust::File, name: &str, len: usize) -> MpasResult<Vec<i32>> {
    if file.variable(name).is_some() {
        return shaped_i32(file, name, len);
    }
    Ok((1..=len as i32).collect())
}

fn zero_based(raw: &[i32], upper: usize, name: &str, allow_zero_padding: bool) -> MpasResult<Vec<usize>> {
    raw.iter().copied()
        .enumerate()
        .map(|(k, v)| {
            if allow_zero_padding && v == 0 {
                return Ok(usize::MAX);
            }
            if v < 1 || v as usize > upper {
                return Err(MpasError::Refusal(format!(
                    "grid {name}[{k}]={v} is outside 1..={upper}"
                )));
            }
            Ok(v as usize - 1)
        })
        .collect()
}

impl Mesh {
    /// Read `path` at the representation the grid asks its static to carry
    /// ([`crate::staticfile::coordframe::for_static_from_grid`]): the probe and
    /// the tests read a grid the way a build of it would.
    fn read(path: &Path) -> MpasResult<Self> {
        let coordinates = crate::staticfile::coordframe::for_static_from_grid(path)?;
        Self::read_as(path, coordinates)
    }

    /// Read `path` for a static that will DECLARE `coordinates`.
    ///
    /// THE PRECISION A GRID IS READ AT FOLLOWS THE REPRESENTATION ITS STATIC
    /// DECLARES, because the writer stores the coordinates exactly as read
    /// (`put_coord!`) and the declaration is a promise about those bytes.
    ///
    /// * binary32 -- every published mesh and every cull of one -- is the
    ///   native RKIND=4 mesh pool (mpas_init_atm_static.F:292-310): each value
    ///   rounded to binary32 on read and carried to the sphere in binary32.
    ///   This is the arithmetic the i2 replay proves word for word against
    ///   native init_atmosphere, and it is unchanged here.
    /// * binary64 -- a mesh this crate GENERATED, which has no native
    ///   counterpart -- is read and carried to the sphere in binary64, so the
    ///   coordinates reach the file as the binary64 values the declaration
    ///   names, and every metric is rounded to binary32 once, at the write.
    ///
    /// THE BREAKAGE THIS PREVENTS (2.9.0, "a defect
    /// found on the way"): bad5408e05a put EVERY grid on the RKIND read, so the
    /// static of a generated grid declared `binary64_earth_centred` while
    /// holding binary32 positions up to 0.44 m off its own grid. gpuwm-hex's
    /// point door then refused its vertical check ("dcEdge disagrees with
    /// spherical cell-center arc length", 2.6e-4 against a 9.5e-7 bound) and
    /// `rw_mpas_mesh --cull-parent` refused the static's cut ("provenance grid
    /// latCell differs", then "dcEdge differs"), so no 3 km regional mesh could
    /// be built through the door.
    fn read_as(path: &Path, coordinates: CoordinateRepresentation) -> MpasResult<Self> {
        let f = netcrust::File::open(path)?;
        let n_cells = dim(&f, "nCells")?;
        let n_edges = dim(&f, "nEdges")?;
        let n_vertices = dim(&f, "nVertices")?;
        let max_edges = dim(&f, "maxEdges")?;
        let max_edges2 = dim(&f, "maxEdges2")?;
        let vertex_degree = dim(&f, "vertexDegree")?;

        let wide = coordinates.is_binary64();
        let readv = |name: &str| -> MpasResult<Vec<f64>> {
            if wide {
                f64v(&f, name)
            } else {
                rkindv(&f, name)
            }
        };
        // Each variable is an independent byte range. Decode them beside
        // one another instead of spending the first large-mesh stage on one
        // core. The indexed collect preserves the original variable order.
        let names = ["xCell", "yCell", "zCell", "xEdge", "yEdge", "zEdge",
                     "xVertex", "yVertex", "zVertex", "dcEdge", "dvEdge",
                     "areaCell", "areaTriangle", "kiteAreasOnVertex", "nominalMinDc"];
        let decoded: Vec<MpasResult<Vec<f64>>> = names.par_iter().map(|name| readv(name)).collect();
        let values: Vec<Vec<f64>> = decoded.into_iter().collect::<MpasResult<_>>()?;
        let [mut x_cell, mut y_cell, mut z_cell, mut x_edge, mut y_edge, mut z_edge,
             mut x_vertex, mut y_vertex, mut z_vertex, mut dc_edge, mut dv_edge,
             mut area_cell, mut area_triangle, mut kite_areas_on_vertex, nominal_values]: [Vec<f64>; 15] =
            values.try_into().map_err(|_| MpasError::Refusal("coordinate inventory changed".into()))?;
        let mut nominal = nominal_values.first().copied()
            .ok_or_else(|| MpasError::Refusal("grid nominalMinDc is empty".into()))?;

        let sample = x_cell
            .iter()
            .zip(&y_cell)
            .zip(&z_cell)
            .take(64)
            .map(|((&x, &y), &z)| (x * x + y * y + z * z).sqrt())
            .sum::<f64>()
            / x_cell.len().min(64).max(1) as f64;
        let scaled = sample < 10.0;
        let r = MPAS_EARTH_RADIUS_M;
        if scaled {
            // binary64: the product rw_mpas_mesh --cull-parent re-derives when
            // it binds a static's cut to its grid (`after * radius_scale`, then
            // binary32 for a stored metric), so the pair binds bit for bit.
            let length = |v: f64| -> f64 {
                if wide {
                    v * r
                } else {
                    ((v as f32) * (r as f32)) as f64
                }
            };
            let area = |v: f64| -> f64 {
                if wide {
                    v * (r * r)
                } else {
                    ((v as f32) * ((r as f32) * (r as f32))) as f64
                }
            };
            [&mut x_cell, &mut y_cell, &mut z_cell, &mut x_edge, &mut y_edge, &mut z_edge,
             &mut x_vertex, &mut y_vertex, &mut z_vertex].into_par_iter()
                .for_each(|values| values.par_iter_mut().for_each(|v| *v = length(*v)));
            [&mut dc_edge, &mut dv_edge].into_par_iter()
                .for_each(|values| values.par_iter_mut().for_each(|v| *v = length(*v)));
            [&mut area_cell, &mut area_triangle, &mut kite_areas_on_vertex].into_par_iter()
                .for_each(|values| values.par_iter_mut().for_each(|v| *v = area(*v)));
            nominal = length(nominal);
        }

        let neoc_i32 = i32v(&f, "nEdgesOnCell")?;
        if neoc_i32.len() != n_cells {
            return Err(MpasError::Refusal("nEdgesOnCell shape mismatch".to_string()));
        }
        let n_edges_on_cell: Vec<usize> = neoc_i32.into_par_iter().map(|v| v as usize).collect();
        let cells_on_cell_raw = i32v(&f, "cellsOnCell")?;
        let edges_on_cell_raw = i32v(&f, "edgesOnCell")?;
        let vertices_on_cell_raw = i32v(&f, "verticesOnCell")?;
        for (name, len) in [
            ("cellsOnCell", cells_on_cell_raw.len()),
            ("edgesOnCell", edges_on_cell_raw.len()),
            ("verticesOnCell", vertices_on_cell_raw.len()),
        ] {
            if len != n_cells * max_edges {
                return Err(MpasError::Refusal(format!(
                    "{name} has {len} values, expected {}",
                    n_cells * max_edges
                )));
            }
        }

        // The boundary masks come BEFORE the topology normalisation: on a
        // regional (culled) mesh the sentinel rules below are keyed off them.
        let bdy_mask_cell = optional_i32v(&f, "bdyMaskCell", n_cells, 0)?;
        let bdy_mask_edge = optional_i32v(&f, "bdyMaskEdge", n_edges, 0)?;
        let bdy_mask_vertex = optional_i32v(&f, "bdyMaskVertex", n_vertices, 0)?;

        // Padding slots in MPAS topology are often zero.  Replace them with a
        // harmless zero-based 0 only after active slots are validated.
        //
        // `absent_ok`: on a regional mesh, a VALID slot of 0 is the culler's
        // "that neighbour was culled" sentinel and is legitimate exactly on
        // outermost-ring cells (bdyMaskCell == 7). Those slots become
        // ABSENT_NEIGHBOR so every stencil that reads them has to decide, and
        // can never silently read cell 0.
        let normalize_padded = |raw: &[i32],
                                upper: usize,
                                name: &str,
                                absent_ok: Option<&[i32]>|
         -> MpasResult<Vec<usize>> {
            let mut out = vec![0usize; raw.len()];
            let error = out.par_chunks_mut(max_edges).enumerate().map(|(c, row)| -> MpasResult<()> {
                let used = n_edges_on_cell[c];
                if used > max_edges {
                    return Err(MpasError::Refusal(format!(
                        "cell {c} nEdgesOnCell={used} exceeds maxEdges={max_edges}"
                    )));
                }
                for s in 0..max_edges {
                    let k = c * max_edges + s;
                    if s < used {
                        let v = raw[k];
                        if v == 0 {
                            if let Some(mask) = absent_ok {
                                if mask[c] == REGIONAL_OUTERMOST_MASK {
                                    row[s] = ABSENT_NEIGHBOR;
                                    continue;
                                }
                                return Err(MpasError::Refusal(format!(
                                    "grid {name}[cell={c},slot={s}]=0 in a VALID slot of a \
                                     cell with bdyMaskCell={}; only outermost regional cells \
                                     (mask {REGIONAL_OUTERMOST_MASK}) may carry absent \
                                     neighbours -- anywhere else the mesh is torn and every \
                                     stencil over it would read a cell that does not exist",
                                    mask[c]
                                )));
                            }
                        }
                        if v < 1 || v as usize > upper {
                            return Err(MpasError::Refusal(format!(
                                "grid {name}[cell={c},slot={s}]={v} is outside 1..={upper}"
                            )));
                        }
                        row[s] = v as usize - 1;
                    }
                }
                Ok(())
            }).find_first(|result| result.is_err());
            if let Some(error) = error { error?; }
            Ok(out)
        };

        let raw_cells_on_cell = cells_on_cell_raw;
        let raw_edges_on_cell = edges_on_cell_raw;
        let raw_vertices_on_cell = vertices_on_cell_raw;
        let cells_on_cell =
            normalize_padded(&raw_cells_on_cell, n_cells, "cellsOnCell", Some(&bdy_mask_cell))?;
        let edges_on_cell = normalize_padded(&raw_edges_on_cell, n_edges, "edgesOnCell", None)?;
        let vertices_on_cell =
            normalize_padded(&raw_vertices_on_cell, n_vertices, "verticesOnCell", None)?;

        let coe_raw = i32v(&f, "cellsOnEdge")?;
        let voe_raw = i32v(&f, "verticesOnEdge")?;
        if coe_raw.len() != 2 * n_edges || voe_raw.len() != 2 * n_edges {
            return Err(MpasError::Refusal("edge-pair topology shape mismatch".to_string()));
        }
        let raw_cells_on_edge = coe_raw;
        let raw_vertices_on_edge = voe_raw;
        // A rim edge of a regional mesh has ONE cell; the culler stores 0 in
        // the other slot, and only on outermost (mask 7) edges.
        let mut coe = vec![0usize; 2 * n_edges];
        for e in 0..n_edges {
            for s in 0..2 {
                let v = raw_cells_on_edge[2 * e + s];
                if v == 0 {
                    if bdy_mask_edge[e] == REGIONAL_OUTERMOST_MASK {
                        coe[2 * e + s] = ABSENT_NEIGHBOR;
                        continue;
                    }
                    return Err(MpasError::Refusal(format!(
                        "grid cellsOnEdge[edge={e},slot={s}]=0 on an edge with \
                         bdyMaskEdge={}; only outermost regional edges (mask \
                         {REGIONAL_OUTERMOST_MASK}) are one-sided -- anywhere else \
                         the mesh is torn",
                        bdy_mask_edge[e]
                    )));
                }
                if v < 1 || v as usize > n_cells {
                    return Err(MpasError::Refusal(format!(
                        "grid cellsOnEdge[{}]={v} is outside 1..={n_cells}",
                        2 * e + s
                    )));
                }
                coe[2 * e + s] = v as usize - 1;
            }
        }
        let voe = zero_based(&raw_vertices_on_edge, n_vertices, "verticesOnEdge", false)?;
        let cells_on_edge = coe.par_chunks_exact(2).map(|row| [row[0], row[1]]).collect();
        let vertices_on_edge = voe.par_chunks_exact(2).map(|row| [row[0], row[1]]).collect();
        drop(coe);
        drop(voe);

        // The mesh DUAL, read verbatim. A static that omits it leaves the
        // dycore with no vertex triangles: `cellsOnVertex` and
        // `kiteAreasOnVertex` are what the vorticity and the potential-
        // vorticity flux are built on, and `weightsOnEdge` / `edgesOnEdge`
        // are the tangential reconstruction stencil itself.
        let raw_edges_on_edge = shaped_i32(&f, "edgesOnEdge", n_edges * max_edges2)?;
        let raw_cells_on_vertex = shaped_i32(&f, "cellsOnVertex", n_vertices * vertex_degree)?;
        let raw_edges_on_vertex = shaped_i32(&f, "edgesOnVertex", n_vertices * vertex_degree)?;
        let n_edges_on_edge = shaped_i32(&f, "nEdgesOnEdge", n_edges)?;
        let weights_on_edge = shaped_f64(&f, "weightsOnEdge", n_edges * max_edges2)?;
        if kite_areas_on_vertex.len() != n_vertices * vertex_degree {
            return Err(MpasError::Refusal(format!(
                "grid kiteAreasOnVertex has {} values, expected {}",
                kite_areas_on_vertex.len(),
                n_vertices * vertex_degree
            )));
        }

        let mesh = Self {
            n_cells,
            n_edges,
            n_vertices,
            max_edges,
            max_edges2,
            vertex_degree,
            scaled_from_unit_sphere: scaled,
            // MPAS: mpas_bootstrapping.F:287-298: absent means spherical;
            // otherwise the native test is a case-sensitive YES substring.
            on_a_sphere: f.attribute("on_a_sphere").map(|a| a.as_string().unwrap_or("").contains("YES")).unwrap_or(true),
            sphere_radius: r,
            lat_cell: readv("latCell")?,
            lon_cell: readv("lonCell")?,
            lat_edge: readv("latEdge")?,
            lon_edge: readv("lonEdge")?,
            lat_vertex: readv("latVertex")?,
            lon_vertex: readv("lonVertex")?,
            x_cell,
            y_cell,
            z_cell,
            x_edge,
            y_edge,
            z_edge,
            x_vertex,
            y_vertex,
            z_vertex,
            dc_edge,
            dv_edge,
            area_cell,
            area_triangle,
            angle_edge: readv("angleEdge")?,
            mesh_density: readv("meshDensity")?,
            nominal_min_dc: nominal,
            n_edges_on_cell,
            cells_on_cell,
            edges_on_cell,
            vertices_on_cell,
            cells_on_edge,
            vertices_on_edge,
            raw_cells_on_cell,
            raw_edges_on_cell,
            raw_vertices_on_cell,
            raw_cells_on_edge,
            raw_vertices_on_edge,
            raw_edges_on_edge,
            raw_cells_on_vertex,
            raw_edges_on_vertex,
            n_edges_on_edge,
            weights_on_edge,
            kite_areas_on_vertex,
            index_to_cell_id: identity_ids(&f, "indexToCellID", n_cells)?,
            index_to_edge_id: identity_ids(&f, "indexToEdgeID", n_edges)?,
            index_to_vertex_id: identity_ids(&f, "indexToVertexID", n_vertices)?,
            bdy_mask_cell,
            bdy_mask_edge,
            bdy_mask_vertex,
        };
        mesh.validate()?;
        Ok(mesh)
    }

    fn validate(&self) -> MpasResult<()> {
        for (name, len, want) in [
            ("latCell", self.lat_cell.len(), self.n_cells),
            ("lonCell", self.lon_cell.len(), self.n_cells),
            ("latEdge", self.lat_edge.len(), self.n_edges),
            ("lonEdge", self.lon_edge.len(), self.n_edges),
            ("latVertex", self.lat_vertex.len(), self.n_vertices),
            ("lonVertex", self.lon_vertex.len(), self.n_vertices),
        ] {
            if len != want {
                return Err(MpasError::Refusal(format!(
                    "grid {name} has {len} values, expected {want}"
                )));
            }
        }
        // Regional (limited-area) admission. The mpas_in_cell containment the
        // old blanket bdyMaskCell==7 refusal named as missing is ported now
        // (see `point_in_cell` and its use in `tile_map`), so a regional mesh
        // is admitted -- against the sentinel geometry the culler declares,
        // with every violation named.
        for (name, mask) in [
            ("bdyMaskCell", &self.bdy_mask_cell),
            ("bdyMaskEdge", &self.bdy_mask_edge),
            ("bdyMaskVertex", &self.bdy_mask_vertex),
        ] {
            if let Some((i, &v)) = mask
                .iter()
                .enumerate()
                .find(|&(_, &v)| !(0..=REGIONAL_OUTERMOST_MASK).contains(&v))
            {
                return Err(MpasError::Refusal(format!(
                    "{name}[{i}]={v} is outside 0..={REGIONAL_OUTERMOST_MASK}; the \
                     boundary-zone stages would index a relaxation ring that does \
                     not exist"
                )));
            }
        }
        let regional = self.bdy_mask_cell.iter().any(|&v| v != 0);
        if regional {
            // edgesOnCell, verticesOnCell and verticesOnEdge never carry a
            // zero valid slot on ANY mesh -- every edge and vertex of a kept
            // cell is kept by the cull -- and the reader above already
            // refused them by name. What is left to check here are the raw
            // dual rows the reader carries verbatim.
            //
            // The sentinel-bearing dual rows: zeros are legitimate only on
            // outermost (mask 7) elements. Anywhere else the boundary zone
            // would start inside the relaxation rings the dycore nudges.
            for e in 0..self.n_edges {
                let used = (self.n_edges_on_edge[e].max(0) as usize).min(self.max_edges2);
                let row = &self.raw_edges_on_edge[e * self.max_edges2..e * self.max_edges2 + used];
                if row.iter().any(|&v| v == 0) && self.bdy_mask_edge[e] != REGIONAL_OUTERMOST_MASK {
                    return Err(MpasError::Refusal(format!(
                        "regional grid edgesOnEdge row {e} carries a 0 inside its \
                         declared nEdgesOnEdge={used} but bdyMaskEdge={}; absent \
                         reconstruction stencils belong to mask-{REGIONAL_OUTERMOST_MASK} \
                         edges only",
                        self.bdy_mask_edge[e]
                    )));
                }
            }
            for vx in 0..self.n_vertices {
                let base = vx * self.vertex_degree;
                let cz = self.raw_cells_on_vertex[base..base + self.vertex_degree]
                    .iter()
                    .any(|&v| v == 0);
                let ez = self.raw_edges_on_vertex[base..base + self.vertex_degree]
                    .iter()
                    .any(|&v| v == 0);
                if (cz || ez) && self.bdy_mask_vertex[vx] != REGIONAL_OUTERMOST_MASK {
                    return Err(MpasError::Refusal(format!(
                        "regional grid vertex {vx} has absent adjacent cells/edges but \
                         bdyMaskVertex={}; one-sided vertices belong to \
                         mask-{REGIONAL_OUTERMOST_MASK} only",
                        self.bdy_mask_vertex[vx]
                    )));
                }
            }
        }
        Ok(())
    }

    /// True when this mesh carries a regional boundary zone.
    fn is_regional(&self) -> bool {
        self.bdy_mask_cell.iter().any(|&v| v != 0)
    }

    fn operator_view(&self) -> OperatorMesh<'_> {
        OperatorMesh {
            n_cells: self.n_cells,
            n_edges: self.n_edges,
            max_edges: self.max_edges,
            sphere_radius: self.sphere_radius,
            on_a_sphere: self.on_a_sphere,
            n_edges_on_cell: &self.n_edges_on_cell,
            cells_on_cell: &self.cells_on_cell,
            edges_on_cell: &self.edges_on_cell,
            cells_on_edge: &self.cells_on_edge,
            vertices_on_cell: &self.vertices_on_cell,
            vertices_on_edge: &self.vertices_on_edge,
            x_cell: &self.x_cell,
            y_cell: &self.y_cell,
            z_cell: &self.z_cell,
            x_vertex: &self.x_vertex,
            y_vertex: &self.y_vertex,
            z_vertex: &self.z_vertex,
            angle_edge: &self.angle_edge,
            dc_edge: &self.dc_edge,
        }
    }

    /// MPAS: mpas_init_atm_static.F:314-329,2241-2289.
    fn max_search_distance2(&self) -> f64 {
        // Each arc keeps the native binary32 arithmetic. A maximum over
        // nonnegative distances has the same words in any reduction order;
        // avoid a whole-mesh trigonometric pass on the protocol thread.
        let max_r = (0..self.n_cells).into_par_iter().map(|c| {
            let mut max_r = 0.0f32;
            for s in 0..self.n_edges_on_cell[c] {
                let v = self.vertices_on_cell[c * self.max_edges + s];
                let arc = crate::static_native::sphere_distance(
                    self.lat_cell[c] as f32, self.lon_cell[c] as f32,
                    self.lat_vertex[v] as f32, self.lon_vertex[v] as f32,
                    self.sphere_radius as f32);
                max_r = max_r.max(arc);
            }
            max_r
        }).reduce(|| 0.0f32, f32::max);
        let diameter = 2.0 * max_r;
        (2.0 * (diameter * diameter)) as f64
    }

}

/// What a grid file declares, through the same read-and-validate path the
/// static builder itself uses.
#[derive(Debug, Clone, serde::Serialize)]
pub struct GridProbe {
    pub n_cells: usize,
    pub n_edges: usize,
    pub n_vertices: usize,
    /// True when the grid carries a regional boundary zone.
    pub regional: bool,
    /// Cells at each bdyMaskCell value 0..7. All in slot 0 for a global mesh.
    pub ring_cell_counts: [usize; 8],
    /// cellsOnCell valid slots pointing at culled cells.
    pub absent_neighbor_slots: usize,
    /// Edges with one culled side.
    pub one_sided_edges: usize,
}

/// Admit or refuse a grid through the static builder's own reader, WITHOUT
/// touching any geography. This is the probe a door runs before hours of
/// tile streaming: a regional grid whose sentinel geometry is torn is refused
/// here with the breakage named, instead of after the archive was read.
pub fn probe_grid(path: &Path) -> MpasResult<GridProbe> {
    let mesh = Mesh::read(path)?;
    let mut ring_cell_counts = [0usize; 8];
    for &m in &mesh.bdy_mask_cell {
        ring_cell_counts[m as usize] += 1;
    }
    let mut absent_neighbor_slots = 0usize;
    for c in 0..mesh.n_cells {
        for s in 0..mesh.n_edges_on_cell[c] {
            if mesh.cells_on_cell[c * mesh.max_edges + s] == ABSENT_NEIGHBOR {
                absent_neighbor_slots += 1;
            }
        }
    }
    let one_sided_edges = (0..mesh.n_edges)
        .filter(|&e| mesh.cells_on_edge[e].contains(&ABSENT_NEIGHBOR))
        .count();
    Ok(GridProbe {
        n_cells: mesh.n_cells,
        n_edges: mesh.n_edges,
        n_vertices: mesh.n_vertices,
        regional: mesh.is_regional(),
        ring_cell_counts,
        absent_neighbor_slots,
        one_sided_edges,
    })
}

// ---------------------------------------------------------------------------
// 3-D kd tree; one compact permutation and split-axis array.
// ---------------------------------------------------------------------------

struct KdTree {
    points: Vec<[f64; 3]>,
    order: Vec<u32>,
    axis: Vec<u8>,
}

/// MPAS: mpas_kd_tree.F:64-159: cyclic axes, lower median.
fn kd_build(points: &[[f64; 3]], order: &mut [u32], axis: &mut [u8], lo: usize, hi: usize) {
    kd_build_axis(points, order, axis, lo, hi, 0);
}
/// MPAS: mpas_kd_tree.F:64-159 (recursive cyclic-axis construction).
fn kd_build_axis(points: &[[f64; 3]], order: &mut [u32], axis: &mut [u8], lo: usize, hi: usize, ax: usize) {
    kd_build_slice(points, &mut order[lo..hi], &mut axis[lo..hi], ax);
}

fn kd_build_slice(points: &[[f64; 3]], order: &mut [u32], axis: &mut [u8], ax: usize) {
    if order.is_empty() { return; }
    let mid = (order.len() - 1) / 2;
    order.select_nth_unstable_by(mid, |&a,&b| {
        (points[a as usize][ax] as f32).total_cmp(&(points[b as usize][ax] as f32))
    });
    axis[mid] = ax as u8;
    let parallel = order.len() >= 16_384;
    let (left, rest) = order.split_at_mut(mid);
    let (left_axis, rest_axis) = axis.split_at_mut(mid);
    let right = &mut rest[1..];
    let right_axis = &mut rest_axis[1..];
    if parallel {
        rayon::join(|| kd_build_slice(points, left, left_axis, (ax+1)%3),
                    || kd_build_slice(points, right, right_axis, (ax+1)%3));
    } else {
        kd_build_slice(points, left, left_axis, (ax+1)%3);
        kd_build_slice(points, right, right_axis, (ax+1)%3);
    }
}

impl KdTree {
    /// MPAS: mpas_kd_tree.F:64-159 (mpas_kd_build_tree).
    fn new(mesh: &Mesh) -> Self {
        let points: Vec<[f64; 3]> = (0..mesh.n_cells).into_par_iter()
            .map(|c| [mesh.x_cell[c], mesh.y_cell[c], mesh.z_cell[c]])
            .collect();
        let mut order: Vec<u32> = (0..mesh.n_cells as u32).collect();
        let mut axis = vec![0u8; order.len()];
        let n = order.len();
        kd_build(&points, &mut order, &mut axis, 0, n);
        Self { points, order, axis }
    }

    #[cfg(test)]
    fn nearest(&self, target: [f64; 3]) -> (usize, f64) {
        self.nearest_bounded(target, f64::INFINITY).expect("nonempty KD tree")
    }

    /// MPAS: mpas_kd_tree.F:285-332: initialize distance to the supplied bound.
    fn nearest_bounded(&self, target: [f64; 3], limit: f64) -> Option<(usize,f64)> {
        let mut best = (usize::MAX, limit);
        self.search(0, self.order.len(), target, &mut best);
        (best.0 != usize::MAX).then_some(best)
    }

    /// MPAS: mpas_kd_tree.F:226-279 (mpas_kd_search_internal).
    fn search(&self, lo: usize, hi: usize, t: [f64; 3], best: &mut (usize, f64)) {
        if hi <= lo {
            return;
        }
        let mid = lo + (hi - lo - 1) / 2;
        let idx = self.order[mid] as usize;
        let p = self.points[idx];
        let d = crate::static_native::distance2(p.map(|v| v as f32), t.map(|v| v as f32)) as f64;
        // MPAS: mpas_kd_tree.F:175-201,226-279: lexicographic coordinate tie.
        let lex_less = best.0 != usize::MAX && p < self.points[best.0];
        if d < best.1 || (d == best.1 && lex_less) { *best = (idx, d); }
        let a = self.axis[mid] as usize;
        let delta = t[a] as f32 - p[a] as f32;
        let (nlo, nhi, flo, fhi) = if delta < 0.0 {
            (lo, mid, mid + 1, hi)
        } else {
            (mid + 1, hi, lo, mid)
        };
        self.search(nlo, nhi, t, best);
        if (delta * delta) as f64 <= best.1 {
            self.search(flo, fhi, t, best);
        }
    }
}

#[cfg(test)]
fn xyz(radius: f64, lat: f64, lon: f64) -> [f64; 3] {
    let c = lat.cos();
    [radius * c * lon.cos(), radius * c * lon.sin(), radius * lat.sin()]
}

/// MPAS: src/operators/mpas_geometry_utils.F:1821-1868 (mpas_in_cell).
fn point_in_cell(mesh: &Mesh, cell: usize, p: [f64; 3]) -> bool {
    let cc = [mesh.x_cell[cell] as f32, mesh.y_cell[cell] as f32, mesh.z_cell[cell] as f32];
    let vertices: Vec<[f32; 3]> = (0..mesh.n_edges_on_cell[cell]).map(|i| {
        let v = mesh.vertices_on_cell[cell * mesh.max_edges + i];
        [mesh.x_vertex[v] as f32, mesh.y_vertex[v] as f32, mesh.z_vertex[v] as f32]
    }).collect();
    crate::static_native::in_cell(p.map(|v| v as f32), cc, &vertices)
}

/// Destination cell for every supersampled subpixel of one tile interior.
/// Map layout is `[interior_pixel * factor^2 + subpixel]`.
/// The supersample factor squares into the per-tile destination map, so it is
/// the one knob that can turn a bounded tile into an unbounded allocation.
fn validate_supersample(factor: usize) -> MpasResult<()> {
    // MPAS: mpas_init_atm_static.F:1404-1410 permits an empty sampling
    // loop. Only the Rust allocation's address-space overflow is refused.
    if factor.checked_mul(factor).is_none() {
        return Err(MpasError::Refusal(format!(
            "supersample factor {factor} squared overflows the tile-map address space"
        )));
    }
    Ok(())
}

/// MPAS: mpas_init_atm_static.F:1294-1461 (init_atm_map_static_data).
/// Rust visits archive tiles directly; integer accumulations retain native words.
///
/// Two things keep a REGIONAL mesh off one core (WOOF run controls:
/// a 937.5 m region of 382k cells spent 590 s in its
/// halo static, every 30-second dataset building 648 tile maps of 5.8 M kd
/// lookups each, and the few tiles under the fine core finishing on one or two
/// threads while the rest of the pool sat idle):
///
/// * A tile no pixel of which can reach a cell is answered without a lookup.
///   Its centre's nearest cell, searched within `sqrt(max_d2) + R + slack`
///   (R contains the tile from its centre), is absent, so by the triangle
///   inequality every pixel is farther than `sqrt(max_d2)` from every cell:
///   `nearest_bounded` would refuse each one, and the map is all `-1`. The
///   slack (1 km) is many orders above the f32 rounding of the coordinates.
/// * `parallel_rows`: the rows of a tile are filled on the pool. Each row is
///   the same pure function of its pixels, written to its own slice, so the
///   map is the same `Vec` in the same order.
fn tile_map(
    mesh: &Mesh,
    tree: &KdTree,
    max_distance2: f64,
    ds: &GeogDataset,
    tile: &TileRef,
    factor: usize,
    parallel_rows: bool,
) -> MpasResult<Vec<i32>> {
    validate_supersample(factor)?;
    let pixels = (ds.index.tile_x as usize)
        .checked_mul(ds.index.tile_y as usize)
        .and_then(|v| v.checked_mul(factor * factor))
        .ok_or_else(|| MpasError::Refusal("tile-map element count overflow".to_string()))?;
    let mut out: Vec<i32> = Vec::new();
    out.try_reserve_exact(pixels).map_err(|e| {
        MpasError::Refusal(format!(
            "host memory allocation failed for tile destination map: {pixels} i32 values: {e}"
        ))
    })?;
    if tile_reaches_no_cell(mesh, tree, max_distance2, ds, tile, factor) {
        out.resize(pixels, -1);
        return Ok(out);
    }
    out.resize(pixels, -1);
    let row = ds.index.tile_x as usize * factor * factor;
    if row == 0 {
        return Ok(out);
    }
    if !mesh.is_regional() && std::env::var_os(KD_ONLY_ENV).is_none() {
        let walk = GlobalWalk::new(mesh, tree, max_distance2, ds, tile, factor);
        if parallel_rows {
            out.par_chunks_mut(row).enumerate().for_each(|(j, dst)| walk.fill_row(j, dst));
        } else {
            for (j, dst) in out.chunks_mut(row).enumerate() {
                walk.fill_row(j, dst);
            }
        }
        return Ok(out);
    }
    let fill = |j: usize, dst: &mut [i32]| {
        let mut k = 0usize;
        for i in 0..ds.index.tile_x as usize {
            for sj in 0..factor {
                for si in 0..factor {
                    let (lat, lon) = ds.tile_to_latlon(tile, i * factor + si, j * factor + sj, factor);
                    let p = crate::static_native::xyz(mesh.sphere_radius as f32, lat, lon).map(f64::from);
                    dst[k] = match tree.nearest_bounded(p, max_distance2) {
                        None => -1,
                        Some((cell, d2)) => {
                            // The native regional rule (init_atm_static, nBdyLayers=7):
                            // a pixel is accepted unconditionally at mask < 7, and on
                            // an outermost-ring cell only when it is geometrically
                            // INSIDE that cell -- otherwise its true owner was culled
                            // and the pixel belongs to nobody in this mesh.
                            let accepted = d2 < max_distance2
                                && (mesh.bdy_mask_cell[cell] != REGIONAL_OUTERMOST_MASK
                                    || point_in_cell(mesh, cell, p));
                            if accepted { cell as i32 } else { -1 }
                        }
                    };
                    k += 1;
                }
            }
        }
    };
    if parallel_rows {
        out.par_chunks_mut(row).enumerate().for_each(|(j, dst)| fill(j, dst));
    } else {
        for (j, dst) in out.chunks_mut(row).enumerate() {
            fill(j, dst);
        }
    }
    Ok(out)
}

/// Set to any value to map every pixel through the kd search alone (the
/// reference the walk below is checked against; same bytes, slower).
const KD_ONLY_ENV: &str = "RW_MPAS_STATIC_KD_ONLY";

/// The destination map of a GLOBAL mesh's tile by walking the cell graph from
/// the previous pixel's cell, with the kd search kept as the authority.
///
/// THE BREAKAGE THIS PREVENTS (moving start member, 2026-10-25): on a
/// 16-core box the static of a 98,073-cell storm-following mesh took 66 s, and
/// 38 s of it was three tile-map passes, each a full kd search (about 220 ns)
/// for each of the 933 million 30-arc-second pixels; the static is the long
/// pole of "Preparing the start member" once the mesh is premade.
///
/// Why the bytes cannot change. Neighbouring pixels almost always share a
/// cell, so the walk starts at the previous pixel's cell and steps to the
/// neighbour (cellsOnCell) with the smallest distance until none is smaller:
/// on a global Voronoi mesh a cell no Delaunay neighbour beats is the nearest
/// generator. The kd search ranks the f32-rounded cell centres with f32
/// distances, so the walk's answer is taken only when every neighbour is at
/// least WALK_MARGIN_M farther than the cell found: that margin exceeds twice
/// the f32 rounding of a centre (0.43 m at the earth's radius) plus the f32
/// distance error, so no other cell can rank level with or ahead of it in the
/// kd search's own arithmetic. Anything closer than the margin (a pixel on a
/// Voronoi edge, a lexicographic tie) is answered by the kd search itself.
/// Regional meshes keep the kd path: a culled neighbour ends a walk early.
/// The pixel's position is the native `xyz` of the native `tile_latlon`
/// (latitude from the row, longitude from the column), its cosines and sines
/// computed once per row and column with the same f32 operations in the same
/// order.
const WALK_MARGIN_M: f64 = 3.0;

struct GlobalWalk<'a> {
    mesh: &'a Mesh,
    tree: &'a KdTree,
    max_distance2: f64,
    factor: usize,
    nx: usize,
    /// Per column: radius * cos(lon), radius * sin(lon) (f32, native order).
    rcos_lon: Vec<f32>,
    rsin_lon: Vec<f32>,
    /// Per row: cos(lat), radius * sin(lat) (f32).
    cos_lat: Vec<f32>,
    rsin_lat: Vec<f32>,
    cells: Vec<[f32; 3]>,
}

impl<'a> GlobalWalk<'a> {
    fn new(mesh: &'a Mesh, tree: &'a KdTree, max_distance2: f64, ds: &GeogDataset, tile: &TileRef,
           factor: usize) -> Self {
        let nx = ds.index.tile_x as usize * factor;
        let ny = ds.index.tile_y as usize * factor;
        let radius = mesh.sphere_radius as f32;
        let mut rcos_lon = Vec::with_capacity(nx);
        let mut rsin_lon = Vec::with_capacity(nx);
        for ix in 0..nx {
            let (_, lon) = ds.tile_to_latlon(tile, ix, 0, factor);
            rcos_lon.push(radius * lon.cos());
            rsin_lon.push(radius * lon.sin());
        }
        let mut cos_lat = Vec::with_capacity(ny);
        let mut rsin_lat = Vec::with_capacity(ny);
        for iy in 0..ny {
            let (lat, _) = ds.tile_to_latlon(tile, 0, iy, factor);
            cos_lat.push(lat.cos());
            rsin_lat.push(radius * lat.sin());
        }
        let cells = (0..mesh.n_cells)
            .map(|c| [mesh.x_cell[c] as f32, mesh.y_cell[c] as f32, mesh.z_cell[c] as f32])
            .collect();
        Self { mesh, tree, max_distance2, factor, nx, rcos_lon, rsin_lon, cos_lat, rsin_lat, cells }
    }

    #[inline]
    fn d2(&self, cell: usize, t: [f32; 3]) -> f32 {
        crate::static_native::distance2(self.cells[cell], t)
    }

    /// The kd search's answer for `t`, when the walk from `seed` proves it; None to ask the kd search.
    #[inline]
    fn walk(&self, seed: usize, t: [f32; 3]) -> Option<(usize, f32)> {
        let mesh = self.mesh;
        let mut cell = seed;
        let mut best = self.d2(cell, t);
        loop {
            let mut next = usize::MAX;
            let mut next_d = best;
            let mut runner_up = f32::INFINITY;
            let base = cell * mesh.max_edges;
            for k in 0..mesh.n_edges_on_cell[cell] {
                let n = mesh.cells_on_cell[base + k];
                if n >= mesh.n_cells {
                    return None;
                }
                let d = self.d2(n, t);
                if d < next_d {
                    next_d = d;
                    next = n;
                }
                runner_up = runner_up.min(d);
            }
            if next == usize::MAX {
                // A local minimum: accepted only with every neighbour clearly farther.
                let gap = (runner_up as f64).sqrt() - (best as f64).sqrt();
                return (gap > WALK_MARGIN_M).then_some((cell, best));
            }
            cell = next;
            best = next_d;
        }
    }

    fn fill_row(&self, j: usize, dst: &mut [i32]) {
        let factor = self.factor;
        let mut seed: Option<usize> = None;
        let mut k = 0usize;
        for i in 0..self.nx / factor {
            for sj in 0..factor {
                let iy = j * factor + sj;
                let (clat, z) = (self.cos_lat[iy], self.rsin_lat[iy]);
                for si in 0..factor {
                    let ix = i * factor + si;
                    let t32 = [self.rcos_lon[ix] * clat, self.rsin_lon[ix] * clat, z];
                    let found = match seed.and_then(|c| self.walk(c, t32)) {
                        Some((cell, d)) => ((d as f64) < self.max_distance2).then_some((cell, d as f64)),
                        None => self.tree.nearest_bounded(t32.map(f64::from), self.max_distance2),
                    };
                    dst[k] = match found {
                        // Global mesh: every bdyMaskCell is 0, so the native regional rule accepts any
                        // pixel the bounded search returned (tile_map's d2 < max_distance2).
                        Some((cell, d2)) if d2 < self.max_distance2 => {
                            seed = Some(cell);
                            cell as i32
                        }
                        _ => -1,
                    };
                    k += 1;
                }
            }
        }
    }
}

/// True when no pixel of `tile` can lie within `sqrt(max_distance2)` of any
/// cell (see [`tile_map`]). Decided from the tile's centre and corners and one
/// bounded kd search; `false` whenever the bound is not finite.
fn tile_reaches_no_cell(
    mesh: &Mesh,
    tree: &KdTree,
    max_distance2: f64,
    ds: &GeogDataset,
    tile: &TileRef,
    factor: usize,
) -> bool {
    if !max_distance2.is_finite() || max_distance2 <= 0.0 {
        return false;
    }
    let nx = ds.index.tile_x as usize * factor;
    let ny = ds.index.tile_y as usize * factor;
    if nx == 0 || ny == 0 {
        return false;
    }
    // The corner bound below holds only while the tile spans at most 180
    // degrees of longitude (beyond that the farthest point can be an edge
    // midpoint or the antipode, e.g. (0, 180) of a one-tile global grid
    // centred at (0, 0), 2 R away against the corners' 1.41 R). Release
    // review 14 (2.9.7): such a tile was skipped and a CONUS region got no
    // samples. Tiles wider than 90 degrees either way take the full lookup.
    if ds.index.tile_x as f64 * ds.index.dx.abs() > 90.0
        || ds.index.tile_y as f64 * ds.index.dy.abs() > 90.0
    {
        return false;
    }
    let at = |i: usize, j: usize| -> [f64; 3] {
        let (lat, lon) = ds.tile_to_latlon(tile, i, j, factor);
        crate::static_native::xyz(mesh.sphere_radius as f32, lat, lon).map(f64::from)
    };
    let d = |a: [f64; 3], b: [f64; 3]| {
        ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
    };
    let centre = at(nx / 2, ny / 2);
    // On a latitude-longitude box the farthest point from an interior point
    // is a corner; the 1 % and the one-pixel pad cover the corner pixels'
    // own extent and the sampling offsets.
    let corners = [at(0, 0), at(nx - 1, 0), at(0, ny - 1), at(nx - 1, ny - 1)];
    let reach = corners.iter().map(|&c| d(centre, c)).fold(0.0f64, f64::max);
    let pixel = d(at(0, 0), at(1.min(nx - 1), 1.min(ny - 1))).max(1.0);
    const SLACK_M: f64 = 1000.0;
    let bound = max_distance2.sqrt() + reach * 1.01 + pixel + SLACK_M;
    if !bound.is_finite() {
        return false;
    }
    tree.nearest_bounded(centre, bound * bound).is_none()
}

// ---------------------------------------------------------------------------
// Reusable tile maps, and the bounded pool that fills them.
// ---------------------------------------------------------------------------

/// A destination map identified by what actually determines it.
///
/// Not by dataset name: the 30-arcsec products share tile geometry, so nine
/// datasets would otherwise pay for the same kd search nine times.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct TileMapKey {
    geometry: SourceGeometry,
    xs: i64,
    ys: i64,
    factor: usize,
}

/// A cached map, run-length encoded when that is smaller.
///
/// A destination cell of a global mesh spans many source pixels, so a scan
/// line crosses few cells and the raw map is enormously redundant.  Storing
/// the runs is what lets every tile of every shared-geometry dataset stay
/// cached inside a budget the admission gate has actually granted.  When a
/// mesh is fine enough that runs stop paying -- source pixels coarser than
/// destination cells -- the raw form is kept instead, so the encoding can
/// never cost more than not having it.
#[derive(Debug)]
enum CachedTileMap {
    Runs(Vec<(i32, u32)>),
    Raw(Vec<i32>),
}

impl CachedTileMap {
    fn encode(map: &[i32]) -> MpasResult<Self> {
        let mut runs: Vec<(i32, u32)> = Vec::new();
        let raw_cap = map.len() / 2 + 1; // where runs stop being smaller
        for &c in map {
            match runs.last_mut() {
                Some((cell, len)) if *cell == c && *len < u32::MAX => *len += 1,
                _ => {
                    if runs.len() >= raw_cap {
                        return Ok(CachedTileMap::Raw(map.to_vec()));
                    }
                    runs.push((c, 1));
                }
            }
        }
        Ok(CachedTileMap::Runs(runs))
    }

    fn bytes(&self) -> u64 {
        match self {
            CachedTileMap::Runs(r) => (r.len() * std::mem::size_of::<(i32, u32)>()) as u64,
            CachedTileMap::Raw(v) => (v.len() * 4) as u64,
        }
    }

    fn decode(&self, len: usize) -> MpasResult<Vec<i32>> {
        match self {
            CachedTileMap::Raw(v) => Ok(v.clone()),
            CachedTileMap::Runs(runs) => {
                let mut out =
                    checked_vec::<i32>(len, "geog-tile-map", "cached tile destination map")?;
                let mut at = 0usize;
                for &(cell, run) in runs {
                    let end = at + run as usize;
                    out[at..end].fill(cell);
                    at = end;
                }
                debug_assert_eq!(at, len);
                Ok(out)
            }
        }
    }
}

#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct TileMapCacheReceipt {
    pub budget_bytes: u64,
    pub stored_bytes: u64,
    pub stored_maps: usize,
    pub hits: u64,
    pub misses: u64,
    /// Maps recomputed because storing them would have passed the budget.
    pub declined_for_budget: u64,
}

#[derive(Default)]
struct TileMapCacheInner {
    maps: HashMap<TileMapKey, CachedTileMap>,
    bytes: u64,
    hits: u64,
    misses: u64,
    declined: u64,
}

/// Tile maps kept between datasets, inside a granted byte budget.
///
/// A miss is only ever a cost in time: the map is a pure function of the
/// mesh, the kd tree and the geometry key, so a recomputed map and a cached
/// one are the same bytes.  That is why running the budget out changes how
/// long a build takes and never what it produces.
struct TileMapCache {
    budget_bytes: u64,
    inner: Mutex<TileMapCacheInner>,
}

impl TileMapCache {
    fn new(budget_bytes: u64) -> Self {
        Self {
            budget_bytes,
            inner: Mutex::new(TileMapCacheInner::default()),
        }
    }

    fn lock(&self) -> MpasResult<std::sync::MutexGuard<'_, TileMapCacheInner>> {
        self.inner.lock().map_err(|_| {
            MpasError::Refusal(
                "the tile-map cache lock was poisoned by a panicking tile worker; \
                 the build cannot continue without knowing which tiles were mapped"
                    .to_string(),
            )
        })
    }

    fn get(&self, key: &TileMapKey, len: usize) -> MpasResult<Option<Vec<i32>>> {
        let mut inner = self.lock()?;
        match inner.maps.get(key) {
            Some(entry) => {
                let decoded = entry.decode(len)?;
                inner.hits += 1;
                Ok(Some(decoded))
            }
            None => {
                inner.misses += 1;
                Ok(None)
            }
        }
    }

    fn insert(&self, key: TileMapKey, map: &[i32]) -> MpasResult<()> {
        if self.budget_bytes == 0 {
            return Ok(());
        }
        let encoded = CachedTileMap::encode(map)?;
        let bytes = encoded.bytes();
        let mut inner = self.lock()?;
        if inner.maps.contains_key(&key) {
            return Ok(());
        }
        if inner.bytes + bytes > self.budget_bytes {
            inner.declined += 1;
            return Ok(());
        }
        inner.bytes += bytes;
        inner.maps.insert(key, encoded);
        Ok(())
    }

    fn receipt(&self) -> MpasResult<TileMapCacheReceipt> {
        let inner = self.lock()?;
        Ok(TileMapCacheReceipt {
            budget_bytes: self.budget_bytes,
            stored_bytes: inner.bytes,
            stored_maps: inner.maps.len(),
            hits: inner.hits,
            misses: inner.misses,
            declined_for_budget: inner.declined,
        })
    }
}

/// The mesh-side state every geography pass needs, plus the pool and cache.
struct GeogContext<'a> {
    mesh: &'a Mesh,
    tree: &'a KdTree,
    max_d2: f64,
    workers: usize,
    pool: Option<&'a rayon::ThreadPool>,
    cache: &'a TileMapCache,
}

impl GeogContext<'_> {
    /// The destination map for one tile, from the cache when it is there.
    fn tile_map(&self, ds: &GeogDataset, tile: &TileRef, factor: usize) -> MpasResult<Vec<i32>> {
        validate_supersample(factor)?;
        let len = (ds.index.tile_x as usize)
            .checked_mul(ds.index.tile_y as usize)
            .and_then(|v| v.checked_mul(factor * factor))
            .ok_or_else(|| MpasError::Refusal("tile-map element count overflow".to_string()))?;
        let key = TileMapKey {
            geometry: ds.geometry(),
            xs: tile.xs,
            ys: tile.ys,
            factor,
        };
        if let Some(hit) = self.cache.get(&key, len)? {
            return Ok(hit);
        }
        // The rows of one tile go to the admitted pool (the same threads the
        // tiles run on), so the few tiles under a regional mesh's fine core
        // no longer finish on one thread each while the rest wait.
        let map = match self.pool {
            Some(pool) if self.workers > 1 => pool.install(|| {
                tile_map(self.mesh, self.tree, self.max_d2, ds, tile, factor, true)
            })?,
            _ => tile_map(self.mesh, self.tree, self.max_d2, ds, tile, factor, false)?,
        };
        self.cache.insert(key, &map)?;
        Ok(map)
    }

    /// MPAS: mpas_init_atm_static.F:1375-1469; mpas_geotile_manager.F:390-412,505-708,789-885,900-993.
    /// Shared mapping returns on a failed cell seed; a failed neighbor stops
    /// that neighbor push only. Integer accumulation remains parallel for a
    /// complete readable archive. Incomplete archives use the native cell/stack
    /// order, including the category/count criterion after each processed tile.
    fn for_each_mapped_tile<F, C>(&self, ds: &GeogDataset, factor: usize, criteria: C, body: F) -> MpasResult<()>
    where F: Fn(&TileRef) -> MpasResult<()> + Sync + Send,
          C: Fn(usize) -> bool,
    {
        self.for_each_mapped_tile_policy(ds, factor, false, None, criteria, body)
    }

    /// MPAS: mpas_init_atm_static.F:520-610,766-852,1006-1086,1375-1469.
    /// The bespoke readers abort on failed seed or neighbor loads. The shared
    /// mapper returns from a failed seed and only ends a failed neighbor push.
    /// Vegetation pushes neighbors only for pixels reaching owned land cells;
    /// snow and albedo also push when a pixel reaches an owned water cell.
    fn for_each_mapped_tile_policy<F, C>(&self, ds: &GeogDataset, factor: usize,
        critical_load: bool, neighbor_landmask: Option<&[i32]>, criteria: C, body: F) -> MpasResult<()>
    where F: Fn(&TileRef) -> MpasResult<()> + Sync + Send,
          C: Fn(usize) -> bool,
    {
        let pixel_nx = (360.0f32 / (ds.index.dx as f32).abs()).round() as i64;
        let pixel_ny = (180.0f32 / (ds.index.dy as f32).abs()).round() as i64;
        let nx = pixel_nx / ds.index.tile_x;
        let ny = pixel_ny / ds.index.tile_y;
        let tiles: std::collections::HashMap<(i64,i64), usize> = ds.tiles.iter().enumerate()
            .map(|(i,t)| (((t.xs-1)/ds.index.tile_x,(t.ys-1)/ds.index.tile_y),i)).collect();
        let readable = |i: usize| std::fs::metadata(&ds.tiles[i].path).map_or(false, |m| m.len()>0);
        if tiles.len() as i64 == nx*ny && tiles.keys().all(|&(x,y)| x>=0 && x<nx && y>=0 && y<ny)
            && (0..ds.tiles.len()).all(readable) {
            return self.for_each_tile(&ds.tiles, body);
        }
        // Native gen_filename uses zero-based rounded global pixels, not
        // known_x/known_y. Preserve RKIND normalization and degree conversion.
        let locate = |lat: f32, mut lon: f32| {
            let pi = std::f32::consts::PI;
            if lon>pi { lon-=2.0*pi; } else if lon < -pi { lon+=2.0*pi; }
            let mut x=((lon*crate::static_native::DEG_PER_RAD-ds.index.known_lon as f32)/(ds.index.dx as f32)).round() as i64;
            let mut y=((lat*crate::static_native::DEG_PER_RAD-ds.index.known_lat as f32)/(ds.index.dy as f32)).round() as i64;
            if x<0 { x+=pixel_nx; } else if x>=pixel_nx { x-=pixel_nx; }
            if y<0 { y=0; } else if y>=pixel_ny { y=pixel_ny-1; }
            ((x-x.rem_euclid(ds.index.tile_x))/ds.index.tile_x,
             (y-y.rem_euclid(ds.index.tile_y))/ds.index.tile_y)
        };
        let mut processed=vec![false;ds.tiles.len()];
        let mut stack=Vec::new();
        for c in 0..self.mesh.n_cells {
            if criteria(c) {
                let key=locate(self.mesh.lat_cell[c] as f32,self.mesh.lon_cell[c] as f32);
                match tiles.get(&key).copied().filter(|&i| readable(i)) {
                    Some(i)=>stack.push(i),
                    None if critical_load=>return Err(MpasError::Refusal(format!(
                        "{}: native field interpolation cannot load the tile containing cell {}",
                        ds.path.display(), c+1))),
                    None=>return Ok(()),
                }
            }
            while let Some(i)=stack.pop() {
                if processed[i] {continue;}
                let tile=&ds.tiles[i];
                body(tile)?;
                processed[i]=true;
                let map=self.tile_map(ds,tile,factor)?;
                if !map.iter().any(|&c| c>=0 && neighbor_landmask.map_or(true, |m| m[c as usize]==1)) {continue;}
                let x=(tile.xs-1)/ds.index.tile_x;
                let y=(tile.ys-1)/ds.index.tile_y;
                let up=if y+1>ny-1 {((x+nx/2).rem_euclid(nx),y)} else {(x,y+1)};
                let down=if y-1<0 {((x+nx/2).rem_euclid(nx),y)} else {(x,y-1)};
                let right=if x+1>nx-1 {(0,y)} else {(x+1,y)};
                let left=if x-1<0 {(nx-1,y)} else {(x-1,y)};
                for (hx,hy) in [up,down,right,left] {
                    // hash_to_ll followed by get_tile is intentionally a
                    // round trip through RKIND, including odd tile widths.
                    let lat=((hy*ds.index.tile_y+ds.index.tile_y/2) as f32*ds.index.dy as f32+ds.index.known_lat as f32)
                        *(std::f32::consts::PI/180.0f32);
                    let lon=((hx*ds.index.tile_x+ds.index.tile_x/2) as f32*ds.index.dx as f32+ds.index.known_lon as f32)
                        *(std::f32::consts::PI/180.0f32);
                    match tiles.get(&locate(lat,lon)).copied().filter(|&j| readable(j)) {
                        Some(j)=>stack.push(j),
                        None if critical_load=>return Err(MpasError::Refusal(format!(
                            "{}: native field interpolation cannot load a neighbor of {}",
                            ds.path.display(), tile.path.display()))),
                        None=>break,
                    }
                }
            }
        }
        Ok(())
    }

    /// Run `body` over every tile, on the admitted number of workers.
    ///
    /// Failures are collected and the lowest-indexed one is returned, so the
    /// refusal a user reads names the same tile on every run rather than
    /// whichever worker happened to finish first.
    fn for_each_tile<F>(&self, tiles: &[TileRef], body: F) -> MpasResult<()>
    where
        F: Fn(&TileRef) -> MpasResult<()> + Sync + Send,
    {
        let pool = match self.pool {
            Some(pool) if self.workers > 1 && tiles.len() > 1 => pool,
            _ => {
                for tile in tiles {
                    body(tile)?;
                }
                return Ok(());
            }
        };
        let results: Vec<MpasResult<()>> =
            pool.install(|| tiles.par_iter().map(&body).collect());
        for result in results {
            result?;
        }
        Ok(())
    }
}

/// True when no pixel of a tile reaches a cell of this mesh.
///
/// THE COST THIS REMOVES. Every geography pass used to read every tile of
/// its dataset whatever the mesh covered, because the destination map is
/// built per tile and the data was read beside it. On a regional mesh most
/// tiles map entirely to nobody: measured on a 66,886-cell cut 1,735 km in
/// radius, the build read the whole 27 GB archive (52.65 million 512-byte
/// blocks) from a cold disk, 227 s of a 125 MB/s volume, for a region that
/// covers about 2 % of the globe. A tile whose map is all refusals adds
/// nothing to any sum, count or category tally, so not reading it changes no
/// value in the static; the map itself is still built, from geometry alone.
#[inline]
fn map_reaches_no_cell(map: &[i32]) -> bool {
    map.iter().all(|&cell| cell < 0)
}

/// Add a run of equal-cell contributions with one atomic operation.
///
/// Consecutive source pixels overwhelmingly land in the same destination
/// cell, so folding the run before touching the shared accumulator is what
/// keeps the parallel loop from being a queue at a handful of cache lines.
/// The fold is integer addition, which is what makes it free of consequences.
#[inline]
// MPAS: mpas_init_atm_static.F:1511,1537. I8KIND integer accumulation.
fn flush_i64(target: &[AtomicI64], index: usize, acc: i64) {
    if acc != 0 {
        target[index].fetch_add(acc, Ordering::Relaxed);
    }
}

#[inline]
// MPAS: mpas_init_atm_static.F:140,1512,1538,1772. Default INTEGER counts.
fn flush_i32(target: &[AtomicI32], index: usize, acc: i32) {
    if acc != 0 {
        target[index].fetch_add(acc, Ordering::Relaxed);
    }
}

#[inline]
fn plane_raw_at(ds: &GeogDataset, plane: &[i64], i: usize, j: usize) -> i64 {
    let b = ds.index.tile_bdr as usize;
    let nx = ds.index.full_tile_nx();
    plane[(j + b) * nx + (i + b)]
}

/// MPAS: mpas_init_atm_static.F:1736-1937,2126-2182 (category criteria,
/// accumulation, landuse/soil-category/Noah-MP texture entry points).
fn categorical(
    ctx: &GeogContext<'_>,
    path: &Path,
    factor: usize,
) -> MpasResult<(Vec<i32>, i64, i64, GeogDataset)> {
    categorical_with(ctx, path, factor, false)
}

/// [`categorical`], optionally voting a dataset's lake category as its water
/// category, pixel by pixel, before the dominant-category vote.
///
/// Native MPAS-A v8.4.2 (`config_landuse_data = 'MODIFIED_IGBP_MODIS_NOAH'`,
/// mpas_init_atm_static.F:401-403) reads `modis_landuse_20class_30s`, whose 20
/// categories hold inland water as water (17). The archive `gpuwm fetch-geog`
/// stages is `modis_landuse_20class_30s_with_lakes`: the same pixels with inland
/// water relabelled 21 (`islake=21`, `category_max=21`). A native static voted
/// from it carried category 21, which MPAS Noah's 20-row VEGPARM.TBL rejects at
/// its first call ("too many input landuse types", module_sf_noahlsm.F:2440).
/// Folding per pixel, before the vote, gives the 20-category histogram native
/// counts; folding the voted cell afterwards would not (lake plus water pixels
/// can together outvote land that neither outvotes alone). A dataset whose
/// index declares no `islake` is read unchanged.
fn categorical_with(
    ctx: &GeogContext<'_>,
    path: &Path,
    factor: usize,
    lake_as_water: bool,
) -> MpasResult<(Vec<i32>, i64, i64, GeogDataset)> {
    let mesh = ctx.mesh;
    let ds = GeogDataset::open(path)?;
    if ds.index.kind != SourceKind::Categorical || ds.index.nz() != 1 {
        return Err(MpasError::Refusal(format!(
            "{} must be a one-plane categorical WPS_GEOG dataset",
            path.display()
        )));
    }
    let cmin = ds.index.category_min.ok_or_else(|| {
        MpasError::Refusal(format!("{} lacks category_min", path.display()))
    })?;
    let mut cmax = ds.index.category_max.ok_or_else(|| {
        MpasError::Refusal(format!("{} lacks category_max", path.display()))
    })?;
    let lake_fold = match (lake_as_water, ds.index.islake, ds.index.iswater) {
        (true, Some(lake), Some(water))
            if lake != water && (cmin..=cmax).contains(&lake) && (cmin..=cmax).contains(&water) =>
        {
            Some((lake, water))
        }
        _ => None,
    };
    if let Some((lake, _)) = lake_fold {
        if lake == cmax {
            // The lake row is the archive's extra top category; native's
            // histogram has no such row.
            cmax = lake - 1;
        }
    }
    // MPAS: mpas_init_atm_static.F:1835-1845. A descending bound
    // allocates a zero-size histogram; MAXLOC then returns zero.
    let ncat = (cmax - cmin + 1).max(0) as usize;
    let counts = checked_vec_with(
        checked_len(mesh.n_cells, ncat, "categorical", "category counts")?,
        "categorical",
        "category counts",
        || AtomicI32::new(0),
    )?;

    ctx.for_each_mapped_tile(&ds, factor, |c| counts[c*ncat..(c+1)*ncat].iter().all(|v| v.load(Ordering::Relaxed)==0), |tile| {
        let map = ctx.tile_map(&ds, tile, factor)?;
        if map_reaches_no_cell(&map) {
            return Ok(());
        }
        let plane = ds.read_plane(tile, 0)?;
        let sub = factor * factor;
        // One atomic per (cell, category) run rather than per subpixel.
        let mut run_slot = usize::MAX;
        let mut run_len = 0i32;
        for j in 0..ds.index.tile_y as usize {
            for i in 0..ds.index.tile_x as usize {
                let mut raw = plane_raw_at(&ds, &plane, i, j);
                if let Some((lake, water)) = lake_fold {
                    if raw == lake {
                        raw = water;
                    }
                }
                if raw < cmin || raw > cmax {
                    continue;
                }
                let cat = (raw - cmin) as usize;
                let p = j * ds.index.tile_x as usize + i;
                for q in 0..sub {
                    let c = map[p * sub + q];
                    if c < 0 {
                        continue;
                    }
                    let slot = c as usize * ncat + cat;
                    if slot != run_slot {
                        if run_len != 0 {
                            counts[run_slot].fetch_add(run_len, Ordering::Relaxed);
                        }
                        run_slot = slot;
                        run_len = 0;
                    }
                    run_len = run_len.wrapping_add(1);
                }
            }
        }
        if run_len != 0 {
            counts[run_slot].fetch_add(run_len, Ordering::Relaxed);
        }
        Ok(())
    })?;

    // A cell no pixel reached votes with the pixel under its centre (see
    // `centre_pixels`), folded and range-checked like any pixel.
    let empty: Vec<usize> = (0..mesh.n_cells)
        .filter(|&c| counts[c * ncat..(c + 1) * ncat].iter().all(|v| v.load(Ordering::Relaxed) == 0))
        .collect();
    let mut seeded = 0usize;
    for (k, hits) in centre_pixels(&ds, mesh, &empty) {
        let plane = ds.read_plane(&ds.tiles[k], 0)?;
        for &(c, i, j) in &hits {
            let mut raw = plane_raw_at(&ds, &plane, i, j);
            if let Some((lake, water)) = lake_fold {
                if raw == lake {
                    raw = water;
                }
            }
            if raw < cmin || raw > cmax {
                continue;
            }
            counts[c * ncat + (raw - cmin) as usize].store(1, Ordering::Relaxed);
            seeded += 1;
        }
    }
    CENTRE_SEEDED.with(|c| c.set(seeded));
    let mut out =
        checked_vec_filled::<i32>(mesh.n_cells, cmin as i32, "categorical", "category field")?;
    for c in 0..mesh.n_cells {
        let row = &counts[c * ncat..(c + 1) * ncat];
        let row: Vec<i32> = row.iter().map(|v| v.load(Ordering::Relaxed)).collect();
        let best = crate::static_native::dominant_category(&row);
        // MPAS: mpas_init_atm_static.F:1842,1918,2163: MAXLOC returns
        // the first category even when every count is zero.
        out[c] = best.map_or(cmin-1,|best| cmin+best as i64) as i32;
    }
    Ok((out, cmin, cmax, ds))
}

thread_local! {
    /// How many cells the last continuous or categorical pass seeded from the
    /// pixel under their centre because no tile pixel reached them.  Read and
    /// cleared by the receipt on the building thread.
    static CENTRE_SEEDED: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

fn take_centre_seeded() -> usize {
    CENTRE_SEEDED.with(|c| c.replace(0))
}

/// The source pixel under each listed cell's centre, grouped by the tile that
/// holds it: `tile index -> [(cell, i, j)]`, `(i, j)` interior to the tile.
///
/// THE DEFECT THIS CLOSES. A destination cell gathers the source pixels whose
/// centres land nearest to it.  A cell finer than the source -- a 1 km cell
/// under 30-arc-second pixels, about 0.9 km -- can be reached by no pixel
/// centre at all; native's own assignment (mpas_init_atm_static.F:1610,
/// `ter = ter_integer / nhs`) is then 0/0, and its category vote (1842,
/// MAXLOC over all-zero counts) is category 1.  rw_mpas_static built from
/// integrate/2.9.2 wrote NaN terrain for 4,869 of 353,100 cells of a 1 km
/// moving case, every one in the 1 km core over open ocean, and the land-use
/// vote called those same ocean cells category 1 land.  Such a cell takes the
/// one sample a point lookup gives: the pixel under its own centre.  A cell
/// any pixel reached is not touched, so a static whose cells all had samples
/// keeps every byte.  A tile that is absent or shorter than it declares gives
/// nothing, which leaves native's incomplete-archive result as it was.
fn centre_pixels(ds: &GeogDataset, mesh: &Mesh, cells: &[usize]) -> BTreeMap<usize, Vec<(usize, usize, usize)>> {
    let mut by_tile: BTreeMap<usize, Vec<(usize, usize, usize)>> = BTreeMap::new();
    if cells.is_empty() || ds.tiles.is_empty() { return by_tile; }
    let (tx, ty) = (ds.index.tile_x, ds.index.tile_y);
    let tiles: HashMap<(i64, i64), usize> = ds.tiles.iter().enumerate()
        .map(|(k, t)| (((t.xs - 1).div_euclid(tx), (t.ys - 1).div_euclid(ty)), k)).collect();
    let declared = ds.plane_bytes() * ds.index.nz() as u64;
    let mut complete: HashMap<usize, bool> = HashMap::new();
    for &c in cells {
        // Source pixel k is centred at source coordinate k (known_x names pixel 1).
        let (x, y) = ds.latlon_rad_to_source_xy(mesh.lat_cell[c], mesh.lon_cell[c]);
        if !(x.is_finite() && y.is_finite()) { continue; }
        let mut ix = x.round() as i64;
        if ds.wraps_x && ds.nx_global > 0 { ix = (ix - 1).rem_euclid(ds.nx_global) + 1; }
        let ix = ix.clamp(1, ds.nx_global.max(1));
        let iy = (y.round() as i64).clamp(1, ds.ny_global.max(1));
        let Some(&k) = tiles.get(&((ix - 1).div_euclid(tx), (iy - 1).div_euclid(ty))) else { continue };
        let t = &ds.tiles[k];
        if ix < t.xs || ix > t.xe || iy < t.ys || iy > t.ye { continue; }
        let whole = *complete.entry(k).or_insert_with(|| {
            std::fs::metadata(&t.path).map_or(false, |m| m.len() >= declared)
        });
        if !whole { continue; }
        by_tile.entry(k).or_default().push((c, (ix - t.xs) as usize, (iy - t.ys) as usize));
    }
    by_tile
}

/// One source pixel under a mean policy's missing-value rule: the word to
/// accumulate, or nothing when the policy skips the pixel.
/// MPAS: mpas_init_atm_static.F:564,585 converts snow's missing value to
/// I8KIND; monthly fields compare RKIND directly at 809,828,1043,1061.
fn mean_sample(ds: &GeogDataset, policy: MeanPolicy, raw: i64) -> Option<i64> {
    let missing = match policy {
        MeanPolicy::Green | MeanPolicy::Albedo => ds.index.missing_value_rkind == Some(raw as f32),
        _ => ds.index.missing_value == Some(raw),
    };
    if !missing { return Some(raw); }
    match policy {
        MeanPolicy::Snow => None,
        MeanPolicy::Green => Some(0),
        MeanPolicy::Albedo => Some(8),
        MeanPolicy::Terrain => Some(raw),
    }
}

/// MPAS: mpas_init_atm_static.F:1506-1514,736-879,972-1113,488-640.
#[derive(Clone, Copy)]
enum MeanPolicy { Terrain, Green, Albedo, Snow }

/// Integer accumulation followed by the RKIND assignment in each native driver.
/// MPAS: mpas_init_atm_static.F:1559-1620,736-879,972-1113,488-640.
fn continuous_mean(
    ctx: &GeogContext<'_>, path: &Path, factor: usize,
    landmask: Option<&[i32]>, policy: MeanPolicy,
) -> MpasResult<(Vec<f32>, GeogDataset)> {
    let mesh = ctx.mesh;
    let ds = GeogDataset::open(path)?;
    if ds.index.kind != SourceKind::Continuous {
        return Err(MpasError::Refusal(format!("{} must be continuous", path.display())));
    }
    // MPAS: mpas_init_atm_static.F:522-527,770-776,1008-1014.
    // These bespoke consumers make a failed get_tile critical; the shared
    // terrain mapper instead returns to its caller with untouched counts.
    if ds.tiles.is_empty() && !matches!(policy, MeanPolicy::Terrain) {
        return Err(MpasError::Refusal(format!("{}: native field interpolation cannot load its first tile", path.display())));
    }
    let nz = ds.index.nz();
    let slots = checked_len(mesh.n_cells, nz, "continuous", "plane accumulator")?;
    let sums = checked_vec_with(slots, "continuous", "plane accumulator", || AtomicI64::new(0))?;
    // MPAS: mpas_init_atm_static.F:140,566,587,816,835,1050,1068,1512.
    let count = checked_vec_with(slots, "continuous", "sample counts", || AtomicI32::new(0))?;
    let body = |tile: &TileRef| {
        let map = ctx.tile_map(&ds, tile, factor)?;
        if map_reaches_no_cell(&map) { return Ok(()); }
        let sub = factor * factor;
        for z in 0..nz {
            let plane = ds.read_plane(tile, z)?;
            let mut run_slot = usize::MAX;
            let mut acc = 0i64;
            let mut n = 0i32;
            for j in 0..ds.index.tile_y as usize {
                for i in 0..ds.index.tile_x as usize {
                    let Some(raw) = mean_sample(&ds, policy, plane_raw_at(&ds, &plane, i, j)) else { continue };
                    let pixel = j * ds.index.tile_x as usize + i;
                    for q in 0..sub {
                        let c = map[pixel * sub + q];
                        if c < 0 || landmask.map_or(false, |m| m[c as usize] == 0) { continue; }
                        let slot = c as usize * nz + z;
                        if slot != run_slot {
                            if run_slot != usize::MAX {
                                flush_i64(&sums, run_slot, acc);
                                flush_i32(&count, run_slot, n);
                            }
                            run_slot = slot; acc = 0; n = 0;
                        }
                        acc = acc.wrapping_add(raw); n = n.wrapping_add(1);
                    }
                }
            }
            if run_slot != usize::MAX {
                flush_i64(&sums, run_slot, acc);
                flush_i32(&count, run_slot, n);
            }
        }
        Ok(())
    };
    if matches!(policy, MeanPolicy::Terrain) {
        ctx.for_each_mapped_tile(&ds, factor, |c| count[c*nz].load(Ordering::Relaxed)==0, body)?;
    } else {
        // MPAS: mpas_init_atm_static.F:522-610,766-852,1006-1086.
        // Incomplete archives still follow the native cell/stack load order;
        // the failure must occur even when other readable tiles add samples.
        let neighbor_landmask=if matches!(policy,MeanPolicy::Green) {landmask} else {None};
        ctx.for_each_mapped_tile_policy(&ds, factor, true, neighbor_landmask,
            |c| count[c*nz].load(Ordering::Relaxed)==0, body)?;
    }
    // A cell no pixel reached takes the pixel under its centre (see
    // `centre_pixels`), under the policy's missing-value rule like any pixel.
    let empty: Vec<usize> = (0..mesh.n_cells)
        .filter(|&c| !landmask.map_or(false, |m| m[c] == 0)
            && (0..nz).all(|z| count[c * nz + z].load(Ordering::Relaxed) == 0))
        .collect();
    let mut seeded = 0usize;
    for (k, hits) in centre_pixels(&ds, mesh, &empty) {
        let tile = &ds.tiles[k];
        for z in 0..nz {
            let plane = ds.read_plane(tile, z)?;
            for &(c, i, j) in &hits {
                let Some(raw) = mean_sample(&ds, policy, plane_raw_at(&ds, &plane, i, j)) else { continue };
                let slot = c * nz + z;
                sums[slot].store(raw, Ordering::Relaxed);
                count[slot].store(1, Ordering::Relaxed);
                if z == 0 { seeded += 1; }
            }
        }
    }
    CENTRE_SEEDED.with(|c| c.set(seeded));
    let mut out = checked_vec::<f32>(slots, "continuous", "destination field")?;
    for c in 0..mesh.n_cells {
        if landmask.map_or(false, |m| m[c] == 0) {
            // MPAS: mpas_init_atm_static.F:1092-1098: no accumulated
            // albedo samples (including water) use unscaled fillVal=8.
            if matches!(policy,MeanPolicy::Albedo) {out[c*nz..(c+1)*nz].fill(8.0);}
            continue;
        }
        for z in 0..nz {
            let slot = c * nz + z;
            let n = count[slot].load(Ordering::Relaxed);
            let sum = sums[slot].load(Ordering::Relaxed);
            out[slot] = if n == 0 && !matches!(policy, MeanPolicy::Terrain) {
                if matches!(policy, MeanPolicy::Albedo) { 8.0 } else { 0.0 }
            } else {
                let scale = if matches!(policy, MeanPolicy::Green) { 1.0 } else { ds.index.scale_factor as f32 };
                crate::static_native::scaled_mean(sum, n, scale)
            };
        }
    }
    Ok((out, ds))
}

/// MPAS: mpas_init_atm_static.F:1529-1541,1642-1718 (soilcomp criteria/accumulation).
fn soilcomp(
    ctx: &GeogContext<'_>,
    path: &Path,
    factor: usize,
    landmask: &[i32],
) -> MpasResult<(Vec<f32>, GeogDataset)> {
    let mesh = ctx.mesh;
    let ds = GeogDataset::open(path)?;
    let nz = ds.index.nz();
    if nz == 0 {
        return Err(MpasError::Refusal(format!("{} has no soil components", path.display())));
    }
    let slots = checked_len(mesh.n_cells, nz, "soilcomp", "component accumulator")?;
    let sums = checked_vec_with(slots, "soilcomp", "component accumulator", || AtomicI64::new(0))?;
    let count =
        checked_vec_with(mesh.n_cells, "soilcomp", "sample counts", || AtomicI32::new(0))?;

    ctx.for_each_mapped_tile(&ds, factor, |c| count[c].load(Ordering::Relaxed)==0, |tile| {
        let map = ctx.tile_map(&ds, tile, factor)?;
        if map_reaches_no_cell(&map) {
            return Ok(());
        }
        let mut planes = Vec::with_capacity(nz);
        for z in 0..nz {
            planes.push(ds.read_plane(tile, z)?);
        }
        let sub = factor * factor;
        // A whole soil column is folded per destination-cell run, so one run
        // costs nz+1 atomics instead of (nz+1) per subpixel.
        let mut run_cell = usize::MAX;
        let mut acc = vec![0i64; nz];
        let mut acc_count = 0i32;
        for j in 0..ds.index.tile_y as usize {
            for i in 0..ds.index.tile_x as usize {
                let first = plane_raw_at(&ds, &planes[0], i, j);
                // MPAS: mpas_init_atm_static.F:93,1536,1679. The integer
                // source pixel is promoted to RKIND for this comparison.
                if ds.index.missing_value_rkind == Some(first as f32) {
                    continue;
                }
                let p = j * ds.index.tile_x as usize + i;
                for q in 0..sub {
                    let c = map[p * sub + q];
                    if c < 0 || landmask[c as usize] == 0 {
                        continue;
                    }
                    let c = c as usize;
                    if c != run_cell {
                        if run_cell != usize::MAX {
                            for (z, &value) in acc.iter().enumerate() {
                                flush_i64(&sums, run_cell * nz + z, value);
                            }
                            flush_i32(&count, run_cell, acc_count);
                        }
                        run_cell = c;
                        acc.iter_mut().for_each(|v| *v = 0);
                        acc_count = 0;
                    }
                    for (slot, plane) in acc.iter_mut().zip(&planes) {
                        *slot = slot.wrapping_add(plane_raw_at(&ds, plane, i, j));
                    }
                    acc_count = acc_count.wrapping_add(1);
                }
            }
        }
        if run_cell != usize::MAX {
            for (z, &value) in acc.iter().enumerate() {
                flush_i64(&sums, run_cell * nz + z, value);
            }
            flush_i32(&count, run_cell, acc_count);
        }
        Ok(())
    })?;

    let sums: Vec<i64> = sums.iter().map(|v| v.load(Ordering::Relaxed)).collect();
    let count: Vec<i32> = count.iter().map(|v| v.load(Ordering::Relaxed)).collect();
    let mut out = checked_vec::<f32>(slots, "soilcomp", "destination field")?;
    for c in 0..mesh.n_cells {
        for z in 0..nz {
            out[c * nz + z] =
                crate::static_native::soilcomp_mean(sums[c * nz + z], count[c], ds.index.scale_factor as f32);
        }
    }
    Ok((out, ds))
}

/// MPAS: mpas_init_atm_static.F:1990-2103: tiled mosaic includes the
/// original three-pixel halo, followed by MPAS's interpolation chain.
fn climatology_plane(ds: &GeogDataset, z: usize) -> MpasResult<crate::static_native::Plane> {
    // MPAS: mpas_init_atm_static.F:2048-2062,2185-2193.
    if ds.tiles.is_empty() {
        return Err(MpasError::Refusal(format!("{}: native climatology read is critical when its tile is missing", ds.path.display())));
    }
    let border = ds.index.tile_bdr;
    let nx = (ds.nx_global + 2 * border) as usize;
    let ny = (ds.ny_global + 2 * border) as usize;
    let mut plane = crate::static_native::Plane {
        x0: 1-border, y0: 1-border, nx, ny,
        values: checked_vec::<f32>(checked_len(nx,ny,"climatology","halo plane")?, "climatology", "halo plane")?,
    };
    for tile in &ds.tiles {
        let raw = ds.read_plane(tile,z)?;
        for j in 0..ds.index.full_tile_ny() {
            for i in 0..ds.index.full_tile_nx() {
                let x=tile.xs-border+i as i64;
                let y=tile.ys-border+j as i64;
                // At an internal seam copy each tile's owned interior;
                // only global perimeter tiles supply the outer halo.
                if (x < tile.xs && tile.xs != 1) || (x > tile.xe && tile.xe != ds.nx_global)
                    || (y < tile.ys && tile.ys != 1) || (y > tile.ye && tile.ye != ds.ny_global) { continue; }
                if x < plane.x0 || y < plane.y0 || x >= plane.x0+nx as i64 || y >= plane.y0+ny as i64 { continue; }
                plane.values[(y-plane.y0) as usize*nx+(x-plane.x0) as usize] =
                    raw[j*ds.index.full_tile_nx()+i] as f32 * ds.index.scale_factor as f32;
            }
        }
    }
    Ok(plane)
}

/// MPAS: mpas_init_atm_static.F:2075-2097; mpas_init_atm_llxy.F:1389-1416.
fn climatology_xy(ds: &GeogDataset, lat: f32, lon: f32) -> (f32,f32) {
    let mut lon = lon * crate::static_native::DEG_PER_RAD;
    let lat = lat * crate::static_native::DEG_PER_RAD;
    let mut x = (lon-ds.index.known_lon as f32)/ds.index.dx as f32 + ds.index.known_x as f32;
    let y = (lat-ds.index.known_lat as f32)/ds.index.dy as f32 + ds.index.known_y as f32;
    if x < 0.5 {
        lon += 360.0;
        x = (lon-ds.index.known_lon as f32)/ds.index.dx as f32 + ds.index.known_x as f32;
    } else if x >= ds.nx_global as f32+0.5 {
        lon -= 360.0;
        x = (lon-ds.index.known_lon as f32)/ds.index.dx as f32 + ds.index.known_x as f32;
    }
    (x, y.clamp(1.0, (ds.ny_global-1) as f32))
}

/// MPAS: mpas_init_atm_static.F:1990-2103 (interp_soiltemp).
fn soil_temperature(mesh: &Mesh, path: &Path, landmask: &[i32]) -> MpasResult<(Vec<f32>, GeogDataset)> {
    let ds = GeogDataset::open(path)?;
    soil_temperature_dataset(mesh,ds,landmask)
}

/// MPAS: mpas_init_atm_static.F:2032-2103.
fn soil_temperature_dataset(mesh: &Mesh, ds: GeogDataset, landmask: &[i32]) -> MpasResult<(Vec<f32>,GeogDataset)> {
    if ds.index.nz() != 1 {
        return Err(MpasError::Refusal(format!("{} soil temperature must have one plane", ds.path.display())));
    }
    let plane = climatology_plane(&ds,0)?;
    let mut out = checked_vec::<f32>(mesh.n_cells, "soiltemp", "destination field")?;
    for c in 0..mesh.n_cells {
        if landmask[c] == 0 { continue; }
        let (x,y) = climatology_xy(&ds,mesh.lat_cell[c] as f32,mesh.lon_cell[c] as f32);
        out[c] = plane.interpolate(x,y,0.0);
    }
    Ok((out, ds))
}

/// MPAS: mpas_init_atm_static.F:640-722,879-959,1113-1198.
/// The driver initializes interp_list only in the NCEP snow branch (687-691).
/// With MODIS snow its NCEP monthly calls (943,1177) use an unset method list;
/// mpas_init_atm_hinterp.F:151-231 leaves the result unset for unknown methods.
/// The pinned GNU strict binary returns the projected y before latitude clamps
/// in that path: the method dispatcher does not change the SSE return register.
/// Whole-driver repeats/ranks/host witnesses are recorded in the lane receipts.
fn direct_climatology(mesh: &Mesh, path: &Path, mask: &[i32], missing: f32,
                       ocean: f32, divisor: f32, sequence_initialized: bool) -> MpasResult<(Vec<f32>,GeogDataset)> {
    let ds=GeogDataset::open_native_climatology(path,divisor==1.0)?;
    let nz=ds.index.nz();
    let mut out=checked_vec::<f32>(checked_len(mesh.n_cells,nz,"NCEP","months")?,"NCEP","field")?;
    for z in 0..nz {
        let plane=climatology_plane(&ds,z)?;
        for c in 0..mesh.n_cells {
            let value=if mask[c]==0 { ocean } else {
                let (x,y)=climatology_xy(&ds,mesh.lat_cell[c] as f32,mesh.lon_cell[c] as f32);
                if sequence_initialized { plane.interpolate(x,y,missing) } else {
                    let lat=mesh.lat_cell[c] as f32 * crate::static_native::DEG_PER_RAD;
                    (lat-ds.index.known_lat as f32)/ds.index.dy as f32 + ds.index.known_y as f32
                }
            };
            out[c*nz+z]=value/divisor;
        }
    }
    Ok((out,ds))
}

/// MPAS: mpas_init_atm_static.F:640-722,879-959,1113-1198.
fn open_stage_dataset(cfg: &StaticBuildConfig, field: &str, path: &Path) -> MpasResult<GeogDataset> {
    match field {
        "greenfrac" if cfg.stages.vegetation_ncep => GeogDataset::open_native_climatology(path,true),
        "albedo12m" if cfg.stages.albedo_ncep => GeogDataset::open_native_climatology(path,true),
        "snoalb" if cfg.stages.snow_ncep => GeogDataset::open_native_climatology(path,false),
        "soiltemp" if cfg.stages.native_init => GeogDataset::open_native_soil_temperature(path),
        _ => GeogDataset::open(path),
    }
}

/// MPAS: mpas_init_atm_static.F:488-640 (MODIS snow driver).
fn snow_albedo(
    ctx: &GeogContext<'_>,
    path: &Path,
    factor: usize,
    landmask: &[i32],
) -> MpasResult<(Vec<f32>, GeogDataset)> {
    let (mut v, ds) = continuous_mean(ctx, path, factor, Some(landmask), MeanPolicy::Snow)?;
    if ds.index.nz() != 1 {
        return Err(MpasError::Refusal(format!(
            "{} snow albedo must have one plane",
            path.display()
        )));
    }
    // Archive percent -> model fraction. See [`SNOALB_MODEL_SCALE`].
    for x in &mut v {
        *x *= SNOALB_MODEL_SCALE;
    }
    Ok((v, ds))
}

/// Bytes the tile-map cache would like, capped so the ask stays sane.
///
/// The cache pays for itself on the 30-arcsec products, which share tile
/// geometry: without a cap the ask on a global mesh is the raw size of every
/// unique map at once, tens of gigabytes, most of which the run-length
/// encoding never needs.  The cap keeps the *predicted peak* -- and therefore
/// the admission gate -- from being dominated by a term the cache will not
/// actually use, while still leaving room for every map a global build needs.
///
/// MEASURED 2026-08-26 (stale-guard audit 2026-08-25, tile-cache
/// adjudication), full-geography global builds, receipts beside the packs:
///
/// * 12,002 cells: cache stored 93.5 MiB encoded (1,962 maps, 0 declined);
///   predicted peak 8.42 GiB, OS peak working set 2.73 GB.
/// * 654,432 cells: cache stored 608 MiB encoded (1,962 maps, 0 declined);
///   predicted peak 8.94 GiB, OS peak working set 4.48 GB.
///
/// So the cap is LIVE as the cache's budget -- encoded use grows roughly
/// with the square root of cell count (~2.4 GiB extrapolated at 10 M
/// cells), and 4 GiB holds headroom for every global build in reach while
/// `declined_for_budget` in the receipt makes an overflow visible.  What
/// the measurements also show, recorded as a named follow-up rather than
/// fixed by guess: charging the FULL cap into `predicted_peak_bytes`
/// over-prices a global build by ~2-3x against the measured peak, because
/// the raw-sum ask saturates the cap while the encoded reality is 2-14% of
/// it.  A measured-scaling prediction term needs a third point in the
/// multi-million-cell class before it can be fit.
const TILE_MAP_CACHE_CAP_BYTES: u64 = 4 * 1024 * 1024 * 1024;

/// One dataset's scratch and its share of the reusable-map ask.
struct DatasetFootprint {
    /// What one worker holds while it is inside this dataset's tile loop.
    tile_bytes: u64,
    /// Raw bytes of this dataset's tile maps, before dedup and encoding.
    map_bytes: u64,
    geometry: SourceGeometry,
    factor: usize,
}

fn dataset_footprint(path: &Path, factor: usize, planes_held: usize) -> MpasResult<DatasetFootprint> {
    validate_supersample(factor)?;
    let ds = GeogDataset::open(path)?;
    let sub = (factor * factor) as u64;
    // encoded bytes + the decoded i64 plane(s) the pass holds at once
    let encoded = ds.plane_bytes();
    let decoded = ds.index.plane_words() as u64 * 8 * planes_held.max(1) as u64;
    let one_map = ds.index.tile_x as u64 * ds.index.tile_y as u64 * sub * 4;
    Ok(DatasetFootprint {
        // The map itself, plus the transient its cache encoding costs while
        // it is being built.
        tile_bytes: encoded + decoded + 2 * one_map,
        map_bytes: one_map * ds.tiles.len() as u64,
        geometry: ds.geometry(),
        factor,
    })
}

/// The resident, per-worker and cache terms the admission gate plans against.
fn build_footprint(mesh: &Mesh, cfg: &StaticBuildConfig) -> MpasResult<(u64, u64, u64, u64)> {
    let resident = ((mesh.n_cells * (12 + 8 + 4) * 8)
        + (mesh.n_cells * mesh.max_edges * 4 * 4)
        + (mesh.n_edges * 2 * FIFTEEN * 4)
        + (mesh.n_edges * 32)
        + (mesh.n_vertices * 32)) as u64;

    let g = &cfg.geog;
    // `planes_held`: every pass streams one plane at a time except soilcomp,
    // which needs the whole soil column of a tile resident together.
    let soilcomp_planes = if cfg.stages.noahmp { GeogDataset::open(&g.soilcomp)?.index.nz() } else { 8 };
    // MPAS uses exactly the configured sampling factor. Plan the same maps
    // the passes build, including their supersampling square.
    validate_supersample(cfg.supersample_landuse)?;
    validate_supersample(cfg.supersample_30s)?;
    let specs: [(&PathBuf, usize, usize); 11] = [
        (&g.terrain, cfg.supersample_30s, 1),
        (&g.landuse, cfg.supersample_landuse, 1),
        (&g.soilcat, cfg.supersample_30s, 1),
        (&g.greenfrac, cfg.supersample_30s, 1),
        (&g.albedo, cfg.supersample, 1),
        (&g.snow_albedo, cfg.supersample, 1),
        (&g.soilcomp, cfg.supersample_30s, soilcomp_planes),
        (&g.soilcl1, cfg.supersample_30s, 1),
        (&g.soilcl2, cfg.supersample_30s, 1),
        (&g.soilcl3, cfg.supersample_30s, 1),
        (&g.soilcl4, cfg.supersample_30s, 1),
    ];

    let mut per_worker = 0u64;
    let mut climatology_workspace = 0u64;
    let mut unique: BTreeMap<(u64, usize), u64> = BTreeMap::new();
    for (path, factor, planes) in specs {
        if !cfg.stages.noahmp && [ &g.soilcomp, &g.soilcl1, &g.soilcl2, &g.soilcl3, &g.soilcl4 ].contains(&path) { continue; }
        let direct = (path==&g.greenfrac && cfg.stages.vegetation_ncep)
            || (path==&g.albedo && cfg.stages.albedo_ncep)
            || (path==&g.snow_albedo && cfg.stages.snow_ncep);
        if direct {
            let ds=GeogDataset::open_native_climatology(path,path!=&g.snow_albedo)?;
            let mosaic=(ds.nx_global+6) as u64*(ds.ny_global+6) as u64*4;
            climatology_workspace=climatology_workspace.max(mosaic+ds.plane_bytes()+ds.index.plane_words() as u64*8);
            continue;
        }
        let f = dataset_footprint(path, factor, planes)?;
        per_worker = per_worker.max(f.tile_bytes);
        // Datasets sharing a geometry key share their maps, so their raw map
        // bytes are counted once, not once per dataset.
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        std::hash::Hash::hash(&f.geometry, &mut hasher);
        let geometry_hash = std::hash::Hasher::finish(&hasher);
        let slot = unique.entry((geometry_hash, f.factor)).or_insert(0);
        *slot = (*slot).max(f.map_bytes);
    }
    let cache_request = unique
        .values()
        .fold(0u64, |a, b| a.saturating_add(*b))
        .min(TILE_MAP_CACHE_CAP_BYTES);

    // NetCDF writer temporarily encodes the largest slab.  `deriv_two` is
    // normally the largest field generated here.
    let writer_slab = ((mesh.n_edges * 2 * FIFTEEN * 4) as u64).max(climatology_workspace);
    Ok((resident, per_worker, writer_slab, cache_request))
}

// Output helpers ------------------------------------------------------------


fn f32_from_f64(v: &[f64]) -> Vec<f32> {
    v.iter().map(|&x| x as f32).collect()
}



/// Every variable this writer declares, in declaration order.
///
/// Spelled out so the schema gate can read it without building a file. The
/// `declared_variables_match_the_writer` test below holds it to what
/// [`write_static`] actually emits, so it cannot drift into a comfortable
/// fiction; `staticfile::schema::gate_bounded_writer_matches_pin` holds it to
/// the schema the mesh registry pins.
///
/// THE ORDER IS THE PINNED ORDER, then the extras. The first 69 names are
/// [`crate::staticfile::schema::PINNED_STATIC_VARIABLES`] in exactly the
/// sequence the published statics carry them, so a reader that walks the file
/// linearly meets the same fields in the same places it always has. The 13
/// that follow are the ones this builder adds: the four deformation/gradient
/// operator tables, the Noah-MP soil-composition group, and the land-use
/// aliases and category scalars a surface scheme reads by name.
pub fn declared_variables() -> &'static [&'static str] {
    &[
        // -- the pinned 69, in published order --------------------------
        "xtime",
        "latCell",
        "lonCell",
        "xCell",
        "yCell",
        "zCell",
        "indexToCellID",
        "latEdge",
        "lonEdge",
        "xEdge",
        "yEdge",
        "zEdge",
        "indexToEdgeID",
        "latVertex",
        "lonVertex",
        "xVertex",
        "yVertex",
        "zVertex",
        "indexToVertexID",
        "cellsOnEdge",
        "nEdgesOnCell",
        "nEdgesOnEdge",
        "edgesOnCell",
        "edgesOnEdge",
        "weightsOnEdge",
        "dvEdge",
        "dcEdge",
        "angleEdge",
        "areaCell",
        "areaTriangle",
        "cellsOnCell",
        "verticesOnCell",
        "verticesOnEdge",
        "edgesOnVertex",
        "cellsOnVertex",
        "kiteAreasOnVertex",
        "meshDensity",
        "nominalMinDc",
        "bdyMaskCell",
        "bdyMaskEdge",
        "bdyMaskVertex",
        "edgeNormalVectors",
        "localVerticalUnitVectors",
        "cellTangentPlane",
        "coeffs_reconstruct",
        "deriv_two",
        "fEdge",
        "fVertex",
        "ter",
        "landmask",
        "mminlu",
        "ivgtyp",
        "isltyp",
        "snoalb",
        "soiltemp",
        "greenfrac",
        "shdmin",
        "shdmax",
        "albedo12m",
        "var2d",
        "con",
        "oa1",
        "oa2",
        "oa3",
        "oa4",
        "ol1",
        "ol2",
        "ol3",
        "ol4",
        // -- the 13 this builder adds -----------------------------------
        "cell_gradient_coef_x",
        "cell_gradient_coef_y",
        "defc_a",
        "defc_b",
        "lu_index",
        "soilcat_top",
        "soilcomp",
        "soilcl1",
        "soilcl2",
        "soilcl3",
        "soilcl4",
        "isice_lu",
        "iswater_lu",
    ]
}

/// The geography group, as the streaming passes produced it.
///
/// Grouped rather than passed as nineteen arguments because the writer's job
/// is to lay this out, not to remember which of nineteen slices was which.
struct GeographyFields {
    tc_lakemask: Option<Vec<i32>>,
    ter: Vec<f32>,
    landmask: Vec<i32>,
    ivgtyp: Vec<i32>,
    isltyp: Vec<i32>,
    /// `[cell][month]`, in PERCENT.
    greenfrac: Vec<f32>,
    /// `[cell][month]`, in percent, as the archive delivers it.
    albedo12m: Vec<f32>,
    /// A FRACTION, not a percentage. See [`SNOALB_MODEL_SCALE`].
    snoalb: Vec<f32>,
    soiltemp: Vec<f32>,
    soilcomp: Vec<f32>,
    soilcl: [Vec<i32>; 4],
    isice_lu: i32,
    iswater_lu: i32,
    mminlu: String,
}

/// Length of the `StrLen` dimension the published statics carry.
const STR_LEN: usize = 64;
/// Months in the seasonal fields.
const N_MONTHS: usize = 12;

/// Write the unified static.
///
/// The layout is the PUBLISHED one: the same dimensions, the same variable
/// order and the same attributes the mesh registry's pinned pair carries,
/// followed by the thirteen fields this builder adds. Nothing here is a
/// convenient re-spelling -- a consumer reads the file by name and refuses on
/// a shape it did not expect, and the registry pins it by byte count.
fn write_static(
    cfg: &StaticBuildConfig,
    mesh: &Mesh,
    operators: &OperatorFields,
    geo: &GeographyFields,
    gwd: &crate::static_gwd::GwdFields,
    nominal_dx_m: f64,
    provenance_json: &str,
) -> MpasResult<()> {
    if geo.greenfrac.len() != mesh.n_cells * N_MONTHS
        || geo.albedo12m.len() != mesh.n_cells * N_MONTHS
    {
        return Err(MpasError::Refusal(
            "monthly static fields must have exactly 12 planes".to_string(),
        ));
    }
    let nsoil = if mesh.n_cells == 0 { 0 } else { geo.soilcomp.len() / mesh.n_cells };
    if nsoil == 0 || geo.soilcomp.len() != mesh.n_cells * nsoil {
        return Err(MpasError::Refusal("soilcomp shape is invalid".to_string()));
    }
    for (name, got, want) in [
        ("var2d", gwd.var2d.len(), mesh.n_cells),
        ("con", gwd.con.len(), mesh.n_cells),
    ] {
        if got != want {
            return Err(MpasError::Refusal(format!(
                "the drag band's {name} has {got} values, expected {want}"
            )));
        }
    }

    let (nc, ne, nv) = (mesh.n_cells, mesh.n_edges, mesh.n_vertices);
    let (me, me2, vd) = (mesh.max_edges, mesh.max_edges2, mesh.vertex_degree);

    let mut dims = vec![
        NcDim::fixed("StrLen", STR_LEN),
        NcDim::record("Time"),
        NcDim::fixed("nCells", nc),
        NcDim::fixed("nEdges", ne),
        NcDim::fixed("nVertices", nv),
        NcDim::fixed("TWO", 2),
        NcDim::fixed("maxEdges", me),
        NcDim::fixed("maxEdges2", me2),
        NcDim::fixed("vertexDegree", vd),
        NcDim::fixed("R3", 3),
        NcDim::fixed("nMonths", N_MONTHS),
        NcDim::fixed("FIFTEEN", FIFTEEN),
        NcDim::fixed("nSoilComps", nsoil),
    ];
    let d = |name: &str| -> usize {
        dims.iter()
            .position(|x| x.name == name)
            .expect("dimension declared above")
    };
    let (d_str, d_time, d_c, d_e, d_v) =
        (d("StrLen"), d("Time"), d("nCells"), d("nEdges"), d("nVertices"));
    let (d_two, d_me, d_me2, d_vd, d_r3, d_mon) = (
        d("TWO"),
        d("maxEdges"),
        d("maxEdges2"),
        d("vertexDegree"),
        d("R3"),
        d("nMonths"),
    );
    let (d_fifteen, d_soil) = (d("FIFTEEN"), d("nSoilComps"));

    // MPAS: core_init_atmosphere/Registry.xml:705-944,1111-1142.
    // Variable attributes are native Registry units and descriptions verbatim,
    // including its historical spellings and dimensionless metric labels.
    let flt = |name: &str, dd: Vec<usize>, units: &str, long: &str| -> NcVarDef {
        NcVarDef::new(name, NcType::Float, dd).with_attrs(vec![
            NcAttr::text("units", units),
            NcAttr::text("long_name", long),
        ])
    };
    // The FIFTEEN coordinate arrays -- the three lat/lon/xyz families -- are
    // the only ones whose type follows `cfg.coordinates`.  They move together
    // because the port cross-checks xyz against the lat/lon pair beside it and
    // tightens its comparison the moment it sees binary64 metrics; a binary64
    // `xCell` beside a binary32 `latCell` fails the load.  Every METRIC
    // (dcEdge, dvEdge, areaCell, areaTriangle, kiteAreasOnVertex,
    // weightsOnEdge) stays binary32 whatever the coordinates do: a stored
    // metric is a relative quantity that binary32 carries to 6e-8 at any
    // length, and leaving them alone keeps the port's `metric_rtol` on its
    // 2.0e-5 binary32 branch -- the branch every published mesh is judged on.
    let coord = |name: &str, dd: Vec<usize>, units: &str, long: &str| -> NcVarDef {
        NcVarDef::new(name, cfg.coordinates.nc_type(), dd).with_attrs(vec![
            NcAttr::text("units", units),
            NcAttr::text("long_name", long),
        ])
    };
    let int = |name: &str, dd: Vec<usize>, long: &str| -> NcVarDef {
        NcVarDef::new(name, NcType::Int, dd)
            .with_attrs(vec![NcAttr::text("units", match name { "landmask" | "ivgtyp" | "isltyp" => "unitless", _ => "-" }), NcAttr::text("long_name", long)])
    };

    let mut vars = vec![
        NcVarDef::new("xtime", NcType::Char, vec![d_time, d_str]).with_attrs(vec![
            NcAttr::text("units", "YYYY-MM-DD_hh:mm:ss"),
            NcAttr::text("long_name", "Model valid time"),
        ]),
        coord("latCell", vec![d_c], "rad", "Latitude of cells"),
        coord("lonCell", vec![d_c], "rad", "Longitude of cells"),
        coord("xCell", vec![d_c], "m", "Cartesian x-coordinate of cells"),
        coord("yCell", vec![d_c], "m", "Cartesian y-coordinate of cells"),
        coord("zCell", vec![d_c], "m", "Cartesian z-coordinate of cells"),
        int("indexToCellID", vec![d_c], "Mapping from local array index to global cell ID"),
        coord("latEdge", vec![d_e], "rad", "Latitude of edges"),
        coord("lonEdge", vec![d_e], "rad", "Longitude of edges"),
        coord("xEdge", vec![d_e], "m", "Cartesian x-coordinate of edges"),
        coord("yEdge", vec![d_e], "m", "Cartesian y-coordinate of edges"),
        coord("zEdge", vec![d_e], "m", "Cartesian z-coordinate of edges"),
        int("indexToEdgeID", vec![d_e], "Mapping from local array index to global edge ID"),
        coord("latVertex", vec![d_v], "rad", "Latitude of vertices"),
        coord("lonVertex", vec![d_v], "rad", "Longitude of vertices"),
        coord("xVertex", vec![d_v], "m", "Cartesian x-coordinate of vertices"),
        coord("yVertex", vec![d_v], "m", "Cartesian y-coordinate of vertices"),
        coord("zVertex", vec![d_v], "m", "Cartesian z-coordinate of vertices"),
        int("indexToVertexID", vec![d_v], "Mapping from local array index to global vertex ID"),
        int("cellsOnEdge", vec![d_e, d_two], "IDs of cells divided by an edge"),
        int("nEdgesOnCell", vec![d_c], "Number of edges forming the boundary of a cell"),
        int("nEdgesOnEdge", vec![d_e], "Number of edges involved in reconstruction of tangential velocity for an edge"),
        int("edgesOnCell", vec![d_c, d_me], "IDs of edges forming the boundary of a cell"),
        int("edgesOnEdge", vec![d_e, d_me2], "IDs of edges involved in reconstruction of tangential velocity for an edge"),
        flt("weightsOnEdge", vec![d_e, d_me2], "-", "Weights used in reconstruction of tangential velocity for an edge"),
        flt("dvEdge", vec![d_e], "-", "Spherical distance between vertex endpoints of an edge"),
        flt("dcEdge", vec![d_e], "m", "Spherical distance between cells separated by an edge"),
        flt("angleEdge", vec![d_e], "rad", "Angle between local north and the positive tangential direction of an edge"),
        flt("areaCell", vec![d_c], "m^2", "Spherical area of a Voronoi cell"),
        flt("areaTriangle", vec![d_v], "m^2", "Spherical area of a Delaunay triangle"),
        int("cellsOnCell", vec![d_c, d_me], "IDs of cells neighboring a cell"),
        int("verticesOnCell", vec![d_c, d_me], "IDs of vertices (corner points) of a cell"),
        int("verticesOnEdge", vec![d_e, d_two], "IDs of the two vertex endpoints of an edge"),
        int("edgesOnVertex", vec![d_v, d_vd], "IDs of the edges that meet at a vertex"),
        int("cellsOnVertex", vec![d_v, d_vd], "IDs of the cells that meet at a vertex"),
        flt("kiteAreasOnVertex", vec![d_v, d_vd], "m^2", "Intersection areas between primal (Voronoi) and dual (triangular) mesh cells"),
        flt("meshDensity", vec![d_c], "unitless", "Mesh density function (used when generating the mesh) evaluated at a cell"),
        NcVarDef::new("nominalMinDc", NcType::Float, vec![]).with_attrs(vec![
            NcAttr::text("units", "m"),
            NcAttr::text("long_name", "Nominal minimum dcEdge value where meshDensity == 1.0"),
        ]),
        int("bdyMaskCell", vec![d_c], "Indicator of whether a cell is an interior cell, a relaxation-zone cell, or a specified-zone cell"),
        int("bdyMaskEdge", vec![d_e], "Indicator of whether an edge is an interior edge, a relaxation-zone edge, or a specified-zone edge"),
        int("bdyMaskVertex", vec![d_v], "Indicator of whether an vertex is an interior vertex, a relaxation-zone vertex, or a specified-zone vertex"),
        flt("edgeNormalVectors", vec![d_e, d_r3], "unitless", "Cartesian components of the vector normal to an edge and tangential to the surface of the sphere"),
        flt("localVerticalUnitVectors", vec![d_c, d_r3], "unitless", "Cartesian components of the vector pointing in the local vertical direction for a cell"),
        flt("cellTangentPlane", vec![d_c, d_two, d_r3], "unitless", "Components of a pair of vectors defining the tangent plane at a cell"),
        flt(
            "coeffs_reconstruct",
            vec![d_c, d_me, d_r3],
            "unitless", "Coefficients to reconstruct velocity vectors at cell centers",
        ),
        flt(
            "deriv_two",
            vec![d_e, d_two, d_fifteen],
            "unitless", "weights for cell-centered second derivative, normal to edge, for transport scheme",
        ),
        flt("fEdge", vec![d_e], "unitless", "Coriolis parameter at an edge"),
        flt("fVertex", vec![d_v], "unitless", "Coriolis parameter at a vertex"),
        flt("ter", vec![d_c], "m", "terrain height"),
        int("landmask", vec![d_c], "land-ocean mask (1=land ; 0=ocean)"),
        NcVarDef::new("mminlu", NcType::Char, vec![d_str])
            .with_attrs(vec![NcAttr::text("units", "unitless"), NcAttr::text("long_name", "land use classification")]),
        int("ivgtyp", vec![d_c], "dominant vegetation category"),
        int("isltyp", vec![d_c], "dominate soil category"),
        flt("snoalb", vec![d_c], "unitless", "annual maximum snow albedo"),
        flt("soiltemp", vec![d_c], "K", "annual mean deep soil temperature"),
        flt("greenfrac", vec![d_c, d_mon], "percent", "monthly-mean climatological greenness fraction"),
        flt("shdmin", vec![d_c], "percent", "minimum fractional coverage of annual green vegetation fraction"),
        flt("shdmax", vec![d_c], "percent", "maximum fractional coverage of annual green vegetation fraction"),
        flt("albedo12m", vec![d_c, d_mon], "percent", "monthly-mean climatological surface albedo"),
    ];
    for (name, long) in [
        ("var2d", "standard deviation of subgrid-scale orography"),
        ("con", "orographic convexity"),
        ("oa1", "asymmetry of subgrid-scale orography for westerly flow"),
        ("oa2", "asymmetry of subgrid-scale orography for southerly flow"),
        ("oa3", "asymmetry of subgrid-scale orography for south-westerly flow"),
        ("oa4", "asymmetry of subgrid-scale orography for north-westerly flow"),
        ("ol1", "effective orographic length for westerly flow"),
        ("ol2", "effective orographic length for southerly flow"),
        ("ol3", "effective orographic length for south-westerly flow"),
        ("ol4", "effective orographic length for north-westerly flow"),
    ] {
        vars.push(flt(name, vec![d_c], if name == "var2d" {"m"} else {"unitless"}, long));
    }
    // The thirteen this builder adds beyond the published manifest.
    vars.extend([
        flt("cell_gradient_coef_x", vec![d_c, d_me], "m^-1", "Coefficients for computing the x (zonal) derivative of a cell-centered variable"),
        flt("cell_gradient_coef_y", vec![d_c, d_me], "m^-1", "Coefficients for computing the y (meridional) derivative of a cell-centered variable"),
        flt("defc_a", vec![d_c, d_me], "m^-1", "Deformation weights, symmetric part"),
        flt("defc_b", vec![d_c, d_me], "m^-1", "Deformation weights, antisymmetric part"),
        int("lu_index", vec![d_c], "Dominant land use category, WRF spelling of ivgtyp"),
        int("soilcat_top", vec![d_c], "Dominant top-layer soil category, WRF spelling of isltyp"),
        flt("soilcomp", vec![d_c, d_soil], "percent", "soil composition needed as input in the NOAH-MP land surface model"),
        flt("soilcl1", vec![d_c], "unitless", "soil texture class level 1 needed as input for the NOAH-MP land surface model"),
        flt("soilcl2", vec![d_c], "unitless", "soil texture class level 2 needed as input for the NOAH-MP land surface model"),
        flt("soilcl3", vec![d_c], "unitless", "soil texture class level 3 needed as input for the NOAH-MP land surface model"),
        flt("soilcl4", vec![d_c], "unitless", "soil texture class level 4 needed as input for the NOAH-MP land surface model"),
        NcVarDef::new("isice_lu", NcType::Int, vec![])
            .with_attrs(vec![NcAttr::text("units", "unitless"), NcAttr::text("long_name", "Index category for snow/ice")]),
        NcVarDef::new("iswater_lu", NcType::Int, vec![])
            .with_attrs(vec![NcAttr::text("units", "unitless"), NcAttr::text("long_name", "Index category for water")]),
    ]);

    if geo.tc_lakemask.is_some() {
        if !cfg.stages.native_init {
            return Err(MpasError::Refusal("TC lake mask requires the native static stream".into()));
        }
        vars.push(int("tc_lakemask", vec![d_c],
            "lake exclusion mask from unfurled dominant with-lakes land use; 1 lake, 0 otherwise"));
    }

    // MPAS: Registry.xml:498-585,882-903. Native init's immutable static
    // stream follows active packages and has no WRF aliases or SW weights.
    if cfg.stages.native_init {
        vars.retain(|v|native_static_field_enabled(&v.name,&cfg.stages));
        vars.insert(1,NcVarDef::new("Time",NcType::Float,vec![d_time]).with_attrs(vec![
            NcAttr::text("units",format!("seconds since {}",cfg.valid_time.replace('_'," "))),
            NcAttr::text("long_name","CF-compliant valid time"),
            NcAttr::text("standard_name","time"),
        ]));
        let old_dims=dims.clone();
        dims=old_dims.iter().enumerate().filter(|(i,_)|vars.iter().any(|v|v.dimids.contains(i)))
            .map(|(_,d)|d.clone()).collect();
        for var in &mut vars {
            for dimid in &mut var.dimids {
                *dimid=dims.iter().position(|d|d.name==old_dims[*dimid].name).unwrap();
            }
        }
    }

    // The schema gate reads `declared_variables()`, not this vector. If the
    // two ever part, the gate would be grading a manifest nothing writes --
    // green while the file changed underneath it. Catch that here.
    let names: Vec<String> = vars.iter().map(|v| v.name.clone()).collect();
    {
        let actual: Vec<&str> = vars.iter().map(|v| v.name.as_str()).collect();
        if !cfg.stages.native_init && actual != declared_variables() {
            return Err(MpasError::Refusal(format!(
                "this writer emits {} variables but declared_variables() lists {}; \
                 the schema gate reads the second, so it would be grading a stale \
                 manifest until the two agree",
                actual.len(),
                declared_variables().len()
            )));
        }
    }

    // The manifest IS the registered schema. The mesh registry pins grid and
    // static together by byte count and SHA-256, so a variable added or
    // dropped here makes every previously registered pair unmatchable -- and
    // silently, since the run still exits 0 and the file still opens. Refuse
    // before writing rather than after registering.
    if let Some(said) = if cfg.stages.native_init {None} else {crate::staticfile::schema::divergence_from_pin(&names)} {
        return Err(MpasError::Refusal(format!(
            "the static about to be written does not match the registered \
             schema: {said}. A pair built this way is refused at registration \
             by byte count before anything reads it."
        )));
    }

    let mut gattrs = vec![
        NcAttr::text("mesh_spec", crate::mesh::emit::MESH_SPEC),
        NcAttr::text("on_a_sphere", if mesh.on_a_sphere {"YES"} else {"NO"}),
        NcAttr::floats("sphere_radius", vec![mesh.sphere_radius as f32]),
        NcAttr::text("is_periodic", "NO"),
        NcAttr::floats("x_period", vec![0.0]),
        NcAttr::floats("y_period", vec![0.0]),
        NcAttr::text("Conventions", "MPAS"),
        NcAttr::text("source", "MPAS"),
        NcAttr::text(
            "rw_static_engine",
            concat!("rw-mpas ", env!("CARGO_PKG_VERSION"), " (rust)"),
        ),
        NcAttr::text("rw_mpas_static_schema", schema_tag(&cfg.stages)),
        // The provenance carries no wall clock and no filesystem path: a
        // registry pins this file by byte count and sha256, and either one
        // makes two identical builds unregisterable under the same name.
        NcAttr::text("rw_static_provenance_json", provenance_json.to_string()),
    ];
    // The coordinate frame is DECLARED, never inferred.  See
    // `staticfile::coordframe` for what an inferred one breaks.
    gattrs.extend(cfg.coordinates.declaration_attributes(mesh.sphere_radius));

    let mut w = NcClassicWriter::create(
        &cfg.out_path,
        NcFormat::Offset64,
        dims,
        gattrs,
        vars,
        1,
    )?;

    let mut xtime = [b' '; STR_LEN];
    for (i, b) in cfg.valid_time.bytes().take(STR_LEN).enumerate() {
        xtime[i] = b;
    }
    w.put_record("xtime", 0, NcData::Chars(&xtime))?;
    if cfg.stages.native_init {w.put_record("Time",0,NcData::Floats(&[0.0]))?;}

    macro_rules! put_f {
        ($name:literal, $v:expr) => {{
            let tmp = f32_from_f64($v);
            w.put($name, NcData::Floats(&tmp))?;
        }};
    }
    // A coordinate array goes out in the DECLARED representation, and nothing
    // else does.  At binary32 this is byte-for-byte the `put_f!` path (the
    // same `f32_from_f64` on the same values); at binary64 the f64 the mesh
    // was derived in reaches the file unrounded.
    macro_rules! put_coord {
        ($name:literal, $v:expr) => {{
            if cfg.coordinates.is_binary64() {
                w.put($name, NcData::Doubles($v))?;
            } else {
                let tmp = f32_from_f64($v);
                w.put($name, NcData::Floats(&tmp))?;
            }
        }};
    }
    put_coord!("latCell", &mesh.lat_cell);
    put_coord!("lonCell", &mesh.lon_cell);
    put_coord!("xCell", &mesh.x_cell);
    put_coord!("yCell", &mesh.y_cell);
    put_coord!("zCell", &mesh.z_cell);
    w.put("indexToCellID", NcData::Ints(&mesh.index_to_cell_id))?;
    put_coord!("latEdge", &mesh.lat_edge);
    put_coord!("lonEdge", &mesh.lon_edge);
    put_coord!("xEdge", &mesh.x_edge);
    put_coord!("yEdge", &mesh.y_edge);
    put_coord!("zEdge", &mesh.z_edge);
    w.put("indexToEdgeID", NcData::Ints(&mesh.index_to_edge_id))?;
    put_coord!("latVertex", &mesh.lat_vertex);
    put_coord!("lonVertex", &mesh.lon_vertex);
    put_coord!("xVertex", &mesh.x_vertex);
    put_coord!("yVertex", &mesh.y_vertex);
    put_coord!("zVertex", &mesh.z_vertex);
    w.put("indexToVertexID", NcData::Ints(&mesh.index_to_vertex_id))?;

    // Topology goes out exactly as the grid stored it; see the `raw_*` note
    // on [`Mesh`].
    w.put("cellsOnEdge", NcData::Ints(&mesh.raw_cells_on_edge))?;
    let neoc: Vec<i32> = mesh.n_edges_on_cell.iter().map(|&v| v as i32).collect();
    w.put("nEdgesOnCell", NcData::Ints(&neoc))?;
    w.put("nEdgesOnEdge", NcData::Ints(&mesh.n_edges_on_edge))?;
    w.put("edgesOnCell", NcData::Ints(&static_topology_words(&mesh.raw_edges_on_cell,mesh.max_edges,&mesh.n_edges_on_cell,cfg.stages.native_init)))?;
    let edge_counts:Vec<usize>=mesh.n_edges_on_edge.iter().map(|&n|n as usize).collect();
    w.put("edgesOnEdge", NcData::Ints(&static_topology_words(&mesh.raw_edges_on_edge,mesh.max_edges2,&edge_counts,cfg.stages.native_init)))?;
    put_f!("weightsOnEdge", &mesh.weights_on_edge);
    put_f!("dvEdge", &mesh.dv_edge);
    put_f!("dcEdge", &mesh.dc_edge);
    put_f!("angleEdge", &mesh.angle_edge);
    put_f!("areaCell", &mesh.area_cell);
    put_f!("areaTriangle", &mesh.area_triangle);
    w.put("cellsOnCell", NcData::Ints(&static_topology_words(&mesh.raw_cells_on_cell,mesh.max_edges,&mesh.n_edges_on_cell,cfg.stages.native_init)))?;
    w.put("verticesOnCell", NcData::Ints(&static_topology_words(&mesh.raw_vertices_on_cell,mesh.max_edges,&mesh.n_edges_on_cell,cfg.stages.native_init)))?;
    w.put("verticesOnEdge", NcData::Ints(&mesh.raw_vertices_on_edge))?;
    w.put("edgesOnVertex", NcData::Ints(&mesh.raw_edges_on_vertex))?;
    w.put("cellsOnVertex", NcData::Ints(&mesh.raw_cells_on_vertex))?;
    put_f!("kiteAreasOnVertex", &mesh.kite_areas_on_vertex);
    put_f!("meshDensity", &mesh.mesh_density);
    w.put("nominalMinDc", NcData::Floats(&[nominal_dx_m as f32]))?;

    w.put("bdyMaskCell", NcData::Ints(&mesh.bdy_mask_cell))?;
    w.put("bdyMaskEdge", NcData::Ints(&mesh.bdy_mask_edge))?;
    w.put("bdyMaskVertex", NcData::Ints(&mesh.bdy_mask_vertex))?;

    // The four local-frame arrays are PLACEHOLDERS in a static, not values.
    //
    // Measured on the published `x4.163842.static.nc`: `edgeNormalVectors`,
    // `localVerticalUnitVectors`, `cellTangentPlane` and `coeffs_reconstruct`
    // are ALL exactly zero, every element. The real frames belong to the
    // vertical grid the init stage builds, and a consumer overlays them from
    // the init file. Its pin is the digest of an exact +0 array at this
    // shape, so a derived frame here -- right as geometry -- is refused with
    // `static edgeNormalVectors placeholder bytes changed` before any device
    // memory is allocated.
    //
    // The SLOT is not optional either. Leaving `coeffs_reconstruct` out
    // stops the port at `AttributeError: mesh has no 'coeffs_reconstruct'`.
    w.put("edgeNormalVectors", NcData::Floats(&vec![0.0f32; ne * 3]))?;
    w.put("localVerticalUnitVectors", NcData::Floats(&vec![0.0f32; nc * 3]))?;
    w.put("cellTangentPlane", NcData::Floats(&vec![0.0f32; nc * 2 * 3]))?;
    w.put("coeffs_reconstruct", NcData::Floats(&vec![0.0f32; nc * me * 3]))?;

    w.put("deriv_two", NcData::Floats(&operators.deriv_two))?;

    let fedge: Vec<f32> = mesh.lat_edge.iter().map(|&lat| 2.0f32 * MPAS_OMEGA as f32 * (lat as f32).sin()).collect();
    let fvertex: Vec<f32> = mesh.lat_vertex.iter().map(|&lat| 2.0f32 * MPAS_OMEGA as f32 * (lat as f32).sin()).collect();
    w.put("fEdge", NcData::Floats(&fedge))?;
    w.put("fVertex", NcData::Floats(&fvertex))?;

    w.put("ter", NcData::Floats(&geo.ter))?;
    w.put("landmask", NcData::Ints(&geo.landmask))?;
    let mut mminlu = [b' '; STR_LEN];
    let label = geo.mminlu.as_bytes();
    if label.len() > STR_LEN {
        return Err(MpasError::Refusal(format!(
            "mminlu label is {} bytes, exceeds StrLen={STR_LEN}",
            label.len()
        )));
    }
    mminlu[..label.len()].copy_from_slice(label);
    w.put("mminlu", NcData::Chars(&mminlu))?;
    if let Some(mask) = &geo.tc_lakemask { w.put("tc_lakemask", NcData::Ints(mask))?; }
    w.put("ivgtyp", NcData::Ints(&geo.ivgtyp))?;
    w.put("isltyp", NcData::Ints(&geo.isltyp))?;
    w.put("snoalb", NcData::Floats(&geo.snoalb))?;
    w.put("soiltemp", NcData::Floats(&geo.soiltemp))?;
    w.put("greenfrac", NcData::Floats(&geo.greenfrac))?;

    let mut shdmin = vec![0.0f32; nc];
    let mut shdmax = vec![0.0f32; nc];
    for c in 0..nc {
        let row = &geo.greenfrac[c * N_MONTHS..(c + 1) * N_MONTHS];
        shdmin[c] = row.iter().copied().fold(f32::INFINITY, f32::min);
        shdmax[c] = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    }
    w.put("shdmin", NcData::Floats(&shdmin))?;
    w.put("shdmax", NcData::Floats(&shdmax))?;
    w.put("albedo12m", NcData::Floats(&geo.albedo12m))?;

    if !cfg.stages.native_init || cfg.stages.gwd {
    w.put("var2d", NcData::Floats(&f32_from_f64(&gwd.var2d)))?;
    w.put("con", NcData::Floats(&f32_from_f64(&gwd.con)))?;
    for k in 0..4 {
        w.put(&format!("oa{}", k + 1), NcData::Floats(&f32_from_f64(&gwd.oa[k])))?;
    }
    for k in 0..4 {
        w.put(&format!("ol{}", k + 1), NcData::Floats(&f32_from_f64(&gwd.ol[k])))?;
    }
    }

    w.put("cell_gradient_coef_x", NcData::Floats(&operators.cell_gradient_coef_x))?;
    w.put("cell_gradient_coef_y", NcData::Floats(&operators.cell_gradient_coef_y))?;
    if !cfg.stages.native_init {
    w.put("defc_a", NcData::Floats(&operators.defc_a))?;
    w.put("defc_b", NcData::Floats(&operators.defc_b))?;
    w.put("lu_index", NcData::Ints(&geo.ivgtyp))?;
    w.put("soilcat_top", NcData::Ints(&geo.isltyp))?;
    }
    if !cfg.stages.native_init || cfg.stages.noahmp {
    w.put("soilcomp", NcData::Floats(&geo.soilcomp))?;
    for (name, src) in [
        ("soilcl1", &geo.soilcl[0]),
        ("soilcl2", &geo.soilcl[1]),
        ("soilcl3", &geo.soilcl[2]),
        ("soilcl4", &geo.soilcl[3]),
    ] {
        let tmp: Vec<f32> = src.iter().map(|&v| v as f32).collect();
        w.put(name, NcData::Floats(&tmp))?;
    }
    }
    w.put("isice_lu", NcData::Ints(&[geo.isice_lu]))?;
    w.put("iswater_lu", NcData::Ints(&[geo.iswater_lu]))?;
    w.finish()?;
    Ok(())
}

/// How many source rows the sub-grid orography pass may hold at once when the
/// budget allows it. Below this the pass still works; above it the extra rows
/// buy nothing, because a band is only useful up to the tallest box in it.
const GWD_PREFERRED_BAND_ROWS: i64 = 768;

/// Each cell's mean `dcEdge`, in metres. The box the drag statistics are cut
/// from is sized from this and nothing else.
fn mean_edge_length_m(mesh: &Mesh) -> Vec<f64> {
    let mut out = vec![0.0f64; mesh.n_cells];
    for c in 0..mesh.n_cells {
        let k = mesh.n_edges_on_cell[c];
        if k == 0 {
            continue;
        }
        let mut sum = 0.0;
        for s in 0..k {
            let e = mesh.edges_on_cell[c * mesh.max_edges + s];
            if e < mesh.n_edges {
                sum += mesh.dc_edge[e];
            }
        }
        out[c] = sum / k as f64;
    }
    out
}

/// Fold a land-use archive's declared lake class into its water class.
///
/// WPS indexes name the lake class as `islake` (21 in the with-lakes MODIS
/// set). MPAS's Noah has no lake category, and WRF real.exe folds the
/// declared lake class into water the same way. Without it a lake cell whose STATSGO soil is
/// not water stayed category 21 and land. An index with no lake class (the
/// parser's -1) changes nothing. Returns the number of cells folded.
pub fn fold_declared_lake(landuse: &mut [i32], islake: Option<i64>, water: i32) -> usize {
    let Some(lake) = islake.filter(|&c| c > 0).map(|c| c as i32) else { return 0 };
    let mut folded = 0;
    for lu in landuse.iter_mut().filter(|lu| **lu == lake) {
        *lu = water;
        folded += 1;
    }
    folded
}

/// Every geography directory this build will open that has no WPS `index`,
/// in the order the build opens them, each named once.
///
/// Defect (hex global run 2026-10-16, two refusals in a row on a WRF-only
/// archive): the check looked at [`StaticGeogPaths::slots`] alone, but the
/// native GWD stage opens its own terrain and the USGS `landuse_30s` mask
/// ([`crate::native_gwd::NativeGwdConfig`]). A box missing that mask passed
/// the check, ran 26 s, and was refused at the tile read, after the first
/// refusal had already been "fixed". The set here is everything the build
/// opens, so one refusal names it all.
pub fn missing_geog_datasets(cfg: &StaticBuildConfig) -> Vec<String> {
    let mut opened: Vec<&Path> = cfg.geog.slots().iter()
        .filter(|(name,_)| (cfg.stages.noahmp || !name.starts_with("soilcl") && *name != "soilcomp")
            && !(*name=="greenfrac" && cfg.stages.vegetation_ncep)
            && !(*name=="albedo" && cfg.stages.albedo_ncep)
            && !(*name=="snow_albedo" && cfg.stages.snow_ncep)
            && !(*name=="soil_temperature" && cfg.stages.native_init))
        .map(|(_,path)| *path).collect();
    if let Some(gwd) = cfg.native_gwd.as_ref().filter(|_| cfg.stages.gwd) {
        opened.push(&gwd.topo);
        opened.push(&gwd.landuse);
    }
    let mut missing: Vec<String> = Vec::new();
    for path in opened {
        let shown = path.display().to_string();
        if !path.join("index").is_file() && !missing.contains(&shown) {
            missing.push(shown);
        }
    }
    missing
}

/// Build a static, silently. See [`build_static_reporting`].
/// MPAS: mpas_init_atm_static.F:110-1260 (init_atm_static).
pub fn build_static(cfg: &StaticBuildConfig) -> MpasResult<StaticBuildReceipt> {
    build_static_reporting(cfg, &|_| {})
}

/// Build a static, streaming one tab-separated progress line per stage.
///
/// The grammar is the door's: `GRID`, `NOMINALDX`, `VECTORS`, `OPERATORS`,
/// `GEOGGRID`, `GWDBAND`. A wrapper parses these, so a token that stops being
/// printed is a wrapper that stops reporting rather than a wrapper that fails.
/// MPAS: mpas_init_atm_static.F:110-1260 (complete static driver boundary).
pub fn build_static_reporting(
    cfg: &StaticBuildConfig,
    progress: &dyn Fn(&str),
) -> MpasResult<StaticBuildReceipt> {
    if cfg.out_path.exists() && !cfg.clobber {
        return Err(MpasError::Refusal(format!(
            "{} exists; pass --clobber to replace it. Overwriting a static \
             silently would leave every init file and run receipt that pinned \
             its sha256 pointing at different bytes under the same name",
            cfg.out_path.display()
        )));
    }
    let missing = missing_geog_datasets(cfg);
    if !missing.is_empty() {
        return Err(MpasError::Refusal(format!(
            "the geography archive is missing {} dataset(s) a static needs: {}. \
             Terrain, land use, soil class and composition, deep-soil \
             temperature, green-ness, albedo and the gravity-wave-drag \
             land-use mask all come from these directories, and a static \
             written without one would hand the model a field of zeros that \
             the run would carry as real geography.\n\
             what does work:\n  \
             gpuwm fetch-geog --datasets mesh, which stages every dataset this \
             builder opens\n  \
             --geog DIR pointing at a complete archive",
            missing.len(),
            missing.join(", ")
        )));
    }
    // Read at the precision this static will DECLARE; see `Mesh::read_as`.
    let mesh = Mesh::read_as(&cfg.grid_path, cfg.coordinates)?;
    progress(&format!(
        "GRID\t{}\t{}\t{}\t{}",
        mesh.n_cells, mesh.n_edges, mesh.n_vertices, mesh.sphere_radius
    ));

    // The one value that is DECLARED rather than measured. A consumer compares
    // its bit pattern against the mesh binding's nominal dx and refuses on a
    // near-miss, because that scalar is what the drag length scale is rebound
    // to.
    let implied = mesh.nominal_min_dc;
    let nominal_dx_m = match cfg.nominal_dx_m {
        None => implied,
        Some(metres) => {
            if !(metres.is_finite() && metres > 0.0) {
                return Err(MpasError::Refusal(format!(
                    "nominal dx {metres} is not a positive length; the drag \
                     length scale rebound to it would be meaningless"
                )));
            }
            if (implied - metres).abs() > NOMINAL_DX_CROSS_CHECK_RELATIVE * metres {
                return Err(MpasError::Refusal(format!(
                    "the grid's nominalMinDc implies {implied:.3} m, but the \
                     static is being told to declare {metres:.3} m. They are \
                     more than {:.3} % apart, so this grid and this static are \
                     not the same mesh and pairing them would give every cell a \
                     horizontal mixing length taken from a different resolution",
                    NOMINAL_DX_CROSS_CHECK_RELATIVE * 100.0
                )));
            }
            metres
        }
    };
    progress(&format!(
        "NOMINALDX\t{:.6}\t{:.9}\t{:#010x}",
        nominal_dx_m,
        nominal_dx_m as f32,
        (nominal_dx_m as f32).to_bits()
    ));
    progress(
        "VECTORS\tfEdge fVertex edgeNormalVectors localVerticalUnitVectors cellTangentPlane",
    );

    let (resident, per_worker, writer_slab, cache_request) = build_footprint(&mesh, cfg)?;

    // The drag band's own resident cost, sized from the geography it will
    // actually read. It does not overlap the tile workers -- the pool is gone
    // by then -- so it competes with the writer slab for the same one-shot
    // slot rather than adding to it.
    let dc_m = mean_edge_length_m(&mesh);
    let native_mesh = cfg.native_gwd.as_ref().filter(|_| cfg.stages.gwd)
        .map(|gwd| crate::native_gwd::NativeMesh::read(&cfg.grid_path, gwd.cell_scaling))
        .transpose()?;
    let gwd_topo = GeogDataset::open(cfg.native_gwd.as_ref().map_or(&cfg.geog.terrain, |gwd| &gwd.topo))?;
    let gwd_min_rows = if !cfg.stages.gwd { 0 } else { match &native_mesh {
        Some(native) => native.minimum_band_rows()?,
        None => crate::static_gwd::minimum_band_rows(&gwd_topo, &mesh.lat_cell, &dc_m)?,
    } };
    let gwd_rows = gwd_min_rows.max(GWD_PREFERRED_BAND_ROWS);
    let one_shot = |rows: i64| if cfg.stages.gwd { writer_slab.max(crate::static_gwd::band_bytes(&gwd_topo, rows)) } else { writer_slab };

    let mut gwd_band_rows = gwd_rows;
    let plan = match plan_tile_parallelism(
        "preflight",
        resident,
        per_worker,
        one_shot(gwd_rows),
        1024 * 1024 * 1024,
        resolve_tile_workers(cfg.tile_workers),
        cache_request,
        cfg.host_memory_limit_bytes,
    ) {
        Ok(plan) => plan,
        Err(wide) if gwd_min_rows < gwd_rows => {
            // Fall back to the narrowest band that still serves every cell.
            // The ANSWER is identical either way -- a band always covers the
            // whole of every box it serves -- so this trades throughput for
            // headroom and never a number in the file.
            gwd_band_rows = gwd_min_rows;
            plan_tile_parallelism(
                "preflight",
                resident,
                per_worker,
                one_shot(gwd_min_rows),
                1024 * 1024 * 1024,
                resolve_tile_workers(cfg.tile_workers),
                cache_request,
                cfg.host_memory_limit_bytes,
            )
            .map_err(|narrow| {
                MpasError::Refusal(format!(
                    "{narrow}\n\nA narrower sub-grid orography band was tried \
                     first and also refused; the wide-band refusal was: {wide}"
                ))
            })?
        }
        Err(e) => return Err(e),
    };
    drop(gwd_topo);
    let predicted = plan.predicted_peak_bytes;
    let limit = plan.limit_bytes;
    let mut memory = MemoryReceipt::new(limit, Some(predicted));
    memory.event(
        "preflight",
        None,
        "mesh loaded and host-memory admission passed",
        None,
        Some(predicted),
        None,
        vec![
            BufferLedger {
                name: "mesh geometry/topology".to_string(),
                bytes: resident,
                lifetime: "whole build".to_string(),
            },
            BufferLedger {
                name: format!("tile scratch x {} concurrent workers", plan.workers),
                bytes: (plan.workers as u64).saturating_mul(per_worker),
                lifetime: "one geography pass".to_string(),
            },
            BufferLedger {
                name: "reusable tile-map cache".to_string(),
                bytes: plan.tile_map_cache_bytes,
                lifetime: "whole build".to_string(),
            },
        ],
    );

    let tree = KdTree::new(&mesh);
    let max_d2 = mesh.max_search_distance2();
    let cache = TileMapCache::new(plan.tile_map_cache_bytes);
    let pool = if plan.workers > 1 {
        Some(
            rayon::ThreadPoolBuilder::new().stack_size(crate::execution::BRIDGE_STACK_BYTES)
                .num_threads(plan.workers)
                .thread_name(|i| format!("rw-mpas-tile-{i}"))
                .build()
                .map_err(|e| {
                    MpasError::Refusal(format!(
                        "cannot start {} admitted tile workers: {e}. The build would \
                         otherwise run on whatever pool happened to exist, which is not \
                         the worker count host-memory admission sized. Remedy: set \
                         --tile-workers 1 to build serially.",
                        plan.workers
                    ))
                })?,
        )
    } else {
        None
    };
    let ctx = GeogContext {
        mesh: &mesh,
        tree: &tree,
        max_d2,
        workers: plan.workers,
        pool: pool.as_ref(),
        cache: &cache,
    };

    memory.event("operators", None, "before operator construction", None, None, None, vec![]);
    let operators = build_operator_fields(&mesh.operator_view())?;
    progress(&format!(
        "OPERATORS\tderiv_two\t{}\t{}",
        operators.deriv_two.len(),
        FIFTEEN
    ));
    memory.event(
        "operators",
        None,
        "after operator construction",
        None,
        Some(
            (operators.cell_gradient_coef_x.len()
                + operators.cell_gradient_coef_y.len()
                + operators.defc_a.len()
                + operators.defc_b.len()
                + operators.deriv_two.len()) as u64
                * 4,
        ),
        None,
        vec![],
    );

    validate_supersample(cfg.supersample_landuse)?;
    validate_supersample(cfg.supersample_30s)?;

    let mut datasets = Vec::new();
    let mut time_field = |field: &str, path: &Path, factor: usize, f: &mut dyn FnMut() -> MpasResult<(u64, u64)>| -> MpasResult<()> {
        let started = Instant::now();
        memory.event(field, Some(path), "before dataset", None, None, None, vec![]);
        take_centre_seeded();
        let (plane_bytes, dest_bytes) = f()?;
        let centre_seeded_cells = take_centre_seeded();
        memory.event(
            field,
            Some(path),
            "after dataset; source tile/plane workspace dropped",
            Some(plane_bytes),
            Some(dest_bytes),
            None,
            vec![],
        );
        let ds = open_stage_dataset(cfg,field,path)?;
        let elapsed = started.elapsed().as_secs_f64();
        progress(&format!(
            "GEOGGRID\t{field}\t{}\t{}\t{}\t{elapsed:.2}",
            path.file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default(),
            ds.tiles.len(),
            factor,
        ));
        datasets.push(DatasetReceipt {
            field: field.to_string(),
            path: path.display().to_string(),
            tiles: ds.tiles.len(),
            source_plane_bytes: ds.plane_bytes(),
            supersample: factor,
            destination_bytes: dest_bytes,
            elapsed_seconds: elapsed,
            spread_rounds: 0,
            centre_seeded_cells,
        });
        Ok(())
    };

    let mut ter: Vec<f32> = Vec::new();
    time_field("ter", &cfg.geog.terrain, cfg.supersample_30s, &mut || {
        let (v, ds) = continuous_mean(&ctx, &cfg.geog.terrain, cfg.supersample_30s, None, MeanPolicy::Terrain)?;
        if ds.index.nz() != 1 {
            return Err(MpasError::Refusal("terrain source must have one plane".to_string()));
        }
        ter = v;
        Ok((ds.plane_bytes(), (ter.len() * 4) as u64))
    })?;

    let mut ivgtyp = Vec::new();
    let mut landuse_ds_opt = None;
    time_field("landuse", &cfg.geog.landuse, cfg.supersample_landuse, &mut || {
        let (v, _, _, ds) =
            categorical_with(&ctx, &cfg.geog.landuse, cfg.supersample_landuse, cfg.stages.native_init)?;
        ivgtyp = v;
        let pb = ds.plane_bytes();
        landuse_ds_opt = Some(ds);
        Ok((pb, (ivgtyp.len() * 4) as u64))
    })?;
    let mut tc_lakemask = None;
    if let Some(path) = &cfg.stages.lake_mask_geog {
        time_field("tc_lakemask", path, cfg.supersample_landuse, &mut || {
            // Independently retain the original classification. Native's pixel-wise
            // lake-to-water vote and its landmask remain byte-for-byte unchanged.
            let (v, _, _, ds) = categorical(&ctx, path, cfg.supersample_landuse)?;
            let lake = ds.index.islake.filter(|&x| x > 0).ok_or_else(||
                MpasError::Refusal("lake-mask geography must declare a positive islake class".into()))?;
            tc_lakemask = Some(v.into_iter().map(|x| i32::from(x as i64 == lake)).collect());
            Ok((ds.plane_bytes(), (mesh.n_cells * 4) as u64))
        })?;
    }
    let landuse_ds = landuse_ds_opt.take().expect("landuse populated");
    let iswater_lu = landuse_ds.index.iswater.unwrap_or(17) as i32;
    let isice_lu = landuse_ds.index.isice.unwrap_or(15) as i32;
    let mut iswater_soil = 14;
    let mut isltyp = Vec::new();
    time_field("soilcat_top", &cfg.geog.soilcat, cfg.supersample_30s, &mut || {
        let (v, _, _, ds) = categorical(&ctx, &cfg.geog.soilcat, cfg.supersample_30s)?;
        isltyp = v;
        iswater_soil = ds.index.isoilwater.unwrap_or(14) as i32;
        Ok((ds.plane_bytes(), (isltyp.len() * 4) as u64))
    })?;

    // A land-use archive that declares a lake class (the with-lakes MODIS set
    // read by `--legacy-static`: islake=21) has its lakes folded into its water
    // class before the native reconciliation, so they leave as water (land use
    // and soil) with landmask 0. The native data sets declare no lake class, so
    // the native default is unchanged.
    fold_declared_lake(&mut ivgtyp, landuse_ds.index.islake, iswater_lu);
    let landmask = crate::static_native::reconcile_surface(
        &mut ivgtyp, &mut isltyp, isice_lu, iswater_lu, iswater_soil);

    let mut greenfrac = Vec::new();
    time_field("greenfrac", &cfg.geog.greenfrac, cfg.supersample_30s, &mut || {
        let (v, ds) = if cfg.stages.vegetation_ncep {
            direct_climatology(&mesh,&cfg.geog.greenfrac,&landmask,-1.0e30,0.0,1.0,
                cfg.stages.snow_ncep || !cfg.stages.native_init)?
        } else { continuous_mean(&ctx, &cfg.geog.greenfrac, cfg.supersample_30s, Some(&landmask), MeanPolicy::Green)? };
        if ds.index.nz() != 12 {
            return Err(MpasError::Refusal(format!(
                "greenfrac must carry 12 months, got {}",
                ds.index.nz()
            )));
        }
        // MPAS: mpas_init_atm_static.F:860-863: raw MODIS values are percent.
        greenfrac = v;
        Ok((ds.plane_bytes(), (greenfrac.len() * 4) as u64))
    })?;

    let mut albedo12m = Vec::new();
    time_field("albedo12m", &cfg.geog.albedo, cfg.supersample, &mut || {
        let (v, ds) = if cfg.stages.albedo_ncep {
            direct_climatology(&mesh,&cfg.geog.albedo,&landmask,0.0,8.0,1.0,
                cfg.stages.snow_ncep || !cfg.stages.native_init)?
        } else { continuous_mean(&ctx, &cfg.geog.albedo, cfg.supersample, Some(&landmask), MeanPolicy::Albedo)? };
        if ds.index.nz() != 12 {
            return Err(MpasError::Refusal(format!(
                "albedo must carry 12 months, got {}",
                ds.index.nz()
            )));
        }
        albedo12m = v;
        // MPAS: mpas_init_atm_static.F:1104-1108.
        for c in 0..mesh.n_cells {
            if !cfg.stages.albedo_ncep && ivgtyp[c] == isice_lu { albedo12m[c*12..(c+1)*12].fill(70.0); }
        }
        Ok((ds.plane_bytes(), (albedo12m.len() * 4) as u64))
    })?;

    let mut snoalb = Vec::new();
    time_field("snoalb", &cfg.geog.snow_albedo, cfg.supersample, &mut || {
        let (v, ds) = if cfg.stages.snow_ncep {
            direct_climatology(&mesh,&cfg.geog.snow_albedo,&landmask,0.0,0.0,100.0,true)?
        } else { snow_albedo(&ctx, &cfg.geog.snow_albedo, cfg.supersample, &landmask)? };
        snoalb = v;
        Ok((ds.plane_bytes(), (snoalb.len() * 4) as u64))
    })?;

    let mut soiltemp = Vec::new();
    time_field("soiltemp", &cfg.geog.soil_temperature, 1, &mut || {
        let (v, ds) = if cfg.stages.native_init {
            soil_temperature_dataset(&mesh,GeogDataset::open_native_soil_temperature(&cfg.geog.soil_temperature)?,&landmask)?
        } else { soil_temperature(&mesh, &cfg.geog.soil_temperature, &landmask)? };
        soiltemp = v;
        Ok((ds.plane_bytes(), (soiltemp.len() * 4) as u64))
    })?;

    let mut soilcomp_v = vec![0.0; mesh.n_cells * 8];
    if cfg.stages.noahmp { time_field("soilcomp", &cfg.geog.soilcomp, cfg.supersample_30s, &mut || {
        let (v, ds) = soilcomp(&ctx, &cfg.geog.soilcomp, cfg.supersample_30s, &landmask)?;
        soilcomp_v = v;
        Ok((ds.plane_bytes(), (soilcomp_v.len() * 4) as u64))
    })?;

    }
    let mut soilcls: [Vec<i32>; 4] = std::array::from_fn(|_| vec![0; mesh.n_cells]);
    let paths = [&cfg.geog.soilcl1, &cfg.geog.soilcl2, &cfg.geog.soilcl3, &cfg.geog.soilcl4];
    for idx in 0..4 {
        if !cfg.stages.noahmp { break; }
        let field = format!("soilcl{}", idx + 1);
        time_field(&field, paths[idx], cfg.supersample_30s, &mut || {
            let (v, _, _, ds) = categorical(&ctx, paths[idx], cfg.supersample_30s)?;
            soilcls[idx] = v;
            Ok((ds.plane_bytes(), (soilcls[idx].len() * 4) as u64))
        })?;
    }

    let tile_map_cache = cache.receipt()?;
    // Release the workers before the drag band and the writer allocate: the
    // peak the admission gate was shown does not include them at once.
    drop(ctx);
    drop(pool);

    // The sub-grid orography band. It runs after the tile pool is gone and
    // holds two full-width source bands; that is the term the admission gate
    // was shown as this phase's one-shot.
    let gwd_bytes = if cfg.stages.gwd { crate::static_gwd::band_bytes(
        &GeogDataset::open(&cfg.geog.terrain)?,
        gwd_band_rows,
    ) } else { 0 };
    memory.event(
        "gwd",
        Some(&cfg.geog.terrain),
        "before the sub-grid orography band",
        None,
        None,
        Some(gwd_bytes),
        vec![BufferLedger {
            name: format!("terrain + land-use bands, {gwd_band_rows} rows"),
            bytes: gwd_bytes,
            lifetime: "one orography band".to_string(),
        }],
    );
    let t_gwd = Instant::now();
    let gwd = if !cfg.stages.gwd {
        crate::static_gwd::GwdFields {
            var2d: vec![0.0; mesh.n_cells], con: vec![0.0; mesh.n_cells],
            oa: std::array::from_fn(|_| vec![0.0; mesh.n_cells]),
            ol: std::array::from_fn(|_| vec![0.0; mesh.n_cells]),
            ..Default::default()
        }
    } else if let (Some(native), Some(native_mesh)) = (&cfg.native_gwd, &native_mesh) {
        let mut native = native.clone();
        native.host_memory_limit_bytes = cfg.host_memory_limit_bytes;
        native.tile_workers = cfg.tile_workers;
        native.extra_resident_bytes = resident;
        crate::native_gwd::compute(&native, native_mesh, gwd_band_rows, progress)?
    } else {
        let topo = GeogDataset::open(&cfg.geog.terrain)?;
        let lu = GeogDataset::open(&cfg.geog.landuse)?;
        crate::static_gwd::compute(
            &topo,
            &lu,
            &mesh.lat_cell,
            &mesh.lon_cell,
            &dc_m,
            &mesh.n_edges_on_cell,
            &mesh.cells_on_cell,
            mesh.max_edges,
            gwd_band_rows,
            progress,
        )?
    };
    let gwd_seconds = t_gwd.elapsed().as_secs_f64();
    memory.event(
        "gwd",
        Some(&cfg.geog.terrain),
        "after the sub-grid orography band; source bands dropped",
        Some(gwd_bytes),
        Some((mesh.n_cells * 10 * 8) as u64),
        None,
        vec![],
    );

    // What FP32 storage does to this file's own edge lengths, measured on the
    // values about to be written rather than predicted from the resolution,
    // and judged against the port's LIVE load contract (the 1.73 m storage
    // atol and the 0.02 dvEdge/dcEdge admission floor -- the retired
    // rtol 2e-5 / atol 0.0 comparison died with the port's 2026-08-23
    // contract change; stale-guard audit 2026-08-25, finding 3).  This
    // reading decides whether the mesh being built can reach a forecast.
    let f32v = |v: &[f64]| -> Vec<f32> { v.iter().map(|&x| x as f32).collect() };
    // Coordinates go through the DECLARED representation, metrics always
    // through f32: that is exactly what the writer below does, so this
    // reading is taken on the bytes the file will carry rather than on a
    // model of them.
    let coordv = |v: &[f64]| -> Vec<f64> {
        if cfg.coordinates.is_binary64() {
            v.to_vec()
        } else {
            v.iter().map(|&x| (x as f32) as f64).collect()
        }
    };
    let fp32_metric_agreement = crate::staticfile::fp32metrics::measure(
        &f32v(&mesh.dv_edge),
        &coordv(&mesh.x_vertex),
        &coordv(&mesh.y_vertex),
        &coordv(&mesh.z_vertex),
        &mesh
            .raw_vertices_on_edge
            .iter()
            .map(|&v| v as i64)
            .collect::<Vec<i64>>(),
        &f32v(&mesh.dc_edge),
        mesh.sphere_radius,
        cfg.coordinates,
    );
    progress(&format!(
        "FP32METRICS\t{:.3e}\t{:.3}\t{}\t{:.3e}\t{}\t{}",
        fp32_metric_agreement.max_dv_edge_absolute_m,
        fp32_metric_agreement.min_dv_edge_m,
        fp32_metric_agreement.edges_past_port_storage_tolerance,
        fp32_metric_agreement.min_dv_over_dc,
        fp32_metric_agreement.edges_below_admission_floor,
        fp32_metric_agreement.port_accepts()
    ));

    let provenance = StaticProvenance {
        engine: concat!("rw-mpas ", env!("CARGO_PKG_VERSION"), " (rust)").to_string(),
        schema: schema_tag(&cfg.stages).to_string(),
        grid_sha256: crate::sha256_file(&cfg.grid_path)?,
        n_cells: mesh.n_cells,
        n_edges: mesh.n_edges,
        n_vertices: mesh.n_vertices,
        sphere_radius: mesh.sphere_radius,
        nominal_dx_m,
        nominal_dx_f32_bits: (nominal_dx_m as f32).to_bits(),
        gwd_water_category: gwd.water_category,
        gwd_water_category_source: gwd.water_category_source.clone(),
        gwd_variance_smoothed_cells: gwd.smoothed_cells,
        mminlu: cfg.stages.landuse_label.clone().unwrap_or_else(||landuse_ds.index.mminlu.clone()),
        native_undefined_ol1_cells: native_mesh.as_ref().map_or_else(Vec::new, |m|
            (0..m.lat.len()).filter(|&c|
                crate::native_gwd::box_size(m.lat[c], m.dc[c]).1 < 4).collect()),
        native_undefined_exclusion_status: native_mesh.as_ref().and_then(|m|
            (0..m.lat.len()).any(|c| crate::native_gwd::box_size(m.lat[c], m.dc[c]).1 < 4)
                .then(|| "PROPOSED FOR OWNER SIGN-OFF: case-specific sign-off required".into())),
    };
    let provenance_json = serde_json::to_string(&provenance)
        .map_err(|e| MpasError::Refusal(format!("cannot serialise the provenance: {e}")))?;

    let geo = GeographyFields {
        tc_lakemask,
        ter,
        landmask,
        ivgtyp,
        isltyp,
        greenfrac,
        albedo12m,
        snoalb,
        soiltemp,
        soilcomp: soilcomp_v,
        soilcl: soilcls,
        isice_lu,
        iswater_lu,
        mminlu: cfg.stages.landuse_label.clone().unwrap_or_else(||landuse_ds.index.mminlu.clone()),
    };

    memory.event("write", None, "before static write", None, None, None, vec![]);
    write_static(cfg, &mesh, &operators, &geo, &gwd, nominal_dx_m, &provenance_json)?;
    memory.event("write", None, "static fsync complete", None, None, None, vec![]);

    let output_bytes = std::fs::metadata(&cfg.out_path)?.len();
    let mut generated_fields: Vec<String> = vec![
        "cell_gradient_coef_x", "cell_gradient_coef_y", "defc_a", "defc_b",
        "deriv_two", "fEdge", "fVertex", "ter", "landmask", "lu_index", "ivgtyp",
        "soilcat_top", "isltyp", "greenfrac", "albedo12m", "shdmin", "shdmax",
        "snoalb", "soiltemp", "soilcomp", "soilcl1", "soilcl2", "soilcl3", "soilcl4",
        "var2d", "con", "oa1", "oa2", "oa3", "oa4", "ol1", "ol2", "ol3", "ol4",
    ]
    .into_iter()
    .filter(|name| !cfg.stages.native_init || native_static_field_enabled(name,&cfg.stages))
    .map(str::to_string)
    .collect();

    if geo.tc_lakemask.is_some() { generated_fields.push("tc_lakemask".into()); }

    let operator_provenance = BTreeMap::from([
        (
            "cell_gradient_coef_x/y".to_string(),
            "MPAS-A v8.4.2 core_init_atmosphere/mpas_atm_advection.F:744-923::atm_initialize_deformation_weights".to_string(),
        ),
        (
            "deriv_two".to_string(),
            "MPAS-A v8.4.2 core_init_atmosphere/mpas_atm_advection.F:21-394::atm_initialize_advection_rk".to_string(),
        ),
        (
            "defc_a/b".to_string(),
            "MPAS v8.4.1 core_sw/mpas_sw_advection.F::sw_initialize_deformation_weights; consumed by gpuwm-hex Smagorinsky strain operator".to_string(),
        ),
    ]);

    let receipt = StaticBuildReceipt {
        schema: if cfg.stages.native_init { "rw-mpas-static.build/v4-native" } else { "rw-mpas-static.build/v3-unified" }.to_string(),
        grid_path: cfg.grid_path.display().to_string(),
        output_path: cfg.out_path.display().to_string(),
        n_cells: mesh.n_cells,
        n_edges: mesh.n_edges,
        n_vertices: mesh.n_vertices,
        max_edges: mesh.max_edges,
        grid_scaled_from_unit_sphere: mesh.scaled_from_unit_sphere,
        output_bytes,
        datasets,
        parallel: plan,
        tile_map_cache,
        memory,
        generated_fields,
        operator_provenance,
        provenance: cfg.provenance.clone(),
        stamped_provenance: provenance,
        gwd_seconds,
        gwd_band_rows,
        gwd_bands: gwd.bands,
        gwd_band_bytes: gwd.band_bytes,
        gwd_variance_smoothed_cells: gwd.smoothed_cells,
        regional: if mesh.is_regional() {
            let mut ring_cell_counts = [0usize; 8];
            for &m in &mesh.bdy_mask_cell {
                ring_cell_counts[m as usize] += 1;
            }
            Some(RegionalStaticNotes {
                ring_cell_counts,
                deriv_two_zero_stencil_cells: operators.deriv_two_zero_stencil_cells,
                gwd_smooth_skipped_boundary_cells: gwd.smooth_skipped_boundary_cells,
            })
        } else {
            None
        },
        fp32_metric_agreement,
        sha256: crate::sha256_file(&cfg.out_path)?,
        variables: if cfg.stages.native_init {
            std::iter::once("Time").chain(declared_variables().iter().copied())
                .filter(|name|native_static_field_enabled(name,&cfg.stages)).map(str::to_string).collect()
        } else {declared_variables().iter().map(|s| s.to_string()).collect()},
    };

    if let Some(path) = &cfg.receipt_path {
        let json = serde_json::to_vec_pretty(&receipt).map_err(|e| {
            MpasError::Refusal(format!("cannot serialize static receipt: {e}"))
        })?;
        std::fs::write(path, json)?;
    }
    Ok(receipt)
}


/// The sparse overlay the global vertical authority needs, without unrelated
/// land, soil, climatology or gravity-wave-drag geography. This is not a
/// runnable static. Each stored value is produced by the full static's own
/// terrain and operator paths at the full static's binary32 precision.
#[derive(Debug, serde::Serialize)]
pub struct TerrainBuildReceipt {
    pub schema: String,
    pub grid_path: String,
    pub grid_sha256: String,
    pub output_path: String,
    pub output_bytes: u64,
    pub sha256: String,
    pub datasets: Vec<DatasetReceipt>,
    pub parallel: TileParallelPlan,
    pub variables: Vec<String>,
}

fn write_terrain_overlay(
    cfg: &StaticBuildConfig,
    mesh: &Mesh,
    terrain: &[f32],
    operators: &OperatorFields,
) -> MpasResult<()> {
    if terrain.len() != mesh.n_cells {
        return Err(MpasError::Refusal("terrain overlay has the wrong cell count".to_string()));
    }
    let dims = vec![
        NcDim::fixed("nCells", mesh.n_cells),
        NcDim::fixed("nEdges", mesh.n_edges),
        NcDim::fixed("TWO", 2),
        NcDim::fixed("FIFTEEN", FIFTEEN),
        NcDim::fixed("nVertices", mesh.n_vertices),
        NcDim::fixed("vertexDegree", mesh.vertex_degree),
        NcDim::fixed("maxEdges2", mesh.max_edges2),
    ];
    let float = |name: &str, dimensions: Vec<usize>, units: &str| {
        NcVarDef::new(name, NcType::Float, dimensions)
            .with_attrs(vec![NcAttr::text("units", units)])
    };
    let mut variables = vec![
        float("ter", vec![0], "m"),
        float("areaCell", vec![0], "m^2"),
        float("dvEdge", vec![1], "m"),
        float("dcEdge", vec![1], "m"),
        float("deriv_two", vec![1, 2, 3], "unitless"),
    ];
    variables.extend([
        float("areaTriangle", vec![4], "m^2"),
        float("kiteAreasOnVertex", vec![4, 5], "m^2"),
        float("weightsOnEdge", vec![1, 6], "unitless"),
        float("nominalMinDc", vec![], "m"),
    ]);
    for (entity, dimension) in [("Cell", 0), ("Edge", 1), ("Vertex", 4)] {
        for axis in ["lat", "lon", "x", "y", "z"] {
            variables.push(NcVarDef::new(&format!("{axis}{entity}"), cfg.coordinates.nc_type(), vec![dimension]));
        }
    }
    let mut attrs = vec![
        NcAttr::text("rw_mpas_terrain_schema", "rw-mpas-terrain-overlay/v1"),
        NcAttr::text("rw_mpas_terrain_grid_sha256", crate::sha256_file(&cfg.grid_path)?),
        NcAttr::text("rw_mpas_terrain_role", "vertical-authority-overlay-not-runnable-static"),
    ];
    attrs.push(NcAttr::text("on_a_sphere", if mesh.on_a_sphere {"YES"} else {"NO"}));
    attrs.push(NcAttr::floats("sphere_radius", vec![mesh.sphere_radius as f32]));
    attrs.extend(cfg.coordinates.declaration_attributes(mesh.sphere_radius));
    let mut writer = NcClassicWriter::create(
        &cfg.out_path, NcFormat::Offset64, dims, attrs, variables, 0,
    )?;
    writer.put("ter", NcData::Floats(terrain))?;
    writer.put("areaCell", NcData::Floats(&f32_from_f64(&mesh.area_cell)))?;
    writer.put("dvEdge", NcData::Floats(&f32_from_f64(&mesh.dv_edge)))?;
    writer.put("dcEdge", NcData::Floats(&f32_from_f64(&mesh.dc_edge)))?;
    writer.put("deriv_two", NcData::Floats(&operators.deriv_two))?;
    writer.put("areaTriangle", NcData::Floats(&f32_from_f64(&mesh.area_triangle)))?;
    writer.put("kiteAreasOnVertex", NcData::Floats(&f32_from_f64(&mesh.kite_areas_on_vertex)))?;
    writer.put("weightsOnEdge", NcData::Floats(&f32_from_f64(&mesh.weights_on_edge)))?;
    writer.put("nominalMinDc", NcData::Floats(&[cfg.nominal_dx_m.unwrap_or(mesh.nominal_min_dc) as f32]))?;
    for (name, values) in [
        ("latCell", &mesh.lat_cell), ("lonCell", &mesh.lon_cell),
        ("xCell", &mesh.x_cell), ("yCell", &mesh.y_cell), ("zCell", &mesh.z_cell),
        ("latEdge", &mesh.lat_edge), ("lonEdge", &mesh.lon_edge),
        ("xEdge", &mesh.x_edge), ("yEdge", &mesh.y_edge), ("zEdge", &mesh.z_edge),
        ("latVertex", &mesh.lat_vertex), ("lonVertex", &mesh.lon_vertex),
        ("xVertex", &mesh.x_vertex), ("yVertex", &mesh.y_vertex), ("zVertex", &mesh.z_vertex),
    ] {
        if cfg.coordinates.is_binary64() {
            writer.put(name, NcData::Doubles(values))?;
        } else {
            writer.put(name, NcData::Floats(&f32_from_f64(values)))?;
        }
    }
    writer.finish()?;
    Ok(())
}

/// Build the full parent terrain and vertical operator overlay exactly,
/// while omitting every geography field the vertical authority never reads.
pub fn build_terrain_reporting(
    cfg: &StaticBuildConfig,
    progress: &dyn Fn(&str),
) -> MpasResult<TerrainBuildReceipt> {
    if cfg.out_path.exists() && !cfg.clobber {
        return Err(MpasError::Refusal(format!(
            "{} exists; pass --clobber to replace the terrain overlay", cfg.out_path.display()
        )));
    }
    if !cfg.geog.terrain.join("index").is_file() {
        return Err(MpasError::Refusal(format!(
            "terrain overlay needs the terrain dataset {}; the vertical authority cannot invent terrain",
            cfg.geog.terrain.display()
        )));
    }
    // Read at the precision this static will DECLARE; see `Mesh::read_as`.
    let mesh = Mesh::read_as(&cfg.grid_path, cfg.coordinates)?;
    progress(&format!("GRID\t{}\t{}\t{}\t{}", mesh.n_cells, mesh.n_edges, mesh.n_vertices, mesh.sphere_radius));
    let footprint = dataset_footprint(&cfg.geog.terrain, cfg.supersample, 1)?;
    // The same conservative mesh/operator resident term used by a full static.
    let resident = ((mesh.n_cells * (12 + 8 + 4) * 8)
        + (mesh.n_cells * mesh.max_edges * 4 * 4)
        + (mesh.n_edges * 2 * FIFTEEN * 4)
        + (mesh.n_edges * 32)
        + (mesh.n_vertices * 32)) as u64;
    let writer_slab = (mesh.n_edges * 2 * FIFTEEN * 4) as u64;
    let plan = plan_tile_parallelism(
        "terrain-overlay", resident, footprint.tile_bytes, writer_slab,
        1024 * 1024 * 1024, resolve_tile_workers(cfg.tile_workers),
        footprint.map_bytes.min(TILE_MAP_CACHE_CAP_BYTES), cfg.host_memory_limit_bytes,
    )?;
    let tree = KdTree::new(&mesh);
    let cache = TileMapCache::new(plan.tile_map_cache_bytes);
    let pool = if plan.workers > 1 {
        Some(rayon::ThreadPoolBuilder::new().stack_size(crate::execution::BRIDGE_STACK_BYTES).num_threads(plan.workers).build()
            .map_err(|e| MpasError::Refusal(format!("cannot start terrain tile workers: {e}")))?)
    } else { None };
    let ctx = GeogContext {
        mesh: &mesh, tree: &tree, max_d2: mesh.max_search_distance2(),
        workers: plan.workers, pool: pool.as_ref(), cache: &cache,
    };
    let started = Instant::now();
    take_centre_seeded();
    let (terrain, dataset) = continuous_mean(&ctx, &cfg.geog.terrain, cfg.supersample_30s, None, MeanPolicy::Terrain)?;
    let centre_seeded_cells = take_centre_seeded();
    if dataset.index.nz() != 1 {
        return Err(MpasError::Refusal("terrain source must have one plane".to_string()));
    }
    let elapsed = started.elapsed().as_secs_f64();
    progress(&format!("GEOGGRID\tter\t{}\t{}\t{}\t{elapsed:.2}",
        cfg.geog.terrain.display(), dataset.tiles.len(), cfg.supersample));
    drop(ctx);
    drop(pool);
    let operators = build_operator_fields(&mesh.operator_view())?;
    progress(&format!("OPERATORS\tderiv_two\t{}\t{}", operators.deriv_two.len(), FIFTEEN));
    write_terrain_overlay(cfg, &mesh, &terrain, &operators)?;
    let receipt = TerrainBuildReceipt {
        schema: "rw-mpas-terrain-overlay.build/v1".to_string(),
        grid_path: cfg.grid_path.display().to_string(),
        grid_sha256: crate::sha256_file(&cfg.grid_path)?,
        output_path: cfg.out_path.display().to_string(),
        output_bytes: std::fs::metadata(&cfg.out_path)?.len(),
        sha256: crate::sha256_file(&cfg.out_path)?,
        datasets: vec![DatasetReceipt {
            field: "ter".to_string(), path: cfg.geog.terrain.display().to_string(),
            tiles: dataset.tiles.len(), source_plane_bytes: dataset.plane_bytes(),
            supersample: cfg.supersample, destination_bytes: (terrain.len() * 4) as u64,
            elapsed_seconds: elapsed, spread_rounds: 0, centre_seeded_cells,
        }],
        parallel: plan,
        variables: ["ter", "areaCell", "dvEdge", "dcEdge", "deriv_two", "areaTriangle", "kiteAreasOnVertex", "weightsOnEdge", "nominalMinDc", "latCell", "lonCell", "xCell", "yCell", "zCell", "latEdge", "lonEdge", "xEdge", "yEdge", "zEdge", "latVertex", "lonVertex", "xVertex", "yVertex", "zVertex"].iter().map(|s| s.to_string()).collect(),
    };
    if let Some(path) = &cfg.receipt_path {
        std::fs::write(path, serde_json::to_vec_pretty(&receipt)
            .map_err(|e| MpasError::Refusal(format!("cannot serialize terrain receipt: {e}")))?)?;
    }
    Ok(receipt)
}

#[cfg(test)]
mod tests {
    /// The exact defect: a WRF-only archive (with-lakes MODIS land use, the
    /// soil container, no native land use and no USGS mask) gets ONE refusal
    /// that names both missing land-use datasets and the mesh fetch.
    #[test]
    fn one_refusal_names_every_missing_native_dataset() {
        let root = std::env::temp_dir().join(format!("rw-mpas-wrf-only-geog-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        for dir in ["topo_gmted2010_30s","modis_landuse_20class_30s_with_lakes","soilgrids/texture_top",
                    "greenfrac_fpar_modis","albedo_modis","maxsnowalb_modis","soiltemp_era5_1deg",
                    "soilgrids/soilcomp","soilgrids/texture_layer1","soilgrids/texture_layer2",
                    "soilgrids/texture_layer3","soilgrids/texture_layer4"] {
            std::fs::create_dir_all(root.join(dir)).unwrap();
            std::fs::write(root.join(dir).join("index"), "type = categorical\n").unwrap();
        }
        let mut cfg = StaticBuildConfig {
            grid_path: root.join("no-grid.nc"), out_path: root.join("out.nc"),
            geog: StaticGeogPaths::under(&root), stages: StaticStageOptions::default(),
            native_gwd: None, supersample: 1, supersample_landuse: 1, supersample_30s: 1,
            host_memory_limit_bytes: None, tile_workers: None, receipt_path: None,
            provenance: String::new(), valid_time: DEFAULT_VALID_TIME.into(), nominal_dx_m: None,
            clobber: false,
            coordinates: crate::staticfile::coordframe::CoordinateRepresentation::Binary32EarthCentred,
        };
        // What `rw_mpas_static --native-init --native-gwd-static` runs: the
        // native registry defaults, then the native GWD stage.
        crate::static_namelist::StaticNamelist::parse("").unwrap().apply(&mut cfg, &root).unwrap();
        let error = build_static(&cfg).unwrap_err().to_string();
        std::fs::remove_dir_all(&root).unwrap();
        assert!(error.contains("missing 2 dataset(s)"), "{error}");
        assert!(error.contains("modis_landuse_20class_30s,") || error.contains("modis_landuse_20class_30s "), "{error}");
        assert!(error.contains("landuse_30s"), "{error}");
        assert!(error.contains("gpuwm fetch-geog --datasets mesh"), "{error}");
    }

    use super::*;
    use crate::static_geog::fixture::{dataset, plane_1234_be, MINIMAL_INDEX, TILE_NAME};
    use crate::sha256_file;

    #[test]
    fn declared_lake_class_leaves_as_water_whatever_its_soil() {
        // With-lakes MODIS: water 17, lake 21, ice 15; STATSGO water soil 14.
        // The third cell is the defect's shape: a lake whose soil is loam (6).
        let mut landuse = [13, 17, 21, 15, 4, 21];
        let mut soil = [6, 2, 6, 14, 14, 14];
        assert_eq!(fold_declared_lake(&mut landuse, Some(21), 17), 2);
        let mask = crate::static_native::reconcile_surface(&mut landuse, &mut soil, 15, 17, 14);
        assert_eq!(landuse, [13, 17, 17, 15, 17, 17]);
        assert_eq!(soil, [6, 14, 14, 16, 14, 14]);
        assert_eq!(mask, [1, 0, 0, 1, 0, 0]);
        assert!(!landuse.contains(&21));
    }

    #[test]
    fn an_index_without_a_lake_class_folds_nothing() {
        // The native MODIS and USGS indexes declare no islake; the parser
        // stores -1. Category 21 is then not a lake and must not move.
        for islake in [None, Some(-1), Some(0)] {
            let mut landuse = [21, 4, 17];
            assert_eq!(fold_declared_lake(&mut landuse, islake, 17), 0);
            assert_eq!(landuse, [21, 4, 17]);
        }
    }

    // -- where a dataset directory is, in both layouts fetch-geog stages ----

    /// A throwaway WPS_GEOG root for the layout tests.
    fn layout_root(label: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "rw-mpas-geog-layout-{}-{}",
            std::process::id(),
            label
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("layout root");
        dir
    }

    /// A dataset directory that counts as present: it carries an `index`.
    fn stage(dir: &Path) {
        std::fs::create_dir_all(dir).expect("dataset directory");
        std::fs::write(dir.join("index"), "type=continuous\n").expect("index");
    }

    /// The container a directory is staged under, per the table.
    fn container_of(directory: &str) -> Option<&'static str> {
        GEOG_CONTAINER_TABLE
            .iter()
            .find(|(child, _)| *child == directory)
            .map(|(_, container)| *container)
    }

    /// The root laid out exactly as `gpuwm fetch-geog` stages it: single
    /// dataset archives flat, the soil group under its container.
    fn stage_as_fetched(root: &Path) {
        for (_, directory) in GEOG_DATASET_TABLE.iter() {
            match container_of(directory) {
                Some(container) => stage(&root.join(container).join(directory)),
                None => stage(&root.join(directory)),
            }
        }
    }

    #[test]
    fn a_flat_archive_resolves_every_dataset_at_the_root() {
        let root = layout_root("flat");
        for (_, directory) in GEOG_DATASET_TABLE.iter() {
            stage(&root.join(directory));
        }
        let paths = StaticGeogPaths::under(&root);
        for (slot, path) in paths.slots() {
            let directory = GEOG_DATASET_TABLE
                .iter()
                .find(|(s, _)| *s == slot)
                .expect("slot is a row")
                .1;
            assert_eq!(path, root.join(directory), "{slot}");
        }
        assert!(paths.missing().is_empty());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn the_layout_fetch_geog_stages_resolves_the_soil_group_under_its_container() {
        // The shipped defect: this exact tree, staged by the product's own
        // fetcher, was refused for five datasets it carried.
        let root = layout_root("fetched");
        stage_as_fetched(&root);
        let paths = StaticGeogPaths::under(&root);
        assert_eq!(paths.terrain, root.join("topo_gmted2010_30s"));
        assert_eq!(paths.soilcomp, root.join("soilgrids").join("soilcomp"));
        assert_eq!(paths.soilcl1, root.join("soilgrids").join("texture_layer1"));
        assert_eq!(paths.soilcl4, root.join("soilgrids").join("texture_layer4"));
        assert!(paths.missing().is_empty(), "{:?}", paths.missing());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_dataset_in_neither_place_is_named_by_its_own_directory() {
        let root = layout_root("short");
        for (_, directory) in GEOG_DATASET_TABLE.iter() {
            if container_of(directory).is_none() {
                stage(&root.join(directory));
            }
        }
        // An empty container directory (an interrupted extraction) is not a
        // dataset either.
        std::fs::create_dir_all(root.join("soilgrids").join("soilcomp")).unwrap();
        let paths = StaticGeogPaths::under(&root);
        assert_eq!(
            paths.missing(),
            vec![
                // The default soil category (2026-10-21) is a container child too.
                "texture_top".to_string(),
                "soilcomp".to_string(),
                "texture_layer1".to_string(),
                "texture_layer2".to_string(),
                "texture_layer3".to_string(),
                "texture_layer4".to_string(),
            ]
        );
        // The unresolved slot keeps the flat path so the refusal names the
        // directory, never a container it was not found in.
        assert_eq!(paths.soilcomp, root.join("soilcomp"));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn the_flat_path_wins_when_both_layouts_carry_a_dataset() {
        let root = layout_root("both");
        stage_as_fetched(&root);
        stage(&root.join("soilcomp"));
        assert_eq!(resolve_geog_dataset_dir(&root, "soilcomp"), root.join("soilcomp"));
        assert_eq!(
            resolve_geog_dataset_dir(&root, "texture_layer2"),
            root.join("soilgrids").join("texture_layer2")
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn every_soil_directory_the_builder_reads_has_a_container_row() {
        // The five soil slots are the ones the fetcher stages nested; a slot
        // added to GEOG_DATASET_TABLE from that archive without a row here
        // would reproduce the refusal for that one directory.
        for slot in ["soilcomp", "soilcl1", "soilcl2", "soilcl3", "soilcl4"] {
            let directory = GEOG_DATASET_TABLE
                .iter()
                .find(|(s, _)| *s == slot)
                .expect("slot is a row")
                .1;
            assert!(container_of(directory).is_some(), "{directory} has no container row");
        }
        // And every container row names a directory, not a slot.
        for (child, container) in GEOG_CONTAINER_TABLE.iter() {
            assert!(!child.is_empty() && !container.is_empty());
            assert_ne!(child, container);
        }
    }

    // -- the regional load path, graded on bytes this module writes ----------

    /// One tiny grid file with configurable boundary masks and sentinel
    /// placement, through the same classic writer the crate ships. Geometry
    /// is unit-sphere garbage on purpose: what is under test is the sentinel
    /// ADMISSION, which reads topology and masks only.
    fn tiny_grid_nc(
        label: &str,
        bdy_cell: Option<[i32; 4]>,
        bdy_edge: [i32; 4],
        bdy_vertex: [i32; 4],
        cells_on_cell: [[i32; 3]; 4],
        cells_on_edge: [[i32; 2]; 4],
        vertices_on_cell: [[i32; 3]; 4],
        edges_on_edge_rows: [[i32; 6]; 4],
    ) -> PathBuf {
        use rw_store::netcdf_classic::{NcClassicWriter, NcData, NcDim, NcFormat, NcVarDef, NcType};
        let path = std::env::temp_dir().join(format!(
            "rw-mpas-regional-admission-{}-{label}.nc",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&path);
        let dims = vec![
            NcDim::fixed("nCells", 4),
            NcDim::fixed("nEdges", 4),
            NcDim::fixed("nVertices", 4),
            NcDim::fixed("maxEdges", 3),
            NcDim::fixed("maxEdges2", 6),
            NcDim::fixed("TWO", 2),
            NcDim::fixed("vertexDegree", 3),
        ];
        let (c, e, v, me, me2, two, vd) = (0usize, 1, 2, 3, 4, 5, 6);
        let mut vars = vec![
            NcVarDef::new("latCell", NcType::Double, vec![c]),
            NcVarDef::new("lonCell", NcType::Double, vec![c]),
            NcVarDef::new("latEdge", NcType::Double, vec![e]),
            NcVarDef::new("lonEdge", NcType::Double, vec![e]),
            NcVarDef::new("latVertex", NcType::Double, vec![v]),
            NcVarDef::new("lonVertex", NcType::Double, vec![v]),
            NcVarDef::new("xCell", NcType::Double, vec![c]),
            NcVarDef::new("yCell", NcType::Double, vec![c]),
            NcVarDef::new("zCell", NcType::Double, vec![c]),
            NcVarDef::new("xEdge", NcType::Double, vec![e]),
            NcVarDef::new("yEdge", NcType::Double, vec![e]),
            NcVarDef::new("zEdge", NcType::Double, vec![e]),
            NcVarDef::new("xVertex", NcType::Double, vec![v]),
            NcVarDef::new("yVertex", NcType::Double, vec![v]),
            NcVarDef::new("zVertex", NcType::Double, vec![v]),
            NcVarDef::new("dcEdge", NcType::Double, vec![e]),
            NcVarDef::new("dvEdge", NcType::Double, vec![e]),
            NcVarDef::new("areaCell", NcType::Double, vec![c]),
            NcVarDef::new("areaTriangle", NcType::Double, vec![v]),
            NcVarDef::new("kiteAreasOnVertex", NcType::Double, vec![v, vd]),
            NcVarDef::new("nominalMinDc", NcType::Double, vec![]),
            NcVarDef::new("angleEdge", NcType::Double, vec![e]),
            NcVarDef::new("meshDensity", NcType::Double, vec![c]),
            NcVarDef::new("weightsOnEdge", NcType::Double, vec![e, me2]),
            NcVarDef::new("nEdgesOnCell", NcType::Int, vec![c]),
            NcVarDef::new("nEdgesOnEdge", NcType::Int, vec![e]),
            NcVarDef::new("cellsOnCell", NcType::Int, vec![c, me]),
            NcVarDef::new("edgesOnCell", NcType::Int, vec![c, me]),
            NcVarDef::new("verticesOnCell", NcType::Int, vec![c, me]),
            NcVarDef::new("cellsOnEdge", NcType::Int, vec![e, two]),
            NcVarDef::new("verticesOnEdge", NcType::Int, vec![e, two]),
            NcVarDef::new("edgesOnEdge", NcType::Int, vec![e, me2]),
            NcVarDef::new("cellsOnVertex", NcType::Int, vec![v, vd]),
            NcVarDef::new("edgesOnVertex", NcType::Int, vec![v, vd]),
        ];
        if bdy_cell.is_some() {
            vars.push(NcVarDef::new("bdyMaskCell", NcType::Int, vec![c]));
            vars.push(NcVarDef::new("bdyMaskEdge", NcType::Int, vec![e]));
            vars.push(NcVarDef::new("bdyMaskVertex", NcType::Int, vec![v]));
        }
        let mut w =
            NcClassicWriter::create(&path, NcFormat::Offset64, dims, Vec::new(), vars, 0).unwrap();
        // Unit-sphere-ish geometry: four distinct points, nothing degenerate.
        let ang = [0.1f64, 0.7, 1.3, 1.9];
        let sc: Vec<f64> = ang.iter().map(|a| a.cos()).collect();
        let ss: Vec<f64> = ang.iter().map(|a| a.sin()).collect();
        let zz = vec![0.0f64; 4];
        for name in ["latCell", "latEdge", "latVertex"] {
            w.put(name, NcData::Doubles(&zz)).unwrap();
        }
        for name in ["lonCell", "lonEdge", "lonVertex"] {
            w.put(name, NcData::Doubles(&ang)).unwrap();
        }
        for name in ["xCell", "xEdge", "xVertex"] {
            w.put(name, NcData::Doubles(&sc)).unwrap();
        }
        for name in ["yCell", "yEdge", "yVertex"] {
            w.put(name, NcData::Doubles(&ss)).unwrap();
        }
        for name in ["zCell", "zEdge", "zVertex"] {
            w.put(name, NcData::Doubles(&zz)).unwrap();
        }
        for name in ["dcEdge", "dvEdge"] {
            w.put(name, NcData::Doubles(&[0.1; 4])).unwrap();
        }
        for name in ["areaCell", "areaTriangle", "meshDensity"] {
            w.put(name, NcData::Doubles(&[1.0; 4])).unwrap();
        }
        w.put("kiteAreasOnVertex", NcData::Doubles(&[1.0; 12])).unwrap();
        w.put("nominalMinDc", NcData::Doubles(&[0.1])).unwrap();
        w.put("angleEdge", NcData::Doubles(&[0.0; 4])).unwrap();
        w.put("weightsOnEdge", NcData::Doubles(&[0.0; 24])).unwrap();
        w.put("nEdgesOnCell", NcData::Ints(&[3; 4])).unwrap();
        w.put("nEdgesOnEdge", NcData::Ints(&[2; 4])).unwrap();
        let flat3 = |rows: [[i32; 3]; 4]| rows.concat();
        let flat2 = |rows: [[i32; 2]; 4]| rows.concat();
        w.put("cellsOnCell", NcData::Ints(&flat3(cells_on_cell))).unwrap();
        w.put("edgesOnCell", NcData::Ints(&flat3([[1, 2, 3], [2, 3, 4], [3, 4, 1], [4, 1, 2]])))
            .unwrap();
        w.put("verticesOnCell", NcData::Ints(&flat3(vertices_on_cell))).unwrap();
        w.put("cellsOnEdge", NcData::Ints(&flat2(cells_on_edge))).unwrap();
        w.put("verticesOnEdge", NcData::Ints(&flat2([[1, 2], [2, 3], [3, 4], [4, 1]])))
            .unwrap();
        w.put("edgesOnEdge", NcData::Ints(&edges_on_edge_rows.concat())).unwrap();
        w.put("cellsOnVertex", NcData::Ints(&flat3([[1, 2, 3], [2, 3, 4], [3, 4, 1], [4, 1, 2]])))
            .unwrap();
        w.put("edgesOnVertex", NcData::Ints(&flat3([[1, 2, 3], [2, 3, 4], [3, 4, 1], [4, 1, 2]])))
            .unwrap();
        if let Some(bc) = bdy_cell {
            w.put("bdyMaskCell", NcData::Ints(&bc)).unwrap();
            w.put("bdyMaskEdge", NcData::Ints(&bdy_edge)).unwrap();
            w.put("bdyMaskVertex", NcData::Ints(&bdy_vertex)).unwrap();
        }
        w.finish().unwrap();
        path
    }

    /// The clean regional layout every corruption below is one delta from:
    /// cells 2 and 3 are outermost (mask 7) and carry one absent neighbour
    /// each; edges 2 and 3 are one-sided; vertices 2 and 3 are one-sided.
    fn clean_regional() -> PathBuf {
        tiny_grid_nc(
            "clean",
            Some([5, 6, 7, 7]),
            [5, 6, 7, 7],
            [0, 0, 7, 7],
            [[2, 3, 4], [1, 3, 4], [1, 2, 0], [1, 2, 0]],
            [[1, 2], [2, 3], [3, 0], [4, 0]],
            [[1, 2, 3], [2, 3, 4], [3, 4, 1], [4, 1, 2]],
            [
                [1, 2, 0, 0, 0, 0],
                [2, 3, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0],
                [0, 2, 0, 0, 0, 0],
            ],
        )
    }

    /// A culler-shaped regional grid loads: sentinels at mask-7 only.
    #[test]
    fn a_regional_grid_with_mask7_sentinels_is_admitted() {
        let path = clean_regional();
        let probe = probe_grid(&path).expect("the clean regional layout admits");
        assert!(probe.regional);
        assert_eq!(probe.ring_cell_counts, [0, 0, 0, 0, 0, 1, 1, 2]);
        assert_eq!(probe.absent_neighbor_slots, 2);
        assert_eq!(probe.one_sided_edges, 2);
        let _ = std::fs::remove_file(&path);
    }

    /// An absent neighbour on a NON-outermost cell is a torn mesh, refused
    /// with the rule named.
    #[test]
    fn a_sentinel_inside_the_relaxation_rings_is_refused() {
        let path = tiny_grid_nc(
            "torn-cell",
            Some([5, 6, 3, 7]),
            [5, 6, 7, 7],
            [0, 0, 7, 7],
            [[2, 3, 4], [1, 3, 4], [1, 2, 0], [1, 2, 0]],
            [[1, 2], [2, 3], [3, 0], [4, 0]],
            [[1, 2, 3], [2, 3, 4], [3, 4, 1], [4, 1, 2]],
            [
                [1, 2, 0, 0, 0, 0],
                [2, 3, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0],
                [0, 2, 0, 0, 0, 0],
            ],
        );
        let err = probe_grid(&path).unwrap_err().to_string();
        assert!(err.contains("only outermost regional cells"), "{err}");
        let _ = std::fs::remove_file(&path);
    }

    /// A one-sided edge below mask 7 is refused with the rule named.
    #[test]
    fn a_one_sided_edge_inside_the_rings_is_refused() {
        let path = tiny_grid_nc(
            "torn-edge",
            Some([5, 6, 7, 7]),
            [5, 6, 2, 7],
            [0, 0, 7, 7],
            [[2, 3, 4], [1, 3, 4], [1, 2, 0], [1, 2, 0]],
            [[1, 2], [2, 3], [3, 0], [4, 0]],
            [[1, 2, 3], [2, 3, 4], [3, 4, 1], [4, 1, 2]],
            [
                [1, 2, 0, 0, 0, 0],
                [2, 3, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0],
                [0, 2, 0, 0, 0, 0],
            ],
        );
        let err = probe_grid(&path).unwrap_err().to_string();
        assert!(err.contains("one-sided"), "{err}");
        let _ = std::fs::remove_file(&path);
    }

    /// A mask value past the seven rings is refused with its location.
    #[test]
    fn a_mask_outside_the_seven_rings_is_refused() {
        let path = tiny_grid_nc(
            "bad-mask",
            Some([5, 9, 7, 7]),
            [5, 6, 7, 7],
            [0, 0, 7, 7],
            [[2, 3, 4], [1, 3, 4], [1, 2, 0], [1, 2, 0]],
            [[1, 2], [2, 3], [3, 0], [4, 0]],
            [[1, 2, 3], [2, 3, 4], [3, 4, 1], [4, 1, 2]],
            [
                [1, 2, 0, 0, 0, 0],
                [2, 3, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0],
                [0, 2, 0, 0, 0, 0],
            ],
        );
        let err = probe_grid(&path).unwrap_err().to_string();
        assert!(err.contains("outside 0..=7"), "{err}");
        let _ = std::fs::remove_file(&path);
    }

    /// A zero in verticesOnCell's VALID slots can never be a boundary: the
    /// cull keeps every vertex of a kept cell.
    #[test]
    fn a_missing_own_vertex_is_refused_on_a_regional_grid() {
        let path = tiny_grid_nc(
            "torn-vertex",
            Some([5, 6, 7, 7]),
            [5, 6, 7, 7],
            [0, 0, 7, 7],
            [[2, 3, 4], [1, 3, 4], [1, 2, 0], [1, 2, 0]],
            [[1, 2], [2, 3], [3, 0], [4, 0]],
            [[1, 2, 3], [2, 3, 4], [3, 4, 0], [4, 1, 2]],
            [
                [1, 2, 0, 0, 0, 0],
                [2, 3, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0],
                [0, 2, 0, 0, 0, 0],
            ],
        );
        let err = probe_grid(&path).unwrap_err().to_string();
        assert!(
            err.contains("verticesOnCell") && err.contains("outside 1..="),
            "{err}"
        );
        let _ = std::fs::remove_file(&path);
    }

    /// The mpas_in_cell port, on real spherical geometry: a triangle cell
    /// around the north pole. The edge through two vertices at colatitude
    /// 0.1 rad with 120 degrees of longitude between them passes closest to
    /// the pole at colatitude atan(tan(0.1)*cos(60 deg)) ~ 0.0501 rad.
    #[test]
    fn point_in_cell_matches_the_voronoi_edges_of_a_polar_triangle() {
        let r = MPAS_EARTH_RADIUS_M;
        let mut mesh = synthetic_mesh(4);
        let colat = 0.1f64;
        mesh.x_cell[0] = 0.0;
        mesh.y_cell[0] = 0.0;
        mesh.z_cell[0] = r;
        mesh.n_edges_on_cell[0] = 3;
        for (k, lon_deg) in [0.0f64, 120.0, 240.0].iter().enumerate() {
            let lon = lon_deg.to_radians();
            mesh.x_vertex[k] = r * colat.sin() * lon.cos();
            mesh.y_vertex[k] = r * colat.sin() * lon.sin();
            mesh.z_vertex[k] = r * colat.cos();
            mesh.vertices_on_cell[k] = k; // slots 0..3 of cell 0
        }
        let at = |colat: f64, lon_deg: f64| -> [f64; 3] {
            let lon = lon_deg.to_radians();
            [
                r * colat.sin() * lon.cos(),
                r * colat.sin() * lon.sin(),
                r * colat.cos(),
            ]
        };
        assert!(point_in_cell(&mesh, 0, at(0.0, 0.0)), "the generating point itself");
        assert!(point_in_cell(&mesh, 0, at(0.03, 77.0)), "deep inside");
        assert!(point_in_cell(&mesh, 0, at(0.049, 60.0)), "just inside the far edge");
        assert!(!point_in_cell(&mesh, 0, at(0.08, 60.0)), "just past the far edge");
        assert!(!point_in_cell(&mesh, 0, at(0.3, 200.0)), "far outside");
    }

    // -- a synthetic case, built from bytes this module owns -----------------

    const NX_TILES: i64 = 6;
    const NY_TILES: i64 = 3;
    const TILE: i64 = 30;

    /// A multi-tile WPS_GEOG tree covering the whole globe at 2 degrees.
    ///
    /// Every value varies with its own global coordinate, so a destination
    /// cell accumulates a different number from every tile that reaches it.
    /// That is what makes the order tiles are visited in observable at all.
    fn geog_fixture(label: &str, kind: &str, nz: usize, wordsize: usize) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "rw-mpas-static-{}-{}",
            std::process::id(),
            label
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("fixture directory");
        let mut index = format!(
            "type={kind}\nprojection=regular_ll\ndx=2.0\ndy=2.0\n\
             known_x=1.0\nknown_y=1.0\nknown_lat=-89.0\nknown_lon=-179.0\n\
             wordsize={wordsize}\ntile_x={TILE}\ntile_y={TILE}\ntile_z={nz}\n\
             tile_bdr=0\nendian=big\nsigned=no\nscale_factor=1.0\n"
        );
        if kind == "categorical" {
            index.push_str("category_min=1\ncategory_max=8\niswater=3\nmminlu=FIXTURE\n");
        }
        std::fs::write(dir.join("index"), &index).expect("fixture index");
        for ty in 0..NY_TILES {
            for tx in 0..NX_TILES {
                let xs = tx * TILE + 1;
                let ys = ty * TILE + 1;
                let mut bytes = Vec::with_capacity(nz * (TILE * TILE) as usize * wordsize);
                for z in 0..nz as i64 {
                    for j in 0..TILE {
                        for i in 0..TILE {
                            let x = xs + i;
                            let y = ys + j;
                            let v = if kind == "categorical" {
                                (x * 7 + y * 13).rem_euclid(8) + 1
                            } else {
                                (x * 31 + y * 17 + z * 101).rem_euclid(4096)
                            };
                            match wordsize {
                                1 => bytes.push(v as u8),
                                _ => bytes.extend_from_slice(&(v as u16).to_be_bytes()),
                            }
                        }
                    }
                }
                let name = format!("{xs:05}-{:05}.{ys:05}-{:05}", xs + TILE - 1, ys + TILE - 1);
                std::fs::write(dir.join(name), &bytes).expect("fixture tile");
            }
        }
        dir
    }

    /// A mesh whose cells cover a latitude band, not the whole sphere.
    ///
    /// Deliberately a band: the tile mapper refuses a source pixel further
    /// than the search radius from any cell, so a global geography source
    /// over a band mesh produces destination maps with refused entries in
    /// them.  A mesh covering the sphere would leave that branch of the
    /// accumulate loop untested by the determinism gate.
    ///
    /// Only the geometry the geography passes read has to be physical: the
    /// topology is filled with in-range indices so the writer can serialize
    /// it, because what is under test is which pixels reach which cell.
    fn synthetic_mesh(n_cells: usize) -> Mesh {
        let r = MPAS_EARTH_RADIUS_M;
        let max_edges = 3usize;
        let golden = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
        let mut lat_cell = Vec::with_capacity(n_cells);
        let mut lon_cell = Vec::with_capacity(n_cells);
        for c in 0..n_cells {
            let z = 0.5 - (c as f64 + 0.5) / n_cells as f64;
            lat_cell.push(z.asin());
            lon_cell.push((golden * c as f64).rem_euclid(std::f64::consts::TAU) - std::f64::consts::PI);
        }
        let xyz_of = |lat: &[f64], lon: &[f64]| -> (Vec<f64>, Vec<f64>, Vec<f64>) {
            let mut x = Vec::with_capacity(lat.len());
            let mut y = Vec::with_capacity(lat.len());
            let mut z = Vec::with_capacity(lat.len());
            for (la, lo) in lat.iter().zip(lon) {
                let p = xyz(r, *la, *lo);
                x.push(p[0]);
                y.push(p[1]);
                z.push(p[2]);
            }
            (x, y, z)
        };
        // Vertices sit a fixed arc from their cell, which sets the search
        // radius the tile mapper refuses beyond.  Small enough that a polar
        // source pixel is outside every cell's reach.
        let lat_vertex: Vec<f64> = lat_cell
            .iter()
            .map(|v| (v + 0.2).clamp(-1.5, 1.5))
            .collect();
        let lon_vertex = lon_cell.clone();
        let (x_cell, y_cell, z_cell) = xyz_of(&lat_cell, &lon_cell);
        let (x_vertex, y_vertex, z_vertex) = xyz_of(&lat_vertex, &lon_vertex);
        let n_edges = n_cells;
        let n_vertices = n_cells;
        let max_edges2 = 2 * max_edges;
        let vertex_degree = 3usize;
        let cells_on_cell: Vec<usize> = (0..n_cells * max_edges).map(|k| k % n_cells).collect();
        let edges_on_cell: Vec<usize> = (0..n_cells * max_edges).map(|k| k % n_edges).collect();
        let vertices_on_cell: Vec<usize> =
            (0..n_cells * max_edges).map(|k| k / max_edges).collect();
        let one_based = |v: &[usize]| -> Vec<i32> { v.iter().map(|&k| k as i32 + 1).collect() };
        Mesh {
            n_cells,
            n_edges,
            n_vertices,
            max_edges,
            max_edges2,
            vertex_degree,
            raw_cells_on_cell: one_based(&cells_on_cell),
            raw_edges_on_cell: one_based(&edges_on_cell),
            raw_vertices_on_cell: one_based(&vertices_on_cell),
            raw_cells_on_edge: (0..n_edges)
                .flat_map(|e| [(e % n_cells) as i32 + 1, ((e + 1) % n_cells) as i32 + 1])
                .collect(),
            raw_vertices_on_edge: (0..n_edges)
                .flat_map(|e| [(e % n_vertices) as i32 + 1, ((e + 1) % n_vertices) as i32 + 1])
                .collect(),
            raw_edges_on_edge: (0..n_edges * max_edges2)
                .map(|k| (k % n_edges) as i32 + 1)
                .collect(),
            raw_cells_on_vertex: (0..n_vertices * vertex_degree)
                .map(|k| (k % n_cells) as i32 + 1)
                .collect(),
            raw_edges_on_vertex: (0..n_vertices * vertex_degree)
                .map(|k| (k % n_edges) as i32 + 1)
                .collect(),
            n_edges_on_edge: vec![max_edges2 as i32; n_edges],
            weights_on_edge: vec![0.0; n_edges * max_edges2],
            kite_areas_on_vertex: vec![1.0; n_vertices * vertex_degree],
            index_to_cell_id: (1..=n_cells as i32).collect(),
            index_to_edge_id: (1..=n_edges as i32).collect(),
            index_to_vertex_id: (1..=n_vertices as i32).collect(),
            bdy_mask_edge: vec![0; n_edges],
            bdy_mask_vertex: vec![0; n_vertices],
            scaled_from_unit_sphere: false,
            on_a_sphere: true,
            sphere_radius: r,
            lat_edge: lat_cell.clone(),
            lon_edge: lon_cell.clone(),
            x_edge: x_cell.clone(),
            y_edge: y_cell.clone(),
            z_edge: z_cell.clone(),
            lat_vertex,
            lon_vertex,
            x_vertex,
            y_vertex,
            z_vertex,
            dc_edge: vec![1.0; n_edges],
            dv_edge: vec![1.0; n_edges],
            area_cell: vec![1.0; n_cells],
            area_triangle: vec![1.0; n_vertices],
            angle_edge: vec![0.0; n_edges],
            mesh_density: vec![1.0; n_cells],
            nominal_min_dc: 1.0,
            n_edges_on_cell: vec![max_edges; n_cells],
            cells_on_cell,
            edges_on_cell,
            vertices_on_cell,
            cells_on_edge: (0..n_edges).map(|e| [e % n_cells, (e + 1) % n_cells]).collect(),
            vertices_on_edge: (0..n_edges)
                .map(|e| [e % n_vertices, (e + 1) % n_vertices])
                .collect(),
            bdy_mask_cell: vec![0; n_cells],
            lat_cell,
            lon_cell,
            x_cell,
            y_cell,
            z_cell,
        }
    }

    /// A drag band of the right shape and no content, for the writer tests.
    ///
    /// The sub-grid orography has its own case in `static_gwd`; what the
    /// writer tests measure is byte-identity of the file it lays out, and a
    /// constant band keeps that measurement about the writer.
    fn flat_gwd(mesh: &Mesh) -> crate::static_gwd::GwdFields {
        crate::static_gwd::GwdFields {
            var2d: vec![0.0; mesh.n_cells],
            con: vec![0.0; mesh.n_cells],
            oa: std::array::from_fn(|_| vec![0.0; mesh.n_cells]),
            ol: std::array::from_fn(|_| vec![0.0; mesh.n_cells]),
            hlanduse: vec![1; mesh.n_cells],
            smoothed_cells: 0,
            smooth_skipped_boundary_cells: 0,
            water_category: 3,
            water_category_source: "fixture".to_string(),
            band_rows: 0,
            bands: 0,
            band_bytes: 0,
        }
    }

    fn fixture_geography(
        ter: Vec<f32>,
        landmask: Vec<i32>,
        ivgtyp: Vec<i32>,
        isltyp: Vec<i32>,
        greenfrac: Vec<f32>,
        albedo12m: Vec<f32>,
        snoalb: Vec<f32>,
        soiltemp: Vec<f32>,
        soilcomp: Vec<f32>,
        soilcl: [Vec<i32>; 4],
        iswater_lu: i32,
    ) -> GeographyFields {
        GeographyFields {
            tc_lakemask: None,
            ter,
            landmask,
            ivgtyp,
            isltyp,
            greenfrac,
            albedo12m,
            snoalb,
            soiltemp,
            soilcomp,
            soilcl,
            isice_lu: 15,
            iswater_lu,
            mminlu: "FIXTURE".to_string(),
        }
    }

    fn zero_operators(mesh: &Mesh) -> OperatorFields {
        let per_cell = mesh.n_cells * mesh.max_edges;
        OperatorFields {
            cell_gradient_coef_x: vec![0.0; per_cell],
            cell_gradient_coef_y: vec![0.0; per_cell],
            defc_a: vec![0.0; per_cell],
            defc_b: vec![0.0; per_cell],
            deriv_two: vec![0.0; mesh.n_edges * 2 * FIFTEEN],
            deriv_two_zero_stencil_cells: 0,
        }
    }

    /// Pristine static driver accumulation and mean blocks, no tolerance.
    #[test]
    fn signed_native_accumulators_and_means_match_pristine_words() {
        let mut rows=0;
        for line in include_str!("../tests/native_static/counts.words").lines() {
            let words:Vec<_>=line.split_whitespace().collect();
            assert_eq!(words.len(),9);
            let i32word=|k:usize|u32::from_str_radix(words[k],16).unwrap() as i32;
            let i64word=|k:usize|u64::from_str_radix(words[k],16).unwrap() as i64;
            let count=[AtomicI32::new(i32word(0))];
            let sum=[AtomicI64::new(i64word(3))];
            flush_i32(&count,0,1);flush_i64(&sum,0,7);
            let n=count[0].load(Ordering::Relaxed);
            let s=sum[0].load(Ordering::Relaxed);
            assert_eq!(n,i32word(1));assert_eq!(n,i32word(2));
            assert_eq!(s,i64word(4));assert_eq!(s,i64word(5));
            let scale=f32::from_bits(i32word(6) as u32);
            assert_eq!(crate::static_native::scaled_mean(s,n,scale).to_bits(),i32word(7) as u32);
            assert_eq!(crate::static_native::soilcomp_mean(s,n,scale).to_bits(),i32word(8) as u32);
            rows+=1;
        }
        assert_eq!(rows,128);
    }

    #[test]
    fn signed_native_category_maxloc_matches_pristine_words() {
        let mut rows=0;
        for line in include_str!("../tests/native_static/categories.words").lines() {
            if let Some(empty)=line.strip_prefix("EMPTY ") {
                let fields:Vec<i32>=empty.split_whitespace().map(|v|v.parse().unwrap()).collect();
                assert!(fields[1]<fields[0]);
                assert_eq!(crate::static_native::dominant_category(&[]),None);
                assert_eq!(fields[0]-1,fields[2]);
                rows+=1;
                continue;
            }
            let words:Vec<i32>=line.split_whitespace().map(|v|u32::from_str_radix(v,16).unwrap() as i32).collect();
            assert_eq!(words.len(),10);
            let counts:Vec<_>=words[..4].iter().map(|&v|AtomicI32::new(v)).collect();
            if (1..=4).contains(&words[4]) {flush_i32(&counts,(words[4]-1) as usize,1);}
            let counts:Vec<_>=counts.iter().map(|v|v.load(Ordering::Relaxed)).collect();
            assert_eq!(counts,words[5..9]);
            assert_eq!(crate::static_native::dominant_category(&counts).unwrap() as i32+1,words[9]);
            rows+=1;
        }
        assert_eq!(rows,49);
    }

    /// Complete pristine atm_advection module, real grid, no tolerance.
    #[test]
    #[ignore = "requires an owned strict native operator dump and its input grid"]
    fn static_operators_match_pristine_real_grid_words() {
        let grid=std::env::var("MPAS_STATIC_ORACLE_GRID").expect("native oracle grid");
        let dump=std::env::var("MPAS_STATIC_ORACLE_OPERATORS").expect("native oracle dump");
        let mut mesh=Mesh::read(Path::new(&grid)).unwrap();
        if let Ok(padding)=std::env::var("MPAS_STATIC_ORACLE_PADDING") {
            let padding:usize=padding.parse().unwrap();
            assert!(padding>=mesh.max_edges);
            let pad=|values:&[usize]| {
                let mut rows=vec![0;mesh.n_cells*padding];
                for c in 0..mesh.n_cells {
                    rows[c*padding..c*padding+mesh.max_edges]
                        .copy_from_slice(&values[c*mesh.max_edges..(c+1)*mesh.max_edges]);
                }
                rows
            };
            mesh.cells_on_cell=pad(&mesh.cells_on_cell);
            mesh.edges_on_cell=pad(&mesh.edges_on_cell);
            mesh.vertices_on_cell=pad(&mesh.vertices_on_cell);
            mesh.max_edges=padding;
        }
        let fields=build_operator_fields(&mesh.operator_view()).unwrap();
        let bytes=std::fs::read(dump).unwrap();
        let expected:Vec<u32>=bytes.chunks_exact(4).map(|v|u32::from_le_bytes(v.try_into().unwrap())).collect();
        let actual:Vec<u32>=fields.cell_gradient_coef_x.iter().chain(&fields.cell_gradient_coef_y)
            .chain(&fields.deriv_two).map(|v|v.to_bits()).collect();
        assert_eq!(actual.len(),expected.len());
        let different=actual.iter().zip(&expected).filter(|(a,b)|a!=b).count();
        let examples:Vec<_>=actual.iter().zip(&expected).enumerate().filter(|(_, (a,b))|a!=b).take(12).collect();
        println!("native operator words={}, differing={different}",actual.len());
        assert_eq!(different,0,"native operator words differ: {examples:?}");
    }

    #[test]
    #[ignore = "requires owned synthetic native climatology files from regenerate.py"]
    fn static_climatology_matches_pristine_driver_words() {
        let root=std::path::PathBuf::from(std::env::var("MPAS_STATIC_ORACLE_CLIMATOLOGY").unwrap());
        let rows:Vec<Vec<&str>>=include_str!("../tests/native_static/climatology.words")
            .lines().map(|line|line.split_whitespace().collect()).collect();
        let mut mesh=synthetic_mesh(rows.len());
        let bits=|s:&str|u32::from_str_radix(s,16).unwrap();
        mesh.lat_cell=rows.iter().map(|r|f32::from_bits(bits(r[1])) as f64).collect();
        mesh.lon_cell=rows.iter().map(|r|f32::from_bits(bits(r[2])) as f64).collect();
        let mask:Vec<i32>=rows.iter().map(|r|r[3].parse().unwrap()).collect();
        let (green,_)=direct_climatology(&mesh,&root.join("greenfrac"),&mask,-1.0e30,0.0,1.0,true).unwrap();
        let (albedo,_)=direct_climatology(&mesh,&root.join("albedo_ncep"),&mask,0.0,8.0,1.0,true).unwrap();
        let (snow,_)=direct_climatology(&mesh,&root.join("maxsnowalb"),&mask,0.0,0.0,100.0,true).unwrap();
        let (temp,_)=soil_temperature_dataset(&mesh,
            GeogDataset::open_native_soil_temperature(&root.join("soiltemp_1deg")).unwrap(),&mask).unwrap();
        let mut different=0;
        let mut examples=Vec::new();
        for (c,row) in rows.iter().enumerate() {
            let g=&green[c*12..(c+1)*12];
            let values:Vec<f32>=g.iter().chain(&albedo[c*12..(c+1)*12]).copied()
                .chain([snow[c],temp[c],g.iter().copied().fold(f32::INFINITY,f32::min),
                    g.iter().copied().fold(f32::NEG_INFINITY,f32::max)]).collect();
            for (field,(&actual,expected)) in values.iter().zip(&row[4..]).enumerate() {
                if actual.to_bits()!=bits(expected) {
                    different+=1;
                    if examples.len()<12 {examples.push((c,field,actual.to_bits(),bits(expected)));}
                }
            }
        }
        println!("native climatology words={}, differing={different}",rows.len()*28);
        assert_eq!(different,0,"native climatology differences: {examples:?}");
    }

    struct SyntheticCase {
        cont1: PathBuf,
        cont12: PathBuf,
        cat: PathBuf,
        comp: PathBuf,
    }

    impl SyntheticCase {
        fn build(tag: &str) -> Self {
            Self {
                cont1: geog_fixture(&format!("{tag}-cont1"), "continuous", 1, 2),
                cont12: geog_fixture(&format!("{tag}-cont12"), "continuous", 12, 2),
                cat: geog_fixture(&format!("{tag}-cat"), "categorical", 1, 1),
                comp: geog_fixture(&format!("{tag}-comp"), "continuous", 4, 2),
            }
        }

        fn config(&self, out: PathBuf, workers: usize) -> StaticBuildConfig {
            StaticBuildConfig {
                grid_path: PathBuf::from("<synthetic>"),
                out_path: out,
                geog: StaticGeogPaths {
                    terrain: self.cont1.clone(),
                    landuse: self.cat.clone(),
                    soilcat: self.cat.clone(),
                    greenfrac: self.cont12.clone(),
                    albedo: self.cont12.clone(),
                    snow_albedo: self.cont1.clone(),
                    soil_temperature: self.cont1.clone(),
                    soilcomp: self.comp.clone(),
                    soilcl1: self.cat.clone(),
                    soilcl2: self.cat.clone(),
                    soilcl3: self.cat.clone(),
                    soilcl4: self.cat.clone(),
                },
                stages: StaticStageOptions::default(),
                supersample: 2,
                supersample_landuse: 1,
                supersample_30s: 1,
                host_memory_limit_bytes: None,
                tile_workers: Some(workers),
                native_gwd: None,
                receipt_path: None,
                provenance: "synthetic determinism case".to_string(),
                valid_time: DEFAULT_VALID_TIME.to_string(),
                nominal_dx_m: None,
                clobber: true,
                coordinates: CoordinateRepresentation::default(),
            }
        }
    }

    /// Run every geography pass and write the static, at a chosen worker
    /// count and cache budget.  Returns the output path and the cache tally.
    fn run_case(
        case: &SyntheticCase,
        mesh: &Mesh,
        workers: usize,
        cache_bytes: u64,
        tag: &str,
    ) -> (PathBuf, TileMapCacheReceipt) {
        let out = std::env::temp_dir().join(format!(
            "rw-mpas-static-{}-{tag}.nc",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&out);
        let cfg = case.config(out.clone(), workers);
        let tree = KdTree::new(mesh);
        let max_d2 = mesh.max_search_distance2();
        let cache = TileMapCache::new(cache_bytes);
        let pool = if workers > 1 {
            Some(
                rayon::ThreadPoolBuilder::new().stack_size(crate::execution::BRIDGE_STACK_BYTES)
                    .num_threads(workers)
                    .build()
                    .expect("pool"),
            )
        } else {
            None
        };
        let ctx = GeogContext {
            mesh,
            tree: &tree,
            max_d2,
            workers,
            pool: pool.as_ref(),
            cache: &cache,
        };

        let (ter, _) = continuous_mean(&ctx, &cfg.geog.terrain, cfg.supersample, None, MeanPolicy::Terrain)
            .expect("terrain");
        let (ivgtyp, _, _, lu) =
            categorical(&ctx, &cfg.geog.landuse, cfg.supersample_landuse).expect("landuse");
        let iswater = lu.index.iswater.unwrap_or(17) as i32;
        let landmask: Vec<i32> = ivgtyp
            .iter()
            .map(|&v| if v == iswater { 0 } else { 1 })
            .collect();
        let (isltyp, _, _, _) =
            categorical(&ctx, &cfg.geog.soilcat, cfg.supersample).expect("soilcat");
        let (greenfrac, _) = continuous_mean(
            &ctx,
            &cfg.geog.greenfrac,
            cfg.supersample_30s,
            Some(&landmask),
            MeanPolicy::Green,
        )
        .expect("greenfrac");
        let (albedo, _) = continuous_mean(
            &ctx,
            &cfg.geog.albedo,
            cfg.supersample,
            Some(&landmask),
            MeanPolicy::Albedo,
        )
        .expect("albedo");
        let (snoalb, _) = snow_albedo(
            &ctx,
            &cfg.geog.snow_albedo,
            cfg.supersample_30s,
            &landmask,
        )
        .expect("snoalb");
        let (soiltemp, _) =
            soil_temperature(mesh, &cfg.geog.soil_temperature, &landmask).expect("soiltemp");
        let (comp, _) = soilcomp(&ctx, &cfg.geog.soilcomp, cfg.supersample_30s, &landmask)
            .expect("soilcomp");
        let mut cls: Vec<Vec<i32>> = Vec::new();
        for path in [
            &cfg.geog.soilcl1,
            &cfg.geog.soilcl2,
            &cfg.geog.soilcl3,
            &cfg.geog.soilcl4,
        ] {
            let (v, _, _, _) = categorical(&ctx, path, cfg.supersample_30s).expect("soilcl");
            cls.push(v);
        }
        let receipt = cache.receipt().expect("cache receipt");

        let geo = fixture_geography(
            ter,
            landmask,
            ivgtyp,
            isltyp,
            greenfrac,
            albedo,
            snoalb,
            soiltemp,
            comp,
            [cls[0].clone(), cls[1].clone(), cls[2].clone(), cls[3].clone()],
            iswater,
        );
        write_static(
            &cfg,
            mesh,
            &zero_operators(mesh),
            &geo,
            &flat_gwd(mesh),
            1.0,
            "{\"fixture\":true}",
        )
        .expect("static writes");
        (out, receipt)
    }

    #[test]
    fn an_equal_split_bound_still_visits_the_lower_indexed_cell() {
        // The root is cell 1 at x=0,y=0.5. Its nearer branch reaches cell
        // 2, one unit from the target. Cell 0 in the farther branch is also
        // one unit away, exactly on the split bound, and has the lower ID.
        let mut mesh = synthetic_mesh(3);
        mesh.x_cell = vec![0.0, 0.0, 2.0];
        mesh.y_cell = vec![0.0, 0.5, 0.0];
        mesh.z_cell = vec![0.0; 3];
        assert_eq!(KdTree::new(&mesh).nearest([1.0, 0.0, 0.0]), (0, 1.0));
    }

    #[test]
    fn native_static_stream_follows_noahmp_and_gwd_packages() {
        let case=SyntheticCase::build("native-stream");
        let mesh=synthetic_mesh(8);
        let geo=fixture_geography(vec![0.0;8],vec![1;8],vec![1;8],vec![1;8],
            vec![0.0;96],vec![8.0;96],vec![0.0;8],vec![0.0;8],vec![0.0;64],
            std::array::from_fn(|_|vec![1;8]),17);
        for noahmp in [false,true] {for gwd in [false,true] {
            let path=std::env::temp_dir().join(format!("rw-mpas-native-stream-{}-{noahmp}-{gwd}.nc",std::process::id()));
            let mut cfg=case.config(path.clone(),1);
            cfg.stages.native_init=true;
            cfg.stages.noahmp=noahmp;
            cfg.stages.gwd=gwd;
            write_static(&cfg,&mesh,&zero_operators(&mesh),&geo,&flat_gwd(&mesh),1.0,"{}").unwrap();
            let file=netcrust::File::open(&path).unwrap();
            assert_eq!(file.variable("soilcomp").is_some(),noahmp);
            assert_eq!(file.dimension("nSoilComps").is_some(),noahmp);
            assert_eq!(file.variable("var2d").is_some(),gwd);
            for alias in ["defc_a","defc_b","lu_index","soilcat_top"] {assert!(file.variable(alias).is_none());}
            assert_eq!(file.read_f64("Time").unwrap(),[0.0]);
            std::fs::remove_file(path).unwrap();
        }}
    }

    #[test]
    fn explicit_lake_mask_keeps_native_landuse_words() {
        let case = SyntheticCase::build("native-lake-mask");
        let mesh = synthetic_mesh(8);
        let mut geo = fixture_geography(vec![0.0;8],vec![0;8],vec![17;8],vec![14;8],
            vec![0.0;96],vec![8.0;96],vec![0.0;8],vec![0.0;8],vec![0.0;64],
            std::array::from_fn(|_|vec![1;8]),17);
        let path = std::env::temp_dir().join(format!("rw-mpas-lake-mask-{}.nc",std::process::id()));
        let mut cfg = case.config(path.clone(),1);
        cfg.stages.native_init = true;
        geo.tc_lakemask = Some(vec![1,0,0,1,0,0,0,0]);
        write_static(&cfg,&mesh,&zero_operators(&mesh),&geo,&flat_gwd(&mesh),1.0,"{}").unwrap();
        let file = netcrust::File::open(&path).unwrap();
        assert_eq!(file.read_f64("tc_lakemask").unwrap(), [1.,0.,0.,1.,0.,0.,0.,0.]);
        assert_eq!(file.read_f64("ivgtyp").unwrap(), vec![17.;8]);
        assert_eq!(file.read_f64("landmask").unwrap(), vec![0.;8]);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn static_modis_water_albedo_is_native_unscaled_eight() {
        let case=SyntheticCase::build("native-water-albedo");let mesh=synthetic_mesh(8);
        let tree=KdTree::new(&mesh);let cache=TileMapCache::new(0);
        let ctx=GeogContext {mesh:&mesh,tree:&tree,max_d2:mesh.max_search_distance2(),workers:1,pool:None,cache:&cache};
        let mask=vec![0;mesh.n_cells];
        let (values,_)=continuous_mean(&ctx,&case.config(PathBuf::from("unused"),1).geog.albedo,2,Some(&mask),MeanPolicy::Albedo).unwrap();
        assert_eq!(values,vec![8.0;mesh.n_cells*12]);
    }

    #[test]
    fn static_native_topology_clears_only_unused_slots() {
        let input=[7,8,9,10,11,12,13,14];
        assert_eq!(&*static_topology_words(&input,4,&[2,3],true),&[7,8,0,0,11,12,13,0]);
        assert_eq!(&*static_topology_words(&input,4,&[2,3],false),&input);
    }

    #[test]
    fn nearest_cell_ties_keep_the_same_owner_after_a_parent_subset() {
        // These are three distinct centres on a sphere. A source pixel at
        // (1,0,0) is exactly equidistant from the first two. Removing the
        // farther third centre changes the kd root and used to change the
        // pixel's owner, even though both nearest centres are retained.
        let mut parent = synthetic_mesh(3);
        let k = std::f64::consts::FRAC_1_SQRT_2;
        parent.x_cell = vec![0.0, 0.0, -k];
        parent.y_cell = vec![1.0, -1.0, k];
        parent.z_cell = vec![0.0; 3];
        let mut subset = synthetic_mesh(2);
        subset.x_cell = parent.x_cell[..2].to_vec();
        subset.y_cell = parent.y_cell[..2].to_vec();
        subset.z_cell = parent.z_cell[..2].to_vec();
        let target = [1.0, 0.0, 0.0];
        let whole = KdTree::new(&parent).nearest(target);
        let cut = KdTree::new(&subset).nearest(target);
        assert_eq!(whole, cut, "removing a farther cell moved a tied source pixel");
        assert_eq!(whole, (1, 2.0), "MPAS ties choose lexicographically smaller coordinates");
    }

    #[test]
    fn terrain_overlay_values_are_the_full_static_values_bit_for_bit() {
        let case = SyntheticCase::build("terrain-overlay");
        let mesh = synthetic_mesh(120);
        let (full, _) = run_case(&case, &mesh, 2, 0, "terrain-overlay-full");
        let overlay = std::env::temp_dir().join(format!("rw-mpas-terrain-overlay-{}.nc", std::process::id()));
        let _ = std::fs::remove_file(&overlay);
        let mut config = case.config(overlay.clone(), 2);
        config.grid_path = full.clone();
        let full_file = netcrust::File::open(&full).unwrap();
        let terrain: Vec<f32> = f64v(&full_file, "ter").unwrap().iter().map(|&v| v as f32).collect();
        write_terrain_overlay(&config, &mesh, &terrain, &zero_operators(&mesh)).unwrap();
        let overlay_file = netcrust::File::open(&overlay).unwrap();
        for name in ["ter", "areaCell", "dvEdge", "dcEdge", "deriv_two", "areaTriangle", "kiteAreasOnVertex", "weightsOnEdge", "nominalMinDc", "latCell", "lonCell", "xCell", "yCell", "zCell", "latEdge", "lonEdge", "xEdge", "yEdge", "zEdge", "latVertex", "lonVertex", "xVertex", "yVertex", "zVertex"] {
            let actual: Vec<u64> = f64v(&overlay_file, name).unwrap().iter().map(|v| v.to_bits()).collect();
            let expected: Vec<u64> = f64v(&full_file, name).unwrap().iter().map(|v| v.to_bits()).collect();
            assert_eq!(actual, expected, "terrain overlay {name} moved a stored bit");
        }
        assert!(overlay_file.variable("landmask").is_none(), "the terrain overlay must not claim a runnable static");
        drop(overlay_file);
        drop(full_file);
        std::fs::remove_file(overlay).unwrap();
    }

    #[test]
    fn a_parallel_build_is_byte_identical_to_a_serial_one() {
        // THE GATE. Splitting the tile loop across workers changes the order
        // contributions reach a destination cell.  If any accumulator were
        // floating point, that reordering would move low bits and two runs of
        // the same build would disagree; a static that differs run to run
        // then breaks every downstream identity check that compares one.
        // Every accumulator here is an integer for exactly that reason, and
        // this is what holds it true.
        let case = SyntheticCase::build("gate");
        let mesh = synthetic_mesh(120);
        let (serial, _) = run_case(&case, &mesh, 1, 0, "serial");
        let (parallel, _) = run_case(&case, &mesh, 8, 0, "parallel");
        let a = sha256_file(&serial).expect("serial digest");
        let b = sha256_file(&parallel).expect("parallel digest");
        assert_eq!(
            a, b,
            "serial {} and 8-worker {} statics differ",
            serial.display(),
            parallel.display()
        );
        assert!(
            std::fs::metadata(&serial).expect("serial static").len() > 4096,
            "the comparison must be of a real static, not an empty file"
        );
    }

    /// Cells on a 4 degree lattice inside a small lat/lon window, each with
    /// its vertex 0.02 rad away, so the search radius is a few pixels and the
    /// far tiles of a global source map to nobody.
    fn synthetic_regional_mesh() -> Mesh {
        let mut mesh = synthetic_mesh(50);
        let (mut lat, mut lon) = (Vec::new(), Vec::new());
        for j in 0..5 {
            for i in 0..10 {
                lat.push((10.0 + 4.0 * j as f64).to_radians());
                lon.push((2.0 + 4.0 * i as f64).to_radians());
            }
        }
        let r = MPAS_EARTH_RADIUS_M;
        for c in 0..50 {
            let p = xyz(r, lat[c], lon[c]);
            mesh.x_cell[c] = p[0];
            mesh.y_cell[c] = p[1];
            mesh.z_cell[c] = p[2];
            let v = xyz(r, lat[c] + 0.02, lon[c]);
            mesh.x_vertex[c] = v[0];
            mesh.y_vertex[c] = v[1];
            mesh.z_vertex[c] = v[2];
        }
        mesh.lat_vertex = lat.iter().map(|&v| v+0.02).collect();
        mesh.lon_vertex = lon.clone();
        mesh.lat_cell = lat;
        mesh.lon_cell = lon;
        mesh
    }

    #[test]
    fn incomplete_archives_follow_native_neighbor_failure_order() {
        // MPAS: mpas_geotile_manager.F:900-993;
        // mpas_init_atm_static.F:1375-1469. Even a tile reaching no cells
        // can stop neighbor discovery when its read fails. The earlier
        // assertion that emptying these tiles preserves all fields was
        // incompatible with native's early return. Whole-driver fixtures
        // compare the resulting fields against pristine Fortran.
        let case = SyntheticCase::build("skip");
        let mesh = synthetic_regional_mesh();
        let (intact, _) = run_case(&case, &mesh, 4, 0, "skip-intact");

        let tree = KdTree::new(&mesh);
        let cache = TileMapCache::new(0);
        let ctx = GeogContext {
            mesh: &mesh,
            tree: &tree,
            max_d2: mesh.max_search_distance2(),
            workers: 1,
            pool: None,
            cache: &cache,
        };
        let copy = |from: &PathBuf, tag: &str| -> (PathBuf, usize, usize) {
            let to = std::env::temp_dir().join(format!(
                "rw-mpas-static-{}-gutted-{tag}",
                std::process::id()
            ));
            let _ = std::fs::remove_dir_all(&to);
            std::fs::create_dir_all(&to).expect("copy directory");
            for entry in std::fs::read_dir(from).expect("fixture listing") {
                let entry = entry.expect("fixture entry");
                std::fs::copy(entry.path(), to.join(entry.file_name())).expect("fixture copy");
            }
            let ds = GeogDataset::open(&to).expect("copy opens");
            let (mut emptied, mut kept) = (0usize, 0usize);
            for tile in &ds.tiles {
                if map_reaches_no_cell(&ctx.tile_map(&ds, tile, 2).expect("tile map"))
                    && map_reaches_no_cell(&ctx.tile_map(&ds, tile, 1).expect("tile map"))
                {
                    std::fs::write(&tile.path, b"").expect("empty the tile");
                    emptied += 1;
                } else {
                    kept += 1;
                }
            }
            (to, emptied, kept)
        };
        let (cont12, emptied, kept) = copy(&case.cont12, "cont12");
        let (cat, _, _) = copy(&case.cat, "cat");
        let (comp, _, _) = copy(&case.comp, "comp");
        assert!(emptied > 0 && kept > 0, "the case reaches every tile or none: {emptied} emptied, {kept} kept");
        // MPAS: mpas_init_atm_static.F:520-610,766-852,1006-1086.
        // The bespoke readers abort on a failed neighbor load, even when
        // that neighbor would reach no cells. The strict whole-driver
        // missing/empty matrix witnesses this separately from the shared
        // mapper's successful early return.
        let landmask = vec![1; mesh.n_cells];
        for policy in [MeanPolicy::Green, MeanPolicy::Albedo] {
            let error = continuous_mean(&ctx, &cont12, 1, Some(&landmask), policy)
                .expect_err("native bespoke reader must stop at the missing neighbor");
            assert!(error.to_string().contains("cannot load a neighbor"), "{error}");
        }
        let (snow, _, _) = copy(&case.cont1, "snow");
        let error = snow_albedo(&ctx, &snow, 1, &landmask)
            .expect_err("native snow reader must stop at the missing neighbor");
        assert!(error.to_string().contains("cannot load a neighbor"), "{error}");

        // Climate and soil temperature stay intact for the complete shared
        // mapper replay; only the categorical and soil-component archives
        // use native's successful early-return path.
        let gutted = SyntheticCase { cont1: case.cont1.clone(),
            cont12: case.cont12.clone(), cat, comp };
        let (skipped, _) = run_case(&gutted, &mesh, 4, 0, "skip-gutted");
        assert_ne!(
            sha256_file(&intact).expect("intact digest"),
            sha256_file(&skipped).expect("gutted digest"),
            "the fixture must exercise native's failed-neighbor path"
        );
        let (serial, _) = run_case(&gutted, &mesh, 1, 0, "skip-gutted-serial");
        assert_eq!(sha256_file(&serial).unwrap(), sha256_file(&skipped).unwrap());
    }

    #[test]
    fn a_one_tile_global_grid_is_never_skipped_for_a_far_region() {
        // Release review 14 (2.9.7): a 1 degree global grid stored as one
        // tile, centred about 160 degrees of longitude from the region. Its
        // corners sit near the poles, so the corner bound came out near 93
        // degrees of arc and the tile was skipped although the region lies
        // inside it: every cell lost its samples.
        let mesh = synthetic_regional_mesh();
        let tree = KdTree::new(&mesh);
        let index = crate::static_geog::GeogIndex {
            kind: SourceKind::Continuous,
            projection: "regular_ll".to_string(),
            dx: 1.0,
            dy: 1.0,
            known_x: 1.0,
            known_y: 1.0,
            known_lat: -89.5,
            known_lon: 20.5,
            wordsize: 2,
            tile_x: 360,
            tile_y: 180,
            tile_z_start: 1,
            tile_z_end: 1,
            tile_bdr: 0,
            signed: false,
            big_endian: true,
            scale_factor: 1.0,
            missing_value: None,
            missing_value_rkind: None,
            category_min: None,
            category_max: None,
            iswater: None,
            islake: None,
            isice: None,
            isoilwater: None,
            mminlu: String::new(),
            row_order_top_bottom: false,
        };
        let tile = TileRef { xs: 1, xe: 360, ys: 1, ye: 180, path: PathBuf::from("00001-00360.00001-00180") };
        let ds = GeogDataset {
            path: PathBuf::from("global-one-tile"),
            index,
            tiles: vec![tile.clone()],
            nx_global: 360,
            ny_global: 180,
            wraps_x: true,
        };
        let max_d2 = mesh.max_search_distance2();
        assert!(!tile_reaches_no_cell(&mesh, &tree, max_d2, &ds, &tile, 1),
            "a tile containing the region must not be skipped");
        let map = tile_map(&mesh, &tree, max_d2, &ds, &tile, 1, false).expect("tile map");
        assert!(map.iter().any(|&c| c >= 0), "the region's cells must receive samples");
    }

    #[test]
    fn the_gate_case_really_does_map_some_pixels_to_nothing() {
        // The gate is only a gate over the branches it reaches.  A band mesh
        // under a global source puts refused entries in the middle of a tile
        // map, which is the case the run-folding accumulator has to carry
        // without merging across the gap.
        let case = SyntheticCase::build("refused");
        let mesh = synthetic_mesh(120);
        let ds = GeogDataset::open(&case.cont1).expect("fixture opens");
        let tree = KdTree::new(&mesh);
        let cache = TileMapCache::new(0);
        let ctx = GeogContext {
            mesh: &mesh,
            tree: &tree,
            max_d2: mesh.max_search_distance2(),
            workers: 1,
            pool: None,
            cache: &cache,
        };
        let mut refused = 0usize;
        let mut mapped = 0usize;
        for tile in &ds.tiles {
            for c in ctx.tile_map(&ds, tile, 1).expect("tile map") {
                if c < 0 {
                    refused += 1;
                } else {
                    mapped += 1;
                }
            }
        }
        assert!(refused > 0, "no pixel was out of reach: {refused}/{mapped}");
        assert!(mapped > 0, "no pixel reached a cell at all");
    }

    #[test]
    fn the_byte_comparison_catches_a_single_moved_value() {
        // A gate never shown to fail is not evidence.  Move one destination
        // cell by one ulp and the digests must part.
        let case = SyntheticCase::build("perturb");
        let mesh = synthetic_mesh(120);
        let (baseline, _) = run_case(&case, &mesh, 1, 0, "perturb-base");
        let baseline_digest = sha256_file(&baseline).expect("baseline digest");

        let out = std::env::temp_dir()
            .join(format!("rw-mpas-static-{}-perturbed.nc", std::process::id()));
        let _ = std::fs::remove_file(&out);
        let cfg = case.config(out.clone(), 1);
        let tree = KdTree::new(&mesh);
        let cache = TileMapCache::new(0);
        let ctx = GeogContext {
            mesh: &mesh,
            tree: &tree,
            max_d2: mesh.max_search_distance2(),
            workers: 1,
            pool: None,
            cache: &cache,
        };
        let (mut ter, _) =
            continuous_mean(&ctx, &cfg.geog.terrain, cfg.supersample, None, MeanPolicy::Terrain).expect("ter");
        ter[7] = f32::from_bits(ter[7].to_bits() ^ 1);
        let (ivgtyp, _, _, lu) =
            categorical(&ctx, &cfg.geog.landuse, cfg.supersample_landuse).expect("landuse");
        let iswater = lu.index.iswater.unwrap_or(17) as i32;
        let landmask: Vec<i32> = ivgtyp
            .iter()
            .map(|&v| if v == iswater { 0 } else { 1 })
            .collect();
        let (isltyp, _, _, _) =
            categorical(&ctx, &cfg.geog.soilcat, cfg.supersample).expect("soilcat");
        let (greenfrac, _) = continuous_mean(
            &ctx,
            &cfg.geog.greenfrac,
            cfg.supersample_30s,
            Some(&landmask),
            MeanPolicy::Green,
        )
        .expect("greenfrac");
        let (snoalb, _) =
            snow_albedo(&ctx, &cfg.geog.snow_albedo, cfg.supersample, &landmask)
                .expect("snoalb");
        let (soiltemp, _) =
            soil_temperature(&mesh, &cfg.geog.soil_temperature, &landmask).expect("soiltemp");
        let (comp, _) = soilcomp(&ctx, &cfg.geog.soilcomp, cfg.supersample_30s, &landmask)
            .expect("soilcomp");
        let (cl, _, _, _) =
            categorical(&ctx, &cfg.geog.soilcl1, cfg.supersample_30s).expect("soilcl");
        let geo = fixture_geography(
            ter,
            landmask,
            ivgtyp,
            isltyp,
            greenfrac.clone(),
            greenfrac,
            snoalb,
            soiltemp,
            comp,
            [cl.clone(), cl.clone(), cl.clone(), cl],
            iswater,
        );
        write_static(
            &cfg,
            &mesh,
            &zero_operators(&mesh),
            &geo,
            &flat_gwd(&mesh),
            1.0,
            "{\"fixture\":true}",
        )
        .expect("perturbed static writes");
        assert_ne!(
            baseline_digest,
            sha256_file(&out).expect("perturbed digest"),
            "a one-ulp move slipped past the digest, so the gate proves nothing"
        );
    }

    #[test]
    fn the_reordering_would_be_visible_if_the_accumulator_were_floating_point() {
        // Why integers, stated as a measurement rather than an assurance.
        // These are contributions of the size the terrain pass accumulates.
        // Summed forwards and backwards as f32 they disagree; summed as the
        // i64 the pass actually uses they cannot.
        let contributions: Vec<i64> = (0..20_000)
            .map(|k: i64| (k * 31 + 17).rem_euclid(4096) + 1_000_000)
            .collect();
        let forward = contributions.iter().fold(0.0f32, |a, &b| a + b as f32);
        let backward = contributions.iter().rev().fold(0.0f32, |a, &b| a + b as f32);
        assert_ne!(
            forward.to_bits(),
            backward.to_bits(),
            "the fixture cannot demonstrate order sensitivity at all"
        );
        let f_int: i64 = contributions.iter().sum();
        let b_int: i64 = contributions.iter().rev().sum();
        assert_eq!(f_int, b_int);
    }

    #[test]
    fn the_reusable_tile_map_changes_the_cost_and_not_the_bytes() {
        // Datasets that share tile geometry reuse one map.  A cached map and
        // a recomputed one are the same bytes -- the map is a pure function
        // of the mesh and the geometry -- so the cache may only ever change
        // how long a build takes.
        let case = SyntheticCase::build("cache");
        let mesh = synthetic_mesh(120);
        let (uncached, cold) = run_case(&case, &mesh, 4, 0, "uncached");
        let (cached, warm) = run_case(&case, &mesh, 4, 64 * 1024 * 1024, "cached");
        assert_eq!(cold.hits, 0, "a zero budget cannot serve a hit");
        assert!(
            warm.hits > 0,
            "the shared-geometry datasets never reused a map: {warm:?}"
        );
        assert!(warm.stored_maps > 0 && warm.stored_bytes > 0, "{warm:?}");
        assert_eq!(
            sha256_file(&uncached).expect("uncached digest"),
            sha256_file(&cached).expect("cached digest")
        );
    }

    #[test]
    fn a_budget_too_small_to_hold_a_map_still_builds_the_same_static() {
        // Running the budget out must degrade to recomputation, not to a
        // partial map or a refusal.
        let case = SyntheticCase::build("tight");
        let mesh = synthetic_mesh(120);
        let (roomy, _) = run_case(&case, &mesh, 4, 64 * 1024 * 1024, "roomy");
        let (tight, tally) = run_case(&case, &mesh, 4, 512, "tight");
        assert!(tally.declined_for_budget > 0, "{tally:?}");
        assert_eq!(
            sha256_file(&roomy).expect("roomy digest"),
            sha256_file(&tight).expect("tight digest")
        );
    }

    #[test]
    fn a_run_length_encoded_map_decodes_to_what_it_encoded() {
        let map: Vec<i32> = vec![-1, -1, 4, 4, 4, 4, 7, 7, 7, 2];
        let encoded = CachedTileMap::encode(&map).expect("encodes");
        assert!(matches!(encoded, CachedTileMap::Runs(_)));
        assert_eq!(encoded.decode(map.len()).expect("decodes"), map);
    }

    #[test]
    fn a_map_that_run_length_encoding_would_grow_is_kept_raw() {
        // A mesh finer than the source has no runs to fold.  Storing the
        // runs anyway would cost twice the raw map, so it is not done.
        let map: Vec<i32> = (0..1000).collect();
        let encoded = CachedTileMap::encode(&map).expect("encodes");
        assert!(matches!(encoded, CachedTileMap::Raw(_)), "{encoded:?}");
        assert_eq!(encoded.bytes(), 4000);
        assert_eq!(encoded.decode(map.len()).expect("decodes"), map);
    }

    #[test]
    fn the_worker_request_comes_from_the_flag_then_the_environment() {
        assert_eq!(resolve_tile_workers(Some(3)), 3);
        assert_eq!(resolve_tile_workers(Some(0)), 1, "zero cannot stall a build");
        assert!(resolve_tile_workers(None) >= 1);
    }

    // -- real WPS_GEOG measurement, on demand ------------------------------
    //
    // Ignored by default: these need a real mesh and a real WPS_GEOG tree,
    // which no CI machine is promised to have.  Run one arm at a time so an
    // external peak-working-set measurement belongs to that arm:
    //
    //   RW_MPAS_REAL_GRID=... RW_MPAS_REAL_TERRAIN=... \
    //   RW_MPAS_REAL_WORKERS=1 RW_MPAS_REAL_OUT=ter-serial.bin \
    //   cargo test -p rw-mpas --release -- --ignored --nocapture real_inputs

    #[test]
    fn a_cell_no_pixel_reaches_takes_the_pixel_under_its_centre() {
        // Two-degree fixture pixels under a 20,000-cell band mesh: most cells
        // are finer than a pixel, so no pixel centre lands nearest to them.
        // Before the fix those cells were 0/0 terrain (NaN) and a category-1
        // vote; the 1 km ocean core of a moving case showed exactly that.
        let case = SyntheticCase::build("centre-seed");
        let mesh = synthetic_mesh(20_000);
        let tree = KdTree::new(&mesh);
        let cache = TileMapCache::new(0);
        let ctx = GeogContext {
            mesh: &mesh,
            tree: &tree,
            max_d2: mesh.max_search_distance2(),
            workers: 1,
            pool: None,
            cache: &cache,
        };
        let (ter, ds) =
            continuous_mean(&ctx, &case.cont1, 1, None, MeanPolicy::Terrain).expect("terrain");
        let seeded = take_centre_seeded();
        assert!(seeded > 0, "the fixture must leave some cells without a pixel");
        let non_finite = ter.iter().filter(|v| !v.is_finite()).count();
        assert_eq!(non_finite, 0, "terrain holds non-finite values");
        // A seeded cell carries the pixel under its centre, scaled as native would.
        let every: Vec<usize> = (0..mesh.n_cells).collect();
        let mut carried = 0usize;
        for (k, hits) in centre_pixels(&ds, &mesh, &every) {
            let plane = ds.read_plane(&ds.tiles[k], 0).expect("plane");
            for &(c, i, j) in &hits {
                let raw = plane_raw_at(&ds, &plane, i, j);
                if ter[c] == crate::static_native::scaled_mean(raw, 1, ds.index.scale_factor as f32) {
                    carried += 1;
                }
            }
        }
        assert!(carried >= seeded, "{carried} cells carry their centre pixel, {seeded} were seeded");
        let (cat, cmin, cmax, _) = categorical(&ctx, &case.cat, 1).expect("categorical");
        assert!(take_centre_seeded() > 0, "the category vote must seed the same cells");
        assert!(cat.iter().all(|&v| (cmin as i32..=cmax as i32).contains(&v)));
    }

    fn real_path(key: &str) -> PathBuf {
        PathBuf::from(std::env::var(key).unwrap_or_else(|_| {
            panic!(
                "{key} must name a real input for this measurement; it is \
                 ignored by default because no machine is promised to have one"
            )
        }))
    }

    #[test]
    #[ignore = "needs a real mesh and a real WPS_GEOG tree"]
    fn real_inputs_terrain_pass() {
        let grid = real_path("RW_MPAS_REAL_GRID");
        let terrain = real_path("RW_MPAS_REAL_TERRAIN");
        let workers = std::env::var("RW_MPAS_REAL_WORKERS")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or_else(|| resolve_tile_workers(None));

        let read_started = Instant::now();
        let mesh = Mesh::read(&grid).expect("mesh reads");
        let tree = KdTree::new(&mesh);
        let max_d2 = mesh.max_search_distance2();
        let mesh_seconds = read_started.elapsed().as_secs_f64();

        let cache = TileMapCache::new(0);
        let pool = if workers > 1 {
            Some(
                rayon::ThreadPoolBuilder::new().stack_size(crate::execution::BRIDGE_STACK_BYTES)
                    .num_threads(workers)
                    .build()
                    .expect("pool"),
            )
        } else {
            None
        };
        let ctx = GeogContext {
            mesh: &mesh,
            tree: &tree,
            max_d2,
            workers,
            pool: pool.as_ref(),
            cache: &cache,
        };
        let started = Instant::now();
        let (ter, ds) =
            continuous_mean(&ctx, &terrain, 1, None, MeanPolicy::Terrain).expect("terrain pass");
        let seconds = started.elapsed().as_secs_f64();

        let out = PathBuf::from(
            std::env::var("RW_MPAS_REAL_OUT")
                .unwrap_or_else(|_| format!("ter-{workers}-workers.bin")),
        );
        let mut bytes = Vec::with_capacity(ter.len() * 4);
        for v in &ter {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        std::fs::write(&out, &bytes).expect("terrain bytes write");
        println!(
            "REAL-TERRAIN workers={workers} cells={} tiles={} mesh_seconds={mesh_seconds:.2} \
             pass_seconds={seconds:.2} out={} sha256={}",
            mesh.n_cells,
            ds.tiles.len(),
            out.display(),
            sha256_file(&out).expect("digest")
        );
    }

    #[test]
    #[ignore = "needs a real mesh and a real WPS_GEOG tree"]
    fn real_inputs_shared_geometry_reuses_maps() {
        // Two 30-arcsec products on the same grid.  The second pass pays for
        // reading and accumulating, and not for the kd search again.
        let grid = real_path("RW_MPAS_REAL_GRID");
        let first = real_path("RW_MPAS_REAL_CAT_A");
        let second = real_path("RW_MPAS_REAL_CAT_B");
        let workers = std::env::var("RW_MPAS_REAL_WORKERS")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or_else(|| resolve_tile_workers(None));
        let budget: u64 = std::env::var("RW_MPAS_REAL_CACHE_BYTES")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(4 * 1024 * 1024 * 1024);

        let mesh = Mesh::read(&grid).expect("mesh reads");
        let tree = KdTree::new(&mesh);
        let max_d2 = mesh.max_search_distance2();
        let cache = TileMapCache::new(budget);
        let pool = if workers > 1 {
            Some(
                rayon::ThreadPoolBuilder::new().stack_size(crate::execution::BRIDGE_STACK_BYTES)
                    .num_threads(workers)
                    .build()
                    .expect("pool"),
            )
        } else {
            None
        };
        let ctx = GeogContext {
            mesh: &mesh,
            tree: &tree,
            max_d2,
            workers,
            pool: pool.as_ref(),
            cache: &cache,
        };
        let t0 = Instant::now();
        let (a, _, _, _) = categorical(&ctx, &first, 1).expect("first categorical");
        let cold = t0.elapsed().as_secs_f64();
        let t1 = Instant::now();
        let (b, _, _, _) = categorical(&ctx, &second, 1).expect("second categorical");
        let warm = t1.elapsed().as_secs_f64();
        let tally = cache.receipt().expect("cache receipt");
        println!(
            "REAL-SHARED workers={workers} cold_seconds={cold:.2} warm_seconds={warm:.2} \
             hits={} misses={} stored_maps={} stored_bytes={} declined={} \
             a_len={} b_len={}",
            tally.hits,
            tally.misses,
            tally.stored_maps,
            tally.stored_bytes,
            tally.declined_for_budget,
            a.len(),
            b.len()
        );
        assert!(tally.hits > 0, "the second product remapped the same grid");
    }

    /// MPAS: mpas_kd_tree.F:64-159,226-332. Golden IDs and distances from
    /// the pristine module, including exact cutoff and coordinate ties.
    #[test]
    fn native_kd_matches_pristine_ids_and_distance_words() {
        let words=include_str!("../tests/native_static/kd.words");
        let mut trees=Vec::new();
        for n in 1..=37 {
            let mut mesh=synthetic_mesh(n);
            for c in 0..n {
                let i=c+1;
                mesh.x_cell[c]=(((i*17%23) as i32-11) as f32*0.125) as f64;
                mesh.y_cell[c]=(((i*11%19) as i32-9) as f32*0.25) as f64;
                mesh.z_cell[c]=(((i*13%29) as i32-14) as f32*0.125) as f64;
            }
            trees.push(KdTree::new(&mesh));
        }
        let mut mismatch=Vec::new();
        let mut total=0;
        for line in words.lines() {
            let f:Vec<_>=line.split_whitespace().collect();
            let n:usize=f[0].parse().unwrap();
            let q:usize=f[1].parse().unwrap();
            let id:i32=f[2].parse().unwrap();
            let expected=u32::from_str_radix(f[3],16).unwrap();
            let t=[((q*71%41) as i32-20) as f32*0.125,
                   ((q*37%43) as i32-21) as f32*0.125,
                   ((q*17%47) as i32-23) as f32*0.125].map(f64::from);
            let limit=[0.0,0.5,8.0,f32::MAX][q%4] as f64;
            let actual=trees[n-1].nearest_bounded(t,limit);
            let got=actual.map(|(c,d)| (c as i32+1,(d as f32).to_bits())).unwrap_or((-1,(-1.0f32).to_bits()));
            if got!=(id,expected) && mismatch.len()<20 { mismatch.push((n,q,(id,expected),got)); }
            total+=1;
        }
        assert_eq!(total,18944);
        assert!(mismatch.is_empty(), "native KD word mismatches: {mismatch:?}");
    }

    #[test]
    #[ignore = "real large-mesh startup stage; requires RW_MPAS_REAL_GRID"]
    fn real_startup_search_radius_keeps_serial_words() {
        let mesh = Mesh::read(&real_path("RW_MPAS_REAL_GRID")).unwrap();
        let started = Instant::now();
        let parallel = mesh.max_search_distance2();
        let parallel_seconds = started.elapsed().as_secs_f64();
        let started = Instant::now();
        let mut max_r = 0.0f32;
        for c in 0..mesh.n_cells {
            for s in 0..mesh.n_edges_on_cell[c] {
                let v = mesh.vertices_on_cell[c * mesh.max_edges + s];
                let arc = crate::static_native::sphere_distance(
                    mesh.lat_cell[c] as f32, mesh.lon_cell[c] as f32,
                    mesh.lat_vertex[v] as f32, mesh.lon_vertex[v] as f32,
                    mesh.sphere_radius as f32);
                max_r = max_r.max(arc);
            }
        }
        let diameter = 2.0 * max_r;
        let serial = (2.0 * (diameter * diameter)) as f64;
        assert_eq!(serial.to_bits(), parallel.to_bits());
        println!("startup radius: cells={} workers={} serial_seconds={} parallel_seconds={} words={:016x}",
                 mesh.n_cells, rayon::current_num_threads(), started.elapsed().as_secs_f64(),
                 parallel_seconds, parallel.to_bits());
    }

    #[test]
    #[ignore = "10 km startup stage; run explicitly on a shared host with room"]
    fn ten_km_parallel_kd_keeps_serial_words_without_stack_environment() {
        fn serial(points: &[[f64; 3]], order: &mut [u32], axes: &mut [u8], ax: usize) {
            if order.is_empty() { return; }
            let mid = (order.len() - 1) / 2;
            order.select_nth_unstable_by(mid, |&a, &b|
                (points[a as usize][ax] as f32).total_cmp(&(points[b as usize][ax] as f32)));
            axes[mid] = ax as u8;
            let (left, rest) = order.split_at_mut(mid);
            let (la, ra) = axes.split_at_mut(mid);
            serial(points, left, la, (ax+1)%3);
            serial(points, &mut rest[1..], &mut ra[1..], (ax+1)%3);
        }
        let n = 5_898_242usize;
        let points: Vec<_> = (0..n).map(|i| [
            (i.wrapping_mul(17)%100003) as f64,
            (i.wrapping_mul(71)%100019) as f64,
            (i.wrapping_mul(37)%100043) as f64]).collect();
        let mut expected: Vec<u32> = (0..n as u32).collect();
        let mut expected_axes = vec![0; n];
        serial(&points, &mut expected, &mut expected_axes, 0);
        let mut actual: Vec<u32> = (0..n as u32).collect();
        let mut actual_axes = vec![0; n];
        let pool = rayon::ThreadPoolBuilder::new().num_threads(8)
            .stack_size(crate::execution::BRIDGE_STACK_BYTES).build().unwrap();
        pool.install(|| kd_build(&points, &mut actual, &mut actual_axes, 0, n));
        assert_eq!(actual, expected);
        assert_eq!(actual_axes, expected_axes);
        println!("10km KD: {n} cells, 8 workers, serial words match");
    }

    #[test]
    fn native_sampling_accepts_configured_empty_and_positive_factors() {
        for factor in [0,1,4,16,17,1024] { validate_supersample(factor).unwrap(); }
        assert!(validate_supersample(usize::MAX).is_err());
    }

    // -- the coordinate representation a static DECLARES is what it HOLDS ---
    //
    // THE BREAKAGE (2.9.0): bad5408e05a put every grid
    // on the native RKIND read, so a GENERATED grid's static declared
    // binary64_earth_centred and held binary32 positions up to 0.44 m off its
    // grid. gpuwm-hex's point door refused its vertical check and
    // `rw_mpas_mesh --cull-parent` refused the cut ("provenance grid latCell
    // differs"), so no 3 km regional mesh could be built. These two tests hold
    // both halves: the generated route binds bit for bit, the native route is
    // still the RKIND read the i2 replay proves.

    /// Generate a grid, build its static through the shipping entry point, and
    /// cut both the way the point door does: the static's cut must bind to the
    /// grid's cut under `--cull-provenance-grid`.
    #[test]
    fn a_generated_grid_gets_a_static_that_holds_its_binary64_declaration() {
        use crate::mesh::cull::{cull_file, cull_file_with_provenance};
        use crate::mesh::density::{MeshSpec, Shape};
        use crate::mesh::emit::{write_grid, Provenance};
        use crate::mesh::{generate, GenerateRequest, LloydOptions};
        use crate::staticfile::coordframe::for_static_from_grid;

        let dir = std::env::temp_dir().join(format!("rw-mpas-static-binary64-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let request = GenerateRequest {
            spec: MeshSpec::uniform(480.0),
            target_cells: Some(2562),
            lloyd: LloydOptions { tolerance: 3e-3, max_sweeps: 200, ..Default::default() },
            sizing_samples: 20_000,
            ..Default::default()
        };
        let generated = generate(&request, |_| {}).expect("generate");
        let grid = dir.join("generated.grid.nc");
        write_grid(
            &generated.mesh,
            &grid,
            &Provenance {
                static_coordinates: Some(CoordinateRepresentation::for_generated_mesh()),
                ..Default::default()
            },
            true,
        )
        .expect("write the generated grid");
        let representation = for_static_from_grid(&grid).unwrap();
        assert!(representation.is_binary64(), "a generated grid asks for binary64 statics");

        let case = SyntheticCase::build("binary64-static");
        let static_path = dir.join("generated.static.nc");
        let mut cfg = case.config(static_path.clone(), 2);
        cfg.grid_path = grid.clone();
        cfg.coordinates = representation;
        build_static_reporting(&cfg, &|_| {}).expect("static builds on the generated grid");

        let g = netcrust::File::open(&grid).unwrap();
        let s = netcrust::File::open(&static_path).unwrap();
        let r = MPAS_EARTH_RADIUS_M;
        let bits = |v: &[f64]| -> Vec<u64> { v.iter().map(|x| x.to_bits()).collect() };
        for entity in ["Cell", "Edge", "Vertex"] {
            for axis in ["x", "y", "z"] {
                let name = format!("{axis}{entity}");
                let want: Vec<f64> = f64v(&g, &name).unwrap().iter().map(|v| v * r).collect();
                assert_eq!(bits(&f64v(&s, &name).unwrap()), bits(&want),
                    "{name}: a binary64 static holds the grid's own position scaled in binary64");
            }
            for axis in ["lat", "lon"] {
                let name = format!("{axis}{entity}");
                assert_eq!(bits(&f64v(&s, &name).unwrap()), bits(&f64v(&g, &name).unwrap()),
                    "{name}: a binary64 static holds the grid's own angle");
            }
        }
        for (name, power) in [("dcEdge", 1), ("dvEdge", 1), ("areaCell", 2), ("kiteAreasOnVertex", 2)] {
            let scale = if power == 1 { r } else { r * r };
            let want: Vec<u32> = f64v(&g, name).unwrap().iter().map(|v| ((v * scale) as f32).to_bits()).collect();
            let got: Vec<u32> = f64v(&s, name).unwrap().iter().map(|&v| (v as f32).to_bits()).collect();
            assert_eq!(got, want, "{name}: one binary32 rounding of the binary64 metric");
        }
        drop((g, s));

        let region = Shape::Cap { center_deg: [34.0, -118.3], radius_km: 1500.0 };
        let cut_grid = dir.join("cut.grid.nc");
        let receipt = cull_file(&grid, &region, &cut_grid, None).expect("cut the grid");
        assert!(receipt.region_cells > 20, "the cut holds too few cells to test anything");
        cull_file_with_provenance(&static_path, &region, &dir.join("cut.static.nc"), None, Some(&cut_grid))
            .expect("the static's cut binds to the grid's cut, as rw_mpas_mesh --cull-parent demands");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A grid that declares nothing -- every published mesh -- is still read as
    /// the native RKIND=4 mesh pool, value for value.
    #[test]
    fn a_grid_with_no_declaration_keeps_the_native_rkind_read() {
        // Its own file: `clean_regional` is removed by another test running
        // in parallel.
        let path = tiny_grid_nc(
            "rkind-read",
            Some([5, 6, 7, 7]),
            [5, 6, 7, 7],
            [0, 0, 7, 7],
            [[2, 3, 4], [1, 3, 4], [1, 2, 0], [1, 2, 0]],
            [[1, 2], [2, 3], [3, 0], [4, 0]],
            [[1, 2, 3], [2, 3, 4], [3, 4, 1], [4, 1, 2]],
            [[1, 2, 0, 0, 0, 0], [2, 3, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [0, 2, 0, 0, 0, 0]],
        );
        let f = netcrust::File::open(&path).unwrap();
        let mesh = Mesh::read(&path).expect("the grid reads");
        let r = MPAS_EARTH_RADIUS_M as f32;
        let native = |name: &str| -> Vec<u64> {
            f64v(&f, name).unwrap().iter().map(|&v| (((v as f32) * r) as f64).to_bits()).collect()
        };
        let bits = |v: &[f64]| -> Vec<u64> { v.iter().map(|x| x.to_bits()).collect() };
        assert_eq!(bits(&mesh.x_cell), native("xCell"));
        assert_eq!(bits(&mesh.y_vertex), native("yVertex"));
        assert_eq!(bits(&mesh.dc_edge), native("dcEdge"));
        let rounded: Vec<u64> = f64v(&f, "lonCell").unwrap().iter().map(|&v| ((v as f32) as f64).to_bits()).collect();
        assert_eq!(bits(&mesh.lon_cell), rounded);
        drop(f);
        let _ = std::fs::remove_file(&path);
    }
}
