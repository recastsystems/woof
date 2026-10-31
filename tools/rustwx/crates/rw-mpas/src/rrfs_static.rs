//! RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_static.F:110-1424,1957-2120 at 1d6e370e.
//! Complete sibling builder; geography/mesh/operator plumbing is retained from i2.
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
use crate::rrfs_static_operators::{build_operator_fields, OperatorFields, OperatorMesh, FIFTEEN};
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

/// The schema tag the unified writer stamps into every file it writes.
pub const UNIFIED_SCHEMA_TAG: &str = "rw-mpas-static/v3-unified";

/// MPAS: core_init_atmosphere/Registry.xml:498-585,882-903.
/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_static.F:110-1422 at 1d6e370e (whole module driver boundary).
fn native_static_field_enabled(name: &str, stages: &StaticStageOptions) -> bool {
    match name {
        "lu_index"|"soilcat_top" => false,
        "soilcomp"|"soilcl1"|"soilcl2"|"soilcl3"|"soilcl4" => stages.noahmp,
        "var2d"|"con"|"oa1"|"oa2"|"oa3"|"oa4"|"ol1"|"ol2"|"ol3"|"ol4" => stages.gwd,
        _=>true,
    }
}

/// MPAS: src/framework/mpas_stream_manager.F:4898-4900,5037-5162.
/// Native streams discard connectivity outside each entity's active stencil.
/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_static.F:110-1422 at 1d6e370e (whole module driver boundary).
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
/// They are constants rather than branches so the next field's conversion is a
/// value and not a code path.
pub const GREENFRAC_MODEL_SCALE: f32 = 100.0;
/// See [`GREENFRAC_MODEL_SCALE`].
pub const SNOALB_MODEL_SCALE: f32 = 0.01;

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StaticGeogPaths {
    pub lai: PathBuf,
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

/// MPAS: mpas_init_atm_static.F:501-1210,1199-1250.
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields)]
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
}
impl Default for StaticStageOptions {
    /// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_static.F:110-1422 at 1d6e370e (whole module driver boundary).
    fn default() -> Self {
        Self { vegetation_ncep: false, albedo_ncep: false, snow_ncep: false, noahmp: true, gwd: true, landuse_label: None, native_init: false }
    }
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StaticBuildConfig {
    pub nvegopt: usize,
    pub gwd_topo_shift: i32,
    pub gwd_cell_scaling: f32,
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
    #[serde(deserialize_with="crate::rrfs_init::job::coordinates") ]
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
pub const GEOG_DATASET_TABLE: [(&str, &str); 13] = [
    ("lai12m","lai_modis_30s"),
    ("terrain", "topo_gmted2010_30s"),
    ("landuse", "modis_landuse_20class_30s_with_lakes"),
    ("soilcat", "soiltype_top_30s"),
    ("greenfrac", "greenfrac_fpar_modis"),
    ("albedo", "albedo_modis"),
    ("snow_albedo", "maxsnowalb_modis"),
    ("soil_temperature", "soiltemp_1deg"),
    ("soilcomp", "soilcomp"),
    ("soilcl1", "texture_layer1"),
    ("soilcl2", "texture_layer2"),
    ("soilcl3", "texture_layer3"),
    ("soilcl4", "texture_layer4"),
];

/// MPAS: mpas_init_atm_static.F:383-460,488-1198.
/// Option selection is data; the current door's staged defaults stay in the
/// GEOG_DATASET_TABLE above.
pub const MPAS_STATIC_SOURCE_TABLE: &[(&str,&str,&str)] = &[
    ("topo-data","GTOPO30","topo_30s"),
    ("topo-data","GMTED2010","topo_gmted2010_30s"),
    ("landuse-data","USGS","landuse_30s"),
    ("landuse-data","MODIFIED_IGBP_MODIS_NOAH","modis_landuse_20class_30s"),
    ("landuse-data","MODIFIED_IGBP_MODIS_NOAH_15s","modis_landuse_20class_15s"),
    ("soilcat-data","STATSGO","soiltype_top_30s"),
    ("soilcat-data","BNU","bnu_soiltype_top"),
    ("vegfrac-data","MODIS","greenfrac_fpar_modis"),
    ("vegfrac-data","NCEP","greenfrac"),
    ("albedo-data","MODIS","albedo_modis"),
    ("albedo-data","NCEP","albedo_ncep"),
    ("maxsnowalbedo-data","MODIS","maxsnowalb_modis"),
    ("maxsnowalbedo-data","NCEP","maxsnowalb"),
];

/// MPAS: mpas_init_atm_static.F:383-460,488-1198 (dataset selection).
/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_static.F:110-1422 at 1d6e370e (whole module driver boundary).
pub fn select_static_source(cfg: &mut StaticBuildConfig, root: &Path, slot: &str, choice: &str) -> MpasResult<()> {
    let dataset=MPAS_STATIC_SOURCE_TABLE.iter().find(|(s,c,_)| *s==slot && *c==choice)
        .ok_or_else(|| MpasError::Refusal(format!("unsupported MPAS static {slot}={choice}")))?.2;
    let path=resolve_geog_dataset_dir(root,dataset);
    match slot {
        "topo-data" => cfg.geog.terrain=path,
        "landuse-data" => {
            cfg.geog.landuse=path;
            cfg.stages.landuse_label=Some(if choice=="USGS" { "USGS" } else { "MODIFIED_IGBP_MODIS_NOAH" }.into());
        },
        "soilcat-data" => cfg.geog.soilcat=path,
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
/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_static.F:110-1422 at 1d6e370e (whole module driver boundary).
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
    /// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_static.F:110-1422 at 1d6e370e (whole module driver boundary).
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
            lai:at("lai12m"),
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
    /// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_static.F:110-1422 at 1d6e370e (whole module driver boundary).
    pub fn slots(&self) -> [(&'static str, &Path); 13] {
        [
            ("terrain", &self.terrain),
            ("landuse", &self.landuse),
            ("soilcat", &self.soilcat),
            ("greenfrac", &self.greenfrac),
            ("lai12m",&self.lai),
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
    /// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_static.F:110-1422 at 1d6e370e (whole module driver boundary).
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
/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_static.F:110-1422 at 1d6e370e (whole module driver boundary).
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

/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_static.F:110-1422 at 1d6e370e (whole module driver boundary).
fn dim(file: &netcrust::File, name: &str) -> MpasResult<usize> {
    file.dimension(name)
        .map(|d| d.len())
        .ok_or_else(|| MpasError::Refusal(format!("grid has no {name} dimension")))
}

/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_static.F:110-1422 at 1d6e370e (whole module driver boundary).
fn f64v(file: &netcrust::File, name: &str) -> MpasResult<Vec<f64>> {
    Ok(file
        .read_array_f64(name)
        .map_err(|e| MpasError::Refusal(format!("grid has no readable {name}: {e}")))?
        .into_values())
}

/// MPAS: mpas_init_atm_static.F:307-325: mesh pools are RKIND.
/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_static.F:110-1422 at 1d6e370e (whole module driver boundary).
fn rkindv(file: &netcrust::File, name: &str) -> MpasResult<Vec<f64>> {
    Ok(f64v(file, name)?.into_iter().map(|v| (v as f32) as f64).collect())
}

/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_static.F:110-1422 at 1d6e370e (whole module driver boundary).
fn i32v(file: &netcrust::File, name: &str) -> MpasResult<Vec<i32>> {
    Ok(f64v(file, name)?
        .into_iter()
        .map(|v| v.round() as i32)
        .collect())
}

/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_static.F:110-1422 at 1d6e370e (whole module driver boundary).
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
/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_static.F:110-1422 at 1d6e370e (whole module driver boundary).
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
/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_static.F:110-1422 at 1d6e370e (whole module driver boundary).
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
/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_static.F:110-1422 at 1d6e370e (whole module driver boundary).
fn identity_ids(file: &netcrust::File, name: &str, len: usize) -> MpasResult<Vec<i32>> {
    if file.variable(name).is_some() {
        return shaped_i32(file, name, len);
    }
    Ok((1..=len as i32).collect())
}

/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_static.F:110-1422 at 1d6e370e (whole module driver boundary).
fn zero_based(raw: Vec<i32>, upper: usize, name: &str, allow_zero_padding: bool) -> MpasResult<Vec<usize>> {
    raw.into_iter()
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
    /// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_static.F:110-1422 at 1d6e370e (whole module driver boundary).
    fn read(path: &Path) -> MpasResult<Self> {
        let f = netcrust::File::open(path)?;
        let n_cells = dim(&f, "nCells")?;
        let n_edges = dim(&f, "nEdges")?;
        let n_vertices = dim(&f, "nVertices")?;
        let max_edges = dim(&f, "maxEdges")?;
        let max_edges2 = dim(&f, "maxEdges2")?;
        let vertex_degree = dim(&f, "vertexDegree")?;

        let mut x_cell = rkindv(&f, "xCell")?;
        let mut y_cell = rkindv(&f, "yCell")?;
        let mut z_cell = rkindv(&f, "zCell")?;
        let mut x_edge = rkindv(&f, "xEdge")?;
        let mut y_edge = rkindv(&f, "yEdge")?;
        let mut z_edge = rkindv(&f, "zEdge")?;
        let mut x_vertex = rkindv(&f, "xVertex")?;
        let mut y_vertex = rkindv(&f, "yVertex")?;
        let mut z_vertex = rkindv(&f, "zVertex")?;
        let mut dc_edge = rkindv(&f, "dcEdge")?;
        let mut dv_edge = rkindv(&f, "dvEdge")?;
        let mut area_cell = rkindv(&f, "areaCell")?;
        let mut area_triangle = rkindv(&f, "areaTriangle")?;
        let mut kite_areas_on_vertex = rkindv(&f, "kiteAreasOnVertex")?;
        let mut nominal = rkindv(&f, "nominalMinDc")?
            .first()
            .copied()
            .ok_or_else(|| MpasError::Refusal("grid nominalMinDc is empty".to_string()))?;

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
            for v in x_cell.iter_mut().chain(y_cell.iter_mut()).chain(z_cell.iter_mut())
                .chain(x_edge.iter_mut()).chain(y_edge.iter_mut()).chain(z_edge.iter_mut())
                .chain(x_vertex.iter_mut()).chain(y_vertex.iter_mut()).chain(z_vertex.iter_mut())
            {
                *v = ((*v as f32) * (r as f32)) as f64;
            }
            for v in dc_edge.iter_mut().chain(dv_edge.iter_mut()) {
                *v = ((*v as f32) * (r as f32)) as f64;
            }
            for v in area_cell
                .iter_mut()
                .chain(area_triangle.iter_mut())
                .chain(kite_areas_on_vertex.iter_mut())
            {
                *v = ((*v as f32) * ((r as f32) * (r as f32))) as f64;
            }
            nominal = ((nominal as f32) * (r as f32)) as f64;
        }

        let neoc_i32 = i32v(&f, "nEdgesOnCell")?;
        if neoc_i32.len() != n_cells {
            return Err(MpasError::Refusal("nEdgesOnCell shape mismatch".to_string()));
        }
        let n_edges_on_cell: Vec<usize> = neoc_i32.into_iter().map(|v| v as usize).collect();
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
        let normalize_padded = |raw: Vec<i32>,
                                upper: usize,
                                name: &str,
                                absent_ok: Option<&[i32]>|
         -> MpasResult<Vec<usize>> {
            let mut out = vec![0usize; raw.len()];
            for c in 0..n_cells {
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
                                    out[k] = ABSENT_NEIGHBOR;
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
                        out[k] = v as usize - 1;
                    }
                }
            }
            Ok(out)
        };

        let raw_cells_on_cell = cells_on_cell_raw.clone();
        let raw_edges_on_cell = edges_on_cell_raw.clone();
        let raw_vertices_on_cell = vertices_on_cell_raw.clone();
        let cells_on_cell =
            normalize_padded(cells_on_cell_raw, n_cells, "cellsOnCell", Some(&bdy_mask_cell))?;
        let edges_on_cell = normalize_padded(edges_on_cell_raw, n_edges, "edgesOnCell", None)?;
        let vertices_on_cell =
            normalize_padded(vertices_on_cell_raw, n_vertices, "verticesOnCell", None)?;

        let coe_raw = i32v(&f, "cellsOnEdge")?;
        let voe_raw = i32v(&f, "verticesOnEdge")?;
        if coe_raw.len() != 2 * n_edges || voe_raw.len() != 2 * n_edges {
            return Err(MpasError::Refusal("edge-pair topology shape mismatch".to_string()));
        }
        let raw_cells_on_edge = coe_raw.clone();
        let raw_vertices_on_edge = voe_raw.clone();
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
        let voe = zero_based(voe_raw, n_vertices, "verticesOnEdge", false)?;
        let cells_on_edge = (0..n_edges).map(|e| [coe[2 * e], coe[2 * e + 1]]).collect();
        let vertices_on_edge = (0..n_edges).map(|e| [voe[2 * e], voe[2 * e + 1]]).collect();

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
            sphere_radius: r,
            lat_cell: rkindv(&f, "latCell")?,
            lon_cell: rkindv(&f, "lonCell")?,
            lat_edge: rkindv(&f, "latEdge")?,
            lon_edge: rkindv(&f, "lonEdge")?,
            lat_vertex: rkindv(&f, "latVertex")?,
            lon_vertex: rkindv(&f, "lonVertex")?,
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
            angle_edge: rkindv(&f, "angleEdge")?,
            mesh_density: rkindv(&f, "meshDensity")?,
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

    /// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_static.F:110-1422 at 1d6e370e (whole module driver boundary).
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
    /// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_static.F:110-1422 at 1d6e370e (whole module driver boundary).
    fn is_regional(&self) -> bool {
        self.bdy_mask_cell.iter().any(|&v| v != 0)
    }

    /// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_static.F:110-1422 at 1d6e370e (whole module driver boundary).
    fn operator_view(&self) -> OperatorMesh<'_> {
        OperatorMesh {
            n_cells: self.n_cells,
            n_edges: self.n_edges,
            max_edges: self.max_edges,
            sphere_radius: self.sphere_radius,
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

    /// MPAS: mpas_init_atm_static.F:329-344,2241-2289.
    /// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_static.F:110-1422 at 1d6e370e (whole module driver boundary).
    fn max_search_distance2(&self) -> f64 {
        let mut max_r = 0.0f32;
        for c in 0..self.n_cells {
            for s in 0..self.n_edges_on_cell[c] {
                let v = self.vertices_on_cell[c * self.max_edges + s];
                let arc = crate::static_native::sphere_distance(
                    self.lat_cell[c] as f32, self.lon_cell[c] as f32,
                    self.lat_vertex[v] as f32, self.lon_vertex[v] as f32,
                    self.sphere_radius as f32);
                max_r = max_r.max(arc);
            }
        }
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
/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_static.F:110-1422 at 1d6e370e (whole module driver boundary).
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
/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_static.F:110-1422 at 1d6e370e (whole module driver boundary).
fn kd_build(points: &[[f64; 3]], order: &mut [u32], axis: &mut [u8], lo: usize, hi: usize) {
    kd_build_axis(points, order, axis, lo, hi, 0);
}
/// MPAS: mpas_kd_tree.F:64-159 (recursive cyclic-axis construction).
/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_static.F:110-1422 at 1d6e370e (whole module driver boundary).
fn kd_build_axis(points: &[[f64; 3]], order: &mut [u32], axis: &mut [u8], lo: usize, hi: usize, ax: usize) {
    if hi <= lo { return; }
    let mid = lo + (hi - lo - 1) / 2;
    order[lo..hi].select_nth_unstable_by(mid-lo, |&a,&b| {
        (points[a as usize][ax] as f32).total_cmp(&(points[b as usize][ax] as f32))
    });
    axis[mid] = ax as u8;
    kd_build_axis(points, order, axis, lo, mid, (ax+1)%3);
    kd_build_axis(points, order, axis, mid+1, hi, (ax+1)%3);
}

impl KdTree {
    /// MPAS: mpas_kd_tree.F:64-159 (mpas_kd_build_tree).
    /// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_static.F:110-1422 at 1d6e370e (whole module driver boundary).
    fn new(mesh: &Mesh) -> Self {
        let points: Vec<[f64; 3]> = (0..mesh.n_cells)
            .map(|c| [mesh.x_cell[c], mesh.y_cell[c], mesh.z_cell[c]])
            .collect();
        let mut order: Vec<u32> = (0..mesh.n_cells as u32).collect();
        let mut axis = vec![0u8; order.len()];
        let n = order.len();
        kd_build(&points, &mut order, &mut axis, 0, n);
        Self { points, order, axis }
    }

    #[cfg(test)]
    /// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_static.F:110-1422 at 1d6e370e (whole module driver boundary).
    fn nearest(&self, target: [f64; 3]) -> (usize, f64) {
        self.nearest_bounded(target, f64::INFINITY).expect("nonempty KD tree")
    }

    /// MPAS: mpas_kd_tree.F:285-332: initialize distance to the supplied bound.
    /// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_static.F:110-1422 at 1d6e370e (whole module driver boundary).
    fn nearest_bounded(&self, target: [f64; 3], limit: f64) -> Option<(usize,f64)> {
        let mut best = (usize::MAX, limit);
        self.search(0, self.order.len(), target, &mut best);
        (best.0 != usize::MAX).then_some(best)
    }

    /// MPAS: mpas_kd_tree.F:226-279 (mpas_kd_search_internal).
    /// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_static.F:110-1422 at 1d6e370e (whole module driver boundary).
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
/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_static.F:110-1422 at 1d6e370e (whole module driver boundary).
fn xyz(radius: f64, lat: f64, lon: f64) -> [f64; 3] {
    let c = lat.cos();
    [radius * c * lon.cos(), radius * c * lon.sin(), radius * lat.sin()]
}

/// MPAS: src/operators/mpas_geometry_utils.F:1821-1868 (mpas_in_cell).
/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_static.F:110-1422 at 1d6e370e (whole module driver boundary).
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
/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_static.F:110-1422 at 1d6e370e (whole module driver boundary).
fn validate_supersample(factor: usize) -> MpasResult<()> {
    // MPAS: mpas_init_atm_static.F:1566-1572 permits an empty sampling
    // loop. Only the Rust allocation's address-space overflow is refused.
    if factor.checked_mul(factor).is_none() {
        return Err(MpasError::Refusal(format!(
            "supersample factor {factor} squared overflows the tile-map address space"
        )));
    }
    Ok(())
}

/// MPAS: mpas_init_atm_static.F:1456-1623 (init_atm_map_static_data).
/// Rust visits archive tiles directly; integer accumulations retain native words.
/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_static.F:110-1422 at 1d6e370e (whole module driver boundary).
fn tile_map(
    mesh: &Mesh,
    tree: &KdTree,
    max_distance2: f64,
    ds: &GeogDataset,
    tile: &TileRef,
    factor: usize,
) -> MpasResult<Vec<i32>> {
    validate_supersample(factor)?;
    let pixels = (ds.index.tile_x as usize)
        .checked_mul(ds.index.tile_y as usize)
        .and_then(|v| v.checked_mul(factor * factor))
        .ok_or_else(|| MpasError::Refusal("tile-map element count overflow".to_string()))?;
    let mut out = Vec::new();
    out.try_reserve_exact(pixels).map_err(|e| {
        MpasError::Refusal(format!(
            "host memory allocation failed for tile destination map: {pixels} i32 values: {e}"
        ))
    })?;
    for j in 0..ds.index.tile_y as usize {
        for i in 0..ds.index.tile_x as usize {
            for sj in 0..factor {
                for si in 0..factor {
                    let (lat, lon) = ds.tile_to_latlon(tile, i * factor + si, j * factor + sj, factor);
                    let p = crate::static_native::xyz(mesh.sphere_radius as f32, lat, lon).map(f64::from);
                    let Some((cell, d2)) = tree.nearest_bounded(p,max_distance2) else {
                        out.push(-1); continue;
                    };
                    // The native regional rule (init_atm_static, nBdyLayers=7):
                    // a pixel is accepted unconditionally at mask < 7, and on
                    // an outermost-ring cell only when it is geometrically
                    // INSIDE that cell -- otherwise its true owner was culled
                    // and the pixel belongs to nobody in this mesh.
                    let accepted = d2 < max_distance2
                        && (mesh.bdy_mask_cell[cell] != REGIONAL_OUTERMOST_MASK
                            || point_in_cell(mesh, cell, p));
                    out.push(if accepted { cell as i32 } else { -1 });
                }
            }
        }
    }
    Ok(out)
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
    /// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_static.F:110-1422 at 1d6e370e (whole module driver boundary).
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

    /// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_static.F:110-1422 at 1d6e370e (whole module driver boundary).
    fn bytes(&self) -> u64 {
        match self {
            CachedTileMap::Runs(r) => (r.len() * std::mem::size_of::<(i32, u32)>()) as u64,
            CachedTileMap::Raw(v) => (v.len() * 4) as u64,
        }
    }

    /// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_static.F:110-1422 at 1d6e370e (whole module driver boundary).
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
    /// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_static.F:110-1422 at 1d6e370e (whole module driver boundary).
    fn new(budget_bytes: u64) -> Self {
        Self {
            budget_bytes,
            inner: Mutex::new(TileMapCacheInner::default()),
        }
    }

    /// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_static.F:110-1422 at 1d6e370e (whole module driver boundary).
    fn lock(&self) -> MpasResult<std::sync::MutexGuard<'_, TileMapCacheInner>> {
        self.inner.lock().map_err(|_| {
            MpasError::Refusal(
                "the tile-map cache lock was poisoned by a panicking tile worker; \
                 the build cannot continue without knowing which tiles were mapped"
                    .to_string(),
            )
        })
    }

    /// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_static.F:110-1422 at 1d6e370e (whole module driver boundary).
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

    /// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_static.F:110-1422 at 1d6e370e (whole module driver boundary).
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

    /// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_static.F:110-1422 at 1d6e370e (whole module driver boundary).
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
    /// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_static.F:110-1422 at 1d6e370e (whole module driver boundary).
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
        let map = tile_map(self.mesh, self.tree, self.max_d2, ds, tile, factor)?;
        self.cache.insert(key, &map)?;
        Ok(map)
    }

    /// Run `body` over every tile, on the admitted number of workers.
    ///
    /// Failures are collected and the lowest-indexed one is returned, so the
    /// refusal a user reads names the same tile on every run rather than
    /// whichever worker happened to finish first.
    /// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_static.F:110-1422 at 1d6e370e (whole module driver boundary).
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
/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_static.F:110-1422 at 1d6e370e (whole module driver boundary).
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
/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_static.F:110-1422 at 1d6e370e (whole module driver boundary).
fn flush_i64(target: &[AtomicI64], index: usize, acc: i64) {
    if acc != 0 {
        target[index].fetch_add(acc, Ordering::Relaxed);
    }
}

#[inline]
// MPAS: mpas_init_atm_static.F:140,1512,1538,1772. Default INTEGER counts.
/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_static.F:110-1422 at 1d6e370e (whole module driver boundary).
fn flush_i32(target: &[AtomicI32], index: usize, acc: i32) {
    if acc != 0 {
        target[index].fetch_add(acc, Ordering::Relaxed);
    }
}

#[inline]
/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_static.F:110-1422 at 1d6e370e (whole module driver boundary).
fn plane_raw_at(ds: &GeogDataset, plane: &[i64], i: usize, j: usize) -> i64 {
    let b = ds.index.tile_bdr as usize;
    let nx = ds.index.full_tile_nx();
    plane[(j + b) * nx + (i + b)]
}

/// MPAS: mpas_init_atm_static.F:1899-2119,2126-2182 (category criteria,
/// accumulation, landuse/soil-category/Noah-MP texture entry points).
/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_static.F:110-1422 at 1d6e370e (whole module driver boundary).
fn categorical_fractions(
    ctx: &GeogContext<'_>,
    path: &Path,
    factor: usize,
    soil: bool,
) -> MpasResult<(Vec<i32>, Vec<f32>, i64, i64, GeogDataset)> {
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
    let cmax = ds.index.category_max.ok_or_else(|| {
        MpasError::Refusal(format!("{} lacks category_max", path.display()))
    })?;
    // MPAS: mpas_init_atm_static.F:2001-2011. A descending bound
    // allocates a zero-size histogram; MAXLOC then returns zero.
    let ncat = (cmax - cmin + 1).max(0) as usize;
    let counts = checked_vec_with(
        checked_len(mesh.n_cells, ncat, "categorical", "category counts")?,
        "categorical",
        "category counts",
        || AtomicI32::new(0),
    )?;

    ctx.for_each_tile(&ds.tiles, |tile| {
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
                let raw = plane_raw_at(&ds, &plane, i, j);
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

    let mut out =
        checked_vec_filled::<i32>(mesh.n_cells, cmin as i32, "categorical", "category field")?;
    let mut fractions=vec![0.;mesh.n_cells*ncat];
    for c in 0..mesh.n_cells {
        let row = &counts[c * ncat..(c + 1) * ncat];
        let row: Vec<i32> = row.iter().map(|v| v.load(Ordering::Relaxed)).collect();
        // Fork soil_sum is REAL; lu_sum is INTEGER. Keep both accumulation orders.
        let total=if soil {row.iter().fold(0f32,|a,&b|a+b as f32)} else {row.iter().fold(0i32,|a,&b|a.wrapping_add(b)) as f32};
        for (i,&count) in row.iter().enumerate(){fractions[c*ncat+i]=count as f32/total;}
        let best = crate::static_native::dominant_category(&row);
        // MPAS: mpas_init_atm_static.F:1842,1918,2163: MAXLOC returns
        // the first category even when every count is zero.
        out[c] = best.map_or(cmin-1,|best| cmin+best as i64) as i32;
    }
    Ok((out, fractions, cmin, cmax, ds))
}

/// RRFSx/MPAS-Model mpas_init_atm_static.F:2288-2351 at 1d6e370e.
/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_static.F:110-1422 at 1d6e370e (whole module driver boundary).
fn categorical(ctx:&GeogContext<'_>,path:&Path,factor:usize)->MpasResult<(Vec<i32>,i64,i64,GeogDataset)> {
    let(a,_,b,c,d)=categorical_fractions(ctx,path,factor,false)?;Ok((a,b,c,d))
}

// Kept in the receipt ABI. MPAS uses field-specific fixed fills and never
// propagates a cell mean across neighboring cells.
/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_static.F:110-1422 at 1d6e370e (whole module driver boundary).
fn take_spread_rounds() -> usize { 0 }

/// MPAS: mpas_init_atm_static.F:1668-1676,736-879,972-1113,488-640.
#[derive(Clone, Copy)]
enum MeanPolicy { Terrain, Green, Albedo, Snow, Lai }

/// Integer accumulation followed by the RKIND assignment in each native driver.
/// MPAS: mpas_init_atm_static.F:1721-1783,736-879,972-1113,488-640.
/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_static.F:110-1422 at 1d6e370e (whole module driver boundary).
fn continuous_mean(
    ctx: &GeogContext<'_>, path: &Path, factor: usize,
    landmask: Option<&[i32]>, policy: MeanPolicy,
) -> MpasResult<(Vec<f32>, GeogDataset)> {
    let mesh = ctx.mesh;
    let ds = GeogDataset::open(path)?;
    if ds.index.kind != SourceKind::Continuous {
        return Err(MpasError::Refusal(format!("{} must be continuous", path.display())));
    }
    let nz = ds.index.nz();
    let slots = checked_len(mesh.n_cells, nz, "continuous", "plane accumulator")?;
    let sums = checked_vec_with(slots, "continuous", "plane accumulator", || AtomicI64::new(0))?;
    // MPAS: mpas_init_atm_static.F:140,566,587,816,835,1050,1068,1512.
    let count = checked_vec_with(slots, "continuous", "sample counts", || AtomicI32::new(0))?;
    ctx.for_each_tile(&ds.tiles, |tile| {
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
                    let mut raw = plane_raw_at(&ds, &plane, i, j);
                    // MPAS: mpas_init_atm_static.F:564,585 converts snow's
                    // missing value to I8KIND; monthly fields compare RKIND
                    // directly at 809,828,1043,1061.
                    let missing = match policy {
                        MeanPolicy::Green | MeanPolicy::Albedo | MeanPolicy::Lai => ds.index.missing_value_rkind == Some(raw as f32),
                        _ => ds.index.missing_value == Some(raw),
                    };
                    if missing {
                        match policy {
                            MeanPolicy::Snow => continue,
                            MeanPolicy::Green | MeanPolicy::Lai => raw = 0,
                            MeanPolicy::Albedo => raw = 8,
                            MeanPolicy::Terrain => {},
                        }
                    }
                    let pixel = j * ds.index.tile_x as usize + i;
                    for q in 0..sub {
                        let c = map[pixel * sub + q];
                        if c < 0 || landmask.map_or(false, |m| if matches!(policy,MeanPolicy::Lai) {m[c as usize]!=1} else {m[c as usize]==0}) { continue; }
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
    })?;
    let mut out = checked_vec::<f32>(slots, "continuous", "destination field")?;
    for c in 0..mesh.n_cells {
        if landmask.map_or(false, |m| if matches!(policy,MeanPolicy::Lai) {m[c]!=1} else {m[c]==0}) {
            // MPAS: mpas_init_atm_static.F:1104-1110: no accumulated
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

/// MPAS: mpas_init_atm_static.F:1691-1703,1642-1718 (soilcomp criteria/accumulation).
/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_static.F:110-1422 at 1d6e370e (whole module driver boundary).
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

    ctx.for_each_tile(&ds.tiles, |tile| {
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

/// MPAS: mpas_init_atm_static.F:2172-2285: tiled mosaic includes the
/// original three-pixel halo, followed by MPAS's interpolation chain.
/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_static.F:110-1422 at 1d6e370e (whole module driver boundary).
fn climatology_plane(ds: &GeogDataset, z: usize) -> MpasResult<crate::static_native::Plane> {
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

/// MPAS: mpas_init_atm_static.F:2257-2279; mpas_init_atm_llxy.F:1389-1416.
/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_static.F:110-1422 at 1d6e370e (whole module driver boundary).
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

/// MPAS: mpas_init_atm_static.F:2172-2285 (interp_soiltemp).
/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_static.F:110-1422 at 1d6e370e (whole module driver boundary).
fn soil_temperature(mesh: &Mesh, path: &Path, landmask: &[i32]) -> MpasResult<(Vec<f32>, GeogDataset)> {
    let ds = GeogDataset::open(path)?;
    soil_temperature_dataset(mesh,ds,landmask)
}

/// MPAS: mpas_init_atm_static.F:2214-2285.
/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_static.F:110-1422 at 1d6e370e (whole module driver boundary).
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

/// MPAS: mpas_init_atm_static.F:653-735,879-959,1113-1198.
/// The driver initializes interp_list only in the NCEP snow branch (687-691).
/// With MODIS snow its NCEP monthly calls (943,1177) use an unset method list;
/// mpas_init_atm_hinterp.F:151-231 leaves the result unset for unknown methods.
/// The pinned GNU strict binary returns the projected y before latitude clamps
/// in that path: the method dispatcher does not change the SSE return register.
/// Whole-driver repeats/ranks/host witnesses are recorded in the lane receipts.
/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_static.F:110-1422 at 1d6e370e (whole module driver boundary).
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

/// MPAS: mpas_init_atm_static.F:653-735,879-959,1113-1198.
/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_static.F:110-1422 at 1d6e370e (whole module driver boundary).
fn open_stage_dataset(cfg: &StaticBuildConfig, field: &str, path: &Path) -> MpasResult<GeogDataset> {
    match field {
        "greenfrac" if cfg.stages.vegetation_ncep => GeogDataset::open_native_climatology(path,true),
        "albedo12m" if cfg.stages.albedo_ncep => GeogDataset::open_native_climatology(path,true),
        "snoalb" if cfg.stages.snow_ncep => GeogDataset::open_native_climatology(path,false),
        "soiltemp" if cfg.stages.native_init => GeogDataset::open_native_soil_temperature(path),
        _ => GeogDataset::open(path),
    }
}

/// MPAS: mpas_init_atm_static.F:501-653 (MODIS snow driver).
/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_static.F:110-1422 at 1d6e370e (whole module driver boundary).
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

/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_static.F:110-1422 at 1d6e370e (whole module driver boundary).
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

/// The sampling rate each CATEGORICAL dataset gets for one mesh.
///
/// Derived once and used in two places -- the memory the admission gate plans
/// against and the passes that spend it -- because a factor computed twice is
/// two things to keep equal, and the one that would drift is the one the
/// budget was granted for.
#[derive(Debug, Clone, Copy)]
struct CategoricalFactors {
    landuse: usize,
    soilcat: usize,
    soilcl: [usize; 4],
}

/// MPAS: mpas_init_atm_static.F:417-460,1199-1250.
/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_static.F:110-1422 at 1d6e370e (whole module driver boundary).
fn categorical_factors(_mesh: &Mesh, cfg: &StaticBuildConfig) -> MpasResult<CategoricalFactors> {
    validate_supersample(cfg.supersample_landuse)?;
    validate_supersample(cfg.supersample_30s)?;
    Ok(CategoricalFactors {
        landuse: cfg.supersample_landuse,
        soilcat: cfg.supersample_30s,
        soilcl: [cfg.supersample_30s; 4],
    })
}

/// The resident, per-worker and cache terms the admission gate plans against.
/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_static.F:110-1422 at 1d6e370e (whole module driver boundary).
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
    let cat = categorical_factors(mesh, cfg)?;
    let specs: [(&PathBuf, usize, usize); 11] = [
        (&g.terrain, cfg.supersample_30s, 1),
        (&g.landuse, cat.landuse, 1),
        (&g.soilcat, cat.soilcat, 1),
        (&g.greenfrac, cfg.supersample_30s, 1),
        (&g.albedo, cfg.supersample, 1),
        (&g.snow_albedo, cfg.supersample, 1),
        (&g.soilcomp, cfg.supersample_30s, soilcomp_planes),
        (&g.soilcl1, cat.soilcl[0], 1),
        (&g.soilcl2, cat.soilcl[1], 1),
        (&g.soilcl3, cat.soilcl[2], 1),
        (&g.soilcl4, cat.soilcl[3], 1),
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


/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_static.F:110-1422 at 1d6e370e (whole module driver boundary).
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
/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_static.F:110-1422 at 1d6e370e (whole module driver boundary).
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
        "lai12m","landusef","soilf",
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
    lai12m: Vec<f32>,
    landusef: Vec<f32>,
    soilf: Vec<f32>,
    ter: Vec<f32>,
    landmask: Vec<i32>,
    ivgtyp: Vec<i32>,
    isltyp: Vec<i32>,
    /// `[cell][month]`, in PERCENT. See [`GREENFRAC_MODEL_SCALE`].
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
/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_static.F:110-1422 at 1d6e370e (whole module driver boundary).
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
        NcDim::fixed("nlcat",geo.landusef.len()/mesh.n_cells),
        NcDim::fixed("nscat",geo.soilf.len()/mesh.n_cells),
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
            .with_attrs(vec![NcAttr::text("units", "-"), NcAttr::text("long_name", long)])
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
        int("nEdgesOnEdge", vec![d_e], "Number of edges involved in reconstruction"),
        int("edgesOnCell", vec![d_c, d_me], "IDs of edges forming the boundary of a cell"),
        int("edgesOnEdge", vec![d_e, d_me2], "IDs of edges used in reconstruction"),
        flt("weightsOnEdge", vec![d_e, d_me2], "-", "Weights used in reconstruction"),
        flt("dvEdge", vec![d_e], "m", "Spherical distance between vertices"),
        flt("dcEdge", vec![d_e], "m", "Spherical distance between cells"),
        flt("angleEdge", vec![d_e], "rad", "Angle between local north and the edge normal"),
        flt("areaCell", vec![d_c], "m^2", "Spherical area of a Voronoi cell"),
        flt("areaTriangle", vec![d_v], "m^2", "Spherical area of a Delaunay triangle"),
        int("cellsOnCell", vec![d_c, d_me], "IDs of cells neighbouring a cell"),
        int("verticesOnCell", vec![d_c, d_me], "IDs of vertices bounding a cell"),
        int("verticesOnEdge", vec![d_e, d_two], "IDs of the vertices bounding an edge"),
        int("edgesOnVertex", vec![d_v, d_vd], "IDs of the edges meeting at a vertex"),
        int("cellsOnVertex", vec![d_v, d_vd], "IDs of the cells meeting at a vertex"),
        flt("kiteAreasOnVertex", vec![d_v, d_vd], "m^2", "Intersection area of a cell and a triangle"),
        flt("meshDensity", vec![d_c], "-", "Value of the density function used to generate the mesh"),
        NcVarDef::new("nominalMinDc", NcType::Float, vec![]).with_attrs(vec![
            NcAttr::text("units", "m"),
            NcAttr::text("long_name", "Nominal minimum dcEdge value where meshDensity == 1.0"),
        ]),
        int("bdyMaskCell", vec![d_c], "Limited-area boundary relaxation zone of a cell"),
        int("bdyMaskEdge", vec![d_e], "Limited-area boundary relaxation zone of an edge"),
        int("bdyMaskVertex", vec![d_v], "Limited-area boundary relaxation zone of a vertex"),
        flt("edgeNormalVectors", vec![d_e, d_r3], "-", "Cartesian components of the edge normal"),
        flt("localVerticalUnitVectors", vec![d_c, d_r3], "-", "Cartesian components of the local vertical"),
        flt("cellTangentPlane", vec![d_c, d_two, d_r3], "-", "Cartesian basis of a cell's tangent plane"),
        flt(
            "coeffs_reconstruct",
            vec![d_c, d_me, d_r3],
            "-",
            "Coefficients for reconstructing a cell-centre vector from its \
             edge normal components (bitwise +0 placeholder; a consumer \
             overlays the exact values from the init file)",
        ),
        flt(
            "deriv_two",
            vec![d_e, d_two, d_fifteen],
            "unitless",
            "weights for cell-centered second derivative, normal to edge, for \
             transport scheme",
        ),
        flt("fEdge", vec![d_e], "s^-1", "Coriolis parameter at an edge"),
        flt("fVertex", vec![d_v], "s^-1", "Coriolis parameter at a vertex"),
        flt("ter", vec![d_c], "m", "Terrain height"),
        int("landmask", vec![d_c], "Land mask, 1 for land and 0 for water"),
        NcVarDef::new("mminlu", NcType::Char, vec![d_str])
            .with_attrs(vec![NcAttr::text("long_name", "Land use dataset")]),
        int("ivgtyp", vec![d_c], "Dominant land use category"),
        int("isltyp", vec![d_c], "Dominant soil category"),
        flt("snoalb", vec![d_c], "-", "Annual maximum snow albedo"),
        flt("soiltemp", vec![d_c], "K", "Deep-soil temperature"),
        flt("greenfrac", vec![d_c, d_mon], "-", "Monthly climatological green-ness fraction"),
        flt("shdmin", vec![d_c], "-", "Minimum climatological green-ness fraction"),
        flt("shdmax", vec![d_c], "-", "Maximum climatological green-ness fraction"),
        flt("albedo12m", vec![d_c, d_mon], "-", "Monthly climatological background albedo"),
        flt("lai12m", vec![d_c,d_mon],"unitless","Monthly climatological leaf area index"),
        flt("landusef", vec![d_c,d("nlcat")],"unitless","Land use category fractions"),
        flt("soilf", vec![d_c,d("nscat")],"unitless","Soil category fractions"),
    ];
    for (name, long) in [
        ("var2d", "Standard deviation of sub-grid orography"),
        ("con", "Orographic convexity"),
        ("oa1", "Orographic asymmetry, westward"),
        ("oa2", "Orographic asymmetry, southward"),
        ("oa3", "Orographic asymmetry, south-westward"),
        ("oa4", "Orographic asymmetry, north-westward"),
        ("ol1", "Orographic effective length, westward"),
        ("ol2", "Orographic effective length, southward"),
        ("ol3", "Orographic effective length, south-westward"),
        ("ol4", "Orographic effective length, north-westward"),
    ] {
        vars.push(flt(name, vec![d_c], "-", long));
    }
    // The thirteen this builder adds beyond the published manifest.
    vars.extend([
        flt("cell_gradient_coef_x", vec![d_c, d_me], "m^-1", "Zonal cell-gradient weights"),
        flt("cell_gradient_coef_y", vec![d_c, d_me], "m^-1", "Meridional cell-gradient weights"),
        flt("defc_a", vec![d_c, d_me], "unitless", "Deformation weights, symmetric part"),
        flt("defc_b", vec![d_c, d_me], "unitless", "Deformation weights, antisymmetric part"),
        int("lu_index", vec![d_c], "Dominant land use category, WRF spelling of ivgtyp"),
        int("soilcat_top", vec![d_c], "Dominant top-layer soil category, WRF spelling of isltyp"),
        flt("soilcomp", vec![d_c, d_soil], "-", "Soil composition by depth"),
        flt("soilcl1", vec![d_c], "-", "Soil texture class, layer 1"),
        flt("soilcl2", vec![d_c], "-", "Soil texture class, layer 2"),
        flt("soilcl3", vec![d_c], "-", "Soil texture class, layer 3"),
        flt("soilcl4", vec![d_c], "-", "Soil texture class, layer 4"),
        NcVarDef::new("isice_lu", NcType::Int, vec![])
            .with_attrs(vec![NcAttr::text("units", "-"), NcAttr::text("long_name", "Ice land-use category")]),
        NcVarDef::new("iswater_lu", NcType::Int, vec![])
            .with_attrs(vec![NcAttr::text("units", "-"), NcAttr::text("long_name", "Water land-use category")]),
    ]);

    // MPAS: Registry.xml:498-585,882-903. Native init's immutable static
    // stream follows active packages and has no WRF aliases or SW weights.
    if cfg.stages.native_init {
        vars.retain(|v|native_static_field_enabled(&v.name,&cfg.stages));
        vars.insert(1,NcVarDef::new("Time",NcType::Float,vec![d_time]).with_attrs(vec![
            NcAttr::text("units",format!("seconds since {}",cfg.valid_time.replace('_'," "))),
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
        NcAttr::text("on_a_sphere", "YES"),
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
        NcAttr::text("rw_mpas_static_schema", UNIFIED_SCHEMA_TAG),
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
    w.put("lai12m",NcData::Floats(&geo.lai12m))?;
    w.put("landusef",NcData::Floats(&geo.landusef))?;
    w.put("soilf",NcData::Floats(&geo.soilf))?;
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
    w.put("defc_a", NcData::Floats(&operators.defc_a))?;
    w.put("defc_b", NcData::Floats(&operators.defc_b))?;
    if !cfg.stages.native_init {
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
/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_static.F:110-1422 at 1d6e370e (whole module driver boundary).
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
                sum = (sum as f32 + mesh.dc_edge[e] as f32) as f64;
            }
        }
        out[c] = (sum as f32 / k as f32) as f64;
    }
    out
}

/// Build a static, silently. See [`build_static_reporting`].
/// MPAS: mpas_init_atm_static.F:110-1422 (init_atm_static).
/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_static.F:110-1422 at 1d6e370e (whole module driver boundary).
pub fn build_static(cfg: &StaticBuildConfig) -> MpasResult<StaticBuildReceipt> {
    build_static_reporting(cfg, &|_| {})
}

/// Build a static, streaming one tab-separated progress line per stage.
///
/// The grammar is the door's: `GRID`, `NOMINALDX`, `VECTORS`, `OPERATORS`,
/// `GEOGGRID`, `GWDBAND`. A wrapper parses these, so a token that stops being
/// printed is a wrapper that stops reporting rather than a wrapper that fails.
/// MPAS: mpas_init_atm_static.F:110-1422 (complete static driver boundary).
/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_static.F:110-1422 at 1d6e370e (whole module driver boundary).
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
    let missing: Vec<_> = cfg.geog.slots().iter()
        .filter(|(name,path)| (cfg.stages.noahmp || !name.starts_with("soilcl") && *name != "soilcomp")
            && !(*name=="greenfrac" && cfg.stages.vegetation_ncep)
            && !(*name=="albedo" && cfg.stages.albedo_ncep)
            && !(*name=="snow_albedo" && cfg.stages.snow_ncep)
            && !(*name=="soil_temperature" && cfg.stages.native_init)
            && !path.join("index").is_file())
        .map(|(_,path)| path.display().to_string()).collect();
    if !missing.is_empty() {
        return Err(MpasError::Refusal(format!(
            "the geography archive is missing {} dataset(s) a static needs: {}. \
             Terrain, land use, soil class and composition, deep-soil \
             temperature, green-ness and albedo all come from these \
             directories, and a static written without one would hand the model \
             a field of zeros that the run would carry as real geography.\n\
             what does work:\n  \
             gpuwm fetch-geog --datasets all, which stages every one of them\n  \
             --geog DIR pointing at a complete archive",
            missing.len(),
            missing.join(", ")
        )));
    }
    let mesh = Mesh::read(&cfg.grid_path)?;
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
    let gwd_band_rows=0;
    let native_gwd_bytes=32u64*1200*1200*3+64u64*1024*1024;
    let plan=plan_tile_parallelism("preflight",resident,per_worker,
        if cfg.stages.gwd {writer_slab.max(native_gwd_bytes)} else {writer_slab},
        1024*1024*1024,resolve_tile_workers(cfg.tile_workers),cache_request,cfg.host_memory_limit_bytes)?;
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
            rayon::ThreadPoolBuilder::new()
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

    // The sampling rate every CATEGORICAL source needs for THIS mesh, derived
    // from the mesh's own smallest cell.  Same function the admission gate
    // planned the memory with, so the pass cannot spend more than was granted.
    let cat_factors = categorical_factors(&mesh, &cfg)?;

    let mut datasets = Vec::new();
    let mut time_field = |field: &str, path: &Path, factor: usize, f: &mut dyn FnMut() -> MpasResult<(u64, u64)>| -> MpasResult<()> {
        let started = Instant::now();
        memory.event(field, Some(path), "before dataset", None, None, None, vec![]);
        take_spread_rounds();
        let (plane_bytes, dest_bytes) = f()?;
        let spread_rounds = take_spread_rounds();
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
            spread_rounds,
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

    let mut landusef=Vec::new();
    let mut soilf=Vec::new();
    let mut ivgtyp = Vec::new();
    let mut landuse_ds_opt = None;
    time_field("landuse", &cfg.geog.landuse, cat_factors.landuse, &mut || {
        let (v, frac, _, _, ds) = categorical_fractions(&ctx, &cfg.geog.landuse, cat_factors.landuse,false)?;
        landusef=frac;
        ivgtyp = v;
        let pb = ds.plane_bytes();
        landuse_ds_opt = Some(ds);
        Ok((pb, (ivgtyp.len() * 4) as u64))
    })?;
    let landuse_ds = landuse_ds_opt.take().expect("landuse populated");
    let iswater_lu = landuse_ds.index.iswater.unwrap_or(17) as i32;
    let isice_lu = landuse_ds.index.isice.unwrap_or(15) as i32;
    let mut iswater_soil = 14;
    let mut isltyp = Vec::new();
    time_field("soilcat_top", &cfg.geog.soilcat, cat_factors.soilcat, &mut || {
        let (v, frac, _, _, ds) = categorical_fractions(&ctx, &cfg.geog.soilcat, cat_factors.soilcat,true)?;
        soilf=frac;
        isltyp = v;
        iswater_soil = ds.index.isoilwater.unwrap_or(14) as i32;
        Ok((ds.plane_bytes(), (isltyp.len() * 4) as u64))
    })?;

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
        // MPAS: mpas_init_atm_static.F:873-876: raw MODIS values are percent.
        greenfrac = v;
        Ok((ds.plane_bytes(), (greenfrac.len() * 4) as u64))
    })?;

    let mut lai12m=Vec::new();
    time_field("lai12m",&cfg.geog.lai,1,&mut || {
        let(v,ds)=continuous_mean(&ctx,&cfg.geog.lai,1,Some(&landmask),MeanPolicy::Lai)?;
        if ds.index.nz()!=12 {return Err(MpasError::Refusal("MODIS LAI must carry twelve monthly planes".into()));}
        lai12m=v;Ok((ds.plane_bytes(),(lai12m.len()*4) as u64))
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
        // MPAS: mpas_init_atm_static.F:1116-1120.
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
        time_field(&field, paths[idx], cat_factors.soilcl[idx], &mut || {
            let (v, _, _, ds) = categorical(&ctx, paths[idx], cat_factors.soilcl[idx])?;
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
    let gwd_bytes=if cfg.stages.gwd {native_gwd_bytes} else {0};
    memory.event(
        "gwd",
        Some(&cfg.geog.terrain),
        "before native fixed-format tile statistics",
        None,
        None,
        Some(gwd_bytes),
        vec![BufferLedger {
            name: "native fixed-format terrain/USGS tile cache and cell box".into(),
            bytes: gwd_bytes,
            lifetime: "GWD stage".to_string(),
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
    } else {
        let root=cfg.geog.terrain.parent().ok_or_else(||MpasError::Refusal("terrain directory has no geography root".into()))?;
        crate::rrfs_static_gwd::compute(&cfg.geog.terrain,&root.join("landuse_30s"),
            cfg.gwd_topo_shift,cfg.gwd_cell_scaling,&mesh.lat_cell,&mesh.lon_cell,
            &dc_m,&mesh.n_edges_on_cell,&mesh.cells_on_cell,mesh.max_edges,progress)?
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
        schema: UNIFIED_SCHEMA_TAG.to_string(),
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
        mminlu: cfg.stages.landuse_label.as_ref().map(|label| if label == "MODIFIED_IGBP_MODIS_NOAH_15s" {"MODIFIED_IGBP_MODIS_NOAH".to_string()} else {label.clone()}).unwrap_or_else(||landuse_ds.index.mminlu.clone()),
    };
    let provenance_json = serde_json::to_string(&provenance)
        .map_err(|e| MpasError::Refusal(format!("cannot serialise the provenance: {e}")))?;

    let geo = GeographyFields {
        lai12m,landusef,soilf,
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
        mminlu: cfg.stages.landuse_label.as_ref().map(|label| if label == "MODIFIED_IGBP_MODIS_NOAH_15s" {"MODIFIED_IGBP_MODIS_NOAH".to_string()} else {label.clone()}).unwrap_or_else(||landuse_ds.index.mminlu.clone()),
    };

    memory.event("write", None, "before static write", None, None, None, vec![]);
    write_static(cfg, &mesh, &operators, &geo, &gwd, nominal_dx_m, &provenance_json)?;
    memory.event("write", None, "static fsync complete", None, None, None, vec![]);

    let output_bytes = std::fs::metadata(&cfg.out_path)?.len();
    let generated_fields = vec![
        "cell_gradient_coef_x", "cell_gradient_coef_y", "defc_a", "defc_b",
        "deriv_two", "fEdge", "fVertex", "ter", "landmask", "lu_index", "ivgtyp",
        "lai12m","landusef","soilf",
        "soilcat_top", "isltyp", "greenfrac", "albedo12m", "shdmin", "shdmax",
        "snoalb", "soiltemp", "soilcomp", "soilcl1", "soilcl2", "soilcl3", "soilcl4",
        "var2d", "con", "oa1", "oa2", "oa3", "oa4", "ol1", "ol2", "ol3", "ol4",
    ]
    .into_iter()
    .filter(|name| !cfg.stages.native_init || native_static_field_enabled(name,&cfg.stages))
    .map(str::to_string)
    .collect();

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
        schema: "rw-mpas-static.build/v3-unified".to_string(),
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

/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_static.F:110-1422 at 1d6e370e (whole module driver boundary).
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
    attrs.push(NcAttr::text("on_a_sphere", "YES"));
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
/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_static.F:110-1422 at 1d6e370e (whole module driver boundary).
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
    let mesh = Mesh::read(&cfg.grid_path)?;
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
        Some(rayon::ThreadPoolBuilder::new().num_threads(plan.workers).build()
            .map_err(|e| MpasError::Refusal(format!("cannot start terrain tile workers: {e}")))?)
    } else { None };
    let ctx = GeogContext {
        mesh: &mesh, tree: &tree, max_d2: mesh.max_search_distance2(),
        workers: plan.workers, pool: pool.as_ref(), cache: &cache,
    };
    let started = Instant::now();
    take_spread_rounds();
    let (terrain, dataset) = continuous_mean(&ctx, &cfg.geog.terrain, cfg.supersample_30s, None, MeanPolicy::Terrain)?;
    let spread_rounds = take_spread_rounds();
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
            elapsed_seconds: elapsed, spread_rounds,
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
