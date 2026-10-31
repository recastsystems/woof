//! The case-7 initial-state builder.
//!
//! Reads a WPS intermediate file, a mesh, a statics capsule and a
//! vertical-metric capsule, and writes an MPAS initial-conditions file: the
//! whole of `init_atm_case_gfs`'s meteorology plus
//! `mpas_atmphys_initialize_real.F`'s soil and surface companions, in Rust.
//! MPAS v8.4.2 core_init_atmosphere/mpas_init_atm_cases.F:2655-5417;
//! core_atmosphere/physics/mpas_atmphys_initialize_real.F:131-513,586-729.
//!
//! The front door builds vertical fields through [`vertical`] and supplies
//! the resulting artifact contract through [`capsule`]. Its receipt names
//! the producer and the separate native identity evidence.
//!
//! Indexed loops are used throughout rather than iterator chains: every loop
//! here walks several parallel arrays at once against a Fortran original whose
//! index arithmetic is the thing being reproduced, and rewriting them as zips
//! would make the correspondence unreadable at exactly the places it matters.
#![allow(clippy::needless_range_loop)]

pub mod aerosols;
pub mod capsule;
pub mod dynamics;
pub mod emit;
pub mod hinterp;
pub mod metfile;
pub mod soil;
pub mod ruc;
pub mod aerosol;
pub mod surface;
mod surface_met;
pub mod vinterp;
pub mod vertical;
pub mod advection;
pub mod boundary;
pub mod projection;
pub mod reconstruction;
pub mod auxiliary;
pub mod artifact;
pub mod cold_artifact;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use rayon::prelude::*;

use crate::error::{MpasError, MpasResult};
use dynamics::VirtualFactor;
use hinterp::{InterpContext, Method, Slab, Underflow};
#[cfg(test)]
use hinterp::LatLonProjection;
use metfile::MetSlab;
use soil::DeepMoisture;
use vinterp::{sorted_column, vertical_interp, Extrap, SURFACE_LEVEL_TAG, SURFACE_SENTINEL};

/// Radians to degrees, derived the way `mpas_init_atm_llxy.F` derives it.
///
/// This is **not** the correctly rounded `180/pi`.  The Fortran declares
/// `REAL(KIND=RKIND), PARAMETER :: PI = 3.141592653589793` and then
/// `DEG_PER_RAD = 180./PI`, so in the single-precision build the division is
/// performed on `f32(pi)` and the result lands one ULP *below* the correctly
/// rounded constant: `0x42652ee0` (57.295776367) rather than `0x42652ee1`
/// (57.295780182).
///
/// One ULP here is not cosmetic.  It is the constant that turns a mesh cell's
/// latitude and longitude into a position on the first-guess grid, so a 6.7e-8
/// relative difference displaces every interpolation point by up to 1.22e-4 of
/// a grid cell at longitudes near 360 degrees.  Over most of the mesh that is
/// a sub-ULP change in the interpolated value; at the roughly twenty cells per
/// global mesh whose position lands *within that band of a grid line*, it
/// changes `floor(x)` and therefore selects an entirely different four- or
/// sixteen-point stencil, which moves the value by as much as the field varies
/// between neighbouring source points.
///
/// Writing the constant as a decimal literal picked the accurate value and
/// produced exactly that: a systematic -6.6e-5 grid-cell shift plus a handful
/// of cells sampling the wrong stencil.  It is derived rather than typed here
/// so the port stands where the reference stands.  See
/// `deg_per_rad_is_the_fortran_parameter_not_the_accurate_one`.
pub(crate) const DEG_PER_RAD: f32 = 180.0f32 / std::f32::consts::PI;
const MSGVAL: f32 = -1.0e30;

/// Which first-guess array a met field feeds, and how it is sampled.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Target {
    /// A cell-indexed 3-D field, by level.
    T,
    Rh,
    Sh,
    Z,
    P,
    /// An edge-indexed 3-D wind component, by level.
    U,
    V,
    /// Cell-indexed surface scalars.
    Psfc,
    Pmsl,
    SoilHgt,
    Skintemp,
    Snow,
    Seaice,
    /// Soil temperature / moisture at a fixed first-guess layer.
    SoilT(usize, f32),
    SoilM(usize, f32),
    NodeT(u16),
    NodeM(u16),
    Number(&'static str),
    /// A first-guess hydrometeor mass or number, by its MPAS scalar name.
    /// Read only under [`InitConfig::first_guess_hydrometeors`].
    Species(&'static str),
}

struct FieldRule {
    target: Target,
    methods: &'static [Method],
    /// `Some((maskval, masked, fillval))` when the land-sea mask applies.
    mask: Option<(f32, i32, f32)>,
}

const PROFILE_METHODS: &[Method] = &[Method::SixteenPoint, Method::Search];
const SOIL_T_METHODS: &[Method] = &[
    Method::SixteenPoint,
    Method::FourPoint,
    Method::WtAverage4,
    Method::Search,
];
const SOIL_M_METHODS: &[Method] = &[Method::FourPoint, Method::WtAverage4, Method::Search];
const SNOW_METHODS: &[Method] = &[Method::FourPoint, Method::WtAverage4];
const SEAICE_METHODS: &[Method] = &[Method::FourPoint, Method::WtAverage4, Method::Search];

/// The dispatch table out of `init_atm_case_gfs`, one row per field name.
///
/// The soil rows carry the layer index and the layer *thickness in
/// centimetres* the Fortran assigns alongside them.  Only the thickness
/// survives: `init_soil_layers_depth` recomputes the mid-depths cumulatively
/// and overwrites whatever the dispatch put in `zs_fg`.
fn rule_for(field: &str) -> Option<FieldRule> {
    if field.len() == 8 && (field.starts_with("ST") || field.starts_with("SM")) {
        if let (Ok(a), Ok(b)) = (field[2..5].parse::<u16>(), field[5..8].parse::<u16>()) {
            if a == b {
                return Some(FieldRule { target: if field.starts_with("ST") { Target::NodeT(a) } else { Target::NodeM(a) },
                    methods: if field.starts_with("ST") { SOIL_T_METHODS } else { SOIL_M_METHODS },
                    mask: Some((0.0, 0, if field.starts_with("ST") {285.0} else {1.0})) });
            }
        }
    }
    let profile = |t: Target| FieldRule {
        target: t,
        methods: PROFILE_METHODS,
        mask: None,
    };
    let soil_t = |k: usize, dz: f32| FieldRule {
        target: Target::SoilT(k, dz),
        methods: SOIL_T_METHODS,
        mask: Some((0.0, 0, 285.0)),
    };
    let soil_m = |k: usize, dz: f32| FieldRule {
        target: Target::SoilM(k, dz),
        methods: SOIL_M_METHODS,
        mask: Some((0.0, 0, 1.0)),
    };
    Some(match field {
        "QNWFA" => profile(Target::Number("nwfa")),
        "QNIFA" => profile(Target::Number("nifa")),
        "TT" => profile(Target::T),
        "RH" => profile(Target::Rh),
        "SPECHUMD" => profile(Target::Sh),
        "GHT" => profile(Target::Z),
        "PRES" | "PRESSURE" => profile(Target::P),
        "UU" => profile(Target::U),
        "VV" => profile(Target::V),
        "PSFC" => profile(Target::Psfc),
        "PMSL" => profile(Target::Pmsl),
        "SOILHGT" => profile(Target::SoilHgt),
        "SKINTEMP" => profile(Target::Skintemp),
        "SNOW" => FieldRule {
            target: Target::Snow,
            methods: SNOW_METHODS,
            mask: None,
        },
        "SEAICE" => FieldRule {
            target: Target::Seaice,
            methods: SEAICE_METHODS,
            mask: Some((1.0, 1, 0.0)),
        },
        "ST000010" => soil_t(0, 10.0),
        "ST010200" => soil_t(1, 190.0),
        "ST010040" => soil_t(1, 30.0),
        "ST040100" => soil_t(2, 60.0),
        "ST100200" => soil_t(3, 100.0),
        "ST000007" => soil_t(0, 7.0),
        "ST007028" => soil_t(1, 21.0),
        "ST028100" => soil_t(2, 72.0),
        "ST100255" => soil_t(3, 155.0),
        "ST100289" => soil_t(3, 189.0),
        "SOILT001" => soil_t(0, 1.0),
        "SOILT002" => soil_t(1, 2.0),
        "SOILT006" => soil_t(2, 6.0),
        "SOILT018" => soil_t(3, 18.0),
        "SOILT054" => soil_t(4, 54.0),
        "SOILT162" => soil_t(5, 162.0),
        "SOILT486" => soil_t(6, 486.0),
        "SOILT999" => soil_t(7, 1458.0),
        "SM000010" => soil_m(0, 10.0),
        "SM010200" => soil_m(1, 190.0),
        "SM010040" => soil_m(1, 30.0),
        "SM040100" => soil_m(2, 60.0),
        "SM100200" => soil_m(3, 100.0),
        "SM000007" => soil_m(0, 7.0),
        "SM007028" => soil_m(1, 21.0),
        "SM028100" => soil_m(2, 72.0),
        "SM100255" => soil_m(3, 155.0),
        "SM100289" => soil_m(3, 189.0),
        "SOILM001" => soil_m(0, 1.0),
        "SOILM002" => soil_m(1, 2.0),
        "SOILM006" => soil_m(2, 6.0),
        "SOILM018" => soil_m(3, 18.0),
        "SOILM054" => soil_m(4, 54.0),
        "SOILM162" => soil_m(5, 162.0),
        "SOILM486" => soil_m(6, 486.0),
        "SOILM999" => soil_m(7, 1458.0),
        _ => return None,
    })
}

/// The first guess's own hydrometeors and number concentrations, by
/// intermediate field name, and the MPAS scalar each one fills.
///
/// MPAS-A v8.4.2 `init_atm_case_gfs` has no row for any of these, so native
/// case 7 ignores them and its hour zero carries no condensate whatever the
/// first guess holds.  The RRFS fork reads all of them (its case 7/9 scalar
/// table, `crate::rrfs_init::scalars::input_index`).  This port reads them
/// only when [`InitConfig::first_guess_hydrometeors`] says so; off, the table
/// is the native one and they land in `met_records_ignored` exactly as before.
///
/// THE BREAKAGE THIS PREVENTS (hex-meshes-sources review, 2026-10-29): a hex
/// run on MPAS-A physics started from HRRR 2026-10-28 12Z had no composite
/// reflectivity at all at f000 and only weak (<25 dBZ) echo at f001 under
/// observed 40-50 dBZ cells, because the HRRR intermediate's QC/QR/QI/QS/QG
/// and QNC/QNI/QNR were dropped here while the hrrrv5 route (the RRFS
/// producer) read them.
pub const FIRST_GUESS_SPECIES: [(&str, &str); 8] = [
    ("QC", "qc"), ("QR", "qr"), ("QI", "qi"), ("QS", "qs"), ("QG", "qg"),
    ("QNC", "nc"), ("QNI", "ni"), ("QNR", "nr"),
];

fn species_rule(field: &str) -> Option<FieldRule> {
    let (_, name) = FIRST_GUESS_SPECIES.iter().find(|(f, _)| *f == field)?;
    Some(FieldRule { target: Target::Species(name), methods: PROFILE_METHODS, mask: None })
}

/// Whether a field's level participates in the first-guess level table.
fn contributes_a_level(target: Target) -> bool {
    matches!(
        target,
        Target::T | Target::Rh | Target::Sh | Target::Z | Target::P | Target::U | Target::V | Target::Number(_)
            | Target::Species(_)
    )
}

/// Everything the caller must state.  There are no defaults on the switches
/// that select physics: a wrong guess produces a structurally perfect file
/// with wrong numbers, which is exactly the failure mode this project has
/// already been bitten by once on `--map-source`.
#[derive(Debug, Clone)]
pub struct InitConfig {
    pub met_path: PathBuf,
    /// Explicit climatology path; None uses MPAS's local filename.
    pub gocart_path: Option<PathBuf>,
    /// config_gocartlevels; None uses the capsule dimension or native 30.
    pub gocart_levels: Option<usize>,
    /// Optional case-7 auxiliary intermediate inputs, read after main meteorology.
    pub seaice_met_path: Option<PathBuf>,
    pub omld_met_path: Option<PathBuf>,
    /// config_input_sst: reread this optional config_sfc_prefix input.
    pub sfc_met_path: Option<PathBuf>,
    pub static_path: PathBuf,
    pub capsule_path: PathBuf,
    /// The native init the capsule's `zgrid` is asserted against.
    pub reference_path: PathBuf,
    pub out_path: PathBuf,
    pub start_time: String,
    pub n_fg_levels: usize,
    pub n_fg_soil_levels: usize,
    pub land_surface: String,
    pub extrap_airtemp: Extrap,
    pub use_spechumd: bool,
    pub theta_adv_order: i32,
    pub coef_3rd_order: f32,
    pub virtual_factor: VirtualFactor,
    pub deep_moisture: DeepMoisture,
    /// `config_landuse_data`, which becomes the `mminlu` label the physics
    /// reads its land-use table by.  No default: the wrong table silently
    /// renumbers every vegetation category.
    pub landuse_table: String,
    /// `config_frac_seaice`.  With it on the sea-ice threshold is 0.02 and the
    /// fraction is kept; with it off the fraction is collapsed to 0/1 at 0.5
    /// first.  No default: the two give different coastlines.
    pub frac_seaice: bool,
    /// `config_tsk_seaice_threshold`, in kelvin.  MPAS's own default is 100 K,
    /// which never fires; a value near freezing converts every cold water
    /// point to land.
    pub tsk_seaice_threshold: f32,
    /// What `oned` does when its guard products underflow.  Not a namelist
    /// option: it is a property of how the reference was *compiled*.  See
    /// [`Underflow`].  No default, because the two answers differ by several
    /// relative-humidity points at the cells where it fires and both files
    /// look perfectly well formed.
    pub oned_underflow: Underflow,
    /// Read the first guess's hydrometeor masses (QC QR QI QS QG) and number
    /// concentrations (QNC QNI QNR) into the initial state; see
    /// [`FIRST_GUESS_SPECIES`].  `false` is MPAS-A v8.4.2 case 7, which reads
    /// none of them, and is what every native-identity proof runs.  With
    /// `true` and a first guess that carries none of them (GFS, ERA5, IFS as
    /// their rows stand) the file is byte-identical to `false`.
    pub first_guess_hydrometeors: bool,
    pub provenance: String,
    /// Who this file says it is: `model_name`, `core_name`, `version` and
    /// `git_version`, the four lineage attributes `rw_mpas_lbc` refuses a
    /// `--grid` for lacking.  See [`Lineage`].
    pub lineage: Lineage,
}

/// The four global attributes that say which model a file belongs to.
///
/// THE BREAKAGE THIS EXISTS TO PREVENT, MEASURED (2026-08-26): this writer
/// re-emitted the capsule's global attributes and added `file_id`,
/// `parent_id` and `gpuwm_provenance`, and nothing else.  A capsule carries
/// the MESH's lineage -- `on_a_sphere`, `sphere_radius`, `mesh_spec` and so
/// on -- but not the MODEL's, so no init this program produced carried
/// `model_name`, `core_name`, `version` or `git_version`.  `rw_mpas_lbc`
/// reads all four off its `--grid` and refuses rather than invent one, so
/// NO init this program minted could drive a boundary stream, regional or
/// global, and the limited-area chain ran through a hand-written patcher
/// instead.
///
/// The producer cannot know which model will integrate the file, so the
/// caller says.  The default is this engine's own name, which is at least
/// true; it is never `mpas`, because a file carrying this port's numerics
/// claiming MPAS's identity is exactly what the consumer's refusal is for.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Lineage {
    pub model_name: String,
    pub core_name: String,
    pub version: String,
    pub git_version: String,
}

impl Default for Lineage {
    fn default() -> Self {
        let version = env!("CARGO_PKG_VERSION").to_string();
        Lineage {
            model_name: "rw-mpas".to_string(),
            core_name: "atmosphere".to_string(),
            git_version: format!("rw-mpas-{version}"),
            version,
        }
    }
}

/// The receipt: counts, not adjectives.
#[derive(Debug, Default, serde::Serialize)]
pub struct InitReceipt {
    pub met_path: String,
    pub sfc_met_path: Option<String>,
    pub sfc_met_records_used: usize,
    pub seaice_met_path: Option<String>,
    pub seaice_met_records_used: usize,
    pub omld_met_path: Option<String>,
    pub omld_met_records_used: usize,
    pub static_path: String,
    pub capsule_path: String,
    pub reference_path: String,
    pub out_path: String,
    pub start_time: String,
    pub n_cells: usize,
    pub n_edges: usize,
    pub n_vert_levels: usize,
    pub met_records_read: usize,
    pub met_records_used: usize,
    pub met_records_ignored: Vec<String>,
    pub first_guess_levels_found: usize,
    pub surface_level_present: bool,
    pub zgrid_values_asserted_identical: usize,
    pub extrap_airtemp: String,
    pub use_spechumd: bool,
    pub virtual_factor: String,
    pub deep_moisture_rule: String,
    /// Which side of the reference's `-ftz` this run stands on.
    pub oned_underflow: String,
    /// How many `oned` calls hit a subnormal guard product, i.e. how many
    /// interpolation points the two `oned_underflow` modes can disagree at.
    /// Counted in both modes.  A non-zero count here is the whole reason the
    /// switch exists.
    pub oned_underflow_sites: u64,
    pub hydrostatic_iterations_max: u32,
    pub hydrostatic_iterations_mean: f64,
    pub hydrostatic_cells_hitting_the_cap: usize,
    pub aerosols: aerosols::AerosolReceipt,
    pub soil: soil::SoilCounts,
    pub soil_categories_reconciled: usize,
    pub seaice: soil::SeaiceCounts,
    pub emit: emit::EmitLedger,
    pub first_guess_hydrometeors: HydrometeorReceipt,
    /// Named loudly so nobody markets a capsule-fed init as a from-scratch one.
    pub deferrals: Vec<String>,
    pub seconds: f64,
}

/// What the initial state took from the first guess's hydrometeors.
#[derive(Debug, Default, serde::Serialize)]
pub struct HydrometeorReceipt {
    /// `read` or `off` (MPAS-A v8.4.2 case 7: none read).
    pub mode: String,
    /// MPAS scalar names whose first-guess records were read, interpolated
    /// and written (masses and numbers alike).
    pub species_read: Vec<String>,
    /// Number concentrations written that the first guess did not carry, or
    /// carried as zero where the mass is positive, made from the mass with
    /// the RRFS fork's calcnfromq (`rrfs_init::scalars::number_from_mass`):
    /// points of rain, cloud and ice number.  Thompson reads (q>0, n=0) rain
    /// as 2.5 mm drops.
    pub numbers_from_mass: [u64; 3],
    /// Largest written mixing ratio per mass species, kg/kg.
    pub max_mixing_ratio: BTreeMap<String, f32>,
}

#[derive(serde::Serialize)]
pub struct FirstGuess {
    vert_level: Vec<f32>,
    t: Vec<Vec<f32>>,
    rh: Vec<Vec<f32>>,
    sh: Vec<Vec<f32>>,
    z: Vec<Vec<f32>>,
    p: Vec<Vec<f32>>,
    u: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    psfc: Vec<f32>,
    pmsl: Vec<f32>,
    soilz: Vec<f32>,
    skintemp: Vec<f32>,
    snow: Vec<f32>,
    xice: Vec<f32>,
    u10: Vec<f32>,
    v10: Vec<f32>,
    st: Vec<Vec<f32>>,
    sm: Vec<Vec<f32>>,
    dzs_cm: Vec<f32>,
    numbers: BTreeMap<&'static str, Vec<Vec<f32>>>,
    species: BTreeMap<&'static str, Vec<Vec<f32>>>,
    node_t: BTreeMap<u16, Vec<f32>>,
    node_m: BTreeMap<u16, Vec<f32>>,
}

/// Complete first-guess surface extraction and pressure/height completion.
/// MPAS-A v8.4.2 core_init_atmosphere/mpas_init_atm_cases.F:4710-4769.
pub fn prepare_vertical_pass(fg: &mut FirstGuess) -> (Vec<f32>,Vec<f32>,Vec<f32>) {
    let n_cells = fg.t.len();
    let nfg = fg.vert_level.len();
    let sfc = fg.vert_level.iter().position(|&v|v == SURFACE_LEVEL_TAG);
    // Native initializes all three surface fields to positive zero and only
    // diagnoses this first q2 when a tagged profile level exists. The later
    // final q2 diagnosis remains unconditional (mpas_init_atm_cases.F:5239-5242).
    let t2m: Vec<f32> = (0..n_cells).into_par_iter().map(|c| sfc.map_or(0.0, |k| fg.t[c][k])).collect();
    let rh2: Vec<f32> = (0..n_cells).into_par_iter().map(|c| sfc.map_or(0.0, |k| fg.rh[c][k])).collect();
    let q2 = (0..n_cells).into_par_iter().map(|c|sfc.map_or(0.0, |_| dynamics::diagnose_q2_surface_pass(t2m[c],fg.psfc[c],rh2[c]))).collect();

    // Isobaric data carries no 3-D pressure; fill it from the level tags.
    let isobaric = fg
        .p
        .par_iter()
        .all(|col| col[..nfg].iter().all(|&v| v == 0.0));
    if isobaric {
        let levels = &fg.vert_level;
        fg.p.par_iter_mut().zip(fg.psfc.par_iter()).for_each(|(col, &ps)| {
            for (k, &lvl) in levels.iter().enumerate() {
                col[k] = if lvl == SURFACE_LEVEL_TAG { ps } else { lvl };
            }
        });
    } else if let Some(sfc) = sfc {
        fg.p.par_iter_mut().zip(fg.psfc.par_iter()).for_each(|(col, &v)| col[sfc] = v);
        fg.z.par_iter_mut().zip(fg.soilz.par_iter()).for_each(|(col, &v)| col[sfc] = v);
    }

    (t2m,rh2,q2)
}

#[allow(clippy::too_many_lines)]
/// Complete meteorological horizontal driver: MPAS-A v8.4.2
/// core_init_atmosphere/mpas_init_atm_cases.F:3450-4680.
pub fn horizontal_pass(
    slabs: &[MetSlab],
    mesh: &capsule::MeshGeometry,
    statics: &capsule::Statics,
    nfg: usize,
    nfgs: usize,
    underflow: hinterp::Underflow,
    hydrometeors: bool,
    receipt: &mut InitReceipt,
) -> MpasResult<FirstGuess> {
    let n_cells = mesh.n_cells;
    let n_edges = mesh.n_edges;

    // MPAS-A v8.4.2 mpas_init_atm_cases.F:3472-3475 allocates on each
    // LANDSEA record without deallocating; a second mask aborts native.
    if slabs.iter().filter(|s|s.field == "LANDSEA").count() > 1 {
        return Err(MpasError::Refusal("duplicate LANDSEA records: native case7 aborts when allocating its already allocated maskslab (mpas_init_atm_cases.F:3472-3475)".into()));
    }
    let landsea = slabs
        .iter()
        .find(|s| s.field == "LANDSEA")
        .ok_or_else(|| {
            MpasError::Refusal(
                "the intermediate file carries no LANDSEA field.  Case 7 uses it as the \
                 interpolation mask for every soil and sea-ice field; without it the coastal \
                 values would come from whatever the source grid holds over the other surface \
                 type, silently"
                    .to_string(),
            )
        })?;
    let mask_slab = Slab::from_met(&landsea.values, landsea.nx, landsea.ny);
    let underflow_hits = std::sync::atomic::AtomicU64::new(0);

    let mut fg = FirstGuess {
        numbers: BTreeMap::new(), species: BTreeMap::new(), node_t: BTreeMap::new(), node_m: BTreeMap::new(),
        vert_level: Vec::new(),
        t: par_columns(n_cells, nfg),
        rh: par_columns(n_cells, nfg),
        sh: par_columns(n_cells, nfg),
        z: par_columns(n_cells, nfg),
        p: par_columns(n_cells, nfg),
        u: par_columns(n_edges, nfg),
        v: par_columns(n_edges, nfg),
        psfc: vec![0.0; n_cells],
        pmsl: vec![0.0; n_cells],
        soilz: vec![0.0; n_cells],
        skintemp: vec![0.0; n_cells],
        snow: vec![0.0; n_cells],
        xice: vec![0.0; n_cells],
        u10: vec![0.0; n_cells],
        v10: vec![0.0; n_cells],
        st: par_columns(n_cells, nfgs),
        sm: par_columns(n_cells, nfgs),
        dzs_cm: vec![0.0; nfgs],
    };

    let cell_lat: Vec<f32> = mesh.lat_cell.iter().map(|v| v * DEG_PER_RAD).collect();
    let cell_lon: Vec<f32> = mesh.lon_cell.iter().map(|v| v * DEG_PER_RAD).collect();
    let edge_lat: Vec<f32> = mesh.lat_edge.iter().map(|v| v * DEG_PER_RAD).collect();
    let edge_lon: Vec<f32> = mesh.lon_edge.iter().map(|v| v * DEG_PER_RAD).collect();

    for s in slabs {
        receipt.met_records_read += 1;
        let Some(rule) = rule_for(&s.field)
            .or_else(|| if hydrometeors { species_rule(&s.field) } else { None }) else {
            if s.field != "LANDSEA" && !receipt.met_records_ignored.contains(&s.field) {
                receipt.met_records_ignored.push(s.field.clone());
            }
            continue;
        };
        receipt.met_records_used += 1;

        // The level table: first sighting wins, and it is capped.
        let level_index = if contributes_a_level(rule.target) {
            match fg.vert_level.iter().position(|&v| v == s.xlvl) {
                Some(i) => i,
                None => {
                    if fg.vert_level.len() >= nfg {
                        return Err(MpasError::Refusal(format!(
                            "the intermediate file holds more than {nfg} distinct levels; raise \
                             config_nfglevels to at least {} and re-run",
                            fg.vert_level.len() + 1
                        )));
                    }
                    fg.vert_level.push(s.xlvl);
                    fg.vert_level.len() - 1
                }
            }
        } else {
            0
        };

        let surface_wind=matches!(rule.target,Target::U|Target::V) && s.xlvl==SURFACE_LEVEL_TAG;
        let proj = projection::Projection::from_met(s)?;
        let array = Slab::from_met(&s.values, s.nx, s.ny);
        let ctx = InterpContext {
            array: &array,
            msgval: MSGVAL,
            maskval: if surface_wind {Some(-1.)} else {rule.mask.map(|(m, _, _)| m)},
            mask: if surface_wind {Some(&mask_slab)} else {rule.mask.map(|_| &mask_slab)},
            underflow,
            underflow_hits: Some(&underflow_hits),
        };

        let at_surface = s.xlvl == SURFACE_LEVEL_TAG;
        let _ = at_surface;
        let on_edges = matches!(rule.target, Target::U | Target::V) && !at_surface;

        let sample = |i: usize, lat: &[f32], lon: &[f32]| -> f32 {
            let (x,y)=if surface_wind {proj.locate_surface_wind(lat[i],lon[i])}
                else {proj.locate(lat[i],lon[i])};
            hinterp::interp_sequence(x, y, &ctx, rule.methods, 0)
        };

        if on_edges {
            let values: Vec<f32> = (0..n_edges)
                .into_par_iter()
                .map(|e| sample(e, &edge_lat, &edge_lon))
                .collect();
            let dest = if rule.target == Target::U {
                &mut fg.u
            } else {
                &mut fg.v
            };
            dest.par_iter_mut().zip(values.par_iter()).for_each(|(col, v)| col[level_index] = *v);
            continue;
        }

        // MPAS v8.4.2 mpas_init_atm_cases.F:3532-3534,4591-4623,4654-4663.
        // Default -1 masks only the destination; the surface-wind pass
        // ignores destination masks and always supplies the source mask.
        let masked_flag = if surface_wind {None} else {Some(rule.mask.map(|(_,m,_)|m).unwrap_or(-1))};
        let fillval = rule.mask.map(|(_, _, f)| f).unwrap_or(0.0);
        let values: Vec<f32> = (0..n_cells)
            .into_par_iter()
            .map(|c| {
                if let Some(m) = masked_flag {
                    if statics.landmask[c] == m {
                        return fillval;
                    }
                }
                sample(c, &cell_lat, &cell_lon)
            })
            .collect();

        match rule.target {
            Target::T => {
                // The surface level lands in the same column slot as any
                // other: it is the 200100.0 tag, not a separate array, that
                // makes it the surface later on.
                fg.t.par_iter_mut().zip(values.par_iter()).for_each(|(col, v)| col[level_index] = *v);
            }
            Target::Rh => {
                fg.rh.par_iter_mut().zip(values.par_iter()).for_each(|(col, v)| col[level_index] = *v);
            }
            Target::Sh => {
                fg.sh.par_iter_mut().zip(values.par_iter()).for_each(|(col, v)| col[level_index] = *v);
            }
            Target::Z => {
                fg.z.par_iter_mut().zip(values.par_iter()).for_each(|(col, v)| col[level_index] = *v);
            }
            Target::P => {
                fg.p.par_iter_mut().zip(values.par_iter()).for_each(|(col, v)| col[level_index] = *v);
            }
            Target::U => fg.u10.copy_from_slice(&values),
            Target::V => fg.v10.copy_from_slice(&values),
            Target::Psfc => fg.psfc.copy_from_slice(&values),
            Target::Pmsl => fg.pmsl.copy_from_slice(&values),
            Target::SoilHgt => fg.soilz.copy_from_slice(&values),
            Target::Skintemp => fg.skintemp.copy_from_slice(&values),
            Target::Snow => fg.snow.copy_from_slice(&values),
            Target::Seaice => fg.xice.copy_from_slice(&values),
            Target::Number(name) => {
                let field = fg.numbers.entry(name).or_insert_with(||par_columns(n_cells,nfg));
                field.par_iter_mut().zip(values.par_iter()).for_each(|(col,v)| col[level_index]=*v);
            }
            Target::Species(name) => {
                let field = fg.species.entry(name).or_insert_with(||par_columns(n_cells,nfg));
                field.par_iter_mut().zip(values.par_iter()).for_each(|(col,v)| col[level_index]=*v);
            }
            Target::NodeT(depth) => { fg.node_t.insert(depth, values); }
            Target::NodeM(depth) => { fg.node_m.insert(depth, values); }
            Target::SoilT(k, dz) => {
                if k >= nfgs {
                    return Err(MpasError::Refusal(format!(
                        "{} lands on first-guess soil level {} but only {nfgs} were declared",
                        s.field,
                        k + 1
                    )));
                }
                fg.dzs_cm[k] = dz;
                fg.st.par_iter_mut().zip(values.par_iter()).for_each(|(col, v)| col[k] = *v);
            }
            Target::SoilM(k, dz) => {
                if k >= nfgs {
                    return Err(MpasError::Refusal(format!(
                        "{} lands on first-guess soil level {} but only {nfgs} were declared",
                        s.field,
                        k + 1
                    )));
                }
                fg.dzs_cm[k] = dz;
                fg.sm.par_iter_mut().zip(values.par_iter()).for_each(|(col, v)| col[k] = *v);
            }
        }
    }

    receipt.oned_underflow = underflow.label().to_string();
    receipt.oned_underflow_sites =
        underflow_hits.load(std::sync::atomic::Ordering::Relaxed);
    receipt.first_guess_levels_found = fg.vert_level.len();
    receipt.surface_level_present = fg.vert_level.contains(&SURFACE_LEVEL_TAG);
    if fg.dzs_cm.iter().any(|&v| v <= 0.0) {
        return Err(MpasError::Refusal(format!(
            "the first-guess soil column is incomplete: layer thicknesses {:?} (cm).  A zero \
             means a soil level this run declared was never delivered by the intermediate file",
            fg.dzs_cm
        )));
    }
    Ok(fg)
}

/// `n` zeroed columns of `len` words, allocated on rayon workers.  The serial
/// `vec![vec![0.0; len]; n]` spent seconds on one thread on the 15 km mesh
/// (tens of millions of small allocations); the columns are the same.
fn par_columns(n: usize, len: usize) -> Vec<Vec<f32>> {
    (0..n).into_par_iter().map(|_| vec![0.0f32; len]).collect()
}

/// Build one init file.
/// Profile-only: run `f`, then stamp `name` (silent by default, like `stage`).
fn timed<T>(stage: &(impl Fn(&str) + Sync), name: &str, f: impl FnOnce() -> T) -> T {
    let value = f();
    stage(name);
    value
}

pub fn build_init(cfg: &InitConfig) -> MpasResult<InitReceipt> {
    let started = std::time::Instant::now();
    // Stage timing and stage dumps for the batched GPU init's oracle. Silent
    // unless INIT_GPU_ORACLE (dump directory) or RW_MPAS_INIT_PROFILE is set,
    // so a default init writes the same stderr and the same file as before.
    let profile = std::env::var_os("INIT_GPU_ORACLE").is_some()
        || std::env::var_os("RW_MPAS_INIT_PROFILE").is_some();
    let stage = |name: &str| {
        if profile {
            eprintln!("INIT_STAGE {name} {:.6}", started.elapsed().as_secs_f64());
        }
    };
    let mut receipt = InitReceipt {
        met_path: cfg.met_path.display().to_string(),
        sfc_met_path: cfg.sfc_met_path.as_ref().map(|p|p.display().to_string()),
        seaice_met_path: cfg.seaice_met_path.as_ref().map(|p|p.display().to_string()),
        omld_met_path: cfg.omld_met_path.as_ref().map(|p|p.display().to_string()),
        static_path: cfg.static_path.display().to_string(),
        capsule_path: cfg.capsule_path.display().to_string(),
        reference_path: cfg.reference_path.display().to_string(),
        out_path: cfg.out_path.display().to_string(),
        start_time: cfg.start_time.clone(),
        extrap_airtemp: cfg.extrap_airtemp.label().to_string(),
        use_spechumd: cfg.use_spechumd,
        virtual_factor: format!("{:?}", cfg.virtual_factor),
        deep_moisture_rule: format!("{:?}", cfg.deep_moisture),
        deferrals: vec![
            "vertical fields are read from the capsule contract; the front-door receipt identifies \
             their native or rw_mpas_geometry producer".to_string(),
            "statics are read from the supplied static file; the front-door receipt identifies \
             their producer".to_string(),
            "coeffs_reconstruct and edgeNormalVectors are carried from the capsule, so this \
             file is only drop-in for a consumer that pins the SAME grid/static bytes the \
             capsule was built from.  The forecast driver's \
             overlay_exact_init_reconstruction_coefficients refuses an init built against a \
             different grid/static pair by design, and reports it as changed coefficient \
             bytes rather than as a provenance mismatch"
                .to_string(),
        ],
        ..Default::default()
    };

    // The five inputs are independent reads, so they run side by side on
    // rayon workers; their results are then taken in the order the serial
    // reader used, so a failing input names the same error it always did.
    let (n_cells_static, n_edges_static) = capsule::mesh_sizes(&cfg.static_path)?;
    let ((mesh, statics), ((terrain, vertical), slabs)) = rayon::join(
        || {
            rayon::join(
                || timed(&stage, "read.mesh", || capsule::read_mesh_geometry(&cfg.static_path)),
                || timed(&stage, "read.statics", || capsule::read_statics(&cfg.static_path, n_cells_static)),
            )
        },
        || {
            rayon::join(
                || {
                    rayon::join(
                        || timed(&stage, "read.terrain", || capsule::read_smoothed_terrain(&cfg.capsule_path, n_cells_static)),
                        || timed(&stage, "read.vertical", || -> MpasResult<_> {
                            let metrics = capsule::read_vertical_metrics(
                                &cfg.capsule_path,
                                n_cells_static,
                                n_edges_static,
                            )?;
                            let asserted =
                                capsule::assert_zgrid_identical(&metrics, &cfg.reference_path);
                            Ok((metrics, asserted))
                        }),
                    )
                },
                || timed(&stage, "read.met", || metfile::read_met_file(&cfg.met_path)),
            )
        },
    );
    let mesh = mesh?;
    let mut statics = statics?;
    // The terrain every later step sees is the SMOOTHED one the vertical-grid
    // block wrote back, not the raw static field.  See read_smoothed_terrain.
    statics.ter = terrain?;
    let (metrics, asserted) = vertical?;
    receipt.zgrid_values_asserted_identical = asserted?;
    receipt.n_cells = mesh.n_cells;
    receipt.n_edges = mesh.n_edges;
    receipt.n_vert_levels = metrics.n_vert_levels;

    let slabs = slabs?;
    stage("read");
    let mut fg = horizontal_pass(&slabs, &mesh, &statics, cfg.n_fg_levels,
        cfg.n_fg_soil_levels,cfg.oned_underflow,cfg.first_guess_hydrometeors,&mut receipt)?;
    if let Some(path)=&cfg.seaice_met_path {
        receipt.seaice_met_records_used=auxiliary::interpolate(&metfile::read_met_file(path)?,auxiliary::Kind::SeaIce,
            &mesh.lat_cell,&mesh.lon_cell,&statics.landmask,&mut fg.xice,cfg.oned_underflow)?;
    }
    let mut ocean_depth=vec![0.;mesh.n_cells];
    if let Some(path)=&cfg.omld_met_path {
        receipt.omld_met_records_used=auxiliary::interpolate(&metfile::read_met_file(path)?,auxiliary::Kind::OceanDepth,
            &mesh.lat_cell,&mesh.lon_cell,&statics.landmask,&mut ocean_depth,cfg.oned_underflow)?;
    }
    stage("horizontal");
    if let Ok(dir) = std::env::var("INIT_GPU_ORACLE") {
        let mut values: Vec<(String, Vec<f32>)> = Vec::new();
        for (name, cols) in [("t", &fg.t), ("rh", &fg.rh), ("sh", &fg.sh), ("z", &fg.z), ("p", &fg.p),
                             ("u", &fg.u), ("v", &fg.v), ("st", &fg.st), ("sm", &fg.sm)] {
            values.push((name.into(), cols.iter().flatten().copied().collect()));
        }
        for (name, v) in [("vert_level", &fg.vert_level), ("psfc", &fg.psfc), ("pmsl", &fg.pmsl),
                           ("soilz", &fg.soilz), ("skintemp", &fg.skintemp), ("snow", &fg.snow),
                           ("xice", &fg.xice), ("u10", &fg.u10), ("v10", &fg.v10), ("dzs_cm", &fg.dzs_cm)] {
            values.push((name.into(), v.clone()));
        }
        gpu_oracle_dump(&dir, "horizontal", values)?;
    }

    let nz = metrics.n_vert_levels;
    let n_cells = mesh.n_cells;
    let n_edges = mesh.n_edges;
    let nfg = fg.vert_level.len();

    let (t2m,rh2,_first_q2) = prepare_vertical_pass(&mut fg);

    // Vertical interpolation, cell by cell.
    let zgrid = &metrics.zgrid;
    let extrap_t = cfg.extrap_airtemp;
    type Column = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, f32);
    let columns: Vec<MpasResult<Column>> = (0..n_cells)
        .into_par_iter()
        .map(|c| {
            let mut coord = vec![0.0f32; nfg];
            let mut value = vec![0.0f32; nfg];
            let prep = |coord: &mut Vec<f32>, value: &mut Vec<f32>, src: &dyn Fn(usize) -> f32| {
                for k in 0..nfg {
                    coord[k] = if fg.vert_level[k] == SURFACE_LEVEL_TAG {
                        SURFACE_SENTINEL
                    } else {
                        fg.z[c][k]
                    };
                    value[k] = src(k);
                }
                sorted_column(coord, value);
            };

            let mut t = vec![0.0f32; nz];
            let mut relhum = vec![0.0f32; nz];
            let mut spechum = vec![0.0f32; nz];
            let mut pressure = vec![0.0f32; nz];
            let mut gfs_z = vec![0.0f32; nz];

            prep(&mut coord, &mut value, &|k| fg.t[c][k]);
            for k in 0..nz {
                let target = 0.5 * (zgrid[c][k] + zgrid[c][k + 1]);
                t[k] = vertical_interp(target, &coord[..nfg - 1], &value[..nfg - 1], extrap_t)
                    .map_err(|_| {
                        MpasError::Refusal(format!(
                            "temperature extrapolation above the first-guess column top is not \
                             implemented for the lapse-rate mode; cell {c} level {k} asked for \
                             it.  This is the Fortran's own fatal error, not a port limitation"
                        ))
                    })?;
            }

            prep(&mut coord, &mut value, &|k| fg.rh[c][k]);
            for k in (0..nz).rev() {
                let target = 0.5 * (zgrid[c][k] + zgrid[c][k + 1]);
                relhum[k] = vertical_interp(
                    target,
                    &coord[..nfg - 1],
                    &value[..nfg - 1],
                    Extrap::Constant,
                )
                .unwrap_or(value[0]);
            }

            prep(&mut coord, &mut value, &|k| fg.sh[c][k].max(0.0));
            for k in (0..nz).rev() {
                let target = 0.5 * (zgrid[c][k] + zgrid[c][k + 1]);
                spechum[k] = vertical_interp(
                    target,
                    &coord[..nfg - 1],
                    &value[..nfg - 1],
                    Extrap::Constant,
                )
                .unwrap_or(value[0]);
            }

            // MPAS v8.4.2 mpas_init_atm_cases.F:5078-5092 (GHT diagnostic).
            prep(&mut coord, &mut value, &|k| fg.z[c][k]);
            for k in 0..nz {
                let target=0.5*(zgrid[c][k]+zgrid[c][k+1]);
                gfs_z[k]=vertical_interp(target,&coord[..nfg-1],&value[..nfg-1],Extrap::Linear)
                    .expect("linear extrapolation is defined for the admitted column");
            }

            prep(&mut coord, &mut value, &|k| fg.p[c][k].ln());
            for k in 0..nz {
                let target = 0.5 * (zgrid[c][k] + zgrid[c][k + 1]);
                pressure[k] =
                    vertical_interp(target, &coord[..nfg - 1], &value[..nfg - 1], Extrap::Linear)
                        .unwrap_or(value[nfg - 2])
                        .exp();
            }

            // Surface pressure is the one field that keeps the surface level:
            // the full column, sentinel included, targeted at zgrid(1).
            prep(&mut coord, &mut value, &|k| fg.p[c][k].ln());
            let psfc = vertical_interp(zgrid[c][0], &coord, &value, Extrap::Linear)
                .unwrap_or(value[nfg - 1])
                .exp();

            Ok((t, relhum, spechum, pressure, gfs_z, psfc))
        })
        .collect();

    let mut t_col = Vec::with_capacity(n_cells);
    let mut rh_col = Vec::with_capacity(n_cells);
    let mut sh_col = Vec::with_capacity(n_cells);
    let mut p_col = Vec::with_capacity(n_cells);
    let mut psfc = Vec::with_capacity(n_cells);
    let mut gfs_z_col = Vec::with_capacity(n_cells);
    for c in columns {
        let (t, rh, sh, p, gfs_z, ps) = c?;
        t_col.push(t);
        rh_col.push(rh);
        sh_col.push(sh);
        p_col.push(p);
        psfc.push(ps);
        gfs_z_col.push(gfs_z);
    }

    stage("vertical");
    if let Ok(dir) = std::env::var("INIT_GPU_ORACLE") {
        let mut values: Vec<(String, Vec<f32>)> = Vec::new();
        for (name, cols) in [("t", &t_col), ("rh", &rh_col), ("sh", &sh_col), ("p", &p_col)] {
            values.push((name.into(), cols.iter().flatten().copied().collect()));
        }
        values.push(("psfc".into(), psfc.clone()));
        gpu_oracle_dump(&dir, "vertical", values)?;
    }
    // Edge winds: rotate to the edge normal, then interpolate on the two-cell
    // mean coordinate to the four-point mean target.
    let u_edge: Vec<Vec<f32>> = (0..n_edges)
        .into_par_iter()
        .map(|e| {
            let [c1, c2] = mesh.cells_on_edge[e];
            let c1 = (c1.max(1) - 1).min(n_cells - 1);
            let c2 = (c2.max(1) - 1).min(n_cells - 1);
            let angle = mesh.angle_edge[e];
            let mut coord = vec![0.0f32; nfg];
            let mut value = vec![0.0f32; nfg];
            for k in 0..nfg {
                coord[k] = if fg.vert_level[k] == SURFACE_LEVEL_TAG {
                    SURFACE_SENTINEL
                } else {
                    dynamics::edge_column_coordinate(fg.z[c1][k], fg.z[c2][k])
                };
                value[k] = dynamics::edge_normal_wind(angle, fg.u[e][k], fg.v[e][k]);
            }
            sorted_column(&mut coord, &mut value);
            (0..nz)
                .map(|k| {
                    let target = dynamics::edge_target_height(
                        zgrid[c1][k],
                        zgrid[c1][k + 1],
                        zgrid[c2][k],
                        zgrid[c2][k + 1],
                    );
                    dynamics::interp_edge_column(target, &coord[..nfg - 1], &value[..nfg - 1])
                })
                .collect()
        })
        .collect();

    stage("edge");
    // Moisture, then RH over ice, then the column chain.
    let use_sh = cfg.use_spechumd
        && sh_col
            .iter()
            .any(|col| col.iter().any(|&v| v != 0.0));
    let factor = cfg.virtual_factor;
    let states: Vec<dynamics::ColumnState> = (0..n_cells)
        .into_par_iter()
        .map(|c| {
            dynamics::build_column(
                &t_col[c],
                &p_col[c],
                &rh_col[c],
                &sh_col[c],
                use_sh,
                &zgrid[c],
                &metrics.zz[c],
                &metrics.fzm,
                &metrics.fzp,
                &metrics.dzu,
                metrics.rdzw[0],
                factor,
            )
        })
        .collect();

    rh_col.par_iter_mut().zip(t_col.par_iter()).for_each(|(rh, t)| {
        dynamics::convert_relhum_wrt_ice(t, rh);
    });

    let (it_max, it_sum, it_n, capped) = states
        .par_iter()
        .map(|s| {
            let its = &s.hydrostatic_iterations[1..];
            let max = its.iter().copied().max().unwrap_or(0);
            let sum: u64 = its.iter().map(|&i| i as u64).sum();
            let hit = its.iter().any(|&i| i >= 30);
            (max, sum, its.len() as u64, usize::from(hit))
        })
        .reduce(|| (0u32, 0u64, 0u64, 0usize), |a, b| (a.0.max(b.0), a.1 + b.1, a.2 + b.2, a.3 + b.3));
    receipt.hydrostatic_iterations_max = it_max;
    receipt.hydrostatic_iterations_mean = if it_n > 0 {
        it_sum as f64 / it_n as f64
    } else {
        0.0
    };
    receipt.hydrostatic_cells_hitting_the_cap = capped;

    stage("dynamics");
    let nzp1=nz+1;
    // The default init stream excludes these met-stage diagnostics. If the
    // requested schema carries them, compute them from this run's winds.
    let reconstruction_contract=capsule::read_reconstruction_contract(&cfg.capsule_path,
        n_cells,mesh.edges_on_cell[0].len())?;
    let reconstructed=if let Some((_,coefficients,sphere))=&reconstruction_contract {
        Some(reconstruction::reconstruct(&u_edge,&mesh.n_edges_on_cell,&mesh.edges_on_cell,
            coefficients,&mesh.lat_cell,&mesh.lon_cell,*sphere)?)
    } else {None};
    let rho_zz:Vec<&[f32]>=states.iter().map(|state|state.rho_zz.as_slice()).collect();
    let winds=dynamics::diagnose_winds(&u_edge,&rho_zz,&mesh.cells_on_edge,&mesh.n_edges_on_cell,
        &mesh.edges_on_cell,&metrics,cfg.theta_adv_order,cfg.coef_3rd_order);
    let w=&winds.w;

    stage("w");
    // Soil and the surface companions.
    // SST is a copy of the skin temperature taken in the case code, BEFORE
    // physics_initialize_real adjusts the skin temperature onto the model's
    // terrain.  Copying it afterwards would give every land cell an SST that
    // carries the terrain correction.
    let mut sst = fg.skintemp.clone();

    // MPAS mpas_init_atm_cases.F:4780-4800 precedes the physics initializer.
    let (flagged, repaired) =
        soil::first_guess_consistency_pass(&statics.landmask, &fg.st, &mut fg.sm);
    // MPAS v8.4.2 mpas_atmphys_initialize_real.F:143-148,516-583.
    // Optional surface input precedes terrain adjustment and soil initialization.
    if let Some(path)=&cfg.sfc_met_path {
        let slabs=metfile::read_met_file(path)?;
        receipt.sfc_met_records_used = surface::initial::interpolate(&slabs,&mesh.lat_cell,&mesh.lon_cell,&statics.landmask,
            &mut sst,&mut fg.xice,cfg.oned_underflow)?;
        surface::initial::initialize_sst(cfg.frac_seaice,&statics.landmask,&sst,
            &mut fg.skintemp,&mut fg.xice);
    }
    let mut skintemp = fg.skintemp.clone();
    let mut tmn = soil::adjust_input_soiltemps(
        &statics.landmask,
        &statics.ter,
        &statics.soiltemp,
        &fg.soilz,
        &mut skintemp,
        &mut fg.st,
    );
    let (mut tslb, mut smois, mut sh2o, mut smcrel, mut soil_counts) =
        soil::init_soil_layers_properties(
        &statics.landmask,
        &skintemp,
        &tmn,
        &fg.st,
        &fg.sm,
        &fg.dzs_cm,
        cfg.deep_moisture,
    )?;
    let mut tslb: Vec<Vec<f32>> = tslb.into_iter().map(Vec::from).collect();
    let mut smois: Vec<Vec<f32>> = smois.into_iter().map(Vec::from).collect();
    let mut sh2o: Vec<Vec<f32>> = sh2o.into_iter().map(Vec::from).collect();
    let mut smcrel: Vec<Vec<f32>> = smcrel.into_iter().map(Vec::from).collect();
    if cfg.land_surface == "ruc" {
        let has_nodes = !fg.node_t.is_empty() || !fg.node_m.is_empty();
        if has_nodes && (fg.node_t.keys().collect::<Vec<_>>() != fg.node_m.keys().collect::<Vec<_>>()) {
            return Err(MpasError::Refusal("source soil node temperature and moisture depths disagree".into()));
        }
        // WRF's reader forms a layer source's depth as the INTEGER mid-depth
        // of its bounds, (top + bottom) / 2 in centimetres (char2int2), so a
        // 0-7 cm layer samples at 3 cm. Identical to the float mid-depth for
        // every even-bounded source (GFS: 5, 25, 70, 150).
        let mut layer_mid_cm = Vec::with_capacity(fg.dzs_cm.len());
        let mut top = 0i64;
        for &dz in &fg.dzs_cm {
            let bottom = top + dz.round() as i64;
            layer_mid_cm.push(((top + bottom) / 2) as f32);
            top = bottom;
        }
        let node_cm: Vec<f32> = fg.node_t.keys().map(|&z| f32::from(z)).collect();
        for c in 0..n_cells {
            if statics.landmask[c] == 0 {
                // init_soil_3_real's open-water fill; never interpolated.
                tslb[c] = vec![skintemp[c]; 9]; smois[c] = vec![1.; 9];
            } else if has_nodes {
                // adjust_soil_temp_new moves every source node onto the
                // model terrain, as the layer path's first guess already is.
                let shift = 0.0065 * (statics.ter[c] - fg.soilz[c]);
                let t: Vec<f32> = fg.node_t.values().map(|v| v[c] - shift).collect();
                let m: Vec<f32> = fg.node_m.values().map(|v| v[c]).collect();
                (tslb[c], smois[c]) = ruc::node_column(&node_cm, &t, &m)
                    .map_err(|e| MpasError::Refusal(format!("cell {c}: {e}")))?;
            } else {
                (tslb[c], smois[c]) = ruc::layer_column(&layer_mid_cm, &fg.st[c], &fg.sm[c], skintemp[c], tmn[c])
                    .map_err(|e| MpasError::Refusal(format!("cell {c}: {e}")))?;
                let (raised, lowest) = ruc::floor_layer_moisture(&mut smois[c]);
                if raised > 0 {
                    soil_counts.ruc_layer_floor_values += raised;
                    soil_counts.ruc_layer_floor_columns += 1;
                    soil_counts.ruc_layer_floor_min = Some(soil_counts.ruc_layer_floor_min.map_or(lowest, |m| m.min(lowest)));
                }
            }
            sh2o[c] = smois[c].clone(); smcrel[c] = vec![0.;9];
        }
    }
    soil_counts.land_moisture_repaired = repaired;
    soil_counts.land_temperature_flagged = flagged;
    receipt.soil = soil_counts;

    let (year, month, day) = parse_ymd(&cfg.start_time)?;
    let surface=soil::init_surface_companions(&statics.landmask,&fg.snow,
        &statics.albedo12m,&statics.greenfrac,year,month,day);
    let sfc_albbck=surface.albedo;let vegfra=surface.vegetation;
    let shdmin=surface.minimum;let shdmax=surface.maximum;
    let mut snowc=surface.snow_cover;let mut snowh=surface.snow_height;
    let mut xland=surface.land;
    let mut xice: Vec<f32> = fg.xice.iter().map(|&v| v.clamp(0.0, 1.0)).collect();
    let mut seaice: Vec<f32> = vec![0.0; n_cells];
    let mut snow = fg.snow.clone();
    let mut vegfra = vegfra;
    let mut snoalb = statics.snoalb.clone();
    let mut ivgtyp = statics.ivgtyp.clone();
    let mut isltyp = statics.isltyp.clone();

    // The sea-ice initialisation runs last and rewrites what the soil step
    // produced at every converted point.
    receipt.seaice = soil::physics_init_seaice(
        cfg.frac_seaice,
        cfg.tsk_seaice_threshold,
        statics.isice_lu,
        &statics.landmask,
        &skintemp,
        &mut xice,
        &mut seaice,
        &mut snow,
        &mut snowc,
        &mut snowh,
        &mut vegfra,
        &mut snoalb,
        &mut ivgtyp,
        &mut isltyp,
        &mut xland,
        &mut tmn,
        &mut tslb,
        &mut smois,
        &mut sh2o,
        &mut smcrel,
    );
    // Preserve the legacy MPAS Noah initializer's field identity. Selected
    // WRF land models need real.exe's category reconciliation before LSMINIT.
    if cfg.land_surface != "noah" {
        receipt.soil_categories_reconciled = soil::reconcile_soil_category(&xland, &mut isltyp);
    }
    let q2: Vec<f32> = (0..n_cells)
        .into_par_iter()
        .map(|c| dynamics::diagnose_q2(t2m[c], psfc[c], rh2[c]))
        .collect();

    stage("soil");
    // Assemble the computed set, cell-major with the vertical fastest.
    // Each row is written in place by a rayon worker: the same words in the
    // same cell-major order the serial loops produced.
    let flat3 = |f: &(dyn Fn(usize, usize) -> f32 + Sync)| -> Vec<f32> {
        let mut out = vec![0.0f32; n_cells * nz];
        if nz > 0 {
            out.par_chunks_mut(nz).enumerate().for_each(|(c, row)| {
                for k in 0..nz {
                    row[k] = f(c, k);
                }
            });
        }
        out
    };
    let rows = |src: &Vec<Vec<f32>>, n: usize, width: usize| -> Vec<f32> {
        let mut out = vec![0.0f32; n * width];
        if width > 0 {
            out.par_chunks_mut(width).enumerate().for_each(|(i, row)| {
                row.copy_from_slice(&src[i][..width]);
            });
        }
        out
    };
    let flat3_edges = |src: &Vec<Vec<f32>>| -> Vec<f32> { rows(src, n_edges, nz) };
    let flat3p1 = |src: &Vec<Vec<f32>>| -> Vec<f32> { rows(src, n_cells, nzp1) };
    let flat_soil = |src: &Vec<Vec<f32>>| -> Vec<f32> {
        // Rows of one width (every Noah and RUC column) go to rayon workers;
        // anything ragged keeps the serial concatenation.
        let width = src.first().map_or(0, Vec::len);
        if src.len() == n_cells && src.par_iter().all(|r| r.len() == width) {
            return rows(src, n_cells, width);
        }
        let mut out = Vec::with_capacity(n_cells * 4);
        for c in 0..n_cells {
            out.extend_from_slice(&src[c]);
        }
        out
    };

    // MPAS: src/core_init_atmosphere/mpas_init_atm_cases.F:261-267;
    // src/core_init_atmosphere/mpas_init_atm_thompson_aerosols.F:36-205;
    // src/core_init_atmosphere/mpas_init_atm_core_interface.F:229-231.
    let aerosol_capsule = netcrust::File::open(&cfg.capsule_path)?;
    let aerosol_vars = aerosol_capsule.variables()?;
    let has_nifa = aerosol_vars.iter().any(|v| v.name() == "nifa");
    let has_nwfa = aerosol_vars.iter().any(|v| v.name() == "nwfa");
    let native_path = PathBuf::from(aerosols::NATIVE_FILENAME);
    let gocart_path = cfg.gocart_path.as_deref().unwrap_or(&native_path);
    let source_present = gocart_path.try_exists()?;
    if has_nifa != has_nwfa && !source_present {
        return Err(MpasError::Refusal("init capsule must carry both nifa and nwfa; without climatology the missing paired scalar cannot be initialized".into()));
    }
    // Native activates mp_thompson_aers_in from file presence even when the
    // input stream has no aerosol variables. The emitter declares the pair.
    let aerosol_state = if has_nifa || source_present {
        let n_gocart = cfg
            .gocart_levels
            .or_else(|| {
                aerosol_capsule
                    .dimensions()
                    .ok()?
                    .iter()
                    .find(|d| d.name() == "nGocartLevels")
                    .map(|d| d.len())
            })
            .unwrap_or(30);
        let pressure_base = flat3(&|c, k| states[c].pressure_base[k]);
        let (state, audit) = aerosols::initialize(
            gocart_path,
            &mesh.lat_cell,
            &mesh.lon_cell,
            &statics.landmask,
            n_gocart,
            &cfg.start_time,
            &pressure_base,
            nz,
        )?;
        receipt.aerosols = audit;
        Some(state)
    } else {
        None
    };

    // The first guess's own hydrometeors (FIRST_GUESS_SPECIES), when this run
    // reads them.  Each column is interpolated the way the RRFS fork
    // interpolates its scalars (RRFSx/MPAS-Model mpas_init_atm_cases.F:
    // 7244-7302 at 1d6e370e, `rrfs_init::scalars::interpolate`): sorted on
    // the first guess's heights with the tagged surface set aside, values
    // held nonnegative, constant beyond the column.  Off, or with a first
    // guess that carries none, nothing here is written and the file is
    // MPAS-A v8.4.2 case 7's (every hydrometeor zero).
    receipt.first_guess_hydrometeors.mode =
        if cfg.first_guess_hydrometeors { "read" } else { "off" }.to_string();
    let mut species_out: BTreeMap<&'static str, Vec<f32>> = BTreeMap::new();
    if cfg.first_guess_hydrometeors && !fg.species.is_empty() && nz > 0 {
        for (&name, field) in &fg.species {
            let mut out = vec![0.0f32; n_cells * nz];
            out.par_chunks_mut(nz).enumerate().for_each(|(c, column)| {
                let mut coord = vec![0.0f32; nfg];
                let mut value = vec![0.0f32; nfg];
                for k in 0..nfg {
                    coord[k] = if fg.vert_level[k] == SURFACE_LEVEL_TAG { SURFACE_SENTINEL } else { fg.z[c][k] };
                    value[k] = field[c][k].max(0.0);
                }
                sorted_column(&mut coord, &mut value);
                for k in (0..nz).rev() {
                    let target = 0.5 * (zgrid[c][k] + zgrid[c][k + 1]);
                    column[k] = vertical_interp(target, &coord[..nfg - 1], &value[..nfg - 1], Extrap::Constant)
                        .unwrap_or(value[0])
                        .max(0.0);
                }
            });
            species_out.insert(name, out);
        }
        receipt.first_guess_hydrometeors.species_read = fg.species.keys().map(|s| s.to_string()).collect();
        // A mass with no number of its own (a one-moment first guess such as
        // GFS, or a point where the interpolated number is zero): Thompson
        // reads (qr>0, nr=0) as 2.5 mm drops and the first hour rains out over
        // the whole domain (the RRFS lam7k cut, rrfs_init::scalars).  Those
        // numbers are made from the mass with the fork's calcnfromq, on this
        // init's dry density; numbers the first guess carried are kept.
        use crate::rrfs_init::scalars::{number_from_mass, Species};
        for (slot, (mass, number, kind)) in
            [("qr", "nr", Species::Rain), ("qc", "nc", Species::Cloud), ("qi", "ni", Species::Ice)].into_iter().enumerate()
        {
            let carried = species_out.remove(number);
            let Some(q) = species_out.get(mass) else {
                if let Some(n) = carried { species_out.insert(number, n); }
                continue;
            };
            let mut n = carried.unwrap_or_else(|| vec![0.0f32; n_cells * nz]);
            let made: u64 = n.par_chunks_mut(nz).enumerate().map(|(c, column)| {
                let mut made = 0u64;
                for k in 0..nz {
                    let mass_value = q[c * nz + k];
                    if mass_value > 0.0 && column[k] == 0.0 {
                        column[k] = number_from_mass(kind, mass_value, states[c].rho[k]);
                        made += u64::from(column[k] > 0.0);
                    }
                }
                made
            }).sum();
            receipt.first_guess_hydrometeors.numbers_from_mass[slot] = made;
            species_out.insert(number, n);
        }
        for name in ["qc", "qr", "qi", "qs", "qg"] {
            if let Some(values) = species_out.get(name) {
                let max = values.par_iter().copied().reduce(|| 0.0f32, f32::max);
                receipt.first_guess_hydrometeors.max_mixing_ratio.insert(name.to_string(), max);
            }
        }
    }

    let mut computed: BTreeMap<String, emit::Computed> = BTreeMap::new();
    let mut put = |name: &str, v: Vec<f32>| {
        computed.insert(name.to_string(), emit::Computed::Floats(v));
    };
    let mut ints: Vec<(String, Vec<i32>)> = Vec::new();
    let mut put_ints = |name: &str, v: Vec<i32>| ints.push((name.to_string(), v));
    // Only the names the init stream actually carries.  The dycore's working
    // state (pressure, exner, rho_zz, theta_m, ru, rw and the perturbation
    // pair) is rebuilt by the model at start-up and is not in this file; the
    // emitter refuses on any computed name the file has no variable for, so a
    // rename upstream cannot silently drop one of these.
    if let Some(aerosols) = aerosol_state {
        put("nifa", aerosols.nifa);
        put("nwfa", aerosols.nwfa);
    }
    put("theta", flat3(&|c, k| states[c].theta[k]));
    put("rho", flat3(&|c, k| states[c].rho[k]));
    put("rho_base", flat3(&|c, k| states[c].rho_base[k]));
    put("theta_base", flat3(&|c, k| states[c].theta_base[k]));
    put("qv", flat3(&|c, k| states[c].qv[k]));
    // The other three scalars and the turbulence kinetic energy start at zero
    // (or, under first_guess_hydrometeors, at the first guess's own); they are
    // computed, not carried, so a capsule with a warm start in it cannot leak
    // into a cold one.
    put("qc", species_out.remove("qc").unwrap_or_else(|| vec![0.0; n_cells * nz]));
    put("qr", species_out.remove("qr").unwrap_or_else(|| vec![0.0; n_cells * nz]));
    put("tke", vec![0.0; n_cells * nz]);
    put("relhum", flat3(&|c, k| rh_col[c][k]));
    put("w", flat3p1(&w));
    put("u", flat3_edges(&u_edge));
    // MPAS v8.4.2 mpas_init_atm_cases.F:5014-5410. Streams may request
    // working diagnostics in addition to the standard init stream fields.
    // Recompute requested fields rather than retaining capsule state.
    let optional=capsule::requested_fields(&cfg.capsule_path,&[
        "gfs_z","spechum","pressure","pressure_base","pressure_p","exner","exner_base",
        "rho_zz","rho_p","theta_m","rtheta_base","ru","rw","qi","qs","qg",
        "v","u_init","v_init","qv_init","t_init","dz","smcrel"])?;
    for name in optional {
        let values=match name.as_str() {
            "gfs_z"=>flat3(&|c,k|gfs_z_col[c][k]),
            "spechum"=>flat3(&|c,k|sh_col[c][k]),
            "pressure"=>flat3(&|c,k|states[c].pressure[k]),
            "pressure_base"=>flat3(&|c,k|states[c].pressure_base[k]),
            "pressure_p"=>flat3(&|c,k|states[c].pressure_p[k]),
            "exner"=>flat3(&|c,k|states[c].exner[k]),
            "exner_base"=>flat3(&|c,k|states[c].exner_base[k]),
            "rho_zz"=>flat3(&|c,k|states[c].rho_zz[k]),
            "rho_p"=>flat3(&|c,k|states[c].rho_p[k]),
            "theta_m"=>flat3(&|c,k|states[c].theta_m[k]),
            "rtheta_base"=>flat3(&|c,k|states[c].rtheta_base[k]),
            "ru"=>flat3_edges(&winds.ru),
            "rw"=>flat3p1(&winds.rw),
            "qi"|"qs"|"qg"=>species_out.remove(name.as_str()).unwrap_or_else(||vec![0.;n_cells*nz]),
            "t_init"=>vec![0.;n_cells*nz],
            // Native case7 leaves these slots at the framework's cold-start
            // zero (Registry.xml:1088-1103,1286-1312,1398-1400).
            "v"=>vec![0.;n_edges*nz],
            "u_init"|"v_init"|"qv_init"=>vec![0.;nz],
            "dz"|"smcrel"=>vec![0.;n_cells*4],
            _=>unreachable!("requested diagnostic list is closed"),
        };
        put(&name,values);
    }
    // First-guess species the capsule's schema has no slot for (qi, qs, qg,
    // nc, ni, nr): the emitter creates exactly these transported species.
    for (name, values) in std::mem::take(&mut species_out) {
        put(name, values);
    }
    if let (Some((names,_,_)),Some(state))=(reconstruction_contract,reconstructed) {
        for (name,values) in reconstruction::FIELDS.into_iter().zip(state.fields) {
            if names.iter().any(|s|s==name) {put(name,values);}
        }
    }
    put("precipw", states.par_iter().map(|s| s.precipw).collect());
    put(
        "surface_pressure",
        states.par_iter().map(|s| s.surface_pressure).collect(),
    );
    put("t2m", t2m.clone());
    put("rh2", rh2.clone());
    put("q2", q2);
    put("u10", fg.u10.clone());
    put("v10", fg.v10.clone());
    put("skintemp", skintemp.clone());
    put("sst", sst);
    put("tmn", tmn);
    put("xice", xice);
    put("seaice", seaice);
    put("h_oml_initial",ocean_depth);
    put("snow", snow);
    put("snowc", snowc);
    put("snowh", snowh);
    put("xland", xland);
    put("sfc_albbck", sfc_albbck);
    put("vegfra", vegfra);
    put("snoalb", snoalb);
    put_ints("ivgtyp", ivgtyp);
    put_ints("isltyp", isltyp);
    put("shdmin", shdmin);
    put("shdmax", shdmax);
    put("tslb", flat_soil(&tslb));
    put("smois", flat_soil(&smois));
    put("sh2o", flat_soil(&sh2o));
    // smcrel is computed (identically zero) but has no variable in the init
    // stream; it is dropped here deliberately rather than by accident, and
    // the emitter would have refused if it were passed.
    let _ = &smcrel;
    // Soil layer geometry, which the file carries per cell.
    let zs_m = if cfg.land_surface == "ruc" {ruc::DEPTHS.to_vec()} else {soil::noah_mid_depths().to_vec()};
    let dzs_m = if cfg.land_surface == "ruc" {ruc::thicknesses()} else {soil::NOAH_LAYER_THICKNESS.to_vec()};
    let tile = |layers: &[f32]| -> Vec<f32> {
        let mut out = vec![0.0f32; n_cells * layers.len()];
        if !layers.is_empty() {
            out.par_chunks_mut(layers.len()).for_each(|row| row.copy_from_slice(layers));
        }
        out
    };
    let dzs_flat = tile(&dzs_m);
    let zs_flat = tile(&zs_m);
    put("dzs", dzs_flat);
    put("zs", zs_flat);
    if cfg.land_surface == "ruc" {
        // This init-stream scratch slot is zero in native real-data
        // initialization. Its geometry must still agree with RUC's nodes.
        put("dz", vec![0.; n_cells * 9]);
    }
    if ["nwfa","nifa"].iter().all(|name|fg.numbers.contains_key(name)) {
        for (name, field) in &fg.numbers {
            // Columns on rayon workers; the first failing column (in cell
            // order) is still the one reported.
            let columns:Vec<MpasResult<Vec<f32>>>=(0..n_cells).into_par_iter().map(|c| {
                let mut pairs:Vec<(f32,f32)>=fg.vert_level.iter().enumerate()
                    .filter(|(_,level)|**level!=SURFACE_LEVEL_TAG)
                    .map(|(k,_)|(fg.z[c][k],field[c][k])).collect();
                pairs.sort_by(|a,b|a.0.total_cmp(&b.0));
                let coord:Vec<f32>=pairs.iter().map(|p|p.0).collect();
                let values:Vec<f32>=pairs.iter().map(|p|p.1).collect();
                let mut col=Vec::with_capacity(nz);
                for k in 0..nz {
                    let target=0.5*(zgrid[c][k]+zgrid[c][k+1]);
                    col.push(vertical_interp(target,&coord,&values,Extrap::Constant)
                        .map_err(|_|MpasError::Refusal(format!("invalid analysis aerosol column {c}")))?.max(0.));
                }
                Ok(col)
            }).collect();
            let mut out=Vec::with_capacity(n_cells*nz);
            for col in columns { out.extend_from_slice(&col?); }
            computed.insert(name.to_string(),emit::Computed::Floats(out));
        }
    }

    for (name, v) in ints {
        computed.insert(name, emit::Computed::Ints(v));
    }

    // The three identity labels.  They are this file's, not the capsule's.
    computed.insert(
        "xtime".to_string(),
        emit::Computed::Text(cfg.start_time.clone()),
    );
    computed.insert(
        "initial_time".to_string(),
        emit::Computed::Text(cfg.start_time.clone()),
    );
    computed.insert(
        "mminlu".to_string(),
        emit::Computed::Text(cfg.landuse_table.clone()),
    );

    stage("assemble");
    let seed = format!(
        "{}|{}|{}|{}",
        cfg.met_path.display(),
        cfg.static_path.display(),
        cfg.start_time,
        cfg.capsule_path.display()
    );
    let file_id = emit::mint_file_id(&seed);
    let parent = capsule_file_id(&cfg.capsule_path).unwrap_or_default();
    receipt.emit = emit::write_init(
        &cfg.out_path,
        &cfg.capsule_path,
        computed,
        &file_id,
        &[parent],
        &cfg.provenance,
        &cfg.lineage,
    )?;

    // The per-column working sets are tens of millions of small vectors, and
    // freeing them took about 1 s on one thread on the 30 km mesh after the
    // file was already written.  A helper thread frees them instead, so the
    // caller is not held up (the binary exits; a library caller still gets
    // the memory back).
    drop(rho_zz);
    let working = (fg, states, metrics, mesh, u_edge, t_col, rh_col, sh_col, p_col,
        gfs_z_col, winds, tslb, smois, sh2o, smcrel);
    std::thread::spawn(move || drop(working));

    stage("emit");
    receipt.seconds = started.elapsed().as_secs_f64();
    Ok(receipt)
}

fn capsule_file_id(path: &Path) -> Option<String> {
    let file = netcrust::File::open(path).ok()?;
    file.attribute("file_id")?.as_string().map(str::to_string)
}

fn parse_ymd(stamp: &str) -> MpasResult<(i32, u32, u32)> {
    let bytes: Vec<char> = stamp.chars().collect();
    if bytes.len() < 10 {
        return Err(MpasError::Refusal(format!(
            "start time '{stamp}' is not YYYY-MM-DD_HH:MM:SS"
        )));
    }
    let take = |from: usize, n: usize| -> MpasResult<i64> {
        let s: String = bytes[from..from + n].iter().collect();
        s.parse::<i64>()
            .map_err(|_| MpasError::Refusal(format!("start time '{stamp}' is not a date")))
    };
    Ok((take(0, 4)? as i32, take(5, 2)? as u32, take(8, 2)? as u32))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_soil_field_name_lands_on_a_declared_layer_and_thickness() {
        for (name, k, dz) in [
            ("ST000010", 0usize, 10.0f32),
            ("ST010040", 1, 30.0),
            ("ST040100", 2, 60.0),
            ("ST100200", 3, 100.0),
            ("ST000007", 0, 7.0),
            ("ST007028", 1, 21.0),
            ("ST028100", 2, 72.0),
            ("ST100289", 3, 189.0),
        ] {
            let rule = rule_for(name).unwrap_or_else(|| panic!("{name} has no rule"));
            match rule.target {
                Target::SoilT(got_k, got_dz) => {
                    assert_eq!(got_k, k, "{name}");
                    assert_eq!(got_dz, dz, "{name}");
                }
                other => panic!("{name} mapped to {other:?}"),
            }
        }
    }

    #[test]
    fn soil_temperature_and_moisture_take_different_method_lists() {
        // Temperature leads with the parabolic sixteen-point; moisture does
        // not.  Swapping them changes coastal soil values.
        let t = rule_for("ST000010").unwrap();
        let m = rule_for("SM000010").unwrap();
        assert_eq!(t.methods[0], Method::SixteenPoint);
        assert_eq!(m.methods[0], Method::FourPoint);
        assert_eq!(t.mask.unwrap().2, 285.0, "water fill for temperature");
        assert_eq!(m.mask.unwrap().2, 1.0, "water fill for moisture");
    }

    #[test]
    fn only_profile_fields_contribute_first_guess_levels() {
        assert!(contributes_a_level(Target::T));
        assert!(contributes_a_level(Target::U));
        assert!(!contributes_a_level(Target::Psfc));
        assert!(!contributes_a_level(Target::SoilT(0, 10.0)));
        assert!(!contributes_a_level(Target::Skintemp));
    }

    #[test]
    fn first_guess_hydrometeors_are_read_only_when_asked() {
        // MPAS-A v8.4.2 case 7 has no row for them: the native table stays native.
        for (field, name) in FIRST_GUESS_SPECIES {
            assert!(rule_for(field).is_none(), "{field} must stay outside the native table");
            let rule = species_rule(field).unwrap_or_else(|| panic!("{field} has no species rule"));
            assert_eq!(rule.target, Target::Species(name));
            assert!(rule.mask.is_none());
            assert!(contributes_a_level(rule.target));
        }
        assert!(species_rule("QNWFA").is_none(), "the aerosol pair keeps its own native row");
        assert!(species_rule("TT").is_none());
    }

    #[test]
    fn a_field_the_table_does_not_know_is_ignored_not_guessed() {
        assert!(rule_for("TOTALCLOUD").is_none());
    }

    #[test]
    fn the_start_date_parses_into_its_parts() {
        assert_eq!(parse_ymd("2025-03-14_12:00:00").unwrap(), (2025, 3, 14));
        assert!(parse_ymd("nope").is_err());
    }

    #[test]
    fn deg_per_rad_is_the_fortran_parameter_not_the_accurate_one() {
        // `mpas_init_atm_llxy.F`: PI is a REAL(RKIND) parameter, so in the
        // single-precision build `180./PI` divides by f32(pi) and lands one
        // ULP below the correctly rounded 180/pi.  Pinned by bit pattern
        // because the whole point is that the two neighbouring f32 values
        // print the same to seven digits.
        assert_eq!(
            DEG_PER_RAD.to_bits(),
            0x4265_2ee0,
            "DEG_PER_RAD must be f32(180)/f32(pi) = 57.295776367, \
             not the correctly rounded 57.295780182"
        );
        let accurate = 57.295_78_f32;
        assert_eq!(accurate.to_bits(), 0x4265_2ee1, "the value NOT to use");
        assert_ne!(DEG_PER_RAD, accurate);
        assert_eq!(
            f32::from_bits(DEG_PER_RAD.to_bits() + 1),
            accurate,
            "the two differ by exactly one ULP"
        );
    }

    #[test]
    fn one_ulp_of_deg_per_rad_can_change_which_stencil_a_cell_samples() {
        // The reason the constant is pinned: at a longitude where the two
        // constants straddle a grid line, `floor(x)` differs and the
        // interpolation reads a different pair of source columns.
        let accurate = 57.295_78_f32;
        let proj = LatLonProjection {
            lat1: 90.0,
            lon1: 0.0,
            latinc: -0.25,
            loninc: 0.25,
            knowni: 1.0,
            knownj: 1.0,
            nx: 1440,
            ny: 721,
        };
        // Search a band of longitudes for a radian value whose degree
        // conversion straddles a grid line under the two constants.
        let mut found = None;
        for n in 0..2_000_000u32 {
            let lon_rad = 4.0 + (n as f32) * 1.0e-6;
            let a = lon_rad * DEG_PER_RAD;
            let b = lon_rad * accurate;
            let (xa, _) = proj.locate(0.0, a);
            let (xb, _) = proj.locate(0.0, b);
            if xa.floor() != xb.floor() {
                found = Some((lon_rad, xa, xb));
                break;
            }
        }
        let (lon_rad, xa, xb) = found.expect(
            "the two constants must disagree about a grid line somewhere in \
             4.0..6.0 radians of longitude",
        );
        assert_ne!(xa.floor(), xb.floor(), "lon_rad = {lon_rad}");
    }
}

// Validation-only stage capture; never active unless explicitly requested.
fn gpu_oracle_dump(dir: &str, stage: &str, arrays: Vec<(String, Vec<f32>)>) -> MpasResult<()> {
    use std::io::Write;
    let root = Path::new(dir).join(stage);
    std::fs::create_dir_all(&root).map_err(|e| MpasError::Refusal(e.to_string()))?;
    let mut index = BTreeMap::new();
    for (name, values) in arrays {
        let path = root.join(format!("{name}.f32"));
        let mut out = std::io::BufWriter::new(std::fs::File::create(&path)
            .map_err(|e| MpasError::Refusal(e.to_string()))?);
        for x in &values { out.write_all(&x.to_le_bytes()).map_err(|e| MpasError::Refusal(e.to_string()))?; }
        out.flush().map_err(|e| MpasError::Refusal(e.to_string()))?;
        index.insert(name, values.len());
    }
    let json = serde_json::to_vec_pretty(&index).map_err(|e| MpasError::Refusal(e.to_string()))?;
    std::fs::write(root.join("index.json"), json).map_err(|e| MpasError::Refusal(e.to_string()))?;
    Ok(())
}
