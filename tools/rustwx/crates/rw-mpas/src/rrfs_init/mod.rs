//! RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_cases.F:4043-7742 at 1d6e370e.
//! Separate case-7 producer; the ordinary MPAS and ArWen selections are unchanged.
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

pub use crate::init::{capsule,emit,hinterp,advection,reconstruction,artifact};
pub mod dynamics;
pub mod aerosols;
pub mod vinterp;
pub mod metfile;
pub mod projection;
pub mod auxiliary;
pub mod surface;
pub mod vertical;
pub mod scalars;
pub mod soil;
pub mod boundary;
pub mod job;
pub mod prepare;

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
    Scalar(usize),
    SnowH,
    Canwat,
    Vegfra,
    Shdmin,
    Shdmax,
    Skintemp,
    Snow,
    Seaice,
    /// Soil temperature / moisture at a fixed first-guess layer.
    SoilT(usize, f32),
    SoilM(usize, f32),
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

/// RRFS fork src/core_init_atmosphere/mpas_init_atm_cases.F:5637-6246 at 1d6e370e.
fn soil_rule(s:&MetSlab)->Option<FieldRule> {
    let rap=["NCEP RAP Model","NCEP HRRR Model","NOAA GSD HRRR Model","unknown model from NCEP"].contains(&s.map_source.trim());
    let (k,dz)=match s.field.get(5..)? {"000"=>(0,0.),"001" if rap=>(1,0.01),"004"=>(2,0.03),"010"=>(3,0.06),"030"=>(4,0.20),"060"=>(5,0.30),"100"=>(6,0.40),"160"=>(7,0.60),"300"=>(8,1.40),_=>return None};
    match s.field.get(..5)? {"SOILT"=>Some(FieldRule{target:Target::SoilT(k,dz),methods:SOIL_T_METHODS,mask:Some((0.,0,285.))}),"SOILM"=>Some(FieldRule{target:Target::SoilM(k,dz),methods:SOIL_M_METHODS,mask:Some((0.,0,1.))}),_=>None}
}
/// RRFS fork src/core_init_atmosphere/mpas_init_atm_cases.F:5455-6246 at 1d6e370e.
fn soil_endpoint(s:&MetSlab,k:usize)->f32 {
    if soil_rule(s).is_some() {return [0.,0.01,0.04,0.10,0.30,0.60,1.,1.60,3.][k];}
    s.field.get(s.field.len().saturating_sub(3)..).and_then(|x|x.parse::<f32>().ok()).unwrap_or(0.)
}

/// The dispatch table out of `init_atm_case_gfs`, one row per field name.
///
/// The soil rows carry the layer index and the layer *thickness in
/// centimetres* the Fortran assigns alongside them.  Only the thickness
/// survives: `init_soil_layers_depth` recomputes the mid-depths cumulatively
/// and overwrites whatever the dispatch put in `zs_fg`.
/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_cases.F:4043-7742 at 1d6e370e.
fn rule_for(field: &str) -> Option<FieldRule> {
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
    if let Some(i)=scalars::input_index(field) {return Some(profile(Target::Scalar(i)));}
    Some(match field {
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
        "SNOWH" => FieldRule {target:Target::SnowH,methods:SNOW_METHODS,mask:None},
        "CANWAT" => FieldRule {target:Target::Canwat,methods:SNOW_METHODS,mask:Some((-1.,0,0.))},
        "VEGFRA" => FieldRule {target:Target::Vegfra,methods:SNOW_METHODS,mask:Some((-1.,0,0.))},
        "SHDMIN" => FieldRule {target:Target::Shdmin,methods:SNOW_METHODS,mask:Some((-1.,0,0.))},
        "SHDMAX" => FieldRule {target:Target::Shdmax,methods:SNOW_METHODS,mask:Some((-1.,0,0.))},
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

/// Whether a field's level participates in the first-guess level table.
/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_cases.F:4043-7742 at 1d6e370e.
fn contributes_a_level(target: Target) -> bool {
    matches!(
        target,
        Target::Scalar(_) | Target::T | Target::Rh | Target::Sh | Target::Z | Target::P | Target::U | Target::V
    )
}

/// Everything the caller must state.  There are no defaults on the switches
/// that select physics: a wrong guess produces a structurally perfect file
/// with wrong numbers, which is exactly the failure mode this project has
/// already been bitten by once on `--map-source`.
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InitConfig {
    /// RRFS namelist switches; caller supplies the suite row.
    pub tempo_rap: bool,
    pub smoke: bool,
    pub dust: bool,
    pub n_soil_layers: usize,
    pub nvegopt: usize,
    pub met_path: PathBuf,
    /// Optional case-7 auxiliary intermediate inputs, read after main meteorology.
    pub seaice_met_path: Option<PathBuf>,
    pub omld_met_path: Option<PathBuf>,
    /// config_input_sst: reread this optional config_sfc_prefix input.
    pub sfc_met_path: Option<PathBuf>,
    /// Explicit climatology path; None uses MPAS's local filename.
    pub gocart_path: Option<PathBuf>,
    /// config_gocartlevels; None uses the capsule dimension or native 30.
    pub gocart_levels: Option<usize>,
    pub static_path: PathBuf,
    pub capsule_path: PathBuf,
    /// The native init the capsule's `zgrid` is asserted against.
    pub reference_path: PathBuf,
    pub out_path: PathBuf,
    pub start_time: String,
    pub n_fg_levels: usize,
    pub n_fg_soil_levels: usize,
    #[serde(deserialize_with="crate::rrfs_init::job::extrap") ]
    pub extrap_airtemp: Extrap,
    pub use_spechumd: bool,
    pub theta_adv_order: i32,
    pub coef_3rd_order: f32,
    #[serde(deserialize_with="crate::rrfs_init::job::virtual_factor") ]
    pub virtual_factor: VirtualFactor,
    #[serde(deserialize_with="crate::rrfs_init::job::deep_moisture") ]
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
    #[serde(default)]
    pub seaice_cleanup: soil::SeaiceCleanup,
    /// What `oned` does when its guard products underflow.  Not a namelist
    /// option: it is a property of how the reference was *compiled*.  See
    /// [`Underflow`].  No default, because the two answers differ by several
    /// relative-humidity points at the cells where it fires and both files
    /// look perfectly well formed.
    #[serde(deserialize_with="crate::rrfs_init::job::underflow") ]
    pub oned_underflow: Underflow,
    pub provenance: String,
    /// Who this file says it is: `model_name`, `core_name`, `version` and
    /// `git_version`, the four lineage attributes `rw_mpas_lbc` refuses a
    /// `--grid` for lacking.  See [`Lineage`].
    #[serde(deserialize_with="crate::rrfs_init::job::lineage") ]
    pub lineage: Lineage,
    /// Rain, cloud and ice number where the first guess carries the mass but
    /// not the number (a one-moment parent such as GFS). `from-mass` (the
    /// default) diagnoses them with the fork's calcnfromq, as the boundaries
    /// do; `zero` keeps the fork's case 7 as shipped, for native replays.
    /// See [`scalars::initialize_init_numbers`].
    #[serde(default="crate::rrfs_init::job::absent_numbers_default",deserialize_with="crate::rrfs_init::job::absent_numbers")]
    pub absent_numbers: scalars::AbsentNumbers,
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
pub use crate::init::Lineage;

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
    /// `zero` or `from-mass`, and the rain, cloud and ice points diagnosed.
    pub absent_numbers: String,
    pub numbers_from_mass: [u64; 3],
    pub aerosols: aerosols::AerosolReceipt,
    pub soil: soil::SoilCounts,
    pub seaice: soil::SeaiceCounts,
    pub emit: emit::EmitLedger,
    /// Named loudly so nobody markets a capsule-fed init as a from-scratch one.
    pub deferrals: Vec<String>,
    /// Wall-clock seconds per stage of this run, in order (StageClock).
    pub stage_seconds: Vec<(String, f64)>,
    pub seconds: f64,
}

/// Per-stage wall-clock marks for a producer run: each mark is the time since
/// the previous one, written to the stage log (stderr) and kept for the
/// receipt. It reads the clock only; it adds no syncs, copies or work.
pub struct StageClock {
    label: &'static str,
    last: std::time::Instant,
    pub marks: Vec<(String, f64)>,
}

impl StageClock {
    pub fn new(label: &'static str) -> Self {
        Self { label, last: std::time::Instant::now(), marks: Vec::new() }
    }

    pub fn mark(&mut self, stage: &str) {
        let now = std::time::Instant::now();
        let seconds = (now - self.last).as_secs_f64();
        self.last = now;
        let epoch = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)
            .map_or(0.0, |d| d.as_secs_f64());
        eprintln!("[{}] {stage}: {seconds:.3} s (ends {epoch:.1})", self.label);
        self.marks.push((stage.to_string(), (seconds * 1000.0).round() / 1000.0));
    }
}

#[derive(serde::Serialize)]
pub struct FirstGuess {
    pub scalars: Vec<Vec<Vec<f32>>>,
    pub snowh: Vec<f32>,
    pub canwat: Vec<f32>,
    pub vegfra: Vec<f32>,
    pub shdmin: Vec<f32>,
    pub shdmax: Vec<f32>,
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
    zs_fg:Vec<f32>,
}

/// `n` zeroed columns of `width`, allocated on every core (one small heap
/// block per cell or edge: about 45 million of them for the first guess on
/// conus3km, which one thread spent seconds allocating).
pub(crate) fn zero_columns(n: usize, width: usize) -> Vec<Vec<f32>> {
    (0..n).into_par_iter().map(|_| vec![0.0f32; width]).collect()
}

/// Interpolate every point on every core. Each run of points counts the
/// subnormal guard products of `oned` (the receipt's oned_underflow_sites) in
/// its own counter and adds it to the shared one once: the moisture slabs hit
/// that guard often, and one counter shared by every core turned each hit into
/// a cross-core cache-line transfer. Same values, same total count.
pub(crate) fn sample_points(n: usize, ctx: &InterpContext,
    f: impl Fn(usize, &InterpContext) -> f32 + Sync) -> Vec<f32>
{
    const RUN: usize = 2048;
    let mut out = vec![0.0f32; n];
    out.par_chunks_mut(RUN).enumerate().for_each(|(run, slots)| {
        let local = std::sync::atomic::AtomicU64::new(0);
        let own = InterpContext {
            array: ctx.array,
            msgval: ctx.msgval,
            maskval: ctx.maskval,
            mask: ctx.mask,
            underflow: ctx.underflow,
            underflow_hits: ctx.underflow_hits.map(|_| &local),
        };
        for (k, slot) in slots.iter_mut().enumerate() {
            *slot = f(run * RUN + k, &own);
        }
        if let Some(shared) = ctx.underflow_hits {
            let hits = local.load(std::sync::atomic::Ordering::Relaxed);
            if hits > 0 {
                shared.fetch_add(hits, std::sync::atomic::Ordering::Relaxed);
            }
        }
    });
    out
}

/// Write one first-guess level into every column, on every core: column i
/// takes values[i] at `slot`, exactly what the serial store loop did.
pub(crate) fn scatter_level(columns: &mut [Vec<f32>], slot: usize, values: &[f32]) {
    columns.par_iter_mut().zip(values.par_iter()).for_each(|(column, &value)| column[slot] = value);
}

/// Complete first-guess surface extraction and pressure/height completion.
/// MPAS-A v8.4.2 core_init_atmosphere/mpas_init_atm_cases.F:4710-4769.
/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_cases.F:4043-7742 at 1d6e370e.
pub fn prepare_vertical_pass(fg: &mut FirstGuess) -> (Vec<f32>,Vec<f32>,Vec<f32>) {
    let n_cells = fg.t.len();
    let nfg = fg.vert_level.len();
    let sfc = fg.vert_level.iter().position(|&v|v == SURFACE_LEVEL_TAG).expect("horizontal pass checked surface level");
    // Surface fields lifted off the first-guess surface level.
    let t2m: Vec<f32> = (0..n_cells).map(|c| fg.t[c][sfc]).collect();
    let rh2: Vec<f32> = (0..n_cells).map(|c| fg.rh[c][sfc]).collect();

    let q2 = (0..n_cells).map(|c|dynamics::diagnose_q2_surface_pass(t2m[c],fg.psfc[c],rh2[c])).collect();

    // Isobaric data carries no 3-D pressure; fill it from the level tags.
    let isobaric = fg
        .p
        .iter()
        .all(|col| col[..nfg].iter().all(|&v| v == 0.0));
    if isobaric {
        for c in 0..n_cells {
            for (k, &lvl) in fg.vert_level.iter().enumerate() {
                fg.p[c][k] = if lvl == SURFACE_LEVEL_TAG {
                    fg.psfc[c]
                } else {
                    lvl
                };
            }
        }
    } else {
        for c in 0..n_cells {
            fg.p[c][sfc] = fg.psfc[c];
            fg.z[c][sfc] = fg.soilz[c];
        }
    }

    (t2m,rh2,q2)
}

#[allow(clippy::too_many_lines)]
/// Complete meteorological horizontal driver: MPAS-A v8.4.2
/// core_init_atmosphere/mpas_init_atm_cases.F:3450-4680.
/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_cases.F:4043-7742 at 1d6e370e.
pub fn horizontal_pass(
    slabs: &[MetSlab],
    mesh: &capsule::MeshGeometry,
    statics: &capsule::Statics,
    nfg: usize,
    nfgs: usize,
    underflow: hinterp::Underflow,
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
        scalars:(0..scalars::COUNT).map(|_| zero_columns(n_cells, nfg)).collect(),
        snowh:vec![0.;n_cells],canwat:vec![0.;n_cells],vegfra:vec![0.;n_cells],shdmin:vec![0.;n_cells],shdmax:vec![0.;n_cells],
        vert_level: Vec::new(),
        t: zero_columns(n_cells, nfg),
        rh: zero_columns(n_cells, nfg),
        sh: zero_columns(n_cells, nfg),
        z: zero_columns(n_cells, nfg),
        p: zero_columns(n_cells, nfg),
        u: zero_columns(n_edges, nfg),
        v: zero_columns(n_edges, nfg),
        psfc: vec![0.0; n_cells],
        pmsl: vec![0.0; n_cells],
        soilz: vec![0.0; n_cells],
        skintemp: vec![0.0; n_cells],
        snow: vec![0.0; n_cells],
        xice: vec![0.0; n_cells],
        u10: vec![0.0; n_cells],
        v10: vec![0.0; n_cells],
        st: vec![vec![0.0; nfgs]; n_cells],
        sm: vec![vec![0.0; nfgs]; n_cells],
        dzs_cm: vec![0.0; nfgs],
        zs_fg:vec![0.;nfgs],
    };

    let cell_lat: Vec<f32> = mesh.lat_cell.iter().map(|v| v * DEG_PER_RAD).collect();
    let cell_lon: Vec<f32> = mesh.lon_cell.iter().map(|v| v * DEG_PER_RAD).collect();
    let edge_lat: Vec<f32> = mesh.lat_edge.iter().map(|v| v * DEG_PER_RAD).collect();
    let edge_lon: Vec<f32> = mesh.lon_edge.iter().map(|v| v * DEG_PER_RAD).collect();

    for s in slabs {
        receipt.met_records_read += 1;
        let Some(rule) = soil_rule(s).or_else(||rule_for(&s.field)) else {
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

        let sample = |i: usize, lat: &[f32], lon: &[f32], ctx: &InterpContext| -> f32 {
            let (x,y)=if surface_wind {proj.locate_surface_wind(lat[i],lon[i])}
                else {proj.locate(lat[i],lon[i])};
            hinterp::interp_sequence(x, y, ctx, rule.methods, 0)
        };

        if on_edges {
            let values = sample_points(n_edges, &ctx, |e, ctx| sample(e, &edge_lat, &edge_lon, ctx));
            let dest = if rule.target == Target::U {
                &mut fg.u
            } else {
                &mut fg.v
            };
            scatter_level(dest, level_index, &values);
            continue;
        }

        // MPAS v8.4.2 cases.F:3532-3534,4591-4623,4654-4663.
        // Default -1 masks only the destination; the surface-wind pass
        // ignores destination masks and always supplies the source mask.
        let masked_flag = if surface_wind {None} else {Some(rule.mask.map(|(_,m,_)|m).unwrap_or(-1))};
        let fillval = rule.mask.map(|(_, _, f)| f).unwrap_or(0.0);
        let values = sample_points(n_cells, &ctx, |c, ctx| {
            if let Some(m) = masked_flag {
                if statics.landmask[c] == m {
                    return fillval;
                }
            }
            sample(c, &cell_lat, &cell_lon, ctx)
        });

        match rule.target {
            Target::Scalar(i)=> scatter_level(&mut fg.scalars[i], level_index, &values),
            Target::SnowH=>fg.snowh.copy_from_slice(&values),
            Target::Canwat=>fg.canwat.copy_from_slice(&values),
            Target::Vegfra=>fg.vegfra.copy_from_slice(&values),
            Target::Shdmin=>fg.shdmin.copy_from_slice(&values),
            Target::Shdmax=>fg.shdmax.copy_from_slice(&values),
            Target::T => {
                // The surface level lands in the same column slot as any
                // other: it is the 200100.0 tag, not a separate array, that
                // makes it the surface later on.
                scatter_level(&mut fg.t, level_index, &values);
            }
            Target::Rh => {
                scatter_level(&mut fg.rh, level_index, &values);
            }
            Target::Sh => {
                scatter_level(&mut fg.sh, level_index, &values);
            }
            Target::Z => {
                scatter_level(&mut fg.z, level_index, &values);
            }
            Target::P => {
                scatter_level(&mut fg.p, level_index, &values);
            }
            Target::U => fg.u10.copy_from_slice(&values),
            Target::V => fg.v10.copy_from_slice(&values),
            Target::Psfc => fg.psfc.copy_from_slice(&values),
            Target::Pmsl => fg.pmsl.copy_from_slice(&values),
            Target::SoilHgt => fg.soilz.copy_from_slice(&values),
            Target::Skintemp => fg.skintemp.copy_from_slice(&values),
            Target::Snow => fg.snow.copy_from_slice(&values),
            Target::Seaice => fg.xice.copy_from_slice(&values),
            Target::SoilT(k, dz) => {
                if k >= nfgs {
                    return Err(MpasError::Refusal(format!(
                        "{} lands on first-guess soil level {} but only {nfgs} were declared",
                        s.field,
                        k + 1
                    )));
                }
                fg.dzs_cm[k] = dz;
                fg.zs_fg[k]=soil_endpoint(s,k);
                scatter_level(&mut fg.st, k, &values);
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
                fg.zs_fg[k]=soil_endpoint(s,k);
                scatter_level(&mut fg.sm, k, &values);
            }
        }
    }

    receipt.oned_underflow = underflow.label().to_string();
    receipt.oned_underflow_sites =
        underflow_hits.load(std::sync::atomic::Ordering::Relaxed);
    receipt.first_guess_levels_found = fg.vert_level.len();
    receipt.surface_level_present = fg.vert_level.contains(&SURFACE_LEVEL_TAG);
    if !receipt.surface_level_present {
        return Err(MpasError::Refusal(
            "no first-guess level is tagged 200100.0.  Case 7 identifies the surface level by \
             that tag alone: without it the surface pressure column has no surface, every \
             profile silently keeps a level it should have dropped, and t2m/rh2 stay zero"
                .to_string(),
        ));
    }
    if fg.dzs_cm.iter().enumerate().any(|(i,&v)| v < 0.0 || (v==0.0 && !(nfgs==9 && i==0))) {
        return Err(MpasError::Refusal(format!(
            "the first-guess soil column is incomplete: layer thicknesses {:?} (cm).  A zero \
             means a soil level this run declared was never delivered by the intermediate file",
            fg.dzs_cm
        )));
    }
    Ok(fg)
}

/// Build one init file.
/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_cases.F:4043-7742 at 1d6e370e.
pub fn build_init(cfg: &InitConfig) -> MpasResult<InitReceipt> {
    let started = std::time::Instant::now();
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
            "reconstruction geometry is recomputed from the supplied mesh at the native \
             case7 boundary; vertical metrics remain supplied by the capsule contract".to_string(),
        ],
        ..Default::default()
    };
    let mut clock = StageClock::new("case7");

    let mesh = capsule::read_mesh_geometry(&cfg.static_path)?;
    let mut statics = capsule::read_statics(&cfg.static_path, mesh.n_cells)?;
    // The terrain every later step sees is the SMOOTHED one the vertical-grid
    // block wrote back, not the raw static field.  See read_smoothed_terrain.
    statics.ter = capsule::read_smoothed_terrain(&cfg.capsule_path, mesh.n_cells)?;
    let metrics = capsule::read_vertical_metrics(&cfg.capsule_path, mesh.n_cells, mesh.n_edges)?;
    receipt.zgrid_values_asserted_identical =
        capsule::assert_zgrid_identical(&metrics, &cfg.reference_path)?;
    receipt.n_cells = mesh.n_cells;
    receipt.n_edges = mesh.n_edges;
    receipt.n_vert_levels = metrics.n_vert_levels;
    clock.mark("read mesh, statics, vertical metrics");

    let slabs = metfile::read_met_file(&cfg.met_path)?;
    clock.mark("read first-guess file");
    let mut fg = horizontal_pass(&slabs, &mesh, &statics, cfg.n_fg_levels,
        cfg.n_fg_soil_levels,cfg.oned_underflow,&mut receipt)?;
    clock.mark("horizontal interpolation");
    if let Some(path)=&cfg.seaice_met_path {
        receipt.seaice_met_records_used=auxiliary::interpolate(&metfile::read_met_file(path)?,auxiliary::Kind::SeaIce,
            &mesh.lat_cell,&mesh.lon_cell,&statics.landmask,&mut fg.xice,cfg.oned_underflow)?;
    }
    let mut ocean_depth=vec![0.;mesh.n_cells];
    if let Some(path)=&cfg.omld_met_path {
        receipt.omld_met_records_used=auxiliary::interpolate(&metfile::read_met_file(path)?,auxiliary::Kind::OceanDepth,
            &mesh.lat_cell,&mesh.lon_cell,&statics.landmask,&mut ocean_depth,cfg.oned_underflow)?;
    }

    let last_rotation=if let Some(path)=&cfg.omld_met_path {
        metfile::read_met_file(path)?.pop()
    } else if let Some(path)=&cfg.seaice_met_path {metfile::read_met_file(path)?.pop()} else {slabs.last().cloned()};
    if let Some(header)=last_rotation.as_ref().filter(|h|h.wind_grid_rel) {
        projection::rotate_velocities(header,&mesh.lat_edge,&mesh.lon_edge,&mut fg.u,&mut fg.v)?;
    }
    let nz = metrics.n_vert_levels;
    let n_cells = mesh.n_cells;
    let n_edges = mesh.n_edges;
    let nfg = fg.vert_level.len();

    let (t2m,rh2,_first_q2) = prepare_vertical_pass(&mut fg);
    clock.mark("rotate winds, surface pass");
    let mut scalar_state=scalars::interpolate(&fg.scalars,&fg.vert_level,&fg.z,&metrics.zgrid,false,cfg.smoke,cfg.dust)?;
    clock.mark("scalar vertical interpolation");

    // Vertical interpolation, cell by cell.
    let zgrid = &metrics.zgrid;
    let extrap_t = cfg.extrap_airtemp;
    type Column = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, f32);
    // Fork cases.F:7180-7413: logging and pressure libm calls seed the next
    // temperature column's live strict weights. Preserve that driver order.
    // vinterp::carried_columns runs the columns on every core and hands each
    // piece the registers the serial loop would have carried into it.
    let first_weights = vinterp::TemperatureWeights::from_log(
        mesh.lat_cell.iter().copied().fold(f32::NEG_INFINITY, f32::max));
    let column = |c: usize, temperature_weights: &mut vinterp::TemperatureWeights| -> MpasResult<Column> {
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
            if coord.iter().copied().fold(f32::NEG_INFINITY,f32::max)>1.0e6
                || coord.iter().copied().fold(f32::INFINITY,f32::min)<0.1 {
                *temperature_weights=vinterp::TemperatureWeights::from_log(coord[nfg-1]);
            }
            for k in 0..nz {
                let target = 0.5 * (zgrid[c][k] + zgrid[c][k + 1]);
                t[k] = temperature_weights.interpolate(target, &coord[..nfg - 1], &value[..nfg - 1], extrap_t)
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

            let log_pressure:Vec<f32>=(0..nfg).map(|k|temperature_weights.log(fg.p[c][k])).collect();
            prep(&mut coord, &mut value, &|k| log_pressure[k]);
            for k in 0..nz {
                let target = 0.5 * (zgrid[c][k] + zgrid[c][k + 1]);
                let interpolated=vertical_interp(target, &coord[..nfg - 1], &value[..nfg - 1], Extrap::Linear)
                    .unwrap_or(value[nfg - 2]);
                pressure[k]=temperature_weights.exp(interpolated);
            }

            // Surface pressure is the one field that keeps the surface level:
            // the full column, sentinel included, targeted at zgrid(1).
            prep(&mut coord, &mut value, &|k| fg.p[c][k].ln());
            let psfc = vertical_interp(zgrid[c][0], &coord, &value, Extrap::Linear)
                .unwrap_or(value[nfg - 1])
                .exp();

            Ok((t, relhum, spechum, pressure, gfs_z, psfc))
        };
    let columns: Vec<MpasResult<Column>> = vinterp::carried_columns(n_cells, first_weights, &column);

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
    clock.mark("cell column interpolation");

    // Edge winds: rotate to the edge normal, then interpolate on the two-cell
    // mean coordinate to the four-point mean target.
    let u_edge: Vec<Vec<f32>> = (0..n_edges)
        .into_par_iter()
        .map(|e| {
            let [c1, c2] = mesh.cells_on_edge[e];
            // Fork cases.F:7419-7435 reads the cold extra-cell pool at a
            // regional boundary. It does not substitute the last real cell.
            let c1 = c1.checked_sub(1).filter(|&c|c<n_cells);
            let c2 = c2.checked_sub(1).filter(|&c|c<n_cells);
            let height = |cell:Option<usize>,k:usize|cell.map_or(0.0,|c|fg.z[c][k]);
            let target = |cell:Option<usize>,k:usize|cell.map_or(0.0,|c|zgrid[c][k]);
            let angle = mesh.angle_edge[e];
            let mut coord = vec![0.0f32; nfg];
            let mut value = vec![0.0f32; nfg];
            for k in 0..nfg {
                coord[k] = if fg.vert_level[k] == SURFACE_LEVEL_TAG {
                    SURFACE_SENTINEL
                } else {
                    dynamics::edge_column_coordinate(height(c1,k), height(c2,k))
                };
                value[k] = dynamics::edge_normal_wind(angle, fg.u[e][k], fg.v[e][k]);
            }
            sorted_column(&mut coord, &mut value);
            (0..nz)
                .map(|k| {
                    let target = dynamics::edge_target_height(
                        target(c1,k),
                        target(c1,k + 1),
                        target(c2,k),
                        target(c2,k + 1),
                    );
                    vertical_interp(target, &coord[..nfg-1], &value[..nfg-1], Extrap::Constant).expect("constant edge extrapolation is defined")
                })
                .collect()
        })
        .collect();

    // Moisture, then RH over ice, then the column chain.
    // RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_cases.F:7484-7528
    // at 1d6e370e. Strict MINVAL/MAXVAL ignore NaNs when an ordered value
    // exists, while an entirely NaN field retains NaN. A NaN-only "nonzero"
    // test selects absent specific humidity incorrectly in mixed columns.
    let sh_min = sh_col.iter().flatten().copied().fold(f32::NAN, f32::min);
    let sh_max = sh_col.iter().flatten().copied().fold(f32::NAN, f32::max);
    let use_sh = cfg.use_spechumd && !(sh_min == 0.0 && sh_max == 0.0);
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

    rh_col.par_iter_mut().zip(t_col.par_iter())
        .for_each(|(rh, t)| dynamics::convert_relhum_wrt_ice(t, rh));
    clock.mark("edge winds, column dynamics");

    // Fork cases.F:7600-7612: number from mass at the first rho_zz
    // (dynamics::build_column step 4, recomputed per point, no array).
    {
        use dynamics::constants::{CP, P0, RGAS, RVORD};
        let nzc = nz.max(1);
        receipt.numbers_from_mass = scalars::initialize_init_numbers(&mut scalar_state, n_cells * nz, |i| {
            let (c, k) = (i / nzc, i % nzc);
            let p = p_col[c][k];
            let exner = (p / P0).powf(RGAS / CP);
            let theta = t_col[c][k] * (P0 / p).powf(RGAS / CP);
            let qv = states[c].qv[k];
            p / RGAS / (exner * theta * (1.0 + (RVORD - 1.0) * qv)) / (1.0 + qv)
        }, cfg.absent_numbers);
        receipt.absent_numbers = cfg.absent_numbers.label().to_string();
    }
    clock.mark("numbers from mass");

    let mut it_max = 0u32;
    let mut it_sum = 0u64;
    let mut it_n = 0u64;
    let mut capped = 0usize;
    for s in &states {
        let mut hit = false;
        for &i in &s.hydrostatic_iterations[1..] {
            it_max = it_max.max(i);
            it_sum += i as u64;
            it_n += 1;
            if i >= 30 {
                hit = true;
            }
        }
        if hit {
            capped += 1;
        }
    }
    receipt.hydrostatic_iterations_max = it_max;
    receipt.hydrostatic_iterations_mean = if it_n > 0 {
        it_sum as f64 / it_n as f64
    } else {
        0.0
    };
    receipt.hydrostatic_cells_hitting_the_cap = capped;

    let nzp1=nz+1;
    // The default init stream excludes these met-stage diagnostics. If the
    // requested schema carries them, compute them from this run's winds.
    // Fork cases.F:7442-7451 recomputes geometry even when vertical-grid
    // construction was disabled and a public invariant supplied its metrics.
    let reconstruction_geometry=prepare::reconstruction_geometry(&cfg.static_path)
        .map_err(MpasError::Refusal)?;
    let mut reconstruction_contract=capsule::read_reconstruction_contract(&cfg.capsule_path,
        n_cells,mesh.edges_on_cell[0].len())?;
    if let Some((_,coefficients,_))=&mut reconstruction_contract {
        *coefficients=reconstruction_geometry.iter().find(|v|v.name=="coeffs_reconstruct")
            .and_then(|v|v.values.clone()).expect("complete geometry always emits coefficients");
    }
    let reconstructed=if let Some((_,coefficients,sphere))=&reconstruction_contract {
        Some(reconstruction::reconstruct(&u_edge,&mesh.n_edges_on_cell,&mesh.edges_on_cell,
            coefficients,&mesh.lat_cell,&mesh.lon_cell,*sphere)?)
    } else {None};
    clock.mark("reconstruction geometry");
    let rho_zz:Vec<&[f32]>=states.iter().map(|state|state.rho_zz.as_slice()).collect();
    let winds=dynamics::diagnose_winds(&u_edge,&rho_zz,&mesh.cells_on_edge,&mesh.n_edges_on_cell,
        &mesh.edges_on_cell,&metrics,cfg.theta_adv_order,cfg.coef_3rd_order);
    let w=&winds.w;
    clock.mark("ru, rw, w");

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
        receipt.sfc_met_records_used = surface::interpolate(&slabs,&mesh.lat_cell,&mesh.lon_cell,&statics.landmask,
            &mut sst,&mut fg.xice,cfg.oned_underflow)?;
        surface::initialize_sst(cfg.frac_seaice,&statics.landmask,&sst,
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
        &fg.zs_fg,
        cfg.n_soil_layers,
        cfg.deep_moisture,
    )?;
    soil_counts.land_moisture_repaired = repaired;
    soil_counts.land_temperature_flagged = flagged;
    receipt.soil = soil_counts;

    let (year, month, day) = parse_ymd(&cfg.start_time)?;
    let mut surface=soil::init_surface_companions(&statics.landmask,&fg.snow,
        &statics.albedo12m,&statics.greenfrac,year,month,day);
    if cfg.nvegopt!=1 && fg.vegfra.iter().any(|&v|v!=0.) {
        surface.vegetation=fg.vegfra.clone();surface.minimum=fg.shdmin.clone();surface.maximum=fg.shdmax.clone();
    }
    if fg.snowh.iter().any(|&v|v!=0.) {surface.snow_height=fg.snowh.clone();}
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
        cfg.seaice_cleanup,
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
    let q2: Vec<f32> = (0..n_cells)
        .into_par_iter()
        .map(|c| dynamics::diagnose_q2(t2m[c], psfc[c], rh2[c]))
        .collect();
    clock.mark("soil, surface, sea ice");

    // Assemble the computed set, cell-major with the vertical fastest. Each
    // cell's (or edge's) run of levels is filled on its own core; the words
    // and their order are those of the serial loops.
    let flat3 = |f: &(dyn Fn(usize, usize) -> f32 + Sync)| -> Vec<f32> {
        let mut out = vec![0.0f32; n_cells * nz];
        out.par_chunks_mut(nz.max(1)).enumerate().for_each(|(c, column)| {
            for (k, slot) in column.iter_mut().enumerate() {
                *slot = f(c, k);
            }
        });
        out
    };
    let flat_rows = |src: &[Vec<f32>], width: usize| -> Vec<f32> {
        let mut out = vec![0.0f32; src.len() * width];
        out.par_chunks_mut(width.max(1)).zip(src.par_iter())
            .for_each(|(row, values)| row.copy_from_slice(&values[..width]));
        out
    };
    let flat3_edges = |src: &Vec<Vec<f32>>| -> Vec<f32> { flat_rows(&src[..n_edges], nz) };
    let flat3p1 = |src: &Vec<Vec<f32>>| -> Vec<f32> { flat_rows(&src[..n_cells], nzp1) };
    let flat_soil = |src: &Vec<Vec<f32>>| -> Vec<f32> {
        let mut out = Vec::with_capacity(n_cells * cfg.n_soil_layers);
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
    if has_nifa != has_nwfa {
        return Err(MpasError::Refusal("init capsule must carry both nifa and nwfa; emitting only one loses a Thompson aerosol scalar".into()));
    }
    let native_path = PathBuf::from(aerosols::NATIVE_FILENAME);
    let gocart_path = cfg.gocart_path.as_deref().unwrap_or(&native_path);
    let use_rap=scalars::use_rap_aerosols(cfg.tempo_rap,&scalar_state);
    let aerosol_state = if has_nifa && use_rap {
        Some(aerosols::AerosolState {nifa:std::mem::take(&mut scalar_state[11]),nwfa:std::mem::take(&mut scalar_state[10])})
    } else if has_nifa {
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
        if cfg.gocart_path.is_some() || gocart_path.try_exists()? {
            return Err(MpasError::Refusal(format!(
                "GOCART climatology {} selected but the init capsule has no nifa/nwfa slots; emitting would discard the climatology",
                gocart_path.display()
            )));
        }
        None
    };
    clock.mark("GOCART aerosols");

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
    // The other three scalars and the turbulence kinetic energy start at zero;
    // they are computed-as-zero, not carried, so a capsule with a warm start
    // in it cannot leak into a cold one.
    // Each species is handed to the emitter once, so it is moved, not copied
    // (a 552 MB copy per species on one core on conus3km).
    put("qc", std::mem::take(&mut scalar_state[0]));
    put("qr", std::mem::take(&mut scalar_state[1]));
    // The fork's standard initial-conditions stream declares no tke slot.
    // Preserve an explicitly requested diagnostic without adding a field to
    // Registry.xml's immutable stream (fork Registry.xml:541-692 at 1d6e370e).
    if !capsule::requested_fields(&cfg.capsule_path,&["tke"])?.is_empty() {
        put("tke", vec![0.0; n_cells * nz]);
    }
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
            "qi"=>std::mem::take(&mut scalar_state[2]),"qs"=>std::mem::take(&mut scalar_state[3]),"qg"=>std::mem::take(&mut scalar_state[4]),
            "t_init"=>vec![0.;n_cells*nz],
            // Native case7 leaves these slots at the framework's cold-start
            // zero (Registry.xml:1088-1103,1286-1312,1398-1400).
            "v"=>vec![0.;n_edges*nz],
            "u_init"|"v_init"|"qv_init"=>vec![0.;nz],
            "dz"|"smcrel"=>vec![0.;n_cells*cfg.n_soil_layers],
            _=>unreachable!("requested diagnostic list is closed"),
        };
        put(&name,values);
    }
    for (i,name) in scalars::OUTPUT.iter().enumerate().skip(5) {
        if i==10 || i==11 {continue;}
        if capsule::requested_fields(&cfg.capsule_path,&[*name])?.len()==1 {
            put(name,std::mem::take(&mut scalar_state[i]));
        }
    }
    if capsule::requested_fields(&cfg.capsule_path,&["canwat"])?.len()==1 {put("canwat",fg.canwat.clone());}
    for field in reconstruction_geometry {
        if capsule::requested_fields(&cfg.capsule_path,&[&field.name])?.len()==1 {
            put(&field.name,field.values.expect("geometry field carries typed values"));
        }
    }
    if let (Some((names,_,_)),Some(state))=(reconstruction_contract,reconstructed) {
        for (name,values) in reconstruction::FIELDS.into_iter().zip(state.fields) {
            if names.iter().any(|s|s==name) {put(name,values);}
        }
    }
    put("precipw", states.iter().map(|s| s.precipw).collect());
    put(
        "surface_pressure",
        states.iter().map(|s| s.surface_pressure).collect(),
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
    let (zs_m, dzs_m) = soil::layer_geometry(cfg.n_soil_layers)?;
    let mut dzs_flat = Vec::with_capacity(n_cells * cfg.n_soil_layers);
    let mut zs_flat = Vec::with_capacity(n_cells * cfg.n_soil_layers);
    for _ in 0..n_cells {
        dzs_flat.extend_from_slice(&dzs_m);
        zs_flat.extend_from_slice(&zs_m);
    }
    put("dzs", dzs_flat);
    put("zs", zs_flat);

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
    // RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_cases.F:300-311
    // at 1d6e370e: preserve the static classification label, defaulting only blanks.
    let labels=netcrust::File::open(&cfg.capsule_path)?.read_strings("mminlu")?;
    let label=labels.first().ok_or_else(||MpasError::Refusal("static mminlu has no label storage".into()))?.trim_end_matches(' ');
    computed.insert("mminlu".to_string(),emit::Computed::Text(
        if label.is_empty() {"USGS".into()} else {label.into()}));

    let seed = format!(
        "{}|{}|{}|{}",
        cfg.met_path.display(),
        cfg.static_path.display(),
        cfg.start_time,
        cfg.capsule_path.display()
    );
    let file_id = emit::mint_file_id(&seed);
    let parent = capsule_file_id(&cfg.capsule_path).unwrap_or_default();
    clock.mark("assemble output fields");
    receipt.emit = emit::write_init(
        &cfg.out_path,
        &cfg.capsule_path,
        computed,
        &file_id,
        &[parent],
        &match cfg.absent_numbers {
            scalars::AbsentNumbers::Zero => cfg.provenance.clone(),
            scalars::AbsentNumbers::FromMass => format!(
                "{}; absent rain/cloud/ice number diagnosed from mass (fork calcnfromq, cases.F:7600-7612 pattern)",
                cfg.provenance),
        },
        &cfg.lineage,
    )?;
    clock.mark("write init file");
    // The column arrays (first guess, column states, edge winds: ~100 million
    // small heap blocks on conus3km) are freed off the main thread; one
    // thread spent seconds freeing them after the file was already written.
    let spent = (fg, states, t_col, rh_col, sh_col, p_col, gfs_z_col, u_edge, winds, metrics, mesh, statics, slabs, scalar_state);
    std::thread::spawn(move || drop(spent));

    receipt.stage_seconds = clock.marks;
    receipt.seconds = started.elapsed().as_secs_f64();
    Ok(receipt)
}

/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_cases.F:4043-7742 at 1d6e370e.
fn capsule_file_id(path: &Path) -> Option<String> {
    let file = netcrust::File::open(path).ok()?;
    file.attribute("file_id")?.as_string().map(str::to_string)
}

/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_cases.F:4043-7742 at 1d6e370e.
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
    /// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_cases.F:4043-7742 at 1d6e370e.
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
    /// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_cases.F:4043-7742 at 1d6e370e.
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
    /// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_cases.F:4043-7742 at 1d6e370e.
    fn only_profile_fields_contribute_first_guess_levels() {
        assert!(contributes_a_level(Target::T));
        assert!(contributes_a_level(Target::U));
        assert!(!contributes_a_level(Target::Psfc));
        assert!(!contributes_a_level(Target::SoilT(0, 10.0)));
        assert!(!contributes_a_level(Target::Skintemp));
    }

    #[test]
    /// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_cases.F:4043-7742 at 1d6e370e.
    fn a_field_the_table_does_not_know_is_ignored_not_guessed() {
        assert!(rule_for("TOTALCLOUD").is_none());
    }

    #[test]
    /// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_cases.F:4043-7742 at 1d6e370e.
    fn the_start_date_parses_into_its_parts() {
        assert_eq!(parse_ymd("2025-03-14_12:00:00").unwrap(), (2025, 3, 14));
        assert!(parse_ymd("nope").is_err());
    }

    #[test]
    /// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_cases.F:4043-7742 at 1d6e370e.
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
    /// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_cases.F:4043-7742 at 1d6e370e.
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
