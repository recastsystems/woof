//! The RRFS v2 UPP field set: every field of NOAA's
//! `rrfs_mpas_postcntrl_L60.xml` (rrfs-workflow 570c81f, which pins UPP
//! 1296eeb), written as GRIB2 with NOAA's parameter identities, levels and
//! product templates, in the three files RRFS v2 writes per forecast hour:
//!
//! * `prslev`: POSTPRS (the 50-1013.2 hPa block and soil) + POSTTWO;
//! * `natlev`: POSTNAT (the 59 native levels and soil) + POSTTWO;
//! * `testbed`: POSTTWO (surface, column and single-level fields).
//!
//! Clean room: UPP is LGPL-3.0 and none of its code is read or ported.
//! The field list and identities come from NOAA's control tables
//! (`profiles/rrfs_mpas_postcntrl_L60.json`, built by
//! `profiles/gen_rrfs_upp_l60.py`); every plane comes from the WOOF post
//! (rw-post Groups A to E, GPU), the RRFS extras kernels
//! (`kernels/rrfs_upp.cu`, GPU) or the frame's own carriers under their
//! MPASSIT names.  A field whose input the frame does not carry is left out
//! and named in the manifest with the carrier it needs; nothing is filled.
//!
//! Input: one WRF-shaped frame on the output grid, the layout MPASSIT
//! writes (`rw_mpas_convert --field-set mpassit` gathers a hex history onto
//! any Lambert window, RRFS v2's 1800 x 1060 3 km grid among them).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use rayon::prelude::*;
use serde::Deserialize;
use serde_json::{json, Value};
use wx_core::grib2::{MessageBuilder, PackingMethod, ProductDefinition, StatisticalInterval};
#[allow(unused_imports)]
use wx_core::grib2::Grib2Writer;

use crate::export::{self, Devices};
use crate::grid;
use crate::request::Request;
use crate::{fail, refuse, Result};

pub const PROFILE_JSON: &str = include_str!("../profiles/rrfs_mpas_postcntrl_L60.json");
pub const PROFILE: &str = "rrfs-mpas-postcntrl-L60";

/// The no-echo value of UPP's composite reflectivity (dBZ), as the UPP 1296eeb referee writes it.
pub const UPP_REFC_FLOOR_DBZ: f32 = -20.0;

/// UPP's no-echo value on a reflectivity plane: the 1296eeb referee writes -20 dBZ where there is no
/// echo on composite, 1 km above ground and -10 C reflectivity alike (rrfs-products integration, mv3km
/// f003/f006: the 1 km field's whole difference was this floor, 15.0 dB on every clear column).
/// Missing points stay missing.
pub fn upp_no_echo_floor(v: Vec<f32>) -> Vec<f32> {
    v.into_iter().map(|x| if x.is_finite() { x.max(UPP_REFC_FLOOR_DBZ) } else { x }).collect()
}

/// One field of the control file.
#[derive(Clone, Debug, Deserialize)]
pub struct Field {
    pub file: String,
    pub datset: String,
    pub shortname: String,
    pub pname: String,
    pub discipline: u8,
    pub category: u8,
    pub number: u8,
    pub ncep_local: bool,
    pub level_type: u8,
    pub levels: Vec<f64>,
    pub level2_type: Option<u8>,
    pub levels2: Vec<f64>,
    pub template: u8,
    pub stat: Option<u8>,
    pub aerosol: Option<String>,
    pub upp_scale: f64,
    /// `last-hour-label`: a run total UPP labels as the last hour's window (its
    /// RAPR/MPAS runoff); absent: the window the field's statistic implies.
    #[serde(default)]
    pub window: Option<String>,
}

#[derive(Deserialize)]
struct ProfileDoc {
    fields: Vec<Field>,
}

pub fn profile() -> Vec<Field> {
    serde_json::from_str::<ProfileDoc>(PROFILE_JSON).expect("profile JSON is checked in").fields
}

/// Number of messages a field writes (one per level, at least one).
pub fn message_count(f: &Field) -> usize {
    f.levels.len().max(1)
}

/// The pressure levels the profile needs (POSTPRS and POSTTWO), Pa.
pub fn isobaric_levels(fields: &[Field]) -> Vec<f64> {
    let mut v: Vec<f64> = fields.iter().filter(|f| f.level_type == 100).flat_map(|f| f.levels.iter().copied()).collect();
    v.sort_by(|a, b| b.partial_cmp(a).unwrap());
    v.dedup();
    v
}

/// Options of one export.
#[derive(Clone, Debug)]
pub struct Options {
    pub out: PathBuf,
    /// The model time step, s: INST_PRATE is the last step's precipitation over it.
    pub dt_seconds: Option<f64>,
    /// The bucket interval of PREC_ACC_* and SNOW_ACC_NC, s (RRFS: 3600).
    pub bucket_seconds: i64,
    pub bits: u8,
    pub post_device: String,
    /// File stem prefix, for example `hexrrfs`.
    pub prefix: String,
}

/// Carried between consecutive frames (bucket fields with no bucket carrier).
#[derive(Default)]
pub struct Carry {
    valid: i64,
    acgraup: Option<Vec<f32>>,
}

// ---------------------------------------------------------------- reading

fn read_var(path: &Path, name: &str) -> Option<Vec<f32>> {
    let file = netcrust::File::open(path).ok()?;
    let var = file.variable(name)?;
    let dtype = var.dtype();
    let v: Vec<f32> = match dtype {
        netcrust::DataType::F32 => match file.read_array::<f32>(name) {
            Ok(a) => a.iter().copied().collect(),
            Err(_) => file.read_array_f64(name).ok()?.values().iter().map(|&x| x as f32).collect(),
        },
        _ => file.read_array_f64(name).ok()?.values().iter().map(|&x| x as f32).collect(),
    };
    let fill = var.attribute("_FillValue").and_then(|a| a.as_f64());
    Some(match fill {
        Some(f) => v.into_iter().map(|x| if f64::from(x) == f { f32::NAN } else { x }).collect(),
        None => v,
    })
}

/// Every carrier the profile can use, read in parallel (one handle per thread).
const CARRIERS: &[&str] = &[
    "QCLOUD", "QICE", "QRAIN", "QSNOW", "QGRAUP", "QNCLOUD", "QNICE", "QNRAIN", "QNGRAUPEL", "QNGRAUP", "QNWFA", "QNIFA",
    "COMPOSITE_REFL_10CM",
    "REFL_10CM", "CLDFRA_BL", "QKE", "EL_PBL", "W", "TSLB", "SMOIS", "HGT", "PSFC", "T2", "Q2", "U10", "V10", "PBLH",
    "RAINNCV", "SNOWNCV", "GRAUPELNCV", "RAINCV", "ISLTYP", "LANDMASK", "XLAND", "LH", "MAVAIL", "CANWAT", "RMOL", "ALBEDO",
    "SWDOWN", "SWDDNI", "SWDDIF", "SWUPT", "OLR", "SFROFF", "UDROFF", "ACSNOW", "ACGRAUP", "ACFRAIN", "SNOWFALLAC",
    "PREC_ACC_NC", "PREC_ACC_C", "SNOW_ACC_NC", "REFC", "REFL_10CM_1KM", "REFD_MAX", "UP_HELI_MAX", "UP_HELI_MIN",
    "UP_HELI_MAX02", "UP_HELI_MIN02", "UP_HELI_MAX03", "UP_HELI_MIN03", "W_UP_MAX", "W_MEAN", "WSPD10MAX", "U10MAX",
    "V10MAX", "HAIL_MAX2D", "HAIL_MAXK1", "SNOW", "SNOWH", "GLW", "LWUPB", "GRDFLX", "LAI", "PRATE_MAX",
    "PREC_MAX_1MIN", "PREC_MAX_5MIN", "PREC_MAX_10MIN", "REFD_MAX_M10C", "REFC_MAX", "VORT_MAX01", "VORT_MAX02",
    "GRPL_MAX", "LTG3_MAX", "SMOKE_FINE", "AOD", "FRP_MEAN", "HWP", "EBB_SMOKE",
];

// ---------------------------------------------------------------- geometry

/// Earth-to-grid rotation of the output grid (Lambert conformal): alpha =
/// n (lon - stand_lon), n the cone constant (Snyder, Map Projections - A
/// Working Manual, USGS PP 1395 (1987) eq. 15-3, tangent and secant cases).
/// Grid components from earth ones: u_g = u cos - v sin, v_g = v cos + u sin.
fn lambert_rotation(meta: &export::FrameMeta) -> Option<(Vec<f32>, Vec<f32>, f64)> {
    let p = &meta.projection;
    if p.map_proj != 1 {
        return None;
    }
    let (t1, t2) = (p.truelat1.to_radians(), p.truelat2.to_radians());
    let n = if (p.truelat1 - p.truelat2).abs() < 1e-9 {
        t1.sin()
    } else {
        (t1.cos() / t2.cos()).ln()
            / ((std::f64::consts::FRAC_PI_4 + t2 / 2.0).tan() / (std::f64::consts::FRAC_PI_4 + t1 / 2.0).tan()).ln()
    };
    let (s, c): (Vec<f32>, Vec<f32>) = meta
        .lon
        .par_iter()
        .map(|&lon| {
            let mut d = lon - p.stand_lon;
            while d > 180.0 {
                d -= 360.0;
            }
            while d < -180.0 {
                d += 360.0;
            }
            let a = n * d.to_radians() * p.truelat1.signum();
            (a.sin() as f32, a.cos() as f32)
        })
        .unzip();
    Some((s, c, n))
}

/// Lambert conformal map factor and the Coriolis parameter per point.
fn map_factor(meta: &export::FrameMeta, n: f64) -> (Vec<f32>, Vec<f32>) {
    let p = &meta.projection;
    let t1 = p.truelat1.to_radians();
    let tq = |phi: f64| (std::f64::consts::FRAC_PI_4 - phi / 2.0).tan();
    meta.lat
        .par_iter()
        .map(|&lat| {
            let phi = lat.to_radians();
            let m = (t1.cos() / phi.cos()) * (tq(phi) / tq(t1)).powf(n);
            (m as f32, (2.0 * 7.292e-5 * phi.sin()) as f32)
        })
        .unzip()
}

// ---------------------------------------------------------------- planes

struct Ctx<'a> {
    n: usize,
    nx: usize,
    ny: usize,
    nz: usize,
    surface: &'a BTreeMap<&'static str, Vec<f32>>,
    plev_index: BTreeMap<i64, usize>,
    plev: &'a [Vec<Vec<f32>>],
    iso: &'a [f32],
    iso_levels: &'a [f64],
    absv: &'a [f32],
    extras: &'a [f32],
    vars: &'a BTreeMap<&'static str, Vec<f32>>,
    state_mass: &'a [f32],
    omega_native: &'a [f32],
    rot: Option<&'a (Vec<f32>, Vec<f32>, f64)>,
    earth_frame: bool,
    lead: i64,
    dt: Option<f64>,
    bucket_graupel: Option<Vec<f32>>,
}

/// Isobaric extras, in `iso` field order.
const ISO_FIELDS: [&str; 10] = ["SPFH", "VVEL", "CLMR", "ICMR", "RWMR", "SNMR", "GRLE", "PMTF", "PMTC", "DZDT"];

type Got = std::result::Result<Vec<f32>, String>;

impl<'a> Ctx<'a> {
    fn s(&self, id: &str) -> Got {
        self.surface.get(id).cloned().ok_or_else(|| format!("WOOF post plane {id} was not computed for this frame"))
    }
    fn v(&self, name: &str) -> Got {
        self.vars.get(name).filter(|v| v.len() >= self.n).map(|v| v[..self.n].to_vec()).ok_or_else(|| format!("the frame carries no {name}"))
    }
    fn v3(&self, name: &str, k: usize) -> Got {
        let v = self.vars.get(name).ok_or_else(|| format!("the frame carries no {name}"))?;
        if v.len() < (k + 1) * self.n {
            return Err(format!("{name} has no level {}", k + 1));
        }
        Ok(v[k * self.n..(k + 1) * self.n].to_vec())
    }
    fn mass(&self, slot: rw_post::state::MassState, k: usize) -> Vec<f32> {
        let vol = self.nz * self.n;
        self.state_mass[slot as usize * vol + k * self.n..slot as usize * vol + (k + 1) * self.n].to_vec()
    }
    fn x(&self, name: &str) -> Got {
        let i = crate::rrfs_gpu::OUT_NAMES.iter().position(|o| *o == name).expect("extras plane name");
        Ok(self.extras[i * self.n..(i + 1) * self.n].to_vec())
    }
    fn plev(&self, pa: f64, f: usize) -> Got {
        let li = *self.plev_index.get(&(pa.round() as i64)).ok_or("pressure level not computed")?;
        Ok(self.plev[li][f].clone())
    }
    fn iso(&self, name: &str, pa: f64) -> Got {
        let fi = ISO_FIELDS.iter().position(|f| *f == name).unwrap();
        let li = self.iso_levels.iter().position(|l| (*l - pa).abs() < 0.5).ok_or("pressure level not computed")?;
        let nl = self.iso_levels.len();
        Ok(self.iso[(fi * nl + li) * self.n..(fi * nl + li + 1) * self.n].to_vec())
    }
    /// The grid-relative `want_u` component of a wind pair in the frame's own orientation.
    fn wind(&self, u: Got, v: Got, want_u: bool) -> Got {
        let (u, v) = (u?, v?);
        match (self.earth_frame, self.rot) {
            (true, Some((s, c, _))) => Ok((0..self.n)
                .into_par_iter()
                .map(|i| if want_u { u[i] * c[i] - v[i] * s[i] } else { v[i] * c[i] + u[i] * s[i] })
                .collect()),
            (true, None) => Err("earth-relative winds on a grid with no Lambert rotation".into()),
            (false, _) => Ok(if want_u { u } else { v }),
        }
    }
    fn map2(&self, a: Got, f: impl Fn(f32) -> f32 + Sync + Send) -> Got {
        Ok(a?.into_par_iter().map(f).collect())
    }
    fn zip2(&self, a: Got, b: Got, f: impl Fn(f32, f32) -> f32 + Sync + Send) -> Got {
        let (a, b) = (a?, b?);
        Ok(a.par_iter().zip(b.par_iter()).map(|(&x, &y)| f(x, y)).collect())
    }

    /// The plane of field `f` at level index `li`.
    fn plane(&self, f: &Field, li: usize) -> Got {
        use rw_post::state::MassState as M;
        let lev = f.levels.get(li).copied().unwrap_or(0.0);
        let k = || (lev.round() as usize).saturating_sub(1);
        let name = f.shortname.as_str();
        let need44 = |what: &str| Err(format!("needs {what}, an RRFSx/MPAS-Model 44c0849 output (54064b34) the hex 1d6e370e route does not write yet"));
        let smoke = || Err("smoke and dust are off in the hrrrv5 run (RRFS-SD, gap 10): the frame carries no smoke carrier".to_string());
        match (f.datset.as_str(), name) {
            // ---------------- isobaric
            (_, "TMP_ON_ISOBARIC_SFC") => self.plev(lev, 1),
            (_, "HGT_ON_ISOBARIC_SFC") => self.plev(lev, 0),
            (_, "DPT_ON_ISOBARIC_SFC") => self.plev(lev, 2),
            (_, "RH_ON_ISOBARIC_SFC") => self.plev(lev, 3),
            (_, "UGRD_ON_ISOBARIC_SFC") => self.wind(self.plev(lev, 4), self.plev(lev, 5), true),
            (_, "VGRD_ON_ISOBARIC_SFC") => self.wind(self.plev(lev, 4), self.plev(lev, 5), false),
            (_, "SPFH_ON_ISOBARIC_SFC") => self.iso("SPFH", lev),
            (_, "VVEL_ON_ISOBARIC_SFC") => self.iso("VVEL", lev),
            (_, "DZDT_ON_ISOBARIC_SFC") => self.iso("DZDT", lev),
            (_, "CLMR_ON_ISOBARIC_SFC") => self.iso("CLMR", lev),
            (_, "ICMR_ON_ISOBARIC_SFC") => self.iso("ICMR", lev),
            (_, "RWMR_ON_ISOBARIC_SFC") => self.iso("RWMR", lev),
            (_, "SNMR_ON_ISOBARIC_SFC") => self.iso("SNMR", lev),
            (_, "GRLE_ON_ISOBARIC_SFC") => self.iso("GRLE", lev),
            (_, "NCWFA_ON_ISOBARIC_SFC") => self.iso("PMTF", lev),
            (_, "NCIFA_ON_ISOBARIC_SFC") => self.iso("PMTC", lev),
            (_, "ABSV_ON_ISOBARIC_SFC") => {
                let li = self.iso_levels.iter().position(|l| (*l - lev).abs() < 0.5).ok_or("pressure level not computed")?;
                if self.absv.is_empty() {
                    return Err("absolute vorticity needs a Lambert output grid".into());
                }
                Ok(self.absv[li * self.n..(li + 1) * self.n].to_vec())
            }
            (_, "LFTX_ON_ISOBARIC_SFC_500-1000hpa") => self.x("lftx"),
            (_, "MAX_MAXUVV_ON_ISOBARIC_SFC_100-1000hpa") => self.v("W_UP_MAX"),
            // ---------------- native levels
            (_, "PRES_ON_HYBRID_LVL") => Ok(self.mass(M::PFull, k())),
            (_, "HGT_ON_HYBRID_LVL") => Ok(self.mass(M::ZMass, k())),
            (_, "TMP_ON_HYBRID_LVL") => Ok(self.mass(M::Tk, k())),
            (_, "SPFH_ON_HYBRID_LVL") => Ok(self.mass(M::Q, k())),
            (_, "UGRD_ON_HYBRID_LVL") => self.wind(Ok(self.mass(M::U, k())), Ok(self.mass(M::V, k())), true),
            (_, "VGRD_ON_HYBRID_LVL") => self.wind(Ok(self.mass(M::U, k())), Ok(self.mass(M::V, k())), false),
            (_, "VVEL_ON_HYBRID_LVL") => Ok(self.omega_native[k() * self.n..(k() + 1) * self.n].to_vec()),
            (_, "CLMR_ON_HYBRID_LVL") => self.v3("QCLOUD", k()),
            (_, "ICMR_ON_HYBRID_LVL") => self.v3("QICE", k()),
            (_, "RWMR_ON_HYBRID_LVL") => self.v3("QRAIN", k()),
            (_, "SNMR_ON_HYBRID_LVL") => self.v3("QSNOW", k()),
            (_, "GRLE_ON_HYBRID_LVL") => self.v3("QGRAUP", k()),
            (_, "NCCD_ON_HYBRID_LVL") => self.v3("QNCLOUD", k()),
            (_, "NCIP_ON_HYBRID_LVL") => self.v3("QNICE", k()),
            (_, "NCRAIN_ON_HYBRID_LVL") => self.v3("QNRAIN", k()),
            (_, "NCGRAUP_ON_HYBRID_LVL") => self.v3("QNGRAUPEL", k()).or_else(|_| self.v3("QNGRAUP", k())),
            (_, "NCWFA_ON_HYBRID_LVL") => self.v3("QNWFA", k()),
            (_, "NCIFA_ON_HYBRID_LVL") => self.v3("QNIFA", k()),
            (_, "REFD_ON_HYBRID_LVL") => self.v3("REFL_10CM", k()),
            (_, "TKE_ON_HYBRID_LVL") => self.map2(self.v3("QKE", k()), |q| 0.5 * q),
            (_, "MIXL_ON_HYBRID_LVL") => self.v3("EL_PBL", k()),
            (_, "FRACCC_ON_HYBRID_LVL") => self.v3("CLDFRA_BL", k()),
            (_, "SMOKE_ON_HYBRID_LVL") => smoke(),
            // ---------------- soil (RUC, 9 levels: 0 1 4 10 30 60 100 160 300 cm)
            (_, "TSOIL_ON_DEPTH_BEL_LAND_SFC") => self.v3("TSLB", li),
            (_, "SOILW_ON_DEPTH_BEL_LAND_SFC") => self.v3("SMOIS", li),
            (_, "RUC_MSTAV_ON_DEPTH_BEL_LAND_SFC") => self.map2(self.v("MAVAIL"), |m| 100.0 * m),
            // ---------------- 2 m, 10 m, 80 m, surface
            (_, "TMP_ON_SPEC_HGT_LVL_ABOVE_GRND_2m") => self.s("t2"),
            (_, "DPT_ON_SPEC_HGT_LVL_ABOVE_GRND_2m") => self.s("td2"),
            (_, "RH_ON_SPEC_HGT_LVL_ABOVE_GRND_2m") => self.s("rh2"),
            (_, "GSD_POT_ON_SPEC_HGT_LVL_ABOVE_GRND_2m") => self.s("potential_temperature2"),
            (_, "SPFH_ON_SPEC_HGT_LVL_ABOVE_GRND_2m") => self.s("specific_humidity2"),
            (_, "UGRD_ON_SPEC_HGT_LVL_ABOVE_GRND_10m") => self.wind(self.v("U10"), self.v("V10"), true),
            (_, "VGRD_ON_SPEC_HGT_LVL_ABOVE_GRND_10m") => self.wind(self.v("U10"), self.v("V10"), false),
            (_, "UGRD_ON_SPEC_HGT_LVL_ABOVE_GRND") => self.wind(self.s("u80"), self.s("v80"), true),
            (_, "VGRD_ON_SPEC_HGT_LVL_ABOVE_GRND") => self.wind(self.s("u80"), self.s("v80"), false),
            (_, "PRES_ON_SURFACE") => self.s("surface_pressure"),
            (_, "HGT_ON_SURFACE") => self.s("terrain"),
            (_, "TMP_ON_SURFACE") => self.s("skin_temperature"),
            (_, "WEASD_ON_SURFACE") => self.s("snow_water"),
            (_, "SNOD_ON_SURFACE") => self.s("snow_depth"),
            (_, "SNOWC_ON_SURFACE") => self.s("snow_cover"),
            (_, "SDEN_ON_SURFACE") => self.zip2(self.v("SNOW"), self.v("SNOWH"), |w, d| if d > 0.0 { w / d } else { f32::NAN }),
            (_, "VEG_ON_SURFACE") => self.s("vegetation_fraction"),
            (_, "VEG_MIN_ON_SURFACE") => self.s("vegetation_min"),
            (_, "VEG_MAX_ON_SURFACE") => self.s("vegetation_max"),
            (_, "LAI_ON_SURFACE") => self.s("leaf_area_index"),
            (_, "SOTYP_ON_SURFACE") => self.v("ISLTYP"),
            (_, "VGTYP_ON_SURFACE") => self.s("vegetation_type"),
            (_, "ICEC_ON_SURFACE") => self.s("sea_ice_fraction"),
            (_, "LAND_ON_SURFACE") => self.v("LANDMASK").or_else(|_| self.map2(self.v("XLAND"), |x| if x.is_finite() { if x < 1.5 { 1.0 } else { 0.0 } } else { f32::NAN })),
            (_, "SFCR_ON_SURFACE") => self.s("roughness_length"),
            (_, "FRICV_ON_SURFACE") => self.s("friction_velocity"),
            (_, "OBUKHOV_RECIPROCAL_ON_SURFACE") => self.v("RMOL"),
            (_, "CNWAT_ON_SURFACE") => self.v("CANWAT"),
            (_, "HPBL_ON_SURFACE") => self.v("PBLH"),
            (_, "INST_GFLUX_ON_SURFACE") => self.s("ground_heat_flux"),
            (_, "INST_SHTFL_ON_SURFACE") => self.s("sensible_heat_flux"),
            (_, "INST_LHTFL_ON_SURFACE") => self.s("latent_heat_flux").and_then(|p| if p.iter().any(|x| x.is_finite()) { Ok(p) } else { Err(String::new()) }).or_else(|_| self.v("LH")),
            (_, "INST_DSWRF_ON_SURFACE") => self.s("downward_shortwave"),
            (_, "INST_DLWRF_ON_SURFACE") => self.s("downward_longwave"),
            (_, "INST_USWRF_ON_SURFACE") => self.zip2(self.v("SWDOWN"), self.v("ALBEDO"), |s, a| s * a),
            (_, "INST_ULWRF_ON_SURFACE") => self.v("LWUPB").map_err(|_| "needs LWUPB (upward longwave at the surface), which the hrrrv5 diag stream does not write".into()),
            (_, "INST_SWDDNI_ON_SURFACE") => self.v("SWDDNI"),
            (_, "INST_SWDDIF_ON_SURFACE") => self.v("SWDDIF"),
            (_, "INST_ULWRF_ON_TOP_OF_ATMOS") => self.v("OLR"),
            (_, "INST_USWRF_ON_TOP_OF_ATMOS") => self.v("SWUPT"),
            (_, "MAPS_PRMSL_ON_MEAN_SEA_LVL") => self.s("maps_mslp"),
            (_, "GUST_ON_SURFACE") => self.s("gust"),
            // ---------------- convective
            (_, "CAPE_ON_SURFACE") => self.s("sbcape"),
            (_, "CIN_ON_SURFACE") => self.s("sbcin"),
            (_, "BEST_CAPE_ON_SPEC_PRES_ABOVE_GRND") => self.s("cape_best180"),
            (_, "BEST_CIN_ON_SPEC_PRES_ABOVE_GRND") => self.s("cin_best180"),
            (_, "MIXED_LAYER_CAPE_ON_SPEC_PRES_ABOVE_GRND") => self.s("mlcape"),
            (_, "MIXED_LAYER_CIN_ON_SPEC_PRES_ABOVE_GRND") => self.s("mlcin"),
            (_, "UNSTABLE_CAPE_ON_SPEC_PRES_ABOVE_GRND") => self.x("mucape255"),
            (_, "UNSTABLE_CIN_ON_SPEC_PRES_ABOVE_GRND") => self.x("mucin255"),
            (_, "PLPL_ON_SPEC_PRES_ABOVE_GRND") => self.x("plpl255"),
            (_, "CAPE_ON_0_3KM_ABOVE_GRND") => self.x("cape_0_3km"),
            (_, "4LFTX_ON_SPEC_PRES_ABOVE_GRND") => self.x("best_lftx"),
            (_, "HGT_ON_LFC") => self.x("lfc_height"),
            (_, "HGT_ON_EQUIL_LVL") => self.x("el_height"),
            (_, "HGT_ON_LVL_OF_ADIAB_COND_FROM_SFC") => self.zip2(self.s("lcl_height"), self.s("terrain"), |h, z| h + z),
            (_, "PWAT_ON_ENTIRE_ATMOS_SINGLE_LYR") => self.s("pwat"),
            (_, "HLCY_ON_SPEC_HGT_LVL_ABOVE_GRND") => {
                if lev >= 2999.0 { self.s("srh_0_3km") } else { self.s("srh_0_1km") }
            }
            (_, "EFF_HLCY_ON_SPEC_HGT_LVL_ABOVE_GRND") => self.x("effective_srh"),
            (_, "DENDRITIC_LAYER_DEPTH") => self.x("dendritic_depth"),
            (_, "ENHANCED_STRETCHING_POTENTIAL") => self.x("esp"),
            (_, "CRITICAL_ANGLE") => self.x("critical_angle"),
            (_, "USTM_ON_SPEC_HGT_LVL_ABOVE_GRND") => self.wind(self.s("storm_motion_u"), self.s("storm_motion_v"), true),
            (_, "VSTM_ON_SPEC_HGT_LVL_ABOVE_GRND") => self.wind(self.s("storm_motion_u"), self.s("storm_motion_v"), false),
            (_, "VUCSH_ON_SPEC_HGT_LVL_ABOVE_GRND_0-1km") => self.wind(self.s("shear_u_0_1km"), self.s("shear_v_0_1km"), true),
            (_, "VVCSH_ON_SPEC_HGT_LVL_ABOVE_GRND_0-1km") => self.wind(self.s("shear_u_0_1km"), self.s("shear_v_0_1km"), false),
            (_, "VUCSH_ON_SPEC_HGT_LVL_ABOVE_GRND_0-6km") => self.wind(self.s("shear_u_0_6km"), self.s("shear_v_0_6km"), true),
            (_, "VVCSH_ON_SPEC_HGT_LVL_ABOVE_GRND_0-6km") => self.wind(self.s("shear_u_0_6km"), self.s("shear_v_0_6km"), false),
            (_, "UGRD_ON_PLANETARY_BOUND_LYR") => self.wind(self.x("u_pbl"), self.x("v_pbl"), true),
            (_, "VGRD_ON_PLANETARY_BOUND_LYR") => self.wind(self.x("u_pbl"), self.x("v_pbl"), false),
            (_, "VRATE_ON_PLANETARY_BOUND_LYR") => self.x("ventilation_rate"),
            // ---------------- reflectivity, cloud, visibility
            // UPP's composite floor is -20 dBZ (the 1296eeb referee writes -20 where there is no echo;
            // the column maximum here bottoms out at -35), so the no-echo value matches the referee.
            (_, "REFC_ON_ENTIRE_ATMOS") => self.v("REFC").or_else(|_| self.v("COMPOSITE_REFL_10CM"))
                .or_else(|_| self.s("composite_reflectivity"))
                .map(|v| v.into_iter().map(|x| if x.is_finite() { x.max(UPP_REFC_FLOOR_DBZ) } else { x }).collect()),
            // 1 km and -10 C reflectivity carry the same -20 dBZ no-echo value in the referee. 4 km keeps
            // its own: the referee leaves it 0 everywhere on MPASSIT input (no REFL_10CM_4KM carrier).
            (_, "GSD_REFL_ON_SPEC_HGT_LVL_ABOVE_GRND_1km") => self.v("REFL_10CM_1KM").or_else(|_| self.x("refl_1km")).map(upp_no_echo_floor),
            (_, "GSD_REFL_ON_SPEC_HGT_LVL_ABOVE_GRND_4km") => self.x("refl_4km"),
            (_, "REFD_ON_ISOTHERMAL") => self.x("refl_minus10c").map(upp_no_echo_floor),
            (_, "GSD_ECHOTOP_ON_CLOUD_TOP") => self.x("echo_top"),
            (_, "VIL_ON_ENTIRE_ATMOS") => self.x("vil"),
            (_, "GSD_VIS_ON_SURFACE") => self.s("visibility"),
            (_, "GSD_HGT_ON_CLOUD_CEILING") => self.s("cloud_ceiling"),
            (_, "GSD_EXP_CEILING") => self.zip2(self.s("cloud_ceiling"), self.s("terrain"), |c, z| c - z),
            (_, "GSD_EXP_CEILING_2") => self.zip2(self.s("cloud_base"), self.s("terrain"), |c, z| c - z),
            (_, "HGT_ON_CLOUD_BASE") => self.s("cloud_base"),
            (_, "HGT_ON_CLOUD_TOP") => self.s("cloud_top"),
            (_, "PRES_ON_CLOUD_BASE") => self.x("pres_cloud_base"),
            (_, "PRES_ON_CLOUD_TOP") => self.x("pres_cloud_top"),
            (_, "LCDC_ON_LOW_CLOUD_LYR") => self.s("low_cloud"),
            (_, "MCDC_ON_MID_CLOUD_LYR") => self.s("mid_cloud"),
            (_, "HCDC_ON_HIGH_CLOUD_LYR") => self.s("high_cloud"),
            (_, "INST_TCDC_ON_ENTIRE_ATMOS") => self.x("tcdc_total"),
            (_, "TCDC_ON_BOUND_LYR") => self.x("tcdc_pbl"),
            (_, "TCOLW_ON_ENTIRE_ATMOS") => self.x("tcolw"),
            (_, "TCOLI_ON_ENTIRE_ATMOS") => self.x("tcoli"),
            (_, "TCOLP_ON_ENTIRE_ATMOS") => self.x("tcolp"),
            (_, "GSD_RH_WRT_PRECIP_WATER_ON_ENTIRE_ATMOS") => self.x("rh_wrt_pw"),
            // ---------------- freezing levels
            (_, "HGT_ON_0C_ISOTHERM") => self.s("freezing_height"),
            (_, "GSD_PRES_ON_0C_ISOTHERM") => self.s("freezing_pressure"),
            (_, "RH_ON_0C_ISOTHERM") => self.x("rh_freezing"),
            (_, "HGT_ON_HGHST_TROP_FRZ_LVL") => self.s("highest_freezing_height"),
            (_, "GSD_PRES_ON_HGHST_TROP_FRZ_LVL") => self.s("highest_freezing_pressure"),
            (_, "RH_ON_HGHST_TROP_FRZ_LVL") => self.x("rh_highest_freezing"),
            (_, "HGT_ON_HGHST_TROP_-10C_LVL") => self.s("minus10_height"),
            (_, "HGT_ON_HGHST_TROP_-20C_LVL") => self.s("minus20_height"),
            // ---------------- precipitation type and rate
            (_, "GSD_INST_CRAIN_ON_SURFACE") => self.x("crain"),
            (_, "GSD_CSNOW_ON_SURFACE") => self.x("csnow"),
            (_, "GSD_CICEP_ON_SURFACE") => self.x("cicep"),
            (_, "GSD_CFRZR_ON_SURFACE") => self.x("cfrzr"),
            (_, "CPOFP_ON_SURFACE") => self.x("cpofp"),
            (_, "INST_PRATE_ON_SURFACE") => {
                let dt = self.dt.ok_or("needs the model time step (--dt): RAINNCV is the last step's precipitation")?;
                let nc = self.v("RAINNCV")?;
                let c = self.v("RAINCV").ok();
                Ok((0..self.n).into_par_iter().map(|i| (nc[i] + c.as_ref().map_or(0.0, |c| c[i])) / dt as f32).collect())
            }
            (_, "MAX_PRATE_ON_SURFACE") => self.v("PRATE_MAX").or_else(|_| need44("prate_max (1 h maximum precipitation rate)")),
            (_, "MAX_PRATE_1MIN_ON_SURFACE") => self.v("PREC_MAX_1MIN").or_else(|_| need44("the 1 min maximum precipitation rate")),
            (_, "MAX_PRATE_5MIN_ON_SURFACE") => self.v("PREC_MAX_5MIN").or_else(|_| need44("the 5 min maximum precipitation rate")),
            (_, "MAX_PRATE_10MIN_ON_SURFACE") => self.v("PREC_MAX_10MIN").or_else(|_| need44("the 10 min maximum precipitation rate")),
            // ---------------- accumulations (template 4.8)
            (_, "ACM_APCP_ON_SURFACE") => self.s("total_precipitation"),
            (_, "ACM_WEASD_ON_SURFACE") => self.v("ACSNOW"),
            (_, "ACM_GRAUPEL_ON_SURFACE") => self.v("ACGRAUP"),
            (_, "ACM_FRAIN_ON_SURFACE") => self.v("ACFRAIN"),
            (_, "GSD_ACM_SNOD_ON_SURFACE") => self.map2(self.v("SNOWFALLAC"), |mm| mm / 1000.0),
            (_, "ACM_SSRUN_ON_SURFACE") => self.v("SFROFF"),
            (_, "ACM_BGRUN_ON_SURFACE") => self.v("UDROFF").map_err(|_| "needs UDROFF (subsurface runoff), which the hrrrv5 history does not write".into()),
            (_, "BUCKET_APCP_ON_SURFACE") => match self.v("PREC_ACC_C") {
                Ok(c) => self.zip2(self.v("PREC_ACC_NC"), Ok(c), |a, b| a + b),
                Err(_) => self.v("PREC_ACC_NC"),
            },
            (_, "BUCKET_WEASD_ON_SURFACE") => self.v("SNOW_ACC_NC"),
            (_, "BUCKET_GRAUPEL_ON_SURFACE") => self.bucket_graupel.clone().ok_or_else(|| "needs the previous hour's ACGRAUP (process frames in order)".into()),
            // ---------------- window extremes (template 4.8)
            (_, "MAX_WIND_ON_SPEC_HGT_LVL_ABOVE_GRND_10m") => self.v("WSPD10MAX"),
            (_, "MAX_UGD_ON_SPEC_HGT_LVL_ABOVE_GRND_10m") => self.wind(self.v("U10MAX"), self.v("V10MAX"), true).map_err(|_| "needs U10MAX/V10MAX (hourly maximum 10 m wind components), which the hrrrv5 diag stream does not write".into()),
            (_, "MAX_VGD_ON_SPEC_HGT_LVL_ABOVE_GRND_10m") => self.wind(self.v("U10MAX"), self.v("V10MAX"), false).map_err(|_| "needs U10MAX/V10MAX, which the hrrrv5 diag stream does not write".into()),
            (_, "MAX_UPHL_ON_SPEC_HGT_LVL_ABOVE_GRND_2-5km") => self.v("UP_HELI_MAX"),
            (_, "GSD_MIN_UPHL_ON_SPEC_HGT_LVL_ABOVE_GRND_2-5km") => self.v("UP_HELI_MIN").map_err(|_| "needs UP_HELI_MIN (2-5 km minimum updraft helicity): the hrrrv5 diag stream writes updraft_helicity_min only for 0-1, 0-2, 0-3 and 1-6 km".into()),
            (_, "GSD_MAX_UPHL_ON_SPEC_HGT_LVL_ABOVE_GRND_0-2km") => self.v("UP_HELI_MAX02"),
            (_, "GSD_MIN_UPHL_ON_SPEC_HGT_LVL_ABOVE_GRND_0-2km") => self.v("UP_HELI_MIN02"),
            (_, "GSD_MAX_UPHL_ON_SPEC_HGT_LVL_ABOVE_GRND_0-3km") => self.v("UP_HELI_MAX03"),
            (_, "GSD_MIN_UPHL_ON_SPEC_HGT_LVL_ABOVE_GRND_0-3km") => self.v("UP_HELI_MIN03"),
            (_, "GSD_MAX_REL_VORT_ON_SPEC_HGT_LVL_ABOVE_GRND_0-2km") => self.v("VORT_MAX02").map_err(|_| "needs an hourly 0-2 km maximum relative vorticity carrier, which no MPAS diag stream writes".into()),
            (_, "GSD_MAX_REL_VORT_ON_SPEC_HGT_LVL_ABOVE_GRND_0-1km") => self.v("VORT_MAX01").map_err(|_| "needs an hourly 0-1 km maximum relative vorticity carrier, which no MPAS diag stream writes".into()),
            (_, "MAX_GRAUPEL_HAIL_DIAMETER_IN_COLUMN") => self.v("HAIL_MAX2D"),
            (_, "MAX_GRAUPEL_HAIL_DIAMETER_AT_LOWEST_LVL") => self.v("HAIL_MAXK1"),
            (_, "AVE_DZDT_ON_SIGMA_LVL_0.5-0.8") => self.v("W_MEAN"),
            (_, "MAXVIG_ON_ENTIRE_ATMOS_SINGLE_LYR") => self.v("GRPL_MAX").map_err(|_| "needs an hourly maximum vertically integrated graupel carrier, which the hrrrv5 diag stream does not write".into()),
            (_, "MAX_REF_ON_SPEC_HGT_LVL_ABOVE_GRND_1km") => self.v("REFD_MAX"),
            (_, "MAX_REF_ON_ISOTHERMAL_-10C") => self.v("REFD_MAX_M10C").map_err(|_| "needs an hourly maximum -10 C reflectivity carrier, which the hrrrv5 diag stream does not write".into()),
            (_, "MAX_REFC_ON_ENTIRE_ATMOS") => self.v("REFC_MAX").map_err(|_| "needs an hourly maximum composite reflectivity carrier; refl10cm_max is the instantaneous column maximum (written as REFC)".into()),
            (_, "GSD_MAX_LTG_THREAT3_ON_ENTIRE_ATMOS") => self.v("LTG3_MAX").map_err(|_| "needs the McCaul lightning threat carrier (LTG3_MAX), which the hrrrv5 diag stream does not write".into()),
            // ---------------- smoke, dust, fire (off in hrrrv5)
            (_, "SMOKE_ON_SPEC_HGT_LVL_ABOVE_GRND_8m") | (_, "SMOKE_ON_ENTIRE_ATMOS_SINGLE_LYR") | (_, "BIOMASS_BURNING_EMISSIONS")
            | (_, "AOD_ON_ENTIRE_ATMOS_SINGLE_LYR") | (_, "AOD_ON_SURFACE") | (_, "MEAN_FIRE_RDIATV_PWR") | (_, "HWP_ON_SURFACE") => smoke(),
            _ => Err(format!("{name}: no source mapped")),
        }
    }
}

// ---------------------------------------------------------------- writing

fn builder(f: &Field, li: usize, plane: &[f32], grid_def: &wx_core::grib2::GridDefinition, meta: &export::FrameMeta, opts: &Options, bucket: i64) -> Result<MessageBuilder> {
    let reference = export::utc(meta.start)?;
    let valid = export::utc(meta.valid)?;
    let lead_s = meta.valid - meta.start;
    let (unit, lead_v) = export::lead(lead_s)?;
    let lev = f.levels.get(li).copied().unwrap_or(0.0);
    let values: Vec<f64> = plane.iter().map(|&x| if x.is_finite() { f64::from(x) } else { f64::NAN }).collect();
    // Statistical window: accumulations from the start (ACM) or over the
    // bucket (BUCKET_*), extremes and averages over the last output interval.
    let stat = f.stat.map(|process| {
        let last_hour_label = f.window.as_deref() == Some("last-hour-label");
        let window = if f.shortname.starts_with("BUCKET_") || f.stat != Some(1) || last_hour_label {
            bucket.min(lead_s)
        } else {
            lead_s
        };
        let (wu, wl) = export::lead(window).unwrap_or((1, 0));
        let (ou, ov) = export::lead(lead_s - window).unwrap_or((1, 0));
        let (u, off, len) = if wu == ou { (wu, ov, wl) } else { (13, (lead_s - window) as u32, window as u32) };
        (process, u, off, len)
    });
    let product = ProductDefinition {
        template: if stat.is_some() { 8 } else { 0 },
        parameter_category: f.category,
        parameter_number: f.number,
        generating_process: export::GENERATING_PROCESS,
        forecast_time: stat.map_or(lead_v, |s| s.2),
        time_range_unit: stat.map_or(unit, |s| s.1),
        level_type: f.level_type,
        level_value: lev,
    };
    let is_wind = matches!(f.pname.as_str(), "UGRD" | "VGRD" | "USTM" | "VSTM" | "VUCSH" | "VVCSH" | "MAXUW" | "MAXVW");
    let mut m = MessageBuilder::new(f.discipline, values)
        .center(export::CENTRE, 0)
        .generating_process_identifier(export::GENERATING_PROCESS_ID)
        .master_table_version(export::MASTER_TABLES)
        .local_table_version(export::LOCAL_TABLES)
        .reference_time(reference)
        .grid(grid_def.clone())
        .product(product)
        .packing(PackingMethod::ComplexSpatial { bits_per_value: opts.bits, order: 2 })
        // The grid section NCEP's own rapid-refresh GRIB2 carries (HRRR
        // wrfsfc, decoded with eccodes): earth shape 6 and resolution flags 8
        // on every message, so hex files sit on "the HRRR grid" in NOAA's tools.
        // Vector components are grid-relative (bit 0x08): earth-relative inputs
        // are rotated in `Ctx::wind`.
        .earth_shape(6)
        .resolution_flags(0x08);
    let _ = is_wind;
    if let Some(t2) = f.level2_type {
        let v2 = if f.levels2.len() == f.levels.len() { f.levels2[li] } else { f.levels2.first().copied().unwrap_or(0.0) };
        m = m.second_surface(t2, v2);
    }
    if let Some((process, u, _, len)) = stat {
        m = m.statistical_interval(StatisticalInterval { end_time: valid, statistical_process: process, time_unit: u, length: len });
    }
    let _ = &f.aerosol;
    Ok(m)
}

/// Export one frame; returns its manifest record.
pub fn export_frame(path: &Path, opts: &Options, devices: &mut Devices, gpu: &crate::rrfs_gpu::RrfsGpu, carry: &mut Carry) -> Result<Value> {
    let t0 = Instant::now();
    let mut timings: BTreeMap<&str, f64> = BTreeMap::new();
    let fields = profile();
    let meta = export::frame_meta(path, None)?;
    let n = meta.nx * meta.ny;
    let levels = isobaric_levels(&fields);
    let mut req: Request = serde_json::from_value(json!({
        "schema": crate::request::SCHEMA, "inputs": [path], "out": opts.out, "fields": "surface",
        "winds": "grid", "post_device": opts.post_device,
    }))
    .map_err(|e| fail(e.to_string()))?;
    req.levels_pa_override = Some(levels.clone());
    req.skip_vertical_extras = true;
    let t = Instant::now();
    let mut planes = export::compute(path, &req, devices, &meta)?;
    timings.insert("woof_post_groups_a_to_e", t.elapsed().as_secs_f64());
    let state = planes.state.take().ok_or_else(|| fail("the post kept no column state"))?;
    let nz = state.shape.nz;

    // ---- carriers, read in parallel (one netCDF handle per thread)
    let t = Instant::now();
    let vars: BTreeMap<&'static str, Vec<f32>> = CARRIERS.par_iter().filter_map(|&name| read_var(path, name).map(|v| (name, v))).collect();
    timings.insert("read_carriers", t.elapsed().as_secs_f64());

    // ---- frame orientation: an MPAS frame carries earth-relative winds
    // (SINALPHA = 0, COSALPHA = 1); a WRF frame carries grid-relative ones.
    let earth_frame = match (read_var(path, "SINALPHA"), read_var(path, "COSALPHA")) {
        (Some(s), Some(c)) => s.iter().all(|x| x.abs() < 1e-6) && c.iter().all(|x| (x - 1.0).abs() < 1e-6),
        _ => true,
    };
    let rot = lambert_rotation(&meta);

    // ---- GPU extras: the column kernel
    use rw_post::state::{InterfaceState, MassState as M};
    let t = Instant::now();
    let vol = nz * n;
    let mass = |slot: M| &state.mass[slot as usize * vol..(slot as usize + 1) * vol];
    let nan3 = vec![f32::NAN; vol];
    let carrier3 = |name: &str| vars.get(name).filter(|v| v.len() == vol).map(Vec::as_slice).unwrap_or(&nan3);
    // Packed on every core (one slot per task), not by a serial memcpy.
    let pack = |parts: Vec<&[f32]>| -> Vec<f32> {
        let mut out = vec![0.0f32; parts.len() * vol];
        out.par_chunks_mut(vol).zip(parts.par_iter()).for_each(|(d, s)| d.copy_from_slice(s));
        out
    };
    let v3 = pack(vec![
        mass(M::PFull), mass(M::Tk), mass(M::R), mass(M::ZMass), mass(M::U), mass(M::V),
        carrier3("QCLOUD"), carrier3("QICE"), carrier3("QRAIN"), carrier3("QSNOW"), carrier3("QGRAUP"),
        carrier3("REFL_10CM"), carrier3("CLDFRA_BL"),
    ]);
    let ivol = (nz + 1) * n;
    let vi = [
        &state.interface[InterfaceState::PInt as usize * ivol..(InterfaceState::PInt as usize + 1) * ivol],
        &state.interface[InterfaceState::ZInt as usize * ivol..(InterfaceState::ZInt as usize + 1) * ivol],
    ]
    .concat();
    let nan2 = vec![f32::NAN; n];
    let sp = |id: &str| planes.surface.get(id).map(Vec::as_slice).unwrap_or(&nan2);
    let cv = |name: &str| vars.get(name).filter(|v| v.len() == n).map(Vec::as_slice).unwrap_or(&nan2);
    let s2 = [
        sp("terrain"), cv("PSFC"), cv("T2"), cv("Q2"), sp("shelter_pressure"), cv("U10"), cv("V10"), cv("PBLH"),
        sp("storm_motion_u"), sp("storm_motion_v"), sp("mlcape"), sp("freezing_height"), sp("highest_freezing_height"),
        sp("cloud_base"), sp("cloud_top"), sp("low_cloud"), sp("mid_cloud"), sp("high_cloud"), cv("RAINNCV"),
        cv("SNOWNCV"), cv("GRAUPELNCV"),
    ]
    .concat();
    let extras = gpu.column(nz, n, &v3, &vi, &s2).map_err(fail)?;
    drop(v3);
    timings.insert("gpu_extras_column", t.elapsed().as_secs_f64());

    // ---- GPU extras: isobaric interpolation of the extra 3-D fields
    let t = Instant::now();
    let w = vars.get("W").filter(|w| w.len() == ivol);
    let w_mass: Vec<f32> = match w {
        Some(w) => (0..vol).into_par_iter().map(|i| 0.5 * (w[i] + w[i + n])).collect(),
        None => nan3.clone(),
    };
    let (p_full, tv) = (mass(M::PFull), mass(M::Tv));
    let omega_native: Vec<f32> = (0..vol)
        .into_par_iter()
        .map(|i| -(p_full[i] / (rw_post::consts::RD * tv[i])) * rw_post::consts::G * w_mass[i])
        .collect();
    let f3 = pack(vec![
        mass(M::Q), &omega_native, carrier3("QCLOUD"), carrier3("QICE"), carrier3("QRAIN"), carrier3("QSNOW"),
        carrier3("QGRAUP"), carrier3("QNWFA"), carrier3("QNIFA"), &w_mass,
    ]);
    let iso = gpu.isobaric(nz, n, p_full, &f3, ISO_FIELDS.len(), &levels).map_err(fail)?;
    drop(f3);
    timings.insert("gpu_extras_isobaric", t.elapsed().as_secs_f64());

    // ---- GPU extras: absolute vorticity on the isobaric levels (grid-relative winds)
    let t = Instant::now();
    let plev_index: BTreeMap<i64, usize> = levels.iter().enumerate().map(|(i, p)| (p.round() as i64, i)).collect();
    let absv = match &rot {
        Some((s, c, cone)) => {
            let (mapf, fcor) = map_factor(&meta, *cone);
            let nl = levels.len();
            let mut ug = vec![0.0f32; nl * n];
            let mut vg = vec![0.0f32; nl * n];
            ug.par_chunks_mut(n).zip(vg.par_chunks_mut(n)).enumerate().for_each(|(l, (ub, vb))| {
                let (u, v) = (&planes.plev[l][4], &planes.plev[l][5]);
                for i in 0..n {
                    if earth_frame {
                        ub[i] = u[i] * c[i] - v[i] * s[i];
                        vb[i] = v[i] * c[i] + u[i] * s[i];
                    } else {
                        ub[i] = u[i];
                        vb[i] = v[i];
                    }
                }
            });
            gpu.absv(meta.nx, meta.ny, nl, meta.projection.dx, &ug, &vg, &mapf, &fcor).map_err(fail)?
        }
        None => Vec::new(),
    };
    timings.insert("gpu_extras_absv", t.elapsed().as_secs_f64());

    // ---- bucket graupel from the previous frame
    let acg = vars.get("ACGRAUP").filter(|v| v.len() == n).cloned();
    let bucket_graupel = match (&carry.acgraup, &acg) {
        (Some(prev), Some(now)) if meta.valid - carry.valid == opts.bucket_seconds => {
            Some(now.par_iter().zip(prev.par_iter()).map(|(a, b)| a - b).collect())
        }
        _ => None,
    };

    let ctx = Ctx {
        n,
        nx: meta.nx,
        ny: meta.ny,
        nz,
        surface: &planes.surface,
        plev_index,
        plev: &planes.plev,
        iso: &iso,
        iso_levels: &levels,
        absv: &absv,
        extras: &extras,
        vars: &vars,
        state_mass: &state.mass,
        omega_native: &omega_native,
        rot: rot.as_ref(),
        earth_frame,
        lead: meta.valid - meta.start,
        dt: opts.dt_seconds,
        bucket_graupel,
    };
    let _ = (ctx.nx, ctx.ny, ctx.lead);

    // ---- plane -> message -> bytes for every (field, level), all cores.
    // Each job keeps only its packed message: holding every plane (and its
    // f64 copy for the packer) of a 1800 x 1060 frame at once took a 64 GB host past
    // 60 GB and the kernel killed the export.
    let t = Instant::now();
    let lead_s = meta.valid - meta.start;
    let grid_def = grid::definition(&meta.projection, meta.nx, meta.ny, &meta.lat, &meta.lon)?;
    let jobs: Vec<(usize, usize)> = fields.iter().enumerate().flat_map(|(fi, f)| (0..message_count(f)).map(move |li| (fi, li))).collect();
    let packed: Vec<std::result::Result<Vec<u8>, String>> = jobs
        .par_iter()
        .map(|&(fi, li)| {
            let f = &fields[fi];
            if f.stat.is_some() && lead_s == 0 {
                return Err("zero-length statistical window at the initial time".to_string());
            }
            let p = ctx.plane(f, li)?;
            if p.iter().all(|x| !x.is_finite()) {
                return Err("missing in every grid point".into());
            }
            let m = builder(f, li, &p, &grid_def, &meta, opts, opts.bucket_seconds).map_err(|e| format!("FATAL {e}"))?;
            drop(p);
            wx_core::grib2::Grib2Writer::new().add_message(m).to_bytes().map_err(|e| format!("FATAL GRIB2 encoding: {e}"))
        })
        .collect();
    timings.insert("planes_and_encode", t.elapsed().as_secs_f64());
    let mut bytes: BTreeMap<&str, Vec<u8>> = BTreeMap::new();
    let mut written: BTreeMap<&str, Vec<Value>> = BTreeMap::new();
    let mut omitted: Vec<Value> = Vec::new();
    for (&(fi, li), g) in jobs.iter().zip(packed.into_iter()) {
        let f = &fields[fi];
        let lev = f.levels.get(li).copied();
        match g {
            Ok(b) => {
                bytes.entry(f.datset.as_str()).or_default().extend_from_slice(&b);
                written.entry(f.datset.as_str()).or_default().push(json!({"shortname": f.shortname, "level": lev}));
            }
            Err(why) if why.starts_with("FATAL") => return Err(fail(why)),
            Err(why) => omitted.push(json!({"shortname": f.shortname, "datset": f.datset, "level": lev, "reason": why})),
        }
    }
    let empty = Vec::new();
    let two = bytes.get("POSTTWO").unwrap_or(&empty);
    let reference = chrono::DateTime::from_timestamp(meta.start, 0).ok_or_else(|| refuse("bad start time"))?;
    let stem = format!("{}.t{}z", opts.prefix, reference.format("%H"));
    let fhr = format!("f{:03}", lead_s / 3600);
    let mut files = Vec::new();
    std::fs::create_dir_all(&opts.out)?;
    for (kind, set) in [("prslev", "POSTPRS"), ("natlev", "POSTNAT"), ("testbed", "POSTTWO")] {
        let mut b = if set == "POSTTWO" { Vec::new() } else { bytes.get(set).cloned().unwrap_or_default() };
        b.extend_from_slice(two);
        let dest = opts.out.join(format!("{stem}.{kind}.{fhr}.grib2"));
        export::write_atomic(&dest, &b)?;
        let count = written.get(set).map_or(0, Vec::len) + if set == "POSTTWO" { 0 } else { written.get("POSTTWO").map_or(0, Vec::len) };
        files.push(json!({"path": dest.file_name().unwrap().to_string_lossy(), "kind": kind, "bytes": b.len(), "messages": count}));
    }
    // carry for the next frame
    carry.valid = meta.valid;
    carry.acgraup = acg;

    let field_status: Vec<Value> = fields
        .iter()
        .map(|f| {
            let total = message_count(f);
            let miss: Vec<&Value> = omitted.iter().filter(|o| o["shortname"] == f.shortname.as_str() && o["datset"] == f.datset.as_str()).collect();
            json!({"datset": f.datset, "shortname": f.shortname, "pname": f.pname,
                   "grib": format!("{}.{}.{}", f.discipline, f.category, f.number), "level_type": f.level_type,
                   "template": if f.stat.is_some() { 8 } else { 0 }, "messages": total, "written": total - miss.len(),
                   "reason": miss.first().map(|o| o["reason"].clone())})
        })
        .collect();
    Ok(json!({
        "source": path.display().to_string(),
        "profile": PROFILE,
        "valid": crate::export::utc(meta.valid)?.format("%Y-%m-%dT%H:%M:%SZ").to_string(),
        "reference_time": crate::export::utc(meta.start)?.format("%Y-%m-%dT%H:%M:%SZ").to_string(),
        "files": files,
        "fields": field_status,
        "devices": planes.devices,
        "extras_device": gpu.label,
        "earth_relative_input_winds": earth_frame,
        "winds_written": "grid-relative (resolution flags 0x38) where a Lambert rotation exists",
        "vertical_source": planes.vertical_source,
        "pressure_source": planes.pressure_source,
        "woof_post_timings": planes.timings,
        "timings": timings,
        "seconds": t0.elapsed().as_secs_f64(),
    }))
}

#[cfg(test)]
mod referee_identity_tests {
    use super::*;

    /// What the UPP 1296eeb referee wrote on MPASSIT output of an hrrrv5 history
    /// (rrfs-products integration, 2026-10-29), where the earlier profile differed.
    #[test]
    fn identities_follow_the_upp_1296eeb_referee() {
        let fields = profile();
        let named = |name: &str| fields.iter().find(|f| f.shortname == name).unwrap_or_else(|| panic!("{name}"));
        for name in ["REFC_ON_ENTIRE_ATMOS", "VIL_ON_ENTIRE_ATMOS", "INST_TCDC_ON_ENTIRE_ATMOS",
                     "TCOLW_ON_ENTIRE_ATMOS", "TCOLI_ON_ENTIRE_ATMOS", "PWAT_ON_ENTIRE_ATMOS_SINGLE_LYR"] {
            assert_eq!(named(name).level_type, 200, "{name}: the referee writes entire atmosphere (single layer)");
        }
        for f in fields.iter().filter(|f| f.level_type == 100) {
            assert!(!f.levels.contains(&101320.0), "{}: UPP's default pressure set stops at 100000 Pa", f.shortname);
        }
        assert_eq!(named("TMP_ON_ISOBARIC_SFC").levels.len() + named("HGT_ON_ISOBARIC_SFC").levels.len() > 0, true);
        for name in ["ACM_SSRUN_ON_SURFACE", "ACM_BGRUN_ON_SURFACE"] {
            assert_eq!(named(name).window.as_deref(), Some("last-hour-label"), "{name}");
        }
        assert_eq!(named("ACM_APCP_ON_SURFACE").window, None, "precipitation keeps its run-total window");
    }

    #[test]
    fn reflectivity_no_echo_is_the_referee_minus_20_dbz() {
        let v = upp_no_echo_floor(vec![-35.0, -20.5, -20.0, -3.25, 47.5, f32::NAN]);
        assert_eq!(&v[..5], &[-20.0, -20.0, -20.0, -3.25, 47.5]);
        assert!(v[5].is_nan(), "missing stays missing");
    }
}
