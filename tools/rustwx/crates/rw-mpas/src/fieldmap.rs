//! The field map: every WRF variable a catalogue product actually reads,
//! paired with the MPAS history (or pinned init) variable carrying the same
//! quantity.
//!
//! What this does not do is invent a field the MPAS history does not carry. A
//! WRF variable with no MPAS source is simply absent, its products report
//! missing fields, and the absence is recorded in the emitted file's
//! `MPAS_ABSENT_WRF_FIELDS` attribute. A missing field is an output-stream gap
//! to fix upstream, never something to synthesize here.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rank {
    Surface,
    Mass3d,
    W3d,
    U3d,
    V3d,
    Soil,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldSource {
    History,
    Init,
    /// A value the WRF schema requires that MPAS has no equivalent for, and
    /// whose correct value is a constant, not an interpolation.
    Constant,
}

#[derive(Debug, Clone, Copy)]
pub struct FieldMapping {
    pub wrf_name: &'static str,
    pub mpas_name: Option<&'static str>,
    pub rank: Rank,
    pub units: &'static str,
    pub description: &'static str,
    pub source: FieldSource,
    pub note: &'static str,
}

const fn f(
    wrf_name: &'static str,
    mpas_name: Option<&'static str>,
    rank: Rank,
    units: &'static str,
    description: &'static str,
    source: FieldSource,
    note: &'static str,
) -> FieldMapping {
    FieldMapping {
        wrf_name,
        mpas_name,
        rank,
        units,
        description,
        source,
        note,
    }
}

/// MPAS-A v8.4.2 Registry.xml:1891-1896,2754-2756,3038-3040,
/// 3183-3185,3810-3812: native stream names for existing rendered quantities.
/// Legacy hex/WRF-derived history names remain the fallback; nothing is derived.
pub fn native_history_name(wrf_name: &str) -> Option<&'static str> {
    match wrf_name {
        "U" => Some("uReconstructZonal"),
        "V" => Some("uReconstructMeridional"),
        "T2" => Some("t2m"),
        "TSK" => Some("skintemp"),
        "PBLH" => Some("hpbl"),
        "SWDOWN" => Some("swdnb"),
        _ => None,
    }
}

pub static FIELD_MAP: &[FieldMapping] = &[
    // --- three-dimensional mass fields --------------------------------------
    f("T", Some("theta"), Rank::Mass3d, "K",
      "perturbation potential temperature theta-t0", FieldSource::History,
      "theta - 300 K; MPAS theta is the dry potential temperature"),
    f("PB", Some("pressure"), Rank::Mass3d, "Pa", "BASE STATE PRESSURE", FieldSource::History,
      "MPAS has no base/perturbation split; PB carries the full model pressure so P+PB is exactly it"),
    f("P", None, Rank::Mass3d, "Pa", "perturbation pressure", FieldSource::Constant,
      "identically zero; see PB"),
    f("QVAPOR", Some("qv"), Rank::Mass3d, "kg kg-1", "Water vapor mixing ratio", FieldSource::History, ""),
    f("QCLOUD", Some("qc"), Rank::Mass3d, "kg kg-1", "Cloud water mixing ratio", FieldSource::History, ""),
    f("QRAIN", Some("qr"), Rank::Mass3d, "kg kg-1", "Rain water mixing ratio", FieldSource::History, ""),
    f("QICE", Some("qi"), Rank::Mass3d, "kg kg-1", "Ice mixing ratio", FieldSource::History, ""),
    f("QSNOW", Some("qs"), Rank::Mass3d, "kg kg-1", "Snow mixing ratio", FieldSource::History, ""),
    f("QGRAUP", Some("qg"), Rank::Mass3d, "kg kg-1", "Graupel mixing ratio", FieldSource::History, ""),
    f("QNCLOUD", Some("nc"), Rank::Mass3d, "kg-1", "Cloud droplet number mixing ratio", FieldSource::History, "published moments only; never synthesized"),
    f("QNICE", Some("ni"), Rank::Mass3d, "kg-1", "Cloud ice number mixing ratio", FieldSource::History, ""),
    f("QNRAIN", Some("nr"), Rank::Mass3d, "kg-1", "Rain number mixing ratio", FieldSource::History, ""),
    f("QNSNOW", Some("ns"), Rank::Mass3d, "kg-1", "Snow number mixing ratio", FieldSource::History, ""),
    f("QNGRAUPEL", Some("ng"), Rank::Mass3d, "kg-1", "Graupel number mixing ratio", FieldSource::History, ""),
    f("QNHAIL", Some("nh"), Rank::Mass3d, "kg-1", "Hail number mixing ratio", FieldSource::History, ""),
    f("REFL_10CM", Some("refl10cm"), Rank::Mass3d, "dBZ", "Radar reflectivity (lambda = 10 cm)",
      FieldSource::History,
      "the model's own microphysics-time diagnostic; when the history carries it, reflectivity products read the model rather than the renderer's hydrometeor fallback"),
    f("U", Some("u_zonal"), Rank::U3d, "m s-1", "x-wind component", FieldSource::History,
      "earth-relative zonal wind, restaggered in x"),
    f("V", Some("v_meridional"), Rank::V3d, "m s-1", "y-wind component", FieldSource::History,
      "earth-relative meridional wind, restaggered in y"),
    f("W", Some("w"), Rank::W3d, "m s-1", "z-wind component", FieldSource::History, ""),
    f("PHB", Some("zgrid"), Rank::W3d, "m2 s-2", "base-state geopotential", FieldSource::Init,
      "g * zgrid from the pinned init file; MPAS is height based"),
    f("PH", None, Rank::W3d, "m2 s-2", "perturbation geopotential", FieldSource::Constant,
      "identically zero; see PHB"),
    // --- surface diagnostics ------------------------------------------------
    f("T2", Some("t2"), Rank::Surface, "K", "TEMP at 2 M", FieldSource::History, ""),
    f("Q2", Some("q2"), Rank::Surface, "kg kg-1", "QV at 2 M", FieldSource::History,
      "published bitwise by the history writer; native q2 carries occasional small negatives, so dewpoint consumers clamp at their own boundary"),
    f("PSFC", Some("surface_pressure"), Rank::Surface, "Pa", "SFC PRESSURE", FieldSource::History, ""),
    f("U10", Some("u10"), Rank::Surface, "m s-1", "U at 10 M", FieldSource::History, ""),
    f("V10", Some("v10"), Rank::Surface, "m s-1", "V at 10 M", FieldSource::History, ""),
    f("TSK", Some("tsk"), Rank::Surface, "K", "SURFACE SKIN TEMPERATURE", FieldSource::History, ""),
    f("HGT", Some("ter"), Rank::Surface, "m", "Terrain Height", FieldSource::History, ""),
    f("PBLH", Some("pblh"), Rank::Surface, "m", "PBL HEIGHT", FieldSource::History, ""),
    f("HFX", Some("hfx"), Rank::Surface, "W m-2", "UPWARD HEAT FLUX AT THE SURFACE", FieldSource::History, ""),
    f("LH", Some("lh"), Rank::Surface, "W m-2", "LATENT HEAT FLUX AT THE SURFACE", FieldSource::History, ""),
    f("QFX", Some("qfx"), Rank::Surface, "kg m-2 s-1", "UPWARD MOISTURE FLUX AT THE SURFACE", FieldSource::History, ""),
    f("SWDOWN", Some("swdown"), Rank::Surface, "W m-2", "DOWNWARD SHORT WAVE FLUX AT GROUND SURFACE", FieldSource::History, ""),
    f("GLW", Some("glw"), Rank::Surface, "W m-2", "DOWNWARD LONG WAVE FLUX AT GROUND SURFACE", FieldSource::History, ""),
    f("REFC", Some("refl10cm_max"), Rank::Surface, "dBZ", "COMPOSITE REFLECTIVITY (COLUMN MAXIMUM, LAMBDA = 10 CM)",
      FieldSource::History,
      "the model's own column-maximum reflectivity; MPAS-A writes it to the diagnostics stream (diag.*.nc), not to output, so a run that writes no 3-D refl10cm still delivers its radar picture"),
    f("CUPREC", Some("cuprec"), Rank::Surface, "mm s-1", "CONVECTIVE PRECIPITATION RATE", FieldSource::History,
      "the cumulus scheme's rain rate at the output time (MPAS-A output stream; absent when no cumulus scheme runs); REFC adds its reflectivity (convective_reflectivity)"),
    // --- precipitation accumulations ----------------------------------------
    f("RAINC", Some("rainc"), Rank::Surface, "mm", "ACCUMULATED TOTAL CUMULUS PRECIPITATION", FieldSource::History, ""),
    f("RAINNC", Some("rainnc"), Rank::Surface, "mm", "ACCUMULATED TOTAL GRID SCALE PRECIPITATION", FieldSource::History, ""),
    f("SNOWNC", Some("snownc"), Rank::Surface, "mm", "ACCUMULATED TOTAL GRID SCALE SNOW AND ICE", FieldSource::History, ""),
    f("GRAUPELNC", Some("graupelnc"), Rank::Surface, "mm", "ACCUMULATED TOTAL GRID SCALE GRAUPEL", FieldSource::History, ""),
    // --- land surface --------------------------------------------------------
    f("TSLB", Some("tslb"), Rank::Soil, "K", "SOIL TEMPERATURE", FieldSource::History, ""),
    f("SMOIS", Some("smois"), Rank::Soil, "m3 m-3", "SOIL MOISTURE", FieldSource::History, ""),
    f("LANDMASK", Some("landmask"), Rank::Surface, "1", "LAND MASK (1 FOR LAND, 0 FOR WATER)", FieldSource::Init,
      "units 1, not WRF's empty string: a CF reader (the engine's mapped NetCDF route) pins units, and an empty unit is no unit"),
    f("LU_INDEX", Some("ivgtyp"), Rank::Surface, "", "LAND USE CATEGORY", FieldSource::Init, ""),
    // Sea ice and snow ride the history (hex's ArWen backend publishes the
    // land surface's own xice, snow and snowh) so a hex frame can be a
    // regional model's initial and boundary surface, not only a picture.
    f("SEAICE", Some("xice"), Rank::Surface, "1", "SEA ICE FLAG", FieldSource::History,
      "fractional sea-ice cover, MPAS xice"),
    f("SNOW", Some("snow"), Rank::Surface, "kg m-2", "SNOW WATER EQUIVALENT", FieldSource::History, ""),
    f("SNOWH", Some("snowh"), Rank::Surface, "m", "PHYSICAL SNOW DEPTH", FieldSource::History, ""),
    // --- grid geometry the renderer reads ------------------------------------
    f("SINALPHA", None, Rank::Surface, "", "Local sine of map rotation", FieldSource::Constant,
      "0: the emitted U/V/U10/V10 are already earth-relative, so the renderer's grid-to-earth rotation must be the identity"),
    f("COSALPHA", None, Rank::Surface, "", "Local cosine of map rotation", FieldSource::Constant,
      "1; see SINALPHA"),
    // --- the MPASSIT field set (`--field-set mpassit`): every remaining variable
    // rrfs-workflow 570c81f fix/mpassit histlist_2d, histlist_3d and diaglist name,
    // under the WRF name MPASSIT writes it as, so a gathered RRFS frame carries what
    // UPP reads.  Only the mpassit set emits these rows ([`MPASSIT_ONLY`]).
    f("XLAND", Some("xland"), Rank::Surface, "1", "LAND MASK (1 FOR LAND, 2 FOR WATER)", FieldSource::History, "mpassit"),
    f("SNOWFALLAC", Some("snowfallac"), Rank::Surface, "mm", "RUN TOTAL SNOWFALL DEPTH", FieldSource::History, "mpassit"),
    f("SST", Some("sst"), Rank::Surface, "K", "SEA SURFACE TEMPERATURE", FieldSource::History, "mpassit"),
    f("RMOL", Some("rmol"), Rank::Surface, "m-1", "1/MONIN-OBUKHOV LENGTH", FieldSource::History, "mpassit"),
    f("BR", Some("br"), Rank::Surface, "1", "BULK RICHARDSON NUMBER IN SURFACE LAYER", FieldSource::History, "mpassit"),
    f("ALBEDO", Some("sfc_albedo"), Rank::Surface, "1", "SURFACE ALBEDO", FieldSource::History, "mpassit"),
    f("ZNT", Some("znt"), Rank::Surface, "m", "TIME-VARYING ROUGHNESS LENGTH", FieldSource::History, "mpassit"),
    f("UST", Some("ust"), Rank::Surface, "m s-1", "U* IN SIMILARITY THEORY", FieldSource::History, "mpassit"),
    f("ISLTYP", Some("isltyp"), Rank::Surface, "", "DOMINANT SOIL CATEGORY", FieldSource::History, "mpassit"),
    f("IVGTYP", Some("ivgtyp"), Rank::Surface, "", "DOMINANT VEGETATION CATEGORY", FieldSource::History, "mpassit"),
    f("VEGFRA", Some("vegfra"), Rank::Surface, "%", "VEGETATION FRACTION", FieldSource::History, "mpassit"),
    f("SHDMIN", Some("shdmin"), Rank::Surface, "%", "ANNUAL MIN VEG FRACTION", FieldSource::History, "mpassit"),
    f("SHDMAX", Some("shdmax"), Rank::Surface, "%", "ANNUAL MAX VEG FRACTION", FieldSource::History, "mpassit"),
    f("CANWAT", Some("canwat"), Rank::Surface, "kg m-2", "CANOPY WATER", FieldSource::History, "mpassit"),
    f("SNOWC", Some("snowc"), Rank::Surface, "1", "FLAG INDICATING SNOW COVERAGE (1 FOR SNOW COVER)", FieldSource::History, "mpassit"),
    f("ACSNOW", Some("snownc"), Rank::Surface, "kg m-2", "ACCUMULATED SNOW", FieldSource::History, "mpassit"),
    f("ACGRAUP", Some("graupelnc"), Rank::Surface, "kg m-2", "ACCUMULATED GRAUPEL", FieldSource::History, "mpassit"),
    f("SFROFF", Some("sfcrunoff"), Rank::Surface, "mm", "SURFACE RUNOFF", FieldSource::History, "mpassit"),
    f("MAVAIL", Some("mavail"), Rank::Surface, "1", "SURFACE MOISTURE AVAILABILITY", FieldSource::History, "mpassit"),
    f("ACFRAIN", Some("frainnc"), Rank::Surface, "mm", "ACCUMULATED FREEZING RAIN", FieldSource::History, "mpassit"),
    f("SNOWNCV", Some("snowncv"), Rank::Surface, "mm", "TIME-STEP NONCONVECTIVE SNOW AND ICE", FieldSource::History, "mpassit"),
    f("RAINNCV", Some("rainncv"), Rank::Surface, "mm", "TIME-STEP NONCONVECTIVE PRECIPITATION", FieldSource::History, "mpassit"),
    f("GRAUPELNCV", Some("graupelncv"), Rank::Surface, "mm", "TIME-STEP NONCONVECTIVE GRAUPEL", FieldSource::History, "mpassit"),
    f("PREC_ACC_C", Some("prec_acc_c"), Rank::Surface, "mm", "ACCUMULATED CUMULUS PRECIPITATION OVER THE BUCKET INTERVAL", FieldSource::History, "mpassit"),
    f("PREC_ACC_NC", Some("prec_acc_nc"), Rank::Surface, "mm", "ACCUMULATED GRID SCALE PRECIPITATION OVER THE BUCKET INTERVAL", FieldSource::History, "mpassit"),
    f("SNOW_ACC_NC", Some("snow_acc_nc"), Rank::Surface, "mm", "ACCUMULATED SNOW WATER EQUIVALENT OVER THE BUCKET INTERVAL", FieldSource::History, "mpassit"),
    f("REFL_10CM_1KM", Some("refl10cm_1km"), Rank::Surface, "dBZ", "1 KM AGL REFLECTIVITY (LAMBDA = 10 CM)", FieldSource::History, "mpassit"),
    f("REFD_MAX", Some("refl10cm_1km_max"), Rank::Surface, "dBZ", "MAX 1 KM AGL REFLECTIVITY OVER THE OUTPUT INTERVAL", FieldSource::History, "mpassit"),
    f("TH2", Some("th2m"), Rank::Surface, "K", "POT TEMP at 2 M", FieldSource::History, "mpassit"),
    f("UP_HELI_MAX", Some("updraft_helicity_max"), Rank::Surface, "m2 s-2", "MAX UPDRAFT HELICITY 2-5 KM", FieldSource::History, "mpassit"),
    f("W_UP_MAX", Some("w_velocity_max"), Rank::Surface, "m s-1", "MAX UPDRAFT VELOCITY", FieldSource::History, "mpassit"),
    f("W_MIN", Some("w_velocity_min"), Rank::Surface, "m s-1", "MAX DOWNDRAFT VELOCITY", FieldSource::History, "mpassit"),
    f("W_MEAN", Some("w_velocity_mean"), Rank::Surface, "m s-1", "MEAN VERTICAL VELOCITY", FieldSource::History, "mpassit"),
    f("UP_HELI_MAX01", Some("updraft_helicity_max_01"), Rank::Surface, "m2 s-2", "MAX UPDRAFT HELICITY 0-1 KM", FieldSource::History, "mpassit"),
    f("UP_HELI_MAX16", Some("updraft_helicity_max_16"), Rank::Surface, "m2 s-2", "MAX UPDRAFT HELICITY 1-6 KM", FieldSource::History, "mpassit"),
    f("UP_HELI_MAX0_15", Some("updraft_helicity_max0_15"), Rank::Surface, "m2 s-2", "MAX UPDRAFT HELICITY", FieldSource::History, "mpassit"),
    f("UP_HELI_MAX05_15", Some("updraft_helicity_max05_15"), Rank::Surface, "m2 s-2", "MAX UPDRAFT HELICITY", FieldSource::History, "mpassit"),
    f("UP_HELI_MAX02", Some("updraft_helicity_max_02"), Rank::Surface, "m2 s-2", "MAX UPDRAFT HELICITY 0-2 KM", FieldSource::History, "mpassit"),
    f("UP_HELI_MAX03", Some("updraft_helicity_max_03"), Rank::Surface, "m2 s-2", "MAX UPDRAFT HELICITY 0-3 KM", FieldSource::History, "mpassit"),
    f("UP_HELI_MIN01", Some("updraft_helicity_min_01"), Rank::Surface, "m2 s-2", "MIN UPDRAFT HELICITY 0-1 KM", FieldSource::History, "mpassit"),
    f("UP_HELI_MIN16", Some("updraft_helicity_min_16"), Rank::Surface, "m2 s-2", "MIN UPDRAFT HELICITY 1-6 KM", FieldSource::History, "mpassit"),
    f("UP_HELI_MIN0_15", Some("updraft_helicity_min0_15"), Rank::Surface, "m2 s-2", "MIN UPDRAFT HELICITY", FieldSource::History, "mpassit"),
    f("UP_HELI_MIN05_15", Some("updraft_helicity_min05_15"), Rank::Surface, "m2 s-2", "MIN UPDRAFT HELICITY", FieldSource::History, "mpassit"),
    f("UP_HELI_MIN02", Some("updraft_helicity_min_02"), Rank::Surface, "m2 s-2", "MIN UPDRAFT HELICITY 0-2 KM", FieldSource::History, "mpassit"),
    f("UP_HELI_MIN03", Some("updraft_helicity_min_03"), Rank::Surface, "m2 s-2", "MIN UPDRAFT HELICITY 0-3 KM", FieldSource::History, "mpassit"),
    f("WIND02MAX", Some("wind02_max"), Rank::Surface, "m s-1", "MAX 0-2 KM WIND SPEED", FieldSource::History, "mpassit"),
    f("SPD02INTMAX", Some("wind02_integral_max"), Rank::Surface, "m2 s-1", "MAX 0-2 KM INTEGRATED WIND SPEED", FieldSource::History, "mpassit"),
    f("WSPD10MAX", Some("wind_speed_10m_max"), Rank::Surface, "m s-1", "MAX 10 M WIND SPEED", FieldSource::History, "mpassit"),
    f("SPDUV10MEAN", Some("wind_speed_10m_mean"), Rank::Surface, "m s-1", "MEAN 10 M WIND SPEED", FieldSource::History, "mpassit"),
    f("U10MEAN", Some("u10m_mean"), Rank::Surface, "m s-1", "MEAN U AT 10 M", FieldSource::History, "mpassit"),
    f("V10MEAN", Some("v10m_mean"), Rank::Surface, "m s-1", "MEAN V AT 10 M", FieldSource::History, "mpassit"),
    f("SWDDIR", Some("swddir"), Rank::Surface, "W m-2", "SHORTWAVE SURFACE DOWNWARD DIRECT IRRADIANCE", FieldSource::History, "mpassit"),
    f("SWDDNI", Some("swddni"), Rank::Surface, "W m-2", "SHORTWAVE SURFACE DOWNWARD DIRECT NORMAL IRRADIANCE", FieldSource::History, "mpassit"),
    f("SWDDIF", Some("swddif"), Rank::Surface, "W m-2", "SHORTWAVE SURFACE DOWNWARD DIFFUSE IRRADIANCE", FieldSource::History, "mpassit"),
    f("SWUPT", Some("swupt"), Rank::Surface, "W m-2", "INSTANTANEOUS UPWELLING SHORTWAVE FLUX AT TOP", FieldSource::History, "mpassit"),
    f("OLR", Some("olrtoa"), Rank::Surface, "W m-2", "TOA OUTGOING LONG WAVE", FieldSource::History, "mpassit"),
    f("HAIL_MAXK1", Some("max_hail_diameter_sfc_acc"), Rank::Surface, "m", "MAX HAIL DIAMETER AT THE LOWEST LEVEL", FieldSource::History, "mpassit"),
    f("HAIL_MAX2D", Some("max_hail_diameter_column_acc"), Rank::Surface, "m", "MAX HAIL DIAMETER IN THE COLUMN", FieldSource::History, "mpassit"),
    f("QVGRAUPEL", Some("volg"), Rank::Mass3d, "m3 kg-1", "Graupel particle volume mixing ratio", FieldSource::History, "mpassit"),
    f("QNWFA", Some("nwfa"), Rank::Mass3d, "kg-1", "water-friendly aerosol number concentration", FieldSource::History, "mpassit"),
    f("QNIFA", Some("nifa"), Rank::Mass3d, "kg-1", "ice-friendly aerosol number concentration", FieldSource::History, "mpassit"),
    f("CLDFRA_BL", Some("cldfrac_bl"), Rank::Mass3d, "1", "boundary-layer cloud fraction", FieldSource::History, "mpassit"),
    f("QKE", Some("qke"), Rank::Mass3d, "m2 s-2", "twice the turbulent kinetic energy", FieldSource::History, "mpassit"),
    f("EL_PBL", Some("el_pbl"), Rank::Mass3d, "m", "mixing length", FieldSource::History, "mpassit"),
];

/// WRF variables some catalogue product wants that the port's history stream
/// does not carry. Gaps to file against the output stream, not things to
/// synthesize.
pub static ABSENT_WRF_FIELDS: &[(&str, &str, &str)] = &[
    ("UP_HELI_MAX", "maximum updraught helicity",
     "not in the port history stream; blocks composite_reflectivity_uh"),
    ("WSPD10MAX", "maximum 10 m wind speed",
     "not in the port history stream; the gust product has no store selector in any WRF lane anyway"),
    ("OLR", "top-of-atmosphere outgoing longwave",
     "not in the port history stream; blocks the OLR raw-extra plane"),
    ("SST", "sea surface temperature",
     "not in the port history stream: hex carries no ocean model, so the water skin temperature TSK is the SST the run was initialised with, and a consumer takes SST from TSK over water by a named policy (the engine's hex-global profile does)"),
    ("RAINSH", "shallow-cumulus precipitation accumulation",
     "the pinned suite runs Grell-Freitas with no separate shallow bucket; total precipitation is RAINC + RAINNC and is complete"),
];

/// How a product conversion carries each field onto the render grid
/// (`crate::remap`).  Metadata, not code paths: a new history field takes
/// the class of what it physically is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RemapClass {
    /// Continuous state: barycentric on the mesh's Delaunay triangles.
    Linear,
    /// Accumulations: first-order area-conservative.
    Conservative,
    /// Categories and land-only arrays: the cell's own value.
    Nearest,
}

impl RemapClass {
    pub fn label(self) -> &'static str {
        match self {
            RemapClass::Linear => "linear",
            RemapClass::Conservative => "conservative",
            RemapClass::Nearest => "nearest",
        }
    }
}

/// Accumulated water: a pixel must carry the area mean of what fell on it,
/// and the window's integral must be the mesh's.
pub static CONSERVATIVE_FIELDS: &[&str] = &[
    "RAINC", "RAINNC", "SNOWNC", "GRAUPELNC", "ACSNOW", "ACGRAUP", "SFROFF", "ACFRAIN", "PREC_ACC_C",
    "PREC_ACC_NC", "SNOW_ACC_NC", "SNOWFALLAC", "RAINNCV", "SNOWNCV", "GRAUPELNCV",
];

/// Categories have no average; soil arrays carry placeholders over water
/// that a blend would smear into the coast.
pub static NEAREST_FIELDS: &[&str] = &["LANDMASK", "LU_INDEX", "TSLB", "SMOIS", "XLAND", "ISLTYP", "IVGTYP"];

/// The note that marks a row only the `mpassit` field set emits, so the
/// `full` and `surface` render frames keep exactly the fields they had.
pub const MPASSIT_ONLY: &str = "mpassit";

pub fn remap_class(wrf_name: &str) -> RemapClass {
    if CONSERVATIVE_FIELDS.contains(&wrf_name) {
        RemapClass::Conservative
    } else if NEAREST_FIELDS.contains(&wrf_name) {
        RemapClass::Nearest
    } else {
        RemapClass::Linear
    }
}

/// The named field sets. A global 0.25 degree window carries four million
/// points per level, so the surface set keeps only what the surface,
/// precipitation, MSLP, precipitable-water and 100 m wind products read -- which is the
/// accurate content of a coarse overview anyway.
pub fn field_set_allows(field_set: &str, mapping: &FieldMapping) -> bool {
    if mapping.note == MPASSIT_ONLY {
        return field_set == "mpassit";
    }
    match field_set {
        "full" | "mpassit" => true,
        // REFL_10CM rides in the surface set because its consumer product --
        // composite reflectivity, the MRMS-comparable quantity -- is a
        // surface-deliverable map even though the carrier is 3-D.
        "surface" => matches!(
            mapping.wrf_name,
            "T" | "P" | "PB" | "QVAPOR" | "PH" | "PHB" | "U" | "V" | "W" | "REFL_10CM"
        ) || matches!(mapping.rank, Rank::Surface | Rank::Soil),
        _ => true,
    }
}

pub fn field_set_is_known(field_set: &str) -> bool {
    matches!(field_set, "full" | "surface" | "mpassit")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mapping(wrf_name: &str) -> &'static FieldMapping {
        FIELD_MAP
            .iter()
            .find(|m| m.wrf_name == wrf_name)
            .unwrap_or_else(|| panic!("{wrf_name} is not mapped; the obs referee's model bundle refuses frames without it"))
    }

    #[test]
    fn export_carriers_survive_conversion_field_sets() {
        for name in ["P", "PB", "PH", "PHB", "QVAPOR", "W", "U", "V", "T", "HGT"] {
            let row = FIELD_MAP.iter().find(|row| row.wrf_name == name).unwrap();
            assert!(field_set_allows("full", row), "full lost {name}");
            assert!(field_set_allows("surface", row), "surface lost {name}");
        }
    }

    #[test]
    fn every_accumulation_is_conservative_and_every_category_nearest() {
        for m in FIELD_MAP {
            let class = remap_class(m.wrf_name);
            if m.units == "mm" {
                assert_eq!(class, RemapClass::Conservative, "{} is an accumulation", m.wrf_name);
            }
            if m.units.is_empty() && m.source != FieldSource::Constant {
                assert_eq!(class, RemapClass::Nearest, "{} is a category", m.wrf_name);
            }
        }
        assert_eq!(remap_class("T2"), RemapClass::Linear);
        assert_eq!(remap_class("REFL_10CM"), RemapClass::Linear);
    }

    #[test]
    fn q2_is_a_mapped_surface_history_field() {
        let q2 = mapping("Q2");
        assert_eq!(q2.mpas_name, Some("q2"));
        assert_eq!(q2.rank, Rank::Surface);
        assert_eq!(q2.source, FieldSource::History);
        assert!(field_set_allows("surface", q2));
    }

    #[test]
    fn refl_10cm_is_a_mapped_mass3d_history_field_in_both_sets() {
        let refl = mapping("REFL_10CM");
        assert_eq!(refl.mpas_name, Some("refl10cm"));
        assert_eq!(refl.rank, Rank::Mass3d);
        assert_eq!(refl.source, FieldSource::History);
        assert!(field_set_allows("surface", refl));
        assert!(field_set_allows("full", refl));
    }

    #[test]
    fn the_declared_absent_list_no_longer_names_the_mapped_pair() {
        for (name, _, _) in ABSENT_WRF_FIELDS {
            assert_ne!(*name, "Q2");
            assert_ne!(*name, "REFL_10CM");
        }
    }
}

/// The convective Z-R relation, Z = A R^B (Z in mm^6 m^-3, R in mm h^-1): the WSR-88D
/// default convective relation, Z = 300 R^1.4 (Fulton et al. 1998, Wea. Forecasting 13).
pub const CONVECTIVE_ZR_A: f64 = 300.0;
pub const CONVECTIVE_ZR_B: f64 = 1.4;
/// Below this a resolved REFC carries no echo (MPAS-A's diagnostic floor is -35 dBZ).
const NO_ECHO_DBZ: f64 = -30.0;

/// Adds the parameterised convection's reflectivity to the resolved composite reflectivity, cell by
/// cell: REFC = 10 log10(Z_resolved + 300 R^1.4), R the cumulus scheme's rain rate (`CUPREC`, mm/s,
/// as mm/h).  A cell the cumulus scheme does not rain on keeps its value exactly.  Returns how many
/// cells changed.
///
/// THE BREAKAGE THIS PREVENTS (WOOF Hex with Regional, run_9a0796b2e9c8e251, 2026-10-28 20Z): on a
/// 120 km mesh the storms are the cumulus scheme's, and its rain makes no hydrometeors, so the
/// microphysics-only REFC drew nothing over Minnesota and Wisconsin while MRMS had a 55-65 dBZ
/// line; the same frame with this term draws 20-30 dBZ there.  Post-processors add the same term
/// (NCEP's UPP adds a convective reflectivity from the convective rain rate to its composite).
pub fn add_convective_reflectivity(refc: &mut [f32], cuprec_mm_per_s: &[f32]) -> usize {
    let mut changed = 0;
    for (dbz, rate) in refc.iter_mut().zip(cuprec_mm_per_s) {
        let rate = f64::from(*rate);
        if !(rate > 0.0) || !rate.is_finite() {
            continue;
        }
        let r = rate * 3600.0;
        let z_conv = CONVECTIVE_ZR_A * r.powf(CONVECTIVE_ZR_B);
        let resolved = f64::from(*dbz);
        let z_resolved = if resolved.is_finite() && resolved > NO_ECHO_DBZ {
            10f64.powf(resolved / 10.0)
        } else {
            0.0
        };
        let combined = 10.0 * (z_resolved + z_conv).log10();
        if combined.is_finite() && (resolved.is_nan() || combined > resolved) {
            *dbz = combined as f32;
            changed += 1;
        }
    }
    changed
}

#[cfg(test)]
mod convective_tests {
    use super::*;

    #[test]
    fn no_cumulus_rain_leaves_refc_exactly() {
        let mut refc = vec![-35.0f32, 12.5, 41.0];
        assert_eq!(add_convective_reflectivity(&mut refc, &[0.0, 0.0, -1.0]), 0);
        assert_eq!(refc, vec![-35.0, 12.5, 41.0]);
    }

    #[test]
    fn convective_rain_alone_gives_its_z_r_reflectivity() {
        // 2.32 mm/h, the strongest cumulus rate over MN/WI at 20Z: 300 * 2.32^1.4 -> 29.9 dBZ.
        let mut refc = vec![-35.0f32];
        add_convective_reflectivity(&mut refc, &[2.32 / 3600.0]);
        assert!((refc[0] - 29.9).abs() < 0.05, "{}", refc[0]);
        // 10 mm/h: 10 log10(300 * 10^1.4) = 38.8 dBZ.
        let mut refc = vec![-35.0f32];
        add_convective_reflectivity(&mut refc, &[10.0 / 3600.0]);
        assert!((refc[0] - 38.77).abs() < 0.05, "{}", refc[0]);
    }

    #[test]
    fn resolved_and_convective_echoes_add_in_linear_z() {
        // 30 dBZ resolved + 30 dBZ convective = 33.0 dBZ.
        let r = (1000.0f64 / 300.0).powf(1.0 / 1.4); // mm/h giving Z = 1000
        let mut refc = vec![30.0f32];
        add_convective_reflectivity(&mut refc, &[(r / 3600.0) as f32]);
        assert!((refc[0] - 33.01).abs() < 0.02, "{}", refc[0]);
    }

    #[test]
    fn cuprec_is_mapped_and_read_with_the_surface_set() {
        let m = FIELD_MAP.iter().find(|m| m.wrf_name == "CUPREC").expect("CUPREC is mapped");
        assert_eq!(m.mpas_name, Some("cuprec"));
        assert!(field_set_allows("surface", m));
    }
}
