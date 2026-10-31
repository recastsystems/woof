//! Fail-closed native HRRR GRIB2 subset bridge for gpuwm initialization.
//!
//! The bridge is intentionally narrow.  It accepts one contiguous public
//! source-lead window from a single HRRR cycle, proves the exact atmosphere/surface/soil
//! inventory needed by the WSM6 initialization lane, and writes a
//! south-to-north row-major FP32 source window.  No field is selected by
//! display name or message position.

use gpuwm_preprocess_cpu::quantization::{
    decode_quantum, BoundKind, BoundVerdict, Bounds, ClampTally,
};
use grib_core::grib2::{unpack_message, Grib2File, Grib2Message, GridDefinition};
use std::env;
use std::collections::BTreeMap;
use gpuwm_preprocess_cpu::grib2_supplement::{is_pmsl_pa, select_pmsl};
use std::error::Error;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;
use rayon::prelude::*;

const HYBRID_LEVEL_TYPE: u8 = 105;
const SOIL_LEVEL_TYPE: u8 = 106;
const SURFACE_LEVEL_TYPE: u8 = 1;
const HEIGHT_LEVEL_TYPE: u8 = 103;
const N_HYBRID_LEVELS: usize = 50;
const SOIL_DEPTHS_M: [f64; 9] = [0.0, 0.01, 0.04, 0.1, 0.3, 0.6, 1.0, 1.6, 3.0];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Parameter {
    discipline: u8,
    category: u8,
    number: u8,
}

#[derive(Clone, Copy, Debug)]
struct HybridSpec {
    name: &'static str,
    parameter: Parameter,
    /// Codes an earlier HRRR published this same field under.  One is read
    /// only from a file that publishes the field under `parameter` on no
    /// hybrid level at all, and then on all 50 levels: a file is read under
    /// exactly one code, never a mixture of the two.
    earlier_parameters: &'static [Parameter],
    nonnegative: bool,
    require_any_positive: bool,
}

impl HybridSpec {
    /// Every code this field may be published under, current first.
    fn candidate_parameters(&self) -> impl Iterator<Item = Parameter> + '_ {
        std::iter::once(self.parameter).chain(self.earlier_parameters.iter().copied())
    }
}

/// HRRRv3 and later (July 2018 on) publish cloud ice as CIMIXR, 0/1/82.
/// HRRRv1 and v2 wrfnat files publish the same mixing ratio as CICE,
/// 0/6/0, on the same 50 hybrid levels and carry no CIMIXR; the archive's
/// 2017-01-19 00Z wrfnatf00 and f01 hold CICE on 50 levels and no CIMIXR,
/// the 2026-09-27 00Z wrfnatf00 holds CIMIXR on 50 levels and no CICE.
/// Reading only 0/1/82 refused every cycle before July 2018 with
/// `missing required field QI hybrid level 1`.
const CLOUD_ICE_BEFORE_HRRR_V3: Parameter = Parameter {
    discipline: 0,
    category: 6,
    number: 0,
};

/// The analyzed water- and ice-friendly aerosol number concentrations
/// (kg-1), read only when the caller asks for them (`--analyzed-aerosol`:
/// a configuration whose preparation reads the analyzed aerosol,
/// gpuwm.preparation_assets.analyzed_aerosol_domains) and the file
/// publishes the pair on all 50 hybrid levels. Masked values follow the
/// operational missing-value chain at decode. Unrequested, the
/// pair is never selected, 2.8.6's inventory: NCEP masks PMTF at some
/// leads of some cycles (2026-10-04 06Z f06 and f07, clean at f00-f05 and
/// f08), and an as-posted series that read the pair at its reference lead
/// cannot withhold it at a later one, so reading it unasked stopped
/// default-profile runs that never use it.
/// NOAA's operational HRRR Vtable (parm/conus/hrrr_vtable) binds QNWFA and
/// QNIFA to the NCEP local codes 0/13/193 and 0/13/192 on surface 105;
/// indexes label them PMTF/PMTC (gpuwm/authorities/
/// rw-wps-hrrr-native-grib2.provenance.json).  Optional: a file with
/// neither publishes the same 561 records as before.  Without them an
/// mp=28 tree on this route asking for the analyzed aerosol
/// (use_rap_aero_icbc) refused, "missing QNIFA, QNWFA", although the
/// HRRR file in hand carries both.
const OPTIONAL_HYBRID_SPECS: [HybridSpec; 2] = [
    HybridSpec {
        name: "QNWFA",
        parameter: Parameter {
            discipline: 0,
            category: 13,
            number: 193,
        },
        earlier_parameters: &[],
        nonnegative: true,
        require_any_positive: false,
    },
    HybridSpec {
        name: "QNIFA",
        parameter: Parameter {
            discipline: 0,
            category: 13,
            number: 192,
        },
        earlier_parameters: &[],
        nonnegative: true,
        require_any_positive: false,
    },
];

/// Optional native-level records. Requests come from --extras, never defaults.
///
/// QNC/QNI/QNR are HRRR's own analyzed Thompson number concentrations (cloud
/// droplets NCONCD 0/6/28, cloud ice NCCICE 0/6/29, rain SPNCR 0/1/100, all
/// kg-1 on the 50 hybrid levels).  An HRRR start that carries HRRR's mass
/// but diagnoses the numbers from it is not HRRR's state; the names are the
/// WPS names the RRFS fork's case 7/9 read (mpas_init_atm_cases.F:5007-5009).
const EXTRA_SPECS: [HybridSpec; 6] = [
    HybridSpec { name: "MASSDEN", parameter: Parameter { discipline: 0, category: 20, number: 0 }, earlier_parameters: &[], nonnegative: true, require_any_positive: false },
    HybridSpec { name: "PMTF", parameter: Parameter { discipline: 0, category: 13, number: 193 }, earlier_parameters: &[], nonnegative: true, require_any_positive: false },
    HybridSpec { name: "PMTC", parameter: Parameter { discipline: 0, category: 13, number: 192 }, earlier_parameters: &[], nonnegative: true, require_any_positive: false },
    HybridSpec { name: "QNC", parameter: Parameter { discipline: 0, category: 6, number: 28 }, earlier_parameters: &[], nonnegative: true, require_any_positive: false },
    HybridSpec { name: "QNI", parameter: Parameter { discipline: 0, category: 6, number: 29 }, earlier_parameters: &[], nonnegative: true, require_any_positive: false },
    HybridSpec { name: "QNR", parameter: Parameter { discipline: 0, category: 1, number: 100 }, earlier_parameters: &[], nonnegative: true, require_any_positive: false },
];

/// Optional single-level records, requested through the same --extras list.
/// RH2 is HRRR's 2 m relative humidity (0/1/1 at 2 m above ground, percent),
/// the record the RAP/HRRR Vtable maps to RH at 200100: MPAS case 7 forms
/// its first-guess q2 from that record alone, so a start without it has
/// q2 = 0 everywhere.
const EXTRA_SURFACE_SPECS: [SurfaceSpec; 1] = [SurfaceSpec {
    name: "RH2",
    parameter: Parameter { discipline: 0, category: 1, number: 1 },
    level_type: 103,
    level_value: 2.0,
    nonnegative: true,
}];
const EXTRA_SURFACE_UNITS: [&str; 1] = ["%"];

#[cfg(test)]
fn requested_extras(text: &str) -> Result<Vec<HybridSpec>, Box<dyn Error>> {
    let (hybrid, surface) = requested_extra_sets(text)?;
    if let Some(spec) = surface.first() {
        return Err(format!("extra {} is a single-level record, not a hybrid one", spec.name).into());
    }
    Ok(hybrid)
}

/// The --extras list, split into the hybrid and the single-level tables.
fn requested_extra_sets(text: &str) -> Result<(Vec<HybridSpec>, Vec<SurfaceSpec>), Box<dyn Error>> {
    let mut hybrid: Vec<HybridSpec> = Vec::new();
    let mut surface: Vec<SurfaceSpec> = Vec::new();
    for name in text.split(',') {
        if hybrid.iter().any(|item| item.name == name) || surface.iter().any(|item| item.name == name) {
            return Err(format!("duplicate extra {name}: would duplicate the manifest inventory").into());
        }
        if let Some(spec) = EXTRA_SPECS.iter().find(|spec| spec.name == name) {
            hybrid.push(*spec);
        } else if let Some(spec) = EXTRA_SURFACE_SPECS.iter().find(|spec| spec.name == name) {
            surface.push(*spec);
        } else {
            return Err(format!("unknown extra {name:?}: no numeric selector exists to prevent decoding the wrong tracer").into());
        }
    }
    hybrid.sort_by_key(|spec| spec.name);
    surface.sort_by_key(|spec| spec.name);
    Ok((hybrid, surface))
}

const HYBRID_SPECS: [HybridSpec; 11] = [
    HybridSpec {
        name: "PRES",
        parameter: Parameter {
            discipline: 0,
            category: 3,
            number: 0,
        },
        earlier_parameters: &[],
        nonnegative: true,
        require_any_positive: false,
    },
    HybridSpec {
        name: "QC",
        parameter: Parameter {
            discipline: 0,
            category: 1,
            number: 22,
        },
        earlier_parameters: &[],
        nonnegative: true,
        require_any_positive: true,
    },
    HybridSpec {
        name: "QI",
        parameter: Parameter {
            discipline: 0,
            category: 1,
            number: 82,
        },
        earlier_parameters: &[CLOUD_ICE_BEFORE_HRRR_V3],
        nonnegative: true,
        require_any_positive: true,
    },
    HybridSpec {
        name: "QR",
        parameter: Parameter {
            discipline: 0,
            category: 1,
            number: 24,
        },
        earlier_parameters: &[],
        nonnegative: true,
        require_any_positive: true,
    },
    HybridSpec {
        name: "QS",
        parameter: Parameter {
            discipline: 0,
            category: 1,
            number: 25,
        },
        earlier_parameters: &[],
        nonnegative: true,
        require_any_positive: true,
    },
    HybridSpec {
        name: "QG",
        parameter: Parameter {
            discipline: 0,
            category: 1,
            number: 32,
        },
        earlier_parameters: &[],
        nonnegative: true,
        require_any_positive: true,
    },
    HybridSpec {
        name: "HGT",
        parameter: Parameter {
            discipline: 0,
            category: 3,
            number: 5,
        },
        earlier_parameters: &[],
        nonnegative: false,
        require_any_positive: false,
    },
    HybridSpec {
        name: "TT",
        parameter: Parameter {
            discipline: 0,
            category: 0,
            number: 0,
        },
        earlier_parameters: &[],
        nonnegative: true,
        require_any_positive: false,
    },
    HybridSpec {
        name: "SPFH",
        parameter: Parameter {
            discipline: 0,
            category: 1,
            number: 0,
        },
        earlier_parameters: &[],
        nonnegative: true,
        require_any_positive: false,
    },
    HybridSpec {
        name: "U_MASS",
        parameter: Parameter {
            discipline: 0,
            category: 2,
            number: 2,
        },
        earlier_parameters: &[],
        nonnegative: false,
        require_any_positive: false,
    },
    HybridSpec {
        name: "V_MASS",
        parameter: Parameter {
            discipline: 0,
            category: 2,
            number: 3,
        },
        earlier_parameters: &[],
        nonnegative: false,
        require_any_positive: false,
    },
];

#[derive(Clone, Copy, Debug)]
struct SurfaceSpec {
    name: &'static str,
    parameter: Parameter,
    level_type: u8,
    level_value: f64,
    nonnegative: bool,
}

const SURFACE_SPECS: [SurfaceSpec; 11] = [
    SurfaceSpec {
        name: "PSFC",
        parameter: Parameter {
            discipline: 0,
            category: 3,
            number: 0,
        },
        level_type: SURFACE_LEVEL_TYPE,
        level_value: 0.0,
        nonnegative: true,
    },
    SurfaceSpec {
        name: "SOILHGT",
        parameter: Parameter {
            discipline: 0,
            category: 3,
            number: 5,
        },
        level_type: SURFACE_LEVEL_TYPE,
        level_value: 0.0,
        nonnegative: false,
    },
    SurfaceSpec {
        name: "SKINTEMP",
        parameter: Parameter {
            discipline: 0,
            category: 0,
            number: 0,
        },
        level_type: SURFACE_LEVEL_TYPE,
        level_value: 0.0,
        nonnegative: true,
    },
    SurfaceSpec {
        name: "SNOW",
        parameter: Parameter {
            discipline: 0,
            category: 1,
            number: 13,
        },
        level_type: SURFACE_LEVEL_TYPE,
        level_value: 0.0,
        nonnegative: true,
    },
    SurfaceSpec {
        name: "SNOWH",
        parameter: Parameter {
            discipline: 0,
            category: 1,
            number: 11,
        },
        level_type: SURFACE_LEVEL_TYPE,
        level_value: 0.0,
        nonnegative: true,
    },
    SurfaceSpec {
        name: "T2",
        parameter: Parameter {
            discipline: 0,
            category: 0,
            number: 0,
        },
        level_type: HEIGHT_LEVEL_TYPE,
        level_value: 2.0,
        nonnegative: true,
    },
    SurfaceSpec {
        name: "Q2",
        parameter: Parameter {
            discipline: 0,
            category: 1,
            number: 0,
        },
        level_type: HEIGHT_LEVEL_TYPE,
        level_value: 2.0,
        nonnegative: true,
    },
    SurfaceSpec {
        name: "U10_MASS",
        parameter: Parameter {
            discipline: 0,
            category: 2,
            number: 2,
        },
        level_type: HEIGHT_LEVEL_TYPE,
        level_value: 10.0,
        nonnegative: false,
    },
    SurfaceSpec {
        name: "V10_MASS",
        parameter: Parameter {
            discipline: 0,
            category: 2,
            number: 3,
        },
        level_type: HEIGHT_LEVEL_TYPE,
        level_value: 10.0,
        nonnegative: false,
    },
    SurfaceSpec {
        name: "LANDSEA",
        parameter: Parameter {
            discipline: 2,
            category: 0,
            number: 0,
        },
        level_type: SURFACE_LEVEL_TYPE,
        level_value: 0.0,
        nonnegative: true,
    },
    SurfaceSpec {
        name: "XICE",
        parameter: Parameter {
            discipline: 10,
            category: 2,
            number: 0,
        },
        level_type: SURFACE_LEVEL_TYPE,
        level_value: 0.0,
        nonnegative: true,
    },
];

/// The PBL height the cycle's own forecast carried into this analysis
/// (HPBL, WMO 0/3/18, surface), read when the wrfnat publishes it.  An HRRR
/// f00 is a cycled state: its wrfinput holds the background forecast's
/// PBLH, so NOAA's f00 products (the GUST message, every PBL-depth
/// product) see a boundary layer.  A cold start that seeds zero, as real.exe
/// does, wrote PBLH = 0 at every point of the initial history, so the f00
/// gust was the bare 10 m wind (measured 2026-10-21 06Z: WOOF f00 gust
/// 2.6 m/s weaker than the same wrf-rust gust on NOAA's own f00 state,
/// where PBLH averaged 347 m).  Optional: a file without it selects exactly
/// the records it always did, and the run keeps the stock zero.
///
/// The low, middle and high cloud cover the cycle carries (LCDC, MCDC,
/// HCDC, WMO 0/6/3-5 on the layer surfaces 214/224/234, percent) seed the
/// initial CLDFRA's cover the same way: the wrfnat publishes no 3-D cloud
/// fraction, and HRRR's cover is mostly MYNN subgrid cloud that a start
/// state does not hold (2026-10-21 06Z: the radiation's own cloud fraction
/// of the start state gave a column-maximum mean of 6 % against NOAA's 54 %).
const OPTIONAL_SURFACE_SPECS: [SurfaceSpec; 4] = [
    SurfaceSpec {
        name: "PBLH",
        parameter: Parameter { discipline: 0, category: 3, number: 18 },
        level_type: SURFACE_LEVEL_TYPE,
        level_value: 0.0,
        nonnegative: true,
    },
    SurfaceSpec {
        name: "LCDC",
        parameter: Parameter { discipline: 0, category: 6, number: 3 },
        level_type: 214,
        level_value: 0.0,
        nonnegative: true,
    },
    SurfaceSpec {
        name: "MCDC",
        parameter: Parameter { discipline: 0, category: 6, number: 4 },
        level_type: 224,
        level_value: 0.0,
        nonnegative: true,
    },
    SurfaceSpec {
        name: "HCDC",
        parameter: Parameter { discipline: 0, category: 6, number: 5 },
        level_type: 234,
        level_value: 0.0,
        nonnegative: true,
    },
];
/// Units the gate declares for each optional surface field, in table order.
const OPTIONAL_SURFACE_UNITS: [&str; 4] = ["m", "%", "%", "%"];

// Optional analyzed fields carried beside the soil subset. The fetch source
// metadata declares which records it appends; selection is numeric here.
const OPTIONAL_SOIL_SURFACE_SPECS: [SurfaceSpec; 1] = [SurfaceSpec {
    name: "VEGFRA", parameter: Parameter { discipline: 2, category: 0, number: 4 },
    level_type: SURFACE_LEVEL_TYPE, level_value: 0.0, nonnegative: true,
}];

#[derive(Clone, Debug)]
struct SelectedField {
    index: usize,
    variable: &'static str,
    level_value: f64,
    /// The code the record was selected under, so the cross-time check
    /// refuses a series whose frames publish a field under different
    /// codes and the gate names the cloud-ice code it bound.
    parameter: Parameter,
}

#[derive(Clone, Debug)]
struct AtmosInventory {
    source_grid: GridDefinition,
    selected: Vec<SelectedField>,
    reference_time: String,
    forecast_hour: u32,
    grid: GridFingerprint,

}

#[derive(Clone, Debug)]
struct SoilInventory {
    selected: Vec<SelectedField>,
    optional_surface: Vec<SelectedField>,
    reference_time: String,
    forecast_hour: u32,
    grid: GridFingerprint,
}

#[derive(Clone, Debug)]
struct SeriesInput {
    forecast_hour: u32,
    atmosphere: String,
    soil: String,
    supplements: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct GridFingerprint {
    template: u16,
    nx: u32,
    ny: u32,
    lat1: u64,
    lon1: u64,
    dx: u64,
    dy: u64,
    latin1: u64,
    latin2: u64,
    lov: u64,
    scan_mode: u8,
    shape_of_earth: u8,
    resolution_flags: u8,
}

impl GridFingerprint {
    fn from_grid(grid: &GridDefinition) -> Self {
        Self {
            template: grid.template,
            nx: grid.nx,
            ny: grid.ny,
            lat1: grid.lat1.to_bits(),
            lon1: grid.lon1.to_bits(),
            dx: grid.dx.to_bits(),
            dy: grid.dy.to_bits(),
            latin1: grid.latin1.to_bits(),
            latin2: grid.latin2.to_bits(),
            lov: grid.lov.to_bits(),
            scan_mode: grid.scan_mode,
            shape_of_earth: grid.shape_of_earth,
            resolution_flags: grid.resolution_flags,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Window {
    i_start: usize,
    i_end: usize,
    j_start: usize,
    j_end: usize,
}

impl Window {
    fn nx(self) -> usize {
        self.i_end - self.i_start + 1
    }
    fn ny(self) -> usize {
        self.j_end - self.j_start + 1
    }
}

fn parameter_matches(message: &Grib2Message, parameter: Parameter) -> bool {
    message.discipline == parameter.discipline
        && message.product.parameter_category == parameter.category
        && message.product.parameter_number == parameter.number
}

fn level_matches(actual: f64, expected: f64) -> bool {
    (actual - expected).abs() <= 1.0e-9
}

/// HRRR's first grid point.  Current files publish it in microdegrees
/// (21.138123, 237.280472); the 2016 archive files publish the same point
/// rounded to millidegrees (21.138, 237.28) under an otherwise identical
/// grid definition -- same 1799 x 1059, 3 km, 38.5/38.5/262.5 cone.  Both
/// are this grid; any other corner is a different grid and stays refused,
/// because the crop window is placed on the canonical grid's own geometry.
fn canonical_corner(lat1: f64, lon1: f64) -> bool {
    (level_matches(lat1, 21.138123) && level_matches(lon1, 237.280472))
        || (level_matches(lat1, 21.138) && level_matches(lon1, 237.28))
}

fn validate_canonical_grid(grid: &GridDefinition) -> Result<(), Box<dyn Error>> {
    if grid.template != 30
        || grid.nx != 1799
        || grid.ny != 1059
        || grid.scan_mode != 0x40
        || grid.shape_of_earth != 6
        || grid.resolution_flags != 0x08
        || !canonical_corner(grid.lat1, grid.lon1)
        || !level_matches(grid.dx, 3000.0)
        || !level_matches(grid.dy, 3000.0)
        || !level_matches(grid.latin1, 38.5)
        || !level_matches(grid.latin2, 38.5)
        || !level_matches(grid.lov, 262.5)
    {
        return Err(format!("field is not on the canonical HRRR Lambert grid: {grid:?}").into());
    }
    Ok(())
}

fn validate_message_common(
    message: &Grib2Message,
    expected_cycle: &str,
    expected_forecast_hour: u32,
) -> Result<(), Box<dyn Error>> {
    if message.reference_time.to_string() != expected_cycle {
        return Err(format!(
            "reference time {} does not equal requested cycle {expected_cycle}",
            message.reference_time
        )
        .into());
    }
    if message.product.time_range_unit != 1
        || message.product.forecast_time != expected_forecast_hour
    {
        return Err(format!(
            "field does not resolve to requested forecast hour f{expected_forecast_hour:02}: unit={} time={}",
            message.product.time_range_unit, message.product.forecast_time
        ).into());
    }
    if message.product.template != 0 {
        return Err(format!(
            "selected field uses PDT {}, expected instantaneous PDT 4.0",
            message.product.template
        )
        .into());
    }
    validate_native_packing(message)?;
    validate_canonical_grid(&message.grid)
}

fn validate_native_packing(message: &Grib2Message) -> Result<(), Box<dyn Error>> {
    if !matches!(message.data_rep.template, 0 | 3) {
        return Err(format!(
            "selected field uses unsupported DRT 5.{}",
            message.data_rep.template
        )
        .into());
    }
    Ok(())
}

fn validate_payload(message: &Grib2Message) -> Result<(), Box<dyn Error>> {
    let expected_points = u64::from(message.grid.nx) * u64::from(message.grid.ny);
    let packed_points = if let Some(mask) = &message.bitmap {
        if mask.len() as u64 != expected_points {
            return Err(format!("selected field bitmap has {} cells, expected {expected_points}", mask.len()).into());
        }
        mask.iter().filter(|present| **present).count() as u64
    } else { expected_points };
    if u64::from(message.data_rep.section5_num_data_points) != packed_points {
        return Err(format!("selected field declares {} packed points, expected {packed_points} present cells",
            message.data_rep.section5_num_data_points).into());
    }
    Ok(())
}

fn payload_context(message: &Grib2Message, index: usize, variable: &str, window: Window) -> String {
    let nx = message.grid.nx as usize;
    let masked = message.bitmap.as_ref().map(|mask| {
        mask.iter().enumerate().filter(|(cell, present)| {
            !**present && nx != 0 && window.i_start <= cell % nx && cell % nx <= window.i_end
                && window.j_start <= cell / nx && cell / nx <= window.j_end
        }).count()
    }).unwrap_or(0);
    format!("{variable} level {} (level_type={}, message {index}, f{:02}) crop i={}..{} j={}..{} masked_inside_crop={masked}",
        message.product.level_value, message.product.level_type, message.product.forecast_time,
        window.i_start, window.i_end, window.j_start, window.j_end)
}

fn validate_crop_payload(message: &Grib2Message, index: usize, variable: &str, window: Window)
    -> Result<(), Box<dyn Error>> {
    let context = payload_context(message, index, variable, window);
    validate_payload(message).map_err(|error| format!("{context}: {error}"))?;
    if let Some(mask) = &message.bitmap {
        let nx = message.grid.nx as usize;
        let masked = (window.j_start..=window.j_end).flat_map(|j|
            (window.i_start..=window.i_end).map(move |i| j * nx + i))
            .filter(|cell| !mask[*cell]).count();
        if masked != 0 && !OPTIONAL_HYBRID_NAMES.contains(&variable) {
            return Err(format!("{context}: missing source cells prevent finite initialization; no missing-value donor policy is declared for this field").into());
        }
    }
    Ok(())
}

fn unique_match<F>(
    messages: &[Grib2Message],
    description: &str,
    predicate: F,
) -> Result<usize, Box<dyn Error>>
where
    F: Fn(&Grib2Message) -> bool,
{
    let matches: Vec<usize> = messages
        .iter()
        .enumerate()
        .filter_map(|(index, message)| predicate(message).then_some(index))
        .collect();
    match matches.as_slice() {
        [index] => Ok(*index),
        [] => Err(format!("missing required field {description}").into()),
        _ => Err(format!("duplicate required field {description}: indices {matches:?}").into()),
    }
}

/// The one code a file publishes `spec` under: the first candidate the
/// file carries on any hybrid level.  All 50 levels are then selected under
/// that code alone.  A file carrying none of them keeps the current code,
/// so the refusal names the field as current HRRR publishes it.
fn published_parameter(messages: &[Grib2Message], spec: &HybridSpec) -> Parameter {
    spec.candidate_parameters()
        .find(|&parameter| {
            messages.iter().any(|message| {
                parameter_matches(message, parameter)
                    && message.product.template == 0
                    && message.product.level_type == HYBRID_LEVEL_TYPE
            })
        })
        .unwrap_or(spec.parameter)
}

fn inventory_atmosphere(
    path: &str,
    expected_cycle: &str,
    forecast_hour: u32,
    extras: &[HybridSpec],
    surface_extras: &[SurfaceSpec],
    analyzed_aerosol: bool,
) -> Result<AtmosInventory, Box<dyn Error>> {
    let file = Grib2File::open(path)?;
    let mut inventory = inventory_atmosphere_with_extras(
        &file.messages, expected_cycle, forecast_hour, extras, analyzed_aerosol)?;
    select_surface_extras(&file.messages, expected_cycle, forecast_hour, surface_extras, &mut inventory)?;
    Ok(inventory)
}

/// A requested single-level extra must be published exactly once, on the
/// hybrid grid: requested and absent is a refusal, never a silent gap.
fn select_surface_extras(
    messages: &[Grib2Message], expected_cycle: &str, forecast_hour: u32,
    surface_extras: &[SurfaceSpec], inventory: &mut AtmosInventory,
) -> Result<(), Box<dyn Error>> {
    for spec in surface_extras {
        let description = format!("{} level type {} value {}", spec.name, spec.level_type, spec.level_value);
        let index = unique_match(messages, &description, |message| {
            parameter_matches(message, spec.parameter)
                && message.product.template == 0
                && message.product.level_type == spec.level_type
                && level_matches(message.product.level_value, spec.level_value)
        })?;
        let message = &messages[index];
        validate_message_common(message, expected_cycle, forecast_hour)?;
        if inventory.grid != GridFingerprint::from_grid(&message.grid) {
            return Err(format!("extra {description} grid differs from the hybrid grid").into());
        }
        inventory.selected.push(SelectedField {
            index,
            variable: spec.name,
            level_value: spec.level_value,
            parameter: spec.parameter,
        });
    }
    Ok(())
}

/// A test inventory as a configuration that requests the analyzed aerosol
/// pair takes it (`--analyzed-aerosol`).
#[cfg(test)]
fn inventory_atmosphere_messages(
    messages: &[Grib2Message],
    expected_cycle: &str,
    forecast_hour: u32,
) -> Result<AtmosInventory, Box<dyn Error>> {
    inventory_atmosphere_with_extras(messages, expected_cycle, forecast_hour, &[], true)
}

/// A test inventory as a configuration that does not request the pair
/// (the default HRRR profile) takes it.
#[cfg(test)]
fn inventory_atmosphere_unrequested(
    messages: &[Grib2Message],
    expected_cycle: &str,
    forecast_hour: u32,
) -> Result<AtmosInventory, Box<dyn Error>> {
    inventory_atmosphere_with_extras(messages, expected_cycle, forecast_hour, &[], false)
}

fn inventory_atmosphere_with_extras(
    messages: &[Grib2Message], expected_cycle: &str, forecast_hour: u32,
    extras: &[HybridSpec], analyzed_aerosol: bool,
) -> Result<AtmosInventory, Box<dyn Error>> {
    let mut selected = Vec::with_capacity((HYBRID_SPECS.len() + extras.len()) * N_HYBRID_LEVELS + SURFACE_SPECS.len());
    let mut common_grid: Option<GridFingerprint> = None;
    for spec in HYBRID_SPECS.iter().chain(extras.iter()) {
        let parameter = published_parameter(messages, spec);
        for level in 1..=N_HYBRID_LEVELS {
            let description = format!("{} hybrid level {level}", spec.name);
            let index = unique_match(messages, &description, |message| {
                parameter_matches(message, parameter)
                    && message.product.template == 0
                    && message.product.level_type == HYBRID_LEVEL_TYPE
                    && level_matches(message.product.level_value, level as f64)
            })?;
            let message = &messages[index];
            validate_message_common(message, expected_cycle, forecast_hour)?;
            let fingerprint = GridFingerprint::from_grid(&message.grid);
            if let Some(ref expected) = common_grid {
                if &fingerprint != expected {
                    return Err(
                        format!("{description} grid differs from the first selected grid").into(),
                    );
                }
            } else {
                common_grid = Some(fingerprint);
            }
            selected.push(SelectedField {
                index,
                variable: spec.name,
                level_value: level as f64,
                parameter,
            });
        }
    }
    // The optional pair, only when requested: both on every hybrid level,
    // or neither at all.  Unrequested, its records are never looked at.
    let published: Vec<usize> = OPTIONAL_HYBRID_SPECS
        .iter()
        .filter(|_| analyzed_aerosol)
        .map(|spec| {
            messages
                .iter()
                .filter(|message| {
                    parameter_matches(message, spec.parameter)
                        && message.product.template == 0
                        && message.product.level_type == HYBRID_LEVEL_TYPE
                })
                .count()
        })
        .collect();
    if published.iter().any(|&count| count != 0) {
        if published.iter().any(|&count| count < N_HYBRID_LEVELS) {
            return Err(format!(
                "analyzed aerosol numbers QNWFA/QNIFA are published on {published:?} hybrid \
                 records of {N_HYBRID_LEVELS} each; the pair is read whole or not at all"
            )
            .into());
        }
        for spec in OPTIONAL_HYBRID_SPECS {
            for level in 1..=N_HYBRID_LEVELS {
                let description = format!("{} hybrid level {level}", spec.name);
                let index = unique_match(messages, &description, |message| {
                    parameter_matches(message, spec.parameter)
                        && message.product.template == 0
                        && message.product.level_type == HYBRID_LEVEL_TYPE
                        && level_matches(message.product.level_value, level as f64)
                })?;
                let message = &messages[index];
                validate_message_common(message, expected_cycle, forecast_hour)?;
                if common_grid.as_ref() != Some(&GridFingerprint::from_grid(&message.grid)) {
                    return Err(format!("{description} grid differs from the hybrid grid").into());
                }
                selected.push(SelectedField {
                    index,
                    variable: spec.name,
                    level_value: level as f64,
                    parameter: spec.parameter,
                });
            }
        }
    }
    for spec in SURFACE_SPECS {
        let description = format!(
            "{} level type {} value {}",
            spec.name, spec.level_type, spec.level_value
        );
        let index = unique_match(messages, &description, |message| {
            parameter_matches(message, spec.parameter)
                && message.product.template == 0
                && message.product.level_type == spec.level_type
                && level_matches(message.product.level_value, spec.level_value)
        })?;
        let message = &messages[index];
        validate_message_common(message, expected_cycle, forecast_hour)?;
        let fingerprint = GridFingerprint::from_grid(&message.grid);
        if common_grid.as_ref() != Some(&fingerprint) {
            return Err(format!("{description} grid differs from the hybrid grid").into());
        }
        selected.push(SelectedField {
            index,
            variable: spec.name,
            level_value: spec.level_value,
            parameter: spec.parameter,
        });
    }
    for spec in OPTIONAL_SURFACE_SPECS {
        let matches: Vec<usize> = messages.iter().enumerate().filter(|(_, message)| {
            parameter_matches(message, spec.parameter)
                && message.product.template == 0
                && message.product.level_type == spec.level_type
                && level_matches(message.product.level_value, spec.level_value)
        }).map(|(index, _)| index).collect();
        if matches.len() > 1 {
            return Err(format!("duplicate optional surface {} in atmosphere input: indices {matches:?}", spec.name).into());
        }
        if let Some(&index) = matches.first() {
            let message = &messages[index];
            validate_message_common(message, expected_cycle, forecast_hour)?;
            if common_grid.as_ref() != Some(&GridFingerprint::from_grid(&message.grid)) {
                return Err(format!("optional surface {} grid differs from the hybrid grid", spec.name).into());
            }
            selected.push(SelectedField {
                index,
                variable: spec.name,
                level_value: spec.level_value,
                parameter: spec.parameter,
            });
        }
    }
    Ok(AtmosInventory {
        source_grid: messages[selected[0].index].grid.clone(),
        selected,
        reference_time: expected_cycle.to_owned(),
        forecast_hour,
        grid: common_grid.ok_or("empty atmosphere inventory")?,
    })
}

/// The requested pair is declared whole. Bitmap handling happens with the
/// concrete crop at decode, including the operational missing-value chain.
fn optional_hybrid_gate_lines(inventory: &AtmosInventory) -> Vec<String> {
    let mut lines = Vec::new();
    let optional_hybrid = optional_hybrid_fields(inventory);
    if !optional_hybrid.is_empty() {
        lines.push(format!("optional_hybrid_fields\t{}", optional_hybrid.join(",")));
        let units = optional_hybrid.iter().map(|name| format!("{name}=kg-1"))
            .collect::<Vec<_>>().join(",");
        lines.push(format!("optional_hybrid_units\t{units}"));
    }
    lines
}

const OPTIONAL_HYBRID_NAMES: [&str; 2] = [OPTIONAL_HYBRID_SPECS[0].name, OPTIONAL_HYBRID_SPECS[1].name];

fn inventory_soil(
    path: &str,
    expected_cycle: &str,
    forecast_hour: u32,
) -> Result<SoilInventory, Box<dyn Error>> {
    let file = Grib2File::open(path)?;
    let mut selected = Vec::with_capacity(18);
    let mut common_grid: Option<GridFingerprint> = None;
    for (name, parameter_number) in [("SOILT", 2u8), ("SOILW", 192u8)] {
        let parameter = Parameter {
            discipline: 2,
            category: 0,
            number: parameter_number,
        };
        for depth in SOIL_DEPTHS_M {
            let description = format!("{name} depth {depth} m");
            let index = unique_match(&file.messages, &description, |message| {
                parameter_matches(message, parameter) && message.product.template == 0
                    && message.product.level_type == SOIL_LEVEL_TYPE
                    && level_matches(message.product.level_value, depth)
            })?;
            let message = &file.messages[index];
            validate_message_common(message, expected_cycle, forecast_hour)?;
            let fingerprint = GridFingerprint::from_grid(&message.grid);
            if let Some(ref expected) = common_grid {
                if &fingerprint != expected {
                    return Err(format!(
                        "{description} grid differs from the first selected soil grid"
                    )
                    .into());
                }
            } else {
                common_grid = Some(fingerprint);
            }
            selected.push(SelectedField {
                index,
                variable: name,
                level_value: depth,
                parameter,
            });
        }
    }
    if selected.len() != 18 {
        return Err(format!(
            "soil inventory has {} selected records, expected 18",
            selected.len()
        )
        .into());
    }
    let mut optional_surface = Vec::new();
    for spec in OPTIONAL_SOIL_SURFACE_SPECS {
        let matches: Vec<_> = file.messages.iter().enumerate().filter(|(_, message)| {
            parameter_matches(message, spec.parameter) && message.product.template == 0
                && message.product.level_type == spec.level_type
                && level_matches(message.product.level_value, spec.level_value)
        }).collect();
        if matches.len() > 1 {
            return Err(format!("duplicate optional surface {} in soil input", spec.name).into());
        }
        if let Some((index, message)) = matches.first() {
            validate_message_common(message, expected_cycle, forecast_hour)?;
            if common_grid.as_ref() != Some(&GridFingerprint::from_grid(&message.grid)) {
                return Err(format!("optional surface {} grid differs from soil", spec.name).into());
            }
            optional_surface.push(SelectedField { index: *index, variable: spec.name,
                level_value: spec.level_value, parameter: spec.parameter });
        }
    }
    Ok(SoilInventory {
        selected,
        optional_surface,
        reference_time: expected_cycle.to_owned(),
        forecast_hour,
        grid: common_grid.ok_or("empty soil inventory")?,
    })
}

/// The optional surface fields an atmosphere inventory selected, in table order.
fn optional_surface_fields(inventory: &AtmosInventory) -> Vec<&'static str> {
    OPTIONAL_SURFACE_SPECS
        .iter()
        .filter(|spec| inventory.selected.iter().any(|field| field.variable == spec.name))
        .map(|spec| spec.name)
        .collect()
}

/// Declared with its unit, so the reader binds exactly what was written.
fn optional_surface_gate_lines(inventory: &AtmosInventory) -> Vec<String> {
    let fields = optional_surface_fields(inventory);
    if fields.is_empty() {
        return Vec::new();
    }
    let units = fields
        .iter()
        .map(|name| {
            let index = OPTIONAL_SURFACE_SPECS.iter().position(|spec| spec.name == *name).unwrap();
            format!("{name}={}", OPTIONAL_SURFACE_UNITS[index])
        })
        .collect::<Vec<_>>()
        .join(",");
    vec![
        format!("optional_surface_fields\t{}", fields.join(",")),
        format!("optional_surface_units\t{units}"),
    ]
}

/// The optional hybrid fields an atmosphere inventory selected, in table order.
fn optional_hybrid_fields(inventory: &AtmosInventory) -> Vec<&'static str> {
    OPTIONAL_HYBRID_SPECS
        .iter()
        .filter(|spec| inventory.selected.iter().any(|field| field.variable == spec.name))
        .map(|spec| spec.name)
        .collect()
}

fn compare_atmosphere_inventory(
    reference: &AtmosInventory,
    candidate: &AtmosInventory,
    expected_forecast_hour: u32,
) -> Result<(), Box<dyn Error>> {
    if reference.reference_time != candidate.reference_time
        || candidate.forecast_hour != expected_forecast_hour
    {
        return Err(format!(
            "atmosphere f{expected_forecast_hour:02} time inventory does not resolve to requested cycle/hour"
        )
        .into());
    }
    if reference.grid != candidate.grid {
        return Err(format!(
            "atmosphere f{expected_forecast_hour:02} grid does not exactly equal the first series frame"
        )
        .into());
    }
    let left: Vec<(&str, u64, Parameter)> = reference
        .selected
        .iter()
        .map(|field| (field.variable, field.level_value.to_bits(), field.parameter))
        .collect();
    let right: Vec<(&str, u64, Parameter)> = candidate
        .selected
        .iter()
        .map(|field| (field.variable, field.level_value.to_bits(), field.parameter))
        .collect();
    if left != right {
        let missing: Vec<_> = left.iter().filter(|key| !right.contains(key)).collect();
        let extra: Vec<_> = right.iter().filter(|key| !left.contains(key)).collect();
        return Err(format!(
            "atmosphere f{expected_forecast_hour:02} selected inventory does not exactly equal the first series frame; missing={missing:?}; extra={extra:?}"
        )
        .into());
    }
    Ok(())
}

fn compare_soil_inventory(
    reference: &SoilInventory,
    candidate: &SoilInventory,
    expected_forecast_hour: u32,
) -> Result<(), Box<dyn Error>> {
    if reference.reference_time != candidate.reference_time
        || candidate.forecast_hour != expected_forecast_hour
    {
        return Err(format!(
            "soil f{expected_forecast_hour:02} time inventory does not resolve to requested cycle/hour"
        )
        .into());
    }
    if reference.grid != candidate.grid {
        return Err(format!(
            "soil f{expected_forecast_hour:02} grid does not exactly equal the first series frame"
        )
        .into());
    }
    let left: Vec<(&str, u64, Parameter)> = reference
        .selected
        .iter()
        .map(|field| (field.variable, field.level_value.to_bits(), field.parameter))
        .collect();
    let right: Vec<(&str, u64, Parameter)> = candidate
        .selected
        .iter()
        .map(|field| (field.variable, field.level_value.to_bits(), field.parameter))
        .collect();
    let optional_left: Vec<_> = reference.optional_surface.iter()
        .map(|field| (field.variable, field.parameter)).collect();
    let optional_right: Vec<_> = candidate.optional_surface.iter()
        .map(|field| (field.variable, field.parameter)).collect();
    if optional_left != optional_right {
        return Err(format!("soil f{expected_forecast_hour:02} optional surface inventory differs from the first frame").into());
    }
    if left != right || left.len() != 18 {
        let missing: Vec<_> = left.iter().filter(|key| !right.contains(key)).collect();
        let extra: Vec<_> = right.iter().filter(|key| !left.contains(key)).collect();
        return Err(format!(
            "soil f{expected_forecast_hour:02} 18-record depth inventory does not exactly equal the first series frame; missing={missing:?}; extra={extra:?}"
        )
        .into());
    }
    Ok(())
}

/// The gate's cloud-ice line: the one code every QI record of the series
/// was selected under (the cross-time check has already made the frames
/// agree), then the value checks the decode applies to it.
fn qice_mapping_line(reference: &AtmosInventory) -> Result<String, Box<dyn Error>> {
    let qi = reference
        .selected
        .iter()
        .find(|field| field.variable == "QI")
        .ok_or("atmosphere inventory selected no QI record")?
        .parameter;
    Ok(format!(
        "qice_mapping\tPASS discipline={} category={} parameter={} level_type={HYBRID_LEVEL_TYPE}; finite/nonnegative/nonzero",
        qi.discipline, qi.category, qi.number
    ))
}

fn validate_window(window: Window, grid: &GridFingerprint) -> Result<(), Box<dyn Error>> {
    if window.i_start > window.i_end || window.j_start > window.j_end {
        return Err("window starts must not exceed window ends".into());
    }
    if window.i_end >= grid.nx as usize || window.j_end >= grid.ny as usize {
        return Err(format!(
            "window i={}..{}, j={}..{} exceeds source {}x{}",
            window.i_start, window.i_end, window.j_start, window.j_end, grid.nx, grid.ny
        )
        .into());
    }
    Ok(())
}

/// The inventory manifest's column names.
///
/// Rows are rendered by `manifest_row` and nowhere else, and a test pins
/// the two to the same width.  Three hand-written rows once gained two
/// columns while this line did not, and nothing in the build noticed: a
/// 13-column row shipped under an 11-column header, which is a receipt
/// that reads wrong rather than one that fails loudly.
const MANIFEST_HEADER: &str = "role\tindex\tvariable\tlevel_value\tlevel_type\tdrt\tbitmap\tdecoded_count\tminimum\tmaximum\tclamped\tmax_excursion\tfilename\tmasked_inside_crop\tnearest_neighbor\tfour_pt\taverage_4pt\tfill_zero";

#[allow(clippy::too_many_arguments)]
fn manifest_row(
    role: &str,
    index: usize,
    variable: &str,
    level_value: f64,
    level_type: u8,
    drt: u16,
    bitmap: bool,
    stats: Stats,
    filename: &str,
) -> String {
    format!(
        "{role}\t{index}\t{variable}\t{level_value}\t{level_type}\t{drt}\t{bitmap}\t{}\t{}\t{}\t{}\t{}\t{filename}\t{}\t{}\t{}\t{}\t{}",
        stats.count,
        stats.minimum,
        stats.maximum,
        stats.clamped.clamps,
        stats.clamped.max_excursion,
        stats.fill.masked, stats.fill.nearest_neighbor, stats.fill.four_pt,
        stats.fill.average_4pt, stats.fill.zero,
    )
}

#[derive(Clone, Copy, Debug)]
struct Stats {
    minimum: f64,
    maximum: f64,
    /// Preserve the pre-crop hydrometeor completeness check over the source.
    any_positive_in_source: bool,
    count: usize,
    /// Cells clamped back onto a physical bound they overshot by no more
    /// than this record's own packing step.
    clamped: ClampTally,
    fill: gpuwm_preprocess_cpu::missing_value_chain::Counts,
}

impl Stats {
    fn source_has_positive(&self) -> bool { self.any_positive_in_source }
}

/// The physical bounds a variable is held to, and how far past each the
/// packing grid is allowed to have pushed a cell that sits exactly on it.
///
/// A mixing ratio is zero across most of a domain and a fraction is one
/// over solid ice; both are encoded AT the limit and both can decode a
/// step outside it.  See `quantization` for the derivation -- this bridge
/// makes the same distinction, with `f64::INFINITY` standing in for "no
/// bound declared on this side".
///
/// These fields carry no declared range, so the tolerance ceiling is
/// anchored on `scale`: the magnitude the record's own data occupies,
/// measured before any clamping.  A field that is zero everywhere gets a
/// scale of zero and therefore no tolerance at all, which is right --
/// there is nothing there to have been quantized.
fn value_bounds(nonnegative: bool, unit_fraction: bool, scale: f64) -> Bounds {
    Bounds {
        minimum: if nonnegative { 0.0 } else { f64::NEG_INFINITY },
        maximum: if unit_fraction { 1.0 } else { f64::INFINITY },
        minimum_kind: if nonnegative {
            BoundKind::Physical
        } else {
            BoundKind::Sanity
        },
        maximum_kind: if unit_fraction {
            BoundKind::Physical
        } else {
            BoundKind::Sanity
        },
        scale,
    }
}

fn decode_crop_write(
    message: &Grib2Message,
    message_index: usize,
    writer: &mut BufWriter<File>,
    window: Window,
    variable: &str,
    nonnegative: bool,
    unit_fraction: bool,
) -> Result<Stats, Box<dyn Error>> {
    let (bytes, stats) =
        decode_crop(message, message_index, window, variable, nonnegative, unit_fraction)?;
    writer.write_all(&bytes)?;
    Ok(stats)
}

/// One record decoded and cropped to the window's FP32 little-endian bytes.
/// Pure in its inputs, so records decode on all cores at once and are then
/// written in their fixed order: the bytes are the serial loop's bytes.
/// The error is its text (`Send`), the same text the serial loop reported.
fn decode_crop(
    message: &Grib2Message,
    message_index: usize,
    window: Window,
    variable: &str,
    nonnegative: bool,
    unit_fraction: bool,
) -> Result<(Vec<u8>, Stats), String> {
    let mut out = Vec::with_capacity(window.nx() * window.ny() * 4);
    decode_crop_into(message, message_index, &mut out, window, variable, nonnegative, unit_fraction)
        .map(|stats| (out, stats))
        .map_err(|error| error.to_string())
}

fn decode_crop_into(
    message: &Grib2Message,
    message_index: usize,
    writer: &mut Vec<u8>,
    window: Window,
    variable: &str,
    nonnegative: bool,
    unit_fraction: bool,
) -> Result<Stats, Box<dyn Error>> {
    validate_window(window, &GridFingerprint::from_grid(&message.grid))?;
    validate_crop_payload(message, message_index, variable, window)?;
    let context = payload_context(message, message_index, variable, window);
    let mut values = unpack_message(message).map_err(|error| format!("{context}: {error}"))?;
    let nx = message.grid.nx as usize;
    let ny = message.grid.ny as usize;
    if values.len() != nx * ny {
        return Err(format!(
            "{variable} decoded {} values, expected {}",
            values.len(),
            nx * ny
        )
        .into());
    }
    // Preserve the field-wide finite check and tolerance anchor. A crop-only
    // scale could lower the packing tolerance in a dry window and refuse
    // values the original decoder clamped. Only the bound verdict below is
    // crop-local. The caller includes its remapping halo in this window;
    // integer-lattice missing-value donors need no additional halo.
    let mut scale = 0.0f64;
    let mut any_positive_in_source = false;
    for (cell, value) in values.iter().copied().enumerate() {
        if !value.is_finite() {
            if message.bitmap.as_ref().map(|mask| !mask[cell]).unwrap_or(false) {
                continue;
            }
            return Err(format!("{context}: non-finite present value at source row {}, column {}; bitmap does not mark it missing", cell / nx, cell % nx).into());
        }
        scale = scale.max(value.abs());
        any_positive_in_source |= value > 0.0;
    }
    let bounds = value_bounds(nonnegative, unit_fraction, scale);
    let quantum = decode_quantum(
        message.data_rep.template,
        message.data_rep.binary_scale,
        message.data_rep.decimal_scale,
    );
    let mut minimum = f64::INFINITY;
    let mut maximum = f64::NEG_INFINITY;
    let mut clamped = ClampTally::default();
    for row in values.chunks_exact_mut(nx).skip(window.j_start).take(window.ny()) {
      for value in &mut row[window.i_start..=window.i_end] {
        if !value.is_finite() { continue; }
        match bounds.check(*value, quantum) {
            BoundVerdict::Inside => {}
            BoundVerdict::Clamped {
                value: bound,
                excursion,
            } => {
                *value = bound;
                clamped.record(excursion);
            }
            BoundVerdict::Refuse {
                excursion,
                tolerance,
            } => {
                return Err(format!(
                    "{variable} value {value} outside [{},{}] by {excursion} \
                     (quantization tolerance {tolerance}); a bound-kissing value \
                     is clamped, this one is not",
                    bounds.minimum, bounds.maximum
                )
                .into());
            }
        }
        minimum = minimum.min(*value);
        maximum = maximum.max(*value);
      }
    }
    let mut fill = gpuwm_preprocess_cpu::missing_value_chain::Counts::default();
    // The output window lies on the native integer lattice. Preserve the
    // original masked plane while each missing target exhausts the chain.
    for j in window.j_start..=window.j_end {
        for i in window.i_start..=window.i_end {
            if !values[j * nx + i].is_finite() {
                fill.masked += 1;
                let (value, stage) = gpuwm_preprocess_cpu::missing_value_chain::interpolate(
                    &values, nx, ny, i as f64, j as f64);
                match stage {
                    gpuwm_preprocess_cpu::missing_value_chain::Stage::NearestNeighbor => fill.nearest_neighbor += 1,
                    gpuwm_preprocess_cpu::missing_value_chain::Stage::FourPt => fill.four_pt += 1,
                    gpuwm_preprocess_cpu::missing_value_chain::Stage::Average4pt => fill.average_4pt += 1,
                    gpuwm_preprocess_cpu::missing_value_chain::Stage::Zero => fill.zero += 1,
                }
                let value = value as f32;
                minimum = minimum.min(value as f64);
                maximum = maximum.max(value as f64);
                writer.write_all(&value.to_le_bytes())?;
                continue;
            }
            let value = values[j * nx + i] as f32;
            if !value.is_finite() {
                return Err(format!("{context}: overflows FP32 in output window").into());
            }
            writer.write_all(&value.to_le_bytes())?;
        }
    }
    Ok(Stats {
        minimum,
        maximum,
        any_positive_in_source,
        count: window.nx() * window.ny(),
        clamped,
        fill,
    })
}

fn atmosphere_nonnegative(variable: &str) -> bool {
    HYBRID_SPECS
        .iter()
        .chain(OPTIONAL_HYBRID_SPECS.iter())
        .chain(EXTRA_SPECS.iter())
        .find(|spec| spec.name == variable)
        .map(|spec| spec.nonnegative)
        .or_else(|| {
            SURFACE_SPECS
                .iter()
                .chain(OPTIONAL_SURFACE_SPECS.iter())
                .chain(EXTRA_SURFACE_SPECS.iter())
                .find(|spec| spec.name == variable)
                .map(|spec| spec.nonnegative)
        })
        .unwrap_or(false)
}

/// The variables whose upper limit is a saturating physical one rather
/// than a plausibility ceiling: land and ice are areal fractions, and
/// volumetric soil moisture is a saturation fraction.  Water vapour at
/// 2 m is deliberately absent -- one kg/kg is an impossibility, not a
/// limit real cells sit on -- so it keeps its exact refusal.
fn unit_fraction(variable: &str) -> bool {
    matches!(variable, "LANDSEA" | "XICE" | "SOILW")
}

fn write_atmosphere(
    input: &str,
    inventory: &AtmosInventory,
    output: &Path,
    role: &str,
    window: Window,
    manifest: &mut BufWriter<File>,
) -> Result<(), Box<dyn Error>> {
    let file = Grib2File::open(input)?;
    fs::create_dir(output)?;
    // The analyzed aerosol number pair (written when the file publishes it)
    // and the --extras records a caller requested, each only when selected.
    let optional = OPTIONAL_HYBRID_SPECS.iter().chain(EXTRA_SPECS.iter()).copied().filter(|spec| {
        inventory.selected.iter().any(|field| field.variable == spec.name)
    });
    for spec in HYBRID_SPECS.into_iter().chain(optional) {
        let path = output.join(format!("{}.f32le", spec.name));
        let mut writer = BufWriter::new(File::create(&path)?);
        let needs_mask = OPTIONAL_HYBRID_NAMES.contains(&spec.name);
        let mut mask_writer = if needs_mask {
            Some(BufWriter::new(File::create(output.join(format!("{}.mask", spec.name)))?))
        } else { None };
        let mut any_positive = false;
        // Every level of this variable decodes at once (one record per
        // core); bytes and manifest rows are then written in level order.
        let fields: Vec<&SelectedField> = inventory
            .selected
            .iter()
            .filter(|field| field.variable == spec.name)
            .collect();
        type DecodedHybrid = Result<(Vec<u8>, Option<Vec<u8>>, Stats), String>;
        let decoded: Vec<DecodedHybrid> = fields
            .par_iter()
            .map(|selected| {
                let message = file
                    .messages
                    .get(selected.index)
                    .ok_or("selected atmosphere index disappeared on reopen")?;
                let (bytes, stats) = decode_crop(
                    message,
                    selected.index,
                    window,
                    spec.name,
                    spec.nonnegative,
                    unit_fraction(spec.name),
                )?;
                let mask = needs_mask.then(|| {
                    let nx = message.grid.nx as usize;
                    let mut mask_bytes = Vec::with_capacity(window.nx() * window.ny());
                    for j in window.j_start..=window.j_end {
                        for i in window.i_start..=window.i_end {
                            let present = message.bitmap.as_ref().map(|mask| mask[j * nx + i]).unwrap_or(true);
                            mask_bytes.push(u8::from(present));
                        }
                    }
                    mask_bytes
                });
                Ok((bytes, mask, stats))
            })
            .collect();
        for (selected, result) in fields.into_iter().zip(decoded) {
            let (bytes, mask, stats) = result?;
            let message = &file.messages[selected.index];
            writer.write_all(&bytes)?;
            drop(bytes);
            if let (Some(mask_writer), Some(mask)) = (&mut mask_writer, mask) {
                mask_writer.write_all(&mask)?;
            }
            any_positive |= stats.source_has_positive();
            writeln!(
                manifest,
                "{}",
                manifest_row(
                    role,
                    selected.index,
                    spec.name,
                    selected.level_value,
                    message.product.level_type,
                    message.data_rep.template,
                    message.bitmap.is_some(),
                    stats,
                    &path.file_name().unwrap().to_string_lossy(),
                )
            )?;
        }
        writer.flush()?;
        if let Some(mask_writer) = &mut mask_writer { mask_writer.flush()?; }
        if spec.require_any_positive && !any_positive {
            return Err(format!(
                "{role} {} is finite/nonnegative but zero on all 50 full-domain levels",
                spec.name
            )
            .into());
        }
    }
    // The surface records decode together, then are written in spec order.
    type DecodedSurface<'a> = Result<(&'a SelectedField, Vec<u8>, Stats), String>;
    let decoded: Vec<DecodedSurface<'_>> = SURFACE_SPECS
        .par_iter()
        .map(|spec| {
            let selected = inventory
                .selected
                .iter()
                .find(|field| field.variable == spec.name)
                .ok_or_else(|| format!("selected surface {} disappeared", spec.name))?;
            let message = file
                .messages
                .get(selected.index)
                .ok_or("selected surface index disappeared on reopen")?;
            let (bytes, stats) = decode_crop(
                message,
                selected.index,
                window,
                spec.name,
                atmosphere_nonnegative(spec.name),
                unit_fraction(spec.name),
            )?;
            Ok((selected, bytes, stats))
        })
        .collect();
    for (spec, result) in SURFACE_SPECS.iter().zip(decoded) {
        let (selected, bytes, stats) = result?;
        let message = &file.messages[selected.index];
        let path = output.join(format!("{}.f32le", spec.name));
        let mut writer = BufWriter::new(File::create(&path)?);
        writer.write_all(&bytes)?;
        drop(bytes);
        writer.flush()?;
        // Q2 keeps its post-hoc ceiling: one kg/kg of water vapour is a
        // plausibility limit, not a saturating one, so no quantization
        // argument applies to it.  LANDSEA and XICE are fractions and are
        // now held to one cell at a time, inside the decoder.
        if spec.name == "Q2" && stats.maximum > 1.0 {
            return Err(format!("{role} {} exceeds one: {}", spec.name, stats.maximum).into());
        }
        writeln!(
            manifest,
            "{}",
            manifest_row(
                role,
                selected.index,
                spec.name,
                selected.level_value,
                message.product.level_type,
                message.data_rep.template,
                message.bitmap.is_some(),
                stats,
                &path.file_name().unwrap().to_string_lossy(),
            )
        )?;
    }
    for spec in OPTIONAL_SURFACE_SPECS.iter().chain(EXTRA_SURFACE_SPECS.iter()) {
        let Some(selected) = inventory.selected.iter().find(|field| field.variable == spec.name) else {
            continue;
        };
        let message = file
            .messages
            .get(selected.index)
            .ok_or("selected optional surface index disappeared on reopen")?;
        let (bytes, stats) = decode_crop(
            message, selected.index, window, spec.name, spec.nonnegative, unit_fraction(spec.name))?;
        let path = output.join(format!("{}.f32le", spec.name));
        let mut writer = BufWriter::new(File::create(&path)?);
        writer.write_all(&bytes)?;
        writer.flush()?;
        writeln!(
            manifest,
            "{}",
            manifest_row(
                role,
                selected.index,
                spec.name,
                selected.level_value,
                message.product.level_type,
                message.data_rep.template,
                message.bitmap.is_some(),
                stats,
                &path.file_name().unwrap().to_string_lossy(),
            )
        )?;
    }
    Ok(())
}

fn write_soil(
    input: &str,
    inventory: &SoilInventory,
    output: &Path,
    role: &str,
    window: Window,
    manifest: &mut BufWriter<File>,
) -> Result<(), Box<dyn Error>> {
    let file = Grib2File::open(input)?;
    fs::create_dir(output)?;
    for variable in ["SOILT", "SOILW"] {
        let path = output.join(format!("{variable}.f32le"));
        let mut writer = BufWriter::new(File::create(&path)?);
        let fields: Vec<&SelectedField> = inventory
            .selected
            .iter()
            .filter(|field| field.variable == variable)
            .collect();
        let decoded: Vec<Result<(Vec<u8>, Stats), String>> = fields
            .par_iter()
            .map(|selected| {
                let message = file
                    .messages
                    .get(selected.index)
                    .ok_or("selected soil index disappeared on reopen")?;
                decode_crop(message, selected.index, window, variable, true, unit_fraction(variable))
            })
            .collect();
        for (selected, result) in fields.into_iter().zip(decoded) {
            let (bytes, stats) = result?;
            let message = &file.messages[selected.index];
            writer.write_all(&bytes)?;
            writeln!(
                manifest,
                "{}",
                manifest_row(
                    role,
                    selected.index,
                    variable,
                    selected.level_value,
                    message.product.level_type,
                    message.data_rep.template,
                    message.bitmap.is_some(),
                    stats,
                    &path.file_name().unwrap().to_string_lossy(),
                )
            )?;
        }
        writer.flush()?;
    }
    let decoded: Vec<Result<(Vec<u8>, Stats), String>> = inventory
        .optional_surface
        .par_iter()
        .map(|selected| {
            let message = file.messages.get(selected.index)
                .ok_or("selected optional surface index disappeared on reopen")?;
            decode_crop(message, selected.index, window, selected.variable, true, false)
        })
        .collect();
    for (selected, result) in inventory.optional_surface.iter().zip(decoded) {
        let (bytes, stats) = result?;
        let message = &file.messages[selected.index];
        let path = output.join(format!("{}.f32le", selected.variable));
        let mut writer = BufWriter::new(File::create(&path)?);
        writer.write_all(&bytes)?;
        if stats.maximum > 100.0 {
            return Err(format!("{} exceeds 100 percent", selected.variable).into());
        }
        writer.flush()?;
        writeln!(manifest, "{}", manifest_row(role, selected.index,
            selected.variable, selected.level_value, message.product.level_type,
            message.data_rep.template, message.bitmap.is_some(), stats,
            &path.file_name().unwrap().to_string_lossy()))?;
    }
    Ok(())
}

fn parse_usize(value: Option<String>, name: &str) -> Result<usize, Box<dyn Error>> {
    value
        .ok_or_else(|| format!("missing {name}"))?
        .parse::<usize>()
        .map_err(|error| format!("invalid {name}: {error}").into())
}

fn normalized_cycle(value: String) -> String {
    value.trim_end_matches('Z').replace('T', " ")
}

fn write_atomic_signal(path: &Path, lines: &[String]) -> Result<(), Box<dyn Error>> {
    let temporary = path.with_extension("tmp");
    let mut writer = BufWriter::new(File::create(&temporary)?);
    for line in lines {
        writeln!(writer, "{line}")?;
    }
    writer.flush()?;
    drop(writer);
    fs::rename(temporary, path)?;
    Ok(())
}

/// Replace `path` with `bytes` so a concurrent reader sees the old file or
/// the new one, never a prefix: a unique temp beside it, flushed and synced,
/// then renamed over it.  Windows refuses the rename while a reader holds
/// the target open without delete sharing (Python's `open`), so it is
/// retried for a few seconds there; a POSIX rename never waits.
fn replace_file_atomically(path: &Path, bytes: &[u8]) -> Result<(), Box<dyn Error>> {
    let name = path.file_name().ok_or("atomic replace target has no file name")?;
    let temporary = path.with_file_name(format!(
        ".{}.replace-{}",
        name.to_string_lossy(),
        std::process::id()
    ));
    {
        let mut file = File::create(&temporary)?;
        file.write_all(bytes)?;
        file.sync_all()?;
    }
    let mut attempts = 0u32;
    loop {
        match fs::rename(&temporary, path) {
            Ok(()) => return Ok(()),
            Err(error) if cfg!(windows) && attempts < 200
                && error.kind() == std::io::ErrorKind::PermissionDenied => {
                attempts += 1;
                std::thread::sleep(std::time::Duration::from_millis(20));
            }
            Err(error) => {
                let _ = fs::remove_file(&temporary);
                return Err(error.into());
            }
        }
    }
}

fn clone_tree_for_publish(source: &Path, destination: &Path) -> Result<(), Box<dyn Error>> {
    fs::create_dir(destination)?;
    for entry in fs::read_dir(source)? {
        let entry = entry?;
        let source_path = entry.path();
        let destination_path = destination.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            clone_tree_for_publish(&source_path, &destination_path)?;
        } else if fs::hard_link(&source_path, &destination_path).is_err() {
            fs::copy(&source_path, &destination_path)?;
        }
    }
    Ok(())
}

fn publish_completed_tree(
    partial: &Path,
    output: &Path,
    retain_staging: bool,
) -> Result<bool, Box<dyn Error>> {
    if output.exists() {
        return Err(format!("refusing existing canonical output: {output:?}").into());
    }
    if !retain_staging {
        // Legacy, --series, and --series-workers calls have no live consumer
        // holding paths into staging.  Transfer the completed tree directly
        // so those modes cannot leak a second full hidden tree.
        fs::rename(partial, output)?;
        return Ok(false);
    }
    // A live consumer opens many payloads after observing an hourly READY
    // receipt.  Renaming the shared staging root here creates a TOCTOU: the
    // consumer can open one field from staging, then lose the remaining path
    // when final publication moves the root.  Publish a closed same-filesystem
    // tree from hard links (copy fallback) and retain staging until the Python
    // consumer explicitly calls finish().  This is required on POSIX as well
    // as Windows; open file descriptors survive a rename, later path opens do
    // not.
    let parent = output.parent().unwrap_or_else(|| Path::new("."));
    let stem = output
        .file_name()
        .ok_or("OUTPUT_DIR has no filename")?
        .to_string_lossy();
    let publish = parent.join(format!(".{stem}.publish-{}", std::process::id()));
    if publish.exists() {
        return Err(format!("stale publish tree exists: {publish:?}").into());
    }
    if let Err(error) = clone_tree_for_publish(partial, &publish) {
        let _ = fs::remove_dir_all(&publish);
        return Err(error);
    }
    if let Err(error) = fs::rename(&publish, output) {
        let _ = fs::remove_dir_all(&publish);
        return Err(error.into());
    }
    Ok(true)
}

fn publish_completed_tree_owned(
    partial: &Path,
    output: &Path,
    retain_staging: bool,
) -> Result<bool, Box<dyn Error>> {
    match publish_completed_tree(partial, output, retain_staging) {
        Ok(retained) => Ok(retained),
        Err(error) => {
            // Streaming staging remains owned by the Python consumer until it
            // has observed producer termination and calls cancel()/finish().
            // Removing it here can race a consumer still opening a READY hour.
            if !retain_staging {
                let _ = fs::remove_dir_all(partial);
            }
            Err(error)
        }
    }
}

fn validate_series_hours(observed: &[u32]) -> Result<(), String> {
    if observed.len() < 2 || observed.len() > 49 {
        return Err(format!(
            "series manifest must contain 2..49 contiguous hourly source leads; observed={observed:?}"
        ));
    }
    let first = observed[0];
    let expected: Vec<u32> = (first..first + observed.len() as u32).collect();
    if observed != expected {
        return Err(format!(
            "series manifest source leads must be contiguous and ordered f{first:02}..f{:02}; observed={observed:?}",
            expected.last().unwrap()
        ));
    }
    if observed.last().copied().unwrap() > 48 {
        return Err(format!(
            "series manifest source lead exceeds the public f48 maximum; observed={observed:?}"
        ));
    }
    Ok(())
}

fn validate_series_cycle_horizon(observed: &[u32], cycle: &str) -> Result<(), String> {
    validate_series_hours(observed)?;
    if !cycle.is_ascii() || cycle.len() != 19 || &cycle[10..11] != " " || &cycle[13..19] != ":00:00"
    {
        return Err(format!(
            "expected cycle must be YYYY-MM-DD HH:00:00, got {cycle:?}"
        ));
    }
    let cycle_hour = cycle[11..13]
        .parse::<u32>()
        .map_err(|error| format!("invalid expected cycle hour: {error}"))?;
    let horizon = if matches!(cycle_hour, 0 | 6 | 12 | 18) {
        48
    } else {
        18
    };
    let last = observed.last().copied().unwrap();
    if last > horizon {
        return Err(format!(
            "source window ends at f{last:02}, beyond cycle {cycle_hour:02}Z horizon f{horizon:02}"
        ));
    }
    Ok(())
}

fn parse_series_manifest(path: &Path) -> Result<Vec<SeriesInput>, Box<dyn Error>> {
    parse_series_manifest_with(path, true)
}

/// The series as written.  `require_files` false is the as-posted mode: a
/// lead's GRIB is not on disk until the lead posts, the first lead's
/// included (the decoder starts before the window's first lead posts), and
/// each lead's files are opened only once the parent has admitted it
/// (`wait_admitted`).
fn parse_series_manifest_with(
    path: &Path,
    require_files: bool,
) -> Result<Vec<SeriesInput>, Box<dyn Error>> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let mut inputs = Vec::new();
    for (line_number, raw) in fs::read_to_string(path)?.lines().enumerate() {
        if raw.trim().is_empty() || raw.trim_start().starts_with('#') {
            continue;
        }
        let fields: Vec<&str> = raw.split('\t').collect();
        if fields.len() < 3 {
            return Err(format!(
                "series manifest line {} must be HOUR<TAB>WRFNAT<TAB>SOIL[<TAB>PMSL=GRIB...]",
                line_number + 1
            )
            .into());
        }
        let forecast_hour = fields[0].parse::<u32>().map_err(|error| {
            format!(
                "invalid series forecast hour on line {}: {error}",
                line_number + 1
            )
        })?;
        let resolve = |value: &str| {
            let candidate = PathBuf::from(value);
            if candidate.is_absolute() {
                candidate
            } else {
                parent.join(candidate)
            }
        };
        let atmosphere = resolve(fields[1]);
        let soil = resolve(fields[2]);
        if require_files && (!atmosphere.is_file() || !soil.is_file()) {
            return Err(format!(
                "series manifest line {} references a missing file: atmosphere={atmosphere:?}, soil={soil:?}",
                line_number + 1
            )
            .into());
        }
        let mut supplements = Vec::new();
        for binding in &fields[3..] {
            let value = binding.strip_prefix("PMSL=")
                .ok_or("native supplement must declare the supported canonical quantity PMSL=GRIB")?;
            let path = resolve(value);
            if value.is_empty() || !path.is_file() {
                return Err(format!("missing PMSL donor file {path:?}").into());
            }
            let path = path.to_string_lossy().into_owned();
            if supplements.contains(&path) {
                return Err(format!("duplicate PMSL donor path {path}").into());
            }
            supplements.push(path);
        }
        inputs.push(SeriesInput {
            forecast_hour,
            atmosphere: atmosphere.to_string_lossy().into_owned(),
            soil: soil.to_string_lossy().into_owned(),
            supplements,
        });
    }
    let observed: Vec<u32> = inputs.iter().map(|item| item.forecast_hour).collect();
    validate_series_hours(&observed)?;
    Ok(inputs)
}

/// Block until the parent has admitted `input`'s lead: its
/// `ADMIT_DIR/fNN.admitted` names the same forecast hour and the same two
/// files, written by the parent only after it held both files to the
/// lead's posted marker (`tools/hrrr_pipeline.py`).  A lead whose files
/// were never admitted is never opened.  Ends when another worker failed
/// (`cancelled`) or, on Unix, when the parent that admits leads is gone,
/// so an orphaned decoder does not wait for a lead nothing will admit.
fn wait_admitted(
    directory: &Path,
    input: &SeriesInput,
    cancelled: &AtomicBool,
) -> Result<(), Box<dyn Error>> {
    #[cfg(unix)]
    let parent = std::os::unix::process::parent_id();
    let path = directory.join(format!("f{:02}.admitted", input.forecast_hour));
    loop {
        if cancelled.load(Ordering::Acquire) {
            return Err(format!(
                "f{:02}: decode cancelled while waiting for its admission",
                input.forecast_hour
            )
            .into());
        }
        if path.is_file() {
            let text = fs::read_to_string(&path)?;
            let expected = [
                format!("forecast_hour\t{}", input.forecast_hour),
                format!("atmosphere\t{}", input.atmosphere),
                format!("soil\t{}", input.soil),
            ];
            let lines: Vec<&str> = text.lines().collect();
            if lines.len() != expected.len()
                || lines.iter().zip(&expected).any(|(line, want)| *line != want.as_str())
            {
                return Err(format!(
                    "f{:02}: admission {path:?} does not name this lead's series files",
                    input.forecast_hour
                )
                .into());
            }
            return Ok(());
        }
        #[cfg(unix)]
        if std::os::unix::process::parent_id() != parent {
            return Err(format!(
                "f{:02}: the parent that admits leads exited",
                input.forecast_hour
            )
            .into());
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    // Keep the release-provenance stamp in the binary: the cut
    // proves a staged bridge by these bytes (see lib.rs).
    let _ = std::hint::black_box(gpuwm_preprocess_cpu::SOURCE_REV_STAMP);
    let process_started = Instant::now();
    let mut args: Vec<String> = env::args().skip(1).collect();
    let mut extras = Vec::new();
    let mut surface_extras: Vec<SurfaceSpec> = Vec::new();
    // --analyzed-aerosol: the caller's configuration reads the analyzed
    // aerosol pair (gpuwm.preparation_assets.analyzed_aerosol_domains), so
    // QNWFA/QNIFA are selected when published.  Absent, they never are.
    let mut analyzed_aerosol = false;
    loop {
        match args.first().map(String::as_str) {
            Some("--extras") if extras.is_empty() && surface_extras.is_empty() => {
                if args.len() < 2 { return Err("--extras needs a comma-separated selector list to prevent an ambiguous inventory".into()); }
                (extras, surface_extras) = requested_extra_sets(&args[1])?;
                args.drain(..2);
            }
            Some("--analyzed-aerosol") if !analyzed_aerosol => {
                analyzed_aerosol = true;
                args.drain(..1);
            }
            _ => break,
        }
    }
    let usage = "usage: hrrr_grib2_bridge WRFNAT_F00 WRFNAT_F01 SOIL_F00 SOIL_F01 OUTPUT_DIR EXPECTED_CYCLE I_START I_END J_START J_END\n       hrrr_grib2_bridge --series SERIES_TSV OUTPUT_DIR EXPECTED_CYCLE I_START I_END J_START J_END\n       hrrr_grib2_bridge --series-workers WORKERS SERIES_TSV OUTPUT_DIR EXPECTED_CYCLE I_START I_END J_START J_END\n       hrrr_grib2_bridge --series-workers-ready WORKERS SERIES_TSV OUTPUT_DIR SIGNAL_DIR EXPECTED_CYCLE I_START I_END J_START J_END\n       hrrr_grib2_bridge --series-workers-posted WORKERS SERIES_TSV OUTPUT_DIR SIGNAL_DIR ADMIT_DIR EXPECTED_CYCLE I_START I_END J_START J_END\n       any form may be preceded by --analyzed-aerosol (select QNWFA/QNIFA when published) and --extras LIST";
    // As posted (A136 L7c): the first lead is inventoried up front and is
    // the reference; every later lead is inventoried, held to that
    // reference and decoded once the parent admits it (ADMIT_DIR).
    let mut admissions: Option<PathBuf> = None;
    let (inputs, output, expected_cycle, coordinate_start, decode_workers, signals) =
        if args.first().map(String::as_str) == Some("--series-workers-posted") {
            if args.len() != 11 {
                return Err(usage.into());
            }
            let workers = parse_usize(args.get(1).cloned(), "WORKERS")?;
            if workers == 0 || workers > 13 {
                return Err("WORKERS must be in 1..13".into());
            }
            admissions = Some(PathBuf::from(&args[5]));
            (
                parse_series_manifest_with(Path::new(&args[2]), false)?,
                PathBuf::from(&args[3]),
                normalized_cycle(args[6].clone()),
                7usize,
                workers,
                Some(PathBuf::from(&args[4])),
            )
        } else if args.first().map(String::as_str) == Some("--series-workers-ready") {
            if args.len() != 10 {
                return Err(usage.into());
            }
            let workers = parse_usize(args.get(1).cloned(), "WORKERS")?;
            if workers == 0 || workers > 13 {
                return Err("WORKERS must be in 1..13".into());
            }
            (
                parse_series_manifest(Path::new(&args[2]))?,
                PathBuf::from(&args[3]),
                normalized_cycle(args[5].clone()),
                6usize,
                workers,
                Some(PathBuf::from(&args[4])),
            )
        } else if args.first().map(String::as_str) == Some("--series-workers") {
            if args.len() != 9 {
                return Err(usage.into());
            }
            let workers = parse_usize(args.get(1).cloned(), "WORKERS")?;
            if workers == 0 || workers > 13 {
                return Err("WORKERS must be in 1..13".into());
            }
            (
                parse_series_manifest(Path::new(&args[2]))?,
                PathBuf::from(&args[3]),
                normalized_cycle(args[4].clone()),
                5usize,
                workers,
                None,
            )
        } else if args.first().map(String::as_str) == Some("--series") {
            if args.len() != 8 {
                return Err(usage.into());
            }
            (
                parse_series_manifest(Path::new(&args[1]))?,
                PathBuf::from(&args[2]),
                normalized_cycle(args[3].clone()),
                4usize,
                1usize,
                None,
            )
        } else {
            if args.len() != 10 {
                return Err(usage.into());
            }
            (
                vec![
                    SeriesInput {
                        forecast_hour: 0,
                        atmosphere: args[0].clone(),
                        soil: args[2].clone(),
                        supplements: Vec::new(),
                    },
                    SeriesInput {
                        forecast_hour: 1,
                        atmosphere: args[1].clone(),
                        soil: args[3].clone(),
                        supplements: Vec::new(),
                    },
                ],
                PathBuf::from(&args[4]),
                normalized_cycle(args[5].clone()),
                6usize,
                1usize,
                None,
            )
        };
    let observed: Vec<u32> = inputs.iter().map(|item| item.forecast_hour).collect();
    validate_series_cycle_horizon(&observed, &expected_cycle)?;
    let window = Window {
        i_start: parse_usize(args.get(coordinate_start).cloned(), "I_START")?,
        i_end: parse_usize(args.get(coordinate_start + 1).cloned(), "I_END")?,
        j_start: parse_usize(args.get(coordinate_start + 2).cloned(), "J_START")?,
        j_end: parse_usize(args.get(coordinate_start + 3).cloned(), "J_END")?,
    };
    if output.exists() {
        return Err(format!("refusing to overwrite existing output {output:?}").into());
    }
    if let Some(path) = &signals {
        if path.exists() {
            return Err(format!("refusing to overwrite existing signal directory {path:?}").into());
        }
    }

    // Inventory every file before creating output.  This is the primary
    // fail-closed gate: no partial bridge is published for mismatched times,
    // fields, levels, grids, duplicates, or packing support.  As posted only
    // the first lead is here: it is the reference, and each later lead is
    // held to it with the same comparisons before it is decoded, so the
    // gate's claims hold for every lead of a published bridge.
    if admissions.is_some() && inputs.iter().any(|input| !input.supplements.is_empty()) {
        // Breakage it prevents: a donor's selection and its coverage check
        // read every lead's inventory before any decode, which a lead not
        // posted yet does not have.
        return Err("--series-workers-posted decodes no PMSL donor; a series with donors is decoded whole (--series-workers-ready)".into());
    }
    let inventoried = if admissions.is_some() { 1 } else { inputs.len() };
    let never_cancelled = AtomicBool::new(false);
    if let Some(directory) = &admissions {
        wait_admitted(directory, &inputs[0], &never_cancelled)?;
    }
    let mut atmosphere_inventories = Vec::with_capacity(inventoried);
    let mut soil_inventories = Vec::with_capacity(inventoried);
    // Every lead's two files are read and inventoried at once; the results
    // are taken in lead order, so the first failure reported is the one the
    // serial loop reported.
    let inventories: Vec<(Result<AtmosInventory, String>, Result<SoilInventory, String>)> =
        inputs[..inventoried]
            .par_iter()
            .map(|input| {
                rayon::join(
                    || {
                        inventory_atmosphere(
                            &input.atmosphere,
                            &expected_cycle,
                            input.forecast_hour, &extras, &surface_extras, analyzed_aerosol,
                        )
                        .map_err(|error| error.to_string())
                    },
                    || {
                        inventory_soil(&input.soil, &expected_cycle, input.forecast_hour)
                            .map_err(|error| error.to_string())
                    },
                )
            })
            .collect();
    for (atmosphere, soil) in inventories {
        atmosphere_inventories.push(atmosphere?);
        soil_inventories.push(soil?);
    }
    let atmosphere_reference = &atmosphere_inventories[0];
    let soil_reference = &soil_inventories[0];
    for index in 0..inventoried {
        let hour = inputs[index].forecast_hour;
        compare_atmosphere_inventory(atmosphere_reference, &atmosphere_inventories[index], hour)?;
        compare_soil_inventory(soil_reference, &soil_inventories[index], hour)?;
        if atmosphere_inventories[index].grid != soil_inventories[index].grid {
            return Err(format!(
                "soil f{hour:02} grid does not exactly equal atmosphere f{hour:02}"
            )
            .into());
        }
    }
    validate_window(window, &atmosphere_reference.grid)?;

    // Inventory each explicitly declared donor once, then retain only its
    // selected compressed messages. No reopen can change a validated donor
    // between inventory and decode; unrelated messages are released here.
    let mut donor_files = BTreeMap::new();
    for input in &inputs {
        for path in &input.supplements {
            if !donor_files.contains_key(path) {
                let file = Grib2File::open(path)?;
                // Drop unrelated fields before opening the next donor file.
                // Keep original message indices for the selection receipt.
                let selected: Vec<_> = file.messages.into_iter().enumerate()
                    .filter(|(_, message)| is_pmsl_pa(message)).collect();
                donor_files.insert(path.clone(), selected);
            }
        }
    }
    let mut donors = Vec::with_capacity(inputs.len());
    let mut donor_rows = Vec::new();
    if admissions.is_some() {
        // Refused above: a posted series names no donor.
        donors.resize(inputs.len(), None);
    }
    for (input, atmosphere) in inputs.iter().zip(&atmosphere_inventories) {
        if admissions.is_some() {
            break;
        }
        let message = if input.supplements.is_empty() {
            None
        } else {
            let messages = input.supplements.iter()
                .flat_map(|path| donor_files[path].iter().map(|(_, message)| message));
            let selected = select_pmsl(messages, &expected_cycle,
                input.forecast_hour, &atmosphere.source_grid)?;
            let (path, message_index) = input.supplements.iter().find_map(|path| {
                donor_files[path].iter().find_map(|(index, message)|
                    std::ptr::eq(message, selected).then_some((path, *index)))
            }).ok_or("selected donor origin disappeared")?;
            donor_rows.push(format!("{}\tPMSL\t{}\t{}\t{}\t{}\t{}/{}/{}\t{}\tPa\t{:?}",
                input.forecast_hour, path, message_index, selected.reference_time,
                selected.identification.center_id, selected.discipline,
                selected.product.parameter_category, selected.product.parameter_number,
                selected.product.level_type, selected.grid));
            donor_rows.last_mut().unwrap().push_str(&format!("\t{}\t{:?}\t{:?}\t{:?}",
                selected.product.template, selected.product.ensemble_type,
                selected.product.perturbation_number, selected.product.num_forecasts_in_ensemble));
            validate_crop_payload(selected, message_index, "PMSL", window)?;
            Some((message_index, selected.clone()))
        };
        donors.push(message);
    }
    drop(donor_files);
    let supplemented = donors.iter().filter(|item| item.is_some()).count();
    if supplemented != 0 && supplemented != inputs.len() {
        return Err("PMSL donor coverage must include every requested forcing time".into());
    }
    let optional_hybrid = optional_hybrid_fields(atmosphere_reference);
    let payload_files = 24 + extras.len() + surface_extras.len() + usize::from(supplemented != 0)
        + soil_reference.optional_surface.len()
        + optional_hybrid.len()
        + optional_surface_fields(atmosphere_reference).len();

    let parent = output.parent().unwrap_or_else(|| Path::new("."));
    let stem = output
        .file_name()
        .ok_or("OUTPUT_DIR has no filename")?
        .to_string_lossy();
    let partial = parent.join(format!(".{stem}.partial-{}", std::process::id()));
    let retain_staging = signals.is_some();
    if partial.exists() {
        return Err(format!("stale partial output already exists: {partial:?}").into());
    }
    fs::create_dir(&partial)?;
    let result = (|| -> Result<(), Box<dyn Error>> {
        let mut gate = BufWriter::new(File::create(partial.join("gate.txt"))?);
        writeln!(gate, "status\tPASS")?;
        writeln!(gate, "aerosol_missing_policy\tnearest_neighbor+four_pt+average_4pt;fill_missing=0;native_grid_coordinates;source_mask_preserved_for_target_mapping")?;
        writeln!(gate, "cycle\t{expected_cycle}")?;
        let valid_times = inputs
            .iter()
            .map(|input| format!("{expected_cycle}+{:02}h", input.forecast_hour))
            .collect::<Vec<_>>()
            .join(",");
        let forecast_hours = inputs
            .iter()
            .map(|input| input.forecast_hour.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let model_forcing_hours = (0..inputs.len())
            .map(|hour| hour.to_string())
            .collect::<Vec<_>>()
            .join(",");
        writeln!(gate, "valid_times\t{valid_times}")?;
        writeln!(gate, "forecast_hours\t{forecast_hours}")?;
        writeln!(gate, "source_forecast_hours\t{forecast_hours}")?;
        writeln!(gate, "model_forcing_hours\t{model_forcing_hours}")?;
        writeln!(gate, "series_count\t{}", inputs.len())?;
        writeln!(
            gate,
            "atmosphere_selected_per_time\t{}",
            atmosphere_reference.selected.len()
        )?;
        if supplemented != 0 {
            writeln!(gate, "supplement_fields\tPMSL")?;
            writeln!(gate, "supplement_alignment\texact_primary_grid_and_source_time")?;
            writeln!(gate, "supplement_units\tPMSL=Pa")?;
        }
        if !extras.is_empty() || !surface_extras.is_empty() {
            let names = extras.iter().map(|spec| spec.name).chain(surface_extras.iter().map(|spec| spec.name));
            writeln!(gate, "extra_fields\t{}", names.collect::<Vec<_>>().join(","))?;
            for spec in &extras {
                writeln!(gate, "extra_{}\tPASS discipline={} category={} parameter={} level_type=105", spec.name, spec.parameter.discipline, spec.parameter.category, spec.parameter.number)?;
            }
            for spec in &surface_extras {
                let unit = EXTRA_SURFACE_SPECS.iter().position(|item| item.name == spec.name)
                    .map(|index| EXTRA_SURFACE_UNITS[index]).unwrap_or("");
                writeln!(gate, "extra_{}\tPASS discipline={} category={} parameter={} level_type={} level_value={} units={unit}", spec.name, spec.parameter.discipline, spec.parameter.category, spec.parameter.number, spec.level_type, spec.level_value)?;
            }
        }
        writeln!(gate, "hybrid_levels\t{N_HYBRID_LEVELS}")?;
        for line in optional_hybrid_gate_lines(atmosphere_reference) {
            writeln!(gate, "{line}")?;
        }
        for line in optional_surface_gate_lines(atmosphere_reference) {
            writeln!(gate, "{line}")?;
        }
        if !soil_reference.optional_surface.is_empty() {
            let names = soil_reference.optional_surface.iter().map(|field| field.variable)
                .collect::<Vec<_>>().join(",");
            writeln!(gate, "optional_soil_surface_fields\t{names}")?;
            writeln!(gate, "optional_soil_surface_units\tVEGFRA=percent")?;
        }
        writeln!(
            gate,
            "soil_selected_per_time\t{}",
            soil_reference.selected.len()
        )?;
        writeln!(
            gate,
            "grid\tHRRR Lambert GDT3.30 1799x1059 3000m shape6 scan0x40 uv-grid-relative"
        )?;
        writeln!(
            gate,
            "window_zero_based_inclusive\ti={}..{} j={}..{}",
            window.i_start, window.i_end, window.j_start, window.j_end
        )?;
        writeln!(gate, "window_shape\t{}x{}", window.ny(), window.nx())?;
        writeln!(
            gate,
            "output\tFP32 little-endian south-to-north row-major; 3-D level-major"
        )?;
        writeln!(gate, "{}", qice_mapping_line(atmosphere_reference)?)?;
        writeln!(
            gate,
            "cross_time_inventory\tPASS exact selected keys/levels/grid; packing may differ"
        )?;
        gate.flush()?;
        if supplemented != 0 {
            let mut receipt = BufWriter::new(File::create(partial.join("supplement-inventory.tsv"))?);
            writeln!(receipt, "forecast_hour\tfield\tpath\tmessage_index\tcycle\tcenter\tparameter\tlevel_type\tunits\tgrid\tpdt\tensemble_type\tmember\tensemble_size")?;
            for row in &donor_rows { writeln!(receipt, "{row}")?; }
            receipt.flush()?;
        }

        if let Some(path) = &signals {
            fs::create_dir(path)?;
            write_atomic_signal(
                &path.join("preflight.ready"),
                &[
                    "status\tPASS".to_owned(),
                    format!("staging_root\t{}", partial.display()),
                    "staging_retention\tuntil_consumer_finish".to_owned(),
                    format!("canonical_output\t{}", output.display()),
                    format!("workers\t{decode_workers}"),
                    format!("series_count\t{}", inputs.len()),
                    format!("cycle\t{expected_cycle}"),
                    format!("window_shape\t{}x{}", window.ny(), window.nx()),
                    if admissions.is_some() {
                        format!(
                            "inventory\tPASS f{:02} reference; f{:02}..f{:02} each held to it when admitted",
                            inputs.first().unwrap().forecast_hour,
                            inputs.first().unwrap().forecast_hour,
                            inputs.last().unwrap().forecast_hour
                        )
                    } else {
                        format!(
                            "inventory\tPASS all f{:02}..f{:02} before decode",
                            inputs.first().unwrap().forecast_hour,
                            inputs.last().unwrap().forecast_hour
                        )
                    },
                    format!(
                        "producer_elapsed_seconds\t{:.9}",
                        process_started.elapsed().as_secs_f64()
                    ),
                ],
            )?;
        }

        let next = AtomicUsize::new(0);
        let cancelled = AtomicBool::new(false);
        let errors = Mutex::new(Vec::<String>::new());
        std::thread::scope(|scope| {
            for _ in 0..decode_workers.min(inputs.len()) {
                scope.spawn(|| loop {
                    if cancelled.load(Ordering::Acquire) {
                        break;
                    }
                    let index = next.fetch_add(1, Ordering::Relaxed);
                    if index >= inputs.len() {
                        break;
                    }
                    let input = &inputs[index];
                    let hour = input.forecast_hour;
                    let atmosphere_role = format!("atmosphere-f{hour:02}");
                    let soil_role = format!("soil-f{hour:02}");
                    let fragment_path = partial.join(format!(".inventory-f{hour:02}.tsv"));
                    let decoded = (|| -> Result<(), Box<dyn Error>> {
                        // As posted, a later lead is read only once
                        // admitted, then held to the reference exactly as
                        // the up-front inventory holds every lead.
                        let admitted: (AtmosInventory, SoilInventory);
                        let (atmosphere_inventory, soil_inventory) =
                            if index < atmosphere_inventories.len() {
                                (&atmosphere_inventories[index], &soil_inventories[index])
                            } else {
                                let directory = admissions
                                    .as_ref()
                                    .ok_or("a lead was not inventoried before decode")?;
                                wait_admitted(directory, input, &cancelled)?;
                                let atmosphere = inventory_atmosphere(
                                        &input.atmosphere, &expected_cycle, hour, &extras,
                                        &surface_extras, analyzed_aerosol)?;
                                let soil = inventory_soil(&input.soil, &expected_cycle, hour)?;
                                compare_atmosphere_inventory(atmosphere_reference, &atmosphere, hour)?;
                                compare_soil_inventory(soil_reference, &soil, hour)?;
                                if atmosphere.grid != soil.grid {
                                    return Err(format!(
                                        "soil f{hour:02} grid does not exactly equal atmosphere f{hour:02}"
                                    )
                                    .into());
                                }
                                admitted = (atmosphere, soil);
                                (&admitted.0, &admitted.1)
                            };
                        let mut fragment = BufWriter::new(File::create(&fragment_path)?);
                        write_atmosphere(
                            &input.atmosphere,
                            atmosphere_inventory,
                            &partial.join(&atmosphere_role),
                            &atmosphere_role,
                            window,
                            &mut fragment,
                        )?;
                        write_soil(
                            &input.soil,
                            soil_inventory,
                            &partial.join(&soil_role),
                            &soil_role,
                            window,
                            &mut fragment,
                        )?;
                        if let Some((message_index, message)) = &donors[index] {
                            let path = partial.join(&atmosphere_role).join("PMSL.f32le");
                            let mut writer = BufWriter::new(File::create(&path)?);
                            let stats = decode_crop_write(message, *message_index, &mut writer, window,
                                "PMSL", true, false)?;
                            writer.flush()?;
                            writeln!(fragment, "{}", manifest_row(&atmosphere_role, *message_index,
                                "PMSL", message.product.level_value, message.product.level_type,
                                message.data_rep.template, message.bitmap.is_some(), stats,
                                "PMSL.f32le"))?;
                        }
                        fragment.flush()?;
                        drop(fragment);
                        if let Some(path) = &signals {
                            write_atomic_signal(
                                &path.join(format!("f{hour:02}.ready")),
                                &[
                                    "status\tPASS".to_owned(),
                                    format!("forecast_hour\t{hour}"),
                                    format!("atmosphere_dir\t{atmosphere_role}"),
                                    format!("soil_dir\t{soil_role}"),
                                    format!("inventory_fragment\t{}", fragment_path.display()),
                                    format!("payload_files\t{payload_files}"),
                                    format!(
                                        "producer_elapsed_seconds\t{:.9}",
                                        process_started.elapsed().as_secs_f64()
                                    ),
                                ],
                            )?;
                        }
                        Ok(())
                    })();
                    if let Err(error) = decoded {
                        cancelled.store(true, Ordering::Release);
                        errors.lock().unwrap().push(format!("f{hour:02}: {error}"));
                        break;
                    }
                });
            }
        });
        let errors = errors.into_inner().unwrap();
        if !errors.is_empty() {
            return Err(format!("parallel decode failed: {errors:?}").into());
        }
        let mut manifest = BufWriter::new(File::create(partial.join("inventory.tsv"))?);
        writeln!(manifest, "{MANIFEST_HEADER}")?;
        for input in &inputs {
            let fragment_path = partial.join(format!(".inventory-f{:02}.tsv", input.forecast_hour));
            let mut fragment = File::open(&fragment_path)?;
            std::io::copy(&mut fragment, &mut manifest)?;
            fs::remove_file(fragment_path)?;
        }
        manifest.flush()?;
        drop(manifest);
        // Counters are reduced in lead order, after every decoder finished.
        // Early streaming consumers see the policy; the completed gate and
        // its sealed hash carry the final counts for all leads.
        //
        // The completed gate replaces the early one whole (temp + rename),
        // never by appending in place.  A streaming consumer re-reads
        // gate.txt for every lead while this runs; the unbuffered append
        // issued one write per format piece, so a reader could see
        // "aerosol_QNWFA_" with no tab yet and refuse "malformed gate.txt
        // line 23/25" (36-member HRRR ensemble, members 13 and 33,
        // 2026-10-24).  A reader now sees the early gate or the final one.
        drop(gate);
        let gate_path = partial.join("gate.txt");
        let mut completed = fs::read_to_string(&gate_path)?;
        let text = fs::read_to_string(partial.join("inventory.tsv"))?;
        for variable in OPTIONAL_HYBRID_NAMES {
            let mut total = gpuwm_preprocess_cpu::missing_value_chain::Counts::default();
            for line in text.lines().skip(1) {
                let columns: Vec<_> = line.split('\t').collect();
                if columns[2] != variable { continue; }
                total.add(gpuwm_preprocess_cpu::missing_value_chain::Counts {
                    masked: columns[13].parse()?, nearest_neighbor: columns[14].parse()?,
                    four_pt: columns[15].parse()?, average_4pt: columns[16].parse()?, zero: columns[17].parse()?,
                });
            }
            for (stage, count) in [("masked", total.masked), ("nearest_neighbor", total.nearest_neighbor),
                ("four_pt", total.four_pt), ("average_4pt", total.average_4pt), ("zero", total.zero)] {
                completed.push_str(&format!("aerosol_{variable}_{stage}\t{count}\n"));
            }
        }
        replace_file_atomically(&gate_path, completed.as_bytes())?;
        Ok(())
    })();
    if let Err(error) = result {
        if !retain_staging {
            let _ = fs::remove_dir_all(&partial);
        }
        if let Some(path) = &signals {
            if path.is_dir() {
                let _ = write_atomic_signal(
                    &path.join("failure.ready"),
                    &[
                        "status\tFAIL".to_owned(),
                        format!("error\t{error}"),
                        format!("staging_root\t{}", partial.display()),
                        "staging_retained\ttrue".to_owned(),
                        format!(
                            "producer_elapsed_seconds\t{:.9}",
                            process_started.elapsed().as_secs_f64()
                        ),
                    ],
                );
            }
        }
        return Err(error);
    }
    let staging_retained = match publish_completed_tree_owned(&partial, &output, retain_staging) {
        Ok(retained) => retained,
        Err(error) => {
            if let Some(path) = &signals {
                if path.is_dir() {
                    let _ = write_atomic_signal(
                        &path.join("failure.ready"),
                        &[
                            "status\tFAIL".to_owned(),
                            format!("error\t{error}"),
                            format!("staging_root\t{}", partial.display()),
                            "staging_retained\ttrue".to_owned(),
                            format!(
                                "producer_elapsed_seconds\t{:.9}",
                                process_started.elapsed().as_secs_f64()
                            ),
                        ],
                    );
                }
            }
            return Err(error);
        }
    };
    if let Some(path) = &signals {
        write_atomic_signal(
            &path.join("complete.ready"),
            &[
                "status\tPASS".to_owned(),
                format!("canonical_output\t{}", output.display()),
                format!("series_count\t{}", inputs.len()),
                format!("staging_retained\t{staging_retained}"),
                format!(
                    "producer_elapsed_seconds\t{:.9}",
                    process_started.elapsed().as_secs_f64()
                ),
            ],
        )?;
    }
    println!("PASS\t{}", output.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn dry_crop_keeps_positive_source_guard_field_wide() {
        let mut record = bitmap_record(None);
        record.raw_data = vec![0, 0, 5, 0, 0, 0, 0, 0, 0];
        let window = Window { i_start: 0, i_end: 1, j_start: 0, j_end: 0 };
        let (bytes, stats) = decode_crop(&record, 0, window, "QG", true, false).unwrap();
        assert_eq!(bytes, vec![0u8; 8]);
        assert_eq!(stats.maximum, 0.0);
        assert!(stats.source_has_positive());
        record.raw_data.fill(0);
        let (_, stats) = decode_crop(&record, 0, window, "QG", true, false).unwrap();
        assert!(!stats.source_has_positive());
    }

    #[test]
    #[ignore = "requires the retained public archive message"]
    fn archived_graupel_dry_crop_has_positive_source() {
        let path = env::var("HRRR_GRAUPEL_MESSAGE").expect("HRRR_GRAUPEL_MESSAGE");
        let file = Grib2File::open(&path).unwrap();
        let message = &file.messages[0];
        let northeast = Window { i_start: 1314, i_end: 1792, j_start: 557, j_end: 1035 };
        let (bytes, stats) = decode_crop(message, 0, northeast, "QG", true, false).unwrap();
        assert!(bytes.iter().all(|v| *v == 0));
        assert_eq!(stats.maximum, 0.0);
        assert!(stats.source_has_positive());
    }

    #[test]
    fn crop_bounds_excludes_unread_cells_and_keeps_included_refusal() {
        let mut record = bitmap_record(None);
        record.data_rep.decimal_scale = 2;
        record.raw_data[0] = 101; // 1.01, outside physical unit-fraction bounds.
        let window = Window { i_start: 1, i_end: 2, j_start: 0, j_end: 2 };
        let (bytes, stats) = decode_crop(&record, 0, window, "SOILW", true, true).unwrap();
        let expected: Vec<u8> = [0.02f32, 0.03, 0.05, 0.06, 0.08, 0.09]
            .iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(bytes, expected);
        assert_eq!(stats.count, 6);
        assert_eq!(stats.maximum, 0.09);
        let includes_bad = Window { i_start: 0, i_end: 1, j_start: 0, j_end: 1 };
        let error = decode_crop(&record, 0, includes_bad, "SOILW", true, true).unwrap_err();
        assert!(error.contains("SOILW value 1.01 outside [0,1]"), "{error}");
    }

    #[test]
    fn dry_crop_preserves_field_anchored_quantization_clamp() {
        let mut record = bitmap_record(None);
        record.data_rep.bits_per_value = 16;
        record.data_rep.reference_value = -1.0;
        record.data_rep.decimal_scale = 6;
        record.raw_data = [0u16, 2, 65535, 2, 2, 2, 2, 2, 2]
            .iter().flat_map(|value| value.to_be_bytes()).collect();
        let window = Window { i_start: 0, i_end: 1, j_start: 0, j_end: 0 };
        let (bytes, stats) = decode_crop(&record, 0, window, "QCLOUD", true, false).unwrap();
        let expected: Vec<u8> = [0.0f32, 1.0e-6]
            .iter().flat_map(|value| value.to_le_bytes()).collect();
        assert_eq!(bytes, expected);
        assert_eq!(stats.clamped.clamps, 1);
        assert_eq!(stats.count, 2);
        // The dry crop alone would narrow the existing quantum tolerance.
        assert!(matches!(value_bounds(true, false, 1.0e-6).check(-1.0e-6, 1.0e-6),
            BoundVerdict::Refuse { .. }));
    }

    /// Real anonymous HRRR soil message, retained separately with URL and SHA256.
    /// The native Northeast source window already includes the remapper halo.
    /// Run with HRRR_SOIL_MESSAGE=/path/to/message.grib2 cargo test ... --ignored.
    #[test]
    #[ignore = "requires the retained public archive message"]
    fn archived_crop_bounds_regression() {
        let path = env::var("HRRR_SOIL_MESSAGE").expect("HRRR_SOIL_MESSAGE");
        let file = Grib2File::open(&path).unwrap();
        assert_eq!(file.messages.len(), 1);
        let message = &file.messages[0];
        let values = unpack_message(message).unwrap();
        assert_eq!(values[921 * message.grid.nx as usize + 558], 1.01);
        let northeast = Window { i_start: 1314, i_end: 1792, j_start: 557, j_end: 1035 };
        let (bytes, stats) = decode_crop(message, 0, northeast, "SOILW", true, true).unwrap();
        assert_eq!(bytes.len(), 479 * 479 * 4);
        assert_eq!(stats.count, 479 * 479);
        assert!(stats.minimum >= 0.0 && stats.maximum <= 1.0);
        let reservoir = Window { i_start: 557, i_end: 559, j_start: 920, j_end: 922 };
        let error = decode_crop(message, 0, reservoir, "SOILW", true, true).unwrap_err();
        assert!(error.contains("SOILW value 1.01 outside [0,1]"), "{error}");
        println!("Northeast {} cells PASS [{},{}]; included source cell REFUSED: {}",
            stats.count, stats.minimum, stats.maximum, error);
    }

    #[test]
    #[ignore = "requires the retained public archive roundoff message"]
    fn archived_one_quantum_soil_roundoff_clamps() {
        let path = env::var("HRRR_SOIL_ROUNDOFF_MESSAGE").expect("HRRR_SOIL_ROUNDOFF_MESSAGE");
        let file = Grib2File::open(&path).unwrap();
        assert_eq!(file.messages.len(), 1);
        let message = &file.messages[0];
        let values = unpack_message(message).unwrap();
        assert_eq!(values[998 * message.grid.nx as usize + 1332], 1.0010000000000001);
        let northeast = Window { i_start: 1314, i_end: 1792, j_start: 557, j_end: 1035 };
        let (bytes, stats) = decode_crop(message, 0, northeast, "SOILW", true, true).unwrap();
        assert_eq!(stats.clamped.clamps, 1);
        assert_eq!(stats.maximum, 1.0);
        let index = ((998 - 557) * 479 + (1332 - 1314)) * 4;
        assert_eq!(f32::from_le_bytes(bytes[index..index + 4].try_into().unwrap()), 1.0);
        assert!(matches!(value_bounds(true, true, 1.002).check(1.002, 0.001),
            BoundVerdict::Refuse { .. }));
        println!("Archived one-quantum SOILW cell clamps to 1; two quanta still refused");
    }

    fn bitmap_record(mask: Option<Vec<bool>>) -> Grib2Message {
        let mut record = canonical_record(PMTF, HYBRID_LEVEL_TYPE, 1.0, 0);
        record.grid.nx = 3;
        record.grid.ny = 3;
        record.data_rep.template = 0;
        record.data_rep.bits_per_value = 8;
        record.data_rep.section5_num_data_points = mask.as_ref()
            .map(|m| m.iter().filter(|p| **p).count() as u32).unwrap_or(9);
        record.raw_data = (1u8..=9).enumerate()
            .filter(|(i, _)| mask.as_ref().map(|m| m[*i]).unwrap_or(true))
            .map(|(_, v)| v).collect();
        record.bitmap = mask;
        record
    }

    fn bitmap_crop(record: &Grib2Message, variable: &str, window: Window)
        -> Result<(Vec<u8>, Stats), String> {
        let path = std::env::temp_dir().join(format!("bitmap-check-{}-{}.f32le",
            std::process::id(), SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos()));
        let mut writer = BufWriter::new(File::create(&path).unwrap());
        let result = decode_crop_write(record, 17, &mut writer, window, variable, true, false);
        writer.flush().unwrap();
        drop(writer);
        let bytes = fs::read(&path).unwrap();
        fs::remove_file(&path).unwrap();
        result.map(|stats| (bytes, stats)).map_err(|e| e.to_string())
    }

    #[test]
    fn bitmap_check_outside_crop_preserves_values() {
        let window = Window { i_start: 1, i_end: 2, j_start: 0, j_end: 2 };
        let masked = bitmap_record(Some(vec![false, true, true, true, true, true, true, true, true]));
        validate_payload(&masked).unwrap();
        assert_eq!(bitmap_crop(&masked, "TT", window).unwrap().0,
            bitmap_crop(&bitmap_record(None), "TT", window).unwrap().0);
    }

    #[test]
    fn bitmap_check_all_present_is_byte_identical() {
        let window = Window { i_start: 0, i_end: 2, j_start: 0, j_end: 2 };
        assert_eq!(bitmap_crop(&bitmap_record(Some(vec![true; 9])), "TT", window).unwrap().0,
            bitmap_crop(&bitmap_record(None), "TT", window).unwrap().0);
    }

    #[test]
    fn bitmap_check_named_non_aerosol_refusal() {
        let window = Window { i_start: 0, i_end: 2, j_start: 0, j_end: 2 };
        let masked = bitmap_record(Some(vec![false, true, true, true, true, true, true, true, true]));
        let error = bitmap_crop(&masked, "TT", window).unwrap_err();
        for required in ["TT", "level 1", "message", "i=0..2", "j=0..2", "masked_inside_crop=1"] {
            assert!(error.contains(required), "missing {required}: {error}");
        }
    }

    #[test]
    fn bitmap_check_aerosol_donor_chain_and_counts() {
        let window = Window { i_start: 0, i_end: 2, j_start: 0, j_end: 2 };
        let masked = bitmap_record(Some(vec![false, true, true, true, true, true, true, true, false]));
        let (bytes, stats) = bitmap_crop(&masked, "QNWFA", window).unwrap();
        let expected: Vec<u8> = [0.0f32, 2., 3., 4., 5., 6., 7., 8., 0.]
            .iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(bytes, expected);
        let counts = format!("{stats:?}");
        for required in ["masked: 2", "nearest_neighbor: 0", "four_pt: 0", "average_4pt: 0", "zero: 2"] {
            assert!(counts.contains(required), "missing {required}: {counts}");
        }
    }

    #[test]
    fn bitmap_check_shape_and_packed_count() {
        let mut record = bitmap_record(Some(vec![true; 9]));
        record.data_rep.section5_num_data_points = 8;
        assert!(validate_payload(&record).unwrap_err().to_string().contains("expected 9 present cells"));
        record.bitmap = Some(vec![true; 8]);
        assert!(validate_payload(&record).unwrap_err().to_string().contains("bitmap has 8 cells, expected 9"));
        let window = Window { i_start: 0, i_end: 2, j_start: 0, j_end: 2 };
        let error = bitmap_crop(&record, "TT", window).unwrap_err();
        for required in ["TT", "level 1", "message 17", "i=0..2", "j=0..2", "masked_inside_crop=0", "bitmap has 8 cells"] {
            assert!(error.contains(required), "{error}");
        }
    }
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn publication_root(prefix: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("{prefix}-{}-{unique}", std::process::id()))
    }

    #[test]
    fn completed_publication_retains_staging_and_canonical_survives_cleanup() {
        let root = publication_root("gpuwm-hrrr-publish-test");
        let partial = root.join(format!(".native-bridge.partial-{}", std::process::id()));
        let nested = partial.join("atmosphere-f00");
        let output = root.join("native-bridge");
        fs::create_dir_all(&nested).unwrap();
        fs::write(nested.join("TT.f32le"), b"stable-payload").unwrap();

        let retained = publish_completed_tree(&partial, &output, true).unwrap();

        assert!(retained);
        assert!(partial.is_dir());
        assert!(output.is_dir());
        assert_eq!(
            fs::read(partial.join("atmosphere-f00/TT.f32le")).unwrap(),
            b"stable-payload"
        );
        assert_eq!(
            fs::read(output.join("atmosphere-f00/TT.f32le")).unwrap(),
            b"stable-payload"
        );
        fs::remove_dir_all(&partial).unwrap();
        assert_eq!(
            fs::read(output.join("atmosphere-f00/TT.f32le")).unwrap(),
            b"stable-payload"
        );
        fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn non_streaming_publication_transfers_tree_without_hidden_leak() {
        let root = publication_root("gpuwm-hrrr-direct-publish-test");
        let partial = root.join(format!(".native-bridge.partial-{}", std::process::id()));
        let output = root.join("native-bridge");
        fs::create_dir_all(partial.join("atmosphere-f00")).unwrap();
        fs::write(partial.join("atmosphere-f00/TT.f32le"), b"direct").unwrap();

        let retained = publish_completed_tree_owned(&partial, &output, false).unwrap();

        assert!(!retained);
        assert!(!partial.exists());
        assert_eq!(
            fs::read(output.join("atmosphere-f00/TT.f32le")).unwrap(),
            b"direct"
        );
        fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn publication_failure_cleans_direct_but_retains_streaming_owner_tree() {
        let root = publication_root("gpuwm-hrrr-publish-failure-test");
        let output = root.join("native-bridge");
        fs::create_dir_all(&output).unwrap();

        let direct = root.join(".native-bridge.partial-111111");
        fs::create_dir_all(&direct).unwrap();
        assert!(publish_completed_tree_owned(&direct, &output, false).is_err());
        assert!(!direct.exists());

        let streaming = root.join(".native-bridge.partial-222222");
        fs::create_dir_all(&streaming).unwrap();
        assert!(publish_completed_tree_owned(&streaming, &output, true).is_err());
        assert!(streaming.is_dir());

        fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn qice_reads_cimixr_first_and_cice_only_as_its_earlier_code() {
        let qi = HYBRID_SPECS.iter().find(|spec| spec.name == "QI").unwrap();
        assert_eq!(qi.parameter, CIMIXR);
        assert_eq!(qi.earlier_parameters, &[CICE]);
        assert!(qi.require_any_positive);
        // 0/6/0 is cloud ice and nothing else in this table: no other
        // field may be selected under it, as its current or earlier code.
        for spec in HYBRID_SPECS.iter().filter(|spec| spec.name != "QI") {
            assert!(
                !spec.candidate_parameters().any(|code| code == CICE),
                "{}",
                spec.name
            );
            assert!(spec.earlier_parameters.is_empty(), "{}", spec.name);
        }
    }

    const CIMIXR: Parameter = Parameter {
        discipline: 0,
        category: 1,
        number: 82,
    };
    const CICE: Parameter = Parameter {
        discipline: 0,
        category: 6,
        number: 0,
    };
    const CYCLE: &str = "2017-01-19 00:00:00";

    /// A header-only record on the canonical HRRR grid: everything the
    /// inventory checks, and no payload, because the inventory never
    /// decodes one.
    fn canonical_record(
        parameter: Parameter,
        level_type: u8,
        level_value: f64,
        forecast_hour: u32,
    ) -> Grib2Message {
        use grib_core::grib2::{DataRepresentation, Identification, ProductDefinition};
        Grib2Message {
            discipline: parameter.discipline,
            identification: Identification {
                center_id: 7,
                ..Default::default()
            },
            reference_time: chrono::NaiveDateTime::parse_from_str(CYCLE, "%Y-%m-%d %H:%M:%S")
                .unwrap(),
            grid: GridDefinition {
                template: 30,
                nx: 1799,
                ny: 1059,
                lat1: 21.138123,
                lon1: 237.280472,
                dx: 3000.0,
                dy: 3000.0,
                latin1: 38.5,
                latin2: 38.5,
                lov: 262.5,
                scan_mode: 0x40,
                shape_of_earth: 6,
                resolution_flags: 0x08,
                ..Default::default()
            },
            product: ProductDefinition {
                template: 0,
                parameter_category: parameter.category,
                parameter_number: parameter.number,
                level_type,
                level_value,
                time_range_unit: 1,
                forecast_time: forecast_hour,
                ..Default::default()
            },
            data_rep: DataRepresentation {
                template: 3,
                section5_num_data_points: 1799 * 1059,
                ..Default::default()
            },
            bitmap: None,
            raw_data: vec![],
        }
    }

    /// Every record the atmosphere inventory requires, with cloud ice
    /// published under `cloud_ice` on each hybrid level.
    fn wrfnat_records(cloud_ice: Parameter, forecast_hour: u32) -> Vec<Grib2Message> {
        let mut records = Vec::new();
        for spec in HYBRID_SPECS {
            let parameter = if spec.name == "QI" { cloud_ice } else { spec.parameter };
            for level in 1..=N_HYBRID_LEVELS {
                records.push(canonical_record(
                    parameter,
                    HYBRID_LEVEL_TYPE,
                    level as f64,
                    forecast_hour,
                ));
            }
        }
        for spec in SURFACE_SPECS {
            records.push(canonical_record(
                spec.parameter,
                spec.level_type,
                spec.level_value,
                forecast_hour,
            ));
        }
        records
    }

    fn qi_codes(inventory: &AtmosInventory) -> Vec<Parameter> {
        inventory
            .selected
            .iter()
            .filter(|field| field.variable == "QI")
            .map(|field| field.parameter)
            .collect()
    }

    #[test]
    fn a_2016_archive_corner_in_millidegrees_is_the_canonical_grid() {
        // The 2016-05-24 wrfnat/wrfsfc files publish lat1 21.138 and lon1
        // 237.28 for the same grid; the microdegree corner stays accepted
        // and a corner that is neither is still refused.
        let mut record = canonical_record(CICE, HYBRID_LEVEL_TYPE, 1.0, 0);
        assert!(validate_canonical_grid(&record.grid).is_ok());
        record.grid.lat1 = 21.138;
        record.grid.lon1 = 237.28;
        assert!(validate_canonical_grid(&record.grid).is_ok());
        record.grid.lat1 = 21.139;
        assert!(validate_canonical_grid(&record.grid).is_err());
        record.grid.lat1 = 21.138;
        record.grid.lon1 = 237.281;
        assert!(validate_canonical_grid(&record.grid).is_err());
        record.grid.lon1 = 237.28;
        record.grid.nx = 1800;
        assert!(validate_canonical_grid(&record.grid).is_err());
    }

    #[test]
    fn a_wrfnat_file_from_before_hrrr_v3_reads_its_cloud_ice_as_cice() {
        // HRRRv1 and v2 wrfnat files (2017-01-19 among them) publish cloud
        // ice as CICE, 0/6/0, on all 50 hybrid levels and no CIMIXR.  Read
        // under 0/1/82 alone, every such cycle was refused with `missing
        // required field QI hybrid level 1`.
        let records = wrfnat_records(CICE, 0);
        let inventory = inventory_atmosphere_messages(&records, CYCLE, 0).unwrap();
        assert_eq!(inventory.selected.len(), 561);
        assert_eq!(qi_codes(&inventory), vec![CICE; N_HYBRID_LEVELS]);
        assert_eq!(
            qice_mapping_line(&inventory).unwrap(),
            "qice_mapping\tPASS discipline=0 category=6 parameter=0 level_type=105; \
             finite/nonnegative/nonzero"
        );
    }

    #[test]
    fn a_current_wrfnat_file_is_selected_and_gated_exactly_as_before() {
        let records = wrfnat_records(CIMIXR, 0);
        let inventory = inventory_atmosphere_messages(&records, CYCLE, 0).unwrap();
        assert_eq!(qi_codes(&inventory), vec![CIMIXR; N_HYBRID_LEVELS]);
        // The literal the gate carried before CICE was readable, byte for
        // byte, so a current cycle's publication does not move.
        assert_eq!(
            qice_mapping_line(&inventory).unwrap(),
            "qice_mapping\tPASS discipline=0 category=1 parameter=82 level_type=105; \
             finite/nonnegative/nonzero"
        );
        // And the very records a CIMIXR-only table selected.
        let expected: Vec<usize> = (0..records.len()).collect();
        let indices: Vec<usize> = inventory.selected.iter().map(|field| field.index).collect();
        assert_eq!(indices, expected);
    }

    #[test]
    fn cimixr_wins_wherever_a_file_publishes_it_and_the_codes_never_mix() {
        // A file carrying both codes on every level is read under CIMIXR
        // alone; the CICE records are not selected and are no duplicate.
        let mut both = wrfnat_records(CIMIXR, 0);
        let cice: Vec<Grib2Message> = (1..=N_HYBRID_LEVELS)
            .map(|level| canonical_record(CICE, HYBRID_LEVEL_TYPE, level as f64, 0))
            .collect();
        both.extend(cice);
        let inventory = inventory_atmosphere_messages(&both, CYCLE, 0).unwrap();
        assert_eq!(qi_codes(&inventory), vec![CIMIXR; N_HYBRID_LEVELS]);

        // A file publishing CIMIXR on 49 levels and CICE on the 50th is
        // not stitched together from two codes: it lacks CIMIXR level 50.
        let mut mixed = wrfnat_records(CIMIXR, 0);
        let last = mixed
            .iter()
            .position(|record| {
                parameter_matches(record, CIMIXR)
                    && level_matches(record.product.level_value, N_HYBRID_LEVELS as f64)
            })
            .unwrap();
        mixed[last] = canonical_record(CICE, HYBRID_LEVEL_TYPE, N_HYBRID_LEVELS as f64, 0);
        let error = inventory_atmosphere_messages(&mixed, CYCLE, 0)
            .unwrap_err()
            .to_string();
        assert_eq!(error, "missing required field QI hybrid level 50");

        // A file publishing cloud ice under neither code is refused, the
        // refusal naming the field.
        let mut none = wrfnat_records(CIMIXR, 0);
        none.retain(|record| !parameter_matches(record, CIMIXR));
        let error = inventory_atmosphere_messages(&none, CYCLE, 0)
            .unwrap_err()
            .to_string();
        assert_eq!(error, "missing required field QI hybrid level 1");
    }

    #[test]
    fn a_series_whose_frames_publish_cloud_ice_under_different_codes_is_refused() {
        let current = inventory_atmosphere_messages(&wrfnat_records(CIMIXR, 0), CYCLE, 0).unwrap();
        let earlier = inventory_atmosphere_messages(&wrfnat_records(CICE, 1), CYCLE, 1).unwrap();
        let error = compare_atmosphere_inventory(&current, &earlier, 1)
            .unwrap_err()
            .to_string();
        assert!(error.contains("selected inventory does not exactly equal"), "{error}");
        let same = inventory_atmosphere_messages(&wrfnat_records(CICE, 1), CYCLE, 1).unwrap();
        let first = inventory_atmosphere_messages(&wrfnat_records(CICE, 0), CYCLE, 0).unwrap();
        compare_atmosphere_inventory(&first, &same, 1).unwrap();
    }

    fn with_aerosol_numbers(mut records: Vec<Grib2Message>, specs: &[HybridSpec]) -> Vec<Grib2Message> {
        for spec in specs {
            for level in 1..=N_HYBRID_LEVELS {
                records.push(canonical_record(spec.parameter, HYBRID_LEVEL_TYPE, level as f64, 0));
            }
        }
        records
    }

    #[test]
    fn a_file_publishing_the_aerosol_number_pair_has_it_selected_and_gated() {
        let records = with_aerosol_numbers(wrfnat_records(CIMIXR, 0), &OPTIONAL_HYBRID_SPECS);
        let inventory = inventory_atmosphere_messages(&records, CYCLE, 0).unwrap();
        assert_eq!(optional_hybrid_fields(&inventory), vec!["QNWFA", "QNIFA"]);
        assert_eq!(
            inventory.selected.len(),
            (HYBRID_SPECS.len() + 2) * N_HYBRID_LEVELS + SURFACE_SPECS.len()
        );
        // A file without the pair selects exactly the 561 it always did.
        let plain = inventory_atmosphere_messages(&wrfnat_records(CIMIXR, 0), CYCLE, 0).unwrap();
        assert!(optional_hybrid_fields(&plain).is_empty());
        assert_eq!(plain.selected.len(), 561);
    }

    #[test]
    fn a_file_publishing_layer_cloud_cover_has_it_selected_and_gated() {
        let mut records = wrfnat_records(CIMIXR, 0);
        for spec in OPTIONAL_SURFACE_SPECS {
            records.push(canonical_record(spec.parameter, spec.level_type, spec.level_value, 0));
        }
        let inventory = inventory_atmosphere_messages(&records, CYCLE, 0).unwrap();
        assert_eq!(optional_surface_fields(&inventory), vec!["PBLH", "LCDC", "MCDC", "HCDC"]);
        assert_eq!(inventory.selected.len(), 565);
        assert_eq!(optional_surface_gate_lines(&inventory), vec![
            "optional_surface_fields\tPBLH,LCDC,MCDC,HCDC",
            "optional_surface_units\tPBLH=m,LCDC=%,MCDC=%,HCDC=%"]);
    }

    #[test]
    fn a_file_publishing_hpbl_has_it_selected_and_gated() {
        let mut records = wrfnat_records(CIMIXR, 0);
        let spec = OPTIONAL_SURFACE_SPECS[0];
        records.push(canonical_record(spec.parameter, spec.level_type, spec.level_value, 0));
        let inventory = inventory_atmosphere_messages(&records, CYCLE, 0).unwrap();
        assert_eq!(optional_surface_fields(&inventory), vec!["PBLH"]);
        assert_eq!(inventory.selected.len(), 562);
        assert_eq!(optional_surface_gate_lines(&inventory), vec![
            "optional_surface_fields\tPBLH", "optional_surface_units\tPBLH=m"]);
        // Without it: the 561 records and no declaration, as before.
        let plain = inventory_atmosphere_messages(&wrfnat_records(CIMIXR, 0), CYCLE, 0).unwrap();
        assert!(optional_surface_fields(&plain).is_empty());
        assert!(optional_surface_gate_lines(&plain).is_empty());
        // Twice is refused rather than one silently chosen.
        records.push(canonical_record(spec.parameter, spec.level_type, spec.level_value, 0));
        let error = inventory_atmosphere_messages(&records, CYCLE, 0).unwrap_err().to_string();
        assert!(error.contains("duplicate optional surface PBLH"), "{error}");
    }

    #[test]
    fn half_the_aerosol_number_pair_is_refused() {
        let records = with_aerosol_numbers(wrfnat_records(CIMIXR, 0), &OPTIONAL_HYBRID_SPECS[..1]);
        let error = inventory_atmosphere_messages(&records, CYCLE, 0).unwrap_err().to_string();
        assert!(error.contains("read whole or not at all"), "{error}");
    }

    const PMTF: Parameter = OPTIONAL_HYBRID_SPECS[0].parameter;
    const POINTS: usize = 1799 * 1059;

    /// A whole wrfnat lead with the aerosol pair, every record carrying a
    /// real packed payload: simple packing at zero bits, so the decoder
    /// expands the reference value over the grid exactly as it does a
    /// constant field read from a file.
    fn decodable_lead(forecast_hour: u32) -> Vec<Grib2Message> {
        let mut records = wrfnat_records(CIMIXR, forecast_hour);
        for spec in OPTIONAL_HYBRID_SPECS {
            for level in 1..=N_HYBRID_LEVELS {
                records.push(canonical_record(
                    spec.parameter,
                    HYBRID_LEVEL_TYPE,
                    level as f64,
                    forecast_hour,
                ));
            }
        }
        for record in &mut records {
            record.data_rep.template = 0;
            record.data_rep.bits_per_value = 0;
            record.data_rep.reference_value = 0.5;
        }
        records
    }

    /// Mask two points of `parameter` on hybrid `level` as NCEP does PMTF
    /// on level 1: a Section 6 bitmap and two fewer packed values.
    fn mask(records: &mut [Grib2Message], parameter: Parameter, level: usize) -> usize {
        let index = records
            .iter()
            .position(|record| {
                parameter_matches(record, parameter)
                    && record.product.level_type == HYBRID_LEVEL_TYPE
                    && level_matches(record.product.level_value, level as f64)
            })
            .unwrap();
        let mut bitmap = vec![true; POINTS];
        bitmap[0] = false;
        bitmap[POINTS - 1] = false;
        records[index].bitmap = Some(bitmap);
        records[index].data_rep.section5_num_data_points = (POINTS - 2) as u32;
        index
    }

    #[test]
    fn masked_pair_is_selected_whole_across_requested_leads() {
        let mut first = decodable_lead(0);
        mask(&mut first, PMTF, 1);
        let reference = inventory_atmosphere_messages(&first, CYCLE, 0).unwrap();
        assert_eq!(optional_hybrid_fields(&reference), vec!["QNWFA", "QNIFA"]);
        assert_eq!(reference.selected.len(), 661);
        for hour in [1, 6, 7, 8] {
            let mut records = decodable_lead(hour);
            mask(&mut records, PMTF, 1);
            let candidate = inventory_atmosphere_messages(&records, CYCLE, hour).unwrap();
            compare_atmosphere_inventory(&reference, &candidate, hour).unwrap();
        }
        let unrequested = inventory_atmosphere_unrequested(&first, CYCLE, 0).unwrap();
        assert_eq!(unrequested.selected.len(), 561);
        assert!(optional_hybrid_gate_lines(&unrequested).is_empty());
        assert_eq!(optional_hybrid_gate_lines(&reference), vec![
            "optional_hybrid_fields\tQNWFA,QNIFA", "optional_hybrid_units\tQNWFA=kg-1,QNIFA=kg-1"]);
    }

    #[test]
    fn inventories_have_frozen_required_cardinality() {
        assert_eq!(
            HYBRID_SPECS.len() * N_HYBRID_LEVELS + SURFACE_SPECS.len(),
            561
        );
        assert_eq!(SOIL_DEPTHS_M.len() * 2, 18);
        assert!(HYBRID_SPECS
            .iter()
            .filter(|spec| spec.require_any_positive)
            .all(|spec| { matches!(spec.name, "QC" | "QI" | "QR" | "QS" | "QG") }));
    }

    #[test]
    fn cycle_normalization_accepts_iso_spelling() {
        assert_eq!(
            normalized_cycle("2026-07-18T00:00:00Z".into()),
            "2026-07-18 00:00:00"
        );
    }

    #[test]
    fn series_hours_accept_absolute_contiguous_windows_and_reject_invalid_series() {
        assert!(validate_series_hours(&[0, 1]).is_ok());
        assert!(validate_series_hours(&(12..=18).collect::<Vec<_>>()).is_ok());
        assert!(validate_series_hours(&(40..=46).collect::<Vec<_>>()).is_ok());
        assert!(validate_series_hours(&(0..=48).collect::<Vec<_>>()).is_ok());
        assert!(validate_series_hours(&[0]).is_err());
        assert!(validate_series_hours(&[12, 14]).is_err());
        assert!(validate_series_hours(&[47, 48, 49]).is_err());
    }

    #[test]
    fn series_cycle_horizon_uses_extended_hrrr_cycles_only() {
        assert!(validate_series_cycle_horizon(&[47, 48], "2026-07-18 18:00:00").is_ok());
        assert!(validate_series_cycle_horizon(&[17, 18], "2026-07-18 05:00:00").is_ok());
        assert!(validate_series_cycle_horizon(&[18, 19], "2026-07-18 05:00:00").is_err());
        assert!(validate_series_cycle_horizon(&[0, 1], "not-a-cycle").is_err());
    }

    /// One step of a `2^-19 * 10^-3` packing grid -- the spacing that put
    /// a saturated GFS soil cell above 1.0, and the same shape of thing
    /// that puts a hydrometeor-free HRRR cell below 0.0.
    const ONE_STEP: f64 = 1.9073486328125e-9;

    #[test]
    fn a_hydrometeor_free_cell_clamps_to_zero_rather_than_refusing() {
        // Cloud water is exactly zero over most of a domain.  Encoded at
        // the bound, it can decode a step below it; that is the packing
        // grid talking, not corruption.
        let bounds = value_bounds(true, false, 0.02);
        match bounds.check(-ONE_STEP, ONE_STEP) {
            BoundVerdict::Clamped { value, excursion } => {
                assert_eq!(value, 0.0);
                assert_eq!(excursion, ONE_STEP);
            }
            other => panic!("a dry cell must clamp, got {other:?}"),
        }
        // Genuinely negative cloud water is still refused.
        assert!(matches!(
            bounds.check(-1.0e-4, ONE_STEP),
            BoundVerdict::Refuse { .. }
        ));
    }

    #[test]
    fn a_saturated_soil_moisture_cell_clamps_to_one() {
        let bounds = value_bounds(true, true, 1.0);
        assert!(matches!(
            bounds.check(1.0 + ONE_STEP, ONE_STEP),
            BoundVerdict::Clamped { value: 1.0, .. }
        ));
        assert!(matches!(
            bounds.check(1.05, ONE_STEP),
            BoundVerdict::Refuse { .. }
        ));
    }

    #[test]
    fn only_the_true_fractions_are_held_to_one() {
        // Water vapour at 2 m has no saturating unit limit, so its
        // ceiling stays an exact refusal rather than a clamped one.
        for variable in ["LANDSEA", "XICE", "SOILW"] {
            assert!(unit_fraction(variable), "{variable}");
        }
        for variable in ["Q2", "SPFH", "QC", "PSFC", "TT"] {
            assert!(!unit_fraction(variable), "{variable}");
        }
        assert!(matches!(
            value_bounds(true, false, 1.0).check(1.0 + ONE_STEP, ONE_STEP),
            BoundVerdict::Inside
        ));
    }

    #[test]
    fn a_field_that_is_zero_everywhere_is_offered_nothing() {
        // With no magnitude to measure a negligible fraction of, there is
        // nothing that could have been quantized: the gate stays shut.
        let bounds = value_bounds(true, false, 0.0);
        assert_eq!(bounds.low_tolerance(ONE_STEP), 0.0);
        assert!(matches!(
            bounds.check(-ONE_STEP, ONE_STEP),
            BoundVerdict::Refuse { .. }
        ));
    }

    #[test]
    fn every_manifest_row_is_as_wide_as_the_header() {
        // The receipt this pins is not hypothetical: three hand-written
        // rows gained `clamped` and `max_excursion` while the header
        // line did not, and the build could not notice -- a 13-column
        // row shipped under an 11-column header, a receipt that reads
        // wrong rather than one that fails.
        let row = manifest_row(
            "atmosphere",
            7,
            "SOILW",
            0.05,
            SOIL_LEVEL_TYPE,
            0,
            false,
            Stats {
                minimum: 0.0,
                maximum: 1.0,
                any_positive_in_source: true,
                count: 4,
                clamped: ClampTally {
                    clamps: 2,
                    max_excursion: 1.9073486328125e-9,
                },
                fill: Default::default(),
            },
            "SOILW.f32le",
        );
        assert_eq!(
            row.split('\t').count(),
            MANIFEST_HEADER.split('\t').count(),
            "row {row}\nheader {MANIFEST_HEADER}"
        );
        assert!(!row.contains('\n'));
        // And the two clamp columns really carry the census.
        let columns: Vec<&str> = row.split('\t').collect();
        let header: Vec<&str> = MANIFEST_HEADER.split('\t').collect();
        let clamped = header.iter().position(|name| *name == "clamped").unwrap();
        assert_eq!(columns[clamped], "2");
        assert_eq!(
            columns[clamped + 1].parse::<f64>().unwrap(),
            1.9073486328125e-9
        );
    }

    fn posted_test_root(name: &str) -> PathBuf {
        // Parallel tests read one clock tick on the 2.8.6 windows-2025 runner and collided on this name; the counter keeps each call distinct.
        static NEXT_SCRATCH: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "gpuwm-hrrr-posted-{name}-{}-{unique}-{}",
            std::process::id(),
            NEXT_SCRATCH.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
        ));
        fs::create_dir_all(&root).unwrap();
        root
    }

    #[test]
    fn a_posted_series_needs_no_lead_on_disk_before_its_admission() {
        let root = posted_test_root("series");
        for name in ["nat00.grib2", "soil00.grib2"] {
            fs::write(root.join(name), b"GRIB").unwrap();
        }
        let series = root.join("series.tsv");
        fs::write(
            &series,
            format!(
                "0\t{0}/nat00.grib2\t{0}/soil00.grib2\n1\t{0}/nat01.grib2\t{0}/soil01.grib2\n",
                root.display()
            ),
        )
        .unwrap();
        // The whole-series reader refuses a lead not posted yet; the posted
        // reader takes it, the first lead included: the decoder starts
        // before the window's first lead posts and opens a lead only once
        // it is admitted.
        assert!(parse_series_manifest(&series).is_err());
        let inputs = parse_series_manifest_with(&series, false).unwrap();
        assert_eq!(inputs.len(), 2);
        fs::remove_file(root.join("nat00.grib2")).unwrap();
        assert!(parse_series_manifest(&series).is_err());
        assert_eq!(parse_series_manifest_with(&series, false).unwrap().len(), 2);
        fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn a_lead_is_read_only_once_admitted_for_its_own_files() {
        let root = posted_test_root("admit");
        let input = SeriesInput {
            forecast_hour: 3,
            atmosphere: "/data/hrrr.t12z.wrfnatf03.grib2".to_owned(),
            soil: "/data/hrrr.t12z.soilf03.grib2".to_owned(),
            supplements: Vec::new(),
        };
        let cancelled = AtomicBool::new(true);
        // Not admitted and cancelled: the wait ends, naming the lead.
        let error = wait_admitted(&root, &input, &cancelled).unwrap_err();
        assert!(error.to_string().contains("f03"));
        let open = AtomicBool::new(false);
        fs::write(
            root.join("f03.admitted"),
            "forecast_hour\t3\natmosphere\t/data/hrrr.t12z.wrfnatf03.grib2\nsoil\t/data/hrrr.t12z.soilf03.grib2\n",
        )
        .unwrap();
        wait_admitted(&root, &input, &open).unwrap();
        // An admission naming other files is refused, never read past.
        fs::write(
            root.join("f03.admitted"),
            "forecast_hour\t3\natmosphere\t/data/other.grib2\nsoil\t/data/hrrr.t12z.soilf03.grib2\n",
        )
        .unwrap();
        assert!(wait_admitted(&root, &input, &open).is_err());
        fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn hrrr_numbers_and_two_metre_humidity_are_requestable_extras() {
        let (hybrid, surface) = requested_extra_sets("QNR,RH2,QNC,QNI").unwrap();
        assert_eq!(hybrid.iter().map(|spec| spec.name).collect::<Vec<_>>(), vec!["QNC", "QNI", "QNR"]);
        assert_eq!(hybrid[0].parameter, Parameter { discipline: 0, category: 6, number: 28 });
        assert_eq!(hybrid[1].parameter, Parameter { discipline: 0, category: 6, number: 29 });
        assert_eq!(hybrid[2].parameter, Parameter { discipline: 0, category: 1, number: 100 });
        assert_eq!(surface.len(), 1);
        assert_eq!((surface[0].name, surface[0].level_type, surface[0].level_value), ("RH2", 103, 2.0));
        assert!(requested_extra_sets("RH2,RH2").is_err());
        assert!(requested_extra_sets("QNC,RH3").is_err());
        assert!(requested_extras("RH2").is_err());
    }

    #[test]
    fn extra_table_is_explicit_and_numeric() {
        assert!(requested_extras("NO_SUCH_FIELD").is_err());
        assert!(requested_extras("MASSDEN,MASSDEN").is_err());
        let specs = requested_extras("PMTF,MASSDEN,PMTC").unwrap();
        assert_eq!(specs[0].name, "MASSDEN");
        assert_eq!(specs[0].parameter, Parameter { discipline: 0, category: 20, number: 0 });
        assert!(specs.iter().all(|spec| spec.nonnegative && !spec.require_any_positive));
    }

    #[test]
    fn extras_require_fifty_matching_records_without_changing_default_inventory() {
        let extras = requested_extras("MASSDEN").unwrap();
        let mut records = wrfnat_records(CIMIXR, 0);
        for level in 1..=50 {
            records.push(canonical_record(extras[0].parameter, 105, level as f64, 0));
        }
        let default = inventory_atmosphere_messages(&records, CYCLE, 0).unwrap();
        assert_eq!(default.selected.len(), 561);
        let selected = inventory_atmosphere_with_extras(&records, CYCLE, 0, &extras, false).unwrap();
        assert_eq!(selected.selected.len(), 611);
        records.pop();
        assert!(inventory_atmosphere_with_extras(&records, CYCLE, 0, &extras, false).unwrap_err().to_string().contains("MASSDEN hybrid level 50"));
        records.push(canonical_record(Parameter { discipline: 0, category: 20, number: 1 }, 105, 50., 0));
        assert!(inventory_atmosphere_with_extras(&records, CYCLE, 0, &extras, false).is_err());
        records.push(canonical_record(extras[0].parameter, 105, 50., 0));
        records.push(canonical_record(extras[0].parameter, 105, 50., 0));
        assert!(inventory_atmosphere_with_extras(&records, CYCLE, 0, &extras, false).is_err());
    }

}
