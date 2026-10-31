//! Fail-closed GFS pressure-level GRIB2 bridge for native WRF initialization.
//!
//! Fields are selected only by GRIB2 discipline/category/parameter and level
//! coordinates.  The complete series is inventoried before an output path is
//! created, and publication is an atomic directory rename.

use gpuwm_preprocess_cpu::quantization::{
    decode_quantum, is_coded_value, BoundKind, BoundVerdict, Bounds, ClampTally,
};
use grib_core::grib2::{flip_rows, unpack_message, Grib2File, Grib2Message, GridDefinition};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::env;
use std::error::Error;
use std::fs::{self, File};
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;

const PRESSURE_LEVEL_TYPE: u8 = 100;
const SURFACE_LEVEL_TYPE: u8 = 1;
const HEIGHT_LEVEL_TYPE: u8 = 103;
const SOIL_LEVEL_TYPE: u8 = 106;
/// The isobaric ladder every GFS subset must carry, in Pa.
///
/// This is the floor, not the ceiling. A case whose model top sits above 100
/// hPa needs the ladder EXTENDED UPWARD -- `woof fetch --source gfs
/// --p-top-pa 5000` fetches the 70 and 50 hPa levels as well -- and a bridge
/// that insists on exactly these 21 would refuse the very file that was
/// fetched to satisfy it. The observed ladder is therefore derived from the
/// source (see `observed_pressure_levels`) and validated to CONTAIN this one,
/// so a deeper top adds levels and can never quietly drop one.
const CERTIFIED_PRESSURE_LEVELS_PA: [f64; 21] = [
    10_000.0, 15_000.0, 20_000.0, 25_000.0, 30_000.0, 35_000.0, 40_000.0, 45_000.0, 50_000.0,
    55_000.0, 60_000.0, 65_000.0, 70_000.0, 75_000.0, 80_000.0, 85_000.0, 90_000.0, 92_500.0,
    95_000.0, 97_500.0, 100_000.0,
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Parameter {
    discipline: u8,
    category: u8,
    number: u8,
}

#[derive(Clone, Copy, Debug)]
struct FieldSpec {
    name: &'static str,
    parameter: Parameter,
    level_type: u8,
    level_value: f64,
    minimum: f64,
    maximum: f64,
    /// What each end of the declared range IS.  `Physical` marks a
    /// saturating limit real cells sit exactly on -- zero snow, a unit
    /// soil-moisture or ice fraction -- which is where GRIB2's packing
    /// quantum can land a value a hair outside; those clamp back onto
    /// the bound and are counted.  `Sanity` marks a slack plausibility
    /// range no real value approaches, which is never widened.
    minimum_kind: BoundKind,
    maximum_kind: BoundKind,
    allow_missing: bool,
}

impl FieldSpec {
    fn bounds(&self) -> Bounds {
        Bounds::closed(
            self.minimum,
            self.maximum,
            self.minimum_kind,
            self.maximum_kind,
        )
    }
}

const PRESSURE_SPECS: [FieldSpec; 5] = [
    FieldSpec {
        name: "GHT",
        parameter: Parameter {
            discipline: 0,
            category: 3,
            number: 5,
        },
        level_type: PRESSURE_LEVEL_TYPE,
        level_value: 0.0,
        minimum: -1000.0,
        maximum: 80_000.0,
        minimum_kind: BoundKind::Sanity,
        maximum_kind: BoundKind::Sanity,
        allow_missing: false,
    },
    FieldSpec {
        name: "T",
        parameter: Parameter {
            discipline: 0,
            category: 0,
            number: 0,
        },
        level_type: PRESSURE_LEVEL_TYPE,
        level_value: 0.0,
        minimum: 120.0,
        maximum: 400.0,
        minimum_kind: BoundKind::Sanity,
        maximum_kind: BoundKind::Sanity,
        allow_missing: false,
    },
    FieldSpec {
        name: "RH",
        parameter: Parameter {
            discipline: 0,
            category: 1,
            number: 1,
        },
        level_type: PRESSURE_LEVEL_TYPE,
        level_value: 0.0,
        minimum: 0.0,
        maximum: 110.0,
        minimum_kind: BoundKind::Physical,
        maximum_kind: BoundKind::Sanity,
        allow_missing: false,
    },
    FieldSpec {
        name: "U",
        parameter: Parameter {
            discipline: 0,
            category: 2,
            number: 2,
        },
        level_type: PRESSURE_LEVEL_TYPE,
        level_value: 0.0,
        minimum: -250.0,
        maximum: 250.0,
        minimum_kind: BoundKind::Sanity,
        maximum_kind: BoundKind::Sanity,
        allow_missing: false,
    },
    FieldSpec {
        name: "V",
        parameter: Parameter {
            discipline: 0,
            category: 2,
            number: 3,
        },
        level_type: PRESSURE_LEVEL_TYPE,
        level_value: 0.0,
        minimum: -250.0,
        maximum: 250.0,
        minimum_kind: BoundKind::Sanity,
        maximum_kind: BoundKind::Sanity,
        allow_missing: false,
    },
];

const SURFACE_SPECS: [FieldSpec; 11] = [
    FieldSpec {
        name: "PSFC",
        parameter: Parameter {
            discipline: 0,
            category: 3,
            number: 0,
        },
        level_type: SURFACE_LEVEL_TYPE,
        level_value: 0.0,
        minimum: 10_000.0,
        maximum: 120_000.0,
        minimum_kind: BoundKind::Sanity,
        maximum_kind: BoundKind::Sanity,
        allow_missing: false,
    },
    FieldSpec {
        name: "SOURCE_OROGRAPHY",
        parameter: Parameter {
            discipline: 0,
            category: 3,
            number: 5,
        },
        level_type: SURFACE_LEVEL_TYPE,
        level_value: 0.0,
        minimum: -1000.0,
        maximum: 10_000.0,
        minimum_kind: BoundKind::Sanity,
        maximum_kind: BoundKind::Sanity,
        allow_missing: false,
    },
    FieldSpec {
        name: "SKINTEMP",
        parameter: Parameter {
            discipline: 0,
            category: 0,
            number: 0,
        },
        level_type: SURFACE_LEVEL_TYPE,
        level_value: 0.0,
        minimum: 170.0,
        maximum: 400.0,
        minimum_kind: BoundKind::Sanity,
        maximum_kind: BoundKind::Sanity,
        allow_missing: false,
    },
    FieldSpec {
        name: "SNOW",
        parameter: Parameter {
            discipline: 0,
            category: 1,
            number: 13,
        },
        level_type: SURFACE_LEVEL_TYPE,
        level_value: 0.0,
        minimum: 0.0,
        maximum: 100_000.0,
        minimum_kind: BoundKind::Physical,
        maximum_kind: BoundKind::Sanity,
        allow_missing: true,
    },
    FieldSpec {
        name: "SNOWH",
        parameter: Parameter {
            discipline: 0,
            category: 1,
            number: 11,
        },
        level_type: SURFACE_LEVEL_TYPE,
        level_value: 0.0,
        minimum: 0.0,
        maximum: 100.0,
        minimum_kind: BoundKind::Physical,
        maximum_kind: BoundKind::Sanity,
        allow_missing: true,
    },
    FieldSpec {
        name: "T2",
        parameter: Parameter {
            discipline: 0,
            category: 0,
            number: 0,
        },
        level_type: HEIGHT_LEVEL_TYPE,
        level_value: 2.0,
        minimum: 170.0,
        maximum: 400.0,
        minimum_kind: BoundKind::Sanity,
        maximum_kind: BoundKind::Sanity,
        allow_missing: false,
    },
    FieldSpec {
        name: "RH2",
        parameter: Parameter {
            discipline: 0,
            category: 1,
            number: 1,
        },
        level_type: HEIGHT_LEVEL_TYPE,
        level_value: 2.0,
        minimum: 0.0,
        maximum: 110.0,
        minimum_kind: BoundKind::Physical,
        maximum_kind: BoundKind::Sanity,
        allow_missing: false,
    },
    FieldSpec {
        name: "U10",
        parameter: Parameter {
            discipline: 0,
            category: 2,
            number: 2,
        },
        level_type: HEIGHT_LEVEL_TYPE,
        level_value: 10.0,
        minimum: -200.0,
        maximum: 200.0,
        minimum_kind: BoundKind::Sanity,
        maximum_kind: BoundKind::Sanity,
        allow_missing: false,
    },
    FieldSpec {
        name: "V10",
        parameter: Parameter {
            discipline: 0,
            category: 2,
            number: 3,
        },
        level_type: HEIGHT_LEVEL_TYPE,
        level_value: 10.0,
        minimum: -200.0,
        maximum: 200.0,
        minimum_kind: BoundKind::Sanity,
        maximum_kind: BoundKind::Sanity,
        allow_missing: false,
    },
    FieldSpec {
        name: "LANDSEA",
        parameter: Parameter {
            discipline: 2,
            category: 0,
            number: 0,
        },
        level_type: SURFACE_LEVEL_TYPE,
        level_value: 0.0,
        minimum: 0.0,
        maximum: 1.0,
        minimum_kind: BoundKind::Physical,
        maximum_kind: BoundKind::Physical,
        allow_missing: false,
    },
    FieldSpec {
        name: "XICE",
        parameter: Parameter {
            discipline: 10,
            category: 2,
            number: 0,
        },
        level_type: SURFACE_LEVEL_TYPE,
        level_value: 0.0,
        minimum: 0.0,
        maximum: 1.0,
        minimum_kind: BoundKind::Physical,
        maximum_kind: BoundKind::Physical,
        allow_missing: false,
    },
];

const SOIL_SPECS: [FieldSpec; 8] = [
    FieldSpec {
        name: "GFS_ST000010",
        parameter: Parameter {
            discipline: 2,
            category: 0,
            number: 2,
        },
        level_type: SOIL_LEVEL_TYPE,
        level_value: 0.0,
        minimum: 170.0,
        maximum: 400.0,
        minimum_kind: BoundKind::Sanity,
        maximum_kind: BoundKind::Sanity,
        allow_missing: true,
    },
    FieldSpec {
        name: "GFS_ST010040",
        parameter: Parameter {
            discipline: 2,
            category: 0,
            number: 2,
        },
        level_type: SOIL_LEVEL_TYPE,
        level_value: 0.1,
        minimum: 170.0,
        maximum: 400.0,
        minimum_kind: BoundKind::Sanity,
        maximum_kind: BoundKind::Sanity,
        allow_missing: true,
    },
    FieldSpec {
        name: "GFS_ST040100",
        parameter: Parameter {
            discipline: 2,
            category: 0,
            number: 2,
        },
        level_type: SOIL_LEVEL_TYPE,
        level_value: 0.4,
        minimum: 170.0,
        maximum: 400.0,
        minimum_kind: BoundKind::Sanity,
        maximum_kind: BoundKind::Sanity,
        allow_missing: true,
    },
    FieldSpec {
        name: "GFS_ST100200",
        parameter: Parameter {
            discipline: 2,
            category: 0,
            number: 2,
        },
        level_type: SOIL_LEVEL_TYPE,
        level_value: 1.0,
        minimum: 170.0,
        maximum: 400.0,
        minimum_kind: BoundKind::Sanity,
        maximum_kind: BoundKind::Sanity,
        allow_missing: true,
    },
    FieldSpec {
        name: "GFS_SM000010",
        parameter: Parameter {
            discipline: 2,
            category: 0,
            number: 192,
        },
        level_type: SOIL_LEVEL_TYPE,
        level_value: 0.0,
        minimum: 0.0,
        maximum: 1.0,
        minimum_kind: BoundKind::Physical,
        maximum_kind: BoundKind::Physical,
        allow_missing: true,
    },
    FieldSpec {
        name: "GFS_SM010040",
        parameter: Parameter {
            discipline: 2,
            category: 0,
            number: 192,
        },
        level_type: SOIL_LEVEL_TYPE,
        level_value: 0.1,
        minimum: 0.0,
        maximum: 1.0,
        minimum_kind: BoundKind::Physical,
        maximum_kind: BoundKind::Physical,
        allow_missing: true,
    },
    FieldSpec {
        name: "GFS_SM040100",
        parameter: Parameter {
            discipline: 2,
            category: 0,
            number: 192,
        },
        level_type: SOIL_LEVEL_TYPE,
        level_value: 0.4,
        minimum: 0.0,
        maximum: 1.0,
        minimum_kind: BoundKind::Physical,
        maximum_kind: BoundKind::Physical,
        allow_missing: true,
    },
    FieldSpec {
        name: "GFS_SM100200",
        parameter: Parameter {
            discipline: 2,
            category: 0,
            number: 192,
        },
        level_type: SOIL_LEVEL_TYPE,
        level_value: 1.0,
        minimum: 0.0,
        maximum: 1.0,
        minimum_kind: BoundKind::Physical,
        maximum_kind: BoundKind::Physical,
        allow_missing: true,
    },
];

#[derive(Clone, Debug, Eq, PartialEq)]
struct GridFingerprint {
    template: u16,
    nx: u32,
    ny: u32,
    lat1: u64,
    lon1: u64,
    lat2: u64,
    lon2: u64,
    dx: u64,
    dy: u64,
    num_data_points: u32,
    scan_mode: u8,
    shape: u8,
    flags: u8,
}

impl GridFingerprint {
    fn from_grid(grid: &GridDefinition) -> Self {
        Self {
            template: grid.template,
            nx: grid.nx,
            ny: grid.ny,
            lat1: grid.lat1.to_bits(),
            lon1: grid.lon1.to_bits(),
            lat2: grid.lat2.to_bits(),
            lon2: grid.lon2.to_bits(),
            dx: grid.dx.to_bits(),
            dy: grid.dy.to_bits(),
            num_data_points: grid.num_data_points,
            scan_mode: grid.scan_mode,
            shape: grid.shape_of_earth,
            flags: grid.resolution_flags,
        }
    }
}

#[derive(Clone, Debug)]
struct Selected {
    index: usize,
    name: &'static str,
    level: f64,
    spec: FieldSpec,
}

#[derive(Clone, Debug)]
struct Inventory {
    selected: Vec<Selected>,
    grid: GridFingerprint,
    land_mask_parameter: u8,
}

#[derive(Clone, Debug)]
struct SeriesInput {
    hour: u32,
    path: PathBuf,
    expected_forecast_process_id: u8,
}

fn sha256_bytes(data: &[u8]) -> [u8; 32] {
    // RustCrypto selects the CPU's SHA implementation at runtime. The
    // digest contract is unchanged; no padded copy of the input is needed.
    Sha256::digest(data).into()
}

fn hex_sha256(data: &[u8]) -> String {
    sha256_bytes(data)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn sha256_file(path: &Path) -> Result<String, Box<dyn Error>> {
    let mut file = File::open(path)?;
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let count = match file.read(&mut buffer) {
            Ok(count) => count,
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(error.into()),
        };
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

fn matches_parameter(message: &Grib2Message, expected: Parameter) -> bool {
    message.discipline == expected.discipline
        && message.product.parameter_category == expected.category
        && message.product.parameter_number == expected.number
}

/// Historical floor for recognizing an exact coded value.  A record
/// whose packing is coarser than this earns its own, larger tolerance;
/// none earns a smaller one.
const CODE_EPSILON: f64 = 1.0e-9;

fn same_level(left: f64, right: f64) -> bool {
    (left - right).abs() <= CODE_EPSILON
}

fn unique_match<F>(
    messages: &[Grib2Message],
    description: &str,
    predicate: F,
) -> Result<usize, Box<dyn Error>>
where
    F: Fn(&Grib2Message) -> bool,
{
    let found: Vec<usize> = messages
        .iter()
        .enumerate()
        .filter_map(|(index, message)| predicate(message).then_some(index))
        .collect();
    match found.as_slice() {
        [index] => Ok(*index),
        [] => Err(format!("missing required GFS field {description}").into()),
        _ => Err(format!("duplicate required GFS field {description}: {found:?}").into()),
    }
}

fn matching_indices<F>(messages: &[Grib2Message], predicate: F) -> Vec<usize>
where
    F: Fn(&Grib2Message) -> bool,
{
    messages
        .iter()
        .enumerate()
        .filter_map(|(index, message)| predicate(message).then_some(index))
        .collect()
}

fn land_mask_match(messages: &[Grib2Message]) -> Result<(usize, u8), Box<dyn Error>> {
    let level_match = |message: &Grib2Message| {
        message.discipline == 2
            && message.product.parameter_category == 0
            && message.product.level_type == SURFACE_LEVEL_TYPE
            && same_level(message.product.level_value, 0.0)
    };
    let modern = matching_indices(messages, |message| {
        level_match(message) && message.product.parameter_number == 218
    });
    let legacy = matching_indices(messages, |message| {
        level_match(message) && message.product.parameter_number == 0
    });
    match (modern.as_slice(), legacy.as_slice()) {
        ([index], _) => Ok((*index, 218)),
        ([], [index]) => Ok((*index, 0)),
        ([], []) => Err("missing required GFS LANDN/LAND field".into()),
        ([..], _) if modern.len() > 1 => {
            Err(format!("duplicate GFS LANDN fields: {modern:?}").into())
        }
        ([], [..]) => Err(format!("duplicate GFS LAND fields: {legacy:?}").into()),
        _ => unreachable!(),
    }
}

fn validate_common(
    message: &Grib2Message,
    cycle: &str,
    hour: u32,
    expected_forecast_process: u8,
    allow_missing: bool,
) -> Result<(), Box<dyn Error>> {
    let identity = &message.identification;
    if identity.center_id != 7
        || identity.subcenter_id != 0
        || identity.master_table_version != 2
        || identity.local_table_version != 1
        || identity.reference_time_significance != 1
        || identity.production_status != 0
        || identity.processed_data_type != 1
    {
        return Err(format!(
            "field is not the certified operational NCEP GFS product: {identity:?}"
        )
        .into());
    }
    if message.reference_time.to_string() != cycle {
        return Err(format!(
            "reference time {} differs from {cycle}",
            message.reference_time
        )
        .into());
    }
    if message.product.template != 0
        || message.product.generating_process != 2
        || message.product.background_generating_process_id != 0
        || message.product.forecast_generating_process_id != expected_forecast_process
        || message.product.time_range_unit != 1
        || message.product.forecast_time != hour
    {
        return Err(format!(
            "field does not resolve to instantaneous f{hour:03}: PDT={}, process_type={}, background_process={}, forecast_process={} (expected {expected_forecast_process}), time_unit={}, forecast_time={}",
            message.product.template,
            message.product.generating_process,
            message.product.background_generating_process_id,
            message.product.forecast_generating_process_id,
            message.product.time_range_unit,
            message.product.forecast_time,
        )
        .into());
    }
    // Simple packing and complex packing with second-order differencing.
    // Historical GFS also declares embedded primary/secondary missing codes.
    // The shared decoder excludes those cells from the differencing chain;
    // scan_normalized carries their mask alongside the Section-6 bitmap.
    // Required atmospheric fields still reject every missing cell.
    match message.data_rep.template {
        0 => {}
        3 => {
            if message.data_rep.group_splitting_method != 1 {
                return Err(format!(
                    "unsupported GFS complex-packing group splitting method {} \
                     (only 1, general group splitting, is certified)",
                    message.data_rep.group_splitting_method
                )
                .into());
            }
            if message.data_rep.missing_value_management > 2 {
                return Err(format!(
                    "unsupported GFS complex-packing missing-value management {} \
                     (reserved Code Table 5.5 value; cannot identify missing codes)",
                    message.data_rep.missing_value_management
                )
                .into());
            }
            if message.data_rep.spatial_diff_order != 2 {
                return Err(format!(
                    "unsupported GFS spatial differencing order {} \
                     (only the second order the product publishes is certified)",
                    message.data_rep.spatial_diff_order
                )
                .into());
            }
            if !(1..=4).contains(&message.data_rep.spatial_diff_bytes) {
                return Err(format!(
                    "unsupported GFS spatial differencing descriptor width {} bytes",
                    message.data_rep.spatial_diff_bytes
                )
                .into());
            }
        }
        other => {
            return Err(format!("unsupported GFS DRT 5.{other}").into());
        }
    }
    if message.data_rep.bits_per_value > 63 {
        return Err(format!(
            "unsupported GFS packing bits_per_value {}",
            message.data_rep.bits_per_value
        )
        .into());
    }
    if message.data_rep.original_field_type != 0 {
        return Err(format!(
            "unsupported GFS original-field type {}",
            message.data_rep.original_field_type
        )
        .into());
    }
    // A field that may have missing cells (soil and snow, undefined over
    // water) is not required to have any.  The NOMADS subregion crop
    // drops Section 6's bitmap when every cell of the crop is defined, so
    // an all-land crop carries these fields with no bitmap at all; the
    // old equality check refused that valid crop.  Missing cells still
    // need an explicit bitmap or declared embedded code. The decode mask
    // combines both; emit_values still refuses missing required fields.
    if message.bitmap.is_some() && !allow_missing {
        return Err(format!(
            "GFS missing-value policy mismatch: bitmap={} expected={allow_missing}",
            message.bitmap.is_some()
        )
        .into());
    }
    let grid = &message.grid;
    // Two row orders are legal here, and only two.  The NOMADS
    // grib-filter `subregion` crop re-encodes south-to-north (0x40);
    // the raw noaa-gfs-bdp-pds S3 objects -- and, as it happens, an
    // uncropped grib-filter request, which is a pure byte-range
    // extractor -- carry the WMO-default north-to-south (0x00).  They
    // are the same numbers in the opposite order: a flipped raw-S3
    // decode is bit-identical to the NOMADS-crop decode of the same
    // cycle and fields (proved by tests/test_gfs_scan_order.py against
    // a committed matched pair).  Row order is normalized once, on
    // decode, so everything downstream still sees exactly one form.
    // Every other scan mode -- i-negative, j-consecutive, alternating
    // rows -- stays fail-closed.
    let south_to_north = grid.scan_mode == SOUTH_TO_NORTH_SCAN;
    let north_to_south = grid.scan_mode == NORTH_TO_SOUTH_SCAN;
    // lat1 is the first stored row either way, so the span it must
    // close runs with the scan direction.
    let latitude_span = if south_to_north {
        grid.lat1 + (grid.ny - 1) as f64 * grid.dy
    } else {
        grid.lat1 - (grid.ny - 1) as f64 * grid.dy
    };
    if grid.template != 0
        || grid.nx < 2
        || grid.ny < 2
        || !(south_to_north || north_to_south)
        || grid.shape_of_earth != 6
        || grid.resolution_flags != 0x30
        || grid.num_data_points as u64 != grid.nx as u64 * grid.ny as u64
        || !grid.lat1.is_finite()
        || !grid.lon1.is_finite()
        || !grid.lat2.is_finite()
        || !grid.lon2.is_finite()
        || !grid.dx.is_finite()
        || !grid.dy.is_finite()
        || ![0.25, 0.5, 1.0].iter().any(|dx| same_level(grid.dx, *dx))
        || !same_level(grid.dy, grid.dx)
        || !same_level(grid.lat2, latitude_span)
        || !same_level(grid.lon2, grid.lon1 + (grid.nx - 1) as f64 * grid.dx)
    {
        return Err(format!(
            "field is not a supported regular GFS grid (scan mode must be \
             0x40 south-to-north or 0x00 north-to-south): {grid:?}"
        )
        .into());
    }
    Ok(())
}

/// The scan mode this bridge emits, and the one NOMADS crops carry:
/// +i, +j, i-consecutive -- row 0 is the southernmost.
const SOUTH_TO_NORTH_SCAN: u8 = 0x40;

/// The WMO default the raw S3 objects carry: +i, -j, i-consecutive --
/// row 0 is the northernmost.  Accepted, then flipped on decode.
const NORTH_TO_SOUTH_SCAN: u8 = 0x00;

/// Southernmost latitude of a validated grid, whichever way it scans.
fn southern_latitude(grid: &GridDefinition) -> f64 {
    if grid.scan_mode == SOUTH_TO_NORTH_SCAN {
        grid.lat1
    } else {
        grid.lat2
    }
}

/// Northernmost latitude of a validated grid.
fn northern_latitude(grid: &GridDefinition) -> f64 {
    if grid.scan_mode == SOUTH_TO_NORTH_SCAN {
        grid.lat2
    } else {
        grid.lat1
    }
}

/// Reverse the row order of a boolean mask, matching [`flip_rows`].
fn flip_mask_rows(mask: &mut [bool], nx: usize, ny: usize) {
    if nx == 0 || ny < 2 || mask.len() != nx * ny {
        return;
    }
    for row in 0..ny / 2 {
        let (head, tail) = mask.split_at_mut((ny - 1 - row) * nx);
        head[row * nx..row * nx + nx].swap_with_slice(&mut tail[..nx]);
    }
}

type ScannedValues = (Vec<f64>, Option<Vec<bool>>);

/// Decode one message into this bridge's canonical south-to-north
/// row-major order, with its missing-value mask in the same order.
///
/// `unpack_message` deliberately preserves storage order (matching
/// ecCodes), so the flip belongs here rather than in the decoder.  The
/// bitmap is indexed by the same flat position as the values, so it has
/// to travel with them or a north-to-south field would attribute its
/// missing cells to the wrong rows.
fn scan_normalized(
    message: &Grib2Message,
) -> Result<ScannedValues, Box<dyn Error>> {
    let nx = message.grid.nx as usize;
    let ny = message.grid.ny as usize;
    let mut values = unpack_message(message)?;
    let mut present = message.bitmap.clone();
    if message.data_rep.template == 3
        && matches!(message.data_rep.missing_value_management, 1 | 2)
    {
        // unpack_message creates NaN only for a declared missing code or
        // bitmap cell. Do this before the row flip so both masks move together.
        let mask = present.get_or_insert_with(|| vec![true; values.len()]);
        if mask.len() != values.len() {
            return Err("embedded missing mask length differs from decoded grid".into());
        }
        for (cell, value) in mask.iter_mut().zip(&values) {
            if value.is_nan() { *cell = false; }
        }
    }
    if message.grid.scan_mode == NORTH_TO_SOUTH_SCAN {
        if values.len() != nx * ny {
            return Err(format!(
                "cannot normalize scan order: decoded {} values for a {nx}x{ny} grid",
                values.len()
            )
            .into());
        }
        flip_rows(&mut values, nx, ny);
        if let Some(mask) = present.as_mut() {
            if mask.len() != nx * ny {
                return Err(format!(
                    "cannot normalize scan order: bitmap has {} cells for a {nx}x{ny} grid",
                    mask.len()
                )
                .into());
            }
            flip_mask_rows(mask, nx, ny);
        }
    }
    Ok((values, present))
}

/// The isobaric ladder this file actually carries, in ascending Pa.
///
/// A level counts only when EVERY 3-D spec appears on it exactly once:
/// four-fifths of a level is not a level this bridge can decode, and adopting
/// one would produce arrays with a hole in them.
///
/// The result must be the certified ladder, optionally extended upward.  Two
/// clauses say all of it:
///
///   * it must END with the certified 21 exactly -- so a source missing a
///     level the certified runs have always used is refused rather than
///     silently decoded one level short;
///   * it must be strictly increasing -- the arrays and the vertical
///     interpolation downstream are ordered by construction.
///
/// Together those force every extra level strictly above the certified top: an
/// increasing ladder whose last 21 begin at 10000 Pa cannot carry anything at
/// or below 10000 Pa ahead of them.  So an interior insertion, a dropped
/// certified level and a reordering are all refused, and no third clause is
/// needed (one would be unreachable).
fn observed_pressure_levels(file: &Grib2File) -> Result<Vec<f64>, Box<dyn Error>> {
    let mut candidates: Vec<f64> = Vec::new();
    for message in &file.messages {
        if message.product.level_type != PRESSURE_LEVEL_TYPE {
            continue;
        }
        let level = message.product.level_value;
        if !level.is_finite() || level <= 0.0 {
            continue;
        }
        if !candidates.iter().any(|seen| same_level(*seen, level)) {
            candidates.push(level);
        }
    }
    candidates.sort_by(|left, right| left.partial_cmp(right).unwrap());
    let mut levels: Vec<f64> = Vec::new();
    for level in candidates {
        let complete = PRESSURE_SPECS.iter().all(|spec| {
            file.messages
                .iter()
                .filter(|m| {
                    matches_parameter(m, spec.parameter)
                        && m.product.level_type == PRESSURE_LEVEL_TYPE
                        && same_level(m.product.level_value, level)
                })
                .count()
                == 1
        });
        if complete {
            levels.push(level);
        }
    }
    validate_ladder_shape(&levels, "source")?;
    Ok(levels)
}

/// The two clauses every decode ladder must satisfy, whichever side
/// declared it -- see `observed_pressure_levels` for why they suffice.
fn validate_ladder_shape(levels: &[f64], what: &str) -> Result<(), Box<dyn Error>> {
    if levels.len() < CERTIFIED_PRESSURE_LEVELS_PA.len() {
        return Err(format!(
            "{what} carries {} complete isobaric levels, fewer than the {} \
             certified levels every GFS subset must have",
            levels.len(),
            CERTIFIED_PRESSURE_LEVELS_PA.len()
        )
        .into());
    }
    let tail = &levels[levels.len() - CERTIFIED_PRESSURE_LEVELS_PA.len()..];
    if tail
        .iter()
        .zip(CERTIFIED_PRESSURE_LEVELS_PA.iter())
        .any(|(observed, expected)| !same_level(*observed, *expected))
    {
        return Err(format!(
            "{what} isobaric ladder {tail:?} Pa does not end with the certified \
             ladder {CERTIFIED_PRESSURE_LEVELS_PA:?} Pa"
        )
        .into());
    }
    if levels.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(format!("{what} isobaric ladder is not strictly increasing").into());
    }
    Ok(())
}

/// The `--pressure-levels-pa` declaration, parsed and held to the same
/// shape as a derived ladder.  Levels the file turns out not to carry
/// refuse by name in the per-hour inventory, so nothing needs checking
/// against the file here.
fn requested_pressure_levels(csv: &str) -> Result<Vec<f64>, Box<dyn Error>> {
    let levels: Vec<f64> = csv
        .split(',')
        .map(|item| {
            let level: f64 = item.trim().parse().map_err(|_| {
                format!("--pressure-levels-pa entry {item:?} is not a number")
            })?;
            if !level.is_finite() || level <= 0.0 {
                return Err(format!(
                    "--pressure-levels-pa entry {item:?} is not a positive pressure"
                ));
            }
            Ok(level)
        })
        .collect::<Result<_, String>>()?;
    validate_ladder_shape(&levels, "requested")?;
    Ok(levels)
}

fn inventory(
    file: &Grib2File,
    cycle: &str,
    hour: u32,
    expected_forecast_process_id: u8,
    pressure_levels_pa: &[f64],
    snow_convention: SnowConvention,
) -> Result<Inventory, Box<dyn Error>> {
    let mut selected = Vec::with_capacity(PRESSURE_SPECS.len() * pressure_levels_pa.len() + 19);
    let mut fingerprint: Option<GridFingerprint> = None;
    for spec in PRESSURE_SPECS {
        for &level in pressure_levels_pa {
            let index = unique_match(
                &file.messages,
                &format!("{} at {} Pa", spec.name, level),
                |m| {
                    matches_parameter(m, spec.parameter)
                        && m.product.level_type == PRESSURE_LEVEL_TYPE
                        && same_level(m.product.level_value, level)
                },
            )?;
            let message = &file.messages[index];
            validate_common(message, cycle, hour, expected_forecast_process_id, false)?;
            if message.product.second_level_type != 255 {
                return Err(format!(
                    "{} at {} Pa unexpectedly has a second fixed surface",
                    spec.name, level
                )
                .into());
            }
            let observed = GridFingerprint::from_grid(&message.grid);
            if let Some(expected) = &fingerprint {
                if expected != &observed {
                    return Err("GFS selected grids differ".into());
                }
            } else {
                fingerprint = Some(observed);
            }
            selected.push(Selected {
                index,
                name: spec.name,
                level,
                spec,
            });
        }
    }
    let mut land_mask_parameter = None;
    for mut spec in SURFACE_SPECS.into_iter().chain(SOIL_SPECS) {
        if spec.name == "SNOWH" && snow_convention == SnowConvention::Wps {
            // Vtable.GFS reads WEASD; WPS derives depth from that field.
            // A WPS decode must not depend on SNOD being present.
            spec.parameter.number = 13;
            spec.maximum = 100_000.0; // input WEASD bounds, before conversion
        }
        let (index, actual_spec) = if spec.name == "LANDSEA" {
            let (index, parameter_number) = land_mask_match(&file.messages)?;
            land_mask_parameter = Some(parameter_number);
            let mut selected_spec = spec;
            selected_spec.parameter.number = parameter_number;
            if parameter_number == 0 {
                selected_spec.maximum = 2.0;
            }
            (index, selected_spec)
        } else {
            let index = unique_match(&file.messages, spec.name, |m| {
                matches_parameter(m, spec.parameter)
                    && m.product.level_type == spec.level_type
                    && same_level(m.product.level_value, spec.level_value)
            })?;
            (index, spec)
        };
        let message = &file.messages[index];
        validate_common(
            message,
            cycle,
            hour,
            expected_forecast_process_id,
            actual_spec.allow_missing,
        )?;
        if actual_spec.level_type == SOIL_LEVEL_TYPE {
            let expected_top = match actual_spec.level_value {
                value if same_level(value, 0.0) => 0.1,
                value if same_level(value, 0.1) => 0.4,
                value if same_level(value, 0.4) => 1.0,
                value if same_level(value, 1.0) => 2.0,
                _ => {
                    return Err(format!(
                        "unrecognized GFS soil lower bound {}",
                        actual_spec.level_value
                    )
                    .into())
                }
            };
            if message.product.second_level_type != SOIL_LEVEL_TYPE
                || !same_level(message.product.second_level_value, expected_top)
            {
                return Err(format!(
                    "{} must span exactly {}..{} m below land surface; observed second surface type={} value={}",
                    actual_spec.name,
                    actual_spec.level_value,
                    expected_top,
                    message.product.second_level_type,
                    message.product.second_level_value
                )
                .into());
            }
        } else if message.product.second_level_type != 255 {
            return Err(format!(
                "{} unexpectedly has a second fixed surface",
                actual_spec.name
            )
            .into());
        }
        let observed = GridFingerprint::from_grid(&message.grid);
        if fingerprint.as_ref() != Some(&observed) {
            return Err(format!("{} grid differs from pressure grid", actual_spec.name).into());
        }
        selected.push(Selected {
            index,
            name: actual_spec.name,
            level: actual_spec.level_value,
            spec: actual_spec,
        });
    }
    // Five records per isobaric level plus the single-level ones.  The count
    // is a function of the ladder, not the constant 124 that a 21-level
    // ladder happens to produce -- a case whose model top sits above 100 hPa
    // decodes a longer ladder and 124 would refuse the very file that was
    // fetched to satisfy it.  Still exact, and still fail-closed.
    let expected =
        PRESSURE_SPECS.len() * pressure_levels_pa.len() + SURFACE_SPECS.len() + SOIL_SPECS.len();
    if selected.len() != expected {
        return Err(format!(
            "selected {} GFS records, expected {} ({} isobaric levels x {} fields + {} \
             single-level records)",
            selected.len(),
            expected,
            pressure_levels_pa.len(),
            PRESSURE_SPECS.len(),
            SURFACE_SPECS.len() + SOIL_SPECS.len()
        )
        .into());
    }
    Ok(Inventory {
        selected,
        grid: fingerprint.ok_or("empty GFS inventory")?,
        land_mask_parameter: land_mask_parameter.ok_or("GFS LAND mask was not selected")?,
    })
}

#[cfg(test)]
fn parse_series(path: &Path) -> Result<Vec<SeriesInput>, Box<dyn Error>> {
    parse_series_rows(path, false)
}

/// The rows of a series file.  `lead_batch` is a decode of some leads of a
/// series (`--lead-batch`): a preparation decoding each lead as it posts
/// hands the bridge the leads it has.  Nothing a decode writes reads across
/// hours (every consumed record is an instantaneous product, and each hour
/// is written from its own file), so the one rule that needs two times, a
/// boundary series having at least one interval, belongs to the series
/// the caller assembles from its batches, and a batch may carry one lead.
fn parse_series_rows(path: &Path, lead_batch: bool) -> Result<Vec<SeriesInput>, Box<dyn Error>> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let mut result = Vec::new();
    for (line_index, raw) in fs::read_to_string(path)?.lines().enumerate() {
        if raw.trim().is_empty() || raw.trim_start().starts_with('#') {
            continue;
        }
        let columns: Vec<&str> = raw.split('\t').collect();
        if !matches!(columns.len(), 2 | 3) {
            return Err(format!(
                "series line {} must be HOUR<TAB>GRIB2[<TAB>FORECAST_PROCESS_ID]",
                line_index + 1
            )
            .into());
        }
        let hour = columns[0].parse::<u32>()?;
        // A two-column legacy row declares only the analysis process
        // capability.  Process 96 is never inferred from the hour: it must
        // appear explicitly in the series written by a source adapter whose
        // forecast products have real-sample certification.
        let expected_forecast_process_id = if columns.len() == 3 {
            columns[2].parse::<u8>()?
        } else {
            81
        };
        if !matches!(expected_forecast_process_id, 81 | 96) {
            return Err(format!(
                "series line {} declares uncertified forecast process ID {}",
                line_index + 1,
                expected_forecast_process_id
            )
            .into());
        }
        if hour == 0 && expected_forecast_process_id != 81 {
            return Err(format!(
                "series line {} must declare analysis process ID 81 for f000",
                line_index + 1
            )
            .into());
        }
        let candidate = PathBuf::from(columns[1]);
        let resolved = if candidate.is_absolute() {
            candidate
        } else {
            parent.join(candidate)
        };
        if !resolved.is_file() {
            return Err(format!(
                "series line {} references missing {resolved:?}",
                line_index + 1
            )
            .into());
        }
        result.push(SeriesInput {
            hour,
            path: resolved,
            expected_forecast_process_id,
        });
    }
    // The series names the SOURCE leads it carries and need not begin at
    // f000.  Initializing from a forecast lead is ordinary practice, and
    // this anchor was one of the places that forced a user wanting the
    // f174..f240 window to decode from f000.  Everything that made the
    // anchor appear necessary is still enforced below and above: `hour`
    // is unsigned, the process ID is declared per line and never inferred
    // (so an f000 row must still say 81 and a lead row must say 96), and
    // the cadence must be positive, uniform and on published source leads.
    let hours: Vec<u32> = result.iter().map(|value| value.hour).collect();
    check_series_hours(&hours, lead_batch)?;
    Ok(result)
}

/// The rules a series' hours obey: at least two times (one for a lead
/// batch, whose series is assembled from its batches), within the
/// certified horizon, a positive uniform cadence, on published leads.
/// The same rules hold the hours of merged lead batches (`--merge-batches`).
fn check_series_hours(hours: &[u32], lead_batch: bool) -> Result<(), Box<dyn Error>> {
    if hours.is_empty() || (hours.len() < 2 && !lead_batch) {
        return Err("GFS series must contain at least two times".into());
    }
    if hours.last().copied().unwrap_or(0) > 384 {
        return Err("GFS series exceeds the certified f384 product horizon".into());
    }
    let deltas: Vec<u32> = hours
        .windows(2)
        .map(|pair| pair[1].checked_sub(pair[0]))
        .collect::<Option<_>>()
        .ok_or("GFS hours must strictly increase")?;
    if deltas
        .iter()
        .any(|delta| *delta == 0 || *delta != deltas[0])
    {
        return Err("GFS series cadence must be positive and uniform".into());
    }
    if let Some(hour) = hours.iter().find(|hour| **hour > 120 && **hour % 3 != 0) {
        return Err(format!(
            "GFS does not publish f{:03}: leads are hourly through f120 and every 3 hours afterward; choose a published lead window",
            hour
        ).into());
    }
    Ok(())
}

/// What one decoded field contributed to the receipts.
#[derive(Clone, Copy, Debug)]
struct FieldSummary {
    finite: usize,
    missing: usize,
    minimum: f64,
    maximum: f64,
    /// Cells that sat outside a physical bound by no more than this
    /// record's derived quantization tolerance and were clamped onto it.
    clamped: ClampTally,
}

/// The decode quantum of the record carrying this field.
fn field_quantum(message: &Grib2Message) -> f64 {
    decode_quantum(
        message.data_rep.template,
        message.data_rep.binary_scale,
        message.data_rep.decimal_scale,
    )
}

/// The quantization census, as gate lines.
///
/// A bounded physical field whose cells sit exactly on their limit
/// decodes a fraction of one packing step outside it; those cells were
/// clamped onto the bound, and the receipt says which fields and how
/// far, so the clamping is auditable rather than invisible.  The three
/// lines are always written -- a run with no clamps says so with zeros
/// and an empty field list, which is a stronger statement than silence.
fn clamp_receipt_lines(census: &BTreeMap<&'static str, ClampTally>) -> Vec<String> {
    clamp_receipt_lines_by_name(census)
}

/// [`clamp_receipt_lines`] over any field-name key, so the census merged
/// from lead batches (`--merge-batches`) is written by the same lines.
fn clamp_receipt_lines_by_name<K: AsRef<str> + Ord>(census: &BTreeMap<K, ClampTally>) -> Vec<String> {
    let total: usize = census.values().map(|tally| tally.clamps).sum();
    let worst = census
        .values()
        .map(|tally| tally.max_excursion)
        .fold(0.0f64, f64::max);
    let fields = census
        .iter()
        .filter(|(_, tally)| tally.clamps > 0)
        .map(|(name, tally)| format!("{}:{}:{}", name.as_ref(), tally.clamps, tally.max_excursion))
        .collect::<Vec<_>>()
        .join(",");
    vec![
        format!("bound_clamp_total\t{total}"),
        format!("bound_clamp_max_excursion\t{worst}"),
        format!("bound_clamp_fields\t{fields}"),
    ]
}

fn out_of_range(spec: FieldSpec, value: f64, excursion: f64, tolerance: f64) -> Box<dyn Error> {
    format!(
        "{} value {value} outside [{},{}] by {excursion} \
         (quantization tolerance {tolerance}); a bound-kissing value is clamped, \
         this one is not",
        spec.name, spec.minimum, spec.maximum
    )
    .into()
}

fn write_field(
    message: &Grib2Message,
    writer: &mut BufWriter<File>,
    spec: FieldSpec,
    all_water_source: bool,
    snow_convention: SnowConvention,
) -> Result<FieldSummary, Box<dyn Error>> {
    let (values, present) = scan_normalized(message)?;
    let expected = message.grid.nx as usize * message.grid.ny as usize;
    if values.len() != expected {
        return Err(format!(
            "{} decoded {} values, expected {expected}",
            spec.name,
            values.len()
        )
        .into());
    }
    emit_converted_values(
        &values,
        present.as_deref(),
        spec,
        field_quantum(message),
        all_water_source,
        writer,
        snow_convention,
    )
}

/// Bound-check, clamp, and emit one decoded field.  Split from
/// `write_field` so the decision this patch changed can be exercised on
/// bare values, without a hand-packed GRIB2 record standing in the way.
#[cfg(test)]
fn emit_values<W: Write>(
    values: &[f64],
    present: Option<&[bool]>,
    spec: FieldSpec,
    quantum: f64,
    writer: &mut W,
) -> Result<FieldSummary, Box<dyn Error>> {
    emit_values_with_surface_support(values, present, spec, quantum, false, writer)
}

#[cfg(test)]
fn emit_values_with_surface_support<W: Write>(
    values: &[f64],
    present: Option<&[bool]>,
    spec: FieldSpec,
    quantum: f64,
    all_water_source: bool,
    writer: &mut W,
) -> Result<FieldSummary, Box<dyn Error>> {
    emit_converted_values(values, present, spec, quantum, all_water_source, writer, SnowConvention::Raw)
}

fn emit_converted_values<W: Write>(
    values: &[f64],
    present: Option<&[bool]>,
    spec: FieldSpec,
    quantum: f64,
    all_water_source: bool,
    writer: &mut W,
    snow_convention: SnowConvention,
) -> Result<FieldSummary, Box<dyn Error>> {
    let bounds = spec.bounds();
    let mut finite = 0usize;
    let mut missing = 0usize;
    let mut minimum = f64::INFINITY;
    let mut maximum = f64::NEG_INFINITY;
    let mut clamped = ClampTally::default();
    for (value_index, raw_value) in values.iter().copied().enumerate() {
        let mut value = normalize_for_output(raw_value, spec, quantum)?;
        if value.is_finite() {
            match bounds.check(value, quantum) {
                BoundVerdict::Inside => {}
                BoundVerdict::Clamped {
                    value: bound,
                    excursion,
                } => {
                    // The cell is physically AT the limit; the packing
                    // grid simply has no representation exactly on it.
                    value = bound;
                    clamped.record(excursion);
                }
                BoundVerdict::Refuse {
                    excursion,
                    tolerance,
                } => {
                    return Err(out_of_range(spec, value, excursion, tolerance));
                }
            }
            value = snow_convention.convert(spec.name, value);
            finite += 1;
            minimum = minimum.min(value);
            maximum = maximum.max(value);
        } else if value.is_nan() {
            // `present` travels in the same row order as `values`, so a
            // north-to-south field's missing cells stay on their own
            // rows through the flip.
            let bitmap_missing = present
                .as_ref()
                .and_then(|bitmap| bitmap.get(value_index))
                .map(|cell| !*cell)
                .unwrap_or(false);
            if !spec.allow_missing || !bitmap_missing {
                return Err(format!(
                    "{} contains NaN outside an explicit missing bitmap cell",
                    spec.name
                )
                .into());
            }
            missing += 1;
        } else {
            return Err(format!("{} contains infinite data", spec.name).into());
        }
        let output = value as f32;
        if value.is_finite() && !output.is_finite() {
            return Err(format!("{} overflows FP32", spec.name).into());
        }
        writer.write_all(&output.to_le_bytes())?;
    }
    if finite == 0 {
        if !(all_water_source && spec.allow_missing && missing > 0 && missing == values.len()) {
            return Err(format!("{} has no finite support", spec.name).into());
        }
        // Soil and snow have no support on an ocean-only crop. Retain the
        // complete missing array and an accurate missing range in its receipt.
        minimum = f64::NAN;
        maximum = f64::NAN;
    }
    Ok(FieldSummary {
        finite,
        missing,
        minimum,
        maximum,
        clamped,
    })
}

/// The land mask is a code field, not a range: its contract is the exact
/// set 0/1 (or 0/1/2 on the legacy parameter).  The same packing grid
/// that pushes a saturated soil cell past 1.0 pushes a land cell off its
/// code, so the codes are recognized to the record's own tolerance
/// rather than to a fixed epsilon, then re-emitted exactly.
fn normalize_for_output(
    value: f64,
    spec: FieldSpec,
    quantum: f64,
) -> Result<f64, Box<dyn Error>> {
    if spec.name != "LANDSEA" || !value.is_finite() {
        return Ok(value);
    }
    // Codes sit at 0, 1 and (legacy) 2; the widest offer any of them
    // earns covers all of them, and never falls below the historical
    // epsilon this contract already shipped with.
    let tolerance = spec
        .bounds()
        .tolerance_at(1.0, BoundKind::Physical, quantum)
        .max(CODE_EPSILON);
    if spec.parameter.number == 0 {
        if is_coded_value(value, 0.0, tolerance) || is_coded_value(value, 2.0, tolerance) {
            return Ok(0.0);
        }
        if is_coded_value(value, 1.0, tolerance) {
            return Ok(1.0);
        }
        return Err(format!("legacy GFS LAND value {value} is not 0/1/2").into());
    }
    if is_coded_value(value, 0.0, tolerance) {
        return Ok(0.0);
    }
    if is_coded_value(value, 1.0, tolerance) {
        return Ok(1.0);
    }
    Err(format!("GFS LANDN value {value} is not 0/1").into())
}

fn decoded_f32_bits(
    file: &Grib2File,
    inventory: &Inventory,
    name: &str,
) -> Result<Vec<u32>, Box<dyn Error>> {
    let selected: Vec<&Selected> = inventory
        .selected
        .iter()
        .filter(|field| field.name == name)
        .collect();
    let [field] = selected.as_slice() else {
        return Err(format!("invariant field {name} is not unique").into());
    };
    let (values, _) = scan_normalized(&file.messages[field.index])?;
    values
        .into_iter()
        .map(|value| {
            let output = value as f32;
            if value.is_finite() && !output.is_finite() {
                Err(format!("invariant field {name} overflows FP32").into())
            } else {
                Ok(output.to_bits())
            }
        })
        .collect()
}

fn source_is_all_water(file: &Grib2File, inventory: &Inventory) -> Result<bool, Box<dyn Error>> {
    let fields: Vec<_> = inventory.selected.iter().filter(|field| field.name == "LANDSEA").collect();
    let [field] = fields.as_slice() else {
        return Err("source water support requires one validated LANDSEA field".into());
    };
    let message = &file.messages[field.index];
    let (values, _) = scan_normalized(message)?;
    if values.is_empty() {
        return Err("source LANDSEA has no grid values".into());
    }
    for value in values {
        if normalize_for_output(value, field.spec, field_quantum(message))? != 0.0 {
            return Ok(false);
        }
    }
    Ok(true)
}

fn require_exact_invariance(
    reference: &[u32],
    candidate: &[u32],
    name: &str,
    hour: u32,
) -> Result<(), Box<dyn Error>> {
    if reference != candidate {
        return Err(format!("GFS {name} differs between f000 and f{hour:03}").into());
    }
    Ok(())
}

fn fnv1a64(values: &[u32]) -> u64 {
    let mut digest = 0xcbf29ce484222325u64;
    for value in values {
        for byte in value.to_le_bytes() {
            digest ^= byte as u64;
            digest = digest.wrapping_mul(0x100000001b3);
        }
    }
    digest
}

/// Worker count for the series' independent work (one input object, or
/// one forecast hour, per task): `GPUWM_GFS_BRIDGE_THREADS` when it
/// names a positive count (`woof prep --preprocess-workers`), else
/// every core this process may run on.
///
/// Named breakage: the whole series was read, hashed and decoded on one
/// core -- every forecast hour one after another, with the input and
/// every written array hashed by the old scalar SHA-256 -- so the
/// decode of a multi-day GFS series took minutes on a 16-core box.
const THREADS_ENV: &str = "GPUWM_GFS_BRIDGE_THREADS";

/// WPS 4.6 rrpr.F:877-889 and its SNOWH derivation. Keep the raw GFS
/// WEASD/SNOD choice explicit; both the bridge gate and preparation receipt
/// name it. Default agrees with the MPAS met_intermediate.rs GFS path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SnowConvention { Wps, Raw }

impl SnowConvention {
    fn parse(value: &str) -> Result<Self, Box<dyn Error>> {
        match value {
            "wps" => Ok(Self::Wps),
            "raw" => Ok(Self::Raw),
            _ => Err("GPUWM_GFS_SNOW_CONVENTION must be wps or raw".into()),
        }
    }

    fn name(self) -> &'static str {
        match self { Self::Wps => "wps", Self::Raw => "raw" }
    }

    fn convert(self, name: &str, value: f64) -> f64 {
        // Native ungrib stores the decoded field in REAL before rrpr.
        // Retain its two FP32 operations for the derived depth.
        match (self, name) {
            (Self::Wps, "SNOW") => ((value as f32) * 2.0_f32) as f64,
            (Self::Wps, "SNOWH") => (((value as f32) * 2.0_f32) * 0.005_f32) as f64,
            _ => value,
        }
    }
}

fn bridge_threads() -> usize {
    env::var(THREADS_ENV)
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|count| count.get())
                .unwrap_or(1)
        })
}

/// An error carried out of a worker thread with the exact text the serial
/// bridge would have printed: `main` returns it, and the runtime prints
/// its `Debug`, which is the original error's `Debug`.
struct Relayed {
    debug: String,
    display: String,
}

impl Relayed {
    fn from_error(error: Box<dyn Error>) -> Self {
        Self { debug: format!("{error:?}"), display: error.to_string() }
    }
}

impl std::fmt::Debug for Relayed {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.debug)
    }
}

impl std::fmt::Display for Relayed {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.display)
    }
}

impl Error for Relayed {}

/// `work(0..count)` on up to [`bridge_threads`] threads, drained IN ORDER:
/// the answers come back in index order and the error returned is the
/// first one in index order -- the one the serial loop would have raised.
/// Indices are claimed in order and no new one is claimed after a
/// failure, so every index before a failing one has run.
fn in_index_order<T: Send>(
    count: usize,
    work: impl Fn(usize) -> Result<T, Box<dyn Error>> + Sync,
) -> Result<Vec<T>, Box<dyn Error>> {
    let threads = bridge_threads().min(count).max(1);
    let next = AtomicUsize::new(0);
    let failed = AtomicBool::new(false);
    let slots: Vec<Mutex<Option<Result<T, Relayed>>>> =
        (0..count).map(|_| Mutex::new(None)).collect();
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| loop {
                if failed.load(Ordering::Relaxed) {
                    break;
                }
                let index = next.fetch_add(1, Ordering::Relaxed);
                if index >= count {
                    break;
                }
                let outcome = work(index).map_err(Relayed::from_error);
                if outcome.is_err() {
                    failed.store(true, Ordering::Relaxed);
                }
                *slots[index].lock().unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(outcome);
            });
        }
    });
    let mut answers = Vec::with_capacity(count);
    for slot in slots {
        match slot.into_inner().unwrap_or_else(|poisoned| poisoned.into_inner()) {
            Some(Ok(answer)) => answers.push(answer),
            Some(Err(relayed)) => return Err(Box::new(relayed)),
            None => return Err("a series task was never run".into()),
        }
    }
    Ok(answers)
}

/// One forecast hour's arrays, written, with what the series-level
/// documents record about them.
struct HourOutput {
    manifest_rows: String,
    decoded_hashes: Vec<(u32, &'static str, u64, String)>,
    clamp_census: BTreeMap<&'static str, ClampTally>,
}

/// Decode and write one forecast hour's directory under `partial`.
fn write_hour(
    partial: &Path,
    input: &SeriesInput,
    inv: &Inventory,
    file: &Grib2File,
    all_water_source: bool,
    snow_convention: SnowConvention,
) -> Result<HourOutput, Box<dyn Error>> {
    use std::fmt::Write as _;
    let time_dir = partial.join(format!("f{:03}", input.hour));
    fs::create_dir(&time_dir)?;
    let mut output = HourOutput {
        manifest_rows: String::new(),
        decoded_hashes: Vec::new(),
        clamp_census: BTreeMap::new(),
    };
    for name in PRESSURE_SPECS
        .iter()
        .map(|v| v.name)
        .chain(SURFACE_SPECS.iter().map(|v| v.name))
        .chain(SOIL_SPECS.iter().map(|v| v.name))
    {
        let path = time_dir.join(format!("{name}.f32le"));
        let mut writer = BufWriter::new(File::create(&path)?);
        for selected in inv.selected.iter().filter(|field| field.name == name) {
            let message = &file.messages[selected.index];
            let summary = write_field(message, &mut writer, selected.spec, all_water_source, snow_convention)?;
            output
                .clamp_census
                .entry(name)
                .or_default()
                .merge(summary.clamped);
            writeln!(
                output.manifest_rows,
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\tf{:03}/{}.f32le",
                input.hour,
                selected.index,
                name,
                selected.spec.parameter.discipline,
                selected.spec.parameter.category,
                selected.spec.parameter.number,
                selected.level,
                message.product.level_type,
                message.data_rep.template,
                message.bitmap.is_some(),
                summary.finite,
                summary.missing,
                summary.minimum,
                summary.maximum,
                summary.clamped.clamps,
                summary.clamped.max_excursion,
                input.hour,
                name
            )?;
        }
        writer.flush()?;
        drop(writer);
        output.decoded_hashes.push((
            input.hour,
            name,
            path.metadata()?.len(),
            sha256_file(&path)?,
        ));
    }
    Ok(output)
}

/// A lead batch's record of what the whole-series checks compare, one row
/// per hour: the grid fingerprint, the selected inventory keys, the LAND
/// mask choice and the SHA-256 of each invariant field's decoded FP32 bits.
/// A series decode compares every hour with its first hour in memory; lead
/// batches decoded apart are compared here instead (`--merge-batches`), on
/// the same values, so a batched series refuses what the one-shot decode
/// of it refuses.
const BATCH_CONTRACT_NAME: &str = "batch-contract.tsv";
const BATCH_CONTRACT_HEADER: &str = "hour\tgrid\tselected\tland_mask_parameter\tinvariants";
const INVENTORY_HEADER: &str = "hour\tindex\tvariable\tdiscipline\tcategory\tparameter\tlevel_value\tlevel_type\tdrt\tbitmap\tfinite\tmissing\tminimum\tmaximum\tclamped\tmax_excursion\tfilename";
const DECODED_MANIFEST_HEADER: &str = "hour\tvariable\tbytes\tsha256\tfilename";
/// Gate rows that list one value per hour, joined across batches.
const GATE_SERIES_ROWS: [&str; 3] = ["forecast_hours", "forecast_generating_process_ids", "source_sha256"];
/// Gate rows derived from the whole series' receipts, written again.
const GATE_DERIVED_ROWS: [&str; 5] = [
    "bound_clamp_total",
    "bound_clamp_max_excursion",
    "bound_clamp_fields",
    "inventory_sha256",
    "decoded_manifest_sha256",
];

fn batch_contract_rows(
    inputs: &[SeriesInput],
    inventories: &[Inventory],
    loaded: &[(Grib2File, String)],
) -> Result<String, Box<dyn Error>> {
    let mut text = format!("{BATCH_CONTRACT_HEADER}\n");
    for ((input, inventory), (file, _)) in inputs.iter().zip(inventories).zip(loaded) {
        let selected = inventory
            .selected
            .iter()
            .map(|field| format!("{}:{:016x}", field.name, field.level.to_bits()))
            .collect::<Vec<_>>()
            .join(",");
        let mut invariants = Vec::new();
        for name in ["SOURCE_OROGRAPHY", "LANDSEA"] {
            let bits = decoded_f32_bits(file, inventory, name)?;
            let bytes: Vec<u8> = bits.iter().flat_map(|value| value.to_le_bytes()).collect();
            invariants.push(format!("{name}:{}", hex_sha256(&bytes)));
        }
        text.push_str(&format!(
            "{}\t{:?}\t{}\t{}\t{}\n",
            input.hour,
            inventory.grid,
            selected,
            inventory.land_mask_parameter,
            invariants.join(",")
        ));
    }
    Ok(text)
}

fn read_rows(path: &Path) -> Result<Vec<(String, String)>, Box<dyn Error>> {
    let mut rows = Vec::new();
    for raw in fs::read_to_string(path)?.lines() {
        let (key, value) = raw
            .split_once('\t')
            .ok_or_else(|| format!("{path:?} has a malformed row {raw:?}"))?;
        rows.push((key.to_string(), value.to_string()));
    }
    Ok(rows)
}

/// The body rows of a receipt table after its header, which must be `header`.
fn table_body(path: &Path, header: &str) -> Result<Vec<String>, Box<dyn Error>> {
    let text = fs::read_to_string(path)?;
    let mut lines = text.lines();
    if lines.next() != Some(header) {
        return Err(format!("{path:?} does not start with its header").into());
    }
    Ok(lines.map(str::to_string).collect())
}

fn gate_value<'a>(rows: &'a [(String, String)], key: &str) -> Result<&'a str, Box<dyn Error>> {
    rows.iter()
        .find(|(name, _)| name == key)
        .map(|(_, value)| value.as_str())
        .ok_or_else(|| format!("lead batch gate has no {key}").into())
}

/// Write the whole series' receipts (`gate.tsv`, `inventory.tsv`,
/// `decoded-sha256.tsv`) from lead batches this bridge decoded with
/// `--lead-batch`, byte for byte what one decode of the whole series
/// writes: every per-hour row is the hour's own, the per-hour gate lists
/// are joined in hour order, the clamp census is merged with the decode's
/// own `ClampTally::merge`, and the two digests are taken again.  Every
/// check the series decode makes across hours is made across the
/// batches first, in its words: the hours obey the series rules, and each
/// hour's grid, inventory, LAND mask choice and invariant fields equal the
/// first hour's.  The decoded arrays stay in their batch directories.
fn merge_batches(output: &Path, cycle: &str, batches: &[PathBuf]) -> Result<(), Box<dyn Error>> {
    if batches.is_empty() {
        return Err("--merge-batches needs at least one lead batch".into());
    }
    if output.exists() {
        return Err(format!("refusing to overwrite {output:?}").into());
    }
    let mut gates = Vec::new();
    let mut inventory_rows = Vec::new();
    let mut decoded_rows = Vec::new();
    let mut contract_rows = Vec::new();
    for batch in batches {
        let gate = read_rows(&batch.join("gate.tsv"))?;
        if gate_value(&gate, "status")? != "PASS" {
            return Err(format!("lead batch {batch:?} did not pass").into());
        }
        if gate_value(&gate, "cycle")? != cycle {
            return Err(format!("lead batch {batch:?} decoded another cycle").into());
        }
        for (name, key) in [
            ("inventory.tsv", "inventory_sha256"),
            ("decoded-sha256.tsv", "decoded_manifest_sha256"),
        ] {
            if sha256_file(&batch.join(name))? != gate_value(&gate, key)? {
                return Err(format!("lead batch {batch:?} {name} differs from its gate").into());
            }
        }
        inventory_rows.extend(table_body(&batch.join("inventory.tsv"), INVENTORY_HEADER)?);
        decoded_rows.extend(table_body(&batch.join("decoded-sha256.tsv"), DECODED_MANIFEST_HEADER)?);
        let contract = table_body(&batch.join(BATCH_CONTRACT_NAME), BATCH_CONTRACT_HEADER)?;
        let hours: Vec<String> = contract
            .iter()
            .map(|row| row.split('\t').next().unwrap_or("").to_string())
            .collect();
        if hours.join(",") != gate_value(&gate, "forecast_hours")? {
            return Err(format!("lead batch {batch:?} contract hours differ from its gate").into());
        }
        contract_rows.extend(contract);
        gates.push(gate);
    }
    let first = &gates[0];
    let keys: Vec<&str> = first.iter().map(|(key, _)| key.as_str()).collect();
    for (gate, batch) in gates.iter().zip(batches).skip(1) {
        let other: Vec<&str> = gate.iter().map(|(key, _)| key.as_str()).collect();
        if other != keys {
            return Err(format!("lead batch {batch:?} gate rows differ from the first batch's").into());
        }
    }
    let mut hours = Vec::new();
    for gate in &gates {
        for value in gate_value(gate, "forecast_hours")?.split(',') {
            hours.push(value.parse::<u32>()?);
        }
    }
    check_series_hours(&hours, false)?;
    // The cross-hour checks of the series decode, each hour against the
    // first, in the series decode's own words.
    let reference: Vec<&str> = contract_rows[0].split('\t').collect();
    if reference.len() != 5 {
        return Err("lead batch contract row is malformed".into());
    }
    for row in &contract_rows {
        let columns: Vec<&str> = row.split('\t').collect();
        if columns.len() != 5 {
            return Err("lead batch contract row is malformed".into());
        }
        let hour = columns[0].parse::<u32>()?;
        if columns[1] != reference[1] {
            return Err(format!("f{hour:03} grid differs from f000").into());
        }
        if columns[3] != reference[3] {
            return Err(format!("f{hour:03} LAND mask choice differs from f000").into());
        }
        if columns[2] != reference[2] {
            return Err(format!("f{hour:03} selected inventory differs from f000").into());
        }
        let expected: Vec<&str> = reference[4].split(',').collect();
        for (index, item) in columns[4].split(',').enumerate() {
            if expected.get(index) != Some(&item) {
                let name = item.split(':').next().unwrap_or("invariant field");
                return Err(format!("GFS {name} differs between f000 and f{hour:03}").into());
            }
        }
    }
    let mut census: BTreeMap<String, ClampTally> = BTreeMap::new();
    let mut total = 0usize;
    let mut worst = 0.0f64;
    for gate in &gates {
        total += gate_value(gate, "bound_clamp_total")?.parse::<usize>()?;
        worst = worst.max(gate_value(gate, "bound_clamp_max_excursion")?.parse::<f64>()?);
        let fields = gate_value(gate, "bound_clamp_fields")?;
        for item in fields.split(',').filter(|item| !item.is_empty()) {
            let parts: Vec<&str> = item.split(':').collect();
            if parts.len() != 3 {
                return Err(format!("lead batch clamp census entry {item:?} is malformed").into());
            }
            census.entry(parts[0].to_string()).or_default().merge(ClampTally {
                clamps: parts[1].parse::<usize>()?,
                max_excursion: parts[2].parse::<f64>()?,
            });
        }
    }
    let clamp_lines = clamp_receipt_lines_by_name(&census);
    if clamp_lines[0] != format!("bound_clamp_total\t{total}")
        || clamp_lines[1] != format!("bound_clamp_max_excursion\t{worst}")
    {
        return Err("lead batch clamp census totals disagree with their fields".into());
    }
    let parent = output.parent().unwrap_or_else(|| Path::new("."));
    let stem = output
        .file_name()
        .ok_or("OUTPUT_DIR has no filename")?
        .to_string_lossy();
    let partial = parent.join(format!(".{stem}.partial-{}", std::process::id()));
    if partial.exists() {
        return Err(format!("stale partial output exists: {partial:?}").into());
    }
    fs::create_dir(&partial)?;
    let written = (|| -> Result<(), Box<dyn Error>> {
        let mut inventory = format!("{INVENTORY_HEADER}\n");
        for row in &inventory_rows {
            inventory.push_str(row);
            inventory.push('\n');
        }
        fs::write(partial.join("inventory.tsv"), &inventory)?;
        let mut decoded = format!("{DECODED_MANIFEST_HEADER}\n");
        for row in &decoded_rows {
            decoded.push_str(row);
            decoded.push('\n');
        }
        fs::write(partial.join("decoded-sha256.tsv"), &decoded)?;
        let mut gate = String::new();
        for (key, value) in first {
            if GATE_SERIES_ROWS.contains(&key.as_str()) {
                let joined = gates
                    .iter()
                    .map(|batch| gate_value(batch, key).map(str::to_string))
                    .collect::<Result<Vec<_>, _>>()?
                    .join(",");
                gate.push_str(&format!("{key}\t{joined}\n"));
            } else if GATE_DERIVED_ROWS.contains(&key.as_str()) {
                continue;
            } else {
                for (other, batch) in gates.iter().zip(batches) {
                    if gate_value(other, key)? != value {
                        return Err(format!("lead batch {batch:?} {key} differs from the first batch's").into());
                    }
                }
                gate.push_str(&format!("{key}\t{value}\n"));
            }
        }
        for line in &clamp_lines {
            gate.push_str(line);
            gate.push('\n');
        }
        gate.push_str(&format!("inventory_sha256\t{}\n", hex_sha256(inventory.as_bytes())));
        gate.push_str(&format!("decoded_manifest_sha256\t{}\n", hex_sha256(decoded.as_bytes())));
        fs::write(partial.join("gate.tsv"), gate)?;
        Ok(())
    })();
    if let Err(error) = written {
        let _ = fs::remove_dir_all(&partial);
        return Err(error);
    }
    fs::rename(&partial, output)?;
    println!("PASS\t{}", output.display());
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    // Keep the release-provenance stamp in the binary: the cut
    // proves a staged bridge by these bytes (see lib.rs).
    let _ = std::hint::black_box(gpuwm_preprocess_cpu::SOURCE_REV_STAMP);
    let args: Vec<String> = env::args().skip(1).collect();
    let usage = "usage: gfs_grib2_bridge --series SERIES_TSV OUTPUT_DIR EXPECTED_CYCLE \
                 [--pressure-levels-pa PA,PA,...] [--lead-batch]\n       \
                 gfs_grib2_bridge --merge-batches OUTPUT_DIR EXPECTED_CYCLE BATCH_DIR...";
    if args.first().map(String::as_str) == Some("--merge-batches") {
        if args.len() < 4 {
            return Err(usage.into());
        }
        let cycle = args[2].trim_end_matches('Z').replace('T', " ");
        let batches: Vec<PathBuf> = args[3..].iter().map(PathBuf::from).collect();
        return merge_batches(Path::new(&args[1]), &cycle, &batches);
    }
    // `--lead-batch` goes last, so every argument vector this bridge took
    // before still means what it meant.
    let lead_batch = args.last().map(String::as_str) == Some("--lead-batch");
    let args: Vec<String> = if lead_batch {
        args[..args.len() - 1].to_vec()
    } else {
        args
    };
    if !(args.len() == 4 || args.len() == 6)
        || args[0] != "--series"
        || (args.len() == 6 && args[4] != "--pressure-levels-pa")
    {
        return Err(usage.into());
    }
    let snow_convention = SnowConvention::parse(
        &env::var("GPUWM_GFS_SNOW_CONVENTION").unwrap_or_else(|_| "wps".to_string()))?;
    let inputs = parse_series_rows(Path::new(&args[1]), lead_batch)?;
    let output = PathBuf::from(&args[2]);
    let cycle = args[3].trim_end_matches('Z').replace('T', " ");
    let requested_levels_csv = args.get(5).cloned();
    if output.exists() {
        return Err(format!("refusing to overwrite {output:?}").into());
    }

    // Read and parse every source exactly once.  All subsequent inventory,
    // invariance, and decode work uses these immutable parsed messages, so a
    // path replacement cannot change the bytes between validation and write.
    //
    // One object per task, several at once, answered in series order: the
    // objects are independent, and the read, the SHA-256 and the
    // parse of each one were the serial half of the bridge.
    let loaded: Vec<(Grib2File, String)> = in_index_order(inputs.len(), |index| {
        let bytes = fs::read(&inputs[index].path)?;
        let digest = hex_sha256(&bytes);
        let file = Grib2File::from_bytes(&bytes)?;
        Ok::<_, Box<dyn Error>>((file, digest))
    })?;
    // The ladder comes from the fetched files, not from a constant: a case
    // whose model top sits above 100 hPa fetches extra levels, and a bridge
    // with the ladder baked in would refuse exactly the file that was fetched
    // to satisfy it. Derived once from f000 and applied to every hour, so a
    // series whose hours disagree about their levels fails the per-hour
    // inventory checks below rather than decoding a ragged column.
    //
    // `--pressure-levels-pa` is the full-file route's declaration.  A NOMADS
    // subset carries exactly the requested ladder, so deriving it from the
    // file IS the request; a whole pgrb2.0p25 object carries every published
    // level up to 1 Pa, where real geopotential heights sit beyond this
    // bridge's certified plausibility range and no tornado-scale case has any
    // use for the mesosphere.  The caller that knows the requested ladder
    // (the fetch manifest records it) passes it here; the same
    // certified-tail/strictly-increasing shape is enforced either way, and a
    // requested level the file does not carry still refuses by name in the
    // per-hour inventory.
    let pressure_levels_pa = match requested_levels_csv {
        Some(csv) => requested_pressure_levels(&csv)?,
        None => observed_pressure_levels(&loaded[0].0)?,
    };
    let inventories: Vec<Inventory> = inputs
        .iter()
        .zip(&loaded)
        .map(|(input, (file, _))| {
            inventory(
                file,
                &cycle,
                input.hour,
                input.expected_forecast_process_id,
                &pressure_levels_pa,
                snow_convention,
            )
        })
        .collect::<Result<_, _>>()?;
    for (input, observed) in inputs.iter().zip(&inventories) {
        if observed.grid != inventories[0].grid {
            return Err(format!("f{:03} grid differs from f000", input.hour).into());
        }
        if observed.land_mask_parameter != inventories[0].land_mask_parameter {
            return Err(format!("f{:03} LAND mask choice differs from f000", input.hour).into());
        }
        let keys: Vec<(&str, u64)> = observed
            .selected
            .iter()
            .map(|field| (field.name, field.level.to_bits()))
            .collect();
        let reference: Vec<(&str, u64)> = inventories[0]
            .selected
            .iter()
            .map(|field| (field.name, field.level.to_bits()))
            .collect();
        if keys != reference {
            return Err(format!("f{:03} selected inventory differs from f000", input.hour).into());
        }
    }
    let mut invariant_fingerprints = Vec::new();
    for name in ["SOURCE_OROGRAPHY", "LANDSEA"] {
        let reference = decoded_f32_bits(&loaded[0].0, &inventories[0], name)?;
        for ((input, inventory), (file, _)) in inputs.iter().zip(&inventories).zip(&loaded).skip(1)
        {
            let candidate = decoded_f32_bits(file, inventory, name)?;
            require_exact_invariance(&reference, &candidate, name, input.hour)?;
        }
        invariant_fingerprints.push((name, fnv1a64(&reference)));
    }
    // LANDSEA is already proven byte-for-byte invariant across the series.
    // Only an entirely water crop may have no finite soil/snow support.
    let all_water_source = source_is_all_water(&loaded[0].0, &inventories[0])?;

    let parent = output.parent().unwrap_or_else(|| Path::new("."));
    let stem = output
        .file_name()
        .ok_or("OUTPUT_DIR has no filename")?
        .to_string_lossy();
    let partial = parent.join(format!(".{stem}.partial-{}", std::process::id()));
    if partial.exists() {
        return Err(format!("stale partial output exists: {partial:?}").into());
    }
    fs::create_dir(&partial)?;
    let generated = (|| -> Result<(), Box<dyn Error>> {
        let first_message = &loaded[0].0.messages[inventories[0].selected[0].index];
        let mut gate = BufWriter::new(File::create(partial.join("gate.tsv"))?);
        writeln!(gate, "status\tPASS")?;
        writeln!(gate, "schema\tgpuwm-gfs-grib2-bridge-v1")?;
        writeln!(gate, "cycle\t{cycle}")?;
        writeln!(
            gate,
            "forecast_hours\t{}",
            inputs
                .iter()
                .map(|v| v.hour.to_string())
                .collect::<Vec<_>>()
                .join(",")
        )?;
        writeln!(
            gate,
            "pressure_levels_pa\t{}",
            pressure_levels_pa
                .iter()
                .map(|v| format!("{v:.0}"))
                .collect::<Vec<_>>()
                .join(",")
        )?;
        writeln!(gate, "nx\t{}", first_message.grid.nx)?;
        writeln!(gate, "ny\t{}", first_message.grid.ny)?;
        // Normalized geometry: lat1 is the southernmost row of the
        // emitted arrays whichever way the source scanned, matching the
        // south-to-north output contract and the ascending latitude axis
        // gpuwm/gfs_direct.py builds from these four numbers.
        writeln!(gate, "lat1\t{}", southern_latitude(&first_message.grid))?;
        writeln!(gate, "lon1\t{}", first_message.grid.lon1)?;
        writeln!(gate, "lat2\t{}", northern_latitude(&first_message.grid))?;
        writeln!(gate, "lon2\t{}", first_message.grid.lon2)?;
        writeln!(gate, "dx\t{}", first_message.grid.dx)?;
        writeln!(gate, "dy\t{}", first_message.grid.dy)?;
        writeln!(
            gate,
            "num_data_points\t{}",
            first_message.grid.num_data_points
        )?;
        // The scan mode of the ARRAYS THIS BRIDGE WROTE, always
        // south-to-north; `source_scan_mode` records what arrived, so a
        // receipt still says whether a flip happened.
        writeln!(gate, "scan_mode\t0x{SOUTH_TO_NORTH_SCAN:02x}")?;
        writeln!(
            gate,
            "source_scan_mode\t0x{:02x}",
            first_message.grid.scan_mode
        )?;
        writeln!(
            gate,
            "scan_order_normalized\t{}",
            first_message.grid.scan_mode != SOUTH_TO_NORTH_SCAN
        )?;
        writeln!(gate, "originating_center\t7")?;
        writeln!(gate, "master_table_version\t2")?;
        writeln!(gate, "local_table_version\t1")?;
        writeln!(
            gate,
            "forecast_generating_process_ids\t{}",
            inputs
                .iter()
                .map(|input| { format!("{}:{}", input.hour, input.expected_forecast_process_id) })
                .collect::<Vec<_>>()
                .join(",")
        )?;
        writeln!(
            gate,
            "source_sha256\t{}",
            inputs
                .iter()
                .zip(&loaded)
                .map(|(input, (_, digest))| format!("{}:{digest}", input.hour))
                .collect::<Vec<_>>()
                .join(",")
        )?;
        writeln!(
            gate,
            "land_mask_parameter\t{}",
            inventories[0].land_mask_parameter
        )?;
        writeln!(gate, "snow_convention\t{}", snow_convention.name())?;
        writeln!(gate, "invariant_fields\tSOURCE_OROGRAPHY,LANDSEA")?;
        writeln!(
            gate,
            "invariant_fingerprint_fnv1a64\t{}",
            invariant_fingerprints
                .iter()
                .map(|(name, value)| format!("{name}:{value:016x}"))
                .collect::<Vec<_>>()
                .join(",")
        )?;
        let mut manifest = BufWriter::new(File::create(partial.join("inventory.tsv"))?);
        let mut decoded_hashes = Vec::new();
        // Per-field clamp census, summed over every hour and level, so a
        // receipt reader can see exactly which bounded fields were
        // kissed by the packing grid and by how much.
        let mut clamp_census: BTreeMap<&'static str, ClampTally> = BTreeMap::new();
        writeln!(manifest, "{INVENTORY_HEADER}")?;
        // Every forecast hour is its own directory and its own arrays, so
        // the hours are decoded and written concurrently; their manifest
        // rows, digests and clamp tallies are then taken IN HOUR ORDER, so
        // every document below is the serial bridge's byte for byte.
        let hours = in_index_order(inputs.len(), |index| {
            write_hour(&partial, &inputs[index], &inventories[index], &loaded[index].0,
                       all_water_source, snow_convention)
        })?;
        for hour in hours {
            manifest.write_all(hour.manifest_rows.as_bytes())?;
            decoded_hashes.extend(hour.decoded_hashes);
            for (name, tally) in hour.clamp_census {
                clamp_census.entry(name).or_default().merge(tally);
            }
        }
        manifest.flush()?;
        drop(manifest);
        if lead_batch {
            fs::write(
                partial.join(BATCH_CONTRACT_NAME),
                batch_contract_rows(&inputs, &inventories, &loaded)?,
            )?;
        }
        let inventory_sha256 = sha256_file(&partial.join("inventory.tsv"))?;
        let decoded_manifest_path = partial.join("decoded-sha256.tsv");
        let mut decoded_manifest = BufWriter::new(File::create(&decoded_manifest_path)?);
        writeln!(decoded_manifest, "{DECODED_MANIFEST_HEADER}")?;
        for (hour, name, bytes, digest) in &decoded_hashes {
            writeln!(
                decoded_manifest,
                "{hour}\t{name}\t{bytes}\t{digest}\tf{hour:03}/{name}.f32le"
            )?;
        }
        decoded_manifest.flush()?;
        drop(decoded_manifest);
        for line in clamp_receipt_lines(&clamp_census) {
            writeln!(gate, "{line}")?;
        }
        writeln!(gate, "inventory_sha256\t{inventory_sha256}")?;
        writeln!(
            gate,
            "decoded_manifest_sha256\t{}",
            sha256_file(&decoded_manifest_path)?
        )?;
        gate.flush()?;
        Ok(())
    })();
    if let Err(error) = generated {
        let _ = fs::remove_dir_all(&partial);
        return Err(error);
    }
    fs::rename(&partial, &output)?;
    println!("PASS\t{}", output.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;

    #[test]
    fn wps_snow_uses_weasd_for_both_outputs_and_preserves_missing() {
        let mut spec = spec_named("SNOW");
        for (name, expected) in [("SNOW", 24.0_f32), ("SNOWH", 24.0_f32 * 0.005_f32)] {
            spec.name = name;
            let mut bytes = Vec::new();
            let summary = emit_converted_values(
                &[12.0, f64::NAN], Some(&[true, false]), spec, 0.0,
                false, &mut bytes, SnowConvention::Wps).unwrap();
            assert_eq!(f32::from_le_bytes(bytes[0..4].try_into().unwrap()), expected);
            assert!(f32::from_le_bytes(bytes[4..8].try_into().unwrap()).is_nan());
            assert_eq!((summary.finite, summary.missing), (1, 1));
            assert_eq!(summary.maximum, expected as f64);
        }
        assert_eq!(SnowConvention::Raw.convert("SNOW", 12.0), 12.0);
        assert_eq!(SnowConvention::Raw.convert("SNOWH", 0.3), 0.3);
        assert!(SnowConvention::parse("unknown").is_err());
    }

    fn valid_message(parameter_number: u8) -> Grib2Message {
        Grib2Message {
            discipline: 2,
            identification: grib_core::grib2::Identification {
                center_id: 7,
                subcenter_id: 0,
                master_table_version: 2,
                local_table_version: 1,
                reference_time_significance: 1,
                production_status: 0,
                processed_data_type: 1,
            },
            reference_time: NaiveDate::from_ymd_opt(2026, 7, 20)
                .unwrap()
                .and_hms_opt(0, 0, 0)
                .unwrap(),
            grid: grib_core::grib2::GridDefinition {
                template: 0,
                nx: 2,
                ny: 2,
                lat1: 20.0,
                lon1: 250.0,
                lat2: 20.25,
                lon2: 250.25,
                dx: 0.25,
                dy: 0.25,
                scan_mode: 0x40,
                num_data_points: 4,
                shape_of_earth: 6,
                resolution_flags: 0x30,
                ..grib_core::grib2::GridDefinition::default()
            },
            product: grib_core::grib2::ProductDefinition {
                template: 0,
                parameter_category: 0,
                parameter_number,
                generating_process: 2,
                background_generating_process_id: 0,
                forecast_generating_process_id: 81,
                forecast_time: 0,
                time_range_unit: 1,
                level_type: SURFACE_LEVEL_TYPE,
                level_value: 0.0,
                ..grib_core::grib2::ProductDefinition::default()
            },
            data_rep: grib_core::grib2::DataRepresentation::default(),
            bitmap: None,
            raw_data: Vec::new(),
        }
    }

    /// The same 2x2 field as `valid_message`, published the other way
    /// up: `lat1` is the northern edge and the rows run south.
    fn north_to_south_message(parameter_number: u8) -> Grib2Message {
        let mut message = valid_message(parameter_number);
        message.grid.scan_mode = NORTH_TO_SOUTH_SCAN;
        message.grid.lat1 = 20.25;
        message.grid.lat2 = 20.0;
        message
    }

    #[test]
    fn both_published_scan_orders_are_accepted() {
        // NOMADS' subregion crop is 0x40; raw noaa-gfs-bdp-pds S3 (and
        // an uncropped grib-filter request) is 0x00.  Same numbers, both
        // orders, one bridge.
        validate_common(&valid_message(0), "2026-07-20 00:00:00", 0, 81, false).unwrap();
        validate_common(
            &north_to_south_message(0),
            "2026-07-20 00:00:00",
            0,
            81,
            false,
        )
        .unwrap();
    }

    #[test]
    fn a_latitude_span_that_contradicts_the_scan_direction_fails_closed() {
        // 0x00 with an ascending span, and 0x40 with a descending one:
        // each declares a row order its own endpoints deny.
        let mut ascending_north_to_south = valid_message(0);
        ascending_north_to_south.grid.scan_mode = NORTH_TO_SOUTH_SCAN;
        let error = validate_common(
            &ascending_north_to_south,
            "2026-07-20 00:00:00",
            0,
            81,
            false,
        )
        .unwrap_err()
        .to_string();
        assert!(
            error.contains("not a supported regular GFS grid"),
            "{error}"
        );

        let mut descending_south_to_north = north_to_south_message(0);
        descending_south_to_north.grid.scan_mode = SOUTH_TO_NORTH_SCAN;
        assert!(validate_common(
            &descending_south_to_north,
            "2026-07-20 00:00:00",
            0,
            81,
            false
        )
        .is_err());
    }

    #[test]
    fn every_other_scan_mode_still_fails_closed() {
        // i-negative, j-consecutive, alternating rows, and combinations:
        // none of these are a row flip away from the output contract.
        for scan_mode in [0x80u8, 0x20, 0x10, 0x60, 0xc0, 0x50, 0x08] {
            let mut message = valid_message(0);
            message.grid.scan_mode = scan_mode;
            let outcome = validate_common(&message, "2026-07-20 00:00:00", 0, 81, false);
            assert!(
                outcome.is_err(),
                "scan mode 0x{scan_mode:02x} must not be accepted"
            );
        }
    }

    #[test]
    fn the_latitude_helpers_report_the_true_edges_either_way() {
        let south_up = valid_message(0);
        assert_eq!(southern_latitude(&south_up.grid), 20.0);
        assert_eq!(northern_latitude(&south_up.grid), 20.25);
        let north_up = north_to_south_message(0);
        assert_eq!(southern_latitude(&north_up.grid), 20.0);
        assert_eq!(northern_latitude(&north_up.grid), 20.25);
    }

    #[test]
    fn the_mask_flip_tracks_the_value_flip_exactly() {
        // The bitmap is indexed by the same flat position as the values,
        // so the two must move together or a north-to-south field would
        // attribute its missing cells to the wrong rows.
        let (nx, ny) = (3usize, 4usize);
        let mut values: Vec<f64> = (0..(nx * ny)).map(|value| value as f64).collect();
        let mut mask: Vec<bool> = (0..(nx * ny)).map(|index| index % 3 == 0).collect();
        flip_rows(&mut values, nx, ny);
        flip_mask_rows(&mut mask, nx, ny);
        for (position, value) in values.iter().enumerate() {
            assert_eq!(
                mask[position],
                (*value as usize).is_multiple_of(3),
                "mask and values disagree at {position}"
            );
        }
        // Row order really did reverse.
        assert_eq!(&values[..nx], &[9.0, 10.0, 11.0]);
    }

    #[test]
    fn the_mask_flip_leaves_degenerate_shapes_alone() {
        let mut single = vec![true];
        flip_mask_rows(&mut single, 1, 1);
        assert_eq!(single, vec![true]);
        let mut mismatched = vec![true, false];
        flip_mask_rows(&mut mismatched, 3, 4);
        assert_eq!(mismatched, vec![true, false]);
    }

    /// The two committed fixtures: the same GFS field, same cycle, from
    /// the two publishers, in the two row orders.
    ///
    /// * `s3-raw-tmp2m-...` -- one message lifted by `.idx` byte range
    ///   from `noaa-gfs-bdp-pds`: global 1440x721, `scan 0x00`, `lat1`
    ///   the north pole.
    /// * `nomads-crop-...` -- the NOMADS grib-filter `subregion` crop of
    ///   the same object over 30..40N, 260..270E: 41x41, `scan 0x40`,
    ///   `lat1` the southern edge.
    fn scan_order_fixture(name: &str) -> Grib2File {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/gfs-scan-order")
            .join(name);
        Grib2File::open(path.to_str().expect("fixture path is UTF-8"))
            .unwrap_or_else(|error| panic!("{}: {error}", path.display()))
    }

    fn all_missing_ocean_fixture() -> Grib2Message {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/gfs-all-water/gfs-20260909t00z-f000-tsoil-ocean.grib2");
        let bytes = fs::read(path).unwrap();
        assert_eq!(hex_sha256(&bytes), "c336254c3adac4689ff670d2618a8b5770d6469dfcbf8d5565d28299e9faeff6");
        let mut file = Grib2File::from_bytes(&bytes).unwrap();
        assert_eq!(file.messages.len(), 1);
        file.messages.remove(0)
    }

    #[test]
    fn historical_embedded_missing_soil_keeps_its_mask_and_required_fields_refuse() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/gfs-archive/soil-embedded-missing-2017091812.grib2");
        let bytes = fs::read(path).unwrap();
        assert_eq!(hex_sha256(&bytes),
            "5cc4fc37e3ecfca0a51690ff0f88d2dc831ac0a738e1ad8dbd27794f29626169");
        let file = Grib2File::from_bytes(&bytes).unwrap();
        let message = &file.messages[0];
        assert_eq!(message.data_rep.missing_value_management, 1);
        assert!(message.bitmap.is_none());
        validate_common(message, "2017-09-18 12:00:00", 0, 81, true).unwrap();
        let (values, present) = scan_normalized(message).unwrap();
        let mask = present.unwrap();
        assert_eq!(values.len(), 1_038_240);
        assert_eq!(mask.iter().filter(|cell| !**cell).count(), 687_174);
        assert!(values.iter().zip(&mask).all(|(value, present)| value.is_finite() == *present));
        let summary = emit_values(&values, Some(&mask), SOIL_SPECS[0], 0.0,
                                  &mut Vec::new()).unwrap();
        assert_eq!((summary.finite, summary.missing), (351_066, 687_174));
        let mut required = SOIL_SPECS[0];
        required.allow_missing = false;
        assert!(emit_values(&values, Some(&mask), required, 0.0, &mut Vec::new()).is_err());
        assert!(emit_values(&values, None, SOIL_SPECS[0], 0.0, &mut Vec::new()).is_err());
    }

    #[test]
    fn all_missing_ocean_record_preserves_full_grid_and_row_window_missing_values() {
        let message = all_missing_ocean_fixture();
        assert_eq!(message.grid.num_data_points, 43584);
        assert_eq!(message.data_rep.section5_num_data_points, 0);
        assert_eq!(message.data_rep.template, 0);
        assert!(message.raw_data.is_empty());
        let (values, present) = scan_normalized(&message).unwrap();
        assert_eq!(values.len(), 43584);
        assert!(values.iter().all(|value| value.is_nan()));
        assert!(present.unwrap().iter().all(|value| !value));
        let window = grib_core::grib2::unpack_message_scan_normalized_row_window(&message, 2, 4).unwrap();
        assert_eq!(window.len(), 2 * message.grid.nx as usize);
        assert!(window.iter().all(|value| value.is_nan()));
    }

    #[test]
    fn all_missing_ocean_record_does_not_relax_corrupt_bitmap_or_payload_checks() {
        for change in 0..5 {
            let mut message = all_missing_ocean_fixture();
            match change {
                0 => message.bitmap = None,
                1 => { message.bitmap.as_mut().unwrap().pop(); },
                2 => message.bitmap.as_mut().unwrap()[0] = true,
                3 => message.raw_data.push(0),
                _ => message.data_rep.template = 40,
            }
            assert!(unpack_message(&message).is_err(), "corruption {change}");
            assert!(grib_core::grib2::unpack_message_scan_normalized_row_window(&message, 0, 1).is_err(), "window corruption {change}");
        }
    }

    #[test]
    fn all_missing_soil_is_emitted_only_with_confirmed_water_only_source() {
        let message = all_missing_ocean_fixture();
        let (values, present) = scan_normalized(&message).unwrap();
        let mut bytes = Vec::new();
        assert!(emit_values(&values, present.as_deref(), SOIL_SPECS[0], 0.0, &mut bytes).is_err());
        bytes.clear();
        let summary = emit_values_with_surface_support(&values, present.as_deref(), SOIL_SPECS[0], 0.0, true, &mut bytes).unwrap();
        assert_eq!((summary.finite, summary.missing), (0, 43584));
        assert!(summary.minimum.is_nan() && summary.maximum.is_nan());
        assert_eq!(bytes.len(), 4 * 43584);
        assert!(bytes.as_chunks::<4>().0.iter().all(|chunk| f32::from_le_bytes(*chunk).is_nan()));
        assert!(emit_values_with_surface_support(&values, present.as_deref(), PRESSURE_SPECS[0], 0.0, true, &mut Vec::new()).is_err());
        let mut incorrect = present.unwrap(); incorrect[0] = true;
        assert!(emit_values_with_surface_support(&values, Some(&incorrect), SOIL_SPECS[0], 0.0, true, &mut Vec::new()).is_err());
    }

    /// A real NOMADS subregion crop over central Nebraska: 2x2 cells, all
    /// land, so the crop carries no bitmap on any record, soil and snow
    /// included.  Provenance beside it in tests/fixtures/gfs-all-land.
    fn all_land_crop_fixture() -> Grib2File {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/gfs-all-land/gfs-20260927t12z-f000-inland-crop.grib2");
        let bytes = fs::read(&path).unwrap_or_else(|error| panic!("{}: {error}", path.display()));
        assert_eq!(
            hex_sha256(&bytes),
            "a7f47bf20d4e5c51e5c65c769285fd4e2ce8b226449229d0a009d7b8e0e1e979"
        );
        Grib2File::from_bytes(&bytes).unwrap()
    }

    #[test]
    fn a_field_that_may_have_missing_cells_is_admitted_without_a_bitmap() {
        // Permission for missing cells is not a requirement to have some.
        validate_common(&valid_message(0), "2026-07-20 00:00:00", 0, 81, true).unwrap();
        let mut marked = valid_message(0);
        marked.bitmap = Some(vec![true, false, true, true]);
        validate_common(&marked, "2026-07-20 00:00:00", 0, 81, true).unwrap();
        // A field that may not have missing cells still refuses a bitmap.
        let refusal = validate_common(&marked, "2026-07-20 00:00:00", 0, 81, false)
            .unwrap_err()
            .to_string();
        assert!(refusal.contains("missing-value policy mismatch"), "{refusal}");
    }

    #[test]
    fn a_missing_cell_without_a_bitmap_is_still_refused() {
        // Admitting the bitmap's absence admits no undeclared missing cell.
        let mut bytes = Vec::new();
        let refusal = emit_values(
            &[0.25, f64::NAN, 0.25, 0.25],
            None,
            spec_named("GFS_SM000010"),
            0.0,
            &mut bytes,
        )
        .unwrap_err()
        .to_string();
        assert!(refusal.contains("NaN outside an explicit missing bitmap cell"), "{refusal}");
    }

    #[test]
    fn an_all_land_nomads_crop_decodes_its_soil_and_snow() {
        let file = all_land_crop_fixture();
        let levels = observed_pressure_levels(&file).unwrap();
        let selected = inventory(&file, "2026-09-27 12:00:00", 0, 81, &levels, SnowConvention::Raw).unwrap();
        assert!(!source_is_all_water(&file, &selected).unwrap());
        let mut names = Vec::new();
        for field in selected.selected.iter().filter(|field| field.spec.allow_missing) {
            let message = &file.messages[field.index];
            assert!(message.bitmap.is_none(), "{} carries a bitmap", field.name);
            let (values, present) = scan_normalized(message).unwrap();
            let mut bytes = Vec::new();
            let summary = emit_values_with_surface_support(
                &values,
                present.as_deref(),
                field.spec,
                field_quantum(message),
                false,
                &mut bytes,
            )
            .unwrap();
            assert_eq!((summary.finite, summary.missing), (4, 0), "{}", field.name);
            assert_eq!(bytes.len(), 4 * 4);
            names.push(field.name);
        }
        names.sort_unstable();
        assert_eq!(
            names,
            [
                "GFS_SM000010", "GFS_SM010040", "GFS_SM040100", "GFS_SM100200",
                "GFS_ST000010", "GFS_ST010040", "GFS_ST040100", "GFS_ST100200",
                "SNOW", "SNOWH",
            ]
        );
    }

    #[test]
    fn all_water_source_requires_decoded_land_mask_codes_and_accepts_legacy_water() {
        for (parameter, code, expected) in [(192, 0.0, true), (192, 1.0, false), (0, 2.0, true), (0, 1.0, false)] {
            let mut message = valid_message(parameter);
            message.data_rep.section5_num_data_points = 4;
            message.data_rep.reference_value = code;
            let mut spec = *SURFACE_SPECS.iter().find(|spec| spec.name == "LANDSEA").unwrap();
            spec.parameter.number = parameter;
            let inv = Inventory { grid: GridFingerprint::from_grid(&message.grid), land_mask_parameter: parameter,
                selected: vec![Selected { index: 0, name: "LANDSEA", level: 0.0, spec }] };
            let file = Grib2File { messages: vec![message] };
            assert_eq!(source_is_all_water(&file, &inv).unwrap(), expected);
        }
    }

    fn gdas_process_fixture(name: &str) -> (Vec<u8>, Grib2File) {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/gdas-process-id")
            .join(name);
        let bytes = fs::read(&path).unwrap_or_else(|error| panic!("{}: {error}", path.display()));
        let file = Grib2File::from_bytes(&bytes)
            .unwrap_or_else(|error| panic!("{}: {error}", path.display()));
        (bytes, file)
    }

    #[test]
    fn real_gdas_forecast_corpus_certifies_only_declared_process_96() {
        let cases = [
            (
                0,
                81,
                "nomads-gdas-20260729t12z-f000.grib2",
                "046ff2dade6e08e921e18c0ad75dccc4e43d1f8dc47c550cfa9be9d3323d2afe",
            ),
            (
                3,
                96,
                "nomads-gdas-20260729t12z-f003.grib2",
                "afa3f6b83c159ae7cfb2ce9425c36aa003a8163e8edc91911c79b776b5d4919a",
            ),
            (
                6,
                96,
                "nomads-gdas-20260729t12z-f006.grib2",
                "581bcf5d8bce2abd39392d449013d46c52ba9b60b732b5585d6a5922eb47227c",
            ),
            // The published endpoint itself.  Without this row the
            // corpus stopped at f006 while the fetch gate advertised
            // f009, so the last certified hour rested on the ladder
            // constant rather than on bytes.
            (
                9,
                96,
                "nomads-gdas-20260729t12z-f009.grib2",
                "1df03c6114a039dc4ffdfe96974c216b0311834e629d5966c424805dfe0b7f5f",
            ),
        ];
        for (hour, process_id, name, digest) in cases {
            let (bytes, file) = gdas_process_fixture(name);
            assert_eq!(hex_sha256(&bytes), digest);
            assert_eq!(file.messages.len(), 124);
            assert!(file.messages.iter().all(|message| {
                let signature_matches = message.identification.center_id == 7
                    && message.identification.master_table_version == 2
                    && message.identification.local_table_version == 1
                    && message.product.template == 0
                    && message.product.generating_process == 2
                    && message.product.background_generating_process_id == 0
                    && message.product.forecast_generating_process_id == process_id
                    && message.product.forecast_time == hour
                    && message.data_rep.template == 0
                    && message.grid.scan_mode == SOUTH_TO_NORTH_SCAN
                    && message.grid.shape_of_earth == 6
                    && same_level(message.grid.dx, 0.25)
                    && same_level(message.grid.dy, 0.25);
                signature_matches
                    && validate_common(
                        message,
                        "2026-07-29 12:00:00",
                        hour,
                        process_id,
                        message.bitmap.is_some(),
                    )
                    .is_ok()
            }));

            if process_id == 96 {
                let refusal = validate_common(
                    &file.messages[0],
                    "2026-07-29 12:00:00",
                    hour,
                    81,
                    file.messages[0].bitmap.is_some(),
                )
                .unwrap_err()
                .to_string();
                assert!(refusal.contains("forecast_process=96 (expected 81)"));
            }
        }
    }

    #[test]
    fn a_flipped_raw_s3_decode_is_bit_identical_to_the_nomads_crop() {
        // This is the whole justification for accepting scan 0x00: the
        // two publishers carry the same numbers in opposite row order,
        // so normalizing the order is lossless.  Compared as raw f64
        // bit patterns, not with a tolerance -- an epsilon here would
        // hide exactly the re-encoding difference worth knowing about.
        let raw = scan_order_fixture("s3-raw-tmp2m-20260729t18z-f000.grib2");
        let crop = scan_order_fixture("nomads-crop-20260729t18z-f000.grib2");
        let global = &raw.messages[0];
        assert_eq!(global.grid.scan_mode, NORTH_TO_SOUTH_SCAN);
        let cropped = crop
            .messages
            .iter()
            .find(|message| {
                message.product.level_type == 103 && same_level(message.product.level_value, 2.0)
            })
            .expect("the crop carries TMP at 2 m above ground");
        assert_eq!(cropped.grid.scan_mode, SOUTH_TO_NORTH_SCAN);

        // The NOMADS crop passes the whole gate, as it always has.
        validate_common(cropped, &cropped.reference_time.to_string(), 0, 81, false).unwrap();

        // The raw S3 object passes the WHOLE gate too: its row order
        // was certified first, and its DRT 5.3 packing later earned its
        // own real-product proof (the SOILW matched pair below, which
        // is where the missing-value semantics the 5.3 gate was waiting
        // on are pinned).  This fixture is complex-packed with no
        // bitmap; the SOILW pair is the bitmap case.
        assert_eq!(global.data_rep.template, 3);
        assert_eq!(global.data_rep.missing_value_management, 0);
        assert!(global.bitmap.is_none());
        validate_common(global, &global.reference_time.to_string(), 0, 81, false).unwrap();

        let (global_values, _) = scan_normalized(global).unwrap();
        let (crop_values, _) = scan_normalized(cropped).unwrap();
        let (gnx, gny) = (global.grid.nx as usize, global.grid.ny as usize);
        let (cnx, cny) = (cropped.grid.nx as usize, cropped.grid.ny as usize);

        // Where the crop sits inside the normalized global array, read
        // off the geometry rather than hardcoded -- both are now
        // south-first, so the offsets are a straight subtraction.
        let row_offset = ((southern_latitude(&cropped.grid) - southern_latitude(&global.grid))
            / global.grid.dy)
            .round() as usize;
        let column_offset =
            ((cropped.grid.lon1 - global.grid.lon1) / global.grid.dx).round() as usize;
        assert!(row_offset + cny <= gny && column_offset + cnx <= gnx);

        let mut compared = 0usize;
        for row in 0..cny {
            for column in 0..cnx {
                let expected = global_values[(row_offset + row) * gnx + column_offset + column];
                let observed = crop_values[row * cnx + column];
                assert_eq!(
                    observed.to_bits(),
                    expected.to_bits(),
                    "row {row} column {column}: crop {observed} != flipped raw {expected}"
                );
                compared += 1;
            }
        }
        assert_eq!(compared, cnx * cny);
        assert_eq!(compared, 41 * 41);
    }

    #[test]
    fn the_raw_s3_fixture_would_be_upside_down_without_the_flip() {
        // A negative control for the test above: comparing the crop
        // against the *unflipped* storage order must fail, or the
        // comparison proves nothing about row order.
        let raw = scan_order_fixture("s3-raw-tmp2m-20260729t18z-f000.grib2");
        let crop = scan_order_fixture("nomads-crop-20260729t18z-f000.grib2");
        let global = &raw.messages[0];
        let cropped = crop
            .messages
            .iter()
            .find(|message| {
                message.product.level_type == 103 && same_level(message.product.level_value, 2.0)
            })
            .unwrap();
        let stored = unpack_message(global).unwrap();
        let (crop_values, _) = scan_normalized(cropped).unwrap();
        let gnx = global.grid.nx as usize;
        let (cnx, cny) = (cropped.grid.nx as usize, cropped.grid.ny as usize);
        // Storage order: row 0 is the north pole, so the crop's southern
        // row sits this far down instead.
        let row_offset = ((northern_latitude(&global.grid) - northern_latitude(&cropped.grid))
            / global.grid.dy)
            .round() as usize;
        let column_offset =
            ((cropped.grid.lon1 - global.grid.lon1) / global.grid.dx).round() as usize;
        let mismatches = (0..cny)
            .flat_map(|row| (0..cnx).map(move |column| (row, column)))
            .filter(|(row, column)| {
                stored[(row_offset + row) * gnx + column_offset + column].to_bits()
                    != crop_values[row * cnx + column].to_bits()
            })
            .count();
        assert!(
            mismatches > cnx * cny / 2,
            "an unflipped comparison should disagree almost everywhere, saw \
             {mismatches} mismatches of {}",
            cnx * cny
        );
    }

    #[test]
    fn the_raw_53_bitmap_decode_matches_the_nomads_crop_cell_for_cell() {
        // THE missing-value proof the DRT 5.3 gate was waiting on.  The
        // TMP pair proves value arithmetic; it carries no bitmap.  This
        // pair is the same SOILW 0-0.1 m field from the same cycle in
        // both published packings -- raw S3 (DRT 5.3, north-to-south,
        // bitmap over water) and the NOMADS crop (DRT 5.0,
        // south-to-north, same bitmap semantics) -- and admission
        // stands on two exact statements over the overlap: the missing
        // cells land in IDENTICAL positions, and every present cell
        // decodes to the IDENTICAL f64 bit pattern.  No tolerance on
        // either statement.
        let raw = scan_order_fixture("s3-raw-soilw-20260729t18z-f000.grib2");
        let crop = scan_order_fixture("nomads-crop-soilw-20260729t18z-f000.grib2");
        let global = &raw.messages[0];
        let cropped = &crop.messages[0];

        // The raw record is exactly the published envelope the gate
        // admits -- and nothing wider.
        assert_eq!(global.grid.scan_mode, NORTH_TO_SOUTH_SCAN);
        assert_eq!(global.data_rep.template, 3);
        assert_eq!(global.data_rep.group_splitting_method, 1);
        assert_eq!(global.data_rep.missing_value_management, 0);
        assert_eq!(global.data_rep.spatial_diff_order, 2);
        assert_eq!(global.data_rep.spatial_diff_bytes, 2);
        assert!(global.bitmap.is_some());
        assert_eq!(cropped.grid.scan_mode, SOUTH_TO_NORTH_SCAN);
        assert_eq!(cropped.data_rep.template, 0);
        assert!(cropped.bitmap.is_some());

        // Both pass the whole gate as the bitmap-carrying soil field
        // they are.
        validate_common(global, &global.reference_time.to_string(), 0, 81, true).unwrap();
        validate_common(cropped, &cropped.reference_time.to_string(), 0, 81, true).unwrap();

        let (global_values, global_present) = scan_normalized(global).unwrap();
        let (crop_values, crop_present) = scan_normalized(cropped).unwrap();
        let global_present = global_present.expect("raw SOILW carries a bitmap");
        let crop_present = crop_present.expect("cropped SOILW carries a bitmap");
        let (gnx, gny) = (global.grid.nx as usize, global.grid.ny as usize);
        let (cnx, cny) = (cropped.grid.nx as usize, cropped.grid.ny as usize);

        let row_offset = ((southern_latitude(&cropped.grid) - southern_latitude(&global.grid))
            / global.grid.dy)
            .round() as usize;
        let column_offset =
            ((cropped.grid.lon1 - global.grid.lon1) / global.grid.dx).round() as usize;
        assert!(row_offset + cny <= gny && column_offset + cnx <= gnx);

        let mut compared = 0usize;
        let mut missing = 0usize;
        for row in 0..cny {
            for column in 0..cnx {
                let flat_global = (row_offset + row) * gnx + column_offset + column;
                let flat_crop = row * cnx + column;
                assert_eq!(
                    global_present[flat_global], crop_present[flat_crop],
                    "row {row} column {column}: the two packings disagree on \
                     whether the cell is missing"
                );
                if global_present[flat_global] {
                    assert_eq!(
                        crop_values[flat_crop].to_bits(),
                        global_values[flat_global].to_bits(),
                        "row {row} column {column}: crop {} != flipped raw {}",
                        crop_values[flat_crop],
                        global_values[flat_global]
                    );
                    compared += 1;
                } else {
                    assert!(global_values[flat_global].is_nan());
                    assert!(crop_values[flat_crop].is_nan());
                    missing += 1;
                }
            }
        }
        assert_eq!(compared + missing, cnx * cny);
        assert_eq!(compared + missing, 41 * 41);
        // The overlap genuinely exercises both branches: the box was
        // chosen over the Gulf coast so it holds land AND water.
        assert!(compared > 0, "no present cells compared");
        assert!(missing > 0, "no missing cells compared; the pair no longer \
                              demonstrates bitmap semantics");
    }

    /// A file carrying the certified ladder plus ``extra`` levels, minus
    /// ``omit``, with all five 3-D fields on every level it keeps.
    fn pressure_ladder_file(extra: &[f64], omit: &[f64]) -> Grib2File {
        let mut messages = Vec::new();
        let levels: Vec<f64> = CERTIFIED_PRESSURE_LEVELS_PA
            .iter()
            .copied()
            .chain(extra.iter().copied())
            .filter(|level| !omit.iter().any(|dropped| same_level(*dropped, *level)))
            .collect();
        for level in levels {
            for spec in PRESSURE_SPECS {
                let mut message = valid_message(spec.parameter.number);
                message.discipline = spec.parameter.discipline;
                message.product.parameter_category = spec.parameter.category;
                message.product.parameter_number = spec.parameter.number;
                message.product.level_type = PRESSURE_LEVEL_TYPE;
                message.product.level_value = level;
                messages.push(message);
            }
        }
        Grib2File { messages }
    }

    #[test]
    fn frozen_gfs_inventory_cardinality() {
        assert_eq!(
            PRESSURE_SPECS.len() * CERTIFIED_PRESSURE_LEVELS_PA.len()
                + SURFACE_SPECS.len()
                + SOIL_SPECS.len(),
            124
        );
    }

    /// The selected-record count is a function of the ladder.
    ///
    /// A live 23-level decode caught this as a hardcoded `!= 124`: the fetch
    /// was right, the ladder derivation was right, and the bridge refused its
    /// own correctly decoded 134 records at the last cardinality check. The
    /// arithmetic is asserted at both ladder lengths so the constant cannot
    /// creep back.
    #[test]
    fn the_selected_record_count_follows_the_ladder() {
        let single_level = SURFACE_SPECS.len() + SOIL_SPECS.len();
        assert_eq!(single_level, 19);
        for (levels, expected) in [(21usize, 124usize), (23, 134), (28, 159), (41, 224)] {
            assert_eq!(
                PRESSURE_SPECS.len() * levels + single_level,
                expected,
                "{levels} isobaric levels"
            );
        }
    }

    #[test]
    fn pressure_levels_are_strictly_increasing() {
        assert!(CERTIFIED_PRESSURE_LEVELS_PA
            .windows(2)
            .all(|pair| pair[0] < pair[1]));
    }

    /// `--pressure-levels-pa` is how the full-file route keeps a
    /// whole-globe object from dragging the mesosphere into the decode:
    /// the request's ladder wins over the file's, under the same shape
    /// rule, and levels the file lacks still refuse in the inventory.
    #[test]
    fn a_requested_ladder_is_parsed_and_held_to_the_certified_shape() {
        let certified_csv = CERTIFIED_PRESSURE_LEVELS_PA
            .iter()
            .map(|level| format!("{level}"))
            .collect::<Vec<_>>()
            .join(",");
        assert_eq!(
            requested_pressure_levels(&certified_csv).expect("certified ladder"),
            CERTIFIED_PRESSURE_LEVELS_PA.to_vec()
        );
        let extended = format!("5000,7000,{certified_csv}");
        assert_eq!(
            requested_pressure_levels(&extended).expect("extended ladder")[..2],
            [5_000.0, 7_000.0]
        );

        let dropped = requested_pressure_levels(
            &certified_csv.replace("50000,", ""))
            .expect_err("a dropped certified level must refuse")
            .to_string();
        assert!(dropped.contains("certified") || dropped.contains("fewer than"),
                "{dropped}");
        assert!(requested_pressure_levels("")
            .unwrap_err()
            .to_string()
            .contains("not a number"));
        assert!(requested_pressure_levels(&format!("{certified_csv},nonsense"))
            .unwrap_err()
            .to_string()
            .contains("not a number"));
        assert!(requested_pressure_levels(&format!("-5,{certified_csv}"))
            .unwrap_err()
            .to_string()
            .contains("not a positive pressure"));
    }

    /// The requested ladder drives selection: a file carrying MORE
    /// complete levels than the request decodes exactly the request,
    /// and a request the file cannot fill refuses by name.
    #[test]
    fn a_fuller_file_decodes_only_the_requested_ladder() {
        // The file carries two levels above the certified top, the way
        // a whole pgrb2.0p25 object carries everything up to 1 Pa.
        let file = pressure_ladder_file(&[5_000.0, 7_000.0], &[]);
        let certified: Vec<f64> = CERTIFIED_PRESSURE_LEVELS_PA.to_vec();
        let mut messages = file.messages;
        for spec in SURFACE_SPECS.iter().chain(SOIL_SPECS.iter()) {
            if spec.name == "LANDSEA" {
                messages.push(valid_message(218));
                continue;
            }
            let mut message = valid_message(spec.parameter.number);
            message.discipline = spec.parameter.discipline;
            message.product.parameter_category = spec.parameter.category;
            message.product.parameter_number = spec.parameter.number;
            message.product.level_type = spec.level_type;
            message.product.level_value = spec.level_value;
            if spec.level_type == SOIL_LEVEL_TYPE {
                message.product.second_level_type = SOIL_LEVEL_TYPE;
                message.product.second_level_value = match spec.level_value {
                    value if same_level(value, 0.0) => 0.1,
                    value if same_level(value, 0.1) => 0.4,
                    value if same_level(value, 0.4) => 1.0,
                    _ => 2.0,
                };
            }
            if spec.allow_missing {
                message.bitmap = Some(vec![true; 4]);
            }
            messages.push(message);
        }
        let file = Grib2File { messages };

        let selected = inventory(&file, "2026-07-20 00:00:00", 0, 81, &certified, SnowConvention::Raw)
            .expect("the certified request against a fuller file");
        assert_eq!(
            selected.selected.len(),
            PRESSURE_SPECS.len() * certified.len() + SURFACE_SPECS.len() + SOIL_SPECS.len()
        );
        assert!(selected
            .selected
            .iter()
            .all(|field| !same_level(field.level, 5_000.0)
                && !same_level(field.level, 7_000.0)));

        let mut too_deep = certified.clone();
        too_deep.insert(0, 3_000.0);
        let refusal = inventory(&file, "2026-07-20 00:00:00", 0, 81, &too_deep, SnowConvention::Raw)
            .expect_err("a level the file lacks must refuse")
            .to_string();
        assert!(refusal.contains("missing required GFS field"), "{refusal}");
        assert!(refusal.contains("3000"), "{refusal}");
    }

    /// A ladder with 50 and 70 hPa on top is what a p_top=5000 Pa case
    /// fetches, and the bridge must decode it rather than refuse it.
    #[test]
    fn an_upward_extended_ladder_is_accepted() {
        let file = pressure_ladder_file(&[5_000.0, 7_000.0], &[]);
        let observed = observed_pressure_levels(&file).expect("extended ladder");
        assert_eq!(observed.len(), CERTIFIED_PRESSURE_LEVELS_PA.len() + 2);
        assert_eq!(observed[0], 5_000.0);
        assert_eq!(observed[1], 7_000.0);
        assert_eq!(
            &observed[2..],
            &CERTIFIED_PRESSURE_LEVELS_PA[..],
            "the certified ladder must survive underneath the extension"
        );
    }

    #[test]
    fn the_certified_ladder_alone_is_accepted_unchanged() {
        let file = pressure_ladder_file(&[], &[]);
        assert_eq!(
            observed_pressure_levels(&file).expect("certified ladder"),
            CERTIFIED_PRESSURE_LEVELS_PA.to_vec()
        );
    }

    /// A level short of the certified ladder is incompleteness, not a
    /// shallower request: refuse rather than decode one level short.
    #[test]
    fn a_ladder_missing_a_certified_level_is_refused() {
        let file = pressure_ladder_file(&[], &[50_000.0]);
        let message = observed_pressure_levels(&file)
            .expect_err("a missing certified level must refuse")
            .to_string();
        assert!(
            message.contains("certified") || message.contains("fewer than"),
            "{message}"
        );
    }

    /// Four fifths of a level is not a level: a hole in one field must not
    /// become a decoded array with a hole in it.
    #[test]
    fn a_level_missing_one_field_is_not_a_level() {
        let mut file = pressure_ladder_file(&[5_000.0], &[]);
        // Drop RH at 5000 Pa only.
        let rh = PRESSURE_SPECS
            .iter()
            .find(|spec| spec.name == "RH")
            .expect("RH spec")
            .parameter;
        file.messages.retain(|m| {
            !(matches_parameter(m, rh)
                && m.product.level_type == PRESSURE_LEVEL_TYPE
                && same_level(m.product.level_value, 5_000.0))
        });
        assert_eq!(
            observed_pressure_levels(&file).expect("incomplete top level dropped"),
            CERTIFIED_PRESSURE_LEVELS_PA.to_vec()
        );
    }

    /// An "extra" level below the certified top is a different ladder, not an
    /// extension of this one.
    #[test]
    fn an_interior_extra_level_is_refused() {
        let file = pressure_ladder_file(&[12_500.0], &[]);
        let message = observed_pressure_levels(&file)
            .expect_err("an interior extra level must refuse")
            .to_string();
        assert!(message.contains("certified"), "{message}");
    }

    #[test]
    fn certified_product_and_quarter_degree_grid_fail_closed() {
        let message = valid_message(0);
        validate_common(&message, "2026-07-20 00:00:00", 0, 81, false).unwrap();

        let mut wrong_center = message.clone();
        wrong_center.identification.center_id = 8;
        assert!(
            validate_common(&wrong_center, "2026-07-20 00:00:00", 0, 81, false)
                .unwrap_err()
                .to_string()
                .contains("certified operational NCEP GFS")
        );

        let mut wrong_process = message.clone();
        wrong_process.product.forecast_generating_process_id = 82;
        assert!(
            validate_common(&wrong_process, "2026-07-20 00:00:00", 0, 81, false)
                .unwrap_err()
                .to_string()
                .contains("instantaneous")
        );

        let mut forecast = message.clone();
        forecast.product.forecast_generating_process_id = 96;
        forecast.product.forecast_time = 3;
        validate_common(&forecast, "2026-07-20 00:00:00", 3, 96, false).unwrap();
        forecast.product.forecast_generating_process_id = 81;
        assert!(
            validate_common(&forecast, "2026-07-20 00:00:00", 3, 96, false)
                .unwrap_err()
                .to_string()
                .contains("expected 96")
        );

        let mut wrong_resolution = message.clone();
        wrong_resolution.grid.dx = 0.5;
        assert!(
            validate_common(&wrong_resolution, "2026-07-20 00:00:00", 0, 81, false)
                .unwrap_err()
                .to_string()
                .contains("regular GFS grid")
        );

        let mut wrong_endpoint = message;
        wrong_endpoint.grid.lon2 = 251.0;
        assert!(
            validate_common(&wrong_endpoint, "2026-07-20 00:00:00", 0, 81, false)
                .unwrap_err()
                .to_string()
                .contains("regular GFS grid")
        );

        // DRT 5.3 is admitted only inside the envelope NCEP publishes
        // and the committed matched pairs prove: general group
        // splitting, declared missing-value modes, second-order
        // spatial differencing.  Each departure refuses by name, and
        // plain complex packing (5.2) has no proof corpus at all.
        let raw_s3_form = |message: &mut Grib2Message| {
            message.data_rep.template = 3;
            message.data_rep.group_splitting_method = 1;
            message.data_rep.missing_value_management = 0;
            message.data_rep.spatial_diff_order = 2;
            message.data_rep.spatial_diff_bytes = 2;
        };
        let mut certified_53 = valid_message(0);
        raw_s3_form(&mut certified_53);
        validate_common(&certified_53, "2026-07-20 00:00:00", 0, 81, false).unwrap();

        let mut plain_complex = valid_message(0);
        raw_s3_form(&mut plain_complex);
        plain_complex.data_rep.template = 2;
        assert!(
            validate_common(&plain_complex, "2026-07-20 00:00:00", 0, 81, false)
                .unwrap_err()
                .to_string()
                .contains("unsupported GFS DRT 5.2")
        );

        let mut embedded_missing = valid_message(0);
        raw_s3_form(&mut embedded_missing);
        embedded_missing.data_rep.missing_value_management = 1;
        for mode in [1, 2] {
            embedded_missing.data_rep.missing_value_management = mode;
            validate_common(&embedded_missing, "2026-07-20 00:00:00", 0, 81, false).unwrap();
        }
        embedded_missing.data_rep.missing_value_management = 3;
        assert!(validate_common(&embedded_missing, "2026-07-20 00:00:00", 0, 81, false)
                .unwrap_err().to_string().contains("missing-value management 3"));

        let mut foreign_splitting = valid_message(0);
        raw_s3_form(&mut foreign_splitting);
        foreign_splitting.data_rep.group_splitting_method = 0;
        assert!(
            validate_common(&foreign_splitting, "2026-07-20 00:00:00", 0, 81, false)
                .unwrap_err()
                .to_string()
                .contains("group splitting method 0")
        );

        let mut first_order = valid_message(0);
        raw_s3_form(&mut first_order);
        first_order.data_rep.spatial_diff_order = 1;
        assert!(
            validate_common(&first_order, "2026-07-20 00:00:00", 0, 81, false)
                .unwrap_err()
                .to_string()
                .contains("spatial differencing order 1")
        );

        let mut degenerate_descriptor = valid_message(0);
        raw_s3_form(&mut degenerate_descriptor);
        degenerate_descriptor.data_rep.spatial_diff_bytes = 0;
        assert!(
            validate_common(&degenerate_descriptor, "2026-07-20 00:00:00", 0, 81, false)
                .unwrap_err()
                .to_string()
                .contains("descriptor width 0")
        );

        let mut oversized_packing = valid_message(0);
        oversized_packing.data_rep.bits_per_value = 64;
        assert!(
            validate_common(&oversized_packing, "2026-07-20 00:00:00", 0, 81, false)
                .unwrap_err()
                .to_string()
                .contains("bits_per_value 64")
        );
    }

    #[test]
    fn sha256_padding_boundaries_and_streamed_file_keep_standard_words() {
        let cases = [
            (0, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
            (55, "463eb28e72f82e0a96c0a4cc53690c571281131f672aa229e0d45ae59b598b59"),
            (56, "da2ae4d6b36748f2a318f23e7ab1dfdf45acdc9d049bd80e59de82a60895f562"),
            (63, "29af2686fd53374a36b0846694cc342177e428d1647515f078784d69cdb9e488"),
            (64, "fdeab9acf3710362bd2658cdc9a29e8f9c757fcf9811603a8c447cd1d9151108"),
            (65, "4bfd2c8b6f1eec7a2afeb48b934ee4b2694182027e6d0fc075074f2fabb31781"),
            (65553, "db321256cd80da245f26f881e6b51650d0c95881f47d47e0e8c51a455b66af34"),
        ];
        let path = env::temp_dir().join(format!("gpuwm-gfs-sha-{}", std::process::id()));
        for (length, expected) in cases {
            let bytes: Vec<_> = (0..length).map(|index| (index % 251) as u8).collect();
            assert_eq!(hex_sha256(&bytes), expected);
            fs::write(&path, &bytes).unwrap();
            assert_eq!(sha256_file(&path).unwrap(), expected);
        }
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn sha256_matches_the_standard_known_answer() {
        assert_eq!(
            hex_sha256(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn simple_packing_rejects_short_and_extra_section7_payloads() {
        let mut message = valid_message(0);
        message.data_rep = grib_core::grib2::DataRepresentation {
            template: 0,
            bits_per_value: 8,
            section5_num_data_points: 4,
            ..grib_core::grib2::DataRepresentation::default()
        };
        message.raw_data = vec![1, 2, 3];
        assert!(unpack_message(&message)
            .unwrap_err()
            .to_string()
            .contains("requires 4"));
        message.raw_data = vec![1, 2, 3, 4, 5];
        assert!(unpack_message(&message)
            .unwrap_err()
            .to_string()
            .contains("requires 4"));
        message.raw_data = vec![1, 2, 3, 4];
        assert_eq!(unpack_message(&message).unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn landn_is_preferred_and_legacy_land_is_an_explicit_fallback() {
        let legacy = valid_message(0);
        let modern = valid_message(218);
        assert_eq!(
            land_mask_match(std::slice::from_ref(&legacy)).unwrap(),
            (0, 0)
        );
        assert_eq!(
            land_mask_match(&[legacy.clone(), modern.clone()]).unwrap(),
            (1, 218)
        );
        assert!(land_mask_match(&[modern.clone(), modern])
            .unwrap_err()
            .to_string()
            .contains("duplicate GFS LANDN"));
        assert!(land_mask_match(&[legacy.clone(), legacy])
            .unwrap_err()
            .to_string()
            .contains("duplicate GFS LAND"));

        let legacy_spec = SURFACE_SPECS
            .iter()
            .find(|field| field.name == "LANDSEA")
            .copied()
            .unwrap();
        assert_eq!(normalize_for_output(2.0, legacy_spec, 0.0).unwrap(), 0.0);
        assert_eq!(normalize_for_output(1.0, legacy_spec, 0.0).unwrap(), 1.0);
        assert!(normalize_for_output(0.5, legacy_spec, 0.0).is_err());
        let mut modern_spec = legacy_spec;
        modern_spec.parameter.number = 218;
        assert!(normalize_for_output(2.0, modern_spec, 0.0).is_err());
    }

    #[test]
    fn surface_humidity_contract_uses_wps_gfs_rh2_identifier() {
        let rh2 = SURFACE_SPECS
            .iter()
            .find(|field| field.name == "RH2")
            .unwrap();
        assert_eq!(
            rh2.parameter,
            Parameter {
                discipline: 0,
                category: 1,
                number: 1,
            }
        );
        assert_eq!(rh2.level_type, HEIGHT_LEVEL_TYPE);
        assert_eq!(rh2.level_value, 2.0);
        assert!(!SURFACE_SPECS.iter().any(|field| field.name == "D2"));
    }

    #[test]
    fn invariant_surface_contract_rejects_a_one_cell_change() {
        let reference = [0.0f32.to_bits(), 1.0f32.to_bits(), 2.0f32.to_bits()];
        require_exact_invariance(&reference, &reference, "LANDSEA", 3).unwrap();
        let changed = [0.0f32.to_bits(), 0.0f32.to_bits(), 2.0f32.to_bits()];
        assert!(require_exact_invariance(&reference, &changed, "LANDSEA", 3)
            .unwrap_err()
            .to_string()
            .contains("differs between f000 and f003"));
    }

    #[test]
    fn truncated_complex_payload_cannot_be_zero_padded() {
        let mut message = valid_message(0);
        message.grid.num_data_points = 1;
        message.grid.nx = 1;
        message.grid.ny = 1;
        message.data_rep = grib_core::grib2::DataRepresentation {
            template: 2,
            bits_per_value: 8,
            num_groups: 1,
            group_width_bits: 1,
            group_length_ref: 1,
            group_length_inc: 1,
            last_group_length: 1,
            group_length_bits: 1,
            section5_num_data_points: 1,
            ..grib_core::grib2::DataRepresentation::default()
        };
        message.raw_data.clear();
        assert!(unpack_message(&message)
            .unwrap_err()
            .to_string()
            .contains("truncated packed payload"));
    }

    #[test]
    fn zero_length_or_unterminated_grib_envelopes_fail_without_scanning() {
        let mut envelope = vec![0u8; 20];
        envelope[0..4].copy_from_slice(b"GRIB");
        envelope[7] = 2;
        assert!(Grib2File::from_bytes(&envelope)
            .unwrap_err()
            .to_string()
            .contains("below the 20-byte minimum"));

        envelope[8..16].copy_from_slice(&20u64.to_be_bytes());
        assert!(Grib2File::from_bytes(&envelope)
            .unwrap_err()
            .to_string()
            .contains("lacks 7777"));

        let mut prefixed = b"junk".to_vec();
        prefixed.extend_from_slice(&envelope);
        assert!(Grib2File::from_bytes(&prefixed)
            .unwrap_err()
            .to_string()
            .contains("non-GRIB bytes"));

        let mut section1 = vec![0u8; 21];
        section1[0..4].copy_from_slice(&21u32.to_be_bytes());
        section1[4] = 1;
        section1[5..7].copy_from_slice(&7u16.to_be_bytes());
        section1[9] = 2;
        section1[10] = 1;
        section1[11] = 1;
        section1[12..14].copy_from_slice(&2026u16.to_be_bytes());
        section1[14] = 7;
        section1[15] = 20;
        section1[20] = 1;
        let mut missing_fields = vec![0u8; 16];
        missing_fields[0..4].copy_from_slice(b"GRIB");
        missing_fields[7] = 2;
        missing_fields.extend_from_slice(&section1);
        missing_fields.extend_from_slice(b"7777");
        let missing_length = missing_fields.len() as u64;
        missing_fields[8..16].copy_from_slice(&missing_length.to_be_bytes());
        assert!(Grib2File::from_bytes(&missing_fields)
            .unwrap_err()
            .to_string()
            .contains("missing required Sections"));

        let mut early_marker = missing_fields;
        early_marker.extend_from_slice(b"7777");
        let early_length = early_marker.len() as u64;
        early_marker[8..16].copy_from_slice(&early_length.to_be_bytes());
        assert!(Grib2File::from_bytes(&early_marker)
            .unwrap_err()
            .to_string()
            .contains("not at the parsed envelope boundary"));
    }

    /// A GFS record's packing: reference plus `X * 2^-19`, scaled by
    /// `10^-3`.  One step of that grid is 1.9073486328125e-9, which is
    /// exactly the distance the field report's soil-moisture value sat
    /// above saturation.
    fn reported_packing() -> grib_core::grib2::DataRepresentation {
        grib_core::grib2::DataRepresentation {
            template: 0,
            binary_scale: -19,
            decimal_scale: 3,
            ..grib_core::grib2::DataRepresentation::default()
        }
    }

    fn reported_quantum() -> f64 {
        let mut message = valid_message(0);
        message.data_rep = reported_packing();
        field_quantum(&message)
    }

    fn spec_named(name: &str) -> FieldSpec {
        PRESSURE_SPECS
            .iter()
            .chain(SURFACE_SPECS.iter())
            .chain(SOIL_SPECS.iter())
            .find(|spec| spec.name == name)
            .copied()
            .unwrap_or_else(|| panic!("no spec named {name}"))
    }

    #[test]
    fn saturated_soil_moisture_clamps_instead_of_refusing() {
        // The externally reported blocker, to the digit:
        //   GFS_SM010040 value 1.0000000019073487 outside [0,1]
        // A cell whose soil is physically saturated is encoded at 1.0 and
        // decodes one packing step above it.  It is now clamped, counted
        // and reported -- not read as corruption.
        let spec = spec_named("GFS_SM010040");
        let quantum = reported_quantum();
        assert_eq!(quantum, 1.9073486328125e-9);
        let mut sink = Vec::new();
        let summary = emit_values(
            &[0.0, 0.25, 1.0000000019073487, 1.0],
            None,
            spec,
            quantum,
            &mut sink,
        )
        .unwrap();
        assert_eq!(summary.finite, 4);
        assert_eq!(summary.clamped.clamps, 1);
        // One packing step, as far as f64 can resolve a step that small
        // beside 1.0 -- comfortably inside the offer at that bound.
        assert_eq!(summary.clamped.max_excursion, 1.0000000019073487f64 - 1.0);
        assert!(summary.clamped.max_excursion <= spec.bounds().high_tolerance(quantum));
        // The emitted value is the bound itself, not the excursion.
        assert_eq!(summary.maximum, 1.0);
        let emitted: Vec<f32> = sink
            .as_chunks::<4>().0.iter()
            .map(|bytes| f32::from_le_bytes(*bytes))
            .collect();
        assert_eq!(emitted, vec![0.0f32, 0.25, 1.0, 1.0]);
    }

    #[test]
    fn a_genuinely_impossible_soil_moisture_still_refuses() {
        let spec = spec_named("GFS_SM010040");
        let quantum = reported_quantum();
        let error = emit_values(&[0.5, 1.05], None, spec, quantum, &mut Vec::new())
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("GFS_SM010040 value 1.05 outside [0,1]"),
            "{error}"
        );
        // The refusal shows its own arithmetic: how far out it was, and
        // how far out it would have tolerated.
        assert!(error.contains("by 0.05"), "{error}");
        assert!(error.contains("quantization tolerance 0.000001"), "{error}");
        // The gate has an edge, not a slope: just past the offer refuses.
        let offer = spec.bounds().high_tolerance(quantum);
        assert!(emit_values(
            &[0.5, 1.0 + offer * 1.000001],
            None,
            spec,
            quantum,
            &mut Vec::new()
        )
        .is_err());
        assert!(emit_values(&[0.5, 1.0 + offer], None, spec, quantum, &mut Vec::new()).is_ok());
    }

    #[test]
    fn a_slack_sanity_bound_is_not_widened_by_the_same_packing() {
        // Relative humidity's zero is physical; its 110 % ceiling is a
        // plausibility range.  Same record, same quantum, opposite
        // verdicts.
        let spec = spec_named("RH");
        let quantum = reported_quantum();
        let summary = emit_values(&[-quantum, 50.0], None, spec, quantum, &mut Vec::new()).unwrap();
        assert_eq!(summary.clamped.clamps, 1);
        assert_eq!(summary.minimum, 0.0);
        assert_eq!(spec.bounds().high_tolerance(quantum), 0.0);
        assert!(emit_values(
            &[50.0, 110.0 + quantum],
            None,
            spec,
            quantum,
            &mut Vec::new()
        )
        .is_err());
    }

    #[test]
    fn every_bounded_physical_field_survives_a_bound_kissing_cell() {
        // The class, not the instance.  Each field declaring a physical
        // end is fed a value exactly one offered tolerance outside that
        // end and must clamp; the roster is pinned so a newly added
        // bounded field has to state which of its ends is physical
        // rather than inherit silence.
        let quantum = reported_quantum();
        let mut roster = Vec::new();
        for spec in PRESSURE_SPECS
            .iter()
            .chain(SURFACE_SPECS.iter())
            .chain(SOIL_SPECS.iter())
            .copied()
        {
            let bounds = spec.bounds();
            if spec.minimum_kind == BoundKind::Physical {
                roster.push(format!("{}:minimum", spec.name));
            }
            if spec.maximum_kind == BoundKind::Physical {
                roster.push(format!("{}:maximum", spec.name));
            }
            if spec.name == "LANDSEA" {
                // A code field, not a range: its quantization fix lives
                // in the discretizer and is asserted on its own below.
                continue;
            }
            if spec.minimum_kind == BoundKind::Physical {
                let summary = emit_values(
                    &[spec.minimum - bounds.low_tolerance(quantum), spec.maximum],
                    None,
                    spec,
                    quantum,
                    &mut Vec::new(),
                )
                .unwrap_or_else(|error| panic!("{} low bound: {error}", spec.name));
                assert_eq!(summary.clamped.clamps, 1, "{}", spec.name);
            }
            if spec.maximum_kind == BoundKind::Physical {
                let summary = emit_values(
                    &[spec.minimum, spec.maximum + bounds.high_tolerance(quantum)],
                    None,
                    spec,
                    quantum,
                    &mut Vec::new(),
                )
                .unwrap_or_else(|error| panic!("{} high bound: {error}", spec.name));
                assert_eq!(summary.clamped.clamps, 1, "{}", spec.name);
            }
        }
        assert_eq!(
            roster,
            vec![
                "RH:minimum",
                "SNOW:minimum",
                "SNOWH:minimum",
                "RH2:minimum",
                "LANDSEA:minimum",
                "LANDSEA:maximum",
                "XICE:minimum",
                "XICE:maximum",
                "GFS_SM000010:minimum",
                "GFS_SM000010:maximum",
                "GFS_SM010040:minimum",
                "GFS_SM010040:maximum",
                "GFS_SM040100:minimum",
                "GFS_SM040100:maximum",
                "GFS_SM100200:minimum",
                "GFS_SM100200:maximum",
            ]
        );
    }

    #[test]
    fn the_land_mask_codes_are_recognized_to_the_same_tolerance() {
        // LANDSEA is a code field, and the same packing grid that pushed
        // soil moisture past saturation pushes a land cell off its code.
        let spec = spec_named("LANDSEA");
        let quantum = reported_quantum();
        let mut modern = spec;
        modern.parameter.number = 218;
        assert_eq!(
            normalize_for_output(1.0000000019073487, modern, quantum).unwrap(),
            1.0
        );
        assert_eq!(normalize_for_output(-quantum, modern, quantum).unwrap(), 0.0);
        // Still exactly two codes: half land is not a code.
        assert!(normalize_for_output(0.5, modern, quantum).is_err());
    }

    #[test]
    fn the_receipt_names_the_clamped_fields_and_their_worst_excursion() {
        let mut census: BTreeMap<&'static str, ClampTally> = BTreeMap::new();
        // A field that decoded clean must not appear in the field list.
        census.insert("GFS_SM000010", ClampTally::default());
        census.insert(
            "GFS_SM010040",
            ClampTally {
                clamps: 41,
                max_excursion: 1.9073486328125e-9,
            },
        );
        census.insert(
            "XICE",
            ClampTally {
                clamps: 7,
                max_excursion: 5.0e-10,
            },
        );
        let lines = clamp_receipt_lines(&census);
        assert_eq!(lines[0], "bound_clamp_total\t48");
        assert_eq!(
            lines[1]
                .strip_prefix("bound_clamp_max_excursion\t")
                .unwrap()
                .parse::<f64>()
                .unwrap(),
            1.9073486328125e-9
        );
        let fields: Vec<&str> = lines[2]
            .strip_prefix("bound_clamp_fields\t")
            .unwrap()
            .split(',')
            .collect();
        assert_eq!(fields.len(), 2);
        assert!(fields[0].starts_with("GFS_SM010040:41:"), "{fields:?}");
        assert_eq!(
            fields[0].rsplit(':').next().unwrap().parse::<f64>().unwrap(),
            1.9073486328125e-9
        );
        assert!(fields[1].starts_with("XICE:7:"), "{fields:?}");
        // A clean run still says so, in the same three lines.
        assert_eq!(
            clamp_receipt_lines(&BTreeMap::new()),
            vec![
                "bound_clamp_total\t0".to_string(),
                "bound_clamp_max_excursion\t0".to_string(),
                "bound_clamp_fields\t".to_string(),
            ]
        );
    }

    /// One scratch directory of series rows, and the parse of them.
    fn parsed_series(label: &str, rows: &[(&str, &str)]) -> Result<Vec<SeriesInput>, String> {
        let root = env::temp_dir().join(format!(
            "gpuwm-gfs-series-{}-{}",
            std::process::id(),
            label
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        let mut text = String::new();
        for (line, name) in rows {
            fs::write(root.join(name), b"GRIB").unwrap();
            text.push_str(line);
            text.push('\n');
        }
        let series = root.join("series.tsv");
        fs::write(&series, text).unwrap();
        let outcome = parse_series(&series).map_err(|error| error.to_string());
        fs::remove_dir_all(&root).unwrap();
        outcome
    }

    #[test]
    fn a_lead_batch_may_carry_one_lead_and_a_series_may_not() {
        let root = env::temp_dir().join(format!("gpuwm-gfs-batch-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("f003.grib2"), b"GRIB").unwrap();
        let series = root.join("series.tsv");
        fs::write(&series, "3\tf003.grib2\t96\n").unwrap();
        let batch = parse_series_rows(&series, true).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(batch[0].hour, 3);
        let whole = parse_series_rows(&series, false).map_err(|error| error.to_string());
        assert_eq!(whole.unwrap_err(), "GFS series must contain at least two times");
        fs::write(&series, "").unwrap();
        let empty = parse_series_rows(&series, true).map_err(|error| error.to_string());
        assert_eq!(empty.unwrap_err(), "GFS series must contain at least two times");
        fs::remove_dir_all(&root).unwrap();
    }

    fn write_batch(root: &Path, name: &str, hours: &[u32], landsea: &str, clamps: &str) -> PathBuf {
        let dir = root.join(name);
        fs::create_dir_all(&dir).unwrap();
        let mut inventory = format!("{INVENTORY_HEADER}\n");
        let mut decoded = format!("{DECODED_MANIFEST_HEADER}\n");
        let mut contract = format!("{BATCH_CONTRACT_HEADER}\n");
        for hour in hours {
            inventory.push_str(&format!("{hour}\t0\tT\t0\t0\t0\t50000\t100\t0\t255\t4\t0\t1\t2\t0\t0\tf{hour:03}/T.f32le\n"));
            decoded.push_str(&format!("{hour}\tT\t16\t{:064x}\tf{hour:03}/T.f32le\n", hour));
            contract.push_str(&format!("{hour}\tGrid\tT:0\t0\tSOURCE_OROGRAPHY:aa,LANDSEA:{landsea}\n"));
        }
        fs::write(dir.join("inventory.tsv"), &inventory).unwrap();
        fs::write(dir.join("decoded-sha256.tsv"), &decoded).unwrap();
        fs::write(dir.join(BATCH_CONTRACT_NAME), &contract).unwrap();
        let (total, worst) = if clamps.is_empty() {
            (0, "0".to_string())
        } else {
            let parts: Vec<&str> = clamps.split(':').collect();
            (parts[1].parse::<usize>().unwrap(), parts[2].to_string())
        };
        let list = |f: &dyn Fn(u32) -> String| hours.iter().map(|h| f(*h)).collect::<Vec<_>>().join(",");
        let gate = format!(
            "status\tPASS\nschema\tgpuwm-gfs-grib2-bridge-v1\ncycle\t2026-09-30 00:00:00\nforecast_hours\t{}\n\
             pressure_levels_pa\t50000\nnx\t2\nforecast_generating_process_ids\t{}\nsource_sha256\t{}\n\
             bound_clamp_total\t{total}\nbound_clamp_max_excursion\t{worst}\nbound_clamp_fields\t{clamps}\n\
             inventory_sha256\t{}\ndecoded_manifest_sha256\t{}\n",
            list(&|h| h.to_string()),
            list(&|h| format!("{h}:{}", if h == 0 { 81 } else { 96 })),
            list(&|h| format!("{h}:s{h}")),
            hex_sha256(inventory.as_bytes()),
            hex_sha256(decoded.as_bytes()),
        );
        fs::write(dir.join("gate.tsv"), gate).unwrap();
        dir
    }

    #[test]
    fn merged_lead_batches_are_the_series_receipts_and_refuse_what_it_refuses() {
        let root = env::temp_dir().join(format!("gpuwm-gfs-merge-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        let whole = write_batch(&root, "whole", &[0, 1, 2], "bb", "T:3:0.25");
        let first = write_batch(&root, "b0", &[0], "bb", "T:1:0.125");
        let second = write_batch(&root, "b1", &[1, 2], "bb", "T:2:0.25");
        let merged = root.join("merged");
        merge_batches(&merged, "2026-09-30 00:00:00", &[first.clone(), second.clone()]).unwrap();
        for name in ["gate.tsv", "inventory.tsv", "decoded-sha256.tsv"] {
            assert_eq!(
                fs::read_to_string(merged.join(name)).unwrap(),
                fs::read_to_string(whole.join(name)).unwrap(),
                "{name}"
            );
        }
        // A LAND field that moved between batches is the series decode's refusal.
        let moved = write_batch(&root, "b2", &[1, 2], "cc", "");
        let refused = merge_batches(&root.join("refused"), "2026-09-30 00:00:00", &[first.clone(), moved])
            .map_err(|error| error.to_string());
        assert_eq!(refused.unwrap_err(), "GFS LANDSEA differs between f000 and f001");
        // Batches that skip a lead are not a series.
        let gap = write_batch(&root, "b3", &[4], "bb", "");
        let refused = merge_batches(&root.join("gap"), "2026-09-30 00:00:00", &[first, second, gap])
            .map_err(|error| error.to_string());
        assert_eq!(refused.unwrap_err(), "GFS series cadence must be positive and uniform");
        fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn a_series_may_begin_at_a_forecast_lead() {
        // The defect: this used to be "GFS series must contain at least
        // two times and begin at f000", so a user who wanted a window
        // deep in the forecast had to decode from f000 to reach it.
        let parsed = parsed_series(
            "lead",
            &[
                ("18\tf018.grib2\t96", "f018.grib2"),
                ("21\tf021.grib2\t96", "f021.grib2"),
                ("24\tf024.grib2\t96", "f024.grib2"),
            ],
        )
        .unwrap();
        assert_eq!(
            parsed.iter().map(|input| input.hour).collect::<Vec<_>>(),
            vec![18, 21, 24]
        );
        // Every lead row still declares process 96 explicitly: a lead is
        // never relabelled as an analysis, and 96 is never inferred.
        assert!(parsed
            .iter()
            .all(|input| input.expected_forecast_process_id == 96));

        // A two-column lead row still declares only the analysis
        // capability, so it fails closed at decode rather than being
        // promoted to a forecast product here.
        let legacy = parsed_series(
            "legacy",
            &[
                ("18\tf018.grib2", "f018.grib2"),
                ("21\tf021.grib2", "f021.grib2"),
            ],
        )
        .unwrap();
        assert!(legacy
            .iter()
            .all(|input| input.expected_forecast_process_id == 81));

        // f000 still must say 81, and the cadence rules are untouched.
        assert!(parsed_series("f000", &[("0\tf000.grib2\t96", "f000.grib2")])
            .unwrap_err()
            .contains("analysis process ID 81"));
        assert!(parsed_series(
            "single",
            &[("18\tf018.grib2\t96", "f018.grib2")]
        )
        .unwrap_err()
        .contains("at least two times"));
        let subsample = parsed_series(
            "cadence",
            &[
                ("18\tf018.grib2\t96", "f018.grib2"),
                ("20\tf020.grib2\t96", "f020.grib2"),
            ]
        ).unwrap();
        assert_eq!(subsample.iter().map(|input| input.hour).collect::<Vec<_>>(), vec![18, 20]);
        assert!(parsed_series(
            "horizon",
            &[
                ("381\tf381.grib2\t96", "f381.grib2"),
                ("385\tf385.grib2\t96", "f385.grib2"),
            ]
        )
        .unwrap_err()
        .contains("f384"));
    }
    #[test]
    fn supported_uniform_subsets_preserve_their_source_leads() {
        for (index, leads) in [
            vec![0, 2, 4, 6], vec![0, 6, 12], vec![1, 3, 5],
            vec![5, 9, 13], vec![120, 126, 132], vec![123, 135, 147],
            vec![119, 123], vec![117, 123], vec![360, 384],
        ].iter().enumerate() {
            let owned: Vec<(String, String)> = leads.iter().map(|lead| {
                let name = format!("f{lead:03}.grib2");
                (format!("{lead}\t{name}\t{}", if *lead == 0 { 81 } else { 96 }), name)
            }).collect();
            let rows: Vec<(&str, &str)> = owned.iter().map(|(line, name)| (line.as_str(), name.as_str())).collect();
            let parsed = parsed_series(&format!("subset-{index}"), &rows).unwrap();
            assert_eq!(parsed.iter().map(|input| input.hour).collect::<Vec<_>>(), *leads);
        }
    }

    #[test]
    fn uniform_spacing_does_not_admit_an_unpublished_source_lead() {
        for (index, leads) in [vec![120, 122, 124], vec![121, 124], vec![118, 122, 126]].iter().enumerate() {
            let owned: Vec<(String, String)> = leads.iter().map(|lead| {
                let name = format!("f{lead:03}.grib2");
                (format!("{lead}\t{name}\t96"), name)
            }).collect();
            let rows: Vec<(&str, &str)> = owned.iter().map(|(line, name)| (line.as_str(), name.as_str())).collect();
            assert!(parsed_series(&format!("missing-{index}"), &rows).unwrap_err().contains("does not publish"));
        }
    }

}
