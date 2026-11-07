//! `rw_netcdf` -- the NetCDF front door.
//!
//! Two passes, deliberately the same shape as the `grib2_inventory` /
//! `grib2_dump` pair that `gpuwm.mapped_source` already drives for
//! GRIB2, so one module reads both formats through one idiom:
//!
//! * `rw_netcdf inventory FILE` prints a JSON description of the file --
//!   format, dimensions, global attributes, and every variable with its
//!   dimensions, shape, dtype and attributes.  Selector resolution
//!   (name, `standard_name`, units checks) happens on the Python side
//!   against this document.  The only values read are one element per
//!   variable of a classic file, to prove the file holds every byte its
//!   header describes (see [`check_classic_extent`]).
//! * `rw_netcdf dump FILE OUTDIR VAR...` decodes the named variables and
//!   writes little-endian `f64` numeric planes or fixed-width ASCII bytes,
//!   plus `metadata.json` giving each variable's dtype, shape and filename.
//!   numpy maps those with the declared `<f8` or `|S1` dtype.
//!
//! CF reference-time decoding lives HERE rather than in Python, because
//! turning `hours since 1970-01-01` plus a number into an instant is
//! decoding meteorological data, not plumbing.  A variable whose `units`
//! parses as a CF time gets an extra `times` array of RFC-3339 strings
//! in the dump metadata; Python reads strings and never runs a calendar.
//!
//! Everything is read through `netcrust`, the vendored pure-Rust
//! NetCDF/NetCDF-4 facade already used by rw-wrfbatch, rw-sat and
//! rw-fieldcmp.  There is no C NetCDF library anywhere in this path.

use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use chrono::{DateTime, Duration, NaiveDate, NaiveDateTime, NaiveTime, Utc};
use serde::Serialize;
mod water_layer;
mod soil_recovery;
mod invariant_binding;
mod compose_source;
mod storm_diagnostics;
mod composition_stack;
mod aeronet_aod;
mod stored_words;

/// The contract marker. Printed by `--abi`, and -- because it is a
/// literal in the binary -- readable straight out of the bytes by the
/// release cut, the same convention every other bundled artifact uses.
const ABI: &str = concat!(
    "gpuwm-rw-netcdf-inventory-v1\tformat\tdimensions\tglobal_attributes\tvariables",
    "\tgpuwm-rw-netcdf-dump-v1\tvariables\tfilename\tshape\ttimes",
    "\tdtype\t<f8\t|S1\twater_layer_conversion\tsource_soil_recovery",
    "\tdump_window_v1",
    "\tdump_sample_window_v1",
    "\tdump_sample_window_masked_v1",
    "\tbind_published_invariants_v1",
    "\tcompose_source_v1 storm_diagnostics_v1 composition_stack_v1 aeronet_aod_v1",
    "\tstored_words_v1",
    "\tsurface_analysis_v1",
    "\tsurface_analysis_v2",
    "\tsurface_analysis_v3",
    "\tsurface_analysis_v4 wrflowinp_v1",
    "\tstored_words_record_v1",
);

const INVENTORY_SCHEMA: &str = "gpuwm-rw-netcdf-inventory-v1";
const DUMP_SCHEMA: &str = "gpuwm-rw-netcdf-dump-v1";
mod surface;

/// `GPUWM_BRIDGE_SOURCE_REV=<40-hex commit>`: the source revision this
/// binary was built from, embedded as a literal so the release cut can
/// read it straight out of the bytes without executing anything.  A
/// bundled artifact the cut cannot prove must not ship, and this is what
/// makes "built from the release commit" a property of the file itself.
pub static GPUWM_BRIDGE_SOURCE_REV_STAMP: &str =
    concat!("GPUWM_BRIDGE_SOURCE_REV=", env!("GPUWM_BRIDGE_SOURCE_REV"));

const USAGE: &str = "\
usage: rw_netcdf inventory FILE
       rw_netcdf stored-words FILE OUTPUT_DIR VARIABLE [VARIABLE...]
       rw_netcdf stored-words-record RECORD FILE OUTPUT_DIR VARIABLE [VARIABLE...]
       rw_netcdf dump [--raw|--no-mask] [--unit-scale=N] [--unit-offset=N] [--water-layer-thickness=VARIABLE] [--window=I0:NI,J0:NJ] [--sample-window=X0:NI,Y0:NJ --sample-method=nearest|bilinear|bilinear-masked [--sample-mask=VARIABLE]] FILE OUTPUT_DIR VARIABLE [VARIABLE...]
       rw_netcdf recover-wrf-soil WRFINPUT MET_EM AUTHORITY_JSON OUTPUT_DIR
       rw_netcdf bind-invariants HEIGHT_FILE HEIGHT_VAR LAND_FILE LAND_VAR OUTPUT_FILE TIME_FILE [TIME_FILE...]
       rw_netcdf storm-diagnostics OUTPUT_DIR HEIGHTS_CSV GRAVITY JOBS FILE...
       rw_netcdf aeronet-aod MANIFEST OUTPUT_DIR JOBS
       rw_netcdf composition-stack MANIFEST OUTPUT_DIR JOBS
       rw_netcdf compose-source MANIFEST OUTPUT_DIR JOBS
       rw_netcdf --abi | --help

  inventory  print a JSON description of FILE; a classic file that ends
             before its data does is refused
  dump       decode VARIABLEs into OUTPUT_DIR as flat little-endian f64
             files plus metadata.json
  --unit-scale=N / --unit-offset=N  explicit quantity conversion after CF unpacking
  --water-layer-thickness=VARIABLE  convert declared layer water mass/depth to volume fraction
  --window=I0:NI,J0:NJ  keep NI columns from I0 and NJ rows from J0 of the
             last two dimensions (x last, 0-based), after CF decoding; a
             window that leaves the variable, or a variable with fewer
             than two dimensions, is refused
  --sample-mask=VARIABLE  with --sample-method=bilinear-masked: a 2-D
             class field of FILE (a land mask); each output point
             averages only the source points of its own class (the class
             of its nearest source point), the masked interpolation WPS
             applies to land-only fields so coast cells do not blend in
             the other class's fill value
  --raw     skip CF decoding: no _FillValue/missing_value masking and no
             scale_factor/add_offset, so stored sentinels survive.  This
             is what netCDF4's set_auto_mask(False) asks for, and some
             products (gridded radar/satellite) read their own fill
             sentinel deliberately rather than a mask.
";

#[derive(Serialize)]
struct Inventory {
    schema: &'static str,
    format: String,
    metadata: MetadataProvenance,
    /// True when this reader proved the file holds every byte its header
    /// describes (a classic file; see [`check_classic_extent`]).  False
    /// for a NetCDF-4 file, whose own library checks its end when it
    /// opens.  A reader older than the check omits the field, which is
    /// how a caller tells the two apart.
    extent_checked: bool,
    dimensions: Vec<DimensionRecord>,
    global_attributes: BTreeMap<String, serde_json::Value>,
    variables: Vec<VariableRecord>,
}

#[derive(Serialize)]
struct DimensionRecord {
    name: String,
    len: usize,
    unlimited: bool,
}

#[derive(Serialize)]
struct VariableRecord {
    name: String,
    dimensions: Vec<String>,
    shape: Vec<usize>,
    dtype: String,
    attributes: BTreeMap<String, serde_json::Value>,
    /// True when this variable was recovered from the raw HDF5 index
    /// because the NetCDF-4 variable table omitted it -- see
    /// [`recover_dimension_scales`].  The reader lists coordinate
    /// variables itself, so a record carries this only when a scale
    /// with data was hidden from the table all the same.  Its
    /// attributes are limited to the CF string attributes that can be
    /// read by name, so a consumer that needs numeric attributes on it
    /// must say so rather than assume.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    recovered_dimension_scale: bool,
}

/// CF string attributes probed by name on a recovered dimension scale.
///
/// The raw-HDF5 escape hatch reads ONE named string attribute at a time
/// -- there is no enumeration -- so the set has to be written down.
/// These are the attributes the coordinate contract actually consults:
/// `units` and `standard_name` decide selector resolution, `calendar`
/// and `axis` describe the time axis, `long_name` is for diagnostics.
const RECOVERED_ATTRIBUTES: [&str; 5] =
    ["units", "standard_name", "long_name", "calendar", "axis"];

/// The netCDF-4 marker on an HDF5 dimension scale that has no coordinate
/// variable behind it.  The C library writes this exact sentence into the
/// scale's `NAME` attribute and hides such datasets from `variables`.
const PHONY_DIMENSION_PREFIX: &str =
    "This is a netCDF dimension but not a netCDF variable.";

#[derive(Serialize)]
struct DumpMetadata {
    schema: &'static str,
    metadata: MetadataProvenance,
    variables: Vec<DumpRecord>,
}

#[derive(Serialize)]
struct DumpRecord {
    name: String,
    filename: String,
    shape: Vec<usize>,
    dimensions: Vec<String>,
    dtype: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    units: Option<String>,
    /// RFC-3339 instants, present only when `units` parses as a CF
    /// reference time. Decoding a calendar is this binary's job.
    #[serde(skip_serializing_if = "Option::is_none")]
    times: Option<Vec<String>>,
    /// What CF decoding was applied, so a reader can audit it rather
    /// than infer it. `missing_count` is the number of elements written
    /// as NaN because they matched `_FillValue`/`missing_value`.
    cf: CfApplied,
    #[serde(skip_serializing_if = "Option::is_none")]
    unit_transform: Option<[f64; 2]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    water_layer_conversion: Option<water_layer::Conversion>,
    /// `[i0, ni, j0, nj]` when `--window` cut the last two dimensions,
    /// so a caller can prove the reader applied the window it asked for.
    #[serde(skip_serializing_if = "Option::is_none")]
    window: Option<[usize; 4]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    sample_window: Option<SampleWindow>,
    /// The class field a `bilinear-masked` sample was restricted by.
    #[serde(skip_serializing_if = "Option::is_none")]
    sample_mask: Option<String>,
}

/// Same-spacing rectangular sampling in the last two axes. Origins are
/// zero-based source mass points and output points advance one source cell.
#[derive(Clone, Copy, Debug, Serialize)]
struct SampleWindow {
    i0: f64,
    ni: usize,
    j0: f64,
    nj: usize,
    method: SampleMethod,
}

#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
enum SampleMethod {
    Nearest,
    Bilinear,
    /// Bilinear over the source points that share the class (mask zero
    /// or nonzero) of the output point's nearest source point.  A point
    /// whose positive-weight neighbours are all of one class gets exactly
    /// the `bilinear` bits; only class boundaries differ.
    #[serde(rename = "bilinear-masked")]
    BilinearMasked,
}

/// The class plane a masked sample reads: the last two dimensions of the
/// mask variable, one plane.
#[derive(Clone, Copy, Debug)]
struct SampleMask<'a> {
    name: &'a str,
    values: &'a [f64],
    ny: usize,
    nx: usize,
}

impl SampleWindow {
    fn parse(text: &str, method: &str) -> Result<Self, String> {
        let bad = || format!("--sample-window needs finite X0:NI,Y0:NJ with positive whole counts, got {text:?}");
        let (x, y) = text.split_once(',').ok_or_else(bad)?;
        let (i0, ni) = x.split_once(':').ok_or_else(bad)?;
        let (j0, nj) = y.split_once(':').ok_or_else(bad)?;
        let parsed = Self {
            i0: i0.parse().map_err(|_| bad())?, ni: ni.parse().map_err(|_| bad())?,
            j0: j0.parse().map_err(|_| bad())?, nj: nj.parse().map_err(|_| bad())?,
            method: match method { "nearest" => SampleMethod::Nearest,
                "bilinear" => SampleMethod::Bilinear,
                "bilinear-masked" => SampleMethod::BilinearMasked,
                _ => return Err(format!("--sample-method must be nearest, bilinear or bilinear-masked, got {method:?}")), },
        };
        if !parsed.i0.is_finite() || !parsed.j0.is_finite()
            || parsed.i0 < 0.0 || parsed.j0 < 0.0 || parsed.ni == 0 || parsed.nj == 0 {
            return Err(bad());
        }
        Ok(parsed)
    }

    #[cfg_attr(not(test), allow(dead_code))]
    fn apply(&self, name: &str, values: &[f64], shape: &[usize])
             -> Result<(Vec<f64>, Vec<usize>), String> {
        self.apply_masked(name, values, shape, None)
    }

    fn apply_masked(&self, name: &str, values: &[f64], shape: &[usize], mask: Option<SampleMask<'_>>)
             -> Result<(Vec<f64>, Vec<usize>), String> {
        if (self.method == SampleMethod::BilinearMasked) != mask.is_some() {
            return Err(format!("{name}: bilinear-masked sampling and a sample mask go together"));
        }
        if shape.len() < 2 {
            return Err(format!("{name}: sampling needs at least two dimensions, got {shape:?}"));
        }
        let nx = shape[shape.len() - 1];
        let ny = shape[shape.len() - 2];
        let exact_count_limit = 1_u64 << 53;
        if self.ni as u64 > exact_count_limit || self.nj as u64 > exact_count_limit {
            return Err(format!("{name}: sampling counts overflow exact source-index arithmetic"));
        }
        let end_x = self.i0 + (self.ni - 1) as f64;
        let end_y = self.j0 + (self.nj - 1) as f64;
        if nx == 0 || ny == 0 || !end_x.is_finite() || !end_y.is_finite()
            || self.i0 < 0.0 || self.j0 < 0.0
            || end_x > (nx - 1) as f64 || end_y > (ny - 1) as f64 {
            return Err(format!("{name}: sampled points {}..{} and {}..{} leave the variable's {ny} x {nx} plane",
                               self.i0, end_x, self.j0, end_y));
        }
        let checked_product = |dims: &[usize]| dims.iter().try_fold(1_usize,
            |a, b| a.checked_mul(*b)).ok_or_else(|| format!("{name}: sampling shape overflows the index range"));
        if checked_product(shape)? != values.len() {
            return Err(format!("{name}: sampling input shape differs from decoded value count"));
        }
        let lead = checked_product(&shape[..shape.len() - 2])?;
        if let Some(mask) = mask {
            if mask.ny != ny || mask.nx != nx || mask.values.len() != ny * nx {
                return Err(format!("{name}: sample mask {} is {} x {}, the variable's plane {ny} x {nx}",
                                   mask.name, mask.ny, mask.nx));
            }
        }
        let count = checked_product(&[lead, self.ni, self.nj])?;
        let mut out = Vec::new();
        out.try_reserve_exact(count).map_err(|_| format!("{name}: sampled output allocation is too large"))?;
        for plane in 0..lead {
            let base = plane * ny * nx;
            for j in 0..self.nj {
                let y = self.j0 + j as f64;
                for i in 0..self.ni {
                    let x = self.i0 + i as f64;
                    let get = |xx: usize, yy: usize| -> Result<f64, String> {
                        let value = values[base + yy * nx + xx];
                        if !value.is_finite() {
                            Err(format!("{name}: non-finite source value needed for sampling at ({xx}, {yy})"))
                        } else { Ok(value) }
                    };
                    let bilinear = || -> Result<f64, String> {
                        let xx = x.floor() as usize;
                        let yy = y.floor() as usize;
                        let fx = x - xx as f64;
                        let fy = y - yy as f64;
                        let a = get(xx, yy)?;
                        // Exact integer origins keep their original f64 bits and
                        // do not read zero-weight neighbors at the source edge.
                        let top = if fx == 0.0 { a } else {
                            a * (1.0 - fx) + get(xx + 1, yy)? * fx };
                        Ok(if fy == 0.0 { top } else {
                            let c = get(xx, yy + 1)?;
                            let bottom = if fx == 0.0 { c } else {
                                c * (1.0 - fx) + get(xx + 1, yy + 1)? * fx };
                            top * (1.0 - fy) + bottom * fy
                        })
                    };
                    let value = match self.method {
                        SampleMethod::Nearest => get((x + 0.5).floor() as usize, (y + 0.5).floor() as usize)?,
                        SampleMethod::Bilinear => bilinear()?,
                        SampleMethod::BilinearMasked => {
                            let mask = mask.expect("checked above");
                            let class = |xx: usize, yy: usize| -> Result<bool, String> {
                                let flag = mask.values[yy * nx + xx];
                                if !flag.is_finite() {
                                    Err(format!("{name}: sample mask {} is non-finite at ({xx}, {yy})", mask.name))
                                } else { Ok(flag != 0.0) }
                            };
                            let own = class((x + 0.5).floor() as usize, (y + 0.5).floor() as usize)?;
                            let xx = x.floor() as usize;
                            let yy = y.floor() as usize;
                            let fx = x - xx as f64;
                            let fy = y - yy as f64;
                            // Positive-weight neighbours only: a zero weight never
                            // reads past the source edge.  The nearest point is
                            // always among them, so `own` carries weight.
                            let mut corners = [(xx, yy, (1.0 - fx) * (1.0 - fy)), (0, 0, 0.0),
                                               (0, 0, 0.0), (0, 0, 0.0)];
                            let mut used = 1;
                            if fx != 0.0 { corners[used] = (xx + 1, yy, fx * (1.0 - fy)); used += 1; }
                            if fy != 0.0 {
                                corners[used] = (xx, yy + 1, (1.0 - fx) * fy); used += 1;
                                if fx != 0.0 { corners[used] = (xx + 1, yy + 1, fx * fy); used += 1; }
                            }
                            let mut mixed = false;
                            for &(cx, cy, _) in &corners[..used] {
                                mixed |= class(cx, cy)? != own;
                            }
                            if !mixed { bilinear()? } else {
                                let (mut sum, mut weight) = (0.0_f64, 0.0_f64);
                                for &(cx, cy, w) in &corners[..used] {
                                    if w > 0.0 && class(cx, cy)? == own {
                                        sum += w * get(cx, cy)?;
                                        weight += w;
                                    }
                                }
                                if weight <= 0.0 {
                                    return Err(format!("{name}: no source point of the output point's own class carries weight"));
                                }
                                sum / weight
                            }
                        },
                    };
                    if !value.is_finite() { return Err(format!("{name}: sampling produced a non-finite result")); }
                    out.push(value);
                }
            }
        }
        let mut sampled_shape = shape[..shape.len() - 2].to_vec();
        sampled_shape.extend_from_slice(&[self.nj, self.ni]);
        Ok((out, sampled_shape))
    }
}

/// A rectangle of the last two dimensions of a variable: `ni` columns
/// from `i0` along the last (x) dimension and `nj` rows from `j0` along
/// the one before it, both 0-based.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Window {
    i0: usize,
    ni: usize,
    j0: usize,
    nj: usize,
}

impl Window {
    fn parse(text: &str) -> Result<Self, String> {
        let bad = || format!("--window must be I0:NI,J0:NJ with whole numbers, got {text:?}");
        let (x, y) = text.split_once(',').ok_or_else(bad)?;
        let pair = |part: &str| -> Result<(usize, usize), String> {
            let (start, count) = part.split_once(':').ok_or_else(bad)?;
            Ok((start.trim().parse().map_err(|_| bad())?,
                count.trim().parse().map_err(|_| bad())?))
        };
        let (i0, ni) = pair(x)?;
        let (j0, nj) = pair(y)?;
        if ni == 0 || nj == 0 {
            return Err(format!("--window {text:?} selects no cells"));
        }
        Ok(Window { i0, ni, j0, nj })
    }

    /// The window of `values` (row-major, `shape`), and its shape.
    ///
    /// Refused, by name, for a variable with fewer than two dimensions
    /// or a window that is not wholly inside the last two: a window cut
    /// short at the edge would hand back a smaller field than the grid
    /// it was asked for, under that grid's name.
    fn apply(&self, name: &str, values: &[f64], shape: &[usize])
             -> Result<(Vec<f64>, Vec<usize>), String> {
        self.check(name, shape)?;
        let nx = shape[shape.len() - 1];
        let ny = shape[shape.len() - 2];
        let lead: usize = shape[..shape.len() - 2].iter().product();
        let mut out = Vec::with_capacity(lead * self.ni * self.nj);
        for plane in 0..lead {
            for row in 0..self.nj {
                let start = plane * ny * nx + (self.j0 + row) * nx + self.i0;
                out.extend_from_slice(&values[start..start + self.ni]);
            }
        }
        let mut cut = shape[..shape.len() - 2].to_vec();
        cut.push(self.nj);
        cut.push(self.ni);
        Ok((out, cut))
    }

    /// The hyperslab of a variable of `shape` this window covers: every
    /// leading index, then the window's rows and columns.  Call after
    /// [`Window::check`].
    fn selection(&self, shape: &[usize]) -> netcrust::NcSliceInfo {
        let mut selections: Vec<netcrust::NcSliceInfoElem> = shape[..shape.len() - 2]
            .iter()
            .map(|&n| netcrust::NcSliceInfoElem::Slice { start: 0, end: n as u64, step: 1 })
            .collect();
        selections.push(netcrust::NcSliceInfoElem::Slice {
            start: self.j0 as u64, end: (self.j0 + self.nj) as u64, step: 1 });
        selections.push(netcrust::NcSliceInfoElem::Slice {
            start: self.i0 as u64, end: (self.i0 + self.ni) as u64, step: 1 });
        netcrust::NcSliceInfo { selections }
    }

    /// Refuse, by name, a window this variable cannot supply whole.
    fn check(&self, name: &str, shape: &[usize]) -> Result<(), String> {
        if shape.len() < 2 {
            return Err(format!("{name}: --window needs at least two dimensions, got shape {shape:?}"));
        }
        let nx = shape[shape.len() - 1];
        let ny = shape[shape.len() - 2];
        let i_end = self.i0.checked_add(self.ni).ok_or_else(|| format!(
            "{name}: --window column start {} plus count {} overflows the index range",
            self.i0, self.ni))?;
        let j_end = self.j0.checked_add(self.nj).ok_or_else(|| format!(
            "{name}: --window row start {} plus count {} overflows the index range",
            self.j0, self.nj))?;
        if i_end > nx || j_end > ny {
            return Err(format!(
                "{name}: --window columns {}..{} rows {}..{} leave the variable's {ny} x {nx} plane",
                self.i0, i_end, self.j0, j_end));
        }
        Ok(())
    }
}

#[derive(Serialize, Default)]
struct CfApplied {
    #[serde(skip_serializing_if = "Option::is_none")]
    scale_factor: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    add_offset: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    fill_value: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    missing_value: Option<f64>,
    /// The NetCDF default fill of the stored type, masked because the
    /// variable declares no `_FillValue`: a value the writer never set
    /// (a masked element written without an explicit fill) is stored as
    /// this finite number, and passing it on made 9.97e36 K a
    /// temperature.  Byte types have no default fill in this rule, the
    /// same exemption netCDF4-python makes.
    #[serde(skip_serializing_if = "Option::is_none")]
    default_fill: Option<f64>,
    /// True when the reader applied the default-fill rule above, so a
    /// caller can tell this reader from one that predates it.
    default_fill_rule: bool,
    missing_count: usize,
    /// False under `--raw`: the attributes above are reported but were
    /// deliberately not acted on, so stored sentinels survive.
    applied: bool,
}

fn main() {
    // Keep the stamp in the binary: an unreferenced static can be
    // dropped by the linker, and a stamp that optimises away is a
    // release the cut refuses for a reason nobody can see.
    let _ = std::hint::black_box(GPUWM_BRIDGE_SOURCE_REV_STAMP);
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprint!("{USAGE}");
        std::process::exit(2);
    }
    match args[0].as_str() {
        "storm-diagnostics" => {
            if args.len() < 6 { fail("storm-diagnostics takes OUTPUT_DIR HEIGHTS_CSV GRAVITY JOBS FILE..."); }
            let heights = args[2].split(',').map(str::parse::<f64>).collect::<Result<Vec<_>,_>>().unwrap_or_else(|_|fail("invalid heights"));
            let gravity = args[3].parse().unwrap_or_else(|_|fail("invalid gravity"));
            let jobs = args[4].parse().unwrap_or_else(|_|fail("invalid jobs"));
            let files = args[5..].iter().map(PathBuf::from).collect::<Vec<_>>();
            if let Err(error) = storm_diagnostics::run(Path::new(&args[1]), &heights, gravity, jobs, &files) { fail(&error); }
        }
        "aeronet-aod" => {
            if args.len()!=4 { fail("aeronet-aod takes MANIFEST OUTPUT_DIR JOBS"); }
            let jobs=args[3].parse().unwrap_or_else(|_|fail("invalid jobs"));
            if let Err(error)=aeronet_aod::run(Path::new(&args[1]),Path::new(&args[2]),jobs){fail(&error);}
        }
        "composition-stack" => {
            if args.len()!=4 { fail("composition-stack takes MANIFEST OUTPUT_DIR JOBS"); }
            let jobs=args[3].parse().unwrap_or_else(|_|fail("invalid jobs"));
            if let Err(error)=composition_stack::run(Path::new(&args[1]),Path::new(&args[2]),jobs){fail(&error);}
        }
        "compose-source" => {
            if args.len() != 4 { fail("compose-source takes MANIFEST OUTPUT_DIR JOBS"); }
            let jobs = args[3].parse().unwrap_or_else(|_| { fail("JOBS must be an unsigned integer"); });
            if let Err(error) = compose_source::compose(Path::new(&args[1]), Path::new(&args[2]), jobs) { fail(&error); }
        }
        "bind-invariants" => {
            if args.len() < 7 { fail("bind-invariants takes HEIGHT_FILE HEIGHT_VAR LAND_FILE LAND_VAR OUTPUT_FILE TIME_FILE [TIME_FILE...]"); }
            let time_files: Vec<PathBuf> = args[6..].iter().map(PathBuf::from).collect();
            if let Err(error) = invariant_binding::bind(Path::new(&args[1]), &args[2], Path::new(&args[3]), &args[4], Path::new(&args[5]), &time_files) { fail(&error); }
        }
        "recover-wrf-soil" => {
            if args.len() != 5 { fail("recover-wrf-soil takes WRFINPUT MET_EM AUTHORITY_JSON OUTPUT_DIR"); }
            if let Err(error) = soil_recovery::recover(Path::new(&args[1]),Path::new(&args[2]),Path::new(&args[3]),Path::new(&args[4])) { fail(&error); }
        }
        "--abi" => {
            println!("{ABI}");
        }
        "--help" | "-h" => {
            print!("{USAGE}");
        }
        "stored-words-record" => {
            if args.len() < 5 { fail("stored-words-record takes RECORD FILE OUTPUT_DIR VARIABLE [VARIABLE...]"); }
            let record = args[1].parse::<usize>().unwrap_or_else(|_| fail("RECORD must be a nonnegative integer"));
            if let Err(error) = stored_words::run_record(Path::new(&args[2]), Path::new(&args[3]), &args[4..], record) {
                fail(&error);
            }
        }
        "stored-words" => {
            if args.len() < 4 { fail("stored-words takes FILE OUTPUT_DIR VARIABLE [VARIABLE...]"); }
            if let Err(error) = stored_words::run(Path::new(&args[1]), Path::new(&args[2]), &args[3..]) {
                fail(&error);
            }
        }
        "inventory" => {
            if args.len() != 2 {
                fail("inventory takes exactly one FILE");
            }
            if let Err(error) = inventory(Path::new(&args[1])) {
                fail(&error);
            }
        }
        "wrflowinp" => {
            let (offset, surface_physics) = if args.get(1).map(String::as_str) == Some("--sf-surface-physics") {
                if args.len() < 5 {fail("wrflowinp --sf-surface-physics N OUTPUT INPUT [INPUT ...]");}
                (3, args[2].parse::<i32>().unwrap_or_else(|_|fail("invalid surface physics")))
            } else { (1, 2) };
            if args.len()<offset+2 {fail("wrflowinp OUTPUT INPUT [INPUT ...]");}
            let inputs:Vec<_>=args[offset+1..].iter().map(std::path::PathBuf::from).collect();
            match wrf_surface::run_with_surface_physics(&inputs,Path::new(&args[offset]),surface_physics) {
                Ok(receipt)=>println!("{}",serde_json::to_string_pretty(&receipt).unwrap()),
                Err(error)=>fail(&error),
            }
        }
        "surface" => {
            if args.len() != 3 { fail("surface takes SPEC.json OUTPUT_DIR"); }
            if let Err(error) = surface::run(Path::new(&args[1]), Path::new(&args[2])) { fail(&error); }
        }
        "dump" => {
            // Two switches, because netCDF4-python has two.  `--raw` is
            // `set_auto_maskandscale(False)`: no masking and no scaling.
            // `--no-mask` is `set_auto_mask(False)`: no masking, but
            // scale_factor/add_offset STILL applied.  Collapsing the two
            // silently unpacked every packed variable a caller asked to
            // see unmasked, which is a value change, not a policy change.
            let raw = args.iter().any(|a| a == "--raw");
            let no_mask = raw || args.iter().any(|a| a == "--no-mask");
            let apply_scale = !raw;
            let water_layer = args.iter().find_map(|a| a.strip_prefix("--water-layer-thickness="));
            let window = match args.iter().find_map(|a| a.strip_prefix("--window=")) {
                Some(text) => Some(Window::parse(text).unwrap_or_else(|error| fail(&error))),
                None => None,
            };
            let sample_text = args.iter().find_map(|a| a.strip_prefix("--sample-window="));
            let sample_method = args.iter().find_map(|a| a.strip_prefix("--sample-method="));
            let sample_window = match (sample_text, sample_method) {
                (Some(text), Some(method)) => Some(SampleWindow::parse(text, method).unwrap_or_else(|error| fail(&error))),
                (None, None) => None,
                _ => fail("--sample-window and --sample-method must be specified together"),
            };
            if window.is_some() && sample_window.is_some() { fail("--window and --sample-window cannot be combined"); }
            let sample_mask = args.iter().find_map(|a| a.strip_prefix("--sample-mask="));
            if sample_mask.is_some() != sample_window.is_some_and(|s| s.method == SampleMethod::BilinearMasked) {
                fail("--sample-mask goes with --sample-method=bilinear-masked, and only with it");
            }
            if water_layer.is_some() && (raw || no_mask) {
                fail("layer water conversion requires CF masking and unpacking");
            }
            let mut unit_transform = [1.0_f64, 0.0_f64];
            for argument in &args[1..] {
                for (index, prefix) in [(0, "--unit-scale="), (1, "--unit-offset=")] {
                    if let Some(text) = argument.strip_prefix(prefix) {
                        unit_transform[index] = text.parse::<f64>()
                            .unwrap_or_else(|_| fail("unit transform must be finite numbers"));
                    }
                }
            }
            if !unit_transform.iter().all(|v| v.is_finite()) || unit_transform[0] == 0.0 {
                fail("unit transform must have a finite nonzero scale and finite offset");
            }
            let positional: Vec<&String> = args[1..]
                .iter()
                .filter(|a| a.as_str() != "--raw" && a.as_str() != "--no-mask"
                    && !a.starts_with("--unit-scale=") && !a.starts_with("--unit-offset=")
                    && !a.starts_with("--water-layer-thickness=")
                    && !a.starts_with("--window=") && !a.starts_with("--sample-window=")
                    && !a.starts_with("--sample-method=") && !a.starts_with("--sample-mask="))
                .collect();
            if positional.len() < 3 {
                fail("dump takes FILE, OUTPUT_DIR and at least one VARIABLE");
            }
            let names: Vec<String> =
                positional[2..].iter().map(|s| (*s).clone()).collect();
            if let Err(error) = dump_sampled_masked(
                Path::new(positional[0]),
                Path::new(positional[1]),
                &names,
                !no_mask,
                apply_scale,
                unit_transform,
                water_layer,
                window,
                sample_window,
                sample_mask,
            ) {
                fail(&error);
            }
        }
        other => fail(&format!("unknown subcommand {other:?}\n{USAGE}")),
    }
}

fn fail(message: &str) -> ! {
    eprintln!("rw_netcdf: {message}");
    std::process::exit(2);
}

fn attribute_json(value: &netcrust::AttributeValue) -> serde_json::Value {
    use netcrust::AttributeValue as V;
    fn numbers<T: Copy + Into<f64>>(values: &[T]) -> serde_json::Value {
        let mapped: Vec<serde_json::Value> = values
            .iter()
            .map(|&value| {
                serde_json::Number::from_f64(value.into())
                    .map(serde_json::Value::Number)
                    .unwrap_or(serde_json::Value::Null)
            })
            .collect();
        if mapped.len() == 1 {
            mapped.into_iter().next().unwrap()
        } else {
            serde_json::Value::Array(mapped)
        }
    }
    match value {
        // NC_CHAR text as netCDF4-python and this tool returned it before the reader kept every declared
        // byte: trailing NULs are padding, so "" stored as one NUL reads back "" (an empty attribute
        // read as a lone NUL defeated every empty-value check; 2.9.4 final pass R5, test_obs_radar_grid).
        V::Chars(text) => serde_json::Value::String(text.trim_end_matches('\0').to_string()),
        V::Strings(values) if values.len() == 1 => serde_json::Value::String(values[0].clone()),
        V::Strings(values) => serde_json::Value::Array(
            values
                .iter()
                .map(|value| serde_json::Value::String(value.clone()))
                .collect(),
        ),
        V::Bytes(values) => numbers(values),
        V::Shorts(values) => numbers(values),
        V::Ints(values) => numbers(values),
        V::Floats(values) => numbers(values),
        V::Doubles(values) => numbers(values),
        V::UBytes(values) => numbers(values),
        V::UShorts(values) => numbers(values),
        V::UInts(values) => numbers(values),
        // i64/u64 beyond f64's exact range would be silently rounded by
        // the numeric path; emit them as integers instead.
        V::Int64s(values) if values.len() == 1 => serde_json::Value::from(values[0]),
        V::Int64s(values) => serde_json::Value::Array(
            values.iter().map(|&v| serde_json::Value::from(v)).collect(),
        ),
        V::UInt64s(values) if values.len() == 1 => serde_json::Value::from(values[0]),
        V::UInt64s(values) => serde_json::Value::Array(
            values.iter().map(|&v| serde_json::Value::from(v)).collect(),
        ),
    }
}

fn attributes_map(attributes: &[netcrust::Attribute]) -> BTreeMap<String, serde_json::Value> {
    attributes
        .iter()
        .map(|attribute| {
            (
                attribute.name().to_string(),
                attribute_json(attribute.value()),
            )
        })
        .collect()
}

/// Open strictly, and fall back to size-inferred NetCDF-4 metadata only
/// when strict reconstruction is impossible -- reporting which happened.
///
/// Strict mode requires every NetCDF-4 variable to carry a
/// `DIMENSION_LIST` attribute naming its dimension scales.  Real files
/// from major producers do not always have one: ERA5 as delivered by the
/// Copernicus CDS omits it on `number` and `expver`, and the RRTMGP
/// coefficient tables omit it throughout.  Refusing those outright would
/// be refusing the project's own canonical forcing source.
///
/// The fallback resolves a variable's dimensions BY SIZE, which is a
/// heuristic and can mis-name axes whenever two dimensions share a
/// length.  So it is never silent: the mode travels in the output
/// document, and `dimension_lengths_ambiguous` says whether the file
/// contains a length collision that could make the inference wrong.  A
/// consumer that cares about axis identity can then refuse, instead of
/// being handed a guess it cannot distinguish from a fact.
fn open(path: &Path) -> Result<(netcrust::File, MetadataProvenance), String> {
    // The open itself is lazy: a NetCDF-4 file whose metadata cannot be
    // reconstructed still opens, and only reports it when the dimension
    // or variable tables are first built.  So strictness is decided by
    // PROBING those tables, not by whether the handle came back.
    let probe = |file: &netcrust::File| -> Result<(), String> {
        file.dimensions()
            .map_err(|error| error.to_string())
            .and_then(|_| file.variables().map_err(|error| error.to_string()))
            .map(|_| ())
    };

    let strict_error = match netcrust::File::open(path) {
        Ok(file) => match probe(&file) {
            Ok(()) => return Ok((file, MetadataProvenance::strict())),
            Err(error) => error,
        },
        Err(error) => error.to_string(),
    };

    let options = netcrust::NcOpenOptions {
        metadata_mode: netcrust::NcMetadataMode::Lossy,
        ..Default::default()
    };
    let file = netcrust::File::open_with_options(path, options).map_err(|lossy_error| {
        format!(
            "cannot open {}: {strict_error} (and size-inferred metadata \
             also failed: {lossy_error})",
            path.display()
        )
    })?;
    probe(&file).map_err(|lossy_error| {
        format!(
            "cannot read {}: {strict_error} (and size-inferred metadata \
             also failed: {lossy_error})",
            path.display()
        )
    })?;
    let ambiguous = has_duplicate_dimension_lengths(&file);
    Ok((
        file,
        MetadataProvenance {
            mode: "size-inferred",
            strict_error: Some(strict_error),
            dimension_lengths_ambiguous: ambiguous,
        },
    ))
}

/// Add back any NetCDF-4 dimension scale with data that the variable
/// table omits.
///
/// In NetCDF-4 a coordinate variable is stored as an HDF5 *dimension
/// scale*.  The vendored reader's `variables()` lists those itself, with
/// their own dimensions, stored type and every attribute, and hides only
/// the scales netCDF-C marks as dimension-only (no coordinate values):
/// so `latitude`, `longitude`, `pressure_level` and `valid_time` of an
/// ERA5 file, and `time` and the `xh/yh/zh` axes of a CM1 history file,
/// come from the table like any field.  This pass is the fallback for
/// a scale with data the table still hides (a stored type the reader
/// cannot name, for one): those are exactly the variables a coordinate
/// contract resolves against, so a reader that silently dropped one
/// would list every field and not that axis.  A name already in the
/// table is never recovered a second time.
///
/// Recovery uses only public netcrust API -- `hdf5_root_datasets` for
/// the names and shapes, `hdf5_dataset_attribute_string` for the CF
/// string attributes, and `read_array_f64`, which already falls back to
/// a raw-HDF5 read by name for values.
///
/// A 1-D dataset whose length matches a declared dimension is given that
/// dimension. Ambiguity is not guessed at: a dataset whose own name is a
/// dimension takes that dimension (the CF coordinate-variable
/// convention), and anything else that stays ambiguous is reported with
/// an empty dimension list rather than a plausible-looking wrong one.
fn recover_dimension_scales(
    file: &netcrust::File,
    dimensions: &[DimensionRecord],
    variables: &mut Vec<VariableRecord>,
) {
    let Ok(datasets) = file.hdf5_root_datasets() else {
        return;
    };
    let known: std::collections::BTreeSet<String> =
        variables.iter().map(|v| v.name.clone()).collect();
    for dataset in datasets {
        let name = dataset.name().to_string();
        if known.contains(&name) {
            continue;
        }
        // A dimension with no coordinate variable is still stored as an
        // HDF5 dimension scale, and the netCDF-4 convention marks it:
        // its NAME attribute begins with the sentence below.  netCDF4
        // hides exactly those, so recovering them would invent
        // variables the C library does not report -- which broke a
        // consumer that compares `set(dataset.variables)` against an
        // expected inventory and saw `Time`, `bottom_top`, `south_north`
        // appear as fields.  This is the convention's own answer, not a
        // heuristic about attributes.
        if file
            .hdf5_dataset_attribute_string(&name, "NAME")
            .is_some_and(|value| value.starts_with(PHONY_DIMENSION_PREFIX))
        {
            continue;
        }
        let shape: Vec<usize> = dataset.shape().iter().map(|&n| n as usize).collect();
        let mut axes = Vec::with_capacity(shape.len());
        for (index, &length) in shape.iter().enumerate() {
            // The coordinate-variable convention first: a 1-D dataset
            // named after a dimension IS that dimension.
            let named = (shape.len() == 1)
                .then(|| dimensions.iter().find(|d| d.name == name && d.len == length))
                .flatten();
            let resolved = named.or_else(|| {
                let mut matches = dimensions.iter().filter(|d| d.len == length);
                let first = matches.next();
                // Only accept a length match when it is unique; two
                // dimensions of equal length cannot be told apart here.
                match (first, matches.next()) {
                    (Some(only), None) => Some(only),
                    _ => None,
                }
            });
            match resolved {
                Some(dimension) => axes.push(dimension.name.clone()),
                None => {
                    axes.clear();
                    let _ = index;
                    break;
                }
            }
        }
        let mut attributes = BTreeMap::new();
        for key in RECOVERED_ATTRIBUTES {
            if let Some(value) = file.hdf5_dataset_attribute_string(&name, key) {
                attributes.insert(key.to_string(), serde_json::Value::String(value));
            }
        }
        variables.push(VariableRecord {
            name,
            dimensions: axes,
            shape,
            // Values come back promoted; the stored type is not exposed
            // by the raw index, and saying F64 would be a claim about
            // storage that this path cannot make.
            dtype: "Promoted".to_string(),
            attributes,
            recovered_dimension_scale: true,
        });
    }
}

fn has_duplicate_dimension_lengths(file: &netcrust::File) -> bool {
    let Ok(dimensions) = file.dimensions() else {
        return true;
    };
    let mut seen = std::collections::BTreeSet::new();
    dimensions
        .iter()
        .any(|dimension| !seen.insert(dimension.len()))
}

#[derive(Serialize, Clone)]
struct MetadataProvenance {
    /// `"strict"` when every dimension came from an explicit
    /// `DIMENSION_LIST`; `"size-inferred"` when it was reconstructed.
    mode: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    strict_error: Option<String>,
    /// True when two dimensions share a length, which is exactly when
    /// size inference can silently attach the wrong axis name.
    dimension_lengths_ambiguous: bool,
}

impl MetadataProvenance {
    fn strict() -> Self {
        Self {
            mode: "strict",
            strict_error: None,
            dimension_lengths_ambiguous: false,
        }
    }
}

/// The file's dimension table as the inventory reports it.
fn dimension_table(file: &netcrust::File, path: &Path) -> Result<Vec<DimensionRecord>, String> {
    Ok(file
        .dimensions()
        .map_err(|error| format!("cannot read dimensions of {}: {error}", path.display()))?
        .into_iter()
        .map(|dimension| DimensionRecord {
            name: dimension.name().to_string(),
            len: dimension.len(),
            unlimited: dimension.is_unlimited(),
        })
        .collect())
}

/// The file's variable table as the inventory reports it: every variable
/// the reader lists, with its dimension names, shape, stored type and
/// attributes, then any dimension scale with data the reader hid.
fn variable_table(
    file: &netcrust::File,
    path: &Path,
    dimensions: &[DimensionRecord],
) -> Result<Vec<VariableRecord>, String> {
    let mut variables: Vec<VariableRecord> = file
        .variables()
        .map_err(|error| format!("cannot read variables of {}: {error}", path.display()))?
        .into_iter()
        .map(|variable| VariableRecord {
            name: variable.name().to_string(),
            dimensions: variable
                .dimensions()
                .iter()
                .map(|dimension| dimension.name().to_string())
                .collect(),
            shape: variable.shape(),
            dtype: format!("{:?}", variable.dtype()),
            attributes: attributes_map(variable.attributes()),
            recovered_dimension_scale: false,
        })
        .collect();
    recover_dimension_scales(file, dimensions, &mut variables);
    Ok(variables)
}

fn inventory(path: &Path) -> Result<(), String> {
    let (file, metadata) = open(path)?;
    let extent_checked = check_classic_extent(&file)?;
    let dimensions = dimension_table(&file, path)?;
    let global_attributes = attributes_map(
        &file
            .attributes()
            .map_err(|error| format!("cannot read global attributes: {error}"))?,
    );
    let variables = variable_table(&file, path, &dimensions)?;

    let record = Inventory {
        schema: INVENTORY_SCHEMA,
        format: format!("{:?}", file.format()),
        metadata,
        extent_checked,
        dimensions,
        global_attributes,
        variables,
    };
    let text = serde_json::to_string(&record)
        .map_err(|error| format!("cannot serialize inventory: {error}"))?;
    println!("{text}");
    Ok(())
}

/// Does a classic file hold every byte its header describes?
///
/// A CDF-1/2/5 file cut off partway through its data (a copy that
/// stopped, a disk that filled, a writer that died) keeps its header, so
/// it still opens, and the C library hands the missing values back as
/// zeros without a word.  A truncated forecast then reads as a real one
/// whose fields happen to be zero.  So the inventory reads the LAST
/// element of every variable, one value each whatever the variable's
/// size: the decoder checks the bytes that element needs against the
/// file's length, and between them those elements reach the file's last
/// byte whatever order the variables were laid out in.  A character
/// variable is read whole, being text.
///
/// Only running past the end is this check's to report.  A variable that
/// cannot be read for another reason is left to the read that asks for
/// it, which names its own problem, so a file that inventoried before
/// still does.  `Ok(false)` for a NetCDF-4 file: the HDF5 library checks
/// that container's end when it opens, and this reader did not.
fn check_classic_extent(file: &netcrust::File) -> Result<bool, String> {
    use netcrust::{DataType, NcFormat, NcSliceInfo, NcSliceInfoElem};
    if !matches!(file.format(), NcFormat::Classic | NcFormat::Offset64 | NcFormat::Cdf5) {
        return Ok(false);
    }
    let variables = file
        .variables()
        .map_err(|error| format!("cannot read variables: {error}"))?;
    for variable in &variables {
        let shape = variable.shape();
        if shape.contains(&0) {
            continue;
        }
        let name = variable.name();
        let read = if matches!(variable.dtype(), DataType::Char) {
            file.read_strings(name).map(|_| ())
        } else if shape.is_empty() {
            file.read_array_f64(name).map(|_| ())
        } else {
            let last = NcSliceInfo {
                selections: shape
                    .iter()
                    .map(|&len| NcSliceInfoElem::Index(len as u64 - 1))
                    .collect(),
            };
            file.read_array_f64_slice(name, &last).map(|_| ())
        };
        if let Err(error) = read {
            if error.to_string().contains("beyond file") {
                return Err(format!(
                    "the file ends before its data does: variable '{name}' runs \
                     past the end, so it was cut short while it was written or \
                     copied"
                ));
            }
        }
    }
    Ok(true)
}

/// Preserve fixed-width ASCII character arrays in their declared shape.
/// Classic readers return one string per row; HDF5 NC_CHAR returns one
/// string per byte. Neither packing nor CF masking applies to characters.
///
/// `vlen` names an NC_STRING variable (NetCDF-4 variable-length text, one
/// text per element and no character dimension in its shape, e.g. RFMIP's
/// `expt_label`): its texts are padded to the widest one, and that width
/// is returned so the caller can append it to the shape. Before this such
/// a variable was refused as "does not fit its 1-byte slot", and one such
/// variable failed every variable read beside it.
fn character_bytes(file: &netcrust::File, variable: &netcrust::Variable, vlen: bool)
                   -> Result<(Vec<u8>, Option<usize>), String> {
    let name = variable.name();
    let shape = variable.shape();
    let total = shape.iter().try_fold(1usize, |n, &size| n.checked_mul(size))
        .ok_or_else(|| format!("{name}: character shape overflows"))?;
    let width = shape.last().copied().unwrap_or(1);
    let strings = file.read_strings(name).map_err(|error| format!("cannot decode {name}: {error}"))?;
    if total == 0 && strings.is_empty() { return Ok((Vec::new(), None)); }
    // An HDF5 NC_CHAR array also answers as the string type, one text per
    // byte (shape (19, 32) -> 608 one-character texts): that is the classic
    // form below.  Variable-length text is the case where one text per
    // element carries more than a character.
    if vlen && strings.len() == total && strings.iter().any(|text| text.len() > 1) {
        let chunk = strings.iter().map(|text| text.len()).max().unwrap_or(1);
        let mut bytes = Vec::new();
        bytes.try_reserve_exact(total * chunk)
            .map_err(|error| format!("{name}: cannot allocate {} character bytes: {error}", total * chunk))?;
        for text in strings {
            if !text.is_ascii() {
                return Err(format!("{name}: expected ASCII text; decoded text is not ASCII"));
            }
            bytes.extend_from_slice(text.as_bytes());
            bytes.resize(bytes.len() + chunk - text.len(), 0);
        }
        return Ok((bytes, Some(chunk)));
    }
    let chunk = if width > 0 && strings.len() == total / width {
        width
    } else if strings.len() == total { 1 } else {
        return Err(format!("{name}: decoded {} strings do not match character shape {shape:?}", strings.len()));
    };
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(total)
        .map_err(|error| format!("{name}: cannot allocate {total} character bytes: {error}"))?;
    for text in strings {
        if !text.is_ascii() || text.len() > chunk {
            return Err(format!("{name}: expected fixed-width ASCII characters; decoded text does not fit its {chunk}-byte slot"));
        }
        bytes.extend_from_slice(text.as_bytes());
        bytes.resize(bytes.len() + chunk - text.len(), 0);
    }
    Ok((bytes, None))
}

#[cfg(test)]
fn dump(path: &Path, out_dir: &Path, names: &[String], apply_mask: bool,
        apply_scale: bool) -> Result<(), String> {
    dump_transformed(path, out_dir, names, apply_mask, apply_scale, [1.0, 0.0])
}

fn dump_transformed(path: &Path, out_dir: &Path, names: &[String], apply_mask: bool,
                    apply_scale: bool, unit_transform: [f64; 2]) -> Result<(), String> {
    dump_with_water_layer(path, out_dir, names, apply_mask, apply_scale, unit_transform, None)
}

/// The NetCDF library's default fill for a stored type (netcdf.h
/// NC_FILL_*), as the f64 the decoder compares against.  A float variable
/// is compared in its own precision, so the f32 default widens exactly as
/// a stored f32 element does.  Byte and character types return None.
fn nc_default_fill(dtype: &netcrust::DataType) -> Option<f64> {
    use netcrust::DataType as T;
    match dtype {
        T::I16 => Some(-32767.0),
        T::I32 => Some(-2147483647.0),
        T::F32 => Some(f64::from(9.969_209_968_386_869e36_f32)),
        T::F64 => Some(9.969_209_968_386_869e36),
        T::U16 => Some(65535.0),
        T::U32 => Some(4294967295.0),
        T::I64 => Some(-9223372036854775806.0),
        T::U64 => Some(18446744073709551614.0),
        _ => None,
    }
}

fn dump_with_water_layer(path: &Path, out_dir: &Path, names: &[String], apply_mask: bool,
                    apply_scale: bool, unit_transform: [f64; 2], water_layer: Option<&str>) -> Result<(), String> {
    dump_windowed(path, out_dir, names, apply_mask, apply_scale, unit_transform, water_layer, None)
}

#[allow(clippy::too_many_arguments)]
fn dump_windowed(path: &Path, out_dir: &Path, names: &[String], apply_mask: bool,
                 apply_scale: bool, unit_transform: [f64; 2], water_layer: Option<&str>,
                 window: Option<Window>) -> Result<(), String> {
    dump_sampled(path, out_dir, names, apply_mask, apply_scale, unit_transform,
                 water_layer, window, None)
}

#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(test), allow(dead_code))]
fn dump_sampled(path: &Path, out_dir: &Path, names: &[String], apply_mask: bool,
                 apply_scale: bool, unit_transform: [f64; 2], water_layer: Option<&str>,
                 window: Option<Window>, sample_window: Option<SampleWindow>) -> Result<(), String> {
    dump_sampled_masked(path, out_dir, names, apply_mask, apply_scale, unit_transform,
                        water_layer, window, sample_window, None)
}

#[allow(clippy::too_many_arguments)]
fn dump_sampled_masked(path: &Path, out_dir: &Path, names: &[String], apply_mask: bool,
                 apply_scale: bool, unit_transform: [f64; 2], water_layer: Option<&str>,
                 window: Option<Window>, sample_window: Option<SampleWindow>,
                 sample_mask: Option<&str>) -> Result<(), String> {
    if window.is_some() && sample_window.is_some() {
        return Err("exact and sampled windows cannot be combined".into());
    }
    if sample_mask.is_some() != sample_window.is_some_and(|s| s.method == SampleMethod::BilinearMasked) {
        return Err("a sample mask goes with bilinear-masked sampling, and only with it".into());
    }
    let transformed = unit_transform != [1.0, 0.0];
    if !unit_transform.iter().all(|v| v.is_finite()) || unit_transform[0] == 0.0 {
        return Err("unit transform must have a finite nonzero scale and finite offset".into());
    }
    let (file, metadata) = open(path)?;
    fs::create_dir_all(out_dir)
        .map_err(|error| format!("cannot create {}: {error}", out_dir.display()))?;
    // The class plane is decoded once and shared by every sampled variable.
    let mask_plane = match sample_mask {
        Some(mask_name) => {
            let array = file.read_array_f64(mask_name)
                .map_err(|error| format!("cannot decode sample mask {mask_name}: {error}"))?;
            let shape = array.shape().to_vec();
            if shape.len() < 2 || shape[..shape.len() - 2].iter().product::<usize>() != 1 {
                return Err(format!("sample mask {mask_name} must be one plane, got shape {shape:?}"));
            }
            Some((mask_name, array.into_values(), shape[shape.len() - 2], shape[shape.len() - 1]))
        }
        None => None,
    };
    let sample_mask = mask_plane.as_ref().map(|(name, values, ny, nx)|
        SampleMask { name: *name, values: values.as_slice(), ny: *ny, nx: *nx });

    // Variables decode concurrently (each one's chunks are already
    // decompressed in parallel underneath), gated by a byte budget so a
    // file of many large variables does not hold all of them at once.
    // Records and files keep the order and names the caller asked for,
    // and the first refusal in that order is the one reported.
    let options = DumpOptions { apply_mask, apply_scale, unit_transform, transformed,
                                water_layer, window, sample_window, sample_mask };
    let results = run_bounded(names.len(), |index| {
        let estimate = estimated_dump_bytes(&file, &names[index]);
        let _held = DUMP_BUDGET.acquire(estimate);
        dump_one(path, &file, out_dir, index, &names[index], &options)
    });
    let mut records = Vec::with_capacity(names.len());
    for result in results {
        records.push(result?);
    }

    let metadata = DumpMetadata {
        schema: DUMP_SCHEMA,
        metadata,
        variables: records,
    };
    let text = serde_json::to_string(&metadata)
        .map_err(|error| format!("cannot serialize dump metadata: {error}"))?;
    let target = out_dir.join("metadata.json");
    fs::write(&target, text)
        .map_err(|error| format!("cannot write {}: {error}", target.display()))?;
    Ok(())
}

struct DumpOptions<'a> {
    apply_mask: bool,
    apply_scale: bool,
    unit_transform: [f64; 2],
    transformed: bool,
    water_layer: Option<&'a str>,
    window: Option<Window>,
    sample_window: Option<SampleWindow>,
    sample_mask: Option<SampleMask<'a>>,
}

/// Decode one variable into `out_dir/<index>` and describe it.
fn dump_one(path: &Path, file: &netcrust::File, out_dir: &Path, index: usize, name: &str,
            options: &DumpOptions<'_>) -> Result<DumpRecord, String> {
    let DumpOptions { apply_mask, apply_scale, unit_transform, transformed, water_layer,
                      window, sample_window, sample_mask } = *options;
    // An HDF5 dataset the variable table does not list (a dimension
    // scale the reader hides -- see `recover_dimension_scales`) is
    // still reachable: `read_array_f64` falls back to a raw-HDF5 read
    // by name, so the values come either way; only the attribute
    // surface differs, which is why the CF lookups below go through a
    // closure that knows which path it is on.
    let variable = file.variable(name);
    if let Some(variable) = variable.as_ref() {
        if matches!(variable.dtype(), netcrust::DataType::Char | netcrust::DataType::String) {
            if transformed || water_layer.is_some() {
                return Err(format!("{name}: unit conversion requires a numeric variable"));
            }
            if window.is_some() || sample_window.is_some() {
                return Err(format!("{name}: --window requires a numeric variable"));
            }
            let mut shape = variable.shape();
            let mut dimensions: Vec<String> =
                variable.dimensions().iter().map(|d| d.name().to_string()).collect();
            let vlen = matches!(variable.dtype(), netcrust::DataType::String);
            let (values, string_width) = character_bytes(&file, variable, vlen)?;
            if let Some(width) = string_width {
                // NC_STRING: the texts have no character dimension of their
                // own, so the record gets one, the widest text, and the
                // reader sees the same (..., width) S1 array a classic
                // char variable gives it.
                shape.push(width);
                dimensions.push(format!("{name}_strlen"));
            }
            let filename = format!("{index:04}.chars");
            fs::write(out_dir.join(&filename), values)
                .map_err(|error| format!("cannot write {name}: {error}"))?;
            return Ok(DumpRecord {
                name: name.to_string(), filename, shape, dimensions,
                dtype: "|S1", units: None, times: None, cf: CfApplied::default(),
                unit_transform: None,
                water_layer_conversion: None,
                window: None,
                sample_window: None,
                sample_mask: None,
            });
        }
    } else if !file.has_hdf5_dataset(name) {
        return Err(format!("variable not found in {}: {name}", path.display()));
    }
    // An exact window of a listed variable reads only the chunks under
    // the window (the decoded values are the same numbers the full read
    // would cut out: masking, unpacking and unit conversion are
    // elementwise).  Layer water reads its thickness over the whole
    // variable and a sampled window needs neighbours past its edge, so
    // both still read everything.
    let window_read = match (window, variable.as_ref(), water_layer, sample_window) {
        (Some(window), Some(variable), None, None) => {
            let full: Vec<usize> = variable.shape();
            window.check(name, &full)?;
            Some(window.selection(&full))
        }
        _ => None,
    };
    let array = match window_read.as_ref() {
        Some(selection) => file.read_array_f64_slice(name, selection),
        None => file.read_array_f64(name),
    }
    .map_err(|error| format!("cannot decode {name}: {error}"))?;
    let mut shape = array.shape().to_vec();
    let mut values = array.into_values();

    // CF unpacking, applied HERE so Python receives decoded numbers.
    // Order matters and follows CF-1.x and netCDF4-python: a value is
    // tested against the fill markers in its STORED representation,
    // before scale_factor/add_offset are applied.  Applying the
    // packing first would compare a physical quantity against a
    // sentinel that was never in that space, which is how packed
    // fields silently keep their fill values as real data.
    // A recovered dimension scale exposes only the CF STRING
    // attributes reachable by name, so numeric CF attributes are
    // absent there rather than assumed to be zero/one.  Coordinate
    // axes are not packed in any format this reads, so that is a
    // real limitation and not a silent one: it is stated in the
    // inventory by `recovered_dimension_scale`.
    let number = |key: &str| -> Option<f64> {
        variable
            .as_ref()
            .and_then(|v| v.attribute(key))
            .and_then(|a| a.as_f64())
    };
    // `--raw` reports the attributes it did NOT act on, so the
    // record still says what the file declares while `applied`
    // says whether anything was done about it.
    let fill_value = number("_FillValue");
    let default_fill = match (&variable, fill_value) {
        (Some(variable), None) => nc_default_fill(variable.dtype()),
        _ => None,
    };
    let cf = CfApplied {
        scale_factor: number("scale_factor"),
        add_offset: number("add_offset"),
        fill_value,
        missing_value: number("missing_value"),
        default_fill,
        default_fill_rule: apply_mask,
        missing_count: 0,
        applied: apply_mask || apply_scale,
    };
    let mut missing_count = 0usize;
    for value in values.iter_mut() {
        if apply_mask {
            let is_missing = cf.fill_value.is_some_and(|fill| *value == fill)
                || cf.default_fill.is_some_and(|fill| *value == fill)
                || cf.missing_value.is_some_and(|marker| *value == marker);
            if is_missing || !value.is_finite() {
                if is_missing {
                    missing_count += 1;
                }
                *value = f64::NAN;
                continue;
            }
        }
        // Scaling is independent of masking, exactly as it is in
        // netCDF4-python: with masking off the sentinel itself comes
        // back, and it comes back scaled like every other element.
        if apply_scale {
            if let Some(scale) = cf.scale_factor {
                *value *= scale;
            }
            if let Some(offset) = cf.add_offset {
                *value += offset;
            }
        }
    }
    // Explicit quantity-unit conversion follows CF unpacking. Masked values
    // stay NaN; overflow of a finite input is a named decoding refusal.
    if transformed {
        for value in &mut values {
            if value.is_finite() {
                *value = *value * unit_transform[0] + unit_transform[1];
                if !value.is_finite() {
                    return Err(format!("{name}: unit conversion overflowed a finite value"));
                }
            }
        }
    }
    let water_layer_conversion = match water_layer {
        Some(thickness) => Some(water_layer::normalize(&file,
            variable.as_ref().ok_or_else(|| format!("{name}: layer water needs complete variable metadata"))?,
            &mut values, thickness)?),
        None => None,
    };
    let cf = CfApplied {
        missing_count,
        ..cf
    };
    // The window is cut from the decoded values, so masking, unpacking
    // and unit conversion are the same arithmetic with or without it.
    // `missing_count` counts the elements read: the window's when only
    // the window was read, the whole variable's otherwise.
    let applied_window = match window {
        Some(window) if window_read.is_some() => Some([window.i0, window.ni, window.j0, window.nj]),
        Some(window) => {
            let (cut, cut_shape) = window.apply(name, &values, &shape)?;
            values = cut;
            shape = cut_shape;
            Some([window.i0, window.ni, window.j0, window.nj])
        }
        None => None,
    };
    if let Some(sample) = sample_window {
        let (sampled, sampled_shape) = sample.apply_masked(name, &values, &shape, sample_mask)?;
        values = sampled;
        shape = sampled_shape;
    }

    // The index prefix keeps the filename unique and filesystem-safe
    // whatever the variable is called: NetCDF names may differ only
    // by case, which collides on Windows and macOS.
    let filename = format!("{index:04}.f64");
    let target: PathBuf = out_dir.join(&filename);
    let mut handle = fs::File::create(&target)
        .map_err(|error| format!("cannot write {}: {error}", target.display()))?;
    handle
        .write_all(&le_bytes(&values))
        .map_err(|error| format!("cannot write {}: {error}", target.display()))?;

    let units = match variable.as_ref() {
        Some(v) => v
            .attribute("units")
            .and_then(|attribute| attribute.as_string().map(str::to_string)),
        None => file.hdf5_dataset_attribute_string(name, "units"),
    };
    let times = units
        .as_deref()
        .map(parse_cf_units)
        .transpose()?
        .flatten()
        .map(|reference| decode_times(&reference, &values))
        .transpose()?;

    Ok(DumpRecord {
        name: name.to_string(),
        filename,
        shape,
        dimensions: variable
            .as_ref()
            .map(|v| {
                v.dimensions()
                    .iter()
                    .map(|dimension| dimension.name().to_string())
                    .collect()
            })
            .unwrap_or_default(),
        dtype: "<f8",
        units,
        times,
        cf,
        unit_transform: transformed.then_some(unit_transform),
        water_layer_conversion,
        window: applied_window,
        sample_window,
        sample_mask: sample_mask.map(|mask| mask.name.to_string()),
    })
}

/// Little-endian bytes of `values`, borrowed on a little-endian host.
fn le_bytes(values: &[f64]) -> std::borrow::Cow<'_, [u8]> {
    #[cfg(target_endian = "little")]
    {
        // SAFETY: f64 has no padding and any byte view of it is valid u8;
        // the slice covers exactly the values' bytes.
        std::borrow::Cow::Borrowed(unsafe {
            std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
        })
    }
    #[cfg(not(target_endian = "little"))]
    {
        std::borrow::Cow::Owned(values.iter().flat_map(|value| value.to_le_bytes()).collect())
    }
}

/// Bytes of decoded data one dump may hold at once across the variables
/// it decodes concurrently.  A single variable larger than this still
/// runs, alone.  Breakage it prevents: decoding the 93 4D variables of
/// NCAR's 2.5 km restart all at once would hold ~200 GB.
const DUMP_BUDGET_BYTES: u64 = 8 << 30;

static DUMP_BUDGET: ByteBudget = ByteBudget::new(DUMP_BUDGET_BYTES);

/// Peak bytes one variable's decode holds: the stored array, its f64
/// promotion and (on a window or sample) the cut copy.
fn estimated_dump_bytes(file: &netcrust::File, name: &str) -> u64 {
    file.variable(name)
        .map(|variable| variable.shape().iter().map(|&n| n as u64).product::<u64>())
        .unwrap_or(0)
        .saturating_mul(24)
}

struct ByteBudget {
    used: std::sync::Mutex<u64>,
    freed: std::sync::Condvar,
    cap: u64,
}

struct BudgetHold<'a> {
    budget: &'a ByteBudget,
    bytes: u64,
}

impl ByteBudget {
    const fn new(cap: u64) -> Self {
        Self { used: std::sync::Mutex::new(0), freed: std::sync::Condvar::new(), cap }
    }

    fn acquire(&self, bytes: u64) -> BudgetHold<'_> {
        let mut used = self.used.lock().unwrap_or_else(|poison| poison.into_inner());
        while *used > 0 && used.saturating_add(bytes) > self.cap {
            used = self.freed.wait(used).unwrap_or_else(|poison| poison.into_inner());
        }
        *used = used.saturating_add(bytes);
        BudgetHold { budget: self, bytes }
    }
}

impl Drop for BudgetHold<'_> {
    fn drop(&mut self) {
        let mut used = self.budget.used.lock().unwrap_or_else(|poison| poison.into_inner());
        *used = used.saturating_sub(self.bytes);
        self.budget.freed.notify_all();
    }
}

/// `work(0..count)` on up to 16 scoped threads, results in index order.
fn run_bounded<T: Send>(count: usize, work: impl Fn(usize) -> T + Sync) -> Vec<T> {
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get()).min(16).min(count);
    if threads <= 1 {
        return (0..count).map(&work).collect();
    }
    let next = std::sync::atomic::AtomicUsize::new(0);
    let slots: Vec<std::sync::Mutex<Option<T>>> =
        (0..count).map(|_| std::sync::Mutex::new(None)).collect();
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| loop {
                let index = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if index >= count {
                    break;
                }
                let value = work(index);
                *slots[index].lock().unwrap_or_else(|poison| poison.into_inner()) = Some(value);
            });
        }
    });
    slots
        .into_iter()
        .map(|slot| {
            slot.into_inner()
                .unwrap_or_else(|poison| poison.into_inner())
                .expect("every index ran")
        })
        .collect()
}

/// A CF reference time: the unit the values are counted in, and the
/// instant they are counted from.
#[derive(Debug)]
struct CfReference {
    unit: CfUnit,
    epoch: DateTime<Utc>,
}

#[derive(Debug)]
enum CfUnit {
    Seconds,
    Minutes,
    Hours,
    Days,
}

/// Parse `"<unit> since <timestamp>"`.  `Ok(None)` when the units
/// string is not a CF reference time at all (which is the common case
/// -- `K`, `m s-1`); `Err` when it plainly is one but its UTC-offset
/// designator is malformed, because dumping such a variable without its
/// time axis would silently strip the calendar off a file whose author
/// wrote one down.
///
/// Deliberately strict about the unit and forgiving about the timestamp
/// spelling, because that is where real files vary: `1970-01-01`,
/// `1970-01-01 00:00:0.0`, `1970-01-01T00:00:00Z`, and a trailing
/// `+05:30` or ` UTC` all appear in published archives.  A well-formed
/// UTC offset is APPLIED -- the epoch converts to UTC -- so the decoded
/// instants land where the equivalent UTC-spelled units would put them.
fn parse_cf_units(units: &str) -> Result<Option<CfReference>, String> {
    let lowered = units.trim().to_ascii_lowercase();
    let Some((unit_text, rest)) = lowered.split_once(" since ") else {
        return Ok(None);
    };
    let unit = match unit_text.trim() {
        "s" | "sec" | "secs" | "second" | "seconds" => CfUnit::Seconds,
        "min" | "mins" | "minute" | "minutes" => CfUnit::Minutes,
        "h" | "hr" | "hrs" | "hour" | "hours" => CfUnit::Hours,
        "d" | "day" | "days" => CfUnit::Days,
        _ => return Ok(None),
    };
    Ok(parse_cf_epoch(rest.trim())?.map(|epoch| CfReference { unit, epoch }))
}

/// The widest UTC-offset designator accepted, in seconds: 18 hours
/// either side.  The widest civil offset on Earth is +14:00 (the Line
/// Islands) and the udunits/`java.time` grammars both cap the field at
/// +/-18:00; a larger number is a mistyped timestamp, not a timezone,
/// and applying it would move every decoded instant by most of a day.
const MAX_UTC_OFFSET_SECONDS: i64 = 18 * 3600;

/// `Ok(None)` when the text is not a timestamp this parser reads;
/// `Err` when its UTC-offset designator cannot be trusted, saying why.
fn parse_cf_epoch(text: &str) -> Result<Option<DateTime<Utc>>, String> {
    let mut stamp = text.trim();
    // Textual zone designators, every one of which NAMES the zero
    // offset.  udunits reads them; so do published archives.
    let mut named_utc = false;
    for suffix in [" utc", "z", " gmt"] {
        if let Some(head) = stamp.strip_suffix(suffix) {
            stamp = head.trim();
            named_utc = true;
        }
    }
    let body = stamp.replace('t', " ");
    let mut tokens = body.split_whitespace();
    let Some(date_text) = tokens.next() else {
        return Ok(None);
    };
    // A signed token is a numeric UTC offset -- a time of day is never
    // signed.  The offset may ride attached to the tail of the time
    // token (`00:00:00+05:30`) or stand alone after the date or time
    // (`1992-10-8 15:15:42.5 -6:00`, the canonical udunits spelling).
    // It is never recognised attached to a bare date, whose own `-`
    // separators make that spelling unreadable without guessing.
    let (time_text, offset_text) = match (tokens.next(), tokens.next(), tokens.next()) {
        (None, _, _) => ("", None),
        (Some(second), None, _) => match second.find(['+', '-']) {
            Some(at) => (&second[..at], Some(&second[at..])),
            None => (second, None),
        },
        (Some(second), Some(third), None) => {
            if !third.starts_with(['+', '-']) {
                return Ok(None);
            }
            if second.contains(['+', '-']) {
                return Err(format!(
                    "reference time {text:?} carries two UTC offsets; \
                     one epoch cannot be shifted twice"
                ));
            }
            (second, Some(third))
        }
        _ => return Ok(None),
    };
    let offset_seconds = match offset_text {
        Some(token) => {
            if named_utc {
                return Err(format!(
                    "reference time {text:?} names UTC and also carries \
                     the numeric offset {token:?}; an epoch with two zone \
                     designators cannot be placed on the timeline"
                ));
            }
            parse_utc_offset(token, text)?
        }
        None => 0,
    };
    let date = match NaiveDate::parse_from_str(date_text, "%Y-%m-%d")
        .or_else(|_| NaiveDate::parse_from_str(date_text, "%Y-%-m-%-d"))
    {
        Ok(date) => date,
        Err(_) => return Ok(None),
    };
    let time = if time_text.is_empty() {
        NaiveTime::from_hms_opt(0, 0, 0).expect("midnight is a valid time")
    } else {
        match ["%H:%M:%S%.f", "%H:%M:%S", "%H:%M", "%H"]
            .iter()
            .find_map(|format| NaiveTime::parse_from_str(time_text, format).ok())
        {
            Some(time) => time,
            None => return Ok(None),
        }
    };
    // A timestamp at +05:00 reads five hours AHEAD of UTC, so the UTC
    // instant it names is the naive reading minus the offset.
    let epoch = NaiveDateTime::new(date, time) - Duration::seconds(offset_seconds);
    Ok(Some(epoch.and_utc()))
}

/// One numeric UTC-offset designator -- a sign, then `hh:mm`, `hhmm`,
/// or a bare hour count `h`/`hh` -- as signed seconds east of UTC.
/// These are the spellings udunits reads.  Anything else is refused
/// with the reason, because a mis-read offset does not crash: it
/// silently moves every instant decoded from the file.
fn parse_utc_offset(token: &str, timestamp: &str) -> Result<i64, String> {
    let malformed = |why: String| {
        format!(
            "UTC offset {token:?} in reference time {timestamp:?} is \
             malformed: {why}"
        )
    };
    let (sign, magnitude) = if let Some(rest) = token.strip_prefix('+') {
        (1i64, rest)
    } else if let Some(rest) = token.strip_prefix('-') {
        (-1i64, rest)
    } else {
        return Err(malformed("it does not start with a sign".into()));
    };
    if magnitude.is_empty() {
        return Err(malformed("the sign has no digits behind it".into()));
    }
    let digits = |text: &str| text.bytes().all(|byte| byte.is_ascii_digit());
    let (hour_text, minute_text) = match magnitude.split_once(':') {
        Some((hours, minutes)) => {
            if minutes.len() != 2 {
                return Err(malformed(format!(
                    "minutes must be exactly two digits, got {minutes:?}; \
                     an offset carries no seconds field"
                )));
            }
            (hours, minutes)
        }
        None => match magnitude.len() {
            1 | 2 => (magnitude, "0"),
            4 => magnitude.split_at(2),
            _ => {
                return Err(malformed(format!(
                    "a colonless offset must be one or two hour digits or \
                     exactly four digits (hhmm); {magnitude:?} cannot be \
                     split into hours and minutes without guessing"
                )));
            }
        },
    };
    if hour_text.is_empty() || hour_text.len() > 2 || !digits(hour_text) || !digits(minute_text) {
        return Err(malformed(format!(
            "{hour_text:?} hours and {minute_text:?} minutes are not one- \
             or two-digit numbers"
        )));
    }
    let hours: i64 = hour_text.parse().expect("checked ascii digits");
    let minutes: i64 = minute_text.parse().expect("checked ascii digits");
    if minutes >= 60 {
        return Err(malformed(format!(
            "{minutes} is not a minute count below 60"
        )));
    }
    let magnitude_seconds = hours * 3600 + minutes * 60;
    if magnitude_seconds > MAX_UTC_OFFSET_SECONDS {
        return Err(malformed(
            "its magnitude passes 18:00, and no timezone is that far from \
             UTC (the widest civil offset on Earth is +14:00)"
            .into(),
        ));
    }
    Ok(sign * magnitude_seconds)
}

fn decode_times(reference: &CfReference, values: &[f64]) -> Result<Vec<String>, String> {
    values
        .iter()
        .map(|&value| {
            if !value.is_finite() {
                return Err(format!("time coordinate is not finite: {value}"));
            }
            // Split whole seconds from the fractional part. Converting the
            // entire reference offset to i64 nanoseconds incorrectly excludes
            // valid distant epochs such as hours since 0001-01-01.
            let seconds = match reference.unit {
                CfUnit::Seconds => value,
                CfUnit::Minutes => value * 60.0,
                CfUnit::Hours => value * 3600.0,
                CfUnit::Days => value * 86_400.0,
            };
            if !seconds.is_finite() || seconds.abs() >= i64::MAX as f64 {
                return Err(format!("time coordinate out of representable range: {value}"));
            }
            let whole = Duration::try_seconds(seconds.trunc() as i64)
                .ok_or_else(|| format!("time coordinate out of representable range: {value}"))?;
            let fraction = Duration::nanoseconds((seconds.fract() * 1e9).round() as i64);
            let delta = whole.checked_add(&fraction)
                .ok_or_else(|| format!("time coordinate out of representable range: {value}"))?;
            let instant = reference
                .epoch
                .checked_add_signed(delta)
                .ok_or_else(|| format!("time coordinate out of range: {value}"))?;
            Ok(instant.format("%Y-%m-%dT%H:%M:%S%.fZ").to_string())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A parsed CF reference, for units that must be well-formed ones.
    fn reference(units: &str) -> CfReference {
        parse_cf_units(units)
            .unwrap_or_else(|error| panic!("{units}: {error}"))
            .unwrap_or_else(|| panic!("{units} is a CF reference time"))
    }

    #[test]
    fn plain_units_are_not_reference_times() {
        for units in ["K", "m s-1", "kg kg-1"] {
            assert!(
                parse_cf_units(units).expect("nothing to malform").is_none(),
                "{units}"
            );
        }
    }

    #[test]
    fn hours_since_epoch_decodes() {
        let reference = reference("hours since 1970-01-01 00:00:00");
        let times = decode_times(&reference, &[0.0, 1.5]).expect("decoded");
        assert_eq!(times[0], "1970-01-01T00:00:00Z");
        assert_eq!(times[1], "1970-01-01T01:30:00Z");
    }

    #[test]
    fn distant_reference_and_fractional_offsets_decode() {
        let ancient = reference("hours since 0001-01-01 00:00:00.0");
        assert_eq!(decode_times(&ancient, &[17_739_672.0]).unwrap(),
                   vec!["2024-09-25T00:00:00Z"]);
        let modern = reference("seconds since 2024-09-25 00:00:00");
        assert_eq!(decode_times(&modern, &[-0.125, 0.125]).unwrap(),
                   vec!["2024-09-24T23:59:59.875Z", "2024-09-25T00:00:00.125Z"]);
        for value in [f64::INFINITY, f64::NAN, f64::MAX] {
            assert!(decode_times(&modern, &[value]).is_err());
        }
    }

    #[test]
    fn iso_and_zulu_spellings_decode() {
        for spelling in [
            "hours since 2024-03-01T00:00:00Z",
            "hours since 2024-03-01 00:00:00",
            "hours since 2024-03-01",
            "hours since 2024-03-01 00:00:0.0",
        ] {
            let reference = reference(spelling);
            let times = decode_times(&reference, &[6.0]).expect("decoded");
            assert_eq!(times[0], "2024-03-01T06:00:00Z", "{spelling}");
        }
    }

    /// The one exact equivalence the offset handling promises: units
    /// carrying a UTC-offset designator decode to the SAME instants as
    /// the same epoch respelled in UTC by hand.  Every offset spelling
    /// udunits reads is here -- `+hh:mm`, `+hhmm`, bare `-h`/`-hh`,
    /// attached and spaced, and the zero-offset designators.
    #[test]
    fn offset_references_decode_like_their_utc_respelling() {
        for (offset_spelling, utc_respelling) in [
            // +05:00 reads five hours EAST: the epoch is EARLIER in UTC.
            (
                "hours since 2020-01-01 00:00:00 +05:00",
                "hours since 2019-12-31 19:00:00",
            ),
            (
                "hours since 2020-01-01 00:00:00+0500",
                "hours since 2019-12-31 19:00:00",
            ),
            // Negative forms read WEST: the epoch is LATER in UTC.
            (
                "hours since 2020-01-01 00:00:00 -06:00",
                "hours since 2020-01-01 06:00:00",
            ),
            (
                "hours since 2020-01-01 00:00:00-0600",
                "hours since 2020-01-01 06:00:00",
            ),
            (
                "hours since 2020-01-01 00:00:00 -6",
                "hours since 2020-01-01 06:00:00",
            ),
            // A half-hour zone exercises both halves of h*3600 + m*60.
            (
                "hours since 2020-01-01 00:00:00 +05:30",
                "hours since 2019-12-31 18:30:00",
            ),
            (
                "hours since 2020-01-01 00:00:00+0530",
                "hours since 2019-12-31 18:30:00",
            ),
            // Zero-offset designators are exactly UTC.
            (
                "hours since 2020-01-01 00:00:00Z",
                "hours since 2020-01-01 00:00:00",
            ),
            (
                "hours since 2020-01-01 00:00:00 +00:00",
                "hours since 2020-01-01 00:00:00",
            ),
            (
                "hours since 2020-01-01 00:00:00-0000",
                "hours since 2020-01-01 00:00:00",
            ),
            // A date-only stamp still takes a spaced offset (midnight).
            (
                "hours since 2020-01-01 +01:00",
                "hours since 2019-12-31 23:00:00",
            ),
        ] {
            let with_offset = reference(offset_spelling);
            let respelled = reference(utc_respelling);
            assert_eq!(
                with_offset.epoch, respelled.epoch,
                "{offset_spelling} vs {utc_respelling}"
            );
            assert_eq!(
                decode_times(&with_offset, &[0.0, 1.5]).expect("decoded"),
                decode_times(&respelled, &[0.0, 1.5]).expect("decoded"),
                "{offset_spelling}"
            );
        }
    }

    /// Ground truth for the shift direction, pinned as absolute strings
    /// so a sign error that broke both sides of the equivalence test the
    /// same way would still be caught.
    #[test]
    fn a_positive_offset_epoch_lands_earlier_on_the_utc_timeline() {
        let reference = reference("hours since 2020-01-01 00:00:00 +05:30");
        assert_eq!(
            decode_times(&reference, &[0.0, 5.5]).expect("decoded"),
            vec!["2019-12-31T18:30:00Z", "2020-01-01T00:00:00Z"],
        );
    }

    #[test]
    fn a_negative_offset_epoch_lands_later_on_the_utc_timeline() {
        let reference = reference("seconds since 2020-06-01 12:00:00 -06");
        assert_eq!(
            decode_times(&reference, &[0.0]).expect("decoded")[0],
            "2020-06-01T18:00:00Z",
        );
    }

    /// The designator bounds: 59 minutes and 18:00 are the last values
    /// inside the grammar, and the first value past each is refused by
    /// name.  udunits and java.time both cap the field at +/-18:00; the
    /// widest civil offset on Earth is +14:00.
    #[test]
    fn offset_bounds_admit_18_hours_and_59_minutes_and_nothing_past_them() {
        assert_eq!(
            reference("hours since 2020-01-01 00:00:00 +18:00").epoch,
            reference("hours since 2019-12-31 06:00:00").epoch,
        );
        assert_eq!(
            reference("hours since 2020-01-01 00:59:00 +00:59").epoch,
            reference("hours since 2020-01-01 00:00:00").epoch,
        );
        for (units, names) in [
            ("hours since 2020-01-01 00:00:00 +18:01", "18:00"),
            ("hours since 2020-01-01 00:00:00 -19:00", "18:00"),
            ("hours since 2020-01-01 00:00:00 +05:60", "below 60"),
            ("hours since 2020-01-01 00:00:00 +05:61", "below 60"),
        ] {
            let error = match parse_cf_units(units) {
                Err(error) => error,
                Ok(parsed) => panic!("{units} was not refused: {parsed:?}"),
            };
            assert!(error.contains("malformed"), "{units}: {error}");
            assert!(error.contains(names), "{units}: {error}");
        }
    }

    /// Every refusal names what is wrong; nothing is guessed at and
    /// nothing is silently dropped.  Silently dropping was the old
    /// behaviour, and it stripped the time axis off any file whose
    /// author spelled a real timezone.
    #[test]
    fn malformed_offsets_are_refused_with_the_reason() {
        for (units, names) in [
            // Three packed digits cannot be split into hours and
            // minutes without guessing between 5:30 and 53:0.
            ("hours since 2020-01-01 00:00:00 +530", "guessing"),
            ("hours since 2020-01-01 00:00:00 +05300", "guessing"),
            // udunits offsets carry no seconds field.
            ("hours since 2020-01-01 00:00:00 +05:30:00", "two digits"),
            ("hours since 2020-01-01 00:00:00 +05:3", "two digits"),
            // A sign with nothing behind it, a missing hours field, and
            // letters where digits go.
            ("hours since 2020-01-01 00:00:00 +", "no digits"),
            ("hours since 2020-01-01 00:00:00 +:30", "hours"),
            ("hours since 2020-01-01 00:00:00 +0a", "hours"),
            // Two zone designators cannot both place the epoch.
            ("hours since 2020-01-01 00:00:00+05:00Z", "names utc"),
            (
                "hours since 2020-01-01 00:00:00+05:00 +06:00",
                "two utc offsets",
            ),
        ] {
            let error = match parse_cf_units(units) {
                Err(error) => error,
                Ok(parsed) => panic!("{units} was not refused: {parsed:?}"),
            };
            assert!(
                error.to_ascii_lowercase().contains(names),
                "{units}: {error}"
            );
        }
    }

    /// `minutes since` is a spelling this decoder claimed and never
    /// checked.
    ///
    /// Both halves of `value * 60.0 * 1e9` and the whole
    /// `"min" | "mins" | "minute" | "minutes"` match arm could be
    /// deleted or turned into another operator without a single test
    /// noticing, which is a history file whose valid times are wrong by
    /// a factor of sixty and an initial condition read off the wrong
    /// frame.  The 90-minute value is the one that separates them: it
    /// decodes to 01:30 under a multiply, to 00:01:00.00000009 under an
    /// add, and to 00:00:00.0000000015 under a divide.
    #[test]
    fn minutes_since_decodes_as_minutes_and_not_as_some_other_arithmetic() {
        let minutes = reference("minutes since 2024-03-01 00:00:00");
        let times = decode_times(&minutes, &[0.0, 90.0]).expect("decoded");
        assert_eq!(times[0], "2024-03-01T00:00:00Z");
        assert_eq!(times[1], "2024-03-01T01:30:00Z");
        for spelling in ["min", "mins", "minute", "minutes"] {
            let spelled = reference(&format!("{spelling} since 2024-03-01"));
            assert_eq!(
                decode_times(&spelled, &[90.0]).expect("decoded")[0],
                "2024-03-01T01:30:00Z",
                "{spelling}"
            );
        }
    }

    /// The unpadded date spelling, which the epoch parser carries a
    /// second format for and no test ever reached.
    #[test]
    fn unpadded_month_and_day_spellings_decode() {
        let days = reference("days since 2000-1-1");
        assert_eq!(
            decode_times(&days, &[1.5]).expect("decoded")[0],
            "2000-01-02T12:00:00Z"
        );
        let hours = reference("hours since 2000-1-1 6:30");
        assert_eq!(
            decode_times(&hours, &[0.0]).expect("decoded")[0],
            "2000-01-01T06:30:00Z"
        );
    }

    #[test]
    fn days_and_seconds_decode() {
        let days = reference("days since 2000-01-01");
        assert_eq!(
            decode_times(&days, &[1.5]).expect("decoded")[0],
            "2000-01-02T12:00:00Z"
        );
        let seconds = reference("seconds since 2000-01-01");
        assert_eq!(
            decode_times(&seconds, &[90.0]).expect("decoded")[0],
            "2000-01-01T00:01:30Z"
        );
    }

    // ------------------------------------------------------------------
    // JSON attribute encoding.  The single/plural distinction below is a
    // CONTRACT with the Python side: netCDF4-python hands back a scalar
    // for a one-element attribute and an array otherwise, and selector
    // resolution compares `units` and `standard_name` against strings.
    // An attribute that flips shape stops resolving without an error.

    #[test]
    fn char_and_single_string_attributes_encode_as_json_strings() {
        use netcrust::AttributeValue as V;
        assert_eq!(
            attribute_json(&V::Chars("degrees_east".into())),
            serde_json::json!("degrees_east")
        );
        assert_eq!(
            attribute_json(&V::Strings(vec!["longitude".into()])),
            serde_json::json!("longitude")
        );
    }

    #[test]
    fn plural_string_attributes_encode_as_json_arrays() {
        use netcrust::AttributeValue as V;
        assert_eq!(
            attribute_json(&V::Strings(vec!["a".into(), "b".into()])),
            serde_json::json!(["a", "b"])
        );
    }

    #[test]
    fn single_numbers_are_scalars_and_plural_numbers_are_arrays() {
        use netcrust::AttributeValue as V;
        assert_eq!(attribute_json(&V::Ints(vec![7])), serde_json::json!(7.0));
        assert_eq!(
            attribute_json(&V::Floats(vec![1.5, 2.5])),
            serde_json::json!([1.5, 2.5])
        );
        assert_eq!(
            attribute_json(&V::Doubles(vec![0.5])),
            serde_json::json!(0.5)
        );
        // JSON has no NaN; the numeric path must emit null, not panic.
        assert_eq!(
            attribute_json(&V::Doubles(vec![f64::NAN])),
            serde_json::Value::Null
        );
    }

    /// i64/u64 take their own arms because f64 rounds them beyond 2^53;
    /// each arm has the same single/plural guard as the string arm.
    #[test]
    fn wide_integers_encode_exactly_with_the_same_shape_rule() {
        use netcrust::AttributeValue as V;
        let big = (1i64 << 53) + 1;
        assert_eq!(
            attribute_json(&V::Int64s(vec![big])),
            serde_json::json!(9007199254740993i64)
        );
        assert_eq!(
            attribute_json(&V::Int64s(vec![1, 2])),
            serde_json::json!([1, 2])
        );
        assert_eq!(
            attribute_json(&V::UInt64s(vec![u64::MAX])),
            serde_json::json!(18446744073709551615u64)
        );
        assert_eq!(
            attribute_json(&V::UInt64s(vec![3, 4])),
            serde_json::json!([3, 4])
        );
    }

    // ------------------------------------------------------------------
    // File-backed decode contracts.  The classic fixtures are BUILT here
    // by the workspace's own writer (an independent implementation from
    // the netcrust read path under test); the NetCDF-4/HDF5 fixtures are
    // checked in under tests/fixtures because nothing in the Rust stack
    // writes HDF5, with tests/fixtures/make_fixtures.py to regenerate.

    fn fixture(name: &str) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("fixtures")
            .join(name)
    }

    /// A scratch directory unique to one test, wiped at entry so a
    /// failed earlier run cannot leak state into this one.
    fn scratch(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir()
            .join(format!("rw-netcdf-suite-{}-{tag}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("create scratch dir");
        dir
    }

    /// One classic (CDF-1) file carrying the whole CF unpack surface:
    /// a packed short with both fill markers, a CF time coordinate, a
    /// char variable (the named refusal), and two global attributes.
    ///
    /// Data layout of `packed`, stored as i16:
    ///   [-32768 (_FillValue), -32767 (missing_value), 4, 6]
    /// so with scale_factor 0.5 and add_offset 100 the decoded truth is
    ///   [NaN, NaN, 102.0, 103.0]      (mask on, scale on)
    ///   [-32768.0, -32767.0, 4.0, 6.0] (raw)
    ///   [-16284.0, -16283.5, 102.0, 103.0] (mask off, scale on)
    fn write_packed_classic(path: &Path) {
        use netcdf_writer::{AttrValue, NcFormat, NcType, NcWriter, Schema, VarData};
        let mut schema = Schema::new(NcFormat::Classic);
        let x = schema.def_dim("x", 4, false).expect("dim");
        schema
            .put_global_attr("title", AttrValue::Text("packed fixture".into()))
            .expect("gattr");
        schema
            .put_global_attr("version", AttrValue::Ints(vec![3]))
            .expect("gattr");
        let packed = schema.def_var("packed", NcType::Short, &[x]).expect("var");
        schema
            .put_var_attr(packed, "scale_factor", AttrValue::Doubles(vec![0.5]))
            .expect("attr");
        schema
            .put_var_attr(packed, "add_offset", AttrValue::Doubles(vec![100.0]))
            .expect("attr");
        schema
            .put_var_attr(packed, "_FillValue", AttrValue::Shorts(vec![-32768]))
            .expect("attr");
        schema
            .put_var_attr(packed, "missing_value", AttrValue::Shorts(vec![-32767]))
            .expect("attr");
        let hours = schema.def_var("hours", NcType::Double, &[x]).expect("var");
        schema
            .put_var_attr(
                hours,
                "units",
                AttrValue::Text("hours since 2024-03-01 00:00:00".into()),
            )
            .expect("attr");
        let label = schema.def_var("label", NcType::Char, &[x]).expect("var");
        let mut writer = NcWriter::create(path, schema).expect("create");
        writer
            .write_var(packed, VarData::I16(&[-32768, -32767, 4, 6]))
            .expect("write packed");
        writer
            .write_var(hours, VarData::F64(&[0.0, 1.5, 24.0, 36.0]))
            .expect("write hours");
        writer
            .write_var(label, VarData::Char(b"abcd"))
            .expect("write label");
        writer.finish().expect("finish");
    }

    fn read_f64_plane(path: &Path) -> Vec<f64> {
        let bytes = fs::read(path).expect("read plane");
        assert_eq!(bytes.len() % 8, 0, "not a whole number of f64s");
        bytes
            .chunks_exact(8)
            .map(|chunk| f64::from_le_bytes(chunk.try_into().unwrap()))
            .collect()
    }

    fn read_metadata(out_dir: &Path) -> serde_json::Value {
        let text = fs::read_to_string(out_dir.join("metadata.json"))
            .expect("read metadata.json");
        serde_json::from_str(&text).expect("parse metadata.json")
    }

    fn record<'a>(metadata: &'a serde_json::Value, name: &str) -> &'a serde_json::Value {
        metadata["variables"]
            .as_array()
            .expect("variables array")
            .iter()
            .find(|record| record["name"] == name)
            .unwrap_or_else(|| panic!("{name} not in metadata"))
    }

    /// A float written with a masked element and no `_FillValue` stores
    /// the library's default fill, 9.969e36.  That is a finite number, so
    /// a reader that only honours declared markers hands it on as data
    /// (a 9.97e36 K temperature).  Masked on, it is missing; raw, it is
    /// the stored number; a declared `_FillValue` replaces the default.
    #[test]
    fn an_undeclared_default_fill_is_missing() {
        use netcdf_writer::{AttrValue, NcFormat, NcType, NcWriter, Schema, VarData};
        let dir = scratch("default-fill");
        let source = dir.join("fill.nc");
        let mut schema = Schema::new(NcFormat::Classic);
        let x = schema.def_dim("x", 3, false).expect("dim");
        let t = schema.def_var("T", NcType::Float, &[x]).expect("var");
        let d = schema.def_var("D", NcType::Double, &[x]).expect("var");
        let declared = schema.def_var("declared", NcType::Float, &[x]).expect("var");
        schema
            .put_var_attr(declared, "_FillValue", AttrValue::Floats(vec![-999.0]))
            .expect("attr");
        let mut writer = NcWriter::create(&source, schema).expect("create");
        let fill32 = 9.969_209_968_386_869e36_f32;
        writer.write_var(t, VarData::F32(&[1.0, fill32, 3.0])).expect("T");
        writer.write_var(d, VarData::F64(&[9.969_209_968_386_869e36, 2.0, 3.0])).expect("D");
        writer.write_var(declared, VarData::F32(&[fill32, -999.0, 3.0])).expect("declared");
        writer.finish().expect("finish");

        let out = dir.join("out");
        dump(&source, &out, &["T".into(), "D".into(), "declared".into()], true, true)
            .expect("dump");
        let t_values = read_f64_plane(&out.join("0000.f64"));
        assert_eq!(t_values[0], 1.0);
        assert!(t_values[1].is_nan(), "the f32 default fill must mask");
        let d_values = read_f64_plane(&out.join("0001.f64"));
        assert!(d_values[0].is_nan(), "the f64 default fill must mask");
        let declared_values = read_f64_plane(&out.join("0002.f64"));
        assert!(declared_values[0].is_finite(), "a declared fill replaces the default");
        assert!(declared_values[1].is_nan());
        let metadata = read_metadata(&out);
        assert_eq!(record(&metadata, "T")["cf"]["missing_count"], 1);
        assert_eq!(record(&metadata, "T")["cf"]["default_fill_rule"], true);
        assert!(record(&metadata, "declared")["cf"].get("default_fill").is_none());

        let raw = dir.join("raw");
        dump(&source, &raw, &["T".into()], false, false).expect("raw dump");
        assert!(read_f64_plane(&raw.join("0000.f64"))[1] > 9.0e36);
    }

    #[test]
    fn default_dump_masks_in_stored_space_then_scales() {
        let dir = scratch("dump-default");
        let source = dir.join("packed.nc");
        write_packed_classic(&source);
        let out = dir.join("out");
        dump(&source, &out, &["packed".into()], true, true).expect("dump");

        let values = read_f64_plane(&out.join("0000.f64"));
        assert!(values[0].is_nan(), "fill value must mask to NaN");
        assert!(values[1].is_nan(), "missing value must mask to NaN");
        assert_eq!(&values[2..], &[102.0, 103.0]);

        let metadata = read_metadata(&out);
        assert_eq!(metadata["schema"], "gpuwm-rw-netcdf-dump-v1");
        let packed = record(&metadata, "packed");
        assert_eq!(packed["filename"], "0000.f64");
        assert_eq!(packed["shape"], serde_json::json!([4]));
        assert_eq!(packed["dimensions"], serde_json::json!(["x"]));
        assert_eq!(packed["dtype"], "<f8");
        assert_eq!(packed["cf"]["scale_factor"], 0.5);
        assert_eq!(packed["cf"]["add_offset"], 100.0);
        assert_eq!(packed["cf"]["fill_value"], -32768.0);
        assert_eq!(packed["cf"]["missing_value"], -32767.0);
        assert_eq!(packed["cf"]["missing_count"], 2);
        assert_eq!(packed["cf"]["applied"], true);
        assert!(
            packed.get("times").is_none(),
            "a packed physical variable has no time axis"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn raw_dump_preserves_stored_sentinels_and_reports_not_applied() {
        let dir = scratch("dump-raw");
        let source = dir.join("packed.nc");
        write_packed_classic(&source);
        let out = dir.join("out");
        dump(&source, &out, &["packed".into()], false, false).expect("dump");

        assert_eq!(
            read_f64_plane(&out.join("0000.f64")),
            &[-32768.0, -32767.0, 4.0, 6.0]
        );
        let metadata = read_metadata(&out);
        let packed = record(&metadata, "packed");
        // The declared attributes still travel; `applied` says nothing
        // was done about them, and nothing was masked.
        assert_eq!(packed["cf"]["scale_factor"], 0.5);
        assert_eq!(packed["cf"]["missing_count"], 0);
        assert_eq!(packed["cf"]["applied"], false);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn no_mask_dump_scales_the_surviving_sentinels_like_netcdf4_python() {
        let dir = scratch("dump-nomask");
        let source = dir.join("packed.nc");
        write_packed_classic(&source);
        let out = dir.join("out");
        dump(&source, &out, &["packed".into()], false, true).expect("dump");

        assert_eq!(
            read_f64_plane(&out.join("0000.f64")),
            &[-16284.0, -16283.5, 102.0, 103.0]
        );
        let metadata = read_metadata(&out);
        let packed = record(&metadata, "packed");
        assert_eq!(packed["cf"]["missing_count"], 0);
        assert_eq!(packed["cf"]["applied"], true);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_cf_time_variable_gets_decoded_instants_in_the_metadata() {
        let dir = scratch("dump-times");
        let source = dir.join("packed.nc");
        write_packed_classic(&source);
        let out = dir.join("out");
        dump(&source, &out, &["hours".into()], true, true).expect("dump");

        assert_eq!(
            read_f64_plane(&out.join("0000.f64")),
            &[0.0, 1.5, 24.0, 36.0]
        );
        let hours = read_metadata(&out);
        let hours = record(&hours, "hours");
        assert_eq!(hours["units"], "hours since 2024-03-01 00:00:00");
        assert_eq!(
            hours["times"],
            serde_json::json!([
                "2024-03-01T00:00:00Z",
                "2024-03-01T01:30:00Z",
                "2024-03-02T00:00:00Z",
                "2024-03-02T12:00:00Z",
            ])
        );
        let _ = fs::remove_dir_all(&dir);
    }

    /// A classic file whose single `hours` variable carries *units*.
    fn write_hours_classic(path: &Path, units: &str, values: &[f64]) {
        use netcdf_writer::{AttrValue, NcFormat, NcType, NcWriter, Schema, VarData};
        let mut schema = Schema::new(NcFormat::Classic);
        let x = schema.def_dim("x", values.len(), false).expect("dim");
        let hours = schema.def_var("hours", NcType::Double, &[x]).expect("var");
        schema
            .put_var_attr(hours, "units", AttrValue::Text(units.into()))
            .expect("attr");
        let mut writer = NcWriter::create(path, schema).expect("create");
        writer.write_var(hours, VarData::F64(values)).expect("write");
        writer.finish().expect("finish");
    }

    /// The writer-path round trip for an offset-bearing reference: the
    /// units travel through the workspace's own writer, back through
    /// the netcrust read path, and decode to the same UTC instants the
    /// UTC respelling of that epoch would give.
    #[test]
    fn a_written_offset_reference_dumps_utc_instants() {
        let dir = scratch("dump-offset");
        let source = dir.join("offset.nc");
        write_hours_classic(
            &source,
            "hours since 2024-03-01 05:30:00 +05:30",
            &[0.0, 12.0],
        );
        let out = dir.join("out");
        dump(&source, &out, &["hours".into()], true, true).expect("dump");

        let metadata = read_metadata(&out);
        let hours = record(&metadata, "hours");
        assert_eq!(hours["units"], "hours since 2024-03-01 05:30:00 +05:30");
        assert_eq!(
            hours["times"],
            serde_json::json!(["2024-03-01T00:00:00Z", "2024-03-01T12:00:00Z"])
        );
        let _ = fs::remove_dir_all(&dir);
    }

    /// A malformed offset fails the whole dump with its sentence, so a
    /// mis-spelled zone can never silently strip a file's time axis.
    #[test]
    fn a_malformed_offset_fails_the_dump_by_name() {
        let dir = scratch("dump-badoffset");
        let source = dir.join("bad.nc");
        write_hours_classic(&source, "hours since 2024-03-01 00:00:00 +19:00", &[0.0]);
        let out = dir.join("out");
        let error = dump(&source, &out, &["hours".into()], true, true)
            .expect_err("a malformed offset must fail the dump");
        assert!(error.contains("+19:00"), "{error}");
        assert!(error.contains("18:00"), "{error}");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn char_variables_preserve_bytes_and_missing_variables_are_refused() {
        let dir = scratch("dump-refusals");
        let source = dir.join("packed.nc");
        write_packed_classic(&source);
        let out = dir.join("out");

        dump(&source, &out, &["label".into()], true, true).expect("dump characters");
        assert_eq!(fs::read(out.join("0000.chars")).unwrap(), b"abcd");
        let metadata = read_metadata(&out);
        assert_eq!(record(&metadata, "label")["dtype"], "|S1");

        let missing = dump(&source, &out, &["absent".into()], true, true)
            .expect_err("unknown variables are refused");
        assert!(missing.contains("variable not found"), "{missing}");
        assert!(missing.contains("absent"), "{missing}");
        let _ = fs::remove_dir_all(&dir);
    }

    /// `--window` cuts the last two dimensions of a (time, category, y, x)
    /// variable to exactly the rectangle asked for, keeps every leading
    /// plane, echoes the window, and refuses a rectangle that leaves the
    /// plane or a variable with one dimension.
    #[test]
    fn a_window_cuts_the_last_two_dimensions_exactly() {
        use netcdf_writer::{NcFormat, NcType, NcWriter, Schema, VarData};
        let dir = scratch("dump-window");
        let source = dir.join("window.nc");
        let mut schema = Schema::new(NcFormat::Classic);
        let t = schema.def_dim("Time", 1, false).expect("dim");
        let c = schema.def_dim("cat", 2, false).expect("dim");
        let y = schema.def_dim("south_north", 3, false).expect("dim");
        let x = schema.def_dim("west_east", 4, false).expect("dim");
        let field = schema.def_var("FIELD", NcType::Float, &[t, c, y, x]).expect("var");
        let line = schema.def_var("LINE", NcType::Float, &[x]).expect("var");
        let mut writer = NcWriter::create(&source, schema).expect("create");
        // value = 100 * cat + 10 * y + x
        let values: Vec<f32> = (0..2)
            .flat_map(|c| (0..3).flat_map(move |y| (0..4).map(move |x| (100 * c + 10 * y + x) as f32)))
            .collect();
        writer.write_var(field, VarData::F32(&values)).expect("write");
        writer.write_var(line, VarData::F32(&[0.0, 1.0, 2.0, 3.0])).expect("write");
        writer.finish().expect("finish");

        let out = dir.join("out");
        let window = Window::parse("1:2,1:2").expect("parse");
        dump_windowed(&source, &out, &["FIELD".into()], true, true, [1.0, 0.0], None,
                      Some(window)).expect("windowed dump");
        let metadata = read_metadata(&out);
        let rec = record(&metadata, "FIELD");
        assert_eq!(rec["shape"], serde_json::json!([1, 2, 2, 2]));
        assert_eq!(rec["window"], serde_json::json!([1, 2, 1, 2]));
        assert_eq!(read_f64_plane(&out.join("0000.f64")),
                   vec![11.0, 12.0, 21.0, 22.0, 111.0, 112.0, 121.0, 122.0]);
        let exact_bytes = fs::read(out.join("0000.f64")).unwrap();
        for method in ["nearest", "bilinear"] {
            let sample = SampleWindow::parse("1:2,1:2", method).unwrap();
            dump_sampled(&source, &out, &["FIELD".into()], true, true,
                         [1.0, 0.0], None, None, Some(sample)).unwrap();
            assert_eq!(fs::read(out.join("0000.f64")).unwrap(), exact_bytes);
            let sampled_metadata = read_metadata(&out);
            assert_eq!(record(&sampled_metadata, "FIELD")["sample_window"]["method"], method);
        }
        dump_sampled(&source, &out, &["FIELD".into()], true, true,
                     [1.0, 0.0], None, None,
                     Some(SampleWindow::parse("0.25:2,0.5:2", "bilinear").unwrap())).unwrap();
        assert_eq!(read_f64_plane(&out.join("0000.f64")),
                   vec![5.25, 6.25, 15.25, 16.25, 105.25, 106.25, 115.25, 116.25]);
        dump_sampled(&source, &out, &["FIELD".into()], true, true,
                     [1.0, 0.0], None, None,
                     Some(SampleWindow::parse("0.25:2,0.5:2", "nearest").unwrap())).unwrap();
        assert_eq!(read_f64_plane(&out.join("0000.f64")),
                   vec![10.0, 11.0, 20.0, 21.0, 110.0, 111.0, 120.0, 121.0]);

        let outside = dump_windowed(&source, &out, &["FIELD".into()], true, true, [1.0, 0.0],
                                    None, Some(Window::parse("3:2,0:1").unwrap()))
            .expect_err("a window past the edge is refused");
        assert!(outside.contains("leave the variable"), "{outside}");
        let flat = dump_windowed(&source, &out, &["LINE".into()], true, true, [1.0, 0.0],
                                 None, Some(window))
            .expect_err("a one-dimensional variable has no window");
        assert!(flat.contains("at least two dimensions"), "{flat}");
        assert!(Window::parse("1:0,0:1").is_err());
        assert!(Window::parse("1-2,0:1").is_err());
        for overflow in [format!("{}:2,0:1", usize::MAX),
                         format!("0:1,{}:2", usize::MAX)] {
            let error = Window::parse(&overflow).expect("whole numbers")
                .apply("FIELD", &[0.0; 12], &[3, 4])
                .expect_err("an overflowing window is refused before slicing");
            assert!(error.contains("overflows the index range"), "{error}");
        }
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn masked_bilinear_averages_only_the_output_points_own_class() {
        // 2 x 3 plane; the right column is water (mask 0) holding a 0 fill.
        let values = vec![280.0, 290.0, 0.0,
                          300.0, 310.0, 0.0];
        let mask_values = vec![1.0, 1.0, 0.0,
                               1.0, 1.0, 0.0];
        let mask = SampleMask { name: "LANDMASK", values: &mask_values, ny: 2, nx: 3 };
        // Interior land point: every neighbour is land, so the bits are bilinear's.
        let inner = SampleWindow::parse("0.25:1,0.5:1", "bilinear-masked").unwrap();
        let plain = SampleWindow::parse("0.25:1,0.5:1", "bilinear").unwrap();
        assert_eq!(inner.apply_masked("F", &values, &[2, 3], Some(mask)).unwrap().0[0].to_bits(),
                   plain.apply("F", &values, &[2, 3]).unwrap().0[0].to_bits());
        // A land point beside the water column: plain bilinear blends the
        // 0 fill in; the masked sample keeps only the land neighbours.
        let coast = SampleWindow::parse("1.25:1,0.5:1", "bilinear-masked").unwrap();
        let blended = SampleWindow::parse("1.25:1,0.5:1", "bilinear").unwrap()
            .apply("F", &values, &[2, 3]).unwrap().0[0];
        assert!(blended < 250.0, "{blended}");
        assert_eq!(coast.apply_masked("F", &values, &[2, 3], Some(mask)).unwrap().0[0], 300.0);
        // A water point keeps the water class's own value.
        let sea = SampleWindow::parse("1.75:1,0.5:1", "bilinear-masked").unwrap();
        assert_eq!(sea.apply_masked("F", &values, &[2, 3], Some(mask)).unwrap().0[0], 0.0);
        // Integer origins stay exact copies.
        let exact = SampleWindow::parse("1:2,0:2", "bilinear-masked").unwrap();
        assert_eq!(exact.apply_masked("F", &values, &[2, 3], Some(mask)).unwrap().0,
                   vec![290.0, 0.0, 310.0, 0.0]);
        // The method and the mask go together; a mask of another plane is refused.
        assert!(coast.apply("F", &values, &[2, 3]).unwrap_err().contains("go together"));
        let small = SampleMask { name: "LANDMASK", values: &mask_values[..4], ny: 2, nx: 2 };
        assert!(coast.apply_masked("F", &values, &[2, 3], Some(small)).unwrap_err().contains("plane"));
    }

    #[test]
    fn sampled_windows_refuse_missing_neighbors_outside_and_overflow() {
        let sample = SampleWindow::parse("0.25:2,0.5:1", "bilinear").unwrap();
        let values = vec![1.0, 2.0, 3.0, 4.0, f64::NAN, 6.0];
        assert!(sample.apply("FIELD", &values, &[2, 3]).unwrap_err().contains("non-finite source"));
        let outside = SampleWindow::parse("2.25:1,0:1", "nearest").unwrap();
        assert!(outside.apply("FIELD", &[0.0; 6], &[2, 3]).unwrap_err().contains("leave the variable"));
        assert!(SampleWindow::parse("NaN:1,0:1", "nearest").is_err());
        assert!(SampleWindow::parse("-0.1:1,0:1", "bilinear").is_err());
        assert!(SampleWindow::parse("0:0,0:1", "bilinear").is_err());
        assert!(SampleWindow::parse("0:1,0:1", "cubic").is_err());
        let overflow = SampleWindow::parse(&format!("0:{},0:1", usize::MAX), "bilinear").unwrap();
        assert!(overflow.apply("FIELD", &[0.0; 6], &[2, 3]).unwrap_err().contains("overflow"));
        let one = SampleWindow::parse("2:1,1:1", "bilinear").unwrap();
        let mut edge = vec![0.0; 6];
        edge[5] = -0.0;
        assert_eq!(one.apply("FIELD", &edge, &[2, 3]).unwrap().0[0].to_bits(), (-0.0_f64).to_bits());
    }

    /// A NetCDF-4 coordinate variable is an HDF5 dimension scale; the
    /// reader lists it, so its dump is an ordinary one: values, units
    /// and the decoded times, over its OWN dimension rather than an
    /// empty or invented list.
    #[test]
    fn a_coordinate_variable_dumps_over_its_own_dimension() {
        let dir = scratch("dump-coordinate");
        let out = dir.join("out");
        dump(&fixture("axes.nc4"), &out, &["time".into()], true, true)
            .expect("dump coordinate variable");

        assert_eq!(read_f64_plane(&out.join("0000.f64")), &[0.0, 1.0, 2.0, 3.0]);
        let metadata = read_metadata(&out);
        let time = record(&metadata, "time");
        assert_eq!(time["units"], "hours since 2024-03-01");
        assert_eq!(
            time["times"],
            serde_json::json!([
                "2024-03-01T00:00:00Z",
                "2024-03-01T01:00:00Z",
                "2024-03-01T02:00:00Z",
                "2024-03-01T03:00:00Z",
            ])
        );
        assert_eq!(time["dimensions"], serde_json::json!(["time"]));
        assert_eq!(time["shape"], serde_json::json!([4]));
        let _ = fs::remove_dir_all(&dir);
    }

    /// A multi-axis coordinate variable (a 2-D dimension scale) dumps
    /// every element over every axis.  Describing it over its first
    /// dimension alone, as a reader that treats a scale as 1-D does,
    /// made this very read fail with "slice has 1 dimensions but
    /// dataset has 2".
    #[test]
    fn a_two_dimensional_coordinate_dumps_every_element() {
        let dir = scratch("dump-coordinate-2d");
        let out = dir.join("out");
        dump(&fixture("axes.nc4"), &out, &["corners".into()], true, true)
            .expect("dump 2-D coordinate variable");
        let values = read_f64_plane(&out.join("0000.f64"));
        assert_eq!(values, (0..30).map(f64::from).collect::<Vec<_>>());
        let metadata = read_metadata(&out);
        let corners = record(&metadata, "corners");
        assert_eq!(corners["shape"], serde_json::json!([5, 6]));
        assert_eq!(corners["dimensions"], serde_json::json!(["corners", "zvals"]));
        let _ = fs::remove_dir_all(&dir);
    }

    /// An HDF5 dataset outside the variable table has no entry to
    /// describe it, so its dump takes the raw-HDF5 path end to end:
    /// existence via `has_hdf5_dataset`, values by name, and an empty
    /// dimension list rather than an invented one.  `bnds` is such a
    /// dataset: the scale netCDF-C writes for a dimension with no
    /// coordinate variable, hidden from the table by its NAME marker
    /// and never written to, so it reads as the library's zero fill.
    #[test]
    fn a_hidden_hdf5_dataset_dumps_through_the_raw_hdf5_path() {
        let dir = scratch("dump-raw-hdf5");
        let out = dir.join("out");
        let (file, _) = open(&fixture("axes.nc4")).expect("open");
        assert!(file.variable("bnds").is_none(), "bnds is a dimension-only scale");
        assert!(file.has_hdf5_dataset("bnds"));
        dump(&fixture("axes.nc4"), &out, &["bnds".into()], true, true)
            .expect("dump hidden dataset");
        assert_eq!(read_f64_plane(&out.join("0000.f64")), &[0.0, 0.0]);
        let metadata = read_metadata(&out);
        let bnds = record(&metadata, "bnds");
        assert_eq!(bnds["shape"], serde_json::json!([2]));
        assert_eq!(bnds["dimensions"], serde_json::json!([]));
        assert_eq!(bnds["units"], serde_json::Value::Null);

        let missing = dump(&fixture("axes.nc4"), &out, &["absent".into()], true, true)
            .expect_err("unknown names are refused on HDF5 files too");
        assert!(missing.contains("variable not found"), "{missing}");
        let _ = fs::remove_dir_all(&dir);
    }

    // ------------------------------------------------------------------
    // Metadata provenance: strict against size-inferred.

    #[test]
    fn a_classic_file_opens_strict_with_nothing_to_confess() {
        let dir = scratch("open-strict");
        let source = dir.join("packed.nc");
        write_packed_classic(&source);
        let (_, provenance) = open(&source).expect("open");
        assert_eq!(provenance.mode, "strict");
        assert_eq!(provenance.strict_error, None);
        assert!(!provenance.dimension_lengths_ambiguous);
        let _ = fs::remove_dir_all(&dir);
    }

    /// sizes.nc4 is a bare HDF5 file with no DIMENSION_LIST anywhere:
    /// strict reconstruction must fail, the size-inferred fallback must
    /// open it, and the provenance must carry the strict error so a
    /// consumer can refuse the guess.
    #[test]
    fn a_bare_hdf5_file_falls_back_to_size_inferred_metadata() {
        let (file, provenance) = open(&fixture("sizes.nc4")).expect("open");
        assert_eq!(provenance.mode, "size-inferred");
        let error = provenance.strict_error.expect("strict error is reported");
        assert!(error.contains("DIMENSION_LIST"), "{error}");
        let names: Vec<String> = file
            .variables()
            .expect("variables")
            .into_iter()
            .map(|variable| variable.name().to_string())
            .collect();
        assert!(names.contains(&"first".to_string()), "{names:?}");
        assert!(names.contains(&"second".to_string()), "{names:?}");
    }

    /// mixed.nc4 keeps its real dimension table (row and col, both 3)
    /// into the fallback, so the length collision that can mis-name an
    /// axis under size inference MUST be confessed.
    #[test]
    fn equal_dimension_lengths_are_confessed_as_ambiguous() {
        let (_, provenance) = open(&fixture("mixed.nc4")).expect("open");
        assert_eq!(provenance.mode, "size-inferred");
        assert!(provenance.strict_error.is_some());
        assert!(
            provenance.dimension_lengths_ambiguous,
            "row and col share a length; size inference cannot tell them apart"
        );
    }

    #[test]
    fn duplicate_dimension_lengths_are_detected_and_absence_is_clean() {
        use netcdf_writer::{NcFormat, NcType, NcWriter, Schema, VarData};
        let dir = scratch("dup-dims");

        let twins = dir.join("twins.nc");
        let mut schema = Schema::new(NcFormat::Classic);
        let a = schema.def_dim("a", 3, false).expect("dim");
        let b = schema.def_dim("b", 3, false).expect("dim");
        let va = schema.def_var("va", NcType::Double, &[a]).expect("var");
        let vb = schema.def_var("vb", NcType::Double, &[b]).expect("var");
        let mut writer = NcWriter::create(&twins, schema).expect("create");
        writer.write_var(va, VarData::F64(&[1.0, 2.0, 3.0])).expect("write");
        writer.write_var(vb, VarData::F64(&[4.0, 5.0, 6.0])).expect("write");
        writer.finish().expect("finish");
        let file = netcrust::File::open(&twins).expect("open");
        assert!(has_duplicate_dimension_lengths(&file));

        let distinct = dir.join("distinct.nc");
        let mut schema = Schema::new(NcFormat::Classic);
        let a = schema.def_dim("a", 3, false).expect("dim");
        let b = schema.def_dim("b", 4, false).expect("dim");
        let va = schema.def_var("va", NcType::Double, &[a]).expect("var");
        let vb = schema.def_var("vb", NcType::Double, &[b]).expect("var");
        let mut writer = NcWriter::create(&distinct, schema).expect("create");
        writer.write_var(va, VarData::F64(&[1.0, 2.0, 3.0])).expect("write");
        writer
            .write_var(vb, VarData::F64(&[4.0, 5.0, 6.0, 7.0]))
            .expect("write");
        writer.finish().expect("finish");
        let file = netcrust::File::open(&distinct).expect("open");
        assert!(!has_duplicate_dimension_lengths(&file));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn attributes_map_carries_every_attribute_by_its_own_name() {
        let dir = scratch("attrs-map");
        let source = dir.join("packed.nc");
        write_packed_classic(&source);
        let file = netcrust::File::open(&source).expect("open");
        let map = attributes_map(&file.attributes().expect("attributes"));
        assert_eq!(map.len(), 2, "{map:?}");
        assert_eq!(map["title"], serde_json::json!("packed fixture"));
        assert_eq!(map["version"], serde_json::json!(3.0));
        let _ = fs::remove_dir_all(&dir);
    }

    // ------------------------------------------------------------------
    // The variable table against the checked-in NetCDF-4 fixtures:
    // coordinate variables (HDF5 dimension scales) come from the reader
    // like any field, and the recovery pass adds only what it hides.

    /// The table exactly as `inventory` builds it, through the same
    /// code path, for a fixture that must open strict.
    fn strict_table(name: &str) -> (Vec<DimensionRecord>, Vec<VariableRecord>) {
        let path = fixture(name);
        let (file, provenance) = open(&path).expect("open");
        assert_eq!(provenance.mode, "strict", "{:?}", provenance.strict_error);
        let dimensions = dimension_table(&file, &path).expect("dimensions");
        let variables = variable_table(&file, &path, &dimensions).expect("variables");
        (dimensions, variables)
    }

    fn axes_table() -> Vec<VariableRecord> {
        strict_table("axes.nc4").1
    }

    #[test]
    fn coordinate_variables_are_listed_with_their_own_dimensions() {
        let variables = axes_table();
        let mut names: Vec<&str> = variables.iter().map(|v| v.name.as_str()).collect();
        names.sort_unstable();
        assert_eq!(names, vec!["corners", "lat", "lon", "t2", "time", "zvals"]);
        assert!(
            variables.iter().all(|v| !v.recovered_dimension_scale),
            "the reader lists every coordinate variable, so nothing is \
             left for the raw-HDF5 recovery"
        );

        // lon and lat share length 3, so only the scale's own identity
        // (the reader resolves a scale by its dataset address, never by
        // length) can attach each to its OWN dimension rather than the
        // first one of matching length.
        for name in ["time", "lon", "lat", "zvals"] {
            let record = variables
                .iter()
                .find(|v| v.name == name)
                .unwrap_or_else(|| panic!("{name} not listed"));
            assert_eq!(
                record.dimensions,
                vec![name.to_string()],
                "{name} must resolve to its own dimension"
            );
            assert_eq!(record.dtype, "F64", "{name} carries its stored type");
        }

        let time = variables.iter().find(|v| v.name == "time").expect("time");
        assert_eq!(time.shape, vec![4]);
        assert_eq!(
            time.attributes,
            BTreeMap::from([
                ("units".to_string(), serde_json::json!("hours since 2024-03-01")),
                ("standard_name".to_string(), serde_json::json!("time")),
                ("calendar".to_string(), serde_json::json!("standard")),
                ("axis".to_string(), serde_json::json!("T")),
            ]),
            "every attribute the fixture wrote, none it did not, and \
             no HDF5 bookkeeping (CLASS, NAME, REFERENCE_LIST)"
        );
        let zvals = variables.iter().find(|v| v.name == "zvals").expect("zvals");
        assert_eq!(zvals.attributes.get("units"), Some(&serde_json::json!("m")));
    }

    /// The unique-length rule for a scale from a producer that writes no
    /// coordinate record: `corners` (5 x 6, made with h5py) is its own
    /// first dimension, and its second axis resolves only because exactly
    /// one dimension has length 6.
    #[test]
    fn a_two_dimensional_scale_resolves_axes_by_unique_length() {
        let variables = axes_table();
        let corners = variables
            .iter()
            .find(|v| v.name == "corners")
            .expect("corners listed");
        assert_eq!(corners.shape, vec![5, 6]);
        assert_eq!(corners.dimensions, vec!["corners", "zvals"]);
        assert_eq!(corners.dtype, "F64");
        assert_eq!(
            corners.attributes.get("long_name"),
            Some(&serde_json::json!("cell corner offsets"))
        );
    }

    /// netCDF-C's own multi-axis coordinate variable: `station(station,
    /// pair)` where `pair` and `other` both have length 2, so no length
    /// rule can name the second axis; only the library's
    /// `_Netcdf4Coordinates` record can, and the reader reads it.
    #[test]
    fn a_netcdf_c_multi_axis_coordinate_resolves_by_its_dimension_ids() {
        let (dimensions, variables) = strict_table("coord2d.nc4");
        let twos: Vec<&str> = dimensions
            .iter()
            .filter(|d| d.len == 2)
            .map(|d| d.name.as_str())
            .collect();
        assert_eq!(twos, vec!["pair", "other"], "the fixture's ambiguity must stand");
        let station = variables
            .iter()
            .find(|v| v.name == "station")
            .expect("station listed");
        assert_eq!(station.shape, vec![3, 2]);
        assert_eq!(station.dimensions, vec!["station", "pair"]);
        assert_eq!(
            station.attributes.get("long_name"),
            Some(&serde_json::json!("station corner"))
        );
        let temp = variables.iter().find(|v| v.name == "temp").expect("temp listed");
        assert_eq!(temp.dimensions, vec!["station"]);
        assert!(variables.iter().all(|v| !v.recovered_dimension_scale));
    }

    #[test]
    fn phony_dimensions_are_not_listed_and_known_variables_are_not_doubled() {
        let (dimensions, variables) = strict_table("axes.nc4");
        assert!(dimensions.iter().any(|d| d.name == "bnds" && d.len == 2));
        assert!(
            !variables.iter().any(|v| v.name == "bnds"),
            "bnds has no coordinate variable; listing it would invent \
             a variable netCDF4-python does not report"
        );
        for name in ["t2", "time", "corners"] {
            assert_eq!(
                variables.iter().filter(|v| v.name == name).count(),
                1,
                "{name} is in the variable table once and must not be doubled"
            );
        }
    }

    /// The recovery pass adds only what the reader hides, and nothing
    /// twice: run over the complete table it is a no-op, and run over a
    /// table missing `time` it brings `time` back over its own dimension,
    /// with the Promoted type and the by-name CF string attributes the
    /// raw path can reach, flagged as recovered.
    #[test]
    fn recovery_adds_only_hidden_scales_and_never_doubles_a_listed_one() {
        let path = fixture("axes.nc4");
        let (file, _) = open(&path).expect("open");
        let dimensions = dimension_table(&file, &path).expect("dimensions");
        let mut complete = variable_table(&file, &path, &dimensions).expect("variables");
        let before = complete.len();
        recover_dimension_scales(&file, &dimensions, &mut complete);
        assert_eq!(complete.len(), before, "a listed scale is never recovered again");

        let mut hidden: Vec<VariableRecord> =
            complete.into_iter().filter(|v| v.name != "time").collect();
        recover_dimension_scales(&file, &dimensions, &mut hidden);
        let time = hidden.iter().find(|v| v.name == "time").expect("time recovered");
        assert!(time.recovered_dimension_scale);
        assert_eq!(time.dimensions, vec!["time"]);
        assert_eq!(time.shape, vec![4]);
        assert_eq!(time.dtype, "Promoted");
        assert_eq!(
            time.attributes.get("units"),
            Some(&serde_json::json!("hours since 2024-03-01"))
        );
        assert_eq!(time.attributes.get("calendar"), Some(&serde_json::json!("standard")));
        assert_eq!(time.attributes.get("axis"), Some(&serde_json::json!("T")));
        assert!(!hidden.iter().any(|v| v.name == "bnds"), "a phony dimension stays hidden");
    }
}
