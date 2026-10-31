//! Ordinary WRF/rw-store inputs; no cycling-layout or assimilation assumptions.
use crate::store::{RawColumn, Res, Sfc, Store};
use rw_store::{
    grid::{GridFile, GridLocator},
    reader::HourReader,
};
use nc4_reader::NcFile as Nc4File;
use netcdf_reader::{NcFile, NcSliceInfo, NcSliceInfoElem};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{path::PathBuf, sync::Arc};
use wrf_core::{pure_reader::PureRustFile, WrfFile};

#[derive(Clone, Deserialize)]
pub struct Source {
    pub path: PathBuf,
    pub domain: String,
    pub member: String,
}
#[derive(Clone, Deserialize)]
pub struct Point {
    pub label: String,
    pub lat: f64,
    pub lon: f64,
}
pub struct Frame {
    pub domain: String,
    pub member: String,
    pub valid: i64,
    pub columns: Vec<RawColumn>,
    /// What the run says about itself (its start, model, shape, init source), for the sounding's top-right label.
    pub run: RunInfo,
}

/// A run's own description of itself, read from its files: the start (ISO, 2024-04-26T18:00:00Z), the model words
/// ("Regional", "Hex"), the shape ("Global 120 km") and the init source ("HRRR"). Each part is only what the file
/// states; a missing part stays missing. Column stores carry it in their header (`run`), JSON column frames as
/// top-level keys, wrfouts as their globals.
#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
pub struct RunInfo {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub init: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shape: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
}
impl RunInfo {
    /// Every part `self` lacks, taken from `other`.
    pub fn or(mut self, other: &RunInfo) -> RunInfo {
        let take = |a: &mut Option<String>, b: &Option<String>| {
            if a.as_deref().is_none_or(|s| s.trim().is_empty()) {
                *a = b.clone().filter(|s| !s.trim().is_empty());
            }
        };
        take(&mut self.init, &other.init);
        take(&mut self.model, &other.model);
        take(&mut self.shape, &other.shape);
        take(&mut self.source, &other.source);
        self
    }
    /// What a wrfout's globals say: its start (SIMULATION_START_DATE, else START_DATE), the init source an ArWen
    /// run records (GPUWM_INITIAL_CONDITION_SOURCE, else the prepared route's TITLE), and the model: "Hex" for a hex export (its GPUWM_MODEL_LABEL
    /// names hex, or its TITLE names the MPAS port), "Regional" for an ArWen wrfout (GPUWM_VERSION).
    pub fn from_wrf(f: &WrfFile) -> RunInfo {
        let attr = |k: &str| f.global_attr_str(k).ok().map(|s| s.trim().trim_matches('\0').to_string()).filter(|s| !s.is_empty());
        let init = ["SIMULATION_START_DATE", "START_DATE"]
            .iter()
            .find_map(|k| attr(k).and_then(|s| timestamp(&s).ok()))
            .map(iso);
        let label = attr("GPUWM_MODEL_LABEL").unwrap_or_default();
        let title = attr("TITLE").unwrap_or_default();
        // The prepared-domain route titles its files "gpuwm prepared HRRR domain tree ..." and writes no
        // GPUWM_INITIAL_CONDITION_SOURCE, so its source is the title's word after "prepared".
        let source = attr("GPUWM_INITIAL_CONDITION_SOURCE")
            .filter(|s| !s.eq_ignore_ascii_case("unknown"))
            .or_else(|| title.strip_prefix("gpuwm prepared ").and_then(|t| t.split_whitespace().next()).map(str::to_string));
        let model = if label.starts_with("WOOF") {
            Some(label.trim_start_matches("WOOF").trim().to_string()).filter(|s| !s.is_empty())
        } else if label.to_ascii_lowercase().contains("hex") || title.contains("MPAS") {
            Some("Hex".to_string())
        } else if attr("GPUWM_VERSION").is_some() || title.contains("GPUWM") {
            Some("Regional".to_string())
        } else {
            None
        };
        RunInfo { init, model, shape: None, source }
    }
}
/// Unix seconds as ISO UTC (2024-04-26T18:00:00Z).
pub fn iso(t: i64) -> String {
    chrono::DateTime::from_timestamp(t, 0).unwrap().format("%Y-%m-%dT%H:%M:%SZ").to_string()
}
fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}
/// A WRF time (2024-07-14_16:00:00) or ISO time (2024-07-14T16:00:00Z) as Unix seconds.
pub fn parse_time(s: &str) -> Res<i64> {
    timestamp(s)
}
fn timestamp(s: &str) -> Res<i64> {
    let s = s.replace('_', "T");
    chrono::NaiveDateTime::parse_from_str(s.trim_end_matches('Z'), "%Y-%m-%dT%H:%M:%S")
        .map(|t| t.and_utc().timestamp())
        .map_err(err)
}
fn locate(g: &GridFile, point: &Point) -> Res<(f64, f64)> {
    GridLocator::build(g)
        .locate(point.lat, point.lon)
        .filter(|(x, y)| {
            *x >= 0.0 && *y >= 0.0 && *x <= (g.nx - 1) as f64 && *y <= (g.ny - 1) as f64
        })
        .ok_or_else(|| {
            format!(
                "{} ({},{}) is outside the model grid",
                point.label, point.lat, point.lon
            )
        })
}
/// Bolton vapour-pressure inversion: q is water-vapour mixing ratio (kg/kg).
pub(crate) fn dewpoint(p_pa: f64, q: f64) -> f64 {
    let e = (p_pa * q.max(1e-12) / (0.622 + q.max(1e-12)) / 100.0).max(1e-12);
    let l = (e / 6.112).ln();
    243.5 * l / (17.67 - l)
}
pub fn read(source: &Source, points: &[Point]) -> Res<Vec<Frame>> {
    if source.path.extension().is_some_and(|x| x == "rws") {
        let r = HourReader::open(&source.path).map_err(err)?;
        // Whole-hour stores encode their UTC origin in the run key.
        let origin = {
            let run = &r.meta().run;
            let stamp = run.strip_prefix("local_").and_then(|s| s.split('_').next()).unwrap_or(run);
            ["%Y%m%d%H%M%S", "%Y%m%d%H%M", "%Y%m%d%H"].iter().find_map(|fmt| {
                chrono::NaiveDateTime::parse_from_str(stamp, fmt).ok().map(|t|t.and_utc().timestamp())
            })
        };
        let valid = r.meta().exact_time().map(|t| t.valid_unix).or_else(|| {
            origin.map(|t| t + i64::from(r.meta().forecast_hour)*3600)
        }).ok_or("rw-store input needs exact time metadata or a UTC run origin; cannot infer valid time from a storage slot")?;
        let grid = GridFile::open(&source.path.parent().unwrap().join("grid.rwg")).map_err(err)?;
        let columns = points
            .iter()
            .map(|p| {
                let (x, y) = locate(&grid, p)?;
                Store::point_column(&r, x, y)
            })
            .collect::<Res<_>>()?;
        return Ok(vec![Frame {
            domain: source.domain.clone(),
            member: source.member.clone(),
            valid,
            columns,
            run: RunInfo { init: origin.map(iso), ..Default::default() },
        }]);
    }
    if source.path.extension().is_some_and(|x| x == "json") {
        return read_json_frame(source, points);
    }
    if source.path.extension().is_some_and(|x| x == "wcol") {
        return read_column_store(source, points);
    }
    let f = WrfFile::open(&source.path).map_err(err)?;
    let run = RunInfo::from_wrf(&f);
    // Classic, 64-bit-offset and CDF-5 files (ArWen wrfouts, hex exports) are memory-mapped, so a
    // hyperslab read touches only the profile columns. NetCDF-4 (compressed) frames read whole slices,
    // every chunk inflated once and in parallel ([`nc4_time_slice`]).
    let (classic, nc4) = match f.hdf5() {
        PureRustFile::Classic(_) => (Some(NcFile::open(&source.path).map_err(err)?), None),
        PureRustFile::Hdf5(_) => (None, Some(Nc4File::open(&source.path).map_err(err)?)),
    };
    let mut frames = vec![];
    for (ti, time) in f.times().map_err(err)?.iter().enumerate() {
        let g = GridFile {
            nx: f.nx,
            ny: f.ny,
            lat: f.xlat(ti).map_err(err)?.iter().map(|x| *x as f32).collect(),
            lon: f
                .xlong(ti)
                .map_err(err)?
                .iter()
                .map(|x| *x as f32)
                .collect(),
            projection: None,
            hash: String::new(),
        };
        f.clear_cache();
        let xy = points
            .iter()
            .map(|p| locate(&g, p))
            .collect::<Res<Vec<_>>>()?;
        let columns = match (&classic, &nc4) {
            (Some(nc), _) => {
                let read = |name: &str, levels, rows, cols| hyperslab(nc, name, ti, levels, rows, cols);
                xy.par_iter()
                    .map(|&(x, y)| column(&f, &Stencil::new(&f, x, y), &read))
                    .collect::<Res<Vec<_>>>()?
            }
            (None, nc4) => {
                let nc4 = nc4.as_ref().ok_or("a NetCDF-4 frame without its reader")?;
                let whole = WholeSlices::read(&f, nc4, ti)?;
                let read = |name: &str, _, _, _| whole.get(name);
                xy.par_iter()
                    .map(|&(x, y)| column(&f, &Stencil::new(&f, x, y), &read))
                    .collect::<Res<Vec<_>>>()?
            }
        };
        frames.push(Frame {
            domain: source.domain.clone(),
            member: source.member.clone(),
            valid: timestamp(time)?,
            columns,
            run: run.clone(),
        });
    }
    Ok(frames)
}

// wrf-core's own constants for its whole-field diagnostics.
const P0: f64 = 100_000.0; // Pa
const KAPPA: f64 = 0.2857142857; // Rd / Cp
const G: f64 = 9.80665;
const PROFILE_VARS: [&str; 17] = [
    "P", "PB", "T", "QVAPOR", "PH", "PHB", "W", "U", "V", "PSFC", "HGT", "T2", "Q2", "U10", "V10",
    "COSALPHA", "SINALPHA",
];

/// One variable over a box of (level, row, column), row-major like the file.
struct Window {
    data: Vec<f64>,
    y0: usize,
    x0: usize,
    ny: usize,
    nx: usize,
}
impl Window {
    fn at(&self, k: usize, j: usize, i: usize) -> f64 {
        self.data[(k * self.ny + j - self.y0) * self.nx + i - self.x0]
    }
}

/// The bilinear corners of one point: columns ix..=jx, rows iy..=jy.
struct Stencil {
    ix: usize,
    iy: usize,
    jx: usize,
    jy: usize,
    ax: f64,
    ay: f64,
}
impl Stencil {
    fn new(f: &WrfFile, x: f64, y: f64) -> Self {
        let ix = x.floor() as usize;
        let iy = y.floor() as usize;
        Stencil {
            ix,
            iy,
            jx: (ix + 1).min(f.nx - 1),
            jy: (iy + 1).min(f.ny - 1),
            ax: x - ix as f64,
            ay: y - iy as f64,
        }
    }
    /// The same expression, in the same order, as the whole-field sampler it replaced.
    fn sample(&self, g: impl Fn(usize, usize) -> f64) -> f64 {
        let (ax, ay) = (self.ax, self.ay);
        (g(self.ix, self.iy) * (1.0 - ax) + g(self.jx, self.iy) * ax) * (1.0 - ay)
            + (g(self.ix, self.jy) * (1.0 - ax) + g(self.jx, self.jy) * ax) * ay
    }
}

/// `read(name, levels, rows, cols)` gives a window covering at least levels 0..levels and the
/// half-open row and column ranges.
type Reader<'a> = dyn Fn(&str, usize, (usize, usize), (usize, usize)) -> Res<Arc<Window>> + Sync + 'a;

/// The exact selection wrf-core's classic reader makes (time index, then every element through
/// `read_variable_slice_as_f64`), narrowed to the stencil box, so each value converts identically.
fn hyperslab(
    nc: &NcFile,
    name: &str,
    ti: usize,
    levels: usize,
    (y0, y1): (usize, usize),
    (x0, x1): (usize, usize),
) -> Res<Arc<Window>> {
    use NcSliceInfoElem::{Index, Slice};
    let var = nc.variable(name).map_err(err)?;
    let has_time = var
        .dimensions()
        .first()
        .is_some_and(|d| d.is_unlimited || d.name == "Time");
    let mut selections = vec![];
    if has_time {
        selections.push(Index(ti as u64));
    }
    if var.shape().len() - usize::from(has_time) == 3 {
        selections.push(Slice { start: 0, end: levels as u64, step: 1 });
    }
    selections.push(Slice { start: y0 as u64, end: y1 as u64, step: 1 });
    selections.push(Slice { start: x0 as u64, end: x1 as u64, step: 1 });
    let data = nc
        .read_variable_slice_as_f64(name, &NcSliceInfo { selections })
        .map_err(err)?;
    Ok(Arc::new(Window {
        data: data.iter().copied().collect(),
        y0,
        x0,
        ny: y1 - y0,
        nx: x1 - x0,
    }))
}

/// NetCDF-4 inputs: each profile variable's whole time slice, the variables and their chunks read concurrently.
struct WholeSlices(std::collections::HashMap<&'static str, Arc<Window>>);
impl WholeSlices {
    fn read(f: &WrfFile, nc: &Nc4File, ti: usize) -> Res<Self> {
        let windows = PROFILE_VARS
            .par_iter()
            .map(|&name| {
                let shape = f.var_shape_no_time(name).map_err(err)?;
                let (ny, nx) = (shape[shape.len() - 2], shape[shape.len() - 1]);
                let data = nc4_time_slice(nc, name, ti)?;
                Ok((name, Arc::new(Window { data, y0: 0, x0: 0, ny, nx })))
            })
            .collect::<Res<_>>()?;
        Ok(WholeSlices(windows))
    }
    fn get(&self, name: &str) -> Res<Arc<Window>> {
        self.0
            .get(name)
            .cloned()
            .ok_or_else(|| format!("{name} was not read"))
    }
}

/// One raw column from its stencil cells only. Each derived value is wrf-core's whole-field
/// expression evaluated at that cell (P+PB; (T+300)*(p/P0)^kappa - 273.15; destaggered PH+PHB
/// over g; destaggered U, V and W), so every sample is bitwise what the whole-field read gave.
fn column(f: &WrfFile, s: &Stencil, read: &Reader) -> Res<RawColumn> {
    let (nz, nzs) = (f.nz, f.nz_stag);
    let rows = (s.iy, s.jy + 1);
    let cols = (s.ix, s.jx + 1);
    let mut w = std::collections::HashMap::new();
    for name in PROFILE_VARS {
        let window = match name {
            "U" => read(name, nz, rows, (s.ix, s.jx + 2))?,
            "V" => read(name, nz, (s.iy, s.jy + 2), cols)?,
            "PH" | "PHB" | "W" => read(name, nzs, rows, cols)?,
            "P" | "PB" | "T" | "QVAPOR" => read(name, nz, rows, cols)?,
            _ => read(name, 1, rows, cols)?,
        };
        w.insert(name, window);
    }
    let v = |name: &str, k: usize, j: usize, i: usize| w[name].at(k, j, i);
    let pressure = |k, j, i| v("P", k, j, i) + v("PB", k, j, i);
    let temp_c = |k, j, i| {
        let tk = (v("T", k, j, i) + 300.0) * (pressure(k, j, i) / P0).powf(KAPPA);
        tk - 273.15
    };
    let geopt = |k, j, i| v("PH", k, j, i) + v("PHB", k, j, i);
    let height = |k, j, i| 0.5 * (geopt(k, j, i) + geopt(k + 1, j, i)) / G;
    let u_grid = |k, j, i| 0.5 * (v("U", k, j, i) + v("U", k, j, i + 1));
    let v_grid = |k, j, i| 0.5 * (v("V", k, j, i) + v("V", k, j + 1, i));
    let w_mass = |k, j, i| 0.5 * (v("W", k, j, i) + v("W", k + 1, j, i));
    // Rotation must precede horizontal interpolation. WRF stores grid-relative winds.
    let ca = |j, i| v("COSALPHA", 0, j, i);
    let sa = |j, i| v("SINALPHA", 0, j, i);
    let earth_u = |u: f64, vv: f64, j, i| u * ca(j, i) - vv * sa(j, i);
    let earth_v = |u: f64, vv: f64, j, i| vv * ca(j, i) + u * sa(j, i);
    // Hydrostatic conversion of WRF vertical velocity to pressure velocity (Pa/s).
    let omega = |k, j, i| {
        let (p, t, q) = (pressure(k, j, i), temp_c(k, j, i), v("QVAPOR", k, j, i));
        -9.80665 * p * w_mass(k, j, i) / (287.05 * (t + 273.15) * (1.0 + 0.61 * q))
    };
    // Derive Td on the grid before bilinear sampling, like the site store.
    let td = |k, j, i| dewpoint(pressure(k, j, i), v("QVAPOR", k, j, i));
    let sfc = |name: &str| s.sample(|i, j| v(name, 0, j, i));
    let mut c = RawColumn {
        sfc: Sfc {
            psfc_hpa: sfc("PSFC") / 100.0,
            orog_m: sfc("HGT"),
            t2_c: sfc("T2") - 273.15,
            td2_c: dewpoint(sfc("PSFC"), sfc("Q2")),
            u10: s.sample(|i, j| earth_u(v("U10", 0, j, i), v("V10", 0, j, i), j, i)),
            v10: s.sample(|i, j| earth_v(v("U10", 0, j, i), v("V10", 0, j, i), j, i)),
        },
        ..Default::default()
    };
    for k in 0..nz {
        c.p.push(s.sample(|i, j| pressure(k, j, i)) / 100.0);
        c.z.push(s.sample(|i, j| height(k, j, i)));
        c.t.push(s.sample(|i, j| temp_c(k, j, i)));
        c.td.push(s.sample(|i, j| td(k, j, i)));
        c.u.push(s.sample(|i, j| earth_u(u_grid(k, j, i), v_grid(k, j, i), j, i)));
        c.v.push(s.sample(|i, j| earth_v(u_grid(k, j, i), v_grid(k, j, i), j, i)));
        c.omega.push(s.sample(|i, j| omega(k, j, i)));
    }
    Ok(c)
}

/// One WRF-shaped history time on its whole grid, in bridge units, ready to sample: pressure in Pa, heights m MSL,
/// temperature and dewpoint C (dewpoint derived on the grid before any sampling, like the site store), winds
/// earth-relative m/s (rotated before horizontal interpolation; WRF stores grid-relative winds), omega Pa/s.
pub struct WrfGrid {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub lat: Vec<f64>,
    pub lon: Vec<f64>,
    pub p: Vec<f64>,
    pub z: Vec<f64>,
    pub t: Vec<f64>,
    pub td: Vec<f64>,
    pub u: Vec<f64>,
    pub v: Vec<f64>,
    pub omega: Vec<f64>,
    pub ps: Vec<f64>,
    pub terrain: Vec<f64>,
    pub t2: Vec<f64>,
    pub q2: Vec<f64>,
    pub u10: Vec<f64>,
    pub v10: Vec<f64>,
}

/// The raw WRF variables of one history time, read in windows of (levels, rows, columns) in the file's own layout,
/// every value promoted to f64 exactly as wrf-core's reader promotes it.
enum Raw {
    /// Classic, 64-bit-offset and CDF-5 files (ArWen wrfouts, hex exports): hyperslabs straight from the
    /// memory-mapped file, so every window is read by the task that needs it, on its own core.
    Classic { nc: NcFile, ti: usize },
    /// NetCDF-4 (compressed history frames): each variable's whole time slice, then windowed. Read with every
    /// chunk decompressed once, the variables and their chunks side by side on the pool ([`nc4_words`]).
    Whole(std::collections::HashMap<&'static str, (Words, Vec<usize>)>),
}

/// A NetCDF-4 variable's stored words: `f32` as the file holds them (promoted to f64 as each window is copied, on
/// the task that reads it), any other type already promoted.
enum Words {
    F32(Vec<f32>),
    F64(Vec<f64>),
}

/// A netcdf-reader array's words in row-major order: its own buffer when it is already that (always, for these
/// reads), else a copy. (A macro: the crate names no `ndarray` type of its own.)
macro_rules! standard_vec {
    ($a:expr) => {{
        let a = $a;
        let len = a.len();
        if a.is_standard_layout() {
            let (v, offset) = a.into_raw_vec_and_offset();
            let o = offset.unwrap_or(0);
            if o == 0 && v.len() == len { v } else { v[o..o + len].to_vec() }
        } else {
            a.iter().copied().collect::<Vec<_>>()
        }
    }};
}

/// Variable `name` at time `ti` of a NetCDF-4 file, every remaining axis whole: the selection wrf-core
/// makes (the time index when the variable has a Time axis, then everything) through netcdf-reader's chunk-parallel
/// read, which decompresses each overlapping chunk (shuffle + deflate tile) exactly once, in parallel in the
/// caller's rayon pool. Its words promote to f64 exactly as wrf-core's (`as f64`), so every value is the same.
///
/// THE BREAKAGE THIS PREVENTS (gpuwm 2.9.10 history frames are lossless NetCDF-4 by default): wrf-core's
/// `read_var` inflates a variable's chunks one after another, so the column store held ~10 of 22 cores (one per
/// big 3-D variable) and built the same bytes 3-3.5x slower than from classic frames (57 frames of a 12-3-1 km
/// run: 30.0 vs 8.6 s), and a box sounding inflated its 17 variables the same way.
fn nc4_words(nc: &Nc4File, name: &str, ti: usize) -> Res<Words> {
    use nc4_reader::NcSliceInfoElem::{Index, Slice};
    let var = nc.variable(name).map_err(|e| format!("{name}: {e}"))?;
    let shape = var.shape();
    let has_time = shape.len() >= 2 && var.dimensions().first().is_some_and(|d| d.is_unlimited || d.name == "Time");
    let selections = shape
        .iter()
        .enumerate()
        .map(|(d, &len)| if d == 0 && has_time { Index(ti as u64) } else { Slice { start: 0, end: len as u64, step: 1 } })
        .collect();
    let selection = nc4_reader::NcSliceInfo { selections };
    let at = |e: nc4_reader::Error| format!("{name}: {e}");
    if *var.dtype() == nc4_reader::NcType::Float {
        // The stored words as they are: no whole-array promotion on one thread (the reader's own `as_f64`), and
        // half the fresh memory. f64::from(f32) is the exact promotion wrf-core's `as f64` makes.
        return Ok(Words::F32(standard_vec!(nc.read_variable_slice_parallel::<f32>(name, &selection).map_err(at)?)));
    }
    Ok(Words::F64(standard_vec!(nc.read_variable_slice_as_f64_parallel(name, &selection).map_err(at)?)))
}

/// [`nc4_words`] promoted to f64 on every core (the point soundings' whole slices).
fn nc4_time_slice(nc: &Nc4File, name: &str, ti: usize) -> Res<Vec<f64>> {
    Ok(match nc4_words(nc, name, ti)? {
        Words::F32(w) => w.par_iter().map(|&v| f64::from(v)).collect(),
        Words::F64(w) => w,
    })
}

/// The 3-D and 2-D variables the grid needs.
const GRID_VARS_3D: [&str; 9] = ["P", "PB", "T", "QVAPOR", "PH", "PHB", "W", "U", "V"];
const GRID_VARS_2D: [&str; 10] = ["PSFC", "HGT", "T2", "Q2", "U10", "V10", "COSALPHA", "SINALPHA", "XLAT", "XLONG"];

impl Raw {
    fn open(f: &WrfFile, path: &std::path::Path, ti: usize) -> Res<Raw> {
        match f.hdf5() {
            PureRustFile::Classic(_) => Ok(Raw::Classic { nc: NcFile::open(path).map_err(err)?, ti }),
            PureRustFile::Hdf5(_) => {
                let nc = Nc4File::open(path).map_err(err)?;
                let whole = GRID_VARS_3D
                    .par_iter()
                    .chain(GRID_VARS_2D.par_iter())
                    .map(|&name| {
                        let shape = f.var_shape_no_time(name).map_err(err)?;
                        let data = nc4_words(&nc, name, ti)?;
                        let len = match &data {
                            Words::F32(w) => w.len(),
                            Words::F64(w) => w.len(),
                        };
                        if len != shape.iter().product::<usize>() {
                            return Err(format!("{name}: {len} values for the {shape:?} grid"));
                        }
                        Ok((name, (data, shape)))
                    })
                    .collect::<Res<_>>()?;
                Ok(Raw::Whole(whole))
            }
        }
    }

    /// Levels k0..k1 (ignored for a 2-D variable), rows j0..j1 and columns i0..i1 of `name` at this time, row-major.
    /// The classic route makes wrf-core's own selection (the time index when the variable has a Time axis, then
    /// every remaining axis) narrowed to the window, so each value converts identically.
    fn window(&self, name: &str, (k0, k1): (usize, usize), (j0, j1): (usize, usize), (i0, i1): (usize, usize)) -> Res<Vec<f64>> {
        use NcSliceInfoElem::{Index, Slice};
        match self {
            Raw::Classic { nc, ti } => {
                let var = nc.variable(name).map_err(|e| format!("{name}: {e}"))?;
                let shape = var.shape();
                let has_time =
                    shape.len() >= 2 && var.dimensions().first().is_some_and(|d| d.is_unlimited || d.name == "Time");
                let mut selections = vec![];
                if has_time {
                    selections.push(Index(*ti as u64));
                }
                let rank = shape.len() - usize::from(has_time);
                if rank == 3 {
                    selections.push(Slice { start: k0 as u64, end: k1 as u64, step: 1 });
                } else if rank != 2 {
                    return Err(format!("{name} has {rank} spatial axes; a grid variable needs 2 or 3"));
                }
                selections.push(Slice { start: j0 as u64, end: j1 as u64, step: 1 });
                selections.push(Slice { start: i0 as u64, end: i1 as u64, step: 1 });
                let a = nc
                    .read_variable_slice_as_f64(name, &NcSliceInfo { selections })
                    .map_err(|e| format!("{name}: {e}"))?;
                // The reader's own buffer when it is the window in order (always, for this read); else a copy.
                Ok(standard_vec!(a))
            }
            Raw::Whole(whole) => {
                let (data, shape) = whole.get(name).ok_or_else(|| format!("{name} was not read"))?;
                let (nyv, nxv) = (shape[shape.len() - 2], shape[shape.len() - 1]);
                let (k0, k1) = if shape.len() == 3 { (k0, k1) } else { (0, 1) };
                let mut out = Vec::with_capacity((k1 - k0) * (j1 - j0) * (i1 - i0));
                for k in k0..k1 {
                    for j in j0..j1 {
                        let row = (k * nyv + j) * nxv;
                        match data {
                            Words::F32(w) => out.extend(w[row + i0..row + i1].iter().map(|&v| f64::from(v))),
                            Words::F64(w) => out.extend_from_slice(&w[row + i0..row + i1]),
                        }
                    }
                }
                Ok(out)
            }
        }
    }
}

#[cfg(target_os = "linux")]
unsafe extern "C" {
    fn madvise(addr: *mut std::ffi::c_void, len: usize, advice: i32) -> i32;
}

/// A zeroed f64 field whose pages the kernel may back with 2 MB pages (Linux transparent huge pages, in the
/// "madvise" mode Ubuntu ships). A 1 km globe's grid is ~30 GB: at 4 KB pages its first touch and its free are ~8 M
/// page faults and unmaps the kernel serialises, which held the parallel grid at ~30 cores and its free at one core
/// for 5.6 s. The allocation is untouched until written (calloc maps zero pages), so the advice lands first.
fn huge_zeroed(len: usize) -> Vec<f64> {
    let v = vec![0.0f64; len];
    #[cfg(target_os = "linux")]
    if len * 8 >= 64 << 20 {
        const PAGE: usize = 4096;
        const MADV_HUGEPAGE: i32 = 14;
        let start = (v.as_ptr() as usize).next_multiple_of(PAGE);
        let end = (v.as_ptr() as usize + len * 8) & !(PAGE - 1);
        if end > start {
            // Advice only: a kernel without THP refuses it and the field stays on 4 KB pages.
            unsafe { madvise(start as *mut std::ffi::c_void, end - start, MADV_HUGEPAGE) };
        }
    }
    v
}

/// The grid of one history time, every level and row band on its own core: each task reads only its own window of
/// the raw variables and evaluates wrf-core's whole-field expressions on it in the same order (P+PB;
/// (T+300)*(p/P0)^kappa - 273.15; 0.5*(PH+PHB below + above)/g; 0.5*(U|V|W pair)), then omega, the earth rotation
/// and the dewpoint exactly as before, so every value is bit for bit what the serial whole-field route gave
/// (`tests::parallel_grid_is_bitwise_the_serial_one`). Classic files are read from the map by the tasks
/// themselves; NetCDF-4 files are read whole, one variable per core, first. Runs in the caller's rayon pool.
pub fn wrf_grid(f: &WrfFile, path: &std::path::Path, ti: usize) -> Res<WrfGrid> {
    let (nx, ny, nz) = (f.nx, f.ny, f.nz);
    if f.nx_stag != nx + 1 || f.ny_stag != ny + 1 || f.nz_stag != nz + 1 {
        return Err(format!(
            "{}: staggered axes U {} / V {} / PH {} do not fit the {nx}x{ny}x{nz} mass grid; the columns would misalign",
            path.display(),
            f.nx_stag,
            f.ny_stag,
            f.nz_stag
        ));
    }
    let n = nx * ny;
    let raw = Raw::open(f, path, ti)?;
    let cols = (0, nx);
    // The 2-D fields, one per core (each is a single plane).
    let mut planes: std::collections::HashMap<&str, Vec<f64>> = GRID_VARS_2D
        .par_iter()
        .map(|&name| Ok((name, raw.window(name, (0, 1), (0, ny), cols)?)))
        .collect::<Res<_>>()?;
    let mut take = |name: &str| planes.remove(name).ok_or_else(|| format!("{name} was not read"));
    let (ca, sa) = (take("COSALPHA")?, take("SINALPHA")?);
    // Rotation must precede horizontal interpolation. WRF stores grid-relative winds.
    let earth = |us: &[f64], vs: &[f64], out_u: &mut [f64], out_v: &mut [f64], at: usize| {
        for (r, ((u, v), (eu, ev))) in us.iter().zip(vs).zip(out_u.iter_mut().zip(out_v.iter_mut())).enumerate() {
            let i = at + r;
            *eu = u * ca[i] - v * sa[i];
            *ev = v * ca[i] + u * sa[i];
        }
    };
    let (u10g, v10g) = (take("U10")?, take("V10")?);
    let (mut u10, mut v10) = (vec![0.0; n], vec![0.0; n]);
    earth(&u10g, &v10g, &mut u10, &mut v10, 0);
    // Row bands: enough that every core has work when the levels are fewer than the cores, and small (~32 K points,
    // 256 KB a window) so each task's windows reuse the allocator's warm memory instead of faulting in fresh pages:
    // 8 MB windows held the 1 km globe's grid at ~23 cores.
    let min_bands = (4 * rayon::current_num_threads()).div_ceil(nz.max(1)).clamp(1, ny);
    let rows = (32_768 / nx).clamp(1, ny.div_ceil(min_bands));
    let band = rows * nx;
    let mut out: [Vec<f64>; 7] = std::array::from_fn(|_| huge_zeroed(nz * n));
    let [p, z, t, td, u, v, omega] = &mut out;
    let levels = (
        p.par_chunks_mut(n),
        z.par_chunks_mut(n),
        t.par_chunks_mut(n),
        td.par_chunks_mut(n),
        u.par_chunks_mut(n),
        v.par_chunks_mut(n),
        omega.par_chunks_mut(n),
    );
    levels.into_par_iter().enumerate().try_for_each(|(k, (p, z, t, td, u, v, omega))| -> Res<()> {
        let parts = (
            p.par_chunks_mut(band),
            z.par_chunks_mut(band),
            t.par_chunks_mut(band),
            td.par_chunks_mut(band),
            u.par_chunks_mut(band),
            v.par_chunks_mut(band),
            omega.par_chunks_mut(band),
        );
        parts.into_par_iter().enumerate().try_for_each(|(b, (p, z, t, td, u, v, omega))| -> Res<()> {
            let (j0, j1) = (b * rows, ((b + 1) * rows).min(ny));
            let m = (j1 - j0) * nx;
            let mass = |name: &str| raw.window(name, (k, k + 1), (j0, j1), cols);
            let stag = |name: &str| raw.window(name, (k, k + 2), (j0, j1), cols);
            let (pp, pb, th, qv) = (mass("P")?, mass("PB")?, mass("T")?, mass("QVAPOR")?);
            let (ph, phb, ww) = (stag("PH")?, stag("PHB")?, stag("W")?);
            let us = raw.window("U", (k, k + 1), (j0, j1), (0, nx + 1))?;
            let vs = raw.window("V", (k, k + 1), (j0, j1 + 1), cols)?;
            let mut ug = vec![0.0; m];
            let mut vg = vec![0.0; m];
            for r in 0..m {
                // wrf-core, in its order: full_pressure; full_theta -> temperature -> temperature_c;
                // full_geopotential (destagger_z) -> height_msl; w_destag; u_destag (destagger_x);
                // v_destag (destagger_y).
                let pr = pp[r] + pb[r];
                let theta = th[r] + 300.0;
                let tk = theta * (pr / P0).powf(KAPPA);
                let tc = tk - 273.15;
                let g_lo = ph[r] + phb[r];
                let g_hi = ph[m + r] + phb[m + r];
                let geopt = 0.5 * (g_lo + g_hi);
                let wm = 0.5 * (ww[r] + ww[m + r]);
                let (jj, i) = (r / nx, r % nx);
                ug[r] = 0.5 * (us[jj * (nx + 1) + i] + us[jj * (nx + 1) + i + 1]);
                vg[r] = 0.5 * (vs[jj * nx + i] + vs[(jj + 1) * nx + i]);
                let q = qv[r];
                p[r] = pr;
                z[r] = geopt / G;
                t[r] = tc;
                // Hydrostatic conversion of WRF vertical velocity to pressure velocity (Pa/s).
                omega[r] = -9.80665 * pr * wm / (287.05 * (tc + 273.15) * (1.0 + 0.61 * q));
                // Derive Td on the grid before bilinear sampling, like the site store.
                td[r] = dewpoint(pr, q);
            }
            earth(&ug, &vg, u, v, j0 * nx);
            Ok(())
        })
    })?;
    let [p, z, t, td, u, v, omega] = out;
    Ok(WrfGrid {
        nx,
        ny,
        nz,
        lat: take("XLAT")?,
        lon: take("XLONG")?,
        p,
        z,
        t,
        td,
        u,
        v,
        omega,
        ps: take("PSFC")?,
        terrain: take("HGT")?,
        t2: take("T2")?,
        q2: take("Q2")?,
        u10,
        v10,
    })
}

/// The serial whole-field route the parallel grid replaced (wrf-core's cached derived fields, then omega, the
/// rotation and the dewpoint), kept as the referee of `tests::parallel_grid_is_bitwise_the_serial_one`.
#[cfg(test)]
pub(crate) fn wrf_grid_serial(f: &WrfFile, ti: usize) -> Res<WrfGrid> {
    f.prepare_cache_for_time(ti);
    let p = f.full_pressure(ti).map_err(err)?;
    let z = f.height_msl(ti).map_err(err)?;
    let t = f.temperature_c(ti).map_err(err)?;
    let q = f.qvapor(ti).map_err(err)?;
    let w = f.w_destag(ti).map_err(err)?;
    let omega: Vec<f64> = p
        .iter()
        .zip(t.iter())
        .zip(q.iter())
        .zip(w.iter())
        .map(|(((p, t), q), w)| -9.80665 * p * w / (287.05 * (t + 273.15) * (1.0 + 0.61 * q)))
        .collect();
    let u = f.u_destag(ti).map_err(err)?;
    let v = f.v_destag(ti).map_err(err)?;
    let ca = f.cosalpha(ti).map_err(err)?;
    let sa = f.sinalpha(ti).map_err(err)?;
    let n = f.nxy();
    let earth = |us: &[f64], vs: &[f64]| -> (Vec<f64>, Vec<f64>) {
        us.iter()
            .zip(vs)
            .enumerate()
            .map(|(i, (u, v))| (u * ca[i % n] - v * sa[i % n], v * ca[i % n] + u * sa[i % n]))
            .unzip()
    };
    let (u, v) = earth(&u, &v);
    let (u10, v10) = earth(&f.u10(ti).map_err(err)?, &f.v10(ti).map_err(err)?);
    let td: Vec<f64> = p.iter().zip(q.iter()).map(|(p, q)| dewpoint(*p, *q)).collect();
    let g = WrfGrid {
        nx: f.nx,
        ny: f.ny,
        nz: f.nz,
        lat: f.xlat(ti).map_err(err)?.to_vec(),
        lon: f.xlong(ti).map_err(err)?.to_vec(),
        p: p.to_vec(),
        z: z.to_vec(),
        t: t.to_vec(),
        td,
        u,
        v,
        omega,
        ps: f.psfc(ti).map_err(err)?.to_vec(),
        terrain: f.terrain(ti).map_err(err)?.to_vec(),
        t2: f.t2(ti).map_err(err)?.to_vec(),
        q2: f.q2(ti).map_err(err)?.to_vec(),
        u10,
        v10,
    };
    f.clear_cache();
    Ok(g)
}

impl WrfGrid {
    pub(crate) fn locator(&self) -> GridFile {
        GridFile {
            nx: self.nx,
            ny: self.ny,
            lat: self.lat.iter().map(|x| *x as f32).collect(),
            lon: self.lon.iter().map(|x| *x as f32).collect(),
            projection: None,
            hash: String::new(),
        }
    }
    /// 2 m dewpoint (C) at one grid point.
    pub fn td2(&self, i: usize) -> f64 {
        dewpoint(self.ps[i], self.q2[i])
    }
    /// The bilinear column at fractional grid position (x, y).
    pub fn column(&self, x: f64, y: f64) -> RawColumn {
        let n = self.nx * self.ny;
        let sample = |a: &[f64], k: usize| {
            let ix = x.floor() as usize;
            let iy = y.floor() as usize;
            let jx = (ix + 1).min(self.nx - 1);
            let jy = (iy + 1).min(self.ny - 1);
            let ax = x - ix as f64;
            let ay = y - iy as f64;
            let g = |i, j| a[k * n + j * self.nx + i];
            (g(ix, iy) * (1.0 - ax) + g(jx, iy) * ax) * (1.0 - ay)
                + (g(ix, jy) * (1.0 - ax) + g(jx, jy) * ax) * ay
        };
        let mut c = RawColumn {
            sfc: Sfc {
                psfc_hpa: sample(&self.ps, 0) / 100.0,
                orog_m: sample(&self.terrain, 0),
                t2_c: sample(&self.t2, 0) - 273.15,
                td2_c: dewpoint(sample(&self.ps, 0), sample(&self.q2, 0)),
                u10: sample(&self.u10, 0),
                v10: sample(&self.v10, 0),
            },
            ..Default::default()
        };
        // Derive Td on the grid before bilinear sampling, like the site store.
        for k in 0..self.nz {
            c.p.push(sample(&self.p, k) / 100.0);
            c.z.push(sample(&self.z, k));
            c.t.push(sample(&self.t, k));
            c.td.push(sample(&self.td, k));
            c.u.push(sample(&self.u, k));
            c.v.push(sample(&self.v, k));
            c.omega.push(sample(&self.omega, k));
        }
        c
    }
}

/// A column frame already sampled elsewhere (the any-point sounding service reads one column of a run's column store
/// and hands it over as JSON): `{"valid": "2024-07-14T16:00:00Z", "columns": [RawColumn, ...]}`, one column per
/// requested point in order. Missing values are null. The run's own description (`init`, `model`, `shape`,
/// `source`, as the column store's header `run` holds them) may ride along for the label.
#[derive(Deserialize)]
struct JsonFrame {
    valid: String,
    columns: Vec<RawColumn>,
    #[serde(flatten)]
    run: RunInfo,
}

/// One any-point column store file (rw-store version 2, or a 2.9.6 `GPWMCOL1` file): the grid from its own
/// lat/lon, each point located on it as on a history file, then the one tile that holds the point, sampled
/// bilinearly. The stored units are the bridge units already (hPa, m MSL, C, m/s, Pa/s).
fn read_column_store(source: &Source, points: &[Point]) -> Res<Vec<Frame>> {
    use rw_store::columns::ColumnFile;
    let at = |e: String| format!("{}: {e}", source.path.display());
    let f = ColumnFile::open(&source.path).map_err(|e| at(e.to_string()))?;
    let h = &f.header;
    let valid = match h.valid_unix {
        Some(t) => t,
        None => timestamp(&h.valid).map_err(at)?,
    };
    let (lat, lon) = f.lat_lon().map_err(|e| at(e.to_string()))?;
    let grid = GridFile { nx: h.nx, ny: h.ny, lat, lon, projection: None, hash: String::new() };
    let f3 = |name: &str| h.field3(name).ok_or_else(|| at(format!("the store has no {name}")));
    let f2 = |name: &str| h.field2(name).ok_or_else(|| at(format!("the store has no {name}")));
    let (ip, iz, it, itd, iu, iv) = (f3("p_hpa")?, f3("z_m")?, f3("t_c")?, f3("td_c")?, f3("u_ms")?, f3("v_ms")?);
    let iw = h.field3("omega_pas");
    let surface = [f2("psfc_hpa")?, f2("orog_m")?, f2("t2_c")?, f2("td2_c")?, f2("u10_ms")?, f2("v10_ms")?];
    let mut columns = Vec::with_capacity(points.len());
    for p in points {
        let (x, y) = locate(&grid, p)?;
        let tile = f.tile_at(x, y).map_err(|e| at(e.to_string()))?;
        let prof = |i: usize| tile.profile3(i, x, y, h.nx, h.ny).into_iter().map(f64::from).collect::<Vec<_>>();
        let (x0, y0) = (x.floor() as usize, y.floor() as usize);
        let (x1, y1) = ((x0 + 1).min(h.nx - 1), (y0 + 1).min(h.ny - 1));
        let (ax, ay) = (x - x0 as f64, y - y0 as f64);
        let point = |i: usize| {
            let g = |ix, iy| f64::from(tile.point2(i, ix, iy));
            (g(x0, y0) * (1.0 - ax) + g(x1, y0) * ax) * (1.0 - ay) + (g(x0, y1) * (1.0 - ax) + g(x1, y1) * ax) * ay
        };
        let s: Vec<f64> = surface.iter().map(|i| point(*i)).collect();
        let p3 = prof(ip);
        columns.push(RawColumn {
            omega: iw.map(prof).unwrap_or_else(|| vec![f64::NAN; p3.len()]),
            p: p3,
            z: prof(iz),
            t: prof(it),
            td: prof(itd),
            u: prof(iu),
            v: prof(iv),
            sfc: Sfc { psfc_hpa: s[0], orog_m: s[1], t2_c: s[2], td2_c: s[3], u10: s[4], v10: s[5] },
        });
    }
    // The writer's record of its run (`meta.run`: init, model, shape, source), for the label.
    let run = h.meta.get("run").and_then(|v| serde_json::from_value::<RunInfo>(v.clone()).ok()).unwrap_or_default();
    Ok(vec![Frame { domain: source.domain.clone(), member: source.member.clone(), valid, columns, run }])
}

fn read_json_frame(source: &Source, points: &[Point]) -> Res<Vec<Frame>> {
    let bytes = std::fs::read(&source.path).map_err(err)?;
    let mut frame: JsonFrame =
        serde_json::from_slice(&bytes).map_err(|e| format!("{}: {e}", source.path.display()))?;
    for c in &mut frame.columns {
        if c.omega.is_empty() {
            c.omega = vec![f64::NAN; c.p.len()];
        }
    }
    if frame.columns.len() != points.len() {
        return Err(format!(
            "{} carries {} columns for {} points",
            source.path.display(),
            frame.columns.len(),
            points.len()
        ));
    }
    if let Some(bad) = frame.columns.iter().find(|c| {
        let n = c.p.len();
        n < 2 || [&c.z, &c.t, &c.td, &c.u, &c.v, &c.omega].iter().any(|a| a.len() != n)
    }) {
        return Err(format!("a column in {} has unequal or too few levels ({})", source.path.display(), bad.p.len()));
    }
    Ok(vec![Frame {
        domain: source.domain.clone(),
        member: source.member.clone(),
        valid: timestamp(&frame.valid)?,
        columns: frame.columns,
        run: frame.run,
    }])
}
/// Common log-pressure ladder for ensemble statistics. Never average unrelated model levels.
pub fn common_columns(cols: &[RawColumn]) -> Vec<RawColumn> {
    let levels: Vec<f64> = (2..=40).rev().map(|n| n as f64 * 25.0).collect();
    cols.iter()
        .map(|c| {
            let mut o = RawColumn {
                sfc: c.sfc,
                ..Default::default()
            };
            for p in &levels {
                let pair = c.p.windows(2).position(|w| w[0] >= *p && w[1] <= *p);
                let interp = |a: &[f64]| {
                    pair.map(|k| {
                        let w = (p.ln() - c.p[k].ln()) / (c.p[k + 1].ln() - c.p[k].ln());
                        a[k] + w * (a[k + 1] - a[k])
                    })
                    .unwrap_or(f64::NAN)
                };
                o.p.push(*p);
                o.z.push(interp(&c.z));
                o.t.push(interp(&c.t));
                o.td.push(interp(&c.td));
                o.u.push(interp(&c.u));
                o.v.push(interp(&c.v));
                o.omega.push(interp(&c.omega));
            }
            o
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn synthetic_profile_surface_units_common_pressure_and_spread() {
        let base = RawColumn {
            p: vec![950., 800., 600., 400., 200., 75.],
            z: vec![500., 1800., 4000., 7000., 11500., 18000.],
            t: vec![23., 12., -2., -20., -52., -65.],
            td: vec![19., 8., -9., -30., -60., -75.],
            u: vec![3., 8., 15., 22., 30., 40.],
            v: vec![5., 7., 10., 15., 20., 25.],
            omega: vec![f64::NAN; 6],
            sfc: Sfc {
                psfc_hpa: 980.,
                orog_m: 250.,
                t2_c: 26.,
                td2_c: 21.,
                u10: 2.,
                v10: 4.,
            },
        };
        let mut warm = base.clone();
        for t in &mut warm.t {
            *t += 2.;
        }
        let cols = common_columns(&[base.clone(), warm]);
        let k = cols[0].p.iter().position(|p| *p == 800.).unwrap();
        assert_eq!(cols[0].t[k], 12.);
        let mean = crate::store::mean_columns(&cols).unwrap();
        assert_eq!(mean.t[k], 13.);
        let mut samples = vec![cols[0].t[k], cols[1].t[k]];
        assert!((crate::store::percentile(&mut samples, 10.) - 12.2).abs() < 1e-9);
        assert!((crate::store::percentile(&mut samples, 90.) - 13.8).abs() < 1e-9);
        let column = crate::analysis::to_column(&base, Default::default()).unwrap();
        assert_eq!(column.pressure_hpa[0], 980.);
        assert_eq!(column.temperature_c[0], 26.);
        let a = crate::analysis::analyze(column, &Default::default()).unwrap();
        assert!(!a.prof.inner.u.is_empty());
        assert!(dewpoint(100000., 0.014) > 18.);
    }
}
