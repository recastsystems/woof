//! An MPAS-to-Lambert post-processing remap that writes what MPASSIT writes.
//!
//! RRFS v2 post-processes every MPAS forecast hour in two steps: MPASSIT
//! interpolates the MPAS history and diag files onto a WRF-shaped Lambert grid
//! (for conus3km the HRRR grid: `nx = 1800, ny = 1060` staggered points,
//! 3 km, centred 38.5N 97.5W), then UPP turns that file into GRIB2.  This
//! module is the first step: given MPASSIT's own namelist and variable lists
//! it writes a netCDF file with the variables, dimensions and values MPASSIT
//! writes, so NOAA's downstream tools (UPP and everything after it) read hex
//! output exactly as they read RRFS output.
//!
//! LICENCE AND PROVENANCE.  MPASSIT (NOAA-GSL/MPASSIT 58fc18d, the commit
//! rrfs-workflow `rrfs-mpas-jedi` 570c81f pins) is GPL-3.0, which this
//! Apache-2.0 tree may not carry.  No MPASSIT source was read or ported.
//! What is here comes from three permitted places:
//!
//! * MPASSIT's documentation (README, 2024 WRF/MPAS workshop slides): integer
//!   fields use nearest neighbour, `snow`/`snowh` are first-order
//!   conservative, everything else is bilinear; 3-D history variables have
//!   `nVertLevels` levels except `zgrid` and `w`; `wrf_mod_vars` makes the
//!   file WRF-shaped (metadata, staggered winds, extra variables).
//! * WRF's public-domain map projection (`set_lc`/`ijll_lc`, already in
//!   [`crate::window`]) for the target grid, and the published geometry of
//!   barycentric interpolation and spherical polygon areas.
//! * Black-box measurement: the referee binary built on one lab host only, run on
//!   native MPAS-A/RRFS history, and its output compared value by value
//!   against candidate methods.  What the measurement fixed:
//!   - bilinear = barycentric weights in the triangle of the three cell
//!     centres meeting at an MPAS vertex (`cellsOnVertex`), after RADIAL
//!     projection of the target onto the triangle's plane (an orthogonal
//!     projection misses by 1e-4 relative on a 120 km mesh); points in no
//!     triangle are missing (-9999);
//!   - nearest = the cell whose centre is nearest (chord distance), and only
//!     where the bilinear map has a value; soil fields are nearest too;
//!   - the grid's known point is `(e_we/2, e_sn/2)` in WRF's 1-based mass
//!     coordinates, `CEN_LAT/CEN_LON` are at `(nx/2, ny/2)` of the mass grid;
//!   - `SINALPHA/COSALPHA` come from centred differences of latitude and
//!     longitude along the grid's y direction (one-sided on the first and last
//!     rows); `U10/V10` and the 3-D winds are rotated with them, the 10 m
//!     means are not;
//!   - 3-D `U`/`V` are the rotated mass-point winds carried to the staggered
//!     points by a bilinear regrid from the mass grid's own cells (radial
//!     projection, the cell on the point's side of the grid line); 0 where a
//!     staggered point lies outside those cells (the outer U columns and V
//!     rows, and outer-column V points the grid's curvature leaves outside),
//!     missing where the staggered point is off the MPAS mesh;
//!   - `T = theta - 300`, `PHB = zgrid` (metres), `Z_C = 9.81 zgrid` (its top
//!     level unwritten, 0 off the mesh), `PB = P_HYD = pressure`,
//!     `MUB = rho`, and `PH`, `P`, `MU` are zero;
//!   - `XTIME`/`ITIMESTEP` count from the valid time back to the start (both
//!     negative for a forecast), exactly as the referee writes them.
//!
//! What was NOT reproduced bit for bit, and is stated in every receipt: the
//! referee's own floating-point order (ESMF's sparse multiply), a handful of
//! nearest-cell near-ties (two cell centres equidistant to 1e-7), and the
//! last few bits of the staggered-wind average.  The proof compares every
//! variable against the referee on native history and reports the maximum
//! difference per variable.

use rayon::prelude::*;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use rw_store::netcdf_classic::{
    NcAttr, NcClassicWriter, NcData, NcDim, NcFormat, NcSlab, NcType, NcVarDef,
};

use crate::error::{MpasError, MpasResult};
use crate::remap::spherical_barycentric;
use crate::weights::{detect_index_base, IndexBase, KdTree};
use crate::window::LambertProjection;

/// MPASSIT's missing value (its `missing_value` namelist default).
pub const MISSING: f32 = -9999.0;
/// netCDF's default float fill, which the referee leaves in the never-written
/// top level of `Z_C`.
pub const NC_FILL_FLOAT: f32 = 9.969_209_968_386_869e36;
/// The factor the referee writes into `Z_C` (`Z_C = 9.81 * zgrid`).
pub const ZC_GRAVITY: f64 = 9.81;
/// WRF's perturbation potential temperature offset (`T = theta - 300`).
pub const THETA_OFFSET: f64 = 300.0;
/// A point is inside a triangle when its smallest barycentric weight is at
/// least this.
const INSIDE_EPS: f64 = -1e-9;

pub const SCHEMA: &str = "gpuwm.mpassit-equivalent-remap/v1";
pub const METHOD: &str = "MPASSIT-equivalent: radial barycentric on the cellsOnVertex triangles \
    (bilinear class), nearest cell centre (nearest class, soil), exact spherical-polygon overlap \
    over destination area (snow, snowh); WRF Lambert grid; implemented from MPASSIT's documented \
    method and black-box measurement, not from its GPL-3.0 source";

fn refusal(text: impl Into<String>) -> MpasError {
    MpasError::Refusal(text.into())
}

// ---------------------------------------------------------------------------
// Namelist and variable lists (MPASSIT's own formats, read as-is).
// ---------------------------------------------------------------------------

/// The `&config` namelist MPASSIT runs from.  Paths are resolved against the
/// namelist's directory, which is the run directory in RRFS's job script.
#[derive(Debug, Clone)]
pub struct Namelist {
    pub grid_file: PathBuf,
    pub hist_file: PathBuf,
    pub diag_file: PathBuf,
    pub output_file: PathBuf,
    pub target_grid_type: String,
    pub interp_diag: bool,
    pub interp_hist: bool,
    pub wrf_mod_vars: bool,
    pub is_regional: bool,
    pub nx: usize,
    pub ny: usize,
    pub dx: f64,
    pub dy: f64,
    pub ref_lat: f64,
    pub ref_lon: f64,
    pub truelat1: f64,
    pub truelat2: f64,
    pub stand_lon: f64,
    pub missing_value: f32,
}

fn split_outside_quotes(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut quote: Option<char> = None;
    let mut comment = false;
    for ch in text.chars() {
        if comment {
            if ch == '\n' {
                comment = false;
                if !cur.trim().is_empty() {
                    out.push(cur.trim().to_string());
                }
                cur.clear();
            }
            continue;
        }
        match quote {
            Some(q) => {
                cur.push(ch);
                if ch == q {
                    quote = None;
                }
            }
            None => match ch {
                '"' | '\'' => {
                    quote = Some(ch);
                    cur.push(ch);
                }
                ',' | '\n' | '\r' => {
                    if !cur.trim().is_empty() {
                        out.push(cur.trim().to_string());
                    }
                    cur.clear();
                }
                '!' => comment = true,
                _ => cur.push(ch),
            },
        }
    }
    if !cur.trim().is_empty() {
        out.push(cur.trim().to_string());
    }
    out
}

fn parse_logical(key: &str, value: &str) -> MpasResult<bool> {
    let v = value.trim().trim_matches('.').to_ascii_lowercase();
    match v.as_str() {
        "true" | "t" => Ok(true),
        "false" | "f" => Ok(false),
        _ => Err(refusal(format!("namelist {key} = {value} is not a Fortran logical"))),
    }
}

impl Namelist {
    pub fn parse(text: &str, base: &Path) -> MpasResult<Namelist> {
        let body = text
            .find('&')
            .map(|at| &text[at..])
            .ok_or_else(|| refusal("namelist has no &config group"))?;
        let body = body.splitn(2, |c: char| c.is_whitespace()).nth(1).unwrap_or("");
        let body = match body.rfind('/') {
            Some(end) if body[end..].trim() == "/" => &body[..end],
            _ => body,
        };
        let mut kv: BTreeMap<String, String> = BTreeMap::new();
        for item in split_outside_quotes(body) {
            if let Some((k, v)) = item.split_once('=') {
                let v = v.trim();
                let v = v
                    .strip_prefix('"')
                    .and_then(|s| s.strip_suffix('"'))
                    .or_else(|| v.strip_prefix('\'').and_then(|s| s.strip_suffix('\'')))
                    .unwrap_or(v);
                kv.insert(k.trim().to_ascii_lowercase(), v.to_string());
            }
        }
        let get = |k: &str| kv.get(k).cloned();
        let need = |k: &str| {
            get(k).ok_or_else(|| refusal(format!("namelist has no {k}; MPASSIT needs it too")))
        };
        let path = |k: &str| -> MpasResult<PathBuf> {
            let p = PathBuf::from(need(k)?);
            Ok(if p.is_absolute() { p } else { base.join(p) })
        };
        let num = |k: &str| -> MpasResult<f64> {
            let v = need(k)?;
            v.replace(['d', 'D'], "e")
                .parse::<f64>()
                .map_err(|_| refusal(format!("namelist {k} = {v} is not a number")))
        };
        let logical = |k: &str, default: bool| -> MpasResult<bool> {
            match get(k) {
                Some(v) => parse_logical(k, &v),
                None => Ok(default),
            }
        };
        let target_grid_type = get("target_grid_type").unwrap_or_else(|| "lambert".into());
        let nx = num("nx")?;
        let ny = num("ny")?;
        if nx.fract() != 0.0 || ny.fract() != 0.0 || nx < 3.0 || ny < 3.0 {
            return Err(refusal(format!("namelist nx = {nx}, ny = {ny} must be whole and at least 3")));
        }
        let ref_lat = num("ref_lat")?;
        let ref_lon = num("ref_lon")?;
        Ok(Namelist {
            grid_file: path("grid_file_input_grid")?,
            hist_file: path("hist_file_input_grid")?,
            diag_file: path("diag_file_input_grid")?,
            output_file: path("output_file")?,
            target_grid_type,
            interp_diag: logical("interp_diag", true)?,
            interp_hist: logical("interp_hist", true)?,
            wrf_mod_vars: logical("wrf_mod_vars", false)?,
            is_regional: logical("is_regional", true)?,
            nx: nx as usize,
            ny: ny as usize,
            dx: num("dx")?,
            dy: get("dy").map(|_| num("dy")).transpose()?.unwrap_or(num("dx")?),
            ref_lat,
            ref_lon,
            truelat1: get("truelat1").map(|_| num("truelat1")).transpose()?.unwrap_or(ref_lat),
            truelat2: get("truelat2").map(|_| num("truelat2")).transpose()?.unwrap_or(ref_lat),
            stand_lon: get("stand_lon").map(|_| num("stand_lon")).transpose()?.unwrap_or(ref_lon),
            missing_value: get("missing_value")
                .map(|_| num("missing_value"))
                .transpose()?
                .map(|v| v as f32)
                .unwrap_or(MISSING),
        })
    }

    /// The configurations this remap reproduces.  Anything else is refused by
    /// name, because the referee's output for it was never measured and a
    /// file that only looks like MPASSIT's would be read by UPP as if it were.
    pub fn check_supported(&self) -> MpasResult<()> {
        if !self.target_grid_type.eq_ignore_ascii_case("lambert") {
            return Err(refusal(format!(
                "target_grid_type '{}': only 'lambert' (RRFS v2's grid type) is implemented; the \
                 'file' and 'lat-lon' targets were never measured against MPASSIT, so their output \
                 would carry MPASSIT's name without its values",
                self.target_grid_type
            )));
        }
        if !self.wrf_mod_vars {
            return Err(refusal(
                "wrf_mod_vars = .false.: only the WRF-shaped file RRFS v2 writes (wrf_mod_vars = \
                 .true.) is implemented and measured; the unstaggered layout was never compared \
                 against MPASSIT",
            ));
        }
        if (self.dx - self.dy).abs() > 1e-9 {
            return Err(refusal(format!(
                "dx {} != dy {}: a Lambert WRF grid has one spacing",
                self.dx, self.dy
            )));
        }
        if self.missing_value != MISSING {
            return Err(refusal(format!(
                "missing_value {}: only MPASSIT's default -9999 was measured",
                self.missing_value
            )));
        }
        Ok(())
    }
}

/// One line of a MPASSIT variable list: MPAS name, output name.
#[derive(Debug, Clone, PartialEq)]
pub struct ListEntry {
    pub mpas: String,
    pub wrf: String,
}

pub fn read_list(path: &Path) -> MpasResult<Vec<ListEntry>> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| refusal(format!("variable list {}: {e}", path.display())))?;
    let mut out = Vec::new();
    for line in text.lines() {
        let mut words = line.split_whitespace();
        let (Some(mpas), Some(wrf)) = (words.next(), words.next()) else {
            continue;
        };
        if mpas.starts_with('#') || mpas.starts_with('!') {
            continue;
        }
        out.push(ListEntry {
            mpas: mpas.to_string(),
            wrf: wrf.to_string(),
        });
    }
    Ok(out)
}

/// MPASSIT's four lists, read from one directory (its run directory).
#[derive(Debug, Clone, Default)]
pub struct VariableLists {
    pub diag: Vec<ListEntry>,
    pub hist_2d: Vec<ListEntry>,
    pub hist_3d: Vec<ListEntry>,
    pub hist_soil: Vec<ListEntry>,
}

impl VariableLists {
    pub fn read(dir: &Path) -> MpasResult<VariableLists> {
        let opt = |name: &str| -> MpasResult<Vec<ListEntry>> {
            let p = dir.join(name);
            if p.exists() {
                read_list(&p)
            } else {
                Ok(Vec::new())
            }
        };
        Ok(VariableLists {
            diag: opt("diaglist")?,
            hist_2d: opt("histlist_2d")?,
            hist_3d: opt("histlist_3d")?,
            hist_soil: opt("histlist_soil")?,
        })
    }
}

/// How a 2-D history variable is remapped (MPASSIT's documented classes).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Class {
    Bilinear,
    Nearest,
    Conservative,
}

pub fn class_of_hist_2d(mpas: &str) -> Class {
    match mpas {
        "snow" | "snowh" => Class::Conservative,
        "ivgtyp" | "isltyp" | "xland" | "landmask" => Class::Nearest,
        _ => Class::Bilinear,
    }
}

// ---------------------------------------------------------------------------
// Target grid.
// ---------------------------------------------------------------------------

/// The WRF Lambert grid MPASSIT lays down from its namelist.
#[derive(Debug, Clone)]
pub struct TargetGrid {
    /// Staggered sizes (`nx`, `ny` in the namelist; WRF's `e_we`, `e_sn`).
    pub e_we: usize,
    pub e_sn: usize,
    /// Mass-point sizes.
    pub nx: usize,
    pub ny: usize,
    pub dx: f64,
    pub truelat1: f64,
    pub truelat2: f64,
    pub stand_lon: f64,
    pub proj: LambertProjection,
}

impl TargetGrid {
    pub fn from_namelist(nl: &Namelist) -> TargetGrid {
        let proj = LambertProjection::set_lc(
            nl.truelat1,
            nl.truelat2,
            nl.stand_lon,
            nl.dx,
            nl.ref_lat,
            nl.ref_lon,
            nl.nx as f64 / 2.0,
            nl.ny as f64 / 2.0,
        );
        TargetGrid {
            e_we: nl.nx,
            e_sn: nl.ny,
            nx: nl.nx - 1,
            ny: nl.ny - 1,
            dx: nl.dx,
            truelat1: nl.truelat1,
            truelat2: nl.truelat2,
            stand_lon: nl.stand_lon,
            proj,
        }
    }

    /// `(lat, lon)` in degrees of a `ni x nj` lattice whose first point is at
    /// 1-based grid coordinate `(i0, j0)`, row-major (j slowest).
    pub fn lattice(&self, ni: usize, nj: usize, i0: f64, j0: f64) -> (Vec<f64>, Vec<f64>) {
        let rows: Vec<(Vec<f64>, Vec<f64>)> = (0..nj)
            .into_par_iter()
            .map(|j| {
                let mut la = Vec::with_capacity(ni);
                let mut lo = Vec::with_capacity(ni);
                for i in 0..ni {
                    let (a, b) = self.proj.ijll(i0 + i as f64, j0 + j as f64);
                    la.push(a);
                    lo.push(b);
                }
                (la, lo)
            })
            .collect();
        let mut lat = Vec::with_capacity(ni * nj);
        let mut lon = Vec::with_capacity(ni * nj);
        for (a, b) in rows {
            lat.extend(a);
            lon.extend(b);
        }
        (lat, lon)
    }

    pub fn mass(&self) -> (Vec<f64>, Vec<f64>) {
        self.lattice(self.nx, self.ny, 1.0, 1.0)
    }
    pub fn u_stag(&self) -> (Vec<f64>, Vec<f64>) {
        self.lattice(self.nx + 1, self.ny, 0.5, 1.0)
    }
    pub fn v_stag(&self) -> (Vec<f64>, Vec<f64>) {
        self.lattice(self.nx, self.ny + 1, 1.0, 0.5)
    }
    pub fn corners(&self) -> (Vec<f64>, Vec<f64>) {
        self.lattice(self.nx + 1, self.ny + 1, 0.5, 0.5)
    }
    /// `CEN_LAT`, `CEN_LON` as the referee writes them.
    pub fn centre(&self) -> (f64, f64) {
        self.proj.ijll(self.nx as f64 / 2.0, self.ny as f64 / 2.0)
    }

    pub fn cone(&self) -> f64 {
        let rad = std::f64::consts::PI / 180.0;
        if (self.truelat1 - self.truelat2).abs() > 0.1 {
            (libm::log10(libm::cos(self.truelat1 * rad)) - libm::log10(libm::cos(self.truelat2 * rad)))
                / (libm::log10(libm::tan((45.0 - self.truelat1.abs() / 2.0) * rad))
                    - libm::log10(libm::tan((45.0 - self.truelat2.abs() / 2.0) * rad)))
        } else {
            libm::sin(self.truelat1.abs() * rad)
        }
    }

    /// Lambert map factor at a latitude (degrees).
    pub fn map_factor(&self, lat: f64) -> f64 {
        let rad = std::f64::consts::PI / 180.0;
        let hemi = if self.truelat1 >= 0.0 { 1.0 } else { -1.0 };
        let colat0 = (90.0 - hemi * self.truelat1) * rad;
        let colat = (90.0 - hemi * lat) * rad;
        (libm::sin(colat0) / libm::sin(colat))
            * libm::pow(libm::tan(colat / 2.0) / libm::tan(colat0 / 2.0), self.cone())
    }
}

/// `SINALPHA`, `COSALPHA` at mass points: the grid's y direction from centred
/// differences of latitude/longitude along each column (one-sided on the
/// first and last rows).
pub fn rotation(lat_deg: &[f64], lon_deg: &[f64], nx: usize, ny: usize) -> (Vec<f64>, Vec<f64>) {
    let rad = std::f64::consts::PI / 180.0;
    let mut sina = vec![0.0; nx * ny];
    let mut cosa = vec![0.0; nx * ny];
    sina.par_chunks_mut(nx)
        .zip(cosa.par_chunks_mut(nx))
        .enumerate()
        .for_each(|(j, (srow, crow))| {
            let (ja, jb) = if j == 0 {
                (0, 1)
            } else if j == ny - 1 {
                (ny - 2, ny - 1)
            } else {
                (j - 1, j + 1)
            };
            for i in 0..nx {
                let dlat = (lat_deg[jb * nx + i] - lat_deg[ja * nx + i]) * rad;
                let mut dlon = (lon_deg[jb * nx + i] - lon_deg[ja * nx + i]) * rad;
                if dlon > std::f64::consts::PI {
                    dlon -= 2.0 * std::f64::consts::PI;
                } else if dlon < -std::f64::consts::PI {
                    dlon += 2.0 * std::f64::consts::PI;
                }
                let east = dlon * libm::cos(lat_deg[j * nx + i] * rad);
                let hyp = libm::hypot(east, dlat);
                srow[i] = -east / hyp;
                crow[i] = dlat / hyp;
            }
        });
    (sina, cosa)
}

pub fn unit(lat_deg: f64, lon_deg: f64) -> [f64; 3] {
    let rad = std::f64::consts::PI / 180.0;
    unit_rad(lat_deg * rad, lon_deg * rad)
}

pub fn unit_rad(lat: f64, lon: f64) -> [f64; 3] {
    let c = libm::cos(lat);
    [c * libm::cos(lon), c * libm::sin(lon), libm::sin(lat)]
}

// ---------------------------------------------------------------------------
// Source mesh.
// ---------------------------------------------------------------------------

/// What the remap needs from an MPAS grid/init/invariant file.
#[derive(Debug)]
pub struct SourceMesh {
    pub n_cells: usize,
    pub n_vert_levels: usize,
    pub centres: Vec<[f64; 3]>,
    pub vertices: Vec<[f64; 3]>,
    /// Zero-based; `u32::MAX` where the vertex lacks a cell (limited area).
    pub cells_on_vertex: Vec<[u32; 3]>,
    /// `offsets[c]..offsets[c+1]` indexes `vertices_on_cell` and `cells_on_cell`.
    pub offsets: Vec<u32>,
    pub vertices_on_cell: Vec<u32>,
    pub cells_on_cell: Vec<u32>,
    pub terrain: Vec<f64>,
    pub(crate) tree: KdTree,
    /// A spherical cap holding every cell polygon (centre, cosine of its
    /// angular radius); `None` when the mesh covers the sphere.  A target
    /// outside it is in no triangle and overlaps no cell, so it is skipped
    /// before the nearest-cell search, which is slowest for far points.
    pub cap: Option<([f64; 3], f64)>,
}

fn read_index(file: &netcrust::File, name: &str, path: &Path) -> MpasResult<Vec<i64>> {
    if file.variable(name).is_none() {
        return Err(refusal(format!(
            "grid file {} has no {name}; the remap's triangles and cell polygons come from it",
            path.display()
        )));
    }
    Ok(file.read_f64(name)?.into_iter().map(|v| v.round() as i64).collect())
}

impl SourceMesh {
    pub fn read(path: &Path) -> MpasResult<SourceMesh> {
        let file = netcrust::File::open(path)?;
        let lat = file.read_f64("latCell")?;
        let lon = file.read_f64("lonCell")?;
        let n_cells = lat.len();
        let latv = file.read_f64("latVertex")?;
        let lonv = file.read_f64("lonVertex")?;
        let n_vertices = latv.len();
        let n_vert_levels = file
            .dimension("nVertLevels")
            .map(|d| d.len())
            .ok_or_else(|| {
                refusal(format!(
                    "grid file {} has no nVertLevels dimension: give the init/invariant file \
                     (MPASSIT refuses a static file for the same reason)",
                    path.display()
                ))
            })?;
        let n_edges = read_index(&file, "nEdgesOnCell", path)?;
        let voc = read_index(&file, "verticesOnCell", path)?;
        let coc = read_index(&file, "cellsOnCell", path)?;
        let cov = read_index(&file, "cellsOnVertex", path)?;
        if n_edges.len() != n_cells || voc.len() % n_cells != 0 || cov.len() != 3 * n_vertices {
            return Err(refusal(format!("grid file {} connectivity shapes disagree", path.display())));
        }
        let max_edges = voc.len() / n_cells;
        let (mut lo, mut hi) = (i64::MAX, i64::MIN);
        for c in 0..n_cells {
            for s in 0..n_edges[c] as usize {
                lo = lo.min(voc[c * max_edges + s]);
                hi = hi.max(voc[c * max_edges + s]);
            }
        }
        let shift = if detect_index_base(lo, hi, n_vertices, "verticesOnCell")? == IndexBase::One {
            1
        } else {
            0
        };
        let fix = |v: i64, n: usize| -> u32 {
            let z = v - shift;
            if z >= 0 && (z as usize) < n {
                z as u32
            } else {
                u32::MAX
            }
        };
        let mut offsets = Vec::with_capacity(n_cells + 1);
        let mut vertices_on_cell = Vec::with_capacity(n_cells * 6);
        let mut cells_on_cell = Vec::with_capacity(n_cells * 6);
        offsets.push(0u32);
        for c in 0..n_cells {
            let used = n_edges[c];
            if used < 3 || used as usize > max_edges {
                return Err(refusal(format!("grid file {} nEdgesOnCell[{c}] = {used}", path.display())));
            }
            for s in 0..used as usize {
                vertices_on_cell.push(fix(voc[c * max_edges + s], n_vertices));
                cells_on_cell.push(fix(coc[c * max_edges + s], n_cells));
            }
            offsets.push(vertices_on_cell.len() as u32);
        }
        let cells_on_vertex: Vec<[u32; 3]> = cov
            .chunks_exact(3)
            .map(|r| [fix(r[0], n_cells), fix(r[1], n_cells), fix(r[2], n_cells)])
            .collect();
        let centres: Vec<[f64; 3]> = lat.iter().zip(&lon).map(|(&a, &b)| unit_rad(a, b)).collect();
        let vertices: Vec<[f64; 3]> = latv.iter().zip(&lonv).map(|(&a, &b)| unit_rad(a, b)).collect();
        let terrain = if file.variable("ter").is_some() {
            file.read_f64("ter")?
        } else {
            return Err(refusal(format!(
                "grid file {} has no ter; HGT is interpolated from it",
                path.display()
            )));
        };
        let tree = KdTree::build(centres.clone());
        let cap = bounding_cap(&centres, &vertices, &offsets, &vertices_on_cell);
        Ok(SourceMesh {
            n_cells,
            n_vert_levels,
            centres,
            vertices,
            cells_on_vertex,
            offsets,
            vertices_on_cell,
            cells_on_cell,
            terrain,
            tree,
            cap,
        })
    }

    fn range(&self, c: usize) -> std::ops::Range<usize> {
        self.offsets[c] as usize..self.offsets[c + 1] as usize
    }

    /// Whether a unit vector (with an angular margin, radians) can touch the mesh.
    pub fn may_touch(&self, p: [f64; 3], margin: f64) -> bool {
        match self.cap {
            None => true,
            Some((c, cos_r)) => {
                let r = libm::acos(cos_r.clamp(-1.0, 1.0)) + margin;
                r >= std::f64::consts::PI || dot(p, c) >= libm::cos(r)
            }
        }
    }
}

fn bounding_cap(
    centres: &[[f64; 3]],
    vertices: &[[f64; 3]],
    offsets: &[u32],
    vertices_on_cell: &[u32],
) -> Option<([f64; 3], f64)> {
    let mut s = [0.0f64; 3];
    for c in centres {
        s = [s[0] + c[0], s[1] + c[1], s[2] + c[2]];
    }
    let n = libm::sqrt(dot(s, s));
    if !(n > 1e-6 * centres.len() as f64) {
        return None;
    }
    let axis = [s[0] / n, s[1] / n, s[2] / n];
    let mut cos_min = 1.0f64;
    for c in 0..centres.len() {
        cos_min = cos_min.min(dot(centres[c], axis));
        for k in offsets[c] as usize..offsets[c + 1] as usize {
            let v = vertices_on_cell[k];
            if v != u32::MAX {
                cos_min = cos_min.min(dot(vertices[v as usize], axis));
            }
        }
    }
    // Half the sphere or more: no useful cap.
    if cos_min <= 0.0 {
        return None;
    }
    Some((axis, cos_min))
}

// ---------------------------------------------------------------------------
// Bilinear (barycentric) and nearest maps.
// ---------------------------------------------------------------------------

/// Per target point: the triangle, its weights, whether it is inside the
/// triangulated mesh, and the nearest cell.
#[derive(Debug, Clone)]
pub struct PointMap {
    pub tri: Vec<[u32; 3]>,
    pub w: Vec<[f64; 3]>,
    pub inside: Vec<bool>,
    pub near: Vec<u32>,
}

impl PointMap {
    pub fn len(&self) -> usize {
        self.near.len()
    }
    pub fn is_empty(&self) -> bool {
        self.near.is_empty()
    }
    pub fn inside_count(&self) -> usize {
        self.inside.iter().filter(|&&b| b).count()
    }
}

fn min3(w: &[f64; 3]) -> f64 {
    w[0].min(w[1]).min(w[2])
}

/// Locate every target in the mesh's vertex triangles.
pub fn locate(mesh: &SourceMesh, lat: &[f64], lon: &[f64]) -> PointMap {
    let rows: Vec<([u32; 3], [f64; 3], bool, u32)> = lat
        .par_iter()
        .zip(lon.par_iter())
        .map(|(&la, &lo)| {
            let p = unit(la, lo);
            if !mesh.may_touch(p, 1e-9) {
                return ([u32::MAX; 3], [1.0, 0.0, 0.0], false, u32::MAX);
            }
            let (near, _) = mesh.tree.nearest(p);
            let nearc = near as usize;
            let mut best: Option<([u32; 3], [f64; 3])> = None;
            let mut try_cell = |c: usize, best: &mut Option<([u32; 3], [f64; 3])>| {
                for k in mesh.range(c) {
                    let v = mesh.vertices_on_cell[k];
                    if v == u32::MAX {
                        continue;
                    }
                    let t = mesh.cells_on_vertex[v as usize];
                    if t.iter().any(|&x| x == u32::MAX) {
                        continue;
                    }
                    if let Some(w) = spherical_barycentric(
                        p,
                        mesh.centres[t[0] as usize],
                        mesh.centres[t[1] as usize],
                        mesh.centres[t[2] as usize],
                    ) {
                        if best.is_none_or(|(_, bw)| min3(&w) > min3(&bw)) {
                            *best = Some((t, w));
                        }
                    }
                }
            };
            try_cell(nearc, &mut best);
            if !best.is_some_and(|(_, w)| min3(&w) >= INSIDE_EPS) {
                for k in mesh.range(nearc) {
                    let c = mesh.cells_on_cell[k];
                    if c != u32::MAX {
                        try_cell(c as usize, &mut best);
                    }
                }
            }
            match best {
                Some((t, w)) if min3(&w) >= INSIDE_EPS => (t, w, true, near),
                _ => ([near; 3], [1.0, 0.0, 0.0], false, near),
            }
        })
        .collect();
    let mut map = PointMap {
        tri: Vec::with_capacity(rows.len()),
        w: Vec::with_capacity(rows.len()),
        inside: Vec::with_capacity(rows.len()),
        near: Vec::with_capacity(rows.len()),
    };
    for (t, w, inside, near) in rows {
        map.tri.push(t);
        map.w.push(w);
        map.inside.push(inside);
        map.near.push(near);
    }
    map
}

/// A source field: `values[cell * levels + level]`.
pub struct Field<'a> {
    pub values: &'a [f64],
    pub levels: usize,
}

impl PointMap {
    /// Barycentric value at point `p`, level `k` (f64; caller decides missing).
    #[inline]
    pub fn linear_at(&self, f: &Field<'_>, p: usize, k: usize) -> f64 {
        let t = self.tri[p];
        let w = self.w[p];
        w[0] * f.values[t[0] as usize * f.levels + k]
            + w[1] * f.values[t[1] as usize * f.levels + k]
            + w[2] * f.values[t[2] as usize * f.levels + k]
    }

    /// Level-major `(levels, points)` bilinear field with `MISSING` outside,
    /// after `post` (identity for most fields).
    pub fn linear(&self, f: &Field<'_>, post: impl Fn(f64) -> f64 + Sync) -> Vec<f32> {
        let n = self.len();
        let mut out = vec![MISSING; n * f.levels];
        out.par_chunks_mut(n).enumerate().for_each(|(k, row)| {
            for p in 0..n {
                if self.inside[p] {
                    row[p] = post(self.linear_at(f, p, k)) as f32;
                }
            }
        });
        out
    }

    /// Level-major nearest-cell field, `MISSING` where the bilinear map has none.
    pub fn nearest(&self, f: &Field<'_>) -> Vec<f32> {
        let n = self.len();
        let mut out = vec![MISSING; n * f.levels];
        out.par_chunks_mut(n).enumerate().for_each(|(k, row)| {
            for p in 0..n {
                if self.inside[p] {
                    row[p] = f.values[self.near[p] as usize * f.levels + k] as f32;
                }
            }
        });
        out
    }
}

// ---------------------------------------------------------------------------
// Conservative: exact overlap of great-circle polygons.
// ---------------------------------------------------------------------------

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn normalize(v: [f64; 3]) -> [f64; 3] {
    let n = libm::sqrt(dot(v, v));
    [v[0] / n, v[1] / n, v[2] / n]
}

/// Area of a spherical triangle on the unit sphere (Van Oosterom & Strackee).
fn tri_area(a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> f64 {
    let num = dot(a, cross(b, c)).abs();
    let den = 1.0 + dot(a, b) + dot(b, c) + dot(c, a);
    2.0 * libm::atan2(num, den)
}

/// Area of a convex great-circle polygon given on the sphere.
fn polygon_area(v: &[[f64; 3]]) -> f64 {
    let mut a = 0.0;
    for k in 1..v.len().saturating_sub(1) {
        a += tri_area(v[0], v[k], v[k + 1]);
    }
    a
}

/// Gnomonic chart about a unit vector `c`: great circles are straight lines.
struct Chart {
    c: [f64; 3],
    e1: [f64; 3],
    e2: [f64; 3],
}

impl Chart {
    fn at(c: [f64; 3]) -> Chart {
        let z = if c[2].abs() < 0.9 { [0.0, 0.0, 1.0] } else { [1.0, 0.0, 0.0] };
        let e1 = normalize(cross(z, c));
        let e2 = cross(c, e1);
        Chart { c, e1, e2 }
    }
    fn to_plane(&self, v: [f64; 3]) -> Option<[f64; 2]> {
        let d = dot(v, self.c);
        if d <= 1e-6 {
            return None;
        }
        Some([dot(v, self.e1) / d, dot(v, self.e2) / d])
    }
    fn to_sphere(&self, p: [f64; 2]) -> [f64; 3] {
        normalize([
            self.c[0] + p[0] * self.e1[0] + p[1] * self.e2[0],
            self.c[1] + p[0] * self.e1[1] + p[1] * self.e2[1],
            self.c[2] + p[0] * self.e1[2] + p[1] * self.e2[2],
        ])
    }
}

fn signed_area2(p: &[[f64; 2]]) -> f64 {
    let mut s = 0.0;
    for k in 0..p.len() {
        let a = p[k];
        let b = p[(k + 1) % p.len()];
        s += a[0] * b[1] - a[1] * b[0];
    }
    s
}

/// Sutherland-Hodgman: `subject` clipped by the convex counter-clockwise `clip`.
fn clip_convex(subject: &[[f64; 2]], clip: &[[f64; 2]]) -> Vec<[f64; 2]> {
    let mut out: Vec<[f64; 2]> = subject.to_vec();
    for k in 0..clip.len() {
        if out.is_empty() {
            break;
        }
        let a = clip[k];
        let b = clip[(k + 1) % clip.len()];
        let side = |p: [f64; 2]| (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]);
        let input = std::mem::take(&mut out);
        for m in 0..input.len() {
            let cur = input[m];
            let prev = input[(m + input.len() - 1) % input.len()];
            let sc = side(cur);
            let sp = side(prev);
            if sc >= 0.0 {
                if sp < 0.0 {
                    let t = sp / (sp - sc);
                    out.push([prev[0] + t * (cur[0] - prev[0]), prev[1] + t * (cur[1] - prev[1])]);
                }
                out.push(cur);
            } else if sp >= 0.0 {
                let t = sp / (sp - sc);
                out.push([prev[0] + t * (cur[0] - prev[0]), prev[1] + t * (cur[1] - prev[1])]);
            }
        }
    }
    out
}

/// Sparse destination-area weights: per target cell, the overlapping source
/// cells and `overlap / target cell area`.
#[derive(Debug, Clone)]
pub struct Conservative {
    pub offsets: Vec<u32>,
    pub cells: Vec<u32>,
    pub weights: Vec<f64>,
}

/// Build conservative weights for the `nx x ny` target cells whose corners
/// are the `(nx+1) x (ny+1)` lattice.
pub fn build_conservative(
    mesh: &SourceMesh,
    centre_lat: &[f64],
    centre_lon: &[f64],
    corner_lat: &[f64],
    corner_lon: &[f64],
    nx: usize,
    ny: usize,
) -> Conservative {
    let corner = |j: usize, i: usize| unit(corner_lat[j * (nx + 1) + i], corner_lon[j * (nx + 1) + i]);
    let rows: Vec<Vec<(u32, f64)>> = (0..nx * ny)
        .into_par_iter()
        .map(|slot| {
            let (j, i) = (slot / nx, slot % nx);
            let quad = [corner(j, i), corner(j, i + 1), corner(j + 1, i + 1), corner(j + 1, i)];
            let centre = unit(centre_lat[slot], centre_lon[slot]);
            let chart = Chart::at(centre);
            let Some(mut clip): Option<Vec<[f64; 2]>> = quad.iter().map(|&v| chart.to_plane(v)).collect() else {
                return Vec::new();
            };
            if signed_area2(&clip) < 0.0 {
                clip.reverse();
            }
            let reach = quad
                .iter()
                .map(|&v| libm::acos(dot(v, centre).clamp(-1.0, 1.0)))
                .fold(0.0f64, f64::max);
            if !mesh.may_touch(centre, reach + 1e-9) {
                return Vec::new();
            }
            let dst_area = polygon_area(&quad);
            if !(dst_area > 0.0) {
                return Vec::new();
            }
            let (start, _) = mesh.tree.nearest(centre);
            let mut acc: Vec<(u32, f64)> = Vec::new();
            let mut seen: Vec<u32> = vec![start];
            let mut queue: std::collections::VecDeque<(u32, u32)> = std::collections::VecDeque::new();
            queue.push_back((start, 0));
            while let Some((c, depth)) = queue.pop_front() {
                let cu = c as usize;
                let mut poly: Vec<[f64; 2]> = Vec::with_capacity(8);
                let mut ok = true;
                for k in mesh.range(cu) {
                    let v = mesh.vertices_on_cell[k];
                    if v == u32::MAX {
                        ok = false;
                        break;
                    }
                    match chart.to_plane(mesh.vertices[v as usize]) {
                        Some(q) => poly.push(q),
                        None => {
                            ok = false;
                            break;
                        }
                    }
                }
                let mut overlap = 0.0;
                if ok && poly.len() >= 3 {
                    if signed_area2(&poly) < 0.0 {
                        poly.reverse();
                    }
                    let cut = clip_convex(&poly, &clip);
                    if cut.len() >= 3 {
                        let sph: Vec<[f64; 3]> = cut.iter().map(|&q| chart.to_sphere(q)).collect();
                        overlap = polygon_area(&sph);
                    }
                }
                if overlap > 0.0 {
                    acc.push((c, overlap / dst_area));
                }
                // Expand from every overlapping cell, and unconditionally for
                // two rings around the start (a target centre off a limited-area
                // mesh can have a non-overlapping nearest cell).
                if overlap > 0.0 || depth < 2 {
                    for k in mesh.range(cu) {
                        let n = mesh.cells_on_cell[k];
                        if n != u32::MAX && !seen.contains(&n) {
                            seen.push(n);
                            queue.push_back((n, depth + 1));
                        }
                    }
                }
                if seen.len() > 100_000 {
                    break;
                }
            }
            acc.sort_by_key(|(c, _)| *c);
            acc
        })
        .collect();
    let mut out = Conservative {
        offsets: Vec::with_capacity(nx * ny + 1),
        cells: Vec::new(),
        weights: Vec::new(),
    };
    out.offsets.push(0);
    for r in rows {
        for (c, w) in r {
            out.cells.push(c);
            out.weights.push(w);
        }
        out.offsets.push(out.cells.len() as u32);
    }
    out
}

impl Conservative {
    pub fn apply(&self, f: &Field<'_>) -> Vec<f32> {
        let n = self.offsets.len() - 1;
        let mut out = vec![MISSING; n * f.levels];
        out.par_chunks_mut(n).enumerate().for_each(|(k, row)| {
            for (p, slot) in row.iter_mut().enumerate() {
                let r = self.offsets[p] as usize..self.offsets[p + 1] as usize;
                if r.is_empty() {
                    continue;
                }
                let mut s = 0.0;
                for q in r {
                    s += self.weights[q] * f.values[self.cells[q] as usize * f.levels + k];
                }
                *slot = s as f32;
            }
        });
        out
    }
}

// ---------------------------------------------------------------------------
// Staggered winds.
// ---------------------------------------------------------------------------

/// Bilinear position `(s, t)` of `p` in the quad `a(0,0) b(1,0) c(1,1) d(0,1)`
/// after radial projection: Newton on `X(s,t) = lambda p`.
fn quad_st(p: [f64; 3], a: [f64; 3], b: [f64; 3], c: [f64; 3], d: [f64; 3]) -> (f64, f64) {
    let (mut s, mut t, mut lam) = (0.5f64, 0.0f64, 1.0f64);
    for _ in 0..8 {
        let mut x = [0.0; 3];
        let mut xs = [0.0; 3];
        let mut xt = [0.0; 3];
        for k in 0..3 {
            x[k] = (1.0 - s) * (1.0 - t) * a[k] + s * (1.0 - t) * b[k] + s * t * c[k] + (1.0 - s) * t * d[k];
            xs[k] = (1.0 - t) * (b[k] - a[k]) + t * (c[k] - d[k]);
            xt[k] = (1.0 - s) * (d[k] - a[k]) + s * (c[k] - b[k]);
        }
        let f = [x[0] - lam * p[0], x[1] - lam * p[1], x[2] - lam * p[2]];
        let m = [xs, xt, [-p[0], -p[1], -p[2]]];
        // Solve [xs xt -p] * delta = -f by Cramer's rule (columns m[0..3]).
        let det = dot(m[0], cross(m[1], m[2]));
        if det.abs() < 1e-300 {
            break;
        }
        let nf = [-f[0], -f[1], -f[2]];
        s += dot(nf, cross(m[1], m[2])) / det;
        t += dot(m[0], cross(nf, m[2])) / det;
        lam += dot(m[0], cross(m[1], nf)) / det;
    }
    (s, t)
}

/// One staggered point: the four mass points of the grid cell holding it and
/// their bilinear weights; `None` when it lies outside the mass grid's cells.
type Corners = Option<([u32; 4], [f64; 4])>;

/// Weights from the mass grid to a staggered point between mass points `ia`
/// and `ib`, with candidate opposite edges `(ic1, id1)` and `(ic2, id2)` (the
/// cells on either side; `None` past the grid's edge).  The cell is the one on
/// the point's side of the great circle through `a` and `b`.
fn stag_corners(
    m: &[[f64; 3]],
    p: [f64; 3],
    ia: usize,
    ib: usize,
    side1: Option<(usize, usize)>,
    side2: Option<(usize, usize)>,
) -> Corners {
    let n = cross(m[ia], m[ib]);
    let sp = dot(p, n);
    let pick = |side: Option<(usize, usize)>| -> Option<(usize, usize)> {
        let (c, d) = side?;
        (dot(m[c], n) * sp >= 0.0).then_some((c, d))
    };
    let (ic, id) = pick(side1).or_else(|| pick(side2))?;
    let (s, t) = quad_st(p, m[ia], m[ib], m[ic], m[id]);
    Some((
        [ia as u32, ib as u32, ic as u32, id as u32],
        [(1.0 - s) * (1.0 - t), s * (1.0 - t), s * t, (1.0 - s) * t],
    ))
}

/// Rotated mass-point winds carried to the staggered points the way the
/// referee does it: a bilinear regrid from the mass grid's own cells, 0 at a
/// staggered point outside those cells (the outer U columns and V rows, and
/// the outer-column V points the grid's curvature leaves outside), missing
/// where the staggered point is off the MPAS mesh.
pub struct Stagger {
    /// `(ny, nx+1)` U points.
    pub u: Vec<Corners>,
    pub inside_u: Vec<bool>,
    /// `(ny+1, nx)` V points.
    pub v: Vec<Corners>,
    pub inside_v: Vec<bool>,
}

impl Stagger {
    pub fn build(mesh: &SourceMesh, grid: &TargetGrid, mlat: &[f64], mlon: &[f64]) -> Stagger {
        let (nx, ny) = (grid.nx, grid.ny);
        let (ulat, ulon) = grid.u_stag();
        let (vlat, vlon) = grid.v_stag();
        let um = locate(mesh, &ulat, &ulon);
        let vm = locate(mesh, &vlat, &vlon);
        let m: Vec<[f64; 3]> = mlat.par_iter().zip(mlon.par_iter()).map(|(&a, &b)| unit(a, b)).collect();
        let at = |j: usize, i: usize| j * nx + i;
        let u: Vec<Corners> = (0..ny * (nx + 1))
            .into_par_iter()
            .map(|s| {
                let (j, i) = (s / (nx + 1), s % (nx + 1));
                if i == 0 || i == nx {
                    return None;
                }
                let up = (j + 1 < ny).then(|| (at(j + 1, i), at(j + 1, i - 1)));
                let down = (j > 0).then(|| (at(j - 1, i), at(j - 1, i - 1)));
                stag_corners(&m, unit(ulat[s], ulon[s]), at(j, i - 1), at(j, i), up, down)
            })
            .collect();
        let v: Vec<Corners> = (0..(ny + 1) * nx)
            .into_par_iter()
            .map(|s| {
                let (j, i) = (s / nx, s % nx);
                if j == 0 || j == ny {
                    return None;
                }
                let right = (i + 1 < nx).then(|| (at(j, i + 1), at(j - 1, i + 1)));
                let left = (i > 0).then(|| (at(j, i - 1), at(j - 1, i - 1)));
                stag_corners(&m, unit(vlat[s], vlon[s]), at(j - 1, i), at(j, i), right, left)
            })
            .collect();
        Stagger {
            u,
            inside_u: um.inside,
            v,
            inside_v: vm.inside,
        }
    }

    fn apply(corners: &[Corners], inside: &[bool], field: &[f64]) -> Vec<f32> {
        corners
            .par_iter()
            .zip(inside.par_iter())
            .map(|(c, &ins)| {
                if !ins {
                    return MISSING;
                }
                match c {
                    None => 0.0,
                    Some((k, w)) => {
                        (w[0] * field[k[0] as usize]
                            + w[1] * field[k[1] as usize]
                            + w[2] * field[k[2] as usize]
                            + w[3] * field[k[3] as usize]) as f32
                    }
                }
            })
            .collect()
    }

    /// `U(ny, nx+1)` from a rotated mass-point field (0 where off the mesh).
    pub fn u(&self, um: &[f64]) -> Vec<f32> {
        Self::apply(&self.u, &self.inside_u, um)
    }

    /// `V(ny+1, nx)`.
    pub fn v(&self, vm: &[f64]) -> Vec<f32> {
        Self::apply(&self.v, &self.inside_v, vm)
    }
}

// ---------------------------------------------------------------------------
// The whole run.
// ---------------------------------------------------------------------------

/// One stage's wall time and how many threads it ran on.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Stage {
    pub name: String,
    pub seconds: f64,
    pub parallelism: String,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct Receipt {
    pub schema: &'static str,
    pub method: &'static str,
    pub output: String,
    pub grid: String,
    pub source_cells: usize,
    pub target_points: usize,
    pub inside_points: usize,
    pub variables_written: Vec<String>,
    pub absent_inputs: Vec<String>,
    pub threads: usize,
    pub stages: Vec<Stage>,
}

struct Timer {
    stages: Vec<Stage>,
    t: std::time::Instant,
    threads: usize,
}

impl Timer {
    fn lap(&mut self, name: &str, parallel: bool) {
        let now = std::time::Instant::now();
        self.stages.push(Stage {
            name: name.into(),
            seconds: (now - self.t).as_secs_f64(),
            parallelism: if parallel {
                format!("{} threads", self.threads)
            } else {
                "1 thread (file layout)".into()
            },
        });
        self.t = now;
    }
}

/// A variable read from an MPAS file: values (cell-major) and its levels.
struct Input {
    values: Vec<f64>,
    levels: usize,
    level_dim: Option<String>,
    units: String,
    long_name: String,
}

fn read_input(file: &netcrust::File, name: &str, n_cells: usize) -> MpasResult<Option<Input>> {
    let Some(var) = file.variable(name) else {
        return Ok(None);
    };
    let dims: Vec<String> = var.dimensions().iter().map(|d| d.name().to_string()).collect();
    let values = file.read_f64_first_record_or_all(name)?;
    if values.len() % n_cells != 0 || values.is_empty() {
        return Err(refusal(format!(
            "{name} holds {} values, not a whole number of rows of {n_cells} cells",
            values.len()
        )));
    }
    let levels = values.len() / n_cells;
    let level_dim = dims.last().filter(|d| d.as_str() != "nCells").cloned();
    let text = |a: &str| {
        var.attribute(a)
            .and_then(|x| x.as_string())
            .map(|s| s.to_string())
            .unwrap_or_default()
    };
    Ok(Some(Input {
        values,
        levels,
        level_dim,
        units: text("units"),
        long_name: text("long_name"),
    }))
}

fn text_attr(file: &netcrust::File, name: &str) -> Option<String> {
    file.attribute(name).and_then(|a| a.as_string().map(|s| s.trim().to_string()))
}

fn f64_attr(file: &netcrust::File, name: &str) -> Option<f64> {
    file.attribute(name).and_then(|a| {
        a.as_f64()
            .or_else(|| a.as_string().and_then(|s| s.trim().parse::<f64>().ok()))
    })
}

/// `YYYY-MM-DD_hh:mm:ss` -> seconds since 1970 (proleptic Gregorian).
fn parse_stamp(text: &str) -> Option<i64> {
    let t = text.trim();
    let b = t.as_bytes();
    if b.len() < 19 {
        return None;
    }
    let n = |r: std::ops::Range<usize>| t.get(r)?.parse::<i64>().ok();
    let (y, mo, d, h, mi, s) = (n(0..4)?, n(5..7)?, n(8..10)?, n(11..13)?, n(14..16)?, n(17..19)?);
    let (yy, mm) = if mo <= 2 { (y - 1, mo + 9) } else { (y, mo - 3) };
    let era = yy.div_euclid(400);
    let yoe = yy - era * 400;
    let doy = (153 * mm + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146_097 + doe - 719_468;
    Some(days * 86_400 + h * 3600 + mi * 60 + s)
}

/// WRF physics option numbers the referee writes in the header for MPAS
/// scheme names.  Measured: sf_ruc -> 3, mp_tempo -> 0, off -> 0.  The other
/// rows are WRF's own numbering for the same scheme.
fn scheme_number(kind: &str, name: &str) -> i32 {
    match (kind, name) {
        ("lsm", "sf_noah") => 2,
        ("lsm", "sf_ruc") => 3,
        ("lsm", "sf_noahmp") => 4,
        ("mp", "mp_kessler") => 1,
        ("mp", "mp_wsm6") => 6,
        ("mp", "mp_thompson") | ("mp", "mp_thompson_aerosols") => 8,
        ("cu", "cu_kain_fritsch") => 1,
        ("cu", "cu_grell_freitas") => 3,
        ("cu", "cu_tiedtke") => 6,
        ("cu", "cu_ntiedtke") => 16,
        _ => 0,
    }
}

fn var_attrs(description: &str, units: &str, order: &str, stagger: &str, fill: bool, coords: &str) -> Vec<NcAttr> {
    let mut a = vec![
        NcAttr::text("MemoryOrder", order),
        NcAttr::text("coordinates", coords),
        NcAttr::text("units", units),
        NcAttr::text("description", description),
        NcAttr::text("stagger", stagger),
        NcAttr::int("FieldType", 104),
    ];
    if fill {
        a.push(NcAttr::float("_FillValue", MISSING));
    }
    a
}

fn geo_attrs(description: &str, units: &str, coords: &str, stagger: &str) -> Vec<NcAttr> {
    vec![
        NcAttr::text("description", description),
        NcAttr::text("units", units),
        NcAttr::text("MemoryOrder", "XY"),
        NcAttr::text("coordinates", coords),
        NcAttr::text("stagger", stagger),
        NcAttr::int("FieldType", 104),
    ]
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Shape {
    Mass2,
    U2,
    V2,
    Mass3,
    Stag3,
    U3,
    V3,
    Soil,
    Scalar,
}

/// One planned output variable.
struct Planned {
    name: String,
    shape: Shape,
    attrs: Vec<NcAttr>,
    job: Job,
}

enum Job {
    /// From an input variable with a remap class and a post-op.
    Remap { file: Src, mpas: String, class: Class, post: Post },
    /// Rotated 2-D wind (U10 from u10/v10 or V10).
    Wind2 { file: Src, u: String, v: String, which: char },
    /// Rotated, staggered 3-D wind.
    Wind3 { u: String, v: String, which: char },
    /// Interpolated terrain.
    Terrain,
    /// All zeros (PH, P, MU).
    Zero,
    /// A value we already hold (lat/lon/map factors/rotation), level-major.
    Fixed(Vec<f32>),
    /// `PB` without `pressure` in the list: missing everywhere.
    Missing,
    /// `P_TOP` from the PB just written.
    PTop,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Src {
    Diag,
    Hist,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Post {
    None,
    ThetaToT,
    /// `Z_C`: 9.81 * zgrid, 0 off the mesh, top level never written.
    ZC,
}

/// Run one MPASSIT-equivalent remap.  `lists_dir` holds the four variable
/// lists (MPASSIT reads them from its run directory).
pub fn run(nl: &Namelist, lists_dir: &Path, allow_absent: bool, format: NcFormat) -> MpasResult<Receipt> {
    nl.check_supported()?;
    let threads = rayon::current_num_threads();
    let mut timer = Timer {
        stages: Vec::new(),
        t: std::time::Instant::now(),
        threads,
    };
    let lists = VariableLists::read(lists_dir)?;
    let mesh = SourceMesh::read(&nl.grid_file)?;
    let hist = netcrust::File::open(&nl.hist_file)?;
    let diag = netcrust::File::open(&nl.diag_file)?;
    timer.lap("read mesh and open inputs", false);

    let grid = TargetGrid::from_namelist(nl);
    let (nx, ny) = (grid.nx, grid.ny);
    let (mlat, mlon) = grid.mass();
    let (ulat, ulon) = grid.u_stag();
    let (vlat, vlon) = grid.v_stag();
    let (sina, cosa) = rotation(&mlat, &mlon, nx, ny);
    timer.lap("target grid and rotation", true);

    let map = locate(&mesh, &mlat, &mlon);
    timer.lap("locate mass points", true);

    // ---- plan --------------------------------------------------------------
    let nz = mesh.n_vert_levels;
    let mut absent: Vec<String> = Vec::new();
    let mut plan: Vec<Planned> = Vec::new();
    let f32s = |v: &[f64]| -> Vec<f32> { v.iter().map(|&x| x as f32).collect() };
    let mf = |lat: &[f64]| -> Vec<f32> { lat.iter().map(|&a| grid.map_factor(a) as f32).collect() };
    for (name, shape, attrs, data) in [
        ("XLONG", Shape::Mass2, geo_attrs("LONGITUDE, WEST IS NEGATIVE", "degree_east", "XLONG XLAT", ""), f32s(&mlon)),
        ("XLONG_U", Shape::U2, geo_attrs("LONGITUDE, WEST IS NEGATIVE", "degree_east", "XLONG_U XLAT_U", "X"), f32s(&ulon)),
        ("XLONG_V", Shape::V2, geo_attrs("LONGITUDE, WEST IS NEGATIVE", "degree_east", "XLONG_V XLAT_V", "Y"), f32s(&vlon)),
        ("XLAT", Shape::Mass2, geo_attrs("LATITUDE, SOUTH IS NEGATIVE", "degree_north", "XLONG XLAT", ""), f32s(&mlat)),
        ("XLAT_U", Shape::U2, geo_attrs("LATITUDE, SOUTH IS NEGATIVE", "degree_north", "XLONG_U XLAT_U", "X"), f32s(&ulat)),
        ("XLAT_V", Shape::V2, geo_attrs("LATITUDE, SOUTH IS NEGATIVE", "degree_north", "XLONG_V XLAT_V", "Y"), f32s(&vlat)),
        // The referee labels its map factors with latitude's description.
        ("MAPFAC_M", Shape::Mass2, geo_attrs("LATITUDE, SOUTH IS NEGATIVE", "degree_north", "XLONG XLAT", ""), mf(&mlat)),
        ("MAPFAC_U", Shape::U2, geo_attrs("LATITUDE, SOUTH IS NEGATIVE", "degree_north", "XLONG_U XLAT_U", "X"), mf(&ulat)),
        ("MAPFAC_V", Shape::V2, geo_attrs("LATITUDE, SOUTH IS NEGATIVE", "degree_north", "XLONG_V XLAT_V", "Y"), mf(&vlat)),
        ("SINALPHA", Shape::Mass2, geo_attrs("SINE OF GRID ROTATION ANGLE ALPHA", "", "XLONG XLAT", ""), f32s(&sina)),
        ("COSALPHA", Shape::Mass2, geo_attrs("COSINE OF GRID ROTATION ANGLE ALPHA", "", "XLONG XLAT", ""), f32s(&cosa)),
    ] {
        plan.push(Planned { name: name.into(), shape, attrs, job: Job::Fixed(data) });
    }
    let has_hist = |n: &str| hist.variable(n).is_some();
    let has_diag = |n: &str| diag.variable(n).is_some();
    let zgrid_listed = nl.interp_hist && lists.hist_3d.iter().any(|e| e.mpas == "zgrid") && has_hist("zgrid");
    plan.push(Planned {
        name: "Z_C".into(),
        shape: Shape::Stag3,
        attrs: {
            let mut a = var_attrs("Layer center height above mean sea level", "m AMSL", "XYZ", "", false, "XLAT XLONG Z_C");
            a.retain(|x| x.name != "_FillValue");
            a
        },
        job: if zgrid_listed {
            Job::Remap { file: Src::Hist, mpas: "zgrid".into(), class: Class::Bilinear, post: Post::ZC }
        } else {
            Job::Zero
        },
    });
    plan.push(Planned {
        name: "HGT".into(),
        shape: Shape::Mass2,
        attrs: var_attrs("TERRAIN HEIGHT", "m AMSL", "XY", "", true, "XLAT XLONG"),
        job: Job::Terrain,
    });

    let attrs_of = |file: &netcrust::File, mpas: &str, order: &str, stagger: &str| -> Vec<NcAttr> {
        let v = file.variable(mpas);
        let t = |a: &str| {
            v.as_ref()
                .and_then(|v| v.attribute(a).and_then(|x| x.as_string().map(|s| s.to_string())))
                .unwrap_or_default()
        };
        var_attrs(&t("long_name"), &t("units"), order, stagger, true, "XLONG XLAT XTIME")
    };

    if nl.interp_diag {
        for e in &lists.diag {
            if !has_diag(&e.mpas) {
                absent.push(format!("diag:{}", e.mpas));
                continue;
            }
            let dims = diag.variable(&e.mpas).map(|v| v.dimensions().len()).unwrap_or(0);
            let shape = if dims >= 3 { Shape::Mass3 } else { Shape::Mass2 };
            let order = if shape == Shape::Mass3 { "XYZ" } else { "XY" };
            let job = match e.mpas.as_str() {
                "u10" => Job::Wind2 { file: Src::Diag, u: "u10".into(), v: "v10".into(), which: 'u' },
                "v10" => Job::Wind2 { file: Src::Diag, u: "u10".into(), v: "v10".into(), which: 'v' },
                _ => Job::Remap { file: Src::Diag, mpas: e.mpas.clone(), class: Class::Bilinear, post: Post::None },
            };
            plan.push(Planned { name: e.wrf.clone(), shape, attrs: attrs_of(&diag, &e.mpas, order, ""), job });
        }
    }
    if nl.interp_hist {
        // Conservative, then bilinear, then nearest: the referee's order.
        for want in [Class::Conservative, Class::Bilinear, Class::Nearest] {
            for e in lists.hist_2d.iter().filter(|e| class_of_hist_2d(&e.mpas) == want) {
                if !has_hist(&e.mpas) {
                    absent.push(format!("hist:{}", e.mpas));
                    continue;
                }
                plan.push(Planned {
                    name: e.wrf.clone(),
                    shape: Shape::Mass2,
                    attrs: attrs_of(&hist, &e.mpas, "XY", ""),
                    job: Job::Remap { file: Src::Hist, mpas: e.mpas.clone(), class: want, post: Post::None },
                });
            }
        }
        for e in &lists.hist_soil {
            if !has_hist(&e.mpas) {
                absent.push(format!("hist:{}", e.mpas));
                continue;
            }
            plan.push(Planned {
                name: e.wrf.clone(),
                shape: Shape::Soil,
                attrs: attrs_of(&hist, &e.mpas, "XYZ", ""),
                job: Job::Remap { file: Src::Hist, mpas: e.mpas.clone(), class: Class::Nearest, post: Post::None },
            });
        }
        // Mass-level 3-D fields (with their WRF companions), then the
        // interface-level ones, then PH/P/PB.
        let mut stag: Vec<Planned> = Vec::new();
        let mut pressure_seen = false;
        for e in &lists.hist_3d {
            if !has_hist(&e.mpas) {
                absent.push(format!("hist:{}", e.mpas));
                continue;
            }
            match e.mpas.as_str() {
                "uReconstructZonal" | "uReconstructMeridional" => {
                    let which = if e.mpas == "uReconstructZonal" { 'u' } else { 'v' };
                    plan.push(Planned {
                        name: e.wrf.clone(),
                        shape: if which == 'u' { Shape::U3 } else { Shape::V3 },
                        attrs: attrs_of(&hist, &e.mpas, "XYZ", if which == 'u' { "X" } else { "Y" }),
                        job: Job::Wind3 { u: "uReconstructZonal".into(), v: "uReconstructMeridional".into(), which },
                    });
                }
                "zgrid" | "w" => {
                    stag.push(Planned {
                        name: e.wrf.clone(),
                        shape: Shape::Stag3,
                        attrs: if e.mpas == "zgrid" {
                            var_attrs("Base Geopotential Height", "gpm", "XYZ", "Z", true, "XLONG XLAT XTIME")
                        } else {
                            attrs_of(&hist, &e.mpas, "XYZ", "Z")
                        },
                        job: Job::Remap { file: Src::Hist, mpas: e.mpas.clone(), class: Class::Bilinear, post: Post::None },
                    });
                }
                other => {
                    let post = if other == "theta" { Post::ThetaToT } else { Post::None };
                    let attrs = if other == "theta" {
                        var_attrs("Potential temperature", "K", "XYZ", "", true, "XLONG XLAT XTIME")
                    } else {
                        attrs_of(&hist, other, "XYZ", "")
                    };
                    plan.push(Planned {
                        name: e.wrf.clone(),
                        shape: Shape::Mass3,
                        attrs,
                        job: Job::Remap { file: Src::Hist, mpas: other.into(), class: Class::Bilinear, post },
                    });
                    if other == "rho" {
                        plan.push(Planned {
                            name: "MU".into(),
                            shape: Shape::Mass3,
                            attrs: var_attrs("perturbation dry air mass in column", "Pa", "XYZ", "", true, "XLONG XLAT XTIME"),
                            job: Job::Zero,
                        });
                    }
                    if other == "pressure" {
                        pressure_seen = true;
                    }
                }
            }
        }
        plan.extend(stag);
        plan.push(Planned {
            name: "PH".into(),
            shape: Shape::Stag3,
            attrs: var_attrs("perturbation geopotential", "m2 s-2", "XYZ", "Z", true, "XLONG XLAT XTIME"),
            job: Job::Zero,
        });
        plan.push(Planned {
            name: "P".into(),
            shape: Shape::Mass3,
            attrs: var_attrs("perturbation pressure", "Pa", "XYZ", "", true, "XLONG XLAT XTIME"),
            job: Job::Zero,
        });
        plan.push(Planned {
            name: "PB".into(),
            shape: Shape::Mass3,
            attrs: var_attrs("BASE STATE PRESSURE", "Pa", "XYZ", "", true, "XLONG XLAT XTIME"),
            job: if pressure_seen {
                Job::Remap { file: Src::Hist, mpas: "pressure".into(), class: Class::Bilinear, post: Post::None }
            } else {
                Job::Missing
            },
        });
        if pressure_seen {
            // The referee writes P_TOP whenever pressure is listed.  Its formula
            // was not identified (x1.2562: 1817.92 Pa; on a limited-area mesh it
            // writes -10045.9, a value built from missing points); this writes
            // the lowest top-interface pressure extrapolated from the two top
            // PB levels over the points on the mesh.
            plan.push(Planned {
                name: "P_TOP".into(),
                shape: Shape::Scalar,
                attrs: vec![
                    NcAttr::text("MemoryOrder", "0"),
                    NcAttr::text("units", "Pa"),
                    NcAttr::text("description", "PRESSURE TOP OF THE MODEL"),
                    NcAttr::text("stagger", ""),
                    NcAttr::int("FieldType", 104),
                    NcAttr::float("_FillValue", MISSING),
                ],
                job: Job::PTop,
            });
        }
    }
    if !absent.is_empty() && !allow_absent {
        return Err(refusal(format!(
            "the variable lists name {} field(s) the input does not carry ({}); UPP reads every \
             listed field from this file and stops at the first one it cannot find, so the file \
             would fail downstream. Pass --allow-absent to write the file without them (they are \
             listed in the receipt)",
            absent.len(),
            absent.join(", ")
        )));
    }

    // ---- define the file ---------------------------------------------------
    let soil_levels = lists
        .hist_soil
        .iter()
        .find_map(|e| hist.variable(&e.mpas).and_then(|v| v.shape().last().copied()))
        .or_else(|| hist.dimension("nSoilLevels").map(|d| d.len()))
        .unwrap_or(4);
    let dims = vec![
        NcDim::record("Time"),
        NcDim::fixed("west_east", nx),
        NcDim::fixed("west_east_stag", nx + 1),
        NcDim::fixed("south_north", ny),
        NcDim::fixed("south_north_stag", ny + 1),
        NcDim::fixed("bottom_top", nz),
        NcDim::fixed("bottom_top_stag", nz + 1),
        NcDim::fixed("soil_layers_stag", soil_levels),
        NcDim::fixed("StrLen", 19),
    ];
    let (d_t, d_we, d_wes, d_sn, d_sns, d_bt, d_bts, d_soil, d_str) = (0, 1, 2, 3, 4, 5, 6, 7, 8);
    let dimids = |s: Shape| -> Vec<usize> {
        match s {
            Shape::Mass2 => vec![d_t, d_sn, d_we],
            Shape::U2 => vec![d_t, d_sn, d_wes],
            Shape::V2 => vec![d_t, d_sns, d_we],
            Shape::Mass3 => vec![d_t, d_bt, d_sn, d_we],
            Shape::Stag3 => vec![d_t, d_bts, d_sn, d_we],
            Shape::U3 => vec![d_t, d_bt, d_sn, d_wes],
            Shape::V3 => vec![d_t, d_bt, d_sns, d_we],
            Shape::Soil => vec![d_t, d_soil, d_sn, d_we],
            Shape::Scalar => vec![d_t],
        }
    };
    let start = text_attr(&hist, "config_start_time").unwrap_or_default();
    let valid = hist
        .read_strings("xtime")
        .ok()
        .and_then(|v| v.into_iter().next())
        .map(|s| s.trim().trim_end_matches('\0').to_string())
        .unwrap_or_default();
    let dt = f64_attr(&hist, "config_dt").unwrap_or(0.0);
    let xtime_min = match (parse_stamp(&start), parse_stamp(&valid)) {
        (Some(s), Some(v)) => (s - v) as f64 / 60.0,
        _ => 0.0,
    };
    let itimestep = if dt > 0.0 { (xtime_min * 60.0 / dt).round() as i32 } else { 0 };
    let start19: String = start.chars().take(19).collect();
    let mut vars: Vec<NcVarDef> = Vec::new();
    for p in plan.iter().take(12) {
        vars.push(NcVarDef::new(p.name.clone(), NcType::Float, dimids(p.shape)).with_attrs(p.attrs.clone()));
    }
    vars.push(NcVarDef::new("ZS", NcType::Float, vec![d_t, d_soil]).with_attrs(vec![
        NcAttr::text("description", "DEPTHS OF CENTERS OF SOIL LAYERS"),
        NcAttr::text("units", "m"),
        NcAttr::text("MemoryOrder", "Z"),
        NcAttr::text("coordinates", "ZS XTIME"),
        NcAttr::text("stagger", ""),
        NcAttr::int("FieldType", 104),
    ]));
    vars.push(NcVarDef::new("DZS", NcType::Float, vec![d_t, d_soil]).with_attrs(vec![
        NcAttr::text("description", "SOIL LAYER THICKNESS"),
        NcAttr::text("units", "m"),
        NcAttr::text("MemoryOrder", "Z"),
        NcAttr::text("coordinates", "ZS XTIME"),
        NcAttr::text("stagger", ""),
        NcAttr::int("FieldType", 104),
    ]));
    for p in plan.iter().skip(12).take(1) {
        vars.push(NcVarDef::new(p.name.clone(), NcType::Float, dimids(p.shape)).with_attrs(p.attrs.clone()));
    }
    vars.push(NcVarDef::new("Times", NcType::Char, vec![d_t, d_str]).with_attrs(vec![
        NcAttr::text("description", "Times"),
        NcAttr::text("units", "m"),
        NcAttr::text("coordinates", "Time"),
        NcAttr::text("stagger", ""),
        NcAttr::int("FieldType", 104),
    ]));
    vars.push(NcVarDef::new("ITIMESTEP", NcType::Int, vec![d_t]).with_attrs(vec![
        NcAttr::text("description", ""),
        NcAttr::text("units", ""),
        NcAttr::text("stagger", ""),
        NcAttr::int("FieldType", 106),
        NcAttr::text("MemoryOrder", "O"),
    ]));
    let since = format!("minutes since {start19}");
    vars.push(NcVarDef::new("XTIME", NcType::Float, vec![d_t]).with_attrs(vec![
        NcAttr::text("description", since.clone()),
        NcAttr::text("units", since),
        NcAttr::text("stagger", ""),
    ]));
    for p in plan.iter().skip(13) {
        vars.push(NcVarDef::new(p.name.clone(), NcType::Float, dimids(p.shape)).with_attrs(p.attrs.clone()));
    }
    let (cen_lat, cen_lon) = grid.centre();
    let scheme = |k: &str, attr: &str| scheme_number(k, &text_attr(&hist, attr).unwrap_or_default());
    let gattrs = vec![
        NcAttr::int("WEST-EAST_GRID_DIMENSION", grid.e_we as i32),
        NcAttr::int("SOUTH-NORTH_GRID_DIMENSION", grid.e_sn as i32),
        NcAttr::int("BOTTOM-TOP_GRID_DIMENSION", (nz + 1) as i32),
        NcAttr::text("SIMULATION_START_DATE", start19.clone()),
        NcAttr::text("START_DATE", start19.clone()),
        NcAttr::doubles("DT", vec![dt]),
        NcAttr::int("SF_SURFACE_PHYSICS", scheme("lsm", "config_lsm_scheme")),
        NcAttr::int("MP_PHYSICS", scheme("mp", "config_microp_scheme")),
        NcAttr::int("CU_PHYSICS", scheme("cu", "config_convection_scheme")),
        NcAttr::doubles("CEN_LAT", vec![cen_lat]),
        NcAttr::doubles("CEN_LON", vec![cen_lon]),
        NcAttr::doubles("TRUELAT1", vec![nl.truelat1]),
        NcAttr::doubles("TRUELAT2", vec![nl.truelat2]),
        NcAttr::doubles("MOAD_CEN_LAT", vec![cen_lat]),
        NcAttr::doubles("STAND_LON", vec![nl.stand_lon]),
        NcAttr::doubles("POLE_LAT", vec![90.0]),
        NcAttr::doubles("POLE_LON", vec![0.0]),
        NcAttr::doubles("POL_ELAT", vec![90.0]),
        NcAttr::int("MAP_PROJ", 1),
        NcAttr::text("MAP_PROJ_CHAR", "Lambert Conformal"),
        NcAttr::doubles("DX", vec![nl.dx]),
        NcAttr::doubles("DY", vec![nl.dy]),
        NcAttr::doubles("PREC_ACC_DT", vec![f64_attr(&hist, "config_prec_acc_dt").unwrap_or(60.0)]),
        NcAttr::int("I_PARENT_START", 1),
        NcAttr::int("J_PARENT_START", 1),
        NcAttr::int("WEST-EAST_PATCH_START_UNSTAG", 1),
        NcAttr::int("WEST-EAST_PATCH_START_STAG", 1),
        NcAttr::int("SOUTH-NORTH_PATCH_START_UNSTAG", 1),
        NcAttr::int("SOUTH-NORTH_PATCH_START_STAG", 1),
        NcAttr::int("BOTTOM-TOP_PATCH_START_UNSTAG", 1),
        NcAttr::int("BOTTOM-TOP_PATCH_START_STAG", 1),
        NcAttr::int("WEST-EAST_PATCH_END_UNSTAG", nx as i32),
        NcAttr::int("WEST-EAST_PATCH_END_STAG", (nx + 1) as i32),
        NcAttr::int("SOUTH-NORTH_PATCH_END_UNSTAG", ny as i32),
        NcAttr::int("SOUTH-NORTH_PATCH_END_STAG", (ny + 1) as i32),
        NcAttr::int("BOTTOM-TOP_PATCH_END_UNSTAG", nz as i32),
        NcAttr::int("BOTTOM-TOP_PATCH_END_STAG", (nz + 1) as i32),
    ];
    if let Some(parent) = nl.output_file.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let tmp = nl.output_file.with_extension("nc.partial");
    let mut writer = NcClassicWriter::create(&tmp, format, dims, gattrs, vars, 1)
        .map_err(|e| refusal(format!("create {}: {e}", tmp.display())))?;
    let werr = |e: rw_store::error::RwStoreError| refusal(format!("write {}: {e}", tmp.display()));
    timer.lap("plan and lay out the file", false);

    // ---- fixed pieces ------------------------------------------------------
    writer.put_record_text("Times", 0, &valid.chars().take(19).collect::<String>()).map_err(werr)?;
    writer.put_record("ITIMESTEP", 0, NcData::Ints(&[itimestep])).map_err(werr)?;
    writer.put_record("XTIME", 0, NcData::Floats(&[xtime_min as f32])).map_err(werr)?;
    for (name, mpas) in [("ZS", "zs"), ("DZS", "dzs")] {
        let v: Vec<f32> = match hist.variable(mpas) {
            Some(_) => hist.read_f64_first_record_or_all(mpas)?.iter().take(soil_levels).map(|&x| x as f32).collect(),
            None => vec![0.0; soil_levels],
        };
        let mut v = v;
        v.resize(soil_levels, 0.0);
        writer.put_record(name, 0, NcData::Floats(&v)).map_err(werr)?;
    }
    let terrain = Field { values: &mesh.terrain, levels: 1 };
    let hgt = map.linear(&terrain, |x| x);
    writer.put_record("HGT", 0, NcData::Floats(&hgt)).map_err(werr)?;
    let mut written: Vec<String> = vec!["HGT".into()];
    timer.lap("write header fields and HGT", true);

    // ---- the heavier maps, built only if a variable needs them -----------
    let need_cons = plan.iter().any(|p| matches!(p.job, Job::Remap { class: Class::Conservative, .. }));
    let cons = if need_cons {
        let (clat, clon) = grid.corners();
        let c = build_conservative(&mesh, &mlat, &mlon, &clat, &clon, nx, ny);
        timer.lap("conservative weights (exact polygon overlap)", true);
        Some(c)
    } else {
        None
    };
    let need_stag = plan.iter().any(|p| matches!(p.job, Job::Wind3 { .. }));
    let stagger = if need_stag {
        let s = Stagger::build(&mesh, &grid, &mlat, &mlon);
        timer.lap("locate staggered points", true);
        Some(s)
    } else {
        None
    };

    // ---- every variable ----------------------------------------------------
    let mut wind3_cache: BTreeMap<char, Vec<f32>> = BTreeMap::new();
    let mut p_top: f32 = MISSING;
    let points = nx * ny;
    for p in &plan {
        let levels_of = |s: Shape| match s {
            Shape::Mass3 | Shape::U3 | Shape::V3 => nz,
            Shape::Stag3 => nz + 1,
            Shape::Soil => soil_levels,
            _ => 1,
        };
        let size_of = |s: Shape| match s {
            Shape::U2 | Shape::U3 => ny * (nx + 1),
            Shape::V2 | Shape::V3 => (ny + 1) * nx,
            Shape::Scalar => 1,
            _ => points,
        } * levels_of(s);
        let data: Vec<f32> = match &p.job {
            Job::Fixed(v) => v.clone(),
            Job::Terrain => continue,
            Job::Zero => vec![0.0; size_of(p.shape)],
            Job::Missing => vec![MISSING; size_of(p.shape)],
            Job::PTop => vec![p_top],
            Job::Remap { file, mpas, class, post } => {
                let f = if *file == Src::Diag { &diag } else { &hist };
                let input = read_input(f, mpas, mesh.n_cells)?
                    .ok_or_else(|| refusal(format!("{mpas} vanished between planning and reading")))?;
                let want = levels_of(p.shape);
                if input.levels != want {
                    return Err(refusal(format!(
                        "{mpas} has {} level(s) ({:?}) but {} is declared with {want}",
                        input.levels, input.level_dim, p.name
                    )));
                }
                let field = Field { values: &input.values, levels: input.levels };
                match (class, post) {
                    (Class::Bilinear, Post::None) => map.linear(&field, |x| x),
                    (Class::Bilinear, Post::ThetaToT) => map.linear(&field, |x| x - THETA_OFFSET),
                    (Class::Bilinear, Post::ZC) => {
                        let mut v = map.linear(&field, |x| x * ZC_GRAVITY);
                        let n = points;
                        for k in 0..want {
                            let row = &mut v[k * n..(k + 1) * n];
                            for (q, x) in row.iter_mut().enumerate() {
                                if k == want - 1 {
                                    *x = NC_FILL_FLOAT;
                                } else if !map.inside[q] {
                                    *x = 0.0;
                                }
                            }
                        }
                        v
                    }
                    (Class::Nearest, _) => map.nearest(&field),
                    (Class::Conservative, _) => cons.as_ref().expect("built above").apply(&field),
                }
            }
            Job::Wind2 { file, u, v, which } => {
                let f = if *file == Src::Diag { &diag } else { &hist };
                let ui = read_input(f, u, mesh.n_cells)?.ok_or_else(|| refusal(format!("{u} absent")))?;
                let vi = read_input(f, v, mesh.n_cells)?.ok_or_else(|| refusal(format!("{v} absent")))?;
                let (uf, vf) = (Field { values: &ui.values, levels: 1 }, Field { values: &vi.values, levels: 1 });
                let mut out = vec![MISSING; points];
                out.par_iter_mut().enumerate().for_each(|(q, slot)| {
                    if map.inside[q] {
                        let a = map.linear_at(&uf, q, 0);
                        let b = map.linear_at(&vf, q, 0);
                        *slot = if *which == 'u' {
                            (a * cosa[q] + b * sina[q]) as f32
                        } else {
                            (-a * sina[q] + b * cosa[q]) as f32
                        };
                    }
                });
                out
            }
            Job::Wind3 { u, v, which } => {
                if let Some(done) = wind3_cache.remove(which) {
                    done
                } else {
                    let ui = read_input(&hist, u, mesh.n_cells)?.ok_or_else(|| refusal(format!("{u} absent")))?;
                    let vi = read_input(&hist, v, mesh.n_cells)?.ok_or_else(|| refusal(format!("{v} absent")))?;
                    let st = stagger.as_ref().expect("built above");
                    let (uf, vf) = (Field { values: &ui.values, levels: nz }, Field { values: &vi.values, levels: nz });
                    let mut uo = Vec::with_capacity(nz * ny * (nx + 1));
                    let mut vo = Vec::with_capacity(nz * (ny + 1) * nx);
                    let per_level: Vec<(Vec<f32>, Vec<f32>)> = (0..nz)
                        .into_par_iter()
                        .map(|k| {
                            let mut um = vec![0.0f64; points];
                            let mut vm = vec![0.0f64; points];
                            for q in 0..points {
                                if map.inside[q] {
                                    let a = map.linear_at(&uf, q, k);
                                    let b = map.linear_at(&vf, q, k);
                                    um[q] = a * cosa[q] + b * sina[q];
                                    vm[q] = -a * sina[q] + b * cosa[q];
                                }
                            }
                            (st.u(&um), st.v(&vm))
                        })
                        .collect();
                    for (a, b) in per_level {
                        uo.extend(a);
                        vo.extend(b);
                    }
                    let other = if *which == 'u' { 'v' } else { 'u' };
                    if *which == 'u' {
                        wind3_cache.insert(other, vo);
                        uo
                    } else {
                        wind3_cache.insert(other, uo);
                        vo
                    }
                }
            }
        };
        if p.name == "PB" && nz >= 2 {
            let (top, below) = (&data[(nz - 1) * points..nz * points], &data[(nz - 2) * points..(nz - 1) * points]);
            p_top = top
                .par_iter()
                .zip(below.par_iter())
                .filter(|(a, b)| **a != MISSING && **b != MISSING)
                .map(|(a, b)| 2.0 * *a - *b)
                .reduce(|| f32::INFINITY, f32::min);
            if !p_top.is_finite() {
                p_top = MISSING;
            }
        }
        writer
            .put_parallel(&[(p.name.as_str(), Some(0))], |_| Ok(NcSlab::Borrowed(NcData::Floats(&data))))
            .map_err(werr)?;
        written.push(p.name.clone());
    }
    timer.lap("remap and write every listed variable", true);
    writer.finish().map_err(werr)?;
    std::fs::rename(&tmp, &nl.output_file)?;
    timer.lap("finish and rename", false);

    Ok(Receipt {
        schema: SCHEMA,
        method: METHOD,
        output: nl.output_file.display().to_string(),
        grid: format!(
            "lambert e_we {} e_sn {} ({} x {} mass points) dx {} ref ({}, {}) truelat ({}, {}) stand_lon {}",
            grid.e_we, grid.e_sn, nx, ny, nl.dx, nl.ref_lat, nl.ref_lon, nl.truelat1, nl.truelat2, nl.stand_lon
        ),
        source_cells: mesh.n_cells,
        target_points: points,
        inside_points: map.inside_count(),
        variables_written: written,
        absent_inputs: absent,
        threads,
        stages: timer.stages,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rrfs_namelist() -> Namelist {
        let text = "&config\n grid_file_input_grid = \"invariant.nc\"\n hist_file_input_grid = 'h.nc'\n \
                    diag_file_input_grid = \"d.nc\"\n output_file = \"o.nc\"\n target_grid_type = 'lambert'\n \
                    interp_diag = .true.\n interp_hist = .true.\n wrf_mod_vars = .true.\n esmf_log = .false.\n \
                    nx = 1800\n ny = 1060\n dx = 3000.0\n dy = 3000.0\n ref_lat = 38.5\n ref_lon = -97.5\n \
                    truelat1 = 38.5\n truelat2 = 38.5\n stand_lon = -97.5\n/\n";
        Namelist::parse(text, Path::new("/run")).unwrap()
    }

    #[test]
    fn rrfs_v2_namelist_is_the_hrrr_grid() {
        let nl = rrfs_namelist();
        nl.check_supported().unwrap();
        assert_eq!(nl.grid_file, PathBuf::from("/run/invariant.nc"));
        let g = TargetGrid::from_namelist(&nl);
        assert_eq!((g.nx, g.ny), (1799, 1059));
        // HRRR's published first point: 21.138123N, 237.280472E (-122.719528).
        let (lat, lon) = g.proj.ijll(1.0, 1.0);
        assert!((lat - 21.138123).abs() < 1e-5, "{lat}");
        assert!((lon + 122.719528).abs() < 1e-5, "{lon}");
        let (lat, lon) = g.proj.ijll(900.0, 530.0);
        assert!((lat - 38.5).abs() < 1e-9 && (lon + 97.5).abs() < 1e-9);
        // The referee's CEN_LAT/CEN_LON (measured 38.4865067949845, -97.5172364865423).
        let (clat, clon) = g.centre();
        assert!((clat - 38.4865067949845).abs() < 1e-9 && (clon + 97.5172364865423).abs() < 1e-9);
        assert!((g.map_factor(38.5) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn rotation_matches_the_cone_inside_the_grid() {
        let g = TargetGrid::from_namelist(&rrfs_namelist());
        let (lat, lon) = g.lattice(40, 30, 1.0, 1.0);
        let (s, c) = rotation(&lat, &lon, 40, 30);
        let cone = g.cone();
        for j in 1..29 {
            for i in 0..40 {
                let a = cone * (lon[j * 40 + i] + 97.5) * std::f64::consts::PI / 180.0;
                assert!((s[j * 40 + i] + a.sin()).abs() < 1e-6);
                assert!((c[j * 40 + i] - a.cos()).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn lists_read_tabs_and_blank_lines() {
        let dir = std::env::temp_dir().join(format!("mpassit-lists-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("histlist_soil"), "tslb\t\t\tTSLB\nsmois\tSMOIS\n\nsh2o\t\t\tSH2O\t\n").unwrap();
        let l = VariableLists::read(&dir).unwrap();
        assert_eq!(l.hist_soil.len(), 3);
        assert_eq!(l.hist_soil[2], ListEntry { mpas: "sh2o".into(), wrf: "SH2O".into() });
        assert!(l.diag.is_empty());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn exact_overlap_of_a_cell_with_itself_is_its_area() {
        let c = unit(40.0, -100.0);
        let chart = Chart::at(c);
        let sq: Vec<[f64; 3]> = [(-0.01, -0.01), (0.01, -0.01), (0.01, 0.01), (-0.01, 0.01)]
            .iter()
            .map(|&(a, b)| chart.to_sphere([a, b]))
            .collect();
        let plane: Vec<[f64; 2]> = sq.iter().map(|&v| chart.to_plane(v).unwrap()).collect();
        let cut = clip_convex(&plane, &plane);
        let back: Vec<[f64; 3]> = cut.iter().map(|&q| chart.to_sphere(q)).collect();
        let a = polygon_area(&sq);
        assert!((polygon_area(&back) - a).abs() < 1e-15);
        // half-overlap
        let shifted: Vec<[f64; 2]> = plane.iter().map(|p| [p[0] + 0.01, p[1]]).collect();
        let half = clip_convex(&shifted, &plane);
        let hb: Vec<[f64; 3]> = half.iter().map(|&q| chart.to_sphere(q)).collect();
        assert!((polygon_area(&hb) / a - 0.5).abs() < 1e-3);
    }

    #[test]
    fn stamps_count_minutes() {
        let s = parse_stamp("2023-03-31_12:00:00").unwrap();
        let v = parse_stamp("2023-03-31_18:00:00").unwrap();
        assert_eq!((s - v) / 60, -360);
        assert_eq!(parse_stamp("1970-01-01_00:00:00"), Some(0));
    }

    #[test]
    fn unsupported_configurations_are_refused_by_name() {
        let mut nl = rrfs_namelist();
        nl.target_grid_type = "lat-lon".into();
        assert!(format!("{:?}", nl.check_supported()).contains("lat-lon"));
        let mut nl = rrfs_namelist();
        nl.wrf_mod_vars = false;
        assert!(format!("{:?}", nl.check_supported()).contains("wrf_mod_vars"));
    }
}
