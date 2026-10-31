//! GSL UGWP geography generator, MPAS-A v8.4.2.
//! Source: mpas_init_atm_gwd_gsl.F:49-203,
//! mpas_gsl_oro_data_sm_scale.F:66-750, mpas_gsl_oro_data_lg_scale.F:68-1036.
//! Reductions keep Fortran's j-then-i order. Geography decoding is always
//! binary32; the selected RKIND controls geometry, detrending and statistics.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::path::Path;

pub type Result<T> = std::result::Result<T, String>;

// CPU libm is the same glibc used by the strict GNU reference. Explicit
// calls avoid replacing COS with a different polynomial or wider argument.
// Unix only, like the crate's other libm blocks: Windows has no m.lib and
// takes cosf/cos from the C runtime (2.9.10 pass 2, R8 link LNK1181).
#[cfg_attr(unix, link(name = "m"))]
unsafe extern "C" {
    fn cosf(x: f32) -> f32;
    fn cos(x: f64) -> f64;
}

pub trait Real:
    Copy
    + PartialOrd
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
{
    fn from_f64(x: f64) -> Self;
    fn to_f64(self) -> f64;
    fn sqrt(self) -> Self;
    fn cos(self) -> Self;
    fn abs(self) -> Self;
    fn finite(self) -> bool;
}
macro_rules! real {
    ($t:ty, $cos:ident) => {
        impl Real for $t {
            fn from_f64(x: f64) -> Self {
                x as Self
            }
            fn to_f64(self) -> f64 {
                self as f64
            }
            fn sqrt(self) -> Self {
                self.sqrt()
            }
            fn cos(self) -> Self {
                unsafe { $cos(self) }
            }
            fn abs(self) -> Self {
                self.abs()
            }
            fn finite(self) -> bool {
                self.is_finite()
            }
        }
    };
}
real!(f32, cosf);
real!(f64, cos);
fn r<T: Real>(x: f64) -> T {
    T::from_f64(x)
}
// Pi parameter: mpas_gsl_oro_data_sm_scale.F:51, mpas_gsl_oro_data_lg_scale.F:50.
fn pi<T: Real>() -> T {
    r(std::f64::consts::PI)
}

/// Source rows are south to north, points west to east, with column 1 at
/// 180 W. This fixed frame is specified by the native GSL archive contract.
pub trait Topography {
    fn shape(&self) -> (usize, usize);
    fn row(&mut self, j: usize) -> Result<&[f32]>;
    fn read_error(&self) -> Option<&str> { None }
}

/// The MPAS decoder's 4-byte big-endian signed integer conversion followed
/// by its c_float scale multiplication. Source: read_geogrid.c:131-138;
/// mpas_gsl_oro_data_sm_scale.F:144-186; mpas_gsl_oro_data_lg_scale.F:148-190.
pub fn decode_height(bytes: [u8; 4]) -> f32 {
    i32::from_be_bytes(bytes) as f32 * 0.1f32
}

/// At most one global source row plus one tile-row buffer is resident.
/// Tile files stay open; seeking has no effect on reduction order.
pub struct Tiles {
    files: Vec<File>,
    nx: usize,
    ny: usize,
    row: Vec<f32>,
    raw: Vec<u8>,
    cached: Option<usize>,
    read_error: Option<String>,
}
impl Tiles {
    pub fn open(root: &Path, small: bool) -> Result<Self> {
        let (nx, ny, subdir) = if small {
            (43200, 21600, "topo_ugwp_30s")
        } else {
            (8640, 4320, "topo_ugwp_2.5m")
        };
        let mut files = Vec::new();
        let mut read_error = None;
        'tiles: for j in (0..ny).step_by(4320) {
            for i in (0..nx).step_by(8640) {
                let path = root.join(subdir).join(format!(
                    "{:05}-{:05}.{:05}-{:05}",
                    i + 1,
                    i + 8640,
                    j + 1,
                    j + 4320
                ));
                // Native logs an ERR and returns from this scale, not init.
                // Source: read_geogrid.c:50-88; mpas_gsl_oro_data_sm_scale.F:159-169;
                // mpas_gsl_oro_data_lg_scale.F:163-174.
                let mut file = match File::open(&path) {
                    Ok(file) => file,
                    Err(e) => { read_error = Some(format!("{}: {e}", path.display())); break 'tiles; }
                };
                // A successful open does not imply a readable regular file.
                // Native tests fread's byte count, not stat's length; directories
                // open successfully but produce a nonfatal zero-read error.
                // Source: read_geogrid.c:50-88; strict full-init both-host witness.
                let mut first = [0u8; 1];
                loop {
                    match file.read(&mut first) {
                        Ok(0) => {
                            read_error = Some(format!("{}: native fread returns zero bytes", path.display()));
                            break 'tiles;
                        }
                        Ok(_) => break,
                        Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                        Err(e) => { read_error = Some(format!("{}: {e}", path.display())); break 'tiles; }
                    }
                }
                files.push(file);
            }
        }
        Ok(Self {
            files,
            nx,
            ny,
            row: vec![0.; nx],
            raw: vec![0; 8640 * 4],
            cached: None,
            read_error,
        })
    }
}
impl Topography for Tiles {
    fn shape(&self) -> (usize, usize) {
        (self.nx, self.ny)
    }
    fn read_error(&self) -> Option<&str> { self.read_error.as_deref() }
    fn row(&mut self, j: usize) -> Result<&[f32]> {
        if j >= self.ny {
            return Err("GSL topography row outside source".into());
        }
        if let Some(error) = &self.read_error { return Err(error.clone()); }
        if self.cached != Some(j) {
            for tx in 0..self.nx / 8640 {
                let file = &mut self.files[(j / 4320) * (self.nx / 8640) + tx];
                file.seek(SeekFrom::Start(((j % 4320) * 8640 * 4) as u64))
                    .map_err(|e| e.to_string())?;
                // Positive short fread is accepted by the pinned decoder.
                // Its large malloc tail is binary zero in the measured strict
                // full-init repeats on both hosts. Source: read_geogrid.c:66-138.
                self.raw.fill(0);
                let mut offset = 0;
                while offset < self.raw.len() {
                    match file.read(&mut self.raw[offset..]) {
                        Ok(0) => break,
                        Ok(n) => offset += n,
                        Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                        Err(e) => return Err(e.to_string()),
                    }
                }
                for (i, b) in self.raw.chunks_exact(4).enumerate() {
                    self.row[tx * 8640 + i] = decode_height(b.try_into().unwrap());
                }
            }
            self.cached = Some(j);
        }
        Ok(&self.row)
    }
}

/// Native fields ordered stddev, convexity, oa1..4, ol1..4.
pub const BASE_NAMES: [&str; 10] = [
    "var2d", "con", "oa1", "oa2", "oa3", "oa4", "ol1", "ol2", "ol3", "ol4",
];
pub struct Fields<T> {
    pub small: Vec<[T; 10]>,
    pub large: Vec<[T; 10]>,
    pub duplicate: Vec<bool>,
}

/// Approximate square cell areas. Source: mpas_init_atm_gwd_gsl.F:122-165.
/// Edge indices are MPAS 1-based; padding is excluded by nEdgesOnCell.
pub fn cell_areas<T: Real>(
    counts: &[usize],
    edges: &[usize],
    max_edges: usize,
    dc_edge: &[T],
    sphere_radius: T,
    scaling: T,
) -> Result<Vec<T>> {
    if dc_edge.is_empty()
        || edges.len()
            != counts
                .len()
                .checked_mul(max_edges)
                .ok_or("edge shape overflow")?
        || !sphere_radius.finite()
        || sphere_radius <= r(0.)
        || !scaling.finite()
        || scaling <= r(0.)
    {
        return Err(
            "invalid GSL mesh dimensions, radius or scaling; cannot define cell boxes".into(),
        );
    }
    if dc_edge.iter().any(|x| !x.finite() || *x <= r(0.)) {
        return Err("nonpositive/nonfinite dcEdge gives invalid GSL boxes".into());
    }
    let unit = dc_edge.iter().all(|x| *x < r(1.));
    let mut out = Vec::with_capacity(counts.len());
    for (cell, &count) in counts.iter().enumerate() {
        if count == 0 || count > max_edges {
            return Err(
                "invalid nEdgesOnCell; GSL mean diameter would divide by zero or include padding"
                    .into(),
            );
        }
        let mut dc = r::<T>(0.);
        for &edge in &edges[cell * max_edges..cell * max_edges + count] {
            let x = edge
                .checked_sub(1)
                .and_then(|e| dc_edge.get(e))
                .ok_or("edgesOnCell outside dcEdge")?;
            dc = dc + *x;
        }
        dc = dc / r(count as f64);
        if unit {
            dc = dc * sphere_radius;
        }
        dc = dc * scaling;
        out.push(dc * dc);
    }
    Ok(out)
}

struct Frame<T> {
    lat: Vec<T>,
    lon: Vec<T>,
    small: bool,
}
impl<T: Real> Frame<T> {
    /// Source coordinates: mpas_gsl_oro_data_sm_scale.F:178-186, mpas_gsl_oro_data_lg_scale.F:182-190.
    fn new(nx: usize, ny: usize, small: bool) -> Self {
        let coord = |i: usize, n: usize, span: f64| {
            (-r::<T>(span / 2.) + (r::<T>(span) / r(n as f64)) * (r::<T>((i + 1) as f64) - r(0.5)))
                * pi::<T>()
                / r(180.)
        };
        Self {
            lat: (0..ny).map(|j| coord(j, ny, 180.)).collect(),
            lon: (0..nx).map(|i| coord(i, nx, 360.)).collect(),
            small,
        }
    }
    /// nearest_i_east/west: mpas_gsl_oro_data_sm_scale.F:607-666, mpas_gsl_oro_data_lg_scale.F:893-952.
    /// Binary search is equivalent to the native monotonic linear scan,
    /// including the unusual western wrap below the first point.
    fn longitude(&self, mut x: T, east: bool) -> usize {
        while x < -pi::<T>() || x > pi::<T>() {
            if x < -pi::<T>() {
                x = x + r::<T>(2.) * pi::<T>();
            }
            if x > pi::<T>() {
                x = x - r::<T>(2.) * pi::<T>();
            }
        }
        let n = self.lon.len();
        if east {
            if x > self.lon[n - 1] {
                1
            } else {
                self.lon.partition_point(|p| *p < x) + 1
            }
        } else if x < self.lon[0] || x >= self.lon[n - 1] {
            n
        } else {
            self.lon.partition_point(|p| *p <= x)
        }
    }
    /// nearest_j_north/south: mpas_gsl_oro_data_sm_scale.F:673-726, mpas_gsl_oro_data_lg_scale.F:959-1012.
    /// Small scale uses j<=topo_y, so its terminal out-of-array comparison
    /// cannot change the returned index topo_y. Large scale uses j<topo_y
    /// and returns at most topo_y-1. Preserve both native terminal indices.
    fn latitude(&self, x: T, north: bool) -> i32 {
        if x.abs() > r::<T>(0.5) * pi::<T>() {
            -999
        } else if north {
            (self.lat.partition_point(|p| *p < x) + 1).min(self.lat.len()) as i32
        } else {
            let limit = self.lat.len() - usize::from(!self.small);
            self.lat.partition_point(|p| *p <= x).max(1).min(limit) as i32
        }
    }
}

struct Blocks<T> {
    lat: [T; 3],
    lon: [T; 3],
    si: [usize; 3],
    ei: [usize; 3],
    sj: [i32; 3],
    ej: [i32; 3],
}
/// Box selection, polar shaving: mpas_gsl_oro_data_sm_scale.F:231-305, mpas_gsl_oro_data_lg_scale.F:237-335.
fn blocks<T: Real>(frame: &Frame<T>, lat: T, mut lon: T, area: T) -> Result<Blocks<T>> {
    if lon > pi::<T>() {
        lon = lon - r::<T>(2.) * pi::<T>();
    }
    let dlat = area.sqrt() / r(6371229.);
    let dlon = area.sqrt() / (r::<T>(6371229.) * lat.cos());
    if !dlon.finite() {
        return Err("latitude yields undefined GSL longitude width".into());
    }
    let mut b = Blocks {
        lat: [r(0.); 3],
        lon: [r(0.); 3],
        si: [0; 3],
        ei: [0; 3],
        sj: [0; 3],
        ej: [0; 3],
    };
    for k in 0..3 {
        b.lon[k] = lon + r::<T>(k as f64 - 1.) * dlon;
        b.lat[k] = lat + r::<T>(k as f64 - 1.) * dlat;
        b.si[k] = frame.longitude(b.lon[k] - r::<T>(0.5) * dlon, true);
        b.ei[k] = frame.longitude(b.lon[k] + r::<T>(0.5) * dlon, false);
        b.sj[k] = frame.latitude(b.lat[k] - r::<T>(0.5) * dlat, true);
        b.ej[k] = frame.latitude(b.lat[k] + r::<T>(0.5) * dlat, false);
    }
    let ny = frame.lat.len() as i32;
    // Repeat the native shaving assignments once: later i_blk passes are
    // no-ops because their sentinel was already replaced on the first pass.
    for k in 0..3 {
        if b.sj[k] == -999 && (k == 1 || (k == 0 && b.ej[k] != -999)) {
            if b.ej[k] == -999 {
                return Err(
                    "cell spans both poles; native GSL polar-shaving assumption fails".into(),
                );
            }
            b.sj[k] = 1;
            b.lat[k] = r::<T>(0.5) * (frame.lat[0] + frame.lat[b.ej[k] as usize - 1]);
        }
        if b.ej[k] == -999 && (k == 1 || (k == 2 && b.sj[k] != -999)) {
            b.ej[k] = ny;
            b.lat[k] = r::<T>(0.5) * (frame.lat[b.sj[k] as usize - 1] + frame.lat[ny as usize - 1]);
        }
    }
    if b.sj[1] < 1 || b.ej[1] < b.sj[1] {
        return Err("no fine terrain rows inside GSL center block".into());
    }
    Ok(b)
}

// Date-line box width: mpas_gsl_oro_data_sm_scale.F:299-306.
fn width(si: usize, ei: usize, nx: usize) -> usize {
    if si > ei {
        nx - si + 1 + ei
    } else {
        ei - si + 1
    }
}
// Wrapped sample order: mpas_gsl_oro_data_sm_scale.F:313-331.
fn column(si: usize, local: usize, nx: usize) -> usize {
    (si - 1 + local) % nx
}

/// calc_mean_HGT: mpas_gsl_oro_data_lg_scale.F:701-751.
fn mean<T: Real>(source: &mut impl Topography, b: &Blocks<T>, i: usize, j: usize) -> Result<T> {
    if b.sj[j] == -999 && b.ej[j] == -999 {
        return Ok(r(1e10));
    }
    if b.sj[j] < 1 || b.ej[j] < b.sj[j] {
        return Err("empty neighboring GSL box; native mean would divide by zero".into());
    }
    let nx = source.shape().0;
    let n = width(b.si[i], b.ei[i], nx);
    let mut sum = r::<T>(0.);
    let mut count = 0;
    for y in b.sj[j]..=b.ej[j] {
        let row = source.row(y as usize - 1)?;
        for x in 0..n {
            sum = sum + r(row[column(b.si[i], x, nx)] as f64);
            count += 1;
        }
    }
    Ok(sum / r(count as f64))
}

/// interp_1d: mpas_gsl_oro_data_lg_scale.F:1023-1036, mpas_gsl_oro_data_sm_scale.F:737-750.
fn interp<T: Real>(x: T, x1: T, x2: T, y1: T, y2: T) -> T {
    let slope = (y2 - y1) / (x2 - x1);
    y1 + slope * (x - x1)
}

/// HGT_interpolate: mpas_gsl_oro_data_lg_scale.F:762-886.
/// Quadrants share the same expression. Missing northern/southern rows
/// select the same latitude pair for interpolation and extrapolation.
fn detrend<T: Real>(lat: T, mut lon: T, b: &Blocks<T>, h: &[[T; 3]; 3]) -> T {
    if (lon - b.lon[1]).abs() > pi::<T>() {
        if lon > r(0.) {
            lon = lon - r::<T>(2.) * pi::<T>();
        } else if lon < r(0.) {
            lon = lon + r::<T>(2.) * pi::<T>();
        }
    }
    let i = if lon >= b.lon[1] { 1 } else { 0 };
    let j = if h.iter().any(|col| col[2] == r(1e10)) {
        0
    } else if h.iter().any(|col| col[0] == r(1e10)) || lat >= b.lat[1] {
        1
    } else {
        0
    };
    interp(
        lon,
        b.lon[i],
        b.lon[i + 1],
        interp(lat, b.lat[j], b.lat[j + 1], h[i][j], h[i][j + 1]),
        interp(lat, b.lat[j], b.lat[j + 1], h[i + 1][j], h[i + 1][j + 1]),
    )
}

/// All ten statistics: mpas_gsl_oro_data_sm_scale.F:338-586, mpas_gsl_oro_data_lg_scale.F:407-674.
/// Odd box halves and diagonal equality overlap are native definitions.
fn statistics<T: Real>(z: &[T], nx: usize, ny: usize, small: bool) -> [T; 10] {
    let mut out = [r::<T>(0.); 10];
    // Unsuffixed Fortran thresholds are binary32, even in double RKIND.
    let threshold = r(if small {
        0.1f32 as f64
    } else {
        0.001f32 as f64
    });
    if !z.iter().any(|x| x.abs() > threshold) {
        return out;
    }
    let count = r(z.len() as f64);
    let mut sum = r::<T>(0.);
    for &x in z {
        sum = sum + x;
    }
    let avg = sum / count;
    let mut sum2 = r::<T>(0.);
    let mut sum4 = r::<T>(0.);
    for &x in z {
        let d = x - avg;
        let d2 = d * d;
        sum2 = sum2 + d2;
        sum4 = sum4 + d2 * d2;
    }
    let var = sum2 / count;
    out[0] = var.sqrt();
    out[1] = if var.abs() < r(1e-5) {
        r(0.)
    } else {
        let c = sum4 / ((var * var) * count);
        if c < r(10.) { c } else { r(10.) }
    };
    let mut up = [0i64; 4];
    let mut down = [0i64; 4];
    let mut above = [0i64; 4];
    let mut total = [0i64; 4];
    let ratio = r::<T>(ny as f64) / r(nx as f64);
    for j in 1..=ny {
        for i in 1..=nx {
            let high = z[(j - 1) * nx + i - 1] > avg;
            let diagonal = (r::<T>(i as f64) * ratio).to_f64().round() as usize;
            let u = [
                i <= nx / 2 + nx % 2,
                j <= ny / 2 + ny % 2,
                diagonal <= ny - j + 1,
                diagonal <= j,
            ];
            let d = [
                i >= nx / 2 + 1,
                j >= ny / 2 + 1,
                diagonal >= ny - j + 1,
                diagonal >= j,
            ];
            let band = [
                j >= (ny / 4).max(1) && j <= 3 * ny / 4,
                i >= (nx / 4).max(1) && i <= 3 * nx / 4,
                (j <= ny / 2 && i <= nx / 2) || (j > ny / 2 && i > nx / 2),
                (j > ny / 2 && i <= nx / 2) || (j <= ny / 2 && i > nx / 2),
            ];
            for k in 0..4 {
                if high && u[k] {
                    up[k] += 1;
                }
                if high && d[k] {
                    down[k] += 1;
                }
                if band[k] {
                    total[k] += 1;
                    if high {
                        above[k] += 1;
                    }
                }
            }
        }
    }
    for k in 0..4 {
        if up[k] + down[k] > 0 {
            out[2 + k] = r::<T>((up[k] - down[k]) as f64) / r((up[k] + down[k]) as f64);
        }
        if total[k] != 0 {
            out[6 + k] = r::<T>(above[k] as f64) / r(total[k] as f64);
        }
    }
    out
}

/// Entire single-scale cell loop: mpas_gsl_oro_data_sm_scale.F:231-598, mpas_gsl_oro_data_lg_scale.F:233-689.
fn scale<T: Real>(
    lat: &[T],
    lon: &[T],
    areas: &[T],
    skip: &[bool],
    source: &mut impl Topography,
    small: bool,
) -> Result<Vec<[T; 10]>> {
    let (nx, ny) = source.shape();
    if nx == 0 || ny == 0 {
        return Err("empty GSL source grid".into());
    }
    let frame = Frame::<T>::new(nx, ny, small);
    let mut out = vec![[r(0.); 10]; lat.len()];
    for cell in 0..lat.len() {
        if !small && skip[cell] {
            continue;
        }
        let b = blocks(&frame, lat[cell], lon[cell], areas[cell])?;
        let mut h = [[r::<T>(0.); 3]; 3];
        if !small {
            for (i, col) in h.iter_mut().enumerate() {
                for (j, value) in col.iter_mut().enumerate() {
                    *value = mean(source, &b, i, j)?;
                }
            }
        }
        let w = width(b.si[1], b.ei[1], nx);
        let height = (b.ej[1] - b.sj[1] + 1) as usize;
        let len = w
            .checked_mul(height)
            .ok_or("GSL center block allocation overflow")?;
        if len == 0 {
            return Err("empty GSL box; native statistics would divide by zero".into());
        }
        let mut z = Vec::new();
        z.try_reserve_exact(len).map_err(|e| e.to_string())?;
        for y in b.sj[1]..=b.ej[1] {
            let row = source.row(y as usize - 1)?;
            for local in 0..w {
                let x = column(b.si[1], local, nx);
                let mut value = r::<T>(row[x] as f64);
                if !small {
                    value = value - detrend(frame.lat[y as usize - 1], frame.lon[x], &b, &h);
                }
                z.push(value);
            }
        }
        out[cell] = statistics(&z, w, height, small);
    }
    Ok(out)
}

/// The same cell loop on every core. Cells are independent: each worker reads the
/// terrain through its own tile reader (`open` makes one) and writes only its own
/// cells, so every word equals the serial loop's, and the first error in cell order
/// is the one returned. THE BREAKAGE THIS FIXES: the serial loop held one core for
/// ~259 s of a 167,254-cell zoom globe's 9 min RRFS init (2026-10-28).
fn scale_parallel<T: Real + Send + Sync, S: Topography>(
    lat: &[T],
    lon: &[T],
    areas: &[T],
    skip: &[bool],
    open: &(dyn Fn() -> Result<S> + Sync),
    small: bool,
) -> Result<Vec<[T; 10]>> {
    use rayon::prelude::*;
    const CHUNK: usize = 256;
    let starts: Vec<usize> = (0..lat.len()).step_by(CHUNK).collect();
    let parts: Vec<Result<Vec<[T; 10]>>> = starts
        .into_par_iter()
        .map(|start| {
            let end = (start + CHUNK).min(lat.len());
            let mut source = open()?;
            scale(&lat[start..end], &lon[start..end], &areas[start..end], &skip[start..end], &mut source, small)
        })
        .collect();
    let mut out = Vec::with_capacity(lat.len());
    for part in parts {
        out.extend(part?);
    }
    Ok(out)
}

/// Native generator boundary, including duplication below 7500 m.
/// Source: mpas_init_atm_gwd_gsl.F:167-191; mpas_gsl_oro_data_sm_scale.F:212-217.
pub fn generate<T: Real>(
    lat: &[T],
    lon: &[T],
    areas: &[T],
    small: &mut impl Topography,
    large: &mut impl Topography,
) -> Result<Fields<T>> {
    let failed_small = small.read_error().is_some();
    let failed_large = large.read_error().is_some();
    generate_by(lat, lon, areas, failed_small, failed_large,
                |skip| scale(lat, lon, areas, skip, small, true),
                |skip| scale(lat, lon, areas, skip, large, false))
}

/// [`generate`] on every core; `open_small`/`open_large` each make a fresh tile reader.
pub fn generate_parallel<T: Real + Send + Sync, S: Topography>(
    lat: &[T],
    lon: &[T],
    areas: &[T],
    open_small: &(dyn Fn() -> Result<S> + Sync),
    open_large: &(dyn Fn() -> Result<S> + Sync),
) -> Result<Fields<T>> {
    let failed_small = open_small()?.read_error().is_some();
    let failed_large = open_large()?.read_error().is_some();
    generate_by(lat, lon, areas, failed_small, failed_large,
                |skip| scale_parallel(lat, lon, areas, skip, open_small, true),
                |skip| scale_parallel(lat, lon, areas, skip, open_large, false))
}

fn generate_by<T: Real>(
    lat: &[T],
    lon: &[T],
    areas: &[T],
    failed_small: bool,
    failed_large: bool,
    small: impl FnOnce(&[bool]) -> Result<Vec<[T; 10]>>,
    large: impl FnOnce(&[bool]) -> Result<Vec<[T; 10]>>,
) -> Result<Fields<T>> {
    if lat.len() != lon.len() || lat.len() != areas.len() || lat.is_empty() {
        return Err("GSL cell arrays have inconsistent or empty dimensions".into());
    }
    for i in 0..lat.len() {
        if !lat[i].finite()
            || lat[i].abs() > r::<T>(0.5) * pi::<T>()
            || !lon[i].finite()
            || !areas[i].finite()
            || areas[i] <= r(0.)
        {
            return Err("invalid GSL cell coordinate or area".into());
        }
    }
    // A failed small-scale read leaves duplicate_oro_data unset. Full pinned
    // init repeats at fixed 1/2 ranks on both hosts produce all-zero outputs,
    // even with valid large-scale terrain. Keep these words inside the claim.
    // Source: mpas_gsl_oro_data_sm_scale.F:159-169;
    // mpas_init_atm_gwd_gsl.F:144-145,167-203.
    let duplicate: Vec<_> = areas.iter().map(|a| failed_small || a.sqrt() < r(7500.)).collect();
    let ss = if failed_small { vec![[r(0.); 10]; lat.len()] }
             else { small(&duplicate)? };
    // Framework real fields start at zero; an ERR return leaves them there.
    // Source: mpas_gsl_oro_data_lg_scale.F:163-174;
    // mpas_block_creator.F:1649; registry/gen_inc.c:615-623.
    let mut ls = if failed_large || failed_small { vec![[r(0.); 10]; lat.len()] }
                 else { large(&duplicate)? };
    for i in 0..ls.len() {
        if duplicate[i] {
            ls[i] = ss[i];
        }
    }
    Ok(Fields {
        small: ss,
        large: ls,
        duplicate,
    })
}

#[cfg(test)]
mod latitude_tests {
    use super::*;

    // Verbatim strict-Fortran witnesses: tests/gsl_oracle/latitude_witness.py.
    // Source: mpas_gsl_oro_data_sm_scale.F:673-726 and
    // mpas_gsl_oro_data_lg_scale.F:959-1012. Their terminal bounds differ.
    fn native_terminal_indices<T: Real>() {
        for (small, nx, ny) in [(true, 43200, 21600), (false, 8640, 4320)] {
            let frame = Frame::<T>::new(nx, ny, small);
            let last = frame.lat[ny - 1];
            let pole = r::<T>(0.5) * pi::<T>();
            for x in [last, (last + pole) * r(0.5), pole] {
                assert_eq!(frame.latitude(x, true), ny as i32);
                assert_eq!(frame.latitude(x, false), (ny - usize::from(!small)) as i32);
            }
            for x in [frame.lat[0], -pole] {
                assert_eq!(frame.latitude(x, true), 1);
                assert_eq!(frame.latitude(x, false), 1);
            }
            for x in [-pi::<T>(), pi::<T>()] {
                assert_eq!(frame.latitude(x, true), -999);
                assert_eq!(frame.latitude(x, false), -999);
            }
        }
    }

    #[test]
    fn single_native_latitude_terminal_indices() {
        native_terminal_indices::<f32>();
    }

    #[test]
    fn double_native_latitude_terminal_indices() {
        native_terminal_indices::<f64>();
    }
}
