//! WRF v4.8.0's orographic statistics for gwd_opt = 1, as WPS writes them: ELVMAX plus VAR, CON, OA1-4, OL1-4.
//!
//! Authority: WPS PR #281 (merged into WPS `develop` 2026-07-01), `util/compute_gwdo.py`,
//! `util/OrographyStats.py`, `util/SourceData.py`, `util/TileData.py`.  WRF v4.8.0's revised KIM drag
//! (`gwd_opt = 1`, Registry.EM_COMMON `ELVMAX`) blocks the flow below ELVMAX and gives no drag where
//! ELVMAX <= 50 m (bl_gwdo.F90:416), so without this field the scheme silently turns itself off.
//!
//! Per mass point (XLAT_M, XLONG_M) the script takes a box of the global 30" terrain
//! (`topo_gmted2010_30s`, geogrid format) whose side is `dc = 2 * min(DX, DY)`, floored at four source pixels
//! (and then scales VAR by `hratio / 4`), and writes, over the whole box with no land mask:
//! ELVMAX = max, VAR = standard deviation, CON = fourth moment / (n std^4) (0 when std < 1),
//! OA1-4 = (nu - nd) / (nu + nd) counts above the mean on the upstream / downstream halves (diagonals counted
//! on both sides by a geometric slope rule), OL1-4 = fraction above the mean of the interior half.
//!
//! The arithmetic follows the script's numpy path (numpy 2 rules; netCDF4 0-d masked arrays promote to float64): the box as float32, numpy's pairwise float32
//! summation for the mean, the variance and the fourth moment, means divided in float64 by the integer count
//! and rounded back to float32 (numpy `_mean`/`_var` with an intp count), `powf` for the fourth power, and
//! float64 count ratios for OA/OL rounded to float32 on write.  ELVMAX (a max) and the counts are
//! order-independent; VAR and CON follow numpy's summation order.  One worker per point on every core.
//!
//! Boxes that cross a pole are refused by name: the script rotates whole source tiles there, and at the last
//! tile row it indexes past its tile list (IndexError), so there is no defined native value to match.

use rayon::prelude::*;

use crate::error::StaticError;
use crate::geog::GeogDataset;

type Result<T> = std::result::Result<T, StaticError>;

/// geogrid's earth radius (SourceData._earth_radius), not MPAS's.
const EARTH_RADIUS_M: f64 = 6_370_000.0;

/// The eleven fields, each `(n,)` float32 in the order of the input points.
#[derive(Debug, Clone, Default)]
pub struct Wps281Fields {
    pub var: Vec<f32>,
    pub con: Vec<f32>,
    pub oa: [Vec<f32>; 4],
    pub ol: [Vec<f32>; 4],
    pub elvmax: Vec<f32>,
    /// Source pixels per box side actually used (diagnostics): (nx, ny) of the first point.
    pub box_first: (usize, usize),
    pub hratio: f32,
}

/// numpy's pairwise float32 sum (numpy/_core/src/umath/loops_utils.h.src, PW_BLOCKSIZE 128).
fn pairwise_sum(a: &[f32]) -> f32 {
    let n = a.len();
    if n < 8 {
        let mut res = 0.0f32;
        for &v in a {
            res += v;
        }
        res
    } else if n <= 128 {
        let mut r = [0f32; 8];
        r.copy_from_slice(&a[..8]);
        let mut i = 8;
        while i < n - (n % 8) {
            for j in 0..8 {
                r[j] += a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[i];
            i += 1;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        pairwise_sum(&a[..n2]) + pairwise_sum(&a[n2..])
    }
}

/// `np.add.reduce` over a C-contiguous box with no `initial`: the iterator coalesces the box into one run and
/// the pairwise loop sums all of it (checked against numpy 2.5 on 300 random boxes, every one equal).
fn np_sum(a: &[f32]) -> f32 {
    pairwise_sum(a)
}

/// numpy `_mean`: float32 sum, divided in float64 by the intp count, rounded to float32.
fn np_mean(a: &[f32]) -> f32 {
    (np_sum(a) as f64 / a.len() as f64) as f32
}

/// numpy `_var`/`std` (ddof 0): mean as above, float32 deviations squared, float32 sum, float64 division,
/// float32 square root.
fn np_std(a: &[f32], mean: f32) -> f32 {
    let sq: Vec<f32> = a.iter().map(|&v| {
        let d = v - mean;
        d * d
    }).collect();
    ((np_sum(&sq) as f64 / a.len() as f64) as f32).sqrt()
}

#[derive(Debug, Clone, Copy)]
struct Stats {
    max: f32,
    std: f32,
    con: f32,
    oa: [f32; 4],
    ol: [f32; 4],
}

fn ratio(nu: i64, nd: i64) -> f32 {
    if nu + nd > 0 { ((nu - nd) as f64 / (nu + nd) as f64) as f32 } else { 0.0 }
}

/// OrographyStats(box) for a row-major `(ny, nx)` box, row 0 the southernmost.
fn box_stats(b: &[f32], ny: usize, nx: usize) -> Stats {
    let mean = np_mean(b);
    let std = np_std(b, mean);
    let max = b.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let con = if std < 1.0 {
        0.0
    } else {
        let p: Vec<f32> = b.iter().map(|&h| rw_libm::powf(h - mean, 4.0)).collect();
        let var4 = np_sum(&p);
        var4 / ((b.len() as f32) * rw_libm::powf(std, 4.0))
    };
    let above = |j: usize, i: usize| b[j * nx + i] > mean;
    let count = |j0: usize, j1: usize, i0: usize, i1: usize| -> i64 {
        let mut c = 0i64;
        for j in j0..j1 {
            for i in i0..i1 {
                c += above(j, i) as i64;
            }
        }
        c
    };
    let mut oa = [0f32; 4];
    // oa1 west: columns [0, (nx + nx%2)/2) vs [(nx - nx%2)/2, nx)
    oa[0] = ratio(count(0, ny, 0, (nx + nx % 2) / 2), count(0, ny, (nx - nx % 2) / 2, nx));
    // oa2 south: rows likewise
    oa[1] = ratio(count(0, (ny + ny % 2) / 2, 0, nx), count((ny - ny % 2) / 2, ny, 0, nx));
    // oa3/oa4: geometric diagonal, vals = (i+1)*slope - (ny - j); upstream vals <= slope, downstream vals >= -1.
    let slope = ny as f64 / nx as f64;
    let (mut u3, mut d3, mut u4, mut d4) = (0i64, 0i64, 0i64, 0i64);
    for j in 0..ny {
        for i in 0..nx {
            let vals = (i as f64 + 1.0) * slope - (ny as f64 - j as f64);
            let up = vals <= slope;
            let down = vals >= -1.0;
            // oa4 flips both masks along rows: the mask at row j is the one computed for row ny-1-j.
            let vflip = (i as f64 + 1.0) * slope - (ny as f64 - (ny - 1 - j) as f64);
            let a = above(j, i) as i64;
            if up { u3 += a; }
            if down { d3 += a; }
            if vflip <= slope { u4 += a; }
            if vflip >= -1.0 { d4 += a; }
        }
    }
    oa[2] = ratio(u3, d3);
    oa[3] = ratio(u4, d4);
    let frac = |c: i64, n: usize| (c as f64 / n as f64) as f32;
    let mut ol = [0f32; 4];
    let (r0, r1) = ((ny as f64 * 0.25).floor() as usize, (ny as f64 * 0.75).ceil() as usize);
    ol[0] = frac(count(r0, r1, 0, nx), (r1 - r0) * nx);
    let (c0, c1) = ((nx as f64 * 0.25).floor() as usize, (nx as f64 * 0.75).ceil() as usize);
    ol[1] = frac(count(0, ny, c0, c1), ny * (c1 - c0));
    let (hy, hx) = (ny / 2, nx / 2);
    let size_a = hy * hx;
    let size_b = (ny - hy) * (nx - hx);
    ol[2] = frac(count(0, hy, 0, hx) + count(hy, ny, hx, nx), size_a + size_b);
    let size_a4 = (ny - hy) * hx;
    let size_b4 = hy * (nx - hx);
    ol[3] = frac(count(hy, ny, 0, hx) + count(0, hy, hx, nx), size_a4 + size_b4);
    Stats { max, std, con, oa, ol }
}

/// compute_gwdo.py on one domain: `lat`/`lon` the mass points (float32, degrees, XLAT_M/XLONG_M order),
/// `dx`/`dy` the domain's grid spacing attributes (float32, m).
pub fn compute(topo: &GeogDataset, lat: &[f32], lon: &[f32], dx: f32, dy: f32) -> Result<Wps281Fields> {
    if lat.len() != lon.len() || lat.is_empty() {
        return Err(StaticError::Invalid("ELVMAX: lat and lon must be the same non-zero length".into()));
    }
    let idx = &topo.index;
    if idx.projection != "regular_ll" {
        return Err(StaticError::Invalid(format!(
            "ELVMAX: {} is {} -- WPS PR #281 needs a global regular_ll terrain dataset",
            topo.path.display(), idx.projection)));
    }
    // SourceData: npts = int(360 / dx), pts_per_deg = int(1 / dx), sg_delta = 2 pi R / npts_x.
    let npts_x = (360.0 / idx.dx) as i64;
    let npts_y = (180.0 / idx.dy) as i64;
    let pts_per_deg = (1.0 / idx.dx) as i64;
    let sg_delta = 2.0 * std::f64::consts::PI * EARTH_RADIUS_M / npts_x as f64;
    // dc = min(DX, DY) * 2 (float32); hratio = dc / sg_delta (numpy 2 weak-scalar rules: float32); below 4,
    // dc = 4 sg_delta, a Python float, which turns the box-size products into float64 until they meet the
    // float32 cosine.
    let dc32 = dx.min(dy) * 2.0;
    let hratio = dc32 / sg_delta as f32;
    let small = hratio < 4.0;
    let size64 = sg_delta * 4.0;
    let ppd = pts_per_deg as f32;
    let pi_r = std::f64::consts::PI * EARTH_RADIUS_M;
    // ny = int(ceil(180 * size * ppd / (pi R)))
    let ny = if small {
        ((180.0 * size64 * pts_per_deg as f64) / pi_r).ceil() as i64
    } else {
        ((180.0f32 * dc32 * ppd) / (pi_r as f32)).ceil() as i64
    };
    // cos(lat) > 2 ppd size 180 / (npts_x pi R) decides between a box and half the globe.
    let thresh32 = (240.0f32 * dc32 * 180.0f32) / ((npts_x as f64 * pi_r) as f32);
    let thresh64 = (2.0 * pts_per_deg as f64 * size64 * 180.0) / (npts_x as f64 * pi_r);
    let numerator = if small { (180.0 * size64 * pts_per_deg as f64) as f32 } else { 180.0f32 * dc32 * ppd };
    // Per point: (j0, i0 unwrapped, nx).
    let deg2rad = (std::f64::consts::PI / 180.0) as f32;
    let geometry: Vec<(i64, i64, i64)> = lat.iter().zip(lon).map(|(&la, &lo)| {
        let c = ((la * deg2rad) as f64).cos() as f32;
        // XLAT_M/XLONG_M come out of netCDF4 as 0-d masked arrays, which promote with Python floats to float64:
        // the cosine stays float32, but pi*R*cos and the source index arithmetic are float64.
        let wide = if small { c as f64 > thresh64 } else { c > thresh32 };
        let nx = if wide {
            (numerator as f64 / (pi_r * c as f64)).ceil() as i64
        } else {
            npts_x / 2
        };
        // latlon_to_ij: i = ((lon - known_lon)/dx + known_x) % npts_x; j = (lat - known_lat)/dy + known_y
        let ti = ((lo as f64 - idx.known_lon) / idx.dx + idx.known_x).rem_euclid(npts_x as f64);
        let tj = (la as f64 - idx.known_lat) / idx.dy + idx.known_y;
        // int() truncates toward zero.
        let j0 = (tj - ny as f64 / 2.0).trunc() as i64;
        let i0 = (ti - nx as f64 / 2.0).trunc() as i64;
        (j0, i0, nx)
    }).collect();
    for (k, &(j0, _, _)) in geometry.iter().enumerate() {
        if j0 < 0 || j0 + ny > npts_y {
            return Err(StaticError::Invalid(format!(
                "ELVMAX: the box of point {k} (lat {}) crosses a pole; WPS PR #281 rotates whole source tiles there \
                 and fails at the last tile row, so it has no value to reproduce", lat[k])));
        }
    }
    // One mosaicked window covering every box (array index I is geogrid pixel I + 1; x wraps).
    let mut starts: Vec<i64> = geometry.iter().map(|g| g.1).collect();
    let spread = starts.iter().max().unwrap() - starts.iter().min().unwrap();
    if spread > npts_x / 2 {
        // A domain across the dateline: unwrap the western starts by one turn.
        for s in starts.iter_mut() {
            if *s < npts_x / 2 { *s += npts_x; }
        }
    }
    let x_lo = *starts.iter().min().unwrap();
    let x_hi = starts.iter().zip(&geometry).map(|(s, g)| s + g.2 - 1).max().unwrap();
    let y_lo = geometry.iter().map(|g| g.0).min().unwrap();
    let y_hi = geometry.iter().map(|g| g.0 + ny - 1).max().unwrap();
    let window = topo.read_window(x_lo + 1, x_hi + 1, y_lo + 1, y_hi + 1)?;
    let (wnx, scale) = (window.nx, idx.scale_factor as f32);
    let plane: Vec<f32> = window.raw[..window.ny * window.nx].iter().map(|&r| r as f32 * scale).collect();
    let stats: Vec<Stats> = (0..lat.len()).into_par_iter().map(|k| {
        let (j0, _, nx) = geometry[k];
        let (nx, nyu) = (nx as usize, ny as usize);
        let i0 = (starts[k] - x_lo) as usize;
        let jj = (j0 - y_lo) as usize;
        let mut b = Vec::with_capacity(nx * nyu);
        for r in 0..nyu {
            let row = (jj + r) * wnx + i0;
            b.extend_from_slice(&plane[row..row + nx]);
        }
        box_stats(&b, nyu, nx)
    }).collect();
    let pick = |f: &dyn Fn(&Stats) -> f32| stats.iter().map(f).collect::<Vec<f32>>();
    let factor = if small { hratio * 0.25 } else { 1.0 };
    Ok(Wps281Fields {
        var: pick(&|s| if small { s.std * factor } else { s.std }),
        con: pick(&|s| s.con),
        oa: std::array::from_fn(|m| pick(&|s| s.oa[m])),
        ol: std::array::from_fn(|m| pick(&|s| s.ol[m])),
        elvmax: pick(&|s| s.max),
        box_first: (geometry[0].2 as usize, ny as usize),
        hratio,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pairwise_matches_numpy_block_rules() {
        let a: Vec<f32> = (0..300).map(|i| (i as f32) * 0.1 + 1.0e-3).collect();
        let s = np_sum(&a);
        let exact: f64 = a.iter().map(|&v| v as f64).sum();
        assert!(((s as f64) - exact).abs() < 1e-2);
    }

    #[test]
    fn flat_box_has_no_statistics() {
        let b = vec![100.0f32; 12 * 10];
        let s = box_stats(&b, 12, 10);
        assert_eq!(s.max, 100.0);
        assert_eq!(s.std, 0.0);
        assert_eq!(s.con, 0.0);
        assert_eq!(s.oa, [0.0; 4]);
        assert_eq!(s.ol, [0.0; 4]);
    }

    #[test]
    fn west_high_box_is_asymmetric_west() {
        let (ny, nx) = (6usize, 8usize);
        let b: Vec<f32> = (0..ny * nx).map(|k| if k % nx < nx / 2 { 500.0 } else { 0.0 }).collect();
        let s = box_stats(&b, ny, nx);
        assert_eq!(s.oa[0], 1.0);
        assert_eq!(s.max, 500.0);
    }
}
