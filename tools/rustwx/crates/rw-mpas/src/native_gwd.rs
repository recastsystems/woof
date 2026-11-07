//! Native MPAS-A v8.4.2 orographic statistics, RKIND=binary32.
//!
//! Source: src/core_init_atmosphere/mpas_init_atm_gwd.F:8-1255 at f34984b.
//! All active fields are ported together. The commented-out elvmax/theta/gamma/
//! sigma calls (lines 300-303) do not produce fields in native init_atmosphere.
//! Geography decoding remains the existing bounded Rust reader. Native uses
//! the fixed 43200x21600 frame, not index-header pixel centres; GMTED2010 is
//! shifted half a globe, independently of the USGS land-use frame.

use crate::error::{MpasError, MpasResult};
use crate::static_geog::GeogDataset;
use crate::static_gwd::GwdFields;
use rayon::prelude::*;
use serde::Serialize;
use std::path::{Path, PathBuf};

pub const NX: i64 = 43200;
pub const NY: i64 = 21600;
pub const RE: f32 = 6371229.0;
pub const PI: f32 = std::f32::consts::PI;
pub const RAD2DEG: f32 = 180.0 / PI;
const PPD: f32 = 120.0;
/// Source: mpas_init_atm_gwd.F:73, including native constant-folding order.
pub const SG_DELTA: f32 = ((2.0 * PI) * RE) / (360.0 * PPD);

/// MPAS dataset selection, mpas_init_atm_gwd.F:183-204.
/// No model/suite default is changed by this option.
#[derive(Debug, Clone)]
pub struct NativeGwdConfig {
    pub topo: PathBuf,
    pub landuse: PathBuf,
    pub topo_shift: i64,
    pub cell_scaling: f32,
    pub host_memory_limit_bytes: Option<u64>,
    pub tile_workers: Option<usize>,
    /// Other static stages resident concurrently with the native GWD pass.
    pub extra_resident_bytes: u64,
}
impl NativeGwdConfig {
    pub fn under(root: &Path, topo_data: &str, cell_scaling: f32) -> MpasResult<Self> {
        const SOURCES: [(&str, &str, i64); 2] = [
            ("GTOPO30", "topo_30s", 0),
            ("GMTED2010", "topo_gmted2010_30s", NX / 2),
        ];
        let (_, subpath, topo_shift) = SOURCES.iter().find(|(name, _, _)| *name == topo_data)
            .ok_or_else(|| MpasError::Refusal(format!("invalid config_topo_data {topo_data:?}; native GWD supports GTOPO30 and GMTED2010")))?;
        if !cell_scaling.is_finite() || cell_scaling <= 0.0 {
            return Err(MpasError::Refusal("config_gwd_cell_scaling must be finite and positive; otherwise native boxes are empty or unbounded".into()));
        }
        Ok(Self {
            topo: root.join(subpath),
            landuse: root.join("landuse_30s"),
            topo_shift: *topo_shift,
            cell_scaling,
            host_memory_limit_bytes: None,
            tile_workers: None,
            extra_resident_bytes: 0,
        })
    }
}

/// Source: mpas_init_atm_gwd.F:350-370, preserving expression evaluation order.
pub fn box_size(lat: f32, dx: f32) -> (i64, i64) {
    let cos_lat = lat.cos();
    let nx = if cos_lat > (((2.0 * PPD) * dx) * 180.0) / ((NX as f32 * PI) * RE) {
        (((180.0 * dx) * PPD) / ((PI * RE) * cos_lat)).ceil() as i64
    } else {
        NX / 2
    };
    let ny = (((180.0 * dx) * PPD) / (PI * RE)).ceil() as i64;
    (nx, ny)
}

/// Source: mpas_init_atm_gwd.F:730-758 (NINT uses ties away from zero).
pub fn centre(lat: f32, lon: f32) -> (i64, i64) {
    let mut ic = (((lon * RAD2DEG) - (-180.0)) * PPD).round() as i64 + 1;
    let jc = (((lat * RAD2DEG) - (-90.0)) * PPD).round() as i64 + 1;
    if ic <= 0 {
        ic += NX;
    }
    if ic > NX {
        ic -= NX;
    }
    (ic, jc)
}

/// Source: mpas_init_atm_gwd.F:767-785. Return canonical USGS indices.
pub fn wrap(mut ii: i64, mut jj: i64) -> (i64, i64) {
    if jj <= 0 {
        jj = -jj + 1;
        ii += NY;
    }
    if jj > NY {
        jj = NY - (jj - NY - 1);
        ii += NY;
    }
    ((ii - 1).rem_euclid(NX) + 1, jj)
}

/// Band coverage of the native gather (mpas_init_atm_gwd.F:763-785).
/// Includes the last row and rows reflected over both poles.
pub fn reflected_rows(jc: i64, ny: i64) -> (i64, i64) {
    let j0 = jc - ny / 2 + 1;
    let j1 = j0 + ny - 1;
    let mut lo = i64::MAX;
    let mut hi = i64::MIN;
    let mut note = |a: i64, b: i64| {
        if b >= a {
            lo = lo.min(a);
            hi = hi.max(b);
        }
    };
    note(j0.max(1), j1.min(NY));
    if j0 <= 0 {
        note(1, 1 - j0);
    }
    if j1 > NY {
        note(2 * NY + 1 - j1, NY);
    }
    (lo, hi)
}

/// Source: mpas_init_atm_gwd.F:263-271; cast before each sum, not after it.
pub fn diameter(edges: &[f32], sphere_radius: f32, on_unit_sphere: bool, scaling: f32) -> f32 {
    let mut dc = 0.0f32;
    for &edge in edges {
        dc += edge;
    }
    dc /= edges.len() as f32;
    if on_unit_sphere {
        dc *= sphere_radius;
    }
    dc * scaling
}

#[derive(Debug, Clone, Copy)]
pub struct CellStats {
    pub var2d: f32,
    pub con: f32,
    pub oa: [f32; 4],
    pub ol: [f32; 4],
    pub land: u8,
}

/// Entire active reduction, not separately selectable routines.
/// Source: get_box mean mpas_init_atm_gwd.F:760-802;
/// get_dom_landmask :666-693; get_var :818-839; get_con :852-919;
/// get_oa1/2/3/4 :934-1099; critical height :291;
/// get_ol1/2/3/4 :1112-1255. Input storage is Fortran i-fast order.
pub fn reduce_box(topo: &[f32], landuse: &[i32], nx: i64, ny: i64) -> MpasResult<CellStats> {
    if nx < 1
        || ny < 1
        || nx.checked_mul(ny) != Some(topo.len() as i64)
        || topo.len() != landuse.len()
    {
        return Err(MpasError::Refusal("native GWD requires nonempty, dimensionally complete terrain and land-use boxes".into()));
    }
    let n = (nx * ny) as f32;
    let mut mean = 0.0f32;
    for &v in topo {
        mean += v;
    }
    mean /= n;
    let (mut s2, mut s4, mut mean_land, mut mean_water, mut xland) =
        (0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32);
    for (&v, &lu) in topo.iter().zip(landuse) {
        if lu != 16 {
            xland += 1.0;
            mean_land += v;
        } else {
            mean_water += v;
        }
    }
    if xland > 0.0 {
        mean_land /= xland;
    }
    if xland < n {
        mean_water /= n - xland;
    }
    xland /= n;
    let oro = if xland >= 0.5 { mean_land } else { mean_water };
    for &v in topo {
        let d = v - mean;
        s2 += d * d;
        let d = v - oro;
        let square = d * d;
        s4 += square * square;
    }
    let variance = s2 / n;
    let var2d = variance.sqrt();
    let mut con = if var2d < 1.0 {
        0.0
    } else {
        s4 / ((variance * variance) * n)
    };
    if xland < 0.5 {
        con = 0.0;
    }
    let ratio = ny as f32 / nx as f32;
    let hc = 1116.2f32 - 0.878f32 * var2d;
    let mut nu = [0i32; 4];
    let mut nd = [0i32; 4];
    let mut nw = [0i32; 4];
    let mut nt = [0i32; 4];
    for j in 1..=ny {
        for i in 1..=nx {
            let v = topo[((j - 1) * nx + i - 1) as usize];
            let split = [
                i <= nx / 2,
                j <= ny / 2,
                ((i as f32 * ratio).round() as i64) < ny - j,
                ((i as f32 * ratio).round() as i64) < j,
            ];
            let include = [
                // :1126-1128 reads an entire row zero for ny<4. Repeated
                // strict native runs vary there; use the central in-bounds
                // rows and declare exactly those OL1 words undefined.
                j >= (ny / 4).max(1) && j <= (3 * ny / 4).max(1),
                i >= nx / 4 && i <= 3 * nx / 4,
                (j <= ny / 2 && i <= nx / 2) || (j > ny / 2 && i > nx / 2),
                (j > ny / 2 && i <= nx / 2) || (j <= ny / 2 && i > nx / 2),
            ];
            for k in 0..4 {
                if v > mean {
                    if split[k] {
                        nu[k] += 1;
                    } else {
                        nd[k] += 1;
                    }
                }
                if include[k] {
                    nt[k] += 1;
                    if v > hc {
                        nw[k] += 1;
                    }
                }
            }
        }
    }
    if nx < 4 {
        // :1164-1166's column zero aliases the previous row's last element
        // for j>1. For j=1 it is the zero high word of glibc's allocation
        // size header. Strict native repeats reproduce this OL2 behavior.
        for j in 1..=ny {
            let v = if j == 1 { 0.0 } else { topo[((j - 1) * nx - 1) as usize] };
            nt[1] += 1;
            if v > hc { nw[1] += 1; }
        }
    }
    Ok(CellStats {
        var2d,
        con,
        oa: std::array::from_fn(|k| {
            if nu[k] + nd[k] > 0 {
                (nu[k] - nd[k]) as f32 / (nu[k] + nd[k]) as f32
            } else {
                0.0
            }
        }),
        ol: std::array::from_fn(|k| nw[k] as f32 / nt[k] as f32),
        land: u8::from(xland >= 0.5),
    })
}

/// Dormant native diagnostics, mpas_init_atm_gwd.F:1268-1449.
/// Native's calls at :300-303 are commented out; these are not output fields.
#[derive(Debug, Clone, Copy)]
pub struct DormantDiagnostics {
    pub elvmax: f32,
    pub htheta: f32,
    pub hgamma: f32,
    pub hsigma: f32,
}

/// Source: get_box metric, mpas_init_atm_gwd.F:794-795.
pub fn pixel_dx(jj: i64) -> f32 {
    let sg_lat = (-90.0 + ((jj - 1) as f32 + 0.5) / PPD) / RAD2DEG;
    SG_DELTA * sg_lat.cos()
}

/// Full dormant get_elvmax/get_htheta/get_hgamma/get_hsigma reductions.
/// Source: mpas_init_atm_gwd.F:1268-1287, :1300-1339, :1352-1396,
/// :1409-1449. Identical gradient sums are shared without reordering.
pub fn dormant_diagnostics(
    topo: &[f32],
    dxm: &[f32],
    nx: usize,
    ny: usize,
) -> MpasResult<DormantDiagnostics> {
    if nx < 3
        || ny < 3
        || nx.checked_mul(ny) != Some(topo.len())
        || topo.len() != dxm.len()
        || dxm.iter().any(|&v| !v.is_finite() || v <= 0.0)
    {
        return Err(MpasError::Refusal("native GWD gradient diagnostics need complete nx,ny >= 3 boxes and positive finite zonal pixel metrics; otherwise native divides by zero or indexes outside the box".into()));
    }
    let mut elvmax = topo[0];
    for &v in topo {
        if v > elvmax {
            elvmax = v;
        }
    }
    let (mut hx2, mut hy2, mut hxy) = (0.0f32, 0.0f32, 0.0f32);
    for j in 1..ny - 1 {
        for i in 1..nx - 1 {
            let k = j * nx + i;
            let xfp = (topo[k + 1] - topo[k - 1]) / (2.0 * dxm[k]);
            let yfp = (topo[k + nx] - topo[k - nx]) / (2.0 * SG_DELTA);
            hx2 += xfp * xfp;
            hy2 += yfp * yfp;
            hxy += xfp * yfp;
        }
    }
    let n = ((nx - 2) * (ny - 2)) as f32;
    hx2 /= n;
    hy2 /= n;
    hxy /= n;
    let hk = 0.5 * (hx2 + hy2);
    let hl = 0.5 * (hx2 - hy2);
    let hlp = (hl * hl + hxy * hxy).sqrt();
    Ok(DormantDiagnostics {
        elvmax,
        htheta: 0.5 * hxy.atan2(hl),
        hgamma: if hk + hlp > 0.0 && hk - hlp >= 0.0 {
            ((hk - hlp) / (hk + hlp)).sqrt()
        } else {
            0.0
        },
        hsigma: (hk + hlp).sqrt(),
    })
}

/// Source: mpas_init_atm_gwd.F:307-326. Native smooth is sequential/in-place.
pub fn smooth(
    var: &mut [f32],
    land: &[u8],
    neoc: &[usize],
    coc: &[usize],
    max_edges: usize,
) -> MpasResult<usize> {
    if land.len() != var.len() || neoc.len() != var.len()
        || coc.len() != var.len() * max_edges || neoc.iter().any(|&v| v > max_edges) {
        return Err(MpasError::Refusal("native GWD smoothing storage does not match mesh dimensions".into()));
    }
    let mut count = 0;
    for c in 0..var.len() {
        let mut sum_land = 0i8;
        let mut sum_var = 0.0f32;
        for e in 0..neoc[c] {
            let nb = coc[c * max_edges + e];
            if nb >= var.len() {
                // :218 allocates hlanduse(nCells+1), :313-314 reads the
                // absent-neighbour sentinel. Strict regional repeats on
                // both hosts and 1/4 ranks reproduce zero mask/variance
                // contributions; preserve this effective sentinel behavior.
                sum_var += 0.0;
                continue;
            }
            sum_land = sum_land.wrapping_add(land[nb] as i8);
            sum_var += var[nb];
        }
        if (sum_land == neoc[c] as i8 && land[c] == 0) || (sum_land == 0 && land[c] == 1) {
            var[c] = sum_var / neoc[c] as f32;
            count += 1;
        }
    }
    Ok(count)
}

/// Exact values read by compute_gwd_fields, mpas_init_atm_gwd.F:218-271.
#[derive(Debug)]
pub struct NativeMesh {
    pub lat: Vec<f32>,
    pub lon: Vec<f32>,
    pub dc: Vec<f32>,
    pub neoc: Vec<usize>,
    pub coc: Vec<usize>,
    pub max_edges: usize,
}
impl NativeMesh {
    pub fn read(grid: &Path, scaling: f32) -> MpasResult<Self> {
        let f = netcrust::File::open(grid)?;
        let n = f
            .dimension("nCells")
            .ok_or_else(|| MpasError::Refusal("grid lacks nCells".into()))?
            .len();
        let max_edges = f
            .dimension("maxEdges")
            .ok_or_else(|| MpasError::Refusal("grid lacks maxEdges".into()))?
            .len();
        let read = |name: &str| -> MpasResult<Vec<f32>> {
            Ok(f.read_f64(name)?.into_iter().map(|v| v as f32).collect())
        };
        let lat = read("latCell")?;
        let lon = read("lonCell")?;
        let edges = read("dcEdge")?;
        let on_unit = edges.iter().copied().fold(f32::NEG_INFINITY, f32::max) < 1.0;
        let radius = f
            .attribute("sphere_radius")
            .and_then(|a| a.as_f64())
            .unwrap_or(RE as f64) as f32;
        // Native init config supplies Earth radius even for unit-sphere grids.
        let radius = if radius < 1.0 + f32::EPSILON {
            RE
        } else {
            radius
        };
        let neoc: Vec<usize> = f
            .read_f64("nEdgesOnCell")?
            .iter()
            .map(|&v| v as usize)
            .collect();
        let eoc = f.read_f64("edgesOnCell")?;
        let coc_raw = f.read_f64("cellsOnCell")?;
        if lat.len() != n
            || lon.len() != n
            || neoc.len() != n
            || eoc.len() != n * max_edges
            || coc_raw.len() != n * max_edges
        {
            return Err(MpasError::Refusal(
                "native GWD mesh dimensions do not match connectivity/coordinate arrays".into(),
            ));
        }
        if lat.iter().any(|v| !v.is_finite() || v.abs() > PI / 2.0)
            || lon.iter().any(|v| !v.is_finite() || v.abs() > 2.0 * PI)
            || edges.iter().any(|v| !v.is_finite() || *v <= 0.0)
        {
            return Err(MpasError::Refusal("native GWD needs finite on-sphere coordinates and positive finite dcEdge; malformed values would create invalid boxes".into()));
        }
        let mut dc = Vec::with_capacity(n);
        let mut coc = vec![usize::MAX; n * max_edges];
        for c in 0..n {
            if neoc[c] == 0 || neoc[c] > max_edges {
                return Err(MpasError::Refusal(
                    "native GWD cell has no edges or exceeds maxEdges".into(),
                ));
            }
            let mut ee = Vec::with_capacity(neoc[c]);
            for e in 0..neoc[c] {
                let k = c * max_edges + e;
                let edge = eoc[k];
                if edge < 1.0 || edge > edges.len() as f64 {
                    return Err(MpasError::Refusal(
                        "native GWD edgesOnCell references an absent edge".into(),
                    ));
                }
                ee.push(edges[edge as usize - 1]);
                let nb = coc_raw[k];
                if nb >= 1.0 && nb <= n as f64 {
                    coc[k] = nb as usize - 1;
                }
            }
            dc.push(diameter(&ee, radius, on_unit, scaling));
        }
        Ok(Self {
            lat,
            lon,
            dc,
            neoc,
            coc,
            max_edges,
        })
    }
    pub fn minimum_band_rows(&self) -> MpasResult<i64> {
        let mut most = 1;
        for c in 0..self.lat.len() {
            let (nx, ny) = box_size(self.lat[c], self.dc[c]);
            if nx < 1 || ny < 1 || ny > NY {
                return Err(MpasError::Refusal(
                    "native GWD box must be nonempty and may reflect across at most one pole"
                        .into(),
                ));
            }
            let (_, jc) = centre(self.lat[c], self.lon[c]);
            let (lo, hi) = reflected_rows(jc, ny);
            most = most.max(hi - lo + 1);
        }
        Ok(most)
    }
}

/// MPAS tile layout, mpas_init_atm_gwd.F:536-654. Reject incompatible decoding
/// instead of silently applying scale factors or a MODIS water convention.
fn validate_dataset(ds: &GeogDataset, topo: bool) -> MpasResult<()> {
    let i = &ds.index;
    if ds.nx_global != NX
        || ds.ny_global != NY
        || i.tile_x != 1200
        || i.tile_y != 1200
        || i.nz() != 1
        || i.wordsize != if topo { 2 } else { 1 }
        || i.tile_bdr != if topo { 3 } else { 0 }
        || !i.big_endian
        || i.row_order_top_bottom
        || i.scale_factor != 1.0
        || (topo && !i.signed)
        // Native WATER=16 is compiled into this module (:78); NCAR's actual
        // USGS index omits iswater. A contradictory explicit category is not
        // the native dataset, but omitted optional metadata is valid.
        || (!topo && i.iswater.unwrap_or(16) != 16)
    {
        return Err(MpasError::Refusal(format!(
            "{} does not have MPAS's native 30s {} tile layout; applying another decode or water convention changes drag statistics",
            ds.path.display(),
            if topo {
                "signed terrain"
            } else {
                "USGS land-use (WATER=16)"
            }
        )));
    }
    Ok(())
}

/// Complete compute_gwd_fields data path, mpas_init_atm_gwd.F:114-333.
/// Tile decoding and band allocation are the seeded Rust implementation.
/// The band stores the native canonical longitude frame: GMTED shifts the
/// terrain lookup, never the land-use lookup (get_tile_from_box_point :399-408).
pub fn compute(
    cfg: &NativeGwdConfig,
    mesh: &NativeMesh,
    band_rows: i64,
    progress: &dyn Fn(&str),
) -> MpasResult<GwdFields> {
    let topo = GeogDataset::open(&cfg.topo)?;
    let lu = GeogDataset::open(&cfg.landuse)?;
    validate_dataset(&topo, true)?;
    validate_dataset(&lu, false)?;
    let needed = mesh.minimum_band_rows()?;
    if band_rows < needed {
        return Err(MpasError::Refusal(format!(
            "native GWD granted {band_rows} rows but complete reflected boxes require {needed}; shorter bands would substitute zero terrain"
        )));
    }
    let n = mesh.lat.len();
    let max_box_bytes = (0..n)
        .map(|c| {
            let (nx, ny) = box_size(mesh.lat[c], mesh.dc[c]);
            (nx * ny) as u64 * 8
        })
        .max()
        .unwrap_or(0);
    let resident = cfg.extra_resident_bytes + n as u64 * (192 + mesh.max_edges as u64 * 8);
    let decode_scratch = topo.plane_bytes() + topo.index.plane_words() as u64 * 8;
    let band_bytes = band_rows.min(NY) as u64 * NX as u64 * 8;
    let requested = cfg
        .tile_workers
        .or_else(|| {
            std::env::var("RW_MPAS_TILE_WORKERS")
                .ok()
                .and_then(|v| v.parse().ok())
        })
        .unwrap_or_else(rayon::current_num_threads);
    let plan = crate::static_memory::plan_tile_parallelism(
        "native-gwd",
        resident,
        max_box_bytes,
        band_bytes + decode_scratch,
        16 * 1024 * 1024,
        requested,
        0,
        cfg.host_memory_limit_bytes,
    )?;
    let pool = rayon::ThreadPoolBuilder::new()
        .stack_size(crate::execution::BRIDGE_STACK_BYTES)
        .num_threads(plan.workers)
        .build()
        .map_err(|e| MpasError::Refusal(format!("cannot create native GWD worker pool: {e}")))?;
    let geometry: Vec<_> = (0..n)
        .map(|c| {
            let (nx, ny) = box_size(mesh.lat[c], mesh.dc[c]);
            let (ic, jc) = centre(mesh.lat[c], mesh.lon[c]);
            (ic, jc, nx, ny)
        })
        .collect();
    let mut order: Vec<_> = (0..n).collect();
    order.sort_by_key(|&c| (reflected_rows(geometry[c].1, geometry[c].3).0, c));
    let mut stats = vec![
        CellStats {
            var2d: 0.0,
            con: 0.0,
            oa: [0.0; 4],
            ol: [0.0; 4],
            land: 0
        };
        n
    ];
    let (mut pos, mut bands, mut peak_bytes) = (0, 0, 0);
    while pos < n {
        let c = order[pos];
        let (mut lo, mut hi) = reflected_rows(geometry[c].1, geometry[c].3);
        let mut end = pos + 1;
        while end < n {
            let c = order[end];
            let (a, b) = reflected_rows(geometry[c].1, geometry[c].3);
            if hi.max(b) - lo.min(a) + 1 > band_rows {
                break;
            }
            lo = lo.min(a);
            hi = hi.max(b);
            end += 1;
        }
        // NaN/-1 detect holes rather than passing them off as native terrain.
        let t = topo.read_band_with(lo, hi, 0, f32::NAN, |raw| raw as f32)?;
        let l = lu.read_band_with(lo, hi, 0, -1i32, |raw| raw as i8 as i32)?;
        peak_bytes = peak_bytes.max((t.len() + l.len()) as u64 * 4);
        bands += 1;
        progress(&format!(
            "GWDBAND\t{lo}\t{hi}\t{}\t{}",
            hi - lo + 1,
            end - pos
        ));
        let results: MpasResult<Vec<_>> = pool.install(|| order[pos..end].par_iter().map(|&c| {
            let (ic, jc, nx, ny) = geometry[c];
            let mut tt = vec![0.0f32; (nx * ny) as usize]; let mut ll = vec![0i32; tt.len()];
            for j in 1..=ny { for i in 1..=nx {
                let (ii, jj) = wrap(i - nx / 2 + ic, j - ny / 2 + jc);
                let topo_i = (ii - cfg.topo_shift - 1).rem_euclid(NX);
                let row = (jj - lo) * NX;
                let k = ((j-1)*nx+i-1) as usize;
                tt[k] = t[(row + topo_i) as usize]; ll[k] = l[(row + ii-1) as usize];
                if tt[k].is_nan() || ll[k] == -1 { return Err(MpasError::Refusal(format!("native GWD source tile missing at pixel ({ii},{jj}); zero-fill would create false drag fields"))); }
            } }
            Ok((c, reduce_box(&tt, &ll, nx, ny)?))
        }).collect());
        for (c, s) in results? {
            stats[c] = s;
        }
        pos = end;
    }
    let land: Vec<_> = stats.iter().map(|s| s.land).collect();
    let mut var: Vec<_> = stats.iter().map(|s| s.var2d).collect();
    let smoothed = smooth(&mut var, &land, &mesh.neoc, &mesh.coc, mesh.max_edges)?;
    Ok(GwdFields {
        var2d: var.into_iter().map(f64::from).collect(),
        con: stats.iter().map(|s| s.con as f64).collect(),
        oa: std::array::from_fn(|k| stats.iter().map(|s| s.oa[k] as f64).collect()),
        ol: std::array::from_fn(|k| stats.iter().map(|s| s.ol[k] as f64).collect()),
        hlanduse: land,
        smoothed_cells: smoothed,
        smooth_skipped_boundary_cells: 0,
        water_category: 16,
        water_category_source: "MPAS native USGS landuse_30s WATER=16".into(),
        band_rows,
        bands,
        band_bytes: peak_bytes,
    })
}

/// Write the ten-field native GWD overlay for the static-init composition seam.
/// Source fields: mpas_init_atm_gwd.F:224-235. Other static fields are owned by i2.
#[derive(Debug, Serialize)]
pub struct OverlayReceipt {
    pub schema: &'static str,
    pub grid_sha256: String,
    pub output_sha256: String,
    pub n_cells: usize,
    pub bands: usize,
    pub band_rows: i64,
    pub smoothed_cells: usize,
    pub topo_shift: i64,
    pub cell_scaling: f32,
    pub rkind_bytes: u8,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub native_undefined_ol1_cells: Vec<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub native_undefined_exclusion_status: Option<&'static str>,
}
pub fn write_overlay(
    grid: &Path,
    out: &Path,
    cfg: &NativeGwdConfig,
    clobber: bool,
    progress: &dyn Fn(&str),
) -> MpasResult<OverlayReceipt> {
    use rw_store::netcdf_classic::{
        NcAttr, NcClassicWriter, NcData, NcDim, NcFormat, NcType, NcVarDef,
    };
    if out.exists() && !clobber {
        return Err(MpasError::Refusal(format!(
            "{} already exists; pass --clobber to replace it",
            out.display()
        )));
    }
    let mesh = NativeMesh::read(grid, cfg.cell_scaling)?;
    let rows = mesh.minimum_band_rows()?.max(768);
    let fields = compute(cfg, &mesh, rows, progress)?;
    let names = [
        "var2d", "con", "oa1", "oa2", "oa3", "oa4", "ol1", "ol2", "ol3", "ol4",
    ];
    let vars = names
        .iter()
        .map(|&name| NcVarDef::new(name, NcType::Float, vec![0]))
        .collect();
    let undefined_ol1: Vec<usize> = (0..mesh.lat.len())
        .filter(|&c| box_size(mesh.lat[c], mesh.dc[c]).1 < 4).collect();
    let mut attrs = vec![
        NcAttr::text("rw_mpas_gwd_role", "native-gwd-overlay-not-runnable-static"),
        NcAttr::text("rw_mpas_gwd_source", "MPAS-A v8.4.2 mpas_init_atm_gwd.F f34984b RKIND=4"),
    ];
    if !undefined_ol1.is_empty() {
        attrs.push(NcAttr::text("rw_mpas_native_undefined", "EXCLUSIONS PROPOSED FOR OWNER SIGN-OFF: OL1 row zero, mpas_init_atm_gwd.F:1126-1128; case-specific sign-off required"));
        attrs.push(NcAttr::text("rw_mpas_native_undefined_ol1_cells", undefined_ol1.iter().map(usize::to_string).collect::<Vec<_>>().join(",")));
    }
    let mut w = NcClassicWriter::create(
        out,
        NcFormat::Offset64,
        vec![NcDim::fixed("nCells", mesh.lat.len())],
        attrs,
        vars,
        0,
    )?;
    for (name, values) in [("var2d", &fields.var2d), ("con", &fields.con)] {
        w.put(
            name,
            NcData::Floats(&values.iter().map(|&v| v as f32).collect::<Vec<_>>()),
        )?;
    }
    for k in 0..4 {
        w.put(
            &format!("oa{}", k + 1),
            NcData::Floats(&fields.oa[k].iter().map(|&v| v as f32).collect::<Vec<_>>()),
        )?;
        w.put(
            &format!("ol{}", k + 1),
            NcData::Floats(&fields.ol[k].iter().map(|&v| v as f32).collect::<Vec<_>>()),
        )?;
    }
    w.finish()?;
    Ok(OverlayReceipt {
        schema: "rw-mpas-native-gwd/v1",
        grid_sha256: crate::sha256_file(grid)?,
        output_sha256: crate::sha256_file(out)?,
        n_cells: mesh.lat.len(),
        bands: fields.bands,
        band_rows: rows,
        smoothed_cells: fields.smoothed_cells,
        topo_shift: cfg.topo_shift,
        cell_scaling: cfg.cell_scaling,
        rkind_bytes: 4,
        native_undefined_exclusion_status: (!undefined_ol1.is_empty())
            .then_some("PROPOSED FOR OWNER SIGN-OFF: case-specific sign-off required"),
        native_undefined_ol1_cells: undefined_ol1,
    })
}
