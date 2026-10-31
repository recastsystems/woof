//! Thompson GOCART initialization, including the LBC reuse path.
//! RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_thompson_aerosols.F:36-865 at 1d6e370e.
//! Separate strict caller-state sibling; horizontal source and methods are shared in origin.
//! RKIND is binary32; only Gaussian latitude construction uses binary64.

use super::{
    hinterp::{self, InterpContext, Method, Slab, Underflow},

    vinterp::{carried_columns, AerosolWeights},
    DEG_PER_RAD,
};
use crate::init::{metfile::{MetReader, MetSlab}, soil};
use crate::error::{MpasError, MpasResult};
use rayon::prelude::*;
use std::cell::Cell;
use std::path::Path;

pub const NATIVE_FILENAME: &str = "QNWFA_QNIFA_SIGMA_MONTHLY.dat";
const MONTHS: [&str; 12] = [
    "JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC",
];
const METHODS: [Method; 4] = [
    Method::FourPoint,
    Method::WtAverage4,
    Method::WtAverage16,
    Method::Search,
];

#[derive(Debug, Default, serde::Serialize)]
pub struct AerosolReceipt {
    pub source: String,
    pub source_present: bool,
    pub records_read: usize,
    pub records_interpolated: usize,
    pub masked_cells: usize,
    pub n_gocart_levels: usize,
    pub values_computed: usize,
}

/// Layout: month outermost, then cell, with GOCART level fastest.
#[derive(Debug, Clone)]
pub struct Climatology {
    pub n_cells: usize,
    pub n_levels: usize,
    pub nifa: Vec<Vec<f32>>,
    pub nwfa: Vec<Vec<f32>>,
    pub pressure: Vec<Vec<f32>>,
    // get_julgmt DATA mmd keeps February=29 after the first leap-year call.
    february_is_29: Cell<bool>,
}

#[derive(Debug)]
pub struct AerosolState {
    /// Cell-major, model vertical level fastest, as the init stream carries it.
    pub nifa: Vec<f32>,
    pub nwfa: Vec<f32>,
}

/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_thompson_aerosols.F:360-383,673-689 at 1d6e370e.
/// Aerosols wrap with strict `>` and do NOT clamp latitude like case 7.
fn locate(field: &MetSlab, lat: f32, lon: f32, gaussian: Option<&[f32]>) -> MpasResult<(f32, f32)> {
    let loninc = if gaussian.is_some() {
        360.0 / field.nx as f32
    } else {
        field.delta_lon
    };
    let ij = |lon: f32| -> MpasResult<(f32, f32)> {
        let x = (lon - field.start_lon) / loninc + 1.0;
        let y = if let Some(lats) = gaussian {
            gaussian_j(lats, lat)?
        } else {
            (lat - field.start_lat) / field.delta_lat + 1.0
        };
        Ok((x, y))
    };
    let (x, y) = ij(lon)?;
    if x < 0.5 {
        ij(lon + 360.0)
    } else if x > field.nx as f32 + 0.5 {
        ij(lon - 360.0)
    } else {
        Ok((x, y))
    }
}

/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_llxy.F:2095-2140 (lgord) at 1d6e370e.
fn lgord(cosc: f64, n: usize) -> f64 {
    let colat = cosc.acos();
    let mut c1 = 2.0_f64.sqrt();
    for k in 1..=n {
        // This subexpression is default REAL in MPAS, even in the REAL64 routine.
        c1 *= (1.0_f32 - 1.0_f32 / (4 * k * k) as f32).sqrt() as f64;
    }
    let fn_ = n as f64;
    let mut ang = fn_ * colat;
    let (mut s1, mut c4, mut a, mut b) = (0.0, 1.0, -1.0, 0.0);
    for k in (0..=n).step_by(2) {
        if k == n {
            c4 *= 0.5;
        }
        s1 += c4 * ang.cos();
        a += 2.0;
        b += 1.0;
        ang = colat * (fn_ - k as f64 - 2.0);
        c4 = (a * (fn_ - b + 1.0) / (b * (fn_ + fn_ - a))) * c4;
    }
    s1 * c1
}

/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_llxy.F:1914-2092 (set_gauss, gausll, lggaus) at 1d6e370e.
/// Weight/colat outputs unused by gausll are omitted; root operations keep order.
pub fn gaussian_latitudes(nlat: usize, start_lat: f32) -> MpasResult<Vec<f32>> {
    if nlat < 2 || nlat % 2 != 0 {
        return Err(MpasError::Refusal(
            "Gaussian input requires two hemispheres of latitude rows".into(),
        ));
    }
    // Both MPAS REAL64 pi parameters are initialized by a default REAL literal.
    let fi = nlat as f64;
    let fi1 = fi + 1.0;
    let a = fi * fi1 / (4.0 * fi1 * fi1 - 1.0).sqrt();
    let b = fi1 * fi / (4.0 * fi * fi - 1.0).sqrt();
    let mut lats = vec![0.0; nlat];
    for i in 1..=nlat / 2 {
        // i-0.5 is default REAL before multiplication by pi.
        let mut root = (((i as f32 - 0.5) as f64) * (std::f32::consts::PI as f64) / fi
            + (std::f32::consts::PI as f64) * 0.5)
            .sin();
        loop {
            let g = lgord(root, nlat);
            let gm = lgord(root, nlat - 1);
            let gp = lgord(root, nlat + 1);
            let gt = (root * root - 1.0) / (a * gp - b * gm);
            let delta = g * gt;
            root -= delta;
            if delta.abs() <= 1.0e-14_f32 as f64 {
                break;
            }
        }
        let sinc = root.acos().sin();
        let lat = (sinc.acos() * 180.0 / (std::f32::consts::PI as f64)) as f32;
        lats[i - 1] = lat;
        lats[nlat - i] = -lat;
    }
    if (lats[0] - start_lat).abs() > 0.01 {
        for lat in &mut lats {
            *lat *= -1.0;
        }
    }
    if (lats[0] - start_lat).abs() > 0.01 {
        return Err(MpasError::Refusal(format!("Gaussian starting latitude {start_lat} disagrees with MPAS-computed {} by more than MPAS's 0.01 degree check", lats[0])));
    }
    Ok(lats)
}

/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_llxy.F:2143-2222 (llij_gauss) at 1d6e370e.
fn gaussian_j(lats: &[f32], lat: f32) -> MpasResult<f32> {
    if lat.abs() > lats[0].abs() {
        return Ok(
            if (lat - lats[0]).abs() < (lat - lats[lats.len() - 1]).abs() {
                1.0
            } else {
                lats.len() as f32
            },
        );
    }
    for n in 0..lats.len() - 1 {
        if (lats[n] - lat) * (lats[n + 1] - lat) <= 0.0 {
            return Ok(
                ((lats[n] - lat) * (n + 2) as f32 + (lat - lats[n + 1]) * (n + 1) as f32)
                    / (lats[n] - lats[n + 1]),
            );
        }
    }
    Err(MpasError::Refusal(
        "MPAS Gaussian grid has no bounding latitudes for this cell".into(),
    ))
}

/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_thompson_aerosols.F:317-359,389-662 at 1d6e370e.
fn destination(name: &str) -> Option<(usize, usize)> {
    for (family, prefix) in ["QNIFA_", "QNWFA_", "P_WIF_"].iter().enumerate() {
        for (month, suffix) in MONTHS.iter().enumerate() {
            if name.contains(&format!("{prefix}{suffix}")) {
                return Some((family, month));
            }
        }
    }
    None
}

/// One climatology record, read and checked, ready to sample.
struct GocartRecord { family: usize, month: usize, slot: usize, field: MetSlab, gaussian: Option<Vec<f32>>, slab: Slab }

/// The GOCART horizontal pass a month at a time: every record of one
/// (family, month) is read first, then each cell samples all of that month's
/// levels on its own core and writes its own contiguous run of levels.
/// Record by record, each of the 1,080 records rewrote one level of every
/// cell (a 120-byte stride through a 280 MB array on conus3km), so the pass
/// moved the whole climatology through memory a thousand times. Within a
/// cell the records still apply in file order, so a repeated level keeps the
/// last record's value, as before. Any refusal returns Err and the caller
/// reruns record by record for the exact first refusal.
fn grouped(path: &Path, lat: &[f32], lon: &[f32], landmask: &[i32], n_levels: usize,
    clim: &mut Climatology) -> MpasResult<(usize, usize)>
{
    let mut reader = MetReader::open(path, true)?;
    let mut records: Vec<GocartRecord> = Vec::new();
    let mut read = 0usize;
    while let Some((mut field, projection)) = reader.next_field()? {
        read += 1;
        let Some((family, month)) = destination(&field.field) else { continue; };
        let k = field.xlvl as i32; // Fortran REAL-to-INTEGER truncates.
        if k < 1 || k as usize > n_levels || field.nx < 3 {
            return Err(MpasError::Refusal("GOCART record outside its level/halo bounds".into()));
        }
        let gaussian = if projection == 4 {
            let nlat = field.delta_lat.round() as usize * 2;
            if nlat != field.ny {
                return Err(MpasError::Refusal("Gaussian header nlat does not match slab rows".into()));
            }
            Some(gaussian_latitudes(nlat, field.start_lat)?)
        } else {
            None
        };
        let slab = Slab::from_met(&field.values, field.nx, field.ny);
        field.values = Vec::new();
        records.push(GocartRecord { family, month, slot: k as usize - 1, field, gaussian, slab });
    }
    for family in 0..3 {
        for month in 0..12 {
            let group: Vec<&GocartRecord> = records.iter()
                .filter(|r| r.family == family && r.month == month).collect();
            if group.is_empty() { continue; }
            let contexts: Vec<InterpContext> = group.iter().map(|r| InterpContext {
                array: &r.slab,
                msgval: 0.0,
                maskval: None,
                mask: None,
                underflow: Underflow::Preserve,
                underflow_hits: None,
            }).collect();
            let dest = match family {
                0 => &mut clim.nifa,
                1 => &mut clim.nwfa,
                _ => &mut clim.pressure,
            };
            dest[month].par_chunks_mut(n_levels).enumerate().try_for_each(|(c, column)| {
                if landmask[c] == -1 {
                    for r in &group { column[r.slot] = 0.0; }
                    return Ok(());
                }
                let (cell_lat, cell_lon) = (lat[c] * DEG_PER_RAD, lon[c] * DEG_PER_RAD);
                for (r, ctx) in group.iter().zip(&contexts) {
                    let (x, y) = locate(&r.field, cell_lat, cell_lon, r.gaussian.as_deref())?;
                    column[r.slot] = hinterp::interp_sequence(x, y, ctx, &METHODS, 0);
                }
                Ok::<(), MpasError>(())
            })?;
        }
    }
    Ok((read, records.len()))
}

/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_thompson_aerosols.F:209-719 (init_hinterp_gocart) at 1d6e370e.
/// LANDSEA is read by native but maskval=-1 means it never affects sampling.
pub fn horizontal(
    path: &Path,
    lat: &[f32],
    lon: &[f32],
    landmask: &[i32],
    n_levels: usize,
) -> MpasResult<(Climatology, AerosolReceipt)> {
    let n_cells = lat.len();
    if lon.len() != n_cells || landmask.len() != n_cells || n_levels < 2 {
        return Err(MpasError::Refusal(
            "GOCART mesh array shapes or level count are inconsistent".into(),
        ));
    }
    let mut clim = Climatology {
        n_cells,
        n_levels,
        // One zeroed allocation per month (vec![v; 12] would copy the first
        // eleven times on one core: 9 GB of copies on conus3km).
        nifa: (0..12).map(|_| vec![0.0; n_cells * n_levels]).collect(),
        nwfa: (0..12).map(|_| vec![0.0; n_cells * n_levels]).collect(),
        pressure: (0..12).map(|_| vec![0.0; n_cells * n_levels]).collect(),
        february_is_29: Cell::new(false),
    };
    let mut receipt = AerosolReceipt {
        source: path.display().to_string(),
        source_present: true,
        n_gocart_levels: n_levels,
        masked_cells: landmask.iter().filter(|&&m| m == -1).count(),
        ..Default::default()
    };
    match grouped(path, lat, lon, landmask, n_levels, &mut clim) {
        Ok((read, interpolated)) => {
            receipt.records_read = read;
            receipt.records_interpolated = interpolated;
            return Ok((clim, receipt));
        }
        Err(_) => {
            // A refusal somewhere: rerun record by record from zeroed months,
            // so the refusal returned is the one that loop meets first.
            for months in [&mut clim.nifa, &mut clim.nwfa, &mut clim.pressure] {
                for month in months.iter_mut() { month.iter_mut().for_each(|v| *v = 0.0); }
            }
        }
    }
    let mut reader = MetReader::open(path, true)?;
    while let Some((field, projection)) = reader.next_field()? {
        receipt.records_read += 1;
        let Some((family, month)) = destination(&field.field) else {
            continue;
        };
        let k = field.xlvl as i32; // Fortran REAL-to-INTEGER truncates.
        if k < 1 || k as usize > n_levels || field.nx < 3 {
            return Err(MpasError::Refusal(format!("GOCART field {} has level {} or grid width {} outside its allocated level/halo bounds", field.field, field.xlvl, field.nx)));
        }
        let gaussian = if projection == 4 {
            let nlat = field.delta_lat.round() as usize * 2;
            if nlat != field.ny {
                return Err(MpasError::Refusal(
                    "Gaussian header nlat does not match slab rows".into(),
                ));
            }
            Some(gaussian_latitudes(nlat, field.start_lat)?)
        } else {
            None
        };
        let slab = Slab::from_met(&field.values, field.nx, field.ny);
        let ctx = InterpContext {
            array: &slab,
            msgval: 0.0,
            maskval: None,
            mask: None,
            underflow: Underflow::Preserve,
            underflow_hits: None,
        };
        let dest = match family {
            0 => &mut clim.nifa,
            1 => &mut clim.nwfa,
            _ => &mut clim.pressure,
        };
        // Each cell's sample depends on that cell alone, so the cells run on
        // every core (this loop was one core: about a tenth of the conus3km
        // init). A locate refusal is rare (a Gaussian row with no bounding
        // latitudes); when one occurs the serial loop reruns, so the refusal
        // returned is the first in cell order, as before.
        let slot = k as usize - 1;
        let sample = |c: usize| -> MpasResult<f32> {
            Ok(if landmask[c] == -1 {
                0.0
            } else {
                let (x, y) = locate(
                    &field,
                    lat[c] * DEG_PER_RAD,
                    lon[c] * DEG_PER_RAD,
                    gaussian.as_deref(),
                )?;
                hinterp::interp_sequence(x, y, &ctx, &METHODS, 0)
            })
        };
        let parallel: MpasResult<()> = dest[month].par_chunks_mut(n_levels).enumerate()
            .try_for_each(|(c, column)| { column[slot] = sample(c)?; Ok(()) });
        if parallel.is_err() {
            for c in 0..n_cells {
                dest[month][c * n_levels + slot] = sample(c)?;
            }
        }
        receipt.records_interpolated += 1;
    }
    Ok((clim, receipt))
}

/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_thompson_aerosols.F:87-205 (init_vinterp_gocart) at 1d6e370e.
/// Also the identical interpolation slice of init_atm_thompson_aerosols_lbc:790-844.
impl Climatology {
    /// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_thompson_aerosols.F:693-839 at 1d6e370e.
    pub fn at_date(
        &self,
        date: &str,
        pressure_base: &[f32],
        n_vert: usize,
    ) -> MpasResult<AerosolState> {
        let parse = |range: std::ops::Range<usize>| -> MpasResult<u32> {
            date.get(range).and_then(|s| s.parse().ok()).ok_or_else(|| {
                MpasError::Refusal(format!("invalid GOCART interpolation date {date}"))
            })
        };
        let (year, month, day) = (parse(0..4)? as i32, parse(5..7)?, parse(8..10)?);
        let leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
        let days = [
            31,
            if leap { 29 } else { 28 },
            31,
            30,
            31,
            30,
            31,
            31,
            30,
            31,
            30,
            31,
        ];
        if !(1..=12).contains(&month) || day < 1 || day > days[(month - 1) as usize] {
            return Err(MpasError::Refusal(format!(
                "invalid GOCART interpolation date {date}"
            )));
        }
        let size = self.n_cells * self.n_levels;
        if self.n_levels < 2
            || n_vert == 0
            || pressure_base.len() != self.n_cells * n_vert
            || [&self.nifa, &self.nwfa, &self.pressure]
                .iter()
                .any(|a| a.len() != 12 || a.iter().any(|v| v.len() != size))
        {
            return Err(MpasError::Refusal(
                "GOCART climatology and pressure_base have inconsistent array shapes".into(),
            ));
        }
        self.february_is_29.set(self.february_is_29.get() || leap);
        let native_leap = self.february_is_29.get();
        let p = soil::monthly_interp_with_february(&self.pressure, year, month, day, native_leap);
        let interpolate = |monthly: &[Vec<f32>]| {
            let v = soil::monthly_interp_with_february(monthly, year, month, day, native_leap);
            // The weights registers run on from cell to cell as in the native
            // serial loop; vinterp::carried_columns runs the cells on every
            // core and gives each cell the registers the serial loop would.
            let n_levels = self.n_levels;
            let column = |c: usize, weights: &mut AerosolWeights| -> Vec<f32> {
                let offset = c * n_levels;
                let coords: Vec<_> = p[offset + 1..offset + n_levels]
                    .iter()
                    .rev()
                    .copied()
                    .collect();
                let values: Vec<_> = v[offset + 1..offset + n_levels]
                    .iter()
                    .rev()
                    .copied()
                    .collect();
                let mut out = vec![0.0; n_vert];
                for k in (0..n_vert).rev() {
                    let target = pressure_base[c * n_vert + k];
                    // Constant extrapolation also supports a one-level retained column.
                    out[k] = if coords.len() == 1 {
                        values[0]
                    } else {
                        weights.interpolate(target, &coords, &values)
                    };
                    if target >= p[offset] && k + 1 < n_vert {
                        out[k] = out[k + 1];
                    }
                }
                out
            };
            let first = AerosolWeights(monthly_day_weights(year, month, day, native_leap));
            let columns = carried_columns(self.n_cells, first, &column);
            let mut out = vec![0.0; pressure_base.len()];
            out.par_chunks_mut(n_vert).zip(columns.par_iter())
                .for_each(|(slot, column)| slot.copy_from_slice(column));
            out
        };
        Ok(AerosolState {
            nifa: interpolate(&self.nifa),
            nwfa: interpolate(&self.nwfa),
        })
    }
}

/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_thompson_aerosols.F:36-81 at 1d6e370e.
/// No file means zero cold-start scalars, never the synthetic aerosol profile.
pub fn initialize(
    path: &Path,
    lat: &[f32],
    lon: &[f32],
    landmask: &[i32],
    n_levels: usize,
    date: &str,
    pressure_base: &[f32],
    n_vert: usize,
) -> MpasResult<(AerosolState, AerosolReceipt)> {
    if !path.try_exists()? {
        return Ok((
            AerosolState {
                nifa: vec![0.0; lat.len() * n_vert],
                nwfa: vec![0.0; lat.len() * n_vert],
            },
            AerosolReceipt {
                source: path.display().to_string(),
                ..Default::default()
            },
        ));
    }
    let started = std::time::Instant::now();
    let (clim, mut receipt) = horizontal(path, lat, lon, landmask, n_levels)?;
    eprintln!("[case7]   GOCART climatology read + horizontal: {:.3} s", started.elapsed().as_secs_f64());
    let started = std::time::Instant::now();
    let state = clim.at_date(date, pressure_base, n_vert)?;
    eprintln!("[case7]   GOCART vertical (at_date): {:.3} s", started.elapsed().as_secs_f64());
    receipt.values_computed = state.nifa.len() + state.nwfa.len();
    Ok((state, receipt))
}

/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_thompson_aerosols.F:722-862 at 1d6e370e.
/// Caller-owned cache mirrors mesh climatology arrays: reload at timestart,
/// reuse thereafter; a missing file leaves the caller's LBC state untouched.
pub fn initialize_lbc(
    path: &Path,
    timestamp: &str,
    timestart: &str,
    lat: &[f32],
    lon: &[f32],
    landmask: &[i32],
    n_levels: usize,
    pressure_base: &[f32],
    n_vert: usize,
    cache: &mut Option<Climatology>,
) -> MpasResult<Option<AerosolState>> {
    if !path.try_exists()? {
        return Ok(None);
    }
    if timestamp == timestart {
        let started = std::time::Instant::now();
        *cache = Some(horizontal(path, lat, lon, landmask, n_levels)?.0);
        eprintln!("[case9]   GOCART climatology read + horizontal: {:.3} s", started.elapsed().as_secs_f64());
    }
    let clim = cache.as_ref().ok_or_else(|| {
        MpasError::Refusal(
            "GOCART LBC requested after timestart without initialized mesh climatology".into(),
        )
    })?;
    Ok(Some(clim.at_date(timestamp, pressure_base, n_vert)?))
}

/// RRFSx/MPAS-Model src/core_atmosphere/physics/mpas_atmphys_date_time.F:148-219 at 1d6e370e.
/// The strict monthly routine retains the unnormalized day distances in xmm3
/// and xmm4. These seed each subsequent whole aerosol interpolation pass.
fn monthly_day_weights(year:i32, month:u32, day:u32, leap:bool)->[f32;2] {
    let days=[31,if leap {29}else{28},31,30,31,30,31,31,30,31,30,31];
    let julian=|m:usize,d:i32|days[..m-1].iter().sum::<i32>()+d;
    let mut middle=[0;14];
    for m in 1..=12 {middle[m]=year*1000+julian(m,15);}
    middle[0]=middle[1]-31; middle[13]=middle[12]+31;
    let target=year*1000+julian(month as usize,day as i32);
    for m in 0..=12 {
        if middle[m]<target && middle[m+1]>=target {
            return [(middle[m+1]-target) as f32,(target-middle[m]) as f32];
        }
    }
    unreachable!("validated calendar date always lies between monthly anchors")
}
