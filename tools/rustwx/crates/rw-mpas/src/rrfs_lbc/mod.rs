//! RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_cases.F:7759-8986 at 1d6e370e.
//! Complete case-9 first-guess boundary producer. The reader, projections,
//! optional hydrometeors, number concentrations and aerosol streams follow
//! the fork. Horizontal interpolation and saturation arithmetic are shared
//! only where the source-identity receipt records identical native sources.
//! Wind extrapolation is constant. Hydrometeor flags control emitted fields;
//! workflow metadata is supplied explicitly by the table-selected caller.
//! Outer edges retain native zero-neighbour arithmetic. Its strict binary
//! witness remains part of the required full init/forecast proof.

#![allow(clippy::needless_range_loop)]

pub use crate::lbc::{compare,sphere};
pub mod emit;
use crate::rrfs_init::{scalars,projection,scatter_level,sample_points,zero_columns};

use std::path::PathBuf;

use rayon::prelude::*;

use crate::error::{MpasError, MpasResult};
use crate::init::capsule::{self, MeshGeometry, VerticalMetrics};
use crate::init::dynamics::constants::{CP, CV, GRAVITY, P0, RGAS, T0B};
use crate::init::dynamics::rslf;
use crate::init::hinterp::{
    interp_sequence, InterpContext, Method, Slab, Underflow,
};
use crate::rrfs_init::metfile::{read_lbc_met_file, MetSlab};
use crate::init::vinterp::{
    sorted_column, vertical_interp, Extrap, SURFACE_LEVEL_TAG, SURFACE_SENTINEL,
};
use crate::init::DEG_PER_RAD;

const MSGVAL: f32 = -1.0e30;

/// The literal `1.61` the Fortran uses in the hydrostatic loop, in `theta_m`
/// and in the recovery of `lbc_theta`, not `rvord - 1`.  See
/// `init::dynamics` for the history of that mix; the lbc case carries no
/// switch because `init_atm_case_lbc` has no `rvord - 1` site at all.
const VIRTUAL_161: f32 = 1.61;

/// The nine met fields `init_atm_case_lbc` consumes.  Everything else in the
/// intermediate file is read past and reported, never guessed at.
const LBC_MET_FIELDS: [&str; 9] = [
    "UU", "VV", "TT", "RH", "SPECHUMD", "GHT", "SOILHGT", "PRES", "PRESSURE",
];

/// One boundary time to produce: the valid time and the intermediate file
/// carrying that time's first guess.  Which source model produced the
/// intermediate is invisible here by design: cadence and availability are
/// the caller's registry data, and the file format is the source-agnostic WPS
/// intermediate the arbitrary ingest already emits for every registered
/// source.
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BoundaryInterval {
    /// `YYYY-MM-DD_HH:MM:SS`.
    pub valid_time: String,
    pub met_path: PathBuf,
}

/// Everything the caller must state.  As with `rw_mpas_init`, the switches
/// that select physics have no defaults.
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LbcConfig {
    pub hydrometeors_rrfs: bool,
    pub hydrometeors_gfs: bool,
    pub tempo_rap: bool,
    pub aerosolaware: bool,
    pub smoke: bool,
    pub dust: bool,
    pub gocart_path: Option<PathBuf>,
    pub gocart_levels: usize,
    /// The regional initial-conditions file: mesh geometry, vertical metrics
    /// and the parent identity all come from here, exactly as native case 9
    /// reads its `input` stream.
    pub grid_path: PathBuf,
    pub out_dir: PathBuf,
    /// `config_start_time`: the epoch of the `Time` axis.
    pub start_time: String,
    /// `config_stop_time`, stamped into the header.
    pub stop_time: String,
    pub intervals: Vec<BoundaryInterval>,
    pub n_fg_levels: usize,
    #[serde(deserialize_with="crate::rrfs_init::job::extrap") ]
    pub extrap_airtemp: Extrap,
    pub use_spechumd: bool,
    pub theta_adv_order: i32,
    pub coef_3rd_order: f32,
    /// `config_fg_interval` in seconds, stamped into the header.  The data in
    /// each file depends only on that file's own valid time; the interval is
    /// registry metadata carried for the consumer.
    pub fg_interval_seconds: i64,
    #[serde(deserialize_with="crate::rrfs_init::job::underflow") ]
    pub oned_underflow: Underflow,
    /// Overrides for header config attributes that are namelist metadata
    /// rather than producer switches (met prefix, geography table names, ...).
    /// Names must exist in the fork Registry table and must not shadow a producer
    /// switch.  See [`emit::config_attributes`].
    pub attr_overrides: std::collections::BTreeMap<String, serde_json::Value>,
    pub provenance: String,
    /// Rain, cloud and ice number where the first guess carries the mass but
    /// not the number (GFS and every other one-moment parent). `zero` is the
    /// fork as shipped and is what the native replays pass; `from-mass`, the
    /// default, diagnoses them with the fork's calcnfromq.  With `zero` a
    /// GFS-driven limited-area run carries a 20-30 dBZ echo ring along every
    /// edge (TEMPO turns mass without number into 2.5 mm drops); see
    /// [`scalars::AbsentNumbers`].
    #[serde(default="absent_numbers_default",deserialize_with="absent_numbers")]
    pub absent_numbers: scalars::AbsentNumbers,
}
fn absent_numbers_default()->scalars::AbsentNumbers {scalars::AbsentNumbers::FromMass}
fn absent_numbers<'de,D:serde::Deserializer<'de>>(d:D)->Result<scalars::AbsentNumbers,D::Error> {
    let s:String=serde::Deserialize::deserialize(d)?;
    scalars::AbsentNumbers::parse(&s).map_err(serde::de::Error::custom)
}

/// Receipt for one produced boundary time.
#[derive(Debug, Default, serde::Serialize)]
pub struct IntervalReceipt {
    pub valid_time: String,
    pub met_path: String,
    pub out_path: String,
    pub time_seconds: f64,
    pub met_records_read: usize,
    pub met_records_used: usize,
    pub met_records_ignored: Vec<String>,
    pub first_guess_levels_found: usize,
    /// `isobaric` or `model-level`, the branch the pressure content selected.
    pub pressure_branch: String,
    pub oned_underflow_sites: u64,
    pub hydrostatic_iterations_max: u32,
    pub hydrostatic_cells_hitting_the_cap: usize,
    /// Edges whose second (or first) adjacent cell is the stored-0 garbage
    /// cell; their columns carry the reference's zero contributions.
    pub one_cell_edges: usize,
    pub out_sha256: String,
    /// Wall-clock seconds per stage of this boundary time, in order. The
    /// output digest runs beside the next time (or the run's end) and is
    /// listed as "digest" when it was waited for.
    pub stage_seconds: Vec<(String, f64)>,
    pub seconds: f64,
    /// `zero` or `from-mass`, and the points whose rain, cloud and ice number
    /// were diagnosed from mass at this time (`[nr, nc, ni]`).
    pub absent_numbers: String,
    pub numbers_from_mass: [u64; 3],
}

/// The whole run's receipt.
#[derive(Debug, Default, serde::Serialize)]
pub struct LbcReceipt {
    pub grid_path: String,
    pub start_time: String,
    pub n_cells: usize,
    pub n_edges: usize,
    pub n_vert_levels: usize,
    pub extrap_airtemp: String,
    pub use_spechumd: bool,
    pub oned_underflow: String,
    pub intervals: Vec<IntervalReceipt>,
    /// Seconds to read the grid file's mesh and vertical metrics.
    pub setup_seconds: f64,
    /// Seconds spent waiting for output digests after the last boundary time.
    pub digest_wait_seconds: f64,
    pub seconds: f64,
}

/// First-guess arrays for one boundary time, `[point][level]`.
struct FgLbc {
    scalars: Vec<Vec<Vec<f32>>>,
    vert_level: Vec<f32>,
    t: Vec<Vec<f32>>,
    rh: Vec<Vec<f32>>,
    sh: Vec<Vec<f32>>,
    z: Vec<Vec<f32>>,
    p: Vec<Vec<f32>>,
    u: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    soilz: Vec<f32>,
}

/// Seconds between two `YYYY-MM-DD_HH:MM:SS` stamps on the proleptic
/// Gregorian calendar (`config_calendar_type = 'gregorian'`).
/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_cases.F:7759-8986 at 1d6e370e.
pub fn seconds_between(start: &str, end: &str) -> MpasResult<i64> {
    Ok(epoch_seconds(end)? - epoch_seconds(start)?)
}

/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_cases.F:7759-8986 at 1d6e370e.
fn epoch_seconds(stamp: &str) -> MpasResult<i64> {
    let bad = || MpasError::Refusal(format!("time stamp '{stamp}' is not YYYY-MM-DD_HH:MM:SS"));
    let b: Vec<char> = stamp.chars().collect();
    if b.len() < 19 {
        return Err(bad());
    }
    let num = |from: usize, n: usize| -> MpasResult<i64> {
        b[from..from + n]
            .iter()
            .collect::<String>()
            .parse::<i64>()
            .map_err(|_| bad())
    };
    let (y, m, d) = (num(0, 4)?, num(5, 2)?, num(8, 2)?);
    let (hh, mm, ss) = (num(11, 2)?, num(14, 2)?, num(17, 2)?);
    if !(1..=12).contains(&m) || !(1..=31).contains(&d) {
        return Err(bad());
    }
    // Days-from-civil (Howard Hinnant's algorithm), proleptic Gregorian.
    let y_adj = if m <= 2 { y - 1 } else { y };
    let era = if y_adj >= 0 { y_adj } else { y_adj - 399 } / 400;
    let yoe = y_adj - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146097 + doe - 719468;
    Ok(days * 86400 + hh * 3600 + mm * 60 + ss)
}

/// Build the whole boundary series.
/// MPAS v8.4.2 mpas_init_atm_cases.F:320-369 (clock loop) and
/// :5637-5660 (vertical grid validation).
/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_cases.F:7759-8986 at 1d6e370e.
pub fn build_lbc(cfg: &LbcConfig) -> MpasResult<LbcReceipt> {
    let started = std::time::Instant::now();

    if cfg.intervals.is_empty() {
        return Err(MpasError::Refusal(
            "no boundary intervals were given.  Each --interval names one valid time and the \
             intermediate file carrying it; without at least one there is nothing to produce"
                .to_string(),
        ));
    }

    let mesh = capsule::read_mesh_geometry(&cfg.grid_path)?;
    let metrics = capsule::read_vertical_metrics(&cfg.grid_path, mesh.n_cells, mesh.n_edges)?;
    let max_zgrid = metrics
        .zgrid
        .iter()
        .flat_map(|col| col.iter())
        .fold(f32::MIN, |a, &b| a.max(b));
    if max_zgrid == 0.0 {
        // The native check, kept verbatim in spirit: an input stream without
        // vertical grid information interpolates everything to height zero.
        return Err(MpasError::Refusal(format!(
            "the maximum value of zgrid in {} is 0; the grid file carries no vertical grid \
             information, and every column would be interpolated to height zero",
            cfg.grid_path.display()
        )));
    }

    let header = emit::HeaderSource::from_grid(&cfg.grid_path)?;

    let mut receipt = LbcReceipt {
        grid_path: cfg.grid_path.display().to_string(),
        start_time: cfg.start_time.clone(),
        n_cells: mesh.n_cells,
        n_edges: mesh.n_edges,
        n_vert_levels: metrics.n_vert_levels,
        extrap_airtemp: cfg.extrap_airtemp.label().to_string(),
        use_spechumd: cfg.use_spechumd,
        oned_underflow: cfg.oned_underflow.label().to_string(),
        ..Default::default()
    };

    std::fs::create_dir_all(&cfg.out_dir)?;
    receipt.setup_seconds = started.elapsed().as_secs_f64();

    let mut aerosol_cache=None;
    // RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_cases.F:382-421,
    // 8038-8040,8421-8441,8717-8727 at 1d6e370e. The clock loop reuses
    // its first-guess pool; only the level lookup is allocated afresh. In
    // particular, isobaric pressure filled at the first time remains in
    // that pool and selects the model-level branch at subsequent times.
    let mut first_guess_cache=None;
    // The SHA-256 of each written file (one core, about 13 GB an hour on
    // conus3km) runs on its own thread beside the next boundary time instead
    // of on the main thread between them. Same bytes, same digest.
    // The file's fsync rides the same thread, before its digest, so the
    // main thread never waits on the disk between hours; every file is
    // synced before the run reports success.
    let mut digests: Vec<std::thread::JoinHandle<MpasResult<String>>> = Vec::new();
    for interval in &cfg.intervals {
        let (one, file) = build_one_time(cfg, &mesh, &metrics, &header, interval,&mut aerosol_cache,&mut first_guess_cache)?;
        let path = PathBuf::from(&one.out_path);
        digests.push(std::thread::spawn(move || {
            file.sync_all()?;
            crate::sha256_file(&path)
        }));
        receipt.intervals.push(one);
    }
    let waited = std::time::Instant::now();
    for (one, digest) in receipt.intervals.iter_mut().zip(digests) {
        one.out_sha256 = digest.join().map_err(|_| MpasError::Refusal(format!(
            "the digest thread for {} panicked", one.out_path)))??;
    }
    receipt.digest_wait_seconds = waited.elapsed().as_secs_f64();
    // The first-guess pool and the grid arrays are freed off the main thread;
    // the process ends right after the receipt is written.
    let spent = (first_guess_cache, mesh, metrics, aerosol_cache);
    std::thread::spawn(move || drop(spent));

    receipt.seconds = started.elapsed().as_secs_f64();
    Ok(receipt)
}

/// MPAS v8.4.2 mpas_init_atm_cases.F:5869-6218: pressure dispatch,
/// vertical interpolation, humidity, hydrostatic balance and all LBC fields.
/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_cases.F:7759-8986 at 1d6e370e.
fn build_one_time(
    cfg: &LbcConfig,
    mesh: &MeshGeometry,
    metrics: &VerticalMetrics,
    header: &emit::HeaderSource,
    interval: &BoundaryInterval,
    aerosol_cache: &mut Option<crate::rrfs_init::aerosols::Climatology>,
    first_guess_cache: &mut Option<FgLbc>,
) -> MpasResult<(IntervalReceipt, std::fs::File)> {
    let started = std::time::Instant::now();
    let n_cells = mesh.n_cells;
    let n_edges = mesh.n_edges;
    let nz = metrics.n_vert_levels;

    let mut receipt = IntervalReceipt {
        valid_time: interval.valid_time.clone(),
        met_path: interval.met_path.display().to_string(),
        ..Default::default()
    };

    let mut clock = crate::rrfs_init::StageClock::new("case9");
    let slabs = read_lbc_met_file(&interval.met_path)?;
    clock.mark("read first-guess file");
    let mut fg = horizontal_pass(&slabs, mesh, cfg, &mut receipt,first_guess_cache.take())?;
    clock.mark("horizontal interpolation");
    let nfg = fg.vert_level.len();
    if nfg < 2 {
        return Err(MpasError::Refusal(format!(
            "{} delivered {nfg} first-guess level(s) among the fields the lbc case consumes; \
             a column cannot be interpolated from fewer than two",
            interval.met_path.display()
        )));
    }

    // Isobaric vs model-level, decided the way the Fortran decides it: on
    // whether ANY pressure content arrived.
    let isobaric = fg.p.iter().all(|col| col[..nfg].iter().all(|&v| v == 0.0));
    if isobaric {
        receipt.pressure_branch = "isobaric".to_string();
        let levels = &fg.vert_level;
        fg.p.par_iter_mut().for_each(|column| {
            for (k, &lvl) in levels.iter().enumerate() {
                if lvl != SURFACE_LEVEL_TAG {
                    column[k] = lvl;
                }
            }
        });
    } else {
        receipt.pressure_branch = "model-level".to_string();
        for k in 0..nfg {
            if fg.vert_level[k] == SURFACE_LEVEL_TAG {
                crate::rrfs_init::scatter_level(&mut fg.z, k, &fg.soilz);
            }
        }
    }

    if let Some((header,_))=slabs.last().filter(|(s,_)|s.wind_grid_rel) {
        projection::rotate_velocities(header,&mesh.lat_edge,&mesh.lon_edge,&mut fg.u,&mut fg.v)?;
    }
    clock.mark("pressure branch, rotation");
    let mut scalar_state=scalars::interpolate(&fg.scalars,&fg.vert_level,&fg.z,&metrics.zgrid,true,cfg.smoke,cfg.dust)?;
    clock.mark("scalar vertical interpolation");
    // RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_cases.F:8717-8727
    // at 1d6e370e. The surface pressure log sentinel mutates the retained
    // first-guess pool after scalar interpolation, not just its sorted copy.
    for (k,&level) in fg.vert_level.iter().enumerate() {
        if level==SURFACE_LEVEL_TAG {fg.p.par_iter_mut().for_each(|column| column[k]=1.0);}
    }

    // Rotate the first-guess winds onto the edge normal, in place, exactly as
    // the Fortran overwrites its own fg % u.
    fg.u.par_iter_mut().zip(fg.v.par_iter()).enumerate().for_each(|(e, (u, v))| {
        let angle = mesh.angle_edge[e];
        let (sin, cos) = (angle.sin(), angle.cos());
        for k in 0..nfg {
            u[k] = cos * u[k] + sin * v[k];
        }
    });

    // Vertical interpolation on cells: t, relhum, spechum, pressure.
    let zgrid = &metrics.zgrid;
    let extrap_t = cfg.extrap_airtemp;
    type CellCol = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>);
    let columns: Vec<MpasResult<CellCol>> = (0..n_cells)
        .into_par_iter()
        .map(|c| {
            let mut coord = vec![0.0f32; nfg];
            let mut value = vec![0.0f32; nfg];
            let prep = |coord: &mut Vec<f32>, value: &mut Vec<f32>, src: &dyn Fn(usize) -> f32| {
                for k in 0..nfg {
                    coord[k] = if fg.vert_level[k] == SURFACE_LEVEL_TAG {
                        SURFACE_SENTINEL
                    } else {
                        fg.z[c][k]
                    };
                    value[k] = src(k);
                }
                sorted_column(coord, value);
            };

            let mut t = vec![0.0f32; nz];
            let mut relhum = vec![0.0f32; nz];
            let mut spechum = vec![0.0f32; nz];
            let mut pressure = vec![0.0f32; nz];

            prep(&mut coord, &mut value, &|k| fg.t[c][k]);
            for k in 0..nz {
                let target = 0.5 * (zgrid[c][k] + zgrid[c][k + 1]);
                t[k] = vertical_interp(target, &coord[..nfg - 1], &value[..nfg - 1], extrap_t)
                    .map_err(|_| {
                        MpasError::Refusal(format!(
                            "temperature extrapolation above the first-guess column top under \
                             the lapse-rate mode at cell {c} level {k}.  This is the Fortran's \
                             own fatal error in init_atm_case_lbc, not a port limitation"
                        ))
                    })?;
            }

            prep(&mut coord, &mut value, &|k| fg.rh[c][k]);
            for k in (0..nz).rev() {
                let target = 0.5 * (zgrid[c][k] + zgrid[c][k + 1]);
                relhum[k] = vertical_interp(
                    target,
                    &coord[..nfg - 1],
                    &value[..nfg - 1],
                    Extrap::Constant,
                )
                .unwrap_or(value[0]);
            }

            prep(&mut coord, &mut value, &|k| fg.sh[c][k].max(0.0));
            for k in (0..nz).rev() {
                let target = 0.5 * (zgrid[c][k] + zgrid[c][k + 1]);
                spechum[k] = vertical_interp(
                    target,
                    &coord[..nfg - 1],
                    &value[..nfg - 1],
                    Extrap::Constant,
                )
                .unwrap_or(value[0]);
            }

            // The surface pressure slot is given `1.0` (a value whose log is
            // valid) before the log; the sentinel then sorts it out of the
            // profile.  There is no PSFC in this case.
            prep(&mut coord, &mut value, &|k| {
                if fg.vert_level[k] == SURFACE_LEVEL_TAG {
                    1.0f32.ln()
                } else {
                    fg.p[c][k].ln()
                }
            });
            for k in 0..nz {
                let target = 0.5 * (zgrid[c][k] + zgrid[c][k + 1]);
                pressure[k] =
                    vertical_interp(target, &coord[..nfg - 1], &value[..nfg - 1], Extrap::Linear)
                        .unwrap_or(value[nfg - 2])
                        .exp();
            }

            Ok((t, relhum, spechum, pressure))
        })
        .collect();

    let mut t_col = Vec::with_capacity(n_cells);
    let mut rh_col = Vec::with_capacity(n_cells);
    let mut sh_col = Vec::with_capacity(n_cells);
    let mut p_col = Vec::with_capacity(n_cells);
    for c in columns {
        let (t, rh, sh, p) = c?;
        t_col.push(t);
        rh_col.push(rh);
        sh_col.push(sh);
        p_col.push(p);
    }
    clock.mark("cell column interpolation");

    // RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_cases.F:8720-8756
    // at 1d6e370e. Pressure extrapolates linearly; edge wind holds the endpoint.
    // The stored-0 garbage cell contributes zeros to the coordinate means.
    let garbage_z = vec![0.0f32; nfg];
    let garbage_zgrid = vec![0.0f32; nz + 1];
    let zfg_of = |c: usize| -> &[f32] {
        if c >= 1 && c <= n_cells {
            &fg.z[c - 1]
        } else {
            &garbage_z
        }
    };
    let zgrid_of = |c: usize| -> &[f32] {
        if c >= 1 && c <= n_cells {
            &zgrid[c - 1]
        } else {
            &garbage_zgrid
        }
    };
    let one_cell_edges = mesh
        .cells_on_edge
        .iter()
        .filter(|&&[c1, c2]| c1 == 0 || c1 > n_cells || c2 == 0 || c2 > n_cells)
        .count();
    receipt.one_cell_edges = one_cell_edges;

    let u_edge: Vec<Vec<f32>> = (0..n_edges)
        .into_par_iter()
        .map(|e| {
            let [c1, c2] = mesh.cells_on_edge[e];
            let z1 = zfg_of(c1);
            let z2 = zfg_of(c2);
            let g1 = zgrid_of(c1);
            let g2 = zgrid_of(c2);
            let mut coord = vec![0.0f32; nfg];
            let mut value = vec![0.0f32; nfg];
            for k in 0..nfg {
                coord[k] = if fg.vert_level[k] == SURFACE_LEVEL_TAG {
                    SURFACE_SENTINEL
                } else {
                    0.5 * (z1[k] + z2[k])
                };
                value[k] = fg.u[e][k];
            }
            sorted_column(&mut coord, &mut value);
            (0..nz)
                .map(|k| {
                    let target = 0.25 * (g1[k] + g1[k + 1] + g2[k] + g2[k + 1]);
                    vertical_interp(target, &coord[..nfg - 1], &value[..nfg - 1], Extrap::Constant)
                        .unwrap_or(value[nfg - 2])
                })
                .collect()
        })
        .collect();

    // Water vapour: the spechum path only when it was asked for AND delivered.
    let sh_all_zero = sh_col.par_iter().all(|col| col.iter().all(|&v| v == 0.0));
    let use_sh = cfg.use_spechumd && !sh_all_zero;
    let qv: Vec<Vec<f32>> = (0..n_cells)
        .into_par_iter()
        .map(|c| {
            (0..nz)
                .map(|k| {
                    if use_sh {
                        sh_col[c][k] / (1.0 - sh_col[c][k])
                    } else {
                        let rs = rslf(p_col[c][k], t_col[c][k]);
                        0.01 * rs * rh_col[c][k]
                    }
                })
                .collect()
        })
        .collect();

    // Fork cases.F:8820-8850 diagnoses number from the pre-hydrostatic density.
    let mut initial_density=vec![0.0f32;n_cells*nz];
    initial_density.par_chunks_mut(nz.max(1)).enumerate().for_each(|(c,column)| {for k in 0..nz {
        let p=p_col[c][k];let exner=(p/P0).powf(RGAS/CP);
        let theta=t_col[c][k]*(P0/p).powf(RGAS/CP);
        column[k]=(p/RGAS/(exner*theta))/(1.+qv[c][k]);
    }});
    receipt.numbers_from_mass=scalars::initialize_lbc_numbers(&mut scalar_state,&initial_density,cfg.absent_numbers);
    receipt.absent_numbers=cfg.absent_numbers.label().to_string();
    clock.mark("edge winds, moisture, numbers");

    // The dynamics chain, per cell.  `theta` is the reused `t` array of the
    // Fortran: temperature, then potential temperature, then theta_m.
    struct CellState {
        theta_m: Vec<f32>,
        rho_zz: Vec<f32>,
        iterations: Vec<u32>,
    }
    let states: Vec<CellState> = (0..n_cells)
        .into_par_iter()
        .map(|c| {
            let zz = &metrics.zz[c];
            let mut theta = vec![0.0f32; nz];
            let mut exner = vec![0.0f32; nz];
            let mut rho_zz = vec![0.0f32; nz];
            let mut pressure = p_col[c].clone();

            // Exner, theta, and the first rho_zz, with NO virtual factor,
            // which is where this case departs from case 7.
            for k in 0..nz {
                exner[k] = (pressure[k] / P0).powf(RGAS / CP);
                theta[k] = t_col[c][k] * (P0 / pressure[k]).powf(RGAS / CP);
                rho_zz[k] = pressure[k] / RGAS / (exner[k] * theta[k]);
                rho_zz[k] /= 1.0 + qv[c][k];
            }

            // Base state: dry isothermal at t0b.
            let mut ppb = vec![0.0f32; nz];
            let mut rb = vec![0.0f32; nz];
            let mut pp = vec![0.0f32; nz];
            let mut rr = vec![0.0f32; nz];
            for k in 0..nz {
                let ztemp = 0.5 * (zgrid[c][k + 1] + zgrid[c][k]);
                ppb[k] = P0 * (-GRAVITY * ztemp / (RGAS * T0B)).exp();
                let pb = (ppb[k] / P0).powf(RGAS / CP);
                rb[k] = ppb[k] / (RGAS * T0B);
                exner[k] = pb;
                pp[k] = 0.0;
                rr[k] = 0.0;
            }

            // Couple to the vertical metric.
            for k in 0..nz {
                rb[k] /= zz[k];
                rho_zz[k] /= zz[k];
                pp[k] = pressure[k] - ppb[k];
                rr[k] = rho_zz[k] - rb[k];
            }

            // Hydrostatic rebalance: level 1 seeded, then the 30-pass 1e-4 Pa
            // fixed point, all with the literal 1.61.
            let mut iterations = vec![0u32; nz];
            rho_zz[0] = ((pressure[0] / P0).powf(CV / CP)) * (P0 / RGAS)
                / (theta[0] * (1.0 + VIRTUAL_161 * qv[c][0]))
                / zz[0];
            rr[0] = rho_zz[0] - rb[0];
            for k in 1..nz {
                let mut it = 0u32;
                let mut p_check = 2.0 * 0.0001f32;
                while it < 30 && p_check > 0.0001 {
                    p_check = pp[k];
                    pp[k] = pp[k - 1]
                        - (metrics.fzm[k] * rr[k] + metrics.fzp[k] * rr[k - 1])
                            * GRAVITY
                            * metrics.dzu[k]
                        - (metrics.fzm[k] * rho_zz[k] * qv[c][k]
                            + metrics.fzp[k] * rho_zz[k - 1] * qv[c][k - 1])
                            * GRAVITY
                            * metrics.dzu[k];
                    pressure[k] = pp[k] + ppb[k];
                    exner[k] = (pressure[k] / P0).powf(RGAS / CP);
                    rho_zz[k] = pressure[k]
                        / RGAS
                        / (exner[k] * theta[k] * (1.0 + VIRTUAL_161 * qv[c][k]))
                        / zz[k];
                    rr[k] = rho_zz[k] - rb[k];
                    p_check = (p_check - pp[k]).abs();
                    it += 1;
                }
                iterations[k] = it;
            }

            // theta_m.  (The Fortran also decouples rho_p here; nothing this
            // case writes reads it, so it is not carried.)
            let mut theta_m = theta;
            for k in 0..nz {
                theta_m[k] *= 1.0 + VIRTUAL_161 * qv[c][k];
            }

            CellState {
                theta_m,
                rho_zz,
                iterations,
            }
        })
        .collect();

    let mut it_max = 0u32;
    let mut capped = 0usize;
    for s in &states {
        let mut hit = false;
        for &i in &s.iterations[1..] {
            it_max = it_max.max(i);
            if i >= 30 {
                hit = true;
            }
        }
        if hit {
            capped += 1;
        }
    }
    receipt.hydrostatic_iterations_max = it_max;
    receipt.hydrostatic_cells_hitting_the_cap = capped;
    clock.mark("column dynamics");

    // ru on every edge, garbage cells contributing zero density.
    let rz_of = |c: usize, k: usize| -> f32 {
        if c >= 1 && c <= n_cells {
            states[c - 1].rho_zz[k]
        } else {
            0.0
        }
    };
    let mut ru = zero_columns(n_edges, nz);
    ru.par_iter_mut().enumerate().for_each(|(e, ru_e)| {
        let [c1, c2] = mesh.cells_on_edge[e];
        for k in 0..nz {
            ru_e[k] = u_edge[e][k] * 0.5 * (rz_of(c1, k) + rz_of(c2, k));
        }
    });

    // rw, accumulated per cell over its edges in stored order, then w. The
    // edge check runs first, in cell order, so the refusal is the one the
    // serial loop gave; then every cell sums its own edges on its own core.
    let nzp1 = nz + 1;
    for c in 0..n_cells {
        for i in 0..mesh.n_edges_on_cell[c] {
            let e = mesh.edges_on_cell[c][i];
            if e == 0 || e > n_edges {
                return Err(MpasError::Refusal(format!(
                    "edgesOnCell names edge {e} at cell {} slot {i}, outside 1..{n_edges}; the \
                     reference cull stores no zeros in valid edgesOnCell slots, so this grid \
                     file is not one this producer understands",
                    c + 1
                )));
            }
        }
    }
    let mut rw = zero_columns(n_cells, nzp1);
    rw.par_iter_mut().enumerate().for_each(|(c, rw_c)| {
        for i in 0..mesh.n_edges_on_cell[c] {
            let e = mesh.edges_on_cell[c][i] - 1;
            let side_one = mesh.cells_on_edge[e][0] == c + 1;
            for k in 1..nz {
                let flux = metrics.fzm[k] * ru[e][k] + metrics.fzp[k] * ru[e][k - 1];
                let metric =
                    metrics.fzm[k] * metrics.zz[c][k] + metrics.fzp[k] * metrics.zz[c][k - 1];
                if side_one {
                    rw_c[k] -= metric * metrics.zb[e][0][k] * flux;
                } else {
                    rw_c[k] += metric * metrics.zb[e][1][k] * flux;
                }
                if cfg.theta_adv_order == 3 {
                    // MPAS v8.4.2 mpas_init_atm_cases.F:6181-6188:
                    // SIGN copies the sign bit, including that of -0.0.
                    let sign = 1.0f32.copysign(ru[e][k]);
                    if side_one {
                        rw_c[k] +=
                            sign * cfg.coef_3rd_order * metric * metrics.zb3[e][0][k] * flux;
                    } else {
                        rw_c[k] -=
                            sign * cfg.coef_3rd_order * metric * metrics.zb3[e][1][k] * flux;
                    }
                }
            }
        }
    });
    let mut w = zero_columns(n_cells, nzp1);
    w.par_iter_mut().enumerate().for_each(|(c, w_c)| {
        for k in 1..nz {
            w_c[k] = rw[c][k]
                / (metrics.fzp[k] * states[c].rho_zz[k - 1] + metrics.fzm[k] * states[c].rho_zz[k]);
        }
    });
    clock.mark("ru, rw, w");

    // The lbc fields.
    // Cell-major, level fastest, each cell's run filled on its own core.
    let nzc = nz.max(1);
    let mut lbc_qv = vec![0.0f32; n_cells * nz];
    let mut lbc_rho = vec![0.0f32; n_cells * nz];
    let mut lbc_theta = vec![0.0f32; n_cells * nz];
    lbc_qv.par_chunks_mut(nzc).zip(lbc_rho.par_chunks_mut(nzc)).zip(lbc_theta.par_chunks_mut(nzc)).enumerate()
        .for_each(|(c, ((out_qv, out_rho), out_theta))| {
            for k in 0..nz {
                out_qv[k] = qv[c][k];
                out_rho[k] = states[c].rho_zz[k] * metrics.zz[c][k];
                out_theta[k] = states[c].theta_m[k] / (1.0 + VIRTUAL_161 * qv[c][k]);
            }
        });
    let mut lbc_u = vec![0.0f32; n_edges * nz];
    lbc_u.par_chunks_mut(nzc).zip(u_edge[..n_edges].par_iter())
        .for_each(|(out, column)| out.copy_from_slice(&column[..nz]));
    let mut lbc_w = vec![0.0f32; n_cells * nzp1];
    lbc_w.par_chunks_mut(nzp1).zip(w[..n_cells].par_iter())
        .for_each(|(out, column)| out.copy_from_slice(&column[..nzp1]));

    let time_seconds = seconds_between(&cfg.start_time, &interval.valid_time)?;
    receipt.time_seconds = time_seconds as f64;

    let out_path = cfg.out_dir.join(emit::lbc_file_name(&interval.valid_time));
    // RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_cases.F:8553-8721
    // at 1d6e370e. The resolved stream packages select the interpolated scalars.
    let zeros = vec![0.0f32; n_cells * nz];
    if cfg.aerosolaware && !scalars::use_rap_aerosols(cfg.tempo_rap,&scalar_state) {
        let landmask=netcrust::File::open(&cfg.grid_path)?.read_array_f64("landmask")?.into_values().into_iter().map(|v|v as i32).collect::<Vec<_>>();
        // The fallback interpolates against pressure_base, not the adjusted pressure.
        let mut ppb=vec![0.0f32;n_cells*nz];
        ppb.par_chunks_mut(nzc).enumerate().for_each(|(c,out)| {for k in 0..nz {let z=0.5*(zgrid[c][k+1]+zgrid[c][k]);out[k]=P0*(-GRAVITY*z/(RGAS*T0B)).exp();}});
        let default=PathBuf::from(crate::rrfs_init::aerosols::NATIVE_FILENAME);
        if let Some(a)=crate::rrfs_init::aerosols::initialize_lbc(cfg.gocart_path.as_deref().unwrap_or(&default),
            &interval.valid_time,&cfg.start_time,&mesh.lat_cell,&mesh.lon_cell,&landmask,cfg.gocart_levels,&ppb,nz,aerosol_cache)? {
            scalar_state[10]=a.nwfa;scalar_state[11]=a.nifa;
        }
    }
    let mut extra=std::collections::BTreeMap::new();
    if cfg.hydrometeors_rrfs || cfg.hydrometeors_gfs {
        for i in 0..5 {extra.insert(format!("lbc_{}",scalars::OUTPUT[i]),scalar_state[i].as_slice());}
    }
    if cfg.hydrometeors_rrfs {
        for i in 5..10 {extra.insert(format!("lbc_{}",scalars::OUTPUT[i]),scalar_state[i].as_slice());}
        for name in ["qh","nh","nccn","volg","volh"] {extra.insert(format!("lbc_{name}"),zeros.as_slice());}
    }
    if cfg.hydrometeors_rrfs || cfg.aerosolaware {
        extra.insert("lbc_nwfa".into(),scalar_state[10].as_slice());extra.insert("lbc_nifa".into(),scalar_state[11].as_slice());
    }
    if cfg.smoke {extra.insert("lbc_smoke_fine".into(),scalar_state[12].as_slice());}
    if cfg.dust {extra.insert("lbc_dust_fine".into(),scalar_state[13].as_slice());extra.insert("lbc_dust_coarse".into(),scalar_state[14].as_slice());}
    let fields = emit::LbcFields {
        qv: &lbc_qv,
        extra,
        u: &lbc_u,
        w: &lbc_w,
        rho: &lbc_rho,
        theta: &lbc_theta,
    };
    clock.mark("output fields, GOCART aerosols");
    let written = emit::write_lbc_file(
        &out_path,
        cfg,
        header,
        n_cells,
        n_edges,
        nz,
        &interval.valid_time,
        time_seconds as f32,
        &fields,
    )?;
    clock.mark("write boundary file");
    // This hour's column arrays are tens of millions of small heap blocks on
    // conus3km; one thread frees them beside the next hour instead of the
    // main thread freeing them between hours.
    let spent = (t_col, rh_col, sh_col, p_col, u_edge, qv, states, ru, rw, w, scalar_state, slabs, initial_density);
    std::thread::spawn(move || drop(spent));
    receipt.out_path = out_path.display().to_string();
    receipt.stage_seconds = clock.marks;
    receipt.seconds = started.elapsed().as_secs_f64();
    *first_guess_cache=Some(fg);
    Ok((receipt, written))
}

/// MPAS v8.4.2 mpas_init_atm_cases.F:5666-5898: complete case-9
/// first-guess field dispatch, level accounting and horizontal interpolation.
/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_cases.F:7759-8986 at 1d6e370e.
fn horizontal_pass(
    slabs: &[(MetSlab, i32)],
    mesh: &MeshGeometry,
    cfg: &LbcConfig,
    receipt: &mut IntervalReceipt,
    previous: Option<FgLbc>,
) -> MpasResult<FgLbc> {
    let n_cells = mesh.n_cells;
    let n_edges = mesh.n_edges;
    let nfg_cap = cfg.n_fg_levels;

    let mut fg = previous.unwrap_or_else(|| FgLbc {
        scalars:(0..scalars::COUNT).map(|_| zero_columns(n_cells, nfg_cap)).collect(),
        vert_level: Vec::new(),
        t: zero_columns(n_cells, nfg_cap),
        rh: zero_columns(n_cells, nfg_cap),
        sh: zero_columns(n_cells, nfg_cap),
        z: zero_columns(n_cells, nfg_cap),
        p: zero_columns(n_cells, nfg_cap),
        u: zero_columns(n_edges, nfg_cap),
        v: zero_columns(n_edges, nfg_cap),
        soilz: vec![0.0; n_cells],
    });
    fg.vert_level.clear();

    let cell_lat: Vec<f32> = mesh.lat_cell.iter().map(|v| v * DEG_PER_RAD).collect();
    let cell_lon: Vec<f32> = mesh.lon_cell.iter().map(|v| v * DEG_PER_RAD).collect();
    let edge_lat: Vec<f32> = mesh.lat_edge.iter().map(|v| v * DEG_PER_RAD).collect();
    let edge_lon: Vec<f32> = mesh.lon_edge.iter().map(|v| v * DEG_PER_RAD).collect();

    // Distinct levels among the consumed fields, counted past the cap the way
    // the Fortran keeps counting through its hash table, so the refusal can
    // name the config_nfglevels actually needed.
    let mut distinct_levels: std::collections::BTreeSet<u32> = std::collections::BTreeSet::new();
    let mut too_many = false;

    let underflow_hits = std::sync::atomic::AtomicU64::new(0);
    let methods: [Method; 2] = [Method::SixteenPoint, Method::Search];

    for (s, _iproj) in slabs {
        receipt.met_records_read += 1;
        if !LBC_MET_FIELDS.contains(&s.field.as_str()) && scalars::input_index(&s.field).is_none() {
            if !receipt.met_records_ignored.contains(&s.field) {
                receipt.met_records_ignored.push(s.field.clone());
            }
            continue;
        }
        receipt.met_records_used += 1;

        let level_index = if s.field != "SOILHGT" {
            distinct_levels.insert(s.xlvl.to_bits());
            if distinct_levels.len() > nfg_cap {
                too_many = true;
            }
            if too_many {
                continue;
            }
            match fg.vert_level.iter().position(|&v| v == s.xlvl) {
                Some(i) => i,
                None => {
                    fg.vert_level.push(s.xlvl);
                    fg.vert_level.len() - 1
                }
            }
        } else {
            0
        };

        let proj=projection::Projection::from_met(s)?;
        let array = Slab::from_met(&s.values, s.nx, s.ny);
        let ctx = InterpContext {
            array: &array,
            msgval: MSGVAL,
            maskval: None,
            mask: None,
            underflow: cfg.oned_underflow,
            underflow_hits: Some(&underflow_hits),
        };
        let sample = |i: usize, lat: &[f32], lon: &[f32], ctx: &InterpContext| -> f32 {
            let (x,y)=proj.locate(lat[i],lon[i]);
            interp_sequence(x, y, ctx, &methods, 0)
        };

        let on_edges = s.field == "UU" || s.field == "VV";
        if on_edges {
            let values = sample_points(n_edges, &ctx, |e, ctx| sample(e, &edge_lat, &edge_lon, ctx));
            let dest = if s.field == "UU" {
                &mut fg.u
            } else {
                &mut fg.v
            };
            scatter_level(dest, level_index, &values);
            continue;
        }

        let values = sample_points(n_cells, &ctx, |c, ctx| sample(c, &cell_lat, &cell_lon, ctx));
        match s.field.as_str() {
            "TT" => {
                scatter_level(&mut fg.t, level_index, &values);
            }
            "RH" => {
                scatter_level(&mut fg.rh, level_index, &values);
            }
            "SPECHUMD" => {
                scatter_level(&mut fg.sh, level_index, &values);
            }
            "GHT" => {
                scatter_level(&mut fg.z, level_index, &values);
            }
            "PRES" | "PRESSURE" => {
                scatter_level(&mut fg.p, level_index, &values);
            }
            "SOILHGT" => fg.soilz.copy_from_slice(&values),
            other => {let i=scalars::input_index(other).expect("scalar passed field filter");scatter_level(&mut fg.scalars[i], level_index, &values);},
        }
    }

    if too_many {
        return Err(MpasError::Refusal(format!(
            "the intermediate file holds {} distinct levels among the fields the lbc case \
             consumes, more than the declared {nfg_cap}; raise --nfglevels to at least {} and \
             re-run",
            distinct_levels.len(),
            distinct_levels.len()
        )));
    }

    receipt.oned_underflow_sites = underflow_hits.load(std::sync::atomic::Ordering::Relaxed);
    receipt.first_guess_levels_found = fg.vert_level.len();
    Ok(fg)
}
