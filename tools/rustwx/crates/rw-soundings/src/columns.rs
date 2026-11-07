//! The any-point column store: every column of a run's grid at one valid time, written as an rw-store version 2
//! column file (`rw_store::columns`, schema `rw-store.columns.v2`), so one sounding anywhere in the domain is one
//! small read.
//!
//! A forecast box writes one file per domain, member and valid time from its WRF-shaped history (ArWen's wrfout,
//! hex's lat-lon export). An always-on service later answers "the sounding at this point and time" by reading the
//! file's header once, then the one tile that holds the point (an HTTP range read), sampling the column bilinearly
//! and handing it to `rw_soundings` as a JSON column frame. Nobody has to pick sounding points before a run.
//!
//! The fields are the store's canonical ones (`rw_store::columns::FIELDS_3D`, `FIELDS_2D`): hPa, m MSL, C, m/s
//! (earth-relative), Pa/s; degrees. By default the 3-D fields are quantized (`QUANTA_3D`: 0.05 hPa, 0.5 m, 0.02 C,
//! 0.05 C, 0.05 m/s, 0.01 Pa/s), about 250 bytes a column; `"exact": true` stores every value's f32 instead.
//!
//! Every file also carries the SURFACE SECTION (rw-store `SurfaceWrite`, ABI `columns-surface-v1`): the planes a
//! point forecast or a meteogram reads (precipitation, sea-level pressure, cloud, gust, CAPE and the rest,
//! [`crate::surface`]), exact f32, computed here from the full-precision frame by the map renderer's own functions.
//! The main tiles are unchanged, so soundings read every file as before. At a grid's finish, `--series` folds its
//! hourly files' surface sections into one time-major file (rw-store `series`), so a whole meteogram is three reads.
use crate::input::{self, RunInfo, Source, WrfGrid};
use crate::surface;
use crate::store::Res;
use rw_store::columns::{self as cs, ColumnsWrite, Field2, Field3, Values};
use serde::Deserialize;
use serde_json::json;
use std::path::{Path, PathBuf};
use wrf_core::WrfFile;

pub use rw_store::columns::{FIELDS_2D, FIELDS_3D, QUANTA_3D};

#[derive(Deserialize)]
pub struct ColumnsRequest {
    pub sources: Vec<Source>,
    pub out: PathBuf,
    #[serde(default = "default_tile")]
    pub tile: usize,
    #[serde(default = "default_threads")]
    pub threads: usize,
    /// Store every 3-D value's f32 exactly instead of quantized (about 3x the bytes).
    #[serde(default)]
    pub exact: bool,
    /// The run's description from the caller (a host knows the shape and the product's name); the history's own
    /// globals fill what it leaves out. Written to the header's `meta.run` for the sounding's label.
    #[serde(default)]
    pub run: RunInfo,
}
fn default_tile() -> usize {
    16
}
fn default_threads() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4)
}

fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}

fn stamp(t: i64) -> String {
    chrono::DateTime::from_timestamp(t, 0).unwrap().format("%Y%m%dT%H%M%SZ").to_string()
}

/// One written file, as the caller's receipt wants it.
pub struct Written {
    pub path: PathBuf,
    pub bytes: u64,
    pub ms: f64,
}

pub fn run(req: ColumnsRequest) -> Res<Vec<Written>> {
    if req.sources.is_empty() {
        return Err("a column store needs at least one history file".into());
    }
    if !(4..=128).contains(&req.tile) {
        return Err(format!("tile must be 4..128 columns, got {}", req.tile));
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(req.threads.max(1))
        .build()
        .map_err(err)?;
    let mut written = vec![];
    // Files one at a time (a 3 km nest's time is ~1 GB of f64 fields, a 1 km zoom globe's ~30 GB); within each,
    // the grid's levels and row bands, then the tiles, on every core of the pool.
    for source in &req.sources {
        let f = WrfFile::open(&source.path).map_err(|e| format!("{}: {e}", source.path.display()))?;
        let run_info = req.run.clone().or(&RunInfo::from_wrf(&f));
        for (ti, time) in f.times().map_err(err)?.iter().enumerate() {
            let started = std::time::Instant::now();
            let grid = pool.install(|| input::wrf_grid(&f, &source.path, ti))?;
            let read_ms = started.elapsed().as_secs_f64() * 1e3;
            let valid = input::parse_time(time)?;
            let dir = if source.member == "det" {
                req.out.join(&source.domain)
            } else {
                req.out.join("members").join(&source.member).join(&source.domain)
            };
            let path = dir.join(format!("{}.wcol", stamp(valid)));
            let w = pool.install(|| {
                write_frame(&path, grid, &f, &source.path, ti, &source.domain, &source.member, valid, req.tile, req.exact, &run_info)
            })?;
            f.clear_cache();
            let bytes = w.bytes;
            let ms = started.elapsed().as_secs_f64() * 1e3;
            // Stage times for the caller's receipt (stderr: stdout's COLUMNS rows are parsed).
            eprintln!(
                "COLUMNS-STAGES {} grid_ms={read_ms:.0} encode_write_ms={:.0} surface_ms={:.0} free_ms={:.0} threads={} \
                 surface_planes={} surface_stages={}",
                path.display(),
                w.encode_ms + w.write_ms,
                w.surface_ms,
                (ms - read_ms - w.encode_ms - w.write_ms - w.surface_ms).max(0.0),
                pool.current_num_threads(),
                w.planes,
                w.stages.iter().map(|(k, v)| format!("{k}={v:.0}")).collect::<Vec<_>>().join(",")
            );
            println!("COLUMNS {} bytes={} ms={:.0}", path.display(), bytes, ms);
            written.push(Written { path, bytes, ms });
        }
    }
    Ok(written)
}

/// One frame's file as written, with its stage times.
pub struct FrameWritten {
    pub bytes: u64,
    pub encode_ms: f64,
    pub surface_ms: f64,
    pub write_ms: f64,
    pub planes: usize,
    pub stages: Vec<(&'static str, f64)>,
}

/// The six sounding surface planes as the main tiles store them: pressure Pa -> hPa, 2 m temperature K -> C by
/// scale and offset, the 2 m dewpoint derived per grid point, the 10 m wind earth-relative.
fn sounding_planes<'a>(ps: &'a [f64], terrain: &'a [f64], t2: &'a [f64], td2: &'a [f64], u10: &'a [f64], v10: &'a [f64]) -> Vec<Field2<'a>> {
    let data2: [(&'a [f64], f64, f64); 6] =
        [(ps, 0.01, 0.0), (terrain, 1.0, 0.0), (t2, 1.0, -273.15), (td2, 1.0, 0.0), (u10, 1.0, 0.0), (v10, 1.0, 0.0)];
    data2
        .iter()
        .enumerate()
        .map(|(i, (a, scale, offset))| Field2 {
            name: cs::FIELDS_2D[i + 2],
            units: cs::UNITS_2D[i + 2],
            data: Values::F64(a),
            scale: *scale,
            offset: *offset,
        })
        .collect()
}

/// Write one frame's file with its surface section: the main tiles from the grid, then (the grid's winds, dewpoint
/// and omega freed, its pressure, temperature and height handed on) the surface planes, then the layout. A surface
/// plane that cannot be computed is left out and named; a surface section that cannot be computed at all leaves the
/// six sounding planes and the reason, and the file is still written.
#[allow(clippy::too_many_arguments)]
pub fn write_frame(
    path: &Path,
    g: WrfGrid,
    f: &WrfFile,
    src: &Path,
    ti: usize,
    domain: &str,
    member: &str,
    valid: i64,
    tile: usize,
    exact: bool,
    run: &RunInfo,
) -> Res<FrameWritten> {
    use rayon::prelude::*;
    let started = std::time::Instant::now();
    let td2: Vec<f64> = (0..g.nx * g.ny).into_par_iter().map(|i| g.td2(i)).collect();
    let encoded = cs::encode_main(&columns_write(&g, &td2, domain, member, valid, tile, exact, run))
        .map_err(|e| format!("{}: {e}", path.display()))?;
    let encode_ms = started.elapsed().as_secs_f64() * 1e3;
    let WrfGrid { nx, ny, nz, lat, lon, p, z, t, td, u, v, omega, ps, terrain, t2, q2, u10, v10 } = g;
    // Freed side by side: a 1 km globe's grid is tens of GB of pages to hand back.
    vec![td, u, v, omega].into_par_iter().for_each(drop);
    let started = std::time::Instant::now();
    let kept = surface::Kept {
        nx, ny, nz, p, t_c: t, z, lat: &lat, lon: &lon, ps: &ps, terrain: &terrain, t2: &t2, q2: &q2, u10: &u10, v10: &v10,
    };
    let (planes, notes, stages) = match surface::compute(f, src, ti, kept) {
        Ok(s) => (s.planes, s.notes, s.stages),
        Err(e) => (vec![], json!({"absent": {"surface": format!("no surface planes: {e}")}}), vec![]),
    };
    f.clear_cache();
    let surface_ms = started.elapsed().as_secs_f64() * 1e3;
    let started = std::time::Instant::now();
    let mut fields = sounding_planes(&ps, &terrain, &t2, &td2, &u10, &v10);
    fields.extend(planes.iter().map(|p| Field2 { name: p.name, units: p.units, data: Values::F32(&p.data), scale: 1.0, offset: 0.0 }));
    let count = fields.len();
    let s = cs::SurfaceWrite { lat: Values::F64(&lat), lon: Values::F64(&lon), fields, notes };
    let out = cs::finish(encoded, Some(&s)).map_err(|e| format!("{}: {e}", path.display()))?;
    let bytes = cs::write_bytes(path, &out).map_err(|e| format!("{}: {e}", path.display()))?;
    let write_ms = started.elapsed().as_secs_f64() * 1e3;
    Ok(FrameWritten { bytes, encode_ms, surface_ms, write_ms, planes: count, stages })
}

/// `rw_soundings --series STORE_DIR [DOMAIN...]`: each fixed grid's time series file from its hourly files'
/// surface sections (every grid of the store when none is named). One `SERIES <path> bytes=.. ms=..` line per file
/// written and one `SERIES-SKIP <domain> <reason>` per grid that gets none (a moving grid, a store written before
/// the surface section).
pub fn series(store: &Path, domains: &[String], threads: usize) -> Res<Vec<rw_store::series::SeriesWritten>> {
    let pool = rayon::ThreadPoolBuilder::new().num_threads(threads.max(1)).build().map_err(err)?;
    let domains: Vec<String> = if domains.is_empty() { rw_store::series::grids(store) } else { domains.to_vec() };
    let mut out = vec![];
    for d in &domains {
        let w = pool.install(|| rw_store::series::write_series(store, d, 4)).map_err(|e| format!("{d}: {e}"))?;
        match (&w.path, &w.skipped) {
            (Some(p), _) => println!(
                "SERIES {} bytes={} ms={:.0} members={} times={} fields={}",
                p.display(),
                w.bytes,
                w.ms,
                w.members,
                w.times,
                w.fields
            ),
            (None, why) => println!("SERIES-SKIP {d} {}", why.as_deref().unwrap_or("nothing to write")),
        }
        out.push(w);
    }
    Ok(out)
}

/// The version 2 description of one WRF-shaped grid: the canonical fields, pressure Pa -> hPa and 2 m
/// temperature K -> C by scale and offset, 2 m dewpoint derived per grid point.
pub fn columns_write<'a>(
    g: &'a WrfGrid,
    td2: &'a [f64],
    domain: &'a str,
    member: &'a str,
    valid: i64,
    tile: usize,
    exact: bool,
    run: &RunInfo,
) -> ColumnsWrite<'a> {
    let data3: [(&'a [f64], f64); 7] =
        [(&g.p, 0.01), (&g.z, 1.0), (&g.t, 1.0), (&g.td, 1.0), (&g.u, 1.0), (&g.v, 1.0), (&g.omega, 1.0)];
    let data2: [(&'a [f64], f64, f64); 6] = [
        (&g.ps, 0.01, 0.0),
        (&g.terrain, 1.0, 0.0),
        (&g.t2, 1.0, -273.15),
        (td2, 1.0, 0.0),
        (&g.u10, 1.0, 0.0),
        (&g.v10, 1.0, 0.0),
    ];
    ColumnsWrite {
        nx: g.nx,
        ny: g.ny,
        nz: g.nz,
        lat: Values::F64(&g.lat),
        lon: Values::F64(&g.lon),
        fields3: data3
            .iter()
            .enumerate()
            .map(|(i, (a, scale))| Field3 {
                name: cs::FIELDS_3D[i],
                units: cs::UNITS_3D[i],
                quantum: if exact { None } else { Some(cs::QUANTA_3D[i]) },
                data: Values::F64(a),
                scale: *scale,
                offset: 0.0,
                affine: None,
            })
            .collect(),
        fields2: data2
            .iter()
            .enumerate()
            .map(|(i, (a, scale, offset))| Field2 {
                name: cs::FIELDS_2D[i + 2],
                units: cs::UNITS_2D[i + 2],
                data: Values::F64(a),
                scale: *scale,
                offset: *offset,
            })
            .collect(),
        domain,
        member,
        valid_unix: valid,
        tile,
        // `run`: what the run says about itself (init, model, shape, source), for the sounding's top-right label.
        meta: json!({"writer": concat!("rw_soundings ", env!("CARGO_PKG_VERSION"), " --columns"), "source": "wrf", "run": run}),
    }
}

#[allow(clippy::too_many_arguments)]
pub fn write_file(path: &Path, g: &WrfGrid, domain: &str, member: &str, valid: i64, tile: usize, exact: bool, run: &RunInfo) -> Res<u64> {
    use rayon::prelude::*;
    let td2: Vec<f64> = (0..g.nx * g.ny).into_par_iter().map(|i| g.td2(i)).collect();
    cs::write(path, &columns_write(g, &td2, domain, member, valid, tile, exact, run)).map_err(|e| format!("{}: {e}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid(nx: usize, ny: usize, nz: usize) -> WrfGrid {
        let n = nx * ny;
        let v3 = |f: &dyn Fn(usize, usize, usize) -> f64| -> Vec<f64> {
            (0..nz * n).map(|i| f(i / n, (i % n) % nx, (i % n) / nx)).collect()
        };
        let v2 = |f: &dyn Fn(usize, usize) -> f64| -> Vec<f64> { (0..n).map(|i| f(i % nx, i / nx)).collect() };
        WrfGrid {
            nx,
            ny,
            nz,
            lat: v2(&|_, y| 35.0 + y as f64 * 0.03),
            lon: v2(&|x, _| -98.0 + x as f64 * 0.03),
            p: v3(&|k, _, _| 100_000.0 - k as f64 * 8_000.0),
            z: v3(&|k, _, _| 300.0 + k as f64 * 800.0),
            t: v3(&|k, x, _| 25.0 - k as f64 * 6.5 + x as f64 * 0.01),
            td: v3(&|k, _, y| 18.0 - k as f64 * 8.0 + y as f64 * 0.01),
            u: v3(&|k, _, _| 2.0 + k as f64 * 3.0),
            v: v3(&|k, _, _| 5.0 - k as f64),
            omega: v3(&|_, _, _| f64::NAN),
            ps: v2(&|_, _| 98_000.0),
            terrain: v2(&|_, _| 250.0),
            t2: v2(&|_, _| 300.0),
            q2: v2(&|_, _| 0.012),
            u10: v2(&|_, _| 1.0),
            v10: v2(&|_, _| 2.0),
        }
    }

    /// The written file is an rw-store v2 column file the store's own reader opens: header, tiles, a missing omega
    /// counted as filled, and the stored values the canonical units.
    #[test]
    fn store_layout_is_what_readers_read() {
        let dir = std::env::temp_dir().join(format!("rw-columns-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.wcol");
        let g = grid(37, 20, 6);
        let run = RunInfo { init: Some("2024-07-14T12:00:00Z".into()), model: Some("Regional".into()), source: Some("HRRR".into()), shape: None };
        let bytes = write_file(&path, &g, "d01", "det", 1_720_972_800, 16, false, &run).unwrap();
        assert_eq!(std::fs::metadata(&path).unwrap().len(), bytes);
        let f = rw_store::columns::ColumnFile::open(&path).unwrap();
        let h = &f.header;
        assert_eq!((h.tiles_x, h.tiles_y), (3, 2));
        // Counted per stored value, and tiles overlap by one column: (17 + 17 + 5) x (17 + 4) columns x 6 levels.
        assert_eq!(h.filled, 39 * 21 * 6);
        assert_eq!(h.valid, "2024-07-14T16:00:00Z");
        assert_eq!(h.fields2d, FIELDS_2D);
        let t = f.tile_at(5.0, 5.0).unwrap();
        assert_eq!(t.column3(0, 5, 5)[1], 920.0);
        assert_eq!(t.point2(4, 5, 5), (300.0f64 - 273.15) as f32);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// On a real history file (RW_COLUMNS_WRFOUT=path, `cargo test -- --ignored`): the exact store holds every
    /// value of the grid bit for bit, and the quantized store every value as round(x / q) * q, at every point.
    #[test]
    #[ignore]
    fn real_history_round_trips() {
        let Ok(path) = std::env::var("RW_COLUMNS_WRFOUT") else { return };
        let f = WrfFile::open(Path::new(&path)).unwrap();
        let g = input::wrf_grid(&f, Path::new(&path), 0).unwrap();
        let n = g.nx * g.ny;
        let td2: Vec<f64> = (0..n).map(|i| g.td2(i)).collect();
        for exact in [true, false] {
            let w = columns_write(&g, &td2, "d01", "det", 0, 16, exact, &RunInfo::default());
            let raw = cs::encode(&w).unwrap();
            let file = cs::ColumnFile::from_source(raw.clone()).unwrap();
            let (f3, f2) = file.read_full().unwrap();
            let mut differ = 0usize;
            for (i, fld) in w.fields3.iter().enumerate() {
                for j in 0..g.nz * n {
                    let v = match fld.data {
                        Values::F64(a) => a[j] * fld.scale,
                        Values::F32(a) => a[j] as f64 * fld.scale,
                    };
                    let want = if exact { v as f32 } else { ((v / cs::QUANTA_3D[i]).round() * cs::QUANTA_3D[i]) as f32 };
                    // Exact: bit for bit. Quantized: the same number (a small negative rounds to code 0, which
                    // decodes to +0.0 where the formula says -0.0).
                    let same = if exact { f3[i][j].to_bits() == want.to_bits() } else { f3[i][j] == want };
                    if v.is_finite() && !same {
                        differ += 1;
                    }
                }
            }
            for (i, fld) in w.fields2.iter().enumerate() {
                for j in 0..n {
                    let want = match fld.data {
                        Values::F64(a) => (a[j] * fld.scale + fld.offset) as f32,
                        Values::F32(a) => (a[j] as f64 * fld.scale + fld.offset) as f32,
                    };
                    if f2[i + 2][j].to_bits() != want.to_bits() {
                        differ += 1;
                    }
                }
            }
            println!(
                "REAL {} exact={exact} {}x{}x{} bytes={} differing={differ}",
                path,
                g.nx,
                g.ny,
                g.nz,
                raw.len()
            );
            assert_eq!(differ, 0);
        }
    }

    /// On a real history file (RW_COLUMNS_WRFOUT=path, `cargo test -- --ignored`): the parallel grid (levels and
    /// row bands on every core, windows read by the tasks) holds every value of the serial whole-field route
    /// (wrf-core's derived fields), bit for bit, in every field.
    #[test]
    #[ignore]
    fn parallel_grid_is_bitwise_the_serial_one() {
        let Ok(path) = std::env::var("RW_COLUMNS_WRFOUT") else { return };
        let f = WrfFile::open(Path::new(&path)).unwrap();
        let started = std::time::Instant::now();
        let a = input::wrf_grid(&f, Path::new(&path), 0).unwrap();
        let parallel_s = started.elapsed().as_secs_f64();
        let started = std::time::Instant::now();
        let b = input::wrf_grid_serial(&f, 0).unwrap();
        let serial_s = started.elapsed().as_secs_f64();
        assert_eq!((a.nx, a.ny, a.nz), (b.nx, b.ny, b.nz));
        let fields = |g: &WrfGrid| -> Vec<(&'static str, Vec<u64>)> {
            [
                ("lat", &g.lat),
                ("lon", &g.lon),
                ("p", &g.p),
                ("z", &g.z),
                ("t", &g.t),
                ("td", &g.td),
                ("u", &g.u),
                ("v", &g.v),
                ("omega", &g.omega),
                ("ps", &g.ps),
                ("terrain", &g.terrain),
                ("t2", &g.t2),
                ("q2", &g.q2),
                ("u10", &g.u10),
                ("v10", &g.v10),
            ]
            .into_iter()
            .map(|(k, v)| (k, v.iter().map(|x| x.to_bits()).collect()))
            .collect()
        };
        let mut differ = 0usize;
        for ((name, x), (_, y)) in fields(&a).into_iter().zip(fields(&b)) {
            assert_eq!(x.len(), y.len(), "{name}");
            let d = x.iter().zip(&y).filter(|(p, q)| p != q).count();
            println!("FIELD {name} values={} differing={d}", x.len());
            differ += d;
        }
        println!(
            "GRID {path} {}x{}x{} parallel_s={parallel_s:.2} serial_s={serial_s:.2} threads={} differing={differ}",
            a.nx,
            a.ny,
            a.nz,
            rayon::current_num_threads()
        );
        assert_eq!(differ, 0);
    }
}
