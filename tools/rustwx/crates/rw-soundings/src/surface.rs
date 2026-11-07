//! The column file's surface section (rw-store `SurfaceWrite`): every surface plane a point forecast (an
//! Open-Meteo-style hourly series, a meteogram) reads, computed once per frame on the box from the full-precision
//! history and stored as exact f32.
//!
//! Each plane is computed by the function the map renderer (rw-wrfbatch) uses for the same quantity, so a value the
//! point forecast serves is the value the map draws at that point:
//!
//! | plane | function |
//! |---|---|
//! | precip_mm, precip_conv_mm | rustwx-calc `surface::total_precip` (the renderer's `apcp`: RAINC + RAINNC + RAINSH; RAINC + RAINSH) |
//! | mslp_hpa | wrf-core `slp` (WRF frames) or rw-mpas Shuell (frames converted from MPAS, `MPAS_BRIDGE_SCHEMA`), then rustwx-calc `mslp_smoothing::display_mslp`, as the renderer draws it |
//! | cloud_low_pct, cloud_mid_pct, cloud_high_pct | wrf-core `cloudfrac` |
//! | sbcape_jkg, sbcin_jkg, mlcape_jkg, mucape_jkg | wrf-core `sbcape`, `sbcin`, `mlcape`, `mucape` |
//! | freezing_level_agl_m | wrf-core `freezing_level` |
//! | gust_ms | rustwx-calc `surface::gust_10m` on wrf-core `height_agl`, `wspd`, `wspd10` and PBLH |
//! | wspd80_ms, wdir80_deg, wspd100_ms, wdir100_deg | rustwx-calc `hubwind::interpolate` (the hub-height products' interpolation) |
//! | visibility_m | rustwx-calc `visibility::compute_visibility` |
//! | refl_comp_dbz | rustwx-calc `surface::composite_reflectivity_from_dbz` on REFL_10CM, else the model's REFC |
//! | snow_mm, graupel_mm, hail_mm, snow_depth_m, swdown_wm2, pblh_m, wind10_max_ms, gust_max_ms | the history's SNOWNC, GRAUPELNC, HAILNC, SNOWH, SWDOWN, PBLH, WSPD10MAX, GUST_EST_MAX |
//! | uh25_max_m2s2, uh01_max_m2s2, uh02_max_m2s2, uh03_max_m2s2 | the history's UP_HELI_MAX, UP_HELI_MAX01, UP_HELI_MAX02, UP_HELI_MAX03 |
//!
//! A plane whose inputs the frame does not carry is left out and named in the notes' `absent`, never filled.
//!
//! Memory: the caller hands over the grid's pressure, temperature and height (moved, not copied); they seed
//! wrf-core's per-time cache (its own keys for P+PB, T-273.15 and geopotential/g, the values the parallel grid makes
//! bit for bit), so wrf-core's diagnostics read no 3-D field the grid already read. The cache is cleared between
//! stages so at most about nine 3-D fields are held at once (the grid itself is seven).
use crate::input;
use crate::store::Res;
use rayon::prelude::*;
use rustwx_calc::surface as sfc;
use serde_json::{Map, Value, json};
use std::path::Path;
use wrf_core::{ComputeOpts, WrfFile};

/// One stored plane.
pub struct Plane {
    pub name: &'static str,
    pub units: &'static str,
    pub data: Vec<f32>,
}

/// What the grid hands over once its main tiles are encoded: its 3-D pressure (Pa), temperature (C) and height
/// (m MSL), [nz][ny][nx], moved; its 2-D planes, borrowed (the caller stores them too).
pub struct Kept<'a> {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub p: Vec<f64>,
    pub t_c: Vec<f64>,
    pub z: Vec<f64>,
    pub lat: &'a [f64],
    pub lon: &'a [f64],
    pub ps: &'a [f64],
    pub terrain: &'a [f64],
    pub t2: &'a [f64],
    pub q2: &'a [f64],
    /// Earth-relative 10 m wind (m/s).
    pub u10: &'a [f64],
    pub v10: &'a [f64],
}

/// The planes, the notes the header carries (what each plane is; `absent`: what the frame could not give) and the
/// time each stage took (ms), for the receipt.
pub struct Surface {
    pub planes: Vec<Plane>,
    pub notes: Value,
    pub stages: Vec<(&'static str, f64)>,
}

const P0: f64 = 100_000.0;
const KAPPA: f64 = 0.2857142857;

/// Run `f`, a panic inside a diagnostic becoming an error naming it (as the renderer's `isolate_panics`): one plane
/// lost, never the store.
fn isolate<T>(what: &str, f: impl FnOnce() -> Result<T, String>) -> Result<T, String> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).unwrap_or_else(|payload| {
        let message = payload
            .downcast_ref::<&str>()
            .map(|m| (*m).to_string())
            .or_else(|| payload.downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "unknown panic".into());
        Err(format!("panicked computing {what}: {message}"))
    })
}

fn getvar(f: &WrfFile, ti: usize, name: &str, units: Option<&str>, cells: usize, planes: usize) -> Result<Vec<f64>, String> {
    let opts = ComputeOpts { units: units.map(str::to_string), ..ComputeOpts::default() };
    let out = isolate(name, || wrf_core::getvar(f, name, Some(ti), &opts).map_err(|e| e.to_string()))?;
    if out.data.len() != cells * planes {
        return Err(format!("{name}: {} values, expected {}", out.data.len(), cells * planes));
    }
    Ok(out.data)
}

fn notes_table() -> Map<String, Value> {
    let mut m = Map::new();
    for (k, v) in [
        ("precip_mm", "total precipitation since the run's start (RAINC + RAINNC + RAINSH, the map's apcp); an hour's amount is the difference of two times"),
        ("precip_conv_mm", "convective (cumulus) precipitation since the start (RAINC + RAINSH); 0 where the run has no cumulus scheme"),
        ("snow_mm", "snow since the start, liquid water equivalent (SNOWNC)"),
        ("graupel_mm", "graupel since the start, liquid water equivalent (GRAUPELNC)"),
        ("hail_mm", "hail since the start, liquid water equivalent (HAILNC)"),
        ("snow_depth_m", "snow depth on the ground (SNOWH)"),
        ("mslp_hpa", "sea-level pressure as the map draws it: the reduction (wrf-core slp, or NCEP Shuell on frames converted from MPAS), smoothed over ground above 500 m"),
        ("cloud_low_pct", "low cloud (surface to 800 hPa), wrf-core cloudfrac: 100 where any level holds cloud water or ice above 1e-6 kg/kg, else 0"),
        ("cloud_mid_pct", "mid cloud (800 to 450 hPa), wrf-core cloudfrac"),
        ("cloud_high_pct", "high cloud (above 450 hPa), wrf-core cloudfrac"),
        ("sbcape_jkg", "surface-based CAPE (wrf-core sbcape)"),
        ("sbcin_jkg", "surface-based CIN (wrf-core sbcin)"),
        ("mlcape_jkg", "mixed-layer CAPE (wrf-core mlcape)"),
        ("mucape_jkg", "most-unstable CAPE (wrf-core mucape)"),
        ("freezing_level_agl_m", "height above ground of the first 0 C crossing going up (wrf-core freezing_level); 0 where the ground is below freezing"),
        ("gust_ms", "10 m gust at the hour: the wind at the top of the boundary layer mixed down (the map's gust)"),
        ("wspd80_ms", "wind speed 80 m above ground (the hub-height products' interpolation)"),
        ("wdir80_deg", "wind direction 80 m above ground, degrees the wind blows from"),
        ("wspd100_ms", "wind speed 100 m above ground"),
        ("wdir100_deg", "wind direction 100 m above ground, degrees the wind blows from"),
        ("visibility_m", "visibility at the surface (Stoelinga-Warner hydrometeor extinction with the RUC humidity limit, capped at 90 km)"),
        ("refl_comp_dbz", "composite (column maximum) reflectivity"),
        ("swdown_wm2", "downward shortwave radiation at the surface at the hour"),
        ("pblh_m", "boundary-layer height (the PBL scheme's PBLH)"),
        ("wind10_max_ms", "largest 10 m wind since the previous history time (WSPD10MAX)"),
        ("gust_max_ms", "largest estimated gust since the previous history time (GUST_EST_MAX)"),
        ("uh25_max_m2s2", "largest 2-5 km updraft helicity since the previous history time (UP_HELI_MAX)"),
        ("uh01_max_m2s2", "largest 0-1 km updraft helicity since the previous history time (UP_HELI_MAX01)"),
        ("uh02_max_m2s2", "largest 0-2 km updraft helicity since the previous history time (UP_HELI_MAX02)"),
        ("uh03_max_m2s2", "largest 0-3 km updraft helicity since the previous history time (UP_HELI_MAX03)"),
        ("psfc_hpa", "surface pressure"),
        ("orog_m", "model terrain height"),
        ("t2_c", "2 m temperature"),
        ("td2_c", "2 m dew point"),
        ("u10_ms", "10 m wind, eastward (earth-relative)"),
        ("v10_ms", "10 m wind, northward (earth-relative)"),
    ] {
        m.insert(k.into(), json!(v));
    }
    m
}

/// The surface planes of one frame. Runs in the caller's rayon pool.
pub fn compute(f: &WrfFile, path: &Path, ti: usize, g: Kept<'_>) -> Res<Surface> {
    let Kept { nx, ny, nz, lat, lon, p, t_c, z, ps, terrain, t2, q2, u10, v10 } = g;
    let n = nx * ny;
    let mut planes: Vec<Plane> = vec![];
    let mut absent: Map<String, Value> = Map::new();
    let mut stages: Vec<(&'static str, f64)> = vec![];
    let mut clock = std::time::Instant::now();
    let mut lap = |stages: &mut Vec<(&'static str, f64)>, name: &'static str| {
        stages.push((name, clock.elapsed().as_secs_f64() * 1e3));
        clock = std::time::Instant::now();
    };
    let miss = |absent: &mut Map<String, Value>, names: &[&str], why: String| {
        for name in names {
            absent.insert((*name).to_string(), json!(why));
        }
    };
    let plane32 = |v: &[f64]| sfc::clean_values(v);

    // 1. The history's own 2-D planes, read side by side.
    let names2 = [
        "RAINC", "RAINNC", "RAINSH", "SNOWNC", "GRAUPELNC", "HAILNC", "SNOWH", "SWDOWN", "PBLH", "REFC", "WSPD10MAX",
        "GUST_EST_MAX", "COSALPHA", "SINALPHA", "UP_HELI_MAX", "UP_HELI_MAX01", "UP_HELI_MAX02", "UP_HELI_MAX03",
    ];
    let mut two = input::read_whole(f, path, ti, &names2)?;
    let precip_parts: Vec<&[f64]> = ["RAINC", "RAINNC", "RAINSH"].iter().filter_map(|k| two.get(k).map(Vec::as_slice)).collect();
    match sfc::total_precip(&precip_parts, n) {
        Some(v) => planes.push(Plane { name: "precip_mm", units: "mm", data: v }),
        None => miss(&mut absent, &["precip_mm"], "the frame carries no RAINC, RAINNC or RAINSH".into()),
    }
    let conv_parts: Vec<&[f64]> = ["RAINC", "RAINSH"].iter().filter_map(|k| two.get(k).map(Vec::as_slice)).collect();
    match sfc::total_precip(&conv_parts, n) {
        Some(v) => planes.push(Plane { name: "precip_conv_mm", units: "mm", data: v }),
        None => miss(&mut absent, &["precip_conv_mm"], "the frame carries no RAINC or RAINSH".into()),
    }
    for (var, name, units) in [
        ("SNOWNC", "snow_mm", "mm"),
        ("GRAUPELNC", "graupel_mm", "mm"),
        ("HAILNC", "hail_mm", "mm"),
        ("SNOWH", "snow_depth_m", "m"),
        ("SWDOWN", "swdown_wm2", "W/m2"),
        ("PBLH", "pblh_m", "m"),
        ("WSPD10MAX", "wind10_max_ms", "m/s"),
        ("GUST_EST_MAX", "gust_max_ms", "m/s"),
        ("UP_HELI_MAX", "uh25_max_m2s2", "m2/s2"),
        ("UP_HELI_MAX01", "uh01_max_m2s2", "m2/s2"),
        ("UP_HELI_MAX02", "uh02_max_m2s2", "m2/s2"),
        ("UP_HELI_MAX03", "uh03_max_m2s2", "m2/s2"),
    ] {
        match two.get(var) {
            Some(v) => planes.push(Plane { name, units, data: plane32(v) }),
            None => miss(&mut absent, &[name], format!("the frame carries no {var}")),
        }
    }
    // Composite reflectivity: the REFL_10CM volume's column maximum, else the model's own REFC (the renderer's order).
    let mut refl = None;
    if f.has_var("REFL_10CM") {
        let vol = input::read_whole(f, path, ti, &["REFL_10CM"])?;
        if let Some(v) = vol.get("REFL_10CM") {
            refl = sfc::composite_reflectivity_from_dbz(v, v.len() / n, n).ok().map(|c| plane32(&c));
        }
    } else if let Some(v) = two.get("REFC") {
        refl = Some(v.iter().map(|x| if sfc::valid_reflectivity_dbz(*x) { sfc::clean_value(*x) } else { f32::NAN }).collect());
    }
    match refl {
        Some(v) => planes.push(Plane { name: "refl_comp_dbz", units: "dBZ", data: v }),
        None => miss(&mut absent, &["refl_comp_dbz"], "the frame carries no REFL_10CM or REFC".into()),
    }
    lap(&mut stages, "planes2d_ms");

    // 2. Seed wrf-core's cache with the grid's own fields (its keys; the same values bit for bit), then T -> K and
    // QVAPOR, read in parallel.
    f.prepare_cache_for_time(ti);
    let seed = |key: &str, v: Vec<f64>| {
        f.store_cached_field(format!("{key}_{ti}"), v);
    };
    seed("pressure", p);
    seed("temp_c", t_c);
    seed("height_msl", z);
    seed("terrain", terrain.to_vec());
    seed("psfc", ps.to_vec());
    seed("t2", t2.to_vec());
    seed("q2", q2.to_vec());
    let pressure = f.full_pressure(ti).map_err(|e| e.to_string())?;
    let mut raw = input::read_whole(f, path, ti, &["T", "QVAPOR"])?;
    let (Some(theta), Some(qv)) = (raw.remove("T"), raw.remove("QVAPOR")) else {
        return Err("the frame has no T or QVAPOR".into());
    };
    if theta.len() != nz * n || qv.len() != nz * n {
        return Err(format!("T/QVAPOR hold {}/{} values for {nz} levels of {n} columns", theta.len(), qv.len()));
    }
    // wrf-core's temperature, in its order: (T + 300) * (p / P0)^kappa.
    let tk: Vec<f64> = theta.par_iter().zip(pressure.par_iter()).map(|(th, pr)| (th + 300.0) * (pr / P0).powf(KAPPA)).collect();
    drop(theta);
    seed("temp", tk);
    seed("qvapor", qv);
    lap(&mut stages, "seed_ms");

    // 3. Sea-level pressure, the map's: the reduction, then the smoothing over high ground.
    let reduced: Result<Vec<f32>, String> = if f.global_attr_str("MPAS_BRIDGE_SCHEMA").is_ok() {
        (|| {
            let tk = f.temperature(ti).map_err(|e| e.to_string())?;
            let qv = f.qvapor(ti).map_err(|e| e.to_string())?;
            let z = f.height_msl(ti).map_err(|e| e.to_string())?;
            Ok((0..n)
                .into_par_iter()
                .map(|ij| rw_mpas::mslp::shuell_mslp_pa(ps[ij], terrain[ij], tk[ij], qv[ij], z[ij]) as f32)
                .collect())
        })()
    } else {
        getvar(f, ti, "slp", Some("Pa"), n, 1).map(|v| plane32(&v))
    };
    match reduced {
        Ok(mut values) => {
            let terrain32: Vec<f32> = terrain.iter().map(|z| *z as f32).collect();
            let lat32: Vec<f32> = lat.iter().map(|v| *v as f32).collect();
            let lon32: Vec<f32> = lon.iter().map(|v| *v as f32).collect();
            rustwx_calc::mslp_smoothing::display_mslp(&mut values, &terrain32, &lat32, &lon32, nx, ny);
            // Stored in hPa: the map's Pa word divided by 100 (one f32 rounding).
            planes.push(Plane { name: "mslp_hpa", units: "hPa", data: values.iter().map(|v| (*v as f64 / 100.0) as f32).collect() });
        }
        Err(e) => miss(&mut absent, &["mslp_hpa"], e),
    }
    lap(&mut stages, "mslp_ms");

    // 4. Instability and the freezing level (wrf-core, grid-parallel).
    for (var, name) in [("sbcape", "sbcape_jkg"), ("sbcin", "sbcin_jkg"), ("mlcape", "mlcape_jkg"), ("mucape", "mucape_jkg")] {
        match getvar(f, ti, var, None, n, 1) {
            Ok(v) => planes.push(Plane { name, units: "J/kg", data: plane32(&v) }),
            Err(e) => miss(&mut absent, &[name], e),
        }
    }
    lap(&mut stages, "cape_ms");
    match getvar(f, ti, "freezing_level", None, n, 1) {
        Ok(v) => planes.push(Plane { name: "freezing_level_agl_m", units: "m", data: plane32(&v) }),
        Err(e) => miss(&mut absent, &["freezing_level_agl_m"], e),
    }

    // 5. Cloud (wrf-core cloudfrac reads QCLOUD and QICE itself). Without QCLOUD it would read clear sky everywhere,
    // so the plane is left out instead.
    if f.has_var("QCLOUD") {
        match getvar(f, ti, "cloudfrac", None, n, 3) {
            Ok(v) => {
                for (i, name) in ["cloud_low_pct", "cloud_mid_pct", "cloud_high_pct"].into_iter().enumerate() {
                    planes.push(Plane { name, units: "%", data: plane32(&v[i * n..(i + 1) * n]) });
                }
            }
            Err(e) => miss(&mut absent, &["cloud_low_pct", "cloud_mid_pct", "cloud_high_pct"], e),
        }
    } else {
        miss(&mut absent, &["cloud_low_pct", "cloud_mid_pct", "cloud_high_pct"], "the frame carries no QCLOUD".into());
    }
    lap(&mut stages, "cloud_ms");

    // 6. Winds above the ground: the gust and the hub heights. Keep what the later stages need, clear the rest.
    let hagl_vec = f.height_agl(ti).map_err(|e| e.to_string())?.to_vec();
    let (tk_arc, qv_arc) = (f.temperature(ti).map_err(|e| e.to_string())?, f.qvapor(ti).map_err(|e| e.to_string())?);
    f.clear_cache();
    f.prepare_cache_for_time(ti);
    let hagl = f.store_cached_field(format!("height_agl_{ti}"), hagl_vec);
    let mut uv = input::read_whole(f, path, ti, &["U", "V"])?;
    if let (Some(us), Some(vs)) = (uv.remove("U"), uv.remove("V")) {
        if us.len() == nz * ny * (nx + 1) && vs.len() == nz * (ny + 1) * nx {
            // wrf-core's destaggering, in its order: 0.5 * (left + right), 0.5 * (below + above).
            let ud: Vec<f64> = (0..nz * n)
                .into_par_iter()
                .map(|i| {
                    let (k, j, x) = (i / n, (i % n) / nx, i % nx);
                    let a = k * ny * (nx + 1) + j * (nx + 1) + x;
                    0.5 * (us[a] + us[a + 1])
                })
                .collect();
            drop(us);
            let vd: Vec<f64> = (0..nz * n)
                .into_par_iter()
                .map(|i| {
                    let (k, j, x) = (i / n, (i % n) / nx, i % nx);
                    let a = k * (ny + 1) * nx + j * nx + x;
                    0.5 * (vs[a] + vs[a + nx])
                })
                .collect();
            drop(vs);
            let one = vec![1.0; n];
            let zero = vec![0.0; n];
            let cos = two.remove("COSALPHA").unwrap_or(one);
            let sin = two.remove("SINALPHA").unwrap_or(zero);
            match rustwx_calc::hubwind::interpolate(
                rustwx_calc::hubwind::Columns {
                    cells: n,
                    levels: nz,
                    z_agl_m: &hagl,
                    u_ms: &ud,
                    v_ms: &vd,
                    sin: &sin,
                    cos: &cos,
                    tke: None,
                    surface: Some((u10, v10)),
                },
                &[80.0, 100.0],
            ) {
                Ok(hub) => {
                    let mut hub = hub.into_iter();
                    let (h80, h100) = (hub.next().unwrap(), hub.next().unwrap());
                    planes.push(Plane { name: "wspd80_ms", units: "m/s", data: h80.speed });
                    planes.push(Plane { name: "wdir80_deg", units: "degrees", data: h80.direction });
                    planes.push(Plane { name: "wspd100_ms", units: "m/s", data: h100.speed });
                    planes.push(Plane { name: "wdir100_deg", units: "degrees", data: h100.direction });
                }
                Err(e) => miss(&mut absent, &["wspd80_ms", "wdir80_deg", "wspd100_ms", "wdir100_deg"], e),
            }
            f.store_cached_field(format!("u_destag_{ti}"), ud);
            f.store_cached_field(format!("v_destag_{ti}"), vd);
            match two.get("PBLH") {
                Some(pblh) => {
                    let gust = (|| {
                        let wspd = getvar(f, ti, "wspd", None, n, nz)?;
                        let w10 = getvar(f, ti, "wspd10", None, n, 1)?;
                        Ok::<_, String>(sfc::gust_10m(&hagl, &wspd, pblh, &w10, nz, n))
                    })();
                    match gust {
                        Ok(v) => planes.push(Plane { name: "gust_ms", units: "m/s", data: v }),
                        Err(e) => miss(&mut absent, &["gust_ms"], e),
                    }
                }
                None => miss(&mut absent, &["gust_ms"], "the frame carries no PBLH".into()),
            }
        } else {
            miss(&mut absent, &["gust_ms", "wspd80_ms", "wdir80_deg", "wspd100_ms", "wdir100_deg"], "U/V are not on the staggered grid".into());
        }
    } else {
        miss(&mut absent, &["gust_ms", "wspd80_ms", "wdir80_deg", "wspd100_ms", "wdir100_deg"], "the frame carries no U or V".into());
    }
    drop(hagl);
    f.clear_cache();
    lap(&mut stages, "wind_ms");

    // 7. Visibility, from the hydrometeors. Without rain and snow mixing ratios it would read clear air through
    // precipitation, so it needs QCLOUD, QRAIN and QSNOW (ice and graupel are used when present).
    let mut q = input::read_whole(f, path, ti, &["QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP"])?;
    if let (Some(c), true, true) = (q.remove("QCLOUD"), q.contains_key("QRAIN"), q.contains_key("QSNOW")) {
        let shape = rustwx_calc::GridShape::new(nx, ny).map_err(|e| e.to_string())?;
        let pressure = pressure; // the grid's own P+PB (kept by Arc)
        let vis = isolate("visibility", || {
            rustwx_calc::visibility::compute_visibility(rustwx_calc::visibility::VisibilityInputs {
                grid: shape,
                nz,
                pressure_pa: &pressure,
                temperature_k: &tk_arc,
                vapor_kgkg: &qv_arc,
                cloud_kgkg: &c,
                rain_kgkg: q.get("QRAIN").map(Vec::as_slice),
                ice_kgkg: q.get("QICE").map(Vec::as_slice),
                snow_kgkg: q.get("QSNOW").map(Vec::as_slice),
                graupel_kgkg: q.get("QGRAUP").map(Vec::as_slice),
            })
            .map_err(|e| e.to_string())
        });
        match vis {
            Ok(out) => planes.push(Plane { name: "visibility_m", units: "m", data: out.visibility_m.iter().map(|v| *v as f32).collect() }),
            Err(e) => miss(&mut absent, &["visibility_m"], e),
        }
    } else {
        miss(&mut absent, &["visibility_m"], "the frame carries no QCLOUD, QRAIN or QSNOW".into());
    }
    drop(q);
    lap(&mut stages, "visibility_ms");

    let mut notes = notes_table();
    let held: std::collections::HashSet<&str> = planes.iter().map(|p| p.name).collect();
    notes.retain(|k, _| held.contains(k.as_str()) || ["psfc_hpa", "orog_m", "t2_c", "td2_c", "u10_ms", "v10_ms"].contains(&k.as_str()));
    notes.insert("absent".into(), Value::Object(absent));
    Ok(Surface { planes, notes: Value::Object(notes), stages })
}
