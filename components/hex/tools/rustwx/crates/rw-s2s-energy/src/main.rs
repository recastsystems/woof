//! Native-grid S2S energy conversion and reduction. No Python array path.
use chrono::{Datelike, Duration, NaiveDateTime};
use rayon::prelude::*;
use rw_store::netcdf_classic::{
    NcAttr, NcClassicWriter, NcData, NcDim, NcFormat, NcSlab, NcType, NcVarDef,
};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    fs,
    path::{Path, PathBuf},
    time::Instant,
};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;
mod normal;
#[derive(Deserialize)]
struct Request {
    config: Config,
    start: String,
    cadence_hours: usize,
    days: usize,
    grid: PathBuf,
    output: PathBuf,
    step_degrees: f64,
    windows: Vec<Window>,
    members: Vec<Member>,
}
#[derive(Deserialize)]
struct Config {
    rows: Vec<Value>,
    table_sha256: String,
    heights_m_agl: Vec<f64>,
    climatology: Option<PathBuf>,
}
#[derive(Deserialize)]
struct Window {
    name: String,
    low: usize,
    high: usize,
}
#[derive(Deserialize)]
struct Member {
    id: String,
    init: PathBuf,
    histories: Vec<History>,
}
#[derive(Deserialize)]
struct History {
    valid: String,
    path: PathBuf,
}
#[derive(Clone)]
struct Product {
    id: String,
    units: String,
    height: Option<f64>,
    row: Value,
    scale: f32,
    energy: bool,
}
const FORMAT: &str = "%Y-%m-%d_%H:%M:%S";
fn num(row: &Value, key: &str) -> f64 {
    row[key].as_f64().unwrap_or(f64::NAN)
}
fn text<'a>(row: &'a Value, key: &str) -> &'a str {
    row[key].as_str().unwrap_or("")
}
fn attr(file: &netcrust::File, key: &str) -> Result<String> {
    Ok(file
        .attribute(key)
        .and_then(|v| v.as_string().map(str::to_owned))
        .ok_or_else(|| format!("missing string attribute {key}"))?)
}
fn read(file: &netcrust::File, names: &[&str], shape: &[usize]) -> Result<Vec<f64>> {
    for name in names {
        if file.variable(name).is_none() {
            continue;
        }
        let array = file.read_array_f64(name)?;
        // Only a single leading Time record may be stripped. Never transpose by length.
        let actual = array.shape();
        let has_time = file
            .variable(name)
            .unwrap()
            .dimensions()
            .first()
            .is_some_and(|d| d.name() == "Time");
        let actual = if has_time && actual.first() == Some(&1) {
            &actual[1..]
        } else {
            actual
        };
        if actual != shape {
            return Err(format!("{name}: shape {actual:?}, expected {shape:?}").into());
        }
        let variable = file.variable(name).unwrap();
        let fill = variable.attribute("_FillValue").and_then(|a| a.as_f64());
        let missing = variable.attribute("missing_value").and_then(|a| a.as_f64());
        let mut values = array.into_values();
        values.par_iter_mut().for_each(|v| {
            if Some(*v) == fill || Some(*v) == missing || v.abs() > 1e30 {
                *v = f64::NAN;
            }
        });
        return Ok(values);
    }
    Err(format!("missing native field, tried {names:?}").into())
}
fn read_units(
    file: &netcrust::File,
    names: &[&str],
    shape: &[usize],
    units: &[&str],
) -> Result<Vec<f64>> {
    let name = names
        .iter()
        .find(|n| file.variable(n).is_some())
        .ok_or_else(|| format!("missing field {names:?}"))?;
    let variable = file.variable(name).unwrap();
    let unit = variable
        .attribute("units")
        .and_then(|a| a.as_string().map(str::to_owned))
        .unwrap_or_default();
    if !units.contains(&unit.as_str()) {
        return Err(format!("{name} units {unit:?}, expected {units:?}").into());
    }
    read(file, &[*name], shape)
}
fn products(config: &Config) -> Result<Vec<Product>> {
    let mut output = Vec::new();
    for row in &config.rows {
        let kind = text(row, "kind");
        if !["wind", "ghi", "turbine", "pv"].contains(&kind)
            || !num(row, "display_max").is_finite()
            || num(row, "display_max") <= 0.
        {
            return Err("invalid energy row".into());
        }
        let heights = if kind == "wind" || kind == "turbine" {
            config.heights_m_agl.iter().copied().map(Some).collect()
        } else {
            vec![None]
        };
        for height in heights {
            let id = if let Some(h) = height {
                format!("{}_{}m", text(row, "id"), h.to_string().replace('.', "p"))
            } else {
                text(row, "id").into()
            };
            output.push(Product {
                id: id.clone(),
                units: text(row, "units").into(),
                height,
                row: row.clone(),
                scale: num(row, "display_max") as f32,
                energy: false,
            });
            if kind == "pv" || kind == "turbine" {
                output.push(Product {
                    id: format!("{id}_energy"),
                    units: "MWh".into(),
                    height,
                    row: row.clone(),
                    scale: num(row, "display_max") as f32,
                    energy: true,
                });
            }
        }
    }
    if output.is_empty() {
        return Err("empty energy products".into());
    }
    Ok(output)
}
fn hub_speed(z: &[f64], u: &[f64], v: &[f64], u10: f64, v10: f64, height: f64) -> f64 {
    if height <= 10.
        || !height.is_finite()
        || z.len() != u.len() + 1
        || u.len() != v.len()
        || z.iter().any(|a| !a.is_finite())
        || z.windows(2).any(|a| a[1] <= a[0])
    {
        return f64::NAN;
    }
    let mut lo = 10.;
    let mut a = u10;
    let mut b = v10;
    for k in 0..u.len() {
        let hi = (z[k] + z[k + 1]) * 0.5 - z[0];
        if hi <= 10. {
            continue;
        }
        if height <= hi {
            let weight = (height - lo) / (hi - lo);
            return (a + (u[k] - a) * weight).hypot(b + (v[k] - b) * weight);
        }
        lo = hi;
        a = u[k];
        b = v[k];
    }
    f64::NAN
}
fn power(row: &Value, speed: f64) -> f64 {
    if !speed.is_finite() || speed < 0. {
        return f64::NAN;
    }
    if speed >= num(row, "cut_out_ms") {
        return 0.;
    }
    let Some(curve) = row["curve"].as_array() else {
        return f64::NAN;
    };
    for pair in curve.windows(2) {
        let (a, b) = (pair[0][0].as_f64().unwrap(), pair[1][0].as_f64().unwrap());
        if speed <= b {
            let (p, q) = (pair[0][1].as_f64().unwrap(), pair[1][1].as_f64().unwrap());
            return num(row, "rated_mw") * (p + (q - p) * (speed - a) / (b - a));
        }
    }
    num(row, "rated_mw") * curve.last().unwrap()[1].as_f64().unwrap()
}
fn pv(row: &Value, ghi: f64, temperature_k: f64) -> f64 {
    if !ghi.is_finite()
        || ghi < 0.
        || !temperature_k.is_finite()
        || temperature_k < 150.
        || temperature_k > 400.
    {
        return f64::NAN;
    }
    let cell_c = temperature_k - 273.15 + (num(row, "noct_c") - 20.) * ghi / 800.;
    let dc_fraction =
        ghi / 1000. * (1. + num(row, "temperature_coefficient_per_k") * (cell_c - 25.)).max(0.);
    (num(row, "rated_mw")
        * num(row, "dc_ac_ratio")
        * dc_fraction
        * (1. - num(row, "loss_fraction"))
        * num(row, "inverter_efficiency"))
    .clamp(0., num(row, "rated_mw"))
}
fn clock_matches(stamps: &[String], valid: &str) -> bool {
    // HDF5 fixed-width S1 arrays are exposed as one string per character;
    // classic NC_CHAR arrays are exposed as one string per record.
    if stamps.len() == 1 {
        stamps[0].trim_matches(['\0', ' ']) == valid
    } else {
        stamps.iter().all(|s| s.len() <= 1) && stamps.concat().trim_matches(['\0', ' ']) == valid
    }
}
fn diagnostic(
    path: &Path,
    products: &[Product],
    z: &[f64],
    n: usize,
    nz: usize,
    valid: &str,
) -> Result<Vec<Vec<f64>>> {
    let file = netcrust::open(path)?;
    // Native histories carry the true clock in xtime, independent of basename.
    let stamps = file.read_strings("xtime")?;
    let clock = file.variable("xtime").ok_or("missing xtime")?;
    let dims = clock.dimensions();
    let one_record = dims.len() == 2 && dims[0].len() == 1;
    if !one_record || !clock_matches(&stamps, valid) {
        return Err(format!("{} xtime does not match {valid}", path.display()).into());
    }
    let speed_units = ["m s-1", "m s^{-1}", "m s^-1", "m/s"];
    let (u, v) = rayon::join(
        || {
            read_units(
                &file,
                &["u_zonal", "uReconstructZonal"],
                &[n, nz],
                &speed_units,
            )
        },
        || {
            read_units(
                &file,
                &["v_meridional", "uReconstructMeridional"],
                &[n, nz],
                &speed_units,
            )
        },
    );
    let (u, v) = (u?, v?);
    let (u10, v10) = rayon::join(
        || read_units(&file, &["u10"], &[n], &speed_units),
        || read_units(&file, &["v10"], &[n], &speed_units),
    );
    let (u10, v10) = (u10?, v10?);
    let (ghi, t) = rayon::join(
        || {
            read_units(
                &file,
                &["swdown", "swdnb", "SWDOWN"],
                &[n],
                &["W m-2", "W m^{-2}", "W m^-2"],
            )
        },
        || read_units(&file, &["t2", "t2m"], &[n], &["K"]),
    );
    let (ghi, t) = (ghi?, t?);
    Ok(products
        .par_iter()
        .map(|p| {
            (0..n)
                .into_par_iter()
                .map(|i| {
                    let speed = || {
                        hub_speed(
                            &z[i * (nz + 1)..(i + 1) * (nz + 1)],
                            &u[i * nz..(i + 1) * nz],
                            &v[i * nz..(i + 1) * nz],
                            u10[i],
                            v10[i],
                            p.height.unwrap(),
                        )
                    };
                    match text(&p.row, "kind") {
                        "wind" => speed(),
                        "turbine" => power(&p.row, speed()),
                        "ghi" => {
                            if ghi[i] >= 0. {
                                ghi[i]
                            } else {
                                f64::NAN
                            }
                        }
                        "pv" => pv(&p.row, ghi[i], t[i]),
                        _ => f64::NAN,
                    }
                })
                .collect()
        })
        .collect())
}
fn stats(values: &[f64], baseline: Option<f64>) -> [f32; 7] {
    let count = values.iter().filter(|x| x.is_finite()).count();
    if count != values.len() {
        return [
            f32::NAN,
            f32::NAN,
            count as f32,
            f32::NAN,
            f32::NAN,
            0.,
            f32::NAN,
        ];
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let sd = (values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64).sqrt();
    let Some(b) = baseline.filter(|b| b.is_finite()) else {
        return [
            mean as f32,
            sd as f32,
            count as f32,
            f32::NAN,
            f32::NAN,
            0.,
            f32::NAN,
        ];
    };
    [
        mean as f32,
        sd as f32,
        count as f32,
        (mean - b) as f32,
        sd as f32,
        count as f32,
        values.iter().filter(|&&x| x > b).count() as f32 / values.len() as f32,
    ]
}
fn write_window(
    req: &Request,
    window: &Window,
    products: &[Product],
    planes: &[Vec<Vec<f64>>],
    lat: &[f32],
    lon: &[f32],
    ny: usize,
    nx: usize,
) -> Result<Value> {
    let normal = req
        .config
        .climatology
        .as_ref()
        .map(netcrust::open)
        .transpose()?;
    if let Some(ref file) = normal {
        if attr(file, "s2s_energy_table_sha256")? != req.config.table_sha256
            || attr(file, "s2s_energy_heights_m_agl")?
                != serde_json::to_string(&req.config.heights_m_agl)?
            || attr(file, "s2s_climatology_start_mmdd")?
                != NaiveDateTime::parse_from_str(&req.start, FORMAT)?
                    .format("%m-%d")
                    .to_string()
        {
            return Err(
                "energy climatology table, hub heights or calendar differs from forecast".into(),
            );
        }
        let year = NaiveDateTime::parse_from_str(&req.start, FORMAT)?.year() as f64;
        let first = file
            .attribute("climatology_year_start")
            .and_then(|a| a.as_f64())
            .ok_or("normal year start")?;
        let last = file
            .attribute("climatology_year_end")
            .and_then(|a| a.as_f64())
            .ok_or("normal year end")?;
        if first.fract() != 0. || last.fract() != 0. || first >= last || last >= year {
            return Err(
                "climatology must span multiple complete years preceding the forecast".into(),
            );
        }
        for (key, expected) in [("XLAT", lat), ("XLONG", lon)] {
            let actual = read(file, &[key], &[ny, nx])?;
            if actual
                .iter()
                .zip(expected)
                .any(|(a, b)| !a.is_finite() || (a - *b as f64).abs() > 1e-5)
            {
                return Err("energy climatology coordinates differ".into());
            }
        }
    }
    let baseline = if normal.is_some() {
        "historical energy climatology"
    } else {
        "historical energy anomalies unavailable; no energy climatology"
    };
    let valid = (NaiveDateTime::parse_from_str(&req.start, FORMAT)?
        + Duration::days(window.high as i64))
    .format(FORMAT)
    .to_string();
    let mut attrs = vec![
        NcAttr::text("ensemble_contract", "gpuwm-resident-ensemble-products.v1"),
        NcAttr::text("probability_scale", "fraction"),
        NcAttr::int("ensemble_members", req.members.len() as i32),
        NcAttr::text("valid_time", valid),
        NcAttr::text("s2s_window", &window.name),
        NcAttr::text("s2s_baseline", baseline),
        NcAttr::text("s2s_energy_table_sha256", &req.config.table_sha256),
        NcAttr::text(
            "s2s_energy_heights_m_agl",
            serde_json::to_string(&req.config.heights_m_agl)?,
        ),
        NcAttr::text(
            "s2s_sampling",
            "6-hour sampled endpoint trapezoids; conversion before averaging",
        ),
        NcAttr::text("s2s_energy_rows", serde_json::to_string(&req.config.rows)?),
    ];
    let mut defs = Vec::new();
    let mut payloads: Vec<(String, Vec<f32>)> = Vec::new();
    for (name, data, unit) in [
        ("XLAT", lat, "degrees_north"),
        ("XLONG", lon, "degrees_east"),
    ] {
        defs.push(
            NcVarDef::new(name, NcType::Float, vec![0, 1])
                .with_attrs(vec![NcAttr::text("units", unit)]),
        );
        payloads.push((name.into(), data.to_vec()));
    }
    let dims = vec![
        NcDim::fixed("south_north", ny),
        NcDim::fixed("west_east", nx),
        NcDim::fixed("energy_threshold", 1),
    ];
    let mut summary = serde_json::Map::new();
    for (pi, p) in products.iter().enumerate() {
        let baseline_values = if let Some(ref file) = normal {
            let key = format!("{}_{}", window.name.replace('-', "_"), p.id);
            Some(read_units(file, &[&key], &[ny, nx], &[&p.units])?)
        } else {
            None
        };
        let computed: Vec<_> = (0..ny * nx)
            .into_par_iter()
            .map(|i| {
                stats(
                    &planes[pi].iter().map(|m| m[i]).collect::<Vec<_>>(),
                    baseline_values.as_ref().map(|b| b[i]),
                )
            })
            .collect();
        let scale = if p.energy {
            p.scale * ((window.high - window.low) * 24) as f32
        } else {
            p.scale
        };
        // A weekly generation anomaly can be much smaller than nameplate
        // capacity. Keep its diverging scale symmetric and show its actual
        // finite range instead of washing out the PV signal at the mean scale.
        let anomalies: Vec<_> = computed.iter().map(|s| s[3])
            .filter(|v| v.is_finite()).collect();
        let anomaly_extent = anomalies.iter().fold(0f32, |m, v| m.max(v.abs()));
        let anomaly_scale = if anomaly_extent > 0. { anomaly_extent } else { scale };
        for (field, anomaly) in [(&p.id, false), (&format!("{}_anomaly", p.id), true)] {
            attrs.push(NcAttr::float(format!("{field}_s2s_display_max"),
                if anomaly { anomaly_scale } else { scale }));
            attrs.push(NcAttr::int(
                format!("{field}_s2s_is_anomaly"),
                anomaly as i32,
            ));
        }
        for (k, suffix) in [
            "mean",
            "spread",
            "finite_count",
            "anomaly_mean",
            "anomaly_spread",
            "anomaly_finite_count",
            "anomaly_probability",
        ]
        .iter()
        .enumerate()
        {
            let name = format!("{}_{suffix}", p.id);
            let prob = k == 6;
            defs.push(
                NcVarDef::new(
                    &name,
                    NcType::Float,
                    if prob { vec![2, 0, 1] } else { vec![0, 1] },
                )
                .with_attrs(vec![
                    NcAttr::text("units", if k == 2 || k >= 5 { "1" } else { &p.units }),
                    NcAttr::text("comparison", "gt"),
                ]),
            );
            payloads.push((name, computed.iter().map(|s| s[k]).collect()));
        }
        let name = format!("{}_anomaly_thresholds", p.id);
        // The renderer requires each probability's first dimension to be named after its field.
        // A shared dimension is renamed below through distinct per-field definitions.
        defs.push(
            NcVarDef::new(&name, NcType::Float, vec![2])
                .with_attrs(vec![NcAttr::text("units", &p.units)]),
        );
        payloads.push((name, vec![0.]));
        let finite: Vec<_> = computed
            .iter()
            .map(|s| s[0])
            .filter(|v| v.is_finite())
            .collect();
        summary.insert(p.id.clone(),json!({"units":p.units,"finite":finite.len(),"cells":ny*nx,
            "min":finite.iter().copied().reduce(f32::min),"max":finite.iter().copied().reduce(f32::max),
            "anomaly":{"finite":anomalies.len(),"cells":ny*nx,
                "min":anomalies.iter().copied().reduce(f32::min),
                "max":anomalies.iter().copied().reduce(f32::max),
                "symmetric_display_max":anomaly_scale}}));
    }
    let mut dims = dims;
    for p in products {
        let dim = dims.len();
        dims.push(NcDim::fixed(format!("{}_anomaly_threshold", p.id), 1));
        for def in &mut defs {
            if def.name == format!("{}_anomaly_probability", p.id) {
                def.dimids = vec![dim, 0, 1];
            }
            if def.name == format!("{}_anomaly_thresholds", p.id) {
                def.dimids = vec![dim];
            }
        }
    }
    let path = req.output.join(format!("{}.nc", window.name));
    let temporary = path.with_extension("nc.part");
    let mut writer = NcClassicWriter::create(&temporary, NcFormat::Data64, dims, attrs, defs, 0)?;
    let jobs: Vec<_> = payloads
        .iter()
        .map(|(name, _)| (name.as_str(), None))
        .collect();
    writer.put_parallel(&jobs, |i| {
        Ok(NcSlab::Borrowed(NcData::Floats(&payloads[i].1)))
    })?;
    writer.finish()?;
    fs::rename(temporary, &path)?;
    Ok(
        json!({"name":window.name,"file":path,"days":[window.low,window.high],"field_summary":summary}),
    )
}
/// The s2s door's own minimum (hexcore.s2s_door.MIN_DAYS = 7: week-1 is the shortest window) and no
/// ceiling, as the door has none. Breakage: the old 30..=90 bound refused every s2s run of 7 to 29 days
/// at the products stage ("invalid S2S energy request"; a 7-day GEFS run, 2026-10-23) and every run
/// past 90 days.
fn days_supported(days: usize) -> bool {
    days >= 7
}
fn run(req: Request) -> Result<()> {
    let started = Instant::now();
    let start = NaiveDateTime::parse_from_str(&req.start, FORMAT)?;
    if req.cadence_hours != 6
        || !days_supported(req.days)
        || req.members.is_empty()
        || req.members.len() > 1000
        || ![0.5, 1.].contains(&req.step_degrees)
        || req.windows.is_empty()
        || req.config.heights_m_agl.is_empty()
        || req
            .config
            .heights_m_agl
            .iter()
            .any(|h| !h.is_finite() || *h <= 10. || *h > 1000.)
    {
        return Err("invalid S2S energy request".into());
    }
    for w in &req.windows {
        if w.high <= w.low || w.high > req.days || ![7, 14].contains(&(w.high - w.low)) {
            return Err("invalid S2S window".into());
        }
    }
    let prods = products(&req.config)?;
    let grid = netcrust::open(&req.grid)?;
    let n = grid.dimension("nCells").ok_or("grid nCells")?.len();
    let lat = read_units(&grid, &["latCell"], &[n], &["radians", "radian", "rad"])?;
    let lon = read_units(&grid, &["lonCell"], &[n], &["radians", "radian", "rad"])?;
    if lat.iter().zip(&lon).any(|(a, b)| {
        !a.is_finite() || a.abs() > std::f64::consts::FRAC_PI_2 + 1e-6 || !b.is_finite()
    }) {
        return Err("invalid native coordinates".into());
    }
    let init = netcrust::open(&req.members[0].init)?;
    let nz = init
        .dimension("nVertLevels")
        .ok_or("init nVertLevels")?
        .len();
    let z = read_units(&init, &["zgrid"], &[n, nz + 1], &["m", "m MSL"])?;
    if z.par_chunks(nz + 1)
        .any(|col| col.iter().any(|v| !v.is_finite()) || col.windows(2).any(|v| v[1] <= v[0]))
    {
        return Err("invalid native height interfaces".into());
    }
    let nx = (360. / req.step_degrees).round() as usize;
    let ny = (180. / req.step_degrees).round() as usize + 1;
    let targets: Vec<_> = (0..ny * nx)
        .into_par_iter()
        .map(|i| {
            let (a, b) = (
                -90. + (i / nx) as f64 * req.step_degrees,
                -180. + (i % nx) as f64 * req.step_degrees,
            );
            let (a, b) = (a.to_radians(), b.to_radians());
            [a.cos() * b.cos(), a.cos() * b.sin(), a.sin()]
        })
        .collect();
    let sources: Vec<_> = lat
        .par_iter()
        .zip(&lon)
        .map(|(a, b)| [a.cos() * b.cos(), a.cos() * b.sin(), a.sin()])
        .collect();
    let index: Vec<_> = targets
        .par_iter()
        .map(|t| {
            sources
                .iter()
                .enumerate()
                .map(|(i, s)| (i, t[0] * s[0] + t[1] * s[1] + t[2] * s[2]))
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .unwrap()
                .0
        })
        .collect();
    let ylats: Vec<f32> = (0..ny * nx)
        .map(|i| (-90. + (i / nx) as f64 * req.step_degrees) as f32)
        .collect();
    let xlons: Vec<f32> = (0..ny * nx)
        .map(|i| (-180. + (i % nx) as f64 * req.step_degrees) as f32)
        .collect();
    let remap_seconds = started.elapsed().as_secs_f64();
    let mut reduced: Vec<Vec<Vec<Vec<f64>>>> = req
        .windows
        .iter()
        .map(|_| prods.iter().map(|_| Vec::new()).collect())
        .collect();
    let mut coverage = Vec::new();
    for member in &req.members {
        let init = netcrust::open(&member.init)?;
        if read_units(&init, &["zgrid"], &[n, nz + 1], &["m", "m MSL"])? != z {
            return Err("member height interfaces differ".into());
        }
        if member.histories.len() != req.days * 24 / req.cadence_hours + 1 {
            return Err("missing scheduled histories".into());
        }
        let mut sums: Vec<Vec<Vec<f64>>> = req
            .windows
            .iter()
            .map(|_| prods.iter().map(|_| vec![0.; n]).collect())
            .collect();
        for (at, history) in member.histories.iter().enumerate() {
            let valid = (start + Duration::hours((at * req.cadence_hours) as i64))
                .format(FORMAT)
                .to_string();
            if history.valid != valid {
                return Err("history clock not on exact schedule".into());
            }
            let sample = diagnostic(&history.path, &prods, &z, n, nz, &valid)?;
            for (wi, w) in req.windows.iter().enumerate() {
                let (lo, hi) = (
                    w.low * 24 / req.cadence_hours,
                    w.high * 24 / req.cadence_hours,
                );
                if at < lo || at > hi {
                    continue;
                }
                let weight = if at == lo || at == hi { 0.5 } else { 1. };
                sums[wi]
                    .par_iter_mut()
                    .zip(&sample)
                    .for_each(|(sum, plane)| {
                        sum.par_iter_mut()
                            .zip(plane)
                            .for_each(|(a, b)| *a += b * weight)
                    });
            }
        }
        for (wi, w) in req.windows.iter().enumerate() {
            let intervals = ((w.high - w.low) * 24 / req.cadence_hours) as f64;
            for (pi, p) in prods.iter().enumerate() {
                let scale = if p.energy {
                    ((w.high - w.low) * 24) as f64
                } else {
                    1.
                };
                reduced[wi][pi].push(
                    index
                        .par_iter()
                        .map(|&i| sums[wi][pi][i] / intervals * scale)
                        .collect(),
                );
            }
        }
        coverage.push(json!({"member":member.id,"frames":member.histories.len(),"first":member.histories.first().unwrap().valid,"last":member.histories.last().unwrap().valid}));
    }
    let reduction_seconds = started.elapsed().as_secs_f64() - remap_seconds;
    fs::create_dir_all(&req.output)?;
    let windows: Result<Vec<_>> = req
        .windows
        .par_iter()
        .enumerate()
        .map(|(wi, w)| write_window(&req, w, &prods, &reduced[wi], &ylats, &xlons, ny, nx))
        .collect();
    let receipt = json!({"schema":"gpuwm-hex.s2s-energy-reduction/v1","windows":windows?,"coverage":coverage,
        "baseline":if req.config.climatology.is_some(){"historical energy climatology"}else{"historical energy anomalies unavailable; no energy climatology"},
        "table_sha256":req.config.table_sha256,"heights_m_agl":req.config.heights_m_agl,"native_cells":n,"levels":nz,
        "seconds":started.elapsed().as_secs_f64(),"remap_seconds":remap_seconds,"reduction_seconds":reduction_seconds,
        "threads":rayon::current_num_threads(),"peak_vram_mib":0,"backend":"Rust Rayon; native-grid conversion before reduction; spherical nearest-cell remap"});
    fs::write(
        req.output.join("reduction-receipt.json"),
        serde_json::to_vec_pretty(&receipt)?,
    )?;
    Ok(())
}
fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let path = args.next().ok_or("usage: rw_s2s_energy REQUEST.json")?;
    if path == "--build-climatology" {
        let manifest = args.next().ok_or("--build-climatology MANIFEST.json")?;
        return normal::build(&PathBuf::from(manifest));
    }
    if path == "--abi" {
        println!("rw-s2s-energy/v1 native-netcdf rayon sampled-power dated-climatology");
        return Ok(());
    }
    // 0.6.0 read `--help` as a request file and answered "No such file or directory" (exit 1).
    if path == "--help" || path == "-h" {
        println!("{}", USAGE);
        return Ok(());
    }
    run(serde_json::from_slice(&fs::read(path)?)?)
}
/// What `--help` prints. `woof hex s2s` writes the request file and calls this program; it is not run by hand.
const USAGE: &str = "usage: rw_s2s_energy REQUEST.json\n       rw_s2s_energy --build-climatology MANIFEST.json\n       rw_s2s_energy --abi | --help\n\nThe hub-height wind and solar energy tables of a woof hex s2s forecast.\nREQUEST.json is written by `woof hex s2s` (its products stage).";
#[cfg(test)]
mod usage_tests {
    #[test]
    fn help_names_every_form() {
        for word in ["REQUEST.json", "--build-climatology", "--abi", "--help"] {
            assert!(super::USAGE.contains(word), "{word}");
        }
    }
}
#[cfg(test)]
mod tests {
    #[test]
    fn the_request_bound_is_the_door_minimum() {
        assert!(!super::days_supported(6));
        for days in [7, 14, 29, 30, 46, 90, 120] {
            assert!(super::days_supported(days), "days {days}");
        }
    }
    use super::*;
    #[test]
    fn native_s1_and_classic_clock_encodings() {
        let valid = "2026-10-20_00:00:00";
        let padded = format!("{valid:64}");
        let chars: Vec<String> = padded.chars().map(|c| c.to_string()).collect();
        assert!(clock_matches(&chars, valid));
        assert!(clock_matches(&[padded], valid));
        assert!(!clock_matches(&chars, "2026-10-20_06:00:00"));
        assert!(!clock_matches(&[valid.into(), valid.into()], valid));
    }
    #[test]
    fn vectors_before_speed_and_agl_anchor() {
        assert_eq!(
            hub_speed(&[1200., 1400., 1600.], &[0., 0.], &[10., 20.], 10., 0., 55.),
            (5f64).hypot(5.)
        );
        assert!(
            hub_speed(
                &[1200., 1400., 1600.],
                &[0., 0.],
                &[10., 20.],
                10.,
                0.,
                500.
            )
            .is_nan()
        );
    }
    #[test]
    fn convert_before_average_cutout_and_missing() {
        let row =
            json!({"rated_mw":2.,"cut_out_ms":25.,"curve":[[0.,0.],[3.,0.],[12.,1.],[24.99,1.]]});
        assert_eq!(power(&row, 25.), 0.);
        assert_eq!(power(&row, 12.), 2.);
        assert_ne!((power(&row, 0.) + power(&row, 24.)) / 2., power(&row, 12.));
        assert!(power(&row, f64::NAN).is_nan());
    }
    #[test]
    fn pv_night_clipping_temperature() {
        let row = json!({"rated_mw":1.,"dc_ac_ratio":1.2,"loss_fraction":0.14,"inverter_efficiency":0.96,"noct_c":45.,"temperature_coefficient_per_k":-0.004});
        assert_eq!(pv(&row, 0., 300.), 0.);
        assert!(pv(&row, 1000., 310.) < pv(&row, 1000., 280.));
        assert_eq!(pv(&row, 3000., 250.), 1.);
        assert!(pv(&row, -1., 300.).is_nan());
    }
    #[test]
    fn fixed_roster_historical_baseline() {
        assert_eq!(stats(&[1., 3.], Some(1.)), [2., 1., 2., 1., 1., 2., 0.5]);
        assert!(stats(&[1., f64::NAN], Some(1.))[0].is_nan());
        assert!(stats(&[1., 3.], None)[3].is_nan());
        assert_eq!(stats(&[1., 3.], None)[0], 2.);
    }
    #[test]
    fn aggregate_roundtrip_and_dated_normal_contract() -> Result<()> {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test-output");
        fs::create_dir_all(&dir)?;
        let normal = dir.join("normal.nc");
        let lat = vec![-1f32, -1., 1., 1.];
        let lon = vec![-1f32, 1., -1., 1.];
        let arrays = [
            ("XLAT", lat.clone(), "degrees_north"),
            ("XLONG", lon.clone(), "degrees_east"),
            ("week_1_wind", vec![1., 2., 3., 4.], "m s-1"),
        ];
        let defs = arrays
            .iter()
            .map(|(name, _, unit)| {
                NcVarDef::new(*name, NcType::Float, vec![0, 1])
                    .with_attrs(vec![NcAttr::text("units", *unit)])
            })
            .collect();
        let mut writer = NcClassicWriter::create(
            &normal,
            NcFormat::Data64,
            vec![NcDim::fixed("south_north", 2), NcDim::fixed("west_east", 2)],
            vec![
                NcAttr::text("s2s_energy_table_sha256", "test-row-hash"),
                NcAttr::text("s2s_energy_heights_m_agl", "[87.5]"),
                NcAttr::text("s2s_climatology_start_mmdd", "03-31"),
                NcAttr::int("climatology_year_start", 2000),
                NcAttr::int("climatology_year_end", 2020),
            ],
            defs,
            0,
        )?;
        let jobs: Vec<_> = arrays.iter().map(|(name, _, _)| (*name, None)).collect();
        writer.put_parallel(&jobs, |i| {
            Ok(NcSlab::Borrowed(NcData::Floats(&arrays[i].1)))
        })?;
        writer.finish()?;
        let mut req = Request {
            config: Config {
                rows: vec![],
                table_sha256: "test-row-hash".into(),
                heights_m_agl: vec![87.5],
                climatology: Some(normal),
            },
            start: "2023-03-31_12:00:00".into(),
            cadence_hours: 6,
            days: 30,
            grid: PathBuf::new(),
            output: dir.clone(),
            step_degrees: 1.,
            windows: vec![],
            members: vec![
                Member {
                    id: "a".into(),
                    init: PathBuf::new(),
                    histories: vec![],
                },
                Member {
                    id: "b".into(),
                    init: PathBuf::new(),
                    histories: vec![],
                },
            ],
        };
        let window = Window {
            name: "week-1".into(),
            low: 0,
            high: 7,
        };
        let product = Product {
            id: "wind".into(),
            units: "m s-1".into(),
            height: Some(87.5),
            row: json!({}),
            scale: 30.,
            energy: false,
        };
        let planes = vec![vec![vec![1., 3., f64::NAN, 5.], vec![3., 5., 7., 7.]]];
        write_window(&req, &window, &[product.clone()], &planes, &lat, &lon, 2, 2)?;
        let file = netcrust::open(dir.join("week-1.nc"))?;
        assert_eq!(read(&file, &["wind_mean"], &[2, 2])?[0], 2.);
        assert_eq!(read(&file, &["wind_anomaly_mean"], &[2, 2])?[1], 2.);
        assert_eq!(file.attribute("wind_s2s_display_max").and_then(|v| v.as_f64()), Some(30.));
        assert_eq!(file.attribute("wind_anomaly_s2s_display_max").and_then(|v| v.as_f64()), Some(2.));
        let probability = read(&file, &["wind_anomaly_probability"], &[1, 2, 2])?;
        assert_eq!(probability[0], 0.5);
        assert!(probability[2].is_nan());
        assert_eq!(
            file.variable("wind_anomaly_probability")
                .unwrap()
                .dimensions()[0]
                .name(),
            "wind_anomaly_threshold"
        );
        req.start = "2020-03-31_12:00:00".into();
        assert!(
            write_window(&req, &window, &[product.clone()], &planes, &lat, &lon, 2, 2)
                .unwrap_err()
                .to_string()
                .contains("preceding")
        );
        req.start = "2023-04-01_12:00:00".into();
        assert!(
            write_window(&req, &window, &[product.clone()], &planes, &lat, &lon, 2, 2)
                .unwrap_err()
                .to_string()
                .contains("calendar")
        );
        req.start = "2023-03-31_12:00:00".into();
        req.config.table_sha256 = "other".into();
        assert!(
            write_window(&req, &window, &[product], &planes, &lat, &lon, 2, 2)
                .unwrap_err()
                .to_string()
                .contains("table")
        );
        println!("retained contract-test NetCDF fixtures: {}", dir.display());
        Ok(())
    }
}
