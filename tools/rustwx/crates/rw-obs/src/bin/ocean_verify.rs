//! SST verification. NetCDF decoding delegates to the existing Rust bridge;
//! all observation parsing, matching, temporal averaging and scores stay here.
//! Engine MIT Rust bridge interfaces are reused; new lane code is Apache-2.0.
use chrono::{DateTime, NaiveDate, NaiveDateTime, Utc};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fs,
    io::Read,
    path::{Path, PathBuf},
    process::Command,
};
type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;
const ABI: &str = "gpuwm-ocean-sst-verify-v1\tdaily-mean\tndbc-WTMP\tgreat-circle\tpaired-change";
pub static GPUWM_BRIDGE_SOURCE_REV_STAMP: &str =
    concat!("GPUWM_BRIDGE_SOURCE_REV=", env!("GPUWM_BRIDGE_SOURCE_REV"));
fn bad(message: impl Into<String>) -> Box<dyn Error + Send + Sync> {
    std::io::Error::other(message.into()).into()
}
#[derive(Deserialize)]
struct Request {
    forecast: Vec<PathBuf>,
    #[serde(default = "sst_variable")]
    sst_variable: String,
    #[serde(default)] ocean_mask: Option<PathBuf>,
    oisst: Vec<PathBuf>,
    buoys: Vec<Buoy>,
    output: PathBuf,
    /// Optional raw planes for rw_wrfbatch; no Python field transforms.
    #[serde(default)]
    products: Option<PathBuf>,
    #[serde(default = "jobs")]
    jobs: usize,
    #[serde(default = "gap")]
    daily_max_gap_hours: u32,
    #[serde(default = "radius")]
    buoy_max_distance_km: f64,
    #[serde(default = "minutes")]
    buoy_match_minutes: i64,
}
fn sst_variable() -> String { "OCEAN_SST".into() }
fn jobs() -> usize {
    8
}
fn gap() -> u32 {
    3
}
fn radius() -> f64 {
    25.
}
fn minutes() -> i64 {
    30
}
#[derive(Deserialize)]
struct Buoy {
    id: String,
    path: PathBuf,
    latitude: f64,
    longitude: f64,
}
#[derive(Deserialize)]
struct Dump {
    name: String,
    filename: String,
    shape: Vec<usize>,
    dtype: String,
    units: Option<String>,
    times: Option<Vec<String>>,
}
struct Plane {
    record: Dump,
    bytes: Vec<u8>,
}
impl Plane {
    fn numbers(&self) -> Result<Vec<f64>> {
        if self.record.dtype != "<f8"
            || self.bytes.len() != self.record.shape.iter().product::<usize>() * 8
        {
            return Err(bad("invalid numeric dump extent/type"));
        }
        Ok(self
            .bytes
            .chunks_exact(8)
            .map(|b| f64::from_le_bytes(b.try_into().unwrap()))
            .collect())
    }
}
struct Work(PathBuf);
impl Drop for Work {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn dump(
    bridge: &Path,
    path: &Path,
    names: &[&str],
    work: &Path,
) -> Result<BTreeMap<String, Plane>> {
    fs::create_dir(work)?;
    let cleanup = Work(work.to_owned());
    let result = Command::new(bridge)
        .args(["dump"])
        .arg(path)
        .arg(work)
        .args(names)
        .output()?;
    if !result.status.success() {
        return Err(bad(format!(
            "{}: {}",
            path.display(),
            String::from_utf8_lossy(&result.stderr)
        )));
    }
    let metadata: Value = serde_json::from_slice(&fs::read(work.join("metadata.json"))?)?;
    if metadata["schema"] != "gpuwm-rw-netcdf-dump-v1" {
        return Err(bad("wrong NetCDF dump schema"));
    }
    let records: Vec<Dump> = serde_json::from_value(metadata["variables"].clone())?;
    let mut out = BTreeMap::new();
    for record in records {
        if Path::new(&record.filename).components().count() != 1 {
            return Err(bad("unsafe dump filename"));
        }
        let bytes = fs::read(work.join(&record.filename))?;
        out.insert(record.name.clone(), Plane { record, bytes });
    }
    drop(cleanup);
    Ok(out)
}
fn instant(s: &str) -> Result<DateTime<Utc>> {
    if let Ok(t) = DateTime::parse_from_rfc3339(s) {
        return Ok(t.with_timezone(&Utc));
    }
    Ok(NaiveDateTime::parse_from_str(s, "%Y-%m-%d_%H:%M:%S")?.and_utc())
}
fn physical(v: f64) -> bool {
    v.is_finite() && (260.0..=330.0).contains(&v)
}
fn unit_offset(units: &Option<String>) -> Result<f64> {
    match units.as_deref().unwrap_or("").to_lowercase().as_str() {
        "k" | "kelvin" => Ok(0.),
        "celsius" | "degc" | "degree_celsius" | "degrees_celsius" => Ok(273.15),
        other => Err(bad(format!(
            "SST units must explicitly be K or Celsius; got {other:?}"
        ))),
    }
}
#[derive(Clone)]
struct Frame {
    time: DateTime<Utc>,
    sst: Vec<f64>,
    lat: Vec<f64>,
    lon: Vec<f64>,
}
fn forecast(bridge: &Path, path: &Path, work: &Path, variable: &str) -> Result<Vec<Frame>> {
    let d = dump(bridge, path, &[variable, "XLAT", "XLONG", "XLAND", "Times"], work)?;
    let s = &d[variable];
    let land = d["XLAND"].numbers()?;
    let shape = &s.record.shape;
    if shape.len() != 3 || shape.contains(&0) {
        return Err(bad("forecast SST must have Time,y,x axes"));
    }
    let nt = shape[0];
    let n = shape[1] * shape[2];
    let values = s.numbers()?;
    if ![n,n*nt].contains(&land.len()) {return Err(bad("XLAND shape differs from SST"));}
    let offset = unit_offset(&s.record.units)?;
    let lat = d["XLAT"].numbers()?;
    let lon = d["XLONG"].numbers()?;
    if ![n, n * nt].contains(&lat.len()) || ![n, n * nt].contains(&lon.len()) {
        return Err(bad("forecast coordinate shape differs from SST"));
    }
    let times = &d["Times"];
    if times.record.dtype != "|S1"
        || times.record.shape.len() != 2
        || times.record.shape[0] != nt
        || times.bytes.len() != times.record.shape.iter().product::<usize>()
    {
        return Err(bad("forecast Times shape/type is invalid"));
    }
    let mut out = Vec::new();
    for t in 0..nt {
        let time = instant(
            std::str::from_utf8(
                &times.bytes[t * times.record.shape[1]..(t + 1) * times.record.shape[1]],
            )?
            .trim_end_matches('\0'),
        )?;
        let coords = |a: &Vec<f64>| {
            if a.len() == n {
                a.clone()
            } else {
                a[t * n..(t + 1) * n].to_vec()
            }
        };
        out.push(Frame {
            time,
            sst: values[t * n..(t + 1) * n]
                .iter()
                .enumerate().map(|(i,v)| if land[if land.len()==n{i}else{t*n+i}]>1.5 {v + offset} else {f64::NAN})
                .collect(),
            lat: coords(&lat),
            lon: coords(&lon),
        });
    }
    Ok(out)
}
fn angle(lat: f64, lon: f64, y: f64, x: f64) -> f64 {
    let dlat = ((lat - y).to_radians() / 2.).sin();
    let dlon = ((lon - x).to_radians() / 2.).sin();
    2. * (dlat * dlat + lat.to_radians().cos() * y.to_radians().cos() * dlon * dlon)
        .clamp(0., 1.)
        .sqrt()
        .asin()
}
#[derive(Default, Serialize)]
struct Score {
    count: usize,
    bias_k: Option<f64>,
    rmse_k: Option<f64>,
    mae_k: Option<f64>,
}
fn score(pairs: &[(f64, f64, f64)]) -> Score {
    let weight: f64 = pairs.iter().map(|p| p.2).sum();
    if pairs.is_empty() || weight <= 0. {
        return Score::default();
    }
    Score {
        count: pairs.len(),
        bias_k: Some(pairs.iter().map(|(m, o, w)| (m - o) * w).sum::<f64>() / weight),
        rmse_k: Some(
            (pairs
                .iter()
                .map(|(m, o, w)| (m - o).powi(2) * w)
                .sum::<f64>()
                / weight)
                .sqrt(),
        ),
        mae_k: Some(pairs.iter().map(|(m, o, w)| (m - o).abs() * w).sum::<f64>() / weight),
    }
}
fn daily(frames: &[Frame], day: NaiveDate, gap: u32) -> Result<Option<Vec<f64>>> {
    let selected: Vec<_> = frames
        .iter()
        .filter(|f| f.time.date_naive() == day)
        .collect();
    if selected.is_empty() {
        return Ok(None);
    }
    let start = day.and_hms_opt(0, 0, 0).unwrap().and_utc();
    if selected[0].time != start
        || (start + chrono::Duration::days(1) - selected.last().unwrap().time).num_seconds()
            > i64::from(gap) * 3600
        || selected
            .windows(2)
            .any(|p| (p[1].time - p[0].time).num_seconds() > i64::from(gap) * 3600)
    {
        return Ok(None);
    }
    // Time-weighted left-interval mean over exactly one UTC day; missing at
    // any contributing time remains missing, rather than changing support.
    let mut mean = vec![0.; frames[0].sst.len()];
    for (k, f) in selected.iter().enumerate() {
        let end = selected
            .get(k + 1)
            .map_or(start + chrono::Duration::days(1), |f| f.time);
        let weight = (end - f.time).num_seconds() as f64 / 86400.;
        for (m, v) in mean.iter_mut().zip(&f.sst) {
            *m += if physical(*v) { v * weight } else { f64::NAN };
        }
    }
    Ok(Some(mean))
}
fn nearest_axis(axis: &[f64], x: f64) -> usize {
    let k = axis.partition_point(|v| *v < x);
    if k == 0 {
        0
    } else if k == axis.len() {
        k - 1
    } else if x - axis[k - 1] <= axis[k] - x {
        k - 1
    } else {
        k
    }
}
fn digest(path: &Path) -> Result<String> {
    let mut file = fs::File::open(path)?;
    let mut sha = Sha256::new();
    let mut buf = vec![0; 1024 * 1024];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        sha.update(&buf[..n]);
    }
    Ok(format!("{:x}", sha.finalize()))
}
fn buoy_reports(path: &Path) -> Result<Vec<(DateTime<Utc>, f64)>> {
    let bytes = fs::read(path)?;
    let mut plain = Vec::new();
    if bytes.starts_with(&[31, 139]) {
        flate2::read::GzDecoder::new(bytes.as_slice()).read_to_end(&mut plain)?;
    } else {
        plain = bytes;
    }
    parse_buoy(std::str::from_utf8(&plain)?)
}
fn parse_buoy(text: &str) -> Result<Vec<(DateTime<Utc>, f64)>> {
    let mut lines = text.lines();
    let header = lines.next().ok_or_else(|| bad("empty NDBC file"))?;
    let names: Vec<_> = header.trim_start_matches('#').split_whitespace().collect();
    if names.get(..5) != Some(&["YY", "MM", "DD", "hh", "mm"][..]) {
        return Err(bad("NDBC needs a four-digit-year standard-met header"));
    }
    let index = names
        .iter()
        .position(|s| *s == "WTMP")
        .ok_or_else(|| bad("NDBC file has no WTMP"))?;
    let units: Vec<_> = lines
        .next()
        .ok_or_else(|| bad("missing NDBC units"))?
        .trim_start_matches('#')
        .split_whitespace()
        .collect();
    if units.get(index) != Some(&"degC") {
        return Err(bad("NDBC WTMP must be degC"));
    }
    let mut out = Vec::new();
    for line in lines.filter(|s| !s.trim().is_empty() && !s.starts_with('#')) {
        let a: Vec<_> = line.split_whitespace().collect();
        if a.len() != names.len() {
            return Err(bad("truncated NDBC row"));
        }
        if a[0].len() != 4 {
            return Err(bad("NDBC year must have four digits"));
        }
        let date = NaiveDate::from_ymd_opt(a[0].parse()?, a[1].parse()?, a[2].parse()?)
            .ok_or_else(|| bad("bad NDBC date"))?;
        let time = date
            .and_hms_opt(a[3].parse()?, a[4].parse()?, 0)
            .ok_or_else(|| bad("bad NDBC clock"))?
            .and_utc();
        if let Ok(v) = a[index].parse::<f64>() {
            let k = v + 273.15;
            if physical(k) {
                out.push((time, k));
            }
        }
    }
    out.sort_by_key(|p| p.0);
    if out.windows(2).any(|p| p[0].0 == p[1].0) {
        return Err(bad("duplicate NDBC report times"));
    }
    Ok(out)
}
fn ocean_mask(path: &Option<PathBuf>) -> Result<Option<Vec<f32>>> {
    path.as_ref().map(|p| {
        let bytes=fs::read(p)?;
        if bytes.len()%4!=0 {return Err(bad("ocean mask must be raw float32"));}
        let values:Vec<f32>=bytes.chunks_exact(4).map(|b|f32::from_le_bytes(b.try_into().unwrap())).collect();
        if values.iter().any(|v|!v.is_finite()||!(*v==0.||*v==1.)) {return Err(bad("ocean mask must contain zero or one"));}
        Ok(values)
    }).transpose()
}

fn run(args: &[String]) -> Result<Value> {
    if args.len() != 3 || args[0] != "score" {
        return Err(bad("rw_ocean_verify score REQUEST.json RW_NETCDF"));
    }
    let begin = std::time::Instant::now();
    let req: Request = serde_json::from_slice(&fs::read(&args[1])?)?;
    if req.forecast.is_empty()
        || req.oisst.is_empty()
        || req.buoys.is_empty()
        || req.jobs == 0
        || req.jobs > 64
        || !(1..=6).contains(&req.daily_max_gap_hours)
        || !req.buoy_max_distance_km.is_finite()
        || !(0.0..=100.0).contains(&req.buoy_max_distance_km)
        || !(1..=60).contains(&req.buoy_match_minutes)
    {
        return Err(bad(
            "invalid/missing SST verification inputs or matching limits",
        ));
    }
    let parent = req
        .output
        .parent()
        .ok_or_else(|| bad("output needs a parent directory"))?;
    fs::create_dir_all(parent)?;
    let work = Work(parent.join(format!(
        ".ocean-verify-{}-{}",
        std::process::id(),
        Utc::now().timestamp_nanos_opt().unwrap()
    )));
    fs::create_dir(&work.0)?;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(req.jobs)
        .build()?;
    let decoded = pool.install(|| {
        req.forecast
            .par_iter()
            .enumerate()
            .map(|(i, p)| forecast(Path::new(&args[2]), p, &work.0.join(format!("f{i}")), &req.sst_variable))
            .collect::<Vec<_>>()
    });
    let mut frames = Vec::new();
    for item in decoded {
        frames.extend(item?);
    }
    if let Some(mask)=ocean_mask(&req.ocean_mask)? {
        for frame in &mut frames {
            if frame.sst.len()!=mask.len(){return Err(bad("ocean mask shape differs from SST"));}
            for (v,m) in frame.sst.iter_mut().zip(&mask){if *m<0.5{*v=f64::NAN;}}
        }
    }
    frames.sort_by_key(|f| f.time);
    if frames.is_empty() || frames.windows(2).any(|p| p[0].time == p[1].time) {
        return Err(bad("empty/duplicate forecast times"));
    }
    let first = &frames[0];
    if first
        .lat
        .iter()
        .zip(&first.lon)
        .any(|(y, x)| !y.is_finite() || !x.is_finite() || y.abs() > 90.)
        || frames
            .iter()
            .any(|f| f.lat != first.lat || f.lon != first.lon || f.sst.len() != first.sst.len())
    {
        return Err(bad("SST verifier requires a fixed finite geographic grid"));
    }
    let mut daily_scores = Vec::new();
    let mut product_days = Vec::new();
    let mut used_days = BTreeSet::new();
    let mut last_day = None;
    let mut baseline: Option<(Vec<f64>, Vec<f64>)> = None;
    for (k, path) in req.oisst.iter().enumerate() {
        let d = dump(
            Path::new(&args[2]),
            path,
            &["sst", "lat", "lon", "time"],
            &work.0.join(format!("o{k}")),
        )?;
        let times = d["time"]
            .record
            .times
            .as_ref()
            .ok_or_else(|| bad("OISST has no CF time"))?;
        if times.len() != 1 {
            return Err(bad("OISST input must contain one daily time"));
        }
        let day = instant(&times[0])?.date_naive();
        if last_day.is_some_and(|d| day < d) {
            return Err(bad("OISST files must be chronological"));
        }
        last_day = Some(day);
        if !used_days.insert(day) {
            return Err(bad("duplicate OISST day"));
        }
        let Some(mean) = daily(&frames, day, req.daily_max_gap_hours)? else {
            daily_scores.push(json!({"date":day.to_string(),"status":"incomplete_model_day"}));
            continue;
        };
        let lat = d["lat"].numbers()?;
        let lon = d["lon"].numbers()?;
        let obs = d["sst"].numbers()?;
        let off = unit_offset(&d["sst"].record.units)?;
        if lat.len() < 2
            || lon.len() < 2
            || obs.len() != lat.len() * lon.len()
            || lat.windows(2).any(|p| p[0] >= p[1])
            || lon.windows(2).any(|p| p[0] >= p[1])
            || lat.iter().chain(&lon).any(|v| !v.is_finite())
        {
            return Err(bad(
                "OISST needs finite increasing coordinate axes and one SST plane",
            ));
        }
        let mut matched = vec![f64::NAN; mean.len()];
        let mut pairs = Vec::new();
        let mut changes = Vec::new();
        for i in 0..mean.len() {
            let y = first.lat[i];
            let mut x = first.lon[i];
            while x < lon[0] {
                x += 360.;
            }
            while x > lon[lon.len() - 1] + 0.5 * (lon[1] - lon[0]) {
                x -= 360.;
            }
            let j = nearest_axis(&lat, y);
            let h = nearest_axis(&lon, x);
            let o = obs[j * lon.len() + h] + off;
            if physical(mean[i])
                && physical(o)
                && angle(y, x, lat[j], lon[h]) <= 0.5_f64.to_radians()
            {
                matched[i] = o;
                let w = y.to_radians().cos();
                pairs.push((mean[i], o, w));
                if let Some((m0, o0)) = &baseline {
                    if physical(m0[i]) && physical(o0[i]) {
                        changes.push((mean[i] - m0[i], o - o0[i], w));
                    }
                }
            }
        }
        if let Some(dir) = &req.products {
            fs::create_dir_all(dir)?;
            let write = |name: &str, values: Vec<f64>| -> Result<PathBuf> {
                let path = dir.join(format!("{day}-{name}.f32"));
                let mut bytes = vec![0u8; values.len()*4];
                bytes.par_chunks_mut(4).zip(values.par_iter()).for_each(|(b,v)|
                    b.copy_from_slice(&(if v.is_finite(){*v as f32}else{0.}).to_le_bytes()));
                fs::write(&path,bytes)?;
                Ok(path)
            };
            let valid: Vec<bool> = mean.iter().zip(&matched).map(|(&m,&o)|physical(m)&&physical(o)).collect();
            let mut fields = BTreeMap::new();
            fields.insert("model-sst",write("model-sst",mean.iter().map(|v|v-273.15).collect())?);
            fields.insert("observed-sst",write("observed-sst",matched.iter().map(|v|v-273.15).collect())?);
            if let Some((m0,o0)) = &baseline {
                fields.insert("model-change",write("model-change",mean.iter().zip(m0).map(|(a,b)|a-b).collect())?);
                fields.insert("observed-change",write("observed-change",matched.iter().zip(o0).map(|(a,b)|a-b).collect())?);
                fields.insert("change-wet",write("change-wet",valid.iter().zip(m0.iter().zip(o0)).map(|(v,(m,o))|if *v&&physical(*m)&&physical(*o){1.}else{0.}).collect())?);
            }
            product_days.push(json!({"date":day.to_string(),"latitude":write("latitude",first.lat.clone())?,
                "longitude":write("longitude",first.lon.clone())?,"wet":write("wet",valid.iter().map(|v|if *v{1.}else{0.}).collect())?,"fields":fields}));
        }
        if baseline.is_none() {
            baseline = Some((mean, matched));
        }
        daily_scores.push(json!({"date":day.to_string(),"status":if pairs.is_empty(){"no_matches"}else{"scored"},"sst":score(&pairs),"change_from_first_complete_day":score(&changes)}));
    }
    let mut buoy_scores = Vec::new();
    for b in &req.buoys {
        if b.id.is_empty()
            || !b.latitude.is_finite()
            || b.latitude.abs() > 90.
            || !b.longitude.is_finite()
        {
            return Err(bad("invalid buoy position/id"));
        }
        let nearest = (0..first.sst.len())
            .filter(|i| frames.iter().all(|f| physical(f.sst[*i])))
            .map(|i| {
                (
                    i,
                    angle(b.latitude, b.longitude, first.lat[i], first.lon[i]) * 6371.,
                )
            })
            .min_by(|a, b| a.1.total_cmp(&b.1));
        let reports = buoy_reports(&b.path)?;
        let mut pairs = Vec::new();
        let mut changes = Vec::new();
        let mut matched = Vec::new();
        let mut base = None;
        let mut used = BTreeSet::new();
        if let Some((i, distance)) = nearest.filter(|p| p.1 <= req.buoy_max_distance_km) {
            for f in &frames {
                let k = reports.partition_point(|p| p.0 < f.time);
                let picked = [k.checked_sub(1), Some(k)]
                    .into_iter()
                    .flatten()
                    .filter_map(|k| reports.get(k).map(|v| (k, v)))
                    .min_by_key(|(_, p)| (p.0 - f.time).num_seconds().abs());
                if let Some((k, (t, o))) = picked {
                    if (*t - f.time).num_seconds().abs() <= req.buoy_match_minutes * 60
                        && used.insert(k)
                    {
                        let m = f.sst[i];
                        pairs.push((m, *o, 1.));
                        let (m0, o0) = *base.get_or_insert((m, *o));
                        changes.push((m - m0, *o - o0, 1.));
                        matched.push(json!({"forecast_time":f.time.to_rfc3339(),"observation_time":t.to_rfc3339(),"model_k":m,"observed_k":o}));
                    }
                }
            }
            buoy_scores.push(json!({"id":b.id,"distance_km":distance,"grid_index":i,"sst":score(&pairs),"change_from_first_pair":score(&changes),"pairs":matched}));
        } else {
            buoy_scores.push(json!({"id":b.id,"status":"no_wet_grid_within_radius"}));
        }
    }
    let sources = req
        .forecast
        .iter()
        .chain(&req.oisst)
        .chain(req.buoys.iter().map(|b| &b.path))
        .chain(req.ocean_mask.iter())
        .collect::<Vec<_>>();
    let refs = pool.install(|| {
        sources
            .par_iter()
            .map(|p| Ok(json!({"path":p,"sha256":digest(p)?})))
            .collect::<Vec<Result<Value>>>()
    });
    let mut references = Vec::new();
    for r in refs {
        references.push(r?);
    }
    let record = json!({"schema":"gpuwm-ocean-sst-verification-v1","output":req.output,"forecast_frames":frames.len(),"daily":daily_scores,"buoys":buoy_scores,"products":product_days,"sources":references,"wall_seconds":begin.elapsed().as_secs_f64(),"worker_threads":req.jobs,
        "limitations":["Daily time-weighted left-interval model means matched to nearest OISST cell, cosine-latitude weighted on model support; no area-conservative regrid.","Model top-layer SST and OISST foundation SST/buoy sensor depths differ; representativeness error is uncorrected.","OISST includes buoy SST, so these comparisons are not independent.","NDBC historical archive QC and missing-value masks are used; no extra gross-error or sea-ice filter."]});
    let tmp = work.0.join("receipt.json");
    fs::write(&tmp, serde_json::to_vec_pretty(&record)?)?;
    fs::rename(tmp, &req.output)?;
    Ok(record)
}
#[derive(Deserialize)]
struct DiagnosticRequest {
    #[serde(default = "surface_height_limit")] maximum_ocean_surface_height_m: f64,
    forecast: Vec<PathBuf>,
    output: PathBuf,
    products: PathBuf,
    fields: Vec<String>,
    #[serde(default)] ocean_mask: Option<PathBuf>,
    stations: Vec<Station>,
    #[serde(default = "jobs")] jobs: usize,
}
#[derive(Deserialize)]
struct Station { id: String, latitude: f64, longitude: f64 }
fn surface_height_limit() -> f64 {10.}
fn diagnostics(args: &[String]) -> Result<Value> {
    if args.len()!=2 {return Err(bad("diagnostics REQUEST.json RW_NETCDF"));}
    let req: DiagnosticRequest=serde_json::from_slice(&fs::read(&args[0])?)?;
    let bridge=Path::new(&args[1]);
    let mask=ocean_mask(&req.ocean_mask)?;
    if !req.maximum_ocean_surface_height_m.is_finite() || req.maximum_ocean_surface_height_m<0. {return Err(bad("surface-height limit must be finite and nonnegative"));}
    fs::create_dir_all(&req.products)?;
    let begin=std::time::Instant::now();
    let pool=rayon::ThreadPoolBuilder::new().num_threads(req.jobs).build()?;
    let rows: Vec<Result<Vec<Value>>>=pool.install(||req.forecast.par_iter().enumerate().map(|(index,path)| {
        let mut names: Vec<&str>=req.fields.iter().map(String::as_str).collect();
        for name in ["PSFC","U10","V10","XLAND","XLAT","XLONG","Times","PH","PHB"] {if !names.contains(&name){names.push(name);}}
        let d=dump(bridge,path,&names,&req.products.join(format!("decode-{index}")))?;
        let shape=&d["PSFC"].record.shape;
        if shape.len()!=3 {return Err(bad("PSFC must have time,y,x dimensions"));}
        let (nt,ny,nx)=(shape[0],shape[1],shape[2]);let n=ny*nx;
        if mask.as_ref().is_some_and(|m|m.len()!=n){return Err(bad("ocean mask shape differs from diagnostics"));}
        let mut numbers=BTreeMap::new();
        for name in names.iter().filter(|&&x|x!="Times") {
            let mut values=d[*name].numbers()?;
            if *name=="PH" || *name=="PHB" {
                let shape=&d[*name].record.shape;
                if shape.len()!=4 || shape[0]!=nt || shape[1]==0 || shape[2]!=ny || shape[3]!=nx {return Err(bad("geopotential shape differs from surface"));}
                values=values.chunks_exact(shape[1]*n).flat_map(|v|v[..n].iter().copied()).collect();
            }
            if values.len()!=n&&values.len()!=n*nt {return Err(bad(format!("{name} is not a surface field")));}
            numbers.insert(*name,values);
        }
        let times=&d["Times"];let width=times.record.shape[1];
        let mut frames=Vec::new();
        for t in 0..nt {
            let time=instant(std::str::from_utf8(&times.bytes[t*width..(t+1)*width])?.trim_end_matches('\0'))?;
            let plane=|name:&str|->&[f64]{let a=&numbers[name];if a.len()==n{a}else{&a[t*n..(t+1)*n]}};
            let lat=plane("XLAT");let lon=plane("XLONG");let land=plane("XLAND");
            let p=plane("PSFC");let u=plane("U10");let v=plane("V10");
            let sea:Vec<usize>=(0..n).filter(|&i|land[i]>1.5&&mask.as_ref().is_none_or(|m|m[i]>0.5) ).collect();
            if sea.iter().any(|&i|!p[i].is_finite()||p[i]<=10000.) {return Err(bad("nonfinite or invalid ocean surface pressure"));}
            let ph=plane("PH");let phb=plane("PHB");
            let center=sea.iter().copied().filter(|&i|((ph[i]+phb[i])/9.81).abs()<=req.maximum_ocean_surface_height_m).min_by(|&a,&b|p[a].total_cmp(&p[b])).ok_or_else(||bad("no finite ocean pressure"))?;
            let wind=sea.iter().filter(|&&i|angle(lat[center],lon[center],lat[i],lon[i])*6371.<=300.)
                .map(|&i|u[i].hypot(v[i])).filter(|v|v.is_finite()).fold(0.,f64::max);
            let mut stations=Vec::new();
            for s in &req.stations {
                let i=sea.iter().copied().min_by(|&a,&b|angle(s.latitude,s.longitude,lat[a],lon[a]).total_cmp(&angle(s.latitude,s.longitude,lat[b],lon[b]))).unwrap();
                let mut values=BTreeMap::new();
                for name in &req.fields {values.insert(name.clone(),plane(name)[i]);}
                stations.push(json!({"id":s.id,"distance_km":angle(s.latitude,s.longitude,lat[i],lon[i])*6371.,"latitude":lat[i],"longitude":lon[i],"values":values}));
            }
            let write=|name:&str,values:&[f64]|->Result<PathBuf>{
                let path=req.products.join(format!("{}-{name}.f32",time.format("%Y%m%dT%H%M%S")));
                let bytes:Vec<u8>=values.iter().flat_map(|&x|(x as f32).to_le_bytes()).collect();fs::write(&path,bytes)?;Ok(path)
            };
            let mut fields=BTreeMap::new();
            let mut field_health=BTreeMap::new();
            for name in &req.fields {
                let values=plane(name);
                let nonfinite=sea.iter().filter(|&&i|!values[i].is_finite()).count();
                // The published OCEAN_SST diagnostic deliberately fills inactive
                // PWP land/lake cells. Retain the missing-location fingerprint
                // so evidence can distinguish fixed support from new failures.
                if nonfinite>0 && name!="OCEAN_SST" {return Err(bad(format!("{name} has {nonfinite} nonfinite ocean values at {time}")));}
                let missing:Vec<u8>=sea.iter().map(|&i|u8::from(!values[i].is_finite())).collect();
                field_health.insert(name.clone(),json!({"ocean_cells":sea.len(),"nonfinite":nonfinite,"missing_location_sha256":format!("{:x}",Sha256::digest(&missing)),
                    "min":sea.iter().map(|&i|values[i]).filter(|v|v.is_finite()).min_by(f64::total_cmp),
                    "max":sea.iter().map(|&i|values[i]).filter(|v|v.is_finite()).max_by(f64::total_cmp)}));
                fields.insert(name.clone(),write(name,values)?);
            }
            let wet:Vec<f64>=land.iter().enumerate().map(|(i,&x)|if x>1.5&&mask.as_ref().is_none_or(|m|m[i]>0.5){1.}else{0.}).collect();
            frames.push(json!({"time":time.to_rfc3339(),"nx":nx,"ny":ny,"latitude":write("latitude",lat)?,"longitude":write("longitude",lon)?,"wet":write("wet",&wet)?,"fields":fields,"field_health":field_health,"stations":stations,"ocean_min_pressure_hpa":p[center]/100.,"center_latitude":lat[center],"center_longitude":lon[center],"max_ocean_wind_300km_ms":wind}));
        }
        Ok(frames)
    }).collect());
    let mut frames=Vec::new();for row in rows{frames.extend(row?);}frames.sort_by_key(|r|r["time"].as_str().unwrap().to_owned());
    let result=json!({"schema":"gpuwm-ocean-diagnostics-v1","frames":frames,"worker_threads":req.jobs,"wall_seconds":begin.elapsed().as_secs_f64(),"maximum_ocean_surface_height_m":req.maximum_ocean_surface_height_m,"pressure_method":"Minimum surface pressure over ocean cells within the configured surface-height limit of sea level; maximum ocean wind within 300 km of that minimum. Center over land is not represented."});
    fs::write(&req.output,serde_json::to_vec_pretty(&result)?)?;Ok(result)
}

fn main() {
    std::hint::black_box(GPUWM_BRIDGE_SOURCE_REV_STAMP);
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("--abi") => {
            println!("{ABI}");
            return;
        }
        Some("--version") => {
            println!("rw_ocean_verify {}", env!("CARGO_PKG_VERSION"));
            return;
        }
        _ => {}
    }
    let result = if args.first().map(String::as_str) == Some("ndbc-inventory") {
        buoy_inventory(&args[1..])
    } else if args.first().map(String::as_str)==Some("diagnostics") {
        diagnostics(&args[1..])
    } else {
        run(&args)
    };
    match result {
        Ok(v) => println!("{v}"),
        Err(e) => {
            eprintln!("rw_ocean_verify: {e}");
            std::process::exit(2);
        }
    }
}

/// Admit entire real archives before a costly forecast. This does not score
/// a model: counts and water-temperature ranges are source evidence only.
fn buoy_inventory(args: &[String]) -> Result<Value> {
    if args.len() < 3 {
        return Err(bad("rw_ocean_verify ndbc-inventory START END FILE..."));
    }
    let begin = std::time::Instant::now();
    let start = instant(&args[0])?;
    let end = instant(&args[1])?;
    if end < start {
        return Err(bad("inventory END precedes START"));
    }
    let jobs = (args.len() - 2).min(8);
    let pool = rayon::ThreadPoolBuilder::new().num_threads(jobs).build()?;
    let results = pool.install(|| {
        args[2..].par_iter().map(|p| -> Result<Value> {
        let path = Path::new(p);
        let reports = buoy_reports(path)?;
        let window: Vec<_> = reports.iter().filter(|r| r.0>=start && r.0<=end).collect();
        Ok(json!({"path":path,"sha256":digest(path)?,"valid_wtmp_archive_reports":reports.len(),
            "window_reports":window.len(),"window_first":window.first().map(|r|r.0.to_rfc3339()),
            "window_last":window.last().map(|r|r.0.to_rfc3339()),
            "window_min_k":window.iter().map(|r|r.1).min_by(f64::total_cmp),
            "window_max_k":window.iter().map(|r|r.1).max_by(f64::total_cmp)}))
    }).collect::<Vec<_>>()
    });
    let mut records = Vec::new();
    for record in results {
        records.push(record?);
    }
    Ok(
        json!({"schema":"gpuwm-ocean-buoy-inventory-v1","start":start.to_rfc3339(),
        "end":end.to_rfc3339(),"sources":records,"worker_threads":jobs,
        "wall_seconds":begin.elapsed().as_secs_f64(),"model_scores":null}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ndbc_uses_water_temperature_masks_missing_and_rejects_bad_units() {
        let text = "#YY MM DD hh mm ATMP WTMP\n#yr mo dy hr mn degC degC\n2022 09 26 00 00 17.0 29.0\n2022 09 26 00 10 18.0 99.0\n2022 09 26 00 20 18.0 MM\n";
        let reports = parse_buoy(text).unwrap();
        assert_eq!(reports.len(), 1);
        assert!((reports[0].1 - 302.15).abs() < 1e-10);
        assert!(parse_buoy(&text.replacen("degC degC", "degC K", 1)).is_err());
    }
    #[test]
    fn daily_integrates_irregular_intervals_and_preserves_missing_support() {
        let date = NaiveDate::from_ymd_opt(2022, 9, 26).unwrap();
        let frames: Vec<_> = (0..24)
            .step_by(3)
            .map(|h| Frame {
                time: date.and_hms_opt(h, 0, 0).unwrap().and_utc(),
                sst: vec![300. + h as f64, f64::NAN],
                lat: vec![0., 0.],
                lon: vec![0., 0.],
            })
            .collect();
        let mean = daily(&frames, date, 3).unwrap().unwrap();
        assert_eq!(mean[0], 310.5);
        assert!(mean[1].is_nan());
        assert!(daily(&frames[1..], date, 3).unwrap().is_none());
        let mut gap = frames.clone();
        gap.remove(2);
        assert!(daily(&gap, date, 3).unwrap().is_none());
    }
    #[test]
    fn spherical_distance_crosses_dateline_and_paired_scores_use_cooling_sign() {
        assert!((angle(0., 179.9, 0., -179.9).to_degrees() - 0.2).abs() < 1e-10);
        let s = score(&[(-2., -1., 1.), (-3., -1., 1.)]);
        assert_eq!(s.bias_k, Some(-1.5));
        assert_eq!(s.rmse_k, Some(2.5_f64.sqrt()));
    }
}
