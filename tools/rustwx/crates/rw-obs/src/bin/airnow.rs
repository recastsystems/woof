//! AirNow hourly front door. Format: https://docs.airnowapi.org/docs/HourlyDataFactSheet.pdf
//! Published HourlyData dates and times are GMT, marking sample BEGINNING.
//! GMTOffset is GMT to local standard time, not an instruction to shift GMT.
//! --time-basis local explicitly supports supplied local-standard-time files.
use chrono::{DateTime, Duration, NaiveDateTime, Utc};
use rw_nexrad::s3::{parse_http_date, parse_time};
use rw_obs::{
    err, hex_sha256,
    table::{
        AIRNOW_VARIABLES, MEAS_AIRNOW_HOURLY, RowProvenance, TABLE_SCHEMA, TableRow, TableWriter,
    },
};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    path::{Path, PathBuf},
    process::ExitCode,
};
pub static GPUWM_BRIDGE_SOURCE_REV_STAMP: &str =
    concat!("GPUWM_BRIDGE_SOURCE_REV=", env!("GPUWM_BRIDGE_SOURCE_REV"));
const ABI: &str = "gpuwm-obs.airnow-fetch.v1\tgpuwm-obs.airnow-table.v1\tgpuwm-obs.table.v2";
const ARCHIVE: &str = "https://files.airnowtech.org/airnow";
#[derive(Clone, Debug, Serialize, Deserialize)]
struct FileRecord {
    name: String,
    url: String,
    sha256: String,
    bytes: usize,
    last_modified: Option<String>,
    fetched_at: String,
}
#[derive(Serialize, Deserialize)]
struct FetchRecord {
    schema: String,
    files: Vec<FileRecord>,
}
#[derive(Clone, Debug, Serialize)]
struct Site {
    station_id: String,
    latitude: f64,
    longitude: f64,
    elevation_m: f64,
}
fn stamp(t: DateTime<Utc>) -> String {
    t.format("%Y-%m-%dT%H:%M:%SZ").to_string()
}
fn options(args: &[String]) -> Result<BTreeMap<String, String>, Box<dyn Error>> {
    let mut o = BTreeMap::new();
    if args.len() % 2 != 0 {
        return Err(err("options require values"));
    }
    for pair in args.chunks(2) {
        if ![
            "--start",
            "--end",
            "--out",
            "--dir",
            "--bbox",
            "--fetch-record",
            "--time-basis",
            "--archive",
            "--request-pause-ms",
        ]
        .contains(&pair[0].as_str())
            || o.insert(pair[0].clone(), pair[1].clone()).is_some()
        {
            return Err(err(format!("unknown or repeated option {}", pair[0])));
        }
    }
    Ok(o)
}
fn required<'a>(o: &'a BTreeMap<String, String>, k: &str) -> Result<&'a str, Box<dyn Error>> {
    o.get(k)
        .map(String::as_str)
        .ok_or_else(|| err(format!("{k} required")))
}
fn hours(start: DateTime<Utc>, end: DateTime<Utc>) -> Result<Vec<DateTime<Utc>>, Box<dyn Error>> {
    if end < start || end - start > Duration::days(31) {
        return Err(err("window must be ordered and at most 31 days"));
    }
    let mut t = DateTime::from_timestamp(start.timestamp().div_euclid(3600) * 3600, 0).unwrap();
    let mut out = Vec::new();
    while t <= end {
        out.push(t);
        t += Duration::hours(1);
    }
    Ok(out)
}
fn fetch(o: &BTreeMap<String, String>, hs: &[DateTime<Utc>]) -> Result<String, Box<dyn Error>> {
    let dir = Path::new(required(o, "--out")?);
    std::fs::create_dir_all(dir)?;
    let record_path = dir.join("fetch.json");
    let mut record = if record_path.exists() {
        serde_json::from_slice::<FetchRecord>(&std::fs::read(&record_path)?)?
    } else {
        FetchRecord {
            schema: "gpuwm-obs.airnow-fetch.v1".into(),
            files: vec![],
        }
    };
    if record.schema != "gpuwm-obs.airnow-fetch.v1" {
        return Err(err("wrong fetch schema"));
    }
    let archive = o
        .get("--archive")
        .map(String::as_str)
        .unwrap_or(ARCHIVE)
        .trim_end_matches('/');
    let mut names = BTreeSet::new();
    for t in hs {
        let day = t.format("%Y/%Y%m%d");
        names.insert(format!("{day}/HourlyData_{}.dat", t.format("%Y%m%d%H")));
        names.insert(format!("{day}/Monitoring_Site_Locations_V2.dat"));
    }
    let pause = o
        .get("--request-pause-ms")
        .map(|v| v.parse::<u64>())
        .transpose()?
        .unwrap_or(100);
    let agent = rw_obs::net::agent();
    for name in names {
        let url = format!("{archive}/{name}");
        let path = dir.join(&name);
        if record.files.iter().any(|f| {
            f.name == name
                && f.url == url
                && std::fs::read(&path)
                    .map(|b| hex_sha256(&b) == f.sha256)
                    .unwrap_or(false)
        }) {
            continue;
        }
        std::thread::sleep(std::time::Duration::from_millis(pause));
        let mut response = rw_http::get(&agent, &url)
            .map_err(|e| err(format!("GET {url}: {e}")))?;
        let modified = response
            .headers()
            .get("last-modified")
            .and_then(|h| h.to_str().ok())
            .map(str::to_string);
        let bytes = response
            .body_mut()
            .with_config()
            .limit(rw_obs::net::MAX_RESPONSE_BYTES)
            .read_to_vec()?;
        if bytes.is_empty() {
            return Err(err(format!("empty response {url}")));
        }
        std::fs::create_dir_all(path.parent().unwrap())?;
        let tmp = path.with_extension("partial");
        std::fs::write(&tmp, &bytes)?;
        std::fs::rename(tmp, &path)?;
        record.files.retain(|f| f.name != name);
        record.files.push(FileRecord {
            name,
            url,
            sha256: hex_sha256(&bytes),
            bytes: bytes.len(),
            last_modified: modified,
            fetched_at: stamp(Utc::now()),
        });
        std::fs::write(&record_path, serde_json::to_vec_pretty(&record)?)?;
    }
    Ok(serde_json::to_string_pretty(&record)?)
}
fn sites(text: &str) -> Result<BTreeMap<String, Site>, Box<dyn Error>> {
    let mut lines = text.lines();
    let header: Vec<_> = lines
        .next()
        .ok_or_else(|| err("empty site list"))?
        .split('|')
        .collect();
    let col = |name: &str| {
        header
            .iter()
            .position(|h| *h == name)
            .ok_or_else(|| err(format!("site header missing {name}")))
    };
    let (sid, aqs, full, lat, lon, elev) = (
        col("StationID")?,
        col("AQSID")?,
        col("FullAQSID")?,
        col("Latitude")?,
        col("Longitude")?,
        col("Elevation")?,
    );
    let mut out = BTreeMap::new();
    for line in lines {
        let f: Vec<_> = line.split('|').collect();
        if f.len() != header.len() {
            return Err(err("malformed site record"));
        }
        let latitude: f64 = f[lat].parse()?;
        let longitude: f64 = f[lon].parse()?;
        // A surface table requires measured elevation. Unknown elevations are not zero.
        let Ok(elevation_m) = f[elev].parse::<f64>() else {
            continue;
        };
        if !latitude.is_finite()
            || latitude.abs() > 90.0
            || !longitude.is_finite()
            || longitude.abs() > 180.0
            || !elevation_m.is_finite()
        {
            return Err(err("invalid site coordinates"));
        }
        let site = Site {
            station_id: f[sid].into(),
            latitude,
            longitude: rw_obs::seam::wrap_longitude(longitude),
            elevation_m,
        };
        for id in [f[sid], f[full]] {
            out.insert(id.to_string(), site.clone());
        }
        // Legacy AQSID can collide across countries. Never resolve ambiguous aliases.
        if let Some(old) = out.get(f[aqs]) {
            if old.station_id != site.station_id {
                out.insert(
                    f[aqs].into(),
                    Site {
                        station_id: String::new(),
                        ..site.clone()
                    },
                );
            }
        } else {
            out.insert(f[aqs].into(), site);
        }
    }
    Ok(out)
}
fn valid_time(
    date: &str,
    time: &str,
    offset: &str,
    local: bool,
) -> Result<DateTime<Utc>, Box<dyn Error>> {
    let t = NaiveDateTime::parse_from_str(&format!("{date} {time}"), "%m/%d/%y %H:%M")?.and_utc();
    let hours: f64 = offset.parse()?;
    if !hours.is_finite() || !(-12.0..=14.0).contains(&hours) {
        return Err(err("invalid GMT offset"));
    }
    Ok(if local {
        t - Duration::seconds((hours * 3600.0).round() as i64)
    } else {
        t
    })
}
fn decode(
    line: &str,
    local: bool,
    ss: &BTreeMap<String, Site>,
    prov: RowProvenance,
) -> Result<TableRow, String> {
    let f: Vec<_> = line.trim_end().split('|').collect();
    if f.len() != 9 {
        return Err("malformed_or_flagged_record".into());
    }
    let spec = AIRNOW_VARIABLES
        .iter()
        .find(|r| r.0 == f[5])
        .ok_or("unsupported_parameter")?;
    if f[6].to_ascii_uppercase() != spec.2 {
        return Err("unexpected_units".into());
    }
    let value: f64 = f[7].parse().map_err(|_| "invalid_value")?;
    if !value.is_finite() {
        return Err("invalid_value".into());
    }
    if value < 0.0 {
        return Err("negative_sentinel".into());
    }
    if value < spec.3 || value > spec.4 {
        return Err("gross_bound".into());
    }
    let site = ss
        .get(f[2])
        .filter(|s| !s.station_id.is_empty())
        .ok_or("unknown_or_ambiguous_site")?;
    let valid_time = valid_time(f[0], f[1], f[4], local).map_err(|_| "invalid_time")?;
    Ok(TableRow {
        source: "airnow".into(),
        station_id: site.station_id.clone(),
        latitude_deg: site.latitude,
        longitude_deg: site.longitude,
        elevation_m: site.elevation_m,
        level_pa: None,
        valid_time,
        variable: spec.1.into(),
        value,
        error: spec.5,
        provenance: RowProvenance {
            measurement: MEAS_AIRNOW_HOURLY,
            ..prov
        },
    })
}
fn table(
    o: &BTreeMap<String, String>,
    hs: &[DateTime<Utc>],
    start: DateTime<Utc>,
    end: DateTime<Utc>,
) -> Result<String, Box<dyn Error>> {
    let dir = Path::new(required(o, "--dir")?);
    let output = PathBuf::from(required(o, "--out")?);
    let local = match o.get("--time-basis").map(String::as_str).unwrap_or("utc") {
        "utc" => false,
        "local" => true,
        _ => return Err(err("--time-basis must be utc or local")),
    };
    let bbox: Option<Vec<f64>> = o
        .get("--bbox")
        .map(|v| {
            v.split(',')
                .map(str::parse)
                .collect::<Result<Vec<f64>, _>>()
        })
        .transpose()?;
    if let Some(b) = &bbox {
        if b.len() != 4
            || b.iter().any(|x| !x.is_finite())
            || b[0].abs() > 180.0
            || b[2].abs() > 180.0
            || b[1] < -90.0
            || b[3] > 90.0
            || b[1] > b[3]
        {
            return Err(err("--bbox expects W,S,E,N on signed globe"));
        }
    }
    let fetch = o
        .get("--fetch-record")
        .map(|p| -> Result<FetchRecord, Box<dyn Error>> {
            Ok(serde_json::from_slice(&std::fs::read(p)?)?)
        })
        .transpose()?;
    if fetch
        .as_ref()
        .is_some_and(|r| r.schema != "gpuwm-obs.airnow-fetch.v1")
    {
        return Err(err("wrong fetch record schema"));
    }
    let mut writer = TableWriter::new();
    let mut counts = BTreeMap::new();
    let mut variables = BTreeMap::new();
    let mut seen = BTreeSet::new();
    let mut inputs = vec![];
    for t in hs {
        let day = t.format("%Y/%Y%m%d").to_string();
        let site_name = format!("{day}/Monitoring_Site_Locations_V2.dat");
        let ss = sites(&std::fs::read_to_string(dir.join(&site_name))?)?;
        let name = format!("{day}/HourlyData_{}.dat", t.format("%Y%m%d%H"));
        let bytes = std::fs::read(dir.join(&name))?;
        let digest = hex_sha256(&bytes);
        let file = fetch
            .as_ref()
            .and_then(|r| r.files.iter().find(|f| f.name == name));
        if fetch.is_some() && file.is_none() {
            return Err(err(format!("fetch record missing {name}")));
        }
        if let Some(f) = file {
            if f.sha256 != digest {
                return Err(err(format!("digest mismatch {name}")));
            }
        }
        let site_digest = hex_sha256(&std::fs::read(dir.join(&site_name))?);
        if let Some(r) = &fetch {
            let f = r
                .files
                .iter()
                .find(|f| f.name == site_name)
                .ok_or_else(|| err("fetch record missing site list"))?;
            if f.sha256 != site_digest {
                return Err(err("site digest mismatch"));
            }
        }
        inputs.push(serde_json::json!({"name":name,"sha256":digest,"site_name":site_name,"site_sha256":site_digest}));
        let prov = RowProvenance::of_source(
            &digest,
            file.and_then(|f| f.last_modified.as_deref())
                .and_then(parse_http_date),
            file.map(|f| parse_time(&f.fetched_at)).transpose()?,
        );
        let text = match std::str::from_utf8(&bytes) {
            Ok(s) => s.to_string(),
            Err(_) => bytes.iter().map(|b| char::from(*b)).collect(),
        };
        for line in text.lines().filter(|l| !l.trim().is_empty()) {
            match decode(line, local, &ss, prov.clone()) {
                Ok(row) => {
                    let reason = if row.valid_time < start || row.valid_time > end {
                        Some("outside_window")
                    } else if bbox.as_ref().is_some_and(|b| {
                        row.latitude_deg < b[1]
                            || row.latitude_deg > b[3]
                            || !(if b[0] <= b[2] {
                                row.longitude_deg >= b[0] && row.longitude_deg <= b[2]
                            } else {
                                row.longitude_deg >= b[0] || row.longitude_deg <= b[2]
                            })
                    }) {
                        Some("outside_bbox")
                    } else {
                        None
                    };
                    if let Some(r) = reason {
                        *counts.entry(r.to_string()).or_insert(0usize) += 1;
                        continue;
                    }
                    if !seen.insert((row.station_id.clone(), row.valid_time, row.variable.clone()))
                    {
                        return Err(err("duplicate station/time/variable"));
                    }
                    writer.push(row, &mut variables);
                }
                Err(reason) => {
                    *counts.entry(reason).or_insert(0usize) += 1;
                }
            }
        }
    }
    let mut stations = BTreeMap::new();
    let mut reports: BTreeMap<(String, String), BTreeMap<String, f64>> = BTreeMap::new();
    for row in writer.rows() {
        stations.insert(
            row.station_id.clone(),
            Site {
                station_id: row.station_id.clone(),
                latitude: row.latitude_deg,
                longitude: row.longitude_deg,
                elevation_m: row.elevation_m,
            },
        );
        reports
            .entry((
                row.station_id.clone(),
                row.valid_time.format("%Y-%m-%dT%H:%M:%S").to_string(),
            ))
            .or_default()
            .insert(row.variable.clone(), row.value);
    }
    let (rows, digest, bytes) = writer.write(&output)?;
    let reports:Vec<_>=reports.into_iter().map(|((station_id,valid_time),values)|serde_json::json!({"station_id":station_id,"valid_time":valid_time,"values":values})).collect();
    let record = serde_json::json!({"schema":"gpuwm-obs.airnow-table.v1","table_schema":TABLE_SCHEMA,"rows":rows,"bytes":bytes,"sha256":digest,"counts":counts,"rows_by_variable":variables,"time_basis":if local {"local_standard"} else {"utc"},"latency_verified":fetch.is_some(),"inputs":inputs,"stations":stations.into_values().collect::<Vec<_>>(),"reports":reports,"created_at":Utc::now().format("%Y-%m-%dT%H:%M:%S").to_string()});
    let text = serde_json::to_string_pretty(&record)?;
    std::fs::write(output.with_extension("json"), &text)?;
    Ok(text)
}
fn run(a: &[String]) -> Result<String, Box<dyn Error>> {
    match a.first().map(String::as_str) {Some("--abi")=>return Ok(ABI.into()),Some("--version")=>return Ok(format!("rw_airnow {}",env!("CARGO_PKG_VERSION"))),Some("--help")|None=>return Ok("rw_airnow fetch --start TIME --end TIME --out DIR\nrw_airnow table --dir DIR --start TIME --end TIME --out FILE.csv [--bbox W,S,E,N] [--fetch-record FILE] [--time-basis utc|local]".into()),_=>{}}
    // The subcommand is judged before its options, as the sibling front
    // doors do: a mistyped one is named, not reported as a missing --start.
    if !matches!(a[0].as_str(), "fetch" | "table") {
        return Err(err(format!("unknown subcommand {:?}", a[0])));
    }
    let o = options(&a[1..])?;
    let start = parse_time(required(&o, "--start")?)?;
    let end = parse_time(required(&o, "--end")?)?;
    let hs = hours(start, end)?;
    match a[0].as_str() {
        "fetch" => fetch(&o, &hs),
        "table" => table(&o, &hs, start, end),
        _ => Err(err("unknown subcommand")),
    }
}
fn main() -> ExitCode {
    let _ = std::hint::black_box(GPUWM_BRIDGE_SOURCE_REV_STAMP);
    rw_obs::cli::main("rw_airnow", |args| run(args).map(|output| format!("{output}\n")))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn utc_and_local() {
        assert_eq!(
            stamp(valid_time("07/30/25", "21:00", "-4", false).unwrap()),
            "2025-07-30T21:00:00Z"
        );
        assert_eq!(
            stamp(valid_time("07/30/25", "21:00", "-4", true).unwrap()),
            "2025-07-31T01:00:00Z"
        );
        assert_eq!(
            stamp(valid_time("01/01/25", "00:00", "5.5", true).unwrap()),
            "2024-12-31T18:30:00Z"
        );
    }
    #[test]
    fn real_fixture_and_rejections() {
        let ss = sites(include_str!("../../tests/data/airnow/sites.dat")).unwrap();
        let line = include_str!("../../tests/data/airnow/hourly.dat")
            .lines()
            .next()
            .unwrap();
        let row = decode(line, false, &ss, RowProvenance::default()).unwrap();
        assert_eq!(row.variable, "ozone_mole_fraction");
        assert_eq!(row.value, 30.0);
        let found: BTreeSet<_> = include_str!("../../tests/data/airnow/hourly.dat")
            .lines()
            .filter_map(|l| decode(l, false, &ss, RowProvenance::default()).ok())
            .map(|r| r.variable)
            .collect();
        assert_eq!(
            found.len(),
            6,
            "all six published parameter/unit pairs are decoded"
        );

        assert_eq!(
            decode(
                &line.replace("|30|", "|-999|"),
                false,
                &ss,
                RowProvenance::default()
            )
            .unwrap_err(),
            "negative_sentinel"
        );
        assert_eq!(
            decode(
                &line.replace("|30|", "|NaN|"),
                false,
                &ss,
                RowProvenance::default()
            )
            .unwrap_err(),
            "invalid_value"
        );
        assert_eq!(
            decode(
                &format!("{line}|INVALID"),
                false,
                &ss,
                RowProvenance::default()
            )
            .unwrap_err(),
            "malformed_or_flagged_record"
        );
        assert_eq!(
            decode(
                &line.replace("|PPB|", "|PPM|"),
                false,
                &ss,
                RowProvenance::default()
            )
            .unwrap_err(),
            "unexpected_units"
        );
    }
    #[test]
    fn window_validation() {
        let t = parse_time("2025-07-30T21:00:00Z").unwrap();
        assert_eq!(hours(t, t).unwrap().len(), 1);
        assert!(hours(t, t - Duration::hours(1)).is_err());
    }
}
