//! `rw_madis` -- the MADIS public-archive front door: aircraft, radiosondes
//! and mesonets.
//!
//! NOAA's Meteorological Assimilation Data Ingest System publishes its
//! public archive as one gzip-wrapped classic NetCDF file per hour and
//! platform under
//! `https://madis-data.ncep.noaa.gov/madisPublic1/data/archive/YYYY/MM/DD/`:
//! `point/acars/netcdf/` (commercial aircraft: AMDAR, ACARS, TAMDAR, about
//! 46,000 reports an hour), `point/raob/netcdf/` (radiosondes; the hours
//! nobody launched are simply absent) and `LDAD/mesonet/netCDF/` (the public
//! part of the local mesonet collection, five-minute reports from tens of
//! thousands of stations).  This is the data stream an operational regional
//! analysis draws its aircraft from; NCEP's public `.nr` prepbufr dumps
//! carry almost none of it, and the dumps leave NOMADS after two days.
//!
//! The same files are also published in real time, anonymously, under
//! `https://madis-data.ncep.noaa.gov/madisPublic1/data/` with the same
//! sub-paths and names (`point/acars/netcdf/YYYYMMDD_HH00.gz` and so on):
//! the real-time directory keeps about five and a half days and each hour
//! file is rewritten every few minutes while its reports arrive (until about
//! 1.5 h after the hour), while the dated archive fills about two days
//! behind (measured 2026-10-16: archive day 10-14 held 8 of 24 hours, 10-15
//! none; the real-time directory held 10-10 20Z to 10-16 08Z).  No account
//! is needed for either: the account-gated part of MADIS is the restricted
//! providers, which are absent from the public files by MADIS policy.
//!
//! `fetch` downloads the hours of a window with a SHA-256, the server's
//! `Last-Modified` and the instant the bytes arrived, and counts an hour
//! neither feed has rather than failing on it.  `--feed auto` (the default)
//! asks the real-time directory first and the dated archive when real time
//! answers 404 (an hour older than its retention); `--feed realtime` and
//! `--feed archive` ask one.  Each file's record names the feed it came from.
//! A real-time hour file fetched again later holds the reports that arrived
//! since; its rows carry the later arrival instant as `received_time`, so an
//! information cutoff never admits them into an analysis that ran earlier.  `table` decodes the
//! files (gzip through flate2, classic NetCDF through the vendored
//! `netcdf-reader`, nothing else) into the neutral observation table
//! `gpuwm-obs.table.v2` (`rw_obs::table`), one row per report, level and
//! variable:
//!
//! * acars: `temperature_k`, `wind_u_m_s`/`wind_v_m_s` and, where the
//!   aircraft carries a humidity sensor, `dewpoint_k`; measurement
//!   `aircraft_level`, anchored at the ICAO standard-atmosphere pressure of
//!   the reported pressure altitude (pressure altitude IS that pressure by
//!   definition), `elevation_m` the pressure altitude, valid at `timeObs`,
//!   station id the archive's encrypted tail number;
//! * raob: every mandatory, significant-temperature, significant-wind (by
//!   pressure), tropopause and maximum-wind level that has a pressure;
//!   measurement `sonde_level`; dewpoint from the archive's dewpoint
//!   depression.  The MADIS file carries no per-level elapsed time, so
//!   every level is dated to the release time (`relTime`, else the synoptic
//!   time) and the record says so.  MADIS's `relTime` carries the release
//!   hour and minute on the SYNOPTIC day: the 00 UTC soundings released at
//!   23:0x the day before read 23:0x on the synoptic day, 23 h after their
//!   synoptic time (measured: every US 00 UTC sounding of 2026-10-14, e.g.
//!   72249 relTime 2026-10-14 23:04 for synTime 2026-10-14 00:00), and a
//!   few stations carry a placeholder (74794: 2012-01-01).  So a release
//!   time is moved by one day to the one nearest the synoptic time, and one
//!   further off, or still more than 3 h from it, is not believed: the report is dated
//!   to the synoptic time.  Both are counted.  Before this, every US 00 UTC
//!   sounding was dated a day late and fell outside every live window; significant wind levels stated by height
//!   only have no pressure and are counted, not written;
//! * mesonet: `temperature_k` (`screen_temperature_2m`), `dewpoint_k`
//!   (`screen_dewpoint_2m`), `surface_pressure_pa` (`station_pressure`) and
//!   wind as `platform_wind`, because mesonet anemometers stand anywhere
//!   from 2 to 10 m and the archive does not say which.
//!
//! **Quality control** is the archive's own: every variable carries a MADIS
//! QC summary character (`xxxDD`).  `V` (verified), `S` (screened), `C`
//! (coarse pass), `G`/`K` (subjectively good) are kept; `X` (failed the
//! validity check), `Q` (questioned), `B`/`k` (subjectively bad) and
//! anything else are dropped, each counted by character and variable; `Z`
//! (no QC applied) is dropped unless `--accept-unchecked`.  Position and
//! altitude carry their own summary characters on aircraft and are screened
//! the same way; `timeObs` is never checked by MADIS (`Z` on every report),
//! so there only a failing character drops the report.  A value equal to the variable's `_FillValue` or
//! `missing_value`, or outside the table's gross bounds, is counted and not
//! written.  An exact repeat of a row (same platform, time, place, level and
//! variable) is written once and counted.
//!
//! **Errors** are GSI's, from the regional conventional error table the
//! operational rapid-refresh analysis reads (`rw_obs::errtable`, default
//! NCEP's published table, or `--error-table`): report types 133/233 for
//! aircraft, 120/220 for radiosondes, 188/288 for mesonets (temperature and
//! humidity / wind), interpolated at the row's pressure exactly as GSI
//! interpolates and floored as GSI floors.  The humidity column is a
//! relative-humidity error in tenths; a dewpoint row's error is that error
//! carried to dewpoint at the observed temperature and dewpoint:
//! `sigma_Td = (0.1 * e_q / RH) / (L / (Rv * Td^2))`, with RH from Bolton's
//! (1980) saturation vapour pressure, L = 2.5e6 J/kg, Rv = 461.5 J/kg/K,
//! floored at 1 K and capped at 10 K.  Surface pressure is the table's hPa
//! column in Pa.  A type the table does not carry for a variable is
//! counted (`error_not_in_table`) and that row is not written.
//!
//! ```text
//! rw_madis fetch --kind acars|raob|mesonet --start TIME --end TIME --out DIR
//!                [--feed auto|realtime|archive]
//! rw_madis table --kind K --files F ... --out FILE.csv [--fetch-record R.json]
//!                [--start T --end T] [--bbox S,N,W,E] [--error-table T]
//!                [--accept-unchecked] [--thin-minutes M]
//! ```

use std::collections::{BTreeMap, HashSet};
use std::error::Error;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use chrono::{DateTime, Duration, TimeZone, Timelike, Utc};
use netcdf_reader::NcFile;
use serde_json::{json, Value};

use rw_nexrad::s3::parse_time;
use rw_obs::errtable::{Column, ErrorTable};
use rw_obs::seam::seam_time;
use rw_obs::table::{
    isa_altitude_m, revision_of, wind_components, RowProvenance, TableRow, TableWriter,
    GROSS_DEWPOINT_K, GROSS_SURFACE_PRESSURE_PA, GROSS_TEMPERATURE_K, GROSS_WIND_M_S,
    MEAS_AIRCRAFT_LEVEL, MEAS_PLATFORM_WIND, MEAS_SCREEN_DEWPOINT_2M, MEAS_SCREEN_TEMPERATURE_2M,
    MEAS_SONDE_LEVEL, MEAS_STATION_PRESSURE, TABLE_SCHEMA, VAR_DEWPOINT, VAR_SURFACE_PRESSURE,
    VAR_TEMPERATURE, VAR_WIND_U, VAR_WIND_V,
};
use rw_obs::{err, hex_sha256};

const VERSION: &str = env!("CARGO_PKG_VERSION");

pub static GPUWM_BRIDGE_SOURCE_REV_STAMP: &str =
    concat!("GPUWM_BRIDGE_SOURCE_REV=", env!("GPUWM_BRIDGE_SOURCE_REV"));

const ARCHIVE: &str = "https://madis-data.ncep.noaa.gov/madisPublic1/data/archive";
/// The real-time directory: the archive's sub-paths and file names, without
/// the date directories (module notes).
const REALTIME: &str = "https://madis-data.ncep.noaa.gov/madisPublic1/data";
const FETCH_SCHEMA: &str = "gpuwm-obs.madis-fetch.v1";
const TABLE_RECORD_SCHEMA: &str = "gpuwm-obs.madis-table.v1";
const ABI_MARKER: &str = "gpuwm-obs.madis-fetch.v1\tgpuwm-obs.madis-table.v1\tgpuwm-obs.table.v2\t\
acars\traob\tmesonet\tqc-summary\terrtable\trealtime";

/// GSI's report types for each platform: (temperature and humidity, wind).
const TYPES_ACARS: (u16, u16) = (133, 233);
const TYPES_RAOB: (u16, u16) = (120, 220);
const TYPES_MESONET: (u16, u16) = (188, 288);

/// Latent heat of vaporisation and the water-vapour gas constant of the
/// dewpoint-error conversion (module notes).
const LATENT_HEAT: f64 = 2.5e6;
const RV: f64 = 461.5;
const DEWPOINT_ERROR_FLOOR_K: f64 = 1.0;
const DEWPOINT_ERROR_CAP_K: f64 = 10.0;
/// GSI's fill for a variable a report type does not carry.
const TABLE_FILL_THRESHOLD: f64 = 1.0e8;

const USAGE: &str = "\
usage: rw_madis <fetch|table> [OPTIONS]
       rw_madis --version | --help | --abi

  fetch   download the archive's hourly files of a window, with a sha256, the
          server's Last-Modified and the arrival instant per file
  table   decode files into a `gpuwm-obs.table.v2` CSV (record beside it)

options
  --kind K              acars, raob or mesonet
  --start TIME          window start, e.g. 2026-10-01T17:00:00Z
  --end TIME            window end (inclusive); fetch takes every hour file
                        from start's hour through end's hour; table keeps
                        reports valid inside [start, end]
  --out DIR|FILE        fetch: directory; table: the CSV (record = .json beside)
  --feed F              fetch: auto (default: the real-time directory, then the
                        dated archive for an hour real time no longer holds),
                        realtime or archive
  --files F ...         table: the files (.gz or decompressed); repeatable
  --fetch-record R      table: the fetch record, for Last-Modified and arrival
  --bbox S,N,W,E        table: keep reports inside this box (degrees)
  --error-table T       table: a GSI errtable (default NCEP's regional table)
  --accept-unchecked    table: keep values whose QC summary is Z (no QC)
  --thin-minutes M      table, mesonet: per station keep the report nearest
                        each M-minute mark (default: keep every report)
";

fn main() -> ExitCode {
    let _ = std::hint::black_box(GPUWM_BRIDGE_SOURCE_REV_STAMP);
    rw_obs::cli::main("rw_madis", run)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Acars,
    Raob,
    Mesonet,
}

impl Kind {
    fn parse(text: &str) -> Result<Self, Box<dyn Error>> {
        match text {
            "acars" => Ok(Kind::Acars),
            "raob" => Ok(Kind::Raob),
            "mesonet" => Ok(Kind::Mesonet),
            other => Err(err(format!("--kind {other:?} is not acars, raob or mesonet"))),
        }
    }
    fn name(self) -> &'static str {
        match self {
            Kind::Acars => "acars",
            Kind::Raob => "raob",
            Kind::Mesonet => "mesonet",
        }
    }
    fn archive_dir(self) -> &'static str {
        match self {
            Kind::Acars => "point/acars/netcdf",
            Kind::Raob => "point/raob/netcdf",
            Kind::Mesonet => "LDAD/mesonet/netCDF",
        }
    }
    fn source(self) -> &'static str {
        match self {
            Kind::Acars => "madis-acars",
            Kind::Raob => "madis-raob",
            Kind::Mesonet => "madis-mesonet",
        }
    }
    fn report_types(self) -> (u16, u16) {
        match self {
            Kind::Acars => TYPES_ACARS,
            Kind::Raob => TYPES_RAOB,
            Kind::Mesonet => TYPES_MESONET,
        }
    }
}

/// Which public directory a fetch reads (module notes).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum Feed {
    #[default]
    Auto,
    Realtime,
    Archive,
}

impl Feed {
    fn parse(text: &str) -> Result<Self, Box<dyn Error>> {
        match text {
            "auto" => Ok(Feed::Auto),
            "realtime" => Ok(Feed::Realtime),
            "archive" => Ok(Feed::Archive),
            other => Err(err(format!("--feed {other:?} is not auto, realtime or archive"))),
        }
    }
    fn name(self) -> &'static str {
        match self {
            Feed::Auto => "auto",
            Feed::Realtime => "realtime",
            Feed::Archive => "archive",
        }
    }
}

#[derive(Debug, Default)]
struct Options {
    feed: Feed,
    kind: Option<Kind>,
    start: Option<DateTime<Utc>>,
    end: Option<DateTime<Utc>>,
    out: Option<PathBuf>,
    files: Vec<PathBuf>,
    fetch_records: Vec<PathBuf>,
    bbox: Option<[f64; 4]>,
    error_table: Option<PathBuf>,
    accept_unchecked: bool,
    thin_minutes: Option<i64>,
}

impl Options {
    fn parse(args: &[String]) -> Result<Self, Box<dyn Error>> {
        let mut o = Options::default();
        let mut i = 0;
        let value = |i: usize, name: &str| -> Result<String, Box<dyn Error>> {
            args.get(i + 1).cloned().ok_or_else(|| err(format!("{name} needs a value")))
        };
        while i < args.len() {
            let a = args[i].as_str();
            match a {
                "--kind" => {
                    o.kind = Some(Kind::parse(&value(i, a)?)?);
                    i += 2;
                }
                "--feed" => {
                    o.feed = Feed::parse(&value(i, a)?)?;
                    i += 2;
                }
                "--start" => {
                    o.start = Some(parse_time(&value(i, a)?)?);
                    i += 2;
                }
                "--end" => {
                    o.end = Some(parse_time(&value(i, a)?)?);
                    i += 2;
                }
                "--out" => {
                    o.out = Some(PathBuf::from(value(i, a)?));
                    i += 2;
                }
                "--fetch-record" => {
                    o.fetch_records.push(PathBuf::from(value(i, a)?));
                    i += 2;
                }
                "--error-table" => {
                    o.error_table = Some(PathBuf::from(value(i, a)?));
                    i += 2;
                }
                "--accept-unchecked" => {
                    o.accept_unchecked = true;
                    i += 1;
                }
                "--thin-minutes" => {
                    let m: i64 = value(i, a)?
                        .parse()
                        .map_err(|_| err("--thin-minutes is a whole number of minutes"))?;
                    if m <= 0 || m > 1440 {
                        return Err(err("--thin-minutes must be 1..1440"));
                    }
                    o.thin_minutes = Some(m);
                    i += 2;
                }
                "--bbox" => {
                    let text = value(i, a)?;
                    let parts: Vec<f64> = text
                        .split(',')
                        .map(|p| p.trim().parse::<f64>())
                        .collect::<Result<_, _>>()
                        .map_err(|_| err(format!("--bbox {text:?} is not S,N,W,E numbers")))?;
                    if parts.len() != 4 || parts[0] >= parts[1] || parts[2] >= parts[3] {
                        return Err(err(format!("--bbox {text:?} must be S,N,W,E with S<N and W<E")));
                    }
                    o.bbox = Some([parts[0], parts[1], parts[2], parts[3]]);
                    i += 2;
                }
                "--files" => {
                    i += 1;
                    while i < args.len() && !args[i].starts_with("--") {
                        for part in args[i].split(',').filter(|p| !p.is_empty()) {
                            o.files.push(PathBuf::from(part));
                        }
                        i += 1;
                    }
                }
                other => return Err(err(format!("unknown option {other:?}\n{USAGE}"))),
            }
        }
        Ok(o)
    }
}

fn run(args: &[String]) -> Result<String, Box<dyn Error>> {
    let Some(first) = args.first() else {
        return Ok(USAGE.to_string());
    };
    match first.as_str() {
        "--version" => Ok(format!("rw_madis {VERSION}\n")),
        "--help" | "-h" => Ok(USAGE.to_string()),
        "--abi" => Ok(format!("{ABI_MARKER}\n")),
        "fetch" => fetch(&Options::parse(&args[1..])?),
        "table" => table(&Options::parse(&args[1..])?),
        other => Err(err(format!("unknown command {other:?}\n{USAGE}"))),
    }
}

// ---------------------------------------------------------------------------
// fetch
// ---------------------------------------------------------------------------

fn hour_floor(t: DateTime<Utc>) -> DateTime<Utc> {
    t.with_minute(0).and_then(|t| t.with_second(0)).and_then(|t| t.with_nanosecond(0)).unwrap_or(t)
}

fn hour_url(kind: Kind, hour: DateTime<Utc>) -> (String, String) {
    let name = format!("{}.gz", hour.format("%Y%m%d_%H00"));
    let url = format!("{ARCHIVE}/{}/{}/{}", hour.format("%Y/%m/%d"), kind.archive_dir(), name);
    (url, name)
}

/// The real-time directory's URL of the same hour file.
fn realtime_url(kind: Kind, hour: DateTime<Utc>) -> String {
    format!("{REALTIME}/{}/{}.gz", kind.archive_dir(), hour.format("%Y%m%d_%H00"))
}

/// The URLs a fetch asks for one hour, in order, with the feed each is.
fn hour_candidates(kind: Kind, hour: DateTime<Utc>, feed: Feed) -> (Vec<(String, Feed)>, String) {
    let (archive, name) = hour_url(kind, hour);
    let realtime = realtime_url(kind, hour);
    let urls = match feed {
        Feed::Auto => vec![(realtime, Feed::Realtime), (archive, Feed::Archive)],
        Feed::Realtime => vec![(realtime, Feed::Realtime)],
        Feed::Archive => vec![(archive, Feed::Archive)],
    };
    (urls, name)
}

/// GET with the status and Last-Modified kept: `Ok(None)` is a 404, which
/// for radiosondes is the archive saying nobody launched that hour.
fn get_with_headers(
    agent: &ureq::Agent,
    url: &str,
) -> Result<Option<(Vec<u8>, Option<String>)>, Box<dyn Error>> {
    match rw_http::get(agent, url) {
        Ok(mut response) => {
            let last_modified = response
                .headers()
                .get("last-modified")
                .and_then(|v| v.to_str().ok())
                .map(|s| s.to_string());
            let bytes = response
                .body_mut()
                .with_config()
                .limit(rw_obs::net::MAX_RESPONSE_BYTES)
                .read_to_vec()
                .map_err(|e| err(format!("reading {url} failed: {e}")))?;
            if bytes.is_empty() {
                return Err(err(format!("{url} answered with an empty body")));
            }
            Ok(Some((bytes, last_modified)))
        }
        Err(ureq::Error::StatusCode(404)) => Ok(None),
        Err(e) => Err(err(format!("GET {url} failed: {e}"))),
    }
}

fn http_date(text: &str) -> Option<DateTime<Utc>> {
    DateTime::parse_from_rfc2822(text.trim()).ok().map(|t| t.with_timezone(&Utc))
}

fn fetch(o: &Options) -> Result<String, Box<dyn Error>> {
    let kind = o.kind.ok_or_else(|| err("fetch needs --kind"))?;
    let start = o.start.ok_or_else(|| err("fetch needs --start"))?;
    let end = o.end.ok_or_else(|| err("fetch needs --end"))?;
    let out = o.out.clone().ok_or_else(|| err("fetch needs --out DIR"))?;
    if end < start {
        return Err(err("--end is before --start"));
    }
    std::fs::create_dir_all(&out).map_err(|e| err(format!("cannot create {}: {e}", out.display())))?;
    let agent = rw_obs::net::agent();
    let mut files = Vec::new();
    let mut missing = Vec::new();
    let mut hour = hour_floor(start);
    while hour <= end {
        let (candidates, name) = hour_candidates(kind, hour, o.feed);
        let mut hit = None;
        for (url, feed) in &candidates {
            if let Some(found) = get_with_headers(&agent, url)? {
                hit = Some((url.clone(), *feed, found));
                break;
            }
        }
        match hit {
            Some((url, feed, (bytes, last_modified))) => {
                let path = out.join(&name);
                std::fs::write(&path, &bytes).map_err(|e| err(format!("cannot write {}: {e}", path.display())))?;
                files.push(json!({
                    "hour": seam_time(hour),
                    "url": url,
                    "feed": feed.name(),
                    "path": rw_obs::absolute_uri(&path),
                    "name": name,
                    "bytes": bytes.len(),
                    "sha256": hex_sha256(&bytes),
                    "last_modified": last_modified.as_deref().and_then(http_date).map(seam_time),
                    "fetched_at": seam_time(Utc::now()),
                }));
            }
            None => missing.push(json!({
                "hour": seam_time(hour),
                "urls": candidates.iter().map(|(u, _)| u.clone()).collect::<Vec<_>>(),
            })),
        }
        hour += Duration::hours(1);
    }
    let record = json!({
        "schema": FETCH_SCHEMA,
        "status": if files.is_empty() { "EMPTY" } else { "READY" },
        "kind": kind.name(),
        "archive": ARCHIVE,
        "realtime": REALTIME,
        "feed": o.feed.name(),
        "start": seam_time(start),
        "end": seam_time(end),
        "files": files,
        "hours_not_in_archive": missing,
        "note": "times are UTC without a zone suffix (the seam spelling); a missing hour is \
a 404 from every feed asked, counted rather than failed (radiosonde hours nobody launched); \
a real-time hour file grows until about 1.5 h after its hour, so a later fetch holds more reports",
    });
    let text = serde_json::to_string_pretty(&record)? + "\n";
    let record_path = out.join(format!("fetch-{}.json", kind.name()));
    std::fs::write(&record_path, &text).map_err(|e| err(format!("cannot write {}: {e}", record_path.display())))?;
    Ok(text)
}

// ---------------------------------------------------------------------------
// decode helpers
// ---------------------------------------------------------------------------

fn read_file_bytes(path: &Path) -> Result<(Vec<u8>, Vec<u8>), Box<dyn Error>> {
    let raw = std::fs::read(path).map_err(|e| err(format!("cannot read {}: {e}", path.display())))?;
    let data = if raw.starts_with(&[0x1f, 0x8b]) {
        let mut out = Vec::with_capacity(raw.len() * 8);
        flate2::read::GzDecoder::new(&raw[..])
            .read_to_end(&mut out)
            .map_err(|e| err(format!("{}: gzip: {e}", path.display())))?;
        out
    } else {
        raw.clone()
    };
    Ok((raw, data))
}

/// A numeric variable as f64 with its fill and missing values replaced by
/// NaN (so one `is_finite` is the presence test), flattened.
fn numeric(nc: &NcFile, name: &str) -> Result<Vec<f64>, Box<dyn Error>> {
    let var = nc.variable(name).map_err(|e| err(format!("variable {name}: {e}")))?;
    let mut sentinels = Vec::new();
    for attr in ["_FillValue", "missing_value"] {
        if let Some(a) = var.attribute(attr) {
            if let Some(v) = a.value.as_f64() {
                sentinels.push(v);
            }
        }
    }
    let data = nc.read_variable_as_f64(name).map_err(|e| err(format!("variable {name}: {e}")))?;
    Ok(data
        .iter()
        .map(|&v| {
            if !v.is_finite() || v.abs() >= 1.0e30 || sentinels.iter().any(|s| v == *s || ((v - s).abs() <= 1e-6 * s.abs().max(1.0)))
            {
                f64::NAN
            } else {
                v
            }
        })
        .collect())
}

fn numeric_opt(nc: &NcFile, name: &str) -> Result<Option<Vec<f64>>, Box<dyn Error>> {
    if nc.variable(name).is_err() {
        return Ok(None);
    }
    numeric(nc, name).map(Some)
}

/// A char variable of shape `(recNum,)` or `(recNum, n)` as `count * width`
/// bytes (NUL where the archive wrote nothing), so per-record characters
/// stay aligned however the strings were trimmed on the way out.
fn chars(nc: &NcFile, name: &str, count: usize, width: usize) -> Result<Vec<u8>, Box<dyn Error>> {
    let strings = nc.read_variable_as_strings(name).map_err(|e| err(format!("variable {name}: {e}")))?;
    let mut out = vec![0u8; count * width];
    if width == 1 && strings.len() == 1 {
        let b = strings[0].as_bytes();
        if b.len() > count {
            return Err(err(format!("{name}: {} bytes decoded for {count} records (non-ASCII summary)", b.len())));
        }
        out[..b.len()].copy_from_slice(b);
        return Ok(out);
    }
    if strings.len() != count {
        return Err(err(format!("{name}: {} rows for {count} records", strings.len())));
    }
    for (r, s) in strings.iter().enumerate() {
        let b = s.as_bytes();
        if b.len() > width {
            return Err(err(format!("{name}: row {r} has {} bytes for width {width}", b.len())));
        }
        out[r * width..r * width + b.len()].copy_from_slice(b);
    }
    Ok(out)
}

fn ids(nc: &NcFile, name: &str) -> Result<Vec<String>, Box<dyn Error>> {
    Ok(nc
        .read_variable_as_strings(name)
        .map_err(|e| err(format!("variable {name}: {e}")))?
        .into_iter()
        .map(|s| s.trim_matches(|c: char| c == '\0' || c.is_whitespace()).to_string())
        .collect())
}

/// The MADIS QC summary screen: `Ok` keeps the value, `Err(c)` names the
/// character that dropped it.
fn qc_screen(c: u8, accept_unchecked: bool) -> Result<(), char> {
    match c {
        b'V' | b'S' | b'C' | b'G' | b'K' => Ok(()),
        b'Z' if accept_unchecked => Ok(()),
        0 => Err('0'),
        b' ' => Err('_'),
        other => Err(other as char),
    }
}

/// The ICAO standard-atmosphere pressure of an altitude: the exact inverse
/// of `rw_obs::table::isa_altitude_m` (same constants).
fn isa_pressure_pa(altitude_m: f64) -> f64 {
    const P0: f64 = 101_325.0;
    const T0: f64 = 288.15;
    const LAPSE: f64 = 0.0065;
    const EXPONENT: f64 = 5.255877;
    const P_TROP: f64 = 22_632.06;
    const Z_TROP: f64 = 11_000.0;
    const SCALE: f64 = 6341.62;
    if altitude_m <= Z_TROP {
        P0 * (1.0 - LAPSE * altitude_m / T0).powf(EXPONENT)
    } else {
        P_TROP * (-(altitude_m - Z_TROP) / SCALE).exp()
    }
}

/// Bolton's (1980) saturation vapour pressure over water, hPa.
fn bolton_es_hpa(t_k: f64) -> f64 {
    let c = t_k - 273.15;
    6.112 * (17.67 * c / (c + 243.5)).exp()
}

/// The table's relative-humidity error (tenths) carried to dewpoint (K).
fn dewpoint_error_k(table_q_tenths: f64, t_k: f64, td_k: f64) -> f64 {
    let rh = (bolton_es_hpa(td_k) / bolton_es_hpa(t_k)).clamp(0.01, 1.0);
    let sigma_rh = 0.1 * table_q_tenths;
    let dlnes_dtd = LATENT_HEAT / (RV * td_k * td_k);
    ((sigma_rh / rh) / dlnes_dtd).clamp(DEWPOINT_ERROR_FLOOR_K, DEWPOINT_ERROR_CAP_K)
}

/// The instant a sounding was released, from MADIS's `relTime` and
/// `synTime` (module notes): the release time moved by a day to the one
/// nearest the synoptic time, and the synoptic time itself when the release
/// time is missing, more than a day off, or still more than 3 h away.  The second value
/// names the correction made, for the counters.
fn release_instant(rel: f64, syn: f64) -> (Option<DateTime<Utc>>, Option<&'static str>) {
    match (epoch(rel), epoch(syn)) {
        (Some(released), Some(synoptic)) => {
            let days = ((released - synoptic).num_seconds() as f64 / 86_400.0).round() as i64;
            let moved = released - Duration::days(days);
            if days.abs() > 1 || (moved - synoptic).num_seconds().abs() > 3 * 3600 {
                (Some(synoptic), Some("reports_release_time_implausible_dated_to_synoptic_time"))
            } else if days != 0 {
                (Some(moved), Some("reports_release_time_moved_to_the_synoptic_day"))
            } else {
                (Some(moved), None)
            }
        }
        (Some(released), None) => (Some(released), None),
        (None, synoptic) => (synoptic, Some("reports_dated_to_synoptic_time")),
    }
}

fn epoch(seconds: f64) -> Option<DateTime<Utc>> {
    if !seconds.is_finite() || seconds <= 0.0 {
        return None;
    }
    Utc.timestamp_opt(seconds.round() as i64, 0).single()
}

struct Source {
    revision_sha: String,
    published: Option<DateTime<Utc>>,
    received: Option<DateTime<Utc>>,
}

/// Collects rows, counts every screen, and writes each row once.
struct Builder<'a> {
    kind: Kind,
    accept_unchecked: bool,
    table: &'a ErrorTable,
    window: (Option<DateTime<Utc>>, Option<DateTime<Utc>>),
    bbox: Option<[f64; 4]>,
    writer: TableWriter,
    rows_by_variable: BTreeMap<String, usize>,
    rows_by_hour: BTreeMap<String, usize>,
    counts: BTreeMap<String, usize>,
    dropped_by_qc: BTreeMap<String, BTreeMap<String, usize>>,
    seen: HashSet<String>,
}

impl<'a> Builder<'a> {
    fn count(&mut self, key: &str) {
        *self.counts.entry(key.to_string()).or_insert(0) += 1;
    }

    fn qc(&mut self, variable: &str, c: u8) -> bool {
        self.qc_with(variable, c, self.accept_unchecked)
    }

    /// The screen with `Z` accepted: for a field MADIS never checks (an
    /// aircraft's `timeObs` is `Z` on every report of the archive), where
    /// "unchecked" is the field's only state and not a warning.
    fn qc_never_checked(&mut self, variable: &str, c: u8) -> bool {
        self.qc_with(variable, c, true)
    }

    fn qc_with(&mut self, variable: &str, c: u8, accept_unchecked: bool) -> bool {
        match qc_screen(c, accept_unchecked) {
            Ok(()) => true,
            Err(ch) => {
                *self
                    .dropped_by_qc
                    .entry(variable.to_string())
                    .or_default()
                    .entry(ch.to_string())
                    .or_insert(0) += 1;
                false
            }
        }
    }

    /// Report-level screens: time window and box.  `true` keeps it.
    fn place(&mut self, lat: f64, lon: f64, when: Option<DateTime<Utc>>) -> bool {
        if !lat.is_finite() || !lon.is_finite() || lat.abs() > 90.0 {
            self.count("reports_without_position");
            return false;
        }
        let Some(when) = when else {
            self.count("reports_without_time");
            return false;
        };
        if self.window.0.is_some_and(|s| when < s) || self.window.1.is_some_and(|e| when > e) {
            self.count("reports_outside_time_window");
            return false;
        }
        if let Some([s, n, w, e]) = self.bbox {
            let lon = rw_obs::seam::wrap_longitude(lon);
            if lat < s || lat > n || lon < w || lon > e {
                self.count("reports_outside_bbox");
                return false;
            }
        }
        true
    }

    fn table_error(&mut self, report_type: u16, pressure_pa: f64, column: Column, what: &str) -> Option<f64> {
        let p_hpa = if pressure_pa.is_finite() { Some(pressure_pa / 100.0) } else { None };
        let raw = self.table.raw(report_type, p_hpa, column);
        if raw >= TABLE_FILL_THRESHOLD {
            self.count(&format!("error_not_in_table:{what}:{report_type}"));
            return None;
        }
        Some(raw.max(column.floor()))
    }

    #[allow(clippy::too_many_arguments)]
    fn push(
        &mut self,
        source: &Source,
        measurement: &'static str,
        station: &str,
        lat: f64,
        lon: f64,
        elevation_m: f64,
        level_pa: Option<f64>,
        when: DateTime<Utc>,
        nominal: Option<DateTime<Utc>>,
        variable: &str,
        value: f64,
        error: f64,
    ) {
        let (lo, hi) = match variable {
            v if v == VAR_TEMPERATURE => GROSS_TEMPERATURE_K,
            v if v == VAR_DEWPOINT => GROSS_DEWPOINT_K,
            v if v == VAR_SURFACE_PRESSURE => GROSS_SURFACE_PRESSURE_PA,
            _ => (-GROSS_WIND_M_S.1, GROSS_WIND_M_S.1),
        };
        if !value.is_finite() || value < lo || value > hi {
            self.count(&format!("gross_bounds:{variable}"));
            return;
        }
        if !(error.is_finite() && error > 0.0) || !elevation_m.is_finite() {
            self.count(&format!("invalid_error_or_anchor:{variable}"));
            return;
        }
        let key = format!(
            "{station}|{}|{:.4}|{:.4}|{}|{variable}",
            when.timestamp(),
            lat,
            lon,
            level_pa.map(|p| format!("{p:.0}")).unwrap_or_default()
        );
        if !self.seen.insert(key) {
            self.count(&format!("exact_repeat:{variable}"));
            return;
        }
        let mut provenance = RowProvenance::of_source(&source.revision_sha, source.published, source.received)
            .measuring(measurement);
        if let Some(n) = nominal {
            provenance = provenance.nominal(n);
        }
        *self
            .rows_by_hour
            .entry(hour_floor(when).format("%Y-%m-%dT%H").to_string())
            .or_insert(0) += 1;
        self.writer.push(
            TableRow {
                source: self.kind.source().to_string(),
                station_id: station.replace(',', "_"),
                latitude_deg: lat,
                longitude_deg: lon,
                elevation_m,
                level_pa,
                valid_time: when,
                variable: variable.to_string(),
                value,
                error,
                provenance,
            },
            &mut self.rows_by_variable,
        );
    }

    /// One wind (u and v rows) from direction and speed.
    #[allow(clippy::too_many_arguments)]
    fn push_wind(
        &mut self,
        source: &Source,
        measurement: &'static str,
        station: &str,
        lat: f64,
        lon: f64,
        elevation_m: f64,
        level_pa: Option<f64>,
        when: DateTime<Utc>,
        nominal: Option<DateTime<Utc>>,
        direction: f64,
        speed: f64,
        error: f64,
        zero_direction_is_variable: bool,
    ) {
        if !(0.0..=360.0).contains(&direction) || !(0.0..=GROSS_WIND_M_S.1).contains(&speed) {
            self.count("gross_bounds:wind");
            return;
        }
        if zero_direction_is_variable && direction == 0.0 && speed > 0.0 {
            self.count("wind_direction_variable");
            return;
        }
        let (u, v) = wind_components(direction, speed);
        self.push(source, measurement, station, lat, lon, elevation_m, level_pa, when, nominal, VAR_WIND_U, u, error);
        self.push(source, measurement, station, lat, lon, elevation_m, level_pa, when, nominal, VAR_WIND_V, v, error);
    }
}

// ---------------------------------------------------------------------------
// the three platforms
// ---------------------------------------------------------------------------

fn record_count(nc: &NcFile, probe: &str) -> Result<usize, Box<dyn Error>> {
    Ok(numeric(nc, probe)?.len())
}

fn decode_acars(nc: &NcFile, source: &Source, b: &mut Builder) -> Result<(), Box<dyn Error>> {
    let lat = numeric(nc, "latitude")?;
    let n = lat.len();
    let lon = numeric(nc, "longitude")?;
    let alt = numeric(nc, "altitude")?;
    let time = numeric(nc, "timeObs")?;
    let temp = numeric(nc, "temperature")?;
    let wdir = numeric(nc, "windDir")?;
    let wspd = numeric(nc, "windSpeed")?;
    let dewp = numeric_opt(nc, "dewpoint")?.unwrap_or_else(|| vec![f64::NAN; n]);
    let dd = |name: &str| chars(nc, name, n, 1);
    let (lat_dd, lon_dd, alt_dd, time_dd) = (dd("latitudeDD")?, dd("longitudeDD")?, dd("altitudeDD")?, dd("timeObsDD")?);
    let (t_dd, wd_dd, ws_dd) = (dd("temperatureDD")?, dd("windDirDD")?, dd("windSpeedDD")?);
    let td_dd = if nc.variable("dewpointDD").is_ok() { dd("dewpointDD")? } else { vec![b'Z'; n] };
    let tails = ids(nc, "en_tailNumber").unwrap_or_default();
    let (t_type, w_type) = Kind::Acars.report_types();
    for r in 0..n {
        b.count("reports_read");
        if !(b.qc("latitude", lat_dd[r]) && b.qc("longitude", lon_dd[r]) && b.qc("altitude", alt_dd[r]) && b.qc_never_checked("timeObs", time_dd[r])) {
            b.count("reports_failed_position_or_time_qc");
            continue;
        }
        let when = epoch(time[r]);
        if !b.place(lat[r], lon[r], when) {
            continue;
        }
        let when = when.expect("placed reports have a time");
        if !alt[r].is_finite() {
            b.count("reports_without_altitude");
            continue;
        }
        let p = isa_pressure_pa(alt[r]);
        let station = tails.get(r).filter(|s| !s.is_empty()).cloned().unwrap_or_else(|| format!("acars-{r}"));
        b.count("reports_placed");
        if temp[r].is_finite() {
            if b.qc(VAR_TEMPERATURE, t_dd[r]) {
                if let Some(e) = b.table_error(t_type, p, Column::Temperature, VAR_TEMPERATURE) {
                    b.push(source, MEAS_AIRCRAFT_LEVEL, &station, lat[r], lon[r], alt[r], Some(p), when, None, VAR_TEMPERATURE, temp[r], e);
                }
            }
        } else {
            b.count("value_missing:temperature_k");
        }
        if wdir[r].is_finite() && wspd[r].is_finite() {
            if b.qc("wind_direction", wd_dd[r]) & b.qc("wind_speed", ws_dd[r]) {
                if let Some(e) = b.table_error(w_type, p, Column::Wind, "wind") {
                    b.push_wind(source, MEAS_AIRCRAFT_LEVEL, &station, lat[r], lon[r], alt[r], Some(p), when, None, wdir[r], wspd[r], e, false);
                }
            }
        } else {
            b.count("value_missing:wind");
        }
        if dewp[r].is_finite() && temp[r].is_finite() {
            if b.qc(VAR_DEWPOINT, td_dd[r]) {
                if dewp[r] > temp[r] + 0.5 {
                    b.count("dewpoint_above_temperature");
                } else if let Some(eq) = b.table_error(t_type, p, Column::Humidity, VAR_DEWPOINT) {
                    let e = dewpoint_error_k(eq, temp[r], dewp[r].min(temp[r]));
                    b.push(source, MEAS_AIRCRAFT_LEVEL, &station, lat[r], lon[r], alt[r], Some(p), when, None, VAR_DEWPOINT, dewp[r].min(temp[r]), e);
                }
            }
        }
    }
    Ok(())
}

fn decode_raob(nc: &NcFile, source: &Source, b: &mut Builder) -> Result<(), Box<dyn Error>> {
    let lat = numeric(nc, "staLat")?;
    let n = lat.len();
    let lon = numeric(nc, "staLon")?;
    let elev = numeric(nc, "staElev")?;
    let syn = numeric(nc, "synTime")?;
    let rel = numeric(nc, "relTime")?;
    let wmo = numeric(nc, "wmoStaNum")?;
    let names = ids(nc, "staName").unwrap_or_default();
    let (t_type, w_type) = Kind::Raob.report_types();
    // (pressure, height, temperature, dewpoint depression, direction, speed) per block
    struct Block {
        name: &'static str,
        p: &'static str,
        z: Option<&'static str>,
        t: Option<&'static str>,
        dd: Option<&'static str>,
        wd: Option<&'static str>,
        ws: Option<&'static str>,
    }
    let blocks = [
        Block { name: "mandatory", p: "prMan", z: Some("htMan"), t: Some("tpMan"), dd: Some("tdMan"), wd: Some("wdMan"), ws: Some("wsMan") },
        Block { name: "significant_temperature", p: "prSigT", z: None, t: Some("tpSigT"), dd: Some("tdSigT"), wd: None, ws: None },
        Block { name: "significant_wind_by_pressure", p: "prSigW", z: None, t: None, dd: None, wd: Some("wdSigPrW"), ws: Some("wsSigPrW") },
        Block { name: "tropopause", p: "prTrop", z: None, t: Some("tpTrop"), dd: Some("tdTrop"), wd: Some("wdTrop"), ws: Some("wsTrop") },
        Block { name: "maximum_wind", p: "prMaxW", z: None, t: None, dd: None, wd: Some("wdMaxW"), ws: Some("wsMaxW") },
    ];
    if let Ok(levels) = numeric(nc, "htSigW") {
        let count = levels.iter().filter(|v| v.is_finite()).count();
        *b.counts.entry("levels_height_only_significant_wind_not_written".into()).or_insert(0) += count;
    }
    let mut reports = vec![true; n];
    let mut released = vec![None; n];
    for r in 0..n {
        b.count("reports_read");
        let (release, correction) = release_instant(rel[r], syn[r]);
        released[r] = release;
        if let Some(counter) = correction {
            b.count(counter);
        }
        reports[r] = b.place(lat[r], lon[r], release);
    }
    for block in &blocks {
        if nc.variable(block.p).is_err() {
            b.count(&format!("block_absent:{}", block.name));
            continue;
        }
        let p = numeric(nc, block.p)?;
        let width = if n == 0 { 0 } else { p.len() / n };
        let get = |name: Option<&'static str>| -> Result<Option<(Vec<f64>, Vec<u8>)>, Box<dyn Error>> {
            let Some(name) = name else { return Ok(None) };
            if nc.variable(name).is_err() {
                return Ok(None);
            }
            let values = numeric(nc, name)?;
            let flags = chars(nc, &format!("{name}DD"), n, width)?;
            Ok(Some((values, flags)))
        };
        let p_dd = chars(nc, &format!("{}DD", block.p), n, width)?;
        let z = get(block.z)?;
        let t = get(block.t)?;
        let dd = get(block.dd)?;
        let wd = get(block.wd)?;
        let ws = get(block.ws)?;
        for r in 0..n {
            if !reports[r] {
                continue;
            }
            let when = released[r].expect("placed");
            let nominal = epoch(syn[r]);
            let station = if wmo[r].is_finite() {
                format!("{:05}", wmo[r] as i64)
            } else {
                names.get(r).cloned().unwrap_or_else(|| format!("raob-{r}"))
            };
            for k in 0..width {
                let i = r * width + k;
                let pa = p[i] * 100.0;
                if !p[i].is_finite() {
                    continue;
                }
                b.count(&format!("levels_read:{}", block.name));
                if !b.qc("pressure", p_dd[i]) {
                    continue;
                }
                let height = z
                    .as_ref()
                    .filter(|(v, f)| v[i].is_finite() && qc_screen(f[i], b.accept_unchecked).is_ok())
                    .map(|(v, _)| v[i])
                    .unwrap_or_else(|| {
                        let isa = isa_altitude_m(pa);
                        if k == 0 && elev[r].is_finite() && block.name == "mandatory" && isa < elev[r] {
                            elev[r]
                        } else {
                            isa
                        }
                    });
                if let Some((tv, tf)) = t.as_ref() {
                    if tv[i].is_finite() && b.qc(VAR_TEMPERATURE, tf[i]) {
                        if let Some(e) = b.table_error(t_type, pa, Column::Temperature, VAR_TEMPERATURE) {
                            b.push(source, MEAS_SONDE_LEVEL, &station, lat[r], lon[r], height, Some(pa), when, nominal, VAR_TEMPERATURE, tv[i], e);
                        }
                        if let Some((dv, df)) = dd.as_ref() {
                            if dv[i].is_finite() && dv[i] >= 0.0 && b.qc(VAR_DEWPOINT, df[i]) {
                                let td = tv[i] - dv[i];
                                if let Some(eq) = b.table_error(t_type, pa, Column::Humidity, VAR_DEWPOINT) {
                                    let e = dewpoint_error_k(eq, tv[i], td);
                                    b.push(source, MEAS_SONDE_LEVEL, &station, lat[r], lon[r], height, Some(pa), when, nominal, VAR_DEWPOINT, td, e);
                                }
                            }
                        }
                    }
                }
                if let (Some((dv, df)), Some((sv, sf))) = (wd.as_ref(), ws.as_ref()) {
                    if dv[i].is_finite() && sv[i].is_finite() && (b.qc("wind_direction", df[i]) & b.qc("wind_speed", sf[i])) {
                        if let Some(e) = b.table_error(w_type, pa, Column::Wind, "wind") {
                            b.push_wind(source, MEAS_SONDE_LEVEL, &station, lat[r], lon[r], height, Some(pa), when, nominal, dv[i], sv[i], e, false);
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

fn decode_mesonet(nc: &NcFile, source: &Source, b: &mut Builder, thin_minutes: Option<i64>) -> Result<(), Box<dyn Error>> {
    let lat = numeric(nc, "latitude")?;
    let n = lat.len();
    let lon = numeric(nc, "longitude")?;
    let elev = numeric(nc, "elevation")?;
    let time = numeric(nc, "observationTime")?;
    let temp = numeric(nc, "temperature")?;
    let dewp = numeric(nc, "dewpoint")?;
    let wdir = numeric(nc, "windDir")?;
    let wspd = numeric(nc, "windSpeed")?;
    let psta = numeric_opt(nc, "stationPressure")?.unwrap_or_else(|| vec![f64::NAN; n]);
    let dd = |name: &str| chars(nc, name, n, 1);
    let (t_dd, td_dd, wd_dd, ws_dd) = (dd("temperatureDD")?, dd("dewpointDD")?, dd("windDirDD")?, dd("windSpeedDD")?);
    let ps_dd = if nc.variable("stationPressureDD").is_ok() { dd("stationPressureDD")? } else { vec![b'Z'; n] };
    let stations = ids(nc, "stationId")?;
    let providers = ids(nc, "dataProvider").unwrap_or_default();
    let (t_type, w_type) = Kind::Mesonet.report_types();

    // Thinning: per station, the report nearest each M-minute mark.
    let mut keep = vec![true; n];
    if let Some(m) = thin_minutes {
        let step = m * 60;
        let mut best: BTreeMap<(String, i64), (i64, usize)> = BTreeMap::new();
        for r in 0..n {
            let Some(t) = epoch(time[r]) else { continue };
            let s = t.timestamp();
            let mark = (s + step / 2).div_euclid(step) * step;
            let distance = (s - mark).abs();
            let key = (stations.get(r).cloned().unwrap_or_default(), mark);
            match best.get(&key) {
                Some(&(d, _)) if d <= distance => {}
                _ => {
                    best.insert(key, (distance, r));
                }
            }
        }
        keep = vec![false; n];
        for (_, (_, r)) in best {
            keep[r] = true;
        }
    }
    let mut providers_kept: BTreeMap<String, usize> = BTreeMap::new();
    for r in 0..n {
        b.count("reports_read");
        if !keep[r] {
            b.count("reports_thinned");
            continue;
        }
        let when = epoch(time[r]);
        if !b.place(lat[r], lon[r], when) {
            continue;
        }
        let when = when.expect("placed");
        if !elev[r].is_finite() {
            b.count("reports_without_elevation");
            continue;
        }
        let station = stations.get(r).filter(|s| !s.is_empty()).cloned().unwrap_or_else(|| format!("mesonet-{r}"));
        b.count("reports_placed");
        *providers_kept.entry(providers.get(r).cloned().unwrap_or_default()).or_insert(0) += 1;
        let p_for_table = if psta[r].is_finite() { psta[r] } else { isa_pressure_pa(elev[r]) };
        if temp[r].is_finite() && b.qc(VAR_TEMPERATURE, t_dd[r]) {
            if let Some(e) = b.table_error(t_type, p_for_table, Column::Temperature, VAR_TEMPERATURE) {
                b.push(source, MEAS_SCREEN_TEMPERATURE_2M, &station, lat[r], lon[r], elev[r], None, when, None, VAR_TEMPERATURE, temp[r], e);
            }
        }
        if dewp[r].is_finite() && temp[r].is_finite() && b.qc(VAR_DEWPOINT, td_dd[r]) {
            if dewp[r] > temp[r] + 0.5 {
                b.count("dewpoint_above_temperature");
            } else if let Some(eq) = b.table_error(t_type, p_for_table, Column::Humidity, VAR_DEWPOINT) {
                let td = dewp[r].min(temp[r]);
                let e = dewpoint_error_k(eq, temp[r], td);
                b.push(source, MEAS_SCREEN_DEWPOINT_2M, &station, lat[r], lon[r], elev[r], None, when, None, VAR_DEWPOINT, td, e);
            }
        }
        if psta[r].is_finite() && b.qc(VAR_SURFACE_PRESSURE, ps_dd[r]) {
            if let Some(e_hpa) = b.table_error(t_type, p_for_table, Column::SurfacePressure, VAR_SURFACE_PRESSURE) {
                b.push(source, MEAS_STATION_PRESSURE, &station, lat[r], lon[r], elev[r], None, when, None, VAR_SURFACE_PRESSURE, psta[r], e_hpa * 100.0);
            }
        }
        if wdir[r].is_finite() && wspd[r].is_finite() && (b.qc("wind_direction", wd_dd[r]) & b.qc("wind_speed", ws_dd[r])) {
            if let Some(e) = b.table_error(w_type, p_for_table, Column::Wind, "wind") {
                b.push_wind(source, MEAS_PLATFORM_WIND, &station, lat[r], lon[r], elev[r], None, when, None, wdir[r], wspd[r], e, true);
            }
        }
    }
    for (provider, count) in providers_kept {
        *b.counts.entry(format!("reports_placed_by_provider:{provider}")).or_insert(0) += count;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// table
// ---------------------------------------------------------------------------

/// name -> (sha256, Last-Modified, fetched_at) from the fetch records.
fn fetch_index(paths: &[PathBuf]) -> Result<BTreeMap<String, (String, Option<DateTime<Utc>>, Option<DateTime<Utc>>)>, Box<dyn Error>> {
    let mut index = BTreeMap::new();
    for path in paths {
        let text = std::fs::read_to_string(path).map_err(|e| err(format!("cannot read {}: {e}", path.display())))?;
        let record: Value = serde_json::from_str(&text).map_err(|e| err(format!("{}: {e}", path.display())))?;
        if record.get("schema").and_then(Value::as_str) != Some(FETCH_SCHEMA) {
            return Err(err(format!("{} is not a {FETCH_SCHEMA} record", path.display())));
        }
        for f in record.get("files").and_then(Value::as_array).cloned().unwrap_or_default() {
            let name = f.get("name").and_then(Value::as_str).unwrap_or_default().to_string();
            let sha = f.get("sha256").and_then(Value::as_str).unwrap_or_default().to_string();
            let time = |k: &str| f.get(k).and_then(Value::as_str).and_then(|s| parse_time(&format!("{s}Z")).ok());
            index.insert(name, (sha, time("last_modified"), time("fetched_at")));
        }
    }
    Ok(index)
}

fn table(o: &Options) -> Result<String, Box<dyn Error>> {
    let kind = o.kind.ok_or_else(|| err("table needs --kind"))?;
    let out = o.out.clone().ok_or_else(|| err("table needs --out FILE.csv"))?;
    if o.files.is_empty() {
        return Err(err("table needs --files"));
    }
    if o.thin_minutes.is_some() && kind != Kind::Mesonet {
        return Err(err("--thin-minutes thins mesonet stations; it has no meaning for this kind"));
    }
    let owned;
    let error_table: &ErrorTable = match &o.error_table {
        Some(path) => {
            let text = std::fs::read_to_string(path).map_err(|e| err(format!("cannot read {}: {e}", path.display())))?;
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("errtable").to_string();
            owned = ErrorTable::parse(&text, &name)?;
            &owned
        }
        None => ErrorTable::default_table(),
    };
    let index = fetch_index(&o.fetch_records)?;
    let mut builder = Builder {
        kind,
        accept_unchecked: o.accept_unchecked,
        table: error_table,
        window: (o.start, o.end),
        bbox: o.bbox,
        writer: TableWriter::new(),
        rows_by_variable: BTreeMap::new(),
        rows_by_hour: BTreeMap::new(),
        counts: BTreeMap::new(),
        dropped_by_qc: BTreeMap::new(),
        seen: HashSet::new(),
    };
    let mut inputs = Vec::new();
    for path in &o.files {
        let (raw, data) = read_file_bytes(path)?;
        let sha = hex_sha256(&raw);
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or_default().to_string();
        let gz_name = if name.ends_with(".gz") { name.clone() } else { format!("{name}.gz") };
        let (published, received, fetched_sha) = match index.get(&gz_name) {
            Some((s, p, r)) => (*p, *r, Some(s.clone())),
            None => (None, None, None),
        };
        if let Some(fs) = &fetched_sha {
            if raw.starts_with(&[0x1f, 0x8b]) && fs != &sha {
                return Err(err(format!("{}: sha256 {sha} differs from its fetch record's {fs}", path.display())));
            }
        }
        let source = Source { revision_sha: fetched_sha.clone().unwrap_or_else(|| sha.clone()), published, received };
        let nc = NcFile::from_bytes(&data).map_err(|e| err(format!("{}: not a NetCDF file the reader takes: {e}", path.display())))?;
        let before = builder.writer.len();
        match kind {
            Kind::Acars => decode_acars(&nc, &source, &mut builder)?,
            Kind::Raob => decode_raob(&nc, &source, &mut builder)?,
            Kind::Mesonet => decode_mesonet(&nc, &source, &mut builder, o.thin_minutes)?,
        }
        let probe = match kind {
            Kind::Raob => "staLat",
            _ => "latitude",
        };
        inputs.push(json!({
            "path": rw_obs::absolute_uri(path),
            "sha256": sha,
            "revision": revision_of(&source.revision_sha),
            "bytes": raw.len(),
            "records": record_count(&nc, probe)?,
            "rows_written": builder.writer.len() - before,
            "last_modified": published.map(seam_time),
            "fetched_at": received.map(seam_time),
        }));
    }
    let (rows, table_sha, bytes) = builder.writer.write(&out)?;
    let (t_type, w_type) = kind.report_types();
    let record = json!({
        "schema": TABLE_RECORD_SCHEMA,
        "status": if rows > 0 { "READY" } else { "EMPTY" },
        "kind": kind.name(),
        "table_schema": TABLE_SCHEMA,
        "source": kind.source(),
        "window": {"start": o.start.map(seam_time), "end": o.end.map(seam_time)},
        "bbox_s_n_w_e": o.bbox,
        "accept_unchecked": o.accept_unchecked,
        "thin_minutes": o.thin_minutes,
        "inputs": inputs,
        "table": {"path": rw_obs::absolute_uri(&out), "rows": rows, "sha256": table_sha, "bytes": bytes},
        "rows_by_variable": builder.rows_by_variable,
        "rows_by_hour": builder.rows_by_hour,
        "counts": builder.counts,
        "dropped_by_qc_summary": builder.dropped_by_qc,
        "errors": {
            "table": error_table.name,
            "table_sha256": error_table.sha256,
            "report_type_temperature_humidity": t_type,
            "report_type_wind": w_type,
            "dewpoint": "sigma_Td = (0.1*e_q/RH)/(L/(Rv*Td^2)), RH from Bolton (1980) es, L=2.5e6, Rv=461.5, clamped to [1, 10] K",
            "surface_pressure": "table hPa column times 100",
        },
        "qc": "MADIS QC summary: V S C G K kept; Z kept only with --accept-unchecked; every other character dropped and counted",
        "notes": match kind {
            Kind::Acars => "level_pa is the ISA pressure of the reported pressure altitude; elevation_m is that altitude",
            Kind::Raob => "every level is dated to the release time (relTime, else synTime): the MADIS file carries no per-level elapsed time; levels without a pressure are not written",
            Kind::Mesonet => "wind is platform_wind: mesonet anemometer heights vary and the archive does not state them",
        },
    });
    let text = serde_json::to_string_pretty(&record)? + "\n";
    let record_path = out.with_extension("json");
    std::fs::write(&record_path, &text).map_err(|e| err(format!("cannot write {}: {e}", record_path.display())))?;
    Ok(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn isa_pressure_inverts_isa_altitude() {
        for p in [101_325.0, 85_000.0, 50_000.0, 30_000.0, 22_632.06, 20_000.0, 10_000.0, 5_000.0] {
            let z = isa_altitude_m(p);
            assert!((isa_pressure_pa(z) - p).abs() < 1e-6 * p, "{p} -> {z} -> {}", isa_pressure_pa(z));
        }
        assert!((isa_pressure_pa(0.0) - 101_325.0).abs() < 1e-9);
        assert!((isa_pressure_pa(11_000.0) - 22_632.06).abs() < 1.0);
    }

    #[test]
    fn the_qc_screen_keeps_passes_and_names_failures() {
        for c in [b'V', b'S', b'C', b'G', b'K'] {
            assert!(qc_screen(c, false).is_ok());
        }
        assert_eq!(qc_screen(b'Z', false), Err('Z'));
        assert!(qc_screen(b'Z', true).is_ok());
        assert_eq!(qc_screen(b'X', true), Err('X'));
        assert_eq!(qc_screen(b'Q', false), Err('Q'));
        assert_eq!(qc_screen(b'B', false), Err('B'));
        assert_eq!(qc_screen(b'k', false), Err('k'));
        assert_eq!(qc_screen(0, false), Err('0'));
    }

    #[test]
    fn the_dewpoint_error_follows_the_table_and_its_clamp() {
        // saturated air at 290 K: sigma_RH 0.1 -> about 1.55 K
        let e = dewpoint_error_k(1.0, 290.0, 290.0);
        let expected = 0.1 / (LATENT_HEAT / (RV * 290.0 * 290.0));
        assert!((e - expected).abs() < 1e-9 && (e - 1.5525).abs() < 0.001, "{e}");
        // drier air has a larger dewpoint error for the same RH error
        assert!(dewpoint_error_k(1.0, 300.0, 280.0) > e);
        // the floor and the cap
        assert_eq!(dewpoint_error_k(0.1, 290.0, 290.0), DEWPOINT_ERROR_FLOOR_K);
        assert_eq!(dewpoint_error_k(5.0, 300.0, 230.0), DEWPOINT_ERROR_CAP_K);
        // Bolton at 0 C is 6.112 hPa
        assert!((bolton_es_hpa(273.15) - 6.112).abs() < 1e-12);
    }

    #[test]
    fn the_hour_urls_follow_the_archive_layout() {
        let t = Utc.with_ymd_and_hms(2026, 10, 1, 18, 37, 0).unwrap();
        let (url, name) = hour_url(Kind::Mesonet, hour_floor(t));
        assert_eq!(name, "20261001_1800.gz");
        assert_eq!(url, format!("{ARCHIVE}/2026/10/01/LDAD/mesonet/netCDF/20261001_1800.gz"));
        let (url, _) = hour_url(Kind::Acars, hour_floor(t));
        assert!(url.ends_with("/2026/10/01/point/acars/netcdf/20261001_1800.gz"));
        // Real time: the same sub-path and name without the date directories,
        // asked first by auto, alone by realtime, never by archive.
        assert_eq!(
            realtime_url(Kind::Raob, hour_floor(t)),
            format!("{REALTIME}/point/raob/netcdf/20261001_1800.gz")
        );
        let (auto, name) = hour_candidates(Kind::Mesonet, hour_floor(t), Feed::Auto);
        assert_eq!(name, "20261001_1800.gz");
        assert_eq!(auto[0], (format!("{REALTIME}/LDAD/mesonet/netCDF/20261001_1800.gz"), Feed::Realtime));
        assert_eq!(auto[1].1, Feed::Archive);
        assert_eq!(hour_candidates(Kind::Acars, t, Feed::Realtime).0.len(), 1);
        assert_eq!(hour_candidates(Kind::Acars, t, Feed::Archive).0[0].1, Feed::Archive);
        let a = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert_eq!(Options::parse(&a(&["--feed", "realtime"])).unwrap().feed, Feed::Realtime);
        assert_eq!(Options::parse(&a(&[])).unwrap().feed, Feed::Auto);
        assert!(Options::parse(&a(&["--feed", "ldm"])).is_err());
    }

    #[test]
    fn a_release_time_on_the_synoptic_day_is_moved_to_the_day_before() {
        let at = |d, h, m| Utc.with_ymd_and_hms(2026, 10, d, h, m, 0).unwrap().timestamp() as f64;
        // 72249 in the 2026-10-14 00 UTC file: relTime 10-14 23:04, synTime 10-14 00:00.
        let (t, why) = release_instant(at(14, 23, 4), at(14, 0, 0));
        assert_eq!(t, Some(Utc.with_ymd_and_hms(2026, 10, 13, 23, 4, 0).unwrap()));
        assert_eq!(why, Some("reports_release_time_moved_to_the_synoptic_day"));
        // A release time already beside its synoptic time is kept as it is.
        assert_eq!(release_instant(at(14, 11, 2), at(14, 12, 0)), (Some(Utc.with_ymd_and_hms(2026, 10, 14, 11, 2, 0).unwrap()), None));
        // 74794's placeholder (2012-01-01) is not believed.
        let (t, why) = release_instant(1_325_376_000.0, at(14, 0, 0));
        assert_eq!(t, Some(Utc.with_ymd_and_hms(2026, 10, 14, 0, 0, 0).unwrap()));
        assert_eq!(why, Some("reports_release_time_implausible_dated_to_synoptic_time"));
        // No release time: the synoptic time, counted.
        assert_eq!(release_instant(-1e38, at(14, 0, 0)).1, Some("reports_dated_to_synoptic_time"));
    }

    #[test]
    fn the_regional_table_carries_the_platform_types() {
        let t = ErrorTable::default_table();
        for ty in [TYPES_ACARS.0, TYPES_ACARS.1, TYPES_RAOB.0, TYPES_RAOB.1, TYPES_MESONET.0, TYPES_MESONET.1] {
            assert!(t.has_type(ty), "type {ty}");
        }
        assert!(t.error(TYPES_ACARS.0, Some(300.0), Column::Temperature) < TABLE_FILL_THRESHOLD);
        assert!(t.error(TYPES_ACARS.1, Some(300.0), Column::Wind) < TABLE_FILL_THRESHOLD);
    }

    #[test]
    fn options_parse_and_refuse() {
        let a = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let o = Options::parse(&a(&["--kind", "acars", "--files", "a.gz", "b.gz", "--out", "o.csv", "--bbox", "20,55,-130,-60"])).unwrap();
        assert_eq!(o.files.len(), 2);
        assert_eq!(o.bbox, Some([20.0, 55.0, -130.0, -60.0]));
        assert!(Options::parse(&a(&["--kind", "sat"])).is_err());
        assert!(Options::parse(&a(&["--bbox", "55,20,-130,-60"])).is_err());
        assert!(Options::parse(&a(&["--thin-minutes", "0"])).is_err());
    }
}
