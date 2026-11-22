//! `rw_gnssro` -- the GNSS radio-occultation front door.
//!
//! Two public routes to the same neutral table (`gpuwm-obs.table.v2`):
//! `refractivity_n` rows at the tangent-point positions, anchored in
//! HEIGHT (`elevation_m` is the retrieval's altitude above mean sea level)
//! with the retrieval's dry pressure carried in `level_pa` for the
//! analysis's column-span gates and the ln p localisation, source
//! `gnss-ro`, station id `<receiver>-<transmitter>`, valid at the
//! occultation's start time.
//!
//! **The CDAAC near-real-time route** (`cdaac-fetch`, `table --tarballs`):
//! UCAR's COSMIC Data Analysis and Archive Center publishes one tarball
//! per mission and day of `atmPrf` occultation files (classic NetCDF, one
//! per occultation) under
//! `https://data.cosmic.ucar.edu/gnss-ro/<mission>/nrt/level2/<YYYY>/<DDD>/atmPrf_nrt_<YYYY>_<DDD>.tar.gz`,
//! anonymously (measured 2026-09-06: the directory and the tarballs answer
//! HTTP 200 with no credential; the earlier note that the feed needs an
//! account was wrong and is retired).  Measured 2026-09-06: COSMIC-2 day
//! 244 (2026-09-01) is 6,227 occultations in 2.23 GB gzipped, published at
//! 04:53 UTC the next day; PAZ and KOMPSAT-5 publish the same layout at 46
//! and 9 MB; day 248's tarball appeared at 04:42 UTC on day 249, so the
//! route is 4 h 42 min to 4 h 53 min behind the END of the day it covers
//! and up to 29 h behind the day's first occultation: the delayed-replay
//! class, never the fast hourly one.  The tarball is streamed (gzip, then
//! the tar members one at a time through `rw_obs::tar`), never expanded
//! onto a disk, and each member is decoded in memory.
//!
//! `atmPrf` layout, verified 2026-09-06 against
//! `atmPrf_C2E1.2026.244.00.07.G09_0001.0001_nc` (classic NetCDF): the
//! dimension `MSL_alt` (about 3,800 levels); `MSL_alt` (km above mean sea
//! level), `Lat`, `Lon` (the perigee point per level), `Ref` (N-units),
//! `Pres` (dry pressure, mb), `Temp` (dry temperature, C), the bending
//! angles and impact heights; fill -999; global attributes `year`, `month`,
//! `day`, `hour`, `minute`, `second` (the occultation start), `fileStamp`
//! (`C2E1.2026.244.00.07.G09`: receiver, date, transmitter) and `bad`
//! (the string `"0"` for a retrieval that passed CDAAC's own checks; any
//! other value refuses the profile by name).
//!
//! **The AWS archive route** (`list`, `fetch`, `table --files`): the AWS
//! Open Data archive `s3://gnss-ro-data/` (UCAR COSMIC DAAC, ROM SAF and
//! JPL contributions, anonymous access): Level 2a retrievals, one NetCDF-4
//! file per occultation under
//! `contributed/v2.0/gnssro_<mission>_<center>_l2a/YYYY/MM/DD/`, layout
//! verified 2026-09-06 against
//! `gnssro_cosmic2_ucar_l2a_0001.0001_cosmic2e1-E03-202507290058.nc4`: root
//! `time` (seconds since 1980-01-06, the GPS epoch, read as a plain offset),
//! group `post_Abel` with `altitude` (m), `latitude`, `longitude`,
//! `refractivity`, `dry_pressure` (Pa), `superrefraction_impact_height`;
//! fill -9.99e20.  The bucket's collections end at 2025-07-29 (measured
//! 2026-09-06), so this route is retrospective; a window it does not
//! cover is reported EMPTY with the latest day the bucket holds.
//!
//! Vertical thinning (both routes): one row per `--level-step-m` (default
//! 200 m) between `--min-altitude-m` (default 0) and `--max-altitude-m`
//! (default 30,000), the level nearest each target height; the
//! refractivity below about 2 km can carry a negative bias from
//! super-refraction over the marine boundary layer (the AWS file states
//! `superrefraction_impact_height` when it saw one; rows below it are
//! dropped and counted).  Observation error: `rw_obs::table::
//! refractivity_error_fraction` of N by tangent height and latitude (the
//! rule is in the table record).
//!
//! ```text
//! rw_gnssro cdaac-fetch --start DATE --end DATE --cache DIR [--missions cosmic2,paz,kompsat5]
//! rw_gnssro table --tarballs a.tar.gz,b.tar.gz --out FILE.csv [--start T --end T]
//!                 [--level-step-m 200] [--fetch-record FILE]
//! rw_gnssro list  --start DATE --end DATE [--mission cosmic2] [--center ucar]
//! rw_gnssro fetch --start DATE --end DATE --cache DIR [--limit N]
//! rw_gnssro table --files a.nc4,b.nc4 --out FILE.csv [--level-step-m 200] [--fetch-record FILE]
//! rw_gnssro verify --file FILE.csv --record FILE.json
//! ```

use std::collections::BTreeMap;
use std::error::Error;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use chrono::{DateTime, Datelike, Duration, NaiveDate, TimeZone, Utc};
use serde::{Deserialize, Serialize};

use rw_nexrad::s3::{
    build_agent, download_object, list_s3, parse_http_date, parse_s3_timestamp, parse_time, ListRequest,
};
use rw_obs::seam::seam_time;
use rw_obs::table::{
    refractivity_error_fraction, RowProvenance, TableRow, TableWriter, ERROR_REFRACTIVITY_RULE,
    GROSS_REFRACTIVITY_N, MEAS_RO_REFRACTIVITY_TANGENT, TABLE_SCHEMA, VAR_REFRACTIVITY,
};
use rw_obs::tar::tar_members;
use rw_obs::{err, hex_sha256};
use rw_sat::netcdf::open_goes_netcdf_lossy;

const VERSION: &str = env!("CARGO_PKG_VERSION");

pub static GPUWM_BRIDGE_SOURCE_REV_STAMP: &str =
    concat!("GPUWM_BRIDGE_SOURCE_REV=", env!("GPUWM_BRIDGE_SOURCE_REV"));

const BUCKET: &str = "gnss-ro-data";
const DEFAULT_MISSION: &str = "cosmic2";
const DEFAULT_CENTER: &str = "ucar";
/// The row's `source` names the ROUTE the occultation came by, because
/// the two routes are two streams of different latency class (the AWS
/// archive is retrospective, the CDAAC tarballs a delayed replay) and the
/// analysis receipt's stream roster reads the class from the source name:
/// `table --files` writes `gnss-ro`, `table --tarballs` writes `cdaac-ro`,
/// and one table never mixes them.
const SOURCE_AWS: &str = "gnss-ro";
const SOURCE_CDAAC: &str = "cdaac-ro";
const DEFAULT_LEVEL_STEP_M: f64 = 200.0;
const DEFAULT_MIN_ALTITUDE_M: f64 = 0.0;
const DEFAULT_MAX_ALTITUDE_M: f64 = 30_000.0;

/// The CDAAC data portal and the missions with a near-real-time level-2
/// directory there (measured 2026-09-06 for day 244 of 2026: cosmic2, paz
/// and kompsat5 answered 200; metopb, metopc, spire, planetiq, geoopt,
/// tsx and sentinel6a answered 404 under `nrt/level2`).
const CDAAC_BASE: &str = "https://data.cosmic.ucar.edu/gnss-ro";
const CDAAC_DEFAULT_MISSIONS: &[&str] = &["cosmic2", "paz", "kompsat5"];
const CDAAC_PRODUCT: &str = "atmPrf";
/// A day of COSMIC-2 is 2.3 GB; four is more than any mission publishes.
const CDAAC_MAX_TARBALL_BYTES: u64 = 4 * 1024 * 1024 * 1024;
/// One day's tarball at a few megabytes a second: hours, not the shared
/// agent's ten minutes.
const CDAAC_BODY_TIMEOUT_S: u64 = 4 * 3600;

const LIST_SCHEMA: &str = "gpuwm-obs.gnssro-list.v1";
const FETCH_SCHEMA: &str = "gpuwm-obs.gnssro-fetch.v1";
const CDAAC_FETCH_SCHEMA: &str = "gpuwm-obs.gnssro-cdaac-fetch.v1";
const TABLE_RECORD_SCHEMA: &str = "gpuwm-obs.gnssro-table.v1";
const VERIFY_SCHEMA: &str = "gpuwm-obs.gnssro-verify.v1";
const ABI_MARKER: &str = "gpuwm-obs.gnssro-fetch.v1\tgpuwm-obs.gnssro-cdaac-fetch.v1\tgpuwm-obs.gnssro-table.v1\t\
gpuwm-obs.table.v2\trefractivity_n\theight_anchored\tdry_pressure\tcdaac_tarballs";

const USAGE: &str = "\
usage: rw_gnssro <cdaac-fetch|list|fetch|table|verify> [OPTIONS]
       rw_gnssro --version | --help | --abi

  cdaac-fetch  download the CDAAC near-real-time daily atmPrf tarballs of a day range
  table        decode occultation files (--files, the AWS layout) or the CDAAC daily
               tarballs (--tarballs) into a `gpuwm-obs.table.v2` CSV
  list         report the AWS archive files a (mission, center, day range) resolves to
  fetch        download them into the cache with a sha256 per file
  verify       re-hash a table against the record written beside it

cdaac-fetch options
  --start DATE          first day, 2026-09-01 or 2026-09-01T00:00:00Z
  --end DATE            last day (inclusive)
  --cache DIR           where the tarballs land (<cache>/cdaac/<mission>/)
  --missions LIST       comma-separated (default cosmic2,paz,kompsat5)

acquisition options (list, fetch: the AWS archive)
  --mission NAME        cosmic2 (default), metop, planetiq, paz, kompsat5, ...
  --center NAME         ucar (default), romsaf, jpl
  --start DATE / --end DATE / --cache DIR / --limit N

table options
  --tarballs LIST       CDAAC daily tarballs, comma-separated (or repeat --tarballs)
  --files LIST          AWS occultation files, comma-separated (or repeat --files)
  --start T --end T     with --tarballs: keep the occultations inside [start, end]
  --out FILE.csv        the table; a .json record is written beside it
  --level-step-m M      vertical spacing of the rows (default 200)
  --min-altitude-m M    lowest row (default 0)
  --max-altitude-m M    highest row (default 30000)
  --fetch-record FILE   the fetch record, so each row carries the object's
                        Last-Modified (published) and fetched_at (received)

verify options
  --file FILE.csv --record FILE.json
";

fn main() -> ExitCode {
    let _ = std::hint::black_box(GPUWM_BRIDGE_SOURCE_REV_STAMP);
    rw_obs::cli::main("rw_gnssro", run)
}

fn run(args: &[String]) -> Result<String, Box<dyn Error>> {
    if let Some(output) = rw_obs::cli::information(args, "rw_gnssro", VERSION, USAGE, ABI_MARKER) {
        return Ok(output);
    }
    let first = &args[0];
    let options = Options::parse(&args[1..])?;
    match first.as_str() {
        "cdaac-fetch" => cmd_cdaac_fetch(&options),
        "list" => cmd_list(&options),
        "fetch" => cmd_fetch(&options),
        "table" => cmd_table(&options),
        "verify" => cmd_verify(&options),
        other => Err(err(format!("unknown subcommand {other:?}\n\n{USAGE}"))),
    }
}

#[derive(Debug, Default)]
struct Options {
    mission: Option<String>,
    center: Option<String>,
    missions: Option<Vec<String>>,
    start: Option<String>,
    end: Option<String>,
    cache: Option<PathBuf>,
    limit: Option<usize>,
    files: Vec<PathBuf>,
    tarballs: Vec<PathBuf>,
    out: Option<PathBuf>,
    level_step_m: Option<f64>,
    min_altitude_m: Option<f64>,
    max_altitude_m: Option<f64>,
    fetch_record: Option<PathBuf>,
    file: Option<PathBuf>,
    record: Option<PathBuf>,
}

fn token(value: String, flag: &str) -> Result<String, Box<dyn Error>> {
    let v = value.trim().to_ascii_lowercase();
    if v.is_empty() || !v.bytes().all(|b| b.is_ascii_alphanumeric()) {
        return Err(err(format!("{flag} expects letters and digits, got {value:?}; it is concatenated into a URL")));
    }
    Ok(v)
}

impl Options {
    fn parse(args: &[String]) -> Result<Self, Box<dyn Error>> {
        let mut options = Options::default();
        let mut index = 0;
        while index < args.len() {
            let flag = args[index].as_str();
            let mut value = || rw_obs::cli::value(args, &mut index, flag);
            let number = |raw: String, flag: &str| -> Result<f64, Box<dyn Error>> {
                let v: f64 = raw.parse().map_err(|_| err(format!("{flag} expects a number, got {raw:?}")))?;
                if !v.is_finite() {
                    return Err(err(format!("{flag} must be finite")));
                }
                Ok(v)
            };
            match flag {
                "--mission" => options.mission = Some(token(value()?, "--mission")?),
                "--center" => options.center = Some(token(value()?, "--center")?),
                "--missions" => {
                    let raw = value()?;
                    let list = raw
                        .split(',')
                        .map(|t| token(t.to_string(), "--missions"))
                        .collect::<Result<Vec<_>, _>>()?;
                    if list.is_empty() {
                        return Err(err("--missions named no mission"));
                    }
                    options.missions = Some(list);
                }
                "--start" => options.start = Some(value()?),
                "--end" => options.end = Some(value()?),
                "--cache" => options.cache = Some(PathBuf::from(value()?)),
                "--limit" => {
                    let raw = value()?;
                    options.limit = Some(raw.parse().map_err(|_| err(format!("--limit expects a count, got {raw:?}")))?);
                }
                "--files" => {
                    let raw = value()?;
                    options.files.extend(raw.split(',').map(str::trim).filter(|t| !t.is_empty()).map(PathBuf::from));
                }
                "--tarballs" => {
                    let raw = value()?;
                    options.tarballs.extend(raw.split(',').map(str::trim).filter(|t| !t.is_empty()).map(PathBuf::from));
                }
                "--out" => options.out = Some(PathBuf::from(value()?)),
                "--level-step-m" => {
                    let v = number(value()?, "--level-step-m")?;
                    if v <= 0.0 {
                        return Err(err("--level-step-m must be positive"));
                    }
                    options.level_step_m = Some(v);
                }
                "--min-altitude-m" => options.min_altitude_m = Some(number(value()?, "--min-altitude-m")?),
                "--max-altitude-m" => options.max_altitude_m = Some(number(value()?, "--max-altitude-m")?),
                "--fetch-record" => options.fetch_record = Some(PathBuf::from(value()?)),
                "--file" => options.file = Some(PathBuf::from(value()?)),
                "--record" => options.record = Some(PathBuf::from(value()?)),
                other => return Err(err(format!("unknown option {other:?}\n\n{USAGE}"))),
            }
            index += 1;
        }
        Ok(options)
    }

    fn collection(&self) -> String {
        format!(
            "contributed/v2.0/gnssro_{}_{}_l2a/",
            self.mission.as_deref().unwrap_or(DEFAULT_MISSION),
            self.center.as_deref().unwrap_or(DEFAULT_CENTER)
        )
    }

    fn days(&self) -> Result<Vec<NaiveDate>, Box<dyn Error>> {
        let parse_day = |raw: &str| -> Result<NaiveDate, Box<dyn Error>> {
            if let Ok(d) = NaiveDate::parse_from_str(raw.trim(), "%Y-%m-%d") {
                return Ok(d);
            }
            Ok(parse_time(raw)?.date_naive())
        };
        let start = parse_day(self.start.as_deref().ok_or_else(|| err("--start is required"))?)?;
        let end = parse_day(self.end.as_deref().ok_or_else(|| err("--end is required"))?)?;
        if end < start {
            return Err(err(format!("--end {end} precedes --start {start}")));
        }
        let mut days = Vec::new();
        let mut d = start;
        while d <= end {
            days.push(d);
            d += Duration::days(1);
        }
        Ok(days)
    }

    /// The instant window of `table --tarballs` (both, one or neither bound).
    fn window(&self) -> Result<(Option<DateTime<Utc>>, Option<DateTime<Utc>>), Box<dyn Error>> {
        let parse = |raw: Option<&str>, flag: &str| -> Result<Option<DateTime<Utc>>, Box<dyn Error>> {
            match raw {
                None => Ok(None),
                Some(text) => {
                    if let Ok(d) = NaiveDate::parse_from_str(text.trim(), "%Y-%m-%d") {
                        return Ok(Some(Utc.from_utc_datetime(&d.and_hms_opt(0, 0, 0).unwrap())));
                    }
                    parse_time(text).map(Some).map_err(|e| err(format!("{flag}: {e}")))
                }
            }
        };
        let start = parse(self.start.as_deref(), "--start")?;
        let end = parse(self.end.as_deref(), "--end")?;
        if let (Some(a), Some(b)) = (start, end) {
            if b < a {
                return Err(err(format!("--end {} precedes --start {}", seam_time(b), seam_time(a))));
            }
        }
        Ok((start, end))
    }

    fn cache(&self) -> PathBuf {
        self.cache.clone().unwrap_or_else(|| PathBuf::from(".rw-gnssro-cache"))
    }

    fn missions(&self) -> Vec<String> {
        self.missions
            .clone()
            .unwrap_or_else(|| CDAAC_DEFAULT_MISSIONS.iter().map(|m| m.to_string()).collect())
    }
}

// ------------------------------------------------------------- CDAAC fetch

fn cdaac_url(mission: &str, day: NaiveDate) -> String {
    format!(
        "{CDAAC_BASE}/{mission}/nrt/level2/{}/{:03}/{CDAAC_PRODUCT}_nrt_{}_{:03}.tar.gz",
        day.year(),
        day.ordinal(),
        day.year(),
        day.ordinal()
    )
}

/// The shared agent's TLS stack with a body timeout sized for a day's
/// tarball; `build_agent` runs first so the crypto provider is installed
/// exactly once.
fn cdaac_agent() -> ureq::Agent {
    let _ = build_agent();
    let crypto = std::sync::Arc::new(rustls_rustcrypto::provider());
    ureq::Agent::config_builder()
        .timeout_resolve(Some(std::time::Duration::from_secs(10)))
        .timeout_connect(Some(std::time::Duration::from_secs(10)))
        .timeout_send_request(Some(std::time::Duration::from_secs(30)))
        .timeout_recv_response(Some(std::time::Duration::from_secs(CDAAC_BODY_TIMEOUT_S)))
        .timeout_recv_body(Some(std::time::Duration::from_secs(CDAAC_BODY_TIMEOUT_S)))
        .tls_config(
            ureq::tls::TlsConfig::builder()
                .provider(ureq::tls::TlsProvider::Rustls)
                .root_certs(ureq::tls::RootCerts::WebPki)
                .unversioned_rustls_crypto_provider(crypto)
                .build(),
        )
        .build()
        .new_agent()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CdaacFile {
    mission: String,
    day: String,
    url: String,
    path: String,
    bytes: u64,
    sha256: String,
    /// The server's Last-Modified: when the day's tarball was published.
    last_modified: Option<String>,
    fetched_at: Option<String>,
    cache_hit: bool,
    /// Publication minus the end of the day the tarball covers.
    latency_behind_day_end_s: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CdaacFetchRecord {
    schema: String,
    status: String,
    #[serde(default)]
    files: Vec<CdaacFile>,
}

/// `path` with `suffix` appended to its file name (`Path::with_extension`
/// would replace the `.gz` of `.tar.gz` and lose the `.tar`).
fn sibling(path: &Path, suffix: &str) -> PathBuf {
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("object");
    path.with_file_name(format!("{name}{suffix}"))
}

/// Download `url` to `target` through a sha256, streaming: the body is
/// never held whole (a day is two gigabytes).
fn stream_download(agent: &ureq::Agent, url: &str, target: &Path) -> Result<(u64, String), Box<dyn Error>> {
    use sha2::Digest;

    let mut response = rw_http::get(agent, url).map_err(|e| err(format!("GET {url}: {e}")))?;
    let status = response.status().as_u16();
    if !(200..300).contains(&status) {
        return Err(err(format!("GET {url} answered HTTP {status}")));
    }
    if let Some(parent) = target.parent() {
        std::fs::create_dir_all(parent).map_err(|e| err(format!("cannot create {}: {e}", parent.display())))?;
    }
    let partial = sibling(target, ".partial");
    let mut file = std::fs::File::create(&partial).map_err(|e| err(format!("cannot write {}: {e}", partial.display())))?;
    let mut reader = response.body_mut().with_config().limit(CDAAC_MAX_TARBALL_BYTES).reader();
    let mut hasher = sha2::Sha256::new();
    let mut buffer = vec![0u8; 1 << 20];
    let mut total: u64 = 0;
    loop {
        let n = reader.read(&mut buffer).map_err(|e| err(format!("reading {url}: {e}")))?;
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
        file.write_all(&buffer[..n]).map_err(|e| err(format!("writing {}: {e}", partial.display())))?;
        total += n as u64;
    }
    file.flush()?;
    drop(file);
    std::fs::rename(&partial, target).map_err(|e| err(format!("cannot place {}: {e}", target.display())))?;
    Ok((total, format!("{:x}", hasher.finalize())))
}

fn cmd_cdaac_fetch(options: &Options) -> Result<String, Box<dyn Error>> {
    let days = options.days()?;
    let cache = options.cache();
    let agent = cdaac_agent();
    let started = std::time::Instant::now();
    let mut files = Vec::new();
    let mut missing: Vec<String> = Vec::new();
    let mut hits = 0usize;
    for mission in options.missions() {
        for day in &days {
            let url = cdaac_url(&mission, *day);
            let head = rw_http::head(&agent, &url);
            let response = match head {
                Ok(r) if (200..300).contains(&r.status().as_u16()) => r,
                // ureq 3 hands a 4xx back as an error; a 404 is a day the
                // portal has not published (or never will for this mission).
                Ok(r) if r.status().as_u16() == 404 => {
                    missing.push(url.clone());
                    continue;
                }
                Err(ureq::Error::StatusCode(404)) => {
                    missing.push(url.clone());
                    continue;
                }
                Ok(r) => return Err(err(format!("HEAD {url} answered HTTP {}", r.status().as_u16()))),
                Err(e) => return Err(err(format!("HEAD {url}: {e}"))),
            };
            let stated: Option<u64> = response
                .headers()
                .get("content-length")
                .and_then(|v| v.to_str().ok())
                .and_then(|v| v.trim().parse().ok());
            let last_modified = response
                .headers()
                .get("last-modified")
                .and_then(|v| v.to_str().ok())
                .and_then(parse_http_date);
            let target = cache
                .join("cdaac")
                .join(&mission)
                .join(format!("{CDAAC_PRODUCT}_nrt_{}_{:03}.tar.gz", day.year(), day.ordinal()));
            let sidecar = sibling(&target, ".sha256");
            let mut cache_hit = false;
            let (bytes, sha) = match (target.metadata().ok(), std::fs::read_to_string(&sidecar).ok(), stated) {
                (Some(meta), Some(text), Some(size)) if meta.len() == size && text.trim().len() == 64 => {
                    cache_hit = true;
                    hits += 1;
                    (meta.len(), text.trim().to_string())
                }
                _ => {
                    let (n, sha) = stream_download(&agent, &url, &target)?;
                    if let Some(size) = stated {
                        if n != size {
                            return Err(err(format!("{url}: downloaded {n} bytes where the server stated {size}")));
                        }
                    }
                    std::fs::write(&sidecar, format!("{sha}\n"))?;
                    (n, sha)
                }
            };
            let day_end = Utc.from_utc_datetime(&(*day + Duration::days(1)).and_hms_opt(0, 0, 0).unwrap());
            files.push(CdaacFile {
                mission: mission.clone(),
                day: day.format("%Y-%m-%d").to_string(),
                url,
                path: rw_obs::absolute_uri(&target),
                bytes,
                sha256: sha,
                last_modified: last_modified.map(seam_time),
                fetched_at: Some(seam_time(Utc::now())),
                cache_hit,
                latency_behind_day_end_s: last_modified.map(|t| (t - day_end).num_seconds()),
            });
        }
    }
    // Nothing published for the window is a state of the source, not a
    // failure of this door: the record says EMPTY with every URL that
    // answered 404 (the directory holds the last few days only and a day
    // appears about five hours after it ends), so a cycle asking for a day
    // not yet published carries the hour by name instead of dying.
    let status = if files.is_empty() { "EMPTY" } else { "READY" };
    #[derive(Serialize)]
    struct Record {
        schema: &'static str,
        status: &'static str,
        base: &'static str,
        product: &'static str,
        missions: Vec<String>,
        cache_dir: String,
        files: Vec<CdaacFile>,
        missing: Vec<String>,
        total_bytes: u64,
        cache_hits: usize,
        latency_behind_real_time_s: Option<i64>,
        latency_basis: &'static str,
        wall_s: f64,
    }
    Ok(format!(
        "{}\n",
        serde_json::to_string_pretty(&Record {
            schema: CDAAC_FETCH_SCHEMA,
            status,
            base: CDAAC_BASE,
            product: CDAAC_PRODUCT,
            missions: options.missions(),
            cache_dir: rw_obs::absolute_uri(&cache),
            total_bytes: files.iter().map(|f| f.bytes).sum(),
            cache_hits: hits,
            latency_behind_real_time_s: files.iter().filter_map(|f| f.latency_behind_day_end_s).max(),
            latency_basis: "the daily tarball's Last-Modified minus the end of the day it covers (its first \
                            occultation waits a day longer)",
            files,
            missing,
            wall_s: started.elapsed().as_secs_f64(),
        })?
    ))
}

// ------------------------------------------------------------------ list

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Listed {
    key: String,
    bytes: u64,
    last_modified: Option<String>,
}

/// The latest day prefix the collection holds, walked year, month, day
/// with delimited listings, so an empty window can say what the bucket
/// does have.
fn latest_day(agent: &ureq::Agent, collection: &str) -> Result<Option<String>, Box<dyn Error>> {
    let mut prefix = collection.to_string();
    for _ in 0..3 {
        let listing = list_s3(agent, ListRequest::new(BUCKET, &prefix).delimiter("/"))?;
        let mut children: Vec<String> = listing
            .common_prefixes
            .into_iter()
            .filter(|p| p != &prefix)
            .collect();
        children.sort();
        // Some listings answer with the queried prefix itself as a
        // common prefix; a child must be longer than the query.
        children.retain(|c| c.len() > prefix.len());
        let Some(last) = children.pop() else {
            return Ok(None);
        };
        prefix = last;
    }
    Ok(Some(prefix))
}

fn list_files(options: &Options) -> Result<(Vec<Listed>, Option<String>), Box<dyn Error>> {
    let agent = build_agent();
    let collection = options.collection();
    let mut files = Vec::new();
    for day in options.days()? {
        let prefix = format!("{collection}{}", day.format("%Y/%m/%d/"));
        let listing = list_s3(&agent, ListRequest::new(BUCKET, &prefix))?;
        for object in listing.objects {
            files.push(Listed {
                key: object.key,
                bytes: object.size_bytes,
                last_modified: parse_s3_timestamp(&object.last_modified).map(seam_time),
            });
        }
    }
    files.sort_by(|a, b| a.key.cmp(&b.key));
    let latest = if files.is_empty() { latest_day(&agent, &collection)? } else { None };
    Ok((files, latest))
}

fn cmd_list(options: &Options) -> Result<String, Box<dyn Error>> {
    let (files, latest) = list_files(options)?;
    #[derive(Serialize)]
    struct Record {
        schema: &'static str,
        status: &'static str,
        bucket: &'static str,
        collection: String,
        files: usize,
        total_bytes: u64,
        /// When the window is empty: the latest day the collection holds,
        /// so the gap is a number and not a guess.
        latest_day_in_bucket: Option<String>,
        objects: Vec<Listed>,
    }
    Ok(format!(
        "{}\n",
        serde_json::to_string_pretty(&Record {
            schema: LIST_SCHEMA,
            status: if files.is_empty() { "EMPTY" } else { "READY" },
            bucket: BUCKET,
            collection: options.collection(),
            files: files.len(),
            total_bytes: files.iter().map(|f| f.bytes).sum(),
            latest_day_in_bucket: latest,
            objects: files,
        })?
    ))
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Fetched {
    #[serde(flatten)]
    listed: Listed,
    path: String,
    sha256: String,
    cache_hit: bool,
    #[serde(default)]
    fetched_at: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct FetchRecord {
    schema: String,
    status: String,
    #[serde(default)]
    files: Vec<Fetched>,
}

fn cmd_fetch(options: &Options) -> Result<String, Box<dyn Error>> {
    let (mut files, latest) = list_files(options)?;
    if files.is_empty() {
        return Err(err(format!(
            "no occultation files under s3://{BUCKET}/{} for the window; the collection's latest \
             day is {}.  The bucket updates monthly and holds nothing for 2026 (measured \
             2026-09-06); the CDAAC near-real-time route (cdaac-fetch) covers the recent days",
            options.collection(),
            latest.unwrap_or_else(|| "unknown".to_string())
        )));
    }
    if let Some(limit) = options.limit {
        files.truncate(limit);
    }
    let cache = options.cache();
    let agent = build_agent();
    let started = std::time::Instant::now();
    let mut fetched = Vec::with_capacity(files.len());
    let mut hits = 0usize;
    for listed in files {
        let object = rw_nexrad::s3::S3Object {
            key: listed.key.clone(),
            size_bytes: listed.bytes,
            last_modified: listed.last_modified.clone().unwrap_or_default(),
        };
        let got = download_object(&agent, BUCKET, &cache, &object, true)?;
        if got.cache_hit {
            hits += 1;
        }
        fetched.push(Fetched {
            listed,
            path: rw_obs::absolute_uri(&got.path),
            sha256: got.sha256,
            cache_hit: got.cache_hit,
            fetched_at: Some(seam_time(Utc::now())),
        });
    }
    #[derive(Serialize)]
    struct Record {
        schema: &'static str,
        status: &'static str,
        bucket: &'static str,
        collection: String,
        cache_dir: String,
        files: Vec<Fetched>,
        total_bytes: u64,
        cache_hits: usize,
        wall_s: f64,
    }
    Ok(format!(
        "{}\n",
        serde_json::to_string_pretty(&Record {
            schema: FETCH_SCHEMA,
            status: "READY",
            bucket: BUCKET,
            collection: options.collection(),
            cache_dir: rw_obs::absolute_uri(&cache),
            total_bytes: fetched.iter().map(|f| f.listed.bytes).sum(),
            cache_hits: hits,
            files: fetched,
            wall_s: started.elapsed().as_secs_f64(),
        })?
    ))
}

// ----------------------------------------------------------------- table

/// GPS epoch, the AWS file's time origin.
fn gps_epoch() -> DateTime<Utc> {
    Utc.with_ymd_and_hms(1980, 1, 6, 0, 0, 0).unwrap()
}

#[derive(Debug, Clone)]
pub struct Profile {
    pub station_id: String,
    pub time: DateTime<Utc>,
    pub altitude_m: Vec<f64>,
    pub latitude_deg: Vec<f64>,
    pub longitude_deg: Vec<f64>,
    pub refractivity: Vec<f64>,
    pub dry_pressure_pa: Vec<f64>,
    pub superrefraction_height_m: Option<f64>,
    pub levels_in_file: usize,
}

fn attr_string(file: &netcrust::File, name: &str) -> Option<String> {
    file.attribute(name).and_then(|a| a.as_string().map(str::to_string))
}

fn attr_f64(file: &netcrust::File, name: &str) -> Option<f64> {
    file.attribute(name).and_then(|a| a.as_f64())
}

/// One dataset of an HDF5 group as f64, whatever its stored type.  The
/// L2a files keep `altitude`, `latitude`, `longitude` and `quality` as
/// F32 and the rest as F64; the reader is asked for f64 first and f32
/// second and refuses anything else by name.
fn group_values(group: &hdf5_reader::group::Group, name: &str, subject: &str) -> Result<Vec<f64>, Box<dyn Error>> {
    let dataset = group.dataset(name).map_err(|e| {
        err(format!(
            "{subject}: {name} is missing from the post_Abel group ({e}); the L2a layout this reader              knows is root time and the post_Abel group with altitude, latitude, longitude,              refractivity and dry_pressure"
        ))
    })?;
    if let Ok(values) = dataset.read_array::<f64>() {
        return Ok(values.iter().copied().collect());
    }
    if let Ok(values) = dataset.read_array::<f32>() {
        return Ok(values.iter().map(|v| f64::from(*v)).collect());
    }
    Err(err(format!("{subject}: {name} is neither F64 nor F32; this reader takes no other type")))
}

fn read_profile(path: &Path) -> Result<Profile, Box<dyn Error>> {
    let subject = path.display().to_string();
    // Root attributes and the root `time` through the NetCDF facade; the
    // profile arrays live in the `post_Abel` HDF5 group, which the facade
    // does not walk, so they are read through the HDF5 reader it is built on.
    let file = open_goes_netcdf_lossy(path)?;
    let seconds = file
        .read_f64("time")
        .map_err(|e| err(format!("{subject}: root time is unreadable ({e})")))?
        .first()
        .copied()
        .ok_or_else(|| err(format!("{subject}: time is empty")))?;
    if !seconds.is_finite() || seconds <= 0.0 {
        return Err(err(format!("{subject}: time is fill")));
    }
    let instant = gps_epoch() + Duration::milliseconds((seconds * 1000.0).round() as i64);
    let h5 = hdf5_reader::Hdf5File::open(path).map_err(|e| err(format!("{subject}: HDF5 open failed: {e}")))?;
    let group = h5.group("post_Abel").map_err(|e| {
        err(format!(
            "{subject}: no post_Abel group ({e}); this is not a v2.0 L2a occultation file"
        ))
    })?;
    let altitude = group_values(&group, "altitude", &subject)?;
    let latitude = group_values(&group, "latitude", &subject)?;
    let longitude = group_values(&group, "longitude", &subject)?;
    let refractivity = group_values(&group, "refractivity", &subject)?;
    let dry_pressure = group_values(&group, "dry_pressure", &subject)?;
    let n = altitude.len();
    if latitude.len() != n || longitude.len() != n || refractivity.len() != n || dry_pressure.len() != n {
        return Err(err(format!(
            "{subject}: post_Abel arrays disagree on length ({n}, {}, {}, {}, {})",
            latitude.len(),
            longitude.len(),
            refractivity.len(),
            dry_pressure.len()
        )));
    }
    let superrefraction = group_values(&group, "superrefraction_impact_height", &subject)
        .ok()
        .and_then(|v| v.first().copied())
        .filter(|v| v.is_finite() && *v > 0.0 && *v < 1.0e19);
    let receiver = attr_string(&file, "receiver").unwrap_or_else(|| "rx".to_string());
    let transmitter = attr_string(&file, "transmitter").unwrap_or_else(|| "tx".to_string());
    Ok(Profile {
        station_id: format!("{receiver}-{transmitter}"),
        time: instant,
        altitude_m: altitude,
        latitude_deg: latitude,
        longitude_deg: longitude,
        refractivity,
        dry_pressure_pa: dry_pressure,
        superrefraction_height_m: superrefraction,
        levels_in_file: n,
    })
}

/// Why a CDAAC member was not a profile: counted by name.
#[derive(Debug)]
enum CdaacSkip {
    /// The retrieval failed CDAAC's own checks (`bad` other than "0").
    BadFlag(String),
    /// A member that is not an atmPrf occultation file.
    NotAProfile,
}

/// The receiver and transmitter of a CDAAC file name
/// (`atmPrf_C2E1.2026.244.00.07.G09_0001.0001_nc` -> `C2E1`, `G09`).
fn cdaac_station_id(name: &str, file_stamp: Option<&str>) -> String {
    let stem = name.rsplit('/').next().unwrap_or(name);
    let stamp = file_stamp
        .map(str::to_string)
        .or_else(|| stem.strip_prefix("atmPrf_").and_then(|s| s.split('_').next()).map(str::to_string))
        .unwrap_or_default();
    let mut parts = stamp.split('.');
    let receiver = parts.next().unwrap_or("rx");
    let transmitter = parts.last().unwrap_or("tx");
    format!("{receiver}-{transmitter}")
}

/// A CDAAC `atmPrf` occultation from its bytes: the perigee-point
/// positions per level, the refractivity and the dry pressure, the
/// occultation start from the calendar attributes.
fn read_cdaac_profile(name: &str, bytes: &[u8]) -> Result<Result<Profile, CdaacSkip>, Box<dyn Error>> {
    if !name.rsplit('/').next().unwrap_or(name).starts_with("atmPrf_") {
        return Ok(Err(CdaacSkip::NotAProfile));
    }
    let subject = name.to_string();
    let file = netcrust::File::from_bytes(bytes).map_err(|e| err(format!("{subject}: not a NetCDF file ({e})")))?;
    if let Some(bad) = attr_string(&file, "bad") {
        if bad.trim() != "0" {
            return Ok(Err(CdaacSkip::BadFlag(bad.trim().to_string())));
        }
    }
    let read = |variable: &str| -> Result<Vec<f64>, Box<dyn Error>> {
        file.read_f64(variable).map_err(|e| {
            err(format!(
                "{subject}: {variable} is unreadable ({e}); the atmPrf layout this reader knows is MSL_alt, \
                 Lat, Lon, Ref and Pres on the MSL_alt dimension"
            ))
        })
    };
    let altitude_km = read("MSL_alt")?;
    let latitude = read("Lat")?;
    let longitude = read("Lon")?;
    let refractivity = read("Ref")?;
    let pressure_mb = read("Pres")?;
    let n = altitude_km.len();
    if latitude.len() != n || longitude.len() != n || refractivity.len() != n || pressure_mb.len() != n {
        return Err(err(format!(
            "{subject}: the profile arrays disagree on length ({n}, {}, {}, {}, {})",
            latitude.len(),
            longitude.len(),
            refractivity.len(),
            pressure_mb.len()
        )));
    }
    let calendar = |attr: &str| -> Result<i64, Box<dyn Error>> {
        attr_f64(&file, attr)
            .filter(|v| v.is_finite())
            .map(|v| v.floor() as i64)
            .ok_or_else(|| err(format!("{subject}: the {attr} attribute is missing; the occultation has no start time")))
    };
    let (year, month, day, hour, minute) = (calendar("year")?, calendar("month")?, calendar("day")?, calendar("hour")?, calendar("minute")?);
    let second = attr_f64(&file, "second").filter(|v| v.is_finite()).unwrap_or(0.0);
    let date = NaiveDate::from_ymd_opt(year as i32, month as u32, day as u32)
        .and_then(|d| d.and_hms_opt(hour as u32, minute as u32, 0))
        .ok_or_else(|| err(format!("{subject}: the calendar attributes {year}-{month}-{day} {hour}:{minute} are not a date")))?;
    let instant = Utc.from_utc_datetime(&date) + Duration::milliseconds((second * 1000.0).round() as i64);
    // Fill is -999 in every array; a fill altitude or refractivity is dropped
    // by rows_for's fill test (values below -1e19 or non-finite) after the
    // conversion below marks it non-finite.
    let fill = |v: f64| if v <= -998.0 || !v.is_finite() { f64::NAN } else { v };
    Ok(Ok(Profile {
        station_id: cdaac_station_id(name, attr_string(&file, "fileStamp").as_deref()),
        time: instant,
        altitude_m: altitude_km.iter().map(|&v| fill(v) * 1000.0).collect(),
        latitude_deg: latitude.iter().map(|&v| fill(v)).collect(),
        longitude_deg: longitude.iter().map(|&v| fill(v)).collect(),
        refractivity: refractivity.iter().map(|&v| fill(v)).collect(),
        dry_pressure_pa: pressure_mb.iter().map(|&v| fill(v) * 100.0).collect(),
        superrefraction_height_m: None,
        levels_in_file: n,
    }))
}

#[derive(Debug, Default, Serialize, Clone)]
pub struct Counters {
    files_read: usize,
    tarballs_read: usize,
    tarball_members: usize,
    profiles_read: usize,
    profiles_bad_flag: usize,
    profiles_outside_window: usize,
    profiles_unreadable: usize,
    members_not_profiles: usize,
    levels_in_files: usize,
    levels_fill: usize,
    levels_out_of_range: usize,
    levels_below_superrefraction: usize,
    levels_outside_altitude_window: usize,
    target_heights_without_level: usize,
    rows_by_variable: BTreeMap<String, usize>,
    profiles_with_superrefraction: usize,
    profiles_by_station_prefix: BTreeMap<String, usize>,
    /// CDAAC's own `bad` attribute values of the refused profiles.
    bad_flags: BTreeMap<String, usize>,
}

fn refractivity_error(altitude_m: f64, latitude_deg: f64, value: f64) -> f64 {
    (refractivity_error_fraction(altitude_m, latitude_deg) * value).max(1.0e-3)
}

/// The rows of one profile: the level nearest each target height in
/// `[min, max]` at `step`, each target used once.
pub fn rows_for(
    profile: &Profile,
    source: &str,
    step_m: f64,
    min_m: f64,
    max_m: f64,
    provenance: &RowProvenance,
    counters: &mut Counters,
    writer: &mut TableWriter,
) {
    let provenance = provenance.measuring(MEAS_RO_REFRACTIVITY_TANGENT);
    counters.levels_in_files += profile.levels_in_file;
    if profile.superrefraction_height_m.is_some() {
        counters.profiles_with_superrefraction += 1;
    }
    let floor = profile.superrefraction_height_m.unwrap_or(f64::NEG_INFINITY);
    // Good levels, sorted by altitude.
    let mut good: Vec<usize> = Vec::new();
    for i in 0..profile.levels_in_file {
        let z = profile.altitude_m[i];
        let n = profile.refractivity[i];
        let (lat, lon) = (profile.latitude_deg[i], profile.longitude_deg[i]);
        let fill = |v: f64| !v.is_finite() || v < -1.0e19;
        if fill(z) || fill(n) || fill(lat) || fill(lon) {
            counters.levels_fill += 1;
            continue;
        }
        if !(GROSS_REFRACTIVITY_N.0 <= n && n <= GROSS_REFRACTIVITY_N.1) || lat.abs() > 90.0 {
            counters.levels_out_of_range += 1;
            continue;
        }
        if z <= floor {
            counters.levels_below_superrefraction += 1;
            continue;
        }
        if z < min_m || z > max_m {
            counters.levels_outside_altitude_window += 1;
            continue;
        }
        good.push(i);
    }
    good.sort_by(|a, b| profile.altitude_m[*a].partial_cmp(&profile.altitude_m[*b]).unwrap());
    if good.is_empty() {
        return;
    }
    let mut used = std::collections::BTreeSet::new();
    let mut target = (min_m / step_m).ceil() * step_m;
    while target <= max_m {
        // nearest good level by altitude (binary search on sorted altitudes)
        let pos = good.partition_point(|&i| profile.altitude_m[i] < target);
        let mut best: Option<usize> = None;
        for cand in [pos.checked_sub(1), (pos < good.len()).then_some(pos)].into_iter().flatten() {
            let i = good[cand];
            match best {
                None => best = Some(i),
                Some(b) if (profile.altitude_m[i] - target).abs() < (profile.altitude_m[b] - target).abs() => {
                    best = Some(i)
                }
                _ => {}
            }
        }
        match best {
            Some(i) if (profile.altitude_m[i] - target).abs() <= step_m / 2.0 && used.insert(i) => {
                let dry_p = profile.dry_pressure_pa[i];
                writer.push(
                    TableRow {
                        source: source.to_string(),
                        station_id: profile.station_id.clone(),
                        latitude_deg: profile.latitude_deg[i],
                        longitude_deg: profile.longitude_deg[i],
                        elevation_m: profile.altitude_m[i],
                        level_pa: (dry_p.is_finite() && dry_p > 0.0 && dry_p < 1.2e5).then_some(dry_p),
                        valid_time: profile.time,
                        variable: VAR_REFRACTIVITY.to_string(),
                        value: profile.refractivity[i],
                        error: refractivity_error(profile.altitude_m[i], profile.latitude_deg[i], profile.refractivity[i]),
                        provenance: provenance.clone(),
                    },
                    &mut counters.rows_by_variable,
                );
            }
            _ => counters.target_heights_without_level += 1,
        }
        target += step_m;
    }
}

#[derive(Serialize)]
struct SourceFile {
    path: String,
    bytes: usize,
    sha256: String,
    station_id: String,
    time: String,
    levels_in_file: usize,
    rows: usize,
}

#[derive(Serialize)]
struct SourceTarball {
    path: String,
    bytes: u64,
    sha256: String,
    members: usize,
    profiles_read: usize,
    profiles_kept: usize,
    profiles_bad_flag: usize,
    profiles_outside_window: usize,
    profiles_unreadable: usize,
    rows: usize,
    first_time: Option<String>,
    last_time: Option<String>,
    unreadable_examples: Vec<String>,
}

/// The published (Last-Modified) and received (fetched_at) instants per
/// source object name, from a fetch record of either route.
fn provenance_times(path: &Path) -> Result<(BTreeMap<String, DateTime<Utc>>, BTreeMap<String, DateTime<Utc>>), Box<dyn Error>> {
    let text = std::fs::read_to_string(path).map_err(|e| err(format!("cannot read {}: {e}", path.display())))?;
    let value: serde_json::Value = serde_json::from_str(&text).map_err(|e| err(format!("{} is not a fetch record: {e}", path.display())))?;
    let schema = value.get("schema").and_then(|s| s.as_str()).unwrap_or("");
    let name_of = |key: &str| Path::new(key).file_name().and_then(|n| n.to_str()).map(str::to_string);
    let mut published = BTreeMap::new();
    let mut received = BTreeMap::new();
    match schema {
        FETCH_SCHEMA => {
            let record: FetchRecord = serde_json::from_value(value)?;
            for f in &record.files {
                if let Some(name) = name_of(&f.listed.key) {
                    if let Some(t) = f.listed.last_modified.as_deref().and_then(|t| parse_time(&format!("{t}Z")).ok()) {
                        published.insert(name.clone(), t);
                    }
                    if let Some(t) = f.fetched_at.as_deref().and_then(|t| parse_time(&format!("{t}Z")).ok()) {
                        received.insert(name, t);
                    }
                }
            }
        }
        CDAAC_FETCH_SCHEMA => {
            let record: CdaacFetchRecord = serde_json::from_value(value)?;
            for f in &record.files {
                if let Some(name) = name_of(&f.path) {
                    if let Some(t) = f.last_modified.as_deref().and_then(|t| parse_time(&format!("{t}Z")).ok()) {
                        published.insert(name.clone(), t);
                    }
                    if let Some(t) = f.fetched_at.as_deref().and_then(|t| parse_time(&format!("{t}Z")).ok()) {
                        received.insert(name, t);
                    }
                }
            }
        }
        other => {
            return Err(err(format!(
                "{} declares schema {other:?}, expected {FETCH_SCHEMA:?} or {CDAAC_FETCH_SCHEMA:?}",
                path.display()
            )))
        }
    }
    Ok((published, received))
}

fn cmd_table(options: &Options) -> Result<String, Box<dyn Error>> {
    if options.files.is_empty() && options.tarballs.is_empty() {
        return Err(err("--files (AWS occultation files) or --tarballs (CDAAC daily tarballs) is required"));
    }
    let out = options.out.as_deref().ok_or_else(|| err("--out FILE.csv is required"))?;
    if out.is_dir() {
        return Err(err(format!("--out {} is a directory; give the CSV path", out.display())));
    }
    if !options.files.is_empty() && !options.tarballs.is_empty() {
        return Err(err(
            "--files and --tarballs in one table: the two routes are two streams (gnss-ro, the              retrospective AWS archive; cdaac-ro, the delayed-replay CDAAC tarballs) and a row's              source names its route, so write them as two tables",
        ));
    }
    let source: &'static str = if options.tarballs.is_empty() { SOURCE_AWS } else { SOURCE_CDAAC };
    let step = options.level_step_m.unwrap_or(DEFAULT_LEVEL_STEP_M);
    let min_m = options.min_altitude_m.unwrap_or(DEFAULT_MIN_ALTITUDE_M);
    let max_m = options.max_altitude_m.unwrap_or(DEFAULT_MAX_ALTITUDE_M);
    if max_m <= min_m {
        return Err(err("--max-altitude-m must exceed --min-altitude-m"));
    }
    let (window_start, window_end) = if options.tarballs.is_empty() { (None, None) } else { options.window()? };
    // Per object: the source's Last-Modified (published) and the fetch
    // instant (received), keyed by file name, when a fetch record is given.
    let (published, received) = match &options.fetch_record {
        Some(path) => provenance_times(path)?,
        None => (BTreeMap::new(), BTreeMap::new()),
    };
    let mut counters = Counters::default();
    let mut writer = TableWriter::new();
    let mut files = Vec::new();
    let mut tarballs = Vec::new();
    for path in &options.files {
        let bytes = std::fs::read(path).map_err(|e| err(format!("cannot read {}: {e}", path.display())))?;
        let sha = hex_sha256(&bytes);
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("").to_string();
        let provenance = RowProvenance::of_source(&sha, published.get(&name).copied(), received.get(&name).copied());
        let profile = read_profile(path)?;
        counters.files_read += 1;
        counters.profiles_read += 1;
        let before = writer.len();
        rows_for(&profile, source, step, min_m, max_m, &provenance, &mut counters, &mut writer);
        *counters
            .profiles_by_station_prefix
            .entry(profile.station_id.split('-').next().unwrap_or("").to_string())
            .or_insert(0) += 1;
        files.push(SourceFile {
            path: rw_obs::absolute_uri(path),
            bytes: bytes.len(),
            sha256: sha,
            station_id: profile.station_id.clone(),
            time: seam_time(profile.time),
            levels_in_file: profile.levels_in_file,
            rows: writer.len() - before,
        });
    }
    for path in &options.tarballs {
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("").to_string();
        let file = std::fs::File::open(path).map_err(|e| err(format!("cannot open {}: {e}", path.display())))?;
        let size = file.metadata()?.len();
        // The tarball's own digest is its revision; the sidecar the fetch
        // wrote is trusted when its size still matches, else re-hashed.
        let sidecar = sibling(path, ".sha256");
        let sha = match std::fs::read_to_string(&sidecar) {
            Ok(text) if text.trim().len() == 64 => text.trim().to_string(),
            _ => {
                let mut hasher = <sha2::Sha256 as sha2::Digest>::new();
                let mut reader = std::io::BufReader::new(std::fs::File::open(path)?);
                let mut buffer = vec![0u8; 1 << 20];
                loop {
                    let n = reader.read(&mut buffer)?;
                    if n == 0 {
                        break;
                    }
                    sha2::Digest::update(&mut hasher, &buffer[..n]);
                }
                format!("{:x}", sha2::Digest::finalize(hasher))
            }
        };
        let provenance = RowProvenance::of_source(&sha, published.get(&name).copied(), received.get(&name).copied());
        let gzipped = name.ends_with(".gz") || name.ends_with(".tgz");
        let reader = std::io::BufReader::with_capacity(1 << 20, file);
        let subject = path.display().to_string();
        let mut record = SourceTarball {
            path: rw_obs::absolute_uri(path),
            bytes: size,
            sha256: sha,
            members: 0,
            profiles_read: 0,
            profiles_kept: 0,
            profiles_bad_flag: 0,
            profiles_outside_window: 0,
            profiles_unreadable: 0,
            rows: 0,
            first_time: None,
            last_time: None,
            unreadable_examples: Vec::new(),
        };
        let mut first: Option<DateTime<Utc>> = None;
        let mut last: Option<DateTime<Utc>> = None;
        let before_rows = writer.len();
        for member in tar_members(reader, gzipped, subject.clone()) {
            let member = member?;
            record.members += 1;
            counters.tarball_members += 1;
            let profile = match read_cdaac_profile(&member.name, &member.bytes) {
                Ok(Ok(profile)) => profile,
                Ok(Err(CdaacSkip::BadFlag(flag))) => {
                    record.profiles_bad_flag += 1;
                    counters.profiles_bad_flag += 1;
                    *counters.bad_flags.entry(flag).or_insert(0) += 1;
                    continue;
                }
                Ok(Err(CdaacSkip::NotAProfile)) => {
                    counters.members_not_profiles += 1;
                    continue;
                }
                Err(e) => {
                    record.profiles_unreadable += 1;
                    counters.profiles_unreadable += 1;
                    if record.unreadable_examples.len() < 5 {
                        record.unreadable_examples.push(e.to_string());
                    }
                    continue;
                }
            };
            record.profiles_read += 1;
            counters.profiles_read += 1;
            first = Some(first.map_or(profile.time, |t| t.min(profile.time)));
            last = Some(last.map_or(profile.time, |t| t.max(profile.time)));
            let inside = window_start.is_none_or(|s| profile.time >= s) && window_end.is_none_or(|e| profile.time <= e);
            if !inside {
                record.profiles_outside_window += 1;
                counters.profiles_outside_window += 1;
                continue;
            }
            record.profiles_kept += 1;
            *counters
                .profiles_by_station_prefix
                .entry(profile.station_id.split('-').next().unwrap_or("").to_string())
                .or_insert(0) += 1;
            rows_for(&profile, source, step, min_m, max_m, &provenance, &mut counters, &mut writer);
        }
        record.rows = writer.len() - before_rows;
        record.first_time = first.map(seam_time);
        record.last_time = last.map(seam_time);
        counters.tarballs_read += 1;
        tarballs.push(record);
    }
    let (rows, csv_sha, csv_bytes) = writer.write(out)?;
    #[derive(Serialize)]
    struct Record {
        schema: &'static str,
        status: &'static str,
        source: &'static str,
        table_schema: &'static str,
        path: String,
        sha256: String,
        rows: usize,
        bytes: usize,
        level_step_m: f64,
        min_altitude_m: f64,
        max_altitude_m: f64,
        window_start: Option<String>,
        window_end: Option<String>,
        anchor: &'static str,
        error_rule: &'static str,
        counters: Counters,
        files: Vec<SourceFile>,
        tarballs: Vec<SourceTarball>,
    }
    let text = format!(
        "{}\n",
        serde_json::to_string_pretty(&Record {
            schema: TABLE_RECORD_SCHEMA,
            status: if rows > 0 { "READY" } else { "EMPTY" },
            source,
            table_schema: TABLE_SCHEMA,
            path: rw_obs::absolute_uri(out),
            sha256: csv_sha,
            rows,
            bytes: csv_bytes,
            level_step_m: step,
            min_altitude_m: min_m,
            max_altitude_m: max_m,
            window_start: window_start.map(seam_time),
            window_end: window_end.map(seam_time),
            anchor: "elevation_m is the tangent-point altitude above mean sea level and the row's \
                     vertical anchor; level_pa is the retrieval's dry pressure, carried for the \
                     column-span gates and the ln p localisation",
            error_rule: ERROR_REFRACTIVITY_RULE,
            counters,
            files,
            tarballs,
        })?
    );
    std::fs::write(out.with_extension("json"), &text).map_err(|e| err(format!("cannot write the table record: {e}")))?;
    Ok(text)
}

fn cmd_verify(options: &Options) -> Result<String, Box<dyn Error>> {
    rw_obs::table::verify(
        options.file.as_deref(), options.record.as_deref(), TABLE_RECORD_SCHEMA, VERIFY_SCHEMA,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_profile(superrefraction: Option<f64>) -> Profile {
        // 10 m levels from 0 to 40 km, N = 300 exp(-z / 7 km).
        let n = 4001;
        let altitude: Vec<f64> = (0..n).map(|i| i as f64 * 10.0).collect();
        let refractivity: Vec<f64> = altitude.iter().map(|z| 300.0 * (-z / 7000.0).exp()).collect();
        let dry_pressure: Vec<f64> = altitude.iter().map(|z| 101325.0 * (-z / 8000.0).exp()).collect();
        Profile {
            station_id: "cosmic2e1-E03".into(),
            time: Utc.with_ymd_and_hms(2025, 7, 29, 0, 57, 43).unwrap(),
            latitude_deg: vec![12.0; n],
            longitude_deg: vec![-23.0; n],
            altitude_m: altitude,
            refractivity,
            dry_pressure_pa: dry_pressure,
            superrefraction_height_m: superrefraction,
            levels_in_file: n,
        }
    }

    #[test]
    fn one_row_per_target_height_with_the_exact_level_when_it_exists() {
        let mut c = Counters::default();
        let mut w = TableWriter::new();
        rows_for(&synthetic_profile(None), SOURCE_CDAAC, 200.0, 0.0, 30_000.0, &RowProvenance::default(), &mut c, &mut w);
        // targets 0, 200, ..., 30000 = 151 rows
        assert_eq!(w.len(), 151, "{c:?}");
        assert!(w.rows().iter().all(|r| r.provenance.measurement == MEAS_RO_REFRACTIVITY_TANGENT));
        let r = &w.rows()[10];
        assert_eq!(r.elevation_m, 2000.0);
        assert!((r.value - 300.0 * (-2000.0f64 / 7000.0).exp()).abs() < 1e-9);
        // the error follows the stated shape: tropical (12 N) at 2 km
        let expected = refractivity_error_fraction(2000.0, 12.0) * r.value;
        assert!((r.error - expected).abs() < 1e-12);
        assert!(r.error / r.value > 0.015 && r.error / r.value < 0.02, "{}", r.error / r.value);
        let high = w.rows().iter().find(|r| r.elevation_m == 20_000.0).unwrap();
        assert!((high.error / high.value - 0.003).abs() < 5e-5);
        assert!(high.level_pa.unwrap() < 10_000.0);
    }

    #[test]
    fn levels_under_a_superrefraction_layer_are_dropped_and_counted() {
        let mut c = Counters::default();
        let mut w = TableWriter::new();
        rows_for(&synthetic_profile(Some(1500.0)), SOURCE_AWS, 200.0, 0.0, 30_000.0, &RowProvenance::default(), &mut c, &mut w);
        assert!(w.rows().iter().all(|r| r.elevation_m > 1500.0));
        assert_eq!(c.levels_below_superrefraction, 151);
        assert_eq!(c.profiles_with_superrefraction, 1);
        assert_eq!(c.target_heights_without_level, 8, "{c:?}"); // 0..1400 have no level
    }

    #[test]
    fn a_sparse_profile_leaves_targets_without_a_level_counted() {
        let mut p = synthetic_profile(None);
        // keep every 50th level (500 m spacing)
        let keep: Vec<usize> = (0..p.levels_in_file).step_by(50).collect();
        p.altitude_m = keep.iter().map(|&i| p.altitude_m[i]).collect();
        p.refractivity = keep.iter().map(|&i| p.refractivity[i]).collect();
        p.dry_pressure_pa = keep.iter().map(|&i| p.dry_pressure_pa[i]).collect();
        p.latitude_deg = vec![12.0; keep.len()];
        p.longitude_deg = vec![-23.0; keep.len()];
        p.levels_in_file = keep.len();
        let mut c = Counters::default();
        let mut w = TableWriter::new();
        rows_for(&p, SOURCE_CDAAC, 200.0, 0.0, 30_000.0, &RowProvenance::default(), &mut c, &mut w);
        // Levels at 0, 500, 1000, ...: the targets 0, 1000, 2000, ... hit their
        // level exactly (31 rows); target 400 takes the 500 m level (100 m
        // away, inside step/2) and target 600 finds it already used; 200 and
        // 800 are 200 m from anything.  Two rows per kilometre plus the top.
        assert_eq!(w.len(), 61, "{c:?}");
        assert_eq!(c.target_heights_without_level, 151 - 61);
    }

    #[test]
    fn a_cdaac_fill_level_is_dropped_and_the_station_id_reads_the_file_stamp() {
        assert_eq!(cdaac_station_id("atmPrf_C2E1.2026.244.00.07.G09_0001.0001_nc", None), "C2E1-G09");
        assert_eq!(cdaac_station_id("x/atmPrf_PAZ1.2026.244.01.02.R21_0001.0001_nc", Some("PAZ1.2026.244.01.02.R21")), "PAZ1-R21");
        let mut p = synthetic_profile(None);
        p.refractivity[3] = f64::NAN;
        p.altitude_m[7] = f64::NAN;
        let mut c = Counters::default();
        let mut w = TableWriter::new();
        rows_for(&p, SOURCE_CDAAC, 200.0, 0.0, 30_000.0, &RowProvenance::default(), &mut c, &mut w);
        assert_eq!(c.levels_fill, 2);
        assert_eq!(w.len(), 151);
    }

    #[test]
    fn cdaac_urls_follow_the_portal_layout_and_a_member_that_is_no_profile_is_skipped() {
        assert_eq!(
            cdaac_url("cosmic2", NaiveDate::from_ymd_opt(2026, 9, 1).unwrap()),
            "https://data.cosmic.ucar.edu/gnss-ro/cosmic2/nrt/level2/2026/244/atmPrf_nrt_2026_244.tar.gz"
        );
        match read_cdaac_profile("README", b"not a file").unwrap() {
            Err(CdaacSkip::NotAProfile) => {}
            other => panic!("{other:?}"),
        }
        assert!(read_cdaac_profile("atmPrf_x_nc", b"not a netcdf file").is_err());
    }

    #[test]
    fn the_sidecar_and_partial_names_keep_the_whole_file_name() {
        let p = Path::new("/c/cosmic2/atmPrf_nrt_2026_244.tar.gz");
        assert!(sibling(p, ".sha256").ends_with("atmPrf_nrt_2026_244.tar.gz.sha256"));
        assert!(sibling(p, ".partial").ends_with("atmPrf_nrt_2026_244.tar.gz.partial"));
    }

    #[test]
    fn abi_marker_names_the_contracts_it_pins() {
        assert!(ABI_MARKER.contains(FETCH_SCHEMA));
        assert!(ABI_MARKER.contains(CDAAC_FETCH_SCHEMA));
        assert!(ABI_MARKER.contains(TABLE_RECORD_SCHEMA));
        assert!(ABI_MARKER.contains(TABLE_SCHEMA));
    }
}
