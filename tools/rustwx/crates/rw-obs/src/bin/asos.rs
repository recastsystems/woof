//! `rw_asos`: ArWen's surface-observation front door.
//!
//! ASOS/AWOS routine and special METARs from the Iowa Environmental Mesonet
//! archive, frozen into a hash-pinned station table and a report set in
//! seam units, matched to hourly valid times (or, from the one-minute pages,
//! to every minute).
//!
//! This is the headless retelling of the query pattern that already works in
//! The project's Rust meteorology stack, with three deliberate departures, each of
//! which the battery needs:
//!
//! 1. **The query is bounded.** The GUI's fetch sends no `station=` and no
//!    `network=`, so it pulls every station on Earth for the window and
//!    filters locally. A battery that does that seven times over 24-hour
//!    windows is asking the archive for orders of magnitude more than it
//!    scores. Here the station list comes from a frozen table and goes into
//!    the request.
//! 2. **The station table is frozen and hashed, not read off each fetch.**
//!    Coordinates that arrive with the observations can change between the
//!    fetch that registered a case and the fetch that scores it, and the
//!    battery pins its station set at registration.
//! 3. **`mslp` and `p01i` are requested.** The GUI asks for
//!    `tmpf,dwpf,drct,sknt,gust,alti`; the spec reports MSLP as a
//!    diagnostic, and hourly precipitation is worth carrying while the
//!    request is being made anyway.
//!
//! Everything else is kept: the `asos.py` endpoint, `tz=Etc/UTC`,
//! `format=onlycomma`, `missing=empty`, `report_type=3&report_type=4`, the
//! unpadded date components, and resolving CSV columns by name.
//!
//! ```text
//! rw_asos stations --networks IA_ASOS,IL_ASOS --out stations.json
//! rw_asos fetch    --stations stations.json --start ... --end ... --out obs.csv
//! rw_asos decode   --obs obs.csv --stations stations.json --start ... --end ... --out obs.json
//! rw_asos verify   --file obs.json
//! ```
//!
//! **One-minute ASOS (`--product asos1min`).** The same frozen station table
//! and the same seam, fed from the archive's `asos1min.py` route (the
//! one-minute ASOS pages) instead of the METARs. An hourly or special METAR
//! reaches a cycle of a few minutes only at the few analyses after the top
//! of the hour; the one-minute record reaches every one. Its CSV spells the
//! time column `valid(UTC)` and carries the same `tmpf`, `dwpf` and `sknt`
//! columns, so decode is the same code with the column resolved by the
//! product's own name, a one-minute valid-time stride by default, and a
//! provenance that says which product the record holds.

use std::collections::BTreeMap;
use std::error::Error;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use chrono::{DateTime, Datelike, NaiveDateTime, TimeZone, Timelike, Utc};
use serde::{Deserialize, Serialize};

use rw_nexrad::s3::parse_time;
use rw_obs::net::{agent, get_text, get_text_cached, query_encode};
use rw_obs::seam::{seam_time, wrap_longitude, Provenance, TIME_FORMAT};
use rw_obs::table::{
    altimeter_inhg_to_station_pa, fahrenheit_to_kelvin,
    knots_to_m_s, wind_components, RowProvenance, TableRow, TableWriter,
    ERROR_DEWPOINT_SURFACE_K, ERROR_SURFACE_PRESSURE_PA, ERROR_TEMPERATURE_SURFACE_K,
    ERROR_WIND_SURFACE_M_S, GROSS_DEWPOINT_K, GROSS_SURFACE_PRESSURE_PA, GROSS_TEMPERATURE_K,
    GROSS_WIND_M_S, MEAS_ANEMOMETER_WIND_10M, MEAS_SCREEN_DEWPOINT_2M,
    MEAS_SCREEN_TEMPERATURE_2M, MEAS_STATION_PRESSURE_FROM_ALTIMETER, TABLE_SCHEMA,
    VAR_DEWPOINT, VAR_SURFACE_PRESSURE, VAR_TEMPERATURE, VAR_WIND_U, VAR_WIND_V,
};
use rw_obs::{err, hex_sha256};

const VERSION: &str = env!("CARGO_PKG_VERSION");

/// `GPUWM_BRIDGE_SOURCE_REV=<40-hex commit>`: the source revision this
/// binary was built from, embedded so the gpuwm release cut can prove a
/// staged bridge matches the commit being released by reading bytes
/// alone (`tools/build_bridge_bundle.py pin --source-rev`).  `build.rs`
/// injects the value; `main` references the constant so the linker
/// cannot discard it.
pub static GPUWM_BRIDGE_SOURCE_REV_STAMP: &str =
    concat!("GPUWM_BRIDGE_SOURCE_REV=", env!("GPUWM_BRIDGE_SOURCE_REV"));

const DEFAULT_ARCHIVE: &str = "https://mesonet.agron.iastate.edu";
/// The columns requested, in the order the request states them.
const DATA_COLUMNS: &[&str] = &[
    "tmpf", "dwpf", "drct", "sknt", "gust", "alti", "mslp", "p01i", "vsby",
];
/// The one-minute route's variables, in the order the request states them.
/// The one-minute pages carry no MSLP, gust, altimeter or hourly
/// precipitation; temperature, dewpoint and wind are what the seam scores.
const ONE_MINUTE_VARS: &[&str] = &["tmpf", "dwpf", "sknt", "drct"];

/// One archive route `--product` may name. Everything that differs between
/// the routes after the request is a column of this table: the CSV's time
/// column, the provenance label the record carries, and the valid-time
/// stride and match window a report of that cadence takes by default.
struct Product {
    name: &'static str,
    time_column: &'static str,
    provenance: &'static str,
    default_step_minutes: u32,
    default_match_seconds: i64,
}

const PRODUCT_METAR: &str = "metar";
const PRODUCT_ONE_MINUTE: &str = "asos1min";

/// The METAR archive (`asos.py`, the default) and the one-minute pages
/// (`asos1min.py`). A one-minute report serves the minute it was taken:
/// the stride is one minute and the match window half of it.
const PRODUCTS: &[Product] = &[
    Product {
        name: PRODUCT_METAR,
        time_column: "valid",
        provenance: "iem-asos-metar",
        default_step_minutes: 60,
        default_match_seconds: DEFAULT_MATCH_SECONDS,
    },
    Product {
        name: PRODUCT_ONE_MINUTE,
        time_column: "valid(UTC)",
        provenance: "iem-asos-1min",
        default_step_minutes: 1,
        default_match_seconds: 30,
    },
];

fn product_named(name: &str) -> Option<&'static Product> {
    PRODUCTS.iter().find(|product| product.name == name)
}

/// The registered gross-error screen, in the units the seam pins.
/// Transcribed from the battery spec: T2 in [-40, 55] C, dewpoint at or
/// below temperature, wind speed in [0, 75] m/s.
const TEMPERATURE_MIN_K: f64 = 233.15; // -40 C
const TEMPERATURE_MAX_K: f64 = 328.15; // +55 C
const WIND_MIN_MS: f64 = 0.0;
const WIND_MAX_MS: f64 = 75.0;
/// A station whose screen fires on more than this share of its reports is
/// dropped entirely rather than partly trusted.
const DEFAULT_MAX_SCREEN_FAILURE_RATE: f64 = 0.05;
/// Nearest report within this many seconds of a valid time is that hour's.
const DEFAULT_MATCH_SECONDS: i64 = 600;
/// A station reporting fewer than this share of the scored hours is dropped.
const DEFAULT_MIN_REPORT_RATE: f64 = 0.80;
/// How many stations one window request may name.
///
/// Measured against the archive 2026-08-04 while pulling the battery's own
/// case boxes: a frozen table of 642 stations answered 200, and 698 answered
/// **HTTP 414 URI Too Long**. A 1440 x 1200 km box over the dense Midwest or
/// Southeast freezes 575-802 stations, so the bounded query has to be split.
/// 400 sits well inside the measured ceiling and costs two or three requests
/// for a battery case rather than one.
const DEFAULT_STATIONS_PER_REQUEST: usize = 400;
/// How long to wait between the chunk requests of one window.
///
/// Also measured 2026-08-04: pulling seven case boxes back to back, three
/// chunks each, earned an **HTTP 429** on the twenty-first request. The
/// archive is free and is asking to be paced, so the pace is a default
/// rather than something an operator has to remember.
const DEFAULT_REQUEST_PAUSE_MS: u64 = 2000;

const STATIONS_SCHEMA: &str = "gpuwm-obs.asos-stations.v1";
const FETCH_SCHEMA: &str = "gpuwm-obs.asos-fetch.v1";
/// The record `decode` writes.  `v2` carries each report's own
/// `observation_time` beside the `valid_time` it was matched to, and a
/// report serves exactly one valid time; `v1` collapsed the two and could
/// emit one report under two hours.  `verify` reads both.
const SURFACE_SCHEMA: &str = "gpuwm-obs.asos-surface.v2";
const SURFACE_SCHEMA_V1: &str = "gpuwm-obs.asos-surface.v1";
const VERIFY_SCHEMA: &str = "gpuwm-obs.asos-verify.v1";
/// The neutral-table record `rw_asos table` prints (design item 2 of the
/// global DA program): the IEM window CSV converted to `gpuwm-obs.table.v1`.
const TABLE_RECORD_SCHEMA: &str = "gpuwm-obs.asos-table.v1";
/// Networks per `fetch --networks` request.  Measured 2026-09-06: ten
/// networks over a 32 h window answered in 1 to 2 s with up to 26,000 rows
/// (2.1 MB); the whole ASOS estate is 266 networks, so a global window is
/// 27 requests.
const DEFAULT_NETWORKS_PER_REQUEST: usize = 10;
const TABLE_SOURCE: &str = "iem-metar";
/// A window that ended this long ago is settled archive content: its CSV is
/// cached for good, so a second arm or a rerun scoring the same frame never
/// asks IEM again (the 2026-11-09 429 defect: the arm that asked second was
/// refused). Late METARs and corrections land within hours, so 48 h is well
/// past settling.
const SETTLED_WINDOW_HOURS: i64 = 48;
/// A window newer than that is cached for this long: long enough for the
/// arms of one scoring run to share it, short enough that a live window
/// picks up new reports on the next run.
const RECENT_WINDOW_CACHE_SECONDS: u64 = 15 * 60;

fn window_cache_rule(end: DateTime<Utc>, now: DateTime<Utc>) -> rw_http::CacheRule {
    if now - end >= chrono::Duration::hours(SETTLED_WINDOW_HOURS) {
        rw_http::CacheRule::Forever
    } else {
        rw_http::CacheRule::Ttl(std::time::Duration::from_secs(RECENT_WINDOW_CACHE_SECONDS))
    }
}

/// Only an ASOS CSV is cached; a 200 carrying an error page is not.
fn is_asos_csv(body: &[u8]) -> bool {
    body.starts_with(b"station,")
}

const ABI_MARKER: &str = "gpuwm-obs.asos-surface.v2\tstations\treports\tprovenance\t\
observation_time\ttemperature_2m\tdewpoint_2m\twind_speed_10m\tmslp\tK\tm s-1\tPa\t\
gpuwm-obs.asos-table.v1\tgpuwm-obs.table.v2\tiem-asos-1min";

const USAGE: &str = "\
usage: rw_asos <stations|fetch|decode|verify> [OPTIONS]
       rw_asos --version | --help | --abi

  stations  freeze a station table from the archive's own network metadata
  fetch     download one bounded CSV window and print its sha256
  decode    screen, convert to seam units, match each report to the one
            valid time nearest it, write a `gpuwm-obs.asos-surface.v2`
            record (every report carries its observation_time beside the
            valid_time it serves)
  verify    re-hash a decoded record's source against the digest it carries
            and prove no report serves two valid times
  table     convert a fetched CSV to the neutral observation table
            (`gpuwm-obs.table.v2`: station pressure from the altimeter
            setting, 2 m temperature and dewpoint in K, 10 m u and v)
  networks  list the archive's networks (default: every *ASOS network, the
            METAR estate) from its own networks.geojson, so a global fetch
            names them without a hand-kept list
  awc       fetch the Aviation Weather Center METAR cache (the last hour,
            worldwide, rewritten every minute) and convert it to the neutral
            table: the complementary surface source

archive options
  --archive URL          default: https://mesonet.agron.iastate.edu
  --networks LIST        comma-separated IEM networks, e.g. IA_ASOS,IL_ASOS
  --bbox W,S,E,N         stations: keep only sites inside this lon/lat box
  --stations FILE        a frozen station table from `stations`
  --start TIME           window start
  --end TIME             window end (inclusive)
  --out PATH             the destination file

fetch options
  --networks LIST        fetch whole IEM networks (comma-separated, e.g.
                         GB__ASOS,FR__ASOS) instead of a frozen station
                         table: the global route, ten networks per request
  --networks-per-request N
                         how many networks one request may name (default 10)
  --stations-per-request N
                         how many stations one window request may name.
                         Default 400. Measured 2026-08-04: 642 stations
                         answered 200 and 698 answered HTTP 414, and a
                         battery-shaped box freezes 575-802, so the window
                         is fetched in chunks and the CSV bodies joined
  --request-pause-ms N   wait this long between chunk requests. Default 2000.
                         Seven case boxes pulled back to back earned an
                         HTTP 429 on the twenty-first request; the archive is
                         free and asks to be paced

fetch and decode options
  --report-types TYPES   live METAR report types, comma-separated 1,3,4.
                         Default 3,4; 1 adds high-frequency HFMETAR reports.
  --product NAME         metar (default): routine and special METARs from
                         asos.py. asos1min: the one-minute ASOS pages from
                         asos1min.py for the same stations and seam, a
                         report every minute instead of at the hour (asked
                         for by station only: the route takes no network).
                         decode refuses a CSV whose time column is the other
                         product's

decode options
  --obs FILE             the CSV from `fetch`
  --step-hours N         valid-time stride in hours, in [1, 24]
  --step-minutes N       valid-time stride in minutes, in [1, 1440]. Default
                         60 for metar and 1 for asos1min; give one of the two
                         stride options
  --match-seconds N      a report serves the valid time nearest it, and only
                         when within this many seconds of it. Default 600
                         for metar and 30 for asos1min
  --min-report-rate F    drop a station reporting fewer than this share of
                         the valid times. Default 0.80
  --max-screen-rate F    drop a station whose gross-error screen fires on more
                         than this share of its reports. Default 0.05

verify options
  --file FILE            the decoded record to re-prove

table options
  --obs FILE             the CSV from `fetch`
  --start TIME           keep reports at or after this instant (optional)
  --end TIME             keep reports at or before this instant (optional)
  --out PATH             the table CSV; a .json record is written beside it
  --fetch-record FILE    the `fetch` record, so the table record can bound
                         how far behind real time the archive answered
";

fn main() -> ExitCode {
    let _ = std::hint::black_box(GPUWM_BRIDGE_SOURCE_REV_STAMP);
    rw_obs::cli::main("rw_asos", run)
}

fn run(args: &[String]) -> Result<String, Box<dyn Error>> {
    if let Some(output) = rw_obs::cli::information(args, "rw_asos", VERSION, USAGE, ABI_MARKER) {
        return Ok(output);
    }
    let first = &args[0];
    let options = Options::parse(&args[1..])?;
    match first.as_str() {
        "stations" => cmd_stations(&options),
        "fetch" => cmd_fetch(&options),
        "decode" => cmd_decode(&options),
        "verify" => cmd_verify(&options),
        "table" => cmd_table(&options),
        "networks" => cmd_networks(&options),
        "awc" => cmd_awc(&options),
        other => Err(err(format!("unknown subcommand {other:?}\n\n{USAGE}"))),
    }
}

#[derive(Debug, Default)]
struct Options {
    archive: Option<String>,
    networks: Option<String>,
    bbox: Option<[f64; 4]>,
    stations: Option<PathBuf>,
    obs: Option<PathBuf>,
    file: Option<PathBuf>,
    start: Option<String>,
    end: Option<String>,
    out: Option<PathBuf>,
    step_hours: Option<u32>,
    match_seconds: Option<i64>,
    min_report_rate: Option<f64>,
    max_screen_rate: Option<f64>,
    stations_per_request: Option<usize>,
    request_pause_ms: Option<u64>,
    networks_per_request: Option<usize>,
    fetch_record: Option<PathBuf>,
    product: Option<String>,
    step_minutes: Option<u32>,
    report_types: Option<Vec<u8>>,
}

impl Options {
    fn parse(args: &[String]) -> Result<Self, Box<dyn Error>> {
        let mut options = Options::default();
        let mut index = 0;
        while index < args.len() {
            let flag = args[index].as_str();
            let mut value = || rw_obs::cli::value(args, &mut index, flag);
            let rate = |raw: String, flag: &str| -> Result<f64, Box<dyn Error>> {
                let parsed: f64 = raw
                    .parse()
                    .map_err(|_| err(format!("{flag} expects a fraction, got {raw:?}")))?;
                if !parsed.is_finite() || !(0.0..=1.0).contains(&parsed) {
                    return Err(err(format!("{flag} must lie in [0, 1], got {raw:?}")));
                }
                Ok(parsed)
            };
            match flag {
                "--archive" => options.archive = Some(value()?),
                "--networks" => options.networks = Some(value()?),
                "--bbox" => {
                    let raw = value()?;
                    let parts: Vec<&str> = raw.split(',').map(str::trim).collect();
                    if parts.len() != 4 {
                        return Err(err(format!("--bbox expects W,S,E,N, got {raw:?}")));
                    }
                    let mut values = [0.0f64; 4];
                    for (slot, text) in values.iter_mut().zip(parts) {
                        *slot = text
                            .parse()
                            .map_err(|_| err(format!("--bbox component {text:?} is not a number")))?;
                        if !slot.is_finite() {
                            return Err(err("--bbox components must be finite"));
                        }
                    }
                    if values[0] >= values[2] || values[1] >= values[3] {
                        return Err(err("--bbox must be W<E and S<N"));
                    }
                    options.bbox = Some(values);
                }
                "--stations" => options.stations = Some(PathBuf::from(value()?)),
                "--obs" => options.obs = Some(PathBuf::from(value()?)),
                "--file" => options.file = Some(PathBuf::from(value()?)),
                "--start" => options.start = Some(value()?),
                "--end" => options.end = Some(value()?),
                "--out" => options.out = Some(PathBuf::from(value()?)),
                "--step-hours" => {
                    let raw = value()?;
                    let hours: u32 = raw
                        .parse()
                        .map_err(|_| err(format!("--step-hours expects a count, got {raw:?}")))?;
                    if hours == 0 || hours > 24 {
                        return Err(err("--step-hours must lie in [1, 24]"));
                    }
                    options.step_hours = Some(hours);
                }
                "--match-seconds" => {
                    let raw = value()?;
                    let seconds: i64 = raw
                        .parse()
                        .map_err(|_| err(format!("--match-seconds expects a count, got {raw:?}")))?;
                    if seconds <= 0 {
                        return Err(err("--match-seconds must be positive"));
                    }
                    options.match_seconds = Some(seconds);
                }
                "--min-report-rate" => {
                    let raw = value()?;
                    options.min_report_rate = Some(rate(raw, "--min-report-rate")?)
                }
                "--max-screen-rate" => {
                    let raw = value()?;
                    options.max_screen_rate = Some(rate(raw, "--max-screen-rate")?)
                }
                "--stations-per-request" => {
                    let raw = value()?;
                    let count: usize = raw.parse().map_err(|_| {
                        err(format!("--stations-per-request expects a count, got {raw:?}"))
                    })?;
                    if count == 0 {
                        return Err(err("--stations-per-request must be positive"));
                    }
                    options.stations_per_request = Some(count);
                }
                "--request-pause-ms" => {
                    let raw = value()?;
                    options.request_pause_ms = Some(raw.parse().map_err(|_| {
                        err(format!("--request-pause-ms expects milliseconds, got {raw:?}"))
                    })?);
                }
                "--networks-per-request" => {
                    let raw = value()?;
                    let count: usize = raw.parse().map_err(|_| {
                        err(format!("--networks-per-request expects a count, got {raw:?}"))
                    })?;
                    if count == 0 {
                        return Err(err("--networks-per-request must be positive"));
                    }
                    options.networks_per_request = Some(count);
                }
                "--fetch-record" => options.fetch_record = Some(PathBuf::from(value()?)),
                "--product" => {
                    let raw = value()?;
                    if product_named(&raw).is_none() {
                        let names: Vec<&str> = PRODUCTS.iter().map(|p| p.name).collect();
                        return Err(err(format!(
                            "--product must be one of {}, got {raw:?}",
                            names.join(", ")
                        )));
                    }
                    options.product = Some(raw);
                }
                "--report-types" => {
                    let raw = value()?;
                    let mut selected = Vec::new();
                    for token in raw.split(',').map(str::trim) {
                        let kind: u8 = token.parse().map_err(|_| err("--report-types expects comma-separated 1,3,4"))?;
                        if ![1,3,4].contains(&kind) { return Err(err("--report-types permits only 1 (HFMETAR), 3 (routine), 4 (special)")); }
                        if !selected.contains(&kind) { selected.push(kind); }
                    }
                    selected.sort_unstable();
                    options.report_types = Some(selected);
                }
                "--step-minutes" => {
                    let raw = value()?;
                    let minutes: u32 = raw
                        .parse()
                        .map_err(|_| err(format!("--step-minutes expects a count, got {raw:?}")))?;
                    if minutes == 0 || minutes > 1440 {
                        return Err(err("--step-minutes must lie in [1, 1440]"));
                    }
                    options.step_minutes = Some(minutes);
                }
                other => return Err(err(format!("unknown option {other:?}\n\n{USAGE}"))),
            }
            index += 1;
        }
        if options.step_hours.is_some() && options.step_minutes.is_some() {
            // Two strides for one decode: whichever won, the other would be
            // a flag the record silently did not follow.
            return Err(err("--step-hours and --step-minutes both set the valid-time stride; give one"));
        }
        if options.one_minute() && options.report_types.is_some() {
            return Err(err("--report-types applies only to the live METAR route"));
        }
        Ok(options)
    }

    fn product(&self) -> &'static Product {
        self.product
            .as_deref()
            .and_then(product_named)
            .unwrap_or(&PRODUCTS[0])
    }

    fn one_minute(&self) -> bool {
        self.product().name == PRODUCT_ONE_MINUTE
    }
    fn report_types(&self) -> &[u8] {
        self.report_types.as_deref().unwrap_or(&[3,4])
    }

    /// The decode stride: the one the caller gave, else the product's own.
    fn step_minutes(&self) -> u32 {
        match (self.step_minutes, self.step_hours) {
            (Some(minutes), _) => minutes,
            (None, Some(hours)) => hours * 60,
            (None, None) => self.product().default_step_minutes,
        }
    }

    fn match_seconds(&self) -> i64 {
        self.match_seconds
            .unwrap_or(self.product().default_match_seconds)
    }

    fn archive(&self) -> &str {
        self.archive
            .as_deref()
            .unwrap_or(DEFAULT_ARCHIVE)
            .trim_end_matches('/')
    }

    fn stations_per_request(&self) -> usize {
        self.stations_per_request
            .unwrap_or(DEFAULT_STATIONS_PER_REQUEST)
    }

    fn request_pause_ms(&self) -> u64 {
        self.request_pause_ms.unwrap_or(DEFAULT_REQUEST_PAUSE_MS)
    }

    fn window(&self) -> Result<(DateTime<Utc>, DateTime<Utc>), Box<dyn Error>> {
        rw_obs::cli::window(self.start.as_deref(), self.end.as_deref())
    }

    fn out(&self) -> Result<&Path, Box<dyn Error>> {
        let out = self.out.as_deref().ok_or_else(|| err("--out is required"))?;
        if out.is_dir() {
            return Err(err(format!(
                "--out {} is a directory; give the file path",
                out.display()
            )));
        }
        Ok(out)
    }
}

// -------------------------------------------------------------- stations

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Station {
    station_id: String,
    name: String,
    latitude: f64,
    longitude: f64,
    elevation_m: f64,
    network: String,
    state: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StationTable {
    schema: String,
    status: String,
    archive: String,
    networks: Vec<String>,
    frozen_at: String,
    /// Digest over the station rows, so a table that changed between the
    /// registration that froze it and the scoring pass that read it is
    /// caught rather than silently scored.
    content_sha256: String,
    stations: Vec<Station>,
}

/// The digest a station table pins: the rows, canonically spelled, and
/// nothing about when or from where they were fetched.
///
/// Hashing the whole document would make the digest change every time the
/// table was re-frozen from the same archive state, which would make it
/// useless as an identity for the *station set*.
fn station_rows_digest(stations: &[Station]) -> String {
    let mut text = String::new();
    for station in stations {
        text.push_str(&format!(
            "{}\t{:.6}\t{:.6}\t{:.3}\n",
            station.station_id, station.latitude, station.longitude, station.elevation_m
        ));
    }
    hex_sha256(text.as_bytes())
}

fn cmd_stations(options: &Options) -> Result<String, Box<dyn Error>> {
    let networks_raw = options
        .networks
        .as_deref()
        .ok_or_else(|| err("--networks is required, e.g. --networks IA_ASOS,IL_ASOS"))?;
    let mut networks: Vec<String> = Vec::new();
    for token in networks_raw.split(',').map(str::trim).filter(|t| !t.is_empty()) {
        if !token
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_')
        {
            return Err(err(format!(
                "network {token:?} may carry only letters, digits and '_'; it is concatenated \
                 into an archive URL"
            )));
        }
        let upper = token.to_ascii_uppercase();
        if !networks.contains(&upper) {
            networks.push(upper);
        }
    }
    if networks.is_empty() {
        return Err(err("--networks named no network"));
    }
    let out = options.out()?;
    let archive = options.archive().to_string();
    let agent = agent();

    let mut stations: Vec<Station> = Vec::new();
    for network in &networks {
        let url = format!("{archive}/geojson/network/{network}.geojson");
        let body = get_text(&agent, &url, "IEM network metadata")?;
        let document: serde_json::Value = serde_json::from_str(&body)
            .map_err(|e| err(format!("{network} metadata is not JSON: {e}")))?;
        let features = document
            .get("features")
            .and_then(|f| f.as_array())
            .ok_or_else(|| err(format!("{network} metadata carries no feature list")))?;
        for feature in features {
            let properties = feature.get("properties").unwrap_or(&serde_json::Value::Null);
            let coordinates = feature
                .get("geometry")
                .and_then(|g| g.get("coordinates"))
                .and_then(|c| c.as_array());
            let (Some(coordinates), Some(id)) = (
                coordinates,
                properties
                    .get("sid")
                    .and_then(|s| s.as_str())
                    .or_else(|| feature.get("id").and_then(|s| s.as_str())),
            ) else {
                continue;
            };
            if coordinates.len() < 2 {
                continue;
            }
            let (Some(lon), Some(lat)) = (coordinates[0].as_f64(), coordinates[1].as_f64()) else {
                continue;
            };
            let elevation = properties
                .get("elevation")
                .and_then(|e| e.as_f64())
                .unwrap_or(f64::NAN);
            if !lat.is_finite() || !lon.is_finite() || !elevation.is_finite() {
                // A station the battery cannot place, or cannot compare
                // against model terrain, is not a station the battery can
                // screen. Dropping it here is accurate; carrying a NaN
                // elevation into the seam is not.
                continue;
            }
            if !(-90.0..=90.0).contains(&lat) {
                continue;
            }
            let lon = wrap_longitude(lon);
            if let Some([west, south, east, north]) = options.bbox {
                if lon < west || lon > east || lat < south || lat > north {
                    continue;
                }
            }
            let station = Station {
                station_id: id.to_ascii_uppercase(),
                name: properties
                    .get("sname")
                    .and_then(|s| s.as_str())
                    .unwrap_or("")
                    .to_string(),
                latitude: lat,
                longitude: lon,
                elevation_m: elevation,
                network: network.clone(),
                state: properties
                    .get("state")
                    .and_then(|s| s.as_str())
                    .unwrap_or("")
                    .to_string(),
            };
            if !stations.iter().any(|s| s.station_id == station.station_id) {
                stations.push(station);
            }
        }
    }
    if stations.is_empty() {
        return Err(err(
            "no station survived; a frozen table with no stations would score every arm on \
             nothing and report a clean zero",
        ));
    }
    stations.sort_by(|a, b| a.station_id.cmp(&b.station_id));

    let table = StationTable {
        schema: STATIONS_SCHEMA.to_string(),
        status: "READY".to_string(),
        archive,
        networks,
        frozen_at: seam_time(Utc::now()),
        content_sha256: station_rows_digest(&stations),
        stations,
    };
    let text = serde_json::to_string_pretty(&table)?;
    std::fs::write(out, format!("{text}\n"))
        .map_err(|e| err(format!("cannot write {}: {e}", out.display())))?;

    #[derive(Serialize)]
    struct Record<'a> {
        schema: &'static str,
        status: &'static str,
        path: String,
        networks: &'a [String],
        stations: usize,
        content_sha256: &'a str,
    }
    Ok(format!(
        "{}\n",
        serde_json::to_string_pretty(&Record {
            schema: STATIONS_SCHEMA,
            status: "READY",
            path: out.to_string_lossy().to_string(),
            networks: &table.networks,
            stations: table.stations.len(),
            content_sha256: &table.content_sha256,
        })?
    ))
}

fn read_station_table(path: &Path) -> Result<StationTable, Box<dyn Error>> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| err(format!("cannot read station table {}: {e}", path.display())))?;
    let table: StationTable = serde_json::from_str(&text)
        .map_err(|e| err(format!("{} is not a station table: {e}", path.display())))?;
    if table.schema != STATIONS_SCHEMA {
        return Err(err(format!(
            "{} declares schema {:?}, expected {STATIONS_SCHEMA:?}",
            path.display(),
            table.schema
        )));
    }
    let digest = station_rows_digest(&table.stations);
    if digest != table.content_sha256 {
        return Err(err(format!(
            "station table {} has been edited since it was frozen: it states {}, its rows hash \
             to {digest}",
            path.display(),
            table.content_sha256
        )));
    }
    Ok(table)
}

// ----------------------------------------------------------------- fetch

/// The archive's historical range parameters: UTC, unpadded, exactly the
/// spelling the working implementation pins.
fn range_params(start: DateTime<Utc>, end: DateTime<Utc>) -> String {
    format!(
        "year1={}&month1={}&day1={}&hour1={}&minute1={}\
         &year2={}&month2={}&day2={}&hour2={}&minute2={}",
        start.year(),
        start.month(),
        start.day(),
        start.hour(),
        start.minute(),
        end.year(),
        end.month(),
        end.day(),
        end.hour(),
        end.minute()
    )
}

fn fetch_url(archive: &str, stations: &[Station], start: DateTime<Utc>, end: DateTime<Utc>) -> String {
    let mut url = format!("{archive}/cgi-bin/request/asos.py?");
    for station in stations {
        url.push_str("station=");
        url.push_str(&query_encode(&station.station_id));
        url.push('&');
    }
    for column in DATA_COLUMNS {
        url.push_str("data=");
        url.push_str(column);
        url.push('&');
    }
    url.push_str(&range_params(start, end));
    url.push_str(
        "&tz=Etc%2FUTC&format=onlycomma&latlon=yes&elev=yes&missing=empty&trace=T\
         &report_type=3&report_type=4",
    );
    url
}

/// Preserve the historic routine/special default; the scoreboard explicitly
/// asks for HFMETAR as well. IEM report type 1 is the live high-frequency
/// METAR product, distinct from the delayed one-minute archive.
fn select_report_types(url: String, types: &[u8]) -> String {
    let mut url = url.replace("&report_type=3&report_type=4", "");
    for kind in types { url.push_str(&format!("&report_type={kind}")); }
    url
}

/// The one-minute route's request: `asos1min.py` takes its window as two
/// ISO stamps and its variables as `vars=`, and `tz=UTC` is what makes its
/// `valid(UTC)` column UTC. Its `ets` is exclusive (asked for 11:50 to
/// 12:40 it answers 11:50 to 12:39), so the request asks one minute past
/// `--end` to keep the window inclusive, as the METAR route's is.
fn one_minute_url(
    archive: &str,
    stations: &[Station],
    start: DateTime<Utc>,
    end: DateTime<Utc>,
) -> String {
    let mut url = format!("{archive}/cgi-bin/request/asos1min.py?");
    for station in stations {
        url.push_str("station=");
        url.push_str(&query_encode(&station.station_id));
        url.push('&');
    }
    for column in ONE_MINUTE_VARS {
        url.push_str("vars=");
        url.push_str(column);
        url.push('&');
    }
    url.push_str(&format!(
        "sts={}&ets={}",
        start.format("%Y-%m-%dT%H:%MZ"),
        (end + chrono::Duration::minutes(1)).format("%Y-%m-%dT%H:%MZ")
    ));
    url.push_str("&sample=1min&what=download&tz=UTC&delim=comma&gis=yes");
    url
}

fn parse_network_list(raw: &str) -> Result<Vec<String>, Box<dyn Error>> {
    let mut networks: Vec<String> = Vec::new();
    for token in raw.split(',').map(str::trim).filter(|t| !t.is_empty()) {
        if !token.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_') {
            return Err(err(format!(
                "network {token:?} may carry only letters, digits and '_'; it is concatenated \
                 into an archive URL"
            )));
        }
        let upper = token.to_ascii_uppercase();
        if !networks.contains(&upper) {
            networks.push(upper);
        }
    }
    if networks.is_empty() {
        return Err(err("--networks named no network"));
    }
    Ok(networks)
}

/// The same request as [`fetch_url`] with `network=` in place of the
/// station list: the archive filters by its own network membership, which
/// is how a window is pulled for a whole country without a frozen table.
fn fetch_url_networks(archive: &str, networks: &[String], start: DateTime<Utc>, end: DateTime<Utc>) -> String {
    let mut url = format!("{archive}/cgi-bin/request/asos.py?");
    for network in networks {
        url.push_str("network=");
        url.push_str(&query_encode(network));
        url.push('&');
    }
    for column in DATA_COLUMNS {
        url.push_str("data=");
        url.push_str(column);
        url.push('&');
    }
    url.push_str(&range_params(start, end));
    url.push_str(
        "&tz=Etc%2FUTC&format=onlycomma&latlon=yes&elev=yes&missing=empty&trace=T\
         &report_type=3&report_type=4",
    );
    url
}

fn cmd_fetch(options: &Options) -> Result<String, Box<dyn Error>> {
    if let Some(raw) = options.networks.as_deref() {
        if options.one_minute() {
            // The one-minute route answers a request without station= with
            // a validation error, not a CSV.
            return Err(err(
                "--product asos1min is asked for by station: asos1min.py takes no network \
                 query. Freeze a table with `rw_asos stations` and pass --stations",
            ));
        }
        if options.stations.is_some() {
            return Err(err("--networks and --stations are two different bounded queries; give one"));
        }
        return cmd_fetch_networks(options, &parse_network_list(raw)?);
    }
    let table = read_station_table(
        options
            .stations
            .as_deref()
            .ok_or_else(|| err("--stations or --networks is required (a table from `rw_asos stations`, or IEM network ids)"))?,
    )?;
    let (start, end) = options.window()?;
    let out = options.out()?;
    let chunk = options.stations_per_request();
    let fetched_at = seam_time(Utc::now());
    let client = agent();

    // The bounded query is bounded by a station list, and a battery-shaped
    // box freezes enough stations to make that list longer than the endpoint
    // will accept. So the window is fetched in station chunks and the CSV
    // bodies are concatenated; the request stays bounded, and the assembled
    // file is what the digest and the decoder see.
    let pause = std::time::Duration::from_millis(options.request_pause_ms());
    let mut urls: Vec<String> = Vec::new();
    let mut body = String::new();
    let mut rows = 0usize;
    let rule = window_cache_rule(end, Utc::now());
    let mut last_from_network = false;
    for group in table.stations.chunks(chunk.max(1)) {
        if last_from_network && !pause.is_zero() {
            std::thread::sleep(pause);
        }
        let url = if options.one_minute() {
            one_minute_url(options.archive(), group, start, end)
        } else {
            select_report_types(fetch_url(options.archive(), group, start, end), options.report_types())
        };
        let (text, hit) = get_text_cached(&client, &url, "IEM ASOS window", rule, &is_asos_csv)?;
        last_from_network = !hit;
        let mut lines = text.lines().filter(|line| !line.trim().is_empty());
        let header = lines.next().unwrap_or("");
        if !header.starts_with("station,") {
            return Err(err(format!(
                "the archive did not answer with an ASOS CSV: its first line is {header:?}"
            )));
        }
        if body.is_empty() {
            body.push_str(header);
            body.push('\n');
        } else if body.lines().next() != Some(header) {
            return Err(err(format!(
                "the archive answered chunk {} with a different CSV header ({header:?}); \
                 concatenating columns that do not line up would shift every field",
                urls.len() + 1
            )));
        }
        for line in lines {
            body.push_str(line);
            body.push('\n');
            rows += 1;
        }
        urls.push(url);
    }
    if urls.is_empty() {
        return Err(err(
            "the frozen station table is empty; there is no bounded window to request",
        ));
    }
    std::fs::write(out, &body)
        .map_err(|e| err(format!("cannot write {}: {e}", out.display())))?;
    let sha256 = hex_sha256(body.as_bytes());

    #[derive(Serialize)]
    struct Record {
        schema: &'static str,
        status: &'static str,
        path: String,
        url: String,
        request_urls: Vec<String>,
        requests: usize,
        stations_per_request: usize,
        stations_requested: usize,
        report_types: Vec<u8>,
        start: String,
        end: String,
        rows: usize,
        bytes: usize,
        sha256: String,
        fetched_at: String,
    }
    Ok(format!(
        "{}\n",
        serde_json::to_string_pretty(&Record {
            schema: FETCH_SCHEMA,
            status: "READY",
            path: out.to_string_lossy().to_string(),
            url: urls[0].clone(),
            requests: urls.len(),
            request_urls: urls,
            stations_per_request: chunk,
            stations_requested: table.stations.len(),
            report_types: if options.one_minute() { vec![] } else { options.report_types().to_vec() },
            start: seam_time(start),
            end: seam_time(end),
            rows,
            bytes: body.len(),
            sha256,
            fetched_at,
        })?
    ))
}

// ---------------------------------------------------------------- decode

/// One parsed CSV row, still in the archive's own units.
#[derive(Debug, Clone)]
struct RawReport {
    station_id: String,
    valid: DateTime<Utc>,
    tmpf: Option<f64>,
    dwpf: Option<f64>,
    sknt: Option<f64>,
    mslp_hpa: Option<f64>,
    gust_kt: Option<f64>,
    precipitation_in: Option<f64>,
    visibility_sm: Option<f64>,
}

fn split_csv(line: &str) -> Vec<String> {
    rw_obs::csv::parse(line).ok().and_then(|rows|rows.into_iter().next())
        .unwrap_or_default().into_iter().map(|f|f.trim().to_string()).collect()
}

fn parse_csv(text: &str) -> Result<Vec<RawReport>, Box<dyn Error>> {
    parse_product_csv(text, &PRODUCTS[0])
}

/// The CSV the archive answers for `product`, read by that product's own
/// time column. A CSV carrying the other product's time column is refused
/// by name: a one-minute record stamped with the METAR provenance (or the
/// reverse) would tell every reader the wrong cadence.
fn parse_product_csv(text: &str, product: &Product) -> Result<Vec<RawReport>, Box<dyn Error>> {
    let mut lines = text.lines().filter(|line| !line.trim().is_empty());
    let header = lines.next().ok_or_else(|| err("the ASOS CSV is empty"))?;
    if !header.starts_with("station,") {
        return Err(err(format!(
            "the ASOS CSV header does not begin with 'station,': {header:?}"
        )));
    }
    let columns = split_csv(header);
    let index_of = |name: &str| columns.iter().position(|c| c == name);
    let station_at = index_of("station").ok_or_else(|| err("the ASOS CSV has no station column"))?;
    let valid_at = index_of(product.time_column).ok_or_else(|| {
        match PRODUCTS
            .iter()
            .find(|other| other.name != product.name && index_of(other.time_column).is_some())
        {
            Some(other) => err(format!(
                "the CSV's time column is {:?}, the {} product's; decode it with --product {}",
                other.time_column, other.name, other.name
            )),
            None => err(format!("the ASOS CSV has no {} column", product.time_column)),
        }
    })?;
    let tmpf_at = index_of("tmpf");
    let dwpf_at = index_of("dwpf");
    let sknt_at = index_of("sknt");
    let mslp_at = index_of("mslp");
    let gust_at = index_of("gust");
    let precipitation_at = index_of("p01i");
    let visibility_at = index_of("vsby");

    let mut reports = Vec::new();
    for line in lines {
        let fields = split_csv(line);
        let get = |at: Option<usize>| -> Option<f64> {
            let value = fields.get(at?)?;
            if value.is_empty() {
                return None;
            }
            // `M` (missing) and `T` (trace) are not numbers and are not
            // errors; `missing=empty` means they should not appear, and
            // parsing them to None rather than failing keeps one archive
            // quirk from voiding a whole window.
            value.parse::<f64>().ok().filter(|v| v.is_finite())
        };
        let (Some(station), Some(valid)) = (fields.get(station_at), fields.get(valid_at)) else {
            continue;
        };
        if station.is_empty() {
            continue;
        }
        // `tz=Etc/UTC` is in the request, so the naive stamp is UTC. That is
        // the only reason reading it as UTC is sound, and it is why the
        // parameter is not optional in `fetch_url`.
        let parsed = NaiveDateTime::parse_from_str(valid, "%Y-%m-%d %H:%M")
            .or_else(|_| NaiveDateTime::parse_from_str(valid, "%Y-%m-%d %H:%M:%S"));
        let Ok(naive) = parsed else { continue };
        reports.push(RawReport {
            station_id: station.to_ascii_uppercase(),
            valid: Utc.from_utc_datetime(&naive),
            tmpf: get(tmpf_at),
            dwpf: get(dwpf_at),
            sknt: get(sknt_at),
            mslp_hpa: get(mslp_at),
            gust_kt: get(gust_at),
            precipitation_in: get(precipitation_at),
            visibility_sm: get(visibility_at).filter(|v| (0.0..=100.0).contains(v)),
        });
    }
    Ok(reports)
}

fn knots_to_ms(value: f64) -> f64 {
    value * 0.5144444444444445
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SeamReport {
    station_id: String,
    /// The valid time this report serves: the hourly slot it was matched
    /// to, and the leg a consumer binds it to.
    valid_time: String,
    /// When the report was actually taken, the archive's own `valid`
    /// column.  Kept beside the slot because the two differ by up to
    /// `--match-seconds`, and an innovation dated to the slot is dated up
    /// to that far from the instant the instrument read.  Empty on a `v1`
    /// record, which never carried it.
    #[serde(default)]
    observation_time: String,
    values: BTreeMap<String, f64>,
    flags: Vec<String>,
}

/// Which valid time each report serves.
///
/// Every report goes to the ONE valid time nearest it (ties to the earlier
/// one) and only when that is within `match_seconds`; each valid time then
/// takes the nearest of the reports that chose it (ties to the earlier
/// report).  Returns `(target index, report)` pairs in target order.
///
/// The rule used to run the other way round, each valid time taking the
/// nearest report, and nothing marked a report as taken: with a window
/// wider than half the stride one 12:52 report was written under 12:00 and
/// under 13:00, the first of them before it was taken.  Assigning reports
/// to targets makes "used once" a property of the construction rather than
/// a bookkeeping promise.
fn match_reports<'a>(
    candidates: &'a [RawReport],
    targets: &[DateTime<Utc>],
    match_seconds: i64,
) -> Vec<(usize, &'a RawReport)> {
    let mut chosen: Vec<Option<&'a RawReport>> = vec![None; targets.len()];
    for report in candidates {
        let nearest = targets
            .iter()
            .enumerate()
            .map(|(index, target)| ((report.valid - *target).num_seconds().abs(), index))
            .min();
        let Some((distance, index)) = nearest else { continue };
        if distance > match_seconds {
            continue;
        }
        let incumbent = chosen[index];
        let closer = match incumbent {
            None => true,
            Some(current) => {
                let current_distance = (current.valid - targets[index]).num_seconds().abs();
                (distance, report.valid) < (current_distance, current.valid)
            }
        };
        if closer {
            chosen[index] = Some(report);
        }
    }
    chosen
        .into_iter()
        .enumerate()
        .filter_map(|(index, report)| report.map(|r| (index, r)))
        .collect()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ScreenReport {
    temperature_min_k: f64,
    temperature_max_k: f64,
    wind_min_ms: f64,
    wind_max_ms: f64,
    dewpoint_above_temperature_drops: usize,
    range_drops: usize,
    stations_dropped_by_screen: Vec<String>,
    stations_dropped_by_completeness: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SurfaceRecord {
    schema: String,
    status: String,
    provenance: Provenance,
    station_table_sha256: String,
    valid_times: Vec<String>,
    match_seconds: i64,
    min_report_rate: f64,
    max_screen_rate: f64,
    screen: ScreenReport,
    stations: Vec<Station>,
    reports: Vec<SeamReport>,
}

fn cmd_decode(options: &Options) -> Result<String, Box<dyn Error>> {
    let table = read_station_table(
        options
            .stations
            .as_deref()
            .ok_or_else(|| err("--stations is required"))?,
    )?;
    let obs_path = options
        .obs
        .as_deref()
        .ok_or_else(|| err("--obs is required (the CSV from `rw_asos fetch`)"))?;
    let out = options.out()?;
    let (start, end) = options.window()?;
    let step = options.step_minutes();
    let match_seconds = options.match_seconds();
    let min_report_rate = options.min_report_rate.unwrap_or(DEFAULT_MIN_REPORT_RATE);
    let max_screen_rate = options.max_screen_rate.unwrap_or(DEFAULT_MAX_SCREEN_FAILURE_RATE);

    let text = std::fs::read_to_string(obs_path)
        .map_err(|e| err(format!("cannot read {}: {e}", obs_path.display())))?;
    let source_sha = hex_sha256(text.as_bytes());
    let fetched_at = match &options.fetch_record {
        Some(path) => {
            let record: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path)?)?;
            if record.get("schema").and_then(|v| v.as_str()) != Some(FETCH_SCHEMA)
                || record.get("status").and_then(|v| v.as_str()) != Some("READY")
                || record.get("sha256").and_then(|v| v.as_str()) != Some(source_sha.as_str()) {
                return Err(err("fetch record does not bind this CSV; a provenance clock from another source is refused"));
            }
            let clock = record.get("fetched_at").and_then(|v| v.as_str())
                .ok_or_else(|| err("fetch record has no fetched_at acquisition clock"))?;
            seam_time(parse_time(clock)?)
        }
        None => seam_time(Utc::now()),
    };
    let raw = parse_product_csv(&text, options.product())?;

    let mut valid_times = Vec::new();
    let mut when = start;
    while when <= end {
        valid_times.push(when);
        when += chrono::Duration::minutes(i64::from(step));
    }
    if valid_times.is_empty() {
        return Err(err("the window contains no valid time at this stride"));
    }

    let known: std::collections::BTreeSet<String> = table
        .stations
        .iter()
        .map(|s| s.station_id.clone())
        .collect();

    // Group by station, screening as we go.
    let mut by_station: BTreeMap<String, Vec<RawReport>> = BTreeMap::new();
    let mut screened: BTreeMap<String, (usize, usize)> = BTreeMap::new(); // (fired, seen)
    let mut dewpoint_drops = 0usize;
    let mut range_drops = 0usize;
    for report in raw {
        if !known.contains(&report.station_id) {
            // The archive answers with what it has; a station outside the
            // frozen table is not part of this case's set and is not scored
            // into it by accident.
            continue;
        }
        let entry = screened.entry(report.station_id.clone()).or_insert((0, 0));
        entry.1 += 1;
        let mut fired = false;
        if let Some(tmpf) = report.tmpf {
            let kelvin = fahrenheit_to_kelvin(tmpf);
            if !(TEMPERATURE_MIN_K..=TEMPERATURE_MAX_K).contains(&kelvin) {
                fired = true;
                range_drops += 1;
            }
        }
        if let (Some(tmpf), Some(dwpf)) = (report.tmpf, report.dwpf) {
            if dwpf > tmpf {
                fired = true;
                dewpoint_drops += 1;
            }
        }
        if let Some(sknt) = report.sknt {
            let ms = knots_to_ms(sknt);
            if !(WIND_MIN_MS..=WIND_MAX_MS).contains(&ms) {
                fired = true;
                range_drops += 1;
            }
        }
        if fired {
            entry.0 += 1;
            continue;
        }
        by_station.entry(report.station_id.clone()).or_default().push(report);
    }

    let mut dropped_by_screen = Vec::new();
    for (station, (fired, seen)) in &screened {
        if *seen > 0 && (*fired as f64 / *seen as f64) > max_screen_rate {
            dropped_by_screen.push(station.clone());
            by_station.remove(station);
        }
    }

    // Match each station's reports to the valid times: each report serves
    // the one valid time nearest it, so no report is written twice.
    let mut reports = Vec::new();
    let mut matched_hours: BTreeMap<String, usize> = BTreeMap::new();
    for (station, mut candidates) in by_station {
        candidates.sort_by_key(|r| r.valid);
        for (index, best) in match_reports(&candidates, &valid_times, match_seconds) {
            let target = &valid_times[index];
            let mut values = BTreeMap::new();
            if let Some(tmpf) = best.tmpf {
                values.insert("temperature_2m".to_string(), fahrenheit_to_kelvin(tmpf));
            }
            if let Some(dwpf) = best.dwpf {
                values.insert("dewpoint_2m".to_string(), fahrenheit_to_kelvin(dwpf));
            }
            if let Some(sknt) = best.sknt {
                values.insert("wind_speed_10m".to_string(), knots_to_ms(sknt));
            }
            if let Some(hpa) = best.mslp_hpa {
                values.insert("mslp".to_string(), hpa * 100.0);
            }
            // Missing gust is not calm wind. Missing or trace precipitation
            // remains missing, rather than being scored as an observed zero.
            if let Some(gust) = best.gust_kt.filter(|v| (0.0..=250.0).contains(v)) {
                values.insert("wind_gust_10m".into(), knots_to_ms(gust));
            }
            if let Some(rain) = best.precipitation_in.filter(|v| (0.0..=30.0).contains(v)) {
                values.insert("precipitation_1h".into(), rain * 25.4);
            }
            let mut flags = Vec::new();
            if let Some(sm) = best.visibility_sm {
                // ASOS METAR visibility at 10 SM is right censored. Retain
                // the lower bound; never treat it as exact clear-air truth.
                values.insert("visibility".into(), sm.min(10.0) * 1609.344);
                if sm >= 10.0 { flags.push("metar_visibility_ge_10sm".into()); }
            }
            if values.is_empty() {
                // A report that carries no scored variable is not a report
                // for this purpose; emitting it would inflate the
                // completeness rate with empty rows.
                continue;
            }
            *matched_hours.entry(station.clone()).or_insert(0) += 1;
            reports.push(SeamReport {
                station_id: station.clone(),
                valid_time: seam_time(*target),
                observation_time: seam_time(best.valid),
                values,
                flags,
            });
        }
    }

    // Completeness: a station reporting too few of the scored hours is
    // dropped for every arm, symmetrically, so the pairing stays exact.
    let mut dropped_by_completeness = Vec::new();
    let required = (min_report_rate * valid_times.len() as f64).ceil() as usize;
    let mut keep: std::collections::BTreeSet<String> = Default::default();
    for station in &table.stations {
        let matched = matched_hours.get(&station.station_id).copied().unwrap_or(0);
        if matched >= required && matched > 0 {
            keep.insert(station.station_id.clone());
        } else if matched_hours.contains_key(&station.station_id) {
            dropped_by_completeness.push(station.station_id.clone());
        }
    }
    reports.retain(|r| keep.contains(&r.station_id));
    let stations: Vec<Station> = table
        .stations
        .iter()
        .filter(|s| keep.contains(&s.station_id))
        .cloned()
        .collect();
    if stations.is_empty() {
        // The overwhelmingly common cause is a decode window reaching past
        // the window the CSV was fetched for: every station then matches
        // only part of the hours and none clears the completeness bar. Say
        // so with the numbers rather than making the caller guess.
        let best = matched_hours.values().max().copied().unwrap_or(0);
        return Err(err(format!(
            "no station cleared the screens for {} .. {}: {} valid times were requested, the \
             best-covered station matched {best} of them, and {required} are required at \
             --min-report-rate {min_report_rate}. {} station(s) were dropped for completeness \
             and {} by the gross-error screen. If the CSV was fetched for a shorter window \
             than this one, refetch it before lowering the bar",
            seam_time(start),
            seam_time(end),
            valid_times.len(),
            dropped_by_completeness.len(),
            dropped_by_screen.len(),
        )));
    }
    reports.sort_by(|a, b| {
        (a.station_id.as_str(), a.valid_time.as_str())
            .cmp(&(b.station_id.as_str(), b.valid_time.as_str()))
    });

    let record = SurfaceRecord {
        schema: SURFACE_SCHEMA.to_string(),
        status: "READY".to_string(),
        provenance: Provenance::new(
            "asos",
            options.product().provenance,
            rw_obs::absolute_uri(obs_path),
            source_sha,
            fetched_at,
        ),
        station_table_sha256: table.content_sha256.clone(),
        valid_times: valid_times.iter().map(|t| seam_time(*t)).collect(),
        match_seconds,
        min_report_rate,
        max_screen_rate,
        screen: ScreenReport {
            temperature_min_k: TEMPERATURE_MIN_K,
            temperature_max_k: TEMPERATURE_MAX_K,
            wind_min_ms: WIND_MIN_MS,
            wind_max_ms: WIND_MAX_MS,
            dewpoint_above_temperature_drops: dewpoint_drops,
            range_drops,
            stations_dropped_by_screen: dropped_by_screen,
            stations_dropped_by_completeness: dropped_by_completeness,
        },
        stations,
        reports,
    };
    let text = serde_json::to_string_pretty(&record)?;
    std::fs::write(out, format!("{text}\n"))
        .map_err(|e| err(format!("cannot write {}: {e}", out.display())))?;

    #[derive(Serialize)]
    struct Summary<'a> {
        schema: &'static str,
        status: &'static str,
        path: String,
        valid_times: usize,
        stations: usize,
        reports: usize,
        provenance: &'a Provenance,
        station_table_sha256: &'a str,
        screen: &'a ScreenReport,
    }
    Ok(format!(
        "{}\n",
        serde_json::to_string_pretty(&Summary {
            schema: SURFACE_SCHEMA,
            status: "READY",
            path: out.to_string_lossy().to_string(),
            valid_times: record.valid_times.len(),
            stations: record.stations.len(),
            reports: record.reports.len(),
            provenance: &record.provenance,
            station_table_sha256: &record.station_table_sha256,
            screen: &record.screen,
        })?
    ))
}

fn cmd_verify(options: &Options) -> Result<String, Box<dyn Error>> {
    let path = options
        .file
        .as_deref()
        .or(options.out.as_deref())
        .ok_or_else(|| err("verify needs the record path in --file"))?;
    let text = std::fs::read_to_string(path)
        .map_err(|e| err(format!("cannot read {}: {e}", path.display())))?;
    let record: SurfaceRecord = serde_json::from_str(&text)
        .map_err(|e| err(format!("{} is not a surface record: {e}", path.display())))?;
    if record.schema != SURFACE_SCHEMA && record.schema != SURFACE_SCHEMA_V1 {
        return Err(err(format!(
            "{} declares schema {:?}, expected {SURFACE_SCHEMA:?} or {SURFACE_SCHEMA_V1:?}",
            path.display(),
            record.schema
        )));
    }
    let carries_observation_times = record.schema == SURFACE_SCHEMA;
    if carries_observation_times {
        // The two promises a v2 record makes about time: every report says
        // when it was taken, within the match window of the slot it
        // serves, and no observation serves two slots.
        let mut served: std::collections::BTreeSet<(&str, &str)> = Default::default();
        for report in &record.reports {
            let observed = NaiveDateTime::parse_from_str(&report.observation_time, TIME_FORMAT)
                .map(|naive| Utc.from_utc_datetime(&naive))
                .map_err(|_| {
                    err(format!(
                        "the record's report for {} at {} carries observation_time {:?}, \
                         which is not a seam time",
                        report.station_id, report.valid_time, report.observation_time
                    ))
                })?;
            let slot = NaiveDateTime::parse_from_str(&report.valid_time, TIME_FORMAT)
                .map(|naive| Utc.from_utc_datetime(&naive))
                .map_err(|_| {
                    err(format!(
                        "the record's report for {} carries valid_time {:?}, which is not a \
                         seam time",
                        report.station_id, report.valid_time
                    ))
                })?;
            let apart = (observed - slot).num_seconds().abs();
            if apart > record.match_seconds {
                return Err(err(format!(
                    "the record's report for {} was taken at {} and serves {}, {apart} s \
                     apart, beyond the {} s match window the record itself declares",
                    report.station_id,
                    report.observation_time,
                    report.valid_time,
                    record.match_seconds
                )));
            }
            if !served.insert((report.station_id.as_str(), report.observation_time.as_str())) {
                return Err(err(format!(
                    "the record uses {}'s observation of {} twice; a report serves one \
                     valid time",
                    report.station_id, report.observation_time
                )));
            }
        }
    }
    let source = Path::new(&record.provenance.uri);
    let bytes = std::fs::read(source).map_err(|e| {
        err(format!(
            "cannot re-read the source this record was built from ({}): {e}",
            source.display()
        ))
    })?;
    let digest = hex_sha256(&bytes);
    if digest != record.provenance.sha256 {
        return Err(err(format!(
            "the source {} has changed since it was decoded: the record states {}, the bytes \
             hash to {digest}",
            source.display(),
            record.provenance.sha256
        )));
    }
    let known: std::collections::BTreeSet<&str> = record
        .stations
        .iter()
        .map(|s| s.station_id.as_str())
        .collect();
    if let Some(orphan) = record
        .reports
        .iter()
        .find(|r| !known.contains(r.station_id.as_str()))
    {
        return Err(err(format!(
            "the record carries a report for {}, which is not in its own station set",
            orphan.station_id
        )));
    }

    #[derive(Serialize)]
    struct Record<'a> {
        schema: &'static str,
        status: &'static str,
        path: String,
        record_schema: &'a str,
        /// Whether every report carries its own observation_time and was
        /// proved to serve one valid time.  False on a v1 record, whose
        /// reports are dated to their slot and could serve two.
        observation_times_proved: bool,
        source: String,
        source_sha256: String,
        stations: usize,
        reports: usize,
        valid_times: usize,
    }
    Ok(format!(
        "{}\n",
        serde_json::to_string_pretty(&Record {
            schema: VERIFY_SCHEMA,
            status: "PASS",
            path: path.to_string_lossy().to_string(),
            record_schema: &record.schema,
            observation_times_proved: carries_observation_times,
            source: source.to_string_lossy().to_string(),
            source_sha256: digest,
            stations: record.stations.len(),
            reports: record.reports.len(),
            valid_times: record.valid_times.len(),
        })?
    ))
}

fn cmd_fetch_networks(options: &Options, networks: &[String]) -> Result<String, Box<dyn Error>> {
    let (start, end) = options.window()?;
    let out = options.out()?;
    let chunk = options.networks_per_request.unwrap_or(DEFAULT_NETWORKS_PER_REQUEST);
    let fetched_at = seam_time(Utc::now());
    let client = agent();
    let pause = std::time::Duration::from_millis(options.request_pause_ms());
    let started = std::time::Instant::now();
    let mut urls: Vec<String> = Vec::new();
    let mut walls: Vec<f64> = Vec::new();
    let mut body = String::new();
    let mut rows = 0usize;
    let rule = window_cache_rule(end, Utc::now());
    let mut last_from_network = false;
    for group in networks.chunks(chunk.max(1)) {
        if last_from_network && !pause.is_zero() {
            std::thread::sleep(pause);
        }
        let url = select_report_types(fetch_url_networks(options.archive(), group, start, end), options.report_types());
        let t = std::time::Instant::now();
        let (text, hit) = get_text_cached(&client, &url, "IEM ASOS network window", rule, &is_asos_csv)?;
        last_from_network = !hit;
        walls.push(t.elapsed().as_secs_f64());
        let mut lines = text.lines().filter(|line| !line.trim().is_empty());
        let header = lines.next().unwrap_or("");
        if !header.starts_with("station,") {
            return Err(err(format!(
                "the archive did not answer with an ASOS CSV for networks {group:?}: its first line is {header:?}"
            )));
        }
        if body.is_empty() {
            body.push_str(header);
            body.push('\n');
        } else if body.lines().next() != Some(header) {
            return Err(err(format!(
                "the archive answered chunk {} with a different CSV header ({header:?}); \
                 concatenating columns that do not line up would shift every field",
                urls.len() + 1
            )));
        }
        for line in lines {
            body.push_str(line);
            body.push('\n');
            rows += 1;
        }
        urls.push(url);
    }
    std::fs::write(out, &body).map_err(|e| err(format!("cannot write {}: {e}", out.display())))?;
    let sha256 = hex_sha256(body.as_bytes());

    #[derive(Serialize)]
    struct Record {
        schema: &'static str,
        status: &'static str,
        path: String,
        url: String,
        request_urls: Vec<String>,
        request_wall_s: Vec<f64>,
        requests: usize,
        networks_per_request: usize,
        networks_requested: usize,
        report_types: Vec<u8>,
        networks: Vec<String>,
        start: String,
        end: String,
        rows: usize,
        bytes: usize,
        sha256: String,
        fetched_at: String,
        wall_s: f64,
    }
    Ok(format!(
        "{}\n",
        serde_json::to_string_pretty(&Record {
            schema: FETCH_SCHEMA,
            status: "READY",
            path: rw_obs::absolute_uri(out),
            url: urls[0].clone(),
            requests: urls.len(),
            request_urls: urls,
            request_wall_s: walls,
            networks_per_request: chunk,
            networks_requested: networks.len(),
            report_types: options.report_types().to_vec(),
            networks: networks.to_vec(),
            start: seam_time(start),
            end: seam_time(end),
            rows,
            bytes: body.len(),
            sha256,
            fetched_at,
            wall_s: started.elapsed().as_secs_f64(),
        })?
    ))
}

// -------------------------------------------------------------- networks

const NETWORKS_SCHEMA: &str = "gpuwm-obs.asos-networks.v1";

/// The network ids in the archive's `networks.geojson` whose id ends in
/// `ASOS` (every METAR-carrying network, 266 on 2026-09-06), or every id
/// when `--networks all` is asked for.
fn cmd_networks(options: &Options) -> Result<String, Box<dyn Error>> {
    let archive = options.archive().to_string();
    let url = format!("{archive}/geojson/networks.geojson");
    let client = agent();
    let t = std::time::Instant::now();
    let body = get_text(&client, &url, "IEM network list")?;
    let document: serde_json::Value = serde_json::from_str(&body)
        .map_err(|e| err(format!("networks.geojson is not JSON: {e}")))?;
    let features = document
        .get("features")
        .and_then(|f| f.as_array())
        .ok_or_else(|| err("networks.geojson carries no feature list"))?;
    let all = options.networks.as_deref().is_some_and(|n| n.eq_ignore_ascii_case("all"));
    let mut ids: Vec<String> = Vec::new();
    for feature in features {
        let id = feature
            .get("id")
            .and_then(|s| s.as_str())
            .or_else(|| feature.get("properties").and_then(|p| p.get("id")).and_then(|s| s.as_str()));
        let Some(id) = id else { continue };
        if !id.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_') {
            continue;
        }
        if all || id.to_ascii_uppercase().ends_with("ASOS") {
            let upper = id.to_ascii_uppercase();
            if !ids.contains(&upper) {
                ids.push(upper);
            }
        }
    }
    ids.sort();
    if ids.is_empty() {
        return Err(err("networks.geojson names no ASOS network; the layout changed"));
    }
    #[derive(Serialize)]
    struct Record {
        schema: &'static str,
        status: &'static str,
        url: String,
        selection: &'static str,
        count: usize,
        networks: Vec<String>,
        sha256: String,
        fetched_at: String,
        wall_s: f64,
    }
    let record = Record {
        schema: NETWORKS_SCHEMA,
        status: "READY",
        url,
        selection: if all { "all" } else { "ids ending in ASOS" },
        count: ids.len(),
        networks: ids,
        sha256: hex_sha256(body.as_bytes()),
        fetched_at: seam_time(Utc::now()),
        wall_s: t.elapsed().as_secs_f64(),
    };
    let text = format!("{}\n", serde_json::to_string_pretty(&record)?);
    if let Some(out) = options.out.as_deref() {
        std::fs::write(out, &text).map_err(|e| err(format!("cannot write {}: {e}", out.display())))?;
    }
    Ok(text)
}

// ------------------------------------------------------------------- awc

const AWC_CACHE_URL: &str = "https://aviationweather.gov/data/cache/metars.cache.csv.gz";
const AWC_SOURCE: &str = "awc-metar";
const AWC_RECORD_SCHEMA: &str = "gpuwm-obs.asos-awc.v1";

/// One row of the AWC cache in its own units (Celsius, knots, inches of
/// mercury, metres), located by the header's column names.
#[derive(Debug, Clone)]
struct AwcRow {
    station_id: String,
    valid: DateTime<Utc>,
    lat: f64,
    lon: f64,
    elevation_m: f64,
    temp_c: Option<f64>,
    dewpoint_c: Option<f64>,
    wind_dir_deg: Option<f64>,
    wind_speed_kt: Option<f64>,
    altim_in_hg: Option<f64>,
    corrected: bool,
}

/// The cache is a CSV with a five-line preamble (errors, warnings, the
/// query wall, `data source=metars`, the result count) before the header;
/// the header is the first line naming `station_id`.
fn parse_awc_csv(text: &str, counters: &mut TableCounters) -> Result<Vec<AwcRow>, Box<dyn Error>> {
    let mut lines = text.lines().filter(|line| !line.trim().is_empty());
    let mut header: Option<Vec<String>> = None;
    for _ in 0..10 {
        let Some(line) = lines.next() else { break };
        let columns = split_csv(line);
        if columns.iter().any(|c| c == "station_id") && columns.iter().any(|c| c == "observation_time") {
            header = Some(columns);
            break;
        }
    }
    let columns = header.ok_or_else(|| {
        err("the AWC cache has no header naming station_id and observation_time in its first ten lines; the layout changed")
    })?;
    let index_of = |name: &str| -> Result<usize, Box<dyn Error>> {
        columns.iter().position(|c| c == name).ok_or_else(|| {
            err(format!(
                "the AWC cache has no {name} column; this converter needs station_id, observation_time,                  latitude, longitude, elevation_m, temp_c, dewpoint_c, wind_dir_degrees, wind_speed_kt and altim_in_hg"
            ))
        })
    };
    let station_at = index_of("station_id")?;
    let time_at = index_of("observation_time")?;
    let lat_at = index_of("latitude")?;
    let lon_at = index_of("longitude")?;
    let elev_at = index_of("elevation_m")?;
    let temp_at = index_of("temp_c")?;
    let dew_at = index_of("dewpoint_c")?;
    let dir_at = index_of("wind_dir_degrees")?;
    let spd_at = index_of("wind_speed_kt")?;
    let alti_at = index_of("altim_in_hg")?;
    let corrected_at = columns.iter().position(|c| c == "corrected");
    let mut out = Vec::new();
    for line in lines {
        counters.rows_scanned += 1;
        let fields = split_csv(line);
        let get = |at: usize| -> Option<f64> {
            let value = fields.get(at)?;
            if value.is_empty() {
                return None;
            }
            value.parse::<f64>().ok().filter(|v| v.is_finite())
        };
        let (Some(station), Some(time_text)) = (fields.get(station_at), fields.get(time_at)) else {
            counters.rows_malformed += 1;
            continue;
        };
        // The cache spells the instant with milliseconds
        // (2026-09-06T02:02:00.000Z); the fraction is always zero for a
        // METAR and is dropped before the parse.
        let whole = match time_text.find('.') {
            Some(dot) if time_text.ends_with('Z') => format!("{}Z", &time_text[..dot]),
            _ => time_text.to_string(),
        };
        let Ok(valid) = parse_time(&whole) else {
            counters.rows_malformed += 1;
            continue;
        };
        let (Some(lat), Some(lon), Some(elevation)) = (get(lat_at), get(lon_at), get(elev_at)) else {
            counters.rows_without_position += 1;
            continue;
        };
        if lat.abs() > 90.0 || lon < -180.0 || lon > 360.0 {
            counters.rows_without_position += 1;
            continue;
        }
        out.push(AwcRow {
            station_id: station.to_ascii_uppercase(),
            valid,
            lat,
            lon,
            elevation_m: elevation,
            temp_c: get(temp_at),
            dewpoint_c: get(dew_at),
            wind_dir_deg: get(dir_at),
            wind_speed_kt: get(spd_at),
            altim_in_hg: get(alti_at),
            corrected: corrected_at
                .and_then(|i| fields.get(i))
                .is_some_and(|v| v.eq_ignore_ascii_case("TRUE")),
        });
    }
    Ok(out)
}

fn awc_rows_for(raw: &AwcRow, provenance: &RowProvenance, counters: &mut TableCounters, writer: &mut TableWriter) {
    // The same arithmetic as the IEM route, from Celsius instead of
    // Fahrenheit; a corrected report carries COR in its revision.
    let converted = RawTableRow {
        station_id: raw.station_id.clone(),
        valid: raw.valid,
        lon: raw.lon,
        lat: raw.lat,
        elevation_m: raw.elevation_m,
        tmpf: raw.temp_c.map(|c| c * 9.0 / 5.0 + 32.0),
        dwpf: raw.dewpoint_c.map(|c| c * 9.0 / 5.0 + 32.0),
        drct: raw.wind_dir_deg,
        sknt: raw.wind_speed_kt,
        alti: raw.altim_in_hg,
    };
    let mut provenance = provenance.clone();
    if raw.corrected {
        provenance.revision = format!("{}:COR", provenance.revision);
    }
    let before = writer.len();
    table_rows_for(&converted, &provenance, counters, writer);
    // table_rows_for stamps the IEM source; this door's rows are the AWC's.
    for row in writer.rows_mut().iter_mut().skip(before) {
        row.source = AWC_SOURCE.to_string();
    }
}

fn cmd_awc(options: &Options) -> Result<String, Box<dyn Error>> {
    let out = options.out()?;
    let url = options.archive.as_deref().map(|a| a.to_string()).unwrap_or_else(|| AWC_CACHE_URL.to_string());
    let client = agent();
    let fetched_at = Utc::now();
    let t = std::time::Instant::now();
    let mut response = rw_http::get(&client, &url).map_err(|e| err(format!("GET {url} failed: {e}")))?;
    let status = response.status().as_u16();
    if !(200..300).contains(&status) {
        return Err(err(format!("GET {url} answered HTTP {status}")));
    }
    let last_modified = response
        .headers()
        .get("last-modified")
        .and_then(|v| v.to_str().ok())
        .and_then(rw_nexrad::s3::parse_http_date);
    let raw = response
        .body_mut()
        .with_config()
        .limit(rw_obs::net::MAX_RESPONSE_BYTES)
        .read_to_vec()
        .map_err(|e| err(format!("reading {url} failed: {e}")))?;
    let wall = t.elapsed().as_secs_f64();
    if raw.is_empty() {
        return Err(err(format!("{url} answered with an empty body")));
    }
    let raw_sha = hex_sha256(&raw);
    let raw_path = out.with_extension("cache.csv.gz");
    std::fs::write(&raw_path, &raw).map_err(|e| err(format!("cannot write {}: {e}", raw_path.display())))?;
    let (bytes, _) = rw_obs::gunzip_if_wrapped(&raw, "AWC METAR cache")?;
    let text = String::from_utf8_lossy(&bytes).to_string();
    let mut counters = TableCounters::default();
    let rows = parse_awc_csv(&text, &mut counters)?;
    let provenance = RowProvenance::of_source(&raw_sha, last_modified, Some(fetched_at));
    let mut writer = TableWriter::new();
    let mut seen: std::collections::BTreeSet<(String, i64)> = std::collections::BTreeSet::new();
    let mut stations: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    let mut latest: Option<DateTime<Utc>> = None;
    let mut corrected = 0usize;
    for report in &rows {
        if !seen.insert((report.station_id.clone(), report.valid.timestamp())) {
            counters.rows_duplicate_station_time += 1;
            continue;
        }
        counters.reports_kept += 1;
        stations.insert(report.station_id.clone());
        latest = Some(latest.map_or(report.valid, |l| l.max(report.valid)));
        if report.corrected {
            corrected += 1;
        }
        let hour = report.valid.with_minute(0).and_then(|t| t.with_second(0)).unwrap_or(report.valid);
        *counters.reports_by_hour.entry(seam_time(hour)).or_insert(0) += 1;
        awc_rows_for(report, &provenance, &mut counters, &mut writer);
    }
    counters.stations = stations.len();
    let (table_rows, csv_sha, csv_bytes) = writer.write(out)?;
    let mut errors = BTreeMap::new();
    errors.insert(VAR_SURFACE_PRESSURE, ERROR_SURFACE_PRESSURE_PA);
    errors.insert(VAR_TEMPERATURE, ERROR_TEMPERATURE_SURFACE_K);
    errors.insert(VAR_DEWPOINT, ERROR_DEWPOINT_SURFACE_K);
    errors.insert(VAR_WIND_U, ERROR_WIND_SURFACE_M_S);
    errors.insert(VAR_WIND_V, ERROR_WIND_SURFACE_M_S);

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
        url: String,
        cache_path: String,
        cache_bytes: usize,
        cache_sha256: String,
        cache_last_modified: Option<String>,
        fetched_at: String,
        fetch_wall_s: f64,
        corrected_reports: usize,
        errors: BTreeMap<&'static str, f64>,
        counters: TableCounters,
        latest_report: Option<String>,
        /// fetched_at minus the latest report: how far behind real time the
        /// cache was when read (an upper bound on the source's latency).
        latency_behind_real_time_s: Option<i64>,
        latency_basis: &'static str,
        /// The cache's Last-Modified minus the latest report it holds: how
        /// long after the last report the file was rewritten.
        publication_lag_s: Option<i64>,
    }
    let record = Record {
        schema: AWC_RECORD_SCHEMA,
        status: if table_rows > 0 { "READY" } else { "EMPTY" },
        source: AWC_SOURCE,
        table_schema: TABLE_SCHEMA,
        path: rw_obs::absolute_uri(out),
        sha256: csv_sha,
        rows: table_rows,
        bytes: csv_bytes,
        url,
        cache_path: rw_obs::absolute_uri(&raw_path),
        cache_bytes: raw.len(),
        cache_sha256: raw_sha,
        cache_last_modified: last_modified.map(seam_time),
        fetched_at: seam_time(fetched_at),
        fetch_wall_s: wall,
        corrected_reports: corrected,
        errors,
        counters,
        latest_report: latest.map(seam_time),
        latency_behind_real_time_s: latest.map(|l| (fetched_at - l).num_seconds()),
        latency_basis: "fetched_at minus the latest observation_time in the cache (an upper bound)",
        publication_lag_s: match (last_modified, latest) {
            (Some(m), Some(l)) => Some((m - l).num_seconds()),
            _ => None,
        },
    };
    let text = format!("{}\n", serde_json::to_string_pretty(&record)?);
    std::fs::write(out.with_extension("json"), &text).map_err(|e| err(format!("cannot write the record: {e}")))?;
    Ok(text)
}

// ----------------------------------------------------------------- table

#[derive(Debug, Default, Serialize, Clone)]
struct TableCounters {
    rows_scanned: usize,
    rows_malformed: usize,
    rows_outside_window: usize,
    rows_duplicate_station_time: usize,
    rows_without_position: usize,
    values_pressure_out_of_range: usize,
    values_temperature_out_of_range: usize,
    values_dewpoint_out_of_range: usize,
    values_dewpoint_above_temperature: usize,
    values_wind_direction_out_of_range: usize,
    values_wind_variable_direction: usize,
    values_wind_speed_out_of_range: usize,
    values_underivable: usize,
    reports_kept: usize,
    stations: usize,
    rows_by_variable: BTreeMap<String, usize>,
    reports_by_hour: BTreeMap<String, usize>,
}

/// One IEM CSV row in the archive's own units, with the position columns
/// `latlon=yes&elev=yes` add.
#[derive(Debug, Clone)]
struct RawTableRow {
    station_id: String,
    valid: DateTime<Utc>,
    lon: f64,
    lat: f64,
    elevation_m: f64,
    tmpf: Option<f64>,
    dwpf: Option<f64>,
    drct: Option<f64>,
    sknt: Option<f64>,
    alti: Option<f64>,
}

fn parse_table_csv(text: &str, counters: &mut TableCounters) -> Result<Vec<RawTableRow>, Box<dyn Error>> {
    let mut lines = text.lines().filter(|line| !line.trim().is_empty());
    let header = lines.next().ok_or_else(|| err("the ASOS CSV is empty"))?;
    if !header.starts_with("station,") {
        return Err(err(format!("the ASOS CSV header does not begin with 'station,': {header:?}")));
    }
    let columns = split_csv(header);
    let index_of = |name: &str| -> Result<usize, Box<dyn Error>> {
        columns.iter().position(|c| c == name).ok_or_else(|| {
            err(format!(
                "the ASOS CSV has no {name} column; `rw_asos fetch` requests it (latlon=yes, elev=yes, \
                 data=tmpf,dwpf,drct,sknt,alti) and a CSV without it is not one this converter reads"
            ))
        })
    };
    let station_at = index_of("station")?;
    if columns.iter().any(|c| c == "valid(UTC)") {
        // The neutral table's station pressure comes from the altimeter
        // setting, and the one-minute pages carry none.
        return Err(err(
            "this is a one-minute ASOS CSV (its time column is valid(UTC)); the neutral table \
             needs the METAR CSV's altimeter setting. Decode it with `rw_asos decode --product \
             asos1min`",
        ));
    }
    let valid_at = index_of("valid")?;
    let lon_at = index_of("lon")?;
    let lat_at = index_of("lat")?;
    let elev_at = index_of("elevation")?;
    let tmpf_at = index_of("tmpf")?;
    let dwpf_at = index_of("dwpf")?;
    let drct_at = index_of("drct")?;
    let sknt_at = index_of("sknt")?;
    let alti_at = index_of("alti")?;
    let mut out = Vec::new();
    for line in lines {
        counters.rows_scanned += 1;
        let fields = split_csv(line);
        let get = |at: usize| -> Option<f64> {
            let value = fields.get(at)?;
            if value.is_empty() || value == "M" || value == "T" {
                return None;
            }
            value.parse::<f64>().ok().filter(|v| v.is_finite())
        };
        let (Some(station), Some(valid)) = (fields.get(station_at), fields.get(valid_at)) else {
            counters.rows_malformed += 1;
            continue;
        };
        let parsed = NaiveDateTime::parse_from_str(valid, "%Y-%m-%d %H:%M")
            .or_else(|_| NaiveDateTime::parse_from_str(valid, "%Y-%m-%d %H:%M:%S"));
        let Ok(naive) = parsed else {
            counters.rows_malformed += 1;
            continue;
        };
        let (Some(lon), Some(lat), Some(elevation)) = (get(lon_at), get(lat_at), get(elev_at)) else {
            counters.rows_without_position += 1;
            continue;
        };
        if lat.abs() > 90.0 || lon < -180.0 || lon > 360.0 {
            counters.rows_without_position += 1;
            continue;
        }
        out.push(RawTableRow {
            station_id: station.to_ascii_uppercase(),
            valid: Utc.from_utc_datetime(&naive),
            lon,
            lat,
            elevation_m: elevation,
            tmpf: get(tmpf_at),
            dwpf: get(dwpf_at),
            drct: get(drct_at),
            sknt: get(sknt_at),
            alti: get(alti_at),
        });
    }
    Ok(out)
}

fn table_rows_for(
    raw: &RawTableRow,
    provenance: &RowProvenance,
    counters: &mut TableCounters,
    writer: &mut TableWriter,
) {
    let base = TableRow {
        source: TABLE_SOURCE.to_string(),
        station_id: raw.station_id.clone(),
        latitude_deg: raw.lat,
        longitude_deg: raw.lon,
        elevation_m: raw.elevation_m,
        level_pa: None,
        valid_time: raw.valid,
        variable: String::new(),
        value: 0.0,
        error: 0.0,
        provenance: provenance.clone(),
    };
    match raw.alti {
        Some(alti) => {
            // The altimeter setting is the ISA sea-level reduction of the
            // station pressure by definition, so inverting it is exact and
            // the row is a station pressure, labelled with its route.
            let pa = altimeter_inhg_to_station_pa(alti, raw.elevation_m);
            if GROSS_SURFACE_PRESSURE_PA.0 <= pa && pa <= GROSS_SURFACE_PRESSURE_PA.1 {
                let mut r = base.clone();
                r.variable = VAR_SURFACE_PRESSURE.to_string();
                r.value = pa;
                r.error = ERROR_SURFACE_PRESSURE_PA;
                r.provenance = provenance.measuring(MEAS_STATION_PRESSURE_FROM_ALTIMETER);
                writer.push(r, &mut counters.rows_by_variable);
            } else {
                counters.values_pressure_out_of_range += 1;
            }
        }
        None => counters.values_underivable += 1,
    }
    let t_k = raw.tmpf.map(fahrenheit_to_kelvin);
    match t_k {
        Some(t) if GROSS_TEMPERATURE_K.0 <= t && t <= GROSS_TEMPERATURE_K.1 => {
            let mut r = base.clone();
            r.variable = VAR_TEMPERATURE.to_string();
            r.value = t;
            r.error = ERROR_TEMPERATURE_SURFACE_K;
            r.provenance = provenance.measuring(MEAS_SCREEN_TEMPERATURE_2M);
            writer.push(r, &mut counters.rows_by_variable);
        }
        Some(_) => counters.values_temperature_out_of_range += 1,
        None => counters.values_underivable += 1,
    }
    match raw.dwpf.map(fahrenheit_to_kelvin) {
        Some(td) if !(GROSS_DEWPOINT_K.0 <= td && td <= GROSS_DEWPOINT_K.1) => {
            counters.values_dewpoint_out_of_range += 1
        }
        Some(td) => {
            // The archive reports both to a tenth of a degree Fahrenheit;
            // a dewpoint above the temperature by more than that rounding
            // is not a moist report, it is a sensor fault.
            if matches!(t_k, Some(t) if td > t + 0.06) {
                counters.values_dewpoint_above_temperature += 1;
            } else {
                let mut r = base.clone();
                r.variable = VAR_DEWPOINT.to_string();
                r.value = td;
                r.error = ERROR_DEWPOINT_SURFACE_K;
                r.provenance = provenance.measuring(MEAS_SCREEN_DEWPOINT_2M);
                writer.push(r, &mut counters.rows_by_variable);
            }
        }
        None => counters.values_underivable += 1,
    }
    match (raw.drct, raw.sknt) {
        (Some(dir), Some(kt)) => {
            let speed = knots_to_m_s(kt);
            if !(0.0..=360.0).contains(&dir) {
                counters.values_wind_direction_out_of_range += 1;
            } else if speed > 0.0 && dir == 0.0 {
                // Direction 0 with a nonzero speed is the archive's
                // variable-direction wind (true north is 360); a northerly
                // of that speed would be an error of up to 180 degrees.
                counters.values_wind_variable_direction += 1;
            } else if !(GROSS_WIND_M_S.0..=GROSS_WIND_M_S.1).contains(&speed) {
                counters.values_wind_speed_out_of_range += 1;
            } else {
                let (u, v) = wind_components(dir, speed);
                let mut ru = base.clone();
                ru.variable = VAR_WIND_U.to_string();
                ru.value = u;
                ru.error = ERROR_WIND_SURFACE_M_S;
                ru.provenance = provenance.measuring(MEAS_ANEMOMETER_WIND_10M);
                writer.push(ru, &mut counters.rows_by_variable);
                let mut rv = base.clone();
                rv.variable = VAR_WIND_V.to_string();
                rv.value = v;
                rv.error = ERROR_WIND_SURFACE_M_S;
                rv.provenance = provenance.measuring(MEAS_ANEMOMETER_WIND_10M);
                writer.push(rv, &mut counters.rows_by_variable);
            }
        }
        _ => counters.values_underivable += 1,
    }
}

fn cmd_table(options: &Options) -> Result<String, Box<dyn Error>> {
    let obs_path = options
        .obs
        .as_deref()
        .ok_or_else(|| err("--obs is required (the CSV from `rw_asos fetch`)"))?;
    let out = options.out()?;
    let start = match options.start.as_deref() {
        Some(raw) => Some(parse_time(raw)?),
        None => None,
    };
    let end = match options.end.as_deref() {
        Some(raw) => Some(parse_time(raw)?),
        None => None,
    };
    if let (Some(s), Some(e)) = (start, end) {
        if e < s {
            return Err(err(format!("--end {} precedes --start {}", seam_time(e), seam_time(s))));
        }
    }
    let fetched_at: Option<DateTime<Utc>> = match &options.fetch_record {
        Some(path) => {
            let text = std::fs::read_to_string(path)
                .map_err(|e| err(format!("cannot read {}: {e}", path.display())))?;
            let record: serde_json::Value = serde_json::from_str(&text)
                .map_err(|e| err(format!("{} is not JSON: {e}", path.display())))?;
            if record.get("schema").and_then(|s| s.as_str()) != Some(FETCH_SCHEMA) {
                return Err(err(format!("{} does not declare {FETCH_SCHEMA}", path.display())));
            }
            record
                .get("fetched_at")
                .and_then(|s| s.as_str())
                .and_then(|s| parse_time(&format!("{s}Z")).ok())
        }
        None => None,
    };
    let text = std::fs::read_to_string(obs_path)
        .map_err(|e| err(format!("cannot read {}: {e}", obs_path.display())))?;
    let source_sha = hex_sha256(text.as_bytes());
    // The IEM archive states no publication time per report, so the row's
    // published_time stays empty; its receipt is the fetch record's instant.
    let provenance = RowProvenance::of_source(&source_sha, None, fetched_at);
    let mut counters = TableCounters::default();
    let raw = parse_table_csv(&text, &mut counters)?;
    let mut seen: std::collections::BTreeSet<(String, i64)> = std::collections::BTreeSet::new();
    let mut stations: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    let mut writer = TableWriter::new();
    let mut latest: Option<DateTime<Utc>> = None;
    for report in &raw {
        if start.is_some_and(|s| report.valid < s) || end.is_some_and(|e| report.valid > e) {
            counters.rows_outside_window += 1;
            continue;
        }
        if !seen.insert((report.station_id.clone(), report.valid.timestamp())) {
            counters.rows_duplicate_station_time += 1;
            continue;
        }
        counters.reports_kept += 1;
        stations.insert(report.station_id.clone());
        latest = Some(latest.map_or(report.valid, |l| l.max(report.valid)));
        let hour = report
            .valid
            .with_minute(0)
            .and_then(|t| t.with_second(0))
            .unwrap_or(report.valid);
        *counters.reports_by_hour.entry(seam_time(hour)).or_insert(0) += 1;
        table_rows_for(report, &provenance, &mut counters, &mut writer);
    }
    counters.stations = stations.len();
    let (rows, csv_sha, csv_bytes) = writer.write(out)?;
    let mut errors = BTreeMap::new();
    errors.insert(VAR_SURFACE_PRESSURE, ERROR_SURFACE_PRESSURE_PA);
    errors.insert(VAR_TEMPERATURE, ERROR_TEMPERATURE_SURFACE_K);
    errors.insert(VAR_DEWPOINT, ERROR_DEWPOINT_SURFACE_K);
    errors.insert(VAR_WIND_U, ERROR_WIND_SURFACE_M_S);
    errors.insert(VAR_WIND_V, ERROR_WIND_SURFACE_M_S);

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
        obs_csv: String,
        obs_csv_sha256: String,
        start: Option<String>,
        end: Option<String>,
        errors: BTreeMap<&'static str, f64>,
        counters: TableCounters,
        /// The fetch's own instant minus the latest report it returned: an
        /// UPPER bound on how far behind real time the archive publishes,
        /// because the fetch may have run long after the reports.
        latency_upper_bound_s: Option<i64>,
        latency_basis: &'static str,
    }
    let record = Record {
        schema: TABLE_RECORD_SCHEMA,
        status: if rows > 0 { "READY" } else { "EMPTY" },
        source: TABLE_SOURCE,
        table_schema: TABLE_SCHEMA,
        path: rw_obs::absolute_uri(out),
        sha256: csv_sha,
        rows,
        bytes: csv_bytes,
        obs_csv: rw_obs::absolute_uri(obs_path),
        obs_csv_sha256: source_sha,
        start: start.map(seam_time),
        end: end.map(seam_time),
        errors,
        counters,
        latency_upper_bound_s: match (fetched_at, latest) {
            (Some(f), Some(l)) => Some((f - l).num_seconds()),
            _ => None,
        },
        latency_basis: "fetch record's fetched_at minus the latest report kept (an upper bound)",
    };
    let text = format!("{}\n", serde_json::to_string_pretty(&record)?);
    std::fs::write(out.with_extension("json"), &text)
        .map_err(|e| err(format!("cannot write the table record: {e}")))?;
    Ok(text)
}

#[cfg(test)]
mod table_tests {
    use super::*;
    #[test]
    fn live_high_frequency_types_are_explicit_and_defaults_remain_routine_special() {
        let parse = |args: &[&str]| Options::parse(&args.iter().map(|s|s.to_string()).collect::<Vec<_>>());
        assert_eq!(parse(&[]).unwrap().report_types(), &[3,4]);
        let options = parse(&["--report-types","4,1,3,1"]).unwrap();
        assert_eq!(options.report_types(), &[1,3,4]);
        let start=parse_time("2026-10-01T00:00:00Z").unwrap();
        let url=select_report_types(fetch_url_networks(DEFAULT_ARCHIVE,&["CA_ASOS".into()],start,start),options.report_types());
        assert!(url.contains("/cgi-bin/request/asos.py?"));
        assert_eq!(url.matches("report_type=1").count(),1);
        assert_eq!(url.matches("report_type=3").count(),1);
        assert_eq!(url.matches("report_type=4").count(),1);
        for raw in ["0","2","5","","1,x"] { assert!(parse(&["--report-types",raw]).is_err()); }
        assert!(parse(&["--product","asos1min","--report-types","1,3,4"]).is_err());
    }

    const CSV: &str = "station,valid,lon,lat,elevation,tmpf,dwpf,drct,sknt,gust,alti,mslp,p01i\n\
YPXM,2026-09-01 00:00,105.6875,-10.4528,259.00,77.00,73.40,100.00,11.00,,30.03,,0.00\n\
YPXM,2026-09-01 00:00,105.6875,-10.4528,259.00,77.00,73.40,100.00,11.00,,30.03,,0.00\n\
EGGP,2026-09-01 00:20,-2.8500,53.3333,26.00,57.20,59.00,0.00,7.00,,30.06,,\n\
EGNM,2026-09-01 01:20,-1.6653,53.8618,208.00,,,280.00,0.00,,,,\n";

    #[test]
    fn the_iem_csv_converts_to_neutral_rows_with_the_screens_counted() {
        let mut c = TableCounters::default();
        let raw = parse_table_csv(CSV, &mut c).unwrap();
        assert_eq!(raw.len(), 4);
        let mut w = TableWriter::new();
        let provenance = RowProvenance::of_source("096a3cbdffc71e24e264d99fa3f4a81f", None, None);
        for r in &raw {
            table_rows_for(r, &provenance, &mut c, &mut w);
        }
        let rows = w.rows();
        assert!(rows.iter().all(|r| r.provenance.revision == "096a3cbdffc7"));
        assert_eq!(
            rows.iter().find(|r| r.variable == VAR_SURFACE_PRESSURE).unwrap().provenance.measurement,
            MEAS_STATION_PRESSURE_FROM_ALTIMETER
        );
        // YPXM twice (the duplicate is the caller's to drop): ps, T, Td, u, v
        // (x2 = 10); EGGP: ps, T, no Td (above T), wind variable-direction
        // (0 deg, 7 kt) = 2; EGNM: no alti, no T, no Td, wind 280 at 0 kt = u, v = 2
        assert_eq!(rows.len(), 14, "{c:?}");
        assert_eq!(c.values_dewpoint_above_temperature, 1);
        assert_eq!(c.values_wind_variable_direction, 1);
        let ps = rows.iter().find(|r| r.station_id == "YPXM" && r.variable == VAR_SURFACE_PRESSURE).unwrap();
        assert!((ps.value - altimeter_inhg_to_station_pa(30.03, 259.0)).abs() < 1e-9);
        assert!(ps.value < 101_000.0 && ps.value > 98_000.0, "{}", ps.value);
        let t = rows.iter().find(|r| r.station_id == "YPXM" && r.variable == VAR_TEMPERATURE).unwrap();
        assert!((t.value - 298.15).abs() < 1e-9);
        let u = rows.iter().find(|r| r.station_id == "YPXM" && r.variable == VAR_WIND_U).unwrap();
        // 100 deg at 11 kt: u = -5.659*sin(100)
        assert!((u.value - (-(11.0 * 0.514444) * 100.0f64.to_radians().sin())).abs() < 1e-9);
    }

    #[test]
    fn the_awc_cache_parses_past_its_preamble_and_marks_corrections() {
        let text = "No errors\nNo warnings\n12 ms\ndata source=metars\n3 results\n\
raw_text,station_id,observation_time,latitude,longitude,temp_c,dewpoint_c,wind_dir_degrees,wind_speed_kt,wind_gust_kt,visibility_statute_mi,altim_in_hg,sea_level_pressure_mb,corrected,auto,elevation_m\n\
\"METAR KDEN 060153Z ...\",KDEN,2026-09-06T01:53:00.000Z,39.85,-104.65,21.1,7.2,180,8,,10,30.12,1015.2,FALSE,TRUE,1656.0\n\
EGLL 060150Z COR ...,EGLL,2026-09-06T01:50:00Z,51.48,-0.46,15.0,12.0,240,6,,6.2,29.94,,TRUE,,25.0\n\
XXXX 060150Z ...,XXXX,2026-09-06T01:50:00Z,,,15.0,12.0,240,6,,6.2,29.94,,FALSE,,25.0\n";
        let mut c = TableCounters::default();
        let rows = parse_awc_csv(text, &mut c).unwrap();
        assert_eq!(rows.len(), 2, "{c:?}");
        assert_eq!(c.rows_without_position, 1);
        assert!(rows[1].corrected && !rows[0].corrected);
        let provenance = RowProvenance::of_source("deadbeefcafe0123", None, None);
        let mut w = TableWriter::new();
        for r in &rows {
            awc_rows_for(r, &provenance, &mut c, &mut w);
        }
        let out = w.rows();
        assert_eq!(out.len(), 10);
        assert!(out.iter().all(|r| r.source == AWC_SOURCE));
        let den_t = out.iter().find(|r| r.station_id == "KDEN" && r.variable == VAR_TEMPERATURE).unwrap();
        assert!((den_t.value - (21.1 + 273.15)).abs() < 1e-9, "{}", den_t.value);
        assert_eq!(den_t.provenance.revision, "deadbeefcafe");
        let egll_p = out.iter().find(|r| r.station_id == "EGLL" && r.variable == VAR_SURFACE_PRESSURE).unwrap();
        assert_eq!(egll_p.provenance.revision, "deadbeefcafe:COR");
        assert!((egll_p.value - altimeter_inhg_to_station_pa(29.94, 25.0)).abs() < 1e-9);
    }

    #[test]
    fn network_urls_name_every_network_and_the_pinned_parameters() {
        let start = parse_time("2026-08-31T17:00:00Z").unwrap();
        let end = parse_time("2026-09-02T01:00:00Z").unwrap();
        let url = fetch_url_networks(DEFAULT_ARCHIVE, &["GB__ASOS".to_string(), "FR__ASOS".to_string()], start, end);
        assert!(url.contains("network=GB__ASOS&network=FR__ASOS&"));
        for pinned in ["tz=Etc%2FUTC", "format=onlycomma", "latlon=yes", "elev=yes", "data=alti", "report_type=4", "year1=2026&month1=8&day1=31&hour1=17"] {
            assert!(url.contains(pinned), "{url}");
        }
        assert!(parse_network_list("gb__asos, GB__ASOS,fr__asos").unwrap() == vec!["GB__ASOS", "FR__ASOS"]);
        assert!(parse_network_list("GB__ASOS;rm").is_err());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn report(hour: u32, minute: u32) -> RawReport {
        RawReport {
            station_id: "AAAA".to_string(),
            valid: Utc.with_ymd_and_hms(2026, 9, 19, hour, minute, 0).unwrap(),
            tmpf: Some(70.0),
            dwpf: None,
            sknt: None,
            mslp_hpa: None,
            gust_kt: None,
            precipitation_in: None,
            visibility_sm: None,
        }
    }

    fn targets(hours: &[u32]) -> Vec<DateTime<Utc>> {
        hours
            .iter()
            .map(|h| Utc.with_ymd_and_hms(2026, 9, 19, *h, 0, 0).unwrap())
            .collect()
    }

    #[test]
    fn one_report_serves_one_valid_time_and_only_the_nearest() {
        // A single 12:52 report against 12:00 and 13:00 with an hour's
        // window: the rule that took the nearest report for every valid
        // time wrote it under both, the first of them before it was taken.
        let reports = vec![report(12, 52)];
        let matched = match_reports(&reports, &targets(&[12, 13]), 3600);
        assert_eq!(matched.len(), 1);
        assert_eq!(matched[0].0, 1, "13:00 is the nearest slot to 12:52");

        // Inside the default ten-minute window it serves nothing at all:
        // 12:52 is eight minutes from 13:00 and fifty-two from 12:00.
        let matched = match_reports(&reports, &targets(&[12, 13]), 600);
        assert_eq!(matched.len(), 1);
        assert_eq!(matched[0].0, 1);
        let matched = match_reports(&reports, &targets(&[12]), 600);
        assert!(matched.is_empty());
    }

    #[test]
    fn a_valid_time_takes_the_nearest_of_the_reports_that_chose_it() {
        // 12:52 and 12:58 both choose 13:00; 13:00 takes 12:58, and 12:00
        // is not handed the leftover 12:52 it is fifty-two minutes from.
        let reports = vec![report(12, 52), report(12, 58)];
        let matched = match_reports(&reports, &targets(&[12, 13]), 3600);
        assert_eq!(matched.len(), 1);
        assert_eq!(matched[0].0, 1);
        assert_eq!(matched[0].1.valid.minute(), 58);

        // 11:57 chooses 12:00 and 12:52 chooses 13:00: two slots, two
        // different observations, each used once.
        let reports = vec![report(11, 57), report(12, 52)];
        let matched = match_reports(&reports, &targets(&[12, 13]), 3600);
        assert_eq!(matched.len(), 2);
        assert_eq!((matched[0].0, matched[0].1.valid.minute()), (0, 57));
        assert_eq!((matched[1].0, matched[1].1.valid.minute()), (1, 52));

        // Equidistant between two slots goes to the earlier slot; two
        // reports equidistant from one slot leaves the earlier report.
        let reports = vec![report(12, 30)];
        let matched = match_reports(&reports, &targets(&[12, 13]), 3600);
        assert_eq!(matched[0].0, 0);
        let reports = vec![report(12, 55), report(13, 5)];
        let matched = match_reports(&reports, &targets(&[13]), 3600);
        assert_eq!(matched.len(), 1);
        assert_eq!(matched[0].1.valid.minute(), 55);
    }

    #[test]
    fn help_version_and_abi_are_stable_surfaces() {
        assert!(run(&[]).unwrap().contains("usage: rw_asos"));
        let help = run(&["--help".to_string()]).unwrap();
        for subcommand in ["stations", "fetch", "decode", "verify"] {
            assert!(help.contains(subcommand), "usage must document {subcommand}");
        }
        assert!(run(&["--version".to_string()]).unwrap().starts_with("rw_asos "));
        assert_eq!(run(&["--abi".to_string()]).unwrap(), format!("{ABI_MARKER}\n"));
    }

    #[test]
    fn unknown_subcommands_and_options_fail_closed() {
        assert!(run(&["sound".to_string()]).is_err());
        assert!(Options::parse(&["--nope".to_string()]).is_err());
        assert!(Options::parse(&["--min-report-rate".to_string(), "2".to_string()]).is_err());
        assert!(Options::parse(&["--step-hours".to_string(), "0".to_string()]).is_err());
    }

    #[test]
    fn the_range_parameters_are_unpadded_utc_exactly_as_the_archive_wants() {
        let start = parse_time("2024-05-21T20:30:00Z").unwrap();
        let end = parse_time("2024-05-21T22:30:00Z").unwrap();
        assert_eq!(
            range_params(start, end),
            "year1=2024&month1=5&day1=21&hour1=20&minute1=30\
             &year2=2024&month2=5&day2=21&hour2=22&minute2=30"
        );
    }

    #[test]
    fn the_query_is_bounded_by_station_and_carries_the_pinned_parameters() {
        let stations = vec![
            Station {
                station_id: "DSM".to_string(),
                name: String::new(),
                latitude: 41.53,
                longitude: -93.65,
                elevation_m: 294.0,
                network: "IA_ASOS".to_string(),
                state: "IA".to_string(),
            },
            Station {
                station_id: "DMX".to_string(),
                name: String::new(),
                latitude: 41.73,
                longitude: -93.72,
                elevation_m: 306.0,
                network: "IA_ASOS".to_string(),
                state: "IA".to_string(),
            },
        ];
        let url = fetch_url(
            DEFAULT_ARCHIVE,
            &stations,
            parse_time("2024-05-21T20:00:00Z").unwrap(),
            parse_time("2024-05-21T22:00:00Z").unwrap(),
        );
        assert!(url.contains("station=DSM"), "{url}");
        assert!(url.contains("station=DMX"), "{url}");
        for pinned in [
            "tz=Etc%2FUTC",
            "format=onlycomma",
            "missing=empty",
            "latlon=yes",
            "elev=yes",
            "report_type=3",
            "report_type=4",
            "data=tmpf",
            "data=mslp",
            "data=p01i",
        ] {
            assert!(url.contains(pinned), "the request must carry {pinned}: {url}");
        }
    }

    #[test]
    fn a_battery_shaped_station_list_is_split_into_requests_the_endpoint_accepts() {
        // Measured against the archive 2026-08-04: 642 stations answered 200
        // and 698 answered HTTP 414, so a table this size must go out as
        // several bounded requests rather than one long one.
        let stations: Vec<Station> = (0..802)
            .map(|index| Station {
                station_id: format!("S{index:03}"),
                name: String::new(),
                latitude: 41.0,
                longitude: -94.0,
                elevation_m: 300.0,
                network: "IA_ASOS".to_string(),
                state: "IA".to_string(),
            })
            .collect();
        let chunk = DEFAULT_STATIONS_PER_REQUEST;
        let groups: Vec<&[Station]> = stations.chunks(chunk).collect();
        assert_eq!(groups.len(), 3, "802 stations at {chunk} per request");
        let start = parse_time("2024-05-21T12:00:00Z").unwrap();
        let end = parse_time("2024-05-22T12:00:00Z").unwrap();
        let mut named = 0usize;
        for group in &groups {
            assert!(group.len() <= chunk);
            let url = fetch_url(DEFAULT_ARCHIVE, group, start, end);
            // Every chunk carries the whole pinned parameter set, not just
            // the first: a chunk fetched under different columns would
            // concatenate into a CSV whose rows do not line up.
            for pinned in ["tz=Etc%2FUTC", "format=onlycomma", "data=tmpf", "report_type=4"] {
                assert!(url.contains(pinned), "chunk request must carry {pinned}");
            }
            for station in group.iter() {
                assert!(url.contains(&format!("station={}", station.station_id)));
                named += 1;
            }
        }
        assert_eq!(named, stations.len(), "every frozen station is requested once");

        // The chunk size is an option, and a zero would loop forever rather
        // than fetch nothing, so it is refused at parse time.
        assert_eq!(Options::default().stations_per_request(), chunk);
        assert_eq!(
            Options::parse(&["--stations-per-request".to_string(), "120".to_string()])
                .unwrap()
                .stations_per_request(),
            120
        );
        for hostile in ["0", "-1", "many"] {
            assert!(
                Options::parse(&["--stations-per-request".to_string(), hostile.to_string()])
                    .is_err(),
                "{hostile:?} must be refused"
            );
        }
    }

    #[test]
    fn units_convert_to_the_ones_the_seam_pins() {
        // Freezing and boiling, and a knot that is exactly a knot.
        assert!((fahrenheit_to_kelvin(32.0) - 273.15).abs() < 1e-9);
        assert!((fahrenheit_to_kelvin(212.0) - 373.15).abs() < 1e-9);
        assert!((fahrenheit_to_kelvin(80.0) - 299.8166666666667).abs() < 1e-9);
        assert!((knots_to_ms(1.0) - 0.5144444444444445).abs() < 1e-12);
        assert!((knots_to_ms(0.0)).abs() < 1e-12);
        // The screen's bounds are the spec's, expressed in seam units.
        assert!((TEMPERATURE_MIN_K - fahrenheit_to_kelvin(-40.0)).abs() < 1e-9);
        assert!((TEMPERATURE_MAX_K - 328.15).abs() < 1e-9);
    }

    fn one_minute_station() -> Station {
        Station {
            station_id: "AAA".into(),
            name: "STATION A".into(),
            latitude: 40.0,
            longitude: -100.0,
            elevation_m: 500.0,
            network: "XX_ASOS".into(),
            state: "XX".into(),
        }
    }

    #[test]
    fn the_one_minute_route_asks_asos1min_for_the_window_in_utc() {
        let start = Utc.with_ymd_and_hms(2025, 5, 1, 17, 0, 0).unwrap();
        let end = Utc.with_ymd_and_hms(2025, 5, 1, 19, 10, 0).unwrap();
        let url = one_minute_url("https://example.org", &[one_minute_station()], start, end);
        assert!(url.starts_with("https://example.org/cgi-bin/request/asos1min.py?station=AAA&"));
        for pinned in [
            "vars=tmpf&",
            "vars=dwpf&",
            "vars=sknt&",
            "sts=2025-05-01T17:00Z",
            "ets=2025-05-01T19:11Z",
            "sample=1min",
            "tz=UTC",
            "delim=comma",
        ] {
            assert!(url.contains(pinned), "{pinned} missing from {url}");
        }
    }

    #[test]
    fn a_one_minute_csv_reads_by_its_own_time_column_and_refuses_the_metar_label() {
        let one_minute = &PRODUCTS[1];
        let text = "station,station_name,lat,lon,valid(UTC),tmpf,dwpf,sknt,drct\n\
                    AAA,STATION A,40.0,-100.0,2025-05-01 17:01,88,66,12,150\n\
                    AAA,STATION A,40.0,-100.0,2025-05-01 17:02,88,,12,150\n";
        let rows = parse_product_csv(text, one_minute).unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].valid, Utc.with_ymd_and_hms(2025, 5, 1, 17, 1, 0).unwrap());
        assert_eq!(rows[0].tmpf, Some(88.0));
        assert_eq!(rows[0].dwpf, Some(66.0));
        assert_eq!(rows[1].dwpf, None);
        assert_eq!(rows[0].mslp_hpa, None);
        let refused = parse_product_csv(text, &PRODUCTS[0]).unwrap_err().to_string();
        assert!(refused.contains("--product asos1min"), "{refused}");
        let metar = "station,valid,tmpf,dwpf,sknt,mslp\nAAA,2025-05-01 16:51,86,65,10,1012.0\n";
        let refused = parse_product_csv(metar, one_minute).unwrap_err().to_string();
        assert!(refused.contains("--product metar"), "{refused}");
        let mut counters = TableCounters::default();
        let refused = parse_table_csv(text, &mut counters).unwrap_err().to_string();
        assert!(refused.contains("--product"), "{refused}");
    }

    #[test]
    fn product_is_metar_or_asos1min_and_sets_the_stride() {
        let parse = |words: &[&str]| {
            let args: Vec<String> = words.iter().map(|s| s.to_string()).collect();
            Options::parse(&args)
        };
        let one_minute = parse(&["--product", "asos1min"]).unwrap();
        assert!(one_minute.one_minute());
        assert_eq!(one_minute.step_minutes(), 1);
        assert_eq!(one_minute.match_seconds(), 30);
        let metar = parse(&["--product", "metar"]).unwrap();
        assert!(!metar.one_minute());
        assert_eq!(metar.step_minutes(), 60);
        assert_eq!(metar.match_seconds(), DEFAULT_MATCH_SECONDS);
        assert!(!parse(&[]).unwrap().one_minute());
        assert_eq!(parse(&["--step-hours", "3"]).unwrap().step_minutes(), 180);
        assert_eq!(parse(&["--product", "asos1min", "--step-minutes", "5"]).unwrap().step_minutes(), 5);
        assert!(parse(&["--product", "hads"]).is_err());
        assert!(parse(&["--step-minutes", "0"]).is_err());
        assert!(parse(&["--step-hours", "1", "--step-minutes", "5"]).is_err());
        let refused = cmd_fetch(&parse(&["--product", "asos1min", "--networks", "XX_ASOS"]).unwrap())
            .unwrap_err()
            .to_string();
        assert!(refused.contains("by station"), "{refused}");
    }

    #[test]
    fn a_one_minute_decode_serves_every_minute_with_its_own_report() {
        let dir = std::env::temp_dir().join(format!("rw_asos_1min_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let station = one_minute_station();
        let table = StationTable {
            schema: STATIONS_SCHEMA.to_string(),
            status: "READY".to_string(),
            archive: DEFAULT_ARCHIVE.to_string(),
            networks: vec!["XX_ASOS".to_string()],
            frozen_at: "2025-05-01T00:00:00".to_string(),
            content_sha256: station_rows_digest(std::slice::from_ref(&station)),
            stations: vec![station],
        };
        let stations = dir.join("stations.json");
        std::fs::write(&stations, serde_json::to_string(&table).unwrap()).unwrap();
        let csv = dir.join("obs.csv");
        std::fs::write(
            &csv,
            "station,station_name,lat,lon,valid(UTC),tmpf,dwpf,sknt,drct
             AAA,STATION A,40.0,-100.0,2025-05-01 17:00,80,60,10,180
             AAA,STATION A,40.0,-100.0,2025-05-01 17:01,80,61,10,180
             AAA,STATION A,40.0,-100.0,2025-05-01 17:03,81,62,11,180
             AAA,STATION A,40.0,-100.0,2025-05-01 17:04,81,62,11,180
",
        )
        .unwrap();
        let out = dir.join("surface.json");
        let args: Vec<String> = [
            "--product", "asos1min", "--stations", stations.to_str().unwrap(),
            "--obs", csv.to_str().unwrap(), "--start", "2025-05-01T17:00:00Z",
            "--end", "2025-05-01T17:04:00Z", "--out", out.to_str().unwrap(),
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        cmd_decode(&Options::parse(&args).unwrap()).unwrap();
        let record: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&out).unwrap()).unwrap();
        let clock_receipt = dir.join("fetch.json");
        std::fs::write(&clock_receipt, serde_json::to_vec(&serde_json::json!({
            "schema": FETCH_SCHEMA, "status": "READY", "sha256": hex_sha256(&std::fs::read(&csv).unwrap()),
            "fetched_at": "2025-05-02T00:00:00Z"})).unwrap()).unwrap();
        let mut pinned_args = args.clone();
        pinned_args.extend(["--fetch-record".into(), clock_receipt.display().to_string()]);
        cmd_decode(&Options::parse(&pinned_args).unwrap()).unwrap();
        let pinned = std::fs::read(&out).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(1100));
        cmd_decode(&Options::parse(&pinned_args).unwrap()).unwrap();
        assert_eq!(pinned, std::fs::read(&out).unwrap(), "same source and acquisition clock must reproduce seam bytes");
        std::fs::write(&clock_receipt, format!(r#"{{"schema":"{FETCH_SCHEMA}","sha256":"wrong","fetched_at":"2025-05-02T00:00:00Z"}}"#)).unwrap();
        assert!(cmd_decode(&Options::parse(&pinned_args).unwrap()).is_err());
        std::fs::remove_dir_all(&dir).ok();
        assert_eq!(record["provenance"]["product"], "iem-asos-1min");
        assert_eq!(record["match_seconds"], 30);
        assert_eq!(record["valid_times"].as_array().unwrap().len(), 5);
        let reports = record["reports"].as_array().unwrap();
        // 17:02 has no report and borrows none: each report serves only the
        // minute it was taken.
        let served: Vec<(&str, &str)> = reports
            .iter()
            .map(|r| (r["valid_time"].as_str().unwrap(), r["observation_time"].as_str().unwrap()))
            .collect();
        assert_eq!(
            served,
            vec![
                ("2025-05-01T17:00:00", "2025-05-01T17:00:00"),
                ("2025-05-01T17:01:00", "2025-05-01T17:01:00"),
                ("2025-05-01T17:03:00", "2025-05-01T17:03:00"),
                ("2025-05-01T17:04:00", "2025-05-01T17:04:00"),
            ]
        );
        let dewpoint = reports[1]["values"]["dewpoint_2m"].as_f64().unwrap();
        assert!((dewpoint - fahrenheit_to_kelvin(61.0)).abs() < 1e-12);
    }

    #[test]
    fn the_csv_splitter_survives_a_quoted_field_holding_commas() {
        assert_eq!(split_csv("a,b,c"), vec!["a", "b", "c"]);
        assert_eq!(
            split_csv("DSM,2024-05-21 20:54,\"METAR KDSM 21Z, AUTO\",80.00"),
            vec!["DSM", "2024-05-21 20:54", "METAR KDSM 21Z, AUTO", "80.00"]
        );
        assert_eq!(split_csv("a,,c"), vec!["a", "", "c"]);
    }
    #[test]
    fn decoded_gust_and_precipitation_keep_missing_and_trace_absent() {
        let dir=std::env::temp_dir().join(format!("rw_asos_gust_{}",std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let station=one_minute_station();
        let table=StationTable{schema:STATIONS_SCHEMA.into(),status:"READY".into(),archive:DEFAULT_ARCHIVE.into(),networks:vec!["XX_ASOS".into()],frozen_at:"2025-05-01T00:00:00".into(),content_sha256:station_rows_digest(std::slice::from_ref(&station)),stations:vec![station]};
        let stations=dir.join("stations.json");let csv=dir.join("obs.csv");let out=dir.join("surface.json");
        std::fs::write(&stations,serde_json::to_vec(&table).unwrap()).unwrap();
        std::fs::write(&csv,"station,valid,tmpf,dwpf,sknt,gust,p01i,vsby\nAAA,2025-05-01 17:00,80,60,10,20,0.1,0.25\nAAA,2025-05-01 18:00,80,60,10,,T,10\n").unwrap();
        let args=["--stations",stations.to_str().unwrap(),"--obs",csv.to_str().unwrap(),"--start","2025-05-01T17:00:00Z","--end","2025-05-01T18:00:00Z","--out",out.to_str().unwrap()].map(str::to_string);
        cmd_decode(&Options::parse(&args).unwrap()).unwrap();
        let record:serde_json::Value=serde_json::from_slice(&std::fs::read(&out).unwrap()).unwrap();
        assert!((record["reports"][0]["values"]["wind_gust_10m"].as_f64().unwrap()-knots_to_ms(20.0)).abs()<1e-12);
        assert!((record["reports"][0]["values"]["precipitation_1h"].as_f64().unwrap()-2.54).abs()<1e-12);
        assert!(record["reports"][1]["values"].get("wind_gust_10m").is_none());
        assert!(record["reports"][1]["values"].get("precipitation_1h").is_none());
        assert!((record["reports"][0]["values"]["visibility"].as_f64().unwrap()-402.336).abs()<1e-10);
        assert_eq!(record["reports"][1]["values"]["visibility"], 16093.44);
        assert_eq!(record["reports"][1]["flags"][0], "metar_visibility_ge_10sm");
        for entry in std::fs::read_dir(&dir).unwrap() {let entry=entry.unwrap();println!("deleted {} {}",entry.path().display(),entry.metadata().unwrap().len());std::fs::remove_file(entry.path()).unwrap();}
        std::fs::remove_dir(&dir).unwrap();
    }

    #[test]
    fn columns_are_resolved_by_name_so_a_reordered_csv_still_reads() {
        let normal = "station,valid,tmpf,dwpf,sknt,mslp\n\
                      DSM,2024-05-21 20:54,80.00,68.00,17.00,993.70\n";
        let reordered = "station,mslp,valid,sknt,dwpf,tmpf\n\
                         DSM,993.70,2024-05-21 20:54,17.00,68.00,80.00\n";
        for text in [normal, reordered] {
            let rows = parse_csv(text).unwrap();
            assert_eq!(rows.len(), 1);
            assert_eq!(rows[0].station_id, "DSM");
            assert_eq!(rows[0].tmpf, Some(80.0));
            assert_eq!(rows[0].dwpf, Some(68.0));
            assert_eq!(rows[0].sknt, Some(17.0));
            assert_eq!(rows[0].mslp_hpa, Some(993.70));
            assert_eq!(seam_time(rows[0].valid), "2024-05-21T20:54:00");
        }
    }

    #[test]
    fn an_empty_field_is_missing_and_a_trace_token_is_not_a_number() {
        let text = "station,valid,tmpf,dwpf,sknt,mslp\n\
                    DSM,2024-05-21 21:11,80.00,67.00,13.00,\n\
                    DSM,2024-05-21 21:33,76.00,70.00,20.00,T\n";
        let rows = parse_csv(text).unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].mslp_hpa, None, "an empty field is an absent value");
        assert_eq!(rows[1].mslp_hpa, None, "a trace token is not a pressure");
    }

    #[test]
    fn a_csv_that_is_not_one_is_refused_rather_than_read_as_zero_rows() {
        assert!(parse_csv("").is_err());
        assert!(parse_csv("<html>service unavailable</html>\n").is_err());
        assert!(parse_csv("valid,station\n2024-05-21 20:54,DSM\n").is_err());
    }

    #[test]
    fn a_station_table_digest_covers_the_rows_and_catches_an_edit() {
        let mut stations = vec![Station {
            station_id: "DSM".to_string(),
            name: "DES MOINES".to_string(),
            latitude: 41.5339,
            longitude: -93.6531,
            elevation_m: 294.0,
            network: "IA_ASOS".to_string(),
            state: "IA".to_string(),
        }];
        let before = station_rows_digest(&stations);
        // A cosmetic change does not move the digest...
        stations[0].name = "DES MOINES INTL".to_string();
        assert_eq!(station_rows_digest(&stations), before);
        // ...but moving a station does.
        stations[0].latitude = 41.6;
        assert_ne!(station_rows_digest(&stations), before);
    }

    #[test]
    fn every_command_requires_its_arguments() {
        assert!(cmd_stations(&Options::default()).is_err());
        assert!(cmd_fetch(&Options::default()).is_err());
        assert!(cmd_decode(&Options::default()).is_err());
        assert!(cmd_verify(&Options::default()).is_err());
    }

    #[test]
    fn a_network_name_that_would_reshape_the_url_is_refused() {
        for hostile in ["IA_ASOS/../x", "IA ASOS", "IA-ASOS", "../etc"] {
            let options = Options {
                networks: Some(hostile.to_string()),
                out: Some(PathBuf::from("x.json")),
                ..Options::default()
            };
            assert!(cmd_stations(&options).is_err(), "{hostile:?} must be refused");
        }
    }
}
