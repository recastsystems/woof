//! `rw_prepbufr` -- the conventional-observation front door for NCEP's
//! prepbufr.
//!
//! NOAA posts the rapid-refresh prepbufr every hour on NOMADS
//! (`obsproc/prod/rap.YYYYMMDD/rap.tHHz.prepbufr.tm00.nr`, and the early
//! 00Z and 12Z dumps under `rap_e.YYYYMMDD`; about two days are kept).  It
//! is the public route to soundings, aircraft, wind profilers, radar VAD
//! winds and the quality-controlled surface network, every value carrying
//! NCEP's quality mark and its event history.
//!
//! * `fetch` downloads one hour into a directory with its SHA-256, size,
//!   the server's `Last-Modified` and the receipt instant.
//! * `decode` reads a file (its own dictionary messages first, then every
//!   data message, subset by subset) and states the counts by message type
//!   and report type; `--listing FILE` also writes the oracle listing, one
//!   line per message, subset and level with every value as the bits of
//!   the double the NCEP library returns for it, which a program on
//!   NCEPLIBS-bufr making GSI's own calls prints identically.
//! * `table` writes the neutral observation table `gpuwm-obs.table.v2`
//!   with GSI's use rule applied in the door (`rw_obs::prepbufr`): a value
//!   whose mark is 4 or more, or whose level's pressure mark is, or that
//!   GSI's surface rules refuse, is never written; every refusal is
//!   counted by report type, variable and reason.  Each row's error is
//!   GSI's: the conventional error table's for the report type at the
//!   level's pressure (NCEP's published table unless `--error-table`
//!   names another), raised as GSI raises it from the observation alone.
//! * `verify` re-hashes a table against its record.
//!
//! The decode is pure Rust (`rw_obs::ncep_bufr`); NCEPLIBS-bufr is a test
//! oracle only and is never linked here.
//!
//! ```text
//! rw_prepbufr fetch  --cycle TIME --out DIR [--early] [--url-template T]
//! rw_prepbufr decode --file F [--listing OUT.txt]
//! rw_prepbufr table  --file F --out FILE.csv [--fetch-record R.json] [--error-source S] [--error-table T]
//! rw_prepbufr verify --file FILE.csv --record FILE.json
//! ```

use std::collections::BTreeMap;
use std::error::Error;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use chrono::{DateTime, Timelike, Utc};
use serde::{Deserialize, Serialize};

use rw_nexrad::s3::{parse_http_date, parse_time};
use rw_obs::ncep_bufr::{read_file, NcepFile};
use rw_obs::errtable::ErrorTable;
use rw_obs::prepbufr::{
    census, dump, table_of, Counts, ErrorSource, RowContext, ERROR_INFLATION, GSI_LEVEL_LIMIT, MARK_LIMIT,
    NO_WEIGHT_ERROR, REPORT_TYPES, SOURCE, UPPER_MOISTURE_LAYER_HPA, UPPER_MOISTURE_REPORT_TYPES,
};
use rw_obs::seam::seam_time;
use rw_obs::table::{
    revision_of, RowProvenance, MEAS_AIRCRAFT_LEVEL, MEAS_PLATFORM_WIND,
    MEAS_PROFILER_LEVEL, MEAS_STATION_PRESSURE_FROM_SEA_LEVEL, MEAS_VAD_LEVEL, MEAS_VAD_SUPEROB,
    TABLE_SCHEMA,
};
use rw_obs::{err, hex_sha256};

const VERSION: &str = env!("CARGO_PKG_VERSION");

pub static GPUWM_BRIDGE_SOURCE_REV_STAMP: &str =
    concat!("GPUWM_BRIDGE_SOURCE_REV=", env!("GPUWM_BRIDGE_SOURCE_REV"));

const FETCH_SCHEMA: &str = "gpuwm-obs.prepbufr-fetch.v1";
const DECODE_SCHEMA: &str = "gpuwm-obs.prepbufr-decode.v1";
/// v2: the error is GSI's (the error table, or the file's own), and the
/// record states the source and the table's hash in place of v1's
/// per-kind constants.
const TABLE_RECORD_SCHEMA: &str = "gpuwm-obs.prepbufr-table.v2";
const VERIFY_SCHEMA: &str = "gpuwm-obs.prepbufr-verify.v1";
const LISTING_SCHEMA: &str = "gpuwm-obs.prepbufr-listing.v1";

const ABI_MARKER: &str = "gpuwm-obs.prepbufr-fetch.v1\tgpuwm-obs.prepbufr-decode.v1\t\
gpuwm-obs.prepbufr-table.v2\tgpuwm-obs.table.v2\tgpuwm-obs.prepbufr-listing.v1\t\
mark-limit-4\tpressure-mark\tupper-moisture-9-to-2\tvad-superob\tprofiler-400-hpa\t\
surface-gsdqc-2\terror-table";

/// NOMADS' public obsproc tree; `{YYYYMMDD}` and `{HH}` are the cycle's.
const DEFAULT_URL_TEMPLATE: &str =
    "https://nomads.ncep.noaa.gov/pub/data/nccf/com/obsproc/prod/rap.{YYYYMMDD}/rap.t{HH}z.prepbufr.tm00.nr";
/// The early dump NOAA posts at 00Z and 12Z, about half an hour sooner.
const EARLY_URL_TEMPLATE: &str =
    "https://nomads.ncep.noaa.gov/pub/data/nccf/com/obsproc/prod/rap_e.{YYYYMMDD}/rap_e.t{HH}z.prepbufr.tm00.nr";
/// A prepbufr hour is 4.7 to 9.1 MB (MEASURED, NOMADS listing 2026-10-03);
/// a file past this is not one, and is refused before it is decoded.
const MAX_FILE_BYTES: usize = 64 * 1024 * 1024;

const USAGE: &str = "\
usage: rw_prepbufr <fetch|decode|table|verify> [OPTIONS]
       rw_prepbufr --version | --help | --abi

  fetch   download one prepbufr hour with its sha256 and the server's Last-Modified
  decode  read a file and state its messages, subsets and levels by message type and
          report type; --listing writes the oracle listing beside it
  table   write the neutral observation table `gpuwm-obs.table.v2` with GSI's use rule
  verify  re-hash a table against the record written beside it

fetch options
  --cycle TIME          the hour, e.g. 2026-10-03T12:00:00Z
  --out DIR             the file lands in DIR under its own name; a .json record beside it
  --early               the early dump (posted at 00Z and 12Z only)
  --url-template T      a URL with {YYYYMMDD} and {HH} (default: NOMADS obsproc)

decode options
  --file F              a prepbufr file
  --listing OUT.txt     also write the oracle listing (one line per message, subset, level)

table options
  --file F              a prepbufr file
  --out FILE.csv        the table; a .json record is written beside it
  --fetch-record R      the fetch record, for the published and received instants
  --error-source S      table (default): GSI's conventional error table, the error of the
                        row's report type at the level's pressure, floored, which is what
                        the operational analysis uses; file: the file's own POE/TOE/QOE/WOE
                        (GSI without a table), a value without one counted, not written.
                        Either way raised by 1.2 for marks 3 and 7, temperatures above
                        100 hPa and winds above 50 hPa, as GSI does
  --error-table T       a table in GSI's errtable format (default: NCEP's published table,
                        built in; its SHA-256 is in the record)
  --raw-vad             write every radar VAD level; by default VAD winds are read as GSI
                        reads them (its time windows, every sixth level, six-level superobs
                        checked against the file's own background)

verify options
  --file FILE.csv --record FILE.json
";

fn main() -> ExitCode {
    let _ = std::hint::black_box(GPUWM_BRIDGE_SOURCE_REV_STAMP);
    rw_obs::cli::main("rw_prepbufr", run)
}

fn run(args: &[String]) -> Result<String, Box<dyn Error>> {
    if let Some(output) = rw_obs::cli::information(args, "rw_prepbufr", VERSION, USAGE, ABI_MARKER) {
        return Ok(output);
    }
    let first = &args[0];
    let options = Options::parse(&args[1..])?;
    match first.as_str() {
        "fetch" => cmd_fetch(&options),
        "decode" => cmd_decode(&options),
        "table" => cmd_table(&options),
        "verify" => cmd_verify(&options),
        other => Err(err(format!("unknown subcommand {other:?}\n\n{USAGE}"))),
    }
}

#[derive(Debug, Default)]
struct Options {
    cycle: Option<String>,
    out: Option<PathBuf>,
    early: bool,
    url_template: Option<String>,
    file: Option<PathBuf>,
    listing: Option<PathBuf>,
    fetch_record: Option<PathBuf>,
    error_source: Option<String>,
    error_table: Option<PathBuf>,
    raw_vad: bool,
    record: Option<PathBuf>,
}

impl Options {
    fn parse(args: &[String]) -> Result<Self, Box<dyn Error>> {
        let mut options = Options::default();
        let mut index = 0;
        while index < args.len() {
            let flag = args[index].as_str();
            let mut value = || rw_obs::cli::value(args, &mut index, flag);
            match flag {
                "--cycle" => options.cycle = Some(value()?),
                "--out" => options.out = Some(PathBuf::from(value()?)),
                "--early" => options.early = true,
                "--url-template" => options.url_template = Some(value()?),
                "--file" => options.file = Some(PathBuf::from(value()?)),
                "--listing" => options.listing = Some(PathBuf::from(value()?)),
                "--fetch-record" => options.fetch_record = Some(PathBuf::from(value()?)),
                "--error-source" => {
                    let source = value()?;
                    if source != "table" && source != "file" {
                        return Err(err(format!("--error-source {source:?} is neither table nor file")));
                    }
                    options.error_source = Some(source);
                }
                "--error-table" => options.error_table = Some(PathBuf::from(value()?)),
                "--record" => options.record = Some(PathBuf::from(value()?)),
                "--raw-vad" => options.raw_vad = true,
                other => return Err(err(format!("unknown option {other:?}\n\n{USAGE}"))),
            }
            index += 1;
        }
        Ok(options)
    }
}

// ----------------------------------------------------------------- fetch

#[derive(Debug, Serialize, Deserialize)]
struct FetchRecord {
    schema: String,
    status: String,
    url: String,
    cycle: String,
    early: bool,
    path: String,
    bytes: usize,
    sha256: String,
    /// The server's `Last-Modified`, seam spelling; empty when absent.
    published_time: String,
    /// The instant this system held the bytes, seam spelling.
    fetched_at: String,
}

fn cycle_of(text: &str) -> Result<DateTime<Utc>, Box<dyn Error>> {
    let cycle = parse_time(text)?;
    if cycle.minute() != 0 || cycle.second() != 0 {
        return Err(err(format!("--cycle {text} is not on the hour; prepbufr is posted per hour")));
    }
    Ok(cycle)
}

fn cmd_fetch(options: &Options) -> Result<String, Box<dyn Error>> {
    let cycle = cycle_of(options.cycle.as_deref().ok_or_else(|| err("--cycle is required"))?)?;
    let out = options.out.as_deref().ok_or_else(|| err("--out DIR is required"))?;
    if options.early && !(cycle.hour() == 0 || cycle.hour() == 12) {
        return Err(err(format!("--early names the early dump, which NOAA posts at 00Z and 12Z only; {} is neither", seam_time(cycle))));
    }
    let template = options
        .url_template
        .clone()
        .unwrap_or_else(|| if options.early { EARLY_URL_TEMPLATE } else { DEFAULT_URL_TEMPLATE }.to_string());
    if !template.contains("{YYYYMMDD}") || !template.contains("{HH}") {
        return Err(err(format!("--url-template {template:?} lacks {{YYYYMMDD}} or {{HH}}, so it names no hour")));
    }
    let url = template
        .replace("{YYYYMMDD}", &cycle.format("%Y%m%d").to_string())
        .replace("{HH}", &cycle.format("%H").to_string());
    let name = url.rsplit('/').next().filter(|n| !n.is_empty()).ok_or_else(|| err(format!("{url} ends in no file name")))?;
    let agent = rw_obs::net::agent();
    let subject = format!("prepbufr {}", seam_time(cycle));
    let mut response = rw_http::get(&agent, &url).map_err(|e| err(format!("{subject}: GET {url} failed: {e}")))?;
    let status = response.status().as_u16();
    if !(200..300).contains(&status) {
        return Err(err(format!("{subject}: GET {url} answered HTTP {status}; NOMADS keeps about two days")));
    }
    let published = response
        .headers()
        .get("last-modified")
        .and_then(|v| v.to_str().ok())
        .and_then(parse_http_date);
    let bytes = response
        .body_mut()
        .with_config()
        .limit(MAX_FILE_BYTES as u64)
        .read_to_vec()
        .map_err(|e| err(format!("{subject}: reading {url} failed: {e}")))?;
    let fetched_at = Utc::now();
    // A body that is not BUFR (an error page served with 200) is refused here.
    let (messages, _) = rw_obs::ncep_bufr::split_messages(&bytes, &subject)?;
    std::fs::create_dir_all(out).map_err(|e| err(format!("cannot create {}: {e}", out.display())))?;
    let path = out.join(name);
    std::fs::write(&path, &bytes).map_err(|e| err(format!("cannot write {}: {e}", path.display())))?;
    let record = FetchRecord {
        schema: FETCH_SCHEMA.to_string(),
        status: if messages.is_empty() { "EMPTY" } else { "READY" }.to_string(),
        url,
        cycle: seam_time(cycle),
        early: options.early,
        path: rw_obs::absolute_uri(&path),
        bytes: bytes.len(),
        sha256: hex_sha256(&bytes),
        published_time: published.map(seam_time).unwrap_or_default(),
        fetched_at: seam_time(fetched_at),
    };
    let text = format!("{}\n", serde_json::to_string_pretty(&record)?);
    std::fs::write(path.with_extension("fetch.json"), &text)
        .map_err(|e| err(format!("cannot write the fetch record: {e}")))?;
    Ok(text)
}

// ---------------------------------------------------------------- decode

#[derive(Debug, Serialize)]
struct InputRecord {
    path: String,
    bytes: usize,
    sha256: String,
    revision: String,
}

#[derive(Debug, Serialize)]
struct FramingRecord {
    messages: usize,
    dictionary_messages: usize,
    data_messages: usize,
    dictionaries: usize,
    control_word_records: usize,
    padding_bytes: usize,
    table_a_entries: usize,
    table_b_entries: usize,
    table_d_entries: usize,
}

fn load(path: &Path) -> Result<(Vec<u8>, NcepFile, InputRecord), Box<dyn Error>> {
    let bytes = std::fs::read(path).map_err(|e| err(format!("cannot read {}: {e}", path.display())))?;
    if bytes.len() > MAX_FILE_BYTES {
        return Err(err(format!(
            "{} is {} bytes; a prepbufr hour is under 10 MB, so this is not one",
            path.display(),
            bytes.len()
        )));
    }
    let what = path.file_name().and_then(|n| n.to_str()).unwrap_or("input").to_string();
    let file = read_file(&bytes, &what)?;
    if file.messages.is_empty() {
        return Err(err(format!("{what}: the file carries dictionary messages and no data message")));
    }
    let sha = hex_sha256(&bytes);
    let input = InputRecord { path: rw_obs::absolute_uri(path), bytes: bytes.len(), revision: revision_of(&sha), sha256: sha };
    Ok((bytes, file, input))
}

fn framing(file: &NcepFile) -> FramingRecord {
    let last = file.dictionaries.last();
    FramingRecord {
        messages: file.framing.messages,
        dictionary_messages: file.dictionary_messages,
        data_messages: file.messages.len(),
        dictionaries: file.dictionaries.len(),
        control_word_records: file.framing.control_word_records,
        padding_bytes: file.framing.padding_bytes,
        table_a_entries: last.map(|d| d.table_a.len()).unwrap_or(0),
        table_b_entries: last.map(|d| d.elements.len()).unwrap_or(0),
        table_d_entries: last.map(|d| d.sequences.len()).unwrap_or(0),
    }
}

#[derive(Debug, Serialize)]
struct ListingRecord {
    schema: &'static str,
    path: String,
    lines: usize,
    bytes: usize,
    sha256: String,
    format: &'static str,
}

#[derive(Debug, Serialize)]
struct DecodeRecord {
    schema: &'static str,
    status: &'static str,
    input: InputRecord,
    cycle: String,
    framing: FramingRecord,
    messages_by_type: BTreeMap<String, usize>,
    subsets_by_type: BTreeMap<String, usize>,
    levels_by_type: BTreeMap<String, usize>,
    subsets_by_type_and_report_type: BTreeMap<String, usize>,
    listing: Option<ListingRecord>,
}

fn cycle_text(file: &NcepFile) -> Result<String, Box<dyn Error>> {
    let first = file.messages.first().ok_or_else(|| err("the file holds no data message"))?;
    Ok(seam_time(rw_obs::prepbufr::cycle_time(first)?))
}

fn cmd_decode(options: &Options) -> Result<String, Box<dyn Error>> {
    let path = options.file.as_deref().ok_or_else(|| err("--file is required"))?;
    let (_, file, input) = load(path)?;
    let census = census(&file)?;
    let listing = match options.listing.as_deref() {
        Some(out) => {
            let text = dump(&file)?;
            if let Some(parent) = out.parent().filter(|p| !p.as_os_str().is_empty()) {
                std::fs::create_dir_all(parent).map_err(|e| err(format!("cannot create {}: {e}", parent.display())))?;
            }
            std::fs::write(out, text.as_bytes()).map_err(|e| err(format!("cannot write {}: {e}", out.display())))?;
            Some(ListingRecord {
                schema: LISTING_SCHEMA,
                path: rw_obs::absolute_uri(out),
                lines: text.lines().count(),
                bytes: text.len(),
                sha256: hex_sha256(text.as_bytes()),
                format: "V <virtual-temperature program code>; M <message type> <YYYYMMDDHH> <subsets>; \
                         S <subset> <SID XOB YOB DHR TYP ELV SAID T29 as Z16 bits> <levels read by ufbint for \
                         obs, marks, errors (at most 255)> <levels by ufbevn for TPC> <levels by ufbint for \
                         XDR YDR HRDR>; L <level> <POB..PMO> <PQM..PMQ> <POE..PWE> <XDR YDR HRDR> <top TPC> \
                         <V virtual or S>, each value the hex bits of the \
                         double NCEPLIBS-bufr returns (10E10 missing)",
            })
        }
        None => None,
    };
    let record = DecodeRecord {
        schema: DECODE_SCHEMA,
        status: "READY",
        cycle: cycle_text(&file)?,
        framing: framing(&file),
        messages_by_type: census.messages_by_type,
        subsets_by_type: census.subsets_by_type,
        levels_by_type: census.levels_by_type,
        subsets_by_type_and_report_type: census.subsets_by_type_and_report_type,
        listing,
        input,
    };
    Ok(format!("{}\n", serde_json::to_string_pretty(&record)?))
}

// ----------------------------------------------------------------- table

#[derive(Debug, Serialize)]
struct UseRule {
    rule: &'static str,
    mark_limit: i64,
    upper_moisture_report_types: [u16; 4],
    upper_moisture_layer_hpa: (f64, f64),
    gsi_level_limit: usize,
    vad: &'static str,
    thinning: &'static str,
}

#[derive(Debug, Serialize)]
struct ErrorRecord {
    source: &'static str,
    rule: &'static str,
    /// The table's name, SHA-256 and report-type count; empty under `file`.
    table: String,
    table_sha256: String,
    table_types: usize,
    inflation: f64,
    no_weight_error: f64,
}

#[derive(Debug, Serialize)]
struct TableRecord {
    schema: &'static str,
    status: &'static str,
    source: &'static str,
    table_schema: &'static str,
    path: String,
    sha256: String,
    rows: usize,
    bytes: usize,
    input: InputRecord,
    cycle: String,
    published_time: String,
    received_time: String,
    use_rule: UseRule,
    errors: ErrorRecord,
    measurements_added_to_the_vocabulary: Vec<&'static str>,
    report_types_mapped: Vec<String>,
    messages_by_type: BTreeMap<String, usize>,
    subsets_by_type: BTreeMap<String, usize>,
    levels_by_type: BTreeMap<String, usize>,
    rows_by_variable: BTreeMap<String, usize>,
    counts: Counts,
}

fn cmd_table(options: &Options) -> Result<String, Box<dyn Error>> {
    if options.error_table.is_some() && options.error_source.as_deref() == Some("file") {
        return Err(err("--error-table names a table and --error-source file says not to use one; give one or the other"));
    }
    let path = options.file.as_deref().ok_or_else(|| err("--file is required"))?;
    let out = options.out.as_deref().ok_or_else(|| err("--out FILE.csv is required"))?;
    if out.is_dir() {
        return Err(err(format!("--out {} is a directory; give the CSV path", out.display())));
    }
    let (_, file, input) = load(path)?;
    let (mut published, mut received) = (None, None);
    if let Some(record_path) = options.fetch_record.as_deref() {
        let text = std::fs::read_to_string(record_path).map_err(|e| err(format!("cannot read {}: {e}", record_path.display())))?;
        let record: FetchRecord = serde_json::from_str(&text).map_err(|e| err(format!("{} is not a fetch record: {e}", record_path.display())))?;
        if record.schema != FETCH_SCHEMA {
            return Err(err(format!("{} declares schema {:?}, expected {FETCH_SCHEMA:?}", record_path.display(), record.schema)));
        }
        if record.sha256 != input.sha256 {
            return Err(err(format!(
                "{} records sha256 {} and {} hashes to {}; the record is another file's, so its instants cannot be this file's",
                record_path.display(),
                record.sha256,
                path.display(),
                input.sha256
            )));
        }
        published = parse_time(&record.published_time).ok();
        received = parse_time(&record.fetched_at).ok();
    }
    let named_table = match options.error_table.as_deref() {
        Some(table_path) => {
            let text = std::fs::read_to_string(table_path)
                .map_err(|e| err(format!("cannot read {}: {e}", table_path.display())))?;
            let name = table_path.file_name().and_then(|n| n.to_str()).unwrap_or("errtable").to_string();
            Some(ErrorTable::parse(&text, &name)?)
        }
        None => None,
    };
    let table = named_table.as_ref().unwrap_or_else(|| ErrorTable::default_table());
    let errors = if options.error_source.as_deref() == Some("file") { ErrorSource::File } else { ErrorSource::Table(table) };
    let provenance = RowProvenance::of_source(&input.sha256, published, received);
    let context = RowContext { provenance: &provenance, errors, vad_superob: !options.raw_vad };
    let (writer, rows_by_variable, counts) = table_of(&file, &context)?;
    let (rows, sha, bytes) = writer.write(out)?;
    let census = census(&file)?;
    let (table_name, table_sha256, table_types) = match errors {
        ErrorSource::Table(t) => (t.name.clone(), t.sha256.clone(), t.types),
        ErrorSource::File => (String::new(), String::new(), 0),
    };
    let record = TableRecord {
        schema: TABLE_RECORD_SCHEMA,
        status: if rows > 0 { "READY" } else { "EMPTY" },
        source: SOURCE,
        table_schema: TABLE_SCHEMA,
        path: rw_obs::absolute_uri(out),
        sha256: sha,
        rows,
        bytes,
        cycle: cycle_text(&file)?,
        published_time: published.map(seam_time).unwrap_or_default(),
        received_time: received.map(seam_time).unwrap_or_default(),
        use_rule: UseRule {
            rule: "GSI read_prepbufr with noiqc false: a value whose own mark is 4 or more (or missing) is not \
                   written; a temperature, humidity or wind whose level's pressure mark is 4 or more is not \
                   written; a station pressure only from a category-0 level at 500 hPa or more whose height mark \
                   is below 4 (9 and 15 excepted), never from types 192 to 195; humidity of types 120, 131, 133 \
                   and 134 between 300 and 10 hPa with mark 9 is taken with mark 2 (i_gsdqc 2); a surface \
                   humidity of types 180 to 189 is not written when TDO is under min(-40 C, TOB - 10) (usage \
                   116), TOB - TDO exceeds 70 (117) or TDO exceeds 32.2 C (118), missing taken as missing \
                   (i_gsdqc 2); a type-288 wind with both components under 0.01 m/s is not written (115); \
                   multi-agency profiler (227) winds above 400 hPa are not written (GSI's regional rule); radar \
                   VAD (224) winds are read as GSI reads them unless --raw-vad",
            mark_limit: MARK_LIMIT,
            upper_moisture_report_types: UPPER_MOISTURE_REPORT_TYPES,
            upper_moisture_layer_hpa: UPPER_MOISTURE_LAYER_HPA,
            gsi_level_limit: GSI_LEVEL_LIMIT,
            vad: if options.raw_vad {
                "raw: every level as measured"
            } else {
                "GSI's new-VAD read: subtype 2 reports in six |DHR| windows, every sixth level, a superob of \
                 it and the next five within 301 m, refused past 10 m/s from the file's background (8 m/s \
                 in v, 5 m/s in v below 5000 m), above 7000 m, past 5 m/s spread or under three members; \
                 a superob over 60 m/s ends the report"
            },
            thinning: "dthin 0 on every prepbufr row of the regional analysis; the only thinning is GSI's VAD read",
        },
        errors: ErrorRecord {
            source: errors.name(),
            rule: "GSI read_prepbufr: under 'error-table' the table's error for the report type at the level's \
                   pressure (clamped to 0..2000 hPa, GSI's level search and weight), floored at 0.5 K, 0.05 tenths \
                   of saturation, 1 m/s, 0.3 hPa; under 'file' the file's POE, TOE, QOE, WOE, a value without one \
                   not written. Then times 1.2 for a value whose mark is 3 or 7, a temperature times 1.2 again \
                   above 100 hPa, a wind above 50 hPa. A humidity error (tenths of saturation) is carried into \
                   dewpoint at the level's temperature. An error at or past no_weight_error is not written. NOT \
                   applied: errormod's factor for a level of a profile (at least 1; needs the background's layer \
                   depth), counted as rows_without_errormod_factor, and the setup routines' adjustments",
            table: table_name,
            table_sha256,
            table_types,
            inflation: ERROR_INFLATION,
            no_weight_error: NO_WEIGHT_ERROR,
        },
        measurements_added_to_the_vocabulary: vec![
            MEAS_AIRCRAFT_LEVEL,
            MEAS_PROFILER_LEVEL,
            MEAS_VAD_LEVEL,
            MEAS_VAD_SUPEROB,
            MEAS_PLATFORM_WIND,
            MEAS_STATION_PRESSURE_FROM_SEA_LEVEL,
        ],
        report_types_mapped: REPORT_TYPES
            .iter()
            .map(|row| match row.message_type {
                Some(m) => format!("{}/{m}", row.report_type),
                None => row.report_type.to_string(),
            })
            .collect(),
        messages_by_type: census.messages_by_type,
        subsets_by_type: census.subsets_by_type,
        levels_by_type: census.levels_by_type,
        rows_by_variable,
        counts,
        input,
    };
    let text = format!("{}\n", serde_json::to_string_pretty(&record)?);
    std::fs::write(out.with_extension("json"), &text).map_err(|e| err(format!("cannot write the table record: {e}")))?;
    Ok(text)
}

// ---------------------------------------------------------------- verify

fn cmd_verify(options: &Options) -> Result<String, Box<dyn Error>> {
    rw_obs::table::verify(
        options.file.as_deref(), options.record.as_deref(), TABLE_RECORD_SCHEMA, VERIFY_SCHEMA,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abi_marker_names_the_contracts_it_pins() {
        for schema in [FETCH_SCHEMA, DECODE_SCHEMA, TABLE_RECORD_SCHEMA, TABLE_SCHEMA, LISTING_SCHEMA] {
            assert!(ABI_MARKER.contains(schema), "{schema}");
        }
    }

    #[test]
    fn options_refuse_what_they_cannot_read() {
        let args = |list: &[&str]| list.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert!(Options::parse(&args(&["--error-source", "loose"])).unwrap_err().to_string().contains("loose"));
        // a table named beside the file source is refused before anything is read
        let o = Options::parse(&args(&["--file", "absent.bufr", "--out", "o.csv", "--error-source", "file", "--error-table", "t"])).unwrap();
        assert!(cmd_table(&o).unwrap_err().to_string().contains("one or the other"));
        assert!(Options::parse(&args(&["--bogus"])).unwrap_err().to_string().contains("--bogus"));
        assert!(cycle_of("2026-10-03T12:30:00Z").unwrap_err().to_string().contains("not on the hour"));
        let o = Options::parse(&args(&["--early", "--cycle", "2026-10-03T13:00:00Z", "--out", "x"])).unwrap();
        assert!(cmd_fetch(&o).unwrap_err().to_string().contains("00Z and 12Z"));
        let o = Options::parse(&args(&["--cycle", "2026-10-03T13:00:00Z", "--out", "x", "--url-template", "https://h/f"])).unwrap();
        assert!(cmd_fetch(&o).unwrap_err().to_string().contains("names no hour"));
    }
}
