//! `rw_wis2` -- the WMO Information System 2 subscriber.
//!
//! WIS2 publishes every WMO member's core data as notification messages on
//! a global broker (MQTT over TLS, port 8883, the public `everyone` /
//! `everyone` credentials) whose `links[rel=canonical]` point at the
//! payload on a Global Cache (24 h retention, no access restriction).  This
//! bin connects to a broker, subscribes to the surface-based-observation
//! topics (SYNOP, TEMP, BUOY, SHIP and the rest under
//! `cache/a/wis2/+/data/core/weather/surface-based-observations/#`),
//! archives every notification message the moment it arrives, downloads
//! each payload once and archives it beside the message with its SHA-256
//! and the publisher's own integrity digest checked, and measures what the
//! design asks for: coverage (messages per centre and per topic) and
//! latency (the report's `datetime` to the notification's `pubtime`, and
//! `pubtime` to receipt here).
//!
//! `table` decodes the archived payloads into the neutral observation
//! table (`gpuwm-obs.table.v2`) through `rw_obs::bufr` (FM 94 BUFR
//! against the vendored WMO master tables) and `rw_obs::bufr_obs` (the
//! SYNOP, SHIP and TEMP templates to rows): station pressure, screen
//! temperature and dewpoint and the 10 m wind of a surface report, every
//! level of a sounding at its own time, each row carrying the
//! notification's pubtime as its published time, the archive receipt as
//! its received time and the payload's digest as its revision.  A payload
//! this reader cannot decode is counted with its refusal named; a
//! template it does not read (a buoy, a profiler) is counted under its
//! descriptors.
//!
//! MQTT 3.1.1 is framed by hand (CONNECT, CONNACK, SUBSCRIBE, SUBACK,
//! PUBLISH, PUBACK, PINGREQ, PINGRESP, DISCONNECT): the client needs six
//! packet types and no crate in the vendor closure provides them.  TLS is
//! rustls with the RustCrypto provider and the webpki roots, the same stack
//! every HTTPS door here uses.
//!
//! ```text
//! rw_wis2 subscribe --out DIR --seconds N [--broker HOST:PORT] [--topics LIST]
//!                   [--no-download] [--max-downloads N]
//! rw_wis2 table --archive DIR --out FILE.csv [--start T --end T]
//! rw_wis2 table --payloads a.bufr4,b.bufr4 --out FILE.csv
//! rw_wis2 probe [--broker HOST:PORT]      connect, subscribe, disconnect; the handshake wall
//! rw_wis2 --version | --help | --abi
//! ```

use std::collections::BTreeMap;
use std::error::Error;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::time::{Duration, Instant};

use chrono::{DateTime, Utc};
use serde::Serialize;

use rw_nexrad::s3::{build_agent, parse_time};
use rw_obs::bufr;
use rw_obs::bufr_obs;
use rw_obs::seam::seam_time;
use rw_obs::table::{RowProvenance, TableWriter, TABLE_SCHEMA};
use rw_obs::{err, hex_sha256};

const VERSION: &str = env!("CARGO_PKG_VERSION");

pub static GPUWM_BRIDGE_SOURCE_REV_STAMP: &str =
    concat!("GPUWM_BRIDGE_SOURCE_REV=", env!("GPUWM_BRIDGE_SOURCE_REV"));

const DEFAULT_BROKER: &str = "globalbroker.meteo.fr:8883";
const DEFAULT_USER: &str = "everyone";
const DEFAULT_PASSWORD: &str = "everyone";
const DEFAULT_TOPICS: &[&str] = &[
    "cache/a/wis2/+/data/core/weather/surface-based-observations/#",
];
const DEFAULT_SECONDS: u64 = 600;
const KEEPALIVE_S: u16 = 60;
/// A WIS2 payload is a BUFR message of a few kilobytes; a megabyte is a
/// GRIB someone mis-filed and 32 MB is the ceiling past which a download
/// is refused rather than stored.
const MAX_PAYLOAD_BYTES: u64 = 32 * 1024 * 1024;

const SUBSCRIBE_SCHEMA: &str = "gpuwm-obs.wis2-subscribe.v1";
const PROBE_SCHEMA: &str = "gpuwm-obs.wis2-probe.v1";
const TABLE_RECORD_SCHEMA: &str = "gpuwm-obs.wis2-table.v1";
const ABI_MARKER: &str = "gpuwm-obs.wis2-subscribe.v1\tgpuwm-obs.wis2-probe.v1\tgpuwm-obs.wis2-table.v1\t\
gpuwm-obs.table.v2\tmqtt-3.1.1\tmessages\tpayloads\tcoverage\tlatency\tbufr\tsynop\tship\ttemp";

const USAGE: &str = "\
usage: rw_wis2 <subscribe|probe> [OPTIONS]
       rw_wis2 --version | --help | --abi

  subscribe  connect to a WIS2 global broker, subscribe to the surface-based
             observation topics, archive every notification and payload for
             --seconds, and write the coverage and latency record
  probe      connect, subscribe and disconnect; report the handshake wall
  table      decode the archived BUFR payloads (SYNOP, SHIP, TEMP) into a
             `gpuwm-obs.table.v2` CSV with a record beside it
  dump       print every decoded element of --payloads (the diagnostic route)

table options
  --archive DIR         a subscribe archive (its record.json gives every row its
                        published and received times and its revision)
  --payloads LIST       BUFR files, comma-separated (or repeat --payloads)
  --start T --end T     keep the rows inside [start, end] (ISO-8601 UTC)
  --out FILE.csv        the table; a .json record is written beside it

options
  --broker HOST:PORT    default globalbroker.meteo.fr:8883 (MQTT over TLS)
  --topics LIST         comma-separated topic filters (default
                        cache/a/wis2/+/data/core/weather/surface-based-observations/#)
  --out DIR             archive root: messages/<centre>/<id>.json and
                        payloads/<centre>/<file>; record.json beside them
  --seconds N           how long to listen (default 600)
  --max-downloads N     stop downloading payloads after N (messages still archived)
  --no-download         archive messages only
";

fn main() -> ExitCode {
    let _ = std::hint::black_box(GPUWM_BRIDGE_SOURCE_REV_STAMP);
    rw_obs::cli::main("rw_wis2", run)
}

fn run(args: &[String]) -> Result<String, Box<dyn Error>> {
    if let Some(output) = rw_obs::cli::information(args, "rw_wis2", VERSION, USAGE, ABI_MARKER) {
        return Ok(output);
    }
    let first = &args[0];
    let options = Options::parse(&args[1..])?;
    match first.as_str() {
        "subscribe" => cmd_subscribe(&options),
        "probe" => cmd_probe(&options),
        "table" => cmd_table(&options),
        "dump" => cmd_dump(&options),
        other => Err(err(format!("unknown subcommand {other:?}\n\n{USAGE}"))),
    }
}

#[derive(Debug, Default)]
struct Options {
    broker: Option<String>,
    topics: Option<Vec<String>>,
    out: Option<PathBuf>,
    seconds: Option<u64>,
    max_downloads: Option<usize>,
    no_download: bool,
    archive: Option<PathBuf>,
    payloads: Vec<PathBuf>,
    start: Option<String>,
    end: Option<String>,
}

impl Options {
    fn parse(args: &[String]) -> Result<Self, Box<dyn Error>> {
        let mut options = Options::default();
        let mut index = 0;
        while index < args.len() {
            let flag = args[index].as_str();
            let mut value = || rw_obs::cli::value(args, &mut index, flag);
            match flag {
                "--broker" => options.broker = Some(value()?),
                "--topics" => {
                    let raw = value()?;
                    let list: Vec<String> = raw.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect();
                    if list.is_empty() {
                        return Err(err("--topics named no topic"));
                    }
                    options.topics = Some(list);
                }
                "--out" => options.out = Some(PathBuf::from(value()?)),
                "--seconds" => {
                    let raw = value()?;
                    let s: u64 = raw.parse().map_err(|_| err(format!("--seconds expects a count, got {raw:?}")))?;
                    if s == 0 {
                        return Err(err("--seconds must be positive"));
                    }
                    options.seconds = Some(s);
                }
                "--max-downloads" => {
                    let raw = value()?;
                    options.max_downloads = Some(raw.parse().map_err(|_| err(format!("--max-downloads expects a count, got {raw:?}")))?);
                }
                "--no-download" => options.no_download = true,
                "--archive" => options.archive = Some(PathBuf::from(value()?)),
                "--payloads" => {
                    let raw = value()?;
                    options.payloads.extend(raw.split(',').map(str::trim).filter(|t| !t.is_empty()).map(PathBuf::from));
                }
                "--start" => options.start = Some(value()?),
                "--end" => options.end = Some(value()?),
                other => return Err(err(format!("unknown option {other:?}\n\n{USAGE}"))),
            }
            index += 1;
        }
        Ok(options)
    }

    fn broker(&self) -> Result<(String, u16), Box<dyn Error>> {
        let raw = self.broker.as_deref().unwrap_or(DEFAULT_BROKER);
        let (host, port) = raw
            .rsplit_once(':')
            .ok_or_else(|| err(format!("--broker expects HOST:PORT, got {raw:?}")))?;
        let port: u16 = port.parse().map_err(|_| err(format!("--broker port {port:?} is not a number")))?;
        if host.is_empty() || !host.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'.' || b == b'-') {
            return Err(err(format!("--broker host {host:?} is not a host name")));
        }
        Ok((host.to_string(), port))
    }

    fn topics(&self) -> Vec<String> {
        self.topics
            .clone()
            .unwrap_or_else(|| DEFAULT_TOPICS.iter().map(|t| t.to_string()).collect())
    }
}

// ------------------------------------------------------------------- mqtt

/// The MQTT remaining-length encoding (one to four bytes, seven bits each).
fn encode_remaining(mut n: usize, out: &mut Vec<u8>) {
    loop {
        let mut byte = (n % 128) as u8;
        n /= 128;
        if n > 0 {
            byte |= 0x80;
        }
        out.push(byte);
        if n == 0 {
            break;
        }
    }
}

fn push_string(out: &mut Vec<u8>, text: &str) {
    let bytes = text.as_bytes();
    out.extend_from_slice(&(bytes.len() as u16).to_be_bytes());
    out.extend_from_slice(bytes);
}

fn connect_packet(client_id: &str, user: &str, password: &str, keepalive_s: u16) -> Vec<u8> {
    let mut body = Vec::new();
    push_string(&mut body, "MQTT");
    body.push(4); // protocol level 3.1.1
    body.push(0x80 | 0x40 | 0x02); // username, password, clean session
    body.extend_from_slice(&keepalive_s.to_be_bytes());
    push_string(&mut body, client_id);
    push_string(&mut body, user);
    push_string(&mut body, password);
    let mut packet = vec![0x10];
    encode_remaining(body.len(), &mut packet);
    packet.extend_from_slice(&body);
    packet
}

fn subscribe_packet(packet_id: u16, topics: &[String], qos: u8) -> Vec<u8> {
    let mut body = Vec::new();
    body.extend_from_slice(&packet_id.to_be_bytes());
    for topic in topics {
        push_string(&mut body, topic);
        body.push(qos);
    }
    let mut packet = vec![0x82];
    encode_remaining(body.len(), &mut packet);
    packet.extend_from_slice(&body);
    packet
}

fn puback_packet(packet_id: u16) -> Vec<u8> {
    let mut packet = vec![0x40, 0x02];
    packet.extend_from_slice(&packet_id.to_be_bytes());
    packet
}

const PINGREQ: [u8; 2] = [0xC0, 0x00];
const DISCONNECT: [u8; 2] = [0xE0, 0x00];

/// One decoded control packet.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Packet {
    ConnAck { session_present: bool, return_code: u8 },
    SubAck { packet_id: u16, codes: Vec<u8> },
    Publish { topic: String, packet_id: Option<u16>, qos: u8, payload: Vec<u8> },
    PingResp,
    Other(u8),
}

/// Parse the packets a byte buffer holds, returning them and the count of
/// bytes consumed; a trailing partial packet is left for the next read.
fn parse_packets(buffer: &[u8]) -> Result<(Vec<Packet>, usize), Box<dyn Error>> {
    let mut packets = Vec::new();
    let mut at = 0usize;
    while at < buffer.len() {
        let first = buffer[at];
        // remaining length
        let mut multiplier = 1usize;
        let mut remaining = 0usize;
        let mut cursor = at + 1;
        let mut complete = false;
        for _ in 0..4 {
            let Some(&byte) = buffer.get(cursor) else { break };
            remaining += (byte as usize & 0x7F) * multiplier;
            multiplier *= 128;
            cursor += 1;
            if byte & 0x80 == 0 {
                complete = true;
                break;
            }
        }
        if !complete {
            break;
        }
        if cursor + remaining > buffer.len() {
            break;
        }
        let body = &buffer[cursor..cursor + remaining];
        let kind = first >> 4;
        let packet = match kind {
            2 => {
                if body.len() < 2 {
                    return Err(err("CONNACK shorter than two bytes"));
                }
                Packet::ConnAck { session_present: body[0] & 1 == 1, return_code: body[1] }
            }
            9 => {
                if body.len() < 2 {
                    return Err(err("SUBACK shorter than two bytes"));
                }
                Packet::SubAck {
                    packet_id: u16::from_be_bytes([body[0], body[1]]),
                    codes: body[2..].to_vec(),
                }
            }
            3 => {
                let qos = (first >> 1) & 0x03;
                if body.len() < 2 {
                    return Err(err("PUBLISH without a topic length"));
                }
                let topic_len = u16::from_be_bytes([body[0], body[1]]) as usize;
                if body.len() < 2 + topic_len {
                    return Err(err("PUBLISH topic runs past the packet"));
                }
                let topic = String::from_utf8_lossy(&body[2..2 + topic_len]).to_string();
                let mut rest = &body[2 + topic_len..];
                let packet_id = if qos > 0 {
                    if rest.len() < 2 {
                        return Err(err("PUBLISH at QoS > 0 without a packet id"));
                    }
                    let id = u16::from_be_bytes([rest[0], rest[1]]);
                    rest = &rest[2..];
                    Some(id)
                } else {
                    None
                };
                Packet::Publish { topic, packet_id, qos, payload: rest.to_vec() }
            }
            13 => Packet::PingResp,
            other => Packet::Other(other),
        };
        packets.push(packet);
        at = cursor + remaining;
    }
    Ok((packets, at))
}

/// A TLS connection to the broker with the client's read/write plumbing.
struct Broker {
    tcp: TcpStream,
    tls: rustls::ClientConnection,
    buffer: Vec<u8>,
}

impl Broker {
    fn connect(host: &str, port: u16) -> Result<Self, Box<dyn Error>> {
        let provider = Arc::new(rustls_rustcrypto::provider());
        let mut roots = rustls::RootCertStore::empty();
        roots.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
        let config = rustls::ClientConfig::builder_with_provider(provider)
            .with_safe_default_protocol_versions()
            .map_err(|e| err(format!("TLS versions: {e}")))?
            .with_root_certificates(roots)
            .with_no_client_auth();
        let server = rustls::pki_types::ServerName::try_from(host.to_string())
            .map_err(|e| err(format!("{host} is not a valid TLS server name: {e}")))?;
        let tls = rustls::ClientConnection::new(Arc::new(config), server)
            .map_err(|e| err(format!("TLS client: {e}")))?;
        let tcp = TcpStream::connect((host, port))
            .map_err(|e| err(format!("connect {host}:{port}: {e}")))?;
        tcp.set_read_timeout(Some(Duration::from_millis(500)))?;
        tcp.set_write_timeout(Some(Duration::from_secs(20)))?;
        tcp.set_nodelay(true)?;
        Ok(Self { tcp, tls, buffer: Vec::new() })
    }

    fn send(&mut self, packet: &[u8]) -> Result<(), Box<dyn Error>> {
        let mut stream = rustls::Stream::new(&mut self.tls, &mut self.tcp);
        stream.write_all(packet).map_err(|e| err(format!("send: {e}")))?;
        stream.flush().map_err(|e| err(format!("flush: {e}")))?;
        Ok(())
    }

    /// Read whatever is available (up to the socket timeout) and return the
    /// complete packets; a timeout is an empty answer, not an error.
    fn poll(&mut self) -> Result<Vec<Packet>, Box<dyn Error>> {
        let mut chunk = [0u8; 65536];
        {
            let mut stream = rustls::Stream::new(&mut self.tls, &mut self.tcp);
            match stream.read(&mut chunk) {
                Ok(0) => return Err(err("the broker closed the connection")),
                Ok(n) => self.buffer.extend_from_slice(&chunk[..n]),
                Err(e) if matches!(e.kind(), std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut) => {}
                Err(e) => return Err(err(format!("read: {e}"))),
            }
        }
        let (packets, consumed) = parse_packets(&self.buffer)?;
        self.buffer.drain(..consumed);
        Ok(packets)
    }

    /// Poll until a packet the predicate accepts arrives or the deadline
    /// passes; other packets are returned in order for the caller.
    fn wait_for(&mut self, deadline: Instant, want: impl Fn(&Packet) -> bool) -> Result<(Packet, Vec<Packet>), Box<dyn Error>> {
        let mut others = Vec::new();
        while Instant::now() < deadline {
            for packet in self.poll()? {
                if want(&packet) {
                    return Ok((packet, others));
                }
                others.push(packet);
            }
        }
        Err(err("the broker did not answer within the deadline"))
    }
}

fn handshake(options: &Options) -> Result<(Broker, f64, f64, Vec<u8>), Box<dyn Error>> {
    let (host, port) = options.broker()?;
    let t0 = Instant::now();
    let mut broker = Broker::connect(&host, port)?;
    let client_id = format!("gpuwm-rw-wis2-{}", Utc::now().format("%Y%m%d%H%M%S%f"));
    broker.send(&connect_packet(&client_id, DEFAULT_USER, DEFAULT_PASSWORD, KEEPALIVE_S))?;
    let (ack, _) = broker.wait_for(Instant::now() + Duration::from_secs(20), |p| matches!(p, Packet::ConnAck { .. }))?;
    let connect_wall = t0.elapsed().as_secs_f64();
    if let Packet::ConnAck { return_code, .. } = ack {
        if return_code != 0 {
            return Err(err(format!(
                "the broker refused the connection (CONNACK return code {return_code}); \
                 the public credentials are {DEFAULT_USER}/{DEFAULT_PASSWORD}"
            )));
        }
    }
    let t1 = Instant::now();
    let topics = options.topics();
    broker.send(&subscribe_packet(1, &topics, 1))?;
    let (suback, _) = broker.wait_for(Instant::now() + Duration::from_secs(20), |p| matches!(p, Packet::SubAck { .. }))?;
    let subscribe_wall = t1.elapsed().as_secs_f64();
    let codes = match suback {
        Packet::SubAck { codes, .. } => codes,
        _ => Vec::new(),
    };
    if codes.iter().any(|c| *c == 0x80) {
        return Err(err(format!("the broker refused a topic filter (SUBACK codes {codes:?}) for {topics:?}")));
    }
    Ok((broker, connect_wall, subscribe_wall, codes))
}

// -------------------------------------------------------------------- dump

/// Every decoded element of the named payloads, one line each (the
/// diagnostic route: what a centre's message carries, and under which
/// element the reader ran out when it refuses one).
fn cmd_dump(options: &Options) -> Result<String, Box<dyn Error>> {
    if options.payloads.is_empty() {
        return Err(err("--payloads is required"));
    }
    let mut out = String::new();
    for path in &options.payloads {
        let bytes = std::fs::read(path).map_err(|e| err(format!("cannot read {}: {e}", path.display())))?;
        let subject = path.display().to_string();
        let tables = bufr::Tables::get();
        let describe = |items: &[bufr::Item], out: &mut String| {
            for (n, item) in items.iter().enumerate() {
                let name = tables.element(item.code).map(|e| e.abbreviation).unwrap_or("");
                out.push_str(&format!("  {n:5} {:06} {name:40} {:?} path={:?}
", item.code, item.value, item.path));
            }
        };
        match bufr::decode_with_partial(&bytes, &subject) {
            Ok(message) => {
                out.push_str(&format!(
                    "{subject}: edition {} centre {}/{} category {} master table version {} subsets {} compressed {} descriptors {:?}
",
                    message.edition, message.centre, message.subcentre, message.data_category,
                    message.master_table_version, message.subsets.len(), message.compressed,
                    message.descriptors.iter().map(|d| format!("{d:06}")).collect::<Vec<_>>()
                ));
                for (k, subset) in message.subsets.iter().enumerate() {
                    out.push_str(&format!(" subset {} ({} items)
", k + 1, subset.len()));
                    describe(subset, &mut out);
                }
            }
            Err((e, partial)) => {
                out.push_str(&format!("{subject}: REFUSED: {e}
"));
                for (k, subset) in partial.iter().enumerate() {
                    out.push_str(&format!(" subset {} ({} items decoded before the refusal; the first 120 and the last 40)
", k + 1, subset.len()));
                    describe(&subset[..subset.len().min(120)], &mut out);
                    if subset.len() > 120 {
                        out.push_str("  ...
");
                        let tail = subset.len().saturating_sub(40).max(120);
                        describe(&subset[tail..], &mut out);
                    }
                }
            }
        }
    }
    Ok(out)
}

// ------------------------------------------------------------------- table

#[derive(Serialize)]
struct PayloadRecord {
    path: String,
    centre: String,
    bytes: usize,
    sha256: String,
    messages: usize,
    subsets: usize,
    rows: usize,
    kind: Option<bufr_obs::Kind>,
    master_table_version: Option<u8>,
    error: Option<String>,
}

/// The rows of the archived (or named) payloads: every BUFR message of
/// every payload through the reader and the template layer, the window
/// applied to the rows' own times.
fn cmd_table(options: &Options) -> Result<String, Box<dyn Error>> {
    let out = options.out.as_deref().ok_or_else(|| err("--out FILE.csv is required"))?;
    if out.is_dir() {
        return Err(err(format!("--out {} is a directory; give the CSV path", out.display())));
    }
    let window = |raw: Option<&str>, flag: &str| -> Result<Option<DateTime<Utc>>, Box<dyn Error>> {
        match raw {
            None => Ok(None),
            Some(text) => parse_time(text).map(Some).map_err(|e| err(format!("{flag}: {e}"))),
        }
    };
    let start = window(options.start.as_deref(), "--start")?;
    let end = window(options.end.as_deref(), "--end")?;
    // The payload files and, from a subscribe archive, each one's
    // notification: pubtime (published), receipt (received), digest.
    let mut files: Vec<(PathBuf, String)> = Vec::new();
    let mut notes: BTreeMap<String, (Option<DateTime<Utc>>, Option<DateTime<Utc>>, Option<String>)> = BTreeMap::new();
    if let Some(archive) = &options.archive {
        let record_path = archive.join("record.json");
        let text = std::fs::read_to_string(&record_path)
            .map_err(|e| err(format!("cannot read {} (a subscribe archive carries record.json): {e}", record_path.display())))?;
        let record: serde_json::Value = serde_json::from_str(&text).map_err(|e| err(format!("{} is not JSON: {e}", record_path.display())))?;
        if record.get("schema").and_then(|s| s.as_str()) != Some(SUBSCRIBE_SCHEMA) {
            return Err(err(format!("{} does not declare {SUBSCRIBE_SCHEMA}", record_path.display())));
        }
        for note in record.get("notifications").and_then(|n| n.as_array()).into_iter().flatten() {
            let Some(path) = note.get("payload_path").and_then(|v| v.as_str()) else { continue };
            let name = Path::new(path).file_name().and_then(|n| n.to_str()).unwrap_or(path).to_string();
            let published = note.get("pubtime").and_then(|v| v.as_str()).and_then(|t| parse_time(t).ok())
                .or_else(|| note.get("pubtime").and_then(|v| v.as_str()).and_then(first_instant));
            let received = note.get("received").and_then(|v| v.as_str()).and_then(|t| parse_time(t).ok());
            let sha = note.get("payload_sha256").and_then(|v| v.as_str()).map(str::to_string);
            notes.insert(name, (published, received, sha));
        }
        let payloads_dir = archive.join("payloads");
        let mut walk: Vec<PathBuf> = Vec::new();
        if payloads_dir.is_dir() {
            for centre in std::fs::read_dir(&payloads_dir)? {
                let centre = centre?.path();
                if centre.is_dir() {
                    for entry in std::fs::read_dir(&centre)? {
                        let p = entry?.path();
                        if p.is_file() {
                            walk.push(p);
                        }
                    }
                }
            }
        }
        walk.sort();
        for p in walk {
            let centre = p.parent().and_then(|c| c.file_name()).and_then(|c| c.to_str()).unwrap_or("unknown").to_string();
            files.push((p, centre));
        }
    }
    for p in &options.payloads {
        files.push((p.clone(), "named".to_string()));
    }
    if files.is_empty() {
        return Err(err("--archive DIR (with payloads) or --payloads is required"));
    }
    let mut counters = bufr_obs::Counters::default();
    let mut writer = TableWriter::new();
    let mut records = Vec::new();
    let mut payloads_decoded = 0usize;
    let mut payloads_unreadable = 0usize;
    let mut rows_outside_window = 0usize;
    let mut by_centre: BTreeMap<String, usize> = BTreeMap::new();
    let mut by_kind: BTreeMap<String, usize> = BTreeMap::new();
    let mut unreadable_reasons: BTreeMap<String, usize> = BTreeMap::new();
    for (path, centre) in &files {
        let bytes = match std::fs::read(path) {
            Ok(b) => b,
            Err(e) => {
                payloads_unreadable += 1;
                records.push(PayloadRecord {
                    path: rw_obs::absolute_uri(path), centre: centre.clone(), bytes: 0, sha256: String::new(),
                    messages: 0, subsets: 0, rows: 0, kind: None, master_table_version: None,
                    error: Some(format!("cannot read: {e}")),
                });
                continue;
            }
        };
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("").to_string();
        let sha = notes.get(&name).and_then(|n| n.2.clone()).unwrap_or_else(|| hex_sha256(&bytes));
        let (published, received) = notes.get(&name).map(|n| (n.0, n.1)).unwrap_or((None, None));
        let provenance = RowProvenance::of_source(&sha, published, received);
        let subject = format!("{centre}/{name}");
        match bufr::decode_all(&bytes, &subject) {
            Ok(messages) => {
                payloads_decoded += 1;
                let before = writer.len();
                let mut subsets = 0usize;
                let mut kind = None;
                let mut mtv = None;
                for message in &messages {
                    subsets += message.subsets.len();
                    kind = Some(bufr_obs::kind_of(&message.descriptors));
                    mtv = Some(message.master_table_version);
                    for row in bufr_obs::rows_of(message, &provenance, &mut counters) {
                        let inside = start.is_none_or(|s| row.valid_time >= s) && end.is_none_or(|e| row.valid_time <= e);
                        if inside {
                            writer.push(row, &mut BTreeMap::new());
                        } else {
                            rows_outside_window += 1;
                        }
                    }
                }
                let rows = writer.len() - before;
                *by_centre.entry(centre.clone()).or_insert(0) += rows;
                if let Some(k) = kind {
                    *by_kind.entry(format!("{k:?}")).or_insert(0) += 1;
                }
                records.push(PayloadRecord {
                    path: rw_obs::absolute_uri(path), centre: centre.clone(), bytes: bytes.len(), sha256: sha,
                    messages: messages.len(), subsets, rows, kind, master_table_version: mtv, error: None,
                });
            }
            Err(e) => {
                payloads_unreadable += 1;
                let reason = e.to_string();
                // The reason without the payload name, so like refusals pool.
                let key = reason.split(':').skip(1).collect::<Vec<_>>().join(":").trim().to_string();
                *unreadable_reasons.entry(if key.is_empty() { reason.clone() } else { key }).or_insert(0) += 1;
                records.push(PayloadRecord {
                    path: rw_obs::absolute_uri(path), centre: centre.clone(), bytes: bytes.len(), sha256: sha,
                    messages: 0, subsets: 0, rows: 0, kind: None, master_table_version: None, error: Some(reason),
                });
            }
        }
    }
    // The writer's per-variable count was bypassed above (the window filter
    // pushes rows one by one); the counters carry rows_by_variable from the
    // template layer, before the window.
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
        window_start: Option<String>,
        window_end: Option<String>,
        payloads: usize,
        payloads_decoded: usize,
        payloads_unreadable: usize,
        rows_outside_window: usize,
        rows_by_centre: BTreeMap<String, usize>,
        payloads_by_kind: BTreeMap<String, usize>,
        unreadable_reasons: BTreeMap<String, usize>,
        master_table_version_of_reader: u8,
        counters: bufr_obs::Counters,
        errors: BTreeMap<&'static str, f64>,
        measurements: Vec<&'static str>,
        records: Vec<PayloadRecord>,
    }
    let mut errors = BTreeMap::new();
    errors.insert("surface_pressure_pa", rw_obs::table::ERROR_SURFACE_PRESSURE_PA);
    errors.insert("temperature_surface_k", rw_obs::table::ERROR_TEMPERATURE_SURFACE_K);
    errors.insert("dewpoint_surface_k", rw_obs::table::ERROR_DEWPOINT_SURFACE_K);
    errors.insert("wind_surface_m_s", rw_obs::table::ERROR_WIND_SURFACE_M_S);
    errors.insert("temperature_sonde_k", rw_obs::table::ERROR_TEMPERATURE_ALOFT_K);
    errors.insert("dewpoint_sonde_k", rw_obs::table::ERROR_DEWPOINT_ALOFT_K);
    errors.insert("wind_sonde_m_s", rw_obs::table::ERROR_WIND_ALOFT_M_S);
    let text = format!(
        "{}\n",
        serde_json::to_string_pretty(&Record {
            schema: TABLE_RECORD_SCHEMA,
            status: if rows > 0 { "READY" } else { "EMPTY" },
            source: bufr_obs::SOURCE,
            table_schema: TABLE_SCHEMA,
            path: rw_obs::absolute_uri(out),
            sha256: csv_sha,
            rows,
            bytes: csv_bytes,
            window_start: start.map(seam_time),
            window_end: end.map(seam_time),
            payloads: files.len(),
            payloads_decoded,
            payloads_unreadable,
            rows_outside_window,
            rows_by_centre: by_centre,
            payloads_by_kind: by_kind,
            unreadable_reasons,
            master_table_version_of_reader: bufr::MASTER_TABLE_VERSION,
            counters,
            errors,
            measurements: vec![
                rw_obs::table::MEAS_STATION_PRESSURE, rw_obs::table::MEAS_SEA_LEVEL_PRESSURE,
                rw_obs::table::MEAS_SCREEN_TEMPERATURE_2M, rw_obs::table::MEAS_SCREEN_DEWPOINT_2M,
                rw_obs::table::MEAS_PLATFORM_TEMPERATURE, rw_obs::table::MEAS_PLATFORM_DEWPOINT,
                rw_obs::table::MEAS_ANEMOMETER_WIND_10M, rw_obs::table::MEAS_SONDE_LEVEL,
            ],
            records,
        })?
    );
    std::fs::write(out.with_extension("json"), &text).map_err(|e| err(format!("cannot write the table record: {e}")))?;
    Ok(text)
}

fn cmd_probe(options: &Options) -> Result<String, Box<dyn Error>> {
    let (mut broker, connect_wall, subscribe_wall, codes) = handshake(options)?;
    let _ = broker.send(&DISCONNECT);
    #[derive(Serialize)]
    struct Record {
        schema: &'static str,
        status: &'static str,
        broker: String,
        topics: Vec<String>,
        connect_wall_s: f64,
        subscribe_wall_s: f64,
        suback_codes: Vec<u8>,
        probed_at: String,
    }
    Ok(format!(
        "{}\n",
        serde_json::to_string_pretty(&Record {
            schema: PROBE_SCHEMA,
            status: "READY",
            broker: options.broker.clone().unwrap_or_else(|| DEFAULT_BROKER.to_string()),
            topics: options.topics(),
            connect_wall_s: connect_wall,
            subscribe_wall_s: subscribe_wall,
            suback_codes: codes,
            probed_at: seam_time(Utc::now()),
        })?
    ))
}

// ------------------------------------------------------------ notifications

/// What one WIS2 notification says about itself.
#[derive(Debug, Clone, Serialize)]
struct Notification {
    topic: String,
    centre: String,
    topic_suffix: String,
    message_id: String,
    data_id: String,
    datetime: Option<String>,
    pubtime: Option<String>,
    received: String,
    canonical_href: Option<String>,
    canonical_type: Option<String>,
    integrity_method: Option<String>,
    integrity_value: Option<String>,
    /// pubtime minus the report's datetime, seconds (the source's publication delay).
    publication_delay_s: Option<i64>,
    /// receipt here minus pubtime, seconds (the broker path).
    transport_delay_s: Option<i64>,
    message_path: String,
    payload_path: Option<String>,
    payload_bytes: Option<usize>,
    payload_sha256: Option<String>,
    integrity_verified: Option<bool>,
    download_error: Option<String>,
}

fn first_instant(text: &str) -> Option<DateTime<Utc>> {
    // a datetime may be an interval "start/end"; the start is the report's time
    let head = text.split('/').next()?.trim();
    parse_time(head).ok()
}

fn sanitise(name: &str) -> String {
    let mut out: String = name
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '.' || c == '-' || c == '_' { c } else { '_' })
        .collect();
    if out.len() > 180 {
        out.truncate(180);
    }
    if out.is_empty() {
        "unnamed".to_string()
    } else {
        out
    }
}

fn topic_parts(topic: &str) -> (String, String) {
    // cache/a/wis2/<centre-id>/data/core/weather/surface-based-observations/<suffix>
    let parts: Vec<&str> = topic.split('/').collect();
    let centre = parts.get(3).map(|s| s.to_string()).unwrap_or_else(|| "unknown".to_string());
    let suffix = if parts.len() > 8 { parts[8..].join("/") } else { String::new() };
    (centre, suffix)
}

fn verify_integrity(method: &str, value_b64: &str, payload: &[u8]) -> Option<bool> {
    use base64::Engine;
    use sha2::Digest;

    let expected = base64::engine::general_purpose::STANDARD
        .decode(value_b64)
        .or_else(|_| base64::engine::general_purpose::URL_SAFE.decode(value_b64))
        .ok()?;
    let actual: Vec<u8> = match method.to_ascii_lowercase().as_str() {
        "sha512" => sha2::Sha512::digest(payload).to_vec(),
        "sha256" => sha2::Sha256::digest(payload).to_vec(),
        "sha384" => sha2::Sha384::digest(payload).to_vec(),
        _ => return None,
    };
    Some(actual == expected)
}

fn download(agent: &ureq::Agent, url: &str) -> Result<Vec<u8>, Box<dyn Error>> {
    let mut response = rw_http::get(agent, url).map_err(|e| err(format!("GET {url}: {e}")))?;
    let status = response.status().as_u16();
    if !(200..300).contains(&status) {
        return Err(err(format!("GET {url} answered HTTP {status}")));
    }
    let bytes = response
        .body_mut()
        .with_config()
        .limit(MAX_PAYLOAD_BYTES)
        .read_to_vec()
        .map_err(|e| err(format!("reading {url}: {e}")))?;
    Ok(bytes)
}

#[derive(Debug, Default, Serialize)]
struct Coverage {
    messages: usize,
    messages_not_json: usize,
    messages_without_canonical_link: usize,
    payloads_downloaded: usize,
    payloads_bytes: usize,
    download_failures: usize,
    downloads_skipped: usize,
    integrity_verified: usize,
    integrity_failed: usize,
    integrity_unchecked: usize,
    messages_by_centre: BTreeMap<String, usize>,
    messages_by_topic_suffix: BTreeMap<String, usize>,
    payload_types: BTreeMap<String, usize>,
}

#[derive(Debug, Default, Serialize)]
struct LatencyStats {
    count: usize,
    min_s: Option<i64>,
    median_s: Option<i64>,
    max_s: Option<i64>,
}

fn stats(values: &mut Vec<i64>) -> LatencyStats {
    if values.is_empty() {
        return LatencyStats::default();
    }
    values.sort_unstable();
    LatencyStats {
        count: values.len(),
        min_s: values.first().copied(),
        median_s: values.get(values.len() / 2).copied(),
        max_s: values.last().copied(),
    }
}

fn cmd_subscribe(options: &Options) -> Result<String, Box<dyn Error>> {
    let out = options.out.as_deref().ok_or_else(|| err("--out DIR is required"))?;
    std::fs::create_dir_all(out.join("messages")).map_err(|e| err(format!("cannot create {}: {e}", out.display())))?;
    std::fs::create_dir_all(out.join("payloads"))?;
    let seconds = options.seconds.unwrap_or(DEFAULT_SECONDS);
    let (mut broker, connect_wall, subscribe_wall, _) = handshake(options)?;
    let agent = build_agent();
    let started = Instant::now();
    let started_at = Utc::now();
    let deadline = started + Duration::from_secs(seconds);
    let mut last_ping = Instant::now();
    let mut coverage = Coverage::default();
    let mut notifications: Vec<Notification> = Vec::new();
    let mut publication_delays = Vec::new();
    let mut transport_delays = Vec::new();
    let mut downloads = 0usize;
    let mut disconnected: Option<String> = None;
    while Instant::now() < deadline {
        if last_ping.elapsed() >= Duration::from_secs(u64::from(KEEPALIVE_S) / 2) {
            broker.send(&PINGREQ)?;
            last_ping = Instant::now();
        }
        let packets = match broker.poll() {
            Ok(p) => p,
            Err(e) => {
                disconnected = Some(e.to_string());
                break;
            }
        };
        for packet in packets {
            let Packet::Publish { topic, packet_id, qos, payload } = packet else { continue };
            if qos > 0 {
                if let Some(id) = packet_id {
                    broker.send(&puback_packet(id))?;
                }
            }
            let received = Utc::now();
            coverage.messages += 1;
            let (centre, suffix) = topic_parts(&topic);
            *coverage.messages_by_centre.entry(centre.clone()).or_insert(0) += 1;
            *coverage.messages_by_topic_suffix.entry(suffix.clone()).or_insert(0) += 1;
            let document: serde_json::Value = match serde_json::from_slice(&payload) {
                Ok(v) => v,
                Err(_) => {
                    coverage.messages_not_json += 1;
                    let path = out.join("messages").join(&centre).join(format!("{}-notjson.bin", received.format("%Y%m%dT%H%M%S%f")));
                    std::fs::create_dir_all(path.parent().unwrap())?;
                    std::fs::write(&path, &payload)?;
                    continue;
                }
            };
            let message_id = document.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string();
            let properties = document.get("properties").cloned().unwrap_or(serde_json::Value::Null);
            let data_id = properties.get("data_id").and_then(|v| v.as_str()).unwrap_or("").to_string();
            let datetime = properties
                .get("datetime")
                .or_else(|| properties.get("start_datetime"))
                .and_then(|v| v.as_str())
                .map(str::to_string);
            let pubtime = properties.get("pubtime").and_then(|v| v.as_str()).map(str::to_string);
            let (integrity_method, integrity_value) = match properties.get("integrity") {
                Some(i) => (
                    i.get("method").and_then(|v| v.as_str()).map(str::to_string),
                    i.get("value").and_then(|v| v.as_str()).map(str::to_string),
                ),
                None => (None, None),
            };
            let canonical = document
                .get("links")
                .and_then(|l| l.as_array())
                .and_then(|links| {
                    links.iter().find(|l| l.get("rel").and_then(|r| r.as_str()) == Some("canonical"))
                        .or_else(|| links.first())
                });
            let canonical_href = canonical.and_then(|l| l.get("href")).and_then(|v| v.as_str()).map(str::to_string);
            let canonical_type = canonical.and_then(|l| l.get("type")).and_then(|v| v.as_str()).map(str::to_string);
            if canonical_href.is_none() {
                coverage.messages_without_canonical_link += 1;
            }
            if let Some(t) = &canonical_type {
                *coverage.payload_types.entry(t.clone()).or_insert(0) += 1;
            }
            let dt_instant = datetime.as_deref().and_then(first_instant);
            let pub_instant = pubtime.as_deref().and_then(|t| parse_time(t).ok());
            let publication_delay_s = match (dt_instant, pub_instant) {
                (Some(d), Some(p)) => Some((p - d).num_seconds()),
                _ => None,
            };
            let transport_delay_s = pub_instant.map(|p| (received - p).num_seconds());
            if let Some(v) = publication_delay_s {
                publication_delays.push(v);
            }
            if let Some(v) = transport_delay_s {
                transport_delays.push(v);
            }
            // Archive the original message the moment it is understood.
            let stamp = received.format("%Y%m%dT%H%M%S%f");
            let message_name = format!("{stamp}-{}.json", sanitise(if message_id.is_empty() { "noid" } else { &message_id }));
            let message_path = out.join("messages").join(&centre).join(message_name);
            std::fs::create_dir_all(message_path.parent().unwrap())?;
            std::fs::write(&message_path, &payload)?;
            let mut note = Notification {
                topic: topic.clone(),
                centre: centre.clone(),
                topic_suffix: suffix,
                message_id,
                data_id: data_id.clone(),
                datetime,
                pubtime,
                received: seam_time(received),
                canonical_href: canonical_href.clone(),
                canonical_type,
                integrity_method: integrity_method.clone(),
                integrity_value: integrity_value.clone(),
                publication_delay_s,
                transport_delay_s,
                message_path: rw_obs::absolute_uri(&message_path),
                payload_path: None,
                payload_bytes: None,
                payload_sha256: None,
                integrity_verified: None,
                download_error: None,
            };
            let may_download = !options.no_download
                && options.max_downloads.is_none_or(|n| downloads < n)
                && Instant::now() < deadline;
            match (&canonical_href, may_download) {
                (Some(href), true) => {
                    downloads += 1;
                    match download(&agent, href) {
                        Ok(bytes) => {
                            let file_name = sanitise(
                                Path::new(href).file_name().and_then(|n| n.to_str()).unwrap_or(&data_id),
                            );
                            let payload_path = out.join("payloads").join(&centre).join(format!("{stamp}-{file_name}"));
                            std::fs::create_dir_all(payload_path.parent().unwrap())?;
                            std::fs::write(&payload_path, &bytes)?;
                            coverage.payloads_downloaded += 1;
                            coverage.payloads_bytes += bytes.len();
                            note.payload_bytes = Some(bytes.len());
                            note.payload_sha256 = Some(hex_sha256(&bytes));
                            note.payload_path = Some(rw_obs::absolute_uri(&payload_path));
                            match (&integrity_method, &integrity_value) {
                                (Some(m), Some(v)) => match verify_integrity(m, v, &bytes) {
                                    Some(true) => {
                                        coverage.integrity_verified += 1;
                                        note.integrity_verified = Some(true);
                                    }
                                    Some(false) => {
                                        coverage.integrity_failed += 1;
                                        note.integrity_verified = Some(false);
                                    }
                                    None => coverage.integrity_unchecked += 1,
                                },
                                _ => coverage.integrity_unchecked += 1,
                            }
                        }
                        Err(e) => {
                            coverage.download_failures += 1;
                            note.download_error = Some(e.to_string());
                        }
                    }
                }
                (Some(_), false) => coverage.downloads_skipped += 1,
                (None, _) => {}
            }
            notifications.push(note);
        }
    }
    let _ = broker.send(&DISCONNECT);
    let wall = started.elapsed().as_secs_f64();
    let publication = stats(&mut publication_delays);
    let transport = stats(&mut transport_delays);
    let latency_behind_real_time_s = match (publication.median_s, transport.median_s) {
        (Some(p), Some(t)) => Some(p + t),
        _ => None,
    };

    #[derive(Serialize)]
    struct Record {
        schema: &'static str,
        status: &'static str,
        broker: String,
        topics: Vec<String>,
        started_at: String,
        listened_s: f64,
        connect_wall_s: f64,
        subscribe_wall_s: f64,
        disconnected: Option<String>,
        out_dir: String,
        coverage: Coverage,
        publication_delay_s: LatencyStats,
        transport_delay_s: LatencyStats,
        latency_behind_real_time_s: Option<i64>,
        latency_basis: &'static str,
        messages_per_minute: f64,
        decoded_into_table: bool,
        note: &'static str,
        notifications: Vec<Notification>,
    }
    let record = Record {
        schema: SUBSCRIBE_SCHEMA,
        status: if coverage.messages > 0 { "READY" } else { "EMPTY" },
        broker: options.broker.clone().unwrap_or_else(|| DEFAULT_BROKER.to_string()),
        topics: options.topics(),
        started_at: seam_time(started_at),
        listened_s: wall,
        connect_wall_s: connect_wall,
        subscribe_wall_s: subscribe_wall,
        disconnected,
        out_dir: rw_obs::absolute_uri(out),
        messages_per_minute: coverage.messages as f64 / (wall / 60.0).max(1.0e-9),
        coverage,
        publication_delay_s: publication,
        transport_delay_s: transport,
        latency_behind_real_time_s,
        latency_basis: "median (pubtime - report datetime) + median (receipt - pubtime) over the listened window",
        decoded_into_table: false,
        note: "payloads are archived and counted as bytes here; `rw_wis2 table --archive DIR` decodes them into the neutral table",
        notifications,
    };
    let text = format!("{}\n", serde_json::to_string_pretty(&record)?);
    std::fs::write(out.join("record.json"), &text).map_err(|e| err(format!("cannot write the record: {e}")))?;
    Ok(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remaining_length_encodes_like_the_specification() {
        let mut v = Vec::new();
        encode_remaining(0, &mut v);
        assert_eq!(v, vec![0]);
        v.clear();
        encode_remaining(127, &mut v);
        assert_eq!(v, vec![0x7F]);
        v.clear();
        encode_remaining(128, &mut v);
        assert_eq!(v, vec![0x80, 0x01]);
        v.clear();
        encode_remaining(16_383, &mut v);
        assert_eq!(v, vec![0xFF, 0x7F]);
    }

    #[test]
    fn connect_and_subscribe_packets_have_the_documented_shape() {
        let c = connect_packet("id", "everyone", "everyone", 60);
        assert_eq!(c[0], 0x10);
        // remaining = 2+4 + 1 + 1 + 2 + (2+2) + (2+8) + (2+8) = 34
        assert_eq!(c[1], 34);
        assert_eq!(&c[2..8], &[0, 4, b'M', b'Q', b'T', b'T']);
        assert_eq!(c[8], 4);
        assert_eq!(c[9], 0xC2);
        assert_eq!(&c[10..12], &[0, 60]);
        let s = subscribe_packet(7, &["a/#".to_string()], 1);
        assert_eq!(s[0], 0x82);
        assert_eq!(&s[2..4], &[0, 7]);
        assert_eq!(&s[4..6], &[0, 3]);
        assert_eq!(&s[6..9], b"a/#");
        assert_eq!(s[9], 1);
    }

    #[test]
    fn publish_packets_parse_with_and_without_a_packet_id_and_partials_wait() {
        // QoS 1 PUBLISH: topic "t/x", packet id 5, payload "{}"
        let mut p = vec![0x32, 0x00];
        let body: Vec<u8> = [&[0u8, 3][..], b"t/x", &[0, 5][..], b"{}"].concat();
        p[1] = body.len() as u8;
        p.extend_from_slice(&body);
        // then a PINGRESP and half of another packet
        p.extend_from_slice(&[0xD0, 0x00, 0x30]);
        let (packets, consumed) = parse_packets(&p).unwrap();
        assert_eq!(packets.len(), 2);
        assert_eq!(
            packets[0],
            Packet::Publish { topic: "t/x".into(), packet_id: Some(5), qos: 1, payload: b"{}".to_vec() }
        );
        assert_eq!(packets[1], Packet::PingResp);
        assert_eq!(consumed, p.len() - 1);
        let ack = parse_packets(&[0x20, 0x02, 0x00, 0x00]).unwrap().0;
        assert_eq!(ack, vec![Packet::ConnAck { session_present: false, return_code: 0 }]);
        let sub = parse_packets(&[0x90, 0x03, 0x00, 0x01, 0x01]).unwrap().0;
        assert_eq!(sub, vec![Packet::SubAck { packet_id: 1, codes: vec![1] }]);
    }

    #[test]
    fn topic_parts_name_the_centre_and_the_suffix() {
        let (centre, suffix) = topic_parts("cache/a/wis2/de-dwd/data/core/weather/surface-based-observations/synop");
        assert_eq!(centre, "de-dwd");
        assert_eq!(suffix, "synop");
        assert_eq!(topic_parts("x").0, "unknown");
    }

    #[test]
    fn integrity_checks_sha512_and_sha256_and_declines_unknown_methods() {
        use base64::Engine;
        use sha2::Digest;
        let payload = b"BUFR....";
        let value = base64::engine::general_purpose::STANDARD.encode(sha2::Sha512::digest(payload));
        assert_eq!(verify_integrity("sha512", &value, payload), Some(true));
        assert_eq!(verify_integrity("sha512", &value, b"other"), Some(false));
        let v256 = base64::engine::general_purpose::STANDARD.encode(sha2::Sha256::digest(payload));
        assert_eq!(verify_integrity("sha256", &v256, payload), Some(true));
        assert_eq!(verify_integrity("md5", &v256, payload), None);
    }

    #[test]
    fn datetimes_take_the_start_of_an_interval() {
        assert_eq!(
            first_instant("2026-09-06T00:00:00Z/2026-09-06T01:00:00Z"),
            Some(parse_time("2026-09-06T00:00:00Z").unwrap())
        );
        assert!(first_instant("not a time").is_none());
        assert_eq!(sanitise("a/b:c d.bufr"), "a_b_c_d.bufr");
    }

    #[test]
    fn abi_marker_names_the_contracts_it_pins() {
        assert!(ABI_MARKER.contains(SUBSCRIBE_SCHEMA));
        assert!(ABI_MARKER.contains(PROBE_SCHEMA));
        assert!(ABI_MARKER.contains(TABLE_RECORD_SCHEMA));
        assert!(ABI_MARKER.contains(TABLE_SCHEMA));
    }
}
