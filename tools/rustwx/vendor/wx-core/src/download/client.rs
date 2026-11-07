use std::fs::{self, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use ureq::http::header::{CONTENT_LENGTH, CONTENT_RANGE, LOCATION};
use ureq::unversioned::resolver::DefaultResolver;
use ureq::unversioned::transport::time::Duration as WaitDuration;
use ureq::unversioned::transport::{
    Buffers, ConnectionDetails, Connector, DefaultConnector, NextTimeout, Transport,
};

use super::cache::DiskCache;

/// HTTP client for downloading GRIB2 data with byte-range support.
///
/// Uses ureq (blocking HTTP) with rustls + rustcrypto for TLS.
/// Supports a stall limit in place of a whole-request timeout, retry
/// with exponential backoff, chunk downloads over a bounded number of
/// streams that resume where a broken one stopped, byte progress, and
/// optional disk caching.
pub struct DownloadClient {
    agent: ureq::Agent,
    /// The same client with no connection pool: every request it sends
    /// opens a new connection.  Retries go through it, see
    /// [`DownloadClient::with_retry`].
    fresh: ureq::Agent,
    max_retries: u32,
    streams: usize,
    cache: Option<DiskCache>,
    progress: Option<Arc<TransferProgress>>,
}

/// Maximum body size for full file downloads.
///
/// Full HRRR/RRFS family files can exceed the older subset-oriented 500 MB cap,
/// especially `wrfnat`. Keep the cap comfortably above current operational
/// artifacts while still guarding against obviously runaway downloads.
const MAX_BODY_SIZE: u64 = 8 * 1024 * 1024 * 1024;

/// Chunk size for whole-file parallel range downloads.
const FULL_FILE_RANGE_CHUNK_BYTES: u64 = 16 * 1024 * 1024;

/// Longest wait to open a connection: name lookup, TCP and TLS.
const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(30);

/// A response that delivers fewer than [`DEFAULT_STALL_MIN_BYTES`] in any
/// window this long has stopped moving.
///
/// This replaces a 300 s limit on each WHOLE request.  That limit could
/// not tell a stalled stream from a slow one: a 16 MiB chunk still
/// arriving at 50 KiB/s was cut off at 300 s and asked for again from its
/// first byte, and a stream that stopped dead was waited on for the full
/// 300 s first.  On a 36 h HRRR fetch (74 whole files, 44 GB) one such
/// stream per file was enough to fail the run.  A stall is now judged by
/// what arrived, so a slow link finishes and a dead stream is dropped
/// after this window and resumed.
const DEFAULT_STALL_WINDOW: Duration = Duration::from_secs(30);

/// Bytes a response must deliver per stall window to count as moving.
///
/// 64 KiB per 30 s is about 2 KiB/s: far under any link that can carry a
/// weather file, far over a connection that has stopped.
const DEFAULT_STALL_MIN_BYTES: u64 = 64 * 1024;

/// Chunk streams one ranged transfer keeps open at once.
///
/// Was rayon's pool: one stream per CPU thread, so 24 streams per file on
/// a 24-thread machine and 96 on a 96-thread one, each holding a 16 MiB
/// chunk in memory, times every file a caller fetched at once (gpuwm runs
/// six).  Measured from a link about 100 ms from S3, one stream carries
/// about 1 MB/s, so a lone object still wants a dozen or more; 16 fills
/// that link for one object without scaling with the CPU.  A caller that
/// runs several transfers side by side divides its own budget and says so
/// through `RUSTWX_DOWNLOAD_STREAMS`.
const DEFAULT_STREAMS: usize = 16;

/// The most requests one chunk may take in all.
///
/// A resumed chunk that made progress does not spend the retry budget,
/// because the bytes it kept are the proof the link works.  This ceiling
/// is what still ends a chunk that keeps breaking after a few bytes.
const CHUNK_ATTEMPT_CEILING: u32 = 32;

/// Default maximum number of retries.
const DEFAULT_MAX_RETRIES: u32 = 3;

/// Maximum redirects we will follow manually.
///
/// NOMADS file URLs should generally be direct. We disable ureq's built-in
/// redirect handling so malformed upstream 3xx responses do not bubble up as
/// opaque protocol errors such as "missing a location header", then follow
/// only well-formed redirects ourselves.
const MAX_REDIRECTS: u32 = 10;

/// Backoff durations for each retry attempt.
const BACKOFF_DURATIONS: [Duration; 3] = [
    Duration::from_millis(500),
    Duration::from_millis(1000),
    Duration::from_millis(2000),
];

/// Longer backoff for the Akamai "Over Rate Limit" behavior seen on NOMADS.
const NOMADS_RATE_LIMIT_BACKOFF_DURATIONS: [Duration; 3] = [
    Duration::from_secs(5),
    Duration::from_secs(10),
    Duration::from_secs(20),
];

/// Default spacing between NOMADS requests across all RustWX processes on this node.
const NOMADS_DEFAULT_MIN_REQUEST_GAP: Duration = Duration::from_millis(2500);

/// If NOMADS returns its Akamai over-rate-limit page, pause all RustWX NOMADS
/// requests on this node long enough for the block to cool off.
const NOMADS_DEFAULT_COOLDOWN: Duration = Duration::from_secs(15 * 60);

const NOMADS_LOCK_STALE_AFTER: Duration = Duration::from_secs(120);

/// Configuration for creating a DownloadClient.
///
/// There is deliberately no limit on how long a whole request may take:
/// a large object on a slow link legitimately takes long, and what ends a
/// request that has gone wrong is `stall`.
pub struct DownloadConfig {
    /// Longest wait to open a connection (name lookup, TCP, TLS) and to
    /// send a request.
    pub connect_timeout: Duration,
    /// When an open connection counts as stopped.
    pub stall: StallLimit,
    /// Maximum number of retry attempts that make no progress.
    pub max_retries: u32,
    /// Chunk streams one ranged transfer keeps open at once (at least 1).
    pub streams: usize,
}

impl Default for DownloadConfig {
    fn default() -> Self {
        Self {
            connect_timeout: DEFAULT_CONNECT_TIMEOUT,
            stall: StallLimit::default(),
            max_retries: DEFAULT_MAX_RETRIES,
            streams: default_streams(),
        }
    }
}

/// A connection is stalled when fewer than `min_bytes` arrive within
/// `window` of its request going out, or of the last time `min_bytes`
/// had arrived.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StallLimit {
    pub window: Duration,
    pub min_bytes: u64,
}

impl Default for StallLimit {
    /// 64 KiB per 30 s, the window overridable through
    /// `RUSTWX_DOWNLOAD_STALL_SECONDS`.
    fn default() -> Self {
        Self {
            window: env_seconds("RUSTWX_DOWNLOAD_STALL_SECONDS").unwrap_or(DEFAULT_STALL_WINDOW),
            min_bytes: DEFAULT_STALL_MIN_BYTES,
        }
    }
}

fn env_seconds(name: &str) -> Option<Duration> {
    std::env::var(name)
        .ok()
        .and_then(|value| value.trim().parse::<u64>().ok())
        .filter(|seconds| *seconds > 0)
        .map(Duration::from_secs)
}

/// `RUSTWX_DOWNLOAD_STREAMS`, else [`DEFAULT_STREAMS`].
fn default_streams() -> usize {
    std::env::var("RUSTWX_DOWNLOAD_STREAMS")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|streams| *streams > 0)
        .unwrap_or(DEFAULT_STREAMS)
}

/// Payload bytes a client has received, readable from another thread.
///
/// A whole-file transfer is assembled in memory and written out only
/// when it is complete, so nothing on disk grows while it moves.  This is
/// what a caller reads instead: every body byte is counted as it arrives,
/// on whichever stream it arrives, and `total` is the object's size once
/// the transfer knows it.
#[derive(Debug, Default)]
pub struct TransferProgress {
    received: AtomicU64,
    total: AtomicU64,
}

impl TransferProgress {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Start counting a new object.
    pub fn reset(&self) {
        self.received.store(0, Ordering::Relaxed);
        self.total.store(0, Ordering::Relaxed);
    }

    pub fn add(&self, bytes: u64) {
        self.received.fetch_add(bytes, Ordering::Relaxed);
    }

    pub fn set_total(&self, total: u64) {
        self.total.store(total, Ordering::Relaxed);
    }

    /// Body bytes received since the last reset, retried spans included.
    pub fn received(&self) -> u64 {
        self.received.load(Ordering::Relaxed)
    }

    /// The object's size, once known.
    pub fn total(&self) -> Option<u64> {
        match self.total.load(Ordering::Relaxed) {
            0 => None,
            total => Some(total),
        }
    }
}

/// A body reader that counts what it hands over.
struct Counted<'a, R> {
    inner: R,
    progress: Option<&'a TransferProgress>,
}

impl<R: Read> Read for Counted<'_, R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let read = self.inner.read(buf)?;
        if let Some(progress) = self.progress {
            progress.add(read as u64);
        }
        Ok(read)
    }
}

/// Wraps every connection the agent opens in a [`StallGuard`].
#[derive(Debug)]
struct StallConnector(StallLimit);

impl Connector<Box<dyn Transport>> for StallConnector {
    type Out = StallGuard;

    fn connect(
        &self,
        _details: &ConnectionDetails,
        chained: Option<Box<dyn Transport>>,
    ) -> Result<Option<StallGuard>, ureq::Error> {
        Ok(chained.map(|inner| StallGuard::new(inner, self.0)))
    }
}

/// A transport that fails a connection which has stopped moving.
///
/// ureq's timeouts all count from a fixed point: the whole call, the
/// whole body.  None of them notices a body that arrived steadily for
/// four minutes and then stopped, until the whole budget is gone, and
/// every one of them also ends a body that is still arriving, only
/// slowly.  This sits outside TLS and caps each wait for input at the end
/// of the current window.  The window opens when a request goes out and
/// opens again each time `min_bytes` have arrived; reaching its end
/// therefore means fewer than that came in the last `window`, and the
/// wait fails as a timed-out read, which the chunk loop answers by
/// resuming the chunk on a new connection.
#[derive(Debug)]
struct StallGuard {
    inner: Box<dyn Transport>,
    limit: StallLimit,
    window_start: Instant,
    window_bytes: u64,
}

impl StallGuard {
    fn new(inner: Box<dyn Transport>, limit: StallLimit) -> Self {
        Self {
            inner,
            limit,
            window_start: Instant::now(),
            window_bytes: 0,
        }
    }

    fn stalled(&self) -> ureq::Error {
        ureq::Error::Io(std::io::Error::new(
            std::io::ErrorKind::TimedOut,
            format!(
                "the connection stalled: {} B arrived in the last {} s, under the {} B that count as moving",
                self.window_bytes,
                self.limit.window.as_secs_f64(),
                self.limit.min_bytes
            ),
        ))
    }
}

fn is_timeout(error: &ureq::Error) -> bool {
    match error {
        ureq::Error::Timeout(_) => true,
        ureq::Error::Io(io) => matches!(
            io.kind(),
            std::io::ErrorKind::TimedOut | std::io::ErrorKind::WouldBlock
        ),
        _ => false,
    }
}

impl Transport for StallGuard {
    fn buffers(&mut self) -> &mut dyn Buffers {
        self.inner.buffers()
    }

    fn transmit_output(&mut self, amount: usize, timeout: NextTimeout) -> Result<(), ureq::Error> {
        // A request is going out: the wait for its answer starts now,
        // whatever this pooled connection did before.
        self.window_start = Instant::now();
        self.window_bytes = 0;
        self.inner.transmit_output(amount, timeout)
    }

    fn await_input(&mut self, timeout: NextTimeout) -> Result<bool, ureq::Error> {
        loop {
            let now = Instant::now();
            // The window restarts each time `min_bytes` have arrived, so
            // reaching its end means fewer than that came in the last
            // `window`: a stream that stops is caught one window after its
            // last bytes, not up to two.
            let window_end = self.window_start + self.limit.window;
            if now >= window_end {
                return Err(self.stalled());
            }
            let left = window_end - now;
            let ours = left < *timeout.after;
            let capped = if ours {
                NextTimeout {
                    after: WaitDuration::Exact(left),
                    reason: timeout.reason,
                }
            } else {
                timeout
            };
            let before = self.inner.buffers().input().len();
            match self.inner.await_input(capped) {
                Ok(progress) => {
                    let after = self.inner.buffers().input().len();
                    self.window_bytes += after.saturating_sub(before) as u64;
                    if self.window_bytes >= self.limit.min_bytes {
                        self.window_start = Instant::now();
                        self.window_bytes = 0;
                    }
                    return Ok(progress);
                }
                // Our window ran out, not the caller's limit: judge it.
                Err(error) if ours && is_timeout(&error) => continue,
                Err(error) => return Err(error),
            }
        }
    }

    fn is_open(&mut self) -> bool {
        self.inner.is_open()
    }

    fn is_tls(&self) -> bool {
        self.inner.is_tls()
    }
}

/// Check whether an error from ureq should be retried.
///
/// Retries on: connection/transport errors, 429 (rate limit),
/// 500, 502, 503, 504 (server errors).
/// Does NOT retry on: 400, 404, or other 4xx client errors.
fn is_retryable(err: &ureq::Error) -> bool {
    match err {
        ureq::Error::StatusCode(code) => {
            let c = *code;
            c == 429 || c == 500 || c == 502 || c == 503 || c == 504
        }
        // Timeout, DNS, connection reset, etc. all retryable.
        _ => true,
    }
}

/// A body read that failed, typed by whose doing it was.
///
/// A body over the size limit is a fact about the object, which another
/// attempt would only repeat; every other read failure (a reset, a
/// timeout, a connection that closed early) is the network's.
fn body_read_error(url: &str, err: ureq::Error) -> crate::RustmetError {
    let text = format!("failed to read {}: {}", url, err);
    if matches!(err, ureq::Error::BodyExceedsLimit(_)) {
        crate::RustmetError::Http(text)
    } else {
        crate::RustmetError::Transfer(text)
    }
}

fn is_nomads_url(url: &str) -> bool {
    url.contains("nomads.ncep.noaa.gov")
}

fn is_probable_nomads_rate_limit(url: &str, err: &ureq::Error) -> bool {
    is_nomads_url(url) && err.to_string().contains("missing a location header")
}

fn is_redirect_status(status: ureq::http::StatusCode) -> bool {
    status.is_redirection()
}

fn resolve_redirect_url(current_url: &str, location: &str) -> crate::error::Result<String> {
    if location.starts_with("http://") || location.starts_with("https://") {
        return Ok(location.to_string());
    }

    let current_uri: ureq::http::Uri = current_url.parse().map_err(|err| {
        crate::RustmetError::Http(format!(
            "failed to parse redirect source URL {}: {}",
            current_url, err
        ))
    })?;

    let scheme = current_uri.scheme_str().ok_or_else(|| {
        crate::RustmetError::Http(format!(
            "redirect source URL {} is missing a scheme",
            current_url
        ))
    })?;
    let authority = current_uri.authority().ok_or_else(|| {
        crate::RustmetError::Http(format!(
            "redirect source URL {} is missing an authority",
            current_url
        ))
    })?;

    if location.starts_with('/') {
        return Ok(format!("{}://{}{}", scheme, authority, location));
    }

    let path = current_uri.path();
    let directory = path.rsplit_once('/').map(|(dir, _)| dir).unwrap_or("");
    let joined = if directory.is_empty() {
        format!("/{}", location)
    } else {
        format!("{}/{}", directory, location)
    };
    Ok(format!("{}://{}{}", scheme, authority, joined))
}

fn env_duration_ms(name: &str, fallback: Duration) -> Duration {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|millis| *millis > 0)
        .map(Duration::from_millis)
        .unwrap_or(fallback)
}

fn nomads_state_path() -> PathBuf {
    std::env::var("RUSTWX_NOMADS_RATE_STATE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir().join("rustwx_nomads_rate_limit.state"))
}

fn now_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

/// Read the shared pacing state: `(last_request_ms, cooldown_until_ms, sound)`.
///
/// `sound` is false when the file EXISTS but carries no usable
/// `last_request_ms`. That case used to be indistinguishable from "nobody has
/// fetched yet": both mapped to zero timestamps, which puts `last_request +
/// gap` in 1970 and waves the request straight through. A governor that
/// protects a shared public service must not treat corruption as permission to
/// send, so an unusable state is reported as "a request just happened" and the
/// caller waits a full gap. A genuinely absent file is still a real zero.
fn read_nomads_state(path: &Path) -> (u128, u128, bool) {
    let text = match fs::read_to_string(path) {
        Ok(text) => text,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return (0, 0, true),
        Err(_) => return (now_millis(), 0, false),
    };
    let mut last_request_ms: Option<u128> = None;
    let mut cooldown_until_ms = 0;
    for line in text.lines() {
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        let Ok(parsed) = value.trim().parse::<u128>() else {
            continue;
        };
        match key.trim() {
            "last_request_ms" => last_request_ms = Some(parsed),
            "cooldown_until_ms" => cooldown_until_ms = parsed,
            _ => {}
        }
    }
    match last_request_ms {
        Some(value) => (value, cooldown_until_ms, true),
        None => (now_millis(), cooldown_until_ms, false),
    }
}

/// Publish the shared pacing state; false when the bytes did not land.
///
/// The result used to be discarded. A process that held the sentinel, failed
/// to record its request and proceeded left the next process seeing an older
/// `last_request_ms` and free to send immediately -- the failure opened the
/// gate instead of closing it. Callers now absorb the gap locally instead.
fn write_nomads_state(path: &Path, last_request_ms: u128, cooldown_until_ms: u128) -> bool {
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let tmp = path.with_extension("tmp");
    let body = format!(
        "last_request_ms={}\ncooldown_until_ms={}\n",
        last_request_ms, cooldown_until_ms
    );
    fs::write(&tmp, body).is_ok() && fs::rename(tmp, path).is_ok()
}

struct NomadsRateLock {
    path: PathBuf,
}

impl Drop for NomadsRateLock {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

fn nomads_lock_is_stale(lock_path: &Path) -> bool {
    if fs::metadata(lock_path)
        .and_then(|meta| meta.modified())
        .ok()
        .and_then(|modified| modified.elapsed().ok())
        .is_some_and(|elapsed| elapsed > NOMADS_LOCK_STALE_AFTER)
    {
        return true;
    }

    #[cfg(unix)]
    {
        if let Ok(text) = fs::read_to_string(lock_path) {
            if let Some(pid) = text.split_whitespace().next() {
                if pid.parse::<u32>().is_ok() && !Path::new("/proc").join(pid).exists() {
                    return true;
                }
            }
        }
    }

    false
}

fn acquire_nomads_rate_lock(state_path: &Path) -> Option<NomadsRateLock> {
    let lock_path = state_path.with_extension("lock");
    if let Some(parent) = lock_path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    loop {
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&lock_path)
        {
            Ok(mut file) => {
                let _ = writeln!(file, "{} {}", std::process::id(), now_millis());
                return Some(NomadsRateLock { path: lock_path });
            }
            Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => {
                if nomads_lock_is_stale(&lock_path) {
                    let _ = fs::remove_file(&lock_path);
                    continue;
                }
                std::thread::sleep(Duration::from_millis(50));
            }
            Err(_) => return None,
        }
    }
}

fn log_nomads_event(url: &str, kind: &str, status: &str, elapsed_ms: Option<u128>) {
    let Ok(path) = std::env::var("RUSTWX_NOMADS_REQUEST_LOG") else {
        return;
    };
    let escaped_url = url.replace('\\', "\\\\").replace('"', "\\\"");
    let elapsed = elapsed_ms
        .map(|value| value.to_string())
        .unwrap_or_else(|| "null".to_string());
    let line = format!(
        "{{\"ts_ms\":{},\"pid\":{},\"kind\":\"{}\",\"status\":\"{}\",\"elapsed_ms\":{},\"url\":\"{}\"}}\n",
        now_millis(),
        std::process::id(),
        kind,
        status.replace('"', "'"),
        elapsed,
        escaped_url
    );
    if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(path) {
        let _ = file.write_all(line.as_bytes());
    }
}

fn mark_nomads_rate_limited(url: &str, reason: &str) {
    if !is_nomads_url(url) {
        return;
    }
    let cooldown = env_duration_ms("RUSTWX_NOMADS_COOLDOWN_MS", NOMADS_DEFAULT_COOLDOWN);
    let state_path = nomads_state_path();
    let _lock = acquire_nomads_rate_lock(&state_path);
    let (last_request_ms, existing_cooldown_until_ms, _sound) = read_nomads_state(&state_path);
    let now = now_millis();
    if existing_cooldown_until_ms > now {
        log_nomads_event(url, "cooldown_existing", reason, None);
        return;
    }
    let cooldown_until_ms = now.saturating_add(cooldown.as_millis());
    write_nomads_state(&state_path, last_request_ms, cooldown_until_ms);
    log_nomads_event(url, "cooldown", reason, None);
}

fn pace_request(url: &str) {
    if !is_nomads_url(url) {
        return;
    }

    let min_gap = env_duration_ms(
        "RUSTWX_NOMADS_MIN_INTERVAL_MS",
        NOMADS_DEFAULT_MIN_REQUEST_GAP,
    );
    let state_path = nomads_state_path();
    // An unusable state buys exactly one full gap, then this process
    // republishes a sound one. Charging it every round would spin forever
    // against a file nobody can parse: refusing to send is not the same as
    // refusing to finish.
    let mut corruption_paid = false;
    loop {
        let Some(_lock) = acquire_nomads_rate_lock(&state_path) else {
            std::thread::sleep(min_gap);
            continue;
        };

        let (mut last_request_ms, cooldown_until_ms, sound) = read_nomads_state(&state_path);
        if !sound {
            if !corruption_paid {
                drop(_lock);
                std::thread::sleep(min_gap);
                corruption_paid = true;
                continue;
            }
            last_request_ms = 0; // the gap has been served in full
        }
        let now = now_millis();
        let sleep_until =
            cooldown_until_ms.max(last_request_ms.saturating_add(min_gap.as_millis()));
        if sleep_until > now {
            drop(_lock);
            std::thread::sleep(Duration::from_millis(
                (sleep_until - now).min(u64::MAX as u128) as u64,
            ));
            continue;
        }
        if !write_nomads_state(&state_path, now, cooldown_until_ms) {
            // The record did not land, so the next process will not see this
            // request. Absorb the gap here instead of letting it send now.
            drop(_lock);
            std::thread::sleep(min_gap);
        }
        return;
    }
}

/// What this build's cross-process NOMADS governor is configured to do.
///
/// Everything here is read from the same places `pace_request` reads, so a
/// consumer that prints or asserts on it is describing the governor that will
/// actually run -- not a constant that happens to sit beside it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NomadsGovernor {
    /// The node-wide state file this process paces against.
    pub state_path: PathBuf,
    /// Minimum spacing between NOMADS requests across all processes.
    pub min_request_gap: Duration,
    /// How long an over-rate-limit answer pauses the whole node.
    pub cooldown: Duration,
    /// When a held sentinel is treated as abandoned.
    pub lock_stale_after: Duration,
}

/// The NOMADS governor this build carries.
///
/// A **capability probe**, and the reason it exists is worth stating. Two
/// different wx-core crates were vendored under the same `0.3.9` version
/// string: this one, and the published crates.io copy, which has no governor
/// at all -- no pacing, no cooldown, no state file. Only a path dependency
/// kept the right one in the graph, and "we must have got the right one
/// because of where it lives" is an assumption, not a check.
///
/// So consumers ask the resolved crate what it can do. A build wired to a
/// governorless wx-core does not link (this symbol is absent there), and a
/// build wired to this one can be made to *demonstrate* the pacing via
/// [`pace_nomads_request`] rather than infer it.
pub fn nomads_governor() -> NomadsGovernor {
    NomadsGovernor {
        state_path: nomads_state_path(),
        min_request_gap: env_duration_ms(
            "RUSTWX_NOMADS_MIN_INTERVAL_MS",
            NOMADS_DEFAULT_MIN_REQUEST_GAP,
        ),
        cooldown: env_duration_ms("RUSTWX_NOMADS_COOLDOWN_MS", NOMADS_DEFAULT_COOLDOWN),
        lock_stale_after: NOMADS_LOCK_STALE_AFTER,
    }
}

/// Block until this process may send `url`, exactly as the HTTP path does.
///
/// Public so a consumer can PROVE the governor runs -- issue two paced calls
/// for a NOMADS URL and observe the spacing and the state file -- instead of
/// trusting that the crate it linked is the one with the governor in it. A
/// no-op for every host but NOMADS, and it makes no network request.
pub fn pace_nomads_request(url: &str) {
    pace_request(url);
}

/// What a HEAD request learned about an object (see
/// [`DownloadClient::head_status`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HeadOutcome {
    /// The origin answered 200.
    Present,
    /// The origin answered 404 or 403, or refused the request outright.
    Absent,
    /// No answer: the request failed on the network, or with a 429 or
    /// 5xx, on both attempts (for NOMADS also a redirect with no usable
    /// Location, its over-rate-limit answer).  Carries the last error.
    Unreachable(String),
}

/// Build a ureq agent with TLS configured via rustls-rustcrypto.
///
/// `pooled` false gives an agent that keeps no idle connection, so each
/// request it sends opens a new one.
fn build_agent(config: &DownloadConfig, pooled: bool) -> ureq::Agent {
    // Install the rustcrypto provider as the process-wide default.
    rustls::crypto::CryptoProvider::install_default(rustls_rustcrypto::provider()).ok();

    let crypto = Arc::new(rustls_rustcrypto::provider());

    let mut builder = ureq::Agent::config_builder();
    if !pooled || std::env::var_os("GPUWM_FETCH_NO_IDLE_CONNECTIONS").is_some() {
        builder = builder
            .max_idle_connections(0)
            .max_idle_connections_per_host(0);
    }
    let agent_config = builder
        .tls_config(
            ureq::tls::TlsConfig::builder()
                .provider(ureq::tls::TlsProvider::Rustls)
                .root_certs(ureq::tls::RootCerts::WebPki)
                .unversioned_rustls_crypto_provider(crypto)
                .build(),
        )
        .max_redirects(0)
        // No whole-request limit: see DEFAULT_STALL_WINDOW.  Opening a
        // connection and sending a request are bounded; everything after
        // that is judged by whether bytes arrive.
        .timeout_global(None)
        .timeout_resolve(Some(config.connect_timeout))
        .timeout_connect(Some(config.connect_timeout))
        .timeout_send_request(Some(config.connect_timeout))
        .build();
    let connector = DefaultConnector::new().chain(StallConnector(config.stall));
    ureq::Agent::with_parts(agent_config, connector, DefaultResolver::default())
}

impl DownloadClient {
    /// One GET, on a pooled connection or, with `fresh`, a new one.
    fn perform_get(
        &self,
        url: &str,
        range_header: Option<&str>,
        fresh: bool,
    ) -> Result<ureq::http::Response<ureq::Body>, ureq::Error> {
        let agent = if fresh { &self.fresh } else { &self.agent };
        // Byte ranges and size/digest pins describe the identity representation.
        let mut request = agent.get(url).header("Accept-Encoding", "identity");
        if let Some(range_header) = range_header {
            request = request.header("Range", range_header);
        }
        let started = now_millis();
        let result = request.call();
        if is_nomads_url(url) {
            let elapsed = now_millis().saturating_sub(started);
            match &result {
                Ok(response) => log_nomads_event(
                    url,
                    if range_header.is_some() {
                        "get_range"
                    } else {
                        "get"
                    },
                    response.status().as_str(),
                    Some(elapsed),
                ),
                Err(err) => log_nomads_event(
                    url,
                    if range_header.is_some() {
                        "get_range"
                    } else {
                        "get"
                    },
                    &format!("error:{err}"),
                    Some(elapsed),
                ),
            }
        }
        result
    }

    fn get_response_following_redirects(
        &self,
        url: &str,
        range_header: Option<&str>,
    ) -> crate::error::Result<ureq::http::Response<ureq::Body>> {
        let mut current_url = url.to_string();
        let mut malformed_redirect_retries = 0u32;

        for redirect_count in 0..=MAX_REDIRECTS {
            let request_url = current_url.clone();
            let response = self.with_retry(&request_url, |fresh| {
                self.perform_get(&request_url, range_header, fresh)
            })?;
            let status = response.status();

            if is_redirect_status(status) {
                if redirect_count == MAX_REDIRECTS {
                    return Err(crate::RustmetError::Http(format!(
                        "too many redirects while requesting {}",
                        url
                    )));
                }

                let location = response
                    .headers()
                    .get(LOCATION)
                    .and_then(|value| value.to_str().ok());

                let Some(location) = location else {
                    if is_nomads_url(&request_url) && malformed_redirect_retries < self.max_retries
                    {
                        malformed_redirect_retries += 1;
                        mark_nomads_rate_limited(&request_url, "redirect_missing_location");
                        eprintln!(
                            "  NOMADS cooldown {}/{} for {} (probable over-rate-limit redirect {})",
                            malformed_redirect_retries, self.max_retries, request_url, status
                        );
                        continue;
                    }

                    return Err(crate::RustmetError::Http(format!(
                        "redirect response missing Location header for {} (status {})",
                        request_url, status
                    )));
                };

                current_url = resolve_redirect_url(&request_url, location)?;
                continue;
            }

            return Ok(response);
        }

        Err(crate::RustmetError::Http(format!(
            "too many redirects while requesting {}",
            url
        )))
    }

    /// Whether NOMADS serves `url`, asked with a one-byte range GET.
    ///
    /// Present on any answer that is not a redirect, absent on 404 or 403
    /// or a request the origin refused outright, and unreachable when it
    /// could not tell: a network error, a 429 or 5xx, or a redirect with
    /// no usable Location (the over-rate-limit answer NOMADS gives through
    /// Akamai), on both attempts.  This used to be a bool, so every one of
    /// those read as absent: a fetch whose probe met a dropped link or a
    /// throttled NOMADS refused the hour as "no source served this
    /// object", which a caller never asks again.
    fn probe_nomads_range(&self, url: &str) -> HeadOutcome {
        let mut why = String::new();
        for attempt in 0..=1u32 {
            let mut current_url = url.to_string();
            let mut redirects = 0u32;
            why = loop {
                pace_request(&current_url);
                match self.perform_get(&current_url, Some("bytes=0-0"), attempt > 0) {
                    Ok(response) => {
                        let status = response.status();
                        if !is_redirect_status(status) {
                            return HeadOutcome::Present;
                        }
                        let Some(location) = response
                            .headers()
                            .get(LOCATION)
                            .and_then(|value| value.to_str().ok())
                        else {
                            if is_nomads_url(&current_url) {
                                mark_nomads_rate_limited(
                                    &current_url,
                                    "range_probe_redirect_missing_location",
                                );
                            }
                            break format!(
                                "{status} with no Location header (the NOMADS over-rate-limit answer)"
                            );
                        };
                        if redirects == MAX_REDIRECTS {
                            break format!("more than {MAX_REDIRECTS} redirects");
                        }
                        match resolve_redirect_url(&current_url, location) {
                            Ok(next_url) => {
                                redirects += 1;
                                current_url = next_url;
                            }
                            Err(err) => break format!("{status} to an unusable Location: {err}"),
                        }
                    }
                    Err(ureq::Error::StatusCode(code)) if code == 404 || code == 403 => {
                        return HeadOutcome::Absent;
                    }
                    Err(err) => {
                        if is_probable_nomads_rate_limit(&current_url, &err) {
                            mark_nomads_rate_limited(&current_url, "range_probe_rate_limit_error");
                        }
                        if !is_retryable(&err) {
                            return HeadOutcome::Absent;
                        }
                        break err.to_string();
                    }
                }
            };
            if attempt == 0 {
                std::thread::sleep(Duration::from_millis(500));
            }
        }
        HeadOutcome::Unreachable(format!("GET {url} (bytes=0-0) failed twice: {why}"))
    }

    /// Create a new download client with TLS configured via rustls-rustcrypto.
    ///
    /// Uses ureq's built-in TlsConfig with the rustcrypto provider and
    /// webpki root certificates (Mozilla's CA bundle). No caching.
    pub fn new() -> crate::error::Result<Self> {
        Self::new_with_config(DownloadConfig::default())
    }

    /// Create a new download client with custom timeout and retry settings.
    /// No caching.
    pub fn new_with_config(config: DownloadConfig) -> crate::error::Result<Self> {
        Ok(Self {
            agent: build_agent(&config, true),
            fresh: build_agent(&config, false),
            max_retries: config.max_retries,
            streams: config.streams.max(1),
            cache: None,
            progress: None,
        })
    }

    /// Create a new download client with disk caching enabled.
    ///
    /// If `cache_dir` is `Some`, files are cached there. If `None`, the
    /// platform default is used (`~/.cache/metrust/` on Linux/macOS,
    /// `%LOCALAPPDATA%/metrust/cache/` on Windows).
    pub fn new_with_cache(cache_dir: Option<&str>) -> crate::error::Result<Self> {
        let config = DownloadConfig::default();
        let cache = match cache_dir {
            Some(dir) => DiskCache::with_dir(std::path::PathBuf::from(dir)),
            None => DiskCache::new(),
        };
        Ok(Self {
            agent: build_agent(&config, true),
            fresh: build_agent(&config, false),
            max_retries: config.max_retries,
            streams: config.streams.max(1),
            cache: Some(cache),
            progress: None,
        })
    }

    /// Attach a `DiskCache` to this client. Replaces any existing cache.
    pub fn set_cache(&mut self, cache: DiskCache) {
        self.cache = Some(cache);
    }

    /// Count every body byte this client receives into `progress`.
    pub fn set_progress(&mut self, progress: Arc<TransferProgress>) {
        self.progress = Some(progress);
    }

    /// Chunk streams one ranged transfer keeps open at once.
    pub fn streams(&self) -> usize {
        self.streams
    }

    /// Range width after the origin's existing politeness cap.
    pub fn streams_for(&self, url: &str) -> usize {
        if is_nomads_url(url) { 1 } else { self.streams }
    }

    /// Return a reference to the underlying HTTP agent.
    ///
    /// Used by the streaming download module to make requests with
    /// manual body reading.
    pub fn agent(&self) -> &ureq::Agent {
        &self.agent
    }

    /// Return a reference to the cache, if one is attached.
    pub fn cache(&self) -> Option<&DiskCache> {
        self.cache.as_ref()
    }

    /// Execute a request-producing closure with retry and exponential backoff.
    ///
    /// `attempt_fn` is called on each attempt and must produce the final result
    /// or a ureq::Error. This avoids needing to name the ureq Response type.
    ///
    /// Its argument says whether the attempt must open a new connection:
    /// the first attempt may reuse a pooled one, every retry does not.  A
    /// pooled connection can be dead by the time it is reused (the origin
    /// or a proxy between closed it after the last answer), and a retry
    /// that took the next pooled connection could meet another: a 36 h
    /// HRRR fetch through a buffering proxy on a slow link lost a 420 MB
    /// object at 419 MB when four attempts in a row went out on connections
    /// the proxy had already closed ("io: Peer disconnected"), and moved
    /// the whole object again.
    fn with_retry<T, F>(&self, url: &str, attempt_fn: F) -> crate::error::Result<T>
    where
        F: Fn(bool) -> Result<T, ureq::Error>,
    {
        let mut last_err = String::new();
        let mut last_retryable = false;

        for attempt in 0..=self.max_retries {
            pace_request(url);
            match attempt_fn(attempt > 0) {
                Ok(val) => return Ok(val),
                Err(e) => {
                    let probable_nomads_rate_limit = is_probable_nomads_rate_limit(url, &e);
                    if probable_nomads_rate_limit {
                        mark_nomads_rate_limited(url, "retry_rate_limit_error");
                    }
                    last_err = if probable_nomads_rate_limit {
                        format!("probable NOMADS rate-limit response for {}: {}", url, e)
                    } else {
                        format!("{}", e)
                    };

                    last_retryable = is_retryable(&e);
                    if attempt < self.max_retries && last_retryable {
                        let backoff = if probable_nomads_rate_limit {
                            NOMADS_RATE_LIMIT_BACKOFF_DURATIONS
                                .get(attempt as usize)
                                .copied()
                                .unwrap_or(
                                    NOMADS_RATE_LIMIT_BACKOFF_DURATIONS
                                        [NOMADS_RATE_LIMIT_BACKOFF_DURATIONS.len() - 1],
                                )
                        } else {
                            BACKOFF_DURATIONS
                                .get(attempt as usize)
                                .copied()
                                .unwrap_or(BACKOFF_DURATIONS[BACKOFF_DURATIONS.len() - 1])
                        };
                        eprintln!(
                            "  Retry {}/{} for {} after {:?} ({})",
                            attempt + 1,
                            self.max_retries,
                            url,
                            backoff,
                            e
                        );
                        std::thread::sleep(backoff);
                    } else {
                        break;
                    }
                }
            }
        }

        // A connection that failed or a 429/5xx that outlived every retry
        // is the network's doing; a 4xx is the origin's answer.
        let text = format!("HTTP request failed for {}: {}", url, last_err);
        Err(if last_retryable {
            crate::RustmetError::Transfer(text)
        } else {
            crate::RustmetError::Http(text)
        })
    }

    /// Send a HEAD request and return true if the server responds with 200 OK.
    ///
    /// Does NOT retry on 404, only retries on transient/server errors.
    /// Useful for probing whether a remote file exists (e.g., .idx files).
    /// A server that could not be reached reads as absent here; see
    /// [`DownloadClient::head_status`] for a caller that must tell the two
    /// apart.
    pub fn head_ok(&self, url: &str) -> bool {
        self.head_status(url) == HeadOutcome::Present
    }

    /// What a HEAD request learned about `url`: there, not there, or no
    /// answer at all.
    ///
    /// The third case is why this exists.  `head_ok` folds a connection
    /// that failed twice into "absent", so a probe that met a dropped
    /// link reported the object as not published, and a fetch refused it
    /// ("no source served this object") without another attempt or
    /// another host -- a network fault presented as a fact about the
    /// origin.  Same requests as `head_ok`: one retry on a transient
    /// error, none on 404 or 403.
    pub fn head_status(&self, url: &str) -> HeadOutcome {
        if is_nomads_url(url) {
            return self.probe_nomads_range(url);
        }

        for attempt in 0..=1u32 {
            match self.agent.head(url).call() {
                Ok(_) => return HeadOutcome::Present,
                Err(ureq::Error::StatusCode(code)) if code == 404 || code == 403 => {
                    return HeadOutcome::Absent;
                }
                Err(e) => {
                    if !is_retryable(&e) {
                        return HeadOutcome::Absent;
                    }
                    if attempt == 0 {
                        std::thread::sleep(std::time::Duration::from_millis(300));
                        continue;
                    }
                    return HeadOutcome::Unreachable(format!(
                        "HEAD {} failed twice: {}",
                        url, e
                    ));
                }
            }
        }
        HeadOutcome::Absent
    }

    /// Download a full URL and return the response body as bytes.
    ///
    /// If caching is enabled, checks cache first and stores the result after
    /// a successful download. Cache failures are silently ignored.
    pub fn get_bytes(&self, url: &str) -> crate::error::Result<Vec<u8>> {
        let key = DiskCache::cache_key(url, None);

        // Try cache first
        if let Some(cache) = &self.cache {
            if let Some(data) = cache.get(&key) {
                return Ok(data);
            }
        }

        let mut response = self.get_response_following_redirects(url, None)?;
        let mut data = Vec::new();
        Counted {
            inner: response.body_mut().with_config().limit(MAX_BODY_SIZE).reader(),
            progress: self.progress.as_deref(),
        }
        .read_to_end(&mut data)
        .map_err(|err| body_read_error(url, ureq::Error::from(err)))?;

        // Store in cache (errors silently ignored)
        if let Some(cache) = &self.cache {
            cache.put(&key, &data);
        }

        Ok(data)
    }

    /// Download a full URL via byte ranges and return the concatenated bytes.
    ///
    /// This does not require an external `.idx` file. It first probes range
    /// support with `Range: bytes=0-0`; if the origin does not respond with a
    /// usable `Content-Range`, it falls back to the normal full-body download.
    pub fn get_bytes_parallel_whole(&self, url: &str) -> crate::error::Result<Vec<u8>> {
        let key = DiskCache::cache_key(url, None);

        if let Some(cache) = &self.cache {
            if let Some(data) = cache.get(&key) {
                return Ok(data);
            }
        }

        let total_len = match self.probe_range_total_length(url, false) {
            Ok(Some(total_len)) if total_len > 0 => total_len,
            _ => return self.get_bytes(url),
        };
        if let Some(progress) = &self.progress {
            progress.set_total(total_len);
        }
        // Even a single chunk goes by range: a range is what a broken body
        // can be resumed from.
        let ranges = full_file_ranges(total_len, FULL_FILE_RANGE_CHUNK_BYTES);
        if ranges.is_empty() {
            return self.get_bytes(url);
        }

        let data = self.get_ranges(url, &ranges)?;
        if data.len() as u64 != total_len {
            return Err(crate::RustmetError::Http(format!(
                "parallel whole-file download for {} returned {} bytes, expected {}",
                url,
                data.len(),
                total_len
            )));
        }

        // `get_ranges` has already stored these bytes under the URL+ranges
        // key. Storing them again under the URL-only key is what a later
        // whole-file request looks itself up by, and it costs a reference
        // rather than a second payload: the cache recognises the content.
        if let Some(cache) = &self.cache {
            cache.put(&key, &data);
        }

        Ok(data)
    }

    /// The object's total length as the origin states it in the
    /// `Content-Range` of a one-byte answer or the `Content-Length` of a
    /// complete HTTP 200 answer. Unknown totals return `None`.
    pub fn object_length(&self, url: &str) -> crate::error::Result<Option<u64>> {
        self.probe_range_total_length(url, true)
    }

    fn probe_range_total_length(&self, url: &str, allow_full: bool) -> crate::error::Result<Option<u64>> {
        let response = self.get_response_following_redirects(url, Some("bytes=0-0"))?;
        if allow_full && response.status().as_u16() == 200 {
            // A server may ignore the probe's Range and answer with the whole
            // object. Its Content-Length still proves the full object size.
            return Ok(response.headers().get(CONTENT_LENGTH)
                .and_then(|value| value.to_str().ok())
                .and_then(|value| value.parse().ok()));
        }
        if response.status().as_u16() != 206 {
            return Ok(None);
        }
        Ok(response
            .headers()
            .get(CONTENT_RANGE)
            .and_then(|value| value.to_str().ok())
            .and_then(parse_content_range_total))
    }

    /// Download a URL and return the response body as a string (for .idx files).
    ///
    /// Text responses (like .idx) are NOT cached because they are small and
    /// may change between model runs.
    pub fn get_text(&self, url: &str) -> crate::error::Result<String> {
        let mut response = self.get_response_following_redirects(url, None)?;
        let text = response
            .body_mut()
            .read_to_string()
            .map_err(|err| body_read_error(url, err))?;
        Ok(text)
    }

    /// Download a specific byte range from a URL.
    ///
    /// If caching is enabled, the result is keyed by URL + byte range.
    /// Cache failures are silently ignored.
    pub fn get_range(&self, url: &str, start: u64, end: u64) -> crate::error::Result<Vec<u8>> {
        let key = DiskCache::cache_key(url, Some((start, end)));

        // Try cache first
        if let Some(cache) = &self.cache {
            if let Some(data) = cache.get(&key) {
                return Ok(data);
            }
        }

        // A body that breaks off or stalls is RESUMED here, per chunk: the
        // bytes that arrived are kept and the next request asks for the
        // rest of the span only.  `with_retry` covers getting a response;
        // the body of a 16 MiB chunk is read after it returns, so a reset
        // or a stall part-way through used to cost the whole chunk, and
        // before that the whole object.  The response-level refusals below
        // (a 200, a foreign span, more bytes than asked for) are the
        // origin's answer rather than the network's and are not retried.
        //
        // Budget: an attempt that delivered bytes is free, because those
        // bytes prove the link works and nothing is asked for twice.  Only
        // attempts that deliver nothing spend `max_retries`, and
        // CHUNK_ATTEMPT_CEILING bounds the whole loop.
        let mut data: Vec<u8> = Vec::new();
        // Inclusive last byte of the span.  An open-ended request learns
        // it from its first answer, so a resume can name it.
        let mut last = (end != u64::MAX).then_some(end);
        let mut attempts = 0u32;
        let mut idle_attempts = 0u32;
        loop {
            let from = start + data.len() as u64;
            let range_header = match last {
                Some(last) => format!("bytes={}-{}", from, last),
                None => format!("bytes={}-", from),
            };
            let mut response = self.get_response_following_redirects(url, Some(&range_header))?;
            // Validate the ANSWER before adopting it, not after concatenating
            // it.  A range GET whose reply is a 200 (the origin ignored the
            // header and sent the whole object), or a 206 for some other span,
            // or a body of the wrong length, used to be accepted, cached, and
            // concatenated into the caller's subset -- the outer GRIB bars
            // then rejected the assembled file with no indication which chunk
            // was wrong, and the bad bytes stayed in the cache.  This is the
            // same three-clause check the Python range transport has always
            // applied, made against the span THIS request asked for.
            let status = response.status().as_u16();
            // HTTP 200 is safe only when this request is exactly the whole
            // object. A subset or a resumed request must still answer 206.
            let whole_answer = status == 200 && from == 0 && last.is_some_and(|end|
                response.headers().get(CONTENT_LENGTH)
                    .and_then(|value| value.to_str().ok())
                    .and_then(|value| value.parse::<u64>().ok()) == end.checked_add(1));
            if status != 206 && !whole_answer {
                return Err(crate::RustmetError::Http(format!(
                    "range request for {} ({}) returned HTTP {}, not 206; the \
                     origin did not serve the requested span",
                    url, range_header, status
                )));
            }
            let content_range = response
                .headers()
                .get(CONTENT_RANGE)
                .and_then(|value| value.to_str().ok())
                .unwrap_or_default()
                .to_string();
            let expected_prefix = match last {
                Some(last) => format!("bytes {}-{}/", from, last),
                None => format!("bytes {}-", from),
            };
            if !whole_answer && !content_range.starts_with(&expected_prefix) {
                return Err(crate::RustmetError::Http(format!(
                    "range request for {} ({}) answered with Content-Range {:?}, \
                     which is not the requested span",
                    url, range_header, content_range
                )));
            }
            if last.is_none() {
                last = parse_content_range_last(&content_range);
            }
            let before = data.len();
            let read = Counted {
                inner: response.body_mut().with_config().limit(MAX_BODY_SIZE).reader(),
                progress: self.progress.as_deref(),
            }
            .read_to_end(&mut data);
            attempts += 1;
            let wanted = last.map(|last| last - start + 1);
            let reason = match (read, wanted) {
                (Ok(_), Some(wanted)) if data.len() as u64 == wanted => break,
                (Ok(_), None) => break,
                (Ok(_), Some(wanted)) if data.len() as u64 > wanted => {
                    return Err(crate::RustmetError::Http(format!(
                        "range request for {} ({}) returned {} bytes past the {} requested",
                        url,
                        range_header,
                        data.len() as u64 - wanted,
                        wanted
                    )));
                }
                (Ok(_), Some(wanted)) => format!(
                    "range request for {} ({}) returned {} bytes, expected {}",
                    url,
                    range_header,
                    data.len() - before,
                    wanted - (before as u64)
                ),
                (Err(err), _) => format!("failed to read {} ({}): {}", url, range_header, err),
            };
            if data.len() > before {
                idle_attempts = 0;
            } else {
                idle_attempts += 1;
            }
            let span = wanted
                .map(|wanted| wanted.to_string())
                .unwrap_or_else(|| "?".to_string());
            if idle_attempts > self.max_retries || attempts >= CHUNK_ATTEMPT_CEILING {
                // A body that broke off, stalled or came up short and
                // would not resume: the network's doing, unlike the 200
                // and foreign-span refusals above.
                return Err(crate::RustmetError::Transfer(format!(
                    "{} (gave up after {} attempts; {} of {} bytes of the span arrived)",
                    reason,
                    attempts,
                    data.len(),
                    span
                )));
            }
            let backoff = match idle_attempts {
                0 => Duration::ZERO,
                n => BACKOFF_DURATIONS
                    .get(n as usize - 1)
                    .copied()
                    .unwrap_or(BACKOFF_DURATIONS[BACKOFF_DURATIONS.len() - 1]),
            };
            // Own line: the chunk counter draws with bare `\r`.
            eprintln!(
                "\n  Resuming at byte {} of {} ({} of {} bytes of the span kept) after {:?}: {}",
                start + data.len() as u64,
                url,
                data.len(),
                span,
                backoff,
                reason
            );
            std::thread::sleep(backoff);
        }

        // Store in cache (errors silently ignored)
        if let Some(cache) = &self.cache {
            cache.put(&key, &data);
        }

        Ok(data)
    }

    /// Download multiple byte ranges from a URL in parallel and concatenate the results.
    ///
    /// Each range is downloaded as a separate HTTP request with a Range
    /// header, over at most [`DownloadClient::streams`] connections at once
    /// (one for NOMADS, which is served serially on purpose).  The pool
    /// used to be rayon's, one stream per CPU thread, which put 24 streams
    /// per object on a 24-thread machine however many objects the caller
    /// was already fetching.  Progress is printed to stderr.
    ///
    /// The first chunk that fails stops the pool from starting more; the
    /// chunks already in flight finish, and the earliest failure in range
    /// order is returned.
    ///
    /// If caching is enabled, the combined result is cached under a key derived
    /// from the URL and all ranges. Individual ranges are also cached by
    /// `get_range`, so partial overlaps with future requests benefit from the
    /// cache too.
    ///
    /// The combined store is not a second copy of the object. A whole-file
    /// caller stores the same bytes again under its URL-only key, and those
    /// two entries used to be two complete payloads at two paths -- the store
    /// is content-addressed, so the second entry is a reference to the first.
    /// The two keys are still distinct keys: a range list that does not cover
    /// the object yields different bytes and gets its own payload, which is
    /// why the cache decides this on the content and not on the key shape.
    pub fn get_ranges(&self, url: &str, ranges: &[(u64, u64)]) -> crate::error::Result<Vec<u8>> {
        let total = ranges.len();
        if total == 0 {
            return Ok(Vec::new());
        }

        // Check for the combined result in cache
        let combined_key = DiskCache::cache_key_ranges(url, ranges);
        if let Some(cache) = &self.cache {
            if let Some(data) = cache.get(&combined_key) {
                return Ok(data);
            }
        }

        if let Some(progress) = &self.progress {
            if progress.total().is_none() && ranges.iter().all(|&(_, end)| end != u64::MAX) {
                progress.set_total(ranges.iter().map(|&(start, end)| end - start + 1).sum());
            }
        }

        let streams = self.streams_for(url).clamp(1, total);
        let completed = AtomicUsize::new(0);
        let next = AtomicUsize::new(0);
        let failed = AtomicBool::new(false);
        let slots: Vec<Mutex<Option<crate::error::Result<Vec<u8>>>>> =
            (0..total).map(|_| Mutex::new(None)).collect();
        // Each chunk is individually cached via get_range.
        let work = || loop {
            if failed.load(Ordering::Relaxed) {
                break;
            }
            let index = next.fetch_add(1, Ordering::Relaxed);
            if index >= total {
                break;
            }
            let (start, end) = ranges[index];
            let result = self.get_range(url, start, end);
            if result.is_err() {
                failed.store(true, Ordering::Relaxed);
            } else {
                let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
                eprint!("\r  Downloading chunks {}/{}...", done, total);
            }
            *slots[index].lock().unwrap_or_else(|poison| poison.into_inner()) = Some(result);
        };
        if streams == 1 {
            work();
        } else {
            std::thread::scope(|scope| {
                for _ in 0..streams {
                    scope.spawn(work);
                }
            });
        }

        // Concatenate in order, propagating the earliest failure.
        let mut combined = Vec::new();
        let mut first_failure = None;
        for (index, slot) in slots.into_iter().enumerate() {
            match slot.into_inner().unwrap_or_else(|poison| poison.into_inner()) {
                Some(Ok(data)) if first_failure.is_none() => combined.extend_from_slice(&data),
                Some(Ok(_)) => {}
                Some(Err(error)) => {
                    first_failure.get_or_insert(error);
                }
                None if first_failure.is_none() && !failed.load(Ordering::Relaxed) => {
                    first_failure = Some(crate::RustmetError::Http(format!(
                        "internal error: chunk {} of {} was never fetched",
                        index + 1,
                        total
                    )));
                }
                None => {}
            }
        }
        if let Some(error) = first_failure {
            return Err(error);
        }

        eprintln!(
            "\r  Downloaded {} chunks, {} bytes total.    ",
            total,
            combined.len()
        );

        // Cache the combined result (errors silently ignored)
        if let Some(cache) = &self.cache {
            cache.put(&combined_key, &combined);
        }

        Ok(combined)
    }
}

/// The inclusive last byte of `bytes FIRST-LAST/TOTAL`.
fn parse_content_range_last(value: &str) -> Option<u64> {
    let span = value.strip_prefix("bytes ")?.split('/').next()?;
    span.split_once('-')?.1.trim().parse().ok()
}

fn parse_content_range_total(value: &str) -> Option<u64> {
    let (_, total) = value.rsplit_once('/')?;
    if total == "*" {
        return None;
    }
    total.parse().ok()
}

fn full_file_ranges(total_len: u64, chunk_size: u64) -> Vec<(u64, u64)> {
    if total_len == 0 || chunk_size == 0 {
        return Vec::new();
    }

    let mut ranges = Vec::new();
    let mut start = 0u64;
    while start < total_len {
        let end = start.saturating_add(chunk_size - 1).min(total_len - 1);
        ranges.push((start, end));
        start = end.saturating_add(1);
    }
    ranges
}

#[cfg(test)]
mod tests {
    use super::{
        full_file_ranges, now_millis, parse_content_range_last, parse_content_range_total,
        read_nomads_state, write_nomads_state, DownloadClient, DownloadConfig, HeadOutcome,
        StallLimit, TransferProgress,
    };
    use std::fs;
    use std::io::{Read, Write};
    use std::net::{TcpListener, TcpStream};
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    use std::thread;
    use std::time::{Duration, Instant};

    fn spawn_http_server(responses: Vec<Vec<u8>>) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind test server");
        let addr = listener.local_addr().expect("server addr");
        thread::spawn(move || {
            for response in responses {
                let (mut stream, _) = listener.accept().expect("accept connection");
                let mut buf = [0u8; 4096];
                let _ = stream.read(&mut buf);
                stream.write_all(&response).expect("write response");
                stream.flush().expect("flush response");
            }
        });
        format!("http://{}", addr)
    }

    fn test_client() -> DownloadClient {
        DownloadClient::new_with_config(DownloadConfig {
            connect_timeout: Duration::from_secs(5),
            stall: StallLimit {
                window: Duration::from_secs(5),
                min_bytes: 1,
            },
            max_retries: 1,
            streams: 4,
        })
        .expect("client")
    }

    /// A client whose stall window is `window` and which needs
    /// `min_bytes` per window to call a connection moving.
    fn stall_client(window: Duration, min_bytes: u64, streams: usize) -> DownloadClient {
        DownloadClient::new_with_config(DownloadConfig {
            connect_timeout: Duration::from_secs(5),
            stall: StallLimit { window, min_bytes },
            max_retries: 1,
            streams,
        })
        .expect("client")
    }

    /// How one connection to a [`RangeOrigin`] behaves.
    #[derive(Clone, Copy)]
    enum Serve {
        /// The requested span, at once.
        Whole,
        /// `n` bytes of the span, then an open connection that sends
        /// nothing more for a minute.
        StallAfter(usize),
        /// The span one byte at a time, this far apart.
        Trickle(Duration),
    }

    /// A local origin that answers `Range: bytes=A-B` (or `A-`) from
    /// `payload`, one request per connection, connection N behaving as
    /// `plan[N]` (Whole past the end of the plan).  It records every
    /// Range header asked for and the most connections open at once.
    struct RangeOrigin {
        base: String,
        asked: Arc<Mutex<Vec<String>>>,
        peak: Arc<AtomicUsize>,
    }

    fn range_origin(payload: Vec<u8>, plan: Vec<Serve>, hold: Duration) -> RangeOrigin {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind test origin");
        let base = format!("http://{}", listener.local_addr().expect("origin addr"));
        let asked = Arc::new(Mutex::new(Vec::new()));
        let peak = Arc::new(AtomicUsize::new(0));
        let open = Arc::new(AtomicUsize::new(0));
        let payload = Arc::new(payload);
        let (asked_in, peak_in) = (asked.clone(), peak.clone());
        thread::spawn(move || {
            for (number, stream) in listener.incoming().enumerate() {
                let Ok(stream) = stream else { continue };
                let serve = plan.get(number).copied().unwrap_or(Serve::Whole);
                let (payload, asked, peak, open) =
                    (payload.clone(), asked_in.clone(), peak_in.clone(), open.clone());
                thread::spawn(move || {
                    let now_open = open.fetch_add(1, Ordering::SeqCst) + 1;
                    peak.fetch_max(now_open, Ordering::SeqCst);
                    serve_range(stream, &payload, serve, hold, &asked);
                    open.fetch_sub(1, Ordering::SeqCst);
                });
            }
        });
        RangeOrigin { base, asked, peak }
    }

    fn serve_range(
        mut stream: TcpStream,
        payload: &[u8],
        serve: Serve,
        hold: Duration,
        asked: &Mutex<Vec<String>>,
    ) {
        let mut head = Vec::new();
        let mut byte = [0u8; 1];
        while !head.ends_with(b"\r\n\r\n") {
            match stream.read(&mut byte) {
                Ok(1) => head.push(byte[0]),
                _ => return,
            }
        }
        let head = String::from_utf8_lossy(&head).to_string();
        let range = head
            .lines()
            .find_map(|line| {
                let (name, value) = line.split_once(':')?;
                name.eq_ignore_ascii_case("range")
                    .then(|| value.trim().to_string())
            })
            .unwrap_or_default();
        asked.lock().unwrap().push(range.clone());
        let span = range.strip_prefix("bytes=").unwrap_or("0-");
        let (first, last) = span.split_once('-').unwrap_or(("0", ""));
        let first: usize = first.parse().unwrap_or(0);
        let last: usize = last.parse().unwrap_or(payload.len() - 1).min(payload.len() - 1);
        let body = &payload[first..=last];
        thread::sleep(hold);
        let reply = format!(
            "HTTP/1.1 206 Partial Content\r\nContent-Range: bytes {}-{}/{}\r\n\
             Content-Length: {}\r\nConnection: close\r\n\r\n",
            first,
            last,
            payload.len(),
            body.len()
        );
        if stream.write_all(reply.as_bytes()).is_err() {
            return;
        }
        match serve {
            Serve::Whole => {
                let _ = stream.write_all(body);
            }
            Serve::StallAfter(count) => {
                let _ = stream.write_all(&body[..count.min(body.len())]);
                let _ = stream.flush();
                thread::sleep(Duration::from_secs(60));
            }
            Serve::Trickle(gap) => {
                for single in body.chunks(1) {
                    if stream.write_all(single).and_then(|_| stream.flush()).is_err() {
                        return;
                    }
                    thread::sleep(gap);
                }
            }
        }
        let _ = stream.flush();
    }

    fn numbered(len: usize) -> Vec<u8> {
        (0..len).map(|index| (index % 251) as u8).collect()
    }

    /// A body that stops moving with its connection still open is
    /// dropped after one stall window and resumed from the byte it
    /// stopped at, instead of being waited on for a whole-request
    /// timeout and then asked for again from its first byte.
    #[test]
    fn a_stalled_body_is_resumed_from_the_byte_it_stopped_at() {
        let payload = numbered(4000);
        let origin = range_origin(payload.clone(), vec![Serve::StallAfter(1500)], Duration::ZERO);
        let started = Instant::now();
        let data = stall_client(Duration::from_secs(2), 1, 4)
            .get_range(&format!("{}/x", origin.base), 0, 3999)
            .expect("the resumed chunk completes");
        assert_eq!(data, payload);
        assert_eq!(
            *origin.asked.lock().unwrap(),
            vec!["bytes=0-3999".to_string(), "bytes=1500-3999".to_string()],
            "the resume asks for the rest of the span only"
        );
        // The 1500 bytes arrive at once and then nothing: the stall is
        // called one window after the last byte.  A window counted from
        // the request instead of from the last progress would take up to
        // two (4 s here); a whole-request timeout, far longer.
        let elapsed = started.elapsed();
        assert!(elapsed >= Duration::from_millis(1900), "{elapsed:?}");
        assert!(elapsed < Duration::from_millis(3500), "{elapsed:?}");
    }

    /// A body that keeps arriving, however slowly, is never cut off:
    /// there is no whole-request limit, only the stall window.
    #[test]
    fn a_slow_body_that_keeps_moving_outlasts_the_stall_window() {
        let payload = numbered(24);
        let origin = range_origin(
            payload.clone(),
            vec![Serve::Trickle(Duration::from_millis(125))],
            Duration::ZERO,
        );
        let started = Instant::now();
        let data = stall_client(Duration::from_secs(1), 1, 4)
            .get_range(&format!("{}/x", origin.base), 0, 23)
            .expect("a slow body completes");
        assert_eq!(data, payload);
        assert!(
            started.elapsed() > Duration::from_secs(2),
            "the body took several stall windows to arrive"
        );
        assert_eq!(origin.asked.lock().unwrap().len(), 1, "and was never re-asked");
    }

    /// The limit is bytes per window, not "any byte at all": a trickle
    /// under it counts as stalled and the chunk resumes elsewhere.
    #[test]
    fn a_trickle_under_the_byte_rate_counts_as_stalled() {
        let payload = numbered(60);
        let origin = range_origin(
            payload.clone(),
            vec![Serve::Trickle(Duration::from_millis(200))],
            Duration::ZERO,
        );
        let data = stall_client(Duration::from_secs(1), 32, 4)
            .get_range(&format!("{}/x", origin.base), 0, 59)
            .expect("the resume on a healthy connection completes");
        assert_eq!(data, payload);
        let asked = origin.asked.lock().unwrap().clone();
        assert_eq!(asked.len(), 2, "{asked:?}");
        assert_eq!(asked[0], "bytes=0-59");
        assert!(asked[1].starts_with("bytes=") && asked[1] != "bytes=0-59", "{asked:?}");
    }

    /// Attempts that deliver nothing spend the retry budget, and a chunk
    /// that never moves gives up as a network failure saying how far it
    /// got.
    #[test]
    fn a_chunk_that_never_moves_gives_up_after_its_retry_budget() {
        let origin = range_origin(
            numbered(100),
            vec![Serve::StallAfter(0), Serve::StallAfter(0), Serve::StallAfter(0)],
            Duration::ZERO,
        );
        let error = stall_client(Duration::from_secs(1), 1, 4)
            .get_range(&format!("{}/x", origin.base), 0, 99)
            .expect_err("a chunk that never moves");
        assert!(error.is_transfer(), "{error}");
        let message = error.to_string();
        assert!(message.contains("stalled"), "{message}");
        assert!(message.contains("gave up after 2 attempts"), "{message}");
        assert!(message.contains("0 of 100 bytes"), "{message}");
    }

    /// A ranged transfer keeps at most `streams` connections open, not
    /// one per CPU thread.
    #[test]
    fn chunk_streams_are_bounded_by_the_configured_count() {
        let payload = numbered(12 * 100);
        let origin = range_origin(payload.clone(), Vec::new(), Duration::from_millis(150));
        let ranges: Vec<(u64, u64)> = (0..12u64).map(|i| (i * 100, i * 100 + 99)).collect();
        let data = stall_client(Duration::from_secs(5), 1, 3)
            .get_ranges(&format!("{}/x", origin.base), &ranges)
            .expect("all chunks");
        assert_eq!(data, payload);
        let peak = origin.peak.load(Ordering::SeqCst);
        assert!(peak <= 3, "{peak} connections were open at once");
        assert!(peak >= 2, "the chunks did not overlap at all ({peak})");
    }

    /// Every body byte is counted as it arrives, and a whole-file
    /// transfer states its size before its first chunk.
    #[test]
    fn progress_counts_the_bytes_that_arrived() {
        // 17 MiB: two chunks, so the object moves by range.
        let payload = numbered(17 * 1024 * 1024);
        let origin = range_origin(payload.clone(), vec![Serve::Whole], Duration::ZERO);
        let progress = TransferProgress::new();
        let mut client = stall_client(Duration::from_secs(5), 1, 2);
        client.set_progress(progress.clone());
        let data = client
            .get_bytes_parallel_whole(&format!("{}/x", origin.base))
            .expect("whole object");
        assert_eq!(data.len(), payload.len());
        assert_eq!(progress.total(), Some(payload.len() as u64));
        assert_eq!(progress.received(), payload.len() as u64);
        progress.reset();
        assert_eq!((progress.received(), progress.total()), (0, None));
    }

    /// A retry goes out on a new connection, never on the next pooled one.
    ///
    /// The origin answers the first request on each of three connections
    /// and keeps them open, then closes each one when the next request
    /// arrives on it without answering: pooled connections that are dead
    /// when reused, as behind a proxy that has already closed them.  With
    /// one retry allowed, a retry that took the next pooled connection
    /// would meet a second dead one and fail the request.
    #[test]
    fn a_retry_opens_a_new_connection_instead_of_reusing_a_pooled_one() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind test origin");
        let base = format!("http://{}", listener.local_addr().expect("origin addr"));
        let dead_reuses = Arc::new(AtomicUsize::new(0));
        let dead = dead_reuses.clone();
        fn read_head(stream: &mut TcpStream) -> bool {
            let mut head = Vec::new();
            let mut byte = [0u8; 1];
            while !head.ends_with(b"\r\n\r\n") {
                match stream.read(&mut byte) {
                    Ok(1) => head.push(byte[0]),
                    _ => return false,
                }
            }
            true
        }
        thread::spawn(move || {
            for (number, stream) in listener.incoming().enumerate() {
                let Ok(mut stream) = stream else { continue };
                let dead = dead.clone();
                thread::spawn(move || {
                    if !read_head(&mut stream) {
                        return;
                    }
                    if number < 3 {
                        // Long enough for all three first requests to be
                        // in flight at once, each on its own connection.
                        thread::sleep(Duration::from_millis(300));
                        let _ = stream.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\n\r\nhello");
                        let _ = stream.flush();
                        if read_head(&mut stream) {
                            dead.fetch_add(1, Ordering::SeqCst);
                        }
                        // Dropped unanswered: the reused connection is dead.
                    } else {
                        let _ = stream.write_all(
                            b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\nConnection: close\r\n\r\nfresh",
                        );
                        let _ = stream.flush();
                    }
                });
            }
        });
        let client = test_client();
        let url = format!("{}/x", base);
        thread::scope(|scope| {
            let firsts: Vec<_> = (0..3).map(|_| scope.spawn(|| client.get_bytes(&url))).collect();
            for first in firsts {
                assert_eq!(first.join().unwrap().expect("first request"), b"hello");
            }
        });
        let data = client.get_bytes(&url).expect("the retry on a new connection");
        assert_eq!(data, b"fresh");
        assert_eq!(dead_reuses.load(Ordering::SeqCst), 1, "only the first attempt reused a pooled connection");
    }

    /// A URL this crate treats as NOMADS, on a local origin.
    ///
    /// NOMADS is recognised by its host name anywhere in the URL, so the
    /// local origin carries it in the path.  The governor is pointed at a
    /// scratch state file, 1 ms apart with a 1 ms cooldown: the probe is
    /// really paced, but no test waits 2.5 s per request or 15 minutes
    /// after an over-rate-limit answer, and the node's own state file is
    /// never touched.  No other test in this crate sends a NOMADS URL.
    fn nomads_url(base: &str) -> String {
        static GOVERNOR: std::sync::Once = std::sync::Once::new();
        GOVERNOR.call_once(|| {
            let dir = std::env::temp_dir()
                .join(format!("rustwx_nomads_probe_{}", std::process::id()));
            fs::create_dir_all(&dir).expect("scratch state directory");
            std::env::set_var("RUSTWX_NOMADS_RATE_STATE", dir.join("nomads.state"));
            std::env::set_var("RUSTWX_NOMADS_MIN_INTERVAL_MS", "1");
            std::env::set_var("RUSTWX_NOMADS_COOLDOWN_MS", "1");
        });
        format!("{base}/nomads.ncep.noaa.gov/hrrr.t00z.wrfprsf00.grib2")
    }

    const BUSY: &[u8] =
        b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
    const THROTTLED: &[u8] = b"HTTP/1.1 302 Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
    const MISSING: &[u8] = b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
    const PARTIAL: &[u8] = b"HTTP/1.1 206 Partial Content\r\nContent-Range: bytes 0-0/10\r\n\
        Content-Length: 1\r\nConnection: close\r\n\r\nG";

    fn unreachable_reason(outcome: HeadOutcome) -> String {
        match outcome {
            HeadOutcome::Unreachable(why) => why,
            other => panic!("expected an unreachable object, got {other:?}"),
        }
    }

    /// NOMADS asked with nothing listening: the probe learned nothing about
    /// the object, so it is unreachable, not absent.  It used to read as
    /// absent, and rw_fetch refused the hour as "no source served this
    /// object" without asking again.
    #[test]
    fn a_nomads_probe_that_gets_no_answer_is_unreachable_not_absent() {
        let closed = {
            let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
            format!("http://{}", listener.local_addr().expect("address"))
        };
        let why = unreachable_reason(test_client().head_status(&nomads_url(&closed)));
        assert!(why.contains("(bytes=0-0) failed twice"), "{why}");
    }

    /// A 503 on both attempts is the service's state, not the object's.
    #[test]
    fn a_nomads_probe_answered_503_twice_is_unreachable() {
        let base = spawn_http_server(vec![BUSY.to_vec(), BUSY.to_vec()]);
        let why = unreachable_reason(test_client().head_status(&nomads_url(&base)));
        assert!(why.contains("503"), "{why}");
    }

    /// NOMADS says "too fast" through Akamai with a redirect that has no
    /// Location.  Twice is a throttled service, never an absent object.
    #[test]
    fn a_nomads_over_rate_limit_redirect_twice_is_unreachable() {
        let base = spawn_http_server(vec![THROTTLED.to_vec(), THROTTLED.to_vec()]);
        let why = unreachable_reason(test_client().head_status(&nomads_url(&base)));
        assert!(why.contains("no Location header"), "{why}");
    }

    /// The origin's own answers still decide: 206 is there, 404 is not,
    /// and one throttled answer followed by the object is there.
    #[test]
    fn a_nomads_probe_still_tells_present_from_absent() {
        let present = spawn_http_server(vec![PARTIAL.to_vec()]);
        assert_eq!(
            test_client().head_status(&nomads_url(&present)),
            HeadOutcome::Present
        );
        let absent = spawn_http_server(vec![MISSING.to_vec()]);
        assert_eq!(
            test_client().head_status(&nomads_url(&absent)),
            HeadOutcome::Absent
        );
        let throttled_once = spawn_http_server(vec![THROTTLED.to_vec(), PARTIAL.to_vec()]);
        assert_eq!(
            test_client().head_status(&nomads_url(&throttled_once)),
            HeadOutcome::Present
        );
    }

    #[test]
    fn content_range_last_byte_parses() {
        assert_eq!(parse_content_range_last("bytes 10-20/100"), Some(20));
        assert_eq!(parse_content_range_last("bytes 10-20/*"), Some(20));
        assert_eq!(parse_content_range_last("bytes */100"), None);
    }

    #[test]
    fn get_bytes_follows_relative_redirects() {
        let base = spawn_http_server(vec![
            b"HTTP/1.1 302 Found\r\nLocation: /final\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                .to_vec(),
            b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\nConnection: close\r\n\r\nhello".to_vec(),
        ]);
        let client = test_client();
        let body = client
            .get_bytes(&format!("{}/start", base))
            .expect("redirected body");
        assert_eq!(body, b"hello");
    }

    #[test]
    fn get_bytes_surfaces_clear_error_for_redirect_without_location() {
        let base = spawn_http_server(vec![
            b"HTTP/1.1 302 Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".to_vec(),
        ]);
        let client = test_client();
        let err = client
            .get_bytes(&format!("{}/broken", base))
            .expect_err("missing location should fail");
        let message = err.to_string();
        assert!(message.contains("redirect response missing Location header"));
        assert!(!message.contains("protocol: missing a location header"));
    }

    #[test]
    fn content_range_total_parses_known_total() {
        assert_eq!(parse_content_range_total("bytes 0-0/12345"), Some(12345));
        assert_eq!(parse_content_range_total("bytes 10-20/*"), None);
        assert_eq!(parse_content_range_total("not a range"), None);
    }

    #[test]
    fn full_file_ranges_cover_file_once_in_order() {
        assert_eq!(full_file_ranges(0, 4), Vec::<(u64, u64)>::new());
        assert_eq!(full_file_ranges(1, 4), vec![(0, 0)]);
        assert_eq!(full_file_ranges(10, 4), vec![(0, 3), (4, 7), (8, 9)]);
    }

    #[test]
    fn full_object_answer_proves_length_and_can_resume() {
        let base = spawn_http_server(vec![
            b"HTTP/1.1 200 OK\r\nContent-Length: 11\r\nConnection: close\r\n\r\nhello world".to_vec(),
            b"HTTP/1.1 200 OK\r\nContent-Length: 11\r\nConnection: close\r\n\r\nhello".to_vec(),
            b"HTTP/1.1 206 Partial Content\r\nContent-Range: bytes 5-10/11\r\nContent-Length: 6\r\nConnection: close\r\n\r\n world".to_vec(),
        ]);
        let client = test_client();
        let url = format!("{base}/object");
        assert_eq!(client.object_length(&url).unwrap(), Some(11));
        assert_eq!(client.get_range(&url, 0, 10).unwrap(), b"hello world");
    }

    /// A range GET must be answered with the span it asked for.
    #[test]
    fn a_range_reply_that_is_not_the_requested_span_is_refused() {
        // The origin ignores the Range header and sends the whole object.
        let whole = spawn_http_server(vec![
            b"HTTP/1.1 200 OK\r\nContent-Length: 11\r\nConnection: close\r\n\r\nhello world"
                .to_vec(),
        ]);
        let message = test_client()
            .get_range(&format!("{}/x", whole), 0, 4)
            .expect_err("a 200 is not the requested span");
        assert!(!message.is_transfer(), "the origin's answer, not the network's");
        let message = message.to_string();
        assert!(message.contains("not 206"), "{}", message);

        // The origin answers 206 for a different span.
        let wrong_span = spawn_http_server(vec![
            b"HTTP/1.1 206 Partial Content\r\nContent-Range: bytes 6-10/11\r\n\
              Content-Length: 5\r\nConnection: close\r\n\r\nworld"
                .to_vec(),
        ]);
        let message = test_client()
            .get_range(&format!("{}/x", wrong_span), 0, 4)
            .expect_err("a foreign span must be refused");
        assert!(!message.is_transfer(), "the origin's answer, not the network's");
        let message = message.to_string();
        assert!(message.contains("not the requested span"), "{}", message);

        // A resume answered with the ORIGINAL span instead of the rest of
        // it is a foreign span too, and is refused rather than spliced.
        let short_reply = b"HTTP/1.1 206 Partial Content\r\nContent-Range: bytes 0-4/11\r\n\
              Content-Length: 3\r\nConnection: close\r\n\r\nhel"
            .to_vec();
        let short = spawn_http_server(vec![short_reply.clone(), short_reply]);
        let message = test_client()
            .get_range(&format!("{}/x", short), 0, 4)
            .expect_err("a resume answered with another span must be refused");
        assert!(!message.is_transfer(), "the origin's answer, not the network's");
        let message = message.to_string();
        assert!(message.contains("bytes=3-4"), "{}", message);
        assert!(message.contains("not the requested span"), "{}", message);
    }

    /// A body that comes up short is resumed for the bytes it lacks.
    #[test]
    fn a_short_range_body_is_resumed_for_the_bytes_it_lacks() {
        let base = spawn_http_server(vec![
            b"HTTP/1.1 206 Partial Content\r\nContent-Range: bytes 0-4/11\r\n\
              Content-Length: 3\r\nConnection: close\r\n\r\nhel"
                .to_vec(),
            b"HTTP/1.1 206 Partial Content\r\nContent-Range: bytes 3-4/11\r\n\
              Content-Length: 2\r\nConnection: close\r\n\r\nlo"
                .to_vec(),
        ]);
        assert_eq!(
            test_client()
                .get_range(&format!("{}/x", base), 0, 4)
                .expect("the resume supplies the rest"),
            b"hello"
        );
    }

    /// Whose doing a failure was travels in its type: a 4xx is the
    /// origin's answer, a 5xx that outlived every retry and a body that
    /// broke off are the network's.  The message text is the same shape
    /// for both, so a caller that must choose between "ask again later"
    /// and "this will only repeat" reads the variant, not the prose.
    #[test]
    fn a_failure_says_whether_the_network_or_the_origin_ended_it() {
        let missing = spawn_http_server(vec![
            b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".to_vec(),
        ]);
        let error = test_client()
            .get_bytes(&format!("{}/x", missing))
            .expect_err("a 404 is a refusal");
        assert!(!error.is_transfer(), "{}", error);
        assert!(
            error.to_string().starts_with("HTTP error: HTTP request failed"),
            "{}",
            error
        );

        let unavailable =
            b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                .to_vec();
        let busy = spawn_http_server(vec![unavailable.clone(), unavailable]);
        let error = test_client()
            .get_bytes(&format!("{}/x", busy))
            .expect_err("a 503 on every attempt");
        assert!(error.is_transfer(), "{}", error);

        let broken = spawn_http_server(vec![
            b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\nConnection: close\r\n\r\nhe".to_vec(),
        ]);
        let error = test_client()
            .get_bytes(&format!("{}/x", broken))
            .expect_err("a body that broke off");
        assert!(error.is_transfer(), "{}", error);
        assert!(error.to_string().contains("failed to read"), "{}", error);
    }

    /// A chunk whose body breaks off part-way is resumed from where it
    /// broke, not allowed to end the whole object on its first try.
    #[test]
    fn a_range_body_that_breaks_off_is_resumed() {
        let base = spawn_http_server(vec![
            // Promises five bytes, sends two, closes: a dropped connection.
            b"HTTP/1.1 206 Partial Content\r\nContent-Range: bytes 0-4/11\r\n\
              Content-Length: 5\r\nConnection: close\r\n\r\nhe"
                .to_vec(),
            b"HTTP/1.1 206 Partial Content\r\nContent-Range: bytes 2-4/11\r\n\
              Content-Length: 3\r\nConnection: close\r\n\r\nllo"
                .to_vec(),
        ]);
        assert_eq!(
            test_client()
                .get_range(&format!("{}/x", base), 0, 4)
                .expect("the second attempt serves the chunk"),
            b"hello"
        );
    }

    #[test]
    fn a_well_formed_range_reply_is_accepted() {
        let base = spawn_http_server(vec![
            b"HTTP/1.1 206 Partial Content\r\nContent-Range: bytes 0-4/11\r\n\
              Content-Length: 5\r\nConnection: close\r\n\r\nhello"
                .to_vec(),
        ]);
        assert_eq!(
            test_client()
                .get_range(&format!("{}/x", base), 0, 4)
                .expect("well-formed range"),
            b"hello"
        );
    }

    fn state_scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("rustwx_state_{}", name));
        let _ = fs::create_dir_all(&dir);
        dir.join("rustwx_nomads_rate_limit.state")
    }

    /// An absent state is a real zero: nobody has fetched yet.
    #[test]
    fn absent_state_reads_as_a_genuine_zero() {
        let path = state_scratch("absent");
        let _ = fs::remove_file(&path);
        assert_eq!(read_nomads_state(&path), (0, 0, true));
    }

    /// The lie: corruption parsed as zero is permission to send.
    #[test]
    fn corrupt_state_reads_as_just_now_not_as_zero() {
        for body in [
            "",
            "garbage without an equals sign\n",
            "last_request_ms=not-a-number\n",
            "cooldown_until_ms=5\n",
        ] {
            let path = state_scratch("corrupt");
            fs::write(&path, body).unwrap();
            let (last, _cooldown, sound) = read_nomads_state(&path);
            assert!(!sound, "body {:?} should be reported unsound", body);
            assert!(
                last > 0 && last >= now_millis().saturating_sub(60_000),
                "body {:?} must read as a recent request, got {}",
                body,
                last
            );
        }
        let path = state_scratch("corrupt");
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn a_sound_state_round_trips_through_both_halves() {
        let path = state_scratch("roundtrip");
        assert!(write_nomads_state(&path, 111, 222));
        assert_eq!(fs::read_to_string(&path).unwrap(), "last_request_ms=111\ncooldown_until_ms=222\n");
        assert_eq!(read_nomads_state(&path), (111, 222, true));
        let _ = fs::remove_file(&path);
    }

    /// A write that cannot land must be reported, not swallowed: the caller
    /// absorbs the gap locally instead of letting the next process send.
    #[test]
    fn a_state_write_that_cannot_land_reports_failure() {
        let dir = std::env::temp_dir().join("rustwx_state_unwritable");
        let _ = fs::create_dir_all(&dir);
        // A directory where the state file should be: neither write nor
        // rename can succeed onto it.
        let path = dir.join("occupied.state");
        let _ = fs::remove_file(&path);
        let _ = fs::create_dir_all(&path);
        assert!(!write_nomads_state(&path, 1, 2));
        let _ = fs::remove_dir_all(&dir);
    }
}
