//! Retry, pacing and caching for the ureq front doors.
//!
//! Defect this answers (2026-11-09): `rw_asos fetch` sent one GET per chunk
//! and returned the first error it saw. Iowa Mesonet rate-limits with
//! HTTP 429, and a scorer that runs one `rw_asos fetch` process per frame
//! lost ASOS on 4 of 5 frames. Every other ureq door in rw-obs, rw-nexrad
//! (MRMS goes through its S3 client) and rw-sat had the same gap.
//!
//! What a request made through [`send`] gets:
//!
//! 1. **Bounded retry.** HTTP 429, 502, 503 and 504, and transport
//!    failures (timeout, refused or reset connection), are retried with
//!    exponential backoff and jitter. A `Retry-After` header (seconds or an
//!    HTTP date) is honoured: the wait is the larger of it and the backoff.
//!    Attempts are bounded (7 for statuses, 3 for transport failures by
//!    default), and a `Retry-After` past [`Policy::max_retry_after`] is not
//!    waited out: the request fails at once and says so.
//! 2. **A per-host gap shared across processes.** Before each attempt the
//!    request reserves a start slot in a small per-host file under the temp
//!    directory (`RW_HTTP_STATE_DIR` overrides). Hosts with a measured limit
//!    carry a minimum gap ([`HOST_MIN_GAP`]); every host carries the
//!    cooldown a 429 or 503 sets, so one process being told to slow down
//!    slows every process talking to that host.
//! 3. **The caller's error contract is unchanged.** A final status of 400
//!    or more comes back as `ureq::Error::StatusCode(code)`, exactly what a
//!    plain `agent.get(url).call()` returns, so call sites that match on a
//!    404 keep working.
//!
//! [`get_bytes_cached`] adds an opt-in URL-keyed body cache on top, for
//! archive windows that are asked for more than once (two arms of an A/B
//! scoring the same frame, a rerun of a scorer).

use std::collections::hash_map::RandomState;
use std::error::Error;
use std::fs::{self, File, OpenOptions};
use std::hash::{BuildHasher, Hasher};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use sha2::{Digest, Sha256};
use ureq::http::Response;
use ureq::{Agent, Body};

/// Minimum gap between request starts, per host, in milliseconds.
///
/// Only hosts with a measured limit are listed; any other host is paced by
/// the cooldowns its own 429/503 answers set. `RW_HTTP_PACE`
/// (`host=ms,host=ms`) adds or overrides entries.
///
/// * Iowa Environmental Mesonet: seven case boxes pulled back to back at a
///   2 s pause earned a 429 on the twenty-first request (2026-08-04, the
///   measurement behind `rw_asos --request-pause-ms`); a scorer running one
///   process per frame sent requests with no pause at all.
/// * NOMADS: the 2.5 s gap wx-core's NOMADS governor already holds model
///   fetches to; prepbufr comes from the same host.
pub const HOST_MIN_GAP: &[(&str, u64)] = &[
    ("mesonet.agron.iastate.edu", 2000),
    ("nomads.ncep.noaa.gov", 2500),
];

/// Retry policy. [`Policy::from_env`] is what [`send`] uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Policy {
    /// Attempts in all for a retryable status (429/502/503/504).
    pub max_attempts: u32,
    /// Attempts in all for a transport failure.
    pub max_transport_attempts: u32,
    /// First backoff; doubles per retry.
    pub base_delay: Duration,
    /// Ceiling on any one backoff (before Retry-After is applied).
    pub max_delay: Duration,
    /// A Retry-After longer than this is not waited out.
    pub max_retry_after: Duration,
}

impl Default for Policy {
    fn default() -> Self {
        Self {
            max_attempts: 7,
            max_transport_attempts: 3,
            base_delay: Duration::from_secs(2),
            max_delay: Duration::from_secs(60),
            max_retry_after: Duration::from_secs(300),
        }
    }
}

impl Policy {
    /// The default policy with `RW_HTTP_MAX_ATTEMPTS`, `RW_HTTP_BASE_DELAY_MS`
    /// and `RW_HTTP_MAX_DELAY_MS` applied when set.
    pub fn from_env() -> Self {
        let mut policy = Self::default();
        if let Some(v) = env_u64("RW_HTTP_MAX_ATTEMPTS") {
            policy.max_attempts = (v as u32).max(1);
        }
        if let Some(v) = env_u64("RW_HTTP_BASE_DELAY_MS") {
            policy.base_delay = Duration::from_millis(v);
        }
        if let Some(v) = env_u64("RW_HTTP_MAX_DELAY_MS") {
            policy.max_delay = Duration::from_millis(v);
        }
        policy
    }

    /// The backoff before retry number `retry` (1-based), before jitter.
    pub fn backoff(&self, retry: u32) -> Duration {
        let factor = 1u64.checked_shl(retry.saturating_sub(1).min(30)).unwrap_or(u64::MAX);
        let ms = (self.base_delay.as_millis() as u64).saturating_mul(factor);
        Duration::from_millis(ms).min(self.max_delay)
    }
}

fn env_u64(name: &str) -> Option<u64> {
    std::env::var(name).ok().and_then(|v| v.trim().parse().ok())
}

/// The HTTP method of a request made through [`send`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Method {
    Get,
    Head,
}

/// GET `url` with retry and pacing. Same error contract as `call()`.
pub fn get(agent: &Agent, url: &str) -> Result<Response<Body>, ureq::Error> {
    send(agent, Method::Get, url, &[])
}

/// HEAD `url` with retry and pacing. Same error contract as `call()`.
pub fn head(agent: &Agent, url: &str) -> Result<Response<Body>, ureq::Error> {
    send(agent, Method::Head, url, &[])
}

/// `method url` with `headers`, retried and paced under [`Policy::from_env`].
pub fn send(
    agent: &Agent,
    method: Method,
    url: &str,
    headers: &[(&str, &str)],
) -> Result<Response<Body>, ureq::Error> {
    send_with(agent, method, url, headers, &Policy::from_env())
}

/// [`send`] under an explicit policy.
pub fn send_with(
    agent: &Agent,
    method: Method,
    url: &str,
    headers: &[(&str, &str)],
    policy: &Policy,
) -> Result<Response<Body>, ureq::Error> {
    let host = host_of(url);
    let gap = host.as_deref().map(min_gap_for).unwrap_or(Duration::ZERO);
    let mut attempt = 0u32;
    loop {
        attempt += 1;
        if let Some(host) = host.as_deref() {
            pace(host, gap);
        }
        let outcome = match method {
            Method::Get => {
                let mut request = agent.get(url);
                for (name, value) in headers {
                    request = request.header(*name, *value);
                }
                request.config().http_status_as_error(false).build().call()
            }
            Method::Head => {
                let mut request = agent.head(url);
                for (name, value) in headers {
                    request = request.header(*name, *value);
                }
                request.config().http_status_as_error(false).build().call()
            }
        };
        match outcome {
            Ok(response) => {
                let status = response.status().as_u16();
                if status < 400 {
                    return Ok(response);
                }
                if !retryable_status(status) {
                    return Err(ureq::Error::StatusCode(status));
                }
                let retry_after = response
                    .headers()
                    .get("retry-after")
                    .and_then(|v| v.to_str().ok())
                    .and_then(|v| parse_retry_after(v, SystemTime::now()));
                if let Some(asked) = retry_after {
                    if asked > policy.max_retry_after {
                        eprintln!(
                            "rw-http: HTTP {status} from {} asks Retry-After {} s, past the {} s \
                             this client waits out; giving up on {url}",
                            host.as_deref().unwrap_or("?"),
                            asked.as_secs(),
                            policy.max_retry_after.as_secs()
                        );
                        if let Some(host) = host.as_deref() {
                            cool_down(host, policy.max_retry_after);
                        }
                        return Err(ureq::Error::StatusCode(status));
                    }
                }
                if attempt >= policy.max_attempts {
                    eprintln!(
                        "rw-http: HTTP {status} from {} on all {attempt} attempts; giving up on {url}",
                        host.as_deref().unwrap_or("?")
                    );
                    return Err(ureq::Error::StatusCode(status));
                }
                let wait = jittered(policy.backoff(attempt)).max(retry_after.unwrap_or(Duration::ZERO));
                eprintln!(
                    "rw-http: HTTP {status} from {} (attempt {attempt}/{}); retrying in {:.1} s{}",
                    host.as_deref().unwrap_or("?"),
                    policy.max_attempts,
                    wait.as_secs_f64(),
                    retry_after
                        .map(|d| format!(" (Retry-After {} s)", d.as_secs()))
                        .unwrap_or_default()
                );
                // 429 and 503 are the host saying "slow down": every process
                // on this machine hears it, not only this one.
                if matches!(status, 429 | 503) {
                    if let Some(host) = host.as_deref() {
                        cool_down(host, wait);
                    }
                }
                std::thread::sleep(wait);
            }
            Err(error) => {
                if !retryable_transport(&error) || attempt >= policy.max_transport_attempts {
                    return Err(error);
                }
                let wait = jittered(policy.backoff(attempt));
                eprintln!(
                    "rw-http: {error} talking to {} (attempt {attempt}/{}); retrying in {:.1} s",
                    host.as_deref().unwrap_or("?"),
                    policy.max_transport_attempts,
                    wait.as_secs_f64()
                );
                std::thread::sleep(wait);
            }
        }
    }
}

/// Statuses that mean "ask again later".
pub fn retryable_status(status: u16) -> bool {
    matches!(status, 429 | 502 | 503 | 504)
}

fn retryable_transport(error: &ureq::Error) -> bool {
    matches!(
        error,
        ureq::Error::Timeout(_) | ureq::Error::Io(_) | ureq::Error::ConnectionFailed | ureq::Error::BodyStalled
    )
}

/// A `Retry-After` value as a wait from `now`: delta-seconds or an HTTP date.
pub fn parse_retry_after(value: &str, now: SystemTime) -> Option<Duration> {
    let value = value.trim();
    if let Ok(seconds) = value.parse::<u64>() {
        return Some(Duration::from_secs(seconds));
    }
    let when = chrono::DateTime::parse_from_rfc2822(value).ok()?;
    let when_ms = when.timestamp_millis();
    let now_ms = now.duration_since(UNIX_EPOCH).ok()?.as_millis() as i64;
    Some(Duration::from_millis(when_ms.saturating_sub(now_ms).max(0) as u64))
}

/// `delay` scaled into [delay/2, delay] ("equal jitter"), so processes told
/// to back off at the same moment do not all come back at the same moment.
fn jittered(delay: Duration) -> Duration {
    let mut hasher = RandomState::new().build_hasher();
    hasher.write_u128(SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0));
    hasher.write_u32(std::process::id());
    let fraction = (hasher.finish() >> 11) as f64 / (1u64 << 53) as f64;
    delay.mul_f64(0.5 + 0.5 * fraction)
}

/// The lower-cased host of an http(s) URL.
pub fn host_of(url: &str) -> Option<String> {
    let rest = url.split_once("://").map(|(_, r)| r)?;
    let authority = rest.split(['/', '?', '#']).next()?;
    let authority = authority.rsplit('@').next()?;
    let host = if authority.starts_with('[') {
        authority.split(']').next().map(|h| format!("{h}]"))?
    } else {
        authority.split(':').next()?.to_string()
    };
    (!host.is_empty()).then(|| host.to_ascii_lowercase())
}

/// The minimum gap for `host`: `RW_HTTP_PACE` first, then [`HOST_MIN_GAP`].
pub fn min_gap_for(host: &str) -> Duration {
    if let Ok(spec) = std::env::var("RW_HTTP_PACE") {
        for entry in spec.split(',') {
            if let Some((name, ms)) = entry.split_once('=') {
                if name.trim().eq_ignore_ascii_case(host) {
                    if let Ok(ms) = ms.trim().parse::<u64>() {
                        return Duration::from_millis(ms);
                    }
                }
            }
        }
    }
    HOST_MIN_GAP
        .iter()
        .find(|(name, _)| *name == host)
        .map(|(_, ms)| Duration::from_millis(*ms))
        .unwrap_or(Duration::ZERO)
}

fn state_dir() -> PathBuf {
    std::env::var_os("RW_HTTP_STATE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("rw-http-pace"))
}

fn now_ms() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0)
}

/// Open and exclusively lock `host`'s state file, run `update` on the
/// stored next-free instant (unix ms), store what it returns.
///
/// Pacing is best effort: if the state file cannot be used (read-only temp
/// dir, odd filesystem) the request goes ahead unpaced rather than failing.
fn with_host_state(host: &str, update: impl FnOnce(u64) -> u64) -> Option<u64> {
    let dir = state_dir();
    fs::create_dir_all(&dir).ok()?;
    let name: String = host
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '.' || c == '-' { c } else { '_' })
        .collect();
    let path = dir.join(format!("{name}.next"));
    let mut file = OpenOptions::new().read(true).write(true).create(true).truncate(false).open(&path).ok()?;
    file.lock().ok()?;
    let mut text = String::new();
    let _ = file.read_to_string(&mut text);
    let stored = text.trim().parse::<u64>().unwrap_or(0);
    let next = update(stored);
    if next != stored {
        let _ = file.seek(SeekFrom::Start(0));
        let _ = file.set_len(0);
        let _ = file.write_all(next.to_string().as_bytes());
    }
    let _ = file.unlock();
    Some(next)
}

/// Wait for this process's slot on `host`, reserving the next one `gap` on.
fn pace(host: &str, gap: Duration) {
    let now = now_ms();
    let mut slot = now;
    with_host_state(host, |next_free| {
        slot = next_free.max(now);
        if gap.is_zero() { next_free } else { slot + gap.as_millis() as u64 }
    });
    if slot > now {
        std::thread::sleep(Duration::from_millis(slot - now));
    }
}

/// Push `host`'s next-free instant at least `wait` into the future.
fn cool_down(host: &str, wait: Duration) {
    let until = now_ms() + wait.as_millis() as u64;
    with_host_state(host, |next_free| next_free.max(until));
}

// ---------------------------------------------------------------- cache

/// How long a cached body may answer for its URL.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheRule {
    /// Always fetch; nothing is read from or written to the cache.
    Off,
    /// A cached body younger than this answers.
    Ttl(Duration),
    /// The URL names settled archive content: a cached body always answers.
    Forever,
}

/// Cached bodies untouched this long are swept when a new body is written.
const CACHE_SWEEP_AGE: Duration = Duration::from_secs(14 * 24 * 3600);

/// Where cached bodies live: `RW_HTTP_CACHE_DIR`, else the user cache dir.
pub fn cache_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("RW_HTTP_CACHE_DIR") {
        return PathBuf::from(dir);
    }
    if let Some(dir) = std::env::var_os("XDG_CACHE_HOME") {
        return PathBuf::from(dir).join("rw-http");
    }
    if let Some(dir) = std::env::var_os("LOCALAPPDATA") {
        return PathBuf::from(dir).join("rw-http");
    }
    if let Some(home) = std::env::var_os("HOME") {
        return PathBuf::from(home).join(".cache").join("rw-http");
    }
    std::env::temp_dir().join("rw-http-cache")
}

fn cache_disabled() -> bool {
    matches!(
        std::env::var("RW_HTTP_CACHE").map(|v| v.to_ascii_lowercase()).as_deref(),
        Ok("off") | Ok("0") | Ok("no") | Ok("false")
    )
}

fn cache_path(dir: &Path, url: &str) -> PathBuf {
    let digest = Sha256::digest(url.as_bytes());
    let mut key = String::with_capacity(64);
    for byte in digest {
        key.push_str(&format!("{byte:02x}"));
    }
    dir.join(format!("{key}.body"))
}

fn cached_body(path: &Path, rule: CacheRule) -> Option<Vec<u8>> {
    let meta = fs::metadata(path).ok()?;
    if let CacheRule::Ttl(ttl) = rule {
        let age = meta.modified().ok().and_then(|m| SystemTime::now().duration_since(m).ok())?;
        if age > ttl {
            return None;
        }
    }
    let bytes = fs::read(path).ok()?;
    (!bytes.is_empty()).then_some(bytes)
}

fn store_body(dir: &Path, path: &Path, bytes: &[u8]) {
    if fs::create_dir_all(dir).is_err() {
        return;
    }
    let tmp = path.with_extension(format!("tmp{}", std::process::id()));
    let written = File::create(&tmp).and_then(|mut f| f.write_all(bytes).and_then(|_| f.sync_all()));
    if written.is_err() || fs::rename(&tmp, path).is_err() {
        let _ = fs::remove_file(&tmp);
        return;
    }
    // No hoarding: bodies nobody has asked for in two weeks go.
    if let Ok(entries) = fs::read_dir(dir) {
        let now = SystemTime::now();
        for entry in entries.flatten() {
            let stale = entry
                .metadata()
                .ok()
                .and_then(|m| m.modified().ok())
                .and_then(|m| now.duration_since(m).ok())
                .is_some_and(|age| age > CACHE_SWEEP_AGE);
            if stale {
                let _ = fs::remove_file(entry.path());
            }
        }
    }
}

/// A body fetched through [`get_bytes_cached`].
#[derive(Debug)]
pub struct Fetched {
    pub bytes: Vec<u8>,
    /// True when the body came from the cache and no request was sent.
    pub cache_hit: bool,
}

/// GET `url` (retried and paced), answered from the cache when `rule`
/// allows. Only a 2xx, non-empty body that `accept` approves is ever
/// cached (so a 200 carrying an error page is not kept). Errors keep
/// `call()`'s contract; a body past `limit` is `BodyExceedsLimit`.
pub fn get_bytes_cached(
    agent: &Agent,
    url: &str,
    limit: u64,
    rule: CacheRule,
    accept: &dyn Fn(&[u8]) -> bool,
) -> Result<Fetched, Box<dyn Error>> {
    let use_cache = rule != CacheRule::Off && !cache_disabled();
    let dir = cache_dir();
    let path = cache_path(&dir, url);
    if use_cache {
        if let Some(bytes) = cached_body(&path, rule).filter(|b| accept(b)) {
            return Ok(Fetched { bytes, cache_hit: true });
        }
    }
    let mut response = get(agent, url)?;
    let bytes = response.body_mut().with_config().limit(limit).read_to_vec()?;
    if use_cache && !bytes.is_empty() && accept(&bytes) {
        store_body(&dir, &path, &bytes);
    }
    Ok(Fetched { bytes, cache_hit: false })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader};
    use std::net::TcpListener;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    /// A one-thread HTTP/1.1 server answering each connection with the next
    /// canned response; returns its base URL and the request count.
    fn serve(responses: Vec<String>) -> (String, Arc<AtomicUsize>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://127.0.0.1:{}", listener.local_addr().unwrap().port());
        let count = Arc::new(AtomicUsize::new(0));
        let seen = count.clone();
        let queue = Arc::new(Mutex::new(responses.into_iter()));
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { break };
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut line = String::new();
                loop {
                    line.clear();
                    if reader.read_line(&mut line).unwrap_or(0) == 0 || line == "\r\n" {
                        break;
                    }
                }
                seen.fetch_add(1, Ordering::SeqCst);
                let reply = queue.lock().unwrap().next();
                let Some(reply) = reply else { break };
                let _ = stream.write_all(reply.as_bytes());
            }
        });
        (base, count)
    }

    fn reply(status: &str, extra: &str, body: &str) -> String {
        format!(
            "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n{extra}\r\n{body}",
            body.len()
        )
    }

    fn fast() -> Policy {
        Policy {
            max_attempts: 4,
            max_transport_attempts: 2,
            base_delay: Duration::from_millis(10),
            max_delay: Duration::from_millis(40),
            max_retry_after: Duration::from_secs(5),
        }
    }

    fn isolated_state() {
        // Each test process gets its own pacing directory, so tests on one
        // machine never wait on each other's reservations.
        let dir = std::env::temp_dir().join(format!("rw-http-test-{}", std::process::id()));
        // SAFETY: tests set this before any request and never unset it.
        unsafe { std::env::set_var("RW_HTTP_STATE_DIR", dir) };
    }

    #[test]
    fn a_429_is_retried_until_the_body_arrives() {
        isolated_state();
        let (base, count) = serve(vec![
            reply("429 Too Many Requests", "", "slow down"),
            reply("429 Too Many Requests", "Retry-After: 0\r\n", "slow down"),
            reply("200 OK", "", "station,valid\n"),
        ]);
        let agent: Agent = Agent::config_builder().build().into();
        let mut response = send_with(&agent, Method::Get, &format!("{base}/x"), &[], &fast()).unwrap();
        assert_eq!(response.body_mut().read_to_string().unwrap(), "station,valid\n");
        assert_eq!(count.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn retries_are_bounded_and_end_in_the_plain_status_error() {
        isolated_state();
        let (base, count) = serve((0..10).map(|_| reply("503 Service Unavailable", "", "")).collect());
        let agent: Agent = Agent::config_builder().build().into();
        let error = send_with(&agent, Method::Get, &format!("{base}/x"), &[], &fast()).unwrap_err();
        assert!(matches!(error, ureq::Error::StatusCode(503)), "{error}");
        assert_eq!(count.load(Ordering::SeqCst), 4);
    }

    #[test]
    fn a_404_is_returned_at_once_as_call_would() {
        isolated_state();
        let (base, count) = serve(vec![reply("404 Not Found", "", ""), reply("200 OK", "", "x")]);
        let agent: Agent = Agent::config_builder().build().into();
        let error = send_with(&agent, Method::Get, &format!("{base}/x"), &[], &fast()).unwrap_err();
        assert!(matches!(error, ureq::Error::StatusCode(404)));
        assert_eq!(count.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn a_retry_after_past_the_ceiling_is_not_waited_out() {
        isolated_state();
        let (base, count) = serve(vec![reply("429 Too Many Requests", "Retry-After: 3600\r\n", ""), reply("200 OK", "", "x")]);
        let agent: Agent = Agent::config_builder().build().into();
        let started = std::time::Instant::now();
        let error = send_with(&agent, Method::Get, &format!("{base}/x"), &[], &fast()).unwrap_err();
        assert!(matches!(error, ureq::Error::StatusCode(429)));
        assert_eq!(count.load(Ordering::SeqCst), 1);
        assert!(started.elapsed() < Duration::from_secs(2));
    }

    #[test]
    fn retry_after_is_read_as_seconds_or_an_http_date() {
        let now = UNIX_EPOCH + Duration::from_secs(784_111_777); // Sun, 06 Nov 1994 08:49:37 GMT
        assert_eq!(parse_retry_after("120", now), Some(Duration::from_secs(120)));
        assert_eq!(parse_retry_after("Sun, 06 Nov 1994 08:49:47 GMT", now), Some(Duration::from_secs(10)));
        assert_eq!(parse_retry_after("Sun, 06 Nov 1994 08:49:00 GMT", now), Some(Duration::ZERO));
        assert_eq!(parse_retry_after("soon", now), None);
    }

    #[test]
    fn backoff_doubles_to_its_ceiling_and_jitter_stays_inside_half_to_whole() {
        let p = Policy::default();
        assert_eq!(p.backoff(1), Duration::from_secs(2));
        assert_eq!(p.backoff(2), Duration::from_secs(4));
        assert_eq!(p.backoff(6), Duration::from_secs(60));
        assert_eq!(p.backoff(40), Duration::from_secs(60));
        for _ in 0..100 {
            let j = jittered(Duration::from_secs(10));
            assert!(j >= Duration::from_secs(5) && j <= Duration::from_secs(10), "{j:?}");
        }
    }

    #[test]
    fn hosts_are_parsed_and_the_measured_gaps_apply() {
        assert_eq!(host_of("https://Mesonet.Agron.IAState.edu/cgi-bin/x?y=1").as_deref(), Some("mesonet.agron.iastate.edu"));
        assert_eq!(host_of("http://127.0.0.1:8080/x").as_deref(), Some("127.0.0.1"));
        assert_eq!(host_of("https://u:p@host.example/x").as_deref(), Some("host.example"));
        assert_eq!(host_of("not a url"), None);
        assert_eq!(min_gap_for("mesonet.agron.iastate.edu"), Duration::from_millis(2000));
        assert_eq!(min_gap_for("noaa-mrms-pds.s3.amazonaws.com"), Duration::ZERO);
    }

    #[test]
    fn the_host_gap_is_shared_through_the_state_file() {
        isolated_state();
        let host = format!("pace-test-{}.example", std::process::id());
        let gap = Duration::from_millis(150);
        let started = std::time::Instant::now();
        pace(&host, gap);
        pace(&host, gap);
        pace(&host, gap);
        // Three starts at a 150 ms gap: the third cannot begin before 300 ms.
        assert!(started.elapsed() >= Duration::from_millis(290), "{:?}", started.elapsed());
    }

    #[test]
    fn a_cached_body_answers_without_a_request() {
        isolated_state();
        let dir = std::env::temp_dir().join(format!("rw-http-cache-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        // SAFETY: as above.
        unsafe { std::env::set_var("RW_HTTP_CACHE_DIR", &dir) };
        let (base, count) = serve(vec![reply("200 OK", "", "a,b\n1,2\n"), reply("200 OK", "", "changed\n")]);
        let agent: Agent = Agent::config_builder().build().into();
        let url = format!("{base}/window");
        let any = |_: &[u8]| true;
        let first = get_bytes_cached(&agent, &url, 1 << 20, CacheRule::Forever, &any).unwrap();
        let second = get_bytes_cached(&agent, &url, 1 << 20, CacheRule::Forever, &any).unwrap();
        assert!(!first.cache_hit && second.cache_hit);
        assert_eq!(first.bytes, second.bytes);
        assert_eq!(count.load(Ordering::SeqCst), 1);
        // Off always asks.
        let third = get_bytes_cached(&agent, &url, 1 << 20, CacheRule::Off, &any).unwrap();
        assert!(!third.cache_hit);
        assert_eq!(third.bytes, b"changed\n");
        let _ = fs::remove_dir_all(&dir);
    }
}
