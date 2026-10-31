//! The HTTP store the reader fetches a public Zarr archive through.
//!
//! `zarrs_http::HTTPStore` builds `reqwest::blocking::Client::new()`, whose
//! whole-request timeout is 30 seconds.  One ERA5 chunk on Google's public
//! copy is one hour of one field on 37 levels, about 80 MB compressed, so on
//! any link slower than about 2.7 MB/s every chunk is cut off mid-body and
//! the fetch fails with "error decoding response body" (measured on a node
//! whose link fell to 1.3 MB/s: 60 s for one chunk).  This store is the same
//! reads with a timeout sized for the chunk, not for a page, and a retry for
//! a transfer the network drops.  The reads follow zarrs_http 0.3.1
//! (MIT OR Apache-2.0), which stays the dependency its notices name.
//!
//! A server error (5xx) or 429 is retried too, with backoff, and every retry
//! is logged on stderr.  Without it one HTTP 500 from the ERA5 ARCO store
//! aborted the hydro init door ("potential_evaporation: http size_key has
//! status code 500"; a rerun succeeded; hydro-world calibration NOTE gap 3,
//! 2026-10-18).

use std::str::FromStr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use reqwest::{
    header::{HeaderValue, CONTENT_LENGTH, RANGE},
    StatusCode, Url,
};
use zarrs::storage::{
    byte_range::ByteRangeIterator, MaybeBytes, MaybeBytesIterator, ReadableStorageTraits, StorageError, StoreKey,
};

/// One request may take this long: an 80 MB chunk at 150 kB/s.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(600);
const CONNECT_TIMEOUT: Duration = Duration::from_secs(30);
// A CONUS 90-day extract left its last slice idle inside one HTTP read for
// minutes. Keep the 600 s whole-transfer allowance for large slow chunks,
// but retry a connection that delivers no bytes for a minute.
const READ_TIMEOUT: Duration = Duration::from_secs(60);
/// Tries per request; a dropped transfer, a 5xx or a 429 is tried again after
/// a backoff of 2, 4, 8 and 16 s (at most 30 s of waiting per request).
const TRIES: u32 = 5;

fn backoff(attempt: u32) -> Duration {
    Duration::from_secs(1u64 << attempt.clamp(1, 4))
}

fn transient(status: StatusCode) -> bool {
    status.is_server_error() || status == StatusCode::TOO_MANY_REQUESTS
}

#[derive(Debug)]
pub struct HttpStore {
    base_url: Url,
    client: reqwest::Client,
    runtime: tokio::runtime::Runtime,
    /// Body bytes received and requests answered, for the run's receipt.
    received: AtomicU64,
    answered: AtomicU64,
}

/// One range read: the bytes, and whether the server ignored the range and
/// sent the entire object instead.
pub struct RangeBody {
    pub bytes: bytes::Bytes,
    pub whole: bool,
}

fn other(err: impl ToString) -> StorageError {
    StorageError::Other(err.to_string())
}

impl HttpStore {
    pub fn new(base_url: &str) -> Result<Self, String> {
        Self::with_read_timeout(base_url, READ_TIMEOUT)
    }

    fn with_read_timeout(base_url: &str, read_timeout: Duration) -> Result<Self, String> {
        let base_url = Url::from_str(base_url).map_err(|_| format!("base URL {base_url} is not valid"))?;
        let runtime = tokio::runtime::Builder::new_multi_thread().worker_threads(2)
            .enable_all().build().map_err(|e| e.to_string())?;
        let client = reqwest::Client::builder()
            .timeout(REQUEST_TIMEOUT)
            .connect_timeout(CONNECT_TIMEOUT)
            .read_timeout(read_timeout)
            .build()
            .map_err(|e| e.to_string())?;
        Ok(Self { base_url, client, runtime, received: AtomicU64::new(0), answered: AtomicU64::new(0) })
    }

    /// Body bytes received and requests answered so far.
    pub fn transfer(&self) -> (u64, u64) {
        (self.received.load(Ordering::Relaxed), self.answered.load(Ordering::Relaxed))
    }

    /// Bytes `start..end` of `key`, or fewer when the object ends sooner;
    /// None when the object does not exist.
    pub fn get_range(&self, key: &StoreKey, start: u64, end: u64) -> Result<Option<RangeBody>, StorageError> {
        let url = self.url(key)?;
        let header = HeaderValue::from_str(&format!("bytes={}-{}", start, end - 1)).map_err(other)?;
        match self.fetch(&format!("GET {url} range {start}-{}", end - 1), || {
            self.client.get(url.clone()).header(RANGE, header.clone())
        })? {
            (StatusCode::PARTIAL_CONTENT, Some(body), _) if body.len() as u64 <= end - start => {
                Ok(Some(RangeBody { bytes: body, whole: false }))
            }
            (StatusCode::OK, Some(body), _) => Ok(Some(RangeBody { bytes: body, whole: true })),
            (StatusCode::NOT_FOUND, _, _) => Ok(None),
            (status, _, _) => Err(other(format!("the http server responded with status {status} for a byte range"))),
        }
    }

    fn url(&self, key: &StoreKey) -> Result<Url, StorageError> {
        let mut url = self.base_url.as_str().to_string();
        if !key.as_str().is_empty() {
            url += &("/".to_string() + key.as_str().strip_prefix('/').unwrap_or(key.as_str()));
        }
        Url::parse(&url).map_err(other)
    }

    /// Send a request and read its whole body, trying again when the transfer
    /// fails or the server answers 5xx/429.  `what` names the request in the log.
    fn fetch(
        &self,
        what: &str,
        build: impl Fn() -> reqwest::RequestBuilder,
    ) -> Result<(StatusCode, Option<bytes::Bytes>, Option<u64>), StorageError> {
        let mut last = String::new();
        for attempt in 0..TRIES {
            if attempt > 0 {
                let wait = backoff(attempt);
                eprintln!(
                    "rw_zarr: {what}: {last}; retry {attempt}/{} in {} s",
                    TRIES - 1,
                    wait.as_secs()
                );
                std::thread::sleep(wait);
            }
            let response = match self.runtime.block_on(async { build().send().await }) {
                Ok(r) => r,
                Err(e) => {
                    last = e.to_string();
                    continue;
                }
            };
            let status = response.status();
            let length = response
                .headers()
                .get(CONTENT_LENGTH)
                .and_then(|v| v.to_str().ok())
                .and_then(|s| u64::from_str(s).ok());
            if transient(status) && attempt + 1 < TRIES {
                last = format!("status {status}");
                continue;
            }
            if status != StatusCode::OK && status != StatusCode::PARTIAL_CONTENT {
                if transient(status) {
                    eprintln!("rw_zarr: {what}: status {status} after {TRIES} tries; giving up");
                }
                return Ok((status, None, length));
            }
            match self.runtime.block_on(async { response.bytes().await }) {
                Ok(body) => {
                    self.received.fetch_add(body.len() as u64, Ordering::Relaxed);
                    self.answered.fetch_add(1, Ordering::Relaxed);
                    return Ok((status, Some(body), length));
                }
                Err(e) => last = e.to_string(),
            }
        }
        Err(other(format!("http transfer failed after {TRIES} tries: {last}")))
    }
}

impl ReadableStorageTraits for HttpStore {
    fn get(&self, key: &StoreKey) -> Result<MaybeBytes, StorageError> {
        let url = self.url(key)?;
        match self.fetch(&format!("GET {url}"), || self.client.get(url.clone()))? {
            (StatusCode::OK, body, _) => Ok(body),
            (StatusCode::NOT_FOUND, _, _) => Ok(None),
            (status, _, _) => Err(other(format!("http unexpected status code: {status}"))),
        }
    }

    fn get_partial_many<'a>(
        &'a self,
        key: &StoreKey,
        byte_ranges: ByteRangeIterator<'a>,
    ) -> Result<MaybeBytesIterator<'a>, StorageError> {
        let url = self.url(key)?;
        let Some(size) = self.size_key(key)? else {
            return Ok(None);
        };
        // One single-part range request per range: every server answers those.
        let mut parts = Vec::new();
        for range in byte_ranges {
            let (start, end) = (range.start(size), range.end(size));
            let header = HeaderValue::from_str(&format!("bytes={}-{}", start, end - 1)).map_err(other)?;
            match self.fetch(&format!("GET {url} range {start}-{}", end - 1), || {
                self.client.get(url.clone()).header(RANGE, header.clone())
            })? {
                (StatusCode::PARTIAL_CONTENT, Some(body), _) if body.len() as u64 == end - start => parts.push(Ok(body)),
                (StatusCode::OK, Some(body), _) => {
                    let (s, e) = (usize::try_from(start).map_err(other)?, usize::try_from(end).map_err(other)?);
                    parts.push(Ok(body.slice(s..e)));
                }
                (status, _, _) => {
                    return Err(other(format!("the http server responded with status {status} for a byte range")));
                }
            }
        }
        Ok(Some(Box::new(parts.into_iter())))
    }

    fn size_key(&self, key: &StoreKey) -> Result<Option<u64>, StorageError> {
        let url = self.url(key)?;
        match self.fetch(&format!("HEAD {url}"), || self.client.head(url.clone()))? {
            (StatusCode::OK, _, Some(length)) => Ok(Some(length)),
            (StatusCode::OK, _, None) => Err(other("content length response is invalid")),
            (StatusCode::NOT_FOUND, _, _) => Ok(None),
            (status, _, _) => Err(other(format!("http size_key has status code {status}"))),
        }
    }

    fn supports_get_partial(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Arc;

    #[test]
    fn stalled_partial_body_retries_without_shortening_whole_chunk_timeout() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/era5.zarr", listener.local_addr().unwrap());
        let server = std::thread::spawn(move || {
            for (attempt, stream) in listener.incoming().take(2).enumerate() {
                let mut stream = stream.unwrap();
                let mut request = Vec::new();
                let mut byte = [0u8; 1];
                while !request.ends_with(b"\r\n\r\n") && stream.read(&mut byte).unwrap() == 1 {
                    request.push(byte[0]);
                }
                let header = b"HTTP/1.1 200 OK\r\nContent-Length: 4\r\nConnection: close\r\n\r\n";
                if attempt == 0 {
                    // The stalled attempt's client may already have timed out and closed (its read timeout is
                    // 50 ms), so its writes may fail. Breakage this prevents (public ci native windows-2025, run
                    // 40418223243): the unwrap panicked on WSAECONNABORTED (10053), the server thread died, and
                    // every retry then failed to connect.
                    let _ = stream.write_all(header).and_then(|_| stream.write_all(b"a"));
                    std::thread::sleep(Duration::from_millis(250));
                } else {
                    stream.write_all(header).unwrap();
                    stream.write_all(b"abcd").unwrap();
                }
            }
        });
        let store = HttpStore::with_read_timeout(&url, Duration::from_millis(50)).unwrap();
        let bytes = store.get(&StoreKey::new("chunk").unwrap()).unwrap().unwrap();
        assert_eq!(bytes.as_ref(), b"abcd");
        server.join().unwrap();
    }

    /// The defect: one HTTP 500 from the store aborted the init door.  A HEAD
    /// that gets 500 once and then 200 must return the size, after one retry.
    #[test]
    fn size_key_retries_a_server_500() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let served = Arc::new(AtomicU32::new(0));
        let count = served.clone();
        std::thread::spawn(move || {
            for stream in listener.incoming().take(2) {
                let mut stream = stream.unwrap();
                let mut request = Vec::new();
                let mut byte = [0u8; 1];
                while !request.ends_with(b"\r\n\r\n") && stream.read(&mut byte).unwrap() == 1 {
                    request.push(byte[0]);
                }
                let reply: &[u8] = if count.fetch_add(1, Ordering::SeqCst) == 0 {
                    b"HTTP/1.1 500 Internal Server Error\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                } else {
                    b"HTTP/1.1 200 OK\r\nContent-Length: 1234\r\nConnection: close\r\n\r\n"
                };
                stream.write_all(reply).unwrap();
            }
        });
        let store = HttpStore::new(&format!("http://127.0.0.1:{port}/era5.zarr")).unwrap();
        let key = StoreKey::new("potential_evaporation/.zarray").unwrap();
        assert_eq!(store.size_key(&key).unwrap(), Some(1234));
        assert_eq!(served.load(Ordering::SeqCst), 2);
    }
}
