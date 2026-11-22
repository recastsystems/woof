//! Plain HTTPS retrieval for the archives that are not S3 buckets.
//!
//! MRMS lives in an S3 bucket and goes through the hardened `ListObjectsV2`
//! client; Stage-IV and the surface archive are ordinary web servers, so
//! they need a GET and a byte ceiling and nothing else. The agent is the
//! same one (same pure-Rust TLS, same timeouts) because two HTTP clients
//! in one binary is two sets of timeout behavior to reason about.
//!
//! Every GET here goes through `rw_http`: bounded backoff with jitter on
//! 429/502/503/504 honouring Retry-After, and a per-host gap shared across
//! processes (the 2026-11-09 rw_asos 429 defect).

use std::error::Error;
use chrono::{DateTime, Utc};

use crate::err;

/// Ceiling on any single retrieval this module performs.
///
/// The largest object any of these routes serves is a Stage-IV daily source
/// tar at under a megabyte and a surface CSV of a few tens of megabytes, so
/// 256 MB is far past anything expected. It exists so a server that answers
/// a range request with an endless stream cannot exhaust memory.
pub const MAX_RESPONSE_BYTES: u64 = 256 * 1024 * 1024;

/// The agent every front door in this crate uses.
pub fn agent() -> ureq::Agent {
    rw_nexrad::s3::build_agent()
}

/// GET `url`, returning its body, refusing anything past the ceiling.
///
/// `subject` names the thing being fetched so a failure says which archive
/// object went missing rather than only which URL.
pub fn get_bytes(
    agent: &ureq::Agent,
    url: &str,
    subject: &str,
) -> Result<Vec<u8>, Box<dyn Error>> {
    let mut response = rw_http::get(agent, url)
        .map_err(|e| err(format!("{subject}: GET {url} failed: {e}")))?;
    let status = response.status().as_u16();
    if !(200..300).contains(&status) {
        return Err(err(format!("{subject}: GET {url} answered HTTP {status}")));
    }
    let bytes = response
        .body_mut()
        .with_config()
        .limit(MAX_RESPONSE_BYTES)
        .read_to_vec()
        .map_err(|e| err(format!("{subject}: reading {url} failed: {e}")))?;
    if bytes.is_empty() {
        return Err(err(format!("{subject}: {url} answered with an empty body")));
    }
    Ok(bytes)
}

/// GET `url` as UTF-8 text through the URL-keyed body cache (`rule` says
/// how long a cached body may answer). Returns the text and whether it was
/// a cache hit (no request sent). Only bodies `accept` approves are cached.
pub fn get_text_cached(
    agent: &ureq::Agent,
    url: &str,
    subject: &str,
    rule: rw_http::CacheRule,
    accept: &dyn Fn(&[u8]) -> bool,
) -> Result<(String, bool), Box<dyn Error>> {
    let fetched = rw_http::get_bytes_cached(agent, url, MAX_RESPONSE_BYTES, rule, accept)
        .map_err(|e| err(format!("{subject}: GET {url} failed: {e}")))?;
    if fetched.bytes.is_empty() {
        return Err(err(format!("{subject}: {url} answered with an empty body")));
    }
    let text = String::from_utf8(fetched.bytes)
        .map_err(|e| err(format!("{subject}: {url} is not UTF-8: {e}")))?;
    Ok((text, fetched.cache_hit))
}

/// GET `url` as UTF-8 text.
pub fn get_text(agent: &ureq::Agent, url: &str, subject: &str) -> Result<String, Box<dyn Error>> {
    let bytes = get_bytes(agent, url, subject)?;
    String::from_utf8(bytes)
        .map_err(|e| err(format!("{subject}: {url} is not UTF-8: {e}")))
}

/// Percent-encode one query-string value.
///
/// Station lists carry only letters and digits and the archive's own
/// separators, but the encoder is applied to every value rather than to the
/// ones that look like they need it: the alternative is a rule somebody has
/// to re-derive each time a parameter is added.
pub fn query_encode(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for byte in value.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(byte as char)
            }
            _ => out.push_str(&format!("%{byte:02X}")),
        }
    }
    out
}

type PublishedBytes = (Vec<u8>, Option<DateTime<Utc>>);

/// Retrieve an archive object with the server's publication time.
pub fn get_with_last_modified(
    client: &ureq::Agent,
    url: &str,
) -> Result<PublishedBytes, Box<dyn Error>> {
    let mut response = rw_http::get(client, url)
        .map_err(|e| err(format!("GET {url} failed: {e}")))?;
    let status = response.status().as_u16();
    if !(200..300).contains(&status) {
        return Err(err(format!("GET {url} answered HTTP {status}")));
    }
    let last_modified = response
        .headers()
        .get("last-modified")
        .and_then(|v| v.to_str().ok())
        .and_then(rw_nexrad::s3::parse_http_date);
    let bytes = response
        .body_mut()
        .with_config()
        .limit(MAX_RESPONSE_BYTES)
        .read_to_vec()
        .map_err(|e| err(format!("reading {url} failed: {e}")))?;
    if bytes.is_empty() {
        return Err(err(format!("{url} answered with an empty body")));
    }
    Ok((bytes, last_modified))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn query_encoding_leaves_unreserved_characters_and_escapes_the_rest() {
        assert_eq!(query_encode("KOKC"), "KOKC");
        assert_eq!(query_encode("A-b_c.d~e"), "A-b_c.d~e");
        assert_eq!(query_encode("Etc/UTC"), "Etc%2FUTC");
        assert_eq!(query_encode("a b"), "a%20b");
        assert_eq!(query_encode("x&y=z"), "x%26y%3Dz");
    }
}
