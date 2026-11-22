//! Anonymous AWS S3 access to the NOAA GOES open-data buckets over plain
//! HTTPS: adapted from the old rustwx `satellite/native_sequence.rs` client
//! (the paginated variant with URL encoding and XML unescaping), extended
//! with `start-after` incremental listing, which is the polling primitive:
//! keys under a band-specific prefix sort lexicographically by scan start
//! time, so `start-after={last seen key}` returns exactly the newer objects.
//!
//! No SDK, no credentials: `ListObjectsV2` via
//! `https://{bucket}.s3.amazonaws.com/?list-type=2&prefix=...` and plain
//! GETs for objects, TLS through rustls + rustls-rustcrypto (pure Rust).

use std::error::Error;
use std::path::{Path, PathBuf};
use std::time::Duration;

use chrono::{DateTime, Datelike, Timelike, Utc};

use rw_store::atomic::atomic_write_bytes;

use crate::boxed_error;
use crate::goes::{GoesSatellite, parse_goes_abi_filename};

/// One listed S3 object.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct S3Object {
    pub key: String,
    pub size_bytes: u64,
    pub last_modified: String,
}

/// A downloaded (or cache-hit) object on local disk.
#[derive(Debug, Clone)]
pub struct DownloadedObject {
    pub object: S3Object,
    pub path: PathBuf,
    pub cache_hit: bool,
}

/// The ABI sectors the follow engine knows. Mesoscale sectors share the
/// `ABI-L2-CMIPM` S3 prefix; the filename product token (`CMIPM1`/`CMIPM2`)
/// disambiguates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sector {
    Conus,
    FullDisk,
    Meso1,
    Meso2,
}

impl Sector {
    pub fn parse(value: &str) -> Option<Self> {
        let normalized = value.trim().to_ascii_lowercase().replace(['-', ' '], "_");
        match normalized.as_str() {
            "conus" | "continental_us" | "c" => Some(Self::Conus),
            "full" | "full_disk" | "fulldisk" | "fd" | "f" => Some(Self::FullDisk),
            "meso1" | "mesoscale1" | "mesoscale_1" | "m1" => Some(Self::Meso1),
            "meso2" | "mesoscale2" | "mesoscale_2" | "m2" => Some(Self::Meso2),
            _ => None,
        }
    }

    /// The filename product token (what `parse_goes_abi_filename` reports).
    pub fn abi_product(self) -> &'static str {
        match self {
            Self::Conus => "ABI-L2-CMIPC",
            Self::FullDisk => "ABI-L2-CMIPF",
            Self::Meso1 => "ABI-L2-CMIPM1",
            Self::Meso2 => "ABI-L2-CMIPM2",
        }
    }

    pub fn slug(self) -> &'static str {
        match self {
            Self::Conus => "conus",
            Self::FullDisk => "fulldisk",
            Self::Meso1 => "meso1",
            Self::Meso2 => "meso2",
        }
    }

    /// Observed scan cadence, for poll scheduling.
    pub fn cadence_secs(self) -> u64 {
        match self {
            Self::Conus => 300,
            Self::FullDisk => 600,
            Self::Meso1 | Self::Meso2 => 60,
        }
    }

    /// Default poll interval per Recon: meso 15 s, CONUS 30 s, FD 60 s.
    pub fn default_poll_secs(self) -> u64 {
        match self {
            Self::Conus => 30,
            Self::FullDisk => 60,
            Self::Meso1 | Self::Meso2 => 15,
        }
    }
}

/// Map a satellite name to its open-data bucket.
pub fn bucket_for_satellite(satellite: &str) -> Result<String, Box<dyn Error>> {
    let normalized = satellite.trim().to_ascii_lowercase().replace('-', "");
    match normalized.as_str() {
        "g16" | "goes16" => Ok("noaa-goes16".to_string()),
        "g17" | "goes17" => Ok("noaa-goes17".to_string()),
        "g18" | "goes18" => Ok("noaa-goes18".to_string()),
        "g19" | "goes19" => Ok("noaa-goes19".to_string()),
        value if value.starts_with("noaagoes") => Ok(value.replacen("noaagoes", "noaa-goes", 1)),
        value if value.starts_with("noaa-goes") => Ok(value.to_string()),
        _ => Err(boxed_error(format!(
            "unsupported GOES satellite: {satellite}"
        ))),
    }
}

/// `ABI-L2-CMIPC/{year}/{doy:03}/{hour:02}/`, the hour directory prefix.
/// Mesoscale products share one prefix: `CMIPM1`/`CMIPM2` -> `CMIPM`.
pub fn goes_hour_prefix(product: &str, hour: DateTime<Utc>) -> String {
    let product = goes_s3_prefix_product(product);
    format!(
        "{}/{:04}/{:03}/{:02}/",
        product,
        hour.year(),
        hour.ordinal(),
        hour.hour()
    )
}

fn goes_s3_prefix_product(product: &str) -> String {
    let trimmed = product.trim();
    let upper = trimmed.to_ascii_uppercase();
    if upper.ends_with("M1") || upper.ends_with("M2") {
        trimmed[..trimmed.len().saturating_sub(1)].to_string()
    } else {
        trimmed.to_string()
    }
}

/// The band-specific hour prefix used for incremental polling, e.g.
/// `ABI-L2-CMIPC/2026/161/18/OR_ABI-L2-CMIPC-M6C13_G19_`. Keys under it
/// differ only in their s/e/c timestamps, so they sort chronologically and
/// `start-after` listing yields exact incremental diffs.
///
/// `mode` is the ABI scan mode token (6 since 2019; 3 is the contingency
/// mode): configurable so a mode flip degrades to a config change.
pub fn band_hour_prefix(
    abi_product: &str,
    satellite: &GoesSatellite,
    mode: u8,
    band: u8,
    hour: DateTime<Utc>,
) -> String {
    format!(
        "{}OR_{}-M{}C{:02}_{}_",
        goes_hour_prefix(abi_product, hour),
        abi_product.trim().to_ascii_uppercase(),
        mode,
        band,
        satellite.as_str()
    )
}

/// The filename prefix for channel-less single-layer L2 products (the
/// cloud suite: ACHA/ACM/ACTP/COD/CPS/CTP), e.g.
/// `ABI-L2-ACHAC/2026/216/18/OR_ABI-L2-ACHAC-M6_G19_`. The
/// [`band_hour_prefix`] twin for products whose filenames carry no
/// `C{band}` token; mesoscale directory merging (`ACHAM1` -> `ACHAM/`)
/// follows [`goes_hour_prefix`], with the sector digit kept in the
/// filename token exactly as the bucket lays keys out.
pub fn product_hour_prefix(
    abi_product: &str,
    satellite: &GoesSatellite,
    mode: u8,
    hour: DateTime<Utc>,
) -> String {
    format!(
        "{}OR_{}-M{}_{}_",
        goes_hour_prefix(abi_product, hour),
        abi_product.trim().to_ascii_uppercase(),
        mode,
        satellite.as_str()
    )
}

/// Whether a parsed filename product matches the requested one (a bare
/// `...M` request accepts both mesoscale sectors).
pub fn abi_filename_product_matches_request(
    actual_product: &str,
    requested_product: &str,
) -> bool {
    let actual = actual_product.trim().to_ascii_uppercase();
    let requested = requested_product.trim().to_ascii_uppercase();
    actual == requested
        || (requested.ends_with('M')
            && (actual == format!("{requested}1") || actual == format!("{requested}2")))
}

pub fn object_url(bucket: &str, key: &str) -> String {
    format!("https://{bucket}.s3.amazonaws.com/{key}")
}

pub fn object_filename(key: &str) -> &str {
    key.rsplit('/').next().unwrap_or(key)
}

/// List every object under `prefix`, following continuation tokens. When
/// `start_after` is given, only keys lexicographically greater are returned
/// (the incremental polling primitive).
pub fn list_s3_objects(
    agent: &ureq::Agent,
    bucket: &str,
    prefix: &str,
    start_after: Option<&str>,
) -> Result<Vec<S3Object>, Box<dyn Error>> {
    let mut objects = Vec::new();
    let mut token = None::<String>;
    loop {
        let mut url = format!(
            "https://{bucket}.s3.amazonaws.com/?list-type=2&prefix={}&max-keys=1000",
            url_query_encode(prefix)
        );
        match (&token, start_after) {
            // continuation-token supersedes start-after on follow-up pages.
            (Some(token), _) => {
                url.push_str("&continuation-token=");
                url.push_str(&url_query_encode(token));
            }
            (None, Some(after)) => {
                url.push_str("&start-after=");
                url.push_str(&url_query_encode(after));
            }
            (None, None) => {}
        }
        let mut response = rw_http::get(agent, &url)?;
        let xml = response.body_mut().read_to_string()?;
        let page = parse_s3_list_xml(&xml);
        objects.extend(page.objects);
        token = page.next_continuation_token;
        if token.is_none() {
            break;
        }
    }
    Ok(objects)
}

/// Local cache path of one object: `cache_dir/satellite/{bucket}/{key}`.
pub fn cached_object_path(cache_dir: &Path, bucket: &str, key: &str) -> PathBuf {
    cache_dir.join("satellite").join(bucket).join(key)
}

/// Download `object` into [`cached_object_path`] (atomic write); a cache
/// hit is an existing file with the exact listed byte size.
pub fn download_object(
    agent: &ureq::Agent,
    bucket: &str,
    cache_dir: &Path,
    object: &S3Object,
    use_cache: bool,
) -> Result<DownloadedObject, Box<dyn Error>> {
    let target = cached_object_path(cache_dir, bucket, &object.key);
    if use_cache && target.exists() && target.metadata()?.len() == object.size_bytes {
        return Ok(DownloadedObject {
            object: object.clone(),
            path: target,
            cache_hit: true,
        });
    }
    let url = object_url(bucket, &object.key);
    let mut response = rw_http::get(agent, &url)?;
    let limit = object
        .size_bytes
        .saturating_add(16 * 1024 * 1024)
        .max(32 * 1024 * 1024);
    let bytes = response
        .body_mut()
        .with_config()
        .limit(limit)
        .read_to_vec()?;
    if object.size_bytes > 0 && bytes.len() as u64 != object.size_bytes {
        return Err(boxed_error(format!(
            "downloaded byte count mismatch for {}: expected {}, got {}",
            object.key,
            object.size_bytes,
            bytes.len()
        )));
    }
    atomic_write_bytes(&target, &bytes)?;
    Ok(DownloadedObject {
        object: object.clone(),
        path: target,
        cache_hit: false,
    })
}

/// What one [`prune_object_cache`] pass removed.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CachePruneReport {
    pub removed_files: usize,
    pub removed_bytes: u64,
}

/// Best-effort sweep of the raw object cache under
/// `cache_dir/satellite/{bucket}`: every file whose scan start (the
/// filename `s` token, via [`parse_goes_abi_filename`]) (or modified time
/// when the name does not parse) is older than `cutoff` is deleted, and
/// directories left empty are removed (downloads recreate them). The cache
/// is exempt from the store's `enforce_window`, so without this sweep a
/// continuous follow grows disk without bound. Individual IO errors skip
/// the entry: cache housekeeping must never kill the poller.
pub fn prune_object_cache(
    cache_dir: &Path,
    bucket: &str,
    cutoff: DateTime<Utc>,
) -> CachePruneReport {
    let mut report = CachePruneReport::default();
    let root = cache_dir.join("satellite").join(bucket);
    let mut dirs = vec![root];
    let mut index = 0;
    while index < dirs.len() {
        let dir = dirs[index].clone();
        index += 1;
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let Ok(file_type) = entry.file_type() else {
                continue;
            };
            if file_type.is_dir() {
                dirs.push(path);
                continue;
            }
            if !file_type.is_file() {
                continue;
            }
            let age_reference = parse_goes_abi_filename(&path)
                .map(|parsed| parsed.start_time_utc)
                .ok()
                .or_else(|| {
                    entry
                        .metadata()
                        .ok()
                        .and_then(|meta| meta.modified().ok())
                        .map(DateTime::<Utc>::from)
                });
            let Some(when) = age_reference else {
                continue;
            };
            if when >= cutoff {
                continue;
            }
            let bytes = entry.metadata().map(|meta| meta.len()).unwrap_or(0);
            if std::fs::remove_file(&path).is_ok() {
                report.removed_files += 1;
                report.removed_bytes += bytes;
            }
        }
    }
    // Deepest-first so cascading empties collapse (`.../18` first, then a
    // now-empty `.../161`, and so on up to the bucket dir itself).
    for dir in dirs.iter().rev() {
        let _ = std::fs::remove_dir(dir); // refuses (keeps) non-empty dirs
    }
    report
}

struct S3ListPage {
    objects: Vec<S3Object>,
    next_continuation_token: Option<String>,
}

fn parse_s3_list_xml(xml: &str) -> S3ListPage {
    let mut objects = Vec::new();
    for contents in xml.split("<Contents>").skip(1) {
        let end = contents.find("</Contents>").unwrap_or(contents.len());
        let block = &contents[..end];
        let key = extract_xml_tag(block, "Key").unwrap_or_default();
        if key.is_empty() {
            continue;
        }
        let size_bytes = extract_xml_tag(block, "Size")
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(0);
        let last_modified = extract_xml_tag(block, "LastModified").unwrap_or_default();
        objects.push(S3Object {
            key,
            size_bytes,
            last_modified,
        });
    }
    S3ListPage {
        objects,
        next_continuation_token: extract_xml_tag(xml, "NextContinuationToken"),
    }
}

fn extract_xml_tag(xml: &str, tag: &str) -> Option<String> {
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");
    let start = xml.find(&open)? + open.len();
    let end = xml[start..].find(&close)? + start;
    Some(xml_unescape(&xml[start..end]))
}

fn xml_unescape(value: &str) -> String {
    value
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
}

fn url_query_encode(value: &str) -> String {
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

/// Max duration for DNS resolution and for establishing the TCP+TLS
/// connection. A half-open socket (sleep/resume, S3 hiccup) must surface
/// as an `Err` that feeds the follow loop's backoff, never hang the
/// 24/7 poller.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
/// Max duration for sending the request line + headers.
const SEND_REQUEST_TIMEOUT: Duration = Duration::from_secs(30);
/// Max duration for receiving the response headers.
const RECV_RESPONSE_TIMEOUT: Duration = Duration::from_secs(60);
/// Max duration for receiving a response body, generous enough for
/// full-disk objects on a slow link.
const RECV_BODY_TIMEOUT: Duration = Duration::from_secs(300);

/// Build the pure-Rust TLS agent. The OnceLock guards the process-global
/// provider install so repeated calls (parallel downloads) never clash.
///
/// Every phase gets an explicit timeout: ureq 3 defaults them all to
/// `None`, which would let one stalled connection block a poll (and the
/// whole follow session) forever.
pub fn build_agent() -> ureq::Agent {
    static CRYPTO_PROVIDER: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    CRYPTO_PROVIDER.get_or_init(|| {
        rustls::crypto::CryptoProvider::install_default(rustls_rustcrypto::provider()).ok();
    });
    let crypto = std::sync::Arc::new(rustls_rustcrypto::provider());
    ureq::Agent::config_builder()
        .timeout_resolve(Some(CONNECT_TIMEOUT))
        .timeout_connect(Some(CONNECT_TIMEOUT))
        .timeout_send_request(Some(SEND_REQUEST_TIMEOUT))
        .timeout_recv_response(Some(RECV_RESPONSE_TIMEOUT))
        .timeout_recv_body(Some(RECV_BODY_TIMEOUT))
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

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    #[test]
    fn s3_xml_parser_reads_continuation_token() {
        let xml = "<ListBucketResult><Contents><Key>a.nc</Key><Size>42</Size><LastModified>x</LastModified></Contents><NextContinuationToken>abc&amp;123</NextContinuationToken></ListBucketResult>";
        let page = parse_s3_list_xml(xml);
        assert_eq!(page.objects.len(), 1);
        assert_eq!(page.objects[0].key, "a.nc");
        assert_eq!(page.objects[0].size_bytes, 42);
        assert_eq!(page.next_continuation_token.as_deref(), Some("abc&123"));
    }

    #[test]
    fn hour_prefix_uses_day_of_year_and_merges_meso_sectors() {
        let hour = Utc.with_ymd_and_hms(2026, 6, 10, 18, 0, 0).unwrap();
        assert_eq!(
            goes_hour_prefix("ABI-L2-CMIPC", hour),
            "ABI-L2-CMIPC/2026/161/18/"
        );
        assert_eq!(
            goes_hour_prefix("ABI-L2-CMIPM1", hour),
            "ABI-L2-CMIPM/2026/161/18/"
        );
        assert_eq!(
            goes_hour_prefix("ABI-L2-CMIPM2", hour),
            "ABI-L2-CMIPM/2026/161/18/"
        );
    }

    #[test]
    fn band_hour_prefix_pins_product_mode_band_and_satellite() {
        let hour = Utc.with_ymd_and_hms(2026, 6, 10, 18, 0, 0).unwrap();
        assert_eq!(
            band_hour_prefix("ABI-L2-CMIPC", &GoesSatellite::G19, 6, 13, hour),
            "ABI-L2-CMIPC/2026/161/18/OR_ABI-L2-CMIPC-M6C13_G19_"
        );
        // Mesoscale: directory prefix is the shared CMIPM, the filename
        // token keeps the sector digit.
        assert_eq!(
            band_hour_prefix("ABI-L2-CMIPM1", &GoesSatellite::G18, 6, 2, hour),
            "ABI-L2-CMIPM/2026/161/18/OR_ABI-L2-CMIPM1-M6C02_G18_"
        );
    }

    #[test]
    fn sector_parse_and_products() {
        assert_eq!(Sector::parse("CONUS"), Some(Sector::Conus));
        assert_eq!(Sector::parse("m2"), Some(Sector::Meso2));
        assert_eq!(Sector::parse("full-disk"), Some(Sector::FullDisk));
        assert_eq!(Sector::parse("bogus"), None);
        assert_eq!(Sector::Meso1.abi_product(), "ABI-L2-CMIPM1");
        assert_eq!(Sector::Conus.abi_product(), "ABI-L2-CMIPC");
    }

    #[test]
    fn filename_product_matching_handles_bare_meso_requests() {
        assert!(abi_filename_product_matches_request(
            "ABI-L2-CMIPC",
            "ABI-L2-CMIPC"
        ));
        assert!(abi_filename_product_matches_request(
            "ABI-L2-CMIPM1",
            "ABI-L2-CMIPM"
        ));
        assert!(!abi_filename_product_matches_request(
            "ABI-L2-CMIPM2",
            "ABI-L2-CMIPM1"
        ));
    }

    #[test]
    fn bucket_mapping_accepts_aliases() {
        assert_eq!(bucket_for_satellite("goes19").unwrap(), "noaa-goes19");
        assert_eq!(bucket_for_satellite("G18").unwrap(), "noaa-goes18");
        assert_eq!(bucket_for_satellite("noaa-goes16").unwrap(), "noaa-goes16");
        assert!(bucket_for_satellite("himawari").is_err());
    }

    #[test]
    fn query_encoding_escapes_reserved_bytes() {
        assert_eq!(url_query_encode("a/b c+d"), "a%2Fb%20c%2Bd");
        assert_eq!(url_query_encode("OR_ABI-L2"), "OR_ABI-L2");
    }

    #[test]
    fn cache_prune_removes_stale_objects_and_empty_dirs() {
        let dir = std::env::temp_dir().join(format!(
            "rw-sat-cache-prune-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        let bucket = "noaa-goes19";
        let old_key = "ABI-L2-CMIPC/2026/161/18/OR_ABI-L2-CMIPC-M6C13_G19_s20261611851176_e20261611853549_c20261611854020.nc";
        let new_key = "ABI-L2-CMIPC/2026/161/19/OR_ABI-L2-CMIPC-M6C13_G19_s20261611901176_e20261611903549_c20261611904020.nc";
        for key in [old_key, new_key] {
            let path = cached_object_path(&dir, bucket, key);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, b"netcdf bytes").unwrap();
        }

        // Cutoff between the two scan starts: 18:51 goes, 19:01 stays.
        let cutoff = Utc.with_ymd_and_hms(2026, 6, 10, 18, 55, 0).unwrap();
        let report = prune_object_cache(&dir, bucket, cutoff);
        assert_eq!(report.removed_files, 1);
        assert_eq!(report.removed_bytes, 12);
        assert!(!cached_object_path(&dir, bucket, old_key).exists());
        assert!(
            !cached_object_path(&dir, bucket, old_key)
                .parent()
                .unwrap()
                .exists(),
            "emptied hour dir removed"
        );
        assert!(cached_object_path(&dir, bucket, new_key).is_file());

        // Second pass is a no-op; a missing cache root never errors.
        assert_eq!(prune_object_cache(&dir, bucket, cutoff), CachePruneReport::default());
        assert_eq!(
            prune_object_cache(&dir.join("absent"), bucket, cutoff),
            CachePruneReport::default()
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn cache_prune_falls_back_to_mtime_for_unparseable_names() {
        let dir = std::env::temp_dir().join(format!(
            "rw-sat-cache-prune-mtime-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        let bucket = "noaa-goes19";
        let path = cached_object_path(&dir, bucket, "leftover.partial");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, b"junk").unwrap();

        // Fresh mtime survives a cutoff in the past...
        let kept = prune_object_cache(&dir, bucket, Utc::now() - chrono::Duration::hours(1));
        assert_eq!(kept, CachePruneReport::default());
        assert!(path.is_file());
        // ... and falls to a cutoff beyond it.
        let removed = prune_object_cache(&dir, bucket, Utc::now() + chrono::Duration::hours(1));
        assert_eq!(removed.removed_files, 1);
        assert!(!path.exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn agent_has_every_io_phase_timeout_set() {
        let timeouts = build_agent().config().timeouts();
        assert_eq!(timeouts.resolve, Some(CONNECT_TIMEOUT));
        assert_eq!(timeouts.connect, Some(CONNECT_TIMEOUT));
        assert_eq!(timeouts.send_request, Some(SEND_REQUEST_TIMEOUT));
        assert_eq!(timeouts.recv_response, Some(RECV_RESPONSE_TIMEOUT));
        assert_eq!(
            timeouts.recv_body,
            Some(RECV_BODY_TIMEOUT),
            "body timeout must cover full-disk objects"
        );
    }
}
