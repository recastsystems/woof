//! Anonymous AWS S3 access to the `noaa-nexrad-level2` open-data bucket
//! over plain HTTPS.
//!
//! This is the `rw-sat`/`rw-glm` `ListObjectsV2` pattern (paginated with
//! continuation tokens, query-encoded prefixes, XML unescaping, pure-Rust
//! rustls) retold for a bucket whose key space is a day directory per site
//! rather than an hour directory per product:
//!
//! ```text
//! {YYYY}/{MM}/{DD}/{SITE}/{SITE}{YYYYMMDD}_{HHMMSS}_V06
//! ```
//!
//! There is no hour level, so a `(site, time window)` request expands to
//! one prefix per UTC day it touches and the volume timestamp is read back
//! out of the object name.  No SDK, no credentials.

use std::error::Error;
use std::io;
use std::path::{Component, Path, PathBuf};
use std::time::Duration;

use chrono::{DateTime, Datelike, NaiveDate, NaiveDateTime, TimeZone, Timelike, Utc};

use rw_store::atomic::atomic_write_bytes;

/// The NOAA open-data bucket that is the *archive of record* for WSR-88D
/// Level-II: nominally complete, and not the default.
///
/// It is not the default because it does not currently work anonymously.
/// Probed twice on 2026-07-30, hours apart, from a host on which
/// [`DEFAULT_BUCKET`] answered 200 in the same second:
///
/// ```text
/// ListObjectsV2  HTTP 403 AccessDenied
/// GET one object HTTP 403
/// ```
///
/// So it grants neither `s3:ListBucket` nor `s3:GetObject` to an anonymous
/// caller, and a `list`, `fetch` or decode against it fails at the first
/// request.  That is a *capability* statement about the endpoint, not a
/// judgement about which bucket is authoritative: this one still is, and
/// the moment it grants anonymous access again it is the better choice.
/// It stays selectable with `--bucket` for exactly that reason, and
/// [`anonymous_access_refused`] names the capability rather than guessing
/// when a request comes back 403.
pub const ARCHIVE_OF_RECORD_BUCKET: &str = "noaa-nexrad-level2";

/// Unidata's mirror of the same key space, and the default.
///
/// The default is chosen by capability, checked at the endpoint rather
/// than assumed from which organisation runs it: on 2026-07-30 this bucket
/// answered anonymous `ListObjectsV2` with 200 and an anonymous ranged
/// `GET` with 206, over the whole span the pipeline reads (a 2011 `.gz`
/// key, a 2013 `.gz` key, and 2017 through 2026 plain keys) while
/// [`ARCHIVE_OF_RECORD_BUCKET`] answered 403 to both.
///
/// It is the same layout and the same objects, so this is a `--bucket`
/// value and not a code path.  Its coverage is *not* identical: it is the
/// shallower of the two and a window it cannot serve should be retried
/// against the archive of record with credentials.
pub const DEFAULT_BUCKET: &str = "unidata-nexrad-level2";

/// Unidata's **real-time** bucket: the same radars, published one LDM
/// chunk at a time while the antenna is still turning, instead of one
/// Archive-II file per finished volume.
///
/// This is a different key space, not a different `--bucket` value for the
/// same one, `{SITE}/{VOLUME_ID}/{YYYYMMDD}-{HHMMSS}-{NNN}-{S|I|E}`, so
/// it is the default for the `live-list`/`live-fetch` subcommands and is
/// never the default for `list`/`fetch`.  Measured against the S3 `Date`
/// header on 2026-08-05: the newest chunk was 0–4 s old on every poll,
/// while the assembled volume file in [`DEFAULT_BUCKET`] only exists once
/// the volume ends, so its age at a random poll is half the volume period
/// and its worst case is the whole period.
pub const LIVE_DEFAULT_BUCKET: &str = "unidata-nexrad-level2-chunks";


/// Bucket names are data, but a malformed one would build a URL that
/// resolves somewhere unintended, so it is validated as a DNS label set.
pub fn normalize_bucket(bucket: &str) -> Result<String, Box<dyn Error>> {
    let name = bucket.trim().to_ascii_lowercase();
    let well_formed = (3..=63).contains(&name.len())
        && name
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-' || c == '.')
        && !name.starts_with(['-', '.'])
        && !name.ends_with(['-', '.']);
    if well_formed {
        Ok(name)
    } else {
        Err(boxed_error(format!("malformed S3 bucket name {bucket:?}")))
    }
}

/// One listed S3 object.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct S3Object {
    pub key: String,
    pub size_bytes: u64,
    pub last_modified: String,
}

/// A listed object that parsed as a Level-II volume for a known site.
#[derive(Debug, Clone)]
pub struct VolumeObject {
    pub object: S3Object,
    pub site: String,
    pub valid_time: DateTime<Utc>,
    /// Trailing volume-format token, e.g. `V06`.  Never includes the `.gz`
    /// suffix; that is [`VolumeObject::gzipped`].
    pub format: String,
    /// True when the archived key ends `.gz`.  Everything before roughly
    /// 2016 is stored that way and everything after is not.
    pub gzipped: bool,
}

/// What [`parse_volume_key`] read out of an object name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VolumeKey {
    pub site: String,
    pub valid_time: DateTime<Utc>,
    pub format: String,
    pub gzipped: bool,
}

/// A downloaded (or cache-hit) volume on local disk.
#[derive(Debug, Clone)]
pub struct DownloadedVolume {
    pub volume: VolumeObject,
    pub path: PathBuf,
    pub cache_hit: bool,
    pub sha256: String,
}

/// Reject anything that is not a four-character alphanumeric ICAO-style id.
/// Site ids are *data* (the bin never branches on a particular one) but a
/// malformed id would silently list an empty prefix, so it fails closed.
pub fn normalize_site(site: &str) -> Result<String, Box<dyn Error>> {
    let upper = site.trim().to_ascii_uppercase();
    if upper.len() == 4 && upper.chars().all(|c| c.is_ascii_alphanumeric()) {
        Ok(upper)
    } else {
        Err(boxed_error(format!(
            "radar site id must be four alphanumeric characters, got {site:?}"
        )))
    }
}

/// `{YYYY}/{MM}/{DD}/{SITE}/`: the day directory prefix for one site.
pub fn day_prefix(site: &str, day: NaiveDate) -> String {
    format!(
        "{:04}/{:02}/{:02}/{}/",
        day.year(),
        day.month(),
        day.day(),
        site
    )
}

/// Every UTC day the inclusive window `[start, end]` touches.
pub fn window_days(start: DateTime<Utc>, end: DateTime<Utc>) -> Vec<NaiveDate> {
    let mut days = Vec::new();
    let mut day = start.date_naive();
    let last = end.date_naive();
    while day <= last {
        days.push(day);
        match day.succ_opt() {
            Some(next) => day = next,
            None => break,
        }
        // A pathological window can never exhaust the calendar, but the
        // loop must be bounded regardless of what the caller passed.
        if days.len() > 400 {
            break;
        }
    }
    days
}

/// Parse `KTLX20230520_200356_V06`, or its gzipped pre-2016 spelling
/// `KTLX20130520_195111_V06.gz`, out of an object key.
///
/// Returns `None` for anything that does not look like a Level-II volume
/// name: the listing is data, so an unrecognised sibling object is skipped
/// rather than fataled, but it is never counted as a volume either.
///
/// **The `.gz` suffix is a storage detail, not a different product.**  The
/// archive holds roughly 2011 through 2016 gzip-wrapped and everything
/// after it plain, and this used to reject the whole gzipped era because
/// the format token had to be alphanumeric and `V06.gz` is not.  That
/// silently removed every pre-2016 volume from any window a caller asked
/// for -- the listing simply came back shorter, with no error and nothing
/// in provenance.  `..._V06_MDM` (a metadata sidecar) and `..._V08.001` (a
/// chunk remnant) are still refused: they are not volumes, and a `.gz`
/// suffix is accepted only as a whole trailing `.gz` on an otherwise
/// alphanumeric token.
pub fn parse_volume_key(key: &str) -> Option<VolumeKey> {
    let name = key.rsplit('/').next()?;
    // `SSSS` + `YYYYMMDD` + `_` + `HHMMSS` + `_` + `FMT` [+ `.gz`]
    if name.len() < 20 {
        return None;
    }
    // `str::len` is a BYTE count and every slice below is at a fixed BYTE
    // offset, so a name whose bytes are not one-to-one with its characters
    // makes `&name[..4]` land inside a character and PANIC -- not return
    // None.  S3 keys are arbitrary UTF-8 and nothing upstream restricts the
    // character set: `parse_s3_list_xml_with` checks the prefix, the
    // trailing `/` and `key_localization_fault` (backslash, colon, leading
    // `/`, empty/`.`/`..` segments), none of which mentions encoding.  One
    // junk sibling object would therefore kill the whole listing -- and with
    // it every volume in the window -- where this function's own contract
    // three paragraphs up says an unrecognised sibling is SKIPPED.  A
    // Level-II volume name is ASCII by construction, so the refusal is
    // exact rather than defensive.
    if !name.is_ascii() {
        return None;
    }
    let site = &name[..4];
    if !site.chars().all(|c| c.is_ascii_alphanumeric()) {
        return None;
    }
    let stamp = &name[4..12];
    if !stamp.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    if name.as_bytes()[12] != b'_' {
        return None;
    }
    let clock = &name[13..19];
    if !clock.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    if name.as_bytes()[19] != b'_' {
        return None;
    }
    let tail = &name[20..];
    let (format, gzipped) = match tail.strip_suffix(".gz") {
        Some(stem) => (stem, true),
        None => (tail, false),
    };
    // `V06` / `V03` and nothing else: `..._V06_MDM` is a per-volume
    // metadata sidecar and `..._V08.001` is a chunk remnant on the rolling
    // mirror.  Neither is a volume, and both would otherwise be counted as
    // one and then fail at decode.
    if format.is_empty() || !format.chars().all(|c| c.is_ascii_alphanumeric()) {
        return None;
    }
    let naive = NaiveDateTime::parse_from_str(
        &format!("{stamp}{clock}"),
        "%Y%m%d%H%M%S",
    )
    .ok()?;
    Some(VolumeKey {
        site: site.to_string(),
        valid_time: Utc.from_utc_datetime(&naive),
        format: format.to_string(),
        gzipped,
    })
}

/// List every Level-II volume for `site` whose scan start falls inside the
/// inclusive window, sorted by valid time.
/// [`list_volumes_observed`] returns the roster and the bucket's own clock
/// when it answered.
///
/// The clock is here so a receipt can state how far behind the sky this
/// route is *in the same terms* the real-time route states its own lag: the
/// age of the newest object, measured against the clock of the machine that
/// stamped it.  A lag computed against this host's clock measures the skew
/// as well as the feed, and the two are not separable afterwards.
pub fn list_volumes_observed(
    agent: &ureq::Agent,
    bucket: &str,
    site: &str,
    start: DateTime<Utc>,
    end: DateTime<Utc>,
) -> Result<(Vec<VolumeObject>, Option<DateTime<Utc>>), Box<dyn Error>> {
    if end < start {
        return Err(boxed_error(format!(
            "time window ends before it starts: {} .. {}",
            iso8601(start),
            iso8601(end)
        )));
    }
    let mut volumes = Vec::new();
    let mut observed_at = None;
    for day in window_days(start, end) {
        let prefix = day_prefix(site, day);
        let listing = list_s3(agent, ListRequest::new(bucket, &prefix))?;
        if observed_at.is_none() {
            observed_at = listing.server_date;
        }
        for object in listing.objects {
            let Some(parsed) = parse_volume_key(&object.key) else {
                continue;
            };
            if parsed.site != site {
                continue;
            }
            if parsed.valid_time < start || parsed.valid_time > end {
                continue;
            }
            volumes.push(VolumeObject {
                object,
                site: parsed.site,
                valid_time: parsed.valid_time,
                format: parsed.format,
                gzipped: parsed.gzipped,
            });
        }
    }
    volumes.sort_by(|a, b| {
        a.valid_time
            .cmp(&b.valid_time)
            .then_with(|| a.object.key.cmp(&b.object.key))
    });
    Ok((volumes, observed_at))
}

/// Hard ceiling on pages followed for one prefix.
///
/// At the 1000 keys per page this client asks for, that is ten million
/// objects under a single day directory for one radar: six orders of
/// magnitude past the ~290 a busy site actually writes.  It is not a
/// pagination policy, it is the guarantee that a bucket which keeps handing
/// back fresh tokens forever cannot hang the CLI in a loop that looks like
/// progress.
const MAX_LIST_PAGES: usize = 10_000;

/// What one [`list_s3`] call asked the bucket for.
///
/// The archive route wants every object under a day directory and nothing
/// else, which is [`ListRequest::new`].  The live route wants one other
/// shape, and it is *deliberate* rather than incidental, so it is stated
/// here rather than inferred from the response: a **delimited** listing,
/// which rolls the volume-id directories up into `<CommonPrefixes>` and
/// returns no objects at all.  A rolled-up page is otherwise a refusal
/// ([`parse_s3_list_xml`]) precisely because it hides objects; asking for
/// one and then reading the roll-up is a different act from being handed
/// one nobody asked for, and only the client that sent the delimiter can
/// tell them apart.
///
/// A delimited request still follows every page: the roll-up is the roster
/// it wants, and a short roll-up hides volume ids exactly as a short object
/// page hides volumes.
#[derive(Debug, Clone)]
pub struct ListRequest<'a> {
    bucket: &'a str,
    prefix: &'a str,
    delimiter: Option<&'a str>,
}

impl<'a> ListRequest<'a> {
    /// Every object under `prefix`, all pages, no roll-up.
    pub fn new(bucket: &'a str, prefix: &'a str) -> Self {
        Self {
            bucket,
            prefix,
            delimiter: None,
        }
    }

    pub fn delimiter(mut self, delimiter: &'a str) -> Self {
        self.delimiter = Some(delimiter);
        self
    }
}

/// What a listing returned, including the two facts `Vec<S3Object>` cannot
/// carry: the roll-up prefixes of a delimited request, and the bucket's own
/// clock.
///
/// `server_date` is the `Date` response header, and it is here because the
/// only accurate way to state how old an object is, is against the clock of
/// the machine that stamped it.  This host's clock is not that clock, and a
/// lag receipt computed against a skewed local clock reports the skew as
/// freshness.
#[derive(Debug, Clone)]
pub struct S3Listing {
    pub objects: Vec<S3Object>,
    pub common_prefixes: Vec<String>,
    pub server_date: Option<DateTime<Utc>>,
}

/// Run one [`ListRequest`], following continuation tokens to the end.
///
/// **Every page must prove it is complete before any of it is used.**  This
/// function's result is the roster a volume selection is made from, and a
/// short roster is indistinguishable from a quiet day: the caller sees
/// fewer volumes and no error.  So a page that does not say whether it was
/// truncated, says it was truncated without handing over the token that
/// continues it, contradicts its own `KeyCount`, or returns a token that
/// does not advance, is a refusal of the whole listing rather than a
/// partial answer.  See [`parse_s3_list_xml`] for the per-page contract.
pub fn list_s3(agent: &ureq::Agent, request: ListRequest<'_>) -> Result<S3Listing, Box<dyn Error>> {
    let ListRequest {
        bucket,
        prefix,
        delimiter,
    } = request;
    let mut objects = Vec::new();
    let mut common_prefixes = Vec::new();
    let mut server_date = None;
    let mut token = None::<String>;
    let mut spent_tokens: Vec<String> = Vec::new();
    let mut pages = 0usize;
    loop {
        let mut url = format!(
            "https://{bucket}.s3.amazonaws.com/?list-type=2&prefix={}&max-keys=1000",
            url_query_encode(prefix)
        );
        if let Some(delimiter) = delimiter {
            url.push_str("&delimiter=");
            url.push_str(&url_query_encode(delimiter));
        }
        if let Some(token) = &token {
            url.push_str("&continuation-token=");
            url.push_str(&url_query_encode(token));
        }
        let mut response = rw_http::get(agent, &url)
            .map_err(|err| describe_request_failure(err, bucket, prefix))?;
        if server_date.is_none() {
            server_date = response
                .headers()
                .get("date")
                .and_then(|value| value.to_str().ok())
                .and_then(parse_http_date);
        }
        let xml = response.body_mut().read_to_string()?;
        // Named at the call site rather than folded into one entry point:
        // the two are different contracts, and which one a page is being
        // held to is decided by what this client sent, not by what came
        // back.
        let mut page = match delimiter {
            None => parse_s3_list_xml(&xml, bucket, prefix)?,
            Some(delimiter) => parse_s3_list_xml_with(&xml, bucket, prefix, Some(delimiter))?,
        };
        objects.append(&mut page.objects);
        common_prefixes.append(&mut page.common_prefixes);
        pages += 1;
        token = advance_continuation(token.as_deref(), &page, &mut spent_tokens, pages)?;
        if token.is_none() {
            break;
        }
    }
    Ok(S3Listing {
        objects,
        common_prefixes,
        server_date,
    })
}

/// Turn a failed listing request into a statement about what the endpoint
/// will and will not do for an anonymous caller.
///
/// A bare `HTTP 403` at a CLI's exit is indistinguishable from a typo in
/// the site id or a bad window.  It is neither: it is the bucket declining
/// to grant `s3:ListBucket` without credentials, which is a property of
/// the endpoint that no argument the user changes will fix, and the remedy
/// is a different `--bucket`.  Named here rather than assumed from the
/// bucket's identity -- the check is what the endpoint answered, so if the
/// archive of record starts granting anonymous access again, nothing here
/// has to be edited for it to work.
pub fn anonymous_access_refused(bucket: &str, prefix: &str, status: u16) -> Box<dyn Error> {
    let alternative = if bucket == DEFAULT_BUCKET {
        ARCHIVE_OF_RECORD_BUCKET
    } else {
        DEFAULT_BUCKET
    };
    boxed_error(format!(
        "s3://{bucket}/ answered HTTP {status} to an anonymous ListObjectsV2 for {prefix:?}. \
         The bucket is not granting s3:ListBucket without credentials, so no site, window or \
         retry will change the answer. Try --bucket {alternative}, or supply credentials out \
         of band. (On 2026-07-30 {ARCHIVE_OF_RECORD_BUCKET} refused anonymous list *and* get, \
         and {DEFAULT_BUCKET} served both, which is why the default is what it is.)"
    ))
}

fn describe_request_failure(
    error: ureq::Error,
    bucket: &str,
    prefix: &str,
) -> Box<dyn Error> {
    match error {
        ureq::Error::StatusCode(status @ (401 | 403)) => {
            anonymous_access_refused(bucket, prefix, status)
        }
        ureq::Error::StatusCode(status) => boxed_error(format!(
            "s3://{bucket}/ answered HTTP {status} to a ListObjectsV2 for {prefix:?}"
        )),
        other => Box::new(other),
    }
}

/// The token the next request must carry, or `None` when the listing is
/// complete.
///
/// Split out from [`list_s3_objects`] so the pagination rules can be tested
/// without a socket.  Three of them:
///
/// * a page that is not truncated ends the walk, [`parse_s3_list_xml`] has
///   already refused the contradictory combinations, so `is_truncated` and
///   the presence of a token agree by the time this sees them;
/// * the token must **advance**.  Handing back the token the request was
///   made with, or one already spent earlier in this walk, means the next
///   request re-reads a page this walk has already read.  Following it
///   duplicates objects forever and never terminates; ignoring it silently
///   truncates the roster.  Both are refusals.
/// * the walk is bounded by [`MAX_LIST_PAGES`].
fn advance_continuation(
    used: Option<&str>,
    page: &S3ListPage,
    spent: &mut Vec<String>,
    pages_read: usize,
) -> Result<Option<String>, Box<dyn Error>> {
    let Some(next) = page.next_continuation_token.clone() else {
        return Ok(None);
    };
    if Some(next.as_str()) == used {
        return Err(boxed_error(format!(
            "S3 listing did not advance: page {pages_read} was fetched with \
             continuation token {next:?} and handed back the same token. \
             Following it re-reads the page just read; ignoring it drops \
             every object after this page from the roster"
        )));
    }
    if spent.iter().any(|seen| seen == &next) {
        return Err(boxed_error(format!(
            "S3 listing looped: page {pages_read} handed back continuation \
             token {next:?}, which this walk has already spent. The listing \
             is revisiting pages, so no roster it produces is complete"
        )));
    }
    if pages_read >= MAX_LIST_PAGES {
        return Err(boxed_error(format!(
            "S3 listing ran past {MAX_LIST_PAGES} pages and is still \
             truncated; refusing rather than paginating without end"
        )));
    }
    spent.push(next.clone());
    Ok(Some(next))
}

pub fn object_url(bucket: &str, key: &str) -> String {
    format!("https://{bucket}.s3.amazonaws.com/{key}")
}

pub fn object_filename(key: &str) -> &str {
    key.rsplit('/').next().unwrap_or(key)
}

/// Why `key` is not a name this client will turn into a local path, or
/// `None` when every one of its `/`-separated segments is an ordinary name.
///
/// A listed key is *data* -- nothing here branches on a particular one --
/// but it is not only data, because [`cached_volume_path`] **localizes** it:
/// it pushes each segment onto the cache root in turn, and `Path::push`
/// accepts `..` exactly as readily as `KTLX`.  RV8-PATH is that sentence
/// measured end to end.  The key
///
/// ```text
/// 2026/07/29/KTLX/../../../../../../../../elsewhere/KTLX20260729_000234_V06
/// ```
///
/// starts with the requested prefix, does not end in `/`, and parses as a
/// KTLX volume out of its last segment, so every question the roster asked
/// of it answered yes -- and the bytes it downloads land wherever the `..`
/// segments walk to, outside the cache root entirely, or in another bucket's
/// slot inside it.  No forged response is needed: S3 keys may legally
/// contain `..`, so this is available to anyone who can PUT into the listed
/// bucket, and the listed bucket is a public mirror.
///
/// This is a **pre-existing** defect, byte-identical at the merge target and
/// untouched by this branch; it is closed here because this branch is what
/// carries the reader into the integration branch.
///
/// The refusal is deliberate and the sanitisation is deliberately absent.
/// Stripping the `..` segments would invent a key S3 never listed and then
/// download something under it; refusing says which key was refused and
/// why, which is the doctrine every other check in this file follows.
fn key_localization_fault(key: &str) -> Option<&'static str> {
    if key.contains('\\') {
        return Some(
            "carries a backslash, which is a path separator on the platform \
             this client caches on",
        );
    }
    if key.contains(':') {
        return Some(
            "carries a colon, which names a drive or an alternate data \
             stream on the platform this client caches on",
        );
    }
    if key.starts_with('/') {
        return Some(
            "begins with '/', which localizes to a path from the filesystem \
             root rather than to one under the cache",
        );
    }
    key.split('/').find_map(|segment| match segment {
        "" => Some("carries an empty path segment"),
        "." => Some("carries a '.' path segment"),
        ".." => Some("carries a '..' path segment, which walks back out of the cache directory"),
        _ => None,
    })
}

/// Local cache path of one volume: `cache_dir/nexrad/{bucket}/{key}`, or a
/// refusal when that is not a path under `cache_dir/nexrad/{bucket}`.
///
/// The second half of RV8-PATH, and the reason it is here as well as in
/// [`parse_s3_list_xml`]: the boundary check refuses a traversing key at the
/// one door it currently comes through, and this refuses the write for any
/// caller that ever reaches this function by another route.  The test is
/// structural rather than textual -- strip the root off the built path and
/// require every remaining component to be a plain name -- so it holds for
/// the spellings a string check would miss, including a segment carrying a
/// drive prefix, which `Path::push` does not append but SUBSTITUTES for the
/// whole path.
pub fn cached_volume_path(
    cache_dir: &Path,
    bucket: &str,
    key: &str,
) -> Result<PathBuf, Box<dyn Error>> {
    let root = cache_dir.join("nexrad").join(bucket);
    let mut path = root.clone();
    for segment in key.split('/') {
        path.push(segment);
    }
    // Windows separators and drive/stream syntax must remain refusals when
    // a cache is prepared on Unix and later used on Windows.
    let portable_key = !key.contains('\\') && !key.contains(':');
    let under_root = portable_key && path.strip_prefix(&root).is_ok_and(|tail| {
        tail.components()
            .all(|component| matches!(component, Component::Normal(_)))
    });
    if !under_root {
        return Err(boxed_error(format!(
            "S3 key {key:?} does not name a file under the cache directory \
             for s3://{bucket}/ ({}); a key is localized one segment at a \
             time, so a key whose segments walk back out of that directory \
             would put bucket bytes somewhere this client never offered to \
             manage, and nothing about the listing that carried it would say \
             so",
            root.display()
        )));
    }
    Ok(path)
}

/// Download `volume` into [`cached_volume_path`] with an atomic
/// temp+rename, then hard-copy it to `out_dir` (also atomically) when one
/// is given.  A cache hit is an existing file with the exact listed byte
/// size; the sha256 is always recomputed from the bytes on disk, so a
/// truncated cache entry cannot pass itself off as a download.
pub fn download_volume(
    agent: &ureq::Agent,
    bucket: &str,
    cache_dir: &Path,
    volume: &VolumeObject,
    use_cache: bool,
) -> Result<DownloadedVolume, Box<dyn Error>> {
    let cached = download_object(agent, bucket, cache_dir, &volume.object, use_cache)?;
    Ok(DownloadedVolume {
        volume: volume.clone(),
        sha256: cached.sha256,
        path: cached.path,
        cache_hit: cached.cache_hit,
    })
}

/// One listed object on local disk, with the bytes that were written.
#[derive(Debug, Clone)]
pub struct CachedObject {
    pub path: PathBuf,
    pub bytes: Vec<u8>,
    pub cache_hit: bool,
    pub sha256: String,
}

/// Download any listed object into the cache with the same rules
/// [`download_volume`] applies to a whole volume: a cache hit is an existing
/// file of the exact listed byte size, a short download is a refusal rather
/// than a shorter file, and the digest is recomputed from the bytes on disk.
///
/// Split out from [`download_volume`] because the real-time feed's unit of
/// acquisition is a chunk rather than a volume, and a chunk that is quietly
/// short is a hole in the middle of an assembled volume: the one failure
/// the assembler cannot see afterwards, because a concatenation of the wrong
/// bytes is still a concatenation.
pub fn download_object(
    agent: &ureq::Agent,
    bucket: &str,
    cache_dir: &Path,
    object: &S3Object,
    use_cache: bool,
) -> Result<CachedObject, Box<dyn Error>> {
    let target = cached_volume_path(cache_dir, bucket, &object.key)?;
    if use_cache && target.is_file() && target.metadata()?.len() == object.size_bytes {
        let bytes = std::fs::read(&target)?;
        return Ok(CachedObject {
            sha256: hex_sha256(&bytes),
            bytes,
            path: target,
            cache_hit: true,
        });
    }
    let url = object_url(bucket, &object.key);
    let mut response = rw_http::get(agent, &url)?;
    let limit = object
        .size_bytes
        .saturating_add(16 * 1024 * 1024)
        .max(64 * 1024 * 1024);
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
    Ok(CachedObject {
        sha256: hex_sha256(&bytes),
        bytes,
        path: target,
        cache_hit: false,
    })
}

/// Publish a cached volume into a caller-owned directory, atomically.
///
/// The out directory gets the key's last segment and nothing else, and that
/// is checked rather than assumed: `object_filename` is a string operation
/// on listed data, and a last segment of `..` -- or one carrying a path
/// separator this platform honours -- is a write outside the directory the
/// caller named.  Same rule as [`cached_volume_path`], one directory level
/// deep instead of many.
pub fn publish_volume(source: &Path, out_dir: &Path, key: &str) -> Result<PathBuf, Box<dyn Error>> {
    let name = object_filename(key);
    let target = out_dir.join(name);
    let portable_name = !name.contains('\\') && !name.contains(':');
    let one_name_under_out_dir = portable_name && target.strip_prefix(out_dir).is_ok_and(|tail| {
        let mut components = tail.components();
        matches!(components.next(), Some(Component::Normal(_))) && components.next().is_none()
    });
    if !one_name_under_out_dir {
        return Err(boxed_error(format!(
            "S3 key {key:?} names {name:?} as its object file name, which is \
             not a file directly inside the requested output directory \
             ({}); a published volume is the listed object under the name \
             the listing gave it, and nowhere else",
            out_dir.display()
        )));
    }
    let bytes = std::fs::read(source)?;
    atomic_write_bytes(&target, &bytes)?;
    Ok(target)
}

pub fn hex_sha256(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let digest = hasher.finalize();
    let mut out = String::with_capacity(64);
    for byte in digest {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

/// RFC3339 with a literal `Z` and no fractional seconds: the one time
/// spelling every receipt and the NetCDF `valid_time` attribute share.
pub fn iso8601(when: DateTime<Utc>) -> String {
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        when.year(),
        when.month(),
        when.day(),
        when.hour(),
        when.minute(),
        when.second()
    )
}

/// RFC3339 with a literal `Z` and exactly three fractional digits
/// (`2026-09-19T12:02:36.123Z`): the spelling of a radial collection
/// instant, which Message 31 records to the millisecond.
///
/// A second spelling rather than a change to [`iso8601`], because that one
/// is shared with every receipt and every `valid_time` attribute already on
/// disk; widening it would move the bytes of files that are pinned by
/// digest.  The two are distinguishable by length alone (20 and 24).
pub fn iso8601_ms(when: DateTime<Utc>) -> String {
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}.{:03}Z",
        when.year(),
        when.month(),
        when.day(),
        when.hour(),
        when.minute(),
        when.second(),
        when.timestamp_subsec_millis()
    )
}

/// Read back exactly what [`iso8601_ms`] wrote, or what [`iso8601`] wrote.
///
/// Only those two spellings, because the verifier that calls this is
/// checking a writer that has exactly those two, and a looser parser would
/// let a pack say `12:02:36+02:00` and be read as UTC.
pub fn parse_iso8601_ms(value: &str) -> Result<DateTime<Utc>, Box<dyn Error>> {
    let trimmed = value.trim();
    for format in ["%Y-%m-%dT%H:%M:%S%.3fZ", "%Y-%m-%dT%H:%M:%SZ"] {
        if let Ok(naive) = NaiveDateTime::parse_from_str(trimmed, format) {
            return Ok(Utc.from_utc_datetime(&naive));
        }
    }
    Err(boxed_error(format!(
        "unparseable collection instant {value:?}: expected 2026-09-19T12:02:36.123Z"
    )))
}

/// Parse the two time spellings the CLI accepts: full ISO8601 with a `Z`
/// (`2023-05-20T20:00:00Z`) and the compact stamp (`20230520T200000`).
pub fn parse_time(value: &str) -> Result<DateTime<Utc>, Box<dyn Error>> {
    let trimmed = value.trim();
    for format in ["%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S", "%Y%m%dT%H%M%S"] {
        if let Ok(naive) = NaiveDateTime::parse_from_str(trimmed, format) {
            return Ok(Utc.from_utc_datetime(&naive));
        }
    }
    Err(boxed_error(format!(
        "unparseable time {value:?}: expected 2023-05-20T20:00:00Z or 20230520T200000"
    )))
}

/// Parse an HTTP `Date` header (RFC 9110 IMF-fixdate), the bucket's own
/// clock at the instant it answered.
///
/// Only the fixed-length GMT spelling is accepted: the two obsolete
/// formats RFC 9110 still tolerates carry a two-digit year, and guessing a
/// century for a timestamp a lag receipt is computed from is a fabrication.
/// A header this cannot read is `None`, and every caller states the lag as
/// unmeasured rather than substituting the local clock.
pub fn parse_http_date(value: &str) -> Option<DateTime<Utc>> {
    NaiveDateTime::parse_from_str(value.trim(), "%a, %d %b %Y %H:%M:%S GMT")
        .ok()
        .map(|naive| Utc.from_utc_datetime(&naive))
}

/// Parse the `<LastModified>` an S3 listing stamps on an object.
pub fn parse_s3_timestamp(value: &str) -> Option<DateTime<Utc>> {
    let trimmed = value.trim();
    for format in ["%Y-%m-%dT%H:%M:%S%.fZ", "%Y-%m-%dT%H:%M:%SZ"] {
        if let Ok(naive) = NaiveDateTime::parse_from_str(trimmed, format) {
            return Some(Utc.from_utc_datetime(&naive));
        }
    }
    None
}

#[derive(Debug)]
struct S3ListPage {
    objects: Vec<S3Object>,
    /// The `<CommonPrefixes>` roll-up, which is empty unless the request
    /// sent a delimiter and is a refusal when it did not.
    common_prefixes: Vec<String>,
    next_continuation_token: Option<String>,
    /// The bucket's own statement about whether more pages follow.  Kept
    /// even though [`advance_continuation`] reads the token, because the
    /// two must agree and the disagreement is the finding.
    #[allow(dead_code)]
    is_truncated: bool,
}

/// One element of a structurally scanned XML document.
#[derive(Debug)]
struct XmlElement {
    name: String,
    /// Position of the parent in [`XmlDocument::elements`], or `None` for a
    /// root element.
    parent: Option<usize>,
    /// The character data directly inside this element, entity-decoded.
    ///
    /// Accumulated span by span while the document is scanned rather than
    /// sliced back out of the source afterwards, so nothing that is not
    /// character data can become part of a value.  Slicing made the value
    /// the raw source bytes between the tags, and a comment or a processing
    /// instruction occupies source bytes there: re-verification #5 read
    /// `2026/07/29/KTLX/<!--c-->KTLX20260729_000234_V06` back as a key and
    /// `2026-07-29<!--c-->T00:09:29.000Z` back as the `last_modified` this
    /// client calls the object's identity in time.
    text: String,
    /// Set when this element contains other elements, so its inner bytes
    /// are a subtree rather than a value.
    has_children: bool,
    /// The attributes this element's start tag declared, name and
    /// entity-decoded value, in order.
    ///
    /// Kept so the schema can refuse one: the ListObjectsV2 response syntax
    /// declares no attribute on any element but the root's namespace, and
    /// an attribute this client does not read is a claim about the entry it
    /// would honour by ignoring -- `<Contents deleted="true">` was accepted
    /// as a volume.
    ///
    /// The VALUE is kept as well as the name.  Round 4 compared names only,
    /// and re-verification #6 measured what that leaves: the root's `xmlns`
    /// value is an unbounded region nothing reads, so a real `<Contents>`
    /// subtree moved verbatim into it -- with `<KeyCount>` lowered to agree
    /// -- was accepted with a short roster (RV6-01).  A value that is never
    /// read is a region a `<Contents>` entry can be carried out of the
    /// roster inside, exactly like the comment and the processing
    /// instruction round 4 refused.
    attributes: Vec<(String, String)>,
}

/// A structurally scanned XML document: elements, their nesting, and the
/// guarantee that every tag opened was closed in order.
#[derive(Debug)]
struct XmlDocument {
    elements: Vec<XmlElement>,
    /// Elements at depth zero.  A well-formed document has exactly one.
    roots: Vec<usize>,
}

/// XML 1.0's `S` production, which is four characters and not the Unicode
/// White_Space property.
///
/// ```text
/// S ::= (#x20 | #x9 | #xD | #xA)+
/// ```
///
/// `char::is_whitespace` and `str::trim` answer for White_Space, which also
/// admits NBSP (`#xA0`), the line and paragraph separators (`#x2028`,
/// `#x2029`), the en/em spaces and a dozen others.  None of them is `S`, so
/// a conforming parser reads `<?xml version="1.0"{NBSP}encoding="UTF-8"?>`
/// as a declaration with one malformed pseudo-attribute region rather than
/// two well-separated ones, and `{NBSP}<ListBucketResult>` as a document
/// with character data before its root.  This round's thesis is that XML
/// 1.0's productions are the bound, so the predicate is the production:
/// used wherever the reader is standing at an `S` position -- separating
/// declaration and start-tag attributes, terminating a tag name, and
/// deciding that what lies outside the root element is nothing.
///
/// It is deliberately NOT used on element VALUES.  `<LastModified>` being
/// blank is a statement about a value this client reads, not about XML
/// syntax, and Unicode-blank is the wider and therefore safer test there.
fn is_xml_space(character: char) -> bool {
    matches!(character, ' ' | '\t' | '\r' | '\n')
}

/// Index of the `>` that ends the tag beginning at `start`, honouring
/// quoted attribute values.
fn xml_tag_end(xml: &str, start: usize) -> Option<usize> {
    let bytes = xml.as_bytes();
    let mut quote: Option<u8> = None;
    let mut index = start + 1;
    while index < bytes.len() {
        let byte = bytes[index];
        match quote {
            Some(open) if byte == open => quote = None,
            Some(_) => {}
            None if byte == b'"' || byte == b'\'' => quote = Some(byte),
            None if byte == b'>' => return Some(index),
            None => {}
        }
        index += 1;
    }
    None
}

/// The attributes declared in `tag`, which is one start tag's body with its
/// element name already taken off the front: `(name, RAW value)` pairs in
/// document order, exactly as the source spells them.
///
/// Parsed rather than skipped because an attribute is a statement about the
/// element, and an unparseable attribute region is a start tag this client
/// cannot claim to have read.  Values are required to be quoted, which XML
/// requires too, so `<Contents deleted=true>` is a refusal rather than a
/// silent nothing.
///
/// Three XML 1.0 well-formedness constraints are enforced here.  The first
/// two are the whole of RV6-01:
///
/// * **an `AttValue` may not contain a raw `<`.**  The production is
///   literally `'"' ([^<&"] | Reference)* '"'`, so a value carrying markup
///   is not well-formed XML at all -- yet round 4's scanner reported such a
///   document as well-formed and handed back a roster.  An attribute value
///   is also the last region in the document that nothing reads and nothing
///   bounds, which is what makes it a carrier: re-verification #6 moved a
///   real `<Contents>` subtree into the root's `xmlns` value, lowered
///   `<KeyCount>` by one, and got an accepted two-object roster.  Refusing
///   the character kills the class -- a value can never carry markup at all
///   -- rather than the one reproduction;
/// * **a start tag may not declare the same attribute twice** (WFC: Unique
///   Att Spec).  Two values for one name are two answers to one question,
///   and the root declaring `xmlns` twice parsed;
/// * **attributes are separated by `S`.**  `STag ::= '<' Name (S Attribute)*
///   S? '>'` puts whitespace before every attribute, so
///   `xmlns="..."deleted="true"` is not well-formed XML.  Round 5 accepted
///   the pair and left the second to the schema one layer later
///   (re-verification #7's nit RV7-04a); the reader now says what it is,
///   which is a body that failed a check S3 output cannot fail.  `S` is
///   [`is_xml_space`] -- the four characters the production names -- and not
///   Unicode White_Space, so `xmlns="..."{NBSP}deleted="true"` is the same
///   unseparated pair spelled with a character that is not a separator
///   (RV8-03).
///
/// The value comes back **raw** because two callers need two different
/// things from it and only one of them is an `AttValue`: [`xml_attributes`]
/// normalises and entity-decodes it as XML 1.0 §3.3.3 requires of a real
/// attribute, while [`require_xml_declaration`] holds it to the literal
/// productions XML 1.0 gives the declaration's pseudo-attributes, in which
/// no reference is permitted and nothing is decoded (RV7-01).
fn xml_attribute_pairs(tag: &str) -> Result<Vec<(&str, &str)>, String> {
    let mut attributes: Vec<(&str, &str)> = Vec::new();
    let mut rest = tag;
    while !rest.trim_start_matches(is_xml_space).is_empty() {
        let separated = rest.starts_with(is_xml_space);
        let head = rest.trim_start_matches(is_xml_space);
        let name_end = head
            .find(|c: char| is_xml_space(c) || c == '=')
            .ok_or_else(|| {
                format!("the document carries attribute {head:?} with no value")
            })?;
        let (name, tail) = head.split_at(name_end);
        if name.is_empty() {
            return Err(
                "the document carries an attribute with no name in a start \
                 tag"
                    .to_string(),
            );
        }
        if !separated {
            return Err(format!(
                "the document carries attribute {name:?} with no whitespace \
                 before it; XML 1.0 separates one attribute from the last \
                 with S, so this body failed a check S3 output cannot fail"
            ));
        }
        let tail = tail.trim_start_matches(is_xml_space);
        let tail = tail.strip_prefix('=').ok_or_else(|| {
            format!("the document carries attribute {name:?} with no value")
        })?;
        let tail = tail.trim_start_matches(is_xml_space);
        let quote = match tail.as_bytes().first() {
            Some(&byte @ (b'"' | b'\'')) => byte as char,
            _ => {
                return Err(format!(
                    "the document carries attribute {name:?} with an \
                     unquoted value"
                ));
            }
        };
        let value_end = tail[1..].find(quote).ok_or_else(|| {
            format!(
                "the document carries attribute {name:?} with a value that \
                 never closes its quote"
            )
        })?;
        let raw = &tail[1..1 + value_end];
        if raw.contains('<') {
            return Err(format!(
                "the document carries a raw '<' inside the value of \
                 attribute {name:?}; XML 1.0's AttValue production forbids \
                 it, so this body failed a check S3 output cannot fail -- \
                 and an attribute value is a region nothing reads and \
                 nothing bounds, which is a region a <Contents> entry can be \
                 carried out of the roster inside"
            ));
        }
        if attributes.iter().any(|(seen, _)| *seen == name) {
            return Err(format!(
                "the document declares attribute {name:?} twice in one start \
                 tag; XML 1.0 gives an element at most one attribute of each \
                 name, and which of the two is the answer is not something \
                 this client will pick"
            ));
        }
        attributes.push((name, raw));
        rest = &tail[1 + value_end + 1..];
    }
    Ok(attributes)
}

/// One attribute value as XML 1.0 says it is to be read: `(name, value)`
/// pairs with §3.3.3 normalisation applied and the predefined entities
/// decoded.
fn xml_attributes(tag: &str) -> Result<Vec<(String, String)>, String> {
    let mut attributes: Vec<(String, String)> = Vec::new();
    for (name, raw) in xml_attribute_pairs(tag)? {
        attributes.push((
            name.to_string(),
            xml_unescape(&normalize_attribute_value(raw))?,
        ));
    }
    Ok(attributes)
}

/// XML 1.0 §3.3.3 attribute-value normalisation, for the part of it this
/// deliberately small language can reach: a literal `#x9`, `#xA` or `#xD` in
/// an attribute value is a single `#x20` in the normalised value, and a
/// literal `#xD#xA` is one `#x20` rather than two, because §2.11 folds the
/// pair to a single `#xA` before §3.3.3 sees it.
///
/// RV7-04b: the value the schema compares against the declared namespace was
/// the un-normalised source text, which is not the value XML 1.0 defines.
/// It changes no verdict for `xmlns` -- normalisation replaces a line break
/// with a space rather than removing it, so a line-wrapped namespace is
/// still not the declared one -- but the comparison is now the one the
/// standard specifies rather than one that happens to agree with it.
fn normalize_attribute_value(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    let mut characters = raw.chars().peekable();
    while let Some(character) = characters.next() {
        match character {
            '\r' => {
                if characters.peek() == Some(&'\n') {
                    characters.next();
                }
                out.push(' ');
            }
            '\n' | '\t' => out.push(' '),
            _ => out.push(character),
        }
    }
    out
}

/// The XML declaration's own body -- everything between `<?xml` and `?>` --
/// against the one grammar XML 1.0 gives it.
///
/// RV6-03: the declaration is skipped wholesale, and a skipped region is a
/// carrier.  A real `<Contents>` planted inside the byte-zero declaration,
/// with `<KeyCount>` lowered to agree, was accepted with a short roster.
/// The declaration is `VersionInfo EncodingDecl? SDDecl?` and nothing else,
/// so anything else in it is refused -- markup by the character, and any
/// pseudo-attribute the production does not name, out of order or repeated,
/// by name.
///
/// RV7-01: holding the NAMES is not holding the declaration.  Round 5 read
/// the three values, entity-decoded them, and then compared them against
/// nothing, so the region stayed unbounded and the escaped half of RV6-03's
/// class stayed open: the real second roster entry, `&lt;`-escaped, in any
/// one of `version` / `encoding` / `standalone`, with `<KeyCount>` lowered
/// by one, was accepted with a two-object roster and the entry still in the
/// bytes.  A 4 KB `encoding` was accepted too.
///
/// The values are held to their own productions now, and the productions
/// are the whole fix, because a value that must match one cannot carry
/// anything else:
///
/// ```text
/// VersionNum ::= '1.' [0-9]+
/// EncName    ::= [A-Za-z] ([A-Za-z0-9._] | '-')*
/// SDDecl     ::= 'yes' | 'no'
/// ```
///
/// Note what these are **not**: they are not `AttValue`.  XML 1.0 spells
/// each of the three out as literal characters with no `Reference`
/// alternative anywhere in it, so an entity reference is not permitted in a
/// declaration value at all -- which is why nothing here decodes one.  The
/// raw source text is what is matched, `&#60;` and `&lt;` are simply
/// characters no production admits, and a declaration value can no longer
/// hold a roster entry in any spelling.
///
/// RV8-04: a grammar is not a meaning.  `encoding` was held to `EncName` and
/// then discarded, so `encoding="UTF-16"` on a UTF-8 body parsed -- this
/// client decodes the response as UTF-8 unconditionally and nothing reads
/// the label.  A declared `encoding` must now be a UTF-8 label
/// ([`is_utf8_encoding_label`]); the alternative was to document that the
/// value is validated and ignored, and a validator that says nothing about
/// what it validated is the residual, not the fix.
fn require_xml_declaration(body: &str) -> Result<(), String> {
    if body.contains('<') {
        return Err(
            "the document's <?xml ...?> declaration carries markup ('<'); \
             XML 1.0 gives the declaration a version, an encoding and a \
             standalone statement and nothing else, so this body failed a \
             check S3 output cannot fail -- and a region a parser skips is a \
             region a <Contents> entry can be carried out of the roster \
             inside"
                .to_string(),
        );
    }
    let after_target = body.strip_prefix("xml").unwrap_or(body);
    let pseudo = xml_attribute_pairs(after_target).map_err(|problem| {
        format!("in the document's <?xml ...?> declaration, {problem}")
    })?;
    const DECLARED: [&str; 3] = ["version", "encoding", "standalone"];
    let mut next = 0usize;
    for (name, value) in &pseudo {
        let offset = DECLARED[next..]
            .iter()
            .position(|declared| declared == name)
            .ok_or_else(|| {
                format!(
                    "the document's <?xml ...?> declaration carries \
                     {name:?}; XML 1.0 declares version, then an optional \
                     encoding, then an optional standalone, in that order \
                     and once each"
                )
            })?;
        next += offset + 1;
        require_declaration_value(name, value)?;
    }
    match pseudo.first() {
        Some((name, _)) if *name == "version" => Ok(()),
        _ => Err(
            "the document's <?xml ...?> declaration states no version; XML \
             1.0 requires one"
                .to_string(),
        ),
    }
}

/// One declaration pseudo-attribute value against its own XML 1.0
/// production, on the raw source text.
///
/// The name is already known to be one of the three by the time this is
/// called; an unknown one is refused by name before it gets here.
fn require_declaration_value(name: &str, value: &str) -> Result<(), String> {
    let (matches_grammar, production) = match name {
        "version" => (
            match value.strip_prefix("1.") {
                Some(digits) => {
                    !digits.is_empty() && digits.bytes().all(|byte| byte.is_ascii_digit())
                }
                None => false,
            },
            "VersionNum ::= '1.' [0-9]+",
        ),
        "encoding" => {
            let mut characters = value.chars();
            (
                match characters.next() {
                    Some(first) => {
                        first.is_ascii_alphabetic()
                            && characters.all(|character| {
                                character.is_ascii_alphanumeric()
                                    || matches!(character, '.' | '_' | '-')
                            })
                    }
                    None => false,
                },
                "EncName ::= [A-Za-z] ([A-Za-z0-9._] | '-')*",
            )
        }
        "standalone" => (
            value == "yes" || value == "no",
            "SDDecl ::= 'yes' | 'no'",
        ),
        _ => return Ok(()),
    };
    if matches_grammar {
        // The grammar is not the whole of what an `encoding` says.  Round 6
        // gave the value a production and stopped there, which left the
        // declaration validated for shape and discarded for meaning:
        // `encoding="UTF-16"` and `encoding="EBCDIC-CP-US"` are both inside
        // `EncName`, both parsed, and both were read as UTF-8 anyway,
        // because the body is decoded UTF-8 unconditionally (ureq's charset
        // feature is off) and nothing downstream reads the label.  That is
        // not a carrier, but it is a statement this client was accepting and
        // then ignoring, and the accurate close is to accept only the one it
        // can honour.  No transcoding is added: the divergence is removed by
        // refusing the input, which is what the rest of this reader does.
        if name == "encoding" && !is_utf8_encoding_label(value) {
            return Err(format!(
                "the document's <?xml ...?> declaration states \
                 encoding={value:?}; this client reads an S3 ListObjectsV2 \
                 response as UTF-8 and only as UTF-8 -- the body is decoded \
                 before the declaration is looked at and nothing here \
                 transcodes -- so a body that declares itself in another \
                 encoding is a body this client would read in a way the \
                 document says is wrong, and two readings of one page are \
                 two different rosters. S3 labels its ListObjectsV2 \
                 responses UTF-8"
            ));
        }
        return Ok(());
    }
    Err(format!(
        "the document's <?xml ...?> declaration states {name}={value:?}, \
         which XML 1.0's {production} does not admit -- and this production \
         is literal characters, not an AttValue, so no entity reference is \
         permitted there and none is decoded here. A declaration value held \
         to no grammar is an unbounded region nothing reads, which is a \
         region a <Contents> entry can be carried out of the roster inside"
    ))
}

/// Whether an `EncName` names the one encoding this client can honour.
///
/// Case-insensitive, because XML 1.0 §4.3.3 says encoding names are matched
/// case-insensitively, and both registered spellings are admitted.  Nothing
/// wider: `US-ASCII` is a strict subset of UTF-8 and would decode correctly,
/// but accepting a label on the grounds that its documents happen to decode
/// is the same reasoning that left the value unread in the first place, and
/// S3 does not emit it.
fn is_utf8_encoding_label(value: &str) -> bool {
    value.eq_ignore_ascii_case("utf-8") || value.eq_ignore_ascii_case("utf8")
}

/// The elements still open, innermost first, as a sentence.
///
/// Written this way rather than as a list because the innermost element is
/// rarely the interesting one: a body cut mid-`<ChecksumAlgorithm>` is a
/// body cut inside a `<Contents>` entry, and naming the chain says which
/// roster element was lost as well as which byte the cut landed on.
fn xml_open_chain(elements: &[XmlElement], stack: &[usize]) -> String {
    let mut parts = Vec::new();
    for &index in stack.iter().rev() {
        parts.push(format!("an unterminated <{}> element", elements[index].name));
    }
    if parts.is_empty() {
        return "no open element".to_string();
    }
    parts.join(", inside ")
}

/// Scan `xml` structurally: every tag opened is closed, in order, and
/// nothing but whitespace lies outside the root element.
///
/// This is the replacement for reading a listing with `str::find`.  The
/// retired parser checked that the opening substring `<ListBucketResult`
/// existed and then pulled each field out of the whole buffer by searching
/// for its tag, so a page with no closing `</ListBucketResult>` at all
/// parsed to `Ok`: re-verification removed exactly that one tag from the
/// real final-page fixture and got its object back.  Nothing in the field
/// checks could have caught it, because none of them is a statement about
/// the document.  A page that arrived short is the case that matters --
/// the roster is short with it -- so the shape of the document is checked
/// as a shape, once, rather than inferred from whichever field happened to
/// go missing with the bytes.
///
/// Deliberately small and deliberately strict: no DOCTYPE, no CDATA, no
/// entity declarations.  S3 sends none of them, and a listing that arrives
/// carrying one is not a listing this client should be interpreting.
///
/// The same doctrine is applied to the two other markup regions a parser is
/// allowed to discard.  Discarding them is what a conforming XML reader
/// does, and it is exactly why they are carriers: re-verification #5 moved
/// a real `<Contents>` inside `<?hide ...?>` and inside `<!-- ... -->` and
/// got `Ok` with a two-object roster and a `<KeyCount>` lowered to agree.
/// So a processing instruction is refused everywhere except the one
/// position XML 1.0 gives the declaration -- byte zero -- and **a comment
/// is refused everywhere, prolog and epilog included** (RV7-02).
///
/// Rounds 4 and 5 left a comment outside the root alone and disclosed it as
/// a residual on the ground that it "can carry nothing out of a roster it is
/// not inside".  Re-verification #7 measured that sentence and it is false:
/// moving an entry OUT of the root and into a prolog or trailing comment,
/// with `<KeyCount>` lowered by one, is precisely how the roster is
/// shortened, and both were accepted with two objects and the entry still in
/// the bytes.  The region is the finding, not the position: it is raw,
/// unparsed markup that a parser skips, which is the same criterion the
/// declaration body was refused on one function away.  S3 sends no comment
/// anywhere in a `ListObjectsV2` response -- none of the captured real pages
/// carries one -- so the doctrine is now applied to the whole document
/// rather than to the half of it that had been probed.
///
/// Re-verification #6 found the two regions that doctrine had not reached,
/// and both are closed here rather than in the schema, because both are
/// statements about well-formedness that round 4 got wrong:
///
/// * an **attribute value** ([`xml_attributes`]) may not contain a raw
///   `<` -- XML 1.0 forbids it outright -- and a start tag may not declare
///   one attribute twice.  A real `<Contents>` moved into the root's
///   `xmlns` value was accepted with a short roster while `scan_xml` called
///   the document well-formed, which it was not (RV6-01);
/// * the **declaration's own body** ([`require_xml_declaration`]) is
///   `version`, then an optional `encoding`, then an optional `standalone`,
///   and nothing else.  Being allowed at byte zero is not being allowed to
///   contain anything (RV6-03).
///
/// Re-verification #8 measured two more places where the reader was reading
/// a language slightly wider than the one it claims, and one where it was
/// not reading at all:
///
/// * `<?>` is three bytes that made this function **panic**.  The terminator
///   was searched for from byte zero, so the `?>` starting at byte 1 was
///   found and the body slice ran backwards.  The search starts at byte 2
///   now and the sequence is what it looks like: a `<?...?>` region that
///   never terminates, or a processing instruction with an unreadable
///   target, refused by name either way (RV8-01);
/// * `S` is [`is_xml_space`], the four characters XML 1.0's production
///   names, wherever this function stands at an `S` position: what lies
///   outside the root, what ends a tag name, what ends a PI target, and
///   (in [`xml_attribute_pairs`]) what separates one attribute from the
///   last.  Unicode White_Space is wider, so NBSP and `#x2028` were
///   separators here and are not separators in XML (RV8-03);
/// * a literal `]]>` in character data is refused.  XML 1.0 §2.4 forbids it
///   outside a CDATA section, this reader has no CDATA sections, and the
///   legal spelling `]]&gt;` still decodes -- the check is on the source
///   span, not on the decoded value.
fn scan_xml(xml: &str) -> Result<XmlDocument, String> {
    let mut elements: Vec<XmlElement> = Vec::new();
    let mut roots: Vec<usize> = Vec::new();
    let mut stack: Vec<usize> = Vec::new();
    let mut pos = 0usize;

    while let Some(offset) = xml[pos..].find('<') {
        let start = pos + offset;
        if stack.is_empty() && !xml[pos..start].trim_matches(is_xml_space).is_empty() {
            return Err(format!(
                "the document carries text outside its root element \
                 ({:?})",
                xml[pos..start].trim_matches(is_xml_space)
            ));
        }
        // XML 1.0 §2.4: the literal string `]]>` may not appear in character
        // data except as the end of a CDATA section, and this reader has no
        // CDATA sections.  The escape XML gives the sequence is `]]&gt;`,
        // which is why the check is on the SOURCE span rather than on the
        // decoded characters: refusing the decoded form would refuse the
        // legal spelling too.
        if xml[pos..start].contains("]]>") {
            return Err(format!(
                "the document carries a literal ']]>' in character data, \
                 within {}; XML 1.0 forbids the sequence there outside a \
                 CDATA section and this reader has none, so this body failed \
                 a check S3 output cannot fail",
                xml_open_chain(&elements, &stack)
            ));
        }
        let characters = xml_unescape(&xml[pos..start]).map_err(|problem| {
            format!(
                "{problem}, within {}",
                xml_open_chain(&elements, &stack)
            )
        })?;
        if let Some(&open) = stack.last() {
            elements[open].text.push_str(&characters);
        }
        let rest = &xml[start..];
        if let Some(after_open) = rest.strip_prefix("<?") {
            // The terminator is searched for PAST the `<?` that opened this
            // region, because `?>` can otherwise be found INSIDE it: in
            // `<?>` the sequence starts at byte 1, so a search over the
            // whole of `rest` reported `end = 1` and the body slice
            // `rest[2..1]` was a backwards range -- RV8-01, a slice-index
            // panic on three bytes any listing body can contain.  It unwound
            // through `parse_s3_list_xml` to `main` (exit 101, a message
            // about a byte range), which is strictly worse than every
            // carrier this ladder has closed: each of those at least
            // produced a named refusal about the page, and a panic produces
            // a verdict about nothing.  The region is taken by its opening
            // delimiter rather than sliced past it, so the offset cannot be
            // got wrong again.
            let end = after_open.find("?>").ok_or_else(|| {
                "the document ends inside an unterminated <?...?> \
                 declaration"
                    .to_string()
            })?;
            let region = &after_open[..end];
            // `PI ::= '<?' PITarget (S ...)? '?>'`, so the target ends at
            // an `S` and at nothing else.
            let target = region.split(is_xml_space).next().unwrap_or("");
            if target != "xml" {
                return Err(format!(
                    "the document carries a <?{target} ...?> processing \
                     instruction; S3 sends none of them and this client will \
                     not interpret one, and a region a parser skips is a \
                     region a <Contents> entry can be carried out of the \
                     roster inside"
                ));
            }
            if start != 0 {
                return Err(format!(
                    "the document carries an <?xml ...?> declaration at byte \
                     {start}; XML 1.0 permits the declaration only at the \
                     very start of the document, so this body failed a check \
                     S3 output cannot fail -- and a region a parser skips is \
                     a region a <Contents> entry can be carried out of the \
                     roster inside"
                ));
            }
            require_xml_declaration(region)?;
            pos = start + 2 + end + 2;
            continue;
        }
        if rest.starts_with("<!--") {
            return Err(match stack.last() {
                Some(&open) => format!(
                    "the document carries a comment inside <{}>; S3 sends \
                     none, a region a parser skips is a region a <Contents> \
                     entry can be carried out of the roster inside, and \
                     comment markup between an element's tags is markup that \
                     would otherwise be read back as part of its value",
                    elements[open].name
                ),
                None => "the document carries a comment outside its root \
                         element; S3 sends none anywhere, and a region a \
                         parser skips is a region a <Contents> entry can be \
                         carried out of the roster inside -- \
                         re-verification #7 shortened the roster by exactly \
                         this move, with a verbatim entry in a prolog \
                         comment and <KeyCount> lowered to agree"
                    .to_string(),
            });
        }
        if rest.starts_with("<!") {
            return Err(
                "the document carries a <!...> declaration (DOCTYPE, CDATA \
                 or entity); S3 sends none of them and this client will not \
                 interpret one"
                    .to_string(),
            );
        }
        let close = xml_tag_end(xml, start).ok_or_else(|| {
            format!(
                "the document ends inside an unterminated tag, within {}",
                xml_open_chain(&elements, &stack)
            )
        })?;
        let body = &xml[start + 1..close];
        if let Some(name) = body.strip_prefix('/') {
            let name = name.trim_matches(is_xml_space);
            match stack.pop() {
                Some(index) if elements[index].name == name => {}
                Some(index) => {
                    return Err(format!(
                        "</{name}> closes an element that was never opened; \
                         the innermost open element is <{}>",
                        elements[index].name
                    ));
                }
                None => {
                    return Err(format!(
                        "</{name}> closes an element that was never opened"
                    ));
                }
            }
        } else {
            let self_closing = body.ends_with('/');
            let core = if self_closing {
                &body[..body.len() - 1]
            } else {
                body
            };
            let core = core.trim_start_matches(is_xml_space);
            let name_end = core.find(is_xml_space).unwrap_or(core.len());
            let name = &core[..name_end];
            if name.is_empty() {
                return Err("the document carries a tag with no name".to_string());
            }
            // Entity references in attribute values are XML too.  We do not
            // consume any ListObjectsV2 attributes, but accepting an
            // undeclared entity there would still accept a document that is
            // not in the deliberately small XML language described above.
            xml_unescape(core)?;
            let attributes = xml_attributes(&core[name_end..])?;
            let index = elements.len();
            match stack.last() {
                Some(&parent) => elements[parent].has_children = true,
                None => {
                    if let Some(&first) = roots.first() {
                        return Err(format!(
                            "the document carries a second root element \
                             <{name}> after <{}>; XML has exactly one, and a \
                             body that appends one has appended something to \
                             the listing",
                            elements[first].name
                        ));
                    }
                    roots.push(index);
                }
            }
            elements.push(XmlElement {
                name: name.to_string(),
                parent: stack.last().copied(),
                text: String::new(),
                has_children: false,
                attributes,
            });
            if !self_closing {
                stack.push(index);
            }
        }
        pos = close + 1;
    }

    if !stack.is_empty() {
        return Err(format!(
            "the document ends inside {}",
            xml_open_chain(&elements, &stack)
        ));
    }
    if !xml[pos..].trim_matches(is_xml_space).is_empty() {
        return Err(format!(
            "the document carries text after its root element ({:?})",
            xml[pos..].trim_matches(is_xml_space)
        ));
    }
    xml_unescape(&xml[pos..])?;
    Ok(XmlDocument { elements, roots })
}

impl XmlDocument {
    /// Positions of `parent`'s DIRECT children, in document order.
    ///
    /// Direct rather than "anywhere below", which is what searching the
    /// whole buffer for `<Prefix>` amounted to: in a delimited listing the
    /// `<Prefix>` inside a `<CommonPrefixes>` roll-up would answer for the
    /// listing's own echoed prefix.
    fn children(&self, parent: usize) -> impl Iterator<Item = usize> + '_ {
        (0..self.elements.len()).filter(move |&index| self.elements[index].parent == Some(parent))
    }

    /// Check `root` and everything below it against the ListObjectsV2
    /// content model.
    ///
    /// Well-nested XML alone does not establish a ListObjectsV2 response:
    /// a wrapper can hide a real `<Contents>` from the roster while leaving
    /// a closed document and an adjusted `<KeyCount>`.  The first repair
    /// checked the direct-child NAMES at the two roster-bearing levels,
    /// which is not a schema, and re-verification #5 measured the
    /// difference: an allow-listed element the parser never descends into
    /// is an unchecked subtree, and **4 of 12** root children (`Delimiter`,
    /// `EncodingType`, `ContinuationToken`, `StartAfter`) and **6 of 9**
    /// `<Contents>` children (`ChecksumAlgorithm`, `ChecksumType`, `ETag`,
    /// `Owner`, `RestoreStatus`, `StorageClass`) each carried a real
    /// `<Contents>` out of the roster while every direct child stayed
    /// allow-listed and `<KeyCount>` agreed.  The names that did refuse
    /// refused only because [`XmlDocument::child_text`] happened to read
    /// them twice, which is incidental and is absent for exactly the four
    /// root names the real page does not already carry.
    ///
    /// So every element in the response syntax declares a content model --
    /// text-only, or a named set of permitted children with their
    /// multiplicity -- and this walks the whole document against it.  A
    /// text-only element refuses ANY element child; a container refuses any
    /// child it does not declare AND its own non-`S` character data
    /// (RV8-02); a child declared [`Occurs::Once`] refuses
    /// a second; and an attribute nothing declares refuses too.  The
    /// attribute pass runs first over every element, so an element carrying
    /// both faults is named for the attribute rather than for whichever
    /// structural check the walk reached first.
    fn require_list_objects_v2_schema(&self, root: usize) -> Result<(), String> {
        for (index, element) in self.elements.iter().enumerate() {
            let declared: &[(&str, &str)] = if index == root {
                LIST_OBJECTS_V2_ROOT_ATTRIBUTES
            } else {
                &[]
            };
            for (attribute, value) in &element.attributes {
                let Some((_, expected)) =
                    declared.iter().find(|(name, _)| name == attribute)
                else {
                    return Err(format!(
                        "unexpected attribute {attribute:?} on <{}>; the \
                         ListObjectsV2 response syntax declares none there. \
                         An attribute this client does not read is a claim \
                         about the entry it would honour by ignoring, which \
                         is how <Contents deleted=\"true\"> becomes a volume \
                         in the roster",
                        element.name
                    ));
                };
                // The one attribute the response syntax does declare has one
                // value, and it is checked.  Round 4 compared names only, so
                // `xmlns="urn:not-s3-at-all"` parsed as a ListObjectsV2
                // listing (RV6-03): an element schema transcribed from one
                // namespace cannot describe a document that says it is in
                // another.
                if value.is_empty() {
                    return Err(format!(
                        "attribute {attribute:?} on <{}> declares {value:?}, \
                         which in XML is the statement that the element is in \
                         NO namespace at all; the ListObjectsV2 response \
                         syntax declares {expected:?} there, and this element \
                         schema is a transcription of that one namespace",
                        element.name
                    ));
                }
                if value != expected {
                    return Err(format!(
                        "attribute {attribute:?} on <{}> declares {value:?}; \
                         the ListObjectsV2 response syntax declares \
                         {expected:?} there, and this element schema is a \
                         transcription of that one namespace -- a document \
                         that says it is in another is not the document this \
                         client can read",
                        element.name
                    ));
                }
            }
        }
        // Declared means required, not merely checked when it turns up.
        // RV7-03: round 5 compared the namespace VALUE when the root stated
        // one and said nothing at all when it did not, so a
        // <ListBucketResult> carrying no `xmlns` parsed to a full
        // three-object roster while `xmlns=""` -- which in XML is the same
        // statement, "in no namespace" -- was refused, and the suite pinned
        // the refusal of the empty one only.  Absent, empty and wrong are
        // three spellings of one fault; all three refuse, each saying which
        // it is.  This runs after the pass above so a root that carries an
        // undeclared attribute is still named for that attribute.
        for (name, expected) in LIST_OBJECTS_V2_ROOT_ATTRIBUTES {
            if !self.elements[root]
                .attributes
                .iter()
                .any(|(declared, _)| declared == name)
            {
                return Err(format!(
                    "<{}> declares no {name:?} attribute; the ListObjectsV2 \
                     response syntax declares {expected:?} there, and this \
                     element schema is a transcription of that one namespace \
                     -- a document that says nothing about the namespace it \
                     is in is no more the document this client can read than \
                     one that says it is in another",
                    self.elements[root].name
                ));
            }
        }
        self.require_content_model(root)
    }

    /// One element and its subtree against the declared content model.
    ///
    /// Element content is element content: a `Content::Children` element's
    /// OWN character data is refused when it is anything but `S`.
    ///
    /// Rounds 4 through 6 disclosed the opposite as a residual -- "a
    /// container's text is never read as a value, so nothing can be
    /// smuggled through it" -- and re-verification #8 measured that sentence
    /// and it is false, in exactly the shape RV7-01 and RV6-01 were false:
    /// a region nothing reads and nothing bounds is a region a `<Contents>`
    /// entry can be carried OUT of the roster inside.  The real second entry
    /// was `&lt;`-escaped into `<ListBucketResult>`'s own character data --
    /// and into a `<Contents>` element's -- with `<KeyCount>` lowered by one,
    /// and both were accepted with a short two-object roster and the entry
    /// still in the bytes.  Being unreadable is not being harmless; the
    /// roster is shortened by what leaves it, not by what is read.
    ///
    /// Whitespace-only text stays accepted, which is what the residual was
    /// actually protecting: a pretty-printed roster is indented, S3 may
    /// indent one, and indentation carries nothing.
    fn require_content_model(&self, element: usize) -> Result<(), String> {
        let name = self.elements[element].name.as_str();
        let Some(content) = list_objects_v2_content(name) else {
            return Err(format!(
                "<{name}> has no declared content model in the ListObjectsV2 \
                 element schema; this client will not walk an element it \
                 cannot describe, because an element it cannot describe is \
                 an element a <Contents> entry can be hidden inside"
            ));
        };
        let permitted: &[(&str, Occurs)] = match content {
            Content::Text => &[],
            Content::Children(children) => children,
        };
        let mut seen: Vec<&str> = Vec::new();
        for index in self.children(element) {
            let child = self.elements[index].name.as_str();
            let Some((_, occurs)) = permitted.iter().find(|(n, _)| *n == child) else {
                return Err(match content {
                    Content::Text => format!(
                        "unexpected <{child}> direct child of <{name}>; \
                         <{name}> is a text-only element in the \
                         ListObjectsV2 element schema and carries no element \
                         children at all, so this document is not a \
                         structurally valid object roster"
                    ),
                    Content::Children(_) => format!(
                        "unexpected <{child}> direct child of <{name}>; it \
                         is not in the ListObjectsV2 element schema, so this \
                         document is not a structurally valid object roster"
                    ),
                });
            };
            if *occurs == Occurs::Once && seen.contains(&child) {
                return Err(format!(
                    "the document declares <{child}> more than once in one \
                     <{name}> element; which of them is the answer is not \
                     something this client will pick"
                ));
            }
            seen.push(child);
            self.require_content_model(index)?;
        }
        if let Content::Children(_) = content {
            let stray = self.elements[element].text.trim_matches(is_xml_space);
            if !stray.is_empty() {
                return Err(format!(
                    "<{name}> is an element-content element in the \
                     ListObjectsV2 element schema, so its own character data \
                     is not a value this client reads -- yet it carries the \
                     text {stray:?}, and a region nothing reads and nothing \
                     bounds is a region a <Contents> entry can be carried out \
                     of the roster inside"
                ));
            }
        }
        Ok(())
    }

    /// The text of `parent`'s one child named `name`.
    ///
    /// `None` when there is none.  An error when there are two, because a
    /// page stating `<KeyCount>` twice states two different things and
    /// taking the first is a choice nobody made deliberately, and when the
    /// child has children of its own, because a subtree is not a value.
    fn child_text(&self, parent: usize, name: &str) -> Result<Option<String>, String> {
        let mut found: Option<usize> = None;
        for index in self.children(parent) {
            if self.elements[index].name != name {
                continue;
            }
            if found.is_some() {
                return Err(format!(
                    "the document declares <{name}> more than once in one \
                     <{}> element; which of them is the answer is not \
                     something this client will pick",
                    self.elements[parent].name
                ));
            }
            found = Some(index);
        }
        let Some(index) = found else {
            return Ok(None);
        };
        let element = &self.elements[index];
        if element.has_children {
            return Err(format!(
                "the document's <{name}> element carries child elements \
                 where a value was expected"
            ));
        }
        Ok(Some(element.text.clone()))
    }
}

/// How often one child may appear inside its declared parent.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Occurs {
    /// At most once.  A second is two answers to one question, and picking
    /// one is a decision nobody made deliberately.
    Once,
    /// Any number of times: the two roll-ups and the checksum algorithm
    /// list, each of which the response syntax prints followed by the
    /// repetition ellipsis.
    Repeated,
}

/// What one ListObjectsV2 element is allowed to contain.
#[derive(Debug)]
enum Content {
    /// Character data only.  No element child, ever -- which is the whole
    /// of RV5-01: `<StartAfter>` is a string in the response syntax, so a
    /// `<Contents>` inside one is not a listing that arrived, it is a
    /// listing that was edited.
    Text,
    /// Exactly these children, with these multiplicities, and nothing else.
    Children(&'static [(&'static str, Occurs)]),
}

/// The ListObjectsV2 response element schema, as a content model.
///
/// Transcribed from the AWS ListObjectsV2 response syntax:
/// <https://docs.aws.amazon.com/AmazonS3/latest/API/API_ListObjectsV2.html>.
/// Every element named as a permitted child appears here in its own right
/// with its own model -- `every_schema_element_declares_a_content_model`
/// asserts that, so a name added to a child list without a model fails the
/// suite instead of becoming a new hiding place.
///
/// `CommonPrefixes` is part of the response schema but is refused later:
/// this client sends no delimiter, so a roll-up would omit objects.  It is
/// declared here so that the more specific refusal keeps explaining that
/// operational fault instead of a structural one.
const LIST_OBJECTS_V2_SCHEMA: &[(&str, Content)] = &[
    (
        "ListBucketResult",
        Content::Children(&[
            ("IsTruncated", Occurs::Once),
            ("Contents", Occurs::Repeated),
            ("Name", Occurs::Once),
            ("Prefix", Occurs::Once),
            ("Delimiter", Occurs::Once),
            ("MaxKeys", Occurs::Once),
            ("CommonPrefixes", Occurs::Repeated),
            ("EncodingType", Occurs::Once),
            ("KeyCount", Occurs::Once),
            ("ContinuationToken", Occurs::Once),
            ("NextContinuationToken", Occurs::Once),
            ("StartAfter", Occurs::Once),
        ]),
    ),
    (
        "Contents",
        Content::Children(&[
            // AWS documents `Object.ChecksumAlgorithm` as `Type: Array of
            // strings` and prints it in the response syntax followed by the
            // repetition ellipsis, exactly as it prints <Contents> and
            // <CommonPrefixes>.  Round 4 transcribed it as at-most-once, so
            // a legitimate object carrying two checksum algorithms
            // hard-refused the whole page -- and `list_s3_objects`
            // propagates that, which costs a site its whole day's roster
            // (RV6-02).  A multiplicity is part of the transcription and
            // this one was wrong.
            ("ChecksumAlgorithm", Occurs::Repeated),
            ("ChecksumType", Occurs::Once),
            ("ETag", Occurs::Once),
            ("Key", Occurs::Once),
            ("LastModified", Occurs::Once),
            ("Owner", Occurs::Once),
            ("RestoreStatus", Occurs::Once),
            ("Size", Occurs::Once),
            ("StorageClass", Occurs::Once),
        ]),
    ),
    (
        "Owner",
        Content::Children(&[
            ("DisplayName", Occurs::Once),
            ("ID", Occurs::Once),
        ]),
    ),
    (
        "RestoreStatus",
        Content::Children(&[
            ("IsRestoreInProgress", Occurs::Once),
            ("RestoreExpiryDate", Occurs::Once),
        ]),
    ),
    ("CommonPrefixes", Content::Children(&[("Prefix", Occurs::Once)])),
    ("IsTruncated", Content::Text),
    ("Name", Content::Text),
    ("Prefix", Content::Text),
    ("Delimiter", Content::Text),
    ("MaxKeys", Content::Text),
    ("EncodingType", Content::Text),
    ("KeyCount", Content::Text),
    ("ContinuationToken", Content::Text),
    ("NextContinuationToken", Content::Text),
    ("StartAfter", Content::Text),
    ("ChecksumAlgorithm", Content::Text),
    ("ChecksumType", Content::Text),
    ("ETag", Content::Text),
    ("Key", Content::Text),
    ("LastModified", Content::Text),
    ("Size", Content::Text),
    ("StorageClass", Content::Text),
    ("DisplayName", Content::Text),
    ("ID", Content::Text),
    ("IsRestoreInProgress", Content::Text),
    ("RestoreExpiryDate", Content::Text),
];

/// The one attribute the real response carries: the default namespace on
/// the root, with the one value the response syntax gives it.  Every other
/// element in the schema declares none.
///
/// The value is part of the declaration.  An attribute whose name is
/// allow-listed and whose value nothing reads is exactly the region RV6-01
/// carried a roster entry through, and `xmlns="urn:not-s3-at-all"` parsed
/// as a ListObjectsV2 listing besides.
///
/// Being declared here makes it **required** on the root, not merely
/// checked when the root happens to state it (RV7-03).
const LIST_OBJECTS_V2_ROOT_ATTRIBUTES: &[(&str, &str)] =
    &[("xmlns", "http://s3.amazonaws.com/doc/2006-03-01/")];

/// The declared content model of `name`, or `None` if the schema does not
/// describe it.
fn list_objects_v2_content(name: &str) -> Option<&'static Content> {
    LIST_OBJECTS_V2_SCHEMA
        .iter()
        .find(|(element, _)| *element == name)
        .map(|(_, content)| content)
}

/// Parse one `ListObjectsV2` page, or say why it is not one.
///
/// The retired parser returned an `S3ListPage` unconditionally: a body that
/// was empty, an `<Error>` document, a page cut off mid-`<Contents>`, or a
/// truncated page whose token had gone missing all produced a short object
/// list and no error, and the caller stopped paginating on the missing
/// token.  A radar volume selection made from a short roster picks a
/// different volume, or none, and reports success either way.
///
/// So every one of the following is a named refusal of the page:
///
/// * the body is not well-formed XML -- every tag opened is closed, in
///   order, nothing but whitespace lies outside the root element, no
///   comment anywhere, no attribute value carries a raw `<`, attributes are
///   separated by whitespace and no start tag declares one twice, and the
///   declaration is the one XML 1.0 grammar down to its three values
///   ([`scan_xml`]).  This is the check the field checks below cannot
///   make: a page missing only its closing `</ListBucketResult>` states
///   every field correctly and is still a document that arrived short;
/// * the document does not match the ListObjectsV2 content model -- every
///   element's permitted children, recursively, down to the text-only
///   leaves, plus the attributes the response syntax declares, which are
///   required and not merely checked when present
///   ([`XmlDocument::require_list_objects_v2_schema`]).  A name check at
///   the roster-bearing levels is not this: an allow-listed element the
///   parser never descends into is an unchecked subtree, and a real
///   `<Contents>` inside one leaves a closed document, an all-allow-listed
///   set of direct children, and a `<KeyCount>` that agrees with the short
///   roster it produced;
/// * the body is not a `<ListBucketResult>` for the bucket and prefix that
///   were asked for;
/// * `<IsTruncated>` is absent or is not exactly `true`/`false`, without
///   it "no continuation token" cannot be distinguished from "the
///   continuation token was lost", which is the whole finding;
/// * the page says it is truncated and carries no usable
///   `<NextContinuationToken>`, or says it is not and carries one anyway;
/// * `<KeyCount>` is absent, unparseable, larger than `<MaxKeys>`, or does
///   not equal the number of `<Contents>` entries actually parsed: this is
///   the bucket's own count of what it put in the page, and it is the check
///   that catches a body cut short by a dropped connection;
/// * any `<Contents>` element is unterminated, or is missing its `<Key>`,
///   `<Size>` or `<LastModified>`.  A missing size used to default to `0`,
///   which silently disarms the byte-count check in
///   [`download_volume`]: a truncated download would then pass;
/// * a key outside the requested prefix, or a key ending in `/`, which is a
///   directory placeholder rather than an object;
/// * a key that cannot be localized -- one carrying a `..` or `.` or empty
///   path segment, a leading `/`, a backslash or a colon
///   ([`key_localization_fault`]).  A key is turned into a local cache path
///   one segment at a time, so `2026/07/29/KTLX/../../../../elsewhere/KTLX…`
///   is a key that starts with the requested prefix, ends in a parseable
///   volume name, and writes its bytes outside the cache root (RV8-PATH);
/// * a `<CommonPrefixes>` element, which this client never asks for (it
///   sends no delimiter) and which, in a delimited listing, is exactly
///   where the objects it did not return went.
fn parse_s3_list_xml(
    xml: &str,
    bucket: &str,
    prefix: &str,
) -> Result<S3ListPage, Box<dyn Error>> {
    parse_s3_list_xml_with(xml, bucket, prefix, None)
}

/// [`parse_s3_list_xml`], plus the one case where a rolled-up page is an
/// answer instead of a fault: the request sent `delimiter`.
///
/// The roll-up refusal is about a page that hides objects *from a client
/// that asked for all of them*.  A delimited request asks for the roll-up
/// and reads nothing else, so for it the refusal inverts: a page that does
/// not echo the delimiter that was sent is a listing of some other key
/// space, exactly as a page echoing the wrong prefix is.
fn parse_s3_list_xml_with(
    xml: &str,
    bucket: &str,
    prefix: &str,
    delimiter: Option<&str>,
) -> Result<S3ListPage, Box<dyn Error>> {
    // The document's shape first.  Every field check below is a statement
    // about a value; this is the one statement about the page itself, and
    // it is the one the retired parser never made.
    let document = scan_xml(xml).map_err(|problem| {
        boxed_error(format!(
            "S3 did not return well-formed XML for {bucket}/{prefix}: \
             {problem}. {} bytes of a document that does not close is not a \
             roster: the objects that were cut off with the closing tag are \
             exactly the ones a short roster silently omits",
            xml.len()
        ))
    })?;
    let root = match document.roots.as_slice() {
        [only] if document.elements[*only].name == "ListBucketResult" => *only,
        roots => {
            let code = roots
                .first()
                .and_then(|&index| document.child_text(index, "Code").ok().flatten())
                .unwrap_or_default();
            let message = roots
                .first()
                .and_then(|&index| document.child_text(index, "Message").ok().flatten())
                .unwrap_or_default();
            return Err(boxed_error(format!(
                "S3 did not return a ListObjectsV2 listing for {bucket}/{prefix} \
                 (code {code:?}, message {message:?}); {} bytes of body that is \
                 not a <ListBucketResult> is not a roster of zero objects",
                xml.len()
            )));
        }
    };
    // The whole document against the content model, once, before any value
    // is read: every element's permitted children, recursively, down to the
    // text-only leaves, plus the attributes nothing declares.
    document
        .require_list_objects_v2_schema(root)
        .map_err(|problem| boxed_error(format!("S3 listing for {prefix:?}: {problem}")))?;
    let text = |name: &str| -> Result<Option<String>, Box<dyn Error>> {
        document.child_text(root, name).map_err(|problem| {
            boxed_error(format!("S3 listing for {prefix:?}: {problem}"))
        })
    };

    match text("Name")? {
        Some(name) if name == bucket => {}
        other => {
            return Err(boxed_error(format!(
                "S3 listing names bucket {other:?}, this request was for \
                 {bucket:?}"
            )));
        }
    }
    match text("Prefix")? {
        Some(echoed) if echoed == prefix => {}
        other => {
            return Err(boxed_error(format!(
                "S3 listing echoes prefix {other:?}, this request was for \
                 {prefix:?}; a listing of some other key space cannot answer \
                 this one"
            )));
        }
    }
    let rolled_up: Vec<usize> = document
        .children(root)
        .filter(|&index| document.elements[index].name == "CommonPrefixes")
        .collect();
    let mut common_prefixes = Vec::with_capacity(rolled_up.len());
    match delimiter {
        None => {
            if !rolled_up.is_empty() {
                return Err(boxed_error(format!(
                    "S3 listing for {prefix:?} carries <CommonPrefixes>, so it was \
                     rolled up on a delimiter this client never sent. The objects \
                     under those prefixes are not in the page and would be missing \
                     from the roster"
                )));
            }
        }
        Some(sent) => {
            match text("Delimiter")? {
                Some(echoed) if echoed == sent => {}
                other => {
                    return Err(boxed_error(format!(
                        "S3 listing echoes delimiter {other:?}, this request sent \
                         {sent:?}; the roll-up boundary decides which prefixes the \
                         page reports and which objects it withholds, so a page \
                         rolled up on some other boundary does not answer this \
                         request"
                    )));
                }
            }
            for entry in rolled_up {
                let rolled = document
                    .child_text(entry, "Prefix")
                    .map_err(|problem| {
                        boxed_error(format!("S3 listing for {prefix:?}: {problem}"))
                    })?
                    .filter(|value| !value.is_empty())
                    .ok_or_else(|| {
                        boxed_error(format!(
                            "S3 listing for {prefix:?} has a <CommonPrefixes> entry \
                             with no <Prefix>"
                        ))
                    })?;
                if !rolled.starts_with(prefix) {
                    return Err(boxed_error(format!(
                        "S3 listing for {prefix:?} rolled up {rolled:?}, which is \
                         outside the prefix that was requested"
                    )));
                }
                common_prefixes.push(rolled);
            }
        }
    }

    let is_truncated = match text("IsTruncated")?.as_deref() {
        Some("true") => true,
        Some("false") => false,
        other => {
            return Err(boxed_error(format!(
                "S3 listing for {prefix:?} declares IsTruncated {other:?}; \
                 without an explicit true/false there is no way to tell a \
                 complete page from one whose continuation token went \
                 missing, and the difference is silently-dropped volumes"
            )));
        }
    };
    let token = text("NextContinuationToken")?.filter(|value| !value.trim().is_empty());
    match (is_truncated, &token) {
        (true, None) => {
            return Err(boxed_error(format!(
                "S3 listing for {prefix:?} says it is truncated but carries \
                 no usable NextContinuationToken; the rest of the objects \
                 are unreachable and stopping here would report a short \
                 roster as a complete one"
            )));
        }
        (false, Some(value)) => {
            return Err(boxed_error(format!(
                "S3 listing for {prefix:?} says it is complete and still \
                 carries NextContinuationToken {value:?}; the page \
                 contradicts itself about whether more objects exist"
            )));
        }
        _ => {}
    }

    let mut objects = Vec::new();
    let roster: Vec<usize> = document
        .children(root)
        .filter(|&index| document.elements[index].name == "Contents")
        .collect();
    for entry in roster {
        let field = |name: &str| -> Result<Option<String>, Box<dyn Error>> {
            document.child_text(entry, name).map_err(|problem| {
                boxed_error(format!("S3 listing for {prefix:?}: {problem}"))
            })
        };
        let key = field("Key")?
            .filter(|value| !value.is_empty())
            .ok_or_else(|| {
                boxed_error(format!(
                    "S3 listing for {prefix:?} has a <Contents> entry with no \
                     <Key>"
                ))
            })?;
        if !key.starts_with(prefix) {
            return Err(boxed_error(format!(
                "S3 listing for {prefix:?} returned key {key:?}, which is \
                 outside the prefix that was requested"
            )));
        }
        if key.ends_with('/') {
            return Err(boxed_error(format!(
                "S3 listing for {prefix:?} returned key {key:?}, a directory \
                 placeholder rather than an object. A prefix counted as an \
                 object is a zero-byte entry in a roster of volumes"
            )));
        }
        if let Some(fault) = key_localization_fault(&key) {
            return Err(boxed_error(format!(
                "S3 listing for {prefix:?} returned key {key:?}, which \
                 {fault}. A listed key is localized one segment at a time \
                 into the cache directory, so a key that names its way out of \
                 that directory is a key this client will not localize: the \
                 bytes it would write land outside the cache root, or in \
                 another bucket's slot inside it, and every other check the \
                 roster makes of a key -- the prefix it starts with, the \
                 volume name it ends with -- passes on one of these"
            )));
        }
        let size_bytes = field("Size")?
            .ok_or_else(|| {
                boxed_error(format!(
                    "S3 listing entry {key:?} carries no <Size>. The listed \
                     size is what the downloaded byte count is checked \
                     against, so defaulting it to zero disarms that check \
                     and lets a truncated object through"
                ))
            })?
            .parse::<u64>()
            .map_err(|error| {
                boxed_error(format!(
                    "S3 listing entry {key:?} declares an unparseable \
                     <Size>: {error}"
                ))
            })?;
        let last_modified = field("LastModified")?
            .filter(|value| !value.trim().is_empty())
            .ok_or_else(|| {
                boxed_error(format!(
                    "S3 listing entry {key:?} carries no <LastModified>; it \
                     travels into the receipt as the object's identity in \
                     time, and an empty string there is a fact nobody stated"
                ))
            })?;
        objects.push(S3Object {
            key,
            size_bytes,
            last_modified,
        });
    }

    let declared = text("KeyCount")?
        .and_then(|value| value.parse::<usize>().ok())
        .ok_or_else(|| {
            boxed_error(format!(
                "S3 listing for {prefix:?} carries no parseable <KeyCount>; \
                 it is the bucket's own count of what it put in this page, \
                 and without it a body that arrived short still parses"
            ))
        })?;
    // `KeyCount` is what the bucket says it put in the page, and in a
    // delimited listing that is the objects *plus* the roll-up prefixes --
    // the 614 volume directories one live site rolls up arrive as
    // KeyCount 614 with zero <Contents>.  Counting only the objects there
    // would refuse every delimited page as short.
    let counted = objects.len() + common_prefixes.len();
    if declared != counted {
        return Err(boxed_error(format!(
            "S3 listing for {prefix:?} declares KeyCount {declared} and \
             carries {} parseable objects and {} rolled-up prefixes; the \
             page is not what it says it is",
            objects.len(),
            common_prefixes.len()
        )));
    }
    let max_keys = text("MaxKeys")?
        .and_then(|value| value.parse::<usize>().ok())
        .ok_or_else(|| {
            boxed_error(format!(
                "S3 listing for {prefix:?} carries no parseable <MaxKeys>"
            ))
        })?;
    if declared > max_keys {
        return Err(boxed_error(format!(
            "S3 listing for {prefix:?} declares KeyCount {declared} over \
             MaxKeys {max_keys}"
        )));
    }

    Ok(S3ListPage {
        objects,
        common_prefixes,
        next_continuation_token: token,
        is_truncated,
    })
}

fn xml_unescape(value: &str) -> Result<String, String> {
    let mut out = String::with_capacity(value.len());
    let mut rest = value;
    while let Some(offset) = rest.find('&') {
        out.push_str(&rest[..offset]);
        let reference = &rest[offset..];
        let (encoded, decoded) = if reference.starts_with("&amp;") {
            ("&amp;", '&')
        } else if reference.starts_with("&lt;") {
            ("&lt;", '<')
        } else if reference.starts_with("&gt;") {
            ("&gt;", '>')
        } else if reference.starts_with("&quot;") {
            ("&quot;", '"')
        } else if reference.starts_with("&apos;") {
            ("&apos;", '\'')
        } else {
            let end = reference.find(';').map_or(reference.len(), |index| index + 1);
            return Err(format!(
                "the document carries undeclared or malformed XML entity \
                 reference {:?}; only &amp;, &lt;, &gt;, &quot; and &apos; \
                 are defined",
                &reference[..end]
            ));
        };
        out.push(decoded);
        rest = &reference[encoded.len()..];
    }
    out.push_str(rest);
    Ok(out)
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
/// connection.  Every phase gets an explicit timeout: ureq 3 defaults them
/// all to `None`, which would let one stalled connection hang the CLI
/// forever instead of failing closed.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
const SEND_REQUEST_TIMEOUT: Duration = Duration::from_secs(30);
const RECV_RESPONSE_TIMEOUT: Duration = Duration::from_secs(60);
/// Generous enough for a ~30 MB super-res volume on a slow link.
const RECV_BODY_TIMEOUT: Duration = Duration::from_secs(600);

/// Build the pure-Rust TLS agent.  The OnceLock guards the process-global
/// provider install so repeated calls never clash.
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

pub fn boxed_error(message: impl Into<String>) -> Box<dyn Error> {
    Box::new(io::Error::new(io::ErrorKind::InvalidData, message.into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- ListObjectsV2 pagination ------------------------------------------
    //
    // The three pages below are the verbatim bodies Amazon S3 returned for
    // `unidata-nexrad-level2` on 2026-07-30, one of each shape the walk can
    // meet: truncated with a token, complete with objects, and complete with
    // none.  Every refusal test is one field of one of them changed, so each
    // says which single statement made the page unusable rather than
    // asserting that hand-written junk fails.  Nothing here is a listing this
    // repository wrote.

    const BUCKET: &str = "unidata-nexrad-level2";
    const DAY_PREFIX: &str = "2026/07/29/KTLX/";
    const ONE_KEY_PREFIX: &str = "2026/07/29/KTLX/KTLX20260729_000234_V06";
    const ABSENT_PREFIX: &str = "2026/07/29/KTLX/KTLX20260729_zzzzzz_V06";

    /// `?list-type=2&prefix=2026/07/29/KTLX/&max-keys=3`
    const REAL_TRUNCATED_PAGE: &str = concat!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n",
        "<ListBucketResult xmlns=\"http://s3.amazonaws.com/doc/2006-03-01/\">",
        "<Name>unidata-nexrad-level2</Name>",
        "<Prefix>2026/07/29/KTLX/</Prefix>",
        "<NextContinuationToken>1BcoaPVA7MopiM/SNQZZzmf+oSSKS3WiZRWr3PZcQnRQguCRWYUXk74M65UTNKIVs1CXKGIcsmtm88ViHGuPYhFyjMap2ZkWD</NextContinuationToken>",
        "<KeyCount>3</KeyCount><MaxKeys>3</MaxKeys><IsTruncated>true</IsTruncated>",
        "<Contents><Key>2026/07/29/KTLX/KTLX20260729_000234_V06</Key>",
        "<LastModified>2026-07-29T00:09:29.000Z</LastModified>",
        "<ETag>&quot;a88362ade3543dbc9a273bf213db51a6&quot;</ETag>",
        "<ChecksumAlgorithm>CRC32</ChecksumAlgorithm><ChecksumType>FULL_OBJECT</ChecksumType>",
        "<Size>5062263</Size><StorageClass>STANDARD</StorageClass></Contents>",
        "<Contents><Key>2026/07/29/KTLX/KTLX20260729_000936_V06</Key>",
        "<LastModified>2026-07-29T00:16:31.000Z</LastModified>",
        "<ETag>&quot;e64cb4a87cbe7471b9a9009fa072d3f2&quot;</ETag>",
        "<ChecksumAlgorithm>CRC32</ChecksumAlgorithm><ChecksumType>FULL_OBJECT</ChecksumType>",
        "<Size>4845918</Size><StorageClass>STANDARD</StorageClass></Contents>",
        "<Contents><Key>2026/07/29/KTLX/KTLX20260729_001638_V06</Key>",
        "<LastModified>2026-07-29T00:35:08.000Z</LastModified>",
        "<ETag>&quot;41f69fac60f59fcec50e28254bf30b85&quot;</ETag>",
        "<ChecksumAlgorithm>CRC32</ChecksumAlgorithm><ChecksumType>FULL_OBJECT</ChecksumType>",
        "<Size>4624604</Size><StorageClass>STANDARD</StorageClass></Contents>",
        "</ListBucketResult>"
    );

    /// The same bucket, a prefix that names exactly one object: a complete
    /// page, so no continuation token at all.
    const REAL_FINAL_PAGE: &str = concat!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n",
        "<ListBucketResult xmlns=\"http://s3.amazonaws.com/doc/2006-03-01/\">",
        "<Name>unidata-nexrad-level2</Name>",
        "<Prefix>2026/07/29/KTLX/KTLX20260729_000234_V06</Prefix>",
        "<KeyCount>1</KeyCount><MaxKeys>1000</MaxKeys><IsTruncated>false</IsTruncated>",
        "<Contents><Key>2026/07/29/KTLX/KTLX20260729_000234_V06</Key>",
        "<LastModified>2026-07-29T00:09:29.000Z</LastModified>",
        "<ETag>&quot;a88362ade3543dbc9a273bf213db51a6&quot;</ETag>",
        "<ChecksumAlgorithm>CRC32</ChecksumAlgorithm><ChecksumType>FULL_OBJECT</ChecksumType>",
        "<Size>5062263</Size><StorageClass>STANDARD</StorageClass></Contents>",
        "</ListBucketResult>"
    );

    /// A prefix nothing matches.  Zero objects is an answer, not a failure.
    const REAL_EMPTY_PAGE: &str = concat!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n",
        "<ListBucketResult xmlns=\"http://s3.amazonaws.com/doc/2006-03-01/\">",
        "<Name>unidata-nexrad-level2</Name>",
        "<Prefix>2026/07/29/KTLX/KTLX20260729_zzzzzz_V06</Prefix>",
        "<KeyCount>0</KeyCount><MaxKeys>1000</MaxKeys><IsTruncated>false</IsTruncated>",
        "</ListBucketResult>"
    );

    /// What `noaa-nexrad-level2` (the archive of record) actually answers
    /// anonymous `ListObjectsV2` with, verbatim, HTTP 403.
    const REAL_ACCESS_DENIED: &str = concat!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n",
        "<Error><Code>AccessDenied</Code><Message>Access Denied</Message>",
        "<RequestId>FJY11D91H40S29SM</RequestId>",
        "<HostId>fgQxZFklxwt+qAtTgfnM7QhA4SLp9x83nr/Kb5UWQTpQ0mV1MhjCbx3n5AT+pSovF2DLi7mVYNJrcU4q6seKhZKhsv6mDN83</HostId>",
        "</Error>"
    );

    fn day_page(xml: &str) -> Result<S3ListPage, Box<dyn Error>> {
        parse_s3_list_xml(xml, BUCKET, DAY_PREFIX)
    }

    /// Replace the first `from` in the real page with `to`.  Every refusal
    /// below is exactly one such substitution.
    fn edited(xml: &str, from: &str, to: &str) -> String {
        assert!(xml.contains(from), "fixture does not contain {from:?}");
        xml.replacen(from, to, 1)
    }

    #[test]
    fn the_real_pages_parse_to_exactly_what_the_bucket_said() {
        let page = day_page(REAL_TRUNCATED_PAGE).unwrap();
        assert_eq!(page.objects.len(), 3);
        assert_eq!(page.objects[0].key, "2026/07/29/KTLX/KTLX20260729_000234_V06");
        assert_eq!(page.objects[0].size_bytes, 5_062_263);
        assert_eq!(page.objects[0].last_modified, "2026-07-29T00:09:29.000Z");
        assert_eq!(page.objects[2].size_bytes, 4_624_604);
        assert!(page.is_truncated);
        assert!(page
            .next_continuation_token
            .as_deref()
            .unwrap()
            .starts_with("1BcoaPVA7MopiM/SNQZZ"));

        let page = parse_s3_list_xml(REAL_FINAL_PAGE, BUCKET, ONE_KEY_PREFIX).unwrap();
        assert_eq!(page.objects.len(), 1);
        assert!(!page.is_truncated);
        assert_eq!(page.next_continuation_token, None);

        // Zero objects under a prefix is a quiet day, not a refusal.
        let page = parse_s3_list_xml(REAL_EMPTY_PAGE, BUCKET, ABSENT_PREFIX).unwrap();
        assert!(page.objects.is_empty());
        assert!(!page.is_truncated);
        assert_eq!(page.next_continuation_token, None);
    }

    #[test]
    fn a_page_that_will_not_say_whether_it_is_truncated_is_refused() {
        // Absent, empty, and two spellings that are not the two the
        // protocol defines: none of them says whether objects are missing.
        for replacement in [
            "",
            "<IsTruncated></IsTruncated>",
            "<IsTruncated>maybe</IsTruncated>",
            "<IsTruncated>TRUE</IsTruncated>",
            "<IsTruncated>1</IsTruncated>",
        ] {
            let xml = edited(REAL_TRUNCATED_PAGE, "<IsTruncated>true</IsTruncated>", replacement);
            let err = day_page(&xml).unwrap_err().to_string();
            assert!(err.contains("IsTruncated"), "{replacement:?}: {err}");
        }
    }

    #[test]
    fn a_truncated_page_whose_token_went_missing_is_refused() {
        let token_element = REAL_TRUNCATED_PAGE
            .split("<NextContinuationToken>")
            .nth(1)
            .and_then(|rest| rest.split_once("</NextContinuationToken>"))
            .map(|(value, _)| format!("<NextContinuationToken>{value}</NextContinuationToken>"))
            .unwrap();
        // Gone entirely, present but empty, present but whitespace: the
        // retired parser read all three as "the listing ended here".
        for replacement in [
            "".to_string(),
            "<NextContinuationToken></NextContinuationToken>".to_string(),
            "<NextContinuationToken>   </NextContinuationToken>".to_string(),
        ] {
            let xml = edited(REAL_TRUNCATED_PAGE, &token_element, &replacement);
            let err = day_page(&xml).unwrap_err().to_string();
            assert!(err.contains("truncated"), "{replacement:?}: {err}");
            assert!(err.contains("NextContinuationToken"), "{err}");
        }
    }

    #[test]
    fn a_complete_page_that_still_carries_a_token_contradicts_itself() {
        for token in ["abc", "1BcoaPVA7MopiM"] {
            let xml = edited(
                REAL_FINAL_PAGE,
                "<KeyCount>1</KeyCount>",
                &format!("<NextContinuationToken>{token}</NextContinuationToken><KeyCount>1</KeyCount>"),
            );
            let err = parse_s3_list_xml(&xml, BUCKET, ONE_KEY_PREFIX)
                .unwrap_err()
                .to_string();
            assert!(err.contains("contradicts itself"), "{err}");
        }
    }

    #[test]
    fn a_body_that_arrived_short_is_refused_rather_than_partly_used() {
        // A dropped connection cuts the body at an arbitrary byte.  Two
        // cuts: one inside the second entry, one after it but before the
        // page's own KeyCount could be contradicted by anything else.
        for cut in [900usize, 1200] {
            let xml = &REAL_TRUNCATED_PAGE[..cut];
            let err = day_page(xml).unwrap_err().to_string();
            assert!(
                err.contains("unterminated <Contents>") || err.contains("KeyCount"),
                "cut at {cut}: {err}"
            );
        }
    }

    #[test]
    fn a_page_whose_root_never_closes_is_refused() {
        // Re-verification's probe, verbatim: remove ONLY the closing
        // </ListBucketResult> from a real page.  Every field the parser
        // reads is still present and still correct -- KeyCount, the
        // contents, IsTruncated=false -- and the retired parser returned
        // Ok with its object, because "the opening substring exists" was
        // the whole of what it checked about the document.
        //
        // All three real pages, because the hole is in the document check
        // and not in any one page's contents: the empty page has no
        // roster to lose, and the truncated page's roster is three.
        for (label, page, prefix) in [
            ("truncated", REAL_TRUNCATED_PAGE, DAY_PREFIX),
            ("final", REAL_FINAL_PAGE, ONE_KEY_PREFIX),
            ("empty", REAL_EMPTY_PAGE, ABSENT_PREFIX),
        ] {
            // The control: unaltered, it parses.
            parse_s3_list_xml(page, BUCKET, prefix)
                .unwrap_or_else(|error| panic!("{label} page: {error}"));
            let xml = edited(page, "</ListBucketResult>", "");
            let err = parse_s3_list_xml(&xml, BUCKET, prefix)
                .unwrap_err()
                .to_string();
            assert!(err.contains("well-formed XML"), "{label}: {err}");
            assert!(
                err.contains("unterminated <ListBucketResult> element"),
                "{label}: {err}"
            );
        }
    }

    #[test]
    fn no_prefix_of_a_real_page_parses_as_a_page() {
        // The class, not the instance.  A dropped connection cuts the body
        // at an ARBITRARY byte, so the property to hold is that no proper
        // prefix of a real listing is ever a listing -- inside a roster
        // element, between two of them, after the last one, and after
        // every scalar field alike.  Exhaustive over all three real pages
        // rather than at hand-picked offsets, because the hand-picked
        // offsets are how a cut after the final </Contents> went unnoticed.
        for (label, page, prefix) in [
            ("truncated", REAL_TRUNCATED_PAGE, DAY_PREFIX),
            ("final", REAL_FINAL_PAGE, ONE_KEY_PREFIX),
            ("empty", REAL_EMPTY_PAGE, ABSENT_PREFIX),
        ] {
            for cut in 0..page.len() {
                if !page.is_char_boundary(cut) {
                    continue;
                }
                assert!(
                    parse_s3_list_xml(&page[..cut], BUCKET, prefix).is_err(),
                    "{label} page cut at {cut} parsed as a complete listing"
                );
            }
            // And the whole of it still does parse, so the sweep above is
            // not passing because the fixture is unparseable to begin with.
            assert!(parse_s3_list_xml(page, BUCKET, prefix).is_ok(), "{label}");
        }
    }

    #[test]
    fn a_document_that_does_not_nest_is_refused() {
        // Three ways to be malformed that a substring search cannot see:
        // a close tag naming the wrong element, a stray close with nothing
        // open, and content after the root element has ended.
        for (edit, from, to) in [
            ("mismatched close", "</Contents>", "</Content>"),
            ("stray close", "<KeyCount>3</KeyCount>", "</Nothing><KeyCount>3</KeyCount>"),
            ("trailing junk", "</ListBucketResult>", "</ListBucketResult><Contents/>"),
        ] {
            let xml = edited(REAL_TRUNCATED_PAGE, from, to);
            let err = day_page(&xml).unwrap_err().to_string();
            assert!(err.contains("well-formed XML"), "{edit}: {err}");
        }
    }

    #[test]
    fn a_page_that_states_one_field_twice_is_refused() {
        // Reading a field by searching the buffer takes the first match and
        // says nothing about the second.  Two KeyCounts, or two IsTruncateds,
        // are two different claims about the same page; picking one is a
        // decision this client has no basis for.
        for (from, to) in [
            ("<KeyCount>3</KeyCount>", "<KeyCount>3</KeyCount><KeyCount>9</KeyCount>"),
            (
                "<IsTruncated>true</IsTruncated>",
                "<IsTruncated>true</IsTruncated><IsTruncated>false</IsTruncated>",
            ),
        ] {
            let xml = edited(REAL_TRUNCATED_PAGE, from, to);
            let err = day_page(&xml).unwrap_err().to_string();
            assert!(err.contains("more than once"), "{to}: {err}");
        }
    }

    #[test]
    fn unexpected_direct_children_at_roster_levels_are_refused() {
        let xml = edited(
            REAL_TRUNCATED_PAGE,
            "<KeyCount>3</KeyCount>",
            "<Unexpected><Payload>not-s3</Payload></Unexpected><KeyCount>3</KeyCount>",
        );
        let err = day_page(&xml).unwrap_err().to_string();
        assert!(err.contains("unexpected <Unexpected>"), "{err}");
        assert!(err.contains("<ListBucketResult>"), "{err}");

        let xml = edited(
            REAL_TRUNCATED_PAGE,
            "<LastModified>2026-07-29T00:09:29.000Z</LastModified>",
            "<LastModified>2026-07-29T00:09:29.000Z</LastModified><Unexpected>not-s3</Unexpected>",
        );
        let err = day_page(&xml).unwrap_err().to_string();
        assert!(err.contains("unexpected <Unexpected>"), "{err}");
        assert!(err.contains("<Contents>"), "{err}");
    }

    #[test]
    fn an_unknown_wrapper_cannot_hide_contents_behind_an_adjusted_count() {
        let xml = edited(
            REAL_TRUNCATED_PAGE,
            "<Contents><Key>2026/07/29/KTLX/KTLX20260729_000936_V06</Key>",
            "<Unexpected><Contents><Key>2026/07/29/KTLX/KTLX20260729_000936_V06</Key>",
        );
        let xml = edited(
            &xml,
            "<Size>4845918</Size><StorageClass>STANDARD</StorageClass></Contents>",
            "<Size>4845918</Size><StorageClass>STANDARD</StorageClass></Contents></Unexpected>",
        );
        let xml = edited(&xml, "<KeyCount>3</KeyCount>", "<KeyCount>2</KeyCount>");
        let err = day_page(&xml).unwrap_err().to_string();
        assert!(err.contains("unexpected <Unexpected>"), "{err}");
        assert!(err.contains("<ListBucketResult>"), "{err}");
    }

    // -- the content model, element by element ----------------------------
    //
    // Re-verification #5's census, made permanent and made a sweep.  The
    // first repair checked direct-child NAMES at two levels and called that
    // the element schema; a name it allowed but never descended into was an
    // unchecked subtree, and ten of them each carried a real <Contents> out
    // of the roster while the document closed cleanly, every direct child
    // stayed allow-listed, and <KeyCount> agreed with the short roster.
    //
    // These three sweeps are generated FROM the schema table rather than
    // written out by hand, so a name added to a child list without a content
    // model -- or with one that leaves a hiding place, a second statement of
    // itself, or an attribute nobody reads -- fails here rather than in a
    // listing.

    /// The second real roster entry, verbatim, as one movable block.
    const HIDDEN_ENTRY: &str = concat!(
        "<Contents><Key>2026/07/29/KTLX/KTLX20260729_000936_V06</Key>",
        "<LastModified>2026-07-29T00:16:31.000Z</LastModified>",
        "<ETag>&quot;e64cb4a87cbe7471b9a9009fa072d3f2&quot;</ETag>",
        "<ChecksumAlgorithm>CRC32</ChecksumAlgorithm><ChecksumType>FULL_OBJECT</ChecksumType>",
        "<Size>4845918</Size><StorageClass>STANDARD</StorageClass></Contents>"
    );

    /// The fields that make a `<Contents>` a syntactically complete roster
    /// entry that is not a volume: the 1-byte decoy the census watched a
    /// 4.8 MB volume be swapped for.
    const DECOY_FIELDS: &[(&str, &str)] = &[
        ("Key", "2026/07/29/KTLX/decoy_V06"),
        ("LastModified", "2026-07-29T00:20:00.000Z"),
        ("Size", "1"),
    ];

    /// Every path from the root down through the schema, as a chain of
    /// element names.  Generated from the table, so it grows with it.
    fn schema_paths() -> Vec<Vec<&'static str>> {
        fn walk(
            name: &str,
            prefix: &mut Vec<&'static str>,
            out: &mut Vec<Vec<&'static str>>,
        ) {
            let Some(Content::Children(children)) = list_objects_v2_content(name) else {
                return;
            };
            for (child, _) in children.iter() {
                prefix.push(child);
                out.push(prefix.clone());
                if !prefix[..prefix.len() - 1].contains(child) {
                    walk(child, prefix, out);
                }
                prefix.pop();
            }
        }
        let mut out = Vec::new();
        walk("ListBucketResult", &mut Vec::new(), &mut out);
        out
    }

    /// `payload` wrapped in the element chain `path`, ready to be inserted
    /// as a direct child of `<ListBucketResult>`.  Wherever the chain passes
    /// through `<Contents>`, the entry is filled in so it is a countable
    /// roster entry rather than an entry with no key -- the census's decoy.
    fn nested(path: &[&str], payload: &str) -> String {
        let mut nest = String::new();
        for (depth, name) in path.iter().enumerate() {
            nest.push_str(&format!("<{name}>"));
            if *name == "Contents" {
                // Every decoy field except the one the chain continues
                // into, so the mutation is the wrapping and not an entry
                // that states its own <Key> twice.
                let next = path.get(depth + 1).copied().unwrap_or("");
                for (field, value) in DECOY_FIELDS {
                    if *field != next {
                        nest.push_str(&format!("<{field}>{value}</{field}>"));
                    }
                }
            }
        }
        nest.push_str(payload);
        for name in path.iter().rev() {
            nest.push_str(&format!("</{name}>"));
        }
        nest
    }

    /// `base` with `nest` inserted as a direct child of the root and the
    /// page's own `<KeyCount>` set to `count`.
    fn page_with(base: &str, nest: &str, count: usize) -> String {
        edited(
            base,
            "<KeyCount>3</KeyCount>",
            &format!("{nest}<KeyCount>{count}</KeyCount>"),
        )
    }

    /// `base` with its own first `<name>...</name>` taken out, or `base`
    /// unchanged when it states none.
    ///
    /// The census wraps one real entry inside each schema element in turn,
    /// and six of the root's children are already stated by the real page.
    /// Without this the mutation would be "the page states <Name> twice",
    /// which is a different refusal and would leave the wrapping untested.
    fn without_root_element(base: &str, name: &str) -> String {
        let open = format!("<{name}>");
        let close = format!("</{name}>");
        match (base.find(&open), base.find(&close)) {
            (Some(start), Some(end)) if end > start => {
                format!("{}{}", &base[..start], &base[end + close.len()..])
            }
            _ => base.to_string(),
        }
    }

    #[test]
    fn every_schema_element_declares_a_content_model() {
        let mut declared: Vec<&str> = Vec::new();
        for (name, _) in LIST_OBJECTS_V2_SCHEMA {
            assert!(
                !declared.contains(name),
                "<{name}> is declared twice in the schema table"
            );
            declared.push(name);
        }
        for (parent, content) in LIST_OBJECTS_V2_SCHEMA {
            let Content::Children(children) = content else {
                continue;
            };
            for (child, _) in children.iter() {
                assert!(
                    list_objects_v2_content(child).is_some(),
                    "<{child}>, permitted inside <{parent}>, has no content \
                     model; an element the walk cannot describe is an element \
                     a <Contents> entry can be hidden inside"
                );
            }
        }
        // And nothing in the table is a model the walk never reaches.
        let reachable: Vec<&str> = schema_paths()
            .into_iter()
            .map(|path| path[path.len() - 1])
            .collect();
        for (name, _) in LIST_OBJECTS_V2_SCHEMA {
            assert!(
                *name == "ListBucketResult" || reachable.contains(name),
                "<{name}> is declared but unreachable from the root"
            );
        }
    }

    #[test]
    fn no_schema_element_can_hide_a_roster_entry() {
        // The control: unaltered, the real page is a three-object roster and
        // the entry the census moves is in it.
        assert_eq!(day_page(REAL_TRUNCATED_PAGE).unwrap().objects.len(), 3);
        assert!(REAL_TRUNCATED_PAGE.contains(HIDDEN_ENTRY));

        let paths = schema_paths();
        assert_eq!(
            paths.len(),
            26,
            "the census covers every element the schema declares"
        );
        let short = edited(REAL_TRUNCATED_PAGE, HIDDEN_ENTRY, "");
        for path in &paths {
            // A wrapper that is itself a <Contents> is a counted roster
            // entry, so the page's count needs no adjusting -- that is the
            // decoy-substitution shape, where a 4.8 MB volume is replaced by
            // a 1-byte entry and <KeyCount>3</KeyCount> still holds.  Every
            // other wrapper shortens the roster by one, and the count is
            // lowered to agree: the short-roster shape.
            let count = if path[0] == "Contents" { 3 } else { 2 };
            let mut base = if path[0] == "Contents" {
                short.clone()
            } else {
                without_root_element(&short, path[0])
            };
            if base.contains("<KeyCount>3</KeyCount>") {
                base = edited(
                    &base,
                    "<KeyCount>3</KeyCount>",
                    &format!("<KeyCount>{count}</KeyCount>"),
                );
            }
            let xml = edited(
                &base,
                "</ListBucketResult>",
                &format!("{}</ListBucketResult>", nested(path, HIDDEN_ENTRY)),
            );
            let label = path.join("/");
            // The whole point of the census: these documents are WELL-FORMED
            // XML made of nothing but schema element names.  Only the content
            // model refuses them.
            assert!(
                scan_xml(&xml).is_ok(),
                "{label}: the mutation must be well-formed, or it proves \
                 nothing about the schema"
            );
            let parent = path[path.len() - 1];
            let err = day_page(&xml).unwrap_err().to_string();
            assert!(
                err.contains(&format!(
                    "unexpected <Contents> direct child of <{parent}>"
                )),
                "{label}: {err}"
            );
        }
    }

    #[test]
    fn every_child_stated_twice_is_judged_by_its_declared_multiplicity() {
        // Duplicate detection used to exist only for the six fields the
        // parser reads, so two <StorageClass> passed where two <Size>
        // refused.  It is a property of the schema now, not of the reader.
        //
        // Round 4 swept the at-most-once children and SKIPPED the repeating
        // ones, which meant the sweep read its multiplicities back out of
        // the table it was testing and could not see a wrong one.  RV6-02 is
        // that gap: <ChecksumAlgorithm> was transcribed at-most-once against
        // an AWS array, so a legitimate multi-checksum object hard-refused a
        // whole day's roster and this test PINNED it.  So both arms are
        // exercised now -- a repeating element stated twice is the response
        // syntax working, and whatever else such a page may be refused for,
        // it is never for saying it twice.
        let mut at_most_once = 0usize;
        let mut repeated = 0usize;
        for path in schema_paths() {
            let name = path[path.len() - 1];
            let chain = &path[..path.len() - 1];
            let parent = if chain.is_empty() {
                "ListBucketResult"
            } else {
                chain[chain.len() - 1]
            };
            let Some(Content::Children(children)) = list_objects_v2_content(parent) else {
                panic!("<{parent}> must be a container to have <{name}> in it");
            };
            let (_, occurs) = *children.iter().find(|(n, _)| *n == name).unwrap();
            let twice = format!("<{name}></{name}><{name}></{name}>");
            let count = if chain.first() == Some(&"Contents") { 4 } else { 3 };
            let xml = page_with(REAL_TRUNCATED_PAGE, &nested(chain, &twice), count);
            let label = path.join("/");
            assert!(scan_xml(&xml).is_ok(), "{label}: not well-formed");
            let duplicate = format!(
                "declares <{name}> more than once in one <{parent}> element"
            );
            match occurs {
                Occurs::Once => {
                    let err = day_page(&xml).unwrap_err().to_string();
                    assert!(err.contains(&duplicate), "{label}: {err}");
                    at_most_once += 1;
                }
                Occurs::Repeated => {
                    // Two empty <Contents> are two entries with no <Key> and
                    // two <CommonPrefixes> are still a delimited roll-up, so
                    // these pages have their own faults.  The one thing they
                    // must never be refused for is repeating.
                    if let Err(error) = day_page(&xml) {
                        let err = error.to_string();
                        assert!(!err.contains(&duplicate), "{label}: {err}");
                    }
                    repeated += 1;
                }
            }
        }
        assert_eq!(
            at_most_once, 23,
            "every element the response syntax states once"
        );
        assert_eq!(
            repeated, 3,
            "<Contents>, <CommonPrefixes> and <ChecksumAlgorithm> are the \
             three the response syntax prints with the repetition ellipsis"
        );

        // The repeating arm, positively: a real page that states <Contents>
        // twice is a two-object roster and both are read.  On the final
        // page, whose <MaxKeys> is the bucket default rather than the
        // three-key request, so the count can rise.
        let start = REAL_FINAL_PAGE.find("<Contents>").unwrap();
        let end = REAL_FINAL_PAGE.find("</Contents>").unwrap() + "</Contents>".len();
        let entry = &REAL_FINAL_PAGE[start..end];
        let xml = edited(REAL_FINAL_PAGE, entry, &format!("{entry}{entry}"));
        let xml = edited(&xml, "<KeyCount>1</KeyCount>", "<KeyCount>2</KeyCount>");
        let page = parse_s3_list_xml(&xml, BUCKET, ONE_KEY_PREFIX).unwrap();
        assert_eq!(page.objects.len(), 2);
        assert_eq!(page.objects[0], page.objects[1]);
    }

    #[test]
    fn an_object_carrying_more_than_one_checksum_algorithm_is_one_object() {
        // RV6-02.  AWS documents `Object.ChecksumAlgorithm` as `Type: Array
        // of strings` with five valid values, and prints it in the response
        // syntax with the repetition ellipsis.  Round 4 transcribed it as
        // at-most-once, so this page -- a legitimate one -- refused, and
        // `list_s3_objects` propagates a page refusal to the whole day.
        let control = day_page(REAL_TRUNCATED_PAGE).unwrap();
        assert_eq!(control.objects.len(), 3);

        let two = edited(
            REAL_TRUNCATED_PAGE,
            "<ChecksumAlgorithm>CRC32</ChecksumAlgorithm>",
            "<ChecksumAlgorithm>CRC32</ChecksumAlgorithm>\
             <ChecksumAlgorithm>SHA256</ChecksumAlgorithm>",
        );
        let page = day_page(&two).unwrap();
        // One object, and the same one: the algorithms are declared content
        // this client reads none of, so a second cannot change the entry.
        assert_eq!(page.objects, control.objects);

        // Every value AWS lists, on one object, all five at once.
        let all = edited(
            REAL_TRUNCATED_PAGE,
            "<ChecksumAlgorithm>CRC32</ChecksumAlgorithm>",
            &["CRC32", "CRC32C", "SHA1", "SHA256", "CRC64NVME"]
                .map(|value| format!("<ChecksumAlgorithm>{value}</ChecksumAlgorithm>"))
                .join(""),
        );
        assert_eq!(day_page(&all).unwrap().objects, control.objects);

        // And the neighbouring field is unaffected: <ChecksumType> is a
        // single string in the same data type and still refuses a second.
        let xml = edited(
            REAL_TRUNCATED_PAGE,
            "<ChecksumType>FULL_OBJECT</ChecksumType>",
            "<ChecksumType>FULL_OBJECT</ChecksumType>\
             <ChecksumType>COMPOSITE</ChecksumType>",
        );
        let err = day_page(&xml).unwrap_err().to_string();
        assert!(
            err.contains("declares <ChecksumType> more than once in one <Contents> element"),
            "{err}"
        );
    }

    #[test]
    fn no_schema_element_carries_an_attribute_this_client_does_not_read() {
        // The control: the real page's one attribute is the root namespace,
        // and it parses.
        assert!(REAL_TRUNCATED_PAGE.contains(ROOT_TAG));
        assert!(day_page(REAL_TRUNCATED_PAGE).is_ok());

        for path in schema_paths() {
            let name = path[path.len() - 1];
            let chain = &path[..path.len() - 1];
            let nest = nested(chain, &format!("<{name} deleted=\"true\"></{name}>"));
            let count = if chain.first() == Some(&"Contents") { 4 } else { 3 };
            let xml = page_with(REAL_TRUNCATED_PAGE, &nest, count);
            let label = path.join("/");
            assert!(scan_xml(&xml).is_ok(), "{label}: not well-formed");
            let err = day_page(&xml).unwrap_err().to_string();
            assert!(
                err.contains(&format!("unexpected attribute \"deleted\" on <{name}>")),
                "{label}: {err}"
            );
        }

        // Not even on the root, where the one declared attribute lives.
        let xml = edited(
            REAL_TRUNCATED_PAGE,
            ROOT_TAG,
            "<ListBucketResult xmlns=\"http://s3.amazonaws.com/doc/2006-03-01/\" deleted=\"true\">",
        );
        let err = day_page(&xml).unwrap_err().to_string();
        assert!(
            err.contains("unexpected attribute \"deleted\" on <ListBucketResult>"),
            "{err}"
        );

        // The audit's probe, verbatim: a semantically loaded attribute on a
        // real entry, which used to parse as a volume.
        let xml = edited(
            REAL_TRUNCATED_PAGE,
            "<Contents><Key>2026/07/29/KTLX/KTLX20260729_000234_V06</Key>",
            "<Contents deleted=\"true\"><Key>2026/07/29/KTLX/KTLX20260729_000234_V06</Key>",
        );
        let err = day_page(&xml).unwrap_err().to_string();
        assert!(
            err.contains("unexpected attribute \"deleted\" on <Contents>"),
            "{err}"
        );
        assert!(err.contains("becomes a volume in the roster"), "{err}");

        // An attribute region that cannot be parsed is a start tag this
        // client cannot claim to have read.
        for (tag, expected) in [
            ("<Contents deleted=true>", "unquoted value"),
            ("<Contents deleted>", "with no value"),
            // A quote that never closes swallows the `>` as well, because
            // xml_tag_end honours quotes; the tag then never ends.
            ("<Contents deleted=\"true>", "unterminated tag"),
        ] {
            let xml = edited(REAL_TRUNCATED_PAGE, "<Contents>", tag);
            let err = day_page(&xml).unwrap_err().to_string();
            assert!(err.contains(expected), "{tag}: {err}");
        }
        // The remaining branch of the attribute reader, directly: the tag
        // scanner keeps a document with an unbalanced quote from reaching it,
        // so it is reachable only on its own terms.  Values come back with
        // it now, entity-decoded, because a name that is allow-listed and a
        // value that nothing reads is the region RV6-01 used.
        assert_eq!(
            xml_attributes(" xmlns=\"a\" b='c&amp;d'").unwrap(),
            [
                ("xmlns".to_string(), "a".to_string()),
                ("b".to_string(), "c&d".to_string()),
            ]
        );
        assert!(xml_attributes(" a=\"b")
            .unwrap_err()
            .contains("never closes its quote"));
        assert!(xml_attributes(" =\"b\"")
            .unwrap_err()
            .contains("attribute with no name"));

        // RV7-04a.  XML separates one attribute from the last with S, and
        // round 5's reader did not, so `xmlns="..."deleted="true"` was read
        // as two attributes and left to the schema one layer later.  It is a
        // well-formedness fault and it is named as one, before the schema is
        // consulted at all.
        let xml = edited(
            REAL_TRUNCATED_PAGE,
            ROOT_TAG,
            "<ListBucketResult xmlns=\"http://s3.amazonaws.com/doc/2006-03-01/\"\
             deleted=\"true\">",
        );
        let problem = scan_xml(&xml).unwrap_err();
        assert!(
            problem.contains("attribute \"deleted\" with no whitespace before it"),
            "{problem}"
        );
        let err = day_page(&xml).unwrap_err().to_string();
        assert!(err.contains("did not return well-formed XML"), "{err}");
        assert!(xml_attributes(" a=\"b\"c=\"d\"")
            .unwrap_err()
            .contains("no whitespace before it"));

        // RV7-04b.  The value the schema compares is the XML 1.0 §3.3.3
        // NORMALISED value: a literal tab, line feed or carriage return is
        // one space, and a literal CRLF is one space rather than two,
        // because §2.11 folds the pair before §3.3.3 sees it.  It changes no
        // verdict for the namespace -- a space is still not the character it
        // replaced, so a line-wrapped namespace is still not the declared
        // one -- but the comparison is the one the standard defines rather
        // than one that happens to agree with it.
        assert_eq!(
            xml_attributes(" a=\"x\ty\nz\r\nw\rv\"").unwrap(),
            [("a".to_string(), "x y z w v".to_string())]
        );
        let xml = edited(
            REAL_TRUNCATED_PAGE,
            ROOT_TAG,
            "<ListBucketResult xmlns=\"http://s3.amazonaws.com/doc/\n2006-03-01/\">",
        );
        let err = day_page(&xml).unwrap_err().to_string();
        assert!(
            err.contains("declares \"http://s3.amazonaws.com/doc/ 2006-03-01/\""),
            "{err}"
        );

        // And the raw pairs are raw: the declaration reads them undecoded,
        // because XML 1.0 permits no reference in a declaration value.
        assert_eq!(
            xml_attribute_pairs(" a=\"c&amp;d\"").unwrap(),
            [("a", "c&amp;d")]
        );
    }

    /// The real page's root start tag, verbatim: the one element in the
    /// response syntax that declares an attribute at all.
    const ROOT_TAG: &str =
        "<ListBucketResult xmlns=\"http://s3.amazonaws.com/doc/2006-03-01/\">";

    #[test]
    fn no_attribute_value_can_carry_a_roster_entry() {
        // RV6-01, and the census axis that could not see it.  Round 4's 76
        // generated mutations all mutate element names and attribute NAMES;
        // none mutates an attribute VALUE, which is the one region of the
        // document that is neither read nor bounded.  This sweeps it: the
        // real second roster entry, verbatim, planted inside an attribute
        // value at every position the schema declares an element -- under
        // the one declared attribute name and under an undeclared one -- and
        // on the root's own `xmlns`, which is where the audit put it.
        //
        // Every one is refused by scan_xml, BEFORE any schema question is
        // asked, because a raw '<' in an AttValue is not well-formed XML.
        // That is what kills the class rather than the reproduction: a value
        // can never carry markup, so there is no position left to plant one.
        assert!(REAL_TRUNCATED_PAGE.contains(HIDDEN_ENTRY));
        assert!(day_page(REAL_TRUNCATED_PAGE).is_ok());

        let mut swept = 0usize;
        for path in schema_paths() {
            let name = path[path.len() - 1];
            let chain = &path[..path.len() - 1];
            let count = if chain.first() == Some(&"Contents") { 4 } else { 3 };
            for attribute in ["xmlns", "deleted"] {
                let nest = nested(
                    chain,
                    &format!("<{name} {attribute}=\"{HIDDEN_ENTRY}\"></{name}>"),
                );
                let xml = page_with(REAL_TRUNCATED_PAGE, &nest, count);
                let label = format!("{}@{attribute}", path.join("/"));
                let problem = scan_xml(&xml).unwrap_err();
                assert!(
                    problem.contains(&format!("value of attribute {attribute:?}")),
                    "{label}: {problem}"
                );
                assert!(problem.contains("raw '<'"), "{label}: {problem}");
                let err = day_page(&xml).unwrap_err().to_string();
                assert!(err.contains("did not return well-formed XML"), "{label}: {err}");
                swept += 1;
            }
        }
        assert_eq!(swept, 52, "26 schema paths, each under two attribute names");

        // The root's own start tag, which is the audit's reproduction: the
        // real entry moved verbatim into the `xmlns` VALUE with <KeyCount>
        // lowered by one to agree.  Round 4 accepted this with a silently
        // short two-object roster and called the document well-formed.
        let short = edited(REAL_TRUNCATED_PAGE, HIDDEN_ENTRY, "");
        let short = edited(&short, "<KeyCount>3</KeyCount>", "<KeyCount>2</KeyCount>");
        for attribute in ["xmlns", "deleted"] {
            let xml = edited(
                &short,
                ROOT_TAG,
                &format!("<ListBucketResult {attribute}=\"{HIDDEN_ENTRY}\">"),
            );
            assert!(xml.contains(HIDDEN_ENTRY), "the entry is still in the bytes");
            let problem = scan_xml(&xml).unwrap_err();
            assert!(
                problem.contains(&format!("value of attribute {attribute:?}")),
                "{problem}"
            );
            let err = day_page(&xml).unwrap_err().to_string();
            assert!(err.contains("did not return well-formed XML"), "{err}");
        }

        // A 4 KB value, which is what the audit used to show the payload is
        // unbounded, and a bare '<' with no markup after it: the character
        // is the refusal, not the shape of what follows it.
        for value in ["<".to_string(), "x".repeat(4096) + "<Contents>"] {
            let xml = edited(
                REAL_TRUNCATED_PAGE,
                ROOT_TAG,
                &format!("<ListBucketResult xmlns=\"{value}\">"),
            );
            assert!(scan_xml(&xml).unwrap_err().contains("raw '<'"));
        }

        // The other half of the class, and the reason the value is declared
        // and not merely the name: an ENTITY-ESCAPED entry is well-formed
        // XML, so the well-formedness layer cannot speak to it -- but the
        // one attribute the response syntax declares has one value, which
        // this compares, and every other name is refused outright.  There is
        // no attribute anywhere in the document an entry can ride in.
        let escaped = HIDDEN_ENTRY.replace('<', "&lt;").replace('>', "&gt;");
        for attribute in ["xmlns", "deleted"] {
            let xml = edited(
                &short,
                ROOT_TAG,
                &format!("<ListBucketResult {attribute}=\"{escaped}\">"),
            );
            assert!(scan_xml(&xml).is_ok(), "escaped markup is well-formed XML");
            let err = day_page(&xml).unwrap_err().to_string();
            assert!(
                err.contains(&format!("attribute {attribute:?} on <ListBucketResult>")),
                "{err}"
            );
        }
    }

    #[test]
    fn a_start_tag_declares_each_attribute_once_and_the_namespace_it_says() {
        // Two halves of RV6-03's tail, both of them things round 4 accepted.
        //
        // A start tag that declares `xmlns` twice is not well-formed XML at
        // all (WFC: Unique Att Spec), and two values for one name are two
        // answers to one question.
        let xml = edited(
            REAL_TRUNCATED_PAGE,
            ROOT_TAG,
            "<ListBucketResult xmlns=\"http://s3.amazonaws.com/doc/2006-03-01/\" \
             xmlns=\"http://s3.amazonaws.com/doc/2006-03-01/\">",
        );
        let problem = scan_xml(&xml).unwrap_err();
        assert!(problem.contains("declares attribute \"xmlns\" twice"), "{problem}");
        let err = day_page(&xml).unwrap_err().to_string();
        assert!(err.contains("did not return well-formed XML"), "{err}");

        // And the value is read: this element schema is a transcription of
        // one namespace, so a document that says it is in another is not the
        // document it describes.
        for namespace in ["urn:not-s3-at-all", "", "http://s3.amazonaws.com/doc/2006-03-02/"] {
            let xml = edited(
                REAL_TRUNCATED_PAGE,
                ROOT_TAG,
                &format!("<ListBucketResult xmlns=\"{namespace}\">"),
            );
            assert!(scan_xml(&xml).is_ok(), "{namespace}: well-formed, just wrong");
            let err = day_page(&xml).unwrap_err().to_string();
            assert!(err.contains("attribute \"xmlns\" on <ListBucketResult>"), "{err}");
            assert!(err.contains(&format!("declares {namespace:?}")), "{err}");
        }

        // The control, which is every real fixture.
        assert!(day_page(REAL_TRUNCATED_PAGE).is_ok());

        // And the value that is compared is character data, not the source
        // bytes: the receipt names one `&`, not five bytes of `&amp;`.
        let xml = edited(
            REAL_TRUNCATED_PAGE,
            ROOT_TAG,
            "<ListBucketResult xmlns=\"http://s3.amazonaws.com/doc/2006-03-01/&amp;x\">",
        );
        let err = day_page(&xml).unwrap_err().to_string();
        assert!(
            err.contains("declares \"http://s3.amazonaws.com/doc/2006-03-01/&x\""),
            "{err}"
        );
    }

    #[test]
    fn the_root_namespace_is_required_and_not_merely_validated() {
        // RV7-03.  Round 5 compared the namespace VALUE when the root stated
        // one and said nothing when it did not: a <ListBucketResult> with no
        // `xmlns` at all parsed to the full three-object roster, while
        // `xmlns=""` -- which in XML says the same thing, "in no namespace"
        // -- was refused, and the suite pinned only the refusal of the empty
        // one.  Absent, empty and wrong are three spellings of one fault.
        //
        // The fixture evidence for requiring it rather than defaulting it:
        // every captured real <ListBucketResult> carries the namespace
        // verbatim, so no real page is lost to the stricter rule.
        for (label, page) in [
            ("truncated", REAL_TRUNCATED_PAGE),
            ("final", REAL_FINAL_PAGE),
            ("empty", REAL_EMPTY_PAGE),
        ] {
            assert!(page.contains(ROOT_TAG), "the real {label} page states no namespace");
        }
        assert_eq!(day_page(REAL_TRUNCATED_PAGE).unwrap().objects.len(), 3);

        let mut swept = 0usize;

        // Absent.  Well-formed XML, and refused for what it does not say.
        let xml = edited(REAL_TRUNCATED_PAGE, ROOT_TAG, "<ListBucketResult>");
        assert!(scan_xml(&xml).is_ok(), "silent, not malformed");
        let err = day_page(&xml).unwrap_err().to_string();
        assert!(
            err.contains("<ListBucketResult> declares no \"xmlns\" attribute"),
            "{err}"
        );
        assert!(
            err.contains("says nothing about the namespace it is in"),
            "{err}"
        );
        assert!(err.contains("http://s3.amazonaws.com/doc/2006-03-01/"), "{err}");
        swept += 1;

        // Empty, which gets its own message because it is its own
        // statement: `xmlns=""` puts the element in no namespace.
        let xml = edited(REAL_TRUNCATED_PAGE, ROOT_TAG, "<ListBucketResult xmlns=\"\">");
        let err = day_page(&xml).unwrap_err().to_string();
        assert!(
            err.contains("attribute \"xmlns\" on <ListBucketResult> declares \"\""),
            "{err}"
        );
        assert!(err.contains("NO namespace at all"), "{err}");
        swept += 1;

        // Wrong, by value, each naming what the document said.
        for namespace in [
            "urn:not-s3-at-all",
            "http://s3.amazonaws.com/doc/2006-03-02/",
            "https://s3.amazonaws.com/doc/2006-03-01/",
            "http://s3.amazonaws.com/doc/2006-03-01",
        ] {
            let xml = edited(
                REAL_TRUNCATED_PAGE,
                ROOT_TAG,
                &format!("<ListBucketResult xmlns=\"{namespace}\">"),
            );
            let err = day_page(&xml).unwrap_err().to_string();
            assert!(err.contains(&format!("declares {namespace:?}")), "{err}");
            assert!(err.contains("is not the document this client can read"), "{err}");
            swept += 1;
        }
        assert_eq!(swept, 6, "absent, empty, and four wrong spellings");

        // The three arms say three different things, which is the point of
        // giving each its own message rather than one refusal for all.
        let absent = day_page(&edited(REAL_TRUNCATED_PAGE, ROOT_TAG, "<ListBucketResult>"))
            .unwrap_err()
            .to_string();
        let empty = day_page(&edited(
            REAL_TRUNCATED_PAGE,
            ROOT_TAG,
            "<ListBucketResult xmlns=\"\">",
        ))
        .unwrap_err()
        .to_string();
        let wrong = day_page(&edited(
            REAL_TRUNCATED_PAGE,
            ROOT_TAG,
            "<ListBucketResult xmlns=\"urn:not-s3-at-all\">",
        ))
        .unwrap_err()
        .to_string();
        assert_ne!(absent, empty);
        assert_ne!(empty, wrong);
        assert_ne!(absent, wrong);

        // A body that is not a <ListBucketResult> is still judged as what it
        // is: the namespace rule belongs to the element schema, and the real
        // AccessDenied page carries no namespace and must not be re-labelled
        // as one that forgot it.
        let err = parse_s3_list_xml(REAL_ACCESS_DENIED, BUCKET, DAY_PREFIX)
            .unwrap_err()
            .to_string();
        assert!(err.contains("not a <ListBucketResult>"), "{err}");
        assert!(err.contains("AccessDenied"), "{err}");
    }

    #[test]
    fn a_numeric_character_reference_is_refused_wherever_it_appears() {
        // Re-verification #7's undisclosed residual, made a decision instead
        // of an accident.  `&#60;` and `&#x3C;` are ordinary, legal XML 1.0;
        // this client refuses every numeric character reference, everywhere,
        // and the cost is a whole page rather than a decode this
        // deliberately small language has no other rule for.  It is
        // fail-closed and it is deliberate.  Pinned here so that a later
        // round widens it on purpose or not at all, and disclosed as an
        // availability narrowing in the round-6 report.
        let mut swept = 0usize;
        for reference in ["&#60;", "&#x3C;", "&#38;", "&#x26;", "&#10;"] {
            // In character data.
            let xml = edited(
                REAL_TRUNCATED_PAGE,
                "<LastModified>2026-07-29T00:09:29.000Z</LastModified>",
                &format!("<LastModified>2026-07-29T00:09:29.000Z{reference}</LastModified>"),
            );
            let err = day_page(&xml).unwrap_err().to_string();
            assert!(err.contains("entity reference"), "{reference}: {err}");
            assert!(err.contains(reference), "{reference}: {err}");
            swept += 1;

            // In an attribute value.  A decoded value here is compared and
            // never re-scanned, so this is strictness rather than a carrier
            // -- which is exactly why it is worth disclosing.
            let xml = edited(
                REAL_TRUNCATED_PAGE,
                ROOT_TAG,
                &format!(
                    "<ListBucketResult xmlns=\"http://s3.amazonaws.com/doc/2006-03-01/{reference}\">"
                ),
            );
            let err = day_page(&xml).unwrap_err().to_string();
            assert!(err.contains("entity reference"), "{reference}: {err}");
            swept += 1;

            // And in a declaration value, where XML 1.0 permits no reference
            // at all: the refusal is the production's, and nothing decodes.
            let xml = edited(
                REAL_TRUNCATED_PAGE,
                "<?xml version=\"1.0\" encoding=\"UTF-8\"?>",
                &format!("<?xml version=\"1.0\" encoding=\"UTF-8{reference}\"?>"),
            );
            let problem = scan_xml(&xml).unwrap_err();
            assert!(
                problem.contains("declaration states encoding="),
                "{reference}: {problem}"
            );
            assert!(
                problem.contains("no entity reference is permitted"),
                "{reference}: {problem}"
            );
            swept += 1;
        }
        assert_eq!(swept, 15, "five numeric references x three positions");

        // The five that ARE declared still decode, everywhere they are legal.
        assert_eq!(
            xml_attributes(" a=\"&amp;&lt;&gt;&quot;&apos;\"").unwrap(),
            [("a".to_string(), "&<>\"'".to_string())]
        );
    }

    #[test]
    fn the_xml_declaration_carries_only_what_xml_1_0_declares() {
        // RV6-03.  Being permitted at byte zero is not being permitted to
        // contain anything: the declaration is a region the scanner skips,
        // and round 4 skipped all of it, so the real second roster entry
        // planted inside it -- with <KeyCount> lowered to agree -- was
        // accepted with a two-object roster.
        let short = edited(REAL_TRUNCATED_PAGE, HIDDEN_ENTRY, "");
        let short = edited(&short, "<KeyCount>3</KeyCount>", "<KeyCount>2</KeyCount>");
        const DECLARATION: &str = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>";
        assert!(REAL_TRUNCATED_PAGE.starts_with(DECLARATION));
        for carrier in [
            // inside the encoding pseudo-attribute's value ...
            format!("<?xml version=\"1.0\" encoding=\"{HIDDEN_ENTRY}\"?>"),
            // ... and loose in the declaration body, quoted by nothing.
            format!("<?xml version=\"1.0\" {HIDDEN_ENTRY}?>"),
        ] {
            let xml = edited(&short, DECLARATION, &carrier);
            assert!(xml.contains(HIDDEN_ENTRY), "the entry is still in the bytes");
            let problem = scan_xml(&xml).unwrap_err();
            assert!(problem.contains("<?xml ...?> declaration"), "{problem}");
            assert!(problem.contains("markup"), "{problem}");
            let err = day_page(&xml).unwrap_err().to_string();
            assert!(err.contains("did not return well-formed XML"), "{err}");
        }

        // RV7-01: the escaped half of the same class, which round 5 left
        // open.  The pseudo-attribute NAMES were held and the VALUES were
        // entity-decoded and then compared against nothing, so an
        // `&lt;`-escaped entry in any one of the three -- with <KeyCount>
        // lowered by one -- was ACCEPTED with a two-object roster and the
        // entry still in the bytes.  The census axis is the three values
        // crossed with both spellings; the raw one is refused by the
        // character, the escaped one by the production.
        let declaration_with = |name: &str, value: &str| -> String {
            match name {
                "version" => format!("<?xml version=\"{value}\"?>"),
                "encoding" => format!("<?xml version=\"1.0\" encoding=\"{value}\"?>"),
                _ => format!(
                    "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"{value}\"?>"
                ),
            }
        };
        let escaped = HIDDEN_ENTRY.replace('<', "&lt;").replace('>', "&gt;");
        let mut swept = 0usize;
        for name in ["version", "encoding", "standalone"] {
            for (payload, spelling) in [
                (HIDDEN_ENTRY.to_string(), "raw"),
                (escaped.clone(), "escaped"),
            ] {
                let xml = edited(&short, DECLARATION, &declaration_with(name, &payload));
                let label = format!("{name}/{spelling}");
                assert!(xml.contains("<KeyCount>2</KeyCount>"), "{label}");
                let problem = scan_xml(&xml).unwrap_err();
                if spelling == "raw" {
                    assert!(
                        xml.contains(HIDDEN_ENTRY),
                        "{label}: the entry is still in the bytes"
                    );
                    assert!(problem.contains("declaration carries markup"), "{label}: {problem}");
                } else {
                    assert!(
                        problem.contains(&format!("declaration states {name}=")),
                        "{label}: {problem}"
                    );
                    assert!(
                        problem.contains("no entity reference is permitted"),
                        "{label}: {problem}"
                    );
                }
                let err = day_page(&xml).unwrap_err().to_string();
                assert!(err.contains("did not return well-formed XML"), "{label}: {err}");
                swept += 1;
            }
        }
        assert_eq!(swept, 6, "three declaration values x two spellings");

        // The other half of the axis: values that carry no entry at all and
        // are still not what XML 1.0 declares.  `version="lolwut"`,
        // `encoding="not an EncName!!"` and `standalone="maybe"` each parsed
        // to the full three-object roster before this, which is what "held
        // to nothing" meant.
        let mut bounded = 0usize;
        for (name, value) in [
            ("version", "lolwut"),
            ("version", ""),
            ("version", "1."),
            ("version", "1.0.0"),
            ("version", "2.0"),
            ("version", "&lt;"),
            ("encoding", "not an EncName!!"),
            ("encoding", ""),
            ("encoding", "1UTF-8"),
            ("encoding", "-UTF-8"),
            ("encoding", "UTF 8"),
            ("encoding", "UTF-8/"),
            ("standalone", "maybe"),
            ("standalone", "YES"),
            ("standalone", ""),
            ("standalone", "yes "),
        ] {
            let xml = edited(
                REAL_TRUNCATED_PAGE,
                DECLARATION,
                &declaration_with(name, value),
            );
            let problem = scan_xml(&xml).unwrap_err();
            assert!(
                problem.contains(&format!("declaration states {name}={value:?}")),
                "{name}={value:?}: {problem}"
            );
            assert!(
                problem.contains("carried out of the roster inside"),
                "{name}={value:?}: {problem}"
            );
            bounded += 1;
        }
        assert_eq!(bounded, 16, "grammar-violating values across the three");

        // Length was the one property the production did not bound, and
        // round 6 disclosed the audit's 4 KB `encoding` as a residual on the
        // ground that capping it would be an invented rule.  It is not
        // capped now either -- it is refused for what it SAYS, which is that
        // the body is in an encoding this client does not read (RV8-04), and
        // the length residual dissolves with it: `version` and `standalone`
        // are finite value sets, so no declaration value is an unbounded
        // region any more.
        let long = "x".repeat(4096);
        assert!(!long.contains('<') && !long.contains('&'));
        let xml = edited(
            REAL_TRUNCATED_PAGE,
            DECLARATION,
            &declaration_with("encoding", &long),
        );
        let problem = scan_xml(&xml).unwrap_err();
        assert!(problem.contains("as UTF-8 and only as UTF-8"), "{problem}");

        // The grammar, refused by name rather than by character: XML 1.0
        // gives the declaration a version, then an optional encoding, then
        // an optional standalone, in that order and once each.
        for (declaration, expected) in [
            ("<?xml encoding=\"UTF-8\"?>", "states no version"),
            ("<?xml?>", "states no version"),
            ("<?xml version=\"1.0\" hide=\"x\"?>", "carries \"hide\""),
            (
                "<?xml encoding=\"UTF-8\" version=\"1.0\"?>",
                "carries \"version\"",
            ),
            (
                "<?xml version=\"1.0\" version=\"1.0\"?>",
                "declares attribute \"version\" twice",
            ),
            ("<?xml version=1.0?>", "unquoted value"),
        ] {
            let xml = edited(REAL_TRUNCATED_PAGE, DECLARATION, declaration);
            let problem = scan_xml(&xml).unwrap_err();
            assert!(problem.contains(expected), "{declaration}: {problem}");
        }

        // The controls: every spelling XML 1.0 does declare, and the real
        // fixtures' own.
        for declaration in [
            "<?xml version=\"1.0\"?>",
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>",
            "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>",
            "<?xml version=\"1.0\" standalone=\"no\"?>",
            "<?xml version='1.0' encoding='UTF-8'?>",
            // Every one of these is inside the productions, and refusing
            // any of them would be a page lost to strictness that XML 1.0
            // does not ask for.
            "<?xml version=\"1.1\"?>",
            "<?xml version=\"1.10\"?>",
            "<?xml version=\"1.0\" encoding=\"utf-8\"?>",
            // `ISO-8859-1`, `US-ASCII` and `EBCDIC.CP-US` used to sit in this
            // list.  They are inside `EncName` and they are still not
            // documents this client reads, because it decodes the body as
            // UTF-8 and does not transcode; they moved to the refusal census
            // in `a_declared_encoding_this_client_cannot_honour_is_refused`,
            // which is a tightening of this list and not a hole in it.
        ] {
            let xml = edited(REAL_TRUNCATED_PAGE, DECLARATION, declaration);
            assert_eq!(
                day_page(&xml).unwrap().objects.len(),
                3,
                "{declaration} is a declaration XML 1.0 declares"
            );
        }
        // A document with no declaration at all is still a document.
        let xml = edited(REAL_TRUNCATED_PAGE, DECLARATION, "");
        assert_eq!(day_page(&xml).unwrap().objects.len(), 3);
    }

    #[test]
    fn a_declared_encoding_this_client_cannot_honour_is_refused() {
        // The undisclosed residual re-verification #8 named: round 6 spent a
        // commit giving `encoding` a grammar without ever stating that its
        // MEANING is discarded.  The body is decoded UTF-8 unconditionally
        // (ureq's charset feature is off) and nothing downstream reads the
        // label, so `encoding="UTF-16"` on UTF-8 bytes parsed happily.  No
        // carrier -- but a validator that validates a statement and then
        // ignores it is telling the operator something untrue about what it
        // checked.  No transcoding is added; the input is refused instead,
        // which is the doctrine the rest of this reader follows.
        const DECLARATION: &str = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>";
        let mut swept = 0usize;
        for label in [
            "UTF-16",
            "UTF-16LE",
            "UTF-32",
            "ISO-8859-1",
            "US-ASCII",
            "EBCDIC-CP-US",
            "EBCDIC.CP-US",
            "windows-1252",
            "Shift_JIS",
        ] {
            let xml = edited(
                REAL_TRUNCATED_PAGE,
                DECLARATION,
                &format!("<?xml version=\"1.0\" encoding=\"{label}\"?>"),
            );
            // Each of these is inside `EncName`, so the grammar has nothing
            // to say about it: this is the capability refusal or it is
            // nobody's.
            let problem = scan_xml(&xml).unwrap_err();
            assert!(
                problem.contains(&format!("declaration states encoding={label:?}")),
                "{label}: {problem}"
            );
            assert!(
                problem.contains("as UTF-8 and only as UTF-8"),
                "{label}: {problem}"
            );
            assert!(
                !problem.contains("EncName ::="),
                "{label} is grammatical; the refusal must not blame the \
                 production: {problem}"
            );
            let err = day_page(&xml).unwrap_err().to_string();
            assert!(err.contains("did not return well-formed XML"), "{label}: {err}");
            swept += 1;
        }
        assert_eq!(swept, 9, "encoding labels this client cannot honour");

        // The labels it can, in the spellings XML 1.0 says are matched
        // case-insensitively, plus a declaration that states no encoding at
        // all -- which is not a claim about one and stays a document.
        for declaration in [
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>",
            "<?xml version=\"1.0\" encoding=\"utf-8\"?>",
            "<?xml version=\"1.0\" encoding=\"UtF-8\"?>",
            "<?xml version=\"1.0\" encoding=\"UTF8\"?>",
            "<?xml version=\"1.0\" encoding=\"utf8\"?>",
            "<?xml version=\"1.0\"?>",
            "<?xml version=\"1.0\" standalone=\"yes\"?>",
        ] {
            let xml = edited(REAL_TRUNCATED_PAGE, DECLARATION, declaration);
            assert_eq!(
                day_page(&xml).unwrap().objects.len(),
                3,
                "{declaration} is an encoding this client reads"
            );
        }
        // And every captured real body declares the one it reads.
        for (label, page) in [
            ("truncated", REAL_TRUNCATED_PAGE),
            ("final", REAL_FINAL_PAGE),
            ("empty", REAL_EMPTY_PAGE),
            ("access denied", REAL_ACCESS_DENIED),
        ] {
            assert!(
                page.contains("encoding=\"UTF-8\""),
                "the real {label} page does not declare UTF-8"
            );
        }
    }

    #[test]
    fn xml_space_is_the_four_characters_the_production_names() {
        // RV8-03.  This round's thesis is that XML 1.0's productions are the
        // bound, and `S ::= (#x20 | #x9 | #xD | #xA)+` is one of them -- but
        // the reader was standing at every `S` position holding
        // `char::is_whitespace`, which is Unicode White_Space and admits
        // NBSP, `#x2028`, `#x2029` and a dozen more.  None of them is a
        // carrier, because the element they separate is still schema-checked;
        // each of them is a document a conforming parser rejects and this one
        // read.  The census is one mutation per `S` position, each of which
        // PARSED before.
        let control = day_page(REAL_TRUNCATED_PAGE).unwrap();
        assert_eq!(control.objects.len(), 3);

        let mut swept = 0usize;
        // Declaration pseudo-attribute separation.  NBSP was whitespace, so
        // this was two well-separated pseudo-attributes; it is one malformed
        // region.
        let xml = edited(
            REAL_TRUNCATED_PAGE,
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>",
            "<?xml version=\"1.0\"\u{a0}encoding=\"UTF-8\"?>",
        );
        let problem = scan_xml(&xml).unwrap_err();
        assert!(problem.contains("no whitespace before it"), "{problem}");
        swept += 1;

        // Outside the root, both sides of it.  A separator that is not `S`
        // is character data, and character data outside the root element is
        // not something a listing may carry.
        for (label, xml) in [
            (
                "before the root",
                edited(
                    REAL_TRUNCATED_PAGE,
                    "<ListBucketResult",
                    "\u{a0}<ListBucketResult",
                ),
            ),
            ("after the root", format!("{REAL_TRUNCATED_PAGE}\u{2028}")),
        ] {
            let problem = scan_xml(&xml).unwrap_err();
            assert!(
                problem.contains("outside its root element")
                    || problem.contains("after its root element"),
                "{label}: {problem}"
            );
            swept += 1;
        }

        // Tag-name termination.  `<KeyCount{NBSP}>` was read as `<KeyCount>`
        // with an empty attribute region, so the page parsed to three
        // objects; the name now runs to the `>` and nothing closes it.
        let xml = edited(REAL_TRUNCATED_PAGE, "<KeyCount>", "<KeyCount\u{a0}>");
        let problem = scan_xml(&xml).unwrap_err();
        assert!(
            problem.contains("closes an element that was never opened"),
            "{problem}"
        );
        swept += 1;
        assert_eq!(swept, 4, "one mutation per S position this round tightened");

        // The four characters the production DOES name, at the same
        // positions, all still parse -- so the tightening is the predicate
        // and not a refusal of whitespace.
        for (label, xml) in [
            (
                "declaration separated by a tab",
                edited(
                    REAL_TRUNCATED_PAGE,
                    "<?xml version=\"1.0\" encoding=\"UTF-8\"?>",
                    "<?xml version=\"1.0\"\tencoding=\"UTF-8\"?>",
                ),
            ),
            (
                "CR, LF and tab outside the root",
                edited(
                    REAL_TRUNCATED_PAGE,
                    "<ListBucketResult",
                    "\r\n\t <ListBucketResult",
                ),
            ),
            (
                "S before the end of a tag",
                edited(
                    REAL_TRUNCATED_PAGE,
                    "<KeyCount>3</KeyCount>",
                    "<KeyCount >3</KeyCount\t>",
                ),
            ),
        ] {
            assert_eq!(
                day_page(&xml).unwrap().objects.len(),
                3,
                "{label} is S, and S is permitted here"
            );
        }

        // XML 1.0 §2.4: `]]>` is forbidden in character data outside a CDATA
        // section, and this reader has no CDATA sections.  Not a carrier
        // either -- but it is the same class of document, one a conforming
        // parser rejects, and the check is on the SOURCE span so the legal
        // spelling still decodes.
        let xml = edited(
            REAL_TRUNCATED_PAGE,
            "<LastModified>2026-07-29T00:09:29.000Z</LastModified>",
            "<LastModified>2026-07-29]]>T00:09:29.000Z</LastModified>",
        );
        let err = day_page(&xml).unwrap_err().to_string();
        assert!(err.contains("literal ']]>' in character data"), "{err}");
        assert!(err.contains("<LastModified>"), "{err}");

        let xml = edited(
            REAL_TRUNCATED_PAGE,
            "<LastModified>2026-07-29T00:09:29.000Z</LastModified>",
            "<LastModified>2026-07-29]]&gt;T00:09:29.000Z</LastModified>",
        );
        let page = day_page(&xml).unwrap();
        assert_eq!(
            page.objects[0].last_modified,
            "2026-07-29]]>T00:09:29.000Z",
            "the escape XML gives the sequence must still decode to it"
        );

        // And the predicate itself, since every position above delegates to
        // it: four characters, and the ones Unicode adds are not among them.
        for character in [' ', '\t', '\r', '\n'] {
            assert!(is_xml_space(character), "{character:?} is S");
        }
        for character in ['\u{a0}', '\u{2028}', '\u{2029}', '\u{2003}', '\u{feff}', 'x'] {
            assert!(!is_xml_space(character), "{character:?} is not S");
        }
    }

    #[test]
    fn a_text_only_element_says_so_when_it_is_handed_a_subtree() {
        let xml = page_with(
            REAL_TRUNCATED_PAGE,
            "<StartAfter><Unexpected>x</Unexpected></StartAfter>",
            3,
        );
        let err = day_page(&xml).unwrap_err().to_string();
        assert!(
            err.contains("unexpected <Unexpected> direct child of <StartAfter>"),
            "{err}"
        );
        assert!(err.contains("text-only element"), "{err}");

        // A container names itself and what it does declare instead.
        let xml = page_with(
            REAL_TRUNCATED_PAGE,
            &nested(&["Contents", "Owner"], "<Unexpected>x</Unexpected>"),
            4,
        );
        let err = day_page(&xml).unwrap_err().to_string();
        assert!(
            err.contains("unexpected <Unexpected> direct child of <Owner>"),
            "{err}"
        );
        assert!(err.contains("not in the ListObjectsV2 element schema"), "{err}");
    }

    #[test]
    fn a_container_element_carries_no_character_data_of_its_own() {
        // RV8-02, and the residual whose stated reason it falsifies.
        //
        // Rounds 4, 5 and 6 all carried the same sentence: a container's
        // text "is never read as a value ... so nothing can be smuggled
        // through it".  Re-verification #8 measured it and it is false in
        // the identical shape RV6-01 and RV7-01 were false.  Unread is not
        // harmless: the roster is shortened by what leaves it, not by what
        // is read, and a region nothing reads and nothing bounds is exactly
        // where an entry goes when it leaves.  The real second entry,
        // `&lt;`-escaped into `<ListBucketResult>`'s own character data --
        // and into a `<Contents>`'s -- with `<KeyCount>` lowered by one, was
        // ACCEPTED with a short two-object roster and the entry still in the
        // bytes.
        let short = edited(REAL_TRUNCATED_PAGE, HIDDEN_ENTRY, "");
        let short = edited(&short, "<KeyCount>3</KeyCount>", "<KeyCount>2</KeyCount>");
        let escaped = HIDDEN_ENTRY.replace('<', "&lt;").replace('>', "&gt;");

        let mut swept = 0usize;
        for (container, xml) in [
            (
                "ListBucketResult",
                edited(
                    &short,
                    "</ListBucketResult>",
                    &format!("{escaped}</ListBucketResult>"),
                ),
            ),
            (
                "Contents",
                edited(
                    &short,
                    "<Size>5062263</Size>",
                    &format!("{escaped}<Size>5062263</Size>"),
                ),
            ),
        ] {
            // Escaped markup is well-formed XML, so the well-formedness
            // layer has nothing to say about it: this is the content model's
            // refusal or it is nobody's.
            assert!(scan_xml(&xml).is_ok(), "{container}: escaped text is well-formed");
            assert!(
                xml.contains("<KeyCount>2</KeyCount>"),
                "{container}: the count agrees with the short roster"
            );
            assert!(
                xml.contains("KTLX20260729_000936_V06"),
                "{container}: the entry is still in the bytes"
            );
            let err = day_page(&xml).unwrap_err().to_string();
            assert!(
                err.contains(&format!(
                    "<{container}> is an element-content element in the \
                     ListObjectsV2 element schema"
                )),
                "{container}: {err}"
            );
            assert!(err.contains("carried out of the roster inside"), "{container}: {err}");
            swept += 1;
        }
        assert_eq!(swept, 2, "the two containers a real page has to hide an entry in");

        // The control, and the thing the residual was actually protecting: a
        // pretty-printed roster.  Indentation between a container's children
        // is `S` and carries nothing, so it still parses -- to the same three
        // objects, byte for byte.
        let control = day_page(REAL_TRUNCATED_PAGE).unwrap();
        let pretty = REAL_TRUNCATED_PAGE.replace("><", ">\n    <");
        assert!(pretty.contains(">\n    <"), "the control must actually be indented");
        assert_eq!(day_page(&pretty).unwrap().objects, control.objects);

        // And the falsification of that control: the same positions filled
        // with something that is not whitespace are refused, which is what
        // makes the acceptance above a statement about `S` rather than a
        // statement about being lenient.
        let junk = REAL_TRUNCATED_PAGE.replace("></", ">x</");
        let err = day_page(&junk).unwrap_err().to_string();
        assert!(err.contains("is an element-content element"), "{err}");
    }

    #[test]
    fn a_processing_instruction_outside_the_prolog_is_refused() {
        // The control: the declaration where XML 1.0 puts it, and only there.
        assert!(REAL_TRUNCATED_PAGE.starts_with("<?xml version=\"1.0\""));
        assert!(day_page(REAL_TRUNCATED_PAGE).is_ok());

        // A second declaration inside the root.  XML 1.0 forbids it anywhere
        // but byte zero, so this body fails a check S3 output cannot fail --
        // and it was accepted, with a three-object roster.
        let xml = edited(
            REAL_TRUNCATED_PAGE,
            "<KeyCount>3</KeyCount>",
            "<?xml version=\"1.0\"?><KeyCount>3</KeyCount>",
        );
        let err = day_page(&xml).unwrap_err().to_string();
        assert!(err.contains("only at the very start of the document"), "{err}");

        // In the prolog but not at byte zero is the same fault.  The lead-in
        // is whitespace rather than the `<!--lead-->` this used to use,
        // because a comment is itself refused everywhere now (RV7-02) and
        // would answer for the position rule before it was reached.
        let xml = format!("\n{}", REAL_TRUNCATED_PAGE);
        let err = day_page(&xml).unwrap_err().to_string();
        assert!(err.contains("only at the very start of the document"), "{err}");
        // The shape it used to be written as still refuses -- for the
        // comment, which is the accurate reason.
        let xml = format!("<!--lead-->{}", REAL_TRUNCATED_PAGE);
        let err = day_page(&xml).unwrap_err().to_string();
        assert!(err.contains("comment outside its root element"), "{err}");

        // Any other target, anywhere, prolog included: S3 sends none.
        for xml in [
            edited(
                REAL_TRUNCATED_PAGE,
                "<KeyCount>3</KeyCount>",
                "<?rewrite roster?><KeyCount>3</KeyCount>",
            ),
            format!("<?rewrite roster?>{}", REAL_TRUNCATED_PAGE),
        ] {
            let err = day_page(&xml).unwrap_err().to_string();
            assert!(
                err.contains("<?rewrite ...?> processing instruction"),
                "{err}"
            );
        }

        // And the reason the position rule is not pedantry: the region a
        // parser skips is a region a real <Contents> fits inside.  Ok with a
        // two-object roster and an agreeing <KeyCount>, before this.
        let xml = edited(
            REAL_TRUNCATED_PAGE,
            HIDDEN_ENTRY,
            &format!("<?hide {}?>", HIDDEN_ENTRY),
        );
        let xml = edited(&xml, "<KeyCount>3</KeyCount>", "<KeyCount>2</KeyCount>");
        let err = day_page(&xml).unwrap_err().to_string();
        assert!(err.contains("<?hide ...?> processing instruction"), "{err}");
    }

    #[test]
    fn a_question_mark_region_that_opens_and_closes_at_once_is_refused_not_panicked_on() {
        // RV8-01, and the only finding on this ladder that was not a carrier
        // but was worse than one.  `<?>` is three bytes: the terminator
        // search started at byte zero, found the `?>` that begins at byte
        // ONE, and sliced the body as `rest[2..1]`.  That is a slice-index
        // panic, reachable from the network through
        // `list_s3_objects` -> `parse_s3_list_xml` on the raw response body,
        // with no `catch_unwind` anywhere in the crate: it unwound to
        // `main`, exit 101, a message about a byte range and no verdict at
        // all about the page.  Every other refusal on this ladder at least
        // said which listing it would not read.
        //
        // Each arm below calls the parser and reads the error.  That IS the
        // no-panic assertion in this suite's style -- a panic fails the test
        // where it happens -- and the message is asserted as well, so a
        // regression that restores the panic and a regression that quietly
        // widens the refusal both fail here.
        let mut swept = 0usize;
        for (label, xml, expected) in [
            (
                "the whole body",
                "<?>".to_string(),
                "unterminated <?...?> declaration",
            ),
            (
                "in the prolog, before the declaration",
                format!("<?>{REAL_TRUNCATED_PAGE}"),
                // The next `?>` in the document is the real declaration's,
                // so this reads as a processing instruction with an
                // unreadable target -- refused by name, which is the point.
                "processing instruction",
            ),
            (
                "in the prolog, after the declaration",
                edited(
                    REAL_TRUNCATED_PAGE,
                    "<ListBucketResult",
                    "<?><ListBucketResult",
                ),
                "unterminated <?...?> declaration",
            ),
            (
                "between two <Contents> entries",
                edited(REAL_TRUNCATED_PAGE, HIDDEN_ENTRY, &format!("<?>{HIDDEN_ENTRY}")),
                "unterminated <?...?> declaration",
            ),
            (
                "after the root element",
                format!("{REAL_TRUNCATED_PAGE}<?>"),
                "unterminated <?...?> declaration",
            ),
        ] {
            let problem = scan_xml(&xml).unwrap_err();
            assert!(problem.contains(expected), "{label}: {problem}");
            let err = day_page(&xml).unwrap_err().to_string();
            assert!(err.contains("did not return well-formed XML"), "{label}: {err}");
            swept += 1;
        }
        assert_eq!(swept, 5, "the positions a <?> can arrive in");

        // The class rather than the five instances: `<?>` inserted at EVERY
        // byte offset of a real page refuses, and none of the insertions
        // panics.  This is the sweep shape `no_prefix_of_a_real_page_parses_
        // as_a_page` uses, for the same reason -- the hand-picked offset is
        // how a byte sequence goes unnoticed at the offsets nobody picked.
        let mut inserted = 0usize;
        for (label, page, prefix) in [
            ("truncated", REAL_TRUNCATED_PAGE, DAY_PREFIX),
            ("final", REAL_FINAL_PAGE, ONE_KEY_PREFIX),
            ("empty", REAL_EMPTY_PAGE, ABSENT_PREFIX),
        ] {
            for at in 0..=page.len() {
                if !page.is_char_boundary(at) {
                    continue;
                }
                let xml = format!("{}<?>{}", &page[..at], &page[at..]);
                assert!(
                    parse_s3_list_xml(&xml, BUCKET, prefix).is_err(),
                    "{label} page with <?> inserted at {at} parsed as a listing"
                );
                inserted += 1;
            }
            // And the page itself still parses, so the sweep is not passing
            // because the fixture was unparseable to begin with.
            assert!(parse_s3_list_xml(page, BUCKET, prefix).is_ok(), "{label}");
        }
        assert_eq!(
            inserted,
            REAL_TRUNCATED_PAGE.len() + REAL_FINAL_PAGE.len() + REAL_EMPTY_PAGE.len() + 3,
            "every byte offset of all three real pages, ends included -- \
             2,281 insertions at the fixtures' present lengths \
             (1374 + 613 + 291, plus the end of each)"
        );

        // The two neighbours of `<?>` in the grammar, unchanged by the fix:
        // `<??>` is a processing instruction with an EMPTY target, and the
        // real declaration is still the one region byte zero admits.
        let xml = edited(
            REAL_TRUNCATED_PAGE,
            "<ListBucketResult",
            "<??><ListBucketResult",
        );
        let problem = scan_xml(&xml).unwrap_err();
        assert!(
            problem.contains("<? ...?> processing instruction"),
            "{problem}"
        );
        assert_eq!(day_page(REAL_TRUNCATED_PAGE).unwrap().objects.len(), 3);
    }

    #[test]
    fn a_comment_anywhere_in_the_document_is_refused() {
        // RV7-02, and the residual whose stated reason it falsifies.
        //
        // Rounds 4 and 5 accepted a comment in the prolog and after the
        // root, disclosed as legal XML Misc that "cannot carry a roster
        // entry out of a roster it is not inside".  Moving the entry OUT of
        // the root and INTO the comment is exactly how the roster is
        // shortened, and re-verification #7 measured it: both positions were
        // accepted with a two-object roster, a <KeyCount> lowered to agree,
        // and the entry still in the bytes.  The two arms below were an
        // acceptance control and are a refusal census now -- deliberately
        // inverted, because what they pinned was measured to be false.
        //
        // The fixture evidence for refusing rather than bounding: not one of
        // the captured real S3 bodies carries a comment in any position, so
        // no real page is lost to the stricter rule.
        for (label, page) in [
            ("truncated", REAL_TRUNCATED_PAGE),
            ("final", REAL_FINAL_PAGE),
            ("empty", REAL_EMPTY_PAGE),
            ("access denied", REAL_ACCESS_DENIED),
        ] {
            assert!(!page.contains("<!--"), "the real {label} page carries a comment");
        }

        // The census axis: the two positions outside the root, crossed with
        // a raw entry, an escaped entry, a bare '<' and nothing at all.
        let short = edited(REAL_TRUNCATED_PAGE, HIDDEN_ENTRY, "");
        let short = edited(&short, "<KeyCount>3</KeyCount>", "<KeyCount>2</KeyCount>");
        let escaped = HIDDEN_ENTRY.replace('<', "&lt;").replace('>', "&gt;");
        let mut swept = 0usize;
        for (payload, spelling) in [
            (HIDDEN_ENTRY.to_string(), "raw entry"),
            (escaped, "escaped entry"),
            ("<".to_string(), "a bare '<'"),
            (String::new(), "nothing at all"),
        ] {
            let comment = format!("<!--{payload}-->");
            for position in ["prolog", "trailing"] {
                let xml = if position == "prolog" {
                    edited(
                        &short,
                        "<ListBucketResult",
                        &format!("{comment}<ListBucketResult"),
                    )
                } else {
                    format!("{short}{comment}")
                };
                let label = format!("{position} comment carrying {spelling}");
                let problem = scan_xml(&xml).unwrap_err();
                assert!(
                    problem.contains("comment outside its root element"),
                    "{label}: {problem}"
                );
                assert!(
                    problem.contains("carried out of the roster inside"),
                    "{label}: {problem}"
                );
                let err = day_page(&xml).unwrap_err().to_string();
                assert!(err.contains("did not return well-formed XML"), "{label}: {err}");
                swept += 1;
            }
        }
        assert_eq!(swept, 8, "two positions outside the root x four payloads");

        // And the audit's own two reproductions, with the assertion that
        // makes them carriers rather than junk: the entry is still in the
        // bytes of a page whose <KeyCount> says two.
        for xml in [
            edited(
                &short,
                "<ListBucketResult",
                &format!("<!--{HIDDEN_ENTRY}--><ListBucketResult"),
            ),
            format!("{short}<!--{HIDDEN_ENTRY}-->"),
        ] {
            assert!(xml.contains(HIDDEN_ENTRY), "the entry is still in the bytes");
            assert!(xml.contains("<KeyCount>2</KeyCount>"));
            let err = day_page(&xml).unwrap_err().to_string();
            assert!(err.contains("comment outside its root element"), "{err}");
        }

        // Inside the root it is refused, named by the element it sits in.
        let xml = edited(
            REAL_TRUNCATED_PAGE,
            "<KeyCount>3</KeyCount>",
            "<!--c--><KeyCount>3</KeyCount>",
        );
        let err = day_page(&xml).unwrap_err().to_string();
        assert!(err.contains("comment inside <ListBucketResult>"), "{err}");

        // The carrier: a real <Contents> moved inside a comment with
        // <KeyCount> lowered to agree.  Ok with two objects, before this.
        let xml = edited(
            REAL_TRUNCATED_PAGE,
            HIDDEN_ENTRY,
            &format!("<!--{}-->", HIDDEN_ENTRY),
        );
        let xml = edited(&xml, "<KeyCount>3</KeyCount>", "<KeyCount>2</KeyCount>");
        let err = day_page(&xml).unwrap_err().to_string();
        assert!(err.contains("comment inside <ListBucketResult>"), "{err}");

        // And the value carrier: comment markup inside character data used to
        // be read back as part of the key and of the last_modified string
        // this client calls the object's identity in time.
        for (from, to, element) in [
            (
                "<Key>2026/07/29/KTLX/KTLX20260729_000234_V06</Key>",
                "<Key>2026/07/29/KTLX/<!--c-->KTLX20260729_000234_V06</Key>",
                "Key",
            ),
            (
                "<LastModified>2026-07-29T00:09:29.000Z</LastModified>",
                "<LastModified>2026-07-29<!--c-->T00:09:29.000Z</LastModified>",
                "LastModified",
            ),
        ] {
            let xml = edited(REAL_TRUNCATED_PAGE, from, to);
            let err = day_page(&xml).unwrap_err().to_string();
            assert!(err.contains(&format!("comment inside <{element}>")), "{err}");
        }
    }

    #[test]
    fn a_value_is_character_data_and_not_the_bytes_between_the_tags() {
        // The control: the real page's values are exactly what it states.
        let page = day_page(REAL_TRUNCATED_PAGE).unwrap();
        assert_eq!(page.objects[0].key, "2026/07/29/KTLX/KTLX20260729_000234_V06");
        assert_eq!(page.objects[0].last_modified, "2026-07-29T00:09:29.000Z");

        // And a value is the DECODED character data rather than the span
        // between the tags: one character in the receipt, five bytes in the
        // source.  Values are accumulated as text while the document is
        // scanned, which is what keeps markup out of them.
        let xml = edited(
            REAL_TRUNCATED_PAGE,
            "<LastModified>2026-07-29T00:09:29.000Z</LastModified>",
            "<LastModified>2026-07-29T00:09:29.000Z&amp;X</LastModified>",
        );
        let page = day_page(&xml).unwrap();
        assert_eq!(page.objects[0].last_modified, "2026-07-29T00:09:29.000Z&X");
    }

    #[test]
    fn an_undeclared_entity_reference_is_refused() {
        let xml = edited(
            REAL_TRUNCATED_PAGE,
            "<LastModified>2026-07-29T00:09:29.000Z</LastModified>",
            "<LastModified>2026-07-29T00:09:29.000Z&undeclared;</LastModified>",
        );
        let err = day_page(&xml).unwrap_err().to_string();
        assert!(err.contains("entity reference"), "{err}");
        assert!(err.contains("&undeclared;"), "{err}");
    }

    #[test]
    fn a_rolled_up_prefix_cannot_answer_for_the_listings_own() {
        // <Prefix> appears twice in a delimited listing: once as the
        // listing's own echoed prefix and once inside each <CommonPrefixes>
        // roll-up.  Reading fields as DIRECT CHILDREN of the root is what
        // keeps the nested one from being mistaken for the echo -- the
        // roll-up itself is refused, and the refusal names it rather than a
        // prefix mismatch that would send an operator after the wrong fault.
        let xml = edited(
            REAL_TRUNCATED_PAGE,
            "<Prefix>2026/07/29/KTLX/</Prefix>",
            "<CommonPrefixes><Prefix>2026/07/29/KOUN/</Prefix></CommonPrefixes>\
             <Prefix>2026/07/29/KTLX/</Prefix>",
        );
        let err = day_page(&xml).unwrap_err().to_string();
        assert!(err.contains("CommonPrefixes"), "{err}");
        assert!(!err.contains("echoes prefix"), "{err}");
    }

    #[test]
    fn a_key_count_the_page_does_not_meet_is_refused() {
        // The bucket's own count of what it put in the page.  Both
        // directions: a page claiming more entries than it carries is the
        // short-body signature; one claiming fewer is a page nobody wrote.
        for (declared, expected) in [("2", "KeyCount 2"), ("4", "KeyCount 4"), ("0", "KeyCount 0")] {
            let xml = edited(
                REAL_TRUNCATED_PAGE,
                "<KeyCount>3</KeyCount>",
                &format!("<KeyCount>{declared}</KeyCount>"),
            );
            let err = day_page(&xml).unwrap_err().to_string();
            assert!(err.contains(expected), "{declared}: {err}");
        }
        for replacement in ["", "<KeyCount>many</KeyCount>"] {
            let xml = edited(REAL_TRUNCATED_PAGE, "<KeyCount>3</KeyCount>", replacement);
            let err = day_page(&xml).unwrap_err().to_string();
            assert!(err.contains("KeyCount"), "{replacement:?}: {err}");
        }
        // KeyCount over MaxKeys is a page that broke its own limit.
        let xml = edited(REAL_TRUNCATED_PAGE, "<MaxKeys>3</MaxKeys>", "<MaxKeys>2</MaxKeys>");
        let err = day_page(&xml).unwrap_err().to_string();
        assert!(err.contains("over MaxKeys 2"), "{err}");
    }

    #[test]
    fn a_size_is_never_defaulted() {
        // The listed size is what `download_volume` checks the downloaded
        // byte count against.  Defaulting a missing one to 0 disables that
        // check, so a truncated object would download and verify.
        for replacement in ["", "<Size></Size>", "<Size>-1</Size>", "<Size>5.0e6</Size>", "<Size>five</Size>"] {
            let xml = edited(REAL_TRUNCATED_PAGE, "<Size>5062263</Size>", replacement);
            let err = day_page(&xml).unwrap_err().to_string();
            assert!(err.contains("<Size>"), "{replacement:?}: {err}");
            assert!(
                err.contains("KTLX20260729_000234_V06"),
                "the refusal must name the entry: {err}"
            );
        }
    }

    #[test]
    fn an_entry_missing_its_key_or_timestamp_is_refused() {
        for replacement in ["", "<Key></Key>"] {
            let xml = edited(
                REAL_TRUNCATED_PAGE,
                "<Key>2026/07/29/KTLX/KTLX20260729_000234_V06</Key>",
                replacement,
            );
            let err = day_page(&xml).unwrap_err().to_string();
            assert!(err.contains("<Key>"), "{replacement:?}: {err}");
        }
        for replacement in ["", "<LastModified></LastModified>", "<LastModified> </LastModified>"] {
            let xml = edited(
                REAL_TRUNCATED_PAGE,
                "<LastModified>2026-07-29T00:09:29.000Z</LastModified>",
                replacement,
            );
            let err = day_page(&xml).unwrap_err().to_string();
            assert!(err.contains("<LastModified>"), "{replacement:?}: {err}");
        }
    }

    #[test]
    fn a_prefix_placeholder_is_not_counted_as_an_object() {
        // Two shapes of the same thing: a zero-byte directory marker for the
        // day prefix itself, and one for a subdirectory under it.
        for key in ["2026/07/29/KTLX/", "2026/07/29/KTLX/chunks/"] {
            let xml = edited(
                REAL_TRUNCATED_PAGE,
                "<Key>2026/07/29/KTLX/KTLX20260729_000234_V06</Key>",
                &format!("<Key>{key}</Key>"),
            );
            let err = day_page(&xml).unwrap_err().to_string();
            assert!(err.contains("directory placeholder"), "{key}: {err}");
        }
        // A key outside the prefix that was asked for is a different listing.
        for key in ["2026/07/29/KOUN/KOUN20260729_000234_V06", "README.txt"] {
            let xml = edited(
                REAL_TRUNCATED_PAGE,
                "<Key>2026/07/29/KTLX/KTLX20260729_000234_V06</Key>",
                &format!("<Key>{key}</Key>"),
            );
            let err = day_page(&xml).unwrap_err().to_string();
            assert!(err.contains("outside the prefix"), "{key}: {err}");
        }
    }

    /// The traversal key, verbatim, as the second auditor measured it.
    const ESCAPING_KEY: &str =
        "2026/07/29/KTLX/../../../../../../../../rv8-escaped/KTLX20260729_000234_V06";

    /// The same move aimed at another bucket's cache slot rather than out of
    /// the cache entirely.
    const CROSS_BUCKET_KEY: &str = "2026/07/29/KTLX/../../../../\
                                    noaa-nexrad-level2/2026/07/29/KTLX/\
                                    KTLX20260729_000234_V06";

    #[test]
    fn a_key_that_walks_out_of_the_cache_directory_is_refused() {
        // RV8-PATH, and a PRE-EXISTING defect: byte-identical at the merge
        // target, untouched by this branch, and closed here because this
        // branch is what carries the reader into the integration branch.
        //
        // The finding is that every question the roster asked of a key
        // answered yes for a key that is not a name at all but a path.
        assert!(ESCAPING_KEY.starts_with(DAY_PREFIX), "it is under the prefix asked for");
        assert!(!ESCAPING_KEY.ends_with('/'), "it is not a directory placeholder");
        assert_eq!(
            parse_volume_key(ESCAPING_KEY).unwrap().site,
            "KTLX",
            "and it parses as a KTLX volume out of its last segment"
        );

        // So it is refused where the key is accepted, by name.
        let mut swept = 0usize;
        for (label, key, fault) in [
            ("out of the cache root", ESCAPING_KEY, "'..' path segment"),
            ("into another bucket's slot", CROSS_BUCKET_KEY, "'..' path segment"),
            (
                "a '.' segment",
                "2026/07/29/KTLX/./KTLX20260729_000234_V06",
                "'.' path segment",
            ),
            (
                "an empty segment",
                "2026/07/29/KTLX//KTLX20260729_000234_V06",
                "empty path segment",
            ),
            (
                "a backslash",
                "2026/07/29/KTLX/..\\..\\KTLX20260729_000234_V06",
                "backslash",
            ),
            (
                "a drive letter",
                "2026/07/29/KTLX/C:/KTLX20260729_000234_V06",
                "colon",
            ),
        ] {
            let xml = edited(
                REAL_TRUNCATED_PAGE,
                "<Key>2026/07/29/KTLX/KTLX20260729_000234_V06</Key>",
                &format!("<Key>{key}</Key>"),
            );
            assert!(scan_xml(&xml).is_ok(), "{label}: the mutation is well-formed");
            let err = day_page(&xml).unwrap_err().to_string();
            assert!(err.contains(fault), "{label}: {err}");
            assert!(
                err.contains(&format!("{key:?}")),
                "{label}: the refusal must name the key: {err}"
            );
            assert!(err.contains("will not localize"), "{label}: {err}");
            swept += 1;
        }
        assert_eq!(swept, 6, "one mutation per segment shape that is not a name");

        // A leading '/' cannot also start with a day prefix, so that arm is
        // asserted on the predicate rather than through a listing.
        assert!(key_localization_fault("/etc/passwd")
            .unwrap()
            .contains("begins with '/'"));
        // And an ordinary key has no fault at all.
        assert_eq!(
            key_localization_fault("2026/07/29/KTLX/KTLX20260729_000234_V06"),
            None
        );

        // The control: the real page's own keys are ordinary names, and the
        // page still parses to its three objects.
        assert_eq!(day_page(REAL_TRUNCATED_PAGE).unwrap().objects.len(), 3);
    }

    #[test]
    fn a_cache_path_is_the_key_under_the_bucket_root_or_it_is_a_refusal() {
        // The defence-in-depth half of RV8-PATH: the boundary check above
        // guards the one door a key currently comes through, and this guards
        // the write for any caller that ever reaches the builder by another
        // route.
        let root = std::env::temp_dir().join(format!("rw_nexrad_rv8_path_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let slot = root.join("nexrad").join(DEFAULT_BUCKET);

        // The control: an ordinary key resolves under the bucket's slot and
        // the bytes land there.
        let key = "2026/07/29/KTLX/KTLX20260729_000234_V06";
        let target = cached_volume_path(&root, DEFAULT_BUCKET, key).unwrap();
        assert!(target.starts_with(&slot), "{}", target.display());
        assert!(
            target
                .strip_prefix(&slot)
                .unwrap()
                .components()
                .all(|component| matches!(component, Component::Normal(_))),
            "{}",
            target.display()
        );
        atomic_write_bytes(&target, b"volume bytes").unwrap();
        assert_eq!(std::fs::read(&target).unwrap(), b"volume bytes");

        // What the builder used to produce for the traversal key, built the
        // way it used to be built: a path with `..` components in it, which
        // is a path outside the slot no matter how the filesystem resolves
        // it.  This is the measurement, and it is done without writing
        // anywhere outside this test's own scratch directory.
        for key in [ESCAPING_KEY, CROSS_BUCKET_KEY] {
            let mut unchecked = slot.clone();
            for segment in key.split('/') {
                unchecked.push(segment);
            }
            assert!(
                unchecked
                    .components()
                    .any(|component| matches!(component, Component::ParentDir)),
                "{key}: the unchecked build is what escapes; without a \
                 ParentDir in it this assertion proves nothing"
            );
            let err = cached_volume_path(&root, DEFAULT_BUCKET, key)
                .unwrap_err()
                .to_string();
            assert!(err.contains("does not name a file under the cache directory"), "{err}");
            assert!(
                err.contains(&format!("{key:?}")),
                "the refusal must name the key: {err}"
            );
        }

        // A segment carrying a drive prefix is the case a string check of
        // the tail would miss: `Path::push` does not append it, it REPLACES
        // the whole path with it.
        for key in [
            "2026/C:/KTLX20260729_000234_V06",
            "2026/C:KTLX20260729_000234_V06",
            "2026/..\\KTLX20260729_000234_V06",
            "2026/KTLX20260729_000234_V06:stream",
        ] {
            let err = cached_volume_path(&root, DEFAULT_BUCKET, key)
                .unwrap_err()
                .to_string();
            assert!(err.contains("does not name a file under the cache directory"), "{err}");
        }

        // And `publish_volume` holds the same rule one directory deep: the
        // key's last segment is a file name in the requested directory or it
        // is a refusal.
        let out = root.join("out");
        let published = publish_volume(&target, &out, key).unwrap();
        assert_eq!(published, out.join("KTLX20260729_000234_V06"));
        assert_eq!(std::fs::read(&published).unwrap(), b"volume bytes");
        for key in [
            "2026/07/29/KTLX/..", "..", "2026/07/29/KTLX/..\\escaped",
            "2026/07/29/KTLX/C:escaped", "2026/07/29/KTLX/name:stream",
        ] {
            let err = publish_volume(&target, &out, key).unwrap_err().to_string();
            assert!(
                err.contains("not a file directly inside the requested output directory"),
                "{key}: {err}"
            );
        }

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_listing_rolled_up_on_a_delimiter_is_refused() {
        for rollup in [
            "<CommonPrefixes><Prefix>2026/07/29/KTLX/a/</Prefix></CommonPrefixes>",
            "<CommonPrefixes><Prefix>2026/07/29/KTLX/b/</Prefix></CommonPrefixes><CommonPrefixes><Prefix>2026/07/29/KTLX/c/</Prefix></CommonPrefixes>",
        ] {
            let xml = edited(REAL_TRUNCATED_PAGE, "<KeyCount>3</KeyCount>", &format!("{rollup}<KeyCount>3</KeyCount>"));
            let err = day_page(&xml).unwrap_err().to_string();
            assert!(err.contains("CommonPrefixes"), "{err}");
        }
    }

    #[test]
    fn a_listing_of_some_other_bucket_or_prefix_is_refused() {
        for other in ["noaa-nexrad-level2", "unidata-nexrad-level2-staging"] {
            let err = parse_s3_list_xml(REAL_TRUNCATED_PAGE, other, DAY_PREFIX)
                .unwrap_err()
                .to_string();
            assert!(err.contains("names bucket"), "{other}: {err}");
        }
        for other in ["2026/07/29/KOUN/", "2026/07/28/KTLX/"] {
            let err = parse_s3_list_xml(REAL_TRUNCATED_PAGE, BUCKET, other)
                .unwrap_err()
                .to_string();
            assert!(err.contains("echoes prefix"), "{other}: {err}");
        }
    }

    #[test]
    fn an_error_document_is_not_a_listing_of_zero_objects() {
        // The archive of record answers exactly this to anonymous listing.
        let err = day_page(REAL_ACCESS_DENIED).unwrap_err().to_string();
        assert!(err.contains("AccessDenied"), "{err}");
        assert!(err.contains("not a <ListBucketResult>"), "{err}");
        // So does an empty body, and a body that is not XML at all.
        for body in ["", "<html><body>502 Bad Gateway</body></html>"] {
            let err = day_page(body).unwrap_err().to_string();
            assert!(err.contains("not a <ListBucketResult>"), "{body:?}: {err}");
        }
    }

    #[test]
    fn the_continuation_token_must_advance() {
        let page = day_page(REAL_TRUNCATED_PAGE).unwrap();
        let token = page.next_continuation_token.clone().unwrap();

        // First page: no token was used, so the bucket's is an advance.
        let mut spent = Vec::new();
        assert_eq!(
            advance_continuation(None, &page, &mut spent, 1).unwrap(),
            Some(token.clone())
        );
        assert_eq!(spent, vec![token.clone()]);

        // Handing back the token the request carried is a page that repeats
        // itself: following it never terminates, ignoring it drops the rest.
        let mut spent = Vec::new();
        let err = advance_continuation(Some(&token), &page, &mut spent, 2)
            .unwrap_err()
            .to_string();
        assert!(err.contains("did not advance"), "{err}");

        // A token already spent earlier in the same walk is the same fault
        // one page removed.
        let mut spent = vec![token.clone()];
        let err = advance_continuation(Some("some-other-token"), &page, &mut spent, 3)
            .unwrap_err()
            .to_string();
        assert!(err.contains("looped"), "{err}");

        // A complete page ends the walk and spends nothing.
        let final_page = parse_s3_list_xml(REAL_FINAL_PAGE, BUCKET, ONE_KEY_PREFIX).unwrap();
        let mut spent = Vec::new();
        assert_eq!(
            advance_continuation(Some("whatever"), &final_page, &mut spent, 4).unwrap(),
            None
        );
        assert!(spent.is_empty());
    }

    #[test]
    fn a_refused_listing_names_the_capability_and_the_way_round_it() {
        // A bare "HTTP 403" at a CLI's exit reads as a bad site id or a bad
        // window.  It is neither: the bucket is declining s3:ListBucket to
        // an anonymous caller, and no argument the user changes fixes it.
        for status in [401u16, 403] {
            let err = anonymous_access_refused(ARCHIVE_OF_RECORD_BUCKET, DAY_PREFIX, status)
                .to_string();
            assert!(err.contains(&format!("HTTP {status}")), "{err}");
            assert!(err.contains("not granting s3:ListBucket"), "{err}");
            // The remedy names the *other* bucket, whichever one failed.
            assert!(err.contains(&format!("--bucket {DEFAULT_BUCKET}")), "{err}");
        }
        let err = anonymous_access_refused(DEFAULT_BUCKET, DAY_PREFIX, 403).to_string();
        assert!(
            err.contains(&format!("--bucket {ARCHIVE_OF_RECORD_BUCKET}")),
            "{err}"
        );
    }

    #[test]
    fn pagination_is_bounded() {
        let page = day_page(REAL_TRUNCATED_PAGE).unwrap();
        let mut spent = Vec::new();
        let err = advance_continuation(None, &page, &mut spent, MAX_LIST_PAGES)
            .unwrap_err()
            .to_string();
        assert!(err.contains(&MAX_LIST_PAGES.to_string()), "{err}");
        // One page below the ceiling is still a normal advance.
        let mut spent = Vec::new();
        assert!(advance_continuation(None, &page, &mut spent, MAX_LIST_PAGES - 1)
            .unwrap()
            .is_some());
    }

    #[test]
    fn day_prefix_is_zero_padded_and_site_scoped() {
        let day = NaiveDate::from_ymd_opt(2023, 5, 7).unwrap();
        assert_eq!(day_prefix("KTLX", day), "2023/05/07/KTLX/");
    }

    #[test]
    fn window_days_covers_every_utc_day_it_touches() {
        let start = Utc.with_ymd_and_hms(2023, 5, 20, 23, 50, 0).unwrap();
        let end = Utc.with_ymd_and_hms(2023, 5, 22, 0, 10, 0).unwrap();
        let days = window_days(start, end);
        assert_eq!(days.len(), 3);
        assert_eq!(days[0], NaiveDate::from_ymd_opt(2023, 5, 20).unwrap());
        assert_eq!(days[2], NaiveDate::from_ymd_opt(2023, 5, 22).unwrap());
        // A window inside one day yields exactly that day.
        assert_eq!(window_days(start, start).len(), 1);
    }

    #[test]
    fn volume_key_parses_site_time_and_format() {
        let parsed = parse_volume_key("2023/05/20/KTLX/KTLX20230520_200356_V06").unwrap();
        assert_eq!(parsed.site, "KTLX");
        assert_eq!(parsed.format, "V06");
        assert!(!parsed.gzipped);
        assert_eq!(iso8601(parsed.valid_time), "2023-05-20T20:03:56Z");
    }

    #[test]
    fn volume_key_reads_the_gzipped_pre_2016_era_and_the_plain_era_alike() {
        // Real keys from both eras of the same archive.  Rejecting the `.gz`
        // spelling silently removed every volume before roughly 2016 from
        // any window a caller asked for -- the listing simply came back
        // shorter, with no error anywhere: the 2013-05-20 Moore case, and
        // the whole `V03` era before it.
        for (key, format, gzipped, when) in [
            (
                "2013/05/20/KTLX/KTLX20130520_195111_V06.gz",
                "V06",
                true,
                "2013-05-20T19:51:11Z",
            ),
            (
                "2011/05/24/KTLX/KTLX20110524_000041_V03.gz",
                "V03",
                true,
                "2011-05-24T00:00:41Z",
            ),
            (
                "2019/05/20/KTLX/KTLX20190520_000034_V06",
                "V06",
                false,
                "2019-05-20T00:00:34Z",
            ),
            (
                "2026/07/29/KTLX/KTLX20260729_000234_V06",
                "V06",
                false,
                "2026-07-29T00:02:34Z",
            ),
        ] {
            let parsed = parse_volume_key(key).unwrap_or_else(|| panic!("{key}"));
            assert_eq!(parsed.site, "KTLX", "{key}");
            assert_eq!(parsed.format, format, "{key}");
            assert_eq!(parsed.gzipped, gzipped, "{key}");
            assert_eq!(iso8601(parsed.valid_time), when, "{key}");
        }
    }

    #[test]
    fn volume_key_rejects_metadata_siblings_and_junk() {
        // `_MDM` is a per-volume metadata sidecar, not radar data -- in
        // either spelling, so the `.gz` suffix cannot smuggle one in.
        assert!(parse_volume_key("2023/05/20/KTLX/KTLX20230520_200356_V06_MDM").is_none());
        assert!(parse_volume_key("2013/05/20/KTLX/KTLX20130520_200356_V06_MDM.gz").is_none());
        // Chunk remnants on the rolling mirror end in a numeric suffix.
        assert!(parse_volume_key("2026/07/29/KTLX/KTLX20260729_200356_V08.001").is_none());
        assert!(parse_volume_key("2013/05/20/KTLX/KTLX20130520_200356_V08.001.gz").is_none());
        assert!(parse_volume_key("2026/07/29/KTLX/KTLX20260729_200356_V08").is_some());
        // A suffix that is not exactly a trailing `.gz` is not a volume.
        for junk in [
            "2013/05/20/KTLX/KTLX20130520_200356_V06.gzip",
            "2013/05/20/KTLX/KTLX20130520_200356_V06.gz.gz",
            "2013/05/20/KTLX/KTLX20130520_200356_.gz",
            "2013/05/20/KTLX/KTLX20130520_200356_V06.bz2",
        ] {
            assert!(parse_volume_key(junk).is_none(), "{junk}");
        }
        assert!(parse_volume_key("2023/05/20/KTLX/").is_none());
        assert!(parse_volume_key("2023/05/20/KTLX/README.txt").is_none());
        // A plausible length but a non-numeric stamp must not parse, in
        // either spelling.
        assert!(parse_volume_key("2023/05/20/KTLX/KTLXYYYYMMDD_200356_V06").is_none());
        assert!(parse_volume_key("2013/05/20/KTLX/KTLXYYYYMMDD_200356_V06.gz").is_none());
        // And a real date that does not exist is refused with or without it.
        assert!(parse_volume_key("2013/02/30/KTLX/KTLX20130230_200356_V06").is_none());
        assert!(parse_volume_key("2013/02/30/KTLX/KTLX20130230_200356_V06.gz").is_none());
    }

    #[test]
    fn volume_key_skips_a_non_ascii_sibling_instead_of_panicking() {
        // audit dat-02-09.  The length guard is in BYTES and every slice
        // below it is at a fixed BYTE offset, so a multi-byte character
        // inside the first twenty bytes used to panic on a non-char-boundary
        // index rather than return None.  `key_localization_fault`'s own
        // note applies verbatim: no forged response is needed, because S3
        // keys may legally contain these bytes and the listed bucket is a
        // public mirror -- so one junk sibling took down the whole window
        // and every volume in it, not just itself.
        for key in [
            // Seven U+20AC (3 bytes each) + `_V06` = 25 bytes: the `< 20`
            // BYTE guard passes and `&name[..4]` lands inside the second one.
            "2023/05/20/KTLX/\u{20ac}\u{20ac}\u{20ac}\u{20ac}\u{20ac}\u{20ac}\u{20ac}_V06",
            // A character boundary inside each of the other fixed slices.
            "2023/05/20/KTLX/KTLX2023\u{20ac}520_200356_V06",
            "2023/05/20/KTLX/KTLX20230520_20\u{20ac}356_V06",
            "2023/05/20/KTLX/KTLX20230520_200356\u{20ac}V06",
        ] {
            assert!(parse_volume_key(key).is_none(), "{key}");
        }
        // The guard reads the NAME, not the whole key: a non-ASCII directory
        // component above a well-formed volume name is still a volume, and
        // the _MDM sidecar beside it is still refused for being a sidecar.
        let parsed = parse_volume_key(
            "2023/05/20/K\u{20ac}LX/KTLX20230520_200356_V06",
        )
        .expect("a non-ASCII directory does not disqualify the object");
        assert_eq!(parsed.site, "KTLX");
        assert!(parse_volume_key(
            "2023/05/20/K\u{20ac}LX/KTLX20230520_200356_V06_MDM"
        )
        .is_none());
    }

    #[test]
    fn bucket_normalization_fails_closed_on_malformed_names() {
        assert_eq!(normalize_bucket("NOAA-NEXRAD-Level2").unwrap(), "noaa-nexrad-level2");
        assert_eq!(normalize_bucket(DEFAULT_BUCKET).unwrap(), DEFAULT_BUCKET);
        assert_eq!(
            normalize_bucket(ARCHIVE_OF_RECORD_BUCKET).unwrap(),
            ARCHIVE_OF_RECORD_BUCKET
        );
        assert!(normalize_bucket("no").is_err());
        assert!(normalize_bucket("-leading").is_err());
        assert!(normalize_bucket("has space").is_err());
    }

    #[test]
    fn site_normalization_fails_closed_on_malformed_ids() {
        assert_eq!(normalize_site("ktlx").unwrap(), "KTLX");
        assert_eq!(normalize_site(" KOUN ").unwrap(), "KOUN");
        assert!(normalize_site("KTL").is_err());
        assert!(normalize_site("KTLX1").is_err());
        assert!(normalize_site("KT-X").is_err());
    }

    #[test]
    fn query_encoding_escapes_reserved_bytes() {
        assert_eq!(url_query_encode("2023/05/20/KTLX/"), "2023%2F05%2F20%2FKTLX%2F");
        assert_eq!(url_query_encode("KTLX_V06"), "KTLX_V06");
    }

    #[test]
    fn time_parsing_accepts_both_spellings_and_refuses_the_rest() {
        assert_eq!(
            iso8601(parse_time("2023-05-20T20:00:00Z").unwrap()),
            "2023-05-20T20:00:00Z"
        );
        assert_eq!(
            iso8601(parse_time("20230520T200000").unwrap()),
            "2023-05-20T20:00:00Z"
        );
        assert!(parse_time("2023-05-20").is_err());
        assert!(parse_time("yesterday").is_err());
    }

    #[test]
    fn cache_path_nests_the_key_under_bucket() {
        let root = Path::new("C:").join("cache");
        let path =
            cached_volume_path(&root, DEFAULT_BUCKET, "2023/05/20/KTLX/KTLX20230520_200356_V06")
                .unwrap();
        assert!(path.ends_with(
            Path::new("nexrad")
                .join(DEFAULT_BUCKET)
                .join("2023")
                .join("05")
                .join("20")
                .join("KTLX")
                .join("KTLX20230520_200356_V06")
        ));
    }

    #[test]
    fn agent_has_every_io_phase_timeout_set() {
        let timeouts = build_agent().config().timeouts();
        assert_eq!(timeouts.resolve, Some(CONNECT_TIMEOUT));
        assert_eq!(timeouts.connect, Some(CONNECT_TIMEOUT));
        assert_eq!(timeouts.send_request, Some(SEND_REQUEST_TIMEOUT));
        assert_eq!(timeouts.recv_response, Some(RECV_RESPONSE_TIMEOUT));
        assert_eq!(timeouts.recv_body, Some(RECV_BODY_TIMEOUT));
    }

    #[test]
    fn sha256_is_the_standard_empty_digest() {
        assert_eq!(
            hex_sha256(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }
}
