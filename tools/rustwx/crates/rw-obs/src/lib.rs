//! Shared substance for the observation front doors.
//!
//! Three instruments, one container, one clock spelling, one receipt shape.
//! Everything an individual bin does *not* need to decide lives here so the
//! three cannot drift into three dialects of the same contract.
//!
//! The pack is the `GPWMRDR1` layout retold for gridded observations: an
//! 8-byte magic, a `u32` version, a length-prefixed JSON metadata block, and
//! one contiguous little-endian payload. A Python consumer needs `json` and
//! `numpy.frombuffer` and nothing else.

pub mod cli;
pub mod pack;
pub mod seam;
pub mod net;
pub mod bufr;
pub mod bufr_obs;
pub mod ncep_bufr;
pub mod errtable;
pub mod prepbufr;
pub mod fddaobs;
pub mod fddaobs_policy;
pub mod table;
pub mod tar;
pub mod precipitation;
pub mod precipitation_hdf;

use std::error::Error;

/// The crate's one error constructor, so every refusal reads alike.
pub fn err(message: impl Into<String>) -> Box<dyn Error> {
    rw_nexrad::s3::boxed_error(message)
}

/// Hex SHA-256, borrowed from the S3 client so a fetch receipt and a pack
/// digest are provably the same function.
pub use rw_nexrad::s3::hex_sha256;

/// The absolute path a receipt should record for `path`.
///
/// Provenance is re-hashed by a scoring pass that runs later, from a
/// different working directory, on a different machine. A relative `uri`
/// resolves against whatever directory the reader happens to be in, so it
/// names the right file exactly once (in the shell the decode was typed
/// into) and a missing file afterwards reads as a corrupted archive rather
/// than as a path that was never portable.
///
/// The `\\?\` verbatim prefix Windows canonicalization adds is stripped: it
/// is correct and it is also rejected by a good deal of tooling that will
/// read these receipts, and the path without it names the same file.
pub fn absolute_uri(path: &std::path::Path) -> String {
    let resolved = match path.canonicalize() {
        Ok(resolved) => resolved,
        // A path that cannot be canonicalized (it was just deleted, or the
        // platform refuses) is still better recorded as-given than dropped.
        Err(_) => match std::env::current_dir() {
            Ok(cwd) => cwd.join(path),
            Err(_) => path.to_path_buf(),
        },
    };
    let text = resolved.to_string_lossy().to_string();
    match text.strip_prefix(r"\\?\") {
        Some(stripped) => stripped.to_string(),
        None => text,
    }
}

/// Largest payload this front door will expand a gzip member into.
///
/// An MRMS composite is ~1.7 MB gzipped and ~49 MB expanded, so 512 MB is
/// two orders of magnitude of headroom over anything the archive holds. It
/// is not a tuning parameter: it is the bound that stops a malformed or
/// hostile member from being decompressed until the process dies.
pub const MAX_EXPANDED_BYTES: u64 = 512 * 1024 * 1024;

/// Expand a gzip-wrapped archive object, or hand the bytes back unchanged.
///
/// This is `rw_nexrad::pack::gunzip_if_wrapped`'s argument in generic
/// wording, including its truncation check, which is the part that matters
/// and the part a naive `GzDecoder` call omits. A deflate stream can end on
/// a block boundary with the 8-byte trailer missing, and the reader then
/// returns the bytes it got and *no error*; the trailer's ISIZE is the
/// member's own statement of its expanded length and is the last thing a
/// truncation removes. A prefix of a reflectivity field is not a small
/// field, it is the northern rows with no way to tell how many are missing.
///
/// The bytes as they arrived stay the object's identity: the archive
/// listing's size and the fetch receipt's SHA-256 are both over those, never
/// over the expansion.
pub fn gunzip_if_wrapped(raw: &[u8], subject: &str) -> Result<(Vec<u8>, bool), Box<dyn Error>> {
    use std::io::Read;
    if !raw.starts_with(&[0x1f, 0x8b]) {
        return Ok((raw.to_vec(), false));
    }
    if raw.len() < 18 {
        return Err(err(format!(
            "{subject} is {} bytes, too few to hold a gzip member's header and trailer",
            raw.len()
        )));
    }
    let mut expanded = Vec::new();
    let mut decoder =
        flate2::read::GzDecoder::new(raw).take(MAX_EXPANDED_BYTES + 1);
    decoder
        .read_to_end(&mut expanded)
        .map_err(|e| err(format!("{subject} will not expand: {e}")))?;
    if expanded.len() as u64 > MAX_EXPANDED_BYTES {
        return Err(err(format!(
            "{subject} expands past the {MAX_EXPANDED_BYTES}-byte ceiling; refusing rather \
             than decompressing without bound"
        )));
    }
    let tail = &raw[raw.len() - 4..];
    let declared = u32::from_le_bytes([tail[0], tail[1], tail[2], tail[3]]);
    if declared != (expanded.len() as u64 % (1u64 << 32)) as u32 {
        return Err(err(format!(
            "{subject} is incomplete: its trailer declares {declared} expanded bytes and {} \
             came out. A member cut short can still decode to a prefix without erroring, and \
             a prefix of a field is not a small field",
            expanded.len()
        )));
    }
    Ok((expanded, true))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn gzip(payload: &[u8]) -> Vec<u8> {
        let mut encoder =
            flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
        encoder.write_all(payload).unwrap();
        encoder.finish().unwrap()
    }

    #[test]
    fn plain_bytes_pass_through_and_say_they_were_not_wrapped() {
        let (out, wrapped) = gunzip_if_wrapped(b"GRIB....", "an object").unwrap();
        assert_eq!(out, b"GRIB....");
        assert!(!wrapped);
    }

    #[test]
    fn a_whole_member_expands_and_reports_that_it_was_wrapped() {
        let payload = vec![7u8; 4096];
        let (out, wrapped) = gunzip_if_wrapped(&gzip(&payload), "an object").unwrap();
        assert_eq!(out, payload);
        assert!(wrapped);
    }

    #[test]
    fn a_member_cut_short_is_refused_rather_than_returned_as_a_prefix() {
        // The failure this check exists for: truncate inside the deflate
        // stream but leave four bytes that will be read as an ISIZE. The
        // decoder may hand back a clean prefix, and only the trailer
        // comparison can tell.
        let payload: Vec<u8> = (0..40_000u32).map(|v| (v % 251) as u8).collect();
        let whole = gzip(&payload);
        let cut = &whole[..whole.len() - 40];
        let outcome = gunzip_if_wrapped(cut, "an archive object");
        assert!(outcome.is_err(), "a truncated member must not pass");
        let message = outcome.unwrap_err().to_string();
        assert!(message.contains("an archive object"), "{message}");
    }

    #[test]
    fn a_stub_too_short_to_hold_a_trailer_is_refused_by_length() {
        let err = gunzip_if_wrapped(&[0x1f, 0x8b, 0x08, 0x00], "an object")
            .unwrap_err()
            .to_string();
        assert!(err.contains("too few"), "{err}");
    }
}
pub mod csv;

pub mod hail;
