//! Durable memos that let one render reuse what an earlier render of the
//! same, unchanged history files already computed: a file's whole-content
//! SHA-256 fingerprint, and the running snowfall accumulation of a WRF
//! history.
//!
//! WHAT BREAKAGE THIS PREVENTS (gate law): `woof go` draws each history
//! frame as it lands, one `rw_wrfbatch` process per frame, and finishes with
//! one windowed pass over the whole series.  Each of those processes hashed
//! every byte of every source it named, and the snowfall windows name every
//! earlier frame of the domain, so drawing frame N re-read all N frames
//! (63 GB at F024 of a 3 km 796x636x49 run) and re-integrated the snowfall
//! from F000, reading 3-D temperature and pressure of every earlier frame.
//! The live render of one frame grew from 14 s at F001 to 63-80 s at F022-F024
//! on a lab RTX 5090 host, and the windowed pass re-read all 63 GB again.
//!
//! A memo is used only when the file is provably the one it was made from:
//! same canonical path, same name, same length, same modification and
//! status-change times to the nanosecond, same device and inode.  Rewriting
//! a file in place changes its status-change time, which no unprivileged
//! process can set back, so a revised frame is never matched.  Only Unix
//! hosts carry those stamps; elsewhere nothing is memoized and every
//! fingerprint is computed in full, exactly as before.  The values a memo
//! returns are bit-for-bit the ones the full computation produces.

use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use sha2::{Digest, Sha256};

/// How long an unused memo is kept before the next write sweeps it.
const MEMO_KEEP: Duration = Duration::from_secs(7 * 24 * 3600);
/// A file modified this recently may still be written; it is not memoized
/// (many times the kernel's coarse timestamp tick).
const SETTLE: Duration = Duration::from_millis(200);

/// Where memos live: `RW_SOURCE_CACHE_DIR`, else `$XDG_CACHE_HOME/rw-wrfbatch`,
/// else `$HOME/.cache/rw-wrfbatch`.  `None` (no memo) when none is set, or
/// when `RW_SOURCE_CACHE_DIR` is `off`.
pub(crate) fn cache_root() -> Option<PathBuf> {
    let set = |name: &str| std::env::var_os(name).filter(|value| !value.is_empty());
    if let Some(dir) = set("RW_SOURCE_CACHE_DIR") {
        if dir == "off" {
            return None;
        }
        return Some(PathBuf::from(dir));
    }
    if let Some(dir) = set("XDG_CACHE_HOME") {
        return Some(PathBuf::from(dir).join("rw-wrfbatch"));
    }
    set("HOME").map(|home| PathBuf::from(home).join(".cache").join("rw-wrfbatch"))
}

/// The identity stamp of one file: canonical path, length, modification and
/// status-change times in ns, device and inode.  `None` off Unix, for a file
/// that cannot be inspected, or for one modified within [`SETTLE`].
pub(crate) fn file_stamp(path: &Path) -> Option<String> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        let canonical = std::fs::canonicalize(path).ok()?;
        let metadata = std::fs::metadata(&canonical).ok()?;
        if !metadata.is_file() {
            return None;
        }
        let modified = metadata.modified().ok()?;
        if SystemTime::now()
            .duration_since(modified)
            .map_or(true, |age| age < SETTLE)
        {
            return None;
        }
        Some(format!(
            "{}\u{0}{} {}.{} {}.{} {} {}",
            canonical.display(),
            metadata.len(),
            metadata.mtime(),
            metadata.mtime_nsec(),
            metadata.ctime(),
            metadata.ctime_nsec(),
            metadata.dev(),
            metadata.ino()
        ))
    }
    #[cfg(not(unix))]
    {
        let _ = path;
        None
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn key_path(folder: &str, key: &str, suffix: &str) -> Option<PathBuf> {
    let digest = Sha256::digest(key.as_bytes());
    Some(
        cache_root()?
            .join(folder)
            .join(format!("{}{suffix}", &hex(&digest)[..32])),
    )
}

/// Write `bytes` to `path` atomically (a sibling temporary, then a rename),
/// then sweep memos in the same folder unused for [`MEMO_KEEP`].  Failures
/// are ignored: a memo that cannot be written is recomputed next time.
fn write_atomically(path: &Path, bytes: &[u8]) {
    let Some(folder) = path.parent() else { return };
    if std::fs::create_dir_all(folder).is_err() {
        return;
    }
    let temporary = folder.join(format!(
        ".{}.{}.partial",
        path.file_name().and_then(|name| name.to_str()).unwrap_or("memo"),
        std::process::id()
    ));
    if std::fs::write(&temporary, bytes).is_err() || std::fs::rename(&temporary, path).is_err() {
        let _ = std::fs::remove_file(&temporary);
        return;
    }
    if let Ok(entries) = std::fs::read_dir(folder) {
        let now = SystemTime::now();
        for entry in entries.flatten() {
            let stale = entry
                .metadata()
                .and_then(|metadata| metadata.modified())
                .ok()
                .and_then(|modified| now.duration_since(modified).ok())
                .is_some_and(|age| age > MEMO_KEEP);
            if stale {
                let _ = std::fs::remove_file(entry.path());
            }
        }
    }
}

/// v2: the digest is `rw-source-file-v3` (pieces hashed in parallel), so a
/// memo of the one-stream digest is never handed back as this one.
const FINGERPRINT_SCHEMA: &str = "rw-source-fingerprint-v2";

/// The memoized whole-file fingerprint of `path` under `name` (the name the
/// digest covers), if a memo was made from this exact file.
pub(crate) fn recall_fingerprint(path: &Path, name: &str) -> Option<[u8; 32]> {
    let stamp = file_stamp(path)?;
    let memo = key_path("source-fingerprints", &format!("{stamp}\u{0}{name}"), ".txt")?;
    let text = std::fs::read_to_string(memo).ok()?;
    let mut lines = text.lines();
    if lines.next()? != FINGERPRINT_SCHEMA || lines.next()? != stamp || lines.next()? != name {
        return None;
    }
    let digest = lines.next()?;
    if digest.len() != 64 {
        return None;
    }
    let mut out = [0u8; 32];
    for (index, byte) in out.iter_mut().enumerate() {
        *byte = u8::from_str_radix(digest.get(2 * index..2 * index + 2)?, 16).ok()?;
    }
    Some(out)
}

/// Memoize `digest` as the whole-file fingerprint of `path` under `name`.
/// `stamp_before` is the file's stamp taken before it was read; nothing is
/// written unless the file still carries it.
pub(crate) fn remember_fingerprint(path: &Path, name: &str, stamp_before: &str, digest: &[u8; 32]) {
    let Some(stamp) = file_stamp(path) else { return };
    if stamp != stamp_before {
        return;
    }
    let Some(memo) = key_path("source-fingerprints", &format!("{stamp}\u{0}{name}"), ".txt") else {
        return;
    };
    let text = format!("{FINGERPRINT_SCHEMA}\n{stamp}\n{name}\n{}\n", hex(digest));
    write_atomically(&memo, text.as_bytes());
}

const SNOWFALL_SCHEMA: &str = "rw-snowfall-window-v1";

/// One saved snowfall integration: through `until`, over the history files
/// whose stamps are `sources`, with its last frozen-precipitation plane and
/// its accumulation.
pub(crate) struct SnowfallMemo {
    pub(crate) until: i64,
    pub(crate) sources: Vec<String>,
    pub(crate) previous: Vec<f32>,
    pub(crate) accumulated: Vec<f32>,
}

fn snowfall_key(window: &str) -> Option<PathBuf> {
    key_path("snowfall-windows", window, ".bin")
}

fn planes_to_bytes(header: &str, previous: &[f32], accumulated: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(header.len() + 16 + 4 * (previous.len() + accumulated.len()));
    bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    for value in previous.iter().chain(accumulated) {
        bytes.extend_from_slice(&value.to_bits().to_le_bytes());
    }
    bytes
}

/// The saved integration of the window named `window`, if any; the caller
/// decides whether its `until` and `sources` still describe the history.
pub(crate) fn recall_snowfall(window: &str, cells: usize) -> Option<SnowfallMemo> {
    let bytes = std::fs::read(snowfall_key(window)?).ok()?;
    let length = u64::from_le_bytes(bytes.get(..8)?.try_into().ok()?) as usize;
    let header: serde_json::Value = serde_json::from_slice(bytes.get(8..8 + length)?).ok()?;
    if header.get("schema")?.as_str()? != SNOWFALL_SCHEMA
        || header.get("window")?.as_str()? != window
        || header.get("cells")?.as_u64()? as usize != cells
    {
        return None;
    }
    let planes = bytes.get(8 + length..)?;
    if planes.len() != 8 * cells {
        return None;
    }
    let values: Vec<f32> = planes
        .chunks_exact(4)
        .map(|chunk| f32::from_bits(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])))
        .collect();
    let (previous, accumulated) = values.split_at(cells);
    Some(SnowfallMemo {
        until: header.get("until")?.as_i64()?,
        sources: header
            .get("sources")?
            .as_array()?
            .iter()
            .map(|value| value.as_str().map(str::to_string))
            .collect::<Option<Vec<_>>>()?,
        previous: previous.to_vec(),
        accumulated: accumulated.to_vec(),
    })
}

/// Save the integration of the window named `window`.
pub(crate) fn remember_snowfall(window: &str, memo: &SnowfallMemo) {
    let Some(path) = snowfall_key(window) else { return };
    let header = serde_json::json!({
        "schema": SNOWFALL_SCHEMA,
        "window": window,
        "cells": memo.previous.len(),
        "until": memo.until,
        "sources": memo.sources,
    })
    .to_string();
    write_atomically(&path, &planes_to_bytes(&header, &memo.previous, &memo.accumulated));
}

const FRAME_PLANE_SCHEMA: &str = "rw-frame-plane-v1";

fn frame_plane_key(stamp: &str, timeidx: usize, kind: &str) -> String {
    format!("{stamp}\u{0}{timeidx}\u{0}{kind}")
}

/// A plane computed from one record of one unchanged history file (the
/// 500 hPa column temperature maximum a Kuchera interval needs), if a memo
/// was made from this exact file.  `kind` names what the plane is.
///
/// WHAT BREAKAGE THIS PREVENTS: a Kuchera window replayed from an earlier
/// memo re-read every earlier frame's 3-D temperature and pressure on one
/// core, because the window memo of the frame before was not written yet
/// while 2-3 live frames drew at once; on an 8x5090 host (1799x1059x50) the
/// frames that replayed from initialization took 251 s at F009 and more
/// than 420 s at F024 against 140-170 s for the rest.  Each frame's own
/// render saves its plane, so a replay reads two-dimensional memos.
pub(crate) fn recall_frame_plane(path: &Path, timeidx: usize, kind: &str, cells: usize) -> Option<Vec<f32>> {
    let stamp = file_stamp(path)?;
    let key = frame_plane_key(&stamp, timeidx, kind);
    let bytes = std::fs::read(key_path("frame-planes", &key, ".bin")?).ok()?;
    let length = u64::from_le_bytes(bytes.get(..8)?.try_into().ok()?) as usize;
    let header: serde_json::Value = serde_json::from_slice(bytes.get(8..8 + length)?).ok()?;
    if header.get("schema")?.as_str()? != FRAME_PLANE_SCHEMA
        || header.get("key")?.as_str()? != key
        || header.get("cells")?.as_u64()? as usize != cells
    {
        return None;
    }
    let plane = bytes.get(8 + length..)?;
    if plane.len() != 4 * cells {
        return None;
    }
    Some(
        plane
            .chunks_exact(4)
            .map(|chunk| f32::from_bits(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])))
            .collect(),
    )
}

/// Memoize `values` as plane `kind` of record `timeidx` of `path`.
/// `stamp_before` is the file's stamp taken before it was read; nothing is
/// written unless the file still carries it.
pub(crate) fn remember_frame_plane(path: &Path, timeidx: usize, kind: &str, stamp_before: &str, values: &[f32]) {
    let Some(stamp) = file_stamp(path) else { return };
    if stamp != stamp_before {
        return;
    }
    let key = frame_plane_key(&stamp, timeidx, kind);
    let Some(memo) = key_path("frame-planes", &key, ".bin") else { return };
    let header = serde_json::json!({
        "schema": FRAME_PLANE_SCHEMA,
        "key": key,
        "cells": values.len(),
    })
    .to_string();
    let mut bytes = Vec::with_capacity(8 + header.len() + 4 * values.len());
    bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    for value in values {
        bytes.extend_from_slice(&value.to_bits().to_le_bytes());
    }
    write_atomically(&memo, &bytes);
}

#[cfg(test)]
mod tests {
    use super::*;

    // file_stamp is None off Unix (no status-change time or inode to key on), so the memo is never
    // written there; the recall contract is a Unix test and Windows checks the memo stays off.
    #[cfg(not(unix))]
    #[test]
    fn no_fingerprint_memo_is_kept_off_unix() {
        let file = std::env::temp_dir().join(format!("rw-memo-off-{}", std::process::id()));
        std::fs::write(&file, b"frame").unwrap();
        let old = SystemTime::now() - Duration::from_secs(60);
        std::fs::File::options().write(true).open(&file).unwrap().set_modified(old).unwrap();
        assert_eq!(file_stamp(&file), None);
        let _ = std::fs::remove_file(&file);
    }

    #[cfg(unix)]
    #[test]
    fn a_fingerprint_memo_is_recalled_only_for_the_unchanged_file() {
        let root = std::env::temp_dir().join(format!("rw-memo-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        // SAFETY: single-threaded test body; nothing else reads the variable.
        unsafe { std::env::set_var("RW_SOURCE_CACHE_DIR", root.join("cache")) };
        let file = root.join("wrfout_d01_2026-10-17_00_00_00");
        std::fs::write(&file, b"frame").unwrap();
        let old = SystemTime::now() - Duration::from_secs(60);
        std::fs::File::options().write(true).open(&file).unwrap().set_modified(old).unwrap();
        let stamp = file_stamp(&file).unwrap();
        let digest = [7u8; 32];
        // A failed memo replacement must leave no partial file and must not
        // prevent the same fingerprint from being saved once the path clears.
        let memo = key_path(
            "source-fingerprints",
            &format!("{stamp}\u{0}wrfout_d01_2026-10-17_00_00_00"),
            ".txt",
        ).unwrap();
        std::fs::create_dir_all(&memo).unwrap();
        remember_fingerprint(&file, "wrfout_d01_2026-10-17_00_00_00", &stamp, &digest);
        assert_eq!(recall_fingerprint(&file, "wrfout_d01_2026-10-17_00_00_00"), None);
        assert!(memo.is_dir());
        let partial = memo.with_file_name(format!(
            ".{}.{}.partial", memo.file_name().unwrap().to_str().unwrap(), std::process::id()
        ));
        assert!(!partial.exists(), "failed memo replacement left a partial file");
        std::fs::remove_dir(&memo).unwrap();
        remember_fingerprint(&file, "wrfout_d01_2026-10-17_00_00_00", &stamp, &digest);
        assert_eq!(recall_fingerprint(&file, "wrfout_d01_2026-10-17_00_00_00"), Some(digest));
        assert_eq!(recall_fingerprint(&file, "other_name"), None);
        // Rewritten in place (even with its old modification time put back):
        // the status-change time moves, so the memo no longer matches.
        std::thread::sleep(Duration::from_millis(20));
        std::fs::write(&file, b"FRAME").unwrap();
        std::fs::File::options().write(true).open(&file).unwrap().set_modified(old).unwrap();
        assert_eq!(recall_fingerprint(&file, "wrfout_d01_2026-10-17_00_00_00"), None);
        let _ = std::fs::remove_dir_all(&root);
    }
}
