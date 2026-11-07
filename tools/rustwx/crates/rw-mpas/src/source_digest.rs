//! Reuse an unchanged source's ordinary SHA-256 across converter processes.
//! Whole-view and tile conversion otherwise each hash the entire initial
//! state, including fields none of those views reads. The digest remains
//! the ordinary whole-file SHA-256, including in converted-file metadata.
use crate::MpasResult;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

fn root() -> Option<PathBuf> {
    if let Some(path) = std::env::var_os("RW_SOURCE_CACHE_DIR") {
        return (path != "off").then(|| PathBuf::from(path).join("mpas-file-digests-v1"));
    }
    std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|p| PathBuf::from(p).join(".cache")))
        .map(|p| p.join("rw-mpas/file-digests-v1"))
}

// Same revision rule as rw-wrfbatch's source memo. In-place writes change
// ctime even when an application restores mtime. Unsettled files and hosts
// without those revision stamps always take the full hashing path.
fn stamp(path: &Path) -> Option<String> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        let path = path.canonicalize().ok()?;
        let m = path.metadata().ok()?;
        if !m.is_file()
            || SystemTime::now().duration_since(m.modified().ok()?).ok()?
                < Duration::from_millis(200)
        {
            return None;
        }
        Some(format!(
            "{}\0{} {} {} {} {} {} {}",
            path.display(),
            m.dev(),
            m.ino(),
            m.len(),
            m.mtime(),
            m.mtime_nsec(),
            m.ctime(),
            m.ctime_nsec()
        ))
    }
    #[cfg(not(unix))]
    {
        let _ = path;
        None
    }
}

pub(super) fn digest(path: &Path) -> MpasResult<String> {
    digest_in(path, root())
}

fn digest_in(path: &Path, cache_root: Option<PathBuf>) -> MpasResult<String> {
    let Some(before) = stamp(path) else {
        return crate::sha256_file_uncached(path);
    };
    let Some(root) = cache_root else {
        return crate::sha256_file_uncached(path);
    };
    let key = crate::weights::hex(&Sha256::digest(before.as_bytes()));
    let dir = root.join(key);
    if std::fs::create_dir_all(&dir).is_err() {
        return crate::sha256_file_uncached(path);
    }
    let Ok(_lock) = rw_store::RunLock::acquire(&dir, Duration::from_secs(300)) else {
        return crate::sha256_file_uncached(path);
    };
    let memo = dir.join("sha256.json");
    if let Ok(bytes) = std::fs::read(&memo) {
        if let Ok((revision, digest)) = serde_json::from_slice::<(String, String)>(&bytes) {
            if revision == before
                && digest.len() == 64
                && digest.bytes().all(|b| b.is_ascii_hexdigit())
                && stamp(path).as_ref() == Some(&before)
            {
                if std::env::var_os("RUSTWX_IMPORT_TIMINGS").is_some() {
                    eprintln!("SOURCE_DIGEST reused {}", path.display());
                }
                return Ok(digest);
            }
        }
    }
    let digest = crate::sha256_file_uncached(path)?;
    if stamp(path).as_ref() == Some(&before) {
        if let Ok(bytes) = serde_json::to_vec(&(&before, &digest)) {
            let temporary = dir.join(format!("{}.tmp", std::process::id()));
            if std::fs::write(&temporary, bytes).is_ok() {
                let _ = std::fs::rename(&temporary, memo);
            }
            let _ = std::fs::remove_file(temporary);
        }
    }
    Ok(digest)
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    #[test]
    fn unchanged_digest_and_restored_mtime_invalidation() {
        let dir = std::env::temp_dir().join(format!("mpas-digest-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("input");
        let cache = dir.join("cache");
        std::fs::write(&path, b"before").unwrap();
        std::thread::sleep(Duration::from_millis(220));
        let modified = path.metadata().unwrap().modified().unwrap();
        let old_stamp = stamp(&path).unwrap();
        let first = digest_in(&path, Some(cache.clone())).unwrap();
        assert_eq!(first, crate::sha256_file_uncached(&path).unwrap());
        assert_eq!(first, digest_in(&path, Some(cache.clone())).unwrap());
        std::fs::write(&path, b"after!").unwrap();
        std::fs::File::options()
            .write(true)
            .open(&path)
            .unwrap()
            .set_times(std::fs::FileTimes::new().set_modified(modified))
            .unwrap();
        assert_ne!(old_stamp, stamp(&path).unwrap());
        let second = digest_in(&path, Some(cache)).unwrap();
        assert_ne!(first, second);
        assert_eq!(second, crate::sha256_file_uncached(&path).unwrap());
        std::fs::remove_dir_all(dir).unwrap();
    }
}
