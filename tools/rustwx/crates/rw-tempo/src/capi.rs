//! The C ABI seam `gpuwm/core/tempo_rust.py` loads through ctypes.
//!
//! Same discipline as the other gpuwm bridges (static-fields,
//! netcdf-writer): cdylib + ctypes, an ABI version probe, a contract-marker
//! symbol (`gpuwm_tempo_init_v1`), a thread-local last-error string, and
//! every entry point guarded so a panic returns the refusal code instead
//! of unwinding into Python.  Array data crosses as raw f64.

use std::cell::RefCell;
use std::path::Path;

use crate::{tempo_init, InitOptions, FIELDS};

pub const TEMPO_ABI_VERSION: u32 = 1;
const OK: i32 = 0;
const ERR: i32 = -1;

thread_local! {
    static LAST_ERROR: RefCell<String> = const { RefCell::new(String::new()) };
}

fn set_error(message: impl Into<String>) -> i32 {
    LAST_ERROR.with(|slot| *slot.borrow_mut() = message.into());
    ERR
}

fn guard<T>(on_panic: T, body: impl FnOnce() -> T) -> T {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(body)) {
        Ok(value) => value,
        Err(payload) => {
            let detail = payload
                .downcast_ref::<&str>()
                .map(|t| (*t).to_string())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "unknown panic payload".to_string());
            LAST_ERROR.with(|slot| {
                if let Ok(mut m) = slot.try_borrow_mut() {
                    *m = format!("panic in the TEMPO init seam: {detail}");
                }
            });
            on_panic
        }
    }
}

unsafe fn utf8<'a>(ptr: *const u8, len: usize) -> Option<&'a str> {
    if ptr.is_null() {
        return None;
    }
    // SAFETY: the caller passes a buffer of `len` bytes.
    std::str::from_utf8(unsafe { std::slice::from_raw_parts(ptr, len) }).ok()
}

#[unsafe(no_mangle)]
pub extern "C" fn gpuwm_tempo_abi_version() -> u32 {
    TEMPO_ABI_VERSION
}

/// The source revision this library was built from, as a NUL-terminated
/// `GPUWM_BRIDGE_SOURCE_REV=<40-hex commit>` literal (`build.rs` injects it).
/// The release cut reads the literal out of the file's bytes to prove a
/// staged library matches the commit being released; exporting it keeps it
/// in the cdylib under LTO.  2.9.0: this library joined the bridge bundle,
/// because no wheel carried it and TEMPO could not initialize from a pip
/// install.
#[unsafe(no_mangle)]
pub extern "C" fn gpuwm_tempo_source_rev() -> *const std::os::raw::c_char {
    static SOURCE_REV_STAMP: &str = concat!(
        "GPUWM_BRIDGE_SOURCE_REV=",
        env!("GPUWM_BRIDGE_SOURCE_REV"),
        "\0"
    );
    SOURCE_REV_STAMP.as_ptr().cast()
}

/// Contract marker: TEMPO 9adf9ef tempo_init, blob layout v1.
#[unsafe(no_mangle)]
pub extern "C" fn gpuwm_tempo_init_v1() -> u32 {
    1
}

#[unsafe(no_mangle)]
pub extern "C" fn gpuwm_tempo_field_count() -> usize {
    FIELDS.len()
}

/// Length of field `i` (0 when out of range).
#[unsafe(no_mangle)]
pub extern "C" fn gpuwm_tempo_field_len(i: usize) -> u64 {
    FIELDS.get(i).map(|(_, n)| *n as u64).unwrap_or(0)
}

/// Copies field `i`'s name into `buf` (no terminator); returns its length.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_tempo_field_name(i: usize, buf: *mut u8, cap: usize) -> usize {
    guard(0, || {
        let Some((name, _)) = FIELDS.get(i) else { return 0 };
        let bytes = name.as_bytes();
        if !buf.is_null() {
            let n = bytes.len().min(cap);
            // SAFETY: the caller owns `cap` writable bytes at `buf`.
            unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), buf, n) };
        }
        bytes.len()
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn gpuwm_tempo_blob_len() -> u64 {
    FIELDS.iter().map(|(_, n)| *n as u64).sum()
}

/// Build the blob from the four TEMPO table files in `table_dir` into
/// `out` (`out_len` doubles, which must equal gpuwm_tempo_blob_len()).
/// Returns 0, or -1 with gpuwm_tempo_last_error set.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_tempo_build_blob(
    table_dir: *const u8,
    table_dir_len: usize,
    hailaware: i32,
    refl10cm_from_melting: i32,
    max_hail_diameter: i32,
    out: *mut f64,
    out_len: u64,
) -> i32 {
    guard(ERR, || {
        // SAFETY: caller contract above.
        let Some(dir) = (unsafe { utf8(table_dir, table_dir_len) }) else {
            return set_error("table_dir is not valid UTF-8");
        };
        if out.is_null() || out_len != gpuwm_tempo_blob_len() {
            return set_error(format!(
                "output buffer holds {out_len} doubles, the TEMPO blob needs {}",
                gpuwm_tempo_blob_len()
            ));
        }
        let opt = InitOptions {
            hailaware: hailaware != 0,
            refl10cm_from_melting: refl10cm_from_melting != 0,
            max_hail_diameter: max_hail_diameter != 0,
        };
        match tempo_init(Path::new(dir), opt) {
            Ok(blob) => {
                // SAFETY: `out` holds out_len == blob length doubles.
                unsafe { std::ptr::copy_nonoverlapping(blob.data.as_ptr(), out, blob.data.len()) };
                OK
            }
            Err(e) => set_error(e),
        }
    })
}

/// Build the blob and write it as the named-record file
/// tools/tempo_ref/tempo_ref_driver.F90 writes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_tempo_write_records(
    table_dir: *const u8,
    table_dir_len: usize,
    hailaware: i32,
    refl10cm_from_melting: i32,
    max_hail_diameter: i32,
    out_path: *const u8,
    out_path_len: usize,
) -> i32 {
    guard(ERR, || {
        // SAFETY: caller passes valid UTF-8 buffers of the given lengths.
        let (Some(dir), Some(path)) =
            (unsafe { utf8(table_dir, table_dir_len) }, unsafe { utf8(out_path, out_path_len) })
        else {
            return set_error("table_dir or out_path is not valid UTF-8");
        };
        let opt = InitOptions {
            hailaware: hailaware != 0,
            refl10cm_from_melting: refl10cm_from_melting != 0,
            max_hail_diameter: max_hail_diameter != 0,
        };
        match tempo_init(Path::new(dir), opt) {
            Ok(blob) => match std::fs::write(path, blob.to_records()) {
                Ok(()) => OK,
                Err(e) => set_error(format!("writing {path}: {e}")),
            },
            Err(e) => set_error(e),
        }
    })
}

/// Copies the last error into `buf`; returns its full length.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_tempo_last_error(buf: *mut u8, cap: usize) -> usize {
    LAST_ERROR.with(|slot| {
        let text = slot.borrow();
        let bytes = text.as_bytes();
        if !buf.is_null() {
            let n = bytes.len().min(cap);
            // SAFETY: the caller owns `cap` writable bytes at `buf`.
            unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), buf, n) };
        }
        bytes.len()
    })
}

/// Contract marker for WRF big-endian f32 sequential table records.
#[unsafe(no_mangle)]
pub extern "C" fn gpuwm_wrf_table_f32_v1() -> u32 { 1 }

/// Read, hash, validate and decode one pinned WRF table during setup.
/// Safety: path addresses path_len UTF-8 bytes, sha256 addresses 64 ASCII
/// bytes, and out addresses out_len writable, aligned floats, disjoint from
/// both input buffers. No pointers are retained after this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_wrf_table_read_f32(
    path: *const u8, path_len: usize, sha256: *const u8,
    records: usize, values: usize, out: *mut f32, out_len: usize,
) -> i32 {
    guard(ERR, || {
        let (Some(path), Some(hash)) =
            (unsafe { utf8(path, path_len) }, unsafe { utf8(sha256, 64) })
        else { return set_error("path or hash is not UTF-8"); };
        if out.is_null() || out_len == 0 || out_len > isize::MAX as usize / 4 {
            return set_error("invalid table output buffer");
        }
        let output = unsafe { std::slice::from_raw_parts_mut(out, out_len) };
        match crate::wrf_tables::read_f32(Path::new(path), hash, records, values, output) {
            Ok(()) => OK,
            Err(e) => set_error(e),
        }
    })
}
