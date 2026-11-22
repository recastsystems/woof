//! The C ABI seam `gpuwm/static/rust_bridge.py` loads through ctypes.
//!
//! Same discipline as `netcdf-writer/src/capi.rs`: cdylib + ctypes (no
//! pyo3 -- one loading discipline across every gpuwm bridge), positional
//! C signatures guarded by an ABI version probe, a thread-local
//! last-error string, and a contract-marker symbol
//! (`gpuwm_static_sampling_portable_v1`, listed in `gpuwm/bridges.py`
//! `BRIDGE_ABI_MARKERS`) naming the capability that distinguishes this
//! contract.
//!
//! Handle model: opaque `u64` handles into process-global registries --
//! grids ([`grid`]) and field sets ([`build`]) -- because grids are
//! built once and queried many times, and field sets are large enough
//! that the Python side must copy plane-by-plane into preallocated
//! numpy buffers rather than round-trip JSON.  Scalars/config cross as
//! UTF-8 JSON; array data crosses as raw little-endian f64, exactly
//! `numpy.tobytes()` of a C-contiguous array.
//!
//! Entry-point ownership (files, not fences): [`grid`] is lane 1,
//! [`build`] is lane 2, [`highres`] is lane 3.  This module owns only
//! the shared plumbing below.

pub mod build;
pub mod grid;
pub mod gwdo_wps281;
pub mod highres;
pub mod hydrofabric;
pub mod polar_halo;
pub mod scalar;
pub mod sfire;
pub mod soil;
pub mod wrf_history;
pub mod wetland;

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use crate::projection::ProjectedGrid;
use crate::types::FieldSet;

/// The seam's ABI version.  Bump when a signature changes shape, never
/// for a rebuild.
pub const STATIC_ABI_VERSION: u32 = 1;

pub(crate) const OK: i32 = 0;
pub(crate) const ERR: i32 = -1;

thread_local! {
    static LAST_ERROR: RefCell<String> = const { RefCell::new(String::new()) };
}

pub(crate) fn set_error(message: impl Into<String>) -> i32 {
    LAST_ERROR.with(|slot| *slot.borrow_mut() = message.into());
    ERR
}

pub(crate) fn clear_error() {
    LAST_ERROR.with(|slot| slot.borrow_mut().clear());
}

/// Turn a panic below this seam into the ABI's own refusal.
///
/// An unwind that reaches an `extern "C"` boundary aborts the process on the
/// edition this workspace pins, so a panic in a decoder would take the host
/// Python interpreter with it instead of returning the negative code and
/// last-error string this ABI documents.  Every entry point runs its body
/// through here, the same discipline `tools/grib1_bridge/src/lib.rs` applies
/// to its own exports.
pub(crate) fn guard<T>(on_panic: T, body: impl FnOnce() -> T) -> T {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(body)) {
        Ok(value) => value,
        Err(payload) => {
            let detail = payload
                .downcast_ref::<&str>()
                .map(|text| (*text).to_string())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "unknown panic payload".to_string());
            // `try_borrow_mut`, not `borrow_mut`: the panic may have come from
            // inside the last-error accessor itself, and a second panic here
            // would abort exactly what this guard exists to prevent.
            LAST_ERROR.with(|slot| {
                if let Ok(mut message) = slot.try_borrow_mut() {
                    *message = format!("panic in the static-fields seam: {detail}");
                }
            });
            on_panic
        }
    }
}

fn next_handle(counter: &AtomicU64) -> u64 {
    counter.fetch_add(1, Ordering::Relaxed) + 1
}

static GRID_COUNTER: AtomicU64 = AtomicU64::new(0);
static GRIDS: Mutex<BTreeMap<u64, ProjectedGrid>> = Mutex::new(BTreeMap::new());

static FIELDSET_COUNTER: AtomicU64 = AtomicU64::new(0);
static FIELDSETS: Mutex<BTreeMap<u64, FieldSet>> = Mutex::new(BTreeMap::new());

pub(crate) fn register_grid(grid: ProjectedGrid) -> u64 {
    let handle = next_handle(&GRID_COUNTER);
    GRIDS.lock().expect("grid registry poisoned").insert(handle, grid);
    handle
}

pub(crate) fn with_grid<T>(
    handle: u64,
    f: impl FnOnce(&ProjectedGrid) -> T,
) -> Option<T> {
    GRIDS.lock().expect("grid registry poisoned").get(&handle).map(f)
}

pub(crate) fn drop_grid(handle: u64) -> bool {
    GRIDS.lock().expect("grid registry poisoned").remove(&handle).is_some()
}

pub(crate) fn register_fieldset(fields: FieldSet) -> u64 {
    let handle = next_handle(&FIELDSET_COUNTER);
    FIELDSETS
        .lock()
        .expect("fieldset registry poisoned")
        .insert(handle, fields);
    handle
}

pub(crate) fn with_fieldset<T>(
    handle: u64,
    f: impl FnOnce(&FieldSet) -> T,
) -> Option<T> {
    FIELDSETS
        .lock()
        .expect("fieldset registry poisoned")
        .get(&handle)
        .map(f)
}

pub(crate) fn drop_fieldset(handle: u64) -> bool {
    FIELDSETS
        .lock()
        .expect("fieldset registry poisoned")
        .remove(&handle)
        .is_some()
}

/// Borrow both inputs under one registry lock without duplicating their arrays.
pub(crate) fn with_fieldsets<T>(
    first: u64,
    second: u64,
    f: impl FnOnce(&FieldSet, &FieldSet) -> T,
) -> Result<T, u64> {
    let registry = FIELDSETS.lock().expect("fieldset registry poisoned");
    let first_set = registry.get(&first).ok_or(first)?;
    let second_set = registry.get(&second).ok_or(second)?;
    Ok(f(first_set, second_set))
}

/// # Safety
/// `ptr` must point to `len` readable bytes, or be null when `len` is 0.
pub(crate) unsafe fn bytes<'a>(ptr: *const u8, len: usize) -> Option<&'a [u8]> {
    if len == 0 {
        return Some(&[]);
    }
    if ptr.is_null() {
        return None;
    }
    Some(unsafe { std::slice::from_raw_parts(ptr, len) })
}

/// # Safety
/// `ptr`/`len` as for [`bytes`]; the bytes must be UTF-8.
pub(crate) unsafe fn utf8<'a>(ptr: *const u8, len: usize) -> Option<&'a str> {
    let raw = unsafe { bytes(ptr, len) }?;
    std::str::from_utf8(raw).ok()
}

#[unsafe(no_mangle)]
pub extern "C" fn gpuwm_static_abi_version() -> u32 {
    guard(0, || {
        STATIC_ABI_VERSION
    })
}

/// Fixed float32 and float64 sampling arithmetic for prepared moving nests.
#[unsafe(no_mangle)]
pub extern "C" fn gpuwm_static_sampling_portable_v1() -> u32 { 1 }

/// The source-revision stamp, same contract as
/// `gpuwm_ncwrite_source_rev` (see `netcdf-writer`): read out of the
/// binary as bytes by the release cut, never executed.
#[unsafe(no_mangle)]
pub extern "C" fn gpuwm_static_source_rev() -> *const std::os::raw::c_char {
    guard(std::ptr::null(), || {
        static SOURCE_REV_STAMP: &str = concat!(
            "GPUWM_BRIDGE_SOURCE_REV=",
            env!("GPUWM_BRIDGE_SOURCE_REV"),
            "\0"
        );
        SOURCE_REV_STAMP.as_ptr().cast()
    })
}

/// Copy the thread-local last error into `buf` (UTF-8, no NUL);
/// returns the full message length so a short buffer is detectable.
///
/// # Safety
/// `buf` must point to `cap` writable bytes, or be null with `cap` 0.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_static_last_error(
    buf: *mut u8,
    cap: usize,
) -> usize {
    guard(0, || {
        LAST_ERROR.with(|slot| {
            let message = slot.borrow();
            let raw = message.as_bytes();
            if !buf.is_null() && cap > 0 {
                let n = raw.len().min(cap);
                unsafe {
                    std::ptr::copy_nonoverlapping(raw.as_ptr(), buf, n);
                }
            }
            raw.len()
        })
    })
}

#[cfg(test)]
mod tests {
    use super::grid::gpuwm_static_grid_free;
    use super::*;

    const CHILD_ENV: &str = "GPUWM_STATIC_SEAM_PANIC_CHILD";
    const MARKER: &str = "SEAM-LAST-ERROR:";

    /// Negative control for the FFI boundary: feed the seam a panic and
    /// require the ABI's own refusal instead of an abort.
    ///
    /// The child re-runs this one test with `CHILD_ENV` set, poisons the grid
    /// registry the way a panic in any `rlib` consumer does, and then calls an
    /// `extern "C"` entry point whose body hits `.expect("grid registry
    /// poisoned")`.  Without the guard that unwind reaches the `extern "C"`
    /// frame and the child dies on SIGABRT -- which is what a panicking
    /// decoder did to the host Python interpreter, with no traceback and no
    /// `gpuwm_static_last_error` for the operator to read.
    #[test]
    fn a_panic_below_the_seam_is_a_refusal_not_a_process_abort() {
        if std::env::var_os(CHILD_ENV).is_some() {
            let poisoned = std::panic::catch_unwind(|| {
                let _held = GRIDS.lock().unwrap();
                panic!("a consumer panicked while holding the registry");
            });
            assert!(poisoned.is_err(), "the registry must end up poisoned");

            gpuwm_static_grid_free(1);

            let mut buffer = [0u8; 256];
            let length = unsafe {
                gpuwm_static_last_error(buffer.as_mut_ptr(), buffer.len())
            };
            let message =
                std::str::from_utf8(&buffer[..length.min(buffer.len())]).unwrap();
            println!("{MARKER}{message}");
            return;
        }

        let child = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "capi::tests::a_panic_below_the_seam_is_a_refusal_not_a_process_abort",
                "--nocapture",
                "--test-threads",
                "1",
            ])
            .env(CHILD_ENV, "1")
            .output()
            .expect("re-running this test binary");
        let stdout = String::from_utf8_lossy(&child.stdout);
        assert!(
            child.status.success(),
            "the seam aborted the process instead of refusing: {:?}\n{stdout}",
            child.status
        );
        assert!(
            stdout.contains(&format!(
                "{MARKER}panic in the static-fields seam: grid registry poisoned"
            )),
            "the refusal must reach the last-error slot: {stdout}"
        );
    }

    #[test]
    fn the_guard_is_transparent_when_nothing_panics() {
        assert_eq!(guard(ERR, || OK), OK);
    }
}
