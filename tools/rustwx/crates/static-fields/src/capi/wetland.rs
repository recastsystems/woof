//! C seam of the Noah-MP wetland static maps (crate::wetland).
use std::ffi::CStr;
use std::os::raw::c_char;
use std::path::PathBuf;

use super::{clear_error, guard, set_error, ERR, OK};

/// Build FSATMX (fraction) and WCAP (mm) for `n` mass points.
///
/// `dem_paths` is a newline-separated list of GeoTIFF paths.  `coverage`
/// receives two f64: the GLWD and DEM cell coverage fractions.
///
/// # Safety
/// `lat`/`lon` point to `n` f64; `fsatmx`/`wcap` to `n` writable f32;
/// `coverage` to 2 writable f64; both strings are NUL-terminated UTF-8.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_static_wetland_v1(
    lat: *const f64, lon: *const f64, n: usize, dx_m: f64,
    glwd_path: *const c_char, dem_paths: *const c_char,
    fsatmx: *mut f32, wcap: *mut f32, coverage: *mut f64,
) -> i32 {
    guard(ERR, || {
        clear_error();
        if lat.is_null() || lon.is_null() || fsatmx.is_null() || wcap.is_null()
            || coverage.is_null() || glwd_path.is_null() || dem_paths.is_null()
            || n > isize::MAX as usize / 8 || !(dx_m > 0.0)
        {
            return set_error("wetland static: invalid pointer, length or dx");
        }
        let (glwd, dems) = unsafe {
            match (CStr::from_ptr(glwd_path).to_str(), CStr::from_ptr(dem_paths).to_str()) {
                (Ok(g), Ok(d)) => (PathBuf::from(g), d.lines().filter(|l| !l.is_empty())
                                   .map(PathBuf::from).collect::<Vec<_>>()),
                _ => return set_error("wetland static: paths are not UTF-8"),
            }
        };
        let lat = unsafe { std::slice::from_raw_parts(lat, n) };
        let lon = unsafe { std::slice::from_raw_parts(lon, n) };
        match crate::wetland::build(lat, lon, dx_m, &glwd, &dems) {
            Ok(out) => {
                unsafe {
                    std::slice::from_raw_parts_mut(fsatmx, n).copy_from_slice(&out.fsatmx);
                    std::slice::from_raw_parts_mut(wcap, n).copy_from_slice(&out.wcap_mm);
                    *coverage = out.glwd_coverage;
                    *coverage.add(1) = out.dem_coverage;
                }
                OK
            }
            Err(e) => set_error(format!("wetland static: {e}")),
        }
    })
}
