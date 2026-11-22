//! WRF v4.8.0's gwd_opt = 1 orographic statistics (ELVMAX, VAR, CON, OA1-4, OL1-4), WPS PR #281's recipe
//! (crate::gwdo_wps281), for the static builder's ctypes seam (gpuwm/static/rust_bridge.py).

use super::{clear_error, guard, set_error, ERR, OK};

/// Compute the eleven fields for `count` mass points.  `output` receives 11 * `count` binary32 words in the
/// order ELVMAX, VAR, CON, OA1, OA2, OA3, OA4, OL1, OL2, OL3, OL4, each a `count` block.
///
/// # Safety
/// `topo_dir` is a NUL-terminated UTF-8 path; `lat`/`lon` hold `count` words; `output` holds 11 * `count`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_static_gwdo_wps281(
    topo_dir: *const std::os::raw::c_char,
    lat: *const f32,
    lon: *const f32,
    count: usize,
    dx: f32,
    dy: f32,
    output: *mut f32,
) -> i32 {
    guard(ERR, || {
        clear_error();
        if topo_dir.is_null() || lat.is_null() || lon.is_null() || output.is_null() || count == 0
            || count > isize::MAX as usize / (11 * std::mem::size_of::<f32>())
        {
            return set_error("gwdo_wps281: invalid buffers");
        }
        let path = match unsafe { std::ffi::CStr::from_ptr(topo_dir) }.to_str() {
            Ok(p) => std::path::PathBuf::from(p),
            Err(_) => return set_error("gwdo_wps281: topo path is not UTF-8"),
        };
        let lat = unsafe { std::slice::from_raw_parts(lat, count) };
        let lon = unsafe { std::slice::from_raw_parts(lon, count) };
        let out = unsafe { std::slice::from_raw_parts_mut(output, 11 * count) };
        let topo = match crate::geog::GeogDataset::open(&path, None) {
            Ok(t) => t,
            Err(e) => return set_error(&format!("gwdo_wps281: {e}")),
        };
        let f = match crate::gwdo_wps281::compute(&topo, lat, lon, dx, dy) {
            Ok(f) => f,
            Err(e) => return set_error(&format!("gwdo_wps281: {e}")),
        };
        let blocks: [&[f32]; 11] = [&f.elvmax, &f.var, &f.con, &f.oa[0], &f.oa[1], &f.oa[2], &f.oa[3],
                                    &f.ol[0], &f.ol[1], &f.ol[2], &f.ol[3]];
        for (k, block) in blocks.iter().enumerate() {
            out[k * count..(k + 1) * count].copy_from_slice(block);
        }
        OK
    })
}
