//! CU99's bounded-workspace CUDA submissions. No field data crosses the
//! host, no callback enters Python, and no stream completion is inserted.
//! The caller retains every device allocation and the compiled module.
/// Source identity retained in the shipped native library.
#[unsafe(no_mangle)]
pub extern "C" fn gpuwm_cu99_source_rev() -> *const std::os::raw::c_char {
    concat!("GPUWM_BRIDGE_SOURCE_REV=", env!("GPUWM_BRIDGE_SOURCE_REV"), "\0").as_ptr().cast()
}

use std::ffi::c_void;

type Launch = unsafe extern "system" fn(*mut c_void, u32, u32, u32,
    u32, u32, u32, u32, *mut c_void, *mut *mut c_void, *mut *mut c_void) -> i32;

#[repr(C)]
pub struct Arguments {
    pub pointers: [u64; 28],
    pub scalars: [f32; 3],
    // adaptive, stepcu, warm_rain, f_qi, f_qs, f_qr, edrates, nz, ny, nx
    pub options: [i32; 10],
}

#[unsafe(no_mangle)]
pub extern "C" fn gpuwm_cu99_launch_abi_version() -> u32 { 2 }

/// Stateless domain-local scheduling. Restore/nest/member ownership stays
/// with the caller's clock; no hidden timer can survive a restart or move.
#[repr(C)]
#[derive(Default, Debug, PartialEq)]
pub struct StepPlan {
    pub clock_dt: f64,
    pub stepcu: i64,
    pub column_stepcu: i64,
    pub due: i32,
    pub advance: i32,
}

/// WRF module_cumulus_driver.F:832-849 and solve_em.F advance_ppt clock.
/// The override is the shared adaptive clock's event (-1 means fixed dt).
/// No field transfer, CUDA completion, Python callback or mutable singleton.
///
/// # Safety
/// `output` must point to writable, aligned StepPlan storage.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_cu99_plan_step(elapsed: f64, epoch: f64,
    dt: f64, clock_dt: f64, cudt: f64, due_override: i32,
    output: *mut StepPlan) -> i32 {
    if output.is_null() || ![elapsed, epoch, dt, clock_dt, cudt].iter().all(|x| x.is_finite())
        || dt <= 0. || clock_dt < 0. || ![-1, 0, 1].contains(&due_override) {
        return -1;
    }
    let clock_dt = if clock_dt > 0. { clock_dt } else { dt };
    let interval = |step: f64| if cudt <= 0. { 1. } else { (cudt * 60. / step + 0.5).floor().max(1.) };
    let stepcu = interval(dt);
    let column_stepcu = interval(clock_dt);
    let timestep = ((elapsed - epoch) / dt + 0.5).floor() + 1.;
    // Reject rather than saturate integer metadata at the FFI boundary.
    if [stepcu, column_stepcu, timestep].iter().any(|x| x.abs() >= (1_u64 << 53) as f64) {
        return -2;
    }
    let stepcu = stepcu as i64;
    let due = if due_override >= 0 { due_override != 0 } else {
        timestep == 1. || cudt == 0. || stepcu == 1 || timestep as i64 % stepcu == 0
    };
    let remainder = ((elapsed + dt) % clock_dt).abs();
    let advance = remainder.min(clock_dt - remainder) <= 1.0e-6 * clock_dt;
    unsafe { *output = StepPlan { clock_dt, stepcu,
        column_stepcu: column_stepcu as i64, due: due as i32, advance: advance as i32 }; }
    0
}

/// Submit every tile on the caller's current stream. Positive errors are
/// unchanged CUDA driver codes; negative codes reject malformed metadata
/// before the first launch. An error after a launch is not safe to retry.
///
/// # Safety
/// `launch` is cuLaunchKernel, `function` a live CUfunction in the current
/// context, `stream` its live CUstream, and `arguments` a readable Arguments.
/// Every nonzero device address names its declared allocation, retained
/// until the stream completes. Kernel arguments are copied by the driver
/// before each call returns; the stack-held col0 is not used asynchronously.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_cu99_launch_tiles(launch: *const c_void,
    function: *mut c_void, stream: *mut c_void, arguments: *const Arguments,
    tile: i32) -> i32 {
    if launch.is_null() || function.is_null() || arguments.is_null() || tile <= 0 {
        return -1;
    }
    let a = unsafe { &*arguments };
    let [adaptive, stepcu, warm, qi, qs, qr, edrates, nz, ny, nx] = a.options;
    if nz < 1 || ny < 1 || nx < 1 || stepcu < 1
        || [adaptive, warm, qi, qs, qr, edrates].iter().any(|x| ![0, 1].contains(x))
        || a.pointers.contains(&0) || !a.scalars.iter().all(|x| x.is_finite())
        || a.scalars[0] <= 0. || a.scalars[1] <= 0. {
        return -2;
    }
    let Some(columns) = ny.checked_mul(nx) else { return -3; };
    // The kernel guards the complete domain, not an intermediate tile's
    // right edge. A non-block-aligned intermediate tile would therefore
    // advance the next tile's W0AVG twice.
    if tile < columns && tile % 256 != 0 { return -4; }
    let launch: Launch = unsafe { std::mem::transmute(launch) };
    let mut col0 = 0_i32;
    let mut params = [std::ptr::null_mut(); 42];
    for (index, value) in a.pointers.iter().enumerate() {
        params[index] = std::ptr::from_ref(value).cast_mut().cast();
    }
    for (index, value) in a.scalars.iter().enumerate() {
        params[28 + index] = std::ptr::from_ref(value).cast_mut().cast();
    }
    for (index, value) in a.options.iter().enumerate() {
        params[31 + index] = std::ptr::from_ref(value).cast_mut().cast();
    }
    params[41] = std::ptr::from_mut(&mut col0).cast();
    while col0 < columns {
        let span = tile.min(columns - col0) as u32;
        let result = unsafe { launch(function, span.div_ceil(256), 1, 1,
            256, 1, 1, 0, stream, params.as_mut_ptr(), std::ptr::null_mut()) };
        if result != 0 { return result; }
        col0 += span as i32;
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;
    fn plan(t: f64, epoch: f64, dt: f64, clock: f64, cudt: f64, due: i32) -> StepPlan {
        let mut result = StepPlan::default();
        assert_eq!(unsafe { gpuwm_cu99_plan_step(t, epoch, dt, clock, cudt, due, &mut result) }, 0);
        result
    }
    #[test]
    fn native_cadence_restart_nest_and_compatibility_clock() {
        let mut events = vec![];
        for n in 0..720 {
            let p = plan(n as f64 * 15., 0., 15., 0., 5., -1);
            assert_eq!(p.advance, 1);
            assert_eq!((p.stepcu, p.column_stepcu), (20, 20));
            if p.due != 0 { events.push(n + 1); }
        }
        assert_eq!(events, std::iter::once(1).chain((20..=720).step_by(20)).collect::<Vec<_>>());
        assert_eq!(plan(3600., 3600., 5., 0., 5., -1).due, 1);
        assert_eq!(plan(3585., 0., 15., 0., 5., -1).due, 1);
        for n in 0..12 {
            let p = plan(n as f64 * 5., 0., 5., 15., 5., -1);
            assert_eq!(p.advance != 0, n % 3 == 2);
            assert_eq!((p.stepcu, p.column_stepcu), (60, 20));
        }
        assert_eq!(plan(21., 0., 11., 0., 5., 1).due, 1);
        assert_eq!(plan(0., 0., 12., 0., 5., 0).due, 0);
        assert_eq!(plan(0., 0., 45., 0., 5., -1).stepcu, 7);
    }
    // Each callback receives its test-owned recorder as the fake stream.
    struct Calls { rows: Vec<(u32, i32)>, fail_at: usize }
    unsafe extern "system" fn record(_function: *mut c_void, gx: u32, gy: u32, gz: u32,
        bx: u32, by: u32, bz: u32, shared: u32, stream: *mut c_void,
        params: *mut *mut c_void, extra: *mut *mut c_void) -> i32 {
        assert_eq!((gy, gz, bx, by, bz, shared), (1, 1, 256, 1, 1, 0));
        assert!(extra.is_null());
        let calls = unsafe { &mut *stream.cast::<Calls>() };
        let params = unsafe { std::slice::from_raw_parts(params, 42) };
        for (i, parameter) in params[..28].iter().enumerate() {
            assert_eq!(unsafe { *parameter.cast::<u64>() }, (i + 1) as u64);
        }
        assert_eq!(unsafe { *params[28].cast::<f32>() }, 15.);
        assert_eq!(unsafe { *params[29].cast::<f32>() }, 3000.);
        assert_eq!(unsafe { *params[30].cast::<f32>() }, 5.);
        assert_eq!(unsafe { *params[38].cast::<i32>() }, 49);
        let col0 = unsafe { *params[41].cast::<i32>() };
        calls.rows.push((gx, col0));
        if calls.rows.len() == calls.fail_at { 719 } else { 0 }
    }
    fn args() -> Arguments {
        Arguments { pointers: std::array::from_fn(|i| (i + 1) as u64),
            scalars: [15., 3000., 5.], options: [0, 20, 0, 1, 1, 1, 1, 49, 3, 257] }
    }
    fn run(a: &Arguments, calls: &mut Calls, tile: i32) -> i32 {
        unsafe { gpuwm_cu99_launch_tiles(record as *const c_void,
            std::ptr::dangling_mut::<c_void>(), std::ptr::from_mut(calls).cast(), a, tile) }
    }
    #[test]
    fn tile_tail_and_argument_words_are_preserved() {
        let mut calls = Calls { rows: vec![], fail_at: usize::MAX };
        assert_eq!(run(&args(), &mut calls, 512), 0);
        assert_eq!(calls.rows, vec![(2, 0), (2, 512)]);
    }
    #[test]
    fn driver_failure_stops_without_retrying_mutated_columns() {
        let mut calls = Calls { rows: vec![], fail_at: 2 };
        assert_eq!(run(&args(), &mut calls, 256), 719);
        assert_eq!(calls.rows, vec![(1, 0), (1, 256)]);
    }
    #[test]
    fn bad_metadata_is_refused_before_any_launch() {
        let mut calls = Calls { rows: vec![], fail_at: usize::MAX };
        let mut a = args();
        assert_eq!(run(&a, &mut calls, 0), -1);
        assert_eq!(run(&a, &mut calls, 300), -4);
        a.options[9] = i32::MAX;
        assert_eq!(run(&a, &mut calls, 256), -3);
        a.options[9] = 257;
        a.scalars[0] = f32::NAN;
        assert_eq!(run(&a, &mut calls, 256), -2);
        assert!(calls.rows.is_empty());
    }
}


/// The revision stamp is exported so it survives shared-library linking.
#[unsafe(no_mangle)]
pub extern "C" fn gpuwm_cu99_launch_source_rev() -> *const std::os::raw::c_char {
    static SOURCE_REV_STAMP: &str = concat!(
        "GPUWM_BRIDGE_SOURCE_REV=", env!("GPUWM_BRIDGE_SOURCE_REV"), "\0");
    SOURCE_REV_STAMP.as_ptr().cast()
}
