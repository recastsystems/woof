//! DFI setup and compiled-pass submission for the forecast driver.
//!
//! WRF v4.6.1 share/dfi.F, dfcoef through dolph, and
//! share/set_timekeeping.F. Source hashes and the public-domain notice live
//! in tools/dfi_wrf461_oracle/SOURCE.json and NOTICE.txt.
//! Opt-in filter 8 retains the identical 4.7.1 formulas with bounded row
//! construction; tools/dfi_wrf471_oracle documents its declared divergence.
//! No dependency on Python, an array package, or a CUDA link library.

/// Source identity retained in the shipped native library.
#[no_mangle]
pub extern "C" fn arwen_dfi_source_rev() -> *const std::os::raw::c_char {
    concat!("GPUWM_BRIDGE_SOURCE_REV=", env!("GPUWM_BRIDGE_SOURCE_REV"), "\0").as_ptr().cast()
}

use std::ffi::c_void;

// glibc libm on Linux (the WRF-matching routines); the C runtime provides
// them on Windows, where MSVC has no m.lib (LNK1181 cannot open m.lib).
#[cfg_attr(unix, link(name = "m"))]
extern "C" {
    fn sinf(x: f32) -> f32;
    fn cosf(x: f32) -> f32;
    fn powf(x: f32, y: f32) -> f32;
    fn tan(x: f64) -> f64;
}

fn sin(x: f32) -> f32 {
    unsafe { sinf(x) }
}
fn cos(x: f32) -> f32 {
    unsafe { cosf(x) }
}
fn pow(x: f32, y: f32) -> f32 {
    unsafe { powf(x, y) }
}

/// Times are integer seconds relative to the unfiltered analysis.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pass {
    pub start: i64,
    pub stop: i64,
    pub backward: bool,
    pub filter: bool,
    pub reset: bool,
}

#[derive(Debug, PartialEq, Eq)]
pub struct Lifecycle {
    pub passes: Vec<Pass>,
    pub forecast_start: i64,
}

/// C representation of the native lifecycle. This is setup state, not a
/// Python-owned counter in the model loop.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct NativePass {
    pub start: i64,
    pub stop: i64,
    pub backward: u32,
    pub filter: u32,
    pub reset: u32,
    pub reserved: u32,
}

/// The caller supplies capacity for two passes. Restart returns an empty
/// lifecycle, exactly as module_check_a_mundo disables DFI on restart.
///
/// # Safety
/// Non-null output pointers must refer to writable objects of their stated
/// size. They may not overlap each other or the pass array.
#[no_mangle]
pub unsafe extern "C" fn aw_dfi_lifecycle(
    opt: i32,
    backward_stop: i64,
    forward_stop: i64,
    restart: u32,
    passes: *mut NativePass,
    capacity: usize,
    count: *mut usize,
    forecast_start: *mut i64,
) -> i32 {
    if count.is_null() || forecast_start.is_null() {
        return -1;
    }
    let plan = match lifecycle(opt, backward_stop, forward_stop, restart != 0) {
        Ok(plan) => plan,
        Err(_) => return -2,
    };
    if capacity < plan.passes.len() || (!plan.passes.is_empty() && passes.is_null()) {
        return -3;
    }
    for (i, p) in plan.passes.iter().enumerate() {
        passes.add(i).write(NativePass {
            start: p.start,
            stop: p.stop,
            backward: p.backward as u32,
            filter: p.filter as u32,
            reset: p.reset as u32,
            reserved: 0,
        });
    }
    count.write(plan.passes.len());
    forecast_start.write(plan.forecast_start);
    0
}

/// WRF disables DFI on restart. DFL's forecast starts at the forward
/// window midpoint, not at the original analysis. TDFI's filtered backward
/// result is placed at its midpoint before the forward pass.
pub fn lifecycle(
    opt: i32,
    backward_stop: i64,
    forward_stop: i64,
    restart: bool,
) -> Result<Lifecycle, &'static str> {
    if restart || opt == 0 {
        return Ok(Lifecycle {
            passes: vec![],
            forecast_start: 0,
        });
    }
    if !(1..=3).contains(&opt) {
        return Err("dfi_opt must be 0, 1, 2 or 3");
    }
    if forward_stop <= 0 {
        return Err("DFI forward stop must follow the analysis");
    }
    if opt != 1 && backward_stop >= 0 {
        return Err("DFI backward stop must precede the analysis");
    }
    let forward = |start| Pass {
        start,
        stop: forward_stop,
        backward: false,
        filter: true,
        reset: true,
    };
    let passes = match opt {
        1 => vec![forward(0)],
        2 => vec![
            Pass {
                start: 0,
                stop: backward_stop,
                backward: true,
                // Native still accumulates backward DDFI samples. Although
                // it skips reset, clear() retains radius accumulators for
                // the forward pass, so omitting this work is not equivalent.
                filter: true,
                reset: false,
            },
            forward(backward_stop),
        ],
        3 => vec![
            Pass {
                start: 0,
                stop: backward_stop,
                backward: true,
                filter: true,
                reset: true,
            },
            forward(backward_stop - backward_stop / 2),
        ],
        _ => unreachable!(),
    };
    Ok(Lifecycle {
        passes,
        forecast_start: if opt == 1 { forward_stop / 2 } else { 0 },
    })
}

// WRF evaluates (X/3.75)**2 in REAL, then widens it for the REAL*8
// polynomial. Computing the square in binary64 changes filter 4's words.
fn bessi0(x: f32) -> f32 {
    if x.abs() < 3.75 {
        let t = x / 3.75;
        let y = (t * t) as f64;
        (1.0 + y
            * (3.5156230
                + y * (3.0899424
                    + y * (1.2067492 + y * (0.2659732 + y * (0.0360768 + y * 0.0045813))))))
            as f32
    } else {
        // KAISER fixes ALPHA=1, so the other WRF BESSI0 branch is unreachable.
        unreachable!("DFI Kaiser argument exceeds ALPHA=1")
    }
}

fn cheby_last(n: usize, x: f32) -> f32 {
    if n == 0 {
        return 1.0;
    }
    let (mut a, mut b) = (1.0, x);
    for _ in 2..=n {
        let c = (2.0 * x) * b - a;
        a = b;
        b = c;
    }
    b
}

/// WRF 4.7.1 RHOFIL/RHOCOF, order 2, initial-condition type 2.
/// Keep the caller's half-row contract and optfil_driver's reflection.
/// The native defect is writing NSTEP+1 (and clearing 101) words into
/// NOSIZE+1 words. Compute the row in sized scratch, then copy its prefix.
/// REAL products round before the DOUBLE PRECISION sums, as in Fortran.
/// Source, the bounded native patch and the declared divergence are in
/// tools/dfi_wrf471_oracle/. There is no bitwise claim for native's UB.
fn rho_half(steps: usize, dt: f32, cutoff: f32) -> Result<Vec<f32>, &'static str> {
    if cutoff <= 2.0 * dt.abs() {
        return Err("RHO cutoff must exceed two timesteps (Nyquist period)");
    }
    let last = steps.checked_mul(2).ok_or("RHO extent overflow")?;
    let size = last.checked_add(1).ok_or("RHO extent overflow")?;
    let theta = (2.0 * std::f64::consts::PI) * dt.abs() as f64 / cutoff as f64;
    let muc = unsafe { tan(theta / 2.0) };
    // RHOCOF: RN=1./FLOAT(NORD) is evaluated in REAL before widening.
    let rn = (1.0_f32 / 2.0_f32) as f64;
    let sigma = 1.0 / (2.0_f64.powf(rn) - 1.0).sqrt();
    let gain = (muc * sigma / (1.0 + muc * sigma)).powi(2);
    let zn = (1.0 - muc * sigma) / (1.0 + muc * sigma);
    let a = [gain as f32, (2.0 * gain) as f32, gain as f32];
    let b = [0.0_f32, (2.0 * zn) as f32, -(zn.powi(2)) as f32];
    // Only the two preceding rows have nonzero BETA. Omitting zero terms
    // preserves the ordered nonzero additions of native SUMBF exactly.
    let mut older = vec![0.0_f32; size];
    let mut previous = vec![0.0_f32; size];
    previous[0] = 1.0;
    let mut row = vec![0.0_f32; size];
    for n in 1..=last {
        for k in 0..n {
            let mut sum = 0.0_f64;
            if n >= 2 {
                sum += (b[2] * older[k]) as f64;
            }
            sum += (b[1] * previous[k]) as f64;
            let mut alpha = if n - k <= 2 { a[n - k] } else { 0.0 };
            if n == 1 && k == 0 {
                // CN is DOUBLE; ACOEF(NN)+BCOEF(NN) is REAL.
                let cn = (a[2] + b[2]) as f64;
                alpha = (alpha as f64 + cn) as f32;
            }
            row[k] = (alpha as f64 + sum) as f32;
        }
        row[n] = a[0];
        std::mem::swap(&mut older, &mut previous);
        std::mem::swap(&mut previous, &mut row);
    }
    previous.truncate(steps + 1);
    Ok(previous)
}

/// Native half-filter, in chronological order, including its center tap.
/// Every REAL arithmetic boundary is f32; host libm matches the native
/// Fortran build on the same machine. No FMA is requested by this crate.
pub fn half_weights(
    steps: usize,
    dt: f32,
    cutoff: f32,
    filter: i32,
) -> Result<Vec<f32>, &'static str> {
    if !(0..=8).contains(&filter) {
        return Err("dfi_nfilter must be in 0..=8");
    }
    if steps == 0
        || steps > (i32::MAX as usize - 1) / 2
        || !dt.is_finite()
        || dt == 0.0
        || !cutoff.is_finite()
        || cutoff <= 0.0
    {
        return Err(
            "DFI weights require native integer extents, finite nonzero dt and positive cutoff",
        );
    }
    if filter == 8 {
        return rho_half(steps, dt, cutoff);
    }
    if filter == 7 && steps > 2500 {
        // DOLPH declares T(0:5000), then CHEBY writes T(0:2*M).
        return Err("native DOLPH CHEBY extent exceeds its T(0:5000) scratch array");
    }
    let pi = std::f32::consts::PI;
    let m = steps as f32;
    let mut h = vec![0.0; steps + 1];
    if filter >= 6 {
        // DOLPHWIN uses 2*PI/M, DOLPH uses 2*PI*ABS(DELTAT/TAUS).
        let theta = if filter == 6 {
            2.0 * pi / m
        } else {
            2.0 * pi * (dt / cutoff).abs()
        };
        let x0 = 1.0 / cos(theta / 2.0);
        let root = (x0 * x0 - 1.0).sqrt();
        let nm1 = 2 * steps;
        let rr = 0.5 * (pow(x0 + root, nm1 as f32) + pow(x0 - root, nm1 as f32));
        let r = 1.0 / rr;
        let n = (nm1 + 1) as f32;
        for (nt, out) in h.iter_mut().enumerate() {
            let mut sum = if filter == 6 { rr } else { 1.0 };
            for i in 1..=steps {
                let arg = x0 * cos(i as f32 * pi / n);
                let term1 = cheby_last(nm1, arg);
                let term2 = cos((2 * nt) as f32 * pi * i as f32 / n);
                sum += if filter == 6 {
                    (2.0 * term1) * term2
                } else {
                    ((r * 2.0) * term1) * term2
                };
            }
            *out = sum / n;
        }
        if filter == 7 {
            // DOLPH returns W2(0:M), already reversed, without NORMLZ.
            h.reverse();
            return Ok(h);
        }
    } else {
        for (i, out) in h.iter_mut().enumerate() {
            let n = i as f32;
            *out = match filter {
                0 => 1.0,
                1 => {
                    if i == 0 {
                        1.0
                    } else {
                        let a = n * pi / (steps + 1) as f32;
                        sin(a) / a
                    }
                }
                2 => {
                    if i == 0 {
                        1.0
                    } else {
                        0.54 + (1.0 - 0.54) * cos(n * pi / m)
                    }
                }
                3 => {
                    if i == 0 {
                        1.0
                    } else {
                        0.42 + 0.50 * cos(n * pi / m) + 0.08 * cos((2 * i) as f32 * pi / m)
                    }
                }
                4 => {
                    let t = n / m;
                    bessi0((1.0 - t * t).sqrt()) / bessi0(1.0)
                }
                5 => {
                    let arg = (pi * n / m) / 2.0;
                    let d = [0.35577019_f32, 0.2436983, 0.07211497, 0.00630165];
                    let mut sum = d[0];
                    for (ip, d) in d.iter().enumerate().skip(1) {
                        sum += (2.0 * d) * cos(arg * ip as f32);
                    }
                    (if i == steps { 0.5 } else { 1.0 }) * sum
                }
                _ => unreachable!(),
            };
        }
    }
    let omega = 2.0 * pi / cutoff;
    for (i, out) in h.iter_mut().enumerate() {
        let x = if i == 0 {
            omega * dt.abs() / pi
        } else {
            sin((i as f32 * omega) * dt.abs()) / (i as f32 * pi)
        };
        *out = x * *out;
    }
    let mut total = h[0];
    for word in h.iter().skip(1) {
        total += 2.0 * word;
    }
    for word in &mut h {
        *word /= total;
    }
    h.reverse();
    Ok(h)
}

/// WRF optfil_driver's odd/even mirroring, including the doubled center
/// for an even number of samples. Kernel weights must not be renormalized
/// here: dfi_accumulate and dfi_array_reset own the sequential total.
pub fn weights(duration: i32, dt: f32, cutoff: f32, filter: i32) -> Result<Vec<f32>, &'static str> {
    if duration <= 0 || !dt.is_finite() || dt == 0.0 {
        return Err("DFI duration must be positive and dt finite and nonzero");
    }
    let count = 1.0_f32 + duration as f32 / dt.abs();
    if !count.is_finite() || count >= 2147483648.0 {
        return Err("DFI sample count exceeds native INTEGER extent");
    }
    let half = (count / 2.0).ceil() as usize;
    if half < 2 {
        return Err("DFI window is shorter than two timesteps");
    }
    let mut out = half_weights(half - 1, dt, cutoff, filter)?;
    let count = count as usize;
    let reflected = if count % 2 == 1 { half - 1 } else { half };
    for i in (0..reflected).rev() {
        out.push(out[i]);
    }
    Ok(out)
}

/// Return values: 0 success, -1 invalid shape/config. Filter 8 repairs the
/// native RHOFIL bounds defect. Caller supplies `capacity` float words.
///
/// # Safety
/// `output` and `written` must be valid writable pointers of the stated sizes.
#[no_mangle]
pub unsafe extern "C" fn aw_dfi_weights(
    duration: i32,
    dt: f32,
    cutoff: f32,
    filter: i32,
    output: *mut f32,
    capacity: usize,
    written: *mut usize,
) -> i32 {
    if output.is_null() || written.is_null() {
        return -1;
    }
    *written = 0;
    let count = 1.0_f32 + duration as f32 / dt.abs();
    if !count.is_finite() || count < 1.0 || count as usize > capacity {
        return -1;
    }
    let Ok(h) = weights(duration, dt, cutoff, filter) else {
        return -1;
    };
    if h.len() > capacity {
        return -1;
    }
    std::ptr::copy_nonoverlapping(h.as_ptr(), output, h.len());
    *written = h.len();
    0
}

/// CUDA driver function pointers are bound once during setup. A native
/// dispatcher can advance one already captured model pass without a Python
/// callback or a host readback. This is not a model graph recorder.
pub type GraphLaunch = unsafe extern "C" fn(*mut c_void, *mut c_void) -> i32;

/// Version of the loader contract, independent of the forecast selector.
#[no_mangle]
pub extern "C" fn aw_dfi_runtime_abi() -> u32 {
    1
}

/// Submit a finite, time-unrolled model graph exactly once. Each solve in
/// this graph was captured with its own native clock, boundary evaluation
/// and physics cadence. No Python callback, allocation or synchronization
/// runs while the graph integrates the atmosphere.
///
/// Validate the complete signed interval before the first device write.
/// A driver error is terminal: retrying might integrate mutated state twice.
///
/// # Safety
/// `launch` must be the CUDA driver graph-launch function. The graph,
/// stream and every captured allocation must remain alive until completion.
#[no_mangle]
pub unsafe extern "C" fn aw_dfi_submit_compiled_pass(
    launch: Option<GraphLaunch>,
    graph: *mut c_void,
    stream: *mut c_void,
    start_ticks: i64,
    stop_ticks: i64,
    step_ticks: i64,
    steps: u64,
) -> i32 {
    let launch = match launch {
        Some(launch) if !graph.is_null() => launch,
        _ => return -1,
    };
    if step_ticks == 0 || steps == 0 || steps > i64::MAX as u64 {
        return -2;
    }
    let duration = match stop_ticks.checked_sub(start_ticks) {
        Some(duration) => duration,
        None => return -2,
    };
    if step_ticks.checked_mul(steps as i64) != Some(duration) {
        return -2;
    }
    launch(graph, stream)
}
pub type KernelLaunch = unsafe extern "C" fn(
    *mut c_void,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    *mut c_void,
    *mut *mut c_void,
    *mut *mut c_void,
) -> i32;

/// Submit model graph then DFI accumulate kernel for each solved step.
/// Initial t=0 accumulation must be submitted by the caller, just as WRF
/// mediation does. The kernel owns the device cursor and coefficient total.
/// A failed submission returns immediately, with no retry on mutated state.
/// No synchronization is added, even after the final submission.
///
/// # Safety
/// Handles, params, graph topology and allocations must remain alive until
/// stream completion. The graph must advance model clocks and boundary slots
/// in device memory. Repeating a graph with frozen clocks is invalid.
#[no_mangle]
pub unsafe extern "C" fn aw_dfi_submit_pass(
    graph_launch: GraphLaunch,
    kernel_launch: KernelLaunch,
    graph: *mut c_void,
    stream: *mut c_void,
    accumulate: *mut c_void,
    params: *mut *mut c_void,
    blocks: u32,
    fields: u32,
    steps: u32,
    advance_cursor: *mut c_void,
    cursor_params: *mut *mut c_void,
) -> i32 {
    if graph.is_null()
        || accumulate.is_null()
        || params.is_null()
        || advance_cursor.is_null()
        || cursor_params.is_null()
        || blocks == 0
        || fields == 0
    {
        return -1;
    }
    for _ in 0..steps {
        let status = graph_launch(graph, stream);
        if status != 0 {
            return status;
        }
        let status = kernel_launch(
            accumulate,
            blocks,
            fields,
            1,
            256,
            1,
            1,
            0,
            stream,
            params,
            std::ptr::null_mut(),
        );
        if status != 0 {
            return status;
        }
        let status = kernel_launch(
            advance_cursor,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            stream,
            cursor_params,
            std::ptr::null_mut(),
        );
        if status != 0 {
            return status;
        }
    }
    0
}

/// One domain's already captured operations. All model, boundary and
/// coefficient clocks are device resident. The model graph MUST advance
/// its signed clock and consume changing boundary/physics slots. A graph
/// that merely repeats an atmosphere at its capture time is not admissible.
///
/// `force_graph` belongs to a child and consumes the latest parent state.
/// Native DFI disables feedback in BOTH backward and forward passes;
/// forecast initialization later restores the configured feedback mode.
#[repr(C)]
pub struct DomainGraphs {
    pub parent: i32,
    pub ratio: u32,
    pub step_ticks: i64,
    pub model_graph: *mut c_void,
    pub force_graph: *mut c_void,
    pub accumulate_graph: *mut c_void,
    pub advance_graph: *mut c_void,
}

fn validate_tree(nodes: &[DomainGraphs], root: usize) -> Result<Vec<Vec<usize>>, i32> {
    if nodes.is_empty() || nodes.len() > 1024 || root >= nodes.len() {
        return Err(-1);
    }
    let mut children = vec![Vec::new(); nodes.len()];
    for (i, node) in nodes.iter().enumerate() {
        if node.model_graph.is_null()
            || node.step_ticks == 0
            || node.step_ticks == i64::MIN
            || (!node.accumulate_graph.is_null() && node.advance_graph.is_null())
        {
            return Err(-2);
        }
        if i == root {
            if node.parent != -1 || node.ratio != 1 {
                return Err(-3);
            }
        } else {
            if node.parent < 0
                || node.parent as usize >= nodes.len()
                || node.parent as usize == i
                || node.ratio == 0
                || node.force_graph.is_null()
            {
                return Err(-3);
            }
            let parent = node.parent as usize;
            if node.step_ticks.checked_mul(node.ratio as i64) != Some(nodes[parent].step_ticks) {
                return Err(-4);
            }
            children[parent].push(i);
        }
    }
    // Reject disconnected cycles before launching anything, including an
    // unrelated cycle beside an otherwise well-formed root component.
    let mut visited = vec![false; nodes.len()];
    let mut stack = vec![root];
    while let Some(i) = stack.pop() {
        if visited[i] {
            return Err(-5);
        }
        visited[i] = true;
        stack.extend(children[i].iter().copied());
    }
    if visited.iter().any(|seen| !seen) {
        return Err(-5);
    }
    Ok(children)
}

unsafe fn submit_tree_interval(
    launch: GraphLaunch,
    nodes: &[DomainGraphs],
    children: &[Vec<usize>],
    i: usize,
    steps: u64,
    stream: *mut c_void,
) -> i32 {
    let node = &nodes[i];
    // frame/module_integrate.F:365 and :490: sample before EACH solve
    // and once more at the end of EACH recursive integrate() invocation.
    // Child endpoint samples are intentionally repeated on re-entry.
    for _ in 0..steps {
        if !node.accumulate_graph.is_null() {
            let status = launch(node.accumulate_graph, stream);
            if status != 0 {
                return status;
            }
        }
        let status = launch(node.model_graph, stream);
        if status != 0 {
            return status;
        }
        if !node.advance_graph.is_null() {
            let status = launch(node.advance_graph, stream);
            if status != 0 {
                return status;
            }
        }
        // Native forces ALL siblings before integrating any sibling.
        for &child in &children[i] {
            let status = launch(nodes[child].force_graph, stream);
            if status != 0 {
                return status;
            }
        }
        for &child in &children[i] {
            let status = submit_tree_interval(
                launch,
                nodes,
                children,
                child,
                nodes[child].ratio as u64,
                stream,
            );
            if status != 0 {
                return status;
            }
        }
    }
    if !node.accumulate_graph.is_null() {
        let status = launch(node.accumulate_graph, stream);
        if status != 0 {
            return status;
        }
    }
    0
}

/// Submit one complete signed, nested DFI pass through the CUDA driver.
/// The native integer ratio recursion is validated BEFORE the first write.
/// There are no Python callbacks, readbacks, waits, per-step allocations or
/// synchronizations here. An error terminates submission immediately; a
/// partially submitted pass must never be retried from its mutated state.
/// Accumulation follows literal native recursive entry/exit ordering, including
/// repeated child endpoint samples with the same coefficient index. The
/// accumulate graph MUST NOT advance that index. advance_graph does so once
/// after an actual model solve; it is mandatory when accumulation is enabled.
///
/// # Safety
/// All graphs, pointer tables, buffers and the stream must remain alive
/// through completion. `nodes` must contain `count` initialized entries.
/// The CUDA launch function must belong to the active CUDA context.
#[no_mangle]
pub unsafe extern "C" fn aw_dfi_submit_tree(
    launch: GraphLaunch,
    nodes: *const DomainGraphs,
    count: usize,
    root: usize,
    duration_ticks: i64,
    stream: *mut c_void,
) -> i32 {
    if nodes.is_null()
        || count == 0
        || count > 1024
        || duration_ticks == 0
        || duration_ticks == i64::MIN
    {
        return -1;
    }
    let nodes = std::slice::from_raw_parts(nodes, count);
    let children = match validate_tree(nodes, root) {
        Ok(children) => children,
        Err(status) => return status,
    };
    let dt = nodes[root].step_ticks;
    if duration_ticks.signum() != dt.signum() || duration_ticks % dt != 0 {
        return -6;
    }
    submit_tree_interval(
        launch,
        nodes,
        &children,
        root,
        (duration_ticks / dt) as u64,
        stream,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    thread_local! {
        static LAUNCHES: RefCell<Vec<usize>> = const { RefCell::new(Vec::new()) };
        static FAIL_AT: RefCell<Option<usize>> = const { RefCell::new(None) };
    }

    unsafe extern "C" fn witnessed_launch(graph: *mut c_void, _: *mut c_void) -> i32 {
        let id = graph as usize;
        LAUNCHES.with(|events| events.borrow_mut().push(id));
        if FAIL_AT.with(|at| *at.borrow() == Some(id)) {
            77
        } else {
            0
        }
    }

    #[test]
    fn compiled_pass_launches_once_in_both_directions() {
        for sign in [-1, 1] {
            LAUNCHES.with(|events| events.borrow_mut().clear());
            FAIL_AT.with(|at| *at.borrow_mut() = None);
            let status = unsafe {
                aw_dfi_submit_compiled_pass(
                    Some(witnessed_launch),
                    41usize as _,
                    std::ptr::null_mut(),
                    300,
                    300 + sign * 120,
                    sign * 15,
                    8,
                )
            };
            assert_eq!(status, 0);
            LAUNCHES.with(|events| assert_eq!(*events.borrow(), vec![41]));
        }
    }

    #[test]
    fn compiled_pass_invalid_clock_writes_nothing() {
        LAUNCHES.with(|events| events.borrow_mut().clear());
        for (start, stop, dt, steps) in [
            (0, 120, -15, 8),
            (0, 119, 15, 8),
            (0, 0, 0, 1),
            (0, 0, 15, 0),
            (i64::MIN, i64::MAX, 1, 1),
            (0, 1, i64::MAX, 2),
            (0, 1, 1, u64::MAX),
        ] {
            let status = unsafe {
                aw_dfi_submit_compiled_pass(
                    Some(witnessed_launch),
                    41usize as _,
                    std::ptr::null_mut(),
                    start,
                    stop,
                    dt,
                    steps,
                )
            };
            assert_eq!(status, -2);
        }
        LAUNCHES.with(|events| assert!(events.borrow().is_empty()));
    }

    #[test]
    fn compiled_pass_failure_does_not_retry() {
        LAUNCHES.with(|events| events.borrow_mut().clear());
        FAIL_AT.with(|at| *at.borrow_mut() = Some(41));
        let status = unsafe {
            aw_dfi_submit_compiled_pass(
                Some(witnessed_launch),
                41usize as _,
                std::ptr::null_mut(),
                0,
                -120,
                -15,
                8,
            )
        };
        assert_eq!(status, 77);
        LAUNCHES.with(|events| assert_eq!(*events.borrow(), vec![41]));
        FAIL_AT.with(|at| *at.borrow_mut() = None);
    }

    fn nested_graphs(sign: i64) -> [DomainGraphs; 3] {
        [
            DomainGraphs {
                parent: -1,
                ratio: 1,
                step_ticks: 6 * sign,
                model_graph: 10usize as _,
                force_graph: std::ptr::null_mut(),
                accumulate_graph: 11usize as _,
                advance_graph: 13usize as _,
            },
            DomainGraphs {
                parent: 0,
                ratio: 2,
                step_ticks: 3 * sign,
                model_graph: 20usize as _,
                force_graph: 22usize as _,
                accumulate_graph: 21usize as _,
                advance_graph: 23usize as _,
            },
            DomainGraphs {
                parent: 1,
                ratio: 3,
                step_ticks: sign,
                model_graph: 30usize as _,
                force_graph: 32usize as _,
                accumulate_graph: 31usize as _,
                advance_graph: 33usize as _,
            },
        ]
    }

    #[test]
    fn signed_nested_native_order() {
        // frame/module_integrate.F: parent solve, nest force, child solves;
        // DFI samples before each solve and once at every recursive return.
        let expected = vec![
            11, 10, 13, 22, 21, 20, 23, 32, 31, 30, 33, 31, 30, 33, 31, 30, 33, 31, 21, 20, 23, 32,
            31, 30, 33, 31, 30, 33, 31, 30, 33, 31, 21, 11,
        ];
        for sign in [-1, 1] {
            LAUNCHES.with(|x| x.borrow_mut().clear());
            FAIL_AT.with(|x| *x.borrow_mut() = None);
            let nodes = nested_graphs(sign);
            let status = unsafe {
                aw_dfi_submit_tree(
                    witnessed_launch,
                    nodes.as_ptr(),
                    3,
                    0,
                    6 * sign,
                    std::ptr::null_mut(),
                )
            };
            assert_eq!(status, 0);
            LAUNCHES.with(|x| assert_eq!(*x.borrow(), expected));
        }
    }

    #[test]
    fn invalid_nested_clock_rejected_before_write() {
        LAUNCHES.with(|x| x.borrow_mut().clear());
        let mut nodes = nested_graphs(-1);
        nodes[2].step_ticks = 1;
        let status = unsafe {
            aw_dfi_submit_tree(
                witnessed_launch,
                nodes.as_ptr(),
                3,
                0,
                -6,
                std::ptr::null_mut(),
            )
        };
        assert_eq!(status, -4);
        LAUNCHES.with(|x| assert!(x.borrow().is_empty()));
    }

    #[test]
    fn partial_launch_failure_is_terminal() {
        LAUNCHES.with(|x| x.borrow_mut().clear());
        FAIL_AT.with(|x| *x.borrow_mut() = Some(20));
        let nodes = nested_graphs(1);
        let status = unsafe {
            aw_dfi_submit_tree(
                witnessed_launch,
                nodes.as_ptr(),
                3,
                0,
                6,
                std::ptr::null_mut(),
            )
        };
        assert_eq!(status, 77);
        LAUNCHES.with(|x| assert_eq!(*x.borrow(), vec![11, 10, 13, 22, 21, 20]));
        FAIL_AT.with(|x| *x.borrow_mut() = None);
    }

    #[test]
    fn disconnected_cycle_is_rejected() {
        let mut nodes = nested_graphs(1);
        nodes[1].parent = 2;
        nodes[1].ratio = 1;
        nodes[1].step_ticks = 1;
        nodes[2].ratio = 1;
        assert!(matches!(validate_tree(&nodes, 0), Err(-5)));
    }

    #[test]
    fn native_lifecycle_times() {
        assert_eq!(
            lifecycle(1, -3600, 3600, false).unwrap().forecast_start,
            1800
        );
        let dd = lifecycle(2, -3600, 3600, false).unwrap();
        assert!(dd.passes[0].filter);
        assert!(!dd.passes[0].reset);
        assert_eq!(dd.passes[1].start, -3600);
        let td = lifecycle(3, -3600, 1800, false).unwrap();
        assert!(td.passes[0].reset);
        assert_eq!(td.passes[1].start, -1800);
        for opt in 0..=3 {
            assert!(lifecycle(opt, -3600, 3600, true).unwrap().passes.is_empty());
        }
        assert!(lifecycle(0, 0, 0, false).unwrap().passes.is_empty());
    }
}


/// The revision stamp is exported so it survives shared-library linking.
#[no_mangle]
pub extern "C" fn gpuwm_dfi_source_rev() -> *const std::os::raw::c_char {
    static SOURCE_REV_STAMP: &str = concat!(
        "GPUWM_BRIDGE_SOURCE_REV=", env!("GPUWM_BRIDGE_SOURCE_REV"), "\0");
    SOURCE_REV_STAMP.as_ptr().cast()
}
