//! WRF init_soil_2_real point sampling, with REAL operation order.
//! Layer values are anchored at 0 and 3 m by the orchestration metadata.
use std::panic::{catch_unwind, AssertUnwindSafe};
use crate::{ERR_DIMENSION, ERR_NULL, ERR_PANIC, OK};

/// # Safety
/// Input contains ns*columns words; depths contains ns; targets contains nt;
/// output contains nt*columns. Buffers must not overlap.
#[no_mangle]
pub unsafe extern "C" fn gpuwm_soil_points_f32(
    input: *const f32, depths: *const f32, targets: *const f32,
    output: *mut f32, ns: usize, nt: usize, columns: usize,
) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        if ns < 2 || nt == 0 || columns == 0
            || ns.checked_mul(columns).is_none() || nt.checked_mul(columns).is_none() {
            return ERR_DIMENSION;
        }
        if input.is_null() || depths.is_null() || targets.is_null() || output.is_null() {
            return ERR_NULL;
        }
        let z = std::slice::from_raw_parts(depths, ns);
        let target = std::slice::from_raw_parts(targets, nt);
        if z.iter().any(|x| !x.is_finite()) || z.windows(2).any(|p| p[0] >= p[1]) {
            return ERR_DIMENSION;
        }
        let mut lower_levels = Vec::with_capacity(nt);
        for &wanted in target {
            match (0..ns-1).find(|&i| wanted >= z[i] && wanted <= z[i+1]) {
                Some(lower) => lower_levels.push(lower),
                None => return ERR_DIMENSION,
            }
        }
        let input_address = input as usize;
        let output_address = output as usize;
        // Each worker owns disjoint columns at every target depth. Keep
        // the scalar WRF operation order inside a column.
        crate::parallel::run_ranges(columns, crate::parallel::resources::available_workers(), |start, stop| {
            let input = input_address as *const f32;
            let output = output_address as *mut f32;
            for (k, (&wanted, &lower)) in target.iter().zip(&lower_levels).enumerate() {
                for c in start..stop {
                    let left = *input.add(lower * columns + c) * (z[lower+1] - wanted);
                    let right = *input.add((lower+1) * columns + c) * (wanted - z[lower]);
                    *output.add(k * columns + c) = (left + right) / (z[lower+1] - z[lower]);
                }
            }
        });
        OK
    })).unwrap_or(ERR_PANIC)
}
