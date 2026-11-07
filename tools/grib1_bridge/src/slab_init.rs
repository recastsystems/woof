//! Slab-specific soil preprocessing. WRF 4.6.1 module_soil_pre.F.
//! f32 geometry and profiles use separate rounded operations, never mul_add.
use rayon::prelude::*;
use std::slice;
fn depths() -> [f32; 5] {
    let mut z = [0.0f32; 5];
    let mut d = 0.01f32;
    z[0] = 0.5 * d;
    for l in 1..5 {
        let prev = d;
        d = 2.0 * d;
        z[l] = (z[l - 1] + 0.5 * prev) + 0.5 * d;
    }
    z
}
#[cfg(test)]
fn profile(tsk: f32, tmn: f32, land: bool, water: f32) -> [f32; 5] {
    let z = depths();
    let mut t = [water; 5];
    if land {
        for l in 0..5 {
            t[l] = (tsk * (z[4] - z[l]) + tmn * (z[l] - z[0])) / (z[4] - z[0]);
        }
    }
    t
}
/// Caller provides nonaliasing contiguous cell buffers of length n.
#[no_mangle]
pub unsafe extern "C" fn gpuwm_slab_threshold_v1(
    n: usize,
    decision: *const f64,
    lake: *const f64,
    skin: *const f64,
    ice: *const f64,
    threshold: f64,
    terrestrial: *mut u8,
    converted: *mut f64,
) -> i32 {
    if n == 0
        || !threshold.is_finite()
        || decision.is_null()
        || lake.is_null()
        || skin.is_null()
        || ice.is_null()
        || terrestrial.is_null()
        || converted.is_null()
    {
        return 1;
    }
    let (d, l, s, i) = (
        slice::from_raw_parts(decision, n),
        slice::from_raw_parts(lake, n),
        slice::from_raw_parts(skin, n),
        slice::from_raw_parts(ice, n),
    );
    let (t, c) = (
        slice::from_raw_parts_mut(terrestrial, n),
        slice::from_raw_parts_mut(converted, n),
    );
    t.par_iter_mut()
        .zip(c.par_iter_mut())
        .enumerate()
        .for_each(|(x, (t, c))| {
            let land = d[x] >= 0.5 && l[x] == 0.0;
            *t = land as u8;
            *c = if !land && s[x] < threshold { 1.0 } else { i[x] };
        });
    0
}
/// Layer-major temperatures and output; all other arrays are cell-major.
#[no_mangle]
pub unsafe extern "C" fn gpuwm_slab_initialize_v1(
    n: usize,
    nl: usize,
    levels: *const f64,
    temps: *const f64,
    scale: f64,
    terrestrial: *const u8,
    land: *const f64,
    skin: *const f64,
    sst: *const f64,
    delta: *const f64,
    delta_zero: *const f64,
    has_terrain: u32,
    exact: u32,
    bottom: *mut f64,
    output: *mut f32,
) -> i32 {
    if n == 0
        || nl == 0
        || n.checked_mul(nl).is_none()
        || n.checked_mul(5).is_none()
        || !scale.is_finite()
        || scale <= 0.0
        || levels.is_null()
        || temps.is_null()
        || terrestrial.is_null()
        || land.is_null()
        || skin.is_null()
        || sst.is_null()
        || delta.is_null()
        || delta_zero.is_null()
        || bottom.is_null()
        || output.is_null()
    {
        return 1;
    }
    let levels = slice::from_raw_parts(levels, nl);
    if levels.iter().any(|v| !v.is_finite()) {
        return 2;
    }
    let target = 30.0 * scale / 100.0;
    let mut chosen = 0;
    for l in 1..nl {
        if (levels[l] - target).abs() < (levels[chosen] - target).abs() {
            chosen = l;
        }
    }
    let near = levels.iter().any(|v| *v <= 40.0 * scale / 100.0);
    let temp = slice::from_raw_parts(temps, n * nl);
    let terr = slice::from_raw_parts(terrestrial, n);
    let (land, skin, sst, delta, delta_zero) = (
        slice::from_raw_parts(land, n),
        slice::from_raw_parts(skin, n),
        slice::from_raw_parts(sst, n),
        slice::from_raw_parts(delta, n),
        slice::from_raw_parts(delta_zero, n),
    );
    let bottom = slice::from_raw_parts_mut(bottom, n);
    let output = slice::from_raw_parts_mut(output, n * 5);
    bottom.par_iter_mut().enumerate().for_each(|(c, bottom)| {
        let is_land = land[c] >= 0.5;
        let orig = terr[c] != 0;
        let ice = is_land && !orig;
        let mut tmn = temp[chosen * n + c];
        if has_terrain != 0 {
            tmn += if near { delta[c] } else { delta_zero[c] };
        }
        if orig && (!tmn.is_finite() || !(170.0..=400.0).contains(&tmn)) {
            tmn = skin[c];
        }
        if !is_land {
            tmn = skin[c];
        }
        if ice {
            tmn = 271.4;
        }
        *bottom = (tmn as f32) as f64;
    });
    output.par_chunks_mut(n).enumerate().for_each(|(l, row)| {
        row.par_iter_mut().enumerate().for_each(|(c, value)| {
            let is_land = land[c] >= 0.5;
            let orig = terr[c] != 0;
            let ice = is_land && !orig;
            let water =
                if exact != 0 && ice && sst[c].is_finite() && sst[c] > 170.0 && sst[c] < 400.0 {
                    sst[c]
                } else {
                    skin[c]
                };
            *value = profile_layer(
                l,
                skin[c] as f32,
                bottom[c] as f32,
                if exact != 0 { orig } else { is_land },
                water as f32,
            );
        });
    });
    0
}
#[no_mangle]
pub unsafe extern "C" fn gpuwm_slab_profile_v1(
    n: usize,
    skin: *const f32,
    bottom: *const f32,
    land: *const u8,
    water: *const f32,
    output: *mut f32,
) -> i32 {
    if n == 0
        || n.checked_mul(5).is_none()
        || skin.is_null()
        || bottom.is_null()
        || land.is_null()
        || water.is_null()
        || output.is_null()
    {
        return 1;
    }
    let (s, b, l, w) = (
        slice::from_raw_parts(skin, n),
        slice::from_raw_parts(bottom, n),
        slice::from_raw_parts(land, n),
        slice::from_raw_parts(water, n),
    );
    let o = slice::from_raw_parts_mut(output, n * 5);
    o.par_chunks_mut(n).enumerate().for_each(|(z, row)| {
        row.par_iter_mut().enumerate().for_each(|(c, value)| {
            *value = profile_layer(z, s[c], b[c], l[c] != 0, w[c]);
        });
    });
    0
}
fn profile_layer(l: usize, tsk: f32, tmn: f32, land: bool, water: f32) -> f32 {
    let z = depths();
    if land {
        (tsk * (z[4] - z[l]) + tmn * (z[l] - z[0])) / (z[4] - z[0])
    } else {
        water
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn soil_geometry_and_boundary() {
        let z = depths();
        assert_eq!(z[0], 0.005f32);
        assert_eq!(z[4].to_bits(), 0.22999999f32.to_bits());
        let t = profile(269.0, 271.4, true, 286.0);
        assert_eq!(t[4], 271.4);
        assert_eq!(profile(269.0, 271.4, false, 286.0), [286.0; 5]);
    }

    #[test]
    fn parallel_preparation_keeps_column_words() {
        // Mixed land, water, converted ice, terrain corrections, invalid
        // source temperatures and both source-order policies. The scalar
        // authority is the pre-parallelization formula, not another FFI call.
        let n = 4097;
        let levels = [0.07, 0.28, 1.0];
        let decision: Vec<_> = (0..n).map(|c| if c % 3 == 0 { 1.0 } else { 0.0 }).collect();
        let lake = vec![0.0; n];
        let skin: Vec<_> = (0..n).map(|c| 265.0 + (c % 60) as f64).collect();
        let source_ice = vec![0.0; n];
        let temps: Vec<_> = (0..3 * n)
            .map(|c| {
                if c % 17 == 0 {
                    f64::NAN
                } else {
                    278.0 + (c % 19) as f64
                }
            })
            .collect();
        let sst = vec![286.0; n];
        let delta = vec![-1.5; n];
        let delta_zero = vec![-3.0; n];
        for exact in [0, 1] {
            let run = |threads| {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .unwrap();
                pool.install(|| {
                    let mut terr = vec![0u8; n];
                    let mut ice = vec![0.0; n];
                    let mut bottom = vec![0.0; n];
                    let mut output = vec![0.0; n * 5];
                    unsafe {
                        assert_eq!(
                            gpuwm_slab_threshold_v1(
                                n,
                                decision.as_ptr(),
                                lake.as_ptr(),
                                skin.as_ptr(),
                                source_ice.as_ptr(),
                                270.0,
                                terr.as_mut_ptr(),
                                ice.as_mut_ptr()
                            ),
                            0
                        );
                    }
                    let land: Vec<_> = (0..n)
                        .map(|c| {
                            if terr[c] != 0 || ice[c] >= 0.5 {
                                1.0
                            } else {
                                0.0
                            }
                        })
                        .collect();
                    unsafe {
                        assert_eq!(
                            gpuwm_slab_initialize_v1(
                                n,
                                3,
                                levels.as_ptr(),
                                temps.as_ptr(),
                                1.0,
                                terr.as_ptr(),
                                land.as_ptr(),
                                skin.as_ptr(),
                                sst.as_ptr(),
                                delta.as_ptr(),
                                delta_zero.as_ptr(),
                                1,
                                exact,
                                bottom.as_mut_ptr(),
                                output.as_mut_ptr()
                            ),
                            0
                        );
                    }
                    for c in 0..n {
                        let orig = decision[c] >= 0.5;
                        assert_eq!(terr[c] != 0, orig);
                        let converted = !orig && skin[c] < 270.0;
                        assert_eq!(ice[c], if converted { 1.0 } else { 0.0 });
                        let mut tmn = temps[n + c] - 1.5;
                        if orig && (!tmn.is_finite() || !(170.0..=400.0).contains(&tmn)) {
                            tmn = skin[c];
                        }
                        if !orig {
                            tmn = if converted { 271.4 } else { skin[c] };
                        }
                        assert_eq!(bottom[c].to_bits(), ((tmn as f32) as f64).to_bits());
                        let expected = profile(
                            skin[c] as f32,
                            tmn as f32,
                            if exact != 0 { orig } else { land[c] >= 0.5 },
                            if exact != 0 && converted {
                                286.0
                            } else {
                                skin[c] as f32
                            },
                        );
                        for l in 0..5 {
                            assert_eq!(output[l * n + c].to_bits(), expected[l].to_bits());
                        }
                    }
                    (
                        bottom.into_iter().map(f64::to_bits).collect::<Vec<_>>(),
                        output.into_iter().map(f32::to_bits).collect::<Vec<_>>(),
                    )
                })
            };
            assert_eq!(run(1), run(4));
        }
    }
}
