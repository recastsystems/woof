//! Sea-level pressure over high terrain, smoothed the way operational map
//! graphics smooth it, so a 3 km map's isobars over the mountains are
//! readable and the lows and highs over flat ground are untouched.
//!
//! THE DEFECT THIS FIXES (hex RRFS v2 conus3km maps, 2026-10-29; the same
//! reduction runs on every ArWen frame).  A sea-level reduction carries the
//! surface pressure down through a fictitious column whose temperature comes
//! from the lowest model level (NCEP Shuell on MPAS-bridge frames,
//! [`rw_mpas::mslp`]; wrf-python's DCOMPUTESEAPRS on WRF frames).  At 3 km
//! the lowest level resolves every valley cold pool and every sunlit ridge,
//! and over 1.5-3 km of terrain a 1 K difference in that temperature moves
//! the reduced pressure by about 1.4 hPa.  The result was right where the
//! ground is low (the conus3km f001 low within ~1 hPa of the model's own
//! diagnostic) and a dense tangle of small closed isobars over the Rockies,
//! the Great Basin and the Sierra Madre.
//! Measured on the conus3km 2023-03-31 12Z frame (1996 x 1301, dx 3.02 km):
//! 56 isobar crossings (4 hPa) per 1,000 grid edges above 1,500 m against
//! 11.5 over the 300-1,000 m Plains and 6.3 over the low East.
//!
//! WHAT OPERATIONAL PRODUCTS DO.  The fictitious column below a mountain is
//! a convention, not a measurement, so operational graphics draw a smoothed
//! sea-level pressure there: the RUC/RAP/HRRR "MAPS" reduction (Benjamin, S.
//! G., and P. A. Miller, 1990: An alternative sea level pressure reduction
//! and a statistical comparison of geostrophic wind estimates with observed
//! surface winds. Mon. Wea. Rev., 118, 2099-2116) replaces the surface
//! temperature by the 700 hPa temperature carried down at 6.5 K/km, to keep
//! the local surface temperature out of the reduction; and grid graphics
//! smooth the field with the classic nine-point smoother, applied as many
//! times as the grid spacing needs (GEMPAK's SM9S: 1/4 centre, 1/8 sides,
//! 1/16 corners, i.e. the 1-2-1 binomial filter in each direction).  On the
//! frame above the MAPS reduction alone still left 30.6 crossings per 1,000
//! edges above 1,500 m: the reduction halves the tangle, the smoothing is
//! what removes it, so the reduction is kept and the smoothing added.
//!
//! THE FILTER HERE.
//!
//! * The field is smoothed by `n` passes of the nine-point smoother, missing
//!   values (off-mesh, boundary rings) excluded and renormalised.  `n` passes
//!   of the 1-2-1 filter are a binomial approximation to a Gaussian of
//!   standard deviation `dx sqrt(n / 2)`, so `n = 2 (L / dx)^2` gives the same
//!   physical smoothing length `L` on any grid: 40 km by default, which is
//!   about 350 passes at 3 km and 19 at 13 km.  `n` is capped at
//!   [`MAX_PASSES`] (a 1 km grid then smooths over about 14 km).
//! * The smoothed field replaces the reduced one by the terrain weight
//!   `w = (z_s - 500 m) / (1,500 m - 500 m)`, clamped to [0, 1]:
//!   `p = p + w (p_smooth - p)`.  Ground at or below 500 m keeps its reduced
//!   pressure word for word, so lows and highs over the Plains, the coasts
//!   and the ocean do not move; above 1,500 m the map draws the smoothed
//!   field.  A frame with no ground above 500 m is returned untouched.
//! * Measured on the conus3km frame: crossings above 1,500 m 56 -> 10.0 per
//!   1,000 edges (Plains 11.5 -> 11.0), largest change 9.2 hPa above 1,500 m
//!   and 2.2 hPa at 500-1,000 m, none below 500 m; the Plains low's minimum
//!   is unchanged (993.29 hPa).  At 25 km more small closed rings stayed
//!   over the Uintas and the Wasatch (12.8 crossings).
//!
//! `RUSTWX_MSLP_TERRAIN_SMOOTH_KM` sets `L` in km; `0` draws the unsmoothed
//! reduction (for a before/after with one binary).

use rayon::prelude::*;

/// Terrain at or below this height keeps its reduced pressure unchanged (m).
pub(crate) const SMOOTH_FLOOR_M: f32 = 500.0;
/// Terrain at or above this height takes the smoothed pressure in full (m).
pub(crate) const SMOOTH_FULL_M: f32 = 1_500.0;
/// The default smoothing length (km): the standard deviation of the
/// Gaussian the repeated nine-point smoother approximates.
pub(crate) const DEFAULT_LENGTH_KM: f64 = 40.0;
/// The most nine-point passes one frame is given.
pub(crate) const MAX_PASSES: usize = 400;

/// The smoothing length the environment asks for, in metres (0 = off).
pub(crate) fn smoothing_length_m() -> f64 {
    std::env::var("RUSTWX_MSLP_TERRAIN_SMOOTH_KM")
        .ok()
        .and_then(|text| text.trim().parse::<f64>().ok())
        .filter(|km| km.is_finite() && *km >= 0.0)
        .unwrap_or(DEFAULT_LENGTH_KM)
        * 1_000.0
}

/// Nine-point passes for a smoothing length `length_m` on spacing `dx_m`.
pub(crate) fn passes_for(length_m: f64, dx_m: f64) -> usize {
    if !(length_m > 0.0) || !(dx_m > 0.0) || !dx_m.is_finite() {
        return 0;
    }
    let ratio = length_m / dx_m;
    ((2.0 * ratio * ratio).round() as usize).clamp(1, MAX_PASSES)
}

/// The terrain weight: 0 at and below [`SMOOTH_FLOOR_M`], 1 at and above
/// [`SMOOTH_FULL_M`], linear between; missing terrain is 0.
pub(crate) fn terrain_weight(terrain_m: f32) -> f32 {
    if !terrain_m.is_finite() {
        return 0.0;
    }
    ((terrain_m - SMOOTH_FLOOR_M) / (SMOOTH_FULL_M - SMOOTH_FLOOR_M)).clamp(0.0, 1.0)
}

/// The grid spacing (m) at the centre of a lat/lon-located grid: the mean
/// great-circle distance to the next point east and north.
pub(crate) fn centre_spacing_m(lat_deg: &[f32], lon_deg: &[f32], nx: usize, ny: usize) -> Option<f64> {
    if nx < 2 || ny < 2 || lat_deg.len() != nx * ny || lon_deg.len() != nx * ny {
        return None;
    }
    let (j, i) = ((ny - 1) / 2, (nx - 1) / 2);
    let at = |jj: usize, ii: usize| (lat_deg[jj * nx + ii] as f64, lon_deg[jj * nx + ii] as f64);
    let gc = |(a1, o1): (f64, f64), (a2, o2): (f64, f64)| {
        let (a1, o1, a2, o2) = (a1.to_radians(), o1.to_radians(), a2.to_radians(), o2.to_radians());
        let h = ((a2 - a1) / 2.0).sin().powi(2) + a1.cos() * a2.cos() * ((o2 - o1) / 2.0).sin().powi(2);
        2.0 * 6_371_000.0 * h.sqrt().min(1.0).asin()
    };
    let dx = 0.5 * (gc(at(j, i), at(j, i + 1)) + gc(at(j, i), at(j + 1, i)));
    (dx.is_finite() && dx > 0.0).then_some(dx)
}

/// Smooth `values` (one plane, row-major `ny` x `nx`) over high `terrain`
/// with `passes` nine-point passes.  Returns the number of grid points
/// changed.
pub(crate) fn smooth_over_high_terrain(
    values: &mut [f32],
    terrain: &[f32],
    nx: usize,
    ny: usize,
    passes: usize,
) -> usize {
    let cells = nx.saturating_mul(ny);
    if passes == 0 || nx < 3 || ny < 3 || values.len() != cells || terrain.len() != cells {
        return 0;
    }
    // The bounding box of the weighted ground, widened by the smoother's
    // reach (one point per pass) so every weighted point sees exactly what
    // a whole-grid pass would give it.
    let (mut j0, mut j1, mut i0, mut i1) = (usize::MAX, 0usize, usize::MAX, 0usize);
    for j in 0..ny {
        for i in 0..nx {
            if terrain_weight(terrain[j * nx + i]) > 0.0 {
                j0 = j0.min(j);
                j1 = j1.max(j);
                i0 = i0.min(i);
                i1 = i1.max(i);
            }
        }
    }
    if j0 == usize::MAX {
        return 0;
    }
    let reach = passes + 1;
    let (j0, j1) = (j0.saturating_sub(reach), (j1 + reach).min(ny - 1));
    let (i0, i1) = (i0.saturating_sub(reach), (i1 + reach).min(nx - 1));
    let (bx, by) = (i1 - i0 + 1, j1 - j0 + 1);

    // Numerator (finite values) and denominator (finite mask) are smoothed
    // alike; their ratio is the smoother over the points that exist.
    let mut sum: Vec<f64> = vec![0.0; bx * by];
    let mut mass: Vec<f64> = vec![0.0; bx * by];
    sum.par_chunks_mut(bx)
        .zip(mass.par_chunks_mut(bx))
        .enumerate()
        .for_each(|(jj, (srow, mrow))| {
            let row = &values[(j0 + jj) * nx + i0..(j0 + jj) * nx + i0 + bx];
            for (k, &v) in row.iter().enumerate() {
                if v.is_finite() {
                    srow[k] = v as f64;
                    mrow[k] = 1.0;
                }
            }
        });
    let mut scratch_s = vec![0.0f64; bx * by];
    let mut scratch_m = vec![0.0f64; bx * by];
    for _ in 0..passes {
        one_two_one_x(&sum, &mut scratch_s, bx);
        one_two_one_x(&mass, &mut scratch_m, bx);
        one_two_one_y(&scratch_s, &mut sum, bx, by);
        one_two_one_y(&scratch_m, &mut mass, bx, by);
    }

    let changed = std::sync::atomic::AtomicUsize::new(0);
    values
        .par_chunks_mut(nx)
        .enumerate()
        .skip(j0)
        .take(by)
        .for_each(|(j, row)| {
            let jj = j - j0;
            let mut n = 0usize;
            for (k, value) in row[i0..i0 + bx].iter_mut().enumerate() {
                let w = terrain_weight(terrain[j * nx + i0 + k]);
                let m = mass[jj * bx + k];
                if w > 0.0 && value.is_finite() && m > 0.0 {
                    let smooth = sum[jj * bx + k] / m;
                    let blended = (*value as f64 + w as f64 * (smooth - *value as f64)) as f32;
                    if blended != *value {
                        *value = blended;
                        n += 1;
                    }
                }
            }
            changed.fetch_add(n, std::sync::atomic::Ordering::Relaxed);
        });
    changed.into_inner()
}

/// One 1-2-1 pass along rows; points beyond the box count as missing.
fn one_two_one_x(src: &[f64], dst: &mut [f64], bx: usize) {
    dst.par_chunks_mut(bx).zip(src.par_chunks(bx)).for_each(|(out, row)| {
        for i in 0..bx {
            let left = if i > 0 { row[i - 1] } else { 0.0 };
            let right = if i + 1 < bx { row[i + 1] } else { 0.0 };
            out[i] = 0.25 * left + 0.5 * row[i] + 0.25 * right;
        }
    });
}

/// One 1-2-1 pass along columns; points beyond the box count as missing.
fn one_two_one_y(src: &[f64], dst: &mut [f64], bx: usize, by: usize) {
    dst.par_chunks_mut(bx).enumerate().for_each(|(j, out)| {
        let mid = &src[j * bx..(j + 1) * bx];
        let up = (j > 0).then(|| &src[(j - 1) * bx..j * bx]);
        let down = (j + 1 < by).then(|| &src[(j + 1) * bx..(j + 2) * bx]);
        for i in 0..bx {
            out[i] = 0.25 * up.map_or(0.0, |r| r[i]) + 0.5 * mid[i] + 0.25 * down.map_or(0.0, |r| r[i]);
        }
    });
}

/// The map's sea-level pressure from a reduced plane: smoothed over high
/// terrain by the environment's (or the default) smoothing length on the
/// grid's own spacing.  Returns the number of grid points changed.
pub(crate) fn display_mslp(
    values: &mut [f32],
    terrain: &[f32],
    lat_deg: &[f32],
    lon_deg: &[f32],
    nx: usize,
    ny: usize,
) -> usize {
    let Some(dx) = centre_spacing_m(lat_deg, lon_deg, nx, ny) else {
        return 0;
    };
    smooth_over_high_terrain(values, terrain, nx, ny, passes_for(smoothing_length_m(), dx))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn low_ground_keeps_its_pressure_word_for_word() {
        // a noisy field over a plain at 400 m beside a 2,000 m plateau
        let (nx, ny) = (40, 30);
        let mut values = Vec::new();
        let mut terrain = Vec::new();
        for j in 0..ny {
            for i in 0..nx {
                let noise = if (i + j) % 2 == 0 { 600.0 } else { -600.0 };
                values.push(100_000.0 + 10.0 * i as f32 + noise);
                terrain.push(if i < 20 { 400.0 } else { 2_000.0 });
            }
        }
        let before = values.clone();
        let changed = smooth_over_high_terrain(&mut values, &terrain, nx, ny, 30);
        assert_eq!(changed, 20 * ny);
        for j in 0..ny {
            for i in 0..20 {
                assert_eq!(values[j * nx + i].to_bits(), before[j * nx + i].to_bits());
            }
        }
        // the plateau's checkerboard is gone: neighbours within 1 Pa of each other
        // away from the edge, where the gradient is 10 Pa per point
        for j in 5..ny - 5 {
            for i in 25..nx - 5 {
                let d = (values[j * nx + i] - values[j * nx + i + 1]).abs();
                assert!(d < 20.0, "{j} {i} {d}");
            }
        }
    }

    #[test]
    fn a_frame_without_high_ground_is_untouched() {
        let (nx, ny) = (10, 10);
        let mut values: Vec<f32> = (0..100).map(|k| 100_000.0 + k as f32).collect();
        let before = values.clone();
        let terrain = vec![499.0; 100];
        assert_eq!(smooth_over_high_terrain(&mut values, &terrain, nx, ny, 50), 0);
        assert_eq!(values, before);
    }

    #[test]
    fn missing_points_stay_missing_and_do_not_leak() {
        let (nx, ny) = (12, 12);
        let mut values = vec![101_000.0f32; nx * ny];
        for j in 0..ny {
            values[j * nx] = f32::NAN;
        }
        let terrain = vec![3_000.0; nx * ny];
        smooth_over_high_terrain(&mut values, &terrain, nx, ny, 20);
        for j in 0..ny {
            assert!(values[j * nx].is_nan());
            for i in 1..nx {
                // a uniform field stays uniform: missing points are renormalised away
                assert!((values[j * nx + i] - 101_000.0).abs() < 0.01, "{}", values[j * nx + i]);
            }
        }
    }

    #[test]
    fn the_box_gives_what_the_whole_grid_gives() {
        // high ground in a corner of a larger grid: the windowed computation
        // must equal a whole-grid one at every weighted point
        let (nx, ny) = (60, 50);
        let mut values = Vec::new();
        let mut terrain = Vec::new();
        for j in 0..ny {
            for i in 0..nx {
                values.push(100_000.0 + ((i * 7 + j * 13) % 17) as f32 * 50.0);
                terrain.push(if i > 40 && j > 30 { 2_500.0 } else { 0.0 });
            }
        }
        let mut windowed = values.clone();
        smooth_over_high_terrain(&mut windowed, &terrain, nx, ny, 5);
        // whole grid by hand
        let mut s: Vec<f64> = values.iter().map(|&v| v as f64).collect();
        let mut m = vec![1.0f64; nx * ny];
        let (mut ts, mut tm) = (vec![0.0; nx * ny], vec![0.0; nx * ny]);
        for _ in 0..5 {
            one_two_one_x(&s, &mut ts, nx);
            one_two_one_x(&m, &mut tm, nx);
            one_two_one_y(&ts, &mut s, nx, ny);
            one_two_one_y(&tm, &mut m, nx, ny);
        }
        for k in 0..nx * ny {
            if terrain[k] > 0.0 {
                let expected = (s[k] / m[k]) as f32;
                assert!((windowed[k] - expected).abs() < 1e-3, "{k}");
            } else {
                assert_eq!(windowed[k], values[k]);
            }
        }
    }

    #[test]
    fn passes_follow_the_physical_length() {
        assert_eq!(passes_for(40_000.0, 3_000.0), 356);
        assert_eq!(passes_for(40_000.0, 13_000.0), 19);
        assert_eq!(passes_for(40_000.0, 1_000.0), MAX_PASSES);
        assert_eq!(passes_for(0.0, 3_000.0), 0);
        assert_eq!(terrain_weight(500.0), 0.0);
        assert_eq!(terrain_weight(1_000.0), 0.5);
        assert_eq!(terrain_weight(2_000.0), 1.0);
        assert_eq!(terrain_weight(f32::NAN), 0.0);
    }
}
