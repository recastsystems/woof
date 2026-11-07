//! Surface diagnostics that more than one writer must compute the same way: the map renderer (rw-wrfbatch, its
//! `apcp` and 10 m gust planes) and the any-point column store's surface section (rw-soundings `--columns`), so a
//! value a point forecast reads from the store is the value the map draws at that point.
//!
//! Moved here from rw-wrfbatch's `wrf_process.rs` (`clean_values`, `total_precip`, `push_gust_10m`) without a change
//! of arithmetic; the renderer now calls these.

/// A history value as the renderer keeps it: NaN for a non-finite value, WRF's fill words (|v| >= 1e30) and the
/// -9999-style missing markers (v <= -9998), else the value as f32.
#[inline]
pub fn clean_value(value: f64) -> f32 {
    if !value.is_finite() || value.abs() >= 1.0e30 || value <= -9998.0 {
        f32::NAN
    } else {
        value as f32
    }
}

/// Every value of a plane through [`clean_value`].
pub fn clean_values(values: &[f64]) -> Vec<f32> {
    values.iter().map(|v| clean_value(*v)).collect()
}

/// The run's total precipitation (kg m-2 = mm) from its accumulation planes, summed in the order given in f32 (the
/// renderer's `apcp`: RAINC, RAINNC, RAINSH, each the planes that exist): a point where any plane is not finite is
/// NaN. `None` when no plane is given.
pub fn total_precip(planes: &[&[f64]], cells: usize) -> Option<Vec<f32>> {
    if planes.is_empty() {
        return None;
    }
    let mut total = vec![0.0f32; cells];
    for plane in planes {
        for (accum, value) in total.iter_mut().zip(plane.iter().map(|v| clean_value(*v))) {
            if value.is_finite() {
                *accum += value;
            } else {
                *accum = f32::NAN;
            }
        }
    }
    Some(total)
}

/// 10 m wind gust: the wind at the top of the boundary layer mixed down, damped as the layer deepens:
/// gust = w10 + (w_pbltop - w10) * (1 - min(0.5, PBLH / 2000 m)), never below w10, where PBLH is the MODEL's own PBL
/// height (the history's PBLH) and w_pbltop is the wind speed on the first mass level at or above it. wrf-python
/// defines no gust, so this is the definition the renderer uses for every model's files.
///
/// It is the single-level form of the NCEP post-processor's gust, NOT the form NCEP applies to RAP/HRRR (UPP diagnoses
/// its own PBL depth and takes the maximum over the levels up to it); see rw-wrfbatch's `push_gust_10m` for the
/// measured comparison. At a run's initial time PBLH is 0 until the PBL scheme has run (as in WRF), so the gust is
/// the 10 m wind.
///
/// `height_agl` and `wspd` are [nz][cells] (mass levels, lowest first), `pblh` and `wspd10` [cells]. A point whose
/// PBLH, 10 m wind or boundary-layer-top wind is not finite is NaN.
pub fn gust_10m(height_agl: &[f64], wspd: &[f64], pblh: &[f64], wspd10: &[f64], nz: usize, cells: usize) -> Vec<f32> {
    let mut values = vec![f32::NAN; cells];
    for (i, value) in values.iter_mut().enumerate() {
        let zpbl = pblh[i];
        let wind10 = wspd10[i];
        if !zpbl.is_finite() || !wind10.is_finite() {
            continue;
        }
        let mut k = 0;
        while k + 1 < nz && height_agl[k * cells + i] < zpbl {
            k += 1;
        }
        let top = wspd[k * cells + i];
        if !top.is_finite() {
            continue;
        }
        let damp = 1.0 - (zpbl.max(0.0) / 2000.0).min(0.5);
        *value = (wind10 + (top - wind10) * damp).max(wind10) as f32;
    }
    values
}

/// The physical range a reflectivity value must lie in (dBZ): WRF and third-party NetCDF writers use several
/// large-magnitude fill conventions (-999, -8888, -9999, +9999), rejected before a vertical maximum or a linear-Z
/// interpolation can turn them into a false echo.
pub const MIN_VALID_REFLECTIVITY_DBZ: f64 = -100.0;
pub const MAX_VALID_REFLECTIVITY_DBZ: f64 = 120.0;

#[inline]
pub fn valid_reflectivity_dbz(value: f64) -> bool {
    value.is_finite() && (MIN_VALID_REFLECTIVITY_DBZ..=MAX_VALID_REFLECTIVITY_DBZ).contains(&value)
}

/// Composite reflectivity: the largest valid value of each column of a [nz][cells] dBZ volume, NaN where no level
/// holds one (the renderer's composite map from REFL_10CM).
pub fn composite_reflectivity_from_dbz(dbz_3d: &[f64], nz: usize, cells: usize) -> Result<Vec<f64>, String> {
    let expected = nz
        .checked_mul(cells)
        .ok_or_else(|| "reflectivity volume length overflows the platform address space".to_string())?;
    if dbz_3d.len() != expected {
        return Err(format!(
            "reflectivity volume requires {expected} values for {nz} levels and {cells} columns, got {}",
            dbz_3d.len()
        ));
    }
    let mut composite = vec![f64::NAN; cells];
    for k in 0..nz {
        let offset = k * cells;
        for ij in 0..cells {
            let value = dbz_3d[offset + ij];
            if valid_reflectivity_dbz(value) && (!composite[ij].is_finite() || value > composite[ij]) {
                composite[ij] = value;
            }
        }
    }
    Ok(composite)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn precip_sums_in_order_and_a_missing_plane_value_is_nan() {
        let a = [1.0, 2.0, f64::NAN];
        let b = [0.5, 1e31, 3.0];
        let t = total_precip(&[&a, &b], 3).unwrap();
        assert_eq!(t[0], 1.5);
        assert!(t[1].is_nan());
        assert!(t[2].is_nan());
        assert!(total_precip(&[], 3).is_none());
    }

    #[test]
    fn gust_mixes_down_the_pbl_top_wind_and_never_falls_below_the_10m_wind() {
        // two columns, three levels at 50, 500, 1500 m
        let cells = 2;
        let hagl = [50.0, 50.0, 500.0, 500.0, 1500.0, 1500.0];
        let wspd = [5.0, 5.0, 10.0, 2.0, 20.0, 1.0];
        let g = gust_10m(&hagl, &wspd, &[400.0, 400.0], &[4.0, 4.0], 3, cells);
        // first level at or above 400 m is 500 m: 4 + (10 - 4) * (1 - 0.2) = 8.8
        assert!((g[0] - 8.8).abs() < 1e-5);
        // the top wind is weaker than the 10 m wind: the gust is the 10 m wind
        assert_eq!(g[1], 4.0);
    }
}
