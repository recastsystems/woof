//! Soil layers and the surface companions, from `mpas_atmphys_initialize_real.F`.
//!
//! Everything here is counts, not flags: the first-guess sanity check, the
//! dry-land floor, the water fill and the unbracketed-layer count all
//! reach the receipt as numbers a reader can weigh.
//!
//! Native cold-start fields are zero-initialized by the framework. An
//! unbracketed NOAH layer retains those words, as does landmask outside 0/1.
//! Sources: v8.4.2 Registry.xml:1298-1312, tools/registry/gen_inc.c:615-622,
//! framework/mpas_field_routines.F:538-575 and physics/
//! mpas_atmphys_initialize_real.F:453-488. There is no undefined local soil
//! output array to repair or refuse.
//!
//! The deep-moisture comparison arm reproduces sm_input(nFGSoilLevels),
//! the native second-deepest first-guess anchor. The pre-existing corrected
//! arm remains available; this lane does not change its selection default.

use crate::error::{MpasError, MpasResult};

/// NOAH's four layer thicknesses, in metres.  MPAS refuses any other count.
pub const NOAH_LAYER_THICKNESS: [f32; 4] = [0.10, 0.30, 0.60, 1.00];

/// WRF real.exe, module_initialize_real.F:3108-3131. Soil and vegetation
/// maps are independent at shorelines. Keep the land-use classification and
/// give a land cell with water soil the defined silty-clay-loam category.
pub fn reconcile_soil_category(xland: &[f32], soil: &mut [i32]) -> usize {
    assert_eq!(xland.len(), soil.len());
    let mut changed = 0;
    for (&surface, category) in xland.iter().zip(soil) {
        let wanted = if surface >= 1.5 { 14 } else if *category == 14 { 8 } else { *category };
        if *category != wanted { *category = wanted; changed += 1; }
    }
    changed
}

#[cfg(test)]
mod category_tests {
    #[test]
    fn wrf_shoreline_categories_are_consistent_and_idempotent() {
        let xland = [1., 2., 1., 1., 2.];
        let mut soil = [14, 6, 8, 16, 14];
        assert_eq!(super::reconcile_soil_category(&xland, &mut soil), 2);
        assert_eq!(soil, [8, 14, 8, 16, 14]);
        assert_eq!(super::reconcile_soil_category(&xland, &mut soil), 0);
    }
}

/// Which first-guess level seeds the bottom of the augmented soil column.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeepMoisture {
    /// `sm_input(nFGSoilLevels)`: the Fortran's own index, which lands on the
    /// second-deepest first-guess level.  The comparison arm.
    ReproduceFortran,
    /// The deepest first-guess level, which is what the line reads as if it
    /// meant.  Opt-in, with its own receipt entry.
    Corrected,
}

/// The numbers a reader needs to know whether to trust the soil column.
#[derive(Debug, Default, Clone, serde::Serialize)]
pub struct SoilCounts {
    /// Land cells whose first-guess soil temperature was at or below zero.
    /// Non-zero is fatal in the Fortran and fatal here.
    pub bad_first_guess_temperature: usize,
    /// Land cells whose first-guess soil moisture was negative.  Fatal.
    pub bad_first_guess_moisture: usize,
    /// Water cells filled with `smois = 1.0`.  A fill, not a measurement.
    pub water_cells_filled: usize,
    /// Land cells raised to the 0.005 dry floor.  A large count is a
    /// decode-side finding, not a soil-side one.
    pub dry_floor_applied: usize,
    /// NOAH layers no first-guess pair bracketed; native cold words retained.
    pub unbracketed_layers: usize,
    /// Cells where the reproduced and corrected deep-moisture choices differ
    /// by more than [`DEEP_MOISTURE_REPORTING_THRESHOLD`].
    pub deep_moisture_choice_differs: usize,
    /// The largest such difference, in m3/m3.
    pub deep_moisture_max_difference: f32,
    /// Land cells whose non-positive first-guess soil moisture was raised to
    /// 0.001 by the consistency pass before the abort check ran.  This is a
    /// repair the reference performs too; a large count is a decode-side
    /// finding about the first-guess soil field, not a soil-side one.
    pub land_moisture_repaired: usize,
    /// Land cells whose first-guess soil temperature was non-positive at the
    /// consistency pass.  The reference logs these and does not repair them,
    /// so they go on to abort the run.
    pub land_temperature_flagged: usize,
    /// NOAH layers whose bracketing pair actually reached the 3.0 m bottom
    /// anchor, and therefore actually saw the deep-moisture choice.
    ///
    /// This is the number that decides whether the off-by-one matters at all
    /// on a given first-guess column.  When the deepest first-guess mid-depth
    /// is at or below 1.50 m (which is the deepest NOAH mid-depth) no layer
    /// ever brackets against the anchor and the quirk is unreachable, however
    /// large `deep_moisture_max_difference` looks.
    pub deep_moisture_anchor_reached: usize,
    /// RUC from a layer source: values raised to real.exe's 0.005 floor
    /// (`module_initialize_real.F:3441`), the columns holding them, and the
    /// smallest value before the floor (null when none).  The extrapolated
    /// 0 m anchor off a dry top layer is what goes below.
    pub ruc_layer_floor_values: usize,
    pub ruc_layer_floor_columns: usize,
    pub ruc_layer_floor_min: Option<f32>,
}

/// Differences below this are round-off in the first-guess field itself and
/// are not counted; above it, the off-by-one moved a number a reader would see.
pub const DEEP_MOISTURE_REPORTING_THRESHOLD: f32 = 1.0e-4;

/// The floor a non-positive land soil moisture is raised to.
pub const LAND_MOISTURE_FLOOR: f32 = 0.001;

/// The consistency pass `init_atm_case_gfs` runs before it hands the soil
/// column to the physics: the block its own author labelled `MGD CHECK`.
///
/// It is asymmetric and the asymmetry is deliberate on both sides:
///
/// * a **land** cell with `sm_fg <= 0` is repaired to
///   [`LAND_MOISTURE_FLOOR`] and logged;
/// * a land cell with `st_fg <= 0` is logged and **not** repaired, so it
///   reaches `init_soil_layers_properties` and aborts the run there;
/// * a **water** cell is not touched at all, and a negative moisture on one
///   still aborts, because the abort counts every cell.
///
/// Returning counts rather than logging them is the one change: the numbers
/// reach the receipt, where a large repair count is legible as the decode-side
/// finding it is.  Reproducing the repair itself is required, without it the
/// run this port is compared against would not have started.
/// MPAS v8.4.2 mpas_init_atm_cases.F:4780-4800.
pub fn first_guess_consistency_pass(
    landmask: &[i32],
    st_fg: &[Vec<f32>],
    sm_fg: &mut [Vec<f32>],
) -> (usize, usize) {
    let mut bad_temperature = 0usize;
    let mut repaired_moisture = 0usize;
    for c in 0..landmask.len() {
        if landmask[c] != 1 {
            continue;
        }
        for v in &st_fg[c] {
            if *v <= 0.0 {
                bad_temperature += 1;
            }
        }
        for v in sm_fg[c].iter_mut() {
            if *v <= 0.0 {
                repaired_moisture += 1;
                *v = LAND_MOISTURE_FLOOR;
            }
        }
    }
    (bad_temperature, repaired_moisture)
}

/// `adjust_input_soiltemps`: move the deep and skin temperatures, and every
/// first-guess soil level, onto the model's own terrain.
/// MPAS v8.4.2 mpas_atmphys_initialize_real.F:217-270.
pub fn adjust_input_soiltemps(
    landmask: &[i32],
    ter: &[f32],
    soiltemp: &[f32],
    soilz: &[f32],
    skintemp: &mut [f32],
    st_fg: &mut [Vec<f32>],
) -> Vec<f32> {
    let n = landmask.len();
    let mut tmn = vec![0.0f32; n];
    for c in 0..n {
        if landmask[c] == 1 {
            tmn[c] = soiltemp[c] - 0.0065 * ter[c];
            let shift = 0.0065 * (ter[c] - soilz[c]);
            skintemp[c] -= shift;
            for v in st_fg[c].iter_mut() {
                *v -= shift;
            }
        } else if landmask[c] == 0 {
            tmn[c] = skintemp[c];
        }
    }
    tmn
}

/// Cumulative mid-depths, in centimetres, from the layer thicknesses.
/// MPAS v8.4.2 mpas_atmphys_initialize_real.F:273-336.
pub fn first_guess_mid_depths_cm(dzs_fg_cm: &[f32]) -> Vec<f32> {
    let mut zs = vec![0.0f32; dzs_fg_cm.len()];
    if zs.is_empty() {
        return zs;
    }
    zs[0] = 0.5 * dzs_fg_cm[0];
    for i in 1..zs.len() {
        zs[i] = zs[i - 1] + 0.5 * dzs_fg_cm[i - 1] + 0.5 * dzs_fg_cm[i];
    }
    zs
}

/// NOAH mid-depths, in metres.
///
/// They come out **0.05, 0.25, 0.70, 1.50**, not the 0.05/0.30/0.70/1.50 the
/// design note wrote down: `zs(2) = zs(1) + 0.5*dzs(1) + 0.5*dzs(2)` is
/// `0.05 + 0.05 + 0.15 = 0.25`.  The note's 0.30 was the layer *interface*, not
/// its midpoint.  This is checked by test rather than asserted in prose
/// because the second NOAH layer's temperature and moisture both hang off it.
/// MPAS v8.4.2 mpas_atmphys_initialize_real.F:273-336.
pub fn noah_mid_depths() -> [f32; 4] {
    let mut zs = [0.0f32; 4];
    zs[0] = 0.5 * NOAH_LAYER_THICKNESS[0];
    for i in 1..4 {
        zs[i] = zs[i - 1] + 0.5 * NOAH_LAYER_THICKNESS[i - 1] + 0.5 * NOAH_LAYER_THICKNESS[i];
    }
    zs
}

/// What [`init_soil_layers_properties`] returns: `tslb`, `smois`, `sh2o` and
/// `smcrel` on NOAH's four layers, one row per cell, plus the counts.
pub type SoilLayers = (
    Vec<[f32; 4]>,
    Vec<[f32; 4]>,
    Vec<[f32; 4]>,
    Vec<[f32; 4]>,
    SoilCounts,
);

/// Interpolate the first-guess soil column onto NOAH's four layers.
#[allow(clippy::too_many_arguments)]
/// MPAS v8.4.2 mpas_atmphys_initialize_real.F:339-513.
pub fn init_soil_layers_properties(
    landmask: &[i32],
    skintemp: &[f32],
    tmn: &[f32],
    st_fg: &[Vec<f32>],
    sm_fg: &[Vec<f32>],
    dzs_fg_cm: &[f32],
    deep_moisture: DeepMoisture,
) -> MpasResult<SoilLayers> {
    let n_cells = landmask.len();
    let n_fg = dzs_fg_cm.len();
    let mut counts = SoilCounts::default();

    // The refusal that runs before anything else.
    for c in 0..n_cells {
        for k in 0..n_fg {
            if st_fg[c][k] <= 0.0 {
                counts.bad_first_guess_temperature += 1;
            }
            if sm_fg[c][k] < 0.0 {
                counts.bad_first_guess_moisture += 1;
            }
        }
    }
    if counts.bad_first_guess_temperature > 0 {
        return Err(MpasError::Refusal(format!(
            "interpolation of the first-guess soil temperature left {} value(s) at or below \
             zero kelvin; init_soil_layers_properties aborts on this count and so does this port",
            counts.bad_first_guess_temperature
        )));
    }
    if counts.bad_first_guess_moisture > 0 {
        return Err(MpasError::Refusal(format!(
            "interpolation of the first-guess soil moisture left {} negative value(s); \
             init_soil_layers_properties aborts on this count and so does this port",
            counts.bad_first_guess_moisture
        )));
    }

    let zs_fg_cm = first_guess_mid_depths_cm(dzs_fg_cm);
    let zs = noah_mid_depths();

    let mut tslb = vec![[0.0f32; 4]; n_cells];
    let mut smois = vec![[0.0f32; 4]; n_cells];
    let mut sh2o = vec![[0.0f32; 4]; n_cells];
    let mut smcrel = vec![[0.0f32; 4]; n_cells];

    for c in 0..n_cells {
        if landmask[c] == 0 {
            counts.water_cells_filled += 1;
            for l in 0..4 {
                tslb[c][l] = skintemp[c];
                smois[c][l] = 1.0;
                sh2o[c][l] = 1.0;
                smcrel[c][l] = 0.0;
            }
            continue;
        }

        // MPAS v8.4.2 mpas_atmphys_initialize_real.F:453-495 handles
        // only landmask 1 and 0; other values retain the cold-start arrays.
        if landmask[c] != 1 {continue;}

        // The augmented column: skin at depth zero, the first-guess levels at
        // their mid-depths in metres, and tmn at 3.0 m.
        let m = n_fg + 2;
        let mut zhave = vec![0.0f32; m];
        let mut st_input = vec![0.0f32; m];
        let mut sm_input = vec![0.0f32; m];
        zhave[0] = 0.0;
        st_input[0] = skintemp[c];
        sm_input[0] = sm_fg[c][1.min(n_fg - 1)];
        for k in 0..n_fg {
            zhave[k + 1] = zs_fg_cm[k] / 100.0;
            st_input[k + 1] = st_fg[c][k];
            sm_input[k + 1] = sm_fg[c][k];
        }
        zhave[m - 1] = 3.0;
        st_input[m - 1] = tmn[c];

        // The quirk.  `sm_input(nFGSoilLevels)` in the Fortran is one-based, so
        // it is `sm_input[n_fg - 1]` here, and `sm_input` is offset one from
        // `sm_fg`: that slot holds `sm_fg[n_fg - 2]`, the second-deepest level.
        let reproduced = sm_input[n_fg - 1];
        let corrected = sm_input[m - 2];
        let difference = (reproduced - corrected).abs();
        if difference > DEEP_MOISTURE_REPORTING_THRESHOLD {
            counts.deep_moisture_choice_differs += 1;
            if difference > counts.deep_moisture_max_difference {
                counts.deep_moisture_max_difference = difference;
            }
        }
        sm_input[m - 1] = match deep_moisture {
            DeepMoisture::ReproduceFortran => reproduced,
            DeepMoisture::Corrected => corrected,
        };

        for l in 0..4 {
            let mut assigned = false;
            for k in 0..m - 1 {
                if zs[l] >= zhave[k] && zs[l] <= zhave[k + 1] {
                    let span = zhave[k + 1] - zhave[k];
                    tslb[c][l] = (st_input[k] * (zhave[k + 1] - zs[l])
                        + st_input[k + 1] * (zs[l] - zhave[k]))
                        / span;
                    smois[c][l] = (sm_input[k] * (zhave[k + 1] - zs[l])
                        + sm_input[k + 1] * (zs[l] - zhave[k]))
                        / span;
                    sh2o[c][l] = 0.0;
                    smcrel[c][l] = 0.0;
                    if k + 1 == m - 1 {
                        counts.deep_moisture_anchor_reached += 1;
                    }
                    assigned = true;
                    break;
                }
            }
            if !assigned {
                counts.unbracketed_layers += 1;
            }
        }
    }

    // The dry-land floor.
    for c in 0..n_cells {
        if landmask[c] == 1 && tslb[c][0] > 170.0 && tslb[c][0] < 400.0 && smois[c][0] < 0.005 {
            counts.dry_floor_applied += 1;
            for l in 0..4 {
                smois[c][l] = 0.005;
            }
        }
    }

    Ok((tslb, smois, sh2o, smcrel, counts))
}

/// The soil depth the sea-ice profile is drawn over, in metres.
pub const SEAICE_TOTAL_DEPTH: f32 = 3.0;

/// The deep temperature a converted water point is given, in kelvin.
pub const SEAICE_TMN: f32 = 271.4;

/// What the sea-ice initialisation changed.
#[derive(Debug, Default, Clone, serde::Serialize)]
pub struct SeaiceCounts {
    /// Cells converted from water to land because they carry enough sea ice.
    pub converted_to_land: usize,
    /// Water cells whose sea-ice fraction was below the threshold and was
    /// zeroed, taking their snow with it.
    pub subthreshold_ice_zeroed: usize,
    /// Water cells whose snow water equivalent was discarded with it.  A
    /// large count here is the reason a raw first-guess SNOW field must not
    /// be read as the model's snow.
    pub snow_cleared_over_water: usize,
    pub xice_threshold: f32,
}

/// `physics_init_seaice`: convert sea-ice points to land points, and clear
/// sub-threshold ice and its snow off the water points that remain.
///
/// This runs *after* the soil interpolation and overwrites its results at the
/// converted points, which is why leaving it out shows up as a soil defect
/// rather than a sea-ice one: `tslb`, `smois`, `sh2o`, `tmn` and `xland` all
/// come out of the soil step correct and are then legitimately rewritten here.
#[allow(clippy::too_many_arguments)]
/// MPAS v8.4.2 mpas_atmphys_initialize_real.F:586-729.
pub fn physics_init_seaice<T: std::ops::IndexMut<usize, Output=f32> + AsRef<[f32]>>(
    frac_seaice: bool,
    tsk_seaice_threshold: f32,
    isice_lu: i32,
    landmask: &[i32],
    skintemp: &[f32],
    xice: &mut [f32],
    seaice: &mut [f32],
    snow: &mut [f32],
    snowc: &mut [f32],
    snowh: &mut [f32],
    vegfra: &mut [f32],
    snoalb: &mut [f32],
    ivgtyp: &mut [i32],
    isltyp: &mut [i32],
    xland: &mut [f32],
    tmn: &mut [f32],
    tslb: &mut [T],
    smois: &mut [T],
    sh2o: &mut [T],
    smcrel: &mut [T],
) -> SeaiceCounts {
    let mut counts = SeaiceCounts::default();
    let threshold = if frac_seaice {
        0.02
    } else {
        // Without fractional sea ice the field is first collapsed to 0/1.
        for v in xice.iter_mut() {
            *v = if *v >= 0.5 { 1.0 } else { 0.0 };
        }
        0.5
    };
    counts.xice_threshold = threshold;
    let n_soil = tslb.first().map(|v| v.as_ref().len()).unwrap_or(4);

    for c in 0..landmask.len() {
        if xice[c] >= threshold
            || (landmask[c] == 0 && skintemp[c] < tsk_seaice_threshold)
        {
            counts.converted_to_land += 1;
            if landmask[c] == 0 {
                tmn[c] = SEAICE_TMN;
            }
            ivgtyp[c] = isice_lu;
            isltyp[c] = 16;
            snoalb[c] = 0.75;
            vegfra[c] = 0.0;
            xland[c] = 1.0;
            for s in 0..n_soil {
                let mid = SEAICE_TOTAL_DEPTH / n_soil as f32 / 2.0
                    + (s as f32) * (SEAICE_TOTAL_DEPTH / n_soil as f32);
                tslb[c][s] = ((SEAICE_TOTAL_DEPTH - mid) * skintemp[c] + mid * tmn[c])
                    / SEAICE_TOTAL_DEPTH;
                smois[c][s] = 1.0;
                sh2o[c][s] = 0.0;
                smcrel[c][s] = 0.0;
            }
        } else {
            if xice[c] > 0.0 {
                counts.subthreshold_ice_zeroed += 1;
            }
            xice[c] = 0.0;
            if landmask[c] == 0 {
                if snow[c] > 0.0 {
                    counts.snow_cleared_over_water += 1;
                }
                snowc[c] = 0.0;
                snowh[c] = 0.0;
                snow[c] = 0.0;
            }
        }
    }

    for c in 0..landmask.len() {
        seaice[c] = if xice[c] > 0.0 { 1.0 } else { 0.0 };
    }
    counts
}

/// Surface companions at the real-physics driver boundary.
pub struct SurfaceState {
    pub albedo:Vec<f32>,pub vegetation:Vec<f32>,pub minimum:Vec<f32>,pub maximum:Vec<f32>,
    pub snow_cover:Vec<f32>,pub snow_height:Vec<f32>,pub land:Vec<f32>,
}
/// MPAS v8.4.2 physics/mpas_atmphys_initialize_real.F:131-178.
pub fn init_surface_companions(land:&[i32],snow:&[f32],albedo:&[Vec<f32>],green:&[Vec<f32>],
    year:i32,month:u32,day:u32)->SurfaceState {
    let mut background=monthly_interp_to_date(albedo,year,month,day);
    for (i,v) in background.iter_mut().enumerate() {*v= *v/100.;if land[i]==0 {*v=0.08;}}
    let vegetation=monthly_interp_to_date(green,year,month,day);
    let (minimum,maximum)=monthly_min_max(green);
    SurfaceState {albedo:background,vegetation,minimum,maximum,
        snow_cover:snow.iter().map(|&v|if v>=10. {1.} else {0.}).collect(),
        snow_height:snow.iter().map(|&v|v*5./1000.).collect(),
        land:land.iter().map(|&v|if v==1 {1.} else if v==0 {2.} else {0.}).collect()}
}

/// Interpolate a twelve-month field to the start date, the way
/// `monthly_interp_to_date` does: mid-month anchors on day 15, linear in the
/// Julian day, wrapping through December and January.
/// MPAS v8.4.2 physics/mpas_atmphys_date_time.F:50-83,125-219.
pub fn monthly_interp_to_date(monthly: &[Vec<f32>], year: i32, month: u32, day: u32) -> Vec<f32> {
    let leap = (year % 4 == 0 && year % 100 != 0) || year % 400 == 0;
    monthly_interp_with_february(monthly, year, month, day, leap)
}

/// MPAS: src/core_atmosphere/physics/mpas_atmphys_date_time.F:51-84,148-219.
/// Aerosol LBC replay retains the native saved mmd(2)=29 after a leap-year call.
/// Other existing callers retain their calendar contract.
pub(crate) fn monthly_interp_with_february(
    monthly: &[Vec<f32>],
    year: i32,
    month: u32,
    day: u32,
    leap: bool,
) -> Vec<f32> {
    let n = monthly[0].len();
    let mmd = [
        31,
        if leap { 29 } else { 28 },
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ];
    let julian = |m: u32, d: u32| -> i32 {
        let mut j = d as i32;
        for i in 0..(m as usize - 1) {
            j += mmd[i];
        }
        j
    };
    let mut middle = [0i32; 14];
    for l in 1..=12u32 {
        middle[l as usize] = year * 1000 + julian(l, 15);
    }
    middle[0] = middle[1] - 31;
    middle[13] = middle[12] + 31;

    let target = year * 1000 + julian(month, day);
    let mut out = vec![0.0f32; n];
    for l in 0..=12usize {
        if middle[l] < target && middle[l + 1] >= target {
            let (m1, m2) = if l == 0 || l == 12 { (12, 1) } else { (l, l + 1) };
            // Element by element, on every core (pure per element).
            use rayon::prelude::*;
            out.par_iter_mut().enumerate().for_each(|(i, slot)| {
                *slot = (monthly[m2 - 1][i] * (target - middle[l]) as f32
                    + monthly[m1 - 1][i] * (middle[l + 1] - target) as f32)
                    / (middle[l + 1] - middle[l]) as f32;
            });
            break;
        }
    }
    out
}

/// Annual minimum and maximum of a twelve-month field.
/// MPAS v8.4.2 physics/mpas_atmphys_date_time.F:222-255.
pub fn monthly_min_max(monthly: &[Vec<f32>]) -> (Vec<f32>, Vec<f32>) {
    let n = monthly[0].len();
    let mut lo = monthly[0].clone();
    let mut hi = monthly[0].clone();
    for m in 1..12 {
        for i in 0..n {
            if monthly[m][i] < lo[i] {
                lo[i] = monthly[m][i];
            }
            if monthly[m][i] > hi[i] {
                hi[i] = monthly[m][i];
            }
        }
    }
    (lo, hi)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noah_mid_depths_are_five_twentyfive_seventy_and_hundredfifty_centimetres() {
        let zs = noah_mid_depths();
        assert!((zs[0] - 0.05).abs() < 1e-6);
        assert!((zs[1] - 0.25).abs() < 1e-6, "{:?}", zs);
        assert!((zs[2] - 0.70).abs() < 1e-6);
        assert!((zs[3] - 1.50).abs() < 1e-6);
    }

    #[test]
    fn first_guess_mid_depths_are_cumulative_from_the_thicknesses() {
        // GFS: 0-10, 10-40, 40-100, 100-200 cm.
        let zs = first_guess_mid_depths_cm(&[10.0, 30.0, 60.0, 100.0]);
        assert_eq!(zs, vec![5.0, 25.0, 70.0, 150.0]);
    }

    #[test]
    fn a_water_cell_is_filled_and_counted_not_measured() {
        let landmask = vec![0i32];
        let (tslb, smois, sh2o, _, counts) = init_soil_layers_properties(
            &landmask,
            &[271.5],
            &[271.5],
            &[vec![280.0, 281.0, 282.0, 283.0]],
            &[vec![0.2, 0.2, 0.2, 0.2]],
            &[10.0, 30.0, 60.0, 100.0],
            DeepMoisture::ReproduceFortran,
        )
        .unwrap();
        assert_eq!(counts.water_cells_filled, 1);
        assert_eq!(tslb[0], [271.5; 4]);
        assert_eq!(smois[0], [1.0; 4]);
        assert_eq!(sh2o[0], [1.0; 4]);
    }

    #[test]
    fn a_land_cell_interpolates_in_depth_and_zeroes_the_liquid_fields() {
        let (tslb, smois, sh2o, smcrel, counts) = init_soil_layers_properties(
            &[1i32],
            &[290.0],
            &[285.0],
            &[vec![288.0, 287.0, 286.0, 285.5]],
            &[vec![0.30, 0.28, 0.26, 0.24]],
            &[10.0, 30.0, 60.0, 100.0],
            DeepMoisture::ReproduceFortran,
        )
        .unwrap();
        assert_eq!(counts.unbracketed_layers, 0);
        // Layer 1 mid-depth 0.05 m sits exactly on the first first-guess
        // mid-depth, so it takes that level's value.
        assert!((tslb[0][0] - 288.0).abs() < 1e-4, "{:?}", tslb[0]);
        assert!(smois[0].iter().all(|&v| v > 0.0));
        assert_eq!(sh2o[0], [0.0; 4]);
        assert_eq!(smcrel[0], [0.0; 4]);
    }

    #[test]
    fn the_dry_land_floor_is_applied_and_counted() {
        let (_, smois, _, _, counts) = init_soil_layers_properties(
            &[1i32],
            &[290.0],
            &[285.0],
            &[vec![288.0, 287.0, 286.0, 285.5]],
            &[vec![0.0, 0.0, 0.0, 0.0]],
            &[10.0, 30.0, 60.0, 100.0],
            DeepMoisture::ReproduceFortran,
        )
        .unwrap();
        assert_eq!(counts.dry_floor_applied, 1);
        assert_eq!(smois[0], [0.005; 4]);
    }

    #[test]
    fn the_deep_moisture_quirk_is_counted_and_switchable() {
        // A column whose two deepest levels differ, and which is SHALLOW
        // enough that the 3.0 m anchor is what brackets the deepest NOAH
        // layer.  On a column reaching 1.50 m or deeper the anchor is never
        // consulted and the quirk cannot move a number: see the test below.
        let st = vec![vec![288.0f32, 287.0, 286.0, 285.5]];
        let sm = vec![vec![0.30f32, 0.28, 0.26, 0.05]];
        let dz = vec![10.0f32, 20.0, 20.0, 20.0];
        let (_, a, _, _, counts) = init_soil_layers_properties(
            &[1i32],
            &[290.0],
            &[285.0],
            &st,
            &sm,
            &dz,
            DeepMoisture::ReproduceFortran,
        )
        .unwrap();
        let (_, b, _, _, _) = init_soil_layers_properties(
            &[1i32],
            &[290.0],
            &[285.0],
            &st,
            &sm,
            &dz,
            DeepMoisture::Corrected,
        )
        .unwrap();
        assert_eq!(counts.deep_moisture_choice_differs, 1);
        assert!(counts.deep_moisture_max_difference > 0.2);
        assert!(counts.deep_moisture_anchor_reached > 0);
        // The deepest NOAH layer is the one that sees the bottom anchor.
        assert!((a[0][3] - b[0][3]).abs() > 1e-3, "{:?} {:?}", a[0], b[0]);
    }

    #[test]
    fn on_a_column_reaching_one_and_a_half_metres_the_quirk_cannot_move_a_number() {
        // GFS's own layering: 0-10, 10-40, 40-100, 100-200 cm, whose deepest
        // mid-depth is exactly 1.50 m, which is exactly the deepest NOAH
        // mid-depth.  Every NOAH layer therefore brackets between two
        // first-guess levels and the 3.0 m anchor is never reached, so the
        // off-by-one is unreachable however different the two candidates are.
        let st = vec![vec![288.0f32, 287.0, 286.0, 285.5]];
        let sm = vec![vec![0.30f32, 0.28, 0.26, 0.05]];
        let dz = vec![10.0f32, 30.0, 60.0, 100.0];
        let (_, a, _, _, counts) = init_soil_layers_properties(
            &[1i32],
            &[290.0],
            &[285.0],
            &st,
            &sm,
            &dz,
            DeepMoisture::ReproduceFortran,
        )
        .unwrap();
        let (_, b, _, _, _) = init_soil_layers_properties(
            &[1i32],
            &[290.0],
            &[285.0],
            &st,
            &sm,
            &dz,
            DeepMoisture::Corrected,
        )
        .unwrap();
        assert_eq!(counts.deep_moisture_choice_differs, 1, "the candidates differ");
        assert_eq!(
            counts.deep_moisture_anchor_reached, 0,
            "but no layer consults the anchor"
        );
        assert_eq!(a[0], b[0], "so the two arms are identical");
    }

    #[test]
    fn a_below_zero_first_guess_temperature_refuses_with_its_count() {
        let err = init_soil_layers_properties(
            &[1i32],
            &[290.0],
            &[285.0],
            &[vec![0.0, 287.0, 286.0, 285.5]],
            &[vec![0.30, 0.28, 0.26, 0.24]],
            &[10.0, 30.0, 60.0, 100.0],
            DeepMoisture::ReproduceFortran,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("at or below zero kelvin"), "{err}");
        assert!(err.contains("1 value(s)"), "{err}");
    }

    #[test]
    fn monthly_interpolation_lands_on_the_month_at_its_midpoint() {
        // Twelve months, each a distinct constant; on 15 June the answer is
        // June's own value.
        let monthly: Vec<Vec<f32>> = (1..=12).map(|m| vec![m as f32]).collect();
        let v = monthly_interp_to_date(&monthly, 2025, 6, 15);
        assert!((v[0] - 6.0).abs() < 1e-4, "{v:?}");
    }

    #[test]
    fn monthly_min_max_spans_the_year() {
        let monthly: Vec<Vec<f32>> = (1..=12).map(|m| vec![m as f32]).collect();
        let (lo, hi) = monthly_min_max(&monthly);
        assert_eq!(lo[0], 1.0);
        assert_eq!(hi[0], 12.0);
    }
}
