//! Derived 2-D planes read straight off a wrfout's native columns, for
//! catalog rows no `getvar` diagnostic writes:
//!
//! * the height of a named isotherm (0, -10 and -20 C), the lowest
//!   crossing above the ground and NaN where the column never crosses;
//! * the supercooled liquid water path, whole column and in two layers,
//!   integrated with the model's own dry-air mass per level;
//! * the column maximum of each hydrometeor mixing ratio, so a
//!   condensate map is one row a user can name instead of a stored
//!   plane spelled `var:`;
//! * a simulated infrared brightness temperature: the cloud-top
//!   temperature of the WRF-Python / NCL `wrfcttcalc` routine, the
//!   temperature at the pressure where the cloud optical depth integrated
//!   down from the model top reaches one, and the lowest model level's
//!   temperature where it never does.
//!
//! Every plane goes to the store under a canonical selector, so the
//! catalog rows in `rustwx-models` resolve it the way they resolve every
//! other stored field, and the import's planned-field preview lists it.
//! The arithmetic is on plain slices so it can be pinned on a synthetic
//! column without a file.

use rustwx_core::{CanonicalField, FieldSelector, GridProjection, LatLonGrid};
use rayon::prelude::*;
use wrf_core::WrfFile;

use crate::wrf_process::{WrfHourFields, WrfProcessOptions, WrfProductGroup, push_canonical_values};

/// Gravity, the constant the model's own column-mass measure uses, and
/// the one `wrfcttcalc` divides its pressure thicknesses by.
const G: f64 = 9.81;
/// Kelvin at 0 C.
const T_FREEZE_K: f64 = 273.15;
/// Mass absorption coefficients for the simulated infrared brightness
/// temperature, in m2 per GRAM of condensate, applied to a mixing ratio
/// in g kg-1 and a layer mass in kg m-2: cloud liquid and cloud ice, as
/// `wrfcttcalc` states them (`ABSCOEF` and `ABSCOEFI` in WRF-Python's
/// `wrf_constants.f90`).  Applied to a ratio in kg kg-1 they give an
/// optical depth a thousand times too small, and thin cirrus drew the
/// ground's temperature.
const IR_ABSORPTION_LIQUID_M2_PER_G: f64 = 0.145;
const IR_ABSORPTION_ICE_M2_PER_G: f64 = 0.272;
/// Grams per kilogram, to put a stored kg kg-1 ratio in the units the
/// coefficients above are per.
const GRAMS_PER_KILOGRAM: f64 = 1000.0;
/// The optical depth, integrated from the model top, at which the column
/// is opaque and its temperature there is the brightness temperature.
const IR_OPAQUE_OPTICAL_DEPTH: f64 = 1.0;
/// `wrfcttcalc`'s surface-pressure extrapolation: the dry-air gas
/// constant (J kg-1 K-1), the US standard atmosphere lapse rate (K m-1)
/// and the ratio of the gas constants of dry air and water vapour.
const IR_RD: f64 = 287.0;
const IR_STANDARD_LAPSE_RATE_K_PER_M: f64 = 0.0065;
const IR_EPS: f64 = 0.622;

/// One catalog row this module writes: the store name a product resolves,
/// the filter key `--only` and `--skip` match beside it, and the row's
/// canonical selector and units.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ColumnPlane {
    pub filter_key: &'static str,
    pub store_name: &'static str,
    pub units: &'static str,
    pub kind: ColumnPlaneKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ColumnPlaneKind {
    IsothermHeight { celsius: i16 },
    SupercooledWaterPath { bottom_m: Option<u16>, top_m: Option<u16> },
    ColumnMaximum { variable: &'static str, field: CanonicalField },
    SimulatedInfrared,
    VapourIntegral { field: CanonicalField },
    VaporTransport { component: usize },
    IceWaterPath,
}

impl ColumnPlane {
    pub(crate) fn selector(&self) -> FieldSelector {
        match self.kind {
            ColumnPlaneKind::VapourIntegral { field } => FieldSelector::entire_atmosphere(field),
            ColumnPlaneKind::VaporTransport { component } => FieldSelector::entire_atmosphere(
                [CanonicalField::IntegratedVaporTransport, CanonicalField::IntegratedVaporTransportU,
                 CanonicalField::IntegratedVaporTransportV][component]),
            ColumnPlaneKind::IsothermHeight { celsius } => {
                FieldSelector::isotherm_celsius(CanonicalField::GeopotentialHeight, celsius)
            }
            ColumnPlaneKind::SupercooledWaterPath {
                bottom_m: Some(bottom_m),
                top_m: Some(top_m),
            } => FieldSelector::height_layer_agl(
                CanonicalField::SupercooledLiquidWaterPath,
                bottom_m,
                top_m,
            ),
            ColumnPlaneKind::SupercooledWaterPath { .. } => {
                FieldSelector::entire_atmosphere(CanonicalField::SupercooledLiquidWaterPath)
            }
            ColumnPlaneKind::ColumnMaximum { field, .. } => FieldSelector::column_maximum(field),
            ColumnPlaneKind::IceWaterPath => FieldSelector::entire_atmosphere(CanonicalField::IceWaterPath),
            ColumnPlaneKind::SimulatedInfrared => FieldSelector::nominal_top(
                CanonicalField::SimulatedInfraredBrightnessTemperature,
            ),
        }
    }
}

/// The rows, in the order they are written.  A table, so a new layer or a
/// new hydrometeor is a row here and a recipe row in `rustwx-models`.
pub(crate) const COLUMN_PLANE_CATALOG: &[ColumnPlane] = &[
    ColumnPlane { filter_key: "ivt", store_name: "integrated_vapor_transport", units: "kg m-1 s-1",
        kind: ColumnPlaneKind::VaporTransport { component: 0 } },
    ColumnPlane { filter_key: "ivt", store_name: "integrated_vapor_transport_u", units: "kg m-1 s-1",
        kind: ColumnPlaneKind::VaporTransport { component: 1 } },
    ColumnPlane { filter_key: "ivt", store_name: "integrated_vapor_transport_v", units: "kg m-1 s-1",
        kind: ColumnPlaneKind::VaporTransport { component: 2 } },

    // The native radiation fraction, not a diagnosed total-cloud plane.
    ColumnPlane {
        filter_key: "cldfra_max",
        store_name: "cloud_fraction_column_maximum",
        units: "%",
        kind: ColumnPlaneKind::ColumnMaximum {
            variable: "CLDFRA",
            field: CanonicalField::TotalCloudCover,
        },
    },
    ColumnPlane { filter_key: "ice_water_path", store_name: "ice_water_path", units: "g m-2", kind: ColumnPlaneKind::IceWaterPath },
    ColumnPlane {
        filter_key: "isotherm",
        store_name: "isotherm_height_0c",
        units: "m",
        kind: ColumnPlaneKind::IsothermHeight { celsius: 0 },
    },
    ColumnPlane {
        filter_key: "isotherm",
        store_name: "isotherm_height_minus10c",
        units: "m",
        kind: ColumnPlaneKind::IsothermHeight { celsius: -10 },
    },
    ColumnPlane {
        filter_key: "isotherm",
        store_name: "isotherm_height_minus20c",
        units: "m",
        kind: ColumnPlaneKind::IsothermHeight { celsius: -20 },
    },
    ColumnPlane {
        filter_key: "supercooled",
        store_name: "supercooled_water_path",
        units: "g m-2",
        kind: ColumnPlaneKind::SupercooledWaterPath {
            bottom_m: None,
            top_m: None,
        },
    },
    ColumnPlane {
        filter_key: "supercooled",
        store_name: "supercooled_water_path_0_3km",
        units: "g m-2",
        kind: ColumnPlaneKind::SupercooledWaterPath {
            bottom_m: Some(0),
            top_m: Some(3000),
        },
    },
    ColumnPlane {
        filter_key: "supercooled",
        store_name: "supercooled_water_path_3_6km",
        units: "g m-2",
        kind: ColumnPlaneKind::SupercooledWaterPath {
            bottom_m: Some(3000),
            top_m: Some(6000),
        },
    },
    ColumnPlane {
        filter_key: "QCLOUD",
        store_name: "cloud_water_column_max",
        units: "kg kg-1",
        kind: ColumnPlaneKind::ColumnMaximum {
            variable: "QCLOUD",
            field: CanonicalField::CloudWaterMixingRatio,
        },
    },
    ColumnPlane {
        filter_key: "QRAIN",
        store_name: "rain_water_column_max",
        units: "kg kg-1",
        kind: ColumnPlaneKind::ColumnMaximum {
            variable: "QRAIN",
            field: CanonicalField::RainWaterMixingRatio,
        },
    },
    ColumnPlane {
        filter_key: "QICE",
        store_name: "cloud_ice_column_max",
        units: "kg kg-1",
        kind: ColumnPlaneKind::ColumnMaximum {
            variable: "QICE",
            field: CanonicalField::CloudIceMixingRatio,
        },
    },
    ColumnPlane {
        filter_key: "QSNOW",
        store_name: "snow_column_max",
        units: "kg kg-1",
        kind: ColumnPlaneKind::ColumnMaximum {
            variable: "QSNOW",
            field: CanonicalField::SnowMixingRatio,
        },
    },
    ColumnPlane {
        filter_key: "QGRAUP",
        store_name: "graupel_column_max",
        units: "kg kg-1",
        kind: ColumnPlaneKind::ColumnMaximum {
            variable: "QGRAUP",
            field: CanonicalField::GraupelMixingRatio,
        },
    },
    ColumnPlane {
        filter_key: "simulated_ir",
        store_name: "simulated_ir_brightness_temperature",
        units: "K",
        kind: ColumnPlaneKind::SimulatedInfrared,
    },
    ColumnPlane { filter_key: "ivt", store_name: "ivt_1000_300", units: "kg m-1 s-1", kind: ColumnPlaneKind::VapourIntegral { field: CanonicalField::IvtMagnitude } },
    ColumnPlane { filter_key: "ivt", store_name: "ivt_eastward_1000_300", units: "kg m-1 s-1", kind: ColumnPlaneKind::VapourIntegral { field: CanonicalField::IvtEastward } },
    ColumnPlane { filter_key: "ivt", store_name: "ivt_northward_1000_300", units: "kg m-1 s-1", kind: ColumnPlaneKind::VapourIntegral { field: CanonicalField::IvtNorthward } },
    ColumnPlane { filter_key: "ivt", store_name: "integrated_water_vapour", units: "kg m-2", kind: ColumnPlaneKind::VapourIntegral { field: CanonicalField::IntegratedWaterVapour } },
    ColumnPlane { filter_key: "ivt", store_name: "ivt_eastward_full_column", units: "kg m-1 s-1", kind: ColumnPlaneKind::VapourIntegral { field: CanonicalField::FullColumnIvtEastward } },
    ColumnPlane { filter_key: "ivt", store_name: "ivt_northward_full_column", units: "kg m-1 s-1", kind: ColumnPlaneKind::VapourIntegral { field: CanonicalField::FullColumnIvtNorthward } },
    ColumnPlane { filter_key: "ivt", store_name: "ivt_full_column", units: "kg m-1 s-1", kind: ColumnPlaneKind::VapourIntegral { field: CanonicalField::FullColumnIvtMagnitude } },
];

/// The store names the current selection would write, for the planned
/// field preview.  Static: a hydrometeor the file does not carry is
/// skipped at process time with a note, like every other planned field.
pub(crate) fn planned_store_fields(options: &WrfProcessOptions) -> Vec<String> {
    COLUMN_PLANE_CATALOG
        .iter()
        .filter(|plane| {
            options.should_process(
                plane.filter_key,
                Some(plane.store_name),
                WrfProductGroup::Diagnostic,
            )
        })
        .map(|plane| plane.store_name.to_string())
        .collect()
}

/// The same rows as [`planned_store_fields`], as the canonical selectors
/// they are written under: the vocabulary a product recipe asks in.
pub(crate) fn planned_store_selectors(options: &WrfProcessOptions) -> Vec<FieldSelector> {
    COLUMN_PLANE_CATALOG
        .iter()
        .filter(|plane| {
            options.should_process(
                plane.filter_key,
                Some(plane.store_name),
                WrfProductGroup::Diagnostic,
            )
        })
        .map(ColumnPlane::selector)
        .collect()
}

/// A column's native fields for one frame, read once and shared by every
/// plane that needs them.
struct Columns {
    nz: usize,
    cells: usize,
    /// Temperature, K, `[nz, cells]`.
    temperature_k: Vec<f64>,
    /// Height above mean sea level, m, `[nz, cells]`.
    height_msl_m: Vec<f64>,
    /// Terrain height, m, `[cells]`.
    terrain_m: Vec<f64>,
}

impl Columns {
    fn read(file: &WrfFile, timeidx: usize) -> Result<Self, String> {
        let nz = file.nz;
        let cells = file.ny * file.nx;
        let temperature_k = file
            .temperature(timeidx)
            .map_err(|err| format!("read temperature: {err}"))?;
        let height_msl_m = file
            .height_msl(timeidx)
            .map_err(|err| format!("read height: {err}"))?;
        let terrain_m = file
            .terrain(timeidx)
            .map_err(|err| format!("read terrain: {err}"))?;
        if temperature_k.len() != nz * cells
            || height_msl_m.len() != nz * cells
            || terrain_m.len() != cells
        {
            return Err(format!(
                "column fields are not {nz}x{cells}: temperature {}, height {}, terrain {}",
                temperature_k.len(),
                height_msl_m.len(),
                terrain_m.len()
            ));
        }
        Ok(Self {
            nz,
            cells,
            temperature_k: temperature_k.to_vec(),
            height_msl_m: height_msl_m.to_vec(),
            terrain_m: terrain_m.to_vec(),
        })
    }

    /// Height above ground, m, `[nz, cells]`.
    fn height_agl_m(&self) -> Vec<f64> {
        self.height_msl_m
            .iter()
            .enumerate()
            .map(|(index, height)| height - self.terrain_m[index % self.cells])
            .collect()
    }
}

/// The dry-air mass per unit area of every model layer, kg m-2,
/// `[nz, cells]`: `(C1H[k] * MU + C2H[k]) * (-DNW[k]) / g`, the model's own
/// eta-coordinate column mass, exact in the discretisation, with MU the
/// total dry column mass (MU + MUB).  A file without the hybrid
/// coefficients is on the terrain-following coordinate, where C1H is one
/// and C2H is zero.  A file without `DNW` but with the full eta levels
/// `ZNW` (this tree's own history stream writes the levels and not the
/// thicknesses) takes each thickness as the difference of the two full
/// levels its layer sits between; a file with neither has no layer mass,
/// and the note names the two variables either of which would give it one.
fn layer_dry_mass(file: &WrfFile, timeidx: usize, nz: usize, cells: usize) -> Result<Vec<f64>, String> {
    let read = |name: &str| {
        file.read_var(name, timeidx)
            .map_err(|err| format!("read {name}: {err}"))
    };
    let mu = read("MU")?;
    let mub = read("MUB")?;
    let dnw = if file.has_var("DNW") {
        read("DNW")?
    } else if file.has_var("ZNW") {
        eta_layer_thickness(&read("ZNW")?)
    } else {
        return Err(String::from(
            "neither DNW nor ZNW is in the file, so no layer has a dry-air mass; write the eta levels (ZNW) or the layer thicknesses (DNW) into the history stream",
        ));
    };
    if mu.len() != cells || mub.len() != cells || dnw.len() != nz {
        return Err(format!(
            "column mass fields are not {nz} levels over {cells} cells: MU {}, MUB {}, DNW {}",
            mu.len(),
            mub.len(),
            dnw.len()
        ));
    }
    let (c1h, c2h) = if file.has_var("C1H") && file.has_var("C2H") {
        let c1h = read("C1H")?;
        let c2h = read("C2H")?;
        if c1h.len() != nz || c2h.len() != nz {
            return Err(format!(
                "hybrid coefficients are not {nz} levels: C1H {}, C2H {}",
                c1h.len(),
                c2h.len()
            ));
        }
        (c1h, c2h)
    } else {
        (vec![1.0; nz], vec![0.0; nz])
    };
    Ok(layer_dry_mass_from(&mu, &mub, &dnw, &c1h, &c2h, nz, cells))
}

/// WRF's `DNW` from its full eta levels `ZNW`: `ZNW[k+1] - ZNW[k]`, one
/// value per mass layer, negative because eta falls with height.
pub(crate) fn eta_layer_thickness(znw: &[f64]) -> Vec<f64> {
    znw.windows(2).map(|pair| pair[1] - pair[0]).collect()
}

/// See [`layer_dry_mass`]; the arithmetic on slices.
pub(crate) fn layer_dry_mass_from(
    mu: &[f64],
    mub: &[f64],
    dnw: &[f64],
    c1h: &[f64],
    c2h: &[f64],
    nz: usize,
    cells: usize,
) -> Vec<f64> {
    let mut mass = vec![0.0; nz * cells];
    for k in 0..nz {
        for cell in 0..cells {
            let column = mu[cell] + mub[cell];
            mass[k * cells + cell] = (c1h[k] * column + c2h[k]) * (-dnw[k]) / G;
        }
    }
    mass
}

/// The height of the isotherm `isotherm_k` in every column: the LOWEST
/// crossing above the ground, in either direction, interpolated linearly
/// in height between the two levels that bracket it, in the units of
/// `height` (MSL or AGL, the caller's choice).  A column whose
/// temperature never reaches the isotherm is NaN, which is what a map
/// paints as nothing rather than as a height of zero.  A level exactly on
/// the isotherm is the crossing.
pub(crate) fn isotherm_height(
    temperature_k: &[f64],
    height: &[f64],
    nz: usize,
    cells: usize,
    isotherm_k: f64,
) -> Vec<f32> {
    let mut out = vec![f32::NAN; cells];
    if nz == 0 {
        return out;
    }
    out.par_iter_mut().enumerate().for_each(|(cell, slot)| {
        let excess = |k: usize| temperature_k[k * cells + cell] - isotherm_k;
        if excess(0) == 0.0 {
            *slot = height[cell] as f32;
            return;
        }
        for k in 0..nz.saturating_sub(1) {
            let below = excess(k);
            let above = excess(k + 1);
            if !(below.is_finite() && above.is_finite()) {
                continue;
            }
            if above == 0.0 {
                *slot = height[(k + 1) * cells + cell] as f32;
                break;
            }
            if (below < 0.0) != (above < 0.0) {
                let fraction = below / (below - above);
                let z0 = height[k * cells + cell];
                let z1 = height[(k + 1) * cells + cell];
                *slot = (z0 + fraction * (z1 - z0)) as f32;
                break;
            }
        }
    });
    out
}

/// The largest value in each column, NaN where no level is finite.
pub(crate) fn column_maximum(field: &[f64], nz: usize, cells: usize) -> Vec<f32> {
    let mut out = vec![f32::NAN; cells];
    out.par_iter_mut().enumerate().for_each(|(cell, slot)| {
        let mut best = f64::NEG_INFINITY;
        for k in 0..nz {
            let value = field[k * cells + cell];
            if value.is_finite() && value > best {
                best = value;
            }
        }
        if best.is_finite() {
            *slot = best as f32;
        }
    });
    out
}

/// The path of `mixing_ratio` (kg kg-1) through every column, g m-2:
/// the sum over levels of the ratio times the layer's dry-air mass, taken
/// only on levels colder than 0 C and, when bounds are given, only on
/// levels whose height above ground lies in `[bottom_m, top_m)`.  A level
/// is counted whole or not at all: the layer bounds fall on the model's
/// own levels, which is the resolution the answer has.
pub(crate) fn supercooled_water_path(
    mixing_ratio: &[f64],
    temperature_k: &[f64],
    layer_mass: &[f64],
    height_agl_m: &[f64],
    nz: usize,
    cells: usize,
    bounds_m: Option<(f64, f64)>,
) -> Vec<f32> {
    let mut out = vec![0.0f32; cells];
    out.par_iter_mut().enumerate().for_each(|(cell, slot)| {
        let mut path = 0.0f64;
        for k in 0..nz {
            let index = k * cells + cell;
            if !(temperature_k[index] < T_FREEZE_K) {
                continue;
            }
            if let Some((bottom, top)) = bounds_m {
                let z = height_agl_m[index];
                if !(z >= bottom && z < top) {
                    continue;
                }
            }
            let q = mixing_ratio[index];
            if q.is_finite() && q > 0.0 {
                path += q * layer_mass[index];
            }
        }
        *slot = (path * 1000.0) as f32;
    });
    out
}

/// A simulated infrared brightness temperature, K: the cloud-top
/// temperature of `wrfcttcalc` (WRF-Python's `fortran/wrf_fctt.f90`, the
/// NCL `wrf_ctt` routine, both carried over from RIP), computed the way
/// that routine computes it:
///
/// * Each mass level's layer is bounded by the full levels halfway in
///   pressure to its neighbours.  The lowest layer reaches down to a
///   surface pressure extrapolated from the lowest level along the US
///   standard atmosphere lapse rate at that level's virtual temperature.
/// * A layer's optical depth is `(0.145 qc + 0.272 qi) dp / g`: the
///   coefficients in m2 g-1, the mixing ratios in g kg-1, `dp` the
///   layer's thickness in total pressure, Pa.  It is summed from the top
///   down, starting one level below the model top, because the
///   reference's loop does not integrate the top level.
/// * A file without cloud ice counts cloud water colder than 0 C at the
///   ice coefficient, the reference's split for a warm-rain scheme.
/// * Where the sum reaches one, the pressure there is interpolated
///   linearly in depth across the crossing layer and kept inside the
///   model's own pressure range, and the brightness temperature is the
///   temperature at that pressure, interpolated linearly in pressure
///   between the two mass levels that bracket it.
/// * A column that never reaches one takes the lowest level's pressure,
///   so it reads the lowest model level's temperature: the reference's
///   default fill, which its documentation calls the surface temperature.
///
/// One divergence, with no effect on any value: the surface pressure is
/// extrapolated over the height of the lowest level above the terrain,
/// as RIP defines it.  `wrfcttcalc` reads `ght(i,j,nz)` there, which on
/// WRF-Python's bottom-up arrays is the model top (RIP's arrays run top
/// down), and extrapolates to several times the real surface pressure.
/// The value cannot tell the two apart: in the lowest layer the crossing
/// pressure is the layer's top plus `(1 - depth above) g / extinction`
/// whatever surface bounds the layer, and the result is clamped to the
/// lowest level's pressure, which is also what a column that does not
/// cross reads.  The test module's transcription keeps the reference's
/// reading, and the two agree on a column whose only condensate is in
/// that layer.
///
/// Inputs are `[nz, cells]` bottom to top with mixing ratios in kg kg-1;
/// `terrain_m` is `[cells]`.  With fewer than two levels there is no
/// layer to integrate, and every cell is NaN, where the reference writes
/// its missing value.
#[allow(clippy::too_many_arguments)]
pub(crate) fn simulated_infrared_brightness_temperature(
    cloud_water: &[f64],
    cloud_ice: Option<&[f64]>,
    water_vapour: &[f64],
    pressure_pa: &[f64],
    temperature_k: &[f64],
    height_msl_m: &[f64],
    terrain_m: &[f64],
    nz: usize,
    cells: usize,
) -> Vec<f32> {
    let mut out = vec![f32::NAN; cells];
    if nz < 2 {
        return out;
    }
    out.par_iter_mut().enumerate().for_each(|(cell, slot)| {
        let at = |field: &[f64], k: usize| field[k * cells + cell];
        let pressure = |k: usize| at(pressure_pa, k);
        let temperature = |k: usize| at(temperature_k, k);

        let vapour = at(water_vapour, 0);
        let virtual_k = temperature(0) * (IR_EPS + vapour) / (IR_EPS * (1.0 + vapour));
        let height_agl_m = at(height_msl_m, 0) - terrain_m[cell];
        let surface_pa = pressure(0)
            * (virtual_k / (virtual_k + IR_STANDARD_LAPSE_RATE_K_PER_M * height_agl_m))
                .powf(-G / (IR_RD * IR_STANDARD_LAPSE_RATE_K_PER_M));

        let mut depth = 0.0f64;
        let mut cloud_top_pa = pressure(0);
        for k in (0..nz - 1).rev() {
            let top_pa = 0.5 * (pressure(k + 1) + pressure(k));
            let bottom_pa = if k == 0 {
                surface_pa
            } else {
                0.5 * (pressure(k) + pressure(k - 1))
            };
            let thickness_pa = bottom_pa - top_pa;
            let liquid_g_per_kg = GRAMS_PER_KILOGRAM * at(cloud_water, k);
            let extinction = match cloud_ice {
                Some(ice) => {
                    IR_ABSORPTION_LIQUID_M2_PER_G * liquid_g_per_kg
                        + IR_ABSORPTION_ICE_M2_PER_G * GRAMS_PER_KILOGRAM * at(ice, k)
                }
                None if temperature(k) < T_FREEZE_K => IR_ABSORPTION_ICE_M2_PER_G * liquid_g_per_kg,
                None => IR_ABSORPTION_LIQUID_M2_PER_G * liquid_g_per_kg,
            };
            let depth_above = depth;
            depth += extinction * thickness_pa / G;
            if depth >= IR_OPAQUE_OPTICAL_DEPTH {
                let fraction = (IR_OPAQUE_OPTICAL_DEPTH - depth_above) / (depth - depth_above);
                cloud_top_pa = (top_pa + fraction * thickness_pa)
                    .max(pressure(nz - 1))
                    .min(pressure(0));
                break;
            }
        }

        for k in (0..nz - 1).rev() {
            let (upper_pa, lower_pa) = (pressure(k + 1), pressure(k));
            if cloud_top_pa >= upper_pa && cloud_top_pa <= lower_pa {
                let fraction = (cloud_top_pa - upper_pa) / (lower_pa - upper_pa);
                *slot = (temperature(k + 1) + fraction * (temperature(k) - temperature(k + 1))) as f32;
                break;
            }
        }
    });
    out
}

/// Write every selected column plane for one frame.  A plane whose inputs
/// the file does not carry (a scheme without graupel, a file without
/// QVAPOR) is a note, never a failed import.
pub(crate) fn push_column_planes(
    fields: &mut WrfHourFields,
    file: &WrfFile,
    timeidx: usize,
    grid: &LatLonGrid,
    projection: Option<GridProjection>,
    options: &WrfProcessOptions,
    progress: &mut dyn FnMut(String),
) {
    let selected: Vec<&ColumnPlane> = COLUMN_PLANE_CATALOG
        .iter()
        .filter(|plane| {
            options.should_process(
                plane.filter_key,
                Some(plane.store_name),
                WrfProductGroup::Diagnostic,
            )
        })
        .collect();
    if selected.is_empty() {
        return;
    }
    progress("Reading native columns for the isotherm, water path and hydrometeor planes".into());
    let columns = match Columns::read(file, timeidx) {
        Ok(columns) => columns,
        Err(err) => {
            fields.notes.push(format!("column planes skipped: {err}"));
            return;
        }
    };
    if columns.cells != grid.shape.len() {
        fields.notes.push(format!(
            "column planes skipped: {} cells in the file, {} in the grid",
            columns.cells,
            grid.shape.len()
        ));
        return;
    }
    let (nz, cells) = (columns.nz, columns.cells);
    let mut layer_mass: Option<Vec<f64>> = None;
    let mut height_agl: Option<Vec<f64>> = None;
    let mut hydrometeor: std::collections::BTreeMap<&'static str, Option<std::rc::Rc<Vec<f64>>>> =
        std::collections::BTreeMap::new();
    let mut read_volume = |name: &'static str| -> Option<std::rc::Rc<Vec<f64>>> {
        hydrometeor
            .entry(name)
            .or_insert_with(|| {
                if !file.has_var(name) {
                    return None;
                }
                match file.read_var(name, timeidx) {
                    Ok(values) if values.len() == nz * cells => Some(std::rc::Rc::new(values)),
                    _ => None,
                }
            })
            .clone()
    };
    let mut vapour: Option<Result<rustwx_calc::ivt::IvtFields, String>> = None;
    let mut ivt = None;
    for plane in selected {
        let values = match plane.kind {
            ColumnPlaneKind::VapourIntegral { field } => {
                let result = vapour.get_or_insert_with(|| read_vapour_integrals(file, timeidx));
                match result {
                    Ok(v) => match field {
                        CanonicalField::IvtMagnitude => v.magnitude.clone(),
                        CanonicalField::IvtEastward => v.eastward.clone(),
                        CanonicalField::IvtNorthward => v.northward.clone(),
                        CanonicalField::IntegratedWaterVapour => v.iwv.clone(),
                        CanonicalField::FullColumnIvtEastward => v.full_eastward.clone(),
                        CanonicalField::FullColumnIvtNorthward => v.full_northward.clone(),
                        CanonicalField::FullColumnIvtMagnitude => v.full_magnitude.clone(),
                        _ => unreachable!("vapour catalogue field"),
                    },
                    Err(err) => { fields.notes.push(format!("{} skipped: {err}", plane.store_name)); continue; }
                }
            },
            ColumnPlaneKind::VaporTransport { component } => {
                let result = ivt.get_or_insert_with(|| crate::ivt::read(file, timeidx));
                match result {
                    Ok(planes) => planes[component].clone(),
                    Err(error) => { fields.notes.push(format!("{} skipped: {error}", plane.store_name)); continue; }
                }
            }
            ColumnPlaneKind::IsothermHeight { celsius } => isotherm_height(
                &columns.temperature_k,
                &columns.height_msl_m,
                nz,
                cells,
                T_FREEZE_K + f64::from(celsius),
            ),
            ColumnPlaneKind::ColumnMaximum { variable, .. } => match read_volume(variable) {
                Some(field) => {
                    let mut maximum = column_maximum(&field, nz, cells);
                    // Radiation CLDFRA is stored as a fraction. The cloud
                    // map's existing color scale and canonical unit are percent.
                    if plane.units == "%" {
                        for value in &mut maximum { *value *= 100.0; }
                    }
                    maximum
                },
                None => {
                    fields
                        .notes
                        .push(format!("{} skipped: {variable} not in the file", plane.store_name));
                    continue;
                }
            },
            ColumnPlaneKind::SupercooledWaterPath { bottom_m, top_m } => {
                let Some(cloud_water) = read_volume("QCLOUD") else {
                    fields
                        .notes
                        .push(format!("{} skipped: QCLOUD not in the file", plane.store_name));
                    continue;
                };
                if layer_mass.is_none() {
                    match layer_dry_mass(file, timeidx, nz, cells) {
                        Ok(mass) => layer_mass = Some(mass),
                        Err(err) => {
                            fields.notes.push(format!("{} skipped: {err}", plane.store_name));
                            continue;
                        }
                    }
                }
                let height_agl = height_agl.get_or_insert_with(|| columns.height_agl_m());
                supercooled_water_path(
                    &cloud_water,
                    &columns.temperature_k,
                    layer_mass.as_deref().expect("set above"),
                    height_agl,
                    nz,
                    cells,
                    match (bottom_m, top_m) {
                        (Some(bottom), Some(top)) => Some((f64::from(bottom), f64::from(top))),
                        _ => None,
                    },
                )
            }
            ColumnPlaneKind::IceWaterPath => {
                let ice: Vec<_> = ["QICE", "QICE2", "QICE3", "QSNOW", "QGRAUP", "QHAIL"]
                    .iter().filter_map(|name| read_volume(name)).collect();
                if ice.is_empty() {
                    fields.notes.push(format!("{} skipped: no frozen-water mass fields", plane.store_name));
                    continue;
                }
                if layer_mass.is_none() {
                    match layer_dry_mass(file, timeidx, nz, cells) {
                        Ok(mass) => layer_mass = Some(mass),
                        Err(err) => { fields.notes.push(format!("{} skipped: {err}", plane.store_name)); continue; }
                    }
                }
                let slices: Vec<&[f64]> = ice.iter().map(|q| q.as_slice()).collect();
                ice_water_path(&slices, layer_mass.as_deref().expect("set above"), nz, cells)
            }
            ColumnPlaneKind::SimulatedInfrared => {
                let (Some(cloud_water), Some(water_vapour)) =
                    (read_volume("QCLOUD"), read_volume("QVAPOR"))
                else {
                    fields.notes.push(format!(
                        "{} skipped: QCLOUD and QVAPOR are both needed",
                        plane.store_name
                    ));
                    continue;
                };
                // Without cloud ice the reference splits cloud water by
                // temperature, so a warm-rain scheme still draws.
                let cloud_ice = read_volume("QICE");
                let pressure_pa = match file.full_pressure(timeidx) {
                    Ok(values) if values.len() == nz * cells => values,
                    _ => {
                        fields.notes.push(format!(
                            "{} skipped: the full pressure P + PB is not in the file",
                            plane.store_name
                        ));
                        continue;
                    }
                };
                simulated_infrared_brightness_temperature(
                    &cloud_water,
                    cloud_ice.as_ref().map(|ice| ice.as_slice()),
                    &water_vapour,
                    &pressure_pa,
                    &columns.temperature_k,
                    &columns.height_msl_m,
                    &columns.terrain_m,
                    nz,
                    cells,
                )
            }
        };
        progress(format!("Computing {}", plane.store_name));
        push_canonical_values(
            fields,
            grid,
            projection.clone(),
            plane.store_name,
            plane.selector(),
            plane.units,
            values,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One column per test, levels bottom to top.
    fn column(values: &[f64]) -> Vec<f64> {
        values.to_vec()
    }

    #[test]
    fn the_isotherm_is_the_lowest_crossing_and_nan_where_the_column_never_crosses() {
        let height = column(&[100.0, 600.0, 1100.0, 1600.0, 2100.0]);
        // 5 C at the ground, cooling 4 K per level: 0 C between levels 1
        // and 2, a quarter of the way up.
        let warm = column(&[278.15, 274.15, 270.15, 266.15, 262.15]);
        let z0 = isotherm_height(&warm, &height, 5, 1, 273.15)[0];
        assert!((z0 - 725.0).abs() < 1e-3, "{z0}");
        // -10 C is crossed between levels 3 and 4.
        let z10 = isotherm_height(&warm, &height, 5, 1, 263.15)[0];
        assert!((z10 - 1975.0).abs() < 1e-3, "{z10}");
        // -20 C is never reached.
        assert!(isotherm_height(&warm, &height, 5, 1, 253.15)[0].is_nan());
        // An inversion: cold ground, warm aloft, then cold again.  The
        // lowest crossing is the one going warm, not the one going cold.
        let inverted = column(&[271.15, 275.15, 274.15, 270.15, 262.15]);
        let z = isotherm_height(&inverted, &height, 5, 1, 273.15)[0];
        assert!((z - 350.0).abs() < 1e-3, "{z}");
        // A level exactly on the isotherm is the crossing.
        let exact = column(&[280.15, 273.15, 260.15]);
        let z = isotherm_height(&exact, &column(&[0.0, 500.0, 1000.0]), 3, 1, 273.15)[0];
        assert!((z - 500.0).abs() < 1e-9, "{z}");
        // Two cells share the level loop: the second is the cold one.
        let two = vec![278.15, 262.15, 274.15, 261.15, 270.15, 260.15];
        let heights = vec![0.0, 0.0, 500.0, 500.0, 1000.0, 1000.0];
        let out = isotherm_height(&two, &heights, 3, 2, 273.15);
        assert!((out[0] - 625.0).abs() < 1e-3, "{}", out[0]);
        assert!(out[1].is_nan());
    }

    #[test]
    fn the_column_maximum_ignores_holes_and_is_nan_on_an_empty_column() {
        let field = vec![1.0, f64::NAN, 3.0, f64::NAN, 2.0, f64::NAN];
        let out = column_maximum(&field, 3, 2);
        assert_eq!(out[0], 3.0);
        assert!(out[1].is_nan());
    }

    #[test]
    fn the_supercooled_path_counts_only_cold_levels_inside_the_layer() {
        // Four levels, one cell: 1 kg m-2 of dry air per level, 1 g kg-1 of
        // cloud water on every level, the two upper levels below freezing.
        let q = column(&[1e-3, 1e-3, 1e-3, 1e-3]);
        let t = column(&[280.0, 275.0, 270.0, 265.0]);
        let mass = column(&[1.0, 1.0, 1.0, 1.0]);
        let z = column(&[500.0, 1500.0, 2500.0, 3500.0]);
        let whole = supercooled_water_path(&q, &t, &mass, &z, 4, 1, None)[0];
        assert!((whole - 2.0).abs() < 1e-6, "{whole} g m-2");
        let low = supercooled_water_path(&q, &t, &mass, &z, 4, 1, Some((0.0, 3000.0)))[0];
        assert!((low - 1.0).abs() < 1e-6, "{low} g m-2");
        let high = supercooled_water_path(&q, &t, &mass, &z, 4, 1, Some((3000.0, 6000.0)))[0];
        assert!((high - 1.0).abs() < 1e-6, "{high} g m-2");
        // A warm column has none, and a negative ratio is not water.
        let warm = column(&[290.0, 285.0, 280.0, 275.0]);
        assert_eq!(supercooled_water_path(&q, &warm, &mass, &z, 4, 1, None)[0], 0.0);
        let negative = column(&[-1e-3, 1e-3, -1e-3, 1e-3]);
        let out = supercooled_water_path(&negative, &t, &mass, &z, 4, 1, None)[0];
        assert!((out - 1.0).abs() < 1e-6, "{out}");
    }

    #[test]
    fn the_layer_mass_is_the_models_own_measure() {
        // Terrain-following coordinate: mass = mu * (-dnw) / g.
        let mass = layer_dry_mass_from(&[500.0], &[80_000.0], &[-0.02, -0.03], &[1.0, 1.0], &[0.0, 0.0], 2, 1);
        assert!((mass[0] - 80_500.0 * 0.02 / G).abs() < 1e-9);
        assert!((mass[1] - 80_500.0 * 0.03 / G).abs() < 1e-9);
        // Hybrid: the c2h term is mass the column does not scale.
        let hybrid = layer_dry_mass_from(&[500.0], &[80_000.0], &[-0.02], &[0.5], &[1000.0], 1, 1);
        assert!((hybrid[0] - (0.5 * 80_500.0 + 1000.0) * 0.02 / G).abs() < 1e-9);
    }

    #[test]
    fn the_layer_thickness_is_the_difference_of_the_full_levels() {
        assert_eq!(eta_layer_thickness(&[1.0, 0.75, 0.5, 0.0]), vec![-0.25, -0.25, -0.5]);
        assert!(eta_layer_thickness(&[1.0]).is_empty());
        // The mass it gives is the mass DNW gives.
        let from_levels = layer_dry_mass_from(
            &[500.0],
            &[80_000.0],
            &eta_layer_thickness(&[1.0, 0.98, 0.95]),
            &[1.0, 1.0],
            &[0.0, 0.0],
            2,
            1,
        );
        assert!((from_levels[0] - 80_500.0 * 0.02 / G).abs() < 1e-6);
        assert!((from_levels[1] - 80_500.0 * 0.03 / G).abs() < 1e-6);
    }

    /// `wrfcttcalc` from WRF-Python's `fortran/wrf_fctt.f90` (develop at
    /// 44bbe878, read 2026-09-29), transcribed line for line for one
    /// column, with its constants from `wrf_constants.f90`.  Fortran's
    /// 1-based `k` is kept; the arrays run bottom to top, as WRF-Python
    /// passes them; pressure is in hPa and the mixing ratios in g kg-1,
    /// as `get_ctt` converts them before the call; `ght(i,j,nz)` is read
    /// where the reference reads it; the fill and the threshold are the
    /// defaults (`fill_nocloud = 0`, `opt_thresh = 1`).  Degrees C, or
    /// `None` where the reference writes its missing value or nothing.
    #[allow(clippy::too_many_arguments)]
    fn reference_wrfcttcalc(
        prs_hpa: &[f64],
        tk_k: &[f64],
        qci_g_per_kg: &[f64],
        qcw_g_per_kg: &[f64],
        qvp_g_per_kg: &[f64],
        ght_m: &[f64],
        ter: f64,
        haveqci: bool,
    ) -> Option<f64> {
        const EPS: f64 = 0.622;
        const USSALR: f64 = 0.0065;
        const RD: f64 = 287.0;
        const G: f64 = 9.81;
        const ABSCOEFI: f64 = 0.272;
        const ABSCOEF: f64 = 0.145;
        const CELKEL: f64 = 273.15;
        let opt_thresh = 1.0;
        let nz = prs_hpa.len();
        let prs = |k: usize| prs_hpa[k - 1];
        let tk = |k: usize| tk_k[k - 1];
        let qci = |k: usize| qci_g_per_kg[k - 1];
        let qcw = |k: usize| qcw_g_per_kg[k - 1];
        let qvp = |k: usize| qvp_g_per_kg[k - 1];
        let ght = |k: usize| ght_m[k - 1];
        let mut pf = vec![0.0f64; nz + 1];

        let ratmix = 0.001 * qvp(1);
        let arg1 = EPS + ratmix;
        let arg2 = EPS * (1.0 + ratmix);
        let vt = tk(1) * arg1 / arg2;
        let agl_hgt = ght(nz) - ter;
        let arg1 = -G / (RD * USSALR);
        pf[nz] = prs(1) * (vt / (vt + USSALR * agl_hgt)).powf(arg1);

        for k in 1..=nz - 1 {
            let ripk = nz - k + 1;
            pf[k] = 0.5 * (prs(ripk) + prs(ripk - 1));
        }

        let mut opdepthd = 0.0f64;
        let mut prsctt = -1.0f64;
        for k in 2..=nz {
            let opdepthu = opdepthd;
            let ripk = nz - k + 1;
            let dp = if k != 1 {
                100.0 * (pf[k] - pf[k - 1])
            } else {
                200.0 * (pf[1] - prs(nz))
            };
            if !haveqci {
                if tk(ripk) < CELKEL {
                    opdepthd = opdepthu + ABSCOEFI * qcw(ripk) * dp / G;
                } else {
                    opdepthd = opdepthu + ABSCOEF * qcw(ripk) * dp / G;
                }
            } else {
                opdepthd += (ABSCOEF * qcw(ripk) + ABSCOEFI * qci(ripk)) * dp / G;
            }
            if opdepthd < opt_thresh && k < nz {
                continue;
            } else if opdepthd < opt_thresh && k == nz {
                prsctt = prs(1);
                break;
            } else {
                let fac = (1.0 - opdepthu) / (opdepthd - opdepthu);
                prsctt = pf[k - 1] + fac * (pf[k] - pf[k - 1]);
                prsctt = prs(1).min(prs(nz).max(prsctt));
                break;
            }
        }

        if prsctt > -1.0 {
            for k in 2..=nz {
                let ripk = nz - k + 1;
                let p1 = prs(ripk + 1);
                let p2 = prs(ripk);
                if prsctt >= p1 && prsctt <= p2 {
                    let fac = (prsctt - p1) / (p2 - p1);
                    let arg1 = fac * (tk(ripk) - tk(ripk + 1)) - CELKEL;
                    return Some(tk(ripk + 1) + arg1);
                }
            }
            None
        } else {
            None
        }
    }

    /// A stand-in sounding, bottom to top: pressure (hPa), temperature
    /// (K), height above sea level (m) and water vapour (kg kg-1), over
    /// terrain at [`STAND_IN_TERRAIN_M`].
    const STAND_IN_PRESSURE_HPA: [f64; 12] =
        [1000.0, 975.0, 925.0, 850.0, 750.0, 650.0, 550.0, 450.0, 350.0, 275.0, 200.0, 150.0];
    const STAND_IN_TEMPERATURE_K: [f64; 12] =
        [293.0, 291.5, 288.0, 283.0, 276.0, 268.0, 259.0, 248.0, 234.0, 224.0, 217.0, 215.0];
    const STAND_IN_HEIGHT_MSL_M: [f64; 12] = [
        190.0, 400.0, 830.0, 1500.0, 2500.0, 3600.0, 4900.0, 6400.0, 8200.0, 9900.0, 12000.0, 13700.0,
    ];
    const STAND_IN_VAPOUR: [f64; 12] = [
        0.012, 0.011, 0.009, 0.007, 0.005, 0.003, 0.0015, 0.0006, 0.0002, 5.0e-5, 2.0e-5, 1.0e-5,
    ];
    const STAND_IN_TERRAIN_M: f64 = 150.0;

    /// One stand-in column's condensate, kg kg-1, on the levels named.
    struct StandIn {
        what: &'static str,
        cloud_water: &'static [(usize, f64)],
        cloud_ice: &'static [(usize, f64)],
    }

    const STAND_INS: &[StandIn] = &[
        StandIn {
            what: "clear",
            cloud_water: &[],
            cloud_ice: &[],
        },
        StandIn {
            what: "thin cirrus, 0.01 g kg-1 of ice at 350 and 275 hPa",
            cloud_water: &[],
            cloud_ice: &[(8, 1.0e-5), (9, 1.0e-5)],
        },
        StandIn {
            what: "stratocumulus, 0.3 g kg-1 of water at 925 and 850 hPa",
            cloud_water: &[(2, 3.0e-4), (3, 3.0e-4)],
            cloud_ice: &[],
        },
        StandIn {
            what: "a deep storm under an anvil",
            cloud_water: &[(3, 1.0e-3), (4, 1.0e-3), (5, 1.0e-3), (6, 5.0e-4), (7, 2.0e-4)],
            cloud_ice: &[(7, 1.0e-4), (8, 1.0e-4), (9, 1.0e-4), (10, 1.0e-4)],
        },
        StandIn {
            what: "fog on the lowest level only, opaque inside its own layer",
            cloud_water: &[(0, 2.0e-4)],
            cloud_ice: &[],
        },
        StandIn {
            what: "haze on the lowest level only, too thin for its own layer",
            cloud_water: &[(0, 5.0e-6)],
            cloud_ice: &[],
        },
        StandIn {
            what: "ice on the top level only, which the reference does not integrate",
            cloud_water: &[],
            cloud_ice: &[(11, 1.0e-3)],
        },
        StandIn {
            what: "supercooled water at 550 hPa",
            cloud_water: &[(6, 2.0e-5)],
            cloud_ice: &[],
        },
    ];

    /// Every stand-in as one `[nz, cells]` field set, a cell per column,
    /// each column's temperatures offset by a tenth of a kelvin per cell so
    /// a column read from its neighbour shows.
    struct StandInFields {
        nz: usize,
        cells: usize,
        pressure_pa: Vec<f64>,
        temperature_k: Vec<f64>,
        height_msl_m: Vec<f64>,
        vapour: Vec<f64>,
        cloud_water: Vec<f64>,
        cloud_ice: Vec<f64>,
        terrain_m: Vec<f64>,
    }

    fn stand_in_fields() -> StandInFields {
        let nz = STAND_IN_PRESSURE_HPA.len();
        let cells = STAND_INS.len();
        let mut fields = StandInFields {
            nz,
            cells,
            pressure_pa: vec![0.0; nz * cells],
            temperature_k: vec![0.0; nz * cells],
            height_msl_m: vec![0.0; nz * cells],
            vapour: vec![0.0; nz * cells],
            cloud_water: vec![0.0; nz * cells],
            cloud_ice: vec![0.0; nz * cells],
            terrain_m: vec![STAND_IN_TERRAIN_M; cells],
        };
        for (cell, stand_in) in STAND_INS.iter().enumerate() {
            for k in 0..nz {
                let index = k * cells + cell;
                fields.pressure_pa[index] = STAND_IN_PRESSURE_HPA[k] * 100.0;
                fields.temperature_k[index] = STAND_IN_TEMPERATURE_K[k] + 0.1 * cell as f64;
                fields.height_msl_m[index] = STAND_IN_HEIGHT_MSL_M[k];
                fields.vapour[index] = STAND_IN_VAPOUR[k];
            }
            for &(k, q) in stand_in.cloud_water {
                fields.cloud_water[k * cells + cell] = q;
            }
            for &(k, q) in stand_in.cloud_ice {
                fields.cloud_ice[k * cells + cell] = q;
            }
        }
        fields
    }

    /// The reference's answer for one cell of `fields`, K.
    fn reference_kelvin(fields: &StandInFields, cell: usize, haveqci: bool) -> f64 {
        let level = |field: &[f64], scale: f64| -> Vec<f64> {
            (0..fields.nz)
                .map(|k| field[k * fields.cells + cell] * scale)
                .collect()
        };
        let celsius = reference_wrfcttcalc(
            &level(&fields.pressure_pa, 0.01),
            &level(&fields.temperature_k, 1.0),
            &level(&fields.cloud_ice, 1000.0),
            &level(&fields.cloud_water, 1000.0),
            &level(&fields.vapour, 1000.0),
            &level(&fields.height_msl_m, 1.0),
            fields.terrain_m[cell],
            haveqci,
        )
        .expect("the reference brackets every stand-in's cloud top");
        celsius + T_FREEZE_K
    }

    #[test]
    fn the_brightness_temperature_is_the_reference_cloud_top_temperature_on_stand_in_columns() {
        let fields = stand_in_fields();
        let ours = simulated_infrared_brightness_temperature(
            &fields.cloud_water,
            Some(fields.cloud_ice.as_slice()),
            &fields.vapour,
            &fields.pressure_pa,
            &fields.temperature_k,
            &fields.height_msl_m,
            &fields.terrain_m,
            fields.nz,
            fields.cells,
        );
        for (cell, stand_in) in STAND_INS.iter().enumerate() {
            let expected = reference_kelvin(&fields, cell, true);
            let got = f64::from(ours[cell]);
            assert!(
                (got - expected).abs() < 1.0e-4,
                "{}: {got} K, the reference {expected} K",
                stand_in.what
            );
        }

        // The units: 0.01 g kg-1 of ice over a 75 hPa layer is an optical
        // depth of 0.272 * 0.01 * 7500 / 9.81 = 2.1, so thin cirrus is
        // opaque and reads its own temperature, about 224 K.  Taken per
        // kilogram, the same ice is 0.002 deep and the column read the
        // lowest level, 293 K.
        let cirrus = f64::from(ours[1]);
        assert!((220.0..228.0).contains(&cirrus), "thin cirrus reads {cirrus} K");
        // A clear column, and a column whose only condensate is too thin
        // or on the top level, reads the lowest model level.
        for cell in [0, 5, 6] {
            let lowest = fields.temperature_k[cell];
            assert!(
                (f64::from(ours[cell]) - lowest).abs() < 1.0e-4,
                "{}: {} K, the lowest level {lowest} K",
                STAND_INS[cell].what,
                ours[cell]
            );
        }
    }

    #[test]
    fn a_file_without_cloud_ice_counts_supercooled_water_at_the_ice_coefficient() {
        let fields = stand_in_fields();
        let without_ice = simulated_infrared_brightness_temperature(
            &fields.cloud_water,
            None,
            &fields.vapour,
            &fields.pressure_pa,
            &fields.temperature_k,
            &fields.height_msl_m,
            &fields.terrain_m,
            fields.nz,
            fields.cells,
        );
        let with_ice = simulated_infrared_brightness_temperature(
            &fields.cloud_water,
            Some(fields.cloud_ice.as_slice()),
            &fields.vapour,
            &fields.pressure_pa,
            &fields.temperature_k,
            &fields.height_msl_m,
            &fields.terrain_m,
            fields.nz,
            fields.cells,
        );
        for (cell, stand_in) in STAND_INS.iter().enumerate() {
            let expected = reference_kelvin(&fields, cell, false);
            let got = f64::from(without_ice[cell]);
            assert!(
                (got - expected).abs() < 1.0e-4,
                "{} without cloud ice: {got} K, the reference {expected} K",
                stand_in.what
            );
        }
        // The supercooled layer is deeper at the ice coefficient, so its
        // top reads colder than it does in a file whose ice field says it
        // is water.
        let supercooled = STAND_INS.len() - 1;
        assert!(
            without_ice[supercooled] + 0.5 < with_ice[supercooled],
            "{} K without cloud ice, {} K with it",
            without_ice[supercooled],
            with_ice[supercooled]
        );
    }

    #[test]
    fn a_column_with_one_level_has_no_layer_and_is_nan() {
        let one = column(&[1.0e-3]);
        let out = simulated_infrared_brightness_temperature(
            &one,
            Some(one.as_slice()),
            &column(&[0.01]),
            &column(&[100_000.0]),
            &column(&[290.0]),
            &column(&[200.0]),
            &[150.0],
            1,
            1,
        );
        assert!(out[0].is_nan(), "{}", out[0]);
    }

    #[test]
    fn every_catalog_row_has_a_selector_the_recipe_catalog_can_resolve() {
        for plane in COLUMN_PLANE_CATALOG {
            let selector = plane.selector();
            let slug = rustwx_models::built_in_plot_recipes()
                .iter()
                .find(|recipe| recipe.filled.selector == Some(selector))
                .map(|recipe| recipe.slug);
            assert!(
                slug.is_some(),
                "{} ({}) has no catalog row asking for it",
                plane.store_name,
                selector.key()
            );
        }
        let planned = planned_store_fields(&WrfProcessOptions::default());
        assert_eq!(planned.len(), COLUMN_PLANE_CATALOG.len());
        let skipped = planned_store_fields(&WrfProcessOptions {
            skip: vec!["supercooled".to_string()],
            ..WrfProcessOptions::default()
        });
        assert_eq!(planned.len() - skipped.len(), 3);
    }
}

#[cfg(test)]
mod radiation_fraction_tests {
    use super::*;

    #[test]
    fn native_radiation_fraction_keeps_fractional_and_binary_column_maxima() {
        let row = COLUMN_PLANE_CATALOG.iter().find(|p| p.filter_key == "cldfra_max").unwrap();
        assert_eq!(row.units, "%");
        assert!(rustwx_models::selector_supported_for_model(
            row.selector(), rustwx_core::ModelId::WrfGdex));
        assert!(matches!(row.kind, ColumnPlaneKind::ColumnMaximum { variable: "CLDFRA", .. }));
        // Two columns share a lower clear layer; only the native upper
        // fraction makes one partial and the other binary overcast.
        let cf = [0.0, 0.0, 0.125, 1.0, 0.75, 0.0];
        assert_eq!(column_maximum(&cf, 3, 2), vec![0.75, 1.0]);
    }
}

/// Shared by the production publisher and native observation verification.
pub(crate) fn read_vapour_integrals(file: &WrfFile, timeidx: usize) -> Result<rustwx_calc::ivt::IvtFields,String> {
    let pressure=file.full_pressure(timeidx).map_err(|e|e.to_string())?;
    let ratio=file.read_var("QVAPOR",timeidx).map_err(|e|e.to_string())?;
    let psfc=file.read_var("PSFC",timeidx).map_err(|e|e.to_string())?;
    let top=file.read_var("P_TOP",timeidx).map_err(|e|format!("P_TOP required to bound native vapour integrals: {e}"))?;
    let model_top_pa=*top.first().ok_or("P_TOP is empty")?;
    let wind=wrf_core::getvar(file,"uvmet",Some(timeidx),&wrf_core::ComputeOpts::default()).map_err(|e|e.to_string())?;
    let count=file.nz*file.ny*file.nx;
    let wind=crate::wrf_volumes::validate_earth_relative_uvmet(wind,file.nz,file.ny,file.nx,count)?;
    rustwx_calc::ivt::compute_ivt(rustwx_calc::ivt::IvtInputs{cells:file.ny*file.nx,levels:file.nz,
        pressure_pa:&pressure,mixing_ratio:&ratio,eastward_ms:&wind[..count],northward_ms:&wind[count..],
        surface_pressure_pa:&psfc,model_top_pa}).map_err(|e|e.to_string())
}

/// Frozen-water mass integrated with the native dry-air layer mass.
/// Categories are separate inputs, so volume and number never enter the sum.
fn ice_water_path(ice: &[&[f64]], mass: &[f64], nz: usize, cells: usize) -> Vec<f32> {
    assert_eq!(mass.len(), nz*cells);
    assert!(ice.iter().all(|q| q.len() == nz*cells));
    let mut out = vec![0.0; cells];
    for k in 0..nz {
        for (c, value) in out.iter_mut().enumerate() {
            let i = k*cells+c;
            let mut frozen = 0.0;
            for q in ice {
                if !q[i].is_finite() { frozen = f64::NAN; break; }
                frozen += q[i].max(0.0);
            }
            *value += frozen * mass[i] * 1000.0;
        }
    }
    out.into_iter().map(|v| v as f32).collect()
}

#[cfg(test)]
mod ice_path_tests {
    use super::ice_water_path;
    #[test]
    fn every_ice_category_contributes_in_native_layer_units() {
        let one = [0.001,0.002,0.003,0.004];
        let two = [0.002,0.0,0.001,0.0];
        let three = [0.003,0.001,0.0,0.002];
        let mass = [10.0,20.0,30.0,40.0];
        assert_eq!(ice_water_path(&[&one,&two,&three],&mass,2,2),vec![180.0,300.0]);
        assert_eq!(ice_water_path(&[&one],&mass,2,2),vec![100.0,200.0]);
    }
    #[test]
    fn missing_or_invalid_ice_cannot_manufacture_path() {
        assert_eq!(ice_water_path(&[&[-0.001]],&[10.0],1,1),vec![0.0]);
        assert!(ice_water_path(&[&[f64::NAN]],&[10.0],1,1)[0].is_nan());
    }
}
