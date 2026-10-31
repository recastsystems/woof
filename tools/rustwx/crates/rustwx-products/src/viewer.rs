//! Store-viewer style resolver: map one STORED variable (its name + the
//! selector JSON the store carries for it) to the production render styling
//! its plot counterpart uses â€” the same `ColorScale`, colormap build
//! options, tick step, legend mode, unit conversion, and title the PNG
//! lanes render with.
//!
//! Identity with production is by construction, not by a parallel table:
//!
//! * derived/heavy slugs resolve through
//!   [`crate::derived::derived_store_variable_style`], which builds a REAL
//!   render request via the same builders the render lanes run and reads
//!   the styling off it;
//! * direct planes resolve their recipe by reverse-matching the stored
//!   `FieldSelector` against the supported recipe catalog (the same
//!   resolution `store_render` performs forward) and then call the direct
//!   lane's own scale/controls/conversion functions
//!   ([`crate::plot_design::operational_fill_scale_for_recipe`],
//!   [`crate::direct::direct_recipe_render_controls`],
//!   [`crate::direct::direct_fill_unit_conversion`]);
//! * the trailing windowed-source planes (`uh_2to5km_max_1h`,
//!   `wind_speed_10m_max_1h`) mirror `build_windowed_render_request`.
//!
//! Variables with NO production fill counterpart (u/v wind components,
//! geopotential height planes â€” production only contours heights â€”
//! `mslp` â€” production contours mslp and fills the companion 10 m wind
//! speed â€” `surface_pressure`, `orography`, 3D volumes) resolve to
//! `None`: the viewer keeps its clearly-labeled generic ramp for those.

use rustwx_core::{CanonicalField, FieldSelector, ModelId, VerticalSelector};
use rustwx_models::{PlotRecipe, built_in_plot_recipes};
use rustwx_render::{
    Color, ColorScale, ColormapBuildOptions, DiscreteColorScale, ExtendMode, LegendControls,
    LegendMode, LevelDensity, MapRenderRequest, ProductVisualMode, RenderDensity, StaticPlotStyle,
    WeatherProduct,
};
use std::collections::HashSet;

use crate::direct::{
    direct_fill_unit_conversion, direct_recipe_render_controls, supported_direct_recipe_slugs,
};
use crate::windowed::HrrrWindowedProduct;

/// The unit conversion applied to raw stored values before the color scale
/// â€” mirrors the direct lane's `convert_filled_field` arithmetic exactly
/// (same f32 expressions), so converted values color identically to the
/// production fill.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnitConvert {
    None,
    /// 2 m temperature/dewpoint: `(K - 273.15) * 9/5 + 32`.
    KelvinToFahrenheit,
    /// Isobaric temperature/dewpoint: `K - 273.15`.
    KelvinToCelsius,
    /// MSLP: `Pa * 0.01`.
    PaToHpa,
    /// QPF / precipitable water: `mm / 25.4` (kg/m^2 == mm of water).
    MmToInches,
    /// Visibility: `m * 0.0006213712`.
    MetersToMiles,
    /// Absolute vorticity: `s^-1 * 1e5`.
    PerSecondToE5PerSecond,
    /// Wind speed/gust: `m/s * 1.9438445`.
    MsToKnots,
    /// Near-surface smoke: `kg/m^3 * 1e9`.
    KgM3ToUgM3,
    /// Column smoke: `kg/m^2 * 1e6`.
    KgM2ToMgM2,
    /// Curated-mapping input already in Celsius against a Fahrenheit
    /// palette: `degC * 9/5 + 32`.
    CelsiusToFahrenheit,
    /// A mixing ratio: `kg/kg * 1000`, the grams per kilogram every route
    /// draws a hydrometeor in.
    KgPerKgToGPerKg,
    /// Move the values onto the power-of-a-thousand decade the legend
    /// states, so a quantity whose whole range sits far from 1 still gets
    /// ticks that read as numbers.  The payload is the exponent the units
    /// carry, so `ScaleByDecade(-3)` multiplies by 1000 against a
    /// `1e-3 <units>` label.  Chosen from the data's own range, never from
    /// a variable's name.
    ScaleByDecade(i32),
}

impl UnitConvert {
    /// Convert one raw stored value into display units with the SAME f32
    /// arithmetic the direct render lane applies before its color scale.
    pub fn apply(self, value: f32) -> f32 {
        match self {
            Self::None => value,
            Self::KelvinToFahrenheit => (value - 273.15) * 9.0 / 5.0 + 32.0,
            Self::KelvinToCelsius => value - 273.15,
            Self::PaToHpa => value * 0.01,
            Self::MmToInches => value / 25.4,
            Self::MetersToMiles => value * 0.000_621_371_2,
            Self::PerSecondToE5PerSecond => value * 100_000.0,
            Self::MsToKnots => value * 1.943_844_5,
            Self::KgM3ToUgM3 => value * 1_000_000_000.0,
            Self::KgM2ToMgM2 => value * 1_000_000.0,
            Self::CelsiusToFahrenheit => value * 9.0 / 5.0 + 32.0,
            Self::KgPerKgToGPerKg => value * 1000.0,
            // The levels this rides against are built in f64, so the
            // factor is taken in f64 and the product narrowed once.  An
            // f32 `powi` of a large exponent is not the f64 one, and a
            // value that lands a hair outside its own end level colours
            // as the extend band rather than as itself.
            Self::ScaleByDecade(exponent) => {
                (f64::from(value) * 10f64.powi(-exponent)) as f32
            }
        }
    }

    pub fn is_none(self) -> bool {
        matches!(self, Self::None)
    }
}

/// The production styling for one stored variable: everything a viewer
/// needs to color pixels and draw a legend that match the variable's plot
/// counterpart. Build the colormap with
/// `rustwx_render::build_colormap(&style.scale, style.colormap_options)`,
/// color CONVERTED values with `cmap.map(...)` (NaN and masked values map
/// to transparent), and label ticks from
/// `rustwx_render::colorbar_ticks(&cmap, style.cbar_tick_step)`, labelled
/// as one set with `rustwx_render::format_tick_labels`.
#[derive(Debug, Clone, PartialEq)]
pub struct StoreVariableStyle {
    /// Production product title (recipe/preset title, no time suffixes).
    pub title: String,
    /// Display units AFTER `convert` (the units the legend is labeled in).
    pub display_units: String,
    /// Conversion from raw stored values to display units.
    pub convert: UnitConvert,
    /// The production color scale (apply to CONVERTED values).
    pub scale: ColorScale,
    /// The colormap build options the production render uses for this
    /// variable's lane, already filtered through the active
    /// `StaticPlotStyle` exactly as the renderer does.
    pub colormap_options: ColormapBuildOptions,
    /// The production colorbar tick step (`None` = auto "nice" ticks).
    pub cbar_tick_step: Option<f64>,
    /// Stepped vs smooth-ramp colorbar painting (also carried inside
    /// `colormap_options.legend.mode`).
    pub legend_mode: LegendMode,
}

/// One operational color-table template a UI can clone into a user-editable
/// table. The style is copied from the same production code path used by
/// rendered products and the native viewer.
#[derive(Debug, Clone, PartialEq)]
pub struct StoreVariableStyleTemplate {
    pub id: String,
    pub slug: String,
    pub label: String,
    pub category: String,
    pub style: StoreVariableStyle,
}

/// The power of a thousand that puts `max_abs` in 1-1000, or 0 when it is
/// already there.  Powers of a thousand, not of ten, so the decade in the
/// units is one a reader recognises (1e-3, 1e-6, 1e3) rather than an
/// arbitrary shift.
pub fn display_exponent(max_abs: f64) -> i32 {
    if !max_abs.is_finite() || max_abs <= 0.0 {
        return 0;
    }
    if (1.0..1000.0).contains(&max_abs) {
        return 0;
    }
    ((max_abs.log10() / 3.0).floor() * 3.0) as i32
}

/// The units a mass mixing ratio is DRAWN in when the stored units are
/// kilograms per kilogram, in any of the spellings a file carries
/// (`kg kg-1`, `kg kg^-1`, `kg kg^{-1}`, `kg/kg`, `kg kg**-1`): grams per
/// kilogram, and the factor that gets there.  `None` for any other units.
///
/// Keyed on the units attribute and never on a variable's name, so a
/// tracer a user added to their own registry with `kg kg-1` on it gets
/// the same bar as QCLOUD, and a plane in any other units is untouched.
/// The section route has always drawn its mixing ratios in g kg-1; the
/// map routes drew the same fields in kg kg-1 moved onto a decade, so one
/// field read `1.2 against 1e-3 kg kg-1` on a map and `1.2 g kg-1` on a
/// cut of the same air.
pub fn grams_per_kilogram(units: &str) -> Option<(f64, &'static str)> {
    let compact: String = units
        .chars()
        .filter(|c| !c.is_whitespace() && !matches!(c, '^' | '{' | '}' | '*'))
        .collect::<String>()
        .to_ascii_lowercase();
    matches!(compact.as_str(), "kgkg-1" | "kg/kg").then_some((1000.0, "g kg-1"))
}

/// `1e-6 kg kg^{-1}`: the units with the decade the values were moved onto.
pub fn scaled_units(units: &str, exponent: i32) -> String {
    if exponent == 0 {
        return units.to_string();
    }
    let units = units.trim();
    if units.is_empty() {
        format!("1e{exponent}")
    } else {
        format!("1e{exponent} {units}")
    }
}

/// Build a neutral full-range style for a stored 2-D variable without a
/// production meteorological counterpart.
///
/// `finite_range` must be the minimum and maximum over finite samples only.
/// A non-degenerate range is represented exactly, without percentile
/// clipping. Constant fields receive display-only padding; absent or invalid
/// ranges use an explicit 0..1 placeholder so renderers still have ordered
/// levels.
///
/// The decade: a plane far from 1-1000 is drawn on the power-of-a-thousand
/// decade that puts the largest of its values in 1-1000, and the decade is
/// stated in the units and the title so a reader can put it back. It was
/// first installed against a colorbar whose every tick read `0` (a real
/// stored 2-D mixing-ratio plane, `kg kg-1`, finite range 1.0071e-3 to
/// 3.8782e-3: fourteen ticks, all fourteen labelled `0`) when the tick
/// labels carried one decimal. The colour bar now labels its ticks as one
/// set (`rustwx_render::format_tick_labels`) with the places the set needs,
/// so telling ticks apart no longer rests on the decade; what the decade
/// still does is keep each label a few characters long, where a plane of
/// order 1e-6 would otherwise print eight or more characters at every tick
/// and the bar keeps only as many labels as its width fits. It is driven
/// by the RANGE, never by a variable's name, the same rule the mesh lane
/// carries. A range already in 1-1000 is untouched, exponent 0, and its
/// style is byte-unchanged.
///
/// A caller that has ALREADY moved its own values onto a decade must use
/// [`generic_style_for_prescaled_store_variable`] instead, or the decade
/// is taken twice and the levels leave the cells they colour behind.
pub fn generic_style_for_store_variable(
    var_name: &str,
    stored_units: &str,
    finite_range: Option<(f32, f32)>,
) -> StoreVariableStyle {
    if let Some(style) = category_style_for_store_variable(var_name, stored_units, finite_range) {
        return style;
    }
    let range = usable_generic_range(finite_range);
    // A mass mixing ratio is drawn in grams per kilogram before any
    // decade is taken: the units decide, never the name.
    let (grams_factor, base_units) = match grams_per_kilogram(stored_units) {
        Some((factor, units)) => (factor, units),
        None => (1.0, stored_units),
    };
    let range = (range.0 * grams_factor, range.1 * grams_factor);
    // The decade the legend will speak in, taken from the data's own range
    // before the levels are cut, so every level and the convert that rides
    // with them agree by construction.
    let exponent = display_exponent(range.0.abs().max(range.1.abs()));
    let factor = 10f64.powi(-exponent);
    let display_units = scaled_units(base_units, exponent);
    let mut style = generic_style_on_a_settled_range(
        var_name,
        &display_units,
        (range.0 * factor, range.1 * factor),
    );
    // The convert carries both moves as one decade: a thousand-fold into
    // grams is three decades, and the legend states the rest.
    let total_exponent = exponent - (grams_factor.log10().round() as i32);
    if total_exponent != 0 {
        style.convert = UnitConvert::ScaleByDecade(total_exponent);
    }
    style
}

/// The same neutral full-range style for a caller that has already moved
/// its values onto the decade `display_units` states.
///
/// WHAT BREAKAGE THIS PREVENTS (gate law): one panel's levels
/// cut on a different decade from the cells they colour. The mesh lane
/// moves its own decade, because it also masks cells below a named floor
/// and has a diverging branch of its own, and it then asked
/// [`generic_style_for_store_variable`] for the levels. That function
/// takes a decade off whatever range it is handed, so a range the caller
/// had already shifted could be shifted a second time and the fill sat a
/// thousand-fold off its own bar. This entry does no decade arithmetic at
/// all: the range it is given is the range its levels span, the units it
/// is given are the units it states, and the returned `convert` is always
/// [`UnitConvert::None`]. Whether a second shift is possible is therefore
/// a question about which function was called, not about which numbers
/// reached it.
pub fn generic_style_for_prescaled_store_variable(
    var_name: &str,
    display_units: &str,
    display_range: Option<(f32, f32)>,
) -> StoreVariableStyle {
    generic_style_on_a_settled_range(
        var_name,
        display_units,
        usable_generic_range(display_range),
    )
}

/// The ordered pair the levels are cut on: the range as given when it is
/// usable, display-only padding around a constant field, and an explicit
/// 0..1 placeholder when there is no usable range at all, so a renderer
/// always has ordered levels.
fn usable_generic_range(finite_range: Option<(f32, f32)>) -> (f64, f64) {
    match finite_range {
        Some((lo, hi)) if lo.is_finite() && hi.is_finite() && lo < hi => {
            (f64::from(lo), f64::from(hi))
        }
        Some((value, other)) if value.is_finite() && other.is_finite() && value == other => {
            let center = f64::from(value);
            let padding = if center == 0.0 {
                1.0
            } else {
                (center.abs() * 0.05).max(1.0e-6)
            };
            (center - padding, center + padding)
        }
        _ => (0.0, 1.0),
    }
}

/// The generic ramp's band count.
const GENERIC_RAMP_BANDS: usize = 9;

/// The one ramp this lane invents itself, so the one ramp a theme may
/// replace wholesale: the theme's sequential colours stepped onto `bands`.
/// No theme, the viridis anchors stepped onto the same count.
fn generic_ramp_colors(bands: usize) -> Vec<Color> {
    const COLORS: [[u8; 4]; GENERIC_RAMP_BANDS] = [
        [68, 1, 84, 255],
        [72, 40, 120, 255],
        [62, 74, 137, 255],
        [49, 104, 142, 255],
        [38, 130, 142, 255],
        [31, 158, 137, 255],
        [53, 183, 121, 255],
        [109, 205, 89, 255],
        [253, 231, 37, 255],
    ];
    rustwx_render::active_theme()
        .sequential_colors(bands)
        .unwrap_or_else(|| {
            if bands == GENERIC_RAMP_BANDS {
                return COLORS
                    .iter()
                    .map(|[r, g, b, a]| Color::rgba(*r, *g, *b, *a))
                    .collect();
            }
            let anchors: Vec<rustwx_render::Rgba> = COLORS
                .iter()
                .map(|[r, g, b, a]| rustwx_render::Rgba::with_alpha(*r, *g, *b, *a))
                .collect();
            rustwx_render::theme::resample(&anchors, bands)
        })
}

/// The headline a generic or category row carries: the variable's name, its
/// units riding along when it has any.
fn generic_title(var_name: &str, display_units: &str) -> String {
    if display_units.trim().is_empty() {
        var_name.to_string()
    } else {
        format!("{var_name} [{display_units}]")
    }
}

/// Stored planes whose values are category codes: a class number (land use,
/// vegetation, soil, slope, crop, urban type, growing stage) or a mask
/// value.  Matched on the plane's own name with or without the `wrf_` prefix
/// the store gives a raw WRF plane, case-insensitively, so a stored plane
/// and a mesh field of the same name agree.  Adding a code plane is a row
/// here, never a code path.
///
/// Deliberately absent: `SNOWC` and `SEAICE` (land-surface and sea-ice
/// options write fractions into both), `ISNOW`, `KPBL` and other integer
/// counts and level indices (a quantity, not an identity), and every
/// fraction-per-class plane (`LANDUSEF`, `SOILCTOP`), which is a continuous
/// quantity per class.
const CATEGORY_CODE_PLANES: [&str; 14] = [
    "lu_index",
    "ivgtyp",
    "isltyp",
    "sct_dom",
    "scb_dom",
    "soilcat",
    "vegcat",
    "slopecat",
    "cropcat",
    "utype_urb2d",
    "pgs",
    "landmask",
    "lakemask",
    "xland",
];

/// More codes than this is not a legend a reader can use, and a plane that
/// wide is not carrying class numbers.
const MAX_CATEGORY_BANDS: f64 = 256.0;

/// True when `name` is a stored plane of category codes.
pub fn is_category_code_plane(name: &str) -> bool {
    let lower = name.trim().to_ascii_lowercase();
    let bare = lower.strip_prefix("wrf_").unwrap_or(&lower);
    CATEGORY_CODE_PLANES.contains(&bare)
}

/// The category style for a plane of category codes: one legend band per
/// integer code from the smallest to the largest present, each band centred
/// on its code and labelled with it, a colour of its own per code, nearest
/// sampling and no densification (both carried by
/// [`LegendMode::Categories`]).
///
/// WHAT BREAKAGE THIS PREVENTS (gate law): category maps drawn as if their
/// codes were a continuous quantity.  A land mask holding only 0 and 1 was
/// interpolated into lake shores of 0.3 and 0.7 and drawn against a
/// continuous 0 to 1 bar; a soil plane holding codes 2, 3 and 14 painted
/// thousands of pixels in colours of codes no cell holds.  `None` when the
/// name is not a code plane, or its finite range does not start and end on
/// whole codes or spans more than [`MAX_CATEGORY_BANDS`] codes: those keep
/// the generic ramp.
pub fn category_style_for_store_variable(
    var_name: &str,
    display_units: &str,
    finite_range: Option<(f32, f32)>,
) -> Option<StoreVariableStyle> {
    if !is_category_code_plane(var_name) {
        return None;
    }
    let (lo, hi) = finite_range?;
    let (lo, hi) = (f64::from(lo), f64::from(hi));
    let whole = |value: f64| value.is_finite() && (value - value.round()).abs() <= 1.0e-6;
    if !whole(lo) || !whole(hi) || hi < lo || hi - lo >= MAX_CATEGORY_BANDS {
        return None;
    }
    let first = lo.round();
    let bands = (hi.round() - first) as usize + 1;
    let levels = (0..=bands).map(|band| first + band as f64 - 0.5).collect();
    let legend = LegendControls {
        density: LevelDensity::default(),
        mode: LegendMode::Categories,
    };
    Some(StoreVariableStyle {
        title: generic_title(var_name, display_units),
        display_units: display_units.to_string(),
        convert: UnitConvert::None,
        scale: ColorScale::Discrete(DiscreteColorScale {
            levels,
            colors: generic_ramp_colors(bands),
            extend: ExtendMode::Neither,
            mask_below: None,
        }),
        colormap_options: ColormapBuildOptions {
            render_density: RenderDensity {
                fill: LevelDensity::default(),
                palette_multiplier: 1,
            },
            legend,
        },
        cbar_tick_step: None,
        legend_mode: legend.mode,
    })
}

/// Levels, palette and legend over a range that is already final: no
/// decade is taken here, and `convert` is always [`UnitConvert::None`].
fn generic_style_on_a_settled_range(
    var_name: &str,
    display_units: &str,
    range: (f64, f64),
) -> StoreVariableStyle {
    let levels = (0..=GENERIC_RAMP_BANDS)
        .map(|index| {
            range.0 + (range.1 - range.0) * index as f64 / GENERIC_RAMP_BANDS as f64
        })
        .collect();
    let legend = LegendControls {
        density: LevelDensity::default(),
        mode: LegendMode::SmoothRamp,
    };
    let colors = generic_ramp_colors(GENERIC_RAMP_BANDS);

    StoreVariableStyle {
        // The clean headline other rows get: the variable's name (its
        // stored units ride along when it has any).  The auto-ranged
        // nature of the ramp is visible in the legend itself and logged
        // per variable at render time; spelling it in the headline made
        // uncurated rows read like errors next to curated ones.
        title: generic_title(var_name, display_units),
        display_units: display_units.to_string(),
        convert: UnitConvert::None,
        scale: ColorScale::Discrete(DiscreteColorScale {
            levels,
            colors,
            extend: ExtendMode::Neither,
            mask_below: None,
        }),
        colormap_options: ColormapBuildOptions {
            render_density: StaticPlotStyle::from_env().render_density(RenderDensity::default()),
            legend,
        },
        cbar_tick_step: None,
        legend_mode: legend.mode,
    }
}

/// Chart levels the curated resolver may borrow a same-field style from
/// when the exact stored level has no filled recipe of its own.
const CURATED_CHART_LEVELS_HPA: [u16; 6] = [850, 700, 500, 300, 250, 200];

/// Stamp a borrowed operational style with the variable's own identity.
/// The palette, scale, conversion, and legend stay the production ones;
/// only the title says whose data is wearing them, so a borrowed table can
/// never impersonate the product it came from.
fn borrowed_style_identity(mut style: StoreVariableStyle, var_name: &str) -> StoreVariableStyle {
    style.title = format!("{var_name} ({} palette)", style.title);
    style
}

/// Plain headlines for curated planes a product page shows as a map of
/// its own (WOOF city templates request `var:wrf_snownc`).  The palette
/// stays the borrowed production one; the headline names the quantity
/// the plane really holds, so the borrowed table still cannot impersonate
/// the product it came from, and the map no longer reads
/// "wrf_snownc (Total QPF palette)".
fn curated_headline(name: &str) -> Option<&'static str> {
    match name {
        "snownc" => Some("Snow since start, liquid equivalent"),
        "graupelnc" => Some("Graupel since start, liquid equivalent"),
        "hailnc" => Some("Hail since start, liquid equivalent"),
        _ => None,
    }
}

/// Curated colortable mapping for stored variables the production style
/// resolver leaves unstyled: same-quantity diagnostics wear the EXISTING
/// operational palette for their physical quantity instead of the
/// auto-ranged generic ramp.
///
/// Two rule families, both resolving through the production style paths
/// (never a parallel color table):
///
/// 1. Canonical isobaric planes whose exact level has no filled recipe
///    borrow the same canonical FIELD's style from another chart level
///    (dewpoint at 200-500 hPa, absolute vorticity at 250 hPa, ...).
/// 2. Derived-marker diagnostics map by normalized name + stored units to
///    the operational style of the same physical quantity (CAPE-likes to
///    the CAPE table, `wrf_t2` to the 2 m temperature table, ...), with a
///    unit conversion override where the stored units differ from the
///    palette's input calibration.
///
/// Deliberate non-mappings (they return `None` and keep the generic fill):
/// signed u/v wind components (no production diverging table exists),
/// mslp/surface pressure and isobaric geopotential heights (production
/// contours these, never fills), and quantities with no operational palette
/// at all (radiation/heat fluxes, fire indices, wind direction, level
/// heights).  A curated claim for those would show legend colors no
/// production chart has ever used for that quantity.
pub fn curated_style_for_store_variable(
    var_name: &str,
    stored_selector: &serde_json::Value,
    stored_units: &str,
    model: ModelId,
) -> Option<StoreVariableStyle> {
    // Family 1: chart-level gap fill for canonical planes.
    if let Ok(selector) = serde_json::from_value::<FieldSelector>(stored_selector.clone()) {
        let VerticalSelector::IsobaricHpa(stored_level) = selector.vertical else {
            return None;
        };
        for level in CURATED_CHART_LEVELS_HPA {
            if level == stored_level {
                continue;
            }
            let candidate = FieldSelector::isobaric(selector.field, level);
            let Ok(candidate_json) = serde_json::to_value(candidate) else {
                continue;
            };
            if let Some(style) =
                operational_style_for_store_variable(var_name, &candidate_json, stored_units, model)
            {
                return Some(borrowed_style_identity(style, var_name));
            }
        }
        return None;
    }

    // Family 2: name/units-driven quantity mapping for derived diagnostics.
    let name = var_name.strip_prefix("wrf_").unwrap_or(var_name);
    let units = stored_units.trim();
    let canonical = |selector: FieldSelector| -> Option<StoreVariableStyle> {
        let json = serde_json::to_value(selector).ok()?;
        operational_style_for_store_variable(var_name, &json, stored_units, model)
    };
    let style = match (name, units) {
        ("cape" | "effective_cape", "J/kg") => weather_product_style("sbcape", units),
        ("cin", "J/kg") => weather_product_style("sbcin", units),
        ("srh" | "effective_srh", "m2/s2") => weather_product_style("srh_0_3km", units),
        ("up_heli_max", _) => weather_product_style("uhel", units),
        ("bulk_shear" | "ebwd", "m/s") => derived_style("bulk_shear_0_6km", "kt")
            .map(|mut style| {
                style.convert = UnitConvert::MsToKnots;
                style.display_units = "kt".to_string();
                style
            }),
        ("lapse_rate", "degC/km") => derived_style("lapse_rate_0_3km", units),
        ("t2" | "tsk" | "tv2m", "K") => {
            canonical(FieldSelector::height_agl(CanonicalField::Temperature, 2))
        }
        ("ctt", "degC") => {
            // Cloud-top temperature wears the isobaric temperature palette
            // (degC-calibrated); the stored values are already Celsius.
            canonical(FieldSelector::isobaric(CanonicalField::Temperature, 500)).map(|mut style| {
                style.convert = UnitConvert::None;
                style.display_units = stored_units.to_string();
                style
            })
        }
        ("dp2m", "degC") => canonical(FieldSelector::height_agl(CanonicalField::Dewpoint, 2))
            .map(|mut style| {
                style.convert = UnitConvert::CelsiusToFahrenheit;
                style
            }),
        ("rh2m", "%") => canonical(FieldSelector::height_agl(CanonicalField::RelativeHumidity, 2)),
        ("cloudfrac_low", "%") => {
            canonical(FieldSelector::entire_atmosphere(CanonicalField::LowCloudCover))
        }
        ("cloudfrac_mid", "%") => {
            canonical(FieldSelector::entire_atmosphere(CanonicalField::MiddleCloudCover))
        }
        ("cloudfrac_high", "%") => {
            canonical(FieldSelector::entire_atmosphere(CanonicalField::HighCloudCover))
        }
        ("pw", "mm") => {
            canonical(FieldSelector::entire_atmosphere(CanonicalField::PrecipitableWater))
        }
        // hailnc joins the QPF fill with graupelnc and snownc: it is the
        // same accumulated-surface-precipitation quantity, in the same
        // units, and without the arm a hail-bearing run's plane wore the
        // generic auto-ranged ramp (audit R-053).
        ("graupelnc" | "snownc" | "hailnc", "mm") => {
            canonical(FieldSelector::surface(CanonicalField::TotalPrecipitation))
        }
        ("snow_level_ft", "ft") => Some(snow_level_style()),
        ("terrain", "m") => canonical(FieldSelector::surface(CanonicalField::GeopotentialHeight)),
        ("wspd10", "m/s") => {
            canonical(FieldSelector::height_agl(CanonicalField::WindSpeed, 10))
        }
        _ => None,
    };
    style.map(|mut style| match curated_headline(name) {
        Some(headline) => {
            style.title = headline.to_string();
            style
        }
        None => borrowed_style_identity(style, var_name),
    })
}

/// Resolve the production styling for the stored variable `var_name`
/// carrying `stored_selector` (the store's per-variable selector JSON:
/// either a `FieldSelector` or a `{"derived": slug}` marker) and
/// `stored_units`. Returns `None` for variables with no production fill
/// counterpart â€” the caller should fall back to a clearly-labeled generic
/// ramp.
pub fn operational_style_for_store_variable(
    var_name: &str,
    stored_selector: &serde_json::Value,
    stored_units: &str,
    model: ModelId,
) -> Option<StoreVariableStyle> {
    if let Some(slug) = stored_selector.get("derived").and_then(|v| v.as_str()) {
        let slug = normalize_wrf_store_slug(slug);
        // A derived product's plane in a unit its colour bar cannot be
        // reached from claims no production palette at all: the fallback
        // below would put the same numbers on a bar in another unit.
        crate::derived::store_plane_conversion(&slug, stored_units).ok()?;
        return derived_style(&slug, stored_units)
            .or_else(|| weather_product_style(&slug, stored_units));
    }
    let selector: FieldSelector = serde_json::from_value(stored_selector.clone()).ok()?;

    // Trailing (h-1)->h window planes: their formal plot counterpart is the
    // windowed product family, mirroring `build_windowed_render_request`
    // (`1h_qpf` is deliberately NOT a direct recipe â€” it routes to the
    // windowed `qpf_1h` product, see `LEGACY_PRODUCT_ALIASES`).
    match var_name {
        "uh_2to5km_max_1h" => return Some(windowed_uh_style(stored_units)),
        "wind_speed_10m_max_1h" => return Some(windowed_wind10m_style()),
        "apcp_1h" => return Some(windowed_qpf_1h_style()),
        _ => {}
    }

    // Production never color-fills ISOBARIC geopotential height (height
    // recipes are contour-only / fill companion wind speed).  The surface
    // member of the same canonical field is orography, which now has a
    // production fill of its own, so it keeps its plot identity instead of
    // the generic ramp.
    if selector.field == CanonicalField::GeopotentialHeight
        && !matches!(selector.vertical, VerticalSelector::Surface)
    {
        return None;
    }

    // Production never color-fills mslp values either: the `mslp_10m_winds`
    // plot fills the companion 10 m wind speed (kt) and only CONTOURS mslp,
    // so no production colorbar exists for the stored pressure plane.
    // Claiming that plot's identity over the catalog WeatherPressure scale
    // would show legend values (960..1044 hPa) that match no production
    // colorbar â€” keep the clearly-labeled generic ramp instead.
    if selector.field == CanonicalField::PressureReducedToMeanSeaLevel {
        return None;
    }

    let recipe = direct_recipe_for_selector(var_name, selector, model)?;
    let scale = crate::plot_design::operational_fill_scale_for_recipe(recipe, selector);
    let (convert, units_override) = direct_fill_unit_conversion(recipe, selector);
    let (legend_mode_override, cbar_tick_step) = direct_recipe_render_controls(recipe, selector);
    let (render_density, mut legend) = operational_request_chrome();
    if let Some(mode) = legend_mode_override {
        legend.mode = mode;
    }
    Some(StoreVariableStyle {
        title: recipe.title.to_string(),
        display_units: units_override.map_or_else(|| stored_units.to_string(), str::to_string),
        convert,
        scale,
        colormap_options: filtered_options(render_density, legend),
        cbar_tick_step,
        legend_mode: legend.mode,
    })
}

/// Derived/heavy slugs: styling read off a real render request built by the
/// production builders (see `derived_store_variable_style`).  The palette
/// is calibrated in the product's own units, so a plane stored in another
/// unit is converted into them the way the named picture converts it
/// ([`crate::derived::store_plane_conversion`]); one the product's units
/// cannot be reached from gets no production style.
fn derived_style(slug: &str, stored_units: &str) -> Option<StoreVariableStyle> {
    use crate::derived::StoreUnitConversion;

    let lane = crate::derived::derived_store_variable_style(slug).ok()?;
    let (convert, display_units) =
        match crate::derived::store_plane_conversion(slug, stored_units).ok()? {
            StoreUnitConversion::Same => (UnitConvert::None, stored_units.to_string()),
            StoreUnitConversion::MetresPerSecondToKnots => (
                UnitConvert::MsToKnots,
                crate::derived::derived_product_units(slug)?.to_string(),
            ),
        };
    Some(StoreVariableStyle {
        title: lane.title,
        display_units,
        convert,
        scale: lane.scale,
        colormap_options: filtered_options(lane.render_density, lane.legend),
        cbar_tick_step: lane.cbar_tick_step,
        legend_mode: lane.legend.mode,
    })
}

/// Every production color table currently reachable through the operational
/// render/style system, packaged as cloneable user-table templates.
pub fn operational_style_templates(model: ModelId) -> Vec<StoreVariableStyleTemplate> {
    let mut out = Vec::new();
    let mut seen = HashSet::<String>::new();

    let supported = style_recipe_slugs(model);
    for recipe in built_in_plot_recipes()
        .iter()
        .filter(|recipe| supported.iter().any(|slug| slug == recipe.slug))
    {
        let Some(selector) = recipe.filled.selector else {
            continue;
        };
        if selector.field == CanonicalField::GeopotentialHeight
            || selector.field == CanonicalField::PressureReducedToMeanSeaLevel
        {
            continue;
        }
        let Ok(selector_json) = serde_json::to_value(selector) else {
            continue;
        };
        if let Some(mut style) = operational_style_for_store_variable(
            recipe.slug,
            &selector_json,
            selector.native_units(),
            model,
        ) {
            style.title = recipe.title.to_string();
            push_template(
                &mut out,
                &mut seen,
                "direct",
                recipe.slug,
                recipe.title,
                style,
            );
        }
    }

    for entry in crate::derived::supported_derived_recipe_inventory() {
        // A derived template is labelled in the units its palette is
        // calibrated in; the slug-shape guess below is for the weather
        // products, which name no units of their own.
        let units = crate::derived::derived_product_units(entry.slug)
            .unwrap_or_else(|| template_units_for_slug(entry.slug));
        if let Some(mut style) = operational_style_for_store_variable(
            entry.slug,
            &serde_json::json!({ "derived": entry.slug }),
            units,
            model,
        ) {
            style.title = entry.title.to_string();
            push_template(
                &mut out,
                &mut seen,
                if entry.heavy { "heavy" } else { "derived" },
                entry.slug,
                entry.title,
                style,
            );
        }
    }

    for product in ALL_WEATHER_PRODUCTS {
        if let Some(style) =
            weather_product_style(product.slug(), template_units_for_slug(product.slug()))
        {
            push_template(
                &mut out,
                &mut seen,
                "weather",
                product.slug(),
                product.display_title(),
                style,
            );
        }
    }

    for &product in HrrrWindowedProduct::supported_products() {
        if let Some(style) = windowed_product_style(product) {
            push_template(
                &mut out,
                &mut seen,
                "windowed",
                product.slug(),
                product.title(),
                style,
            );
        }
    }

    out.sort_by(|a, b| {
        a.category
            .cmp(&b.category)
            .then_with(|| a.label.cmp(&b.label))
            .then_with(|| a.slug.cmp(&b.slug))
    });
    out
}

fn push_template(
    out: &mut Vec<StoreVariableStyleTemplate>,
    seen: &mut HashSet<String>,
    category: &str,
    slug: &str,
    label: &str,
    style: StoreVariableStyle,
) {
    let id = format!("{category}:{slug}");
    if seen.insert(id.clone()) {
        out.push(StoreVariableStyleTemplate {
            id,
            slug: slug.to_string(),
            label: label.to_string(),
            category: category.to_string(),
            style,
        });
    }
}

/// Local WRF processing initially stored `wrf_*` derived markers, while the
/// existing operational style catalog uses model-agnostic product slugs.
/// Normalize those aliases before resolving a derived/weather palette.
fn normalize_wrf_store_slug(slug: &str) -> String {
    let slug = slug.strip_prefix("wrf_").unwrap_or(slug);
    match slug {
        "srh1" => "srh_0_1km".to_string(),
        "srh3" => "srh_0_3km".to_string(),
        "shear_0_1km" => "bulk_shear_0_1km".to_string(),
        "shear_0_6km" => "bulk_shear_0_6km".to_string(),
        "uhel" => "uhel".to_string(),
        other => other.to_string(),
    }
}

/// WRF-core exposes a few already-computed severe parameters that are not
/// HRRR-derived recipe slugs but do have first-class WeatherProduct palettes
/// (STP/SCP/EHI/LCL/LFC/EL/UH, etc.). Use the same weather request builder
/// as the production weather lanes for those.
fn weather_product_style(slug: &str, stored_units: &str) -> Option<StoreVariableStyle> {
    let product = WeatherProduct::from_product_name(slug)?;
    let mut request = MapRenderRequest::for_core_weather_product(probe_core_field(), product);
    apply_probe_static_design(&mut request);
    Some(StoreVariableStyle {
        title: product.display_title().to_string(),
        display_units: stored_units.to_string(),
        convert: UnitConvert::None,
        scale: request.scale,
        colormap_options: filtered_options(request.render_density, request.legend),
        cbar_tick_step: request.cbar_tick_step,
        legend_mode: request.legend.mode,
    })
}

/// `uh_2to5km_max_1h`: the windowed UH family request â€”
/// `for_core_weather_product(WeatherProduct::Uh)` + static map design.
fn windowed_uh_style(stored_units: &str) -> StoreVariableStyle {
    let mut request =
        MapRenderRequest::for_core_weather_product(probe_core_field(), WeatherProduct::Uh);
    apply_probe_static_design(&mut request);
    StoreVariableStyle {
        title: HrrrWindowedProduct::Uh25km1h.title().to_string(),
        display_units: stored_units.to_string(),
        convert: UnitConvert::None,
        scale: request.scale,
        colormap_options: filtered_options(request.render_density, request.legend),
        cbar_tick_step: request.cbar_tick_step,
        legend_mode: request.legend.mode,
    }
}

/// `wind_speed_10m_max_1h`: the windowed 10 m wind family request â€”
/// `from_core_field(windowed_product_scale(...))` + static map design. The
/// stored plane is m/s; the windowed lane displays knots.
fn windowed_wind10m_style() -> StoreVariableStyle {
    let scale = crate::windowed_decoder::windowed_product_scale(HrrrWindowedProduct::Wind10m1hMax);
    let mut request = MapRenderRequest::from_core_field(probe_core_field(), scale);
    apply_probe_static_design(&mut request);
    StoreVariableStyle {
        title: HrrrWindowedProduct::Wind10m1hMax.title().to_string(),
        display_units: "kt".to_string(),
        convert: UnitConvert::MsToKnots,
        scale: request.scale,
        colormap_options: filtered_options(request.render_density, request.legend),
        cbar_tick_step: request.cbar_tick_step,
        legend_mode: request.legend.mode,
    }
}

/// `apcp_1h`: the trailing 1 h QPF window â€” the windowed `qpf_1h` product
/// (its legacy `1h_qpf` recipe slug deliberately aliases to the windowed
/// lane). Stored kg/m^2 == mm; displayed in inches like all QPF products.
fn windowed_qpf_1h_style() -> StoreVariableStyle {
    let scale = crate::windowed_decoder::windowed_product_scale(HrrrWindowedProduct::Qpf1h);
    let mut request = MapRenderRequest::from_core_field(probe_core_field(), scale);
    apply_probe_static_design(&mut request);
    StoreVariableStyle {
        title: HrrrWindowedProduct::Qpf1h.title().to_string(),
        display_units: "in".to_string(),
        convert: UnitConvert::MmToInches,
        scale: request.scale,
        colormap_options: filtered_options(request.render_density, request.legend),
        cbar_tick_step: request.cbar_tick_step,
        legend_mode: request.legend.mode,
    }
}

/// Reverse-resolve one stored direct plane to its plot recipe: the first
/// supported recipe whose `filled.selector` equals the stored selector â€”
/// except `apcp_run_total`, whose store name pins the run-total window
/// identity for the shared plain TotalPrecipitation selector.
fn windowed_product_style(product: HrrrWindowedProduct) -> Option<StoreVariableStyle> {
    let scale = crate::windowed_decoder::windowed_product_scale(product);
    let mut request = MapRenderRequest::from_core_field(probe_core_field(), scale);
    apply_probe_static_design(&mut request);
    let (convert, display_units) = windowed_product_units(product);
    Some(StoreVariableStyle {
        title: product.title().to_string(),
        display_units: display_units.to_string(),
        convert,
        scale: request.scale,
        colormap_options: filtered_options(request.render_density, request.legend),
        cbar_tick_step: request.cbar_tick_step,
        legend_mode: request.legend.mode,
    })
}

fn windowed_product_units(product: HrrrWindowedProduct) -> (UnitConvert, &'static str) {
    let slug = product.slug();
    if slug.starts_with("qpf_") {
        (UnitConvert::MmToInches, "in")
    } else if slug.starts_with("10m_wind_") {
        (UnitConvert::MsToKnots, "kt")
    } else if slug.starts_with("2m_temp_") || slug.starts_with("2m_dewpoint_") {
        (UnitConvert::KelvinToFahrenheit, "degF")
    } else if slug.starts_with("2m_rh_") {
        (UnitConvert::None, "%")
    } else if slug.starts_with("2m_vpd_") {
        (UnitConvert::None, "hPa")
    } else if slug.starts_with("uh_") {
        (UnitConvert::None, "m^2/s^2")
    } else {
        (UnitConvert::None, "")
    }
}

const ALL_WEATHER_PRODUCTS: &[WeatherProduct] = &[
    WeatherProduct::Sbcape,
    WeatherProduct::Mlcape,
    WeatherProduct::Mucape,
    WeatherProduct::Sbecape,
    WeatherProduct::Mlecape,
    WeatherProduct::Muecape,
    WeatherProduct::SbEcapeDerivedCapeRatio,
    WeatherProduct::MlEcapeDerivedCapeRatio,
    WeatherProduct::MuEcapeDerivedCapeRatio,
    WeatherProduct::SbEcapeNativeCapeRatio,
    WeatherProduct::MlEcapeNativeCapeRatio,
    WeatherProduct::MuEcapeNativeCapeRatio,
    WeatherProduct::Sbncape,
    WeatherProduct::Mlncape,
    WeatherProduct::Muncape,
    WeatherProduct::Sbcin,
    WeatherProduct::Mlcin,
    WeatherProduct::Mucin,
    WeatherProduct::Sbecin,
    WeatherProduct::Mlecin,
    WeatherProduct::Muecin,
    WeatherProduct::EcapeCape,
    WeatherProduct::EcapeCin,
    WeatherProduct::Lcl,
    WeatherProduct::Lfc,
    WeatherProduct::El,
    WeatherProduct::EcapeLfc,
    WeatherProduct::EcapeEl,
    WeatherProduct::Srh01km,
    WeatherProduct::Srh03km,
    WeatherProduct::Stp,
    WeatherProduct::StpFixed,
    WeatherProduct::StpEffective,
    WeatherProduct::Scp,
    WeatherProduct::Ehi,
    WeatherProduct::Tehi,
    WeatherProduct::Tts,
    WeatherProduct::VtpMod,
    WeatherProduct::Uh,
    WeatherProduct::EcapeScpExperimental,
    WeatherProduct::EcapeEhi01kmExperimental,
    WeatherProduct::EcapeEhi03kmExperimental,
    WeatherProduct::EcapeStpExperimental,
];

/// Units for a weather-product or heavy template, which name none of their
/// own.  A derived recipe's template takes its product's units instead
/// ([`crate::derived::derived_product_units`]).
fn template_units_for_slug(slug: &str) -> &'static str {
    let slug = normalize_wrf_store_slug(slug);
    if slug.contains("cape") || slug.contains("cin") || slug.contains("ncape") || slug == "dcape" {
        "J/kg"
    } else if slug.contains("lcl") || slug.contains("lfc") || slug == "el" {
        "m"
    } else if slug.contains("srh") || slug.contains("uhel") || slug == "uh" {
        "m^2/s^2"
    } else if slug.contains("stp")
        || slug.contains("scp")
        || slug.contains("ehi")
        || slug.contains("tts")
        || slug.contains("vtp")
        || slug.contains("ratio")
    {
        "dimensionless"
    } else {
        ""
    }
}

fn direct_recipe_for_selector(
    var_name: &str,
    selector: FieldSelector,
    model: ModelId,
) -> Option<&'static PlotRecipe> {
    direct_recipe_for_selector_with_supported(
        var_name,
        selector,
        &style_recipe_slugs(model),
    )
    .or_else(|| {
        (model == ModelId::WrfGdex).then(|| {
            direct_recipe_for_selector_with_supported(
                var_name,
                selector,
                &style_recipe_slugs(ModelId::Hrrr),
            )
        })?
    })
}

// A stored canonical plane can use its registered production style even
// when the model's downloadable files do not carry that derived field.
fn style_recipe_slugs(model: ModelId) -> Vec<String> {
    let mut slugs=supported_direct_recipe_slugs(model);
    slugs.extend(crate::direct::store_direct_recipe_slugs());
    slugs
}

fn direct_recipe_for_selector_with_supported(
    var_name: &str,
    selector: FieldSelector,
    supported: &[String],
) -> Option<&'static PlotRecipe> {
    let is_supported = |slug: &str| supported.iter().any(|s| s == slug);
    if var_name == "apcp_run_total" {
        if let Some(recipe) =
            rustwx_models::plot_recipe("total_qpf").filter(|r| is_supported(r.slug))
        {
            return Some(recipe);
        }
    }
    built_in_plot_recipes()
        .iter()
        .find(|recipe| recipe.filled.selector == Some(selector) && is_supported(recipe.slug))
}

/// The render density + legend controls a single-product operational static
/// plot request ends with â€” read off a real request run through
/// `StaticPlotDesign` (the same code path every PNG lane applies), over a
/// regional domain like the production CONUS products.
fn operational_request_chrome() -> (RenderDensity, LegendControls) {
    let mut request = MapRenderRequest::from_core_field(
        probe_core_field(),
        ColorScale::Discrete(DiscreteColorScale {
            levels: vec![0.0, 1.0],
            colors: vec![rustwx_render::Color::rgba(0, 0, 0, 255)],
            extend: ExtendMode::Neither,
            mask_below: None,
        }),
    );
    apply_probe_static_design(&mut request);
    (request.render_density, request.legend)
}

/// Any non-global bounds select `apply_static_map_design`'s regional
/// branch, matching the production CONUS domains.
fn apply_probe_static_design(request: &mut MapRenderRequest) {
    crate::plot_design::StaticPlotDesign::new(
        (-125.0, -66.0, 24.0, 50.0),
        ProductVisualMode::FilledMeteorology,
    )
    .apply_to_request(request);
}

/// Filter the request's density through the active plot style exactly as
/// the renderer does when it builds the colormap
/// (`plot_style.render_density(request.render_density)`).
fn filtered_options(render_density: RenderDensity, legend: LegendControls) -> ColormapBuildOptions {
    ColormapBuildOptions {
        render_density: StaticPlotStyle::from_env().render_density(render_density),
        legend,
    }
}

fn probe_core_field() -> rustwx_core::Field2D {
    let shape = rustwx_core::GridShape::new(2, 2).expect("probe grid shape");
    let grid = rustwx_core::LatLonGrid::new(
        shape,
        vec![35.0, 35.0, 36.0, 36.0],
        vec![-100.0, -99.0, -100.0, -99.0],
    )
    .expect("probe grid");
    rustwx_core::Field2D::new(
        rustwx_core::ProductKey::named("style-probe"),
        "probe",
        grid,
        vec![0.0, 0.0, 0.0, 0.0],
    )
    .expect("probe field")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::derived::{store_derived_recipe_slugs, store_heavy_recipe_slugs};
    use rustwx_render::{LevelDensity, Rgba, build_colormap, colorbar_ticks};

    fn style_for(
        var_name: &str,
        selector: &FieldSelector,
        units: &str,
    ) -> Option<StoreVariableStyle> {
        operational_style_for_store_variable(
            var_name,
            &serde_json::to_value(selector).expect("selector json"),
            units,
            ModelId::Hrrr,
        )
    }

    fn derived_marker(slug: &str) -> serde_json::Value {
        serde_json::json!({ "derived": slug })
    }

    /// One representative per curated quantity class: the resolved style
    /// must be the borrowed OPERATIONAL one (variable-named title wearing a
    /// production palette), never the auto-ranged generic ramp.
    #[test]
    fn curated_mapping_assigns_operational_palettes_by_quantity() {
        let derived = derived_marker("anything");
        let model = ModelId::WrfGdex;
        let cases: [(&str, &str, UnitConvert, Option<&str>); 6] = [
            // cape-like -> CAPE table, raw J/kg values color as stored
            ("wrf_cape", "J/kg", UnitConvert::None, None),
            // temperature (K) -> 2 m temperature table (K -> degF)
            ("wrf_t2", "K", UnitConvert::KelvinToFahrenheit, None),
            // SRH (m2/s2) -> SRH table
            ("wrf_srh", "m2/s2", UnitConvert::None, None),
            // wind speed (m/s) -> 10 m wind speed table (m/s -> kt, the
            // production fill's own conversion riding along)
            ("wrf_wspd10", "m/s", UnitConvert::MsToKnots, Some("kt")),
            // precip depth (mm) -> QPF table (mm -> inches)
            ("wrf_snownc", "mm", UnitConvert::MmToInches, Some("in")),
            // shear magnitude (m/s) -> bulk shear table (m/s -> kt)
            ("wrf_ebwd", "m/s", UnitConvert::MsToKnots, Some("kt")),
        ];
        for (name, units, convert, display) in cases {
            let style = curated_style_for_store_variable(name, &derived, units, model)
                .unwrap_or_else(|| panic!("{name} [{units}] must resolve a curated style"));
            match curated_headline(name.strip_prefix("wrf_").unwrap_or(name)) {
                Some(headline) => assert_eq!(style.title, headline, "{name} headline"),
                None => assert!(
                    style.title.starts_with(name) && style.title.contains("palette"),
                    "{name}: borrowed identity must name the variable and the palette: {}",
                    style.title
                ),
            }
            assert!(
                !style.title.contains("(generic"),
                "{name}: curated style must not be the generic ramp: {}",
                style.title
            );
            assert_eq!(style.convert, convert, "{name} conversion");
            if let Some(display) = display {
                assert_eq!(style.display_units, display, "{name} display units");
            }
        }

        // degC inputs against degF/degC palettes keep the values accurate.
        let ctt = curated_style_for_store_variable("wrf_ctt", &derived, "degC", model).unwrap();
        assert_eq!(ctt.convert, UnitConvert::None, "ctt values are already degC");
        assert_eq!(ctt.display_units, "degC");
        let dp2m = curated_style_for_store_variable("wrf_dp2m", &derived, "degC", model).unwrap();
        assert_eq!(dp2m.convert, UnitConvert::CelsiusToFahrenheit);
        assert!((UnitConvert::CelsiusToFahrenheit.apply(20.0) - 68.0).abs() < 1.0e-4);
    }

    /// The WOOF city snow map (`var:wrf_snownc`) headlined
    /// "wrf_snownc (Total QPF palette)": it reads as the product page
    /// names it, on the same QPF palette and mm -> in conversion.
    #[test]
    fn accumulated_snow_graupel_hail_wear_plain_headlines() {
        let derived = derived_marker("anything");
        for (name, headline) in [
            ("wrf_snownc", "Snow since start, liquid equivalent"),
            ("wrf_graupelnc", "Graupel since start, liquid equivalent"),
            ("wrf_hailnc", "Hail since start, liquid equivalent"),
        ] {
            let style = curated_style_for_store_variable(name, &derived, "mm", ModelId::WrfGdex)
                .unwrap_or_else(|| panic!("{name} must resolve a curated style"));
            assert_eq!(style.title, headline);
            assert_eq!(style.convert, UnitConvert::MmToInches, "{name} conversion");
            assert_eq!(style.display_units, "in", "{name} display units");
        }
    }

    #[test]
    fn curated_level_gap_borrows_the_same_field_from_a_chart_level() {
        // Dewpoint has filled recipes at 700/850 but not 500: the 500 hPa
        // plane borrows the field's own palette from a level that has one.
        let selector =
            serde_json::to_value(FieldSelector::isobaric(CanonicalField::Dewpoint, 500)).unwrap();
        let style =
            curated_style_for_store_variable("dewpoint_500hpa", &selector, "K", ModelId::WrfGdex)
                .expect("chart-level gap must borrow the same field's style");
        assert!(style.title.starts_with("dewpoint_500hpa ("), "{}", style.title);
        assert_eq!(style.convert, UnitConvert::KelvinToCelsius);
    }

    #[test]
    fn curated_mapping_refuses_components_and_unknown_quantities() {
        let model = ModelId::WrfGdex;
        // Signed wind components have no production fill table; a curated
        // claim would color them with a magnitude palette. Refused.
        let u500 =
            serde_json::to_value(FieldSelector::isobaric(CanonicalField::UWind, 500)).unwrap();
        assert_eq!(
            curated_style_for_store_variable("u_wind_500hpa", &u500, "m/s", model),
            None
        );
        // Unknown quantity in unknown units: the generic ramp is correct.
        assert_eq!(
            curated_style_for_store_variable(
                "mystery_plane",
                &derived_marker("mystery_plane"),
                "widgets",
                model
            ),
            None
        );
        // Units gate the name match: a "cape" in the wrong units is not
        // CAPE and must not wear its table.
        assert_eq!(
            curated_style_for_store_variable("wrf_cape", &derived_marker("wrf_cape"), "m", model),
            None
        );
    }

    #[test]
    fn generic_style_spans_the_exact_finite_range_with_neutral_units() {
        let style = generic_style_for_store_variable("mystery_plane", "widgets", Some((-2.5, 7.5)));
        assert_eq!(style.title, "mystery_plane [widgets]");
        assert_eq!(style.display_units, "widgets");
        assert_eq!(style.convert, UnitConvert::None);
        assert_eq!(style.legend_mode, LegendMode::SmoothRamp);
        assert_eq!(style.colormap_options.legend.mode, LegendMode::SmoothRamp);
        assert_eq!(style.cbar_tick_step, None);

        let scale = style.scale.resolved_discrete();
        assert_eq!(scale.levels.first().copied(), Some(-2.5));
        assert_eq!(scale.levels.last().copied(), Some(7.5));
        assert_eq!(scale.levels.len(), 10);
        assert_eq!(scale.colors.len(), 9);
        assert!(scale.levels.windows(2).all(|pair| pair[0] < pair[1]));
        assert_eq!(scale.extend, ExtendMode::Neither);
        assert_eq!(scale.mask_below, None);
    }

    #[test]
    fn a_mixing_ratio_plane_gets_a_colorbar_whose_ticks_are_numbers() {
        // The measurement this pins: a real stored 2-D mixing-ratio plane,
        // units `kg kg-1`, finite range 1.0071096e-3 to 3.8782053e-3, read
        // through the store reader and styled through this function. Before
        // the decade shift the panel drew fourteen ticks and labelled all
        // fourteen `0`.
        let style = generic_style_for_store_variable(
            "a_mixing_ratio_plane",
            "kg kg-1",
            Some((1.0071096e-3, 3.8782053e-3)),
        );
        // A mass mixing ratio goes to grams per kilogram first, keyed on
        // its units; the range 1.0 to 3.9 g kg-1 then needs no decade.
        assert_eq!(style.display_units, "g kg-1");
        assert_eq!(style.title, "a_mixing_ratio_plane [g kg-1]");
        assert_eq!(style.convert, UnitConvert::ScaleByDecade(-3));

        let cmap = rustwx_render::build_colormap(&style.scale, style.colormap_options);
        let labels = rustwx_render::format_tick_labels(&rustwx_render::colorbar_ticks(
            &cmap,
            style.cbar_tick_step,
        ));
        assert!(
            !labels.is_empty(),
            "a colorbar with no tick reports nothing either"
        );
        assert!(
            labels.iter().any(|label| label != "0"),
            "every tick still reads zero, so the bar measures nothing: {labels:?}"
        );
        assert!(
            labels.iter().collect::<std::collections::BTreeSet<_>>().len() == labels.len(),
            "two ticks that print the same string are one tick: {labels:?}"
        );
    }

    #[test]
    fn every_production_colour_bar_labels_its_ticks_with_their_own_values() {
        // Every label must parse back to its tick to within a hundredth of
        // the tick spacing. Where the usual one-decimal labels already do
        // that, the bar keeps them unchanged; a quarter-step table (0,
        // 0.25, 0.5, ...) used to print 0.25 as `0.2` and 0.75 as `0.8`.
        let templates = operational_style_templates(ModelId::WrfGdex);
        assert!(!templates.is_empty());
        let reads = |labels: &[String], ticks: &[f64], tolerance: f64| {
            labels.iter().zip(ticks).all(|(label, tick)| {
                label
                    .parse::<f64>()
                    .is_ok_and(|shown| (shown - tick).abs() <= tolerance)
            })
        };
        for template in templates {
            let style = &template.style;
            let cmap = rustwx_render::build_colormap(&style.scale, style.colormap_options);
            let ticks = rustwx_render::colorbar_ticks(&cmap, style.cbar_tick_step);
            let spacing = ticks
                .windows(2)
                .map(|pair| (pair[1] - pair[0]).abs())
                .filter(|gap| *gap > 0.0)
                .fold(f64::INFINITY, f64::min);
            let labels = rustwx_render::format_tick_labels(&ticks);
            let usual: Vec<String> =
                ticks.iter().map(|tick| rustwx_render::format_tick(*tick)).collect();
            let context = format!("{} ({}) ticks {ticks:?}", template.id, template.label);
            if !spacing.is_finite() {
                // Fewer than two different ticks: nothing to tell apart.
                assert_eq!(labels, usual, "{context}");
                continue;
            }
            let tolerance = spacing * 0.01;
            assert!(reads(&labels, &ticks, tolerance), "{context}: {labels:?}");
            if reads(&usual, &ticks, tolerance) {
                assert_eq!(labels, usual, "{context}");
            }
        }
    }

    #[test]
    fn a_narrow_generic_plane_gets_a_colour_bar_whose_ticks_differ() {
        // Measured on a real 3 km forecast frame: 200 hPa height spans
        // 12200.066 to 12228.032 gpm; the bar is drawn against `1e3 gpm`
        // and printed `12.2` at all fourteen ticks.
        let style = generic_style_for_store_variable(
            "geopotential_height_200hpa",
            "gpm",
            Some((12200.066, 12228.032)),
        );
        assert_eq!(style.display_units, "1e3 gpm");
        let cmap = rustwx_render::build_colormap(&style.scale, style.colormap_options);
        let ticks = rustwx_render::colorbar_ticks(&cmap, style.cbar_tick_step);
        let labels = rustwx_render::format_tick_labels(&ticks);
        assert_eq!(labels.len(), 14, "{labels:?}");
        assert_eq!(labels.first().map(String::as_str), Some("12.202"));
        assert_eq!(labels.last().map(String::as_str), Some("12.228"));
        assert_eq!(
            labels.iter().collect::<std::collections::BTreeSet<_>>().len(),
            labels.len(),
            "two ticks that print the same string are one tick: {labels:?}"
        );
    }

    #[test]
    fn the_convert_puts_a_value_where_its_own_levels_are() {
        let style = generic_style_for_store_variable(
            "a_mixing_ratio_plane",
            "kg kg-1",
            Some((1.0071096e-3, 3.8782053e-3)),
        );
        let scale = style.scale.resolved_discrete();
        let lowest = style.convert.apply(1.0071096e-3);
        let highest = style.convert.apply(3.8782053e-3);
        assert!(
            (f64::from(lowest) - scale.levels.first().copied().unwrap()).abs() < 1.0e-6,
            "the smallest value has to land on the lowest level: {lowest}"
        );
        assert!(
            (f64::from(highest) - scale.levels.last().copied().unwrap()).abs() < 1.0e-6,
            "the largest value has to land on the highest level: {highest}"
        );
    }

    #[test]
    fn a_range_already_in_one_to_a_thousand_keeps_its_units_and_its_values() {
        for (units, range) in [
            ("K", (271.4_f32, 302.8_f32)),
            ("dBZ", (1.5_f32, 62.0_f32)),
            ("widgets", (-2.5_f32, 7.5_f32)),
        ] {
            let style = generic_style_for_store_variable("plane", units, Some(range));
            assert_eq!(style.display_units, units, "{units} moved decade");
            assert_eq!(style.convert, UnitConvert::None, "{units} gained a convert");
            let scale = style.scale.resolved_discrete();
            assert_eq!(scale.levels.first().copied(), Some(f64::from(range.0)));
            assert_eq!(scale.levels.last().copied(), Some(f64::from(range.1)));
        }
    }

    #[test]
    fn the_decade_is_read_off_the_range_and_never_off_the_name() {
        // Same numbers, four names: a name cannot buy or lose a decade.
        let mut seen = Vec::new();
        for name in ["a_condensate_plane", "smoke", "x", "reflectivity"] {
            let style =
                generic_style_for_store_variable(name, "kg kg-1", Some((2.0e-7, 8.3e-7)));
            seen.push((style.display_units.clone(), style.convert));
        }
        assert!(
            seen.windows(2).all(|pair| pair[0] == pair[1]),
            "the decade moved with the name: {seen:?}"
        );
        // 2e-7 to 8.3e-7 kg kg-1 is 2e-4 to 8.3e-4 g kg-1, which still
        // needs a power of a thousand: 200 to 830 against 1e-6 g kg-1,
        // and the convert carries the thousand into grams as well.
        assert_eq!(seen[0].0, "1e-6 g kg-1");
        assert_eq!(seen[0].1, UnitConvert::ScaleByDecade(-9));
    }

    #[test]
    fn a_kilogram_per_kilogram_plane_is_labelled_in_grams_per_kilogram() {
        // Every spelling a file carries, and only those.
        for units in ["kg kg-1", "kg kg^-1", "kg kg^{-1}", "kg/kg", "kg kg**-1", "KG KG-1"] {
            assert_eq!(grams_per_kilogram(units), Some((1000.0, "g kg-1")), "{units}");
            let style = generic_style_for_store_variable("q", units, Some((1.0e-3, 4.0e-3)));
            assert_eq!(style.display_units, "g kg-1", "{units}");
            assert_eq!(style.convert, UnitConvert::ScaleByDecade(-3));
            assert!((f64::from(style.convert.apply(4.0e-3)) - 4.0).abs() < 1e-6);
            let scale = style.scale.resolved_discrete();
            assert!((scale.levels.first().copied().unwrap() - 1.0).abs() < 1e-6);
            assert!((scale.levels.last().copied().unwrap() - 4.0).abs() < 1e-6);
        }
        for units in ["g kg-1", "kg m-2", "kg/m^2", "K", "", "kg kg-2"] {
            assert_eq!(grams_per_kilogram(units), None, "{units}");
        }
        // The direct route's convert for a catalog row is the same thousand.
        assert_eq!(UnitConvert::KgPerKgToGPerKg.apply(2.5e-3), 2.5);
    }

    #[test]
    fn the_prescaled_entry_never_takes_a_decade_of_its_own() {
        // A caller that has already moved its values must get levels over
        // exactly the range it stated.  Every one of these would shift
        // under the auto entry.
        for (units, range, top) in [
            ("1e-6 kg kg-1", (0.1_f32, 1.0_f32), 1.0_f64),
            ("kg kg-1", (1.0071096e-3_f32, 3.8782053e-3_f32), 3.8782053e-3_f64),
            ("m", (2.0e4_f32, 4.2e7_f32), 4.2e7_f64),
        ] {
            let style = generic_style_for_prescaled_store_variable("plane", units, Some(range));
            assert_eq!(style.convert, UnitConvert::None, "{units} gained a convert");
            assert_eq!(style.display_units, units, "{units} moved decade");
            assert_eq!(style.title, format!("plane [{units}]"));
            let scale = style.scale.resolved_discrete();
            assert_eq!(scale.levels.first().copied(), Some(f64::from(range.0)));
            assert!(
                (scale.levels.last().copied().unwrap() - top).abs() <= top.abs() * 1.0e-6,
                "{units}: top level {:?} is not the range's own top {top}",
                scale.levels.last()
            );
        }
    }

    #[test]
    fn the_two_entries_agree_wherever_no_decade_is_owed() {
        // Same numbers through both doors when the range is already in
        // 1-1000: the split may not have moved the shipped style.
        let auto = generic_style_for_store_variable("plane", "dBZ", Some((1.5, 62.0)));
        let prescaled = generic_style_for_prescaled_store_variable("plane", "dBZ", Some((1.5, 62.0)));
        assert_eq!(auto.title, prescaled.title);
        assert_eq!(auto.display_units, prescaled.display_units);
        assert_eq!(auto.convert, prescaled.convert);
        assert_eq!(
            auto.scale.resolved_discrete().levels,
            prescaled.scale.resolved_discrete().levels
        );
    }

    #[test]
    fn generic_style_handles_constant_and_absent_ranges_deterministically() {
        let constant = generic_style_for_store_variable("constant", "K", Some((300.0, 300.0)));
        let constant_scale = constant.scale.resolved_discrete();
        assert_eq!(constant.title, "constant [K]");
        assert_eq!(constant_scale.levels.first().copied(), Some(285.0));
        assert_eq!(constant_scale.levels.last().copied(), Some(315.0));

        let zero = generic_style_for_store_variable("zero", "1", Some((0.0, 0.0)));
        let zero_scale = zero.scale.resolved_discrete();
        assert_eq!(zero_scale.levels.first().copied(), Some(-1.0));
        assert_eq!(zero_scale.levels.last().copied(), Some(1.0));

        for range in [None, Some((f32::NAN, 2.0)), Some((2.0, -2.0))] {
            let missing = generic_style_for_store_variable("missing", "", range);
            let missing_scale = missing.scale.resolved_discrete();
            assert_eq!(
                missing.title, "missing",
                "empty units carry no bracket segment"
            );
            assert_eq!(missing_scale.levels.first().copied(), Some(0.0));
            assert_eq!(missing_scale.levels.last().copied(), Some(1.0));
        }
    }

    #[test]
    fn every_store_derived_and_heavy_slug_resolves() {
        for slug in store_derived_recipe_slugs()
            .into_iter()
            .chain(store_heavy_recipe_slugs())
        {
            // The units the store holds each grid in when its writer
            // stores the product's own units.
            let units = crate::derived::derived_product_units(slug).unwrap_or("units");
            let style = operational_style_for_store_variable(
                slug,
                &derived_marker(slug),
                units,
                ModelId::Hrrr,
            )
            .unwrap_or_else(|| panic!("derived slug '{slug}' must resolve"));
            assert!(!style.title.is_empty(), "'{slug}' carries a title");
            assert!(
                style.convert.is_none(),
                "a grid stored in its product's units draws as stored ('{slug}')"
            );
            assert_eq!(style.display_units, units, "'{slug}' is labelled as stored");
        }
    }

    #[test]
    fn stored_shear_in_metres_per_second_wears_the_knot_palette_in_knots() {
        for slug in ["bulk_shear_0_1km", "bulk_shear_0_6km", "wrf_shear_0_6km"] {
            let style = operational_style_for_store_variable(
                slug,
                &derived_marker(slug),
                "m/s",
                ModelId::WrfGdex,
            )
            .unwrap_or_else(|| panic!("'{slug}' in m/s must resolve"));
            assert_eq!(style.display_units, "kt", "{slug}");
            assert_eq!(style.convert, UnitConvert::MsToKnots, "{slug}");
            assert!((style.convert.apply(22.689_922) - 44.105_68).abs() < 1.0e-3);
            let knots = operational_style_for_store_variable(
                slug,
                &derived_marker(slug),
                "kt",
                ModelId::WrfGdex,
            )
            .expect("shear in knots resolves");
            assert_eq!(knots.convert, UnitConvert::None, "knots are not converted twice");
            assert_eq!(knots.scale, style.scale, "one palette whatever the stored unit");
        }
    }

    #[test]
    fn every_derived_template_is_labelled_in_its_palettes_units() {
        let templates = operational_style_templates(ModelId::Hrrr);
        for slug in store_derived_recipe_slugs() {
            let template = templates
                .iter()
                .find(|template| template.id == format!("derived:{slug}"))
                .unwrap_or_else(|| panic!("no template for derived '{slug}'"));
            assert_eq!(
                Some(template.style.display_units.as_str()),
                crate::derived::derived_product_units(slug),
                "'{slug}' template"
            );
        }
    }

    #[test]
    fn a_derived_plane_in_a_foreign_unit_claims_no_production_palette() {
        for (slug, units) in [("bulk_shear_0_6km", "K"), ("sbcape", "m/s"), ("srh_0_1km", "kt")] {
            assert!(
                operational_style_for_store_variable(
                    slug,
                    &derived_marker(slug),
                    units,
                    ModelId::WrfGdex,
                )
                .is_none(),
                "'{slug}' stored in {units} must not borrow its product's palette"
            );
        }
    }

    #[test]
    fn sbcape_resolves_to_masked_cape_preset_with_production_ticks() {
        let style = operational_style_for_store_variable(
            "sbcape",
            &derived_marker("sbcape"),
            "J/kg",
            ModelId::Hrrr,
        )
        .expect("sbcape resolves");
        // The derived lane overrides the CAPE preset with mask_below 250
        // (apply_operational_raster_scale) and ticks every 500 J/kg.
        let discrete = style.scale.resolved_discrete();
        assert_eq!(discrete.mask_below, Some(250.0));
        assert_eq!(style.cbar_tick_step, Some(500.0));
        let cmap = build_colormap(&style.scale, style.colormap_options);
        let ticks = colorbar_ticks(&cmap, style.cbar_tick_step);
        assert_eq!(ticks.first().copied(), Some(discrete.levels[0]));
        assert!(ticks.windows(2).all(|w| (w[1] - w[0] - 500.0).abs() < 1e-9));
        // Masked (below 250) and NaN values are transparent, never clamped.
        assert_eq!(cmap.map(100.0), Rgba::TRANSPARENT);
        assert_eq!(cmap.map(f64::NAN), Rgba::TRANSPARENT);
    }

    #[test]
    fn wrf_prefixed_derived_markers_reuse_product_palettes() {
        let sbcape = operational_style_for_store_variable(
            "wrf_sbcape",
            &derived_marker("wrf_sbcape"),
            "J/kg",
            ModelId::Hrrr,
        )
        .expect("legacy wrf_sbcape resolves");
        assert_eq!(sbcape.title, "SBCAPE");
        assert_eq!(
            sbcape.scale.resolved_discrete().mask_below,
            Some(250.0),
            "wrf_sbcape must use the same CAPE masking as sbcape"
        );

        let srh = operational_style_for_store_variable(
            "wrf_srh1",
            &derived_marker("wrf_srh1"),
            "m2/s2",
            ModelId::Hrrr,
        )
        .expect("legacy wrf_srh1 resolves");
        assert_eq!(srh.title, "0-1 km SRH");
        assert_eq!(srh.cbar_tick_step, Some(50.0));

        let shear = operational_style_for_store_variable(
            "wrf_shear_0_6km",
            &derived_marker("wrf_shear_0_6km"),
            "m/s",
            ModelId::Hrrr,
        )
        .expect("legacy wrf_shear_0_6km resolves");
        assert_eq!(shear.title, "0-6 km Bulk Shear");

        let stp = operational_style_for_store_variable(
            "wrf_stp",
            &derived_marker("wrf_stp"),
            "dimensionless",
            ModelId::Hrrr,
        )
        .expect("legacy wrf_stp resolves via WeatherProduct");
        assert_eq!(stp.title, "STP");
        assert!(matches!(stp.scale, ColorScale::Weather(_)));
    }

    #[test]
    fn operational_template_catalog_exposes_common_starting_palettes() {
        let templates = operational_style_templates(ModelId::Hrrr);
        assert!(
            templates
                .iter()
                .any(|template| template.slug == "2m_temperature"),
            "template catalog must include the operational 2 m temperature palette"
        );
        assert!(
            templates
                .iter()
                .any(|template| template.slug == "2m_dewpoint"),
            "template catalog must include the operational dewpoint palette"
        );
        assert!(
            templates.iter().any(|template| template.slug == "stp"),
            "template catalog must include an STP palette"
        );
    }

    #[test]
    fn heavy_slugs_use_the_weather_preset_lane_without_densification() {
        let style = operational_style_for_store_variable(
            "sbecape",
            &derived_marker("sbecape"),
            "J/kg",
            ModelId::Hrrr,
        )
        .expect("sbecape resolves");
        assert!(
            matches!(style.scale, ColorScale::Weather(_)),
            "heavy lane keeps the Weather preset scale"
        );
        assert_eq!(style.cbar_tick_step, Some(500.0));
        assert_eq!(style.legend_mode, LegendMode::Stepped);
        // for_weather_product's reference-discrete defaults: no fill/palette
        // densification requested (the plot style may still bump it; compare
        // against the identically-filtered reference request).
        let reference = filtered_options(
            RenderDensity {
                fill: LevelDensity::default(),
                palette_multiplier: 1,
            },
            LegendControls {
                density: LevelDensity::default(),
                mode: LegendMode::Stepped,
            },
        );
        assert_eq!(style.colormap_options, reference);
    }

    #[test]
    fn direct_planes_resolve_with_production_conversions_and_controls() {
        let temp = style_for(
            "temperature_2m",
            &FieldSelector::height_agl(CanonicalField::Temperature, 2),
            "K",
        )
        .expect("temperature_2m resolves");
        assert_eq!(temp.convert, UnitConvert::KelvinToFahrenheit);
        assert_eq!(temp.display_units, "degF");
        assert_eq!(temp.cbar_tick_step, None);
        assert_eq!(temp.legend_mode, LegendMode::SmoothRamp);
        let discrete = temp.scale.resolved_discrete();
        assert_eq!(discrete.levels.first().copied(), Some(-60.0));
        assert_eq!(discrete.levels.last().copied(), Some(120.0));

        let dewpoint = style_for(
            "dewpoint_2m",
            &FieldSelector::height_agl(CanonicalField::Dewpoint, 2),
            "K",
        )
        .expect("dewpoint_2m resolves");
        assert_eq!(dewpoint.cbar_tick_step, Some(10.0));
        assert_eq!(dewpoint.legend_mode, LegendMode::Stepped);
        assert_eq!(
            dewpoint.colormap_options.legend.mode,
            LegendMode::Stepped,
            "the legend-mode override must reach the colormap options"
        );

        let rh = style_for(
            "rh_2m",
            &FieldSelector::height_agl(CanonicalField::RelativeHumidity, 2),
            "%",
        )
        .expect("rh_2m resolves");
        assert_eq!(rh.cbar_tick_step, Some(25.0));
        assert_eq!(rh.display_units, "%");
        assert_eq!(rh.convert, UnitConvert::None);

        let reflectivity = style_for(
            "composite_reflectivity",
            &FieldSelector::entire_atmosphere(CanonicalField::CompositeReflectivity),
            "dBZ",
        )
        .expect("composite_reflectivity resolves");
        let scale = reflectivity.scale.resolved_discrete();
        assert_eq!(scale.levels.first().copied(), Some(10.0));
        // Re-recorded 2026-10-03: the reflectivity products moved to the
        // radar reflectivity table, which runs to 85 dBZ (was 70).
        assert_eq!(scale.levels.last().copied(), Some(85.0));
        assert_eq!(scale.mask_below, Some(10.0));
    }

    #[test]
    fn wrf_gdex_direct_planes_resolve_standard_product_palettes() {
        let temp = operational_style_for_store_variable(
            "temperature_2m",
            &serde_json::to_value(FieldSelector::height_agl(CanonicalField::Temperature, 2))
                .expect("selector json"),
            "K",
            ModelId::WrfGdex,
        )
        .expect("WRF 2m temperature should use the standard temperature palette");
        assert_eq!(temp.convert, UnitConvert::KelvinToFahrenheit);
        assert_eq!(temp.display_units, "degF");
    }

    #[test]
    fn mslp_falls_back_to_the_generic_ramp() {
        // The production `mslp_10m_winds` plot fills the companion 10 m wind
        // speed (legend 10..60 kt) and only contours mslp â€” there is no
        // production colorbar for the stored pressure values, so claiming
        // production parity with ANY pressure-valued legend would be false.
        assert!(
            style_for(
                "mslp",
                &FieldSelector::mean_sea_level(CanonicalField::PressureReducedToMeanSeaLevel),
                "Pa",
            )
            .is_none(),
            "mslp must keep the generic ramp (no production fill counterpart)"
        );
    }

    #[test]
    fn windowed_source_planes_resolve_through_the_windowed_family() {
        let uh = style_for(
            "uh_2to5km_max_1h",
            &FieldSelector::height_layer_agl(CanonicalField::UpdraftHelicity, 2000, 5000),
            "m^2/s^2",
        )
        .expect("uh_2to5km_max_1h resolves");
        assert!(matches!(uh.scale, ColorScale::Weather(_)));
        assert_eq!(
            uh.cbar_tick_step,
            WeatherProduct::Uh.default_tick_step(),
            "windowed UH carries the UH product tick step"
        );

        let wind = style_for(
            "wind_speed_10m_max_1h",
            &FieldSelector::height_agl(CanonicalField::WindSpeed, 10),
            "m/s",
        )
        .expect("wind_speed_10m_max_1h resolves");
        assert_eq!(wind.convert, UnitConvert::MsToKnots);
        assert_eq!(wind.display_units, "kt");
        let scale = wind.scale.resolved_discrete();
        assert_eq!(scale.levels.first().copied(), Some(10.0));
        assert_eq!(scale.levels.last().copied(), Some(70.0));
    }

    #[test]
    fn qpf_planes_pin_their_window_identity_by_store_name() {
        let selector = FieldSelector::surface(CanonicalField::TotalPrecipitation);
        let total = style_for("apcp_run_total", &selector, "kg/m^2").expect("run total resolves");
        let hourly = style_for("apcp_1h", &selector, "kg/m^2").expect("1h resolves");
        assert_eq!(total.convert, UnitConvert::MmToInches);
        assert_eq!(total.display_units, "in");
        assert_eq!(
            total.scale, hourly.scale,
            "both windows share the QPF scale"
        );
        assert_eq!(total.title, "Total QPF");
        assert_eq!(hourly.title, "1-h QPF", "titles pin the window identity");
    }

    #[test]
    fn unmapped_variables_fall_back_to_none() {
        // Barb inputs, compute inputs, contour-only heights, and the
        // contour-only mslp plane have no production fill counterpart.
        for (name, selector) in [
            (
                "u_10m",
                FieldSelector::height_agl(CanonicalField::UWind, 10),
            ),
            (
                "v_10m",
                FieldSelector::height_agl(CanonicalField::VWind, 10),
            ),
            (
                "surface_pressure",
                FieldSelector::surface(CanonicalField::Pressure),
            ),
            (
                "mslp",
                FieldSelector::mean_sea_level(CanonicalField::PressureReducedToMeanSeaLevel),
            ),
            (
                "geopotential_height_500hpa",
                FieldSelector::isobaric(CanonicalField::GeopotentialHeight, 500),
            ),
            (
                "u_wind_500hpa",
                FieldSelector::isobaric(CanonicalField::UWind, 500),
            ),
        ] {
            assert!(
                style_for(name, &selector, "units").is_none(),
                "'{name}' must keep the generic ramp"
            );
        }
        // `orography` used to be on that list, for want of a production
        // colorbar.  It has one now, and the browser must show it rather
        // than the generic ramp -- this assertion is what stops the
        // isobaric-height exclusion from silently swallowing it again.
        let terrain = style_for(
            "orography",
            &FieldSelector::surface(CanonicalField::GeopotentialHeight),
            "m",
        )
        .expect("orography now has a production fill");
        assert_eq!(terrain.title, "Terrain Height");
        assert_eq!(terrain.display_units, "m");

        // Unknown derived markers fall back too.
        assert!(
            operational_style_for_store_variable(
                "mystery",
                &serde_json::json!({ "derived": "not_a_recipe" }),
                "units",
                ModelId::Hrrr,
            )
            .is_none()
        );
    }

    #[test]
    fn shared_filled_selectors_agree_on_style_and_scale() {
        // `direct_recipe_for_selector` resolves a stored selector by
        // FIRST-MATCH over the supported recipe catalog. That is only safe
        // while every pair of supported recipes sharing a filled selector
        // agrees on render style and operational fill scale â€” otherwise the
        // viewer would silently pick whichever recipe happens to come first.
        // Pin the invariant over ALL pairs (zero offending pairs required;
        // the sweep guards future catalog additions even if no pair exists
        // today).
        let supported = supported_direct_recipe_slugs(ModelId::Hrrr);
        let recipes: Vec<&PlotRecipe> = built_in_plot_recipes()
            .iter()
            .filter(|recipe| supported.iter().any(|slug| slug == recipe.slug))
            .collect();
        assert!(
            !recipes.is_empty(),
            "HRRR must support at least one direct recipe"
        );
        let mut shared_pairs = 0usize;
        for (i, a) in recipes.iter().enumerate() {
            for b in &recipes[i + 1..] {
                let (Some(selector), Some(other)) = (a.filled.selector, b.filled.selector) else {
                    continue;
                };
                if selector != other {
                    continue;
                }
                shared_pairs += 1;
                assert_eq!(
                    a.style, b.style,
                    "supported recipes '{}' and '{}' share filled selector {selector:?} but \
                     disagree on render style â€” first-match resolution in \
                     direct_recipe_for_selector is no longer safe",
                    a.slug, b.slug,
                );
                assert_eq!(
                    crate::plot_design::operational_fill_scale_for_recipe(a, selector),
                    crate::plot_design::operational_fill_scale_for_recipe(b, selector),
                    "supported recipes '{}' and '{}' share filled selector {selector:?} but \
                     disagree on operational fill scale â€” first-match resolution in \
                     direct_recipe_for_selector is no longer safe",
                    a.slug,
                    b.slug,
                );
            }
        }
        eprintln!(
            "shared_filled_selectors_agree_on_style_and_scale: checked {shared_pairs} \
             shared-selector pair(s) across {} supported recipes",
            recipes.len()
        );
    }

    #[test]
    fn unit_conversions_match_the_direct_lane_arithmetic() {
        assert_eq!(UnitConvert::KelvinToFahrenheit.apply(273.15), 32.0);
        assert_eq!(
            UnitConvert::KelvinToFahrenheit.apply(300.0),
            (300.0 - 273.15) * 9.0 / 5.0 + 32.0
        );
        assert_eq!(UnitConvert::KelvinToCelsius.apply(273.15), 0.0);
        assert_eq!(UnitConvert::PaToHpa.apply(101_325.0), 101_325.0 * 0.01);
        assert_eq!(UnitConvert::MmToInches.apply(25.4), 1.0);
        assert_eq!(UnitConvert::MsToKnots.apply(10.0), 10.0_f32 * 1.943_844_5);
        assert_eq!(UnitConvert::KgM3ToUgM3.apply(1.0e-9), 1.0);
        assert!(UnitConvert::KelvinToFahrenheit.apply(f32::NAN).is_nan());
    }
}

/// Category planes through the generic `var:` style, drawn the way the store
/// route draws them: the style's scale, legend, density and tick step on a
/// projected request.  Written against the style resolver's existing entry
/// points only, so it states the behaviour rather than the mechanism.
#[cfg(test)]
mod category_plane_regression {
    use super::*;
    use rustwx_core::{Field2D, GridShape, LatLonGrid, ProductKey};
    use rustwx_render::{
        ProjectedDomain, ProjectedExtent, build_colormap, colorbar_ticks, legend_color_at_rel,
        legend_tick_rel, render_image,
    };

    /// The colormap the renderer builds for `style` under the active plot
    /// style, exactly as the render path builds it.
    fn rendered_colormap(style: &StoreVariableStyle) -> rustwx_render::LeveledColormap {
        build_colormap(
            &style.scale,
            ColormapBuildOptions {
                render_density: StaticPlotStyle::from_env()
                    .render_density(style.colormap_options.render_density),
                legend: style.colormap_options.legend,
            },
        )
    }

    fn rgba(color: rustwx_render::Rgba) -> [u8; 4] {
        [color.r, color.g, color.b, color.a]
    }

    #[test]
    fn a_soil_plane_holding_two_codes_draws_no_third() {
        // Codes 2 and 14 only, in a checkerboard, on a regular mesh and on a
        // skewed one: every code from 3 to 13 on the map was invented.
        for skew in [0.0, 0.35] {
            let (ny, nx) = (4usize, 5usize);
            let values: Vec<f32> = (0..ny * nx)
                .map(|cell| if (cell / nx + cell % nx) % 2 == 0 { 2.0 } else { 14.0 })
                .collect();
            let lat: Vec<f32> = (0..ny * nx).map(|cell| 40.0 + (cell / nx) as f32).collect();
            let lon: Vec<f32> = (0..ny * nx).map(|cell| -100.0 + (cell % nx) as f32).collect();
            let grid = LatLonGrid::new(GridShape::new(nx, ny).unwrap(), lat, lon).unwrap();
            let field =
                Field2D::new(ProductKey::named("var_wrf_isltyp"), "", grid, values).unwrap();
            let style = generic_style_for_store_variable("wrf_isltyp", "", Some((2.0, 14.0)));
            let mut request = MapRenderRequest::from_core_field(field, style.scale.clone());
            request.width = 420;
            request.height = 360;
            request.colorbar = false;
            request.cbar_tick_step = style.cbar_tick_step;
            request.render_density = style.colormap_options.render_density;
            request.legend = style.colormap_options.legend;
            request.projected_domain = Some(ProjectedDomain {
                x: (0..ny * nx)
                    .map(|cell| (cell % nx) as f64 + skew * (cell / nx) as f64)
                    .collect(),
                y: (0..ny * nx).map(|cell| (cell / nx) as f64).collect(),
                extent: ProjectedExtent {
                    x_min: 0.0,
                    x_max: (nx - 1) as f64 + skew * (ny - 1) as f64,
                    y_min: 0.0,
                    y_max: (ny - 1) as f64,
                },
            });
            let image = render_image(&request).unwrap();
            let cmap = rendered_colormap(&style);
            let held = [rgba(cmap.map(2.0)), rgba(cmap.map(14.0))];
            let absent: Vec<[u8; 4]> = (3..=13)
                .map(|code| rgba(cmap.map(f64::from(code))))
                .filter(|color| !held.contains(color))
                .collect();
            assert!(!absent.is_empty());
            let invented = image.pixels().filter(|px| absent.contains(&px.0)).count();
            assert_eq!(invented, 0, "codes 3 to 13 drawn at skew {skew}");
            for color in held {
                assert!(image.pixels().any(|px| px.0 == color), "a held code vanished");
            }
        }
    }

    #[test]
    fn category_legends_label_codes_in_their_own_map_colours() {
        for (name, range) in [
            ("wrf_landmask", (0.0_f32, 1.0_f32)),
            ("wrf_lu_index", (1.0, 21.0)),
            ("wrf_ivgtyp", (1.0, 17.0)),
            ("wrf_isltyp", (2.0, 14.0)),
        ] {
            let style = generic_style_for_store_variable(name, "", Some(range));
            assert!(style.convert.is_none(), "{name}");
            let cmap = rendered_colormap(&style);
            let codes: Vec<f64> =
                (range.0 as i32..=range.1 as i32).map(f64::from).collect();
            // Every code is a tick, and nothing between codes is.
            assert_eq!(colorbar_ticks(&cmap, style.cbar_tick_step), codes, "{name}");
            let mut seen: Vec<[u8; 4]> = Vec::new();
            for &code in &codes {
                let fill = cmap.map(code);
                let rel = legend_tick_rel(&cmap, code).unwrap();
                assert_eq!(
                    legend_color_at_rel(&cmap, style.legend_mode, rel),
                    fill,
                    "{name} code {code}: the bar shows a colour the map does not"
                );
                assert!(!seen.contains(&rgba(fill)), "{name} code {code} shares a colour");
                seen.push(rgba(fill));
            }
        }
    }

    #[test]
    fn continuous_planes_and_ranges_that_are_not_codes_keep_the_ramp() {
        let ramp = generic_style_for_store_variable("wrf_pblh", "m", Some((12.0, 1850.0)));
        assert_eq!(ramp.legend_mode, LegendMode::SmoothRamp);
        // A land-use name whose extremes are not whole codes is not
        // carrying codes, and neither is one spanning more than 256 of them.
        for range in [(1.0_f32, 20.5_f32), (0.0, 400.0)] {
            let style = generic_style_for_store_variable("wrf_lu_index", "", Some(range));
            assert_eq!(style.legend_mode, LegendMode::SmoothRamp, "{range:?}");
        }
    }
}

/// Snow level (wet-bulb 0 C height, ft MSL): 500 ft bins from 2,000 to
/// 10,000 ft, purple-blue where snow reaches the plains, through green and
/// yellow to red where only the high peaks are cold enough.
fn snow_level_style() -> StoreVariableStyle {
    let levels: Vec<f64> = (0..=16).map(|i| 2000.0 + 500.0 * i as f64).collect();
    let colors: Vec<Color> = [
        (0x54, 0x27, 0x88), (0x6A, 0x51, 0xA3), (0x3F, 0x4F, 0xBF), (0x21, 0x66, 0xAC),
        (0x43, 0x93, 0xC3), (0x92, 0xC5, 0xDE), (0x1B, 0x78, 0x37), (0x5A, 0xAE, 0x61),
        (0xA6, 0xDB, 0xA0), (0xFF, 0xFF, 0xBF), (0xFE, 0xE0, 0x90), (0xFD, 0xAE, 0x61),
        (0xF4, 0x6D, 0x43), (0xD7, 0x30, 0x27), (0xA5, 0x00, 0x26), (0x7F, 0x00, 0x1A),
        (0x67, 0x00, 0x0D),
    ]
    .iter()
    .map(|&(r, g, b)| Color::rgba(r, g, b, 255))
    .collect();
    let scale = DiscreteColorScale { levels, colors, extend: ExtendMode::Max, mask_below: None };
    let mut request = MapRenderRequest::from_core_field(probe_core_field(), ColorScale::Discrete(scale));
    apply_probe_static_design(&mut request);
    request.legend.mode = LegendMode::Thresholds;
    StoreVariableStyle {
        title: "Snow level guide: wet-bulb 0 C height".to_string(),
        display_units: "ft MSL".to_string(),
        convert: UnitConvert::None,
        scale: request.scale,
        colormap_options: filtered_options(request.render_density, request.legend),
        cbar_tick_step: request.cbar_tick_step,
        legend_mode: request.legend.mode,
    }
}
