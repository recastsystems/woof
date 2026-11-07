use std::collections::HashMap;
use std::time::Instant;

use rustwx_core::{
    CanonicalField, FieldProduct, FieldSelector, SelectedField2D, SourceId, VerticalSelector,
};
use rustwx_models::{PlotRecipe, RenderStyle};
use rustwx_render::{
    Color, ColorScale, ContourLayer, DiscreteColorScale, ExtendMode, LegendMode, LevelDensity,
    MapRenderRequest, ProductVisualMode, ProjectedContourLineStyle, ProjectedDomain, ProjectedMap,
    RasterSampleMode, WindBarbLayer, WindStreamlineLayer, build_projected_contour_geometry_profile,
    densify_discrete_scale,
};

use crate::derived::NativeContourRenderMode;
use crate::shared_context::{
    static_chrome_scale, static_supersample_factor, static_supersample_sharpen,
    static_title_with_suffix,
};
use crate::plot_design::Hemisphere;
use crate::viewer::UnitConvert;

use super::domain::{
    is_global_scale_domain, longitude_bounds_span_deg, range_step, visible_grid_span,
};
use super::projection::{is_broad_continent_scale_domain, rectilinear_latlon_mesh_for_inverse};
use super::types::DirectRequestBuildTiming;
use super::{
    BarbStrideCacheKey, SharedBarbLayerCache, SharedBarbStrideCache, SharedContourLayerCache,
    SharedStreamlineLayerCache,
};

/// The direct lane's per-recipe colorbar overrides: an optional legend-mode
/// override and an optional tick step. Factored out of
/// [`apply_direct_recipe_render_controls`] so the store viewer resolver
/// reads the SAME controls the render request gets.
pub(crate) fn direct_recipe_render_controls(
    recipe: &PlotRecipe,
    filled_selector: FieldSelector,
) -> (Option<LegendMode>, Option<f64>) {
    if filled_selector.field == CanonicalField::SmokeMassDensity {
        // Smoke aliases retain the sparse-signal legend and transparent scale.
        (None, None)
    } else if filled_selector.field == CanonicalField::FogFlag
        || matches!(recipe.style, RenderStyle::WeatherCategorical)
    {
        (Some(LegendMode::Categories), None)
    } else if matches!(recipe.style, RenderStyle::WeatherVisibility) {
        (Some(LegendMode::Thresholds), Some(200.0))
    } else if matches!(recipe.style, RenderStyle::AirQualityPm25 | RenderStyle::AirQualityOzone | RenderStyle::WeatherSnowfall | RenderStyle::WeatherSnowDepthChange) || filled_selector.field == CanonicalField::SnowfallLiquidEquivalent {
        (Some(LegendMode::Thresholds), None)
    } else if matches!(recipe.style, RenderStyle::WeatherHaines) {
        (Some(LegendMode::Categories), None)
    } else if matches!(recipe.style, RenderStyle::WeatherLightningThreat | RenderStyle::WeatherFlashRate) {
        (Some(LegendMode::Thresholds), None)
    } else if matches!(recipe.style, RenderStyle::WeatherDewpoint) {
        let tick = matches!(
            filled_selector.vertical,
            VerticalSelector::HeightAboveGroundMeters(2)
        )
        .then_some(10.0);
        (Some(LegendMode::Stepped), tick)
    } else if matches!(recipe.style, RenderStyle::WeatherRh)
        && matches!(
            filled_selector.vertical,
            VerticalSelector::HeightAboveGroundMeters(2)
        )
    {
        (Some(LegendMode::Stepped), Some(25.0))
    } else {
        (None, None)
    }
}

/// The physical reason a recipe's map can hold nothing to draw, written on
/// the map when it does (`rustwx_render::empty_map_note`); `None` leaves
/// the renderer's generic note.
pub(crate) fn empty_note_for_recipe(recipe: &PlotRecipe, selector: FieldSelector) -> Option<String> {
    if let VerticalSelector::IsothermCelsius(celsius) = selector.vertical {
        // The model top is always colder than any isotherm drawn here, so a
        // missing level at or below 0 C means the whole column is colder.
        return Some(if celsius <= 0 {
            format!("No {celsius} C level: the whole column is colder than {celsius} C")
        } else {
            format!("No {celsius} C level in the column")
        });
    }
    match selector.field {
        CanonicalField::HainesLow => {
            return Some("Haines low undefined: 950 hPa is below ground here".into());
        }
        CanonicalField::HainesMid => {
            return Some("Haines mid undefined: 850 hPa is below ground here".into());
        }
        CanonicalField::HainesHigh => {
            return Some("Haines high undefined: 700 hPa is below ground here".into());
        }
        CanonicalField::SnowfallLiquidEquivalent => {
            return Some("No snowfall accumulated in this window".into());
        }
        _ => {}
    }
    match recipe.style {
        RenderStyle::WeatherSnowfall => Some("No snowfall accumulated in this window".into()),
        RenderStyle::WeatherQpf => Some("No precipitation accumulated in this window".into()),
        RenderStyle::WeatherCategorical => Some("Not diagnosed anywhere at this time".into()),
        _ => None,
    }
}

fn apply_direct_recipe_render_controls(
    recipe: &PlotRecipe,
    filled_selector: FieldSelector,
    request: &mut MapRenderRequest,
) {
    let (legend_mode, tick_step) = direct_recipe_render_controls(recipe, filled_selector);
    if let Some(mode) = legend_mode {
        request.legend.mode = mode;
    }
    if let Some(step) = tick_step {
        request.cbar_tick_step = Some(step);
    }
}

pub(super) fn build_render_request(
    recipe: &PlotRecipe,
    filled: &SelectedField2D,
    extracted: &HashMap<FieldSelector, SelectedField2D>,
    projected: ProjectedMap,
    bounds: (f64, f64, f64, f64),
    output_width: u32,
    output_height: u32,
    contour_layer_cache: &SharedContourLayerCache,
    barb_layer_cache: &SharedBarbLayerCache,
    streamline_layer_cache: &SharedStreamlineLayerCache,
    barb_stride_cache: &SharedBarbStrideCache,
    contour_mode: NativeContourRenderMode,
    native_fill_level_multiplier: usize,
) -> Result<(MapRenderRequest, DirectRequestBuildTiming), Box<dyn std::error::Error>> {
    let mut timing = DirectRequestBuildTiming::default();
    let field_prepare_start = Instant::now();
    let filled_field = render_filled_field(recipe, filled, extracted)?;
    timing.field_prepare_ms = field_prepare_start.elapsed().as_millis();
    let overlay_only = should_render_overlay_only(filled.selector, recipe.contours.is_some());
    let visual_mode = visual_mode_for_direct_recipe(recipe, filled.selector, overlay_only);
    let mut request = if overlay_only {
        let mut request = MapRenderRequest::contour_only(filled_field.clone().into());
        let contour_prepare_start = Instant::now();
        if let Some(layer) = cached_contour_layer(
            filled.selector,
            &filled.values,
            filled.grid.shape.nx,
            filled.grid.shape.ny,
            contour_layer_cache,
        ) {
            request.contours.push(layer);
        }
        timing.contour_prepare_ms += contour_prepare_start.elapsed().as_millis();
        request
    } else {
        MapRenderRequest::new(
            filled_field.clone().into(),
            scale_for_filled_selector_in(
                recipe,
                filled.selector,
                &filled_field.values,
                Hemisphere::for_bounds(bounds),
            ),
        )
    };
    // Keep a fixed hurricane-range ceiling across member/frame sheets.
    // An unset override preserves the operational standalone wind scale.
    if matches!(recipe.slug, "10m_wind_speed_and_direction" | "mslp_10m_winds" | "gefs_avg_mslp_10m_winds") {
        if let Ok(text) = std::env::var("RUSTWX_WIND_MAX_KT") {
            let ceiling = text.parse::<f64>()
                .map_err(|_| "RUSTWX_WIND_MAX_KT must be numeric")?;
            if !ceiling.is_finite() || ceiling <= 0. {
                return Err("RUSTWX_WIND_MAX_KT must be finite and positive".into());
            }
            let mut scale = request.scale.resolved_discrete();
            let intervals = scale.levels.len() - 1;
            scale.levels = (0..=intervals)
                .map(|i| ceiling * i as f64 / intervals as f64).collect();
            scale.mask_below = None;
            request.scale = ColorScale::Discrete(scale);
        }
    }
    crate::plot_design::StaticPlotDesign::new(bounds, visual_mode)
        .overlay_only(overlay_only)
        .apply_to_request(&mut request);
    apply_direct_recipe_render_controls(recipe, filled.selector, &mut request);
    request.empty_note = empty_note_for_recipe(recipe, filled.selector);
    request.title = Some(static_title_with_suffix(recipe.title));
    request.width = output_width;
    request.height = output_height;
    request.chrome_scale = static_chrome_scale();
    request.supersample_factor = static_supersample_factor();
    request.supersample_sharpen = static_supersample_sharpen();
    request.projected_domain = Some(ProjectedDomain {
        x: projected.projected_x,
        y: projected.projected_y,
        extent: projected.extent,
    });
    request.projected_lines = projected.lines;
    request.projected_polygons = projected.polygons;
    request.inverse_raster_projection = projected.inverse_raster_projection;
    let contour_prepare_start = Instant::now();
    if overlay_only {
        request
            .contours
            .extend(build_contour_layers(recipe, extracted, contour_layer_cache));
    } else {
        request.contours = build_contour_layers(recipe, extracted, contour_layer_cache);
    }
    timing.contour_prepare_ms += contour_prepare_start.elapsed().as_millis();
    let barb_prepare_start = Instant::now();
    request.wind_streamlines = build_streamline_layers(
        recipe,
        extracted,
        bounds,
        streamline_layer_cache,
        barb_stride_cache,
    );
    request.wind_barbs = build_barb_layers(
        recipe,
        extracted,
        bounds,
        barb_layer_cache,
        barb_stride_cache,
    );
    apply_planned_wind_layers(&mut request, output_width, output_height, filled.grid.shape);
    timing.barb_prepare_ms = barb_prepare_start.elapsed().as_millis();
    if !overlay_only {
        let contour_fill_start = Instant::now();
        maybe_apply_below_ground_mask_overlay(filled.selector, extracted, &mut request)?;
        maybe_apply_experimental_projected_contours(
            recipe,
            &mut request,
            contour_mode,
            native_fill_level_multiplier,
        )?;
        timing.contour_prepare_ms += contour_fill_start.elapsed().as_millis();
    }
    Ok((request, timing))
}

/// Wind layers on a planned canvas.
///
/// Barbs are spaced by map pixels, the table's `barb_spacing_px`, from the
/// map's own pixels per grid cell: a stride counted in grid cells from the
/// domain's lat/lon span drew 28 barbs across a 300 km square and across a
/// continent alike, whatever size the map came out.
///
/// Streamlines left on automatic are dropped when the product draws
/// barbs: every WRF grid turned them on, and under barbs they read as
/// hatching.  `--streamlines` still draws them.
fn apply_planned_wind_layers(
    request: &mut MapRenderRequest,
    output_width: u32,
    output_height: u32,
    shape: rustwx_core::GridShape,
) {
    let Some(plan) = rustwx_render::canvas_plan_for(output_width, output_height) else {
        return;
    };
    let cells_x = shape.nx.saturating_sub(1).max(1) as f64;
    let cells_y = shape.ny.saturating_sub(1).max(1) as f64;
    let px_per_cell_x = plan.map.w as f64 / cells_x;
    let px_per_cell_y = plan.map.h as f64 / cells_y;
    // The glyph grows with the size class the way its spacing does: a
    // large frame spaced 20 px barbs 88 px apart, a scatter of ticks.
    let glyph_scale = if plan.scale.is_finite() && plan.scale > 1.0 { plan.scale } else { 1.0 };
    for layer in &mut request.wind_barbs {
        layer.stride_x = planned_barb_stride(plan.barb_spacing_px, px_per_cell_x);
        layer.stride_y = planned_barb_stride(plan.barb_spacing_px, px_per_cell_y);
        layer.spacing_px = 0.0;
        layer.length_px *= glyph_scale;
        layer.width = ((layer.width.max(1) as f64) * glyph_scale).round() as u32;
        layer.halo_width = ((layer.halo_width as f64) * glyph_scale).round() as u32;
    }
    if !request.wind_barbs.is_empty()
        && matches!(static_streamline_setting(), StreamlineSetting::Auto)
    {
        request.wind_streamlines.clear();
    }
}

/// Grid cells between barbs for a pixel spacing: never less than one.
pub(super) fn planned_barb_stride(spacing_px: f64, px_per_cell: f64) -> usize {
    if !(px_per_cell > 0.0) || !spacing_px.is_finite() {
        return 1;
    }
    ((spacing_px / px_per_cell).round() as usize).max(1)
}

pub(super) fn apply_source_raster_policy(source: SourceId, request: &mut MapRenderRequest) {
    if matches!(source, SourceId::AifsInference)
        && request.raster_sample_mode == RasterSampleMode::Nearest
    {
        request.raster_sample_mode = RasterSampleMode::Linear;
    }
}

fn maybe_apply_below_ground_mask_overlay(
    filled_selector: FieldSelector,
    extracted: &HashMap<FieldSelector, SelectedField2D>,
    request: &mut MapRenderRequest,
) -> Result<(), Box<dyn std::error::Error>> {
    let VerticalSelector::IsobaricHpa(level_hpa) = filled_selector.vertical else {
        return Ok(());
    };
    let Some(projected_domain) = request.projected_domain.as_ref() else {
        return Ok(());
    };
    let Some(surface_pressure) = extracted.get(&FieldSelector::surface(CanonicalField::Pressure))
    else {
        return Ok(());
    };

    let nx = surface_pressure.grid.shape.nx;
    let ny = surface_pressure.grid.shape.ny;
    if nx < 2 || ny < 2 {
        return Ok(());
    }
    if projected_domain.x.len() != nx * ny || projected_domain.y.len() != nx * ny {
        return Ok(());
    }

    let target_pa = level_hpa as f32 * 100.0;
    let masked: Vec<bool> = surface_pressure
        .values
        .iter()
        .map(|value| value.is_finite() && *value < target_pa)
        .collect();
    if !masked.iter().any(|value| *value) {
        return Ok(());
    }

    let render_mask = dilate_mask(&masked, nx, ny);
    apply_below_ground_nan_mask(&render_mask, &mut request.field.values);
    for contour in &mut request.contours {
        apply_below_ground_nan_mask(&render_mask, &mut contour.data);
    }
    for barb in &mut request.wind_barbs {
        apply_below_ground_nan_mask(&render_mask, &mut barb.u);
        apply_below_ground_nan_mask(&render_mask, &mut barb.v);
    }

    let idx = |j: usize, i: usize| j * nx + i;
    let cell_masked = |j: usize, i: usize| {
        render_mask[idx(j, i)]
            && render_mask[idx(j, i + 1)]
            && render_mask[idx(j + 1, i)]
            && render_mask[idx(j + 1, i + 1)]
    };

    for j in 0..(ny - 1) {
        let mut i = 0usize;
        while i < nx - 1 {
            if !cell_masked(j, i) {
                i += 1;
                continue;
            }
            let start = i;
            let mut end = i;
            while end + 1 < nx - 1 && cell_masked(j, end + 1) {
                end += 1;
            }

            let mut ring = Vec::with_capacity(((end - start + 2) * 2) + 1);
            for col in start..=end + 1 {
                ring.push((
                    projected_domain.x[idx(j, col)],
                    projected_domain.y[idx(j, col)],
                ));
            }
            for col in (start..=end + 1).rev() {
                ring.push((
                    projected_domain.x[idx(j + 1, col)],
                    projected_domain.y[idx(j + 1, col)],
                ));
            }
            if let Some(first) = ring.first().copied() {
                ring.push(first);
            }
            if ring.iter().all(|(x, y)| x.is_finite() && y.is_finite()) {
                request
                    .projected_data_polygons
                    .push(rustwx_render::ProjectedPolygonFill {
                        rings: vec![ring],
                        color: Color::rgba(210, 200, 181, 255),
                        role: rustwx_render::PolygonRole::Generic,
                    });
            }
            i = end + 1;
        }
    }
    Ok(())
}

fn dilate_mask(mask: &[bool], nx: usize, ny: usize) -> Vec<bool> {
    let mut dilated = vec![false; mask.len()];
    for j in 0..ny {
        let j0 = j.saturating_sub(1);
        let j1 = (j + 1).min(ny - 1);
        for i in 0..nx {
            let i0 = i.saturating_sub(1);
            let i1 = (i + 1).min(nx - 1);
            let masked = (j0..=j1).any(|jj| (i0..=i1).any(|ii| mask[jj * nx + ii]));
            dilated[j * nx + i] = masked;
        }
    }
    dilated
}

fn apply_below_ground_nan_mask(mask: &[bool], values: &mut [f32]) {
    if values.len() != mask.len() {
        return;
    }
    for (value, masked) in values.iter_mut().zip(mask.iter().copied()) {
        if masked {
            *value = f32::NAN;
        }
    }
}

fn maybe_apply_experimental_projected_contours(
    _recipe: &PlotRecipe,
    request: &mut MapRenderRequest,
    contour_mode: NativeContourRenderMode,
    native_fill_level_multiplier: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let enabled = match contour_mode {
        NativeContourRenderMode::Automatic
        | NativeContourRenderMode::LegacyRaster
        | NativeContourRenderMode::Signature => false,
        NativeContourRenderMode::ExperimentalAllProjected => true,
    };
    if !enabled {
        return Ok(());
    }
    let Some(projected_domain) = request.projected_domain.as_ref() else {
        return Ok(());
    };
    request.scale =
        densify_direct_native_contour_scale(request.scale.clone(), native_fill_level_multiplier);
    let (geometry, _) = build_projected_contour_geometry_profile(
        &request.field,
        projected_domain,
        &request.scale,
        &[],
        ProjectedContourLineStyle::default(),
    )?;
    request.projected_data_polygons.extend(geometry.fills);
    request.projected_lines.extend(geometry.lines);
    request.field.values.fill(f32::NAN);
    Ok(())
}

fn densify_direct_native_contour_scale(
    scale: ColorScale,
    native_fill_level_multiplier: usize,
) -> ColorScale {
    if native_fill_level_multiplier <= 1 {
        return scale;
    }
    let discrete = scale.resolved_discrete();
    ColorScale::Discrete(densify_discrete_scale(
        &discrete,
        LevelDensity {
            multiplier: native_fill_level_multiplier,
            min_source_level_count: 2,
        },
    ))
}

pub(super) fn visual_mode_for_direct_recipe(
    recipe: &PlotRecipe,
    selector: FieldSelector,
    overlay_only: bool,
) -> ProductVisualMode {
    if overlay_only {
        return ProductVisualMode::OverlayAnalysis;
    }

    if matches!(recipe.style, RenderStyle::WeatherHeight)
        || matches!(selector.vertical, VerticalSelector::IsobaricHpa(_))
    {
        return ProductVisualMode::UpperAirAnalysis;
    }

    let slug = recipe.slug.to_ascii_lowercase();
    if [
        "cape", "cin", "stp", "scp", "ehi", "srh", "shear", "lapse", "uh", "helicity",
    ]
    .iter()
    .any(|token| slug.contains(token))
    {
        return ProductVisualMode::SevereDiagnostic;
    }

    ProductVisualMode::FilledMeteorology
}

pub(super) fn sanitize_output_suffix(value: &str) -> String {
    let mut out = String::new();
    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
        } else if matches!(ch, '_' | '-' | '.') {
            out.push(ch);
        } else if ch.is_whitespace() {
            out.push('_');
        }
    }
    out.trim_matches(['_', '-', '.']).to_string()
}

fn selector_is_spread_product(selector: FieldSelector) -> bool {
    matches!(
        selector.product,
        FieldProduct::EnsembleStandardDeviation | FieldProduct::EnsembleSpread
    )
}

pub(super) fn render_filled_field(
    recipe: &PlotRecipe,
    field: &SelectedField2D,
    extracted: &HashMap<FieldSelector, SelectedField2D>,
) -> Result<rustwx_core::Field2D, Box<dyn std::error::Error>> {
    if let Some(wind_speed) = derived_companion_wind_speed_fill(recipe, field, extracted)? {
        return Ok(wind_speed);
    }
    Ok(convert_filled_field(recipe, field))
}

fn derived_companion_wind_speed_fill(
    recipe: &PlotRecipe,
    field: &SelectedField2D,
    extracted: &HashMap<FieldSelector, SelectedField2D>,
) -> Result<Option<rustwx_core::Field2D>, Box<dyn std::error::Error>> {
    let should_fill_with_companion_wind = (recipe.slug == "mslp_10m_winds"
        || recipe.slug == "gefs_avg_mslp_10m_winds")
        || (recipe.style == RenderStyle::WeatherHeight
            && field.selector.field == CanonicalField::GeopotentialHeight);
    if !should_fill_with_companion_wind {
        return Ok(None);
    }

    let (Some(u_spec), Some(v_spec)) = (&recipe.barbs_u, &recipe.barbs_v) else {
        return Ok(None);
    };
    let (Some(u_selector), Some(v_selector)) = (u_spec.selector, v_spec.selector) else {
        return Ok(None);
    };
    let (Some(u), Some(v)) = (extracted.get(&u_selector), extracted.get(&v_selector)) else {
        return Ok(None);
    };

    let values: Vec<f32> = u
        .values
        .iter()
        .zip(&v.values)
        .map(|(u_value, v_value)| {
            let speed_ms = ((*u_value as f64).powi(2) + (*v_value as f64).powi(2)).sqrt();
            (speed_ms * 1.943_844_5) as f32
        })
        .collect();

    let field = rustwx_core::Field2D::new(
        rustwx_core::ProductKey::named(format!("{}_wind_speed", recipe.slug)),
        "kt",
        u.grid.clone(),
        values,
    )?;
    Ok(Some(field))
}

/// The unit conversion + display-units override the direct lane applies to
/// a filled field BEFORE its color scale: the single conversion table
/// behind [`convert_filled_field`], factored out so the store viewer
/// resolver converts raw stored values with the SAME arithmetic.
pub(crate) fn direct_fill_unit_conversion(
    recipe: &PlotRecipe,
    selector: FieldSelector,
) -> (UnitConvert, Option<&'static str>) {
    if selector.field == CanonicalField::SmokeMassDensity {
        (UnitConvert::KgM3ToUgM3, Some("ug/m^3"))
    } else if selector.field == CanonicalField::ColumnIntegratedSmoke {
        (UnitConvert::KgM2ToMgM2, Some("mg/m^2"))
    } else if matches!(
        selector.field,
        CanonicalField::SpecificHumidity
            | CanonicalField::CloudWaterMixingRatio
            | CanonicalField::RainWaterMixingRatio
            | CanonicalField::CloudIceMixingRatio
            | CanonicalField::SnowMixingRatio
            | CanonicalField::GraupelMixingRatio
    ) {
        // A mixing ratio is drawn in grams per kilogram on every route: the
        // section route already speaks g kg-1 and a bar in kg kg-1 read
        // its every tick as 0.
        (UnitConvert::KgPerKgToGPerKg, Some("g kg-1"))
    } else if selector.field == CanonicalField::SupercooledLiquidWaterPath {
        (UnitConvert::None, Some("g m-2"))
    } else if matches!(
        recipe.style,
        RenderStyle::WeatherTemperature | RenderStyle::WeatherDewpoint
    ) {
        if selector_is_spread_product(selector) {
            (UnitConvert::None, Some("K"))
        } else if matches!(
            selector.vertical,
            VerticalSelector::HeightAboveGroundMeters(2)
        ) {
            (UnitConvert::KelvinToFahrenheit, Some("degF"))
        } else {
            (UnitConvert::KelvinToCelsius, Some("degC"))
        }
    } else if selector.field == CanonicalField::PressureReducedToMeanSeaLevel {
        (UnitConvert::PaToHpa, Some("hPa"))
    } else if selector.field == CanonicalField::PrecipitableWater {
        (UnitConvert::MmToInches, Some("in"))
    } else if selector.field == CanonicalField::Visibility {
        (UnitConvert::None, Some("m"))
    } else if selector.field == CanonicalField::AbsoluteVorticity {
        (UnitConvert::PerSecondToE5PerSecond, Some("10^-5 s^-1"))
    } else if matches!(
        selector.field,
        CanonicalField::WindSpeed | CanonicalField::WindGust
    ) {
        (UnitConvert::MsToKnots, Some("kt"))
    } else if selector.field == CanonicalField::TotalPrecipitation {
        if matches!(recipe.style, RenderStyle::WeatherQpf) {
            (UnitConvert::MmToInches, Some("in"))
        } else {
            (UnitConvert::None, Some("mm"))
        }
    } else if matches!(selector.field, CanonicalField::ModelSnowfall | CanonicalField::Snowfall10to1 | CanonicalField::SnowfallKuchera | CanonicalField::SnowfallLiquidEquivalent | CanonicalField::SnowDepth | CanonicalField::SnowDepthChange) {
        (UnitConvert::MmToInches, Some("in"))
    } else {
        (UnitConvert::None, None)
    }
}

pub(super) fn convert_filled_field(
    recipe: &PlotRecipe,
    field: &SelectedField2D,
) -> rustwx_core::Field2D {
    let mut core = field.clone().into_field2d();
    let (convert, units_override) = direct_fill_unit_conversion(recipe, field.selector);
    if !matches!(convert, UnitConvert::None) {
        for value in &mut core.values {
            *value = convert.apply(*value);
        }
    }
    if let Some(units) = units_override {
        core.units = units.to_string();
    }
    core
}

pub(super) fn should_render_overlay_only(
    selector: FieldSelector,
    has_explicit_contours: bool,
) -> bool {
    if has_explicit_contours {
        return false;
    }
    // An isobaric geopotential height is a contour-analysis field: the
    // convention is heights as contours, colour reserved for the wind.
    // The SURFACE member of the same canonical field is a different
    // physical quantity -- orography -- and reads only as a fill; sending
    // it down the contour-analysis path would draw a blank map.  The
    // height of an ISOTHERM is a third quantity, where the air reaches a
    // named temperature, and reads as a fill for the same reason: drawn
    // overlay-only it came out as decametre contours on a blank map.
    matches!(selector.field, CanonicalField::GeopotentialHeight)
        && !matches!(
            selector.vertical,
            VerticalSelector::Surface | VerticalSelector::IsothermCelsius(_)
        )
}

pub(super) fn scale_for_filled_selector(
    recipe: &PlotRecipe,
    filled_selector: FieldSelector,
    values: &[f32],
) -> ColorScale {
    scale_for_filled_selector_in(
        recipe, filled_selector, values, Hemisphere::Northern)
}

pub(super) fn scale_for_filled_selector_in(
    recipe: &PlotRecipe,
    filled_selector: FieldSelector,
    values: &[f32],
    hemisphere: Hemisphere,
) -> ColorScale {
    if selector_is_spread_product(filled_selector) {
        return ensemble_spread_scale(values);
    }
    let scale = scale_for_recipe_in(recipe, filled_selector, hemisphere);
    // These pressure recipes fill the companion 10-m wind magnitude in
    // knots, while retaining pressure as the selector for contour overlays.
    if recipe.slug == "mslp_10m_winds" || recipe.slug == "gefs_avg_mslp_10m_winds" {
        return match scale {
            ColorScale::Discrete(discrete) => ColorScale::Discrete(
                crate::plot_design::fit_surface_wind_scale(discrete, values)),
            other => other,
        };
    }
    // Isobaric temperature: the fixed mid-latitude ladder moves onto a
    // field it would otherwise saturate (a tropical 300 hPa map drew one
    // colour); unchanged where the field already fits.
    match (&recipe.style, filled_selector.field, filled_selector.vertical, scale) {
        (_, CanonicalField::WindSpeed, VerticalSelector::HeightAboveGroundMeters(10),
         ColorScale::Discrete(discrete)) => ColorScale::Discrete(
            crate::plot_design::fit_surface_wind_scale(discrete, values)),
        (RenderStyle::WeatherTemperature, CanonicalField::Temperature, VerticalSelector::IsobaricHpa(_),
         ColorScale::Discrete(discrete)) => ColorScale::Discrete(
            crate::plot_design::fit_isobaric_temperature_scale(discrete, values)),
        (_, _, _, scale) => scale,
    }
}

fn ensemble_spread_scale(values: &[f32]) -> ColorScale {
    let mut finite = values
        .iter()
        .filter_map(|value| {
            let value = *value as f64;
            value.is_finite().then_some(value.max(0.0))
        })
        .collect::<Vec<_>>();
    finite.sort_by(|a, b| a.total_cmp(b));
    let p99 = percentile_sorted(&finite, 0.99).unwrap_or(1.0);
    let upper = nice_spread_upper_bound(p99);
    let step = nice_spread_step(upper / 16.0);
    ColorScale::Discrete(DiscreteColorScale {
        levels: range_step(0.0, upper + step * 0.5, step),
        colors: ensemble_spread_colors(),
        extend: ExtendMode::Max,
        mask_below: None,
    })
}

fn percentile_sorted(values: &[f64], percentile: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let index = ((values.len() - 1) as f64 * percentile.clamp(0.0, 1.0)).round() as usize;
    values.get(index).copied()
}

fn nice_spread_upper_bound(value: f64) -> f64 {
    let value = if value.is_finite() && value > 0.0 {
        value
    } else {
        1.0
    };
    let magnitude = 10_f64.powf(value.log10().floor());
    for multiple in [1.0, 2.0, 2.5, 5.0, 10.0] {
        let candidate = multiple * magnitude;
        if candidate >= value {
            return candidate.max(0.1);
        }
    }
    (10.0 * magnitude).max(0.1)
}

fn nice_spread_step(value: f64) -> f64 {
    let value = if value.is_finite() && value > 0.0 {
        value
    } else {
        0.1
    };
    let magnitude = 10_f64.powf(value.log10().floor());
    for multiple in [1.0, 2.0, 2.5, 5.0, 10.0] {
        let candidate = multiple * magnitude;
        if candidate >= value {
            return candidate.max(0.01);
        }
    }
    (10.0 * magnitude).max(0.01)
}

fn ensemble_spread_colors() -> Vec<Color> {
    vec![
        Color::rgba(247, 251, 255, 255),
        Color::rgba(222, 235, 247, 255),
        Color::rgba(198, 219, 239, 255),
        Color::rgba(158, 202, 225, 255),
        Color::rgba(107, 174, 214, 255),
        Color::rgba(49, 130, 189, 255),
        Color::rgba(8, 81, 156, 255),
        Color::rgba(8, 48, 107, 255),
    ]
}

pub(super) fn scale_for_recipe(recipe: &PlotRecipe, filled_selector: FieldSelector) -> ColorScale {
    scale_for_recipe_in(recipe, filled_selector, Hemisphere::Northern)
}

pub(super) fn scale_for_recipe_in(
    recipe: &PlotRecipe,
    filled_selector: FieldSelector,
    hemisphere: Hemisphere,
) -> ColorScale {
    crate::plot_design::operational_fill_scale_for_recipe_in(
        recipe, filled_selector, hemisphere)
}

fn build_contour_layers(
    recipe: &PlotRecipe,
    extracted: &HashMap<FieldSelector, SelectedField2D>,
    contour_layer_cache: &SharedContourLayerCache,
) -> Vec<ContourLayer> {
    let Some(spec) = &recipe.contours else {
        return Vec::new();
    };
    let Some(selector) = spec.selector else {
        return Vec::new();
    };
    let Some(field) = extracted.get(&selector) else {
        return Vec::new();
    };

    cached_contour_layer(
        selector,
        &field.values,
        field.grid.shape.nx,
        field.grid.shape.ny,
        contour_layer_cache,
    )
    .into_iter()
    .collect()
}

fn cached_contour_layer(
    selector: FieldSelector,
    values: &[f32],
    nx: usize,
    ny: usize,
    contour_layer_cache: &SharedContourLayerCache,
) -> Option<ContourLayer> {
    let key = (selector, nx, ny);
    {
        let cache = contour_layer_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(layer) = cache.get(&key) {
            return layer.clone();
        }
    }

    let contour_values = contour_values_for_render(selector, values, nx, ny);
    let layer = crate::plot_design::operational_contour_layer_for_values(selector, &contour_values);
    let mut cache = contour_layer_cache
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    cache.entry(key).or_insert_with(|| layer.clone()).clone()
}

fn contour_values_for_render(
    selector: FieldSelector,
    values: &[f32],
    nx: usize,
    ny: usize,
) -> Vec<f32> {
    if selector.field == CanonicalField::PressureReducedToMeanSeaLevel {
        smooth_contour_values(values, nx, ny, pressure_contour_smoothing_passes())
    } else {
        values.to_vec()
    }
}

fn pressure_contour_smoothing_passes() -> usize {
    std::env::var("RUSTWX_PRESSURE_CONTOUR_SMOOTH_PASSES")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .or_else(|| rustwx_render::theme::active_theme().contour_smooth_passes("mslp"))
        .unwrap_or(0)
        .clamp(0, 12)
}

fn smooth_contour_values(values: &[f32], nx: usize, ny: usize, passes: usize) -> Vec<f32> {
    if passes == 0 || nx < 3 || ny < 3 || values.len() != nx.saturating_mul(ny) {
        return values.to_vec();
    }

    let mut current = values.to_vec();
    let mut next = current.clone();
    for _ in 0..passes {
        for j in 0..ny {
            let y0 = j.saturating_sub(2);
            let y1 = (j + 2).min(ny - 1);
            for i in 0..nx {
                let x0 = i.saturating_sub(2);
                let x1 = (i + 2).min(nx - 1);
                let mut sum = 0.0_f32;
                let mut count = 0usize;
                for yy in y0..=y1 {
                    let row = yy * nx;
                    for xx in x0..=x1 {
                        let value = current[row + xx];
                        if value.is_finite() {
                            sum += value;
                            count += 1;
                        }
                    }
                }
                next[j * nx + i] = if count > 0 {
                    sum / count as f32
                } else {
                    f32::NAN
                };
            }
        }
        std::mem::swap(&mut current, &mut next);
    }
    current
}

fn build_streamline_layers(
    recipe: &PlotRecipe,
    extracted: &HashMap<FieldSelector, SelectedField2D>,
    bounds: (f64, f64, f64, f64),
    streamline_layer_cache: &SharedStreamlineLayerCache,
    barb_stride_cache: &SharedBarbStrideCache,
) -> Vec<WindStreamlineLayer> {
    if recipe.style == RenderStyle::WeatherIvt { return Vec::new(); }
    let (Some(u_spec), Some(v_spec)) = (&recipe.barbs_u, &recipe.barbs_v) else {
        return Vec::new();
    };
    let (Some(u_selector), Some(v_selector)) = (u_spec.selector, v_spec.selector) else {
        return Vec::new();
    };
    let (Some(u), Some(v)) = (extracted.get(&u_selector), extracted.get(&v_selector)) else {
        return Vec::new();
    };
    if u_selector.field == CanonicalField::IvtEastward || !static_streamlines_enabled_for_grid(&u.grid) {
        return Vec::new();
    }
    let key = BarbStrideCacheKey {
        u_selector,
        v_selector,
        bounds_bits: [
            bounds.0.to_bits(),
            bounds.1.to_bits(),
            bounds.2.to_bits(),
            bounds.3.to_bits(),
        ],
    };
    {
        let cache = streamline_layer_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(layers) = cache.get(&key) {
            return layers.clone();
        }
    }

    let (stride_x, stride_y) =
        cached_streamline_strides(u_selector, v_selector, &u.grid, bounds, barb_stride_cache);
    let style = crate::plot_design::operational_wind_streamline_style(stride_x, stride_y);
    let layers = vec![WindStreamlineLayer {
        u: u.values.iter().map(|value| value * 1.943_844_5).collect(),
        v: v.values.iter().map(|value| value * 1.943_844_5).collect(),
        stride_x: style.stride_x,
        stride_y: style.stride_y,
        color: style.color,
        width: style.width,
        max_steps: style.max_steps,
        step_cells: style.step_cells,
        min_speed: style.min_speed,
    }];
    let mut cache = streamline_layer_cache
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    cache.entry(key).or_insert_with(|| layers.clone()).clone()
}

fn build_barb_layers(
    recipe: &PlotRecipe,
    extracted: &HashMap<FieldSelector, SelectedField2D>,
    bounds: (f64, f64, f64, f64),
    barb_layer_cache: &SharedBarbLayerCache,
    barb_stride_cache: &SharedBarbStrideCache,
) -> Vec<WindBarbLayer> {
    let (Some(u_spec), Some(v_spec)) = (&recipe.barbs_u, &recipe.barbs_v) else {
        return Vec::new();
    };
    let (Some(u_selector), Some(v_selector)) = (u_spec.selector, v_spec.selector) else {
        return Vec::new();
    };
    let (Some(u), Some(v)) = (extracted.get(&u_selector), extracted.get(&v_selector)) else {
        return Vec::new();
    };
    let is_transport = u_selector.field == CanonicalField::IvtEastward || recipe.style == RenderStyle::WeatherIvt;
    let key = BarbStrideCacheKey {
        u_selector,
        v_selector,
        bounds_bits: [
            bounds.0.to_bits(),
            bounds.1.to_bits(),
            bounds.2.to_bits(),
            bounds.3.to_bits(),
        ],
    };
    {
        let cache = barb_layer_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(layers) = cache.get(&key) {
            return layers.clone();
        }
    }

    let (stride_x, stride_y) =
        cached_barb_strides(u_selector, v_selector, &u.grid, bounds, barb_stride_cache);
    let layers = vec![WindBarbLayer {
        direction_arrows: is_transport,
        u: u.values.iter().map(|value| value * if is_transport {1.0} else {1.943_844_5}).collect(),
        v: v.values.iter().map(|value| value * if is_transport {1.0} else {1.943_844_5}).collect(),
        stride_x,
        stride_y,
        spacing_px: static_barb_spacing_px(),
        color: Color::BLACK,
        halo_color: Color::WHITE,
        halo_width: static_barb_halo_width(),
        width: static_barb_width(),
        length_px: static_barb_length_px(),
    }];
    let mut cache = barb_layer_cache
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    cache.entry(key).or_insert_with(|| layers.clone()).clone()
}

fn cached_barb_strides(
    u_selector: FieldSelector,
    v_selector: FieldSelector,
    grid: &rustwx_core::LatLonGrid,
    bounds: (f64, f64, f64, f64),
    barb_stride_cache: &SharedBarbStrideCache,
) -> (usize, usize) {
    let key = BarbStrideCacheKey {
        u_selector,
        v_selector,
        bounds_bits: [
            bounds.0.to_bits(),
            bounds.1.to_bits(),
            bounds.2.to_bits(),
            bounds.3.to_bits(),
        ],
    };

    {
        let cache = barb_stride_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(&strides) = cache.get(&key) {
            return strides;
        }
    }

    let (visible_nx, visible_ny) = visible_grid_span(grid, bounds);
    let density = static_barb_density_scale();
    let (target_columns, target_rows) = barb_target_columns_rows(bounds);
    let strides = (
        ((visible_nx as f64 / (target_columns * density)).round() as usize).clamp(2, 128),
        ((visible_ny as f64 / (target_rows * density)).round() as usize).clamp(2, 96),
    );

    let mut cache = barb_stride_cache
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    *cache.entry(key).or_insert(strides)
}

pub(super) fn barb_target_columns_rows(bounds: (f64, f64, f64, f64)) -> (f64, f64) {
    let lat_span = (bounds.3 - bounds.2).abs();
    let lon_span = longitude_bounds_span_deg(bounds);
    if is_global_scale_domain(bounds) {
        (34.0, 16.0)
    } else if is_broad_continent_scale_domain(bounds) {
        (26.0, 13.0)
    } else if lat_span <= 12.0 && lon_span <= 20.0 {
        (28.0, 18.0)
    } else {
        (23.0, 14.0)
    }
}

fn cached_streamline_strides(
    u_selector: FieldSelector,
    v_selector: FieldSelector,
    grid: &rustwx_core::LatLonGrid,
    bounds: (f64, f64, f64, f64),
    barb_stride_cache: &SharedBarbStrideCache,
) -> (usize, usize) {
    let barb_strides = cached_barb_strides(u_selector, v_selector, grid, bounds, barb_stride_cache);
    let density = static_streamline_density_scale();
    (
        ((barb_strides.0 as f64 / density).round() as usize).clamp(2, 96),
        ((barb_strides.1 as f64 / density).round() as usize).clamp(2, 64),
    )
}

fn static_barb_width() -> u32 {
    std::env::var("RUSTWX_BARB_WIDTH")
        .ok()
        .and_then(|value| value.trim().parse::<u32>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(1)
        .clamp(1, 8)
}

fn static_barb_halo_width() -> u32 {
    std::env::var("RUSTWX_BARB_HALO_WIDTH")
        .ok()
        .and_then(|value| value.trim().parse::<u32>().ok())
        .unwrap_or(1)
        .clamp(0, 8)
}

fn static_barb_length_px() -> f64 {
    std::env::var("RUSTWX_BARB_LENGTH_PX")
        .ok()
        .and_then(|value| value.trim().parse::<f64>().ok())
        .filter(|value| value.is_finite() && *value > 0.0)
        .unwrap_or(20.0)
        .clamp(6.0, 48.0)
}

fn static_barb_spacing_px() -> f64 {
    std::env::var("RUSTWX_BARB_SPACING_PX")
        .ok()
        .and_then(|value| value.trim().parse::<f64>().ok())
        .filter(|value| value.is_finite() && *value >= 0.0)
        .unwrap_or(0.0)
        .clamp(0.0, 160.0)
}

fn static_barb_density_scale() -> f64 {
    std::env::var("RUSTWX_BARB_DENSITY")
        .ok()
        .and_then(|value| value.trim().parse::<f64>().ok())
        .filter(|value| value.is_finite() && *value > 0.0)
        .unwrap_or(1.0)
        .clamp(0.25, 4.0)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum StreamlineSetting {
    Auto,
    Enabled,
    Disabled,
}

/// The streamline choice in force: an explicit front-door request first,
/// then `RUSTWX_WIND_STREAMLINES`, then the automatic per-grid decision.
///
/// The command line outranks the environment on purpose. A user who types
/// `--streamlines` and gets barbs because a shell profile exported
/// `RUSTWX_WIND_STREAMLINES=0` months ago has no way to see why, and the
/// flag would silently be a lie. The variable keeps working for every
/// caller that has been setting it.
pub(super) fn static_streamline_setting() -> StreamlineSetting {
    if let Some(request) = crate::shared_context::wind_streamline_request() {
        return if request {
            StreamlineSetting::Enabled
        } else {
            StreamlineSetting::Disabled
        };
    }
    std::env::var("RUSTWX_WIND_STREAMLINES")
        .ok()
        .map(|value| match value.trim().to_ascii_lowercase().as_str() {
            "0" | "false" | "off" | "no" => StreamlineSetting::Disabled,
            "1" | "true" | "on" | "yes" | "force" => StreamlineSetting::Enabled,
            _ => StreamlineSetting::Auto,
        })
        .unwrap_or(StreamlineSetting::Auto)
}

fn static_streamlines_enabled_for_grid(grid: &rustwx_core::LatLonGrid) -> bool {
    streamlines_enabled_for_grid(static_streamline_setting(), grid)
}

pub(super) fn streamlines_enabled_for_grid(
    setting: StreamlineSetting,
    grid: &rustwx_core::LatLonGrid,
) -> bool {
    match setting {
        StreamlineSetting::Disabled => false,
        StreamlineSetting::Enabled => true,
        StreamlineSetting::Auto => {
            !rectilinear_latlon_mesh_for_inverse(&grid.lat_deg, &grid.lon_deg)
        }
    }
}

fn static_streamline_density_scale() -> f64 {
    std::env::var("RUSTWX_STREAMLINE_DENSITY")
        .ok()
        .and_then(|value| value.trim().parse::<f64>().ok())
        .filter(|value| value.is_finite() && *value > 0.0)
        .unwrap_or(1.0)
        .clamp(0.25, 4.0)
}
