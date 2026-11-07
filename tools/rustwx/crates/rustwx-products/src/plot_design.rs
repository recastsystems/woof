use rustwx_core::{CanonicalField, FieldSelector, VerticalSelector};
use rustwx_models::{PlotRecipe, RenderStyle};
use rustwx_render::{
    Color, ColorScale, ContourLayer, ContourLinePattern, DiscreteColorScale, DomainFrame,
    DomainFrameSource, ExtendMode, LegendControls, LegendMode, LevelDensity, MapRenderRequest,
    ProductVisualMode, RenderDensity, WindStreamlineStyle,
    weather::{
        WeatherPalette, dewpoint_palette_celsius_for_levels, temperature_palette_cropped_f,
        weather_palette, winds_palette_segments,
    },
};

#[derive(Debug, Clone, Copy)]
pub struct StaticPlotDesign {
    pub bounds: (f64, f64, f64, f64),
    pub visual_mode: ProductVisualMode,
    pub overlay_only: bool,
}

impl StaticPlotDesign {
    pub fn new(bounds: (f64, f64, f64, f64), visual_mode: ProductVisualMode) -> Self {
        Self {
            bounds,
            visual_mode,
            overlay_only: false,
        }
    }

    pub fn overlay_only(mut self, overlay_only: bool) -> Self {
        self.overlay_only = overlay_only;
        self
    }

    pub fn apply_to_request(self, request: &mut MapRenderRequest) {
        apply_static_map_design(request, self.bounds, self.visual_mode, self.overlay_only);
    }
}

pub fn longitude_bounds_span_deg(bounds: (f64, f64, f64, f64)) -> f64 {
    let raw_span = (bounds.1 - bounds.0).abs();
    if raw_span >= 359.0 {
        return raw_span.min(360.0);
    }

    let west = normalize_longitude_for_bounds(bounds.0);
    let east = normalize_longitude_for_bounds(bounds.1);
    if west <= east {
        east - west
    } else {
        east + 360.0 - west
    }
}

pub fn is_global_scale_domain(bounds: (f64, f64, f64, f64)) -> bool {
    let lat_span = (bounds.3 - bounds.2).abs();
    lat_span >= 100.0 && longitude_bounds_span_deg(bounds) >= 300.0
}

pub fn static_domain_frame_for_bounds(bounds: (f64, f64, f64, f64)) -> Option<DomainFrame> {
    if is_global_scale_domain(bounds) {
        None
    } else if straight_western_domain_frame_enabled(bounds) {
        Some(static_map_viewport_domain_frame())
    } else {
        Some(static_model_data_domain_frame())
    }
}

fn static_map_viewport_domain_frame() -> DomainFrame {
    DomainFrame {
        inset_px: 2,
        outline_width: 2,
        source: DomainFrameSource::MapViewport,
        ..DomainFrame::map_viewport_default()
    }
}

fn static_model_data_domain_frame() -> DomainFrame {
    DomainFrame {
        inset_px: 2,
        outline_width: 2,
        source: DomainFrameSource::ProjectedGrid,
        ..DomainFrame::map_viewport_default()
    }
}

/// The map-viewport frame (and with it the whitespace crop) for western
/// domains, only under the explicit `RUSTWX_STRAIGHT_WEST_PROJECTION`
/// opt-in.  It used to be the default inside a hard-coded western lat/lon
/// box, so the same domain shape came out cropped in one place and
/// letterboxed in another; the canvas is now sized from the grid.
fn straight_western_domain_frame_enabled(bounds: (f64, f64, f64, f64)) -> bool {
    let opted_in = std::env::var("RUSTWX_STRAIGHT_WEST_PROJECTION")
        .ok()
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on" | "mercator" | "straight" | "northup"
            )
        })
        .unwrap_or(false);
    opted_in && is_straight_western_domain_frame_candidate(bounds)
}

fn is_straight_western_domain_frame_candidate(bounds: (f64, f64, f64, f64)) -> bool {
    let west = normalize_longitude_for_bounds(bounds.0);
    let east = normalize_longitude_for_bounds(bounds.1);
    if west > east {
        return false;
    }
    let lat_span = (bounds.3 - bounds.2).abs();
    let lon_span = longitude_bounds_span_deg(bounds);
    bounds.2 >= 25.0
        && bounds.3 <= 55.0
        && west >= -130.0
        && west <= -115.0
        && east >= -123.0
        && east <= -104.0
        && lat_span >= 4.0
        && lon_span <= 28.0
}

pub fn apply_static_map_design(
    request: &mut MapRenderRequest,
    bounds: (f64, f64, f64, f64),
    visual_mode: ProductVisualMode,
    overlay_only: bool,
) {
    request.visual_mode = visual_mode;
    request.render_density = RenderDensity {
        fill: high_detail_fill_density(),
        palette_multiplier: 4,
    };
    request.legend = LegendControls {
        density: LevelDensity::default(),
        mode: LegendMode::SmoothRamp,
    };
    if is_global_scale_domain(bounds) && !overlay_only {
        request.render_density = RenderDensity::default();
        request.legend = LegendControls {
            density: LevelDensity::default(),
            mode: LegendMode::SmoothRamp,
        };
    }
    request.domain_frame = static_domain_frame_for_bounds(bounds);
}

fn high_detail_fill_density() -> LevelDensity {
    LevelDensity {
        multiplier: 4,
        min_source_level_count: 2,
    }
}

/// Which hemisphere's sign convention a plot should be drawn in.
///
/// Only vorticity needs this today.  Absolute vorticity is dominated by
/// the planetary term `f = 2*omega*sin(lat)`, which changes sign at the
/// equator, so CYCLONIC rotation is positive in the north and negative
/// in the south.  A single fixed ramp can only give its structured
/// colours to one of those.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Hemisphere {
    Northern,
    Southern,
}

impl Hemisphere {
    /// The hemisphere a domain sits in, from its (W, E, S, N) bounds.
    ///
    /// The midpoint of the latitude span, so a domain straddling the
    /// equator takes the side it mostly occupies rather than flipping on
    /// one row of cells.
    pub fn for_bounds(bounds: (f64, f64, f64, f64)) -> Self {
        if 0.5 * (bounds.2 + bounds.3) < 0.0 {
            Self::Southern
        } else {
            Self::Northern
        }
    }
}

/// Absolute-vorticity fill, in the sign convention of the hemisphere the
/// domain is in.
///
/// The northern ramp is `-40..60` with the RelVort palette, whose
/// structured half (orange -> red -> purple -> cyan) sits above zero
/// because that is where northern cyclonic vorticity is.  Rendered over
/// South America that produced a nearly uniform grey chart: the whole
/// synoptic cyclonic signal was negative, i.e. in the flat end of the
/// ramp, and the only coloured features were minor positive streaks --
/// the least significant signal on the plot (B-07).
///
/// The southern form is the northern one mirrored: the range reflects
/// about zero and the palette is reversed, so cyclonic rotation gets the
/// same colours, in the same order, that a northern forecaster reads.
/// The northern scale is returned unchanged, byte for byte.
fn vorticity_scale_for_hemisphere(hemisphere: Hemisphere) -> DiscreteColorScale {
    match hemisphere {
        Hemisphere::Northern => DiscreteColorScale {
            levels: range_step(-40.0, 60.1, 1.0),
            colors: weather_palette(WeatherPalette::RelVort),
            extend: ExtendMode::Both,
            mask_below: None,
        },
        Hemisphere::Southern => {
            let mut colors = weather_palette(WeatherPalette::RelVort);
            colors.reverse();
            DiscreteColorScale {
                levels: range_step(-60.0, 40.1, 1.0),
                colors,
                extend: ExtendMode::Both,
                mask_below: None,
            }
        }
    }
}

pub fn operational_fill_scale_for_recipe(
    recipe: &PlotRecipe,
    filled_selector: FieldSelector,
) -> ColorScale {
    operational_fill_scale_for_recipe_in(
        recipe, filled_selector, Hemisphere::Northern)
}

pub fn operational_fill_scale_for_recipe_in(
    recipe: &PlotRecipe,
    filled_selector: FieldSelector,
    hemisphere: Hemisphere,
) -> ColorScale {
    let moisture_levels = match filled_selector.field {
        CanonicalField::IvtMagnitude | CanonicalField::FullColumnIvtMagnitude => Some(vec![0.,100.,250.,500.,750.,1000.,1250.,1500.,2000.]),
        CanonicalField::IvtEastward | CanonicalField::IvtNorthward | CanonicalField::FullColumnIvtEastward | CanonicalField::FullColumnIvtNorthward
            | CanonicalField::IntegratedVaporTransportU | CanonicalField::IntegratedVaporTransportV => Some(vec![-2000.,-1500.,-1000.,-750.,-500.,-250.,0.,250.,500.,750.,1000.,1500.,2000.]),
        CanonicalField::IntegratedWaterVapour => Some(vec![0.,5.,10.,15.,20.,25.,30.,35.,40.,50.,60.,70.]),
        _ if recipe.slug == "freezing_level" => Some(vec![0.,500.,1000.,1500.,2000.,2500.,3000.,3500.,4000.,5000.,6000.]),
        _ => None,
    };
    if let Some(levels)=moisture_levels {
        let signed=matches!(filled_selector.field,CanonicalField::IvtEastward|CanonicalField::IvtNorthward|CanonicalField::FullColumnIvtEastward|CanonicalField::FullColumnIvtNorthward|CanonicalField::IntegratedVaporTransportU|CanonicalField::IntegratedVaporTransportV);
        return ColorScale::Discrete(DiscreteColorScale {levels,colors:weather_palette(if signed {WeatherPalette::Advection} else if recipe.slug=="freezing_level" {WeatherPalette::IsothermHeight} else {WeatherPalette::Rh}),extend:ExtendMode::Both,mask_below:None});
    }

    if recipe.slug == "mslp_10m_winds" || recipe.slug == "gefs_avg_mslp_10m_winds" {
        return ColorScale::Discrete(ten_meter_wind_speed_scale());
    }

    if recipe.slug == "10m_wind_speed_and_direction" {
        return ColorScale::Discrete(standalone_ten_meter_wind_speed_scale());
    }

    if filled_selector.field == CanonicalField::SmokeMassDensity {
        return ColorScale::Discrete(DiscreteColorScale {
            levels: vec![10.0, 20.0, 35.0, 55.0, 100.0, 150.0, 250.0, 500.0],
            colors: smoke_scale_colors(),
            extend: ExtendMode::Max,
            mask_below: Some(10.0),
        });
    }
    if filled_selector.field == CanonicalField::ColumnIntegratedSmoke {
        return ColorScale::Discrete(DiscreteColorScale {
            levels: vec![20.0, 40.0, 80.0, 160.0, 320.0],
            colors: smoke_scale_colors(),
            extend: ExtendMode::Max,
            mask_below: Some(20.0),
        });
    }

    let chem_scale = match recipe.style {
        RenderStyle::AirQualityPm25 => Some(rustwx_render::theme::air_quality_scale(false)),
        RenderStyle::AirQualityOzone => Some(rustwx_render::theme::air_quality_scale(true)),
        RenderStyle::AerosolOpticalDepth => Some(rustwx_render::theme::aerosol_sequential_scale(true)),
        RenderStyle::DustConcentration => Some(rustwx_render::theme::aerosol_sequential_scale(false)),
        _ => None,
    };
    if let Some(scale) = chem_scale { return ColorScale::Discrete(scale); }

    let discrete = match recipe.style {
        RenderStyle::WeatherTemperature => {
            let (lo, hi, step, crop_f) = match filled_selector.vertical {
                VerticalSelector::HeightAboveGroundMeters(2) => {
                    (-60.0, 120.0, 1.0, Some((-60.0, 120.0)))
                }
                VerticalSelector::IsobaricHpa(200) => (-70.0, -29.0, 1.0, Some((-40.0, 70.0))),
                VerticalSelector::IsobaricHpa(250) => (-70.0, -29.0, 1.0, Some((-40.0, 70.0))),
                VerticalSelector::IsobaricHpa(300) => (-70.0, -29.0, 1.0, Some((-40.0, 70.0))),
                VerticalSelector::IsobaricHpa(500) => (-50.0, 6.0, 1.0, Some((-40.0, 70.0))),
                VerticalSelector::IsobaricHpa(700) => (-40.0, 26.0, 1.0, Some((-40.0, 90.0))),
                VerticalSelector::IsobaricHpa(850) => (-40.0, 40.0, 5.0, Some((-40.0, 110.0))),
                _ => (-50.0, 50.5, 0.5, Some((-40.0, 120.0))),
            };
            DiscreteColorScale {
                levels: range_step(lo, hi, step),
                colors: temperature_palette_cropped_f(
                    crop_f,
                    (((hi - lo) / step).round() as usize).max(2),
                ),
                extend: ExtendMode::Both,
                mask_below: None,
            }
        }
        RenderStyle::WeatherReflectivity | RenderStyle::WeatherRadarReflectivity => {
            reflectivity_dbz_scale()
        }
        RenderStyle::WeatherSpecificHumidity => DiscreteColorScale {
            levels: range_step(0.0, 20.1, 1.0),
            colors: weather_palette(WeatherPalette::Rh),
            extend: ExtendMode::Max, mask_below: None,
        },
        RenderStyle::WeatherOmega => DiscreteColorScale {
            levels: range_step(-2.0, 2.01, 0.1),
            colors: weather_palette(WeatherPalette::Advection),
            extend: ExtendMode::Both, mask_below: None,
        },
        RenderStyle::WeatherRh => relative_humidity_scale_for_selector(filled_selector),
        RenderStyle::WeatherProbability => DiscreteColorScale {
            levels: range_step(0.0, 101.0, 1.0),
            colors: weather_palette(WeatherPalette::Rh),
            extend: ExtendMode::Both,
            mask_below: None,
        },
        RenderStyle::WeatherVorticity => vorticity_scale_for_hemisphere(hemisphere),
        RenderStyle::WeatherDewpoint => dewpoint_scale_for_selector(filled_selector),
        RenderStyle::WeatherPressure => mslp_pressure_fill_scale(),
        RenderStyle::WeatherHeight => DiscreteColorScale {
            levels: match filled_selector.vertical {
                VerticalSelector::IsobaricHpa(200) | VerticalSelector::IsobaricHpa(250) => {
                    range_step(50.0, 170.0, 5.0)
                }
                VerticalSelector::IsobaricHpa(300) => range_step(20.0, 160.0, 5.0),
                VerticalSelector::IsobaricHpa(500) => range_step(20.0, 140.0, 5.0),
                VerticalSelector::IsobaricHpa(700) => range_step(20.0, 80.0, 5.0),
                VerticalSelector::IsobaricHpa(850) | VerticalSelector::IsobaricHpa(925) => {
                    range_step(20.0, 80.0, 5.0)
                }
                _ => range_step(10.0, 71.0, 1.0),
            },
            colors: match filled_selector.vertical {
                VerticalSelector::IsobaricHpa(200) | VerticalSelector::IsobaricHpa(250) => {
                    winds_palette_segments(120)
                }
                VerticalSelector::IsobaricHpa(300) => winds_palette_segments(100),
                VerticalSelector::IsobaricHpa(500) => winds_palette_segments(100),
                VerticalSelector::IsobaricHpa(700)
                | VerticalSelector::IsobaricHpa(850)
                | VerticalSelector::IsobaricHpa(925) => winds_palette_segments(70),
                _ => winds_palette_segments(60),
            },
            extend: ExtendMode::Both,
            mask_below: Some(match filled_selector.vertical {
                VerticalSelector::IsobaricHpa(200) | VerticalSelector::IsobaricHpa(250) => 50.0,
                VerticalSelector::IsobaricHpa(300)
                | VerticalSelector::IsobaricHpa(500)
                | VerticalSelector::IsobaricHpa(700)
                | VerticalSelector::IsobaricHpa(850)
                | VerticalSelector::IsobaricHpa(925) => 20.0,
                _ => 10.0,
            }),
        },
        RenderStyle::WeatherWindGust | RenderStyle::WeatherWinds => {
            wind_speed_scale_for_selector(filled_selector)
        }
        // One signed m/s scale for both heights and every frame. Blue is
        // descent, near-zero is neutral, orange/red is ascent.
        RenderStyle::WeatherVerticalVelocity => DiscreteColorScale {
            levels: vec![-20.0, -10.0, -5.0, -2.0, -1.0, -0.5, -0.1,
                         0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0],
            colors: vec![
                Color::rgba(8, 48, 107, 255), Color::rgba(8, 81, 156, 255),
                Color::rgba(33, 113, 181, 255), Color::rgba(66, 146, 198, 255),
                Color::rgba(107, 174, 214, 255), Color::rgba(158, 202, 225, 255),
                Color::rgba(198, 219, 239, 255), Color::rgba(245, 245, 245, 255),
                Color::rgba(254, 224, 182, 255), Color::rgba(253, 184, 99, 255),
                Color::rgba(244, 109, 67, 255), Color::rgba(215, 48, 39, 255),
                Color::rgba(178, 24, 43, 255), Color::rgba(128, 0, 38, 255),
                Color::rgba(77, 0, 25, 255),
            ],
            extend: ExtendMode::Both,
            mask_below: None,
        },
        RenderStyle::WeatherUh => DiscreteColorScale {
            // The low-level layers (WOOF uh-layers, 2026-11-02) run far weaker
            // than 2-5 km UH: on the 2024-05-06 outbreak 0-1 km peaked near 20
            // and 0-2 km near 135 m2/s2, so on the 0-400 bar their swaths were
            // near-white. 0-1 km takes the same palette over 0-100, 0-2 km over
            // 0-200; 0-3 km and 2-5 km keep 0-400.
            levels: match filled_selector.vertical {
                VerticalSelector::HeightAboveGroundLayerMeters { bottom_m: 0, top_m: 1000 } => {
                    let mut levels = range_step(0.0, 50.0, 1.25);
                    levels.extend(range_step(50.0, 100.25, 2.5).into_iter().skip(1));
                    levels
                }
                VerticalSelector::HeightAboveGroundLayerMeters { bottom_m: 0, top_m: 2000 } => {
                    let mut levels = range_step(0.0, 100.0, 2.5);
                    levels.extend(range_step(100.0, 200.5, 5.0).into_iter().skip(1));
                    levels
                }
                _ => {
                    let mut levels = range_step(0.0, 200.0, 5.0);
                    levels.extend(range_step(200.0, 401.0, 10.0).into_iter().skip(1));
                    levels
                }
            },
            colors: weather_palette(WeatherPalette::Uh),
            extend: ExtendMode::Both,
            mask_below: Some(0.0),
        },
        RenderStyle::WeatherTerrain => terrain_elevation_scale(),
        RenderStyle::WeatherIsothermHeight => isotherm_height_scale(),
        RenderStyle::WeatherSupercooledWaterPath => supercooled_water_path_scale(),
        RenderStyle::WeatherHydrometeorMixingRatio => hydrometeor_mixing_ratio_scale(),
        RenderStyle::WeatherCloudCover => cloud_cover_scale(),
        RenderStyle::WeatherPrecipitableWater => precipitable_water_inches_scale(),
        RenderStyle::WeatherIvt => DiscreteColorScale {
            levels: vec![0., 100., 250., 500., 750., 1000., 1250., 1500., 2000.],
            colors: vec![Color::rgba(245,245,245,255), Color::rgba(202,233,246,255),
                Color::rgba(105,194,165,255), Color::rgba(55,150,65,255),
                Color::rgba(242,220,62,255), Color::rgba(244,151,40,255),
                Color::rgba(220,62,42,255), Color::rgba(142,45,144,255)],
            extend: ExtendMode::Max, mask_below: None,
        },
        RenderStyle::WeatherQpf => crate::qpf::qpf_inches_scale(),
        RenderStyle::WeatherSnowfall => crate::qpf::snowfall_inches_scale(),
        RenderStyle::WeatherSnowDepthChange => {
            let mut scale = rustwx_render::difference::difference_scale(rustwx_render::difference::DifferenceTable::builtin(), 24.0);
            // Preserve sub-inch melt/settling without losing the storm-depth tails.
            scale.levels = vec![-24., -8., -2., -0.5, -0.1, 0., 0.1, 0.5, 2., 8., 24.];
            scale
        },
        RenderStyle::WeatherHail => DiscreteColorScale {
            levels: vec![0.0, 5.0, 10.0, 20.0, 25.4, 38.1, 50.8, 76.2, 101.6, 127.0, 152.4, 203.2],
            colors: weather_palette(WeatherPalette::Uh),
            extend: ExtendMode::Max,
            mask_below: Some(5.0),
        },
        // Binary flags (fog flag, categorical rain/snow/freezing rain/ice
        // pellets): one band per code, 0 and 1, drawn with the category
        // legend (`direct_recipe_render_controls`). The old 0-1 ramp
        // (levels 0, 0.5, 1) was densified like a continuous field, so the
        // bar read 0.1 .. 0.9 and the map blended pink fringes at every
        // 0/1 edge (R10 look, 2.9.4). The fog flag draws its 0 band; the
        // categorical precipitation flags keep masking false flags, so the
        // map shows only where the type is diagnosed.
        RenderStyle::WeatherCategorical => DiscreteColorScale {
            levels: vec![-0.5, 0.5, 1.5],
            colors: vec![
                Color::rgba(242, 242, 242, 255),
                Color::rgba(216, 34, 34, 255),
            ],
            extend: ExtendMode::Neither,
            mask_below: (filled_selector.field != CanonicalField::FogFlag).then_some(0.5),
        },
        // Aviation display range; larger values retain the final clear-air color.
        // The diagnostic itself remains uncapped by this 2 km display range.
        RenderStyle::WeatherVisibility => DiscreteColorScale {
            levels: vec![0.0, 100.0, 200.0, 400.0, 600.0, 1000.0, 1600.0, 2000.0],
            colors: vec![
                Color::rgba(128, 0, 38, 255), Color::rgba(189, 0, 38, 255),
                Color::rgba(240, 59, 32, 255), Color::rgba(253, 141, 60, 255),
                Color::rgba(254, 217, 118, 255), Color::rgba(199, 233, 180, 255),
                Color::rgba(127, 205, 187, 255), Color::rgba(65, 182, 196, 255),
            ],
            extend: ExtendMode::Max,
            mask_below: None,
        },
        RenderStyle::WeatherSatellite => DiscreteColorScale {
            levels: range_step(170.0, 321.0, 2.0),
            colors: weather_palette(WeatherPalette::SimIr),
            extend: ExtendMode::Both,
            mask_below: None,
        },
        RenderStyle::WeatherWbgt => DiscreteColorScale {
            levels: range_step(10.0, 46.0, 1.0), colors: weather_palette(WeatherPalette::Temperature),
            extend: ExtendMode::Both, mask_below: None,
        },
        RenderStyle::WeatherHdw => DiscreteColorScale {
            levels: vec![0.0,10.0,25.0,50.0,100.0,150.0,200.0,300.0,400.0,600.0,800.0,1200.0], colors: weather_palette(WeatherPalette::Uh),
            extend: ExtendMode::Max, mask_below: None,
        },
        RenderStyle::WeatherHaines => DiscreteColorScale {
            levels: vec![1.5,2.5,3.5,4.5,5.5,6.5],
            colors: vec![Color::rgba(80,180,95,255), Color::rgba(245,220,70,255), Color::rgba(245,145,40,255), Color::rgba(220,55,40,255), Color::rgba(130,25,60,255)],
            extend: ExtendMode::Neither, mask_below: None,
        },
        RenderStyle::WeatherLightningThreat => DiscreteColorScale {
            // Threshold legend preserves weak, spatially sparse signals.
            levels: vec![0.0,0.001,0.01,0.05,0.1,0.25,0.5,1.0,2.0,5.0,10.0,20.0],
            colors: vec![Color::rgba(255,255,255,255), Color::rgba(170,220,245,255), Color::rgba(50,190,230,255), Color::rgba(40,100,210,255), Color::rgba(40,180,100,255), Color::rgba(245,220,50,255), Color::rgba(245,150,35,255), Color::rgba(225,55,40,255), Color::rgba(160,45,170,255), Color::rgba(230,100,190,255), Color::rgba(100,20,110,255)],
            extend: ExtendMode::Max, mask_below: None,
        },
        RenderStyle::WeatherFlashRate => DiscreteColorScale {
            // Domain-distributed PR92 rates can be much smaller than 1e-4.
            levels: vec![0.0,1e-7,1e-6,1e-5,1e-4,0.001,0.01,0.1,1.0],
            colors: vec![Color::rgba(255,255,255,255), Color::rgba(50,190,230,255), Color::rgba(40,100,210,255), Color::rgba(40,180,100,255), Color::rgba(245,150,35,255), Color::rgba(225,55,40,255), Color::rgba(160,45,170,255), Color::rgba(230,100,190,255)],
            extend: ExtendMode::Max, mask_below: None,
        },
        RenderStyle::WeatherLightning => DiscreteColorScale {
            levels: range_step(0.0, 20.5, 0.5),
            colors: weather_palette(WeatherPalette::Uh),
            extend: ExtendMode::Max,
            mask_below: Some(0.5),
        },
        _ => DiscreteColorScale {
            levels: range_step(-50.0, 5.0, 1.0),
            colors: weather_palette(WeatherPalette::Temperature),
            extend: ExtendMode::Both,
            mask_below: None,
        },
    };
    ColorScale::Discrete(discrete)
}

pub fn operational_contour_layer_for_values(
    selector: FieldSelector,
    values: &[f32],
) -> Option<ContourLayer> {
    let data = if selector.field == CanonicalField::GeopotentialHeight {
        values.iter().map(|value| value * 0.1).collect()
    } else if selector.field == CanonicalField::PressureReducedToMeanSeaLevel {
        values.iter().map(|value| value * 0.01).collect()
    } else {
        values.to_vec()
    };
    let (levels, color, width, labels, major_every, major_width, show_extrema) = match selector {
        FieldSelector { field: CanonicalField::IvtMagnitude, .. } => (vec![250.0], Color::BLACK, 2, true, None, None, false),
        FieldSelector {
            field: CanonicalField::GeopotentialHeight,
            vertical: rustwx_core::VerticalSelector::IsobaricHpa(200),
            ..
        } => operational_height_contour_policy(range_step(1020.0, 1321.0, 6.0)),
        FieldSelector {
            field: CanonicalField::GeopotentialHeight,
            vertical: rustwx_core::VerticalSelector::IsobaricHpa(300),
            ..
        } => operational_height_contour_policy(range_step(700.0, 1101.0, 6.0)),
        FieldSelector {
            field: CanonicalField::GeopotentialHeight,
            vertical: rustwx_core::VerticalSelector::IsobaricHpa(250),
            ..
        } => operational_height_contour_policy(range_step(900.0, 1201.0, 6.0)),
        FieldSelector {
            field: CanonicalField::GeopotentialHeight,
            vertical: rustwx_core::VerticalSelector::IsobaricHpa(500),
            ..
        } => operational_height_contour_policy(range_step(450.0, 651.0, 6.0)),
        FieldSelector {
            field: CanonicalField::GeopotentialHeight,
            vertical: rustwx_core::VerticalSelector::IsobaricHpa(700),
            ..
        } => operational_height_contour_policy(range_step(100.0, 401.0, 6.0)),
        FieldSelector {
            field: CanonicalField::GeopotentialHeight,
            vertical: rustwx_core::VerticalSelector::IsobaricHpa(850),
            ..
        } => operational_height_contour_policy(range_step(0.0, 201.0, 6.0)),
        FieldSelector {
            field: CanonicalField::PressureReducedToMeanSeaLevel,
            ..
        } => {
            // A theme may name its own isobar set (the `tc` sheet: every
            // 4 hPa from 880 hPa); absent one, the operational 960-1045 hPa.
            let theme = rustwx_render::theme::active_theme();
            match theme.contour_levels("mslp") {
                Some(levels) => {
                    let (_, color, width, labels, major_every, major_width, show_extrema) =
                        operational_pressure_contour_policy(Vec::new());
                    (
                        levels.to_vec(),
                        color,
                        width,
                        labels,
                        theme.contour_label_every("mslp").or(major_every),
                        major_width,
                        show_extrema,
                    )
                }
                None => operational_pressure_contour_policy(range_step(960.0, 1045.0, 2.0)),
            }
        }
        FieldSelector {
            field: CanonicalField::UpdraftHelicity,
            vertical:
                rustwx_core::VerticalSelector::HeightAboveGroundLayerMeters {
                    bottom_m: 2000,
                    top_m: 5000,
                },
            ..
        } => (vec![75.0], Color::BLACK, 1, false, None, None, false),
        // The low-level layers (WOOF uh-layers, 2026-11-02) run weaker than
        // 2-5 km UH. 0-1 km: 10, 20 and 40 m2/s2 (a 3 km severe-day run,
        // 2024-05-06 00Z, peaked at 20 there against 276 at 2-5 km); 0-2 and
        // 0-3 km: 25, 50 and 75, the ensemble probability thresholds.
        FieldSelector {
            field: CanonicalField::UpdraftHelicity,
            vertical:
                rustwx_core::VerticalSelector::HeightAboveGroundLayerMeters {
                    bottom_m: 0,
                    top_m: 1000,
                },
            ..
        } => (vec![10.0, 20.0, 40.0], Color::BLACK, 1, false, None, None, false),
        FieldSelector {
            field: CanonicalField::UpdraftHelicity,
            vertical:
                rustwx_core::VerticalSelector::HeightAboveGroundLayerMeters {
                    bottom_m: 0,
                    top_m: 2000 | 3000,
                },
            ..
        } => (vec![25.0, 50.0, 75.0], Color::BLACK, 1, false, None, None, false),
        _ => (
            range_step(0.0, 200.0, 10.0),
            Color::BLACK,
            1,
            true,
            Some(2),
            Some(2),
            false,
        ),
    };

    Some(ContourLayer {
        data,
        levels,
        color,
        width,
        labels,
        show_extrema,
        pattern: ContourLinePattern::Solid,
        major_every,
        major_width,
    })
}

pub fn operational_wind_streamline_style(stride_x: usize, stride_y: usize) -> WindStreamlineStyle {
    WindStreamlineStyle {
        stride_x: stride_x.max(1),
        stride_y: stride_y.max(1),
        color: Color::rgba(18, 24, 32, 72),
        width: 1,
        max_steps: 14,
        step_cells: 0.80,
        min_speed: 2.5,
    }
}

fn operational_height_contour_policy(
    levels: Vec<f64>,
) -> (Vec<f64>, Color, u32, bool, Option<usize>, Option<u32>, bool) {
    (
        levels,
        Color::rgba(0, 0, 0, 220),
        1,
        true,
        Some(2),
        Some(2),
        false,
    )
}

fn operational_pressure_contour_policy(
    levels: Vec<f64>,
) -> (Vec<f64>, Color, u32, bool, Option<usize>, Option<u32>, bool) {
    (levels, Color::BLACK, 1, true, Some(2), Some(2), true)
}

/// The catalog fill scale for MSLP values in hPa (the `WeatherPressure`
/// recipe arm, e.g. the HREF MSLP mean fill). NOTE: the production
/// `mslp_10m_winds` plot fills the companion 10 m wind speed and only
/// CONTOURS mslp at 2 hPa, no single-model production colorbar shows
/// these pressure values, which is why the store viewer resolver keeps the
/// generic ramp for the stored `mslp` plane instead of this scale.
pub(crate) fn mslp_pressure_fill_scale() -> DiscreteColorScale {
    DiscreteColorScale {
        levels: range_step(960.0, 1045.0, 2.0),
        colors: weather_palette(WeatherPalette::Winds),
        extend: ExtendMode::Both,
        mask_below: None,
    }
}

/// The reflectivity products' ladder: composite reflectivity, composite
/// reflectivity with UH, 1 km reflectivity, and through
/// `rw_wrfbatch::scales::reflectivity_scale` the radar PPIs and the
/// observed reflectivity grids. It is the radar reflectivity table
/// (`rustwx_render::RadarTable::Reflectivity`, the AWIPS-style table
/// transcribed from BowEcho) in 1 dBZ bins from the 10 dBZ display floor
/// to 85 dBZ, its top row held above. The ladder it replaced,
/// [`classic_reflectivity_dbz_scale`], is what the `classic` radar colour
/// set selects (`rw_wrfbatch --radar-colors classic`, `RUSTWX_RADAR_COLORS`,
/// `[simulated_radar] color_tables`).
fn reflectivity_dbz_scale() -> DiscreteColorScale {
    reflectivity_dbz_scale_for(rustwx_render::active_radar_color_set())
}

/// The reflectivity ladder of one radar colour set.
pub fn reflectivity_dbz_scale_for(set: rustwx_render::RadarColorSet) -> DiscreteColorScale {
    match set {
        rustwx_render::RadarColorSet::Standard => rustwx_render::RadarTable::Reflectivity
            .scale(10.0, 85.0, 1.0, ExtendMode::Max, Some(10.0)),
        rustwx_render::RadarColorSet::Classic => classic_reflectivity_dbz_scale(),
    }
}

/// The twelve-step reflectivity ladder the reflectivity products wore
/// before the radar reflectivity table: the `classic` radar colour set's
/// reflectivity, and `reflectivity_classic` by table name.
pub fn classic_reflectivity_dbz_scale() -> DiscreteColorScale {
    DiscreteColorScale {
        levels: vec![
            10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0,
        ],
        colors: vec![
            Color::rgba(242, 246, 252, 255),
            Color::rgba(150, 183, 232, 255),
            Color::rgba(55, 105, 195, 255),
            Color::rgba(20, 94, 133, 255),
            Color::rgba(45, 126, 76, 255),
            Color::rgba(132, 169, 80, 255),
            Color::rgba(246, 226, 82, 255),
            Color::rgba(237, 143, 42, 255),
            Color::rgba(211, 32, 28, 255),
            Color::rgba(147, 5, 21, 255),
            Color::rgba(132, 34, 157, 255),
            Color::rgba(178, 178, 178, 255),
        ],
        extend: ExtendMode::Max,
        mask_below: Some(10.0),
    }
}

fn wind_speed_scale_for_selector(selector: FieldSelector) -> DiscreteColorScale {
    let levels = match selector.vertical {
        VerticalSelector::IsobaricHpa(200) | VerticalSelector::IsobaricHpa(250) => {
            range_step(50.0, 170.0, 5.0)
        }
        VerticalSelector::IsobaricHpa(500) => range_step(20.0, 140.0, 5.0),
        VerticalSelector::IsobaricHpa(700)
        | VerticalSelector::IsobaricHpa(850)
        | VerticalSelector::IsobaricHpa(925) => range_step(20.0, 80.0, 5.0),
        VerticalSelector::HeightAboveGroundMeters(10) => range_step(10.0, 60.0, 5.0),
        _ => range_step(10.0, 80.0, 5.0),
    };
    DiscreteColorScale {
        levels,
        colors: winds_palette_segments(90),
        extend: ExtendMode::Max,
        mask_below: Some(match selector.vertical {
            VerticalSelector::IsobaricHpa(200) | VerticalSelector::IsobaricHpa(250) => 50.0,
            VerticalSelector::IsobaricHpa(500)
            | VerticalSelector::IsobaricHpa(700)
            | VerticalSelector::IsobaricHpa(850)
            | VerticalSelector::IsobaricHpa(925) => 20.0,
            VerticalSelector::HeightAboveGroundMeters(10) => 10.0,
            _ => 10.0,
        }),
    }
}

fn ten_meter_wind_speed_scale() -> DiscreteColorScale {
    DiscreteColorScale {
        levels: range_step(10.0, 60.0, 5.0),
        colors: winds_palette_segments(60),
        extend: ExtendMode::Max,
        mask_below: Some(10.0),
    }
}

/// Fill scale for the STANDALONE 10 m wind chart (knots).
///
/// Deliberately NOT [`ten_meter_wind_speed_scale`].  That one masks
/// everything below 10 kt because it is an OVERLAY on an MSLP analysis:
/// the pressure pattern is the subject and the colour only marks where
/// the wind is strong enough to matter, so blanking the calm half of the
/// map is the right call there.
///
/// On a chart whose whole subject IS the wind, the same mask draws an
/// empty map on exactly the day a forecaster most needs to see the
/// field -- a light-wind morning, or the calm side of a wind shift.  So
/// this scale starts at zero, masks nothing, and steps at 2.5 kt so the
/// low end is resolved rather than lumped.  It still extends past the
/// top of the ramp rather than clipping.
fn standalone_ten_meter_wind_speed_scale() -> DiscreteColorScale {
    DiscreteColorScale {
        levels: range_step(0.0, 60.0, 2.5),
        colors: winds_palette_segments(60),
        extend: ExtendMode::Max,
        mask_below: None,
    }
}

fn dewpoint_scale_for_selector(selector: FieldSelector) -> DiscreteColorScale {
    match selector.vertical {
        VerticalSelector::HeightAboveGroundMeters(2) => {
            let levels = range_step(-40.0, 90.0, 1.0);
            DiscreteColorScale {
                colors: surface_dewpoint_colors(),
                levels,
                extend: ExtendMode::Both,
                mask_below: None,
            }
        }
        VerticalSelector::IsobaricHpa(_) => {
            let levels = range_step(-40.0, 31.0, 1.0);
            DiscreteColorScale {
                colors: dewpoint_palette_celsius_for_levels(&levels),
                levels,
                extend: ExtendMode::Both,
                mask_below: None,
            }
        }
        _ => {
            let levels = range_step(-40.0, 90.0, 1.0);
            DiscreteColorScale {
                colors: surface_dewpoint_colors(),
                levels,
                extend: ExtendMode::Both,
                mask_below: None,
            }
        }
    }
}

fn relative_humidity_scale_for_selector(selector: FieldSelector) -> DiscreteColorScale {
    match selector.vertical {
        VerticalSelector::HeightAboveGroundMeters(2) => DiscreteColorScale {
            levels: range_step(0.0, 100.0, 5.0),
            colors: surface_relative_humidity_colors(),
            extend: ExtendMode::Max,
            mask_below: None,
        },
        _ => DiscreteColorScale {
            levels: range_step(0.0, 101.0, 1.0),
            colors: weather_palette(WeatherPalette::Rh),
            extend: ExtendMode::Both,
            mask_below: None,
        },
    }
}

fn surface_dewpoint_colors() -> Vec<Color> {
    let mut colors = weather_palette(WeatherPalette::Dewpoint);
    if colors.len() <= 1 {
        return colors;
    }

    colors.remove(0);
    if let Some(last) = colors.last().copied() {
        colors.push(last);
    }
    colors
}

fn surface_relative_humidity_colors() -> Vec<Color> {
    vec![
        Color::rgba(140, 45, 4, 255),
        Color::rgba(204, 76, 2, 255),
        Color::rgba(236, 112, 20, 255),
        Color::rgba(254, 153, 41, 255),
        Color::rgba(254, 196, 79, 255),
        Color::rgba(255, 247, 188, 255),
        Color::rgba(224, 243, 219, 255),
        Color::rgba(168, 221, 181, 255),
        Color::rgba(67, 162, 202, 255),
        Color::rgba(8, 104, 172, 255),
    ]
}

/// Cloud cover in percent.  Under the first level (10 %) the sky is clear
/// and nothing is drawn, so the basemap shows through as it does under the
/// reflectivity and QPF ladders.  Before, clear sky took the ladder's
/// first colour: a clear 750 m afternoon drew as one white sheet, which on
/// a dark theme is a glaring slab and on a light one hides land and water,
/// and white is the colour a reader takes for cloud.
fn cloud_cover_scale() -> DiscreteColorScale {
    DiscreteColorScale {
        levels: range_step(10.0, 100.0, 10.0),
        colors: vec![
            Color::rgba(255, 255, 255, 255),
            Color::rgba(222, 222, 222, 255),
            Color::rgba(178, 178, 178, 255),
            Color::rgba(128, 128, 128, 255),
            Color::rgba(70, 80, 100, 255),
            Color::rgba(35, 68, 122, 255),
            Color::rgba(38, 111, 166, 255),
            Color::rgba(103, 177, 209, 255),
            Color::rgba(189, 232, 241, 255),
        ],
        extend: ExtendMode::Max,
        mask_below: Some(10.0),
    }
}

fn precipitable_water_inches_scale() -> DiscreteColorScale {
    DiscreteColorScale {
        levels: vec![
            0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00,
        ],
        colors: vec![
            Color::rgba(70, 55, 44, 255),
            Color::rgba(118, 108, 94, 255),
            Color::rgba(213, 211, 189, 255),
            Color::rgba(183, 224, 175, 255),
            Color::rgba(105, 191, 105, 255),
            Color::rgba(32, 137, 67, 255),
            Color::rgba(16, 111, 101, 255),
            Color::rgba(39, 124, 158, 255),
            Color::rgba(63, 95, 168, 255),
            Color::rgba(116, 74, 165, 255),
            Color::rgba(191, 127, 177, 255),
        ],
        extend: ExtendMode::Both,
        mask_below: None,
    }
}

/// The height of an isotherm, metres above sea level, 0 to 9 km in
/// 500 m bands: a -20 C surface reaches 8 km over summer ground and a
/// 0 C surface sits on the ground in winter.  Fixed, like every other
/// operational scale here, so two frames of one run compare.
fn isotherm_height_scale() -> DiscreteColorScale {
    let levels = range_step(0.0, 9001.0, 500.0);
    let colors = resampled_palette(
        &weather_palette(WeatherPalette::IsothermHeight),
        levels.len().saturating_sub(1),
    );
    DiscreteColorScale {
        levels,
        colors,
        extend: ExtendMode::Max,
        mask_below: None,
    }
}

/// A supercooled liquid water path, g m-2, 0 to 1000 in 50 g m-2 bands.
/// Ten g m-2 is the floor an icing reader cares about and is masked
/// below.  The bands are even because the bar is drawn linear in value
/// and the fill is sampled along it: bands that widened upward left the
/// last one a quarter of the bar, drawn purple on the map and navy on the
/// bar.
fn supercooled_water_path_scale() -> DiscreteColorScale {
    let levels = range_step(0.0, 1000.0, 50.0);
    let colors = resampled_palette(
        &weather_palette(WeatherPalette::SupercooledWater),
        levels.len().saturating_sub(1),
    );
    DiscreteColorScale {
        levels,
        colors,
        extend: ExtendMode::Max,
        mask_below: Some(10.0),
    }
}

/// A hydrometeor mixing ratio, g kg-1, 0 to 5 in quarter-gram bands,
/// masked below a hundredth of a gram: the range a column maximum of
/// cloud water, rain, ice, snow or graupel spans in a resolved storm.
/// Even bands for the reason the water path has them.
fn hydrometeor_mixing_ratio_scale() -> DiscreteColorScale {
    let levels = range_step(0.0, 5.0, 0.25);
    let colors = resampled_palette(
        &weather_palette(WeatherPalette::Hydrometeor),
        levels.len().saturating_sub(1),
    );
    DiscreteColorScale {
        levels,
        colors,
        extend: ExtendMode::Max,
        mask_below: Some(0.01),
    }
}

/// `n` bands linearly interpolated through `anchors`, so a level set has
/// exactly one colour per interval whatever its length.
/// An isobaric temperature ladder moved onto the field it colours.
///
/// The ladders above are fixed per level for mid-latitude air (300 hPa:
/// -70..-29 C).  A tropical domain sits wholly above the warm end: Milton's
/// d02 at f000 (2.9.4 rolling pass) drew one saturated colour over the
/// whole map.  When the field's 2nd..98th percentile already lies inside
/// the ladder it is returned unchanged, byte for byte, so mid-latitude maps
/// and the comparability of two of them are untouched.  Otherwise the
/// ladder keeps its step and width and is moved, by whole steps, to centre
/// on the field, narrowed to the field's span with a margin (at least 12
/// steps, never wider than the fixed ladder); a field wider than the ladder
/// widens it to the field.  The palette is resampled over the new count.  The same colours, in the same
/// order, still run cold to warm.
pub fn fit_isobaric_temperature_scale(
    scale: DiscreteColorScale,
    values: &[f32],
) -> DiscreteColorScale {
    let levels = &scale.levels;
    if levels.len() < 3 {
        return scale;
    }
    let mut finite: Vec<f32> = values.iter().copied().filter(|v| v.is_finite()).collect();
    if finite.len() < 16 {
        return scale;
    }
    let mut quantile = |q: f64| {
        let index = ((finite.len() - 1) as f64 * q).round() as usize;
        let (_, value, _) = finite.select_nth_unstable_by(index, |a, b| a.total_cmp(b));
        f64::from(*value)
    };
    let (low, high) = (quantile(0.02), quantile(0.98));
    let step = levels[1] - levels[0];
    let (lo, last) = (levels[0], levels[levels.len() - 1]);
    // The ladder runs lo..=last (range_step is inclusive).
    if !(step > 0.0) || (low >= lo && high <= last) {
        return scale;
    }
    let width = last - lo;
    let (start, stop) = if high - low <= width {
        // As wide as the field needs with a margin (at least 12 steps, at
        // most the fixed ladder), so a narrow tropical field gets the
        // palette's contrast instead of a third of it.
        let fitted = (((high - low) * 1.6 / step).ceil() * step).clamp(12.0 * step, width);
        let start = (((low + high) * 0.5 - fitted * 0.5) / step).floor() * step;
        (start, start + fitted)
    } else {
        ((low / step).floor() * step, (high / step).ceil() * step)
    };
    let new_levels = range_step(start, stop, step);
    // Keep the ladder's own colours-per-level relation (one colour per
    // interval for these ladders).
    let colors = if new_levels.len() == levels.len() {
        scale.colors.clone()
    } else {
        let count = (new_levels.len() + scale.colors.len()).saturating_sub(levels.len());
        resampled_palette(&scale.colors, count)
    };
    DiscreteColorScale { levels: new_levels, colors, ..scale }
}

/// Keep strong surface winds visible instead of saturating the 60 kt bar.
/// Inputs are already in the recipe's display units (knots). Ordinary-wind
/// charts retain their exact scale; stronger fields extend to a 20 kt boundary.
/// Each legend states its range, so comparisons must read the labelled bars.
pub fn fit_surface_wind_scale(scale: DiscreteColorScale, values: &[f32]) -> DiscreteColorScale {
    if scale.levels.len() < 2 { return scale; }
    let low = scale.levels[0];
    let high = *scale.levels.last().unwrap();
    let peak = values.iter().copied().filter(|x| x.is_finite())
        .map(f64::from).fold(high, f64::max);
    if peak <= high { return scale; }
    let upper = (peak / 20.0).ceil() * 20.0;
    let step = (scale.levels[1] - low).max(((upper-low)/48.0/5.0).ceil()*5.0);
    let count = ((upper - low) / step).ceil() as usize;
    let levels: Vec<f64> = (0..=count).map(|i| low + i as f64 * step).collect();
    let colors = resampled_palette(&scale.colors, levels.len().saturating_sub(1));
    DiscreteColorScale { levels, colors, ..scale }
}

fn resampled_palette(anchors: &[Color], n: usize) -> Vec<Color> {
    if n == 0 || anchors.is_empty() {
        return Vec::new();
    }
    if anchors.len() == 1 || n == 1 {
        return vec![anchors[0]; n];
    }
    (0..n)
        .map(|index| {
            let t = index as f64 / (n - 1) as f64;
            let position = t * (anchors.len() - 1) as f64;
            let lower = (position.floor() as usize).min(anchors.len() - 2);
            let fraction = position - lower as f64;
            let a = anchors[lower];
            let b = anchors[lower + 1];
            let mix = |x: u8, y: u8| (f64::from(x) + (f64::from(y) - f64::from(x)) * fraction).round() as u8;
            Color::rgba(mix(a.r, b.r), mix(a.g, b.g), mix(a.b, b.b), mix(a.a, b.a))
        })
        .collect()
}

/// Surface elevation in metres, on a hypsometric ramp.
///
/// The level set is deliberately non-uniform -- 25 m near sea level,
/// 250 m above 2 km -- because a linear ramp sized for the Rockies
/// renders a coastal or plains domain as one flat colour, and the whole
/// point of the product is to show a viewer the relief that is steering
/// the wind.  Fixed levels (not per-frame autoscaling) match every other
/// operational scale in this module: two domains stay comparable.
fn terrain_elevation_scale() -> DiscreteColorScale {
    let levels = terrain_elevation_levels_m();
    let colors = terrain_palette(levels.len().saturating_sub(1));
    DiscreteColorScale {
        levels,
        colors,
        extend: ExtendMode::Both,
        mask_below: None,
    }
}

fn terrain_elevation_levels_m() -> Vec<f64> {
    vec![
        0.0, 25.0, 50.0, 75.0, 100.0, 150.0, 200.0, 250.0, 300.0, 400.0, 500.0, 600.0, 700.0,
        800.0, 900.0, 1000.0, 1200.0, 1400.0, 1600.0, 1800.0, 2000.0, 2250.0, 2500.0, 2750.0,
        3000.0, 3250.0, 3500.0, 3750.0, 4000.0, 4250.0, 4500.0,
    ]
}

/// `n` interpolated bands across the hypsometric control points, so the
/// palette always has exactly one colour per level interval rather than
/// relying on the colormap's coarse index sampling.
fn terrain_palette(n: usize) -> Vec<Color> {
    const CONTROL: &[(u8, u8, u8)] = &[
        (63, 122, 90),
        (104, 155, 92),
        (150, 179, 97),
        (196, 199, 114),
        (222, 205, 140),
        (217, 184, 138),
        (198, 152, 112),
        (171, 122, 92),
        (146, 104, 88),
        (170, 155, 150),
        (216, 214, 216),
        (247, 247, 250),
    ];
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        let (r, g, b) = CONTROL[0];
        return vec![Color::rgba(r, g, b, 255)];
    }
    (0..n)
        .map(|index| {
            let t = index as f64 / (n - 1) as f64;
            let scaled = t * (CONTROL.len() - 1) as f64;
            let lower = scaled.floor() as usize;
            let upper = (lower + 1).min(CONTROL.len() - 1);
            let frac = scaled - lower as f64;
            let mix = |a: u8, b: u8| -> u8 {
                (f64::from(a) + (f64::from(b) - f64::from(a)) * frac).round() as u8
            };
            let (ar, ag, ab) = CONTROL[lower];
            let (br, bg, bb) = CONTROL[upper];
            Color::rgba(mix(ar, br), mix(ag, bg), mix(ab, bb), 255)
        })
        .collect()
}

/// The tracer ramp the map families paint a carried scalar with: cool blue
/// through green and yellow into red and violet, with the alpha ramping up
/// so a thin plume lets the map through underneath it.
pub fn tracer_scale_colors() -> Vec<Color> {
    vec![
        Color::rgba(82, 185, 226, 42),
        Color::rgba(84, 210, 238, 78),
        Color::rgba(116, 230, 140, 116),
        Color::rgba(247, 232, 65, 160),
        Color::rgba(255, 169, 42, 200),
        Color::rgba(244, 76, 31, 226),
        Color::rgba(218, 10, 36, 238),
        Color::rgba(135, 0, 150, 246),
        Color::rgba(78, 0, 138, 252),
        Color::rgba(48, 0, 112, 255),
    ]
}

/// The same ramp at FULL saturation: identical hues, every stop opaque.
///
/// A map overlays its tracer on basemap and fields and needs the alpha
/// ramp; a cross-section's fill is the subject of the panel and has
/// nothing underneath it worth showing through, so the same plume drawn
/// with the map's alphas came out as a pale wash.  One ramp, two
/// exposures.
pub fn tracer_scale_colors_saturated() -> Vec<Color> {
    tracer_scale_colors()
        .into_iter()
        .map(|color| Color::rgba(color.r, color.g, color.b, 255))
        .collect()
}

fn smoke_scale_colors() -> Vec<Color> {
    tracer_scale_colors()
}

#[cfg(test)]
mod isobaric_temperature_fit_tests {
    use super::*;

    fn ladder_300() -> DiscreteColorScale {
        let recipe = rustwx_models::plot_recipe("300mb_temperature_height_winds").unwrap();
        let selector = FieldSelector::isobaric(CanonicalField::Temperature, 300);
        match operational_fill_scale_for_recipe(recipe, selector) {
            ColorScale::Discrete(scale) => scale,
            _ => panic!("discrete"),
        }
    }

    #[test]
    fn a_midlatitude_field_keeps_the_fixed_ladder_byte_for_byte() {
        let fixed = ladder_300();
        let values: Vec<f32> = (0..1000).map(|i| -60.0 + 25.0 * i as f32 / 999.0).collect();
        let fitted = fit_isobaric_temperature_scale(fixed.clone(), &values);
        assert_eq!(fitted.levels, fixed.levels);
        assert_eq!(fitted.colors, fixed.colors);
    }

    #[test]
    fn a_tropical_field_moves_the_ladder_onto_itself() {
        // Milton-like 300 hPa air: -33..-24 C, above the fixed -29 C top.
        let fixed = ladder_300();
        let values: Vec<f32> = (0..1000).map(|i| -33.0 + 9.0 * i as f32 / 999.0).collect();
        let fitted = fit_isobaric_temperature_scale(fixed.clone(), &values);
        let (lo, hi) = (fitted.levels[0], *fitted.levels.last().unwrap());
        assert!(lo <= -33.0 && hi >= -24.0, "ladder {lo}..{hi} misses the field");
        // Fitted to the field's span (9 C, x1.6 margin -> 15 steps), not the
        // fixed 41-step width, so the palette's contrast lands on the field.
        assert!(hi - lo <= 20.0, "ladder {lo}..{hi} is wider than the field needs");
        assert_eq!(fitted.levels.len() - fitted.colors.len(), fixed.levels.len() - fixed.colors.len());
        assert_eq!(fitted.levels[1] - fitted.levels[0], fixed.levels[1] - fixed.levels[0]);
        assert_eq!(fitted.colors.first(), fixed.colors.first());
        assert_eq!(fitted.colors.last(), fixed.colors.last());
    }
}

fn range_step(start: f64, stop: f64, step: f64) -> Vec<f64> {
    let mut out = Vec::new();
    let mut value = start;
    while value <= stop + 1e-9 {
        out.push(value);
        value += step;
    }
    out
}

fn normalize_longitude_for_bounds(lon: f64) -> f64 {
    let mut lon = lon % 360.0;
    if lon > 180.0 {
        lon -= 360.0;
    } else if lon <= -180.0 {
        lon += 360.0;
    }
    lon
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustwx_core::{CanonicalField, Field2D, FieldSelector, GridShape, LatLonGrid, ProductKey};
    use rustwx_render::{ColorScale, DiscreteColorScale, ExtendMode};

    #[test]
    fn surface_wind_scale_keeps_ordinary_fields_and_covers_extremes() {
        let recipe = rustwx_models::plot_recipe("10m_wind_speed_and_direction").unwrap();
        let selector = recipe.filled.selector.unwrap();
        let ColorScale::Discrete(fixed) = operational_fill_scale_for_recipe(recipe, selector) else {
            panic!("surface wind must use discrete colors");
        };
        let unchanged = fit_surface_wind_scale(fixed.clone(), &[0.0, 30.0, f32::NAN]);
        assert_eq!(unchanged.levels, fixed.levels);
        assert_eq!(unchanged.colors, fixed.colors);
        for peak in [60.1_f32, 98.0, 175.0, 507.0] {
            let fitted = fit_surface_wind_scale(fixed.clone(), &[0.0, peak, f32::NAN]);
            assert!(*fitted.levels.last().unwrap() >= f64::from(peak));
            assert!(fitted.levels.len() <= 50);
            assert_eq!(fitted.levels.len(), fitted.colors.len() + 1);
            assert_eq!(fitted.colors.first(), fixed.colors.first());
            assert_eq!(fitted.colors.last(), fixed.colors.last());
        }
    }

    fn sample_request() -> MapRenderRequest {
        let shape = GridShape::new(2, 2).unwrap();
        let grid = LatLonGrid::new(
            shape,
            vec![35.0, 35.0, 36.0, 36.0],
            vec![-100.0, -99.0, -100.0, -99.0],
        )
        .unwrap();
        let field = Field2D::new(
            ProductKey::named("sample"),
            "unit",
            grid,
            vec![0.0, 1.0, 2.0, 3.0],
        )
        .unwrap();
        MapRenderRequest::new(
            field.into(),
            ColorScale::Discrete(DiscreteColorScale {
                levels: vec![0.0, 1.0, 2.0, 3.0],
                colors: vec![
                    rustwx_render::Color::rgba(0, 0, 255, 255),
                    rustwx_render::Color::rgba(255, 0, 0, 255),
                ],
                extend: ExtendMode::Neither,
                mask_below: None,
            }),
        )
    }


    // 2.9.4 rolling test (GFS 24 h, f012): the model-column IVT components ivt_column_eastward/northward drew a
    // signed transport on the 0-100 RH ladder, so every value below 0 or above 100 kg m-1 s-1 clipped to one colour.
    #[test]
    fn model_column_ivt_components_draw_on_the_signed_transport_scale() {
        for slug in ["ivt_column_eastward", "ivt_column_northward", "ivt_eastward_full_column", "ivt_northward_full_column"] {
            let recipe = rustwx_models::plot_recipe(slug).unwrap_or_else(|| panic!("recipe {slug}"));
            let selector = recipe.filled.selector.unwrap_or_else(|| panic!("selector {slug}"));
            match operational_fill_scale_for_recipe_in(recipe, selector, Hemisphere::Northern) {
                ColorScale::Discrete(scale) => {
                    assert_eq!(scale.levels.first().copied(), Some(-2000.0), "{slug}");
                    assert_eq!(scale.levels.last().copied(), Some(2000.0), "{slug}");
                }
                other => panic!("{slug}: {other:?}"),
            }
        }
    }
    #[test]
    fn clear_sky_under_the_first_cloud_level_draws_nothing() {
        let scale = cloud_cover_scale();
        assert_eq!(scale.mask_below, Some(10.0));
        assert_eq!(scale.levels.first().copied(), Some(10.0));
        assert!(matches!(scale.extend, ExtendMode::Max));
    }

    #[test]
    fn clear_sky_pixels_are_the_uncovered_basemap_and_ten_percent_draws_cloud() {
        let mut request = sample_request();
        request.width = 320;
        request.height = 240;
        request.colorbar = false;
        request.background = Color::rgba(21, 37, 58, 255);
        request.scale = ColorScale::Discrete(cloud_cover_scale());
        request.field.values.fill(f32::NAN);
        let uncovered = rustwx_render::render_image(&request).unwrap();
        let evidence = std::env::var_os("RUSTWX_CLOUD_MASK_EVIDENCE").map(std::path::PathBuf::from);
        if let Some(folder) = &evidence {
            std::fs::create_dir_all(folder).unwrap();
            uncovered.save(folder.join("cloud-uncovered.png")).unwrap();
        }
        let mut clear_images = Vec::new();
        for value in [0.0, 5.0, 9.999, 10.0, 95.0] {
            request.field.values.fill(value);
            let image = rustwx_render::render_image(&request).unwrap();
            if let Some(folder) = &evidence {
                image.save(folder.join(format!("cloud-{value}-percent.png"))).unwrap();
            }
            if value < 10.0 {
                clear_images.push((value, image));
            }
        }
        for (value, clear) in clear_images {
            assert_eq!(clear, uncovered, "{value}% cloud painted a basemap pixel");
        }
        request.field.values.fill(10.0);
        let cloudy = rustwx_render::render_image(&request).unwrap();
        let mut old_scale = cloud_cover_scale();
        old_scale.mask_below = None;
        old_scale.extend = ExtendMode::Both;
        request.scale = ColorScale::Discrete(old_scale);
        let first_level_control = rustwx_render::render_image(&request).unwrap();
        assert_eq!(cloudy, first_level_control,
                   "the clear-sky mask changed a pixel at the first drawn cloud level");
        let painted = cloudy.pixels().zip(uncovered.pixels())
            .filter(|(cloud, base)| cloud != base).count();
        assert!(painted > 100, "the first drawn cloud level painted {painted} pixels");
    }

    #[test]
    fn regional_static_design_uses_projected_grid_frame_and_smooth_legend() {
        let mut request = sample_request();

        StaticPlotDesign::new(
            (-125.0, -66.0, 24.0, 50.0),
            ProductVisualMode::FilledMeteorology,
        )
        .apply_to_request(&mut request);

        assert_eq!(request.visual_mode, ProductVisualMode::FilledMeteorology);
        assert_eq!(
            request.domain_frame.map(|frame| frame.source),
            Some(DomainFrameSource::ProjectedGrid)
        );
        assert_eq!(request.legend.mode, LegendMode::SmoothRamp);
        assert_eq!(request.render_density.fill, high_detail_fill_density());
        assert_eq!(request.render_density.palette_multiplier, 4);
    }

    /// A western domain gets the same grid frame as every other domain:
    /// the viewport frame (and the whitespace crop it brought) is only the
    /// explicit opt-in now, never a default chosen by location.
    #[test]
    fn a_western_domain_uses_the_grid_frame_like_any_other() {
        for bounds in [(-124.9, -113.8, 31.9, 42.5), (-125.7, -110.5, 30.5, 49.0)] {
            assert!(is_straight_western_domain_frame_candidate(bounds));
            let mut request = sample_request();
            StaticPlotDesign::new(bounds, ProductVisualMode::FilledMeteorology)
                .apply_to_request(&mut request);
            assert_eq!(
                request.domain_frame.map(|frame| frame.source),
                Some(DomainFrameSource::ProjectedGrid),
                "{bounds:?}"
            );
        }
    }

    #[test]
    fn rockies_static_design_keeps_projected_grid_frame() {
        let mut request = sample_request();

        StaticPlotDesign::new(
            (-112.0, -96.0, 37.0, 49.5),
            ProductVisualMode::FilledMeteorology,
        )
        .apply_to_request(&mut request);

        assert_eq!(
            request.domain_frame.map(|frame| frame.source),
            Some(DomainFrameSource::ProjectedGrid)
        );
    }

    #[test]
    fn global_filled_static_design_uses_smooth_legend_without_viewport_frame() {
        let mut request = sample_request();

        apply_static_map_design(
            &mut request,
            (-180.0, 179.999, -90.0, 90.0),
            ProductVisualMode::FilledMeteorology,
            false,
        );

        assert!(request.domain_frame.is_none());
        assert_eq!(request.legend.mode, LegendMode::SmoothRamp);
        assert_eq!(request.render_density, RenderDensity::default());
    }

    #[test]
    fn global_overlay_static_design_keeps_stepped_legend_policy() {
        let mut request = sample_request();

        apply_static_map_design(
            &mut request,
            (-180.0, 179.999, -90.0, 90.0),
            ProductVisualMode::OverlayAnalysis,
            true,
        );

        assert!(request.domain_frame.is_none());
        assert_eq!(request.legend.mode, LegendMode::SmoothRamp);
        assert_eq!(request.render_density.fill, high_detail_fill_density());
        assert_eq!(request.render_density.palette_multiplier, 4);
    }

    #[test]
    fn operational_pressure_contours_convert_units_and_mark_extrema() {
        let layer = operational_contour_layer_for_values(
            FieldSelector::mean_sea_level(CanonicalField::PressureReducedToMeanSeaLevel),
            &[100000.0, 100200.0, 100400.0, 100600.0],
        )
        .expect("pressure contour layer");

        assert_eq!(layer.data[0], 1000.0);
        assert_eq!(layer.levels.first().copied(), Some(960.0));
        assert_eq!(layer.width, 1);
        assert_eq!(layer.major_every, Some(2));
        assert_eq!(layer.major_width, Some(2));
        assert_eq!(layer.pattern, ContourLinePattern::Solid);
        assert!(layer.labels);
        assert!(layer.show_extrema);
    }

    #[test]
    fn operational_height_contours_convert_to_decameters_without_extrema() {
        let layer = operational_contour_layer_for_values(
            FieldSelector::isobaric(CanonicalField::GeopotentialHeight, 500),
            &[5400.0, 5460.0, 5520.0, 5580.0],
        )
        .expect("height contour layer");

        assert_eq!(layer.data[0], 540.0);
        assert_eq!(layer.levels.first().copied(), Some(450.0));
        assert_eq!(layer.levels.get(1).copied(), Some(456.0));
        assert_eq!(layer.color, Color::rgba(0, 0, 0, 220));
        assert_eq!(layer.major_every, Some(2));
        assert_eq!(layer.major_width, Some(2));
        assert!(layer.labels);
        assert!(!layer.show_extrema);
    }

    #[test]
    fn operational_wind_streamlines_are_subtle_dense_flow_texture() {
        let style = operational_wind_streamline_style(9, 7);

        assert_eq!(style.stride_x, 9);
        assert_eq!(style.stride_y, 7);
        assert_eq!(style.width, 1);
        assert!(style.color.a < 160);
        assert!(style.max_steps >= 12);
        assert!(style.step_cells > 0.0);
    }

    #[test]
    fn operational_fill_scale_masks_sparse_signal_products() {
        let reflectivity = rustwx_models::plot_recipe("composite_reflectivity").unwrap();
        let ColorScale::Discrete(reflectivity_scale) = operational_fill_scale_for_recipe(
            reflectivity,
            FieldSelector::surface(CanonicalField::CompositeReflectivity),
        ) else {
            panic!("expected reflectivity discrete scale");
        };
        assert_eq!(reflectivity_scale.levels.first().copied(), Some(10.0));
        // Re-recorded 2026-10-03: the reflectivity products moved to the
        // radar reflectivity table, which runs to 85 dBZ (was 70).
        assert_eq!(reflectivity_scale.levels.last().copied(), Some(85.0));
        assert_eq!(reflectivity_scale.extend, ExtendMode::Max);
        assert_eq!(reflectivity_scale.mask_below, Some(10.0));

        let mslp_winds = rustwx_models::plot_recipe("mslp_10m_winds").unwrap();
        let ColorScale::Discrete(mslp_wind_scale) = operational_fill_scale_for_recipe(
            mslp_winds,
            FieldSelector::mean_sea_level(CanonicalField::PressureReducedToMeanSeaLevel),
        ) else {
            panic!("expected MSLP/10m wind discrete scale");
        };
        assert_eq!(mslp_wind_scale.levels.first().copied(), Some(10.0));
        assert_eq!(mslp_wind_scale.mask_below, Some(10.0));

        let qpf = rustwx_models::plot_recipe("1h_qpf").unwrap();
        let ColorScale::Discrete(qpf_scale) = operational_fill_scale_for_recipe(
            qpf,
            FieldSelector::surface(CanonicalField::TotalPrecipitation),
        ) else {
            panic!("expected QPF discrete scale");
        };
        assert_eq!(qpf_scale.mask_below, Some(0.01));

        let categorical = rustwx_models::plot_recipe("categorical_snow").unwrap();
        let ColorScale::Discrete(categorical_scale) = operational_fill_scale_for_recipe(
            categorical,
            FieldSelector::surface(CanonicalField::CategoricalSnow),
        ) else {
            panic!("expected categorical discrete scale");
        };
        assert_eq!(categorical_scale.extend, ExtendMode::Neither);
        assert_eq!(categorical_scale.mask_below, Some(0.5));

        let surface_smoke = rustwx_models::plot_recipe("smoke_pm25_native").unwrap();
        let ColorScale::Discrete(surface_smoke_scale) = operational_fill_scale_for_recipe(
            surface_smoke,
            FieldSelector::height_agl(CanonicalField::SmokeMassDensity, 8),
        ) else {
            panic!("expected surface smoke discrete scale");
        };
        assert_eq!(surface_smoke_scale.levels.first().copied(), Some(10.0));
        assert_eq!(surface_smoke_scale.mask_below, Some(10.0));
        assert!(surface_smoke_scale.colors[0].a < 80);

        let column_smoke = rustwx_models::plot_recipe("smoke_column").unwrap();
        let ColorScale::Discrete(column_smoke_scale) = operational_fill_scale_for_recipe(
            column_smoke,
            FieldSelector::entire_atmosphere(CanonicalField::ColumnIntegratedSmoke),
        ) else {
            panic!("expected column smoke discrete scale");
        };
        assert_eq!(column_smoke_scale.levels.first().copied(), Some(20.0));
        assert_eq!(column_smoke_scale.mask_below, Some(20.0));
        assert!(column_smoke_scale.colors[0].a < 80);
    }

    #[test]
    fn the_southern_vorticity_scale_is_the_northern_one_mirrored() {
        // B-07.  Absolute vorticity is dominated by the planetary term,
        // which changes sign at the equator, so cyclonic rotation is
        // POSITIVE in the north and NEGATIVE in the south.  The single
        // fixed -40..60 ramp gave its structured colours (orange, red,
        // purple, cyan) to the positive half, so a South American domain
        // rendered as a near-uniform grey sheet: the whole synoptic
        // cyclonic signal sat in the flat end of the ramp.
        let north = vorticity_scale_for_hemisphere(Hemisphere::Northern);
        let south = vorticity_scale_for_hemisphere(Hemisphere::Southern);

        // The northern scale is the one that shipped, unchanged.
        assert_eq!(north.levels.first().copied(), Some(-40.0));
        assert_eq!(north.levels.last().copied(), Some(60.0));

        // The southern range is the northern one reflected about zero.
        assert_eq!(south.levels.first().copied(), Some(-60.0));
        assert_eq!(south.levels.last().copied(), Some(40.0));
        assert_eq!(south.levels.len(), north.levels.len());

        // ... and so is the palette, so cyclonic rotation gets the same
        // colours in the same order a northern forecaster reads.
        assert_eq!(south.colors.len(), north.colors.len());
        let mut reversed = north.colors.clone();
        reversed.reverse();
        assert_eq!(south.colors, reversed);
        assert_ne!(south.colors, north.colors, "the palette must actually move");
    }

    #[test]
    fn the_hemisphere_comes_from_the_middle_of_the_domain() {
        // (west, east, south, north)
        assert_eq!(Hemisphere::for_bounds((-80.0, -60.0, -40.0, -20.0)),
                   Hemisphere::Southern);
        assert_eq!(Hemisphere::for_bounds((-100.0, -80.0, 25.0, 50.0)),
                   Hemisphere::Northern);
        // Straddling the equator takes the side it mostly occupies,
        // rather than flipping on a single row of cells.
        assert_eq!(Hemisphere::for_bounds((-10.0, 10.0, -30.0, 5.0)),
                   Hemisphere::Southern);
        assert_eq!(Hemisphere::for_bounds((-10.0, 10.0, -5.0, 30.0)),
                   Hemisphere::Northern);
        // Exactly symmetric about the equator is northern by convention,
        // and pinned so the tie does not drift silently.
        assert_eq!(Hemisphere::for_bounds((-10.0, 10.0, -20.0, 20.0)),
                   Hemisphere::Northern);
    }

    #[test]
    fn only_vorticity_is_hemisphere_dependent() {
        // The negative control for the threading: every other product
        // must be identical in both hemispheres, so this fails if a
        // future scale starts keying on latitude without a test.
        let cases: [(&str, FieldSelector); 3] = [
            (
                "composite_reflectivity",
                FieldSelector::surface(CanonicalField::CompositeReflectivity),
            ),
            (
                "2m_temperature",
                FieldSelector::height_agl(CanonicalField::Temperature, 2),
            ),
            (
                "mslp_10m_winds",
                FieldSelector::height_agl(CanonicalField::WindSpeed, 10),
            ),
        ];
        for (slug, selector) in cases {
            let recipe = rustwx_models::plot_recipe(slug)
                .unwrap_or_else(|| panic!("recipe {slug} should exist"));
            let north = operational_fill_scale_for_recipe_in(
                recipe, selector, Hemisphere::Northern);
            let south = operational_fill_scale_for_recipe_in(
                recipe, selector, Hemisphere::Southern);
            assert_eq!(format!("{north:?}"), format!("{south:?}"), "{slug}");
        }
    }

    /// Every value a column-plane map draws is drawn in the colour its bar
    /// shows for that value, the over-range colour at the bar's top end
    /// included.  The map lanes fill a scale densely and read it against a
    /// smooth ramp through the colours sampled at each declared level, so a
    /// sparse level set whose last band spans a quarter of the bar sampled
    /// that band at its lower edge: the supercooled water path drew its
    /// heaviest columns purple beside a bar whose top was navy, and the
    /// hydrometeor maps drew 4 g kg-1 darker than any colour on theirs.
    #[test]
    fn column_plane_maps_are_drawn_in_the_colours_their_bar_shows() {
        let mut request = sample_request();
        StaticPlotDesign::new(
            (-10.0, 10.0, 30.0, 45.0),
            ProductVisualMode::FilledMeteorology,
        )
        .apply_to_request(&mut request);
        let options = rustwx_render::ColormapBuildOptions {
            render_density: request.render_density,
            legend: request.legend,
        };
        let slugs = [
            "isotherm_height_0c",
            "isotherm_height_minus10c",
            "isotherm_height_minus20c",
            "supercooled_water_path",
            "supercooled_water_path_0_3km",
            "supercooled_water_path_3_6km",
            "cloud_water_column_max",
            "rain_water_column_max",
            "cloud_ice_column_max",
            "snow_column_max",
            "graupel_column_max",
        ];
        for slug in slugs {
            let recipe = rustwx_models::plot_recipe(slug)
                .unwrap_or_else(|| panic!("recipe {slug} should exist"));
            let selector = recipe
                .filled
                .selector
                .unwrap_or_else(|| panic!("{slug} names its stored selector"));
            let scale = operational_fill_scale_for_recipe(recipe, selector);
            let cmap = rustwx_render::build_colormap(&scale, options);
            let levels = cmap.legend_levels_for_display();
            let (low, high) = (levels[0], levels[levels.len() - 1]);
            let mut worst = (0u32, low);
            for step in 0..=2200 {
                let value = low + (high - low) * step as f64 / 2000.0;
                let fill = cmap.map(value);
                if fill.a == 0 {
                    continue;
                }
                let rel = rustwx_render::legend_tick_rel(&cmap, value).unwrap();
                let bar = rustwx_render::legend_color_at_rel(&cmap, request.legend.mode, rel);
                let distance = u32::from(fill.r.abs_diff(bar.r))
                    + u32::from(fill.g.abs_diff(bar.g))
                    + u32::from(fill.b.abs_diff(bar.b));
                if distance > worst.0 {
                    worst = (distance, value);
                }
            }
            assert!(
                worst.0 <= 60,
                "{slug}: the map draws {} in a colour {} RGB steps from the bar's colour for it",
                worst.1,
                worst.0
            );
        }
    }
}
