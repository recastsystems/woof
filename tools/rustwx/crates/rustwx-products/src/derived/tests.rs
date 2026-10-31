use super::compute::{PressureFieldSet, haversine_m, pressure_level_slice_or_interp};
use super::*;
use crate::shared_context::{DomainSpec, TitleProvenance};
use rustwx_render::PngCompressionMode;

struct TestPressureFields {
    pressure_levels_hpa: Vec<f64>,
    pressure_3d_pa: Option<Vec<f64>>,
    temperature_c_3d: Vec<f64>,
}

impl PressureFieldSet for TestPressureFields {
    fn pressure_levels_hpa(&self) -> &[f64] {
        &self.pressure_levels_hpa
    }

    fn pressure_3d_pa(&self) -> Option<&[f64]> {
        self.pressure_3d_pa.as_deref()
    }

    fn temperature_c_3d(&self) -> &[f64] {
        &self.temperature_c_3d
    }

    fn qvapor_kgkg_3d(&self) -> &[f64] {
        &[]
    }

    fn u_ms_3d(&self) -> &[f64] {
        &[]
    }

    fn v_ms_3d(&self) -> &[f64] {
        &[]
    }

    fn gh_m_3d(&self) -> &[f64] {
        &[]
    }
}

fn sample_native_contour_grid() -> rustwx_core::LatLonGrid {
    rustwx_core::LatLonGrid::new(
        rustwx_core::GridShape::new(3, 3).unwrap(),
        vec![35.0, 35.0, 35.0, 36.0, 36.0, 36.0, 37.0, 37.0, 37.0],
        vec![
            -99.0, -98.0, -97.0, -99.0, -98.0, -97.0, -99.0, -98.0, -97.0,
        ],
    )
    .unwrap()
}

fn sample_projected_map() -> ProjectedMap {
    ProjectedMap {
        projected_x: vec![-1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0],
        projected_y: vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
        extent: ProjectedExtent {
            x_min: -1.0,
            x_max: 1.0,
            y_min: 0.0,
            y_max: 2.0,
        },
        lines: Vec::new(),
        polygons: Vec::new(),
        inverse_raster_projection: None,
    }
}

fn sample_domain_bounds() -> (f64, f64, f64, f64) {
    (-99.0, -97.0, 35.0, 37.0)
}

fn sample_fire_weather_computed_fields() -> DerivedComputedFields {
    DerivedComputedFields {
        vpd_2m_hpa: Some(vec![0.5, 1.5, 3.0, 2.0, 4.0, 6.0, 5.0, 8.0, 10.0]),
        dewpoint_depression_2m_c: Some(vec![1.0, 3.0, 6.0, 4.0, 8.0, 12.0, 10.0, 16.0, 20.0]),
        wetbulb_2m_c: Some(vec![-6.0, -3.0, 0.0, 2.0, 5.0, 8.0, 11.0, 15.0, 19.0]),
        fire_weather_composite: Some(vec![8.0, 15.0, 25.0, 20.0, 35.0, 55.0, 50.0, 75.0, 92.0]),
        ..DerivedComputedFields::default()
    }
}

#[test]
fn canonical_depth_ehi_slugs_are_supported_and_legacy_aliases_canonicalize() {
    assert_eq!(
        DerivedRecipe::parse("scp_mu_0_3km_0_6km_proxy").unwrap(),
        DerivedRecipe::ScpMu03km06kmProxy
    );
    assert_eq!(
        DerivedRecipe::parse("apparent_temperature_2m").unwrap(),
        DerivedRecipe::ApparentTemperature2m
    );
    assert_eq!(
        DerivedRecipe::parse("2m_apparent_temperature").unwrap(),
        DerivedRecipe::ApparentTemperature2m
    );
    assert_eq!(
        DerivedRecipe::parse("2m_vpd").unwrap(),
        DerivedRecipe::Vpd2m
    );
    assert_eq!(
        DerivedRecipe::parse("vapor_pressure_deficit_2m").unwrap(),
        DerivedRecipe::Vpd2m
    );
    assert_eq!(
        DerivedRecipe::parse("2m_dewpoint_depression").unwrap(),
        DerivedRecipe::DewpointDepression2m
    );
    assert_eq!(
        DerivedRecipe::parse("wet_bulb_2m").unwrap(),
        DerivedRecipe::Wetbulb2m
    );
    assert_eq!(
        DerivedRecipe::parse("fire_weather").unwrap(),
        DerivedRecipe::FireWeatherComposite
    );
    assert_eq!(
        DerivedRecipe::parse("ehi_0_1km").unwrap(),
        DerivedRecipe::Ehi01km
    );
    assert_eq!(
        DerivedRecipe::parse("ehi_sb_0_1km_proxy").unwrap(),
        DerivedRecipe::Ehi01km
    );
    assert_eq!(
        DerivedRecipe::parse("ehi_0_3km").unwrap(),
        DerivedRecipe::Ehi03km
    );
    assert_eq!(
        DerivedRecipe::parse("ehi_sb_0_3km_proxy").unwrap(),
        DerivedRecipe::Ehi03km
    );
    assert!(DerivedRecipe::parse("scp").is_err());
    assert!(DerivedRecipe::parse("stp_effective").is_err());
}

#[test]
fn haversine_is_reasonable_for_one_degree_latitude() {
    let distance = haversine_m(35.0, -97.0, 36.0, -97.0);
    assert!(distance > 100_000.0);
    assert!(distance < 120_000.0);
}

#[test]
fn derived_recipe_dedupe_preserves_first_seen_order() {
    let recipes = plan_derived_recipes(&[
        "mlcape".to_string(),
        "sbcape".to_string(),
        "mlcape".to_string(),
    ])
    .unwrap();
    assert_eq!(recipes, vec![DerivedRecipe::Mlcape, DerivedRecipe::Sbcape]);
}

#[test]
fn derived_inventory_stays_in_sync_with_slug_parser() {
    for recipe in supported_derived_recipe_inventory() {
        assert!(
            DerivedRecipe::parse(recipe.slug).is_ok(),
            "supported inventory slug '{}' should parse",
            recipe.slug
        );
    }
    for recipe in blocked_derived_recipe_inventory() {
        assert!(
            DerivedRecipe::parse(recipe.slug).is_err(),
            "blocked inventory slug '{}' should stay blocked",
            recipe.slug
        );
    }
}

#[test]
fn blocked_inventory_is_narrowed_to_effective_layer_products() {
    let blocked = blocked_derived_recipe_inventory()
        .iter()
        .map(|recipe| recipe.slug)
        .collect::<Vec<_>>();
    assert_eq!(blocked, vec!["stp_effective", "scp", "scp_effective"]);
}

#[test]
fn derived_requirements_stay_narrow_for_surface_only_requests() {
    let requirements = DerivedRequirements::from_recipes(&[DerivedRecipe::HeatIndex2m]);
    assert!(requirements.surface_thermo);
    assert!(!requirements.needs_volume());
    assert!(!requirements.needs_height_agl());
    assert!(!requirements.needs_grid_spacing());
}

#[test]
fn apparent_temperature_is_supported_surface_only_inventory_entry() {
    let recipe = supported_derived_recipe_inventory()
        .iter()
        .find(|recipe| recipe.slug == "apparent_temperature_2m")
        .expect("apparent temperature inventory entry should exist");
    assert_eq!(recipe.title, "2 m Apparent Temperature");
    assert!(!recipe.experimental);

    let requirements = DerivedRequirements::from_recipes(&[DerivedRecipe::ApparentTemperature2m]);
    assert!(requirements.surface_thermo);
    assert!(!requirements.surface_winds);
    assert!(!requirements.needs_volume());
    assert!(!requirements.needs_height_agl());
    assert!(!requirements.needs_grid_spacing());
}

#[test]
fn fire_weather_family_is_supported_surface_only_inventory() {
    let expected = [
        ("vpd_2m", "2 m Vapor Pressure Deficit", DerivedRecipe::Vpd2m),
        (
            "dewpoint_depression_2m",
            "2 m Dewpoint Depression",
            DerivedRecipe::DewpointDepression2m,
        ),
        (
            "wetbulb_2m",
            "2 m Wet-Bulb Temperature",
            DerivedRecipe::Wetbulb2m,
        ),
        (
            "fire_weather_composite",
            "Fire Weather Composite",
            DerivedRecipe::FireWeatherComposite,
        ),
    ];

    for (slug, title, parsed) in expected {
        let recipe = supported_derived_recipe_inventory()
            .iter()
            .find(|recipe| recipe.slug == slug)
            .unwrap_or_else(|| panic!("{slug} inventory entry should exist"));
        assert_eq!(recipe.title, title);
        assert!(!recipe.experimental);
        assert!(!recipe.heavy);
        assert_eq!(DerivedRecipe::parse(slug).unwrap(), parsed);
    }

    let requirements = DerivedRequirements::from_recipes(&[
        DerivedRecipe::Vpd2m,
        DerivedRecipe::DewpointDepression2m,
        DerivedRecipe::Wetbulb2m,
        DerivedRecipe::FireWeatherComposite,
    ]);
    assert!(requirements.surface_thermo);
    assert!(!requirements.surface_winds);
    assert!(!requirements.needs_volume());
    assert!(!requirements.needs_height_agl());
    assert!(!requirements.needs_grid_spacing());
}

#[test]
fn rrfs_public_and_firewx_expose_model_agnostic_derived_inventory() {
    for model in [ModelId::RrfsPublic, ModelId::RrfsFireWx] {
        let slugs = supported_derived_recipe_slugs(model);
        assert!(
            slugs.iter().any(|slug| slug == "vpd_2m"),
            "{model} should expose fire-weather surface diagnostics"
        );
        assert!(
            slugs.iter().any(|slug| slug == "bulk_shear_0_6km"),
            "{model} should expose profile-derived severe diagnostics"
        );
        assert!(
            slugs.iter().any(|slug| slug == "stp_fixed"),
            "{model} should expose model-agnostic severe composite diagnostics"
        );
    }
}

#[test]
fn native_contour_config_covers_multiple_real_products() {
    for recipe in [
        DerivedRecipe::StpFixed,
        DerivedRecipe::Sbcape,
        DerivedRecipe::Mlcape,
        DerivedRecipe::Srh01km,
        DerivedRecipe::Srh03km,
        DerivedRecipe::Ehi01km,
        DerivedRecipe::Ehi03km,
    ] {
        let config = native_contour_product_config(recipe)
            .unwrap_or_else(|| panic!("expected native contour config for {}", recipe.slug()));
        assert!(
            !config.line_levels.is_empty(),
            "{} should define contour lines",
            recipe.slug()
        );
    }
    assert!(native_contour_product_config(DerivedRecipe::LiftedIndex).is_none());
}

#[test]
fn native_projected_contour_scale_uses_operational_tick_density() {
    let config = native_contour_product_config(DerivedRecipe::Sbcape)
        .expect("SBCAPE should have native contour config");
    let ColorScale::Discrete(coarse) =
        native_projected_contour_scale(config.scale, config.tick_step, 1)
    else {
        panic!("expected discrete contour scale");
    };

    assert!(coarse.levels.contains(&250.0));
    assert!(coarse.levels.contains(&500.0));
    assert!(!coarse.levels.contains(&100.0));
    assert!(coarse.levels.len() < 25);

    let config = native_contour_product_config(DerivedRecipe::Sbcape)
        .expect("SBCAPE should have native contour config");
    let ColorScale::Discrete(finer) =
        native_projected_contour_scale(config.scale, config.tick_step, 5)
    else {
        panic!("expected discrete contour scale");
    };

    assert!(finer.levels.len() > coarse.levels.len());
    assert!(finer.levels.contains(&100.0));
}

#[test]
fn wetbulb_uses_raster_temperature_scale_without_contour_promotion() {
    assert!(native_contour_product_config(DerivedRecipe::Wetbulb2m).is_none());

    let grid = sample_native_contour_grid();
    let projected = sample_projected_map();
    let computed = sample_fire_weather_computed_fields();
    let artifact = build_render_artifact_with_contour_mode(
        DerivedRecipe::Wetbulb2m,
        &grid,
        &projected,
        sample_domain_bounds(),
        "20260414",
        23,
        0,
        SourceId::Nomads,
        ModelId::Hrrr,
        1200,
        900,
        &computed,
        NativeContourRenderMode::Automatic,
        1,
    )
    .unwrap();

    assert!(artifact.request.projected_data_polygons.is_empty());
    assert!(
        artifact
            .request
            .field
            .values
            .iter()
            .any(|value| value.is_finite())
    );
    let ColorScale::Discrete(scale) = artifact.request.scale else {
        panic!("wet-bulb scale should be discrete");
    };
    assert_eq!(scale.extend, ExtendMode::Both);
    assert_eq!(scale.levels[0], -50.0);
    assert_eq!(scale.levels[1] - scale.levels[0], 0.5);
    assert!(scale.levels.contains(&0.0));
    assert!(scale.levels.contains(&40.0));
    assert_ne!(scale.levels[1] - scale.levels[0], 5.0);
}

#[test]
fn automatic_contour_mode_keeps_configured_native_products_rasterized() {
    let grid = sample_native_contour_grid();
    let projected = sample_projected_map();
    let values = vec![
        0.0, 500.0, 1000.0, 250.0, 1250.0, 2250.0, 750.0, 2000.0, 3500.0,
    ];

    let automatic = build_native_render_artifact(
        DerivedRecipe::Sbcape,
        &grid,
        &projected,
        sample_domain_bounds(),
        "20260414",
        23,
        0,
        SourceId::Nomads,
        ModelId::Hrrr,
        1200,
        900,
        values.clone(),
        NativeContourRenderMode::Automatic,
        1,
    )
    .unwrap();
    assert!(automatic.request.projected_data_polygons.is_empty());
    assert!(
        automatic
            .request
            .field
            .values
            .iter()
            .any(|value| value.is_finite())
    );

    let legacy = build_native_render_artifact(
        DerivedRecipe::Sbcape,
        &grid,
        &projected,
        sample_domain_bounds(),
        "20260414",
        23,
        0,
        SourceId::Nomads,
        ModelId::Hrrr,
        1200,
        900,
        values,
        NativeContourRenderMode::LegacyRaster,
        1,
    )
    .unwrap();
    assert!(legacy.request.projected_data_polygons.is_empty());
    assert!(
        legacy
            .request
            .field
            .values
            .iter()
            .any(|value| value.is_finite())
    );
}

#[test]
fn aifs_inference_derived_policy_uses_interpolated_raster_sampling() {
    let grid = sample_native_contour_grid();
    let projected = sample_projected_map();
    let values = vec![
        0.0, 500.0, 1000.0, 250.0, 1250.0, 2250.0, 750.0, 2000.0, 3500.0,
    ];

    let artifact = build_native_render_artifact(
        DerivedRecipe::Sbcape,
        &grid,
        &projected,
        sample_domain_bounds(),
        "20260414",
        23,
        0,
        SourceId::AifsInference,
        ModelId::Aifs,
        1200,
        900,
        values,
        NativeContourRenderMode::Automatic,
        1,
    )
    .unwrap();

    assert_eq!(
        artifact.request.raster_sample_mode,
        RasterSampleMode::Linear
    );
    assert!(artifact.request.projected_data_polygons.is_empty());
}

#[test]
fn experimental_contour_mode_can_promote_nonconfigured_derived_products() {
    let grid = sample_native_contour_grid();
    let projected = sample_projected_map();
    let values = vec![-9.0, -6.0, -3.0, -2.0, 0.0, 2.0, 4.0, 7.0, 10.0];

    let automatic = build_native_render_artifact(
        DerivedRecipe::LiftedIndex,
        &grid,
        &projected,
        sample_domain_bounds(),
        "20260414",
        23,
        0,
        SourceId::Nomads,
        ModelId::Hrrr,
        1200,
        900,
        values.clone(),
        NativeContourRenderMode::Automatic,
        1,
    )
    .unwrap();
    assert!(automatic.request.projected_data_polygons.is_empty());

    let experimental = build_native_render_artifact(
        DerivedRecipe::LiftedIndex,
        &grid,
        &projected,
        sample_domain_bounds(),
        "20260414",
        23,
        0,
        SourceId::Nomads,
        ModelId::Hrrr,
        1200,
        900,
        values,
        NativeContourRenderMode::ExperimentalAllProjected,
        1,
    )
    .unwrap();
    assert!(!experimental.request.projected_data_polygons.is_empty());
    assert!(
        experimental
            .request
            .field
            .values
            .iter()
            .all(|value| value.is_nan())
    );
}

#[test]
fn signature_contour_mode_keeps_selected_products_rasterized() {
    let grid = sample_native_contour_grid();
    let projected = sample_projected_map();
    let values = vec![-9.0, -6.0, -3.0, -2.0, 0.0, 2.0, 4.0, 7.0, 10.0];

    let signature = build_native_render_artifact(
        DerivedRecipe::LiftedIndex,
        &grid,
        &projected,
        sample_domain_bounds(),
        "20260414",
        23,
        0,
        SourceId::Nomads,
        ModelId::Hrrr,
        1200,
        900,
        values,
        NativeContourRenderMode::Signature,
        1,
    )
    .unwrap();
    assert!(signature.request.projected_data_polygons.is_empty());
    assert!(
        signature
            .request
            .field
            .values
            .iter()
            .any(|value| value.is_finite())
    );
}

#[test]
fn signature_contour_mode_keeps_non_signature_products_rasterized() {
    let grid = sample_native_contour_grid();
    let projected = sample_projected_map();
    let values = vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 1.0, 0.0, -1.0];

    let signature = build_native_render_artifact(
        DerivedRecipe::Mucin,
        &grid,
        &projected,
        sample_domain_bounds(),
        "20260414",
        23,
        0,
        SourceId::Nomads,
        ModelId::Hrrr,
        1200,
        900,
        values,
        NativeContourRenderMode::Signature,
        1,
    )
    .unwrap();
    assert!(signature.request.projected_data_polygons.is_empty());
    assert!(
        signature
            .request
            .field
            .values
            .iter()
            .any(|value| value.is_finite())
    );
}

#[test]
fn fire_weather_family_render_artifacts_build_and_stay_rasterized() {
    let grid = sample_native_contour_grid();
    let projected = sample_projected_map();
    let computed = sample_fire_weather_computed_fields();

    for recipe in [
        DerivedRecipe::Vpd2m,
        DerivedRecipe::DewpointDepression2m,
        DerivedRecipe::Wetbulb2m,
        DerivedRecipe::FireWeatherComposite,
    ] {
        let artifact = build_render_artifact_with_contour_mode(
            recipe,
            &grid,
            &projected,
            sample_domain_bounds(),
            "20260414",
            23,
            0,
            SourceId::Nomads,
            ModelId::Hrrr,
            1200,
            900,
            &computed,
            NativeContourRenderMode::LegacyRaster,
            1,
        )
        .unwrap();
        assert_eq!(artifact.request.title.as_deref(), Some(recipe.title()));
        assert!(artifact.request.projected_data_polygons.is_empty());
        assert!(artifact.field.values.iter().any(|value| value.is_finite()));
    }

    let signature = build_render_artifact_with_contour_mode(
        DerivedRecipe::FireWeatherComposite,
        &grid,
        &projected,
        sample_domain_bounds(),
        "20260414",
        23,
        0,
        SourceId::Nomads,
        ModelId::Hrrr,
        1200,
        900,
        &computed,
        NativeContourRenderMode::Signature,
        1,
    )
    .unwrap();
    assert!(signature.request.projected_data_polygons.is_empty());
    assert!(
        signature
            .request
            .field
            .values
            .iter()
            .any(|value| value.is_finite())
    );
}

#[test]
fn ecape_inventory_entries_are_marked_heavy() {
    let sbecape = supported_derived_recipe_inventory()
        .iter()
        .find(|recipe| recipe.slug == "sbecape")
        .expect("sbecape inventory entry should exist");
    assert!(sbecape.heavy);
    assert!(!sbecape.experimental);

    let ecape_scp = supported_derived_recipe_inventory()
        .iter()
        .find(|recipe| recipe.slug == "ecape_scp")
        .expect("ecape_scp inventory entry should exist");
    assert!(ecape_scp.heavy);
    assert!(ecape_scp.experimental);
    assert_eq!(
        DerivedRecipe::parse("ecape_scp").unwrap(),
        DerivedRecipe::EcapeScp
    );

    let native_ratio = supported_derived_recipe_inventory()
        .iter()
        .find(|recipe| recipe.slug == "sb_ecape_native_cape_ratio")
        .expect("native ECAPE/CAPE ratio inventory entry should exist");
    assert!(native_ratio.heavy);
    assert!(native_ratio.experimental);
    assert_eq!(
        DerivedRecipe::parse("sb_ecape_native_cape_ratio").unwrap(),
        DerivedRecipe::SbEcapeNativeCapeRatio
    );
}

#[test]
fn canonical_mode_keeps_all_supported_recipes_on_canonical_path() {
    let recipes = vec![
        DerivedRecipe::Sbcape,
        DerivedRecipe::LiftedIndex,
        DerivedRecipe::BulkShear06km,
    ];
    let planned = plan_native_thermo_routes_with_surface_product(
        ModelId::Hrrr,
        &recipes,
        ProductSourceMode::Canonical,
        None,
    )
    .unwrap();
    assert_eq!(planned.output_recipes, recipes);
    assert_eq!(planned.compute_recipes, recipes);
    assert!(planned.native_routes.is_empty());
    assert!(planned.blockers.is_empty());
}

#[test]
fn gfs_canonical_mode_uses_exact_native_thermo_routes() {
    let recipes = vec![DerivedRecipe::Sbcape, DerivedRecipe::Mlcape];
    let planned = plan_native_thermo_routes_with_surface_product(
        ModelId::Gfs,
        &recipes,
        ProductSourceMode::Canonical,
        None,
    )
    .unwrap();

    assert_eq!(planned.output_recipes, recipes);
    assert_eq!(planned.compute_recipes, vec![DerivedRecipe::Mlcape]);
    assert_eq!(planned.native_routes.len(), 1);
    assert_eq!(planned.native_routes[0].recipe, DerivedRecipe::Sbcape);
    assert_eq!(
        planned.native_routes[0].source_route,
        ProductSourceRoute::NativeExact
    );
}

#[test]
fn canonical_mode_routes_ecape_recipes_through_heavy_path() {
    let planned = plan_native_thermo_routes_with_surface_product(
        ModelId::Hrrr,
        &[DerivedRecipe::Sbecape, DerivedRecipe::EcapeScp],
        ProductSourceMode::Canonical,
        None,
    )
    .unwrap();
    assert_eq!(
        planned.output_recipes,
        vec![DerivedRecipe::Sbecape, DerivedRecipe::EcapeScp]
    );
    assert!(planned.compute_recipes.is_empty());
    assert_eq!(
        planned.heavy_recipes,
        vec![DerivedRecipe::Sbecape, DerivedRecipe::EcapeScp]
    );
    assert!(planned.native_routes.is_empty());
    assert!(planned.blockers.is_empty());
}

#[test]
fn fastest_mode_uses_native_exact_and_blocks_non_fast_recipes() {
    let recipes = vec![DerivedRecipe::Sbcape, DerivedRecipe::BulkShear06km];
    let planned = plan_native_thermo_routes_with_surface_product(
        ModelId::Hrrr,
        &recipes,
        ProductSourceMode::Fastest,
        None,
    )
    .unwrap();
    assert_eq!(planned.output_recipes, vec![DerivedRecipe::Sbcape]);
    assert!(planned.compute_recipes.is_empty());
    assert_eq!(planned.native_routes.len(), 1);
    assert_eq!(planned.native_routes[0].recipe, DerivedRecipe::Sbcape);
    assert_eq!(
        planned.native_routes[0].source_route,
        ProductSourceRoute::NativeExact
    );
    assert_eq!(planned.blockers.len(), 1);
    assert_eq!(planned.blockers[0].recipe_slug, "bulk_shear_0_6km");
    assert_eq!(
        planned.blockers[0].source_route,
        ProductSourceRoute::BlockedNoFastRoute
    );
}

#[test]
fn fastest_mode_keeps_proxy_native_routes_when_labeled() {
    let planned = plan_native_thermo_routes_with_surface_product(
        ModelId::Gfs,
        &[DerivedRecipe::Mlcape],
        ProductSourceMode::Fastest,
        None,
    )
    .unwrap();
    assert_eq!(planned.output_recipes, vec![DerivedRecipe::Mlcape]);
    assert_eq!(planned.native_routes.len(), 1);
    assert_eq!(
        planned.native_routes[0].source_route,
        ProductSourceRoute::NativeProxy
    );
    assert!(planned.blockers.is_empty());
}

#[test]
fn fastest_mode_blocks_surface_only_canonical_shortcuts_until_a_true_fast_path_exists() {
    let planned = plan_native_thermo_routes_with_surface_product(
        ModelId::Hrrr,
        &[DerivedRecipe::HeatIndex2m],
        ProductSourceMode::Fastest,
        None,
    )
    .unwrap();
    assert!(planned.output_recipes.is_empty());
    assert!(planned.native_routes.is_empty());
    assert_eq!(planned.blockers.len(), 1);
    assert!(
        planned.blockers[0]
            .reason
            .contains("will not fall back to canonical-derived compute")
    );
}

#[test]
fn fastest_mode_blocks_heavy_ecape_recipes() {
    let planned = plan_native_thermo_routes_with_surface_product(
        ModelId::Hrrr,
        &[DerivedRecipe::Sbecape],
        ProductSourceMode::Fastest,
        None,
    )
    .unwrap();
    assert!(planned.output_recipes.is_empty());
    assert!(planned.compute_recipes.is_empty());
    assert!(planned.heavy_recipes.is_empty());
    assert_eq!(planned.blockers.len(), 1);
    assert!(
        planned.blockers[0]
            .reason
            .contains("cropped heavy ECAPE path")
    );
}

#[test]
fn wrf_gdex_canonical_mode_prefers_native_d612005_2d_recipes() {
    let recipes = vec![
        DerivedRecipe::Sbcape,
        DerivedRecipe::BulkShear06km,
        DerivedRecipe::Srh03km,
        DerivedRecipe::LiftedIndex,
    ];
    let planned = plan_native_thermo_routes_with_surface_product(
        ModelId::WrfGdex,
        &recipes,
        ProductSourceMode::Canonical,
        Some("d612005-future2d"),
    )
    .unwrap();

    assert_eq!(planned.output_recipes, recipes);
    assert_eq!(planned.compute_recipes, vec![DerivedRecipe::LiftedIndex]);
    assert_eq!(planned.native_routes.len(), 3);
    assert_eq!(
        planned.native_routes[0].candidate.fetch_product,
        "d612005-future2d"
    );
    assert_eq!(
        planned
            .native_routes
            .iter()
            .map(|route| route.recipe)
            .collect::<Vec<_>>(),
        vec![
            DerivedRecipe::Sbcape,
            DerivedRecipe::BulkShear06km,
            DerivedRecipe::Srh03km,
        ]
    );
}

#[test]
fn wrf_gdex_non_d612005_products_fall_back_to_compute() {
    let recipes = vec![DerivedRecipe::Sbcape, DerivedRecipe::BulkShear06km];
    let planned = plan_native_thermo_routes_with_surface_product(
        ModelId::WrfGdex,
        &recipes,
        ProductSourceMode::Canonical,
        Some("d010047"),
    )
    .unwrap();

    assert_eq!(planned.output_recipes, recipes);
    assert_eq!(planned.compute_recipes, recipes);
    assert!(planned.native_routes.is_empty());
    assert!(planned.blockers.is_empty());
}

#[test]
fn locally_imported_derived_titles_name_the_grid_and_never_a_dataset_token() {
    // Every isobaric and 2 m product renders through this lane, which is
    // why the stray dataset token was on almost every frame.
    let mut request = sample_local_import_derived_request();
    request.title_provenance = TitleProvenance::LocalImport {
        grid_label: Some("d02 750 m".to_string()),
    };

    let title = derived_title_for_request(&request, "500mb Heights / Winds");

    assert_eq!(title, "500mb Heights / Winds (d02 750 m)");
    assert!(!title.contains("d612005"), "{title}");
}

#[test]
fn a_locally_imported_derived_batch_with_no_readable_grid_stamps_nothing() {
    let mut request = sample_local_import_derived_request();
    request.title_provenance = TitleProvenance::LocalImport { grid_label: None };

    assert_eq!(
        derived_title_for_request(&request, "500mb Heights / Winds"),
        "500mb Heights / Winds"
    );
}

#[test]
fn a_fetched_derived_batch_keeps_its_source_catalog_dataset_token() {
    // Negative control: the token must survive for frames that were really
    // fetched from the catalog it names.
    let request = sample_local_import_derived_request();

    assert_eq!(
        derived_title_for_request(&request, "500mb Heights / Winds"),
        "500mb Heights / Winds (d612005)"
    );
}

/// A batch shaped exactly like the shipped `woof render --engine rust`
/// one: store model `wrf`, a local provenance subtitle, and no product
/// overrides -- which is the combination that used to reach the fallback.
fn sample_local_import_derived_request() -> DerivedBatchRequest {
    DerivedBatchRequest {
        model: ModelId::WrfGdex,
        date_yyyymmdd: "20260801".to_string(),
        cycle_override_utc: Some(8),
        forecast_hour: 6,
        source: SourceId::Gdex,
        domain: DomainSpec::new("native_grid", (-104.0, -100.0, 40.0, 44.0)),
        out_dir: PathBuf::from("target\test-out"),
        cache_root: PathBuf::from("target\test-cache"),
        use_cache: false,
        recipe_slugs: Vec::new(),
        surface_product_override: None,
        pressure_product_override: None,
        source_mode: ProductSourceMode::Canonical,
        allow_large_heavy_domain: false,
        contour_mode: NativeContourRenderMode::Automatic,
        native_fill_level_multiplier: 1,
        output_width: OUTPUT_WIDTH,
        output_height: OUTPUT_HEIGHT,
        png_compression: PngCompressionMode::Default,
        place_label_overlay: None,
        output_suffix: None,
        subtitle_left_override: None,
        subtitle_right_override: Some("source: ArWen".to_string()),
        title_provenance: TitleProvenance::default(),
    }
}

#[test]
fn cycle_pinned_fastest_native_only_run_skips_pair_resolution() {
    let request = DerivedBatchRequest {
        model: ModelId::Hrrr,
        date_yyyymmdd: "20260418".to_string(),
        cycle_override_utc: Some(12),
        forecast_hour: 0,
        source: SourceId::Aws,
        domain: DomainSpec::new("midwest", (-104.0, -80.0, 34.0, 49.0)),
        out_dir: PathBuf::from("target\\test-out"),
        cache_root: PathBuf::from("target\\test-cache"),
        use_cache: true,
        recipe_slugs: vec!["sbcape".to_string()],
        surface_product_override: None,
        pressure_product_override: None,
        source_mode: ProductSourceMode::Fastest,
        allow_large_heavy_domain: false,
        contour_mode: NativeContourRenderMode::Automatic,
        native_fill_level_multiplier: 1,
        output_width: OUTPUT_WIDTH,
        output_height: OUTPUT_HEIGHT,
        png_compression: PngCompressionMode::Default,
        place_label_overlay: None,
        output_suffix: None,
        subtitle_left_override: None,
        subtitle_right_override: None,
        title_provenance: TitleProvenance::default(),
    };
    let planned = plan_native_thermo_routes_with_surface_product(
        request.model,
        &[DerivedRecipe::Sbcape],
        request.source_mode,
        request.surface_product_override.as_deref(),
    )
    .unwrap();

    let latest = resolve_derived_run(
        &request,
        &planned.compute_recipes,
        &planned.heavy_recipes,
        &planned.native_routes,
    )
    .unwrap();

    assert_eq!(latest.model, ModelId::Hrrr);
    assert_eq!(latest.cycle.date_yyyymmdd, "20260418");
    assert_eq!(latest.cycle.hour_utc, 12);
    assert_eq!(latest.source, SourceId::Aws);
}

#[test]
fn pressure_level_slice_or_interp_prefers_exact_isobaric_slice() {
    let pressure = TestPressureFields {
        pressure_levels_hpa: vec![850.0, 700.0],
        pressure_3d_pa: None,
        temperature_c_3d: vec![12.0, 13.0, 1.0, 2.0],
    };

    let slice = pressure_level_slice_or_interp(&pressure, &pressure.temperature_c_3d, 700.0, 2)
        .expect("exact 700 mb slice should be available");

    assert_eq!(slice, vec![1.0, 2.0]);
}

#[test]
fn pressure_level_slice_or_interp_interpolates_native_pressure_columns() {
    let pressure = TestPressureFields {
        pressure_levels_hpa: vec![900.0, 600.0],
        pressure_3d_pa: Some(vec![90000.0, 90000.0, 60000.0, 60000.0]),
        temperature_c_3d: vec![20.0, 24.0, 0.0, 4.0],
    };

    let slice = pressure_level_slice_or_interp(&pressure, &pressure.temperature_c_3d, 700.0, 2)
        .expect("native-pressure interpolation should succeed");

    let log_frac = (700.0_f64.ln() - 900.0_f64.ln()) / (600.0_f64.ln() - 900.0_f64.ln());
    let expected0 = 20.0 + log_frac * (0.0 - 20.0);
    let expected1 = 24.0 + log_frac * (4.0 - 24.0);

    assert!((slice[0] - expected0).abs() < 1.0e-6);
    assert!((slice[1] - expected1).abs() < 1.0e-6);
}
