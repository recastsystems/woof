use super::planning::{build_direct_execution_plan, partition_recipes_by_selector_availability};
use super::*;
use crate::shared_context::TitleProvenance;
use rustwx_core::{GridProjection, GridShape, LatLonGrid, SelectedField2D};

#[test]
fn chem_reference_display_conversions_are_exact_float32_multiplications() {
    for (slug, input, factor) in [("smoke_near_surface", 2.5e-8_f32, 1.0e9_f32),
                                  ("smoke_column", 4.0e-5_f32, 1.0e6_f32),
                                  ("aod_550", 0.5_f32, 1.0_f32),
                                  ("ozone_near_surface", 70.0_f32, 1.0_f32)] {
        let recipe = rustwx_models::plot_recipe(slug).unwrap();
        let (conversion, _) = super::rendering::direct_fill_unit_conversion(
            recipe, recipe.filled.selector.unwrap());
        let actual = conversion.apply(input);
        let expected = input * factor;
        let ulp = actual.to_bits().abs_diff(expected.to_bits());
        println!("ULP display-reference {slug} max={ulp}");
        assert_eq!(ulp, 0, "{slug}");
    }
}

fn sample_grid() -> LatLonGrid {
    LatLonGrid::new(
        GridShape::new(2, 2).unwrap(),
        vec![35.0, 35.0, 36.0, 36.0],
        vec![-100.0, -99.0, -100.0, -99.0],
    )
    .unwrap()
}

fn regular_geographic_grid_3x3() -> LatLonGrid {
    LatLonGrid::new(
        GridShape::new(3, 3).unwrap(),
        vec![34.0, 34.0, 34.0, 35.0, 35.0, 35.0, 36.0, 36.0, 36.0],
        vec![
            -101.0, -100.0, -99.0, -101.0, -100.0, -99.0, -101.0, -100.0, -99.0,
        ],
    )
    .unwrap()
}

fn skewed_geographic_grid_3x3() -> LatLonGrid {
    LatLonGrid::new(
        GridShape::new(3, 3).unwrap(),
        vec![34.0, 34.1, 34.2, 35.0, 35.1, 35.2, 36.0, 36.1, 36.2],
        vec![
            -101.0, -100.0, -99.0, -100.8, -99.8, -98.8, -100.6, -99.6, -98.6,
        ],
    )
    .unwrap()
}

fn sample_selected_field(
    selector: FieldSelector,
    units: &str,
    values: Vec<f32>,
) -> SelectedField2D {
    SelectedField2D::new(selector, units, sample_grid(), values).unwrap()
}

#[test]
fn barb_density_targets_thin_operational_synoptic_domains() {
    assert_eq!(
        barb_target_columns_rows((-127.0, -66.0, 23.0, 51.5)),
        (23.0, 14.0)
    );
    assert_eq!(
        barb_target_columns_rows((-170.0, -50.0, 5.0, 84.0)),
        (26.0, 13.0)
    );
    assert_eq!(
        barb_target_columns_rows((-180.0, 179.999, -90.0, 90.0)),
        (34.0, 16.0)
    );
}

#[test]
fn partition_blocks_recipe_whose_filled_selector_is_missing() {
    // Partial-success regression: direct_batch used to crash the
    // whole batch on the first missing GRIB message (GFS f000
    // missing APCP@Surface, ECMWF f000 missing RH@2m_agl). Now a
    // missing selector produces a per-recipe blocker and the rest
    // of the recipes still render.
    let rh_recipe = plot_recipe("2m_relative_humidity").expect("2m RH recipe should exist");
    let tmp_recipe = plot_recipe("2m_temperature").expect("2m temperature recipe should exist");

    let planned = vec![
        PlannedDirectRecipe {
            recipe: rh_recipe,
            plan: plot_recipe_fetch_plan(rh_recipe.slug, ModelId::Hrrr).unwrap(),
        },
        PlannedDirectRecipe {
            recipe: tmp_recipe,
            plan: plot_recipe_fetch_plan(tmp_recipe.slug, ModelId::Hrrr).unwrap(),
        },
    ];
    let mut missing = HashSet::new();
    missing.insert(
        rh_recipe
            .filled
            .selector
            .expect("2m RH recipe has a filled selector"),
    );

    let (renderable, blockers) = partition_recipes_by_selector_availability(&planned, &missing);
    assert_eq!(renderable.len(), 1);
    assert_eq!(renderable[0].recipe.slug, tmp_recipe.slug);
    assert_eq!(blockers.len(), 1);
    assert_eq!(blockers[0].recipe_slug, rh_recipe.slug);
    assert!(
        blockers[0].reason.contains("filled selector"),
        "blocker reason should mention the missing filled selector; got: {}",
        blockers[0].reason
    );
}

#[test]
fn empty_renderable_batch_returns_without_projected_map_failure() {
    let request = sample_direct_request(ModelId::Hrrr);
    let latest = LatestRun {
        model: ModelId::Hrrr,
        cycle: rustwx_core::CycleSpec::new("20260414", 23).unwrap(),
        source: SourceId::Nomads,
    };

    let rendered = render_direct_recipes(
        &request,
        &latest,
        &[],
        &HashMap::new(),
        &HashMap::new(),
        None,
    )
    .expect("empty renderable batches should not fail projected-map prep");

    assert!(rendered.is_empty());
}

#[test]
fn prepared_projected_maps_use_composite_components_as_projection_sample() {
    let request = sample_direct_request(ModelId::Hrrr);
    let recipe = plot_recipe("cloud_cover_levels").expect("cloud-cover panel recipe should exist");
    let plan = plot_recipe_fetch_plan(recipe.slug, ModelId::Hrrr)
        .expect("cloud-cover panel should have a direct fetch plan");
    let planned = vec![PlannedDirectRecipe { recipe, plan }];
    let mut extracted = HashMap::new();
    for selector in planned[0].plan.selectors() {
        extracted.insert(
            selector,
            sample_selected_field(selector, "%", vec![10.0, 20.0, 30.0, 40.0]),
        );
    }

    let prepared = build_prepared_projected_maps(&request, &planned, &extracted)
        .expect("composite-only batches should prepare panel projections");
    let spec = composite_panel_spec(recipe.slug)
        .expect("cloud-cover recipe should be a composite panel")
        .scaled_for_request(&request);
    let sample_selector =
        projected_sample_selector(&planned[0]).expect("composite panel should expose a selector");
    let sample_field = extracted
        .get(&sample_selector)
        .expect("sample selector should have an extracted field");

    assert!(prepared.contains_key(&projected_map_cache_key(
        spec.panel_width,
        spec.panel_height,
        visual_mode_cache_key(ProductVisualMode::PanelMember),
        sample_field,
    )));
}

fn sample_direct_request(model: ModelId) -> DirectBatchRequest {
    DirectBatchRequest {
        accumulation_window: None,
        model,
        date_yyyymmdd: "20260414".to_string(),
        cycle_override_utc: Some(23),
        forecast_hour: 6,
        source: rustwx_models::model_summary(model).sources[0].id,
        domain: DomainSpec::new("midwest", (-105.0, -80.0, 30.0, 50.0)),
        out_dir: PathBuf::from("C:\\temp\\rustwx-tests"),
        cache_root: PathBuf::from("C:\\temp\\rustwx-tests-cache"),
        use_cache: false,
        recipe_slugs: Vec::new(),
        product_overrides: HashMap::new(),
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
        geographic_overlays: None,
        panel_annotations: None,
    }
}

#[test]
fn snowfall_titles_name_the_resolved_window_before_the_domain() {
    let mut request = sample_direct_request(ModelId::WrfGdex);
    request.title_provenance = TitleProvenance::LocalImport { grid_label: Some("d02 750 m".into()) };
    let parse = |value: &str| chrono::NaiveDateTime::parse_from_str(value, "%Y-%m-%d_%H:%M:%S")
        .unwrap().and_utc().timestamp();
    let init = parse("2026-10-08_18:00:00");
    let end = parse("2026-10-12_18:00:00");
    let whole_hour = local_accumulation_window_from_cycle("20261008", 18, 96, Some("2026-10-11_18:00:00")).unwrap();
    assert_eq!((whole_hour.start_unix, whole_hour.end_unix), (parse("2026-10-11_18:00:00"), end));
    request.accumulation_window = Some(local_accumulation_window(init, end, Some("2026-10-11_18:00:00")).unwrap());
    for (base, expected) in [
        ("Kuchera snowfall, window total", "24 h Kuchera snowfall ending Mon 12 Oct 18Z (d02 750 m)"),
        ("10:1 snowfall, window total", "24 h 10:1 snowfall ending Mon 12 Oct 18Z (d02 750 m)"),
        ("Model snowfall, window total", "24 h Model snowfall ending Mon 12 Oct 18Z (d02 750 m)"),
    ] {
        assert_eq!(direct_title_for_planned_product(&request, "native_grid", base), expected);
    }
    request.panel_annotations = Some(serde_json::from_str(
        r#"{"title_suffix":", 24 h ending Mon 12 Oct 18Z","subtitle_center":"keep"}"#,
    ).unwrap());
    let annotations = titles::product_annotations(&request, "Kuchera snowfall, window total").unwrap();
    assert!(annotations.title_suffix.is_none(), "the legacy caption now repeats the resolved window");
    assert_eq!(annotations.subtitle_center.as_deref(), Some("keep"));
    request.panel_annotations.as_mut().unwrap().title = Some("Caller title".into());
    assert!(titles::product_annotations(&request, "Kuchera snowfall, window total").unwrap().title_suffix.is_some());
    request.panel_annotations.as_mut().unwrap().title = None;
    request.panel_annotations.as_mut().unwrap().title_suffix = Some("EXPERIMENTAL".into());
    assert_eq!(titles::product_annotations(&request, "Kuchera snowfall, window total").unwrap().title_suffix.as_deref(), Some("EXPERIMENTAL"));
    request.accumulation_window = Some(local_accumulation_window(init, end, None).unwrap());
    assert_eq!(direct_title_for_planned_product(&request, "native_grid", "Kuchera snowfall, window total"),
        "Kuchera snowfall since Thu 8 Oct 18Z (d02 750 m)");
    request.accumulation_window = Some(local_accumulation_window(init, end, Some("2026-10-11_18:00:30")).unwrap());
    assert_eq!(direct_title_for_planned_product(&request, "native_grid", "Kuchera snowfall, window total"),
        "Kuchera snowfall since Sun 11 Oct 18:00:30Z (d02 750 m)");
    assert_eq!(direct_title_for_planned_product(&request, "native_grid", "2 m temperature"),
        "2 m temperature (d02 750 m)");
}

/// A title note reaches its own recipe's title only, ahead of a caller's
/// suffix, and JSON without `title_notes` (every `--annotate` file written
/// so far) still parses.
#[test]
fn a_title_note_reaches_only_its_own_recipe() {
    let mut request = sample_direct_request(ModelId::WrfGdex);
    let mut annotations: crate::geographic_overlays::PanelAnnotations =
        serde_json::from_str(r#"{"subtitle_center":"keep"}"#).unwrap();
    assert!(annotations.title_notes.is_empty());
    annotations.title_notes.insert("composite_reflectivity".into(), "incl. parameterised convection".into());
    request.panel_annotations = Some(annotations);
    let refc = titles::recipe_annotations(&request, "composite_reflectivity", "Composite Reflectivity").unwrap();
    assert_eq!(refc.title_suffix.as_deref(), Some("incl. parameterised convection"));
    assert!(refc.title_notes.is_empty());
    assert_eq!(refc.subtitle_center.as_deref(), Some("keep"));
    let t2 = titles::recipe_annotations(&request, "2m_temperature", "2 m temperature").unwrap();
    assert_eq!(t2.title_suffix, None);
    request.panel_annotations.as_mut().unwrap().title_suffix = Some("EXPERIMENTAL".into());
    let refc = titles::recipe_annotations(&request, "composite_reflectivity", "Composite Reflectivity").unwrap();
    assert_eq!(refc.title_suffix.as_deref(), Some("incl. parameterised convection EXPERIMENTAL"));
}

#[test]
fn native_stat_product_overrides_promote_stat_to_static_titles() {
    let mut request = sample_direct_request(ModelId::Sref);
    request.product_overrides.insert(
        "ensprod/pgrb212/mean_3hrly".to_string(),
        "ensprod/pgrb212/p50_3hrly".to_string(),
    );

    assert_eq!(
        native_stat_label_for_request(&request, Some("ensprod/pgrb212/mean_3hrly")).as_deref(),
        Some("P50")
    );
    let title = direct_title_for_planned_product(
        &request,
        "ensprod/pgrb212/mean_3hrly",
        "2m Temperature + 10m Winds",
    );
    assert!(
        title.starts_with("SREF P50 2m Temperature + 10m Winds"),
        "{title}"
    );
}

#[test]
fn native_stat_title_prefix_keeps_existing_model_prefix_first() {
    assert_eq!(
        apply_native_stat_title_prefix(ModelId::Sref, "Spread", "SREF 2m Dewpoint"),
        "SREF Spread 2m Dewpoint"
    );
    assert_eq!(
        apply_native_stat_title_prefix(ModelId::Sref, "Mean", "SREF Mean 2m Dewpoint"),
        "SREF Mean 2m Dewpoint"
    );
}

#[test]
fn local_wrf_netcdf_titles_omit_gdex_dataset_token() {
    let mut request = sample_direct_request(ModelId::WrfGdex);
    request.subtitle_right_override = Some("source: local WRF NetCDF".to_string());

    let title =
        direct_title_for_planned_product(&request, "d612005-hist2d", "Composite Reflectivity / UH");

    assert!(title.starts_with("Composite Reflectivity / UH"), "{title}");
    assert!(!title.contains("d612005"), "{title}");
}

#[test]
fn locally_imported_titles_name_the_grid_and_never_a_dataset_token() {
    // The shipped local-import path: the store model identity is still
    // `wrf`, and the provenance subtitle says `source: ArWen`, not "local
    // WRF NetCDF".  Before the grid identity was plumbed, that combination
    // fell through to the GDEX fallback and stamped a dataset id this lane
    // never read onto every frame anyone screenshots.
    let mut request = sample_direct_request(ModelId::WrfGdex);
    request.subtitle_right_override = Some("source: ArWen".to_string());
    request.title_provenance = TitleProvenance::LocalImport {
        grid_label: Some("d02 750 m".to_string()),
    };

    let title = direct_title_for_planned_product(&request, "native_grid", "MSLP / 10m Winds");

    assert_eq!(title, "MSLP / 10m Winds (d02 750 m)");
    assert!(!title.contains("d612005"), "{title}");
}

#[test]
fn a_locally_imported_batch_with_no_readable_grid_stamps_nothing() {
    // A wrfout with no usable GRID_ID/DX: no parenthetical at all beats a
    // borrowed one.  This is the arm that keeps the fallback from creeping
    // back in as "well, we had to print something".
    let mut request = sample_direct_request(ModelId::WrfGdex);
    request.subtitle_right_override = Some("source: ArWen".to_string());
    request.title_provenance = TitleProvenance::LocalImport { grid_label: None };

    let title = direct_title_for_planned_product(&request, "native_grid", "MSLP / 10m Winds");

    assert_eq!(title, "MSLP / 10m Winds");
}

#[test]
fn a_fetched_batch_keeps_its_source_catalog_dataset_token() {
    // The negative control for the two above: nothing about this change may
    // strip the dataset token from frames that really did come from the
    // catalog, because for those it is the only thing naming the archive.
    let request = sample_direct_request(ModelId::WrfGdex);

    let title = direct_title_for_planned_product(&request, "d612005-hist2d", "Composite Reflectivity / UH");

    assert_eq!(title, "Composite Reflectivity / UH (d612005)");
}

#[test]
fn global_scale_domain_detection_handles_dateline_bounds() {
    assert!(is_global_scale_domain((-180.0, 179.999, -90.0, 90.0)));
    assert!(!is_global_scale_domain((-125.0, -66.0, 24.0, 50.0)));
}

#[test]
fn full_world_geographic_bounds_include_all_longitudes() {
    let bounds = (-180.0, 180.0, -90.0, 90.0);

    assert!(point_in_geographic_bounds(-179.5, 0.0, bounds));
    assert!(point_in_geographic_bounds(-90.0, 0.0, bounds));
    assert!(point_in_geographic_bounds(0.0, 0.0, bounds));
    assert!(point_in_geographic_bounds(90.0, 0.0, bounds));
    assert!(point_in_geographic_bounds(179.5, 0.0, bounds));
}

fn periodic_global_grid() -> rustwx_core::LatLonGrid {
    let nx = 36usize;
    let ny = 3usize;
    let mut lat = Vec::with_capacity(nx * ny);
    let mut lon = Vec::with_capacity(nx * ny);
    for row_lat in [-10.0_f32, 0.0, 10.0] {
        for x in 0..nx {
            lat.push(row_lat);
            lon.push((x as f32) * 10.0);
        }
    }
    rustwx_core::LatLonGrid::new(rustwx_core::GridShape::new(nx, ny).unwrap(), lat, lon).unwrap()
}

#[test]
fn periodic_global_crop_wraps_regional_domains_across_greenwich() {
    let grid = periodic_global_grid();

    let crop = crop_for_direct_grid(&grid, (-12.0, 12.0, -2.0, 2.0), 1, true)
        .unwrap()
        .expect("regional Greenwich crop should trim the periodic axis");

    assert_eq!(
        crop,
        DirectGridCrop::Wrapped {
            x_start: 34,
            x_end: 3,
            y_start: 0,
            y_end: 3,
        }
    );

    let cropped = crop_latlon_grid_for_direct(&grid, crop).unwrap();
    assert_eq!(cropped.shape.nx, 5);
    assert_eq!(cropped.shape.ny, 3);
    assert_eq!(&cropped.lon_deg[0..5], &[340.0, 350.0, 0.0, 10.0, 20.0]);
}

#[test]
fn periodic_global_direct_crop_normalizes_longitudes_near_domain_center() {
    let grid = periodic_global_grid();
    let values = (0..grid.shape.nx * grid.shape.ny)
        .map(|value| value as f32)
        .collect::<Vec<_>>();
    let field = SelectedField2D::new(
        FieldSelector::isobaric(CanonicalField::GeopotentialHeight, 300),
        "m",
        grid,
        values,
    )
    .unwrap();

    let cropped =
        crop_selected_field_for_domain(&field, (-12.0, 12.0, -2.0, 2.0), 1, true).unwrap();

    assert_eq!(cropped.grid.shape.nx, 5);
    assert_eq!(cropped.grid.shape.ny, 3);
    assert_eq!(
        &cropped.grid.lon_deg[0..5],
        &[-20.0, -10.0, 0.0, 10.0, 20.0]
    );
}

#[test]
fn streamline_auto_mode_disables_regular_latlon_grids() {
    assert!(!streamlines_enabled_for_grid(
        StreamlineSetting::Auto,
        &periodic_global_grid()
    ));
    assert!(streamlines_enabled_for_grid(
        StreamlineSetting::Enabled,
        &periodic_global_grid()
    ));
}

/// The front door outranks the environment, and saying nothing changes
/// nothing.
///
/// The concrete breakage: streamlines had no command-line spelling at all,
/// so a user could ask for a wind product and had no visible switch for
/// which wind layer to draw. Once a flag exists it has to WIN: a
/// `--streamlines` overruled by a stale `RUSTWX_WIND_STREAMLINES=0` in a
/// shell profile would be a flag that lies, with nothing on screen saying
/// why.
#[test]
fn a_command_line_streamline_request_outranks_the_environment() {
    // The setting is process-wide; this is the only test that moves it.
    static GUARD: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _held = GUARD.lock().unwrap_or_else(|poisoned| poisoned.into_inner());

    crate::shared_context::set_wind_streamline_request(None);
    let unasked = static_streamline_setting();

    crate::shared_context::set_wind_streamline_request(Some(true));
    assert_eq!(static_streamline_setting(), StreamlineSetting::Enabled);
    assert!(streamlines_enabled_for_grid(
        static_streamline_setting(),
        &periodic_global_grid()
    ));

    crate::shared_context::set_wind_streamline_request(Some(false));
    assert_eq!(static_streamline_setting(), StreamlineSetting::Disabled);
    assert!(!streamlines_enabled_for_grid(
        static_streamline_setting(),
        &periodic_global_grid()
    ));

    crate::shared_context::set_wind_streamline_request(None);
    assert_eq!(
        static_streamline_setting(),
        unasked,
        "clearing the request must hand the choice back untouched"
    );
}

#[test]
fn inverse_raster_latlon_maps_clip_regional_bounds() {
    let bounds = (110.0, 180.0, -50.0, 0.0);
    let grid = regular_geographic_grid_3x3();
    let inverse =
        inverse_raster_projection_for_grid(Some(&GridProjection::Geographic), bounds, &grid)
            .expect("regional regular lat/lon maps should use inverse raster");

    assert!(inverse.clip_bounds.is_some());
    assert_eq!(
        inverse.reference_longitude_deg,
        Some(center_longitude_for_bounds(bounds))
    );
}

#[test]
fn inverse_raster_requires_rectilinear_geographic_mesh() {
    let bounds = (-127.0, -66.0, 23.0, 51.5);
    let grid = skewed_geographic_grid_3x3();

    let inverse =
        inverse_raster_projection_for_grid(Some(&GridProjection::Geographic), bounds, &grid);

    assert!(
        inverse.is_none(),
        "rotated or curvilinear grids tagged geographic must not use regular-axis inverse raster"
    );
}

#[test]
fn inverse_raster_does_not_geo_clip_projected_conus_frames() {
    let clip = inverse_raster_clip_bounds(
        (-127.0, -66.0, 23.0, 51.5),
        &rustwx_render::ProjectionSpec::LambertConformal {
            standard_parallel_1_deg: PIVOTAL_CONUS_STANDARD_PARALLEL_1_DEG,
            standard_parallel_2_deg: PIVOTAL_CONUS_STANDARD_PARALLEL_2_DEG,
            central_meridian_deg: PIVOTAL_CONUS_CENTRAL_MERIDIAN_DEG,
        },
    );

    assert!(clip.is_none());
}

#[test]
fn broad_native_projected_grids_use_full_domain_frame_by_default() {
    let lambert = GridProjection::LambertConformal {
        standard_parallel_1_deg: 33.0,
        standard_parallel_2_deg: 45.0,
        central_meridian_deg: -96.0,
    };
    let conus_bounds = (-127.0, -66.0, 23.0, 51.5);
    let local_bounds = (-124.0, -122.0, 44.5, 46.0);

    assert!(full_domain_projected_frame_default(
        Some(&lambert),
        conus_bounds
    ));
    assert!(!full_domain_projected_frame_default(
        Some(&lambert),
        local_bounds
    ));
    assert!(!full_domain_projected_frame_default(
        Some(&GridProjection::Geographic),
        conus_bounds,
    ));
    assert!(!full_domain_projected_frame_default(None, conus_bounds));
}

#[test]
fn requested_projected_map_builder_bypasses_native_full_domain_default() {
    let lambert = GridProjection::LambertConformal {
        standard_parallel_1_deg: 33.0,
        standard_parallel_2_deg: 45.0,
        central_meridian_deg: -96.0,
    };
    let conus_bounds = (-127.0, -66.0, 23.0, 51.5);
    assert!(full_domain_projected_frame_default(
        Some(&lambert),
        conus_bounds
    ));

    let grid = sample_grid();
    let full_domain = build_projected_map_with_projection(
        &grid.lat_deg,
        &grid.lon_deg,
        Some(&lambert),
        conus_bounds,
        16.0 / 9.0,
    )
    .expect("full-domain projected map should build");
    let requested_domain = build_requested_projected_map_with_projection(
        &grid.lat_deg,
        &grid.lon_deg,
        Some(&lambert),
        conus_bounds,
        16.0 / 9.0,
    )
    .expect("requested-domain projected map should build");

    let full_width = full_domain.extent.x_max - full_domain.extent.x_min;
    let requested_width = requested_domain.extent.x_max - requested_domain.extent.x_min;
    assert!(
        requested_width > full_width * 10.0,
        "requested frame should follow map bounds, not the tiny native mesh extent"
    );
}

#[test]
fn pivotal_lambert_variant_uses_fixed_conus_projection_for_geographic_grids() {
    let bounds = (-127.0, -66.0, 23.0, 51.5);
    let projection = presentation_projection_for_bounds(
        Some(&GridProjection::Geographic),
        bounds,
        ProjectionPresentationVariant::PivotalLambert,
    )
    .expect("CONUS geographic grids should get a presentation projection");

    assert_eq!(
        projection,
        rustwx_render::ProjectionSpec::LambertConformal {
            standard_parallel_1_deg: PIVOTAL_CONUS_STANDARD_PARALLEL_1_DEG,
            standard_parallel_2_deg: PIVOTAL_CONUS_STANDARD_PARALLEL_2_DEG,
            central_meridian_deg: PIVOTAL_CONUS_CENTRAL_MERIDIAN_DEG,
        }
    );
    assert_eq!(
        reference_latitude_for_projection_variant(
            ProjectionPresentationVariant::PivotalLambert,
            Some(&GridProjection::Geographic),
            bounds,
        ),
        Some(PIVOTAL_CONUS_REFERENCE_LATITUDE_DEG)
    );
}

#[test]
fn pivotal_lambert_variant_keeps_global_geographic_grids_on_robinson() {
    let bounds = (-180.0, 179.999, -90.0, 90.0);
    let projection = presentation_projection_for_bounds(
        Some(&GridProjection::Geographic),
        bounds,
        ProjectionPresentationVariant::PivotalLambert,
    )
    .expect("global geographic grids should get a presentation projection");

    assert!(matches!(
        projection,
        rustwx_render::ProjectionSpec::Robinson {
            central_meridian_deg
        } if central_meridian_deg == 0.0
    ));
    assert_eq!(
        reference_latitude_for_projection_variant(
            ProjectionPresentationVariant::PivotalLambert,
            Some(&GridProjection::Geographic),
            bounds,
        ),
        None
    );
}

/// One batch, two panels, two DIFFERENT projections -- the measured
/// finding behind the georeference feature.  A regional CONUS panel is
/// drawn in a presentation Lambert and a global panel in Robinson, so any
/// consumer that assumes one transform per batch is wrong on arrival.
/// Pinned here so a future single-transform assumption fails loudly, and
/// pinned through `panel_resolved_projection` -- the function that
/// PUBLISHES the transform -- rather than through the projection ladder
/// alone, so the published answer itself carries the difference.  The
/// Lambert's resolved reference latitude is asserted too: it exists only
/// on the RESOLVED projection, which is why publishing the spec alone
/// under-determined the map.
#[test]
fn one_batch_publishes_different_projections_for_regional_and_global_panels() {
    let conus_grid = regular_geographic_grid_3x3();
    let regional = panel_resolved_projection(
        &conus_grid.lat_deg,
        &conus_grid.lon_deg,
        None,
        (-127.0, -66.0, 23.0, 51.5),
        16.0 / 9.0,
    )
    .expect("a CONUS panel must resolve a projection");

    let global_grid = LatLonGrid::new(
        GridShape::new(3, 3).unwrap(),
        vec![-80.0, -80.0, -80.0, 0.0, 0.0, 0.0, 80.0, 80.0, 80.0],
        vec![
            -170.0, 0.0, 170.0, -170.0, 0.0, 170.0, -170.0, 0.0, 170.0,
        ],
    )
    .unwrap();
    let global = panel_resolved_projection(
        &global_grid.lat_deg,
        &global_grid.lon_deg,
        None,
        (-180.0, 180.0, -90.0, 90.0),
        16.0 / 9.0,
    )
    .expect("a global panel must resolve a projection");

    match regional {
        rustwx_render::ResolvedProjection::LambertConformal {
            reference_latitude_deg,
            ..
        } => assert_eq!(
            reference_latitude_deg, PIVOTAL_CONUS_REFERENCE_LATITUDE_DEG,
            "the resolved Lambert must carry the derived reference latitude \
             the spec alone does not have"
        ),
        other => panic!("a CONUS panel must present in Lambert, got {other:?}"),
    }
    assert!(
        matches!(
            global,
            rustwx_render::ResolvedProjection::Robinson { .. }
        ),
        "a global panel must present in Robinson, got {global:?}"
    );
    assert_ne!(
        regional, global,
        "two panels of one batch must be free to publish different projections"
    );
}

#[test]
fn albers_variant_uses_conus_equal_area_regionally_and_robinson_globally() {
    let conus_bounds = (-127.0, -66.0, 23.0, 51.5);
    let conus_projection = presentation_projection_for_bounds(
        Some(&GridProjection::Geographic),
        conus_bounds,
        ProjectionPresentationVariant::AlbersEqualArea,
    )
    .expect("CONUS geographic grids should get a presentation projection");

    assert_eq!(
        conus_projection,
        rustwx_render::ProjectionSpec::AlbersEqualArea {
            standard_parallel_1_deg: 29.5,
            standard_parallel_2_deg: 45.5,
            central_meridian_deg: -96.0,
            latitude_of_origin_deg: 23.0,
        }
    );

    let global_projection = presentation_projection_for_bounds(
        Some(&GridProjection::Geographic),
        (-180.0, 179.999, -90.0, 90.0),
        ProjectionPresentationVariant::AlbersEqualArea,
    )
    .expect("global geographic grids should get a presentation projection");

    assert!(matches!(
        global_projection,
        rustwx_render::ProjectionSpec::Robinson {
            central_meridian_deg
        } if central_meridian_deg == 0.0
    ));
}

#[test]
fn mercator_variant_uses_mercator_regionally_and_robinson_globally() {
    let conus_bounds = (-127.0, -66.0, 23.0, 51.5);
    let conus_projection = presentation_projection_for_bounds(
        Some(&GridProjection::Geographic),
        conus_bounds,
        ProjectionPresentationVariant::Mercator,
    )
    .expect("CONUS geographic grids should get a presentation projection");

    assert!(matches!(
        conus_projection,
        rustwx_render::ProjectionSpec::Mercator { .. }
    ));

    let global_projection = presentation_projection_for_bounds(
        Some(&GridProjection::Geographic),
        (-180.0, 179.999, -90.0, 90.0),
        ProjectionPresentationVariant::Mercator,
    )
    .expect("global geographic grids should get a presentation projection");

    assert!(matches!(
        global_projection,
        rustwx_render::ProjectionSpec::Robinson {
            central_meridian_deg
        } if central_meridian_deg == 0.0
    ));
}

#[test]
fn rectangular_variant_uses_geographic_regionally_and_robinson_globally() {
    let europe_bounds = (-25.0, 45.0, 34.0, 72.0);
    let europe_projection = presentation_projection_for_bounds(
        Some(&GridProjection::Geographic),
        europe_bounds,
        ProjectionPresentationVariant::RectangularGeographic,
    )
    .expect("regional geographic grids should get a presentation projection");
    assert_eq!(europe_projection, rustwx_render::ProjectionSpec::Geographic);

    let global_projection = presentation_projection_for_bounds(
        Some(&GridProjection::Geographic),
        (-180.0, 179.999, -90.0, 90.0),
        ProjectionPresentationVariant::RectangularGeographic,
    )
    .expect("global geographic grids should get a presentation projection");
    assert!(matches!(
        global_projection,
        rustwx_render::ProjectionSpec::Robinson {
            central_meridian_deg
        } if central_meridian_deg == 0.0
    ));
}

#[test]
fn adaptive_geographic_regions_use_presentation_projections() {
    assert!(matches!(
        presentation_projection_for_bounds(
            Some(&GridProjection::Geographic),
            (-180.0, 179.999, -90.0, 90.0),
            ProjectionPresentationVariant::Adaptive,
        )
        .unwrap(),
        rustwx_render::ProjectionSpec::Robinson { .. }
    ));

    let conus_projection = presentation_projection_for_bounds(
        Some(&GridProjection::Geographic),
        (-127.0, -66.0, 23.0, 51.5),
        ProjectionPresentationVariant::Adaptive,
    )
    .unwrap();
    assert!(matches!(
        conus_projection,
        rustwx_render::ProjectionSpec::LambertConformal { .. }
    ));

    let europe_projection = presentation_projection_for_bounds(
        Some(&GridProjection::Geographic),
        (-25.0, 45.0, 34.0, 72.0),
        ProjectionPresentationVariant::Adaptive,
    )
    .unwrap();
    assert!(matches!(
        europe_projection,
        rustwx_render::ProjectionSpec::LambertConformal { .. }
    ));

    let north_america_projection = presentation_projection_for_bounds(
        Some(&GridProjection::Geographic),
        (-170.0, -50.0, 5.0, 84.0),
        ProjectionPresentationVariant::Adaptive,
    )
    .unwrap();
    assert_eq!(
        north_america_projection,
        rustwx_render::ProjectionSpec::LambertConformal {
            standard_parallel_1_deg: 25.0,
            standard_parallel_2_deg: 60.0,
            central_meridian_deg: -100.0,
        }
    );

    assert!(matches!(
        presentation_projection_for_bounds(
            Some(&GridProjection::Geographic),
            (-180.0, 179.999, -90.0, -60.0),
            ProjectionPresentationVariant::Adaptive,
        )
        .unwrap(),
        rustwx_render::ProjectionSpec::PolarStereographic { .. }
    ));
}

/// A western domain is drawn in its own grid projection like every other
/// domain.  The north-up Mercator it used to get by default, chosen by a
/// hard-coded western lat/lon box, drew the Lambert grid as a slanted
/// quadrilateral with blank corner wedges; it is now only an explicit
/// `RUSTWX_STRAIGHT_WEST_PROJECTION` opt-in.
#[test]
fn adaptive_native_lambert_keeps_western_coast_domains_on_the_grid_projection() {
    let native = GridProjection::LambertConformal {
        standard_parallel_1_deg: 38.5,
        standard_parallel_2_deg: 38.5,
        central_meridian_deg: -97.5,
    };
    for bounds in [(-124.9, -113.8, 31.9, 42.5), (-126.5, -108.0, 30.0, 50.5)] {
        let projection = presentation_projection_for_bounds(
            Some(&native),
            bounds,
            ProjectionPresentationVariant::Adaptive,
        )
        .unwrap();
        assert!(
            matches!(projection, rustwx_render::ProjectionSpec::LambertConformal { .. }),
            "{bounds:?}: {projection:?}"
        );
    }
}

#[test]
fn adaptive_native_lambert_keeps_rockies_on_native_projection() {
    let native = GridProjection::LambertConformal {
        standard_parallel_1_deg: 38.5,
        standard_parallel_2_deg: 38.5,
        central_meridian_deg: -97.5,
    };

    let projection = presentation_projection_for_bounds(
        Some(&native),
        (-112.0, -96.0, 37.0, 49.5),
        ProjectionPresentationVariant::Adaptive,
    )
    .unwrap();
    assert_eq!(projection, native.into());
}

#[test]
fn explicit_rectangular_variant_overrides_native_lambert() {
    let native = GridProjection::LambertConformal {
        standard_parallel_1_deg: 38.5,
        standard_parallel_2_deg: 38.5,
        central_meridian_deg: -97.5,
    };

    let projection = presentation_projection_for_bounds(
        Some(&native),
        (-124.9, -113.8, 31.9, 42.5),
        ProjectionPresentationVariant::RectangularGeographic,
    )
    .unwrap();
    assert_eq!(projection, rustwx_render::ProjectionSpec::Geographic);
}

#[test]
fn rectangular_variant_expands_tall_bounds_to_target_aspect() {
    let bounds = (110.0, 180.0, -50.0, 0.0);
    let expanded = presentation_frame_bounds_for_grid(
        Some(&GridProjection::Geographic),
        bounds,
        ProjectionPresentationVariant::RectangularGeographic,
        16.0 / 9.0,
    );

    assert!((expanded.3 - expanded.2 - 50.0).abs() < 1.0e-6);
    assert!(
        longitude_bounds_span_deg(expanded) > longitude_bounds_span_deg(bounds),
        "expanded bounds should widen the crop for a 16:9 rectangular map"
    );
}

/// Test-only equivalent of the legacy `build_direct_fetch_request`
/// helper. Tests still want to assert that direct's fetch identity
/// stays consistent across HRRR's nat→sfc routing and product
/// overrides; the production path now builds requests inside the
/// loader, but the same routing logic lives in the planner so this
/// thin shim stays accurate.
fn build_direct_fetch_request(
    request: &DirectBatchRequest,
    latest: &LatestRun,
    forecast_hour: u16,
    group: &FetchGroup,
) -> Result<rustwx_io::FetchRequest, rustwx_core::RustwxError> {
    Ok(rustwx_io::FetchRequest {
        request: rustwx_core::ModelRunRequest::new(
            request.model,
            latest.cycle.clone(),
            forecast_hour,
            group.product.as_str(),
        )?,
        source_override: Some(latest.source),
        variable_patterns: if should_attach_direct_idx_patterns(latest.source) {
            group.variable_patterns.clone()
        } else {
            Vec::new()
        },
    })
}

#[test]
fn planning_hrrr_direct_batch_dedupes_recipe_aliases() {
    let planned = plan_direct_recipes(
        ModelId::Hrrr,
        &[
            "500mb_temperature_height_winds".to_string(),
            "500mb temperature height winds".to_string(),
        ],
    )
    .unwrap();
    assert_eq!(planned.len(), 1);
    assert_eq!(planned[0].recipe.slug, "500mb_temperature_height_winds");
    assert_eq!(planned[0].plan.product, "prs");
}

#[test]
fn grouping_preserves_logical_family_aliases_when_nat_reroutes_to_sfc() {
    let planned = plan_direct_recipes(
        ModelId::Hrrr,
        &[
            "composite_reflectivity".to_string(),
            "2m_temperature_10m_winds".to_string(),
        ],
    )
    .unwrap();
    let request = sample_direct_request(ModelId::Hrrr);
    let groups = group_direct_fetches(&request, &planned);
    // Both recipes share the canonical sfc fetch, but the logical
    // planning recorded "nat" for composite_reflectivity; the alias
    // set must retain both "nat" and "sfc" for provenance.
    let sfc_group = groups
        .iter()
        .find(|group| group.product == "sfc")
        .expect("expected a canonical sfc fetch group");
    assert!(sfc_group.planned_family_aliases.contains("nat"));
    assert!(sfc_group.planned_family_aliases.contains("sfc"));
}

#[test]
fn grouping_keeps_shared_prs_selector_union_under_structured_fetches() {
    let planned = plan_direct_recipes(
        ModelId::Hrrr,
        &[
            "500mb_temperature_height_winds".to_string(),
            "700mb_temperature_height_winds".to_string(),
        ],
    )
    .unwrap();
    let request = sample_direct_request(ModelId::Hrrr);
    let groups = group_direct_fetches(&request, &planned);
    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0].product, "prs");
    assert_eq!(
        groups[0].fetch_mode,
        PlotRecipeFetchMode::WholeFileStructuredExtract
    );
    assert!(
        groups[0]
            .selectors
            .contains(&FieldSelector::isobaric(CanonicalField::Temperature, 500))
    );
    assert!(
        groups[0]
            .selectors
            .contains(&FieldSelector::isobaric(CanonicalField::Temperature, 700))
    );
    assert!(
        groups[0]
            .variable_patterns
            .iter()
            .any(|pattern| pattern.contains("500 mb"))
    );
    assert!(
        groups[0]
            .variable_patterns
            .iter()
            .any(|pattern| pattern.contains("700 mb"))
    );
}

#[test]
fn direct_fetch_request_strips_nomads_subset_patterns() {
    let request = sample_direct_request(ModelId::Hrrr);
    let latest = LatestRun {
        model: ModelId::Hrrr,
        cycle: rustwx_core::CycleSpec::new("20260414", 23).unwrap(),
        source: SourceId::Nomads,
    };
    let group = FetchGroup {
        product: "prs".to_string(),
        fetch_mode: PlotRecipeFetchMode::WholeFileStructuredExtract,
        variable_patterns: vec!["TMP:500 mb".to_string()],
        selectors: vec![FieldSelector::isobaric(CanonicalField::Temperature, 500)],
        planned_family_aliases: std::collections::BTreeSet::from(["prs".to_string()]),
    };
    let fetch = build_direct_fetch_request(&request, &latest, 6, &group).unwrap();
    assert_eq!(fetch.request.product, "prs");
    assert_eq!(fetch.source_override, Some(SourceId::Nomads));
    assert!(fetch.variable_patterns.is_empty());
}

#[test]
fn native_fetches_share_surface_family_file() {
    let request = sample_direct_request(ModelId::Hrrr);
    let latest = LatestRun {
        model: ModelId::Hrrr,
        cycle: rustwx_core::CycleSpec::new("20260414", 23).unwrap(),
        source: SourceId::Aws,
    };
    let group = FetchGroup {
        product: canonical_fetch_product(&request, "nat"),
        fetch_mode: PlotRecipeFetchMode::WholeFileStructuredExtract,
        variable_patterns: Vec::new(),
        selectors: vec![FieldSelector::entire_atmosphere(
            CanonicalField::CompositeReflectivity,
        )],
        planned_family_aliases: std::collections::BTreeSet::from(["nat".to_string()]),
    };
    let fetch = build_direct_fetch_request(&request, &latest, 6, &group).unwrap();
    assert_eq!(fetch.request.product, "sfc");
}

#[test]
fn smoke_fetches_stay_on_hrrr_wrfnat_family() {
    let planned = plan_direct_recipes(
        ModelId::Hrrr,
        &[
            "smoke_pm25_native".to_string(),
            "2m_temperature_10m_winds".to_string(),
        ],
    )
    .unwrap();
    let request = sample_direct_request(ModelId::Hrrr);
    let groups = group_direct_fetches(&request, &planned);

    assert!(groups.iter().any(|group| group.product == "nat"));
    assert!(groups.iter().any(|group| group.product == "sfc"));

    let smoke_group = groups
        .iter()
        .find(|group| {
            group.selectors.contains(&FieldSelector::height_agl(
                CanonicalField::SmokeMassDensity,
                8,
            ))
        })
        .expect("expected a dedicated smoke wrfnat group");
    assert_eq!(smoke_group.product, "nat");
    assert_eq!(
        smoke_group.planned_family_aliases,
        std::collections::BTreeSet::from(["nat".to_string()])
    );
}

#[test]
fn direct_fetch_timing_keeps_planned_vs_actual_family_truth() {
    let request = sample_direct_request(ModelId::Hrrr);
    let latest = LatestRun {
        model: ModelId::Hrrr,
        cycle: rustwx_core::CycleSpec::new("20260414", 23).unwrap(),
        source: SourceId::Nomads,
    };
    let planned_product = "nat";
    let group = FetchGroup {
        product: canonical_fetch_product(&request, planned_product),
        fetch_mode: PlotRecipeFetchMode::WholeFileStructuredExtract,
        variable_patterns: Vec::new(),
        selectors: vec![FieldSelector::entire_atmosphere(
            CanonicalField::CompositeReflectivity,
        )],
        planned_family_aliases: std::collections::BTreeSet::from([planned_product.to_string()]),
    };
    let fetch = build_direct_fetch_request(&request, &latest, 6, &group).unwrap();
    let runtime = HrrrDirectFetchRuntimeInfo {
        fetch_key: crate::publication::fetch_key(planned_product, &fetch.request),
        planned_product: planned_product.into(),
        fetched_product: fetch.request.product.clone(),
        planned_family_aliases: vec![planned_product.into()],
        requested_source: fetch.source_override.unwrap(),
        resolved_source: SourceId::Nomads,
        resolved_url: "https://example.test/hrrr.t23z.wrfsfcf06.grib2".into(),
    };
    assert_eq!(runtime.planned_product, "nat");
    assert_eq!(runtime.fetched_product, "sfc");
    assert_eq!(runtime.planned_family_aliases, vec!["nat".to_string()]);
    assert_eq!(runtime.resolved_source, SourceId::Nomads);
    assert!(runtime.resolved_url.contains("wrfsfc"));
}

#[test]
fn nomads_hrrr_direct_fetch_requests_use_full_grib_files() {
    let request = sample_direct_request(ModelId::Hrrr);
    let latest = LatestRun {
        model: ModelId::Hrrr,
        cycle: rustwx_core::CycleSpec::new("20260414", 23).unwrap(),
        source: SourceId::Nomads,
    };
    let planned = plan_direct_recipes(
        ModelId::Hrrr,
        &[
            "500mb_temperature_height_winds".to_string(),
            "2m_temperature_10m_winds".to_string(),
            "composite_reflectivity".to_string(),
        ],
    )
    .unwrap();
    let groups = group_direct_fetches(&request, &planned);
    assert_eq!(groups.len(), 2);

    for group in &groups {
        let fetch = build_direct_fetch_request(&request, &latest, 6, group).unwrap();
        assert_eq!(
            group.fetch_mode,
            PlotRecipeFetchMode::WholeFileStructuredExtract
        );
        assert!(
            fetch.variable_patterns.is_empty(),
            "NOMADS production direct fetches should not carry .idx subset patterns"
        );
    }
}

#[test]
fn aws_hrrr_direct_fetch_requests_keep_idx_patterns_for_fallback() {
    let request = sample_direct_request(ModelId::Hrrr);
    let latest = LatestRun {
        model: ModelId::Hrrr,
        cycle: rustwx_core::CycleSpec::new("20260414", 23).unwrap(),
        source: SourceId::Aws,
    };
    let planned = plan_direct_recipes(
        ModelId::Hrrr,
        &[
            "500mb_temperature_height_winds".to_string(),
            "2m_temperature_10m_winds".to_string(),
        ],
    )
    .unwrap();
    let groups = group_direct_fetches(&request, &planned);
    let prs_group = groups
        .iter()
        .find(|group| group.product == "prs")
        .expect("expected a pressure fetch group");
    assert!(!prs_group.variable_patterns.is_empty());

    let fetch = build_direct_fetch_request(&request, &latest, 6, prs_group).unwrap();
    assert_eq!(fetch.source_override, Some(SourceId::Aws));
    assert_eq!(fetch.variable_patterns, prs_group.variable_patterns);
}

#[test]
fn direct_execution_plan_strips_group_subset_patterns_for_nomads() {
    let planned = plan_direct_recipes(
        ModelId::Hrrr,
        &[
            "500mb_temperature_height_winds".to_string(),
            "700mb_temperature_height_winds".to_string(),
        ],
    )
    .unwrap();
    let request = sample_direct_request(ModelId::Hrrr);
    let groups = group_direct_fetches(&request, &planned);
    let latest = LatestRun {
        model: ModelId::Hrrr,
        cycle: rustwx_core::CycleSpec::new("20260414", 23).unwrap(),
        source: SourceId::Nomads,
    };
    let plan = build_direct_execution_plan(&latest, 6, &groups);
    let prs_bundle = plan
        .bundles
        .iter()
        .find(|bundle| bundle.id.native_product == "prs")
        .expect("expected a planned HRRR pressure bundle");
    let patterns = prs_bundle
        .aliases
        .iter()
        .flat_map(|alias| alias.variable_patterns.iter())
        .collect::<Vec<_>>();
    assert!(
        patterns.is_empty(),
        "NOMADS production execution should use full GRIB files without .idx subset patterns"
    );
}

#[test]
fn direct_execution_plan_keeps_group_subset_patterns_for_aws_fallback() {
    let planned = plan_direct_recipes(
        ModelId::Hrrr,
        &[
            "500mb_temperature_height_winds".to_string(),
            "700mb_temperature_height_winds".to_string(),
        ],
    )
    .unwrap();
    let request = sample_direct_request(ModelId::Hrrr);
    let groups = group_direct_fetches(&request, &planned);
    let latest = LatestRun {
        model: ModelId::Hrrr,
        cycle: rustwx_core::CycleSpec::new("20260414", 23).unwrap(),
        source: SourceId::Aws,
    };
    let plan = build_direct_execution_plan(&latest, 6, &groups);
    let prs_bundle = plan
        .bundles
        .iter()
        .find(|bundle| bundle.id.native_product == "prs")
        .expect("expected a planned HRRR pressure bundle");
    let patterns = prs_bundle
        .aliases
        .iter()
        .flat_map(|alias| alias.variable_patterns.iter())
        .collect::<Vec<_>>();
    assert!(patterns.iter().any(|pattern| pattern.contains("500 mb")));
    assert!(patterns.iter().any(|pattern| pattern.contains("700 mb")));
}

#[test]
fn grouping_splits_prs_and_nat_recipes() {
    let planned = plan_direct_recipes(
        ModelId::Hrrr,
        &[
            "500mb_temperature_height_winds".to_string(),
            "composite_reflectivity".to_string(),
        ],
    )
    .unwrap();
    let request = sample_direct_request(ModelId::Hrrr);
    let groups = group_direct_fetches(&request, &planned);
    assert_eq!(groups.len(), 2);
    assert!(groups.iter().any(|group| group.product == "prs"));
    assert!(groups.iter().any(|group| group.product == "sfc"));
}

#[test]
fn planning_supports_hrrr_direct_composite_layout_recipes() {
    let planned = plan_direct_recipes(
        ModelId::Hrrr,
        &[
            "cloud_cover_levels".to_string(),
            "precipitation_type".to_string(),
        ],
    )
    .unwrap();
    assert_eq!(planned.len(), 2);

    let request = sample_direct_request(ModelId::Hrrr);
    let groups = group_direct_fetches(&request, &planned);
    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0].product, "sfc");
    assert!(
        groups[0]
            .selectors
            .contains(&FieldSelector::entire_atmosphere(
                CanonicalField::LowCloudCover
            ))
    );
    assert!(
        groups[0]
            .selectors
            .contains(&FieldSelector::surface(CanonicalField::CategoricalSnow))
    );
}

#[test]
fn nbm_pop_direct_recipe_groups_to_core_surface_product() {
    let planned =
        plan_direct_recipes(ModelId::Nbm, &["probability_of_precipitation".to_string()]).unwrap();
    let request = sample_direct_request(ModelId::Nbm);
    let groups = group_direct_fetches(&request, &planned);
    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0].product, "core/co");
    assert!(groups[0].selectors.contains(&FieldSelector::surface(
        CanonicalField::ProbabilityOfPrecipitation
    )));
}

#[test]
fn nbm_qmd_direct_recipes_are_explicit_only_for_all_supported() {
    let supported = supported_direct_recipe_slugs(ModelId::Nbm);
    assert!(!supported.iter().any(|slug| slug.starts_with("nbm_qmd_")));
    let sref_supported = supported_direct_recipe_slugs(ModelId::Sref);
    assert!(
        !sref_supported
            .iter()
            .any(|slug| slug.starts_with("sref_prob_"))
    );
    let gefs_supported = supported_direct_recipe_slugs(ModelId::Gefs);
    assert!(
        !gefs_supported
            .iter()
            .any(|slug| slug.starts_with("gefs_avg_") || slug.starts_with("gefs_spr_"))
    );
    let aigefs_supported = supported_direct_recipe_slugs(ModelId::Aigefs);
    assert!(
        !aigefs_supported
            .iter()
            .any(|slug| slug.starts_with("aigefs_spr_"))
    );
    let hgefs_supported = supported_direct_recipe_slugs(ModelId::Hgefs);
    assert!(
        !hgefs_supported
            .iter()
            .any(|slug| slug.starts_with("hgefs_spr_"))
    );
    let href_supported = supported_direct_recipe_slugs(ModelId::Href);
    assert!(!href_supported.iter().any(|slug| {
        slug.starts_with("href_sprd_")
            || slug.starts_with("href_prob_")
            || slug.starts_with("href_mean_")
    }));
    let refs_supported = supported_direct_recipe_slugs(ModelId::Refs);
    assert!(
        !refs_supported
            .iter()
            .any(|slug| { slug.starts_with("refs_sprd_") || slug.starts_with("refs_prob_") })
    );

    let planned =
        plan_direct_recipes(ModelId::Nbm, &["nbm_qmd_2m_temperature_p50".to_string()]).unwrap();
    assert_eq!(planned[0].plan.product, "qmd/co");

    let sref_planned = plan_direct_recipes(
        ModelId::Sref,
        &["sref_prob_2m_temperature_below_273k".to_string()],
    )
    .unwrap();
    assert_eq!(sref_planned[0].plan.product, "ensprod/pgrb212/prob_3hrly");

    let gefs_planned = plan_direct_recipes(
        ModelId::Gefs,
        &["gefs_spr_2m_temperature_stddev".to_string()],
    )
    .unwrap();
    assert_eq!(gefs_planned[0].plan.product, "pgrb2ap5/gespr");

    let aigefs_planned = plan_direct_recipes(
        ModelId::Aigefs,
        &["aigefs_spr_2m_temperature_stddev".to_string()],
    )
    .unwrap();
    assert_eq!(aigefs_planned[0].plan.product, "sfc/spr");

    let hgefs_planned = plan_direct_recipes(
        ModelId::Hgefs,
        &["hgefs_spr_2m_temperature_stddev".to_string()],
    )
    .unwrap();
    assert_eq!(hgefs_planned[0].plan.product, "sfc/spr");

    let href_planned =
        plan_direct_recipes(ModelId::Href, &["href_sprd_2m_temperature".to_string()]).unwrap();
    assert_eq!(href_planned[0].plan.product, "ensprod/conus/sprd");

    let href_prob_planned = plan_direct_recipes(
        ModelId::Href,
        &["href_prob_2m_temperature_below_273p15k".to_string()],
    )
    .unwrap();
    assert_eq!(href_prob_planned[0].plan.product, "ensprod/conus/prob");

    let href_mean_planned =
        plan_direct_recipes(ModelId::Href, &["href_mean_2m_temperature".to_string()]).unwrap();
    assert_eq!(href_mean_planned[0].plan.product, "ensprod/conus/mean");

    let refs_spread_planned =
        plan_direct_recipes(ModelId::Refs, &["refs_sprd_2m_temperature".to_string()]).unwrap();
    assert_eq!(refs_spread_planned[0].plan.product, "sprd-conus");

    let refs_prob_planned = plan_direct_recipes(
        ModelId::Refs,
        &["refs_prob_2m_temperature_below_273p15k".to_string()],
    )
    .unwrap();
    assert_eq!(refs_prob_planned[0].plan.product, "prob-conus");
}

#[test]
fn unsupported_recipe_error_stays_explicit() {
    let err = plan_direct_recipes(ModelId::Hrrr, &["1h_qpf".to_string()])
        .unwrap_err()
        .to_string();
    assert!(err.contains("windowed lane") || err.contains("not supported"));
}

#[test]
fn gfs_direct_fetches_are_now_whole_file() {
    let planned = plan_direct_recipes(
        ModelId::Gfs,
        &["500mb_temperature_height_winds".to_string()],
    )
    .unwrap();
    let request = sample_direct_request(ModelId::Gfs);
    let groups = group_direct_fetches(&request, &planned);
    assert_eq!(groups.len(), 1);
    assert_eq!(
        groups[0].fetch_mode,
        PlotRecipeFetchMode::WholeFileStructuredExtract
    );
    let request = sample_direct_request(ModelId::Gfs);
    let latest = LatestRun {
        model: ModelId::Gfs,
        cycle: rustwx_core::CycleSpec::new("20260414", 18).unwrap(),
        source: SourceId::Nomads,
    };
    let fetch = build_direct_fetch_request(&request, &latest, 6, &groups[0]).unwrap();
    assert_eq!(fetch.request.product, "pgrb2.0p25");
    assert!(fetch.variable_patterns.is_empty());
}

#[test]
fn rrfs_direct_product_overrides_can_select_na_family() {
    let mut request = sample_direct_request(ModelId::RrfsA);
    request
        .product_overrides
        .insert("prs-conus".to_string(), "prs-na".to_string());
    let latest = LatestRun {
        model: ModelId::RrfsA,
        cycle: rustwx_core::CycleSpec::new("20260414", 20).unwrap(),
        source: SourceId::Aws,
    };
    let group = FetchGroup {
        product: canonical_fetch_product(&request, "prs-conus"),
        fetch_mode: PlotRecipeFetchMode::WholeFileStructuredExtract,
        variable_patterns: Vec::new(),
        selectors: vec![FieldSelector::isobaric(CanonicalField::Temperature, 500)],
        planned_family_aliases: std::collections::BTreeSet::from(["prs-conus".to_string()]),
    };
    let fetch = build_direct_fetch_request(&request, &latest, 2, &group).unwrap();
    assert_eq!(fetch.request.product, "prs-na");
}

#[test]
fn convert_filled_field_applies_operational_unit_transforms() {
    let pressure_recipe = plot_recipe("mslp_10m_winds").unwrap();
    let pressure_field = sample_selected_field(
        FieldSelector::mean_sea_level(CanonicalField::PressureReducedToMeanSeaLevel),
        "Pa",
        vec![100000.0; 4],
    );
    let converted_pressure = convert_filled_field(pressure_recipe, &pressure_field);
    assert_eq!(converted_pressure.units, "hPa");
    assert_eq!(converted_pressure.values[0], 1000.0);

    let pwat_recipe = plot_recipe("precipitable_water").unwrap();
    let pwat_field = sample_selected_field(
        FieldSelector::entire_atmosphere(CanonicalField::PrecipitableWater),
        "kg/m^2",
        vec![25.4; 4],
    );
    let converted_pwat = convert_filled_field(pwat_recipe, &pwat_field);
    assert_eq!(converted_pwat.units, "in");
    assert!((converted_pwat.values[0] - 1.0).abs() < 1.0e-6);

    let vis_recipe = plot_recipe("visibility").unwrap();
    let vis_field = sample_selected_field(
        FieldSelector::surface(CanonicalField::Visibility),
        "m",
        vec![1609.344; 4],
    );
    let converted_vis = convert_filled_field(vis_recipe, &vis_field);
    assert_eq!(converted_vis.units, "m");
    assert_eq!(super::rendering::direct_recipe_render_controls(vis_recipe, vis_field.selector), (Some(rustwx_render::LegendMode::Thresholds), Some(200.0)));
    assert!((converted_vis.values[0] - 1609.344).abs() < 1.0e-4);

    let vort_recipe = plot_recipe("500mb_absolute_vorticity_height_winds").unwrap();
    let vort_field = sample_selected_field(
        FieldSelector::isobaric(CanonicalField::AbsoluteVorticity, 500),
        "s^-1",
        vec![0.0002; 4],
    );
    let converted_vort = convert_filled_field(vort_recipe, &vort_field);
    assert_eq!(converted_vort.units, "10^-5 s^-1");
    assert!((converted_vort.values[0] - 20.0).abs() < 1.0e-6);

    let temp_recipe = plot_recipe("2m_temperature").unwrap();
    let temp_field = sample_selected_field(
        FieldSelector::height_agl(CanonicalField::Temperature, 2),
        "K",
        vec![273.15; 4],
    );
    let converted_temp = convert_filled_field(temp_recipe, &temp_field);
    assert_eq!(converted_temp.units, "degF");
    assert!((converted_temp.values[0] - 32.0).abs() < 1.0e-5);

    let upper_temp_recipe = plot_recipe("500mb_temperature_height_winds").unwrap();
    let upper_temp_field = sample_selected_field(
        FieldSelector::isobaric(CanonicalField::Temperature, 500),
        "K",
        vec![253.15; 4],
    );
    let converted_upper_temp = convert_filled_field(upper_temp_recipe, &upper_temp_field);
    assert_eq!(converted_upper_temp.units, "degC");
    assert!((converted_upper_temp.values[0] + 20.0).abs() < 1.0e-5);

    let wind_speed_recipe = plot_recipe("nbm_qmd_10m_wind_speed_p50").unwrap();
    let wind_speed_field = sample_selected_field(
        FieldSelector::height_agl(CanonicalField::WindSpeed, 10).with_percentile(50),
        "m/s",
        vec![10.0; 4],
    );
    let converted_wind_speed = convert_filled_field(wind_speed_recipe, &wind_speed_field);
    assert_eq!(converted_wind_speed.units, "kt");
    assert!((converted_wind_speed.values[0] - 19.438_445).abs() < 1.0e-5);
}

#[test]
fn overlay_only_rule_only_catches_height_products() {
    assert!(should_render_overlay_only(
        FieldSelector::isobaric(CanonicalField::GeopotentialHeight, 500),
        false
    ));
    assert!(!should_render_overlay_only(
        FieldSelector::mean_sea_level(CanonicalField::PressureReducedToMeanSeaLevel),
        false
    ));
    assert!(!should_render_overlay_only(
        FieldSelector::isobaric(CanonicalField::Temperature, 500),
        true
    ));
    assert!(!should_render_overlay_only(
        FieldSelector::surface(CanonicalField::Visibility),
        false
    ));
    // The height of an isotherm is a scalar map with a fill of its own,
    // not a contour analysis: overlay-only it was drawn as decametre
    // contours on a blank map.
    assert!(!should_render_overlay_only(
        FieldSelector::isotherm_celsius(CanonicalField::GeopotentialHeight, 0),
        false
    ));
    assert!(!should_render_overlay_only(
        FieldSelector::surface(CanonicalField::GeopotentialHeight),
        false
    ));
}

#[test]
fn direct_synoptic_contours_use_operational_emphasis() {
    let pressure = crate::plot_design::operational_contour_layer_for_values(
        FieldSelector::mean_sea_level(CanonicalField::PressureReducedToMeanSeaLevel),
        &[100000.0, 100200.0, 100400.0, 100600.0],
    )
    .expect("pressure contour layer");
    assert_eq!(pressure.levels.first().copied(), Some(960.0));
    assert_eq!(pressure.width, 1);
    assert_eq!(pressure.major_every, Some(2));
    assert_eq!(pressure.major_width, Some(2));
    assert!(pressure.labels);
    assert!(pressure.show_extrema);
    assert_eq!(pressure.pattern, rustwx_render::ContourLinePattern::Solid);
    assert_eq!(pressure.data[0], 1000.0);

    let height = crate::plot_design::operational_contour_layer_for_values(
        FieldSelector::isobaric(CanonicalField::GeopotentialHeight, 500),
        &[5400.0, 5460.0, 5520.0, 5580.0],
    )
    .expect("height contour layer");
    assert_eq!(height.major_every, Some(2));
    assert_eq!(height.major_width, Some(2));
    assert!(height.labels);
    assert!(!height.show_extrema);
    assert_eq!(height.data[0], 540.0);
}

#[test]
fn qmd_stddev_temperature_keeps_spread_units_and_scale() {
    let recipe = plot_recipe("nbm_qmd_2m_temperature_stddev").unwrap();
    let field = sample_selected_field(
        FieldSelector::height_agl(CanonicalField::Temperature, 2)
            .with_ensemble_standard_deviation(),
        "K",
        vec![0.0, 1.5, 3.0, 6.0],
    );
    let converted = convert_filled_field(recipe, &field);
    assert_eq!(converted.units, "K");
    assert_eq!(converted.values, vec![0.0, 1.5, 3.0, 6.0]);

    let ColorScale::Discrete(scale) =
        scale_for_filled_selector(recipe, field.selector, &converted.values)
    else {
        panic!("expected discrete spread scale");
    };
    assert_eq!(scale.levels.first().copied(), Some(0.0));
    assert!(scale.levels.last().copied().unwrap_or(0.0) >= 6.0);
    assert_eq!(scale.extend, ExtendMode::Max);
}

#[test]
fn weather_uh_scale_uses_operational_levels_and_masks_negative_noise() {
    let recipe = plot_recipe("uh_2to5km").unwrap();
    let scale = scale_for_recipe(
        recipe,
        FieldSelector::height_layer_agl(CanonicalField::UpdraftHelicity, 2000, 5000),
    );
    let ColorScale::Discrete(discrete) = scale else {
        panic!("expected discrete UH scale");
    };
    assert_eq!(discrete.levels.first().copied(), Some(0.0));
    assert_eq!(discrete.levels.last().copied(), Some(400.0));
    assert_eq!(discrete.mask_below, Some(0.0));
}

#[test]
fn reflectivity_scale_masks_no_return_values() {
    let recipe = plot_recipe("composite_reflectivity").unwrap();
    let scale = scale_for_recipe(
        recipe,
        FieldSelector::surface(CanonicalField::CompositeReflectivity),
    );
    let ColorScale::Discrete(discrete) = scale else {
        panic!("expected discrete reflectivity scale");
    };
    assert_eq!(discrete.levels.first().copied(), Some(10.0));
    // Re-recorded 2026-10-03: the reflectivity products moved to the
    // radar reflectivity table, which runs to 85 dBZ (was 70).
    assert_eq!(discrete.levels.last().copied(), Some(85.0));
    assert_eq!(discrete.extend, ExtendMode::Max);
    assert_eq!(discrete.mask_below, Some(10.0));
}

#[test]
fn categorical_precip_scale_masks_false_flags() {
    let recipe = plot_recipe("categorical_snow").unwrap();
    let scale = scale_for_recipe(
        recipe,
        FieldSelector::surface(CanonicalField::CategoricalSnow),
    );
    let ColorScale::Discrete(discrete) = scale else {
        panic!("expected discrete categorical scale");
    };
    // Re-recorded 2026-10-22 (fix/render-legends): one band per code (0, 1)
    // for the category legend; false flags stay masked.
    assert_eq!(discrete.levels, vec![-0.5, 0.5, 1.5]);
    assert_eq!(discrete.extend, ExtendMode::Neither);
    assert_eq!(discrete.mask_below, Some(0.5));
}

#[test]
fn height_winds_fill_uses_derived_wind_speed_in_knots() {
    let recipe = plot_recipe("500mb_height_winds").unwrap();
    let filled = sample_selected_field(
        FieldSelector::isobaric(CanonicalField::GeopotentialHeight, 500),
        "gpm",
        vec![540.0, 543.0, 546.0, 549.0],
    );
    let u = sample_selected_field(
        FieldSelector::isobaric(CanonicalField::UWind, 500),
        "m/s",
        vec![10.0, 0.0, 3.0, 4.0],
    );
    let v = sample_selected_field(
        FieldSelector::isobaric(CanonicalField::VWind, 500),
        "m/s",
        vec![0.0, 10.0, 4.0, 3.0],
    );
    let mut extracted = HashMap::new();
    extracted.insert(filled.selector, filled.clone());
    extracted.insert(u.selector, u);
    extracted.insert(v.selector, v);

    let render_field = render_filled_field(recipe, &filled, &extracted).unwrap();

    assert_eq!(render_field.units, "kt");
    assert_eq!(
        render_field.product.as_named(),
        Some("500mb_height_winds_wind_speed")
    );
    assert!((render_field.values[0] - 19.438_445).abs() < 0.01);
    assert!((render_field.values[1] - 19.438_445).abs() < 0.01);
    assert!((render_field.values[2] - 9.719_223).abs() < 0.01);
    assert!((render_field.values[3] - 9.719_223).abs() < 0.01);
}

/// The standalone wind chart colours its own field, in knots, from the
/// stored 10 m speed plane -- not from a companion substitution and not
/// left in raw m/s against a knot-calibrated ramp.
#[test]
fn standalone_10m_wind_fill_is_the_stored_speed_in_knots() {
    let recipe = plot_recipe("10m_wind_speed_and_direction").unwrap();
    let filled = sample_selected_field(
        FieldSelector::height_agl(CanonicalField::WindSpeed, 10),
        "m/s",
        vec![0.0, 2.5, 10.0, 25.0],
    );
    let u = sample_selected_field(
        FieldSelector::height_agl(CanonicalField::UWind, 10),
        "m/s",
        vec![0.0, 2.5, 10.0, 25.0],
    );
    let v = sample_selected_field(
        FieldSelector::height_agl(CanonicalField::VWind, 10),
        "m/s",
        vec![0.0, 0.0, 0.0, 0.0],
    );
    let mut extracted = HashMap::new();
    extracted.insert(filled.selector, filled.clone());
    extracted.insert(u.selector, u);
    extracted.insert(v.selector, v);

    let render_field = render_filled_field(recipe, &filled, &extracted).unwrap();

    assert_eq!(render_field.units, "kt");
    assert!((render_field.values[0] - 0.0).abs() < 0.01);
    assert!((render_field.values[1] - 4.859_611).abs() < 0.01);
    assert!((render_field.values[2] - 19.438_445).abs() < 0.01);
    assert!((render_field.values[3] - 48.596_11).abs() < 0.01);
}

/// A calm morning must still draw a map.
///
/// The MSLP overlay's wind ramp masks everything under 10 kt, which is
/// right for a pressure chart and catastrophic for a chart whose only
/// subject is the wind: on a light-wind day it renders blank.
#[test]
fn standalone_10m_wind_scale_masks_nothing_and_starts_at_calm() {
    let recipe = plot_recipe("10m_wind_speed_and_direction").unwrap();
    let scale = scale_for_recipe(
        recipe,
        FieldSelector::height_agl(CanonicalField::WindSpeed, 10),
    );
    let ColorScale::Discrete(discrete) = scale else {
        panic!("expected a discrete wind speed scale");
    };
    assert_eq!(discrete.mask_below, None, "no calm cutout on a wind chart");
    assert_eq!(discrete.levels.first().copied(), Some(0.0));
    assert_eq!(discrete.extend, ExtendMode::Max);

    // The overlay it must NOT share a ramp with.
    let overlay = plot_recipe("mslp_10m_winds").unwrap();
    let ColorScale::Discrete(overlay_scale) = scale_for_recipe(
        overlay,
        FieldSelector::mean_sea_level(CanonicalField::PressureReducedToMeanSeaLevel),
    ) else {
        panic!("expected a discrete scale");
    };
    assert_eq!(overlay_scale.mask_below, Some(10.0));
}

#[test]
fn aifs_inference_direct_policy_uses_interpolated_raster_sampling() {
    let field = sample_selected_field(
        FieldSelector::height_agl(CanonicalField::Temperature, 2),
        "K",
        vec![290.0; 4],
    )
    .into_field2d();
    let mut request = MapRenderRequest::contour_only(field.into());
    assert_eq!(request.raster_sample_mode, RasterSampleMode::Linear);

    apply_source_raster_policy(SourceId::AifsInference, &mut request);

    assert_eq!(request.raster_sample_mode, RasterSampleMode::Linear);
}

#[test]
fn full_domain_regular_grid_inverse_matches_its_native_frame() {
    let lat = vec![22.,22.,22.,26.5,26.5,26.5,31.,31.,31.];
    let lon = vec![-87.,-82.,-77.,-87.,-82.,-77.,-87.,-82.,-77.];
    let bounds = (-87.,-77.,22.,31.);
    let map = build_full_domain_projected_map_with_projection(&lat,&lon,None,bounds,1.4).unwrap();
    let resolved = full_domain_panel_resolved_projection(&lat,&lon,None,bounds,1.4).unwrap();
    let inverse = map.inverse_raster_projection.unwrap();
    assert_eq!(inverse.projection,resolved.spec());
    match resolved {
        rustwx_render::ResolvedProjection::LambertConformal { reference_latitude_deg, .. } =>
            assert_eq!(inverse.reference_latitude_deg,Some(reference_latitude_deg)),
        rustwx_render::ResolvedProjection::Geographic { central_meridian_deg } =>
            assert_eq!(inverse.reference_longitude_deg,Some(central_meridian_deg)),
        _ => {}
    }
    for i in 0..lat.len() {
        let (x,y) = resolved.project(lat[i] as f64,lon[i] as f64);
        assert!((x-map.projected_x[i]).abs()<1e-6 && (y-map.projected_y[i]).abs()<1e-6);
    }
}
