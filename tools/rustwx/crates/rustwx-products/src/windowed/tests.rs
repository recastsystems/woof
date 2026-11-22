use super::*;
use crate::shared_context::TitleProvenance;
use rustwx_render::ChromeScale;

#[test]
fn plan_windowed_products_blocks_short_forecast_hours() {
    let (planned, blockers, surface_hours, nat_hours, wind_hours, temp_hours) =
        plan_windowed_products(
            &[HrrrWindowedProduct::Qpf24h, HrrrWindowedProduct::Uh25km3h],
            2,
            Some(0),
        );
    assert!(planned.is_empty());
    assert_eq!(blockers.len(), 2);
    assert!(surface_hours.is_empty());
    assert!(nat_hours.is_empty());
    assert!(wind_hours.is_empty());
    assert!(temp_hours.is_empty());
}

/// A blocker is a fact about the frame's hour and the window, the same for
/// every model.  It used to advise "use a HRRR extended cycle" on every
/// model's short frame and blame "HRRR APCP windows" for a 1 h QPF at F000.
#[test]
fn a_short_hour_blocker_names_the_hour_and_no_model() {
    let (_planned, blockers, ..) = plan_windowed_products(
        &[
            HrrrWindowedProduct::Qpf1h,
            HrrrWindowedProduct::Wind10m0to24hMax,
            HrrrWindowedProduct::Temp2m0to24hMax,
        ],
        0,
        Some(3),
    );
    assert_eq!(blockers.len(), 3, "{blockers:?}");
    for blocker in &blockers {
        assert!(blocker.reason.contains("requires forecast hour >= "), "{blocker:?}");
        assert!(!blocker.reason.contains("HRRR"), "{blocker:?}");
    }
}

#[test]
fn qpf_hourly_fallback_is_limited_to_hourly_cadence_models() {
    assert!(qpf_hourly_fallback_supported(ModelId::Hrrr, 48));
    assert!(qpf_hourly_fallback_supported(ModelId::Rap, 51));
    assert!(qpf_hourly_fallback_supported(ModelId::Gfs, 120));
    assert!(!qpf_hourly_fallback_supported(ModelId::Gfs, 123));
    assert!(!qpf_hourly_fallback_supported(ModelId::Gefs, 6));
    assert!(!qpf_hourly_fallback_supported(ModelId::EcmwfOpenData, 6));
}

#[test]
fn plan_windowed_products_adds_wind_max_hours_for_any_extended_cycle() {
    let (planned, blockers, surface_hours, nat_hours, wind_hours, temp_hours) =
        plan_windowed_products(
            &[
                HrrrWindowedProduct::Wind10m1hMax,
                HrrrWindowedProduct::Wind10m0to24hMax,
                HrrrWindowedProduct::Wind10m24to48hMax,
                HrrrWindowedProduct::Wind10m0to48hMax,
            ],
            48,
            Some(0),
        );
    assert_eq!(planned.len(), 4);
    assert!(blockers.is_empty());
    assert!(surface_hours.is_empty());
    assert!(nat_hours.is_empty());
    assert!(temp_hours.is_empty());
    assert_eq!(wind_hours.first(), Some(&1));
    assert_eq!(wind_hours.last(), Some(&48));

    let (planned, blockers, _, _, wind_hours, temp_hours) =
        plan_windowed_products(&[HrrrWindowedProduct::Wind10m0to24hMax], 24, Some(18));
    assert_eq!(planned, vec![HrrrWindowedProduct::Wind10m0to24hMax]);
    assert!(blockers.is_empty());
    assert_eq!(wind_hours.first(), Some(&1));
    assert_eq!(wind_hours.last(), Some(&24));
    assert!(temp_hours.is_empty());
}

#[test]
fn plan_windowed_products_adds_diurnal_temperature_hours() {
    let (planned, blockers, surface_hours, nat_hours, wind_hours, temp_hours) =
        plan_windowed_products(
            &[
                HrrrWindowedProduct::Temp2m0to24hMax,
                HrrrWindowedProduct::Temp2m24to48hMin,
                HrrrWindowedProduct::Temp2m0to48hMax,
                HrrrWindowedProduct::Temp2m0to48hRange,
                HrrrWindowedProduct::Rh2m0to24hMin,
                HrrrWindowedProduct::Dewpoint2m24to48hMax,
                HrrrWindowedProduct::Vpd2m0to48hRange,
            ],
            48,
            Some(0),
        );
    assert_eq!(planned.len(), 7);
    assert!(blockers.is_empty());
    assert!(surface_hours.is_empty());
    assert!(nat_hours.is_empty());
    assert!(wind_hours.is_empty());
    assert_eq!(temp_hours.first(), Some(&1));
    assert_eq!(temp_hours.last(), Some(&48));

    let (planned, blockers, _, _, _, temp_hours) =
        plan_windowed_products(&[HrrrWindowedProduct::Temp2m0to24hMax], 24, Some(18));
    assert_eq!(planned, vec![HrrrWindowedProduct::Temp2m0to24hMax]);
    assert!(blockers.is_empty());
    assert_eq!(temp_hours.first(), Some(&1));
    assert_eq!(temp_hours.last(), Some(&24));
}

#[test]
fn windowed_fetch_truth_can_show_nat_planned_but_sfc_fetched() {
    let fetch = HrrrWindowedHourFetchInfo {
        hour: 1,
        planned_product: "nat".into(),
        fetched_product: "sfc".into(),
        requested_source: SourceId::Nomads,
        resolved_source: SourceId::Nomads,
        resolved_url: "https://example.test/hrrr.t23z.wrfsfcf01.grib2".into(),
        fetch_cache_hit: false,
        input_fetch: None,
    };
    assert_eq!(fetch.planned_product, "nat");
    assert_eq!(fetch.fetched_product, "sfc");
    assert_eq!(fetch.resolved_source, SourceId::Nomads);
    assert!(fetch.resolved_url.contains("wrfsfc"));
}

#[test]
fn windowed_render_request_uses_modern_map_chrome() {
    let shape = rustwx_core::GridShape::new(2, 2).unwrap();
    let grid = rustwx_core::LatLonGrid::new(
        shape,
        vec![36.0, 36.0, 35.0, 35.0],
        vec![-98.0, -97.0, -98.0, -97.0],
    )
    .unwrap();
    let field = rustwx_core::Field2D::new(
        rustwx_core::ProductKey::named("qpf_1h"),
        "in",
        grid,
        vec![0.0, 0.1, 0.2, 0.3],
    )
    .unwrap();
    let computed = crate::windowed_decoder::ComputedWindowedField {
        field,
        title: "1-h QPF".to_string(),
        metadata: HrrrWindowedProductMetadata {
            strategy: "test window".to_string(),
            contributing_forecast_hours: vec![1],
            window_hours: Some(1),
        },
        scale: rustwx_render::ColorScale::Discrete(crate::windowed_decoder::qpf_scale()),
    };
    let request = HrrrWindowedBatchRequest {
        model: ModelId::Hrrr,
        date_yyyymmdd: "20260424".to_string(),
        cycle_override_utc: Some(22),
        forecast_hour: 1,
        source: SourceId::Nomads,
        domain: DomainSpec::new("southern_plains", (-109.0, -90.0, 25.0, 40.5)),
        out_dir: PathBuf::new(),
        cache_root: PathBuf::new(),
        use_cache: false,
        products: vec![HrrrWindowedProduct::Qpf1h],
        output_width: 1200,
        output_height: 900,
        png_compression: PngCompressionMode::Default,
        place_label_overlay: None,
        subtitle_left_suffix: None,
        subtitle_right_override: None,
        title_provenance: TitleProvenance::default(),
    };
    let projected = ProjectedMap {
        projected_x: vec![0.0, 1.0, 0.0, 1.0],
        projected_y: vec![1.0, 1.0, 0.0, 0.0],
        extent: rustwx_render::ProjectedExtent {
            x_min: 0.0,
            x_max: 1.0,
            y_min: 0.0,
            y_max: 1.0,
        },
        lines: Vec::new(),
        polygons: Vec::new(),
        inverse_raster_projection: None,
    };

    let render_request = build_windowed_render_request(
        HrrrWindowedProduct::Qpf1h,
        &computed,
        &request,
        &projected,
        "20260424",
        22,
        1,
        ModelId::Hrrr,
        SourceId::Nomads,
    );

    assert_eq!(render_request.width, 1200);
    assert_eq!(render_request.height, 900);
    assert_eq!(render_request.chrome_scale, ChromeScale::Fixed(0.9));
    // A fetched batch: nothing is appended to the headline.
    assert_eq!(render_request.title.as_deref(), Some("1-h QPF"));
    assert_eq!(render_request.supersample_factor, 1);
    assert_eq!(
        render_request.subtitle_left.as_deref(),
        Some("Init 04/24 22Z | F001 | Valid 04/24 23Z | HRRR")
    );
    assert_eq!(
        render_request.subtitle_right.as_deref(),
        Some("source: nomads")
    );
    assert_eq!(
        render_request.visual_mode,
        ProductVisualMode::FilledMeteorology
    );
    // Rain draws one listed colour per band (`crate::qpf::qpf_inches_scale`
    // names the breakage the smooth ramp over the linear palette caused).
    assert_eq!(
        render_request.legend.mode,
        rustwx_render::LegendMode::Thresholds
    );
    assert!(render_request.domain_frame.is_some());
    assert!(render_request.projected_domain.is_some());
    // A Lambert grid has no inverse raster, and the request carries none.
    assert!(render_request.inverse_raster_projection.is_none());
}

/// A regular lat/lon grid's map carries an inverse-raster projection, and
/// the windowed panel must draw through it exactly as the direct lane does:
/// without it the field was drawn as forward-projected contour polygons,
/// which closed rain bands across the whole map at the date line and framed
/// a regional crop to the strip inscribed in its curved footprint.
#[test]
fn windowed_render_request_carries_the_maps_inverse_raster_projection() {
    let shape = rustwx_core::GridShape::new(2, 2).unwrap();
    let grid = rustwx_core::LatLonGrid::new(
        shape,
        vec![10.0, 10.0, 11.0, 11.0],
        vec![-100.0, -99.0, -100.0, -99.0],
    )
    .unwrap();
    let field = rustwx_core::Field2D::new(
        rustwx_core::ProductKey::named("qpf_6h"),
        "in",
        grid,
        vec![0.1, 0.2, 0.3, 0.4],
    )
    .unwrap();
    let computed = crate::windowed_decoder::ComputedWindowedField {
        field,
        title: "6-h QPF".to_string(),
        metadata: HrrrWindowedProductMetadata {
            strategy: "test window".to_string(),
            contributing_forecast_hours: vec![0, 6],
            window_hours: Some(6),
        },
        scale: rustwx_render::ColorScale::Discrete(crate::windowed_decoder::qpf_scale()),
    };
    let request = HrrrWindowedBatchRequest {
        model: ModelId::Gfs,
        date_yyyymmdd: "20260920".to_string(),
        cycle_override_utc: Some(0),
        forecast_hour: 6,
        source: SourceId::Nomads,
        domain: DomainSpec::new("global", (-180.0, 180.0, -90.0, 90.0)),
        out_dir: PathBuf::new(),
        cache_root: PathBuf::new(),
        use_cache: false,
        products: vec![HrrrWindowedProduct::Qpf6h],
        output_width: 1200,
        output_height: 700,
        png_compression: PngCompressionMode::Default,
        place_label_overlay: None,
        subtitle_left_suffix: None,
        subtitle_right_override: None,
        title_provenance: TitleProvenance::default(),
    };
    let inverse = rustwx_render::InverseRasterProjection {
        projection: rustwx_render::ProjectionSpec::Robinson {
            central_meridian_deg: 0.0,
        },
        reference_latitude_deg: None,
        reference_longitude_deg: None,
        clip_bounds: None,
    };
    let projected = ProjectedMap {
        projected_x: vec![0.0, 1.0, 0.0, 1.0],
        projected_y: vec![0.0, 0.0, 1.0, 1.0],
        extent: rustwx_render::ProjectedExtent {
            x_min: 0.0,
            x_max: 1.0,
            y_min: 0.0,
            y_max: 1.0,
        },
        lines: Vec::new(),
        polygons: Vec::new(),
        inverse_raster_projection: Some(inverse.clone()),
    };

    let render_request = build_windowed_render_request(
        HrrrWindowedProduct::Qpf6h,
        &computed,
        &request,
        &projected,
        "20260920",
        0,
        6,
        ModelId::Gfs,
        SourceId::Nomads,
    );

    assert_eq!(render_request.inverse_raster_projection, Some(inverse));
}

#[test]
fn a_locally_imported_windowed_title_names_its_grid() {
    let shape = rustwx_core::GridShape::new(2, 2).unwrap();
    let grid = rustwx_core::LatLonGrid::new(
        shape,
        vec![36.0, 36.0, 35.0, 35.0],
        vec![-98.0, -97.0, -98.0, -97.0],
    )
    .unwrap();
    let field = rustwx_core::Field2D::new(
        rustwx_core::ProductKey::named("10m_wind_1h_max"),
        "kt",
        grid,
        vec![10.0, 20.0, 30.0, 40.0],
    )
    .unwrap();
    let computed = crate::windowed_decoder::ComputedWindowedField {
        field,
        title: "10 m Wind Max (1 h)".to_string(),
        metadata: HrrrWindowedProductMetadata {
            strategy: "test window".to_string(),
            contributing_forecast_hours: vec![1],
            window_hours: Some(1),
        },
        scale: rustwx_render::ColorScale::Discrete(crate::windowed_decoder::qpf_scale()),
    };
    let request = HrrrWindowedBatchRequest {
        model: ModelId::WrfGdex,
        date_yyyymmdd: "20260801".to_string(),
        cycle_override_utc: Some(8),
        forecast_hour: 1,
        source: SourceId::Gdex,
        domain: DomainSpec::new("native_grid", (-104.0, -100.0, 40.0, 44.0)),
        out_dir: PathBuf::new(),
        cache_root: PathBuf::new(),
        use_cache: false,
        products: vec![HrrrWindowedProduct::Wind10m1hMax],
        output_width: 1200,
        output_height: 900,
        png_compression: PngCompressionMode::Default,
        place_label_overlay: None,
        subtitle_left_suffix: Some("\u{0394}x 750 m".to_string()),
        subtitle_right_override: Some("source: ArWen".to_string()),
        title_provenance: TitleProvenance::LocalImport {
            grid_label: Some("d02 750 m".to_string()),
        },
    };
    let projected = ProjectedMap {
        projected_x: vec![0.0, 1.0, 0.0, 1.0],
        projected_y: vec![1.0, 1.0, 0.0, 0.0],
        extent: rustwx_render::ProjectedExtent {
            x_min: 0.0,
            x_max: 1.0,
            y_min: 0.0,
            y_max: 1.0,
        },
        lines: Vec::new(),
        polygons: Vec::new(),
        inverse_raster_projection: None,
    };

    let render_request = build_windowed_render_request(
        HrrrWindowedProduct::Wind10m1hMax,
        &computed,
        &request,
        &projected,
        "20260801",
        8,
        1,
        ModelId::WrfGdex,
        SourceId::Gdex,
    );

    assert_eq!(
        render_request.title.as_deref(),
        Some("10 m Wind Max (1 h) (d02 750 m)")
    );
}

#[test]
fn windowed_render_request_labels_fixed_window_instead_of_requested_end_hour() {
    let shape = rustwx_core::GridShape::new(2, 2).unwrap();
    let grid = rustwx_core::LatLonGrid::new(
        shape,
        vec![36.0, 36.0, 35.0, 35.0],
        vec![-98.0, -97.0, -98.0, -97.0],
    )
    .unwrap();
    let field = rustwx_core::Field2D::new(
        rustwx_core::ProductKey::named("2m_rh_24_48h_range"),
        "%",
        grid,
        vec![10.0, 20.0, 30.0, 40.0],
    )
    .unwrap();
    let computed = crate::windowed_decoder::ComputedWindowedField {
        field,
        title: "2 m Relative Humidity Range (24-48 h)".to_string(),
        metadata: HrrrWindowedProductMetadata {
            strategy:
                "pointwise max-min range of hourly 2 m relative humidity snapshots across F025-F048"
                    .to_string(),
            contributing_forecast_hours: (25..=48).collect(),
            window_hours: Some(24),
        },
        scale: rustwx_render::ColorScale::Discrete(crate::windowed_decoder::rh2m_scale(true)),
    };
    let request = HrrrWindowedBatchRequest {
        model: ModelId::Hrrr,
        date_yyyymmdd: "20260424".to_string(),
        cycle_override_utc: Some(0),
        forecast_hour: 48,
        source: SourceId::Aws,
        domain: DomainSpec::new("california", (-124.9, -113.8, 31.9, 42.5)),
        out_dir: PathBuf::new(),
        cache_root: PathBuf::new(),
        use_cache: false,
        products: vec![HrrrWindowedProduct::Rh2m24to48hRange],
        output_width: 1200,
        output_height: 900,
        png_compression: PngCompressionMode::Default,
        place_label_overlay: None,
        subtitle_left_suffix: None,
        subtitle_right_override: None,
        title_provenance: TitleProvenance::default(),
    };
    let projected = ProjectedMap {
        projected_x: vec![0.0, 1.0, 0.0, 1.0],
        projected_y: vec![1.0, 1.0, 0.0, 0.0],
        extent: rustwx_render::ProjectedExtent {
            x_min: 0.0,
            x_max: 1.0,
            y_min: 0.0,
            y_max: 1.0,
        },
        lines: Vec::new(),
        polygons: Vec::new(),
        inverse_raster_projection: None,
    };

    let render_request = build_windowed_render_request(
        HrrrWindowedProduct::Rh2m24to48hRange,
        &computed,
        &request,
        &projected,
        "20260424",
        0,
        48,
        ModelId::Hrrr,
        SourceId::Aws,
    );

    assert_eq!(
        render_request.subtitle_left.as_deref(),
        Some("Init 04/24 00Z | F025-F048 | Valid 04/26 00Z | HRRR")
    );
    assert_eq!(
        render_request.subtitle_right.as_deref(),
        Some("source: aws")
    );
}

#[test]
fn from_slug_round_trips_every_supported_windowed_product() {
    for &product in HrrrWindowedProduct::supported_products() {
        assert_eq!(
            HrrrWindowedProduct::from_slug(product.slug()),
            Some(product),
            "slug '{}' must parse back to its product",
            product.slug()
        );
    }
    assert_eq!(HrrrWindowedProduct::from_slug("not_a_windowed_slug"), None);
}

/// Every 10 m wind maximum window has a snapshot-fold row, and no row
/// titles a snapshot as a maximum: the row is what an hourly history with no
/// stored wind maximum is drawn under.
#[test]
fn every_wind_maximum_window_names_its_hourly_snapshot_fold() {
    for &product in HrrrWindowedProduct::supported_products() {
        let row = product.snapshot_fold();
        assert_eq!(row.is_some(), product.is_wind10m(), "{}", product.slug());
        let Some(row) = row else { continue };
        assert_eq!(row.product, product);
        assert_ne!(row.title, product.title(), "{}", product.slug());
        assert!(row.title.contains("snapshot"), "{}", row.title);
        assert!(row.fold.contains("top-of-hour"), "{}", row.fold);
        assert!(row.why.contains("WSPD10MAX") && row.why.contains("not the maximum"));
        // Only the one-hour window, which folds one frame, cannot be
        // partly a stored maximum and partly a snapshot.
        match row.partial_title {
            None => assert_eq!(product, HrrrWindowedProduct::Wind10m1hMax),
            Some(partial) => {
                assert!(partial.contains("partly hourly snapshots"), "{partial}");
                assert_ne!(partial, row.title);
                assert_ne!(partial, product.title());
            }
        }
    }
    assert_eq!(
        SNAPSHOT_FOLD_ROWS.len(),
        SNAPSHOT_FOLD_ROWS
            .iter()
            .map(|row| row.product)
            .collect::<std::collections::HashSet<_>>()
            .len(),
        "one row per product"
    );
}

/// The store-render scale helper reproduces the per-family scales the GRIB
/// compute kernels attach: QPF/wind/snapshot products get the family's
/// discrete scale, UH products the Uh weather preset, for every supported
/// product, so store-rendered windowed PNGs style identically.
#[test]
fn windowed_product_scale_matches_the_kernel_families() {
    use rustwx_render::ColorScale;
    for &product in HrrrWindowedProduct::supported_products() {
        let scale = crate::windowed_decoder::windowed_product_scale(product);
        if product.is_qpf() {
            let ColorScale::Discrete(scale) = scale else {
                panic!("{}: QPF must use a discrete scale", product.slug());
            };
            assert_eq!(scale.levels, crate::windowed_decoder::qpf_scale().levels);
        } else if product.is_wind10m() {
            let ColorScale::Discrete(scale) = scale else {
                panic!("{}: wind must use a discrete scale", product.slug());
            };
            assert_eq!(
                scale.levels,
                crate::windowed_decoder::wind10m_scale().levels
            );
        } else if product.is_surface_snapshot() {
            assert!(
                matches!(scale, ColorScale::Discrete(_)),
                "{}: snapshot windows use discrete scales",
                product.slug()
            );
        } else {
            assert!(
                matches!(scale, ColorScale::Weather(_)),
                "{}: UH products use the Uh weather preset",
                product.slug()
            );
        }
    }
}

#[test]
fn full_longitude_envelope_keeps_windowed_grid_uncropped() {
    use rustwx_core::{GridShape, LatLonGrid};
    let bounds = (-180.05, 180.05, -22.0, 82.0);
    let grid = LatLonGrid::new(
        GridShape::new(3, 2).unwrap(),
        vec![20.0, 20.0, 20.0, 50.0, 50.0, 50.0],
        vec![-179.99, -92.0, 179.99, -179.99, -92.0, 179.99],
    ).unwrap();
    assert!(point_in_geographic_bounds(-92.0, 37.0, bounds));
    assert!(!point_in_geographic_bounds(-92.0, 85.0, bounds));
    assert!(crop_for_domain_grid(&grid, bounds, 0).unwrap().is_none());
    // A genuinely narrow date-line request must remain narrow.
    let narrow = (179.0, -179.0, -22.0, 82.0);
    assert!(point_in_geographic_bounds(179.99, 37.0, narrow));
    assert!(!point_in_geographic_bounds(-92.0, 37.0, narrow));
}
