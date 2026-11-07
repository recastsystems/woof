use super::*;
use image::ImageFormat;

fn sample_field(product: &str) -> Field2D {
    let shape = GridShape::new(4, 3).unwrap();
    let lat = vec![35.0; shape.len()];
    let lon = vec![-97.0; shape.len()];
    let grid = LatLonGrid::new(shape, lat, lon).unwrap();
    let values = vec![
        0.0, 250.0, 750.0, 1500.0, 2000.0, 2400.0, 2600.0, 2800.0, 3000.0, 3200.0, 3400.0, 3600.0,
    ];
    Field2D::new(ProductKey::named(product), "J/kg", grid, values).unwrap()
}

#[test]
fn colorbar_units_reach_the_image_without_changing_the_map() {
    for style in [StaticPlotStyle::OperationalBudget30s, StaticPlotStyle::CleanAtlas] {
        for units in ["degF", "dBZ", "kt", "kg m-2 s-1"] {
            let mut request = MapRenderRequest::for_weather_product(
                sample_field("sbecape"), WeatherProduct::Sbecape);
            request.width = 600;
            request.height = 450;
            request.field.units.clear();
            let render = |request: &MapRenderRequest| {
                with_render_state_with_style(request, style, |data, ny, nx, opts| {
                    Ok(render_to_image_profile(data, ny, nx, opts))
                }).unwrap()
            };
            let (bare, before) = render(&request);
            request.field.units = units.into();
            let (labelled, after) = render(&request);
            assert!(bare != labelled, "colorbar omitted {units} in {style:?}");
            let canvas = RenderPresentation::for_mode_with_style(
                request.visual_mode, style).canvas_background.to_image_rgba();
            for (old, new) in bare.pixels().zip(labelled.pixels()) {
                if old != new {
                    assert!(old == &canvas || old[3] == 0,
                            "unit label overwrote existing legend or header ink");
                }
            }
            assert_eq!((before.map_x, before.map_y, before.map_w, before.map_h),
                       (after.map_x, after.map_y, after.map_w, after.map_h));
            for y in before.map_y..before.map_y + before.map_h {
                for x in before.map_x..before.map_x + before.map_w {
                    assert_eq!(bare.get_pixel(x, y), labelled.get_pixel(x, y),
                               "unit annotation changed a map pixel");
                }
            }
        }
    }
}

#[test]
fn disabled_colorbar_does_not_draw_units() {
    let mut request = MapRenderRequest::for_weather_product(
        sample_field("sbecape"), WeatherProduct::Sbecape);
    request.width = 300;
    request.height = 240;
    request.colorbar = false;
    let before = render_image_with_style(&request, StaticPlotStyle::CleanAtlas).unwrap();
    request.field.units = "different units".into();
    let after = render_image_with_style(&request, StaticPlotStyle::CleanAtlas).unwrap();
    assert_eq!(before, after);
}

#[test]
fn weather_product_mapping_covers_ecape_and_severe_aliases() {
    assert_eq!(
        WeatherProduct::from_product_name("sbecape"),
        Some(WeatherProduct::Sbecape)
    );
    assert_eq!(
        WeatherProduct::from_product_name("mlecin"),
        Some(WeatherProduct::Mlecin)
    );
    assert_eq!(
        WeatherProduct::from_product_name("ecape_scp"),
        Some(WeatherProduct::EcapeScpExperimental)
    );
    assert_eq!(
        WeatherProduct::from_product_name("sb_ecape_derived_cape_ratio"),
        Some(WeatherProduct::SbEcapeDerivedCapeRatio)
    );
    assert_eq!(
        WeatherProduct::from_product_name("mu_ecape_native_cape_ratio"),
        Some(WeatherProduct::MuEcapeNativeCapeRatio)
    );
    assert_eq!(
        WeatherProduct::from_product_name("ecape_ehi"),
        Some(WeatherProduct::EcapeEhi01kmExperimental)
    );
    assert_eq!(
        WeatherProduct::from_product_name("ecape_ehi_0_3km"),
        Some(WeatherProduct::EcapeEhi03kmExperimental)
    );
}

fn sample_canvas_request(product: &str, title: &str, colorbar: bool) -> MapRenderRequest {
    MapRenderRequest {
        field: sample_field(product),
        rgba_grid: None,
        product_metadata: None,
        width: 320,
        height: 240,
        scale: ColorScale::Weather(crate::weather::WeatherPreset::Cape),
        background: Color::WHITE,
        colorbar,
        title: Some(title.into()),
        subtitle_left: None,
        subtitle_center: None,
        subtitle_right: None,
        cbar_tick_step: Some(500.0),
        render_density: RenderDensity::default(),
        legend: LegendControls::default(),
        chrome_scale: ChromeScale::default(),
        supersample_factor: 1,
        supersample_sharpen: true,
        visual_mode: ProductVisualMode::FilledMeteorology,
        raster_sample_mode: RasterSampleMode::default(),
        domain_frame: None,
        projected_domain: None,
        projected_polygons: Vec::new(),
        projected_data_polygons: Vec::new(),
        mesh_cells: None,
        inverse_raster_projection: None,
        resolved_projection: None,
        geographic_bounds: None,
        projected_place_labels: Vec::new(),
        projected_points: Vec::new(),
        projected_lines: Vec::new(),
        contours: Vec::new(),
        wind_barbs: Vec::new(),
        wind_streamlines: Vec::new(),
        semantics: None,
        empty_note: None,
    }
}

#[test]
fn render_png_emits_valid_nonempty_image() {
    let mut request = sample_canvas_request("sbecape", "SBECAPE", true);
    request.subtitle_left = Some("HRRR 2026-04-14 20Z F00".into());
    request.subtitle_center = Some("rustwx-render".into());
    request.subtitle_right = Some("rustwx-render".into());
    let png = render_png(&request).unwrap();
    assert!(png.starts_with(&[137, 80, 78, 71, 13, 10, 26, 10]));

    let image = image::load_from_memory_with_format(&png, ImageFormat::Png)
        .unwrap()
        .to_rgba8();
    assert_eq!(image.width(), 320);
    assert_eq!(image.height(), 240);

    let non_white = image
        .pixels()
        .filter(|px| px.0 != [255, 255, 255, 255])
        .count();
    assert!(non_white > 1000, "image should contain rendered content");
}

#[test]
fn save_png_writes_file() {
    let request = MapRenderRequest::for_weather_product(sample_field("scp"), WeatherProduct::Scp);

    let path = std::env::temp_dir().join(format!("rustwx-render-{}.png", std::process::id()));
    save_png(&request, &path).unwrap();

    let bytes = std::fs::read(&path).unwrap();
    assert!(bytes.starts_with(&[137, 80, 78, 71, 13, 10, 26, 10]));

    let _ = std::fs::remove_file(path);
}

#[test]
fn render_image_emits_rgba_canvas_without_png_decode_in_callers() {
    let request = sample_canvas_request("mucape", "MUCAPE", false);
    let image = render_image(&request).unwrap();
    assert_eq!(image.width(), 320);
    assert_eq!(image.height(), 240);

    let non_white = image
        .pixels()
        .filter(|px| px.0 != [255, 255, 255, 255])
        .count();
    assert!(non_white > 1000, "image should contain rendered content");
}

#[test]
fn with_render_state_carries_projected_place_labels_into_render_opts() {
    let mut request = MapRenderRequest::contour_only(sample_field("overlay"));
    request.projected_domain = Some(ProjectedDomain {
        x: vec![0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0],
        y: vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0],
        extent: ProjectedExtent {
            x_min: 0.0,
            x_max: 3.0,
            y_min: 0.0,
            y_max: 2.0,
        },
    });
    request.projected_place_labels.push(
        ProjectedPlaceLabel::new(1.5, 1.0)
            .with_label("Tulsa")
            .with_priority(ProjectedPlaceLabelPriority::Micro),
    );

    let carried = with_render_state(&request, |_data, _ny, _nx, opts| {
        Ok((
            opts.projected_place_labels.len(),
            opts.projected_place_labels[0].label.clone(),
            opts.projected_place_labels[0].style.marker_radius_px,
            opts.projected_place_labels[0].priority,
        ))
    })
    .unwrap();

    assert_eq!(carried.0, 1);
    assert_eq!(carried.1.as_deref(), Some("Tulsa"));
    assert_eq!(carried.2, 3);
    assert_eq!(carried.3, ProjectedPlaceLabelPriority::Micro);
}

#[test]
fn for_weather_product_sets_expected_titles_for_experimental_fields() {
    let request = MapRenderRequest::for_weather_product(
        sample_field("ecape_scp"),
        WeatherProduct::EcapeScpExperimental,
    );

    assert_eq!(request.title.as_deref(), Some("ECAPE SCP (EXP)"));
    assert_eq!(request.cbar_tick_step, Some(5.0));
    assert!(matches!(
        request.scale,
        ColorScale::Weather(WeatherPreset::Scp)
    ));
}

#[test]
fn derived_product_builder_renders_signed_field_with_builtin_scale() {
    let shape = GridShape::new(4, 3).unwrap();
    let lat = vec![35.0; shape.len()];
    let lon = vec![-97.0; shape.len()];
    let grid = LatLonGrid::new(shape, lat, lon).unwrap();
    let field = Field2D::new(
        ProductKey::named("temperature_advection_850mb"),
        "K/hr",
        grid,
        vec![
            -10.0, -8.0, -6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0,
        ],
    )
    .unwrap();

    let request = MapRenderRequest::for_derived_product(
        field,
        DerivedProductStyle::TemperatureAdvection850mb,
    );
    let image = render_image(&request).unwrap();

    let non_white = image
        .pixels()
        .filter(|px| px.0 != [255, 255, 255, 255])
        .count();
    assert!(non_white > 1000, "derived render should contain content");
}

/// A whole-earth mesh with one point per 10 degrees, plus the request
/// options a caller would build for it.  Robinson is pinned explicitly so
/// the test cannot drift with projection inference; the measured failure
/// this whole feature answers was on a Robinson panel.
fn global_robinson_mesh() -> (Vec<f32>, Vec<f32>, ProjectedMapBuildOptions) {
    let mut lat = Vec::new();
    let mut lon = Vec::new();
    for row in 0..18 {
        for col in 0..36 {
            lat.push(-85.0 + row as f32 * 10.0);
            lon.push(-175.0 + col as f32 * 10.0);
        }
    }
    let options = ProjectedMapBuildOptions::from_bounds((-180.0, 180.0, -90.0, 90.0), 1.6)
        .with_projection(ProjectionSpec::Robinson {
            central_meridian_deg: 0.0,
        })
        .without_basemap();
    (lat, lon, options)
}

fn global_request_with_projected_domain() -> (MapRenderRequest, ProjectedMapBuildOptions) {
    let (lat, lon, options) = global_robinson_mesh();
    let projected = build_projected_map_with_options(&lat, &lon, &options).unwrap();
    let resolved = resolved_projection_for_options(&lat, &lon, &options.domain).unwrap();
    let shape = GridShape::new(36, 18).unwrap();
    let values: Vec<f32> = (0..shape.len()).map(|value| value as f32).collect();
    let grid = LatLonGrid::new(shape, lat, lon).unwrap();
    let field = Field2D::new(ProductKey::named("global"), "K", grid, values).unwrap();
    let mut request = MapRenderRequest::new(
        field,
        ColorScale::Weather(crate::weather::WeatherPreset::Cape),
    );
    request.width = 800;
    request.height = 600;
    request.projected_domain = Some(projected.domain());
    request.resolved_projection = Some(resolved);
    request.geographic_bounds = Some((-180.0, 180.0, -90.0, 90.0));
    (request, options)
}

/// The round trip that proves the published transform IS the drawn
/// transform: for a global Robinson panel, `PanelGeoReference::
/// lonlat_to_pixel` must agree -- to well under a pixel -- with what the
/// renderer's own projection seam (`project_geographic_points_with_options`
/// through the panel's extent and plot rectangle) produces for the same
/// points.  If the published projection resolved a different central
/// meridian, or the published extent were a different box, these numbers
/// would disagree by tens to hundreds of pixels, which is exactly the
/// wrong-ocean failure this feature retires.
#[test]
fn published_georeference_matches_the_drawn_projection_on_a_global_panel() {
    let (request, options) = global_request_with_projected_domain();
    let (lat, lon, _) = global_robinson_mesh();

    let path = std::env::temp_dir().join(format!(
        "rustwx-georef-global-{}.png",
        std::process::id()
    ));
    let timing = save_png_profile(&request, &path).unwrap();
    let _ = std::fs::remove_file(&path);

    assert_eq!(
        timing.georeference_absent_reason, None,
        "a fully-specified request must publish"
    );
    let georeference = timing
        .georeference
        .expect("resolved projection + projected domain + bounds must publish");
    assert_eq!(georeference.image_width_px, 800);
    assert_eq!(georeference.image_height_px, 600);

    let image_timing = &timing.png_timing.image_timing;
    assert!(image_timing.plot_rect_describes_the_png);
    let extent = &request.projected_domain.as_ref().unwrap().extent;
    // The Southern Ocean point the placed-grid cascade got wrong, plus
    // spread-out controls.
    let points = [
        (-60.079, 139.499),
        (0.0, 0.0),
        (45.0, 90.0),
        (30.0, -100.0),
    ];
    let projected_points =
        project_geographic_points_with_options(&lat, &lon, &options, &points).unwrap();
    for ((point_lat, point_lon), (x, y)) in points.iter().zip(projected_points) {
        let (px, py) = georeference
            .lonlat_to_pixel(*point_lat, *point_lon)
            .expect("point inside the frame must land on a pixel");
        // The renderer's own answer for the same point: its projected
        // coordinates normalised through the panel extent onto the plot
        // rectangle.
        let rx = (x - extent.x_min) / (extent.x_max - extent.x_min);
        let ry = 1.0 - (y - extent.y_min) / (extent.y_max - extent.y_min);
        let expected_x =
            image_timing.map_x as f64 + rx * (image_timing.map_w.saturating_sub(1)) as f64;
        let expected_y =
            image_timing.map_y as f64 + ry * (image_timing.map_h.saturating_sub(1)) as f64;
        assert!(
            (px - expected_x).abs() < 0.05 && (py - expected_y).abs() < 0.05,
            "published transform disagrees with the drawn one at \
             ({point_lat},{point_lon}): published ({px},{py}), drawn ({expected_x},{expected_y})"
        );
    }
}

/// Centroid of the pixels within a small channel distance of `color`, in
/// FULL-image coordinates: where a drawn marker of that color actually
/// sits in the finished PNG.  This is the ground truth the pixel gate
/// compares against -- image content, not any rectangle the code under
/// test reported.
fn marker_centroid(image: &RgbaImage, color: Color) -> Option<(f64, f64)> {
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut count = 0usize;
    for (x, y, pixel) in image.enumerate_pixels() {
        let distance = u32::from(pixel.0[0].abs_diff(color.r))
            + u32::from(pixel.0[1].abs_diff(color.g))
            + u32::from(pixel.0[2].abs_diff(color.b));
        if distance < 90 && pixel.0[3] > 200 {
            sum_x += x as f64;
            sum_y += y as f64;
            count += 1;
        }
    }
    (count > 0).then(|| (sum_x / count as f64, sum_y / count as f64))
}

fn regional_lambert_request(
    bounds: (f64, f64, f64, f64),
) -> (MapRenderRequest, Vec<f32>, Vec<f32>, ProjectedMapBuildOptions) {
    let mut lat = Vec::new();
    let mut lon = Vec::new();
    for row in 0..16 {
        for col in 0..26 {
            lat.push(30.0 + row as f32);
            lon.push(-110.0 + col as f32);
        }
    }
    let mut options = ProjectedMapBuildOptions::from_bounds(bounds, 1.6)
        .with_projection(ProjectionSpec::LambertConformal {
            standard_parallel_1_deg: 33.0,
            standard_parallel_2_deg: 45.0,
            central_meridian_deg: -96.0,
        })
        .without_basemap();
    options.domain.reference_latitude_deg = Some(39.0);
    let projected = build_projected_map_with_options(&lat, &lon, &options).unwrap();
    let resolved = resolved_projection_for_options(&lat, &lon, &options.domain).unwrap();
    let shape = GridShape::new(26, 16).unwrap();
    let values: Vec<f32> = (0..shape.len()).map(|value| value as f32).collect();
    let grid = LatLonGrid::new(shape, lat.clone(), lon.clone()).unwrap();
    let field = Field2D::new(ProductKey::named("georef_pixel_probe"), "K", grid, values).unwrap();
    let mut request = MapRenderRequest::contour_only(field);
    request.width = 800;
    request.height = 600;
    request.projected_domain = Some(projected.domain());
    request.resolved_projection = Some(resolved);
    request.geographic_bounds = Some(bounds);
    (request, lat, lon, options)
}

/// The regional pixel gate, both directions.  A regional Lambert panel
/// takes the map-viewport crop -- the biggest of the three post-render
/// passes -- and its plot rectangle must FOLLOW the map through it:
///
/// * markers drawn at known lat/lons are located IN THE WRITTEN PNG by
///   color, and the published `lonlat_to_pixel` must land on them to
///   well under two pixels (marker centroid quantisation);
/// * the tester is then tested: the PRE-crop rectangle (the old
///   behaviour, reconstructed exactly from the reported offsets) must
///   FAIL the same comparison by more than a pixel.  If it did not, the
///   crop moved nothing and this test would be measuring nothing.
///
/// A published-and-wrong georeference is worse than a withheld one; this
/// is the gate that keeps the adjustment accurate.
#[test]
fn published_georeference_survives_the_crop_on_a_regional_lambert_panel() {
    let (mut request, lat, lon, options) =
        regional_lambert_request((-110.0, -85.0, 30.0, 45.0));
    let resolved = request.resolved_projection.unwrap();
    assert!(
        matches!(resolved, ResolvedProjection::LambertConformal { .. }),
        "the regional gate must run on a Lambert panel, got {resolved:?}"
    );

    request.domain_frame = Some(DomainFrame::map_viewport_default());

    // Known points spread across the frame, each marked in a color
    // nothing else in this panel uses (no fill, no basemap; linework and
    // text are black on paper).
    let points = [(33.0, -105.0), (42.0, -89.0), (36.5, -97.0)];
    let colors = [
        Color::rgba(255, 0, 255, 255),
        Color::rgba(0, 200, 0, 255),
        Color::rgba(255, 140, 0, 255),
    ];
    let projected_points =
        project_geographic_points_with_options(&lat, &lon, &options, &points).unwrap();
    for ((x, y), color) in projected_points.iter().zip(colors) {
        request.projected_points.push(ProjectedPointOverlay {
            x: *x,
            y: *y,
            color,
            radius_px: 4,
            width_px: 3,
            shape: ProjectedMarkerShape::Plus,
        });
    }

    let path = std::env::temp_dir().join(format!(
        "rustwx-georef-regional-{}.png",
        std::process::id()
    ));
    let timing = save_png_profile(&request, &path).unwrap();
    let final_image = image::load_from_memory_with_format(
        &std::fs::read(&path).unwrap(),
        ImageFormat::Png,
    )
    .unwrap()
    .to_rgba8();
    let _ = std::fs::remove_file(&path);

    let image_timing = &timing.png_timing.image_timing;
    assert!(
        image_timing.postprocess_offset_x != 0 || image_timing.postprocess_offset_y != 0,
        "the crop moved nothing, so this test is not measuring the crop -- \
         change the panel until it does"
    );
    let georeference = timing
        .georeference
        .expect("a cropped regional panel must still publish its transform");
    assert_eq!(
        (georeference.image_width_px, georeference.image_height_px),
        (final_image.width(), final_image.height()),
        "the published image size must be the written file's, not the request's"
    );

    // The OLD behaviour, reconstructed exactly: the pre-crop rectangle on
    // the pre-crop canvas.
    let stale = PanelGeoReference::new(
        request.width,
        request.height,
        PlotRect {
            x: (i64::from(georeference.plot_rect_px.x) - image_timing.postprocess_offset_x)
                as u32,
            y: (i64::from(georeference.plot_rect_px.y) - image_timing.postprocess_offset_y)
                as u32,
            width: georeference.plot_rect_px.width,
            height: georeference.plot_rect_px.height,
        },
        georeference.projection,
        georeference.extent.clone(),
        georeference.geographic_bounds,
    );

    let mut worst_published: f64 = 0.0;
    let mut worst_stale: f64 = 0.0;
    for (point, color) in points.iter().zip(colors) {
        let truth = marker_centroid(&final_image, color)
            .unwrap_or_else(|| panic!("marker {color:?} not found in the written PNG"));
        let (px, py) = georeference
            .lonlat_to_pixel(point.0, point.1)
            .expect("a point inside the frame must land on a pixel");
        worst_published =
            worst_published.max(((px - truth.0).powi(2) + (py - truth.1).powi(2)).sqrt());
        let (sx, sy) = stale
            .lonlat_to_pixel(point.0, point.1)
            .expect("the stale rectangle still places the point somewhere");
        worst_stale = worst_stale.max(((sx - truth.0).powi(2) + (sy - truth.1).powi(2)).sqrt());
    }
    assert!(
        worst_published < 2.0,
        "published transform misses the drawn markers by {worst_published} px"
    );
    assert!(
        worst_stale > 1.0 && worst_stale > worst_published,
        "the un-adjusted rectangle must measurably fail this comparison or the test \
         proves nothing: stale error {worst_stale} px, published error {worst_published} px"
    );
}

/// A moved rectangle that overhangs the written image is clipped to the
/// pixels that survive, and the extent is cut by the same fraction, so a
/// point lands on the same pixel through the clipped pair as through the
/// unclipped one.  Only a rectangle with no surviving pixel is `None`.
#[test]
fn a_moved_plot_rectangle_is_clipped_to_the_written_image() {
    // Inside: unchanged, nothing cut.
    let fits = clip_plot_rect_to_image(18, 46, 700, 500, 800, 600).unwrap();
    assert_eq!((fits.x, fits.y, fits.width, fits.height), (18, 46, 700, 500));
    assert_eq!((fits.left, fits.right, fits.top, fits.bottom), (0.0, 0.0, 0.0, 0.0));
    // Recentred past the right edge by 186 px on an 800-px image: the
    // right 168 px of the 764-px rectangle are gone.
    let right = clip_plot_rect_to_image(18 + 186, 64, 764, 518, 800, 600).unwrap();
    assert_eq!((right.x, right.y, right.width, right.height), (204, 64, 596, 518));
    assert!((right.right - 168.0 / 763.0).abs() < 1e-12);
    assert_eq!((right.left, right.top, right.bottom), (0.0, 0.0, 0.0));
    // Cut on the left and the top: the origin is clamped to zero.
    let corner = clip_plot_rect_to_image(-50, -10, 700, 500, 800, 600).unwrap();
    assert_eq!((corner.x, corner.y, corner.width, corner.height), (0, 0, 650, 490));
    assert!((corner.left - 50.0 / 699.0).abs() < 1e-12);
    assert!((corner.top - 10.0 / 499.0).abs() < 1e-12);
    // Nothing survives: entirely to the right, or a zero-size rectangle.
    assert!(clip_plot_rect_to_image(800, 64, 764, 518, 800, 600).is_none());
    assert!(clip_plot_rect_to_image(18, 64, 0, 518, 800, 600).is_none());
}

/// The published transform through a clipped rectangle is the transform
/// through the unclipped one: the extent is cut by the fraction the
/// rectangle was, so every surviving pixel keeps its projected
/// coordinate.
#[test]
fn a_clipped_georeference_places_a_point_where_the_unclipped_one_did() {
    let (request, _) = global_request_with_projected_domain();
    let extent = request.projected_domain.as_ref().unwrap().extent.clone();
    let unclipped = georeference::PanelGeoReference::new(
        800,
        600,
        PlotRect { x: 204, y: 64, width: 764, height: 518 },
        request.resolved_projection.unwrap(),
        extent.clone(),
        request.geographic_bounds.unwrap(),
    );
    let clip = clip_plot_rect_to_image(204, 64, 764, 518, 800, 600).unwrap();
    let image_timing = RenderImageTiming {
        map_x: clip.x,
        map_y: clip.y,
        map_w: clip.width,
        map_h: clip.height,
        image_w: 800,
        image_h: 600,
        map_clip_left: clip.left,
        map_clip_right: clip.right,
        map_clip_top: clip.top,
        map_clip_bottom: clip.bottom,
        plot_rect_describes_the_png: true,
        ..RenderImageTiming::default()
    };
    let (published, reason) = panel_georeference_for_save(&request, &image_timing);
    assert_eq!(reason, None);
    let published = published.expect("a clipped rectangle with area publishes");
    assert_eq!(published.plot_rect_px.width, 596);
    assert!(published.extent.x_max < extent.x_max, "the extent's east edge was cut");
    assert_eq!(published.extent.x_min, extent.x_min);
    // A point in the surviving part of the map: the same pixel either
    // way, to floating point.
    let (x, y) = published.projection.project(30.0, -100.0);
    let (ux, uy) = unclipped.projected_to_pixel(x, y).expect("inside the unclipped rectangle");
    let (cx, cy) = published.projected_to_pixel(x, y).expect("inside the clipped rectangle");
    assert!((ux - cx).abs() < 1e-6 && (uy - cy).abs() < 1e-6, "({ux},{uy}) vs ({cx},{cy})");
}

/// A regional grid drawn in a frame wider than its data: the production
/// projected-grid frame recentres the content by more than the map's
/// margin, so the moved rectangle overhangs the written image.  The
/// sidecar was withheld for exactly this ("moved the map in a way its
/// reported offset cannot describe") on a correctly drawn map.  Now the
/// rectangle is clipped and the published transform lands on the drawn
/// markers.
#[test]
fn a_recentred_grid_that_overhangs_the_image_still_publishes_a_clipped_georeference() {
    // The frame reaches 25 degrees east of the data.
    let (mut request, lat, lon, options) =
        regional_lambert_request((-110.0, -60.0, 30.0, 45.0));
    request.domain_frame = Some(DomainFrame::model_data_default());
    // Three points inside the frame the grid draws.
    let points = [(33.0, -105.0), (36.5, -97.0), (34.0, -92.5)];
    let colors = [
        Color::rgba(255, 0, 255, 255),
        Color::rgba(0, 200, 0, 255),
        Color::rgba(255, 140, 0, 255),
    ];
    let projected_points =
        project_geographic_points_with_options(&lat, &lon, &options, &points).unwrap();
    for ((x, y), color) in projected_points.iter().zip(colors) {
        request.projected_points.push(ProjectedPointOverlay {
            x: *x,
            y: *y,
            color,
            radius_px: 4,
            width_px: 3,
            shape: ProjectedMarkerShape::Plus,
        });
    }
    let path = std::env::temp_dir().join(format!(
        "rustwx-georef-overhang-{}.png",
        std::process::id()
    ));
    let timing = save_png_profile(&request, &path).unwrap();
    let final_image = image::load_from_memory_with_format(
        &std::fs::read(&path).unwrap(),
        ImageFormat::Png,
    )
    .unwrap()
    .to_rgba8();
    let _ = std::fs::remove_file(&path);

    let image_timing = &timing.png_timing.image_timing;
    // The tester tested: the recentre must have moved the map past the
    // image's edge, or this panel is not the case this test is about.
    assert!(
        image_timing.map_clip_right > 0.05 && image_timing.postprocess_offset_x > 0,
        "the recentre did not push the map past the image's edge (clip right {}, offset {}), \
         so this test is not measuring the clip -- change the frame until it does",
        image_timing.map_clip_right,
        image_timing.postprocess_offset_x
    );
    assert!(image_timing.plot_rect_describes_the_png);
    assert!(
        image_timing.map_x + image_timing.map_w <= image_timing.image_w,
        "the clipped rectangle must lie inside the written image"
    );
    let georeference = timing
        .georeference
        .expect("a recentred regional panel whose rectangle overhangs must still publish");
    assert_eq!(
        (georeference.image_width_px, georeference.image_height_px),
        (final_image.width(), final_image.height())
    );
    let mut worst = 0.0f64;
    for (point, color) in points.iter().zip(colors) {
        let truth = marker_centroid(&final_image, color)
            .unwrap_or_else(|| panic!("marker {color:?} not found in the written PNG"));
        let (px, py) = georeference
            .lonlat_to_pixel(point.0, point.1)
            .expect("a point inside the surviving map must land on a pixel");
        worst = worst.max(((px - truth.0).powi(2) + (py - truth.1).powi(2)).sqrt());
    }
    assert!(
        worst < 2.0,
        "published transform misses the drawn markers by {worst} px"
    );
}

/// The residual refusal: a rectangle with no surviving pixel.  On every
/// other path the passes report their offsets and the rectangle follows
/// the map, clipped where it overhangs, so the
/// `plot_rect_describes_the_png == false` branch is constructed directly
/// here to pin that it still withholds rather than publishes.
#[test]
fn an_unreportable_map_move_still_withholds_the_georeference() {
    let (request, _) = global_request_with_projected_domain();
    let mut image_timing = RenderImageTiming {
        map_x: 18,
        map_y: 46,
        map_w: 700,
        map_h: 500,
        image_w: 800,
        image_h: 600,
        plot_rect_describes_the_png: false,
        ..RenderImageTiming::default()
    };

    let (georeference, reason) = panel_georeference_for_save(&request, &image_timing);
    assert!(georeference.is_none(), "a retired rectangle must not publish");
    let reason = reason.expect("a withheld georeference must say why");
    assert!(
        reason.contains("moved the map"),
        "the reason must name the concrete breakage: {reason}"
    );

    // The same timing with the rectangle intact publishes -- the refusal
    // above is the flag's doing, nothing else's.
    image_timing.plot_rect_describes_the_png = true;
    let (georeference, reason) = panel_georeference_for_save(&request, &image_timing);
    assert!(georeference.is_some() && reason.is_none());
}

#[test]
fn contour_only_map_with_height_contours_and_barbs_renders_visible_overlays() {
    let base = sample_field("height");
    let contours = sample_field("height_contours");
    let u = sample_field("u_wind");
    let mut v = sample_field("v_wind");
    v.values.iter_mut().for_each(|value| *value = 10.0);

    let request = MapRenderRequest::contour_only(base)
        .with_contour_field(
            &contours,
            vec![500.0, 1500.0, 2500.0, 3500.0],
            ContourStyle {
                labels: true,
                ..Default::default()
            },
        )
        .unwrap()
        .with_wind_barbs(
            &u,
            &v,
            WindBarbStyle {
                stride_x: 2,
                stride_y: 2,
                ..Default::default()
            },
        )
        .unwrap();

    let image = render_image(&request).unwrap();
    let non_white = image
        .pixels()
        .filter(|px| px.0 != [255, 255, 255, 255])
        .count();
    assert!(
        non_white > 1000,
        "overlay-only render should remain visible"
    );
}

/// Three category codes, 0, 1 and 2, as integer-centred bands.
fn three_code_scale() -> ColorScale {
    ColorScale::Discrete(DiscreteColorScale {
        levels: vec![-0.5, 0.5, 1.5, 2.5],
        colors: vec![
            Color::rgba(0, 0, 255, 255),
            Color::rgba(0, 255, 0, 255),
            Color::rgba(255, 0, 0, 255),
        ],
        extend: ExtendMode::Neither,
        mask_below: None,
    })
}

#[test]
fn geographic_viewport_keeps_context_outside_sparse_weather() {
    let (mut request, lat, lon, options) =
        regional_lambert_request((-110.0, -85.0, 30.0, 45.0));
    request.inverse_raster_projection = Some(InverseRasterProjection {
        projection: ProjectionSpec::LambertConformal {
            standard_parallel_1_deg: 33., standard_parallel_2_deg: 45.,
            central_meridian_deg: -96.,
        },
        reference_latitude_deg: Some(39.), reference_longitude_deg: None,
        clip_bounds: None,
    });
    request.scale = three_code_scale();
    request.field.values.fill(f32::NAN);
    for row in 0..5 {
        for col in 0..5 { request.field.values[row * 26 + col] = 1.; }
    }
    let position = project_geographic_points_with_options(
        &lat, &lon, &options, &[(40., -90.)]).unwrap()[0];
    request.projected_points.push(ProjectedPointOverlay {
        x: position.0, y: position.1, color: Color::rgba(255, 0, 255, 255),
        radius_px: 5, width_px: 3, shape: ProjectedMarkerShape::Plus,
    });
    request.domain_frame = Some(DomainFrame::map_viewport_default());
    let image = render_image(&request).unwrap();
    let marker_pixels = image.pixels().filter(|p| p[0] > 200 && p[1] < 50 && p[2] > 200).count();
    assert!(marker_pixels > 10, "geographic context was clipped to the finite weather swath");
}

fn category_legend() -> LegendControls {
    LegendControls {
        density: LevelDensity::default(),
        mode: LegendMode::Categories,
    }
}

#[test]
fn projected_category_maps_draw_only_the_codes_the_grid_holds() {
    // A 3x3 plane of codes 0 and 2 in a checkerboard: any pixel between two
    // grid points that is drawn as code 1 was invented by interpolation.
    // The regular mesh takes the rectilinear rasterizer, the skewed one the
    // triangle rasterizer; both must sample the nearest code.
    for skew in [0.0, 0.35] {
        let shape = GridShape::new(3, 3).unwrap();
        let lat: Vec<f32> = (0..9).map(|cell| 35.0 + (cell / 3) as f32).collect();
        let lon: Vec<f32> = (0..9).map(|cell| -97.0 + (cell % 3) as f32).collect();
        let grid = LatLonGrid::new(shape, lat, lon).unwrap();
        let values: Vec<f32> = (0..9)
            .map(|cell| if cell % 2 == 0 { 0.0 } else { 2.0 })
            .collect();
        let field = Field2D::new(ProductKey::named("category"), "", grid, values).unwrap();
        let mut request = MapRenderRequest::new(field, three_code_scale());
        request.width = 360;
        request.height = 300;
        request.colorbar = false;
        request.legend = category_legend();
        request.projected_domain = Some(ProjectedDomain {
            x: (0..9)
                .map(|cell| (cell % 3) as f64 + skew * (cell / 3) as f64)
                .collect(),
            y: (0..9).map(|cell| (cell / 3) as f64).collect(),
            extent: ProjectedExtent {
                x_min: 0.0,
                x_max: 2.0 + 2.0 * skew,
                y_min: 0.0,
                y_max: 2.0,
            },
        });
        for style in [
            StaticPlotStyle::OperationalFast,
            StaticPlotStyle::OperationalBudget30s,
        ] {
            let image = render_image_with_style(&request, style).unwrap();
            let held_low = image.pixels().filter(|px| px.0 == [0, 0, 255, 255]).count();
            let held_high = image.pixels().filter(|px| px.0 == [255, 0, 0, 255]).count();
            let invented = image.pixels().filter(|px| px.0 == [0, 255, 0, 255]).count();
            assert!(held_low > 1000 && held_high > 1000, "skew {skew} {style:?}");
            assert_eq!(invented, 0, "code 1 invented at skew {skew} under {style:?}");
        }
    }
}

#[test]
fn category_colormap_fill_is_its_legend_under_every_plot_style() {
    let codes: Vec<f64> = (1..=21).map(f64::from).collect();
    let levels: Vec<f64> = (0..=21).map(|edge| edge as f64 + 0.5).collect();
    let palette = [Rgba::new(68, 1, 84), Rgba::new(253, 231, 37)];
    for style in [
        StaticPlotStyle::Default,
        StaticPlotStyle::OperationalFast,
        StaticPlotStyle::CleanAtlasCombined,
    ] {
        let cmap = LeveledColormap::from_palette_with_options(
            &palette,
            &levels,
            Extend::Neither,
            None,
            ColormapBuildOptions {
                render_density: style.render_density(RenderDensity::default()),
                legend: category_legend(),
            },
        );
        assert_eq!(cmap.levels, levels, "{style:?} densified a category legend");
        assert_eq!(colorbar_ticks(&cmap, Some(1.0)), codes);
        let mut seen = Vec::new();
        for &code in &codes {
            let fill = cmap.map(code);
            let rel = legend_tick_rel(&cmap, code).unwrap();
            assert_eq!(
                legend_color_at_rel(&cmap, LegendMode::Categories, rel),
                fill,
                "code {code} under {style:?}"
            );
            assert!(!seen.contains(&fill), "code {code} shares a colour");
            seen.push(fill);
        }
    }
}

#[test]
fn a_large_frame_draws_place_labels_at_twice_the_size() {
    assert_eq!(planned_label_text_scale(1, 1), 1);
    assert_eq!(planned_label_text_scale(2, 1), 2);
    // 12 px doubled is 24 px: the text table's fourth step.
    assert_eq!(planned_label_text_scale(1, 2), 4);
    assert_eq!(planned_label_text_scale(2, 2), 6);
}

#[test]
fn a_dark_theme_turns_a_near_black_label_to_its_ink_and_a_light_theme_keeps_it() {
    let dark = RenderTheme::builtin("dark").expect("built in").presentation;
    let label = Rgba::new(24, 31, 39);
    let inked = themed_label_ink(dark, label);
    assert!(inked.r.min(inked.g).min(inked.b) > 150, "{inked:?}");
    assert_eq!(themed_label_ink(PresentationTheme::default(), label), label);
}

#[test]
fn render_scratch_survives_nested_rayon_render_and_unwind() {
    let pool = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
    pool.install(|| {
        let request = MapRenderRequest::for_weather_product(
            sample_field("sbecape"), WeatherProduct::Sbecape);
        let expected = RustRenderer.render_png(&request).unwrap();
        let actual = with_render_state(&request, |data, ny, nx, opts| {
            // A worker waiting for raster rows may execute another complete
            // product before resuming this callback. Force that schedule.
            let (nested, ()) = rayon::join(
                || RustRenderer.render_png(&request).unwrap(), || ());
            assert_eq!(nested, expected);
            Ok(render_to_png(data, ny, nx, opts))
        }).unwrap();
        assert_eq!(actual, expected);
        let panic = std::panic::catch_unwind(|| {
            with_render_scratch(|_| panic!("interrupted render"));
        });
        assert!(panic.is_err());
        with_render_scratch(|scratch| {
            assert_eq!(scratch.fill_f64_constant(3, 7.0), vec![7.0; 3]);
        });
    });
}
