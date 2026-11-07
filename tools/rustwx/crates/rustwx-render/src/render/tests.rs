use super::*;
use crate::colormap::{ColormapBuildOptions, Extend, LevelDensity};
use crate::presentation::StaticPlotStyle;

fn sample_cmap() -> LeveledColormap {
    LeveledColormap::from_palette(
        &[Rgba::new(0, 0, 255), Rgba::new(255, 0, 0)],
        &[0.0, 1.0, 2.0, 3.0],
        Extend::Neither,
        None,
    )
}

fn sample_masked_cmap() -> LeveledColormap {
    LeveledColormap::from_palette(
        &[Rgba::new(0, 0, 255), Rgba::new(255, 0, 0)],
        &[10.0, 20.0, 30.0],
        Extend::Neither,
        Some(10.0),
    )
}

fn sample_projected_grid() -> ProjectedGrid {
    ProjectedGrid {
        x: vec![0.0, 1.0, 0.0, 1.0],
        y: vec![0.0, 0.0, 1.0, 1.0],
        ny: 2,
        nx: 2,
    }
}

fn sample_projected_opts() -> RenderOpts {
    RenderOpts {
        width: 240,
        height: 160,
        cmap: sample_cmap(),
        background: Rgba::WHITE,
        colorbar: false,
        colorbar_units: None,
        title: Some("Projected".into()),
        subtitle_left: None,
        subtitle_center: None,
        subtitle_right: None,
        cbar_tick_step: None,
        colorbar_mode: crate::colormap::LegendMode::Stepped,
        chrome_scale: ChromeScale::default(),
        supersample_factor: 1,
        supersample_sharpen: true,
        raster_sample_mode: RasterSampleMode::default(),
        domain_frame: None,
        map_extent: Some(MapExtent {
            x_min: 0.0,
            x_max: 1.0,
            y_min: 0.0,
            y_max: 1.0,
        }),
        projected_grid: Some(sample_projected_grid()),
        inverse_projected_grid: None,
        rgba_grid: None,
        projected_polygons: Vec::new(),
        projected_data_polygons: Vec::new(),
        mesh_cells: None,
        projected_place_labels: Vec::new(),
        projected_points: Vec::new(),
        projected_lines: Vec::new(),
        contours: Vec::new(),
        barbs: Vec::new(),
        streamlines: Vec::new(),
        presentation: RenderPresentation::for_mode(ProductVisualMode::FilledMeteorology),
        empty_note: None,
    }
}

#[test]
fn vertical_colorbar_follows_domain_frame_horizontally() {
    let presentation = RenderPresentation::for_mode_with_style(
        ProductVisualMode::FilledMeteorology,
        StaticPlotStyle::OperationalBudget30s,
    );
    let layout = compute_layout(2400, 1600, true, true, presentation, ChromeScale::default());
    let frame = DomainFrame::map_viewport_default();
    let rect = LocalRect {
        min_x: 0,
        max_x: layout.map_w.saturating_sub(160),
        min_y: 0,
        max_y: layout.map_h.saturating_sub(1),
    };

    let (x, y, w) = colorbar_anchor_rect(
        &layout,
        ColorbarOrientation::VerticalRight,
        Some(frame),
        Some(rect),
    );
    let frame_gap = (2u32.saturating_mul(layout.text_scale.max(1))).clamp(4, 8);

    assert_eq!(x, layout.map_x + rect.max_x + frame_gap);
    assert!(x < layout.cbar_x);
    assert_eq!(y, layout.cbar_y);
    assert_eq!(w, layout.cbar_w);
}

#[test]
fn colorbar_units_preserve_map_ticks_and_complete_glyphs() {
    for style in [StaticPlotStyle::OperationalBudget30s, StaticPlotStyle::CleanAtlas] {
    for has_title in [false, true] {
    for has_subtitle in [false, true] {
    for (width, height) in [(600, 450), (300, 240), (1400, 1000)] {
    for units in ["degF", "dBZ", "kt", "kg m-2 s-1", "m2 s-2", "kg kg-1"] {
        let mut opts = sample_projected_opts();
        opts.width = width;
        opts.height = height;
        opts.title = has_title.then(|| "Field".into());
        opts.subtitle_right = has_subtitle.then(|| {
            if width < 600 { "18:00Z" } else { "2026-09-12 18:00:00Z" }.into()
        });
        opts.colorbar = true;
        opts.presentation = RenderPresentation::for_mode_with_style(
            ProductVisualMode::FilledMeteorology, style);
        let (bare, timing) = render_to_image_profile(&[0.5, 1.0, 1.5, 2.0], 2, 2, &opts);
        opts.colorbar_units = Some(units.into());
        let (labelled, _) = render_to_image_profile(&[0.5, 1.0, 1.5, 2.0], 2, 2, &opts);
        let canvas = opts.presentation.canvas_background.to_image_rgba();
        let mut changed = 0;
        for y in 0..bare.height() {
            for x in 0..bare.width() {
                if bare.get_pixel(x, y) != labelled.get_pixel(x, y) {
                    changed += 1;
                    assert!(x < timing.map_x || x >= timing.map_x + timing.map_w
                        || y < timing.map_y || y >= timing.map_y + timing.map_h,
                        "units changed the map: {style:?} {width}x{height} {has_title} {units}");
                    assert_eq!(bare.get_pixel(x, y), &canvas,
                        "units overwrote legend ink: {style:?} {width}x{height} {has_title} {units}");
                }
            }
        }
        // Compare with the complete glyphs drawn by the existing font owner
        // on an unconstrained canvas. Matching the coverage count also catches
        // a label clipped at an image edge, even if it missed all old ink.
        let layout = compute_effective_layout(width, height, true, has_title || has_subtitle,
            opts.presentation, opts.chrome_scale, false);
        let mut reference = RgbaImage::from_pixel(400, 100, canvas);
        text::draw_text_with_factor(&mut reference, units, 32, 32,
            opts.presentation.colorbar.label_color, layout.text_scale, layout.label_factor);
        let expected = reference.pixels().filter(|p| *p != &canvas).count();
        assert!(expected > 0);
        assert_eq!(changed, expected,
            "units lost glyph pixels: {style:?} {width}x{height} {has_title} {units}");
    }
    }
    }
    }
    }
}

fn sample_place_label() -> ProjectedPlaceLabelOverlay {
    ProjectedPlaceLabelOverlay {
        x: 0.52,
        y: 0.48,
        label: Some("Sacramento".into()),
        priority: ProjectedPlaceLabelPriority::Primary,
        style: crate::overlay::ProjectedPlaceLabelStyle {
            marker_radius_px: 4,
            marker_fill: Rgba::with_alpha(255, 255, 255, 235),
            marker_outline: Rgba::with_alpha(24, 28, 34, 240),
            marker_outline_width: 1,
            label_color: Rgba::BLACK,
            label_halo: Rgba::with_alpha(255, 255, 255, 235),
            label_halo_width_px: 2,
            label_scale: 1,
            label_offset_x_px: 6,
            label_offset_y_px: -2,
            label_placement: ProjectedLabelPlacement::AboveRight,
            label_bold: true,
        },
    }
}

fn contour_test_layout() -> Layout {
    Layout {
        map_x: 0,
        map_y: 0,
        map_w: 64,
        map_h: 64,
        halo: Rgba::WHITE,
        title_factor: 1.0,
        label_factor: 1.0,
        cbar_x: 0,
        cbar_y: 0,
        cbar_w: 0,
        cbar_h: 0,
        title_y: 0,
        subtitle_y: 0,
        text_scale: 1,
        label_gap: 14,
        plan: None,
    }
}

fn blank_test_image() -> RgbaImage {
    RgbaImage::from_pixel(80, 80, Rgba::WHITE.to_image_rgba())
}

fn non_white_bounds(img: &RgbaImage) -> Option<(u32, u32, u32, u32)> {
    let mut min_x = u32::MAX;
    let mut max_x = 0u32;
    let mut min_y = u32::MAX;
    let mut max_y = 0u32;
    let mut found = false;

    for (x, y, pixel) in img.enumerate_pixels() {
        if pixel.0 == [255, 255, 255, 255] {
            continue;
        }
        found = true;
        min_x = min_x.min(x);
        max_x = max_x.max(x);
        min_y = min_y.min(y);
        max_y = max_y.max(y);
    }

    found.then_some((min_x, max_x, min_y, max_y))
}

fn sample_domain_frame(outline_color: crate::request::Color) -> DomainFrame {
    DomainFrame {
        inset_px: 5,
        outline_color,
        outline_width: 2,
        clear_outside: true,
        legend_follows_frame: true,
        chrome_follows_frame: true,
        source: crate::request::DomainFrameSource::ProjectedGrid,
    }
}

#[test]
fn projected_pixels_keep_nearby_offscreen_points_for_clipping() {
    let layout = contour_test_layout();
    let grid = ProjectedGrid {
        x: vec![-0.05, 1.05, -0.05, 1.05],
        y: vec![0.0, 0.0, 1.0, 1.0],
        ny: 2,
        nx: 2,
    };
    let extent = MapExtent {
        x_min: 0.0,
        x_max: 1.0,
        y_min: 0.0,
        y_max: 1.0,
    };

    let pixels = projected_grid_to_pixels(&grid, &extent, &layout);

    assert_eq!(pixels.len(), 4);
    assert!(pixels[0].is_some_and(|(x, _)| x < 0.0));
    assert!(pixels[1].is_some_and(|(x, _)| x > layout.map_w as f32 - 1.0));
}

fn visit_rs_files(
    root: &std::path::Path,
    visitor: &mut impl FnMut(&std::path::Path),
) -> std::io::Result<()> {
    for entry in std::fs::read_dir(root)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            visit_rs_files(&path, visitor)?;
        } else if path.extension().and_then(|ext| ext.to_str()) == Some("rs") {
            visitor(&path);
        }
    }
    Ok(())
}

#[test]
fn supersample_scaling_expands_overlay_dimensions() {
    let mut opts = sample_projected_opts();
    opts.projected_lines = vec![ProjectedPolyline {
        points: vec![(0.0, 0.0), (1.0, 1.0)],
        color: Rgba::BLACK,
        width: 2,
        role: crate::presentation::LineworkRole::Generic,
    }];
    opts.projected_points = vec![ProjectedPointOverlay {
        x: 0.50,
        y: 0.50,
        color: Rgba::new(255, 80, 40),
        radius_px: 5,
        width_px: 2,
        shape: ProjectedMarkerShape::Plus,
    }];
    opts.contours = vec![ContourOverlay {
        data: vec![500.0, 504.0, 508.0, 512.0],
        ny: 2,
        nx: 2,
        levels: vec![504.0],
        color: Rgba::BLACK,
        width: 1,
        labels: false,
        show_extrema: false,
        pattern: crate::request::ContourLinePattern::Solid,
        major_every: Some(1),
        major_width: Some(3),
    }];
    opts.barbs = vec![BarbOverlay {
        direction_projection: None,
        direction_arrows: false,
        u: vec![10.0, 10.0, 10.0, 10.0],
        v: vec![0.0, 0.0, 0.0, 0.0],
        ny: 2,
        nx: 2,
        stride_x: 1,
        stride_y: 1,
        spacing_px: 24.0,
        color: Rgba::BLACK,
        halo_color: Rgba::WHITE,
        halo_width: 2,
        width: 1,
        length_px: 18.0,
    }];
    opts.streamlines = vec![StreamlineOverlay {
        u: vec![10.0, 10.0, 10.0, 10.0],
        v: vec![0.0, 0.0, 0.0, 0.0],
        ny: 2,
        nx: 2,
        stride_x: 1,
        stride_y: 1,
        color: Rgba::with_alpha(0, 0, 0, 120),
        width: 1,
        max_steps: 4,
        step_cells: 0.5,
        min_speed: 2.5,
    }];
    opts.projected_place_labels = vec![sample_place_label()];
    opts.domain_frame = Some(sample_domain_frame(crate::request::Color::BLACK));

    let scaled = scale_render_opts_for_supersample(&opts, 2);
    assert_eq!(scaled.width, opts.width * 2);
    assert_eq!(scaled.height, opts.height * 2);
    assert_eq!(scaled.projected_lines[0].width, 4);
    assert_eq!(scaled.projected_points[0].radius_px, 10);
    assert_eq!(scaled.projected_points[0].width_px, 4);
    assert_eq!(scaled.projected_place_labels[0].style.marker_radius_px, 8);
    assert_eq!(scaled.projected_place_labels[0].style.label_scale, 2);
    assert_eq!(scaled.projected_place_labels[0].style.label_offset_x_px, 12);
    assert_eq!(scaled.contours[0].width, 2);
    assert_eq!(scaled.contours[0].major_width, Some(6));
    assert_eq!(scaled.barbs[0].width, 2);
    assert_eq!(scaled.barbs[0].halo_width, 4);
    assert_eq!(scaled.barbs[0].spacing_px, 48.0);
    assert_eq!(scaled.barbs[0].length_px, 36.0);
    assert_eq!(scaled.streamlines[0].width, 2);
    assert_eq!(scaled.domain_frame.unwrap().outline_width, 4);
    assert_eq!(scaled.supersample_factor, 1);
    assert_eq!(scaled.supersample_sharpen, opts.supersample_sharpen);
}

#[test]
fn wind_streamlines_draw_visible_flow_lines() {
    let mut image = blank_test_image();
    let layout = contour_test_layout();
    let overlay = StreamlineOverlay {
        u: vec![10.0; 64],
        v: vec![0.0; 64],
        ny: 8,
        nx: 8,
        stride_x: 2,
        stride_y: 2,
        color: Rgba::BLACK,
        width: 1,
        max_steps: 8,
        step_cells: 0.5,
        min_speed: 2.5,
    };

    draw_streamlines(&mut image, &layout, &overlay, None, None);

    let bounds = non_white_bounds(&image).expect("streamlines should draw");
    assert!(bounds.1 > bounds.0, "flow lines should span horizontally");
}

#[test]
fn render_to_image_supersample_preserves_requested_dimensions() {
    let mut opts = sample_projected_opts();
    opts.supersample_factor = 2;
    let data = vec![10.0, 20.0, 30.0, 25.0];
    let (image, timing) = render_to_image_profile(&data, 2, 2, &opts);
    assert_eq!(image.width(), opts.width);
    assert_eq!(image.height(), opts.height);
    assert!(timing.postprocess_ms <= timing.total_ms);
}

#[test]
fn render_to_image_supersample_can_skip_sharpen_pass() {
    let mut opts = sample_projected_opts();
    opts.supersample_factor = 2;
    opts.supersample_sharpen = false;
    let data = vec![10.0, 20.0, 30.0, 25.0];

    let (image, timing) = render_to_image_profile(&data, 2, 2, &opts);

    assert_eq!(image.width(), opts.width);
    assert_eq!(image.height(), opts.height);
    assert!(timing.downsample_ms <= timing.total_ms);
}

#[test]
fn projected_place_labels_render_visible_marker_and_text() {
    let mut opts = sample_projected_opts();
    opts.projected_place_labels = vec![sample_place_label()];
    let data = vec![0.5, 1.0, 1.5, 2.0];

    let image = render_to_image(&data, 2, 2, &opts);
    let dark_pixels = image
        .pixels()
        .filter(|pixel| pixel.0[0] < 80 && pixel.0[1] < 80 && pixel.0[2] < 80)
        .count();
    let bright_pixels = image
        .pixels()
        .filter(|pixel| pixel.0[0] > 220 && pixel.0[1] > 220 && pixel.0[2] > 220)
        .count();

    assert!(
        dark_pixels > 50,
        "label text and marker outline should be visible"
    );
    assert!(
        bright_pixels > 200,
        "marker fill and halo should be visible"
    );
}

#[test]
fn projected_points_render_visible_marker() {
    let mut opts = sample_projected_opts();
    opts.projected_points = vec![ProjectedPointOverlay {
        x: 0.50,
        y: 0.50,
        color: Rgba::new(255, 50, 20),
        radius_px: 8,
        width_px: 2,
        shape: ProjectedMarkerShape::Plus,
    }];
    let data = vec![0.5, 1.0, 1.5, 2.0];

    let image = render_to_image(&data, 2, 2, &opts);
    let red_pixels = image
        .pixels()
        .filter(|pixel| pixel.0[0] > 200 && pixel.0[1] < 100 && pixel.0[2] < 100)
        .count();

    assert!(red_pixels > 15, "projected point marker should be visible");
}

#[test]
fn projected_place_labels_clamp_text_inside_requested_clip_rect() {
    let presentation = RenderPresentation::for_mode(ProductVisualMode::FilledMeteorology);
    let layout = compute_layout(240, 160, false, false, presentation, ChromeScale::default());
    let extent = MapExtent {
        x_min: 0.0,
        x_max: 1.0,
        y_min: 0.0,
        y_max: 1.0,
    };
    let clip_rect = LocalRect {
        min_x: 20,
        max_x: 80,
        min_y: 20,
        max_y: 60,
    };
    let local_x = clip_rect.max_x.saturating_sub(2) as f64;
    let local_y = clip_rect.max_y.saturating_sub(2) as f64;
    let mut style = sample_place_label().style;
    style.marker_radius_px = 0;
    style.marker_outline_width = 0;
    style.label_halo = Rgba::TRANSPARENT;
    style.label_halo_width_px = 0;
    style.label_offset_x_px = 28;
    style.label_offset_y_px = 14;
    style.label_placement = ProjectedLabelPlacement::BelowRight;
    let label = ProjectedPlaceLabelOverlay {
        x: local_x / layout.map_w.saturating_sub(1) as f64,
        y: 1.0 - (local_y / layout.map_h.saturating_sub(1) as f64),
        label: Some("Sacramento Valley".into()),
        priority: ProjectedPlaceLabelPriority::Primary,
        style,
    };
    let mut img = RgbaImage::from_pixel(240, 160, Rgba::WHITE.to_image_rgba());

    draw_projected_place_labels(&mut img, &layout, &extent, &[label], None, Some(clip_rect));

    let (min_x, max_x, min_y, max_y) =
        non_white_bounds(&img).expect("clipped place label should still render");
    assert!(min_x >= layout.map_x + clip_rect.min_x);
    assert!(max_x <= layout.map_x + clip_rect.max_x);
    assert!(min_y >= layout.map_y + clip_rect.min_y);
    assert!(max_y <= layout.map_y + clip_rect.max_y);
}

#[test]
fn projected_place_labels_skip_marker_and_text_outside_requested_clip_mask() {
    let presentation = RenderPresentation::for_mode(ProductVisualMode::FilledMeteorology);
    let layout = compute_layout(240, 160, false, false, presentation, ChromeScale::default());
    let extent = MapExtent {
        x_min: 0.0,
        x_max: 1.0,
        y_min: 0.0,
        y_max: 1.0,
    };
    let clip_mask = RgbaImage::from_pixel(
        layout.map_w,
        layout.map_h,
        Rgba::TRANSPARENT.to_image_rgba(),
    );
    let mut img = RgbaImage::from_pixel(240, 160, Rgba::WHITE.to_image_rgba());

    draw_projected_place_labels(
        &mut img,
        &layout,
        &extent,
        &[sample_place_label()],
        Some(&clip_mask),
        None,
    );

    assert!(
        non_white_bounds(&img).is_none(),
        "place labels whose marker falls outside the clip mask should not render"
    );
}

#[test]
fn projected_place_label_priorities_reduce_auxiliary_and_micro_visual_weight() {
    let primary = place_label_render_adjustments(ProjectedPlaceLabelPriority::Primary);
    let auxiliary = place_label_render_adjustments(ProjectedPlaceLabelPriority::Auxiliary);
    let micro = place_label_render_adjustments(ProjectedPlaceLabelPriority::Micro);

    assert_eq!(primary.text_size_factor, 1.0);
    assert_eq!(primary.marker_scale_factor, 1.0);
    assert!(auxiliary.text_size_factor < primary.text_size_factor);
    assert!(auxiliary.text_alpha_factor < primary.text_alpha_factor);
    assert!(micro.text_size_factor < auxiliary.text_size_factor);
    assert!(micro.text_alpha_factor < auxiliary.text_alpha_factor);
    assert!(micro.marker_scale_factor < auxiliary.marker_scale_factor);
    assert!(micro.halo_width_factor < auxiliary.halo_width_factor);
}

fn slanted_projected_fixture() -> (Layout, ProjectedGrid, Arc<[Option<(f64, f64)>]>, LocalRect) {
    let layout = compute_layout(
        320,
        240,
        true,
        true,
        RenderPresentation::for_mode(ProductVisualMode::FilledMeteorology),
        ChromeScale::default(),
    );
    let nx = 14usize;
    let ny = 10usize;
    let grid = ProjectedGrid {
        x: vec![0.0; nx * ny],
        y: vec![0.0; nx * ny],
        ny,
        nx,
    };
    let mut pixel_points = Vec::with_capacity(nx * ny);
    for j in 0..ny {
        for i in 0..nx {
            pixel_points.push(Some((
                28.0 + i as f64 * 12.0 + j as f64 * 0.5,
                10.0 + j as f64 * 8.0,
            )));
        }
    }
    let pixel_points: Arc<[Option<(f64, f64)>]> = pixel_points.into();
    let rect = compute_domain_frame_rect(
        sample_domain_frame(crate::request::Color::BLACK),
        layout.map_w,
        layout.map_h,
    )
    .expect("test layout should produce a frame rect");
    (layout, grid, pixel_points, rect)
}

#[test]
fn bucketed_contours_match_legacy_for_sorted_levels() {
    let layout = contour_test_layout();
    let overlay = ContourOverlay {
        data: vec![0.0, 1.0, 2.0, 1.0, 2.0, 3.0, 2.0, 3.0, 4.0],
        ny: 3,
        nx: 3,
        levels: vec![1.5, 2.5],
        color: Rgba::BLACK,
        width: 1,
        labels: false,
        show_extrema: false,
        pattern: crate::request::ContourLinePattern::Solid,
        major_every: None,
        major_width: None,
    };

    let mut legacy = blank_test_image();
    let mut bucketed = blank_test_image();
    let mut legacy_labels = ContourLabelPlacer::default();
    let mut bucketed_labels = ContourLabelPlacer::default();
    draw_contours_legacy(
        &mut legacy,
        &layout,
        &overlay,
        None,
        None,
        &mut legacy_labels,
        1,
    );
    draw_contours_bucketed(
        &mut bucketed,
        &layout,
        &overlay,
        None,
        None,
        &mut bucketed_labels,
        1,
    );

    assert_eq!(legacy, bucketed);
}

#[test]
fn bucketed_contours_match_legacy_with_nan_corner() {
    let layout = contour_test_layout();
    let overlay = ContourOverlay {
        data: vec![0.0, 1.0, f64::NAN, 3.0],
        ny: 2,
        nx: 2,
        levels: vec![0.5, 1.5, 2.5],
        color: Rgba::BLACK,
        width: 1,
        labels: false,
        show_extrema: false,
        pattern: crate::request::ContourLinePattern::Solid,
        major_every: None,
        major_width: None,
    };

    let mut legacy = blank_test_image();
    let mut bucketed = blank_test_image();
    let mut legacy_labels = ContourLabelPlacer::default();
    let mut bucketed_labels = ContourLabelPlacer::default();
    draw_contours_legacy(
        &mut legacy,
        &layout,
        &overlay,
        None,
        None,
        &mut legacy_labels,
        1,
    );
    draw_contours_bucketed(
        &mut bucketed,
        &layout,
        &overlay,
        None,
        None,
        &mut bucketed_labels,
        1,
    );

    assert_eq!(legacy, bucketed);
}

#[test]
fn contour_label_placer_rejects_overlapping_labels() {
    let mut placer = ContourLabelPlacer::default();
    assert!(placer.can_place(LabelRect {
        min_x: 20,
        max_x: 70,
        min_y: 20,
        max_y: 34,
    }));
    assert!(!placer.can_place(LabelRect {
        min_x: 68,
        max_x: 110,
        min_y: 21,
        max_y: 35,
    }));
    assert!(placer.can_place(LabelRect {
        min_x: 120,
        max_x: 160,
        min_y: 21,
        max_y: 35,
    }));
}

#[test]
fn contour_label_state_allows_repeated_spaced_labels_per_level() {
    let mut layout = contour_test_layout();
    layout.map_w = 1500;
    layout.map_h = 850;
    let mut state = ContourLevelLabelState::new(true, &layout);

    assert!(state.max_labels > 1);
    assert!(state.can_try_at((100.0, 100.0)));
    state.record((100.0, 100.0));
    assert!(!state.can_try_at((130.0, 120.0)));
    assert!(state.can_try_at((420.0, 100.0)));

    while state.centers.len() < state.max_labels {
        let x = 100.0 + state.centers.len() as f64 * 260.0;
        assert!(state.can_try_at((x, 650.0)));
        state.record((x, 650.0));
    }
    assert!(!state.can_try_at((5000.0, 5000.0)));
}

#[test]
fn contour_labels_only_use_major_levels_when_configured() {
    let overlay = ContourOverlay {
        data: Vec::new(),
        ny: 0,
        nx: 0,
        levels: vec![540.0, 546.0, 552.0, 558.0],
        color: Rgba::BLACK,
        width: 1,
        labels: true,
        show_extrema: false,
        pattern: crate::request::ContourLinePattern::Solid,
        major_every: Some(2),
        major_width: Some(2),
    };

    assert!(contour_level_gets_label(&overlay, 0));
    assert!(!contour_level_gets_label(&overlay, 1));
    assert!(contour_level_gets_label(&overlay, 2));
    assert!(!contour_level_gets_label(&overlay, 3));
}

#[test]
fn contour_stroke_supports_major_width_and_dashes() {
    let overlay = ContourOverlay {
        data: Vec::new(),
        ny: 0,
        nx: 0,
        levels: vec![1000.0, 1002.0, 1004.0],
        color: Rgba::BLACK,
        width: 1,
        labels: false,
        show_extrema: false,
        pattern: crate::request::ContourLinePattern::Solid,
        major_every: Some(2),
        major_width: Some(3),
    };
    assert_eq!(contour_level_width(&overlay, 0), 3);
    assert_eq!(contour_level_width(&overlay, 1), 1);
    assert_eq!(contour_level_width(&overlay, 2), 3);

    let mut solid = blank_test_image();
    let mut dashed = blank_test_image();
    draw_contour_stroke(
        &mut solid,
        5.0,
        40.0,
        75.0,
        40.0,
        Rgba::BLACK,
        1,
        crate::request::ContourLinePattern::Solid,
    );
    draw_contour_stroke(
        &mut dashed,
        5.0,
        40.0,
        75.0,
        40.0,
        Rgba::BLACK,
        1,
        crate::request::ContourLinePattern::Dashed,
    );
    let solid_pixels = solid
        .pixels()
        .filter(|pixel| pixel.0 != [255, 255, 255, 255])
        .count();
    let dashed_pixels = dashed
        .pixels()
        .filter(|pixel| pixel.0 != [255, 255, 255, 255])
        .count();

    assert!(dashed_pixels > 0);
    assert!(dashed_pixels < solid_pixels);
}

#[test]
fn domain_frame_uses_viewport_when_fill_is_fully_masked() {
    let mut opts = sample_projected_opts();
    opts.cmap = sample_masked_cmap();
    opts.title = None;
    opts.domain_frame = Some(sample_domain_frame(crate::request::Color::rgba(
        250, 10, 10, 255,
    )));

    let data = [0.0f64; 4];
    let (image, timing) = render_to_image_profile(&data, 2, 2, &opts);
    let outline_pixels = image
        .pixels()
        .filter(|px| px.0[0] > 180 && px.0[1] < 120 && px.0[2] < 120)
        .count();

    assert!(
        outline_pixels > 0,
        "domain frame should still render when fill alpha is empty"
    );
    assert_eq!(
        timing.domain_clip_rect,
        Some([
            5,
            timing.map_w.saturating_sub(7),
            5,
            timing.map_h.saturating_sub(7)
        ]),
        "domain frame should follow the map viewport, not the data coverage"
    );
}

#[test]
fn domain_frame_clears_map_outside_rect() {
    let (layout, _, _, rect) = slanted_projected_fixture();
    let presentation = RenderPresentation::for_mode(ProductVisualMode::FilledMeteorology);
    let mut img = RgbaImage::from_pixel(320, 240, presentation.canvas_background.to_image_rgba());

    for py in layout.map_y..layout.map_y + layout.map_h {
        for px in layout.map_x..layout.map_x + layout.map_w {
            img.put_pixel(px, py, presentation.map_background.to_image_rgba());
        }
    }

    let outside_x = layout.map_x + rect.min_x.saturating_sub(1);
    let outside_y = layout.map_y + rect.min_y;
    let inside_x = layout.map_x + rect.min_x + 1;
    let inside_y = layout.map_y + rect.min_y + 1;
    img.put_pixel(outside_x, outside_y, Rgba::BLACK.to_image_rgba());

    clear_map_outside_local_rect(&mut img, &layout, rect, presentation.canvas_background);

    assert_eq!(
        img.get_pixel(outside_x, outside_y).0,
        presentation.canvas_background.to_image_rgba().0
    );
    assert_eq!(
        img.get_pixel(inside_x, inside_y).0,
        presentation.map_background.to_image_rgba().0
    );
}

#[test]
fn domain_frame_keeps_colorbar_in_layout_when_frame_matches_viewport() {
    let (layout, _, _, rect) = slanted_projected_fixture();
    let frame = sample_domain_frame(crate::request::Color::BLACK);

    let (_, cbar_y, _) = colorbar_anchor_rect(
        &layout,
        ColorbarOrientation::HorizontalBottom,
        Some(frame),
        Some(rect),
    );

    assert_eq!(cbar_y, layout.cbar_y);
}

#[test]
fn domain_frame_layout_reserves_space_for_legend_labels() {
    let layout = compute_effective_layout(
        1400,
        1100,
        true,
        true,
        RenderPresentation::for_mode(ProductVisualMode::OverlayAnalysis),
        ChromeScale::Fixed(1.0),
        true,
    );

    let label_top = layout.cbar_y.saturating_sub(layout.label_gap);
    let map_bottom = layout.map_y.saturating_add(layout.map_h).saturating_sub(1);
    assert!(layout.label_gap > text::regular_line_height(layout.text_scale));
    assert!(label_top > map_bottom);
}

#[test]
fn domain_frame_text_anchors_to_rect() {
    let (layout, _, _, rect) = slanted_projected_fixture();
    let frame = sample_domain_frame(crate::request::Color::BLACK);

    let (left, right, center) = chrome_anchor_bounds(&layout, Some(frame), Some(rect));

    assert_eq!(left, layout.map_x + rect.min_x);
    assert_eq!(right, layout.map_x + rect.max_x);
    assert_eq!(center, left + right.saturating_sub(left) / 2);
    assert_ne!(left, layout.map_x);
    assert_ne!(right, layout.map_x + layout.map_w);
}

#[test]
fn domain_frame_text_rows_anchor_just_above_rect() {
    let (layout, _, _, rect) = slanted_projected_fixture();
    let frame = sample_domain_frame(crate::request::Color::BLACK);

    let (title_y, subtitle_y) = chrome_anchor_rows(&layout, Some(frame), Some(rect));
    let frame_top = layout.map_y + rect.min_y;
    let max_gap = text::bold_line_height(layout.text_scale)
        .saturating_add(text::regular_line_height(layout.text_scale))
        .saturating_add(8u32.saturating_mul(layout.text_scale.max(1)));

    assert!(title_y <= subtitle_y);
    assert!(subtitle_y < frame_top);
    assert!(title_y < frame_top);
    assert!(frame_top.saturating_sub(title_y) <= max_gap);
}

#[test]
fn chrome_metadata_uses_space_left_by_short_title() {
    let metadata = "Init 05/04 11Z | F008 | Valid 05/04 19Z | HRRR | source: nomads";

    let (title, fitted_metadata) =
        fit_chrome_title_metadata(Some("2m AGL Temperature"), Some(metadata), 940, 14, 1);

    assert_eq!(title.as_deref(), Some("2m AGL Temperature"));
    assert_eq!(fitted_metadata.as_deref(), Some(metadata));
}

#[test]
fn projected_alpha_mask_clears_linework_outside_mask() {
    let layout = Layout {
        map_x: 1,
        map_y: 1,
        map_w: 4,
        map_h: 4,
        halo: Rgba::WHITE,
        title_factor: 1.0,
        label_factor: 1.0,
        cbar_x: 0,
        cbar_y: 0,
        cbar_w: 0,
        cbar_h: 0,
        title_y: 0,
        subtitle_y: 0,
        text_scale: 1,
        label_gap: 1,
        plan: None,
    };
    let bg = Rgba::new(244, 246, 248);
    let mut img = RgbaImage::from_pixel(6, 6, Rgba::BLACK.to_image_rgba());
    let mut mask = RgbaImage::new(4, 4);
    for y in 1..3 {
        for x in 1..3 {
            mask.put_pixel(x, y, Rgba::WHITE.to_image_rgba());
        }
    }

    clear_map_outside_local_mask(&mut img, &layout, &mask, bg);

    assert_eq!(img.get_pixel(1, 1).0, bg.to_image_rgba().0);
    assert_eq!(img.get_pixel(2, 2).0, Rgba::BLACK.to_image_rgba().0);
}

#[test]
fn trim_vertical_canvas_whitespace_crops_outer_blank_rows() {
    let mut img = RgbaImage::from_pixel(
        6,
        10,
        RenderPresentation::for_mode(ProductVisualMode::FilledMeteorology)
            .canvas_background
            .to_image_rgba(),
    );
    for y in 3..7 {
        for x in 0..6 {
            img.put_pixel(x, y, Rgba::BLACK.to_image_rgba());
        }
    }

    let (trimmed, crop_top) = trim_vertical_canvas_whitespace(
        &img,
        RenderPresentation::for_mode(ProductVisualMode::FilledMeteorology).canvas_background,
    );

    assert_eq!(trimmed.width(), 6);
    assert!(trimmed.height() < 10);
    assert!(trimmed.height() >= 4);
    // Content starts at row 3 and the pass keeps a 2-row pad, so exactly
    // one row came off the top -- the offset the plot rect follows.
    assert_eq!(crop_top, 1);
}

#[test]
fn center_horizontal_canvas_content_balances_outer_margins() {
    let bg = RenderPresentation::for_mode(ProductVisualMode::FilledMeteorology).canvas_background;
    let mut img = RgbaImage::from_pixel(12, 4, bg.to_image_rgba());
    for x in 1..7 {
        img.put_pixel(x, 1, Rgba::BLACK.to_image_rgba());
    }

    let (centered, shift) = center_horizontal_canvas_content(&img, bg);
    let mut min_x = centered.width();
    let mut max_x = 0;
    for y in 0..centered.height() {
        for x in 0..centered.width() {
            if !pixel_matches_background(*centered.get_pixel(x, y), bg) {
                min_x = min_x.min(x);
                max_x = max_x.max(x);
            }
        }
    }
    let left_margin = min_x;
    let right_margin = centered.width().saturating_sub(max_x).saturating_sub(1);

    assert!(left_margin.abs_diff(right_margin) <= 1);
    // Content spans x 1..=6 in a 12-wide canvas: margins 1 and 5, so the
    // pass moves everything 2 px right and must SAY so.
    assert_eq!(shift, 2);
}

#[test]
fn crop_canvas_whitespace_removes_blank_border() {
    let bg = RenderPresentation::for_mode(ProductVisualMode::FilledMeteorology).canvas_background;
    let mut img = RgbaImage::from_pixel(12, 10, bg.to_image_rgba());
    for y in 3..7 {
        for x in 4..9 {
            img.put_pixel(x, y, Rgba::BLACK.to_image_rgba());
        }
    }

    let (cropped, (crop_left, crop_top)) = crop_canvas_whitespace(&img, bg, 1);

    assert_eq!(cropped.width(), 7);
    assert_eq!(cropped.height(), 6);
    // Content starts at (4, 3) and the pass keeps a 1-px pad: 3 columns
    // and 2 rows came off -- the offsets the plot rect follows.
    assert_eq!((crop_left, crop_top), (3, 2));
}

#[test]
fn bucketed_contours_match_legacy_when_projected_corner_is_missing() {
    let layout = contour_test_layout();
    let overlay = ContourOverlay {
        data: vec![0.0, 1.0, 2.0, 3.0],
        ny: 2,
        nx: 2,
        levels: vec![1.5],
        color: Rgba::BLACK,
        width: 1,
        labels: false,
        show_extrema: false,
        pattern: crate::request::ContourLinePattern::Solid,
        major_every: None,
        major_width: None,
    };
    let pixel_points = vec![
        Some((0.0, 0.0)),
        None,
        Some((64.0, 64.0)),
        Some((0.0, 64.0)),
    ];

    let mut legacy = blank_test_image();
    let mut bucketed = blank_test_image();
    let mut legacy_labels = ContourLabelPlacer::default();
    let mut bucketed_labels = ContourLabelPlacer::default();
    draw_contours_legacy(
        &mut legacy,
        &layout,
        &overlay,
        Some(&pixel_points),
        None,
        &mut legacy_labels,
        1,
    );
    draw_contours_bucketed(
        &mut bucketed,
        &layout,
        &overlay,
        Some(&pixel_points),
        None,
        &mut bucketed_labels,
        1,
    );

    assert_eq!(legacy, bucketed);
}

#[test]
fn contour_cells_reject_projected_seam_jumps() {
    let layout = contour_test_layout();
    let overlay = ContourOverlay {
        data: vec![0.0, 1.0, 2.0, 3.0],
        ny: 2,
        nx: 2,
        levels: vec![1.5],
        color: Rgba::BLACK,
        width: 1,
        labels: false,
        show_extrema: false,
        pattern: crate::request::ContourLinePattern::Solid,
        major_every: None,
        major_width: None,
    };
    let pixel_points = vec![
        Some((0.0, 0.0)),
        Some((1000.0, 0.0)),
        Some((0.0, 64.0)),
        Some((1000.0, 64.0)),
    ];

    assert!(contour_cell_corners(&layout, &overlay, Some(&pixel_points), 0, 1).is_none());
}

#[test]
fn projected_pixel_bilinear_rejects_projected_seam_jumps() {
    let pixel_points = vec![
        Some((0.0, 0.0)),
        Some((1000.0, 0.0)),
        Some((0.0, 64.0)),
        Some((1000.0, 64.0)),
    ];

    assert!(projected_pixel_bilinear(&pixel_points, 2, 2, 0.5, 0.5).is_none());
}

fn isobar_cell_overlay(data: Vec<f64>, nx: usize, ny: usize, levels: Vec<f64>) -> ContourOverlay {
    ContourOverlay {
        data,
        ny,
        nx,
        levels,
        color: Rgba::BLACK,
        width: 1,
        labels: false,
        show_extrema: false,
        pattern: crate::request::ContourLinePattern::Solid,
        major_every: None,
        major_width: None,
    }
}

/// U-E9 (2.9.9 R10 look): a small closed isobar on the R11 d01 MSLP/10 m
/// winds map had a one-cell gap with nothing in it.  The ring passed exactly
/// through a grid node (the node's MSLP equalled the 2 hPa level), the cell
/// counted that corner as a crossing on both of its edges, found three
/// points, and drew only the first pair: the corner to itself.  The segment
/// from that corner to the opposite edge was never drawn.
#[test]
fn contour_through_a_grid_node_keeps_the_segment_to_the_opposite_edge() {
    let layout = contour_test_layout();
    // Corners in walk order: c0 (0,0) below, c1 (1,0) exactly ON 1016,
    // c2 (1,1) and c3 (0,1) above.  The isobar runs from c1 to the c3-c0 edge.
    let overlay = isobar_cell_overlay(vec![1015.5, 1016.0, 1016.4, 1016.3], 2, 2, vec![1016.0]);

    let (pts, count, _) =
        contour_cell_intersections(&layout, &overlay, None, 0, 1, 1016.0).expect("the cell is crossed");
    assert_eq!(count, 2, "a node on the level is one crossing, not two: {pts:?}");
    let (c0, c1, _c2, c3) = contour_cell_corners(&layout, &overlay, None, 0, 1).unwrap();
    let t = (1016.0 - 1016.4) / (1015.5 - 1016.4);
    let a = (c3.0 + (c0.0 - c3.0) * t, c3.1 + (c0.1 - c3.1) * t);
    let close = |p: (f64, f64), q: (f64, f64)| (p.0 - q.0).abs() < 1e-9 && (p.1 - q.1).abs() < 1e-9;
    assert!(close(pts[0], c1) && close(pts[1], a), "segment is c1 -> the c3-c0 edge: {pts:?}");

    // And it is drawn: the middle of that segment is inked.
    let mut img = blank_test_image();
    let mut labels = ContourLabelPlacer::default();
    draw_contours_bucketed(&mut img, &layout, &overlay, None, None, &mut labels, 1);
    let mid = ((c1.0 + a.0) * 0.5, (c1.1 + a.1) * 0.5);
    let inked = (-1..=1).any(|dy| {
        (-1..=1).any(|dx| {
            let x = (mid.0.round() as i32 + dx) as u32;
            let y = (mid.1.round() as i32 + dy) as u32;
            img.get_pixel(x, y).0 != [255, 255, 255, 255]
        })
    });
    assert!(inked, "no ink at the middle of the isobar segment {mid:?}");
}

/// Every cell of a field whose nodes often sit exactly on a level (whole
/// hPa values, 2 hPa levels) has 0, 2 or 4 crossings, never 3.
#[test]
fn contour_cells_never_have_an_odd_crossing_count_when_nodes_sit_on_levels() {
    let layout = contour_test_layout();
    let (nx, ny) = (9usize, 9usize);
    let data: Vec<f64> = (0..ny)
        .flat_map(|j| (0..nx).map(move |i| 1010.0 + ((i * 7 + j * 3) % 9) as f64))
        .collect();
    let levels: Vec<f64> = (0..10).map(|k| 1008.0 + 2.0 * k as f64).collect();
    let overlay = isobar_cell_overlay(data, nx, ny, levels.clone());
    let mut on_level_cells = 0usize;
    for j in 0..ny - 1 {
        for i in 0..nx - 1 {
            let base = j * nx + i;
            for &level in &levels {
                let corners = [
                    overlay.data[base],
                    overlay.data[base + 1],
                    overlay.data[base + nx + 1],
                    overlay.data[base + nx],
                ];
                if corners.iter().any(|v| *v == level) {
                    on_level_cells += 1;
                }
                if let Some((pts, count, _)) =
                    contour_cell_intersections(&layout, &overlay, None, base, 1, level)
                {
                    assert!(
                        count == 2 || count == 4,
                        "cell ({i},{j}) level {level}: {count} crossings {pts:?}"
                    );
                }
            }
        }
    }
    assert!(on_level_cells > 20, "the field must exercise nodes on levels ({on_level_cells})");
}

#[test]
fn levels_are_sorted_finite_rejects_unsorted_or_nan_levels() {
    assert!(levels_are_sorted_finite(&[1.0, 2.0, 3.0]));
    assert!(!levels_are_sorted_finite(&[2.0, 1.0]));
    assert!(!levels_are_sorted_finite(&[1.0, f64::NAN, 3.0]));
}

#[test]
fn render_to_png_reuses_projected_pixel_cache_for_identical_meshes() {
    let _guard = PROJECTED_PIXEL_CACHE_TEST_LOCK.lock().unwrap();
    reset_projected_pixel_cache_for_tests();

    let data = [0.0, 1.0, 2.0, 3.0];
    let opts = sample_projected_opts();

    let first = render_to_png(&data, 2, 2, &opts);
    let second = render_to_png(&data, 2, 2, &opts);

    assert_eq!(first, second);
    assert_eq!(projected_pixel_cache_miss_count_for_tests(), 1);
}

#[test]
fn render_to_png_recomputes_projected_pixels_when_extent_changes() {
    let _guard = PROJECTED_PIXEL_CACHE_TEST_LOCK.lock().unwrap();
    reset_projected_pixel_cache_for_tests();

    let data = [0.0, 1.0, 2.0, 3.0];
    let opts = sample_projected_opts();
    let mut shifted = sample_projected_opts();
    shifted.map_extent = Some(MapExtent {
        x_min: -0.25,
        x_max: 0.75,
        y_min: 0.0,
        y_max: 1.0,
    });

    render_to_png(&data, 2, 2, &opts);
    render_to_png(&data, 2, 2, &shifted);

    assert_eq!(projected_pixel_cache_miss_count_for_tests(), 2);
}

#[test]
fn static_base_cache_key_changes_with_plot_style() {
    let opts = sample_projected_opts();
    let layout = compute_layout(
        opts.width,
        opts.height,
        opts.colorbar,
        opts.title.is_some(),
        opts.presentation,
        opts.chrome_scale,
    );
    let baseline_key = static_base_cache_key(
        &opts,
        &layout,
        opts.map_extent.as_ref(),
        None,
        opts.presentation.canvas_background,
        opts.presentation.map_background,
        true,
    );
    let mut clean_opts = opts.clone();
    clean_opts.presentation = RenderPresentation::for_mode_with_style(
        ProductVisualMode::FilledMeteorology,
        crate::presentation::StaticPlotStyle::CleanAtlasFast,
    );
    let clean_key = static_base_cache_key(
        &clean_opts,
        &layout,
        clean_opts.map_extent.as_ref(),
        None,
        clean_opts.presentation.canvas_background,
        clean_opts.presentation.map_background,
        true,
    );

    assert_ne!(baseline_key, clean_key);
}

#[test]
fn map_frame_aspect_ratio_matches_wide_render_layout() {
    let default_layout = compute_layout(
        1200,
        900,
        true,
        true,
        RenderPresentation::for_mode(ProductVisualMode::FilledMeteorology),
        ChromeScale::default(),
    );
    let ratio = default_layout.map_w as f64 / default_layout.map_h as f64;
    assert!(ratio > 1.35);
    assert!(ratio < 1.7);

    let operational_layout = compute_layout(
        1200,
        900,
        true,
        true,
        RenderPresentation::for_mode_with_style(
            ProductVisualMode::FilledMeteorology,
            crate::presentation::StaticPlotStyle::OperationalFast,
        ),
        ChromeScale::default(),
    );
    let operational_ratio = operational_layout.map_w as f64 / operational_layout.map_h as f64;
    assert!(operational_ratio > 1.2);
    assert!(operational_ratio < ratio);
}

#[test]
fn colorbar_tick_levels_follow_legend_levels_when_fill_is_densified() {
    let cmap = LeveledColormap::from_palette_with_options(
        &[Rgba::new(0, 0, 255), Rgba::new(255, 0, 0)],
        &[0.0, 10.0, 20.0, 30.0, 40.0],
        Extend::Neither,
        None,
        ColormapBuildOptions {
            render_density: crate::colormap::RenderDensity::default(),
            legend: crate::colormap::LegendControls {
                density: LevelDensity::default(),
                mode: crate::colormap::LegendMode::Stepped,
            },
        },
    );

    assert!(cmap.levels.len() > cmap.legend_levels.len());
    assert_eq!(
        colorbar_levels_for_ticks(&cmap),
        cmap.legend_levels.as_slice()
    );
}

#[test]
fn colorbar_tick_labels_clamp_to_requested_bounds() {
    let labels =
        filter_tick_labels_to_fit(&[0.0, 50.0, 100.0], 0.0, 100.0, 80, 120, 80, 200, 400, 1);
    assert!(!labels.is_empty());
    for (_, lx, label) in labels {
        let width = text::text_width(&label, 1) as i32;
        assert!(lx >= 80);
        assert!(lx + width <= 200);
    }
}

#[test]
fn extrema_selection_keeps_only_ranked_spaced_centers() {
    let mut layout = contour_test_layout();
    layout.map_w = 1500;
    layout.map_h = 850;
    let lows = vec![
        ExtremaCandidate {
            value: 1009.8,
            score: 1009.8,
            px: 120,
            py: 120,
        },
        ExtremaCandidate {
            value: 1008.2,
            score: 1008.2,
            px: 170,
            py: 145,
        },
        ExtremaCandidate {
            value: 1006.5,
            score: 1006.5,
            px: 760,
            py: 430,
        },
        ExtremaCandidate {
            value: 1004.1,
            score: 1004.1,
            px: 1260,
            py: 240,
        },
        ExtremaCandidate {
            value: 1003.9,
            score: 1003.9,
            px: 1320,
            py: 260,
        },
    ];

    let selected = select_extrema_labels(lows, false, &layout);

    assert_eq!(selected.len(), 3);
    assert!(selected.iter().any(|point| point.value == 1003.9));
    assert!(selected.iter().any(|point| point.value == 1006.5));
    assert!(!selected.iter().any(|point| point.value == 1004.1));
    assert!(!selected.iter().any(|point| point.value == 1009.8));
}

#[test]
fn contour_labels_scale_up_on_operational_sized_maps() {
    let small = contour_test_layout();
    let mut operational = contour_test_layout();
    operational.map_w = 1494;
    operational.map_h = 829;

    assert_eq!(contour_label_scale(&small), 1);
    assert_eq!(contour_label_scale(&operational), 2);
    assert_eq!(contour_label_halo_width(&operational), 2);
    assert!(contour_label_size_factor(&operational) < 1.0);
}

#[test]
fn extrema_analysis_grid_keeps_full_resolution_by_default() {
    assert_eq!(extrema_analysis_stride(100, 100), 1);
    assert_eq!(extrema_analysis_stride(1800, 1059), 1);

    let data = (0..36).map(|value| value as f64).collect::<Vec<_>>();
    let (analysis, nx, ny) = extrema_analysis_grid(&data, 6, 6, 2);

    assert_eq!((nx, ny), (3, 3));
    assert_eq!(analysis.len(), 9);
    assert!((analysis[0] - 3.5).abs() < 1.0e-9);
    assert!((analysis[8] - 31.5).abs() < 1.0e-9);
}

#[test]
fn chrome_scale_grows_layout_for_larger_outputs() {
    let base = compute_layout(
        1200,
        900,
        true,
        true,
        RenderPresentation::for_mode(ProductVisualMode::FilledMeteorology),
        ChromeScale::default(),
    );
    let bigger = compute_layout(
        2400,
        1800,
        true,
        true,
        RenderPresentation::for_mode(ProductVisualMode::FilledMeteorology),
        ChromeScale::default(),
    );

    assert!(bigger.cbar_h > base.cbar_h);
    assert!(bigger.text_scale > base.text_scale);
    assert!(bigger.label_gap > base.label_gap);
}

#[test]
fn filled_layout_keeps_header_and_legend_tight_to_map() {
    let layout = compute_layout(
        1200,
        900,
        true,
        true,
        RenderPresentation::for_mode(ProductVisualMode::FilledMeteorology),
        ChromeScale::Fixed(1.0),
    );

    assert_eq!(layout.map_y, 64);
    assert_eq!(layout.title_y, 5);
    assert!(layout.subtitle_y > layout.title_y);
    assert_eq!(layout.cbar_y + layout.cbar_h, 892);
}

#[test]
fn render_to_png_suppresses_barbs_when_overlay_data_is_nan() {
    // Updated expectation: barb overlays are no longer clipped to the fill
    // raster (that broke height-contour / wind-barb renders when the fill
    // used mask_below). Instead, barbs clip themselves via NaN u/v values.
    let _guard = PROJECTED_PIXEL_CACHE_TEST_LOCK.lock().unwrap();
    let mut opts = sample_projected_opts();
    opts.title = None;
    opts.barbs = vec![BarbOverlay {
        direction_projection: None,
        direction_arrows: false,
        u: vec![f32::NAN; 4],
        v: vec![f32::NAN; 4],
        ny: 2,
        nx: 2,
        stride_x: 1,
        stride_y: 1,
        spacing_px: 24.0,
        color: Rgba::BLACK,
        halo_color: Rgba::WHITE,
        halo_width: 2,
        width: 1,
        length_px: 12.0,
    }];

    let data = [0.5f64; 4];
    let png = render_to_png(&data, 2, 2, &opts);
    let image = image::load_from_memory_with_format(&png, image::ImageFormat::Png)
        .unwrap()
        .to_rgba8();
    // NaN u/v means no barb glyphs are drawn.
    let non_fill = image.pixels().filter(|px| px.0 == [0, 0, 0, 255]).count();
    assert_eq!(non_fill, 0, "NaN barb vectors should produce no glyphs");
}

#[test]
fn barb_glyph_margin_skips_map_edge_anchors() {
    assert!(
        !barb_glyph_fits_map_rect(10.0, 10.0, 100, 100, 18.0, 1),
        "edge anchors can draw outside the map frame"
    );
    assert!(
        barb_glyph_fits_map_rect(50.0, 50.0, 100, 100, 18.0, 1),
        "center anchors should still render"
    );
}

#[test]
fn render_to_png_suppresses_contours_when_overlay_data_is_nan() {
    // Updated expectation: contour overlays self-clip via NaN data, not via
    // the fill raster. Lets height contours render across the whole frame
    // even when the paired CAPE fill uses mask_below.
    let _guard = PROJECTED_PIXEL_CACHE_TEST_LOCK.lock().unwrap();
    let mut opts = sample_projected_opts();
    opts.title = None;
    opts.contours = vec![ContourOverlay {
        data: vec![f64::NAN; 4],
        ny: 2,
        nx: 2,
        levels: vec![1.5],
        color: Rgba::BLACK,
        width: 1,
        labels: false,
        show_extrema: false,
        pattern: crate::request::ContourLinePattern::Solid,
        major_every: None,
        major_width: None,
    }];

    let data = [0.5f64; 4];
    let png = render_to_png(&data, 2, 2, &opts);
    let image = image::load_from_memory_with_format(&png, image::ImageFormat::Png)
        .unwrap()
        .to_rgba8();
    let contour_pixels = image.pixels().filter(|px| px.0 == [0, 0, 0, 255]).count();
    assert_eq!(
        contour_pixels, 0,
        "NaN contour data should produce no contour lines"
    );
}

#[test]
fn crates_do_not_reintroduce_legacy_credit_footers() {
    let crates_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..");
    let forbidden = [
        String::from_utf8(vec![
            67, 111, 108, 111, 114, 32, 84, 97, 98, 108, 101, 115, 58, 32, 83, 111, 108, 97, 114,
            112, 111, 119, 101, 114, 48, 55,
        ])
        .expect("legacy footer bytes should be valid utf-8"),
        ["Pivotal", " Weather"].concat(),
        ["Weather", "Bell"].concat(),
    ];
    let mut offenders = Vec::<String>::new();
    visit_rs_files(&crates_root, &mut |path| {
        if let Ok(contents) = std::fs::read_to_string(path) {
            for term in &forbidden {
                if contents.contains(term) {
                    offenders.push(format!("{} => {}", path.display(), term));
                }
            }
        }
    })
    .expect("crate source tree should be readable");
    assert!(
        offenders.is_empty(),
        "legacy credit/footer strings remain in crates/: {offenders:?}"
    );
}


#[test]
fn subtitle_left_keeps_what_the_source_label_does_not_need() {
    // B-13.  The split was `subtitle_available / 2` whenever a right
    // subtitle existed, so the provenance line was ellipsized to half
    // the row while "source: ArWen" -- a small fraction of it -- left the
    // rest of the row empty.  Measured on a rendered plot the line came
    // out `Init 04/03 12Z | F006 | Valid 04/03 18Z | WRF...`, and what
    // was dropped was the grid spacing.
    let scale = 2u32;
    let gap = 6u32 * scale;
    let right = "source: ArWen";
    let provenance = "Init 04/03 12Z | F006 | Valid 04/03 18Z | WRF | dx 12 km";

    // Widths are measured, not guessed: the row is exactly wide enough
    // for both, which is the case the old rule got wrong.
    let prov_w = measure_text_width(provenance, scale, false);
    let right_w = measure_text_width(right, scale, false);
    let available = prov_w + right_w + gap;

    let left_width = subtitle_left_width(available, Some(right), scale, gap);
    assert_eq!(left_width, prov_w);
    let (fitted, truncated) = fit_text_to_width(provenance, left_width, scale, false);
    assert!(!truncated, "still truncated: {fitted:?}");
    assert_eq!(fitted, provenance);

    // Negative control: the old fixed half-split cuts the same line, so
    // this test fails if the split ever goes back to it.
    assert!(available / 2 < prov_w, "fixture cannot demonstrate B-13");
    let (old_fitted, old_truncated) =
        fit_text_to_width(provenance, available / 2, scale, false);
    assert!(old_truncated);
    assert!(old_fitted.ends_with("..."));
}

#[test]
fn subtitle_left_never_falls_below_the_even_split() {
    // A pathological source label must not squeeze the provenance below
    // what the old fixed split gave it.
    let scale = 2u32;
    let available = 400u32;
    let absurd = "source: ".to_string() + &"x".repeat(400);
    let left_width = subtitle_left_width(available, Some(absurd.as_str()), scale, 12);
    assert_eq!(left_width, available - available / 2);
}

#[test]
fn a_subtitle_that_does_not_fit_reports_that_it_was_cut() {
    // The channel B-13 said was missing: the plain helper returns a bare
    // String, so a caller could not tell a line that fitted from one
    // that lost its tail.
    let scale = 2u32;
    let text = "Init 04/03 12Z | F006 | Valid 04/03 18Z | WRF | dx 12 km";
    let (fitted, truncated) = fit_text_to_width(text, 60, scale, false);
    assert!(truncated);
    assert!(fitted.ends_with("..."));
    let (whole, untruncated) = fit_text_to_width(text, 100_000, scale, false);
    assert!(!untruncated);
    assert_eq!(whole, text);
}

const NEST_SUBTITLE: &str = "Init 09/27 12Z | +000:15 | Valid 09/27 12:15Z | WRF | dx 1 km";
const NEST_SOURCE: &str = "source: ArWen";

fn operational_header_opts(width: u32, height: u32, subtitle_left: &str, colorbar: bool) -> RenderOpts {
    let mut opts = RenderOpts::default();
    opts.width = width;
    opts.height = height;
    opts.colorbar = colorbar;
    opts.cmap = sample_cmap();
    opts.colorbar_units = Some("degF".into());
    opts.presentation = RenderPresentation::for_mode_with_style(
        ProductVisualMode::FilledMeteorology,
        StaticPlotStyle::Operational,
    );
    // A title with no descenders, so nothing of it reaches the subtitle row.
    opts.title = Some("Bulk Shear 0-6 km".into());
    opts.subtitle_left = Some(subtitle_left.into());
    opts.subtitle_right = Some(NEST_SOURCE.into());
    opts.domain_frame = Some(DomainFrame::model_data_default());
    opts
}

/// The chrome of `opts` over a full-height frame `frame_w` pixels wide in
/// the middle of the map: a tall nest on a wide canvas.
fn draw_narrow_frame_header(opts: &RenderOpts, frame_w: u32) -> (RgbaImage, Layout, LocalRect) {
    let layout = compute_layout(
        opts.width,
        opts.height,
        opts.colorbar,
        true,
        opts.presentation,
        opts.chrome_scale,
    );
    let left = layout.map_w.saturating_sub(frame_w) / 2;
    let rect = LocalRect {
        min_x: left,
        max_x: left + frame_w,
        min_y: 0,
        max_y: layout.map_h.saturating_sub(1),
    };
    let mut img = RgbaImage::from_pixel(opts.width, opts.height, Rgba::WHITE.to_image_rgba());
    draw_chrome_and_colorbar(&mut img, &layout, opts, None, Some(rect), None, false, true);
    (img, layout, rect)
}

/// The width the left subtitle gets on a header row `row_width` wide.
fn left_subtitle_room(layout: &Layout, row_width: u32) -> u32 {
    let scale = layout.text_scale;
    subtitle_left_width(row_width.saturating_sub(18 * scale), Some(NEST_SOURCE), scale, 6 * scale)
}

/// Whether `img` shows `text` drawn from `x`, `y`: every pixel the text
/// alone paints there is painted the same in `img`.
fn shows_text_at(img: &RgbaImage, opts: &RenderOpts, layout: &Layout, text: &str, x: u32, y: u32) -> bool {
    let mut alone = RgbaImage::from_pixel(opts.width, opts.height, Rgba::WHITE.to_image_rgba());
    text::draw_text_with_factor(
        &mut alone,
        text,
        x as i32,
        y as i32,
        opts.presentation.chrome.subtitle_color,
        layout.text_scale,
        layout.label_factor,
    );
    let width = measure_text_width_with_factor(text, layout.text_scale, layout.label_factor, false);
    let height = text::regular_line_height(layout.text_scale);
    let expected = crop_imm(&alone, x, y, width, height).to_image();
    assert!(
        expected.pixels().any(|pixel| pixel.0 != [255, 255, 255, 255]),
        "the reference text drew nothing"
    );
    let actual = crop_imm(img, x, y, width, height).to_image();
    expected
        .enumerate_pixels()
        .filter(|(_, _, pixel)| pixel.0 != [255, 255, 255, 255])
        .all(|(px, py, pixel)| actual.get_pixel(px, py) == pixel)
}

/// Where on `row` the whole `text` is drawn, searching from the canvas
/// row's left edge to `last_x`.
fn whole_text_x(img: &RgbaImage, opts: &RenderOpts, layout: &Layout, text: &str, row: u32, last_x: u32) -> Option<u32> {
    (layout.map_x..=last_x).find(|&x| shows_text_at(img, opts, layout, text, x, row))
}

#[test]
fn a_narrow_nest_keeps_its_valid_time_on_a_wide_canvas() {
    // A tall nest drawn at 1200x900 or 2400x900 gets a frame about 380 px
    // wide at either width.  Anchored to that frame the subtitle came out
    // "Init 09/27 12Z | +000:15 | Vali..." with most of the header row
    // unused, and widening the image could not help.  The header row
    // widens, centred on the frame, until its text fits.
    for width in [1200u32, 2400] {
        let opts = operational_header_opts(width, 900, NEST_SUBTITLE, false);
        let (img, layout, rect) = draw_narrow_frame_header(&opts, 380);
        let (_, row) = chrome_anchor_rows(&layout, opts.domain_frame, Some(rect));
        let frame_left = layout.map_x + rect.min_x;
        let needed = measure_text_width(NEST_SUBTITLE, layout.text_scale, false);
        assert!(needed > left_subtitle_room(&layout, rect.max_x - rect.min_x), "{width}: must not fit the frame");
        assert!(needed <= left_subtitle_room(&layout, layout.map_w), "{width}: must fit the canvas row");
        let x = whole_text_x(&img, &opts, &layout, NEST_SUBTITLE, row, frame_left);
        assert!(x.is_some(), "{width}x900: the whole subtitle, valid time included, must be drawn");
        // Only as wide as its text needs, centred on the frame: it starts
        // left of the map, never at the canvas's edge when there is room.
        let x = x.unwrap();
        assert!(x > layout.map_x && x < frame_left, "{width}: the header hugs its map, at {x}");
    }
}

#[test]
fn a_narrow_nest_header_stops_at_the_colour_bar_beside_it() {
    // The operational colour bar follows the frame and stands just right of
    // it, inside the canvas row, so the widened header grows to the left of
    // the frame and ends at the frame's right edge.
    let opts = operational_header_opts(1200, 900, NEST_SUBTITLE, true);
    let (img, layout, rect) = draw_narrow_frame_header(&opts, 380);
    let (_, row) = chrome_anchor_rows(&layout, opts.domain_frame, Some(rect));
    let frame_left = layout.map_x + rect.min_x;
    let frame_right = layout.map_x + rect.max_x;
    let (bar_x, _, _) = colorbar_anchor_rect(
        &layout,
        ColorbarOrientation::VerticalRight,
        opts.domain_frame,
        Some(rect),
    );
    assert!(bar_x > frame_right && bar_x < layout.map_x + layout.map_w, "the bar stands beside the frame");
    let needed = measure_text_width(NEST_SUBTITLE, layout.text_scale, false);
    assert!(needed > left_subtitle_room(&layout, rect.max_x - rect.min_x), "must not fit the frame");
    assert!(needed <= left_subtitle_room(&layout, frame_right - layout.map_x), "must fit up to the frame's edge");
    assert!(
        whole_text_x(&img, &opts, &layout, NEST_SUBTITLE, row, frame_left).is_some(),
        "the whole subtitle, valid time included, must be drawn"
    );
    let source_w = measure_text_width_with_factor(NEST_SOURCE, layout.text_scale, layout.label_factor, false);
    assert!(
        shows_text_at(&img, &opts, &layout, NEST_SOURCE, frame_right - source_w, row),
        "the source label ends at the frame's right edge, clear of the bar"
    );
}

#[test]
fn a_subtitle_that_fits_over_its_frame_stays_anchored_to_it() {
    let opts = operational_header_opts(1200, 900, "Valid 09/27 12:15Z", true);
    let (img, layout, rect) = draw_narrow_frame_header(&opts, 380);
    let (left, _, _) = chrome_anchor_bounds(&layout, opts.domain_frame, Some(rect));
    let (_, row) = chrome_anchor_rows(&layout, opts.domain_frame, Some(rect));
    assert!(left > layout.map_x, "the frame must sit inside the canvas row");
    assert!(shows_text_at(&img, &opts, &layout, "Valid 09/27 12:15Z", left, row));
}

/// A regular latitude/longitude mesh, projected into the presentation
/// projection the direct lane picks for a CONUS-sized window.
///
/// This is the geometry every lat/lon SOURCE renders in -- GFS, GDAS,
/// GEFS, ICON, the AI atmospheres -- once a regional window makes the
/// adaptive presentation conic.  Its screen footprint is a curved
/// quadrilateral, not a rectangle.
fn curved_latlon_footprint(map_w: u32, map_h: u32) -> (ProjectedGrid, Vec<Option<(f32, f32)>>) {
    let (ny, nx) = (36usize, 71usize);
    let projector = crate::projection::ProjectionSpec::LambertConformal {
        standard_parallel_1_deg: 25.833_333_333_333_332,
        standard_parallel_2_deg: 49.166_666_666_666_664,
        central_meridian_deg: -95.0,
    }
    .build_projector(Some(39.0), None, &[20.0f32], &[-95.0f32])
    .expect("lambert projector");

    let mut x = Vec::with_capacity(ny * nx);
    let mut y = Vec::with_capacity(ny * nx);
    for j in 0..ny {
        for i in 0..nx {
            let lat = 20.0 + j as f64;
            let lon = -130.0 + i as f64;
            let (px, py) = projector.project(lat, lon);
            x.push(px);
            y.push(py);
        }
    }
    let (mut x0, mut x1, mut y0, mut y1) = (
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
    );
    for (&px, &py) in x.iter().zip(y.iter()) {
        x0 = x0.min(px);
        x1 = x1.max(px);
        y0 = y0.min(py);
        y1 = y1.max(py);
    }
    let extent = MapExtent {
        x_min: x0,
        x_max: x1,
        y_min: y0,
        y_max: y1,
    };
    let pixel_points = x
        .iter()
        .zip(y.iter())
        .map(|(&px, &py)| {
            extent
                .to_pixel(px, py, map_w, map_h)
                .map(|(a, b)| (a as f32, b as f32))
        })
        .collect::<Vec<_>>();
    (ProjectedGrid { x, y, ny, nx }, pixel_points)
}

/// A native projected grid -- a wrfout on its own Lambert -- is an
/// axis-aligned rectangle on screen.
fn rectangular_footprint(map_w: u32, map_h: u32) -> (ProjectedGrid, Vec<Option<(f32, f32)>>) {
    let (ny, nx) = (36usize, 71usize);
    let mut x = Vec::with_capacity(ny * nx);
    let mut y = Vec::with_capacity(ny * nx);
    for j in 0..ny {
        for i in 0..nx {
            x.push(i as f64);
            y.push(j as f64);
        }
    }
    let extent = MapExtent {
        x_min: 0.0,
        x_max: (nx - 1) as f64,
        y_min: 0.0,
        y_max: (ny - 1) as f64,
    };
    let pixel_points = x
        .iter()
        .zip(y.iter())
        .map(|(&px, &py)| {
            extent
                .to_pixel(px, py, map_w, map_h)
                .map(|(a, b)| (a as f32, b as f32))
        })
        .collect::<Vec<_>>();
    (ProjectedGrid { x, y, ny, nx }, pixel_points)
}

fn covered_pixel_bounds(
    grid: &ProjectedGrid,
    pixels: &[Option<(f32, f32)>],
    w: u32,
    h: u32,
) -> LocalRect {
    let mask = crate::rasterize::rasterize_projected_coverage_mask(grid.ny, grid.nx, pixels, w, h);
    LocalRect::from_bounds(raster_alpha_bounds(&mask).expect("some coverage"))
}

#[test]
fn resampled_lat_lon_domain_frame_holds_every_drawn_row() {
    // The defect: `inner_rect_from_coverage` returns the largest
    // rectangle INSCRIBED in the coverage, which for a curved footprint
    // is a thin horizontal band -- and `clear_outside` then erases the
    // rest of the domain.  The aigfs lane saw it as "rw_wrfbatch draws
    // only about half the south_north rows".
    let (map_w, map_h) = (1200u32, 800u32);
    let (grid, pixels) = curved_latlon_footprint(map_w, map_h);
    let frame = DomainFrame {
        inset_px: 2,
        outline_width: 2,
        source: DomainFrameSource::ProjectedGrid,
        ..DomainFrame::map_viewport_default()
    };

    let covered = covered_pixel_bounds(&grid, &pixels, map_w, map_h);

    // The mechanism, pinned so it cannot come back by accident: asked to
    // INSCRIBE, the same coverage yields a band a fraction of the domain
    // tall.  This is the arm that shipped, and what it drops is model
    // data that `clear_outside` then erases.
    let inscribed =
        compute_projected_domain_frame_rect(frame, &grid, &pixels, map_w, map_h, 0, true)
            .expect("a covered domain has a frame");
    assert!(
        inscribed.height() * 2 < covered.height(),
        "the inscribed rectangle of a curved footprint is a band: {} of \
         {} rows",
        inscribed.height(),
        covered.height()
    );

    let rect = compute_projected_domain_frame_rect(frame, &grid, &pixels, map_w, map_h, 0, false)
        .expect("a covered domain has a frame");

    assert_eq!(
        rect,
        inset_rect(covered, frame.inset_px).expect("inset fits"),
        "a resampled lat/lon field's frame is the box that holds every \
         drawn row, not the widest band inscribed inside the curve"
    );
    assert!(
        rect.height() * 100 >= covered.height() * 95,
        "frame kept {} of {} covered rows",
        rect.height(),
        covered.height()
    );
}

#[test]
fn native_projected_domain_frame_still_inscribes_the_rectangle() {
    // The unchanged half of the contract: a wrfout on its own projection
    // is a screen rectangle, so inscribing and bounding agree, and the
    // inscribing path stays exactly where it was.
    let (map_w, map_h) = (1200u32, 800u32);
    let (grid, pixels) = rectangular_footprint(map_w, map_h);
    let frame = DomainFrame {
        inset_px: 2,
        outline_width: 2,
        source: DomainFrameSource::ProjectedGrid,
        ..DomainFrame::map_viewport_default()
    };

    let inscribed =
        compute_projected_domain_frame_rect(frame, &grid, &pixels, map_w, map_h, 0, true)
            .expect("frame");
    let bounded =
        compute_projected_domain_frame_rect(frame, &grid, &pixels, map_w, map_h, 0, false)
            .expect("frame");

    assert_eq!(inscribed, bounded);
}

// A 480 px map inside a 560 x 600 canvas, the extent spanning 0..1 on both
// axes: one extent unit is 479 px, so y = -40/479 lies 40 px below the map's
// bottom row, inside the 10 % slack `MapExtent::to_pixel` keeps for lines
// that leave the map.
fn frame_test_layout() -> Layout {
    Layout {
        map_x: 40,
        map_y: 40,
        map_w: 480,
        map_h: 480,
        ..contour_test_layout()
    }
}

fn unit_extent() -> MapExtent {
    MapExtent {
        x_min: 0.0,
        x_max: 1.0,
        y_min: 0.0,
        y_max: 1.0,
    }
}

fn test_polyline(points: Vec<(f64, f64)>, width: u32) -> ProjectedPolyline {
    ProjectedPolyline {
        points,
        color: Rgba::BLACK,
        width,
        role: crate::presentation::LineworkRole::Generic,
    }
}

fn is_ink(pixel: &image::Rgba<u8>) -> bool {
    pixel.0 != [255, 255, 255, 255]
}

fn ink_outside_map(img: &RgbaImage, layout: &Layout) -> Vec<(u32, u32)> {
    let right = layout.map_x + layout.map_w;
    let bottom = layout.map_y + layout.map_h;
    img.enumerate_pixels()
        .filter(|(x, y, pixel)| {
            let inside = *x >= layout.map_x && *x < right && *y >= layout.map_y && *y < bottom;
            !inside && is_ink(pixel)
        })
        .map(|(x, y, _)| (x, y))
        .collect()
}

/// Ink in the map's last three rows within 3 px of canvas column `x`.
fn ink_on_bottom_edge(img: &RgbaImage, layout: &Layout, x: u32) -> usize {
    let bottom = layout.map_y + layout.map_h - 1;
    (bottom - 2..=bottom)
        .flat_map(|y| (x - 3..=x + 3).map(move |x| (x, y)))
        .filter(|&(x, y)| is_ink(img.get_pixel(x, y)))
        .count()
}

#[test]
fn basemap_lines_that_cross_the_frame_draw_nothing_outside_it() {
    // F13: county lines leaving the bottom of a 12 km map were drawn in the
    // margin below the frame, because a segment that touched the clip mask
    // kept its outside endpoint. Every mask the renderer can pass is tried:
    // none, one covering the map, and a domain frame inset inside the map.
    let layout = frame_test_layout();
    let extent = unit_extent();
    let presentation = RenderPresentation::for_mode(ProductVisualMode::FilledMeteorology);
    let below = -40.0 / 479.0;
    let whole_map = build_rect_clip_mask(
        480,
        480,
        LocalRect {
            min_x: 0,
            max_x: 479,
            min_y: 0,
            max_y: 479,
        },
    );
    let domain_frame = build_rect_clip_mask(
        480,
        480,
        LocalRect {
            min_x: 5,
            max_x: 474,
            min_y: 5,
            max_y: 474,
        },
    );
    let leaving_x = layout.map_x + (0.30_f64 * 479.0).round() as u32;
    let entering_x = layout.map_x + (0.70_f64 * 479.0).round() as u32;
    let mut problems = Vec::new();
    for (label, mask) in [
        ("no mask", None),
        ("whole-map mask", Some(&whole_map)),
        ("domain-frame mask", Some(&domain_frame)),
    ] {
        for width in [1u32, 3] {
            let mut img = RgbaImage::from_pixel(560, 600, Rgba::WHITE.to_image_rgba());
            let lines = vec![
                // From inside the frame to 40 px below it.
                test_polyline(vec![(0.30, 0.5), (0.30, 0.1), (0.30, below)], width),
                // The same crossing drawn in the other direction.
                test_polyline(vec![(0.70, below), (0.70, 0.1), (0.70, 0.5)], width),
                // Out through the left side, 38 px past it.
                test_polyline(vec![(0.5, 0.8), (-0.08, 0.8)], width),
                // Wholly below the map.
                test_polyline(vec![(0.2, below), (0.8, below)], width),
            ];

            draw_projected_lines(&mut img, &layout, &extent, &lines, presentation, mask);

            let outside = ink_outside_map(&img, &layout);
            if !outside.is_empty() {
                let below = outside.iter().filter(|&&(_, y)| y >= 520).count();
                problems.push(format!(
                    "{label}, width {width}: {} px drawn outside the frame ({below} below it), \
                     first at {:?}",
                    outside.len(),
                    outside.first()
                ));
            }
            if ink_on_bottom_edge(&img, &layout, leaving_x) == 0 {
                problems.push(format!(
                    "{label}, width {width}: the line leaving the map stops short of the frame"
                ));
            }
            if ink_on_bottom_edge(&img, &layout, entering_x) == 0 {
                problems.push(format!(
                    "{label}, width {width}: the line entering the map stops short of the frame"
                ));
            }
        }
    }
    assert!(problems.is_empty(), "{}", problems.join("\n"));
}

#[test]
fn basemap_lines_inside_the_frame_draw_exactly_as_before() {
    // The cut only touches segments that reach the map's edge: a polyline
    // wholly inside it is the same stroke, pixel for pixel.
    let layout = frame_test_layout();
    let extent = unit_extent();
    let presentation = RenderPresentation::for_mode(ProductVisualMode::FilledMeteorology);
    let points = vec![(0.1, 0.1), (0.4, 0.73), (0.62, 0.35), (0.9, 0.88)];
    for width in [1u32, 2, 3] {
        let mut drawn = RgbaImage::from_pixel(560, 600, Rgba::WHITE.to_image_rgba());
        draw_projected_lines(
            &mut drawn,
            &layout,
            &extent,
            &[test_polyline(points.clone(), width)],
            presentation,
            None,
        );
        let mut expected = RgbaImage::from_pixel(560, 600, Rgba::WHITE.to_image_rgba());
        let canvas: Vec<(f64, f64)> = points
            .iter()
            .map(|&(x, y)| {
                let (px, py) = extent.to_pixel(x, y, layout.map_w, layout.map_h).unwrap();
                (layout.map_x as f64 + px, layout.map_y as f64 + py)
            })
            .collect();
        draw::draw_polyline_aa(&mut expected, &canvas, Rgba::BLACK, width);
        assert!(
            drawn == expected,
            "width {width}: an interior polyline changed"
        );
    }
}

#[test]
fn clip_segment_to_rect_keeps_only_the_inside_stretch() {
    let clip = |a, b| clip_segment_to_rect(a, b, 0.0, 4.0, 0.0, 4.0);
    // (from, to, expected, what the case is)
    let cases = [
        ((1.0, 1.0), (3.0, 3.0), Some((0.0, 1.0)), "inside"),
        ((2.0, 2.0), (2.0, 6.0), Some((0.0, 0.5)), "leaves"),
        ((2.0, 6.0), (2.0, 2.0), Some((0.5, 1.0)), "enters"),
        ((-2.0, 2.0), (6.0, 2.0), Some((0.25, 0.75)), "crosses"),
        ((5.0, 5.0), (6.0, 6.0), None, "outside"),
        ((-1.0, 0.0), (-1.0, 4.0), None, "parallel, outside"),
        ((0.0, 1.0), (0.0, 3.0), Some((0.0, 1.0)), "on an edge"),
        ((3.0, 5.0), (5.0, 3.0), None, "touches a corner only"),
        // A repeated vertex stays a (zero-length) segment, as it always drew.
        ((2.0, 2.0), (2.0, 2.0), Some((0.0, 1.0)), "repeat inside"),
        ((6.0, 2.0), (6.0, 2.0), None, "repeat outside"),
        ((f64::NAN, 1.0), (2.0, 2.0), None, "non-finite end"),
    ];
    for (a, b, expected, what) in cases {
        assert_eq!(clip(a, b), expected, "{what}");
    }
}

#[test]
fn clipped_polyline_stroke_writes_nothing_outside_its_rect() {
    // The stroke half of the cut: a line ending exactly on the rect's edge
    // still has width and antialiasing past it.
    for width in [1u32, 2, 3, 5] {
        let mut img = blank_test_image();
        draw::draw_polyline_aa_clipped(
            &mut img,
            &[(40.0, 40.0), (10.0, 10.0), (10.0, 69.0), (69.0, 69.0)],
            Rgba::BLACK,
            width,
            (10, 10, 69, 69),
        );
        let (min_x, max_x, min_y, max_y) = non_white_bounds(&img).expect("the stroke draws");
        assert!(
            min_x >= 10 && max_x <= 69 && min_y >= 10 && max_y <= 69,
            "width {width}: ink spans x {min_x}..={max_x}, y {min_y}..={max_y}"
        );
        assert!(
            min_x == 10 && max_y == 69,
            "width {width}: the stroke reaches the rect"
        );
    }
}

/// The tick set a narrow colour bar gets: `pick_ticks` over ten evenly cut
/// levels spanning `lo..hi`, the levels a generic stored-plane ramp cuts.
fn narrow_bar_ticks(lo: f64, hi: f64) -> Vec<f64> {
    let levels: Vec<f64> = (0..=9).map(|i| lo + (hi - lo) * i as f64 / 9.0).collect();
    pick_ticks(&levels, None)
}

/// Every drawn label must parse back to its own tick to within a hundredth
/// of the spacing between ticks, and no two drawn labels may be the same.
fn assert_labels_read_as_ticks(drawn: &[(f64, i32, String)], spacing: f64, bar: &str) {
    assert!(drawn.len() >= 3, "{bar}: only {} label(s) drawn", drawn.len());
    for (value, _, label) in drawn {
        let shown: f64 = label.parse().unwrap_or(f64::NAN);
        assert!(
            (shown - value).abs() <= spacing * 0.01,
            "{bar}: tick {value} is labelled {label:?}"
        );
    }
    let unique: std::collections::HashSet<_> = drawn.iter().map(|row| &row.2).collect();
    assert_eq!(unique.len(), drawn.len(), "{bar}: repeated labels in {drawn:?}");
}

#[test]
fn a_narrow_colour_bar_labels_every_tick_with_its_own_value() {
    // Measured on a real 3 km forecast frame: 200 hPa height spans
    // 12200.066 to 12228.032 gpm, drawn against `1e3 gpm`, and every tick
    // of the bar read `12.2`; mean sea level pressure 100282.9 to
    // 100986.8 Pa, against `1e3 Pa`, read `100.4` twice and `100.5` three
    // times.
    for (bar, lo, hi, spacing) in [
        ("200 hPa height", 12.20006640625, 12.2280322265625, 0.002),
        ("mean sea level pressure", 100.282859375, 100.98675, 0.05),
    ] {
        let ticks = narrow_bar_ticks(lo, hi);
        let range = hi - lo;
        let horizontal =
            filter_tick_labels_to_fit(&ticks, lo, range, 100, 1400, 50, 1550, 1600, 1);
        assert_labels_read_as_ticks(&horizontal, spacing, bar);
        let vertical = filter_vertical_tick_labels_to_fit(
            &ticks, lo, range, 100, 900, 100, 1000, 1100, 1,
        );
        assert_labels_read_as_ticks(&vertical, spacing, bar);
    }
}

#[test]
fn a_colour_bar_crossing_zero_labels_its_zero_tick_zero() {
    // Ticks are stepped by repeated addition, so on a 0.1 step from -0.5
    // the zero tick arrives as about -3e-17 and used to print `-0`.
    let ticks = pick_ticks(&[-0.5, 0.5], None);
    let zero = ticks
        .iter()
        .position(|value| value.abs() < 1e-9)
        .expect("a zero tick");
    assert_ne!(ticks[zero], 0.0, "the accumulated tick is not exactly zero");
    let drawn = filter_tick_labels_to_fit(&ticks, -0.5, 1.0, 100, 1400, 50, 1550, 1600, 1);
    let label = drawn
        .iter()
        .find(|row| row.0 == ticks[zero])
        .map(|row| row.2.as_str());
    assert_eq!(label, Some("0"));
}

/// A planned canvas, rendered: the image is the plan's size, the map
/// rectangle has no row or column of canvas colour in it, the header is
/// drawn whole inside the canvas, and the colour bar spans exactly the
/// map's side.  Run for a square, a wide and a tall grid.
#[test]
fn a_planned_frame_fills_its_map_and_keeps_its_chrome_inside_the_canvas() {
    let table = crate::layout_plan::LayoutTable::builtin();
    for (index, aspect) in [1.0f64, 2.91, 0.399].into_iter().enumerate() {
        let mut plan = table.plan_map(aspect, crate::layout_plan::SizeClass::Standard, 1.0);
        // A canvas size no other test renders at, so the registry cannot
        // hand this plan to an unrelated render.
        plan.canvas_w += 3 + index as u32;
        crate::layout_plan::register_canvas_plan(plan);
        let (nx, ny) = (41usize, ((41.0 / aspect).round() as usize).max(2));
        let mut x = Vec::with_capacity(nx * ny);
        let mut y = Vec::with_capacity(nx * ny);
        let mut data = Vec::with_capacity(nx * ny);
        for j in 0..ny {
            for i in 0..nx {
                x.push(i as f64 * aspect / (nx - 1) as f64);
                y.push(j as f64 / (ny - 1) as f64);
                data.push(((i + j) % 3) as f64);
            }
        }
        let mut opts = sample_projected_opts();
        opts.width = plan.canvas_w;
        opts.height = plan.canvas_h;
        opts.colorbar = true;
        opts.colorbar_units = Some("degF".into());
        opts.title = Some("2m AGL Temperature (d01 3 km)".into());
        opts.subtitle_left = Some("Init 05/26 15Z | F002 | Valid 05/26 17Z | WRF".into());
        opts.subtitle_right = Some("source: ArWen".into());
        opts.map_extent = Some(MapExtent { x_min: 0.0, x_max: aspect, y_min: 0.0, y_max: 1.0 });
        opts.projected_grid = Some(ProjectedGrid { x, y, nx, ny });
        opts.domain_frame = Some(DomainFrame {
            inset_px: 0,
            chrome_follows_frame: false,
            legend_follows_frame: false,
            source: crate::request::DomainFrameSource::ProjectedGrid,
            ..DomainFrame::map_viewport_default()
        });
        let (image, timing) = render_to_image_profile(&data, ny, nx, &opts);
        assert_eq!((image.width(), image.height()), (plan.canvas_w, plan.canvas_h), "aspect {aspect}");
        assert_eq!(
            (timing.map_x, timing.map_y, timing.map_w, timing.map_h),
            (plan.map.x, plan.map.y, plan.map.w, plan.map.h),
            "aspect {aspect}: the drawn map is the planned map"
        );
        let canvas = opts.presentation.canvas_background;
        let is_canvas = |px: u32, py: u32| {
            let p = image.get_pixel(px, py).0;
            let c = canvas.to_image_rgba().0;
            (0..3).all(|k| p[k].abs_diff(c[k]) <= 2)
        };
        for py in plan.map.y..plan.map.bottom() {
            assert!(
                !(plan.map.x..plan.map.right()).all(|px| is_canvas(px, py)),
                "aspect {aspect}: map row {py} is all canvas colour"
            );
        }
        for px in plan.map.x..plan.map.right() {
            assert!(
                !(plan.map.y..plan.map.bottom()).all(|py| is_canvas(px, py)),
                "aspect {aspect}: map column {px} is all canvas colour"
            );
        }
        // The header is inked, and nothing is inked in the outer margin.
        let header_ink = (0..plan.header.h)
            .flat_map(|py| (0..plan.canvas_w).map(move |px| (px, py)))
            .filter(|&(px, py)| !is_canvas(px, py))
            .count();
        assert!(header_ink > 200, "aspect {aspect}: header drew {header_ink} px");
        for py in 0..plan.canvas_h {
            assert!(is_canvas(0, py) && is_canvas(plan.canvas_w - 1, py), "aspect {aspect}: ink on the side edge at row {py}");
        }
        // The bar runs the map's full side and not a pixel past it.
        let bar = plan.bar.unwrap();
        match plan.bar_side {
            crate::layout_plan::BarSide::Right => {
                let mid = bar.x + bar.w / 2;
                assert!(!is_canvas(mid, bar.y + 1) && !is_canvas(mid, bar.bottom() - 2));
                assert!(is_canvas(mid, bar.y.saturating_sub(2)) && is_canvas(mid, bar.bottom() + 2));
            }
            crate::layout_plan::BarSide::Bottom => {
                let mid = bar.y + bar.h / 2;
                assert!(!is_canvas(bar.x + 1, mid) && !is_canvas(bar.right() - 2, mid));
                assert!(is_canvas(bar.x.saturating_sub(2), mid) && is_canvas(bar.right() + 2, mid));
            }
        }
    }
}

#[test]
fn transport_arrows_follow_projection_aspect_and_seam() {
    use crate::georeference::ResolvedProjection;
    use crate::overlay::DirectionProjection;
    let projection = DirectionProjection {
        projection: ResolvedProjection::Geographic { central_meridian_deg: 0.0 },
        extent: MapExtent { x_min: -180.0, x_max: 180.0, y_min: -90.0, y_max: 90.0 },
        lat_lon: vec![(60.0, 179.99998)],
    };
    let (east, north) = projected_transport_direction(Some(&projection), 0, 1.0, 0.0, 1440, 720).unwrap();
    assert!(east > 0.0 && north.abs() < 1e-10);
    let (_, meridional) = projected_transport_direction(Some(&projection), 0, 0.0, 1.0, 1440, 720).unwrap();
    assert!((east / meridional - 2.0).abs() < 1e-6);
    let projection = DirectionProjection {
        projection: ResolvedProjection::LambertConformal {
            standard_parallel_1_deg: 30.0, standard_parallel_2_deg: 60.0,
            central_meridian_deg: -100.0, reference_latitude_deg: 40.0 },
        extent: projection.extent,
        lat_lon: vec![(40.0, -120.0), (40.0, -80.0)],
    };
    let west = projected_transport_direction(Some(&projection), 0, 1.0, 0.0, 1000, 1000).unwrap();
    let east = projected_transport_direction(Some(&projection), 1, 1.0, 0.0, 1000, 1000).unwrap();
    assert!(west.0 > 0.0 && east.0 > 0.0);
    assert!(west.1 < 0.0 && east.1 > 0.0);
}

/// A smooth height-like field, `levels` lines across the grid, on a grid
/// whose cells are well under a pixel at the given map size.
fn dense_contour_overlay(nx: usize, ny: usize) -> ContourOverlay {
    let mut data = Vec::with_capacity(nx * ny);
    for j in 0..ny {
        for i in 0..nx {
            let x = i as f64 / (nx - 1) as f64;
            let y = j as f64 / (ny - 1) as f64;
            data.push(540.0 + 60.0 * x + 12.0 * (3.0 * y).sin());
        }
    }
    ContourOverlay {
        data,
        ny,
        nx,
        levels: (0..10).map(|index| 546.0 + 6.0 * index as f64).collect(),
        color: Rgba::BLACK,
        width: 1,
        labels: true,
        show_extrema: false,
        pattern: crate::request::ContourLinePattern::Solid,
        major_every: None,
        major_width: None,
    }
}

#[test]
fn contour_labels_draw_on_a_grid_whose_cells_are_under_a_pixel() {
    // A 3 km CONUS grid (1799 x 1059) on a ~1000 px map: every marching
    // squares segment is about half a pixel.  The fixed two-pixel floor
    // rejected all of them and no label was ever placed.
    let mut layout = contour_test_layout();
    layout.map_w = 1000;
    layout.map_h = 600;
    let overlay = dense_contour_overlay(1799, 1059);
    let mut img = RgbaImage::from_pixel(1000, 600, Rgba::WHITE.to_image_rgba());
    let mut placer = ContourLabelPlacer::default();
    draw_contours_bucketed(&mut img, &layout, &overlay, None, None, &mut placer, 1);
    let mut labelled: Vec<String> = placer.labels.iter().map(|label| label.text.clone()).collect();
    labelled.sort();
    labelled.dedup();
    assert!(
        labelled.len() >= 8,
        "expected most of the 10 levels labelled, got {labelled:?}"
    );
    if let Some(folder) = std::env::var_os("RUSTWX_CONTOUR_LABEL_EVIDENCE") {
        let folder = std::path::PathBuf::from(folder);
        std::fs::create_dir_all(&folder).unwrap();
        placer.draw(&mut img);
        img.save(folder.join("dense-grid-contour-labels.png")).unwrap();
    }
}

#[test]
fn the_label_segment_floor_is_unchanged_on_coarse_grids() {
    assert_eq!(contour_label_min_segment_px(4.0), 2.0);
    assert_eq!(contour_label_min_segment_px(40.0), 2.0);
    assert_eq!(contour_label_min_segment_px(f64::INFINITY), 2.0);
    assert_eq!(contour_label_min_segment_px(0.0), 2.0);
    assert!((contour_label_min_segment_px(0.6) - 0.3).abs() < 1.0e-12);
}

#[test]
fn a_label_crossing_the_domain_clip_is_not_placed() {
    let mut layout = contour_test_layout();
    layout.map_x = 10;
    layout.map_y = 10;
    layout.map_w = 200;
    layout.map_h = 200;
    // The drawn domain is the local square 20..180.
    let mut mask = RgbaImage::new(200, 200);
    for y in 20..180 {
        for x in 20..180 {
            mask.put_pixel(x, y, Rgba::WHITE.to_image_rgba());
        }
    }
    let inside = LabelRect { min_x: 60, max_x: 90, min_y: 60, max_y: 72 };
    let across_left = LabelRect { min_x: 25, max_x: 55, min_y: 60, max_y: 72 };
    let across_bottom = LabelRect { min_x: 60, max_x: 90, min_y: 182, max_y: 194 };
    assert!(label_rect_inside_clip_mask(&layout, inside, 2, Some(&mask)));
    assert!(!label_rect_inside_clip_mask(&layout, across_left, 2, Some(&mask)));
    assert!(!label_rect_inside_clip_mask(&layout, across_bottom, 2, Some(&mask)));
    assert!(label_rect_inside_clip_mask(&layout, across_left, 2, None));
}

#[test]
fn masked_dense_contours_keep_every_label_inside_the_domain() {
    let mut layout = contour_test_layout();
    layout.map_w = 800;
    layout.map_h = 500;
    let overlay = dense_contour_overlay(1400, 900);
    let mut mask = RgbaImage::new(800, 500);
    for y in 40..460 {
        for x in 40..760 {
            mask.put_pixel(x, y, Rgba::WHITE.to_image_rgba());
        }
    }
    let mut img = RgbaImage::from_pixel(800, 500, Rgba::WHITE.to_image_rgba());
    let mut placer = ContourLabelPlacer::default();
    draw_contours_bucketed(&mut img, &layout, &overlay, None, Some(&mask), &mut placer, 1);
    assert!(!placer.labels.is_empty());
    for label in &placer.labels {
        assert!(label.x >= 40 && label.y >= 40, "label {} at {},{}", label.text, label.x, label.y);
        assert!(label.x < 760 && label.y < 460);
    }
}

/// A filled map whose values all fall under the scale's mask (a cold column
/// with no 0 C level, an accumulation at F000) carries a note; one drawn
/// cell, or a map with no colour bar, does not.
#[test]
fn empty_filled_map_gets_a_note_and_a_drawn_one_does_not() {
    let mut opts = sample_projected_opts();
    opts.cmap = sample_masked_cmap();
    opts.colorbar = true;
    assert_eq!(empty_map_note(&[f64::NAN, 1.0, 5.0, 9.0], &opts).as_deref(), Some(EMPTY_MAP_NOTE));
    opts.empty_note = Some("No 0 C level: the whole column is colder than 0 C".into());
    assert_eq!(
        empty_map_note(&[f64::NAN; 4], &opts).as_deref(),
        Some("No 0 C level: the whole column is colder than 0 C")
    );
    assert_eq!(empty_map_note(&[f64::NAN, 1.0, 15.0, 9.0], &opts), None);
    opts.colorbar = false;
    assert_eq!(empty_map_note(&[f64::NAN; 4], &opts), None);
    // The note is drawn inside the map rectangle.
    opts.colorbar = true;
    let (bare, _) = {
        let mut drawn = opts.clone();
        drawn.empty_note = None;
        render_to_image_profile_unnoted(&[f64::NAN; 4], 2, 2, &drawn)
    };
    let (noted, _) = render_to_image_profile(&[f64::NAN; 4], 2, 2, &opts);
    assert_ne!(bare.as_raw(), noted.as_raw());
}

/// A mesh panel draws its values on the cells over a 2x2 NaN placeholder
/// raster: a coloured cell means the map is not empty, while a mesh whose
/// cells have no value, or only masked values, still carries the note.
#[test]
fn mesh_cells_with_values_suppress_the_empty_note() {
    let mut opts = sample_projected_opts();
    opts.cmap = sample_masked_cmap();
    opts.colorbar = true;
    let ring = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
    let style = crate::mesh_cells::MeshDrawStyle::from_theme(crate::theme::active_theme().mesh, 1);
    let layer = |values: &[Option<f64>]| MeshCellsOverlay {
        cells: values
            .iter()
            .map(|value| crate::mesh_cells::MeshCell { ring: ring.clone(), value: *value })
            .collect(),
        style,
    };
    let placeholder = [f64::NAN; 4];
    opts.mesh_cells = Some(layer(&[None, Some(5.0), Some(15.0)]));
    assert_eq!(empty_map_note(&placeholder, &opts), None);
    opts.mesh_cells = Some(layer(&[None, Some(5.0), Some(f64::NAN)]));
    assert_eq!(empty_map_note(&placeholder, &opts).as_deref(), Some(EMPTY_MAP_NOTE));
    opts.mesh_cells = Some(layer(&[None, None]));
    assert_eq!(empty_map_note(&placeholder, &opts).as_deref(), Some(EMPTY_MAP_NOTE));
}

/// Dimensionless units label nothing; real units are kept as given.
#[test]
fn dimensionless_units_carry_no_colorbar_label() {
    for units in ["1", "0/1", " 1 ", "", "dimensionless", "-"] {
        assert_eq!(crate::legend_units_label(units), None, "{units:?}");
    }
    for units in ["in", "m", "g kg-1", "%", "dBZ"] {
        assert_eq!(crate::legend_units_label(units).as_deref(), Some(units));
    }
}
