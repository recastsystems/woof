pub mod advisory;
pub mod chrome_plan;
mod color;
mod colorbar;
mod colormap;
mod colormaps;
mod contour_fill;
pub mod difference;
mod draw;
mod error;
mod features;
pub mod footer;
pub mod georeference;
pub mod layout_plan;
pub mod mesh_cells;
mod overlay;
mod panel;
mod presentation;
mod projected_map;
mod projection;
pub mod radar_tables;
mod rasterize;
#[cfg(any(windows, target_os = "linux"))]
mod rasterize_cuda;
mod render;
mod request;
mod text;
pub mod theme;
pub mod weather;

pub use contour_fill::{
    ProjectedContourGeometry, ProjectedContourGeometryTiming, ProjectedContourLineStyle,
    build_projected_contour_geometry, build_projected_contour_geometry_profile,
};
pub use error::RustwxRenderError;
pub use layout_plan::{
    BarSide, CanvasPlan, LayoutMode, LayoutTable, PlanRect, SectionPlan, SheetPlan, SizeClass,
    auto_layout_active, canvas_plan_for, layout_mode, register_canvas_plan, set_layout_mode,
};
pub use rasterize::{cuda_rasterize_stats, print_cuda_rasterize_stats_if_enabled};

pub use features::{
    BasemapDetail, BasemapStyle, StyledLonLatLayer, StyledLonLatPolygonLayer,
    checked_in_natural_earth_110m_root, load_styled_basemap_features_for,
    load_styled_basemap_features_for_detail,
    load_styled_basemap_polygons_for, load_styled_conus_features_for,
    load_styled_conus_polygons_for,
};
pub use image::RgbaImage;
pub use panel::{PanelGridLayout, PanelPadding, compose_panel_images, render_panel_grid};
pub use presentation::{
    LineworkRole, PolygonRole, ProductVisualMode, RenderPresentation, StaticPlotStyle,
};
pub use projected_map::{
    PROJECTED_MAP_CACHE_BYTES,
    resolved_projection_for_options,
    GeographicBounds, ProjectedBasemap, ProjectedBasemapBuildOptions, ProjectedDomainBuildOptions,
    ProjectedFrameSource, ProjectedMap, ProjectedMapBuildOptions, build_projected_domain,
    build_projected_map, build_projected_map_with_options,
    project_geographic_points_with_options, project_geographic_rings_with_options,
};
pub use georeference::{
    PANEL_GEOREFERENCE_SCHEMA, PanelGeoReference, PlotRect, ResolvedProjection,
};
pub use projection::{LambertConformal, ProjectionSpec};
pub use render::{
    PngCompressionMode, PngWriteOptions, RenderImageTiming, RenderPngTiming,
    map_frame_aspect_ratio, map_frame_aspect_ratio_for_mode,
    map_frame_aspect_ratio_for_mode_with_chrome_scale,
    map_frame_aspect_ratio_for_mode_with_domain_frame_and_chrome_scale, render_to_image_profile,
    render_to_png_profile as profile_render_to_png,
};
pub use request::{
    ChromeScale, Color, ColorScale, ContourLayer, ContourLinePattern, ContourStyle,
    DiscreteColorScale, DomainFrame, DomainFrameSource, ExtendMode, Field2D, GeographicClipBounds,
    GridShape, InverseRasterProjection, LatLonGrid, MapRenderRequest, ProductKey, ProductMaturity,
    ProductSemanticFlag, ProductSemantics, ProjectedDomain, ProjectedExtent,
    ProjectedLabelPlacement, ProjectedLineOverlay, ProjectedMarkerShape, ProjectedPlaceLabel,
    ProjectedPlaceLabelPriority, ProjectedPlaceLabelStyle, ProjectedPointOverlay,
    ProjectedPolygonFill, RasterSampleMode, RgbaGridField, WindBarbLayer, WindBarbStyle,
    WindStreamlineLayer, WindStreamlineStyle,
};
pub use rustwx_core::{
    Field2D as CoreField2D, GridProjection as CoreGridProjection, GridShape as CoreGridShape,
    LatLonGrid as CoreLatLonGrid, ProductKey as CoreProductKey,
};
pub use weather::{
    DerivedProductStyle, DerivedScalePreset, ECAPE_SEVERE_PANEL_PRODUCTS,
    SEVERE_CLASSIC_PANEL_PRODUCTS, WeatherPalette, WeatherPreset, WeatherProduct, palette_scale,
};

pub use crate::color::Rgba;
pub use crate::radar_tables::{
    RADAR_COLORS_ENV, RadarColorSet, RadarTable, active_radar_color_set,
    install_radar_color_set, radar_color_set_from_env,
};
pub use crate::colorbar::{legend_color_at_rel, legend_tick_rel};
use crate::colormap::Extend;
pub use crate::colormap::{
    ColormapBuildOptions, LegendControls, LegendMode, LevelDensity, LeveledColormap, RenderDensity,
    densify_discrete_scale,
};
use crate::overlay::{
    BarbOverlay, ContourOverlay, InverseProjectedGrid, MapExtent, ProjectedGrid,
    ProjectedPlaceLabelOverlay, ProjectedPointOverlay as RenderProjectedPointOverlay,
    ProjectedPolygon, ProjectedPolyline,
};
use crate::render::{
    RenderOpts, center_horizontal_canvas_content, crop_canvas_whitespace,
    encode_rgba_png_profile_with_options, render_to_image as native_render_to_image, render_to_png,
    trim_vertical_canvas_whitespace,
};
pub use crate::text::{format_tick, format_tick_labels};
// The text primitives a SHEET of finished panels writes its shared header
// with (`rw_compare`): the same font owner and the same pixel sizes the
// panels' own titles use, so a header band does not bring a second face.
pub use crate::text::{draw_text, draw_text_bold, text_width, text_width_bold};
// The stroke, fill and barb primitives a point product (sounding, meteogram)
// draws with, so its lines and wind barbs are the maps' own.
pub use crate::draw::{
    draw_circle_fill_aa, draw_circle_stroke_aa, draw_line_aa_width, draw_polyline_aa,
    draw_wind_barb, fill_polygon,
};
pub use crate::theme::{
    FooterTheme, MeshTheme, PresentationTheme, RenderTheme, RenderThemeFile, THEME_ENV,
    active_theme, install_theme,
};
pub use crate::footer::{FooterFields, clear_footer_fields, footer_fields, set_footer_fields};
pub use crate::mesh_cells::{MeshCell, MeshCellsLayer, MeshDrawStyle};
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::path::Path;
use std::sync::OnceLock;
use std::time::Instant;

fn trim_vertical_canvas_whitespace_enabled() -> bool {
    std::env::var("RUSTWX_TRIM_VERTICAL_WHITESPACE")
        .ok()
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

/// On a planned canvas a masked field (echo, rain, rotation tracks: the
/// fields drawn as cells over a basemap) whose grid cell spans the
/// table's `sharp_cell_px` or more is sampled nearest, so a coarse grid
/// shows its cells instead of an interpolated blur that claims detail the
/// grid does not have.  Nearest is also the cheaper fill.
fn planned_cells_draw_sharp(request: &MapRenderRequest, masked: bool) -> bool {
    if !masked {
        return false;
    }
    let Some(plan) = layout_plan::canvas_plan_for(request.width, request.height) else {
        return false;
    };
    let shape = &request.field.grid.shape;
    let cells_x = shape.nx.saturating_sub(1).max(1) as f64;
    let cells_y = shape.ny.saturating_sub(1).max(1) as f64;
    let px_per_cell = (plan.map.w as f64 / cells_x).min(plan.map.h as f64 / cells_y);
    plan.sharp_cell_px > 0.0 && px_per_cell >= plan.sharp_cell_px
}

/// The whole-number factor a planned size class draws place labels at: 1
/// on a standard or phone frame and on every fixed canvas, 2 on a large one.
fn planned_label_factor(width: u32, height: u32) -> u32 {
    layout_plan::canvas_plan_for(width, height)
        .map(|plan| plan.scale.round().max(1.0) as u32)
        .unwrap_or(1)
}

/// The text scale step that draws a label `factor` times its size.  The
/// text table grows 4 px a step from 12 px (`text::font_size_px`), so
/// doubling a 12 px label is three steps, not one: one step drew a large
/// frame's labels at 16 px beside a header twice the standard size.
fn planned_label_text_scale(scale: u32, factor: u32) -> u32 {
    let scale = scale.max(1);
    if factor <= 1 {
        return scale;
    }
    let px = (12 + (scale - 1) * 4) * factor;
    1 + (px - 12).div_ceil(4)
}

/// A place label's ink under the active theme: unchanged unless the theme
/// replaces the white halo, in which case a near-black ink takes the
/// theme's contour ink so the text reads on the halo it now sits on.
fn themed_label_ink(theme: theme::PresentationTheme, requested: Rgba) -> Rgba {
    if theme.halo.is_some() {
        theme.substitute_dark_ink(requested)
    } else {
        requested
    }
}

/// The domain frame a planned canvas draws: flush with the map (the map
/// IS the grid, so there is no margin to inset into), never steering the
/// header or the bar (the plan places them), and, when the grid's aspect
/// was clamped, not clearing the band past the grid: that band shows
/// basemap, never blank canvas.
fn planned_domain_frame(frame: Option<DomainFrame>, width: u32, height: u32) -> Option<DomainFrame> {
    let Some(plan) = layout_plan::canvas_plan_for(width, height) else {
        return frame;
    };
    frame.map(|frame| DomainFrame {
        inset_px: 0,
        clear_outside: frame.clear_outside && !plan.aspect_clamped,
        legend_follows_frame: false,
        chrome_follows_frame: false,
        source: DomainFrameSource::ProjectedGrid,
        ..frame
    })
}

#[derive(Debug, Default, Clone, Copy)]
pub struct RustRenderer;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RenderStateTiming {
    pub validate_ms: u128,
    pub data_buffer_ms: u128,
    pub projected_grid_ms: u128,
    pub projected_lines_ms: u128,
    pub projected_polygons_ms: u128,
    pub contour_prep_ms: u128,
    pub barb_prep_ms: u128,
    pub state_prep_ms: u128,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RenderSaveTiming {
    pub state_timing: RenderStateTiming,
    pub png_timing: RenderPngTiming,
    pub file_write_ms: u128,
    pub total_ms: u128,
    /// What the PNG that was just written maps to on the Earth.
    ///
    /// gpuwm addition (VENDOR.md).  `Some` only when every ingredient is
    /// present AND still true of the written bytes: the caller published a
    /// resolved projection and its geographic bounds on the request, the
    /// request carried a projected domain (its extent is the projected
    /// box), and no post-render pass moved the map inside the canvas.
    /// Anything less and this is `None` with the reason below, because a
    /// georeference that is tens of pixels wrong is the defect this exists
    /// to fix, not a lesser version of the fix.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub georeference: Option<georeference::PanelGeoReference>,
    /// Why `georeference` is `None`, naming the missing ingredient.  A
    /// manifest that lists a panel without a transform must say what was
    /// missing -- a silent omission is exactly the defect being fixed.
    /// `None` whenever `georeference` is `Some`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub georeference_absent_reason: Option<String>,
}

/// The publish decision for one finished panel: the georeference when the
/// request supplied every ingredient and the written PNG still matches the
/// layout, otherwise the reason it is withheld (gpuwm addition, VENDOR.md).
fn panel_georeference_for_save(
    request: &MapRenderRequest,
    image_timing: &RenderImageTiming,
) -> (
    Option<georeference::PanelGeoReference>,
    Option<String>,
) {
    let Some(resolved_projection) = request.resolved_projection else {
        return (
            None,
            Some(
                "the render request carried no resolved projection; the caller did not \
                 publish which projection the projected domain was built in"
                    .to_string(),
            ),
        );
    };
    let Some(projected_domain) = request.projected_domain.as_ref() else {
        return (
            None,
            Some(
                "the render request carried no projected domain, so no projected extent \
                 exists to map pixels through"
                    .to_string(),
            ),
        );
    };
    let Some(geographic_bounds) = request.geographic_bounds else {
        return (
            None,
            Some(
                "the render request carried no geographic bounds; publishing a transform \
                 with fabricated bounds would be worse than publishing none"
                    .to_string(),
            ),
        );
    };
    if !image_timing.plot_rect_describes_the_png {
        return (
            None,
            Some(
                "a post-render pass (map-viewport crop, horizontal recentre, or vertical \
                 trim) moved the map entirely outside the written PNG, so no pixel of the \
                 plot rectangle survives for a transform to describe"
                    .to_string(),
            ),
        );
    }
    if image_timing.map_w == 0 || image_timing.map_h == 0 {
        return (
            None,
            Some("the layout reported a zero-size map rectangle".to_string()),
        );
    }
    // The written PNG's own size: the post-render passes can shrink the
    // canvas below the requested size, and a georeference must describe
    // the FILE, not the request.  The request dims are only a fallback
    // for a timing that predates the recorded size.
    let (image_width_px, image_height_px) =
        if image_timing.image_w > 0 && image_timing.image_h > 0 {
            (image_timing.image_w, image_timing.image_h)
        } else {
            (request.width, request.height)
        };
    (
        Some(georeference::PanelGeoReference::new(
            image_width_px,
            image_height_px,
            georeference::PlotRect {
                x: image_timing.map_x,
                y: image_timing.map_y,
                width: image_timing.map_w,
                height: image_timing.map_h,
            },
            resolved_projection,
            clipped_projected_extent(&projected_domain.extent, image_timing),
            geographic_bounds,
        )),
        None,
    )
}

/// The projected box the CLIPPED plot rectangle spans: the domain's
/// extent with each side cut by the fraction of the unclipped pixel span
/// a post-render pass removed (`RenderImageTiming::map_clip_*`).  The
/// rectangle and the extent are cut by the same fractions, so a point
/// maps to the same pixel through the clipped pair as it did through the
/// unclipped one.  All-zero fractions return the extent unchanged.
pub fn clipped_projected_extent(
    extent: &ProjectedExtent,
    image_timing: &RenderImageTiming,
) -> ProjectedExtent {
    let dx = extent.x_max - extent.x_min;
    let dy = extent.y_max - extent.y_min;
    ProjectedExtent {
        x_min: extent.x_min + dx * image_timing.map_clip_left,
        x_max: extent.x_max - dx * image_timing.map_clip_right,
        y_min: extent.y_min + dy * image_timing.map_clip_bottom,
        y_max: extent.y_max - dy * image_timing.map_clip_top,
    }
}

/// The plot rectangle after a post-render pass moved it, intersected
/// with the written image.
///
/// `x`, `y` is the moved rectangle's origin (signed: a pass that cut the
/// left or top edge puts it below zero), `width`/`height` its unclipped
/// size, `image_w`/`image_h` the written image.  The result is the part
/// inside the image plus the fraction of the unclipped pixel span cut
/// off each side, in the span [`georeference::PanelGeoReference`] maps
/// the extent onto (width less one, height less one).  `None` when no
/// pixel survives.
pub fn clip_plot_rect_to_image(
    x: i64,
    y: i64,
    width: u32,
    height: u32,
    image_w: u32,
    image_h: u32,
) -> Option<ClippedPlotRect> {
    if width == 0 || height == 0 || image_w == 0 || image_h == 0 {
        return None;
    }
    let right = x + i64::from(width);
    let bottom = y + i64::from(height);
    let left_in = x.max(0);
    let top_in = y.max(0);
    let right_in = right.min(i64::from(image_w));
    let bottom_in = bottom.min(i64::from(image_h));
    if right_in <= left_in || bottom_in <= top_in {
        return None;
    }
    let x_span = f64::from(width.saturating_sub(1)).max(1.0);
    let y_span = f64::from(height.saturating_sub(1)).max(1.0);
    Some(ClippedPlotRect {
        x: left_in as u32,
        y: top_in as u32,
        width: (right_in - left_in) as u32,
        height: (bottom_in - top_in) as u32,
        left: (left_in - x) as f64 / x_span,
        right: (right - right_in) as f64 / x_span,
        top: (top_in - y) as f64 / y_span,
        bottom: (bottom - bottom_in) as f64 / y_span,
    })
}

/// See [`clip_plot_rect_to_image`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClippedPlotRect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
    pub left: f64,
    pub right: f64,
    pub top: f64,
    pub bottom: f64,
}

#[derive(Default)]
struct RenderScratch {
    f64_buffers: Vec<Vec<f64>>,
    point_buffers: Vec<Vec<(f64, f64)>>,
}

impl RenderScratch {
    fn take_f64_buffer(&mut self, len: usize) -> Vec<f64> {
        let mut buffer = self.f64_buffers.pop().unwrap_or_default();
        buffer.clear();
        if buffer.capacity() < len {
            buffer.reserve(len - buffer.capacity());
        }
        buffer
    }

    fn fill_f64_from_f32(&mut self, src: &[f32]) -> Vec<f64> {
        let mut buffer = self.take_f64_buffer(src.len());
        buffer.extend(src.iter().map(|&value| value as f64));
        buffer
    }

    fn fill_f64_from_f64(&mut self, src: &[f64]) -> Vec<f64> {
        let mut buffer = self.take_f64_buffer(src.len());
        buffer.extend_from_slice(src);
        buffer
    }

    fn fill_f64_constant(&mut self, len: usize, value: f64) -> Vec<f64> {
        let mut buffer = self.take_f64_buffer(len);
        buffer.resize(len, value);
        buffer
    }

    fn reclaim_f64_buffer(&mut self, mut buffer: Vec<f64>) {
        buffer.clear();
        self.f64_buffers.push(buffer);
    }

    fn take_point_buffer(&mut self, len: usize) -> Vec<(f64, f64)> {
        let mut buffer = self.point_buffers.pop().unwrap_or_default();
        buffer.clear();
        if buffer.capacity() < len {
            buffer.reserve(len - buffer.capacity());
        }
        buffer
    }

    fn fill_point_buffer(&mut self, src: &[(f64, f64)]) -> Vec<(f64, f64)> {
        let mut buffer = self.take_point_buffer(src.len());
        buffer.extend_from_slice(src);
        buffer
    }

    fn reclaim_point_buffer(&mut self, mut buffer: Vec<(f64, f64)>) {
        buffer.clear();
        self.point_buffers.push(buffer);
    }

    fn reclaim_render_opts(&mut self, mut opts: RenderOpts, data: Vec<f64>) {
        self.reclaim_f64_buffer(data);

        if let Some(grid) = opts.projected_grid.take() {
            self.reclaim_f64_buffer(grid.x);
            self.reclaim_f64_buffer(grid.y);
        }
        if let Some(grid) = opts.inverse_projected_grid.take() {
            self.reclaim_f64_buffer(grid.lat_deg);
            self.reclaim_f64_buffer(grid.lon_deg);
        }

        for line in opts.projected_lines.drain(..) {
            self.reclaim_point_buffer(line.points);
        }

        for poly in opts.projected_polygons.drain(..) {
            for ring in poly.rings {
                self.reclaim_point_buffer(ring);
            }
        }

        for poly in opts.projected_data_polygons.drain(..) {
            for ring in poly.rings {
                self.reclaim_point_buffer(ring);
            }
        }

        for contour in opts.contours.drain(..) {
            self.reclaim_f64_buffer(contour.data);
            self.reclaim_f64_buffer(contour.levels);
        }

        opts.barbs.clear();

        for streamline in opts.streamlines.drain(..) {
            self.reclaim_f64_buffer(streamline.u);
            self.reclaim_f64_buffer(streamline.v);
        }
    }
}

thread_local! {
    static RENDER_SCRATCH: RefCell<RenderScratch> = RefCell::new(RenderScratch::default());
}

// Raster work can yield to another product on this same Rayon worker. Keep
// only the idle buffer in TLS; an active render owns its scratch and never
// holds a RefCell borrow across a callback or a parallel operation.
fn with_render_scratch<T>(render: impl FnOnce(&mut RenderScratch) -> T) -> T {
    struct Lease(RenderScratch);
    impl Drop for Lease {
        fn drop(&mut self) {
            RENDER_SCRATCH.with(|slot| {
                slot.replace(std::mem::take(&mut self.0));
            });
        }
    }
    let mut scratch = Lease(RENDER_SCRATCH.with(RefCell::take));
    render(&mut scratch.0)
}

impl RustRenderer {
    pub fn render_png(self, request: &MapRenderRequest) -> Result<Vec<u8>, RustwxRenderError> {
        with_render_state(request, |data, ny, nx, opts| {
            Ok(render_to_png(data, ny, nx, opts))
        })
    }

    pub fn render_image(self, request: &MapRenderRequest) -> Result<RgbaImage, RustwxRenderError> {
        with_render_state(request, |data, ny, nx, opts| {
            Ok(native_render_to_image(data, ny, nx, opts))
        })
    }

    pub fn render_image_with_style(
        self,
        request: &MapRenderRequest,
        plot_style: StaticPlotStyle,
    ) -> Result<RgbaImage, RustwxRenderError> {
        with_render_state_with_style(request, plot_style, |data, ny, nx, opts| {
            Ok(native_render_to_image(data, ny, nx, opts))
        })
    }

    pub fn save_png<P: AsRef<Path>>(
        self,
        request: &MapRenderRequest,
        output_path: P,
    ) -> Result<(), RustwxRenderError> {
        self.save_png_profile_with_options(request, output_path, &PngWriteOptions::default())
            .map(|_| ())
    }

    pub fn save_png_profile<P: AsRef<Path>>(
        self,
        request: &MapRenderRequest,
        output_path: P,
    ) -> Result<RenderSaveTiming, RustwxRenderError> {
        self.save_png_profile_with_options(request, output_path, &PngWriteOptions::default())
    }

    pub fn save_png_profile_with_options<P: AsRef<Path>>(
        self,
        request: &MapRenderRequest,
        output_path: P,
        png_options: &PngWriteOptions,
    ) -> Result<RenderSaveTiming, RustwxRenderError> {
        self.save_png_profile_with_options_and_style(
            request,
            output_path,
            png_options,
            StaticPlotStyle::from_env(),
        )
    }

    pub fn save_png_profile_with_options_and_style<P: AsRef<Path>>(
        self,
        request: &MapRenderRequest,
        output_path: P,
        png_options: &PngWriteOptions,
        plot_style: StaticPlotStyle,
    ) -> Result<RenderSaveTiming, RustwxRenderError> {
        // A run difference (`difference.rs`) sees every finished product
        // here, whichever lane built it, so one hook covers every family.
        // With no difference in progress this is one uncontended lock.
        if let Some(outcome) =
            difference::intercept(request, output_path.as_ref(), png_options, plot_style)
        {
            return outcome;
        }
        self.save_drawn_png(request, output_path.as_ref(), png_options, plot_style)
    }

    /// Draw `request` and write it to `output_path`: the save path with no
    /// difference hook in front of it.
    pub(crate) fn save_drawn_png(
        self,
        request: &MapRenderRequest,
        output_path: &Path,
        png_options: &PngWriteOptions,
        plot_style: StaticPlotStyle,
    ) -> Result<RenderSaveTiming, RustwxRenderError> {
        let total_start = Instant::now();
        let (bytes, state_timing, png_timing) =
            with_render_state_profile_with_style(request, plot_style, |data, ny, nx, opts| {
                let (image, mut image_timing) = render_to_image_profile(data, ny, nx, opts);
                let trim_start = Instant::now();
                // gpuwm addition (VENDOR.md): each of these three passes
                // can move the map inside the canvas.  They used to
                // report nothing, so the plot rectangle died with the
                // first pass -- and since every REGIONAL panel's domain
                // frame takes one, only global panels ever published a
                // georeference, which is not "fixed" by default.  Each
                // pass now reports the offset it applied; the rectangle
                // FOLLOWS the map.  Where the moved rectangle overhangs
                // the written image it is CLIPPED to the surviving pixels
                // and the cut fractions are recorded, so the projected
                // extent is cut by the same amount and the transform
                // still describes the file: a regional grid drawn in a
                // frame wider than its data is recentred by more than its
                // margin on every frame, and withholding the sidecar for
                // that left a correctly drawn map with no transform.  It
                // is retired only when no pixel survives.
                let mut moved_x: i64 = 0;
                let mut moved_y: i64 = 0;
                // A planned canvas was sized for its map: there is no
                // padding for a pass to find, and the map stays exactly
                // where the plan put it.
                let planned = layout_plan::canvas_plan_for(opts.width, opts.height).is_some();
                let image = match opts.domain_frame {
                    _ if planned => image,
                    Some(frame) if matches!(frame.source, DomainFrameSource::MapViewport) => {
                        let (cropped, (crop_left, crop_top)) = crop_canvas_whitespace(
                            &image,
                            opts.presentation.canvas_background,
                            8,
                        );
                        moved_x -= i64::from(crop_left);
                        moved_y -= i64::from(crop_top);
                        cropped
                    }
                    Some(_) => {
                        let (centered, shift_x) = center_horizontal_canvas_content(
                            &image,
                            opts.presentation.canvas_background,
                        );
                        moved_x += shift_x;
                        centered
                    }
                    None => image,
                };
                let trimmed = if !planned && trim_vertical_canvas_whitespace_enabled() {
                    let (trimmed, crop_top) = trim_vertical_canvas_whitespace(
                        &image,
                        opts.presentation.canvas_background,
                    );
                    moved_y -= i64::from(crop_top);
                    trimmed
                } else {
                    image
                };
                image_timing.postprocess_offset_x = moved_x;
                image_timing.postprocess_offset_y = moved_y;
                image_timing.image_w = trimmed.width();
                image_timing.image_h = trimmed.height();
                let adjusted_x = i64::from(image_timing.map_x) + moved_x;
                let adjusted_y = i64::from(image_timing.map_y) + moved_y;
                match clip_plot_rect_to_image(
                    adjusted_x,
                    adjusted_y,
                    image_timing.map_w,
                    image_timing.map_h,
                    trimmed.width(),
                    trimmed.height(),
                ) {
                    Some(clip) => {
                        image_timing.map_x = clip.x;
                        image_timing.map_y = clip.y;
                        image_timing.map_w = clip.width;
                        image_timing.map_h = clip.height;
                        image_timing.map_clip_left = clip.left;
                        image_timing.map_clip_right = clip.right;
                        image_timing.map_clip_top = clip.top;
                        image_timing.map_clip_bottom = clip.bottom;
                    }
                    None => image_timing.plot_rect_describes_the_png = false,
                }
                let trim_ms = trim_start.elapsed().as_millis();
                image_timing.postprocess_ms = image_timing.postprocess_ms.saturating_add(trim_ms);
                image_timing.total_ms = image_timing.total_ms.saturating_add(trim_ms);
                let render_to_image_ms = image_timing.total_ms;
                // The strip is composed AFTER every pass that can move the
                // map inside the canvas, and only under it, so map_x, map_y
                // and the published georeference still describe the written
                // PNG.  Both halves are required: a theme that names a footer
                // and caption fields the caller installed.  Neither built-in
                // theme names one, so no existing render grows a strip and
                // the pixel gate stays green.
                let composed =
                    match (theme::active_theme().footer.as_ref(), footer::footer_fields()) {
                        (Some(footer_theme), Some(fields)) => {
                            let fields = fields.with_panel_defaults(
                                request.title.as_deref(),
                                request.subtitle_left.as_deref(),
                            );
                            Some(footer::compose(&trimmed, footer_theme, &fields))
                        }
                        _ => None,
                    };
                let trimmed = composed.unwrap_or(trimmed);
                image_timing.image_w = trimmed.width();
                image_timing.image_h = trimmed.height();
                let (bytes, png_encode_ms) =
                    encode_rgba_png_profile_with_options(&trimmed, png_options);
                Ok((
                    bytes,
                    RenderPngTiming {
                        image_timing,
                        render_to_image_ms,
                        png_encode_ms,
                        png_write_ms: 0,
                        total_ms: render_to_image_ms.saturating_add(png_encode_ms),
                    },
                ))
            })?;
        let path = output_path;
        let write_start = Instant::now();
        std::fs::write(path, bytes).map_err(|source| RustwxRenderError::WriteFile {
            path: path.display().to_string(),
            source,
        })?;
        let file_write_ms = write_start.elapsed().as_millis();
        let mut png_timing = png_timing;
        png_timing.png_write_ms = file_write_ms;
        let (georeference, georeference_absent_reason) =
            panel_georeference_for_save(request, &png_timing.image_timing);
        let timing = RenderSaveTiming {
            state_timing,
            png_timing,
            file_write_ms,
            total_ms: total_start.elapsed().as_millis(),
            georeference,
            georeference_absent_reason,
        };
        // Separate diagnostic runs can expose timers already collected by the
        // renderer without adding timers or serialization to ordinary renders.
        static TRACE_TIMING: OnceLock<bool> = OnceLock::new();
        if *TRACE_TIMING.get_or_init(|| std::env::var_os("RUSTWX_TRACE_RENDER_TIMING").is_some()) {
            eprintln!("RUSTWX_RENDER_TIMING {}", serde_json::json!({
                "path": path, "timing": &timing, "cuda": cuda_rasterize_stats(),
            }));
        }
        Ok(timing)
    }
}

pub fn render_png(request: &MapRenderRequest) -> Result<Vec<u8>, RustwxRenderError> {
    RustRenderer.render_png(request)
}

/// The header text of a planned sheet: what a single map's header would
/// carry.  `units` falls back to the first member's field units.
#[derive(Debug, Clone, Copy, Default)]
pub struct SheetHeader<'a> {
    pub title: Option<&'a str>,
    pub units: Option<&'a str>,
    pub subtitle_left: Option<&'a str>,
    pub subtitle_center: Option<&'a str>,
    pub subtitle_right: Option<&'a str>,
}

/// Draw a multi-panel sheet on its plan (`LayoutTable::plan_sheet`): each
/// member map at one cell's size, a label strip over each, the map header
/// across the top, and one colour bar shared by every member (members of
/// one sheet share a scale).  The members must be sized to
/// `sheet.member`; their own titles and bars are not drawn.
pub fn render_planned_sheet(
    sheet: &SheetPlan,
    members: &[MapRenderRequest],
    labels: &[String],
    header: SheetHeader<'_>,
) -> Result<RgbaImage, RustwxRenderError> {
    if members.len() > sheet.cells.len() {
        return Err(RustwxRenderError::TooManyPanels {
            actual: members.len(),
            capacity: sheet.cells.len(),
        });
    }
    register_canvas_plan(sheet.member);
    struct Chrome {
        canvas: Rgba,
        title: Rgba,
        meta: Rgba,
        cmap: LeveledColormap,
        mode: LegendMode,
        colorbar: presentation::ColorbarPresentation,
        ticks: Vec<f64>,
        units: String,
    }
    let mut chrome: Option<Chrome> = None;
    let mut images = Vec::with_capacity(members.len());
    for (index, request) in members.iter().enumerate() {
        if (request.width, request.height) != (sheet.member.canvas_w, sheet.member.canvas_h) {
            return Err(RustwxRenderError::PanelSizeMismatch {
                index,
                expected_width: sheet.member.canvas_w,
                expected_height: sheet.member.canvas_h,
                actual_width: request.width,
                actual_height: request.height,
            });
        }
        let image = with_render_state(request, |data, ny, nx, opts| {
            if chrome.is_none() {
                chrome = Some(Chrome {
                    canvas: if opts.background == Rgba::WHITE {
                        opts.presentation.canvas_background
                    } else {
                        opts.background
                    },
                    title: opts.presentation.chrome.title_color,
                    meta: opts.presentation.chrome.subtitle_color,
                    cmap: opts.cmap.clone(),
                    mode: opts.colorbar_mode,
                    colorbar: opts.presentation.colorbar,
                    ticks: render::legend_ticks(&opts.cmap, opts.cbar_tick_step),
                    units: opts.colorbar_units.clone().unwrap_or_default(),
                });
            }
            Ok(native_render_to_image(data, ny, nx, opts))
        })?;
        images.push(image);
    }
    let Some(chrome) = chrome else {
        return Ok(RgbaImage::new(sheet.canvas_w, sheet.canvas_h));
    };
    let mut canvas = RgbaImage::from_pixel(sheet.canvas_w, sheet.canvas_h, chrome.canvas.to_image_rgba());
    for (index, image) in images.iter().enumerate() {
        let cell = sheet.cells[index];
        image::imageops::replace(&mut canvas, image, i64::from(cell.x), i64::from(cell.y));
        if let Some(label) = labels.get(index).map(|label| label.trim()).filter(|l| !l.is_empty()) {
            let strip = sheet.labels[index];
            let size = sheet.label_px;
            let baseline = strip.y as f32 + strip.h as f32 * 0.72;
            let top = (baseline - text::ascent_px(size, true)).round() as i32;
            text::draw_text_px(&mut canvas, label, strip.x as i32, top, chrome.title, size, true);
        }
    }
    let units = header.units.map(str::to_string).unwrap_or(chrome.units);
    let header_text = chrome_plan::PlanHeaderText::compose(
        header.title,
        Some(units.as_str()),
        header.subtitle_left,
        header.subtitle_center,
        header.subtitle_right,
    );
    chrome_plan::draw_plan_header(&mut canvas, &sheet.frame, &header_text, chrome.title, chrome.meta);
    if let Some(bar) = sheet.bar {
        let levels = render::colorbar_levels_for_ticks(&chrome.cmap);
        if levels.len() >= 2 {
            chrome_plan::draw_plan_colorbar(
                &mut canvas,
                &sheet.frame,
                bar,
                sheet.bar_side,
                &chrome.cmap,
                chrome.mode,
                chrome.colorbar,
                &chrome.ticks,
                levels[0],
                levels[levels.len() - 1],
            );
        }
    }
    Ok(canvas)
}

pub fn render_image(request: &MapRenderRequest) -> Result<RgbaImage, RustwxRenderError> {
    RustRenderer.render_image(request)
}

pub fn render_image_with_style(
    request: &MapRenderRequest,
    plot_style: StaticPlotStyle,
) -> Result<RgbaImage, RustwxRenderError> {
    RustRenderer.render_image_with_style(request, plot_style)
}

pub fn save_png<P: AsRef<Path>>(
    request: &MapRenderRequest,
    output_path: P,
) -> Result<(), RustwxRenderError> {
    RustRenderer.save_png(request, output_path)
}

pub fn save_png_profile<P: AsRef<Path>>(
    request: &MapRenderRequest,
    output_path: P,
) -> Result<RenderSaveTiming, RustwxRenderError> {
    RustRenderer.save_png_profile(request, output_path)
}

pub fn save_png_profile_with_options<P: AsRef<Path>>(
    request: &MapRenderRequest,
    output_path: P,
    png_options: &PngWriteOptions,
) -> Result<RenderSaveTiming, RustwxRenderError> {
    RustRenderer.save_png_profile_with_options(request, output_path, png_options)
}

pub fn save_png_profile_with_options_and_style<P: AsRef<Path>>(
    request: &MapRenderRequest,
    output_path: P,
    png_options: &PngWriteOptions,
    plot_style: StaticPlotStyle,
) -> Result<RenderSaveTiming, RustwxRenderError> {
    RustRenderer.save_png_profile_with_options_and_style(
        request,
        output_path,
        png_options,
        plot_style,
    )
}

pub fn save_rgba_png_profile_with_options<P: AsRef<Path>>(
    image: &RgbaImage,
    output_path: P,
    png_options: &PngWriteOptions,
) -> Result<RenderSaveTiming, RustwxRenderError> {
    if let Some(refusal) = difference::refuse_composed(output_path.as_ref()) {
        return Err(refusal);
    }
    let total_start = Instant::now();
    let (bytes, png_encode_ms) = encode_rgba_png_profile_with_options(image, png_options);
    let path = output_path.as_ref();
    let write_start = Instant::now();
    std::fs::write(path, bytes).map_err(|source| RustwxRenderError::WriteFile {
        path: path.display().to_string(),
        source,
    })?;
    let file_write_ms = write_start.elapsed().as_millis();
    Ok(RenderSaveTiming {
        state_timing: RenderStateTiming::default(),
        png_timing: RenderPngTiming {
            image_timing: RenderImageTiming::default(),
            render_to_image_ms: 0,
            png_encode_ms,
            png_write_ms: file_write_ms,
            total_ms: png_encode_ms + file_write_ms,
        },
        file_write_ms,
        total_ms: total_start.elapsed().as_millis(),
        georeference: None,
        georeference_absent_reason: Some(
            "the PNG was saved from an already-composed RGBA canvas, not a single map \
             render; no one plot rectangle describes it"
                .to_string(),
        ),
    })
}

fn with_render_state<T>(
    request: &MapRenderRequest,
    render: impl FnOnce(&[f64], usize, usize, &RenderOpts) -> Result<T, RustwxRenderError>,
) -> Result<T, RustwxRenderError> {
    with_render_state_with_style(request, StaticPlotStyle::from_env(), render)
}

fn with_render_state_with_style<T>(
    request: &MapRenderRequest,
    plot_style: StaticPlotStyle,
    render: impl FnOnce(&[f64], usize, usize, &RenderOpts) -> Result<T, RustwxRenderError>,
) -> Result<T, RustwxRenderError> {
    with_render_state_profile_with_style(request, plot_style, |data, ny, nx, opts| {
        Ok((render(data, ny, nx, opts)?, RenderPngTiming::default()))
    })
    .map(|(result, _, _)| result)
}

fn with_render_state_profile_with_style<T>(
    request: &MapRenderRequest,
    plot_style: StaticPlotStyle,
    render: impl FnOnce(
        &[f64],
        usize,
        usize,
        &RenderOpts,
    ) -> Result<(T, RenderPngTiming), RustwxRenderError>,
) -> Result<(T, RenderStateTiming, RenderPngTiming), RustwxRenderError> {
    let total_start = Instant::now();
    let validate_start = Instant::now();
    validate_request(request)?;
    let validate_ms = validate_start.elapsed().as_millis();

    let shape = request.field.grid.shape;
    let overlay_only = request.is_overlay_only();
    let visual_mode = if overlay_only {
        ProductVisualMode::OverlayAnalysis
    } else {
        request.visual_mode
    };
    let presentation = RenderPresentation::for_mode_with_style(visual_mode, plot_style);
    // A theme may name this product's colormap; the scale keeps its own
    // levels, extend mode and mask, only the colours are replaced.
    let themed_scale =
        theme::active_theme().product_scale_override(&request.field.product, &request.scale);
    let cmap = if overlay_only {
        blank_fill_colormap()
    } else {
        build_colormap(
            themed_scale.as_ref().unwrap_or(&request.scale),
            ColormapBuildOptions {
                // A difference panel keeps the stepped bands it asked for
                // under every plot style (`difference::drawing_stepped`).
                render_density: if difference::drawing_stepped() {
                    request.render_density
                } else {
                    plot_style.render_density(request.render_density)
                },
                // A theme that names both a product's band edges and one
                // colour per band (the tc wind ladder) draws those bands as
                // listed, with a tick on every edge (34, 50, 64 ... kt).
                legend: if theme::active_theme().product_has_banded_scale(&request.field.product) {
                    LegendControls { mode: LegendMode::Thresholds, ..request.legend }
                } else {
                    request.legend
                },
            },
        )
    };
    let category_map = cmap.categories;
    let cell_sharp = planned_cells_draw_sharp(request, cmap.mask_below.is_some());
    let projected_domain = request.projected_domain.as_ref();
    let default_title = default_title(&request.field);

    with_render_scratch(|scratch| {

        let data_start = Instant::now();
        let data = if overlay_only {
            scratch.fill_f64_constant(shape.len(), OVERLAY_ONLY_FILL_VALUE)
        } else {
            scratch.fill_f64_from_f32(&request.field.values)
        };
        let data_buffer_ms = data_start.elapsed().as_millis();

        let projected_grid_start = Instant::now();
        let projected_grid = projected_domain.map(|domain| ProjectedGrid {
            x: scratch.fill_f64_from_f64(&domain.x),
            y: scratch.fill_f64_from_f64(&domain.y),
            ny: shape.ny,
            nx: shape.nx,
        });
        let projected_grid_ms = projected_grid_start.elapsed().as_millis();

        let inverse_projected_grid =
            request
                .inverse_raster_projection
                .as_ref()
                .and_then(|inverse| {
                    let projector = inverse
                        .projection
                        .build_projector(
                            inverse.reference_latitude_deg,
                            inverse.reference_longitude_deg,
                            &request.field.grid.lat_deg,
                            &request.field.grid.lon_deg,
                        )
                        .ok()?;
                    Some(InverseProjectedGrid {
                        projector,
                        clip_bounds: inverse.clip_bounds,
                        lat_deg: scratch.fill_f64_from_f32(&request.field.grid.lat_deg),
                        lon_deg: scratch.fill_f64_from_f32(&request.field.grid.lon_deg),
                    })
                });

        let rgba_grid = request.rgba_grid.as_ref().map(|field| {
            field
                .pixels
                .iter()
                .map(|pixel| Rgba {
                    r: pixel.r,
                    g: pixel.g,
                    b: pixel.b,
                    a: pixel.a,
                })
                .collect::<Vec<_>>()
        });

        let projected_lines_start = Instant::now();
        let mut projected_lines = Vec::with_capacity(request.projected_lines.len());
        for line in &request.projected_lines {
            projected_lines.push(ProjectedPolyline {
                points: scratch.fill_point_buffer(&line.points),
                color: line.color.into(),
                width: line.width,
                role: line.role,
            });
        }
        let projected_lines_ms = projected_lines_start.elapsed().as_millis();

        let projected_polygons_start = Instant::now();
        let mut projected_polygons = Vec::with_capacity(request.projected_polygons.len());
        for poly in &request.projected_polygons {
            let rings = poly
                .rings
                .iter()
                .map(|ring| scratch.fill_point_buffer(ring))
                .collect();
            projected_polygons.push(ProjectedPolygon {
                rings,
                color: poly.color.into(),
                role: poly.role,
            });
        }
        let projected_polygons_ms = projected_polygons_start.elapsed().as_millis();

        let mut projected_data_polygons = Vec::with_capacity(request.projected_data_polygons.len());
        for poly in &request.projected_data_polygons {
            let rings = poly
                .rings
                .iter()
                .map(|ring| scratch.fill_point_buffer(ring))
                .collect();
            projected_data_polygons.push(ProjectedPolygon {
                rings,
                color: poly.color.into(),
                role: poly.role,
            });
        }

        // A size class past standard draws its labels at the class's
        // scale, the way it draws the header and the bar: a large frame
        // kept standard-size place labels, a quarter of the type around them.
        let label_factor = planned_label_factor(request.width, request.height);
        let mut projected_place_labels = Vec::with_capacity(request.projected_place_labels.len());
        for place_label in &request.projected_place_labels {
            projected_place_labels.push(ProjectedPlaceLabelOverlay {
                x: place_label.x,
                y: place_label.y,
                label: place_label.label.clone(),
                priority: place_label.priority,
                style: crate::overlay::ProjectedPlaceLabelStyle {
                    marker_radius_px: place_label.style.marker_radius_px * label_factor,
                    marker_fill: place_label.style.marker_fill.into(),
                    // A dark theme swaps the white halo for its surface; the
                    // near-black label and marker inks follow to the
                    // theme's ink, or the label is dark text on a dark halo.
                    marker_outline: themed_label_ink(
                        presentation.theme,
                        place_label.style.marker_outline.into(),
                    ),
                    marker_outline_width: place_label.style.marker_outline_width * label_factor,
                    label_color: themed_label_ink(
                        presentation.theme,
                        place_label.style.label_color.into(),
                    ),
                    label_halo: presentation
                        .theme
                        .substitute_white_halo(place_label.style.label_halo.into()),
                    label_halo_width_px: place_label.style.label_halo_width_px * label_factor,
                    label_scale: planned_label_text_scale(place_label.style.label_scale, label_factor),
                    label_offset_x_px: place_label.style.label_offset_x_px * label_factor as i32,
                    label_offset_y_px: place_label.style.label_offset_y_px * label_factor as i32,
                    label_placement: place_label.style.label_placement,
                    label_bold: place_label.style.label_bold,
                },
            });
        }

        let projected_points = request
            .projected_points
            .iter()
            .map(|point| RenderProjectedPointOverlay {
                x: point.x,
                y: point.y,
                color: point.color.into(),
                radius_px: point.radius_px,
                width_px: point.width_px,
                shape: point.shape,
            })
            .collect::<Vec<_>>();

        let contour_start = Instant::now();
        let mut contours = Vec::with_capacity(request.contours.len());
        for layer in &request.contours {
            contours.push(ContourOverlay {
                data: scratch.fill_f64_from_f32(&layer.data),
                ny: shape.ny,
                nx: shape.nx,
                levels: scratch.fill_f64_from_f64(&layer.levels),
                color: presentation.contour_color(layer.color.into()),
                width: layer.width,
                labels: layer.labels,
                show_extrema: layer.show_extrema,
                pattern: layer.pattern,
                major_every: layer.major_every,
                major_width: layer.major_width,
            });
        }
        let contour_prep_ms = contour_start.elapsed().as_millis();

        let barb_start = Instant::now();
        let direction_projection = request.resolved_projection.zip(projected_domain)
            .filter(|_| request.wind_barbs.iter().any(|layer| layer.direction_arrows))
            .map(|(projection, domain)| {
            std::sync::Arc::new(crate::overlay::DirectionProjection {
                projection,
                extent: MapExtent { x_min: domain.extent.x_min, x_max: domain.extent.x_max,
                    y_min: domain.extent.y_min, y_max: domain.extent.y_max },
                lat_lon: request.field.grid.lat_deg.iter().copied().zip(request.field.grid.lon_deg.iter().copied()).collect(),
            })
        });
        let mut barbs = Vec::with_capacity(request.wind_barbs.len());
        for layer in &request.wind_barbs {
            barbs.push(BarbOverlay {
                direction_projection: if layer.direction_arrows { direction_projection.clone() } else { None },
                direction_arrows: layer.direction_arrows,
                u: layer.u.clone(),
                v: layer.v.clone(),
                ny: shape.ny,
                nx: shape.nx,
                stride_x: layer.stride_x,
                stride_y: layer.stride_y,
                spacing_px: layer.spacing_px,
                color: presentation.barb_color(layer.color.into()),
                halo_color: presentation
                    .theme
                    .substitute_white_halo(layer.halo_color.into()),
                halo_width: layer.halo_width,
                width: layer.width,
                length_px: layer.length_px,
            });
        }
        let mut streamlines = Vec::with_capacity(request.wind_streamlines.len());
        for layer in &request.wind_streamlines {
            streamlines.push(crate::overlay::StreamlineOverlay {
                u: scratch.fill_f64_from_f32(&layer.u),
                v: scratch.fill_f64_from_f32(&layer.v),
                ny: shape.ny,
                nx: shape.nx,
                stride_x: layer.stride_x,
                stride_y: layer.stride_y,
                color: presentation.barb_color(layer.color.into()),
                width: layer.width,
                max_steps: layer.max_steps,
                step_cells: layer.step_cells,
                min_speed: layer.min_speed,
            });
        }
        let barb_prep_ms = barb_start.elapsed().as_millis();

        let opts = RenderOpts {
            width: request.width,
            height: request.height,
            cmap,
            background: request.background.into(),
            colorbar: request.colorbar,
            // Product preparation has already converted this field. Its
            // own units describe the displayed values and legend scale.
            colorbar_units: legend_units_label(&request.field.units),
            title: request.title.clone().or(default_title),
            subtitle_left: request.subtitle_left.clone(),
            subtitle_center: request.subtitle_center.clone(),
            // A theme may name the provenance label drawn here; a product
            // that carries none stays bare.
            subtitle_right: theme::active_theme()
                .source_subtitle(request.subtitle_right.clone()),
            cbar_tick_step: request.cbar_tick_step,
            colorbar_mode: request.legend.mode,
            chrome_scale: request.chrome_scale,
            // A category map draws each grid value's own code. Interpolating
            // codes paints values no cell holds, and averaging a supersampled
            // frame blends two codes' colours into a third code's colour at
            // every class edge, so both are off for a category legend.
            supersample_factor: if category_map {
                1
            } else {
                plot_style.supersample_factor(request.supersample_factor)
            },
            supersample_sharpen: !category_map
                && plot_style.supersample_sharpen(request.supersample_sharpen),
            raster_sample_mode: if category_map || cell_sharp {
                RasterSampleMode::Nearest
            } else {
                request.raster_sample_mode
            },
            domain_frame: planned_domain_frame(request.domain_frame, request.width, request.height),
            map_extent: projected_domain.map(|domain| MapExtent {
                x_min: domain.extent.x_min,
                x_max: domain.extent.x_max,
                y_min: domain.extent.y_min,
                y_max: domain.extent.y_max,
            }),
            projected_grid,
            inverse_projected_grid,
            rgba_grid,
            projected_polygons,
            projected_data_polygons,
            mesh_cells: request.mesh_cells.as_ref().map(|layer| {
                crate::render::MeshCellsOverlay {
                    cells: layer.cells.clone(),
                    style: crate::mesh_cells::MeshDrawStyle::from_theme(
                        theme::active_theme().mesh,
                        1,
                    ),
                }
            }),
            projected_place_labels,
            projected_points,
            projected_lines,
            contours,
            barbs,
            streamlines,
            presentation,
            empty_note: request.empty_note.clone(),
        };

        let state_timing = RenderStateTiming {
            validate_ms,
            data_buffer_ms,
            projected_grid_ms,
            projected_lines_ms,
            projected_polygons_ms,
            contour_prep_ms,
            barb_prep_ms,
            state_prep_ms: total_start.elapsed().as_millis(),
        };

        let result = render(&data, shape.ny, shape.nx, &opts);
        scratch.reclaim_render_opts(opts, data);
        result.map(|(value, png_timing)| (value, state_timing, png_timing))
    })
}

/// Build the exact [`LeveledColormap`] the rasterizer uses for `scale` under
/// `options`: the same function the PNG render path calls, exposed so
/// external viewers (e.g. the egui data viewer) can color values with
/// literally `cmap.map(value)` and stay bit-identical to the plot output.
///
/// To match a production render, pass the request's options filtered through
/// the active plot style:
/// `ColormapBuildOptions { render_density: StaticPlotStyle::from_env().render_density(request.render_density), legend: request.legend }`.
pub fn build_colormap(scale: &ColorScale, options: ColormapBuildOptions) -> LeveledColormap {
    let discrete = scale.resolved_discrete();

    let colors: Vec<Rgba> = discrete.colors.into_iter().map(Into::into).collect();
    LeveledColormap::from_palette_with_options(
        &colors,
        &discrete.levels,
        discrete.extend.into(),
        discrete.mask_below,
        options,
    )
}

/// The colorbar tick VALUES the PNG renderer would label for `cmap` with the
/// request's `cbar_tick_step`: the same `pick_ticks` over the same legend
/// levels the production colorbar uses, or for a category colormap the
/// code at the centre of every band. Label the whole set with
/// [`format_tick_labels`] and position each value at [`legend_tick_rel`] to
/// reproduce the production colorbar's numbers exactly.
pub fn colorbar_ticks(cmap: &LeveledColormap, cbar_tick_step: Option<f64>) -> Vec<f64> {
    crate::render::legend_ticks(cmap, cbar_tick_step)
}

/// The unit label a colour bar carries for a field stored in `units`, or
/// `None` for a dimensionless field.
///
/// WHAT BREAKAGE THIS PREVENTS (gate law): a dimensionless field (a flag, an
/// index such as Haines, a 0/1 category) stores its units as `1` or `0/1`,
/// and the bar printed that above itself, where it read as a tick value (R10
/// look, 2.9.4). The title already names such a quantity; a bare number is
/// not a unit.
pub fn legend_units_label(units: &str) -> Option<String> {
    let trimmed = units.trim();
    let dimensionless = matches!(
        trimmed.to_ascii_lowercase().as_str(),
        "" | "1" | "0/1" | "-" | "none" | "dimensionless" | "unitless" | "(dimensionless)"
    );
    (!dimensionless).then(|| units.to_string())
}

const OVERLAY_ONLY_FILL_VALUE: f64 = 0.5;

fn blank_fill_colormap() -> LeveledColormap {
    static BLANK_FILL_COLORMAP: OnceLock<LeveledColormap> = OnceLock::new();
    BLANK_FILL_COLORMAP
        .get_or_init(|| {
            LeveledColormap::from_palette(&[Rgba::TRANSPARENT], &[0.0, 1.0], Extend::Neither, None)
        })
        .clone()
}

fn default_title(field: &Field2D) -> Option<String> {
    match &field.product {
        ProductKey::Named(name) if !name.is_empty() => Some(name.clone()),
        _ => None,
    }
}

fn validate_request(request: &MapRenderRequest) -> Result<(), RustwxRenderError> {
    let expected = request.field.grid.shape.len();

    if let Some(rgba_grid) = &request.rgba_grid {
        if rgba_grid.grid.shape != request.field.grid.shape || rgba_grid.pixels.len() != expected {
            return Err(RustwxRenderError::LayerShapeMismatch {
                layer: "rgba_grid",
                expected,
                actual: rgba_grid.pixels.len(),
            });
        }
    }

    if let Some(domain) = &request.projected_domain {
        if request.field.grid.shape.nx < 2 || request.field.grid.shape.ny < 2 {
            return Err(RustwxRenderError::DegenerateProjectedGrid);
        }
        if domain.x.len() != domain.y.len() {
            return Err(RustwxRenderError::InvalidProjectedGrid);
        }
        if domain.x.len() != expected {
            return Err(RustwxRenderError::LayerShapeMismatch {
                layer: "projected_domain",
                expected,
                actual: domain.x.len(),
            });
        }
    }

    for layer in &request.contours {
        if layer.data.len() != expected {
            return Err(RustwxRenderError::LayerShapeMismatch {
                layer: "contour",
                expected,
                actual: layer.data.len(),
            });
        }
    }

    for layer in &request.wind_barbs {
        if layer.u.len() != expected {
            return Err(RustwxRenderError::LayerShapeMismatch {
                layer: "wind_barb_u",
                expected,
                actual: layer.u.len(),
            });
        }
        if layer.v.len() != expected {
            return Err(RustwxRenderError::LayerShapeMismatch {
                layer: "wind_barb_v",
                expected,
                actual: layer.v.len(),
            });
        }
    }

    for layer in &request.wind_streamlines {
        if layer.u.len() != expected {
            return Err(RustwxRenderError::LayerShapeMismatch {
                layer: "wind_streamline_u",
                expected,
                actual: layer.u.len(),
            });
        }
        if layer.v.len() != expected {
            return Err(RustwxRenderError::LayerShapeMismatch {
                layer: "wind_streamline_v",
                expected,
                actual: layer.v.len(),
            });
        }
    }

    Ok(())
}

impl From<Color> for Rgba {
    fn from(value: Color) -> Self {
        Self {
            r: value.r,
            g: value.g,
            b: value.b,
            a: value.a,
        }
    }
}

impl From<Rgba> for Color {
    fn from(value: Rgba) -> Self {
        Self {
            r: value.r,
            g: value.g,
            b: value.b,
            a: value.a,
        }
    }
}

impl From<ExtendMode> for Extend {
    fn from(value: ExtendMode) -> Self {
        match value {
            ExtendMode::Neither => Self::Neither,
            ExtendMode::Min => Self::Min,
            ExtendMode::Max => Self::Max,
            ExtendMode::Both => Self::Both,
        }
    }
}

pub fn draw_centered_text_line(img: &mut RgbaImage, text: &str, y: i32, color: Color, scale: u32) {
    text::draw_text_centered(img, text, y, color.into(), scale);
}

pub fn draw_centered_text_line_with_factor(
    img: &mut RgbaImage,
    text: &str,
    y: i32,
    color: Color,
    scale: u32,
    size_factor: f32,
) {
    let width = text::text_width_bold_with_factor(text, scale, size_factor) as i32;
    let x = ((img.width() as i32) - width) / 2;
    text::draw_text_bold_with_factor(img, text, x, y, color.into(), scale, size_factor);
}

pub fn draw_text_line(img: &mut RgbaImage, text: &str, x: i32, y: i32, color: Color, scale: u32) {
    text::draw_text(img, text, x, y, color.into(), scale);
}

pub fn draw_text_line_with_factor(
    img: &mut RgbaImage,
    text: &str,
    x: i32,
    y: i32,
    color: Color,
    scale: u32,
    size_factor: f32,
) {
    text::draw_text_with_factor(img, text, x, y, color.into(), scale, size_factor);
}

pub fn draw_right_text_line(
    img: &mut RgbaImage,
    text: &str,
    x_right: i32,
    y: i32,
    color: Color,
    scale: u32,
) {
    text::draw_text_right(img, text, x_right, y, color.into(), scale);
}

pub fn draw_right_text_line_with_factor(
    img: &mut RgbaImage,
    text: &str,
    x_right: i32,
    y: i32,
    color: Color,
    scale: u32,
    size_factor: f32,
) {
    let width = text::text_width_with_factor(text, scale, size_factor) as i32;
    text::draw_text_with_factor(
        img,
        text,
        x_right - width,
        y,
        color.into(),
        scale,
        size_factor,
    );
}

#[cfg(test)]
mod tests;
