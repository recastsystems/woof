use crate::color::Rgba;
use crate::chrome_plan::{PlanHeaderText, draw_plan_colorbar, draw_plan_header};
use crate::colorbar;
use crate::layout_plan::{CanvasPlan, canvas_plan_for};
use crate::colormap::LeveledColormap;
use crate::draw;
use crate::overlay::{
    BarbOverlay, ContourOverlay, InverseProjectedGrid, MapExtent, ProjectedGrid,
    ProjectedPlaceLabelOverlay, ProjectedPointOverlay, ProjectedPolygon, ProjectedPolyline,
    StreamlineOverlay,
};
use crate::presentation::{
    ColorbarOrientation, ProductVisualMode, RenderPresentation, TitleAnchor,
};
use crate::rasterize;
use crate::request::{
    ChromeScale, DomainFrame, DomainFrameSource, ProjectedLabelPlacement, ProjectedMarkerShape,
    ProjectedPlaceLabelPriority, RasterSampleMode,
};
use crate::text;
use image::ExtendedColorType;
use image::ImageEncoder;
use image::RgbaImage;
use image::codecs::png::{CompressionType, FilterType as PngFilterType, PngEncoder};
use image::imageops::{FilterType, crop_imm, filter3x3, resize};
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::collections::{HashMap, VecDeque, hash_map::DefaultHasher};
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::time::Instant;

#[cfg(test)]
use std::cell::Cell;
#[cfg(test)]
use std::sync::Mutex;

/// Full render configuration.
#[derive(Clone)]
pub struct RenderOpts {
    pub width: u32,
    pub height: u32,
    pub cmap: LeveledColormap,
    pub background: Rgba,
    pub colorbar: bool,
    pub colorbar_units: Option<String>,
    pub title: Option<String>,
    pub subtitle_left: Option<String>,
    pub subtitle_center: Option<String>,
    pub subtitle_right: Option<String>,
    pub cbar_tick_step: Option<f64>,
    pub colorbar_mode: crate::colormap::LegendMode,
    pub chrome_scale: ChromeScale,
    pub supersample_factor: u32,
    pub supersample_sharpen: bool,
    pub raster_sample_mode: RasterSampleMode,
    pub domain_frame: Option<DomainFrame>,
    pub map_extent: Option<MapExtent>,
    pub projected_grid: Option<ProjectedGrid>,
    pub(crate) inverse_projected_grid: Option<InverseProjectedGrid>,
    pub rgba_grid: Option<Vec<Rgba>>,
    /// Filled polygons (lat/lon-derived). Drawn BEFORE the data raster so the
    /// data overlays on top; ordering within the list is bottom-to-top.
    /// Typical stack: ocean → land → lakes.
    pub projected_polygons: Vec<ProjectedPolygon>,
    pub projected_data_polygons: Vec<ProjectedPolygon>,
    /// An unstructured mesh drawn cell by cell in the variable-data pass.
    pub mesh_cells: Option<MeshCellsOverlay>,
    pub projected_place_labels: Vec<ProjectedPlaceLabelOverlay>,
    pub projected_points: Vec<ProjectedPointOverlay>,
    pub projected_lines: Vec<ProjectedPolyline>,
    pub contours: Vec<ContourOverlay>,
    pub barbs: Vec<BarbOverlay>,
    pub streamlines: Vec<StreamlineOverlay>,
    pub presentation: RenderPresentation,
    /// The product's own note for a map with nothing to draw; `None` uses
    /// [`EMPTY_MAP_NOTE`]. See [`empty_map_note`].
    pub empty_note: Option<String>,
}

/// The mesh layer as the renderer holds it: the cells and the resolved
/// style, so the supersample pass can scale the hairline without reaching
/// back into the theme.
#[derive(Debug, Clone)]
pub struct MeshCellsOverlay {
    pub cells: Vec<crate::mesh_cells::MeshCell>,
    pub style: crate::mesh_cells::MeshDrawStyle,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RenderImageTiming {
    pub layout_ms: u128,
    pub background_ms: u128,
    pub polygon_fill_ms: u128,
    pub projected_pixel_ms: u128,
    pub rasterize_ms: u128,
    pub raster_blit_ms: u128,
    pub linework_ms: u128,
    pub contour_ms: u128,
    #[serde(default)]
    pub contour_bucket_ms: u128,
    #[serde(default)]
    pub contour_extrema_ms: u128,
    #[serde(default)]
    pub contour_label_draw_ms: u128,
    #[serde(default)]
    pub contour_segment_count: u64,
    pub barb_ms: u128,
    #[serde(default)]
    pub outside_frame_clear_ms: u128,
    pub chrome_ms: u128,
    pub colorbar_ms: u128,
    #[serde(default)]
    pub downsample_ms: u128,
    pub postprocess_ms: u128,
    pub total_ms: u128,
    /// The map frame's pixel box inside the image: origin then size.
    ///
    /// gpuwm addition (VENDOR.md): `map_w`/`map_h` were already published
    /// here, but a size without an origin cannot place a point -- the
    /// title row and the colourbar push the map down and right by tens of
    /// pixels, which at global scale is degrees.  Anything georeferencing
    /// a finished panel needs all four.
    #[serde(default)]
    pub map_x: u32,
    #[serde(default)]
    pub map_y: u32,
    #[serde(default)]
    pub map_w: u32,
    #[serde(default)]
    pub map_h: u32,
    /// True when the map rectangle above still describes the PNG that was
    /// written.
    ///
    /// gpuwm addition (VENDOR.md).  Three post-render passes can move the
    /// map inside the canvas after the layout is fixed -- the map-viewport
    /// crop, the horizontal recentre, and the optional vertical trim.
    /// Each of them reports the offset it applied and the save path folds
    /// that offset back into `map_x`/`map_y`, so the rectangle FOLLOWS the
    /// map.  Where the moved rectangle overhangs the written image (a
    /// regional grid drawn in a frame wider than its data is recentred by
    /// more than its margin; a crop or trim cuts into the padded
    /// viewport), the rectangle is CLIPPED to the pixels that survive and
    /// the four `map_clip_*` fractions below say how much of each side
    /// went, so the projected extent can be cut by the same fraction and
    /// the published transform still describes the file.  This goes false
    /// only when no pixel of the rectangle survives inside the written
    /// image, because a rectangle with no area places nothing; the flag
    /// lets a consumer refuse instead of guessing.
    #[serde(default)]
    pub plot_rect_describes_the_png: bool,
    /// The fraction of the UNCLIPPED map rectangle's pixel span (its
    /// width or height less one, the span `PanelGeoReference` maps the
    /// extent onto) that a post-render pass cut off each side.  All zero
    /// when the whole rectangle is inside the written image.
    #[serde(default)]
    pub map_clip_left: f64,
    #[serde(default)]
    pub map_clip_right: f64,
    #[serde(default)]
    pub map_clip_top: f64,
    #[serde(default)]
    pub map_clip_bottom: f64,
    /// The written image's own size.
    ///
    /// gpuwm addition (VENDOR.md): the map-viewport crop and the vertical
    /// trim SHRINK the canvas below the requested width/height, so
    /// anything describing the finished file (the published georeference)
    /// must read the file's size here, not the request's.
    #[serde(default)]
    pub image_w: u32,
    #[serde(default)]
    pub image_h: u32,
    /// Net movement the post-render passes applied to the map inside the
    /// canvas: positive x is rightward, positive y is downward, and both
    /// are already folded into `map_x`/`map_y` above.
    ///
    /// gpuwm addition (VENDOR.md).  Published separately so a consumer
    /// (or a test) can reconstruct the PRE-pass layout exactly -- which is
    /// how the pixel gate proves an unadjusted rectangle is measurably
    /// wrong rather than trusting the adjustment by construction.
    #[serde(default)]
    pub postprocess_offset_x: i64,
    #[serde(default)]
    pub postprocess_offset_y: i64,
    #[serde(default)]
    pub has_projected_grid: bool,
    #[serde(default)]
    pub has_inverse_raster: bool,
    #[serde(default)]
    pub projection_clip_mask_present: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub domain_clip_rect: Option<[u32; 4]>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RenderPngTiming {
    pub image_timing: RenderImageTiming,
    #[serde(default)]
    pub render_to_image_ms: u128,
    pub png_encode_ms: u128,
    #[serde(default)]
    pub png_write_ms: u128,
    pub total_ms: u128,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum PngCompressionMode {
    #[default]
    Default,
    Fast,
    Fastest,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PngWriteOptions {
    #[serde(default)]
    pub compression: PngCompressionMode,
}

impl Default for PngWriteOptions {
    fn default() -> Self {
        Self {
            compression: PngCompressionMode::Default,
        }
    }
}

impl Default for RenderOpts {
    fn default() -> Self {
        Self {
            width: 1100,
            height: 850,
            cmap: LeveledColormap {
                levels: vec![],
                colors: vec![],
                legend_levels: vec![],
                legend_colors: vec![],
                under_color: None,
                over_color: None,
                mask_below: None,
                categories: false,
                equal_bands: false,
            },
            background: Rgba::WHITE,
            colorbar: true,
            colorbar_units: None,
            title: None,
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
            map_extent: None,
            projected_grid: None,
            inverse_projected_grid: None,
            rgba_grid: None,
            projected_polygons: vec![],
            projected_data_polygons: vec![],
            mesh_cells: None,
            projected_place_labels: vec![],
            projected_points: vec![],
            projected_lines: vec![],
            contours: vec![],
            barbs: vec![],
            streamlines: vec![],
            presentation: RenderPresentation::for_mode_from_env(
                ProductVisualMode::FilledMeteorology,
            ),
            empty_note: None,
        }
    }
}

struct Layout {
    map_x: u32,
    map_y: u32,
    map_w: u32,
    map_h: u32,
    cbar_x: u32,
    cbar_y: u32,
    cbar_w: u32,
    cbar_h: u32,
    title_y: u32,
    subtitle_y: u32,
    text_scale: u32,
    label_gap: u32,
    /// Label halo (opaque; each use sets its own alpha).  White unless a
    /// theme names the surface colour.
    halo: Rgba,
    /// Title size factor from the theme; 1.0 draws exactly as before.
    title_factor: f32,
    /// Subtitle and colorbar label size factor from the theme.
    label_factor: f32,
    /// The planned geometry, when the canvas was sized from the grid
    /// (`layout_plan`).  The map and bar rectangles above are copied from
    /// it; the chrome reads its header rows and font sizes.
    plan: Option<CanvasPlan>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct LocalRect {
    min_x: u32,
    max_x: u32,
    min_y: u32,
    max_y: u32,
}

impl LocalRect {
    fn from_bounds(bounds: (u32, u32, u32, u32)) -> Self {
        let (min_x, max_x, min_y, max_y) = bounds;
        Self {
            min_x,
            max_x,
            min_y,
            max_y,
        }
    }

    fn width(self) -> u32 {
        self.max_x.saturating_sub(self.min_x).saturating_add(1)
    }

    fn height(self) -> u32 {
        self.max_y.saturating_sub(self.min_y).saturating_add(1)
    }

    fn expanded_within(self, padding: u32, max_w: u32, max_h: u32) -> Self {
        Self {
            min_x: self.min_x.saturating_sub(padding),
            max_x: self
                .max_x
                .saturating_add(padding)
                .min(max_w.saturating_sub(1)),
            min_y: self.min_y.saturating_sub(padding),
            max_y: self
                .max_y
                .saturating_add(padding)
                .min(max_h.saturating_sub(1)),
        }
    }
}

#[derive(Clone)]
struct CachedProjectedPixels {
    grid_hash: u64,
    nx: usize,
    ny: usize,
    map_w: u32,
    map_h: u32,
    extent_bits: [u64; 4],
    pixels: Arc<[Option<(f32, f32)>]>,
}

impl CachedProjectedPixels {
    fn new(
        grid: &ProjectedGrid,
        extent: &MapExtent,
        layout: &Layout,
        pixels: Arc<[Option<(f32, f32)>]>,
    ) -> Self {
        Self {
            grid_hash: hash_projected_grid(grid),
            nx: grid.nx,
            ny: grid.ny,
            map_w: layout.map_w,
            map_h: layout.map_h,
            extent_bits: extent_bits(extent),
            pixels,
        }
    }

    fn matches(&self, grid: &ProjectedGrid, extent: &MapExtent, layout: &Layout) -> bool {
        self.nx == grid.nx
            && self.ny == grid.ny
            && self.map_w == layout.map_w
            && self.map_h == layout.map_h
            && self.extent_bits == extent_bits(extent)
            && self.grid_hash == hash_projected_grid(grid)
    }
}

#[derive(Clone)]
struct CachedStaticBase {
    key: u64,
    image: RgbaImage,
}

struct VariableLayerTiming {
    rasterize_ms: u128,
    raster_blit_ms: u128,
    linework_ms: u128,
    contour_ms: u128,
    contour_profile: ContourDrawTiming,
    barb_ms: u128,
    outside_frame_clear_ms: u128,
    domain_frame_rect: Option<LocalRect>,
    domain_clip_rect: Option<LocalRect>,
    projection_clip_mask_present: bool,
}

#[derive(Debug, Clone, Copy, Default)]
struct ContourDrawTiming {
    bucket_ms: u128,
    extrema_ms: u128,
    label_draw_ms: u128,
    segment_count: u64,
}

impl ContourDrawTiming {
    fn add(&mut self, other: Self) {
        self.bucket_ms = self.bucket_ms.saturating_add(other.bucket_ms);
        self.extrema_ms = self.extrema_ms.saturating_add(other.extrema_ms);
        self.label_draw_ms = self.label_draw_ms.saturating_add(other.label_draw_ms);
        self.segment_count = self.segment_count.saturating_add(other.segment_count);
    }
}

thread_local! {
    static PROJECTED_PIXEL_CACHE: RefCell<Option<CachedProjectedPixels>> = const { RefCell::new(None) };
    static STATIC_BASE_CACHE: RefCell<Option<CachedStaticBase>> = const { RefCell::new(None) };
}

#[cfg(test)]
thread_local! {
    static PROJECTED_PIXEL_CACHE_MISSES: Cell<usize> = const { Cell::new(0) };
}
#[cfg(test)]
static PROJECTED_PIXEL_CACHE_TEST_LOCK: Mutex<()> = Mutex::new(());

fn compute_layout(
    total_w: u32,
    total_h: u32,
    has_cbar: bool,
    has_title: bool,
    presentation: RenderPresentation,
    chrome_scale: ChromeScale,
) -> Layout {
    let chrome_scale = resolve_chrome_scale(total_w, total_h, chrome_scale);
    let metrics = scaled_layout_metrics(presentation.layout, chrome_scale);
    let text_scale = text_scale_from_chrome(chrome_scale);
    let vertical_colorbar = has_cbar
        && matches!(
            presentation.colorbar.orientation,
            ColorbarOrientation::VerticalRight
        );
    let title_line_h = text::bold_line_height(text_scale);
    let subtitle_line_h = text::regular_line_height(text_scale);
    let label_gap = scale_u32(12, chrome_scale).max(subtitle_line_h.saturating_add(6));
    let header_row_gap = scale_u32(3, chrome_scale);
    let header_top_pad = scale_u32(5, chrome_scale);
    let header_bottom_pad = scale_u32(5, chrome_scale);
    let map_x = metrics.margin_x.min(total_w.saturating_sub(1));
    let title_h = if has_title {
        metrics.title_h.max(
            header_top_pad
                .saturating_add(title_line_h)
                .saturating_add(header_row_gap)
                .saturating_add(subtitle_line_h)
                .saturating_add(header_bottom_pad),
        )
    } else {
        0
    };
    let footer_h = if has_cbar && !vertical_colorbar {
        metrics
            .footer_h
            .max(metrics.colorbar_h + metrics.colorbar_gap + 10)
    } else {
        metrics.footer_h.min(18)
    };
    let map_y = title_h.min(total_h.saturating_sub(1));
    let vertical_cbar_w = metrics.colorbar_h.max(8);
    let side_legend_w = if vertical_colorbar {
        vertical_cbar_w
            .saturating_add(metrics.colorbar_gap)
            .saturating_add(metrics.colorbar_margin_x)
    } else {
        0
    };
    let map_w = total_w.saturating_sub(map_x.saturating_mul(2)).max(1);
    let map_w = map_w.saturating_sub(side_legend_w).max(1);
    let map_h = total_h
        .saturating_sub(map_y)
        .saturating_sub(footer_h)
        .max(1);
    let cbar_h = if vertical_colorbar {
        map_h
    } else if has_cbar {
        metrics.colorbar_h.max(8)
    } else {
        0
    };
    let cbar_x = if vertical_colorbar {
        map_x
            .saturating_add(map_w)
            .saturating_add(metrics.colorbar_gap)
            .min(total_w.saturating_sub(1))
    } else if has_cbar {
        map_x
            .saturating_add(metrics.colorbar_margin_x)
            .min(total_w.saturating_sub(1))
    } else {
        0
    };
    let cbar_w = if vertical_colorbar {
        vertical_cbar_w
    } else if has_cbar {
        map_w
            .saturating_sub(metrics.colorbar_margin_x.saturating_mul(2))
            .max(1)
    } else {
        0
    };
    let cbar_y = if vertical_colorbar {
        map_y
    } else if has_cbar {
        total_h
            .saturating_sub(metrics.colorbar_gap)
            .saturating_sub(cbar_h)
            .max(map_y + map_h)
    } else {
        0
    };

    Layout {
        map_x,
        map_y,
        map_w,
        map_h,
        cbar_x,
        cbar_y,
        cbar_w,
        cbar_h,
        title_y: if has_title { header_top_pad } else { 0 },
        subtitle_y: if has_title {
            header_top_pad
                .saturating_add(title_line_h)
                .saturating_add(header_row_gap)
        } else {
            0
        },
        text_scale,
        label_gap,
        halo: presentation.theme.halo_with_alpha(255),
        title_factor: presentation.theme.title_factor(),
        label_factor: presentation.theme.label_factor(),
        plan: None,
    }
}

fn compute_effective_layout(
    total_w: u32,
    total_h: u32,
    has_cbar: bool,
    has_title: bool,
    presentation: RenderPresentation,
    chrome_scale: ChromeScale,
    has_domain_frame: bool,
) -> Layout {
    let mut layout = compute_layout(
        total_w,
        total_h,
        has_cbar,
        has_title,
        presentation,
        chrome_scale,
    );
    reserve_domain_frame_legend_space(
        &mut layout,
        has_cbar,
        has_domain_frame,
        presentation.colorbar.orientation,
    );
    if let Some(plan) = canvas_plan_for(total_w, total_h) {
        apply_canvas_plan(&mut layout, plan, has_cbar);
    }
    layout
}

/// The planned rectangles replace the proportional ones.  Map labels
/// (contours, extrema, places) keep the chrome text scale they always
/// had; only the header and the bar read the plan's pixel sizes.
fn apply_canvas_plan(layout: &mut Layout, plan: CanvasPlan, has_cbar: bool) {
    layout.map_x = plan.map.x;
    layout.map_y = plan.map.y;
    layout.map_w = plan.map.w.max(1);
    layout.map_h = plan.map.h.max(1);
    match plan.bar.filter(|_| has_cbar) {
        Some(bar) => {
            layout.cbar_x = bar.x;
            layout.cbar_y = bar.y;
            layout.cbar_w = bar.w;
            layout.cbar_h = bar.h;
        }
        None => {
            layout.cbar_x = 0;
            layout.cbar_y = 0;
            layout.cbar_w = 0;
            layout.cbar_h = 0;
        }
    }
    layout.plan = Some(plan);
}

fn fit_map_viewport_layout_to_extent(
    layout: &mut Layout,
    total_w: u32,
    has_cbar: bool,
    colorbar_orientation: ColorbarOrientation,
    domain_frame: Option<DomainFrame>,
    extent: Option<&MapExtent>,
) {
    if layout.plan.is_some() {
        return;
    }
    if !matches!(domain_frame, Some(frame) if matches!(frame.source, DomainFrameSource::MapViewport))
    {
        return;
    }
    let Some(extent) = extent else {
        return;
    };
    let dx = extent.x_max - extent.x_min;
    let dy = extent.y_max - extent.y_min;
    if !dx.is_finite() || !dy.is_finite() || dx <= 0.0 || dy <= 0.0 {
        return;
    }

    let target_aspect = dx / dy;
    let current_aspect = layout.map_w as f64 / layout.map_h.max(1) as f64;
    if target_aspect >= current_aspect * 0.98 {
        return;
    }

    let old_map_w = layout.map_w;
    let new_map_w = ((layout.map_h as f64) * target_aspect)
        .round()
        .clamp(1.0, old_map_w as f64) as u32;
    if new_map_w >= old_map_w {
        return;
    }

    let old_cbar_x = layout.cbar_x;
    let cbar_gap = if has_cbar && matches!(colorbar_orientation, ColorbarOrientation::VerticalRight)
    {
        old_cbar_x.saturating_sub(layout.map_x.saturating_add(old_map_w))
    } else {
        0
    };
    let right_span =
        if has_cbar && matches!(colorbar_orientation, ColorbarOrientation::VerticalRight) {
            total_w.saturating_sub(old_cbar_x)
        } else {
            0
        };
    let group_w = new_map_w
        .saturating_add(cbar_gap)
        .saturating_add(right_span)
        .min(total_w);
    let centered_x = total_w.saturating_sub(group_w) / 2;

    layout.map_x = centered_x.max(1);
    layout.map_w = new_map_w.max(1);
    if has_cbar && matches!(colorbar_orientation, ColorbarOrientation::VerticalRight) {
        layout.cbar_x = layout
            .map_x
            .saturating_add(layout.map_w)
            .saturating_add(cbar_gap)
            .min(total_w.saturating_sub(1));
    }
}

fn reserve_domain_frame_legend_space(
    layout: &mut Layout,
    has_cbar: bool,
    has_domain_frame: bool,
    colorbar_orientation: ColorbarOrientation,
) {
    if !has_cbar
        || !has_domain_frame
        || matches!(colorbar_orientation, ColorbarOrientation::VerticalRight)
    {
        return;
    }

    let label_top = layout.cbar_y.saturating_sub(layout.label_gap);
    let required_gap = 6u32.saturating_mul(layout.text_scale.max(1));
    let max_map_bottom = label_top.saturating_sub(required_gap);
    let current_map_bottom = layout.map_y.saturating_add(layout.map_h).saturating_sub(1);
    if current_map_bottom <= max_map_bottom {
        return;
    }

    let shrink_by = current_map_bottom.saturating_sub(max_map_bottom);
    layout.map_h = layout.map_h.saturating_sub(shrink_by).max(1);
}

fn resolve_chrome_scale(total_w: u32, total_h: u32, chrome_scale: ChromeScale) -> f32 {
    match chrome_scale {
        ChromeScale::Fixed(value) => value.clamp(0.5, 4.0),
        ChromeScale::Auto {
            base_width,
            base_height,
            min,
            max,
        } => {
            let base_area = (base_width.max(1) as f64) * (base_height.max(1) as f64);
            let area = (total_w.max(1) as f64) * (total_h.max(1) as f64);
            ((area / base_area).sqrt() as f32).clamp(min, max)
        }
    }
}

fn scale_u32(value: u32, scale: f32) -> u32 {
    ((value as f32) * scale).round().max(1.0) as u32
}

fn scaled_layout_metrics(
    metrics: crate::presentation::LayoutMetrics,
    scale: f32,
) -> crate::presentation::LayoutMetrics {
    crate::presentation::LayoutMetrics {
        margin_x: scale_u32(metrics.margin_x, scale),
        title_h: scale_u32(metrics.title_h, scale),
        footer_h: scale_u32(metrics.footer_h, scale),
        colorbar_h: scale_u32(metrics.colorbar_h, scale),
        colorbar_gap: scale_u32(metrics.colorbar_gap, scale),
        colorbar_margin_x: scale_u32(metrics.colorbar_margin_x, scale),
    }
}

fn text_scale_from_chrome(chrome_scale: f32) -> u32 {
    ((chrome_scale * 3.0).ceil() as u32)
        .saturating_add(1)
        .clamp(3, 12)
}

pub fn map_frame_aspect_ratio(total_w: u32, total_h: u32, has_cbar: bool, has_title: bool) -> f64 {
    map_frame_aspect_ratio_for_mode(
        ProductVisualMode::FilledMeteorology,
        total_w,
        total_h,
        has_cbar,
        has_title,
    )
}

pub fn map_frame_aspect_ratio_for_mode(
    mode: ProductVisualMode,
    total_w: u32,
    total_h: u32,
    has_cbar: bool,
    has_title: bool,
) -> f64 {
    map_frame_aspect_ratio_for_mode_with_chrome_scale(
        mode,
        total_w,
        total_h,
        has_cbar,
        has_title,
        ChromeScale::default(),
    )
}

pub fn map_frame_aspect_ratio_for_mode_with_chrome_scale(
    mode: ProductVisualMode,
    total_w: u32,
    total_h: u32,
    has_cbar: bool,
    has_title: bool,
    chrome_scale: ChromeScale,
) -> f64 {
    // A planned canvas has one map shape whatever lane asks, so every lane
    // frames its extent to the rectangle the map is drawn in.
    if let Some(plan) = canvas_plan_for(total_w, total_h) {
        return plan.map_aspect;
    }
    let layout = compute_layout(
        total_w,
        total_h,
        has_cbar,
        has_title,
        RenderPresentation::for_mode_from_env(mode),
        chrome_scale,
    );
    layout.map_w as f64 / (layout.map_h.max(1) as f64)
}

pub fn map_frame_aspect_ratio_for_mode_with_domain_frame_and_chrome_scale(
    mode: ProductVisualMode,
    total_w: u32,
    total_h: u32,
    has_cbar: bool,
    has_title: bool,
    has_domain_frame: bool,
    chrome_scale: ChromeScale,
) -> f64 {
    let layout = compute_effective_layout(
        total_w,
        total_h,
        has_cbar,
        has_title,
        RenderPresentation::for_mode_from_env(mode),
        chrome_scale,
        has_domain_frame,
    );
    layout.map_w as f64 / (layout.map_h.max(1) as f64)
}

pub(crate) fn pick_ticks(levels: &[f64], step: Option<f64>) -> Vec<f64> {
    if levels.is_empty() {
        return vec![];
    }
    let lo = levels[0];
    let hi = levels[levels.len() - 1];

    if let Some(s) = step {
        let mut ticks = Vec::new();
        let mut v = lo;
        while v <= hi + s * 0.01 {
            ticks.push(v);
            v += s;
        }
        return ticks;
    }

    let range = hi - lo;
    if range <= 0.0 {
        return vec![lo];
    }
    let raw_step = range / 10.0;
    let mag = 10.0_f64.powf(raw_step.log10().floor());
    let nice = if raw_step / mag < 1.5 {
        mag
    } else if raw_step / mag < 3.5 {
        2.0 * mag
    } else if raw_step / mag < 7.5 {
        5.0 * mag
    } else {
        10.0 * mag
    };

    let mut ticks = Vec::new();
    let start = (lo / nice).ceil() * nice;
    let mut v = start;
    while v <= hi + nice * 0.01 {
        ticks.push(v);
        v += nice;
    }
    ticks
}

/// The values the colorbar labels: the category codes at band centres for a
/// category colormap, otherwise [`pick_ticks`] over the legend levels.
pub(crate) fn legend_ticks(cmap: &LeveledColormap, step: Option<f64>) -> Vec<f64> {
    if cmap.equal_bands {
        // A threshold bar labels its boundaries: a stepped tick between two
        // boundaries would sit inside a band at a place that means nothing.
        return cmap.legend_levels_for_display().to_vec();
    }
    cmap.category_codes()
        .unwrap_or_else(|| pick_ticks(cmap.legend_levels_for_display(), step))
}

/// A tick's place on the colour bar (0..1): linear in value, as the bar has
/// always placed ticks, except on an equal-band threshold bar, where it is
/// the boundary's own place ([`LeveledColormap::legend_fraction`]).
pub(crate) fn legend_tick_position(cmap: &LeveledColormap, tick: f64, lo: f64, range: f64) -> f64 {
    if cmap.equal_bands {
        if let Some(frac) = cmap.legend_fraction(tick) {
            return frac;
        }
    }
    (tick - lo) / range
}

pub(crate) fn colorbar_levels_for_ticks(cmap: &LeveledColormap) -> &[f64] {
    cmap.legend_levels_for_display()
}

fn measure_text_width(text: &str, scale: u32, bold: bool) -> u32 {
    if bold {
        text::text_width_bold(text, scale)
    } else {
        text::text_width(text, scale)
    }
}

fn measure_text_width_with_factor(text: &str, scale: u32, size_factor: f32, bold: bool) -> u32 {
    if bold {
        text::text_width_bold_with_factor(text, scale, size_factor)
    } else {
        text::text_width_with_factor(text, scale, size_factor)
    }
}

fn centered_text_left(text: &str, center_x: u32, scale: u32, bold: bool) -> i32 {
    let width = measure_text_width(text, scale, bold) as i32;
    center_x as i32 - width / 2
}

/// `ellipsize_text_to_width`, plus whether anything was actually dropped.
///
/// The plain form returns a bare `String`, so a caller cannot tell a
/// subtitle that fitted from one that lost its tail -- which is how the
/// provenance line came to be cut at "| WRF..." on ordinary domains with
/// nothing anywhere saying so (B-13).
fn fit_text_to_width(text: &str, max_width: u32, scale: u32, bold: bool) -> (String, bool) {
    let fitted = ellipsize_text_to_width(text, max_width, scale, bold);
    let truncated = fitted != text;
    (fitted, truncated)
}

/// Say once, per distinct text, that a subtitle lost its tail.
///
/// Deduplicated because a batch render draws the same provenance line on
/// every product in the catalog, and 150 copies of one warning is a way
/// of not being read.  Silence was the actual defect: the plot claims to
/// carry its provenance and the cut is invisible in the image.
///
/// The say-once bookkeeping lives in [`crate::advisory`] rather than here,
/// because under a batch render's worker pool the copy that survives must
/// be chosen by catalog position, not by which thread arrived first.
fn warn_subtitle_truncated(slot: &str, full: &str, fitted: &str) {
    crate::advisory::advise_once(
        crate::advisory::subtitle_truncated_key(slot, full),
        format!(
            "warning: the {slot} subtitle does not fit the plot width and was \
             drawn as {fitted:?}; the full text is {full:?} -- widen the image \
             (--size WxH) to keep the whole provenance line"
        ),
    );
}

/// Width for the left half of the subtitle row, given what the right
/// half actually needs.
///
/// This used to be `subtitle_available / 2` whenever a right subtitle
/// existed at all.  The right subtitle is the source label -- "source:
/// ArWen", about a seventh of the row -- so a fixed half-and-half split
/// spent 50% of the width on it and ellipsized the provenance to fit the
/// other 50%, while a third of the row stayed empty.  Measure the right
/// text, give it what it needs, and the left keeps the rest.
fn subtitle_left_width(
    available: u32,
    right: Option<&str>,
    scale: u32,
    gap: u32,
) -> u32 {
    let Some(right) = right.map(str::trim).filter(|text| !text.is_empty()) else {
        return available;
    };
    let needed = measure_text_width(right, scale, false).saturating_add(gap);
    // Never give the right MORE than the old even split: a pathologically
    // long source label must not squeeze the provenance below what it
    // used to get.
    let reserved = needed.min(available / 2);
    available.saturating_sub(reserved).max(1)
}

fn ellipsize_text_to_width(text: &str, max_width: u32, scale: u32, bold: bool) -> String {
    ellipsize_text_to_width_with_factor(text, max_width, scale, 1.0, bold)
}

fn fit_chrome_title_metadata(
    title: Option<&str>,
    metadata: Option<&str>,
    row_width: u32,
    _row_gap: u32,
    scale: u32,
) -> (Option<String>, Option<String>) {
    let row_width = row_width.max(1);
    let fitted_title = title.map(|text| ellipsize_text_to_width(text, row_width, scale, true));
    let fitted_metadata =
        metadata.map(|text| ellipsize_text_to_width(text, row_width, scale, false));
    (fitted_title, fitted_metadata)
}

fn ellipsize_text_to_width_with_factor(
    text: &str,
    max_width: u32,
    scale: u32,
    size_factor: f32,
    bold: bool,
) -> String {
    if measure_text_width_with_factor(text, scale, size_factor, bold) <= max_width {
        return text.to_string();
    }

    let ellipsis = "...";
    let ellipsis_w = measure_text_width_with_factor(ellipsis, scale, size_factor, bold);
    if ellipsis_w >= max_width {
        return ellipsis.to_string();
    }

    let mut kept = String::new();
    for ch in text.chars() {
        let mut candidate = kept.clone();
        candidate.push(ch);
        candidate.push_str(ellipsis);
        if measure_text_width_with_factor(&candidate, scale, size_factor, bold) > max_width {
            break;
        }
        kept.push(ch);
    }

    if kept.is_empty() {
        ellipsis.to_string()
    } else {
        format!("{kept}{ellipsis}")
    }
}

#[cfg_attr(not(test), allow(dead_code))]
fn filter_tick_labels_to_fit(
    ticks: &[f64],
    lo: f64,
    range: f64,
    cbar_x: u32,
    cbar_w: u32,
    label_left_bound: u32,
    label_right_bound: u32,
    img_w: u32,
    text_scale: u32,
) -> Vec<(f64, i32, String)> {
    let fracs: Vec<f64> = ticks.iter().map(|tick| (tick - lo) / range).collect();
    filter_tick_labels_at(
        ticks, &fracs, cbar_x, cbar_w, label_left_bound, label_right_bound, img_w, text_scale,
    )
}

/// [`filter_tick_labels_to_fit`] with each tick's place on the bar given
/// (`fracs`, 0..1), so an equal-band threshold bar places its labels where
/// its boundaries are drawn.
fn filter_tick_labels_at(
    ticks: &[f64],
    fracs: &[f64],
    cbar_x: u32,
    cbar_w: u32,
    label_left_bound: u32,
    label_right_bound: u32,
    img_w: u32,
    text_scale: u32,
) -> Vec<(f64, i32, String)> {
    let min_gap_px = (6 * text_scale.max(1)) as i32;
    let min_x = label_left_bound.min(img_w.saturating_sub(1)) as i32;
    let max_x = label_right_bound.min(img_w) as i32;
    if max_x <= min_x {
        return Vec::new();
    }
    let mut labels = Vec::with_capacity(ticks.len());
    let mut last_right = i32::MIN / 4;

    for ((tick_val, label), frac) in ticks.iter().zip(text::format_tick_labels(ticks)).zip(fracs) {
        let px = cbar_x as f64 + frac * cbar_w as f64;
        let lw = text::text_width(&label, text_scale) as i32;
        if lw >= max_x.saturating_sub(min_x) {
            continue;
        }
        let centered_lx = (px as i32) - (lw / 2);
        let max_lx = max_x.saturating_sub(lw);
        let lx = centered_lx.clamp(min_x, max_lx.max(min_x));
        if !labels.is_empty() && lx <= last_right.saturating_add(min_gap_px) {
            continue;
        }
        last_right = lx.saturating_add(lw);
        labels.push((*tick_val, lx, label));
    }

    labels
}

#[cfg_attr(not(test), allow(dead_code))]
fn filter_vertical_tick_labels_to_fit(
    ticks: &[f64],
    lo: f64,
    range: f64,
    cbar_y: u32,
    cbar_h: u32,
    label_top_bound: u32,
    label_bottom_bound: u32,
    img_h: u32,
    text_scale: u32,
) -> Vec<(f64, i32, String)> {
    let fracs: Vec<f64> = ticks.iter().map(|tick| (tick - lo) / range).collect();
    filter_vertical_tick_labels_at(
        ticks, &fracs, cbar_y, cbar_h, label_top_bound, label_bottom_bound, img_h, text_scale,
    )
}

/// [`filter_vertical_tick_labels_to_fit`] with each tick's place on the bar
/// given (`fracs`, 0 at the bottom, 1 at the top).
fn filter_vertical_tick_labels_at(
    ticks: &[f64],
    fracs: &[f64],
    cbar_y: u32,
    cbar_h: u32,
    label_top_bound: u32,
    label_bottom_bound: u32,
    img_h: u32,
    text_scale: u32,
) -> Vec<(f64, i32, String)> {
    let line_h = text::regular_line_height(text_scale) as i32;
    let min_gap_px = (2 * text_scale.max(1)) as i32;
    let min_y = label_top_bound.min(img_h.saturating_sub(1)) as i32;
    let max_y = label_bottom_bound.min(img_h) as i32;
    if max_y <= min_y {
        return Vec::new();
    }

    let mut candidates = Vec::with_capacity(ticks.len());
    for ((tick_val, label), &frac) in ticks.iter().zip(text::format_tick_labels(ticks)).zip(fracs) {
        if !frac.is_finite() {
            continue;
        }
        let py = cbar_y as f64 + (1.0 - frac) * cbar_h as f64;
        let centered_y = (py.round() as i32) - (line_h / 2);
        let max_label_y = max_y.saturating_sub(line_h);
        let y = centered_y.clamp(min_y, max_label_y.max(min_y));
        candidates.push((*tick_val, y, label));
    }
    candidates.sort_by_key(|(_, y, _)| *y);

    let mut labels = Vec::with_capacity(candidates.len());
    let mut last_bottom = i32::MIN / 4;
    for (tick_val, y, label) in candidates {
        if !labels.is_empty() && y <= last_bottom.saturating_add(min_gap_px) {
            continue;
        }
        last_bottom = y.saturating_add(line_h);
        labels.push((tick_val, y, label));
    }

    labels
}

fn grid_to_pixel(i: f64, j: f64, nx: usize, ny: usize, layout: &Layout) -> (f64, f64) {
    let x = layout.map_x as f64
        + i / (nx.saturating_sub(1).max(1)) as f64 * (layout.map_w.saturating_sub(1)) as f64;
    let y = layout.map_y as f64
        + (1.0 - j / (ny.saturating_sub(1).max(1)) as f64)
            * (layout.map_h.saturating_sub(1)) as f64;
    (x, y)
}

fn mask_contains_local_pixel(mask: &RgbaImage, x: f64, y: f64) -> bool {
    if !x.is_finite() || !y.is_finite() {
        return false;
    }
    let px = x.round() as i32;
    let py = y.round() as i32;
    if px < 0 || py < 0 || px >= mask.width() as i32 || py >= mask.height() as i32 {
        return false;
    }
    mask.get_pixel(px as u32, py as u32).0[3] > 0
}

fn segment_intersects_mask(mask: &RgbaImage, x0: f64, y0: f64, x1: f64, y1: f64) -> bool {
    const SAMPLE_STEPS: [f64; 5] = [0.0, 0.25, 0.5, 0.75, 1.0];
    SAMPLE_STEPS.iter().any(|t| {
        let x = x0 + (x1 - x0) * t;
        let y = y0 + (y1 - y0) * t;
        mask_contains_local_pixel(mask, x, y)
    })
}

fn build_alpha_clip_mask(img: &RgbaImage) -> Option<RgbaImage> {
    let mut has_transparent = false;
    let mut has_opaque = false;
    let mut mask = RgbaImage::new(img.width(), img.height());
    for py in 0..img.height() {
        for px in 0..img.width() {
            if img.get_pixel(px, py).0[3] > 0 {
                has_opaque = true;
                mask.put_pixel(px, py, Rgba::WHITE.to_image_rgba());
            } else {
                has_transparent = true;
            }
        }
    }
    (has_opaque && has_transparent).then_some(mask)
}

fn project_ring_unclipped(
    extent: &MapExtent,
    ring: &[(f64, f64)],
    layout: &Layout,
) -> Vec<(f64, f64)> {
    let dx = extent.x_max - extent.x_min;
    let dy = extent.y_max - extent.y_min;
    if dx.abs() < 1e-12 || dy.abs() < 1e-12 {
        return Vec::new();
    }
    let w = layout.map_w.saturating_sub(1) as f64;
    let h = layout.map_h.saturating_sub(1) as f64;
    ring.iter()
        .map(|&(x, y)| {
            let rx = (x - extent.x_min) / dx;
            let ry = 1.0 - (y - extent.y_min) / dy;
            (layout.map_x as f64 + rx * w, layout.map_y as f64 + ry * h)
        })
        .collect()
}

fn draw_projected_polygons(
    img: &mut RgbaImage,
    layout: &Layout,
    extent: &MapExtent,
    polygons: &[ProjectedPolygon],
    presentation: RenderPresentation,
    clip_rect: Option<(i32, i32, i32, i32)>,
) {
    for poly in polygons {
        if poly.rings.is_empty() {
            continue;
        }
        let style = presentation.polygon_style(poly.role, poly.color);
        if !style.visible {
            continue;
        }
        let rings: Vec<Vec<(f64, f64)>> = poly
            .rings
            .iter()
            .map(|ring| project_ring_unclipped(extent, ring, layout))
            .collect();
        draw::fill_polygon(img, &rings, style.color, clip_rect);
    }
}

/// Liang-Barsky: the parameters `(t0, t1)`, `0 <= t0 < t1 <= 1`, of the part
/// of the segment `a -> b` that lies inside the closed rectangle
/// `[min_x, max_x] x [min_y, max_y]`, or `None` when no stretch of it does
/// (a segment that only touches the rectangle at one point has no stretch).
fn clip_segment_to_rect(
    a: (f64, f64),
    b: (f64, f64),
    min_x: f64,
    max_x: f64,
    min_y: f64,
    max_y: f64,
) -> Option<(f64, f64)> {
    if ![a.0, a.1, b.0, b.1].iter().all(|value| value.is_finite()) {
        return None;
    }
    let dx = b.0 - a.0;
    let dy = b.1 - a.1;
    let mut t0 = 0.0_f64;
    let mut t1 = 1.0_f64;
    for (p, q) in [
        (-dx, a.0 - min_x),
        (dx, max_x - a.0),
        (-dy, a.1 - min_y),
        (dy, max_y - a.1),
    ] {
        if p == 0.0 {
            // Parallel to this edge: wholly outside it or never crossing it.
            if q < 0.0 {
                return None;
            }
            continue;
        }
        let r = q / p;
        if p < 0.0 {
            if r > t1 {
                return None;
            }
            t0 = t0.max(r);
        } else {
            if r < t0 {
                return None;
            }
            t1 = t1.min(r);
        }
    }
    (t1 > t0).then_some((t0, t1))
}

/// Draws basemap linework cut at the map rectangle.
///
/// Every segment is clipped to the map rectangle before it is drawn, and the
/// stroke itself is confined to that rectangle, so no line reaches past the
/// frame into the margin. `MapExtent::to_pixel` keeps vertices up to 10 % of
/// the extent outside the map so that a line leaving the map is still drawn up
/// to the edge; before this cut, the polyline kept those vertices as they
/// were, and county lines leaving the bottom of a 12 km map were drawn in the
/// margin below the frame. The clip mask then decides, per clipped segment,
/// whether the segment is drawn at all: one that touches the mask is drawn up
/// to the rectangle, and the pass that clears outside the mask trims the rest.
fn draw_projected_lines(
    img: &mut RgbaImage,
    layout: &Layout,
    extent: &MapExtent,
    lines: &[ProjectedPolyline],
    presentation: RenderPresentation,
    clip_mask: Option<&RgbaImage>,
) {
    if layout.map_w == 0 || layout.map_h == 0 {
        return;
    }
    let local_max_x = layout.map_w.saturating_sub(1) as f64;
    let local_max_y = layout.map_h.saturating_sub(1) as f64;
    let pixel_clip = (
        layout.map_x as i32,
        layout.map_y as i32,
        layout.map_x.saturating_add(layout.map_w).saturating_sub(1) as i32,
        layout.map_y.saturating_add(layout.map_h).saturating_sub(1) as i32,
    );
    let to_canvas = |(x, y): (f64, f64)| (layout.map_x as f64 + x, layout.map_y as f64 + y);
    let along = |a: (f64, f64), b: (f64, f64), t: f64| {
        // The unclipped ends stay bit-identical to the projected vertex.
        if t <= 0.0 {
            a
        } else if t >= 1.0 {
            b
        } else {
            (a.0 + (b.0 - a.0) * t, a.1 + (b.1 - a.1) * t)
        }
    };
    // Collect all projected+clipped polylines first so we can either
    // dispatch them all as one GPU batch (single canvas round-trip) or
    // fall back to per-polyline CPU drawing.
    let mut chunks: Vec<(Vec<(f64, f64)>, crate::color::Rgba, u32)> = Vec::new();
    let push_chunk =
        |current: &mut Vec<(f64, f64)>,
         color: crate::color::Rgba,
         width: u32,
         chunks: &mut Vec<(Vec<(f64, f64)>, crate::color::Rgba, u32)>| {
            if current.len() >= 2 {
                chunks.push((std::mem::take(current), color, width));
            } else {
                current.clear();
            }
        };
    for line in lines {
        let style = presentation.linework_style(line.role, line.color, line.width);
        if !style.visible {
            continue;
        }
        let mut current: Vec<(f64, f64)> = Vec::with_capacity(line.points.len());
        let mut previous_local: Option<(f64, f64)> = None;
        for &(x, y) in &line.points {
            let Some(point) = extent.to_pixel(x, y, layout.map_w, layout.map_h) else {
                push_chunk(&mut current, style.color, style.width, &mut chunks);
                previous_local = None;
                continue;
            };
            if let Some(previous) = previous_local {
                let clipped =
                    clip_segment_to_rect(previous, point, 0.0, local_max_x, 0.0, local_max_y);
                match clipped {
                    Some((t0, t1)) => {
                        let start = along(previous, point, t0);
                        let end = along(previous, point, t1);
                        let visible = clip_mask.is_none_or(|mask| {
                            segment_intersects_mask(mask, start.0, start.1, end.0, end.1)
                        });
                        if visible {
                            // A segment that enters the rectangle starts a new
                            // piece at the edge; one that leaves it ends its
                            // piece at the edge.
                            if t0 > 0.0 || current.is_empty() {
                                push_chunk(&mut current, style.color, style.width, &mut chunks);
                                current.push(to_canvas(start));
                            }
                            current.push(to_canvas(end));
                            if t1 < 1.0 {
                                push_chunk(&mut current, style.color, style.width, &mut chunks);
                            }
                        } else {
                            push_chunk(&mut current, style.color, style.width, &mut chunks);
                        }
                    }
                    None => push_chunk(&mut current, style.color, style.width, &mut chunks),
                }
            }
            previous_local = Some(point);
        }
        push_chunk(&mut current, style.color, style.width, &mut chunks);
    }

    // NOTE: GPU linework swap was tried (cuda_draw_linework) but caused a
    // 100x regression because the per-polyline-launch ordering preservation
    // serializes sync overhead per polyline. Re-enable only when the canvas
    // can stay GPU-resident across multiple draw passes: see the
    // canvas-resident pipeline plan.
    for (points, color, width) in chunks {
        draw::draw_polyline_aa_clipped(img, &points, color, width, pixel_clip);
    }
}

fn draw_projected_points(
    img: &mut RgbaImage,
    layout: &Layout,
    extent: &MapExtent,
    points: &[ProjectedPointOverlay],
    clip_mask: Option<&RgbaImage>,
) {
    for point in points {
        let Some((px, py)) = extent.to_pixel(point.x, point.y, layout.map_w, layout.map_h) else {
            continue;
        };
        if let Some(mask) = clip_mask {
            if !mask_contains_local_pixel(mask, px, py) {
                continue;
            }
        }

        let x = layout.map_x as f64 + px.clamp(0.0, layout.map_w.saturating_sub(1) as f64);
        let y = layout.map_y as f64 + py.clamp(0.0, layout.map_h.saturating_sub(1) as f64);
        let radius = point.radius_px.max(1) as f64;
        let width = point.width_px.max(1);
        match point.shape {
            ProjectedMarkerShape::Circle => {
                draw::draw_circle_stroke_aa(img, x, y, radius, point.color, width);
            }
            ProjectedMarkerShape::Plus => {
                draw::draw_plus_marker_aa(img, x, y, radius, point.color, width);
            }
            ProjectedMarkerShape::Cross => {
                draw::draw_cross_marker_aa(img, x, y, radius, point.color, width);
            }
        }
    }
}

fn label_bounds(layout: &Layout, clip_rect: Option<LocalRect>) -> (i32, i32, i32, i32) {
    if let Some(rect) = clip_rect {
        (
            (layout.map_x + rect.min_x) as i32,
            (layout.map_x + rect.max_x) as i32,
            (layout.map_y + rect.min_y) as i32,
            (layout.map_y + rect.max_y) as i32,
        )
    } else {
        let max_x = layout.map_x.saturating_add(layout.map_w).saturating_sub(1) as i32;
        let max_y = layout.map_y.saturating_add(layout.map_h).saturating_sub(1) as i32;
        (layout.map_x as i32, max_x, layout.map_y as i32, max_y)
    }
}

fn label_line_height_with_factor(scale: u32, size_factor: f32, bold: bool) -> u32 {
    if bold {
        text::bold_line_height_with_factor(scale, size_factor)
    } else {
        text::regular_line_height_with_factor(scale, size_factor)
    }
}

fn draw_styled_text(
    img: &mut RgbaImage,
    text_value: &str,
    x: i32,
    y: i32,
    color: Rgba,
    scale: u32,
    size_factor: f32,
    bold: bool,
) {
    if bold {
        text::draw_text_bold_with_factor(img, text_value, x, y, color, scale, size_factor);
    } else {
        text::draw_text_with_factor(img, text_value, x, y, color, scale, size_factor);
    }
}

fn label_top_left(
    placement: ProjectedLabelPlacement,
    anchor_x: i32,
    anchor_y: i32,
    label_width: u32,
    label_height: u32,
) -> (i32, i32) {
    let width = label_width as i32;
    let height = label_height as i32;
    match placement {
        ProjectedLabelPlacement::Center => (anchor_x - width / 2, anchor_y - height / 2),
        ProjectedLabelPlacement::Left => (anchor_x - width, anchor_y - height / 2),
        ProjectedLabelPlacement::Right => (anchor_x, anchor_y - height / 2),
        ProjectedLabelPlacement::Above => (anchor_x - width / 2, anchor_y - height),
        ProjectedLabelPlacement::Below => (anchor_x - width / 2, anchor_y),
        ProjectedLabelPlacement::AboveLeft => (anchor_x - width, anchor_y - height),
        ProjectedLabelPlacement::AboveRight => (anchor_x, anchor_y - height),
        ProjectedLabelPlacement::BelowLeft => (anchor_x - width, anchor_y),
        ProjectedLabelPlacement::BelowRight => (anchor_x, anchor_y),
    }
}

fn draw_text_halo(
    img: &mut RgbaImage,
    text_value: &str,
    x: i32,
    y: i32,
    label_color: Rgba,
    halo_color: Rgba,
    halo_width_px: u32,
    scale: u32,
    size_factor: f32,
    bold: bool,
) {
    let halo_width_px = halo_width_px as i32;
    if halo_color.a > 0 && halo_width_px > 0 {
        for dy in -halo_width_px..=halo_width_px {
            for dx in -halo_width_px..=halo_width_px {
                if dx == 0 && dy == 0 {
                    continue;
                }
                if dx.abs().max(dy.abs()) > halo_width_px {
                    continue;
                }
                draw_styled_text(
                    img,
                    text_value,
                    x + dx,
                    y + dy,
                    halo_color,
                    scale,
                    size_factor,
                    bold,
                );
            }
        }
    }
    draw_styled_text(img, text_value, x, y, label_color, scale, size_factor, bold);
}

#[derive(Debug, Clone, Copy)]
struct PlaceLabelRenderAdjustments {
    text_size_factor: f32,
    text_alpha_factor: f32,
    halo_alpha_factor: f32,
    marker_scale_factor: f32,
    marker_alpha_factor: f32,
    outline_width_factor: f32,
    halo_width_factor: f32,
    offset_factor: f32,
}

fn place_label_render_adjustments(
    priority: ProjectedPlaceLabelPriority,
) -> PlaceLabelRenderAdjustments {
    match priority {
        ProjectedPlaceLabelPriority::Primary => PlaceLabelRenderAdjustments {
            text_size_factor: 1.0,
            text_alpha_factor: 1.0,
            halo_alpha_factor: 1.0,
            marker_scale_factor: 1.0,
            marker_alpha_factor: 1.0,
            outline_width_factor: 1.0,
            halo_width_factor: 1.0,
            offset_factor: 1.0,
        },
        ProjectedPlaceLabelPriority::Auxiliary => PlaceLabelRenderAdjustments {
            text_size_factor: 0.90,
            text_alpha_factor: 0.84,
            halo_alpha_factor: 0.78,
            marker_scale_factor: 0.82,
            marker_alpha_factor: 0.84,
            outline_width_factor: 0.75,
            halo_width_factor: 0.75,
            offset_factor: 0.92,
        },
        ProjectedPlaceLabelPriority::Micro => PlaceLabelRenderAdjustments {
            text_size_factor: 0.82,
            text_alpha_factor: 0.72,
            halo_alpha_factor: 0.62,
            marker_scale_factor: 0.68,
            marker_alpha_factor: 0.72,
            outline_width_factor: 0.50,
            halo_width_factor: 0.50,
            offset_factor: 0.85,
        },
    }
}

fn scale_alpha(color: Rgba, factor: f32) -> Rgba {
    Rgba::with_alpha(
        color.r,
        color.g,
        color.b,
        ((color.a as f32) * factor.clamp(0.0, 1.0)).round() as u8,
    )
}

fn scale_nonzero_u32(value: u32, factor: f32) -> u32 {
    if value == 0 {
        0
    } else {
        ((value as f32) * factor).round().max(1.0) as u32
    }
}

fn scale_i32(value: i32, factor: f32) -> i32 {
    ((value as f32) * factor).round() as i32
}

fn draw_projected_place_labels(
    img: &mut RgbaImage,
    layout: &Layout,
    extent: &MapExtent,
    place_labels: &[ProjectedPlaceLabelOverlay],
    clip_mask: Option<&RgbaImage>,
    clip_rect: Option<LocalRect>,
) {
    let (min_x, max_x, min_y, max_y) = label_bounds(layout, clip_rect);
    let available_width = max_x.saturating_sub(min_x).saturating_add(1) as u32;
    let available_height = max_y.saturating_sub(min_y).saturating_add(1) as u32;

    for place_label in place_labels {
        let adjustments = place_label_render_adjustments(place_label.priority);
        let Some((px, py)) =
            extent.to_pixel(place_label.x, place_label.y, layout.map_w, layout.map_h)
        else {
            continue;
        };

        if let Some(mask) = clip_mask {
            if !mask_contains_local_pixel(mask, px, py) {
                continue;
            }
        }
        // A place past the map's edge is not on the map.  The selection
        // works from the frame's lat/lon box, which a Lambert grid's corners
        // overhang, and clamping such a place onto the frame line drew a
        // city marker on the border, as if the city were there.
        if !(px >= 0.0 && py >= 0.0 && px < layout.map_w as f64 && py < layout.map_h as f64) {
            continue;
        }

        let marker_x = layout.map_x as f64 + px.clamp(0.0, layout.map_w.saturating_sub(1) as f64);
        let marker_y = layout.map_y as f64 + py.clamp(0.0, layout.map_h.saturating_sub(1) as f64);
        let marker_radius = scale_nonzero_u32(
            place_label.style.marker_radius_px,
            adjustments.marker_scale_factor,
        );
        let marker_outline_width = scale_nonzero_u32(
            place_label.style.marker_outline_width,
            adjustments.outline_width_factor,
        );
        if marker_radius > 0 {
            draw::draw_circle_fill_aa(
                img,
                marker_x,
                marker_y,
                marker_radius as f64,
                scale_alpha(
                    place_label.style.marker_fill,
                    adjustments.marker_alpha_factor,
                ),
            );
            let marker_outline = scale_alpha(
                place_label.style.marker_outline,
                adjustments.marker_alpha_factor,
            );
            if marker_outline_width > 0 && marker_outline.a > 0 {
                draw::draw_circle_stroke_aa(
                    img,
                    marker_x,
                    marker_y,
                    marker_radius as f64,
                    marker_outline,
                    marker_outline_width,
                );
            }
        }

        let Some(label) = place_label
            .label
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
        else {
            continue;
        };

        let bold = place_label.style.label_bold;
        let scale = place_label.style.label_scale.max(1);
        let fitted_label = ellipsize_text_to_width_with_factor(
            label,
            available_width.max(1),
            scale,
            adjustments.text_size_factor,
            bold,
        );
        let label_width = measure_text_width_with_factor(
            &fitted_label,
            scale,
            adjustments.text_size_factor,
            bold,
        );
        let label_height = label_line_height_with_factor(scale, adjustments.text_size_factor, bold);
        if label_width == 0 || label_height == 0 || label_height > available_height {
            continue;
        }

        let anchor_x = marker_x.round() as i32
            + scale_i32(
                place_label.style.label_offset_x_px,
                adjustments.offset_factor,
            );
        let anchor_y = marker_y.round() as i32
            + scale_i32(
                place_label.style.label_offset_y_px,
                adjustments.offset_factor,
            );
        let (mut text_x, mut text_y) = label_top_left(
            place_label.style.label_placement,
            anchor_x,
            anchor_y,
            label_width,
            label_height,
        );
        let max_label_x = max_x.saturating_sub(label_width as i32).saturating_add(1);
        let max_label_y = max_y.saturating_sub(label_height as i32).saturating_add(1);
        text_x = text_x.clamp(min_x, max_label_x.max(min_x));
        text_y = text_y.clamp(min_y, max_label_y.max(min_y));

        draw_text_halo(
            img,
            &fitted_label,
            text_x,
            text_y,
            scale_alpha(place_label.style.label_color, adjustments.text_alpha_factor),
            scale_alpha(place_label.style.label_halo, adjustments.halo_alpha_factor),
            scale_nonzero_u32(
                place_label.style.label_halo_width_px,
                adjustments.halo_width_factor,
            ),
            scale,
            adjustments.text_size_factor,
            bold,
        );
    }
}

fn projected_grid_to_pixels(
    grid: &ProjectedGrid,
    extent: &MapExtent,
    layout: &Layout,
) -> Vec<Option<(f32, f32)>> {
    grid.x
        .iter()
        .zip(grid.y.iter())
        .map(|(&x, &y)| {
            extent
                .to_pixel(x, y, layout.map_w, layout.map_h)
                .filter(|(px, py)| px.is_finite() && py.is_finite())
                .map(|(px, py)| (px as f32, py as f32))
        })
        .collect()
}

fn projected_grid_to_pixels_cached(
    grid: &ProjectedGrid,
    extent: &MapExtent,
    layout: &Layout,
) -> Arc<[Option<(f32, f32)>]> {
    if let Some(pixels) = PROJECTED_PIXEL_CACHE.with(|slot| {
        slot.borrow().as_ref().filter(|cached| cached.matches(grid, extent, layout))
            .map(|cached| Arc::clone(&cached.pixels))
    }) {
        return pixels;
    }
    let pixels: Arc<[Option<(f32, f32)>]> =
        projected_grid_to_pixels(grid, extent, layout).into();
    PROJECTED_PIXEL_CACHE.with(|slot| {
        slot.replace(Some(CachedProjectedPixels::new(
            grid, extent, layout, Arc::clone(&pixels),
        )));
    });
    #[cfg(test)]
    PROJECTED_PIXEL_CACHE_MISSES.with(|count| count.set(count.get() + 1));
    pixels
}

fn hash_rgba(hasher: &mut impl Hasher, color: Rgba) {
    color.r.hash(hasher);
    color.g.hash(hasher);
    color.b.hash(hasher);
    color.a.hash(hasher);
}

fn hash_extent(hasher: &mut impl Hasher, extent: &MapExtent) {
    extent.x_min.to_bits().hash(hasher);
    extent.x_max.to_bits().hash(hasher);
    extent.y_min.to_bits().hash(hasher);
    extent.y_max.to_bits().hash(hasher);
}

fn hash_projected_polygons(hasher: &mut impl Hasher, polygons: &[ProjectedPolygon]) {
    polygons.len().hash(hasher);
    for polygon in polygons {
        hash_rgba(hasher, polygon.color);
        std::mem::discriminant(&polygon.role).hash(hasher);
        polygon.rings.len().hash(hasher);
        for ring in &polygon.rings {
            ring.len().hash(hasher);
            for &(x, y) in ring {
                x.to_bits().hash(hasher);
                y.to_bits().hash(hasher);
            }
        }
    }
}

fn static_base_cache_key(
    opts: &RenderOpts,
    layout: &Layout,
    extent: Option<&MapExtent>,
    domain_frame_rect: Option<LocalRect>,
    canvas_background: Rgba,
    map_background: Rgba,
    draw_static_polygons: bool,
) -> u64 {
    let mut hasher = DefaultHasher::new();
    opts.width.hash(&mut hasher);
    opts.height.hash(&mut hasher);
    std::mem::discriminant(&opts.presentation.mode).hash(&mut hasher);
    std::mem::discriminant(&opts.presentation.plot_style).hash(&mut hasher);
    layout.map_x.hash(&mut hasher);
    layout.map_y.hash(&mut hasher);
    layout.map_w.hash(&mut hasher);
    layout.map_h.hash(&mut hasher);
    hash_rgba(&mut hasher, canvas_background);
    hash_rgba(&mut hasher, map_background);
    draw_static_polygons.hash(&mut hasher);
    opts.domain_frame.is_some().hash(&mut hasher);
    if let Some(frame) = opts.domain_frame {
        frame.clear_outside.hash(&mut hasher);
    }
    domain_frame_rect
        .map(|rect| (rect.min_x, rect.max_x, rect.min_y, rect.max_y).hash(&mut hasher));
    if let Some(extent) = extent {
        hash_extent(&mut hasher, extent);
    } else {
        0u8.hash(&mut hasher);
    }
    hash_projected_polygons(&mut hasher, &opts.projected_polygons);
    hasher.finish()
}

fn build_static_base_image(
    opts: &RenderOpts,
    layout: &Layout,
    extent: Option<&MapExtent>,
    domain_frame_rect: Option<LocalRect>,
    canvas_background: Rgba,
    map_background: Rgba,
    polygon_clip_rect: (i32, i32, i32, i32),
    draw_static_polygons: bool,
) -> (RgbaImage, u128, u128) {
    let background_start = Instant::now();
    let mut img = RgbaImage::from_pixel(opts.width, opts.height, canvas_background.to_image_rgba());
    if matches!(opts.domain_frame, Some(frame) if frame.clear_outside)
        && domain_frame_rect.is_some()
    {
        let rect = domain_frame_rect.expect("checked is_some above");
        for py in rect.min_y..=rect.max_y.min(layout.map_h.saturating_sub(1)) {
            for px in rect.min_x..=rect.max_x.min(layout.map_w.saturating_sub(1)) {
                img.put_pixel(
                    layout.map_x + px,
                    layout.map_y + py,
                    map_background.to_image_rgba(),
                );
            }
        }
    } else {
        let map_right = layout.map_x.saturating_add(layout.map_w).min(img.width());
        let map_bottom = layout.map_y.saturating_add(layout.map_h).min(img.height());
        for py in layout.map_y..map_bottom {
            for px in layout.map_x..map_right {
                img.put_pixel(px, py, map_background.to_image_rgba());
            }
        }
    }
    let background_ms = background_start.elapsed().as_millis();

    let polygon_start = Instant::now();
    if draw_static_polygons {
        if let Some(extent) = extent {
            draw_projected_polygons(
                &mut img,
                layout,
                extent,
                &opts.projected_polygons,
                opts.presentation,
                Some(polygon_clip_rect),
            );
        }
    }
    let polygon_fill_ms = polygon_start.elapsed().as_millis();
    (img, background_ms, polygon_fill_ms)
}

fn cached_static_base_image(
    opts: &RenderOpts,
    layout: &Layout,
    extent: Option<&MapExtent>,
    domain_frame_rect: Option<LocalRect>,
    canvas_background: Rgba,
    map_background: Rgba,
    polygon_clip_rect: (i32, i32, i32, i32),
    draw_static_polygons: bool,
) -> (RgbaImage, u128, u128) {
    let static_base_key = static_base_cache_key(
        opts,
        layout,
        extent,
        domain_frame_rect,
        canvas_background,
        map_background,
        draw_static_polygons,
    );
    if let Some(image) = STATIC_BASE_CACHE.with(|slot| {
        slot.borrow().as_ref().filter(|cached| cached.key == static_base_key)
            .map(|cached| cached.image.clone())
    }) {
        return (image, 0, 0);
    }
    // Keep TLS borrows out of drawing, which may yield to another product.
    let (image, background_ms, polygon_fill_ms) = build_static_base_image(
        opts, layout, extent, domain_frame_rect, canvas_background,
        map_background, polygon_clip_rect, draw_static_polygons,
    );
    STATIC_BASE_CACHE.with(|slot| {
        slot.replace(Some(CachedStaticBase {
            key: static_base_key,
            image: image.clone(),
        }));
    });
    (image, background_ms, polygon_fill_ms)
}

fn field_and_colormap_are_opaque(data: &[f64], opts: &RenderOpts) -> bool {
    if opts.cmap.mask_below.is_some() || opts.cmap.colors.is_empty() || opts.cmap.levels.len() < 2 {
        return false;
    }
    let opaque_color = |color: Rgba| color.a == 255;
    if !opts.cmap.colors.iter().copied().all(opaque_color) {
        return false;
    }
    if !opts.cmap.under_color.is_some_and(opaque_color) {
        return false;
    }
    if !opts.cmap.over_color.is_some_and(opaque_color) {
        return false;
    }
    data.iter().all(|value| value.is_finite())
}

fn draw_static_polygons_for_render(data: &[f64], opts: &RenderOpts) -> bool {
    !field_and_colormap_are_opaque(data, opts)
}

fn draw_projected_grid_boundary(
    img: &mut RgbaImage,
    layout: &Layout,
    grid: &ProjectedGrid,
    pixel_points: &[Option<(f32, f32)>],
    color: Rgba,
    width: u32,
) -> bool {
    if grid.nx < 2 || grid.ny < 2 || pixel_points.len() != grid.nx * grid.ny {
        return false;
    }

    let idx = |j: usize, i: usize| j * grid.nx + i;
    let mut boundary = Vec::with_capacity((grid.nx + grid.ny) * 2 + 1);
    let mut visible_min_x = f64::INFINITY;
    let mut visible_max_x = f64::NEG_INFINITY;
    let mut visible_min_y = f64::INFINITY;
    let mut visible_max_y = f64::NEG_INFINITY;

    for &(px, py) in pixel_points.iter().flatten() {
        let x = layout.map_x as f64 + px as f64;
        let y = layout.map_y as f64 + py as f64;
        visible_min_x = visible_min_x.min(x);
        visible_max_x = visible_max_x.max(x);
        visible_min_y = visible_min_y.min(y);
        visible_max_y = visible_max_y.max(y);
    }

    for i in 0..grid.nx {
        let Some((px, py)) = pixel_points[idx(0, i)] else {
            return draw_visible_projected_grid_bounds(
                img,
                visible_min_x,
                visible_max_x,
                visible_min_y,
                visible_max_y,
                color,
                width,
            );
        };
        boundary.push((
            layout.map_x as f64 + px as f64,
            layout.map_y as f64 + py as f64,
        ));
    }
    for j in 1..grid.ny {
        let Some((px, py)) = pixel_points[idx(j, grid.nx - 1)] else {
            return draw_visible_projected_grid_bounds(
                img,
                visible_min_x,
                visible_max_x,
                visible_min_y,
                visible_max_y,
                color,
                width,
            );
        };
        boundary.push((
            layout.map_x as f64 + px as f64,
            layout.map_y as f64 + py as f64,
        ));
    }
    for i in (0..grid.nx.saturating_sub(1)).rev() {
        let Some((px, py)) = pixel_points[idx(grid.ny - 1, i)] else {
            return draw_visible_projected_grid_bounds(
                img,
                visible_min_x,
                visible_max_x,
                visible_min_y,
                visible_max_y,
                color,
                width,
            );
        };
        boundary.push((
            layout.map_x as f64 + px as f64,
            layout.map_y as f64 + py as f64,
        ));
    }
    for j in (1..grid.ny.saturating_sub(1)).rev() {
        let Some((px, py)) = pixel_points[idx(j, 0)] else {
            return draw_visible_projected_grid_bounds(
                img,
                visible_min_x,
                visible_max_x,
                visible_min_y,
                visible_max_y,
                color,
                width,
            );
        };
        boundary.push((
            layout.map_x as f64 + px as f64,
            layout.map_y as f64 + py as f64,
        ));
    }

    if let Some(first) = boundary.first().copied() {
        boundary.push(first);
    }

    if boundary.len() >= 2 {
        draw::draw_polyline_aa(img, &boundary, color, width);
        true
    } else {
        false
    }
}

fn draw_visible_projected_grid_bounds(
    img: &mut RgbaImage,
    min_x: f64,
    max_x: f64,
    min_y: f64,
    max_y: f64,
    color: Rgba,
    width: u32,
) -> bool {
    if !min_x.is_finite() || !max_x.is_finite() || !min_y.is_finite() || !max_y.is_finite() {
        return false;
    }
    draw::draw_polyline_aa(
        img,
        &[
            (min_x, min_y),
            (max_x, min_y),
            (max_x, max_y),
            (min_x, max_y),
            (min_x, min_y),
        ],
        color,
        width,
    );
    true
}

fn raster_alpha_bounds(map_img: &RgbaImage) -> Option<(u32, u32, u32, u32)> {
    const OUTLINE_ALPHA_THRESHOLD: u8 = 128;
    let mut min_x: Option<u32> = None;
    let mut max_x: Option<u32> = None;
    let mut min_y: Option<u32> = None;
    let mut max_y: Option<u32> = None;

    for py in 0..map_img.height() {
        for px in 0..map_img.width() {
            if map_img.get_pixel(px, py).0[3] < OUTLINE_ALPHA_THRESHOLD {
                continue;
            }
            min_x = Some(min_x.map_or(px, |v| v.min(px)));
            max_x = Some(max_x.map_or(px, |v| v.max(px)));
            min_y = Some(min_y.map_or(py, |v| v.min(py)));
            max_y = Some(max_y.map_or(py, |v| v.max(py)));
        }
    }

    match (min_x, max_x, min_y, max_y) {
        (Some(min_x), Some(max_x), Some(min_y), Some(max_y)) => Some((min_x, max_x, min_y, max_y)),
        _ => None,
    }
}

fn inset_rect(bounds: LocalRect, inset: u32) -> Option<LocalRect> {
    let LocalRect {
        min_x,
        max_x,
        min_y,
        max_y,
    } = bounds;
    if max_x <= min_x.saturating_add(inset.saturating_mul(2))
        || max_y <= min_y.saturating_add(inset.saturating_mul(2))
    {
        return None;
    }
    Some(LocalRect {
        min_x: min_x + inset,
        max_x: max_x - inset,
        min_y: min_y + inset,
        max_y: max_y - inset,
    })
}

fn build_rect_clip_mask(map_w: u32, map_h: u32, rect: LocalRect) -> RgbaImage {
    let mut mask = RgbaImage::new(map_w, map_h);
    for py in rect.min_y..=rect.max_y.min(map_h.saturating_sub(1)) {
        for px in rect.min_x..=rect.max_x.min(map_w.saturating_sub(1)) {
            mask.put_pixel(px, py, Rgba::WHITE.to_image_rgba());
        }
    }
    mask
}

fn intersect_alpha_clip_masks(a: &RgbaImage, b: &RgbaImage) -> RgbaImage {
    let width = a.width().min(b.width());
    let height = a.height().min(b.height());
    let mut mask = RgbaImage::new(width, height);
    for py in 0..height {
        for px in 0..width {
            let alpha = a.get_pixel(px, py).0[3].min(b.get_pixel(px, py).0[3]);
            if alpha > 0 {
                mask.put_pixel(px, py, image::Rgba([255, 255, 255, alpha]));
            }
        }
    }
    mask
}

fn draw_local_rect_outline(
    img: &mut RgbaImage,
    layout: &Layout,
    rect: LocalRect,
    color: Rgba,
    width: u32,
) {
    draw::draw_polyline_aa(
        img,
        &[
            (
                layout.map_x as f64 + rect.min_x as f64,
                layout.map_y as f64 + rect.min_y as f64,
            ),
            (
                layout.map_x as f64 + rect.max_x as f64,
                layout.map_y as f64 + rect.min_y as f64,
            ),
            (
                layout.map_x as f64 + rect.max_x as f64,
                layout.map_y as f64 + rect.max_y as f64,
            ),
            (
                layout.map_x as f64 + rect.min_x as f64,
                layout.map_y as f64 + rect.max_y as f64,
            ),
            (
                layout.map_x as f64 + rect.min_x as f64,
                layout.map_y as f64 + rect.min_y as f64,
            ),
        ],
        color,
        width,
    );
}

fn covered(mask: &RgbaImage, x: u32, y: u32) -> bool {
    mask.get_pixel(x, y).0[3] > 0
}

fn row_coverage_count(mask: &RgbaImage, y: u32, x0: u32, x1: u32) -> u32 {
    (x0..=x1).filter(|&x| covered(mask, x, y)).count() as u32
}

fn col_coverage_count(mask: &RgbaImage, x: u32, y0: u32, y1: u32) -> u32 {
    (y0..=y1).filter(|&y| covered(mask, x, y)).count() as u32
}

fn inner_rect_from_coverage(mask: &RgbaImage, inset: u32) -> Option<LocalRect> {
    let (bx0, bx1, by0, by1) = raster_alpha_bounds(mask)?;
    let mut rect = LocalRect::from_bounds((bx0, bx1, by0, by1));
    const EDGE_COVERAGE_NUM: u32 = 9;
    const EDGE_COVERAGE_DEN: u32 = 10;

    for _ in 0..3 {
        let width = rect.width();
        let min_row_coverage = ((width * EDGE_COVERAGE_NUM) / EDGE_COVERAGE_DEN).max(1);
        let top = (rect.min_y..=rect.max_y)
            .find(|&y| row_coverage_count(mask, y, rect.min_x, rect.max_x) >= min_row_coverage)?;
        let bottom = (rect.min_y..=rect.max_y)
            .rev()
            .find(|&y| row_coverage_count(mask, y, rect.min_x, rect.max_x) >= min_row_coverage)?;
        rect.min_y = top;
        rect.max_y = bottom;
        if rect.min_y >= rect.max_y {
            return None;
        }

        let height = rect.height();
        let min_col_coverage = ((height * EDGE_COVERAGE_NUM) / EDGE_COVERAGE_DEN).max(1);
        let left = (rect.min_x..=rect.max_x)
            .find(|&x| col_coverage_count(mask, x, rect.min_y, rect.max_y) >= min_col_coverage)?;
        let right = (rect.min_x..=rect.max_x)
            .rev()
            .find(|&x| col_coverage_count(mask, x, rect.min_y, rect.max_y) >= min_col_coverage)?;
        rect.min_x = left;
        rect.max_x = right;
        if rect.min_x >= rect.max_x {
            return None;
        }
    }

    inset_rect(rect, inset)
}

/// Every covered pixel, trimmed by the frame inset -- the bounding box of
/// the coverage rather than a rectangle inscribed in it.
///
/// gpuwm divergence (VENDOR.md): the counterpart to
/// [`inner_rect_from_coverage`] for grids whose screen footprint is not a
/// rectangle.
fn covering_rect_from_coverage(mask: &RgbaImage, inset: u32) -> Option<LocalRect> {
    let bounds = raster_alpha_bounds(mask)?;
    inset_rect(LocalRect::from_bounds(bounds), inset)
}

/// The domain frame for a projected grid.
///
/// `inscribe_in_coverage` says whether the grid's screen footprint is an
/// axis-aligned rectangle.  When it is -- a wrfout drawn on its own map
/// projection -- the frame is the largest rectangle INSCRIBED in the
/// coverage, so the ragged antialiased rim is trimmed off the outline.
///
/// gpuwm divergence (VENDOR.md): a regular lat/lon field resampled into a
/// PRESENTATION projection (every lat/lon source at a regional window:
/// GFS, GDAS, GEFS, ICON, the AI atmospheres) has a curved footprint, and
/// on a curve the inscribed rectangle is a thin horizontal band -- the
/// three 90%-coverage erosion passes below keep shrinking because the
/// rows near the top and bottom edges really are clipped by the arcs.
/// `clear_outside` then erases everything outside that band, which is
/// most of the model domain.  Measured on a 0.25 deg 20-55N / 130-60W
/// window: 256 of 900 rows survived.  For those grids the frame is the
/// bounding box of the coverage instead, which is identical to the
/// inscribed rectangle whenever the footprint IS a rectangle, so no
/// native-projection render moves a pixel.
fn compute_projected_domain_frame_rect(
    frame: DomainFrame,
    grid: &ProjectedGrid,
    pixel_points: &[Option<(f32, f32)>],
    map_w: u32,
    map_h: u32,
    _overlay_padding_px: u32,
    inscribe_in_coverage: bool,
) -> Option<LocalRect> {
    let mask =
        rasterize::rasterize_projected_coverage_mask(grid.ny, grid.nx, pixel_points, map_w, map_h);
    if inscribe_in_coverage {
        inner_rect_from_coverage(&mask, frame.inset_px)
    } else {
        covering_rect_from_coverage(&mask, frame.inset_px)
    }
}

fn overlay_frame_padding_px(opts: &RenderOpts, layout: &Layout) -> u32 {
    let mut padding = 0u32;
    for barb in &opts.barbs {
        let width = barb.width.saturating_add(barb.halo_width.saturating_mul(2));
        padding = padding.max(barb_glyph_margin_px(barb.length_px, width).ceil() as u32);
    }
    for contour in &opts.contours {
        let line_width = contour
            .width
            .max(contour.major_width.unwrap_or(contour.width));
        let label_padding = if contour.labels {
            text::regular_line_height(1).saturating_add(8)
        } else {
            0
        };
        padding = padding.max(line_width.saturating_add(label_padding));
    }
    for point in &opts.projected_points {
        padding = padding.max(
            point
                .radius_px
                .saturating_add(point.width_px)
                .saturating_add(2),
        );
    }
    for line in &opts.projected_lines {
        padding = padding.max(line.width.saturating_add(2));
    }
    for label in &opts.projected_place_labels {
        let style = &label.style;
        let marker_padding = style
            .marker_radius_px
            .saturating_add(style.marker_outline_width)
            .saturating_add(2);
        let label_width = label
            .label
            .as_deref()
            .map(|text| {
                measure_text_width_with_factor(
                    text,
                    style.label_scale.max(1),
                    1.0,
                    style.label_bold,
                )
            })
            .unwrap_or(0);
        let label_padding = style
            .label_halo_width_px
            .saturating_mul(2)
            .saturating_add(label_width)
            .saturating_add(style.label_offset_x_px.unsigned_abs())
            .saturating_add(style.label_offset_y_px.unsigned_abs());
        padding = padding.max(marker_padding.saturating_add(label_padding));
    }

    let cap = layout.map_w.min(layout.map_h) / 3;
    padding.min(cap)
}

fn compute_domain_frame_rect(frame: DomainFrame, map_w: u32, map_h: u32) -> Option<LocalRect> {
    if map_w == 0 || map_h == 0 {
        return None;
    }
    inset_rect(
        LocalRect {
            min_x: 0,
            max_x: map_w.saturating_sub(1),
            min_y: 0,
            max_y: map_h.saturating_sub(1),
        },
        frame.inset_px,
    )
}

fn scale_render_opts_for_supersample(opts: &RenderOpts, factor: u32) -> RenderOpts {
    let factor = factor.max(1);
    let mut scaled = opts.clone();
    let resolved_chrome_scale = resolve_chrome_scale(opts.width, opts.height, opts.chrome_scale);
    scaled.width = scaled.width.saturating_mul(factor);
    scaled.height = scaled.height.saturating_mul(factor);
    if let Some(frame) = scaled.domain_frame.as_mut() {
        frame.inset_px = frame.inset_px.saturating_mul(factor);
        frame.outline_width = frame.outline_width.max(1).saturating_mul(factor);
    }
    scaled.chrome_scale = ChromeScale::Fixed(resolved_chrome_scale * factor as f32);
    for line in &mut scaled.projected_lines {
        line.width = line.width.max(1).saturating_mul(factor);
    }
    for point in &mut scaled.projected_points {
        point.radius_px = point.radius_px.max(1).saturating_mul(factor);
        point.width_px = point.width_px.max(1).saturating_mul(factor);
    }
    for place_label in &mut scaled.projected_place_labels {
        place_label.style.marker_radius_px =
            place_label.style.marker_radius_px.saturating_mul(factor);
        place_label.style.marker_outline_width = place_label
            .style
            .marker_outline_width
            .saturating_mul(factor);
        place_label.style.label_halo_width_px =
            place_label.style.label_halo_width_px.saturating_mul(factor);
        place_label.style.label_scale = place_label.style.label_scale.max(1).saturating_mul(factor);
        place_label.style.label_offset_x_px = place_label
            .style
            .label_offset_x_px
            .saturating_mul(factor as i32);
        place_label.style.label_offset_y_px = place_label
            .style
            .label_offset_y_px
            .saturating_mul(factor as i32);
    }
    if let Some(mesh) = scaled.mesh_cells.as_mut() {
        mesh.style.edge_width_px *= factor as f32;
    }
    for contour in &mut scaled.contours {
        contour.width = contour.width.max(1).saturating_mul(factor);
        contour.major_width = contour
            .major_width
            .map(|width| width.max(1).saturating_mul(factor));
    }
    for barb in &mut scaled.barbs {
        barb.width = barb.width.max(1).saturating_mul(factor);
        barb.halo_width = barb.halo_width.saturating_mul(factor);
        barb.spacing_px *= factor as f64;
        barb.length_px *= factor as f64;
    }
    for streamline in &mut scaled.streamlines {
        streamline.width = streamline.width.max(1).saturating_mul(factor);
    }
    scaled.supersample_factor = 1;
    scaled
}

fn clear_map_outside_local_rect(
    img: &mut RgbaImage,
    layout: &Layout,
    rect: LocalRect,
    clear_color: Rgba,
) {
    let x0 = layout.map_x + rect.min_x;
    let x1 = layout.map_x + rect.max_x;
    let y0 = layout.map_y + rect.min_y;
    let y1 = layout.map_y + rect.max_y;

    let clear = clear_color.to_image_rgba();
    let map_right = layout.map_x.saturating_add(layout.map_w).min(img.width());
    let map_bottom = layout.map_y.saturating_add(layout.map_h).min(img.height());

    for py in layout.map_y..map_bottom {
        for px in layout.map_x..map_right {
            if px < x0 || px > x1 || py < y0 || py > y1 {
                img.put_pixel(px, py, clear);
            }
        }
    }
}

fn clear_map_outside_local_mask(
    img: &mut RgbaImage,
    layout: &Layout,
    mask: &RgbaImage,
    background: Rgba,
) {
    let clear = background.to_image_rgba();
    let map_right = layout.map_x.saturating_add(layout.map_w).min(img.width());
    let map_bottom = layout.map_y.saturating_add(layout.map_h).min(img.height());
    for py in layout.map_y..map_bottom {
        let local_y = py - layout.map_y;
        if local_y >= mask.height() {
            continue;
        }
        for px in layout.map_x..map_right {
            let local_x = px - layout.map_x;
            if local_x < mask.width() && mask.get_pixel(local_x, local_y).0[3] == 0 {
                img.put_pixel(px, py, clear);
            }
        }
    }
}

fn local_mask_covered(mask: &RgbaImage, x: i32, y: i32) -> bool {
    if x < 0 || y < 0 || x >= mask.width() as i32 || y >= mask.height() as i32 {
        return false;
    }
    mask.get_pixel(x as u32, y as u32).0[3] > 0
}

fn draw_local_mask_outline(
    img: &mut RgbaImage,
    layout: &Layout,
    mask: &RgbaImage,
    color: Rgba,
    width: u32,
    stroked_rect: Option<(LocalRect, u32)>,
) {
    let radius = width.saturating_sub(1).min(3) as i32;
    for local_y in 0..mask.height() {
        for local_x in 0..mask.width() {
            if mask.get_pixel(local_x, local_y).0[3] == 0 {
                continue;
            }
            if let Some((rect, band)) = stroked_rect {
                let near_edge = local_x.abs_diff(rect.min_x) <= band
                    || local_x.abs_diff(rect.max_x) <= band
                    || local_y.abs_diff(rect.min_y) <= band
                    || local_y.abs_diff(rect.max_y) <= band;
                if near_edge {
                    continue;
                }
            }
            let x = local_x as i32;
            let y = local_y as i32;
            let edge = !local_mask_covered(mask, x - 1, y)
                || !local_mask_covered(mask, x + 1, y)
                || !local_mask_covered(mask, x, y - 1)
                || !local_mask_covered(mask, x, y + 1);
            if !edge {
                continue;
            }
            for dy in -radius..=radius {
                for dx in -radius..=radius {
                    draw::blend_pixel(
                        img,
                        layout.map_x as i32 + x + dx,
                        layout.map_y as i32 + y + dy,
                        color,
                    );
                }
            }
        }
    }
}

fn chrome_anchor_bounds(
    layout: &Layout,
    frame: Option<DomainFrame>,
    frame_rect: Option<LocalRect>,
) -> (u32, u32, u32) {
    if matches!(frame, Some(frame) if frame.chrome_follows_frame) {
        if let Some(rect) = frame_rect {
            let left = layout.map_x + rect.min_x;
            let right = layout.map_x + rect.max_x;
            let center = left + right.saturating_sub(left) / 2;
            return (left, right, center);
        }
    }

    let left = layout.map_x;
    let right = layout.map_x + layout.map_w;
    let center = left + right.saturating_sub(left) / 2;
    (left, right, center)
}

fn chrome_anchor_rows(
    layout: &Layout,
    frame: Option<DomainFrame>,
    frame_rect: Option<LocalRect>,
) -> (u32, u32) {
    if matches!(frame, Some(frame) if frame.chrome_follows_frame) {
        if let Some(rect) = frame_rect {
            let frame_top = layout.map_y + rect.min_y;
            let title_h = text::bold_line_height(layout.text_scale);
            let subtitle_h = text::regular_line_height(layout.text_scale);
            let bottom_gap = 5u32.saturating_mul(layout.text_scale.max(1));
            let row_gap = 2u32.saturating_mul(layout.text_scale.max(1));
            let subtitle_y = frame_top.saturating_sub(subtitle_h.saturating_add(bottom_gap));
            let title_y = subtitle_y.saturating_sub(title_h.saturating_add(row_gap));
            return (title_y, subtitle_y);
        }
    }

    (layout.title_y, layout.subtitle_y)
}

/// Where the header goes: its row bounds `(left, right, center)` and its
/// `(title, subtitle)` rows.
///
/// A header follows its domain frame so the title sits over the map it
/// names.  A narrow frame (a tall nest on a wide canvas) leaves most of
/// the header row unused while its own width cuts the subtitle, which
/// dropped the valid time from every narrow nest at any image width, so
/// the advice to widen the image could not help.  When the header text
/// does not fit over the frame, its row widens, on the same rows and
/// centred on the frame, to the narrowest width the text fits.  It stays
/// inside the canvas row, and when a vertical colour bar stands beside
/// the frame it ends at the frame's right edge, so it never runs into the
/// bar or the units above it.  Text that still does not fit is cut and
/// warned about as before, and a wider image now widens the row.
fn header_anchor(
    layout: &Layout,
    opts: &RenderOpts,
    frame_rect: Option<LocalRect>,
) -> ((u32, u32, u32), (u32, u32)) {
    let bounds = chrome_anchor_bounds(layout, opts.domain_frame, frame_rect);
    let rows = chrome_anchor_rows(layout, opts.domain_frame, frame_rect);
    let (frame_left, frame_right, frame_center) = bounds;
    let frame_width = frame_right.saturating_sub(frame_left);
    if frame_width >= layout.map_w || header_text_fits(layout, opts, frame_width) {
        return (bounds, rows);
    }
    let room_left = layout.map_x.min(frame_left);
    let canvas_right = layout.map_x.saturating_add(layout.map_w);
    let bar_beside_frame = opts.colorbar
        && matches!(
            opts.presentation.colorbar.orientation,
            ColorbarOrientation::VerticalRight
        )
        && colorbar_anchor_rect(
            layout,
            ColorbarOrientation::VerticalRight,
            opts.domain_frame,
            frame_rect,
        )
        .0 < layout.cbar_x;
    let room_right = if bar_beside_frame {
        frame_right.min(canvas_right)
    } else {
        canvas_right.max(frame_right)
    };
    let room = room_right.saturating_sub(room_left);
    if room <= frame_width {
        return (bounds, rows);
    }
    // The narrowest row the header fits, found by bisection: whether it
    // fits only grows with the row's width.
    let width = if header_text_fits(layout, opts, room) {
        let (mut cut, mut fits) = (frame_width, room);
        while fits - cut > 1 {
            let middle = cut + (fits - cut) / 2;
            if header_text_fits(layout, opts, middle) {
                fits = middle;
            } else {
                cut = middle;
            }
        }
        fits
    } else {
        room
    };
    let left = frame_center
        .saturating_sub(width / 2)
        .clamp(room_left, room_right - width);
    let right = left + width;
    ((left, right, left + width / 2), rows)
}

/// Whether every header text fits a row `row_width` pixels wide, measured
/// exactly as `draw_chrome_and_colorbar` fits them.
fn header_text_fits(layout: &Layout, opts: &RenderOpts, row_width: u32) -> bool {
    let row_width = row_width.max(1);
    let scale = layout.text_scale;
    fn present(text: Option<&str>) -> Option<&str> {
        text.map(str::trim).filter(|text| !text.is_empty())
    }
    let fits = |text: &str, width: u32, bold: bool| measure_text_width(text, scale, bold) <= width;
    if !present(opts.title.as_deref()).is_none_or(|title| fits(title, row_width, true)) {
        return false;
    }
    if !opts.presentation.plot_style.uses_operational_presentation() {
        return joined_subtitle_metadata(opts)
            .is_none_or(|metadata| fits(&metadata, row_width, false));
    }
    let available = row_width.saturating_sub(18u32.saturating_mul(scale));
    let right = present(opts.subtitle_right.as_deref());
    let left_fits = present(opts.subtitle_left.as_deref()).is_none_or(|left| {
        let width = subtitle_left_width(available, right, scale, 6u32.saturating_mul(scale));
        fits(left, width.max(1), false)
    });
    let center_fits =
        present(opts.subtitle_center.as_deref()).is_none_or(|center| fits(center, available, false));
    let right_fits = right.is_none_or(|right| {
        let width = subtitle_right_width(available, opts.subtitle_left.is_some(), right, scale);
        fits(right, width.max(1), false)
    });
    left_fits && center_fits && right_fits
}

/// Width for the right half of the subtitle row.
///
/// The right label is measured first (`subtitle_left_width`), so it keeps
/// what it needs; this bound only stops a pathological label from eating
/// the whole row.
fn subtitle_right_width(available: u32, has_left: bool, right: &str, scale: u32) -> u32 {
    if has_left {
        available
            .saturating_sub(subtitle_left_width(
                available,
                Some(right),
                scale,
                6u32.saturating_mul(scale),
            ))
            .max(available / 2)
    } else {
        available
    }
}

fn joined_subtitle_metadata(opts: &RenderOpts) -> Option<String> {
    let mut parts = Vec::with_capacity(3);
    if let Some(text) = opts
        .subtitle_left
        .as_deref()
        .map(str::trim)
        .filter(|t| !t.is_empty())
    {
        parts.push(text);
    }
    if let Some(text) = opts
        .subtitle_center
        .as_deref()
        .map(str::trim)
        .filter(|t| !t.is_empty())
    {
        parts.push(text);
    }
    if let Some(text) = opts
        .subtitle_right
        .as_deref()
        .map(str::trim)
        .filter(|t| !t.is_empty())
    {
        parts.push(text);
    }

    (!parts.is_empty()).then(|| parts.join(" | "))
}

fn colorbar_anchor_rect(
    layout: &Layout,
    orientation: ColorbarOrientation,
    frame: Option<DomainFrame>,
    frame_rect: Option<LocalRect>,
) -> (u32, u32, u32) {
    let mut cbar_x = layout.cbar_x;
    let mut cbar_y = layout.cbar_y;
    let mut cbar_w = layout.cbar_w;

    if matches!(orientation, ColorbarOrientation::VerticalRight) {
        if matches!(frame, Some(frame) if frame.legend_follows_frame) {
            if let Some(rect) = frame_rect {
                let domain_right = layout.map_x.saturating_add(rect.max_x);
                let frame_gap = (2u32.saturating_mul(layout.text_scale.max(1))).clamp(4, 8);
                let anchored_x = domain_right.saturating_add(frame_gap);
                if anchored_x < cbar_x {
                    cbar_x = anchored_x;
                }
            }
        }
        return (cbar_x, cbar_y, cbar_w);
    }

    if matches!(frame, Some(frame) if frame.legend_follows_frame) {
        if let Some(rect) = frame_rect {
            let domain_left = layout.map_x + rect.min_x;
            let domain_right = layout.map_x + rect.max_x;
            let domain_bottom = layout.map_y + rect.max_y;
            let baseline_margin = layout.cbar_x.saturating_sub(layout.map_x);
            let margin = baseline_margin.min(rect.width().saturating_sub(1) / 4);

            cbar_x = domain_left.saturating_add(margin);
            cbar_w = domain_right
                .saturating_sub(domain_left)
                .saturating_sub(margin.saturating_mul(2))
                .max(1);
            cbar_y = domain_bottom
                .saturating_add(layout.label_gap)
                .saturating_add(8)
                .min(layout.cbar_y);
        }
    }

    (cbar_x, cbar_y, cbar_w)
}

fn extent_bits(extent: &MapExtent) -> [u64; 4] {
    [
        extent.x_min.to_bits(),
        extent.x_max.to_bits(),
        extent.y_min.to_bits(),
        extent.y_max.to_bits(),
    ]
}

fn hash_projected_grid(grid: &ProjectedGrid) -> u64 {
    let mut hasher = DefaultHasher::new();
    grid.nx.hash(&mut hasher);
    grid.ny.hash(&mut hasher);
    grid.x.len().hash(&mut hasher);
    grid.y.len().hash(&mut hasher);
    for value in &grid.x {
        value.to_bits().hash(&mut hasher);
    }
    for value in &grid.y {
        value.to_bits().hash(&mut hasher);
    }
    hasher.finish()
}

/// Where `level` crosses the edge from `a` to `b`, if it does.
///
/// Each corner is classified once, at-or-above the level or below it, and
/// an edge is crossed only when its two corners fall on different sides.
/// A corner lying exactly ON the level used to count as a crossing on BOTH
/// of its edges: the cell then had three intersection points (or two copies
/// of the same corner), the walk drew only the first pair, and the real
/// segment from that corner to the opposite edge was never drawn.  A small
/// closed isobar in a nearly flat field showed a one-cell gap with nothing
/// in it (2.9.9 R10 look, U-E9: the R11 d01 MSLP/10 m winds map, north-east
/// corner).  With the classification a cell always has 0, 2 or 4 crossings;
/// values off the level take exactly the arithmetic they always did.
fn interp_point(a: (f64, f64, f64), b: (f64, f64, f64), level: f64) -> Option<(f64, f64)> {
    let (x0, y0, v0) = a;
    let (x1, y1, v1) = b;
    if !v0.is_finite() || !v1.is_finite() {
        return None;
    }
    if (v0 >= level) == (v1 >= level) {
        return None;
    }
    // Opposite sides, so v1 != v0 and t lies in [0, 1].
    let t = (level - v0) / (v1 - v0);
    Some((x0 + (x1 - x0) * t, y0 + (y1 - y0) * t))
}

fn levels_are_sorted_finite(levels: &[f64]) -> bool {
    levels.iter().all(|value| value.is_finite()) && levels.windows(2).all(|w| w[0] <= w[1])
}

fn lower_bound(levels: &[f64], target: f64) -> usize {
    let mut lo = 0usize;
    let mut hi = levels.len();
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if levels[mid] < target {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

fn upper_bound(levels: &[f64], target: f64) -> usize {
    let mut lo = 0usize;
    let mut hi = levels.len();
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if levels[mid] <= target {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

fn finite_minmax_4(v0: f64, v1: f64, v2: f64, v3: f64) -> Option<(f64, f64)> {
    let mut min_v = f64::INFINITY;
    let mut max_v = f64::NEG_INFINITY;
    let mut finite_count = 0usize;
    for value in [v0, v1, v2, v3] {
        if value.is_finite() {
            min_v = min_v.min(value);
            max_v = max_v.max(value);
            finite_count += 1;
        }
    }
    if finite_count >= 2 {
        Some((min_v, max_v))
    } else {
        None
    }
}

fn contour_cell_corners(
    layout: &Layout,
    overlay: &ContourOverlay,
    pixel_points: Option<&[Option<(f32, f32)>]>,
    base: usize,
    cell_step: usize,
) -> Option<((f64, f64), (f64, f64), (f64, f64), (f64, f64))> {
    let cell_step = cell_step.max(1);
    if let Some(points) = pixel_points {
        match (
            points[base],
            points[base + cell_step],
            points[base + cell_step * overlay.nx + cell_step],
            points[base + cell_step * overlay.nx],
        ) {
            (Some(a), Some(b), Some(c), Some(d)) => {
                let a = (a.0 as f64, a.1 as f64);
                let b = (b.0 as f64, b.1 as f64);
                let c = (c.0 as f64, c.1 as f64);
                let d = (d.0 as f64, d.1 as f64);
                projected_quad_is_continuous(a, b, c, d).then_some((a, b, c, d))
            }
            _ => None,
        }
    } else {
        let i = base % overlay.nx;
        let j = base / overlay.nx;
        Some((
            grid_to_pixel(i as f64, j as f64, overlay.nx, overlay.ny, layout),
            grid_to_pixel(
                (i + cell_step) as f64,
                j as f64,
                overlay.nx,
                overlay.ny,
                layout,
            ),
            grid_to_pixel(
                (i + cell_step) as f64,
                (j + cell_step) as f64,
                overlay.nx,
                overlay.ny,
                layout,
            ),
            grid_to_pixel(
                i as f64,
                (j + cell_step) as f64,
                overlay.nx,
                overlay.ny,
                layout,
            ),
        ))
    }
}

fn projected_quad_is_continuous(
    c0: (f64, f64),
    c1: (f64, f64),
    c2: (f64, f64),
    c3: (f64, f64),
) -> bool {
    let top = point_distance(c0, c1);
    let right = point_distance(c1, c2);
    let bottom = point_distance(c2, c3);
    let left = point_distance(c3, c0);
    if ![top, right, bottom, left]
        .iter()
        .all(|value| value.is_finite())
    {
        return false;
    }

    let horizontal = top.max(bottom);
    let vertical = right.max(left);
    let large = horizontal.max(vertical);
    let small = horizontal.min(vertical).max(1.0e-6);
    !(large > 128.0 && large / small > 10.0)
}

fn point_distance(a: (f64, f64), b: (f64, f64)) -> f64 {
    (b.0 - a.0).hypot(b.1 - a.1)
}

fn emit_interp_point(
    pts: &mut [(f64, f64); 4],
    count: &mut usize,
    a: (f64, f64, f64),
    b: (f64, f64, f64),
    level: f64,
) {
    if let Some(point) = interp_point(a, b, level) {
        pts[*count] = point;
        *count += 1;
    }
}

fn contour_cell_intersections(
    layout: &Layout,
    overlay: &ContourOverlay,
    pixel_points: Option<&[Option<(f32, f32)>]>,
    base: usize,
    cell_step: usize,
    level: f64,
) -> Option<([(f64, f64); 4], usize, f64)> {
    let cell_step = cell_step.max(1);
    let (c0, c1, c2, c3) = contour_cell_corners(layout, overlay, pixel_points, base, cell_step)?;
    let extent = point_distance(c0, c1)
        .max(point_distance(c1, c2))
        .max(point_distance(c2, c3))
        .max(point_distance(c3, c0));
    let p0 = (c0.0, c0.1, overlay.data[base]);
    let p1 = (c1.0, c1.1, overlay.data[base + cell_step]);
    let p2 = (
        c2.0,
        c2.1,
        overlay.data[base + cell_step * overlay.nx + cell_step],
    );
    let p3 = (c3.0, c3.1, overlay.data[base + cell_step * overlay.nx]);

    let mut pts = [(0.0, 0.0); 4];
    let mut count = 0usize;
    emit_interp_point(&mut pts, &mut count, p0, p1, level);
    emit_interp_point(&mut pts, &mut count, p1, p2, level);
    emit_interp_point(&mut pts, &mut count, p2, p3, level);
    emit_interp_point(&mut pts, &mut count, p3, p0, level);

    if count >= 2 { Some((pts, count, extent)) } else { None }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct LabelRect {
    min_x: i32,
    max_x: i32,
    min_y: i32,
    max_y: i32,
}

impl LabelRect {
    fn padded(self, padding: i32) -> Self {
        Self {
            min_x: self.min_x.saturating_sub(padding),
            max_x: self.max_x.saturating_add(padding),
            min_y: self.min_y.saturating_sub(padding),
            max_y: self.max_y.saturating_add(padding),
        }
    }

    fn intersects(self, other: Self) -> bool {
        self.min_x <= other.max_x
            && self.max_x >= other.min_x
            && self.min_y <= other.max_y
            && self.max_y >= other.min_y
    }
}

#[derive(Debug, Default)]
struct ContourLabelPlacer {
    occupied: Vec<LabelRect>,
    labels: Vec<DeferredContourLabel>,
}

impl ContourLabelPlacer {
    fn can_place(&mut self, rect: LabelRect) -> bool {
        let padded = rect.padded(4);
        if self
            .occupied
            .iter()
            .any(|existing| padded.intersects(*existing))
        {
            return false;
        }
        self.occupied.push(padded);
        true
    }

    fn push(&mut self, label: DeferredContourLabel) {
        self.labels.push(label);
    }

    fn draw(&self, img: &mut RgbaImage) {
        for label in &self.labels {
            draw_text_halo(
                img,
                &label.text,
                label.x,
                label.y,
                label.color,
                label.halo,
                label.halo_width_px,
                label.scale,
                label.size_factor,
                label.bold,
            );
        }
    }
}

#[derive(Debug, Clone)]
struct DeferredContourLabel {
    text: String,
    x: i32,
    y: i32,
    color: Rgba,
    halo: Rgba,
    halo_width_px: u32,
    scale: u32,
    size_factor: f32,
    bold: bool,
}

#[derive(Debug)]
struct ContourLevelLabelState {
    enabled: bool,
    max_labels: usize,
    min_spacing_sq: f64,
    centers: Vec<(f64, f64)>,
    /// Longest projected side, in pixels, of the grid cell whose segments
    /// are being offered.  Set by the contour walk before each cell.
    cell_extent_px: f64,
}

impl ContourLevelLabelState {
    fn new(enabled: bool, layout: &Layout) -> Self {
        let map_area = layout.map_w as u64 * layout.map_h as u64;
        let max_labels = ((map_area / 430_000) as usize).clamp(1, 3);
        let min_spacing = (layout.map_w.max(layout.map_h) as f64 / 6.5).clamp(120.0, 240.0);
        Self {
            enabled,
            max_labels,
            min_spacing_sq: min_spacing * min_spacing,
            centers: Vec::with_capacity(max_labels),
            cell_extent_px: f64::INFINITY,
        }
    }

    fn can_try_at(&self, center: (f64, f64)) -> bool {
        self.enabled
            && self.centers.len() < self.max_labels
            && self.centers.iter().all(|existing| {
                let dx = center.0 - existing.0;
                let dy = center.1 - existing.1;
                dx * dx + dy * dy >= self.min_spacing_sq
            })
    }

    fn record(&mut self, center: (f64, f64)) {
        self.centers.push(center);
    }
}

fn maybe_place_contour_label(
    layout: &Layout,
    overlay: &ContourOverlay,
    level_index: usize,
    level: f64,
    x0: f64,
    y0: f64,
    x1: f64,
    y1: f64,
    label_state: &mut ContourLevelLabelState,
    label_placer: &mut ContourLabelPlacer,
    clip_mask: Option<&RgbaImage>,
) {
    if !contour_level_gets_label(overlay, level_index) {
        return;
    }
    let segment_len = (x1 - x0).hypot(y1 - y0);
    if segment_len <= contour_label_min_segment_px(label_state.cell_extent_px) {
        return;
    }
    let center = ((x0 + x1) * 0.5, (y0 + y1) * 0.5);
    if !label_state.can_try_at(center) {
        return;
    }

    let label = text::format_tick(level);
    let label_scale = contour_label_scale(layout);
    let label_size_factor = contour_label_size_factor(layout);
    let label_w = text::text_width_with_factor(&label, label_scale, label_size_factor) as i32;
    let label_h = text::regular_line_height_with_factor(label_scale, label_size_factor) as i32;
    if label_w <= 0 || label_h <= 0 {
        return;
    }

    let tx = (layout.map_x as f64 + center.0) as i32 - label_w / 2;
    let ty = (layout.map_y as f64 + center.1) as i32 - label_h / 2;
    let rect = LabelRect {
        min_x: tx,
        max_x: tx.saturating_add(label_w),
        min_y: ty,
        max_y: ty.saturating_add(label_h),
    };
    let map_rect = LabelRect {
        min_x: layout.map_x as i32 + 2,
        max_x: layout.map_x as i32 + layout.map_w as i32 - 3,
        min_y: layout.map_y as i32 + 2,
        max_y: layout.map_y as i32 + layout.map_h as i32 - 3,
    };
    if rect.min_x < map_rect.min_x
        || rect.max_x > map_rect.max_x
        || rect.min_y < map_rect.min_y
        || rect.max_y > map_rect.max_y
        || !label_rect_inside_clip_mask(layout, rect, contour_label_halo_width(layout), clip_mask)
        || !label_placer.can_place(rect)
    {
        return;
    }

    let label_color = Rgba {
        a: 255,
        ..overlay.color
    };
    label_placer.push(DeferredContourLabel {
        text: label,
        x: tx,
        y: ty,
        color: label_color,
        halo: Rgba { a: 248, ..layout.halo },
        halo_width_px: contour_label_halo_width(layout),
        scale: label_scale,
        size_factor: label_size_factor,
        bold: true,
    });
    label_state.record(center);
}

/// The shortest segment a contour label may sit on.  Two pixels keeps a
/// label off a cell corner the line merely clips, which is what the floor
/// was for on grids whose cells span several pixels.  On a grid whose
/// cells are a pixel or less (a 3 km CONUS grid on a 1200 px panel) EVERY
/// segment is two pixels or shorter, so a fixed floor rejected every
/// segment and no contour label was ever drawn, though one was requested
/// (the WOOF-vs-HRRR 500 hPa height sheets).  The floor is therefore half
/// the cell's longest projected side, capped at the old two pixels:
/// identical on coarse grids, and a dense grid's segments qualify.
fn contour_label_min_segment_px(cell_extent_px: f64) -> f64 {
    if cell_extent_px.is_finite() && cell_extent_px > 0.0 {
        (0.5 * cell_extent_px).min(2.0)
    } else {
        2.0
    }
}

/// A label is drawn only where its whole box, halo included, survives the
/// clear outside the drawn domain.  Before, a label placed inside the map
/// rectangle but across the domain frame was cut in half by that clear
/// (a "564" reading "64" at the frame edge).
fn label_rect_inside_clip_mask(
    layout: &Layout,
    rect: LabelRect,
    halo_px: u32,
    clip_mask: Option<&RgbaImage>,
) -> bool {
    let Some(mask) = clip_mask else {
        return true;
    };
    let halo = halo_px as i32;
    let x0 = rect.min_x - halo - layout.map_x as i32;
    let x1 = rect.max_x + halo - layout.map_x as i32;
    let y0 = rect.min_y - halo - layout.map_y as i32;
    let y1 = rect.max_y + halo - layout.map_y as i32;
    let xs = [x0, (x0 + x1) / 2, x1];
    let ys = [y0, (y0 + y1) / 2, y1];
    xs.iter().all(|&x| {
        ys.iter()
            .all(|&y| mask_contains_local_pixel(mask, x as f64, y as f64))
    })
}

fn contour_level_gets_label(overlay: &ContourOverlay, level_index: usize) -> bool {
    overlay
        .major_every
        .filter(|every| *every > 0)
        .is_none_or(|every| level_index % every == 0)
}

fn contour_label_scale(layout: &Layout) -> u32 {
    if layout.map_w >= 1100 || layout.map_h >= 700 {
        2
    } else {
        1
    }
}

fn contour_label_size_factor(layout: &Layout) -> f32 {
    if contour_label_scale(layout) > 1 {
        0.78
    } else {
        1.0
    }
}

fn contour_label_halo_width(layout: &Layout) -> u32 {
    if contour_label_scale(layout) > 1 {
        2
    } else {
        1
    }
}

fn contour_level_width(overlay: &ContourOverlay, level_index: usize) -> u32 {
    let is_major = overlay
        .major_every
        .filter(|every| *every > 0)
        .is_some_and(|every| level_index % every == 0);
    if is_major {
        overlay
            .major_width
            .unwrap_or_else(|| overlay.width.saturating_add(1))
            .max(1)
    } else {
        overlay.width.max(1)
    }
}

fn draw_contour_stroke(
    img: &mut RgbaImage,
    x0: f64,
    y0: f64,
    x1: f64,
    y1: f64,
    color: Rgba,
    width: u32,
    pattern: crate::request::ContourLinePattern,
) {
    if !matches!(pattern, crate::request::ContourLinePattern::Dashed) {
        draw::draw_line_aa_width(img, x0, y0, x1, y1, color, width);
        return;
    }

    let len = (x1 - x0).hypot(y1 - y0);
    if !len.is_finite() || len <= 0.0 {
        return;
    }
    let dash = (7.0 * width.max(1) as f64).clamp(7.0, 18.0);
    let gap = (4.0 * width.max(1) as f64).clamp(4.0, 12.0);
    let period = dash + gap;
    let dx = (x1 - x0) / len;
    let dy = (y1 - y0) / len;
    let phase = (x0 * dx + y0 * dy).rem_euclid(period);
    let mut offset = if phase < dash {
        let end = (dash - phase).min(len);
        draw::draw_line_aa_width(img, x0, y0, x0 + dx * end, y0 + dy * end, color, width);
        end + gap
    } else {
        period - phase
    };
    while offset < len {
        let end = (offset + dash).min(len);
        draw::draw_line_aa_width(
            img,
            x0 + dx * offset,
            y0 + dy * offset,
            x0 + dx * end,
            y0 + dy * end,
            color,
            width,
        );
        offset += period;
    }
}

fn draw_contour_segments(
    img: &mut RgbaImage,
    layout: &Layout,
    overlay: &ContourOverlay,
    level_index: usize,
    level: f64,
    pts: &[(f64, f64); 4],
    count: usize,
    clip_mask: Option<&RgbaImage>,
    label_state: &mut ContourLevelLabelState,
    label_placer: &mut ContourLabelPlacer,
) {
    let segments: &[(usize, usize)] = if count == 4 {
        &[(0, 1), (2, 3)]
    } else {
        &[(0, 1)]
    };

    for &(a, b) in segments {
        let (x0, y0) = pts[a];
        let (x1, y1) = pts[b];
        if clip_mask.is_some_and(|mask| !segment_intersects_mask(mask, x0, y0, x1, y1)) {
            continue;
        }
        draw_contour_stroke(
            img,
            layout.map_x as f64 + x0,
            layout.map_y as f64 + y0,
            layout.map_x as f64 + x1,
            layout.map_y as f64 + y1,
            overlay.color,
            contour_level_width(overlay, level_index),
            overlay.pattern,
        );
        maybe_place_contour_label(
            layout,
            overlay,
            level_index,
            level,
            x0,
            y0,
            x1,
            y1,
            label_state,
            label_placer,
            clip_mask,
        );
    }
}

fn build_contour_buckets(
    overlay: &ContourOverlay,
    pixel_points: Option<&[Option<(f32, f32)>]>,
    cell_step: usize,
) -> Vec<Vec<u32>> {
    let mut buckets: Vec<Vec<u32>> = vec![Vec::new(); overlay.levels.len()];
    if overlay.levels.is_empty() {
        return buckets;
    }
    let cell_step = cell_step.max(1);

    for j in (0..overlay.ny.saturating_sub(cell_step)).step_by(cell_step) {
        let row_base = j * overlay.nx;
        for i in (0..overlay.nx.saturating_sub(cell_step)).step_by(cell_step) {
            let base = row_base + i;
            if let Some(points) = pixel_points {
                if !matches!(
                    (
                        points[base],
                        points[base + cell_step],
                        points[base + cell_step * overlay.nx + cell_step],
                        points[base + cell_step * overlay.nx]
                    ),
                    (Some(_), Some(_), Some(_), Some(_))
                ) {
                    continue;
                }
            }

            let Some((min_v, max_v)) = finite_minmax_4(
                overlay.data[base],
                overlay.data[base + cell_step],
                overlay.data[base + cell_step * overlay.nx + cell_step],
                overlay.data[base + cell_step * overlay.nx],
            ) else {
                continue;
            };

            let lo = lower_bound(&overlay.levels, min_v);
            let hi = upper_bound(&overlay.levels, max_v);
            if lo >= hi {
                continue;
            }

            let cell_id = base as u32;
            for bucket in &mut buckets[lo..hi] {
                bucket.push(cell_id);
            }
        }
    }

    buckets
}

fn draw_contours_bucketed(
    img: &mut RgbaImage,
    layout: &Layout,
    overlay: &ContourOverlay,
    pixel_points: Option<&[Option<(f32, f32)>]>,
    clip_mask: Option<&RgbaImage>,
    label_placer: &mut ContourLabelPlacer,
    cell_step: usize,
) -> ContourDrawTiming {
    let mut timing = ContourDrawTiming::default();
    let bucket_start = Instant::now();
    let buckets = build_contour_buckets(overlay, pixel_points, cell_step);
    timing.bucket_ms = bucket_start.elapsed().as_millis();

    for (level_index, &level) in overlay.levels.iter().enumerate() {
        let mut label_state = ContourLevelLabelState::new(overlay.labels, layout);
        for &base in &buckets[level_index] {
            let Some((pts, count, cell_extent)) = contour_cell_intersections(
                layout,
                overlay,
                pixel_points,
                base as usize,
                cell_step,
                level,
            ) else {
                continue;
            };
            label_state.cell_extent_px = cell_extent;
            timing.segment_count =
                timing
                    .segment_count
                    .saturating_add(if count == 4 { 2 } else { 1 });
            draw_contour_segments(
                img,
                layout,
                overlay,
                level_index,
                level,
                &pts,
                count,
                clip_mask,
                &mut label_state,
                label_placer,
            );
        }
    }
    timing
}

fn draw_contours_legacy(
    img: &mut RgbaImage,
    layout: &Layout,
    overlay: &ContourOverlay,
    pixel_points: Option<&[Option<(f32, f32)>]>,
    clip_mask: Option<&RgbaImage>,
    label_placer: &mut ContourLabelPlacer,
    cell_step: usize,
) -> ContourDrawTiming {
    let mut timing = ContourDrawTiming::default();
    let cell_step = cell_step.max(1);
    for (level_index, &level) in overlay.levels.iter().enumerate() {
        let mut label_state = ContourLevelLabelState::new(overlay.labels, layout);
        for j in (0..overlay.ny.saturating_sub(cell_step)).step_by(cell_step) {
            let row_base = j * overlay.nx;
            for i in (0..overlay.nx.saturating_sub(cell_step)).step_by(cell_step) {
                let base = row_base + i;
                let Some((pts, count, cell_extent)) = contour_cell_intersections(
                    layout,
                    overlay,
                    pixel_points,
                    base,
                    cell_step,
                    level,
                ) else {
                    continue;
                };
                label_state.cell_extent_px = cell_extent;
                timing.segment_count =
                    timing
                        .segment_count
                        .saturating_add(if count == 4 { 2 } else { 1 });

                draw_contour_segments(
                    img,
                    layout,
                    overlay,
                    level_index,
                    level,
                    &pts,
                    count,
                    clip_mask,
                    &mut label_state,
                    label_placer,
                );
            }
        }
    }
    timing
}

fn draw_contours(
    img: &mut RgbaImage,
    layout: &Layout,
    overlay: &ContourOverlay,
    pixel_points: Option<&[Option<(f32, f32)>]>,
    clip_mask: Option<&RgbaImage>,
    label_placer: &mut ContourLabelPlacer,
) -> ContourDrawTiming {
    if overlay.nx < 2 || overlay.ny < 2 {
        return ContourDrawTiming::default();
    }

    let cell_step = contour_render_cell_step(overlay, layout);
    if levels_are_sorted_finite(&overlay.levels) && overlay.data.len() <= u32::MAX as usize {
        draw_contours_bucketed(
            img,
            layout,
            overlay,
            pixel_points,
            clip_mask,
            label_placer,
            cell_step,
        )
    } else {
        draw_contours_legacy(
            img,
            layout,
            overlay,
            pixel_points,
            clip_mask,
            label_placer,
            cell_step,
        )
    }
}

fn contour_render_cell_step(overlay: &ContourOverlay, layout: &Layout) -> usize {
    let _ = (overlay, layout);
    std::env::var("RUSTWX_CONTOUR_CELL_STEP")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(1)
        .clamp(1, 8)
}

fn extrema_analysis_stride(nx: usize, ny: usize) -> usize {
    let _ = (nx, ny);
    std::env::var("RUSTWX_EXTREMA_ANALYSIS_STRIDE")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(1)
        .clamp(1, 16)
}

fn extrema_analysis_grid(
    data: &[f64],
    nx: usize,
    ny: usize,
    stride: usize,
) -> (Vec<f64>, usize, usize) {
    if stride <= 1 {
        return (data.to_vec(), nx, ny);
    }

    let analysis_nx = nx.div_ceil(stride);
    let analysis_ny = ny.div_ceil(stride);
    let mut analysis = Vec::with_capacity(analysis_nx.saturating_mul(analysis_ny));
    for aj in 0..analysis_ny {
        let j0 = aj.saturating_mul(stride);
        let j1 = (j0 + stride).min(ny);
        for ai in 0..analysis_nx {
            let i0 = ai.saturating_mul(stride);
            let i1 = (i0 + stride).min(nx);
            let mut sum = 0.0;
            let mut count = 0usize;
            for j in j0..j1 {
                let row = j * nx;
                for i in i0..i1 {
                    let value = data[row + i];
                    if value.is_finite() {
                        sum += value;
                        count += 1;
                    }
                }
            }
            analysis.push(if count > 0 {
                sum / count as f64
            } else {
                f64::NAN
            });
        }
    }
    (analysis, analysis_nx, analysis_ny)
}

fn finite_box_blur(src: &[f64], nx: usize, ny: usize, radius: usize) -> Vec<f64> {
    if radius == 0 || src.len() != nx.saturating_mul(ny) {
        return src.to_vec();
    }

    let mut horizontal = vec![f64::NAN; src.len()];
    for j in 0..ny {
        let row = j * nx;
        let mut sum = 0.0;
        let mut count = 0usize;
        for i in 0..nx {
            if i == 0 {
                let right = radius.min(nx.saturating_sub(1));
                for ii in 0..=right {
                    let value = src[row + ii];
                    if value.is_finite() {
                        sum += value;
                        count += 1;
                    }
                }
            } else {
                if i > radius {
                    let value = src[row + i - radius - 1];
                    if value.is_finite() {
                        sum -= value;
                        count = count.saturating_sub(1);
                    }
                }
                let add = i + radius;
                if add < nx {
                    let value = src[row + add];
                    if value.is_finite() {
                        sum += value;
                        count += 1;
                    }
                }
            }
            if count > 0 {
                horizontal[row + i] = sum / count as f64;
            }
        }
    }

    let mut blurred = vec![f64::NAN; src.len()];
    for i in 0..nx {
        let mut sum = 0.0;
        let mut count = 0usize;
        for j in 0..ny {
            if j == 0 {
                let bottom = radius.min(ny.saturating_sub(1));
                for jj in 0..=bottom {
                    let value = horizontal[jj * nx + i];
                    if value.is_finite() {
                        sum += value;
                        count += 1;
                    }
                }
            } else {
                if j > radius {
                    let value = horizontal[(j - radius - 1) * nx + i];
                    if value.is_finite() {
                        sum -= value;
                        count = count.saturating_sub(1);
                    }
                }
                let add = j + radius;
                if add < ny {
                    let value = horizontal[add * nx + i];
                    if value.is_finite() {
                        sum += value;
                        count += 1;
                    }
                }
            }
            if count > 0 {
                blurred[j * nx + i] = sum / count as f64;
            }
        }
    }

    blurred
}

fn sliding_extrema_line(input: &[f64], radius: usize, output: &mut [f64], want_max: bool) {
    debug_assert_eq!(input.len(), output.len());
    if input.is_empty() {
        return;
    }

    let mut deque: VecDeque<usize> = VecDeque::new();
    let mut next_right = 0usize;
    for center in 0..input.len() {
        let right = center.saturating_add(radius).min(input.len() - 1);
        while next_right <= right {
            let value = input[next_right];
            if value.is_finite() {
                while let Some(&back) = deque.back() {
                    let back_value = input[back];
                    let discard_back = if want_max {
                        back_value <= value
                    } else {
                        back_value >= value
                    };
                    if discard_back {
                        deque.pop_back();
                    } else {
                        break;
                    }
                }
                deque.push_back(next_right);
            }
            next_right += 1;
        }

        let left = center.saturating_sub(radius);
        while deque.front().is_some_and(|&front| front < left) {
            deque.pop_front();
        }

        output[center] = deque.front().map(|&front| input[front]).unwrap_or(f64::NAN);
    }
}

fn sliding_window_extrema(
    src: &[f64],
    nx: usize,
    ny: usize,
    radius: usize,
    want_max: bool,
) -> Vec<f64> {
    if radius == 0 || src.len() != nx.saturating_mul(ny) {
        return src.to_vec();
    }

    let mut horizontal = vec![f64::NAN; src.len()];
    for j in 0..ny {
        let row = j * nx;
        sliding_extrema_line(
            &src[row..row + nx],
            radius,
            &mut horizontal[row..row + nx],
            want_max,
        );
    }

    let mut output = vec![f64::NAN; src.len()];
    let mut column = vec![f64::NAN; ny];
    let mut filtered = vec![f64::NAN; ny];
    for i in 0..nx {
        for j in 0..ny {
            column[j] = horizontal[j * nx + i];
        }
        sliding_extrema_line(&column, radius, &mut filtered, want_max);
        for j in 0..ny {
            output[j * nx + i] = filtered[j];
        }
    }

    output
}

fn percentile_pair(values: &[f64], low_percent: usize, high_percent: usize) -> Option<(f64, f64)> {
    let mut finite: Vec<f64> = values.iter().filter(|v| v.is_finite()).copied().collect();
    if finite.is_empty() {
        return None;
    }

    let low_idx = finite.len().saturating_mul(low_percent) / 100;
    finite.select_nth_unstable_by(low_idx, |left, right| left.total_cmp(right));
    let low = finite[low_idx];

    let high_idx = finite.len().saturating_mul(high_percent) / 100;
    finite.select_nth_unstable_by(high_idx, |left, right| left.total_cmp(right));
    let high = finite[high_idx];
    Some((low, high))
}

fn draw_extrema_labels(
    img: &mut RgbaImage,
    layout: &Layout,
    overlay: &ContourOverlay,
    pixel_points: Option<&[Option<(f32, f32)>]>,
    clip_mask: Option<&RgbaImage>,
) {
    let source_ny = overlay.ny;
    let source_nx = overlay.nx;
    let source_data = &overlay.data;
    let analysis_stride = extrema_analysis_stride(source_nx, source_ny);
    let (analysis_data, nx, ny) =
        extrema_analysis_grid(source_data, source_nx, source_ny, analysis_stride);
    if nx < 3 || ny < 3 {
        return;
    }
    let data = &analysis_data;

    // Box-blur smoothing (3 passes ≈ Gaussian sigma~3)
    let mut smoothed = data.clone();
    let r = 5usize.min(ny / 4).min(nx / 4).max(1);
    for _ in 0..3 {
        smoothed = finite_box_blur(&smoothed, nx, ny, r);
    }

    // Find local extrema
    let source_window = (source_ny / 10).max(10).min(30);
    let window = source_window.div_ceil(analysis_stride).max(2).min(30);
    let edge = ((source_ny / 15).max(8)).div_ceil(analysis_stride).max(1);
    if nx <= edge.saturating_mul(2) || ny <= edge.saturating_mul(2) {
        return;
    }
    let local_max = sliding_window_extrema(&smoothed, nx, ny, window, true);
    let local_min = sliding_window_extrema(&smoothed, nx, ny, window, false);
    let Some((p20, p90)) = percentile_pair(data, 20, 90) else {
        return;
    };
    let mut highs: Vec<(usize, usize, f64, f64)> = Vec::new();
    let mut lows: Vec<(usize, usize, f64, f64)> = Vec::new();

    for j in edge..(ny - edge) {
        for i in edge..(nx - edge) {
            let idx = j * nx + i;
            let val = smoothed[j * nx + i];
            if !val.is_finite() {
                continue;
            }
            let raw_value = data[idx];
            if raw_value > p90 && val == local_max[idx] {
                highs.push((j, i, raw_value, val));
            }
            if raw_value < p20 && val == local_min[idx] {
                lows.push((j, i, raw_value, val));
            }
        }
    }

    // Convert grid (j,i) to pixel coordinates
    let to_px = |j: usize, i: usize| -> Option<(i32, i32)> {
        let source_j = j
            .saturating_mul(analysis_stride)
            .saturating_add(analysis_stride / 2)
            .min(source_ny.saturating_sub(1));
        let source_i = i
            .saturating_mul(analysis_stride)
            .saturating_add(analysis_stride / 2)
            .min(source_nx.saturating_sub(1));
        if let Some(points) = pixel_points {
            let idx = source_j * source_nx + source_i;
            points.get(idx)?.map(|(px, py)| {
                (
                    layout.map_x as i32 + px as i32,
                    layout.map_y as i32 + py as i32,
                )
            })
        } else {
            let (px, py) = grid_to_pixel(
                source_i as f64,
                source_j as f64,
                source_nx,
                source_ny,
                layout,
            );
            Some((px as i32, py as i32))
        }
    };
    let visible_at = |px: i32, py: i32| -> bool {
        clip_mask.is_none_or(|mask| {
            mask_contains_local_pixel(
                mask,
                (px - layout.map_x as i32) as f64,
                (py - layout.map_y as i32) as f64,
            )
        })
    };
    let highs = select_extrema_labels(
        highs
            .into_iter()
            .filter_map(|(j, i, value, score)| {
                let (px, py) = to_px(j, i)?;
                visible_at(px, py).then_some(ExtremaCandidate {
                    value,
                    score,
                    px,
                    py,
                })
            })
            .collect(),
        true,
        layout,
    );
    let lows = select_extrema_labels(
        lows.into_iter()
            .filter_map(|(j, i, value, score)| {
                let (px, py) = to_px(j, i)?;
                visible_at(px, py).then_some(ExtremaCandidate {
                    value,
                    score,
                    px,
                    py,
                })
            })
            .collect(),
        false,
        layout,
    );

    // Deep royal blue for H, brick red for L: saturated enough to read as
    // labels but muted so they don't feel neon over colored data.
    let h_color = Rgba::new(24, 84, 168);
    let l_color = Rgba::new(176, 46, 42);
    let halo = Rgba { a: 230, ..layout.halo };

    for point in &highs {
        draw_extrema_marker(
            img,
            layout,
            "H",
            point.value,
            point.px,
            point.py,
            h_color,
            halo,
        );
    }

    for point in &lows {
        draw_extrema_marker(
            img,
            layout,
            "L",
            point.value,
            point.px,
            point.py,
            l_color,
            halo,
        );
    }
}

#[derive(Debug, Clone, Copy)]
struct ExtremaCandidate {
    value: f64,
    score: f64,
    px: i32,
    py: i32,
}

fn select_extrema_labels(
    mut candidates: Vec<ExtremaCandidate>,
    high: bool,
    layout: &Layout,
) -> Vec<ExtremaCandidate> {
    if high {
        candidates.sort_by(|left, right| right.score.total_cmp(&left.score));
    } else {
        candidates.sort_by(|left, right| left.score.total_cmp(&right.score));
    }

    let min_spacing = extrema_label_min_spacing_px(layout);
    let min_spacing_sq = min_spacing * min_spacing;
    let max_labels = extrema_label_max_per_kind(layout);
    let mut selected = Vec::with_capacity(max_labels);
    for candidate in candidates {
        if selected.iter().all(|kept: &ExtremaCandidate| {
            let dx = (candidate.px - kept.px) as f64;
            let dy = (candidate.py - kept.py) as f64;
            dx * dx + dy * dy >= min_spacing_sq
        }) {
            selected.push(candidate);
            if selected.len() >= max_labels {
                break;
            }
        }
    }
    selected
}

fn extrema_label_min_spacing_px(layout: &Layout) -> f64 {
    (layout.map_w.min(layout.map_h) as f64 * 0.24).clamp(170.0, 280.0)
}

fn extrema_label_max_per_kind(layout: &Layout) -> usize {
    let area = layout.map_w as f64 * layout.map_h as f64;
    ((area / 450_000.0).round() as usize).clamp(2, 3)
}

fn draw_extrema_marker(
    img: &mut RgbaImage,
    layout: &Layout,
    kind: &str,
    value: f64,
    center_x: i32,
    center_y: i32,
    color: Rgba,
    halo: Rgba,
) {
    let letter_scale = layout.text_scale.max(1).saturating_add(1).min(4);
    let value_scale = layout.text_scale.max(1).min(3);
    let value_label = text::format_tick(value);

    let letter_w = text::text_width_bold(kind, letter_scale) as i32;
    let letter_h = text::bold_line_height(letter_scale) as i32;
    let value_w = text::text_width(&value_label, value_scale) as i32;
    let value_h = text::regular_line_height(value_scale) as i32;
    let gap = layout.text_scale.max(1) as i32;
    let total_h = letter_h + gap + value_h;
    let letter_x = center_x - letter_w / 2;
    let letter_y = center_y - total_h / 2;
    let value_x = center_x - value_w / 2;
    let value_y = letter_y + letter_h + gap;

    draw_text_halo(
        img,
        kind,
        letter_x,
        letter_y,
        color,
        halo,
        2,
        letter_scale,
        1.0,
        true,
    );
    draw_text_halo(
        img,
        &value_label,
        value_x,
        value_y,
        color,
        halo,
        2,
        value_scale,
        1.0,
        false,
    );
}

fn draw_streamlines(
    img: &mut RgbaImage,
    layout: &Layout,
    overlay: &StreamlineOverlay,
    pixel_points: Option<&[Option<(f32, f32)>]>,
    clip_mask: Option<&RgbaImage>,
) {
    if overlay.nx < 2 || overlay.ny < 2 {
        return;
    }

    let sx = overlay.stride_x.max(1);
    let sy = overlay.stride_y.max(1);
    let start_i = (sx / 2).min(overlay.nx.saturating_sub(1));
    let start_j = (sy / 2).min(overlay.ny.saturating_sub(1));

    for j in (start_j..overlay.ny).step_by(sy) {
        for i in (start_i..overlay.nx).step_by(sx) {
            let seed_i = i as f64;
            let seed_j = j as f64;
            draw_streamline_direction(
                img,
                layout,
                overlay,
                pixel_points,
                clip_mask,
                seed_i,
                seed_j,
                1.0,
            );
            draw_streamline_direction(
                img,
                layout,
                overlay,
                pixel_points,
                clip_mask,
                seed_i,
                seed_j,
                -1.0,
            );
        }
    }
}

fn draw_streamline_direction(
    img: &mut RgbaImage,
    layout: &Layout,
    overlay: &StreamlineOverlay,
    pixel_points: Option<&[Option<(f32, f32)>]>,
    clip_mask: Option<&RgbaImage>,
    seed_i: f64,
    seed_j: f64,
    direction: f64,
) {
    let mut i = seed_i;
    let mut j = seed_j;
    let Some(mut prev) =
        grid_coord_to_canvas_pixel(i, j, overlay.nx, overlay.ny, layout, pixel_points)
    else {
        return;
    };

    for _ in 0..overlay.max_steps {
        let Some((u, v)) =
            sample_vector_bilinear(&overlay.u, &overlay.v, overlay.nx, overlay.ny, i, j)
        else {
            break;
        };
        let speed = (u * u + v * v).sqrt();
        if !speed.is_finite() || speed < overlay.min_speed {
            break;
        }

        i += direction * (u / speed) * overlay.step_cells;
        j += direction * (v / speed) * overlay.step_cells;
        if i < 0.0
            || j < 0.0
            || i > overlay.nx.saturating_sub(1) as f64
            || j > overlay.ny.saturating_sub(1) as f64
        {
            break;
        }

        let Some(next) =
            grid_coord_to_canvas_pixel(i, j, overlay.nx, overlay.ny, layout, pixel_points)
        else {
            break;
        };

        if segment_inside_map_and_mask(layout, prev, next, clip_mask) {
            draw::draw_line_aa_width(
                img,
                prev.0,
                prev.1,
                next.0,
                next.1,
                overlay.color,
                overlay.width,
            );
        }
        prev = next;
    }
}

fn sample_vector_bilinear(
    u: &[f64],
    v: &[f64],
    nx: usize,
    ny: usize,
    i: f64,
    j: f64,
) -> Option<(f64, f64)> {
    if nx == 0
        || ny == 0
        || u.len() < nx.saturating_mul(ny)
        || v.len() < nx.saturating_mul(ny)
        || !i.is_finite()
        || !j.is_finite()
    {
        return None;
    }
    let i0 = i.floor().clamp(0.0, nx.saturating_sub(1) as f64) as usize;
    let j0 = j.floor().clamp(0.0, ny.saturating_sub(1) as f64) as usize;
    let i1 = (i0 + 1).min(nx - 1);
    let j1 = (j0 + 1).min(ny - 1);
    let tx = i - i0 as f64;
    let ty = j - j0 as f64;

    let sample = |values: &[f64], ii: usize, jj: usize| values[jj * nx + ii];
    let u00 = sample(u, i0, j0);
    let u10 = sample(u, i1, j0);
    let u01 = sample(u, i0, j1);
    let u11 = sample(u, i1, j1);
    let v00 = sample(v, i0, j0);
    let v10 = sample(v, i1, j0);
    let v01 = sample(v, i0, j1);
    let v11 = sample(v, i1, j1);
    if ![u00, u10, u01, u11, v00, v10, v01, v11]
        .iter()
        .all(|value| value.is_finite())
    {
        return None;
    }

    let lerp = |a: f64, b: f64, t: f64| a + (b - a) * t;
    let uu = lerp(lerp(u00, u10, tx), lerp(u01, u11, tx), ty);
    let vv = lerp(lerp(v00, v10, tx), lerp(v01, v11, tx), ty);
    Some((uu, vv))
}

fn grid_coord_to_canvas_pixel(
    i: f64,
    j: f64,
    nx: usize,
    ny: usize,
    layout: &Layout,
    pixel_points: Option<&[Option<(f32, f32)>]>,
) -> Option<(f64, f64)> {
    if let Some(points) = pixel_points {
        let (local_x, local_y) = projected_pixel_bilinear(points, nx, ny, i, j)?;
        return Some((layout.map_x as f64 + local_x, layout.map_y as f64 + local_y));
    }

    Some(grid_to_pixel(i, j, nx, ny, layout))
}

fn projected_pixel_bilinear(
    points: &[Option<(f32, f32)>],
    nx: usize,
    ny: usize,
    i: f64,
    j: f64,
) -> Option<(f64, f64)> {
    if nx == 0 || ny == 0 || points.len() < nx.saturating_mul(ny) {
        return None;
    }
    let i0 = i.floor().clamp(0.0, nx.saturating_sub(1) as f64) as usize;
    let j0 = j.floor().clamp(0.0, ny.saturating_sub(1) as f64) as usize;
    let i1 = (i0 + 1).min(nx - 1);
    let j1 = (j0 + 1).min(ny - 1);
    let tx = i - i0 as f64;
    let ty = j - j0 as f64;

    let sample = |ii: usize, jj: usize| points.get(jj * nx + ii).and_then(|point| *point);
    let p00 = sample(i0, j0)?;
    let p10 = sample(i1, j0)?;
    let p01 = sample(i0, j1)?;
    let p11 = sample(i1, j1)?;
    if ![p00.0, p00.1, p10.0, p10.1, p01.0, p01.1, p11.0, p11.1]
        .iter()
        .all(|value| value.is_finite())
    {
        return None;
    }
    let p00 = (p00.0 as f64, p00.1 as f64);
    let p10 = (p10.0 as f64, p10.1 as f64);
    let p01 = (p01.0 as f64, p01.1 as f64);
    let p11 = (p11.0 as f64, p11.1 as f64);
    if !projected_quad_is_continuous(p00, p10, p11, p01) {
        return None;
    }

    let lerp = |a: f64, b: f64, t: f64| a + (b - a) * t;
    let x = lerp(lerp(p00.0, p10.0, tx), lerp(p01.0, p11.0, tx), ty);
    let y = lerp(lerp(p00.1, p10.1, tx), lerp(p01.1, p11.1, tx), ty);
    Some((x, y))
}

fn segment_inside_map_and_mask(
    layout: &Layout,
    a: (f64, f64),
    b: (f64, f64),
    clip_mask: Option<&RgbaImage>,
) -> bool {
    let local_a = (a.0 - layout.map_x as f64, a.1 - layout.map_y as f64);
    let local_b = (b.0 - layout.map_x as f64, b.1 - layout.map_y as f64);
    if !point_inside_map(layout, local_a) || !point_inside_map(layout, local_b) {
        return false;
    }
    if let Some(mask) = clip_mask {
        return mask_contains_local_pixel(mask, local_a.0, local_a.1)
            && mask_contains_local_pixel(mask, local_b.0, local_b.1);
    }
    true
}

fn point_inside_map(layout: &Layout, point: (f64, f64)) -> bool {
    point.0 >= 0.0
        && point.1 >= 0.0
        && point.0 < layout.map_w as f64
        && point.1 < layout.map_h as f64
}

fn draw_barbs(
    img: &mut RgbaImage,
    layout: &Layout,
    overlay: &BarbOverlay,
    pixel_points: Option<&[Option<(f32, f32)>]>,
    clip_mask: Option<&RgbaImage>,
) {
    if overlay.nx == 0 || overlay.ny == 0 {
        return;
    }
    let sx = overlay.stride_x.max(1);
    let sy = overlay.stride_y.max(1);
    let spacing_px = overlay.spacing_px;
    let use_pixel_spacing = spacing_px.is_finite() && spacing_px > 0.0;
    let mut occupied_cells: HashMap<(i32, i32), (f64, f64)> = HashMap::new();

    for j in (0..overlay.ny).step_by(sy) {
        for i in (0..overlay.nx).step_by(sx) {
            let idx = j * overlay.nx + i;
            if idx >= overlay.u.len() || idx >= overlay.v.len() {
                continue;
            }
            let (x, y) = if let Some(points) = pixel_points {
                match points.get(idx).and_then(|p| *p) {
                    Some((px, py))
                        if (0.0..layout.map_w as f64).contains(&(px as f64))
                            && (0.0..layout.map_h as f64).contains(&(py as f64)) =>
                    {
                        (
                            layout.map_x as f64 + px as f64,
                            layout.map_y as f64 + py as f64,
                        )
                    }
                    None => continue,
                    _ => continue,
                }
            } else {
                grid_to_pixel(i as f64, j as f64, overlay.nx, overlay.ny, layout)
            };
            if let Some(mask) = clip_mask {
                if !mask_contains_local_pixel(
                    mask,
                    x - layout.map_x as f64,
                    y - layout.map_y as f64,
                ) {
                    continue;
                }
            }
            let local_x = x - layout.map_x as f64;
            let local_y = y - layout.map_y as f64;
            if !barb_glyph_fits_map_rect(
                local_x,
                local_y,
                layout.map_w,
                layout.map_h,
                overlay.length_px,
                overlay
                    .width
                    .saturating_add(overlay.halo_width.saturating_mul(2)),
            ) {
                continue;
            }
            if use_pixel_spacing
                && !accept_pixel_spaced_barb(&mut occupied_cells, local_x, local_y, spacing_px)
            {
                continue;
            }
            // A planned frame draws no glyph for a calm point: the calm
            // circle, repeated on a barb lattice over a light-wind area,
            // read as a grid of dots over the whole map.
            if !overlay.direction_arrows && layout.plan.is_some()
                && (overlay.u[idx] as f64).hypot(overlay.v[idx] as f64) < CALM_KT
            {
                continue;
            }
            let (u, v) = if overlay.direction_arrows {
                match projected_transport_direction(overlay.direction_projection.as_deref(), idx,
                    overlay.u[idx] as f64, overlay.v[idx] as f64, layout.map_w, layout.map_h) {
                    Some(vector) => vector, None => continue,
                }
            } else { (overlay.u[idx] as f64, overlay.v[idx] as f64) };
            draw_wind_barb_with_halo(img, overlay, x, y, u, v);
        }
    }
}

/// Below this speed (knots, the unit barbs are drawn in) a barb has no
/// pennant, half barb or shaft worth drawing.
const CALM_KT: f64 = 2.5;

fn accept_pixel_spaced_barb(
    occupied_cells: &mut HashMap<(i32, i32), (f64, f64)>,
    local_x: f64,
    local_y: f64,
    spacing_px: f64,
) -> bool {
    if spacing_px <= 0.0 || !local_x.is_finite() || !local_y.is_finite() {
        return false;
    }
    let cell_x = (local_x / spacing_px).floor() as i32;
    let cell_y = (local_y / spacing_px).floor() as i32;
    let min_distance_sq = spacing_px * spacing_px;
    for dy in -1..=1 {
        for dx in -1..=1 {
            if let Some((other_x, other_y)) = occupied_cells.get(&(cell_x + dx, cell_y + dy)) {
                let dist_sq = (local_x - other_x).powi(2) + (local_y - other_y).powi(2);
                if dist_sq < min_distance_sq {
                    return false;
                }
            }
        }
    }
    occupied_cells.insert((cell_x, cell_y), (local_x, local_y));
    true
}

// Project a short physical east/north displacement, then apply the map's
// actual pixel aspect. Pick the shorter one-sided derivative at a map seam.
fn projected_transport_direction(
    projection: Option<&crate::overlay::DirectionProjection>, idx: usize,
    u: f64, v: f64, width: u32, height: u32,
) -> Option<(f64, f64)> {
    let Some(projection) = projection else { return Some((u, v)); };
    let &(lat, lon) = projection.lat_lon.get(idx)?;
    let (lat, lon) = (lat as f64, lon as f64);
    if !lat.is_finite() || !lon.is_finite() || lat.abs() > 89.999 { return None; }
    let origin = projection.projection.project(lat, lon);
    let h = 1e-4;
    let tangent = |dlat: f64, dlon: f64| {
        let a = projection.projection.project(lat + dlat, lon + dlon);
        let b = projection.projection.project(lat - dlat, lon - dlon);
        let forward = (a.0-origin.0, a.1-origin.1);
        let backward = (origin.0-b.0, origin.1-b.1);
        if forward.0.hypot(forward.1) < backward.0.hypot(backward.1) { forward } else { backward }
    };
    let east = tangent(0.0, h / lat.to_radians().cos());
    let north = tangent(h, 0.0);
    let e = &projection.extent;
    let x = (u*east.0+v*north.0) * width as f64 / (e.x_max-e.x_min);
    let y = (u*east.1+v*north.1) * height as f64 / (e.y_max-e.y_min);
    (x.is_finite() && y.is_finite()).then_some((x, y))
}

fn draw_wind_barb_with_halo(
    img: &mut RgbaImage,
    overlay: &BarbOverlay,
    x: f64,
    y: f64,
    u: f64,
    v: f64,
) {
    if overlay.direction_arrows {
        let speed = u.hypot(v);
        if !speed.is_finite() || speed <= 0.0 { return; }
        let (dx, dy) = (u / speed, -v / speed);
        let length = overlay.length_px;
        let tip = (x + dx * length * 0.5, y + dy * length * 0.5);
        let tail = (x - dx * length * 0.5, y - dy * length * 0.5);
        let head = length * 0.28;
        let draw_arrow = |img: &mut RgbaImage, color, width| {
            draw::draw_line_aa_width(img, tail.0, tail.1, tip.0, tip.1, color, width);
            for sign in [-1.0, 1.0] {
                draw::draw_line_aa_width(img, tip.0, tip.1,
                    tip.0 - head * dx + sign * head * 0.5 * dy,
                    tip.1 - head * dy - sign * head * 0.5 * dx, color, width);
            }
        };
        if overlay.halo_color.a > 0 && overlay.halo_width > 0 {
            draw_arrow(img, overlay.halo_color, overlay.width + 2 * overlay.halo_width);
        }
        draw_arrow(img, overlay.color, overlay.width);
        return;
    }
    if overlay.halo_color.a > 0 && overlay.halo_width > 0 {
        draw::draw_wind_barb(
            img,
            x,
            y,
            u,
            v,
            overlay.halo_color,
            overlay.length_px,
            overlay
                .width
                .saturating_add(overlay.halo_width.saturating_mul(2)),
        );
    }
    draw::draw_wind_barb(
        img,
        x,
        y,
        u,
        v,
        overlay.color,
        overlay.length_px,
        overlay.width,
    );
}

fn barb_glyph_fits_map_rect(
    local_x: f64,
    local_y: f64,
    map_w: u32,
    map_h: u32,
    length_px: f64,
    width: u32,
) -> bool {
    if map_w == 0 || map_h == 0 {
        return false;
    }
    let margin = barb_glyph_margin_px(length_px, width);
    local_x >= margin
        && local_y >= margin
        && local_x <= map_w.saturating_sub(1) as f64 - margin
        && local_y <= map_h.saturating_sub(1) as f64 - margin
}

fn barb_glyph_margin_px(length_px: f64, width: u32) -> f64 {
    length_px.max(0.0) + (width.max(1) as f64 * 4.0) + 2.0
}

fn effective_domain_frame_rect(
    opts: &RenderOpts,
    map_img: &RgbaImage,
    raster_frame_mask: Option<&RgbaImage>,
    projected_domain_frame_rect: Option<LocalRect>,
    overlay_padding_px: u32,
) -> Option<LocalRect> {
    match opts.domain_frame {
        Some(frame) if matches!(frame.source, DomainFrameSource::RasterAlpha) => {
            raster_alpha_bounds(raster_frame_mask.unwrap_or(map_img))
                .map(LocalRect::from_bounds)
                .and_then(|rect| inset_rect(rect, frame.inset_px))
                .map(|rect| {
                    rect.expanded_within(overlay_padding_px, map_img.width(), map_img.height())
                })
        }
        _ => projected_domain_frame_rect,
    }
}

fn draw_variable_layers(
    img: &mut RgbaImage,
    data: &[f64],
    ny: usize,
    nx: usize,
    opts: &RenderOpts,
    layout: &Layout,
    projected_pixels: Option<&[Option<(f32, f32)>]>,
    domain_frame_rect: Option<LocalRect>,
    overlay_padding_px: u32,
    polygon_clip_rect: (i32, i32, i32, i32),
    canvas_background: Rgba,
) -> VariableLayerTiming {
    if let Some(ref extent) = opts.map_extent {
        draw_projected_polygons(
            img,
            layout,
            extent,
            &opts.projected_data_polygons,
            opts.presentation,
            Some(polygon_clip_rect),
        );
        if let Some(mesh) = opts.mesh_cells.as_ref() {
            // Projected once into pixel space, then filled and stroked.  The
            // fill colour comes from the SAME colormap the colorbar is drawn
            // from, so a reader matching a cell to a swatch is matching it to
            // the value that coloured it.
            let mut rings: Vec<Vec<(f64, f64)>> = Vec::with_capacity(mesh.cells.len());
            let mut colors: Vec<Rgba> = Vec::with_capacity(mesh.cells.len());
            for cell in &mesh.cells {
                rings.push(project_ring_unclipped(extent, &cell.ring, layout));
                colors.push(match cell.value {
                    Some(value) => opts.cmap.map(value),
                    None => mesh.style.empty,
                });
            }
            crate::mesh_cells::draw_mesh_cells_pixels(
                img,
                &rings,
                &colors,
                mesh.style,
                Some(polygon_clip_rect),
            );
        }
    }

    let rasterize_start = Instant::now();
    let map_img = match (
        opts.rgba_grid.as_deref(),
        projected_pixels,
        opts.inverse_projected_grid.as_ref(),
        opts.map_extent.as_ref(),
    ) {
        (Some(rgba_grid), Some(pixel_points), _, _) => rasterize::rasterize_projected_rgba_grid(
            rgba_grid,
            ny,
            nx,
            pixel_points,
            layout.map_w,
            layout.map_h,
        ),
        (Some(rgba_grid), _, _, _) => {
            rasterize::rasterize_rgba_grid(rgba_grid, ny, nx, layout.map_w, layout.map_h)
        }
        (None, _, Some(inverse), Some(extent)) => rasterize::rasterize_inverse_projected_grid(
            data,
            ny,
            nx,
            &inverse.lat_deg,
            &inverse.lon_deg,
            inverse.projector,
            inverse.clip_bounds,
            extent,
            &opts.cmap,
            opts.raster_sample_mode,
            layout.map_w,
            layout.map_h,
        ),
        (None, Some(pixel_points), _, _) => rasterize::rasterize_projected_grid(
            data,
            ny,
            nx,
            pixel_points,
            &opts.cmap,
            opts.raster_sample_mode,
            layout.map_w,
            layout.map_h,
        ),
        (None, None, _, _) => rasterize::rasterize_grid(
            data,
            ny,
            nx,
            &opts.cmap,
            opts.raster_sample_mode,
            layout.map_w,
            layout.map_h,
        ),
    };
    let rasterize_ms = rasterize_start.elapsed().as_millis();
    let projection_clip_mask =
        if opts.inverse_projected_grid.is_some() && opts.cmap.mask_below.is_none()
            // An explicit geographic viewport includes context outside a sparse
            // weather footprint (for example a moving storm's wind swath).
            && !matches!(opts.domain_frame, Some(frame) if matches!(frame.source, DomainFrameSource::MapViewport)) {
            build_alpha_clip_mask(&map_img)
        } else {
            None
        };
    let raster_frame_clip_mask = if matches!(
        opts.domain_frame,
        Some(frame) if frame.clear_outside && matches!(frame.source, DomainFrameSource::RasterAlpha)
    ) {
        match (opts.projected_grid.as_ref(), projected_pixels) {
            (Some(grid), Some(pixel_points)) => Some(rasterize::rasterize_projected_coverage_mask(
                grid.ny,
                grid.nx,
                pixel_points,
                layout.map_w,
                layout.map_h,
            )),
            _ => build_alpha_clip_mask(&map_img),
        }
    } else {
        None
    };
    let effective_domain_frame_rect = effective_domain_frame_rect(
        opts,
        &map_img,
        raster_frame_clip_mask.as_ref(),
        domain_frame_rect,
        overlay_padding_px,
    );

    let frame_clip_rect = match opts.domain_frame {
        Some(frame)
            if frame.clear_outside && !matches!(frame.source, DomainFrameSource::RasterAlpha) =>
        {
            effective_domain_frame_rect
        }
        _ => None,
    };
    let domain_clip_rect = frame_clip_rect.or_else(|| {
        opts.presentation
            .domain_boundary
            .and_then(|domain_boundary| {
                if !domain_boundary.visible {
                    return None;
                }
                if opts.cmap.mask_below.is_some() {
                    return None;
                }
                let inset = domain_boundary.width.saturating_add(3);
                raster_alpha_bounds(&map_img)
                    .map(LocalRect::from_bounds)
                    .and_then(|bounds| inset_rect(bounds, inset))
            })
    });
    let rect_domain_clip_mask =
        domain_clip_rect.map(|rect| build_rect_clip_mask(layout.map_w, layout.map_h, rect));
    let domain_clip_mask = raster_frame_clip_mask
        .as_ref()
        .or(rect_domain_clip_mask.as_ref());
    let combined_clip_mask = match (domain_clip_mask, projection_clip_mask.as_ref()) {
        (Some(domain), Some(projection)) => Some(intersect_alpha_clip_masks(domain, projection)),
        _ => None,
    };
    let draw_clip_mask = combined_clip_mask
        .as_ref()
        .or(domain_clip_mask)
        .or(projection_clip_mask.as_ref());

    let raster_blit_start = Instant::now();
    for py in 0..layout.map_h {
        for px in 0..layout.map_w {
            if let Some(mask) = draw_clip_mask {
                if mask.get_pixel(px, py).0[3] == 0 {
                    continue;
                }
            }
            let src = map_img.get_pixel(px, py);
            let a = src.0[3];
            if a == 0 {
                continue;
            }
            if a == 255 {
                img.put_pixel(layout.map_x + px, layout.map_y + py, *src);
            } else {
                draw::blend_pixel(
                    img,
                    (layout.map_x + px) as i32,
                    (layout.map_y + py) as i32,
                    Rgba {
                        r: src.0[0],
                        g: src.0[1],
                        b: src.0[2],
                        a,
                    },
                );
            }
        }
    }
    let raster_blit_ms = raster_blit_start.elapsed().as_millis();

    let linework_start = Instant::now();
    if let Some(ref extent) = opts.map_extent {
        draw_projected_lines(
            img,
            layout,
            extent,
            &opts.projected_lines,
            opts.presentation,
            draw_clip_mask,
        );
    }
    let linework_ms = linework_start.elapsed().as_millis();

    let point_start = Instant::now();
    if let Some(ref extent) = opts.map_extent {
        draw_projected_points(img, layout, extent, &opts.projected_points, draw_clip_mask);
    }
    let point_ms = point_start.elapsed().as_millis();

    let streamline_start = Instant::now();
    for streamline in &opts.streamlines {
        draw_streamlines(img, layout, streamline, projected_pixels, draw_clip_mask);
    }
    let streamline_ms = streamline_start.elapsed().as_millis();

    let contour_start = Instant::now();
    let mut contour_profile = ContourDrawTiming::default();
    let mut contour_label_placer = ContourLabelPlacer::default();
    for contour in &opts.contours {
        contour_profile.add(draw_contours(
            img,
            layout,
            contour,
            projected_pixels,
            draw_clip_mask,
            &mut contour_label_placer,
        ));
    }
    let contour_line_ms = contour_start.elapsed().as_millis();

    let barb_start = Instant::now();
    for barb in &opts.barbs {
        draw_barbs(img, layout, barb, projected_pixels, draw_clip_mask);
    }
    let barb_ms = streamline_ms.saturating_add(barb_start.elapsed().as_millis());

    // Contour labels and H/L extrema go on top of the barbs: a barb drawn
    // after them covered isobar values on MSLP / 10 m wind maps (2.9.6 R10
    // look, U10).  Lines stay beneath the barbs.
    let contour_label_start = Instant::now();
    let label_draw_start = Instant::now();
    contour_label_placer.draw(img);
    contour_profile.label_draw_ms = contour_profile
        .label_draw_ms
        .saturating_add(label_draw_start.elapsed().as_millis());
    for contour in &opts.contours {
        if contour.show_extrema && contour.nx >= 20 && contour.ny >= 20 {
            let extrema_start = Instant::now();
            draw_extrema_labels(img, layout, contour, projected_pixels, draw_clip_mask);
            contour_profile.extrema_ms = contour_profile
                .extrema_ms
                .saturating_add(extrema_start.elapsed().as_millis());
        }
    }
    let contour_ms = contour_line_ms.saturating_add(contour_label_start.elapsed().as_millis());

    let label_start = Instant::now();
    if let Some(ref extent) = opts.map_extent {
        draw_projected_place_labels(
            img,
            layout,
            extent,
            &opts.projected_place_labels,
            draw_clip_mask,
            domain_clip_rect,
        );
    }
    let label_ms = label_start.elapsed().as_millis();

    let outside_frame_clear_start = Instant::now();
    if let Some(mask) = combined_clip_mask.as_ref() {
        clear_map_outside_local_mask(img, layout, mask, canvas_background);
        draw_domain_clip_mask_outline(img, layout, mask, opts, effective_domain_frame_rect);
    } else if let (Some(frame), Some(rect)) = (opts.domain_frame, effective_domain_frame_rect) {
        if frame.clear_outside {
            clear_map_outside_local_rect(img, layout, rect, canvas_background);
        }
    } else if let Some(mask) = projection_clip_mask.as_ref() {
        clear_map_outside_local_mask(img, layout, mask, canvas_background);
        draw_domain_clip_mask_outline(img, layout, mask, opts, effective_domain_frame_rect);
    }
    let outside_frame_clear_ms = outside_frame_clear_start.elapsed().as_millis();

    VariableLayerTiming {
        rasterize_ms,
        raster_blit_ms,
        linework_ms: linework_ms
            .saturating_add(point_ms)
            .saturating_add(label_ms),
        contour_ms,
        contour_profile,
        barb_ms,
        outside_frame_clear_ms,
        domain_frame_rect: effective_domain_frame_rect,
        domain_clip_rect,
        projection_clip_mask_present: projection_clip_mask.is_some(),
    }
}

/// Outline the clip mask in the domain frame's style.
///
/// `frame_rect` is the rectangle the chrome strokes afterwards
/// (`draw_frame_outlines`) when a non-raster domain frame is drawn; mask
/// edge pixels inside that stroke's band are left to it.
///
/// WHAT BREAKAGE THIS PREVENTS (gate law): a native grid whose projected
/// edge runs a pixel or two inside the frame rectangle got both strokes side
/// by side, so the left frame edge read thicker from the top corner to about
/// a third of the way down and then thinned where the two met (R10 look,
/// 2.9.4). Where the data really stops short of the rectangle the mask
/// outline still draws.
fn draw_domain_clip_mask_outline(
    img: &mut RgbaImage,
    layout: &Layout,
    mask: &RgbaImage,
    opts: &RenderOpts,
    frame_rect: Option<LocalRect>,
) {
    if let Some(frame) = opts.domain_frame {
        let frame_style = opts
            .presentation
            .domain_frame_style(frame.outline_color.into(), frame.outline_width);
        if frame_style.visible {
            let stroked = frame_rect
                .filter(|_| !matches!(frame.source, DomainFrameSource::RasterAlpha))
                .map(|rect| (rect, frame_style.width.saturating_add(1)));
            draw_local_mask_outline(img, layout, mask, frame_style.color, frame_style.width, stroked);
        }
    } else if let Some(frame) = opts.presentation.chrome.frame_color {
        draw_local_mask_outline(img, layout, mask, frame, 1, None);
    }
}

fn draw_chrome_and_colorbar(
    img: &mut RgbaImage,
    layout: &Layout,
    opts: &RenderOpts,
    projected_pixels_ref: Option<&[Option<(f32, f32)>]>,
    domain_frame_rect: Option<LocalRect>,
    domain_clip_rect: Option<LocalRect>,
    projection_clip_mask_present: bool,
    _has_title: bool,
) -> (u128, u128) {
    if let Some(plan) = layout.plan.as_ref() {
        return draw_planned_chrome(
            img,
            layout,
            plan,
            opts,
            projected_pixels_ref,
            domain_frame_rect,
            domain_clip_rect,
            projection_clip_mask_present,
        );
    }
    let chrome_start = Instant::now();
    let ((chrome_left, chrome_right, chrome_center), (title_y, subtitle_y)) =
        header_anchor(layout, opts, domain_frame_rect);
    let title_color = opts.presentation.chrome.title_color;
    let subtitle_color = opts.presentation.chrome.subtitle_color;
    let row_width = chrome_right.saturating_sub(chrome_left).max(1);
    if opts.presentation.plot_style.uses_operational_presentation() {
        if let Some(title) = opts
            .title
            .as_deref()
            .map(str::trim)
            .filter(|text| !text.is_empty())
        {
            let fitted = ellipsize_text_to_width(title, row_width, layout.text_scale, true);
            text::draw_text_bold_with_factor(
                img,
                &fitted,
                chrome_left as i32,
                title_y as i32,
                title_color,
                layout.text_scale,
                layout.title_factor,
            );
        }
        let subtitle_available = row_width.saturating_sub(18u32.saturating_mul(layout.text_scale));
        if let Some(left) = opts
            .subtitle_left
            .as_deref()
            .map(str::trim)
            .filter(|text| !text.is_empty())
        {
            let left_width = subtitle_left_width(
                subtitle_available,
                opts.subtitle_right.as_deref(),
                layout.text_scale,
                6u32.saturating_mul(layout.text_scale),
            );
            let (fitted, truncated) =
                fit_text_to_width(left, left_width.max(1), layout.text_scale, false);
            if truncated {
                warn_subtitle_truncated("left", left, &fitted);
            }
            text::draw_text_with_factor(
                img,
                &fitted,
                chrome_left as i32,
                subtitle_y as i32,
                subtitle_color,
                layout.text_scale,
                layout.label_factor,
            );
        }
        if let Some(center) = opts
            .subtitle_center
            .as_deref()
            .map(str::trim)
            .filter(|text| !text.is_empty())
        {
            let (fitted, truncated) =
                fit_text_to_width(center, subtitle_available, layout.text_scale, false);
            if truncated {
                warn_subtitle_truncated("center", center, &fitted);
            }
            text::draw_text_with_factor(
                img,
                &fitted,
                centered_text_left(&fitted, chrome_center, layout.text_scale, false),
                subtitle_y as i32,
                subtitle_color,
                layout.text_scale,
                layout.label_factor,
            );
        }
        if let Some(right) = opts
            .subtitle_right
            .as_deref()
            .map(str::trim)
            .filter(|text| !text.is_empty())
        {
            let right_width = subtitle_right_width(
                subtitle_available,
                opts.subtitle_left.is_some(),
                right,
                layout.text_scale,
            );
            let (fitted, truncated) =
                fit_text_to_width(right, right_width.max(1), layout.text_scale, false);
            if truncated {
                warn_subtitle_truncated("right", right, &fitted);
            }
            text::draw_text_right_with_factor(
                img,
                &fitted,
                chrome_right as i32,
                subtitle_y as i32,
                subtitle_color,
                layout.text_scale,
                layout.label_factor,
            );
        }
    } else {
        let metadata = joined_subtitle_metadata(opts);
        let row_gap = 14u32.saturating_mul(layout.text_scale.max(1));
        let (fitted_title, fitted_metadata) = fit_chrome_title_metadata(
            opts.title.as_deref(),
            metadata.as_deref(),
            row_width,
            row_gap,
            layout.text_scale,
        );

        if let Some(ref title) = fitted_title {
            let title_x = if fitted_metadata.is_none()
                && matches!(opts.presentation.chrome.title_anchor, TitleAnchor::Center)
            {
                centered_text_left(title, chrome_center, layout.text_scale, true)
            } else {
                chrome_left as i32
            };
            text::draw_text_bold(
                img,
                title,
                title_x,
                title_y as i32,
                title_color,
                layout.text_scale,
            );
        }
        if let Some(ref metadata) = fitted_metadata {
            text::draw_text_right(
                img,
                metadata,
                chrome_right as i32,
                subtitle_y as i32,
                subtitle_color,
                layout.text_scale,
            );
        }
    }
    draw_frame_outlines(
        img,
        layout,
        opts,
        projected_pixels_ref,
        domain_frame_rect,
        domain_clip_rect,
        projection_clip_mask_present,
    );
    let chrome_ms = chrome_start.elapsed().as_millis();

    let colorbar_start = Instant::now();
    if opts.colorbar {
        let colorbar_orientation = opts.presentation.colorbar.orientation;
        let (cbar_x, cbar_y, cbar_w) = colorbar_anchor_rect(
            layout,
            colorbar_orientation,
            opts.domain_frame,
            domain_frame_rect,
        );
        let levels = colorbar_levels_for_ticks(&opts.cmap);
        let ticks = legend_ticks(&opts.cmap, opts.cbar_tick_step);
        match colorbar_orientation {
            ColorbarOrientation::HorizontalBottom => {
                colorbar::draw_colorbar(
                    img,
                    &opts.cmap,
                    cbar_x,
                    cbar_y,
                    cbar_w,
                    layout.cbar_h,
                    opts.colorbar_mode,
                    opts.presentation.colorbar,
                );
                if levels.len() >= 2 {
                    let lo = levels[0];
                    let hi = levels[levels.len() - 1];
                    let range = hi - lo;
                    if range > 0.0 {
                        let tick_positions: Vec<f64> = ticks
                            .iter()
                            .map(|t| legend_tick_position(&opts.cmap, *t, lo, range))
                            .collect();
                        colorbar::draw_colorbar_ticks(
                            img,
                            cbar_x,
                            cbar_y,
                            cbar_w,
                            &tick_positions,
                            opts.presentation.colorbar.tick_color,
                        );
                        let tick_y = cbar_y.saturating_sub(layout.label_gap) as i32;
                        let label_color = opts.presentation.colorbar.label_color;
                        for (_, lx, label) in filter_tick_labels_at(
                            &ticks,
                            &tick_positions,
                            cbar_x,
                            cbar_w,
                            cbar_x,
                            cbar_x.saturating_add(cbar_w),
                            img.width(),
                            layout.text_scale,
                        ) {
                            text::draw_text_with_factor(
                                img,
                                &label,
                                lx,
                                tick_y,
                                label_color,
                                layout.text_scale,
                                layout.label_factor,
                            );
                        }
                    }
                }
            }
            ColorbarOrientation::VerticalRight => {
                colorbar::draw_vertical_colorbar(
                    img,
                    &opts.cmap,
                    cbar_x,
                    cbar_y,
                    cbar_w,
                    layout.cbar_h,
                    opts.colorbar_mode,
                    opts.presentation.colorbar,
                );
                if levels.len() >= 2 {
                    let lo = levels[0];
                    let hi = levels[levels.len() - 1];
                    let range = hi - lo;
                    if range > 0.0 {
                        let tick_positions: Vec<f64> = ticks
                            .iter()
                            .map(|t| legend_tick_position(&opts.cmap, *t, lo, range))
                            .collect();
                        colorbar::draw_vertical_colorbar_ticks(
                            img,
                            cbar_x,
                            cbar_y,
                            cbar_w,
                            layout.cbar_h,
                            &tick_positions,
                            opts.presentation.colorbar.tick_color,
                        );
                        let label_color = opts.presentation.colorbar.label_color;
                        let label_x = cbar_x
                            .saturating_add(cbar_w)
                            .saturating_add(6u32.saturating_mul(layout.text_scale.max(1)))
                            as i32;
                        for (_, ly, label) in filter_vertical_tick_labels_at(
                            &ticks,
                            &tick_positions,
                            cbar_y,
                            layout.cbar_h,
                            cbar_y,
                            cbar_y.saturating_add(layout.cbar_h),
                            img.height(),
                            layout.text_scale,
                        ) {
                            text::draw_text_with_factor(
                                img,
                                &label,
                                label_x,
                                ly,
                                label_color,
                                layout.text_scale,
                                layout.label_factor,
                            );
                        }
                    }
                }
            }
        }
        if let Some(units) = opts.colorbar_units.as_deref().filter(|s| !s.trim().is_empty()) {
            let units = units.trim();
            let width = text::text_width_with_factor(
                units, layout.text_scale, layout.label_factor);
            let height = text::regular_line_height_with_factor(
                layout.text_scale, layout.label_factor);
            let gap = 4u32.saturating_mul(layout.text_scale.max(1));
            let mut vertical_label = None;
            let (x, y) = match colorbar_orientation {
                ColorbarOrientation::HorizontalBottom => (
                    cbar_x.saturating_sub(width.saturating_add(gap)),
                    cbar_y.saturating_add(layout.cbar_h).saturating_sub(height),
                ),
                ColorbarOrientation::VerticalRight
                    if cbar_y >= height + gap
                        && width <= img.width().saturating_sub(cbar_x) => (
                    cbar_x.saturating_add(cbar_w / 2)
                        .saturating_sub(width / 2).max(cbar_x),
                    cbar_y - height - gap,
                ),
                ColorbarOrientation::VerticalRight => {
                    // The header belongs to the map's title and timestamps.
                    // If the label has no complete row within the legend
                    // column, retain every glyph in the side margin.
                    let tick_width = text::format_tick_labels(&ticks).iter().map(|label| {
                        text::text_width_with_factor(
                            label, layout.text_scale, layout.label_factor)
                    }).max().unwrap_or(0);
                    let label = text::vertical_text(units, layout.text_scale, layout.label_factor);
                    let legend_left = cbar_x.saturating_add(cbar_w)
                        .saturating_add(6 * layout.text_scale.max(1))
                        .saturating_add(tick_width).saturating_add(gap);
                    let spare_width = img.width().saturating_sub(legend_left);
                    let position = (
                        legend_left.saturating_add(spare_width.saturating_sub(label.width) / 2),
                        cbar_y.saturating_add(layout.cbar_h / 2).saturating_sub(label.height / 2),
                    );
                    vertical_label = Some(label);
                    position
                }
            };
            if let Some(label) = vertical_label {
                label.draw(img, x, y, opts.presentation.colorbar.label_color);
            } else {
                text::draw_text_with_factor(
                    img, units,
                    x.min(img.width().saturating_sub(width)) as i32,
                    y.min(img.height().saturating_sub(height)) as i32,
                    opts.presentation.colorbar.label_color, layout.text_scale,
                    layout.label_factor,
                );
            }
        }
    }
    let colorbar_ms = colorbar_start.elapsed().as_millis();

    (chrome_ms, colorbar_ms)
}

/// The map's rectangular frame, the domain frame outline and the domain
/// boundary: shared by the fixed-canvas chrome and the planned chrome.
fn draw_frame_outlines(
    img: &mut RgbaImage,
    layout: &Layout,
    opts: &RenderOpts,
    projected_pixels_ref: Option<&[Option<(f32, f32)>]>,
    domain_frame_rect: Option<LocalRect>,
    domain_clip_rect: Option<LocalRect>,
    projection_clip_mask_present: bool,
) {
    if let Some(frame) = opts.presentation.chrome.frame_color {
        let draw_rectangular_frame = opts.domain_frame.is_none() && !projection_clip_mask_present;
        if draw_rectangular_frame {
            let map_right = layout.map_x + layout.map_w.saturating_sub(1);
            let map_bottom = layout.map_y + layout.map_h.saturating_sub(1);
            for px in layout.map_x..=map_right.min(img.width().saturating_sub(1)) {
                if layout.map_y < img.height() {
                    img.put_pixel(px, layout.map_y, frame.to_image_rgba());
                }
                if map_bottom < img.height() {
                    img.put_pixel(px, map_bottom, frame.to_image_rgba());
                }
            }
            for py in layout.map_y..=map_bottom.min(img.height().saturating_sub(1)) {
                if layout.map_x < img.width() {
                    img.put_pixel(layout.map_x, py, frame.to_image_rgba());
                }
                if map_right < img.width() {
                    img.put_pixel(map_right, py, frame.to_image_rgba());
                }
            }
        }
    }

    if let Some(frame) = opts.domain_frame {
        if let Some(rect) = domain_frame_rect {
            if !matches!(frame.source, DomainFrameSource::RasterAlpha) {
                let frame_style = opts
                    .presentation
                    .domain_frame_style(frame.outline_color.into(), frame.outline_width);
                if frame_style.visible {
                    draw_local_rect_outline(
                        img,
                        layout,
                        rect,
                        frame_style.color,
                        frame_style.width,
                    );
                }
            }
        }
    }

    if let Some(domain_boundary) = opts.presentation.domain_boundary {
        if domain_boundary.visible {
            if let Some(rect) = domain_clip_rect.filter(|_| domain_frame_rect.is_none()) {
                draw_local_rect_outline(
                    img,
                    layout,
                    rect,
                    domain_boundary.color,
                    domain_boundary.width,
                );
            } else {
                let drew_grid_boundary = match (&opts.projected_grid, projected_pixels_ref) {
                    (Some(grid), Some(pixel_points)) => draw_projected_grid_boundary(
                        img,
                        layout,
                        grid,
                        pixel_points,
                        domain_boundary.color,
                        domain_boundary.width,
                    ),
                    _ => false,
                };
                if !drew_grid_boundary {
                    let map_right = layout.map_x + layout.map_w.saturating_sub(1);
                    let map_bottom = layout.map_y + layout.map_h.saturating_sub(1);
                    draw::draw_polyline_aa(
                        img,
                        &[
                            (layout.map_x as f64, layout.map_y as f64),
                            (map_right as f64, layout.map_y as f64),
                            (map_right as f64, map_bottom as f64),
                            (layout.map_x as f64, map_bottom as f64),
                            (layout.map_x as f64, layout.map_y as f64),
                        ],
                        domain_boundary.color,
                        domain_boundary.width,
                    );
                }
            }
        }
    }
}

/// The chrome of a planned canvas: the header across the content width,
/// the frame outlines, and the colour bar at exactly the map's length.
#[allow(clippy::too_many_arguments)]
fn draw_planned_chrome(
    img: &mut RgbaImage,
    layout: &Layout,
    plan: &CanvasPlan,
    opts: &RenderOpts,
    projected_pixels_ref: Option<&[Option<(f32, f32)>]>,
    domain_frame_rect: Option<LocalRect>,
    domain_clip_rect: Option<LocalRect>,
    projection_clip_mask_present: bool,
) -> (u128, u128) {
    let chrome_start = Instant::now();
    let header = PlanHeaderText::compose(
        opts.title.as_deref(),
        opts.colorbar_units.as_deref().filter(|_| opts.colorbar),
        opts.subtitle_left.as_deref(),
        opts.subtitle_center.as_deref(),
        opts.subtitle_right.as_deref(),
    );
    if draw_plan_header(
        img,
        plan,
        &header,
        opts.presentation.chrome.title_color,
        opts.presentation.chrome.subtitle_color,
    ) {
        let full = [header.title.as_str(), header.when.as_deref().unwrap_or(""), header.meta.as_str()]
            .join(" / ");
        warn_subtitle_truncated("header", &full, "(cut to fit)");
    }
    draw_frame_outlines(
        img,
        layout,
        opts,
        projected_pixels_ref,
        domain_frame_rect,
        domain_clip_rect,
        projection_clip_mask_present,
    );
    let chrome_ms = chrome_start.elapsed().as_millis();

    let colorbar_start = Instant::now();
    if opts.colorbar {
        if let Some(bar) = plan.bar {
            let levels = colorbar_levels_for_ticks(&opts.cmap);
            if levels.len() >= 2 {
                let ticks = legend_ticks(&opts.cmap, opts.cbar_tick_step);
                draw_plan_colorbar(
                    img,
                    plan,
                    bar,
                    plan.bar_side,
                    &opts.cmap,
                    opts.colorbar_mode,
                    opts.presentation.colorbar,
                    &ticks,
                    levels[0],
                    levels[levels.len() - 1],
                );
            }
        }
    }
    (chrome_ms, colorbar_start.elapsed().as_millis())
}

fn render_to_image_profile_inner(
    data: &[f64],
    ny: usize,
    nx: usize,
    opts: &RenderOpts,
) -> (RgbaImage, RenderImageTiming) {
    let total_start = Instant::now();
    let layout_start = Instant::now();
    let has_title = opts.title.is_some()
        || opts.subtitle_left.is_some()
        || opts.subtitle_center.is_some()
        || opts.subtitle_right.is_some();
    let mut layout = compute_effective_layout(
        opts.width,
        opts.height,
        opts.colorbar,
        has_title,
        opts.presentation,
        opts.chrome_scale,
        opts.domain_frame.is_some(),
    );
    fit_map_viewport_layout_to_extent(
        &mut layout,
        opts.width,
        opts.colorbar,
        opts.presentation.colorbar.orientation,
        opts.domain_frame,
        opts.map_extent.as_ref(),
    );
    let layout_ms = layout_start.elapsed().as_millis();

    let projected_pixel_start = Instant::now();
    let projected_pixels = match (&opts.projected_grid, &opts.map_extent) {
        (Some(grid), Some(extent)) if grid.nx == nx && grid.ny == ny => {
            Some(projected_grid_to_pixels_cached(grid, extent, &layout))
        }
        _ => None,
    };
    let projected_pixel_ms = projected_pixel_start.elapsed().as_millis();
    let overlay_padding_px = overlay_frame_padding_px(opts, &layout);
    let planned_full_frame = layout
        .plan
        .is_some_and(|plan| !plan.aspect_clamped && opts.inverse_projected_grid.is_none());
    let domain_frame_rect = match (
        opts.domain_frame,
        opts.projected_grid.as_ref(),
        projected_pixels.as_deref(),
    ) {
        // A planned map is the grid's own box, edge to edge: the frame is
        // the whole map rectangle, with no coverage scan that could leave
        // the last antialiased pixel row outside it.
        (Some(frame), _, _) if planned_full_frame => compute_domain_frame_rect(
            DomainFrame { inset_px: 0, ..frame },
            layout.map_w,
            layout.map_h,
        ),
        (Some(frame), Some(grid), Some(pixel_points))
            if matches!(frame.source, DomainFrameSource::ProjectedGrid) =>
        {
            compute_projected_domain_frame_rect(
                frame,
                grid,
                pixel_points,
                layout.map_w,
                layout.map_h,
                overlay_padding_px,
                // An inverse-raster render is, by construction, a regular
                // lat/lon field resampled into a different presentation
                // projection: its footprint curves, so the inscribed
                // rectangle is not its frame.
                opts.inverse_projected_grid.is_none(),
            )
        }
        (Some(frame), _, _) if matches!(frame.source, DomainFrameSource::RasterAlpha) => None,
        (Some(frame), _, _) => compute_domain_frame_rect(frame, layout.map_w, layout.map_h),
        _ => None,
    };
    let polygon_clip_rect = domain_frame_rect
        .filter(|_| matches!(opts.domain_frame, Some(frame) if frame.clear_outside))
        .map(|rect| {
            (
                (layout.map_x + rect.min_x) as i32,
                (layout.map_y + rect.min_y) as i32,
                (layout.map_x + rect.max_x) as i32,
                (layout.map_y + rect.max_y) as i32,
            )
        })
        .unwrap_or_else(|| {
            let map_right = layout.map_x.saturating_add(layout.map_w).saturating_sub(1) as i32;
            let map_bottom = layout.map_y.saturating_add(layout.map_h).saturating_sub(1) as i32;
            (
                layout.map_x as i32,
                layout.map_y as i32,
                map_right,
                map_bottom,
            )
        });

    let canvas_background = if opts.background == Rgba::WHITE {
        opts.presentation.canvas_background
    } else {
        opts.background
    };
    let map_background = if opts.background == Rgba::WHITE {
        opts.presentation.map_background
    } else {
        opts.background
    };
    let draw_static_polygons = draw_static_polygons_for_render(data, opts);
    let (mut img, background_ms, polygon_fill_ms) = cached_static_base_image(
        opts,
        &layout,
        opts.map_extent.as_ref(),
        domain_frame_rect,
        canvas_background,
        map_background,
        polygon_clip_rect,
        draw_static_polygons,
    );
    let variable_timing = draw_variable_layers(
        &mut img,
        data,
        ny,
        nx,
        opts,
        &layout,
        projected_pixels.as_deref(),
        domain_frame_rect,
        overlay_padding_px,
        polygon_clip_rect,
        canvas_background,
    );
    let effective_domain_frame_rect = variable_timing.domain_frame_rect.or(domain_frame_rect);
    let effective_domain_clip_rect = variable_timing
        .domain_clip_rect
        .or(effective_domain_frame_rect);
    let (chrome_ms, colorbar_ms) = draw_chrome_and_colorbar(
        &mut img,
        &layout,
        opts,
        projected_pixels.as_deref(),
        effective_domain_frame_rect,
        variable_timing.domain_clip_rect,
        variable_timing.projection_clip_mask_present,
        has_title,
    );

    let timing = RenderImageTiming {
        layout_ms,
        background_ms,
        polygon_fill_ms,
        projected_pixel_ms,
        rasterize_ms: variable_timing.rasterize_ms,
        raster_blit_ms: variable_timing.raster_blit_ms,
        linework_ms: variable_timing.linework_ms,
        contour_ms: variable_timing.contour_ms,
        contour_bucket_ms: variable_timing.contour_profile.bucket_ms,
        contour_extrema_ms: variable_timing.contour_profile.extrema_ms,
        contour_label_draw_ms: variable_timing.contour_profile.label_draw_ms,
        contour_segment_count: variable_timing.contour_profile.segment_count,
        barb_ms: variable_timing.barb_ms,
        outside_frame_clear_ms: variable_timing.outside_frame_clear_ms,
        chrome_ms,
        colorbar_ms,
        downsample_ms: 0,
        postprocess_ms: 0,
        total_ms: total_start.elapsed().as_millis(),
        map_x: layout.map_x,
        map_y: layout.map_y,
        map_w: layout.map_w,
        map_h: layout.map_h,
        // Set true here because nothing in this function moves the map
        // after the layout; `save_png_profile_with_options_and_style`
        // folds each post-render pass's reported offset into the
        // rectangle, clips the rectangle to the pixels that survive in
        // the written image, and clears this only when none survive.
        plot_rect_describes_the_png: true,
        map_clip_left: 0.0,
        map_clip_right: 0.0,
        map_clip_top: 0.0,
        map_clip_bottom: 0.0,
        image_w: img.width(),
        image_h: img.height(),
        postprocess_offset_x: 0,
        postprocess_offset_y: 0,
        has_projected_grid: opts.projected_grid.is_some(),
        has_inverse_raster: opts.inverse_projected_grid.is_some(),
        projection_clip_mask_present: variable_timing.projection_clip_mask_present,
        domain_clip_rect: effective_domain_clip_rect
            .map(|rect| [rect.min_x, rect.max_x, rect.min_y, rect.max_y]),
    };

    (img, timing)
}

/// The note a filled map carries when it has nothing to draw and the product
/// gave no reason of its own.
pub const EMPTY_MAP_NOTE: &str = "No values to plot";

/// The note to draw on a filled map that has nothing to draw: no value of
/// the field takes a visible colour on its scale, no mesh cell does either,
/// and no contour, barb, streamline or RGBA layer draws anything over it.
/// `None` when anything is drawn, and for a map without a colour bar
/// (overlay-only pictures are blank fills by design).
///
/// WHAT BREAKAGE THIS PREVENTS (gate law): a map with nothing to draw for a
/// valid reason (no 0 C level in a cold column, the Haines layer below
/// ground, an accumulation at F000) looked exactly like a failed render (R10
/// look, 2.9.4).
///
/// The mesh layer counts as data. A `mesh:`/`meshdiff:` product carries its
/// values on the cells and hands the raster a 2x2 NaN placeholder, so a
/// check of `data` alone stamped "No values to plot" over every mesh panel
/// whose cells were plainly coloured (autodiff U9 sheets, 2026-11-03).
pub fn empty_map_note(data: &[f64], opts: &RenderOpts) -> Option<String> {
    if !opts.colorbar
        || opts.rgba_grid.is_some()
        || !opts.contours.is_empty()
        || !opts.barbs.is_empty()
        || !opts.streamlines.is_empty()
        || data.is_empty()
    {
        return None;
    }
    if data.iter().any(|value| opts.cmap.map(*value).a > 0) {
        return None;
    }
    if let Some(mesh) = opts.mesh_cells.as_ref() {
        if mesh
            .cells
            .iter()
            .filter_map(|cell| cell.value)
            .any(|value| opts.cmap.map(value).a > 0)
        {
            return None;
        }
    }
    Some(
        opts.empty_note
            .clone()
            .filter(|note| !note.trim().is_empty())
            .unwrap_or_else(|| EMPTY_MAP_NOTE.to_string()),
    )
}

/// Draw `note` centred on the map rectangle of a finished image, on a panel
/// in the canvas colour with a hairline in the frame colour, so it reads as
/// a statement about the map rather than as data.
fn draw_empty_map_note(img: &mut RgbaImage, timing: &RenderImageTiming, note: &str, opts: &RenderOpts) {
    let (map_x, map_y, map_w, map_h) = (timing.map_x, timing.map_y, timing.map_w, timing.map_h);
    if map_w < 40 || map_h < 24 {
        return;
    }
    // The face the rest of the chrome uses: Source Sans 3 on a fixed
    // canvas, the planned chrome face otherwise.
    let fixed = matches!(
        crate::layout_plan::layout_mode(),
        crate::layout_plan::LayoutMode::Fixed
    );
    let width_of = |size: f32| {
        if fixed {
            text::text_width_bold_with_factor(note, 1, size / 15.0)
        } else {
            text::text_width_px(note, size, true)
        }
    };
    let mut size = (map_h as f32 / 28.0).clamp(10.0, 22.0);
    let mut text_w = width_of(size);
    while text_w + 2 * (size * 0.8) as u32 > map_w.saturating_sub(8) && size > 10.0 {
        size -= 1.0;
        text_w = width_of(size);
    }
    let pad = (size * 0.8).round() as u32;
    let line_h = if fixed {
        text::bold_line_height_with_factor(1, size / 15.0)
    } else {
        text::line_height_px(size, true)
    };
    let box_w = (text_w + 2 * pad).min(map_w.saturating_sub(4));
    let box_h = line_h + pad;
    let box_x = map_x + (map_w.saturating_sub(box_w)) / 2;
    let box_y = map_y + (map_h.saturating_sub(box_h)) / 2;
    let fill = opts.presentation.canvas_background;
    let edge = opts.presentation.colorbar.frame_color;
    for y in box_y..(box_y + box_h).min(img.height()) {
        for x in box_x..(box_x + box_w).min(img.width()) {
            let on_edge = y == box_y || x == box_x || y + 1 == box_y + box_h || x + 1 == box_x + box_w;
            let color = if on_edge { edge } else { fill };
            img.put_pixel(x, y, image::Rgba([color.r, color.g, color.b, 255]));
        }
    }
    let text_x = (box_x + (box_w.saturating_sub(text_w)) / 2) as i32;
    let text_y = (box_y as f32 + (box_h as f32 - line_h as f32) / 2.0).round() as i32;
    let color = opts.presentation.chrome.title_color;
    if fixed {
        text::draw_text_bold_with_factor(img, note, text_x, text_y, color, 1, size / 15.0);
    } else {
        text::draw_text_px(img, note, text_x, text_y, color, size, true);
    }
}

pub fn render_to_image_profile(
    data: &[f64],
    ny: usize,
    nx: usize,
    opts: &RenderOpts,
) -> (RgbaImage, RenderImageTiming) {
    let (mut image, timing) = render_to_image_profile_unnoted(data, ny, nx, opts);
    if let Some(note) = empty_map_note(data, opts) {
        draw_empty_map_note(&mut image, &timing, &note, opts);
    }
    (image, timing)
}

fn render_to_image_profile_unnoted(
    data: &[f64],
    ny: usize,
    nx: usize,
    opts: &RenderOpts,
) -> (RgbaImage, RenderImageTiming) {
    let factor = opts.supersample_factor.max(1);
    if factor == 1 {
        return render_to_image_profile_inner(data, ny, nx, opts);
    }

    let total_start = Instant::now();
    let scaled_opts = scale_render_opts_for_supersample(opts, factor);
    let (hires, mut timing) = render_to_image_profile_inner(data, ny, nx, &scaled_opts);
    let downsample_start = Instant::now();

    // Upstream gates a cuda downsample+sharpen fast-path here; this port
    // always takes the CPU Lanczos path.
    let image_opt: Option<RgbaImage> = None;

    let image = match image_opt {
        Some(img) => img,
        None => {
            let image = resize(&hires, opts.width, opts.height, FilterType::Lanczos3);
            if opts.supersample_sharpen {
                sharpen_downsampled_image(&image)
            } else {
                image
            }
        }
    };

    timing.downsample_ms = downsample_start.elapsed().as_millis();
    timing.postprocess_ms = timing.downsample_ms;
    timing.total_ms = total_start.elapsed().as_millis();
    timing.map_x /= factor;
    timing.map_y /= factor;
    timing.map_w /= factor;
    timing.map_h /= factor;
    timing.image_w = image.width();
    timing.image_h = image.height();
    if let Some(rect) = timing.domain_clip_rect.as_mut() {
        for value in rect {
            *value /= factor;
        }
    }
    (image, timing)
}

fn sharpen_downsampled_image(image: &RgbaImage) -> RgbaImage {
    filter3x3(
        image,
        &[0.0, -0.22, 0.0, -0.22, 1.88, -0.22, 0.0, -0.22, 0.0],
    )
}

pub fn render_to_image(data: &[f64], ny: usize, nx: usize, opts: &RenderOpts) -> RgbaImage {
    render_to_image_profile(data, ny, nx, opts).0
}

fn row_is_canvas_background(img: &RgbaImage, y: u32, background: Rgba) -> bool {
    let bg = background.to_image_rgba().0;
    (0..img.width()).all(|x| {
        let px = img.get_pixel(x, y).0;
        let diff = px[0].abs_diff(bg[0]) as u16
            + px[1].abs_diff(bg[1]) as u16
            + px[2].abs_diff(bg[2]) as u16
            + px[3].abs_diff(bg[3]) as u16;
        diff <= 6
    })
}

/// Also returns the rows removed off the TOP -- everything in the image
/// moved up by that many pixels.  gpuwm addition (VENDOR.md): until the
/// pass reported it, the plot rectangle could not follow the map and
/// every trimmed panel shipped without a georeference.
pub(crate) fn trim_vertical_canvas_whitespace(
    img: &RgbaImage,
    background: Rgba,
) -> (RgbaImage, u32) {
    if img.height() <= 2 {
        return (img.clone(), 0);
    }

    let first_non_bg = (0..img.height()).find(|&y| !row_is_canvas_background(img, y, background));
    let last_non_bg = (0..img.height()).rfind(|&y| !row_is_canvas_background(img, y, background));

    let (Some(first), Some(last)) = (first_non_bg, last_non_bg) else {
        return (img.clone(), 0);
    };

    let top_pad = 2u32;
    let bottom_pad = 2u32;
    let crop_top = first.saturating_sub(top_pad);
    let crop_bottom = (last.saturating_add(bottom_pad)).min(img.height().saturating_sub(1));
    let crop_h = crop_bottom.saturating_sub(crop_top).saturating_add(1);
    if crop_top == 0 && crop_h == img.height() {
        return (img.clone(), 0);
    }

    (
        crop_imm(img, 0, crop_top, img.width(), crop_h).to_image(),
        crop_top,
    )
}

/// Also returns `(left, top)` -- the columns and rows removed off the
/// left and top edges, i.e. how far everything in the image moved left
/// and up.  gpuwm addition (VENDOR.md): the map-viewport crop is the
/// biggest mover of the three post-render passes (tens of pixels), and
/// until it reported its offset the plot rectangle could not follow the
/// map through it.
pub(crate) fn crop_canvas_whitespace(
    img: &RgbaImage,
    background: Rgba,
    pad: u32,
) -> (RgbaImage, (u32, u32)) {
    if img.width() <= 2 || img.height() <= 2 {
        return (img.clone(), (0, 0));
    }

    let mut min_x = img.width();
    let mut max_x = 0;
    let mut min_y = img.height();
    let mut max_y = 0;
    for y in 0..img.height() {
        for x in 0..img.width() {
            if !pixel_matches_background(*img.get_pixel(x, y), background) {
                min_x = min_x.min(x);
                max_x = max_x.max(x);
                min_y = min_y.min(y);
                max_y = max_y.max(y);
            }
        }
    }

    if min_x > max_x || min_y > max_y {
        return (img.clone(), (0, 0));
    }

    let crop_x = min_x.saturating_sub(pad);
    let crop_y = min_y.saturating_sub(pad);
    let crop_right = max_x.saturating_add(pad).min(img.width().saturating_sub(1));
    let crop_bottom = max_y
        .saturating_add(pad)
        .min(img.height().saturating_sub(1));
    let crop_w = crop_right.saturating_sub(crop_x).saturating_add(1);
    let crop_h = crop_bottom.saturating_sub(crop_y).saturating_add(1);
    if crop_x == 0 && crop_y == 0 && crop_w == img.width() && crop_h == img.height() {
        return (img.clone(), (0, 0));
    }

    (
        crop_imm(img, crop_x, crop_y, crop_w, crop_h).to_image(),
        (crop_x, crop_y),
    )
}

fn pixel_matches_background(px: image::Rgba<u8>, background: Rgba) -> bool {
    if px.0[3] <= 6 {
        return true;
    }

    let bg = background.to_image_rgba().0;
    let diff = px.0[0].abs_diff(bg[0]) as u16
        + px.0[1].abs_diff(bg[1]) as u16
        + px.0[2].abs_diff(bg[2]) as u16
        + px.0[3].abs_diff(bg[3]) as u16;
    diff <= 6
}

/// Also returns the signed horizontal shift it applied to the content
/// (positive = rightward).  gpuwm addition (VENDOR.md): this is the pass
/// every regional domain-framed panel takes, so until it reported its
/// shift no regional panel could publish a georeference at all.
pub(crate) fn center_horizontal_canvas_content(
    img: &RgbaImage,
    background: Rgba,
) -> (RgbaImage, i64) {
    if img.width() <= 2 {
        return (img.clone(), 0);
    }

    let mut min_x = img.width();
    let mut max_x = 0;
    for y in 0..img.height() {
        for x in 0..img.width() {
            if !pixel_matches_background(*img.get_pixel(x, y), background) {
                min_x = min_x.min(x);
                max_x = max_x.max(x);
            }
        }
    }

    if min_x > max_x {
        return (img.clone(), 0);
    }

    let left_margin = min_x;
    let right_margin = img.width().saturating_sub(max_x).saturating_sub(1);
    let shift = (right_margin as i64 - left_margin as i64) / 2;
    if shift == 0 {
        return (img.clone(), 0);
    }

    let mut centered = RgbaImage::from_pixel(img.width(), img.height(), background.to_image_rgba());
    for y in 0..img.height() {
        for x in 0..img.width() {
            let pixel = *img.get_pixel(x, y);
            if pixel_matches_background(pixel, background) {
                continue;
            }

            let dest_x = x as i64 + shift;
            if (0..img.width() as i64).contains(&dest_x) {
                centered.put_pixel(dest_x as u32, y, pixel);
            }
        }
    }

    (centered, shift)
}

pub fn encode_rgba_png_profile_with_options(
    image: &RgbaImage,
    options: &PngWriteOptions,
) -> (Vec<u8>, u128) {
    let encode_start = Instant::now();
    // Filter selection note: `Adaptive` tries all 5 filter types per
    // scanline and picks the best, that's expensive (~30-40% of encode
    // time for typical RGBA8 image content). `Up` is the next-best
    // single-filter choice for our typical map images and is well
    // within 5% of Adaptive's compressed size on this workload.
    let (compression, filter) = match options.compression {
        PngCompressionMode::Default => (CompressionType::Default, PngFilterType::Up),
        PngCompressionMode::Fast => (CompressionType::Fast, PngFilterType::Up),
        PngCompressionMode::Fastest => (CompressionType::Fast, PngFilterType::NoFilter),
    };
    let mut buf = Vec::new();
    let encoder = PngEncoder::new_with_quality(&mut buf, compression, filter);
    encoder
        .write_image(
            image.as_raw(),
            image.width(),
            image.height(),
            ExtendedColorType::Rgba8,
        )
        .expect("PNG encoding failed");
    (buf, encode_start.elapsed().as_millis())
}

pub fn render_to_png_profile_with_options(
    data: &[f64],
    ny: usize,
    nx: usize,
    opts: &RenderOpts,
    png_options: &PngWriteOptions,
) -> (Vec<u8>, RenderPngTiming) {
    let total_start = Instant::now();
    let (image, image_timing) = render_to_image_profile(data, ny, nx, opts);
    let render_to_image_ms = image_timing.total_ms;
    let (buf, png_encode_ms) = encode_rgba_png_profile_with_options(&image, png_options);
    let timing = RenderPngTiming {
        image_timing,
        render_to_image_ms,
        png_encode_ms,
        png_write_ms: 0,
        total_ms: total_start.elapsed().as_millis(),
    };
    (buf, timing)
}

pub fn render_to_png_profile(
    data: &[f64],
    ny: usize,
    nx: usize,
    opts: &RenderOpts,
) -> (Vec<u8>, RenderPngTiming) {
    render_to_png_profile_with_options(data, ny, nx, opts, &PngWriteOptions::default())
}

pub fn render_to_png(data: &[f64], ny: usize, nx: usize, opts: &RenderOpts) -> Vec<u8> {
    render_to_png_profile(data, ny, nx, opts).0
}

#[cfg(test)]
fn reset_projected_pixel_cache_for_tests() {
    PROJECTED_PIXEL_CACHE.with(|cache_cell| {
        *cache_cell.borrow_mut() = None;
    });
    PROJECTED_PIXEL_CACHE_MISSES.with(|count| count.set(0));
}

#[cfg(test)]
fn projected_pixel_cache_miss_count_for_tests() -> usize {
    PROJECTED_PIXEL_CACHE_MISSES.with(Cell::get)
}

#[cfg(test)]
mod tests;
