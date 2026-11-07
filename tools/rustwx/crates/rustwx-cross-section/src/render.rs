use rusttype::{Font, Scale, point};
use rustwx_contour::{ContourEngine, ContourLevels, RectilinearGrid, ScalarField2D};
use std::sync::OnceLock;
use std::time::Instant;

use crate::data::{ScalarSection, SectionMetadata};
use crate::error::CrossSectionError;
use crate::palette::CrossSectionPalette;
use crate::style::{CrossSectionProduct, CrossSectionStyle};
use crate::vertical::{VerticalAxis, VerticalKind, VerticalUnits};
use crate::wind::DecomposedWindGrid;

const SOURCE_SANS_3_REGULAR: &[u8] =
    include_bytes!("../../rustwx-render/assets/fonts/SourceSans3-Regular.ttf");
const SOURCE_SANS_3_SEMIBOLD: &[u8] =
    include_bytes!("../../rustwx-render/assets/fonts/SourceSans3-Semibold.ttf");
const MS_TO_KT_F32: f32 = 1.943_844_5;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CrossSectionFontKind {
    Regular,
    Semibold,
}

struct CrossSectionFontSet {
    regular: Option<Font<'static>>,
    semibold: Option<Font<'static>>,
}

static CROSS_SECTION_FONTS: OnceLock<CrossSectionFontSet> = OnceLock::new();

/// Font bytes a host installed before the first glyph was drawn (gpuwm
/// addition, VENDOR.md): a render theme names its own faces and every
/// surface should draw them, this crate included.  Installed after the
/// fonts have loaded it is a no-op; a byte slice that is not a font keeps
/// the embedded face for that weight.
static CROSS_SECTION_FONT_OVERRIDE: OnceLock<(Option<Vec<u8>>, Option<Vec<u8>>)> = OnceLock::new();

/// Install replacement font bytes (regular, semibold) for every later
/// cross-section render in this process.  Returns `false` when the fonts
/// were already loaded and the override could not take effect.
pub fn install_cross_section_fonts(regular: Option<Vec<u8>>, semibold: Option<Vec<u8>>) -> bool {
    CROSS_SECTION_FONT_OVERRIDE.set((regular, semibold)).is_ok() && CROSS_SECTION_FONTS.get().is_none()
}

/// Simple RGBA color.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Color {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

impl Color {
    pub const BLACK: Self = Self::rgb(0, 0, 0);
    pub const WHITE: Self = Self::rgb(255, 255, 255);
    pub const TRANSPARENT: Self = Self::rgba(0, 0, 0, 0);

    pub const fn rgb(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b, a: 255 }
    }

    pub const fn rgba(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }

    fn with_alpha(self, alpha: u8) -> Self {
        Self { a: alpha, ..self }
    }

    fn luminance(self) -> f32 {
        (0.2126 * self.r as f32 + 0.7152 * self.g as f32 + 0.0722 * self.b as f32) / 255.0
    }

    fn lerp(self, other: Self, fraction: f32) -> Self {
        let fraction = fraction.clamp(0.0, 1.0);
        let mix = |start: u8, end: u8| -> u8 {
            let start = start as f32;
            let end = end as f32;
            (start + (end - start) * fraction).round() as u8
        };
        Self {
            r: mix(self.r, other.r),
            g: mix(self.g, other.g),
            b: mix(self.b, other.b),
            a: mix(self.a, other.a),
        }
    }
}

/// Margins around the drawable plot area.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Insets {
    pub left: u32,
    pub right: u32,
    pub top: u32,
    pub bottom: u32,
}

impl Default for Insets {
    fn default() -> Self {
        Self {
            left: 82,
            right: 128,
            top: 64,
            bottom: 86,
        }
    }
}

/// Render options for [`render_scalar_section`].
#[derive(Debug, Clone, PartialEq)]
pub struct CrossSectionRenderRequest {
    pub width: u32,
    pub height: u32,
    pub margins: Insets,
    pub page_background_top: Color,
    pub page_background_bottom: Color,
    pub plot_background_top: Color,
    pub plot_background_bottom: Color,
    pub frame_color: Color,
    pub axis_color: Color,
    pub text_color: Color,
    pub grid_major_color: Color,
    pub grid_minor_color: Color,
    pub terrain_fill_top: Color,
    pub terrain_fill_bottom: Color,
    pub terrain_stroke: Color,
    pub terrain_highlight: Color,
    pub palette: Vec<Color>,
    pub value_range: Option<(f32, f32)>,
    pub value_ticks: Vec<f32>,
    /// Colour bar ticks with the text each one prints, for a fill whose
    /// drawn value is not the value a reader wants beside the bar: a fill
    /// drawn in log10 hands decade positions here with the field's own
    /// numbers as labels (-1 reads 0.1, 0 reads 1, 1 reads 10).  Empty
    /// leaves the ticks to `value_ticks` and the automatic labels (gpuwm
    /// addition, VENDOR.md).
    pub colorbar_ticks: Vec<(f32, String)>,
    pub colorbar_label: Option<String>,
    pub show_axes: bool,
    pub show_grid: bool,
    pub show_colorbar: bool,
    pub isotherms_c: Vec<f32>,
    pub highlight_isotherm_c: Option<f32>,
    pub isotherm_color: Color,
    pub highlight_isotherm_color: Color,
    pub wind_overlay: Option<WindOverlayBundle>,
    pub contour_overlays: Vec<ScalarContourOverlayBundle>,
    /// The producer's mark, drawn at the right of the title row.  `None`
    /// -- the default, and what every caller that names no theme passes --
    /// draws nothing and leaves the header exactly where it was.
    pub source_label: Option<String>,
    /// Type and chrome scale.  `None` derives it from the image width, so a
    /// 2400-pixel sheet is not drawn with the type and the margins of a
    /// 960-pixel one (gpuwm addition, VENDOR.md).
    pub type_scale: Option<f32>,
    /// The caller draws the header (title, times, provenance) itself, in
    /// the same header its map products carry: this renderer then draws no
    /// title row, and names the ends of the cut `A` and `B` above the top
    /// corners of the plot instead of in a route row under the axis
    /// (gpuwm addition, VENDOR.md).
    pub external_header: bool,
}

impl CrossSectionRenderRequest {
    pub fn with_dimensions(mut self, width: u32, height: u32) -> Self {
        self.width = width;
        self.height = height;
        self
    }

    pub fn with_palette(mut self, palette: Vec<Color>) -> Self {
        self.palette = palette;
        self
    }

    pub fn with_value_range(mut self, min_value: f32, max_value: f32) -> Self {
        self.value_range = Some((min_value, max_value));
        self
    }

    pub fn with_value_ticks(mut self, ticks: Vec<f32>) -> Self {
        self.value_ticks = ticks;
        self
    }

    pub fn with_colorbar_label(mut self, label: impl Into<String>) -> Self {
        self.colorbar_label = Some(label.into());
        self
    }

    /// Ticks at these drawn values, each printing its own text instead of
    /// the drawn value (see [`Self::colorbar_ticks`]).
    pub fn with_colorbar_ticks(mut self, ticks: Vec<(f32, String)>) -> Self {
        self.colorbar_ticks = ticks;
        self
    }

    /// Name the producer whose mark the title row carries.  Unset, the
    /// row is drawn as it always was.
    pub fn with_source_label(mut self, label: impl Into<String>) -> Self {
        self.source_label = Some(label.into());
        self
    }

    pub fn with_isotherms(mut self, levels_c: Vec<f32>, highlight_c: Option<f32>) -> Self {
        self.isotherms_c = levels_c;
        self.highlight_isotherm_c = highlight_c;
        self
    }

    pub fn with_margins(mut self, margins: Insets) -> Self {
        self.margins = margins;
        self
    }

    pub fn with_contour_overlays(mut self, overlays: Vec<ScalarContourOverlayBundle>) -> Self {
        self.contour_overlays = overlays;
        self
    }

    /// Fix the type and chrome scale instead of deriving it from the width.
    pub fn with_type_scale(mut self, scale: f32) -> Self {
        self.type_scale = Some(scale);
        self
    }

    /// The scale every piece of type and chrome on this render is drawn at.
    ///
    /// The crate's own sizes are laid out for a 960-pixel sheet.  Held
    /// there, a 2400-pixel delivery carries a 22-pixel title -- under one
    /// percent of the width, which is the "fonts are about ten pixels"
    /// complaint.  Dividing by 1400 keeps a title at or above 1.4 % of the
    /// width and axis, colourbar and legend labels at or above 0.9 % of it
    /// at every width, which is the bar this renderer is now held to.
    pub fn resolved_type_scale(&self) -> f32 {
        self.type_scale
            .filter(|scale| scale.is_finite() && *scale > 0.0)
            .unwrap_or_else(|| (self.width as f32 / 1400.0).clamp(1.0, 6.0))
    }
}

impl Default for CrossSectionRenderRequest {
    fn default() -> Self {
        Self {
            width: 960,
            height: 560,
            margins: Insets::default(),
            page_background_top: Color::WHITE,
            page_background_bottom: Color::WHITE,
            plot_background_top: Color::rgb(248, 250, 252),
            plot_background_bottom: Color::rgb(248, 250, 252),
            frame_color: Color::rgb(18, 22, 27),
            axis_color: Color::rgb(38, 45, 53),
            text_color: Color::rgb(20, 24, 29),
            grid_major_color: Color::rgba(0, 0, 0, 50),
            grid_minor_color: Color::rgba(0, 0, 0, 22),
            // A muted earth at its true height: the terrain is the frame's
            // floor, not its subject.  The opaque saturated brown it used
            // to be read as the loudest thing on a section whose subject
            // was the plume above it (gpuwm addition, VENDOR.md).
            terrain_fill_top: Color::rgba(150, 132, 108, 236),
            terrain_fill_bottom: Color::rgba(104, 92, 76, 248),
            terrain_stroke: Color::rgb(72, 60, 46),
            terrain_highlight: Color::TRANSPARENT,
            palette: CrossSectionPalette::default().build(),
            value_range: None,
            value_ticks: Vec::new(),
            colorbar_ticks: Vec::new(),
            colorbar_label: None,
            show_axes: true,
            show_grid: true,
            show_colorbar: true,
            isotherms_c: CrossSectionStyle::default().isotherms_c().to_vec(),
            highlight_isotherm_c: CrossSectionStyle::default().highlight_isotherm_c(),
            isotherm_color: Color::rgba(20, 24, 29, 205),
            // Red, not magenta: magenta is the vertical-velocity ink, and
            // one colour has to mean one thing on a sheet that carries both
            // (gpuwm addition, VENDOR.md).
            highlight_isotherm_color: Color::rgb(228, 26, 28),
            wind_overlay: None,
            contour_overlays: Vec::new(),
            source_label: None,
            type_scale: None,
            external_header: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ScalarContourOverlayBundle {
    pub section: ScalarSection,
    pub levels: Vec<f32>,
    pub highlight_level: Option<f32>,
    pub label: Option<String>,
    pub units: Option<String>,
    pub color: Color,
    pub highlight_color: Color,
}

impl ScalarContourOverlayBundle {
    pub fn new(section: ScalarSection, levels: Vec<f32>) -> Self {
        Self {
            section,
            levels,
            highlight_level: None,
            label: None,
            units: None,
            color: Color::rgba(20, 24, 29, 205),
            highlight_color: Color::rgb(214, 34, 190),
        }
    }

    pub fn with_highlight(mut self, level: f32) -> Self {
        self.highlight_level = Some(level);
        self
    }

    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    pub fn with_units(mut self, units: impl Into<String>) -> Self {
        self.units = Some(units.into());
        self
    }

    pub fn with_color(mut self, color: Color) -> Self {
        self.color = color;
        self
    }

    pub fn with_highlight_color(mut self, color: Color) -> Self {
        self.highlight_color = color;
        self
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct WindOverlayBundle {
    pub grid: DecomposedWindGrid,
    pub style: WindOverlayStyle,
    pub label: Option<String>,
}

impl WindOverlayBundle {
    pub fn new(grid: DecomposedWindGrid, style: WindOverlayStyle) -> Self {
        Self {
            grid,
            style,
            label: None,
        }
    }

    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WindOverlayStyle {
    pub stride_points: usize,
    pub stride_levels: usize,
    pub target_columns: usize,
    pub min_speed_ms: f32,
    pub max_speed_ms: f32,
    pub base_length_px: f32,
    pub max_length_px: f32,
    pub arrow_head_px: f32,
    pub cross_tick_px: f32,
    pub line_width: u32,
    pub color: Color,
}

impl Default for WindOverlayStyle {
    fn default() -> Self {
        Self {
            stride_points: 6,
            stride_levels: 2,
            target_columns: 10,
            min_speed_ms: 6.0,
            max_speed_ms: 35.0,
            base_length_px: 18.0,
            max_length_px: 18.0,
            arrow_head_px: 4.0,
            cross_tick_px: 5.0,
            line_width: 1,
            color: Color::rgba(0, 0, 0, 220),
        }
    }
}

/// Raw RGBA output from the lightweight rasterizer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderedCrossSection {
    width: u32,
    height: u32,
    rgba: Vec<u8>,
}

impl RenderedCrossSection {
    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn rgba(&self) -> &[u8] {
        &self.rgba
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CrossSectionRenderTiming {
    pub plot_layout_ms: u128,
    pub terrain_mask_ms: u128,
    pub scene_resolve_ms: u128,
    pub canvas_init_ms: u128,
    pub scalar_field_ms: u128,
    pub grid_ms: u128,
    pub contour_topology_ms: u128,
    pub contour_draw_ms: u128,
    pub wind_overlay_ms: u128,
    pub terrain_ms: u128,
    pub axes_ms: u128,
    pub header_ms: u128,
    pub footer_ms: u128,
    pub colorbar_ms: u128,
    pub total_ms: u128,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
struct OverlayContourTiming {
    topology_ms: u128,
    draw_ms: u128,
}

#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq)]
struct WindVectorGeometry {
    start: (f64, f64),
    end: (f64, f64),
    head_left: (f64, f64),
    head_right: (f64, f64),
}

#[derive(Debug, Clone, PartialEq)]
struct ResolvedRenderScene {
    palette: Vec<Color>,
    min_value: f32,
    max_value: f32,
    value_ticks: Vec<f32>,
    /// One label per entry of `value_ticks` when the request named its
    /// ticks; empty when the labels are the drawn values themselves.
    tick_labels: Vec<String>,
    /// The step the automatic ticks were cut on, `None` when the ticks
    /// came from a style or a request: the places a drawn-value label
    /// needs so that a tick between whole numbers reads as itself.
    tick_step: Option<f64>,
    colorbar_label: String,
    overlay_levels: Vec<f32>,
    highlight_overlay: Option<f32>,
    field_label: String,
    field_units: Option<String>,
}

impl ResolvedRenderScene {
    fn resolve(
        section: &ScalarSection,
        masked: &ScalarSection,
        request: &CrossSectionRenderRequest,
    ) -> Result<Self, CrossSectionError> {
        let declared_style = detect_declared_style(section.metadata());
        let declared_palette = detect_declared_palette(section.metadata());
        let finite_range = masked.finite_range();
        let (min_value, max_value) = request
            .value_range
            .or_else(|| {
                declared_style
                    .as_ref()
                    .and_then(CrossSectionStyle::value_range)
            })
            .or(finite_range)
            .ok_or(CrossSectionError::NoFiniteData)?;

        let palette = if request_uses_default_palette(request) {
            declared_style
                .as_ref()
                .map(|style| style.palette().build())
                .or_else(|| declared_palette.map(CrossSectionPalette::build))
                .unwrap_or_else(|| request.palette.clone())
        } else {
            request.palette.clone()
        };
        if palette.len() < 2 {
            return Err(CrossSectionError::EmptyColorRamp);
        }

        let (value_ticks, tick_labels, tick_step) = if !request.colorbar_ticks.is_empty() {
            (
                request.colorbar_ticks.iter().map(|(value, _)| *value).collect(),
                request
                    .colorbar_ticks
                    .iter()
                    .map(|(_, label)| label.clone())
                    .collect(),
                None,
            )
        } else if request.value_ticks.is_empty() {
            match declared_style
                .as_ref()
                .map(|style| style.value_ticks().to_vec())
                .filter(|ticks| !ticks.is_empty())
            {
                Some(ticks) => (ticks, Vec::new(), None),
                None => (
                    nice_value_ticks(min_value, max_value, 7),
                    Vec::new(),
                    Some(nice_value_tick_step(min_value, max_value, 7)),
                ),
            }
        } else {
            (request.value_ticks.clone(), Vec::new(), None)
        };

        let uses_default_overlay = request_uses_default_overlays(request);
        let overlay_levels = if uses_default_overlay {
            declared_style
                .as_ref()
                .map(|style| style.isotherms_c().to_vec())
                .unwrap_or_else(|| request.isotherms_c.clone())
        } else {
            request.isotherms_c.clone()
        };
        let mut overlay_levels = overlay_levels;
        normalize_levels(&mut overlay_levels);
        let highlight_overlay = if overlay_levels.is_empty() {
            None
        } else if uses_default_overlay {
            declared_style
                .as_ref()
                .and_then(CrossSectionStyle::highlight_isotherm_c)
                .filter(|value| {
                    overlay_levels
                        .iter()
                        .any(|level| (*level - *value).abs() <= 0.001)
                })
                .or(request.highlight_isotherm_c)
        } else {
            request.highlight_isotherm_c
        };

        let declared_product = declared_style.as_ref().map(CrossSectionStyle::product);
        let field_label = resolve_field_label(section.metadata(), declared_product);
        let field_units = section
            .metadata()
            .field_units
            .as_deref()
            .map(normalize_units_label)
            .or_else(|| {
                declared_product
                    .map(CrossSectionProduct::units)
                    .map(normalize_units_label)
            })
            .filter(|units| !units.is_empty());
        let colorbar_label = request
            .colorbar_label
            .clone()
            .or_else(|| {
                declared_style
                    .as_ref()
                    .and_then(|style| style.colorbar_label().map(str::to_string))
            })
            .unwrap_or_else(|| compose_label_with_units(&field_label, field_units.as_deref()));

        Ok(Self {
            palette,
            min_value,
            max_value,
            value_ticks,
            tick_labels,
            tick_step,
            colorbar_label,
            overlay_levels,
            highlight_overlay,
            field_label,
            field_units,
        })
    }
}

/// Renders a scalar cross-section to a simple RGBA buffer.
pub fn render_scalar_section(
    section: &ScalarSection,
    request: &CrossSectionRenderRequest,
) -> Result<RenderedCrossSection, CrossSectionError> {
    render_scalar_section_profile(section, request).map(|(rendered, _)| rendered)
}

pub fn render_scalar_section_profile(
    section: &ScalarSection,
    request: &CrossSectionRenderRequest,
) -> Result<(RenderedCrossSection, CrossSectionRenderTiming), CrossSectionError> {
    let total_start = Instant::now();
    if request.width < 2 || request.height < 2 {
        return Err(CrossSectionError::InvalidRenderDimensions);
    }

    let type_scale = request.resolved_type_scale();
    let plot_layout_start = Instant::now();
    let plot = PlotRect::from_request(request, type_scale)?;
    let plot_layout_ms = plot_layout_start.elapsed().as_millis();
    let terrain_mask_start = Instant::now();
    let masked = section.masked_with_terrain();
    let terrain_mask_ms = terrain_mask_start.elapsed().as_millis();
    let scene_resolve_start = Instant::now();
    let scene = ResolvedRenderScene::resolve(section, &masked, request)?;
    let scene_resolve_ms = scene_resolve_start.elapsed().as_millis();

    let canvas_init_start = Instant::now();
    let mut canvas = Canvas::new(
        request.width,
        request.height,
        request.page_background_top,
        request.page_background_bottom,
    );
    canvas.type_scale = type_scale;
    canvas.fill_rect_gradient(
        plot,
        request.plot_background_top,
        request.plot_background_bottom,
    );
    let canvas_init_ms = canvas_init_start.elapsed().as_millis();
    let scalar_field_start = Instant::now();
    render_scalar_field(&mut canvas, &plot, section, &scene);
    let scalar_field_ms = scalar_field_start.elapsed().as_millis();

    let mut grid_ms = 0;
    if request.show_grid {
        let grid_start = Instant::now();
        draw_grid(&mut canvas, &plot, &masked, request);
        grid_ms = grid_start.elapsed().as_millis();
    }

    let contour_timing =
        draw_overlay_contours_profile(&mut canvas, &plot, &masked, request, &scene);
    let wind_overlay_start = Instant::now();
    draw_wind_overlay(&mut canvas, &plot, &masked, request)?;
    let wind_overlay_ms = wind_overlay_start.elapsed().as_millis();
    let terrain_start = Instant::now();
    draw_terrain(&mut canvas, &plot, section, request);
    let terrain_ms = terrain_start.elapsed().as_millis();

    let mut axes_ms = 0;
    if request.show_axes {
        let axes_start = Instant::now();
        draw_axes(&mut canvas, &plot, &masked, request);
        axes_ms = axes_start.elapsed().as_millis();
    }

    let header_start = Instant::now();
    draw_header(&mut canvas, &plot, &masked, request, &scene);
    let header_ms = header_start.elapsed().as_millis();
    let footer_start = Instant::now();
    draw_footer(&mut canvas, &plot, &masked, request, &scene);
    let footer_ms = footer_start.elapsed().as_millis();

    let mut colorbar_ms = 0;
    if request.show_colorbar {
        let colorbar_start = Instant::now();
        draw_colorbar(&mut canvas, &plot, request, &scene);
        colorbar_ms = colorbar_start.elapsed().as_millis();
    }

    let rendered = RenderedCrossSection {
        width: request.width,
        height: request.height,
        rgba: canvas.rgba,
    };
    Ok((
        rendered,
        CrossSectionRenderTiming {
            plot_layout_ms,
            terrain_mask_ms,
            scene_resolve_ms,
            canvas_init_ms,
            scalar_field_ms,
            grid_ms,
            contour_topology_ms: contour_timing.topology_ms,
            contour_draw_ms: contour_timing.draw_ms,
            wind_overlay_ms,
            terrain_ms,
            axes_ms,
            header_ms,
            footer_ms,
            colorbar_ms,
            total_ms: total_start.elapsed().as_millis(),
        },
    ))
}

#[derive(Debug, Clone, Copy)]
struct PlotRect {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
}

/// The margins this render lays out with: a caller's own insets exactly as
/// given, and the crate's DEFAULT insets scaled with the type, because type
/// that grew with the sheet does not fit in margins that did not (gpuwm
/// addition, VENDOR.md).
fn effective_margins(request: &CrossSectionRenderRequest, type_scale: f32) -> Insets {
    if request.margins != Insets::default() {
        return request.margins;
    }
    let scaled = |value: u32| ((value as f32) * type_scale).round() as u32;
    Insets {
        left: scaled(request.margins.left),
        right: scaled(request.margins.right),
        top: scaled(request.margins.top),
        bottom: scaled(request.margins.bottom),
    }
}

impl PlotRect {
    fn from_request(
        request: &CrossSectionRenderRequest,
        type_scale: f32,
    ) -> Result<Self, CrossSectionError> {
        let margins = effective_margins(request, type_scale);
        let width = request
            .width
            .checked_sub(margins.left + margins.right)
            .ok_or(CrossSectionError::InvalidPlotMargins)?;
        let height = request
            .height
            .checked_sub(margins.top + margins.bottom)
            .ok_or(CrossSectionError::InvalidPlotMargins)?;

        if width < 2 || height < 2 {
            return Err(CrossSectionError::InvalidPlotMargins);
        }

        Ok(Self {
            x: margins.left,
            y: margins.top,
            width,
            height,
        })
    }

    fn right(&self) -> u32 {
        self.x + self.width - 1
    }

    fn bottom(&self) -> u32 {
        self.y + self.height - 1
    }

    fn contains(&self, x: i32, y: i32) -> bool {
        x >= self.x as i32
            && x <= self.right() as i32
            && y >= self.y as i32
            && y <= self.bottom() as i32
    }
}

struct Canvas {
    width: u32,
    height: u32,
    rgba: Vec<u8>,
    /// What every piece of type and chrome drawn on this canvas is scaled
    /// by; 1.0 until the render entry point sets it.
    type_scale: f32,
}

impl Canvas {
    /// A whole-pixel length from a 960-sheet design length.
    fn px(&self, base: i32) -> i32 {
        ((base as f32) * self.type_scale).round() as i32
    }

    /// The same, unsigned, never below one pixel.
    fn upx(&self, base: u32) -> u32 {
        (((base as f32) * self.type_scale).round() as u32).max(1)
    }

    fn new(width: u32, height: u32, top: Color, bottom: Color) -> Self {
        let mut canvas = Self {
            width,
            height,
            rgba: vec![0u8; (width * height * 4) as usize],
            type_scale: 1.0,
        };
        canvas.fill_vertical_gradient(0, 0, width, height, top, bottom);
        canvas
    }

    fn fill_vertical_gradient(
        &mut self,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
        top: Color,
        bottom: Color,
    ) {
        if width == 0 || height == 0 {
            return;
        }

        for dy in 0..height {
            let fraction = if height == 1 {
                0.0
            } else {
                dy as f32 / (height - 1) as f32
            };
            let color = top.lerp(bottom, fraction);
            for dx in 0..width {
                self.set_pixel(x + dx, y + dy, color);
            }
        }
    }

    fn fill_rect_gradient(&mut self, rect: PlotRect, top: Color, bottom: Color) {
        self.fill_vertical_gradient(rect.x, rect.y, rect.width, rect.height, top, bottom);
    }

    fn fill_rect(&mut self, x: u32, y: u32, width: u32, height: u32, color: Color) {
        for dy in 0..height {
            for dx in 0..width {
                self.blend_pixel((x + dx) as i32, (y + dy) as i32, color);
            }
        }
    }

    fn draw_rect(&mut self, rect: PlotRect, color: Color, thickness: u32) {
        for offset in 0..thickness {
            let x0 = rect.x.saturating_sub(offset);
            let y0 = rect.y.saturating_sub(offset);
            let x1 = (rect.right() + offset).min(self.width.saturating_sub(1));
            let y1 = (rect.bottom() + offset).min(self.height.saturating_sub(1));
            for x in x0..=x1 {
                self.blend_pixel(x as i32, y0 as i32, color);
                self.blend_pixel(x as i32, y1 as i32, color);
            }
            for y in y0..=y1 {
                self.blend_pixel(x0 as i32, y as i32, color);
                self.blend_pixel(x1 as i32, y as i32, color);
            }
        }
    }

    fn draw_line(
        &mut self,
        start: (f64, f64),
        end: (f64, f64),
        color: Color,
        thickness: u32,
        clip: Option<&PlotRect>,
    ) {
        let dx = end.0 - start.0;
        let dy = end.1 - start.1;
        let steps = dx.abs().max(dy.abs()).ceil().max(1.0) as usize;
        let radius = thickness.saturating_sub(1) as i32 / 2;

        for step in 0..=steps {
            let t = step as f64 / steps as f64;
            let x = start.0 + dx * t;
            let y = start.1 + dy * t;
            let xi = x.round() as i32;
            let yi = y.round() as i32;
            for oy in -radius..=radius {
                for ox in -radius..=radius {
                    if clip.is_none_or(|rect| rect.contains(xi + ox, yi + oy)) {
                        self.blend_pixel(xi + ox, yi + oy, color);
                    }
                }
            }
        }
    }

    fn set_pixel(&mut self, x: u32, y: u32, color: Color) {
        if x >= self.width || y >= self.height {
            return;
        }
        let idx = ((y * self.width + x) * 4) as usize;
        self.rgba[idx] = color.r;
        self.rgba[idx + 1] = color.g;
        self.rgba[idx + 2] = color.b;
        self.rgba[idx + 3] = color.a;
    }

    fn blend_pixel(&mut self, x: i32, y: i32, color: Color) {
        if x < 0 || y < 0 || x >= self.width as i32 || y >= self.height as i32 {
            return;
        }
        let idx = ((y as u32 * self.width + x as u32) * 4) as usize;
        if color.a == 255 {
            self.rgba[idx] = color.r;
            self.rgba[idx + 1] = color.g;
            self.rgba[idx + 2] = color.b;
            self.rgba[idx + 3] = 255;
            return;
        }
        if color.a == 0 {
            return;
        }

        let alpha = color.a as f32 / 255.0;
        let inv = 1.0 - alpha;
        self.rgba[idx] = (color.r as f32 * alpha + self.rgba[idx] as f32 * inv).round() as u8;
        self.rgba[idx + 1] =
            (color.g as f32 * alpha + self.rgba[idx + 1] as f32 * inv).round() as u8;
        self.rgba[idx + 2] =
            (color.b as f32 * alpha + self.rgba[idx + 2] as f32 * inv).round() as u8;
        self.rgba[idx + 3] = 255;
    }

    fn draw_text(
        &mut self,
        x: i32,
        y: i32,
        text: &str,
        color: Color,
        scale: u32,
        shadow: Option<Color>,
    ) {
        if let Some(shadow) = shadow {
            self.draw_text(
                x + scale as i32,
                y + scale as i32,
                text,
                shadow,
                scale,
                None,
            );
        }

        if let Some(font) = cross_section_font(scale) {
            self.draw_ttf_text(x, y, text, color, scale, font);
            return;
        }

        let mut cursor_x = x;
        let mut cursor_y = y;
        let bitmap_scale = (((scale.max(1) as f32) * self.type_scale).round() as i32).max(1);
        for ch in text.chars() {
            if ch == '\n' {
                cursor_x = x;
                cursor_y += 8 * bitmap_scale;
                continue;
            }

            let glyph = glyph_rows(ch);
            for (row_index, row) in glyph.iter().enumerate() {
                for col in 0..5 {
                    if (row >> (4 - col)) & 1 == 1 {
                        for sy in 0..bitmap_scale {
                            for sx in 0..bitmap_scale {
                                self.blend_pixel(
                                    cursor_x + col * bitmap_scale + sx,
                                    cursor_y + row_index as i32 * bitmap_scale + sy,
                                    color,
                                );
                            }
                        }
                    }
                }
            }
            cursor_x += 6 * bitmap_scale;
        }
    }

    fn draw_ttf_text(
        &mut self,
        x: i32,
        y: i32,
        text: &str,
        color: Color,
        scale: u32,
        font: &Font<'static>,
    ) {
        let kind = font_kind_for_scale(scale);
        let scale = Scale::uniform(cross_section_font_size_px(scale, kind) * self.type_scale);
        let v_metrics = font.v_metrics(scale);
        let line_height = cross_section_line_height_px(scale, font).max(1.0).ceil() as i32;

        for (line_index, line) in text.split('\n').enumerate() {
            let baseline_y = y + line_index as i32 * line_height;
            let glyphs = font.layout(
                line,
                scale,
                point(x as f32, baseline_y as f32 + v_metrics.ascent),
            );

            for glyph in glyphs {
                if let Some(bb) = glyph.pixel_bounding_box() {
                    glyph.draw(|gx, gy, coverage| {
                        let px = bb.min.x + gx as i32;
                        let py = bb.min.y + gy as i32;
                        let alpha = ((color.a as f32) * coverage).round().clamp(0.0, 255.0) as u8;
                        self.blend_pixel(
                            px,
                            py,
                            Color {
                                r: color.r,
                                g: color.g,
                                b: color.b,
                                a: alpha,
                            },
                        );
                    });
                }
            }
        }
    }
}

/// Below this much of a sample quad being finite, a fill pixel is not drawn
/// at all: the remainder is one corner's dust, not a field.
const FILL_COVERAGE_FLOOR: f32 = 0.04;

fn render_scalar_field(
    canvas: &mut Canvas,
    plot: &PlotRect,
    section: &ScalarSection,
    scene: &ResolvedRenderScene,
) {
    let start_distance = section.distances_km()[0];
    let end_distance = section.distances_km()[section.n_points() - 1];

    for plot_y in 0..plot.height {
        let y_fraction = if plot.height == 1 {
            0.0
        } else {
            plot_y as f64 / (plot.height as f64 - 1.0)
        };
        let axis_value = section.vertical_axis().value_at_plot_fraction(y_fraction);

        for plot_x in 0..plot.width {
            let x_fraction = if plot.width == 1 {
                0.0
            } else {
                plot_x as f64 / (plot.width as f64 - 1.0)
            };
            let distance_km = start_distance + x_fraction * (end_distance - start_distance);

            // Smooth shading between samples, and a smooth FADE where the
            // sample quad is only partly finite.  Dropping a whole quad on
            // one missing corner is what drew a 1 km field as a staircase
            // of blocks (gpuwm addition, VENDOR.md).
            let Some((value, coverage)) =
                section.bilinear_sample_coverage(distance_km, axis_value)
            else {
                continue;
            };
            if coverage <= FILL_COVERAGE_FLOOR {
                continue;
            }
            let color = map_value_to_color(value, scene.min_value, scene.max_value, &scene.palette);
            let alpha = ((color.a as f32) * coverage).round().clamp(0.0, 255.0) as u8;
            if alpha == 0 {
                continue;
            }
            canvas.blend_pixel(
                (plot.x + plot_x) as i32,
                (plot.y + plot_y) as i32,
                color.with_alpha(alpha),
            );
        }
    }
}

fn draw_grid(
    canvas: &mut Canvas,
    plot: &PlotRect,
    section: &ScalarSection,
    request: &CrossSectionRenderRequest,
) {
    for tick in vertical_ticks(section.vertical_axis()) {
        if let Some(y) = axis_value_to_pixel(section.vertical_axis(), plot, tick) {
            canvas.draw_line(
                (plot.x as f64, y),
                (plot.right() as f64, y),
                request.grid_major_color,
                1,
                Some(plot),
            );
        }
    }

    for tick in intermediate_vertical_ticks(section.vertical_axis()) {
        if let Some(y) = axis_value_to_pixel(section.vertical_axis(), plot, tick) {
            canvas.draw_line(
                (plot.x as f64, y),
                (plot.right() as f64, y),
                request.grid_minor_color,
                1,
                Some(plot),
            );
        }
    }

    for tick in distance_ticks(section.distances_km(), 6) {
        if let Some(x) = distance_to_pixel(section, plot, tick) {
            canvas.draw_line(
                (x, plot.y as f64),
                (x, plot.bottom() as f64),
                request.grid_major_color,
                1,
                Some(plot),
            );
        }
    }

    for tick in intermediate_distance_ticks(section.distances_km(), 6) {
        if let Some(x) = distance_to_pixel(section, plot, tick) {
            canvas.draw_line(
                (x, plot.y as f64),
                (x, plot.bottom() as f64),
                request.grid_minor_color,
                1,
                Some(plot),
            );
        }
    }
}

fn draw_overlay_contours_profile(
    canvas: &mut Canvas,
    plot: &PlotRect,
    section: &ScalarSection,
    request: &CrossSectionRenderRequest,
    scene: &ResolvedRenderScene,
) -> OverlayContourTiming {
    let mut timing = OverlayContourTiming::default();
    if !scene.overlay_levels.is_empty() {
        let fallback = draw_single_contour_overlay(
            canvas,
            plot,
            section,
            section,
            &scene.overlay_levels,
            scene.highlight_overlay,
            request.isotherm_color,
            request.highlight_isotherm_color,
            scene.field_units.as_deref(),
        );
        timing.topology_ms += fallback.topology_ms;
        timing.draw_ms += fallback.draw_ms;
    }

    for overlay in &request.contour_overlays {
        let masked_overlay = overlay.section.masked_with_terrain();
        let overlay_timing = draw_single_contour_overlay(
            canvas,
            plot,
            section,
            &masked_overlay,
            &overlay.levels,
            overlay.highlight_level,
            overlay.color,
            overlay.highlight_color,
            overlay.units.as_deref(),
        );
        timing.topology_ms += overlay_timing.topology_ms;
        timing.draw_ms += overlay_timing.draw_ms;
    }

    timing
}

fn draw_single_contour_overlay(
    canvas: &mut Canvas,
    plot: &PlotRect,
    plot_section: &ScalarSection,
    contour_section: &ScalarSection,
    levels: &[f32],
    highlight_overlay: Option<f32>,
    contour_color: Color,
    highlight_color: Color,
    label_units: Option<&str>,
) -> OverlayContourTiming {
    let topology_start = Instant::now();
    if contour_section.n_points() != plot_section.n_points()
        || contour_section.n_levels() != plot_section.n_levels()
    {
        return OverlayContourTiming::default();
    }

    let mut levels = levels.to_vec();
    if let Some(highlight) = highlight_overlay {
        levels.push(highlight);
    }
    normalize_levels(&mut levels);
    if levels.is_empty() {
        return OverlayContourTiming::default();
    }

    let Ok(grid) = RectilinearGrid::new(
        contour_section.distances_km().to_vec(),
        contour_section.vertical_axis().levels().to_vec(),
    ) else {
        return OverlayContourTiming::default();
    };
    let values = contour_section
        .values()
        .iter()
        .map(|value| *value as f64)
        .collect::<Vec<_>>();
    let Ok(field) = ScalarField2D::new(grid, values) else {
        return OverlayContourTiming::default();
    };
    let contour_levels = levels.iter().map(|value| *value as f64).collect::<Vec<_>>();
    let Ok(levels) = ContourLevels::new(contour_levels) else {
        return OverlayContourTiming::default();
    };

    let topology = ContourEngine::new().extract_isolines(&field, &levels);
    let topology_ms = topology_start.elapsed().as_millis();
    let draw_start = Instant::now();

    for layer in &topology.layers {
        let level = layer.level as f32;
        let highlighted =
            highlight_overlay.is_some_and(|candidate| (candidate - level).abs() <= 0.001);
        let color = if highlighted {
            highlight_color
        } else {
            contour_color
        };
        let thickness = canvas.upx(if highlighted { 3 } else { 1 });
        let halo = contour_halo_color(color).with_alpha(if highlighted { 126 } else { 58 });

        for segment in &layer.segments {
            let Some(start) = data_point_to_pixel(
                plot_section,
                plot,
                segment.geometry.start.x,
                segment.geometry.start.y,
            ) else {
                continue;
            };
            let Some(end) = data_point_to_pixel(
                plot_section,
                plot,
                segment.geometry.end.x,
                segment.geometry.end.y,
            ) else {
                continue;
            };
            canvas.draw_line(start, end, halo, thickness + 2, Some(plot));
            canvas.draw_line(start, end, color, thickness, Some(plot));
        }

        // The highlighted level is labelled too.  It is the one line on a
        // section a reader looks for by name -- the -10 C isotherm on a
        // seeding sheet -- and leaving it as the only unlabelled contour
        // made it the only one that had to be guessed at (gpuwm addition,
        // VENDOR.md).
        let label = contour_label_for_level(level, label_units);
        draw_contour_label(canvas, plot, plot_section, layer, color, &label);
    }
    OverlayContourTiming {
        topology_ms,
        draw_ms: draw_start.elapsed().as_millis(),
    }
}

fn draw_wind_overlay(
    canvas: &mut Canvas,
    plot: &PlotRect,
    section: &ScalarSection,
    request: &CrossSectionRenderRequest,
) -> Result<(), CrossSectionError> {
    let Some(overlay) = request.wind_overlay.as_ref() else {
        return Ok(());
    };

    if overlay.grid.n_levels() != section.n_levels()
        || overlay.grid.n_points() != section.n_points()
    {
        return Err(CrossSectionError::ShapeMismatch {
            context: "wind overlay",
            expected: section.n_levels() * section.n_points(),
            actual: overlay.grid.n_levels() * overlay.grid.n_points(),
        });
    }

    let stride_levels = overlay.style.stride_levels.max(1);
    let point_indices = wind_barb_point_indices(
        section.n_points(),
        overlay.style.target_columns,
        overlay.style.stride_points.max(1),
    );
    let axis_levels = section.vertical_axis().levels();
    let terrain = section.terrain();

    for level_index in (0..section.n_levels()).step_by(stride_levels) {
        for &point_index in &point_indices {
            let Some(speed_ms) = overlay.grid.speed_value(level_index, point_index) else {
                continue;
            };
            let Some(along_ms) = overlay.grid.along_section_value(level_index, point_index) else {
                continue;
            };
            let Some(left_ms) = overlay.grid.left_of_section_value(level_index, point_index) else {
                continue;
            };
            if !(speed_ms.is_finite() && along_ms.is_finite() && left_ms.is_finite()) {
                continue;
            }
            if speed_ms < overlay.style.min_speed_ms {
                continue;
            }

            let distance_km = section.distances_km()[point_index];
            let axis_value = axis_levels[level_index];
            if terrain.is_some_and(|terrain| {
                terrain
                    .below_surface(section.vertical_axis(), distance_km, axis_value)
                    .unwrap_or(false)
            }) {
                continue;
            }

            let Some((center_x, center_y)) =
                data_point_to_pixel(section, plot, distance_km, axis_value)
            else {
                continue;
            };

            draw_section_wind_barb(
                canvas,
                (center_x, center_y),
                along_ms,
                left_ms,
                speed_ms,
                overlay.style,
                plot,
            );
        }
    }

    Ok(())
}

fn wind_barb_point_indices(
    n_points: usize,
    target_columns: usize,
    fallback_stride: usize,
) -> Vec<usize> {
    if n_points == 0 {
        return Vec::new();
    }
    if target_columns == 0 {
        return (0..n_points).step_by(fallback_stride.max(1)).collect();
    }
    let count = target_columns.min(n_points);
    if count <= 1 {
        return vec![0];
    }
    (0..count)
        .map(|index| ((index * (n_points - 1) + (count - 1) / 2) / (count - 1)).min(n_points - 1))
        .collect()
}

#[cfg(test)]
fn draw_section_wind_vector(
    canvas: &mut Canvas,
    center: (f64, f64),
    along_ms: f32,
    left_ms: f32,
    speed_ms: f32,
    style: WindOverlayStyle,
    plot: &PlotRect,
) {
    let Some(geometry) =
        resolve_section_wind_vector_geometry(center, along_ms, left_ms, speed_ms, style)
    else {
        return;
    };
    let halo = contour_halo_color(style.color).with_alpha(90);
    canvas.draw_line(
        geometry.start,
        geometry.end,
        halo,
        style.line_width + 2,
        Some(plot),
    );
    canvas.draw_line(
        geometry.start,
        geometry.end,
        style.color,
        style.line_width,
        Some(plot),
    );
    canvas.draw_line(
        geometry.end,
        geometry.head_left,
        halo,
        style.line_width + 2,
        Some(plot),
    );
    canvas.draw_line(
        geometry.end,
        geometry.head_left,
        style.color,
        style.line_width,
        Some(plot),
    );
    canvas.draw_line(
        geometry.end,
        geometry.head_right,
        halo,
        style.line_width + 2,
        Some(plot),
    );
    canvas.draw_line(
        geometry.end,
        geometry.head_right,
        style.color,
        style.line_width,
        Some(plot),
    );
}

fn draw_section_wind_barb(
    canvas: &mut Canvas,
    center: (f64, f64),
    along_ms: f32,
    left_ms: f32,
    speed_ms: f32,
    style: WindOverlayStyle,
    plot: &PlotRect,
) {
    if !(along_ms.is_finite() && left_ms.is_finite() && speed_ms.is_finite()) {
        return;
    }

    let vector_norm = f64::from(along_ms).hypot(f64::from(left_ms));
    if vector_norm < 0.001 {
        return;
    }

    let shaft_length = style.max_length_px.max(style.base_length_px + 4.0) as f64;
    let unit_x = f64::from(along_ms) / vector_norm;
    let unit_y = -f64::from(left_ms) / vector_norm;
    let half_length = shaft_length * 0.5;
    let start = (
        center.0 - unit_x * half_length,
        center.1 - unit_y * half_length,
    );
    let end = (
        center.0 + unit_x * half_length,
        center.1 + unit_y * half_length,
    );
    draw_wind_line(canvas, start, end, style.color, style.line_width, plot);

    let barb_len = style.cross_tick_px.max(5.0) as f64;
    let spacing = (style.cross_tick_px.max(5.0) * 0.82) as f64;
    let perp_x = -unit_y;
    let perp_y = unit_x;
    let mut speed_kt = (speed_ms * MS_TO_KT_F32 / 5.0).round() * 5.0;
    let mut offset = 1.5f64;

    while speed_kt >= 47.5 {
        let base = (end.0 - unit_x * offset, end.1 - unit_y * offset);
        let next = (
            end.0 - unit_x * (offset + spacing),
            end.1 - unit_y * (offset + spacing),
        );
        let tip = (base.0 + perp_x * barb_len, base.1 + perp_y * barb_len);
        draw_wind_line(canvas, base, tip, style.color, style.line_width, plot);
        draw_wind_line(canvas, next, tip, style.color, style.line_width, plot);
        speed_kt -= 50.0;
        offset += spacing * 1.35;
    }

    while speed_kt >= 7.5 {
        let base = (end.0 - unit_x * offset, end.1 - unit_y * offset);
        let tip = (base.0 + perp_x * barb_len, base.1 + perp_y * barb_len);
        draw_wind_line(canvas, base, tip, style.color, style.line_width, plot);
        speed_kt -= 10.0;
        offset += spacing;
    }

    if speed_kt >= 2.5 {
        let base = (end.0 - unit_x * offset, end.1 - unit_y * offset);
        let tip = (
            base.0 + perp_x * barb_len * 0.55,
            base.1 + perp_y * barb_len * 0.55,
        );
        draw_wind_line(canvas, base, tip, style.color, style.line_width, plot);
    }
}

fn draw_wind_line(
    canvas: &mut Canvas,
    start: (f64, f64),
    end: (f64, f64),
    color: Color,
    thickness: u32,
    plot: &PlotRect,
) {
    canvas.draw_line(start, end, color, thickness, Some(plot));
}

#[cfg(test)]
fn resolve_section_wind_vector_geometry(
    center: (f64, f64),
    along_ms: f32,
    left_ms: f32,
    speed_ms: f32,
    style: WindOverlayStyle,
) -> Option<WindVectorGeometry> {
    if !(along_ms.is_finite() && left_ms.is_finite() && speed_ms.is_finite()) {
        return None;
    }

    let vector_norm = f64::from(along_ms).hypot(f64::from(left_ms));
    if vector_norm < 0.001 {
        return None;
    }

    let normalized_speed =
        (speed_ms / style.max_speed_ms.max(style.min_speed_ms + 1.0)).clamp(0.0, 1.0);
    let shaft_length =
        style.base_length_px + (style.max_length_px - style.base_length_px) * normalized_speed;
    let unit_x = f64::from(along_ms) / vector_norm;
    let unit_y = -f64::from(left_ms) / vector_norm;
    let half_length = shaft_length as f64 * 0.5;
    let start = (
        center.0 - unit_x * half_length,
        center.1 - unit_y * half_length,
    );
    let end = (
        center.0 + unit_x * half_length,
        center.1 + unit_y * half_length,
    );

    let head_length = style.arrow_head_px.max(2.0) as f64;
    let head_half_width = (style.cross_tick_px.max(2.0) * 0.5) as f64;
    let back_x = -unit_x * head_length;
    let back_y = -unit_y * head_length;
    let perp_x = -unit_y;
    let perp_y = unit_x;
    let head_left = (
        end.0 + back_x + perp_x * head_half_width,
        end.1 + back_y + perp_y * head_half_width,
    );
    let head_right = (
        end.0 + back_x - perp_x * head_half_width,
        end.1 + back_y - perp_y * head_half_width,
    );

    Some(WindVectorGeometry {
        start,
        end,
        head_left,
        head_right,
    })
}

fn draw_contour_label(
    canvas: &mut Canvas,
    plot: &PlotRect,
    section: &ScalarSection,
    layer: &rustwx_contour::ContourLayer,
    color: Color,
    label: &str,
) {
    let Some(segment) = layer.segments.iter().max_by(|left, right| {
        left.geometry
            .length_squared()
            .partial_cmp(&right.geometry.length_squared())
            .unwrap_or(std::cmp::Ordering::Equal)
    }) else {
        return;
    };

    let Some(start) = data_point_to_pixel(
        section,
        plot,
        segment.geometry.start.x,
        segment.geometry.start.y,
    ) else {
        return;
    };
    let Some(end) = data_point_to_pixel(
        section,
        plot,
        segment.geometry.end.x,
        segment.geometry.end.y,
    ) else {
        return;
    };
    let text_width = measure_text_width(label, 1, canvas.type_scale) as i32;
    let line_height = text_line_height(1, canvas.type_scale) as i32;
    let mid_x = ((start.0 + end.0) * 0.5).round() as i32;
    let mid_y = ((start.1 + end.1) * 0.5).round() as i32;
    let label_x =
        (mid_x - text_width / 2).clamp(plot.x as i32 + 4, plot.right() as i32 - text_width - 4);
    let label_y = (mid_y - line_height / 2).clamp(
        plot.y as i32 + 2,
        plot.bottom() as i32 - line_height - 2,
    );

    canvas.draw_text(
        label_x + 1,
        label_y + 1,
        label,
        Color::WHITE.with_alpha(210),
        1,
        None,
    );
    canvas.draw_text(label_x, label_y, label, color, 1, None);
}

fn draw_terrain(
    canvas: &mut Canvas,
    plot: &PlotRect,
    section: &ScalarSection,
    request: &CrossSectionRenderRequest,
) {
    let Some(terrain) = section.terrain() else {
        return;
    };

    let axis = section.vertical_axis();
    let mut surface_ys = Vec::with_capacity(plot.width as usize);
    for plot_x in 0..plot.width {
        let fraction = if plot.width == 1 {
            0.0
        } else {
            plot_x as f64 / (plot.width as f64 - 1.0)
        };
        let distance_km = section.distances_km()[0]
            + fraction
                * (section.distances_km()[section.n_points() - 1] - section.distances_km()[0]);

        let Some(surface_value) = terrain.surface_value_on_axis(axis, distance_km) else {
            continue;
        };

        let Some(surface_y) = terrain_surface_y(axis, plot, surface_value) else {
            surface_ys.push(None);
            continue;
        };

        surface_ys.push(Some(surface_y.min(plot.bottom())));
    }

    for plot_x in 0..plot.width {
        let Some(surface_y) = surface_ys[plot_x as usize] else {
            continue;
        };
        let column_height = plot.bottom().saturating_sub(surface_y) + 1;

        // Top-to-bottom through the terrain body, not one flat ink: the
        // ground reads as ground and stops competing with the field above
        // it (gpuwm addition, VENDOR.md).
        for offset in 0..column_height {
            let fraction = if column_height <= 1 {
                0.0
            } else {
                offset as f32 / (column_height - 1) as f32
            };
            canvas.blend_pixel(
                (plot.x + plot_x) as i32,
                (surface_y + offset) as i32,
                request
                    .terrain_fill_top
                    .lerp(request.terrain_fill_bottom, fraction),
            );
        }
        let rim = canvas.upx(2).min(4);
        for offset in 0..rim {
            canvas.blend_pixel(
                (plot.x + plot_x) as i32,
                surface_y as i32 + offset as i32,
                request.terrain_stroke,
            );
        }
        if surface_y > plot.y {
            canvas.blend_pixel(
                (plot.x + plot_x) as i32,
                surface_y as i32 - 1,
                request.terrain_stroke.with_alpha(155),
            );
        }
    }
}

fn draw_axes(
    canvas: &mut Canvas,
    plot: &PlotRect,
    section: &ScalarSection,
    request: &CrossSectionRenderRequest,
) {
    canvas.draw_rect(*plot, request.frame_color, 2);

    for tick in vertical_ticks(section.vertical_axis()) {
        let Some(y) = axis_value_to_pixel(section.vertical_axis(), plot, tick) else {
            continue;
        };
        canvas.draw_line(
            (plot.x as f64 - f64::from(canvas.px(7)), y),
            (plot.x as f64, y),
            request.axis_color,
            1,
            None,
        );
        let label = format_axis_tick(section.vertical_axis(), tick);
        let label_x =
            plot.x as i32 - measure_text_width(&label, 1, canvas.type_scale) as i32 - canvas.px(10);
        canvas.draw_text(
            label_x,
            y.round() as i32 - canvas.px(7),
            &label,
            request.text_color,
            1,
            Some(label_halo(request)),
        );
    }

    for tick in distance_ticks(section.distances_km(), 6) {
        let Some(x) = distance_to_pixel(section, plot, tick) else {
            continue;
        };
        canvas.draw_line(
            (x, plot.bottom() as f64),
            (x, plot.bottom() as f64 + f64::from(canvas.px(7))),
            request.axis_color,
            1,
            None,
        );
        let label = format_distance_tick(tick);
        let label_width = measure_text_width(&label, 1, canvas.type_scale) as i32;
        let label_x = (x.round() as i32 - label_width / 2).clamp(
            plot.x as i32,
            plot.right() as i32 - label_width,
        );
        canvas.draw_text(
            label_x,
            plot.bottom() as i32 + canvas.px(10),
            &label,
            request.text_color,
            1,
            Some(label_halo(request)),
        );
    }

    let axis_title = axis_label(section.vertical_axis());
    canvas.draw_text(
        plot.x as i32
            - measure_text_width(&axis_title, 1, canvas.type_scale) as i32
            - canvas.px(12),
        plot.y as i32 - canvas.px(22),
        &axis_title,
        request.text_color,
        1,
        Some(label_halo(request)),
    );
}

/// The halo behind axis labels: the page colour at upstream's alpha, so a
/// dark page halos in its own black instead of white (gpuwm addition,
/// VENDOR.md; on the default white page this is upstream's exact colour).
fn label_halo(request: &CrossSectionRenderRequest) -> Color {
    request.page_background_top.with_alpha(175)
}

/// The header's secondary ink: upstream's grey on a light page, the text
/// ink at reduced alpha on a dark one.
fn muted_text(request: &CrossSectionRenderRequest) -> Color {
    if request.page_background_top.luminance() < 0.5 {
        request.text_color.with_alpha(170)
    } else {
        Color::rgb(82, 88, 96)
    }
}

/// The legend's backing: the page colour at upstream's alpha.
fn legend_backing(request: &CrossSectionRenderRequest) -> Color {
    request.page_background_top.with_alpha(232)
}

fn draw_header(
    canvas: &mut Canvas,
    plot: &PlotRect,
    section: &ScalarSection,
    request: &CrossSectionRenderRequest,
    scene: &ResolvedRenderScene,
) {
    if request.external_header {
        draw_reference_legend(canvas, plot, request, scene);
        return;
    }
    let title = section
        .metadata()
        .title
        .as_deref()
        .map(str::to_string)
        .unwrap_or_else(|| {
            format!(
                "Cross-Section: {}",
                compose_label_with_units(&scene.field_label, scene.field_units.as_deref())
            )
        });
    // ONE header row: the title on the left, the timing and the producer's
    // mark right-aligned on the SAME baseline, the mark furthest right.
    // The mark used to be laid on a second row seven pixels below the
    // title, which at any real delivery size overlapped it (gpuwm addition,
    // VENDOR.md).
    let title_scale = if measure_text_width(&title, 2, canvas.type_scale) + plot.x + canvas.upx(28)
        <= canvas.width
    {
        2
    } else {
        1
    };
    let title_height = text_line_height(title_scale, canvas.type_scale) as i32;
    let row_height = text_line_height(1, canvas.type_scale) as i32;
    let title_y = (plot.y as i32 - canvas.px(20) - title_height).max(canvas.px(8));

    canvas.draw_text(
        plot.x as i32,
        title_y,
        &title,
        request.text_color,
        title_scale,
        None,
    );

    let side_y = title_y + (title_height - row_height) / 2;
    let mut right_edge = plot.right() as i32;
    let source = request
        .source_label
        .as_deref()
        .map(str::trim)
        .filter(|mark| !mark.is_empty());
    if let Some(mark) = source {
        let width = measure_text_width(mark, 1, canvas.type_scale) as i32;
        canvas.draw_text(
            right_edge - width,
            side_y,
            mark,
            muted_text(request),
            1,
            None,
        );
        right_edge -= width + canvas.px(28);
    }

    let timing = format_header_timing(section.metadata());
    if !timing.is_empty() {
        let timing_width = measure_text_width(&timing, 1, canvas.type_scale) as i32;
        canvas.draw_text(
            right_edge - timing_width,
            side_y,
            &timing,
            muted_text(request),
            1,
            None,
        );
    }

    draw_reference_legend(canvas, plot, request, scene);
}

fn draw_footer(
    canvas: &mut Canvas,
    plot: &PlotRect,
    section: &ScalarSection,
    request: &CrossSectionRenderRequest,
    _scene: &ResolvedRenderScene,
) {
    let center_label = section.metadata().attribute("distance_label").unwrap_or("Distance (km)");
    let center_x = plot.x as i32
        + (plot.width as i32 - measure_text_width(center_label, 1, canvas.type_scale) as i32) / 2;
    canvas.draw_text(
        center_x,
        plot.bottom() as i32 + canvas.px(36),
        center_label,
        request.text_color,
        1,
        None,
    );

    let start_label = section
        .metadata()
        .attribute("start_label")
        .unwrap_or("Start");
    let end_label = section.metadata().attribute("end_label").unwrap_or("End");
    if request.external_header {
        // The ends of the cut at the ends of the top axis, where a reader
        // looks for which way the section runs.
        let label_y = plot.y as i32 - canvas.px(22);
        let start_text = format!("A  {start_label}");
        let end_text = format!("B  {end_label}");
        canvas.draw_text(plot.x as i32, label_y, &start_text, request.text_color, 1, None);
        let end_width = measure_text_width(&end_text, 1, canvas.type_scale) as i32;
        canvas.draw_text(
            plot.right() as i32 - end_width,
            label_y,
            &end_text,
            request.text_color,
            1,
            None,
        );
        return;
    }
    let route_label = section.metadata().attribute("route_label").unwrap_or("");
    let route_y = plot.bottom() as i32 + canvas.px(62);
    let start_text = format!("A  {start_label}");
    let end_text = format!("B  {end_label}");
    canvas.draw_text(
        plot.x as i32,
        route_y,
        &start_text,
        request.text_color,
        1,
        None,
    );
    let end_width = measure_text_width(&end_text, 1, canvas.type_scale) as i32;
    canvas.draw_text(
        plot.right() as i32 - end_width,
        route_y,
        &end_text,
        request.text_color,
        1,
        None,
    );

    if !route_label.is_empty() {
        let route_width = measure_text_width(route_label, 1, canvas.type_scale) as i32;
        if route_width + canvas.px(40) < plot.width as i32 {
            canvas.draw_text(
                plot.x as i32 + (plot.width as i32 - route_width) / 2,
                route_y,
                route_label,
                muted_text(request),
                1,
                None,
            );
        }
    }
}

fn draw_colorbar(
    canvas: &mut Canvas,
    plot: &PlotRect,
    request: &CrossSectionRenderRequest,
    scene: &ResolvedRenderScene,
) {
    let bar_x = plot.right() + canvas.upx(28);
    let bar_width = canvas.upx(20);
    let label_x = bar_x + bar_width + canvas.upx(9);
    if label_x + canvas.upx(20) >= canvas.width {
        return;
    }
    let bar_y = plot.y;
    let bar_height = plot.height.max(40);

    for offset in 0..bar_height {
        let fraction = if bar_height == 1 {
            0.0
        } else {
            1.0 - offset as f32 / (bar_height - 1) as f32
        };
        let value = scene.min_value + fraction * (scene.max_value - scene.min_value);
        let color = map_value_to_color(value, scene.min_value, scene.max_value, &scene.palette);
        for dx in 0..bar_width {
            // The bar is BLENDED onto the page, not written over it: a
            // sequential fill's ramp fades out at its absent end, and
            // writing that alpha straight into the buffer punched a
            // transparent hole in the bar -- white on a dark sheet
            // (gpuwm addition, VENDOR.md).
            canvas.set_pixel(bar_x + dx, bar_y + offset, request.page_background_top);
            canvas.blend_pixel((bar_x + dx) as i32, (bar_y + offset) as i32, color);
        }
    }

    canvas.draw_rect(
        PlotRect {
            x: bar_x,
            y: bar_y,
            width: bar_width,
            height: bar_height,
        },
        request.frame_color,
        1,
    );

    let label = scene.colorbar_label.as_str();
    let label_width = measure_text_width(label, 1, canvas.type_scale) as i32;
    let label_left = (canvas.width as i32 - label_width - canvas.px(4))
        .max(canvas.px(4))
        .min(bar_x as i32);
    canvas.draw_text(
        label_left,
        plot.y as i32 - text_line_height(1, canvas.type_scale) as i32 - canvas.px(6),
        label,
        request.text_color,
        1,
        None,
    );

    let value_labels = value_tick_labels(&scene.value_ticks, scene.tick_step);
    for (index, &tick) in scene.value_ticks.iter().enumerate() {
        if tick < scene.min_value || tick > scene.max_value {
            continue;
        }
        let fraction = if (scene.max_value - scene.min_value).abs() <= f32::EPSILON {
            0.5
        } else {
            1.0 - ((tick - scene.min_value) / (scene.max_value - scene.min_value)).clamp(0.0, 1.0)
        };
        let y = bar_y as f32 + fraction * (bar_height.saturating_sub(1) as f32);
        canvas.draw_line(
            (bar_x as f64 + bar_width as f64 + 1.0, y as f64),
            (label_x as f64 - 3.0, y as f64),
            request.axis_color.with_alpha(110),
            1,
            None,
        );
        let tick_label = scene
            .tick_labels
            .get(index)
            .cloned()
            .unwrap_or_else(|| value_labels[index].clone());
        canvas.draw_text(
            label_x as i32,
            y.round() as i32 - text_line_height(1, canvas.type_scale) as i32 / 2,
            &tick_label,
            request.text_color,
            1,
            None,
        );
    }

    if let Some(highlight) = scene.highlight_overlay {
        if highlight >= scene.min_value && highlight <= scene.max_value {
            let fraction = if (scene.max_value - scene.min_value).abs() <= f32::EPSILON {
                0.5
            } else {
                1.0 - ((highlight - scene.min_value) / (scene.max_value - scene.min_value))
                    .clamp(0.0, 1.0)
            };
            let y = bar_y as f32 + fraction * (bar_height.saturating_sub(1) as f32);
            let reach = f64::from(canvas.px(20));
            canvas.draw_line(
                (bar_x as f64 - 2.0, y as f64),
                (label_x as f64 + reach, y as f64),
                contour_halo_color(request.highlight_isotherm_color).with_alpha(86),
                canvas.upx(4),
                None,
            );
            canvas.draw_line(
                (bar_x as f64 - 2.0, y as f64),
                (label_x as f64 + reach, y as f64),
                request.highlight_isotherm_color,
                canvas.upx(2),
                None,
            );
        }
    }
}

enum LegendSymbol {
    Line(Color),
    /// One overlay's ordinary ink and its highlight ink, drawn as the two
    /// halves of a single swatch so the highlight costs no second row.
    Split(Color, Color),
    Barb(Color),
}

struct LegendEntry {
    label: String,
    symbol: LegendSymbol,
}

/// EXACTLY one legend row per thing drawn: the built-in contour set, each
/// contour overlay, the wind barbs.
///
/// WHAT BREAKAGE THIS PREVENTS (gate law): the delivered agent
/// section listed "wa" and "T" twice, once for the overlay and again for
/// its highlighted level.  A highlight is not a second quantity -- it is
/// one level of the same overlay drawn heavier -- so it folds into that
/// overlay's own row, which names the level and carries the highlight ink
/// as the right half of its swatch.
fn legend_entries(
    request: &CrossSectionRenderRequest,
    scene: &ResolvedRenderScene,
) -> Vec<LegendEntry> {
    let mut entries = Vec::new();
    if request.contour_overlays.is_empty() && !scene.overlay_levels.is_empty() {
        let (label, symbol) = match scene.highlight_overlay {
            Some(level) => (
                format!("Contours ({})", contour_label_for_level(level, Some("C"))),
                LegendSymbol::Split(request.isotherm_color, request.highlight_isotherm_color),
            ),
            None => (
                "Contours".to_string(),
                LegendSymbol::Line(request.isotherm_color),
            ),
        };
        entries.push(LegendEntry { label, symbol });
    }

    for overlay in &request.contour_overlays {
        if overlay.levels.is_empty() && overlay.highlight_level.is_none() {
            continue;
        }
        let label = overlay
            .label
            .clone()
            .unwrap_or_else(|| "Contours".to_string());
        let units = overlay
            .units
            .as_deref()
            .map(str::trim)
            .filter(|units| !units.is_empty());
        let (label, symbol) = match overlay.highlight_level {
            Some(level) => {
                let level_text = contour_label_for_level(level, units);
                let named = match units {
                    Some(units) if !units.eq_ignore_ascii_case("c") => {
                        format!("{label} ({level_text} {units})")
                    }
                    _ => format!("{label} ({level_text})"),
                };
                if overlay.highlight_color == overlay.color {
                    (named, LegendSymbol::Line(overlay.color))
                } else {
                    (
                        named,
                        LegendSymbol::Split(overlay.color, overlay.highlight_color),
                    )
                }
            }
            None => (label, LegendSymbol::Line(overlay.color)),
        };
        entries.push(LegendEntry { label, symbol });
    }

    if let Some(overlay) = request.wind_overlay.as_ref() {
        entries.push(LegendEntry {
            label: "Wind Barbs (kt)".to_string(),
            symbol: LegendSymbol::Barb(overlay.style.color),
        });
    }
    entries
}

fn draw_reference_legend(
    canvas: &mut Canvas,
    plot: &PlotRect,
    request: &CrossSectionRenderRequest,
    scene: &ResolvedRenderScene,
) {
    let entries = legend_entries(request, scene);
    if entries.is_empty() {
        return;
    }

    let max_label_width = entries
        .iter()
        .map(|entry| measure_text_width(&entry.label, 1, canvas.type_scale))
        .max()
        .unwrap_or(0);
    let row_height = (text_line_height(1, canvas.type_scale) + canvas.upx(7)).max(canvas.upx(21));
    let swatch_left = canvas.upx(11);
    let swatch_right = canvas.upx(34);
    let text_left = canvas.upx(44);
    let desired_width = (max_label_width + text_left + canvas.upx(20)).max(canvas.upx(160));
    let max_width = plot.width.saturating_sub(canvas.upx(16)).max(canvas.upx(96));
    let width = desired_width.min(max_width);
    let height = canvas.upx(10) + entries.len() as u32 * row_height;
    if height + canvas.upx(16) >= plot.height {
        return;
    }

    let x = plot.x + canvas.upx(10);
    let y = plot.y + canvas.upx(10);
    let rect = PlotRect {
        x,
        y,
        width,
        height,
    };
    canvas.fill_rect(x, y, width, height, legend_backing(request));
    canvas.draw_rect(rect, request.frame_color.with_alpha(145), 1);

    let legend_clip = PlotRect {
        x,
        y,
        width,
        height,
    };
    let halo = canvas.upx(3);
    let stroke = canvas.upx(2);
    for (index, entry) in entries.iter().enumerate() {
        let row_y = y as i32 + canvas.px(5) + index as i32 * row_height as i32;
        let mid_y = row_y as f64 + f64::from(text_line_height(1, canvas.type_scale) as i32) * 0.5;
        let left = x as f64 + f64::from(swatch_left);
        let right = x as f64 + f64::from(swatch_right);
        match entry.symbol {
            LegendSymbol::Line(color) => {
                canvas.draw_line((left, mid_y), (right, mid_y), Color::WHITE.with_alpha(180), halo, None);
                canvas.draw_line((left, mid_y), (right, mid_y), color, stroke, None);
            }
            LegendSymbol::Split(color, highlight) => {
                let middle = (left + right) * 0.5;
                canvas.draw_line((left, mid_y), (right, mid_y), Color::WHITE.with_alpha(180), halo, None);
                canvas.draw_line((left, mid_y), (middle, mid_y), color, stroke, None);
                canvas.draw_line((middle, mid_y), (right, mid_y), highlight, halo, None);
            }
            LegendSymbol::Barb(color) => {
                let barb = 12.0 * canvas.type_scale;
                draw_section_wind_barb(
                    canvas,
                    ((left + right) * 0.5, mid_y),
                    12.0,
                    0.0,
                    barb,
                    WindOverlayStyle {
                        min_speed_ms: 0.0,
                        max_speed_ms: 24.0,
                        base_length_px: 18.0 * canvas.type_scale,
                        max_length_px: 18.0 * canvas.type_scale,
                        cross_tick_px: 6.0 * canvas.type_scale,
                        color,
                        ..WindOverlayStyle::default()
                    },
                    &legend_clip,
                );
            }
        }
        canvas.draw_text(
            x as i32 + text_left as i32,
            row_y,
            &entry.label,
            request.text_color,
            1,
            None,
        );
    }
}

fn format_header_timing(metadata: &SectionMetadata) -> String {
    let mut parts = Vec::new();
    if let Some(init) = metadata
        .attribute("init_label")
        .or_else(|| metadata.attribute("store_cycle"))
    {
        parts.push(format!("Init: {}", compact_datetime_label(init)));
    }
    if let Some(forecast_hour) = metadata.attribute("forecast_hour") {
        parts.push(forecast_hour.to_string());
    }
    if let Some(valid) = metadata
        .attribute("valid_time")
        .or(metadata.valid_label.as_deref())
    {
        parts.push(format!("Valid: {}", compact_datetime_label(valid)));
    }
    parts.join("   ")
}

fn compact_datetime_label(value: &str) -> String {
    value
        .split_whitespace()
        .map(compact_datetime_token)
        .collect::<Vec<_>>()
        .join(" ")
}

fn compact_datetime_token(value: &str) -> String {
    let trimmed = value.trim();
    let bytes = trimmed.as_bytes();
    if bytes.len() >= 13
        && bytes[0..8].iter().all(u8::is_ascii_digit)
        && bytes[8] == b'T'
        && bytes[9..11].iter().all(u8::is_ascii_digit)
    {
        return format!(
            "{}-{}-{} {}Z",
            &trimmed[0..4],
            &trimmed[4..6],
            &trimmed[6..8],
            &trimmed[9..11]
        );
    }
    trimmed.to_string()
}

fn terrain_surface_y(axis: &VerticalAxis, plot: &PlotRect, surface_value: f64) -> Option<u32> {
    if let Some(frac) = axis.plot_fraction_of_value(surface_value) {
        Some(plot.y + (frac * (plot.height as f64 - 1.0)).round() as u32)
    } else {
        match axis.kind() {
            VerticalKind::Pressure if surface_value <= axis.plot_top() => Some(plot.y),
            VerticalKind::Pressure if surface_value >= axis.plot_bottom() => None,
            VerticalKind::Height if surface_value >= axis.plot_top() => Some(plot.y),
            VerticalKind::Height if surface_value <= axis.plot_bottom() => None,
            _ => None,
        }
    }
}

fn map_value_to_color(value: f32, min_value: f32, max_value: f32, palette: &[Color]) -> Color {
    let fraction = if (max_value - min_value).abs() <= f32::EPSILON {
        0.5
    } else {
        ((value - min_value) / (max_value - min_value)).clamp(0.0, 1.0)
    };

    let scaled = fraction * (palette.len() as f32 - 1.0);
    let left = scaled.floor() as usize;
    let right = scaled.ceil().min((palette.len() - 1) as f32) as usize;
    if left == right {
        palette[left]
    } else {
        palette[left].lerp(palette[right], scaled - left as f32)
    }
}

fn distance_to_pixel(section: &ScalarSection, plot: &PlotRect, distance_km: f64) -> Option<f64> {
    let start = section.distances_km()[0];
    let end = section.distances_km()[section.n_points() - 1];
    if distance_km < start || distance_km > end {
        return None;
    }
    let fraction = if (end - start).abs() <= f64::EPSILON {
        0.0
    } else {
        (distance_km - start) / (end - start)
    };
    Some(plot.x as f64 + fraction * (plot.width as f64 - 1.0))
}

fn axis_value_to_pixel(axis: &VerticalAxis, plot: &PlotRect, value: f64) -> Option<f64> {
    let fraction = axis.plot_fraction_of_value(value)?;
    Some(plot.y as f64 + fraction * (plot.height as f64 - 1.0))
}

fn data_point_to_pixel(
    section: &ScalarSection,
    plot: &PlotRect,
    distance_km: f64,
    axis_value: f64,
) -> Option<(f64, f64)> {
    Some((
        distance_to_pixel(section, plot, distance_km)?,
        axis_value_to_pixel(section.vertical_axis(), plot, axis_value)?,
    ))
}

fn axis_label(axis: &VerticalAxis) -> String {
    match (axis.kind(), axis.units()) {
        (VerticalKind::Pressure, _) => "Pressure (hPa)".to_string(),
        (VerticalKind::Height, VerticalUnits::Meters) => "Height (m)".to_string(),
        (VerticalKind::Height, VerticalUnits::Kilometers) => "Height (km)".to_string(),
        (VerticalKind::Height, VerticalUnits::Hectopascals) => "Height".to_string(),
    }
}

fn vertical_ticks(axis: &VerticalAxis) -> Vec<f64> {
    match axis.kind() {
        VerticalKind::Pressure => {
            let preferred = [
                1000.0, 925.0, 850.0, 700.0, 600.0, 500.0, 400.0, 300.0, 250.0, 200.0, 150.0, 100.0,
            ];
            let min = axis.plot_top().min(axis.plot_bottom());
            let max = axis.plot_top().max(axis.plot_bottom());
            let mut ticks = preferred
                .into_iter()
                .filter(|tick| *tick >= min && *tick <= max)
                .collect::<Vec<_>>();
            if ticks.len() < 3 {
                ticks = axis.levels().to_vec();
            }
            ticks
        }
        VerticalKind::Height => {
            let min = axis.plot_bottom();
            let max = axis.plot_top();
            let step = nice_step((max - min).abs() / 6.0).max(0.5);
            ranged_ticks(min, max, step)
        }
    }
}

fn intermediate_vertical_ticks(axis: &VerticalAxis) -> Vec<f64> {
    let major = vertical_ticks(axis);
    major
        .windows(2)
        .filter_map(|pair| {
            let midpoint = (pair[0] + pair[1]) * 0.5;
            axis.plot_fraction_of_value(midpoint).map(|_| midpoint)
        })
        .collect()
}

fn distance_ticks(distances_km: &[f64], desired_count: usize) -> Vec<f64> {
    let start = distances_km[0];
    let end = distances_km[distances_km.len() - 1];
    let step = nice_step((end - start).abs() / desired_count.max(1) as f64).max(1.0);
    ranged_ticks(start, end, step)
}

fn intermediate_distance_ticks(distances_km: &[f64], desired_count: usize) -> Vec<f64> {
    let major = distance_ticks(distances_km, desired_count);
    major
        .windows(2)
        .map(|pair| (pair[0] + pair[1]) * 0.5)
        .collect()
}

fn nice_value_ticks(min: f32, max: f32, desired_count: usize) -> Vec<f32> {
    let min = min as f64;
    let max = max as f64;
    // No floor of 1.0 on the step.
    //
    // WHAT BREAKAGE THIS PREVENTS (gate law): a supercooled
    // liquid section runs 0 to about 0.6 g kg-1, and a step floored at one
    // whole unit gave that colourbar exactly two ticks -- its two ends --
    // with nothing in between to read a value against.  `nice_step` already
    // refuses a non-positive span.
    let step = nice_value_tick_step(min as f32, max as f32, desired_count);
    ranged_ticks(min, max, step)
        .into_iter()
        .map(|tick| tick as f32)
        .collect()
}

/// The step [`nice_value_ticks`] cuts `min..max` on.
fn nice_value_tick_step(min: f32, max: f32, desired_count: usize) -> f64 {
    nice_step((max as f64 - min as f64).abs() / desired_count.max(1) as f64)
}

/// A tick loop may not run away.
///
/// WHAT BREAKAGE THIS PREVENTS (gate law): the loop below walks
/// a range by a step, and a range whose ends are enormous (a field with no
/// data at all, whose min and max both come back as f32::MIN) has a step
/// too small to advance the accumulator at that magnitude -- so the loop
/// never terminates and the process grows a Vec until the kernel kills it.
/// That was measured: 122 GB of resident memory and no PNG.
const MAX_AXIS_TICKS: usize = 256;

fn ranged_ticks(start: f64, end: f64, step: f64) -> Vec<f64> {
    if !start.is_finite() || !end.is_finite() || !step.is_finite() || step <= 0.0 {
        return Vec::new();
    }

    let min = start.min(end);
    let max = start.max(end);
    let mut ticks = Vec::new();
    // Ticks are rounded against the STEP's own magnitude, not against a
    // fixed thousandth.
    //
    // WHAT BREAKAGE THIS PREVENTS (gate law): a supercooled
    // liquid section runs to a fraction of a gram per kilogram, so its
    // ticks are ten-thousandths -- and rounding those to 1/1000 collapsed
    // every one of them to 0.0.  The delivered control sections carried a
    // colourbar with one label on it, reading zero.
    let quantum = 10f64.powi((step.log10().floor() as i32) - 3);
    let round_to = |value: f64| -> f64 {
        if quantum > 0.0 && quantum.is_finite() {
            (value / quantum).round() * quantum
        } else {
            value
        }
    };
    let mut tick = (min / step).ceil() * step;
    while tick <= max + step * 0.25 {
        ticks.push(round_to(tick));
        let advanced = tick + step;
        if !advanced.is_finite() || advanced <= tick || ticks.len() >= MAX_AXIS_TICKS {
            break;
        }
        tick = advanced;
    }
    if ticks
        .first()
        .is_none_or(|first| (*first - min).abs() > step * 0.35)
    {
        ticks.insert(0, min);
    }
    if ticks
        .last()
        .is_none_or(|last| (*last - max).abs() > step * 0.35)
    {
        ticks.push(max);
    }
    ticks
}

fn nice_step(raw: f64) -> f64 {
    if !raw.is_finite() || raw <= 0.0 {
        return 1.0;
    }
    let exponent = raw.log10().floor();
    let base = 10f64.powf(exponent);
    let fraction = raw / base;
    let nice = if fraction <= 1.0 {
        1.0
    } else if fraction <= 2.0 {
        2.0
    } else if fraction <= 2.5 {
        2.5
    } else if fraction <= 5.0 {
        5.0
    } else {
        10.0
    };
    nice * base
}

fn normalize_levels(levels: &mut Vec<f32>) {
    levels.retain(|value| value.is_finite());
    levels.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));
    levels.dedup_by(|left, right| (*left - *right).abs() <= 0.001);
}

fn detect_declared_style(metadata: &SectionMetadata) -> Option<CrossSectionStyle> {
    let mut style = [
        metadata.attribute("product_style"),
        metadata.attribute("render_style"),
        metadata.attribute("product_key"),
        metadata.attribute("field_key"),
        metadata.field_name.as_deref(),
    ]
    .into_iter()
    .flatten()
    .find_map(CrossSectionStyle::from_name)?;
    if let Some(palette) = detect_declared_palette(metadata) {
        style = style.with_palette(palette);
    }
    if let Some(label) = metadata.attribute("colorbar_label") {
        style = style.with_colorbar_label(label);
    }
    Some(style)
}

fn detect_declared_palette(metadata: &SectionMetadata) -> Option<CrossSectionPalette> {
    metadata
        .attribute("palette")
        .or_else(|| metadata.attribute("palette_name"))
        .and_then(CrossSectionPalette::from_name)
}

fn resolve_field_label(
    metadata: &SectionMetadata,
    declared_product: Option<CrossSectionProduct>,
) -> String {
    declared_product
        .map(product_badge_label)
        .map(str::to_string)
        .or_else(|| metadata.field_name.as_deref().map(normalize_field_label))
        .unwrap_or_else(|| "FIELD".to_string())
}

fn product_badge_label(product: CrossSectionProduct) -> &'static str {
    match product {
        CrossSectionProduct::Temperature => "TEMP",
        CrossSectionProduct::WindSpeed => "WIND",
        CrossSectionProduct::ThetaE => "THETA-E",
        CrossSectionProduct::RelativeHumidity => "RH",
        CrossSectionProduct::SpecificHumidity => "Q",
        CrossSectionProduct::Omega => "OMEGA",
        CrossSectionProduct::Vorticity => "VORT",
        CrossSectionProduct::Shear => "SHEAR",
        CrossSectionProduct::LapseRate => "LAPSE RATE",
        CrossSectionProduct::CloudWater => "CLOUD WATER",
        CrossSectionProduct::TotalCondensate => "CONDENSATE",
        CrossSectionProduct::WetBulb => "WET BULB",
        CrossSectionProduct::Icing => "ICING",
        CrossSectionProduct::Frontogenesis => "FRONTO",
        CrossSectionProduct::Smoke => "SMOKE",
        CrossSectionProduct::VaporPressureDeficit => "VPD",
        CrossSectionProduct::DewpointDepression => "DPD",
        CrossSectionProduct::MoistureTransport => "MOISTURE XPORT",
        CrossSectionProduct::PotentialVorticity => "PV",
        CrossSectionProduct::FireWeather => "FIRE WX",
    }
}

fn normalize_field_label(value: &str) -> String {
    CrossSectionProduct::from_name(value)
        .map(product_badge_label)
        .map(str::to_string)
        .unwrap_or_else(|| value.replace(['_', '-'], " ").to_ascii_uppercase())
}

fn normalize_units_label(value: &str) -> String {
    match value.trim().to_ascii_lowercase().as_str() {
        "c" | "degc" | "celsius" => "C".to_string(),
        "f" | "degf" | "fahrenheit" => "F".to_string(),
        "k" | "degk" | "kelvin" => "K".to_string(),
        "hpa" => "HPA".to_string(),
        "m" | "meter" | "meters" => "M".to_string(),
        "km" => "KM".to_string(),
        "m/s" | "ms-1" | "mps" | "ms^-1" => "M/S".to_string(),
        "kt" | "kts" | "knot" | "knots" => "KT".to_string(),
        "dbz" => "DBZ".to_string(),
        "%" => "%".to_string(),
        other => other.to_ascii_uppercase(),
    }
}

fn compose_label_with_units(label: &str, units: Option<&str>) -> String {
    match units {
        Some("%") => format!("{label} %"),
        Some(units) if !units.is_empty() => format!("{label} {units}"),
        _ => label.to_string(),
    }
}

fn request_uses_default_palette(request: &CrossSectionRenderRequest) -> bool {
    request.palette == CrossSectionPalette::default().build()
}

fn request_uses_default_overlays(request: &CrossSectionRenderRequest) -> bool {
    let default_style = CrossSectionStyle::default();
    request.isotherms_c.as_slice() == default_style.isotherms_c()
        && request.highlight_isotherm_c == default_style.highlight_isotherm_c()
}

fn contour_halo_color(color: Color) -> Color {
    if color.luminance() >= 0.6 {
        Color::BLACK
    } else {
        Color::WHITE
    }
}

fn format_axis_tick(axis: &VerticalAxis, value: f64) -> String {
    match axis.units() {
        VerticalUnits::Kilometers => format_scalar_value(value as f32),
        _ => format!("{value:.0}"),
    }
}

fn format_distance_tick(value: f64) -> String {
    format!("{value:.0}")
}

fn format_scalar_value(value: f32) -> String {
    if value == 0.0 {
        return "0".to_string();
    }
    let magnitude = value.abs();
    if magnitude < 1.0 {
        // WHAT BREAKAGE THIS PREVENTS (gate law): one decimal
        // place turned every tick of a cloud-water colourbar -- 0.02, 0.04,
        // 0.06 g kg-1 -- into the string "0", and a scale whose every label
        // reads zero is not a scale.  The places come from the value.
        let decimals = (2 - magnitude.log10().ceil() as i32).clamp(1, 5) as usize;
        let text = format!("{value:.decimals$}");
        let trimmed = text.trim_end_matches('0').trim_end_matches('.');
        return if trimmed.is_empty() || trimmed == "-" {
            "0".to_string()
        } else {
            trimmed.to_string()
        };
    }
    if (value - value.round()).abs() <= 0.05 || magnitude >= 100.0 {
        format!("{value:.0}")
    } else {
        format!("{value:.1}")
    }
}

/// The places `step` needs to be written exactly, at most six.
fn places_for_step(step: f64) -> usize {
    if !step.is_finite() || step <= 0.0 {
        return 0;
    }
    (0..=6)
        .find(|places| {
            let scaled = step * 10f64.powi(*places as i32);
            (scaled - scaled.round()).abs() <= 1e-6 * scaled.abs().max(1.0)
        })
        .unwrap_or(6)
}

/// The places [`format_scalar_value`] gives `value` on its own.
fn own_places(value: f32) -> usize {
    let magnitude = value.abs();
    if value == 0.0 {
        0
    } else if magnitude < 1.0 {
        (2 - magnitude.log10().ceil() as i32).clamp(1, 5) as usize
    } else if (value - value.round()).abs() <= 0.05 || magnitude >= 100.0 {
        0
    } else {
        1
    }
}

/// The colourbar's drawn-value labels, as one set.
///
/// WHAT BREAKAGE THIS PREVENTS (gate law): a section bar whose ticks
/// sit between whole numbers prints them rounded. At 100 and above a
/// value is written with no places, so a real theta section cut to 6 km
/// (305.3 to 320.9 K, ticks every 2.5 K) labelled 307.5 as `308`, 312.5
/// as `312` and 317.5 as `318`, and a quarter step below 100 printed
/// 1.25 as `1.2`. Each label now carries at least the places the tick
/// step needs, so a bar on whole-number steps is spelled exactly as
/// before and a bar on 2.5 or 0.25 steps reads its own values.
fn value_tick_labels(ticks: &[f32], step: Option<f64>) -> Vec<String> {
    let step = step.or_else(|| {
        let mut sorted: Vec<f64> = ticks.iter().map(|tick| f64::from(*tick)).collect();
        sorted.sort_by(f64::total_cmp);
        sorted
            .windows(2)
            .map(|pair| pair[1] - pair[0])
            .filter(|gap| *gap > 0.0)
            .min_by(f64::total_cmp)
    });
    let step_places = step.map_or(0, places_for_step);
    ticks
        .iter()
        .map(|tick| {
            if step_places <= own_places(*tick) {
                return format_scalar_value(*tick);
            }
            let text = format!("{tick:.step_places$}");
            let trimmed = text.trim_end_matches('0').trim_end_matches('.');
            if trimmed.is_empty() || trimmed == "-" || trimmed == "-0" {
                "0".to_string()
            } else {
                trimmed.to_string()
            }
        })
        .collect()
}

fn contour_label_for_level(value: f32, units: Option<&str>) -> String {
    match units.map(str::trim).filter(|units| !units.is_empty()) {
        Some(units) if units.eq_ignore_ascii_case("c") => {
            format!("{}C", format_scalar_value(value))
        }
        _ => format_scalar_value(value),
    }
}

#[cfg(test)]
fn badge_rect(x: i32, y: i32, text: &str, scale: u32) -> (u32, u32, u32, u32) {
    let pad_x = 4 * scale.max(1);
    let width = measure_text_width(text, scale, 1.0) + pad_x * 2;
    let height = text_line_height(scale, 1.0) + 4;
    (x.max(0) as u32, y.max(0) as u32, width, height)
}

#[cfg(test)]
fn measure_badge_width(text: &str, scale: u32) -> u32 {
    badge_rect(0, 0, text, scale).2
}

fn measure_text_width(text: &str, scale: u32, type_scale: f32) -> u32 {
    if let Some(font) = cross_section_font(scale) {
        let font_kind = font_kind_for_scale(scale);
        let scale = Scale::uniform(cross_section_font_size_px(scale, font_kind) * type_scale);
        let v_metrics = font.v_metrics(scale);
        return text
            .split('\n')
            .map(|line| {
                let glyphs: Vec<_> = font
                    .layout(line, scale, point(0.0, v_metrics.ascent))
                    .collect();
                glyphs
                    .iter()
                    .rev()
                    .find_map(|glyph| glyph.pixel_bounding_box().map(|bb| bb.max.x.max(0) as u32))
                    .or_else(|| {
                        glyphs.last().map(|glyph| {
                            let end =
                                glyph.position().x + glyph.unpositioned().h_metrics().advance_width;
                            end.max(0.0).ceil() as u32
                        })
                    })
                    .unwrap_or(0)
            })
            .max()
            .unwrap_or(0);
    }

    ((text.chars().count() as f32) * 6.0 * (scale.max(1) as f32) * type_scale).round() as u32
}

fn text_line_height(scale: u32, type_scale: f32) -> u32 {
    if let Some(font) = cross_section_font(scale) {
        let font_kind = font_kind_for_scale(scale);
        let scale = Scale::uniform(cross_section_font_size_px(scale, font_kind) * type_scale);
        return cross_section_line_height_px(scale, font).ceil() as u32;
    }

    ((8.0 * (scale.max(1) as f32)) * type_scale).round() as u32
}

fn font_kind_for_scale(scale: u32) -> CrossSectionFontKind {
    if scale >= 2 {
        CrossSectionFontKind::Semibold
    } else {
        CrossSectionFontKind::Regular
    }
}

fn cross_section_font(scale: u32) -> Option<&'static Font<'static>> {
    let fonts = CROSS_SECTION_FONTS.get_or_init(load_cross_section_fonts);
    match font_kind_for_scale(scale) {
        CrossSectionFontKind::Regular => fonts.regular.as_ref(),
        CrossSectionFontKind::Semibold => fonts.semibold.as_ref().or(fonts.regular.as_ref()),
    }
}

fn load_cross_section_fonts() -> CrossSectionFontSet {
    let (regular, semibold) = CROSS_SECTION_FONT_OVERRIDE
        .get()
        .cloned()
        .unwrap_or((None, None));
    CrossSectionFontSet {
        regular: regular
            .and_then(Font::try_from_vec)
            .or_else(|| Font::try_from_bytes(SOURCE_SANS_3_REGULAR)),
        semibold: semibold
            .and_then(Font::try_from_vec)
            .or_else(|| Font::try_from_bytes(SOURCE_SANS_3_SEMIBOLD)),
    }
}

fn cross_section_font_size_px(scale: u32, kind: CrossSectionFontKind) -> f32 {
    match (scale.max(1), kind) {
        (1, CrossSectionFontKind::Regular) => 14.0,
        (1, CrossSectionFontKind::Semibold) => 16.0,
        (2, CrossSectionFontKind::Regular) => 18.0,
        (2, CrossSectionFontKind::Semibold) => 22.0,
        (level, CrossSectionFontKind::Regular) => 14.0 + (level as f32 - 1.0) * 4.0,
        (level, CrossSectionFontKind::Semibold) => 16.0 + (level as f32 - 1.0) * 4.5,
    }
}

fn cross_section_line_height_px(scale: Scale, font: &Font<'static>) -> f32 {
    let v_metrics = font.v_metrics(scale);
    (v_metrics.ascent - v_metrics.descent + v_metrics.line_gap).max(scale.y + 2.0)
}

fn glyph_rows(ch: char) -> [u8; 7] {
    match ch.to_ascii_uppercase() {
        'A' => [0x0E, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11],
        'B' => [0x1E, 0x11, 0x11, 0x1E, 0x11, 0x11, 0x1E],
        'C' => [0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E],
        'D' => [0x1E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x1E],
        'E' => [0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F],
        'F' => [0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x10],
        'G' => [0x0F, 0x10, 0x10, 0x17, 0x11, 0x11, 0x0F],
        'H' => [0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11],
        'I' => [0x0E, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E],
        'J' => [0x01, 0x01, 0x01, 0x01, 0x11, 0x11, 0x0E],
        'K' => [0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11],
        'L' => [0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F],
        'M' => [0x11, 0x1B, 0x15, 0x15, 0x11, 0x11, 0x11],
        'N' => [0x11, 0x11, 0x19, 0x15, 0x13, 0x11, 0x11],
        'O' => [0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E],
        'P' => [0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10],
        'Q' => [0x0E, 0x11, 0x11, 0x11, 0x15, 0x12, 0x0D],
        'R' => [0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11],
        'S' => [0x0F, 0x10, 0x10, 0x0E, 0x01, 0x01, 0x1E],
        'T' => [0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04],
        'U' => [0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E],
        'V' => [0x11, 0x11, 0x11, 0x11, 0x11, 0x0A, 0x04],
        'W' => [0x11, 0x11, 0x11, 0x15, 0x15, 0x15, 0x0A],
        'X' => [0x11, 0x11, 0x0A, 0x04, 0x0A, 0x11, 0x11],
        'Y' => [0x11, 0x11, 0x0A, 0x04, 0x04, 0x04, 0x04],
        'Z' => [0x1F, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1F],
        '0' => [0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E],
        '1' => [0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E],
        '2' => [0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F],
        '3' => [0x1E, 0x01, 0x01, 0x0E, 0x01, 0x01, 0x1E],
        '4' => [0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02],
        '5' => [0x1F, 0x10, 0x10, 0x1E, 0x01, 0x01, 0x1E],
        '6' => [0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E],
        '7' => [0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08],
        '8' => [0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E],
        '9' => [0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C],
        '-' => [0x00, 0x00, 0x00, 0x1F, 0x00, 0x00, 0x00],
        '.' => [0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C],
        ':' => [0x00, 0x0C, 0x0C, 0x00, 0x0C, 0x0C, 0x00],
        ',' => [0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C, 0x08],
        '/' => [0x01, 0x02, 0x04, 0x04, 0x08, 0x10, 0x10],
        '|' => [0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04],
        ' ' => [0x00; 7],
        _ => [0x1F, 0x01, 0x02, 0x04, 0x04, 0x00, 0x04],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{ScalarSection, SectionMetadata, TerrainProfile};
    use crate::vertical::VerticalAxis;
    use crate::wind::decompose_wind_grid;

    fn sample_section() -> ScalarSection {
        let axis = VerticalAxis::pressure_hpa(vec![1000.0, 900.0, 800.0, 700.0, 600.0]).unwrap();
        let mut values = Vec::new();
        for level in 0..axis.len() {
            for point in 0..6 {
                values.push(14.0 - point as f32 * 2.0 - level as f32 * 6.0);
            }
        }

        ScalarSection::new(vec![0.0, 50.0, 100.0, 150.0, 200.0, 250.0], axis, values)
            .unwrap()
            .with_metadata(
                SectionMetadata::new()
                    .titled("HRRR Temperature Cross Section")
                    .field("temperature", "C")
                    .sourced_from("nomads")
                    .valid_at("20260414 23Z F000")
                    .with_attribute("start_label", "39.10N 94.58W")
                    .with_attribute("end_label", "41.88N 87.63W")
                    .with_attribute("route_label", "KANSAS CITY TO CHICAGO"),
            )
            .with_terrain(
                TerrainProfile::from_surface_pressure_hpa(
                    vec![0.0, 50.0, 100.0, 150.0, 200.0, 250.0],
                    vec![970.0, 940.0, 910.0, 905.0, 930.0, 960.0],
                )
                .unwrap(),
            )
            .unwrap()
    }

    #[test]
    fn an_unset_source_label_draws_the_header_exactly_as_before() {
        let section = sample_section();
        let request = CrossSectionRenderRequest::default();
        assert!(request.source_label.is_none());
        let plain = render_scalar_section(&section, &request).unwrap();
        let again = render_scalar_section(&section, &CrossSectionRenderRequest::default()).unwrap();
        assert_eq!(plain.rgba(), again.rgba());
    }

    #[test]
    fn a_named_source_label_puts_ink_at_the_right_of_the_title_row() {
        let section = sample_section();
        let plain = render_scalar_section(&section, &CrossSectionRenderRequest::default()).unwrap();
        let marked = render_scalar_section(
            &section,
            &CrossSectionRenderRequest::default().with_source_label("hex-mod 0.2.3"),
        )
        .unwrap();
        assert_eq!(plain.width(), marked.width());
        assert_eq!(plain.height(), marked.height());
        assert_ne!(plain.rgba(), marked.rgba(), "the mark has to reach the pixels");
        // The mark is ink in the title row's right half, above the plot.
        let width = plain.width();
        let differing = plain
            .rgba()
            .chunks_exact(4)
            .zip(marked.rgba().chunks_exact(4))
            .enumerate()
            .filter(|(_, (a, b))| a != b)
            .map(|(index, _)| {
                let pixel = index as u32;
                (pixel % width, pixel / width)
            })
            .collect::<Vec<_>>();
        assert!(!differing.is_empty());
        let rightmost = differing.iter().map(|(x, _)| *x).max().unwrap();
        assert!(
            rightmost > width / 2,
            "the mark belongs on the right of the row, saw {rightmost} of {width}"
        );
    }

    #[test]
    fn named_colorbar_ticks_print_their_own_text_at_the_drawn_value() {
        // A fill drawn in log10 sits on -1..1 and wants 0.1, 1, 10 beside
        // the bar.  Without named ticks the bar prints the drawn values.
        let section = sample_section();
        let auto = CrossSectionRenderRequest::default().with_value_range(-1.0, 1.0);
        let named = CrossSectionRenderRequest::default()
            .with_value_range(-1.0, 1.0)
            .with_colorbar_ticks(vec![
                (-1.0, "0.1".to_string()),
                (0.0, "1".to_string()),
                (1.0, "10".to_string()),
            ]);
        let scene = ResolvedRenderScene::resolve(&section, &section, &named).unwrap();
        assert_eq!(scene.value_ticks, vec![-1.0, 0.0, 1.0]);
        assert_eq!(scene.tick_labels, vec!["0.1", "1", "10"]);
        let auto_scene = ResolvedRenderScene::resolve(&section, &section, &auto).unwrap();
        assert!(auto_scene.tick_labels.is_empty());
        assert!(auto_scene.value_ticks.contains(&0.0));
        let plain = render_scalar_section(&section, &auto).unwrap();
        let labelled = render_scalar_section(&section, &named).unwrap();
        assert_ne!(
            plain.rgba(),
            labelled.rgba(),
            "named tick text must reach the drawn bar"
        );
    }

    #[test]
    fn an_empty_source_label_is_the_same_as_naming_none() {
        let section = sample_section();
        let plain = render_scalar_section(&section, &CrossSectionRenderRequest::default()).unwrap();
        let blank = render_scalar_section(
            &section,
            &CrossSectionRenderRequest::default().with_source_label("   "),
        )
        .unwrap();
        assert_eq!(plain.rgba(), blank.rgba());
    }

    fn count_exact_pixels(
        canvas: &Canvas,
        color: Color,
        x_min: u32,
        x_max: u32,
        y_min: u32,
        y_max: u32,
    ) -> usize {
        let mut count = 0usize;
        for y in y_min..=y_max.min(canvas.height.saturating_sub(1)) {
            for x in x_min..=x_max.min(canvas.width.saturating_sub(1)) {
                let idx = ((y * canvas.width + x) * 4) as usize;
                if canvas.rgba[idx..idx + 4] == [color.r, color.g, color.b, 255] {
                    count += 1;
                }
            }
        }
        count
    }

    fn count_nontransparent_pixels(
        canvas: &Canvas,
        x_min: u32,
        x_max: u32,
        y_min: u32,
        y_max: u32,
    ) -> usize {
        let mut count = 0usize;
        for y in y_min..=y_max.min(canvas.height.saturating_sub(1)) {
            for x in x_min..=x_max.min(canvas.width.saturating_sub(1)) {
                let idx = ((y * canvas.width + x) * 4) as usize;
                if canvas.rgba[idx + 3] > 0 {
                    count += 1;
                }
            }
        }
        count
    }

    fn count_exact_pixels_excluding_row(
        canvas: &Canvas,
        color: Color,
        x_min: u32,
        x_max: u32,
        y_min: u32,
        y_max: u32,
        excluded_y: u32,
    ) -> usize {
        let mut count = 0usize;
        for y in y_min..=y_max.min(canvas.height.saturating_sub(1)) {
            if y == excluded_y {
                continue;
            }
            count += count_exact_pixels(canvas, color, x_min, x_max, y, y);
        }
        count
    }

    #[test]
    fn renderer_emits_header_text_legend_and_terrain_fill() {
        let image = render_scalar_section(
            &sample_section(),
            &CrossSectionRenderRequest {
                width: 360,
                height: 220,
                ..Default::default()
            },
        )
        .unwrap();

        assert_eq!(image.rgba().len(), (360 * 220 * 4) as usize);

        let header_pixels = image
            .rgba()
            .chunks_exact(4)
            .take((360 * 50) as usize)
            .filter(|px| px[0] < 80 && px[1] < 90 && px[2] < 100)
            .count();
        assert!(header_pixels > 40);

        let terrain_pixels = image
            .rgba()
            .chunks_exact(4)
            .filter(|px| px[0] >= 70 && px[1] >= 45 && px[1] <= 160 && px[2] <= 100)
            .count();
        assert!(terrain_pixels > 0);

        let legend_pixels = image
            .rgba()
            .chunks_exact(4)
            .enumerate()
            .filter(|(index, px)| {
                let x = (*index as u32) % 360;
                x >= 300 && px[0..3] != [246, 240, 231]
            })
            .count();
        assert!(legend_pixels > 0);
    }

    #[test]
    fn the_legend_lists_each_overlay_exactly_once_with_its_highlight_folded_in() {
        let section = sample_section();
        let ink = Color::rgb(10, 20, 30);
        let attention = Color::rgb(240, 100, 30);
        let request = CrossSectionRenderRequest {
            width: 360,
            height: 220,
            contour_overlays: vec![
                // Same ink for the highlight: the thicker line, one row.
                ScalarContourOverlayBundle::new(section.clone(), vec![1.0, 2.0, 5.0, 10.0])
                    .with_label("wa")
                    .with_units("m s-1")
                    .with_color(ink)
                    .with_highlight_color(ink)
                    .with_highlight(5.0),
                // Its own ink: STILL one row -- the row names the level and
                // its swatch carries both inks.
                ScalarContourOverlayBundle::new(section.clone(), vec![-20.0, -10.0, 0.0])
                    .with_label("T")
                    .with_units("C")
                    .with_color(ink)
                    .with_highlight_color(attention)
                    .with_highlight(-10.0),
                // No base levels at all: the highlight is the only line.
                ScalarContourOverlayBundle::new(section.clone(), Vec::new())
                    .with_label("QCLOUD")
                    .with_units("g kg-1")
                    .with_color(ink)
                    .with_highlight_color(ink)
                    .with_highlight(0.5),
            ],
            ..Default::default()
        };
        let masked = section.masked_with_terrain();
        let scene = ResolvedRenderScene::resolve(&section, &masked, &request).unwrap();
        let entries = legend_entries(&request, &scene);
        let labels: Vec<&str> = entries.iter().map(|entry| entry.label.as_str()).collect();
        // THREE overlays, THREE rows.  The delivered agent section listed
        // "wa" and "T" twice, once for the overlay and again for its
        // highlighted level; a highlight is one level of the same overlay
        // drawn heavier, not a second quantity.
        assert_eq!(labels, vec!["wa (5 m s-1)", "T (-10C)", "QCLOUD (0.5 g kg-1)"]);
        assert_eq!(entries.len(), request.contour_overlays.len());
        // The overlay whose highlight has its OWN ink shows both inks in
        // one swatch; the two drawn in a single ink keep a plain line.
        assert!(matches!(
            entries[1].symbol,
            LegendSymbol::Split(base, highlight) if base == ink && highlight == attention
        ));
        assert!(matches!(entries[0].symbol, LegendSymbol::Line(color) if color == ink));
        assert!(matches!(entries[2].symbol, LegendSymbol::Line(color) if color == ink));
        // And the section still renders with them.
        let image = render_scalar_section(&section, &request).unwrap();
        assert_eq!(image.rgba().len(), (360 * 220 * 4) as usize);
    }

    /// A planted field that varies smoothly in both directions must render
    /// without visible steps: no two neighbouring fill pixels may jump by
    /// more than a couple of levels of ink.
    ///
    /// WHAT BREAKAGE THIS PREVENTS (gate law): the delivered
    /// winter sections were drawn as per-sample blocks, because one NaN
    /// corner dropped a whole sample quad and eight palette bands stepped
    /// the rest.  A smooth field is the instrument that catches both.
    #[test]
    fn a_smooth_field_renders_with_no_visible_steps() {
        let n_points = 60usize;
        let n_levels = 40usize;
        let distances: Vec<f64> = (0..n_points).map(|i| i as f64 * 2.0).collect();
        let levels: Vec<f64> = (0..n_levels).map(|k| k as f64 * 100.0).collect();
        let axis = VerticalAxis::height_meters(levels).unwrap();
        let mut values = Vec::with_capacity(n_points * n_levels);
        for k in 0..n_levels {
            for i in 0..n_points {
                let x = i as f32 / (n_points - 1) as f32;
                let z = k as f32 / (n_levels - 1) as f32;
                values.push(10.0 * x + 6.0 * z);
            }
        }
        let section = ScalarSection::new(distances, axis, values)
            .unwrap()
            .with_metadata(SectionMetadata::new().field("planted", "unit"));
        let request = CrossSectionRenderRequest {
            width: 900,
            height: 460,
            show_grid: false,
            show_axes: false,
            isotherms_c: Vec::new(),
            highlight_isotherm_c: None,
            palette: CrossSectionPalette::CloudWater.sampled_colors(24),
            ..Default::default()
        };
        let plot = PlotRect::from_request(&request, request.resolved_type_scale()).unwrap();
        let image = render_scalar_section(&section, &request).unwrap();
        let rgba = image.rgba();
        let at = |x: u32, y: u32| -> [i32; 3] {
            let idx = ((y * image.width() + x) * 4) as usize;
            [rgba[idx] as i32, rgba[idx + 1] as i32, rgba[idx + 2] as i32]
        };
        // Inside the plot, away from its own border, neighbouring pixels of
        // a smooth field differ by a few units of ink at most.  A block
        // edge is a jump of tens.
        let mut worst = 0i32;
        for y in (plot.y + 4)..(plot.bottom() - 4) {
            for x in (plot.x + 4)..(plot.right() - 4) {
                let here = at(x, y);
                for other in [at(x + 1, y), at(x, y + 1)] {
                    for channel in 0..3 {
                        worst = worst.max((here[channel] - other[channel]).abs());
                    }
                }
            }
        }
        assert!(worst <= 6, "neighbouring fill pixels jump by {worst}");
    }

    /// The same planted field with a hole punched in it: the edge of the
    /// hole fades, it does not fall off a cliff of whole sample cells.
    #[test]
    fn a_partly_missing_quad_fades_instead_of_dropping_whole_cells() {
        let n_points = 40usize;
        let n_levels = 24usize;
        let distances: Vec<f64> = (0..n_points).map(|i| i as f64 * 2.0).collect();
        let levels: Vec<f64> = (0..n_levels).map(|k| k as f64 * 100.0).collect();
        let axis = VerticalAxis::height_meters(levels).unwrap();
        let mut values = vec![5.0f32; n_points * n_levels];
        for k in 0..n_levels {
            for i in 0..n_points {
                if k > n_levels / 2 {
                    values[k * n_points + i] = f32::NAN;
                }
            }
        }
        let section = ScalarSection::new(distances, axis, values).unwrap();
        // The strict sampler drops the entire quad below the boundary; the
        // coverage sampler keeps it and reports how much of it was there.
        let boundary_axis = (n_levels / 2) as f64 * 100.0 + 50.0;
        assert!(section.bilinear_sample(20.0, boundary_axis).is_none());
        let (value, coverage) = section
            .bilinear_sample_coverage(20.0, boundary_axis)
            .expect("the finite half of the quad still answers");
        assert!((value - 5.0).abs() < 1e-5, "{value}");
        assert!(
            coverage > 0.0 && coverage < 1.0,
            "a half-finite quad reports partial coverage, got {coverage}"
        );
    }

    /// Type is a share of the image, not a fixed pixel count: a 2400-pixel
    /// sheet gets 2400-pixel type.
    #[test]
    fn type_is_scaled_to_the_image_and_clears_the_readability_bars() {
        for width in [1400u32, 1800, 2400, 4000] {
            let request = CrossSectionRenderRequest {
                width,
                height: width / 2,
                ..Default::default()
            };
            let scale = request.resolved_type_scale();
            let title_px = cross_section_font_size_px(2, CrossSectionFontKind::Semibold) * scale;
            let label_px = cross_section_font_size_px(1, CrossSectionFontKind::Regular) * scale;
            assert!(
                title_px >= width as f32 * 0.014,
                "title {title_px} px is under 1.4 % of {width}"
            );
            assert!(
                label_px >= width as f32 * 0.009,
                "labels {label_px} px are under 0.9 % of {width}"
            );
        }
        // A caller may still pin it.
        assert_eq!(
            CrossSectionRenderRequest::default()
                .with_type_scale(1.0)
                .resolved_type_scale(),
            1.0
        );
    }

    /// The default margins grow with the type; a caller's own margins do
    /// not, because those were measured for a purpose.
    #[test]
    fn the_default_margins_scale_with_the_type_and_a_callers_do_not() {
        let request = CrossSectionRenderRequest {
            width: 2800,
            height: 1400,
            ..Default::default()
        };
        let scale = request.resolved_type_scale();
        assert!((scale - 2.0).abs() < 1e-6, "{scale}");
        let margins = effective_margins(&request, scale);
        assert_eq!(margins.left, Insets::default().left * 2);
        assert_eq!(margins.top, Insets::default().top * 2);
        let pinned = request.clone().with_margins(Insets {
            left: 40,
            right: 40,
            top: 40,
            bottom: 40,
        });
        assert_eq!(effective_margins(&pinned, scale).left, 40);
    }

    #[test]
    fn a_colourbar_tick_keeps_the_places_its_own_magnitude_needs() {
        // The rejected control section's cloud-water bar ran 0 to about
        // 0.6 g kg-1 and every tick printed "0".
        assert_eq!(format_scalar_value(0.0), "0");
        assert_eq!(format_scalar_value(0.02), "0.02");
        assert_eq!(format_scalar_value(0.06), "0.06");
        assert_eq!(format_scalar_value(0.5), "0.5");
        assert_eq!(format_scalar_value(0.001), "0.001");
        assert_eq!(format_scalar_value(-0.25), "-0.25");
        // Everything at or above one is spelled the way it always was.
        assert_eq!(format_scalar_value(-10.0), "-10");
        assert_eq!(format_scalar_value(2.5), "2.5");
        assert_eq!(format_scalar_value(140.0), "140");
        // A whole colourbar of small ticks reads as distinct numbers.
        let labels: Vec<String> = [0.0f32, 0.02, 0.04, 0.06]
            .iter()
            .map(|v| format_scalar_value(*v))
            .collect();
        let unique: std::collections::BTreeSet<&String> = labels.iter().collect();
        assert_eq!(unique.len(), labels.len(), "{labels:?}");
    }

    #[test]
    fn a_degenerate_axis_range_ends_its_tick_loop_instead_of_eating_the_machine() {
        // A field with no data at all comes back with min == max == a huge
        // magnitude, where adding the step does not move the accumulator.
        let huge = f64::from(f32::MIN);
        let ticks = ranged_ticks(huge, huge, 1.0);
        assert!(ticks.len() <= MAX_AXIS_TICKS + 2, "{}", ticks.len());
        let ticks = nice_value_ticks(f32::MIN, f32::MIN, 7);
        assert!(ticks.len() <= MAX_AXIS_TICKS + 2, "{}", ticks.len());
        // A sane range is unchanged.
        assert_eq!(ranged_ticks(0.0, 10.0, 2.0), vec![0.0, 2.0, 4.0, 6.0, 8.0, 10.0]);
        assert_eq!(nice_value_ticks(0.0, 7.0, 7), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    }

    #[test]
    fn a_scale_of_ten_thousandths_keeps_its_ticks_apart() {
        // The rejected control sections ran to a fraction of a gram per
        // kilogram and their colourbar carried ONE label, reading zero:
        // every tick had been rounded to the nearest thousandth.
        let ticks = nice_value_ticks(0.0, 0.0009, 7);
        assert!(ticks.len() >= 4, "{ticks:?}");
        let labels: Vec<String> = ticks.iter().map(|t| format_scalar_value(*t)).collect();
        let unique: std::collections::BTreeSet<&String> = labels.iter().collect();
        assert_eq!(unique.len(), labels.len(), "{labels:?}");
        assert!(labels.iter().filter(|l| l.as_str() == "0").count() <= 1, "{labels:?}");
        // And a scale of whole numbers is spelled the way it always was.
        assert_eq!(
            nice_value_ticks(0.0, 0.6, 7)
                .iter()
                .map(|t| format_scalar_value(*t))
                .collect::<Vec<_>>(),
            vec!["0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6"]
        );
    }

    #[test]
    fn a_section_bar_between_whole_numbers_reads_its_own_values() {
        // Measured on a real theta section cut to 6 km: 305.25 to 320.88 K
        // on a 2.5 K step, drawn as 305, 308, 310, 312, 315, 318, 320, 321.
        let (lo, hi) = (305.25204f32, 320.88f32);
        let ticks = nice_value_ticks(lo, hi, 7);
        let labels = value_tick_labels(&ticks, Some(nice_value_tick_step(lo, hi, 7)));
        assert_eq!(
            labels,
            ["305.3", "307.5", "310", "312.5", "315", "317.5", "320", "320.9"]
        );
        // A quarter step below 100 printed 1.25 as 1.2 and 1.75 as 1.8.
        assert_eq!(
            value_tick_labels(&[1.0, 1.25, 1.5, 1.75, 2.0], None),
            ["1", "1.25", "1.5", "1.75", "2"]
        );
    }

    #[test]
    fn a_section_bar_on_whole_steps_is_spelled_as_before() {
        for (lo, hi) in [
            (305.25204f32, 316.4611f32),
            (284.48618, 305.13718),
            (0.0, 0.6),
            (0.0, 0.0009),
            (-30.0, 12.0),
        ] {
            let ticks = nice_value_ticks(lo, hi, 7);
            let usual: Vec<String> = ticks.iter().map(|tick| format_scalar_value(*tick)).collect();
            assert_eq!(
                value_tick_labels(&ticks, Some(nice_value_tick_step(lo, hi, 7))),
                usual,
                "{lo}..{hi}"
            );
        }
        for style in crate::style::ALL_CROSS_SECTION_PRODUCTS {
            let ticks = style.default_value_ticks();
            let usual: Vec<String> = ticks.iter().map(|tick| format_scalar_value(*tick)).collect();
            assert_eq!(value_tick_labels(ticks, None), usual, "{style:?}");
        }
    }

    #[test]
    fn renderer_draws_highlight_isotherm_overlay() {
        let image = render_scalar_section(
            &sample_section(),
            &CrossSectionRenderRequest {
                width: 360,
                height: 220,
                highlight_isotherm_c: Some(0.0),
                isotherms_c: vec![-20.0, -10.0, 0.0],
                ..Default::default()
            },
        )
        .unwrap();

        // The highlight ink is the request's own, so the assertion moves
        // with it: it is RED now, because magenta is the vertical-velocity
        // ink and one colour has to mean one thing on a sheet that carries
        // both.
        let ink = CrossSectionRenderRequest::default().highlight_isotherm_color;
        assert_eq!((ink.r, ink.g, ink.b), (228, 26, 28));
        let highlight_pixels = image
            .rgba()
            .chunks_exact(4)
            .filter(|px| px[0] == ink.r && px[1] == ink.g && px[2] == ink.b)
            .count();
        assert!(highlight_pixels > 20, "{highlight_pixels}");
    }

    #[test]
    fn request_builders_override_ticks_and_colorbar_label() {
        let request = CrossSectionRenderRequest::default()
            .with_value_ticks(vec![-30.0, -10.0, 0.0, 10.0])
            .with_colorbar_label("Temp C")
            .with_isotherms(vec![-15.0, 0.0], Some(0.0));

        assert_eq!(request.value_ticks, vec![-30.0, -10.0, 0.0, 10.0]);
        assert_eq!(request.colorbar_label.as_deref(), Some("Temp C"));
        assert_eq!(request.isotherms_c, vec![-15.0, 0.0]);
        assert_eq!(request.highlight_isotherm_c, Some(0.0));
    }

    #[test]
    fn wind_vector_geometry_uses_section_relative_angle() {
        let style = WindOverlayStyle::default();

        let up_right = resolve_section_wind_vector_geometry((100.0, 60.0), 12.0, 12.0, 17.0, style)
            .expect("nonzero wind should produce drawable geometry");
        assert!(up_right.end.0 > 100.0);
        assert!(up_right.end.1 < 60.0);

        let down_left =
            resolve_section_wind_vector_geometry((100.0, 60.0), -12.0, -12.0, 17.0, style)
                .expect("nonzero wind should produce drawable geometry");
        assert!(down_left.end.0 < 100.0);
        assert!(down_left.end.1 > 60.0);
    }

    #[test]
    fn renderer_draws_section_relative_wind_vectors() {
        let section = sample_section();
        let wind = decompose_wind_grid(
            &[
                10.0, 10.0, 10.0, 10.0, 10.0, 10.0, //
                12.0, 12.0, 12.0, 12.0, 12.0, 12.0, //
                14.0, 14.0, 14.0, 14.0, 14.0, 14.0, //
                16.0, 16.0, 16.0, 16.0, 16.0, 16.0, //
                18.0, 18.0, 18.0, 18.0, 18.0, 18.0, //
            ],
            &[
                2.0, 2.0, 2.0, 2.0, 2.0, 2.0, //
                -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, //
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, //
                -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, //
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
            ],
            section.n_levels(),
            section.n_points(),
            &[90.0; 6],
        )
        .unwrap();

        let image = render_scalar_section(
            &section,
            &CrossSectionRenderRequest {
                width: 360,
                height: 220,
                wind_overlay: Some(
                    WindOverlayBundle::new(
                        wind,
                        WindOverlayStyle {
                            stride_points: 2,
                            stride_levels: 1,
                            min_speed_ms: 1.0,
                            color: Color::rgb(28, 34, 43),
                            ..Default::default()
                        },
                    )
                    .with_label("Section Relative Wind"),
                ),
                ..Default::default()
            },
        )
        .unwrap();

        let vector_pixels = image
            .rgba()
            .chunks_exact(4)
            .filter(|px| px[0] == 28 && px[1] == 34 && px[2] == 43)
            .count();
        assert!(vector_pixels > 30);
    }

    #[test]
    fn wind_barb_point_selection_keeps_target_columns_across_route_lengths() {
        assert_eq!(
            wind_barb_point_indices(51, 10, 6),
            vec![0, 6, 11, 17, 22, 28, 33, 39, 44, 50]
        );
        assert_eq!(wind_barb_point_indices(101, 10, 6).len(), 10);
        assert_eq!(wind_barb_point_indices(6, 10, 6), vec![0, 1, 2, 3, 4, 5]);
        assert_eq!(wind_barb_point_indices(10, 0, 4), vec![0, 4, 8]);
    }

    #[test]
    fn wind_vector_arrowheads_flip_with_along_section_sign() {
        let mut positive_canvas = Canvas::new(80, 40, Color::TRANSPARENT, Color::TRANSPARENT);
        let mut negative_canvas = Canvas::new(80, 40, Color::TRANSPARENT, Color::TRANSPARENT);
        let plot = PlotRect {
            x: 0,
            y: 0,
            width: 80,
            height: 40,
        };
        let style = WindOverlayStyle {
            min_speed_ms: 0.0,
            max_speed_ms: 20.0,
            base_length_px: 10.0,
            max_length_px: 10.0,
            arrow_head_px: 4.0,
            line_width: 1,
            color: Color::rgb(230, 30, 30),
            ..Default::default()
        };

        draw_section_wind_vector(
            &mut positive_canvas,
            (40.0, 20.0),
            10.0,
            0.0,
            20.0,
            style,
            &plot,
        );
        draw_section_wind_vector(
            &mut negative_canvas,
            (40.0, 20.0),
            -10.0,
            0.0,
            20.0,
            style,
            &plot,
        );

        let positive_right_head =
            count_exact_pixels_excluding_row(&positive_canvas, style.color, 41, 45, 16, 24, 20);
        let positive_left_head =
            count_exact_pixels_excluding_row(&positive_canvas, style.color, 35, 39, 16, 24, 20);
        let negative_right_head =
            count_exact_pixels_excluding_row(&negative_canvas, style.color, 41, 45, 16, 24, 20);
        let negative_left_head =
            count_exact_pixels_excluding_row(&negative_canvas, style.color, 35, 39, 16, 24, 20);

        assert!(positive_right_head > 0);
        assert_eq!(positive_left_head, 0);
        assert_eq!(negative_right_head, 0);
        assert!(negative_left_head > 0);
    }

    #[test]
    fn canvas_text_helper_renders_multiline_text_with_shadow_offset() {
        let mut canvas = Canvas::new(48, 28, Color::TRANSPARENT, Color::TRANSPARENT);
        let text_color = Color::rgb(245, 245, 245);
        let shadow_color = Color::rgb(12, 18, 24);

        canvas.draw_text(2, 2, "A\nA", text_color, 1, Some(shadow_color));

        let top_line_pixels = count_nontransparent_pixels(&canvas, 0, 47, 0, 13);
        let bottom_line_pixels = count_nontransparent_pixels(&canvas, 0, 47, 14, 27);
        let shadow_pixels = count_nontransparent_pixels(&canvas, 5, 25, 5, 25);

        assert!(top_line_pixels > 0);
        assert!(bottom_line_pixels > 0);
        assert!(shadow_pixels > 0);
    }

    #[test]
    fn badge_rect_uses_text_width_padding_and_scale() {
        let (x, y, width, height) = badge_rect(-3, 5, "AB", 2);

        assert_eq!(x, 0);
        assert_eq!(y, 5);
        assert_eq!(width, measure_text_width("AB", 2, 1.0) + 16);
        assert_eq!(height, text_line_height(2, 1.0) + 4);
        assert_eq!(measure_badge_width("AB", 2), width);
    }
}
