use crate::RustwxRenderError;
use crate::colormap::{LegendControls, LegendMode, LevelDensity, RenderDensity};
use crate::presentation::{LineworkRole, PolygonRole, ProductVisualMode};
use crate::projected_map::ProjectedMap;
use crate::projection::ProjectionSpec;
use rustwx_core as core;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct GridShape {
    pub nx: usize,
    pub ny: usize,
}

impl GridShape {
    pub fn new(nx: usize, ny: usize) -> Result<Self, RustwxRenderError> {
        if nx == 0 || ny == 0 {
            return Err(RustwxRenderError::InvalidGridShape { nx, ny });
        }
        Ok(Self { nx, ny })
    }

    pub fn len(self) -> usize {
        self.nx * self.ny
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LatLonGrid {
    pub shape: GridShape,
    pub lat_deg: Vec<f32>,
    pub lon_deg: Vec<f32>,
}

impl LatLonGrid {
    pub fn new(
        shape: GridShape,
        lat_deg: Vec<f32>,
        lon_deg: Vec<f32>,
    ) -> Result<Self, RustwxRenderError> {
        if lat_deg.len() != shape.len() || lon_deg.len() != shape.len() {
            return Err(RustwxRenderError::InvalidGridShape {
                nx: shape.nx,
                ny: shape.ny,
            });
        }
        Ok(Self {
            shape,
            lat_deg,
            lon_deg,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProductKey {
    Named(String),
}

impl ProductKey {
    pub fn named<S: Into<String>>(name: S) -> Self {
        Self::Named(name.into())
    }

    pub fn as_named(&self) -> Option<&str> {
        match self {
            Self::Named(name) => Some(name.as_str()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Field2D {
    pub product: ProductKey,
    pub units: String,
    pub grid: LatLonGrid,
    pub values: Vec<f32>,
}

impl Field2D {
    pub fn new<S: Into<String>>(
        product: ProductKey,
        units: S,
        grid: LatLonGrid,
        values: Vec<f32>,
    ) -> Result<Self, RustwxRenderError> {
        if values.len() != grid.shape.len() {
            return Err(RustwxRenderError::InvalidGridShape {
                nx: grid.shape.nx,
                ny: grid.shape.ny,
            });
        }
        Ok(Self {
            product,
            units: units.into(),
            grid,
            values,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RgbaGridField {
    pub grid: LatLonGrid,
    pub pixels: Vec<Color>,
}

impl RgbaGridField {
    pub fn new(grid: LatLonGrid, pixels: Vec<Color>) -> Result<Self, RustwxRenderError> {
        if pixels.len() != grid.shape.len() {
            return Err(RustwxRenderError::InvalidGridShape {
                nx: grid.shape.nx,
                ny: grid.shape.ny,
            });
        }
        Ok(Self { grid, pixels })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Color {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

impl Color {
    pub const TRANSPARENT: Self = Self {
        r: 0,
        g: 0,
        b: 0,
        a: 0,
    };
    pub const WHITE: Self = Self {
        r: 255,
        g: 255,
        b: 255,
        a: 255,
    };
    pub const BLACK: Self = Self {
        r: 0,
        g: 0,
        b: 0,
        a: 255,
    };

    pub const fn rgba(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum DomainFrameSource {
    #[default]
    ProjectedGrid,
    RasterAlpha,
    MapViewport,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DomainFrame {
    pub inset_px: u32,
    pub outline_color: Color,
    pub outline_width: u32,
    pub clear_outside: bool,
    pub legend_follows_frame: bool,
    pub chrome_follows_frame: bool,
    #[serde(default)]
    pub source: DomainFrameSource,
}

impl DomainFrame {
    pub fn map_viewport_default() -> Self {
        Self {
            inset_px: 5,
            outline_color: Color::BLACK,
            outline_width: 3,
            clear_outside: true,
            legend_follows_frame: true,
            chrome_follows_frame: true,
            source: DomainFrameSource::MapViewport,
        }
    }

    pub fn model_data_default() -> Self {
        Self {
            source: DomainFrameSource::ProjectedGrid,
            ..Self::map_viewport_default()
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExtendMode {
    Neither,
    Min,
    Max,
    Both,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProductMaturity {
    Operational,
    Experimental,
    Proof,
}

impl ProductMaturity {
    pub fn is_non_operational(self) -> bool {
        !matches!(self, Self::Operational)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProductSemanticFlag {
    Proxy,
    Composite,
    Alias,
    ProofOriented,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProductSemantics {
    pub maturity: ProductMaturity,
    pub flags: Vec<ProductSemanticFlag>,
}

impl Default for ProductSemantics {
    fn default() -> Self {
        Self::operational()
    }
}

impl ProductSemantics {
    pub fn operational() -> Self {
        Self {
            maturity: ProductMaturity::Operational,
            flags: Vec::new(),
        }
    }

    pub fn experimental() -> Self {
        Self {
            maturity: ProductMaturity::Experimental,
            flags: Vec::new(),
        }
    }

    pub fn proof() -> Self {
        Self {
            maturity: ProductMaturity::Proof,
            flags: vec![ProductSemanticFlag::ProofOriented],
        }
    }

    pub fn with_flag(mut self, flag: ProductSemanticFlag) -> Self {
        if !self.flags.contains(&flag) {
            self.flags.push(flag);
        }
        self
    }

    pub fn has_flag(&self, flag: ProductSemanticFlag) -> bool {
        self.flags.contains(&flag)
    }
}

impl From<core::ProductMaturity> for ProductMaturity {
    fn from(value: core::ProductMaturity) -> Self {
        match value {
            core::ProductMaturity::Operational => Self::Operational,
            core::ProductMaturity::Experimental => Self::Experimental,
            core::ProductMaturity::Proof => Self::Proof,
        }
    }
}

impl From<ProductMaturity> for core::ProductMaturity {
    fn from(value: ProductMaturity) -> Self {
        match value {
            ProductMaturity::Operational => Self::Operational,
            ProductMaturity::Experimental => Self::Experimental,
            ProductMaturity::Proof => Self::Proof,
        }
    }
}

impl From<core::ProductSemanticFlag> for ProductSemanticFlag {
    fn from(value: core::ProductSemanticFlag) -> Self {
        match value {
            core::ProductSemanticFlag::Proxy => Self::Proxy,
            core::ProductSemanticFlag::Composite => Self::Composite,
            core::ProductSemanticFlag::Alias => Self::Alias,
            core::ProductSemanticFlag::ProofOriented => Self::ProofOriented,
        }
    }
}

impl From<ProductSemanticFlag> for core::ProductSemanticFlag {
    fn from(value: ProductSemanticFlag) -> Self {
        match value {
            ProductSemanticFlag::Proxy => Self::Proxy,
            ProductSemanticFlag::Composite => Self::Composite,
            ProductSemanticFlag::Alias => Self::Alias,
            ProductSemanticFlag::ProofOriented => Self::ProofOriented,
        }
    }
}

impl From<core::ProductProvenance> for ProductSemantics {
    fn from(value: core::ProductProvenance) -> Self {
        let mut semantics = ProductSemantics {
            maturity: value.maturity.into(),
            flags: value.flags.into_iter().map(Into::into).collect(),
        };
        semantics.flags.sort_by_key(|flag| match flag {
            ProductSemanticFlag::Proxy => 0,
            ProductSemanticFlag::Composite => 1,
            ProductSemanticFlag::Alias => 2,
            ProductSemanticFlag::ProofOriented => 3,
        });
        semantics.flags.dedup();
        semantics
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DiscreteColorScale {
    pub levels: Vec<f64>,
    pub colors: Vec<Color>,
    pub extend: ExtendMode,
    pub mask_below: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum RasterSampleMode {
    #[default]
    Linear,
    Nearest,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ColorScale {
    Weather(crate::weather::WeatherPreset),
    Discrete(DiscreteColorScale),
}

impl ColorScale {
    pub fn resolved_discrete(&self) -> DiscreteColorScale {
        match self {
            Self::Weather(preset) => preset.scale(),
            Self::Discrete(scale) => scale.clone(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProjectedExtent {
    pub x_min: f64,
    pub x_max: f64,
    pub y_min: f64,
    pub y_max: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProjectedDomain {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub extent: ProjectedExtent,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InverseRasterProjection {
    pub projection: ProjectionSpec,
    pub reference_latitude_deg: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reference_longitude_deg: Option<f64>,
    #[serde(default)]
    pub clip_bounds: Option<GeographicClipBounds>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct GeographicClipBounds {
    pub west_deg: f64,
    pub east_deg: f64,
    pub south_deg: f64,
    pub north_deg: f64,
}

impl GeographicClipBounds {
    pub fn new(west_deg: f64, east_deg: f64, south_deg: f64, north_deg: f64) -> Self {
        Self {
            west_deg,
            east_deg,
            south_deg: south_deg.min(north_deg),
            north_deg: south_deg.max(north_deg),
        }
    }

    pub fn contains(self, lat_deg: f64, lon_deg: f64) -> bool {
        if !lat_deg.is_finite() || !lon_deg.is_finite() {
            return false;
        }
        if lat_deg < self.south_deg || lat_deg > self.north_deg {
            return false;
        }
        if self.longitude_span_deg() >= 359.0 {
            return true;
        }
        let west = normalize_clip_longitude_deg(self.west_deg);
        let east = normalize_clip_longitude_deg(self.east_deg);
        let lon = normalize_clip_longitude_deg(lon_deg);
        if west <= east {
            lon >= west && lon <= east
        } else {
            lon >= west || lon <= east
        }
    }

    fn longitude_span_deg(self) -> f64 {
        let raw_span = (self.east_deg - self.west_deg).abs();
        if raw_span >= 359.0 {
            return raw_span.min(360.0);
        }

        let west = normalize_clip_longitude_deg(self.west_deg);
        let east = normalize_clip_longitude_deg(self.east_deg);
        if west <= east {
            east - west
        } else {
            east + 360.0 - west
        }
    }
}

fn normalize_clip_longitude_deg(lon_deg: f64) -> f64 {
    let mut lon = lon_deg % 360.0;
    if lon > 180.0 {
        lon -= 360.0;
    } else if lon <= -180.0 {
        lon += 360.0;
    }
    lon
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProjectedLineOverlay {
    pub points: Vec<(f64, f64)>,
    pub color: Color,
    pub width: u32,
    #[serde(default)]
    pub role: LineworkRole,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ProjectedMarkerShape {
    Circle,
    #[default]
    Plus,
    Cross,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProjectedPointOverlay {
    pub x: f64,
    pub y: f64,
    pub color: Color,
    pub radius_px: u32,
    pub width_px: u32,
    #[serde(default)]
    pub shape: ProjectedMarkerShape,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ProjectedLabelPlacement {
    Center,
    Left,
    Right,
    Above,
    Below,
    AboveLeft,
    #[default]
    AboveRight,
    BelowLeft,
    BelowRight,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ProjectedPlaceLabelPriority {
    #[default]
    Primary,
    Auxiliary,
    Micro,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProjectedPlaceLabelStyle {
    pub marker_radius_px: u32,
    pub marker_fill: Color,
    pub marker_outline: Color,
    pub marker_outline_width: u32,
    pub label_color: Color,
    pub label_halo: Color,
    pub label_halo_width_px: u32,
    pub label_scale: u32,
    pub label_offset_x_px: i32,
    pub label_offset_y_px: i32,
    #[serde(default)]
    pub label_placement: ProjectedLabelPlacement,
    #[serde(default)]
    pub label_bold: bool,
}

impl Default for ProjectedPlaceLabelStyle {
    fn default() -> Self {
        Self {
            marker_radius_px: 3,
            marker_fill: Color::rgba(255, 255, 255, 235),
            marker_outline: Color::rgba(24, 28, 34, 240),
            marker_outline_width: 1,
            label_color: Color::rgba(24, 28, 34, 255),
            label_halo: Color::rgba(255, 255, 255, 235),
            label_halo_width_px: 2,
            label_scale: 1,
            label_offset_x_px: 6,
            label_offset_y_px: -2,
            label_placement: ProjectedLabelPlacement::AboveRight,
            label_bold: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProjectedPlaceLabel {
    pub x: f64,
    pub y: f64,
    #[serde(default)]
    pub label: Option<String>,
    #[serde(default)]
    pub priority: ProjectedPlaceLabelPriority,
    #[serde(default)]
    pub style: ProjectedPlaceLabelStyle,
}

impl ProjectedPlaceLabel {
    pub fn new(x: f64, y: f64) -> Self {
        Self {
            x,
            y,
            label: None,
            priority: ProjectedPlaceLabelPriority::Primary,
            style: ProjectedPlaceLabelStyle::default(),
        }
    }

    pub fn with_label<S: Into<String>>(mut self, label: S) -> Self {
        self.label = Some(label.into());
        self
    }

    pub fn with_priority(mut self, priority: ProjectedPlaceLabelPriority) -> Self {
        self.priority = priority;
        self
    }

    pub fn with_style(mut self, style: ProjectedPlaceLabelStyle) -> Self {
        self.style = style;
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChromeScale {
    Fixed(f32),
    Auto {
        base_width: u32,
        base_height: u32,
        min: f32,
        max: f32,
    },
}

impl Default for ChromeScale {
    fn default() -> Self {
        Self::Auto {
            base_width: 1200,
            base_height: 900,
            min: 1.0,
            max: 3.0,
        }
    }
}

/// A filled polygon in projected map coordinates. First ring is the outer
/// boundary; additional rings punch holes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProjectedPolygonFill {
    pub rings: Vec<Vec<(f64, f64)>>,
    pub color: Color,
    #[serde(default)]
    pub role: PolygonRole,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContourLayer {
    pub data: Vec<f32>,
    pub levels: Vec<f64>,
    pub color: Color,
    pub width: u32,
    pub labels: bool,
    pub show_extrema: bool,
    #[serde(default)]
    pub pattern: ContourLinePattern,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub major_every: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub major_width: Option<u32>,
}

/// A translucent banded fill drawn beneath a contour layer: a value at or
/// above `levels[i]` (and below `levels[i + 1]`, if any) takes `colors[i]`;
/// below `levels[0]`, NaN and non-finite values are left clear.  The colours
/// carry their own alpha so the field underneath shows through.  It is
/// drawn over the data raster and under linework and contour lines, so the
/// outlines stay on top (UH tracks shaded over reflectivity, the shading
/// inside the UH contours, 2026-11-05).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContourFillLayer {
    pub data: Vec<f32>,
    pub levels: Vec<f64>,
    pub colors: Vec<Color>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ContourLinePattern {
    Solid,
    Dashed,
}

impl Default for ContourLinePattern {
    fn default() -> Self {
        Self::Solid
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContourStyle {
    pub color: Color,
    pub width: u32,
    pub labels: bool,
    pub show_extrema: bool,
    #[serde(default)]
    pub pattern: ContourLinePattern,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub major_every: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub major_width: Option<u32>,
}

impl Default for ContourStyle {
    fn default() -> Self {
        Self {
            color: Color::BLACK,
            width: 1,
            labels: false,
            show_extrema: false,
            pattern: ContourLinePattern::Solid,
            major_every: None,
            major_width: None,
        }
    }
}

impl ContourLayer {
    pub fn from_field(field: &Field2D, levels: Vec<f64>, style: ContourStyle) -> Self {
        Self {
            data: field.values.clone(),
            levels,
            color: style.color,
            width: style.width,
            labels: style.labels,
            show_extrema: style.show_extrema,
            pattern: style.pattern,
            major_every: style.major_every,
            major_width: style.major_width,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WindBarbLayer {
    /// Direction-only arrows, in the same earth-relative convention as winds.
    #[serde(default)]
    pub direction_arrows: bool,
    pub u: Vec<f32>,
    pub v: Vec<f32>,
    pub stride_x: usize,
    pub stride_y: usize,
    pub spacing_px: f64,
    pub color: Color,
    pub halo_color: Color,
    pub halo_width: u32,
    pub width: u32,
    pub length_px: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct WindBarbStyle {
    pub stride_x: usize,
    pub stride_y: usize,
    pub spacing_px: f64,
    pub color: Color,
    pub halo_color: Color,
    pub halo_width: u32,
    pub width: u32,
    pub length_px: f64,
}

impl Default for WindBarbStyle {
    fn default() -> Self {
        Self {
            stride_x: 8,
            stride_y: 8,
            spacing_px: 0.0,
            color: Color::BLACK,
            halo_color: Color::WHITE,
            halo_width: 0,
            width: 1,
            length_px: 20.0,
        }
    }
}

impl WindBarbLayer {
    pub fn from_fields(u: &Field2D, v: &Field2D, style: WindBarbStyle) -> Self {
        Self {
            direction_arrows: false,
            u: u.values.clone(),
            v: v.values.clone(),
            stride_x: style.stride_x.max(1),
            stride_y: style.stride_y.max(1),
            spacing_px: style.spacing_px.max(0.0),
            color: style.color,
            halo_color: style.halo_color,
            halo_width: style.halo_width,
            width: style.width,
            length_px: style.length_px,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WindStreamlineLayer {
    pub u: Vec<f32>,
    pub v: Vec<f32>,
    pub stride_x: usize,
    pub stride_y: usize,
    pub color: Color,
    pub width: u32,
    pub max_steps: usize,
    pub step_cells: f64,
    pub min_speed: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct WindStreamlineStyle {
    pub stride_x: usize,
    pub stride_y: usize,
    pub color: Color,
    pub width: u32,
    pub max_steps: usize,
    pub step_cells: f64,
    pub min_speed: f64,
}

impl Default for WindStreamlineStyle {
    fn default() -> Self {
        Self {
            stride_x: 12,
            stride_y: 12,
            color: Color::rgba(20, 26, 34, 120),
            width: 1,
            max_steps: 18,
            step_cells: 0.75,
            min_speed: 2.5,
        }
    }
}

impl WindStreamlineLayer {
    pub fn from_fields(u: &Field2D, v: &Field2D, style: WindStreamlineStyle) -> Self {
        Self {
            u: u.values.clone(),
            v: v.values.clone(),
            stride_x: style.stride_x.max(1),
            stride_y: style.stride_y.max(1),
            color: style.color,
            width: style.width.max(1),
            max_steps: style.max_steps.max(1),
            step_cells: style.step_cells.max(0.05),
            min_speed: style.min_speed.max(0.0),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MapRenderRequest {
    pub field: Field2D,
    #[serde(default)]
    pub rgba_grid: Option<RgbaGridField>,
    pub product_metadata: Option<core::ProductKeyMetadata>,
    pub width: u32,
    pub height: u32,
    pub scale: ColorScale,
    pub background: Color,
    pub colorbar: bool,
    pub title: Option<String>,
    pub subtitle_left: Option<String>,
    pub subtitle_center: Option<String>,
    pub subtitle_right: Option<String>,
    pub cbar_tick_step: Option<f64>,
    #[serde(default)]
    pub render_density: RenderDensity,
    #[serde(default)]
    pub legend: LegendControls,
    #[serde(default)]
    pub chrome_scale: ChromeScale,
    #[serde(default = "default_supersample_factor")]
    pub supersample_factor: u32,
    #[serde(default = "default_supersample_sharpen")]
    pub supersample_sharpen: bool,
    #[serde(default)]
    pub visual_mode: ProductVisualMode,
    #[serde(default)]
    pub raster_sample_mode: RasterSampleMode,
    #[serde(default)]
    pub domain_frame: Option<DomainFrame>,
    pub projected_domain: Option<ProjectedDomain>,
    /// Filled polygon basemap layers (ocean/land/lakes). Drawn BEFORE the
    /// data raster; ordering within the list is bottom-to-top.
    #[serde(default)]
    pub projected_polygons: Vec<ProjectedPolygonFill>,
    /// Dynamic projected fill polygons drawn during the variable-data pass.
    #[serde(default)]
    pub projected_data_polygons: Vec<ProjectedPolygonFill>,
    /// An unstructured mesh drawn cell by cell, in the same projected space
    /// as `projected_domain`, during the variable-data pass.  `None` -- every
    /// structured product -- draws none.  A caller that sets this hands a
    /// field the rasteriser will not draw (an all-NaN placeholder) and lets
    /// the cells carry the values, because a Voronoi mesh has no (ny, nx).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mesh_cells: Option<crate::mesh_cells::MeshCellsLayer>,
    #[serde(default)]
    pub inverse_raster_projection: Option<InverseRasterProjection>,
    /// The projection `projected_domain` was built in, with every
    /// data-derived parameter resolved.
    ///
    /// gpuwm addition (VENDOR.md).  Purely descriptive: the renderer never
    /// reads it to draw a pixel.  When set (together with
    /// `geographic_bounds` and a `projected_domain`), the save path
    /// publishes a [`crate::georeference::PanelGeoReference`] on its
    /// [`crate::RenderSaveTiming`], which is the only way a consumer
    /// holding the finished PNG can put a lat/lon on a pixel -- recovering
    /// the mapping by registration measurably cannot work (a global panel
    /// is Robinson; no linear fit describes Robinson).  `None` -- every
    /// existing caller -- publishes nothing and changes nothing.
    #[serde(default)]
    pub resolved_projection: Option<crate::georeference::ResolvedProjection>,
    /// `(west, east, south, north)` degrees of the DATA the caller framed,
    /// carried into the published georeference for sanity checks and
    /// captions.  Descriptive only, like `resolved_projection`; a
    /// georeference is withheld rather than published with made-up bounds
    /// when this is `None`, because a silently-zero bounds field is worse
    /// than no georeference at all (gpuwm addition, VENDOR.md).
    #[serde(default)]
    pub geographic_bounds: Option<(f64, f64, f64, f64)>,
    #[serde(default)]
    pub projected_place_labels: Vec<ProjectedPlaceLabel>,
    #[serde(default)]
    pub projected_points: Vec<ProjectedPointOverlay>,
    pub projected_lines: Vec<ProjectedLineOverlay>,
    /// Translucent fills under the contour lines; empty draws nothing and
    /// serialises nothing, so every request without one is unchanged.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub contour_fills: Vec<ContourFillLayer>,
    pub contours: Vec<ContourLayer>,
    pub wind_barbs: Vec<WindBarbLayer>,
    #[serde(default)]
    pub wind_streamlines: Vec<WindStreamlineLayer>,
    pub semantics: Option<ProductSemantics>,
    /// The note the map carries when the field has no value the colour
    /// scale draws (for example "No 0 C level: ..."). `None` uses the
    /// renderer's generic note. See `render::empty_map_note`.
    #[serde(default)]
    pub empty_note: Option<String>,
}

const fn default_supersample_factor() -> u32 {
    1
}

const fn default_supersample_sharpen() -> bool {
    true
}

impl MapRenderRequest {
    pub fn new(field: Field2D, scale: ColorScale) -> Self {
        Self {
            field,
            rgba_grid: None,
            product_metadata: None,
            width: 1100,
            height: 850,
            scale,
            background: Color::WHITE,
            colorbar: true,
            title: None,
            subtitle_left: None,
            subtitle_center: None,
            subtitle_right: None,
            cbar_tick_step: None,
            render_density: RenderDensity::default(),
            legend: LegendControls::default(),
            chrome_scale: ChromeScale::default(),
            supersample_factor: default_supersample_factor(),
            supersample_sharpen: default_supersample_sharpen(),
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
            contour_fills: Vec::new(),
            contours: Vec::new(),
            wind_barbs: Vec::new(),
            wind_streamlines: Vec::new(),
            semantics: None,
            empty_note: None,
        }
    }

    pub fn from_core_field(field: core::Field2D, scale: ColorScale) -> Self {
        Self::new(field.into(), scale)
    }

    pub fn for_weather_product(field: Field2D, product: crate::weather::WeatherProduct) -> Self {
        let mut request = Self::new(field, ColorScale::Weather(product.scale_preset()));
        apply_reference_discrete_defaults(&mut request);
        request.title = Some(product.display_title().to_string());
        request.cbar_tick_step = product.default_tick_step();
        request.semantics = Some(product.semantics());
        request.visual_mode = product.default_visual_mode();
        request.product_metadata = Some(
            core::ProductKeyMetadata::new(product.display_title())
                .with_native_units(request.field.units.clone()),
        );
        request
    }

    pub fn for_core_weather_product(
        field: core::Field2D,
        product: crate::weather::WeatherProduct,
    ) -> Self {
        Self::for_weather_product(field.into(), product)
    }

    pub fn for_derived_product(
        field: Field2D,
        product: crate::weather::DerivedProductStyle,
    ) -> Self {
        let mut request = Self::new(field, ColorScale::Discrete(product.scale()));
        apply_reference_discrete_defaults(&mut request);
        request.title = Some(product.display_title().to_string());
        request.cbar_tick_step = product.default_tick_step();
        request.semantics = Some(product.semantics());
        request.visual_mode = product.default_visual_mode();
        request.product_metadata = Some(
            core::ProductKeyMetadata::new(product.display_title())
                .with_category("derived")
                .with_native_units(request.field.units.clone()),
        );
        request
    }

    pub fn for_palette_fill(
        field: Field2D,
        palette: crate::weather::WeatherPalette,
        levels: Vec<f64>,
        extend: ExtendMode,
    ) -> Self {
        let mut request = Self::new(
            field,
            ColorScale::Discrete(crate::weather::palette_scale(palette, levels, extend, None)),
        );
        apply_reference_discrete_defaults(&mut request);
        request
    }

    pub fn contour_only(field: Field2D) -> Self {
        let mut request = Self::new(field, ColorScale::Discrete(blank_fill_scale()));
        request.colorbar = false;
        request.visual_mode = ProductVisualMode::OverlayAnalysis;
        request
    }

    pub fn with_visual_mode(mut self, visual_mode: ProductVisualMode) -> Self {
        self.visual_mode = visual_mode;
        self
    }

    pub fn with_semantics(mut self, semantics: ProductSemantics) -> Self {
        self.semantics = Some(semantics);
        self
    }

    pub fn with_product_metadata(mut self, product_metadata: core::ProductKeyMetadata) -> Self {
        self.product_metadata = Some(product_metadata);
        self
    }

    pub fn apply_projected_map(&mut self, projected: &ProjectedMap) -> &mut Self {
        self.projected_domain = Some(projected.domain());
        self.projected_lines = projected.lines.clone();
        self.projected_polygons = projected.polygons.clone();
        self.inverse_raster_projection = projected.inverse_raster_projection.clone();
        self
    }

    pub fn with_projected_map(mut self, projected: &ProjectedMap) -> Self {
        self.apply_projected_map(projected);
        self
    }

    pub fn resolved_semantics(&self) -> Option<ProductSemantics> {
        self.semantics.clone().or_else(|| {
            self.product_metadata
                .as_ref()
                .and_then(|metadata| metadata.provenance.clone())
                .map(Into::into)
        })
    }

    pub fn product_provenance(&self) -> Option<&core::ProductProvenance> {
        self.product_metadata
            .as_ref()
            .and_then(|metadata| metadata.provenance.as_ref())
    }

    pub(crate) fn is_overlay_only(&self) -> bool {
        self.rgba_grid.is_none() && !self.colorbar && is_blank_fill_scale(&self.scale)
    }

    pub fn add_contour_field(
        &mut self,
        field: &Field2D,
        levels: Vec<f64>,
        style: ContourStyle,
    ) -> Result<&mut Self, RustwxRenderError> {
        ensure_same_grid(&self.field, field, "contour")?;
        self.contours
            .push(ContourLayer::from_field(field, levels, style));
        Ok(self)
    }

    pub fn with_contour_field(
        mut self,
        field: &Field2D,
        levels: Vec<f64>,
        style: ContourStyle,
    ) -> Result<Self, RustwxRenderError> {
        self.add_contour_field(field, levels, style)?;
        Ok(self)
    }

    pub fn add_wind_barbs(
        &mut self,
        u: &Field2D,
        v: &Field2D,
        style: WindBarbStyle,
    ) -> Result<&mut Self, RustwxRenderError> {
        ensure_same_grid(&self.field, u, "wind_barb_u")?;
        ensure_same_grid(&self.field, v, "wind_barb_v")?;
        ensure_same_grid(u, v, "wind_barb_uv")?;
        self.wind_barbs
            .push(WindBarbLayer::from_fields(u, v, style));
        Ok(self)
    }

    pub fn with_wind_barbs(
        mut self,
        u: &Field2D,
        v: &Field2D,
        style: WindBarbStyle,
    ) -> Result<Self, RustwxRenderError> {
        self.add_wind_barbs(u, v, style)?;
        Ok(self)
    }

    pub fn add_wind_streamlines(
        &mut self,
        u: &Field2D,
        v: &Field2D,
        style: WindStreamlineStyle,
    ) -> Result<&mut Self, RustwxRenderError> {
        ensure_same_grid(&self.field, u, "wind_streamline_u")?;
        ensure_same_grid(&self.field, v, "wind_streamline_v")?;
        ensure_same_grid(u, v, "wind_streamline_uv")?;
        self.wind_streamlines
            .push(WindStreamlineLayer::from_fields(u, v, style));
        Ok(self)
    }

    pub fn with_wind_streamlines(
        mut self,
        u: &Field2D,
        v: &Field2D,
        style: WindStreamlineStyle,
    ) -> Result<Self, RustwxRenderError> {
        self.add_wind_streamlines(u, v, style)?;
        Ok(self)
    }
}

fn apply_reference_discrete_defaults(request: &mut MapRenderRequest) {
    request.render_density = RenderDensity {
        fill: LevelDensity::default(),
        palette_multiplier: 1,
    };
    request.legend = LegendControls {
        density: LevelDensity::default(),
        mode: LegendMode::Stepped,
    };
}

fn ensure_same_grid(
    base: &Field2D,
    overlay: &Field2D,
    layer: &'static str,
) -> Result<(), RustwxRenderError> {
    if base.grid != overlay.grid {
        return Err(RustwxRenderError::OverlayGridMismatch { layer });
    }
    Ok(())
}

fn blank_fill_scale() -> DiscreteColorScale {
    DiscreteColorScale {
        levels: vec![0.0, 1.0],
        colors: vec![Color::TRANSPARENT],
        extend: ExtendMode::Neither,
        mask_below: None,
    }
}

fn is_blank_fill_scale(scale: &ColorScale) -> bool {
    match scale {
        ColorScale::Discrete(scale) => {
            scale.levels == [0.0, 1.0]
                && scale.colors == [Color::TRANSPARENT]
                && scale.extend == ExtendMode::Neither
                && scale.mask_below.is_none()
        }
        ColorScale::Weather(_) => false,
    }
}

impl From<core::GridShape> for GridShape {
    fn from(value: core::GridShape) -> Self {
        Self {
            nx: value.nx,
            ny: value.ny,
        }
    }
}

impl From<GridShape> for core::GridShape {
    fn from(value: GridShape) -> Self {
        Self {
            nx: value.nx,
            ny: value.ny,
        }
    }
}

impl From<core::LatLonGrid> for LatLonGrid {
    fn from(value: core::LatLonGrid) -> Self {
        Self {
            shape: value.shape.into(),
            lat_deg: value.lat_deg,
            lon_deg: value.lon_deg,
        }
    }
}

impl From<LatLonGrid> for core::LatLonGrid {
    fn from(value: LatLonGrid) -> Self {
        Self {
            shape: value.shape.into(),
            lat_deg: value.lat_deg,
            lon_deg: value.lon_deg,
        }
    }
}

impl From<core::ProductKey> for ProductKey {
    fn from(value: core::ProductKey) -> Self {
        match value {
            core::ProductKey::Named(name) => Self::Named(name),
        }
    }
}

impl From<ProductKey> for core::ProductKey {
    fn from(value: ProductKey) -> Self {
        match value {
            ProductKey::Named(name) => Self::Named(name),
        }
    }
}

impl From<core::Field2D> for Field2D {
    fn from(value: core::Field2D) -> Self {
        Self {
            product: value.product.into(),
            units: value.units,
            grid: value.grid.into(),
            values: value.values,
        }
    }
}

impl From<Field2D> for core::Field2D {
    fn from(value: Field2D) -> Self {
        Self {
            product: value.product.into(),
            units: value.units,
            grid: value.grid.into(),
            values: value.values,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_render_field() -> Field2D {
        let shape = GridShape::new(3, 2).unwrap();
        let grid = LatLonGrid::new(
            shape,
            vec![35.0, 35.0, 35.0, 36.0, 36.0, 36.0],
            vec![-99.0, -98.0, -97.0, -99.0, -98.0, -97.0],
        )
        .unwrap();
        Field2D::new(
            ProductKey::named("sbecape"),
            "J/kg",
            grid,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        )
        .unwrap()
    }

    #[test]
    fn field2d_round_trips_through_rustwx_core() {
        let render_field = sample_render_field();
        let core_field: core::Field2D = render_field.clone().into();
        let round_trip = Field2D::from(core_field);

        assert_eq!(round_trip, render_field);
    }

    #[test]
    fn weather_builder_accepts_core_field() {
        let core_field: core::Field2D = sample_render_field().into();
        let request = MapRenderRequest::for_core_weather_product(
            core_field,
            crate::weather::WeatherProduct::Mlecape,
        );

        assert!(matches!(
            request.scale,
            ColorScale::Weather(crate::weather::WeatherPreset::Ecape)
        ));
        assert_eq!(request.title.as_deref(), Some("MLECAPE"));
        assert_eq!(request.cbar_tick_step, Some(500.0));
        assert_eq!(
            request
                .semantics
                .as_ref()
                .map(|semantics| semantics.maturity),
            Some(ProductMaturity::Operational)
        );
    }

    #[test]
    fn contour_only_builder_disables_colorbar_and_uses_blank_fill() {
        let request = MapRenderRequest::contour_only(sample_render_field());

        assert!(!request.colorbar);
        assert!(request.is_overlay_only());
        assert_eq!(request.visual_mode, ProductVisualMode::OverlayAnalysis);
        assert!(matches!(request.scale, ColorScale::Discrete(_)));
        assert_eq!(request.field.values, sample_render_field().values);
        match request.scale {
            ColorScale::Discrete(scale) => assert_eq!(scale.colors, vec![Color::TRANSPARENT]),
            _ => panic!("expected discrete blank fill scale"),
        }
    }

    #[test]
    fn projected_place_label_defaults_to_city_friendly_marker_and_label_style() {
        let label = ProjectedPlaceLabel::new(12.5, -97.25).with_label("Norman");

        assert_eq!(label.x, 12.5);
        assert_eq!(label.y, -97.25);
        assert_eq!(label.label.as_deref(), Some("Norman"));
        assert_eq!(label.priority, ProjectedPlaceLabelPriority::Primary);
        assert_eq!(
            label.style.label_placement,
            ProjectedLabelPlacement::AboveRight
        );
        assert_eq!(label.style.marker_radius_px, 3);
        assert_eq!(label.style.label_scale, 1);
        assert!(!label.style.label_bold);
    }

    #[test]
    fn new_render_requests_start_without_projected_place_labels() {
        let request = MapRenderRequest::new(
            sample_render_field(),
            ColorScale::Weather(crate::weather::WeatherPreset::Cape),
        );

        assert!(request.projected_place_labels.is_empty());
    }

    #[test]
    fn overlay_builders_require_matching_grids() {
        let base = sample_render_field();
        let mut shifted = sample_render_field();
        shifted.grid.lon_deg[0] = -101.0;

        let contour_error = MapRenderRequest::contour_only(base.clone())
            .with_contour_field(&shifted, vec![1.0, 2.0], ContourStyle::default())
            .unwrap_err();
        assert!(matches!(
            contour_error,
            RustwxRenderError::OverlayGridMismatch { layer: "contour" }
        ));

        let wind_error = MapRenderRequest::contour_only(base)
            .with_wind_barbs(&shifted, &sample_render_field(), WindBarbStyle::default())
            .unwrap_err();
        assert!(matches!(
            wind_error,
            RustwxRenderError::OverlayGridMismatch {
                layer: "wind_barb_u"
            }
        ));
    }

    #[test]
    fn palette_fill_builder_uses_requested_palette_scale() {
        let request = MapRenderRequest::for_palette_fill(
            sample_render_field(),
            crate::weather::WeatherPalette::Temperature,
            vec![-40.0, -20.0, 0.0, 20.0, 40.0],
            ExtendMode::Both,
        );

        match request.scale {
            ColorScale::Discrete(scale) => {
                assert_eq!(scale.levels, vec![-40.0, -20.0, 0.0, 20.0, 40.0]);
                assert_eq!(scale.extend, ExtendMode::Both);
                assert!(!scale.colors.is_empty());
            }
            _ => panic!("expected discrete palette scale"),
        }
    }

    #[test]
    fn weather_family_builders_default_to_reference_stepped_density() {
        let weather = MapRenderRequest::for_weather_product(
            sample_render_field(),
            crate::weather::WeatherProduct::Sbcape,
        );
        assert_eq!(weather.render_density.fill, LevelDensity::default());
        assert_eq!(weather.render_density.palette_multiplier, 1);
        assert_eq!(weather.legend.density, LevelDensity::default());
        assert_eq!(weather.legend.mode, LegendMode::Stepped);

        let derived = MapRenderRequest::for_derived_product(
            sample_render_field(),
            crate::weather::DerivedProductStyle::BulkShear06km,
        );
        assert_eq!(derived.render_density.fill, LevelDensity::default());
        assert_eq!(derived.render_density.palette_multiplier, 1);
        assert_eq!(derived.legend.mode, LegendMode::Stepped);
    }

    #[test]
    fn derived_builder_sets_titles_scale_and_tick_steps() {
        let request = MapRenderRequest::for_derived_product(
            sample_render_field(),
            crate::weather::DerivedProductStyle::BulkShear06km,
        );

        assert_eq!(request.title.as_deref(), Some("0-6 KM BULK SHEAR"));
        assert_eq!(request.cbar_tick_step, Some(5.0));
        assert_eq!(
            request
                .semantics
                .as_ref()
                .map(|semantics| semantics.maturity),
            Some(ProductMaturity::Operational)
        );
        match request.scale {
            ColorScale::Discrete(scale) => {
                assert_eq!(
                    scale.levels,
                    vec![
                        0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0
                    ]
                );
                assert_eq!(scale.extend, ExtendMode::Max);
                assert!(!scale.colors.is_empty());
            }
            _ => panic!("expected discrete derived scale"),
        }
    }

    #[test]
    fn semantics_builder_attaches_non_operational_flags() {
        let request = MapRenderRequest::contour_only(sample_render_field())
            .with_semantics(ProductSemantics::proof().with_flag(ProductSemanticFlag::Proxy));

        let semantics = request
            .resolved_semantics()
            .expect("semantics should be attached");
        assert_eq!(semantics.maturity, ProductMaturity::Proof);
        assert!(semantics.has_flag(ProductSemanticFlag::ProofOriented));
        assert!(semantics.has_flag(ProductSemanticFlag::Proxy));
    }

    #[test]
    fn projected_map_helpers_apply_domain_and_basemap_together() {
        let projected = ProjectedMap {
            projected_x: vec![0.0, 1.0, 2.0],
            projected_y: vec![0.0, 1.0, 2.0],
            extent: ProjectedExtent {
                x_min: 0.0,
                x_max: 2.0,
                y_min: 0.0,
                y_max: 2.0,
            },
            lines: vec![ProjectedLineOverlay {
                points: vec![(0.0, 0.0), (1.0, 1.0)],
                color: Color::BLACK,
                width: 2,
                role: LineworkRole::Generic,
            }],
            polygons: vec![ProjectedPolygonFill {
                rings: vec![vec![(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]],
                color: Color::WHITE,
                role: PolygonRole::Generic,
            }],
            inverse_raster_projection: None,
        };

        let request =
            MapRenderRequest::contour_only(sample_render_field()).with_projected_map(&projected);
        assert_eq!(
            request
                .projected_domain
                .as_ref()
                .map(|domain| domain.x.clone()),
            Some(vec![0.0, 1.0, 2.0])
        );
        assert_eq!(request.projected_lines.len(), 1);
        assert_eq!(request.projected_polygons.len(), 1);
    }

    #[test]
    fn metadata_builder_keeps_typed_provenance_visible_on_request() {
        let provenance = core::ProductProvenance::new(
            core::ProductLineage::Windowed,
            core::ProductMaturity::Operational,
        )
        .with_flag(core::ProductSemanticFlag::Alias)
        .with_window(core::ProductWindowSpec::accumulation(Some(1)));

        let request = MapRenderRequest::contour_only(sample_render_field()).with_product_metadata(
            core::ProductKeyMetadata::new("1-h QPF")
                .with_category("windowed")
                .with_native_units("mm")
                .with_provenance(provenance),
        );

        let semantics = request
            .resolved_semantics()
            .expect("typed metadata should resolve render semantics");
        assert_eq!(semantics.maturity, ProductMaturity::Operational);
        assert!(semantics.has_flag(ProductSemanticFlag::Alias));
        let metadata = request
            .product_metadata
            .as_ref()
            .expect("request should expose product metadata");
        assert_eq!(metadata.category.as_deref(), Some("windowed"));
        assert_eq!(
            request.product_provenance().unwrap().window,
            Some(core::ProductWindowSpec::accumulation(Some(1)))
        );
    }
}
