use crate::color::Rgba;
use crate::presentation::{LineworkRole, PolygonRole};
use crate::projection::ProjectionProjector;
use crate::request::{
    GeographicClipBounds, ProjectedLabelPlacement, ProjectedMarkerShape,
    ProjectedPlaceLabelPriority,
};

#[derive(Clone, Debug)]
pub struct MapExtent {
    pub x_min: f64,
    pub x_max: f64,
    pub y_min: f64,
    pub y_max: f64,
}

#[derive(Clone, Debug)]
pub struct ProjectedGrid {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub ny: usize,
    pub nx: usize,
}

#[derive(Clone, Debug)]
pub(crate) struct InverseProjectedGrid {
    pub projector: ProjectionProjector,
    pub clip_bounds: Option<GeographicClipBounds>,
    pub lat_deg: Vec<f64>,
    pub lon_deg: Vec<f64>,
}

impl MapExtent {
    pub fn to_pixel(&self, x: f64, y: f64, img_w: u32, img_h: u32) -> Option<(f64, f64)> {
        let dx = self.x_max - self.x_min;
        let dy = self.y_max - self.y_min;
        if dx.abs() < 1e-12 || dy.abs() < 1e-12 {
            return None;
        }
        let rx = (x - self.x_min) / dx;
        let ry = 1.0 - (y - self.y_min) / dy;
        if !(-0.1..=1.1).contains(&rx) || !(-0.1..=1.1).contains(&ry) {
            return None;
        }
        Some((
            rx * (img_w.saturating_sub(1)) as f64,
            ry * (img_h.saturating_sub(1)) as f64,
        ))
    }
}

#[derive(Clone, Debug)]
pub struct ProjectedPolyline {
    pub points: Vec<(f64, f64)>,
    pub color: Rgba,
    pub width: u32,
    pub role: LineworkRole,
}

#[derive(Clone, Debug)]
pub struct ProjectedPointOverlay {
    pub x: f64,
    pub y: f64,
    pub color: Rgba,
    pub radius_px: u32,
    pub width_px: u32,
    pub shape: ProjectedMarkerShape,
}

/// A filled polygon in projected map coordinates. The first ring is the outer
/// boundary; additional rings punch holes (inlets/lakes). Typically sourced
/// from Natural Earth land/ocean/lake shapefiles.
#[derive(Clone, Debug)]
pub struct ProjectedPolygon {
    pub rings: Vec<Vec<(f64, f64)>>,
    pub color: Rgba,
    pub role: PolygonRole,
}

#[derive(Clone, Debug)]
pub struct ProjectedPlaceLabelStyle {
    pub marker_radius_px: u32,
    pub marker_fill: Rgba,
    pub marker_outline: Rgba,
    pub marker_outline_width: u32,
    pub label_color: Rgba,
    pub label_halo: Rgba,
    pub label_halo_width_px: u32,
    pub label_scale: u32,
    pub label_offset_x_px: i32,
    pub label_offset_y_px: i32,
    pub label_placement: ProjectedLabelPlacement,
    pub label_bold: bool,
}

#[derive(Clone, Debug)]
pub struct ProjectedPlaceLabelOverlay {
    pub x: f64,
    pub y: f64,
    pub label: Option<String>,
    pub priority: ProjectedPlaceLabelPriority,
    pub style: ProjectedPlaceLabelStyle,
}

/// A translucent banded fill under the contour lines (see
/// [`crate::request::ContourFillLayer`]): `cmap` maps each value to its
/// band colour, clear below the first band.
#[derive(Clone, Debug)]
pub struct ContourFillOverlay {
    pub data: Vec<f64>,
    pub ny: usize,
    pub nx: usize,
    pub cmap: crate::colormap::LeveledColormap,
}

#[derive(Clone, Debug)]
pub struct ContourOverlay {
    pub data: Vec<f64>,
    pub ny: usize,
    pub nx: usize,
    pub levels: Vec<f64>,
    pub color: Rgba,
    pub width: u32,
    pub labels: bool,
    /// When true, find and draw H/L extrema labels on the contour field.
    pub show_extrema: bool,
    pub pattern: crate::request::ContourLinePattern,
    pub major_every: Option<usize>,
    pub major_width: Option<u32>,
}

#[derive(Clone, Debug)]
pub struct DirectionProjection {
    pub projection: crate::georeference::ResolvedProjection,
    pub extent: MapExtent,
    pub lat_lon: Vec<(f32, f32)>,
}

#[derive(Clone, Debug)]
pub struct BarbOverlay {
    pub direction_projection: Option<std::sync::Arc<DirectionProjection>>,
    pub direction_arrows: bool,
    pub u: Vec<f32>,
    pub v: Vec<f32>,
    pub ny: usize,
    pub nx: usize,
    pub stride_x: usize,
    pub stride_y: usize,
    pub spacing_px: f64,
    pub color: Rgba,
    pub halo_color: Rgba,
    pub halo_width: u32,
    pub width: u32,
    pub length_px: f64,
}

#[derive(Clone, Debug)]
pub struct StreamlineOverlay {
    pub u: Vec<f64>,
    pub v: Vec<f64>,
    pub ny: usize,
    pub nx: usize,
    pub stride_x: usize,
    pub stride_y: usize,
    pub color: Rgba,
    pub width: u32,
    pub max_steps: usize,
    pub step_cells: f64,
    pub min_speed: f64,
}
