use rayon::prelude::*;
use std::error::Error;
use std::mem::{size_of, size_of_val};
use std::sync::Arc;

#[cfg(test)]
mod cache_tests;

use crate::MapExtent;
use crate::features::{
    BasemapDetail, BasemapStyle, load_styled_basemap_features_for_detail,
    load_styled_basemap_polygons_for_detail,
};
use crate::presentation::LineworkRole;
use crate::projection::{ProjectionProjector, ProjectionSpec};
use crate::request::{
    Color, InverseRasterProjection, ProjectedDomain, ProjectedExtent, ProjectedLineOverlay,
    ProjectedPolygonFill,
};

const DEFAULT_BASEMAP_GRATICULE: bool = false;

/// Maximum retained coordinate and basemap storage shared by product renders.
pub const PROJECTED_MAP_CACHE_BYTES: u64 = 128 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq)]
pub struct ProjectedMap {
    pub projected_x: Vec<f64>,
    pub projected_y: Vec<f64>,
    pub extent: ProjectedExtent,
    pub lines: Vec<ProjectedLineOverlay>,
    pub polygons: Vec<ProjectedPolygonFill>,
    pub inverse_raster_projection: Option<InverseRasterProjection>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct ProjectedBasemap {
    pub lines: Vec<ProjectedLineOverlay>,
    pub polygons: Vec<ProjectedPolygonFill>,
}

impl ProjectedMap {
    pub fn domain(&self) -> ProjectedDomain {
        ProjectedDomain {
            x: self.projected_x.clone(),
            y: self.projected_y.clone(),
            extent: self.extent.clone(),
        }
    }

    pub fn basemap(&self) -> ProjectedBasemap {
        ProjectedBasemap {
            lines: self.lines.clone(),
            polygons: self.polygons.clone(),
        }
    }

    pub fn split(self) -> (ProjectedDomain, ProjectedBasemap) {
        let domain = ProjectedDomain {
            x: self.projected_x,
            y: self.projected_y,
            extent: self.extent,
        };
        let basemap = ProjectedBasemap {
            lines: self.lines,
            polygons: self.polygons,
        };
        (domain, basemap)
    }

    pub fn rotated_degrees(mut self, degrees: f64) -> Self {
        if !degrees.is_finite() || degrees.abs() < 1.0e-9 {
            return self;
        }
        let radians = degrees.to_radians();
        let sin = radians.sin();
        let cos = radians.cos();
        let center = (
            (self.extent.x_min + self.extent.x_max) * 0.5,
            (self.extent.y_min + self.extent.y_max) * 0.5,
        );

        rotate_parallel_points(
            &mut self.projected_x,
            &mut self.projected_y,
            center,
            sin,
            cos,
        );
        for line in &mut self.lines {
            rotate_points(&mut line.points, center, sin, cos);
        }
        for polygon in &mut self.polygons {
            for ring in &mut polygon.rings {
                rotate_points(ring, center, sin, cos);
            }
        }
        self.extent = rotated_extent(self.extent, center, sin, cos);
        self.inverse_raster_projection = None;
        self
    }
}

fn rotate_parallel_points(xs: &mut [f64], ys: &mut [f64], center: (f64, f64), sin: f64, cos: f64) {
    for (x, y) in xs.iter_mut().zip(ys.iter_mut()) {
        (*x, *y) = rotate_point((*x, *y), center, sin, cos);
    }
}

fn rotate_points(points: &mut [(f64, f64)], center: (f64, f64), sin: f64, cos: f64) {
    for point in points {
        *point = rotate_point(*point, center, sin, cos);
    }
}

fn rotate_point(point: (f64, f64), center: (f64, f64), sin: f64, cos: f64) -> (f64, f64) {
    let dx = point.0 - center.0;
    let dy = point.1 - center.1;
    (
        center.0 + dx * cos - dy * sin,
        center.1 + dx * sin + dy * cos,
    )
}

fn rotated_extent(
    extent: ProjectedExtent,
    center: (f64, f64),
    sin: f64,
    cos: f64,
) -> ProjectedExtent {
    let corners = [
        (extent.x_min, extent.y_min),
        (extent.x_max, extent.y_min),
        (extent.x_max, extent.y_max),
        (extent.x_min, extent.y_max),
    ];
    let mut x_min = f64::INFINITY;
    let mut x_max = f64::NEG_INFINITY;
    let mut y_min = f64::INFINITY;
    let mut y_max = f64::NEG_INFINITY;
    for point in corners {
        let (x, y) = rotate_point(point, center, sin, cos);
        x_min = x_min.min(x);
        x_max = x_max.max(x);
        y_min = y_min.min(y);
        y_max = y_max.max(y);
    }
    ProjectedExtent {
        x_min,
        x_max,
        y_min,
        y_max,
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeographicBounds {
    pub west_deg: f64,
    pub east_deg: f64,
    pub south_deg: f64,
    pub north_deg: f64,
}

impl GeographicBounds {
    pub fn new(west_deg: f64, east_deg: f64, south_deg: f64, north_deg: f64) -> Self {
        Self {
            west_deg,
            east_deg,
            south_deg: south_deg.min(north_deg),
            north_deg: south_deg.max(north_deg),
        }
    }

    fn contains(self, lat_deg: f64, lon_deg: f64) -> bool {
        if !lat_deg.is_finite() || !lon_deg.is_finite() {
            return false;
        }
        if lat_deg < self.south_deg || lat_deg > self.north_deg {
            return false;
        }
        if self.longitude_span_deg() >= 359.0 {
            return true;
        }
        let west = normalize_longitude_deg(self.west_deg);
        let east = normalize_longitude_deg(self.east_deg);
        let lon = normalize_longitude_deg(lon_deg);
        if west <= east {
            lon >= west && lon <= east
        } else {
            lon >= west || lon <= east
        }
    }

    fn center_longitude(self) -> f64 {
        if self.longitude_span_deg() >= 359.0 {
            return 0.0;
        }
        let west = normalize_longitude_deg(self.west_deg);
        let mut east = normalize_longitude_deg(self.east_deg);
        if east < west {
            east += 360.0;
        }
        normalize_longitude_deg((west + east) / 2.0)
    }

    fn longitude_span_deg(self) -> f64 {
        let raw_span = (self.east_deg - self.west_deg).abs();
        if raw_span >= 359.0 {
            return raw_span.min(360.0);
        }

        let west = normalize_longitude_deg(self.west_deg);
        let east = normalize_longitude_deg(self.east_deg);
        if west <= east {
            east - west
        } else {
            east + 360.0 - west
        }
    }
}

impl From<(f64, f64, f64, f64)> for GeographicBounds {
    fn from(value: (f64, f64, f64, f64)) -> Self {
        Self::new(value.0, value.1, value.2, value.3)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ProjectedFrameSource {
    FullDomain,
    GeographicBounds(GeographicBounds),
    GeographicGridIntersection(GeographicBounds),
}

impl ProjectedFrameSource {
    fn matches(self, lat_deg: f64, lon_deg: f64) -> bool {
        match self {
            Self::FullDomain => true,
            Self::GeographicBounds(bounds) | Self::GeographicGridIntersection(bounds) => {
                bounds.contains(lat_deg, lon_deg)
            }
        }
    }

    fn geographic_bounds(self) -> Option<GeographicBounds> {
        match self {
            Self::FullDomain => None,
            Self::GeographicBounds(bounds) | Self::GeographicGridIntersection(bounds) => {
                Some(bounds)
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProjectedDomainBuildOptions {
    pub projection: Option<ProjectionSpec>,
    /// Optional latitude of origin for projection families that benefit from a
    /// caller-provided reference latitude. When absent, the builder uses the
    /// lat/lon mesh midpoint.
    pub reference_latitude_deg: Option<f64>,
    pub frame_source: ProjectedFrameSource,
    pub target_aspect_ratio: f64,
    pub fit_to_target_aspect: bool,
    pub pad_fraction: f64,
}

impl ProjectedDomainBuildOptions {
    pub fn from_bounds(bounds: (f64, f64, f64, f64), target_aspect_ratio: f64) -> Self {
        Self {
            projection: None,
            reference_latitude_deg: None,
            frame_source: ProjectedFrameSource::GeographicBounds(bounds.into()),
            target_aspect_ratio,
            fit_to_target_aspect: true,
            pad_fraction: 0.0,
        }
    }

    pub fn full_domain(target_aspect_ratio: f64) -> Self {
        Self {
            projection: None,
            reference_latitude_deg: None,
            frame_source: ProjectedFrameSource::FullDomain,
            target_aspect_ratio,
            fit_to_target_aspect: true,
            pad_fraction: 0.0,
        }
    }

    pub fn with_projection(mut self, projection: impl Into<ProjectionSpec>) -> Self {
        self.projection = Some(projection.into());
        self
    }

    pub fn with_reference_latitude(mut self, reference_latitude_deg: f64) -> Self {
        self.reference_latitude_deg = Some(reference_latitude_deg);
        self
    }

    pub fn with_geographic_grid_intersection_frame(
        mut self,
        bounds: impl Into<GeographicBounds>,
    ) -> Self {
        self.frame_source = ProjectedFrameSource::GeographicGridIntersection(bounds.into());
        self
    }

    pub fn with_natural_frame_aspect(mut self) -> Self {
        self.fit_to_target_aspect = false;
        self
    }

    pub fn with_padding(mut self, pad_fraction: f64) -> Self {
        self.pad_fraction = pad_fraction.max(0.0);
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProjectedBasemapBuildOptions {
    pub style: BasemapStyle,
    pub detail: BasemapDetail,
    pub polygon_pad_fraction: f64,
    pub line_pad_fraction: f64,
}

impl Default for ProjectedBasemapBuildOptions {
    fn default() -> Self {
        Self {
            style: BasemapStyle::Filled,
            detail: BasemapDetail::Regional,
            polygon_pad_fraction: 0.50,
            line_pad_fraction: 0.10,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProjectedMapBuildOptions {
    pub domain: ProjectedDomainBuildOptions,
    pub basemap: Option<ProjectedBasemapBuildOptions>,
}

impl ProjectedMapBuildOptions {
    pub fn from_bounds(bounds: (f64, f64, f64, f64), target_aspect_ratio: f64) -> Self {
        Self {
            domain: ProjectedDomainBuildOptions::from_bounds(bounds, target_aspect_ratio),
            basemap: Some(ProjectedBasemapBuildOptions::default()),
        }
    }

    pub fn full_domain(target_aspect_ratio: f64) -> Self {
        Self {
            domain: ProjectedDomainBuildOptions::full_domain(target_aspect_ratio),
            basemap: Some(ProjectedBasemapBuildOptions::default()),
        }
    }

    pub fn with_projection(mut self, projection: impl Into<ProjectionSpec>) -> Self {
        self.domain = self.domain.with_projection(projection);
        self
    }

    pub fn with_geographic_grid_intersection_frame(
        mut self,
        bounds: impl Into<GeographicBounds>,
    ) -> Self {
        self.domain = self.domain.with_geographic_grid_intersection_frame(bounds);
        self
    }

    pub fn with_natural_frame_aspect(mut self) -> Self {
        self.domain = self.domain.with_natural_frame_aspect();
        self
    }

    pub fn without_basemap(mut self) -> Self {
        self.basemap = None;
        self
    }

    pub fn with_basemap_detail(mut self, detail: BasemapDetail) -> Self {
        let mut basemap = self.basemap.unwrap_or_default();
        basemap.detail = detail;
        self.basemap = Some(basemap);
        self
    }

    pub fn with_basemap_padding(
        mut self,
        line_pad_fraction: f64,
        polygon_pad_fraction: f64,
    ) -> Self {
        let mut basemap = self.basemap.unwrap_or_default();
        basemap.line_pad_fraction = line_pad_fraction.max(0.0);
        basemap.polygon_pad_fraction = polygon_pad_fraction.max(0.0);
        self.basemap = Some(basemap);
        self
    }
}

pub fn build_projected_domain(
    lat_deg: &[f32],
    lon_deg: &[f32],
    options: &ProjectedDomainBuildOptions,
) -> Result<ProjectedDomain, Box<dyn Error>> {
    validate_lat_lon_mesh(lat_deg, lon_deg)?;
    let projector = resolved_projector(lat_deg, lon_deg, options)?;
    let (projected_x, projected_y, extent) = project_domain(
        lat_deg,
        lon_deg,
        projector,
        options.frame_source,
        options.pad_fraction,
        options.target_aspect_ratio,
        options.fit_to_target_aspect,
    )?;

    Ok(ProjectedDomain {
        x: projected_x,
        y: projected_y,
        extent,
    })
}

pub fn build_projected_map_with_options(
    lat_deg: &[f32],
    lon_deg: &[f32],
    options: &ProjectedMapBuildOptions,
) -> Result<ProjectedMap, Box<dyn Error>> {
    // Batch products build separate requests for the same mesh. Share only
    // geometry, never weather values, and bound retained memory across grids.
    static CACHE: std::sync::Mutex<ProjectedMapCache> = std::sync::Mutex::new(ProjectedMapCache {
        entries: Vec::new(),
        bytes: 0,
    });
    let cached = CACHE
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get(lat_deg, lon_deg, options);
    if let Some(map) = cached {
        return Ok((*map).clone());
    }
    // Neither projection nor the caller's full coordinate copy holds the
    // cache mutex. Nested Rayon work must never wait for its own parent.
    let map = Arc::new(build_projected_map_uncached(lat_deg, lon_deg, options)?);
    let map = {
        let mut cache = CACHE.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(cached) = cache.get(lat_deg, lon_deg, options) {
            cached
        } else {
            cache.insert(
                lat_deg,
                lon_deg,
                options,
                PROJECTED_MAP_CACHE_BYTES as usize,
                map,
            )
        }
    };
    Ok(Arc::try_unwrap(map).unwrap_or_else(|map| (*map).clone()))
}

struct CachedMap {
    lat: Vec<f32>,
    lon: Vec<f32>,
    options: String,
    map: Arc<ProjectedMap>,
    bytes: usize,
}

struct ProjectedMapCache {
    entries: Vec<CachedMap>,
    bytes: usize,
}

impl ProjectedMapCache {
    fn get(
        &mut self,
        lat: &[f32],
        lon: &[f32],
        options: &ProjectedMapBuildOptions,
    ) -> Option<Arc<ProjectedMap>> {
        // Debug's float spelling retains signed zero and every finite bit.
        // Comparing the entire request avoids reusing geometry across crops,
        // projections, aspect ratios, basemap styles or moving meshes.
        let options = format!("{options:?}");
        let same_bits = |a: &[f32], b: &[f32]| {
            a.len() == b.len() && a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits())
        };
        if let Some(index) = self.entries.iter().position(|entry| {
            entry.options == options && same_bits(&entry.lat, lat) && same_bits(&entry.lon, lon)
        }) {
            let entry = self.entries.remove(index);
            let map = entry.map.clone();
            self.entries.push(entry);
            return Some(map);
        }
        None
    }

    fn insert(
        &mut self,
        lat: &[f32],
        lon: &[f32],
        options: &ProjectedMapBuildOptions,
        byte_limit: usize,
        map: Arc<ProjectedMap>,
    ) -> Arc<ProjectedMap> {
        let options = format!("{options:?}");
        let mut bytes = (lat.len() + lon.len())
            .saturating_mul(size_of::<f32>())
            .saturating_add(
                (map.projected_x.len() + map.projected_y.len()).saturating_mul(size_of::<f64>()),
            )
            .saturating_add(options.len());
        for line in &map.lines {
            bytes = bytes.saturating_add(size_of_val(line))
                .saturating_add(line.points.len().saturating_mul(size_of::<(f64, f64)>()));
        }
        for polygon in &map.polygons {
            bytes = bytes.saturating_add(size_of_val(polygon));
            for ring in &polygon.rings {
                bytes = bytes.saturating_add(size_of_val(ring))
                    .saturating_add(ring.len().saturating_mul(size_of::<(f64, f64)>()));
            }
        }
        // Large grids still render normally; keeping another full copy could
        // otherwise crowd the forecast's host working set out of memory.
        if bytes <= byte_limit {
            while !self.entries.is_empty()
                && (self.entries.len() >= 4 || self.bytes.saturating_add(bytes) > byte_limit) {
                self.bytes -= self.entries.remove(0).bytes;
            }
            self.entries.push(CachedMap {
                lat: lat.to_vec(), lon: lon.to_vec(), options, map: map.clone(), bytes,
            });
            self.bytes += bytes;
        }
        map
    }

    #[cfg(test)]
    fn get_or_build(
        &mut self,
        lat: &[f32],
        lon: &[f32],
        options: &ProjectedMapBuildOptions,
        byte_limit: usize,
        build: impl FnOnce() -> Result<ProjectedMap, Box<dyn Error>>,
    ) -> Result<Arc<ProjectedMap>, Box<dyn Error>> {
        if let Some(map) = self.get(lat, lon, options) {
            return Ok(map);
        }
        let map = Arc::new(build()?);
        Ok(self.insert(lat, lon, options, byte_limit, map))
    }
}

fn build_projected_map_uncached(
    lat_deg: &[f32],
    lon_deg: &[f32],
    options: &ProjectedMapBuildOptions,
) -> Result<ProjectedMap, Box<dyn Error>> {
    validate_lat_lon_mesh(lat_deg, lon_deg)?;
    let projector = resolved_projector(lat_deg, lon_deg, &options.domain)?;
    let (projected_x, projected_y, extent) = project_domain(
        lat_deg,
        lon_deg,
        projector,
        options.domain.frame_source,
        options.domain.pad_fraction,
        options.domain.target_aspect_ratio,
        options.domain.fit_to_target_aspect,
    )?;

    let basemap = options
        .basemap
        .as_ref()
        .map(|basemap| {
            build_projected_basemap(projector, &extent, options.domain.frame_source, *basemap)
        })
        .transpose()?
        .unwrap_or_default();

    Ok(ProjectedMap {
        projected_x,
        projected_y,
        extent,
        lines: basemap.lines,
        polygons: basemap.polygons,
        inverse_raster_projection: None,
    })
}

pub fn build_projected_map(
    lat_deg: &[f32],
    lon_deg: &[f32],
    bounds: (f64, f64, f64, f64),
    target_ratio: f64,
) -> Result<ProjectedMap, Box<dyn Error>> {
    build_projected_map_with_options(
        lat_deg,
        lon_deg,
        &ProjectedMapBuildOptions::from_bounds(bounds, target_ratio),
    )
}

/// Project arbitrary `(lat, lon)` degree pairs into the SAME projected
/// space [`build_projected_map_with_options`] produces from the same mesh
/// and the same options.
///
/// gpuwm addition (VENDOR.md).  Every projected overlay type in
/// [`crate::request`] -- points, lines, place labels -- takes projected
/// coordinates, and until now the only way to obtain any was to be a
/// basemap feature or a grid cell.  A radar site, a range ring, a storm
/// report and a domain-boundary box are none of those: they are places,
/// known in degrees, that have to land where the map already put that
/// place.  Nothing existing calls this, so it cannot move an existing
/// pixel; it reuses `resolved_projector` rather than rebuilding one so a
/// marker can never land in a different frame than the fill it sits on.
pub fn project_geographic_points_with_options(
    lat_deg: &[f32],
    lon_deg: &[f32],
    options: &ProjectedMapBuildOptions,
    points: &[(f64, f64)],
) -> Result<Vec<(f64, f64)>, Box<dyn Error>> {
    validate_lat_lon_mesh(lat_deg, lon_deg)?;
    let projector = resolved_projector(lat_deg, lon_deg, &options.domain)?;
    Ok(points
        .iter()
        .map(|(lat, lon)| projector.project(*lat, *lon))
        .collect())
}

/// Project `(lat, lon)` rings in the map's own projection, splitting each
/// ring at that projection's cut. Each output retains its input ring index.
/// Point-wise longitude normalization alone reconnects the two map edges
/// across the interior and paints a full-width band for a seam-crossing cell.
pub fn project_geographic_rings_with_options(
    lat_deg: &[f32],
    lon_deg: &[f32],
    options: &ProjectedMapBuildOptions,
    rings: &[Vec<(f64, f64)>],
) -> Result<Vec<(usize, Vec<(f64, f64)>)>, Box<dyn Error>> {
    validate_lat_lon_mesh(lat_deg, lon_deg)?;
    let projector = resolved_projector(lat_deg, lon_deg, &options.domain)?;
    let mut projected = Vec::with_capacity(rings.len());
    for (index, ring) in rings.iter().enumerate() {
        let lonlat: Vec<_> = ring.iter().map(|&(lat, lon)| (lon, lat)).collect();
        for piece in project_ring_at_seam(projector, &lonlat) {
            projected.push((index, piece));
        }
    }
    Ok(projected)
}

/// The projection a panel built with these options is actually drawn in,
/// with every parameter the builder derives from the data resolved.
///
/// gpuwm addition (VENDOR.md).  A caller that wants to publish the
/// transform of a finished panel cannot reconstruct it from
/// [`ProjectedDomainBuildOptions::projection`]: that field is often
/// `None` (the spec is inferred from the mesh), and even when it is set,
/// `Geographic` has no central meridian in it and `LambertConformal` has
/// no reference latitude.  Both are filled in here, from the data, by
/// exactly the code the render uses.  This function reuses
/// `resolved_projector` for that reason -- a second, hand-copied
/// resolution is how a published transform ends up describing a different
/// map than the one that was drawn.
pub fn resolved_projection_for_options(
    lat_deg: &[f32],
    lon_deg: &[f32],
    options: &ProjectedDomainBuildOptions,
) -> Result<crate::georeference::ResolvedProjection, Box<dyn Error>> {
    validate_lat_lon_mesh(lat_deg, lon_deg)?;
    let projector = resolved_projector(lat_deg, lon_deg, options)?;
    Ok(crate::georeference::ResolvedProjection::from_projector(
        projector,
    ))
}

fn resolved_projector(
    lat_deg: &[f32],
    lon_deg: &[f32],
    options: &ProjectedDomainBuildOptions,
) -> Result<ProjectionProjector, Box<dyn Error>> {
    let projection = options
        .projection
        .clone()
        .or_else(|| ProjectionSpec::infer_from_latlon_grid(lat_deg, lon_deg))
        .ok_or("projected map builder requires at least one finite lat/lon point")?;
    let reference_longitude_deg = match (&projection, options.frame_source) {
        (ProjectionSpec::Geographic, source) => source
            .geographic_bounds()
            .map(GeographicBounds::center_longitude),
        _ => None,
    };
    projection
        .build_projector(
            options.reference_latitude_deg,
            reference_longitude_deg,
            lat_deg,
            lon_deg,
        )
        .map_err(Into::into)
}

fn validate_lat_lon_mesh(lat_deg: &[f32], lon_deg: &[f32]) -> Result<(), Box<dyn Error>> {
    if lat_deg.len() != lon_deg.len() {
        return Err("lat/lon arrays must have the same length".into());
    }
    if lat_deg.is_empty() {
        return Err("lat/lon arrays must not be empty".into());
    }
    Ok(())
}

fn project_domain(
    lat_deg: &[f32],
    lon_deg: &[f32],
    projector: ProjectionProjector,
    frame_source: ProjectedFrameSource,
    pad_fraction: f64,
    target_aspect_ratio: f64,
    fit_to_target_aspect: bool,
) -> Result<(Vec<f64>, Vec<f64>, ProjectedExtent), Box<dyn Error>> {
    let (projected_x, projected_y): (Vec<_>, Vec<_>) = lat_deg
        .par_iter()
        .zip(lon_deg)
        .map(|(&lat, &lon)| projector.project(lat as f64, lon as f64))
        .unzip();
    let mut full_bounds = ProjectedBounds::default();
    let mut framed_bounds = ProjectedBounds::default();

    // Keep the bounds fold in input order, including signed-zero ties.
    for (((&lat, &lon), &x), &y) in lat_deg
        .iter()
        .zip(lon_deg)
        .zip(&projected_x)
        .zip(&projected_y)
    {
        let lat = lat as f64;
        let lon = lon as f64;
        if !x.is_finite() || !y.is_finite() {
            continue;
        }
        full_bounds.include(x, y);
        if frame_source.matches(lat, lon) {
            framed_bounds.include(x, y);
        }
    }

    let bounds = match frame_source {
        ProjectedFrameSource::GeographicBounds(bounds) => {
            if !framed_bounds.is_valid() {
                return Err(
                    "requested geographic bounds crop does not intersect the model grid".into(),
                );
            }
            projected_geographic_frame_bounds(projector, bounds).unwrap_or(framed_bounds)
        }
        ProjectedFrameSource::GeographicGridIntersection(_) => {
            if !framed_bounds.is_valid() {
                return Err(
                    "requested geographic bounds crop does not intersect the model grid".into(),
                );
            }
            framed_bounds
        }
        ProjectedFrameSource::FullDomain if framed_bounds.is_valid() => framed_bounds,
        ProjectedFrameSource::FullDomain => full_bounds,
    };
    if !bounds.is_valid() {
        return Err("projected extent produced no finite coordinates".into());
    }

    let padded = bounds.expanded(pad_fraction.max(0.0));
    let extent = if fit_to_target_aspect {
        MapExtent::from_bounds(
            padded.min_x,
            padded.max_x,
            padded.min_y,
            padded.max_y,
            target_aspect_ratio,
        )
    } else {
        MapExtent {
            x_min: padded.min_x,
            x_max: padded.max_x,
            y_min: padded.min_y,
            y_max: padded.max_y,
        }
    };

    Ok((
        projected_x,
        projected_y,
        ProjectedExtent {
            x_min: extent.x_min,
            x_max: extent.x_max,
            y_min: extent.y_min,
            y_max: extent.y_max,
        },
    ))
}

fn projected_geographic_frame_bounds(
    projector: ProjectionProjector,
    bounds: GeographicBounds,
) -> Option<ProjectedBounds> {
    let mut projected = ProjectedBounds::default();
    let segments = if bounds.longitude_span_deg() >= 300.0 {
        180
    } else {
        96
    };
    let west = normalize_longitude_deg(bounds.west_deg);
    let mut east = normalize_longitude_deg(bounds.east_deg);
    if bounds.longitude_span_deg() >= 359.0 {
        east = west + 360.0;
    } else if east < west {
        east += 360.0;
    }

    for step in 0..=segments {
        let t = step as f64 / segments as f64;
        let lon = normalize_longitude_deg(west + (east - west) * t);
        include_projected_point(&mut projected, projector.project(bounds.south_deg, lon));
        include_projected_point(&mut projected, projector.project(bounds.north_deg, lon));
    }
    for step in 0..=segments {
        let t = step as f64 / segments as f64;
        let lat = bounds.south_deg + (bounds.north_deg - bounds.south_deg) * t;
        include_projected_point(&mut projected, projector.project(lat, bounds.west_deg));
        include_projected_point(&mut projected, projector.project(lat, bounds.east_deg));
    }

    // Full-world and near-global frames cannot be represented safely by only
    // sampling the geographic rectangle perimeter. At the antimeridian,
    // normalized -180 and +180 can collapse onto the same projected side, and
    // projections such as Robinson reach their widest x extent around the
    // equator rather than along the north/south frame edges. Sample the
    // interior grid so the fitted projected frame is centered on the real map
    // silhouette instead of being biased toward one seam side.
    if should_sample_geographic_frame_interior(bounds) {
        let lat_segments = 72usize;
        let lon_segments = 180usize;
        for lat_step in 0..=lat_segments {
            let lat_t = lat_step as f64 / lat_segments as f64;
            let lat = bounds.south_deg + (bounds.north_deg - bounds.south_deg) * lat_t;
            for lon_step in 0..=lon_segments {
                let lon_t = lon_step as f64 / lon_segments as f64;
                let lon = normalize_longitude_deg(west + (east - west) * lon_t);
                include_projected_point(&mut projected, projector.project(lat, lon));
            }
        }
    }

    projected.is_valid().then_some(projected)
}

fn should_sample_geographic_frame_interior(bounds: GeographicBounds) -> bool {
    bounds.longitude_span_deg() >= 300.0 || (bounds.north_deg - bounds.south_deg).abs() >= 120.0
}

fn include_projected_point(bounds: &mut ProjectedBounds, point: (f64, f64)) {
    if point.0.is_finite() && point.1.is_finite() {
        bounds.include(point.0, point.1);
    }
}

fn build_projected_basemap(
    projector: ProjectionProjector,
    extent: &ProjectedExtent,
    frame_source: ProjectedFrameSource,
    options: ProjectedBasemapBuildOptions,
) -> Result<ProjectedBasemap, Box<dyn Error>> {
    let line_bbox = expanded_bbox(extent, options.line_pad_fraction.max(0.0));
    let polygon_bbox = expanded_bbox(extent, options.polygon_pad_fraction.max(0.0));
    let geographic_clip = basemap_geographic_clip(frame_source);

    let mut lines = Vec::new();
    if subtle_graticule_enabled(options.detail) {
        append_graticule_lines(
            &mut lines,
            projector,
            line_bbox,
            geographic_clip,
            options.detail,
        );
    }

    let line_densify_step_deg = basemap_line_densify_step_deg(options.detail);
    let max_projected_step = max_projected_basemap_segment_length(line_bbox);
    for layer in load_styled_basemap_features_for_detail(options.style, options.detail) {
        let color = Color::rgba(layer.color.r, layer.color.g, layer.color.b, layer.color.a);
        for line in layer.lines {
            let mut current = Vec::<(f64, f64)>::with_capacity(line.len());
            let mut previous_lonlat: Option<(f64, f64)> = None;
            let mut previous_projected: Option<(f64, f64)> = None;
            for (lon, lat) in line {
                if let Some((prev_lon, prev_lat)) = previous_lonlat {
                    let steps = densified_lonlat_segment_steps(
                        prev_lon,
                        prev_lat,
                        lon,
                        lat,
                        line_densify_step_deg,
                    );
                    for step in 1..=steps {
                        let t = step as f64 / steps as f64;
                        let point_lon = interpolate_longitude(prev_lon, lon, t);
                        let point_lat = prev_lat + (lat - prev_lat) * t;
                        push_projected_line_point(
                            &mut lines,
                            &mut current,
                            &mut previous_projected,
                            projector,
                            geographic_clip,
                            line_bbox,
                            max_projected_step,
                            point_lon,
                            point_lat,
                            color,
                            layer.width,
                            layer.role,
                        );
                    }
                } else {
                    push_projected_line_point(
                        &mut lines,
                        &mut current,
                        &mut previous_projected,
                        projector,
                        geographic_clip,
                        line_bbox,
                        max_projected_step,
                        lon,
                        lat,
                        color,
                        layer.width,
                        layer.role,
                    );
                }
                previous_lonlat = Some((lon, lat));
            }
            if current.len() >= 2 {
                lines.push(ProjectedLineOverlay {
                    points: current,
                    color,
                    width: layer.width,
                    role: layer.role,
                });
            }
        }
    }

    let mut polygons = Vec::new();
    let polygon_densify_step_deg = basemap_polygon_densify_step_deg(options.detail);
    for layer in load_styled_basemap_polygons_for_detail(options.style, options.detail) {
        let color = Color::rgba(layer.color.r, layer.color.g, layer.color.b, layer.color.a);
        for polygon in layer.polygons {
            let rings: Vec<Vec<(f64, f64)>> = polygon
                .into_iter()
                .filter(|ring| {
                    geographic_clip
                        .map(|bounds| ring.iter().any(|&(lon, lat)| bounds.contains(lat, lon)))
                        .unwrap_or(true)
                })
                .flat_map(|ring| project_densified_ring(projector, &ring, polygon_densify_step_deg))
                .filter(|ring| ring_overlaps_bbox(ring, polygon_bbox))
                .collect();
            if !rings.is_empty() {
                polygons.push(ProjectedPolygonFill {
                    rings,
                    color,
                    role: layer.role,
                });
            }
        }
    }

    Ok(ProjectedBasemap { lines, polygons })
}

fn basemap_geographic_clip(frame_source: ProjectedFrameSource) -> Option<GeographicBounds> {
    let _ = frame_source;
    None
}

fn subtle_graticule_enabled(detail: BasemapDetail) -> bool {
    if matches!(detail, BasemapDetail::Regional) {
        return false;
    }
    std::env::var("RUSTWX_BASEMAP_GRATICULE")
        .ok()
        .map(|value| parse_basemap_graticule_flag(&value))
        .unwrap_or(DEFAULT_BASEMAP_GRATICULE)
}

fn parse_basemap_graticule_flag(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on"
    )
}

fn append_graticule_lines(
    lines: &mut Vec<ProjectedLineOverlay>,
    projector: ProjectionProjector,
    bbox: (f64, f64, f64, f64),
    geographic_clip: Option<GeographicBounds>,
    detail: BasemapDetail,
) {
    let color = match detail {
        BasemapDetail::Global => Color::rgba(42, 52, 66, 30),
        BasemapDetail::Broad => Color::rgba(42, 52, 66, 24),
        BasemapDetail::Regional => Color::rgba(42, 52, 66, 0),
    };
    let step_deg = match detail {
        BasemapDetail::Global => 2.0,
        BasemapDetail::Broad => 1.0,
        BasemapDetail::Regional => 0.5,
    };
    let max_projected_step = max_projected_basemap_segment_length(bbox);

    let latitude_lines: &[f64] = match detail {
        BasemapDetail::Global => &[-60.0, -30.0, 0.0, 30.0, 60.0],
        BasemapDetail::Broad => &[-60.0, -40.0, -20.0, 0.0, 20.0, 40.0, 60.0, 80.0],
        BasemapDetail::Regional => &[],
    };
    for &lat in latitude_lines {
        let mut current = Vec::new();
        let mut previous_projected = None;
        let mut lon = -180.0;
        while lon <= 180.0 {
            push_projected_line_point(
                lines,
                &mut current,
                &mut previous_projected,
                projector,
                geographic_clip,
                bbox,
                max_projected_step,
                lon,
                lat,
                color,
                1,
                LineworkRole::Generic,
            );
            lon += step_deg;
        }
        flush_projected_line(lines, &mut current, color, 1, LineworkRole::Generic);
    }

    let lon_step = match detail {
        BasemapDetail::Global => 30,
        BasemapDetail::Broad => 20,
        BasemapDetail::Regional => 30,
    };
    for lon in (-180..=180).step_by(lon_step) {
        let mut current = Vec::new();
        let mut previous_projected = None;
        let mut lat = -80.0;
        while lat <= 80.0 {
            push_projected_line_point(
                lines,
                &mut current,
                &mut previous_projected,
                projector,
                geographic_clip,
                bbox,
                max_projected_step,
                lon as f64,
                lat,
                color,
                1,
                LineworkRole::Generic,
            );
            lat += step_deg;
        }
        flush_projected_line(lines, &mut current, color, 1, LineworkRole::Generic);
    }
}

fn basemap_line_densify_step_deg(detail: BasemapDetail) -> f64 {
    match detail {
        BasemapDetail::Global => 1.25,
        BasemapDetail::Broad => 0.9,
        BasemapDetail::Regional => 0.65,
    }
}

fn basemap_polygon_densify_step_deg(detail: BasemapDetail) -> f64 {
    match detail {
        BasemapDetail::Global => 2.0,
        BasemapDetail::Broad => 1.5,
        BasemapDetail::Regional => 1.0,
    }
}

fn max_projected_basemap_segment_length(bbox: (f64, f64, f64, f64)) -> f64 {
    let width = (bbox.1 - bbox.0).abs();
    let height = (bbox.3 - bbox.2).abs();
    width.max(height).max(1.0) * 0.30
}

fn densified_lonlat_segment_steps(
    lon0: f64,
    lat0: f64,
    lon1: f64,
    lat1: f64,
    max_step_deg: f64,
) -> usize {
    if !lon0.is_finite()
        || !lat0.is_finite()
        || !lon1.is_finite()
        || !lat1.is_finite()
        || !max_step_deg.is_finite()
        || max_step_deg <= 0.0
    {
        return 1;
    }
    let lon_span = wrapped_longitude_delta_deg(lon0, lon1).abs();
    let lat_span = (lat1 - lat0).abs();
    (lon_span.max(lat_span) / max_step_deg).ceil().max(1.0) as usize
}

fn wrapped_longitude_delta_deg(lon0: f64, lon1: f64) -> f64 {
    let mut delta = normalize_longitude_deg(lon1) - normalize_longitude_deg(lon0);
    if delta > 180.0 {
        delta -= 360.0;
    } else if delta < -180.0 {
        delta += 360.0;
    }
    delta
}

fn interpolate_longitude(lon0: f64, lon1: f64, t: f64) -> f64 {
    normalize_longitude_deg(
        normalize_longitude_deg(lon0) + wrapped_longitude_delta_deg(lon0, lon1) * t,
    )
}

fn push_projected_line_point(
    lines: &mut Vec<ProjectedLineOverlay>,
    current: &mut Vec<(f64, f64)>,
    previous_projected: &mut Option<(f64, f64)>,
    projector: ProjectionProjector,
    geographic_clip: Option<GeographicBounds>,
    bbox: (f64, f64, f64, f64),
    max_projected_step: f64,
    lon: f64,
    lat: f64,
    color: Color,
    width: u32,
    role: LineworkRole,
) {
    if geographic_clip.is_some_and(|bounds| !bounds.contains(lat, lon)) {
        flush_projected_line(lines, current, color, width, role);
        *previous_projected = None;
        return;
    }
    let point = projector.project(lat, lon);
    if !point.0.is_finite() || !point.1.is_finite() {
        flush_projected_line(lines, current, color, width, role);
        *previous_projected = None;
        return;
    }

    if let Some(previous) = *previous_projected {
        if projected_distance(previous, point) > max_projected_step {
            flush_projected_line(lines, current, color, width, role);
            if point_in_bbox(point, bbox) {
                current.push(point);
            }
            *previous_projected = Some(point);
            return;
        }
        append_clipped_projected_segment(lines, current, previous, point, bbox, color, width, role);
    } else if point_in_bbox(point, bbox) {
        current.push(point);
    }

    if !point_in_bbox(point, bbox) {
        flush_projected_line(lines, current, color, width, role);
    }
    *previous_projected = Some(point);
}

fn flush_projected_line(
    lines: &mut Vec<ProjectedLineOverlay>,
    current: &mut Vec<(f64, f64)>,
    color: Color,
    width: u32,
    role: LineworkRole,
) {
    if current.len() >= 2 {
        lines.push(ProjectedLineOverlay {
            points: std::mem::take(current),
            color,
            width,
            role,
        });
    } else {
        current.clear();
    }
}

fn append_clipped_projected_segment(
    lines: &mut Vec<ProjectedLineOverlay>,
    current: &mut Vec<(f64, f64)>,
    previous: (f64, f64),
    point: (f64, f64),
    bbox: (f64, f64, f64, f64),
    color: Color,
    width: u32,
    role: LineworkRole,
) {
    let Some((start, end)) = clip_projected_segment_to_bbox(previous, point, bbox) else {
        if !point_in_bbox(point, bbox) {
            flush_projected_line(lines, current, color, width, role);
        }
        return;
    };

    if current.is_empty() {
        current.push(start);
    } else if current
        .last()
        .is_some_and(|&last| !projected_points_close(last, start))
    {
        flush_projected_line(lines, current, color, width, role);
        current.push(start);
    }
    if current
        .last()
        .is_none_or(|&last| !projected_points_close(last, end))
    {
        current.push(end);
    }
}

fn clip_projected_segment_to_bbox(
    p0: (f64, f64),
    p1: (f64, f64),
    bbox: (f64, f64, f64, f64),
) -> Option<((f64, f64), (f64, f64))> {
    let (x_min, x_max, y_min, y_max) = bbox;
    if x_min > x_max || y_min > y_max {
        return None;
    }

    let dx = p1.0 - p0.0;
    let dy = p1.1 - p0.1;
    let mut t0 = 0.0;
    let mut t1 = 1.0;

    for (p, q) in [
        (-dx, p0.0 - x_min),
        (dx, x_max - p0.0),
        (-dy, p0.1 - y_min),
        (dy, y_max - p0.1),
    ] {
        if p.abs() <= f64::EPSILON {
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
            if r > t0 {
                t0 = r;
            }
        } else {
            if r < t0 {
                return None;
            }
            if r < t1 {
                t1 = r;
            }
        }
    }

    Some((
        (p0.0 + dx * t0, p0.1 + dy * t0),
        (p0.0 + dx * t1, p0.1 + dy * t1),
    ))
}

fn projected_points_close(a: (f64, f64), b: (f64, f64)) -> bool {
    projected_distance(a, b) <= 1.0e-6
}

fn projected_distance(a: (f64, f64), b: (f64, f64)) -> f64 {
    let dx = a.0 - b.0;
    let dy = a.1 - b.1;
    (dx * dx + dy * dy).sqrt()
}

/// Projects one lon/lat ring, densified, into one or more projected rings.
///
/// D-08 (2.8.8 acceptance row CY1): a ring that straddles the projection's
/// cut meridian (`central + 180`) used to come back as ONE ring whose
/// vertices jump from one edge of the projected plane to the other.  Its
/// projected bounding box then spans the whole plane, so it passed
/// `ring_overlaps_bbox` and the scanline fill painted every row the ring
/// covers across the full width of the map.  On the cyclone d01 Mercator
/// grid (central meridian -89.3) the Tibetan lakes Nam Co and Yamdrok Tso
/// sit on the cut at 90.7E and drew two full-width lake-coloured bands at
/// their own latitudes (about 30.7N and 29.0N), over the Gulf coast.
///
/// A ring that crosses the cut is therefore split there: each piece runs
/// from the cut back to the cut on one side of the plane and is closed
/// along the cut itself, which is where the true boundary of the visible
/// half lies.  A piece far from the frame is then culled like any other
/// ring.  A ring that never crosses the cut is projected exactly as before.
fn project_densified_ring(
    projector: ProjectionProjector,
    ring: &[(f64, f64)],
    max_step_deg: f64,
) -> Vec<Vec<(f64, f64)>> {
    if ring.is_empty() {
        return Vec::new();
    }
    let mut dense = Vec::with_capacity(ring.len());
    let mut previous_lonlat: Option<(f64, f64)> = None;
    for &(lon, lat) in ring {
        if let Some((prev_lon, prev_lat)) = previous_lonlat {
            let steps = densified_lonlat_segment_steps(prev_lon, prev_lat, lon, lat, max_step_deg);
            for step in 1..=steps {
                let t = step as f64 / steps as f64;
                let point_lon = interpolate_longitude(prev_lon, lon, t);
                let point_lat = prev_lat + (lat - prev_lat) * t;
                dense.push((point_lon, point_lat));
            }
        } else {
            dense.push((lon, lat));
        }
        previous_lonlat = Some((lon, lat));
    }
    project_ring_at_seam(projector, &dense)
}

fn project_ring_at_seam(
    projector: ProjectionProjector,
    ring: &[(f64, f64)],
) -> Vec<Vec<(f64, f64)>> {
    let project_all = |points: &[(f64, f64)]| -> Vec<(f64, f64)> {
        points
            .iter()
            .map(|&(lon, lat)| projector.project(lat, lon))
            .filter(|point| point.0.is_finite() && point.1.is_finite())
            .collect()
    };
    let pieces = projector
        .seam_longitude_deg()
        .and_then(|seam| split_lonlat_ring_at_seam(ring, seam));
    match pieces {
        None => vec![project_all(ring)],
        Some(pieces) => pieces
            .iter()
            .map(|piece| project_all(piece))
            .filter(|piece| piece.len() >= 3)
            .collect(),
    }
}

/// How far inside the cut a seam vertex is placed, in degrees, so it
/// projects onto the intended edge of the plane.
const SEAM_EDGE_EPSILON_DEG: f64 = 1.0e-9;

/// Splits a densified lon/lat ring where it crosses the cut meridian
/// `seam_deg`.  Returns `None` when it never does (the caller keeps the
/// ring untouched), otherwise one closed-along-the-cut piece per side run.
fn split_lonlat_ring_at_seam(dense: &[(f64, f64)], seam_deg: f64) -> Option<Vec<Vec<(f64, f64)>>> {
    if dense.len() < 2 || !seam_deg.is_finite() {
        return None;
    }
    let central = seam_deg - 180.0;
    let mut walk = dense.to_vec();
    if dense.first() != dense.last() {
        walk.push(dense[0]);
    }
    let mut pieces: Vec<Vec<(f64, f64)>> = vec![vec![walk[0]]];
    for pair in walk.windows(2) {
        let (a, b) = (pair[0], pair[1]);
        let rel_a = normalize_longitude_deg(a.0 - central);
        let delta = wrapped_longitude_delta_deg(a.0, b.0);
        let reached = rel_a + delta;
        let (bound, exit_rel, enter_rel) = if reached > 180.0 {
            (
                180.0,
                180.0 - SEAM_EDGE_EPSILON_DEG,
                -180.0 + SEAM_EDGE_EPSILON_DEG,
            )
        } else if reached <= -180.0 {
            (
                -180.0,
                -180.0 + SEAM_EDGE_EPSILON_DEG,
                180.0 - SEAM_EDGE_EPSILON_DEG,
            )
        } else {
            pieces.last_mut().expect("one piece is always open").push(b);
            continue;
        };
        let t = if delta.abs() > 0.0 {
            ((bound - rel_a) / delta).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let lat = a.1 + (b.1 - a.1) * t;
        pieces
            .last_mut()
            .expect("one piece is always open")
            .push((central + exit_rel, lat));
        pieces.push(vec![(central + enter_rel, lat), b]);
    }
    if pieces.len() < 2 {
        return None;
    }
    // The walk is closed, so the last run continues straight into the first.
    let first = pieces.remove(0);
    let last = pieces.last_mut().expect("at least one crossing");
    last.extend(first.into_iter().skip(1));
    Some(pieces)
}

fn point_in_bbox(point: (f64, f64), bbox: (f64, f64, f64, f64)) -> bool {
    point.0 >= bbox.0 && point.0 <= bbox.1 && point.1 >= bbox.2 && point.1 <= bbox.3
}

fn expanded_bbox(extent: &ProjectedExtent, pad_fraction: f64) -> (f64, f64, f64, f64) {
    let pad_x = 0.5 * pad_fraction * (extent.x_max - extent.x_min);
    let pad_y = 0.5 * pad_fraction * (extent.y_max - extent.y_min);
    (
        extent.x_min - pad_x,
        extent.x_max + pad_x,
        extent.y_min - pad_y,
        extent.y_max + pad_y,
    )
}

fn ring_overlaps_bbox(ring: &[(f64, f64)], bbox: (f64, f64, f64, f64)) -> bool {
    let mut bounds = ProjectedBounds::default();
    for &(x, y) in ring {
        bounds.include(x, y);
    }
    bounds.is_valid()
        && !(bounds.max_x < bbox.0
            || bounds.min_x > bbox.1
            || bounds.max_y < bbox.2
            || bounds.min_y > bbox.3)
}

fn normalize_longitude_deg(lon_deg: f64) -> f64 {
    let mut lon = lon_deg % 360.0;
    if lon > 180.0 {
        lon -= 360.0;
    } else if lon <= -180.0 {
        lon += 360.0;
    }
    lon
}

#[derive(Debug, Clone, Copy)]
struct ProjectedBounds {
    min_x: f64,
    max_x: f64,
    min_y: f64,
    max_y: f64,
}

impl Default for ProjectedBounds {
    fn default() -> Self {
        Self {
            min_x: f64::INFINITY,
            max_x: f64::NEG_INFINITY,
            min_y: f64::INFINITY,
            max_y: f64::NEG_INFINITY,
        }
    }
}

impl ProjectedBounds {
    fn include(&mut self, x: f64, y: f64) {
        self.min_x = self.min_x.min(x);
        self.max_x = self.max_x.max(x);
        self.min_y = self.min_y.min(y);
        self.max_y = self.max_y.max(y);
    }

    fn is_valid(self) -> bool {
        self.min_x.is_finite()
            && self.max_x.is_finite()
            && self.min_y.is_finite()
            && self.max_y.is_finite()
    }

    fn expanded(self, pad_fraction: f64) -> Self {
        let width = self.max_x - self.min_x;
        let height = self.max_y - self.min_y;
        let pad_x = width * pad_fraction / 2.0;
        let pad_y = height * pad_fraction / 2.0;
        Self {
            min_x: self.min_x - pad_x,
            max_x: self.max_x + pad_x,
            min_y: self.min_y - pad_y,
            max_y: self.max_y + pad_y,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::projection::ProjectionSpec;

    fn sample_lat_lon() -> (Vec<f32>, Vec<f32>) {
        (
            vec![35.0, 35.0, 35.0, 36.0, 36.0, 36.0],
            vec![-100.0, -99.0, -98.0, -100.0, -99.0, -98.0],
        )
    }

    #[test]
    fn projected_domain_builder_supports_full_domain_geographic_projection() {
        let (lat, lon) = sample_lat_lon();
        let domain = build_projected_domain(
            &lat,
            &lon,
            &ProjectedDomainBuildOptions::full_domain(2.0)
                .with_projection(ProjectionSpec::Geographic),
        )
        .expect("domain should build");

        assert_eq!(domain.x.len(), lat.len());
        assert_eq!(domain.y.len(), lat.len());
        assert!(domain.extent.x_min < 0.0);
        assert!(domain.extent.x_max > 0.0);
        assert!(domain.extent.y_max > domain.extent.y_min);
    }

    #[test]
    fn projected_domain_builder_respects_geographic_crop_bounds() {
        let (lat, lon) = sample_lat_lon();
        let full = build_projected_domain(
            &lat,
            &lon,
            &ProjectedDomainBuildOptions::full_domain(1.5)
                .with_projection(ProjectionSpec::Geographic),
        )
        .expect("full domain");
        let cropped = build_projected_domain(
            &lat,
            &lon,
            &ProjectedDomainBuildOptions::from_bounds((-99.25, -98.25, 35.0, 36.0), 1.5)
                .with_projection(ProjectionSpec::Geographic),
        )
        .expect("cropped domain");

        assert!(
            cropped.extent.x_max - cropped.extent.x_min < full.extent.x_max - full.extent.x_min
        );
    }

    #[test]
    fn grid_intersection_frame_uses_model_cells_not_requested_rectangle() {
        let lat = vec![0.0, 0.0, 1.0, 1.0];
        let lon = vec![0.0, 1.0, 0.2, 0.8];
        let rectangle = build_projected_domain(
            &lat,
            &lon,
            &ProjectedDomainBuildOptions::from_bounds((0.1, 0.9, 0.0, 1.0), 1.0)
                .with_projection(ProjectionSpec::Geographic),
        )
        .expect("rectangle frame");
        let grid_intersection = build_projected_domain(
            &lat,
            &lon,
            &ProjectedDomainBuildOptions::from_bounds((0.1, 0.9, 0.0, 1.0), 1.0)
                .with_projection(ProjectionSpec::Geographic)
                .with_geographic_grid_intersection_frame((0.1, 0.9, 0.0, 1.0)),
        )
        .expect("grid intersection frame");

        assert!(
            grid_intersection.extent.x_max - grid_intersection.extent.x_min
                < rectangle.extent.x_max - rectangle.extent.x_min
        );
    }

    #[test]
    fn projected_frames_do_not_geographically_clip_basemap_linework() {
        let bounds = GeographicBounds::new(-125.0, -110.0, 30.0, 49.0);

        assert!(basemap_geographic_clip(ProjectedFrameSource::GeographicBounds(bounds)).is_none());
        assert!(
            basemap_geographic_clip(ProjectedFrameSource::GeographicGridIntersection(bounds))
                .is_none()
        );
    }

    #[test]
    fn projected_segments_clip_to_bbox_edges() {
        let bbox = (0.0, 10.0, 0.0, 10.0);

        let clipped = clip_projected_segment_to_bbox((-5.0, 5.0), (5.0, 5.0), bbox)
            .expect("segment should enter bbox");
        assert!((clipped.0.0 - 0.0).abs() < 1.0e-9);
        assert!((clipped.0.1 - 5.0).abs() < 1.0e-9);
        assert!((clipped.1.0 - 5.0).abs() < 1.0e-9);
        assert!((clipped.1.1 - 5.0).abs() < 1.0e-9);

        let clipped = clip_projected_segment_to_bbox((5.0, 5.0), (15.0, 5.0), bbox)
            .expect("segment should exit bbox");
        assert!((clipped.0.0 - 5.0).abs() < 1.0e-9);
        assert!((clipped.0.1 - 5.0).abs() < 1.0e-9);
        assert!((clipped.1.0 - 10.0).abs() < 1.0e-9);
        assert!((clipped.1.1 - 5.0).abs() < 1.0e-9);

        assert!(clip_projected_segment_to_bbox((-5.0, -5.0), (-1.0, -1.0), bbox).is_none());
    }

    #[test]
    fn geographic_crop_bounds_can_cross_antimeridian() {
        let lat = vec![-20.0, -18.0, -20.0, -18.0, 0.0, 0.0, 40.0, -40.0];
        let lon = vec![176.0, 178.0, -179.0, -178.0, -60.0, 30.0, 120.0, -100.0];
        let cropped = build_projected_domain(
            &lat,
            &lon,
            &ProjectedDomainBuildOptions::from_bounds((176.0, -178.0, -22.0, -15.0), 1.5)
                .with_projection(ProjectionSpec::Geographic),
        )
        .expect("cropped antimeridian domain");

        assert!(
            cropped.extent.x_max - cropped.extent.x_min < 20.0,
            "antimeridian crop should not frame the whole globe: {:?}",
            cropped.extent
        );
    }

    #[test]
    fn global_geographic_bounds_center_on_greenwich() {
        let bounds = GeographicBounds::new(-180.0, 179.999, -90.0, 90.0);

        assert_eq!(bounds.center_longitude(), 0.0);
        assert!(bounds.contains(0.0, 180.0));
        assert!(bounds.contains(0.0, -179.75));
        assert!(bounds.contains(0.0, 0.0));
    }

    #[test]
    fn global_robinson_frame_is_centered_on_world_silhouette() {
        let mut lat = Vec::new();
        let mut lon = Vec::new();
        for row_lat in [-85.0_f32, -60.0, -30.0, 0.0, 30.0, 60.0, 85.0] {
            for col_lon in (-180..=180).step_by(30) {
                lat.push(row_lat);
                lon.push(col_lon as f32);
            }
        }

        let domain = build_projected_domain(
            &lat,
            &lon,
            &ProjectedDomainBuildOptions::from_bounds((-180.0, 180.0, -85.0, 85.0), 16.0 / 9.0)
                .with_projection(ProjectionSpec::Robinson {
                    central_meridian_deg: 0.0,
                }),
        )
        .expect("global Robinson domain should build");

        let center_x = (domain.extent.x_min + domain.extent.x_max) / 2.0;
        let width = domain.extent.x_max - domain.extent.x_min;
        assert!(
            center_x.abs() < width * 0.01,
            "global Robinson frame should be centered, got extent {:?}",
            domain.extent
        );
    }

    #[test]
    fn basemap_densification_takes_short_antimeridian_path() {
        assert_eq!(
            densified_lonlat_segment_steps(179.0, 0.0, -179.0, 0.0, 1.0),
            2
        );
        assert!((interpolate_longitude(179.0, -179.0, 0.5).abs() - 180.0).abs() < 1.0e-9);
    }

    #[test]
    fn projected_ring_densification_adds_curve_support_points() {
        let projector = ProjectionSpec::Robinson {
            central_meridian_deg: 0.0,
        }
        .build_projector(None, None, &[0.0, 10.0], &[0.0, 20.0])
        .expect("projector");
        let ring = vec![(0.0, 0.0), (20.0, 10.0)];
        let projected = project_densified_ring(projector, &ring, 2.0);

        assert_eq!(projected.len(), 1);
        assert!(projected[0].len() > ring.len());
    }

    fn mercator_gulf_projector() -> ProjectionProjector {
        // The 2.8.8 acceptance CY1 d01 grid: Mercator, stand_lon -89.3, so
        // the projection is cut at 90.7E.
        ProjectionSpec::Mercator {
            latitude_of_true_scale_deg: 24.4,
            central_meridian_deg: -89.3,
        }
        .build_projector(None, None, &[15.0, 33.0], &[-100.0, -78.0])
        .expect("projector")
    }

    /// The width of the projected plane: from just west of the cut at 90.7E
    /// (its east edge) to just east of it (its west edge).
    fn projected_plane_width(projector: ProjectionProjector) -> f64 {
        projector.project(0.0, 90.7 - 1.0e-6).0 - projector.project(0.0, 90.7 + 1.0e-6).0
    }

    fn ring_x_span(ring: &[(f64, f64)]) -> f64 {
        let min = ring.iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
        let max = ring.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max);
        max - min
    }

    /// D-08: a ring straddling the projection cut must not come back as one
    /// ring spanning the whole plane (it drew a full-width lake band).
    #[test]
    fn ring_straddling_the_projection_cut_is_split_at_the_cut() {
        let projector = mercator_gulf_projector();
        assert!((projector.seam_longitude_deg().unwrap() - 90.7).abs() < 1.0e-9);
        // A lake-sized box across 90.7E at the latitude of Nam Co.
        let ring = vec![
            (90.2, 30.5),
            (91.0, 30.5),
            (91.0, 30.9),
            (90.2, 30.9),
            (90.2, 30.5),
        ];
        let pieces = project_densified_ring(projector, &ring, 1.0);
        assert_eq!(pieces.len(), 2, "one piece per side of the cut");
        let world = projected_plane_width(projector);
        for piece in &pieces {
            assert!(
                ring_x_span(piece) < 0.01 * world,
                "a piece spans {} of a {} wide plane",
                ring_x_span(piece),
                world
            );
            let all_east = piece.iter().all(|p| p.0 > 0.0);
            let all_west = piece.iter().all(|p| p.0 < 0.0);
            assert!(all_east || all_west, "a piece stays on one side: {piece:?}");
        }
    }

    #[test]
    fn mesh_ring_projection_keeps_indices_and_splits_rotated_seam() {
        let lat = [15.0, 33.0];
        let lon = [-100.0, -78.0];
        let options = ProjectedMapBuildOptions::full_domain(1.5).with_projection(
            ProjectionSpec::Mercator {
                latitude_of_true_scale_deg: 24.4,
                central_meridian_deg: -89.3,
            });
        // Input rings use (lat, lon), unlike the private basemap helper.
        let rings = vec![
            vec![(30.5, 90.2), (30.5, 91.0), (30.9, 91.0), (30.9, 90.2)],
            vec![(30.0, -90.4), (30.0, -89.7), (30.35, -89.7), (30.35, -90.4)],
        ];
        let pieces = project_geographic_rings_with_options(&lat, &lon, &options, &rings)
            .expect("mesh ring projection");
        assert_eq!(pieces.iter().map(|(i, _)| *i).collect::<Vec<_>>(), vec![0, 0, 1]);
        let world = projected_plane_width(mercator_gulf_projector());
        for (_, ring) in &pieces[..2] {
            assert!(ring_x_span(ring) < world * 0.01, "seam cell spans the map");
        }
        assert_eq!(pieces[2].1,
            project_geographic_points_with_options(&lat, &lon, &options, &rings[1])
                .expect("unchanged off-seam cell"));
    }

    #[test]
    fn ring_clear_of_the_projection_cut_is_projected_unchanged() {
        let projector = mercator_gulf_projector();
        // Lake Pontchartrain-sized box, nowhere near 90.7E.
        let ring = vec![
            (-90.4, 30.0),
            (-89.7, 30.0),
            (-89.7, 30.35),
            (-90.4, 30.35),
            (-90.4, 30.0),
        ];
        let pieces = project_densified_ring(projector, &ring, 1.0);
        assert_eq!(pieces.len(), 1);
        let expected: Vec<(f64, f64)> = ring
            .iter()
            .map(|&(lon, lat)| projector.project(lat, lon))
            .collect();
        assert_eq!(pieces[0], expected);
    }

    #[test]
    fn polar_stereographic_has_no_cut() {
        let projector = ProjectionSpec::PolarStereographic {
            true_latitude_deg: 60.0,
            central_meridian_deg: -105.0,
            south_pole_on_projection_plane: false,
        }
        .build_projector(None, None, &[50.0, 70.0], &[-120.0, -90.0])
        .expect("projector");
        assert!(projector.seam_longitude_deg().is_none());
    }

    /// The widest horizontal step between consecutive vertices of a ring,
    /// its implicit closing edge included.  A ring that jumps across the
    /// projection cut has one step about as wide as the whole plane.
    fn ring_widest_x_step(ring: &[(f64, f64)]) -> f64 {
        (0..ring.len())
            .map(|i| (ring[(i + 1) % ring.len()].0 - ring[i].0).abs())
            .fold(0.0, f64::max)
    }

    /// D-08 end to end: no basemap ring built for the CY1 d01 frame jumps
    /// across the projection cut.  Before the fix the Nam Co and Yamdrok Tso
    /// rings (and the ocean ring) each carried an edge spanning the whole
    /// Mercator plane, and the lake rings filled as full-width bands.
    #[test]
    fn gulf_mercator_basemap_has_no_plane_spanning_lake_rings() {
        let mut lat = Vec::new();
        let mut lon = Vec::new();
        for row in 0..19 {
            for col in 0..26 {
                lat.push(15.0 + row as f32);
                lon.push(-101.0 + col as f32);
            }
        }
        let projected = build_projected_map_with_options(
            &lat,
            &lon,
            &ProjectedMapBuildOptions {
                domain: ProjectedDomainBuildOptions::full_domain(1.4).with_projection(
                    ProjectionSpec::Mercator {
                        latitude_of_true_scale_deg: 24.4,
                        central_meridian_deg: -89.3,
                    },
                ),
                basemap: Some(ProjectedBasemapBuildOptions::default()),
            },
        )
        .expect("map");
        let projector = mercator_gulf_projector();
        let plane = projected_plane_width(projector);
        assert!(!projected.polygons.is_empty(), "the frame has lakes to fill");
        for polygon in &projected.polygons {
            for ring in &polygon.rings {
                assert!(
                    ring_widest_x_step(ring) < 0.5 * plane,
                    "a basemap ring steps {:.0} m sideways on a {:.0} m wide plane",
                    ring_widest_x_step(ring),
                    plane
                );
            }
        }
    }

    #[test]
    fn hrrr_like_crop_outside_footprint_errors_instead_of_framing_full_domain() {
        let (lat, lon) = sample_lat_lon();
        let err = build_projected_domain(
            &lat,
            &lon,
            &ProjectedDomainBuildOptions::from_bounds((8.0, 15.0, 45.0, 52.0), 1.5)
                .with_projection(ProjectionSpec::Geographic),
        )
        .expect_err("outside HRRR-like footprint should error");

        assert!(
            err.to_string().contains("does not intersect"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn rap_like_crop_outside_footprint_errors_instead_of_framing_full_domain() {
        let lat = vec![20.0, 20.0, 55.0, 55.0, 35.0, 45.0];
        let lon = vec![-135.0, -60.0, -135.0, -60.0, -100.0, -80.0];
        let err = build_projected_domain(
            &lat,
            &lon,
            &ProjectedDomainBuildOptions::from_bounds((120.0, 150.0, -40.0, -20.0), 1.5)
                .with_projection(ProjectionSpec::Geographic),
        )
        .expect_err("outside RAP-like footprint should error");

        assert!(
            err.to_string().contains("does not intersect"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn gfs_like_global_crop_inside_footprint_still_builds() {
        let lat = vec![60.0, 60.0, 30.0, 30.0, 0.0, 0.0, -30.0, -30.0];
        let lon = vec![-130.0, -70.0, -120.0, -80.0, 0.0, 90.0, 120.0, -120.0];
        let cropped = build_projected_domain(
            &lat,
            &lon,
            &ProjectedDomainBuildOptions::from_bounds((-125.0, -75.0, 25.0, 50.0), 1.5)
                .with_projection(ProjectionSpec::Geographic),
        )
        .expect("GFS-like in-footprint crop should build");

        assert!(cropped.extent.x_min < cropped.extent.x_max);
        assert!(cropped.extent.y_min < cropped.extent.y_max);
    }

    #[test]
    fn projected_map_builder_can_skip_basemap_for_reusable_domain_scaffolds() {
        let (lat, lon) = sample_lat_lon();
        let projected = build_projected_map_with_options(
            &lat,
            &lon,
            &ProjectedMapBuildOptions::full_domain(1.4)
                .with_projection(ProjectionSpec::Geographic)
                .without_basemap(),
        )
        .expect("projected map");

        assert!(projected.lines.is_empty());
        assert!(projected.polygons.is_empty());
    }

    #[test]
    fn projected_map_rotation_transforms_domain_and_basemap_together() {
        let projected = ProjectedMap {
            projected_x: vec![0.0, 2.0],
            projected_y: vec![0.0, 0.0],
            extent: ProjectedExtent {
                x_min: 0.0,
                x_max: 2.0,
                y_min: 0.0,
                y_max: 2.0,
            },
            lines: vec![ProjectedLineOverlay {
                points: vec![(0.0, 1.0), (2.0, 1.0)],
                color: Color::BLACK,
                width: 1,
                role: crate::presentation::LineworkRole::Generic,
            }],
            polygons: vec![ProjectedPolygonFill {
                rings: vec![vec![(1.0, 0.0), (2.0, 1.0)]],
                color: Color::WHITE,
                role: crate::presentation::PolygonRole::Generic,
            }],
            inverse_raster_projection: None,
        }
        .rotated_degrees(90.0);

        assert!((projected.projected_x[0] - 2.0).abs() < 1.0e-9);
        assert!((projected.projected_y[0] - 0.0).abs() < 1.0e-9);
        assert!((projected.projected_x[1] - 2.0).abs() < 1.0e-9);
        assert!((projected.projected_y[1] - 2.0).abs() < 1.0e-9);
        assert!((projected.lines[0].points[0].0 - 1.0).abs() < 1.0e-9);
        assert!((projected.lines[0].points[1].0 - 1.0).abs() < 1.0e-9);
        assert!((projected.extent.x_min - 0.0).abs() < 1.0e-9);
        assert!((projected.extent.x_max - 2.0).abs() < 1.0e-9);
        assert!((projected.extent.y_min - 0.0).abs() < 1.0e-9);
        assert!((projected.extent.y_max - 2.0).abs() < 1.0e-9);
    }

    #[test]
    fn broad_and_global_graticules_are_opt_in() {
        assert!(!DEFAULT_BASEMAP_GRATICULE);
        assert!(!parse_basemap_graticule_flag(""));
        assert!(!parse_basemap_graticule_flag("false"));
        assert!(parse_basemap_graticule_flag("true"));
        assert!(parse_basemap_graticule_flag("on"));
        assert!(!subtle_graticule_enabled(BasemapDetail::Regional));
    }

    #[test]
    fn projected_map_split_preserves_domain_and_basemap_layers() {
        let projected = ProjectedMap {
            projected_x: vec![0.0, 1.0],
            projected_y: vec![0.0, 1.0],
            extent: ProjectedExtent {
                x_min: 0.0,
                x_max: 1.0,
                y_min: 0.0,
                y_max: 1.0,
            },
            lines: vec![ProjectedLineOverlay {
                points: vec![(0.0, 0.0), (1.0, 1.0)],
                color: Color::BLACK,
                width: 2,
                role: crate::presentation::LineworkRole::Generic,
            }],
            polygons: vec![ProjectedPolygonFill {
                rings: vec![vec![(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]],
                color: Color::WHITE,
                role: crate::presentation::PolygonRole::Generic,
            }],
            inverse_raster_projection: None,
        };

        let (domain, basemap) = projected.split();
        assert_eq!(domain.x, vec![0.0, 1.0]);
        assert_eq!(basemap.lines.len(), 1);
        assert_eq!(basemap.polygons.len(), 1);
    }
}
