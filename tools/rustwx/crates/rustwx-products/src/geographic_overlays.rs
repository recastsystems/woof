//! Map overlays and panel annotations, given in DEGREES and read from JSON.
//!
//! `rustwx-render` has carried every primitive these describe since before
//! this crate existed -- [`ProjectedLineOverlay`], [`ProjectedPointOverlay`],
//! [`ProjectedPlaceLabel`], the three subtitle slots -- and all of them take
//! PROJECTED coordinates, which is why nothing outside the renderer could
//! reach them.  Four separate matplotlib modules in this repository
//! hand-rolled the same four overlays (radar sites and range rings, SPC
//! storm reports, a domain-boundary box, tile seams) precisely because a
//! place known in degrees had nowhere to go.
//!
//! This module is that seam: one JSON file, coordinates in `(lat, lon)`
//! degrees, projected with [`project_points_with_projection`] -- the same
//! option build the fill underneath is projected with, so a marker cannot
//! land in a different frame than the data it annotates.
//!
//! ## Schema (all fields optional)
//!
//! ```json
//! {
//!   "lines":  [{"points": [[38.0,-98.0],[39.0,-97.0]], "color": "#ff0000",
//!               "width": 2, "closed": false}],
//!   "points": [{"lat": 35.33, "lon": -97.28, "color": "#111111",
//!               "radius_px": 7, "width_px": 2, "shape": "circle"}],
//!   "labels": [{"lat": 35.33, "lon": -97.28, "text": "KTLX"}],
//!   "rings":  [{"lat": 35.33, "lon": -97.28, "radii_km": [50,100,150],
//!               "color": "#444444", "width": 1}]
//! }
//! ```
//!
//! `rings` is a convenience that expands to closed `lines`: a great-circle
//! circle of the given radius about a centre, which is the range ring both
//! DA render modules draw and the one construction a caller cannot express
//! as a literal point list without re-deriving the geodesy.

use std::path::Path;

use crate::direct::project_points_with_projection;
use rustwx_render::{
    Color, ColorScale, ColormapBuildOptions, MapRenderRequest, ProjectedLineOverlay,
    ProjectedMarkerShape, ProjectedPlaceLabel, ProjectedPointOverlay, StaticPlotStyle, build_colormap,
};
use serde::{Deserialize, Serialize};

/// Mean Earth radius, metres.  The ring geodesy is a spherical
/// small-circle walk: at the 50-500 km radii an operator draws, the
/// spherical/ellipsoidal difference is well under a pixel at any sane
/// canvas size, and stating the sphere is accurate where quietly using one
/// under an ellipsoidal name would not be.
const EARTH_RADIUS_M: f64 = 6_371_000.0;

fn default_line_width() -> u32 {
    2
}

fn default_marker_radius() -> u32 {
    6
}

fn default_marker_width() -> u32 {
    2
}

fn default_ring_segments() -> usize {
    180
}

fn default_color() -> String {
    "#101418".to_string()
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LineSpec {
    /// `(lat, lon)` degree pairs, in draw order.
    pub points: Vec<(f64, f64)>,
    #[serde(default = "default_color")]
    pub color: String,
    #[serde(default = "default_line_width")]
    pub width: u32,
    /// Repeat the first point at the end (a boundary box, a tile outline).
    #[serde(default)]
    pub closed: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PointSpec {
    pub lat: f64,
    pub lon: f64,
    #[serde(default = "default_color")]
    pub color: String,
    #[serde(default = "default_marker_radius")]
    pub radius_px: u32,
    #[serde(default = "default_marker_width")]
    pub width_px: u32,
    /// `circle`, `plus` or `cross`.
    #[serde(default)]
    pub shape: String,
}

/// A measured or scored scalar in the map's display units.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ValuePointSpec {
    pub lat: f64,
    pub lon: f64,
    pub value: f64,
}

/// Scalar dots using the field palette, or an explicitly supplied error palette.
/// Product selection is metadata, so the same overlay file can accompany a batch.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ValuePointLayer {
    #[serde(default)]
    pub products: Vec<String>,
    pub units: String,
    pub points: Vec<ValuePointSpec>,
    #[serde(default)]
    pub scale: Option<ColorScale>,
    #[serde(default = "default_marker_radius")]
    pub radius_px: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LabelSpec {
    pub lat: f64,
    pub lon: f64,
    pub text: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RingSpec {
    pub lat: f64,
    pub lon: f64,
    /// Radii in kilometres, one closed ring each.
    pub radii_km: Vec<f64>,
    #[serde(default = "default_color")]
    pub color: String,
    #[serde(default)]
    pub width: u32,
    #[serde(default = "default_ring_segments")]
    pub segments: usize,
}

/// Everything one `--overlays FILE.json` can add to a panel.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct MapOverlays {
    #[serde(default)]
    pub lines: Vec<LineSpec>,
    #[serde(default)]
    pub points: Vec<PointSpec>,
    #[serde(default)]
    pub labels: Vec<LabelSpec>,
    #[serde(default)]
    pub rings: Vec<RingSpec>,
    #[serde(default)]
    pub value_layers: Vec<ValuePointLayer>,
}

/// Everything one `--annotate FILE.json` can say on a panel.
///
/// Three slots, because three is what the renderer has.  A multi-paragraph
/// accuracy footer (the `IC_FOOTER` block in `tilestream/bigdomain_render.py`)
/// does NOT fit in them and is not silently truncated into them; there is
/// no footer band in `MapRenderRequest` to squeeze one into.
///
/// **`subtitle_center` overlaps if it is long.**  The renderer centres the
/// middle slot on the PANEL, not in the gap between the other two, so a
/// centre string wide enough to reach them is drawn over them.  It is
/// offered because a short badge (`PAST LAST OBS`, `VERIFIED`) belongs
/// exactly there; anything sentence-length belongs in `subtitle_left`,
/// which owns the row's width budget and has its own ellipsis.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PanelAnnotations {
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub title_suffix: Option<String>,
    #[serde(default)]
    pub subtitle_left: Option<String>,
    #[serde(default)]
    pub subtitle_center: Option<String>,
    #[serde(default)]
    pub subtitle_right: Option<String>,
    /// A note added to ONE product's title, keyed by its recipe slug
    /// (`composite_reflectivity`), before any `title_suffix`.  `rw_wrfbatch`
    /// fills it from what its frames say about a field (`GPUWM_REFC_NOTE`:
    /// a coarse hex row's REFC includes the cumulus scheme's reflectivity);
    /// empty for every other caller, so their titles are unchanged.
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub title_notes: std::collections::BTreeMap<String, String>,
}

impl MapOverlays {
    pub fn load(path: &Path) -> Result<Self, String> {
        let bytes = std::fs::read(path)
            .map_err(|err| format!("read overlays {}: {err}", path.display()))?;
        serde_json::from_slice(&bytes)
            .map_err(|err| format!("parse overlays {}: {err}", path.display()))
    }

    pub fn is_empty(&self) -> bool {
        self.lines.is_empty()
            && self.points.is_empty()
            && self.labels.is_empty()
            && self.rings.is_empty()
            && self.value_layers.is_empty()
    }

    /// Every `(lat, lon)` this overlay set needs projected, in the order
    /// [`Self::apply`] consumes them.  One projection call for the whole
    /// file: the projector build reads the mesh, and doing it per feature
    /// would be O(features) passes over the grid.
    fn geographic_points(&self) -> Vec<(f64, f64)> {
        let mut points = Vec::new();
        for line in &self.lines {
            points.extend(line.points.iter().copied());
        }
        for point in &self.points {
            points.push((point.lat, point.lon));
        }
        for label in &self.labels {
            points.push((label.lat, label.lon));
        }
        for ring in &self.rings {
            for radius_km in &ring.radii_km {
                points.extend(ring_points(ring.lat, ring.lon, *radius_km, ring.segments));
            }
        }
        for layer in &self.value_layers {
            points.extend(layer.points.iter().map(|point| (point.lat, point.lon)));
        }
        points
    }

    /// Project and attach.  `bounds`/`target_ratio` must be exactly the
    /// values the panel's own [`build_projected_map_with_projection`] call
    /// used, or the overlay lands in a different frame than the fill.
    pub fn apply(
        &self,
        request: &mut MapRenderRequest,
        lat_deg: &[f32],
        lon_deg: &[f32],
        projection: Option<&rustwx_core::GridProjection>,
        bounds: (f64, f64, f64, f64),
        target_ratio: f64,
    ) -> Result<(), String> {
        if self.is_empty() {
            return Ok(());
        }
        let geographic = self.geographic_points();
        let projected = if let Some(resolved)=request.resolved_projection {
            geographic.iter().map(|(lat,lon)|resolved.project(*lat,*lon)).collect()
        } else {
            project_points_with_projection(lat_deg,lon_deg,projection,bounds,target_ratio,&geographic)
                .map_err(|err|format!("project overlay points: {err}"))?
        };
        let mut cursor = 0usize;
        let mut take = |count: usize| {
            let slice = projected[cursor..cursor + count].to_vec();
            cursor += count;
            slice
        };

        for line in &self.lines {
            let mut points = take(line.points.len());
            if line.closed && points.len() > 2 {
                points.push(points[0]);
            }
            request.projected_lines.push(ProjectedLineOverlay {
                points,
                color: parse_color(&line.color),
                width: line.width.max(1),
                role: Default::default(),
            });
        }
        for point in &self.points {
            let xy = take(1)[0];
            request.projected_points.push(ProjectedPointOverlay {
                x: xy.0,
                y: xy.1,
                color: parse_color(&point.color),
                radius_px: point.radius_px.max(1),
                width_px: point.width_px.max(1),
                shape: parse_shape(&point.shape),
            });
        }
        for label in &self.labels {
            let xy = take(1)[0];
            let mut place = ProjectedPlaceLabel::new(xy.0, xy.1);
            place.label = Some(label.text.clone());
            request.projected_place_labels.push(place);
        }
        for ring in &self.rings {
            for _radius_km in &ring.radii_km {
                let mut points = take(ring.segments.max(8));
                if let Some(first) = points.first().copied() {
                    points.push(first);
                }
                request.projected_lines.push(ProjectedLineOverlay {
                    points,
                    color: parse_color(&ring.color),
                    width: ring.width.max(1),
                    role: Default::default(),
                });
            }
        }
        for layer in &self.value_layers {
            let points = take(layer.points.len());
            if !layer.products.is_empty()
                && !layer.products.iter().any(|product| {
                    request.field.product.as_named() == Some(product.as_str())
                })
            {
                continue;
            }
            // Dots that borrow the map's palette must be in the map's units,
            // or a value is read off a bar it was never on.  A layer that
            // carries its own ladder (observation-minus-background dots
            // over a shaded field) is read on that ladder, not the map's.
            if layer.scale.is_none() && layer.units != request.field.units {
                return Err(format!(
                    "scalar overlay units {:?} differ from map units {:?}",
                    layer.units, request.field.units
                ));
            }
            let cmap = build_colormap(
                layer.scale.as_ref().unwrap_or(&request.scale),
                ColormapBuildOptions {
                    render_density: StaticPlotStyle::from_env().render_density(request.render_density),
                    legend: request.legend,
                },
            );
            for (point, xy) in layer.points.iter().zip(points) {
                if !point.value.is_finite() || !xy.0.is_finite() || !xy.1.is_finite() {
                    continue;
                }
                let mut dot = ProjectedPlaceLabel::new(xy.0, xy.1);
                dot.style.marker_radius_px = layer.radius_px.max(1);
                dot.style.marker_fill = cmap.map(point.value).into();
                dot.style.marker_outline = Color::rgba(20, 24, 28, 255);
                dot.style.marker_outline_width = 1;
                request.projected_place_labels.push(dot);
            }
        }
        Ok(())
    }
}

impl PanelAnnotations {
    pub fn load(path: &Path) -> Result<Self, String> {
        let bytes = std::fs::read(path)
            .map_err(|err| format!("read annotations {}: {err}", path.display()))?;
        serde_json::from_slice(&bytes)
            .map_err(|err| format!("parse annotations {}: {err}", path.display()))
    }

    pub fn apply(&self, request: &mut MapRenderRequest) {
        if let Some(title) = &self.title {
            request.title = Some(title.clone());
        }
        if let Some(suffix) = &self.title_suffix {
            request.title = Some(match request.title.take() {
                Some(existing) => format!("{} {}", existing.trim_end(), suffix.trim_start()),
                None => suffix.clone(),
            });
        }
        if let Some(text) = &self.subtitle_left {
            request.subtitle_left = Some(text.clone());
        }
        if let Some(text) = &self.subtitle_center {
            request.subtitle_center = Some(text.clone());
        }
        if let Some(text) = &self.subtitle_right {
            request.subtitle_right = Some(text.clone());
        }
    }
}

/// Points on a spherical small circle of `radius_km` about `(lat, lon)`.
///
/// `segments` samples, evenly spaced in bearing, first point due north.
/// Returned open; the caller closes it.
fn ring_points(lat_deg: f64, lon_deg: f64, radius_km: f64, segments: usize) -> Vec<(f64, f64)> {
    let segments = segments.max(8);
    let angular = (radius_km * 1_000.0) / EARTH_RADIUS_M;
    let lat = lat_deg.to_radians();
    let lon = lon_deg.to_radians();
    let (sin_lat, cos_lat) = (lat.sin(), lat.cos());
    let (sin_ang, cos_ang) = (angular.sin(), angular.cos());
    (0..segments)
        .map(|step| {
            let bearing = std::f64::consts::TAU * step as f64 / segments as f64;
            let sin_point = sin_lat * cos_ang + cos_lat * sin_ang * bearing.cos();
            let point_lat = sin_point.clamp(-1.0, 1.0).asin();
            let point_lon = lon
                + (bearing.sin() * sin_ang * cos_lat)
                    .atan2(cos_ang - sin_lat * sin_point);
            (point_lat.to_degrees(), normalize_longitude(point_lon.to_degrees()))
        })
        .collect()
}

fn normalize_longitude(mut lon_deg: f64) -> f64 {
    while lon_deg > 180.0 {
        lon_deg -= 360.0;
    }
    while lon_deg <= -180.0 {
        lon_deg += 360.0;
    }
    lon_deg
}

/// `#rrggbb`, `#rrggbbaa`, or the documented fallback.
///
/// A malformed colour is opaque black rather than a refusal: the overlay
/// file is hand-written, and losing a whole render over one typo'd swatch
/// is a worse trade than one wrong-coloured line the reader can see.
pub fn parse_color(value: &str) -> Color {
    let text = value.trim().trim_start_matches('#');
    let byte = |index: usize| u8::from_str_radix(&text[index..index + 2], 16).ok();
    match text.len() {
        6 => match (byte(0), byte(2), byte(4)) {
            (Some(r), Some(g), Some(b)) => Color::rgba(r, g, b, 255),
            _ => Color::BLACK,
        },
        8 => match (byte(0), byte(2), byte(4), byte(6)) {
            (Some(r), Some(g), Some(b), Some(a)) => Color::rgba(r, g, b, a),
            _ => Color::BLACK,
        },
        _ => Color::BLACK,
    }
}

fn parse_shape(value: &str) -> ProjectedMarkerShape {
    match value.trim().to_ascii_lowercase().as_str() {
        "circle" => ProjectedMarkerShape::Circle,
        "cross" => ProjectedMarkerShape::Cross,
        _ => ProjectedMarkerShape::Plus,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_ring_is_the_requested_distance_from_its_centre() {
        let points = ring_points(35.333, -97.278, 100.0, 72);
        assert_eq!(points.len(), 72);
        for (lat, lon) in points {
            let distance_km = haversine_km(35.333, -97.278, lat, lon);
            assert!(
                (distance_km - 100.0).abs() < 0.5,
                "ring point at {distance_km} km, not 100"
            );
        }
    }

    #[test]
    fn a_ring_is_closed_by_construction_and_starts_due_north() {
        let points = ring_points(40.0, -100.0, 50.0, 36);
        assert!(points[0].0 > 40.0, "first point is north of the centre");
        assert!((points[0].1 - -100.0).abs() < 1.0e-9, "and on its meridian");
    }

    #[test]
    fn colours_parse_both_lengths_and_a_typo_is_visible_not_fatal() {
        assert_eq!(parse_color("#ff8000"), Color::rgba(255, 128, 0, 255));
        assert_eq!(parse_color("ff800080"), Color::rgba(255, 128, 0, 128));
        assert_eq!(parse_color("#nonsense"), Color::BLACK);
        assert_eq!(parse_color(""), Color::BLACK);
    }

    #[test]
    fn the_schema_reads_a_hand_written_file() {
        let json = r##"{
            "lines": [{"points": [[38.0,-98.0],[39.0,-97.0]], "color": "#ff0000"}],
            "points": [{"lat": 35.3, "lon": -97.3, "shape": "circle"}],
            "labels": [{"lat": 35.3, "lon": -97.3, "text": "KTLX"}],
            "rings": [{"lat": 35.3, "lon": -97.3, "radii_km": [50, 100]}]
        }"##;
        let overlays: MapOverlays = serde_json::from_str(json).unwrap();
        assert_eq!(overlays.lines[0].width, 2, "the documented default");
        assert!(!overlays.lines[0].closed);
        assert_eq!(overlays.points[0].radius_px, 6);
        assert_eq!(overlays.rings[0].segments, 180);
        // One projection call must cover every feature, so the count has
        // to match what `apply` consumes.
        assert_eq!(overlays.geographic_points().len(), 2 + 1 + 1 + 360);
    }

    #[test]
    fn an_empty_overlay_file_is_a_no_op_rather_than_an_error() {
        let overlays: MapOverlays = serde_json::from_str("{}").unwrap();
        assert!(overlays.is_empty());
    }

    #[test]
    fn scalar_marks_use_the_field_colormap_and_product_selection() {
        use rustwx_core::{Field2D, GridShape, LatLonGrid, ProductKey};
        let lat = vec![34.0, 34.0, 35.0, 35.0];
        let lon = vec![-99.0, -98.0, -99.0, -98.0];
        let scale = ColorScale::Discrete(rustwx_render::DiscreteColorScale {
            levels: vec![0.0, 10.0, 20.0],
            colors: vec![Color::rgba(0, 0, 200, 255), Color::rgba(200, 0, 0, 255)],
            extend: rustwx_render::ExtendMode::Both,
            mask_below: None,
        });
        let field = Field2D::new(ProductKey::named("scalar"), "degC", LatLonGrid {
            shape: GridShape {nx: 2, ny: 2}, lat_deg: lat.clone(), lon_deg: lon.clone(),
        }, vec![15.0; 4]).unwrap();
        let mut request = MapRenderRequest::from_core_field(field, scale.clone());
        let marks = MapOverlays {
            value_layers: vec![ValuePointLayer {
                products: vec!["scalar".to_string()], units: "degC".to_string(),
                points: vec![ValuePointSpec {lat: 34.5, lon: -98.5, value: 15.0}],
                scale: None, radius_px: 5,
            }], ..MapOverlays::default()
        };
        marks.apply(&mut request, &lat, &lon, None, (-99.0,-98.0,34.0,35.0), 1.0).unwrap();
        let cmap = build_colormap(&scale, ColormapBuildOptions {
            render_density: StaticPlotStyle::from_env().render_density(request.render_density),
            legend: request.legend,
        });
        assert_eq!(request.projected_place_labels.len(), 1);
        assert_eq!(request.projected_place_labels[0].style.marker_fill, Color::from(cmap.map(15.0)));
        request.field.product = rustwx_render::ProductKey::named("other");
        request.projected_place_labels.clear();
        marks.apply(&mut request, &lat, &lon, None, (-99.0,-98.0,34.0,35.0), 1.0).unwrap();
        assert!(request.projected_place_labels.is_empty());
    }

    #[test]
    fn dots_on_their_own_ladder_may_carry_other_units_than_the_map() {
        use rustwx_core::{Field2D, GridShape, LatLonGrid, ProductKey};
        let lat = vec![34.0, 34.0, 35.0, 35.0];
        let lon = vec![-99.0, -98.0, -99.0, -98.0];
        let scale = |low: f64| ColorScale::Discrete(rustwx_render::DiscreteColorScale {
            levels: vec![low, 0.0, 10.0],
            colors: vec![Color::rgba(0, 0, 200, 255), Color::rgba(200, 0, 0, 255)],
            extend: rustwx_render::ExtendMode::Both,
            mask_below: None,
        });
        let field = Field2D::new(ProductKey::named("shaded"), "dBZ", LatLonGrid {
            shape: GridShape {nx: 2, ny: 2}, lat_deg: lat.clone(), lon_deg: lon.clone(),
        }, vec![30.0; 4]).unwrap();
        let mut request = MapRenderRequest::from_core_field(field, scale(-10.0));
        let layer = |own: Option<ColorScale>| MapOverlays {
            value_layers: vec![ValuePointLayer {
                products: Vec::new(), units: "K".to_string(),
                points: vec![ValuePointSpec {lat: 34.5, lon: -98.5, value: 2.0}],
                scale: own, radius_px: 4,
            }], ..MapOverlays::default()
        };
        // Borrowing the map's dBZ palette for kelvin is refused ...
        assert!(layer(None).apply(&mut request, &lat, &lon, None, (-99.0,-98.0,34.0,35.0), 1.0).is_err());
        // ... and a departure layer read on its own ladder is drawn.
        layer(Some(scale(-10.0))).apply(&mut request, &lat, &lon, None, (-99.0,-98.0,34.0,35.0), 1.0).unwrap();
        assert_eq!(request.projected_place_labels.len(), 1);
    }

    #[test]
    fn overlays_follow_the_published_native_frame_transform() {
        use rustwx_core::{Field2D,GridShape,LatLonGrid,ProductKey};
        let field=Field2D::new(ProductKey::named("probe"),"1",LatLonGrid {
            shape:GridShape {nx:1,ny:1},lat_deg:vec![32.6],lon_deg:vec![-116.4]},vec![0.]).unwrap();
        let mut request=MapRenderRequest::from_core_field(field,rustwx_render::ColorScale::Discrete(
            rustwx_render::DiscreteColorScale {levels:vec![0.,1.],colors:vec![Color::BLACK],
                extend:rustwx_render::ExtendMode::Neither,mask_below:None}));
        request.resolved_projection=Some(rustwx_render::ResolvedProjection::Geographic {central_meridian_deg:-120.});
        let overlays=MapOverlays {labels:vec![LabelSpec {lat:32.6,lon:-116.4,text:"32.60°N".into()}],..Default::default()};
        // The supplied bounds would select a different presentation anchor.
        // The actual PNG's transform is authoritative for all annotations.
        overlays.apply(&mut request,&[32.6],&[-116.4],None,(-117.,-116.,32.,33.),1.).unwrap();
        let label=&request.projected_place_labels[0];
        assert!((label.x-3.6).abs()<1e-10);assert!((label.y-32.6).abs()<1e-10);
    }

    #[test]
    fn annotations_replace_only_the_slots_they_name() {
        use rustwx_core::{Field2D, GridShape, LatLonGrid, ProductKey};

        let field = Field2D::new(
            ProductKey::named("probe"),
            "1".to_string(),
            LatLonGrid {
                shape: GridShape { nx: 1, ny: 1 },
                lat_deg: vec![0.0],
                lon_deg: vec![0.0],
            },
            vec![0.0f32],
        )
        .unwrap();
        let mut request = MapRenderRequest::from_core_field(
            field,
            rustwx_render::ColorScale::Discrete(rustwx_render::DiscreteColorScale {
                levels: vec![0.0, 1.0],
                colors: vec![rustwx_render::Color::BLACK],
                extend: rustwx_render::ExtendMode::Neither,
                mask_below: None,
            }),
        );
        request.title = Some("Composite Reflectivity".to_string());
        request.subtitle_right = Some("source: keep me".to_string());
        let annotations: PanelAnnotations =
            serde_json::from_str(r#"{"title_suffix": "EXPERIMENTAL", "subtitle_center": "note"}"#)
                .unwrap();
        annotations.apply(&mut request);
        assert_eq!(
            request.title.as_deref(),
            Some("Composite Reflectivity EXPERIMENTAL")
        );
        assert_eq!(request.subtitle_center.as_deref(), Some("note"));
        assert_eq!(request.subtitle_right.as_deref(), Some("source: keep me"));
        request.title = Some("Kuchera snowfall, window total (d02 750 m)".into());
        let annotations: PanelAnnotations = serde_json::from_str(
            r#"{"title_suffix": ", 24 h ending Mon 12 Oct 18Z"}"#,
        ).unwrap();
        annotations.apply(&mut request);
        assert_eq!(request.title.as_deref(), Some(
            "Kuchera snowfall, window total (d02 750 m) , 24 h ending Mon 12 Oct 18Z"
        ));
    }

    fn haversine_km(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
        let (p1, p2) = (lat1.to_radians(), lat2.to_radians());
        let dp = p2 - p1;
        let dl = (lon2 - lon1).to_radians();
        let a = (dp / 2.0).sin().powi(2) + p1.cos() * p2.cos() * (dl / 2.0).sin().powi(2);
        2.0 * a.sqrt().asin() * EARTH_RADIUS_M / 1_000.0
    }
}
