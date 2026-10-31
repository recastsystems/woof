//! Nearest-cell resample weights: the one-time, per-mesh half of the bridge.
//!
//! Method: nearest cell centre on the sphere, by 3-D chord distance through a
//! k-d tree over unit vectors. Nearest-cell is a deliberate choice, not a
//! shortcut. A resampled value is exactly the value the model carried in that
//! cell, so extremes survive the transfer unchanged: a 70 dBZ core stays
//! 70 dBZ, a 35 m/s gust stays 35 m/s. Every smoothing scheme, inverse
//! distance included, pulls maxima down and pushes minima up, which is the
//! wrong trade for products whose whole point is the tail of the
//! distribution.
//!
//! Nothing here is conservative and nothing produced through this path
//! supports an accumulation-budget claim.

use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};
use rayon::prelude::*;

use crate::error::{MpasError, MpasResult};
use crate::window::{Window, MPAS_EARTH_RADIUS_M};

pub const WEIGHTS_SCHEMA: &str = "mpas-port.nearest-cell-render-weights/v1";
pub const RESAMPLE_METHOD: &str = "nearest-cell (spherical k-d tree over unit vectors)";

const RAD: f64 = std::f64::consts::PI / 180.0;

/// How far past pi/2 a mesh latitude may sit and still be a pole cell.
///
/// MPAS writes `latCell` as float32, and a mesh with cells at the poles stores
/// pi/2 as the nearest float32, which is 4.4e-8 rad *above* pi/2. A tolerance
/// tighter than that refuses every polar global mesh, x1.40962 included. One
/// float32 ulp at 1.57 is 1.19e-7 rad, so the bound below is a few ulps: it
/// admits a rounded pole and nothing else. 1e-6 rad is 5.7e-5 degrees, about
/// 6 m on the ground, so no genuinely out-of-range mesh slips through.
const POLE_TOLERANCE_RAD: f64 = std::f64::consts::FRAC_PI_2 + 1.0e-6;

/// What a window point that is off the mesh carries.
///
/// NaN, because that is what the renderer already treats as "draw nothing"
/// (`rustwx-render::colormap::LeveledColormap::map` returns a transparent
/// pixel for it).  A sentinel number would be drawn: it would land somewhere
/// on the colour scale and a reader would read it as weather.
pub const OFF_MESH_FILL: f32 = f32::NAN;

/// How far outside a cell's own spacing a window point may sit and still be
/// counted as on the mesh.  See [`NearestCellWeights::mark_footprint`].
pub const ON_MESH_SPACING_FACTOR: f64 = 1.0;

/// How a mesh's angular units were established.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AngleUnits {
    /// The file declared radians in a `units` attribute.
    Declared,
    /// The file declared nothing and radians were established from the range
    /// the values actually span. Recorded in the emitted frame's provenance.
    InferredFromRange,
}

impl AngleUnits {
    pub fn label(self) -> &'static str {
        match self {
            AngleUnits::Declared => "declared-radians",
            AngleUnits::InferredFromRange => "inferred-radians-from-range",
        }
    }
}

/// Decide whether `latCell`/`lonCell` are radians.
///
/// A declared unit is obeyed: radians pass, anything else is refused, exactly
/// as before. The new case is an *absent* `units` attribute, which the CUDA
/// port's history writer leaves off. Absence is not a licence -- it is
/// resolved against the numbers, and only when they cannot be degrees:
/// a degree-valued mesh reaches |lat| = 90, which is 57x the radian bound of
/// pi/2, so the two interpretations do not overlap for any mesh that spans
/// more than 1.57 degrees of latitude or 6.28 of longitude.
///
/// The residual ambiguity is named rather than hidden: a *degree*-valued mesh
/// confined to roughly 1.57 deg of the equator and 6.28 deg of the prime
/// meridian -- a patch of the Gulf of Guinea -- would be read as radians.
/// No MPAS mesh in this program is regional, let alone that one, and the
/// emitted frame records `inferred-radians-from-range` so a reader can see
/// which branch was taken.
pub fn resolve_angle_units(
    declared: &str,
    max_abs_lat: f64,
    max_abs_lon: f64,
) -> MpasResult<AngleUnits> {
    let units = declared.trim().to_ascii_lowercase();
    if matches!(units.as_str(), "rad" | "radian" | "radians") {
        return Ok(AngleUnits::Declared);
    }
    if !units.is_empty() {
        return Err(MpasError::Refusal(format!(
            "mesh latCell/lonCell units '{declared}' are not radians"
        )));
    }
    const LAT_LIMIT: f64 = std::f64::consts::FRAC_PI_2 + 1.0e-6;
    const LON_LIMIT: f64 = 2.0 * std::f64::consts::PI + 1.0e-6;
    if max_abs_lat <= LAT_LIMIT && max_abs_lon <= LON_LIMIT {
        Ok(AngleUnits::InferredFromRange)
    } else {
        Err(MpasError::Refusal(format!(
            "mesh latCell/lonCell declare no units and the values they span \
             (max |lat| {max_abs_lat}, max |lon| {max_abs_lon}) are not radians \
             (radians bound them by {LAT_LIMIT} and {LON_LIMIT})"
        )))
    }
}

/// Cell centres read from a grid, static, init or history file.
#[derive(Debug, Clone)]
pub struct MeshCoordinates {
    pub latitude_degrees: Vec<f64>,
    pub longitude_degrees: Vec<f64>,
    pub source_path: PathBuf,
    pub source_sha256: String,
    pub angle_units: AngleUnits,
}

impl MeshCoordinates {
    pub fn n_cells(&self) -> usize {
        self.latitude_degrees.len()
    }
}

/// Read `latCell`/`lonCell` and bind the file by digest.
///
/// Longitudes are normalised to `[-180, 180)`. A rotated mesh -- the
/// `grid_rotate` output the port pins -- stores post-rotation earth
/// coordinates directly, so no de-rotation is applied or wanted: the cell
/// centre in the file is where that cell is on the Earth. The converter
/// cross-checks this against the history file's own coordinates before it
/// resamples anything.
pub fn read_mesh_coordinates(path: &Path, expect_sha256: Option<&str>) -> MpasResult<MeshCoordinates> {
    let digest = crate::sha256_file(path)?;
    if let Some(want) = expect_sha256 {
        if digest != want {
            return Err(MpasError::Refusal(format!(
                "mesh {} is sha256 {digest}, expected {want}",
                path.display()
            )));
        }
    }
    let file = netcrust::File::open(path)?;
    let mut declared: Vec<String> = Vec::new();
    for name in ["latCell", "lonCell"] {
        let variable = file.variable(name).ok_or_else(|| {
            MpasError::Refusal(format!("mesh {} has no {name}", path.display()))
        })?;
        declared.push(
            variable
                .attribute("units")
                .and_then(|a| a.as_string())
                .unwrap_or_default()
                .to_string(),
        );
    }
    let latitude = file.read_f64("latCell")?;
    let longitude = file.read_f64("lonCell")?;
    let max_abs_lat = latitude.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let max_abs_lon = longitude.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let mut angle_units = AngleUnits::Declared;
    for (name, text) in ["latCell", "lonCell"].iter().zip(declared.iter()) {
        let resolved = resolve_angle_units(text, max_abs_lat, max_abs_lon).map_err(|e| {
            MpasError::Refusal(format!("mesh {} {name}: {e}", path.display()))
        })?;
        if resolved == AngleUnits::InferredFromRange {
            angle_units = AngleUnits::InferredFromRange;
        }
    }
    if angle_units == AngleUnits::InferredFromRange {
        eprintln!(
            "rw_mpas_convert: advisory: mesh {} declares no units on latCell/lonCell; \
             radians established from the range (max |lat| {max_abs_lat:.6}, max |lon| \
             {max_abs_lon:.6}) and recorded as {} in the emitted frame",
            path.display(),
            AngleUnits::InferredFromRange.label()
        );
    }
    if latitude.len() != longitude.len() || latitude.is_empty() {
        return Err(MpasError::Refusal(
            "mesh latCell/lonCell shapes disagree".to_string(),
        ));
    }
    if latitude.iter().chain(longitude.iter()).any(|v| !v.is_finite()) {
        return Err(MpasError::Refusal(
            "mesh coordinates are not finite".to_string(),
        ));
    }
    if max_abs_lat > POLE_TOLERANCE_RAD {
        return Err(MpasError::Refusal(format!(
            "mesh latitude leaves [-pi/2, pi/2]: max |lat| is {max_abs_lat} rad,              which is {:e} rad past the pole (tolerance {:e})",
            max_abs_lat - std::f64::consts::FRAC_PI_2,
            POLE_TOLERANCE_RAD - std::f64::consts::FRAC_PI_2
        )));
    }
    let deg = 180.0 / std::f64::consts::PI;
    Ok(MeshCoordinates {
        latitude_degrees: latitude.iter().map(|v| v * deg).collect(),
        longitude_degrees: longitude
            .iter()
            .map(|v| (v * deg + 180.0).rem_euclid(360.0) - 180.0)
            .collect(),
        source_path: path.to_path_buf(),
        source_sha256: digest,
        angle_units,
    })
}

// ---------------------------------------------------------------------------
// Per-cell spacing: what a mesh-derived render window is sized from.
// ---------------------------------------------------------------------------

/// Which variable a mesh's per-cell spacing was measured from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpacingSource {
    /// `areaCell`, as hexagon across-flats `sqrt(2A/sqrt(3))`. This is the
    /// metric every published resolution figure in this program is quoted in
    /// (`mesh::derive::Mesh::spacing_m`), so a window derived from it is
    /// sized in the same units the mesh registry describes the mesh in.
    AreaCell,
    /// The mean `dcEdge` over each cell's own edges, for a file that carries
    /// the connectivity but no `areaCell`.
    MeanEdgeLength,
}

impl SpacingSource {
    pub fn label(self) -> &'static str {
        match self {
            SpacingSource::AreaCell => "areaCell hexagon across-flats",
            SpacingSource::MeanEdgeLength => "mean dcEdge",
        }
    }
}

/// One mesh's per-cell horizontal spacing, in metres.
#[derive(Debug, Clone)]
pub struct MeshSpacing {
    pub spacing_metres: Vec<f64>,
    pub source: SpacingSource,
    /// The radius the file's lengths are stored on: 1.0 for a unit-sphere
    /// grid file, 6371229.0 for an earth-scaled static.
    pub sphere_radius: f64,
}

/// Whether a connectivity array counts from 1 or from 0.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexBase {
    Zero,
    One,
}

/// Decide whether `edgesOnCell` counts from 1 or from 0, from the data.
///
/// Not assumed, because assuming is how this goes wrong quietly: an
/// off-by-one read still lands inside `dcEdge` for every cell but the last,
/// so every spacing belongs to a neighbouring edge and the mesh looks fine.
///
/// The test is decisive rather than heuristic. Every edge lies on exactly two
/// cells, so every edge index appears in some cell's ring: a one-based map
/// spans exactly `[1, nEdges]` and a zero-based map spans exactly
/// `[0, nEdges - 1]`. `min` and `max` are taken over VALID slots only -- the
/// padding past `nEdgesOnCell` is 0 or -1 and says nothing. Anything that
/// matches neither span is refused rather than guessed at.
pub fn detect_index_base(min: i64, max: i64, count: usize, name: &str) -> MpasResult<IndexBase> {
    let n = count as i64;
    if min == 1 && max == n {
        return Ok(IndexBase::One);
    }
    if min == 0 && max == n - 1 {
        return Ok(IndexBase::Zero);
    }
    Err(MpasError::Refusal(format!(
        "{name} spans [{min}, {max}] over its valid slots against {count} target(s). Every \
         target is named by some cell, so a one-based map spans exactly [1, {n}] and a \
         zero-based map spans exactly [0, {}]; this matches neither, so nothing in the file \
         says which it is. Reading it the wrong way shifts every lookup by one and still \
         lands in range, so the mesh would measure clean at the wrong spacing",
        n - 1
    )))
}

/// A limited-area mesh's boundary rings: `bdyMaskCell` per cell, 0 for the
/// interior and 1-7 for MPAS's `nBdyLayers` (rings 1-5 the relaxation zone,
/// 6-7 the specified zone).  `None` when the file has no `bdyMaskCell` or
/// every cell is interior (a global mesh), so nothing changes for those.
pub fn read_boundary_rings(path: &Path) -> MpasResult<Option<Vec<i32>>> {
    let file = netcrust::File::open(path)?;
    if file.variable("bdyMaskCell").is_none() {
        return Ok(None);
    }
    let raw = file.read_f64("bdyMaskCell")?;
    if raw
        .iter()
        .any(|v| !v.is_finite() || *v < 0.0 || *v > 7.0 || v.fract() != 0.0)
    {
        return Err(MpasError::Refusal(format!(
            "mesh {} bdyMaskCell holds a value that is not a ring 0..7; a boundary zone \
             read from it would hide or draw the wrong cells",
            path.display()
        )));
    }
    let rings: Vec<i32> = raw.iter().map(|v| *v as i32).collect();
    Ok(if rings.iter().any(|&r| r > 0) { Some(rings) } else { None })
}

/// Read one horizontal spacing per cell off a mesh file, in metres.
///
/// The file's own `sphere_radius` decides the units: a grid file written on
/// the unit sphere stores `areaCell` and `dcEdge` as unit-sphere quantities
/// despite their `m` and `m^2` attributes, and a static file stores them
/// earth-scaled. Both are handled by dividing out the stored radius and
/// multiplying by the physical one, so the returned metres mean the same
/// thing either way.
///
/// No digest is taken here. The caller has already bound the file by SHA-256
/// through [`read_mesh_coordinates`]; digesting a multi-gigabyte grid twice
/// buys nothing.
pub fn read_mesh_cell_spacing(path: &Path) -> MpasResult<MeshSpacing> {
    let file = netcrust::File::open(path)?;
    let sphere_radius = match file.attribute("sphere_radius").and_then(|a| a.as_f64()) {
        Some(r) if r.is_finite() && r > 0.0 => r,
        Some(r) => {
            return Err(MpasError::Refusal(format!(
                "mesh {} declares sphere_radius = {r}; every stored length is divided by that \
                 radius to reach the unit sphere, and a non-positive radius puts every cell \
                 at infinity or reflects the mesh through the origin",
                path.display()
            )))
        }
        None => {
            return Err(MpasError::Refusal(format!(
                "mesh {} carries no sphere_radius global attribute, so nothing in it says \
                 whether areaCell is a unit-sphere area or square metres. Guessing wrong \
                 scales every derived spacing by 6.4e6 or its inverse, and a render window \
                 sized from it would be the whole planet or a single cell",
                path.display()
            )))
        }
    };

    // areaCell first: it needs no connectivity, so no index base can be got
    // wrong, and it is the metric this program's published resolution figures
    // are quoted in.
    if file.variable("areaCell").is_some() {
        let area = file.read_f64("areaCell")?;
        if area.is_empty() {
            return Err(MpasError::Refusal(format!(
                "mesh {} carries an empty areaCell",
                path.display()
            )));
        }
        let scale = 1.0 / (sphere_radius * sphere_radius);
        let mut spacing = Vec::with_capacity(area.len());
        for (i, &a) in area.iter().enumerate() {
            let unit_area = a * scale;
            if !unit_area.is_finite() || unit_area <= 0.0 {
                return Err(MpasError::Refusal(format!(
                    "mesh {} has areaCell[{i}] = {a} on sphere_radius {sphere_radius}; a cell \
                     with no positive area has no spacing, and the window would be sized from \
                     a zero or a NaN",
                    path.display()
                )));
            }
            spacing.push((2.0 * unit_area / 3f64.sqrt()).sqrt() * MPAS_EARTH_RADIUS_M);
        }
        return Ok(MeshSpacing {
            spacing_metres: spacing,
            source: SpacingSource::AreaCell,
            sphere_radius,
        });
    }

    // Otherwise the edge lengths, with the index base established from the
    // data rather than assumed.
    for name in ["dcEdge", "nEdgesOnCell", "edgesOnCell"] {
        if file.variable(name).is_none() {
            return Err(MpasError::Refusal(format!(
                "mesh {} carries neither areaCell nor the {name} a mean edge length needs, so \
                 nothing in it states the resolution each cell holds. A render window derived \
                 from this file would have to invent its own spacing",
                path.display()
            )));
        }
    }
    let dc_edge = file.read_f64("dcEdge")?;
    let n_edges_on_cell: Vec<i64> = file
        .read_f64("nEdgesOnCell")?
        .into_iter()
        .map(|v| v.round() as i64)
        .collect();
    let edges_on_cell: Vec<i64> = file
        .read_f64("edgesOnCell")?
        .into_iter()
        .map(|v| v.round() as i64)
        .collect();
    let n_cells = n_edges_on_cell.len();
    if n_cells == 0 || dc_edge.is_empty() {
        return Err(MpasError::Refusal(format!(
            "mesh {} carries {n_cells} cell(s) and {} edge(s); a spacing needs both",
            path.display(),
            dc_edge.len()
        )));
    }
    if edges_on_cell.len() % n_cells != 0 {
        return Err(MpasError::Refusal(format!(
            "mesh {} carries {} edgesOnCell values over {n_cells} cells, which is not a whole \
             number of slots per cell; walking it would read one cell's ring out of another's",
            path.display(),
            edges_on_cell.len()
        )));
    }
    let max_edges = edges_on_cell.len() / n_cells;
    let mut min_index = i64::MAX;
    let mut max_index = i64::MIN;
    for c in 0..n_cells {
        let used = n_edges_on_cell[c];
        if used < 1 || used as usize > max_edges {
            return Err(MpasError::Refusal(format!(
                "mesh {} has nEdgesOnCell[{c}] = {used}, outside 1..={max_edges}",
                path.display()
            )));
        }
        for s in 0..used as usize {
            let v = edges_on_cell[c * max_edges + s];
            min_index = min_index.min(v);
            max_index = max_index.max(v);
        }
    }
    let base = detect_index_base(min_index, max_index, dc_edge.len(), "edgesOnCell")?;
    let shift = match base {
        IndexBase::One => 1i64,
        IndexBase::Zero => 0,
    };
    let scale = MPAS_EARTH_RADIUS_M / sphere_radius;
    let mut spacing = vec![0.0f64; n_cells];
    for c in 0..n_cells {
        let used = n_edges_on_cell[c] as usize;
        let mut sum = 0.0;
        for s in 0..used {
            let e = edges_on_cell[c * max_edges + s] - shift;
            let length = dc_edge[e as usize];
            if !length.is_finite() || length <= 0.0 {
                return Err(MpasError::Refusal(format!(
                    "mesh {} has dcEdge[{e}] = {length}; a zero or non-finite edge length \
                     makes the cell's mean spacing meaningless",
                    path.display()
                )));
            }
            sum += length;
        }
        spacing[c] = sum / used as f64 * scale;
    }
    Ok(MeshSpacing {
        spacing_metres: spacing,
        source: SpacingSource::MeanEdgeLength,
        sphere_radius,
    })
}

pub(crate) fn unit_vector(lat_deg: f64, lon_deg: f64) -> [f64; 3] {
    let lat = lat_deg * RAD;
    let lon = lon_deg * RAD;
    // The vendored `libm`, not the platform C library: see the note at the
    // top of window.rs. The nearest-cell index map is built from these.
    let cos_lat = libm::cos(lat);
    [cos_lat * libm::cos(lon), cos_lat * libm::sin(lon), libm::sin(lat)]
}

// ---------------------------------------------------------------------------
// A 3-D k-d tree over the mesh's unit vectors.
// ---------------------------------------------------------------------------

/// Balanced k-d tree built by repeated median selection. Nodes are stored
/// implicitly: `order[lo..hi]` is a subtree and its median is its root, so
/// there is one permutation vector and no per-node allocation.
#[derive(Debug)]
pub(crate) struct KdTree {
    points: Vec<[f64; 3]>,
    order: Vec<u32>,
    /// Split axis chosen for each subtree root, indexed by the median slot.
    axis: Vec<u8>,
}

/// Split `order[lo..hi]` at its median along the widest axis of that
/// subtree's bounding box -- widest, because it is what keeps the tree
/// shallow on a mesh with a refined region.
///
/// The median selection is the standard library's `select_nth_unstable_by`.
/// A hand-rolled quickselect is the obvious thing to write here and the wrong
/// thing to trust: the first version of this function looped forever whenever
/// a partition landed with every element on one side, which a mesh with many
/// coincident coordinates reaches immediately.
fn build_range(
    points: &[[f64; 3]],
    order: &mut [u32],
    axis_of: &mut [u8],
    lo: usize,
    hi: usize,
) {
    if hi <= lo {
        return;
    }
    let mut min = [f64::INFINITY; 3];
    let mut max = [f64::NEG_INFINITY; 3];
    for &idx in &order[lo..hi] {
        let p = points[idx as usize];
        for k in 0..3 {
            if p[k] < min[k] {
                min[k] = p[k];
            }
            if p[k] > max[k] {
                max[k] = p[k];
            }
        }
    }
    let mut axis = 0usize;
    let mut widest = max[0] - min[0];
    for k in 1..3 {
        if max[k] - min[k] > widest {
            widest = max[k] - min[k];
            axis = k;
        }
    }
    let mid = lo + (hi - lo) / 2;
    order[lo..hi].select_nth_unstable_by(mid - lo, |&a, &b| {
        points[a as usize][axis].total_cmp(&points[b as usize][axis])
    });
    axis_of[mid] = axis as u8;
    build_range(points, order, axis_of, lo, mid);
    build_range(points, order, axis_of, mid + 1, hi);
}

impl KdTree {
    /// The `k` nearest stored points, nearest first, with their squared chord
    /// distances.
    ///
    /// The single-nearest search below is the same walk with a one-slot
    /// frontier; this keeps a short sorted list instead, and prunes against
    /// the worst member of it rather than against the best. A boundary
    /// producer needs the *few* nearest, not the one nearest, because the one
    /// nearest does not tell it which triangle a point is in.
    pub(crate) fn nearest_k(&self, target: [f64; 3], k: usize) -> Vec<(u32, f64)> {
        let mut frontier: Vec<(f64, u32)> = Vec::with_capacity(k + 1);
        if k > 0 {
            self.search_k(0, self.order.len(), target, k, &mut frontier);
        }
        frontier.into_iter().map(|(d, i)| (i, d)).collect()
    }

    fn search_k(
        &self,
        lo: usize,
        hi: usize,
        target: [f64; 3],
        k: usize,
        frontier: &mut Vec<(f64, u32)>,
    ) {
        if hi <= lo {
            return;
        }
        let mid = lo + (hi - lo) / 2;
        let idx = self.order[mid];
        let p = self.points[idx as usize];
        let d = (p[0] - target[0]).powi(2) + (p[1] - target[1]).powi(2) + (p[2] - target[2]).powi(2);
        let slot = frontier.partition_point(|(dd, _)| *dd <= d);
        if slot < k {
            frontier.insert(slot, (d, idx));
            frontier.truncate(k);
        }
        let axis = self.axis[mid] as usize;
        let delta = target[axis] - p[axis];
        let (near_lo, near_hi, far_lo, far_hi) = if delta < 0.0 {
            (lo, mid, mid + 1, hi)
        } else {
            (mid + 1, hi, lo, mid)
        };
        self.search_k(near_lo, near_hi, target, k, frontier);
        let worst = if frontier.len() >= k {
            frontier[frontier.len() - 1].0
        } else {
            f64::INFINITY
        };
        if delta * delta < worst {
            self.search_k(far_lo, far_hi, target, k, frontier);
        }
    }

    pub(crate) fn build(points: Vec<[f64; 3]>) -> KdTree {
        let n = points.len();
        let mut order: Vec<u32> = (0..n as u32).collect();
        let mut axis = vec![0u8; n];
        build_range(&points, &mut order, &mut axis, 0, n);
        KdTree {
            points,
            order,
            axis,
        }
    }

    /// Index of the nearest stored point to `target`, by squared chord
    /// distance, with the distance.
    pub(crate) fn nearest(&self, target: [f64; 3]) -> (u32, f64) {
        let mut best = (u32::MAX, f64::INFINITY);
        self.search(0, self.order.len(), target, &mut best);
        best
    }

    fn search(&self, lo: usize, hi: usize, target: [f64; 3], best: &mut (u32, f64)) {
        if hi <= lo {
            return;
        }
        let mid = lo + (hi - lo) / 2;
        let idx = self.order[mid];
        let p = self.points[idx as usize];
        let d = (p[0] - target[0]).powi(2) + (p[1] - target[1]).powi(2) + (p[2] - target[2]).powi(2);
        if d < best.1 {
            *best = (idx, d);
        }
        let axis = self.axis[mid] as usize;
        let delta = target[axis] - p[axis];
        let (near_lo, near_hi, far_lo, far_hi) = if delta < 0.0 {
            (lo, mid, mid + 1, hi)
        } else {
            (mid + 1, hi, lo, mid)
        };
        self.search(near_lo, near_hi, target, best);
        if delta * delta < best.1 {
            self.search(far_lo, far_hi, target, best);
        }
    }
}

/// One nearest-cell index map for one mesh and one window.
#[derive(Debug, Clone)]
pub struct NearestCellWeights {
    pub window_name: String,
    pub window: Window,
    pub target_latitude: Vec<f64>,
    pub target_longitude: Vec<f64>,
    pub cell_index: Vec<i32>,
    /// Window points that are NOT on the mesh: their nearest cell centre is
    /// further away than that cell's own spacing, so the value there would
    /// be a boundary cell's smeared outward rather than a model answer.
    ///
    /// Empty on a mesh nobody asked about (a global mesh covers the sphere,
    /// so every window point is on it), populated by
    /// [`NearestCellWeights::mark_footprint`].
    ///
    /// THE BREAKAGE THIS PREVENTS, MEASURED (2026-08-26, r4.75.11020): a
    /// limited-area mesh is a BOUNDED DISK and `--window mesh` frames it in a
    /// RECTANGLE, so the corners fall outside the domain entirely.  The
    /// nearest-cell resample has no opinion about that -- it returns the
    /// nearest cell however far it is -- so the first rendered composite
    /// reflectivity from a limited-area forecast carried radial streaks of
    /// the outermost ring's value across three corners, with a maximum
    /// nearest-cell distance of 195.8 km on a 4.6 km mesh.  A reader cannot
    /// tell a streak from weather.
    pub off_mesh: Vec<bool>,
    pub distance_km: Vec<f32>,
    pub mesh_sha256: String,
    pub mesh_path: String,
    pub mesh_angle_units: AngleUnits,
    pub n_cells: usize,
    pub ny: usize,
    pub nx: usize,
    pub weights_sha256: String,
    pub build_seconds: f64,
}

impl NearestCellWeights {
    pub fn shape(&self) -> (usize, usize) {
        (self.ny, self.nx)
    }

    /// Mark the window points that are off this mesh.
    ///
    /// A hexagonal cell of spacing `d` puts every point of its own interior
    /// within `d / sqrt(3)` (about 0.577 d) of the centre, so a point further
    /// than `factor * d` from the nearest centre, with `factor >= 1`, is
    /// outside the mesh rather than merely off-centre inside it.  The margin
    /// is deliberate: this is a mask that HIDES data, and it must never hide
    /// a point the model actually solved.
    pub fn mark_footprint(&mut self, spacing_metres: &[f64], factor: f64) -> MpasResult<usize> {
        if spacing_metres.len() != self.n_cells {
            return Err(MpasError::Refusal(format!(
                "the footprint needs one spacing per cell; got {} for {} cells",
                spacing_metres.len(),
                self.n_cells
            )));
        }
        if !(factor.is_finite() && factor >= 1.0) {
            return Err(MpasError::Refusal(format!(
                "a footprint factor of {factor} would mask points inside the                  mesh: every point of a cell of spacing d lies within                  d/sqrt(3) of its centre, so the factor cannot go below 1"
            )));
        }
        let mut marked = 0usize;
        self.off_mesh = self
            .cell_index
            .iter()
            .zip(self.distance_km.iter())
            .map(|(&cell, &km)| {
                let limit = factor * spacing_metres[cell as usize] / 1000.0;
                let outside = f64::from(km) > limit;
                if outside {
                    marked += 1;
                }
                outside
            })
            .collect();
        Ok(marked)
    }

    /// Hide the window points whose nearest cell is in a limited-area
    /// mesh's boundary zone (`bdyMaskCell` 1-7, see [`read_boundary_rings`]),
    /// on top of the footprint.  Returns how many points this newly hides.
    ///
    /// Those cells are not a forecast: rings 6-7 are overwritten from the
    /// boundary file every step and rings 1-5 are relaxed toward it, so a map
    /// of them shows the parent's interpolated state run through the
    /// microphysics, not the limited-area model's weather.  The operational
    /// limited-area products do not draw them either (RRFS posts its
    /// products on a grid inside the computational domain).
    ///
    /// THE BREAKAGE THIS PREVENTS, MEASURED (2026-10-28, RRFS hrrrv5 on the
    /// lam7k cut of conus3km, 2023-03-31 12Z from GFS): composite reflectivity
    /// was >= 15 dBZ in 93% of the specified-zone cells at 15Z, hex and native
    /// MPAS-Model-rrfs alike, against 0% in MRMS there and 17% in the
    /// interior; on the 206,726-cell mv3km cut the same zone drew a
    /// continuous 30-45 dBZ wall down the east edge.  A reader takes a ring
    /// of echo around the frame for weather.
    pub fn mask_boundary_zone(&mut self, rings: &[i32]) -> MpasResult<usize> {
        if rings.len() != self.n_cells {
            return Err(MpasError::Refusal(format!(
                "the boundary zone needs one bdyMaskCell per cell; got {} for {} cells",
                rings.len(),
                self.n_cells
            )));
        }
        if self.off_mesh.is_empty() {
            self.off_mesh = vec![false; self.cell_index.len()];
        }
        let mut marked = 0usize;
        for (outside, &cell) in self.off_mesh.iter_mut().zip(self.cell_index.iter()) {
            if !*outside && rings[cell as usize] > 0 {
                *outside = true;
                marked += 1;
            }
        }
        Ok(marked)
    }

    /// How far the FURTHEST point that is still on the mesh had to reach.
    /// With a footprint marked this is the accurate resample quality figure;
    /// [`Self::max_distance_km`] then describes points nobody will see.
    pub fn max_on_mesh_distance_km(&self) -> f32 {
        if self.off_mesh.is_empty() {
            return self.max_distance_km();
        }
        self.distance_km
            .iter()
            .zip(self.off_mesh.iter())
            .filter(|(_, outside)| !**outside)
            .map(|(&km, _)| km)
            .fold(f32::MIN, f32::max)
    }

    pub fn max_distance_km(&self) -> f32 {
        self.distance_km.iter().copied().fold(f32::MIN, f32::max)
    }

    pub fn mean_distance_km(&self) -> f64 {
        self.distance_km.iter().map(|&v| v as f64).sum::<f64>() / self.distance_km.len() as f64
    }

    /// Gather a cell-indexed field onto the target window.
    ///
    /// The result is exactly the source array's values, reordered: no
    /// arithmetic happens, so every bit pattern in the output came straight
    /// from the model field. `values` is `(n_cells, levels)` row-major, and
    /// the output is `(levels, ny, nx)` -- the WRF layout.
    pub fn gather(&self, values: &[f32], levels: usize) -> MpasResult<Vec<f32>> {
        if values.len() != self.n_cells * levels {
            return Err(MpasError::Refusal(format!(
                "field carries {} value(s); mesh has {} cells x {levels} level(s)",
                values.len(),
                self.n_cells
            )));
        }
        let points = self.cell_index.len();
        let mut out = vec![0.0f32; levels * points];
        if points == 0 {
            return Ok(out);
        }
        // Each level's plane is filled on its own, so levels (and row bands of
        // one plane) run in parallel and write disjoint slots: the same bytes
        // as one thread.
        use rayon::prelude::*;
        const BAND: usize = 64 * 1024;
        out.par_chunks_mut(points)
            .enumerate()
            .for_each(|(level, plane)| {
                plane
                    .par_chunks_mut(BAND)
                    .enumerate()
                    .for_each(|(band, target)| {
                        let first = band * BAND;
                        let cells = &self.cell_index[first..first + target.len()];
                        for (slot, &cell) in target.iter_mut().zip(cells) {
                            *slot = values[cell as usize * levels + level];
                        }
                        if !self.off_mesh.is_empty() {
                            let outside = &self.off_mesh[first..first + target.len()];
                            for (slot, &outside) in target.iter_mut().zip(outside) {
                                if outside {
                                    *slot = OFF_MESH_FILL;
                                }
                            }
                        }
                    });
            });
        Ok(out)
    }
}

fn weights_digest(
    window_name: &str,
    spec_compact: &str,
    mesh_sha256: &str,
    cell_index: &[i32],
) -> String {
    let mut spec_hash = Sha256::new();
    spec_hash.update(spec_compact.as_bytes());
    let spec_digest = hex(&spec_hash.finalize());

    let mut digest = Sha256::new();
    digest.update(WEIGHTS_SCHEMA.as_bytes());
    digest.update([0u8]);
    digest.update(RESAMPLE_METHOD.as_bytes());
    digest.update([0u8]);
    digest.update(window_name.as_bytes());
    digest.update([0u8]);
    digest.update(spec_digest.as_bytes());
    digest.update([0u8]);
    digest.update(mesh_sha256.as_bytes());
    digest.update([0u8]);
    let mut bytes = Vec::with_capacity(cell_index.len() * 4);
    for &v in cell_index {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    digest.update(&bytes);
    hex(&digest.finalize())
}

pub(crate) fn hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push_str(&format!("{b:02x}"));
    }
    out
}

/// Build one nearest-cell index map. No cache is consulted or written.
pub fn build_weights(
    mesh: &MeshCoordinates,
    window_name: &str,
    window: &Window,
) -> MpasResult<NearestCellWeights> {
    let started = std::time::Instant::now();
    let (target_latitude, target_longitude) = window.coordinates()?;
    let (ny, nx) = window.shape()?;
    if target_latitude.len() != ny * nx {
        return Err(MpasError::Refusal(
            "window coordinates do not match its shape".to_string(),
        ));
    }

    let cells: Vec<[f64; 3]> = (0..mesh.n_cells())
        .map(|i| unit_vector(mesh.latitude_degrees[i], mesh.longitude_degrees[i]))
        .collect();
    let tree = KdTree::build(cells);

    // THE COST THIS REMOVES (conus12km limited-area mesh, 157,859 cells, global 0.25 degree window,
    // 2026-10-17): the 1,038,240 nearest-cell queries ran one after another on one core for 495 s. A
    // target far outside a limited-area mesh is nearly equidistant from every cell, so the k-d tree can
    // prune almost nothing and each query walks most of the tree. The queries are independent, so they
    // run on every core; each slot's answer is computed exactly as before and lands in the same slot,
    // so the weights (and their digest) are the same bytes.
    let (cell_index, distance_km): (Vec<i32>, Vec<f32>) = (0..ny * nx)
        .into_par_iter()
        .map(|slot| {
            let target = unit_vector(target_latitude[slot], target_longitude[slot]);
            let (idx, chord_squared) = tree.nearest(target);
            // Chord to great-circle, on the mesh's physical radius.
            let chord = chord_squared.sqrt();
            let great_circle = 2.0 * libm::asin((chord / 2.0).clamp(0.0, 1.0)) * MPAS_EARTH_RADIUS_M / 1000.0;
            (idx as i32, great_circle as f32)
        })
        .unzip();
    if cell_index
        .iter()
        .any(|&i| i < 0 || i as usize >= mesh.n_cells())
    {
        return Err(MpasError::Refusal(
            "nearest-cell index left the mesh".to_string(),
        ));
    }

    let spec_compact = window.spec_json_compact();
    Ok(NearestCellWeights {
        window_name: window_name.to_string(),
        window: window.clone(),
        target_latitude,
        target_longitude,
        cell_index: cell_index.clone(),
        off_mesh: Vec::new(),
        distance_km,
        mesh_sha256: mesh.source_sha256.clone(),
        mesh_path: mesh.source_path.display().to_string(),
        mesh_angle_units: mesh.angle_units,
        n_cells: mesh.n_cells(),
        ny,
        nx,
        weights_sha256: weights_digest(
            window_name,
            &spec_compact,
            &mesh.source_sha256,
            &cell_index,
        ),
        build_seconds: started.elapsed().as_secs_f64(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The k-d tree must agree with brute force on every query, or the whole
    /// map is quietly wrong in a way no eye would catch.
    #[test]
    fn the_kd_tree_agrees_with_brute_force() {
        let mut state = 0x2026_0812_u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        let cells: Vec<[f64; 3]> = (0..4000)
            .map(|_| {
                let lat = next() * 180.0 - 90.0;
                let lon = next() * 360.0 - 180.0;
                unit_vector(lat, lon)
            })
            .collect();
        let tree = KdTree::build(cells.clone());
        for _ in 0..2000 {
            let target = unit_vector(next() * 180.0 - 90.0, next() * 360.0 - 180.0);
            let (got, got_d) = tree.nearest(target);
            let mut best = (usize::MAX, f64::INFINITY);
            for (i, p) in cells.iter().enumerate() {
                let d = (p[0] - target[0]).powi(2)
                    + (p[1] - target[1]).powi(2)
                    + (p[2] - target[2]).powi(2);
                if d < best.1 {
                    best = (i, d);
                }
            }
            assert_eq!(got as usize, best.0, "index differs");
            assert_eq!(got_d, best.1, "distance differs");
        }
    }

    /// Points whose nearest cell sits in rings 1-7 are hidden; interior
    /// points and points already off the footprint keep their state.
    #[test]
    fn the_boundary_zone_hides_ring_points_only() {
        let mut weights = NearestCellWeights {
            window_name: "t".into(),
            window: Window::named("focus").unwrap(),
            target_latitude: vec![0.0; 4],
            target_longitude: vec![0.0; 4],
            cell_index: vec![0, 1, 2, 2],
            distance_km: vec![0.0; 4],
            off_mesh: vec![false, false, false, true],
            mesh_sha256: String::new(),
            mesh_path: String::new(),
            mesh_angle_units: AngleUnits::Declared,
            n_cells: 3,
            ny: 2,
            nx: 2,
            weights_sha256: String::new(),
            build_seconds: 0.0,
        };
        // cell 0 interior, cell 1 relaxation ring 3, cell 2 specified ring 7
        assert_eq!(weights.mask_boundary_zone(&[0, 3, 7]).unwrap(), 2);
        assert_eq!(weights.off_mesh, vec![false, true, true, true]);
        assert!(weights.mask_boundary_zone(&[0, 3]).is_err(), "one ring per cell");
    }

    #[test]
    fn a_gather_reorders_without_arithmetic() {
        let weights = NearestCellWeights {
            window_name: "t".into(),
            window: Window::named("focus").unwrap(),
            target_latitude: vec![0.0; 4],
            target_longitude: vec![0.0; 4],
            cell_index: vec![2, 0, 1, 2],
            distance_km: vec![0.0; 4],
            off_mesh: Vec::new(),
            mesh_sha256: String::new(),
            mesh_path: String::new(),
            mesh_angle_units: AngleUnits::Declared,
            n_cells: 3,
            ny: 2,
            nx: 2,
            weights_sha256: String::new(),
            build_seconds: 0.0,
        };
        // Awkward bit patterns: a value that survives a gather bit-for-bit
        // cannot be hiding behind a benign-looking number.
        let values = vec![
            f32::from_bits(0x7f7f_ffff),
            -0.0,
            f32::from_bits(0x0080_0000),
            1.0,
            2.0,
            3.0,
        ];
        let out = weights.gather(&values, 2).unwrap();
        // level 0 then level 1, each (ny, nx)
        assert_eq!(out[0].to_bits(), values[4].to_bits());
        assert_eq!(out[1].to_bits(), values[0].to_bits());
        assert_eq!(out[2].to_bits(), values[2].to_bits());
        assert_eq!(out[3].to_bits(), values[4].to_bits());
        assert_eq!(out[4].to_bits(), values[5].to_bits());
        assert_eq!(out[5].to_bits(), values[1].to_bits());
    }

    #[test]
    fn the_pole_tolerance_admits_a_float32_pi_over_two() {
        // The exact value a float32 mesh stores for the north pole.
        let f32_pole = (std::f64::consts::FRAC_PI_2 as f32) as f64;
        assert!(
            f32_pole > std::f64::consts::FRAC_PI_2,
            "float32 pi/2 must round up for this test to mean anything"
        );
        assert!(
            f32_pole <= POLE_TOLERANCE_RAD,
            "float32 pi/2 ({f32_pole}) must be inside the pole tolerance ({POLE_TOLERANCE_RAD})"
        );
        // And the tolerance is nowhere near admitting degrees.
        assert!(90.0 > POLE_TOLERANCE_RAD);
        // A milliradian past the pole -- 6 km -- is still refused.
        assert!(std::f64::consts::FRAC_PI_2 + 1.0e-3 > POLE_TOLERANCE_RAD);
    }

    #[test]
    fn declared_radians_are_taken_as_declared() {
        for text in ["rad", "radian", "radians", " RADIANS "] {
            assert_eq!(
                resolve_angle_units(text, 1.5, 3.1).unwrap(),
                AngleUnits::Declared,
                "{text}"
            );
        }
    }

    #[test]
    fn declared_degrees_are_still_refused_whatever_the_range() {
        // The guard that mattered before must not have been weakened: a file
        // that says degrees is refused even when its numbers would pass the
        // range test on their own.
        let err = resolve_angle_units("degrees", 1.5, 3.1)
            .unwrap_err()
            .to_string();
        assert!(err.contains("not radians"), "{err}");
        for text in ["degrees", "degree", "deg", "degrees_north", "m"] {
            assert!(resolve_angle_units(text, 0.1, 0.1).is_err(), "{text}");
        }
    }

    #[test]
    fn absent_units_with_radian_range_are_inferred() {
        // pi/2 and pi exactly: the x1.40962 global mesh's own extremes.
        assert_eq!(
            resolve_angle_units("", std::f64::consts::FRAC_PI_2, std::f64::consts::PI).unwrap(),
            AngleUnits::InferredFromRange
        );
    }

    #[test]
    fn absent_units_with_degree_range_are_refused() {
        // A degree-valued global mesh reaches |lat| = 90, which is 57x the
        // radian bound. There is no overlap to get wrong.
        let err = resolve_angle_units("", 90.0, 180.0).unwrap_err().to_string();
        assert!(err.contains("declare no units"), "{err}");
        assert!(err.contains("90"), "{err}");
        // Latitude alone is enough to refuse, even with a small longitude span.
        assert!(resolve_angle_units("", 41.5, 1.0).is_err());
        // And longitude alone is enough, even with a small latitude span.
        assert!(resolve_angle_units("", 0.5, 96.0).is_err());
    }

    #[test]
    fn absent_units_reject_the_boundary_by_a_hair() {
        // Just outside pi/2 must fail; just inside must pass. This pins the
        // limit rather than trusting it.
        assert!(resolve_angle_units("", std::f64::consts::FRAC_PI_2 + 1.0e-3, 1.0).is_err());
        assert!(resolve_angle_units("", std::f64::consts::FRAC_PI_2 - 1.0e-3, 1.0).is_ok());
    }

    // -----------------------------------------------------------------
    // Per-cell spacing, and the index base a connectivity read depends on.
    // -----------------------------------------------------------------

    /// A grid file carrying only what a spacing needs. `area` and the edge
    /// trio are independently optional so each branch of the reader can be
    /// reached with a real file rather than a stub.
    fn write_spacing_grid(
        label: &str,
        sphere_radius: Option<f64>,
        area_cell: Option<&[f64]>,
        edges: Option<(&[f64], &[i32], &[i32], usize)>,
    ) -> PathBuf {
        use rw_store::netcdf_classic::{
            NcAttr, NcClassicWriter, NcData, NcDim, NcFormat, NcType, NcVarDef,
        };
        let path = std::env::temp_dir().join(format!(
            "rw-mpas-spacing-{}-{label}.nc",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&path);
        let n_cells = area_cell
            .map(|a| a.len())
            .or_else(|| edges.map(|(_, neoc, _, _)| neoc.len()))
            .unwrap_or(1);
        let n_edges = edges.map(|(dc, _, _, _)| dc.len()).unwrap_or(1);
        let max_edges = edges.map(|(_, _, _, me)| me).unwrap_or(1);
        let dims = vec![
            NcDim::fixed("nCells", n_cells),
            NcDim::fixed("nEdges", n_edges),
            NcDim::fixed("maxEdges", max_edges),
        ];
        let (c, e, me) = (0usize, 1, 2);
        let mut vars = vec![NcVarDef::new("latCell", NcType::Double, vec![c])];
        if area_cell.is_some() {
            vars.push(NcVarDef::new("areaCell", NcType::Double, vec![c]));
        }
        if edges.is_some() {
            vars.push(NcVarDef::new("dcEdge", NcType::Double, vec![e]));
            vars.push(NcVarDef::new("nEdgesOnCell", NcType::Int, vec![c]));
            vars.push(NcVarDef::new("edgesOnCell", NcType::Int, vec![c, me]));
        }
        let gattrs = match sphere_radius {
            Some(r) => vec![NcAttr::doubles("sphere_radius", vec![r])],
            None => Vec::new(),
        };
        let mut w =
            NcClassicWriter::create(&path, NcFormat::Offset64, dims, gattrs, vars, 0).unwrap();
        w.put("latCell", NcData::Doubles(&vec![0.0f64; n_cells])).unwrap();
        if let Some(a) = area_cell {
            w.put("areaCell", NcData::Doubles(a)).unwrap();
        }
        if let Some((dc, neoc, eoc, _)) = edges {
            w.put("dcEdge", NcData::Doubles(dc)).unwrap();
            w.put("nEdgesOnCell", NcData::Ints(neoc)).unwrap();
            w.put("edgesOnCell", NcData::Ints(eoc)).unwrap();
        }
        w.finish().unwrap();
        path
    }

    /// The trap this reader exists to not fall into. An `edgesOnCell` read
    /// one off still lands inside `dcEdge` for every cell but the last, so
    /// every spacing would belong to a neighbouring edge and the mesh would
    /// measure clean at the wrong resolution. Both bases are decisive
    /// because every edge lies on two cells and therefore appears in some
    /// cell's ring; anything else is refused rather than guessed.
    #[test]
    fn the_edge_index_base_is_read_from_the_data() {
        // A one-based map reaches nEdges itself -- the signature that says
        // one-based, because a zero-based index never can.
        assert_eq!(
            detect_index_base(1, 12, 12, "edgesOnCell").unwrap(),
            IndexBase::One
        );
        assert_eq!(
            detect_index_base(0, 11, 12, "edgesOnCell").unwrap(),
            IndexBase::Zero
        );
        // Neither span: refused, with the numbers that made it refuse.
        let error = detect_index_base(1, 11, 12, "edgesOnCell")
            .unwrap_err()
            .to_string();
        assert!(error.contains("spans [1, 11] over its valid slots"), "{error}");
        assert!(error.contains("[1, 12]") && error.contains("[0, 11]"), "{error}");
        assert!(error.contains("still lands in range"), "{error}");
        assert!(detect_index_base(0, 12, 12, "edgesOnCell").is_err());
        assert!(detect_index_base(2, 12, 12, "edgesOnCell").is_err());
    }

    /// A grid file on the UNIT sphere and a static file on the earth carry
    /// the same mesh at different scales. Both have to come back in the same
    /// metres, or a window derived from one is 6.4e6 times the other.
    #[test]
    fn the_stored_sphere_radius_sets_the_scale() {
        let target_m = 4_530.0f64;
        let unit_area = (target_m / MPAS_EARTH_RADIUS_M).powi(2) * 3f64.sqrt() / 2.0;

        let unit = write_spacing_grid("unit", Some(1.0), Some(&[unit_area; 3]), None);
        let read = read_mesh_cell_spacing(&unit).unwrap();
        assert_eq!(read.source, SpacingSource::AreaCell);
        assert_eq!(read.sphere_radius, 1.0);
        for v in &read.spacing_metres {
            assert!((v - target_m).abs() < 1.0e-6, "{v} is not {target_m}");
        }

        let scaled_area = unit_area * MPAS_EARTH_RADIUS_M * MPAS_EARTH_RADIUS_M;
        let scaled = write_spacing_grid(
            "scaled",
            Some(MPAS_EARTH_RADIUS_M),
            Some(&[scaled_area; 3]),
            None,
        );
        let read_scaled = read_mesh_cell_spacing(&scaled).unwrap();
        for v in &read_scaled.spacing_metres {
            assert!((v - target_m).abs() < 1.0e-6, "{v} is not {target_m}");
        }
        let _ = std::fs::remove_file(&unit);
        let _ = std::fs::remove_file(&scaled);
    }

    /// A file with the connectivity but no `areaCell` still yields a spacing,
    /// through the mean edge length, with the index base taken off the data.
    #[test]
    fn a_file_without_area_cell_measures_its_edges() {
        // Four cells, three edges each, one-based as MPAS writes it. Every
        // edge index 1..=6 appears, which is what makes the base decisive.
        let dc = vec![0.1f64, 0.2, 0.3, 0.4, 0.5, 0.6];
        let neoc = vec![3i32; 4];
        let eoc = vec![1i32, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5, 6];
        let path = write_spacing_grid("edges", Some(1.0), None, Some((&dc, &neoc, &eoc, 3)));
        let read = read_mesh_cell_spacing(&path).unwrap();
        assert_eq!(read.source, SpacingSource::MeanEdgeLength);
        assert_eq!(read.spacing_metres.len(), 4);
        // Cell 0 averages edges 1, 2 and 3, which are 0.1, 0.2 and 0.3 on the
        // unit sphere: 0.2 radians of arc.
        let want = 0.2 * MPAS_EARTH_RADIUS_M;
        assert!(
            (read.spacing_metres[0] - want).abs() < 1.0e-6,
            "{} is not {want}",
            read.spacing_metres[0]
        );
        assert!(read.spacing_metres[3] > read.spacing_metres[0], "the ring must vary");
        let _ = std::fs::remove_file(&path);
    }

    /// A file that says nothing about its radius, and a file that says
    /// nothing about its resolution, are both refused by name.
    #[test]
    fn a_file_that_cannot_state_its_scale_is_refused() {
        let no_radius = write_spacing_grid("noradius", None, Some(&[1.0e-7; 3]), None);
        let error = read_mesh_cell_spacing(&no_radius).unwrap_err().to_string();
        assert!(error.contains("no sphere_radius"), "{error}");
        assert!(error.contains("6.4e6"), "{error}");
        let _ = std::fs::remove_file(&no_radius);

        let bad_radius = write_spacing_grid("badradius", Some(0.0), Some(&[1.0e-7; 3]), None);
        let error = read_mesh_cell_spacing(&bad_radius).unwrap_err().to_string();
        assert!(error.contains("sphere_radius = 0"), "{error}");
        let _ = std::fs::remove_file(&bad_radius);

        let bare = write_spacing_grid("bare", Some(1.0), None, None);
        let error = read_mesh_cell_spacing(&bare).unwrap_err().to_string();
        assert!(error.contains("neither areaCell nor the dcEdge"), "{error}");
        assert!(error.contains("invent its own spacing"), "{error}");
        let _ = std::fs::remove_file(&bare);

        let zero_area = write_spacing_grid("zeroarea", Some(1.0), Some(&[1.0e-7, 0.0]), None);
        let error = read_mesh_cell_spacing(&zero_area).unwrap_err().to_string();
        assert!(error.contains("areaCell[1] = 0"), "{error}");
        let _ = std::fs::remove_file(&zero_area);
    }

    #[test]
    fn a_field_of_the_wrong_length_is_refused() {
        let weights = NearestCellWeights {
            window_name: "t".into(),
            window: Window::named("focus").unwrap(),
            target_latitude: vec![0.0; 1],
            target_longitude: vec![0.0; 1],
            cell_index: vec![0],
            distance_km: vec![0.0],
            off_mesh: Vec::new(),
            mesh_sha256: String::new(),
            mesh_path: String::new(),
            mesh_angle_units: AngleUnits::Declared,
            n_cells: 3,
            ny: 1,
            nx: 1,
            weights_sha256: String::new(),
            build_seconds: 0.0,
        };
        let err = weights.gather(&[1.0, 2.0], 1).unwrap_err().to_string();
        assert!(err.contains("mesh has 3 cells"), "{err}");
    }
}
