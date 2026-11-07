//! Product remaps: mesh fields onto a render grid the way each quantity needs.
//!
//! The nearest-cell map in [`crate::weights`] stays the frozen default of the
//! converter binary and the composite path.  This module adds the remap a
//! product door ships by default (`rw_mpas_convert --remap product`):
//!
//! * **linear** for continuous state (temperature, pressure, moisture, wind,
//!   heights, reflectivity, fluxes): barycentric interpolation inside the
//!   triangle of the three cell centres that meet at an MPAS vertex.  Those
//!   triangles are the mesh's own Delaunay dual (`cellsOnVertex`), so the
//!   interpolant is the unique piecewise-linear surface through every cell
//!   value: it reproduces a linear field exactly, never overshoots its three
//!   cell values, and has no seams where a nearest-cell map steps.
//! * **conservative** for accumulations (`RAINC`, `RAINNC`, `SNOWNC`,
//!   `GRAUPELNC`): first-order area-weighted remap.  Each render pixel's value
//!   is the area-weighted mean of the cells that overlap its footprint on the
//!   sphere.  An MPAS cell is the Voronoi region of its generator, so a point
//!   belongs to the cell whose centre is nearest; overlap areas are measured by
//!   quadrature, sampling every pixel footprint on an `n x n` sub-grid (`n`
//!   grows where the cells are smaller than the pixel) and weighting each
//!   sample by the exact area element of the footprint at that sample.  A
//!   pixel's value is therefore a true area mean of the model's water, and the
//!   global integral of the remapped field matches the mesh's own
//!   `sum(value * areaCell)` to the quadrature error the converter reports.
//! * **nearest** for categorical and land-only fields (`LANDMASK`,
//!   `LU_INDEX`, soil): a category has no average, and soil arrays carry
//!   placeholder values over water that must not be blended into the coast.
//!
//! Window points off a limited-area mesh keep the nearest-cell footprint mask
//! and stay NaN under every method.

use rayon::prelude::*;
use sha2::{Digest, Sha256};
use std::path::Path;

use crate::error::{MpasError, MpasResult};
use crate::fieldmap::RemapClass;
use crate::weights::{
    IndexBase, KdTree, MeshCoordinates, NearestCellWeights, OFF_MESH_FILL, detect_index_base,
    unit_vector,
};
use crate::window::MPAS_EARTH_RADIUS_M;

pub const PRODUCT_REMAP_SCHEMA: &str = "gpuwm-hex.product-remap/v1";
pub const PRODUCT_REMAP_METHOD: &str = "product remap: linear barycentric on the mesh Delaunay \
     triangles (cellsOnVertex) for state; first-order area-conservative (sampled Voronoi overlap) \
     for accumulations; nearest cell for categorical and soil fields";

/// The connectivity a product remap needs, read once per mesh.
#[derive(Debug, Clone)]
pub struct MeshTopology {
    pub n_cells: usize,
    /// Zero-based cells meeting at each vertex; `u32::MAX` where the mesh has
    /// no cell (the outer ring of a limited-area mesh).
    pub cells_on_vertex: Vec<[u32; 3]>,
    /// Zero-based vertices of each cell, `vertex_offsets[c]..vertex_offsets[c+1]`.
    pub vertex_offsets: Vec<u32>,
    pub vertices: Vec<u32>,
    /// Cell areas on the unit sphere.
    pub area_unit: Vec<f64>,
}

impl MeshTopology {
    fn vertices_of(&self, cell: usize) -> &[u32] {
        &self.vertices[self.vertex_offsets[cell] as usize..self.vertex_offsets[cell + 1] as usize]
    }
}

fn read_int(file: &netcrust::File, name: &str, path: &Path) -> MpasResult<Vec<i64>> {
    if file.variable(name).is_none() {
        return Err(MpasError::Refusal(format!(
            "mesh {} has no {name}; the product remap's triangles and cell areas come from it, \
             and without them only the nearest-cell map (--remap nearest) can be drawn",
            path.display()
        )));
    }
    Ok(file
        .read_f64(name)?
        .into_iter()
        .map(|v| v.round() as i64)
        .collect())
}

/// Read `nEdgesOnCell`, `verticesOnCell`, `cellsOnVertex` and `areaCell`.
pub fn read_mesh_topology(path: &Path, n_cells: usize) -> MpasResult<MeshTopology> {
    let file = netcrust::File::open(path)?;
    let sphere_radius = file
        .attribute("sphere_radius")
        .and_then(|a| a.as_f64())
        .filter(|r| r.is_finite() && *r > 0.0)
        .ok_or_else(|| {
            MpasError::Refusal(format!(
                "mesh {} carries no positive sphere_radius, so its areaCell cannot be put on \
                 the unit sphere and the conservative remap would weight cells in unknown units",
                path.display()
            ))
        })?;
    let n_edges_on_cell = read_int(&file, "nEdgesOnCell", path)?;
    let vertices_on_cell = read_int(&file, "verticesOnCell", path)?;
    let cells_on_vertex = read_int(&file, "cellsOnVertex", path)?;
    let area = file.read_f64("areaCell")?;
    if n_edges_on_cell.len() != n_cells || area.len() != n_cells {
        return Err(MpasError::Refusal(format!(
            "mesh {} carries {} nEdgesOnCell and {} areaCell values for {n_cells} cells",
            path.display(),
            n_edges_on_cell.len(),
            area.len()
        )));
    }
    if vertices_on_cell.len() % n_cells != 0 || cells_on_vertex.len() % 3 != 0 {
        return Err(MpasError::Refusal(format!(
            "mesh {} connectivity is not whole rows (verticesOnCell {} values, cellsOnVertex {})",
            path.display(),
            vertices_on_cell.len(),
            cells_on_vertex.len()
        )));
    }
    let max_edges = vertices_on_cell.len() / n_cells;
    let n_vertices = cells_on_vertex.len() / 3;
    let (mut lo, mut hi) = (i64::MAX, i64::MIN);
    for c in 0..n_cells {
        let used = n_edges_on_cell[c];
        if used < 3 || used as usize > max_edges {
            return Err(MpasError::Refusal(format!(
                "mesh {} has nEdgesOnCell[{c}] = {used}, outside 3..={max_edges}",
                path.display()
            )));
        }
        for s in 0..used as usize {
            let v = vertices_on_cell[c * max_edges + s];
            lo = lo.min(v);
            hi = hi.max(v);
        }
    }
    let base = detect_index_base(lo, hi, n_vertices, "verticesOnCell")?;
    let shift = if base == IndexBase::One { 1 } else { 0 };
    let mut vertex_offsets = Vec::with_capacity(n_cells + 1);
    let mut vertices = Vec::with_capacity(n_cells * 6);
    vertex_offsets.push(0u32);
    for c in 0..n_cells {
        for s in 0..n_edges_on_cell[c] as usize {
            vertices.push((vertices_on_cell[c * max_edges + s] - shift) as u32);
        }
        vertex_offsets.push(vertices.len() as u32);
    }
    let cells_on_vertex: Vec<[u32; 3]> = cells_on_vertex
        .chunks_exact(3)
        .map(|row| {
            let mut out = [u32::MAX; 3];
            for k in 0..3 {
                let c = row[k] - shift;
                if c >= 0 && (c as usize) < n_cells {
                    out[k] = c as u32;
                }
            }
            out
        })
        .collect();
    let scale = 1.0 / (sphere_radius * sphere_radius);
    let area_unit: Vec<f64> = area.iter().map(|a| a * scale).collect();
    if area_unit.iter().any(|a| !a.is_finite() || *a <= 0.0) {
        return Err(MpasError::Refusal(format!(
            "mesh {} has a non-positive or non-finite areaCell",
            path.display()
        )));
    }
    Ok(MeshTopology {
        n_cells,
        cells_on_vertex,
        vertex_offsets,
        vertices,
        area_unit,
    })
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Barycentric coordinates of `p` in the planar triangle through `a, b, c`
/// after radial (gnomonic) projection: the solution of `p = x a + y b + z c`,
/// normalised so `x + y + z = 1`.  `None` for a degenerate triangle or one on
/// the far side of the sphere.
pub fn spherical_barycentric(p: [f64; 3], a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> Option<[f64; 3]> {
    let det = dot(a, cross(b, c));
    if det.abs() < 1e-300 {
        return None;
    }
    let x = dot(p, cross(b, c)) / det;
    let y = dot(a, cross(p, c)) / det;
    let z = dot(a, cross(b, p)) / det;
    let s = x + y + z;
    if !(s > 0.0) {
        return None;
    }
    Some([x / s, y / s, z / s])
}

/// Three cells and three weights per target point.
#[derive(Debug, Clone)]
pub struct LinearWeights {
    pub cells: Vec<[u32; 3]>,
    pub weights: Vec<[f32; 3]>,
    /// Points found inside a triangle of their nearest cell.
    pub contained: usize,
    /// Points that needed the second ring of triangles.
    pub widened: usize,
    /// Points inside no triangle (outside a limited-area mesh's triangulated
    /// hull): the best triangle's weights clamped to non-negative.
    pub clamped: usize,
}

const INSIDE_EPS: f64 = -1e-9;

fn min3(w: &[f64; 3]) -> f64 {
    w[0].min(w[1]).min(w[2])
}

/// Locate every target in the mesh's vertex triangles.
pub fn build_linear(
    centres: &[[f64; 3]],
    topology: &MeshTopology,
    nearest_cell: &[i32],
    targets: &[[f64; 3]],
) -> LinearWeights {
    #[derive(Clone, Copy)]
    enum Found {
        Inside,
        Widened,
        Clamped,
    }
    let triangle = |v: u32| -> Option<[u32; 3]> {
        let t = topology.cells_on_vertex[v as usize];
        (t.iter().all(|&c| c != u32::MAX)).then_some(t)
    };
    let rows: Vec<([u32; 3], [f32; 3], Found)> = targets
        .par_iter()
        .zip(nearest_cell.par_iter())
        .map(|(&p, &near)| {
            let near = near as usize;
            let mut best: Option<([u32; 3], [f64; 3])> = None;
            let mut consider = |tri: [u32; 3], best: &mut Option<([u32; 3], [f64; 3])>| -> bool {
                if let Some(w) = spherical_barycentric(
                    p,
                    centres[tri[0] as usize],
                    centres[tri[1] as usize],
                    centres[tri[2] as usize],
                ) {
                    let inside = min3(&w) >= INSIDE_EPS;
                    if best.is_none_or(|(_, bw)| min3(&w) > min3(&bw)) {
                        *best = Some((tri, w));
                    }
                    return inside;
                }
                false
            };
            for &v in topology.vertices_of(near) {
                if let Some(tri) = triangle(v) {
                    if consider(tri, &mut best) {
                        let (t, w) = best.expect("set by consider");
                        return (t, clamp_weights(w), Found::Inside);
                    }
                }
            }
            // Second ring: every triangle touching a cell of the first ring.
            let mut ring: Vec<u32> = Vec::with_capacity(24);
            for &v in topology.vertices_of(near) {
                if let Some(tri) = triangle(v) {
                    for c in tri {
                        if c as usize != near && !ring.contains(&c) {
                            ring.push(c);
                        }
                    }
                }
            }
            for c in ring {
                for &v in topology.vertices_of(c as usize) {
                    if let Some(tri) = triangle(v) {
                        if consider(tri, &mut best) {
                            let (t, w) = best.expect("set by consider");
                            return (t, clamp_weights(w), Found::Widened);
                        }
                    }
                }
            }
            match best {
                Some((t, w)) => (t, clamp_weights(w), Found::Clamped),
                // No triangle at all near this cell: the nearest value.
                None => ([near as u32; 3], [1.0, 0.0, 0.0], Found::Clamped),
            }
        })
        .collect();
    let mut out = LinearWeights {
        cells: Vec::with_capacity(rows.len()),
        weights: Vec::with_capacity(rows.len()),
        contained: 0,
        widened: 0,
        clamped: 0,
    };
    for (t, w, found) in rows {
        out.cells.push(t);
        out.weights.push(w);
        match found {
            Found::Inside => out.contained += 1,
            Found::Widened => out.widened += 1,
            Found::Clamped => out.clamped += 1,
        }
    }
    out
}

/// Negative round-off (and a clamped outside point) to zero, renormalised.
fn clamp_weights(w: [f64; 3]) -> [f32; 3] {
    let c = [w[0].max(0.0), w[1].max(0.0), w[2].max(0.0)];
    let s = c[0] + c[1] + c[2];
    if s > 0.0 {
        [(c[0] / s) as f32, (c[1] / s) as f32, (c[2] / s) as f32]
    } else {
        [1.0, 0.0, 0.0]
    }
}

impl LinearWeights {
    /// `values` is `(n_cells, levels)`; the result is `(levels, points)`.
    pub fn gather(&self, values: &[f32], levels: usize) -> Vec<f32> {
        let points = self.cells.len();
        let mut out = vec![0.0f32; levels * points];
        if points == 0 { return out; }
        out.par_chunks_mut(points).enumerate().for_each(|(level, plane)| {
          // A surface field has one level. Split its points too, keeping
          // each point's three-term sum in exactly the original order.
          plane.par_chunks_mut(4096).enumerate().for_each(|(band, target)| {
            let first = band * 4096;
            for (local, value) in target.iter_mut().enumerate() {
                let slot = first + local;
                let cells = &self.cells[slot];
                let w = &self.weights[slot];
                let mut acc = 0.0f64;
                for k in 0..3 {
                    acc += f64::from(w[k]) * f64::from(values[cells[k] as usize * levels + level]);
                }
                *value = acc as f32;
            }
          });
        });
        out
    }
}

/// Sparse area weights: per render pixel, the overlapping cells and the
/// unit-sphere area of each overlap.
#[derive(Debug, Clone)]
pub struct ConservativeWeights {
    pub offsets: Vec<u32>,
    pub cells: Vec<u32>,
    /// Overlap area on the unit sphere.
    pub overlap: Vec<f64>,
    /// Sum of overlaps per pixel (the sampled pixel area on the mesh).
    pub pixel_area: Vec<f64>,
    pub samples_min: u32,
    pub samples_max: u32,
    pub samples_total: u64,
    /// Per-cell sampled area (sum of its overlaps over all pixels).
    pub sampled_cell_area: Vec<f64>,
}

/// How many sub-samples per pixel side a pixel of `pixel_km` needs over cells
/// of `cell_km`: at least [`CONSERVATIVE_SAMPLES_PER_CELL`] across the
/// smallest cell, never fewer than [`CONSERVATIVE_MIN_SIDE`].
///
/// MEASURED (2026-10-12) at 4 across a cell and a 6-sample floor: the
/// window integral of RAINC+RAINNC was within 1.9e-5 of the mesh on a
/// uniform 30 km mesh at its own spacing, but 3.5e-4 to 9.0e-4 low on a
/// 48-3 km mesh drawn globally at 48 km, where one pixel holds 250 cells.
/// Eight across a cell costs under a second more per window.
pub const CONSERVATIVE_SAMPLES_PER_CELL: f64 = 8.0;
pub const CONSERVATIVE_MIN_SIDE: u32 = 8;
pub const CONSERVATIVE_MAX_SIDE: u32 = 160;

fn normalized(v: [f64; 3]) -> [f64; 3] {
    let n = dot(v, v).sqrt();
    [v[0] / n, v[1] / n, v[2] / n]
}

/// Build the conservative weights for a `ny x nx` grid of pixel centres.
///
/// Pixel corners are the normalised mean of the four surrounding centres
/// (edges extrapolated one ghost row/column), which is the exact cell
/// boundary for a lat-lon grid and for a conformal projection to second
/// order in the pixel size.  Inside a pixel the footprint is the radial
/// projection of the bilinear patch through its corners; each sample's
/// weight is that projection's area element `|x . (x_s x x_t)| / |x|^3`.
pub(crate) fn build_conservative(
    tree: &KdTree,
    spacing_m: &[f64],
    n_cells: usize,
    centres_grid: &[[f64; 3]],
    ny: usize,
    nx: usize,
    off_mesh: &[bool],
) -> ConservativeWeights {
    let at = |j: isize, i: isize| -> [f64; 3] {
        // Ghost extrapolation: 2*edge - inner, which keeps a lat-lon grid's
        // ghost on its own meridian/parallel (it crosses the pole there).
        let jj = j.clamp(0, ny as isize - 1);
        let ii = i.clamp(0, nx as isize - 1);
        let base = centres_grid[jj as usize * nx + ii as usize];
        let dj = j - jj;
        let di = i - ii;
        if dj == 0 && di == 0 {
            return base;
        }
        let inner_j = (jj - dj).clamp(0, ny as isize - 1);
        let inner_i = (ii - di).clamp(0, nx as isize - 1);
        let inner = centres_grid[inner_j as usize * nx + inner_i as usize];
        normalized([
            2.0 * base[0] - inner[0],
            2.0 * base[1] - inner[1],
            2.0 * base[2] - inner[2],
        ])
    };
    let corner = |j: isize, i: isize| -> [f64; 3] {
        // corner between centres (j-1..j, i-1..i)
        let a = at(j - 1, i - 1);
        let b = at(j - 1, i);
        let c = at(j, i - 1);
        let d = at(j, i);
        normalized([
            a[0] + b[0] + c[0] + d[0],
            a[1] + b[1] + c[1] + d[1],
            a[2] + b[2] + c[2] + d[2],
        ])
    };
    let rows: Vec<(Vec<(u32, f64)>, u32)> = (0..ny * nx)
        .into_par_iter()
        .map(|slot| {
            if !off_mesh.is_empty() && off_mesh[slot] {
                return (Vec::new(), 0);
            }
            let j = (slot / nx) as isize;
            let i = (slot % nx) as isize;
            let c00 = corner(j, i);
            let c10 = corner(j, i + 1);
            let c01 = corner(j + 1, i);
            let c11 = corner(j + 1, i + 1);
            let centre = centres_grid[slot];
            let (near, _) = tree.nearest(centre);
            // Pixel size from its diagonals, in km on the sphere.
            let diag = |a: [f64; 3], b: [f64; 3]| dot(a, b).clamp(-1.0, 1.0).acos();
            let pixel_km = 0.5 * (diag(c00, c11) + diag(c10, c01)) / std::f64::consts::SQRT_2
                * MPAS_EARTH_RADIUS_M
                / 1000.0;
            let cell_km = spacing_m[near as usize] / 1000.0;
            let side = ((CONSERVATIVE_SAMPLES_PER_CELL * pixel_km / cell_km).ceil() as u32)
                .clamp(CONSERVATIVE_MIN_SIDE, CONSERVATIVE_MAX_SIDE);
            let mut acc: Vec<(u32, f64)> = Vec::with_capacity(8);
            let inv = 1.0 / side as f64;
            for sj in 0..side {
                let t = (sj as f64 + 0.5) * inv;
                for si in 0..side {
                    let s = (si as f64 + 0.5) * inv;
                    let mut x = [0.0; 3];
                    let mut xs = [0.0; 3];
                    let mut xt = [0.0; 3];
                    for k in 0..3 {
                        x[k] = (1.0 - s) * (1.0 - t) * c00[k]
                            + s * (1.0 - t) * c10[k]
                            + (1.0 - s) * t * c01[k]
                            + s * t * c11[k];
                        xs[k] = (1.0 - t) * (c10[k] - c00[k]) + t * (c11[k] - c01[k]);
                        xt[k] = (1.0 - s) * (c01[k] - c00[k]) + s * (c11[k] - c10[k]);
                    }
                    let r2 = dot(x, x);
                    let jac = dot(x, cross(xs, xt)).abs() / (r2 * r2.sqrt());
                    let w = jac * inv * inv;
                    let (cell, _) = tree.nearest(normalized(x));
                    match acc.iter_mut().find(|(c, _)| *c == cell) {
                        Some(entry) => entry.1 += w,
                        None => acc.push((cell, w)),
                    }
                }
            }
            acc.sort_by_key(|(c, _)| *c);
            (acc, side * side)
        })
        .collect();
    let mut out = ConservativeWeights {
        offsets: Vec::with_capacity(ny * nx + 1),
        cells: Vec::new(),
        overlap: Vec::new(),
        pixel_area: Vec::with_capacity(ny * nx),
        samples_min: u32::MAX,
        samples_max: 0,
        samples_total: 0,
        sampled_cell_area: vec![0.0; n_cells],
    };
    out.offsets.push(0);
    for (acc, samples) in rows {
        let mut area = 0.0;
        for (cell, w) in acc {
            out.cells.push(cell);
            out.overlap.push(w);
            out.sampled_cell_area[cell as usize] += w;
            area += w;
        }
        out.pixel_area.push(area);
        out.offsets.push(out.cells.len() as u32);
        if samples > 0 {
            out.samples_min = out.samples_min.min(samples);
            out.samples_max = out.samples_max.max(samples);
            out.samples_total += u64::from(samples);
        }
    }
    if out.samples_min == u32::MAX {
        out.samples_min = 0;
    }
    out
}

impl ConservativeWeights {
    /// Area mean over each pixel footprint; NaN where the pixel has none.
    pub fn gather(&self, values: &[f32], levels: usize) -> Vec<f32> {
        let points = self.pixel_area.len();
        let mut out = vec![0.0f32; levels * points];
        if points == 0 { return out; }
        out.par_chunks_mut(points).enumerate().for_each(|(level, plane)| {
          plane.par_chunks_mut(4096).enumerate().for_each(|(band, target)| {
            for (local, value) in target.iter_mut().enumerate() {
                let slot = band * 4096 + local;
                let lo = self.offsets[slot] as usize;
                let hi = self.offsets[slot + 1] as usize;
                if hi == lo || !(self.pixel_area[slot] > 0.0) {
                    *value = OFF_MESH_FILL;
                    continue;
                }
                let mut acc = 0.0f64;
                for e in lo..hi {
                    acc += self.overlap[e] * f64::from(values[self.cells[e] as usize * levels + level]);
                }
                *value = (acc / self.pixel_area[slot]) as f32;
            }
          });
        });
        out
    }

    /// `sum(remapped * pixel_area)` against `sum(value * areaCell)` over the
    /// cells the window covers, for one surface field.  Returns
    /// `(mesh_integral, grid_integral, covered_mesh_area_fraction)` on the
    /// unit sphere; the fraction is 1 for a window covering the whole mesh.
    pub fn budget(&self, values: &[f32], area_unit: &[f64], remapped: &[f32]) -> (f64, f64, f64) {
        let mesh: f64 = values
            .iter()
            .zip(area_unit.iter())
            .map(|(v, a)| f64::from(*v) * a)
            .sum();
        let grid: f64 = remapped
            .iter()
            .zip(self.pixel_area.iter())
            .filter(|(v, a)| v.is_finite() && **a > 0.0)
            .map(|(v, a)| f64::from(*v) * a)
            .sum();
        let covered: f64 = self.sampled_cell_area.iter().sum::<f64>() / area_unit.iter().sum::<f64>();
        (mesh, grid, covered)
    }
}

/// Everything a product conversion needs beyond the nearest map.
pub struct ProductRemap<'a> {
    pub nearest: &'a NearestCellWeights,
    pub linear: LinearWeights,
    pub conservative: ConservativeWeights,
    pub area_unit: Vec<f64>,
    pub sha256: String,
    pub build_seconds: f64,
}

impl<'a> ProductRemap<'a> {
    /// Build both maps for the nearest map's window.
    pub fn build(
        mesh: &MeshCoordinates,
        topology: &MeshTopology,
        spacing_m: &[f64],
        nearest: &'a NearestCellWeights,
    ) -> MpasResult<ProductRemap<'a>> {
        let started = std::time::Instant::now();
        if topology.n_cells != mesh.n_cells() || spacing_m.len() != mesh.n_cells() {
            return Err(MpasError::Refusal(format!(
                "the product remap's topology ({} cells), spacing ({}) and centres ({}) disagree",
                topology.n_cells,
                spacing_m.len(),
                mesh.n_cells()
            )));
        }
        let centres: Vec<[f64; 3]> = (0..mesh.n_cells())
            .map(|i| unit_vector(mesh.latitude_degrees[i], mesh.longitude_degrees[i]))
            .collect();
        let targets: Vec<[f64; 3]> = nearest
            .target_latitude
            .iter()
            .zip(nearest.target_longitude.iter())
            .map(|(&la, &lo)| unit_vector(la, lo))
            .collect();
        let linear = build_linear(&centres, topology, &nearest.cell_index, &targets);
        let tree = KdTree::build(centres);
        let conservative = build_conservative(
            &tree,
            spacing_m,
            mesh.n_cells(),
            &targets,
            nearest.ny,
            nearest.nx,
            &nearest.off_mesh,
        );
        let sha256 = remap_digest(&nearest.weights_sha256, &linear, &conservative);
        Ok(ProductRemap {
            nearest,
            linear,
            conservative,
            area_unit: topology.area_unit.clone(),
            sha256,
            build_seconds: started.elapsed().as_secs_f64(),
        })
    }

    /// One field through its class's remap, `(levels, ny, nx)`, with the
    /// footprint mask applied.
    pub fn gather(&self, class: RemapClass, values: &[f32], levels: usize) -> MpasResult<Vec<f32>> {
        if values.len() != self.nearest.n_cells * levels {
            return Err(MpasError::Refusal(format!(
                "field carries {} value(s); mesh has {} cells x {levels} level(s)",
                values.len(),
                self.nearest.n_cells
            )));
        }
        let mut out = match class {
            RemapClass::Nearest => return self.nearest.gather(values, levels),
            RemapClass::Linear => self.linear.gather(values, levels),
            RemapClass::Conservative => self.conservative.gather(values, levels),
        };
        if !self.nearest.off_mesh.is_empty() {
            let points = self.nearest.off_mesh.len();
            for level in 0..levels {
                for (slot, &outside) in self.nearest.off_mesh.iter().enumerate() {
                    if outside {
                        out[level * points + slot] = OFF_MESH_FILL;
                    }
                }
            }
        }
        Ok(out)
    }

    /// The receipt form: what the remap is, plus how long this process took
    /// to build it.
    pub fn summary_json(&self) -> String {
        self.json(Some(self.build_seconds))
    }

    /// The form a FRAME carries (`MPAS_PRODUCT_REMAP`): what the remap is
    /// and nothing about the process that built it.  The build wall time
    /// was written into every frame, so two conversions of the same history
    /// never produced the same bytes (measured 2026-10-12: two bytes of a
    /// 2.9 GB frame, the last digits of that time), and a frame's receipt
    /// `output_sha256` could not be reproduced by converting it again.
    pub fn identity_json(&self) -> String {
        self.json(None)
    }

    fn json(&self, build_seconds: Option<f64>) -> String {
        let c = &self.conservative;
        let covered: f64 =
            c.sampled_cell_area.iter().sum::<f64>() / self.area_unit.iter().sum::<f64>();
        let build = match build_seconds {
            Some(seconds) => format!(" \"build_seconds\": {seconds:.3},"),
            None => String::new(),
        };
        format!(
            "{{\"schema\": \"{PRODUCT_REMAP_SCHEMA}\", \"sha256\": \"{}\",{} \
             \"linear\": {{\"contained\": {}, \"widened\": {}, \"clamped\": {}}}, \
             \"conservative\": {{\"samples_per_pixel_min\": {}, \"samples_per_pixel_max\": {}, \
             \"samples_total\": {}, \"entries\": {}, \"covered_mesh_area_fraction\": {:.9}}}}}",
            self.sha256,
            build,
            self.linear.contained,
            self.linear.widened,
            self.linear.clamped,
            c.samples_min,
            c.samples_max,
            c.samples_total,
            c.cells.len(),
            covered
        )
    }
}

fn remap_digest(nearest: &str, linear: &LinearWeights, conservative: &ConservativeWeights) -> String {
    let mut h = Sha256::new();
    h.update(PRODUCT_REMAP_SCHEMA.as_bytes());
    h.update([0u8]);
    h.update(nearest.as_bytes());
    for (cells, w) in linear.cells.iter().zip(linear.weights.iter()) {
        for k in 0..3 {
            h.update(cells[k].to_le_bytes());
            h.update(w[k].to_le_bytes());
        }
    }
    for o in &conservative.offsets {
        h.update(o.to_le_bytes());
    }
    for (c, w) in conservative.cells.iter().zip(conservative.overlap.iter()) {
        h.update(c.to_le_bytes());
        h.update(w.to_le_bytes());
    }
    crate::weights::hex(&h.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An icosahedron-derived toy mesh: 12 generators, their 20 triangles as
    /// vertices.  Enough for exactness and conservation checks on the sphere.
    fn icosahedron() -> (Vec<[f64; 3]>, MeshTopology) {
        let t = (1.0 + 5f64.sqrt()) / 2.0;
        let raw = [
            [-1.0, t, 0.0], [1.0, t, 0.0], [-1.0, -t, 0.0], [1.0, -t, 0.0],
            [0.0, -1.0, t], [0.0, 1.0, t], [0.0, -1.0, -t], [0.0, 1.0, -t],
            [t, 0.0, -1.0], [t, 0.0, 1.0], [-t, 0.0, -1.0], [-t, 0.0, 1.0],
        ];
        let centres: Vec<[f64; 3]> = raw.iter().map(|v| normalized(*v)).collect();
        let faces: [[u32; 3]; 20] = [
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
        ];
        let mut vertex_offsets = vec![0u32];
        let mut vertices = Vec::new();
        for c in 0..12u32 {
            for (f, face) in faces.iter().enumerate() {
                if face.contains(&c) {
                    vertices.push(f as u32);
                }
            }
            vertex_offsets.push(vertices.len() as u32);
        }
        let area = 4.0 * std::f64::consts::PI / 12.0;
        (
            centres,
            MeshTopology {
                n_cells: 12,
                cells_on_vertex: faces.to_vec(),
                vertex_offsets,
                vertices,
                area_unit: vec![area; 12],
            },
        )
    }

    fn lcg(state: &mut u64) -> f64 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*state >> 11) as f64) / ((1u64 << 53) as f64)
    }

    #[test]
    fn barycentric_reproduces_the_vertices_and_sums_to_one() {
        let a = normalized([1.0, 0.0, 0.0]);
        let b = normalized([0.0, 1.0, 0.0]);
        let c = normalized([0.0, 0.0, 1.0]);
        assert_eq!(spherical_barycentric(a, a, b, c).map(|w| w.map(|x| (x * 1e12).round())), Some([1e12, 0.0, 0.0]));
        let p = normalized([0.3, 0.3, 0.4]);
        let w = spherical_barycentric(p, a, b, c).unwrap();
        assert!((w[0] + w[1] + w[2] - 1.0).abs() < 1e-14);
        assert!(min3(&w) > 0.0);
        // The far side of the sphere is no triangle's interior.
        assert!(spherical_barycentric([-p[0], -p[1], -p[2]], a, b, c).is_none());
    }

    /// The interpolant through cell values f(x) = a . x is exactly the
    /// gnomonic-plane linear function, so a field linear in the 3-D position
    /// comes back to round-off after normalising by the projection.
    #[test]
    fn linear_weights_reproduce_a_field_linear_in_position() {
        let (centres, topo) = icosahedron();
        let coeff = [0.7, -1.3, 2.1];
        let mut state = 7u64;
        let targets: Vec<[f64; 3]> = (0..4000)
            .map(|_| normalized([lcg(&mut state) - 0.5, lcg(&mut state) - 0.5, lcg(&mut state) - 0.5]))
            .collect();
        let tree = KdTree::build(centres.clone());
        let nearest: Vec<i32> = targets.iter().map(|p| tree.nearest(*p).0 as i32).collect();
        let lw = build_linear(&centres, &topo, &nearest, &targets);
        assert_eq!(lw.clamped, 0, "a closed mesh has a triangle for every point");
        for (k, p) in targets.iter().enumerate() {
            let tri = lw.cells[k];
            let (a, b, c) = (centres[tri[0] as usize], centres[tri[1] as usize], centres[tri[2] as usize]);
            // Exact linear interpolation of g(x)=coeff.x over the planar
            // triangle at the radial projection q of p onto it.
            let n = cross([b[0] - a[0], b[1] - a[1], b[2] - a[2]], [c[0] - a[0], c[1] - a[1], c[2] - a[2]]);
            let s = dot(n, a) / dot(n, *p);
            let q = [p[0] * s, p[1] * s, p[2] * s];
            let want = dot(coeff, q);
            let got: f64 = (0..3)
                .map(|i| f64::from(lw.weights[k][i]) * dot(coeff, centres[tri[i] as usize]))
                .sum();
            assert!((got - want).abs() < 1e-5, "point {k}: {got} vs {want}");
        }
    }

    #[test]
    fn conservative_weights_tile_the_sphere_and_conserve() {
        let (centres, topo) = icosahedron();
        // A global 2-degree lat-lon grid of pixel centres.
        let (ny, nx) = (90usize, 180usize);
        let mut grid = Vec::with_capacity(ny * nx);
        for j in 0..ny {
            let lat = -90.0 + 1.0 + 2.0 * j as f64;
            for i in 0..nx {
                let lon = -180.0 + 1.0 + 2.0 * i as f64;
                grid.push(unit_vector(lat, lon));
            }
        }
        let tree = KdTree::build(centres.clone());
        let spacing = vec![7.0e6; 12];
        let cw = build_conservative(&tree, &spacing, 12, &grid, ny, nx, &[]);
        let total: f64 = cw.pixel_area.iter().sum();
        assert!((total / (4.0 * std::f64::consts::PI) - 1.0).abs() < 1e-5, "pixels tile the sphere: {total}");
        for (c, s) in cw.sampled_cell_area.iter().enumerate() {
            assert!((s / topo.area_unit[c] - 1.0).abs() < 0.01, "cell {c} sampled area {s}");
        }
        let values: Vec<f32> = (0..12).map(|c| (c as f32) * 1.5 + 0.25).collect();
        let out = cw.gather(&values, 1);
        let (mesh, grid_integral, covered) = cw.budget(&values, &topo.area_unit, &out);
        assert!((covered - 1.0).abs() < 1e-5);
        assert!((grid_integral / mesh - 1.0).abs() < 2e-3, "{grid_integral} vs {mesh}");
        // A uniform field stays exactly uniform: every pixel is a mean.
        let flat = cw.gather(&vec![3.5f32; 12], 1);
        assert!(flat.iter().all(|v| (*v - 3.5).abs() < 1e-5));
    }
}
