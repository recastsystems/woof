//! LANE 1.  Parent-extent statics-corridor geometry and crop.
//!
//! Ports the MATH of `gpuwm/static/corridor.py`: `corridor_geometry`,
//! `corridor_grid` (translated + re-extented reference grid),
//! `grid_identity_probes`, `corridor_cost`, and
//! `ChildStaticsCorridor.crop` slice arithmetic.  The receipt/sealing
//! POLICY (JSON receipt equality, digest relay, refusal wording) stays
//! in Python -- it is verification orchestration, not data-path
//! processing -- but the corridor FIELD BYTES are built by lane 2's
//! `build_static` on the corridor grid and sealed through
//! [`crate::npz`], so no array byte crosses the boundary un-Rusted.
//!
//! The bitwise contract carried over verbatim: a footprint cropped from
//! the corridor equals the statics built directly for that footprint
//! (identical source + identical cells = identical bytes), which holds
//! because the corridor grid delegates per-cell transforms to the
//! reference grid's own float arithmetic.

use crate::error::{Result, StaticError};
use crate::projection::ProjectedGrid;
use crate::types::{Field, FieldSet, Grid2, Stack3};

/// Placement-independent corridor geometry for one child
/// (`corridor_geometry`).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct CorridorGeometry {
    pub grid_id: i64,
    pub parent_id: i64,
    pub parent_grid_ratio: i64,
    pub frame_grid_id: i64,
    pub ratio_to_frame: i64,
    pub reference_origin_child_cells: (i64, i64),
    pub reference_i_parent_start: i64,
    pub reference_j_parent_start: i64,
    pub child_nx: i64,
    pub child_ny: i64,
    pub parent_nx: i64,
    pub parent_ny: i64,
    pub corridor_nx: i64,
    pub corridor_ny: i64,
    /// Whole-cell translation from the child's reference grid to the
    /// corridor origin (parent cell 1).
    pub origin_translation_child_cells: (i64, i64),
}

impl CorridorGeometry {
    /// Pure arithmetic; refuses ratio < 1.  Corridor cell 1 is the
    /// first child subcell of parent cell 1, so a footprint at
    /// `i_parent_start = ip` occupies corridor cells
    /// `(ip-1)*ratio + 1 .. (ip-1)*ratio + nx` -- a pure index crop for
    /// every admissible placement.
    #[allow(clippy::too_many_arguments)]
    pub fn derive(
        grid_id: i64,
        parent_id: i64,
        parent_grid_ratio: i64,
        i_parent_start: i64,
        j_parent_start: i64,
        child_nx: i64,
        child_ny: i64,
        parent_nx: i64,
        parent_ny: i64,
    ) -> Result<Self> {
        let ratio = parent_grid_ratio;
        if ratio < 1 {
            return Err(StaticError::Invalid(format!(
                "parent_grid_ratio must be >= 1, got {ratio}"
            )));
        }
        Ok(CorridorGeometry {
            grid_id,
            parent_id,
            parent_grid_ratio: ratio,
            frame_grid_id: parent_id,
            ratio_to_frame: ratio,
            reference_origin_child_cells: (
                (i_parent_start - 1) * ratio,
                (j_parent_start - 1) * ratio,
            ),
            reference_i_parent_start: i_parent_start,
            reference_j_parent_start: j_parent_start,
            child_nx,
            child_ny,
            parent_nx,
            parent_ny,
            corridor_nx: parent_nx * ratio,
            corridor_ny: parent_ny * ratio,
            origin_translation_child_cells: (
                (1 - i_parent_start) * ratio,
                (1 - j_parent_start) * ratio,
            ),
        })
    }
}

/// What one child's corridor costs, without building it
/// (`corridor_cost`).  The plane inventory is the Python-side native
/// static contract's business, so the count crosses as an argument.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct CorridorCost {
    pub grid_id: i64,
    pub parent_id: i64,
    pub corridor_nx: i64,
    pub corridor_ny: i64,
    pub cells: i64,
    pub planes_per_cell: i64,
    pub bytes_per_cell: i64,
    pub host_bytes: i64,
}

/// Pure arithmetic on the geometry and the plane inventory.
pub fn corridor_cost(
    geometry: &CorridorGeometry,
    planes_per_cell: i64,
) -> CorridorCost {
    let cells = geometry.corridor_nx * geometry.corridor_ny;
    let bytes_per_cell = planes_per_cell * std::mem::size_of::<f64>() as i64;
    CorridorCost {
        grid_id: geometry.grid_id,
        parent_id: geometry.parent_id,
        corridor_nx: geometry.corridor_nx,
        corridor_ny: geometry.corridor_ny,
        cells,
        planes_per_cell,
        bytes_per_cell,
        host_bytes: cells * bytes_per_cell,
    }
}

/// The corridor's grid: the reference child grid translated onto parent
/// cell 1 and re-extented on the SAME lattice (per-cell reference
/// arithmetic).
pub fn corridor_grid(
    reference: &ProjectedGrid,
    geometry: &CorridorGeometry,
) -> Result<ProjectedGrid> {
    let (di, dj) = geometry.origin_translation_child_cells;
    reference.translated(
        di,
        dj,
        Some(geometry.corridor_nx + 1),
        Some(geometry.corridor_ny + 1),
    )
}

/// Exact float64 lat/lon probes pinning grid arithmetic
/// (`grid_identity_probes`): sw/se/nw/ne/center in that order, each
/// `[lat, lon]`.  Byte equality between preparation and run is what the
/// runner's verification gate compares.
pub fn grid_identity_probes(
    grid: &ProjectedGrid,
) -> Result<Vec<(String, [f64; 2])>> {
    let nx = grid.spec.e_we - 1;
    let ny = grid.spec.e_sn - 1;
    let points: [(&str, f64, f64); 5] = [
        ("sw", 1.0, 1.0),
        ("se", nx as f64, 1.0),
        ("nw", 1.0, ny as f64),
        ("ne", nx as f64, ny as f64),
        (
            "center",
            grid.spec.e_we as f64 / 2.0,
            grid.spec.e_sn as f64 / 2.0,
        ),
    ];
    Ok(points
        .iter()
        .map(|&(name, x, y)| {
            let (lat, lon) = grid.ij_to_latlon(x, y);
            (name.to_string(), [lat, lon])
        })
        .collect())
}

/// Crop one placement's footprint out of corridor fields
/// (`ChildStaticsCorridor.crop`); refuses off-corridor placements by
/// the same sentence the Python does.
pub fn crop(
    corridor_fields: &FieldSet,
    geometry: &CorridorGeometry,
    i_parent_start: i64,
    j_parent_start: i64,
) -> Result<FieldSet> {
    let ratio = geometry.parent_grid_ratio;
    let nx = geometry.child_nx;
    let ny = geometry.child_ny;
    let ip = i_parent_start;
    let jp = j_parent_start;
    let x0 = (ip - 1) * ratio;
    let y0 = (jp - 1) * ratio;
    if ip < 1 || jp < 1 || x0 + nx > geometry.corridor_nx
        || y0 + ny > geometry.corridor_ny
    {
        return Err(StaticError::Invalid(format!(
            "placement (i_parent_start={ip}, j_parent_start={jp}) lies \
             outside the statics corridor ({}x{} child cells over the \
             whole parent); an admissible move cannot reach here, so \
             this is a wiring defect",
            geometry.corridor_nx, geometry.corridor_ny
        )));
    }
    let (x0, y0, nx, ny) = (x0 as usize, y0 as usize, nx as usize, ny as usize);
    let mut out = FieldSet::default();
    for (name, field) in &corridor_fields.fields {
        let cropped = match field {
            Field::Plane(grid) => {
                let mut plane = Grid2::filled(ny, nx, 0.0);
                for j in 0..ny {
                    let src = (y0 + j) * grid.nx + x0;
                    plane.data[j * nx..(j + 1) * nx]
                        .copy_from_slice(&grid.data[src..src + nx]);
                }
                Field::Plane(plane)
            }
            Field::Stack(stack) => {
                let mut cropped = Stack3::filled(stack.planes, ny, nx, 0.0);
                let src_plane = stack.ny * stack.nx;
                let dst_plane = ny * nx;
                for z in 0..stack.planes {
                    for j in 0..ny {
                        let src = z * src_plane + (y0 + j) * stack.nx + x0;
                        let dst = z * dst_plane + j * nx;
                        cropped.data[dst..dst + nx]
                            .copy_from_slice(&stack.data[src..src + nx]);
                    }
                }
                Field::Stack(cropped)
            }
        };
        out.fields.insert(name.clone(), cropped);
    }
    Ok(out)
}
