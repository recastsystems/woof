//! A moving nest's frames, embedded on their fixed native union lattice.
//!
//! A following nest moves by whole parent cells, and every history frame
//! records where it sat: WRF's `I_PARENT_START`, `J_PARENT_START` and
//! `PARENT_GRID_RATIO` global attributes, which gpuwm's history writer
//! refreshes after every move (`gpuwm/runtime.py`, the post-move seam).  Two
//! frames of one nest at different places therefore share every cell of
//! their overlap exactly, a whole number of nest cells apart, with no
//! interpolation between them.
//!
//! A store holds one grid, so a series that spans a move used to be split
//! into one store per place (`gpuwm/render.py`'s series grouping keys on the
//! placement and the coordinates), and a window whose earlier frame sat at
//! the old place had nothing to difference: the 1 h rain of a 3 km storm-
//! following nest was drawn only at the hours in which the nest did not move
//! (4 of 12 on a live tropical storm), and the render step ended failed.
//!
//! Compatible recorded placements use a bounding grid covering every frame.
//! Coordinates are checked before embedding, and departed strips survive.
//! For moving ancestors whose parent-relative placements cannot define that
//! fixed Earth frame, the coordinate-verified last-frame fallback remains.
//! In that fallback an earlier frame moves onto the series' LAST frame:
//! the cell that sits on the same ground takes its value, and a cell the
//! earlier place did not cover holds NaN, the store's missing value.  A
//! window folded from it is missing there for that window, never zero and
//! never borrowed from a neighbour.  The move is checked against the frames'
//! own coordinates on the whole overlap before any value is taken.

use rustwx_core::LatLonGrid;
use wrf_core::WrfFile;

/// Where one wrfout's grid sits in its parent, as the frame records it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct NestPlacement {
    pub(crate) grid_id: i32,
    pub(crate) parent_grid_ratio: i32,
    pub(crate) i_parent_start: i32,
    pub(crate) j_parent_start: i32,
}

impl NestPlacement {
    /// The placement a raw wrfout records, or `None` when it records none
    /// (a root domain carries `I_PARENT_START = 1` and ratio 1, which is a
    /// placement like any other).
    pub(crate) fn of(file: &WrfFile) -> Option<Self> {
        let placement = Self {
            grid_id: file.global_attr_i32("GRID_ID").ok()?,
            parent_grid_ratio: file.global_attr_i32("PARENT_GRID_RATIO").ok()?,
            i_parent_start: file.global_attr_i32("I_PARENT_START").ok()?,
            j_parent_start: file.global_attr_i32("J_PARENT_START").ok()?,
        };
        (placement.parent_grid_ratio > 0).then_some(placement)
    }
}

/// The whole-cell move between two places of one nest: cell `(x, y)` of the
/// later place sits on the ground of cell `(x + dx, y + dy)` of the earlier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct NestShift {
    pub(crate) dx: i64,
    pub(crate) dy: i64,
}

impl NestShift {
    /// A move that leaves no cell of any grid on the target: the frame is
    /// stored missing everywhere ([`shift_plane`]).
    pub(crate) const NOWHERE: Self = Self { dx: i64::MAX / 4, dy: 0 };

    /// The move from `from` to `to`, or `None` when they are not two places
    /// of one nest (another grid, or another refinement ratio).
    ///
    /// A nest cell `x` lies at parent coordinate `i_parent_start + x / ratio`
    /// (plus a stagger offset both places share), so the same ground is
    /// `x_from = x_to + (i_to - i_from) * ratio`.
    pub(crate) fn between(from: NestPlacement, to: NestPlacement) -> Option<Self> {
        if from.grid_id != to.grid_id || from.parent_grid_ratio != to.parent_grid_ratio {
            return None;
        }
        let ratio = i64::from(to.parent_grid_ratio);
        Some(Self {
            dx: (i64::from(to.i_parent_start) - i64::from(from.i_parent_start)) * ratio,
            dy: (i64::from(to.j_parent_start) - i64::from(from.j_parent_start)) * ratio,
        })
    }

    pub(crate) fn is_null(self) -> bool {
        self.dx == 0 && self.dy == 0
    }

    /// Whether any cell of an `nx` by `ny` nest sits, after this move, on
    /// ground the nest covered before it.
    pub(crate) fn shares_cell(self, nx: usize, ny: usize) -> bool {
        self.dx.unsigned_abs() < nx as u64 && self.dy.unsigned_abs() < ny as u64
    }

    /// The earlier place's cell under cell `(x, y)` of the later place.
    fn source(self, x: usize, y: usize, nx: usize, ny: usize) -> Option<usize> {
        let sx = i64::try_from(x).ok()? + self.dx;
        let sy = i64::try_from(y).ok()? + self.dy;
        let sx = usize::try_from(sx).ok().filter(|&sx| sx < nx)?;
        let sy = usize::try_from(sy).ok().filter(|&sy| sy < ny)?;
        Some(sy * nx + sx)
    }
}

/// One row-major `ny * nx` plane of the earlier place, on the later place:
/// NaN wherever the earlier place did not cover the ground.
pub(crate) fn shift_plane(values: &[f32], nx: usize, ny: usize, shift: NestShift) -> Vec<f32> {
    let mut shifted = vec![f32::NAN; nx * ny];
    if values.len() != nx * ny {
        return shifted;
    }
    for y in 0..ny {
        for x in 0..nx {
            if let Some(source) = shift.source(x, y, nx, ny) {
                shifted[y * nx + x] = values[source];
            }
        }
    }
    shifted
}

/// Copy a native placement onto the bounding fixed frame of all placements.
/// No interpolation and no cropped departure footprint. Holes stay missing.
pub(crate) fn embed_plane(values: &[f32], source: (usize, usize), target: (usize, usize),
                          offset: (usize, usize)) -> Result<Vec<f32>, String> {
    let (nx, ny) = source;
    let (tx, ty) = target;
    let (x, y) = offset;
    if values.len() != nx.checked_mul(ny).ok_or("source grid overflow")?
        || x.checked_add(nx).is_none_or(|n| n > tx)
        || y.checked_add(ny).is_none_or(|n| n > ty) {
        return Err("nest footprint lies outside its fixed frame".into());
    }
    let mut result = vec![f32::NAN; tx.checked_mul(ty).ok_or("fixed grid overflow")?];
    for row in 0..ny {
        result[(row+y)*tx+x..(row+y)*tx+x+nx].copy_from_slice(&values[row*nx..(row+1)*nx]);
    }
    Ok(result)
}

#[cfg(test)]
mod swath_tests {
    use super::embed_plane;
    #[test]
    fn departed_and_exposed_strips_survive_fixed_frame_embedding() {
        let first = embed_plane(&[25., 20., 25., 20.], (2, 2), (3, 2), (0, 0)).unwrap();
        let last = embed_plane(&[30., 35., 30., 35.], (2, 2), (3, 2), (1, 0)).unwrap();
        let maxima: Vec<_> = first.iter().zip(&last).map(|(a,b)| a.max(*b)).collect();
        assert_eq!(maxima, vec![25., 30., 35., 25., 30., 35.]);
        assert!(first[2].is_nan() && last[0].is_nan());
    }
}

/// How far two coordinates of one cell may differ and still be the same
/// ground: a twentieth of the grid spacing, in degrees of latitude, and never
/// less than 4e-5 degrees (a few float32 steps of a longitude near 180).
/// Both places compute their coordinates from one projection, so the same
/// ground agrees to float32 rounding; a place one cell off misses by twenty
/// times this.
pub(crate) fn coordinate_tolerance_deg(dx_m: f64) -> f64 {
    const METRES_PER_DEGREE: f64 = 111_195.0;
    if dx_m.is_finite() && dx_m > 0.0 {
        (dx_m / METRES_PER_DEGREE / 20.0).max(4.0e-5)
    } else {
        4.0e-5
    }
}

/// Prove that `earlier`, moved by `shift`, lands on `later`: every cell of
/// the overlap has the same latitude and longitude within `tolerance_deg`.
/// Returns the number of cells the two places share.
///
/// WHAT BREAKAGE THIS PREVENTS (gate law): a placement that does not match
/// the frames' own coordinates (a descendant that rode along with a moving
/// parent records its place in that parent, not on the ground, and a file
/// edited by hand records whatever it was given) would subtract the rain of
/// one piece of ground from another's and draw the result as an hour of rain.
pub(crate) fn verify_shift(
    earlier: &LatLonGrid,
    later: &LatLonGrid,
    shift: NestShift,
    tolerance_deg: f64,
) -> Result<usize, String> {
    let (nx, ny) = (later.shape.nx, later.shape.ny);
    if (earlier.shape.nx, earlier.shape.ny) != (nx, ny) {
        return Err(format!(
            "the earlier frame is {}x{} and the later one {nx}x{ny}; one nest keeps its size \
             when it moves",
            earlier.shape.nx, earlier.shape.ny
        ));
    }
    let mut shared = 0usize;
    let mut off = 0usize;
    let mut worst = 0.0f64;
    for y in 0..ny {
        for x in 0..nx {
            let Some(source) = shift.source(x, y, nx, ny) else {
                continue;
            };
            let target = y * nx + x;
            shared += 1;
            let dlat = f64::from(later.lat_deg[target]) - f64::from(earlier.lat_deg[source]);
            let mut dlon = f64::from(later.lon_deg[target]) - f64::from(earlier.lon_deg[source]);
            dlon = (dlon + 540.0).rem_euclid(360.0) - 180.0;
            let miss = dlat.abs().max(dlon.abs());
            if !(miss <= tolerance_deg) {
                off += 1;
                if miss.is_finite() {
                    worst = worst.max(miss);
                } else {
                    worst = f64::INFINITY;
                }
            }
        }
    }
    if shared == 0 {
        return Err(format!(
            "the recorded move ({}, {}) nest cells leaves no cell in common",
            shift.dx, shift.dy
        ));
    }
    if off > 0 {
        return Err(format!(
            "moved by its recorded placement ({}, {}) nest cells, {off} of {shared} shared cells \
             land up to {worst:.6} degrees from the later frame's own coordinates (tolerance \
             {tolerance_deg:.6})",
            shift.dx, shift.dy
        ));
    }
    Ok(shared)
}

/// The whole-cell move that lands `earlier` on `later`, read from their
/// coordinates: for the later grid's centre and each of its corners, the
/// earlier cell nearest it gives a candidate move, and the first whose
/// whole overlap [`verify_shift`] accepts is the move.  Two equal grids
/// that overlap at all hold a corner of the later one on the earlier one's
/// ground, so a partial overlap is found too.  `None` when no whole-cell
/// move fits.
///
/// For a nest whose recorded place does not describe its move on the
/// ground: a descendant riding a parent that moved records its place in
/// that parent, which the parent's move changes under it.
pub(crate) fn locate_shift(
    earlier: &LatLonGrid,
    later: &LatLonGrid,
    tolerance_deg: f64,
) -> Option<(NestShift, usize)> {
    let (nx, ny) = (later.shape.nx, later.shape.ny);
    if (earlier.shape.nx, earlier.shape.ny) != (nx, ny) || nx == 0 || ny == 0 {
        return None;
    }
    let anchors = [
        (nx / 2, ny / 2),
        (0, 0),
        (nx - 1, 0),
        (0, ny - 1),
        (nx - 1, ny - 1),
    ];
    let mut tried = Vec::<NestShift>::with_capacity(anchors.len());
    for (ax, ay) in anchors {
        let lat = f64::from(later.lat_deg[ay * nx + ax]);
        let lon = f64::from(later.lon_deg[ay * nx + ax]);
        let scale = lat.to_radians().cos().max(1.0e-6);
        let nearest = (0..nx * ny).min_by(|&a, &b| {
            let distance = |cell: usize| {
                let dlat = f64::from(earlier.lat_deg[cell]) - lat;
                let dlon = ((f64::from(earlier.lon_deg[cell]) - lon + 540.0).rem_euclid(360.0)
                    - 180.0)
                    * scale;
                dlat * dlat + dlon * dlon
            };
            distance(a).total_cmp(&distance(b))
        })?;
        let shift = NestShift {
            dx: (nearest % nx) as i64 - ax as i64,
            dy: (nearest / nx) as i64 - ay as i64,
        };
        if tried.contains(&shift) {
            continue;
        }
        tried.push(shift);
        if let Ok(shared) = verify_shift(earlier, later, shift, tolerance_deg) {
            return Some((shift, shared));
        }
    }
    None
}

/// The ground under a raw wrfout's first and last mass cells, `(lat, lon)`
/// each, at its first time; `None` when the file has no coordinates.
pub(crate) fn corner_ground(file: &WrfFile) -> Option<[(f32, f32); 2]> {
    let lat = file.xlat(0).ok()?;
    let lon = file.xlong(0).ok()?;
    let (first_lat, last_lat) = (*lat.first()?, *lat.last()?);
    let (first_lon, last_lon) = (*lon.first()?, *lon.last()?);
    Some([
        (first_lat as f32, first_lon as f32),
        (last_lat as f32, last_lon as f32),
    ])
}

/// Whether two frames that record one place sit on different ground: their
/// corner cells are more than [`coordinate_tolerance_deg`] apart.  Unknown
/// ground (no coordinates read) is not a difference.
pub(crate) fn ground_differs(
    ground: Option<[(f32, f32); 2]>,
    other: Option<[(f32, f32); 2]>,
    dx_m: Option<f64>,
) -> bool {
    let (Some(ground), Some(other)) = (ground, other) else {
        return false;
    };
    let tolerance = coordinate_tolerance_deg(dx_m.unwrap_or(f64::NAN));
    ground.iter().zip(other.iter()).any(|(&(lat, lon), &(lat_o, lon_o))| {
        let dlat = f64::from(lat) - f64::from(lat_o);
        let dlon = (f64::from(lon) - f64::from(lon_o) + 540.0).rem_euclid(360.0) - 180.0;
        !(dlat.abs() <= tolerance && dlon.abs() <= tolerance)
    })
}

/// Whether two grids' latitude and longitude spans have nothing in common,
/// so no cell of one lies on the other's ground.
pub(crate) fn grids_disjoint(one: &LatLonGrid, other: &LatLonGrid) -> bool {
    fn spans(grid: &LatLonGrid) -> Option<(f64, f64, f64, f64)> {
        let anchor = f64::from(*grid.lon_deg.iter().find(|v| v.is_finite())?);
        let (mut south, mut north, mut west, mut east) =
            (f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY);
        for (&lat, &lon) in grid.lat_deg.iter().zip(&grid.lon_deg) {
            if !(lat.is_finite() && lon.is_finite()) {
                continue;
            }
            let lon = (f64::from(lon) - anchor + 540.0).rem_euclid(360.0) - 180.0 + anchor;
            south = south.min(f64::from(lat));
            north = north.max(f64::from(lat));
            west = west.min(lon);
            east = east.max(lon);
        }
        Some((south, north, west, east))
    }
    let (Some((s1, n1, w1, e1)), Some((s2, n2, w2, e2))) = (spans(one), spans(other)) else {
        return false;
    };
    let offset = (((w1 + e1) - (w2 + e2)) / 720.0).round() * 360.0;
    let (w2, e2) = (w2 + offset, e2 + offset);
    !(s1 < n2 && s2 < n1 && w1 < e2 && w2 < e1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustwx_core::GridShape;

    fn place(i: i32, j: i32) -> NestPlacement {
        NestPlacement {
            grid_id: 2,
            parent_grid_ratio: 4,
            i_parent_start: i,
            j_parent_start: j,
        }
    }

    /// A 3 km nest's grid at parent start (i, j): a plain lat/lon lattice in
    /// nest cells, which is all the shift arithmetic sees.
    fn grid(i: i32, j: i32, nx: usize, ny: usize) -> LatLonGrid {
        let step = 0.027f64;
        let mut lat = Vec::new();
        let mut lon = Vec::new();
        for y in 0..ny {
            for x in 0..nx {
                let gx = f64::from(i) * 4.0 + x as f64;
                let gy = f64::from(j) * 4.0 + y as f64;
                lat.push((10.0 + gy * step) as f32);
                lon.push((-110.0 + gx * step) as f32);
            }
        }
        LatLonGrid::new(GridShape::new(nx, ny).unwrap(), lat, lon).unwrap()
    }

    #[test]
    fn a_move_west_and_north_leaves_the_new_west_and_north_edges_missing() {
        // The nest moved 2 parent cells west and 1 north: 8 nest cells of new
        // ground on its west edge and 4 on its north edge.
        let (from, to) = (place(81, 61), place(79, 62));
        let shift = NestShift::between(from, to).unwrap();
        assert_eq!(shift, NestShift { dx: -8, dy: 4 });
        let (nx, ny) = (20usize, 16usize);
        let values: Vec<f32> = (0..nx * ny).map(|cell| cell as f32).collect();
        let moved = shift_plane(&values, nx, ny, shift);
        for y in 0..ny {
            for x in 0..nx {
                let value = moved[y * nx + x];
                let (sx, sy) = (x as i64 - 8, y as i64 + 4);
                if sx < 0 || sy >= ny as i64 {
                    assert!(value.is_nan(), "({x}, {y}) is new ground: {value}");
                } else {
                    assert_eq!(value, (sy as usize * nx + sx as usize) as f32, "({x}, {y})");
                }
            }
        }
        // The same ground on both grids, cell for cell.
        let shared = verify_shift(
            &grid(81, 61, nx, ny),
            &grid(79, 62, nx, ny),
            shift,
            coordinate_tolerance_deg(3000.0),
        )
        .unwrap();
        assert_eq!(shared, (nx - 8) * (ny - 4));
    }

    #[test]
    fn a_placement_that_misses_the_coordinates_is_refused_by_name() {
        // Recorded one parent cell east of where the coordinates put it: the
        // breakage is differencing the rain of different ground.
        let shift = NestShift::between(place(81, 61), place(80, 61)).unwrap();
        let error = verify_shift(
            &grid(81, 61, 20, 16),
            &grid(79, 61, 20, 16),
            shift,
            coordinate_tolerance_deg(3000.0),
        )
        .unwrap_err();
        assert!(error.contains("shared cells land up to"), "{error}");
    }

    #[test]
    fn a_move_the_placement_does_not_record_is_read_from_the_coordinates() {
        // A descendant that moved inside a parent that moved too: its
        // recorded place changed by one parent cell, the ground by two.
        let (earlier, later) = (grid(81, 61, 20, 16), grid(79, 62, 20, 16));
        let recorded = NestShift::between(place(81, 61), place(80, 62)).unwrap();
        assert!(
            verify_shift(&earlier, &later, recorded, coordinate_tolerance_deg(3000.0)).is_err()
        );
        let (found, shared) =
            locate_shift(&earlier, &later, coordinate_tolerance_deg(3000.0)).unwrap();
        assert_eq!(found, NestShift { dx: -8, dy: 4 });
        assert_eq!(shared, 12 * 12);
        // Grids that share no whole-cell move are not located.
        let mut skewed = grid(79, 62, 20, 16);
        for lon in skewed.lon_deg.iter_mut() {
            *lon += 0.0135;
        }
        assert!(locate_shift(&earlier, &skewed, coordinate_tolerance_deg(3000.0)).is_none());
    }

    #[test]
    fn a_place_a_whole_width_away_shares_no_cell_and_is_not_located() {
        // A 16-cell-wide nest that travelled 6 parent cells (18 nest cells)
        // east: nothing of the earlier place lies on the later one's ground.
        let shift = NestShift::between(place(5, 5), place(11, 5)).unwrap();
        assert_eq!(shift, NestShift { dx: 24, dy: 0 });
        assert!(!shift.shares_cell(16, 12));
        assert!(NestShift { dx: 15, dy: -11 }.shares_cell(16, 12));
        assert!(!NestShift { dx: 0, dy: 12 }.shares_cell(16, 12));
        let (earlier, later) = (grid(5, 5, 16, 12), grid(11, 5, 16, 12));
        assert!(verify_shift(&earlier, &later, shift, coordinate_tolerance_deg(3000.0)).is_err());
        assert!(locate_shift(&earlier, &later, coordinate_tolerance_deg(3000.0)).is_none());
        // Stored on the later place it is missing everywhere.
        let values: Vec<f32> = (0..16 * 12).map(|cell| cell as f32).collect();
        assert!(shift_plane(&values, 16, 12, shift).iter().all(|value| value.is_nan()));
    }

    #[test]
    fn a_partial_overlap_away_from_the_centre_is_located_from_a_corner() {
        // Moved 3 parent cells (12 nest cells) west and 2 (8) north on a
        // 16 by 12 nest: the later centre is off the earlier ground, but
        // its south-east corner is on it.
        let (earlier, later) = (grid(8, 5, 16, 12), grid(5, 7, 16, 12));
        let (found, shared) =
            locate_shift(&earlier, &later, coordinate_tolerance_deg(3000.0)).unwrap();
        assert_eq!(found, NestShift { dx: -12, dy: 8 });
        assert_eq!(shared, 4 * 4);
    }

    #[test]
    fn a_nest_fixed_in_a_moving_parent_is_another_place_on_the_ground() {
        // One recorded place, ground two parent cells (8 nest cells) west.
        let (earlier, later) = (grid(81, 61, 20, 16), grid(79, 61, 20, 16));
        let corners = |g: &LatLonGrid| {
            let n = g.lat_deg.len() - 1;
            Some([(g.lat_deg[0], g.lon_deg[0]), (g.lat_deg[n], g.lon_deg[n])])
        };
        assert!(ground_differs(corners(&earlier), corners(&later), Some(3000.0)));
        assert!(!ground_differs(corners(&later), corners(&later), Some(3000.0)));
        assert!(!ground_differs(None, corners(&later), Some(3000.0)));
        let recorded = NestShift::between(place(80, 60), place(80, 60)).unwrap();
        assert!(recorded.is_null());
        assert!(verify_shift(&earlier, &later, recorded, coordinate_tolerance_deg(3000.0)).is_err());
        let (found, _) = locate_shift(&earlier, &later, coordinate_tolerance_deg(3000.0)).unwrap();
        assert_eq!(found, NestShift { dx: -8, dy: 0 });
        assert!(!grids_disjoint(&earlier, &later));
        // A whole width away: nothing in common, stored missing.
        let far = grid(90, 61, 20, 16);
        assert!(grids_disjoint(&earlier, &far));
        assert!(shift_plane(&[1.0; 20 * 16], 20, 16, NestShift::NOWHERE).iter().all(|v| v.is_nan()));
    }

    #[test]
    fn another_grid_or_ratio_is_not_a_move() {
        let mut other = place(81, 61);
        other.grid_id = 3;
        assert!(NestShift::between(place(81, 61), other).is_none());
        let mut ratio = place(81, 61);
        ratio.parent_grid_ratio = 3;
        assert!(NestShift::between(place(81, 61), ratio).is_none());
        assert!(NestShift::between(place(81, 61), place(81, 61)).unwrap().is_null());
    }

    #[test]
    fn the_tolerance_scales_with_the_grid_and_keeps_a_float32_floor() {
        assert!((coordinate_tolerance_deg(3000.0) - 3000.0 / 111_195.0 / 20.0).abs() < 1e-12);
        // A 50 m grid's twentieth of a cell is 2.2e-5 degrees: the floor.
        assert_eq!(coordinate_tolerance_deg(50.0), 4.0e-5);
        assert_eq!(coordinate_tolerance_deg(f64::NAN), 4.0e-5);
    }
}
