//! The memory order a generated mesh's cells are numbered in.
//!
//! THE BREAKAGE THIS PREVENTS, MEASURED (2026-10-29).  The generator numbered
//! cells in the order it made them: the background crystal first, then every
//! refinement level's insertions appended after it, level by level.  A fine
//! cell's neighbours were therefore spread across the whole index range, and
//! every dycore gather of a neighbour value touched far more memory lines than
//! it needed.  Generated zoom meshes cost 1.6-2.8x the per-cell dycore time of
//! a uniform globe on the same card, and the gap grew with the mesh
//! (HEX-SHAPES-1CARD, one RTX PRO 6000).  On the default 96 -> 3 km zoom globe
//! (149,322 cells), a warp of 32 consecutive edges gathering one float32 level
//! of its two cells touched 10.0 distinct 128-byte lines in generator order
//! (`hex_order --locality`, the census [`neighbour_lines_per_warp`] takes).
//!
//! MEASURED WITH THIS ORDER, same mesh, same GFS 2023-03-31 12Z start, 1 h,
//! MPAS-A convection_permitting, fast numerics, one RTX 5090, one card lease
//! for both runs: the census is 2.31 lines per warp, and the median dycore
//! step is 244.2 ms against 378.3 ms in generator order, 35 % less (the whole
//! forecast 69.7 s against 95.4 s).  The two runs start from the same state:
//! every cell field of the start frame is the same word, and the normal wind
//! differs only where an edge now points the other way (below), by at most
//! 5.8e-5 m/s, the rounding of cos and sin of an angle turned by pi.  By +1 h
//! that rounding has grown the way any last-bit change grows in convection:
//! RMS 0.0015 K in theta and 0.006 K in 2 m temperature, a few kelvin at
//! single convective points.
//!
//! THE ORDER.  Cells are sorted by the Morton (Z-order) key of their centre:
//! each Cartesian coordinate is scaled to 21 bits over the point set's own
//! bounding box and the bits are interleaved, ties broken by the generator's
//! index, so the order is a pure function of the point set.  It is the cell
//! key `hex_order --make-map` uses.  Only names change: every ring keeps its
//! counter-clockwise winding, and
//! [`MpasMesh::derive`](crate::mesh::derive::MpasMesh::derive) then numbers
//! edges and vertices canonically from the renamed cell pairs and triples, so
//! they follow the cells into the same locality with no step of their own and
//! the file stays the canonical numbering a rebuild from its own centres
//! reproduces.  Canonical edges keep `cellsOnEdge` as (lower, higher) label,
//! so about half the edges point the other way after the rename (244,765 of
//! 447,960 on the zoom globe).  That is why a forecast on the renamed mesh
//! matches the generator-order one to rounding and not word for word; a pure
//! file permutation (`hex_order --apply`, every edge kept pointing its old
//! way) measured 247.1 ms and word-identical +1 h history, so the speed is the
//! memory order's, not the orientation's.
//!
//! A rotation of the sphere keeps which cells neighbour which and keeps the
//! numbering, so a zoom globe rotated from a template generated here is in
//! this order too, and a limited area culled from it keeps the parent's
//! relative order.

use rayon::prelude::*;
use serde::Serialize;

use crate::error::{MpasError, MpasResult};
use crate::mesh::derive::{MpasMesh, Rings};
use crate::mesh::geom::V3;

/// The receipt's name for the order `generate` writes.
pub const MORTON_XYZ_21: &str = "morton-xyz-21";

/// What the receipt says about the order a generated file is numbered in.
#[derive(Debug, Clone, Serialize)]
pub struct CellOrder {
    /// The sort key; see the module documentation.
    pub key: &'static str,
    /// The census `hex_order --locality` prints for the emitted numbering:
    /// distinct 128-byte lines one float32 level of `cellsOnEdge` gathers per
    /// warp of 32 consecutive edges, averaged over both cells of every warp.
    /// A static census of the gather pattern, not measured cache traffic.
    pub neighbour_128byte_lines_per_warp: f64,
}

fn spread(v: u64) -> u64 {
    let mut out = 0;
    for bit in 0..21 {
        out |= ((v >> bit) & 1) << (3 * bit);
    }
    out
}

/// `order[new] = old`: the cells sorted by the Morton key of their centres.
pub fn morton_order(points: &[V3]) -> MpasResult<Vec<usize>> {
    if points.iter().any(|p| p.iter().any(|x| !x.is_finite())) {
        return Err(MpasError::Refusal(
            "a cell centre is not finite, so it has no place in a space-filling order and every field indexed by the reorder would be misplaced".into(),
        ));
    }
    let bounds: Vec<(f64, f64)> = (0..3)
        .map(|a| {
            points
                .par_iter()
                .map(|p| (p[a], p[a]))
                .reduce(
                    || (f64::INFINITY, f64::NEG_INFINITY),
                    |x, y| (x.0.min(y.0), x.1.max(y.1)),
                )
        })
        .collect();
    let mut keyed: Vec<(u64, usize)> = points
        .par_iter()
        .enumerate()
        .map(|(i, p)| {
            let mut key = 0;
            for (a, &(lo, hi)) in bounds.iter().enumerate() {
                let v = if hi == lo {
                    0
                } else {
                    (((p[a] - lo) / (hi - lo)) * 2_097_151.0).round() as u64
                };
                key |= spread(v) << a;
            }
            (key, i)
        })
        .collect();
    // `(key, index)` pairs are all distinct, so the unstable sort has one answer.
    keyed.par_sort_unstable();
    Ok(keyed.into_iter().map(|(_, i)| i).collect())
}

/// The cell centres and neighbour rings renamed by `order` (`order[new] = old`).
///
/// Each ring keeps its winding and its starting slot; only the names change.
pub fn renumber(points: &[V3], rings: &Rings, order: &[usize]) -> MpasResult<(Vec<V3>, Rings)> {
    let n = points.len();
    if order.len() != n || rings.n_cells() != n {
        return Err(MpasError::Refusal(format!(
            "a reorder of {} cells was asked for {n} centres and {} rings; the three tables describe different meshes and the renamed rings would name the wrong cells",
            order.len(),
            rings.n_cells()
        )));
    }
    let mut inverse = vec![u32::MAX; n];
    for (new, &old) in order.iter().enumerate() {
        if old >= n || inverse[old] != u32::MAX {
            return Err(MpasError::Refusal(format!(
                "the cell reorder names cell {old} twice or past the end of a {n}-cell mesh; it is not a permutation, so two cells would share one slot and another would vanish"
            )));
        }
        inverse[old] = new as u32;
    }
    let renamed: Vec<V3> = order.par_iter().map(|&old| points[old]).collect();
    let mut offsets = Vec::with_capacity(n + 1);
    offsets.push(0u32);
    for &old in order {
        offsets.push(offsets[offsets.len() - 1] + rings.degree(old) as u32);
    }
    let rows: Vec<Vec<u32>> = order
        .par_iter()
        .map(|&old| rings.ring(old).iter().map(|&c| inverse[c as usize]).collect())
        .collect();
    let mut values = Vec::with_capacity(rings.values.len());
    for row in rows {
        values.extend_from_slice(&row);
    }
    Ok((renamed, Rings { offsets, values }))
}

/// The census `hex_order --locality` takes of a numbering; see [`CellOrder`].
pub fn neighbour_lines_per_warp(mesh: &MpasMesh) -> f64 {
    let warps: Vec<usize> = (0..mesh.n_edges)
        .collect::<Vec<_>>()
        .par_chunks(32)
        .map(|warp| {
            let mut lines = 0;
            for side in 0..2 {
                let mut seen: Vec<i64> = warp
                    .iter()
                    .map(|&e| {
                        let c = mesh.cells_on_edge[2 * e + side];
                        if c < 0 { mesh.n_cells as i64 / 32 } else { c as i64 / 32 }
                    })
                    .collect();
                seen.sort_unstable();
                seen.dedup();
                lines += seen.len();
            }
            lines
        })
        .collect();
    if warps.is_empty() {
        return 0.0;
    }
    warps.iter().sum::<usize>() as f64 / (2 * warps.len()) as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ring_table(rows: &[&[u32]]) -> Rings {
        let mut offsets = vec![0u32];
        let mut values = Vec::new();
        for row in rows {
            values.extend_from_slice(row);
            offsets.push(values.len() as u32);
        }
        Rings { offsets, values }
    }

    #[test]
    fn the_order_is_a_permutation_and_neighbours_land_together() {
        // Two clusters of points, interleaved in the input: Morton order puts
        // each cluster's members next to each other.
        let points: Vec<V3> = (0..64)
            .map(|i| {
                let side = if i % 2 == 0 { -0.9 } else { 0.9 };
                let t = (i / 2) as f64 * 1e-3;
                [side + t, side - t, side + 0.5 * t]
            })
            .collect();
        let order = morton_order(&points).unwrap();
        let mut sorted = order.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..64).collect::<Vec<_>>());
        let first_half: Vec<bool> = order[..32].iter().map(|&i| i % 2 == 0).collect();
        assert!(
            first_half.iter().all(|&b| b == first_half[0]),
            "the two clusters are still interleaved after the Morton sort: {order:?}"
        );
        assert_eq!(order, morton_order(&points).unwrap(), "the order is not a pure function of the points");
    }

    #[test]
    fn renumbering_renames_every_neighbour_and_keeps_every_winding() {
        let points: Vec<V3> = vec![[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]];
        let rings = ring_table(&[&[1, 2, 3], &[0, 3, 2], &[0, 1, 3], &[0, 2, 1]]);
        let order = vec![2, 0, 3, 1];
        let (renamed, renumbered) = renumber(&points, &rings, &order).unwrap();
        for (new, &old) in order.iter().enumerate() {
            assert_eq!(renamed[new], points[old]);
            let expected: Vec<usize> = rings.ring(old).iter().map(|&c| order.iter().position(|&o| o == c as usize).unwrap()).collect();
            let got: Vec<usize> = renumbered.ring(new).iter().map(|&c| c as usize).collect();
            assert_eq!(got, expected, "cell {new} (was {old}) lost its neighbours or their winding");
        }
        assert!(renumber(&points, &rings, &[0, 0, 1, 2]).is_err(), "a repeated index was accepted as a permutation");
        assert!(renumber(&points, &rings, &[0, 1, 2]).is_err(), "a short order was accepted");
    }
}
