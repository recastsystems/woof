//! What is left of the FIRST static writer: the schema it pinned, the grader
//! it shipped, the archive ladder it walked, and the FP32 reading it took.
//!
//! # Why this module is a remnant
//!
//! This crate carried two static writers. `staticfile::emit` wrote the 69
//! variables the mesh registry pins and owned the `rw_mpas_static` name;
//! [`crate::static_builder`] wrote 56 under host-memory admission with a
//! byte-identical parallel tile decode, and was reached only through a second
//! binary. They were built against different consumers and neither knew about
//! the other, so one edit to a `use` line was enough to hand a user the wrong
//! file with the right name -- a static with no `var2d`, no `oa1..oa4`, no
//! `cellsOnVertex` and no `kiteAreasOnVertex`, refused at registration by byte
//! count. [`schema`] is the gate that said so, and it was RED on purpose until
//! the two converged.
//!
//! They have converged. The STREAMING builder is the survivor -- it is the one
//! whose memory is bounded, whose tile decode is parallel and deterministic,
//! and whose refusals name what they prevent -- and the computations this
//! writer owned moved into it rather than being rewritten:
//!
//! * the sub-grid orography statistics are [`crate::static_gwd`], reading the
//!   archive's own georeferencing instead of assuming the Fortran's;
//! * `deriv_two` and the deformation weights are `crate::static_operators`;
//! * the mesh dual and the local frames are read and written by
//!   `crate::static_builder`;
//! * the two measured unit conversions are constants there, still table data.
//!
//! What stayed here is what was never about writing a file: the pinned
//! manifest, the field-by-field grader `--compare` runs, the ladder that
//! agrees with where `woof fetch-geog` stages, and the FP32 metric reading.

pub mod compare;
pub mod coordframe;
pub mod fp32metrics;
pub mod geog;
pub mod schema;

#[cfg(test)]
use crate::mesh::geom::EARTH_RADIUS_M;

/// Fields written as a bitwise +0 PLACEHOLDER, not as values.
///
/// A consumer overlays the real array; this engine writes only the slot and the
/// zeros, so the "the static carries bitwise +0" contract the MPAS port's pins
/// assert is preserved rather than measured.
///
/// The list is not a choice, it is a reading of the published file. Measured on
/// `x4.163842.static.nc`: every element of all four is exactly zero. `fEdge`
/// and `fVertex` are deliberately NOT here -- the published static carries real
/// Coriolis values for those, and so does this engine.
///
/// The SLOT is not optional either, however cheaply a model could recompute the
/// values. `coeffs_reconstruct` was left out entirely until a real forecast was
/// attempted, and the MPAS port stopped at
/// `AttributeError: mesh has no 'coeffs_reconstruct'` before a single device
/// byte was allocated.
pub const ZERO_PLACEHOLDER_FIELDS: [&str; 4] = [
    "coeffs_reconstruct",
    "edgeNormalVectors",
    "localVerticalUnitVectors",
    "cellTangentPlane",
];

#[cfg(test)]
mod tests {
    use super::*;

    /// The three local-frame arrays are placeholders, and `fEdge` / `fVertex`
    /// are not.
    ///
    /// This distinction was measured against the published
    /// `x4.163842.static.nc`, not assumed: the frame arrays are exactly zero
    /// there and the Coriolis arrays are not. Writing derived frames into the
    /// frame slots -- which this engine did -- makes the MPAS port refuse the
    /// file with `static edgeNormalVectors placeholder bytes changed`.
    #[test]
    fn the_local_frames_are_placeholders_and_coriolis_is_not() {
        for name in [
            "edgeNormalVectors",
            "localVerticalUnitVectors",
            "cellTangentPlane",
        ] {
            assert!(
                ZERO_PLACEHOLDER_FIELDS.contains(&name),
                "{name} must be written as bitwise +0; a derived value there \
                 is refused by the port before any device allocation"
            );
        }
        for name in ["fEdge", "fVertex", "ter", "landmask", "nominalMinDc"] {
            assert!(
                !ZERO_PLACEHOLDER_FIELDS.contains(&name),
                "{name} carries real values in the published static"
            );
        }
    }

    /// `coeffs_reconstruct` is written, as a placeholder, and is not absent.
    #[test]
    fn the_reconstruction_placeholder_is_written_not_declared_absent() {
        assert!(ZERO_PLACEHOLDER_FIELDS.contains(&"coeffs_reconstruct"));
        assert!(
            schema::PINNED_STATIC_VARIABLES.contains(&"coeffs_reconstruct"),
            "the slot left the pin; the port will refuse the static before it \
             allocates any device memory"
        );
    }

    /// Every field the first writer declared ABSENT is now written.
    ///
    /// The absent list was a consequence of the schema split, not a property
    /// of the geography: `cell_gradient_coef_x/y` and `defc_a/b` were computed
    /// by one writer and declared missing by the other, and the
    /// soil-composition group was read by one and unknown to the other. A
    /// reader who learnt the old list would go looking for a gap that is not
    /// there, so the pin is asserted here rather than in a comment.
    #[test]
    fn nothing_the_old_absent_list_named_is_absent_now() {
        for name in [
            "cell_gradient_coef_x",
            "cell_gradient_coef_y",
            "defc_a",
            "defc_b",
            "deriv_two",
            "soilcomp",
            "soilcl1",
            "soilcl2",
            "soilcl3",
            "soilcl4",
        ] {
            assert!(
                schema::PINNED_STATIC_VARIABLES.contains(&name),
                "{name} was declared absent by the retired writer and must be \
                 written by the surviving one"
            );
        }
    }

    /// The round trip `from_grid` depends on: a grid stores the nominal
    /// spacing as a unit-sphere angle, and reading it back must land on the
    /// SAME `f32` bits the original metres would have produced. A single bit
    /// of drift here is a refusal at the consumer, because that scalar is what
    /// the gravity-wave drag length scale is rebound to.
    #[test]
    fn the_round_trip_is_fp32_exact() {
        for metres in [
            1_000.0f64, 3_000.0, 15_000.0, 25_000.0, 60_000.0, 120_000.0, 240_000.0, 1_234.5,
            7_500.0, 92_345.678,
        ] {
            let angle = metres / EARTH_RADIUS_M;
            let back = angle * EARTH_RADIUS_M;
            assert_eq!(
                (back as f32).to_bits(),
                (metres as f32).to_bits(),
                "{metres} m round-tripped to {back} m, which is a different f32"
            );
        }
    }
}
