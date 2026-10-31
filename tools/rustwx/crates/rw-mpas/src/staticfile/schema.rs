//! The static schema the mesh registry pins, and the gate that holds the
//! writer to it.
//!
//! THE BREAKAGE THIS PREVENTS. This crate carried two static writers, built
//! against different consumers, neither knowing about the other. One wrote 69
//! variables and owned the `rw_mpas_static` name; the other wrote 56 under
//! host-memory admission. With both compiled in, one edit to a `[[bin]]` path
//! or one `use` line was enough to point `rw_mpas_static` -- and therefore
//! `woof mesh` -- at the wrong one, and nothing would have said so: the build
//! stays green, the binary keeps its name, the run exits 0, and a user gets a
//! static with no `var2d`, no `oa1..oa4`, no `ol1..ol4`, no `cellsOnVertex` and
//! no `kiteAreasOnVertex`. The mesh registry pins grid and static together by
//! byte count and SHA-256, so that file is refused at registration, and a
//! forecast that reached one anyway would run a dycore with no sub-grid
//! orography and no vertex duals.
//!
//! ONE WRITER NOW SERVES THE DOOR. [`crate::static_builder`] is it: the
//! streaming, admission-gated, rayon-parallel builder, carrying the tested
//! computations the 69-variable writer owned. The manifest is pinned here in
//! one place, the writer refuses at write time if what it is about to declare
//! has drifted from it, and [`gate_bounded_writer_matches_pin`] holds its
//! declared manifest to the same list.
//!
//! THE PIN IS THE UNION, 82 NAMES. The first 69 are the published static's own
//! manifest, in the published order. The 13 that follow are what the streaming
//! builder brings and the published file never carried: the four
//! deformation/gradient operator tables, the Noah-MP soil-composition group,
//! and the land-use aliases and category scalars a surface scheme reads by
//! name. Dropping any of them to "match the published file" would put the
//! consumer back where it was -- reading a static and finding the array it
//! needs is not there.

/// Every variable the pinned static carries, in the order it is declared.
///
/// This is the registered schema: the port registry pins a static of exactly
/// these 82 variables, and a file that carries a different set is a different
/// file by byte count before it is a different file by content.
pub const PINNED_STATIC_VARIABLES: [&str; 82] = [
    "xtime",
    "latCell",
    "lonCell",
    "xCell",
    "yCell",
    "zCell",
    "indexToCellID",
    "latEdge",
    "lonEdge",
    "xEdge",
    "yEdge",
    "zEdge",
    "indexToEdgeID",
    "latVertex",
    "lonVertex",
    "xVertex",
    "yVertex",
    "zVertex",
    "indexToVertexID",
    "cellsOnEdge",
    "nEdgesOnCell",
    "nEdgesOnEdge",
    "edgesOnCell",
    "edgesOnEdge",
    "weightsOnEdge",
    "dvEdge",
    "dcEdge",
    "angleEdge",
    "areaCell",
    "areaTriangle",
    "cellsOnCell",
    "verticesOnCell",
    "verticesOnEdge",
    "edgesOnVertex",
    "cellsOnVertex",
    "kiteAreasOnVertex",
    "meshDensity",
    "nominalMinDc",
    "bdyMaskCell",
    "bdyMaskEdge",
    "bdyMaskVertex",
    "edgeNormalVectors",
    "localVerticalUnitVectors",
    "cellTangentPlane",
    "coeffs_reconstruct",
    "deriv_two",
    "fEdge",
    "fVertex",
    "ter",
    "landmask",
    "mminlu",
    "ivgtyp",
    "isltyp",
    "snoalb",
    "soiltemp",
    "greenfrac",
    "shdmin",
    "shdmax",
    "albedo12m",
    "var2d",
    "con",
    "oa1",
    "oa2",
    "oa3",
    "oa4",
    "ol1",
    "ol2",
    "ol3",
    "ol4",
    // -- the 13 beyond the published manifest, see the module note ---------
    "cell_gradient_coef_x",
    "cell_gradient_coef_y",
    "defc_a",
    "defc_b",
    "lu_index",
    "soilcat_top",
    "soilcomp",
    "soilcl1",
    "soilcl2",
    "soilcl3",
    "soilcl4",
    "isice_lu",
    "iswater_lu",
];

/// The variables the pin requires that `names` does not declare.
pub fn missing_from_pin(names: &[String]) -> Vec<&'static str> {
    PINNED_STATIC_VARIABLES
        .iter()
        .copied()
        .filter(|want| !names.iter().any(|have| have == want))
        .collect()
}

/// The variables `names` declares that the pin does not carry.
pub fn beyond_the_pin(names: &[String]) -> Vec<String> {
    names
        .iter()
        .filter(|have| !PINNED_STATIC_VARIABLES.iter().any(|want| *want == have.as_str()))
        .cloned()
        .collect()
}

/// A one-line account of how `names` differs from the pin, or `None` when it
/// matches exactly. Used by both the write-time refusal and the gate test, so
/// the two cannot describe the same gap differently.
pub fn divergence_from_pin(names: &[String]) -> Option<String> {
    let missing = missing_from_pin(names);
    let extra = beyond_the_pin(names);
    if missing.is_empty() && extra.is_empty() {
        return None;
    }
    let mut parts = Vec::new();
    if !missing.is_empty() {
        parts.push(format!(
            "{} variable(s) the pin requires are absent: {}",
            missing.len(),
            missing.join(", ")
        ));
    }
    if !extra.is_empty() {
        parts.push(format!(
            "{} variable(s) are declared that the pin does not carry: {}",
            extra.len(),
            extra.join(", ")
        ));
    }
    Some(parts.join("; "))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_pin_lists_eighty_two_distinct_variables() {
        let mut sorted = PINNED_STATIC_VARIABLES.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(
            sorted.len(),
            82,
            "the pinned manifest must be 82 DISTINCT names; a duplicate would \
             make the registry's byte count unreachable"
        );
    }

    /// The published static's own 69 names are all still in the pin.
    ///
    /// The union was reached by ADDING to the published manifest, never by
    /// trading one of its fields for a new one. Spelled out because the
    /// tempting simplification -- "drop `defc_a`/`defc_b`, the published file
    /// does not carry them" -- takes two arrays the Smagorinsky strain
    /// operator reads out of the file, and the run stops before it allocates.
    #[test]
    fn every_published_field_survives_the_union() {
        for name in [
            "xtime", "latCell", "indexToCellID", "nEdgesOnEdge", "edgesOnEdge",
            "weightsOnEdge", "edgesOnVertex", "cellsOnVertex", "kiteAreasOnVertex",
            "bdyMaskEdge", "bdyMaskVertex", "edgeNormalVectors",
            "localVerticalUnitVectors", "cellTangentPlane", "coeffs_reconstruct",
            "deriv_two", "var2d", "con", "oa1", "oa4", "ol1", "ol4", "mminlu",
        ] {
            assert!(
                PINNED_STATIC_VARIABLES.contains(&name),
                "{name} left the pin; the published static carries it"
            );
        }
    }

    #[test]
    fn an_exact_manifest_shows_no_divergence() {
        let names: Vec<String> = PINNED_STATIC_VARIABLES
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(divergence_from_pin(&names), None);
    }

    #[test]
    fn a_dropped_variable_is_named_not_merely_counted() {
        let names: Vec<String> = PINNED_STATIC_VARIABLES
            .iter()
            .filter(|n| **n != "var2d")
            .map(|s| s.to_string())
            .collect();
        let said = divergence_from_pin(&names).expect("a dropped variable must diverge");
        assert!(said.contains("var2d"), "the gap must name the field: {said}");
    }

    /// THE GATE. The writer that serves the door declares the pinned schema.
    ///
    /// [`crate::static_builder`] exists for its host-memory admission and its
    /// byte-identical parallel tile decode, and it is now how the pinned door
    /// builds geography. That is only true while it declares the same file the
    /// registry admits. This test is what holds it there: it was RED by
    /// construction while the two schemas differed, naming the 26 variables
    /// the streaming writer had to gain, and it goes red again the moment a
    /// field is added to one place and not the other.
    #[test]
    fn gate_bounded_writer_matches_pin() {
        let names: Vec<String> = crate::static_builder::declared_variables()
            .iter()
            .map(|s| s.to_string())
            .collect();
        if let Some(said) = divergence_from_pin(&names) {
            panic!(
                "the writer `rw_mpas_static` runs no longer declares the pinned \
                 schema: {said}\n\n\
                 The mesh registry pins grid and static together by byte count \
                 and SHA-256, so a build made this way is refused at \
                 registration before anything reads it."
            );
        }
    }
}
