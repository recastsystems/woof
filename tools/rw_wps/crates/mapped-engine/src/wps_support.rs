//! Authenticated support for the MPAS case-7 reader. Geometry stays in Rust.
//! All covered rectangle nodes are retained, including coastal search donors.
use super::wps::LatLonLattice;
use crate::{
    digest,
    refusal::{Result, frame_invalid, usage},
};
use serde_json::{Value, json};
use std::{
    collections::{BTreeSet, VecDeque},
    fs,
    path::Path,
};

pub(crate) struct MeshSupport {
    pub required: Vec<bool>,
    points: Vec<(f32, f32)>,
    pub receipt: Value,
}

impl MeshSupport {
    pub fn load(path: &str, sha256: &str, lattice: LatLonLattice) -> Result<Self> {
        let bytes = fs::read(path).map_err(|e| usage(e.to_string()))?;
        if digest::bytes_sha256(&bytes) != sha256 {
            return Err(usage("target support contract digest differs"));
        }
        let doc: Value = serde_json::from_slice(&bytes).map_err(|e| usage(e.to_string()))?;
        if doc["schema"] != "rw-wps.mesh-support.v1" || doc["consumer"] != "mpas-case7-f32" {
            return Err(usage("unsupported target support contract"));
        }
        let target = doc["target_latlon"]
            .as_str()
            .ok_or_else(|| usage("support needs target_latlon"))?;
        if LatLonLattice::parse(target)? != lattice {
            return Err(usage("target support lattice differs"));
        }
        let mesh = doc["mesh"]["path"]
            .as_str()
            .ok_or_else(|| usage("support needs mesh.path"))?;
        let hash = doc["mesh"]["sha256"]
            .as_str()
            .ok_or_else(|| usage("support needs mesh.sha256"))?;
        if digest::file_sha256(mesh).map_err(|e| usage(e.to_string()))? != hash {
            return Err(usage("target support mesh digest differs"));
        }
        let file = netcrust::File::open(Path::new(mesh)).map_err(|e| usage(e.to_string()))?;
        let mut points = Vec::new();
        let mut counts = Vec::new();
        for (lat, lon) in [("latCell", "lonCell"), ("latEdge", "lonEdge")] {
            let read = |name| {
                file.read_array_f64(name)
                    .map(|a| a.into_values())
                    .map_err(|e| usage(e.to_string()))
            };
            let latitudes = read(lat)?;
            let longitudes = read(lon)?;
            if latitudes.is_empty() || latitudes.len() != longitudes.len() {
                return Err(usage(
                    "target support needs matching nonempty cell and edge coordinates",
                ));
            }
            counts.push(latitudes.len());
            for (lat, lon) in latitudes.into_iter().zip(longitudes) {
                if !lat.is_finite() || !lon.is_finite() || lat.abs() > std::f64::consts::FRAC_PI_2 {
                    return Err(usage("target support mesh has invalid coordinates"));
                }
                // Match rw-mpas init's Fortran parameter, one ULP below
                // f32::to_degrees(): the division itself is single precision.
                let degrees_per_radian = 180.0f32 / std::f32::consts::PI;
                let latitude = (lat as f32) * degrees_per_radian;
                let longitude = (lon as f32) * degrees_per_radian;
                points.push(lattice.locate_f32(latitude, longitude));
            }
        }
        let required = Self::stencils(&points, lattice)?;
        let receipt = json!({"schema":"rw-wps.mesh-support.v1", "contract_sha256":sha256,
            "mesh_sha256":hash, "cells":counts[0], "edges":counts[1],
            "required_nodes":required.iter().filter(|x| **x).count(),
            "rectangle_nodes":required.len(), "coastal_search":"checked against uncovered nodes"});
        Ok(Self {
            required,
            points,
            receipt,
        })
    }

    fn stencils(points: &[(f32, f32)], lattice: LatLonLattice) -> Result<Vec<bool>> {
        let mut required = vec![false; lattice.nx * lattice.ny];
        for &(x, y) in points {
            // One-based f32 positions and the reader's 1e-5 truncation. Always
            // retain the whole 4x4 stencil, even at an exact node. Four-point,
            // nearest and weighted-four methods read a subset of these nodes.
            let i = (x + 0.00001) as isize;
            let j = (y + 0.00001) as isize;
            if !x.is_finite()
                || !y.is_finite()
                || i < 2
                || j < 2
                || i + 2 > lattice.nx as isize
                || j + 2 > lattice.ny as isize
            {
                return Err(frame_invalid(
                    "mesh interpolation support leaves the target lattice",
                ));
            }
            for row in j - 1..=j + 2 {
                for col in i - 1..=i + 2 {
                    required[(row as usize - 1) * lattice.nx + col as usize - 1] = true;
                }
            }
        }
        Ok(required)
    }

    /// When a local stencil cannot supply a field, conservatively follow the
    /// reader's search queue (including the queued nearest-distance tie pass).
    /// Refuse if it would read an uncovered node. Covered nodes keep their
    /// original values, so a passing rectangle never changes its search result.
    pub fn check_search(
        &self,
        slabs: &[Vec<super::wps::Slab>],
        lattice: LatLonLattice,
    ) -> Result<()> {
        use rayon::prelude::*;
        let mask = slabs
            .iter()
            .flatten()
            .find(|s| s.field == "LANDSEA")
            .ok_or_else(|| frame_invalid("mesh support requires LANDSEA"))?;
        if !slabs
            .iter()
            .flatten()
            .any(|s| s.uncovered.iter().any(|v| *v))
        {
            return Ok(());
        }
        slabs.par_iter().flatten().try_for_each(|slab| -> Result<()> {
            let uncovered = |node: usize| slab.uncovered.get(node).copied().unwrap_or(false)
                || mask.uncovered.get(node).copied().unwrap_or(false);
            let field = slab.field.as_str();
            if matches!(field, "LANDSEA" | "SNOW" | "SST") { return Ok(()); }
            let maskval = if field.starts_with("ST") || field.starts_with("SM") {Some(0.)}
                else if field == "SEAICE" {Some(1.)} else {None};
            let valid = |index: usize| slab.values[index] != super::wps::MISSING
                && maskval.is_none_or(|v| mask.values[index] != v);
            // The reader wraps three columns on either side. Search uses those
            // columns too; retain their identity until mapping to the WPS slab.
            let index = |i: isize, j: isize| (j as usize - 1) * lattice.nx
                + (i-1).rem_euclid(lattice.nx as isize) as usize;
            for &(x, y) in &self.points {
                let i = (x + 0.00001) as isize;
                let j = (y + 0.00001) as isize;
                let local_ok = (j-1..=j+2).all(|jj| (i-1..=i+2).all(|ii| valid(index(ii,jj))));
                if local_ok { continue; }
                // Four-point and weighted-four fallbacks need no search if all
                // four corners are valid (soil and sea ice reader method lists).
                if maskval.is_some() && [y.floor() as isize,y.ceil() as isize].into_iter()
                    .all(|jj| [x.floor() as isize,x.ceil() as isize].into_iter().all(|ii| valid(index(ii,jj)))) {
                    continue;
                }
                let start = (x.round() as isize, y.round() as isize);
                let mut queue = VecDeque::from([start]);
                let mut seen = BTreeSet::from([start]);
                let mut found = false;
                while let Some((ii,jj)) = queue.pop_front() {
                    let node = index(ii,jj);
                    if uncovered(node) {
                        return Err(frame_invalid(format!("mesh coastal interpolation search leaves source coverage for {field}: choose a smaller region inside the source domain")));
                    }
                    if found { continue; }
                    let hit = valid(node);
                    for (a,b) in [(ii-1,jj),(ii+1,jj),(ii,jj-1),(ii,jj+1)] {
                        if a >= -2 && a <= lattice.nx as isize+3 && b >= 1 && b <= lattice.ny as isize
                            && seen.insert((a,b)) { queue.push_back((a,b)); }
                    }
                    if hit { found = true; }
                }
            }
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::wps::{MISSING, Slab};
    use super::*;

    #[test]
    fn cell_and_edge_stencils_include_the_f32_rounding_guard() {
        assert_eq!((180.0f32 / std::f32::consts::PI).to_bits(), 0x42652ee0);
        let grid = LatLonLattice::parse("-10,170,0.025,0.025,100,100").unwrap();
        let mask = MeshSupport::stencils(&[(10.2, 10.3), (80.999999, 90.5)], grid).unwrap();
        assert_eq!(mask.iter().filter(|x| **x).count(), 32);
        assert!(mask[8 * 100 + 8]);
        assert!(mask[91 * 100 + 82]);
        assert!(MeshSupport::stencils(&[(0.9, 10.)], grid).is_err());
        let (x, y) = grid.locate_f32(-9., -189.);
        assert!((x - 41.).abs() < 0.001 && (y - 41.).abs() < 0.001);
    }

    fn slab(field: &str, values: Vec<f32>) -> Slab {
        Slab {
            field: field.into(),
            units: String::new(),
            level: 0.,
            uncovered: values
                .iter()
                .map(|v| field == "LANDSEA" && *v == MISSING)
                .collect(),
            values,
        }
    }

    #[test]
    fn coastal_search_checks_queued_nodes_as_well_as_local_stencils() {
        let grid = LatLonLattice::parse("0,0,1,1,20,20").unwrap();
        let mut land = vec![1.; 400];
        land[10 * 20 + 15] = MISSING;
        let support = MeshSupport {
            required: vec![false; 400],
            points: vec![(11., 11.)],
            receipt: json!({}),
        };
        let mut values = vec![MISSING; 400];
        values[10 * 20 + 14] = 2.;
        // The first hit queues a neighbour outside coverage. The reader's
        // tie pass would read it even though a valid donor was already found.
        assert!(
            support
                .check_search(
                    &[vec![slab("LANDSEA", land.clone()), slab("TT", values)]],
                    grid
                )
                .is_err()
        );
        let mut narrower = slab("TT", vec![MISSING; 400]);
        narrower.values[10 * 20 + 14] = 2.;
        narrower.uncovered.fill(false);
        narrower.uncovered[10 * 20 + 15] = true;
        assert!(
            support
                .check_search(&[vec![slab("LANDSEA", vec![1.; 400]), narrower]], grid)
                .is_err()
        );
        assert!(
            support
                .check_search(
                    &[vec![slab("LANDSEA", land), slab("TT", vec![2.; 400])]],
                    grid
                )
                .is_ok()
        );
    }

    #[test]
    fn a_changed_contract_is_refused_before_mesh_io() {
        let path = std::env::temp_dir().join(format!("wps-support-{}.json", std::process::id()));
        fs::write(&path, b"{}").unwrap();
        let err = MeshSupport::load(
            path.to_str().unwrap(),
            &"0".repeat(64),
            LatLonLattice::parse("0,0,1,1,10,10").unwrap(),
        )
        .err()
        .unwrap();
        fs::remove_file(path).unwrap();
        assert!(err.message.contains("contract digest differs"));
    }
}
