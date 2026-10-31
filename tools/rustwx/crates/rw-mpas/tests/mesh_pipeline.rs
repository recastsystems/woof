//! The whole generator, end to end, and the file it writes read back through a
//! reader that shares no code with the writer.
//!
//! A file that the writer can read back proves nothing about the writer. These
//! tests emit through `rw_store::netcdf_classic` and read through `netcrust` --
//! a separate codebase with its own header parser -- then rebuild every derived
//! field from the cell centres the file carries and check them against the
//! values the file carries beside them.

use std::path::PathBuf;

use rw_mpas::mesh::density::{MeshSpec, Region, Shape, TransitionField};
use rw_mpas::mesh::footprint;
use rw_mpas::mesh::emit::{Provenance, write_grid};
use rw_mpas::mesh::geom::{EARTH_RADIUS_M, V3, chord};
use rw_mpas::mesh::hull::delaunay_rings;
use rw_mpas::mesh::{GenerateRequest, Limits, LloydOptions, MpasMesh, Rings, generate};

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join("rw_mpas_mesh_tests");
    std::fs::create_dir_all(&dir).expect("scratch directory");
    dir.join(name)
}

fn small_request(cells: usize) -> GenerateRequest {
    GenerateRequest {
        spec: MeshSpec::uniform(1000.0),
        target_cells: Some(cells),
        lloyd: LloydOptions {
            tolerance: 3e-3,
            max_sweeps: 200,
            ..Default::default()
        },
        sizing_samples: 20_000,
        ..Default::default()
    }
}

#[test]
fn the_generator_produces_a_mesh_that_passes_its_own_emit_gate() {
    let out = generate(&small_request(900), |line| eprintln!("  {line}"))
        .unwrap_or_else(|e| panic!("generate refused: {e}"));
    let r = &out.receipt;
    eprintln!(
        "uniform 900: delivered {} cells, {} edges, {} vertices; spacing {:.1}..{:.1} km ratio {:.4}; nonorthogonality {:.3e}; antisymmetry {:.3e} over {} pairs; {} sweeps in {:.2} s",
        r.delivered_cells,
        r.mesh.n_edges,
        r.mesh.n_vertices,
        r.mesh.min_spacing_m / 1000.0,
        r.mesh.max_spacing_m / 1000.0,
        r.mesh.spacing_ratio,
        r.mesh.max_nonorthogonality,
        r.mesh.max_weight_antisymmetry,
        r.mesh.weight_pairs_checked,
        r.relaxation_sweeps,
        r.relaxation_seconds
    );
    // A uniform request seeds from the icosahedral subdivision, so the count
    // SNAPS to the nearest achievable 10*(m^2+mn+n^2)+2 -- 900 lands on
    // GP(9,1) = 912 -- and the receipt records the move instead of the
    // delivered count quietly disagreeing with the request.
    assert_eq!(r.delivered_cells, 912);
    match r.seeding {
        rw_mpas::mesh::Seeding::IcosahedralGoldberg {
            m,
            n,
            requested_cells,
            seeded_cells,
            ..
        } => {
            assert_eq!((m, n), (9, 1));
            assert_eq!(requested_cells, 900);
            assert_eq!(seeded_cells, 912);
        }
        rw_mpas::mesh::Seeding::FibonacciAcceptance { .. } => {
            panic!("a uniform request took the Fibonacci seed, whose dislocations are the defect class the icosahedral seed exists to remove")
        }
        rw_mpas::mesh::Seeding::HierarchicalGoldberg { .. } => {
            panic!("a uniform request took the graded ladder; the ladder's level 0 IS the uniform arm, so routing a uniform request through it does nothing but hide the arm the receipt should name")
        }
    }
    // Dislocation-free: exactly the twelve pentagons topology requires.
    assert_eq!(
        r.mesh.coordination_histogram,
        vec![(5, 12), (6, r.delivered_cells - 12)]
    );
    assert_eq!(r.mesh.euler_characteristic, 2);
    assert_eq!(r.mesh.coordination_defect, 12);
    assert_eq!(r.mesh.nonzero_weight_padding_slots, 0);
    assert!(r.mesh.min_edge_orientation > 0.0);
}

#[test]
fn the_written_grid_file_reads_back_through_an_independent_reader() {
    let out = generate(&small_request(600), |_| {}).expect("generate");
    let path = scratch("independent-reader.grid.nc");
    let _ = std::fs::remove_file(&path);
    let written = write_grid(
        &out.mesh,
        &path,
        &Provenance {
            spec_json: serde_json::to_string(&out.spec).unwrap(),
            request: "600 cells, uniform".into(),
            receipt_json: "{}".into(),
            static_coordinates: None,
        },
        false,
    )
    .expect("write");
    eprintln!(
        "wrote {} B, file_id {}, sha256 {}",
        written.bytes, written.file_id, written.sha256
    );

    let f = netcrust::File::open(&path).expect("netcrust opens the file this crate wrote");

    // --- the published schema: 8 dimensions, 38 variables, 7 named attributes
    for (name, want) in [
        ("nCells", out.mesh.n_cells),
        ("nEdges", out.mesh.n_edges),
        ("nVertices", out.mesh.n_vertices),
        ("maxEdges", out.mesh.max_edges),
        ("maxEdges2", out.mesh.max_edges2),
        ("TWO", 2),
        ("vertexDegree", 3),
        ("codeLen", 1),
    ] {
        let d = f
            .dimension(name)
            .unwrap_or_else(|| panic!("the file has no dimension {name}"));
        assert_eq!(d.len(), want, "dimension {name}");
    }
    let vars = f.variables().expect("variables");
    let names: Vec<&str> = vars.iter().map(|v| v.name()).collect();
    for required in [
        "latCell", "lonCell", "xCell", "yCell", "zCell", "indexToCellID",
        "latEdge", "lonEdge", "xEdge", "yEdge", "zEdge", "indexToEdgeID",
        "latVertex", "lonVertex", "xVertex", "yVertex", "zVertex", "indexToVertexID",
        "nEdgesOnCell", "nEdgesOnEdge", "cellsOnCell", "edgesOnCell", "verticesOnCell",
        "edgesOnEdge", "cellsOnEdge", "verticesOnEdge", "cellsOnVertex", "edgesOnVertex",
        "areaCell", "areaTriangle", "dcEdge", "dvEdge", "angleEdge", "weightsOnEdge",
        "kiteAreasOnVertex", "meshDensity", "nominalMinDc", "densityFunctionCode",
    ] {
        assert!(names.contains(&required), "the file has no variable {required}");
    }
    assert_eq!(names.len(), 38, "variable count: {names:?}");

    assert_eq!(f.attribute("mesh_spec").and_then(|a| a.as_string().map(str::to_string)).as_deref(), Some("1.1"));
    assert_eq!(f.attribute("on_a_sphere").and_then(|a| a.as_string().map(str::to_string)).as_deref(), Some("YES"));
    assert_eq!(f.attribute("is_periodic").and_then(|a| a.as_string().map(str::to_string)).as_deref(), Some("NO"));
    assert_eq!(f.attribute("sphere_radius").and_then(|a| a.as_f64()), Some(1.0));
    assert_eq!(f.attribute("x_period").and_then(|a| a.as_f64()), Some(0.0));
    assert_eq!(f.attribute("y_period").and_then(|a| a.as_f64()), Some(0.0));
    assert_eq!(
        f.attribute("file_id").and_then(|a| a.as_string().map(str::to_string)).as_deref(),
        Some(written.file_id.as_str())
    );
    // The deliverable boundary travels with the file, so a reader that mistook
    // a grid file for a runnable mesh has been told otherwise in the header.
    let boundary = f
        .attribute("rw_mesh_boundary")
        .and_then(|a| a.as_string().map(str::to_string))
        .expect("rw_mesh_boundary");
    assert!(boundary.contains("static file"), "{boundary}");

    // --- values -------------------------------------------------------------
    let read = |name: &str| -> Vec<f64> { f.read_f64(name).unwrap_or_else(|e| panic!("{name}: {e}")) };
    let n = out.mesh.n_cells;

    for (name, k) in [("xCell", 0usize), ("yCell", 1), ("zCell", 2)] {
        let got = read(name);
        assert_eq!(got.len(), n);
        for i in 0..n {
            assert_eq!(got[i], out.mesh.cell_xyz[i][k], "{name}[{i}]");
        }
    }
    // 1-BASED on the way out, which is how MPAS reads them.
    let ids = read("indexToCellID");
    assert_eq!(ids.first().copied(), Some(1.0));
    assert_eq!(ids.last().copied(), Some(n as f64));
    let coe = read("cellsOnEdge");
    assert_eq!(coe.len(), out.mesh.n_edges * 2);
    assert!(coe.iter().all(|&v| v >= 1.0), "cellsOnEdge is not 1-based");

    // --- the padding fill rules the published files carry -------------------
    let coc = read("cellsOnCell");
    let eoc = read("edgesOnCell");
    let voc = read("verticesOnCell");
    let me = out.mesh.max_edges;
    let mut padded_cells = 0usize;
    for i in 0..n {
        let deg = out.mesh.n_edges_on_cell[i] as usize;
        if deg == me {
            continue;
        }
        padded_cells += 1;
        let last = coc[i * me + deg - 1];
        let max_edge = (0..deg).map(|j| eoc[i * me + j]).fold(f64::MIN, f64::max);
        for j in deg..me {
            assert_eq!(coc[i * me + j], last, "cellsOnCell padding at cell {i}");
            assert_eq!(eoc[i * me + j], max_edge, "edgesOnCell padding at cell {i}");
            assert_eq!(voc[i * me + j], 0.0, "verticesOnCell padding at cell {i}");
        }
    }
    assert!(padded_cells > 0, "no cell was padded, so the padding rules were not exercised");

    let w = read("weightsOnEdge");
    let neoe = read("nEdgesOnEdge");
    let me2 = out.mesh.max_edges2;
    for e in 0..out.mesh.n_edges {
        for k in neoe[e] as usize..me2 {
            assert_eq!(w[e * me2 + k], 0.0, "weightsOnEdge padding at edge {e} slot {k}");
        }
    }

    // --- the scalar and the char, which are the two shapes a classic writer
    //     most often gets wrong
    let nominal = read("nominalMinDc");
    assert_eq!(nominal.len(), 1, "nominalMinDc is not a scalar");
    assert!(
        (nominal[0] * EARTH_RADIUS_M / 1000.0 - 1000.0).abs() < 1e-6,
        "nominalMinDc reads {} km",
        nominal[0] * EARTH_RADIUS_M / 1000.0
    );
    let code = f.variable("densityFunctionCode").expect("densityFunctionCode");
    assert_eq!(code.shape(), vec![1], "densityFunctionCode shape");

    let _ = std::fs::remove_file(&path);
}

#[test]
fn a_written_mesh_rebuilds_itself_from_the_cell_centres_it_carries() {
    // The round trip that matters: take the file's OWN cell centres, rebuild
    // the triangulation and every derived field from scratch, and compare
    // against what the file says. This is the same grading the published
    // meshes get in tests/mesh_derive.rs, turned on our own output.
    let out = generate(&small_request(700), |_| {}).expect("generate");
    let path = scratch("round-trip.grid.nc");
    let _ = std::fs::remove_file(&path);
    write_grid(&out.mesh, &path, &Provenance::default(), true).expect("write");

    let f = netcrust::File::open(&path).expect("open");
    let x = f.read_f64("xCell").unwrap();
    let y = f.read_f64("yCell").unwrap();
    let z = f.read_f64("zCell").unwrap();
    let pts: Vec<V3> = (0..x.len()).map(|i| [x[i], y[i], z[i]]).collect();
    let density = f.read_f64("meshDensity").unwrap();
    let nominal = f.read_f64("nominalMinDc").unwrap()[0];

    let rings = delaunay_rings(&pts).expect("the file's own centres retriangulate");
    let rebuilt = MpasMesh::derive(pts.clone(), density, &rings, nominal).expect("derive");

    assert_eq!(rebuilt.n_cells, out.mesh.n_cells);
    assert_eq!(rebuilt.n_edges, out.mesh.n_edges);
    assert_eq!(rebuilt.n_vertices, out.mesh.n_vertices);

    let file_area = f.read_f64("areaCell").unwrap();
    let file_dc = f.read_f64("dcEdge").unwrap();
    let file_w = f.read_f64("weightsOnEdge").unwrap();
    let mut worst_area = 0.0f64;
    for i in 0..rebuilt.n_cells {
        worst_area = worst_area.max((rebuilt.area_cell[i] / file_area[i] - 1.0).abs());
    }
    // Edge and vertex numbering is canonical -- a pure function of the cell
    // centres -- so a rebuild lands on the same numbering and these compare
    // index for index.
    let mut worst_dc = 0.0f64;
    let mut worst_w = 0.0f64;
    for e in 0..rebuilt.n_edges {
        worst_dc = worst_dc.max((rebuilt.dc_edge[e] / file_dc[e] - 1.0).abs());
        for k in 0..rebuilt.max_edges2 {
            worst_w =
                worst_w.max((rebuilt.weights_on_edge[e * rebuilt.max_edges2 + k] - file_w[e * rebuilt.max_edges2 + k]).abs());
        }
    }
    eprintln!(
        "round trip: areaCell rel {worst_area:.3e}, dcEdge rel {worst_dc:.3e}, weightsOnEdge abs {worst_w:.3e}"
    );
    assert!(worst_area < 1e-15, "areaCell {worst_area:.3e}");
    assert!(worst_dc < 1e-15, "dcEdge {worst_dc:.3e}");
    assert!(worst_w < 1e-15, "weightsOnEdge {worst_w:.3e}");

    let _ = std::fs::remove_file(&path);
}

#[test]
fn a_variable_resolution_request_delivers_the_refinement_it_asked_for() {
    // The user experience the whole binary exists for: fine over a named box,
    // coarse everywhere else. Graded on COUNTS and on the delivered spacing
    // inside and outside the box, not on a fit.
    // This request MOVED when graded generation moved to the hierarchical
    // ladder. The old 125-in-500 box with a 300 km ramp asked for a 59% per
    // cell gradient; the Fibonacci arm "served" it by rolling metre-class
    // dual edges (this very test used to name the FP32 floors away to look
    // past them), and the ladder refuses it up front because a band that
    // narrow cannot contain its own repairs. The request below is the same
    // user experience -- 4x refinement over a named box -- at a ramp the
    // annuli can carry, and it passes the DEFAULT gate.
    let spec = MeshSpec {
        background_km: 600.0,
        regions: vec![Region {
            shape: Shape::LatLonBox {
                lat_deg: [25.0, 50.0],
                lon_deg: [-115.0, -80.0],
            },
            spacing_km: 150.0,
            // Centred ON the box boundary; the box is about two ramp widths
            // across, so its middle attains ~180 km against the 150 asked.
            transition: TransitionField::Km(2200.0),
        }],
        name: Some("box refinement".into()),
    };
    let request = GenerateRequest {
        spec: spec.clone(),
        target_cells: None,
        lloyd: LloydOptions {
            tolerance: 3e-3,
            max_sweeps: 250,
            ..Default::default()
        },
        sizing_samples: 50_000,
        // The DEFAULT limits, deliberately. This test used to name the FP32
        // floors away because the Fibonacci arm's dislocations rolled a
        // 1,761.4 m dual edge on this very layout; the hierarchical ladder
        // that now serves graded requests is the mechanism that removed that
        // class, so the same request has to clear the same gate every other
        // mesh clears -- passing only under loosened floors would be the old
        // defect wearing a new seed.
        limits: Limits::default(),
        ..Default::default()
    };
    let out = generate(&request, |line| eprintln!("  {line}")).unwrap_or_else(|e| panic!("{e}"));
    let mesh = &out.mesh;
    let spacing = mesh.spacing_m();

    let inside = Shape::LatLonBox {
        lat_deg: [28.0, 47.0],
        lon_deg: [-111.0, -84.0],
    };
    let mut fine: Vec<f64> = Vec::new();
    let mut coarse: Vec<f64> = Vec::new();
    for i in 0..mesh.n_cells {
        let p = mesh.cell_xyz[i];
        if inside.signed_distance(p) < 0.0 {
            fine.push(spacing[i]);
        } else if spec.regions[0].shape.signed_distance(p) > 6000_000.0 / EARTH_RADIUS_M {
            coarse.push(spacing[i]);
        }
    }
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    eprintln!(
        "box refinement: {} cells total, {} inside the box at a mean of {:.1} km, {} far outside at {:.1} km; requested 150 and 600; steepest requested gradient {:.2}% per cell against the published 1.53%",
        mesh.n_cells,
        fine.len(),
        mean(&fine) / 1000.0,
        coarse.len(),
        mean(&coarse) / 1000.0,
        out.receipt.steepest_requested_gradient_percent_per_cell
    );
    assert!(fine.len() > 30, "only {} cells landed inside the box", fine.len());
    assert!(coarse.len() > 30, "only {} cells landed far outside", coarse.len());
    // The box middle ATTAINS ~180 km against its 150 request (the ramp is
    // centred on the boundary; region_attainment in the receipt says so), so
    // the delivered mean is graded against the attainable field, within 20%.
    assert!(
        (mean(&fine) / 180_000.0 - 1.0).abs() < 0.20,
        "inside the box the mesh delivers {:.1} km against an attainable ~180 km",
        mean(&fine) / 1000.0
    );
    assert!(
        (mean(&coarse) / 600_000.0 - 1.0).abs() < 0.20,
        "far outside the mesh delivers {:.1} km against a requested 600 km",
        mean(&coarse) / 1000.0
    );
    assert!(
        mean(&coarse) / mean(&fine) > 2.5,
        "the delivered refinement ratio is only {:.2}",
        mean(&coarse) / mean(&fine)
    );

    // And the contract the receipt actually states: delivered over REQUESTED,
    // cell by cell, against the spec's continuous field rather than against a
    // region's nominal number.
    let r = &out.receipt;
    eprintln!(
        "delivered/requested: p05 {:.4}  median {:.4}  p95 {:.4}",
        r.delivered_over_requested_p05, r.delivered_over_requested_median, r.delivered_over_requested_p95
    );
    assert!(
        (r.delivered_over_requested_median - 1.0).abs() < 0.10,
        "the median cell is {:.1}% off what the spec asked for at its own location",
        (r.delivered_over_requested_median - 1.0) * 100.0
    );
    assert!(
        r.delivered_over_requested_p05 > 0.6 && r.delivered_over_requested_p95 < 1.6,
        "the delivered spread p05 {:.3} .. p95 {:.3} is wider than an SCVT's own lattice distortion",
        r.delivered_over_requested_p05,
        r.delivered_over_requested_p95
    );
}

#[test]
fn a_graded_request_regenerates_byte_identically() {
    // The mesh registry pins grid files by byte count and SHA-256, so a
    // graded mesh that cannot reproduce its own bytes on regeneration makes
    // every registered graded mesh permanently red. The ladder carries no
    // RNG and every ordering is canonical; this measures that end to end
    // through the real writer.
    let spec = MeshSpec {
        background_km: 600.0,
        regions: vec![Region {
            shape: Shape::Cap {
                center_deg: [39.0, -98.0],
                radius_km: 4000.0,
            },
            spacing_km: 300.0,
            transition: TransitionField::Km(3600.0),
        }],
        name: Some("regeneration identity".into()),
    };
    let request = GenerateRequest {
        spec,
        sizing_samples: 50_000,
        ..Default::default()
    };
    let digest = |tag: &str| -> String {
        let out = generate(&request, |_| {}).unwrap_or_else(|e| panic!("{e}"));
        let path = scratch(&format!("graded-regen-{tag}.grid.nc"));
        let _ = std::fs::remove_file(&path);
        let written = write_grid(
            &out.mesh,
            &path,
            &Provenance {
                spec_json: serde_json::to_string(&out.spec).unwrap(),
                request: "regeneration identity".into(),
                receipt_json: rw_mpas::mesh::provenance_json(&out.receipt).unwrap(),
                static_coordinates: None,
            },
            false,
        )
        .expect("write");
        let _ = std::fs::remove_file(&path);
        written.sha256
    };
    let first = digest("a");
    let second = digest("b");
    eprintln!("graded regeneration digests: {first} / {second}");
    assert_eq!(
        first, second,
        "two generations of the same graded spec wrote different bytes; the registry could never pin this mesh"
    );
}

#[test]
fn a_graded_mesh_is_numbered_so_neighbours_share_memory_lines() {
    // The ladder appends each level's insertions after the cells before it.
    // Written in that order, a fine cell's neighbours sat across the whole
    // index range: the default 96 -> 3 km zoom globe's edges gathered their
    // two cells from 10.0 distinct 128-byte lines per warp, against 2.31 in
    // Morton order, and its dycore step took 378 ms against 244 ms renumbered
    // (mesh::order). This spec, in the generator's own order (rw_mpas_mesh at
    // dd8c2e697c, hex_order --locality, 2,195 cells), gathered 3.73 lines per
    // warp; renumbered, 2.29.
    use rw_mpas::mesh::order::{morton_order, neighbour_lines_per_warp};
    let spec = MeshSpec {
        background_km: 600.0,
        regions: vec![Region {
            shape: Shape::Cap {
                center_deg: [39.0, -98.0],
                radius_km: 4000.0,
            },
            spacing_km: 300.0,
            transition: TransitionField::Km(3600.0),
        }],
        name: Some("cell order".into()),
    };
    let request = GenerateRequest {
        spec,
        sizing_samples: 50_000,
        ..Default::default()
    };
    let out = generate(&request, |_| {}).unwrap_or_else(|e| panic!("{e}"));
    let census = out.receipt.cell_order.neighbour_128byte_lines_per_warp;
    let morton = morton_order(&out.mesh.cell_xyz).expect("finite centres");
    let out_of_place = morton.iter().enumerate().filter(|&(i, &j)| i != j).count();
    eprintln!(
        "cell order: {} cells, key {}, {census:.3} lines per warp, {out_of_place} cells out of Morton place",
        out.mesh.n_cells, out.receipt.cell_order.key
    );
    assert_eq!(census, neighbour_lines_per_warp(&out.mesh), "the receipt's census is not the emitted mesh's");
    assert!(
        out_of_place * 1000 <= out.mesh.n_cells,
        "{out_of_place} of {} cells are not where the Morton order puts them",
        out.mesh.n_cells
    );
    assert!(
        census < 3.0,
        "neighbour gathers touch {census:.2} lines per warp, the generator-order class"
    );
}

#[test]
fn a_request_that_does_not_fit_the_card_is_refused_with_both_numbers() {
    // The measured answer to "8 km over this box, sized to fit a 10 GiB
    // budget on the 70 SM part" is that it does not fit. The door has to say
    // so with the numbers rather than quietly deliver a coarser mesh under
    // the name that was asked for.
    //
    // THE SPACING MOVED, and why it moved is the point. This case used to be
    // 15 km, and 15 km used to overrun because the sizing model quoted ONE
    // card's fixed term -- a 170 SM part's 9,798 MiB -- for every card, which
    // leaves a 10 GiB budget holding 5,349 cells. Derived per part, the 70 SM
    // card pays 5,384 MiB and the same budget holds 58,777, so 15 km over
    // this box now FITS, and so does 12 km. That is the fix, not a regression, and the refusal
    // path is still worth a test: it just needs a request that genuinely
    // overruns the card it names.
    let spec = MeshSpec {
        background_km: 120.0,
        regions: vec![Region {
            shape: Shape::LatLonBox {
                lat_deg: [30.0, 45.0],
                lon_deg: [-105.0, -90.0],
            },
            spacing_km: 8.0,
            transition: TransitionField::Cells(30.0),
        }],
        name: None,
    };
    // A budget is only a cell count once a CARD is named: the footprint
    // model's fixed term is a property of the part.  This is the 70 SM part.
    let card = footprint::card("rtx-5070-ti").expect("a measured card");
    let budget = card.cells_that_fit(10.0 * 1024.0).unwrap();
    let needs = spec.predicted_cells(100_000);
    eprintln!(
        "8 km over the box at a 120 km background needs {needs:.0} cells; a 10 GiB budget on {} holds {budget}",
        card.key
    );
    assert!(needs > budget as f64, "the case this test is built on has stopped being true");

    let request = GenerateRequest {
        spec: spec.clone(),
        budget_mib: Some(10.0 * 1024.0),
        card: Some(card),
        fit_spacing: false,
        sizing_samples: 50_000,
        ..Default::default()
    };
    let err = generate(&request, |_| {}).unwrap_err().to_string();
    assert!(err.contains("device budget holds"), "{err}");
    assert!(err.contains("--fit-spacing"), "the refusal does not name the remedy: {err}");

    // And with the remedy applied, it sizes: every spacing scaled by one
    // factor, the ratio between them untouched.
    let (fitted, k) = spec.fitted_to(budget, 50_000).unwrap();
    eprintln!(
        "--fit-spacing scales by {k:.4}: {:.1} km inside, {:.1} km background, {:.0} cells",
        fitted.finest_km(),
        fitted.background_km,
        fitted.predicted_cells(50_000)
    );
    assert!(k > 1.0, "fitting a too-large request must coarsen it, not refine it");
    assert!(
        (fitted.background_km / fitted.finest_km() - spec.background_km / spec.finest_km()).abs() < 1e-9,
        "the refinement ratio moved"
    );
}

#[test]
fn writing_over_an_existing_mesh_is_refused_unless_asked() {
    let out = generate(&small_request(400), |_| {}).expect("generate");
    let path = scratch("clobber.grid.nc");
    let _ = std::fs::remove_file(&path);
    write_grid(&out.mesh, &path, &Provenance::default(), false).expect("first write");
    let err = write_grid(&out.mesh, &path, &Provenance::default(), false)
        .unwrap_err()
        .to_string();
    assert!(err.contains("--clobber"), "{err}");
    assert!(err.contains("under the same name"), "the refusal does not name the breakage: {err}");
    write_grid(&out.mesh, &path, &Provenance::default(), true).expect("clobbering write");
    let _ = std::fs::remove_file(&path);
}

#[test]
fn the_same_request_produces_the_same_mesh_twice() {
    // Reproducibility is not decoration: a mesh is pinned downstream by
    // sha256, so a generator that wandered between runs would break every
    // registry row that cited it.
    let a = generate(&small_request(500), |_| {}).expect("first");
    let b = generate(&small_request(500), |_| {}).expect("second");
    assert_eq!(a.mesh.n_cells, b.mesh.n_cells);
    let mut worst = 0.0f64;
    for i in 0..a.mesh.n_cells {
        worst = worst.max(chord(a.mesh.cell_xyz[i], b.mesh.cell_xyz[i]));
    }
    assert_eq!(worst, 0.0, "two runs of the same request moved a generator by {worst:.3e}");

    let pa = scratch("repeat-a.grid.nc");
    let pb = scratch("repeat-b.grid.nc");
    let wa = write_grid(&a.mesh, &pa, &Provenance::default(), true).expect("write a");
    let wb = write_grid(&b.mesh, &pb, &Provenance::default(), true).expect("write b");
    assert_eq!(wa.sha256, wb.sha256, "the same mesh wrote two different files");
    assert_eq!(wa.file_id, wb.file_id);
    let _ = std::fs::remove_file(&pa);
    let _ = std::fs::remove_file(&pb);
}

/// The same request twice, written the way the BINARY writes them.
///
/// `the_same_request_produces_the_same_mesh_twice` writes with
/// `Provenance::default()` -- an empty document -- so it could never see what
/// `rw_mpas_mesh` actually stamps. It did not: the binary put the receipt in,
/// receipts carry `relaxation_seconds`, and two runs of one command produced
/// two digests. The port registry pins a grid by byte count and SHA-256, so
/// that made every generated mesh unregisterable. This test writes through the
/// same provenance the binary builds.
#[test]
fn the_binarys_own_provenance_writes_the_same_bytes_twice() {
    let a = generate(&small_request(500), |_| {}).expect("first");
    let b = generate(&small_request(500), |_| {}).expect("second");

    let provenance_of = |g: &rw_mpas::mesh::Generated| Provenance {
        spec_json: serde_json::to_string(&g.spec).expect("spec"),
        request: format!(
            "{} cells, {:.3} km finest, {:.3} km background",
            g.receipt.delivered_cells,
            g.receipt.finest_requested_km,
            g.receipt.background_requested_km
        ),
        receipt_json: rw_mpas::mesh::provenance_json(&g.receipt).expect("provenance"),
        static_coordinates: None,
    };

    let pa = scratch("stamped-a.grid.nc");
    let pb = scratch("stamped-b.grid.nc");
    let wa = write_grid(&a.mesh, &pa, &provenance_of(&a), true).expect("write a");
    let wb = write_grid(&b.mesh, &pb, &provenance_of(&b), true).expect("write b");
    assert_eq!(
        wa.sha256, wb.sha256,
        "two runs of the same command wrote two different grid files, so the \
         pair can never be registered under a stable name"
    );
    assert_eq!(wa.bytes, wb.bytes);

    // And the duration is genuinely absent from the bytes, not merely equal.
    let bytes = std::fs::read(&pa).expect("read back");
    let text = String::from_utf8_lossy(&bytes);
    assert!(
        !text.contains("relaxation_seconds"),
        "the grid file still carries a wall-clock reading"
    );
    // The measurement still exists for the caller, in the receipt itself.
    let side = serde_json::to_string(&a.receipt).expect("receipt");
    assert!(side.contains("relaxation_seconds"));

    let _ = std::fs::remove_file(&pa);
    let _ = std::fs::remove_file(&pb);
}

#[test]
fn a_mesh_that_would_fail_the_gate_is_never_written() {
    // The gate runs BEFORE the file exists. Feed derive a ring table that is
    // not a closed sphere and confirm nothing lands on disk.
    let out = generate(&small_request(300), |_| {}).expect("generate");
    let mut rings = Rings {
        offsets: vec![0u32; out.mesh.n_cells + 1],
        values: Vec::new(),
    };
    // Every cell claims the same three neighbours: a ring table that parses and
    // is not a sphere.
    for i in 0..out.mesh.n_cells {
        rings.offsets[i + 1] = rings.offsets[i] + 3;
        for k in 0..3 {
            rings.values.push(((i + k + 1) % out.mesh.n_cells) as u32);
        }
    }
    let attempt = MpasMesh::derive(
        out.mesh.cell_xyz.clone(),
        vec![1.0; out.mesh.n_cells],
        &rings,
        1e-3,
    );
    let message = match attempt {
        Err(e) => e.to_string(),
        Ok(bad) => rw_mpas::mesh::validate::validate(&bad, Limits::default())
            .unwrap_err()
            .to_string(),
    };
    assert!(
        !message.is_empty(),
        "a ring table that is not a sphere produced a mesh with no complaint"
    );
    eprintln!("the gate said: {message}");
}

/// One mesh, two source paths, one digest.
///
/// The sibling of the `relaxation_seconds` defect, and it survived that fix.
/// The `--from-centres` rebuild route wrote the SOURCE PATH into the grid's own
/// provenance, so two byte-identical copies of one mesh, sitting at two
/// filenames, rebuilt to two different SHA-256s at an identical 2,741,008
/// bytes. Measured against the exe before this test existed: 5d2782cf... and
/// 955bac3b....
///
/// A registry pin is cross-machine by nature. A duration varies with the run; a
/// path varies with the box, which is worse. `source_sha256` already names the
/// content, which is the only thing a reader can act on.
#[test]
fn the_rebuild_route_names_its_source_by_digest_and_not_by_path() {
    let out = generate(&small_request(500), |_| {}).expect("generate");

    // What rw_mpas_mesh's rebuild arm stamps, built the way the binary
    // builds it: everything except the caller's filesystem.
    let stamped = |sha: &str| Provenance {
        spec_json: serde_json::json!({
            "route": "from-centres",
            "source_sha256": sha,
            "source_sphere_radius": 1.0,
        })
        .to_string(),
        request: format!(
            "{} cell centres rebuilt from a grid with sha256 {}",
            out.mesh.n_cells, sha
        ),
        receipt_json: rw_mpas::mesh::provenance_json(&out.receipt).expect("provenance"),
        static_coordinates: None,
    };

    let digest = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    let one = scratch("rebuilt-from-one.grid.nc");
    let two = scratch("rebuilt-from-two.grid.nc");
    let wa = write_grid(&out.mesh, &one, &stamped(digest), true).expect("write one");
    let wb = write_grid(&out.mesh, &two, &stamped(digest), true).expect("write two");
    assert_eq!(
        wa.sha256, wb.sha256,
        "the same source content rebuilt to two different files"
    );

    // And no absolute path reached the bytes on the way.
    let bytes = std::fs::read(&one).expect("read back");
    let text = String::from_utf8_lossy(&bytes);
    assert!(
        !text.contains("rebuilt-from-one"),
        "the grid file carries the filename it was rebuilt from"
    );
    assert!(
        text.contains(digest),
        "the grid file does not name its source content at all"
    );

    let _ = std::fs::remove_file(&one);
    let _ = std::fs::remove_file(&two);
}
