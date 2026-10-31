use rw_mpas::gsl_oro::{self, BASE_NAMES, Topography};
use rw_store::netcdf_classic::{NcClassicWriter, NcData, NcDim, NcFormat, NcType, NcVarDef};
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;
use std::process::Command;

struct Flat {
    row: Vec<f32>,
    ny: usize,
}
impl Topography for Flat {
    fn shape(&self) -> (usize, usize) {
        (self.row.len(), self.ny)
    }
    fn row(&mut self, _j: usize) -> gsl_oro::Result<&[f32]> {
        Ok(&self.row)
    }
}

#[test]
fn flat_terrain_and_native_area_order() {
    let areas = gsl_oro::cell_areas(
        &[3],
        &[3, 1, 2, 0],
        4,
        &[0.0001f32, 0.0002, 0.0003],
        6371229.,
        1.5,
    )
    .unwrap();
    let dc = ((0.0003f32 + 0.0001) + 0.0002) / 3.;
    let dc = (dc * 6371229.) * 1.5;
    assert_eq!(areas[0].to_bits(), (dc * dc).to_bits());
    let mut ss = Flat {
        row: vec![0.; 43200],
        ny: 21600,
    };
    let mut ls = Flat {
        row: vec![0.; 8640],
        ny: 4320,
    };
    let result =
        gsl_oro::generate(&[0.0f32], &[0.0], &[120000f32 * 120000.], &mut ss, &mut ls).unwrap();
    assert_eq!(result.small, vec![[0.; 10]]);
    assert_eq!(result.large, vec![[0.; 10]]);
    assert!(!result.duplicate[0]);
}

#[test]
fn source_conversion_is_single_precision_before_scaling() {
    for n in [i32::MIN, -123456789, -1, 0, 1, 16777217, i32::MAX] {
        assert_eq!(
            gsl_oro::decode_height(n.to_be_bytes()).to_bits(),
            (n as f32 * 0.1).to_bits()
        );
    }
}

#[test]
fn bad_mesh_cannot_pair_wrong_edges_with_terrain() {
    assert!(gsl_oro::cell_areas(&[2], &[1, 9], 2, &[1000f32], 6371229., 1.).is_err());
    assert!(gsl_oro::cell_areas(&[0], &[0], 1, &[1000f32], 6371229., 1.).is_err());
    assert!(gsl_oro::cell_areas(&[1], &[1], 1, &[f32::NAN], 6371229., 1.).is_err());
}

#[test]
fn native_read_errors_continue_with_initialized_fields() {
    use gsl_oro::Tiles;
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(format!(".gsl-read-errors-{}", std::process::id()));
    fs::create_dir(&root).unwrap();
    let mut unavailable = Tiles::open(&root, true).unwrap();
    assert!(unavailable.read_error().is_some());
    let mut ridge = Flat { row: (0..8640).map(|i| (i % 11) as f32 * 100.).collect(), ny: 4320 };
    let fields = gsl_oro::generate(&[0f32], &[0.], &[120000f32 * 120000.],
                                   &mut unavailable, &mut ridge).unwrap();
    assert_eq!(fields.small, vec![[0.; 10]]);
    assert_eq!(fields.large, vec![[0.; 10]]);
    let mut missing_large = Tiles::open(&root, false).unwrap();
    let mut small = Flat { row: (0..43200).map(|i| (i % 11) as f32 * 100.).collect(), ny: 21600 };
    let fields = gsl_oro::generate(&[0f32], &[0.], &[120000f32 * 120000.],
                                   &mut small, &mut missing_large).unwrap();
    assert!(fields.small[0][0] > 0.);
    assert_eq!(fields.large, vec![[0.; 10]]);
    fs::create_dir(root.join("topo_ugwp_2.5m")).unwrap();
    let tile = root.join("topo_ugwp_2.5m/00001-08640.00001-04320");
    fs::write(&tile, []).unwrap();
    assert!(Tiles::open(&root, false).unwrap().read_error().is_some());
    fs::write(&tile, 1000i32.to_be_bytes()).unwrap();
    let mut partial = Tiles::open(&root, false).unwrap();
    assert!(partial.read_error().is_none());
    assert_eq!(partial.row(0).unwrap()[0].to_bits(), 100f32.to_bits());
    assert!(partial.row(0).unwrap()[1..].iter().all(|v| v.to_bits() == 0));
    assert!(partial.row(4319).unwrap().iter().all(|v| v.to_bits() == 0));
    fs::remove_file(&tile).unwrap();
    fs::create_dir(&tile).unwrap();
    assert!(Tiles::open(&root, false).unwrap().read_error().is_some());
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn comparison_detects_one_word_and_preserves_its_inputs() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(format!(".gsl-comparison-{}", std::process::id()));
    fs::create_dir(&root).unwrap();
    let reference = root.join("reference.nc");
    let different = root.join("different.nc");
    let double = root.join("double.nc");
    for (path, wide, bump) in [
        (&reference, false, false),
        (&different, false, true),
        (&double, true, false),
    ] {
        let defs = ["ls", "ss"]
            .into_iter()
            .flat_map(|suffix| {
                BASE_NAMES.into_iter().map(move |name| {
                    NcVarDef::new(
                        format!("{name}{suffix}"),
                        if wide { NcType::Double } else { NcType::Float },
                        vec![0],
                    )
                })
            })
            .collect();
        let mut writer = NcClassicWriter::create(
            path,
            NcFormat::Offset64,
            vec![NcDim::fixed("nCells", 2)],
            vec![],
            defs,
            0,
        )
        .unwrap();
        for suffix in ["ls", "ss"] {
            for name in BASE_NAMES {
                let v = if bump && suffix == "ss" && name == "oa3" {
                    f32::from_bits(1.0f32.to_bits() + 1)
                } else {
                    1.0
                };
                let name = format!("{name}{suffix}");
                if wide {
                    writer
                        .put(&name, NcData::Doubles(&[v as f64, 1.0]))
                        .unwrap();
                } else {
                    writer.put(&name, NcData::Floats(&[v, 1.0])).unwrap();
                }
            }
        }
        writer.finish().unwrap();
    }
    let bin = env!("CARGO_BIN_EXE_rw_mpas_gsl_oro");
    let result = Command::new(bin)
        .args([
            "--compare",
            reference.to_str().unwrap(),
            different.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(!result.status.success());
    let receipt: serde_json::Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(receipt["unequal_words"], 1);
    let result = Command::new(bin)
        .args([
            "--compare",
            double.to_str().unwrap(),
            reference.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("type does not match --precision"));
    let hash = rw_mpas::sha256_file(&reference).unwrap();
    let result = Command::new(bin)
        .args([
            "--compare",
            reference.to_str().unwrap(),
            different.to_str().unwrap(),
            "--receipt",
            reference.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert_eq!(hash, rw_mpas::sha256_file(&reference).unwrap());
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn comparison_preserves_signed_zero_and_signaling_nan_words() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(format!(".gsl-word-bits-{}", std::process::id()));
    fs::create_dir(&root).unwrap();
    for wide in [false, true] {
        let reference = root.join("reference.nc");
        let different = root.join("different.nc");
        for (path, bump) in [(&reference, false), (&different, true)] {
            let defs = ["ls", "ss"].into_iter().flat_map(|suffix| {
                BASE_NAMES.into_iter().map(move |base| NcVarDef::new(
                    format!("{base}{suffix}"),
                    if wide { NcType::Double } else { NcType::Float }, vec![0]))
            }).collect();
            let mut writer = NcClassicWriter::create(path, NcFormat::Offset64,
                vec![NcDim::fixed("nCells", 4)], vec![], defs, 0).unwrap();
            for suffix in ["ls", "ss"] {
                for base in BASE_NAMES {
                    let name = format!("{base}{suffix}");
                    let change = bump && name == "var2dls";
                    if wide {
                        let values = [if change { -0.0 } else { 0.0 }, -0.0,
                            f64::from_bits(if change { 0x7ff8000000000001 } else { 0x7ff0000000000001 }),
                            f64::from_bits(0x7ff8000000000002)];
                        writer.put(&name, NcData::Doubles(&values)).unwrap();
                    } else {
                        let values = [if change { -0.0 } else { 0.0 }, -0.0,
                            f32::from_bits(if change { 0x7fc00001 } else { 0x7f800001 }),
                            f32::from_bits(0x7fc00002)];
                        writer.put(&name, NcData::Floats(&values)).unwrap();
                    }
                }
            }
            writer.finish().unwrap();
        }
        let result = Command::new(env!("CARGO_BIN_EXE_rw_mpas_gsl_oro"))
            .args(["--compare", reference.to_str().unwrap(), different.to_str().unwrap(),
                "--precision", if wide { "double" } else { "single" }]).output().unwrap();
        assert!(!result.status.success());
        let receipt: serde_json::Value = serde_json::from_slice(&result.stdout).unwrap();
        assert_eq!(receipt["words"], 80);
        assert_eq!(receipt["unequal_words"], 2, "signed zero and NaN quiet bit must both differ");
        if !wide {
            let witness_bin = env!("CARGO_BIN_EXE_rw_mpas_gsl_word_witness");
            let witness = root.join("native-words.json");
            let compared = root.join("compared.json");
            assert!(Command::new(witness_bin).args([&reference, &witness]).status().unwrap().success());
            let compare = |candidate: &Path, original: &Path, out: &Path| {
                Command::new(witness_bin).arg("--compare-witness")
                    .args([original, candidate, out]).output().unwrap()
            };
            assert!(compare(&reference, &witness, &compared).status.success());
            assert!(!compare(&different, &witness, &compared).status.success());
            let result: serde_json::Value = serde_json::from_slice(&fs::read(&compared).unwrap()).unwrap();
            assert_eq!(result["words"], 80);
            assert_eq!(result["unequal_words"], 2);
            let original: serde_json::Value = serde_json::from_slice(&fs::read(&witness).unwrap()).unwrap();
            let faulty = root.join("faulty.json");
            for fault in 0..5 {
                let mut bad = original.clone();
                match fault {
                    0 => { bad["fields"].as_array_mut().unwrap().pop(); }
                    1 => bad["fields"][0]["shape"] = serde_json::json!([3]),
                    2 => { bad["fields"][0]["words"].as_array_mut().unwrap().pop(); }
                    3 => bad["fields"][0]["words"][0] = serde_json::json!(4294967296u64),
                    _ => bad["fields"][1]["field"] = bad["fields"][0]["field"].clone(),
                }
                fs::write(&faulty, serde_json::to_vec(&bad).unwrap()).unwrap();
                assert!(!compare(&reference, &faulty, &compared).status.success());
            }
            let hash = rw_mpas::sha256_file(&witness).unwrap();
            assert!(!compare(&reference, &witness, &witness).status.success());
            assert_eq!(hash, rw_mpas::sha256_file(&witness).unwrap());
        }
    }
    fs::remove_dir_all(root).unwrap();
}

fn terrain(x: usize, y: usize, nx: usize, ny: usize) -> i32 {
    if x >= nx / 8 && x < nx / 3 && y >= ny / 3 && y < 2 * ny / 3 {
        return 0;
    }
    if x >= nx / 2 && x < 3 * nx / 4 && y >= ny / 3 && y < 2 * ny / 3 {
        return 750;
    }
    if x % 113 == 0 && y % 79 == 0 {
        return 100000;
    }
    ((x * 37 + y * 29 + (x % 17) * (y % 19) * 101) % 15000) as i32 - 5000
}

fn make_geog(root: &Path) {
    for (sub, nx, ny) in [
        ("topo_ugwp_30s", 43200, 21600),
        ("topo_ugwp_2.5m", 8640, 4320),
    ] {
        let dir = root.join(sub);
        fs::create_dir_all(&dir).unwrap();
        for ty in (0..ny).step_by(4320) {
            for tx in (0..nx).step_by(8640) {
                let path = dir.join(format!(
                    "{:05}-{:05}.{:05}-{:05}",
                    tx + 1,
                    tx + 8640,
                    ty + 1,
                    ty + 4320
                ));
                let mut file = BufWriter::new(File::create(path).unwrap());
                let mut bytes = vec![0u8; 8640 * 4];
                for y in ty..ty + 4320 {
                    for x in 0..8640 {
                        bytes[x * 4..x * 4 + 4]
                            .copy_from_slice(&terrain(tx + x, y, nx, ny).to_be_bytes());
                    }
                    file.write_all(&bytes).unwrap();
                }
                file.flush().unwrap();
            }
        }
    }
}

// Full-resolution source plus the untouched MPAS driver, executed through
// rw_mpas_gsl_oro. This is standalone evidence, never whole-forecast proof.
// Build oracle with tests/gsl_oracle/Makefile, set GSL_ORACLE and GSL_SCRATCH.
#[test]
#[ignore = "requires pristine Fortran oracle and approximately 4 GB owned geography scratch"]
fn native_driver_word_replay() {
    let oracle = std::env::var("GSL_ORACLE").expect("GSL_ORACLE executable");
    let root = std::path::PathBuf::from(
        std::env::var("GSL_SCRATCH").expect("GSL_SCRATCH owned directory"),
    );
    fs::create_dir_all(&root).unwrap();
    let published_geog = std::env::var_os("GSL_GEOG").map(std::path::PathBuf::from);
    let geog = published_geog.clone().unwrap_or_else(|| root.join("geog"));
    // Published archives stay read-only; create synthetic geography only in
    // newly owned scratch. Both paths replay the same pristine whole modules.
    if published_geog.is_none() {
        assert!(
            !geog.exists(),
            "scratch geography must be newly owned by this test"
        );
        make_geog(&geog);
    }
    let mut cells = vec![
        (0., -80., 240000.),
        (45., 10., 120000.),
        (-30., 179.8, 240000.),
        (30., -179.8, 240000.),
        (89.95, 45., 180000.),
        (-89.95, -45., 180000.),
        (88., 179., 40000.),
        (-88., -179., 40000.),
        (0., -120., 120000.),
        (0., 30., 120000.),
        (20., 120., 7500.),
        (20., 120., 7499.),
        (20., 120., 3000.),
        (-45., 200., 15000.),
        (60., 270., 15001.),
        (-60., 300., 22000.),
        (5., -70., 8000.),
        (-5., -70., 9000.),
        (33.021, -110.013, 35000.),
        (-33.027, 110.018, 42000.),
        (89.3, 120.013, 60000.),
        (-89.3, 120.013, 60000.),
        // The center block ends between the final large-scale row and the
        // north pole: lg nearest_j_south uses j<topo_y, sm uses j<=topo_y.
        // Source: mpas_gsl_oro_data_lg_scale.F:988-1012; sm_scale.F:702-726.
        (89.92, 120., 15000.),
        (30.024, 110.027, 7500.),
        (
            (-90. + (180. / 21600.) * (11001. - 0.5)),
            (-180. + (360. / 43200.) * (10001. - 0.5)),
            1000.,
        ),
        // One nonzero fine-grid point reaches the empty effective-length
        // bands; a one-point large-scale box reaches zero variance and OA.
        (
            (-90. + (180. / 21600.) * (11001. - 0.5)),
            (-180. + (360. / 43200.) * (25001. - 0.5)),
            1000.,
        ),
        (
            (-90. + (180. / 4320.) * (2161. - 0.5)),
            (-180. + (360. / 8640.) * (1001. - 0.5)),
            7500.,
        ),
        (42.071, 123.013, 12501.),
        (-42.037, -123.019, 18001.),
    ];
    let double = std::env::var("GSL_DOUBLE").is_ok();
    if !double {
        // COS of binary32 Pi/2 is a finite negative number. Native permits
        // that width and still wraps the resulting longitude bounds.
        cells.extend([(90., 120., 15000.), (-90., 120., 15000.)]);
    }
    let nc = cells.len();
    let me = 3usize;
    let ne = nc * me;
    let bin = env!("CARGO_BIN_EXE_rw_mpas_gsl_oro");
    let mut receipts = Vec::new();
    for (scenario, unit, scaling) in [("earth", false, 1.0), ("unit_scaled", true, 1.375)] {
        let grid = root.join(format!("{scenario}.nc"));
        let native_input = root.join(format!("{scenario}.bin"));
        let native_output = root.join(format!("{scenario}.native"));
        let output = root.join(format!("{scenario}.oro.nc"));
        let rk = |v: f64| if double { v } else { v as f32 as f64 };
        let lat: Vec<_> = cells
            .iter()
            .map(|c| rk(c.0 * std::f64::consts::PI / 180.))
            .collect();
        let lon: Vec<_> = cells
            .iter()
            .map(|c| rk(c.1 * std::f64::consts::PI / 180.))
            .collect();
        let dc: Vec<_> = cells
            .iter()
            .flat_map(|c| [rk(c.2), rk(c.2), rk(c.2)])
            .map(|v| rk(if unit { v / 6371229. } else { v }))
            .collect();
        let counts = vec![me as i32; nc];
        let edges: Vec<_> = (1..=ne as i32).collect();
        let mut w = NcClassicWriter::create(
            &grid,
            NcFormat::Offset64,
            vec![
                NcDim::fixed("nCells", nc),
                NcDim::fixed("nEdges", ne),
                NcDim::fixed("maxEdges", me),
            ],
            vec![],
            vec![
                NcVarDef::new("latCell", NcType::Double, vec![0]),
                NcVarDef::new("lonCell", NcType::Double, vec![0]),
                NcVarDef::new("dcEdge", NcType::Double, vec![1]),
                NcVarDef::new("nEdgesOnCell", NcType::Int, vec![0]),
                NcVarDef::new("edgesOnCell", NcType::Int, vec![0, 2]),
            ],
            0,
        )
        .unwrap();
        for (n, v) in [("latCell", &lat), ("lonCell", &lon), ("dcEdge", &dc)] {
            w.put(n, NcData::Doubles(v)).unwrap();
        }
        w.put("nEdgesOnCell", NcData::Ints(&counts)).unwrap();
        w.put("edgesOnCell", NcData::Ints(&edges)).unwrap();
        w.finish().unwrap();
        let mut f = File::create(&native_input).unwrap();
        for v in [nc as i32, ne as i32, me as i32] {
            f.write_all(&v.to_le_bytes()).unwrap();
        }
        for v in [6371229., scaling]
            .into_iter()
            .chain(lat.iter().copied())
            .chain(lon.iter().copied())
            .chain(dc.iter().copied())
        {
            if double {
                f.write_all(&v.to_le_bytes()).unwrap();
            } else {
                f.write_all(&(v as f32).to_le_bytes()).unwrap();
            }
        }
        for v in counts.iter().chain(&edges) {
            f.write_all(&v.to_le_bytes()).unwrap();
        }
        drop(f);
        assert!(
            Command::new(&oracle)
                .arg(&native_input)
                .arg(&geog)
                .arg(&native_output)
                .status()
                .unwrap()
                .success()
        );
        let result = Command::new(bin)
            .args([
                "--grid",
                grid.to_str().unwrap(),
                "--geog",
                geog.to_str().unwrap(),
                "--out",
                output.to_str().unwrap(),
                "--cell-scaling",
                &scaling.to_string(),
                "--precision",
                if double { "double" } else { "single" },
            ])
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        let ncfile = netcrust::File::open(&output).unwrap();
        let raw = fs::read(&native_output).unwrap();
        assert_eq!(raw.len(), nc * 20 * if double { 8 } else { 4 });
        let mut differences = Vec::new();
        let reference = root.join(format!("{scenario}.native.nc"));
        let defs: Vec<_> = ["ls", "ss"]
            .into_iter()
            .flat_map(|suffix| {
                BASE_NAMES.into_iter().map(move |base| {
                    NcVarDef::new(
                        format!("{base}{suffix}"),
                        if double {
                            NcType::Double
                        } else {
                            NcType::Float
                        },
                        vec![0],
                    )
                })
            })
            .collect();
        let mut reference_writer = NcClassicWriter::create(
            &reference,
            NcFormat::Offset64,
            vec![NcDim::fixed("nCells", nc)],
            vec![],
            defs,
            0,
        )
        .unwrap();
        for (scale, suffix) in ["ls", "ss"].iter().enumerate() {
            for (k, base) in BASE_NAMES.iter().enumerate() {
                let name = format!("{base}{suffix}");
                let values = ncfile.read_f64(&name).unwrap();
                let mut native_values = Vec::new();
                for (cell, &value) in values.iter().enumerate() {
                    let offset = (scale * nc * 10 + cell * 10 + k) * if double { 8 } else { 4 };
                    let (expected, actual) = if double {
                        (
                            u64::from_le_bytes(raw[offset..offset + 8].try_into().unwrap()),
                            value.to_bits(),
                        )
                    } else {
                        (
                            u32::from_le_bytes(raw[offset..offset + 4].try_into().unwrap()) as u64,
                            (value as f32).to_bits() as u64,
                        )
                    };
                    native_values.push(if double {
                        f64::from_bits(expected)
                    } else {
                        f32::from_bits(expected as u32) as f64
                    });
                    if expected != actual {
                        differences.push(serde_json::json!({"field":name,"cell":cell,"expected":expected,"actual":actual}));
                    }
                }
                if double {
                    reference_writer
                        .put(&name, NcData::Doubles(&native_values))
                        .unwrap();
                } else {
                    reference_writer
                        .put(
                            &name,
                            NcData::Floats(
                                &native_values.iter().map(|v| *v as f32).collect::<Vec<_>>(),
                            ),
                        )
                        .unwrap();
                }
            }
        }
        reference_writer.finish().unwrap();
        let comparison = Command::new(bin)
            .args([
                "--compare",
                reference.to_str().unwrap(),
                output.to_str().unwrap(),
                "--precision",
                if double { "double" } else { "single" },
            ])
            .output()
            .unwrap();
        assert_eq!(comparison.status.success(), differences.is_empty());
        let original_hash = rw_mpas::sha256_file(&output).unwrap();
        let refused = Command::new(bin)
            .args([
                "--grid",
                grid.to_str().unwrap(),
                "--geog",
                geog.to_str().unwrap(),
                "--out",
                output.to_str().unwrap(),
            ])
            .output()
            .unwrap();
        assert!(!refused.status.success());
        assert_eq!(
            original_hash,
            rw_mpas::sha256_file(&output).unwrap(),
            "existing output was changed"
        );
        receipts.push(serde_json::json!({"scenario":scenario,"cells":nc,"words":nc*20,"unequal_words":differences.len(),
            "differences":differences,"generation":serde_json::from_slice::<serde_json::Value>(&result.stdout).unwrap(),
            "comparison_command":serde_json::from_slice::<serde_json::Value>(&comparison.stdout).unwrap(),
            "existing_output_preserved":true,
            "native_output_sha256":rw_mpas::sha256_file(&native_output).unwrap()}));
    }
    let receipt = serde_json::json!({"schema":"gsl-standalone-native/v1","rkind_bytes":if double {8} else {4},
        "tier":if published_geog.is_some() {"standalone full-resolution pristine-module published geography"} else {"standalone full-resolution pristine-module synthetic geography"},
        "geography":geog,
        "oracle_sha256":rw_mpas::sha256_file(Path::new(&oracle)).unwrap(),"scenarios":receipts});
    let path = std::env::var("GSL_RECEIPT").expect("GSL_RECEIPT path");
    fs::write(path, serde_json::to_string_pretty(&receipt).unwrap()).unwrap();
    if published_geog.is_none() {
        fs::remove_dir_all(&geog).unwrap();
    }
    for entry in fs::read_dir(&root).unwrap() {
        let p = entry.unwrap().path();
        if p.is_file() {
            fs::remove_file(p).unwrap();
        }
    }
    for scenario in receipt["scenarios"].as_array().unwrap() {
        assert_eq!(scenario["unequal_words"], 0, "{scenario}");
    }
}
