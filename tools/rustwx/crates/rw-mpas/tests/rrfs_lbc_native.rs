//! Full case-9 numerical replay against RRFSx/MPAS-Model
//! src/core_init_atmosphere/mpas_init_atm_cases.F:7759-8986 at 1d6e370e.
//! The production Rust sibling runs directly; this standalone diagnostic is
//! separate from the required gpuwm-hex init-file and forecast proofs.
//! Needs a native LBC oracle built from that source (development harness,
//! not shipped): set MPAS_LBC_ORACLE and MPAS_LBC_REPLAY_DIR.

use rw_mpas::rrfs_lbc::compare::compare_lbc;
use rw_mpas::lbc::projection::gaussian_latitudes;
use rw_store::netcdf_classic::{
    NcAttr, NcClassicWriter, NcData, NcDim, NcFormat, NcType, NcVarDef,
};
use sha2::{Digest, Sha256};
use std::fs;
use std::io::{Read, Write};
use std::path::Path;
use std::process::Command;

const NC: usize = 4;
const NE: usize = 5;
const NZ: usize = 4;
const START: &str = "2024-02-28_21:00:00";
const STOP: &str = "2024-02-29_00:00:00";

fn grid(path: &Path, above_top: bool) {
    let dims = vec![
        NcDim::fixed("nCells", NC),
        NcDim::fixed("nEdges", NE),
        NcDim::fixed("nVertLevels", NZ),
        NcDim::fixed("nVertLevelsP1", NZ + 1),
        NcDim::fixed("TWO", 2),
        NcDim::fixed("maxEdges", 2),
    ];
    let mut attrs = [
        "model_name",
        "core_name",
        "version",
        "source",
        "Conventions",
        "git_version",
        "on_a_sphere",
        "is_periodic",
        "mesh_spec",
        "file_id",
    ]
    .iter()
    .map(|n| NcAttr::text(*n, "synthetic-rrfs-replay"))
    .collect::<Vec<_>>();
    attrs.extend([
        NcAttr::float("sphere_radius", 6371229.0),
        NcAttr::float("x_period", 0.0),
        NcAttr::float("y_period", 0.0),
    ]);
    let mut floats: Vec<(&str, Vec<usize>, Vec<f32>)> = Vec::new();
    let rad = std::f32::consts::PI / 180.0;
    for (name, d, values) in [
        ("latCell", 0, vec![-80.0, -29.75, 7.125, 69.4]),
        ("lonCell", 0, vec![-181.0, -88.4, 47.7, 185.0]),
        ("latEdge", 1, vec![-55.125, -11.31, 38.5, 87.0, -89.0]),
        ("lonEdge", 1, vec![-135.2, -20.35, 115.4, 190.0, -189.0]),
    ] {
        floats.push((name, vec![d], values.iter().map(|x| x * rad).collect()));
    }
    floats.push(("angleEdge", vec![1], vec![0.0, 0.7125, -1.8125, 2.7, -0.0]));
    let z: Vec<f32> = (0..NC)
        .flat_map(|c| {
            [0.0, 450.0, 1800.0, 3700.0, 6400.0]
                .map(|v| v * if above_top { 3.0 } else { 1.0 } + 40.0 * c as f32)
        })
        .collect();
    floats.push(("zgrid", vec![0, 3], z));
    floats.push((
        "zz",
        vec![0, 2],
        (0..NC)
            .flat_map(|c| [1.1, 0.92, 1.04, 0.88].map(|v| v + 0.015 * c as f32))
            .collect(),
    ));
    for (name, values) in [
        ("fzm", vec![0.0, 0.47, 0.43, 0.58, 0.0]),
        ("fzp", vec![0.0, 0.53, 0.57, 0.42, 0.0]),
        ("dzu", vec![0.0, 900.0, 1500.0, 2300.0, 0.0]),
        (
            "rdzw",
            vec![1.0 / 450.0, 1.0 / 1350.0, 1.0 / 1900.0, 1.0 / 2700.0, 0.0],
        ),
    ] {
        floats.push((name, vec![3], values));
    }
    for (name, factor) in [("zb", 0.00019), ("zb3", 0.00031)] {
        floats.push((
            name,
            vec![1, 4, 3],
            (0..NE)
                .flat_map(|e| {
                    (0..2).flat_map(move |s| {
                        (0..NZ + 1).map(move |k| {
                            if k == 0 || k == NZ {
                                0.0
                            } else {
                                factor * (e + 1) as f32 * (s + 1) as f32 * (k + 1) as f32
                            }
                        })
                    })
                })
                .collect(),
        ));
    }
    let ints = [
        (
            "cellsOnEdge",
            vec![1, 4],
            vec![1, 2, 2, 3, 3, 4, 4, 0, 0, 1],
        ),
        ("nEdgesOnCell", vec![0], vec![2; NC]),
        ("edgesOnCell", vec![0, 5], vec![1, 5, 1, 2, 2, 3, 3, 4]),
    ];
    let vars = floats
        .iter()
        .map(|(n, d, _)| NcVarDef::new(*n, NcType::Float, d.clone()))
        .chain(
            ints.iter()
                .map(|(n, d, _)| NcVarDef::new(*n, NcType::Int, d.clone())),
        )
        .collect();
    let mut writer = NcClassicWriter::create(path, NcFormat::Data64, dims, attrs, vars, 0).unwrap();
    for (name, _, values) in floats {
        writer.put(name, NcData::Floats(&values)).unwrap();
    }
    for (name, _, values) in ints {
        writer.put(name, NcData::Ints(&values)).unwrap();
    }
    writer.finish().unwrap();
}

fn record(file: &mut fs::File, payload: &[u8]) {
    let count = (payload.len() as u32).to_be_bytes();
    file.write_all(&count).unwrap();
    file.write_all(payload).unwrap();
    file.write_all(&count).unwrap();
}
fn text(buf: &mut Vec<u8>, value: &str, size: usize) {
    let start = buf.len();
    buf.extend_from_slice(value.as_bytes());
    buf.resize(start + size, b' ');
}
fn float(buf: &mut Vec<u8>, value: f32) {
    buf.extend_from_slice(&value.to_be_bytes());
}

fn met(
    path: &Path,
    gaussian: bool,
    model_level: bool,
    humidity: bool,
    time: &str,
    phase: f32,
    south_first: bool,
    zero_wind: bool,
    version: i32,
) {
    let (nx, ny) = (12, 8);
    let mut file = fs::File::create(path).unwrap();
    // Permuted levels make every column go through the native sort.
    for k in [4, 0, 7, 2, 5, 1, 6, 3] {
        let height = [0.0, 300.0, 1100.0, 2800.0, 5000.0, 7500.0, 10000.0, 14000.0][k];
        let level = if k == 0 {
            200100.0
        } else if model_level {
            k as f32
        } else {
            [
                200100.0, 97000.0, 90000.0, 75000.0, 60000.0, 45000.0, 30000.0, 16000.0,
            ][k]
        };
        for name in [
            "TT", "UU", "VV", "GHT", "RH", "SPECHUMD", "PRESSURE", "PSFC", "SOILHGT", "QC","QR","QI","QS","QG","QNC","QNI","QNR","QNWFA","QNIFA",
        ] {
            if name == "SPECHUMD" && !humidity {
                continue;
            }
            if name == "PRESSURE" && !model_level {
                continue;
            }
            record(&mut file, &version.to_be_bytes());
            let mut header = Vec::new();
            text(&mut header, time, 24);
            float(&mut header, 0.0);
            if version != 3 {
                text(&mut header, "synthetic", 32);
            }
            let field_name = match (name, version) {
                ("GHT", 4) => "HGT",
                ("PRESSURE", 3) => "PRES",
                _ => name,
            };
            text(&mut header, field_name, 9);
            text(&mut header, "test", 25);
            text(&mut header, "case-9 branch replay", 46);
            float(&mut header, level);
            let projection_code = if name == "PSFC" {
                [1, 3, 5][k % 3]
            } else if gaussian {
                4
            } else {
                0
            };
            for i in [nx as i32, ny as i32, projection_code] {
                header.extend_from_slice(&i.to_be_bytes());
            }
            record(&mut file, &header);
            let mut proj = Vec::new();
            if version != 3 {
                text(&mut proj, "SWCORNER", 8);
            }
            float(
                &mut proj,
                if gaussian {
                    gaussian_latitudes(ny)[0] * if south_first { -1.0 } else { 1.0 }
                } else {
                    -70.0
                },
            );
            float(&mut proj, -180.0);
            float(&mut proj, if gaussian { ny as f32 / 2.0 } else { 20.0 });
            float(&mut proj, 30.0);
            let extra = match projection_code {
                1 => 1,
                3 => 3,
                5 => 2,
                _ => 0,
            };
            for v in [0.0, 30.0, 60.0].into_iter().take(extra) {
                float(&mut proj, v);
            }
            if version == 5 {
                float(&mut proj, 6371.229);
            }
            record(&mut file, &proj);
            if version == 5 {
                record(&mut file, &0i32.to_be_bytes());
            }
            let mut values = Vec::new();
            for j in 0..ny {
                for i in 0..nx {
                    let perturb = (((i * 13 + j * 7 + k * 3) % 17) as f32 - 8.0) * 0.025 + phase;
                    let value = match name {
                        "TT" => 290.0 - height * 0.006 + perturb,
                        "UU" => {
                            if zero_wind || (i + j) % 7 == 0 {
                                0.0
                            } else {
                                11.0 - height * 0.0004 + perturb * 4.0
                            }
                        }
                        "VV" => {
                            if zero_wind {
                                -0.0
                            } else {
                                -3.0 + height * 0.0002 + perturb * 2.0
                            }
                        }
                        "GHT" => height + perturb * 8.0,
                        "RH" => 52.0 + perturb * 9.0,
                        "SPECHUMD" => {
                            if k == 7 {
                                -0.001
                            } else {
                                0.012 * (-height / 5500.0).exp() + perturb * 0.0001
                            }
                        }
                        "PRESSURE" => 97000.0 * (-height / 7800.0).exp() + perturb * 30.0,
                        "QC"|"QR"|"QI"|"QS"|"QG"=> 0.001*(1.+perturb),
                        "QNC"|"QNI"|"QNR"=>1000.*(1.+perturb),
                        "QNWFA"|"QNIFA"=>100000.*(1.+perturb),
                        _ => 101000.0,
                    };
                    float(&mut values, value);
                }
            }
            record(&mut file, &values);
        }
    }
}

fn run(command: &mut Command) -> std::process::Output {
    let output = command.output().expect("run artifact");
    assert!(
        output.status.success(),
        "{command:?}\n{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    output
}

fn file_hash(path: &Path) -> String {
    let mut input = fs::File::open(path).unwrap();
    let mut hash = Sha256::new();
    let mut buf = [0u8; 65536];
    loop {
        let n = input.read(&mut buf).unwrap();
        if n == 0 {
            break;
        }
        hash.update(&buf[..n]);
    }
    format!("{:x}", hash.finalize())
}

/// Build additional admitted smoke/dust input branches, without modifying any
/// published GFS file. The height records supply the original clock, levels and
/// projection; only the added scalar names and deterministic test values
/// differ. All intermediate-file processing stays in Rust.
/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_read_met.F:97-412,
/// mpas_init_atm_cases.F:5078-5080,5368-5392,7323-7325 at 1d6e370e.
#[test]
#[ignore = "creates owned whole-driver fixtures from pinned GFS inputs"]
fn whole_optional_aerosol_input_fixture() {
    use std::io::{BufReader, BufWriter};
    fn read_record(reader: &mut impl Read) -> Option<Vec<u8>> {
        let mut size = [0u8; 4];
        let n = reader.read(&mut size).unwrap();
        if n == 0 { return None; }
        reader.read_exact(&mut size[n..]).unwrap();
        let count = u32::from_be_bytes(size) as usize;
        let mut payload = vec![0; count];
        reader.read_exact(&mut payload).unwrap();
        let mut end = [0; 4];
        reader.read_exact(&mut end).unwrap();
        assert_eq!(size, end, "Fortran record markers differ");
        Some(payload)
    }
    fn write_record(writer: &mut impl Write, payload: &[u8]) {
        let marker = (payload.len() as u32).to_be_bytes();
        writer.write_all(&marker).unwrap();
        writer.write_all(payload).unwrap();
        writer.write_all(&marker).unwrap();
    }
    let plan_path = std::env::var("MPAS_OPTIONAL_MET_PLAN").unwrap();
    let plan: serde_json::Value = serde_json::from_slice(&fs::read(plan_path).unwrap()).unwrap();
    let mut outputs = Vec::new();
    for row in plan.as_array().unwrap() {
        let fields: Vec<&str> = row.get("fields").map(|v| v.as_array().unwrap().iter()
            .map(|v| v.as_str().unwrap()).collect()).unwrap_or_else(|| vec!["SMKF", "DSTF", "DSTC"]);
        assert!(!fields.is_empty());
        assert!(fields.iter().all(|v| ["SMKF", "DSTF", "DSTC", "QNWFA", "QNIFA"].contains(v)));
        let zero_fields: Vec<&str> = row.get("zero_fields").map(|v| v.as_array().unwrap().iter()
            .map(|v| v.as_str().unwrap()).collect()).unwrap_or_default();
        assert!(zero_fields.iter().all(|v| fields.contains(v)));
        let source = Path::new(row["source"].as_str().unwrap());
        let destination = Path::new(row["output"].as_str().unwrap());
        let source_sha = file_hash(source);
        assert_eq!(source_sha, row["sha256"].as_str().unwrap());
        fs::create_dir_all(destination.parent().unwrap()).unwrap();
        let mut output = BufWriter::new(fs::OpenOptions::new().write(true).create_new(true).open(destination).unwrap());
        std::io::copy(&mut fs::File::open(source).unwrap(), &mut output).unwrap();
        let mut input = BufReader::new(fs::File::open(source).unwrap());
        let mut levels = 0usize;
        while let Some(version_raw) = read_record(&mut input) {
            let version = i32::from_be_bytes(version_raw.as_slice().try_into().unwrap());
            assert!([3, 4, 5].contains(&version));
            let header = read_record(&mut input).unwrap();
            let projection = read_record(&mut input).unwrap();
            let wind = if version == 5 { read_record(&mut input) } else { None };
            let data = read_record(&mut input).unwrap();
            let field_offset = if version == 3 { 28 } else { 60 };
            let field = std::str::from_utf8(&header[field_offset..field_offset+9]).unwrap().trim();
            assert!(!fields.contains(&field), "fixture input already carries requested fields");
            if !["GHT", "HGT"].contains(&field) { continue; }
            assert_eq!(data.len() % 4, 0);
            for (species, name) in fields.iter().copied().enumerate() {
                let mut added_header = header.clone();
                added_header[field_offset..field_offset+9].fill(b' ');
                added_header[field_offset..field_offset+name.len()].copy_from_slice(name.as_bytes());
                added_header[field_offset+9..field_offset+34].fill(b' ');
                let units = if ["QNWFA", "QNIFA"].contains(&name) { "kg-1" } else { "kg kg-1" };
                added_header[field_offset+9..field_offset+9+units.len()].copy_from_slice(units.as_bytes());
                let mut values = Vec::with_capacity(data.len());
                for cell in 0..data.len()/4 {
                    // Negative and zero inputs exercise the native max(0,q),
                    // while positive values exercise interpolation and x1e9.
                    let value = if zero_fields.contains(&name) { 0.0 } else {
                        ((cell % 17) as f32 - 4.0) * 1.0e-11
                        * (species + 1) as f32 * (1.0 + levels as f32 * 0.03125)
                    };
                    values.extend_from_slice(&value.to_be_bytes());
                }
                for payload in [&version_raw, &added_header, &projection] { write_record(&mut output, payload); }
                if let Some(wind) = &wind { write_record(&mut output, wind); }
                write_record(&mut output, &values);
            }
            levels += 1;
        }
        assert!(levels > 1, "no complete height level table in fixture source");
        output.flush().unwrap();
        assert_eq!(file_hash(source), source_sha, "published input changed");
        outputs.push(serde_json::json!({"source": source, "source_sha256":source_sha,
            "output":destination,"output_sha256":file_hash(destination),
            "added_fields":fields,"zero_fields":zero_fields,"levels_per_field":levels,
            "scope":"synthetic optional scalar slabs appended to original GFS; additional branch proof only"}));
    }
    fs::write(std::env::var("MPAS_OPTIONAL_MET_RECEIPT").unwrap(), serde_json::to_string_pretty(&serde_json::json!({
        "schema":"r10-optional-met-fixture/v1","producer_sha256":file_hash(&std::env::current_exe().unwrap()),
        "source_revision":env!("GPUWM_BRIDGE_SOURCE_REV"),"outputs":outputs})).unwrap()).unwrap();
}

#[test]
#[ignore = "requires strict pinned Fortran oracle and lane-owned replay directory"]
fn entire_rrfs_case9_replay_is_word_identical_to_fortran() {
    let oracle = std::env::var("MPAS_LBC_ORACLE").expect("MPAS_LBC_ORACLE");
    let root = std::path::PathBuf::from(
        std::env::var("MPAS_LBC_REPLAY_DIR").expect("MPAS_LBC_REPLAY_DIR"),
    );
    fs::create_dir_all(&root).unwrap();
    let provenance = serde_json::json!({
        "rust_binary_sha256": format!("{:x}", Sha256::digest(fs::read(std::env::current_exe().unwrap()).unwrap())),
        "oracle_sha256": format!("{:x}", Sha256::digest(fs::read(&oracle).unwrap())),
        "source_revision": env!("GPUWM_BRIDGE_SOURCE_REV"),
        "optimized": !cfg!(debug_assertions),
        "host": String::from_utf8(run(Command::new("hostname").arg("-f")).stdout).unwrap().trim(),
    });
    let mut receipts = Vec::new();
    for (
        index,
        (
            gaussian,
            model_level,
            humidity,
            use_sh,
            order,
            extrap,
            above_top,
            south_first,
            zero_wind,
            version,
        ),
    ) in [
        (
            false,
            false,
            true,
            "yes",
            3,
            "lapse-rate",
            false,
            false,
            false,
            5,
        ),
        (
            false, true, true, "yes", 2, "linear", false, false, false, 5,
        ),
        (
            false, false, false, "yes", 3, "constant", false, false, false, 5,
        ),
        (
            false, true, true, "no", 3, "constant", true, false, false, 5,
        ),
        (
            true,
            false,
            true,
            "yes",
            3,
            "lapse-rate",
            false,
            false,
            false,
            5,
        ),
        (true, true, true, "no", 2, "linear", false, false, false, 5),
        (
            true, false, false, "yes", 3, "constant", true, false, false, 5,
        ),
        (true, true, true, "yes", 3, "linear", false, true, false, 5),
        (
            true, false, true, "yes", 3, "constant", false, false, true, 5,
        ),
        (
            false, true, true, "yes", 3, "linear", false, false, false, 5,
        ),
        (
            false,
            false,
            false,
            "no",
            2,
            "lapse-rate",
            false,
            false,
            false,
            5,
        ),
    ]
    .into_iter()
    .enumerate()
    {
        let dir = root.join(format!("branch-{index}"));
        fs::create_dir_all(dir.join("met")).unwrap();
        let grid_path = dir.join("grid.nc");
        grid(&grid_path, above_top);
        for (time, phase) in [(START, 0.0), (STOP, 0.1375)] {
            met(
                &dir.join("met").join(format!("MET:{}", &time[..13])),
                gaussian,
                model_level,
                humidity,
                time,
                phase,
                south_first,
                zero_wind,
                version,
            );
        }
        use rw_mpas::rrfs_lbc::{build_lbc,LbcConfig,BoundaryInterval};
        let extrap_mode=match extrap {"constant"=>rw_mpas::init::vinterp::Extrap::Constant,"linear"=>rw_mpas::init::vinterp::Extrap::Linear,_=>rw_mpas::init::vinterp::Extrap::LapseRate};
        let config=LbcConfig {grid_path:grid_path.clone(),out_dir:dir.join("ours"),start_time:START.into(),stop_time:STOP.into(),intervals:[START,STOP].into_iter().map(|t|BoundaryInterval{valid_time:t.into(),met_path:dir.join("met").join(format!("MET:{}",&t[..13]))}).collect(),n_fg_levels:8,extrap_airtemp:extrap_mode,use_spechumd:use_sh=="yes",theta_adv_order:order,coef_3rd_order:0.25,fg_interval_seconds:10800,oned_underflow:rw_mpas::init::hinterp::Underflow::Preserve,attr_overrides:Default::default(),provenance:"RRFS standalone replay".into(),hydrometeors_rrfs:true,hydrometeors_gfs:false,tempo_rap:true,aerosolaware:false,smoke:false,dust:false,gocart_path:None,gocart_levels:30,absent_numbers:rw_mpas::rrfs_init::scalars::AbsentNumbers::Zero};
        build_lbc(&config).unwrap();
        for time in [START, STOP] {
            let native = dir.join(format!("native-{}.nc", time.replace(':', ".")));
            let mut reference = Command::new(&oracle);
            reference
                .arg(&grid_path)
                .arg(&native)
                .arg(time)
                .arg(dir.join("met/MET"))
                .args([extrap, use_sh, &order.to_string()]);
            run(&mut reference);
            let ours = dir
                .join("ours")
                .join(rw_mpas::rrfs_lbc::emit::lbc_file_name(time));
            let comparison = compare_lbc(&ours, &native).unwrap();
            assert_eq!(
                comparison.variables.len(),
                17,
                "complete numerical LBC field set"
            );
            let receipt = dir.join(format!("comparison-{}.json", time.replace(':', ".")));
            fs::write(&receipt, serde_json::to_string_pretty(&comparison).unwrap()).unwrap();
            receipts.push(serde_json::json!({"branch":index,"gaussian":gaussian,"model_level":model_level,
                "specific_humidity_present":humidity,"use_spechumd":use_sh,"theta_adv_order":order,
                "extrapolation":extrap,"above_top":above_top,"south_first":south_first,"zero_wind":zero_wind,"intermediate_version":version,"time":time,"comparison":receipt,
                "payload_byte_exact":comparison.payload_byte_exact}));
            fs::write(root.join("replay.json"),serde_json::to_string_pretty(&serde_json::json!({
                "tier":"synthetic full case-9 numerical replay; not r0 native deck proof", "provenance":provenance,"cases":receipts})).unwrap()).unwrap();
            assert!(comparison.payload_byte_exact, "branch {index}: {receipt:?}");
            fs::remove_file(native).unwrap();
            fs::remove_file(ours).unwrap();
        }
        fs::remove_file(grid_path).unwrap();
        for entry in fs::read_dir(dir.join("met")).unwrap() {
            fs::remove_file(entry.unwrap().path()).unwrap();
        }
    }
}
