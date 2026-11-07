//! The front door itself: argument parsing, flag filtering, and the
//! inventory document, exercised through the real binary.
//!
//! `gpuwm.mapped_source` drives `rw_netcdf` as a subprocess, so the
//! argv contract IS the product interface: a flag that stops being
//! filtered out of the positionals, or a switch whose sense inverts,
//! changes every NetCDF initial condition read through it without
//! failing a single in-process test.  These tests run the executable
//! the way Python does and read what it wrote.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn binary() -> &'static str {
    env!("CARGO_BIN_EXE_rw_netcdf")
}

fn run(args: &[&str]) -> Output {
    Command::new(binary())
        .args(args)
        .output()
        .expect("spawn rw_netcdf")
}

fn stdout(output: &Output) -> String {
    String::from_utf8(output.stdout.clone()).expect("stdout is UTF-8")
}

fn stderr(output: &Output) -> String {
    String::from_utf8(output.stderr.clone()).expect("stderr is UTF-8")
}

fn fixture(name: &str) -> String {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(name)
        .to_string_lossy()
        .into_owned()
}

/// A scratch directory unique to one test, wiped at entry.
fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir()
        .join(format!("rw-netcdf-cli-{}-{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create scratch dir");
    dir
}

/// The classic packed fixture, built by the workspace's own writer --
/// same layout as the unit suite's: stored shorts
/// [-32768 (_FillValue), -32767 (missing_value), 4, 6] with
/// scale_factor 0.5 and add_offset 100.
fn write_packed_classic(path: &Path) {
    use netcdf_writer::{AttrValue, NcFormat, NcType, NcWriter, Schema, VarData};
    let mut schema = Schema::new(NcFormat::Classic);
    let x = schema.def_dim("x", 4, false).expect("dim");
    schema
        .put_global_attr("title", AttrValue::Text("cli fixture".into()))
        .expect("gattr");
    let packed = schema.def_var("packed", NcType::Short, &[x]).expect("var");
    schema
        .put_var_attr(packed, "scale_factor", AttrValue::Doubles(vec![0.5]))
        .expect("attr");
    schema
        .put_var_attr(packed, "add_offset", AttrValue::Doubles(vec![100.0]))
        .expect("attr");
    schema
        .put_var_attr(packed, "_FillValue", AttrValue::Shorts(vec![-32768]))
        .expect("attr");
    schema
        .put_var_attr(packed, "missing_value", AttrValue::Shorts(vec![-32767]))
        .expect("attr");
    let mut writer = NcWriter::create(path, schema).expect("create");
    writer
        .write_var(packed, VarData::I16(&[-32768, -32767, 4, 6]))
        .expect("write");
    writer.finish().expect("finish");
}

fn read_f64_plane(path: &Path) -> Vec<f64> {
    let bytes = std::fs::read(path).expect("read plane");
    bytes
        .chunks_exact(8)
        .map(|chunk| f64::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

#[test]
fn no_arguments_is_usage_on_stderr_and_exit_2() {
    let output = run(&[]);
    assert_eq!(output.status.code(), Some(2));
    assert!(stderr(&output).contains("usage:"), "{output:?}");
}

#[test]
fn abi_prints_both_schema_markers() {
    let output = run(&["--abi"]);
    assert_eq!(output.status.code(), Some(0));
    let text = stdout(&output);
    assert!(text.contains("gpuwm-rw-netcdf-inventory-v1"), "{text}");
    assert!(text.contains("gpuwm-rw-netcdf-dump-v1"), "{text}");
}

#[test]
fn help_prints_usage_on_stdout() {
    let output = run(&["--help"]);
    assert_eq!(output.status.code(), Some(0));
    assert!(stdout(&output).contains("usage:"), "{output:?}");
}

#[test]
fn unknown_subcommands_are_refused_by_name() {
    let output = run(&["decode"]);
    assert_eq!(output.status.code(), Some(2));
    let text = stderr(&output);
    assert!(text.contains("unknown subcommand"), "{text}");
    assert!(text.contains("decode"), "{text}");
}

#[test]
fn inventory_takes_exactly_one_file() {
    for arguments in [vec!["inventory"], vec!["inventory", "a.nc", "b.nc"]] {
        let output = run(&arguments);
        assert_eq!(output.status.code(), Some(2), "{arguments:?}");
        assert!(
            stderr(&output).contains("exactly one FILE"),
            "{arguments:?}: {output:?}"
        );
    }
}

#[test]
fn inventory_reports_the_whole_document() {
    let dir = scratch("inventory");
    let source = dir.join("packed.nc");
    write_packed_classic(&source);
    let output = run(&["inventory", source.to_str().expect("path")]);
    assert_eq!(output.status.code(), Some(0), "{output:?}");
    let document: serde_json::Value =
        serde_json::from_str(&stdout(&output)).expect("inventory JSON");

    assert_eq!(document["schema"], "gpuwm-rw-netcdf-inventory-v1");
    assert_eq!(document["metadata"]["mode"], "strict");
    assert_eq!(
        document["dimensions"],
        serde_json::json!([{"name": "x", "len": 4, "unlimited": false}])
    );
    assert_eq!(document["global_attributes"]["title"], "cli fixture");
    let variables = document["variables"].as_array().expect("variables");
    assert_eq!(variables.len(), 1);
    let packed = &variables[0];
    assert_eq!(packed["name"], "packed");
    assert_eq!(packed["dimensions"], serde_json::json!(["x"]));
    assert_eq!(packed["shape"], serde_json::json!([4]));
    assert_eq!(packed["attributes"]["scale_factor"], 0.5);
    assert_eq!(packed["attributes"]["add_offset"], 100.0);
    assert_eq!(packed["attributes"]["_FillValue"], -32768.0);
    let _ = std::fs::remove_dir_all(&dir);
}

/// The provenance travels through the CLI too: a NetCDF-4 file that
/// strict reconstruction refuses arrives marked size-inferred, with the
/// strict error and the length-collision confession beside it.
#[test]
fn inventory_confesses_size_inferred_metadata() {
    let output = run(&["inventory", &fixture("mixed.nc4")]);
    assert_eq!(output.status.code(), Some(0), "{output:?}");
    let document: serde_json::Value =
        serde_json::from_str(&stdout(&output)).expect("inventory JSON");
    assert_eq!(document["metadata"]["mode"], "size-inferred");
    assert!(
        document["metadata"]["strict_error"]
            .as_str()
            .expect("strict_error")
            .contains("DIMENSION_LIST"),
        "{document}"
    );
    assert_eq!(document["metadata"]["dimension_lengths_ambiguous"], true);
}

/// A CDF-2 history shaped like a wrfout: an unlimited `Time`, a record
/// character `Times`, a record field and a fixed field, two records.
fn write_record_history(path: &Path) {
    use netcdf_writer::{NcFormat, NcType, NcWriter, Schema, VarData};
    let mut schema = Schema::new(NcFormat::Offset64);
    let time = schema.def_dim("Time", 0, true).expect("dim");
    let chars = schema.def_dim("DateStrLen", 19, false).expect("dim");
    let y = schema.def_dim("south_north", 3, false).expect("dim");
    let x = schema.def_dim("west_east", 4, false).expect("dim");
    let times = schema
        .def_var("Times", NcType::Char, &[time, chars])
        .expect("var");
    let hgt = schema.def_var("HGT", NcType::Float, &[y, x]).expect("var");
    let t = schema
        .def_var("T", NcType::Float, &[time, y, x])
        .expect("var");
    let mut writer = NcWriter::create(path, schema).expect("create");
    writer
        .write_var(hgt, VarData::F32(&[250.0; 12]))
        .expect("write HGT");
    for (record, stamp) in ["2026-09-26_00:00:00", "2026-09-26_01:00:00"]
        .iter()
        .enumerate()
    {
        writer
            .write_record(record as u64, times, VarData::Char(stamp.as_bytes()))
            .expect("write Times");
        writer
            .write_record(record as u64, t, VarData::F32(&[300.0 + record as f32; 12]))
            .expect("write T");
    }
    writer.finish().expect("finish");
}

#[test]
fn stored_record_preserves_words_and_rejects_unproven_axes() {
    use netcdf_writer::{NcFormat, NcType, NcWriter, Schema, VarData};
    let dir = scratch("stored-record");
    let path = dir.join("surface.nc");
    let mut schema = Schema::new(NcFormat::Offset64);
    let time = schema.def_dim("Time", 0, true).unwrap();
    let x = schema.def_dim("west_east", 4, false).unwrap();
    let chars = schema.def_dim("DateStrLen", 19, false).unwrap();
    let values = schema.def_var("Q2_NDG_OLD", NcType::Float, &[time, x]).unwrap();
    let times = schema.def_var("Times", NcType::Char, &[time, chars]).unwrap();
    let fixed = schema.def_var("fixed", NcType::Float, &[x]).unwrap();
    let mut writer = NcWriter::create(&path, schema).unwrap();
    writer.write_var(fixed, VarData::F32(&[1.; 4])).unwrap();
    let words = [0x80000000u32, 0x7fc01234, 1, 0x3f800001];
    let floats = words.map(f32::from_bits);
    for record in 0..3 {
        writer.write_record(record, values, VarData::F32(if record == 1 { &floats } else { &[17.; 4] })).unwrap();
        writer.write_record(record, times, VarData::Char(if record == 1 { b"2026-11-01_01:00:00" } else { b"2026-11-01_00:00:00" })).unwrap();
    }
    writer.finish().unwrap();
    let out = dir.join("out");
    let result = run(&["stored-words-record", "1", path.to_str().unwrap(), out.to_str().unwrap(), "Q2_NDG_OLD", "Times"]);
    assert!(result.status.success(), "{}", stderr(&result));
    let bytes = std::fs::read(out.join("0.bin")).unwrap();
    assert_eq!(bytes, words.into_iter().flat_map(u32::to_le_bytes).collect::<Vec<_>>());
    assert_eq!(std::fs::read(out.join("1.bin")).unwrap(), b"2026-11-01_01:00:00");
    let metadata: serde_json::Value = serde_json::from_slice(&std::fs::read(out.join("metadata.json")).unwrap()).unwrap();
    assert_eq!(metadata["record"], 1);
    assert_eq!(metadata["variables"][0]["shape"], serde_json::json!([1, 4]));
    for (record, variable) in [("3", "Q2_NDG_OLD"), ("-1", "Q2_NDG_OLD"), ("0", "fixed")] {
        let result = run(&["stored-words-record", record, path.to_str().unwrap(), out.to_str().unwrap(), variable]);
        assert!(!result.status.success(), "record={record}, variable={variable}");
    }
    std::fs::remove_dir_all(dir).unwrap();
}

/// A classic file cut short keeps its header and still opens; the C
/// library then reads the missing values as zeros.  The inventory is
/// where the cut is caught, whatever part of the data it took.
#[test]
fn inventory_refuses_a_classic_file_that_ends_before_its_data() {
    let dir = scratch("truncated");
    let whole = dir.join("wrfout_d01_2026-09-26_00_00_00");
    write_record_history(&whole);
    let output = run(&["inventory", whole.to_str().expect("path")]);
    assert_eq!(output.status.code(), Some(0), "{output:?}");
    let document: serde_json::Value =
        serde_json::from_str(&stdout(&output)).expect("inventory JSON");
    assert_eq!(document["extent_checked"], true);
    assert_eq!(document["format"], "Offset64");

    let bytes = std::fs::read(&whole).expect("read");
    // One byte short, one record short, and cut inside the fixed field
    // the records follow.  The header is intact in every case.
    let record_bytes = 19 + 12 * 4 + 1; // Times + T, Times padded to 20
    for keep in [bytes.len() - 1, bytes.len() - record_bytes, bytes.len() - 2 * record_bytes - 8] {
        let cut = dir.join(format!("cut-{keep}"));
        std::fs::write(&cut, &bytes[..keep]).expect("write cut");
        let output = run(&["inventory", cut.to_str().expect("path")]);
        assert_eq!(output.status.code(), Some(2), "keep {keep}: {output:?}");
        let text = stderr(&output);
        assert!(text.contains("the file ends before its data does"), "keep {keep}: {text}");
        assert!(text.contains("cut short"), "keep {keep}: {text}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// A NetCDF-4 file's end is checked by its own library at open; the
/// inventory says it did not check, rather than claiming it did.
#[test]
fn inventory_says_a_netcdf4_file_was_not_extent_checked() {
    let output = run(&["inventory", &fixture("times.nc4")]);
    assert_eq!(output.status.code(), Some(0), "{output:?}");
    let document: serde_json::Value =
        serde_json::from_str(&stdout(&output)).expect("inventory JSON");
    assert_eq!(document["extent_checked"], false);
}

#[test]
fn inventory_refuses_a_missing_file_with_exit_2() {
    let output = run(&["inventory", "no-such-file.nc"]);
    assert_eq!(output.status.code(), Some(2));
    assert!(stderr(&output).contains("cannot open"), "{output:?}");
}

#[test]
fn dump_requires_file_output_dir_and_a_variable() {
    let dir = scratch("dump-argc");
    let source = dir.join("packed.nc");
    write_packed_classic(&source);
    let out = dir.join("out");
    let arguments = [
        "dump",
        source.to_str().expect("path"),
        out.to_str().expect("path"),
    ];
    let output = run(&arguments);
    assert_eq!(output.status.code(), Some(2), "{output:?}");
    assert!(
        stderr(&output).contains("at least one VARIABLE"),
        "{output:?}"
    );
    assert!(
        !out.join("metadata.json").exists(),
        "a refused dump must write nothing"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_bare_dump_masks_and_scales_by_default() {
    let dir = scratch("dump-default");
    let source = dir.join("packed.nc");
    write_packed_classic(&source);
    let out = dir.join("out");
    let output = run(&[
        "dump",
        source.to_str().expect("path"),
        out.to_str().expect("path"),
        "packed",
    ]);
    assert_eq!(output.status.code(), Some(0), "{output:?}");
    let values = read_f64_plane(&out.join("0000.f64"));
    assert!(values[0].is_nan() && values[1].is_nan(), "{values:?}");
    assert_eq!(&values[2..], &[102.0, 103.0]);
    let _ = std::fs::remove_dir_all(&dir);
}

/// `--raw` is netCDF4's set_auto_maskandscale(False): the stored bytes
/// survive untouched.  The flag must also be FILTERED out of the
/// positionals wherever it appears.
#[test]
fn dump_raw_preserves_stored_sentinels() {
    let dir = scratch("dump-raw");
    let source = dir.join("packed.nc");
    write_packed_classic(&source);
    let out = dir.join("out");
    let output = run(&[
        "dump",
        "--raw",
        source.to_str().expect("path"),
        out.to_str().expect("path"),
        "packed",
    ]);
    assert_eq!(output.status.code(), Some(0), "{output:?}");
    assert_eq!(
        read_f64_plane(&out.join("0000.f64")),
        &[-32768.0, -32767.0, 4.0, 6.0]
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// `--no-mask` is set_auto_mask(False): sentinels survive but arrive
/// SCALED like every other element, exactly as netCDF4-python does it.
#[test]
fn dump_no_mask_scales_the_surviving_sentinels() {
    let dir = scratch("dump-nomask");
    let source = dir.join("packed.nc");
    write_packed_classic(&source);
    let out = dir.join("out");
    let output = run(&[
        "dump",
        source.to_str().expect("path"),
        out.to_str().expect("path"),
        "packed",
        "--no-mask",
    ]);
    assert_eq!(output.status.code(), Some(0), "{output:?}");
    assert_eq!(
        read_f64_plane(&out.join("0000.f64")),
        &[-16284.0, -16283.5, 102.0, 103.0]
    );
    let _ = std::fs::remove_dir_all(&dir);
}


#[test]
fn character_records_match_independent_c_library_fixtures() {
    for filename in ["times.nc1", "times.nc2", "times.nc5", "times.nc4"] {
        let dir = scratch(filename);
        let output = run(&["dump", &fixture(filename), dir.to_str().unwrap(), "Times"]);
        assert!(output.status.success(), "{}: {}", filename, stderr(&output));
        let mut expected = b"2021-12-30_17:00:002021-12-30_18:00:00a b\0c".to_vec();
        expected.resize(57, 0);
        assert_eq!(std::fs::read(dir.join("0000.chars")).unwrap(), expected);
        let metadata: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("metadata.json")).unwrap()).unwrap();
        assert_eq!(metadata["variables"][0]["shape"], serde_json::json!([3, 19]));
        assert_eq!(metadata["variables"][0]["dtype"], "|S1");
        std::fs::remove_dir_all(dir).unwrap();
    }
}


#[test]
fn explicit_units_follow_cf_unpacking_and_preserve_default_bytes() {
    let dir = scratch("units");
    let input = dir.join("packed.nc");
    write_packed_classic(&input);
    let normal = dir.join("normal");
    let identity = dir.join("identity");
    let converted = dir.join("converted");
    for (out, flags) in [(&normal, vec![]), (&identity, vec!["--unit-scale=1", "--unit-offset=0"]),
                         (&converted, vec!["--unit-scale=100", "--unit-offset=-5"])] {
        let mut args = vec!["dump"];
        args.extend(flags);
        args.extend([input.to_str().unwrap(), out.to_str().unwrap(), "packed"]);
        let output = run(&args);
        assert!(output.status.success(), "{}", stderr(&output));
    }
    assert_eq!(std::fs::read(normal.join("0000.f64")).unwrap(),
               std::fs::read(identity.join("0000.f64")).unwrap());
    assert_eq!(std::fs::read(normal.join("metadata.json")).unwrap(),
               std::fs::read(identity.join("metadata.json")).unwrap());
    let data = read_f64_plane(&converted.join("0000.f64"));
    assert!(data[0].is_nan() && data[1].is_nan());
    assert_eq!(&data[2..], &[10195.0, 10295.0]);
    let metadata: serde_json::Value = serde_json::from_slice(
        &std::fs::read(converted.join("metadata.json")).unwrap()).unwrap();
    assert_eq!(metadata["variables"][0]["unit_transform"], serde_json::json!([100.0, -5.0]));
}

#[test]
fn invalid_or_overflowing_unit_transforms_are_refused() {
    let dir = scratch("invalid-units");
    let input = dir.join("packed.nc");
    write_packed_classic(&input);
    for flag in ["--unit-scale=nan", "--unit-scale=0", "--unit-offset=inf", "--unit-scale=oops", "--unit-scale=1e308"] {
        let output = run(&["dump", flag, input.to_str().unwrap(), dir.to_str().unwrap(), "packed"]);
        assert_eq!(output.status.code(), Some(2), "{flag}");
        assert!(stderr(&output).contains("unit"), "{}", stderr(&output));
    }
}

// ---------------------------------------------------------------------------
// recover-wrf-soil, against real files this test writes
// ---------------------------------------------------------------------------
//
// Reported from the field: `--soil-source` pointed at the WPS directory
// that produced the run, and the reader refused with "source moisture
// shape does not match the target domain or authority layers" for a
// met_em set WPS had produced for that same domain.  The sentence
// covered three different conditions and printed none of their numbers.
// These build the pair and drive the binary.

/// The soil authority a producing table yields, with `declared` layer
/// bounds in metres.
fn soil_authority(declared: &[[f64; 2]]) -> String {
    let bounds = declared
        .iter()
        .map(|[top, bottom]| format!("[{top},{bottom}]"))
        .collect::<Vec<_>>()
        .join(",");
    format!(
        "{{\"schema\":\"gpuwm-wrf-soil-authority-v1\",\
         \"source_variable\":\"SOILM\",\"source_depth_variable\":\"SOIL_LEVELS\",\
         \"source_quantity\":\"layer_water_mass\",\"source_units\":\"kg m-2\",\
         \"source_depths_from_metgrid\":true,\"source_layer_bounds_m\":[{bounds}]}}"
    )
}

/// WRF REAL's single-precision soil interpolation, in its own order.
/// The wrfinput this test writes has to BE that, or the reader refuses
/// the reconstruction before it reaches anything under test.
fn wrf_soil_interpolation(values: &[f64], depths_cm: &[f64], centre: f64) -> f32 {
    let depths: Vec<f32> = depths_cm.iter().map(|cm| *cm as f32 / 100.0).collect();
    let target = centre as f32;
    let pair = depths
        .windows(2)
        .position(|z| target >= z[0] && target <= z[1])
        .expect("target centre inside the source depths");
    ((values[pair] as f32 * (depths[pair + 1] - target))
        + (values[pair + 1] as f32 * (target - depths[pair])))
        / (depths[pair + 1] - depths[pair])
}

const CLOCK: &str = "2026-09-12_00:00:00";
const CENTRES: [f64; 4] = [0.05, 0.25, 0.7, 1.5];

/// A cold REAL_EM initial state carrying the source quantity WRF
/// interpolated without converting it, on a `ny` by `nx` mass grid.
fn write_wrfinput(path: &Path, ny: usize, nx: usize, layers: &[f64], depths_cm: &[f64]) {
    use netcdf_writer::{AttrValue, NcFormat, NcType, NcWriter, Schema, VarData};
    let cells = ny * nx;
    let mut schema = Schema::new(NcFormat::Classic);
    let time = schema.def_dim("Time", 1, false).expect("dim");
    let chars = schema.def_dim("DateStrLen", 19, false).expect("dim");
    let south_north = schema.def_dim("south_north", ny, false).expect("dim");
    let west_east = schema.def_dim("west_east", nx, false).expect("dim");
    let soil = schema
        .def_dim("soil_layers_stag", CENTRES.len(), false)
        .expect("dim");
    for (name, value) in [
        ("TITLE", " OUTPUT FROM REAL_EM V4.6.1 PREPROCESSOR"),
        ("START_DATE", CLOCK),
        ("SIMULATION_START_DATE", CLOCK),
    ] {
        schema
            .put_global_attr(name, AttrValue::Text(value.into()))
            .expect("gattr");
    }
    schema
        .put_global_attr("SF_SURFACE_PHYSICS", AttrValue::Ints(vec![2]))
        .expect("gattr");
    let times = schema
        .def_var("Times", NcType::Char, &[time, chars])
        .expect("var");
    let smois = schema
        .def_var("SMOIS", NcType::Float, &[time, soil, south_north, west_east])
        .expect("var");
    let sh2o = schema
        .def_var("SH2O", NcType::Float, &[time, soil, south_north, west_east])
        .expect("var");
    let landmask = schema
        .def_var("LANDMASK", NcType::Float, &[time, south_north, west_east])
        .expect("var");
    let latitude = schema
        .def_var("XLAT", NcType::Float, &[time, south_north, west_east])
        .expect("var");
    let longitude = schema
        .def_var("XLONG", NcType::Float, &[time, south_north, west_east])
        .expect("var");
    let centres = schema.def_var("ZS", NcType::Float, &[time, soil]).expect("var");
    schema
        .put_var_attr(centres, "units", AttrValue::Text("m".into()))
        .expect("attr");
    let mut moisture = Vec::new();
    for centre in CENTRES {
        let value = wrf_soil_interpolation(layers, depths_cm, centre);
        moisture.extend(std::iter::repeat(value).take(cells));
    }
    let mut writer = NcWriter::create(path, schema).expect("create");
    writer
        .write_var(times, VarData::Char(CLOCK.as_bytes()))
        .expect("write");
    writer
        .write_var(smois, VarData::F32(&moisture))
        .expect("write");
    writer
        .write_var(sh2o, VarData::F32(&vec![0.0_f32; CENTRES.len() * cells]))
        .expect("write");
    writer
        .write_var(landmask, VarData::F32(&vec![1.0_f32; cells]))
        .expect("write");
    writer
        .write_var(latitude, VarData::F32(&grid(ny, nx, ny, nx, 30.0)))
        .expect("write");
    writer
        .write_var(longitude, VarData::F32(&grid(ny, nx, ny, nx, -90.0)))
        .expect("write");
    writer
        .write_var(centres, VarData::F32(&CENTRES.map(|z| z as f32)))
        .expect("write");
    writer.finish().expect("finish");
}

/// A coordinate plane on a `sy` by `sx` extent whose leading `ny` by
/// `nx` block is the same domain whatever the extent is.
fn grid(sy: usize, sx: usize, ny: usize, nx: usize, origin: f32) -> Vec<f32> {
    let _ = (ny, nx);
    (0..sy * sx)
        .map(|index| origin + (index / sx) as f32 * 0.1 + (index % sx) as f32 * 0.01)
        .collect()
}

/// The metgrid file under WRF's explicit SOIL_LEVELS contract, on a
/// `sy` by `sx` extent whose leading `ny` by `nx` block is the domain.
fn write_met_em(
    path: &Path,
    (ny, nx): (usize, usize),
    (sy, sx): (usize, usize),
    layers: &[f64],
    depths_cm: &[f64],
    stagger_names: bool,
) {
    use netcdf_writer::{AttrValue, NcFormat, NcType, NcWriter, Schema, VarData};
    let mut schema = Schema::new(NcFormat::Classic);
    let time = schema.def_dim("Time", 1, false).expect("dim");
    let chars = schema.def_dim("DateStrLen", 19, false).expect("dim");
    let (y_name, x_name) = if stagger_names {
        ("south_north_stag", "west_east_stag")
    } else {
        ("south_north", "west_east")
    };
    let south_north = schema.def_dim(y_name, sy, false).expect("dim");
    let west_east = schema.def_dim(x_name, sx, false).expect("dim");
    let levels = schema
        .def_dim("num_soilm_levels", layers.len(), false)
        .expect("dim");
    schema
        .put_global_attr("FLAG_SOIL_LEVELS", AttrValue::Ints(vec![1]))
        .expect("gattr");
    let times = schema
        .def_var("Times", NcType::Char, &[time, chars])
        .expect("var");
    let water = schema
        .def_var("SOILM", NcType::Float, &[time, levels, south_north, west_east])
        .expect("var");
    schema
        .put_var_attr(water, "units", AttrValue::Text("kg m-2".into()))
        .expect("attr");
    let axis = schema
        .def_var(
            "SOIL_LEVELS",
            NcType::Float,
            &[time, levels, south_north, west_east],
        )
        .expect("var");
    schema
        .put_var_attr(axis, "units", AttrValue::Text("cm".into()))
        .expect("attr");
    let latitude = schema
        .def_var("XLAT_M", NcType::Float, &[time, south_north, west_east])
        .expect("var");
    let longitude = schema
        .def_var("XLONG_M", NcType::Float, &[time, south_north, west_east])
        .expect("var");
    let plane = |per_layer: &[f64]| -> Vec<f32> {
        per_layer
            .iter()
            .flat_map(|value| std::iter::repeat(*value as f32).take(sy * sx))
            .collect()
    };
    let mut writer = NcWriter::create(path, schema).expect("create");
    writer
        .write_var(times, VarData::Char(CLOCK.as_bytes()))
        .expect("write");
    writer
        .write_var(water, VarData::F32(&plane(layers)))
        .expect("write");
    writer
        .write_var(axis, VarData::F32(&plane(depths_cm)))
        .expect("write");
    writer
        .write_var(latitude, VarData::F32(&grid(sy, sx, ny, nx, 30.0)))
        .expect("write");
    writer
        .write_var(longitude, VarData::F32(&grid(sy, sx, ny, nx, -90.0)))
        .expect("write");
    writer.finish().expect("finish");
}

/// Six stacked layers and the water mass that makes each one a round
/// volume fraction on its own declared thickness.
const STACKED_CM: [f64; 6] = [1.0, 4.0, 30.0, 100.0, 160.0, 300.0];
const STACKED_KG: [f64; 6] = [1.0, 6.0, 60.0, 160.0, 270.0, 700.0];
const STACKED_VOLUME: [f64; 6] = [0.10, 0.20, 0.30, 0.40, 0.45, 0.50];

/// The nine layers a producing table can declare for that source.
const DECLARED: [[f64; 2]; 9] = [
    [0.0, 0.01],
    [0.01, 0.04],
    [0.04, 0.1],
    [0.1, 0.3],
    [0.3, 0.6],
    [0.6, 1.0],
    [1.0, 1.6],
    [1.6, 3.0],
    [3.0, 10.0],
];

fn soil_case(tag: &str, extent: (usize, usize), stagger_names: bool,
             declared: &[[f64; 2]], stacked_cm: &[f64], stacked_kg: &[f64])
             -> (PathBuf, Output) {
    let dir = scratch(tag);
    let wrfinput = dir.join("wrfinput_d01");
    let met = dir.join("met_em.d01.2026-09-12_00_00_00.nc");
    let authority = dir.join("authority.json");
    let out = dir.join("recovered");
    write_wrfinput(&wrfinput, 2, 2, stacked_kg, stacked_cm);
    write_met_em(&met, (2, 2), extent, stacked_kg, stacked_cm, stagger_names);
    std::fs::write(&authority, soil_authority(declared)).expect("authority");
    let output = run(&[
        "recover-wrf-soil",
        wrfinput.to_str().unwrap(),
        met.to_str().unwrap(),
        authority.to_str().unwrap(),
        out.to_str().unwrap(),
    ]);
    (out, output)
}

#[test]
fn a_table_declaring_more_layers_than_the_file_stacks_recovers_the_column() {
    let (out, output) = soil_case(
        "soil-superset", (2, 2), false, &DECLARED, &STACKED_CM, &STACKED_KG);
    assert_eq!(output.status.code(), Some(0), "{}", stderr(&output));
    let recovered = read_f64_plane(&out.join("SMOIS.f64"));
    assert_eq!(recovered.len(), CENTRES.len() * 4);
    for (index, centre) in CENTRES.iter().enumerate() {
        let want = wrf_soil_interpolation(&STACKED_VOLUME, &STACKED_CM, *centre);
        for cell in 0..4 {
            let got = recovered[index * 4 + cell];
            assert!((got - f64::from(want)).abs() < 1e-6,
                    "layer {index}: {got} against {want}");
        }
    }
    let document = std::fs::read_to_string(out.join("metadata.json")).expect("metadata");
    let pairing: String = document
        .split("\"authority_layers_used\"").nth(1).expect("pairing")
        .split(']').next().expect("pairing list")
        .chars().filter(|c| c.is_ascii_digit()).collect();
    assert_eq!(pairing, "013567", "{document}");
}

#[test]
fn a_stacked_depth_outside_every_declared_layer_is_refused_with_both_lists() {
    let declared: Vec<[f64; 2]> = DECLARED[..7].to_vec();
    let (_, output) = soil_case(
        "soil-outside", (2, 2), false, &declared, &STACKED_CM, &STACKED_KG);
    assert_ne!(output.status.code(), Some(0));
    let text = stderr(&output);
    assert!(text.contains("depths (cm) [1, 4, 30, 100, 160, 300]"), "{text}");
    assert!(text.contains("declares 7 layer(s)"), "{text}");
}

#[test]
fn a_table_matching_the_file_layer_for_layer_recovers_the_same_column() {
    let declared: Vec<[f64; 2]> = [0, 1, 3, 5, 6, 7].iter().map(|i| DECLARED[*i]).collect();
    let (out, output) = soil_case(
        "soil-exact", (2, 2), false, &declared, &STACKED_CM, &STACKED_KG);
    assert_eq!(output.status.code(), Some(0), "{}", stderr(&output));
    let recovered = read_f64_plane(&out.join("SMOIS.f64"));
    let want = wrf_soil_interpolation(&STACKED_VOLUME, &STACKED_CM, CENTRES[0]);
    assert!((recovered[0] - f64::from(want)).abs() < 1e-6);
}

#[test]
fn a_depth_no_declared_layer_holds_is_refused_with_both_lists() {
    let declared = [DECLARED[0], DECLARED[1], DECLARED[2]];
    let (_, output) = soil_case(
        "soil-unmapped", (2, 2), false, &declared, &STACKED_CM, &STACKED_KG);
    assert_ne!(output.status.code(), Some(0));
    let text = stderr(&output);
    assert!(text.contains("stacks 6 soil layer(s)"), "{text}");
    assert!(text.contains("declares only 3"), "{text}");
    assert!(text.contains("[0, 0.01], [0.01, 0.04], [0.04, 0.1]"), "{text}");
}

#[test]
fn a_source_grid_that_is_not_this_domain_names_both_extents() {
    let (_, output) = soil_case(
        "soil-extent", (4, 4), false, &DECLARED, &STACKED_CM, &STACKED_KG);
    assert_ne!(output.status.code(), Some(0));
    let text = stderr(&output);
    assert!(text.contains("XLAT_M covers 4 by 4 points"), "{text}");
    assert!(text.contains("XLAT covers 2 by 2"), "{text}");
}

#[test]
fn a_source_carrying_wrfs_extra_staggered_row_keeps_its_leading_block() {
    let (out, output) = soil_case(
        "soil-stagger", (3, 3), true, &DECLARED, &STACKED_CM, &STACKED_KG);
    assert_eq!(output.status.code(), Some(0), "{}", stderr(&output));
    let recovered = read_f64_plane(&out.join("SMOIS.f64"));
    assert_eq!(recovered.len(), CENTRES.len() * 4);
    let want = wrf_soil_interpolation(&STACKED_VOLUME, &STACKED_CM, CENTRES[0]);
    assert!((recovered[0] - f64::from(want)).abs() < 1e-6);
    let document = std::fs::read_to_string(out.join("metadata.json")).expect("metadata");
    assert!(document.contains("\"source_grid_extra_row_and_column\""), "{document}");
}
