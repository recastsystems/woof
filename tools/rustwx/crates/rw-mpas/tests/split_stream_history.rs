//! A split-stream history (RRFS-MPAS: no cell centres in `history.*.nc`, the
//! 2 m and reflectivity fields in `diag.*.nc`) converts on its `--mesh`, and a
//! history that carries its own centres is read exactly as before.
//!
//! THE BREAKAGE (hex 0.4.0 completeness row #39): `gpuwm-hex render` on an
//! RRFS run's own output stopped with "history has no latCell" although
//! `--mesh` was given, and its 2 m temperature and reflectivity could not be
//! drawn.
use std::path::{Path, PathBuf};

use rw_mpas::history::{
    diag_companion, read_history_cells, read_history_cells_on_mesh, read_history_on_mesh, read_history_with_init,
    Timestamp,
};
use rw_mpas::weights::read_mesh_coordinates;
use rw_store::netcdf_classic::{NcClassicWriter, NcData, NcDim, NcFormat, NcType, NcVarDef};

const LAT: [f64; 3] = [0.1, -0.2, 0.3];
const LON: [f64; 3] = [0.5, 1.0, -2.0];
const SHDMAX: [f32; 3] = [0.25, 0.5, 0.75];
const THETA: [f32; 6] = [300.0, 301.0, 302.0, 303.0, 304.0, 305.0];
const T2M: [f32; 3] = [280.5, 281.5, 282.5];
const REFL: [f32; 6] = [-10.0, 5.0, 20.0, 35.0, 0.0, 12.5];

fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("rw-mpas-split-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn dims(cells: usize) -> Vec<NcDim> {
    vec![NcDim::record("Time"), NcDim::fixed("nCells", cells), NcDim::fixed("nVertLevels", 2)]
}

fn write_mesh(path: &Path, lat: &[f64], lon: &[f64], shdmax: &[f32]) {
    let vars = vec![
        NcVarDef::new("latCell", NcType::Double, vec![1]),
        NcVarDef::new("lonCell", NcType::Double, vec![1]),
        NcVarDef::new("shdmax", NcType::Float, vec![1]),
    ];
    let mut w = NcClassicWriter::create(path, NcFormat::Data64, dims(lat.len()), vec![], vars, 1).unwrap();
    w.put("latCell", NcData::Doubles(lat)).unwrap();
    w.put("lonCell", NcData::Doubles(lon)).unwrap();
    w.put("shdmax", NcData::Floats(shdmax)).unwrap();
    w.finish().unwrap();
}

/// A history in the fork's layout (no centres, no t2m, no refl10cm), or in
/// MPAS-A's (`with_centres`).
fn write_history(path: &Path, with_centres: bool) {
    let mut vars = vec![
        NcVarDef::new("theta", NcType::Float, vec![0, 1, 2]),
        NcVarDef::new("shdmax", NcType::Float, vec![1]),
    ];
    if with_centres {
        vars.push(NcVarDef::new("latCell", NcType::Double, vec![1]));
        vars.push(NcVarDef::new("lonCell", NcType::Double, vec![1]));
    }
    let mut w = NcClassicWriter::create(path, NcFormat::Data64, dims(3), vec![], vars, 1).unwrap();
    w.put_record("theta", 0, NcData::Floats(&THETA)).unwrap();
    w.put("shdmax", NcData::Floats(&SHDMAX)).unwrap();
    if with_centres {
        w.put("latCell", NcData::Doubles(&LAT)).unwrap();
        w.put("lonCell", NcData::Doubles(&LON)).unwrap();
    }
    w.finish().unwrap();
}

fn write_diag(path: &Path) {
    let vars = vec![
        NcVarDef::new("t2m", NcType::Float, vec![0, 1]),
        NcVarDef::new("refl10cm", NcType::Float, vec![0, 1, 2]),
    ];
    let mut w = NcClassicWriter::create(path, NcFormat::Data64, dims(3), vec![], vars, 1).unwrap();
    w.put_record("t2m", 0, NcData::Floats(&T2M)).unwrap();
    w.put_record("refl10cm", 0, NcData::Floats(&REFL)).unwrap();
    w.finish().unwrap();
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn time() -> Option<Timestamp> {
    Some(Timestamp::parse("2023-03-31_13:00:00").unwrap())
}

#[test]
fn the_diag_companion_is_the_same_time_in_the_diag_stream() {
    let p = Path::new("/run/out/history.2023-03-31_13.00.00.nc");
    assert_eq!(diag_companion(p).unwrap(), Path::new("/run/out/diag.2023-03-31_13.00.00.nc"));
    assert!(diag_companion(Path::new("/run/out/mpasout.2023-03-31_13.00.00.nc")).is_none());
}

#[test]
fn a_split_stream_history_takes_centres_from_the_mesh_and_fields_from_its_diag() {
    let dir = scratch("rrfs");
    let mesh_path = dir.join("static.nc");
    let history = dir.join("history.2023-03-31_13.00.00.nc");
    write_mesh(&mesh_path, &LAT, &LON, &SHDMAX);
    write_history(&history, false);
    write_diag(&dir.join("diag.2023-03-31_13.00.00.nc"));
    let mesh = read_mesh_coordinates(&mesh_path, None).unwrap();

    // Before the fix this was the only answer, mesh or not.
    let err = read_history_with_init(&history, None, time(), None).unwrap_err().to_string();
    assert!(err.contains("history has no latCell"), "{err}");

    let frame = read_history_on_mesh(&history, None, time(), None, Some(&mesh)).unwrap();
    assert_eq!(frame.latitude_degrees, mesh.latitude_degrees);
    assert_eq!(frame.longitude_degrees, mesh.longitude_degrees);
    assert_eq!(bits(&frame.fields["T"].0), bits(&THETA));
    assert_eq!(bits(&frame.fields["T2"].0), bits(&T2M), "T2 is the diag stream's t2m, word for word");
    assert_eq!(frame.fields["REFL_10CM"].1, 2);
    assert_eq!(bits(&frame.fields["REFL_10CM"].0), bits(&REFL));
    assert!(!frame.absent.iter().any(|n| n == "T2" || n == "REFL_10CM"));
    assert!(frame.absent.iter().any(|n| n == "Q2"), "a field neither stream carries stays absent");
    let split = frame.split.as_ref().expect("a split-stream frame says where it came from");
    assert_eq!(split.agreed_static_fields, vec!["shdmax".to_string()]);
    assert_eq!(split.diag_fields, vec!["REFL_10CM".to_string(), "T2".to_string()]);
    assert!(split.diag_sha256.is_some());
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn a_mesh_that_is_not_the_runs_is_refused_by_name() {
    let dir = scratch("wrong-mesh");
    let history = dir.join("history.2023-03-31_13.00.00.nc");
    write_history(&history, false);
    // Same cell count, the static field at other cells: a rotated copy.
    let rotated = dir.join("rotated.nc");
    write_mesh(&rotated, &LAT, &LON, &[0.75, 0.5, 0.25]);
    let mesh = read_mesh_coordinates(&rotated, None).unwrap();
    let err = read_history_on_mesh(&history, None, time(), None, Some(&mesh)).unwrap_err().to_string();
    assert!(err.contains("shdmax differs from the mesh's at 2 of 3 cells"), "{err}");
    // A different cell count.
    let bigger = dir.join("bigger.nc");
    write_mesh(&bigger, &[0.1, 0.2, 0.3, 0.4], &[0.0; 4], &[0.0; 4]);
    let mesh = read_mesh_coordinates(&bigger, None).unwrap();
    let err = read_history_on_mesh(&history, None, time(), None, Some(&mesh)).unwrap_err().to_string();
    assert!(err.contains("the mesh has 4 cells and the history 3"), "{err}");
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn a_history_with_its_own_centres_reads_exactly_as_before_and_completes_from_its_diag() {
    let dir = scratch("mpas-a");
    let mesh_path = dir.join("static.nc");
    let history = dir.join("history.2023-03-31_13.00.00.nc");
    write_mesh(&mesh_path, &LAT, &LON, &SHDMAX);
    write_history(&history, true);
    write_diag(&dir.join("diag.2023-03-31_13.00.00.nc"));
    let mesh = read_mesh_coordinates(&mesh_path, None).unwrap();
    let before = read_history_with_init(&history, None, time(), None).unwrap();
    let after = read_history_on_mesh(&history, None, time(), None, Some(&mesh)).unwrap();
    assert!(after.split.is_none());
    assert_eq!(before.fields.keys().collect::<Vec<_>>(), after.fields.keys().collect::<Vec<_>>());
    for (name, (values, levels)) in &before.fields {
        assert_eq!(*levels, after.fields[name].1);
        assert_eq!(bits(values), bits(&after.fields[name].0), "{name}");
    }
    assert_eq!(before.absent, after.absent);
    // Since F-DIAG (ef31c5508b) an MPAS-A history missing a mapped field
    // completes it from its diag stream of the same time, with or without
    // the mesh: T2 comes from diag.*.nc and the frame names the file.
    assert!(!after.absent.iter().any(|n| n == "T2"), "T2 is read from the diag stream");
    assert_eq!(bits(&after.fields["T2"].0), bits(&T2M));
    let diag = after.companion_diag.as_ref().expect("the frame records its diag companion");
    assert!(diag.fields.iter().any(|n| n == "T2"), "{:?}", diag.fields);
    let to_bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(to_bits(&before.latitude_degrees), to_bits(&after.latitude_degrees));
    assert_eq!(to_bits(&before.longitude_degrees), to_bits(&after.longitude_degrees));
    std::fs::remove_dir_all(dir).unwrap();
}

/// The point door (`rw_hex_points`) reads a split-stream history at a few
/// cells as the converter reads the whole mesh (2.9.1: it stopped with
/// "history has no latCell" on RRFS output after the converter was fixed).
#[test]
fn a_split_stream_history_reads_at_cells_on_the_mesh() {
    let dir = scratch("rrfs-cells");
    let mesh_path = dir.join("static.nc");
    let history = dir.join("history.2023-03-31_13.00.00.nc");
    write_mesh(&mesh_path, &LAT, &LON, &SHDMAX);
    write_history(&history, false);
    write_diag(&dir.join("diag.2023-03-31_13.00.00.nc"));
    let mesh = read_mesh_coordinates(&mesh_path, None).unwrap();
    let cells = [2u32, 0];
    let wanted = ["T", "T2", "REFL_10CM", "Q2"];

    let err = read_history_cells(&history, None, &cells, &wanted, time()).unwrap_err().to_string();
    assert!(err.contains("history has no latCell"), "{err}");

    let frame = read_history_cells_on_mesh(&history, None, &cells, &wanted, time(), Some(&mesh)).unwrap();
    assert_eq!(frame.n_cells, 2);
    assert_eq!(frame.latitude_degrees, vec![mesh.latitude_degrees[2], mesh.latitude_degrees[0]]);
    assert_eq!(frame.longitude_degrees, vec![mesh.longitude_degrees[2], mesh.longitude_degrees[0]]);
    assert_eq!(bits(&frame.fields["T"].0), bits(&[THETA[4], THETA[5], THETA[0], THETA[1]]));
    assert_eq!(bits(&frame.fields["T2"].0), bits(&[T2M[2], T2M[0]]));
    assert_eq!(bits(&frame.fields["REFL_10CM"].0), bits(&[REFL[4], REFL[5], REFL[0], REFL[1]]));
    assert!(frame.absent.iter().any(|n| n == "Q2"));
    let split = frame.split.as_ref().expect("the frame names its split sources");
    assert_eq!(split.diag_fields, vec!["REFL_10CM".to_string(), "T2".to_string()]);
    assert!(split.diag_sha256.is_none(), "cell-subset reads hash no file");

    // A history with its own centres reads exactly as read_history_cells does, mesh or not.
    let own = dir.join("own").join("history.2023-03-31_13.00.00.nc");
    std::fs::create_dir_all(own.parent().unwrap()).unwrap();
    write_history(&own, true);
    let a = read_history_cells(&own, None, &cells, &wanted, time()).unwrap();
    let b = read_history_cells_on_mesh(&own, None, &cells, &wanted, time(), Some(&mesh)).unwrap();
    assert_eq!(a.latitude_degrees, b.latitude_degrees);
    assert_eq!(bits(&a.fields["T"].0), bits(&b.fields["T"].0));
    assert_eq!(a.absent, b.absent);
    assert!(b.split.is_none());
    std::fs::remove_dir_all(dir).unwrap();
}

/// The RRFS fork's history stream writes `ter` and never fills it: every word
/// is 0.0 (native MPAS-Model-rrfs and hex alike, 2026-10-28).  The frame's
/// `HGT` is then the init's terrain, on the whole-mesh path and the cell path,
/// so the renderer's MSLP reduction and height-above-ground products start
/// from the ground the run integrated on.  A filled history `ter` keeps its
/// own words, and an all-sea domain (init `ter` all zero too) keeps the zeros.
#[test]
fn an_unfilled_history_terrain_is_the_inits_and_a_filled_one_is_kept() {
    use rw_mpas::history::read_init_fields;
    const TER_INIT: [f32; 3] = [134.69, 344.22, 251.5];
    const TER_OWN: [f32; 3] = [130.0, 340.0, 250.0];
    let write_ter = |path: &Path, ter: &[f32], history: bool| {
        let mut vars = vec![NcVarDef::new("ter", NcType::Float, vec![1])];
        if history {
            vars.push(NcVarDef::new("theta", NcType::Float, vec![0, 1, 2]));
            vars.push(NcVarDef::new("shdmax", NcType::Float, vec![1]));
        }
        let mut w = NcClassicWriter::create(path, NcFormat::Data64, dims(3), vec![], vars, 1).unwrap();
        w.put("ter", NcData::Floats(ter)).unwrap();
        if history {
            w.put_record("theta", 0, NcData::Floats(&THETA)).unwrap();
            w.put("shdmax", NcData::Floats(&SHDMAX)).unwrap();
        }
        w.finish().unwrap();
    };
    let dir = scratch("unset-ter");
    let mesh_path = dir.join("static.nc");
    write_mesh(&mesh_path, &LAT, &LON, &SHDMAX);
    let mesh = read_mesh_coordinates(&mesh_path, None).unwrap();
    let init_path = dir.join("init.nc");
    write_ter(&init_path, &TER_INIT, false);
    let init = read_init_fields(&init_path, None).unwrap();
    let cells = [2u32, 0];

    for (case, history_ter, init_ter, want) in [
        ("unfilled", [0.0f32; 3], TER_INIT, TER_INIT),
        ("filled", TER_OWN, TER_INIT, TER_OWN),
        ("sea", [0.0f32; 3], [0.0f32; 3], [0.0f32; 3]),
    ] {
        let case_dir = dir.join(case);
        std::fs::create_dir_all(&case_dir).unwrap();
        let history = case_dir.join("history.2023-03-31_13.00.00.nc");
        write_ter(&history, &history_ter, true);
        let case_init_path = case_dir.join("init.nc");
        write_ter(&case_init_path, &init_ter, false);
        let case_init = read_init_fields(&case_init_path, None).unwrap();
        let frame = read_history_on_mesh(&history, Some(&case_init), time(), None, Some(&mesh)).unwrap();
        assert_eq!(bits(&frame.fields["HGT"].0), bits(&want), "{case}: whole mesh");
        assert!(!frame.absent.iter().any(|n| n == "HGT"), "{case}");
        let at = read_history_cells_on_mesh(&history, Some(&case_init_path), &cells, &["HGT"], time(), Some(&mesh))
            .unwrap();
        assert_eq!(bits(&at.fields["HGT"].0), bits(&[want[2], want[0]]), "{case}: cells");
    }
    // A history that omits `ter` still takes the init's (F-PG4), unchanged.
    let omitted = dir.join("history.2023-03-31_13.00.00.nc");
    write_history(&omitted, false);
    let frame = read_history_on_mesh(&omitted, Some(&init), time(), None, Some(&mesh)).unwrap();
    assert_eq!(bits(&frame.fields["HGT"].0), bits(&TER_INIT));
    std::fs::remove_dir_all(dir).unwrap();
}
