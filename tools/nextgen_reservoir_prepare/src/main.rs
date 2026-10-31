//! Apache-2.0. Native t-route construction semantics at pin12a8eae0 (NOTICE retained).
//! Startup/fixture preparation only; no reservoir time-stepping on the CPU.
use rayon::prelude::*;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, env, fs, path::{Path, PathBuf}, time::Instant};
mod fabric;

const PIN: &str = "12a8eae0cdfed437143c590659fa7077605a5e70";
const PARAMETERS: [&str; 9] = ["area_km2", "weir_m", "maximum_m", "weir_coefficient", "weir_length_m", "dam_length_m", "orifice_m", "orifice_coefficient", "orifice_area_m2"];
const CONTROLS: [&str; 8] = ["lookback_hours", "observation_interval_seconds", "persistence_interval_seconds", "rfc_enabled", "rfc_interval_seconds", "rfc_persist_until_seconds", "rfc_total_counts", "source_filename_id"];
fn hash(bytes: &[u8]) -> String { format!("{:x}", Sha256::digest(bytes)) }
fn read_column(root: &Path, arrays: &Value, name: &str, dtype: &str, n: usize) -> Vec<u8> {
    let spec = &arrays[name];
    assert_eq!(spec["dtype"], dtype, "native column dtype {name}");
    assert_eq!(spec["shape"], json!([n]), "native column shape {name}");
    let path = Path::new(spec["path"].as_str().expect("column path"));
    assert!(!path.is_absolute() && path.components().all(|c| matches!(c, std::path::Component::Normal(_))), "column path escapes capture");
    let data = fs::read(root.join(path)).expect("read captured column");
    assert_eq!(data.len(), n * 8, "native column length {name}");
    assert_eq!(hash(&data), spec["sha256"].as_str().expect("source SHA256"), "captured column integrity {name}");
    data
}
fn i64s(bytes: &[u8]) -> Vec<i64> { bytes.chunks_exact(8).map(|x|i64::from_le_bytes(x.try_into().unwrap())).collect() }
fn f64s(bytes: &[u8]) -> Vec<f64> { bytes.chunks_exact(8).map(|x|f64::from_le_bytes(x.try_into().unwrap())).collect() }
fn floats(values: &[f32]) -> Vec<u8> { values.iter().flat_map(|x|x.to_le_bytes()).collect() }
fn doubles(values: &[f64]) -> Vec<u8> { values.iter().flat_map(|x|x.to_le_bytes()).collect() }
/// The commit this binary was built from; the release pin step reads it from the bytes.
const SOURCE_REV_STAMP: &str = concat!("GPUWM_BRIDGE_SOURCE_REV=", env!("GPUWM_BRIDGE_SOURCE_REV"));

fn usage() -> ! {
    eprintln!("usage: nextgen-reservoir-prepare NATIVE_MANIFEST NEW_OUTPUT TIME_ORIGIN
       nextgen-reservoir-prepare --fabric FABRIC_MANIFEST INITIAL_STATE_JSON NEW_OUTPUT");
    std::process::exit(2);
}

fn main() {
    std::hint::black_box(SOURCE_REV_STAMP);
    let started = Instant::now();
    let args: Vec<_> = env::args().collect();
    // A wrong argument count prints the usage and exits 2, never a panic (exit 101): the release natives qualification
    // and `woof doctor` probe every bridge with `--version` and refuse a panic as "corrupt or built for another
    // platform" (a release natives build of 05c018fe0c stopped on it).
    if args.get(1).map(String::as_str) == Some("--fabric") {
        if args.len() != 5 { usage(); }
        if let Err(error) = fabric::run(Path::new(&args[2]), Path::new(&args[3]), Path::new(&args[4])) {
            eprintln!("reservoir fabric preparation: {error}");
            std::process::exit(1);
        }
        return;
    }
    if args.len() != 4 { usage(); }
    let manifest_path = PathBuf::from(&args[1]);
    let input_bytes = fs::read(&manifest_path).expect("native manifest");
    let source: Value = serde_json::from_slice(&input_bytes).expect("native manifest JSON");
    assert_eq!(source["schema"], "troute-native-network-observation-v1");
    let a = &source["arrays"];
    let root = manifest_path.parent().unwrap();
    let execution_path = root.parent().unwrap().join("execution.json");
    let execution_bytes = fs::read(&execution_path).expect("native execution receipt beside network capture");
    let execution: Value = serde_json::from_slice(&execution_bytes).expect("native execution receipt JSON");
    assert_eq!(execution["source_pin"], PIN, "native reservoir source pin differs from audited source");
    assert_eq!(execution["returncode"], 0, "native source run did not succeed");
    let n = a["waterbody-parameters-row-index"]["shape"][0].as_u64().expect("reservoir rows") as usize;
    let ids_raw = read_column(root, a, "waterbody-parameters-row-index", "<i8", n);
    let ids = i64s(&ids_raw);
    let types_raw = read_column(root, a, "waterbody-types-column-reservoir_type", "<i8", n);
    let types = i64s(&types_raw);
    assert_eq!(ids_raw, read_column(root, a, "waterbody-types-row-index", "<i8", n), "native reservoir type/parameter ordering differs");
    assert_eq!(ids.iter().collect::<std::collections::BTreeSet<_>>().len(), n, "duplicate native waterbody ID");
    assert!(types.iter().all(|&t|t == 1), "native DA types require actual controls/series captures; do not initialize them as levelpool");
    let columns = ["LkArea", "WeirE", "LkMxE", "WeirC", "WeirL", "OrificeE", "OrificeC", "OrificeA", "ifd", "h0", "qd0"];
    let decoded: BTreeMap<_,_> = columns.par_iter().map(|&name| {
        let key = format!("waterbody-parameters-column-{name}");
        let raw = read_column(root, a, &key, "<f8", n);
        let values = f64s(&raw).into_iter().map(|v| {
            let f = v as f32;
            assert!(f.is_finite(), "nonfinite native parameter {name}");
            f
        }).collect::<Vec<_>>();
        (name, values)
    }).collect();
    let mut parameters = Vec::with_capacity(9*n);
    for name in ["LkArea", "WeirE", "LkMxE", "WeirC", "WeirL", "Dam_Length", "OrificeE", "OrificeC", "OrificeA"] {
        // Actual levelpool.pyx constructor sets dam_length = 10.0, not a data default.
        if name == "Dam_Length" { parameters.extend(std::iter::repeat(10.0f32).take(n)); }
        else { parameters.extend_from_slice(&decoded[name]); }
    }
    let mut cold = 0;
    let heights: Vec<_> = (0..n).map(|i| {
        let h = decoded["h0"][i];
        // levelpool_structs.c converts arguments to float BEFORE this sentinel/math.
        if h < -900000000.0f32 {
            cold += 1;
            let delta = decoded["LkMxE"][i] - decoded["OrificeE"][i];
            let part = delta * decoded["ifd"][i];
            decoded["OrificeE"][i] + part
        } else { h }
    }).collect();
    let mut payloads: Vec<(String,String,Vec<usize>,Vec<u8>)> = Vec::new();
    let mut push = |name: &str, dtype: &str, shape: Vec<usize>, data: Vec<u8>|payloads.push((name.into(),dtype.into(),shape,data));
    push("parameters", "<f4", vec![9,n], floats(&parameters));
    for name in ["height", "initial_height", "levelpool_height"] { push(name,"<f4",vec![n],floats(&heights)); }
    for name in ["previous_inflow", "inflow", "lateral", "levelpool_release"] { push(name,"<f4",vec![n],vec![0;n*4]); }
    push("types", "<i4", vec![n],types.iter().flat_map(|&t|(t as i32).to_le_bytes()).collect());
    for name in ["indices", "dynamic_type", "active_source", "fault"] { push(name,"<i4",vec![n],vec![0;n*4]); }
    push("offsets","<i8",vec![n+1],vec![0;(n+1)*8]);
    push("controls","<f8",vec![8,n],vec![0;8*n*8]);
    // These DA carriers are deliberately inactive for captured type1 rows.
    push("carriers","<f8",vec![3,n],vec![0;3*n*8]);
    push("clock","<f8",vec![n],vec![0;n*8]);
    push("release","<f8",vec![n],doubles(&decoded["qd0"].iter().map(|&x|f64::from(x)).collect::<Vec<_>>()));
    push("assimilated","<f8",vec![n],doubles(&vec![-9999.;n]));
    for name in ["series", "times"] { push(name,"<f8",vec![0],vec![]); }
    let output = Path::new(&args[2]);
    fs::create_dir(output).expect("output must be new; never overwrite evidence");
    let specs: Vec<_> = payloads.par_iter().map(|(name,dtype,shape,data)| {
        let key = format!("reservoir_{name}");
        let path = format!("{key}.bin");
        fs::write(output.join(&path),data).expect("write reservoir payload");
        (key,json!({"path":path,"dtype":dtype,"shape":shape,"bytes":data.len(),"sha256":hash(data)}))
    }).collect();
    let files: BTreeMap<_,_> = specs.into_iter().collect();
    let bindings: BTreeMap<_,_> = payloads.iter().map(|(name,_,_,_)|(name.clone(),format!("reservoir_{name}"))).collect();
    fs::write(output.join("reservoir_ids.i64"), &ids_raw).unwrap();
    let result = json!({
        "schema":"arwen.nextgen.reservoir-payloads.v1", "files":files,
        "reservoirs":{"arrays":bindings,"source_pin":PIN,"parameter_columns":PARAMETERS,"control_columns":CONTROLS,
          "validated_csr":true,"time_origin":args[3],"reservoir_order_sha256":hash(&ids_raw)},
        "reservoir_ids":{"path":"reservoir_ids.i64","dtype":"<i8","shape":[n],"sha256":hash(&ids_raw)},
        "source_manifest":{"path":manifest_path,"sha256":hash(&input_bytes)},
        "native_execution":{"path":execution_path,"sha256":hash(&execution_bytes)},
        "scope":"actual native type1 waterbody construction; not topology, DA proof or reference stepping",
        "construction":{"dam_length_m":10,"native_cold_threshold_m":-900000000,"cold_rows":cold,
           "conversion":"captured f64 -> native constructor f32 before initialization math; no CPU step"},
        "data_scope":"private native research replay; data permission/provenance inherited from native capture, never public demo payloads",
        "rows":n,"parallelism":{"rayon_threads":rayon::current_num_threads(),"work":"independent sealed input columns and payload writes"},
        "wall_ms":started.elapsed().as_secs_f64()*1e3
    });
    fs::write(output.join("manifest.json"),serde_json::to_vec_pretty(&result).unwrap()).unwrap();
    println!("{}",serde_json::to_string_pretty(&result).unwrap());
}
