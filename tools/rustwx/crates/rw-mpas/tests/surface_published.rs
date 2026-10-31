//! Replay published real case-8 inputs through executable front doors.
//! All field decoding, word comparisons and receipts are Rust.
#[path = "common/surface_native.rs"]
mod surface_native;

use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Deserialize)]
struct Deck {
    label: String,
    static_path: PathBuf,
    sfc_prefix: PathBuf,
    start: String,
    stop: String,
    interval: i64,
    graph_prefix: PathBuf,
    /// Optional published full output, compared for the generated records.
    published_native: Option<PathBuf>,
}
fn sha(path: &Path) -> String {
    let mut f = File::open(path).unwrap();
    let mut hash = Sha256::new();
    let mut buf = [0; 65536];
    loop {
        let n = f.read(&mut buf).unwrap();
        if n == 0 {
            break;
        }
        hash.update(&buf[..n]);
    }
    format!("{:x}", hash.finalize())
}
fn compare(generated: &Path, native: &Path, allow_prefix: bool) -> serde_json::Value {
    let rust = netcrust::File::open(generated).unwrap();
    let reference = netcrust::File::open(native).unwrap();
    let records = rust.dimension("Time").unwrap().len();
    let cells = rust.dimension("nCells").unwrap().len();
    assert_eq!(cells, reference.dimension("nCells").unwrap().len());
    let native_records = reference.dimension("Time").unwrap().len();
    if allow_prefix {
        assert!(records <= native_records);
    } else {
        assert_eq!(records, native_records, "fresh native record count");
    }
    let mut fields = Vec::new();
    for name in ["Time", "sst", "xice"] {
        let got = rust.read_array::<f32>(name).unwrap();
        let want = reference.read_array::<f32>(name).unwrap();
        assert_eq!(got.len(), records * if name == "Time" { 1 } else { cells });
        assert_eq!(want.len(), native_records * if name == "Time" { 1 } else { cells });
        assert!(got.len() <= want.len());
        let mut mismatches = Vec::new();
        let mut differing = 0;
        let mut got_hash = Sha256::new();
        let mut want_hash = Sha256::new();
        for (index, (a, b)) in got.iter().zip(want.iter()).enumerate() {
            got_hash.update(a.to_bits().to_be_bytes());
            want_hash.update(b.to_bits().to_be_bytes());
            if a.to_bits() != b.to_bits() {
                differing += 1;
                if mismatches.len() < 12 {
                    mismatches.push(serde_json::json!({"index":index,"rust":format!("{:08x}",a.to_bits()),"native":format!("{:08x}",b.to_bits())}));
                }
            }
        }
        fields.push(serde_json::json!({"field":name,"words":got.len(),"different_words":differing,"first_differences":mismatches,"rust_words_sha256":format!("{:x}",got_hash.finalize()),"native_words_sha256":format!("{:x}",want_hash.finalize())}));
    }
    let got = rust.read_strings("xtime").unwrap();
    let want = reference.read_strings("xtime").unwrap();
    assert_eq!(got.len(), records);
    assert_eq!(want.len(), native_records);
    let different_xtime_bytes: usize = got
        .iter()
        .zip(want.iter())
        .map(|(a, b)| {
            assert_eq!(a.len(), 64);
            assert_eq!(b.len(), 64);
            a.bytes().zip(b.bytes()).filter(|(a, b)| a != b).count()
        })
        .sum();
    serde_json::json!({"records":records,"cells":cells,"fields":fields,"xtime_bytes":records*64,"different_xtime_bytes":different_xtime_bytes,"native_output_sha256":sha(native),"rust_output_sha256":sha(generated)})
}

/// Strict standalone full-module replay of the same real destination cells.
/// MPAS's pristine surface driver is copied by surface_oracle/build.sh;
/// only framework pools, clock and output are replaced by test adapters.
fn standalone(generated: &Path, deck: &Deck, work: &Path) -> Option<serde_json::Value> {
    let oracle = std::env::var_os("MPAS_SURFACE_ORACLE")?;
    let file = netcrust::File::open(generated).unwrap();
    let records = file.dimension("Time").unwrap().len();
    let cells = file.dimension("nCells").unwrap().len();
    let labels = file.read_strings("xtime").unwrap();
    let mut clock = format!("{records} {}\n", deck.interval);
    for label in &labels {
        clock.push_str(label.trim());
        clock.push('\n');
    }
    fs::write(work.join("clock.txt"), clock).unwrap();
    let raw_path = work.join("standalone.words");
    let output = Command::new(&oracle)
        .arg(&deck.static_path)
        .arg(&deck.sfc_prefix)
        .arg(work.join("clock.txt"))
        .arg(&raw_path)
        .output()
        .unwrap();
    assert!(output.status.success(), "{} {}", String::from_utf8_lossy(&output.stdout), String::from_utf8_lossy(&output.stderr));
    let raw = fs::read(&raw_path).unwrap();
    let stride = 64 + 4 * (1 + 2 * cells);
    assert_eq!(raw.len(), records * stride);
    let time: Vec<_> = file.read_array::<f32>("Time").unwrap().into_iter().collect();
    let sst: Vec<_> = file.read_array::<f32>("sst").unwrap().into_iter().collect();
    let ice: Vec<_> = file.read_array::<f32>("xice").unwrap().into_iter().collect();
    let mut different_words = 0;
    let mut different_xtime_bytes = 0;
    let mut expected = Sha256::new();
    for r in 0..records {
        let bytes = &raw[r * stride..(r + 1) * stride];
        different_xtime_bytes += labels[r].bytes().zip(bytes[..64].iter()).filter(|(a,b)| a != *b).count();
        expected.update(labels[r].as_bytes());
        let values = std::iter::once(time[r])
            .chain(sst[r*cells..(r+1)*cells].iter().copied())
            .chain(ice[r*cells..(r+1)*cells].iter().copied());
        for (value, bytes) in values.zip(bytes[64..].chunks_exact(4)) {
            expected.update(value.to_bits().to_be_bytes());
            different_words += usize::from(value.to_bits() != u32::from_be_bytes(bytes.try_into().unwrap()));
        }
    }
    Some(serde_json::json!({"records":records,"cells":cells,"numeric_words":records*(1+2*cells),"xtime_bytes":records*64,"different_words":different_words,"different_xtime_bytes":different_xtime_bytes,"native_record_bytes_sha256":sha(&raw_path),"rust_record_bytes_sha256":format!("{:x}",expected.finalize()),"oracle_sha256":sha(Path::new(&oracle))}))
}

#[test]
#[ignore = "requires f0 published inputs, strict executable and explicit deck JSON"]
fn published_case8_front_door_replay() {
    let decks: Vec<Deck> = serde_json::from_slice(
        &fs::read(std::env::var_os("MPAS_SURFACE_DECKS_JSON").unwrap()).unwrap(),
    )
    .unwrap();
    let engine = std::env::var_os("MPAS_SURFACE_ENGINE")
        .unwrap_or_else(|| env!("CARGO_BIN_EXE_rw_mpas_init").into());
    // Once the integrator exposes the surface door, exercise that checkout's
    // actual CLI. No replacement Python data path or simulated launcher.
    let hex_checkout = std::env::var_os("MPAS_SURFACE_HEX_CHECKOUT").map(PathBuf::from);
    let native = PathBuf::from(std::env::var_os("MPAS_SURFACE_NATIVE").unwrap());
    let root = PathBuf::from(std::env::var_os("MPAS_SURFACE_TEST_OUT").unwrap());
    fs::create_dir_all(&root).unwrap();
    let mut rows = Vec::new();
    for (i, deck) in decks.iter().enumerate() {
        let work = root.join(format!("deck-{i}-{}", std::process::id()));
        fs::create_dir_all(&work).unwrap();
        let generated = work.join("rust.nc");
        let mut command = if let Some(checkout) = &hex_checkout {
            let mut c = Command::new("python");
            c.args(["-m", "hexcore.cli", "surface-update", "--engine"])
                .arg(&engine)
                .env("PYTHONPATH", checkout.join("tree/src"));
            c
        } else {
            let mut c = Command::new(&engine);
            c.args(["--mode", "surface-update"]);
            c
        };
        let output = command
            .args([
                "--start-time",
                &deck.start,
                "--stop-time",
                &deck.stop,
                "--fg-interval",
                &deck.interval.to_string(),
                "--surface-interval",
                &deck.interval.to_string(),
            ])
            .arg("--sfc-prefix")
            .arg(&deck.sfc_prefix)
            .arg("--static")
            .arg(&deck.static_path)
            .arg("--out")
            .arg(&generated)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let run = surface_native::run(
            &native,
            &deck.static_path,
            &deck.sfc_prefix,
            &deck.graph_prefix,
            &deck.start,
            &deck.stop,
            deck.interval,
            1,
            &work.join("native"),
        );
        assert!(run["success"].as_bool().unwrap(), "{}", run["logs"]);
        let replay = compare(&generated, &work.join("native/native.nc"), false);
        let published = deck
            .published_native
            .as_ref()
            .map(|path| compare(&generated, path, true));
        let module = standalone(&generated, deck, &work);
        let pass = |v: &serde_json::Value| {
            v["different_xtime_bytes"].as_u64() == Some(0)
                && v["fields"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .all(|field| field["different_words"].as_u64() == Some(0))
        };
        let exact = pass(&replay) && published.as_ref().is_none_or(pass)
            && module.as_ref().is_none_or(|v| v["different_words"].as_u64()==Some(0) && v["different_xtime_bytes"].as_u64()==Some(0));
        rows.push(serde_json::json!({"label":deck.label,"native_ranks":1,"rust_processes":1,"standalone_processes":1,"static":deck.static_path,"sfc_prefix":deck.sfc_prefix,"start":deck.start,"stop":deck.stop,"strict_native_binary_sha256":sha(&native),"fresh_native":replay,"published_native":published,"standalone_module":module,"exact":exact}));
        fs::write(root.join("real-replay.json"),serde_json::to_vec_pretty(&serde_json::json!({"schema":"mpas.surface-real-replay/v1","host":fs::read_to_string("/proc/sys/kernel/hostname").unwrap_or_default().trim(),"native_ranks":1,"rust_processes":1,"standalone_processes":1,"front_door":if hex_checkout.is_some(){"woof hex surface-update"}else{"rw_mpas_init --mode surface-update"},"hex_checkout":hex_checkout,"hex_front_door_proven":hex_checkout.is_some(),"whole_model_proof":false,"rows":rows})).unwrap()).unwrap();
        fs::remove_dir_all(work).unwrap();
        assert!(
            exact,
            "real case-8 field differences: {}",
            rows.last().unwrap()
        );
    }
}
