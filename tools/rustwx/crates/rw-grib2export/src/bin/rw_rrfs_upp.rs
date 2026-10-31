//! `rw_rrfs_upp`: RRFS v2's downstream GRIB2 (the UPP field set of
//! rrfs_mpas_postcntrl_L60, in prslev / natlev / testbed files) from
//! WRF-shaped frames on the output grid (MPASSIT layout).
//!
//! usage: rw_rrfs_upp --out DIR [--dt SECONDS] [--bits N] [--post-device gpu|auto|cpu]
//!                    [--prefix NAME] [--bucket-seconds S] --frames F1 [F2 ...]
//!
//! Frames are processed in the order given (bucket fields difference
//! consecutive frames); each frame uses every core and the GPU.  A manifest
//! (`rrfs-upp-manifest.json`) records every file, every field's status and
//! per-stage wall time.

#[cfg(any(windows, target_os = "linux"))]
fn main() {
    use rw_grib2export::rrfs_upp::{self, Carry, Options};
    let mut opts = Options {
        out: std::path::PathBuf::from("."),
        dt_seconds: None,
        bucket_seconds: 3600,
        bits: 16,
        post_device: "gpu".into(),
        prefix: "hexrrfs".into(),
    };
    let mut frames: Vec<std::path::PathBuf> = Vec::new();
    let mut args = std::env::args().skip(1).peekable();
    while let Some(a) = args.next() {
        let mut val = || args.next().unwrap_or_else(|| { eprintln!("{a} needs a value"); std::process::exit(2) });
        match a.as_str() {
            "--out" => opts.out = val().into(),
            "--dt" => opts.dt_seconds = Some(val().parse().expect("--dt seconds")),
            "--bits" => opts.bits = val().parse().expect("--bits"),
            "--post-device" => opts.post_device = val(),
            "--prefix" => opts.prefix = val(),
            "--bucket-seconds" => opts.bucket_seconds = val().parse().expect("--bucket-seconds"),
            "--frames" => {
                while let Some(n) = args.peek() {
                    if n.starts_with("--") { break; }
                    frames.push(args.next().unwrap().into());
                }
            }
            "--profile" => {
                // Print the field set and exit.
                for f in rrfs_upp::profile() {
                    println!("{}\t{}\t{}.{}.{}\t{}\t{:?}", f.datset, f.shortname, f.discipline, f.category, f.number, f.level_type, f.levels);
                }
                return;
            }
            other => { eprintln!("unknown flag {other}"); std::process::exit(2) }
        }
    }
    if frames.is_empty() {
        eprintln!("--frames is required");
        std::process::exit(2);
    }
    let gpu = match rw_grib2export::rrfs_gpu::RrfsGpu::open(0) {
        Ok(g) => g,
        Err(e) => { eprintln!("the RRFS extras need a CUDA device: {e}"); std::process::exit(3) }
    };
    let choice = match opts.post_device.as_str() { "cpu" => rw_post::PostDevice::Cpu, "auto" => rw_post::PostDevice::Auto, _ => rw_post::PostDevice::Gpu };
    let mut devices = rw_grib2export::export::Devices::new(choice);
    let mut carry = Carry::default();
    let mut records = Vec::new();
    let t0 = std::time::Instant::now();
    for f in &frames {
        match rrfs_upp::export_frame(f, &opts, &mut devices, &gpu, &mut carry) {
            Ok(r) => {
                println!("{}", serde_json::json!({"event": "frame", "source": f.display().to_string(), "seconds": r["seconds"], "files": r["files"]}));
                records.push(r);
            }
            Err(e) => { eprintln!("{}: {e}", f.display()); std::process::exit(1) }
        }
    }
    let manifest = serde_json::json!({"profile": rrfs_upp::PROFILE, "frames": records, "seconds": t0.elapsed().as_secs_f64()});
    std::fs::write(opts.out.join("rrfs-upp-manifest.json"), serde_json::to_string_pretty(&manifest).unwrap()).expect("write manifest");
}

#[cfg(not(any(windows, target_os = "linux")))]
fn main() {}
