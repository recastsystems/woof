//! `rw_mpas_mpassit`: MPAS history + diag -> the WRF-shaped Lambert file
//! MPASSIT writes, from MPASSIT's own namelist and variable lists.
//!
//! RRFS v2's `exrrfs_mpassit.sh` runs `mpassit.x namelist.mpassit` in a
//! directory holding `diaglist`, `histlist_2d`, `histlist_3d` and
//! `histlist_soil`.  This binary takes the same namelist and lists, so the job
//! script's call can name it instead: `rw_mpas_mpassit namelist.mpassit`.
//! See `rw_mpas::mpassit` for what is reproduced and how it was established
//! (MPASSIT itself is GPL-3.0 and was neither read nor ported).

use std::path::PathBuf;
use std::process::ExitCode;

use rw_store::netcdf_classic::NcFormat;

pub static GPUWM_BRIDGE_SOURCE_REV_STAMP: &str =
    concat!("GPUWM_BRIDGE_SOURCE_REV=", env!("GPUWM_BRIDGE_SOURCE_REV"));

/// The contract a caller can probe without running a remap (`--abi`): the
/// schema, the inputs it reads and the flags an older build would refuse.
pub static ABI: &str = "rw_mpas_mpassit NAMELIST schema=gpuwm.mpassit-equivalent-remap/v1 lists=diaglist,histlist_2d,histlist_3d,histlist_soil target=lambert wrf_mod_vars=true --lists DIR --jobs N --receipt PATH --allow-absent --format cdf5|cdf2";

const USAGE: &str = "usage: rw_mpas_mpassit NAMELIST [--lists DIR] [--jobs N] [--receipt PATH] \
[--allow-absent] [--format cdf5|cdf2]

  NAMELIST        MPASSIT's &config namelist (RRFS v2: parm/namelist.mpassit filled by the job)
  --lists DIR     directory holding diaglist, histlist_2d, histlist_3d, histlist_soil
                  (default: the namelist's directory, which is where MPASSIT looks)
  --jobs N        worker threads (default: every core)
  --receipt PATH  write the JSON receipt here (default: stdout)
  --allow-absent  write the file even when a listed variable is not in the input
                  (the receipt lists what was absent)
  --format        cdf5 (default; 64-bit sizes, any grid) or cdf2";

fn main() -> ExitCode {
    let _stamp = std::hint::black_box(GPUWM_BRIDGE_SOURCE_REV_STAMP);
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args == ["--abi"] {
        println!("{}", std::hint::black_box(ABI));
        return ExitCode::SUCCESS;
    }
    if args == ["--version"] {
        println!("rw_mpas_mpassit {} ({})", env!("CARGO_PKG_VERSION"), GPUWM_BRIDGE_SOURCE_REV_STAMP);
        return ExitCode::SUCCESS;
    }
    if args.is_empty() || args.iter().any(|a| a == "--help" || a == "-h") {
        println!("{USAGE}");
        return if args.is_empty() { ExitCode::FAILURE } else { ExitCode::SUCCESS };
    }
    let mut namelist: Option<PathBuf> = None;
    let mut lists: Option<PathBuf> = None;
    let mut jobs: Option<usize> = None;
    let mut receipt: Option<PathBuf> = None;
    let mut allow_absent = false;
    let mut format = NcFormat::Data64;
    let mut it = args.into_iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--lists" => lists = it.next().map(PathBuf::from),
            "--jobs" => jobs = it.next().and_then(|v| v.parse().ok()),
            "--receipt" => receipt = it.next().map(PathBuf::from),
            "--allow-absent" => allow_absent = true,
            "--format" => match it.next().as_deref() {
                Some("cdf5") => format = NcFormat::Data64,
                Some("cdf2") => format = NcFormat::Offset64,
                other => {
                    eprintln!("rw_mpas_mpassit: --format {other:?}: choose cdf5 or cdf2\n{USAGE}");
                    return ExitCode::FAILURE;
                }
            },
            other if namelist.is_none() && !other.starts_with("--") => namelist = Some(PathBuf::from(other)),
            other => {
                eprintln!("rw_mpas_mpassit: unknown argument {other}\n{USAGE}");
                return ExitCode::FAILURE;
            }
        }
    }
    let Some(namelist) = namelist else {
        eprintln!("{USAGE}");
        return ExitCode::FAILURE;
    };
    if let Some(n) = jobs {
        if let Err(e) = rayon::ThreadPoolBuilder::new().num_threads(n).build_global() {
            eprintln!("rw_mpas_mpassit: --jobs {n}: {e}");
            return ExitCode::FAILURE;
        }
    }
    let base = namelist
        .parent()
        .map(|p| if p.as_os_str().is_empty() { PathBuf::from(".") } else { p.to_path_buf() })
        .unwrap_or_else(|| PathBuf::from("."));
    let text = match std::fs::read_to_string(&namelist) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("rw_mpas_mpassit: {}: {e}", namelist.display());
            return ExitCode::FAILURE;
        }
    };
    let nl = match rw_mpas::mpassit::Namelist::parse(&text, &base) {
        Ok(nl) => nl,
        Err(e) => {
            eprintln!("rw_mpas_mpassit: {e}");
            return ExitCode::FAILURE;
        }
    };
    let lists = lists.unwrap_or(base);
    match rw_mpas::mpassit::run(&nl, &lists, allow_absent, format) {
        Ok(r) => {
            let json = serde_json::to_string_pretty(&r).expect("receipt serialises");
            match receipt {
                Some(p) => {
                    if let Err(e) = std::fs::write(&p, json + "\n") {
                        eprintln!("rw_mpas_mpassit: receipt {}: {e}", p.display());
                        return ExitCode::FAILURE;
                    }
                }
                None => println!("{json}"),
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("rw_mpas_mpassit: {e}");
            ExitCode::FAILURE
        }
    }
}
