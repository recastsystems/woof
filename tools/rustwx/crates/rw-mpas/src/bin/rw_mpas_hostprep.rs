//! Exact initialized MPAS terrain and momentum preparation.
use std::process::ExitCode;

pub static GPUWM_BRIDGE_SOURCE_REV_STAMP: &str =
    concat!("GPUWM_BRIDGE_SOURCE_REV=", env!("GPUWM_BRIDGE_SOURCE_REV"));

fn main() -> ExitCode {
    rw_mpas::execution::bridge(run_bridge)
}

fn run_bridge() -> ExitCode {
    let _stamp = std::hint::black_box(GPUWM_BRIDGE_SOURCE_REV_STAMP);
    let _deformation = std::hint::black_box(rw_mpas::hosttables::DEFORMATION_MARKER);
    let _tables = std::hint::black_box(rw_mpas::hosttables::TABLES_MARKER);
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args == ["--abi"] || args == ["--help"] {
        println!("{}", rw_mpas::hostprep::ABI_MARKER);
        println!("{}", rw_mpas::hosttables::TABLES_MARKER);
        println!("{}", rw_mpas::hosttables::DEFORMATION_MARKER);
        return ExitCode::SUCCESS;
    }
    if args != ["--protocol", "hex-hostprep-v1"] {
        eprintln!("usage: {}", rw_mpas::hostprep::ABI_MARKER);
        return ExitCode::FAILURE;
    }
    match rw_mpas::hostprep::run(
        std::io::stdin().lock(),
        std::io::BufWriter::new(std::io::stdout().lock()),
    ) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("host preparation: {error}");
            ExitCode::FAILURE
        }
    }
}
