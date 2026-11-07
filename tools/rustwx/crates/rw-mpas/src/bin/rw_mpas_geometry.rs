//! Derived vector geometry and exact RBF reconstruction for the hex init door.

use std::process::ExitCode;

pub static GPUWM_BRIDGE_SOURCE_REV_STAMP: &str =
    concat!("GPUWM_BRIDGE_SOURCE_REV=", env!("GPUWM_BRIDGE_SOURCE_REV"));

fn main() -> ExitCode {
    rw_mpas::execution::bridge(run_bridge)
}

fn run_bridge() -> ExitCode {
    let _stamp = std::hint::black_box(GPUWM_BRIDGE_SOURCE_REV_STAMP);
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args == ["--abi"] {
        println!("{}",rw_mpas::geometry::ABI_MARKER);
        return ExitCode::SUCCESS;
    }
    if args == ["--help"] {
        println!("{}\nstdin: HEXGEO1 little-endian arrays; stdout: raw derived arrays",rw_mpas::geometry::ABI_MARKER);
        return ExitCode::SUCCESS;
    }
    if args == ["--protocol", "hex-vertical-v1"] {
        let stdin = std::io::stdin();
        let stdout = std::io::stdout();
        return match rw_mpas::init::vertical::run(stdin.lock(), std::io::BufWriter::new(stdout.lock())) {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => { eprintln!("vertical: {error}"); ExitCode::FAILURE }
        };
    }
    if args == ["--protocol", "hex-vertical-v2"] {
        let stdin = std::io::stdin();
        let stdout = std::io::stdout();
        return match rw_mpas::init::vertical::run_v2(std::io::BufReader::with_capacity(1 << 20, stdin.lock()),
                std::io::BufWriter::with_capacity(1 << 20, stdout.lock())) {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => { eprintln!("vertical: {error}"); ExitCode::FAILURE }
        };
    }
    if args == ["--protocol", "hex-cold-artifact-v1"] {
        return match rw_mpas::init::cold_artifact::run(std::io::stdin().lock(), std::io::stdout().lock()) {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => { eprintln!("cold artifact: {error}"); ExitCode::FAILURE }
        };
    }
    if args == ["--protocol", "hex-init-artifact-v1"] {
        let stdin = std::io::stdin();
        let stdout = std::io::stdout();
        return match rw_mpas::init::artifact::run(stdin.lock(), stdout.lock()) {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => { eprintln!("artifact: {error}"); ExitCode::FAILURE }
        };
    }
    if args == ["--protocol", "hex-advection-compression-v1"] {
        let stdin=std::io::stdin();let stdout=std::io::stdout();
        return match rw_mpas::init::advection::run_compression(stdin.lock(),std::io::BufWriter::new(stdout.lock())) {
            Ok(())=>ExitCode::SUCCESS,Err(error)=>{eprintln!("compression: {error}");ExitCode::FAILURE}
        };
    }
    if args == ["--protocol", "hex-advection-v1"] {
        let stdin = std::io::stdin();
        let stdout = std::io::stdout();
        return match rw_mpas::init::advection::run(stdin.lock(), std::io::BufWriter::new(stdout.lock())) {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => { eprintln!("advection: {error}"); ExitCode::FAILURE }
        };
    }
    if args != ["--protocol","hex-geometry-v1"] {
        eprintln!("usage: {}",rw_mpas::geometry::ABI_MARKER);
        return ExitCode::FAILURE;
    }
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    match rw_mpas::geometry::run(stdin.lock(),std::io::BufWriter::new(stdout.lock())) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => { eprintln!("geometry: {error}"); ExitCode::FAILURE }
    }
}
