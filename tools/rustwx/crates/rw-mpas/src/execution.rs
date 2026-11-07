//! Stack policy for the mesh and forecast-start bridges.
//!
//! These programs also run on hosts with small main-thread stacks. Give
//! the protocol thread and Rayon workers the same explicit budget, rather
//! than relying on a caller's RUST_MIN_STACK (which does not set main).
use std::process::ExitCode;

pub const BRIDGE_STACK_BYTES: usize = 256 * 1024 * 1024;

pub fn bridge(run: fn() -> ExitCode) -> ExitCode {
    match std::thread::Builder::new().name("mpas-bridge".into())
        .stack_size(BRIDGE_STACK_BYTES).spawn(move || {
            if let Err(error) = rayon::ThreadPoolBuilder::new()
                .stack_size(BRIDGE_STACK_BYTES).build_global() {
                eprintln!("MPAS worker pool: {error}");
                return ExitCode::FAILURE;
            }
            run()
        }) {
        Ok(thread) => match thread.join() {
            Ok(code) => code,
            Err(panic) => std::panic::resume_unwind(panic),
        },
        Err(error) => {
            eprintln!("MPAS protocol thread: {error}");
            ExitCode::FAILURE
        }
    }
}
