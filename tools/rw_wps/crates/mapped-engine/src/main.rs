//! `gpuwm_mapped_engine`: the seam executable behind gpuwm's mapped prep.
//!
//! Contract (design doc §3; `gpuwm/mapped_engine_bridge.py` encodes the
//! Python half):
//!
//! * `decode`: one mapping, N input files → a frameset directory.
//!   Mirrors `gpuwm.mapped_source.decode_mapped_source`.
//! * `compose`: mapping + composition + role bindings → a frameset
//!   directory plus composition evidence.  Mirrors
//!   `gpuwm.mapped_composition.decode_composed_source`.
//! * `inspect`: selector resolution report on real bytes, stdout JSON
//!   (`gpuwm-mapped-source-inspection-v1`).  Mirrors
//!   `gpuwm.mapped_source.inspect_mapped_source`.
//!
//! stdout carries one JSON object per line, schema
//! `gpuwm-mapped-engine-progress-v1`; the LAST line is the run receipt (for
//! `inspect`, the inspection document itself).  A refusal exits nonzero
//! with one JSON object, schema `gpuwm-mapped-refusal-v1`, as the LAST
//! stderr line: Python maps `class` onto the exception type it raises.
//!
//! The ABI marker is compiled in from `mapped_engine::ABI_CONTRACT` and
//! anchored in `.rodata` by the usage text below, so a stale staged binary
//! fails the static handshake instead of writing a shape the Python side no
//! longer reads OR refusing bytes this release decodes.

use std::process::ExitCode;

use mapped_engine::engine::{Invocation, USAGE};
use mapped_engine::refusal::Refusal;
use mapped_engine::{ABI_CONTRACT, FRAMESET_SCHEMA, PROGRESS_SCHEMA, REFUSAL_SCHEMA};

/// `GPUWM_BRIDGE_SOURCE_REV=<40-hex commit>`: the source revision this
/// binary was built from, embedded so the gpuwm release cut can prove a
/// staged bridge matches the commit being released by reading bytes
/// alone (`tools/build_bridge_bundle.py pin --source-rev`).  `build.rs`
/// injects the value; `main` references the constant so the linker
/// cannot discard it.
pub static GPUWM_BRIDGE_SOURCE_REV_STAMP: &str =
    concat!("GPUWM_BRIDGE_SOURCE_REV=", env!("GPUWM_BRIDGE_SOURCE_REV"));

fn emit(value: serde_json::Value) {
    println!("{value}");
}

fn run() -> Result<(), Refusal> {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    if arguments.first().map(String::as_str) == Some("index-ranges") {
        emit(mapped_engine::lead_index::run(&arguments[1..])?);
        return Ok(());
    }
    if arguments.first().map(String::as_str) == Some("wps") {
        emit(mapped_engine::wps::run(&arguments[1..])?);
        return Ok(());
    }
    let invocation = Invocation::parse(&arguments).inspect_err(|_| {
        // The usage line doubles as compiled-in contract text, and the
        // schema names ride with it so the ABI marker literal is anchored
        // in .rodata whatever the optimizer does to the JSON writers.
        eprintln!("{USAGE}");
        eprintln!(
            "writes {FRAMESET_SCHEMA}; progress {PROGRESS_SCHEMA}; \
             refusals {REFUSAL_SCHEMA}"
        );
        eprintln!("{ABI_CONTRACT}");
    })?;
    let mut progress = |event: serde_json::Value| {
        let mut line = serde_json::json!({"schema": PROGRESS_SCHEMA});
        if let (Some(target), Some(source)) = (line.as_object_mut(), event.as_object()) {
            for (key, value) in source {
                target.insert(key.clone(), value.clone());
            }
        }
        emit(line);
    };
    match invocation.subcommand.as_str() {
        "capabilities" => {
            let mut capabilities = mapped_engine::engine::run_capabilities();
            capabilities["wps_intermediate_v5"] = serde_json::json!(true);
            capabilities["wps_mesh_support_v1"] = serde_json::json!(true);
            emit(capabilities);
            Ok(())
        }
        "decode" => {
            let receipt = mapped_engine::engine::run_decode(&invocation, &mut progress)?;
            progress(receipt);
            Ok(())
        }
        "inspect" => {
            let document = mapped_engine::engine::run_inspect(&invocation, &mut progress)?;
            emit(document);
            Ok(())
        }
        "inventory" => {
            let document = mapped_engine::engine::run_inventory(&invocation)?;
            emit(document);
            Ok(())
        }
        "compose" => {
            let receipt = mapped_engine::compose::run_compose(&invocation, &mut progress)?;
            progress(receipt);
            Ok(())
        }
        other => Err(mapped_engine::refusal::usage(format!(
            "unknown subcommand '{other}'"
        ))),
    }
}

fn main() -> ExitCode {
    let _ = std::hint::black_box(GPUWM_BRIDGE_SOURCE_REV_STAMP);
    let _ = std::hint::black_box(ABI_CONTRACT);
    // Before the first thread exists: glibc sizes its arena table as threads arrive.
    mapped_engine::threads::limit_allocator_arenas();
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(refusal) => {
            eprintln!("{}", refusal.to_json());
            // The seam contract pins NONZERO, never a per-class integer
            // (the class rides in the refusal JSON; Python's three call
            // sites all test `returncode != 0`).  The VALUE is pinned by
            // the release cut instead: doctor's `_exec_probe` -- run on
            // every staged executable by `tools/verify_release_artifacts
            // .py` and the publish workflow -- accepts exit 0, or 1/2
            // with a diagnostic, and refuses anything else as corrupt or
            // wrong-platform.  3 here burned a probe on a healthy binary;
            // 2 matches the sibling bridges' usage exits.
            ExitCode::from(2)
        }
    }
}
