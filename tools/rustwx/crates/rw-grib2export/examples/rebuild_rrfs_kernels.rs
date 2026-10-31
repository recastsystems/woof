//! Developer-only rebuild of `kernels/rrfs_upp.ptx`, the RRFS UPP-set extras.
//!
//! Normal builds include the checked-in PTX and never call NVRTC.  This
//! example inlines every local `#include "..."` of `kernels/rrfs_upp.cu`
//! (each header once; headers are looked up beside it, then in rw-post's
//! `kernels/`, which holds the shared maths library and the Group C parcel
//! helpers), compiles the unit to compute_60 PTX with NVRTC, and writes
//! `kernels/rrfs_upp.manifest.json` with the source and artifact hashes.
//!
//! Run it where NVRTC 12.0 is first on the loader path (the
//! `nvidia-cuda-nvrtc-cu12==12.0.76` wheel): NVRTC stamps its own PTX ISA
//! version and a driver refuses a newer ISA than it knows, so 12.0 (ISA 8.0)
//! keeps every driver from R525 on (rw-post's `PTX_ISA_CEILING`).

#[cfg(any(windows, target_os = "linux"))]
use std::{
    collections::BTreeSet,
    ffi::CString,
    path::{Path, PathBuf},
};

#[cfg(any(windows, target_os = "linux"))]
use cudarc::nvrtc::{result, sys};
#[cfg(any(windows, target_os = "linux"))]
use sha2::{Digest, Sha256};

#[cfg(any(windows, target_os = "linux"))]
const OPTIONS: &[&str] = &[
    "--ftz=false",
    "--prec-sqrt=true",
    "--prec-div=true",
    "--fmad=false",
    "--std=c++17",
    "--gpu-architecture=compute_60",
];

#[cfg(any(windows, target_os = "linux"))]
fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

#[cfg(any(windows, target_os = "linux"))]
fn inline(dirs: &[PathBuf], file: &str, seen: &mut BTreeSet<String>, sources: &mut Vec<(String, String)>) -> String {
    let path = dirs.iter().map(|d| d.join(file)).find(|p| p.exists()).unwrap_or_else(|| panic!("{file} not found"));
    let text = std::fs::read_to_string(&path).unwrap();
    sources.push((file.to_owned(), sha256(text.as_bytes())));
    let mut out = String::new();
    for line in text.lines() {
        let t = line.trim();
        if t == "#pragma once" {
            continue;
        }
        if let Some(rest) = t.strip_prefix("#include \"") {
            let name = rest.trim_end_matches('"').to_owned();
            if seen.insert(name.clone()) {
                out.push_str(&format!("// ---- begin {name}\n"));
                out.push_str(&inline(dirs, &name, seen, sources));
                out.push_str(&format!("// ---- end {name}\n"));
            }
            continue;
        }
        out.push_str(line);
        out.push('\n');
    }
    out
}

#[cfg(any(windows, target_os = "linux"))]
fn main() {
    let here = Path::new(env!("CARGO_MANIFEST_DIR"));
    let dirs = vec![here.join("kernels"), here.join("../rw-post/kernels")];
    let mut seen = BTreeSet::new();
    let mut sources = Vec::new();
    let unit = inline(&dirs, "rrfs_upp.cu", &mut seen, &mut sources);
    let source = CString::new(unit).unwrap();
    let program = result::create_program(&source, Some(c"rrfs_upp.cu")).unwrap();
    let options: Vec<String> = OPTIONS.iter().map(|o| (*o).to_owned()).collect();
    if let Err(error) = unsafe { result::compile_program(program, &options) } {
        let log = unsafe { result::get_program_log(program) }.unwrap_or_default();
        let log: String = log.iter().map(|&c| c as u8 as char).collect();
        panic!("NVRTC: {error}\n{log}");
    }
    let raw = unsafe { result::get_ptx(program) }.unwrap();
    let mut ptx: Vec<u8> = raw.into_iter().map(|b| b as u8).collect();
    if ptx.last() == Some(&0) {
        ptx.pop();
    }
    let (mut major, mut minor) = (0, 0);
    let _ = unsafe { sys::nvrtcVersion(&mut major, &mut minor) };
    std::fs::write(here.join("kernels/rrfs_upp.ptx"), &ptx).unwrap();
    let manifest = serde_json::json!({
        "kernel_set": "rrfs_upp",
        "kernels": ["rrfs_upp_column_v1", "rrfs_upp_isobaric_v1", "rrfs_upp_absv_v1"],
        "artifact": {"file": "rrfs_upp.ptx", "architecture": "compute_60 PTX", "bytes": ptx.len(),
                     "sha256": sha256(&ptx), "generator": format!("NVIDIA NVRTC {major}.{minor}")},
        "options": OPTIONS,
        "sources": sources.iter().map(|(f, h)| serde_json::json!({"file": f, "sha256": h})).collect::<Vec<_>>(),
    });
    std::fs::write(here.join("kernels/rrfs_upp.manifest.json"), serde_json::to_string_pretty(&manifest).unwrap() + "\n").unwrap();
    println!("rrfs_upp.ptx: {} bytes, NVRTC {major}.{minor}", ptx.len());
}

#[cfg(not(any(windows, target_os = "linux")))]
fn main() {}
