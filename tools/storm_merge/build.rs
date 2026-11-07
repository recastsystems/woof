// SPDX-License-Identifier: Apache-2.0
use std::process::Command;
fn main() {
    println!("cargo:rerun-if-env-changed=GPUWM_BRIDGE_SOURCE_REV");
    for name in ["src", "Cargo.toml", "Cargo.lock", "build.rs"] {
        println!("cargo:rerun-if-changed={name}");
    }
    let git = |args: &[&str]| -> Option<String> {
        let result = Command::new("git").args(args).output().ok()?;
        result.status.success().then_some(())?;
        Some(String::from_utf8(result.stdout).ok()?.trim().to_string())
    };
    for name in ["HEAD", "packed-refs"] {
        if let Some(path) = git(&["rev-parse", "--path-format=absolute", "--git-path", name]) {
            println!("cargo:rerun-if-changed={path}");
        }
    }
    if let Some(reference) = git(&["symbolic-ref", "-q", "HEAD"]) {
        if let Some(path) = git(&[
            "rev-parse",
            "--path-format=absolute",
            "--git-path",
            &reference,
        ]) {
            println!("cargo:rerun-if-changed={path}");
        }
    }
    let clean = git(&[
        "status",
        "--porcelain",
        "--untracked-files=normal",
        "--",
        ".",
    ])
    .is_some_and(|s| s.is_empty());
    let revision = std::env::var("GPUWM_BRIDGE_SOURCE_REV")
        .ok()
        .or_else(|| clean.then(|| git(&["rev-parse", "HEAD"])).flatten())
        .filter(|s| s.len() == 40 && s.bytes().all(|b| b.is_ascii_hexdigit()))
        .unwrap_or_else(|| "unknown".into());
    println!("cargo:rustc-env=GPUWM_BRIDGE_SOURCE_REV={revision}");
}
