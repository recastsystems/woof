//! Two build-time values the simulated-radar binary embeds.
//!
//! `SIMRADAR_IMPLEMENTATION_SHA256` hashes this crate's own sources and the
//! site table, so a manifest names the implementation that wrote it.
//!
//! `GPUWM_BRIDGE_SOURCE_REV` is the source revision every bundled bridge
//! artifact embeds.  `rw_simradar` is in `gpuwm.bridge_assets
//! .BUNDLED_ARTIFACTS`, and the release cut refuses to pin a bundle whose
//! binaries do not carry `GPUWM_BRIDGE_SOURCE_REV=<40-hex commit>` naming
//! the exact commit being released (`tools/build_bridge_bundle.py pin
//! --source-rev`).  Without it the cut cannot prove the staged binary is
//! this commit's and refuses the bundle.  The stamp is read out of the
//! built binary as plain bytes, never by executing it, so it must be a
//! literal the compiler embeds.
//!
//! Resolution order for the revision:
//! * `GPUWM_BRIDGE_SOURCE_REV` set in the environment and shaped like a
//!   full commit wins (a build farm that knows better than the checkout);
//! * otherwise the checkout's HEAD, but only when the workspace tree is
//!   clean -- a binary built from modified sources is NOT that commit,
//!   and a stamp that lies is worse than none;
//! * otherwise `unknown`, which the cut refuses with a message naming
//!   the remedy.  Local development builds are unaffected: nothing
//!   outside the release cut reads the stamp.

use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::process::Command;

fn collect(root: &Path, paths: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(root).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            collect(&path, paths);
        } else {
            paths.push(path);
        }
    }
}

fn main() {
    let root = PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let mut paths = vec![root.join("Cargo.toml"), root.join("build.rs")];
    collect(&root.join("src"), &mut paths);
    paths.push(root.join("data/sites.json"));
    let workspace = root.parent().unwrap().parent().unwrap();
    let operator = workspace.join("vendor/bowecho/crates/bowecho_simradar");
    paths.push(workspace.join("vendor/bowecho/crates/data_source/src/sites.rs"));
    for name in ["src/wrf_band_scattering.rs", "src/wrf_radar_core.rs",
                 "src/wrf_radar_physics.rs", "data/scattering-C.bin",
                 "data/scattering-X.bin", "data/scattering-p3.bin",
                 "data/scattering-C.json", "data/scattering-X.json"] {
        paths.push(operator.join(name));
    }
    paths.sort();
    let mut hash = Sha256::new();
    for path in paths {
        println!("cargo:rerun-if-changed={}", path.display());
        hash.update(
            path.strip_prefix(&root)
                .or_else(|_|path.strip_prefix(workspace)).unwrap()
                .to_string_lossy()
                .replace('\\', "/"),
        );
        // Ignore checkout line-ending policy when identifying the same source.
        let bytes=std::fs::read(&path).unwrap();
        if path.extension().is_some_and(|e|e=="bin") {hash.update(bytes);}
        else {hash.update(std::str::from_utf8(&bytes).unwrap().replace("\r\n","\n"));}
    }
    println!(
        "cargo:rustc-env=SIMRADAR_IMPLEMENTATION_SHA256={:x}",
        hash.finalize()
    );

    println!("cargo:rerun-if-env-changed=GPUWM_BRIDGE_SOURCE_REV");
    let rev = std::env::var("GPUWM_BRIDGE_SOURCE_REV")
        .ok()
        .filter(|value| is_commit(value))
        .or_else(head_of_clean_checkout)
        .unwrap_or_else(|| String::from("unknown"));
    println!("cargo:rustc-env=GPUWM_BRIDGE_SOURCE_REV={rev}");
}

fn is_commit(value: &str) -> bool {
    value.len() == 40 && value.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

/// The crate lives at `crates/<name>` inside the renderer workspace; the
/// tree that must be clean is the workspace root, since the binary is
/// built from crates and vendored dependencies across it.
fn workspace_dir() -> Option<String> {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").ok()?;
    let root = std::path::Path::new(&manifest).parent()?.parent()?;
    Some(root.to_string_lossy().into_owned())
}

fn git(args: &[&str]) -> Option<String> {
    let dir = workspace_dir()?;
    let output = Command::new("git")
        .arg("-C")
        .arg(&dir)
        .args(args)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    Some(String::from_utf8(output.stdout).ok()?.trim().to_string())
}

fn head_of_clean_checkout() -> Option<String> {
    // Re-run when HEAD moves so an incremental target directory cannot
    // hold yesterday's stamp.  `--git-path` resolves worktrees too.
    for tracked in ["HEAD", "packed-refs"] {
        if let Some(path) = git(&["rev-parse", "--path-format=absolute", "--git-path", tracked]) {
            println!("cargo:rerun-if-changed={path}");
        }
    }
    if let Some(reference) = git(&["symbolic-ref", "-q", "HEAD"]) {
        if let Some(path) = git(&["rev-parse", "--path-format=absolute", "--git-path", &reference]) {
            println!("cargo:rerun-if-changed={path}");
        }
    }
    // Tracked modifications under this workspace mean the binary is not
    // HEAD, whatever HEAD says.  Untracked files (target/, dist output)
    // do not enter the build of tracked sources and are ignored.
    let dirty = git(&["status", "--porcelain", "-uno", "--", "."])?;
    if !dirty.is_empty() {
        return None;
    }
    let rev = git(&["rev-parse", "HEAD"])?.to_ascii_lowercase();
    is_commit(&rev).then_some(rev)
}
