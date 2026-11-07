//! Inject the source revision every bridge artifact embeds.
//!
//! The release cut refuses to pin a bundle whose binaries do not carry
//! `GPUWM_BRIDGE_SOURCE_REV=<40-hex commit>` naming the exact commit
//! being released (`tools/build_bridge_bundle.py pin --source-rev`).
//! The stamp is read out of the built binary as plain bytes -- never by
//! executing it -- so it must be a literal the compiler embeds, which
//! is what the `GPUWM_BRIDGE_SOURCE_REV` env this script emits is for.
//!
//! Resolution order:
//! * `GPUWM_BRIDGE_SOURCE_REV` set in the environment and shaped like a
//!   full commit wins (a build farm that knows better than the checkout);
//! * otherwise the checkout's HEAD, but only when the workspace tree is
//!   clean -- a binary built from modified sources is NOT that commit,
//!   and a stamp that lies is worse than none;
//! * otherwise `unknown`, which the cut refuses with a message naming
//!   the remedy.  Local development builds are unaffected: nothing
//!   outside the release cut reads the stamp.

use std::process::Command;

fn main() {
    println!("cargo:rerun-if-env-changed=GPUWM_BRIDGE_SOURCE_REV");
    for input in ["src", "Cargo.toml", "Cargo.lock", "../../woof/data/cam3/state-layout.json", "../../woof/data/cam3/state-defaults.json"] {
        println!("cargo:rerun-if-changed={input}");
    }
    let rev = std::env::var("GPUWM_BRIDGE_SOURCE_REV")
        .ok()
        .filter(|value| is_commit(value))
        .or_else(head_of_clean_checkout)
        .unwrap_or_else(|| String::from("unknown"));
    println!("cargo:rustc-env=GPUWM_BRIDGE_SOURCE_REV={rev}");
}

fn is_commit(value: &str) -> bool {
    value.len() == 40
        && value
            .bytes()
            .all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

/// Source stamping covers this crate and the pinned CAM data directory.
fn workspace_dir() -> Option<String> {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").ok()?;
    let root = std::path::Path::new(&manifest);
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
    for tracked in ["HEAD", "packed-refs", "index"] {
        if let Some(path) = git(&["rev-parse", "--path-format=absolute", "--git-path", tracked]) {
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
    // The crate and its included CAM data must match the stamped commit.
    let dirty = git(&["status", "--porcelain", "-uno", "--", ".", "../../woof/data/cam3"])?;
    if !dirty.is_empty() {
        return None;
    }
    let rev = git(&["rev-parse", "HEAD"])?.to_ascii_lowercase();
    is_commit(&rev).then_some(rev)
}
