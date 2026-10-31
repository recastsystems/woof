//! Where a WPS_GEOG archive is looked for.
//!
//! This is all that is left of a module that also built geography fields. The
//! field builder was one of two, and the streaming, admission-gated one in
//! [`crate::static_builder`] is the survivor; see [`crate::staticfile`]. The
//! ladder stayed because it is not about reduction at all -- it is the
//! agreement between where `woof fetch-geog` writes and where the engine
//! looks, and both doors read it from here.

use std::path::{Path, PathBuf};

/// Every place a geography root is looked for, best first.
///
/// THIS LADDER IS NOT FREE-CHOICE. It is `woof fetch-geog`'s own staging
/// ladder, spelled here so the engine finds the archive the product itself
/// downloaded. It used to stop after `$GPUWM_WPS_GEOG` and
/// `~/.local/share/gpuwm/WPS_GEOG`, which is the non-Windows case-data root and
/// nothing else -- so on Windows, or with `$GPUWM_CASE_DATA_ROOT` exported, the
/// door refused a complete archive that `woof fetch-geog` had just written.
/// A refusal that names data the product staged is a defect, not a refusal.
pub fn geog_root_candidates() -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = Vec::new();
    let mut push = |p: PathBuf| {
        if !out.contains(&p) {
            out.push(p);
        }
    };
    if let Ok(p) = std::env::var("GPUWM_WPS_GEOG") {
        let p = p.trim().to_string();
        if !p.is_empty() {
            push(PathBuf::from(p));
        }
    }
    // `$GPUWM_CASE_DATA_ROOT/WPS_GEOG`: fetch-geog's default `--root`, and the
    // `geog_root` every wizard-emitted config already carries.
    if let Ok(p) = std::env::var("GPUWM_CASE_DATA_ROOT") {
        let p = p.trim().to_string();
        if !p.is_empty() {
            push(PathBuf::from(p).join("WPS_GEOG"));
        }
    }
    // The same variable's unset fallback, per platform, mirroring
    // `gpuwm.case_data.default_case_data_root`.
    if cfg!(windows) {
        if let Ok(home) = std::env::var("USERPROFILE").or_else(|_| std::env::var("HOME")) {
            push(PathBuf::from(home).join("Downloads").join("WPS_GEOG"));
        }
    } else {
        if let Ok(xdg) = std::env::var("XDG_DATA_HOME") {
            let xdg = xdg.trim().to_string();
            if !xdg.is_empty() && Path::new(&xdg).is_absolute() {
                push(PathBuf::from(xdg).join("gpuwm").join("WPS_GEOG"));
            }
        }
        if let Ok(home) = std::env::var("HOME").or_else(|_| std::env::var("USERPROFILE")) {
            push(PathBuf::from(home).join(".local/share/gpuwm/WPS_GEOG"));
        }
    }
    out
}

/// Where a geography root lives when the caller did not say.
pub fn default_geog_root() -> Option<PathBuf> {
    geog_root_candidates().into_iter().find(|p| p.is_dir())
}
