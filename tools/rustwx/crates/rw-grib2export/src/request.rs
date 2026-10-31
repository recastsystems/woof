//! The request a front door writes (schema `grib2-export.request/v1`).

use std::path::PathBuf;

use serde::Deserialize;

use crate::{refuse, Result};

pub const SCHEMA: &str = "grib2-export.request/v1";

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub schema: String,
    #[serde(default = "default_mode")]
    pub mode: String,
    #[serde(default)]
    pub inputs: Vec<PathBuf>,
    pub out: PathBuf,
    #[serde(default = "default_fields")]
    pub fields: String,
    #[serde(default = "default_packing")]
    pub packing: String,
    #[serde(default = "default_bits")]
    pub bits: u8,
    /// hPa.
    #[serde(default)]
    pub levels: Vec<u32>,
    #[serde(default = "default_definitions")]
    pub definitions: String,
    #[serde(default = "default_winds")]
    pub winds: String,
    #[serde(default)]
    pub domains: Vec<String>,
    #[serde(default)]
    pub times: Vec<String>,
    #[serde(default)]
    pub start: Option<String>,
    #[serde(default)]
    pub end: Option<String>,
    #[serde(default)]
    pub zip: bool,
    #[serde(default)]
    pub threads: Option<usize>,
    /// `auto` (default), `gpu` or `cpu` (spec 3.5).
    #[serde(default = "default_device")]
    pub post_device: String,
    /// `auto` (default: `tke` when the history carries TKE, else
    /// `similarity`), `similarity` or `tke` (RULINGS 7, scored against ASOS).
    #[serde(default = "default_gust")]
    pub gust: String,
    /// The window, in seconds, of the extrema rows (maximum and minimum
    /// updraft helicity, maximum 10 m wind speed: [`crate::catalog::EXTREMA`]).
    /// `null` (the default) writes no extrema; 3600 writes hourly extrema.
    /// Each window is built from the running extremes the history frames
    /// of that window carry ([`crate::extrema`]).
    #[serde(default)]
    pub extrema_interval_seconds: Option<u64>,
    /// Level type of `composite_reflectivity` (code table 4.5): 200, the
    /// NCEP local entire-atmosphere layer (default, the SRW control), or 10,
    /// WMO's entire atmosphere (NCEP's rapid-refresh control).  The front door's
    /// `--upp-control srw|...` names choose it; nothing else changes.
    #[serde(default = "default_composite_level")]
    pub composite_level_type: u8,
    /// Test hook: write every plane exactly as handed to the packer
    /// (`<dir>/<grib file stem>/<NNN>_<id>.f32`, little-endian, x fastest)
    /// so an independent decoder can be graded against it.
    #[serde(default)]
    pub dump_planes: Option<PathBuf>,
    /// Pressure levels in Pa, overriding `levels` for the pressure planes
    /// (Group B).  Not part of the request schema: set by profiles that need
    /// a level hPa cannot name, such as the RRFS UPP set's 1013.2 hPa.
    #[serde(skip)]
    pub levels_pa_override: Option<Vec<f64>>,
    /// Skip the 100 m height winds and the pressure-level SPFH/VVEL/DZDT
    /// extras (a serial wrf-core pass over the whole volume): set by profiles
    /// that write neither (the RRFS UPP set computes its own on the GPU).
    #[serde(skip)]
    pub skip_vertical_extras: bool,
}

fn default_mode() -> String {
    "run".into()
}
fn default_fields() -> String {
    "standard".into()
}
fn default_packing() -> String {
    "complex".into()
}
fn default_bits() -> u8 {
    20
}
fn default_definitions() -> String {
    "woof".into()
}
fn default_winds() -> String {
    "grid".into()
}
fn default_device() -> String {
    "auto".into()
}
fn default_gust() -> String {
    "auto".into()
}

impl Request {
    /// The request's default gust word, for tests.
    #[cfg(test)]
    pub(crate) fn default_gust_for_tests() -> String {
        default_gust()
    }
}
fn default_composite_level() -> u8 {
    COMPOSITE_LEVEL_SRW
}

/// `composite_reflectivity` on the NCEP local entire-atmosphere layer (200).
pub const COMPOSITE_LEVEL_SRW: u8 = 200;
/// `composite_reflectivity` on WMO's entire atmosphere (10).
pub const COMPOSITE_LEVEL_WMO: u8 = 10;
/// The longest extrema window: a day.  Longer windows would hold every
/// frame of a run in the window cache for no product anyone asked for.
pub const EXTREMA_MAX_SECONDS: u64 = 86_400;

/// Which definitions compute the catalog rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Definitions {
    /// The clean-room WOOF post-processor (`rw-post`), the default.
    Woof,
    /// The renderer's own diagnostics (`rw_wrfbatch`, wrf-core `getvar`)
    /// for the rows the renderer draws, so the GRIB2 numbers equal the map
    /// numbers; every other row stays WOOF's ([`crate::renderer`]).
    Renderer,
}

impl Request {
    pub fn read(path: &std::path::Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| refuse(format!("the request {} cannot be read ({e}), so there is nothing to export", path.display())))?;
        let request: Request = serde_json::from_str(&text)
            .map_err(|e| refuse(format!("the request {} is not a grib2-export.request/v1 document ({e})", path.display())))?;
        request.validate()?;
        Ok(request)
    }

    /// Each refusal names the breakage it prevents.
    pub fn validate(&self) -> Result<()> {
        if self.schema != SCHEMA {
            return Err(refuse(format!("request schema {} is not {SCHEMA}; an older or newer front door would read different fields", self.schema)));
        }
        if !matches!(self.mode.as_str(), "run" | "append" | "finalize") {
            return Err(refuse(format!("mode {} is not run, append or finalize", self.mode)));
        }
        if self.mode != "finalize" && self.inputs.is_empty() {
            return Err(refuse("no history inputs: an export with no frames would publish an empty folder as a result"));
        }
        if !matches!(self.fields.as_str(), "standard" | "surface" | "pressure" | "all") {
            return Err(refuse(format!("fields {} is not standard, surface, pressure or all", self.fields)));
        }
        if !matches!(self.packing.as_str(), "complex" | "simple") {
            return Err(refuse(format!("packing {} is not complex or simple", self.packing)));
        }
        if !(1..=31).contains(&self.bits) {
            return Err(refuse("bits must be 1 through 31; outside that range the packed integers cannot hold the field"));
        }
        if !matches!(self.definitions.as_str(), "woof" | "upp" | "renderer" | "arwen") {
            return Err(refuse(format!(
                "definitions {} is not woof, renderer or arwen; `upp` is accepted as an alias of `woof` for one release",
                self.definitions
            )));
        }
        if !matches!(self.winds.as_str(), "grid" | "earth") {
            return Err(refuse(format!("winds {} is not grid or earth", self.winds)));
        }
        if !matches!(self.post_device.as_str(), "auto" | "gpu" | "cpu") {
            return Err(refuse(format!("post_device {} is not auto, gpu or cpu", self.post_device)));
        }
        if !matches!(self.gust.as_str(), "auto" | "similarity" | "tke") {
            return Err(refuse(format!("gust {} is not auto, similarity or tke", self.gust)));
        }
        if let Some(n) = self.extrema_interval_seconds {
            if n == 0 || n > EXTREMA_MAX_SECONDS {
                return Err(refuse(format!(
                    "extrema_interval_seconds {n} is outside 1 through {EXTREMA_MAX_SECONDS}: a zero-length window has no extreme, and a longer one would hold a run's every frame"
                )));
            }
        }
        if !matches!(self.composite_level_type, COMPOSITE_LEVEL_SRW | COMPOSITE_LEVEL_WMO) {
            return Err(refuse(format!(
                "composite_level_type {} is not 200 (NCEP entire atmosphere) or 10 (WMO entire atmosphere); any other level would mislabel the composite reflectivity",
                self.composite_level_type
            )));
        }
        let mut seen = std::collections::BTreeSet::new();
        for &p in &self.levels {
            if !(1..=1100).contains(&p) || !seen.insert(p) {
                return Err(refuse("levels must be unique pressures from 1 through 1100 hPa"));
            }
        }
        for d in &self.domains {
            let ok = d.len() == 3 && d.starts_with('d') && d[1..].bytes().all(|b| b.is_ascii_digit()) && d != "d00";
            if !ok {
                return Err(refuse(format!("domain {d} is not dNN (d01, d02, ...)")));
            }
        }
        if self.threads == Some(0) {
            return Err(refuse("threads must be positive"));
        }
        Ok(())
    }

    /// Levels in hPa, defaulting to 100 through 1000 by 25.
    /// Group B's pressure levels, Pa.
    pub fn levels_pa(&self) -> Vec<f64> {
        match &self.levels_pa_override {
            Some(l) => l.clone(),
            None => self.levels_hpa().iter().map(|&p| f64::from(p) * 100.0).collect(),
        }
    }

    pub fn levels_hpa(&self) -> Vec<u32> {
        if self.levels.is_empty() { crate::catalog::default_levels_hpa() } else { self.levels.clone() }
    }

    pub fn writes_surface(&self) -> bool {
        self.fields != "pressure"
    }

    pub fn writes_pressure(&self) -> bool {
        self.fields != "surface"
    }

    /// The definitions that compute the rows; `upp` is `woof` and `arwen`
    /// is `renderer`.
    pub fn definitions_kind(&self) -> Definitions {
        if matches!(self.definitions.as_str(), "renderer" | "arwen") { Definitions::Renderer } else { Definitions::Woof }
    }

    pub fn device(&self) -> rw_post::PostDevice {
        self.post_device.parse().unwrap_or(rw_post::PostDevice::Auto)
    }
}
