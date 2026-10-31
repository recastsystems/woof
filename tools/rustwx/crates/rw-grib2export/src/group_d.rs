//! Group D: reflectivity, cloud, visibility (spec section 7) and the
//! simulated infrared brightness temperature, from `rw_post::group_d`.
//!
//! Group D reads Group A's shared column state and the frame's packed state
//! inputs (hydrometeors), plus the carriers only it needs: `REFL_10CM`,
//! `CLDFRA`, the 550 nm aerosol extinction, `HGT` and `OLR`.  It returns its planes
//! by catalog id (`composite_reflectivity`, `reflectivity_1km`, `low_cloud`,
//! `mid_cloud`, `high_cloud`, `cloud_base`, `cloud_top`, `cloud_ceiling`,
//! `visibility`, `simulated_ir`).  A field the carriers cannot support is listed as omitted
//! with rw-post's reason; absence is never read as zero.
//!
//! Method defaults are rw-post's one-line switches (`group_d::REFL_INTERP`,
//! `CLOUD_FRACTION_SOURCE`, `DIAGNOSED_CLOUD`, `VIS_CONTRAST`,
//! `VIS_WITHOUT_AEROSOL`), set per
//! RULINGS.md; the diagnosed cloud fraction (Xu and Randall 1996) is
//! PROVISIONAL pending a ruling.

use std::collections::BTreeMap;
use std::path::Path;

use rw_post::group_d::{self, history::OwnedCarriers, Options, FIELDS};
use rw_post::state::{ColumnState, StateInputs};
use rw_post::PostDevice;

/// Group D planes of one frame, by catalog id, plus per-field omissions.
pub struct Output {
    pub planes: BTreeMap<&'static str, Vec<f32>>,
    pub omitted: Vec<(&'static str, String)>,
    pub device: String,
    /// Where the cloud fraction came from (manifest text), when used.
    pub cloud_fraction_source: Option<&'static str>,
    /// Which carriers the aerosol extinction came from, or, without one,
    /// the clear-air baseline visibility used instead (RULINGS 10).
    pub aerosol_source: String,
    /// Which branches `simulated_ir` used (manifest text), when written.
    pub ir_source: Option<&'static str>,
}

/// Compute Group D for one frame.
///
/// `history` is the frame's file (for the Group D carriers), `state` the
/// shared column state, `inputs` the frame's packed state inputs, `gpu` the
/// opened kernel set when the frame runs on a card, `device` the device
/// choice for the groups (`Auto` falls back to the CPU on a device error).
pub fn compute(
    history: &Path,
    state: &ColumnState,
    inputs: &StateInputs,
    #[cfg(any(windows, target_os = "linux"))] gpu: Option<&rw_post::gpu::GpuPost>,
    device: PostDevice,
    terrain: Option<&[f32]>,
) -> Result<Output, String> {
    let mut owned = OwnedCarriers::read(history).map_err(|e| format!("{}: {e}", history.display()))?;
    // A height-coordinate frame's terrain is the ground of its coordinate
    // (see `export::compute`), not its HGT carrier.
    if let Some(t) = terrain {
        owned.hgt = Some(t.to_vec());
    }
    let c = owned.carriers();
    let opt = Options::default();
    #[cfg(any(windows, target_os = "linux"))]
    let d = group_d::compute(state, inputs, &c, &opt, gpu, device).map_err(|e| format!("group D: {e}"))?;
    #[cfg(not(any(windows, target_os = "linux")))]
    let d = {
        let _ = device;
        group_d::compute_cpu(state, inputs, &c, &opt).map_err(|e| format!("group D: {e}"))?
    };
    let n = d.nx * d.ny;
    let mut planes = BTreeMap::new();
    for (k, f) in FIELDS.iter().enumerate() {
        if d.omitted.iter().any(|(id, _)| *id == f.id) {
            continue;
        }
        planes.insert(f.id, d.planes[k * n..(k + 1) * n].to_vec());
    }
    Ok(Output {
        planes,
        omitted: d.omitted.clone(),
        device: d.device.clone(),
        cloud_fraction_source: d.cloud_fraction_source,
        // The note describes the visibility that was written; when it was
        // omitted for another carrier, it says so instead of naming a
        // baseline nothing used.
        aerosol_source: match d.omitted.iter().find(|(id, _)| *id == "visibility") {
            Some((_, why)) => format!("not used: visibility omitted ({why})"),
            None => group_d::aerosol_note(&owned.aerosol_source, &opt),
        },
        ir_source: d.ir_source,
    })
}
