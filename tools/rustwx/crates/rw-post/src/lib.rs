//
// Copyright 2026 the WOOF authors.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License.  You may obtain a copy
// of the License at https://www.apache.org/licenses/LICENSE-2.0.  Unless
// required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied.

//! WOOF post-processor: clean-room, GPU-first replacement of the GRIB2
//! exporter's post modules (WOOF engine 2.8.7).
//!
//! Written only from the WOOF post specification and the public sources it
//! cites (WRF ARW technical note, AMS Glossary, Solar Energy 40 (1988) 227,
//! J. Appl. Meteor. 35 (1996) 601, Mon. Wea. Rev. 108 (1980) 1046, WMO-No. 8,
//! WMO-No. 306, J. Geophys. Res. 108 (2003) 8851 and the public WRF
//! parameter tables).
//!
//! One crate, one module per group of the specification, all built on ONE
//! shared column state and ONE maths library:
//! * Group A ([`state`], [`surface`] and the helpers [`thermo`], [`solar`],
//!   [`wind`], [`grib_coords`], [`hybrid`]): the shared column state every
//!   other group consumes, and the shelter and surface products;
//! * Group B ([`plev`]): pressure levels, sea-level pressure, column water;
//! * Group C ([`severe`]): parcels, shear, helicity, severe indices;
//! * Group D ([`group_d`]): reflectivity, cloud cover, cloud base, top and
//!   ceiling, visibility;
//! * Group E ([`group_e`]): boundary layer, near-surface winds, isotherms.
//!
//! [`math`] is the only source of elementary functions (D30): IEEE
//! `+ - * /` and `sqrt` only, no platform C library, and its CUDA twin
//! `kernels/woof_math.cuh` returns the same bits.
//!
//! Every computation has two paths that return the same bits:
//! * CPU: Rust, rayon over columns;
//! * GPU: ONE kernel set, `kernels/woof_post.cu` (one translation unit, one
//!   CUBIN per architecture, one manifest), loaded by [`gpu::GpuPost`].
//!
//! [`reference`] holds binary64 implementations of the same formulas that
//! the tests grade both paths against.

pub mod consts;
pub mod grib_coords;
pub mod history;
pub mod hybrid;
pub mod math;
pub mod plev;
pub mod group_d;
pub mod group_e;
pub mod severe;
pub mod reference;
pub mod snow;
pub mod solar;
pub mod state;
pub mod surface;
pub mod thermo;
pub mod wind;

#[cfg(any(windows, target_os = "linux"))]
pub mod gpu;

use state::{ColumnState, MassState};

/// Where post-processing runs (`--post-device`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PostDevice {
    /// GPU when a driver and enough free memory exist, else CPU.
    Auto,
    Gpu,
    Cpu,
}

impl std::str::FromStr for PostDevice {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "auto" => Ok(Self::Auto),
            "gpu" => Ok(Self::Gpu),
            "cpu" => Ok(Self::Cpu),
            other => Err(format!("--post-device must be auto, gpu or cpu, not {other}")),
        }
    }
}

/// Group A output of one frame.
pub struct GroupA {
    pub state: ColumnState,
    /// `SurfaceField::COUNT` planes.
    pub surface: Vec<f32>,
    /// "cpu" or the device description.
    pub device: String,
}

impl GroupA {
    pub fn plane(&self, f: surface::SurfaceField) -> &[f32] {
        let n = self.state.shape.ncell();
        &self.surface[f as usize * n..(f as usize + 1) * n]
    }
}

/// Lowest mass-level virtual temperature plane of a state.
pub fn tv_lowest(state: &ColumnState) -> Vec<f32> {
    let n = state.shape.ncell();
    state.mass_slot(MassState::Tv)[..n].to_vec()
}

/// Group A on the CPU.
pub fn group_a_cpu(frame: &mut history::Frame) -> Result<GroupA, String> {
    let mut state = state::build_cpu(&frame.state)?;
    height_coordinate_interfaces(frame, &mut state);
    frame.attach_tv_lowest(&tv_lowest(&state));
    let surface = surface::surface_cpu(&frame.surface, &frame.options);
    Ok(GroupA { state, surface, device: "cpu".into() })
}

/// A height-coordinate frame (no MU/ZNW, [`hybrid::CoefficientSource::HeightCoordinate`])
/// gets its interface pressures from its own mass-level pressures and heights
/// ([`state::fill_height_coordinate_interfaces`]); no surface field reads them,
/// so the patch lands before Groups B to E consume the state.
fn height_coordinate_interfaces(frame: &history::Frame, state: &mut ColumnState) {
    if frame.coefficients == hybrid::CoefficientSource::HeightCoordinate {
        let psfc = frame.surface.has(surface::SurfaceInput::Psfc).then(|| frame.surface.plane(surface::SurfaceInput::Psfc));
        state::fill_height_coordinate_interfaces(state, psfc);
    }
}

/// Group A on a GPU (state stays on the device for the surface pass).
#[cfg(any(windows, target_os = "linux"))]
pub fn group_a_gpu(gpu: &gpu::GpuPost, frame: &mut history::Frame) -> Result<GroupA, String> {
    let dstate = gpu.build_state(&frame.state).map_err(|e| e.to_string())?;
    let surface = gpu.surface(&frame.surface, &frame.options, Some(&dstate)).map_err(|e| e.to_string())?;
    let mut state = gpu.download_state(&dstate).map_err(|e| e.to_string())?;
    height_coordinate_interfaces(frame, &mut state);
    frame.attach_tv_lowest(&tv_lowest(&state));
    Ok(GroupA { state, surface, device: gpu.device.describe() })
}

/// Resolve `auto`: a GPU when one opens and has room for the frame.
#[cfg(any(windows, target_os = "linux"))]
pub fn open_device(choice: PostDevice, shape: &state::Shape) -> Result<Option<gpu::GpuPost>, String> {
    match choice {
        PostDevice::Cpu => Ok(None),
        PostDevice::Gpu => gpu::GpuPost::open(0).map(Some).map_err(|e| e.to_string()),
        PostDevice::Auto => match gpu::GpuPost::open(0) {
            Ok(g) => {
                let need = gpu::GpuPost::state_bytes(shape) + (64 << 20);
                match g.memory() {
                    Ok((free, _)) if free >= need => Ok(Some(g)),
                    _ => Ok(None),
                }
            }
            Err(_) => Ok(None),
        },
    }
}
