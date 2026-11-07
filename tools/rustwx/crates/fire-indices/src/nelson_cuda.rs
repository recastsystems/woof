//! CUDA accelerator for the expensive hourly radial transport. CPU fallback is
//! the independently implemented Rust solver; the receipt records the backend.
use crate::{Cell, Result};
use cudarc::{
    driver::{CudaContext, CudaFunction, CudaStream, LaunchConfig, PushKernelArg},
    nvrtc::{CompileOptions, compile_ptx_with_opts},
};
use rayon::prelude::*;
use std::sync::Arc;
pub struct Backend {
    device: Option<(Arc<CudaStream>, CudaFunction, u32)>,
}
impl Backend {
    pub fn open(choice: &str) -> Result<Self> {
        if choice == "cpu" {
            return Ok(Self { device: None });
        }
        // cudarc's dynamic loader panics when libcuda is absent. A CPU-only
        // host is a supported configuration, not a failed run.
        let opened = std::panic::catch_unwind(|| -> std::result::Result<_, String> {
            let ctx = CudaContext::new(0).map_err(|e| e.to_string())?;
            let stream = ctx.new_stream().map_err(|e| e.to_string())?;
            let ptx = compile_ptx_with_opts(
                include_str!("nelson.cu"),
                CompileOptions {
                    fmad: Some(false),
                    ..Default::default()
                },
            )
            .map_err(|e| e.to_string())?;
            let module = ctx.load_module(ptx).map_err(|e| e.to_string())?;
            let function = module
                .load_function("nelson_hour")
                .map_err(|e| e.to_string())?;
            // The radial solver holds several double-precision node arrays.
            // cudarc's 1024-thread convenience launch exceeds its register
            // budget. Let the driver use the compiled kernel's actual resources.
            extern "C" fn no_dynamic_shared(_: std::ffi::c_int) -> usize {
                0
            }
            let (_, threads) = function
                .occupancy_max_potential_block_size(no_dynamic_shared, 0, 0, None)
                .map_err(|e| e.to_string())?;
            if threads == 0 {
                return Err("CUDA returned zero Nelson block size".into());
            }
            eprintln!("Nelson GPU block size: {threads} threads");
            Ok((stream, function, threads))
        });
        match opened {
            Ok(Ok(device)) => Ok(Self {
                device: Some(device),
            }),
            error => {
                let message = match error {
                    Ok(Err(e)) => e,
                    _ => "CUDA driver unavailable".into(),
                };
                if choice == "cuda" {
                    return Err(message.into());
                }
                eprintln!("Nelson uses Rust CPU Rayon: {message}");
                Ok(Self { device: None })
            }
        }
    }
    pub fn active(&self) -> bool {
        self.device.is_some()
    }
    pub fn name(&self) -> &str {
        if self.active() {
            if cudarc::gpuwm_hip::requested() {
                "Rust + HIP Nelson radial transport; Rayon daily indices"
            } else {
                "Rust + CUDA Nelson radial transport; Rayon daily indices"
            }
        } else {
            "Rust CPU Rayon"
        }
    }
    pub fn allocated_mib(&self, cells: usize) -> f64 {
        if self.active() {
            (cells * 4 * 29 * 8) as f64 / 1048576.
        } else {
            0.
        }
    }
    pub fn hour(
        &mut self,
        cells: &mut [Cell],
        temp: &[f64],
        dew: &[f64],
        solar: &[f64],
        rain: &[f64],
    ) -> Result<()> {
        let Some((stream, kernel, threads)) = &self.device else {
            return Ok(());
        };
        let packets: Vec<f64> = cells
            .par_iter()
            .enumerate()
            .flat_map_iter(|(i, c)| {
                let t = temp[i] - 273.15;
                let d = dew[i] - 273.15;
                let rh =
                    ((17.625 * d / (243.04 + d) - 17.625 * t / (243.04 + t)).exp()).clamp(0., 1.);
                let r = rain[i].max(0.) * 1000.;
                let wet = if r > 0. { c.nfdrs.wet_event_mm + r } else { 0. };
                (0..4).flat_map(move |k| {
                    let mut p = [0.; 29];
                    p[..21].copy_from_slice(&c.nfdrs.profiles[k]);
                    p[21..28].copy_from_slice(&[
                        t,
                        rh,
                        solar[i].max(0.) / 3600.,
                        r,
                        c.nfdrs.previous_rain,
                        c.nfdrs.previous_rh,
                        wet,
                    ]);
                    p
                })
            })
            .collect();
        let mut data = stream.clone_htod(&packets)?;
        let n = (cells.len() * 4) as i32;
        unsafe {
            stream
                .launch_builder(kernel)
                .arg(&mut data)
                .arg(&n)
                .launch(LaunchConfig {
                    grid_dim: ((n as u32).div_ceil(*threads), 1, 1),
                    block_dim: (*threads, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        let output = stream.clone_dtoh(&data)?;
        cells.par_iter_mut().enumerate().for_each(|(i, c)| {
            for k in 0..4 {
                let p = &output[(i * 4 + k) * 29..(i * 4 + k + 1) * 29];
                c.nfdrs.profiles[k].copy_from_slice(&p[..21]);
                c.nfdrs.stick_temperature[k] = p[28];
            }
            let t = temp[i] - 273.15;
            let d = dew[i] - 273.15;
            let rh = (17.625 * d / (243.04 + d) - 17.625 * t / (243.04 + t))
                .exp()
                .clamp(0., 1.);
            let r = rain[i].max(0.) * 1000.;
            c.nfdrs.wet_event_mm = if r > 0. { c.nfdrs.wet_event_mm + r } else { 0. };
            c.nfdrs.track_hour(t, rh, r);
        });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fire_indices::{FwiState, KbdiState, nfdrs};
    #[test]
    fn cuda_matches_independent_rust_transport_when_available() {
        let choice = if std::env::var_os("FIRE_INDICES_REQUIRE_CUDA_TEST").is_some() {
            "cuda"
        } else {
            "auto"
        };
        let mut gpu = Backend::open(choice).unwrap();
        if !gpu.active() {
            return;
        }
        let mut cells = vec![
            Cell {
                fwi: FwiState::default(),
                kbdi: KbdiState {
                    deficit: 0.,
                    wet_event_mm: 0.
                },
                nfdrs: nfdrs::State::default(),
                rain: 0.,
                tmax: -90.,
                hours: 0
            };
            1024
        ];
        let mut reference = cells.clone();
        for hour in 0..72 {
            let t: Vec<_> = (0..1024)
                .map(|i| 273.15 + 12. + ((hour + i) % 24) as f64)
                .collect();
            let d = vec![278.15; 1024];
            let s = vec![if hour % 24 < 12 { 400. * 3600. } else { 0. }; 1024];
            let r = vec![if hour % 17 == 0 { 0.003 } else { 0. }; 1024];
            gpu.hour(&mut cells, &t, &d, &s, &r).unwrap();
            reference.par_iter_mut().enumerate().for_each(|(i, c)| {
                let tc = t[i] - 273.15;
                let dc = d[i] - 273.15;
                let rh = (17.625 * dc / (243.04 + dc) - 17.625 * tc / (243.04 + tc))
                    .exp()
                    .clamp(0., 1.);
                c.nfdrs.hour(tc, rh, s[i] / 3600., r[i] * 1000.);
            });
        }
        let error = cells
            .iter()
            .zip(&reference)
            .flat_map(|(a, b)| {
                a.nfdrs
                    .profiles
                    .iter()
                    .flatten()
                    .zip(b.nfdrs.profiles.iter().flatten())
                    .map(|(a, b)| (a - b).abs())
            })
            .fold(0., f64::max);
        assert!(
            error < 1e-10,
            "CUDA/Rust absolute moisture-fraction difference {error}"
        );
    }
}
