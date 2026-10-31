//! The RRFS UPP-set extras on the GPU (`kernels/rrfs_upp.cu`, prebuilt
//! `kernels/rrfs_upp.ptx`; driver-only, no toolkit at run time).
//!
//! Three kernels: the column diagnostics (one thread per column), the
//! isobaric interpolation of extra 3-D fields, and the absolute vorticity
//! stencil.  The planes they return are host vectors in the exporter's
//! layout (x fastest).

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaFunction, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;

const PTX: &str = include_str!("../kernels/rrfs_upp.ptx");
const THREADS: u32 = 128;

/// Column-kernel input slots (`kernels/rrfs_upp.cu` V_*, S_*), output count.
pub const V_COUNT: usize = 13;
pub const S_COUNT: usize = 21;
pub const O_COUNT: usize = 35;

/// Output plane names, in kernel order (O_*).
pub const OUT_NAMES: [&str; O_COUNT] = [
    "lftx", "best_lftx", "mucape255", "mucin255", "plpl255", "cape_0_3km", "lfc_height", "el_height",
    "effective_srh", "esp", "critical_angle", "dendritic_depth", "rh_wrt_pw", "tcolw", "tcoli", "tcolp",
    "vil", "echo_top", "refl_minus10c", "refl_1km", "refl_4km", "u_pbl", "v_pbl", "ventilation_rate",
    "tcdc_pbl", "rh_freezing", "rh_highest_freezing", "pres_cloud_base", "pres_cloud_top", "tcdc_total",
    "cpofp", "crain", "csnow", "cicep", "cfrzr",
];

pub struct RrfsGpu {
    _ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    column: CudaFunction,
    isobaric: CudaFunction,
    absv: CudaFunction,
    pub label: String,
}

fn drv<E: std::fmt::Display>(what: &'static str) -> impl Fn(E) -> String {
    move |e| format!("{what}: {e}")
}

fn cfg(n: usize) -> LaunchConfig {
    LaunchConfig { grid_dim: ((n as u32).div_ceil(THREADS), 1, 1), block_dim: (THREADS, 1, 1), shared_mem_bytes: 0 }
}

impl RrfsGpu {
    pub fn open(ordinal: usize) -> Result<Self, String> {
        // SAFETY: probes for the driver library by name only.
        if !unsafe { cudarc::driver::sys::is_culib_present() } {
            return Err("no NVIDIA driver library (libcuda/nvcuda) could be loaded".into());
        }
        let ctx = CudaContext::new(ordinal).map_err(drv("open CUDA device"))?;
        let name = ctx.name().map_err(drv("device name"))?;
        let module = ctx.load_module(Ptx::from_src(PTX)).map_err(drv("load rrfs_upp.ptx"))?;
        let f = |n: &str| module.load_function(n).map_err(drv("resolve rrfs_upp kernel"));
        Ok(Self {
            stream: ctx.default_stream(),
            column: f("rrfs_upp_column_v1")?,
            isobaric: f("rrfs_upp_isobaric_v1")?,
            absv: f("rrfs_upp_absv_v1")?,
            label: format!("{name}: rrfs_upp.ptx (compute_60, JIT)"),
            _ctx: ctx,
        })
    }

    /// Column diagnostics.  `v3`: V_COUNT * nz * n, `vi`: 2 * (nz + 1) * n,
    /// `s2`: S_COUNT * n.  Returns O_COUNT * n.
    pub fn column(&self, nz: usize, n: usize, v3: &[f32], vi: &[f32], s2: &[f32]) -> Result<Vec<f32>, String> {
        let s = &self.stream;
        let d3 = s.clone_htod(v3).map_err(drv("upload column volumes"))?;
        let di = s.clone_htod(vi).map_err(drv("upload interfaces"))?;
        let d2 = s.clone_htod(s2).map_err(drv("upload column planes"))?;
        let mut out = s.alloc_zeros::<f32>(O_COUNT * n).map_err(drv("allocate column outputs"))?;
        let (nzi, ni) = (nz as i32, n as i32);
        let mut l = s.launch_builder(&self.column);
        l.arg(&nzi).arg(&ni).arg(&d3).arg(&di).arg(&d2).arg(&mut out);
        unsafe { l.launch(cfg(n)) }.map_err(drv("launch rrfs_upp_column_v1"))?;
        s.clone_dtoh(&out).map_err(drv("download column outputs"))
    }

    /// `fields` (nf * nz * n) to `levels` (Pa); returns [field][level][cell].
    pub fn isobaric(&self, nz: usize, n: usize, p: &[f32], fields: &[f32], nf: usize, levels: &[f64]) -> Result<Vec<f32>, String> {
        let s = &self.stream;
        let dp = s.clone_htod(p).map_err(drv("upload pressure"))?;
        let df = s.clone_htod(fields).map_err(drv("upload isobaric fields"))?;
        let dl = s.clone_htod(levels).map_err(drv("upload levels"))?;
        let mut out = s.alloc_zeros::<f32>(nf * levels.len() * n).map_err(drv("allocate isobaric outputs"))?;
        let (nzi, ni, nfi, nli) = (nz as i32, n as i32, nf as i32, levels.len() as i32);
        let mut l = s.launch_builder(&self.isobaric);
        l.arg(&nzi).arg(&ni).arg(&nfi).arg(&nli).arg(&dp).arg(&df).arg(&dl).arg(&mut out);
        unsafe { l.launch(cfg(n)) }.map_err(drv("launch rrfs_upp_isobaric_v1"))?;
        s.clone_dtoh(&out).map_err(drv("download isobaric outputs"))
    }

    /// Absolute vorticity of grid-relative winds on `nlev` planes.
    #[allow(clippy::too_many_arguments)]
    pub fn absv(&self, nx: usize, ny: usize, nlev: usize, dx: f64, u: &[f32], v: &[f32], mapf: &[f32], fcor: &[f32]) -> Result<Vec<f32>, String> {
        let s = &self.stream;
        let du = s.clone_htod(u).map_err(drv("upload u"))?;
        let dv = s.clone_htod(v).map_err(drv("upload v"))?;
        let dm = s.clone_htod(mapf).map_err(drv("upload map factor"))?;
        let df = s.clone_htod(fcor).map_err(drv("upload coriolis"))?;
        let total = nx * ny * nlev;
        let mut out = s.alloc_zeros::<f32>(total).map_err(drv("allocate vorticity"))?;
        let (nxi, nyi, nli) = (nx as i32, ny as i32, nlev as i32);
        let mut l = s.launch_builder(&self.absv);
        l.arg(&nxi).arg(&nyi).arg(&nli).arg(&dx).arg(&du).arg(&dv).arg(&dm).arg(&df).arg(&mut out);
        unsafe { l.launch(cfg(total)) }.map_err(drv("launch rrfs_upp_absv_v1"))?;
        s.clone_dtoh(&out).map_err(drv("download vorticity"))
    }
}
