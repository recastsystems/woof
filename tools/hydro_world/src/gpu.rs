//! The DEM per-cell passes (depression fill, D8 with flat resolution, flow
//! accumulation) on the GPU through cudarc (driver API, dynamically loaded;
//! kernels compiled by NVRTC at start with its defaults: no fast math, no
//! flush to zero). Results are bit-identical to flow.rs; `rw_hydro_world
//! hydrography-compare` proves it on a box. The DEM, the filled surface and
//! the directions stay on the card between passes; only the directions and
//! the accumulated areas come back.
use crate::flow::{FlatStats, AREA_UNITS_PER_M2};
use crate::grid::Grid;
use crate::Result;
use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use rayon::prelude::*;
use std::sync::Arc;
use std::time::Instant;

const SRC: &str = include_str!("cuda/hydrography.cu");
const TS: u32 = 32;
const BY: u32 = 8;
const THREADS: u32 = 256;

fn e<E: std::fmt::Debug>(what: &'static str) -> impl FnOnce(E) -> crate::Error {
    move |x| format!("CUDA {what}: {x:?}").into()
}

pub struct Gpu {
    pub device: String,
    pub compile_ms: f64,
    _ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
}

#[derive(Debug, Default, serde::Serialize)]
pub struct GpuFillStats {
    pub tile_size: u32,
    pub sweeps: u32,
    pub raised_cells: u64,
}

#[derive(Debug, Default, serde::Serialize)]
pub struct GpuAccumStats {
    pub land_cells: u64,
    pub source_cells: u64,
    pub longest_walk_cells: u64,
}

#[derive(Debug, Default, serde::Serialize)]
pub struct GpuRelaxStats {
    pub label_sweeps: u32,
    pub lower_distance_sweeps: u32,
    pub higher_distance_sweeps: u32,
}

pub struct Passes {
    pub dir: Vec<u8>,
    pub acc: Vec<u64>,
    /// Filled surface, only when asked for (the comparison command).
    pub filled: Option<Vec<f32>>,
    pub fill: GpuFillStats,
    pub flats: FlatStats,
    pub flat_sweeps: GpuRelaxStats,
    pub accum: GpuAccumStats,
    /// (stage, milliseconds) in order, each closed by a stream sync.
    pub stages: Vec<(String, f64)>,
}

/// NVRTC to a CUBIN for compute capability `cc` (defaults otherwise: no fast
/// math, no flush to zero, IEEE division and square root).
fn compile_cubin(src: &str, cc: (i32, i32)) -> Result<Vec<u8>> {
    if cudarc::gpuwm_hip::requested() {
        return cudarc::gpuwm_hip::compile(src, &["--std=c++17".into(),"--fmad=false".into()])
            .map_err(|e| -> crate::Error {e.into()});
    }
    use cudarc::nvrtc::{result, sys};
    use std::ffi::CString;
    let csrc = CString::new(src)?;
    let name = CString::new("hydrography.cu")?;
    let prog = result::create_program(&csrc, Some(&name)).map_err(e("NVRTC create"))?;
    let arch = format!("-arch=sm_{}{}", cc.0, cc.1);
    let out = (|| -> Result<Vec<u8>> {
        if let Err(x) = unsafe { result::compile_program(prog, std::slice::from_ref(&arch)) } {
            let log = unsafe { result::get_program_log(prog) }.map(|l| l.iter().map(|&c| c as u8 as char).collect::<String>()).unwrap_or_default();
            return Err(format!("NVRTC {arch}: {x:?}: {log}").into());
        }
        let mut size = 0usize;
        if unsafe { sys::nvrtcGetCUBINSize(prog, &mut size) } != sys::nvrtcResult::NVRTC_SUCCESS || size == 0 {
            return Err(format!("NVRTC {arch}: no CUBIN").into());
        }
        let mut buf = vec![0u8; size];
        if unsafe { sys::nvrtcGetCUBIN(prog, buf.as_mut_ptr() as *mut std::ffi::c_char) } != sys::nvrtcResult::NVRTC_SUCCESS {
            return Err(format!("NVRTC {arch}: CUBIN copy failed").into());
        }
        Ok(buf)
    })();
    unsafe { result::destroy_program(prog) }.ok();
    out
}

impl Gpu {
    /// First CUDA device; Err when there is no driver, no device or no NVRTC.
    pub fn new() -> Result<Gpu> {
        // cudarc's dynamic loader panics when a library is absent; turn that into an error.
        let r = std::panic::catch_unwind(|| -> Result<Gpu> {
            let ctx = CudaContext::new(0).map_err(e("context"))?;
            let device = ctx.name().map_err(e("device name"))?;
            let t = Instant::now();
            // CUBIN for the device's own architecture: the driver then needs no
            // PTX JIT, so an NVRTC newer than the driver still works.
            let cc = ctx.compute_capability().map_err(e("compute capability"))?;
            let module = match compile_cubin(SRC, cc) {
                Ok(cubin) => ctx.load_module(cudarc::nvrtc::Ptx::from_binary(cubin)).map_err(e("module load (cubin)"))?,
                Err(cubin_err) => {
                    let ptx = cudarc::nvrtc::compile_ptx_with_opts(SRC, cudarc::nvrtc::CompileOptions { name: Some("hydrography.cu".into()), ..Default::default() }).map_err(e("NVRTC compile"))?;
                    ctx.load_module(ptx).map_err(|x| -> crate::Error { format!("CUDA module load: cubin path failed ({cubin_err}); PTX path: {x:?}").into() })?
                }
            };
            let compile_ms = t.elapsed().as_secs_f64() * 1e3;
            let stream = ctx.default_stream();
            Ok(Gpu { device, compile_ms, _ctx: ctx, stream, module })
        });
        match r {
            Ok(x) => x,
            Err(p) => Err(format!("CUDA unavailable: {}", p.downcast_ref::<String>().cloned().or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_else(|| "driver or NVRTC library not found".into())).into()),
        }
    }

    fn f(&self, name: &str) -> Result<CudaFunction> {
        self.module.load_function(name).map_err(|x| format!("CUDA function {name}: {x:?}").into())
    }

    fn cells(n: usize) -> LaunchConfig {
        LaunchConfig { grid_dim: ((n as u32).div_ceil(THREADS), 1, 1), block_dim: (THREADS, 1, 1), shared_mem_bytes: 0 }
    }

    fn sync(&self) -> Result<()> {
        self.stream.synchronize().map_err(e("synchronize"))
    }

    fn scalar(&self, v: &CudaSlice<u32>) -> Result<u32> {
        Ok(self.stream.clone_dtoh(v).map_err(e("download"))?[0])
    }

    fn zero(&self) -> Result<CudaSlice<u32>> {
        self.stream.alloc_zeros::<u32>(1).map_err(e("alloc"))
    }

    /// Repeats a tile relaxation kernel until no tile changes. `launch`
    /// receives (act_in, act_out, changed) and pushes the kernel.
    fn relax(&self, g: &Grid, mut launch: impl FnMut(&CudaSlice<u32>, &mut CudaSlice<u32>, &mut CudaSlice<u32>, LaunchConfig) -> Result<()>) -> Result<u32> {
        let (tx, ty) = ((g.nx as u32).div_ceil(TS), (g.ny as u32).div_ceil(TS));
        let nt = (tx * ty) as usize;
        let cfg = LaunchConfig { grid_dim: (tx, ty, 1), block_dim: (TS, BY, 1), shared_mem_bytes: 0 };
        let mut a = self.stream.clone_htod(&vec![1u32; nt]).map_err(e("upload"))?;
        let mut b = self.stream.alloc_zeros::<u32>(nt).map_err(e("alloc"))?;
        let mut changed = self.zero()?;
        let mut sweeps = 0u32;
        loop {
            self.stream.memset_zeros(&mut b).map_err(e("memset"))?;
            self.stream.memset_zeros(&mut changed).map_err(e("memset"))?;
            launch(&a, &mut b, &mut changed, cfg)?;
            sweeps += 1;
            if self.scalar(&changed)? == 0 {
                return Ok(sweeps);
            }
            if sweeps > 1_000_000 {
                return Err("CUDA relaxation did not converge in 1,000,000 sweeps".into());
            }
            std::mem::swap(&mut a, &mut b);
        }
    }

    /// Fill, D8 + flats and accumulation for one grid. `keep_filled` also
    /// downloads the filled surface (for the CPU comparison).
    pub fn passes(&self, g: &Grid, z: &[f32], sea: &[bool], keep_filled: bool) -> Result<Passes> {
        let n = g.len();
        let (ny, nx) = (g.ny as i32, g.nx as i32);
        let nu = n as u32;
        let mut stages = Vec::new();
        let mut t = Instant::now();
        let lap = |name: &str, stages: &mut Vec<(String, f64)>, t: &mut Instant| {
            stages.push((name.to_string(), t.elapsed().as_secs_f64() * 1e3));
            *t = Instant::now();
        };
        let s = &self.stream;
        let sea8: Vec<u8> = sea.par_iter().map(|&b| b as u8).collect();
        let dz = s.clone_htod(z).map_err(e("upload DEM"))?;
        let dsea = s.clone_htod(&sea8).map_err(e("upload sea mask"))?;
        drop(sea8);
        let dist: Vec<f64> = (0..g.ny)
            .flat_map(|r| {
                let (dx, dy) = g.spacing_m(r);
                [dx, (dx * dx + dy * dy).sqrt(), dy]
            })
            .collect();
        let ddist = s.clone_htod(&dist).map_err(e("upload spacing"))?;
        let area: Vec<u64> = (0..g.ny).map(|r| (g.cell_area_m2(r) * AREA_UNITS_PER_M2).round() as u64).collect();
        let darea = s.clone_htod(&area).map_err(e("upload areas"))?;
        self.sync()?;
        lap("upload DEM + sea mask + row tables", &mut stages, &mut t);

        // 1. Fill.
        let mut w = s.alloc_zeros::<f32>(n).map_err(e("alloc"))?;
        let mut fixed = s.alloc_zeros::<u8>(n).map_err(e("alloc"))?;
        {
            let f = self.f("fill_init")?;
            let mut l = s.launch_builder(&f);
            l.arg(&dz).arg(&dsea).arg(&mut w).arg(&mut fixed).arg(&ny).arg(&nx);
            unsafe { l.launch(Self::cells(n)) }.map_err(e("fill_init"))?;
        }
        let relax_fill = self.f("fill_relax")?;
        let sweeps = self.relax(g, |a, b, ch, cfg| {
            let mut l = s.launch_builder(&relax_fill);
            l.arg(&dz).arg(&fixed).arg(&mut w).arg(&ny).arg(&nx).arg(a).arg(b).arg(ch);
            unsafe { l.launch(cfg) }.map_err(e("fill_relax"))?;
            Ok(())
        })?;
        drop(fixed);
        let (mut unreached, mut raised) = (self.zero()?, self.zero()?);
        {
            let f = self.f("fill_check")?;
            let mut l = s.launch_builder(&f);
            l.arg(&dz).arg(&w).arg(&nu).arg(&mut unreached).arg(&mut raised);
            unsafe { l.launch(Self::cells(n)) }.map_err(e("fill_check"))?;
        }
        let un = self.scalar(&unreached)?;
        if un > 0 {
            return Err(format!("fill: {un} cells with no spill path to an outlet").into());
        }
        let fill = GpuFillStats { tile_size: TS, sweeps, raised_cells: self.scalar(&raised)? as u64 };
        lap("depression fill (minimax relaxation)", &mut stages, &mut t);
        drop(dz);

        // 2. D8.
        let mut dir = s.alloc_zeros::<u8>(n).map_err(e("alloc"))?;
        {
            let f = self.f("d8")?;
            let mut l = s.launch_builder(&f);
            l.arg(&w).arg(&dsea).arg(&ddist).arg(&mut dir).arg(&ny).arg(&nx);
            unsafe { l.launch(Self::cells(n)) }.map_err(e("d8"))?;
        }
        self.sync()?;
        lap("D8 steepest descent", &mut stages, &mut t);
        drop(dsea);

        // 3. Flats.
        let mut nm = s.alloc_zeros::<u8>(n).map_err(e("alloc"))?;
        let mut lab = s.alloc_zeros::<u32>(n).map_err(e("alloc"))?;
        let mut dl = s.alloc_zeros::<u32>(n).map_err(e("alloc"))?;
        let mut dh = s.alloc_zeros::<u32>(n).map_err(e("alloc"))?;
        let mut und = self.zero()?;
        {
            let f = self.f("flat_prep")?;
            let mut l = s.launch_builder(&f);
            l.arg(&w).arg(&dir).arg(&mut nm).arg(&mut lab).arg(&mut dl).arg(&mut dh).arg(&ny).arg(&nx).arg(&mut und);
            unsafe { l.launch(Self::cells(n)) }.map_err(e("flat_prep"))?;
        }
        let undrained = self.scalar(&und)? as usize;
        let relax_u = self.f("relax_u32")?;
        let run = |v: &mut CudaSlice<u32>, inc: u32| -> Result<u32> {
            if undrained == 0 {
                return Ok(0);
            }
            self.relax(g, |a, b, ch, cfg| {
                let mut l = s.launch_builder(&relax_u);
                l.arg(&mut *v).arg(&nm).arg(&inc).arg(&ny).arg(&nx).arg(a).arg(b).arg(ch);
                unsafe { l.launch(cfg) }.map_err(e("relax_u32"))?;
                Ok(())
            })
        };
        let flat_sweeps = GpuRelaxStats { label_sweeps: run(&mut lab, 0)?, lower_distance_sweeps: run(&mut dl, 1)?, higher_distance_sweeps: run(&mut dh, 1)? };
        let mut cnt = s.alloc_zeros::<u32>(n).map_err(e("alloc"))?;
        let mut hmax = s.alloc_zeros::<u32>(n).map_err(e("alloc"))?;
        let (mut nflats, mut no_outlet, mut largest) = (self.zero()?, self.zero()?, self.zero()?);
        let mut first_bad = s.clone_htod(&[u32::MAX]).map_err(e("upload"))?;
        {
            let f = self.f("flat_stats")?;
            let mut l = s.launch_builder(&f);
            l.arg(&dir).arg(&lab).arg(&dl).arg(&dh).arg(&nu).arg(&mut cnt).arg(&mut hmax).arg(&mut nflats).arg(&mut no_outlet).arg(&mut first_bad);
            unsafe { l.launch(Self::cells(n)) }.map_err(e("flat_stats"))?;
        }
        {
            let f = self.f("flat_largest")?;
            let mut l = s.launch_builder(&f);
            l.arg(&lab).arg(&cnt).arg(&nu).arg(&mut largest);
            unsafe { l.launch(Self::cells(n)) }.map_err(e("flat_largest"))?;
        }
        drop(cnt);
        let no = self.scalar(&no_outlet)?;
        if no > 0 {
            let c = self.scalar(&first_bad)?;
            return Err(format!("flat at cell {c}: {no} undrained cells have no outlet at their own elevation").into());
        }
        let flats = FlatStats { undrained_cells: undrained, flats: self.scalar(&nflats)? as usize, largest_flat_cells: self.scalar(&largest)? as usize };
        let mut out = s.alloc_zeros::<u8>(n).map_err(e("alloc"))?;
        s.memcpy_dtod(&dir, &mut out).map_err(e("copy"))?;
        let (mut no_lower, mut bad2) = (self.zero()?, s.clone_htod(&[u32::MAX]).map_err(e("upload"))?);
        {
            let f = self.f("flat_dir")?;
            let mut l = s.launch_builder(&f);
            l.arg(&w).arg(&dir).arg(&nm).arg(&lab).arg(&dl).arg(&dh).arg(&hmax).arg(&mut out).arg(&ny).arg(&nx).arg(&mut no_lower).arg(&mut bad2);
            unsafe { l.launch(Self::cells(n)) }.map_err(e("flat_dir"))?;
        }
        let nl = self.scalar(&no_lower)?;
        if nl > 0 {
            return Err(format!("flats: {nl} cells have no lower mask neighbour (first {})", self.scalar(&bad2)?).into());
        }
        drop((nm, lab, dl, dh, hmax, dir));
        lap("flat resolution (labels, two distances, directions)", &mut stages, &mut t);
        let filled = if keep_filled { Some(s.clone_dtoh(&w).map_err(e("download filled"))?) } else { None };
        drop(w);

        // 4. Accumulation.
        let mut acc = s.alloc_zeros::<u64>(n).map_err(e("alloc"))?;
        let mut indeg = s.alloc_zeros::<u32>(n).map_err(e("alloc"))?;
        let mut source = s.alloc_zeros::<u8>(n).map_err(e("alloc"))?;
        let (mut land, mut processed, mut starts, mut longest) = (self.zero()?, self.zero()?, self.zero()?, self.zero()?);
        {
            let f = self.f("acc_init")?;
            let mut l = s.launch_builder(&f);
            l.arg(&out).arg(&darea).arg(&mut acc).arg(&mut indeg).arg(&mut source).arg(&ny).arg(&nx).arg(&mut land);
            unsafe { l.launch(Self::cells(n)) }.map_err(e("acc_init"))?;
        }
        {
            let f = self.f("acc_walk")?;
            let mut l = s.launch_builder(&f);
            l.arg(&out).arg(&mut acc).arg(&mut indeg).arg(&source).arg(&ny).arg(&nx).arg(&mut processed).arg(&mut starts).arg(&mut longest);
            unsafe { l.launch(Self::cells(n)) }.map_err(e("acc_walk"))?;
        }
        let (landn, proc_n) = (self.scalar(&land)?, self.scalar(&processed)?);
        if proc_n != landn {
            return Err(format!("flow directions contain a cycle: {} of {landn} land cells never drained", landn - proc_n).into());
        }
        let accum = GpuAccumStats { land_cells: landn as u64, source_cells: self.scalar(&starts)? as u64, longest_walk_cells: self.scalar(&longest)? as u64 };
        lap("flow accumulation (one walk per source, last arrival continues)", &mut stages, &mut t);
        let dir_h = s.clone_dtoh(&out).map_err(e("download directions"))?;
        let acc_h = s.clone_dtoh(&acc).map_err(e("download accumulation"))?;
        lap("download directions + accumulation", &mut stages, &mut t);
        Ok(Passes { dir: dir_h, acc: acc_h, filled, fill, flats, flat_sweeps, accum, stages })
    }
}
