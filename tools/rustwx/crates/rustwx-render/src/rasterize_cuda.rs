//! GPU fill for structured maps, seeded from the project's rustwx-production-cuda
//! rasterize_grid.cu. Binary64, no contraction, same masks and palette lookup.
//! One shared low-priority nonblocking stream bounds concurrent GPU allocation
//! even when the caller draws many products on CPU workers. No device-wide wait.
use std::sync::{Arc, Mutex, OnceLock};
use std::sync::atomic::{AtomicUsize, Ordering};
use cudarc::{driver::{CudaContext, CudaFunction, CudaStream, LaunchConfig, PushKernelArg}, nvrtc::Ptx};
use crate::{LeveledColormap, RasterSampleMode, Rgba, RgbaImage};

const BUDGET: usize = 256 * 1024 * 1024;
static CALLS: AtomicUsize = AtomicUsize::new(0);
static FAILURES: AtomicUsize = AtomicUsize::new(0);
static DEVICE: OnceLock<Mutex<Result<Renderer, String>>> = OnceLock::new();

pub(crate) fn stats() -> [(&'static str, usize); 2] {
    [("structured raster CUDA OK", CALLS.load(Ordering::Relaxed)),
     ("structured raster CUDA failed", FAILURES.load(Ordering::Relaxed))]
}

struct Renderer {
    stream: Arc<CudaStream>,
    function: CudaFunction,
    robinson: CudaFunction,
}
impl Renderer {
    /// Opens the device `RUSTWX_RASTER_DEVICE=<n>` names, else visible
    /// device 0 (the first card CUDA_VISIBLE_DEVICES exposes), and says
    /// which card the fills run on. `RUSTWX_RASTER_DEVICE=cpu` never gets here.
    fn open() -> Result<Self, String> {
        if !unsafe { cudarc::driver::sys::is_culib_present() } { return Err("no CUDA driver".into()); }
        let ordinal = match std::env::var("RUSTWX_RASTER_DEVICE") {
            Ok(value) if !value.trim().is_empty() => value.trim().parse::<usize>().map_err(|_| {
                let message = format!("RUSTWX_RASTER_DEVICE={value:?} is neither cpu nor a CUDA device number");
                eprintln!("Rust renderer: {message}; raster fills run on the CPU");
                message
            })?,
            _ => 0,
        };
        let ctx = CudaContext::new(ordinal).map_err(|e| e.to_string())?;
        let visible = std::env::var("CUDA_VISIBLE_DEVICES").unwrap_or_else(|_| "unset".into());
        eprintln!("Rust renderer: raster fills on selected GPU device {ordinal} ({}, CUDA_VISIBLE_DEVICES={visible}); \
                   RUSTWX_RASTER_DEVICE=cpu keeps them on the CPU",
                  ctx.name().unwrap_or_else(|_| "unnamed".into()));
        let (least, _) = cudarc::driver::result::stream::get_priority_range().map_err(|e| e.to_string())?;
        let stream = ctx.new_stream_with_priority(least).map_err(|e| e.to_string())?;
        let image=if cudarc::gpuwm_hip::requested() {
            Ptx::from_binary(cudarc::gpuwm_hip::compile(include_str!("rasterize_structured.cu"),&["--std=c++17".into(),"--fmad=false".into(),"--ftz=false".into()])?)
        } else {Ptx::from_src(include_str!("rasterize_structured.ptx"))};
        let module=ctx.load_module(image).map_err(|e|e.to_string())?;
        let function = module.load_function("rasterize_structured").map_err(|e| e.to_string())?;
        let robinson = module.load_function("rasterize_robinson").map_err(|e| e.to_string())?;
        Ok(Self { stream, function, robinson })
    }
    fn render(&self, data: &[f64], ny: usize, nx: usize, cmap: &LeveledColormap,
              sample: RasterSampleMode, width: u32, height: u32,
              axes: Option<[f64; 4]>, inverse: Option<&[f64; 16]>) -> Result<RgbaImage, String> {
        let run = || -> Result<RgbaImage, cudarc::driver::DriverError> {
            let stream = &self.stream;
            stream.context().bind_to_thread()?;
            let values = stream.clone_htod(data)?;
            // Driver allocations accept zero-length slices; an empty palette
            // produces transparent pixels in the kernel before dereferencing.
            let levels = stream.clone_htod(&cmap.levels)?;
            let colors: Vec<u32> = cmap.colors.iter().map(pack).collect();
            let palette = stream.clone_htod(&colors)?;
            // Every in-range thread writes its output, including transparent pixels.
            let mut out = unsafe { stream.alloc::<u32>(width as usize * height as usize) }?;
            let (ny, nx, w, h) = (ny as i32, nx as i32, width as i32, height as i32);
            let (nl, nc) = (cmap.levels.len() as i32, colors.len() as i32);
            let (hu, ho, hm) = (i32::from(cmap.under_color.is_some()), i32::from(cmap.over_color.is_some()), i32::from(cmap.mask_below.is_some()));
            let (under, over) = (cmap.under_color.as_ref().map(pack).unwrap_or(0), cmap.over_color.as_ref().map(pack).unwrap_or(0));
            let mask = cmap.mask_below.unwrap_or(0.0);
            let axis_mode = i32::from(axes.is_some());
            let [x0, y0, dx, dy] = axes.unwrap_or([0.0, 0.0, 1.0, 1.0]);
            let nearest = i32::from(matches!(sample, RasterSampleMode::Nearest));
            let cfg = LaunchConfig { grid_dim: ((width * height).div_ceil(256), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
            let inverse_device = inverse.map(|q| stream.clone_htod(q)).transpose()?;
            unsafe {
                let function = if inverse.is_some() { &self.robinson } else { &self.function };
                let mut launch = stream.launch_builder(function);
                launch.arg(&values).arg(&ny).arg(&nx).arg(&levels).arg(&nl).arg(&palette).arg(&nc)
                    .arg(&hu).arg(&under).arg(&ho).arg(&over).arg(&hm).arg(&mask)
                    .arg(&mut out).arg(&w).arg(&h);
                if let Some(q) = &inverse_device { launch.arg(q).arg(&nearest).launch(cfg)?; }
                else { launch.arg(&axis_mode).arg(&x0).arg(&y0).arg(&dx).arg(&dy).arg(&nearest).launch(cfg)?; }
            }
            // Packed RGBA on the CUDA device is already the image byte layout.
            // Read that byte view directly instead of repacking every pixel on
            // the host. u8 has no alignment or invalid-bit-pattern restriction.
            let byte_view = unsafe { out.transmute::<u8>(out.len() * 4) }.expect("full allocation byte view");
            let bytes = stream.clone_dtoh(&byte_view)?;
            Ok(RgbaImage::from_raw(width, height, bytes).expect("checked image extent"))
        };
        run().map_err(|e| e.to_string())
    }
}
fn pack(c: &Rgba) -> u32 { u32::from_le_bytes([c.r, c.g, c.b, c.a]) }

pub(crate) fn render(data: &[f64], ny: usize, nx: usize, cmap: &LeveledColormap,
                     sample: RasterSampleMode, width: u32, height: u32,
                     axes: Option<[f64; 4]>) -> Option<RgbaImage> {
    render_inner(data, ny, nx, cmap, sample, width, height, axes, None)
}

pub(crate) fn render_inverse(data: &[f64], ny: usize, nx: usize, cmap: &LeveledColormap,
    sample: RasterSampleMode, width: u32, height: u32, inverse: &[f64;16]) -> Option<RgbaImage> {
    render_inner(data, ny, nx, cmap, sample, width, height, None, Some(inverse))
}

fn render_inner(data: &[f64], ny: usize, nx: usize, cmap: &LeveledColormap,
    sample: RasterSampleMode, width: u32, height: u32, axes: Option<[f64;4]>, inverse: Option<&[f64;16]>) -> Option<RgbaImage> {
    if std::env::var_os("RUSTWX_RASTER_DEVICE").as_deref() == Some(std::ffi::OsStr::new("cpu")) { return None; }
    // These are the signed indexing and allocation limits of this kernel.
    let n = ny.checked_mul(nx)?;
    let pixels = (width as usize).checked_mul(height as usize)?;
    let bytes = n.checked_mul(8)?.checked_add(pixels.checked_mul(8)?)?
        .checked_add(128)?.checked_add(cmap.levels.len().checked_mul(8)?)?.checked_add(cmap.colors.len().checked_mul(8)?)?;
    if ny < 2 || nx < 2 || data.len() != n || pixels == 0 || n > i32::MAX as usize
        || pixels > i32::MAX as usize || bytes > BUDGET { return None; }
    let mut device = DEVICE.get_or_init(|| Mutex::new(Renderer::open())).lock().ok()?;
    let result = match device.as_ref() {
        Ok(renderer) => renderer.render(data, ny, nx, cmap, sample, width, height, axes, inverse),
        Err(_) => return None,
    };
    match result {
        Ok(img) => { CALLS.fetch_add(1, Ordering::Relaxed); Some(img) }
        Err(error) => {
            FAILURES.fetch_add(1, Ordering::Relaxed);
            eprintln!("Rust renderer CUDA fill failed; continuing on CPU: {error}");
            *device = Err(error);
            None
        }
    }
}
