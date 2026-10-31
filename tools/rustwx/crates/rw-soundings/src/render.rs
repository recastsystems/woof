//! Headless rendering of the sharppyrs SPC sounding window (and the
//! skew-T-only widget) through egui_kittest's wgpu renderer, plus the
//! ensemble overlays sharppyrs does not have yet (member spaghetti and a
//! percentile envelope on the skew-T).
//!
//! Each render thread owns one Harness (egui contexts are not Send). Hosts
//! with no usable GPU device (a headless server without the Vulkan loader)
//! draw through [`CpuRenderer`], a rasteriser of egui's own meshes.

use egui_wgpu::wgpu;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::time::Instant;

use egui::{Align2, Color32, Pos2, Rect, Shape, Stroke, Vec2};
use sharppyrs::{
    CornerPanel, DiagnosticTablePanelKind, NativeDiagnosticPatchBoard, ParcelType, SkewTStyle,
    SoundingLayout,
};

use crate::analysis::Analysis;

#[derive(Clone, Debug)]
pub enum ViewKind {
    Window {
        omega: Option<bool>,
        interp_winds: bool,
    },
    SkewT {
        omega: Option<bool>,
        interp_winds: bool,
    },
}

#[derive(Clone, Debug, Default)]
pub struct MemberTrace {
    pub label: String,
    pub p: Vec<f64>,
    pub t: Vec<f64>,
    pub td: Vec<f64>,
    /// hodograph (u, v) kt, 0-10 km AGL
    pub hodo: Vec<(f64, f64)>,
}

/// Member percentiles per level: index 0..5 = p10, p25, p50, p75, p90.
#[derive(Clone, Debug, Default)]
pub struct Bands {
    pub p: Vec<f64>,
    pub t: [Vec<f64>; 5],
    pub td: [Vec<f64>; 5],
    /// median hodograph (u, v) kt, 0-10 km AGL
    pub hodo_med: Vec<(f64, f64)>,
}

#[derive(Clone, Debug)]
pub struct Overlay {
    pub members: Vec<MemberTrace>,
    pub bands: Option<Bands>,
    /// envelope mode: p10-p90 and p25-p75 bands; otherwise only p10-p90
    pub inner_band: bool,
    pub show_t: bool,
    pub show_td: bool,
    pub alpha: u8,
    pub legend: String,
}

pub struct SoundingScene {
    pub analysis: Arc<Analysis>,
    pub view: ViewKind,
    pub layout: SoundingLayout,
    pub corner: CornerPanel,
    pub parcel: ParcelType,
    pub title: String,
    pub brand: String,
    pub style: SkewTStyle,
    pub patches: NativeDiagnosticPatchBoard,
    pub overlay: Option<Overlay>,
}

pub enum Job {
    Sounding(SoundingScene),
}

pub struct RenderRequest {
    pub job: Arc<Job>,
    pub size: Vec2,
    pub ppp: f32,
    pub reply: tokio::sync::oneshot::Sender<Result<Rendered, String>>,
}

pub struct Rendered {
    pub png: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub frame_ms: f64,
    pub encode_ms: f64,
}

struct RState {
    fonts: bool,
    job: Option<Arc<Job>>,
}

/// Own the render workers until they have finished. Detached workers raced
/// NVIDIA's process teardown after the last PNG was written.
pub struct RenderPool {
    sender: Option<Sender<RenderRequest>>,
    workers: Vec<std::thread::JoinHandle<()>>,
}

impl RenderPool {
    pub fn send(
        &self,
        request: RenderRequest,
    ) -> Result<(), std::sync::mpsc::SendError<RenderRequest>> {
        self.sender.as_ref().unwrap().send(request)
    }
}

impl Drop for RenderPool {
    fn drop(&mut self) {
        self.sender.take();
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

/// The device every render thread draws with, chosen once per process.
#[derive(Clone, Debug)]
pub enum RenderDevice {
    /// A wgpu adapter that created a device; `backends` recreates it per thread.
    Gpu {
        backends: wgpu::Backends,
        adapter: String,
    },
    /// The CPU rasteriser in this file; `reason` says why no GPU device was used.
    Cpu { reason: String },
}

impl std::fmt::Display for RenderDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RenderDevice::Gpu { adapter, .. } => write!(f, "gpu {adapter}"),
            RenderDevice::Cpu { reason } => write!(f, "cpu rasteriser ({reason})"),
        }
    }
}

/// `GPUWM_SOUNDINGS_RENDERER`: `auto` (default: the CPU rasteriser for a
/// batch of at most [`CPU_PICTURES_PER_THREAD`] pictures per CPU render
/// thread, otherwise GPU, else the CPU rasteriser), `gpu` (refuse without a
/// GPU device) or `cpu`.
pub const RENDERER_ENV: &str = "GPUWM_SOUNDINGS_RENDERER";

/// `auto` draws a batch of at most this many pictures per CPU render thread
/// on the CPU rasteriser without creating a GPU device at all (32 pictures
/// with the default 4 threads). Breakage this prevents: on AWS g7e boxes
/// (NVIDIA RTX PRO 6000 Blackwell Server Edition, 2026-10-28, WOOF runs
/// run_e44b7d322ff71018 and run_a7e4a0df78e5b9f8) creating the Vulkan device
/// took 69 s and about 220 s, then 6 pictures drew in 0.3 s; every
/// `gpuwm soundings` call paid it. On a warm, idle RTX PRO 6000 (2026-10-28)
/// the device costs about 0.8 s (create and exit) and then draws about 19 ms
/// a picture on its 2 threads; the CPU rasteriser takes about 160 ms a
/// picture per thread, so up to 8 pictures a thread the CPU is not slower
/// even where the device is quick (6 pictures: 1.38 s on the GPU, 0.52 s on
/// 4 CPU threads).
pub const CPU_PICTURES_PER_THREAD: usize = 8;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Mode {
    Auto,
    Gpu,
    Cpu,
}

fn mode() -> Result<Mode, String> {
    let mode = std::env::var(RENDERER_ENV).unwrap_or_default();
    match mode.trim().to_ascii_lowercase().as_str() {
        "" | "auto" => Ok(Mode::Auto),
        "gpu" => Ok(Mode::Gpu),
        "cpu" => Ok(Mode::Cpu),
        other => Err(format!("{RENDERER_ENV}={other}: use auto, gpu or cpu")),
    }
}

/// The device a batch of `pictures` drawn on `threads` CPU render threads
/// uses: under `auto` a small batch stays on the CPU rasteriser and no GPU
/// device is created (see [`CPU_PICTURES_PER_THREAD`]); a larger one, or
/// `gpu`/`cpu`, goes to [`render_device`].
pub fn batch_device(pictures: usize, threads: usize) -> Result<RenderDevice, String> {
    let cpu_max = CPU_PICTURES_PER_THREAD * threads.max(1);
    if mode()? == Mode::Auto && pictures <= cpu_max {
        return Ok(RenderDevice::Cpu {
            reason: format!(
                "auto: {pictures} pictures; up to {cpu_max} draw on {threads} CPU threads \
                 without starting a GPU device"
            ),
        });
    }
    render_device()
}

/// What `--probe-device` says: the device [`render_device`] creates, and
/// under `auto` the small-batch rule of [`batch_device`].
pub fn probe_line() -> Result<String, String> {
    let device = render_device()?;
    Ok(match (mode()?, &device) {
        (Mode::Auto, RenderDevice::Gpu { .. }) => format!(
            "{device}; batches of up to {CPU_PICTURES_PER_THREAD} pictures per CPU thread \
             draw on the cpu rasteriser"
        ),
        _ => device.to_string(),
    })
}

static DEVICE: std::sync::OnceLock<Result<RenderDevice, String>> = std::sync::OnceLock::new();

/// Pick the render device once: Vulkan/Metal/DX12, then GL, then the CPU
/// rasteriser. Breakage this prevents: on a headless NVIDIA server without
/// the Vulkan loader (libvulkan1) wgpu's default choice was GL/EGL, whose
/// device request fails with Device(Lost), and egui_kittest panicked inside
/// every render thread.
pub fn render_device() -> Result<RenderDevice, String> {
    DEVICE.get_or_init(select_device).clone()
}

fn select_device() -> Result<RenderDevice, String> {
    let mode = mode()?;
    if mode == Mode::Cpu {
        return Ok(RenderDevice::Cpu {
            reason: format!("{RENDERER_ENV}=cpu"),
        });
    }
    let mut tried = Vec::new();
    for backends in [
        wgpu::Backends::PRIMARY - wgpu::Backends::BROWSER_WEBGPU,
        wgpu::Backends::GL,
    ] {
        match create_render_state(backends) {
            Ok(state) => {
                let info = state.adapter.get_info();
                // Never vkDestroyDevice: see `render_thread` on queue close.
                std::mem::forget(state);
                return Ok(RenderDevice::Gpu {
                    backends,
                    adapter: format!("{} {:?} {:?}", info.name, info.backend, info.device_type),
                });
            }
            Err(e) => tried.push(format!("{backends:?}: {e}")),
        }
    }
    let mut reason = format!("no GPU render device; {}", tried.join("; "));
    if !vulkan_loader_present() {
        reason.push_str(
            "; the Vulkan loader libvulkan.so.1 is not installed \
             (apt-get install libvulkan1) for GPU rendering",
        );
    }
    if mode == Mode::Gpu {
        Err(format!("{RENDERER_ENV}=gpu: {reason}"))
    } else {
        Ok(RenderDevice::Cpu { reason })
    }
}

#[cfg(target_os = "linux")]
fn vulkan_loader_present() -> bool {
    // SAFETY: loading the system Vulkan loader runs no user code beyond its
    // initialisers, the same library wgpu itself opens.
    unsafe { libloading::Library::new("libvulkan.so.1") }.is_ok()
}

#[cfg(not(target_os = "linux"))]
fn vulkan_loader_present() -> bool {
    true
}

fn create_render_state(backends: wgpu::Backends) -> Result<egui_wgpu::RenderState, String> {
    let mut setup = egui_wgpu::WgpuSetupCreateNew::without_display_handle();
    setup.instance_descriptor.backends = backends;
    setup.power_preference = wgpu::PowerPreference::HighPerformance;
    let setup = egui_wgpu::WgpuSetup::CreateNew(setup);
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let instance = pollster::block_on(setup.new_instance());
        pollster::block_on(egui_wgpu::RenderState::create(
            &egui_wgpu::WgpuConfiguration {
                wgpu_setup: setup,
                ..Default::default()
            },
            &instance,
            None,
            egui_wgpu::RendererOptions::PREDICTABLE,
        ))
        .map_err(|e| e.to_string())
    }))
    .unwrap_or_else(|p| Err(panic_text(&p)))
}

fn panic_text(p: &Box<dyn std::any::Any + Send>) -> String {
    p.downcast_ref::<String>()
        .cloned()
        .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_else(|| "render panic".into())
}

/// Spawn `n` render threads sharing one queue, all drawing on `device`
/// (from [`batch_device`] or [`render_device`]).
pub fn spawn_pool(device: RenderDevice, n: usize) -> Result<RenderPool, String> {
    eprintln!("soundings: rendering on {device} with {} threads", n.max(1));
    let (tx, rx) = channel::<RenderRequest>();
    let rx = Arc::new(std::sync::Mutex::new(rx));
    let mut workers = Vec::new();
    for i in 0..n.max(1) {
        let rx = rx.clone();
        let device = device.clone();
        workers.push(
            std::thread::Builder::new()
                .name(format!("sharppyrs-render-{i}"))
                .spawn(move || render_thread(rx, device))
                .expect("spawn render thread"),
        );
    }
    Ok(RenderPool {
        sender: Some(tx),
        workers,
    })
}

enum AnyRenderer {
    Gpu(egui_kittest::wgpu::WgpuTestRenderer),
    Cpu(CpuRenderer),
}

impl egui_kittest::TestRenderer for AnyRenderer {
    fn handle_delta(&mut self, delta: &egui::TexturesDelta) {
        match self {
            AnyRenderer::Gpu(r) => r.handle_delta(delta),
            AnyRenderer::Cpu(r) => r.handle_delta(delta),
        }
    }
    fn render(
        &mut self,
        ctx: &egui::Context,
        output: &egui::FullOutput,
    ) -> Result<image::RgbaImage, String> {
        match self {
            AnyRenderer::Gpu(r) => r.render(ctx, output),
            AnyRenderer::Cpu(r) => r.render(ctx, output),
        }
    }
}

fn build_harness(
    ppp: f32,
    device: &RenderDevice,
) -> Result<egui_kittest::Harness<'static, RState>, String> {
    // The pool's device: the CPU rasteriser for a small batch under `auto`
    // (`batch_device`), else high-performance GPU first (`render_device`).
    let renderer = match device {
        RenderDevice::Gpu { backends, .. } => AnyRenderer::Gpu(
            egui_kittest::wgpu::WgpuTestRenderer::from_render_state(create_render_state(
                *backends,
            )?),
        ),
        RenderDevice::Cpu { .. } => AnyRenderer::Cpu(CpuRenderer::default()),
    };
    let mut h = egui_kittest::Harness::builder()
        .with_size(Vec2::new(1630.0, 1100.0))
        .with_pixels_per_point(ppp)
        .with_max_steps(8)
        .renderer(renderer)
        .build_ui_state(
            |ui: &mut egui::Ui, st: &mut RState| {
                if !st.fonts {
                    sharppyrs::install_fonts(ui.ctx());
                    st.fonts = true;
                    return;
                }
                if let Some(job) = st.job.clone() {
                    draw_job(ui, &job);
                }
            },
            RState {
                fonts: false,
                job: None,
            },
        );
    h.run_steps(2);
    Ok(h)
}

/// One Harness per pixels-per-point value (switching ppp on a live context
/// only lands a frame late, which produced wrong-size images). A failure
/// (device or render) is replied to the request as a plain error, never a
/// panic that closes the queue.
fn render_thread(rx: Arc<std::sync::Mutex<Receiver<RenderRequest>>>, device: RenderDevice) {
    let mut harnesses: std::collections::HashMap<u32, egui_kittest::Harness<'static, RState>> =
        std::collections::HashMap::new();
    loop {
        let req = {
            let guard = rx.lock().unwrap_or_else(|e| e.into_inner());
            match guard.recv() {
                Ok(r) => r,
                Err(_) => {
                    // Every picture is written; leave the device contexts to process exit.
                    // Breakage this prevents: NVIDIA's vkDestroyDevice (driver 610.43) at times
                    // never returns while its helper thread spins one core at 100%, so
                    // rw_soundings hung after its last PNG (3 of 25 runs on a busy RTX 5090;
                    // seen as 2+ minutes at 101% of one core on a hex export).
                    harnesses.drain().for_each(|(_, h)| std::mem::forget(h));
                    return;
                }
            }
        };
        let t0 = Instant::now();
        let key = ((req.ppp * 4.0).round() * 25.0) as u32;
        let out = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            if !harnesses.contains_key(&key) {
                harnesses.insert(key, build_harness(key as f32 / 100.0, &device)?);
            }
            let h = harnesses.get_mut(&key).unwrap();
            h.state_mut().job = Some(req.job.clone());
            h.set_size(req.size);
            h.run_steps(2);
            let out = h.render();
            h.state_mut().job = None;
            out
        }))
        .unwrap_or_else(|p| {
            harnesses.remove(&key);
            Err(format!("sounding render failed: {}", panic_text(&p)))
        });
        let frame_ms = t0.elapsed().as_secs_f64() * 1e3;
        let res = out.and_then(|img| {
            let t1 = Instant::now();
            let (w, hh) = (img.width(), img.height());
            let mut buf = std::io::Cursor::new(Vec::new());
            img.write_to(&mut buf, image::ImageFormat::Png)
                .map_err(|e| e.to_string())?;
            Ok(Rendered {
                png: buf.into_inner(),
                width: w,
                height: hh,
                frame_ms,
                encode_ms: t1.elapsed().as_secs_f64() * 1e3,
            })
        });
        let _ = req.reply.send(res);
    }
}

fn draw_job(ui: &mut egui::Ui, job: &Job) {
    // cover the harness panel margin with the sounding background
    let Job::Sounding(s) = job;
    let bg = s.style.bg_color;
    let screen = ui.ctx().content_rect();
    ui.ctx()
        .layer_painter(egui::LayerId::background())
        .rect_filled(screen, 0.0, bg);
    draw_sounding(ui, s);
}

fn draw_sounding(ui: &mut egui::Ui, s: &SoundingScene) {
    let size = ui.available_size();
    let a = &s.analysis;
    let layout_id = egui::Id::new("gpuwm-sharppyrs-layout");
    let resp = egui::Frame::new()
        .fill(s.style.bg_color)
        .show(ui, |ui| {
            ui.set_min_size(size);
            match &s.view {
                ViewKind::Window {
                    omega,
                    interp_winds,
                } => {
                    sharppyrs::store_layout(ui.ctx(), layout_id, &s.layout);
                    let mut view = sharppyrs::SoundingView::new(&a.prof, &a.derived)
                        .title(s.title.clone())
                        .brand(s.brand.clone())
                        .style(s.style.clone())
                        .parcel(s.parcel)
                        .corner_panel(s.corner)
                        .interactive(false)
                        .layout_memory_id(layout_id)
                        .interp_winds(*interp_winds)
                        .size(size);
                    if let Some(o) = omega {
                        view = view.plot_omega(*o);
                    }
                    let view = if s.patches.patches.is_empty() {
                        view
                    } else {
                        view.native_diagnostic_patches(&s.patches)
                    };
                    ui.add(view)
                }
                ViewKind::SkewT {
                    omega,
                    interp_winds,
                } => {
                    let mut w = sharppyrs::SkewT::new(&a.prof)
                        .title(s.title.clone())
                        .brand(s.brand.clone())
                        .style(s.style.clone())
                        .parcel(s.parcel)
                        .cursor_readout(false)
                        .interp_winds(*interp_winds)
                        .size(size);
                    if let Some(o) = omega {
                        w = w.plot_omega(*o);
                    }
                    ui.add(w)
                }
            }
        })
        .inner;
    if let Some(ov) = &s.overlay {
        let rect = resp.rect;
        let (skew, hodo) = match s.view {
            ViewKind::Window { .. } => {
                let top = s.layout.top_height_fraction.clamp(0.40, 0.85);
                let sw = s.layout.skew_width_fraction.clamp(0.30, 0.70);
                (
                    Rect::from_min_max(
                        rect.min,
                        Pos2::new(
                            rect.min.x + rect.width() * sw,
                            rect.min.y + rect.height() * top,
                        ),
                    ),
                    hodo_rect(&s.layout, rect),
                )
            }
            ViewKind::SkewT { .. } => (rect, None),
        };
        draw_overlay(ui, skew, ov, &s.style);
        if let Some(h) = hodo {
            draw_hodo_overlay(ui, h, s.layout.hodo_zoom_kts, ov);
        }
    }
}

/// Port of the sharppyrs skew-T `Geom` transform (SHARPpy backgroundSkewT):
/// the same padding, pressure range and 100/3-degree skew, so overlays land
/// on the widget's own coordinates.
struct Geom {
    rect: Rect,
    lpad: f64,
    tpad: f64,
    brx: f64,
    bry: f64,
    yrange: f64,
}

impl Geom {
    fn new(rect: Rect) -> Self {
        let (w, h) = (rect.width() as f64, rect.height() as f64);
        Geom {
            rect,
            lpad: 46.0,
            tpad: 20.0,
            brx: w - 65.0,
            bry: h - 20.0,
            yrange: (100.0f64 / 3.0).to_radians().tan() * 100.0,
        }
    }
    fn py(&self, p: f64) -> f64 {
        let (lmax, lmin) = (1050.0f64.ln(), 100.0f64.ln());
        self.bry - ((lmax - p.ln()) / (lmax - lmin)) * (self.bry - self.tpad)
    }
    fn tx(&self, t: f64, p: f64) -> f64 {
        let scl1 = 50.0 - ((self.bry - self.py(p)) / (self.bry - self.tpad)) * self.yrange;
        self.brx - ((scl1 - t) / 100.0) * (self.brx - self.lpad)
    }
    fn pt(&self, t: f64, p: f64) -> Pos2 {
        Pos2::new(
            self.rect.min.x + self.tx(t, p) as f32,
            self.rect.min.y + self.py(p) as f32,
        )
    }
    fn clip(&self) -> Rect {
        Rect::from_min_max(
            Pos2::new(
                self.rect.min.x + self.lpad as f32,
                self.rect.min.y + self.tpad as f32,
            ),
            Pos2::new(
                self.rect.min.x + self.brx as f32,
                self.rect.min.y + self.bry as f32,
            ),
        )
    }
}

fn norm<const N: usize>(f: &[f32; N]) -> [f32; N] {
    let t: f32 = f.iter().map(|x| x.max(0.0)).sum();
    if t <= f32::EPSILON {
        [1.0 / N as f32; N]
    } else {
        std::array::from_fn(|i| f[i].max(0.0) / t)
    }
}

fn split_h<const N: usize>(band: Rect, f: &[f32; N]) -> [Rect; N] {
    let f = norm(f);
    let mut x = band.min.x;
    std::array::from_fn(|i| {
        let x0 = x;
        x = if i == N - 1 {
            band.max.x
        } else {
            x + band.width() * f[i]
        };
        Rect::from_min_max(Pos2::new(x0, band.min.y), Pos2::new(x, band.max.y))
    })
}

/// Where sharppyrs puts the hodograph cell (same track arithmetic as
/// `SoundingView`: 16 px brand strip, right columns, main/inset split).
/// Only the upper-right cells are mapped; a hodograph moved into the bottom
/// band gets no member overlay.
fn hodo_rect(l: &SoundingLayout, rect: Rect) -> Option<Rect> {
    use sharppyrs::PanelKind::Hodograph;
    let band_top = rect.min.y + rect.height() * l.top_height_fraction.clamp(0.40, 0.85);
    let skew_right = rect.min.x + rect.width() * l.skew_width_fraction.clamp(0.30, 0.70);
    let g2 = Rect::from_min_max(
        Pos2::new(skew_right, rect.min.y + 16.0),
        Pos2::new(rect.max.x, band_top),
    );
    let cols = split_h(g2, &l.right_column_fractions);
    let rc = cols[2];
    let main_bottom = rc.min.y + rc.height() * l.right_main_height_fraction.clamp(0.35, 0.85);
    let main = Rect::from_min_max(rc.min, Pos2::new(rc.max.x, main_bottom));
    let insets = split_h(
        Rect::from_min_max(Pos2::new(rc.min.x, main_bottom), rc.max),
        &l.inset_column_fractions,
    );
    if l.main == Hodograph {
        return Some(main);
    }
    for i in 0..4 {
        if l.insets[i] == Hodograph {
            return Some(insets[i]);
        }
    }
    for i in 0..2 {
        if l.strips[i] == Hodograph {
            return Some(cols[i]);
        }
    }
    None
}

fn draw_hodo_overlay(ui: &mut egui::Ui, r: Rect, zoom_kts: f64, ov: &Overlay) {
    let painter = ui.painter().with_clip_rect(r);
    let scale = r.width() as f64 / zoom_kts.clamp(80.0, 500.0);
    let (cx, cy) = (r.width() as f64 / 2.0, r.height() as f64 / 2.0);
    let pt = |u: f64, v: f64| {
        Pos2::new(
            r.min.x + (cx + u * scale) as f32,
            r.min.y + (cy - v * scale) as f32,
        )
    };
    let line = |pts: &Vec<(f64, f64)>, c: Color32, w: f32| {
        let v: Vec<Pos2> = pts
            .iter()
            .filter(|(u, v)| u.is_finite() && v.is_finite())
            .map(|(u, v)| pt(*u, *v))
            .collect();
        if v.len() > 1 {
            painter.add(Shape::line(v, Stroke::new(w, c)));
        }
    };
    for m in &ov.members {
        line(
            &m.hodo,
            Color32::from_rgba_unmultiplied(220, 220, 220, ov.alpha),
            1.0,
        );
    }
    if let Some(b) = &ov.bands
        && ov.members.is_empty()
    {
        line(
            &b.hodo_med,
            Color32::from_rgba_unmultiplied(255, 255, 255, 200),
            2.0,
        );
    }
}

fn draw_overlay(ui: &mut egui::Ui, skew: Rect, ov: &Overlay, st: &SkewTStyle) {
    let g = Geom::new(skew);
    let painter = ui.painter().with_clip_rect(g.clip());
    let ok = |p: f64, v: f64| p.is_finite() && v.is_finite() && p > 100.0;
    if let Some(b) = &ov.bands {
        let band = |lo: &Vec<f64>, hi: &Vec<f64>, c: Color32| {
            for k in 0..b.p.len().saturating_sub(1) {
                let (p0, p1) = (b.p[k], b.p[k + 1]);
                if !(ok(p0, lo[k])
                    && ok(p1, lo[k + 1])
                    && hi[k].is_finite()
                    && hi[k + 1].is_finite())
                {
                    continue;
                }
                let quad = vec![
                    g.pt(lo[k], p0),
                    g.pt(lo[k + 1], p1),
                    g.pt(hi[k + 1], p1),
                    g.pt(hi[k], p0),
                ];
                painter.add(Shape::convex_polygon(quad, c, Stroke::NONE));
            }
        };
        if ov.show_t {
            band(
                &b.t[0],
                &b.t[4],
                Color32::from_rgba_unmultiplied(255, 70, 70, 55),
            );
            if ov.inner_band {
                band(
                    &b.t[1],
                    &b.t[3],
                    Color32::from_rgba_unmultiplied(255, 70, 70, 75),
                );
            }
        }
        if ov.show_td {
            band(
                &b.td[0],
                &b.td[4],
                Color32::from_rgba_unmultiplied(70, 230, 90, 55),
            );
            if ov.inner_band {
                band(
                    &b.td[1],
                    &b.td[3],
                    Color32::from_rgba_unmultiplied(70, 230, 90, 75),
                );
            }
        }
    }
    let line = |p: &Vec<f64>, v: &Vec<f64>, c: Color32, w: f32| {
        let pts: Vec<Pos2> = p
            .iter()
            .zip(v)
            .filter(|(p, v)| ok(**p, **v))
            .map(|(p, v)| g.pt(*v, *p))
            .collect();
        if pts.len() > 1 {
            painter.add(Shape::line(pts, Stroke::new(w, c)));
        }
    };
    for m in &ov.members {
        if ov.show_t {
            line(
                &m.p,
                &m.t,
                Color32::from_rgba_unmultiplied(255, 90, 90, ov.alpha),
                1.0,
            );
        }
        if ov.show_td {
            line(
                &m.p,
                &m.td,
                Color32::from_rgba_unmultiplied(90, 235, 110, ov.alpha),
                1.0,
            );
        }
    }
    if let Some(b) = &ov.bands {
        if ov.show_t {
            line(
                &b.p,
                &b.t[2],
                Color32::from_rgba_unmultiplied(255, 210, 210, 170),
                1.0,
            );
        }
        if ov.show_td {
            line(
                &b.p,
                &b.td[2],
                Color32::from_rgba_unmultiplied(210, 255, 210, 170),
                1.0,
            );
        }
    }
    if !ov.legend.is_empty() {
        let c = g.clip();
        ui.painter().text(
            Pos2::new(c.max.x - 6.0, c.min.y + 4.0),
            Align2::RIGHT_TOP,
            &ov.legend,
            st.regular_font(12.0),
            st.fg_color,
        );
    }
}

/// Rasterises egui's tessellated meshes on the CPU the way egui-wgpu draws
/// them into an Rgba8Unorm target: premultiplied vertex colour times the
/// texture sample, blended `src + dst * (1 - src.a)` in gamma space, pixel
/// centres sampled, egui's own feathering doing the anti-aliasing.
#[derive(Default)]
pub struct CpuRenderer {
    textures: std::collections::HashMap<egui::TextureId, CpuTexture>,
}

struct CpuTexture {
    w: usize,
    h: usize,
    px: Vec<[f32; 4]>,
    linear: bool,
}

impl CpuTexture {
    fn sample(&self, u: f32, v: f32) -> [f32; 4] {
        if self.w == 0 || self.h == 0 {
            return [1.0; 4];
        }
        let at = |x: i64, y: i64| {
            let x = x.clamp(0, self.w as i64 - 1) as usize;
            let y = y.clamp(0, self.h as i64 - 1) as usize;
            self.px[y * self.w + x]
        };
        let (fx, fy) = (u * self.w as f32, v * self.h as f32);
        if !self.linear {
            return at(fx.floor() as i64, fy.floor() as i64);
        }
        let (fx, fy) = (fx - 0.5, fy - 0.5);
        let (x0, y0) = (fx.floor(), fy.floor());
        let (ax, ay) = (fx - x0, fy - y0);
        let (x0, y0) = (x0 as i64, y0 as i64);
        let (a, b, c, d) = (at(x0, y0), at(x0 + 1, y0), at(x0, y0 + 1), at(x0 + 1, y0 + 1));
        std::array::from_fn(|k| {
            let top = a[k] + (b[k] - a[k]) * ax;
            let bot = c[k] + (d[k] - c[k]) * ax;
            top + (bot - top) * ay
        })
    }
}

fn unorm(c: Color32) -> [f32; 4] {
    c.to_array().map(|x| x as f32 / 255.0)
}

impl egui_kittest::TestRenderer for CpuRenderer {
    fn handle_delta(&mut self, delta: &egui::TexturesDelta) {
        for (id, d) in &delta.set {
            let egui::epaint::ImageData::Color(img) = &d.image;
            let [w, h] = img.size;
            let linear = d.options.magnification == egui::epaint::textures::TextureFilter::Linear;
            match d.pos {
                None => {
                    self.textures.insert(
                        *id,
                        CpuTexture {
                            w,
                            h,
                            px: img.pixels.iter().map(|c| unorm(*c)).collect(),
                            linear,
                        },
                    );
                }
                Some([x0, y0]) => {
                    if let Some(t) = self.textures.get_mut(id) {
                        for y in 0..h.min(t.h.saturating_sub(y0)) {
                            for x in 0..w.min(t.w.saturating_sub(x0)) {
                                t.px[(y0 + y) * t.w + x0 + x] = unorm(img.pixels[y * w + x]);
                            }
                        }
                    }
                }
            }
        }
        // Frees are not applied, as in egui_kittest's wgpu renderer: egui
        // never reuses a managed texture id within a harness.
    }

    fn render(
        &mut self,
        ctx: &egui::Context,
        output: &egui::FullOutput,
    ) -> Result<image::RgbaImage, String> {
        let ppp = ctx.pixels_per_point();
        let size = ctx.content_rect().size() * ppp;
        let (w, h) = (size.x.round().max(0.0) as usize, size.y.round().max(0.0) as usize);
        let mut fb = vec![[0f32; 4]; w * h];
        let white = CpuTexture {
            w: 1,
            h: 1,
            px: vec![[1.0; 4]],
            linear: false,
        };
        for egui::epaint::ClippedPrimitive {
            clip_rect,
            primitive,
        } in ctx.tessellate(output.shapes.clone(), ppp)
        {
            let egui::epaint::Primitive::Mesh(mesh) = primitive else {
                continue;
            };
            let clip = |v: f32, max: usize| (v * ppp).round().clamp(0.0, max as f32) as i64;
            let (cx0, cy0) = (clip(clip_rect.min.x, w), clip(clip_rect.min.y, h));
            let (cx1, cy1) = (clip(clip_rect.max.x, w), clip(clip_rect.max.y, h));
            if cx1 <= cx0 || cy1 <= cy0 {
                continue;
            }
            let tex = self.textures.get(&mesh.texture_id).unwrap_or(&white);
            for tri in mesh.indices.chunks_exact(3) {
                let mut v = [
                    &mesh.vertices[tri[0] as usize],
                    &mesh.vertices[tri[1] as usize],
                    &mesh.vertices[tri[2] as usize],
                ];
                let p = |k: &egui::epaint::Vertex| (k.pos.x as f64 * ppp as f64, k.pos.y as f64 * ppp as f64);
                let edge = |a: (f64, f64), b: (f64, f64), c: (f64, f64)| {
                    (b.0 - a.0) * (c.1 - a.1) - (b.1 - a.1) * (c.0 - a.0)
                };
                let mut area = edge(p(v[0]), p(v[1]), p(v[2]));
                if !area.is_finite() || area == 0.0 {
                    continue;
                }
                if area < 0.0 {
                    v.swap(1, 2);
                    area = -area;
                }
                let (p0, p1, p2) = (p(v[0]), p(v[1]), p(v[2]));
                // A shared edge is walked in opposite directions by its two
                // triangles, so this rule gives its pixels to exactly one.
                let owns = |a: (f64, f64), b: (f64, f64)| {
                    let (dx, dy) = (b.0 - a.0, b.1 - a.1);
                    dy > 0.0 || (dy == 0.0 && dx > 0.0)
                };
                let (o0, o1, o2) = (owns(p1, p2), owns(p2, p0), owns(p0, p1));
                let xs = ((p0.0.min(p1.0).min(p2.0) - 0.5).ceil() as i64).max(cx0);
                let xe = ((p0.0.max(p1.0).max(p2.0) - 0.5).floor() as i64).min(cx1 - 1);
                let ys = ((p0.1.min(p1.1).min(p2.1) - 0.5).ceil() as i64).max(cy0);
                let ye = ((p0.1.max(p1.1).max(p2.1) - 0.5).floor() as i64).min(cy1 - 1);
                if xs > xe || ys > ye {
                    continue;
                }
                let col = [unorm(v[0].color), unorm(v[1].color), unorm(v[2].color)];
                let flat = v[0].color == v[1].color
                    && v[1].color == v[2].color
                    && v[0].uv == v[1].uv
                    && v[1].uv == v[2].uv;
                let flat_src = {
                    let t = tex.sample(v[0].uv.x, v[0].uv.y);
                    std::array::from_fn::<f32, 4, _>(|k| col[0][k] * t[k])
                };
                for y in ys..=ye {
                    let py = y as f64 + 0.5;
                    let row = y as usize * w;
                    for x in xs..=xe {
                        let q = (x as f64 + 0.5, py);
                        let (w0, w1, w2) = (edge(p1, p2, q), edge(p2, p0, q), edge(p0, p1, q));
                        if w0 < 0.0
                            || w1 < 0.0
                            || w2 < 0.0
                            || (w0 == 0.0 && !o0)
                            || (w1 == 0.0 && !o1)
                            || (w2 == 0.0 && !o2)
                        {
                            continue;
                        }
                        let src = if flat {
                            flat_src
                        } else {
                            let (l0, l1, l2) =
                                ((w0 / area) as f32, (w1 / area) as f32, (w2 / area) as f32);
                            let u = l0 * v[0].uv.x + l1 * v[1].uv.x + l2 * v[2].uv.x;
                            let vv = l0 * v[0].uv.y + l1 * v[1].uv.y + l2 * v[2].uv.y;
                            let t = tex.sample(u, vv);
                            std::array::from_fn(|k| {
                                (l0 * col[0][k] + l1 * col[1][k] + l2 * col[2][k]) * t[k]
                            })
                        };
                        let d = &mut fb[row + x as usize];
                        let keep = 1.0 - src[3];
                        for k in 0..4 {
                            d[k] = src[k] + d[k] * keep;
                        }
                    }
                }
            }
        }
        let bytes = fb
            .iter()
            .flat_map(|p| p.map(|c| (c.clamp(0.0, 1.0) * 255.0).round() as u8))
            .collect();
        image::RgbaImage::from_raw(w as u32, h as u32, bytes)
            .ok_or_else(|| "cpu rasteriser: bad frame size".into())
    }
}

/// Patches every rendered window carries: the two ECAPE cells are blanked,
/// always (the project owner: never ECAPE).
pub fn base_patches() -> Vec<sharppyrs::NativeDiagnosticPatch> {
    vec![
        sharppyrs::NativeDiagnosticPatch::blank(DiagnosticTablePanelKind::Severe, "severe.ecape"),
        sharppyrs::NativeDiagnosticPatch::blank(DiagnosticTablePanelKind::Severe, "severe.ncape"),
    ]
}
