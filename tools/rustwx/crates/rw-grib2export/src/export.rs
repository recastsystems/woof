//! One export request: discover frames, compute every product with rw-post
//! from one shared column state, pack GRIB2, keep the manifest and the
//! append ledger, and build the archive.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use chrono::NaiveDateTime;
use rayon::prelude::*;
use serde_json::{json, Value};
use wx_core::grib2::{Grib2Writer, GridDefinition, MessageBuilder, PackingMethod, ProductDefinition, StatisticalInterval};

use rw_mlexport::inputs::{self, Candidate};
use rw_mlexport::{times, zipout};
use rw_post::group_e::{self, GustMethod};
use rw_post::plev;
use rw_post::severe;
use rw_post::surface::SurfaceField;
use rw_post::{history, PostDevice};

use crate::catalog::{self, Row, Source, EXTREMA};
use crate::extrema::{self, FrameExtrema};
use crate::grid::{self, Projection};
use crate::request::{Definitions, Request};
use crate::{fail, group_d, refuse, renderer, Result, CATALOG, PROFILE};

/// Originating centre (Section 1, octets 6-7; WMO common code table C-11).
/// The catalog's NCEP local parameters (0.7.198, 0.3.198, 0.16.19x) and level
/// types (200, 204, 214, 215, 224, 234) are NCEP local-table entries (spec
/// [R10]), which decoders apply only under centre 7; the subcentre is 0.
pub const CENTRE: u16 = 7;
/// Centre for WOOF's own local parameters (category 192, the shear
/// components and magnitudes, spec 6.3): 255, "missing", because no NCEP
/// table defines them and claiming one would make decoders mislabel them.
pub const CENTRE_LOCAL: u16 = 255;

/// The centre a row is encoded under.
pub fn centre_for(row: &Row) -> u16 {
    if row.discipline == 0 && row.category == 192 { CENTRE_LOCAL } else { CENTRE }
}
/// GRIB master table version (Section 1, octet 10).
pub const MASTER_TABLES: u8 = 2;
/// Local table version (Section 1, octet 11): NCEP local table 1.
pub const LOCAL_TABLES: u8 = 1;
/// Type of generating process (code table 4.3): 2 = forecast.
pub const GENERATING_PROCESS: u8 = 2;
/// Analysis or forecast generating process identifier (product template
/// 4.0/4.8 octet 14, "defined by the originating centre"), RULINGS 12:
/// identify WOOF.  Under centre 7 a decoder reads it against NCEP Office
/// Note 388 Table A, which assigns or reserves every value; 231-254 is a
/// reserved block with no assignment and 255 means missing.  WOOF writes
/// 254, the top of that block, so its messages are told apart from every
/// NCEP model (Table A assigns them 2 to 220, for example 83, 105, 116 and 134) without claiming
/// any of them.  One constant: if NCEP ever assigns 254, this line moves.
pub const GENERATING_PROCESS_ID: u8 = 254;
/// The identifier's meaning, as the manifest records it.
pub const GENERATING_PROCESS_ID_TEXT: &str =
    "254: WOOF (product template octet 14; NCEP ON388 Table A reserved block 231-254, unassigned there)";

const STATE_FILE: &str = ".woof-grib2-state.json";
const MANIFEST: &str = "manifest.json";

/// One frame's computed planes and their provenance.
pub(crate) struct Planes {
    pub(crate) surface: BTreeMap<&'static str, Vec<f32>>,
    /// `[level][field]` planes of the pressure block.
    pub(crate) plev: Vec<Vec<Vec<f32>>>,
    pub(crate) omitted: Vec<(String, String)>,
    pub(crate) devices: BTreeMap<&'static str, String>,
    pub(crate) timings: BTreeMap<&'static str, f64>,
    pub(crate) pressure_source: String,
    pub(crate) vertical_source: String,
    /// The gust method that ran and, for the TKE gust, its carrier.
    gust_source: String,
    /// Group D inputs as used: cloud-fraction source and aerosol carriers.
    group_d_sources: Value,
    /// Rows the renderer computed (`definitions = renderer`), with their
    /// manifest method, and the rows it could not, with the reason.
    renderer_rows: BTreeMap<&'static str, String>,
    renderer_kept: Vec<(String, String)>,
    /// The shared column state (Group A), kept for profiles that derive
    /// more from it (the RRFS UPP set, `crate::rrfs_upp`).
    pub(crate) state: Option<rw_post::state::ColumnState>,
}

pub(crate) fn utc(seconds: i64) -> Result<NaiveDateTime> {
    chrono::DateTime::from_timestamp(seconds, 0)
        .map(|d| d.naive_utc())
        .ok_or_else(|| refuse(format!("time {seconds} s is outside the GRIB2 calendar range")))
}

/// Forecast time and its unit (code table 4.4): hours when whole, else
/// minutes, else seconds.
pub fn lead(seconds: i64) -> Result<(u8, u32)> {
    if seconds < 0 {
        return Err(refuse("the valid time precedes SIMULATION_START_DATE, so the forecast time would be negative"));
    }
    let s = seconds as u64;
    let (unit, value) = if s % 3600 == 0 { (1, s / 3600) } else if s % 60 == 0 { (0, s / 60) } else { (13, s) };
    let value = u32::try_from(value).map_err(|_| refuse("forecast time does not fit GRIB2 octets 19-22"))?;
    Ok((unit, value))
}

fn compact(seconds: i64) -> String {
    // 2026-10-03T22:00:00 -> 20261003T220000Z
    let mut s = times::iso(seconds).replace(['-', ':'], "");
    if !s.ends_with('Z') {
        s.push('Z');
    }
    s
}

fn dump(req: &Request, file_stem: &str, index: usize, id: &str, level: Option<f64>, plane: &[f32]) -> Result<()> {
    let Some(dir) = &req.dump_planes else { return Ok(()) };
    let dir = dir.join(file_stem);
    std::fs::create_dir_all(&dir)?;
    let name = match level {
        Some(p) => format!("{index:03}_{id}_{p}.f32"),
        None => format!("{index:03}_{id}.f32"),
    };
    let bytes: Vec<u8> = plane.iter().flat_map(|x| x.to_le_bytes()).collect();
    std::fs::write(dir.join(name), bytes)?;
    Ok(())
}

fn attr_f64(file: &netcrust::File, name: &str) -> Option<f64> {
    file.attribute(name).and_then(|a| a.as_f64())
}

fn attr_text(file: &netcrust::File, name: &str) -> Option<String> {
    file.attribute(name).and_then(|a| a.as_string().map(str::to_owned))
}

fn read_first(file: &netcrust::File, name: &str) -> Result<Option<Vec<f64>>> {
    let Some(var) = file.variable(name) else { return Ok(None) };
    // A declared fill is missing, not data (MPASSIT writes -9999 off the mesh,
    // which reached APCP as -9999 mm).
    let fill = var.attribute("_FillValue").or_else(|| var.attribute("missing_value")).and_then(|a| a.as_f64());
    let v = file.read_f64_first_record_or_all(name).map_err(|e| fail(format!("read {name}: {e}")))?;
    Ok(Some(match fill {
        Some(f) if f.is_finite() => v.into_iter().map(|x| if x == f { f64::NAN } else { x }).collect(),
        _ => v,
    }))
}

fn all_missing(v: &[f32]) -> bool {
    v.iter().all(|x| !x.is_finite())
}

/// Frame identity read from the history before any computation.
pub(crate) struct FrameMeta {
    pub(crate) domain: String,
    pub(crate) valid: i64,
    pub(crate) start: i64,
    pub(crate) projection: Projection,
    pub(crate) nx: usize,
    pub(crate) ny: usize,
    pub(crate) lat: Vec<f64>,
    pub(crate) lon: Vec<f64>,
    pub(crate) apcp: Option<Vec<f32>>,
    pub(crate) apcp_reason: String,
}

/// RULINGS 7: no standard exists, so the candidates were scored against
/// ASOS gust reports (merge-and-accept lane, 2026-10-05).  The mixed-layer
/// TKE gust (Brasseur 2001) won wherever the history carries TKE (WOOF
/// MYNN 2024-05-25 Kansas-Oklahoma: MAE 1.84 against 2.97 m/s on reports
/// with a gust group, 0.94 against 1.79 censored), so `auto` takes it
/// there and the similarity gust elsewhere, rather than omitting gust on a
/// history without TKE.  `similarity` and `tke` force one method.
#[must_use]
pub fn gust_method(requested: &str, has_tke: bool) -> GustMethod {
    match requested {
        "tke" => GustMethod::MixedLayerTke,
        "similarity" => GustMethod::Similarity,
        _ if has_tke => GustMethod::MixedLayerTke,
        _ => GustMethod::Similarity,
    }
}

pub(crate) fn frame_meta(path: &Path, hint: Option<&(String, i64)>) -> Result<FrameMeta> {
    let file = netcrust::File::open(path).map_err(|e| fail(format!("open {}: {e}", path.display())))?;
    let times_text = rw_post::history::read_times(&file).map_err(|e| refuse(format!("{} has no readable Times ({e})", path.display())))?;
    if times_text.len() != 1 {
        return Err(refuse(format!(
            "{} holds {} times; the exporter reads one frame per history file (the WOOF history layout), and guessing which record to take would mislabel the output",
            path.display(),
            times_text.len()
        )));
    }
    let valid = times::parse(times_text[0].trim_end_matches('\0').trim())
        .ok_or_else(|| refuse(format!("{} has an unreadable valid time {:?}", path.display(), times_text[0])))?;
    let start_text = attr_text(&file, "SIMULATION_START_DATE")
        .or_else(|| attr_text(&file, "START_DATE"))
        .ok_or_else(|| refuse(format!("{} has no SIMULATION_START_DATE, so the GRIB reference time and forecast hour cannot be set", path.display())))?;
    let start = times::parse(start_text.trim_end_matches('\0').trim())
        .ok_or_else(|| refuse(format!("{} has an unreadable SIMULATION_START_DATE {start_text:?}", path.display())))?;
    let domain = match hint {
        Some((d, _)) => d.clone(),
        None => format!("d{:02}", attr_f64(&file, "GRID_ID").map_or(1, |v| v as i64)),
    };
    let lat = read_first(&file, "XLAT")?.ok_or_else(|| refuse("XLAT is absent, so no GRIB grid point can be placed"))?;
    let lon = read_first(&file, "XLONG")?.ok_or_else(|| refuse("XLONG is absent, so no GRIB grid point can be placed"))?;
    let nx = file.dimension("west_east").map(|d| d.len()).ok_or_else(|| refuse("dimension west_east is absent"))?;
    let ny = file.dimension("south_north").map(|d| d.len()).ok_or_else(|| refuse("dimension south_north is absent"))?;
    let projection = Projection {
        map_proj: attr_f64(&file, "MAP_PROJ").map_or(-1, |v| v as i32),
        truelat1: attr_f64(&file, "TRUELAT1").unwrap_or(0.0),
        truelat2: attr_f64(&file, "TRUELAT2").unwrap_or(0.0),
        stand_lon: attr_f64(&file, "STAND_LON").unwrap_or(0.0),
        pole_lat: attr_f64(&file, "POLE_LAT").unwrap_or(90.0),
        pole_lon: attr_f64(&file, "POLE_LON").unwrap_or(0.0),
        dx: attr_f64(&file, "DX").unwrap_or(0.0),
        dy: attr_f64(&file, "DY").unwrap_or(0.0),
    };
    // Total precipitation since the start: RAINNC + RAINC + bucket counters.
    let (mut apcp, mut apcp_reason) = (None, String::new());
    match read_first(&file, "RAINNC")? {
        None => apcp_reason = "RAINNC is absent; absence is not zero precipitation".into(),
        Some(nc) => {
            let c = read_first(&file, "RAINC")?;
            let inc = read_first(&file, "I_RAINNC")?;
            let ic = read_first(&file, "I_RAINC")?;
            let bucket = attr_f64(&file, "BUCKET_MM").filter(|b| *b > 0.0);
            if (inc.is_some() || ic.is_some()) && bucket.is_none() {
                apcp_reason = "I_RAINNC/I_RAINC bucket counters are present without a positive BUCKET_MM, so the total cannot be rebuilt".into();
            } else {
                let b = bucket.unwrap_or(0.0);
                let total: Vec<f32> = (0..nc.len())
                    .map(|i| {
                        let mut t = nc[i];
                        if let Some(c) = &c {
                            t += c[i];
                        }
                        if let Some(k) = &inc {
                            t += b * k[i];
                        }
                        if let Some(k) = &ic {
                            t += b * k[i];
                        }
                        t as f32
                    })
                    .collect();
                apcp = Some(total);
            }
        }
    }
    if valid == start && apcp.is_some() {
        apcp = None;
        apcp_reason = "the initial frame has a zero-length accumulation window".into();
    }
    Ok(FrameMeta { domain, valid, start, projection, nx, ny, lat, lon, apcp, apcp_reason })
}

/// The kernel set, opened once per process and reused across frames.
pub struct Devices {
    choice: PostDevice,
    #[cfg(any(windows, target_os = "linux"))]
    opened: Option<std::result::Result<rw_post::gpu::GpuPost, String>>,
    /// The one line `auto` owes the user when the card did not open.
    notice: Option<String>,
}

impl Devices {
    pub fn new(choice: PostDevice) -> Self {
        Self {
            choice,
            #[cfg(any(windows, target_os = "linux"))]
            opened: None,
            notice: None,
        }
    }

    /// The pending CPU-fallback notice, once per process.  Breakage this
    /// prevents: through 2.9.0 a card whose driver refused the post kernels
    /// (CUDA_ERROR_UNSUPPORTED_PTX_VERSION on a CUDA 13.2 driver) silently
    /// computed every GRIB2 product on the CPU; only manifest.json said cpu.
    pub fn take_notice(&mut self) -> Option<String> {
        self.notice.take()
    }

    /// The card for a frame of `shape`, or `None` for the CPU.  `gpu`
    /// fails when no card opens; `auto` takes the CPU when no card opens or
    /// the frame does not fit (spec 3.5, D30).
    #[cfg(any(windows, target_os = "linux"))]
    fn card(&mut self, shape: &rw_post::state::Shape) -> Result<Option<&rw_post::gpu::GpuPost>> {
        if self.choice == PostDevice::Cpu {
            return Ok(None);
        }
        if self.opened.is_none() {
            let opened = rw_post::gpu::GpuPost::open(0).map_err(|e| e.to_string());
            if let (Err(e), PostDevice::Auto) = (&opened, self.choice) {
                self.notice = Some(format!(
                    "post_device auto: the GRIB2 products are computed on the CPU (the same bytes, slower) \
                     because the GPU post kernels did not open: {e}"
                ));
            }
            self.opened = Some(opened);
        }
        match self.opened.as_ref().unwrap() {
            Ok(g) => {
                if self.choice == PostDevice::Auto {
                    // State plus the largest group's working set, with margin.
                    let need = rw_post::gpu::GpuPost::state_bytes(shape) * 2 + (512 << 20);
                    match g.memory() {
                        Ok((free, _)) if free >= need => Ok(Some(g)),
                        _ => Ok(None),
                    }
                } else {
                    Ok(Some(g))
                }
            }
            Err(e) if self.choice == PostDevice::Gpu => {
                Err(refuse(format!("post_device gpu was asked for, but no CUDA device opened ({e})")))
            }
            Err(_) => Ok(None),
        }
    }

    #[cfg(not(any(windows, target_os = "linux")))]
    fn card(&mut self, _shape: &rw_post::state::Shape) -> Result<Option<()>> {
        if self.choice == PostDevice::Gpu {
            return Err(refuse("post_device gpu was asked for, but this platform has no CUDA post path"));
        }
        Ok(None)
    }
}

fn rotate_pair(u: &mut [f32], v: &mut [f32], sina: &[f32], cosa: &[f32]) {
    let n = sina.len();
    for i in 0..u.len() {
        let (eu, ev) = rw_post::wind::rotate(u[i], v[i], sina[i % n], cosa[i % n]);
        u[i] = eu;
        v[i] = ev;
    }
}

/// Compute every catalog plane of one frame.
pub(crate) fn compute(path: &Path, req: &Request, devices: &mut Devices, meta: &FrameMeta) -> Result<Planes> {
    let t_read = Instant::now();
    let heights: std::collections::BTreeSet<u32> = catalog::ROWS.iter().filter_map(|r| {
        if let Source::WindHeight(height, _) = r.source { Some(height) } else { None }
    }).collect();
    let extra = if req.skip_vertical_extras && !req.writes_pressure() {
        crate::vertical::Output {
            pressure: Vec::new(),
            wind: heights.iter().map(|&h| (h, vec![f32::NAN; meta.nx * meta.ny], vec![f32::NAN; meta.nx * meta.ny])).collect(),
        }
    } else {
        crate::vertical::compute(path, &req.levels_hpa(), &heights.into_iter().collect::<Vec<_>>(), req.writes_pressure())?
    };
    let mut frame = history::Frame::read(path, 0).map_err(|e| refuse(format!("{}: {e}", path.display())))?;
    let mut owned = group_e::history::OwnedCarriers::read(path).map_err(|e| refuse(format!("{}: {e}", path.display())))?;
    // A height-coordinate frame (MPAS on a WRF-shaped grid) is one whose
    // ground IS its vertical coordinate's lowest level: MPAS's init sets
    // ter = zgrid(1).  Every group takes that ground as the terrain.
    // Breakage prevented: the hex RRFS history writes ter = 0 (mv3km
    // 2023-03-31, rrfs/c3-lam), so MPASSIT-shaped and converter frames carried
    // HGT = 0 and MSLP, below-ground heights and the 1 km AGL reflectivity were
    // computed over flat terrain (1000 hPa heights of -1462 m).
    let terrain: Option<Vec<f32>> = (frame.coefficients == rw_post::hybrid::CoefficientSource::HeightCoordinate).then(|| {
        let n = frame.shape.ncell();
        (0..n).into_par_iter().map(|c| (frame.state.ph[c] + frame.state.phb[c]) / rw_post::consts::G).collect()
    });
    if let Some(t) = &terrain {
        frame.surface.set(rw_post::surface::SurfaceInput::Hgt, t);
        owned.vars.insert("HGT", t.clone());
    }
    let mut timings = BTreeMap::new();
    timings.insert("read", t_read.elapsed().as_secs_f64());
    let shape = frame.shape;
    let n = shape.ncell();
    if (shape.nx, shape.ny) != (meta.nx, meta.ny) {
        return Err(refuse("the mass grid and XLAT/XLONG disagree in size"));
    }
    let mut devices_used = BTreeMap::new();
    let mut omitted: Vec<(String, String)> = Vec::new();
    let earth = req.winds == "earth";
    let choice = req.device();

    let gpu = devices.card(&shape)?;
    #[cfg(any(windows, target_os = "linux"))]
    let device_for_groups = if gpu.is_some() { choice } else { PostDevice::Cpu };
    #[cfg(not(any(windows, target_os = "linux")))]
    let device_for_groups = PostDevice::Cpu;

    // ---- Group A: the ONE shared column state and the surface planes.
    let t = Instant::now();
    #[cfg(any(windows, target_os = "linux"))]
    let a = match gpu {
        Some(g) => match rw_post::group_a_gpu(g, &mut frame) {
            Ok(a) => a,
            Err(e) if choice == PostDevice::Gpu => return Err(fail(format!("group A on the GPU: {e}"))),
            Err(_) => {
                frame = history::Frame::read(path, 0).map_err(|e| refuse(format!("{}: {e}", path.display())))?;
                rw_post::group_a_cpu(&mut frame).map_err(fail)?
            }
        },
        None => rw_post::group_a_cpu(&mut frame).map_err(fail)?,
    };
    #[cfg(not(any(windows, target_os = "linux")))]
    let a = rw_post::group_a_cpu(&mut frame).map_err(fail)?;
    timings.insert("group_a", t.elapsed().as_secs_f64());
    devices_used.insert("a", a.device.clone());
    let st = &a.state;
    let mut surface: BTreeMap<&'static str, Vec<f32>> = BTreeMap::new();
    for r in catalog::ROWS {
        if let Source::WindHeight(height, u) = r.source {
            let (_, up, vp) = extra.wind.iter().find(|(h, _, _)| *h == height).unwrap();
            surface.insert(r.id, if u { up.clone() } else { vp.clone() });
        }
    }
    for spec in SurfaceField::PRODUCTS.iter() {
        surface.insert(spec.id, a.plane(spec.field).to_vec());
    }
    let v = &owned.vars;
    let get = |name: &str| v.get(name).map(Vec::as_slice);
    let sin_cos = match (frame.sinalpha.as_deref(), frame.cosalpha.as_deref()) {
        (Some(s), Some(c)) if s.len() == n && c.len() == n => Some((s.to_vec(), c.to_vec())),
        _ => None,
    };
    if earth && sin_cos.is_none() {
        return Err(refuse(
            "winds = earth needs SINALPHA and COSALPHA, which this history lacks; grid-relative components labelled earth-relative would point the wrong way",
        ));
    }

    // ---- history copies.
    let mut u10 = get("U10").map(<[f32]>::to_vec);
    let mut v10 = get("V10").map(<[f32]>::to_vec);
    if let (Some(u), Some(vv), true) = (u10.as_mut(), v10.as_mut(), earth) {
        let (s, c) = sin_cos.as_ref().unwrap();
        rotate_pair(u, vv, s, c);
    }
    for (id, plane) in [("u10", u10), ("v10", v10)] {
        match plane {
            Some(p) => {
                surface.insert(id, p);
            }
            None => omitted.push((id.into(), format!("{} is absent", id.to_uppercase()))),
        }
    }
    match &meta.apcp {
        Some(p) => {
            surface.insert("total_precipitation", p.clone());
        }
        None => omitted.push(("total_precipitation".into(), meta.apcp_reason.clone())),
    }

    // ---- Group B.
    let dx = meta.projection.dx;
    let pressure_source = if frame.state.has(rw_post::state::MassInput::PHyd) {
        "history P_HYD on mass levels; interfaces by moist hydrostatic integration (Group A state)"
    } else {
        "moist hydrostatic integration of the dry column mass (Group A state, D1)"
    };
    let vertical_source = frame.coefficients.as_str().to_owned();
    let psfc = get("PSFC").ok_or_else(|| refuse("PSFC is absent"))?;
    let hgt = get("HGT").ok_or_else(|| refuse("HGT is absent"))?;
    let bf = plev::frame::Frame::new(st, psfc, hgt, dx, pressure_source, &vertical_source);
    let levels_pa: Vec<f64> = req.levels_pa();
    let t = Instant::now();
    let breq = plev::Request { levels_pa: levels_pa.clone(), device: device_for_groups, ordinal: 0 };
    #[cfg(any(windows, target_os = "linux"))]
    let b = plev::compute(&bf, &breq, gpu).map_err(fail)?;
    #[cfg(not(any(windows, target_os = "linux")))]
    let b = plev::compute(&bf, &breq).map_err(fail)?;
    timings.insert("group_b", t.elapsed().as_secs_f64());
    devices_used.insert("b", b.device.clone());
    surface.insert("shelter_pressure", a.plane(SurfaceField::ShelterPressure).to_vec());
    surface.insert("pwat", b.pwat.clone());
    surface.insert("mslp", b.mslp.clone());
    surface.insert("maps_mslp", b.maps_mslp.clone());
    let mut plev_planes: Vec<Vec<Vec<f32>>> = (0..levels_pa.len())
        .map(|l| (0..6).map(|f| b.plev_plane(l, f).to_vec()).collect())
        .collect();
    if req.writes_pressure() {
        for (l, planes) in plev_planes.iter_mut().enumerate() {
            for field in &extra.pressure { planes.push(field[l*n..(l+1)*n].to_vec()); }
        }
    }
    if earth {
        let (s, c) = sin_cos.as_ref().unwrap();
        for level in plev_planes.iter_mut() {
            let (lo, hi) = level.split_at_mut(5);
            rotate_pair(&mut lo[4], &mut hi[0], s, c);
        }
    }

    // ---- Group C.
    let t = Instant::now();
    let q2 = get("Q2");
    let (u10s, v10s) = (get("U10"), get("V10"));
    match (q2, u10s, v10s) {
        (Some(q2), Some(u10s), Some(v10s)) => {
            let cs = severe::ColumnState::from_shared(
                st,
                hgt,
                psfc,
                a.plane(SurfaceField::T2),
                q2,
                a.plane(SurfaceField::ShelterPressure),
                u10s,
                v10s,
            );
            #[cfg(any(windows, target_os = "linux"))]
            let (c, cdev) = severe::compute(&cs, gpu, device_for_groups).map_err(|e| fail(format!("group C: {e}")))?;
            #[cfg(not(any(windows, target_os = "linux")))]
            let (c, cdev) = severe::compute(&cs, device_for_groups).map_err(|e| fail(format!("group C: {e}")))?;
            devices_used.insert("c", cdev);
            for r in catalog::ROWS {
                if let Source::C(name) = r.source {
                    let i = severe::plane_index(name).ok_or_else(|| fail(format!("rw-post has no group C plane {name}")))?;
                    surface.insert(r.id, c.planes[i * n..(i + 1) * n].to_vec());
                }
            }
            if earth {
                let (s, cc) = sin_cos.as_ref().unwrap();
                for (u, vv) in [("storm_motion_u", "storm_motion_v"), ("shear_u_0_1km", "shear_v_0_1km"), ("shear_u_0_6km", "shear_v_0_6km")] {
                    let mut pu = surface.remove(u).unwrap();
                    let mut pv = surface.remove(vv).unwrap();
                    rotate_pair(&mut pu, &mut pv, s, cc);
                    surface.insert(u, pu);
                    surface.insert(vv, pv);
                }
            }
        }
        _ => {
            for r in catalog::ROWS {
                if let Source::C(_) = r.source {
                    omitted.push((r.id.into(), "Q2, U10 or V10 is absent, so no surface parcel or 10 m wind exists".into()));
                }
            }
        }
    }
    timings.insert("group_c", t.elapsed().as_secs_f64());

    // ---- Group E.
    let t = Instant::now();
    let ec = owned.carriers(st);
    let opt = group_e::Options { earth_winds: earth, gust: gust_method(&req.gust, ec.tke.is_some()) };
    let gust_source = match opt.gust {
        GustMethod::MixedLayerTke => format!("{}: mixed-layer TKE gust ({})", req.gust, owned.tke_source.unwrap_or("TKE absent")),
        GustMethod::Similarity => format!("{}: surface-layer similarity gust", req.gust),
    };
    #[cfg(any(windows, target_os = "linux"))]
    let e = group_e::compute(&ec, opt, gpu, device_for_groups).map_err(fail)?;
    #[cfg(not(any(windows, target_os = "linux")))]
    let e = group_e::compute_cpu(&ec, opt).map_err(fail)?;
    timings.insert("group_e", t.elapsed().as_secs_f64());
    devices_used.insert("e", e.device.clone());
    for (i, name) in group_e::OUT_NAMES.iter().enumerate() {
        surface.insert(*name, e.planes[i * n..(i + 1) * n].to_vec());
    }
    for (name, why) in &e.omitted {
        surface.remove(name);
        omitted.push(((*name).into(), why.clone()));
    }

    // ---- Group D.
    let t = Instant::now();
    #[cfg(any(windows, target_os = "linux"))]
    let d = group_d::compute(path, st, &frame.state, gpu, device_for_groups, terrain.as_deref()).map_err(fail)?;
    #[cfg(not(any(windows, target_os = "linux")))]
    let d = group_d::compute(path, st, &frame.state, device_for_groups, terrain.as_deref()).map_err(fail)?;
    devices_used.insert("d", d.device);
    let group_d_sources = json!({"cloud_fraction": d.cloud_fraction_source, "aerosol": d.aerosol_source, "simulated_ir": d.ir_source});
    for (k, p) in d.planes {
        surface.insert(k, p);
    }
    for (k, why) in d.omitted {
        omitted.push((k.into(), why));
    }
    timings.insert("group_d", t.elapsed().as_secs_f64());

    // ---- definitions = renderer: the renderer's own diagnostics replace
    // the rows it draws (src/renderer.rs); every other row stays WOOF's.
    let mut renderer_rows = BTreeMap::new();
    let mut renderer_kept = Vec::new();
    if req.definitions_kind() == Definitions::Renderer {
        let t = Instant::now();
        let r = renderer::compute(path, n)?;
        for (id, plane) in r.planes {
            surface.insert(id, plane);
            omitted.retain(|(k, _)| k != id);
            if let Some(row) = renderer::row(id) {
                renderer_rows.insert(id, renderer::method(row));
            }
        }
        renderer_kept = r.kept.into_iter().map(|(k, w)| (k.to_owned(), format!("{w}; WOOF's plane kept"))).collect();
        timings.insert("renderer", t.elapsed().as_secs_f64());
    }

    Ok(Planes {
        surface,
        plev: plev_planes,
        omitted,
        devices: devices_used,
        timings,
        pressure_source: pressure_source.into(),
        vertical_source,
        gust_source,
        group_d_sources,
        renderer_rows,
        renderer_kept,
        state: Some(a.state),
    })
}

/// A row as the request writes it: `composite_reflectivity` takes the
/// request's level type (`composite_level_type`, the front door's
/// `--upp-control` names).
pub fn as_requested(mut r: Row, req: &Request) -> Row {
    if r.id == "composite_reflectivity" {
        r.level.0 = req.composite_level_type;
    }
    r
}

/// The statistical-process part of a template 4.8 message.
#[derive(Clone, Copy, Debug)]
pub struct Stat {
    /// Code table 4.10.
    pub process: u8,
    /// Code table 4.4 unit of the offset and the length.
    pub unit: u8,
    /// Forecast time of the window start, in `unit`.
    pub offset: u32,
    /// Window length, in `unit`.
    pub length: u32,
}

/// One message.
#[allow(clippy::too_many_arguments)]
fn message(
    row: &Row,
    level_value: Option<f64>,
    plane: &[f32],
    grid: &GridDefinition,
    reference: NaiveDateTime,
    lead: (u8, u32),
    valid: NaiveDateTime,
    req: &Request,
    stat: Option<Stat>,
) -> MessageBuilder {
    let values: Vec<f64> = plane.iter().map(|&x| if x.is_finite() { f64::from(x) } else { f64::NAN }).collect();
    let accumulated = matches!(row.source, Source::Apcp);
    // Accumulation from the start: offset 0, length = the lead.
    let stat = stat.or(accumulated.then_some(Stat { process: 1, unit: lead.0, offset: 0, length: lead.1 }));
    let product = ProductDefinition {
        template: if stat.is_some() { 8 } else { 0 },
        parameter_category: row.category,
        parameter_number: row.number,
        generating_process: GENERATING_PROCESS,
        forecast_time: stat.map_or(lead.1, |s| s.offset),
        time_range_unit: stat.map_or(lead.0, |s| s.unit),
        level_type: row.level.0,
        level_value: level_value.unwrap_or(row.level.1),
    };
    let packing = if req.packing == "simple" {
        PackingMethod::Simple { bits_per_value: req.bits }
    } else {
        PackingMethod::ComplexSpatial { bits_per_value: req.bits, order: 2 }
    };
    // Bit 5 of the resolution and component flags (code table 3.3): vector
    // components resolved relative to the grid (0x08) or to east/north.
    let flags = if req.winds == "grid" && !matches!(row.source, Source::WindHeight(..)) { 0x38 } else { 0x30 };
    let mut m = MessageBuilder::new(row.discipline, values)
        .center(centre_for(row), 0)
        .generating_process_identifier(GENERATING_PROCESS_ID)
        .master_table_version(MASTER_TABLES)
        .local_table_version(LOCAL_TABLES)
        .reference_time(reference)
        .grid(grid.clone())
        .product(product)
        .packing(packing)
        .earth_radius(grid::WRF_EARTH_RADIUS_M)
        .resolution_flags(flags);
    if let Some((t, v)) = row.second {
        m = m.second_surface(t, v);
    }
    if let Some(st) = stat {
        m = m.statistical_interval(StatisticalInterval { end_time: valid, statistical_process: st.process, time_unit: st.unit, length: st.length });
    }
    m
}

pub(crate) fn encode(messages: Vec<MessageBuilder>) -> Result<Vec<u8>> {
    let parts: Vec<std::result::Result<Vec<u8>, String>> =
        messages.into_par_iter().map(|m| Grib2Writer::new().add_message(m).to_bytes()).collect();
    let mut out = Vec::new();
    for p in parts {
        out.extend_from_slice(&p.map_err(|e| fail(format!("GRIB2 encoding: {e}")))?);
    }
    Ok(out)
}

pub(crate) fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    let tmp = path.with_extension("grib2.part");
    std::fs::write(&tmp, bytes).map_err(|e| fail(format!("write {}: {e}", tmp.display())))?;
    std::fs::rename(&tmp, path).map_err(|e| fail(format!("rename {}: {e}", path.display())))?;
    Ok(())
}

fn row_json(r: &Row, level: Option<f64>) -> Value {
    let mut level_text = if matches!(r.source, Source::Plev(_)) {
        "100/pressure_Pa".to_owned()
    } else {
        catalog::level_text(r)
    };
    if let Some(p) = level {
        level_text = format!("{}/{}", r.level.0, p);
    }
    json!({"id": r.id, "grib": catalog::grib_code(r), "level": level_text, "units": r.units, "method": r.method})
}

/// What one frame produced (manifest record).
fn export_frame(
    path: &Path,
    candidate: &Candidate,
    req: &Request,
    devices: &mut Devices,
    meta: FrameMeta,
    ext: Option<&mut extrema::Context>,
    emit: &mut dyn FnMut(Value),
) -> Result<(Value, u64)> {
    let t0 = Instant::now();
    let planes = compute(path, req, devices, &meta)?;
    let frame_extrema: Option<FrameExtrema> = match ext {
        Some(ctx) => Some(ctx.frame(&meta.domain, meta.valid, meta.start, path, meta.nx * meta.ny, &req.out.join(".scratch"))?),
        None => None,
    };
    let grid = grid::definition(&meta.projection, meta.nx, meta.ny, &meta.lat, &meta.lon)?;
    let reference = utc(meta.start)?;
    let valid = utc(meta.valid)?;
    let lead = lead(meta.valid - meta.start)?;
    let stamp = compact(meta.valid);
    let mut files = Vec::new();
    let mut total = 0u64;
    let mut omitted: Vec<Value> = planes.omitted.iter().map(|(k, w)| json!({"id": k, "reason": w})).collect();
    let omitted_ids: std::collections::BTreeSet<&str> = planes.omitted.iter().map(|(k, _)| k.as_str()).collect();

    if req.writes_surface() {
        let mut messages = Vec::new();
        let mut written = Vec::new();
        for r in catalog::surface_rows() {
            let r = as_requested(r, req);
            if omitted_ids.contains(r.id) {
                continue;
            }
            let Some(plane) = planes.surface.get(r.id) else {
                omitted.push(json!({"id": r.id, "reason": "not computed for this frame"}));
                continue;
            };
            if all_missing(plane) {
                let needs = SurfaceField::PRODUCTS.iter().find(|s| s.id == r.id).map(|s| s.needs.join(", "));
                let reason = match needs {
                    Some(n) => format!("missing in every cell (needs {n})"),
                    None => "missing in every cell".into(),
                };
                omitted.push(json!({"id": r.id, "reason": reason}));
                continue;
            }
            dump(req, &format!("wrfsfc_{}_{}", meta.domain, stamp), messages.len(), r.id, None, plane)?;
            messages.push(message(&r, None, plane, &grid, reference, lead, valid, req, None));
            let mut j = row_json(&r, None);
            if let Some(m) = planes.renderer_rows.get(r.id) {
                j["method"] = json!(m);
                j["definitions"] = json!("renderer");
            }
            written.push(j);
        }
        // Extrema over the request's window (template 4.8).
        if let Some(fx) = &frame_extrema {
            for (id, why) in &fx.omitted {
                omitted.push(json!({"id": id, "reason": why}));
            }
            if let Some(w) = fx.window {
                let (unit, offset, length) = extrema::time_range(w, meta.start).map_err(refuse)?;
                for (i, plane) in &fx.planes {
                    let e = &EXTREMA[*i];
                    if all_missing(plane) {
                        omitted.push(json!({"id": e.row.id, "reason": "missing in every cell"}));
                        continue;
                    }
                    let stat = Stat { process: e.process, unit, offset, length };
                    dump(req, &format!("wrfsfc_{}_{}", meta.domain, stamp), messages.len(), e.row.id, None, plane)?;
                    messages.push(message(&e.row, None, plane, &grid, reference, lead, valid, req, Some(stat)));
                    let mut j = row_json(&e.row, None);
                    j["window"] = json!([times::iso(w.start), times::iso(w.end)]);
                    j["statistical_process"] = json!(e.process);
                    written.push(j);
                }
            }
        }
        let bytes = encode(messages)?;
        let dest = req.out.join(format!("wrfsfc_{}_{}.grib2", meta.domain, stamp));
        write_atomic(&dest, &bytes)?;
        total += bytes.len() as u64;
        emit(json!({"event": "file", "path": dest.display().to_string(), "fields": written.len(), "bytes": bytes.len()}));
        files.push(json!({"path": dest.file_name().unwrap().to_string_lossy(), "kind": "surface", "bytes": bytes.len(), "fields": written}));
    }
    if req.writes_pressure() {
        let mut messages = Vec::new();
        let mut written = Vec::new();
        let levels = req.levels_hpa();
        for (li, &hpa) in levels.iter().enumerate() {
            let pa = f64::from(hpa) * 100.0;
            for r in catalog::PLEV_ROWS.iter() {
                let Source::Plev(fi) = r.source else { continue };
                let plane = &planes.plev[li][fi];
                if all_missing(plane) {
                    omitted.push(json!({"id": r.id, "level_pa": pa, "reason": "missing in every cell (above the model top)"}));
                    continue;
                }
                dump(req, &format!("wrfprs_{}_{}", meta.domain, stamp), messages.len(), r.id, Some(pa), plane)?;
                messages.push(message(r, Some(pa), plane, &grid, reference, lead, valid, req, None));
                written.push(row_json(r, Some(pa)));
            }
        }
        let bytes = encode(messages)?;
        let dest = req.out.join(format!("wrfprs_{}_{}.grib2", meta.domain, stamp));
        write_atomic(&dest, &bytes)?;
        total += bytes.len() as u64;
        emit(json!({"event": "file", "path": dest.display().to_string(), "fields": written.len(), "bytes": bytes.len()}));
        files.push(json!({"path": dest.file_name().unwrap().to_string_lossy(), "kind": "pressure", "bytes": bytes.len(), "fields": written}));
    }
    let seconds = t0.elapsed().as_secs_f64();
    let record = json!({
        "source": candidate.describe(),
        "domain": meta.domain,
        "valid": times::iso(meta.valid),
        "reference_time": times::iso(meta.start),
        "forecast_seconds": meta.valid - meta.start,
        "files": files,
        "omitted": omitted,
        "devices": planes.devices,
        "timings": planes.timings,
        "pressure_source": planes.pressure_source,
        "vertical_source": planes.vertical_source,
        "group_d_sources": planes.group_d_sources,
        "definitions": if planes.renderer_rows.is_empty() && req.definitions_kind() == Definitions::Woof {
            json!("woof")
        } else {
            json!({
                "renderer": planes.renderer_rows.keys().collect::<Vec<_>>(),
                "woof_kept": planes.renderer_kept.iter().map(|(k, w)| json!({"id": k, "reason": w})).collect::<Vec<_>>(),
            })
        },
        "extrema_window": frame_extrema.as_ref().and_then(|f| f.window).map(|w| json!([times::iso(w.start), times::iso(w.end)])),
        "winds": req.winds,
        "gust": planes.gust_source,
        "packing": req.packing,
        "bits": req.bits,
        "seconds": seconds,
    });
    emit(json!({"event": "frame", "domain": meta.domain, "valid": times::iso(meta.valid), "devices": record["devices"], "seconds": seconds}));
    Ok((record, total))
}

fn identity(path: &Path) -> String {
    let md = std::fs::metadata(path).ok();
    let size = md.as_ref().map_or(0, |m| m.len());
    let mtime = md
        .and_then(|m| m.modified().ok())
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map_or(0, |d| d.as_nanos());
    let canon = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    format!("{}|{size}|{mtime}", canon.display())
}

fn load_state(out: &Path) -> BTreeMap<String, Value> {
    std::fs::read_to_string(out.join(STATE_FILE))
        .ok()
        .and_then(|t| serde_json::from_str::<BTreeMap<String, Value>>(&t).ok())
        .unwrap_or_default()
}

fn save_state(out: &Path, state: &BTreeMap<String, Value>) -> Result<()> {
    let bytes = serde_json::to_vec_pretty(state).map_err(|e| fail(e.to_string()))?;
    let tmp = out.join(format!("{STATE_FILE}.part"));
    std::fs::write(&tmp, bytes)?;
    std::fs::rename(&tmp, out.join(STATE_FILE))?;
    Ok(())
}

fn write_manifest(out: &Path, req: &Request, state: &BTreeMap<String, Value>, finalized: bool) -> Result<()> {
    let mut frames: Vec<&Value> = state.values().collect();
    frames.sort_by(|a, b| (a["domain"].as_str(), a["valid"].as_str()).cmp(&(b["domain"].as_str(), b["valid"].as_str())));
    let mut surface: Vec<Value> = catalog::surface_rows().into_iter().map(|r| row_json(&as_requested(r, req), None)).collect();
    if req.extrema_interval_seconds.is_some() {
        surface.extend(EXTREMA.iter().map(|e| {
            let mut j = row_json(&e.row, None);
            j["statistical_process"] = json!(e.process);
            j["carrier"] = json!(e.carrier);
            j
        }));
    }
    let pressure: Vec<Value> = catalog::PLEV_ROWS.iter().map(|r| row_json(r, None)).collect();
    let definitions = match req.definitions_kind() {
        Definitions::Woof => "woof",
        Definitions::Renderer => "renderer",
    };
    let manifest = json!({
        "schema": "grib2-export.manifest/v1",
        "catalog": CATALOG,
        "profile": PROFILE,
        "definitions": definitions,
        "generating_process_id": GENERATING_PROCESS_ID,
        "generating_process_id_meaning": GENERATING_PROCESS_ID_TEXT,
        "composite_level_type": req.composite_level_type,
        "extrema_interval_seconds": req.extrema_interval_seconds,
        "exporter": format!("rw_grib2export {}", env!("CARGO_PKG_VERSION")),
        "post_device": req.post_device,
        "levels_hpa": req.levels_hpa(),
        "grid_geometry": "wrf-native-locations/v1: WRF sphere 6370000 m encoded as earth shape 1, scanning mode 0x40",
        "centre": CENTRE,
        "finalized": finalized,
        "catalog_rows": {"surface": surface, "pressure": pressure},
        "frames": frames,
    });
    let tmp = out.join(format!("{MANIFEST}.part"));
    std::fs::write(&tmp, serde_json::to_vec_pretty(&manifest).map_err(|e| fail(e.to_string()))?)?;
    std::fs::rename(&tmp, out.join(MANIFEST))?;
    Ok(())
}

/// The sibling archive `<out>-grib2.zip`.
pub fn archive_path(out: &Path) -> PathBuf {
    let name = out.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| "grib2".into());
    out.with_file_name(format!("{name}-grib2.zip"))
}

fn build_archive(out: &Path, emit: &mut dyn FnMut(Value)) -> Result<u64> {
    let prefix = out.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| "grib2".into());
    let files: Vec<(String, PathBuf)> = zipout::collect(out, &prefix)?
        .into_iter()
        .filter(|(name, _)| {
            let base = name.rsplit('/').next().unwrap_or(name);
            !base.starts_with('.') && !base.ends_with(".part")
        })
        .collect();
    let dest = archive_path(out);
    let bytes = zipout::write(&dest, &files)?;
    emit(json!({"event": "archive", "path": dest.display().to_string(), "bytes": bytes, "files": files.len()}));
    Ok(bytes)
}

/// Carry out one request.
pub fn run(req: &Request, emit: &mut dyn FnMut(Value)) -> Result<Value> {
    let t0 = Instant::now();
    if req.definitions == "upp" {
        eprintln!(
            "rw_grib2export: definitions `upp` is now `woof` (catalog woof-post/v1); the alias is accepted for this release only"
        );
    }
    std::fs::create_dir_all(&req.out).map_err(|e| fail(format!("create {}: {e}", req.out.display())))?;
    let mut state = if req.mode == "run" { BTreeMap::new() } else { load_state(&req.out) };
    let mut frames = 0usize;
    let mut bytes = 0u64;
    if req.mode != "finalize" {
        let candidates = inputs::discover(&req.inputs)?;
        let wanted_times: Vec<i64> = req
            .times
            .iter()
            .map(|t| times::parse(t).ok_or_else(|| refuse(format!("time {t} is not a UTC time"))))
            .collect::<Result<_>>()?;
        let start = req.start.as_deref().map(|t| times::parse_named(t, "start")).transpose()?;
        let end = req.end.as_deref().map(|t| times::parse_named(t, "end")).transpose()?;
        let scratch = req.out.join(".scratch");
        let mut devices = Devices::new(req.device());
        let mut ext = match req.extrema_interval_seconds {
            Some(seconds) => Some(extrema::Context::new(&req.out, seconds, req.mode == "run", &candidates)?),
            None => None,
        };
        for candidate in &candidates {
            if let Some((d, t)) = &candidate.hint {
                if !req.domains.is_empty() && !req.domains.contains(d) {
                    continue;
                }
                if (!wanted_times.is_empty() && !wanted_times.contains(t)) || start.is_some_and(|s| *t < s) || end.is_some_and(|e| *t > e) {
                    continue;
                }
            }
            let on_disk = inputs::materialize(candidate, &scratch)?;
            let key = match &candidate.source {
                inputs::Source::File(p) => identity(p),
                _ => format!("{}|{}", candidate.describe(), identity(&on_disk.path).rsplit('|').nth(1).unwrap_or("")),
            };
            if req.mode == "append" && state.contains_key(&key) {
                emit(json!({"event": "skipped", "source": candidate.describe(), "reason": "already exported"}));
                continue;
            }
            let meta = frame_meta(&on_disk.path, candidate.hint.as_ref())?;
            if !req.domains.is_empty() && !req.domains.contains(&meta.domain) {
                continue;
            }
            if (!wanted_times.is_empty() && !wanted_times.contains(&meta.valid))
                || start.is_some_and(|s| meta.valid < s)
                || end.is_some_and(|e| meta.valid > e)
            {
                continue;
            }
            let exported = export_frame(&on_disk.path, candidate, req, &mut devices, meta, ext.as_mut(), emit);
            if let Some(message) = devices.take_notice() {
                eprintln!("rw_grib2export: {message}");
                emit(json!({"event": "warning", "code": "post_device_cpu", "message": message}));
            }
            let (record, b) = exported?;
            state.insert(key, record);
            frames += 1;
            bytes += b;
            save_state(&req.out, &state)?;
            write_manifest(&req.out, req, &state, false)?;
        }
        let _ = std::fs::remove_dir(&scratch);
        if frames == 0 && req.mode == "run" {
            return Err(refuse("no frame matched the request's inputs, domains and times, so nothing was exported"));
        }
    }
    let finalized = req.mode != "append";
    write_manifest(&req.out, req, &state, finalized)?;
    if req.zip && req.mode != "append" {
        build_archive(&req.out, emit)?;
    }
    let done = json!({"event": "done", "frames": frames, "bytes": bytes, "seconds": t0.elapsed().as_secs_f64(), "out": req.out.display().to_string()});
    emit(done.clone());
    Ok(done)
}

/// `--probe-post-device`: does the first card open with the post kernels?
/// `gpuwm doctor` reads it for its "GPU post kernels" line, so a card whose
/// driver refuses the kernels is named before a run falls back to the CPU.
pub fn probe_post_device() -> Value {
    #[cfg(any(windows, target_os = "linux"))]
    {
        let floor = rw_post::gpu::PTX_DRIVER_FLOOR;
        if !rw_post::gpu::driver_present() {
            return json!({"ok": false, "driver": false, "error": "no NVIDIA driver library could be loaded", "ptx_driver_floor": floor});
        }
        match rw_post::gpu::GpuPost::open(0) {
            Ok(g) => json!({"ok": true, "driver": true, "device": g.device.describe(), "ptx_driver_floor": floor}),
            Err(e) => {
                // Which step failed: a card that opens but refuses the kernel
                // images ("load post kernels") is the driver-floor case.
                let operation = match &e {
                    rw_post::gpu::GpuError::Driver { operation, .. } => *operation,
                    rw_post::gpu::GpuError::Input(_) => "input",
                };
                json!({"ok": false, "driver": true, "operation": operation, "error": e.to_string(), "ptx_driver_floor": floor})
            }
        }
    }
    #[cfg(not(any(windows, target_os = "linux")))]
    {
        json!({"ok": false, "driver": false, "error": "this platform has no CUDA post path"})
    }
}

/// `--list`: the catalog and options as JSON.
pub fn list() -> Value {
    let surface: Vec<Value> = catalog::surface_rows()
        .iter()
        .map(|r| {
            let mut v = row_json(r, None);
            v["group"] = json!(match r.source {
                Source::A(_) => "A",
                Source::Pwat | Source::Mslp | Source::MapsMslp | Source::Plev(_) => "B",
                Source::C(_) => "C",
                Source::D(_) => "D",
                Source::E(_) => "E",
                Source::U10 | Source::V10 | Source::Apcp | Source::WindHeight(..) => "history",
                Source::Extreme(_) => "extrema",
            });
            v
        })
        .collect();
    json!({
        "catalog": CATALOG,
        "profile": PROFILE,
        "fields": ["standard", "surface", "pressure", "all"],
        "packing": ["complex", "simple"],
        "definitions": ["woof", "renderer", "arwen (alias of renderer)", "upp (alias of woof, this release only)"],
        "renderer_rows": renderer::ROWS.iter().map(|r| json!({"id": r.id, "getvar": r.var})).collect::<Vec<_>>(),
        "extrema_rows": EXTREMA.iter().map(|e| {
            let mut j = row_json(&e.row, None);
            j["carrier"] = json!(e.carrier);
            j["statistical_process"] = json!(e.process);
            j
        }).collect::<Vec<_>>(),
        "composite_level_types": [crate::request::COMPOSITE_LEVEL_SRW, crate::request::COMPOSITE_LEVEL_WMO],
        "generating_process_id": GENERATING_PROCESS_ID,
        "post_device": ["auto", "gpu", "cpu"],
        "winds": ["grid", "earth"],
        "gust": ["auto", "similarity", "tke"],
        "default_levels_hpa": catalog::default_levels_hpa(),
        "surface_fields": surface,
        "pressure_fields": catalog::PLEV_ROWS.iter().map(|r| r.id).collect::<Vec<_>>(),
        "pressure_rows": catalog::PLEV_ROWS.iter().map(|r| row_json(r, None)).collect::<Vec<_>>(),
    })
}

#[cfg(test)]
mod tests {

    /// RULINGS 7 default: TKE gust where the history carries TKE, the
    /// similarity gust where it does not (so gust is never dropped for
    /// want of TKE), and either one when asked for by name.  Breakage
    /// prevented: the ASOS-scored winner silently not running by default.
    #[test]
    fn gust_auto_takes_tke_when_carried_and_similarity_otherwise() {
        assert_eq!(gust_method("auto", true), GustMethod::MixedLayerTke);
        assert_eq!(gust_method("auto", false), GustMethod::Similarity);
        assert_eq!(gust_method("similarity", true), GustMethod::Similarity);
        assert_eq!(gust_method("tke", false), GustMethod::MixedLayerTke);
        assert_eq!(crate::request::Request::default_gust_for_tests(), "auto");
    }

    use super::*;

    #[test]
    fn lead_units() {
        assert_eq!(lead(7200).unwrap(), (1, 2));
        assert_eq!(lead(900).unwrap(), (0, 15));
        assert_eq!(lead(45).unwrap(), (13, 45));
        assert!(lead(-1).is_err());
    }

    #[test]
    fn catalog_ids_unique() {
        let mut ids = std::collections::BTreeSet::new();
        for r in catalog::surface_rows() {
            assert!(ids.insert(r.id), "duplicate {}", r.id);
        }
        assert_eq!(catalog::default_levels_hpa().len(), 37);
    }

    #[test]
    fn added_rows_roundtrip_wmo_identity_template_units_and_wind_flags() {
        let (grid, n) = small_grid();
        let reference = utc(1_759_449_600).unwrap();
        let req = request("complex"); // Default grid winds must not mislabel earth-relative 100 m winds.
        for (id, code, units, level) in [
            ("specific_humidity", (0,1,0), "kg kg-1", (100,75000.)),
            ("omega", (0,2,8), "Pa s-1", (100,75000.)),
            ("vertical_velocity", (0,2,9), "m s-1", (100,75000.)),
            ("u100", (0,2,2), "m s-1", (103,100.)),
            ("v100", (0,2,3), "m s-1", (103,100.)),
        ] {
            let row = catalog::PLEV_ROWS.iter().chain(catalog::ROWS).find(|r| r.id==id).unwrap();
            assert_eq!(row.units, units);
            assert!(!row.method.is_empty());
            let bytes = encode(vec![message(row, (level.0==100).then_some(75000.), &vec![0.25;n], &grid, reference, (1,0), reference, &req, None)]).unwrap();
            let file = wx_core::grib2::Grib2File::from_bytes(&bytes).unwrap();
            let msg = &file.messages[0];
            assert_eq!((msg.discipline,msg.product.parameter_category,msg.product.parameter_number), code);
            assert_eq!((msg.product.level_type,msg.product.level_value), level);
            assert_eq!(&section4(&bytes)[7..9], &[0,0]); // product template 4.0
            if level.0==103 {
                let mut at=16;
                while bytes[at+4]!=3 { at+=u32::from_be_bytes(bytes[at..at+4].try_into().unwrap()) as usize; }
                assert_eq!(bytes[at+46]&0x08, 0, "{id} must be earth-relative");
            }
        }
    }

    fn request(packing: &str) -> Request {
        serde_json::from_value(json!({
            "schema": crate::request::SCHEMA, "inputs": ["x"], "out": "/tmp/x", "packing": packing,
        }))
        .unwrap()
    }

    /// Pack a plane with missing cells through `message`, decode it with the
    /// in-tree wx-core reader, and check codes, both fixed surfaces, the
    /// bitmap and every value against half the packing quantum.
    #[test]
    fn message_round_trips_codes_bitmap_and_values() {
        let p = Projection { map_proj: 1, truelat1: 38.5, truelat2: 38.5, stand_lon: -97.5, pole_lat: 90.0, pole_lon: 0.0, dx: 3000.0, dy: 3000.0 };
        let (nx, ny) = (7usize, 5usize);
        let lat: Vec<f64> = (0..nx * ny).map(|i| 30.0 + (i / nx) as f64 * 0.03).collect();
        let lon: Vec<f64> = (0..nx * ny).map(|i| -100.0 + (i % nx) as f64 * 0.03).collect();
        let g = grid::definition(&p, nx, ny, &lat, &lon).unwrap();
        let plane: Vec<f32> = (0..nx * ny).map(|i| if i % 6 == 0 { f32::NAN } else { 250.0 + (i as f32) * 0.731 }).collect();
        let row = catalog::ROWS.iter().find(|r| r.id == "mlcape").unwrap();
        let reference = utc(1_759_449_600).unwrap();
        let valid = utc(1_759_449_600 + 3 * 3600).unwrap();
        for packing in ["complex", "simple"] {
            let req = request(packing);
            let m = message(row, None, &plane, &g, reference, (1, 3), valid, &req, None);
            let bytes = encode(vec![m]).unwrap();
            let file = wx_core::grib2::Grib2File::from_bytes(&bytes).unwrap();
            assert_eq!(file.messages.len(), 1);
            let msg = &file.messages[0];
            assert_eq!((msg.discipline, msg.product.parameter_category, msg.product.parameter_number), (0, 7, 6));
            assert_eq!(msg.product.level_type, 108);
            assert!((msg.product.level_value - 9000.0).abs() < 1e-6);
            assert_eq!(msg.product.forecast_time, 3);
            assert_eq!((msg.grid.nx as usize, msg.grid.ny as usize, msg.grid.template), (nx, ny, 30));
            let bitmap = msg.bitmap.as_ref().expect("missing cells need a bitmap");
            let values = wx_core::grib2::unpack_message(msg).unwrap();
            let q = 2f64.powi(i32::from(msg.data_rep.binary_scale)) * 10f64.powi(-i32::from(msg.data_rep.decimal_scale));
            for (i, &x) in plane.iter().enumerate() {
                assert_eq!(bitmap[i], x.is_finite(), "bitmap cell {i}");
                if x.is_finite() {
                    assert!((values[i] - f64::from(x)).abs() <= 0.5 * q + 1e-9, "{packing} cell {i}: {} vs {x}", values[i]);
                }
            }
        }
    }

    #[test]
    fn local_category_uses_the_missing_centre() {
        let shear = catalog::ROWS.iter().find(|r| r.id == "bulk_shear_0_6km").unwrap();
        let cape = catalog::ROWS.iter().find(|r| r.id == "sbcape").unwrap();
        assert_eq!(centre_for(shear), CENTRE_LOCAL);
        assert_eq!(centre_for(cape), CENTRE);
    }

    /// simulated_ir keeps 2.8.5's identity (0.192.0 at the nominal top of
    /// the atmosphere, level type 8, K) under the missing centre, so a
    /// decoder that read 2.8.5's message finds the same one (RULINGS 12).
    #[test]
    fn simulated_ir_keeps_its_identity() {
        let ir = catalog::ROWS.iter().find(|r| r.id == "simulated_ir").unwrap();
        assert_eq!((ir.discipline, ir.category, ir.number, ir.level, ir.second, ir.units), (0, 192, 0, (8, 0.0), None, "K"));
        assert_eq!(ir.source, Source::D("simulated_ir"));
        assert_eq!(centre_for(ir), CENTRE_LOCAL);
        assert!(rw_post::group_d::plane_index("simulated_ir").is_some());
    }

    #[test]
    fn request_refusals_name_the_breakage() {
        let mut r = request("complex");
        assert!(r.validate().is_ok());
        r.bits = 40;
        assert!(r.validate().unwrap_err().message().contains("bits"));
        let mut r = request("complex");
        r.definitions = "renderer".into();
        assert!(r.validate().is_ok());
        assert_eq!(r.definitions_kind(), Definitions::Renderer);
        r.definitions = "arwen".into();
        assert_eq!(r.definitions_kind(), Definitions::Renderer);
        r.definitions = "upp".into();
        assert_eq!(r.definitions_kind(), Definitions::Woof);
        r.definitions = "ncep".into();
        assert!(r.validate().is_err());
        let mut r = request("complex");
        r.extrema_interval_seconds = Some(3600);
        assert!(r.validate().is_ok());
        r.extrema_interval_seconds = Some(0);
        assert!(r.validate().is_err());
        r.extrema_interval_seconds = Some(86_401);
        assert!(r.validate().is_err());
        let mut r = request("complex");
        assert_eq!(r.composite_level_type, 200);
        r.composite_level_type = 10;
        assert!(r.validate().is_ok());
        r.composite_level_type = 11;
        assert!(r.validate().unwrap_err().message().contains("composite"));
        let mut r = request("complex");
        r.levels = vec![500, 500];
        assert!(r.validate().is_err());
        assert_eq!(request("complex").device(), PostDevice::Auto);
    }

    /// Section 4 of the first message in `bytes` (octet 1 at index 0).
    fn section4(bytes: &[u8]) -> Vec<u8> {
        let mut at = 16;
        loop {
            let len = u32::from_be_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]]) as usize;
            if bytes[at + 4] == 4 {
                return bytes[at..at + len].to_vec();
            }
            at += len;
        }
    }

    fn small_grid() -> (GridDefinition, usize) {
        let p = Projection { map_proj: 1, truelat1: 38.5, truelat2: 38.5, stand_lon: -97.5, pole_lat: 90.0, pole_lon: 0.0, dx: 3000.0, dy: 3000.0 };
        let (nx, ny) = (4usize, 3usize);
        let lat: Vec<f64> = (0..nx * ny).map(|i| 30.0 + (i / nx) as f64 * 0.03).collect();
        let lon: Vec<f64> = (0..nx * ny).map(|i| -100.0 + (i % nx) as f64 * 0.03).collect();
        (grid::definition(&p, nx, ny, &lat, &lon).unwrap(), nx * ny)
    }

    /// RULINGS 12: centre 7 stays, octet 14 identifies WOOF on every row,
    /// including the centre-255 local rows.
    #[test]
    fn generating_process_identifies_woof() {
        let (g, n) = small_grid();
        let plane: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let reference = utc(1_759_449_600).unwrap();
        for id in ["sbcape", "bulk_shear_0_6km"] {
            let row = catalog::ROWS.iter().find(|r| r.id == id).unwrap();
            let bytes = encode(vec![message(row, None, &plane, &g, reference, (1, 3), reference, &request("simple"), None)]).unwrap();
            let s4 = section4(&bytes);
            assert_eq!(s4[11], GENERATING_PROCESS, "octet 12: forecast");
            assert_eq!(s4[13], GENERATING_PROCESS_ID, "octet 14: WOOF");
            assert_eq!(u16::from_be_bytes([bytes[21], bytes[22]]), centre_for(row), "section 1 centre");
        }
        assert_eq!(CENTRE, 7);
        assert_eq!(GENERATING_PROCESS_ID, 254);
    }

    /// `--upp-control` names choose the composite level label; nothing
    /// else changes.
    #[test]
    fn composite_level_follows_the_request() {
        let mut req = request("simple");
        let comp = *catalog::ROWS.iter().find(|r| r.id == "composite_reflectivity").unwrap();
        assert_eq!(as_requested(comp, &req).level, (200, 0.0));
        req.composite_level_type = crate::request::COMPOSITE_LEVEL_WMO;
        assert_eq!(as_requested(comp, &req).level, (10, 0.0));
        let other = *catalog::ROWS.iter().find(|r| r.id == "reflectivity_1km").unwrap();
        assert_eq!(as_requested(other, &req).level, (103, 1000.0));
        let (g, n) = small_grid();
        let plane = vec![35.0f32; n];
        let reference = utc(1_759_449_600).unwrap();
        let bytes = encode(vec![message(&as_requested(comp, &req), None, &plane, &g, reference, (1, 3), reference, &req, None)]).unwrap();
        let file = wx_core::grib2::Grib2File::from_bytes(&bytes).unwrap();
        assert_eq!(file.messages[0].product.level_type, 10);
    }

    /// An extrema row is template 4.8 with the window's process, unit,
    /// start offset and length (WMO-No. 306 template 4.8 octets 47-53).
    #[test]
    fn extrema_message_carries_the_window() {
        let (g, n) = small_grid();
        let plane: Vec<f32> = (0..n).map(|i| 10.0 + i as f32).collect();
        let start = 1_759_449_600i64;
        let reference = utc(start).unwrap();
        let valid = utc(start + 3 * 3600).unwrap();
        let w = extrema::Window { start: start + 2 * 3600, end: start + 3 * 3600 };
        let (unit, offset, length) = extrema::time_range(w, start).unwrap();
        for e in [&EXTREMA[0], &EXTREMA[1], &EXTREMA[8]] {
            let stat = Stat { process: e.process, unit, offset, length };
            let bytes = encode(vec![message(&e.row, None, &plane, &g, reference, (1, 3), valid, &request("simple"), Some(stat))]).unwrap();
            let s4 = section4(&bytes);
            assert_eq!(u16::from_be_bytes([s4[7], s4[8]]), 8, "template 4.8");
            assert_eq!((s4[9], s4[10]), (e.row.category, e.row.number));
            assert_eq!(s4[17], 1, "hours");
            assert_eq!(u32::from_be_bytes([s4[18], s4[19], s4[20], s4[21]]), 2, "window starts at hour 2");
            assert_eq!(s4[46], e.process, "statistical process");
            assert_eq!(s4[48], 1, "length unit");
            assert_eq!(u32::from_be_bytes([s4[49], s4[50], s4[51], s4[52]]), 1, "one hour");
            let file = wx_core::grib2::Grib2File::from_bytes(&bytes).unwrap();
            let values = wx_core::grib2::unpack_message(&file.messages[0]).unwrap();
            assert!((values[3] - 13.0).abs() < 1e-3);
        }
        // Precipitation keeps its accumulation from the start.
        let apcp = catalog::ROWS.iter().find(|r| r.id == "total_precipitation").unwrap();
        let bytes = encode(vec![message(apcp, None, &plane, &g, reference, (1, 3), valid, &request("simple"), None)]).unwrap();
        let s4 = section4(&bytes);
        assert_eq!((s4[46], u32::from_be_bytes([s4[18], s4[19], s4[20], s4[21]])), (1, 0));
    }

    #[test]
    fn archive_is_a_sibling() {
        assert_eq!(archive_path(Path::new("/r/grib2")), PathBuf::from("/r/grib2-grib2.zip"));
    }
}
