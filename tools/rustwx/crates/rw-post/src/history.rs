//! Read one frame of a WRF-format WOOF history into the packed inputs
//! (specification 3.1).  Absent optional carriers are recorded, never
//! filled with zeros that a product could mistake for data.

use std::path::Path;

use rayon::prelude::*;

use thiserror::Error;

use crate::hybrid::{CoefficientSource, HybridError, InterfaceCoefficients};
use crate::solar::{SolarFrame, UtcTime};
use crate::state::{MassInput, Shape, StateInputs};
use crate::surface::{SurfaceInput, SurfaceInputs, SurfaceOptions};

#[derive(Debug, Error)]
pub enum HistoryError {
    #[error("read {path}: {source}")]
    Open { path: String, source: netcrust::Error },
    #[error("variable {name}: {message}")]
    Variable { name: String, message: String },
    #[error("missing required carrier {0}")]
    Missing(&'static str),
    #[error("bad valid time {0:?}")]
    Time(String),
    #[error(transparent)]
    Hybrid(#[from] HybridError),
}

/// One history frame, ready for either device.
pub struct Frame {
    pub shape: Shape,
    pub valid: UtcTime,
    pub valid_text: String,
    pub state: StateInputs,
    /// 2D carriers; the lowest-level virtual temperature slot is filled
    /// after the state is built ([`Frame::attach_state`]).
    pub surface: SurfaceInputs,
    pub options: SurfaceOptions,
    pub solar: SolarFrame,
    pub coefficients: CoefficientSource,
    pub sinalpha: Option<Vec<f32>>,
    pub cosalpha: Option<Vec<f32>>,
    pub land_use_table: Option<String>,
    pub sf_surface_physics: Option<i32>,
    /// Every carrier name that was present.
    pub carriers: Vec<String>,
}

/// The history's `Times` labels, one per record, on either container.
///
/// A classic file decodes the `Times(Time, DateStrLen)` char matrix one
/// string per record.  A netCDF-4 file (WRF's own `io_form_history = 11`,
/// for example) stores NC_CHAR as one-byte HDF5 strings, and the reader
/// returns one element per character; those are joined back along
/// `DateStrLen` here.  Breakage this prevents: every netCDF-4 WRF history
/// was refused with "holds 19 times" (b1-height lane, 2026-10-05), so the
/// exporter could not run on WRF's own output.
pub fn read_times(file: &netcrust::File) -> Result<Vec<String>, String> {
    let raw = file.read_strings("Times").map_err(|e| e.to_string())?;
    let shape = file.variable("Times").map(|v| v.shape()).unwrap_or_default();
    let labels = match shape.as_slice() {
        [records, width] if *width > 1 && raw.len() == records * width => {
            if let Some(bad) = raw.iter().position(|c| c.trim_end_matches('\0').len() > 1) {
                return Err(format!(
                    "Times decoded {} elements for a [{records}, {width}] char matrix, and element {bad} is not one character",
                    raw.len()
                ));
            }
            raw.chunks(*width).map(|c| c.concat()).collect()
        }
        _ => raw,
    };
    Ok(labels.into_iter().map(|s| s.trim_end_matches('\0').trim().to_owned()).collect())
}

/// The path a handle was opened from, so parallel readers can open their own.
fn file_path(file: &netcrust::File) -> Option<std::path::PathBuf> {
    file.path().map(Path::to_path_buf)
}

fn dims(file: &netcrust::File, name: &str) -> Option<Vec<usize>> {
    file.variable(name).map(|v| v.shape())
}

/// Read record `t` of a variable as f32, dropping the leading Time axis.
fn read_record_f32(file: &netcrust::File, name: &str, t: usize) -> Result<Option<Vec<f32>>, HistoryError> {
    let Some(var) = file.variable(name) else { return Ok(None) };
    let shape = var.shape();
    let has_time = var.dimensions().first().is_some_and(|d| d.name() == "Time");
    // Typed reads first (no promotion).  netCDF-4 variables on an
    // unlimited Time axis whose stored dimensions netcrust resolves through
    // HDF5 overrides refuse a typed read; those go through the promoted
    // f64 read, which is exact for f32 and i32 words.  Breakage prevented:
    // WRF's own netCDF-4 histories failed on T ("typed read requires
    // resolved stored dimensions"), merge-and-accept lane, 2026-10-05.
    let promoted = || -> Result<Vec<f32>, HistoryError> {
        let a = file
            .read_array_f64(name)
            .map_err(|e| HistoryError::Variable { name: name.into(), message: e.to_string() })?;
        Ok(a.values().iter().map(|&v| v as f32).collect())
    };
    let values: Vec<f32> = match var.dtype() {
        netcrust::DataType::F32 => match file.read_array::<f32>(name) {
            Ok(a) => a.iter().copied().collect(),
            Err(_) => promoted()?,
        },
        netcrust::DataType::I32 => match file.read_array::<i32>(name) {
            Ok(a) => a.iter().map(|&v| v as f32).collect(),
            Err(_) => promoted()?,
        },
        _ => promoted()?,
    };
    // A declared fill (MPASSIT writes _FillValue = -9999 off the mesh) is
    // missing, never data: a -9999 Pa pressure or K temperature would reach
    // every product as a value.
    let fill = var.attribute("_FillValue").or_else(|| var.attribute("missing_value")).and_then(|a| a.as_f64());
    let values: Vec<f32> = match fill {
        Some(f) if f.is_finite() => values.into_iter().map(|x| if f64::from(x) == f { f32::NAN } else { x }).collect(),
        _ => values,
    };
    if !has_time {
        return Ok(Some(values));
    }
    let records = shape[0].max(1);
    if t >= records {
        return Err(HistoryError::Variable { name: name.into(), message: format!("record {t} of {records}") });
    }
    let per = values.len() / records;
    Ok(Some(values[t * per..(t + 1) * per].to_vec()))
}

fn attr_f64(file: &netcrust::File, name: &str) -> Option<f64> {
    file.attribute(name).and_then(|a| a.as_f64())
}

fn attr_string(file: &netcrust::File, name: &str) -> Option<String> {
    file.attribute(name).and_then(|a| a.as_string().map(str::to_owned))
}

fn units_scale(file: &netcrust::File, name: &str) -> f32 {
    // Percent unless the units attribute says the carrier is a fraction.
    let units = file
        .variable(name)
        .and_then(|v| v.attribute("units").and_then(|a| a.as_string().map(|s| s.trim().to_ascii_lowercase())));
    match units.as_deref() {
        Some("fraction") | Some("1") | Some("0-1") | Some("dimensionless") => 100.0,
        _ => 1.0,
    }
}

impl Frame {
    /// Read record `t` of a history file.
    pub fn read(path: &Path, t: usize) -> Result<Self, HistoryError> {
        let file = netcrust::File::open(path)
            .map_err(|source| HistoryError::Open { path: path.display().to_string(), source })?;
        let tdims = dims(&file, "T").ok_or(HistoryError::Missing("T"))?;
        let (nz, ny, nx) = (tdims[tdims.len() - 3], tdims[tdims.len() - 2], tdims[tdims.len() - 1]);
        let shape = Shape { nx, ny, nz };
        let mut carriers = Vec::new();

        let mut mass = vec![0.0f32; MassInput::COUNT * shape.mass_volume()];
        let mut present = 0u32;
        // Every carrier on its own core, each task with its own handle: a
        // serial read of a 1800 x 1060 x 59 frame held one core for tens of
        // seconds while the card waited (no single-CPU path, 2026-10-16).
        let open = || netcrust::File::open(path).map_err(|source| HistoryError::Open { path: path.display().to_string(), source });
        let read_all: Vec<Result<Option<Vec<f32>>, HistoryError>> = MassInput::ALL
            .par_iter()
            .map(|slot| read_record_f32(&open()?, slot.carrier(), t))
            .collect();
        let mut prefetched: std::collections::BTreeMap<&'static str, Vec<f32>> = ["PH", "PHB", "U", "V"]
            .par_iter()
            .filter_map(|name| match open().and_then(|f| read_record_f32(&f, name, t)) {
                Ok(Some(v)) => Some(Ok((*name, v))),
                Ok(None) => None,
                Err(e) => Some(Err(e)),
            })
            .collect::<Result<_, _>>()?;
        let mut got_slots: Vec<Option<Vec<f32>>> = Vec::with_capacity(MassInput::COUNT);
        for (slot, got) in MassInput::ALL.into_iter().zip(read_all) {
            let got = got?;
            match &got {
                Some(values) if values.len() != shape.mass_volume() => {
                    return Err(HistoryError::Variable { name: slot.carrier().into(), message: "not a mass-level volume".into() });
                }
                Some(_) => {
                    present |= 1 << slot as u32;
                    carriers.push(slot.carrier().to_owned());
                }
                None if slot.required() => return Err(HistoryError::Missing(slot.carrier())),
                None => {}
            }
            got_slots.push(got);
        }
        mass.par_chunks_mut(shape.mass_volume()).zip(got_slots.par_iter()).for_each(|(dst, src)| {
            if let Some(src) = src {
                dst.copy_from_slice(src);
            }
        });
        drop(got_slots);
        let mut need = |name: &'static str, carriers: &mut Vec<String>| -> Result<Vec<f32>, HistoryError> {
            let v = match prefetched.remove(name) {
                Some(v) => v,
                None => read_record_f32(&file, name, t)?.ok_or(HistoryError::Missing(name))?,
            };
            carriers.push(name.to_owned());
            Ok(v)
        };
        let ph = need("PH", &mut carriers)?;
        let phb = need("PHB", &mut carriers)?;
        let u_stag = need("U", &mut carriers)?;
        let v_stag = need("V", &mut carriers)?;
        // A height-coordinate frame (an MPAS history gathered onto a
        // WRF-shaped grid) carries no dry column mass and no eta levels.  Its
        // mass pressure is its own P + PB, carried as P_HYD, and the
        // interfaces are filled from heights after the state is built
        // (CoefficientSource::HeightCoordinate).  Breakage prevented: every
        // MPAS-on-Lambert frame was refused with "missing required carrier MU",
        // so no hex RRFS run could reach a GRIB2 product.
        let height_coordinate = file.variable("ZNW").is_none() && file.variable("C3F").is_none();
        if height_coordinate {
            // MPASSIT writes PHB as zgrid in metres (units "gpm"); the state
            // kernel takes geopotential, so metres are scaled by its g.
            let metres = |name: &str| {
                file.variable(name)
                    .and_then(|v| v.attribute("units").and_then(|a| a.as_string().map(|u| u.trim().to_ascii_lowercase())))
                    .is_some_and(|u| u == "gpm" || u == "m" || u.starts_with("m amsl"))
            };
            let (mut ph, mut phb) = (ph, phb);
            if metres("PHB") {
                phb.par_iter_mut().for_each(|x| *x *= crate::consts::G);
                carriers.push("PHB metres x g".into());
            }
            if metres("PH") {
                ph.par_iter_mut().for_each(|x| *x *= crate::consts::G);
            }
            let ncell = shape.ncell();
            let v = shape.mass_volume();
            if present & (1 << MassInput::PHyd as u32) == 0 {
                let (p0, pb0) = (MassInput::P as usize * v, MassInput::PB as usize * v);
                let ph0 = MassInput::PHyd as usize * v;
                let (head, tail) = mass.split_at_mut(ph0);
                let (p_part, pb_part) = (&head[p0..p0 + v], &head[pb0..pb0 + v]);
                tail[..v].par_iter_mut().zip(p_part.par_iter().zip(pb_part.par_iter())).for_each(|(d, (a, b))| *d = a + b);
                present |= 1 << MassInput::PHyd as u32;
                carriers.push("P_HYD=P+PB".into());
            }
            let state = StateInputs {
                shape,
                mass,
                present,
                ph,
                phb,
                u_stag,
                v_stag,
                mu: vec![0.0; ncell],
                mub: vec![0.0; ncell],
                c3f: vec![0.0; nz + 1],
                c4f: vec![0.0; nz + 1],
                p_top: 0.0,
            };
            return Self::finish(&file, t, shape, state, CoefficientSource::HeightCoordinate, carriers);
        }
        let mu = need("MU", &mut carriers)?;
        let mub = need("MUB", &mut carriers)?;
        let p_top = need("P_TOP", &mut carriers)?[0];
        let znw = need("ZNW", &mut carriers)?;
        let stored = match (read_record_f32(&file, "C3F", t)?, read_record_f32(&file, "C4F", t)?) {
            (Some(a), Some(b)) => {
                carriers.push("C3F".into());
                carriers.push("C4F".into());
                Some((a, b))
            }
            _ => None,
        };
        let hybrid_opt = attr_f64(&file, "HYBRID_OPT").map_or(0, |v| v as i32);
        let coeff = InterfaceCoefficients::resolve(nz, stored, &znw, hybrid_opt, attr_f64(&file, "ETAC"), p_top)?;

        let state = StateInputs {
            shape,
            mass,
            present,
            ph,
            phb,
            u_stag,
            v_stag,
            mu,
            mub,
            c3f: coeff.c3f.clone(),
            c4f: coeff.c4f.clone(),
            p_top,
        };
        Self::finish(&file, t, shape, state, coeff.source, carriers)
    }

    /// The 2D carriers, time and options of a frame whose state is read.
    fn finish(
        file: &netcrust::File,
        t: usize,
        shape: Shape,
        state: StateInputs,
        coefficients: CoefficientSource,
        mut carriers: Vec<String>,
    ) -> Result<Self, HistoryError> {
        let file = file;
        let ncell = shape.ncell();
        let mut surface = SurfaceInputs::new(ncell);
        // The 2D carriers in parallel too, each task with its own handle.
        let path = file_path(file);
        let reads: Vec<Result<Option<Vec<f32>>, HistoryError>> = SurfaceInput::CARRIERS
            .par_iter()
            .map(|(_, name)| match &path {
                Some(p) => netcrust::File::open(p)
                    .map_err(|source| HistoryError::Open { path: p.display().to_string(), source })
                    .and_then(|f| read_record_f32(&f, name, t)),
                None => read_record_f32(file, name, t),
            })
            .collect();
        for ((slot, name), got) in SurfaceInput::CARRIERS.into_iter().zip(reads) {
            if let Some(values) = got? {
                if values.len() == ncell {
                    surface.set(slot, &values);
                    carriers.push(name.to_owned());
                }
            }
        }

        let times = read_times(&file).map_err(|message| HistoryError::Variable { name: "Times".into(), message })?;
        let valid_text = times.get(t).cloned().unwrap_or_default();
        let valid = UtcTime::parse_wrf(&valid_text).ok_or_else(|| HistoryError::Time(valid_text.clone()))?;
        let solar = SolarFrame::at(&valid);

        let land_use_table = attr_string(&file, "MMINLU");
        let sf_surface_physics = attr_f64(&file, "SF_SURFACE_PHYSICS").map(|v| v as i32);
        let mut options = SurfaceOptions::new(&solar);
        options.vegfra_scale = units_scale(&file, "VEGFRA");
        options.shd_scale = units_scale(&file, "SHDMIN");
        // RUC land surface model (SF_SURFACE_PHYSICS = 3): D28.
        options.latent_from_qfx = u32::from(sf_surface_physics == Some(3));
        if !surface.has(SurfaceInput::SnowC) {
            options.snow_cover_derived = 1;
            if let (Some(table), true) = (land_use_table.as_deref(), surface.has(SurfaceInput::IvgTyp)) {
                if let Some(snup) = crate::snow::snup_plane(table, surface.plane(SurfaceInput::IvgTyp)) {
                    surface.set(SurfaceInput::Snup, &snup);
                }
            }
        }

        Ok(Self {
            shape,
            valid,
            valid_text,
            state,
            surface,
            options,
            solar,
            coefficients,
            sinalpha: read_record_f32(&file, "SINALPHA", t)?,
            cosalpha: read_record_f32(&file, "COSALPHA", t)?,
            land_use_table,
            sf_surface_physics,
            carriers,
        })
    }

    /// Put the lowest mass-level virtual temperature into the surface inputs.
    pub fn attach_tv_lowest(&mut self, tv_lowest: &[f32]) {
        self.surface.set(SurfaceInput::TvLowest, tv_lowest);
    }
}
