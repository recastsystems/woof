use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

#[cfg(feature = "netcdf-backend")]
use ndarray::Axis;

use crate::error::{WrfError, WrfResult};
use crate::grid;

// ── Physical constants ──
const G: f64 = 9.80665;
const P0: f64 = 100_000.0; // Pa
const KAPPA: f64 = 0.2857142857; // Rd / Cp

// ═══════════════════════════════════════════════════════════════════════════
// Backend: netcdf crate (default)
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(feature = "netcdf-backend")]
mod backend {
    use super::*;

    pub(super) fn nc_dim_len(file: &netcdf::File, name: &str) -> WrfResult<usize> {
        file.dimension(name)
            .map(|d| d.len())
            .ok_or_else(|| WrfError::DimMismatch(format!("dimension '{name}' not found")))
    }

    pub(super) fn _nc_read_f64(var: &netcdf::Variable) -> WrfResult<Vec<f64>> {
        let arr: ndarray::ArrayD<f64> =
            var.get(..).map_err(|e| WrfError::NetCdf(format!("{e}")))?;
        Ok(arr.iter().copied().collect())
    }

    pub(super) fn nc_read_f64_time(var: &netcdf::Variable, t: usize) -> WrfResult<Vec<f64>> {
        let ndim = var.dimensions().len();
        let arr: ndarray::ArrayD<f64> =
            var.get(..).map_err(|e| WrfError::NetCdf(format!("{e}")))?;
        if ndim >= 2 {
            let slice = arr.index_axis(Axis(0), t);
            Ok(slice.iter().copied().collect())
        } else {
            Ok(arr.iter().copied().collect())
        }
    }

    pub(super) fn nc_get_global_f64(file: &netcdf::File, name: &str) -> WrfResult<f64> {
        let attr = file
            .attribute(name)
            .ok_or_else(|| WrfError::AttrNotFound(name.to_string()))?;
        let val = attr.value().map_err(|e| WrfError::NetCdf(format!("{e}")))?;
        attr_value_to_f64(&val, name)
    }

    pub(super) fn nc_get_global_i32(file: &netcdf::File, name: &str) -> WrfResult<i32> {
        let attr = file
            .attribute(name)
            .ok_or_else(|| WrfError::AttrNotFound(name.to_string()))?;
        let val = attr.value().map_err(|e| WrfError::NetCdf(format!("{e}")))?;
        attr_value_to_i32(&val, name)
    }

    pub(super) fn nc_get_global_str(file: &netcdf::File, name: &str) -> WrfResult<String> {
        let attr = file
            .attribute(name)
            .ok_or_else(|| WrfError::AttrNotFound(name.to_string()))?;
        let val = attr.value().map_err(|e| WrfError::NetCdf(format!("{e}")))?;
        match val {
            netcdf::AttributeValue::Str(s) => Ok(s),
            other => Err(WrfError::AttrType(format!(
                "{name}: expected string, got {other:?}"
            ))),
        }
    }

    fn attr_value_to_f64(val: &netcdf::AttributeValue, name: &str) -> WrfResult<f64> {
        use netcdf::AttributeValue::*;
        match val {
            Double(d) => Ok(*d),
            Float(f) => Ok(*f as f64),
            Int(i) => Ok(*i as f64),
            Short(s) => Ok(*s as f64),
            Ushort(u) => Ok(*u as f64),
            Uint(u) => Ok(*u as f64),
            Uchar(u) => Ok(*u as f64),
            Schar(s) => Ok(*s as f64),
            Longlong(l) => Ok(*l as f64),
            Ulonglong(u) => Ok(*u as f64),
            Doubles(d) if !d.is_empty() => Ok(d[0]),
            Floats(f) if !f.is_empty() => Ok(f[0] as f64),
            Ints(i) if !i.is_empty() => Ok(i[0] as f64),
            _ => Err(WrfError::AttrType(format!("{name}: cannot convert to f64"))),
        }
    }

    fn attr_value_to_i32(val: &netcdf::AttributeValue, name: &str) -> WrfResult<i32> {
        use netcdf::AttributeValue::*;
        match val {
            Int(i) => Ok(*i),
            Short(s) => Ok(*s as i32),
            Uchar(u) => Ok(*u as i32),
            Schar(s) => Ok(*s as i32),
            Longlong(l) => Ok(*l as i32),
            Float(f) => Ok(*f as i32),
            Double(d) => Ok(*d as i32),
            Ints(i) if !i.is_empty() => Ok(i[0]),
            _ => Err(WrfError::AttrType(format!("{name}: cannot convert to i32"))),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// WrfFile -- unified struct
// ═══════════════════════════════════════════════════════════════════════════

/// A handle to a WRF output file with dimension caching and memoized fields.
///
/// When compiled with the `netcdf-backend` feature the file is read through
/// the `netcdf` crate. When compiled with `pure-rust-reader`, classic NetCDF
/// and NetCDF4/HDF5 files are read through Rust-only parsers.
pub struct WrfFile {
    pub path: PathBuf,

    // -- backend handle --
    #[cfg(feature = "netcdf-backend")]
    nc: netcdf::File,
    #[cfg(feature = "pure-rust-reader")]
    hdf5: crate::pure_reader::PureRustFile,
    /// The frame this file takes its static fields from (gpuwm quick-look frames, GPUWM_STATIC_FRAME).
    #[cfg(feature = "pure-rust-reader")]
    static_frame: Option<crate::pure_reader::PureRustFile>,

    /// Unstaggered grid dimensions.
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub nt: usize,
    /// Staggered dimensions.
    pub nx_stag: usize,
    pub ny_stag: usize,
    pub nz_stag: usize,
    /// Grid spacing (m).
    pub dx: f64,
    pub dy: f64,
    /// Memoization cache keyed by `"{field}_{timeidx}"`.
    cache: Mutex<HashMap<String, SharedField>>,
    cache_timeidx: Mutex<Option<usize>>,
}

pub type SharedField = Arc<[f64]>;

// ── netcdf-backend open / read_var / attribute helpers ──

#[cfg(feature = "netcdf-backend")]
impl WrfFile {
    /// Open a WRF output file and read grid dimensions.
    pub fn open<P: AsRef<Path>>(path: P) -> WrfResult<Self> {
        let path = path.as_ref().to_path_buf();
        let nc = netcdf::open(&path)?;

        let nx = backend::nc_dim_len(&nc, "west_east")?;
        let ny = backend::nc_dim_len(&nc, "south_north")?;
        let nz = backend::nc_dim_len(&nc, "bottom_top")?;
        let nt = backend::nc_dim_len(&nc, "Time")?;

        let nx_stag = backend::nc_dim_len(&nc, "west_east_stag").unwrap_or(nx + 1);
        let ny_stag = backend::nc_dim_len(&nc, "south_north_stag").unwrap_or(ny + 1);
        let nz_stag = backend::nc_dim_len(&nc, "bottom_top_stag").unwrap_or(nz + 1);

        let dx = backend::nc_get_global_f64(&nc, "DX").unwrap_or(1000.0);
        let dy = backend::nc_get_global_f64(&nc, "DY").unwrap_or(1000.0);

        Ok(Self {
            path,
            nc,
            nx,
            ny,
            nz,
            nt,
            nx_stag,
            ny_stag,
            nz_stag,
            dx,
            dy,
            cache: Mutex::new(HashMap::new()),
            cache_timeidx: Mutex::new(None),
        })
    }

    /// Read a raw variable for a single time step.
    /// Returns data with the Time dimension removed.
    pub fn read_var(&self, name: &str, t: usize) -> WrfResult<Vec<f64>> {
        let var = self
            .nc
            .variable(name)
            .ok_or_else(|| WrfError::VarNotFound(name.to_string()))?;
        backend::nc_read_f64_time(&var, t)
    }

    /// Check if a variable exists in the file.
    pub fn has_var(&self, name: &str) -> bool {
        self.nc.variable(name).is_some()
    }

    /// Read a global attribute as f64.
    pub fn global_attr_f64(&self, name: &str) -> WrfResult<f64> {
        backend::nc_get_global_f64(&self.nc, name)
    }

    /// Read a global attribute as i32.
    pub fn global_attr_i32(&self, name: &str) -> WrfResult<i32> {
        backend::nc_get_global_i32(&self.nc, name)
    }

    /// Read a global attribute as String.
    pub fn global_attr_str(&self, name: &str) -> WrfResult<String> {
        backend::nc_get_global_str(&self.nc, name)
    }

    /// Read WRF Times variable as strings.
    pub fn times(&self) -> WrfResult<Vec<String>> {
        let var = self
            .nc
            .variable("Times")
            .ok_or_else(|| WrfError::VarNotFound("Times".to_string()))?;
        let arr: ndarray::ArrayD<u8> = var.get(..).map_err(|e| WrfError::NetCdf(format!("{e}")))?;
        let shape = arr.shape();
        if shape.len() == 2 {
            let nt = shape[0];
            let slen = shape[1];
            let flat: Vec<u8> = arr.iter().copied().collect();
            let mut times = Vec::with_capacity(nt);
            for i in 0..nt {
                let start = i * slen;
                let end = start + slen;
                let s = String::from_utf8_lossy(&flat[start..end])
                    .trim_end_matches('\0')
                    .to_string();
                times.push(s);
            }
            Ok(times)
        } else {
            Err(WrfError::DimMismatch(
                "Times variable has unexpected shape".into(),
            ))
        }
    }

    /// Expose the netcdf file reference for advanced use.
    pub fn nc(&self) -> &netcdf::File {
        &self.nc
    }
}

// ── pure-rust-reader open / read_var / attribute helpers ──

#[cfg(feature = "pure-rust-reader")]
impl WrfFile {
    /// Open a WRF output file and read grid dimensions using Rust-only readers.
    pub fn open<P: AsRef<Path>>(path: P) -> WrfResult<Self> {
        let path = path.as_ref().to_path_buf();
        let hdf5 = crate::pure_reader::PureRustFile::open(&path)?;
        let static_frame = Self::open_static_frame(&path, &hdf5);

        // Derive dimensions from the shape of known variables.
        // netCDF4/WRF stores dimension information as attributes of _nc4_non_coord variables
        // or implicitly via dataset shapes.  The most reliable approach is to
        // probe a variable we know exists (e.g. "T" has shape [Time, bottom_top, south_north, west_east]).
        // A gpuwm quick-look frame may carry no 3-D T (it writes only the fields its maps read): the
        // dimensions then come from the coordinate fields every wrfout keeps (Times, ZNU, XLAT).
        let (nx, ny, nz, nt) = match Self::probe_dims_pure(&hdf5) {
            Ok(dims) => dims,
            Err(error) => Self::probe_dims_coordinates(&hdf5).ok_or(error)?,
        };

        // Staggered dims: probe U for nx_stag, V for ny_stag, PH for nz_stag.
        let nx_stag = hdf5
            .dataset_shape("U")
            .map(|s| if s.len() >= 4 { s[3] } else { nx + 1 })
            .unwrap_or(nx + 1);
        let ny_stag = hdf5
            .dataset_shape("V")
            .map(|s| if s.len() >= 4 { s[2] } else { ny + 1 })
            .unwrap_or(ny + 1);
        let nz_stag = hdf5
            .dataset_shape("PH")
            .map(|s| if s.len() >= 4 { s[1] } else { nz + 1 })
            .unwrap_or(nz + 1);

        let dx = hdf5.global_attr_f64("DX").unwrap_or(1000.0);
        let dy = hdf5.global_attr_f64("DY").unwrap_or(1000.0);

        Ok(Self {
            path,
            hdf5,
            static_frame,
            nx,
            ny,
            nz,
            nt,
            nx_stag,
            ny_stag,
            nz_stag,
            dx,
            dy,
            cache: Mutex::new(HashMap::new()),
            cache_timeidx: Mutex::new(None),
        })
    }

    /// The frame named by the global attribute GPUWM_STATIC_FRAME (a path relative to this file's folder),
    /// when it opens.  gpuwm writes a quick-look frame without the fields that never change over a fixed
    /// domain (PHB) and names the domain's first full frame here; reads of those fields go there.
    fn open_static_frame(
        path: &Path,
        hdf5: &crate::pure_reader::PureRustFile,
    ) -> Option<crate::pure_reader::PureRustFile> {
        let name = hdf5.global_attr_string("GPUWM_STATIC_FRAME").ok()?;
        let name = name.trim_end_matches(char::from(0)).trim();
        if name.is_empty() {
            return None;
        }
        let target = path.parent().unwrap_or_else(|| Path::new(".")).join(name);
        crate::pure_reader::PureRustFile::open(target).ok()
    }

    /// Grid dimensions from the coordinate fields: nz from ZNU [Time, bottom_top], ny/nx from XLAT
    /// [Time, south_north, west_east], nt from Times [Time, DateStrLen].  For a frame without T.
    fn probe_dims_coordinates(
        hdf5: &crate::pure_reader::PureRustFile,
    ) -> Option<(usize, usize, usize, usize)> {
        let znu = hdf5.dataset_shape("ZNU").ok()?;
        let xlat = hdf5.dataset_shape("XLAT").ok()?;
        let times = hdf5.dataset_shape("Times").ok()?;
        if znu.len() < 2 || xlat.len() < 3 || times.is_empty() {
            return None;
        }
        Some((xlat[xlat.len() - 1], xlat[xlat.len() - 2], znu[znu.len() - 1], times[0]))
    }

    /// The fields a quick-look frame may leave to its static frame: they do not change over a fixed domain.
    const STATIC_FRAME_FIELDS: &'static [&'static str] = &["PHB", "PB", "MUB", "HGT"];

    /// Probe grid dimensions from the "T" variable shape
    /// which is [Time, bottom_top, south_north, west_east].
    fn probe_dims_pure(
        hdf5: &crate::pure_reader::PureRustFile,
    ) -> WrfResult<(usize, usize, usize, usize)> {
        for name in ["T", "P", "PB", "QVAPOR"] {
            if let Ok(shape) = hdf5.dataset_shape(name) {
                if shape.len() == 4 {
                    return Ok((shape[3], shape[2], shape[1], shape[0]));
                }
            }
        }
        // A selected surface-only history can omit every mass-level
        // volume. Its latitude plane still defines the horizontal grid;
        // no vertical data is synthesized by opening that file.
        if let Ok(shape) = hdf5.dataset_shape("XLAT") {
            if shape.len() == 3 {
                let nz = hdf5.global_attr_i32("BOTTOM-TOP_GRID_DIMENSION")
                    .ok().filter(|&v| v > 1).map(|v| v as usize - 1).unwrap_or(1);
                return Ok((shape[2], shape[1], nz, shape[0]));
            }
        }
        Err(WrfError::DimMismatch("Cannot determine grid dimensions: no 4-D mass field or 3-D XLAT plane".to_string()))
    }

    /// Read a raw variable for a single time step.
    /// Returns data with the Time dimension removed.
    /// Only reads the bytes for the requested timestep (not the whole variable).
    pub fn read_var(&self, name: &str, t: usize) -> WrfResult<Vec<f64>> {
        if let Some(frame) = self.static_frame_for(name) {
            return frame.read_f64_slice(name, 0);
        }
        self.hdf5.read_f64_slice(name, t)
    }

    /// Check if a variable exists in the file (or, for a static field, in its static frame).
    pub fn has_var(&self, name: &str) -> bool {
        self.hdf5.has_dataset(name) || self.static_frame_for(name).is_some()
    }

    /// The static frame, when `name` is a static field this file does not carry and that frame does.
    fn static_frame_for(&self, name: &str) -> Option<&crate::pure_reader::PureRustFile> {
        if self.hdf5.has_dataset(name) || !Self::STATIC_FRAME_FIELDS.contains(&name) {
            return None;
        }
        self.static_frame.as_ref().filter(|frame| frame.has_dataset(name))
    }

    /// Read a global attribute as f64.
    pub fn global_attr_f64(&self, name: &str) -> WrfResult<f64> {
        self.hdf5.global_attr_f64(name)
    }

    /// Read a global attribute as i32.
    pub fn global_attr_i32(&self, name: &str) -> WrfResult<i32> {
        self.hdf5.global_attr_i32(name)
    }

    /// Read a global attribute as String.
    pub fn global_attr_str(&self, name: &str) -> WrfResult<String> {
        self.hdf5.global_attr_string(name)
    }

    /// Read WRF Times variable as strings.
    pub fn times(&self) -> WrfResult<Vec<String>> {
        let shape = self.hdf5.dataset_shape("Times")?;
        let raw = self.hdf5.read_u8("Times")?;

        if shape.len() == 2 {
            let nt = shape[0];
            let slen = shape[1];
            let mut times = Vec::with_capacity(nt);
            for i in 0..nt {
                let start = i * slen;
                let end = start + slen;
                if end > raw.len() {
                    break;
                }
                let s = String::from_utf8_lossy(&raw[start..end])
                    .trim_end_matches('\0')
                    .to_string();
                times.push(s);
            }
            Ok(times)
        } else {
            Err(WrfError::DimMismatch(
                "Times variable has unexpected shape".into(),
            ))
        }
    }

    /// Expose the pure-Rust file handle for advanced use.
    pub fn hdf5(&self) -> &crate::pure_reader::PureRustFile {
        &self.hdf5
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Shared methods -- identical regardless of backend
// ═══════════════════════════════════════════════════════════════════════════

impl WrfFile {
    /// Number of grid cells in a 2D plane.
    pub fn nxy(&self) -> usize {
        self.nx * self.ny
    }

    /// Number of grid cells in a 3D volume.
    pub fn nxyz(&self) -> usize {
        self.nx * self.ny * self.nz
    }

    /// Keep cached intermediates scoped to a single timestep so related
    /// diagnostics can reuse them without letting memory grow unbounded.
    pub fn prepare_cache_for_time(&self, t: usize) {
        let mut cached_timeidx = self.cache_timeidx.lock().unwrap();
        if *cached_timeidx != Some(t) {
            self.cache.lock().unwrap().clear();
            *cached_timeidx = Some(t);
        }
    }

    /// Clear the intermediate-result cache, freeing memory.
    ///
    /// Called automatically after each `getvar` call so that 3-D
    /// intermediates (full_pressure, temperature, etc.) do not persist
    /// beyond the computation that needed them.
    pub fn clear_cache(&self) {
        self.cache.lock().unwrap().clear();
        *self.cache_timeidx.lock().unwrap() = None;
    }

    // ── Cached derived fields ──

    fn cached_or_compute(
        &self,
        key: &str,
        f: impl FnOnce() -> WrfResult<Vec<f64>>,
    ) -> WrfResult<SharedField> {
        {
            let cache = self.cache.lock().unwrap();
            if let Some(v) = cache.get(key) {
                return Ok(Arc::clone(v));
            }
        }
        let result = Arc::<[f64]>::from(f()?);
        self.cache
            .lock()
            .unwrap()
            .insert(key.to_string(), Arc::clone(&result));
        Ok(result)
    }

    /// Return a cached field by key, if present.
    pub fn cached_field(&self, key: &str) -> Option<SharedField> {
        self.cache.lock().unwrap().get(key).map(Arc::clone)
    }

    /// Insert a precomputed field into the memoization cache.
    pub fn store_cached_field(&self, key: String, data: Vec<f64>) -> SharedField {
        let result = Arc::<[f64]>::from(data);
        let mut cache = self.cache.lock().unwrap();
        if let Some(existing) = cache.get(&key) {
            return Arc::clone(existing);
        }
        cache.insert(key, Arc::clone(&result));
        result
    }

    /// Full pressure = P + PB (Pa). Shape: `[nz, ny, nx]`.
    pub fn full_pressure(&self, t: usize) -> WrfResult<SharedField> {
        let key = format!("pressure_{t}");
        self.cached_or_compute(&key, || {
            let p = self.read_var("P", t)?;
            let pb = self.read_var("PB", t)?;
            Ok(p.iter().zip(pb.iter()).map(|(a, b)| a + b).collect())
        })
    }

    /// Full potential temperature = T + 300 (K). Shape: `[nz, ny, nx]`.
    pub fn full_theta(&self, t: usize) -> WrfResult<SharedField> {
        let key = format!("theta_{t}");
        self.cached_or_compute(&key, || {
            let th = self.read_var("T", t)?;
            Ok(th.iter().map(|v| v + 300.0).collect())
        })
    }

    /// Full geopotential = PH + PHB (m^2/s^2), destaggered in Z.
    /// Shape: `[nz, ny, nx]`.
    pub fn full_geopotential(&self, t: usize) -> WrfResult<SharedField> {
        let key = format!("geopt_{t}");
        self.cached_or_compute(&key, || {
            let ph = self.read_var("PH", t)?;
            let phb = self.read_var("PHB", t)?;
            let stag: Vec<f64> = ph.iter().zip(phb.iter()).map(|(a, b)| a + b).collect();
            Ok(grid::destagger_z(&stag, self.nz_stag, self.ny, self.nx))
        })
    }

    /// Geopotential on staggered Z levels (not destaggered).
    /// Shape: `[nz_stag, ny, nx]`.
    pub fn geopotential_stag(&self, t: usize) -> WrfResult<SharedField> {
        let key = format!("geopt_stag_{t}");
        self.cached_or_compute(&key, || {
            let ph = self.read_var("PH", t)?;
            let phb = self.read_var("PHB", t)?;
            Ok(ph.iter().zip(phb.iter()).map(|(a, b)| a + b).collect())
        })
    }

    /// Temperature = theta * (p / p0)^kappa (K). Shape: `[nz, ny, nx]`.
    pub fn temperature(&self, t: usize) -> WrfResult<SharedField> {
        let key = format!("temp_{t}");
        self.cached_or_compute(&key, || {
            let theta = self.full_theta(t)?;
            let pres = self.full_pressure(t)?;
            Ok(theta
                .iter()
                .zip(pres.iter())
                .map(|(th, p)| th * (p / P0).powf(KAPPA))
                .collect())
        })
    }

    /// Height MSL = geopotential / g (m). Shape: `[nz, ny, nx]`.
    pub fn height_msl(&self, t: usize) -> WrfResult<SharedField> {
        let key = format!("height_msl_{t}");
        self.cached_or_compute(&key, || {
            let geopt = self.full_geopotential(t)?;
            Ok(geopt.iter().map(|g| g / G).collect())
        })
    }

    /// Terrain height (m). Shape: `[ny, nx]`.
    pub fn terrain(&self, t: usize) -> WrfResult<SharedField> {
        let key = format!("terrain_{t}");
        self.cached_or_compute(&key, || self.read_var("HGT", t))
    }

    /// Height AGL = height_msl - terrain (m). Shape: `[nz, ny, nx]`.
    pub fn height_agl(&self, t: usize) -> WrfResult<SharedField> {
        let key = format!("height_agl_{t}");
        self.cached_or_compute(&key, || {
            let h = self.height_msl(t)?;
            let ter = self.terrain(t)?;
            let nxy = self.nxy();
            Ok(h.iter()
                .enumerate()
                .map(|(idx, hv)| hv - ter[idx % nxy])
                .collect())
        })
    }

    /// Water vapour mixing ratio (kg/kg). Shape: `[nz, ny, nx]`.
    pub fn qvapor(&self, t: usize) -> WrfResult<SharedField> {
        let key = format!("qvapor_{t}");
        self.cached_or_compute(&key, || self.read_var("QVAPOR", t))
    }

    /// Surface pressure (Pa). Shape: `[ny, nx]`.
    pub fn psfc(&self, t: usize) -> WrfResult<SharedField> {
        let key = format!("psfc_{t}");
        self.cached_or_compute(&key, || self.read_var("PSFC", t))
    }

    /// 2-m temperature (K). Shape: `[ny, nx]`.
    /// Use `t2_for_opts` when lake_interp may be active.
    pub fn t2(&self, t: usize) -> WrfResult<SharedField> {
        let key = format!("t2_{t}");
        self.cached_or_compute(&key, || self.read_var("T2", t))
    }

    /// 2-m temperature, with optional lake correction from ComputeOpts.
    pub fn t2_for_opts(&self, t: usize, opts: &crate::compute::ComputeOpts) -> WrfResult<Vec<f64>> {
        match opts.lake_interp {
            Some(area) if area > 0.0 => self.t2_lake_corrected(t, area),
            _ => Ok(self.t2(t)?.to_vec()),
        }
    }

    /// 2-m temperature with lake interpolation. Shape: `[ny, nx]`.
    /// If `area_threshold_km2 > 0`, water bodies smaller than that area
    /// are masked and their T2 values are interpolated from surrounding land.
    pub fn t2_lake_corrected(&self, t: usize, area_threshold_km2: f64) -> WrfResult<Vec<f64>> {
        let data = self.read_var("T2", t)?;
        if area_threshold_km2 <= 0.0 {
            return Ok(data);
        }
        let mask = self.lake_mask(t, area_threshold_km2)?;
        Ok(interpolate_masked_2d(&data, &mask, self.ny, self.nx))
    }

    /// 2-m mixing ratio (kg/kg). Shape: `[ny, nx]`.
    /// Use `q2_for_opts` when lake_interp may be active.
    pub fn q2(&self, t: usize) -> WrfResult<SharedField> {
        let key = format!("q2_{t}");
        self.cached_or_compute(&key, || self.read_var("Q2", t))
    }

    /// 2-m mixing ratio, with optional lake correction from ComputeOpts.
    pub fn q2_for_opts(&self, t: usize, opts: &crate::compute::ComputeOpts) -> WrfResult<Vec<f64>> {
        match opts.lake_interp {
            Some(area) if area > 0.0 => self.q2_lake_corrected(t, area),
            _ => Ok(self.q2(t)?.to_vec()),
        }
    }

    /// 2-m mixing ratio with lake interpolation. Shape: `[ny, nx]`.
    pub fn q2_lake_corrected(&self, t: usize, area_threshold_km2: f64) -> WrfResult<Vec<f64>> {
        let data = self.read_var("Q2", t)?;
        if area_threshold_km2 <= 0.0 {
            return Ok(data);
        }
        let mask = self.lake_mask(t, area_threshold_km2)?;
        Ok(interpolate_masked_2d(&data, &mask, self.ny, self.nx))
    }

    /// Compute a lake mask: true for water body grid cells smaller than `area_km2`.
    /// Uses WRF LU_INDEX: categories 16 (water), 17 (ocean/bay), 21 (lake).
    /// Connected components smaller than the area threshold are marked as lakes.
    fn lake_mask(&self, t: usize, area_km2: f64) -> WrfResult<Vec<bool>> {
        let key = format!("lake_mask_{t}_{area_km2}");
        let cached = {
            let cache = self.cache.lock().unwrap();
            cache.get(&key).map(Arc::clone)
        };
        if let Some(mask_f64) = cached {
            return Ok(mask_f64.iter().map(|v| *v > 0.5).collect());
        }

        let lu = self.read_var("LU_INDEX", t)?;
        let nxy = self.nxy();
        let ny = self.ny;
        let nx = self.nx;

        // Water categories in USGS and MODIS land use
        let is_water: Vec<bool> = lu
            .iter()
            .map(|v| {
                let cat = *v as i32;
                cat == 16 || cat == 17 || cat == 21
            })
            .collect();

        // Connected component labeling (flood fill)
        let mut labels = vec![0u32; nxy];
        let mut current_label = 0u32;
        let mut label_sizes: Vec<usize> = vec![0]; // label 0 = no label

        for start in 0..nxy {
            if !is_water[start] || labels[start] != 0 {
                continue;
            }
            current_label += 1;
            let mut size = 0usize;
            let mut stack = vec![start];
            while let Some(idx) = stack.pop() {
                if labels[idx] != 0 {
                    continue;
                }
                labels[idx] = current_label;
                size += 1;

                let j = idx / nx;
                let i = idx % nx;
                // 8-connected neighbors
                for dj in [-1i32, 0, 1] {
                    for di in [-1i32, 0, 1] {
                        if dj == 0 && di == 0 {
                            continue;
                        }
                        let nj = j as i32 + dj;
                        let ni = i as i32 + di;
                        if nj >= 0 && nj < ny as i32 && ni >= 0 && ni < nx as i32 {
                            let nidx = nj as usize * nx + ni as usize;
                            if is_water[nidx] && labels[nidx] == 0 {
                                stack.push(nidx);
                            }
                        }
                    }
                }
            }
            label_sizes.push(size);
        }

        // Determine which labels are "small" (lakes, not oceans)
        let grid_area_km2 = (self.dx * self.dy) / 1e6;
        let count_threshold = (area_km2 / grid_area_km2) as usize;

        let is_lake: Vec<bool> = labels
            .iter()
            .map(|&lbl| {
                if lbl == 0 {
                    return false;
                }
                label_sizes[lbl as usize] < count_threshold
            })
            .collect();

        // Cache as f64
        let mask_f64: SharedField = Arc::<[f64]>::from(
            is_lake
                .iter()
                .map(|&b| if b { 1.0 } else { 0.0 })
                .collect::<Vec<_>>(),
        );
        self.cache
            .lock()
            .unwrap()
            .insert(key, Arc::clone(&mask_f64));

        Ok(is_lake)
    }

    /// 10-m U wind (m/s). Shape: `[ny, nx]`.
    pub fn u10(&self, t: usize) -> WrfResult<SharedField> {
        let key = format!("u10_{t}");
        self.cached_or_compute(&key, || self.read_var("U10", t))
    }

    /// 10-m V wind (m/s). Shape: `[ny, nx]`.
    pub fn v10(&self, t: usize) -> WrfResult<SharedField> {
        let key = format!("v10_{t}");
        self.cached_or_compute(&key, || self.read_var("V10", t))
    }

    /// Destaggered U wind (m/s). Shape: `[nz, ny, nx]`.
    pub fn u_destag(&self, t: usize) -> WrfResult<SharedField> {
        let key = format!("u_destag_{t}");
        self.cached_or_compute(&key, || {
            let u_stag = self.read_var("U", t)?;
            Ok(grid::destagger_x(&u_stag, self.nz, self.ny, self.nx_stag))
        })
    }

    /// Destaggered V wind (m/s). Shape: `[nz, ny, nx]`.
    pub fn v_destag(&self, t: usize) -> WrfResult<SharedField> {
        let key = format!("v_destag_{t}");
        self.cached_or_compute(&key, || {
            let v_stag = self.read_var("V", t)?;
            Ok(grid::destagger_y(&v_stag, self.nz, self.ny_stag, self.nx))
        })
    }

    /// Destaggered W wind (m/s). Shape: `[nz, ny, nx]`.
    pub fn w_destag(&self, t: usize) -> WrfResult<SharedField> {
        let key = format!("w_destag_{t}");
        self.cached_or_compute(&key, || {
            let w_stag = self.read_var("W", t)?;
            Ok(grid::destagger_z(&w_stag, self.nz_stag, self.ny, self.nx))
        })
    }

    /// Latitude (degrees). Shape: `[ny, nx]`.
    pub fn xlat(&self, t: usize) -> WrfResult<SharedField> {
        let key = format!("xlat_{t}");
        self.cached_or_compute(&key, || self.read_var("XLAT", t))
    }

    /// Longitude (degrees). Shape: `[ny, nx]`.
    pub fn xlong(&self, t: usize) -> WrfResult<SharedField> {
        let key = format!("xlong_{t}");
        self.cached_or_compute(&key, || self.read_var("XLONG", t))
    }

    /// Sin of map-rotation angle. Shape: `[ny, nx]`.
    pub fn sinalpha(&self, t: usize) -> WrfResult<SharedField> {
        let key = format!("sinalpha_{t}");
        self.cached_or_compute(&key, || self.read_var("SINALPHA", t))
    }

    /// Cos of map-rotation angle. Shape: `[ny, nx]`.
    pub fn cosalpha(&self, t: usize) -> WrfResult<SharedField> {
        let key = format!("cosalpha_{t}");
        self.cached_or_compute(&key, || self.read_var("COSALPHA", t))
    }

    /// Pressure in hPa. Shape: `[nz, ny, nx]`.
    pub fn pressure_hpa(&self, t: usize) -> WrfResult<SharedField> {
        let key = format!("pressure_hpa_{t}");
        self.cached_or_compute(&key, || {
            let p = self.full_pressure(t)?;
            Ok(p.iter().map(|v| v / 100.0).collect())
        })
    }

    /// Temperature in Celsius. Shape: `[nz, ny, nx]`.
    pub fn temperature_c(&self, t: usize) -> WrfResult<SharedField> {
        let key = format!("temp_c_{t}");
        self.cached_or_compute(&key, || {
            let tk = self.temperature(t)?;
            Ok(tk.iter().map(|v| v - 273.15).collect())
        })
    }

    /// Return the raw dataset shape with the leading Time dimension removed.
    pub fn var_shape_no_time(&self, name: &str) -> WrfResult<Vec<usize>> {
        #[cfg(feature = "netcdf-backend")]
        {
            let var = self
                .nc
                .variable(name)
                .ok_or_else(|| WrfError::VarNotFound(name.to_string()))?;
            let dims: Vec<usize> = var.dimensions().iter().map(|d| d.len()).collect();
            Ok(trim_time_dim(dims, self.nt))
        }

        #[cfg(feature = "pure-rust-reader")]
        {
            let dims = self.hdf5.dataset_shape(name)?;
            Ok(trim_time_dim(dims, self.nt))
        }
    }
}

fn trim_time_dim(mut dims: Vec<usize>, nt: usize) -> Vec<usize> {
    if dims.len() > 1 && dims[0] == nt {
        dims.remove(0);
    }
    dims
}

/// Interpolate masked grid cells from surrounding unmasked values.
/// Uses inverse-distance weighting with a search radius that expands until
/// neighbors are found.
fn interpolate_masked_2d(data: &[f64], mask: &[bool], ny: usize, nx: usize) -> Vec<f64> {
    let mut result = data.to_vec();
    let masked_indices: Vec<usize> = mask
        .iter()
        .enumerate()
        .filter(|(_, &m)| m)
        .map(|(i, _)| i)
        .collect();

    if masked_indices.is_empty() {
        return result;
    }

    let mut nearest_land_radius = vec![i32::MAX; mask.len()];
    let mut queue = VecDeque::new();

    for idx in 0..mask.len() {
        if !mask[idx] {
            nearest_land_radius[idx] = 0;
            queue.push_back(idx);
        }
    }

    if queue.is_empty() {
        return result;
    }

    while let Some(idx) = queue.pop_front() {
        let j = idx / nx;
        let i = idx % nx;
        let next_radius = nearest_land_radius[idx] + 1;

        for dj in -1i32..=1 {
            for di in -1i32..=1 {
                if dj == 0 && di == 0 {
                    continue;
                }
                let nj = j as i32 + dj;
                let ni = i as i32 + di;
                if nj < 0 || nj >= ny as i32 || ni < 0 || ni >= nx as i32 {
                    continue;
                }
                let nidx = nj as usize * nx + ni as usize;
                if next_radius < nearest_land_radius[nidx] {
                    nearest_land_radius[nidx] = next_radius;
                    queue.push_back(nidx);
                }
            }
        }
    }

    let interpolated: Vec<(usize, f64)> = masked_indices
        .iter()
        .map(|&idx| {
            let cj = (idx / nx) as i32;
            let ci = (idx % nx) as i32;

            let radius = nearest_land_radius[idx];
            if radius == i32::MAX || radius <= 0 {
                return (idx, data[idx]);
            }

            let mut sum_val = 0.0f64;
            let mut sum_wt = 0.0f64;

            for dj in -radius..=radius {
                for di in -radius..=radius {
                    if dj.abs() != radius && di.abs() != radius {
                        continue;
                    }
                    let nj = cj + dj;
                    let ni = ci + di;
                    if nj < 0 || nj >= ny as i32 || ni < 0 || ni >= nx as i32 {
                        continue;
                    }
                    let nidx = nj as usize * nx + ni as usize;
                    if !mask[nidx] {
                        let dist = ((dj * dj + di * di) as f64).sqrt();
                        let w = 1.0 / dist;
                        sum_val += data[nidx] * w;
                        sum_wt += w;
                    }
                }
            }

            if sum_wt > 0.0 {
                (idx, sum_val / sum_wt)
            } else {
                (idx, data[idx])
            }
        })
        .collect();

    for (idx, val) in interpolated {
        result[idx] = val;
    }
    result
}
