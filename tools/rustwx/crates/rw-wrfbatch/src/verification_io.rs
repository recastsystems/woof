//! Native readers for observation verification. Input formats and field names
//! arrive as metadata; forecast labels never select a reader or calculation.

use std::collections::{BTreeMap, BTreeSet};
use std::io::Read;
use std::path::{Path, PathBuf};

use rustwx_core::{CanonicalField, FieldSelector, GridProjection};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ArmSpec {
    pub label: String,
    pub kind: String,
    pub path: PathBuf,
    /// Record within a native NetCDF history file. Other formats use their
    /// existing selectors. Zero preserves single-frame requests.
    #[serde(default)]
    pub time_index: usize,
    #[serde(default)]
    pub grid_path: Option<PathBuf>,
    #[serde(default)]
    pub previous_path: Option<PathBuf>,
    #[serde(default)]
    pub points_path: Option<PathBuf>,
    #[serde(default)]
    pub fields: BTreeMap<String, String>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub run: Option<String>,
    #[serde(default)]
    pub hour: Option<u16>,
    #[serde(default)]
    pub source_identity: Option<Value>,
    #[serde(default)]
    pub valid_time: Option<String>,
    #[serde(default)]
    pub init_time: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RadarSpec {
    pub quantity: String,
    pub path: PathBuf,
    pub grid_path: PathBuf,
}

#[derive(Clone, Debug)]
pub struct GridData {
    pub lat: Vec<f32>,
    pub lon: Vec<f32>,
    pub ny: usize,
    pub nx: usize,
    pub projection: Option<GridProjection>,
}

impl GridData {
    pub fn spacing_km(&self) -> Option<(f64, f64)> {
        if self.nx < 2 || self.ny < 2 {
            return None;
        }
        let y = (self.ny / 2).min(self.ny - 2);
        let x = (self.nx / 2).min(self.nx - 2);
        let first = y * self.nx + x;
        let distance = |a: usize, b: usize| {
            let lat = (f64::from(self.lat[a]) + f64::from(self.lat[b])) * 0.5;
            let dy = f64::from(self.lat[b]) - f64::from(self.lat[a]);
            let dx =
                (f64::from(self.lon[b]) - f64::from(self.lon[a]) + 180.0).rem_euclid(360.0) - 180.0;
            111.195 * dy.hypot(dx * lat.to_radians().cos())
        };
        let pair = (distance(first, first + 1), distance(first, first + self.nx));
        (pair.0.is_finite() && pair.1.is_finite() && pair.0 > 0.0 && pair.1 > 0.0).then_some(pair)
    }
    pub fn bbox(&self) -> [f64; 4] {
        let (mut west, mut south, mut east, mut north) =
            (180.0_f64, 90.0_f64, -180.0_f64, -90.0_f64);
        for (&lat, &lon) in self.lat.iter().zip(&self.lon) {
            if lat.is_finite() && lon.is_finite() {
                west = west.min(f64::from(lon));
                east = east.max(f64::from(lon));
                south = south.min(f64::from(lat));
                north = north.max(f64::from(lat));
            }
        }
        [west, south, east, north]
    }

    /// Local bilinear inverse of the native coordinate lattice. No geographic
    /// projection is reimplemented: the stored latitude/longitude cells define
    /// the inverse, and coordinates outside the lattice are not extrapolated.
    pub fn position(&self, latitude: f64, longitude: f64) -> Option<(f64, f64)> {
        if self.nx < 2 || self.ny < 2 {
            return None;
        }
        let cos = latitude.to_radians().cos();
        let delta_lon = |lon: f64| (lon - longitude + 180.0).rem_euclid(360.0) - 180.0;
        let distance = |i: usize| {
            (f64::from(self.lat[i]) - latitude).powi(2)
                + (delta_lon(f64::from(self.lon[i])) * cos).powi(2)
        };
        // Use the first cell's two native axes for an indexed seed, then walk
        // to the closest local point. This is O(grid diameter), rather than an
        // exhaustive continental-grid scan for every station.
        let a = (
            delta_lon(f64::from(self.lon[0])) * cos,
            f64::from(self.lat[0]) - latitude,
        );
        let xaxis = (
            (f64::from(self.lon[1]) - f64::from(self.lon[0])) * cos,
            f64::from(self.lat[1]) - f64::from(self.lat[0]),
        );
        let yaxis = (
            (f64::from(self.lon[self.nx]) - f64::from(self.lon[0])) * cos,
            f64::from(self.lat[self.nx]) - f64::from(self.lat[0]),
        );
        let det = xaxis.0 * yaxis.1 - yaxis.0 * xaxis.1;
        let (fx, fy) = if det.abs() > 1e-15 {
            (
                (-a.0 * yaxis.1 + yaxis.0 * a.1) / det,
                (-xaxis.0 * a.1 + a.0 * xaxis.1) / det,
            )
        } else {
            (0.0, 0.0)
        };
        let mut nearest = (fy.round().clamp(0.0, (self.ny - 1) as f64) as usize) * self.nx
            + (fx.round().clamp(0.0, (self.nx - 1) as f64) as usize);
        for _ in 0..self.nx + self.ny {
            let (y, x) = (nearest / self.nx, nearest % self.nx);
            let mut best = nearest;
            for j in y.saturating_sub(1)..=(y + 1).min(self.ny - 1) {
                for i in x.saturating_sub(1)..=(x + 1).min(self.nx - 1) {
                    let at = j * self.nx + i;
                    if distance(at) < distance(best) {
                        best = at;
                    }
                }
            }
            if best == nearest {
                break;
            }
            nearest = best;
        }
        if !distance(nearest).is_finite() {
            return None;
        }
        self.position_from_nearest(latitude, longitude, nearest)
    }

    /// Invert only the cells incident to an already matched native point.
    /// This provides a constant-work containment check during dense regridding,
    /// without a new nearest-point search for every target cell.
    pub fn position_from_nearest(
        &self,
        latitude: f64,
        longitude: f64,
        nearest: usize,
    ) -> Option<(f64, f64)> {
        if self.nx < 2
            || self.ny < 2
            || nearest >= self.lat.len()
            || nearest >= self.lon.len()
            || !latitude.is_finite()
            || !longitude.is_finite()
        {
            return None;
        }
        let cos = latitude.to_radians().cos();
        let delta_lon = |lon: f64| (lon - longitude + 180.0).rem_euclid(360.0) - 180.0;
        let row = nearest / self.nx;
        let col = nearest % self.nx;
        for y in row.saturating_sub(1)..=row.min(self.ny - 2) {
            for x in col.saturating_sub(1)..=col.min(self.nx - 2) {
                let indices = [
                    y * self.nx + x,
                    y * self.nx + x + 1,
                    (y + 1) * self.nx + x,
                    (y + 1) * self.nx + x + 1,
                ];
                let a = indices.map(|i| delta_lon(f64::from(self.lon[i])) * cos);
                let b = indices.map(|i| f64::from(self.lat[i]) - latitude);
                let mut tx = 0.5;
                let mut ty = 0.5;
                let mut converged = false;
                for _ in 0..8 {
                    let value = |v: [f64; 4]| {
                        v[0] * (1.0 - tx) * (1.0 - ty)
                            + v[1] * tx * (1.0 - ty)
                            + v[2] * (1.0 - tx) * ty
                            + v[3] * tx * ty
                    };
                    let deriv = |v: [f64; 4]| {
                        (
                            (v[1] - v[0]) * (1.0 - ty) + (v[3] - v[2]) * ty,
                            (v[2] - v[0]) * (1.0 - tx) + (v[3] - v[1]) * tx,
                        )
                    };
                    let (ax, ay) = deriv(a);
                    let (bx, by) = deriv(b);
                    let det = ax * by - ay * bx;
                    if !det.is_finite() || det.abs() < 1e-15 {
                        break;
                    }
                    let (av, bv) = (value(a), value(b));
                    if av.abs().max(bv.abs()) < 1e-10 {
                        converged = true;
                        break;
                    }
                    tx -= (av * by - ay * bv) / det;
                    ty -= (ax * bv - av * bx) / det;
                }
                if converged && tx >= -1e-6 && tx <= 1.000001 && ty >= -1e-6 && ty <= 1.000001 {
                    return Some((x as f64 + tx.clamp(0.0, 1.0), y as f64 + ty.clamp(0.0, 1.0)));
                }
            }
        }
        None
    }

    pub fn sample(&self, values: &[f64], position: (f64, f64)) -> Option<f64> {
        if values.len() != self.nx * self.ny {
            return None;
        }
        let (x, y) = position;
        let ix = (x.floor() as usize).min(self.nx - 2);
        let iy = (y.floor() as usize).min(self.ny - 2);
        let (tx, ty) = (x - ix as f64, y - iy as f64);
        let points = [
            (iy * self.nx + ix, (1.0 - tx) * (1.0 - ty)),
            (iy * self.nx + ix + 1, tx * (1.0 - ty)),
            ((iy + 1) * self.nx + ix, (1.0 - tx) * ty),
            ((iy + 1) * self.nx + ix + 1, tx * ty),
        ];
        let mut result = 0.0;
        for (i, w) in points {
            if w > 1e-12 {
                let v = values[i];
                if !v.is_finite() || v.abs() > 1e20 {
                    return None;
                }
                result += w * v;
            }
        }
        Some(result)
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct PointForecast {
    pub station_id: String,
    #[serde(default)]
    pub terrain_m: Option<f64>,
    pub values: BTreeMap<String, f64>,
}

pub struct ArmData {
    pub label: String,
    pub grid: Option<GridData>,
    pub fields: BTreeMap<String, Vec<f64>>,
    pub points: BTreeMap<String, PointForecast>,
    pub provenance: Value,
}

pub fn arm_spacing(arm: &ArmData) -> Option<(f64, f64)> {
    let metadata = &arm.provenance["metadata"];
    metadata["dx_km"]
        .as_f64()
        .zip(metadata["dy_km"].as_f64())
        .filter(|(x, y)| x.is_finite() && y.is_finite() && *x > 0.0 && *y > 0.0)
        .or_else(|| arm.grid.as_ref()?.spacing_km())
}

pub fn validate_arm_time(spec: &ArmSpec, arm: &ArmData, target: &str) -> Result<(), String> {
    let expected = crate::verification::parse_time(target)?;
    let mut verified = false;
    for (i, value) in [
        &arm.provenance["metadata"]["valid_time"],
        &arm.provenance["point_extract"]["valid_time"],
    ]
    .into_iter()
    .enumerate()
    {
        if let Some(text) = value.as_str() {
            if crate::verification::parse_time(text)? != expected {
                return Err(format!(
                    "{} source valid time {text} differs from requested {target}",
                    spec.label
                ));
            }
            if i == 0 || spec.kind == "points" {
                verified = true;
            }
        }
    }
    if let Some(text) = &spec.valid_time {
        if crate::verification::parse_time(text)? != expected {
            return Err("arm declared valid time differs from requested time".into());
        }
    }
    if !verified && spec.kind == "npz" {
        let initial = spec
            .init_time
            .as_deref()
            .ok_or("legacy NPZ requires explicit init_time and hour metadata")?;
        let hour = spec
            .hour
            .ok_or("legacy NPZ requires an explicit forecast hour")?;
        if crate::verification::parse_time(initial)? + chrono::Duration::hours(i64::from(hour))
            != expected
        {
            return Err("legacy NPZ cycle and hour differ from requested valid time".into());
        }
        let stem = spec
            .path
            .file_stem()
            .and_then(|s| s.to_str())
            .ok_or("legacy NPZ filename is unreadable")?;
        let token = stem.rsplit('-').next().unwrap_or(stem);
        let actual = token
            .strip_prefix('f')
            .and_then(|s| s.parse::<u16>().ok())
            .ok_or("legacy NPZ filename must end in -fNN to bind its forecast hour")?;
        if actual != hour {
            return Err("legacy NPZ filename hour differs from its metadata".into());
        }
        verified = true;
    }
    if !verified {
        return Err(format!("{} has no native valid-time metadata", spec.label));
    }
    Ok(())
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StationObservation {
    pub station_id: String,
    pub latitude: f64,
    pub longitude: f64,
    pub elevation_m: f64,
    pub observation_time: String,
    pub values: BTreeMap<String, f64>,
}

pub struct RadarData {
    pub grid: GridData,
    pub values: Vec<f64>,
    pub valid: Vec<bool>,
    pub valid_time: String,
    pub quantity: String,
    pub units: String,
    pub provenance: Value,
}

/// Retain the verification planes before disposable history is removed.
pub fn write_snapshot(path: &Path, arm: &ArmData) -> Result<(), String> {
    let grid = arm
        .grid
        .as_ref()
        .ok_or("a grid snapshot needs native geometry")?;
    let mut arrays = arm.fields.clone();
    arrays.insert(
        "lat".into(),
        grid.lat.iter().copied().map(f64::from).collect(),
    );
    arrays.insert(
        "lon".into(),
        grid.lon.iter().copied().map(f64::from).collect(),
    );
    let mut crc_table = [0_u32; 256];
    for (i, slot) in crc_table.iter_mut().enumerate() {
        let mut c = i as u32;
        for _ in 0..8 {
            c = if c & 1 == 1 {
                0xedb88320 ^ (c >> 1)
            } else {
                c >> 1
            };
        }
        *slot = c;
    }
    let mut bytes = Vec::new();
    let mut directory = Vec::new();
    for (key, values) in arrays {
        let name = format!("{key}.npy");
        let header = format!(
            "{{'descr': '<f8', 'fortran_order': False, 'shape': ({}, {}), }}",
            grid.ny, grid.nx
        );
        let padding = (64 - (10 + header.len() + 1) % 64) % 64;
        let mut npy = b"\x93NUMPY\x01\x00".to_vec();
        npy.extend_from_slice(&((header.len() + padding + 1) as u16).to_le_bytes());
        npy.extend_from_slice(header.as_bytes());
        npy.extend(std::iter::repeat_n(b' ', padding));
        npy.push(b'\n');
        for v in values {
            npy.extend_from_slice(&v.to_le_bytes());
        }
        let mut crc = 0xffffffff_u32;
        for &b in &npy {
            crc = crc_table[((crc ^ u32::from(b)) & 255) as usize] ^ (crc >> 8);
        }
        crc ^= 0xffffffff;
        let size =
            u32::try_from(npy.len()).map_err(|_| "verification snapshot member exceeds 4 GiB")?;
        let offset =
            u32::try_from(bytes.len()).map_err(|_| "verification snapshot exceeds 4 GiB")?;
        bytes.extend_from_slice(b"PK\x03\x04");
        bytes.extend_from_slice(&20_u16.to_le_bytes());
        bytes.extend_from_slice(&[0; 4]);
        bytes.extend_from_slice(&[0, 0, 33, 0]);
        bytes.extend_from_slice(&crc.to_le_bytes());
        bytes.extend_from_slice(&size.to_le_bytes());
        bytes.extend_from_slice(&size.to_le_bytes());
        bytes.extend_from_slice(&(name.len() as u16).to_le_bytes());
        bytes.extend_from_slice(&0_u16.to_le_bytes());
        bytes.extend_from_slice(name.as_bytes());
        bytes.extend_from_slice(&npy);
        directory.push((name, crc, size, offset));
    }
    let central = u32::try_from(bytes.len()).map_err(|_| "verification snapshot exceeds 4 GiB")?;
    for (name, crc, size, offset) in &directory {
        bytes.extend_from_slice(b"PK\x01\x02");
        bytes.extend_from_slice(&20_u16.to_le_bytes());
        bytes.extend_from_slice(&20_u16.to_le_bytes());
        bytes.extend_from_slice(&[0; 4]);
        bytes.extend_from_slice(&[0, 0, 33, 0]);
        bytes.extend_from_slice(&crc.to_le_bytes());
        bytes.extend_from_slice(&size.to_le_bytes());
        bytes.extend_from_slice(&size.to_le_bytes());
        bytes.extend_from_slice(&(name.len() as u16).to_le_bytes());
        bytes.extend_from_slice(&[0; 12]);
        bytes.extend_from_slice(&offset.to_le_bytes());
        bytes.extend_from_slice(name.as_bytes());
    }
    let central_size =
        u32::try_from(bytes.len()).map_err(|_| "verification snapshot exceeds 4 GiB")? - central;
    bytes.extend_from_slice(b"PK\x05\x06");
    bytes.extend_from_slice(&[0; 4]);
    bytes.extend_from_slice(&(directory.len() as u16).to_le_bytes());
    bytes.extend_from_slice(&(directory.len() as u16).to_le_bytes());
    bytes.extend_from_slice(&central_size.to_le_bytes());
    bytes.extend_from_slice(&central.to_le_bytes());
    bytes.extend_from_slice(&[0; 2]);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    std::fs::write(path, bytes).map_err(|e| e.to_string())
}

pub fn read_json(path: &Path) -> Result<Value, String> {
    serde_json::from_slice(
        &std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?,
    )
    .map_err(|e| format!("parse {}: {e}", path.display()))
}

pub fn source_identity(path: &Path) -> Result<Value, String> {
    let mut file =
        std::fs::File::open(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let mut hash = Sha256::new();
    let mut buffer = [0_u8; 65536];
    let mut size = 0_u64;
    loop {
        let count = file.read(&mut buffer).map_err(|e| e.to_string())?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
        size += count as u64;
    }
    Ok(serde_json::json!({"path":path,"sha256":format!("{:x}",hash.finalize()),"bytes":size}))
}

pub fn load_station_observations(
    path: &Path,
    valid_time: &str,
) -> Result<Vec<StationObservation>, String> {
    let record = read_json(path)?;
    let generic = record["schema"] == "gpuwm-obs.station-observations.v1";
    if (!record["schema"]
        .as_str()
        .unwrap_or("")
        .starts_with("gpuwm-obs.asos-surface.v")
        && !generic)
        || record["provenance"]["is_stub"].as_bool() == Some(true)
    {
        return Err("station verification requires decoded real surface observations".into());
    }
    if generic && (record["source"].as_str().is_none() || record["units"].as_object().is_none()
        || record["provenance"]["sha256"].as_str().is_none()) {
        return Err("generic station observations require source, per-quantity units and original-source SHA256 to prevent unbound profile or snow scores".into());
    }
    let stations = record["stations"]
        .as_array()
        .ok_or("station record has no station table")?;
    let reports = record["reports"]
        .as_array()
        .ok_or("station record has no reports")?;
    let mut result = BTreeMap::new();
    let target = crate::verification::parse_time(valid_time)?;
    for report in reports.iter().filter(|r| {
        r["valid_time"]
            .as_str()
            .and_then(|s| crate::verification::parse_time(s).ok())
            == Some(target)
    }) {
        let id = report["station_id"]
            .as_str()
            .ok_or("station report has no id")?;
        let station = stations
            .iter()
            .find(|s| s["station_id"].as_str() == Some(id))
            .ok_or("report station is absent from its frozen table")?;
        let number = |key: &str| {
            station[key]
                .as_f64()
                .ok_or_else(|| format!("station {id} has no {key}"))
        };
        let mut values: BTreeMap<String, f64> =
            serde_json::from_value(report["values"].clone()).map_err(|e| e.to_string())?;
        if !generic {
            if let (Some(&t), Some(&td)) = (values.get("temperature_2m"), values.get("dewpoint_2m")) {
                let (heat, apparent) = rustwx_calc::derived_heat::screen_heat(t, td, values.get("wind_speed_10m").copied());
                for (name, value) in [("heat_index_2m", heat), ("apparent_temperature_2m", apparent)] {
                    if value.is_finite() { values.insert(name.into(), value); }
                }
            }
        }
        let observation_time = report["observation_time"]
            .as_str()
            .filter(|s| !s.is_empty())
            .ok_or("station verification requires each report's actual observation_time")?;
        let observed = StationObservation {
            station_id: id.into(),
            latitude: number("latitude")?,
            longitude: number("longitude")?,
            elevation_m: number("elevation_m")?,
            observation_time: observation_time.into(),
            values,
        };
        if result.insert(id.to_string(), observed).is_some() {
            return Err(format!("two reports serve station {id} at {valid_time}"));
        }
    }
    Ok(result.into_values().collect())
}

#[derive(Clone, Debug)]
struct Array {
    shape: Vec<usize>,
    values: Vec<f64>,
}

/// Read NPY f32/f64 arrays from ordinary stored or deflated NPZ members. ZIP
/// central-directory sizes are authoritative, including ZIP64 local headers.
fn read_npz(
    path: &Path,
    selected: &BTreeSet<String>,
    metadata: &mut BTreeMap<String, String>,
) -> Result<BTreeMap<String, Array>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let u16at = |at: usize| -> Result<usize, String> {
        bytes
            .get(at..at + 2)
            .map(|b| u16::from_le_bytes([b[0], b[1]]) as usize)
            .ok_or("truncated ZIP".into())
    };
    let u32at = |at: usize| -> Result<usize, String> {
        bytes
            .get(at..at + 4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()) as usize)
            .ok_or("truncated ZIP".into())
    };
    let end = bytes
        .windows(4)
        .rposition(|w| w == b"PK\x05\x06")
        .ok_or("NPZ has no ZIP directory")?;
    let members = u16at(end + 10)?;
    let mut offset = u32at(end + 16)?;
    let mut result = BTreeMap::new();
    for _ in 0..members {
        if bytes.get(offset..offset + 4) != Some(b"PK\x01\x02") {
            return Err("invalid NPZ ZIP directory".into());
        }
        let method = u16at(offset + 10)?;
        let compressed = u32at(offset + 20)?;
        let expanded = u32at(offset + 24)?;
        let name_len = u16at(offset + 28)?;
        let extra = u16at(offset + 30)?;
        let comment = u16at(offset + 32)?;
        let local = u32at(offset + 42)?;
        let name = std::str::from_utf8(
            bytes
                .get(offset + 46..offset + 46 + name_len)
                .ok_or("truncated member name")?,
        )
        .map_err(|e| e.to_string())?;
        let next = offset + 46 + name_len + extra + comment;
        let Some(key) = name.strip_suffix(".npy") else {
            offset = next;
            continue;
        };
        let selected_field = selected.contains(key);
        let selected_metadata = matches!(
            key,
            "valid" | "valid_time" | "source_file" | "source_frame" | "crop"
        );
        if !selected_field && !selected_metadata {
            offset = next;
            continue;
        }
        if expanded > 512 * 1024 * 1024 {
            return Err(format!("NPZ member {name} exceeds 512 MiB"));
        }
        let begin = local + 30 + u16at(local + 26)? + u16at(local + 28)?;
        let payload = bytes
            .get(begin..begin + compressed)
            .ok_or("truncated NPZ payload")?;
        let raw = match method {
            0 => payload.to_vec(),
            8 => {
                let mut output = Vec::with_capacity(expanded);
                flate2::read::DeflateDecoder::new(payload)
                    .take(expanded as u64 + 1)
                    .read_to_end(&mut output)
                    .map_err(|e| e.to_string())?;
                output
            }
            _ => return Err(format!("NPZ compression {method} is unsupported")),
        };
        if raw.len() != expanded {
            return Err(format!(
                "NPZ member {name} length differs from its directory"
            ));
        }
        if selected_field {
            result.insert(
                key.into(),
                read_npy(&raw).map_err(|e| format!("NPZ selected member {name}: {e}"))?,
            );
        } else if let Some(text) =
            read_npy_text(&raw).map_err(|e| format!("NPZ metadata member {name}: {e}"))?
        {
            metadata.insert(key.into(), text);
        }
        offset = next;
    }
    Ok(result)
}

fn npy_header(bytes: &[u8]) -> Result<(&str, usize), String> {
    if bytes.get(..6) != Some(b"\x93NUMPY") {
        return Err("not an NPY array".into());
    }
    let (start, length) = match bytes.get(6) {
        Some(1) => (
            10,
            u16::from_le_bytes(bytes.get(8..10).ok_or("truncated NPY")?.try_into().unwrap())
                as usize,
        ),
        Some(2) | Some(3) => (
            12,
            u32::from_le_bytes(bytes.get(8..12).ok_or("truncated NPY")?.try_into().unwrap())
                as usize,
        ),
        _ => return Err("unsupported NPY version".into()),
    };
    let header = std::str::from_utf8(
        bytes
            .get(start..start + length)
            .ok_or("truncated NPY header")?,
    )
    .map_err(|e| e.to_string())?;
    Ok((header, start + length))
}

fn read_npy_text(bytes: &[u8]) -> Result<Option<String>, String> {
    let (header, begin) = npy_header(bytes)?;
    let Some(dtype) = header
        .split("'<U")
        .nth(1)
        .and_then(|s| s.split('\'').next())
    else {
        return Ok(None);
    };
    let scalar_shape = header
        .split("'shape':")
        .nth(1)
        .and_then(|s| s.split_once('('))
        .and_then(|(_, s)| s.split_once(')'))
        .map(|(s, _)| s.trim().is_empty())
        .unwrap_or(false);
    if !scalar_shape {
        return Ok(None);
    }
    let characters = dtype.parse::<usize>().map_err(|e| e.to_string())?;
    if characters > 4096 {
        return Err("scalar NPZ metadata exceeds 4096 characters".into());
    }
    let payload = &bytes[begin..];
    if characters.checked_mul(4) != Some(payload.len()) {
        return Err("Unicode metadata length differs from NPY dtype".into());
    }
    let mut text = String::new();
    let mut padding = false;
    for raw in payload.chunks_exact(4) {
        let code = u32::from_le_bytes(raw.try_into().unwrap());
        if code == 0 {
            padding = true;
            continue;
        }
        if padding {
            return Err("Unicode metadata contains nonzero data after padding".into());
        }
        text.push(char::from_u32(code).ok_or("Unicode metadata contains an invalid code point")?);
    }
    Ok(Some(text))
}

fn read_npy(bytes: &[u8]) -> Result<Array, String> {
    let (header, begin) = npy_header(bytes)?;
    if !header.contains("'fortran_order': False") {
        return Err("NPY verification arrays must be row-major".into());
    }
    let shape_text = header
        .split("'shape':")
        .nth(1)
        .and_then(|s| s.split_once('('))
        .and_then(|(_, s)| s.split_once(')'))
        .ok_or("NPY has no shape")?
        .0;
    let shape = shape_text
        .split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.trim().parse::<usize>().map_err(|e| e.to_string()))
        .collect::<Result<Vec<_>, _>>()?;
    let size = shape
        .iter()
        .try_fold(1_usize, |a, &b| a.checked_mul(b))
        .ok_or("NPY shape overflow")?;
    let raw = &bytes[begin..];
    let width = if header.contains("'<f8'") || header.contains("'=f8'") {
        8
    } else if header.contains("'<f4'") || header.contains("'=f4'") {
        4
    } else {
        return Err("NPY verification arrays must be little-endian f32 or f64".into());
    };
    if size.checked_mul(width) != Some(raw.len()) {
        return Err("NPY shape disagrees with payload length".into());
    }
    let values = raw
        .chunks_exact(width)
        .map(|b| {
            if width == 8 {
                f64::from_le_bytes(b.try_into().unwrap())
            } else {
                f64::from(f32::from_le_bytes(b.try_into().unwrap()))
            }
        })
        .collect();
    Ok(Array { shape, values })
}

fn grid_from_arrays(arrays: &BTreeMap<String, Array>) -> Result<GridData, String> {
    let find = |names: &[&str]| {
        names
            .iter()
            .find_map(|name| arrays.get(*name))
            .ok_or_else(|| format!("coordinate array missing: {names:?}"))
    };
    let lat = find(&["lat", "latitude", "XLAT", "lat_deg"])?;
    let lon = find(&["lon", "longitude", "XLONG", "lon_deg"])?;
    let mut shape = lat.shape.clone();
    while shape.len() > 2 && shape[0] == 1 {
        shape.remove(0);
    }
    if shape.len() != 2 || lat.values.len() != lon.values.len() {
        return Err("coordinate arrays are not paired 2D planes".into());
    }
    Ok(GridData {
        lat: lat.values.iter().map(|&x| x as f32).collect(),
        lon: lon.values.iter().map(|&x| x as f32).collect(),
        ny: shape[0],
        nx: shape[1],
        projection: None,
    })
}

fn standard_fields() -> BTreeMap<String, String> {
    [
        ("temperature_2m", "T2"),
        ("dewpoint_2m", "td2"),
        ("wind_speed_10m", "wind_speed"),
        ("composite_reflectivity", "REFL_10CM"),
        ("precipitation_1h", "precip_1h_mm"),
    ]
    .map(|(a, b)| (a.into(), b.into()))
    .into()
}

pub fn hourly_messages(
    grib: &grib_core::grib2::Grib2File,
    lead: u16,
) -> grib_core::grib2::Grib2File {
    let hours = |unit: u8, value: u32| -> Option<u32> {
        match unit {
            1 => Some(value),
            0 => (value % 60 == 0).then_some(value / 60),
            13 => (value % 3600 == 0).then_some(value / 3600),
            2 => value.checked_mul(24),
            10 => value.checked_mul(3),
            11 => value.checked_mul(6),
            12 => value.checked_mul(12),
            _ => None,
        }
    };
    let cycle = grib.messages.first().map(|m| m.reference_time);
    grib_core::grib2::Grib2File {
        messages: grib
            .messages
            .iter()
            .filter(|m| {
                lead > 0
                    && Some(m.reference_time) == cycle
                    && m.product.statistical_process_type == Some(1)
                    && m.product.statistical_time_range_hours() == Some(1)
                    && hours(m.product.time_range_unit, m.product.forecast_time)
                        == Some(u32::from(lead) - 1)
                    && m.product.end_of_interval
                        == Some(m.reference_time + chrono::Duration::hours(i64::from(lead)))
            })
            .cloned()
            .collect(),
    }
}

pub fn load_arm(spec: &ArmSpec) -> Result<ArmData, String> {
    load_arm_at(spec, spec.time_index)
}

/// Read one native history record; other formats retain their native selector.
pub fn load_arm_at(spec: &ArmSpec, time_index: usize) -> Result<ArmData, String> {
    let fields = if spec.fields.is_empty() {
        standard_fields()
    } else {
        spec.fields.clone()
    };
    let mut result = ArmData {
        label: spec.label.clone(),
        grid: None,
        fields: BTreeMap::new(),
        points: BTreeMap::new(),
        provenance: serde_json::json!({"artifact":source_identity(&spec.path)?,"prepared_from":spec.source_identity}),
    };
    match spec.kind.as_str() {
        "points" => {}
        "npz" => {
            let coordinates: BTreeSet<String> = [
                "lat",
                "latitude",
                "XLAT",
                "lat_deg",
                "lon",
                "longitude",
                "XLONG",
                "lon_deg",
            ]
            .into_iter()
            .map(str::to_string)
            .collect();
            let mut selected: BTreeSet<String> = fields.values().cloned().collect();
            if spec.grid_path.is_none() {
                selected.extend(coordinates.iter().cloned());
            }
            let mut embedded = BTreeMap::new();
            let arrays = read_npz(&spec.path, &selected, &mut embedded)?;
            let grid_arrays = if let Some(path) = &spec.grid_path {
                read_npz(path, &coordinates, &mut BTreeMap::new())?
            } else {
                arrays.clone()
            };
            result.grid = Some(grid_from_arrays(&grid_arrays)?);
            result.provenance["npz_metadata"] =
                serde_json::to_value(&embedded).map_err(|e| e.to_string())?;
            if let Some(time) = embedded.get("valid_time").or_else(|| embedded.get("valid")) {
                crate::verification::parse_time(time)?;
                result.provenance["metadata"] = serde_json::json!({"valid_time":time});
            }
            let metadata_path = spec.path.with_extension("metadata.json");
            if metadata_path.is_file() {
                result.provenance["metadata"] = read_json(&metadata_path)?;
                if let Some(time) = embedded.get("valid_time").or_else(|| embedded.get("valid")) {
                    if result.provenance["metadata"]["valid_time"]
                        .as_str()
                        .map(crate::verification::parse_time)
                        .transpose()?
                        != Some(crate::verification::parse_time(time)?)
                    {
                        return Err("NPZ field timestamp differs from its prepared sidecar".into());
                    }
                }
                result.provenance["metadata_artifact"] = source_identity(&metadata_path)?;
                let metadata = &result.provenance["metadata"];
                if metadata["schema"].as_str() != Some("gpuwm.verify-visuals.inputs.v1")
                    || metadata["artifact"]["sha256"] != result.provenance["artifact"]["sha256"]
                {
                    return Err("verification NPZ sidecar does not bind these field bytes".into());
                }
                if let Some(projection) = metadata.get("projection").filter(|v| !v.is_null()) {
                    result.grid.as_mut().unwrap().projection = Some(
                        serde_json::from_value(projection.clone()).map_err(|e| e.to_string())?,
                    );
                }
            }
            for (quantity, key) in fields {
                if let Some(array) = arrays.get(&key) {
                    result.fields.insert(quantity, array.values.clone());
                }
            }
        }
        "netcdf" | "wrfout" => {
            let file = netcrust::open(&spec.path).map_err(|e| e.to_string())?;
            let wrf = wrf_core::WrfFile::open(&spec.path).ok();
            // Share the renderer's exact time-axis reader. NetCDF-4 char
            // matrices can decode as one string per character; selecting
            // read_strings()[time_index] would label a whole frame "2".
            let axis = crate::local_import::netcdf_source_times(&file, &spec.path)
                .map_err(|e| e.to_string())?;
            let valid = &axis.records.get(time_index)
                .ok_or("verification record is outside the native time axis")?.label;
            result.provenance["metadata"] = serde_json::json!({"valid_time":valid.replace('_',"T"),"init_time":file.attribute("SIMULATION_START_DATE").or_else(||file.attribute("START_DATE")).and_then(|a|a.as_string().map(|s|s.replace('_',"T"))),"dx_km":file.attribute("DX").and_then(|a|a.as_f64()).map(|v|v*0.001),"dy_km":file.attribute("DY").and_then(|a|a.as_f64()).map(|v|v*0.001)});
            let read = |name: &str| -> Result<Array, String> {
                let a = file
                    .read_array_f64_record_or_all(name, time_index as u64)
                    .map_err(|e| e.to_string())?;
                Ok(Array {
                    shape: a.shape().to_vec(),
                    values: a.into_values(),
                })
            };
            let mut coords = BTreeMap::new();
            coords.insert("XLAT".into(), read("XLAT")?);
            coords.insert("XLONG".into(), read("XLONG")?);
            result.grid = Some(grid_from_arrays(&coords)?);
            result.grid.as_mut().unwrap().projection =
                crate::wrf_process::netcdf_projection(&file);
            result.provenance["metadata"]["projection"] =
                serde_json::to_value(result.grid.as_ref().unwrap().projection.as_ref())
                    .map_err(|e| e.to_string())?;
            let cells = result.grid.as_ref().unwrap().nx * result.grid.as_ref().unwrap().ny;
            let requested: BTreeSet<_> = fields.keys().cloned().collect();
            if requested.iter().any(|q|q.starts_with("ivt_") || q=="integrated_water_vapour") {
                let native=crate::wrf_column_planes::read_vapour_integrals(wrf.as_ref().ok_or("IVT requires native WRF model levels")?,time_index)?;
                for (quantity,values) in [
                    ("ivt_1000_300",native.magnitude),("ivt_eastward_1000_300",native.eastward),
                    ("ivt_northward_1000_300",native.northward),("integrated_water_vapour",native.iwv),
                    ("ivt_full_column",native.full_magnitude),("ivt_eastward_full_column",native.full_eastward),
                    ("ivt_northward_full_column",native.full_northward)] {
                    if requested.contains(quantity) { result.fields.insert(quantity.into(),values.into_iter().map(f64::from).collect()); }
                }
                result.provenance["vapour_integral"]=serde_json::json!({"method":"native-pressure trapezoids; specific humidity; earth-relative winds", "operational_bounds_hpa":[1000,300],"reference_bounds":"surface to finite model lid; ERA5 reference surface to atmosphere top"});
            }
            for (quantity, key) in fields {
                if (quantity.starts_with("ivt_") || quantity=="integrated_water_vapour") && result.fields.contains_key(&quantity) { continue; }
                if let Ok(array) = read(&key) {
                    let values =
                        if quantity == "composite_reflectivity" && array.values.len() > cells {
                            let mut plane = vec![f64::NEG_INFINITY; cells];
                            for level in array.values.chunks_exact(cells) {
                                for (p, &v) in plane.iter_mut().zip(level) {
                                    if v.is_finite() {
                                        *p = p.max(v);
                                    }
                                }
                            }
                            plane
                        } else {
                            array.values
                        };
                    if values.len() == cells {
                        result.fields.insert(quantity, values);
                    }
                }
            }
            if !result.fields.contains_key("dewpoint_2m") {
                if let Some(wrf) = wrf.as_ref() {
                    let opts = wrf_core::ComputeOpts {
                        units: Some("K".into()),
                        ..Default::default()
                    };
                    if let Ok(output) = crate::wrf_process::isolate_panics("2 m dewpoint", || {
                        wrf_core::getvar(wrf, "td2", Some(time_index), &opts).map_err(|e| e.to_string())
                    }) {
                        result.fields.insert("dewpoint_2m".into(), output.data);
                    }
                }
            }
            if !result.fields.contains_key("wind_speed_10m") {
                if let (Ok(u), Ok(v)) = (read("U10"), read("V10")) {
                    result.fields.insert(
                        "wind_speed_10m".into(),
                        u.values
                            .iter()
                            .zip(&v.values)
                            .map(|(u, v)| u.hypot(*v))
                            .collect(),
                    );
                }
            }
            if !result.fields.contains_key("precipitation_1h") {
                if let Some(before) = &spec.previous_path {
                    let previous = netcrust::open(before).map_err(|e| e.to_string())?;
                    let now_time = result.provenance["metadata"]["valid_time"].as_str().ok_or(
                        "hourly precipitation needs the current frame's native valid time",
                    )?;
                    let old_axis = crate::local_import::netcdf_source_times(&previous, before)
                        .map_err(|e| e.to_string())?;
                    let old_time = old_axis.records.first()
                        .ok_or("previous frame has no valid time")?.label.replace('_', "T");
                    if (crate::verification::parse_time(now_time)?
                        - crate::verification::parse_time(&old_time)?)
                    .num_seconds()
                        != 3600
                    {
                        return Err(
                            "hourly precipitation frames are not exactly one hour apart".into()
                        );
                    }
                    for name in ["XLAT", "XLONG"] {
                        let a = read(name)?;
                        let b = previous
                            .read_array_f64_first_record_or_all(name)
                            .map_err(|e| e.to_string())?;
                        if a.values != b.values() {
                            return Err(
                                "hourly precipitation frames have different native geometry".into(),
                            );
                        }
                    }
                    let mut now = vec![0.0; cells];
                    let mut old = vec![0.0; cells];
                    let mut carried = false;
                    for name in ["RAINC", "RAINNC", "RAINSH"] {
                        if file.variable(name).is_some() != previous.variable(name).is_some() {
                            return Err(format!(
                                "precipitation component {name} occurs in only one hourly frame"
                            ));
                        }
                        if file.variable(name).is_none() {
                            continue;
                        }
                        if let (Ok(a), Ok(b)) = (
                            read(name),
                            previous.read_array_f64_first_record_or_all(name),
                        ) {
                            if a.values.len() == cells && b.len() == cells {
                                carried = true;
                                for ((n, o), (a, b)) in now
                                    .iter_mut()
                                    .zip(&mut old)
                                    .zip(a.values.iter().zip(b.values()))
                                {
                                    *n += a;
                                    *o += b;
                                }
                            } else {
                                return Err(format!(
                                    "precipitation component {name} changes shape between frames"
                                ));
                            }
                        } else {
                            return Err(format!(
                                "cannot decode hourly precipitation component {name}"
                            ));
                        }
                    }
                    if carried {
                        if now
                            .iter()
                            .zip(&old)
                            .any(|(n, o)| !n.is_finite() || !o.is_finite() || n - o < -0.0001)
                        {
                            return Err("hourly precipitation accumulated totals reset or contain missing values".into());
                        }
                        result.fields.insert(
                            "precipitation_1h".into(),
                            now.iter().zip(old).map(|(n, o)| (n - o).max(0.0)).collect(),
                        );
                    }
                }
            }
        }
        "grib" => {
            let bytes = std::fs::read(&spec.path).map_err(|e| e.to_string())?;
            let grib =
                grib_core::grib2::Grib2File::from_bytes(&bytes).map_err(|e| e.to_string())?;
            if let Some(message) = grib.messages.first() {
                let initial = message.reference_time.and_utc();
                result.provenance["metadata"] = serde_json::json!({"init_time":initial.format("%Y-%m-%dT%H:%M:%S").to_string(),"valid_time":spec.hour.map(|h|(initial+chrono::Duration::hours(i64::from(h))).format("%Y-%m-%dT%H:%M:%S").to_string())});
            }
            let selector = |q: &str| -> Option<FieldSelector> {
                Some(match q {
                    "solar_ghi" => FieldSelector::surface(CanonicalField::DownwardShortwaveRadiationFlux),
                    "temperature_2m" => FieldSelector::height_agl(CanonicalField::Temperature, 2),
                    "dewpoint_2m" => FieldSelector::height_agl(CanonicalField::Dewpoint, 2),
                    "wind_u_10m" => FieldSelector::height_agl(CanonicalField::UWind, 10),
                    "wind_v_10m" => FieldSelector::height_agl(CanonicalField::VWind, 10),
                    "composite_reflectivity" => {
                        FieldSelector::entire_atmosphere(CanonicalField::CompositeReflectivity)
                    }
                    "precipitation_1h" => {
                        FieldSelector::surface(CanonicalField::TotalPrecipitation)
                    }
                    _ => return None,
                })
            };
            let mut wanted: Vec<String> = fields.keys().cloned().collect();
            if wanted.iter().any(|q| q == "wind_speed_10m") {
                wanted.extend(["wind_u_10m".into(), "wind_v_10m".into()]);
            }
            for q in wanted {
                let Some(select) = selector(&q) else { continue };
                let mut windowed = None;
                let hour = if q == "precipitation_1h" {
                    let Some(h) = spec.hour.filter(|&h| h > 0) else {
                        continue;
                    };
                    windowed = Some(hourly_messages(&grib, h));
                    if windowed.as_ref().unwrap().messages.is_empty() {
                        continue;
                    }
                    Some(h - 1)
                } else {
                    spec.hour
                };
                let extract = match hour {
                    Some(h) => rustwx_io::extract_fields_from_grib2_partial_at_forecast_hour(
                        windowed.as_ref().unwrap_or(&grib),
                        &[select],
                        h,
                    ),
                    None => rustwx_io::extract_fields_from_grib2_partial(
                        windowed.as_ref().unwrap_or(&grib),
                        &[select],
                    ),
                }
                .map_err(|e| e.to_string())?;
                if let Some(field) = extract.extracted.into_iter().next() {
                    result.grid.get_or_insert(GridData {
                        ny: field.grid.shape.ny,
                        nx: field.grid.shape.nx,
                        lat: field.grid.lat_deg,
                        lon: field.grid.lon_deg,
                        projection: field.projection,
                    });
                    result
                        .fields
                        .insert(q, field.values.into_iter().map(f64::from).collect());
                }
            }
            if let (Some(u), Some(v)) = (
                result.fields.get("wind_u_10m"),
                result.fields.get("wind_v_10m"),
            ) {
                result.fields.insert(
                    "wind_speed_10m".into(),
                    u.iter().zip(v).map(|(u, v)| u.hypot(*v)).collect(),
                );
            }
        }
        "store" => {
            let reader =
                rw_store::reader::HourReader::open(&spec.path).map_err(|e| e.to_string())?;
            if let Some(time) = reader.meta().exact_time() {
                result.provenance["metadata"] = serde_json::json!({"valid_time":chrono::DateTime::from_timestamp(time.valid_unix,0).map(|t|t.format("%Y-%m-%dT%H:%M:%S").to_string()),"init_time":time.origin_unix().and_then(|t|chrono::DateTime::from_timestamp(t,0)).map(|t|t.format("%Y-%m-%dT%H:%M:%S").to_string())});
            }
            let grid_path = spec.grid_path.clone().unwrap_or_else(|| {
                spec.path
                    .parent()
                    .unwrap_or(Path::new("."))
                    .join("grid.rwg")
            });
            let grid = rw_store::grid::GridFile::open(&grid_path).map_err(|e| e.to_string())?;
            for (q, key) in fields {
                if let Ok(f) = rw_store::read_grid_2d(&reader, &grid, &key) {
                    result.grid.get_or_insert(GridData {
                        ny: f.grid.shape.ny,
                        nx: f.grid.shape.nx,
                        lat: f.grid.lat_deg,
                        lon: f.grid.lon_deg,
                        projection: f.projection,
                    });
                    result
                        .fields
                        .insert(q, f.values.into_iter().map(f64::from).collect());
                }
            }
        }
        other => return Err(format!("unknown verification reader {other}")),
    }
    let points_path = if spec.kind == "points" {
        Some(&spec.path)
    } else {
        spec.points_path.as_ref()
    };
    if let Some(path) = points_path {
        let record = read_json(path)?;
        result.provenance["point_extract"] = serde_json::json!({"artifact":source_identity(path)?,"valid_time":record["valid_time"],"init_time":record["init_time"],"support_hash":record["support_hash"],"method_id":record["method_id"]});
        for row in record["points"]
            .as_array()
            .ok_or("point extract has no points")?
        {
            let point: PointForecast =
                serde_json::from_value(row.clone()).map_err(|e| e.to_string())?;
            if result
                .points
                .insert(point.station_id.clone(), point)
                .is_some()
            {
                return Err("point extract contains duplicate station ids".into());
            }
        }
    }
    if let Some(grid) = &result.grid {
        for (q, values) in &result.fields {
            if values.len() != grid.nx * grid.ny {
                return Err(format!(
                    "{q} has {} values on a {}x{} grid",
                    values.len(),
                    grid.ny,
                    grid.nx
                ));
            }
        }
    }
    Ok(result)
}

pub fn load_radar(spec: &RadarSpec) -> Result<RadarData, String> {
    if matches!(spec.path.extension().and_then(|s|s.to_str()),Some("nc"|"nc4")) {
        return crate::gridded_verification::load(spec);
    }
    fn pack(
        path: &Path,
    ) -> Result<(Value, Vec<u8>, BTreeMap<String, rw_obs::pack::ArrayEntry>), String> {
        let raw = std::fs::read(path).map_err(|e| e.to_string())?;
        let (meta, payload): (Value, Vec<u8>) =
            rw_obs::pack::decode_pack(&raw).map_err(|e| e.to_string())?;
        if meta["content_sha256"].as_str() != Some(rw_obs::pack::payload_digest(&payload).as_str())
        {
            return Err("observation pack payload digest mismatch".into());
        }
        let arrays = serde_json::from_value(meta["arrays"].clone()).map_err(|e| e.to_string())?;
        rw_obs::pack::validate_arrays(&arrays, payload.len()).map_err(|e| e.to_string())?;
        Ok((meta, payload, arrays))
    }
    let (meta, payload, arrays) = pack(&spec.path)?;
    let (geometry, geo, coordinates) = pack(&spec.grid_path)?;
    if meta["grid"] != geometry["grid"]
        || meta["provenance"]["product"] != geometry["source_product"]
    {
        return Err(
            "observation field and geometry describe different grids or source products".into(),
        );
    }
    if meta["schema"].as_str() != Some(rw_obs::pack::GRID_SCHEMA)
        || geometry["schema"].as_str() != Some(rw_obs::pack::GEO_SCHEMA)
    {
        return Err("verification needs native observation grid and geometry schemas".into());
    }
    let floats = |data: &[u8],
                  entries: &BTreeMap<String, rw_obs::pack::ArrayEntry>,
                  key: &str|
     -> Result<Vec<f64>, String> {
        let a = entries
            .get(key)
            .ok_or_else(|| format!("observation array {key} absent"))?;
        if a.dtype != "<f8" {
            return Err(format!("{key} is not f64"));
        }
        Ok(data[a.offset..a.offset + a.bytes]
            .chunks_exact(8)
            .map(|b| f64::from_le_bytes(b.try_into().unwrap()))
            .collect())
    };
    let values = floats(&payload, &arrays, "values")?;
    let valid = arrays
        .get("valid")
        .ok_or("observation validity mask absent")?;
    let lat_key = if coordinates.contains_key("latitude") {
        "latitude"
    } else {
        "lat_deg"
    };
    let lon_key = if coordinates.contains_key("longitude") {
        "longitude"
    } else {
        "lon_deg"
    };
    let lat = floats(&geo, &coordinates, lat_key)?;
    let lon = floats(&geo, &coordinates, lon_key)?;
    let shape = &arrays["values"].shape;
    if shape.len() != 2
        || coordinates[lat_key].shape != *shape
        || coordinates[lon_key].shape != *shape
        || valid.shape != *shape
        || valid.dtype != "<u1"
        || values.len() != lat.len()
        || values.len() != lon.len()
        || valid.bytes != values.len()
    {
        return Err("observation geometry, values and mask differ".into());
    }
    let quantity = meta["quantity"]
        .as_str()
        .ok_or("observation pack has no quantity")?
        .to_string();
    let expected = if spec.quantity == "precipitation_1h" {
        "precipitation_accumulation"
    } else {
        spec.quantity.as_str()
    };
    if quantity != expected {
        return Err(format!(
            "observation quantity {quantity} cannot verify {}",
            spec.quantity
        ));
    }
    let expected_units = if quantity == "precipitation_accumulation" {
        "mm"
    } else {
        "dBZ"
    };
    if meta["units"].as_str() != Some(expected_units) {
        return Err("observation units disagree with the verification quantity".into());
    }
    if quantity == "precipitation_accumulation"
        && meta["accumulation_seconds"].as_u64() != Some(3600)
    {
        return Err("1 h verification requires a 3600 second observed accumulation".into());
    }
    for (&value, &mask) in values
        .iter()
        .zip(&payload[valid.offset..valid.offset + valid.bytes])
    {
        if mask > 1 || mask == 1 && (!value.is_finite() || value.abs() > 1e20) {
            return Err("observation mask or observed cells are invalid".into());
        }
    }
    if meta["provenance"]["is_stub"].as_bool() == Some(true) {
        return Err("stub grids cannot verify a forecast".into());
    }
    Ok(RadarData {
        grid: GridData {
            lat: lat.into_iter().map(|v| v as f32).collect(),
            lon: lon.into_iter().map(|v| v as f32).collect(),
            ny: shape[0],
            nx: shape[1],
            projection: None,
        },
        values,
        valid: payload[valid.offset..valid.offset + valid.bytes]
            .iter()
            .map(|&b| b != 0)
            .collect(),
        valid_time: meta["valid_time"].as_str().unwrap_or("").into(),
        quantity: spec.quantity.clone(),
        units: meta["units"].as_str().unwrap_or("").into(),
        provenance: meta["provenance"].clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn numpy_test_member(dtype: &str, shape: &str, payload: &[u8]) -> Vec<u8> {
        let header =
            format!("{{'descr': '{dtype}', 'fortran_order': False, 'shape': ({shape}), }}");
        let padding = (64 - (10 + header.len() + 1) % 64) % 64;
        let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
        bytes.extend_from_slice(&((header.len() + padding + 1) as u16).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend(std::iter::repeat_n(b' ', padding));
        bytes.push(b'\n');
        bytes.extend_from_slice(payload);
        bytes
    }
    fn numpy_test_archive(members: Vec<(&str, Vec<u8>)>) -> Vec<u8> {
        use std::io::Write;
        let mut archive = Vec::new();
        let mut directory = Vec::new();
        for (name, raw) in members {
            let mut crc = 0xffff_ffff_u32;
            for &byte in &raw {
                crc ^= u32::from(byte);
                for _ in 0..8 {
                    crc = if crc & 1 == 1 {
                        0xedb8_8320 ^ (crc >> 1)
                    } else {
                        crc >> 1
                    };
                }
            }
            crc ^= 0xffff_ffff;
            let mut compressor =
                flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::fast());
            compressor.write_all(&raw).unwrap();
            let compressed = compressor.finish().unwrap();
            let offset = archive.len() as u32;
            archive.extend_from_slice(b"PK\x03\x04");
            archive.extend_from_slice(&20_u16.to_le_bytes());
            archive.extend_from_slice(&0_u16.to_le_bytes());
            archive.extend_from_slice(&8_u16.to_le_bytes());
            archive.extend_from_slice(&[0, 0, 33, 0]);
            archive.extend_from_slice(&crc.to_le_bytes());
            archive.extend_from_slice(&(compressed.len() as u32).to_le_bytes());
            archive.extend_from_slice(&(raw.len() as u32).to_le_bytes());
            archive.extend_from_slice(&(name.len() as u16).to_le_bytes());
            archive.extend_from_slice(&[0; 2]);
            archive.extend_from_slice(name.as_bytes());
            archive.extend_from_slice(&compressed);
            directory.push((name, crc, compressed.len() as u32, raw.len() as u32, offset));
        }
        let central = archive.len() as u32;
        for (name, crc, compressed, expanded, offset) in &directory {
            archive.extend_from_slice(b"PK\x01\x02");
            archive.extend_from_slice(&20_u16.to_le_bytes());
            archive.extend_from_slice(&20_u16.to_le_bytes());
            archive.extend_from_slice(&0_u16.to_le_bytes());
            archive.extend_from_slice(&8_u16.to_le_bytes());
            archive.extend_from_slice(&[0, 0, 33, 0]);
            archive.extend_from_slice(&crc.to_le_bytes());
            archive.extend_from_slice(&compressed.to_le_bytes());
            archive.extend_from_slice(&expanded.to_le_bytes());
            archive.extend_from_slice(&(name.len() as u16).to_le_bytes());
            archive.extend_from_slice(&[0; 12]);
            archive.extend_from_slice(&offset.to_le_bytes());
            archive.extend_from_slice(name.as_bytes());
        }
        let size = archive.len() as u32 - central;
        archive.extend_from_slice(b"PK\x05\x06");
        archive.extend_from_slice(&[0; 4]);
        archive.extend_from_slice(&(directory.len() as u16).to_le_bytes());
        archive.extend_from_slice(&(directory.len() as u16).to_le_bytes());
        archive.extend_from_slice(&size.to_le_bytes());
        archive.extend_from_slice(&central.to_le_bytes());
        archive.extend_from_slice(&[0; 2]);
        archive
    }
    #[test]
    fn packaged_unicode_metadata_does_not_replace_selected_float_fields() {
        let unicode = |text: &str, width: usize| {
            let mut bytes = Vec::new();
            for code in text
                .chars()
                .map(u32::from)
                .chain(std::iter::repeat(0))
                .take(width)
            {
                bytes.extend_from_slice(&code.to_le_bytes());
            }
            bytes
        };
        let path = std::env::temp_dir().join(format!(
            "native-npz-metadata-{}-{}.npz",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let values: [f32; 4] = [1., 2., 3., 4.];
        let numeric: Vec<u8> = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let bytes = numpy_test_archive(vec![
            ("refc_dbz.npy", numpy_test_member("<f4", "2, 2", &numeric)),
            (
                "valid.npy",
                numpy_test_member("<U19", "", &unicode("2030-01-01 12:00:00", 19)),
            ),
            (
                "source_file.npy",
                numpy_test_member("<U25", "", &unicode("reference-field-f10.grib2", 25)),
            ),
            (
                "crop.npy",
                numpy_test_member("<U33", "", &unicode("(slice(1,-1),slice(1,-1))", 33)),
            ),
            (
                "unused_integer.npy",
                numpy_test_member("<i8", "", &7_i64.to_le_bytes()),
            ),
            ("unused_boolean.npy", numpy_test_member("|b1", "", &[1])),
        ]);
        std::fs::write(&path, bytes).unwrap();
        let selected = ["refc_dbz".to_string()].into();
        let mut metadata = BTreeMap::new();
        let arrays = read_npz(&path, &selected, &mut metadata).unwrap();
        assert_eq!(arrays["refc_dbz"].values, vec![1., 2., 3., 4.]);
        assert_eq!(arrays.len(), 1);
        assert_eq!(metadata["valid"], "2030-01-01 12:00:00");
        let spec: ArmSpec =
            serde_json::from_value(serde_json::json!({"label":"A","kind":"npz","path":path}))
                .unwrap();
        let arm = ArmData {
            label: "A".into(),
            grid: None,
            fields: BTreeMap::new(),
            points: BTreeMap::new(),
            provenance: serde_json::json!({"metadata":{"valid_time":metadata["valid"]}}),
        };
        validate_arm_time(&spec, &arm, "2030-01-01T12:00:00Z").unwrap();
        assert!(validate_arm_time(&spec, &arm, "2030-01-01T13:00:00Z").is_err());
        assert_eq!(metadata["source_file"], "reference-field-f10.grib2");
        assert_eq!(metadata["crop"], "(slice(1,-1),slice(1,-1))");
        let invalid = ["valid".to_string()].into();
        let error = read_npz(&path, &invalid, &mut BTreeMap::new())
            .err()
            .unwrap();
        assert!(error.contains("selected member valid.npy"));
        assert!(error.contains("little-endian f32 or f64"));
        std::fs::remove_file(path).unwrap();
    }
    #[test]
    fn native_frame_retains_projection_time_and_one_kilometre_spacing() {
        use netcdf_writer::{AttrValue, NcFormat, NcType, NcWriter, Schema, VarData};
        let path = std::env::temp_dir().join(format!(
            "native-verification-frame-{}-{}.nc",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mut schema = Schema::new(NcFormat::Offset64);
        let time = schema.def_dim("Time", 0, true).unwrap();
        let date = schema.def_dim("DateStrLen", 19, false).unwrap();
        let ny = schema.def_dim("south_north", 2, false).unwrap();
        let nx = schema.def_dim("west_east", 2, false).unwrap();
        schema.def_dim("bottom_top", 1, false).unwrap();
        schema
            .put_global_attr("MAP_PROJ", AttrValue::Ints(vec![1]))
            .unwrap();
        for (key, value) in [
            ("TRUELAT1", 38.5),
            ("TRUELAT2", 38.5),
            ("STAND_LON", -97.5),
            ("DX", 1000.),
            ("DY", 1000.),
        ] {
            schema
                .put_global_attr(key, AttrValue::Doubles(vec![value]))
                .unwrap();
        }
        schema
            .put_global_attr(
                "SIMULATION_START_DATE",
                AttrValue::Text("2030-01-01_00:00:00".into()),
            )
            .unwrap();
        let times = schema
            .def_var("Times", NcType::Char, &[time, date])
            .unwrap();
        let lat = schema
            .def_var("XLAT", NcType::Float, &[time, ny, nx])
            .unwrap();
        let lon = schema
            .def_var("XLONG", NcType::Float, &[time, ny, nx])
            .unwrap();
        let t = schema
            .def_var("T2", NcType::Float, &[time, ny, nx])
            .unwrap();
        let td = schema
            .def_var("Td2", NcType::Float, &[time, ny, nx])
            .unwrap();
        let mut writer = NcWriter::create(&path, schema).unwrap();
        writer
            .write_record(0, times, VarData::Char(b"2030-01-01_01:00:00"))
            .unwrap();
        writer
            .write_record(0, lat, VarData::F32(&[30., 30., 30.009, 30.009]))
            .unwrap();
        writer
            .write_record(0, lon, VarData::F32(&[-100., -99.9896, -100., -99.9896]))
            .unwrap();
        writer.write_record(0, t, VarData::F32(&[300.; 4])).unwrap();
        writer
            .write_record(0, td, VarData::F32(&[290.; 4]))
            .unwrap();
        writer
            .write_record(1, times, VarData::Char(b"2030-01-01_02:00:00"))
            .unwrap();
        writer.write_record(1, lat, VarData::F32(&[30., 30., 30.009, 30.009])).unwrap();
        writer.write_record(1, lon, VarData::F32(&[-100., -99.9896, -100., -99.9896])).unwrap();
        writer.write_record(1, t, VarData::F32(&[301.; 4])).unwrap();
        writer.write_record(1, td, VarData::F32(&[291.; 4])).unwrap();
        writer.finish().unwrap();
        let spec:ArmSpec=serde_json::from_value(serde_json::json!({"label":"A","kind":"netcdf","path":path,"fields":{"temperature_2m":"T2","dewpoint_2m":"Td2"}})).unwrap();
        let arm = load_arm(&spec).unwrap();
        // The fixture is a surface-only frame: it carries no 3-D T.  Read
        // through the netCDF reader load_arm uses; whether the WRF reader
        // also refuses such a file depends on its backend, which is not what
        // this test is about (it failed on that alone at e6240d626).
        assert!(netcrust::open(&spec.path).unwrap()
                    .read_array_f64_first_record_or_all("T").is_err(),
                "the native verification frame remains a surface-only file without T");
        assert_eq!(arm_spacing(&arm), Some((1., 1.)));
        assert!(matches!(
            arm.grid.as_ref().unwrap().projection,
            Some(GridProjection::LambertConformal { .. })
        ));
        assert_eq!(arm.fields["temperature_2m"].iter().map(|v|v.to_bits()).collect::<Vec<_>>(),
                   vec![300.0_f64.to_bits();4]);
        assert_eq!(arm.fields["dewpoint_2m"].iter().map(|v|v.to_bits()).collect::<Vec<_>>(),
                   vec![290.0_f64.to_bits();4]);
        validate_arm_time(&spec, &arm, "2030-01-01T01:00:00").unwrap();
        assert!(validate_arm_time(&spec, &arm, "2030-01-01T02:00:00").is_err());
        let mut next = spec.clone();
        next.time_index = 1;
        let second = load_arm(&next).unwrap();
        validate_arm_time(&next, &second, "2030-01-01T02:00:00").unwrap();
        assert!(validate_arm_time(&next, &second, "2030-01-01T01:00:00").is_err());
        assert_eq!(second.fields["temperature_2m"], vec![301.; 4]);
        assert_eq!(second.fields["dewpoint_2m"], vec![291.; 4]);
        std::fs::remove_file(path).unwrap();
    }
    #[test]
    fn verification_reads_full_netcdf4_char_and_string_timestamps() {
        for name in ["times_char_netcdf4_cut.nc", "times_string_netcdf4_cut.nc"] {
            let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("tests/time_axis_fixtures").join(name);
            let spec: ArmSpec = serde_json::from_value(serde_json::json!({
                "label":"forecast", "kind":"netcdf", "path":path,
                "fields":{"temperature_2m":"T2"}
            })).unwrap();
            let arm = load_arm(&spec).unwrap();
            validate_arm_time(&spec, &arm, "2025-03-16T00:00:00Z").unwrap();
            assert!(validate_arm_time(&spec, &arm, "2025-03-15T12:00:00Z").is_err());
            assert_eq!(arm.fields["temperature_2m"], vec![291., 291., 292., 293.]);
            // The previous-frame path must use the same complete time axis,
            // rather than accepting the first character as a timestamp.
            let mut same_frame = spec.clone();
            same_frame.previous_path = Some(spec.path.clone());
            let error = match load_arm(&same_frame) {
                Ok(_) => panic!("same-time previous frame must be refused"),
                Err(error) => error,
            };
            assert!(error.contains("not exactly one hour apart"), "{error}");
        }
    }

    #[test]
    fn spacing_is_measured_from_native_one_kilometre_geometry() {
        let unit = (1.0 / 111.195) as f32;
        let grid = GridData {
            lat: vec![0., 0., unit, unit],
            lon: vec![0., unit, 0., unit],
            nx: 2,
            ny: 2,
            projection: None,
        };
        let (dx, dy) = grid.spacing_km().unwrap();
        assert!((dx - 1.0).abs() < 1e-5);
        assert!((dy - 1.0).abs() < 1e-5);
    }
    #[test]
    fn snapshot_roundtrips_native_values_and_geometry() {
        let grid = GridData {
            lat: vec![30., 30., 31., 31.],
            lon: vec![-100., -99., -100., -99.],
            ny: 2,
            nx: 2,
            projection: None,
        };
        let arm = ArmData {
            label: "A".into(),
            grid: Some(grid),
            fields: [("temperature_2m".into(), vec![290.125, 291.5, 292.25, 293.])].into(),
            points: BTreeMap::new(),
            provenance: Value::Null,
        };
        let path = std::env::temp_dir().join(format!(
            "native-verification-{}-{}.npz",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        write_snapshot(&path, &arm).unwrap();
        let selected = arm
            .fields
            .keys()
            .cloned()
            .chain(["lat".into(), "lon".into()])
            .collect();
        let arrays = read_npz(&path, &selected, &mut BTreeMap::new()).unwrap();
        assert_eq!(
            arrays["temperature_2m"].values,
            arm.fields["temperature_2m"]
        );
        assert_eq!(
            grid_from_arrays(&arrays).unwrap().lat,
            arm.grid.as_ref().unwrap().lat
        );
        std::fs::remove_file(path).unwrap();
    }
    #[test]
    fn source_hour_binding_rejects_a_correct_point_stamp_on_wrong_grid_hour() {
        let spec:ArmSpec=serde_json::from_value(serde_json::json!({"label":"A","kind":"npz","path":"forecast-f02.npz","init_time":"2030-01-01T00:00:00","hour":1})).unwrap();
        let arm = ArmData {
            label: "A".into(),
            grid: None,
            fields: BTreeMap::new(),
            points: BTreeMap::new(),
            provenance: serde_json::json!({"point_extract":{"valid_time":"2030-01-01T01:00:00"}}),
        };
        assert!(validate_arm_time(&spec, &arm, "2030-01-01T01:00:00")
            .unwrap_err()
            .contains("filename hour"));
    }
    #[test]
    fn bilinear_samples_native_cells_and_rejects_outside() {
        let g = GridData {
            lat: vec![30., 30., 31., 31.],
            lon: vec![-100., -99., -100., -99.],
            ny: 2,
            nx: 2,
            projection: None,
        };
        let p = g.position(30.25, -99.5).unwrap();
        assert!((g.sample(&[0., 2., 4., 6.], p).unwrap() - 2.).abs() < 1e-9);
        assert!(g.position(29.9, -99.5).is_none());
    }
    #[test]
    fn seeded_containment_checks_the_curved_lattice_instead_of_its_bbox() {
        let grid = GridData {
            lat: vec![30., 30.2, 31., 31.2],
            lon: vec![-100., -99., -100., -99.],
            ny: 2,
            nx: 2,
            projection: None,
        };
        assert!(grid.position_from_nearest(30.05, -99.5, 0).is_none());
        let inside = grid.position_from_nearest(30.5, -99.5, 0).unwrap();
        assert!((inside.0 - 0.5).abs() < 1e-6);
        assert!((inside.1 - 0.4).abs() < 1e-5);
    }
    #[test]
    fn nonfinite_nonzero_weight_is_missing() {
        let g = GridData {
            lat: vec![0.; 4],
            lon: vec![0.; 4],
            ny: 2,
            nx: 2,
            projection: None,
        };
        assert_eq!(g.sample(&[1., f64::NAN, 3., 4.], (0., 0.)), Some(1.));
        assert!(g.sample(&[1., f64::NAN, 3., 4.], (0.5, 0.5)).is_none());
    }
}

#[cfg(test)]
mod derived_heat_observation_tests {
    #[test]
    fn decoded_asos_rows_supply_heat_quantities_without_inventing_missing_wind() {
        let path = std::env::temp_dir().join(format!("asos-heat-{}-{}.json", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
        let record = serde_json::json!({
            "schema":"gpuwm-obs.asos-surface.v2", "provenance":{"is_stub":false},
            "stations":[{"station_id":"SCREEN","latitude":40.,"longitude":-100.,"elevation_m":300.}],
            "reports":[{"station_id":"SCREEN","valid_time":"2026-08-19T00:00:00","observation_time":"2026-08-18T23:53:00", "values":{"temperature_2m":308.15,"dewpoint_2m":303.15}}]
        });
        std::fs::write(&path,serde_json::to_vec(&record).unwrap()).unwrap();
        let rows=super::load_station_observations(&path,"2026-08-19T00:00:00").unwrap();
        std::fs::remove_file(path).unwrap();
        assert_eq!(rows.len(),1);
        assert!(rows[0].values["heat_index_2m"]>45.0);
        assert!(!rows[0].values.contains_key("apparent_temperature_2m"));
        assert!(!rows[0].values.contains_key("wbgt_2m"));
        assert_eq!(rows[0].observation_time,"2026-08-18T23:53:00");
    }
}
