//! Geostationary imagers other than GOES ABI, decoded to the same shape the
//! ABI path uses: one band of one scan on its fixed grid (scan angles plus
//! the projection), with radiance, a per-pixel quality flag, brightness
//! temperature and the measurement time of every row.
//!
//! The decoders live beside this module (`himawari` for JMA's Himawari
//! Standard Data, `ami` for KMA's GK-2A AMI Level 1B NetCDF); the sources
//! they are fetched from are a table (`geo_imagers.json`), so adding a
//! satellite on an existing instrument is a table row, not code.

use std::error::Error;
use std::path::{Path, PathBuf};

use chrono::{DateTime, Datelike, Duration, TimeZone, Timelike, Utc};
use serde::{Deserialize, Serialize};

use crate::abi::{AbiFixedGrid, GoesImagerProjection};
use crate::geostationary::lat_lon_to_scan_angles_fast;
use crate::s3::{DownloadedObject, S3Object, download_object, list_s3_objects, object_filename};

/// Per-pixel quality, the ABI L1b DQF enumeration: 0 good, 1 conditionally
/// usable, 2 outside the scan / off the Earth, 3 error.  Only 0 carries a
/// brightness temperature.
pub const DQF_GOOD: f32 = 0.0;
pub const DQF_CONDITIONAL: f32 = 1.0;
pub const DQF_OUTSIDE: f32 = 2.0;
pub const DQF_ERROR: f32 = 3.0;

/// One band of one scan, decoded.
#[derive(Debug, Clone)]
pub struct GeoImagerScan {
    /// `ahi`, `ami` (lower case, the CRTM instrument prefix).
    pub instrument: String,
    /// Platform as the producer names it in file names (`H09`, `GK2A`).
    pub satellite: String,
    /// `F` full disk (the ABI sector token).
    pub sector: String,
    /// Instrument band number (AHI B01..B16, AMI channel 1..16 in wavelength order).
    pub band: u8,
    /// The producer's own channel label (`B13`, `IR105`).
    pub channel_name: String,
    pub central_wavelength_um: f64,
    pub start_time_utc: DateTime<Utc>,
    pub end_time_utc: DateTime<Utc>,
    pub projection: GoesImagerProjection,
    pub fixed_grid: AbiFixedGrid,
    /// Measurement time of each row (POSIX seconds), `fixed_grid.ny` values.
    pub row_time_unix_s: Vec<f64>,
    /// Radiance as the producer calibrates it, in `radiance_units`.
    pub radiance: Vec<f32>,
    pub radiance_units: String,
    pub dqf: Vec<f32>,
    pub bt: Vec<f32>,
    /// The calibration constants read from the file and the formula applied.
    pub calibration: serde_json::Value,
    /// `[x_start, x_count, y_start, y_count]` inside the full grid, when cropped.
    pub window: Option<[usize; 4]>,
    pub full_nx: usize,
    pub full_ny: usize,
    /// Source files (segments for AHI).
    pub sources: Vec<PathBuf>,
}

impl GeoImagerScan {
    pub fn lat_lon_mesh(&self) -> (Vec<f32>, Vec<f32>) {
        let n = self.fixed_grid.nx * self.fixed_grid.ny;
        let mut lat = Vec::with_capacity(n);
        let mut lon = Vec::with_capacity(n);
        for &y in &self.fixed_grid.y_scan_rad {
            for &x in &self.fixed_grid.x_scan_rad {
                match self.projection.scan_angles_to_lat_lon(x, y) {
                    Some((a, o)) => {
                        lat.push(a);
                        lon.push(o);
                    }
                    None => {
                        lat.push(f32::NAN);
                        lon.push(f32::NAN);
                    }
                }
            }
        }
        (lat, lon)
    }

    /// Crop to `[x_start, x_count, y_start, y_count]` of the current grid.
    pub fn crop(mut self, window: [usize; 4]) -> Result<Self, Box<dyn Error>> {
        let [xs, xc, ys, yc] = window;
        let (nx, ny) = (self.fixed_grid.nx, self.fixed_grid.ny);
        if xc == 0 || yc == 0 || xs + xc > nx || ys + yc > ny {
            return Err(format!("window {window:?} is not inside the {nx}x{ny} grid").into());
        }
        let take = |v: &[f32]| -> Vec<f32> {
            let mut out = Vec::with_capacity(xc * yc);
            for r in ys..ys + yc {
                out.extend_from_slice(&v[r * nx + xs..r * nx + xs + xc]);
            }
            out
        };
        self.radiance = take(&self.radiance);
        self.dqf = take(&self.dqf);
        self.bt = take(&self.bt);
        self.fixed_grid = AbiFixedGrid {
            nx: xc,
            ny: yc,
            x_scan_rad: self.fixed_grid.x_scan_rad[xs..xs + xc].to_vec(),
            y_scan_rad: self.fixed_grid.y_scan_rad[ys..ys + yc].to_vec(),
        };
        self.row_time_unix_s = self.row_time_unix_s[ys..ys + yc].to_vec();
        let (ox, oy) = self.window.map(|w| (w[0], w[2])).unwrap_or((0, 0));
        self.window = Some([ox + xs, xc, oy + ys, yc]);
        Ok(self)
    }

    /// The smallest window holding every on-disk pixel inside the
    /// latitude/longitude box `[south, north, west, east]` (degrees; west > east
    /// crosses the antimeridian).  `None` when the box is not in view.
    pub fn window_for_bbox(&self, bbox: [f64; 4]) -> Option<[usize; 4]> {
        window_for_bbox(&self.projection, &self.fixed_grid, bbox)
    }
}

/// See [`GeoImagerScan::window_for_bbox`]; works on any fixed grid.
pub fn window_for_bbox(
    projection: &GoesImagerProjection,
    grid: &AbiFixedGrid,
    bbox: [f64; 4],
) -> Option<[usize; 4]> {
    let [south, north, west, east] = bbox;
    let inside_lon = |lon: f64| {
        if west <= east {
            lon >= west && lon <= east
        } else {
            lon >= west || lon <= east
        }
    };
    // Project a dense ring of box-boundary points plus an interior lattice to
    // scan angles; the window is the index span of those angles.
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    let span = if west <= east { east - west } else { east + 360.0 - west };
    let steps = 64;
    for i in 0..=steps {
        for j in 0..=steps {
            let lat = south + (north - south) * i as f64 / steps as f64;
            let mut lon = west + span * j as f64 / steps as f64;
            if lon > 180.0 {
                lon -= 360.0;
            }
            if !inside_lon(lon) && span < 360.0 {
                continue;
            }
            if let Some((x, y)) = lat_lon_to_scan_angles_fast(
                projection.perspective_point_height_m,
                projection.semi_major_axis_m,
                projection.semi_minor_axis_m,
                projection.longitude_of_projection_origin_deg,
                projection.sweep_angle_axis,
                lat,
                lon,
            ) {
                // the fast inverse does not test visibility: keep only points whose
                // forward projection lands back on them
                if let Some((la, lo)) = projection.scan_angles_to_lat_lon(x, y) {
                    let dlon = ((f64::from(lo) - lon + 540.0) % 360.0 - 180.0).abs();
                    if (f64::from(la) - lat).abs() < 1.0e-2 && dlon < 1.0e-2 {
                        xs.push(x);
                        ys.push(y);
                    }
                }
            }
        }
    }
    if xs.is_empty() {
        return None;
    }
    let index_span = |axis: &[f64], lo: f64, hi: f64| -> Option<(usize, usize)> {
        let hits: Vec<usize> = axis
            .iter()
            .enumerate()
            .filter(|(_, a)| **a >= lo && **a <= hi)
            .map(|(i, _)| i)
            .collect();
        Some((*hits.first()?, *hits.last()?))
    };
    let (xmin, xmax) = xs.iter().fold((f64::MAX, f64::MIN), |(a, b), v| (a.min(*v), b.max(*v)));
    let (ymin, ymax) = ys.iter().fold((f64::MAX, f64::MIN), |(a, b), v| (a.min(*v), b.max(*v)));
    let (x0, x1) = index_span(&grid.x_scan_rad, xmin, xmax)?;
    let (y0, y1) = index_span(&grid.y_scan_rad, ymin, ymax)?;
    Some([x0, x1 - x0 + 1, y0, y1 - y0 + 1])
}

/// Brightness temperature from a spectral radiance per wavelength
/// (W m-2 sr-1 um-1) through the monochromatic inverse Planck function at the
/// band's central wavelength, then the producer's effective-to-brightness
/// quadratic `c0 + c1 T + c2 T^2`.  NaN where the relation has no answer.
pub fn bt_from_radiance_per_um(
    radiance_per_um: f64,
    wavelength_um: f64,
    c: [f64; 3],
    planck_h: f64,
    light_c: f64,
    boltzmann_k: f64,
) -> f64 {
    if !(radiance_per_um.is_finite() && radiance_per_um > 0.0) {
        return f64::NAN;
    }
    let lam = wavelength_um * 1.0e-6;
    let arg = 2.0 * planck_h * light_c * light_c / (radiance_per_um * 1.0e6 * lam.powi(5)) + 1.0;
    if !(arg.is_finite() && arg > 1.0) {
        return f64::NAN;
    }
    let te = planck_h * light_c / boltzmann_k / (lam * arg.ln());
    let tb = c[0] + c[1] * te + c[2] * te * te;
    if tb.is_finite() { tb } else { f64::NAN }
}

/// Brightness temperature from a spectral radiance per wavenumber
/// (mW m-2 sr-1 (cm-1)-1) at the band's central wavenumber (from its central
/// wavelength), then the producer's quadratic.
pub fn bt_from_radiance_per_wavenumber(
    radiance_mw_per_cm1: f64,
    wavelength_um: f64,
    c: [f64; 3],
    planck_h: f64,
    light_c: f64,
    boltzmann_k: f64,
) -> f64 {
    if !(radiance_mw_per_cm1.is_finite() && radiance_mw_per_cm1 > 0.0) {
        return f64::NAN;
    }
    let nu = 1.0 / (wavelength_um * 1.0e-6); // m-1
    let rad = radiance_mw_per_cm1 * 1.0e-5; // W m-2 sr-1 (m-1)-1
    let arg = 2.0 * planck_h * light_c * light_c * nu.powi(3) / rad + 1.0;
    if !(arg.is_finite() && arg > 1.0) {
        return f64::NAN;
    }
    let te = planck_h * light_c * nu / boltzmann_k / arg.ln();
    let tb = c[0] + c[1] * te + c[2] * te * te;
    if tb.is_finite() { tb } else { f64::NAN }
}

// ---------------------------------------------------------------------------
// the sources table
// ---------------------------------------------------------------------------

pub const SOURCES_SCHEMA: &str = "gpuwm-obs.geo-imager-sources.v1";
const SOURCES_JSON: &str = include_str!("geo_imagers.json");

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeoSourcesTable {
    pub schema: String,
    pub sources: Vec<GeoSource>,
}

/// One (satellite, sector) feed in an anonymous open-data bucket.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeoSource {
    pub satellite: String,
    pub instrument: String,
    pub sector: String,
    pub bucket: String,
    /// Nominal scan starts are `offset_minutes + k * cadence_minutes` past the hour.
    pub cadence_minutes: i64,
    #[serde(default)]
    pub offset_minutes: i64,
    /// Listing prefix of one nominal scan: `{Y} {m} {d} {H} {M} {j}` are the
    /// scan's year, month, day, hour, minute and day of year.
    pub prefix: String,
    /// File-name glob of one band of one scan (`*` any run of characters):
    /// the time tokens above plus `{band}` (the `bands` token of the band).
    pub file: String,
    /// Files per band per scan. AHI instead validates the sequence and total
    /// declared in each HSD filename (historical whole disks use S0101).
    #[serde(default = "one")]
    pub files_per_band: usize,
    /// Band number -> file-name token.
    pub bands: std::collections::BTreeMap<String, String>,
    /// Optional same-scan companion products (a clear-sky mask), name -> {prefix, file}.
    #[serde(default)]
    pub companions: std::collections::BTreeMap<String, GeoCompanion>,
    /// First and last nominal scan the bucket holds, when bounded.
    #[serde(default)]
    pub first_scan: Option<String>,
    #[serde(default)]
    pub last_scan: Option<String>,
    #[serde(default)]
    pub note: Option<String>,
    /// Band number -> spectral-response centroid (cm-1), for producers whose
    /// file states only a rounded central wavelength (GK-2A AMI).
    #[serde(default)]
    pub band_centroid_wavenumber_cm1: std::collections::BTreeMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeoCompanion {
    pub prefix: String,
    pub file: String,
}

fn one() -> usize {
    1
}

pub fn sources_table() -> Result<GeoSourcesTable, Box<dyn Error>> {
    let table: GeoSourcesTable = serde_json::from_str(SOURCES_JSON)?;
    if table.schema != SOURCES_SCHEMA {
        return Err(format!("geo_imagers.json schema {} is not {SOURCES_SCHEMA}", table.schema).into());
    }
    Ok(table)
}

pub fn find_source(satellite: &str, sector: &str) -> Result<GeoSource, Box<dyn Error>> {
    let table = sources_table()?;
    let sat = satellite.trim().to_ascii_uppercase().replace('-', "");
    let sec = sector.trim().to_ascii_uppercase();
    table
        .sources
        .iter()
        .find(|s| s.satellite.to_ascii_uppercase().replace('-', "") == sat && s.sector.to_ascii_uppercase() == sec)
        .cloned()
        .ok_or_else(|| {
            let known: Vec<String> = table.sources.iter().map(|s| format!("{}/{}", s.satellite, s.sector)).collect();
            format!("no geostationary source {satellite}/{sector}; the table knows {}", known.join(", ")).into()
        })
}

/// Expand `{Y} {m} {d} {H} {M} {j} {band}` in a template.
pub fn expand(template: &str, t: DateTime<Utc>, band_token: Option<&str>) -> String {
    let mut s = template
        .replace("{Y}", &format!("{:04}", t.year()))
        .replace("{m}", &format!("{:02}", t.month()))
        .replace("{d}", &format!("{:02}", t.day()))
        .replace("{H}", &format!("{:02}", t.hour()))
        .replace("{M}", &format!("{:02}", t.minute()))
        .replace("{j}", &format!("{:03}", t.ordinal()));
    if let Some(b) = band_token {
        s = s.replace("{band}", b);
    }
    s
}

/// `*`-only glob match.
pub fn glob_match(pattern: &str, name: &str) -> bool {
    let parts: Vec<&str> = pattern.split('*').collect();
    if parts.len() == 1 {
        return pattern == name;
    }
    let mut rest = name;
    for (i, part) in parts.iter().enumerate() {
        if i == 0 {
            if !rest.starts_with(part) {
                return false;
            }
            rest = &rest[part.len()..];
        } else if i == parts.len() - 1 {
            return rest.ends_with(part);
        } else if let Some(pos) = rest.find(part) {
            rest = &rest[pos + part.len()..];
        } else {
            return false;
        }
    }
    true
}

/// Nominal scan start at or before `t`.
pub fn scan_at_or_before(source: &GeoSource, t: DateTime<Utc>) -> DateTime<Utc> {
    let hour = Utc
        .with_ymd_and_hms(t.year(), t.month(), t.day(), t.hour(), 0, 0)
        .single()
        .unwrap_or(t);
    let minutes = (t - hour).num_minutes() - source.offset_minutes;
    let k = minutes.div_euclid(source.cadence_minutes.max(1));
    hour + Duration::minutes(source.offset_minutes + k * source.cadence_minutes.max(1))
}

/// The files of one scan: per band, the listed objects matching the band glob.
#[derive(Debug, Clone)]
pub struct ScanListing {
    pub scan_nominal: String,
    pub bands: std::collections::BTreeMap<u8, Vec<S3Object>>,
    pub companions: std::collections::BTreeMap<String, Vec<S3Object>>,
    pub complete: bool,
}

fn complete_ahi_segments<'a>(names: impl IntoIterator<Item = &'a str>) -> bool {
    let mut expected = None;
    let mut sequences = std::collections::BTreeSet::new();
    for name in names {
        let Some((_, suffix)) = name.rsplit_once("_S") else { return false };
        let Some(segment) = suffix.strip_suffix(".DAT.bz2") else { return false };
        if segment.len() != 4 || !segment.bytes().all(|b| b.is_ascii_digit()) {
            return false;
        }
        let sequence = segment[..2].parse::<usize>().unwrap();
        let total = segment[2..].parse::<usize>().unwrap();
        if total == 0 || sequence == 0 || sequence > total || expected.is_some_and(|n| n != total)
            || !sequences.insert(sequence)
        {
            return false;
        }
        expected = Some(total);
    }
    expected.is_some_and(|n| sequences.len() == n)
}

pub fn list_scan(
    agent: &ureq::Agent,
    source: &GeoSource,
    scan: DateTime<Utc>,
    bands: &[u8],
) -> Result<ScanListing, Box<dyn Error>> {
    let mut cache: std::collections::BTreeMap<String, Vec<S3Object>> = Default::default();
    let mut listed = |prefix: String| -> Result<Vec<S3Object>, Box<dyn Error>> {
        if let Some(v) = cache.get(&prefix) {
            return Ok(v.clone());
        }
        let v = list_s3_objects(agent, &source.bucket, &prefix, None)?;
        cache.insert(prefix, v.clone());
        Ok(v)
    };
    let objects = listed(expand(&source.prefix, scan, None))?;
    let mut out = ScanListing {
        scan_nominal: scan.format("%Y-%m-%dT%H:%M:%SZ").to_string(),
        bands: Default::default(),
        companions: Default::default(),
        complete: true,
    };
    for &band in bands {
        let token = source
            .bands
            .get(&band.to_string())
            .ok_or_else(|| format!("{} {} has no band {band} in the table", source.satellite, source.instrument))?;
        let glob = expand(&source.file, scan, Some(token));
        let mut hits: Vec<S3Object> = objects
            .iter()
            .filter(|o| glob_match(&glob, object_filename(&o.key)))
            .cloned()
            .collect();
        hits.sort_by(|a, b| a.key.cmp(&b.key));
        // Two files for one ABI band and scan are reprocessings: keep the newest.
        if source.instrument != "ahi" && source.files_per_band == 1 && hits.len() > 1 {
            hits = vec![hits.pop().unwrap()];
        }
        let complete = if source.instrument == "ahi" {
            complete_ahi_segments(hits.iter().map(|o| object_filename(&o.key)))
        } else {
            hits.len() == source.files_per_band
        };
        if !complete {
            out.complete = false;
        }
        out.bands.insert(band, hits);
    }
    for (name, comp) in &source.companions {
        let objs = listed(expand(&comp.prefix, scan, None))?;
        let glob = expand(&comp.file, scan, None);
        let mut hits: Vec<S3Object> = objs
            .iter()
            .filter(|o| glob_match(&glob, object_filename(&o.key)))
            .cloned()
            .collect();
        hits.sort_by(|a, b| a.key.cmp(&b.key));
        if hits.len() > 1 {
            hits = vec![hits.pop().unwrap()];
        }
        out.companions.insert(name.clone(), hits);
    }
    Ok(out)
}

/// Download every listed object of a scan into the content cache.
pub fn fetch_scan(
    agent: &ureq::Agent,
    source: &GeoSource,
    listing: &ScanListing,
    cache_dir: &Path,
) -> Result<Vec<(String, DownloadedObject)>, Box<dyn Error>> {
    let mut jobs: Vec<(String, S3Object)> = Vec::new();
    for (band, objs) in &listing.bands {
        for o in objs {
            jobs.push((format!("band:{band}"), o.clone()));
        }
    }
    for (name, objs) in &listing.companions {
        for o in objs {
            jobs.push((format!("companion:{name}"), o.clone()));
        }
    }
    // Parallel downloads: the scan's files are independent (AHI is 10 segments per band).
    use rayon::prelude::*;
    let results: Vec<Result<(String, DownloadedObject), String>> = jobs
        .par_iter()
        .map(|(tag, o)| {
            download_object(agent, &source.bucket, cache_dir, o, true)
                .map(|d| (tag.clone(), d))
                .map_err(|e| format!("{}: {e}", o.key))
        })
        .collect();
    let mut out = Vec::with_capacity(results.len());
    for r in results {
        out.push(r.map_err(|e| -> Box<dyn Error> { e.into() })?);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ahi_whole_disk_and_segmented_archives_require_every_segment() {
        let historical = "HS_H08_20191006_0000_B13_FLDK_R20_S0101.DAT.bz2";
        assert!(complete_ahi_segments([historical]));
        let names: Vec<String> = (1..=10).map(|i|
            format!("HS_H09_20241007_0300_B13_FLDK_R20_S{i:02}10.DAT.bz2")
        ).collect();
        assert!(complete_ahi_segments(names.iter().map(String::as_str)));
        assert!(!complete_ahi_segments(names[..9].iter().map(String::as_str)));
        assert!(!complete_ahi_segments([historical, historical]));
        assert!(!complete_ahi_segments([historical, names[0].as_str()]));
        assert!(!complete_ahi_segments(["HS_H08_B13_FLDK_R20_S0001.DAT.bz2"]));
        assert!(!complete_ahi_segments(std::iter::empty()));
        let source = find_source("H08", "F").unwrap();
        let scan = Utc.with_ymd_and_hms(2019, 10, 6, 0, 0, 0).unwrap();
        assert!(glob_match(&expand(&source.file, scan, Some("B13")), historical));
    }

    #[test]
    fn the_table_parses_and_every_row_names_its_bands() {
        let t = sources_table().unwrap();
        assert!(t.sources.len() >= 4);
        for s in &t.sources {
            assert!(!s.bands.is_empty(), "{}", s.satellite);
            assert!(s.file.contains("{band}"), "{}", s.satellite);
        }
        assert_eq!(find_source("h09", "f").unwrap().bucket, "noaa-himawari9");
        let gk = find_source("GK-2A", "F").unwrap();
        assert_eq!(gk.bucket, "noaa-gk2a-pds");
        for b in gk.bands.keys() {
            assert!(gk.band_centroid_wavenumber_cm1.contains_key(b), "GK-2A band {b} has no centroid");
        }
        assert!(find_source("G18", "C").is_ok() && find_source("G18", "F").is_ok());
    }

    #[test]
    fn templates_expand_to_the_public_bucket_names() {
        let t = Utc.with_ymd_and_hms(2026, 10, 15, 0, 0, 0).unwrap();
        let h9 = find_source("H09", "F").unwrap();
        assert_eq!(expand(&h9.prefix, t, None), "AHI-L1b-FLDK/2026/10/15/0000/");
        let g = expand(&h9.file, t, Some("B13"));
        assert!(glob_match(&g, "HS_H09_20261015_0000_B13_FLDK_R20_S0410.DAT.bz2"));
        assert!(!glob_match(&g, "HS_H09_20261015_0000_B13_FLDK_R05_S0410.DAT.bz2"));
        let gk = find_source("GK2A", "F").unwrap();
        assert_eq!(expand(&gk.prefix, t, None), "AMI/L1B/FD/202610/15/00/");
        assert!(glob_match(&expand(&gk.file, t, Some("ir105")), "gk2a_ami_le1b_ir105_fd020ge_202610150000.nc"));
        let g18 = find_source("G18", "F").unwrap();
        assert_eq!(expand(&g18.prefix, t, None), "ABI-L1b-RadF/2026/288/00/");
        assert!(glob_match(
            &expand(&g18.file, t, Some("C13")),
            "OR_ABI-L1b-RadF-M6C13_G18_s20262880000206_e20262880009526_c20262880009557.nc"
        ));
        let g18c = find_source("G18", "C").unwrap();
        let t1 = scan_at_or_before(&g18c, Utc.with_ymd_and_hms(2026, 10, 15, 0, 4, 59).unwrap());
        assert_eq!(t1, Utc.with_ymd_and_hms(2026, 10, 15, 0, 1, 0).unwrap());
        assert!(glob_match(
            &expand(&g18c.file, t1, Some("C13")),
            "OR_ABI-L1b-RadC-M6C13_G18_s20262880001174_e20262880003560_c20262880004013.nc"
        ));
    }

    #[test]
    fn the_wavenumber_and_wavelength_planck_forms_agree() {
        // a 280 K blackbody at 10.5 um, computed both ways, inverts to 280 K both ways
        let (h, c, k): (f64, f64, f64) = (6.62606957e-34, 299792458.0, 1.3806488e-23);
        let lam: f64 = 10.5e-6;
        let b_lam = 2.0 * h * c * c / lam.powi(5) / ((h * c / (lam * k * 280.0)).exp() - 1.0); // W m-2 sr-1 m-1
        let per_um = b_lam * 1.0e-6;
        let nu = 1.0 / lam;
        let b_nu = 2.0 * h * c * c * nu.powi(3) / ((h * c * nu / (k * 280.0)).exp() - 1.0); // per m-1
        let mw_per_cm1 = b_nu * 1.0e5;
        let a = bt_from_radiance_per_um(per_um, 10.5, [0.0, 1.0, 0.0], h, c, k);
        let b = bt_from_radiance_per_wavenumber(mw_per_cm1, 10.5, [0.0, 1.0, 0.0], h, c, k);
        assert!((a - 280.0).abs() < 1.0e-9 && (b - 280.0).abs() < 1.0e-9, "{a} {b}");
    }
}
