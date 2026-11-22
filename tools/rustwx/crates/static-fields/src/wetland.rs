//! Static inputs of the Noah-MP wetland option (Zhang et al. 2022, WRR 58,
//! e2021WR030573; re-implemented from published descriptions, not bitwise
//! to upstream).
//!
//! Two per-cell maps, built once at prep on every core:
//!
//! * `FSATMX`, the maximum saturated fraction.  The paper takes it from the
//!   GIEMS-2 maximum inundation fraction (their Sec. 3.2); GIEMS has no open
//!   licence, so this takes the GLWD v2 combined wetland area
//!   (`GLWD_v2_0_area_pct`, Lehner et al. 2025, CC BY 4.0): the mean percent
//!   of the cell's footprint covered by any of the 33 wetland classes.
//! * `WCAP`, the wetland storage capacity in mm.  The paper's Eq. 14 sums the
//!   90 m DEM elevations below the cell-mean elevation; we take that as the
//!   cell-mean depth below the cell mean, `mean(max(Hbar - H_i, 0))`, from
//!   Copernicus GLO-90 (the paper used MERIT, which is CC BY-NC and never
//!   ships).
//!
//! The cell footprint is the square of side `dx` centred on the mass point,
//! as a latitude/longitude box.  Pixels count when their centre is inside.
use std::path::{Path, PathBuf};

use rayon::prelude::*;

use crate::error::Result;
use crate::raster::geotiff::TiffReader;

const METRES_PER_DEGREE: f64 = 111_195.0;

/// One geographic GeoTIFF read into memory with its pixel lattice.
struct Plane {
    west: f64,
    north: f64,
    dlon: f64,
    dlat: f64,
    nx: usize,
    ny: usize,
    values: Vec<f64>,
}

impl Plane {
    fn read(path: &Path, window: Option<[f64; 4]>) -> Result<Option<Plane>> {
        let mut reader = TiffReader::open(path)?;
        let t = reader.transform;
        let (dlon, dlat) = (t[0], -t[4]);
        let (west0, north0) = (t[2], t[5]);
        let (w, h) = (reader.width, reader.height);
        let (c0, r0, c1, r1) = match window {
            None => (0usize, 0usize, w, h),
            Some([west, south, east, north]) => {
                let c0 = ((west - west0) / dlon).floor().max(0.0) as usize;
                let c1 = (((east - west0) / dlon).ceil() as isize).clamp(0, w as isize) as usize;
                let r0 = ((north0 - north) / dlat).floor().max(0.0) as usize;
                let r1 = (((north0 - south) / dlat).ceil() as isize).clamp(0, h as isize) as usize;
                if c0 >= c1 || r0 >= r1 {
                    return Ok(None);
                }
                (c0, r0, c1, r1)
            }
        };
        let raw = reader.read_window_raw(c0, r0, c1 - c0, r1 - r0)?;
        let nodata = reader.nodata;
        let values = raw
            .into_iter()
            .map(|v| match nodata {
                Some(nd) if v == nd => f64::NAN,
                _ => v,
            })
            .collect();
        Ok(Some(Plane {
            west: west0 + c0 as f64 * dlon,
            north: north0 - r0 as f64 * dlat,
            dlon,
            dlat,
            nx: c1 - c0,
            ny: r1 - r0,
            values,
        }))
    }

    /// Valid pixel values whose centres fall in the box.
    fn each_in(&self, west: f64, south: f64, east: f64, north: f64, mut f: impl FnMut(f64)) {
        if east <= self.west || north <= self.north - self.ny as f64 * self.dlat {
            return;
        }
        let c0 = ((west - self.west) / self.dlon - 0.5).ceil().max(0.0) as usize;
        let c1 = ((east - self.west) / self.dlon - 0.5).floor();
        let r0 = ((self.north - north) / self.dlat - 0.5).ceil().max(0.0) as usize;
        let r1 = ((self.north - south) / self.dlat - 0.5).floor();
        if c1 < 0.0 || r1 < 0.0 {
            return;
        }
        let c1 = (c1 as usize).min(self.nx.saturating_sub(1));
        let r1 = (r1 as usize).min(self.ny.saturating_sub(1));
        for r in r0..=r1 {
            let row = &self.values[r * self.nx..(r + 1) * self.nx];
            for c in c0..=c1.min(row.len().saturating_sub(1)) {
                let v = row[c];
                if v.is_finite() {
                    f(v);
                }
            }
        }
    }
}

/// Footprint box `(west, south, east, north)` of a square cell of side `dx_m`.
pub fn footprint(lat: f64, lon: f64, dx_m: f64) -> [f64; 4] {
    let half_lat = 0.5 * dx_m / METRES_PER_DEGREE;
    let coslat = lat.to_radians().cos().max(0.01);
    let half_lon = half_lat / coslat;
    [lon - half_lon, lat - half_lat, lon + half_lon, lat + half_lat]
}

/// Paper Eq. 14 (cell-mean reading): mean depth below the mean, in mm.
pub fn depression_capacity_mm(heights: &[f64]) -> Option<f64> {
    if heights.is_empty() {
        return None;
    }
    let mean = heights.iter().sum::<f64>() / heights.len() as f64;
    let below = heights.iter().map(|h| (mean - h).max(0.0)).sum::<f64>();
    Some(1000.0 * below / heights.len() as f64)
}

pub struct WetlandStatic {
    pub fsatmx: Vec<f32>,
    pub wcap_mm: Vec<f32>,
    /// Fraction of cells with GLWD and DEM pixels inside their footprint.
    pub glwd_coverage: f64,
    pub dem_coverage: f64,
}

/// Build both maps for `lat.len()` mass points.
pub fn build(lat: &[f64], lon: &[f64], dx_m: f64, glwd: &Path, dem_tiles: &[PathBuf]) -> Result<WetlandStatic> {
    let n = lat.len();
    let boxes: Vec<[f64; 4]> = (0..n).map(|k| footprint(lat[k], lon[k], dx_m)).collect();
    let mut domain = [f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY];
    for b in &boxes {
        domain[0] = domain[0].min(b[0]);
        domain[1] = domain[1].min(b[1]);
        domain[2] = domain[2].max(b[2]);
        domain[3] = domain[3].max(b[3]);
    }
    let glwd_plane = Plane::read(glwd, Some(domain))?;
    // Every DEM tile on its own core.
    let dem: Vec<Plane> = dem_tiles
        .par_iter()
        .map(|p| Plane::read(p, None))
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .flatten()
        .collect();
    let rows: Vec<(f32, f32, bool, bool)> = boxes
        .par_iter()
        .map(|b| {
            let (mut sum, mut count) = (0.0f64, 0usize);
            if let Some(g) = &glwd_plane {
                g.each_in(b[0], b[1], b[2], b[3], |v| {
                    sum += v.min(100.0);
                    count += 1;
                });
            }
            let fsat = if count > 0 { (sum / count as f64 / 100.0) as f32 } else { 0.0 };
            let mut heights = Vec::new();
            for tile in &dem {
                tile.each_in(b[0], b[1], b[2], b[3], |v| heights.push(v));
            }
            let cap = depression_capacity_mm(&heights);
            (fsat, cap.unwrap_or(0.0) as f32, count > 0, cap.is_some())
        })
        .collect();
    let glwd_coverage = rows.iter().filter(|r| r.2).count() as f64 / n.max(1) as f64;
    let dem_coverage = rows.iter().filter(|r| r.3).count() as f64 / n.max(1) as f64;
    Ok(WetlandStatic {
        fsatmx: rows.iter().map(|r| r.0).collect(),
        wcap_mm: rows.iter().map(|r| r.1).collect(),
        glwd_coverage,
        dem_coverage,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capacity_is_mean_depth_below_mean() {
        // Heights 10, 10, 9, 7 m: mean 9; depths below mean 0, 0, 0, 2 m;
        // mean depth 0.5 m = 500 mm.
        let cap = depression_capacity_mm(&[10.0, 10.0, 9.0, 7.0]).unwrap();
        assert!((cap - 500.0).abs() < 1e-9);
        assert!(depression_capacity_mm(&[]).is_none());
        assert_eq!(depression_capacity_mm(&[5.0; 9]).unwrap(), 0.0);
    }

    #[test]
    fn footprint_widens_with_latitude() {
        let b = footprint(60.0, -100.0, 3000.0);
        let half_lat = 1500.0 / METRES_PER_DEGREE;
        assert!((b[3] - 60.0 - half_lat).abs() < 1e-12);
        assert!(((b[2] - (-100.0)) - 2.0 * half_lat).abs() < 1e-9);
    }

    #[test]
    fn plane_box_counts_pixel_centres() {
        let p = Plane { west: 0.0, north: 2.0, dlon: 1.0, dlat: 1.0, nx: 2, ny: 2,
                        values: vec![1.0, 2.0, 3.0, f64::NAN] };
        let mut seen = Vec::new();
        p.each_in(0.0, 0.0, 2.0, 2.0, |v| seen.push(v));
        assert_eq!(seen, vec![1.0, 2.0, 3.0]);
        let mut one = Vec::new();
        p.each_in(0.0, 1.0, 1.0, 2.0, |v| one.push(v));
        assert_eq!(one, vec![1.0]);
    }
}
