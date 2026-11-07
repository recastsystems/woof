//! Point columns from MPAS history: what soundings and meteograms draw.
//!
//! A point's value is the same linear interpolant the product remap draws
//! maps with ([`crate::remap::build_linear`]): barycentric in the triangle of
//! the three cell centres around the point, so a sounding agrees with the map
//! under it.  Nothing here is drawn; the renderer (`rw_hex_points`) draws.

use crate::error::{MpasError, MpasResult};
use crate::history::MpasFrame;
use crate::remap::{LinearWeights, MeshTopology, build_linear};
use crate::weights::{KdTree, MeshCoordinates, unit_vector};
use crate::window::MPAS_EARTH_RADIUS_M;

/// MPAS-A's own constants (`mpas_constants.F`): `rgas`, `cp = 7 rgas / 2`.
pub const RGAS: f64 = 287.0;
pub const CP: f64 = 7.0 * RGAS / 2.0;
pub const GRAVITY: f64 = 9.80616;

#[derive(Debug, Clone, PartialEq)]
pub struct PointSpec {
    pub name: String,
    pub lat: f64,
    pub lon: f64,
}

impl PointSpec {
    /// A file-safe token: lowercase ASCII letters and digits, `-` elsewhere.
    pub fn slug(&self) -> String {
        let mut out = String::new();
        for ch in self.name.chars() {
            if ch.is_ascii_alphanumeric() {
                out.push(ch.to_ascii_lowercase());
            } else if !out.ends_with('-') && !out.is_empty() {
                out.push('-');
            }
        }
        while out.ends_with('-') {
            out.pop();
        }
        if out.is_empty() {
            format!("p{:+.2}{:+.2}", self.lat, self.lon)
        } else {
            out
        }
    }
}

/// `Name@lat,lon;Name@lat,lon`.  Latitude north positive, longitude east
/// positive, degrees.
pub fn parse_points(text: &str) -> MpasResult<Vec<PointSpec>> {
    let mut out = Vec::new();
    for item in text.split(';').map(str::trim).filter(|s| !s.is_empty()) {
        let (name, coords) = item.split_once('@').ok_or_else(|| {
            MpasError::Refusal(format!("point '{item}' is not NAME@LAT,LON"))
        })?;
        let (lat, lon) = coords.split_once(',').ok_or_else(|| {
            MpasError::Refusal(format!("point '{item}' is not NAME@LAT,LON"))
        })?;
        let lat: f64 = lat.trim().parse().map_err(|_| {
            MpasError::Refusal(format!("point '{item}': latitude '{lat}' is not a number"))
        })?;
        let lon: f64 = lon.trim().parse().map_err(|_| {
            MpasError::Refusal(format!("point '{item}': longitude '{lon}' is not a number"))
        })?;
        if !(-90.0..=90.0).contains(&lat) || !lon.is_finite() {
            return Err(MpasError::Refusal(format!(
                "point '{item}' is off the sphere (latitude {lat})"
            )));
        }
        let lon = (lon + 180.0).rem_euclid(360.0) - 180.0;
        out.push(PointSpec {
            name: name.trim().to_string(),
            lat,
            lon,
        });
    }
    if out.is_empty() {
        return Err(MpasError::Refusal("no points were named".to_string()));
    }
    Ok(out)
}

/// The interpolation for a fixed set of points on one mesh.
pub struct PointSampler {
    pub points: Vec<PointSpec>,
    pub linear: LinearWeights,
    pub nearest_cell: Vec<u32>,
    pub nearest_km: Vec<f64>,
    /// The point is further from its nearest centre than that cell's own
    /// spacing: it is not on this (limited-area) mesh.
    pub off_mesh: Vec<bool>,
    /// Spacing of the nearest cell, metres.
    pub local_spacing_m: Vec<f64>,
}

impl PointSampler {
    pub fn build(
        mesh: &MeshCoordinates,
        topology: &MeshTopology,
        spacing_m: &[f64],
        points: Vec<PointSpec>,
    ) -> MpasResult<PointSampler> {
        if topology.n_cells != mesh.n_cells() || spacing_m.len() != mesh.n_cells() {
            return Err(MpasError::Refusal(
                "point sampler: topology, spacing and centres disagree on the cell count"
                    .to_string(),
            ));
        }
        let centres: Vec<[f64; 3]> = (0..mesh.n_cells())
            .map(|i| unit_vector(mesh.latitude_degrees[i], mesh.longitude_degrees[i]))
            .collect();
        let tree = KdTree::build(centres.clone());
        let targets: Vec<[f64; 3]> = points.iter().map(|p| unit_vector(p.lat, p.lon)).collect();
        let mut nearest_cell = Vec::with_capacity(points.len());
        let mut nearest_km = Vec::with_capacity(points.len());
        for t in &targets {
            let (idx, chord2) = tree.nearest(*t);
            nearest_cell.push(idx);
            let chord = chord2.sqrt();
            nearest_km.push(2.0 * (chord / 2.0).clamp(0.0, 1.0).asin() * MPAS_EARTH_RADIUS_M / 1000.0);
        }
        let near_i32: Vec<i32> = nearest_cell.iter().map(|&c| c as i32).collect();
        let linear = build_linear(&centres, topology, &near_i32, &targets);
        let local_spacing_m: Vec<f64> = nearest_cell.iter().map(|&c| spacing_m[c as usize]).collect();
        let off_mesh = nearest_km
            .iter()
            .zip(local_spacing_m.iter())
            .map(|(km, s)| *km > s / 1000.0)
            .collect();
        Ok(PointSampler {
            points,
            linear,
            nearest_cell,
            nearest_km,
            off_mesh,
            local_spacing_m,
        })
    }

    /// The distinct mesh cells this sampler's interpolant reads (ascending),
    /// and a copy of the sampler whose weights index that list instead of the
    /// mesh.  Read those cells with [`crate::history::read_history_cells`]
    /// and [`columns`] on the copy gives exactly the values the whole-mesh
    /// frame gives on `self`: the gather multiplies the same weights by the
    /// same stored values in the same order.  `nearest_cell` keeps its mesh
    /// numbering (it names a cell; nothing gathers through it).
    pub fn compacted(&self) -> (Vec<u32>, PointSampler) {
        let mut cells: Vec<u32> = self.linear.cells.iter().flatten().copied().collect();
        cells.sort_unstable();
        cells.dedup();
        let slot = |c: u32| cells.binary_search(&c).expect("cell is in the list built from it") as u32;
        let linear = LinearWeights {
            cells: self.linear.cells.iter().map(|t| [slot(t[0]), slot(t[1]), slot(t[2])]).collect(),
            weights: self.linear.weights.clone(),
            contained: self.linear.contained,
            widened: self.linear.widened,
            clamped: self.linear.clamped,
        };
        let compact = PointSampler {
            points: self.points.clone(),
            linear,
            nearest_cell: self.nearest_cell.clone(),
            nearest_km: self.nearest_km.clone(),
            off_mesh: self.off_mesh.clone(),
            local_spacing_m: self.local_spacing_m.clone(),
        };
        (cells, compact)
    }

    /// One field at every point: `out[point][level]`.
    pub fn field(&self, frame: &MpasFrame, wrf_name: &str) -> Option<Vec<Vec<f64>>> {
        let (values, levels) = frame.fields.get(wrf_name)?;
        // MPAS initializes unavailable diagnostics (notably F000 U10/V10)
        // to -1e30. NetCDF's default float fill is positive ~1e37. Mask
        // before interpolation so neither sentinel becomes a finite wind.
        let flat = if values.iter().any(|v| !v.is_finite() || v.abs() >= 1e29) {
            let clean: Vec<f32> = values.iter().map(|&v| {
                if v.is_finite() && v.abs() < 1e29 { v } else { f32::NAN }
            }).collect();
            self.linear.gather(&clean, *levels)
        } else {
            self.linear.gather(values, *levels)
        };
        let n = self.points.len();
        Some(
            (0..n)
                .map(|p| (0..*levels).map(|k| f64::from(flat[k * n + p])).collect())
                .collect(),
        )
    }
}

/// One column, bottom to top, plus the surface.
#[derive(Debug, Clone)]
pub struct Column {
    pub pressure_pa: Vec<f64>,
    pub temperature_k: Vec<f64>,
    /// Water vapour mixing ratio, kg/kg.
    pub qv: Vec<f64>,
    /// Mass-level height above sea level, m.
    pub height_m: Vec<f64>,
    pub u: Vec<f64>,
    pub v: Vec<f64>,
    pub terrain_m: f64,
    pub psfc_pa: f64,
    pub t2_k: f64,
    pub q2: f64,
    pub u10: f64,
    pub v10: f64,
    /// RAINC + RAINNC, mm since the start.
    pub precip_mm: f64,
    pub refl_max_dbz: Option<f64>,
    /// Vertical pressure velocity on the mass levels, Pa/s, when the frame carries `W`: the column store's formula
    /// (rw-soundings `input.rs`), so a box sounding's omega panel shows what the store's shows.
    pub omega_pa_s: Option<Vec<f64>>,
}

impl Column {
    /// Sea-level pressure by the NCEP Shuell reduction the maps use
    /// ([`crate::mslp`]): the lowest model level carried to the ground, a
    /// 6.5 K/km fictitious column, its sea-level temperature adjusted only
    /// when it exceeds 290.66 K.  Until 2026-10-14 this reduced from the 2 m
    /// temperature with no warm-column adjustment, which read about 6 hPa low
    /// over warm high terrain (Tibet, the Andes) against the GFS analysis;
    /// a meteogram and the map beside it now say the same pressure.
    pub fn mslp_pa(&self) -> f64 {
        match (self.temperature_k.first(), self.qv.first(), self.height_m.first()) {
            (Some(&t), Some(&q), Some(&z)) => {
                crate::mslp::shuell_mslp_pa(self.psfc_pa, self.terrain_m, t, q, z)
            }
            _ => {
                // A column with no levels: the 2 m air is the lowest level.
                crate::mslp::shuell_mslp_pa(self.psfc_pa, self.terrain_m, self.t2_k, self.q2, self.terrain_m)
            }
        }
    }

    /// Precipitable water, mm (= kg/m2), from the surface to the top level.
    pub fn precipitable_water_mm(&self) -> f64 {
        let mut pw = 0.0;
        for k in 0..self.pressure_pa.len().saturating_sub(1) {
            let dp = self.pressure_pa[k] - self.pressure_pa[k + 1];
            let q = 0.5 * (self.qv[k] + self.qv[k + 1]);
            pw += q / (1.0 + q) * dp / GRAVITY;
        }
        pw
    }
}

/// Every point's column from one frame.  `None` when the frame lacks a field
/// a column needs (named in the error).
pub fn columns(sampler: &PointSampler, frame: &MpasFrame) -> MpasResult<Vec<Column>> {
    let need = |name: &str| -> MpasResult<Vec<Vec<f64>>> {
        sampler.field(frame, name).ok_or_else(|| {
            MpasError::Refusal(format!(
                "history {} carries no {name}, which a sounding needs",
                frame.history_path.display()
            ))
        })
    };
    let theta = need("T")?;
    let pressure = need("PB")?;
    let qv = need("QVAPOR")?;
    let u = need("U")?;
    let v = need("V")?;
    let zgrid = need("PHB").map_err(|e| {
        MpasError::Refusal(format!("{e} (zgrid comes from the init file: pass --init)"))
    })?;
    let surface = |name: &str| -> MpasResult<Vec<f64>> {
        Ok(need(name)?.into_iter().map(|c| c[0]).collect())
    };
    let t2 = surface("T2")?;
    let q2 = surface("Q2")?;
    let psfc = surface("PSFC")?;
    let u10 = surface("U10")?;
    let v10 = surface("V10")?;
    let rainc = sampler.field(frame, "RAINC").map(|f| f.into_iter().map(|c| c[0]).collect::<Vec<_>>());
    let rainnc = sampler.field(frame, "RAINNC").map(|f| f.into_iter().map(|c| c[0]).collect::<Vec<_>>());
    let refl = sampler.field(frame, "REFL_10CM");
    let w = sampler.field(frame, "W");
    let kappa = RGAS / CP;
    let mut out = Vec::with_capacity(sampler.points.len());
    for p in 0..sampler.points.len() {
        let n = theta[p].len();
        let temperature: Vec<f64> = (0..n)
            .map(|k| theta[p][k] * (pressure[p][k] / 1.0e5).powf(kappa))
            .collect();
        let z = &zgrid[p];
        let height: Vec<f64> = (0..n).map(|k| 0.5 * (z[k] + z[k + 1])).collect();
        let omega = w.as_ref().filter(|w| w[p].len() > n).map(|w| {
            (0..n)
                .map(|k| {
                    let wm = 0.5 * (w[p][k] + w[p][k + 1]);
                    -9.80665 * pressure[p][k] * wm / (287.05 * temperature[k] * (1.0 + 0.61 * qv[p][k]))
                })
                .collect::<Vec<f64>>()
        });
        let precip = rainc.as_ref().map(|r| r[p]).unwrap_or(0.0)
            + rainnc.as_ref().map(|r| r[p]).unwrap_or(0.0);
        out.push(Column {
            pressure_pa: pressure[p].clone(),
            temperature_k: temperature,
            qv: qv[p].clone(),
            height_m: height,
            u: u[p].clone(),
            v: v[p].clone(),
            terrain_m: z[0],
            psfc_pa: psfc[p],
            t2_k: t2[p],
            q2: q2[p],
            u10: u10[p],
            v10: v10[p],
            precip_mm: precip,
            refl_max_dbz: refl.as_ref().map(|r| r[p].iter().copied().fold(f64::MIN, f64::max)),
            omega_pa_s: omega,
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn points_parse_and_slug() {
        let pts = parse_points("Oklahoma City@35.47,-97.52; Tokyo@35.68,139.69 ;").unwrap();
        assert_eq!(pts.len(), 2);
        assert_eq!(pts[0].slug(), "oklahoma-city");
        assert_eq!(pts[1].lon, 139.69);
        assert!(parse_points("Nowhere@95,0").is_err());
        assert!(parse_points("bad").is_err());
        let wrapped = parse_points("x@0,190").unwrap();
        assert!((wrapped[0].lon + 170.0).abs() < 1e-12);
    }

    #[test]
    fn compacted_sampler_gathers_the_same_values_from_the_cell_subset() {
        let n_cells = 40usize;
        let levels = 3usize;
        let sampler = PointSampler {
            points: parse_points("a@0,0;b@1,1;c@2,2").unwrap(),
            linear: LinearWeights {
                cells: vec![[31, 7, 12], [7, 12, 2], [39, 0, 31]],
                weights: vec![[0.2, 0.3, 0.5], [0.6, 0.1, 0.3], [0.25, 0.25, 0.5]],
                contained: 3,
                widened: 0,
                clamped: 0,
            },
            nearest_cell: vec![31, 7, 39],
            nearest_km: vec![1.0; 3],
            off_mesh: vec![false; 3],
            local_spacing_m: vec![1000.0; 3],
        };
        let whole: Vec<f32> = (0..n_cells * levels).map(|i| (i as f32).sin() * 300.0).collect();
        let (cells, compact) = sampler.compacted();
        assert_eq!(cells, vec![0, 2, 7, 12, 31, 39]);
        let subset: Vec<f32> = cells
            .iter()
            .flat_map(|&c| whole[c as usize * levels..(c as usize + 1) * levels].to_vec())
            .collect();
        let a = sampler.linear.gather(&whole, levels);
        let b = compact.linear.gather(&subset, levels);
        assert_eq!(a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), b.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
        assert_eq!(compact.nearest_cell, sampler.nearest_cell);
    }

    #[test]
    fn point_fields_mask_fill_before_interpolation() {
        let sampler = PointSampler {
            points: parse_points("sample@0,0").unwrap(),
            linear: LinearWeights { cells: vec![[0, 1, 2]], weights: vec![[0.5, 0.25, 0.25]], contained: 1, widened: 0, clamped: 0 },
            nearest_cell: vec![0], nearest_km: vec![0.0], off_mesh: vec![false], local_spacing_m: vec![1000.0],
        };
        let mut frame = MpasFrame {
            history_path: "unused".into(), history_sha256: String::new(), init_path: None, init_sha256: None,
            valid_time: crate::history::Timestamp::parse("2024-01-01_00:00:00").unwrap(),
            n_cells: 3, n_levels: 1, n_soil_levels: 0, fields: Default::default(),
            latitude_degrees: vec![], longitude_degrees: vec![], absent: vec![], science_attributes: vec![], split: None, companion_diag: None,
        };
        for fill in [-1e30, 9.96921e36, f32::NAN] {
            frame.fields.insert("U10".into(), (vec![fill, 8.0, 12.0], 1));
            assert!(sampler.field(&frame, "U10").unwrap()[0][0].is_nan());
        }
        frame.fields.insert("U10".into(), (vec![4.0, 8.0, 12.0], 1));
        assert_eq!(sampler.field(&frame, "U10").unwrap()[0][0], 7.0);
    }

    #[test]
    fn mslp_is_the_surface_pressure_at_sea_level_and_rises_over_terrain() {
        let mut c = Column {
            pressure_pa: vec![],
            temperature_k: vec![],
            qv: vec![],
            height_m: vec![],
            u: vec![],
            v: vec![],
            terrain_m: 0.0,
            psfc_pa: 101_000.0,
            t2_k: 288.0,
            q2: 0.0,
            u10: 0.0,
            v10: 0.0,
            precip_mm: 0.0,
            refl_max_dbz: None,
            omega_pa_s: None,
        };
        assert_eq!(c.mslp_pa(), 101_000.0);
        c.terrain_m = 1500.0;
        c.psfc_pa = 85_000.0;
        let slp = c.mslp_pa();
        // A standard atmosphere puts 845 hPa near 1500 m; the reduction of
        // 850 hPa at 1500 m lands within a few hPa of 1005-1010 hPa.
        assert!(slp > 100_000.0 && slp < 102_500.0, "{slp}");
    }
}
