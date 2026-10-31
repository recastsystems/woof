//! PPI sweep sampling into the production rw_wrfbatch map renderer.
use crate::manifest::{self, Image};
use bowecho_simradar::radar_core::{ElevationCut, MomentType, RadarVolume};
use rustwx_render::{LegendControls, LegendMode, RenderDensity};
use rw_wrfbatch::{
    panel::{PanelRequest, render_panel},
    scales,
};
use rayon::prelude::*;
use std::path::Path;

const EARTH_M: f64 = 6_371_000.0;

/// The moments drawn as PPI images, in the order BowEcho lists its
/// products: reflectivity, velocity, correlation coefficient, differential
/// reflectivity, specific differential phase. A volume without a moment
/// draws no image for it, so a run without dual-pol moments draws
/// reflectivity and velocity only. `resources::output_bound` prices one
/// image per listed field and drawn tilt.
pub const PPI_FIELDS: &[&str] = &["reflectivity", "velocity", "rhohv", "zdr", "kdp"];

/// BowEcho's own presentation floor for simulated reflectivity
/// (`SyntheticRadarConfig::default().ref_floor_dbz`), used when the
/// reflectivity scale carries no display floor of its own.
const PRESENTATION_FLOOR_DBZ: f32 = 0.0;

/// The fixed radial velocity colour range, m/s either side of zero.
const VELOCITY_HALF_RANGE_M_S: f64 = 60.0;

pub(crate) struct DrawnField {
    pub(crate) moment: MomentType,
    units: &'static str,
    title: &'static str,
    /// Drawn only where the same gate's reflectivity is drawn. BowEcho's
    /// simulator writes a dual-polarization gate only where reflectivity
    /// survives its presentation floor, so its ZDR, CC and KDP cover the
    /// echo and nothing else. WOOF keeps every gate down to -30 dBZ in the
    /// radar files, so the PPI applies that rule at draw time, with the
    /// reflectivity image's own floor.
    echo_masked: bool,
}

pub(crate) fn drawn_field(field: &str) -> DrawnField {
    let (moment, units, title, echo_masked) = match field {
        "reflectivity" => (MomentType::Reflectivity, "dBZ", "reflectivity", false),
        "velocity" => (MomentType::Velocity, "m s-1", "radial velocity", false),
        "rhohv" => (
            MomentType::CorrelationCoefficient,
            "",
            "correlation coefficient CC",
            true,
        ),
        "zdr" => (
            MomentType::DifferentialReflectivity,
            "dB",
            "differential reflectivity ZDR",
            true,
        ),
        "kdp" => (
            MomentType::SpecificDifferentialPhase,
            "deg km-1",
            "specific differential phase KDP",
            true,
        ),
        other => unreachable!("{other} is not a PPI field"),
    };
    DrawnField {
        moment,
        units,
        title,
        echo_masked,
    }
}

/// The reflectivity below which a PPI draws nothing: the reflectivity
/// scale's own display floor (10 dBZ on both radar colour sets), else
/// BowEcho's presentation floor.
fn echo_floor_dbz() -> f32 {
    let floor = match scales::reflectivity_scale() {
        rustwx_render::ColorScale::Discrete(scale) => scale.mask_below,
        rustwx_render::ColorScale::Weather(preset) => preset.scale().mask_below,
    };
    floor.map(|value| value as f32).unwrap_or(PRESENTATION_FLOOR_DBZ)
}

/// One moment of one cut, as the PPI samples it.
struct Sampled<'a> {
    grid: &'a bowecho_simradar::radar_core::MomentGrid,
    rows: Vec<(f64, usize)>,
}

impl<'a> Sampled<'a> {
    fn new(cut: &'a ElevationCut, moment: &MomentType) -> Option<Self> {
        let grid = cut.moments.get(moment)?;
        let rows = azimuth_rows(cut, moment);
        (!rows.is_empty() && grid.gate_range.gate_count > 0).then_some(Self { grid, rows })
    }

    /// The nearest gate to `slant_m` along the nearest ray to `azimuth_deg`.
    fn value(&self, azimuth_deg: f64, slant_m: f64) -> f32 {
        let gate = ((slant_m - f64::from(self.grid.gate_range.first_gate_m))
            / f64::from(self.grid.gate_range.gate_spacing_m))
        .round();
        if gate >= 0.0 && gate < self.grid.gate_range.gate_count as f64 {
            nearest_row(&self.rows, azimuth_deg)
                .and_then(|row| self.grid.scaled_value(row, gate as usize))
                .unwrap_or(f32::NAN)
        } else {
            f32::NAN
        }
    }
}

/// The value a PPI pixel draws. A dual-polarization moment is drawn only
/// where the same gate's reflectivity reaches `floor_dbz`; `echo` is `None`
/// for reflectivity and velocity, which are never masked here.
fn pixel_value(
    field: &Sampled,
    echo: Option<&Sampled>,
    floor_dbz: f32,
    azimuth_deg: f64,
    slant_m: f64,
) -> f32 {
    if let Some(echo) = echo {
        // NaN (no reflectivity gate) fails the comparison and masks too.
        if !(echo.value(azimuth_deg, slant_m) >= floor_dbz) {
            return f32::NAN;
        }
    }
    field.value(azimuth_deg, slant_m)
}

fn destination(lat: f64, lon: f64, azimuth: f64, ground_m: f64) -> (f32, f32) {
    let (lat, lon) = (lat.to_radians(), lon.to_radians());
    let a = ground_m / EARTH_M;
    let y = (lat.sin() * a.cos() + lat.cos() * a.sin() * azimuth.cos()).asin();
    let x = lon + (azimuth.sin() * a.sin() * lat.cos()).atan2(a.cos() - lat.sin() * y.sin());
    (
        y.to_degrees() as f32,
        ((x.to_degrees() + 180.0).rem_euclid(360.0) - 180.0) as f32,
    )
}

fn azimuth_rows(cut: &ElevationCut, moment: &MomentType) -> Vec<(f64, usize)> {
    let Some(grid) = cut.moments.get(moment) else {
        return Vec::new();
    };
    let mut rows: Vec<_> = grid
        .radial_indices
        .iter()
        .enumerate()
        .filter_map(|(row, &radial)| {
            cut.radials
                .get(radial)
                .map(|r| (f64::from(r.azimuth_deg).rem_euclid(360.0), row))
        })
        .collect();
    rows.sort_by(|a, b| a.0.total_cmp(&b.0));
    rows
}

/// Nearest measured ray, including the north crossing.
fn nearest_row(rows: &[(f64, usize)], azimuth: f64) -> Option<usize> {
    if rows.is_empty() {
        return None;
    }
    let right = rows.partition_point(|r| r.0 < azimuth) % rows.len();
    let left = (right + rows.len() - 1) % rows.len();
    let distance = |v: f64| ((v - azimuth + 180.0).rem_euclid(360.0) - 180.0).abs();
    Some(if distance(rows[left].0) <= distance(rows[right].0) {
        rows[left].1
    } else {
        rows[right].1
    })
}

pub fn render(
    root: &Path,
    domain: &str,
    generation: &str,
    volume: &RadarVolume,
    tilt_count: usize,
    width: u32,
    height: u32,
) -> Result<Vec<Image>, String> {
    render_labeled(
        root,
        domain,
        generation,
        volume,
        tilt_count,
        width,
        height,
        "WOOF simulated",
        None,
    )
}

pub fn render_labeled(
    root: &Path,
    domain: &str,
    generation: &str,
    volume: &RadarVolume,
    tilt_count: usize,
    width: u32,
    height: u32,
    label: &str,
    range_limit_m: Option<f64>,
) -> Result<Vec<Image>, String> {
    render_labeled_with_scales(root,domain,generation,volume,tilt_count,width,height,label,range_limit_m,&[])
}

/// Evidence panels may share an expanded scale when finite C/X values exceed
/// the ordinary display range. Empty overrides preserve product image bytes.
pub fn render_labeled_with_scales(
    root: &Path, domain: &str, generation: &str, volume: &RadarVolume,
    tilt_count: usize, width: u32, height: u32, label: &str,
    range_limit_m: Option<f64>, overrides: &[(&str,f64,f64)],
) -> Result<Vec<Image>, String> {
    let lat = f64::from(volume.site.latitude_deg.ok_or("radar lacks latitude")?);
    let lon = f64::from(volume.site.longitude_deg.ok_or("radar lacks longitude")?);
    let stem = volume.volume_time.format("%Y%m%dT%H%M%SZ").to_string();
    let mut tilts: Vec<f32> = volume.cuts.iter().map(|c| c.elevation_deg).collect();
    tilts.sort_by(f32::total_cmp);
    tilts.dedup_by(|a, b| (*a - *b).abs() < 0.01);
    tilts.truncate(tilt_count);
    // Display resolution changes only the image, never the volume gate data.
    let n = width.max(height).clamp(256, 1000) as usize;
    // Every (tilt, field) image is sampled and drawn at once; the images
    // keep the serial order (tilt, then PPI_FIELDS order).
    let jobs: Vec<(usize, f32, &str)> = tilts
        .iter()
        .enumerate()
        .flat_map(|(tilt_index, &elevation)| {
            PPI_FIELDS.iter().map(move |&field| (tilt_index, elevation, field))
        })
        .collect();
    let drawn_images = jobs
        .into_par_iter()
        .map(|(tilt_index, elevation, field)| -> Result<Option<Image>, String> {
            let drawn = drawn_field(field);
            let (moment, units) = (drawn.moment, drawn.units);
            let Some((sweep_index, cut)) = volume.cuts.iter().enumerate().find(|(_, c)| {
                (c.elevation_deg - elevation).abs() < 0.01 && c.moments.contains_key(&moment)
            }) else {
                return Ok(None);
            };
            let Some(sampled) = Sampled::new(cut, &moment) else {
                return Ok(None);
            };
            let grid = sampled.grid;
            // A dual-pol moment with no reflectivity in its cut has no echo
            // to show it against, so nothing of it is drawn.
            let echo = if drawn.echo_masked {
                match Sampled::new(cut, &MomentType::Reflectivity) {
                    Some(echo) => Some(echo),
                    None => return Ok(None),
                }
            } else {
                None
            };
            let floor_dbz = echo_floor_dbz();
            let range = f64::from(grid.gate_range.first_gate_m)
                + f64::from(grid.gate_range.gate_spacing_m)
                    * (grid.gate_range.gate_count - 1) as f64;
            let range = range_limit_m.map(|limit| range.min(limit)).unwrap_or(range);
            let mut lats = vec![0.0f32; n * n];
            let mut lons = vec![0.0f32; n * n];
            let mut values = vec![0.0f32; n * n];
            lats.par_chunks_mut(n)
                .zip(lons.par_chunks_mut(n))
                .zip(values.par_chunks_mut(n))
                .enumerate()
                .for_each(|(j, ((lat_row, lon_row), value_row))| {
                    for i in 0..n {
                        let east = (2.0 * i as f64 / (n - 1) as f64 - 1.0) * range;
                        let north = (2.0 * j as f64 / (n - 1) as f64 - 1.0) * range;
                        let ground = east.hypot(north);
                        let az = east.atan2(north);
                        let (y, x) = destination(lat, lon, az, ground);
                        lat_row[i] = y;
                        lon_row[i] = x;
                        let earth = EARTH_M * (4.0 / 3.0);
                        let arc = ground / earth;
                        let slant =
                            earth * arc.sin() / (f64::from(elevation).to_radians() + arc).cos();
                        value_row[i] = pixel_value(
                            &sampled,
                            echo.as_ref(),
                            floor_dbz,
                            az.to_degrees().rem_euclid(360.0),
                            slant,
                        );
                    }
                });
            let path = root
                .join("radar")
                .join(domain)
                .join(&volume.site.id)
                .join(generation)
                .join("ppi")
                .join(field)
                .join(format!("{stem}_tilt-{tilt_index:02}.png"));
            let mut scale = scales::radar_scale_named(field, VELOCITY_HALF_RANGE_M_S)
                .ok_or_else(|| format!("no colour scale for radar field {field}"))?;
            if let Some(&(_,lo,hi))=overrides.iter().find(|v|v.0==field) {
                if !lo.is_finite() || !hi.is_finite() || hi<=lo {
                    return Err("evidence colour limits must be finite and increasing".into());
                }
                let mut discrete=scale.resolved_discrete();
                let first=discrete.levels[0];
                let span=discrete.levels[discrete.levels.len()-1]-first;
                for value in &mut discrete.levels {*value=lo+(*value-first)/span*(hi-lo);}
                scale=rustwx_render::ColorScale::Discrete(discrete);
            }
            render_panel(PanelRequest {
                lat_deg: &lats,
                lon_deg: &lons,
                projection: None,
                ny: n,
                nx: n,
                values,
                product_slug: format!("simulated_radar_{field}"),
                title: if units.is_empty() {
                    format!("{label} {}", drawn.title)
                } else {
                    format!("{label} {} ({units})", drawn.title)
                },
                display_units: units.into(),
                scale,
                cbar_tick_step: None,
                legend: LegendControls {
                    mode: LegendMode::SmoothRamp,
                    ..Default::default()
                },
                render_density: RenderDensity::default(),
                subtitle_left: format!(
                    "{} | {:.2} deg | {}",
                    volume.site.id,
                    elevation,
                    volume.volume_time.format("%Y-%m-%d %H:%M:%S UTC")
                ),
                subtitle_center: None,
                subtitle_right: domain.into(),
                width,
                height,
                contours: Vec::new(),
                colorbar: true,
                overlays: None,
                annotations: None,
                out_path: path.clone(),
            })?;
            Ok(Some(Image {
                artifact: manifest::artifact(root, &path, "png")?,
                field: field.into(),
                tilt_index,
                sweep_index,
                elevation_deg: elevation,
            }))
        })
        .collect::<Result<Vec<_>, String>>()?;
    Ok(drawn_images.into_iter().flatten().collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn every_drawn_field_has_a_moment_and_a_scale() {
        for &field in PPI_FIELDS {
            let drawn = drawn_field(field);
            assert!(!drawn.title.is_empty());
            assert!(scales::radar_scale_named(field, VELOCITY_HALF_RANGE_M_S).is_some(), "{field}");
        }
    }

    /// Four rays, eight 1 km gates: reflectivity 40 dBZ in the first four
    /// gates, 5 dBZ in the next two, missing in the last two; ZDR 1.5 dB at
    /// every gate.
    fn echo_cut() -> ElevationCut {
        use bowecho_simradar::radar_core::{GateRange, MomentGrid, MomentStorage, Radial};
        let gate_range = GateRange {
            first_gate_m: 500,
            gate_spacing_m: 1000,
            gate_count: 8,
        };
        let mut cut = ElevationCut::new(0.5, Some(1));
        for azimuth in [0.0f32, 90.0, 180.0, 270.0] {
            cut.radials.push(Radial {
                azimuth_deg: azimuth,
                elevation_deg: 0.5,
                time_offset_ms: 0,
                gate_range: gate_range.clone(),
                nyquist_velocity_mps: None,
                radial_status: None,
            });
        }
        let grid = |moment: MomentType, row: [f32; 8]| MomentGrid {
            moment,
            gate_range: gate_range.clone(),
            scale: 1.0,
            offset: 0.0,
            nodata: None,
            range_folded: None,
            radial_indices: vec![0, 1, 2, 3],
            storage: MomentStorage::F32(row.iter().copied().cycle().take(32).collect()),
        };
        let nan = f32::NAN;
        cut.moments.insert(
            MomentType::Reflectivity,
            grid(
                MomentType::Reflectivity,
                [40.0, 40.0, 40.0, 40.0, 5.0, 5.0, nan, nan],
            ),
        );
        cut.moments.insert(
            MomentType::DifferentialReflectivity,
            grid(MomentType::DifferentialReflectivity, [1.5; 8]),
        );
        cut
    }

    #[test]
    fn dual_pol_is_drawn_only_where_reflectivity_is_drawn() {
        let cut = echo_cut();
        let zdr = Sampled::new(&cut, &MomentType::DifferentialReflectivity).unwrap();
        let echo = Sampled::new(&cut, &MomentType::Reflectivity).unwrap();
        let floor = 10.0;
        for azimuth in [0.0, 90.0, 181.0, 359.0] {
            // Inside the 40 dBZ echo the moment is drawn.
            assert_eq!(pixel_value(&zdr, Some(&echo), floor, azimuth, 2500.0), 1.5);
            // Below the reflectivity display floor it is not.
            assert!(pixel_value(&zdr, Some(&echo), floor, azimuth, 4500.0).is_nan());
            // Nor where reflectivity has no gate at all.
            assert!(pixel_value(&zdr, Some(&echo), floor, azimuth, 7500.0).is_nan());
            // Unmasked, the same gates all carry the moment: the mask is
            // the only thing that removed them.
            assert_eq!(pixel_value(&zdr, None, floor, azimuth, 4500.0), 1.5);
            assert_eq!(pixel_value(&zdr, None, floor, azimuth, 7500.0), 1.5);
        }
        // Reflectivity itself is never masked here; its scale hides the weak gates.
        assert_eq!(pixel_value(&echo, None, floor, 0.0, 4500.0), 5.0);
        // Past the last gate there is nothing to draw either way.
        assert!(pixel_value(&zdr, Some(&echo), floor, 0.0, 9500.0).is_nan());
    }

    #[test]
    fn only_the_dual_pol_fields_are_echo_masked() {
        let masked: Vec<&str> = PPI_FIELDS
            .iter()
            .copied()
            .filter(|field| drawn_field(field).echo_masked)
            .collect();
        assert_eq!(masked, ["rhohv", "zdr", "kdp"]);
        assert_eq!(echo_floor_dbz(), 10.0, "the reflectivity image's own floor");
    }

    #[test]
    fn ray_lookup_wraps_at_north() {
        let rows = [(1.0, 7), (90.0, 8), (359.0, 9)];
        assert_eq!(nearest_row(&rows, 359.9), Some(9));
        assert_eq!(nearest_row(&rows, 0.9), Some(7));
    }
}
