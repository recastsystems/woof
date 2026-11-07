//! Fixed-Earth cyclone verification products from owned float32 diagnostics.
//! All reductions and map drawing are native; model time-maxima are collected
//! on the GPU at every committed fine-domain step, before any nest relocation.
use crate::{
    annotate::{parse_color, LineSpec, MapOverlays, PointSpec},
    panel::{layout_path, render_panel_with_unobserved_bounds, PanelRequest},
    scales,
};
use chrono::{DateTime, NaiveDateTime, Utc};
use rayon::prelude::*;
use rustwx_render::{ColorScale, LegendControls, LegendMode, LevelDensity, RenderDensity};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::Read,
    path::{Path, PathBuf},
};

// Preserve rare positive probabilities in a 100-child ensemble. The generic
// 10% display cutoff would erase one- to nine-child events.
fn cyclone_probability_scale() -> ColorScale {
    let mut scale = scales::probability_scale();
    if let ColorScale::Discrete(ref mut discrete) = scale {
        discrete.mask_below = Some(1e-12);
    }
    scale
}

// These sparse shore-only fields retain the geographic context outside their
// finite stencil. Zero remains a real duration/onset value, not a missing event.
fn coastal_time_scale(hours: f64) -> ColorScale {
    let mut scale = scales::spread_scale(hours);
    if let ColorScale::Discrete(ref mut discrete) = scale {
        discrete.mask_below = Some(0.0);
    }
    scale
}

#[derive(Deserialize)]
struct Frame {
    valid_time: String,
    path: PathBuf,
    shape: [usize; 2],
    grid_id: usize,
}
#[derive(Deserialize)]
struct Common {
    directory: PathBuf,
    shape: [usize; 2],
    samples: usize,
    #[serde(default)]
    full_coverage_wind: Option<String>,
}
#[derive(Deserialize)]
struct Member {
    member_id: usize,
    parent_id: serde_json::Value,
    weight: f64,
    frames: Vec<Frame>,
    common_wind: Common,
}
#[derive(Deserialize)]
struct Request {
    analysis_time: String,
    observation_cutoff: String,
    issue_time: String,
    landfall_reference: String,
    out: PathBuf,
    members: Vec<Member>,
    best_track: PathBuf,
    observed_radar: Option<PathBuf>,
    experiment_label: String,
    model_analysis_seconds: f64,
    #[serde(default = "grid_id")]
    grid_id: usize,
    #[serde(default = "threshold")]
    wind_threshold_m_s: f32,
    #[serde(default = "onshore_threshold")]
    onshore_threshold_m_s: f32,
}
fn grid_id() -> usize {
    2
}
fn threshold() -> f32 {
    32.92
}
fn onshore_threshold() -> f32 {
    10.
}
#[derive(Clone, Serialize, Deserialize)]
struct Fix {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    footprint: Vec<(f64, f64)>,
    valid_time: String,
    latitude: f64,
    longitude: f64,
    mslp_hpa: f32,
    max_wind_m_s: f32,
}
#[derive(Serialize)]
struct Track {
    member_id: usize,
    parent_id: serde_json::Value,
    weight: f64,
    color: String,
    fixes: Vec<Fix>,
    plotted_fixes: Vec<Fix>,
}

// Display-only 30-minute centroid. Native minima, intensity and footprints
// remain in `fixes`; neither the model tracker nor any swath uses this filter.
fn plotted_fixes(raw: &[Fix]) -> Result<Vec<Fix>, String> {
    let times = raw
        .iter()
        .map(|f| date(&f.valid_time))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(raw
        .iter()
        .zip(&times)
        .map(|(fix, time)| {
            let nearby: Vec<_> = raw
                .iter()
                .zip(&times)
                .filter(|(_, t)| (**t - *time).num_seconds().abs() <= 15 * 60)
                .map(|(f, _)| f)
                .collect();
            let mut plotted = fix.clone();
            plotted.latitude = nearby.iter().map(|f| f.latitude).sum::<f64>() / nearby.len() as f64;
            let offset = nearby
                .iter()
                .map(|f| (f.longitude - fix.longitude + 180.).rem_euclid(360.) - 180.)
                .sum::<f64>()
                / nearby.len() as f64;
            plotted.longitude = (fix.longitude + offset + 180.).rem_euclid(360.) - 180.;
            plotted
        })
        .collect())
}
fn read(path: &Path, cells: usize) -> Result<Vec<f32>, String> {
    let bytes = fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if bytes.len() != cells.checked_mul(4).ok_or("grid size overflow")? {
        return Err(format!("{}: wrong float32 shape", path.display()));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect())
}
fn plane(frame: &Frame, name: &str) -> Result<Vec<f32>, String> {
    read(
        &frame
            .path
            .parent()
            .ok_or("frame path has no parent")?
            .join(format!("{name}.f32")),
        frame.shape[0] * frame.shape[1],
    )
}
// Invert each curvilinear source cell in parallel; use one Earth frame for
// all stamps. Never stretch a moving member's footprint to fill its panel.
fn regrid(frame: &Frame, target_lat: &[f32], target_lon: &[f32]) -> Result<Vec<f32>, String> {
    let lat = plane(frame, "latitude")?;
    let lon = plane(frame, "longitude")?;
    let field = plane(frame, "refl")?;
    let [ny, nx] = frame.shape;
    if nx < 2 || ny < 2 {
        return Err("reflectivity source grid needs at least two cells per axis".into());
    }
    Ok(target_lat
        .par_iter()
        .zip(target_lon)
        .map(|(&la, &lo)| {
            let (mut x, mut y) = (nx as f64 / 2., ny as f64 / 2.);
            for _ in 0..12 {
                if !x.is_finite() || !y.is_finite() {
                    return f32::NAN;
                }
                let i = x.clamp(0., (nx - 2) as f64) as usize;
                let j = y.clamp(0., (ny - 2) as f64) as usize;
                let (a, b) = (x - i as f64, y - j as f64);
                let k = j * nx + i;
                let sample = |q: &[f32]| {
                    (1. - a) * (1. - b) * q[k] as f64
                        + a * (1. - b) * q[k + 1] as f64
                        + (1. - a) * b * q[k + nx] as f64
                        + a * b * q[k + nx + 1] as f64
                };
                let dl = la as f64 - sample(&lat);
                let dn = (lo as f64 - sample(&lon) + 180.).rem_euclid(360.) - 180.;
                if dl.abs() < 1e-5 && dn.abs() < 1e-5 {
                    if x < 0.
                        || y < 0.
                        || x > (nx - 1) as f64
                        || y > (ny - 1) as f64
                        || [k, k + 1, k + nx, k + nx + 1]
                            .iter()
                            .any(|&v| !field[v].is_finite())
                    {
                        return f32::NAN;
                    }
                    return sample(&field) as f32;
                }
                let derivative = |q: &[f32]| {
                    (
                        (1. - b) * (q[k + 1] - q[k]) as f64
                            + b * (q[k + nx + 1] - q[k + nx]) as f64,
                        (1. - a) * (q[k + nx] - q[k]) as f64
                            + a * (q[k + nx + 1] - q[k + 1]) as f64,
                    )
                };
                let (lx, ly) = derivative(&lat);
                let (nx_, ny_) = derivative(&lon);
                let det = lx * ny_ - ly * nx_;
                x += (dl * ny_ - dn * ly) / det;
                y += (dn * lx - dl * nx_) / det;
            }
            f32::NAN
        })
        .collect())
}
fn date(s: &str) -> Result<DateTime<Utc>, String> {
    DateTime::parse_from_rfc3339(s)
        .map(|v| v.with_timezone(&Utc))
        .map_err(|e| format!("invalid UTC {s}: {e}"))
}
fn color(index: usize) -> String {
    let h = (index as f64 * 0.61803398875) % 1.;
    let sector = h * 6.;
    let x = 1. - (sector % 2. - 1.).abs();
    let (r, g, b) = match sector as usize {
        0 => (1., x, 0.),
        1 => (x, 1., 0.),
        2 => (0., 1., x),
        3 => (0., x, 1.),
        4 => (x, 0., 1.),
        _ => (1., 0., x),
    };
    format!(
        "#{:02x}{:02x}{:02x}",
        ((r * 0.7 + 0.1) * 255.) as u8,
        ((g * 0.7 + 0.1) * 255.) as u8,
        ((b * 0.7 + 0.1) * 255.) as u8
    )
}
fn best_time(hour: &str, minute: &str) -> Result<String, String> {
    // BEST uses the technique-number column for minutes at special fixes,
    // including landfall. A blank means the ordinary synoptic hour.
    let minute = if minute.trim().is_empty() {
        "00"
    } else {
        minute.trim()
    };
    NaiveDateTime::parse_from_str(&format!("{hour}{minute:0>2}"), "%Y%m%d%H%M")
        .map(|v| v.and_utc().to_rfc3339())
        .map_err(|e| e.to_string())
}
fn best_track(path: &Path) -> Result<Vec<Fix>, String> {
    let file = fs::File::open(path).map_err(|e| e.to_string())?;
    let mut text = String::new();
    if path.extension().is_some_and(|e| e == "gz") {
        flate2::read::GzDecoder::new(file)
            .read_to_string(&mut text)
            .map_err(|e| e.to_string())?;
    } else {
        let mut file = file;
        file.read_to_string(&mut text).map_err(|e| e.to_string())?;
    }
    let mut fixes = BTreeMap::new();
    for line in text.lines() {
        let c: Vec<_> = line.split(',').map(str::trim).collect();
        if c.len() < 10 || c[4] != "BEST" {
            continue;
        }
        let time = best_time(c[2], c[3])?;
        let coordinate = |s: &str| -> Result<f64, String> {
            let v = s[..s.len() - 1].parse::<f64>().map_err(|e| e.to_string())? / 10.;
            Ok(if s.ends_with('S') || s.ends_with('W') {
                -v
            } else {
                v
            })
        };
        fixes.entry(time.clone()).or_insert(Fix {
            footprint: vec![],
            valid_time: time,
            latitude: coordinate(c[6])?,
            longitude: coordinate(c[7])?,
            max_wind_m_s: c[8].parse::<f32>().map_err(|e| e.to_string())? * 1852. / 3600.,
            mslp_hpa: c[9]
                .parse()
                .map_err(|e: std::num::ParseFloatError| e.to_string())?,
        });
    }
    if fixes.is_empty() {
        return Err("best-track archive contains no BEST fixes".into());
    }
    Ok(fixes.into_values().collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fix(minute: u32, latitude: f64, longitude: f64) -> Fix {
        Fix {
            footprint: vec![(24., -83.)],
            valid_time: format!("2022-09-28T07:{minute:02}:00Z"),
            latitude,
            longitude,
            mslp_hpa: 965.,
            max_wind_m_s: 47.,
        }
    }

    #[test]
    fn plotted_centroid_uses_thirty_minutes_and_preserves_raw_fixes() {
        let raw: Vec<_> = (0..9)
            .map(|i| fix(i * 5, 25. + (i % 2) as f64 * 0.1, -83.))
            .collect();
        let before = serde_json::to_value(&raw).unwrap();
        let plotted = plotted_fixes(&raw).unwrap();
        assert!((plotted[4].latitude - (25. + 0.4 / 7.)).abs() < 1e-10);
        assert!((plotted[0].latitude - 25.05).abs() < 1e-10);
        assert_eq!(serde_json::to_value(&raw).unwrap(), before);
        assert_eq!(plotted[4].mslp_hpa, raw[4].mslp_hpa);
        assert_eq!(plotted[4].footprint, raw[4].footprint);
        let track = Track {
            member_id: 0,
            parent_id: "parent".into(),
            weight: 1.,
            color: "#123456".into(),
            fixes: raw,
            plotted_fixes: plotted,
        };
        let layers = overlay(&[track], &[], None);
        assert!(layers.labels.is_empty());
        assert_eq!(layers.lines.len(), 2); // One real track and the best-track layer; no legend lines.
        assert!((layers.lines[0].points[4].0 - (25. + 0.4 / 7.)).abs() < 1e-10);
    }

    #[test]
    fn plotted_centroid_does_not_average_across_the_long_way_round_earth() {
        let plotted = plotted_fixes(&[fix(0, 25., 179.), fix(15, 25., -179.)]).unwrap();
        assert!(plotted
            .iter()
            .all(|f| (f.longitude.abs() - 180.).abs() < 1e-10));
    }

    #[test]
    fn atcf_best_retains_special_fix_minutes() {
        assert_eq!(
            super::best_time("2022092819", "05").unwrap(),
            "2022-09-28T19:05:00+00:00"
        );
        assert_eq!(
            super::best_time("2022092806", "  ").unwrap(),
            "2022-09-28T06:00:00+00:00"
        );
    }
}
fn fixes(member: &Member, grid: usize) -> Result<Vec<Fix>, String> {
    let mut result: Vec<_> = member
        .frames
        .par_iter()
        .filter(|f| f.grid_id == grid)
        .map(|f| {
            let p = plane(f, "mslp")?;
            let wind = plane(f, "wind10")?;
            let (k, &minimum) = p
                .iter()
                .enumerate()
                .filter(|(_, x)| x.is_finite())
                .min_by(|a, b| a.1.total_cmp(b.1))
                .ok_or("no finite member MSLP")?;
            let lat = plane(f, "latitude")?;
            let lon = plane(f, "longitude")?;
            let vmax = wind
                .into_iter()
                .filter(|x| x.is_finite())
                .max_by(f32::total_cmp)
                .ok_or("no finite member wind")?;
            let [ny, nx] = f.shape;
            let footprint = [0, nx - 1, ny * nx - 1, (ny - 1) * nx]
                .iter()
                .map(|&k| (lat[k] as f64, lon[k] as f64))
                .collect();
            Ok(Fix {
                footprint,
                valid_time: date(&f.valid_time)?.to_rfc3339(),
                latitude: lat[k] as f64,
                longitude: lon[k] as f64,
                mslp_hpa: minimum,
                max_wind_m_s: vmax,
            })
        })
        .collect::<Result<_, String>>()?;
    result.sort_by(|a, b| a.valid_time.cmp(&b.valid_time));
    if result.is_empty()
        || result
            .windows(2)
            .any(|w| w[0].valid_time == w[1].valid_time)
    {
        return Err("member tracks need unique valid times".into());
    }
    Ok(result)
}
#[allow(clippy::too_many_arguments)]
fn draw(
    r: &Request,
    slug: &str,
    title: String,
    valid: &str,
    lat: &[f32],
    lon: &[f32],
    shape: [usize; 2],
    values: Vec<f32>,
    units: &str,
    scale: ColorScale,
    overlays: Option<&MapOverlays>,
    colorbar: bool,
) -> Result<PathBuf, String> {
    let finite = values.par_iter().filter(|v| v.is_finite()).count();
    let maximum = values
        .par_iter()
        .copied()
        .filter(|v| v.is_finite())
        .reduce(|| f32::NEG_INFINITY, f32::max);
    eprintln!(
        "CYCLONE_FIELD {slug}: finite={finite}/{} max={maximum}",
        values.len()
    );
    let lead = (date(&r.landfall_reference)? - date(&r.issue_time)?).num_seconds() as f64 / 3600.;
    let compact = |s: &str| date(s).map(|d| d.format("%Y-%m-%d %H:%MZ").to_string());
    let subtitle = format!(
        "Analysis {} | cutoff {} | issued {} | lead {lead:.2} h (hindcast)",
        compact(&r.analysis_time)?,
        compact(&r.observation_cutoff)?,
        compact(&r.issue_time)?
    );
    let path = layout_path(
        &r.out,
        &format!("d{:02}", r.grid_id),
        slug,
        &valid[..10],
        &format!("{slug}_{}", valid.replace(':', "-")),
    );
    let unobserved: Vec<_> = values.iter().map(|v| !v.is_finite()).collect();
    let coverage = (slug == "eyewall-observed").then_some(unobserved.as_slice());
    let bounds = rusty_weather::batch_render::native_grid_domain_from_coordinates(lat,lon)?.bounds;
    let result = render_panel_with_unobserved_bounds(
        PanelRequest {
            lat_deg: lat,
            lon_deg: lon,
            projection: None,
            ny: shape[0],
            nx: shape[1],
            values,
            product_slug: slug.into(),
            title,
            display_units: units.into(),
            scale,
            cbar_tick_step: None,
            legend: LegendControls {
                density: LevelDensity::default(),
                mode: LegendMode::Stepped,
            },
            render_density: RenderDensity::default(),
            subtitle_left: subtitle,
            subtitle_center: None,
            subtitle_right: valid.into(),
            width: 1440,
            height: 1080,
            contours: vec![],
            colorbar,
            overlays,
            annotations: None,
            out_path: path,
        },
        coverage,
        bounds,
    )?;
    println!("RENDERED\t{}", result.display());
    Ok(result)
}
fn overlay(tracks: &[Track], best: &[Fix], until: Option<&str>) -> MapOverlays {
    let mut overlays = MapOverlays::default();
    for track in tracks {
        let points: Vec<_> = track
            .plotted_fixes
            .iter()
            .filter(|f| until.is_none_or(|u| f.valid_time.as_str() <= u))
            .map(|f| (f.latitude, f.longitude))
            .collect();
        overlays.lines.push(LineSpec {
            points,
            color: track.color.clone(),
            width: 2,
            closed: false,
        });
        if let Some(fix) = track
            .plotted_fixes
            .iter()
            .rev()
            .find(|f| until.is_none_or(|u| f.valid_time.as_str() <= u))
        {
            overlays.points.push(PointSpec {
                lat: fix.latitude,
                lon: fix.longitude,
                color: track.color.clone(),
                radius_px: 4,
                width_px: 2,
                shape: "circle".into(),
            });
        }
        if let Some(time) = until {
            if let Some(fix) = track.fixes.iter().find(|f| f.valid_time == time) {
                overlays.lines.push(LineSpec {
                    points: fix.footprint.clone(),
                    color: track.color.clone(),
                    width: 1,
                    closed: true,
                });
            }
        }
    }
    // The full best track is explicitly verification; it never enters steering.
    overlays.lines.push(LineSpec {
        points: best.iter().map(|f| (f.latitude, f.longitude)).collect(),
        color: "#000000".into(),
        width: 4,
        closed: false,
    });
    overlays
}
fn legend(path: &Path, tracks: &[Track], has_best: bool) -> Result<(), String> {
    let families: BTreeMap<_, _> = tracks
        .iter()
        .map(|t| (t.parent_id.to_string(), t.color.clone()))
        .collect();
    // Compose a figure-space side panel. It never enters geographic overlays
    // and leaves every original map pixel, including its captions, untouched.
    let map = image::open(path).map_err(|e| e.to_string())?.to_rgba8();
    let rows = ((map.height().saturating_sub(210)) / 30).max(1) as usize;
    let columns = families.len().div_ceil(rows).max(1) as u32;
    let column_width = 300;
    let mut figure = image::RgbaImage::from_pixel(
        map.width() + columns * column_width,
        map.height(),
        image::Rgba([255, 255, 255, 255]),
    );
    image::imageops::replace(&mut figure, &map, 0, 0);
    let ink = rustwx_render::Color::rgba(23, 35, 52, 255);
    let left = map.width() as i32 + 16;
    rustwx_render::draw_text_line(&mut figure, "Ancestor / GEFS member", left, 32, ink, 2);
    for (k, (parent, color)) in families.into_iter().enumerate() {
        let y = 78 + (k % rows) as u32 * 30;
        let x = map.width() + 16 + (k / rows) as u32 * column_width;
        let rgba = parse_color(&color);
        for yy in y + 7..y + 11 {
            for xx in x..x + 26 {
                figure.put_pixel(xx, yy, image::Rgba([rgba.r, rgba.g, rgba.b, 255]));
            }
        }
        rustwx_render::draw_text_line(
            &mut figure,
            parent.trim_matches('"'),
            x as i32 + 36,
            y as i32,
            ink,
            2,
        );
    }
    let y = map.height() as i32 - 116;
    // NHC wording only when a best-track line is actually drawn (tc-tracks plume showed the
    // label with no black line, improve-55).
    if has_best {
        rustwx_render::draw_text_line(&mut figure, "Black: NHC verification", left, y, ink, 2);
    }
    rustwx_render::draw_text_line(&mut figure, "Tracks: 30-min centroid", left, y + 32, ink, 2);
    rustwx_render::draw_text_line(&mut figure, "Raw fixes in receipt", left, y + 60, ink, 2);
    rustwx_render::save_rgba_png_profile_with_options(
        &figure,
        path,
        &rustwx_render::PngWriteOptions::default(),
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}
fn run(path: &Path) -> Result<(), String> {
    let r: Request = serde_json::from_slice(&fs::read(path).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    if r.members.is_empty() {
        return Err("no completed members".into());
    }
    let total: f64 = r.members.iter().map(|m| m.weight).sum();
    if r.members
        .iter()
        .any(|m| !m.weight.is_finite() || m.weight <= 0.)
        || (total - 1.).abs() > 1e-6
    {
        return Err("member probability weights must be positive and sum to one".into());
    }
    let shape = r.members[0].common_wind.shape;
    let cells = shape[0]
        .checked_mul(shape[1])
        .ok_or("common grid overflow")?;
    if shape.iter().any(|&n| n < 2)
        || r.members
            .iter()
            .any(|m| m.common_wind.shape != shape || m.common_wind.samples < 2)
    {
        return Err("common-grid shape or every-step samples missing".into());
    }
    let common =
        |m: &Member, name: &str| read(&m.common_wind.directory.join(format!("{name}.f32")), cells);
    let lat = common(&r.members[0], "latitude")?;
    let lon = common(&r.members[0], "longitude")?;
    let families: Vec<_> = r
        .members
        .iter()
        .map(|m| m.parent_id.to_string())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let tracks: Vec<_> = r
        .members
        .par_iter()
        .map(|m| {
            let fixes = fixes(m, r.grid_id)?;
            Ok(Track {
                member_id: m.member_id,
                parent_id: m.parent_id.clone(),
                weight: m.weight,
                color: color(
                    families
                        .iter()
                        .position(|p| p == &m.parent_id.to_string())
                        .unwrap(),
                ),
                plotted_fixes: plotted_fixes(&fixes)?,
                fixes,
            })
        })
        .collect::<Result<_, String>>()?;
    let analysis = date(&r.analysis_time)?;
    let end = date(&tracks[0].fixes.last().unwrap().valid_time)?;
    let best: Vec<_> = best_track(&r.best_track)?
        .into_iter()
        .filter(|f| {
            date(&f.valid_time).is_ok_and(|t| {
                t >= analysis - chrono::Duration::hours(6) && t <= end + chrono::Duration::hours(6)
            })
        })
        .collect();
    fs::create_dir_all(&r.out).map_err(|e| e.to_string())?;
    fs::write(r.out.join("intensity-tracks.json"),serde_json::to_vec_pretty(&serde_json::json!({"members":tracks,"best_track":best,
        "analysis_time":r.analysis_time,"observation_cutoff":r.observation_cutoff,"issue_time":r.issue_time,"landfall_reference":r.landfall_reference,
        "wind_definition":"model instantaneous 10 m wind; not NHC sustained wind",
        "track_definition":"raw diagnostic MSLP minimum on each member's own following grid",
        "track_rendering":{"method":"geographic centroid of raw fixes within +/-15 minutes",
            "window_minutes":30,"endpoints":"truncated window, renormalized",
            "raw_field":"members[].fixes","plotted_field":"members[].plotted_fixes",
            "legend":"figure-space side panel; no geographic legend overlays",
            "scope":"position display only; intensity, footprints, steering and wind swaths unchanged"}})).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
    let times: Vec<_> = tracks[0]
        .fixes
        .iter()
        .map(|f| f.valid_time.as_str())
        .collect();
    if tracks.iter().any(|t| {
        t.fixes
            .iter()
            .map(|f| f.valid_time.as_str())
            .collect::<Vec<_>>()
            != times
    }) {
        return Err("tracks do not share exact UTC valid times".into());
    }
    let valid = times.last().ok_or("no valid times")?;
    let layers = overlay(&tracks, &best, None);
    let plume = draw(
        &r,
        "track-plume",
        format!(
            "{}: track plume{}",
            r.experiment_label,
            if best.is_empty() { "" } else { "; black=NHC verification" }
        ),
        valid,
        &lat,
        &lon,
        shape,
        vec![f32::NAN; cells],
        "",
        scales::probability_scale(),
        Some(&layers),
        false,
    )?;
    legend(&plume, &tracks, !best.is_empty())?;
    let winds: Vec<_> = r
        .members
        .par_iter()
        .map(|m| {
            if common(m, "latitude")? != lat || common(m, "longitude")? != lon {
                return Err("members have different common Earth grids".into());
            }
            common(m, m.common_wind.full_coverage_wind.as_deref().unwrap_or("time_max_wind10"))
        })
        .collect::<Result<_, String>>()?;
    let probability: Vec<f32> = (0..cells)
        .into_par_iter()
        .map(|i| {
            if winds.iter().any(|w| !w[i].is_finite()) {
                f32::NAN
            } else {
                r.members
                    .iter()
                    .zip(&winds)
                    .filter(|(_, w)| w[i] >= r.wind_threshold_m_s)
                    .map(|(m, _)| m.weight)
                    .sum::<f64>() as f32
            }
        })
        .collect();
    draw(
        &r,
        "wind64-time-max-probability",
        format!(
            "{}: P(max W10 >= {:.2} m/s)",
            r.experiment_label, r.wind_threshold_m_s
        ),
        valid,
        &lat,
        &lon,
        shape,
        probability,
        "fraction",
        cyclone_probability_scale(),
        None,
        true,
    )?;
    let durations: Vec<_> = r
        .members
        .par_iter()
        .map(|m| common(m, "onshore_duration"))
        .collect::<Result<_, String>>()?;
    let onset: Vec<_> = r
        .members
        .par_iter()
        .map(|m| common(m, "onshore_onset"))
        .collect::<Result<_, String>>()?;
    let duration = (0..cells)
        .into_par_iter()
        .map(|i| {
            if durations.iter().any(|d| !d[i].is_finite()) {
                f32::NAN
            } else {
                r.members
                    .iter()
                    .zip(&durations)
                    .map(|(m, d)| m.weight * d[i] as f64)
                    .sum::<f64>() as f32
            }
        })
        .collect();
    let begin = r.members[0]
        .frames
        .iter()
        .filter(|f| f.grid_id == r.grid_id)
        .map(|f| date(&f.valid_time))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .min()
        .ok_or("no first frame")?;
    // Model start offset is explicit in the request, so no onset is inferred
    // from the first event, and a member with no event stays event-free.
    let model_analysis_seconds = r.model_analysis_seconds as f32;
    let onset_hours = (0..cells)
        .into_par_iter()
        .map(|i| {
            if winds.iter().any(|w| !w[i].is_finite())
                || durations.iter().any(|d| !d[i].is_finite())
            {
                return f32::NAN;
            }
            onset
                .iter()
                .map(|d| d[i])
                .filter(|v| v.is_finite())
                .min_by(f32::total_cmp)
                .map(|s| (s - model_analysis_seconds) / 3600.)
                .unwrap_or(f32::NAN)
        })
        .collect();
    let horizon = (date(valid)? - begin).num_seconds() as f64 / 3600.;
    draw(
        &r,
        "coastal-onshore-duration",
        format!(
            "{}: weighted onshore duration (>= {} m/s)",
            r.experiment_label, r.onshore_threshold_m_s
        ),
        valid,
        &lat,
        &lon,
        shape,
        duration,
        "h",
        coastal_time_scale(horizon.max(1.)),
        None,
        true,
    )?;
    draw(
        &r,
        "coastal-onshore-onset",
        format!(
            "{}: earliest onshore onset (>= {} m/s)",
            r.experiment_label, r.onshore_threshold_m_s
        ),
        valid,
        &lat,
        &lon,
        shape,
        onset_hours,
        "h after analysis",
        coastal_time_scale(horizon.max(1.)),
        None,
        true,
    )?;
    times
        .par_iter()
        .try_for_each(|time| -> Result<(), String> {
            let layers = overlay(&tracks, &best, Some(time));
            let frame = draw(
                &r,
                "members-vs-best-track",
                format!(
                    "{}: tracks + nest footprints;{} {} members",
                    r.experiment_label,
                    if best.is_empty() { "" } else { " NHC black verification;" },
                    r.members.len()
                ),
                time,
                &lat,
                &lon,
                shape,
                vec![f32::NAN; cells],
                "",
                scales::probability_scale(),
                Some(&layers),
                false,
            )?;
            legend(&frame, &tracks, !best.is_empty())?;
            Ok(())
        })?;
    // One valid time, identical physical units, scale and Earth coordinates.
    r.members
        .par_iter()
        .try_for_each(|m| -> Result<(), String> {
            let frame = m
                .frames
                .iter()
                .find(|f| f.grid_id == r.grid_id && date(&f.valid_time).ok() == date(valid).ok())
                .ok_or("stamp valid time missing")?;
            draw(
                &r,
                &format!("eyewall-member-{:05}", m.member_id),
                format!(
                    "{}: member {} / ancestor {}; reflectivity",
                    r.experiment_label, m.member_id, m.parent_id
                ),
                valid,
                &lat,
                &lon,
                shape,
                regrid(frame, &lat, &lon)?,
                "dBZ",
                scales::reflectivity_scale(),
                None,
                true,
            )?;
            Ok(())
        })?;
    if let Some(observed) = &r.observed_radar {
        let f: Frame = serde_json::from_slice(&fs::read(&observed).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
        if date(&f.valid_time)? != date(valid)? {
            return Err("observed radar panel must match the stamp UTC exactly".into());
        }
        draw(
            &r,
            "eyewall-observed",
            format!("{}: observed radar; grey=unobserved", r.experiment_label),
            valid,
            &lat,
            &lon,
            shape,
            regrid(&f, &lat, &lon)?,
            "dBZ",
            scales::reflectivity_scale(),
            None,
            true,
        )?;
    } else {
        fs::write(r.out.join("OBSERVED-RADAR-MISSING.txt"),"Observed radar was not supplied. Eyewall comparison is incomplete; missing offshore coverage is not clear air.\n").map_err(|e|e.to_string())?;
    }
    Ok(())
}

// Deterministic go uses the same panels and overlays as the branch products.
// The manifest carries already accumulated GPU fields, never saved-frame maxima.
#[derive(Deserialize)]
struct Accumulator {
    schema: String,
    request: serde_json::Value,
    common_wind: Common,
    fixes: Vec<Fix>,
    out: PathBuf,
}
fn render_accumulator(path: &Path) -> Result<(), String> {
    let a: Accumulator = serde_json::from_slice(&fs::read(path).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    if a.schema != "gpuwm.tc-accumulator.v1" || a.common_wind.samples < 2
        || a.fixes.len() != a.common_wind.samples || a.common_wind.shape.iter().any(|&n| n < 2) {
        return Err("every-step accumulator schema, shape or track/sample count mismatch".into());
    }
    let analysis = a.request["analysis_time"].as_str().ok_or("analysis_time missing")?;
    let r = Request {
        analysis_time: analysis.into(), observation_cutoff: analysis.into(), issue_time: analysis.into(),
        landfall_reference: a.fixes.last().ok_or("no every-step fixes")?.valid_time.clone(), out: a.out, members: vec![], best_track: PathBuf::new(),
        observed_radar: None,
        experiment_label: a.request["experiment_label"].as_str().unwrap_or("Cyclone").into(),
        model_analysis_seconds: 0., grid_id: a.request["grid_id"].as_u64().ok_or("grid_id missing")? as usize,
        wind_threshold_m_s: threshold(), onshore_threshold_m_s: onshore_threshold(),
    };
    let shape = a.common_wind.shape;
    let cells = shape[0].checked_mul(shape[1]).ok_or("accumulator grid overflow")?;
    let plane = |name: &str| read(&a.common_wind.directory.join(format!("{name}.f32")), cells);
    let lat = plane("latitude")?;
    let lon = plane("longitude")?;
    let valid = a.fixes.last().ok_or("no every-step fixes")?.valid_time.clone();
    let tracks = vec![Track {member_id: 0, parent_id: serde_json::json!("go"), weight: 1.,
        color: "#111111".into(), plotted_fixes: a.fixes.clone(), fixes: a.fixes}];
    let layers = overlay(&tracks, &[], None);
    let products = a.request["products"].as_array().ok_or("products missing")?;
    for (product, name, title, unit, scale) in [
        ("max-wind", "time_max_wind10", "Every-step maximum 10 m wind", "m s-1", scales::spread_scale(80.)),
        ("gust", "time_max_gust", "Every-step maximum diagnostic gust (surface-layer similarity)", "m s-1", scales::spread_scale(80.)),
        ("min-pressure", "time_min_mslp", "Every-step minimum sea-level pressure", "hPa", scales::radial_velocity_scale(1050.)),
        ("rain", "nest_sampled_rain", "Nest-integrated rain during sampled exposure (see exposure map)", "mm", scales::spread_scale(rain_display_limit(&a.request)?)),
        ("rain", "nest_rain_exposure_hours", "Nest rain sampling exposure", "hours", scales::spread_scale(48.)),
    ] {
        if !products.iter().any(|p| p.as_str() == Some(product)) { continue; }
        let values = plane(name)?;
        if !values.iter().any(|v| v.is_finite()) {
            return Err(format!("{name}: no finite sample on the requested Earth grid; choose a common_grid intersecting the forecast footprint"));
        }
        let mut scale = scale;
        if name == "nest_rain_exposure_hours" {
            // A fixed 48-hour legend hides partial coverage in shorter runs.
            // Equal-duration complete members retain an equal scale.
            scale = scales::spread_scale(exposure_scale_limit(&values));
        }
        if product == "min-pressure" {
            scale = minimum_pressure_scale(&values);
        }
        let output_name = if name == "nest_rain_exposure_hours" { "rain-exposure" } else { product };
        draw(&r, output_name, format!("{}: {title}", r.experiment_label), &valid,
            &lat, &lon, shape, values, unit, scale, Some(&layers), true)?;
    }
    if products.iter().any(|p| p.as_str() == Some("track")) {
        draw(&r, "track", format!("{}: pressure minimum at every committed step", r.experiment_label),
            &valid, &lat, &lon, shape, vec![f32::NAN;cells], "", scales::probability_scale(), Some(&layers), false)?;
    }
    Ok(())
}

fn minimum_pressure_scale(values:&[f32]) -> ColorScale {
    let lower=values.iter().copied().filter(|v|v.is_finite())
        .fold(930.0_f32,f32::min) as f64;
    let lower=(lower/10.).floor()*10.;
    let mut scale=scales::spread_scale(1050.-lower);
    if let ColorScale::Discrete(ref mut discrete)=scale {
        discrete.levels.iter_mut().for_each(|v|*v+=lower);
        discrete.extend=rustwx_render::ExtendMode::Both;
    }
    scale
}

#[test]
fn pressure_swath_keeps_the_deepest_native_core_visible() {
    let ColorScale::Discrete(scale)=minimum_pressure_scale(&[f32::NAN,892.5,1001.]) else {panic!("discrete pressure scale")};
    assert!(scale.levels[0]<=892.5);
    assert!(*scale.levels.last().unwrap()>=1001.);
    assert_eq!(scale.mask_below,None);
    assert_eq!(scale.extend,rustwx_render::ExtendMode::Both);
}

fn exposure_scale_limit(values: &[f32]) -> f64 {
    values.iter().copied().filter(|v| v.is_finite())
        .fold(0.0_f32, f32::max).max(f32::EPSILON) as f64
}

// Supplemental strip views can resolve light rain without changing the field.
// The default overview retains its existing ceiling; every legend names its limit.
fn rain_display_limit(request: &serde_json::Value) -> Result<f64, String> {
    match request.get("display_rain_max_mm") {
        None => Ok(300.),
        Some(value) => value.as_f64().filter(|v| v.is_finite() && *v > 0.)
            .ok_or_else(|| "display_rain_max_mm must be a finite positive number".into()),
    }
}

#[test]
fn light_rain_strip_view_keeps_explicit_scale_and_default() {
    assert_eq!(rain_display_limit(&serde_json::json!({})).unwrap(), 300.);
    assert_eq!(rain_display_limit(&serde_json::json!({"display_rain_max_mm": 30.})).unwrap(), 30.);
    for value in [serde_json::json!(0), serde_json::json!(-30), serde_json::json!("30")] {
        assert!(rain_display_limit(&serde_json::json!({"display_rain_max_mm": value})).is_err());
    }
}

#[test]
fn six_hour_exposure_legend_resolves_partial_strips() {
    assert_eq!(exposure_scale_limit(&[f32::NAN, 0.0, 0.0020833333, 3.0, 6.0]), 6.0);
    assert_eq!(exposure_scale_limit(&[0.0, 12.0, 48.0]), 48.0);
}

pub fn try_cli(args: &[String]) -> Option<Result<(), String>> {
    let flag = args.first()?;
    if flag == "--tc-accumulator-abi" {
        println!("gpuwm.tc-accumulator.v1");
        return Some(Ok(()));
    }
    if flag == "--tc-accumulator" {
        return Some(if args.len() == 2 {render_accumulator(Path::new(&args[1]))}
            else {Err("usage: rw_wrfbatch --tc-accumulator RECEIPT.json".into())});
    }
    if flag == "--tc-track-landfalls" {
        return Some(if args.len() == 3 {
            track_landfalls(Path::new(&args[1]), Path::new(&args[2]))
        } else {
            Err("usage: rw_wrfbatch --tc-track-landfalls TRACKS.json OUT.json".into())
        });
    }
    if flag == "--tc-track-plume" {
        return Some(if args.len() == 3 {
            render_plume(Path::new(&args[1]), Path::new(&args[2]))
        } else {
            Err("usage: rw_wrfbatch --tc-track-plume TRACKS.json OUT.png".into())
        });
    }
    if flag != "--cyclone-products" && flag != "--cyclone-radar" {
        return None;
    }
    Some(if args.len() != 2 {
        Err(format!("usage: rw_wrfbatch {flag} REQUEST.json"))
    } else if flag == "--cyclone-radar" {
        prepare_radar(Path::new(&args[1]))
    } else {
        run(Path::new(&args[1]))
    })
}

#[derive(Deserialize)]
struct RadarRequest {
    out: PathBuf,
    volumes: Vec<PathBuf>,
    valid_time: String,
    south: f64,
    north: f64,
    west: f64,
    east: f64,
    nx: usize,
    ny: usize,
    spacing_m: f64,
}
fn write_f32(path: &Path, values: &[f32]) -> Result<(), String> {
    fs::write(
        path,
        values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    )
    .map_err(|e| e.to_string())
}
fn prepare_radar(path: &Path) -> Result<(), String> {
    use rw_nexrad::grid_ref::{grid_ref, GridParams, GridRefRequest, ModelGrid, Reduce};
    use sha2::{Digest, Sha256};
    let r: RadarRequest = serde_json::from_slice(&fs::read(path).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    if r.nx < 2 || r.ny < 2 || !r.spacing_m.is_finite() || r.spacing_m <= 0. {
        return Err("invalid radar grid dimensions or spacing".into());
    }
    fs::create_dir_all(&r.out).map_err(|e| e.to_string())?;
    let cells = r.nx * r.ny;
    let center_lat = (r.south + r.north) / 2.;
    let center_lon = (r.west + r.east) / 2.;
    let projection = serde_json::json!({"kind":"mercator","ref_lat":center_lat,"ref_lon":center_lon,"truelat1":center_lat,"truelat2":center_lat,"stand_lon":center_lon,
        "dx":r.spacing_m,"dy":r.spacing_m,"e_we":r.nx+1,"e_sn":r.ny+1,"known_x":(r.nx as f64+1.)/2.,"known_y":(r.ny as f64+1.)/2.,"moad_cen_lat":center_lat,"moad_cen_lon":center_lon});
    let z: Vec<u8> = (0..2 * cells)
        .into_par_iter()
        .map(|k| if k < cells { 0_f64 } else { 20000_f64 }.to_le_bytes())
        .collect::<Vec<_>>()
        .into_iter()
        .flatten()
        .collect();
    fs::write(r.out.join("z_w.f64"), &z).map_err(|e| e.to_string())?;
    let descriptor = serde_json::json!({"schema":"gpuwm-obs.radar-tten-grid.v1","identity_sha256":format!("{:x}",Sha256::digest(serde_json::to_vec(&projection).unwrap())),
        "name":"fixed-Earth radar comparison","nx":r.nx,"ny":r.ny,"nz":1,"projection":projection,
        "z_w":{"file":"z_w.f64","dtype":"<f8","shape":[2,r.ny,r.nx],"sha256":format!("{:x}",Sha256::digest(&z))}});
    let grid_path = r.out.join("grid.json");
    fs::write(&grid_path, serde_json::to_vec_pretty(&descriptor).unwrap())
        .map_err(|e| e.to_string())?;
    let request = GridRefRequest {
        grid: grid_path.clone(),
        volumes: r.volumes,
        out_prefix: r.out.join("radar"),
        out_data: None,
        reduce: Reduce::Max,
        params: GridParams {
            max_range_km: 300.,
            max_elevation_deg: 20.,
        },
        threads: std::thread::available_parallelism().map_or(1, usize::from),
        window_end: Some(date(&r.valid_time)?),
        max_age_s: 900.,
    };
    let receipt = grid_ref(&request).map_err(|e| e.to_string())?;
    println!("{receipt}");
    let grid = ModelGrid::load(&grid_path)?;
    let coords: Vec<_> = (0..cells)
        .into_par_iter()
        .map(|k| {
            grid.projection
                .ij_to_latlon((k % r.nx) as f64 + 1., (k / r.nx) as f64 + 1.)
        })
        .collect();
    let lat: Vec<_> = coords.iter().map(|v| v.0 as f32).collect();
    let lon: Vec<_> = coords.iter().map(|v| v.1 as f32).collect();
    let values: Vec<_> = read(&r.out.join("radar.ref.f32"), cells)?
        .into_par_iter()
        .map(|v| {
            if v == rw_nexrad::grid_ref::NO_COVERAGE {
                f32::NAN
            } else {
                v
            }
        })
        .collect();
    write_f32(&r.out.join("latitude.f32"), &lat)?;
    write_f32(&r.out.join("longitude.f32"), &lon)?;
    write_f32(&r.out.join("refl.f32"), &values)?;
    let frame = serde_json::json!({"valid_time":r.valid_time,"path":r.out.join("frame.json"),"shape":[r.ny,r.nx],"grid_id":2,
        "observation_definition":"maximum sampled reflectivity below 20 km, latest complete volume per radar ending by nominal valid time; maximum age 900 s; no coverage remains missing",
        "source_receipt":r.out.join("radar.json")});
    fs::write(
        r.out.join("frame.json"),
        serde_json::to_vec_pretty(&frame).unwrap(),
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}

#[derive(Deserialize)]
struct PlumeFix {
    valid: String,
    lat: Option<f64>,
    lon: Option<f64>,
    mslp_pa: Option<f32>,
    max_wind10_ms: Option<f32>,
}
#[derive(Deserialize)]
struct PlumeTrack {
    #[serde(default)]
    track_id: String,
    member_id: String,
    weight: f64,
    fixes: Vec<Option<PlumeFix>>,
}
#[derive(Deserialize)]
struct Plume {
    schema: String,
    members: Vec<serde_json::Value>,
    tracks: Vec<PlumeTrack>,
    best_track: Option<serde_json::Value>,
    #[serde(default)]
    plot: serde_json::Value,
}
fn read_plume(bytes: &[u8]) -> Result<Plume, String> {
    let p: Plume = serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
    if p.schema != "gpuwm.tc-tracks.v1" {
        return Err("expected gpuwm.tc-tracks.v1".into());
    }
    for t in &p.tracks {
        if !p.members.iter().any(|m| {
            m["member_id"].as_str() == Some(&t.member_id) && m["weight"].as_f64() == Some(t.weight)
        }) {
            return Err(format!(
                "track {} has no member with its weight",
                t.member_id
            ));
        }
    }
    Ok(p)
}
fn plume_fix(f: &PlumeFix) -> Option<Fix> {
    Some(Fix {
        footprint: vec![],
        valid_time: f.valid.clone(),
        latitude: f.lat?,
        longitude: f.lon?,
        mslp_hpa: f.mslp_pa.unwrap_or(f32::NAN) / 100.,
        max_wind_m_s: f.max_wind10_ms.unwrap_or(f32::NAN),
    })
}
// Fit the occupied ground span with a 20% margin and a small local minimum.
// Unwrap around the first fix so crossing 180 degrees stays local.
fn plume_bounds(points: &[&Fix]) -> (f64, f64, f64, f64) {
    let anchor = points[0].longitude;
    let longitude = |p: &&Fix| anchor + (p.longitude - anchor + 180.).rem_euclid(360.) - 180.;
    let west = points.iter().map(longitude).fold(f64::INFINITY, f64::min);
    let east = points.iter().map(longitude).fold(f64::NEG_INFINITY, f64::max);
    let south = points.iter().map(|p| p.latitude).fold(90., f64::min);
    let north = points.iter().map(|p| p.latitude).fold(-90., f64::max);
    let mid_lat = (south + north) * 0.5;
    let cos_lat = mid_lat.to_radians().cos().abs().max(0.05);
    let half_y = ((north - south) * 0.7).max(0.2);
    let half_x = ((east - west) * cos_lat * 0.7).max(0.2);
    let half_y = half_y.max(half_x / (4. / 3.));
    let half_x = half_x.max(half_y * (4. / 3.)) / cos_lat;
    let mid_lon = (west + east) * 0.5;
    (mid_lon-half_x, mid_lon+half_x, (mid_lat-half_y).max(-90.), (mid_lat+half_y).min(90.))
}
fn render_plume(path: &Path, out: &Path) -> Result<(), String> {
    let p = read_plume(&fs::read(path).map_err(|e| e.to_string())?)?;
    let mut tracks = vec![];
    for t in &p.tracks {
        let index = p
            .members
            .iter()
            .position(|m| m["member_id"] == t.member_id)
            .unwrap();
        let mut segment = vec![];
        let label = serde_json::Value::String(t.member_id.clone());
        let add = |seg: &mut Vec<Fix>, tracks: &mut Vec<Track>| {
            if !seg.is_empty() {
                tracks.push(Track {
                    member_id: index,
                    parent_id: label.clone(),
                    weight: t.weight,
                    color: color(index),
                    fixes: seg.clone(),
                    plotted_fixes: std::mem::take(seg),
                })
            }
        };
        for f in &t.fixes {
            if let Some(f) = f.as_ref().and_then(plume_fix) {
                segment.push(f)
            } else {
                add(&mut segment, &mut tracks)
            }
        }
        add(&mut segment, &mut tracks);
    }
    let best: Vec<Fix> = p
        .best_track
        .as_ref()
        .and_then(|b| b["fixes"].as_array())
        .map(|a| {
            a.iter()
                .filter_map(|v| {
                    serde_json::from_value::<PlumeFix>(v.clone())
                        .ok()
                        .and_then(|f| plume_fix(&f))
                })
                .collect()
        })
        .unwrap_or_default();
    let points: Vec<_> = tracks.iter().flat_map(|t| &t.fixes).chain(&best).collect();
    if points.is_empty() {
        return Err("no finite model or best-track fixes to draw".into());
    }
    let (lo, hi, so, no) = plume_bounds(&points);
    let (ny, nx) = (181, 241);
    let lat: Vec<_> = (0..ny * nx)
        .map(|k| (no - (no - so) * (k / nx) as f64 / (ny - 1) as f64) as f32)
        .collect();
    let lon: Vec<_> = (0..ny * nx)
        .map(|k| (lo + (hi - lo) * (k % nx) as f64 / (nx - 1) as f64) as f32)
        .collect();
    let overlays = overlay(&tracks, &best, None);
    // A track-only canvas has no finite weather footprint to recenter.
    // Keep the requested geographic frame and overlay transform together.
    crate::panel::render_panel_bounds(PanelRequest {
        lat_deg: &lat,
        lon_deg: &lon,
        projection: None,
        ny,
        nx,
        values: vec![f32::NAN; ny * nx],
        product_slug: "tc-track-plume".into(),
        title: p.plot["title"].as_str().unwrap_or("Storm tracks").into(),
        display_units: String::new(),
        scale: scales::probability_scale(),
        cbar_tick_step: None,
        legend: LegendControls {
            density: LevelDensity::default(),
            mode: LegendMode::Stepped,
        },
        render_density: RenderDensity::default(),
        // A request may name its own labels (medicane / polar-low tracks have no agency reference);
        // otherwise the tropical-cyclone wording, keyed on whether an observed best track is drawn.
        subtitle_left: p.plot["subtitle_left"].as_str().map(String::from).unwrap_or_else(|| if best.is_empty() {
            "Forecast positions | missing positions leave gaps".into()
        } else {
            "Forecast positions | black: observed track".into()
        }),
        subtitle_center: None,
        subtitle_right: p.plot["subtitle_right"].as_str().map(String::from).unwrap_or_else(|| if best.is_empty() {
            "".into()
        } else {
            "Instantaneous model wind differs from observed 1-minute sustained wind".into()
        }),
        width: 1440,
        height: 1080,
        contours: vec![],
        colorbar: false,
        overlays: Some(&overlays),
        annotations: None,
        out_path: out.into(),
    }, Some((lo, hi, so, no)))?;
    // The legend names members and their share of the ensemble.
    plume_legend(out, &p, !best.is_empty())?;
    println!("RENDERED\t{}", out.display());
    Ok(())
}
/// Members with at least one located fix, i.e. members that have a line on the map.
/// Breakage it fixes: a partial moving ensemble's legend listed all 8 members with
/// coloured swatches while only 3 had tracks (hexcap-moving-ens ensemble-03), so the
/// legend claimed lines the map does not draw.
fn plume_members_with_fixes(p: &Plume) -> std::collections::BTreeSet<String> {
    p.tracks
        .iter()
        .filter(|t| t.fixes.iter().flatten().any(|f| f.lat.is_some() && f.lon.is_some()))
        .map(|t| t.member_id.clone())
        .collect()
}

fn plume_legend(path: &Path, p: &Plume, has_best: bool) -> Result<(), String> {
    let drawn = plume_members_with_fixes(p);
    let map = image::open(path).map_err(|e| e.to_string())?.to_rgba8();
    let rows = ((map.height() - 180) / 28).max(1) as usize;
    let cols = p.members.len().div_ceil(rows).max(1) as u32;
    let mut fig = image::RgbaImage::from_pixel(
        map.width() + cols * 360,
        map.height(),
        image::Rgba([255, 255, 255, 255]),
    );
    image::imageops::replace(&mut fig, &map, 0, 0);
    let ink = rustwx_render::Color::rgba(23, 35, 52, 255);
    rustwx_render::draw_text_line(
        &mut fig,
        "Forecast members",
        map.width() as i32 + 12,
        28,
        ink,
        2,
    );
    for (i, m) in p.members.iter().enumerate() {
        let x = map.width() + 12 + (i / rows) as u32 * 360;
        let y = 70 + (i % rows) as u32 * 28;
        let id = m["member_id"].as_str().unwrap_or("?");
        let has_track = drawn.contains(id);
        if has_track {
            let c = parse_color(&color(i));
            for yy in y + 4..y + 8 {
                for xx in x..x + 24 {
                    fig.put_pixel(xx, yy, image::Rgba([c.r, c.g, c.b, 255]))
                }
            }
        }
        rustwx_render::draw_text_line(
            &mut fig,
            &format!(
                "{} | share {:.1}% {}{}",
                id,
                100. * m["weight"].as_f64().unwrap_or(0.),
                m["stratum"].as_str().unwrap_or(""),
                if has_track { "" } else { " no positions" }
            ),
            x as i32 + 30,
            y as i32,
            ink,
            1,
        );
    }
    if has_best {
        rustwx_render::draw_text_line(
            &mut fig,
            &format!("Black: {}", p.plot["reference_label"].as_str().unwrap_or("observed best track")),
            map.width() as i32 + 12,
            map.height() as i32 - 60,
            ink,
            2,
        );
    }
    rustwx_render::save_rgba_png_profile_with_options(
        &fig,
        path,
        &rustwx_render::PngWriteOptions::default(),
    )
    .map(|_| ())
    .map_err(|e| e.to_string())
}
#[cfg(test)]
mod plume_tests {
    use super::*;
    #[test]
    fn northern_plume_keeps_track_inside_geographic_frame() {
        let dir=std::env::temp_dir().join(format!("northern-plume-{}",std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let input=dir.join("track.json");let output=dir.join("track.png");
        fs::write(&input,br#"{"schema":"gpuwm.tc-tracks.v1","members":[{"member_id":"a","weight":1.0}],"tracks":[{"member_id":"a","weight":1.0,"fixes":[{"valid":"2024-01-01T00:00:00Z","lat":72.0,"lon":5.75},{"valid":"2024-01-01T03:00:00Z","lat":76.0,"lon":30.0},{"valid":"2024-01-01T06:00:00Z","lat":68.25,"lon":49.0}]}]}"#).unwrap();
        render_plume(&input,&output).unwrap();
        let image=image::open(&output).unwrap().to_rgb8();
        // Exclude the member legend. A clipped northern map leaves no red
        // track pixels in the map panel, even though the PNG write succeeds.
        let red=image.enumerate_pixels().filter(|(x,_,p)| *x<1400 && p[0]>150 && p[1]<80 && p[2]<80).count();
        assert!(red>300,"northern track was clipped: {red} visible red pixels");
        fs::remove_file(input).unwrap();fs::remove_file(output).unwrap();
        // Native georeference sidecars are small test products in this directory.
        for entry in fs::read_dir(&dir).unwrap() { let p=entry.unwrap().path();if p.is_file(){fs::remove_file(p).unwrap();} }
        fs::remove_dir(dir).unwrap();
    }
    #[test]
    fn short_track_bounds_are_local_and_cross_the_dateline() {
        let a = Fix { footprint: vec![], valid_time: String::new(), latitude: 12., longitude: 179.9,
            mslp_hpa: 990., max_wind_m_s: 20. };
        let mut b = a.clone();
        b.latitude = 12.1;
        b.longitude = -179.9;
        let (w,e,s,n) = plume_bounds(&[&a,&b]);
        assert!(e-w < 1. && n-s < 1.);
        assert!(w < 179.9 && e > 180.1 && s < 12. && n > 12.1);
    }
    #[test]
    fn plume_reader_keeps_missing_fixes_and_member_weights() {
        let bytes=br#"{"schema":"gpuwm.tc-tracks.v1","members":[{"member_id":"a","weight":0.25}],"tracks":[{"member_id":"a","weight":0.25,"fixes":[null,{"valid":"2024-09-27T00:00:00Z","lat":29.0,"lon":-84.0,"mslp_pa":97000,"max_wind10_ms":40}]}]}"#;
        let p = read_plume(bytes).unwrap();
        assert!(p.tracks[0].fixes[0].is_none());
        assert_eq!(p.tracks[0].weight, 0.25);
        assert_eq!(
            plume_fix(p.tracks[0].fixes[1].as_ref().unwrap())
                .unwrap()
                .mslp_hpa,
            970.
        );
        assert!(read_plume(
            &String::from_utf8_lossy(bytes)
                .replace("\"weight\":0.25,\"fixes\"", "\"weight\":1.0,\"fixes\"")
                .into_bytes()
        )
        .is_err());
    }

    #[test]
    fn plume_legend_marks_members_without_a_located_fix() {
        let p = read_plume(br#"{"schema":"gpuwm.tc-tracks.v1",
            "members":[{"member_id":"m1","weight":0.5},{"member_id":"m2","weight":0.5}],
            "tracks":[{"member_id":"m1","weight":0.5,"fixes":[null,{"valid":"2022-09-28T01:00:00","lat":24.5,"lon":-83.1}]},
                      {"member_id":"m2","weight":0.5,"fixes":[null,null]}]}"#).unwrap();
        let drawn = plume_members_with_fixes(&p);
        assert!(drawn.contains("m1"));
        assert!(!drawn.contains("m2"));
    }
}

// Native coastline intersection is a sampling diagnostic, with the input
// cadence retained. It does not claim timing finer than the bracketing fixes.
fn track_landfalls(path: &Path, out: &Path) -> Result<(), String> {
    use rustwx_render::{BasemapStyle, PolygonRole};
    let p = read_plume(&fs::read(path).map_err(|e| e.to_string())?)?;
    let land: Vec<_> = rustwx_render::load_styled_basemap_polygons_for(BasemapStyle::Filled)
        .into_iter()
        .filter(|layer| layer.role == PolygonRole::Land)
        .flat_map(|layer| layer.polygons)
        .collect();
    // Derive intersections and land classification from the same boundaries.
    // Independent coastline and fill assets can disagree at short segments.
    let coast: Vec<_> = land
        .iter()
        .flat_map(|poly| poly.iter())
        .flat_map(|ring| ring.windows(2).map(|w| (w[0], w[1])))
        .collect();
    if coast.is_empty() || land.is_empty() {
        return Err(
            "landfall diagnostic requires renderer coastline and land basemap assets".into(),
        );
    }
    let in_ring = |x: f64, y: f64, ring: &Vec<(f64, f64)>| {
        let mut inside = false;
        for w in ring.windows(2) {
            let ((a, b), (c, d)) = (w[0], w[1]);
            if (b > y) != (d > y) && x < (c - a) * (y - b) / (d - b) + a {
                inside = !inside
            }
        }
        inside
    };
    let on_land = |x: f64, y: f64| {
        land.iter()
            .any(|poly| poly.iter().filter(|ring| in_ring(x, y, ring)).count() % 2 == 1)
    };
    let cross = |a: (f64, f64), b: (f64, f64)| a.0 * b.1 - a.1 * b.0;
    let mut results = vec![];
    for track in &p.tracks {
        for pair in track.fixes.windows(2) {
            let (Some(a), Some(b)) = (&pair[0], &pair[1]) else {
                continue;
            };
            let (Some(a), Some(b)) = (plume_fix(a), plume_fix(b)) else {
                continue;
            };
            let pa = (a.longitude, a.latitude);
            let pb = (b.longitude, b.latitude);
            if (pb.0 - pa.0).abs() > 180. {
                continue;
            }
            let v = (pb.0 - pa.0, pb.1 - pa.1);
            let hits: Vec<_> = coast
                .par_iter()
                .filter_map(|&(c, d)| {
                    if c.0.max(d.0) < pa.0.min(pb.0)
                        || c.0.min(d.0) > pa.0.max(pb.0)
                        || c.1.max(d.1) < pa.1.min(pb.1)
                        || c.1.min(d.1) > pa.1.max(pb.1)
                    {
                        return None;
                    }
                    let w = (d.0 - c.0, d.1 - c.1);
                    let den = cross(v, w);
                    if den.abs() < 1e-12 {
                        return None;
                    }
                    let delta = (c.0 - pa.0, c.1 - pa.1);
                    let t = cross(delta, w) / den;
                    let u = cross(delta, v) / den;
                    if !(0.0..=1.0).contains(&t) || !(0.0..=1.0).contains(&u) {
                        return None;
                    }
                    Some(t)
                })
                .collect();
            for t in hits {
                let before = (t - 1e-5).max(0.);
                let after = (t + 1e-5).min(1.);
                if on_land(pa.0 + before * v.0, pa.1 + before * v.1)
                    || !on_land(pa.0 + after * v.0, pa.1 + after * v.1)
                {
                    continue;
                }
                let ta = date(&a.valid_time)?;
                let tb = date(&b.valid_time)?;
                let time = ta
                    + chrono::Duration::seconds(((tb - ta).num_seconds() as f64 * t).round() as i64);
                results.push(serde_json::json!({"member_id":track.member_id,"track_id":track.track_id,"valid":time.to_rfc3339_opts(chrono::SecondsFormat::Secs,true),"lat":pa.1+t*v.1,"lon":pa.0+t*v.0,"bracket":[a.valid_time,b.valid_time],"bracket_hours":(tb-ta).num_seconds()as f64/3600.}));
            }
        }
    }
    fs::write(out,serde_json::to_vec_pretty(&serde_json::json!({"method":"linear interpolation between adjacent finite raw fixes, ocean-to-land crossing of renderer Natural Earth coast; no interpolation across missing fixes","crossings":results})).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
    Ok(())
}
