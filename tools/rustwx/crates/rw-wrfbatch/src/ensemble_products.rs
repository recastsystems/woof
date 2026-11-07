//! Resident ensemble probability planes, read without member histories.
//!
//! The producer has already reduced the resident member roster on the GPU.
//! This reader validates that contract and passes each aggregate plane through
//! the same production panel renderer and probability scale as `rw_ensbatch`.

use std::{path::{Path, PathBuf}, time::Instant, io::{BufWriter, Write}};
use netcrust::File;
use rayon::prelude::*;
use rustwx_render::{ContourLayer, LegendControls, LegendMode, LevelDensity, RenderDensity, advisory};
use crate::{panel::{PanelRequest, layout_path, render_panel_with_rgba, safe_component}, scales};

pub const ABI: &str = "gpuwm-rw-wrfbatch-resident-ensemble-v2\tCDF5\tmean\tspread\tmin\tmax\tprobability\tpaintball\tpostage\tfraction\ttyped-diagnostic\ts2s-window\tglobal-ensemble-projection\tRENDERED\tFAILED";
const CONTRACT: &str = "gpuwm-resident-ensemble-products.v1";

fn attr(file: &File, name: &str) -> Result<String, String> {
    file.attribute(name).and_then(|v| v.as_string().map(str::to_string))
        .ok_or_else(|| format!("ensemble product is missing string attribute {name}"))
}

fn shaped(file: &File, name: &str, expected: &[usize]) -> Result<Vec<f32>, String> {
    let variable = file.variable(name).ok_or_else(|| format!("missing variable {name}"))?;
    let names: Vec<&str> = variable.dimensions().iter().map(|d| d.name()).collect();
    let threshold_dimension = name.strip_suffix("_probability").map(|f| format!("{f}_threshold"));
    let wanted = if let Some(ref threshold) = threshold_dimension {
        vec![threshold.as_str(), "south_north", "west_east"]
    } else { vec!["south_north", "west_east"] };
    if names != wanted {
        return Err(format!("{name} has dimensions {names:?}, expected {wanted:?}; equal square extents do not permit transposition"));
    }
    let array = file.read_array_f64(name).map_err(|e| format!("read {name}: {e}"))?;
    if array.shape() != expected {
        return Err(format!("{name} has shape {:?}, expected {expected:?}; dimensions cannot be interchanged", array.shape()));
    }
    Ok(array.values().iter().map(|&v| v as f32).collect())
}

// S2S window means and anomalies have declared units and fixed comparison
// scales. The generic 0..100 fallback would hide every height/MSLP mean.
fn s2s_scale(field: &str, kind: &str) -> Option<rustwx_render::ColorScale> {
    let anomaly = field.ends_with("_anomaly");
    let base = field.strip_suffix("_anomaly").unwrap_or(field);
    let (low, high, delta, spread) = match base {
        "temperature2" => (240.0, 320.0, 10.0, 10.0),
        "precipitation_rate" | "freezing_rain_rate" => (0.0, 10.0, 5.0, 5.0),
        "z500" => (4500.0, 6200.0, 300.0, 150.0),
        "mslp" => (960.0, 1040.0, 20.0, 15.0),
        _ => return None,
    };
    if kind == "spread" { return Some(scales::spread_scale(spread)); }
    if anomaly {
        use rustwx_render::difference::{difference_scale, DifferenceTable};
        return Some(rustwx_render::ColorScale::Discrete(difference_scale(DifferenceTable::builtin(), delta)));
    }
    if base == "temperature2" { return None; } // canonical operational temperature palette
    let mut scale = scales::spread_scale(high - low);
    if let rustwx_render::ColorScale::Discrete(ref mut ladder) = scale {
        for level in &mut ladder.levels { *level += low; }
        ladder.extend = rustwx_render::ExtendMode::Both;
    }
    Some(scale)
}


fn is_ice(field: &str) -> bool {
    matches!(field, "ice_accretion" | "ice_accretion_horizontal")
}
fn ice_scale() -> rustwx_render::ColorScale {
    use rustwx_render::{ColorScale, DiscreteColorScale, ExtendMode, Color};
    ColorScale::Discrete(DiscreteColorScale {
        levels: vec![0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 1.0, 1.5, 2.0],
        colors: [[220,240,255],[150,210,245],[80,165,220],[45,105,180],
                 [95,60,170],[155,45,155],[210,45,115],[250,90,75]]
            .iter().map(|c| Color::rgba(c[0],c[1],c[2],255)).collect(),
        extend: ExtendMode::Max, mask_below: Some(0.01),
    })
}
fn threshold_display(field: &str, threshold: f64, units: &str) -> (String, String) {
    if is_ice(field) { return (format!("{:.2}", threshold / 25.4), "in".into()); }
    if field == "temperature2_below_freezing" {
        return (format!("{:.2}", if (threshold - 273.15).abs() < 0.0001 { 0.0 } else { threshold - 273.15 }), "C".into());
    }
    (format!("{}", threshold as f32), units.into())
}

fn field_title(field: &str) -> &str {
    match field {
        "qpf_1h" => "1 h precipitation", "qpf_3h" => "3 h precipitation",
        "qpf_6h" => "6 h precipitation", "rain_total" => "forecast precipitation",
        "wind10" => "10 m wind speed", "gust" => "wind gust",
        "temperature2" | "temperature2_below_freezing" => "2 m temperature",
        "freezing_rain_rate" => "saved-frame freezing-rain rate",
        "freezing_rain_total" => "saved-frame freezing-rain liquid",
        "ice_accretion" => "saved-frame FRAM radial ice",
        "ice_accretion_horizontal" => "saved-frame FRAM horizontal ice", "dewpoint2" => "2 m dewpoint",
        "refl" => "composite reflectivity", "uh" => "2-5 km updraft helicity",
        "uh01" => "0-1 km updraft helicity", "uh02" => "0-2 km updraft helicity",
        "uh03" => "0-3 km updraft helicity",
        "humidity2" => "2 m relative humidity",
        "z500" => "500 hPa geopotential height", "mslp" => "mean sea level pressure",
        "precipitation_rate" => "precipitation rate",
        "temperature2_anomaly" => "2 m temperature anomaly",
        "z500_anomaly" => "500 hPa geopotential height anomaly",
        "mslp_anomaly" => "mean sea level pressure anomaly",
        "precipitation_rate_anomaly" => "precipitation rate anomaly", other => other,
    }
}

fn field_recipe(field: &str) -> &str {
    match field {
        "temperature2" | "temperature2_below_freezing" => "2m_temperature", "dewpoint2" => "2m_dewpoint",
        "humidity2" => "2m_relative_humidity", "wind10" => "10m_wind_speed_and_direction",
        "gust" => "10m_wind_gusts", "rain_total" => "total_qpf",
        "refl" => "composite_reflectivity", "uh" => "uh_2to5km",
        "uh01" => "uh_0to1km", "uh02" => "uh_0to2km", "uh03" => "uh_0to3km", other => other,
    }
}

fn field_style(field: &str, units: &str) -> Option<rustwx_products::viewer::StoreVariableStyle> {
    use rustwx_core::{CanonicalField as F, FieldSelector as S};
    let selector = match field {
        "refl" => S::entire_atmosphere(F::CompositeReflectivity),
        "temperature2" | "temperature2_below_freezing" => S::height_agl(F::Temperature, 2),
        "dewpoint2" => S::height_agl(F::Dewpoint, 2),
        "humidity2" => S::height_agl(F::RelativeHumidity, 2),
        "wind10" => S::height_agl(F::WindSpeed, 10),
        "gust" => S::height_agl(F::WindGust, 10),
        "rain_total" | "freezing_rain_total" | "qpf_1h" | "qpf_3h" | "qpf_6h" => S::surface(F::TotalPrecipitation),
        "uh" => S::height_layer_agl(F::UpdraftHelicity, 2000, 5000),
        "uh01" => S::height_layer_agl(F::UpdraftHelicity, 0, 1000),
        "uh02" => S::height_layer_agl(F::UpdraftHelicity, 0, 2000),
        "uh03" => S::height_layer_agl(F::UpdraftHelicity, 0, 3000),
        _ => return rustwx_products::viewer::operational_style_for_store_variable(
            field, &serde_json::json!({"derived": field_recipe(field)}), units, rustwx_core::ModelId::WrfGdex),
    };
    let name = if field.starts_with("qpf_") { "apcp_1h" } else { field };
    rustwx_products::viewer::operational_style_for_store_variable(
        name, &serde_json::to_value(selector).ok()?, units,
        rustwx_core::ModelId::WrfGdex)
}

/// Exact typed transport for bounded diagnostic replay. The Rust decoder owns
/// the NetCDF parsing; the caller receives little-endian float32 stored words.
fn dump_diagnostic(args: &[String]) -> Result<(), String> {
    let path = args.get(1).ok_or("--ensemble-diagnostic-dump requires a CDF5 path")?;
    let mut field = None; let mut output = None; let mut row = 0usize; let mut rows = None;
    let mut i = 2;
    while i < args.len() {
        let value = args.get(i + 1).ok_or_else(|| format!("{} requires a value", args[i]))?;
        match args[i].as_str() {
            "--field" => field = Some(value.clone()), "--output" => output = Some(PathBuf::from(value)),
            "--row" => row = value.parse().map_err(|_| "invalid --row")?,
            "--rows" => rows = Some(value.parse::<usize>().map_err(|_| "invalid --rows")?),
            other => return Err(format!("unknown diagnostic dump option {other}")),
        }
        i += 2;
    }
    let file = netcrust::open(path).map_err(|e| e.to_string())?;
    if attr(&file, "ensemble_diagnostic_contract")? != "gpuwm-ensemble-diagnostic-spool.v1" {
        return Err("diagnostic replay requires the diagnostic spool contract".into());
    }
    let field = field.ok_or("--field is required")?;
    let variable = file.variable(&field).ok_or_else(|| format!("missing diagnostic {field}"))?;
    let names: Vec<_> = variable.dimensions().iter().map(|d| d.name()).collect();
    let shape = variable.shape();
    if names != ["pack_member", "south_north", "west_east"] || shape.len() != 3 {
        return Err("diagnostic dimensions must be pack_member,south_north,west_east".into());
    }
    let rows = rows.ok_or("--rows is required")?;
    if rows == 0 || row.checked_add(rows).is_none_or(|end| end > shape[1]) {
        return Err("diagnostic row selection exceeds the stored grid".into());
    }
    use netcrust::{NcSliceInfo, NcSliceInfoElem::Slice};
    let selection = NcSliceInfo { selections: vec![
        Slice {start: 0, end: shape[0] as u64, step: 1},
        Slice {start: row as u64, end: (row + rows) as u64, step: 1},
        Slice {start: 0, end: shape[2] as u64, step: 1}] };
    let array = file.read_array_slice::<f32>(&field, &selection).map_err(|e| e.to_string())?;
    let output = output.ok_or("--output is required")?;
    let mut writer = BufWriter::new(std::fs::File::create(&output).map_err(|e| e.to_string())?);
    for value in &array { writer.write_all(&value.to_bits().to_le_bytes()).map_err(|e| e.to_string())?; }
    writer.flush().map_err(|e| e.to_string())?;
    println!("{}", serde_json::json!({"schema":"gpuwm-ensemble-diagnostic-dump.v1", "dtype":"<f4",
        "shape":array.shape(), "output":output, "row":row, "rows":rows}));
    Ok(())
}

fn validate_probability(values: &[f32], counts: &[f32], members: usize) -> Result<(), String> {
    for (&v, &n) in values.iter().zip(counts.iter().cycle()) {
        if !n.is_finite() || n.fract() != 0.0 || n < 0.0 || n > members as f32 {
            return Err("finite_count is outside the declared member roster".into());
        }
        if n == members as f32 {
            if !v.is_finite() || !(0.0..=1.0).contains(&v) {
                return Err("complete-roster probability is missing or outside [0,1]".into());
            }
        } else if !v.is_nan() {
            return Err("incomplete-roster probability must be masked; its denominator cannot change".into());
        }
    }
    Ok(())
}

/// The subtitle row of a full-size ensemble map: the roster and the valid
/// time on the left, the source label on the right.
fn roster_subtitles(members: usize, valid: &str, source: &str) -> (String, String) {
    // One plane is a deterministic field (a single run or an observation),
    // not an ensemble of one: no roster to name.
    if members == 1 {
        return (format!("valid {valid}"), format!("source: {source}"));
    }
    (format!("{} | valid {valid}", roster_words(members)), format!("source: {source}"))
}

fn roster_words(members: usize) -> String {
    if members == 1 { "1 member".to_string() } else { format!("{members} members") }
}

/// What the subtitle row of one postage stamp may say, longest line first.
///
/// A stamp is a 480 px panel. The roster line and the source label do not
/// fit its row together, and the cut took the valid time: every stamp read
/// "2 members | va...". The stamp's title already names its member and the
/// full-size maps of the same frame carry the roster and the source, so
/// the stamp's row carries the valid time alone. Where the map frame is
/// narrow inside the stamp (a 500x400 grid at 12 km) even "valid TIME"
/// lost its seconds, so the bare time is the second line a stamp tries.
fn stamp_valid_lines(valid: &str) -> [String; 2] {
    [format!("valid {valid}"), valid.to_string()]
}

/// Draw one stamp with the longest valid-time line its row holds.
///
/// `cut` counts the lines the earlier stamps of this file found too long.
/// Every stamp of a file has the same frame, so a line that was cut once
/// is not drawn again. `draw` receives the (left, right) subtitles and
/// draws the stamp; a stamp drawn with a cut line is drawn again over
/// itself with the next one, and its warning is not said. A stamp whose
/// shortest line is still cut keeps it, and the renderer's own warning is
/// said as for any other map.
fn stamp<F>(cut: &mut usize, valid: &str, draw: F) -> Result<PathBuf, String>
where F: Fn((String, String)) -> Result<PathBuf, String> {
    let lines = stamp_valid_lines(valid);
    loop {
        let line = &lines[(*cut).min(lines.len() - 1)];
        let cut_key = advisory::subtitle_truncated_key("left", line);
        let (drawn, said) = advisory::hold(|| draw((line.clone(), String::new())));
        let was_cut = said.iter().any(|advice| advice.once_key.as_deref() == Some(cut_key.as_str()));
        if was_cut && drawn.is_ok() && *cut + 1 < lines.len() {
            *cut += 1;
            continue;
        }
        for advice in said {
            match advice.once_key {
                Some(key) => advisory::advise_once(key, advice.line),
                None => advisory::advise(advice.line),
            }
        }
        return drawn;
    }
}

#[allow(clippy::too_many_arguments)]
fn plane(out: &Path, domain: &str, valid: &str, subtitles: (String, String),
         latitude: &[f32], longitude: &[f32], ny: usize, nx: usize,
         slug: String, title: String, values: Vec<f32>, units: String,
         scale: rustwx_render::ColorScale, contours: Vec<ContourLayer>,
         colorbar: bool, width: u32, height: u32) -> Result<PathBuf, String> {
    plane_geo(out, domain, valid, subtitles, latitude, longitude, ny, nx, slug, title, values, units, scale,
              contours, colorbar, width, height).map(|(path, _)| path)
}

/// One drawn map's lines on stdout: `RENDERED\t<path>`, then `GEOREF\t<path>\t<json>` with its map transform
/// (one rustwx.render-georef/v1 panel), which the caller folds into the folder's `render-georef.json`.
/// THE BREAKAGE THIS CLOSES: every ensemble product map (WOOF run_6a0d19f802e9bb6c: 2,016 of them) went out with
/// no transform, so a click on it could not become a latitude and longitude and the site's soundings were off.
fn said(target: &Path, geo: Option<&rustwx_render::PanelGeoReference>) {
    println!("RENDERED\t{}", target.display());
    if let Some(json) = geo.and_then(|g| serde_json::to_string(g).ok()) {
        println!("GEOREF\t{}\t{json}", target.display());
    }
}

#[allow(clippy::too_many_arguments)]
fn plane_geo(out: &Path, domain: &str, valid: &str, subtitles: (String, String),
         latitude: &[f32], longitude: &[f32], ny: usize, nx: usize,
         slug: String, title: String, values: Vec<f32>, units: String,
         scale: rustwx_render::ColorScale, contours: Vec<ContourLayer>,
         colorbar: bool, width: u32, height: u32)
         -> Result<(PathBuf, Option<rustwx_render::PanelGeoReference>), String> {
    let day = valid.get(..10).unwrap_or(valid);
    let target = layout_path(out, domain, &slug, day,
                            &format!("{slug}_{}", safe_component(valid, "valid")));
    render_panel_with_rgba(PanelRequest {
        lat_deg: latitude, lon_deg: longitude, projection: None, ny, nx, values,
        product_slug: slug, title, display_units: units, scale, cbar_tick_step: None,
        legend: LegendControls { density: LevelDensity::default(), mode: LegendMode::Stepped },
        render_density: RenderDensity::default(),
        subtitle_left: subtitles.0, subtitle_center: None,
        subtitle_right: subtitles.1, width, height,
        contours, colorbar, overlays: MAP_OVERLAYS.get(), annotations: None, out_path: target,
    }, None)
}

/// The `--overlays` of this process's one `--ensemble-products` call.
static MAP_OVERLAYS: std::sync::OnceLock<crate::annotate::MapOverlays> = std::sync::OnceLock::new();

fn typed_shape(file: &File, name: &str, dims: &[&str], shape: &[usize]) -> Result<(), String> {
    let variable = file.variable(name).ok_or_else(|| format!("missing variable {name}"))?;
    let names: Vec<_> = variable.dimensions().iter().map(|d| d.name()).collect();
    if names != dims || variable.shape() != shape {
        return Err(format!("{name} has dimensions {names:?} and shape {:?}, expected {dims:?} {shape:?}", variable.shape()));
    }
    Ok(())
}

pub fn render(path: &Path, out: &Path, fields: &[String], domain: &str,
              source: &str, width: u32, height: u32, products: &[String]) -> Result<Vec<PathBuf>, String> {
    let file = netcrust::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    if attr(&file, "ensemble_contract")? != CONTRACT {
        return Err(format!("{} is not a {CONTRACT} aggregate file", path.display()));
    }
    if attr(&file, "probability_scale")? != "fraction" {
        return Err("probability_scale must be fraction; percent values would use the wrong colour scale".into());
    }
    let members = file.attribute("ensemble_members").and_then(|a| a.as_f64())
        .filter(|&n| n.is_finite() && n >= 1.0 && n.fract() == 0.0)
        .ok_or("ensemble_members must name the complete integer roster")? as usize;
    let valid = attr(&file, "valid_time")?;
    let is_s2s = file.attribute("s2s_window").is_some();
    let lat = file.read_array_f64("XLAT").map_err(|e| format!("read XLAT: {e}"))?;
    if lat.shape().len() != 2 || lat.shape().contains(&0) {
        return Err("XLAT must be a nonempty (south_north,west_east) plane".into());
    }
    let shape = lat.shape();
    let (ny, nx) = (shape[0], shape[1]);
    let latitude = shaped(&file, "XLAT", shape)?;
    let longitude = shaped(&file, "XLONG", shape)?;
    if latitude.iter().chain(&longitude).any(|v| !v.is_finite()) {
        return Err("ensemble product grid has nonfinite coordinates".into());
    }
    let mut selected = fields.to_vec();
    if selected.is_empty() {
        selected = file.variables().map_err(|e| e.to_string())?.iter()
            .filter_map(|v| v.name().strip_suffix("_mean").map(str::to_string)).collect();
        selected.sort();
    }
    if selected.is_empty() { return Err("aggregate file has no ensemble fields".into()); }
    // Each field writes distinct product paths and owns a native reader. The
    // old serial loop left a card waiting behind minutes of single-core plots.
    // Open a separate reader per field instead of sharing a NetCDF handle.
    let mut seen = std::collections::HashSet::new();
    selected.retain(|field| seen.insert(field.clone()));
    if selected.len() > 1 {
        let frames: Result<Vec<Vec<PathBuf>>, String> = selected.par_iter()
            .map(|field| render(path, out, std::slice::from_ref(field), domain,
                                source, width, height, products))
            .collect();
        return frames.map(|rows| rows.into_iter().flatten().collect());
    }
    let mut written = Vec::new();
    // How many of a stamp's valid-time lines this file's frame has cut.
    let mut stamp_lines_cut = 0usize;
    for field in selected {
        let mean_name = format!("{field}_mean");
        let units = file.variable(&mean_name).and_then(|v| v.attribute("units").and_then(|a| a.as_string()).map(str::to_string))
            .ok_or_else(|| format!("{mean_name} has no declared units"))?;
        let style = field_style(&field, &units);
        for kind in ["mean", "spread", "min", "max", "pmm"] {
            if !products.iter().any(|p| p == kind) || file.variable(&format!("{field}_{kind}")).is_none() { continue; }
            let mut values = shaped(&file, &format!("{field}_{kind}"), &[ny, nx])?;
            let s2s = if is_s2s || field == "freezing_rain_rate" { s2s_scale(&field, kind) } else { None };
            let (scale, display_units) = if is_ice(&field) {
                values.iter_mut().for_each(|v| *v /= 25.4);
                (if kind == "spread" { scales::spread_scale(0.5) } else { ice_scale() }, "in".into())
            } else if let Some(scale) = s2s {
                (scale, units.clone())
            } else if kind == "spread" {
                let upper = if units == "1" { 1.0 } else { match field.as_str() { "temperature2" | "dewpoint2" => 15.0,
                    "humidity2" => 40.0, "refl" => 30.0, "uh" => 150.0,
                    "uh01" | "uh02" | "uh03" => 75.0, _ => 50.0 }
                };
                (scales::spread_scale(upper), units.clone())
            } else if let Some(ref style) = style {
                values.iter_mut().for_each(|v| *v = style.convert.apply(*v));
                (style.scale.clone(), style.display_units.clone())
            } else {
                (if units == "1" { scales::probability_scale() } else { scales::spread_scale(100.0) }, units.clone())
            };
            let slug = format!("ens_{kind}_{}", safe_component(&field, "field"));
            let (target, geo) = plane_geo(out, domain, &valid, roster_subtitles(members, &valid, source), &latitude, &longitude,
                ny, nx, slug,
                if members == 1 && kind == "mean" {
                    // The plain product title for one deterministic plane.
                    format!("{} ({display_units})", field_title(&field))
                } else {
                    format!("Ensemble {kind} {} ({display_units})", field_title(&field))
                },
                values, display_units, scale, vec![], true, width, height)?;
            said(&target, geo.as_ref()); written.push(target);
        }
        if products.iter().any(|p| p == "postage") && file.variable(&format!("{field}_members")).is_some() {
            let member_name = format!("{field}_members"); let member_dim = format!("{field}_member");
            typed_shape(&file, &member_name, &[&member_dim, "south_north", "west_east"], &[members, ny, nx])?;
            let values = file.read_array::<f32>(&member_name).map_err(|e| e.to_string())?;
            let values = values.as_slice().ok_or("member diagnostic is not contiguous")?;
            for first in (0..members).step_by(16) {
                let end = (first + 16).min(members); let columns = (end - first).min(4);
                let rows = (end - first + columns - 1) / columns;
                let mut sheet = image::RgbaImage::new(columns as u32 * 480, rows as u32 * 360);
                let mut parts = Vec::new();
                let slug = format!("ens_postage_{}_page_{}", safe_component(&field, "field"), first / 16 + 1);
                for member in first..end {
                    let mut member_values = values[member * ny * nx..(member + 1) * ny * nx].to_vec();
                    let (scale, display_units) = if is_ice(&field) {
                        member_values.iter_mut().for_each(|v| *v /= 25.4);
                        (ice_scale(), "in".into())
                    } else if let Some(scale) = s2s_scale(&field, "mean") {
                        (scale, units.clone())
                    } else if let Some(ref style) = style {
                        member_values.iter_mut().for_each(|v| *v = style.convert.apply(*v));
                        (style.scale.clone(), style.display_units.clone())
                    } else { (if units == "1" { scales::probability_scale() } else { scales::spread_scale(100.0) }, units.clone()) };
                    let part_slug = format!("{slug}_member_{}", member + 1);
                    let title = format!("Member {} {} ({display_units})", member + 1, field_title(&field));
                    let part = stamp(&mut stamp_lines_cut, &valid, |subtitles| plane(out, domain, &valid, subtitles,
                        &latitude, &longitude, ny, nx, part_slug.clone(), title.clone(),
                        member_values.clone(), display_units.clone(), scale.clone(), vec![], true, 480, 360))?;
                    let pixels = image::open(&part).map_err(|e| e.to_string())?.to_rgba8();
                    let position = member - first;
                    image::imageops::overlay(&mut sheet, &pixels, (position % columns * 480) as i64, (position / columns * 360) as i64);
                    parts.push(part);
                }
                let target = layout_path(out, domain, &slug, valid.get(..10).unwrap_or(&valid),
                    &format!("{slug}_{}", safe_component(&valid, "valid")));
                if let Some(parent) = target.parent() { std::fs::create_dir_all(parent).map_err(|e| e.to_string())?; }
                sheet.save(&target).map_err(|e| e.to_string())?;
                for part in parts { std::fs::remove_file(part).map_err(|e| e.to_string())?; }
                println!("RENDERED\t{}", target.display()); written.push(target);
            }
        }
        let threshold_name = format!("{field}_thresholds");
        if file.variable(&threshold_name).is_none() { continue; }
        let thresholds = file.read_array_f64(&threshold_name).map_err(|e| format!("read {threshold_name}: {e}"))?;
        if thresholds.shape().len() != 1 || thresholds.values().is_empty() || thresholds.values().iter().any(|v| !v.is_finite()) {
            return Err(format!("{threshold_name} must be a nonempty finite threshold vector"));
        }
        let count = thresholds.values().len();
        let variable = file.variable(&threshold_name).ok_or("threshold metadata disappeared")?;
        if variable.dimensions().len() != 1 || variable.dimensions()[0].name() != format!("{field}_threshold") {
            return Err(format!("{threshold_name} does not use its own threshold dimension"));
        }
        let units = variable.attribute("units").and_then(|a| a.as_string()).ok_or("threshold units are missing")?.to_string();
        let probability_name = format!("{field}_probability");
        let probabilities = shaped(&file, &probability_name, &[count, ny, nx])?;
        let variable = file.variable(&probability_name).ok_or("probability metadata disappeared")?;
        let comparison = variable.attribute("comparison").and_then(|a| a.as_string()).ok_or("threshold comparison is missing")?;
        let relation = match comparison { "ge" => ">=", "gt" => ">", "le" => "<=", "lt" => "<", other => return Err(format!("unknown threshold comparison {other:?}")) };
        let counts = shaped(&file, &format!("{field}_finite_count"), &[ny, nx])?;
        validate_probability(&probabilities, &counts, members)?;
        for (index, &threshold) in thresholds.values().iter().enumerate() {
            // The producer stores float32 thresholds. Its shortest round-trip
            // spelling is both exact and readable on a product title.
            let threshold_label = format!("{}", threshold as f32);
            let (display_threshold, threshold_units) = threshold_display(&field, threshold, &units);
            for contour_product in ["paintball", "spaghetti"] {
            if products.iter().any(|p| p == contour_product) && file.variable(&format!("{field}_paintball")).is_some() {
                let paint_name = format!("{field}_paintball");
                let threshold_dim = format!("{field}_threshold"); let word_dim = format!("{field}_member_word");
                let words = (members + 63) / 64;
                typed_shape(&file, &paint_name, &[&threshold_dim, &word_dim, "south_north", "west_east"], &[count, words, ny, nx])?;
                let masks = file.read_array::<u64>(&paint_name).map_err(|e| e.to_string())?;
                let masks = masks.as_slice().ok_or("paintball words are not contiguous")?;
                let contours = (0..members).map(|member| ContourLayer {
                    data: (0..ny * nx).map(|cell| if counts[cell] != members as f32 { f32::NAN }
                        else { ((masks[(index * words + member / 64) * ny * nx + cell] >> (member % 64)) & 1) as f32 }).collect(),
                    levels: vec![0.5], color: crate::annotate::parse_color(rustwx_ensemble::member_color(member as u32 + 1)),
                    width: 2, labels: false, show_extrema: false, pattern: Default::default(), major_every: None, major_width: None,
                }).collect();
                let slug = format!("ens_{contour_product}_{}_{}_{}", safe_component(&field, "field"), comparison, safe_component(&threshold_label, "threshold"));
                let (target, geo) = plane_geo(out, domain, &valid, roster_subtitles(members, &valid, source), &latitude, &longitude, ny, nx, slug,
                    format!("{contour_product} {} {relation} {display_threshold} {threshold_units}", field_title(&field)),
                    vec![f32::NAN; ny * nx], units.clone(), scales::spread_scale(1.0), contours, false, width, height)?;
                said(&target, geo.as_ref()); written.push(target);
            }
            }
            if !products.iter().any(|p| p == "prob") { continue; }
            let slug = format!("ens_prob_{}_{}_{}", safe_component(&field, "field"), comparison,
                               safe_component(&threshold_label, "threshold"));
            let day = valid.get(..10).unwrap_or(&valid);
            let target = layout_path(out, domain, &slug, day,
                                     &format!("{slug}_{}", safe_component(&valid, "valid")));
            let started = Instant::now();
            let (target, geo) = render_panel_with_rgba(PanelRequest {
                lat_deg: &latitude, lon_deg: &longitude, projection: None, ny, nx,
                values: probabilities[index * ny * nx..(index + 1) * ny * nx].to_vec(),
                product_slug: slug, title: format!("P({} {relation} {display_threshold} {threshold_units})", field_title(&field)),
                display_units: "fraction".into(), scale: scales::probability_scale(), cbar_tick_step: Some(0.1),
                legend: LegendControls { density: LevelDensity::default(), mode: LegendMode::Stepped },
                render_density: RenderDensity::default(),
                subtitle_left: roster_subtitles(members, &valid, source).0, subtitle_center: None,
                subtitle_right: roster_subtitles(members, &valid, source).1, width, height,
                contours: Vec::new(), colorbar: true, overlays: None, annotations: None, out_path: target,
            }, None)?;
            said(&target, geo.as_ref());
            println!("ENSEMBLE_PLANE\t{field}\tthreshold={threshold}\tcomparison={comparison}\tmembers={members}\trender_ms={:.3}", started.elapsed().as_secs_f64() * 1000.0);
            written.push(target);
        }
    }
    Ok(written)
}

/// Additive dispatch: existing deterministic WRF invocations retain their ABI.
/// Validated presentation settings shared by the CLI and bounded resident worker.
pub struct RenderOptions {
    input: PathBuf,
    out: PathBuf,
    fields: Vec<String>,
    domain: String,
    source: String,
    width: u32,
    height: u32,
    products: Vec<String>,
}

impl RenderOptions {
    /// Parse only; this never opens a weather frame or initializes CUDA.
    pub fn parse(args: &[String]) -> Result<Self, String> {
        if args.first().map(String::as_str) != Some("--ensemble-products") {
            return Err("expected --ensemble-products".into());
        }
        let input = args.get(1).ok_or("--ensemble-products requires a CDF5 path")?;
        let mut out = None; let mut fields = Vec::new(); let mut domain = "d01".to_string();
        let mut source = concat!("WOOF ", env!("GPUWM_ENGINE_VERSION")).to_string(); let mut width = 1200; let mut height = 900;
        let mut products = vec!["mean", "spread", "min", "max", "prob", "paintball", "postage", "pmm", "spaghetti"].into_iter().map(str::to_string).collect::<Vec<_>>();
        let mut i = 2;
        while i < args.len() {
            let value = args.get(i + 1).ok_or_else(|| format!("{} requires a value", args[i]))?;
            match args[i].as_str() {
                "--out-dir" => out = Some(PathBuf::from(value)),
                "--fields" => fields = value.split(',').map(str::trim).filter(|v| !v.is_empty()).map(str::to_string).collect(),
                "--products" => products = value.split(',').map(str::trim).filter(|v| !v.is_empty()).map(str::to_string).collect(),
                "--domain" => domain = value.clone(), "--source-label" => source = value.clone(),
                "--width" => width = value.parse().map_err(|_| "invalid --width")?,
                "--height" => height = value.parse().map_err(|_| "invalid --height")?,
                // Geographic lines, labels and rings drawn over every map this
                // call paints (the same schema as rw_wrfbatch --overlays): a
                // nest's box and name over its parent, a track, a site.
                "--overlays" => {
                    let loaded = crate::annotate::MapOverlays::load(std::path::Path::new(value))?;
                    let _ = MAP_OVERLAYS.set(loaded);
                }
                other => return Err(format!("unknown ensemble product option {other}")),
            }
            i += 2;
        }
        if width < 160 || height < 120 { return Err("ensemble map dimensions must be at least 160x120".into()); }
        if products.is_empty() || products.iter().any(|p| !["mean", "spread", "min", "max", "prob", "paintball", "postage", "pmm", "spaghetti"].contains(&p.as_str())) {
            return Err("ensemble products must name mean,spread,min,max,prob,paintball,postage".into());
        }
        Ok(Self { input: PathBuf::from(input), out: out.ok_or("--out-dir is required")?, fields, domain, source, width, height, products })
    }

    pub fn render_at(&self, input: &Path, out: &Path) -> Result<(), String> {
        render(input, out, &self.fields, &self.domain, &self.source, self.width, self.height, &self.products).map(|_| ())
    }
}

pub fn try_cli(args: &[String]) -> Option<Result<(), String>> {
    if args.first().map(String::as_str) == Some("--ensemble-sheet") {
        return Some(compose_family_sheet(args));
    }
    if args.first().map(String::as_str) == Some("--ensemble-diagnostic-reduce-abi") {
        println!("{}", crate::ensemble_reduce::ABI);
        return Some(Ok(()));
    }
    if args.first().map(String::as_str) == Some("--ensemble-diagnostic-reduce") {
        return Some(crate::ensemble_reduce::cli(args));
    }
    if args.first().map(String::as_str) == Some("--ensemble-diagnostic-dump") {
        return Some(dump_diagnostic(args));
    }
    if args.first().map(String::as_str) == Some("--ensemble-products-abi") {
        println!("{ABI}"); return Some(Ok(()));
    }
    if args.first().map(String::as_str) != Some("--ensemble-products") { return None; }
    Some(RenderOptions::parse(args).and_then(|options| options.render_at(&options.input, &options.out)))
}

fn compose_family_sheet(args: &[String]) -> Result<(), String> {
    #[derive(serde::Deserialize)]
    struct Request { rows: Vec<Vec<PathBuf>>, output: PathBuf, title: String, valid_time: String }
    let input = args.get(1).ok_or("--ensemble-sheet needs a JSON request")?;
    let request: Request = serde_json::from_slice(&std::fs::read(input).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    if request.rows.is_empty() || request.rows[0].is_empty()
        || request.rows.iter().any(|row| row.len() != request.rows[0].len()) {
        return Err("ensemble sheet needs rectangular nonempty panel rows".into());
    }
    let rows: Result<Vec<Vec<image::RgbaImage>>, String> = request.rows.par_iter().map(|row|
        row.iter().map(|path| image::open(path).map(|image| image.to_rgba8()).map_err(|e| e.to_string())).collect()
    ).collect();
    let sheet = crate::compare::compose_sheet_rows(&rows?, &crate::compare::SheetHeader {
        title: request.title, subtitle: format!("valid {} UTC", request.valid_time),
    })?;
    if let Some(parent) = request.output.parent() { std::fs::create_dir_all(parent).map_err(|e| e.to_string())?; }
    sheet.save(&request.output).map_err(|e| e.to_string())?;
    println!("RENDERED\t{}", request.output.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn roster_mask_is_checked() {
        assert!(validate_probability(&[0.3, f32::NAN], &[10.0, 9.0], 10).is_ok());
        assert!(validate_probability(&[0.3, 0.2], &[10.0, 9.0], 10).is_err());
        assert!(validate_probability(&[1.1], &[10.0], 10).is_err());
        assert!(validate_probability(&[0.5], &[11.0], 10).is_err());
    }
    #[test] fn a_full_map_keeps_its_roster_and_its_source() {
        let valid = "2026-10-02_18:00:00";
        let (left, right) = roster_subtitles(20, valid, "ArWen");
        assert_eq!(left, "20 members | valid 2026-10-02_18:00:00");
        assert_eq!(right, "source: ArWen");
        // One plane is a deterministic field, not an ensemble of one.
        assert_eq!(roster_subtitles(1, valid, "ArWen").0, "valid 2026-10-02_18:00:00");
    }
    /// A stamp row that holds `room` characters and cuts, with the map
    /// renderer's own say-once advisory, anything longer.
    fn stamp_row(room: usize, drawn: &std::cell::RefCell<Vec<String>>)
            -> impl Fn((String, String)) -> Result<PathBuf, String> + '_ {
        move |(left, right)| {
            assert!(right.is_empty(), "a source label beside it cut the valid time on a 480 px stamp");
            if left.len() > room {
                advisory::advise_once(advisory::subtitle_truncated_key("left", &left),
                                      format!("warning: the left subtitle does not fit: {left}"));
            }
            drawn.borrow_mut().push(left);
            Ok(PathBuf::from("stamp.png"))
        }
    }
    #[test] fn a_postage_stamp_takes_the_longest_valid_line_its_row_holds() {
        let valid = "2026-10-03_01:00:00";
        let long = format!("valid {valid}");
        // A row with room for the whole line: one draw, nothing said.
        let (drawn, mut cut) = (std::cell::RefCell::new(Vec::new()), 0);
        let (result, said) = advisory::hold(|| stamp(&mut cut, valid, stamp_row(25, &drawn)));
        assert!(result.is_ok() && said.is_empty() && cut == 0);
        assert_eq!(*drawn.borrow(), [long.clone()]);
        // A narrow frame: "valid TIME" is cut, so the stamp is drawn again
        // with the bare time, and the warning about the copy that was
        // replaced is not said.
        let (drawn, mut cut) = (std::cell::RefCell::new(Vec::new()), 0);
        let (result, said) = advisory::hold(|| stamp(&mut cut, valid, stamp_row(22, &drawn)));
        assert!(result.is_ok() && said.is_empty() && cut == 1);
        assert_eq!(*drawn.borrow(), [long.clone(), valid.to_string()]);
        // The next stamp of the same file does not try the cut line again.
        let (result, said) = advisory::hold(|| stamp(&mut cut, valid, stamp_row(22, &drawn)));
        assert!(result.is_ok() && said.is_empty() && cut == 1);
        assert_eq!(drawn.borrow().len(), 3);
        assert_eq!(drawn.borrow()[2], valid);
        // A frame too narrow for the bare time keeps it, and the cut is said.
        let (drawn, mut cut) = (std::cell::RefCell::new(Vec::new()), 0);
        let (result, said) = advisory::hold(|| stamp(&mut cut, valid, stamp_row(10, &drawn)));
        assert!(result.is_ok() && cut == 1);
        assert_eq!(*drawn.borrow(), [long, valid.to_string()]);
        assert_eq!(said.len(), 1);
        assert_eq!(said[0].once_key.as_deref(), Some(advisory::subtitle_truncated_key("left", valid).as_str()));
        // A stamp that failed to draw is not drawn again.
        let mut cut = 0;
        let failed = stamp(&mut cut, valid, |_| Err::<PathBuf, String>("write PNG".into()));
        assert!(failed.is_err() && cut == 0);
    }
    #[test] fn deterministic_cli_is_not_claimed() {
        assert!(try_cli(&["--products".into(), "t2".into()]).is_none());
    }
    #[test] fn headline_fields_use_the_production_style_and_conversion() {
        for (field, units) in [("temperature2", "K"), ("dewpoint2", "K"), ("wind10", "m s-1"),
            ("humidity2", "%"), ("refl", "dBZ"), ("rain_total", "mm"), ("qpf_1h", "mm"),
            ("qpf_3h", "mm"), ("qpf_6h", "mm"), ("gust", "m s-1"), ("uh", "m2 s-2"),
            ("uh01", "m2 s-2"), ("uh02", "m2 s-2"), ("uh03", "m2 s-2")] {
            assert!(field_style(field, units).is_some(), "no production style for {field}");
        }
        assert_eq!(field_style("temperature2", "K").unwrap().display_units, "degF");
    }
    #[test] fn typed_cdf5_words_keep_high_member_bits_and_float_payloads() {
        use netcdf_writer::{Schema, NcFormat, NcType, NcWriter, VarData};
        let directory = std::env::temp_dir().join(format!("ensemble-typed-{}-{}", std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
        std::fs::create_dir_all(&directory).unwrap();
        let path = directory.join("words.nc");
        let mut schema = Schema::new(NcFormat::Cdf5);
        let n = schema.def_dim("n", 4, false).unwrap();
        let bits = schema.def_var("bits", NcType::UInt64, &[n]).unwrap();
        let floats = schema.def_var("floats", NcType::Float, &[n]).unwrap();
        let masks = [1u64, (1u64 << 53) | 3, (1u64 << 63) | 5, u64::MAX];
        let words = [0x80000000u32, 0x7fc01234, 0x00000001, 0x3f800000];
        let values = words.map(f32::from_bits);
        let mut writer = NcWriter::create(&path, schema).unwrap();
        writer.write_var(bits, VarData::U64(&masks)).unwrap();
        writer.write_var(floats, VarData::F32(&values)).unwrap();
        writer.finish().unwrap();
        let file = netcrust::open(&path).unwrap();
        assert_eq!(file.read_array::<u64>("bits").unwrap().as_slice().unwrap(), masks);
        assert_eq!(file.read_array::<f32>("floats").unwrap().iter().map(|v| v.to_bits()).collect::<Vec<_>>(), words);
        let selection = netcrust::NcSliceInfo { selections: vec![netcrust::NcSliceInfoElem::Slice {start: 1, end: 3, step: 1}] };
        assert_eq!(file.read_array_slice::<u64>("bits", &selection).unwrap().as_slice().unwrap(), &masks[1..3]);
        assert_eq!(file.read_array_slice::<f32>("floats", &selection).unwrap().iter().map(|v| v.to_bits()).collect::<Vec<_>>(), words[1..3]);
        drop(file);
        std::fs::remove_file(path).unwrap();
        std::fs::remove_dir(directory).unwrap();
    }
}
