//! Native CF render door on the production panel and geographic overlay seams.
use crate::{
    cf_grid::{CfGrid, ProductRow, moment, product_rows, window},
    panel::{PanelRequest, layout_path, render_panel_bounds},
};
use chrono::{DateTime, Utc};
use rayon::prelude::*;
use rustwx_core::GridProjection;
use rustwx_products::geographic_overlays::{
    LineSpec, MapOverlays, ValuePointLayer, ValuePointSpec,
};
use rustwx_render::{Color, ContourLayer, LegendControls, LegendMode, RenderDensity};
use serde::Deserialize;
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    time::Instant,
};

pub const HELP: &str = "rw_wrfbatch CF physical-scale-v1 --cf FILE.nc --variable NAME --out-dir DIR [--valid YYYY-MM-DDTHH:MM:SSZ] [--bounds W,E,S,N] [--cf-table FILE.json] [--contours NAME[,NAME]] [--contour-file FILE.nc] [--observation FILE.nc|FILE.grib2.gz --observation-variable NAME] [--overlays FILE.json] [--width N --height N] [--receipt FILE.json]";

#[derive(Default)]
struct Args {
    cf: PathBuf,
    variable: String,
    out: PathBuf,
    valid: Option<DateTime<Utc>>,
    bounds: Option<(f64, f64, f64, f64)>,
    table: Option<PathBuf>,
    contours: Vec<String>,
    contour_file: Option<PathBuf>,
    observation: Option<PathBuf>,
    observation_variable: Option<String>,
    overlays: Option<PathBuf>,
    width: u32,
    height: u32,
    receipt: Option<PathBuf>,
}
fn parse(raw: impl Iterator<Item = String>) -> Result<Args, String> {
    let mut args = Args {
        width: 1200,
        height: 900,
        ..Args::default()
    };
    let mut raw = raw;
    while let Some(key) = raw.next() {
        if key == "--help" || key == "-h" {
            println!("{HELP}");
            return Err(String::new());
        }
        let value = raw.next().ok_or_else(|| format!("{key} needs a value"))?;
        match key.as_str() {
            "--cf" => args.cf = value.into(),
            "--variable" => args.variable = value,
            "--out-dir" => args.out = value.into(),
            "--valid" => args.valid = Some(moment(&value)?),
            "--cf-table" => args.table = Some(value.into()),
            "--bounds" | "--mesh-bounds" => {
                let b = value
                    .split(',')
                    .map(str::parse::<f64>)
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|e| e.to_string())?;
                if b.len() != 4
                    || b.iter().any(|v| !v.is_finite())
                    || b[0] >= b[1]
                    || b[1] - b[0] > 360.
                    || b[0] < -360.
                    || b[1] > 540.
                    || b[2] >= b[3]
                    || b[2] < -90.
                    || b[3] > 90.
                {
                    return Err("bounds require W < E with span <= 360 degrees, -360 <= W, E <= 540 and -90 <= S < N <= 90".into());
                }
                args.bounds = Some((b[0], b[1], b[2], b[3]));
            }
            "--contours" => args.contours = value.split(',').map(str::to_string).collect(),
            "--contour-file" => args.contour_file = Some(value.into()),
            "--observation" => args.observation = Some(value.into()),
            "--observation-variable" => args.observation_variable = Some(value),
            "--overlays" => args.overlays = Some(value.into()),
            "--width" => args.width = value.parse().map_err(|e| format!("width: {e}"))?,
            "--height" => args.height = value.parse().map_err(|e| format!("height: {e}"))?,
            "--receipt" => args.receipt = Some(value.into()),
            _ => return Err(format!("unknown CF option {key}")),
        }
    }
    if args.cf.as_os_str().is_empty() || args.variable.is_empty() || args.out.as_os_str().is_empty()
    {
        return Err(HELP.into());
    }
    if args.width < 400 || args.height < 300 {
        return Err("panel dimensions must be at least 400x300 for map and legend".into());
    }
    Ok(args)
}

/// CSV uses RFC-style quoting, including doubled quote escapes. Records here
/// are one line (station and track exports have no multiline text fields).
fn csv_line(line: &str) -> Result<Vec<String>, String> {
    let mut fields = Vec::new();
    let mut field = String::new();
    let mut quoted = false;
    let mut chars = line.trim_end_matches('\r').chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '"' if quoted && chars.peek() == Some(&'"') => {
                field.push('"');
                chars.next();
            }
            '"' => quoted = !quoted,
            ',' if !quoted => {
                fields.push(std::mem::take(&mut field));
            }
            _ => field.push(c),
        }
    }
    if quoted {
        return Err("CSV record has an unterminated quote or multiline field".into());
    }
    fields.push(field);
    Ok(fields)
}
fn csv(path: &Path) -> Result<Vec<BTreeMap<String, String>>, String> {
    let data = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut lines = data.lines().filter(|l| !l.trim().is_empty());
    let header = csv_line(lines.next().ok_or("empty CSV")?)?;
    lines
        .collect::<Vec<_>>()
        .par_iter()
        .map(|l| {
            let values = csv_line(l)?;
            if values.len() != header.len() {
                return Err(format!(
                    "{}: CSV row has {} columns, expected {}",
                    path.display(),
                    values.len(),
                    header.len()
                ));
            }
            Ok(header.iter().cloned().zip(values).collect())
        })
        .collect()
}
fn default_values() -> Vec<String> {
    vec!["value".into()]
}
fn default_time() -> String {
    "valid".into()
}
#[derive(Deserialize)]
struct CsvPoints {
    #[serde(default)]
    path: PathBuf,
    #[serde(default)]
    paths: Vec<PathBuf>,
    units: String,
    #[serde(default = "default_values")]
    value_columns: Vec<String>,
    #[serde(default = "default_time")]
    time_column: String,
    start: Option<String>,
    end: Option<String>,
}
#[derive(Deserialize, Default)]
struct CsvOverlays {
    #[serde(flatten)]
    maps: MapOverlays,
    #[serde(default)]
    csv_points: Vec<CsvPoints>,
    #[serde(default)]
    csv_tracks: Vec<PathBuf>,
}
fn load_overlays(
    path: Option<&Path>,
    row: &ProductRow,
    valid: DateTime<Utc>,
) -> Result<MapOverlays, String> {
    let Some(path) = path else {
        return Ok(MapOverlays::default());
    };
    let mut spec: CsvOverlays =
        serde_json::from_slice(&std::fs::read(path).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    let root = path.parent().unwrap_or(Path::new("."));
    for points in spec.csv_points {
        let factor = row.factor(&points.units)?;
        let start = points.start.as_deref().map(moment).transpose()?;
        let end = points
            .end
            .as_deref()
            .map(moment)
            .transpose()?
            .unwrap_or(valid);
        let mut stations: BTreeMap<String, ValuePointSpec> = BTreeMap::new();
        let mut paths = points.paths;
        if !points.path.as_os_str().is_empty() {
            paths.push(points.path);
        }
        if paths.is_empty() {
            return Err("csv_points requires path or paths".into());
        }
        let records = paths
            .par_iter()
            .map(|path| csv(&root.join(path)))
            .collect::<Result<Vec<_>, _>>()?;
        for record in records.into_iter().flatten() {
            if start.is_some() || record.contains_key(&points.time_column) {
                let t = moment(
                    record
                        .get(&points.time_column)
                        .ok_or("CSV missing time column")?,
                )?;
                if start.is_some_and(|s| t < s) || t > end {
                    continue;
                }
            }
            let value = points
                .value_columns
                .iter()
                .filter_map(|key| record.get(key)?.parse::<f64>().ok())
                .filter(|v| v.is_finite() && *v >= 0.)
                .reduce(f64::max);
            let Some(value) = value else { continue };
            let lat = record
                .get("lat")
                .and_then(|v| v.parse::<f64>().ok())
                .ok_or("CSV missing numeric lat")?;
            let lon = record
                .get("lon")
                .and_then(|v| v.parse::<f64>().ok())
                .ok_or("CSV missing numeric lon")?;
            if !lat.is_finite() || lat.abs() > 90. || !lon.is_finite() {
                continue;
            }
            let key = record
                .get("station")
                .cloned()
                .unwrap_or_else(|| format!("{lat},{lon}"));
            let point = ValuePointSpec {
                lat,
                lon: (lon + 180.).rem_euclid(360.) - 180.,
                value: value * factor,
            };
            stations
                .entry(key)
                .and_modify(|p| {
                    if point.value > p.value {
                        *p = point.clone();
                    }
                })
                .or_insert(point);
        }
        let points: Vec<_> = stations.into_values().collect();
        println!(
            "OVERLAY stations={} units={} window={:?}..{}",
            points.len(),
            row.display_units,
            start,
            end
        );
        spec.maps.value_layers.push(ValuePointLayer {
            products: vec![row.product.clone()],
            units: row.display_units.clone(),
            points,
            scale: None,
            radius_px: 4,
        });
    }
    for file in spec.csv_tracks {
        let mut paths: BTreeMap<String, Vec<(i64, Option<(f64, f64)>)>> = BTreeMap::new();
        for record in csv(&root.join(file))? {
            let get = |key: &str| {
                record
                    .get(key)
                    .map(String::as_str)
                    .ok_or_else(|| format!("track CSV missing {key}"))
            };
            if moment(get("valid_time")?)? > valid {
                continue;
            }
            let step = get("step")?.parse::<i64>().map_err(|e| e.to_string())?;
            let lat = get("tclat")?.parse::<f64>().map_err(|e| e.to_string())?;
            let lon = get("tclon")?.parse::<f64>().map_err(|e| e.to_string())?;
            let point =
                (lat.is_finite() && lon.is_finite() && lat.abs() <= 90. && lon.abs() <= 360.)
                    .then_some((lat, (lon + 180.).rem_euclid(360.) - 180.));
            paths
                .entry(get("path")?.to_string())
                .or_default()
                .push((step, point));
        }
        for mut track in paths.into_values() {
            track.sort_by_key(|p| p.0);
            let mut segment = Vec::new();
            let mut last = None;
            for (step, point) in track.into_iter().chain(std::iter::once((i64::MAX, None))) {
                if point.is_none()
                    || last.is_some_and(|s| step != s + 1)
                    || point.is_some_and(|p| {
                        segment
                            .last()
                            .is_some_and(|q: &(f64, f64)| (p.1 - q.1).abs() > 180.)
                    })
                {
                    if segment.len() >= 2 {
                        spec.maps.lines.push(LineSpec {
                            points: segment.clone(),
                            color: "#111111".into(),
                            width: 2,
                            closed: false,
                        });
                    }
                    segment.clear();
                }
                if let Some(point) = point {
                    segment.push(point);
                }
                last = Some(step);
            }
        }
    }
    Ok(spec.maps)
}

struct Plane {
    lat: Vec<f32>,
    lon: Vec<f32>,
    values: Vec<f32>,
    nx: usize,
    ny: usize,
}
impl Plane {
    fn crop(
        lat: &[f32],
        lon: &[f32],
        values: &[f32],
        nx: usize,
        bounds: Option<(f64, f64, f64, f64)>,
    ) -> Result<Self, String> {
        let idx = window(lat, lon, nx, bounds)?;
        let width = idx.iter().take_while(|&&k| k / nx == idx[0] / nx).count();
        Ok(Self {
            lat: idx.par_iter().map(|&k| lat[k]).collect(),
            lon: idx.par_iter().map(|&k| lon[k]).collect(),
            values: idx.par_iter().map(|&k| values[k]).collect(),
            nx: width,
            ny: idx.len() / width,
        })
    }
}
fn render(
    plane: Plane,
    row: &ProductRow,
    source: &str,
    subtitle: &str,
    contours: Vec<ContourLayer>,
    overlays: Option<&MapOverlays>,
    args: &Args,
    path: PathBuf,
) -> Result<(PathBuf, Option<rustwx_render::PanelGeoReference>), String> {
    render_panel_bounds(
        PanelRequest {
            lat_deg: &plane.lat,
            lon_deg: &plane.lon,
            projection: Some(&GridProjection::Geographic),
            nx: plane.nx,
            ny: plane.ny,
            values: plane.values,
            product_slug: row.product.clone(),
            title: format!("{} | {}", source, row.title),
            display_units: row.display_units.clone(),
            scale: row.scale()?,
            cbar_tick_step: None,
            legend: LegendControls {
                mode: LegendMode::SmoothRamp,
                ..LegendControls::default()
            },
            render_density: RenderDensity::default(),
            subtitle_left: subtitle.into(),
            subtitle_center: None,
            subtitle_right: String::new(),
            width: args.width,
            height: args.height,
            contours,
            colorbar: true,
            overlays,
            annotations: None,
            out_path: path,
        },
        args.bounds,
    )
}
fn run(mut args: Args) -> Result<(), String> {
    let whole = Instant::now();
    let rows = product_rows(args.table.as_deref())?;
    let mut grid = CfGrid::open(&args.cf, args.valid)?;
    if let Some((w,e,_,_)) = args.bounds {
        if w < -180. || e > 180. { grid.recenter((w+e)*0.5); }
    }
    let (values, row) = grid.field(&args.variable, &rows)?;
    let finite = values.par_iter().filter(|v| v.is_finite()).count();
    let range = values
        .par_iter()
        .copied()
        .filter(|v| v.is_finite())
        .map(|v| (v, v))
        .reduce(
            || (f32::INFINITY, f32::NEG_INFINITY),
            |(a, b), (c, d)| (a.min(c), b.max(d)),
        );
    println!(
        "CF grid={}x{} valid={} source={:?} finite={} range={:?} units={}",
        grid.ny, grid.nx, grid.valid, grid.source, finite, range, row.display_units
    );
    if finite == 0 {
        return Err("selected CF field has no finite values".into());
    }
    if args.observation.is_some() && args.bounds.is_none() {
        args.bounds = Some(
            rusty_weather::batch_render::native_grid_domain_from_coordinates(&grid.lat, &grid.lon)?
                .bounds,
        );
    }
    let read_seconds = whole.elapsed().as_secs_f64();
    let stage = Instant::now();
    // Keep the native data beyond the requested rectangle: a projected
    // viewport can sample outside its curved geographic boundary. Cropping
    // first leaves artificial empty wedges on an otherwise global field.
    let plane = Plane::crop(&grid.lat, &grid.lon, &values, grid.nx, None)?;
    let mut contours = Vec::new();
    let mut contour_details = Vec::new();
    let mut contour_metrics = Vec::new();
    if !args.contours.is_empty() {
        let companion = CfGrid::open(
            args.contour_file.as_deref().unwrap_or(&args.cf),
            Some(grid.valid),
        )?;
        if grid.nx != companion.nx
            || grid.ny != companion.ny
            || grid.lat != companion.lat
            || grid.lon != companion.lon
        {
            return Err("contour and fill grids differ; co-register the CF companion before contouring to avoid misplaced boundaries".into());
        }
        for name in &args.contours {
            let (values, crow) = companion.field(name, &rows)?;
            let values =
                Plane::crop(&companion.lat, &companion.lon, &values, companion.nx, None)?.values;
            let level = crow
                .contour_level
                .ok_or("contour row needs contour_level")?;
            let region = window(&companion.lat, &companion.lon, companion.nx, args.bounds)?;
            let above = region
                .par_iter()
                .filter(|&&i| values[i] as f64 >= level)
                .count();
            contour_metrics.push(serde_json::json!({"variable":name,"source":companion.source,"level":level,"region_cells":region.len(),"region_at_or_above":above}));
            contour_details.push(format!("{name}={level} ({})", companion.source));
            contours.push(ContourLayer {
                data: values,
                levels: vec![level],
                color: Color::rgba(16, 16, 16, 255),
                width: 2,
                labels: false,
                show_extrema: false,
                pattern: Default::default(),
                major_every: None,
                major_width: None,
            });
        }
    }
    let overlays = load_overlays(args.overlays.as_deref(), &row, grid.valid)?;
    let day = grid.valid.format("%Y-%m-%d").to_string();
    let stamp = grid.valid.format("%Y%m%d_%H%Mz").to_string();
    let subtitle = format!(
        "valid {}{}",
        grid.valid.format("%Y-%m-%d %H:%M UTC"),
        if args.contours.is_empty() {
            String::new()
        } else {
            format!(" | {}", contour_details.join(", "))
        }
    );
    let path = layout_path(
        &args.out,
        "cf-grid",
        &row.product,
        &day,
        &format!("cf_{}_{}", args.variable, stamp),
    );
    let (model, model_geo) = render(
        plane,
        &row,
        &grid.source,
        &subtitle,
        contours,
        Some(&overlays),
        &args,
        path,
    )?;
    println!("RENDERED {} {}", row.product, model.display());
    let model_seconds = stage.elapsed().as_secs_f64();
    let stage = Instant::now();
    let mut outputs = vec![model.clone()];
    let mut observation_info = serde_json::Value::Null;
    if let Some(file) = &args.observation {
        let (obs, source) = if file.extension().is_some_and(|s| s == "nc") {
            let obs = CfGrid::open(file, Some(grid.valid))?;
            let (values, orow) = obs.field(
                args.observation_variable
                    .as_deref()
                    .unwrap_or(&args.variable),
                &rows,
            )?;
            if orow.product != row.product || orow.display_units != row.display_units {
                return Err("observation and diagnostic select different product ladders".into());
            }
            (
                Plane::crop(&obs.lat, &obs.lon, &values, obs.nx, None)?,
                obs.source,
            )
        } else {
            let specs = crate::comparison_observations::specifications()?;
            let product = row
                .observation_product
                .as_deref()
                .ok_or("CF product row has no observation_product")?;
            let spec = specs
                .iter()
                .find(|s| s.products.iter().any(|p| p.product == product))
                .ok_or("no matching observation product table row")?;
            let field = crate::comparison_observations::load(
                spec,
                product,
                grid.valid,
                None,
                &args.out.join("reference-cache"),
                Some(file),
                true,
            )?;
            let unit = &spec
                .products
                .iter()
                .find(|p| p.product == product)
                .unwrap()
                .units;
            let factor = row.factor(unit)? as f32;
            let values: Vec<_> = field.values.par_iter().map(|v| v * factor).collect();
            observation_info = serde_json::json!({"path":file,"sha256":field.sha256,"valid":field.valid.to_rfc3339(),"nx":field.nx,"ny":field.ny,"missing_cells":field.missing_cells});
            (
                Plane::crop(&field.lat, &field.lon, &values, field.nx, None)?,
                format!("{} | {}", spec.label, product),
            )
        };
        let path = layout_path(
            &args.out,
            "cf-grid",
            &row.product,
            &day,
            &format!("observed_{}_{}", row.product, stamp),
        );
        let (observed, observed_geo) =
            render(obs, &row, &source, &subtitle, Vec::new(), None, &args, path)?;
        if observation_info.is_null() {
            observation_info = serde_json::json!({"path":file});
        }
        observation_info["georeference"] =
            serde_json::to_value(observed_geo).map_err(|e| e.to_string())?;
        let panels = [&model, &observed]
            .iter()
            .map(|p| {
                image::ImageReader::open(p)
                    .map_err(|e| e.to_string())?
                    .decode()
                    .map(|i| i.to_rgba8())
                    .map_err(|e| e.to_string())
            })
            .collect::<Result<Vec<_>, _>>()?;
        let sheet = crate::compare::compose_sheet(
            &panels,
            &crate::compare::SheetHeader {
                title: format!(
                    "{} ({}) | diagnostic and observation",
                    row.title, row.display_units
                ),
                subtitle: format!("{subtitle} | same ladder and extent; native grids"),
            },
        )?;
        let sheet_path = layout_path(
            &args.out,
            "cf-grid",
            &row.product,
            &day,
            &format!("comparison_{}_{}", row.product, stamp),
        );
        rustwx_render::save_rgba_png_profile_with_options(&sheet, &sheet_path, &Default::default())
            .map_err(|e| e.to_string())?;
        println!("RENDERED observation {}", observed.display());
        println!("RENDERED comparison {}", sheet_path.display());
        outputs.extend([observed, sheet_path]);
    }
    let receipt = serde_json::json!({"source_revision":env!("GPUWM_BRIDGE_SOURCE_REV"),"input":args.cf,"variable":args.variable,"source":grid.source,"valid":grid.valid.to_rfc3339(),"grid":[grid.ny,grid.nx],"finite":finite,"range":range,"units":row.display_units,"bounds":args.bounds,"contours":contour_details,"contour_metrics":contour_metrics,"georeference":model_geo,"station_markers":overlays.value_layers.iter().map(|l|l.points.len()).sum::<usize>(),"track_segments":overlays.lines.len(),"observation":observation_info,"outputs":outputs,"timings_seconds":{"read":read_seconds,"model_render":model_seconds,"observation_render_compose":stage.elapsed().as_secs_f64(),"total":whole.elapsed().as_secs_f64()}});
    if let Some(file) = args.receipt {
        if let Some(p) = file.parent() {
            std::fs::create_dir_all(p).map_err(|e| e.to_string())?;
        }
        std::fs::write(file, serde_json::to_vec_pretty(&receipt).unwrap())
            .map_err(|e| e.to_string())?;
    }
    println!("FINISHED {receipt}");
    Ok(())
}
pub fn main() -> std::process::ExitCode {
    match parse(std::env::args().skip(1)).and_then(run) {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) if e.is_empty() => std::process::ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("CF render: {e}");
            std::process::ExitCode::FAILURE
        }
    }
}
