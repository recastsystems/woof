//! Table driven hub-height products. All atmospheric I/O, interpolation,
//! integration, verification and map rendering stay in Rust/Rayon.
use crate::{
    local_import::wrf_source_times,
    panel::{render_panel, PanelRequest},
    wrf_process::wrf_projection,
};
use chrono::{DateTime, NaiveDate, Utc};
use netcdf_writer::{AttrValue, NcFormat, NcType, NcWriter, Schema, VarData};
use rayon::prelude::*;
use rustwx_calc::hubwind::{self as calc, Columns, Options, Plane, Stats, Turbine};
use rustwx_core::GridProjection;
use rustwx_render::{
    Color, ColorScale, DiscreteColorScale, ExtendMode, LegendControls, RenderDensity,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::HashSet,
    io::Write,
    path::{Path, PathBuf},
    time::Instant,
};
use wrf_core::WrfFile;
const PRODUCTS: &str = include_str!("../../../../../woof/data/hubwind_products.json");
const CURVES: &str = include_str!("../../../../../woof/data/turbine_power_curves.json");
const OBS: &str = include_str!("../../../../../woof/data/hubwind_observations.json");
#[derive(Deserialize, Serialize)]
struct Product {
    name: String,
    title: String,
    units: String,
    kind: String,
    upper: f64,
}
#[derive(Deserialize, Clone)]
#[serde(deny_unknown_fields)]
struct Observation {
    network: String,
    path: PathBuf,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    inputs: Vec<PathBuf>,
    #[serde(default = "default_record_selection")]
    render_frames: String,
    #[serde(default = "default_compute_backend")]
    compute_backend: String,
    #[serde(default)]
    verify_cuda: bool,
    out: PathBuf,
    #[serde(default)]
    receipt_prefix: String,
    #[serde(default = "default_record_selection")]
    record_selection: String,
    #[serde(default)]
    options: Options,
    #[serde(default)]
    products: Vec<String>,
    #[serde(default)]
    turbines: Vec<Turbine>,
    #[serde(default = "default_domain")]
    domain: String,
    #[serde(default = "default_width")]
    width: u32,
    #[serde(default = "default_height")]
    height: u32,
    #[serde(default)]
    observations: Vec<Observation>,
    #[serde(default = "default_distance")]
    observation_max_distance_km: f64,
}
fn default_compute_backend() -> String {
    "auto".into()
}
fn default_record_selection() -> String {
    "all".into()
}
fn default_domain() -> String {
    "d01".into()
}
fn default_width() -> u32 {
    1200
}
fn default_height() -> u32 {
    900
}
fn default_distance() -> f64 {
    25.
}
#[derive(Deserialize)]
struct ObsRow {
    id: String,
    height_m: f64,
    lat: f64,
    lon: f64,
    format: String,
    speed_column: usize,
    direction_column: isize,
}
struct Frame {
    time: i64,
    planes: Vec<Plane>,
}
struct Grid {
    lat: Vec<f32>,
    lon: Vec<f32>,
    nx: usize,
    ny: usize,
    projection: Option<GridProjection>,
}
fn err<E: std::fmt::Display>(e: E) -> String {
    e.to_string()
}
/// A domain folder or receipt prefix: one path component, never `.`/`..`.
///
/// The render tree names a grid by its spacing (`d01-2.25km`, the
/// `render.domain_token` / `native_domain_slug` spelling), so a dot inside
/// the token is allowed. Every 2.25 km or 1.5 km grid was refused here
/// with "invalid hubwind domain" and `woof go --products all` exited 2
/// at the end-of-run render (2.9.5, RTX PRO 4500/6000 boxes, 2026-10-23).
fn token(s: &str) -> bool {
    !s.is_empty()
        && !s.starts_with('.')
        && !s.contains("..")
        && s.chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-' || c == '.')
}
fn height_token(h: f64) -> String {
    format!("{h}m").replace('.', "p")
}
fn stamp(t: i64) -> String {
    DateTime::<Utc>::from_timestamp(t, 0)
        .unwrap()
        .format("%Y%m%dT%H%M%SZ")
        .to_string()
}
fn utc(t: i64) -> String {
    DateTime::<Utc>::from_timestamp(t, 0).unwrap().to_rfc3339()
}
fn masked_read(
    file: &WrfFile,
    metadata: &netcrust::File,
    name: &str,
    t: usize,
) -> Result<Vec<f64>, String> {
    let variable = metadata
        .variable(name)
        .ok_or_else(|| format!("missing native field {name}"))?;
    let mut missing = vec![9.969209968386869e36_f64];
    for key in ["_FillValue", "missing_value"] {
        if let Some(a) = variable.attribute(key).and_then(|a| a.value().as_f64_vec()) {
            missing.extend(a);
        }
    }
    let scale = variable
        .attribute("scale_factor")
        .and_then(|a| a.as_f64())
        .unwrap_or(1.);
    let offset = variable
        .attribute("add_offset")
        .and_then(|a| a.as_f64())
        .unwrap_or(0.);
    Ok(file
        .read_var(name, t)
        .map_err(err)?
        .into_par_iter()
        .map(|v| {
            if !v.is_finite() || missing.contains(&v) {
                f64::NAN
            } else {
                v * scale + offset
            }
        })
        .collect())
}
/// The receipt and map wording of the diagnosed form, so a reader never
/// takes it for the scheme's own turbulence.
const SIMILARITY_TKE_SOURCE: &str = "diagnosed (no PBL TKE): surface-layer similarity sigma_u from UST, PBLH and \
1/L (RMOL, or from HFX, T2/TH2 and PSFC): unstable u*(12+0.5 zi/|L|)^(1/3) (Panofsky et al. 1977) up to zi; \
neutral/stable 12^(1/3) u* (1-z/h)^(3/4) below h (Nieuwstadt 1984 decay), missing above; TKE = 1.5 sigma_u^2";

/// Hub TI where the PBL scheme carries no TKE (YSU, ACM2, ...).
///
/// Breakage it fixes (2.9.4 rolling pass, 2026-10-21 17:10Z): the default
/// product set draws hub TI, and under the default YSU every lead at
/// every height drew "NO VALID TI" because TI read only a scheme TKE.  The
/// wind-energy standard without a turbulence scheme is surface-layer
/// similarity for the streamwise velocity deviation sigma_u, scaled by the
/// friction velocity and the Obukhov length:
///
/// * unstable (1/L < 0), within the mixed layer z <= zi:
///   sigma_u = u* (12 + 0.5 zi/|L|)^(1/3) (Panofsky, Tennekes, Lenschow and
///   Wyngaard 1977), height independent;
/// * neutral and stable (1/L >= 0), below the boundary-layer top h:
///   sigma_u = 12^(1/3) u* (1 - z/h)^(3/4) (the neutral limit of the form
///   above, with the stable boundary layer's decay toward its top,
///   Nieuwstadt 1984); above h the form says nothing and TI stays missing.
///
/// 1/L is the history's RMOL when present, else -k g H / (rho cp theta u*^3)
/// from HFX, T2 (or TH2) and PSFC.  TKE = 1.5 sigma_u^2, so the isotropic
/// TI the products already compute, sqrt(2 TKE/3)/speed, is sigma_u/speed.
/// None when UST or PBLH is absent: TI then stays missing, as before.
fn similarity_tke(
    file: &WrfFile,
    metadata: &netcrust::File,
    t: usize,
    z: &[f64],
) -> Result<Option<Vec<f64>>, String> {
    const K: f64 = 0.4;
    const G: f64 = 9.81;
    const CP: f64 = 1004.5;
    const RD: f64 = 287.04;
    if !(file.has_var("UST") && file.has_var("PBLH")) {
        return Ok(None);
    }
    let n = file.nxy();
    let ust = masked_read(file, metadata, "UST", t)?;
    let pblh = masked_read(file, metadata, "PBLH", t)?;
    let rmol: Vec<f64> = if file.has_var("RMOL") {
        masked_read(file, metadata, "RMOL", t)?
    } else if file.has_var("HFX") && file.has_var("PSFC") && (file.has_var("TH2") || file.has_var("T2")) {
        let hfx = masked_read(file, metadata, "HFX", t)?;
        let psfc = masked_read(file, metadata, "PSFC", t)?;
        let t2 = masked_read(file, metadata, "T2", t)?;
        let theta = if file.has_var("TH2") { masked_read(file, metadata, "TH2", t)? } else { t2.clone() };
        (0..n)
            .into_par_iter()
            .map(|i| {
                let rho = psfc[i] / (RD * t2[i]);
                let u3 = ust[i].max(1e-3).powi(3);
                -K * G * hfx[i] / (rho * CP * theta[i] * u3)
            })
            .collect()
    } else {
        return Ok(None);
    };
    if ust.len() != n || pblh.len() != n || rmol.len() != n || z.len() % n != 0 {
        return Err("similarity TI: UST, PBLH or 1/L dimensions differ from the grid".into());
    }
    Ok(Some(
        (0..z.len())
            .into_par_iter()
            .map(|index| {
                let i = index % n;
                let sigma = similarity_sigma_u(ust[i], pblh[i], rmol[i], z[index]);
                1.5 * sigma * sigma
            })
            .collect(),
    ))
}

/// sigma_u (m/s) at height z (m AGL) from u* (m/s), the boundary-layer
/// depth h (m) and 1/L (1/m); NaN where the form is undefined.
fn similarity_sigma_u(u: f64, h: f64, inv_l: f64, z: f64) -> f64 {
    if !(u.is_finite() && h.is_finite() && inv_l.is_finite() && z.is_finite())
        || u < 0. || h <= 0. || z < 0.
    {
        return f64::NAN;
    }
    if inv_l < 0. {
        if z > h {
            return f64::NAN;
        }
        u * (12. + 0.5 * h * -inv_l).cbrt()
    } else {
        if z >= h {
            return f64::NAN;
        }
        12f64.cbrt() * u * (1. - z / h).powf(0.75)
    }
}

fn planes(
    file: &WrfFile,
    t: usize,
    o: &Options,
    backend: &str,
    verify_cuda: bool,
) -> Result<(Vec<Plane>, String, Value), String> {
    let metadata = netcrust::open(&file.path).map_err(err)?;
    let names = ["PH", "PHB", "HGT", "U", "V"];
    let values: Result<Vec<_>, String> = names
        .par_iter()
        .map(|name| masked_read(file, &metadata, name, t))
        .collect();
    let mut values = values?.into_iter();
    let ph = values.next().unwrap();
    let phb = values.next().unwrap();
    let terrain = values.next().unwrap();
    let uraw = values.next().unwrap();
    let vraw = values.next().unwrap();
    let n = file.nxy();
    let nx = file.nx;
    let ny = file.ny;
    let nz = file.nz;
    if ph.len() != n * (nz + 1)
        || phb.len() != ph.len()
        || terrain.len() != n
        || uraw.len() != nz * ny * (nx + 1)
        || vraw.len() != nz * (ny + 1) * nx
    {
        return Err("hubwind geopotential, terrain or staggered wind dimensions differ".into());
    }
    let z: Vec<_> = (0..n * nz)
        .into_par_iter()
        .map(|i| (ph[i] + phb[i] + ph[i + n] + phb[i + n]) * 0.5 / 9.80665 - terrain[i % n])
        .collect();
    let u: Vec<_> = (0..n * nz)
        .into_par_iter()
        .map(|i| {
            let x = i % nx;
            let row = i / nx;
            let a = row * (nx + 1) + x;
            (uraw[a] + uraw[a + 1]) * 0.5
        })
        .collect();
    let v: Vec<_> = (0..n * nz)
        .into_par_iter()
        .map(|i| {
            let level = i / n;
            let rem = i % n;
            let a = level * (ny + 1) * nx + rem;
            (vraw[a] + vraw[a + nx]) * 0.5
        })
        .collect();
    let (sin, cos) = rayon::join(
        || {
            if file.has_var("SINALPHA") {
                masked_read(file, &metadata, "SINALPHA", t)
            } else {
                file.sinalpha(t).map(|v| v.to_vec()).map_err(err)
            }
        },
        || {
            if file.has_var("COSALPHA") {
                masked_read(file, &metadata, "COSALPHA", t)
            } else {
                file.cosalpha(t).map(|v| v.to_vec()).map_err(err)
            }
        },
    );
    let (sin, cos) = (sin?, cos?);
    #[derive(Deserialize)]
    struct TkeSource {
        variable: String,
        multiplier: f64,
        description: String,
    }
    let sources: Vec<TkeSource> = serde_json::from_str(include_str!(
        "../../../../../woof/data/hubwind_tke_sources.json"
    ))
    .map_err(err)?;
    let (tke, tke_source): (Option<Vec<f64>>, String) =
        if let Some(row) = sources.iter().find(|s| file.has_var(&s.variable)) {
            (
                Some(
                    masked_read(file, &metadata, &row.variable, t)?
                        .into_par_iter()
                        .map(|v| v * row.multiplier)
                        .collect(),
                ),
                format!("{} * {}: {}", row.variable, row.multiplier, row.description),
            )
        } else if let Some(diagnosed) = similarity_tke(file, &metadata, t, &z)? {
            (Some(diagnosed), SIMILARITY_TKE_SOURCE.into())
        } else {
            (None, "absent: TI remains missing (no PBL TKE and no UST/PBLH for the similarity form)".into())
        };
    let tke = match tke {
        Some(v) if v.len() == n * file.nz => Some(v),
        Some(v) if v.len() == n * (file.nz + 1) => Some(
            (0..n * file.nz)
                .into_par_iter()
                .map(|i| {
                    if v[i] >= 0. && v[i + n] >= 0. {
                        (v[i] + v[i + n]) * 0.5
                    } else {
                        f64::NAN
                    }
                })
                .collect(),
        ),
        Some(v) => {
            return Err(format!(
                "TKE has {} values, expected mass or vertical staggered grid",
                v.len()
            ))
        }
        None => None,
    };
    let surface = if file.has_var("U10") && file.has_var("V10") {
        let (u, v) = rayon::join(
            || masked_read(file, &metadata, "U10", t),
            || masked_read(file, &metadata, "V10", t),
        );
        let (u, v) = (u.map_err(err)?, v.map_err(err)?);
        if u.len() != n || v.len() != n {
            return Err("10 m wind dimensions differ".into());
        }
        Some((
            (0..n)
                .into_par_iter()
                .map(|i| u[i] * cos[i] - v[i] * sin[i])
                .collect::<Vec<_>>(),
            (0..n)
                .into_par_iter()
                .map(|i| u[i] * sin[i] + v[i] * cos[i])
                .collect::<Vec<_>>(),
        ))
    } else {
        None
    };
    let columns = Columns {
        cells: n,
        levels: file.nz,
        z_agl_m: &z,
        u_ms: &u,
        v_ms: &v,
        sin: &sin,
        cos: &cos,
        tke: tke.as_deref(),
        surface: surface.as_ref().map(|(u, v)| (u.as_slice(), v.as_slice())),
    };
    if backend != "cpu" {
        if let Some((planes, bytes)) =
            rustwx_calc::hubwind_cuda::try_interpolate(&columns, &o.heights_m)?
        {
            let parity = if verify_cuda {
                let reference = calc::interpolate(columns, &o.heights_m)?;
                Some(cuda_parity(&planes, &reference)?)
            } else {
                None
            };
            return Ok((
                planes,
                tke_source,
                json!({"backend":"CUDA","device_allocation_bytes":bytes,"cpu_parity":parity}),
            ));
        }
        if backend == "cuda" {
            return Err(
                "CUDA hubwind requested but no CUDA driver, NVRTC or visible device is available"
                    .into(),
            );
        }
    }
    Ok((
        calc::interpolate(columns, &o.heights_m)?,
        tke_source,
        json!({"backend":"Rust/Rayon","device_allocation_bytes":0}),
    ))
}

fn cuda_parity(actual: &[Plane], reference: &[Plane]) -> Result<Value, String> {
    let mut maxima = [0f64; 3];
    let mut compared = 0u64;
    for (a, b) in actual.iter().zip(reference) {
        for (kind, (a, b)) in [
            (&a.speed, &b.speed),
            (&a.direction, &b.direction),
            (&a.ti, &b.ti),
        ]
        .into_iter()
        .enumerate()
        {
            let check = a
                .par_iter()
                .zip(b)
                .map(|(&a, &b)| {
                    if a.is_finite() != b.is_finite() {
                        return Err("CUDA/Rust missing mask differs".to_string());
                    }
                    if !a.is_finite() {
                        return Ok((0u64, 0.));
                    }
                    let mut d = (a as f64 - b as f64).abs();
                    if kind == 1 {
                        d = d.min((360. - d).abs());
                    }
                    if d > 1e-5 * (1. + (b as f64).abs()) {
                        return Err(format!("CUDA/Rust value differs: {a} versus {b}"));
                    }
                    Ok((1, d))
                })
                .try_reduce(|| (0, 0f64), |(n, m), (a, b)| Ok((n + a, m.max(b))))?;
            compared += check.0;
            maxima[kind] = maxima[kind].max(check.1);
        }
    }
    Ok(
        json!({"compared_finite_values":compared,"max_abs_speed_ms":maxima[0],"max_abs_circular_direction_deg":maxima[1],"max_abs_ti":maxima[2],"relative_tolerance":1e-5}),
    )
}
fn stat_values(s: &[Stats], name: &str) -> Vec<f32> {
    s.par_iter()
        .map(|s| match name {
            "hub_mean" => s.mean_ms,
            "hub_max" => s.max_ms,
            "hub_exceedance_hours" => s.exceedance_hours,
            "hub_cut_in_hours" => s.cut_in_hours,
            "hub_rated_hours" => s.rated_hours,
            "hub_cut_out_hours" => s.cut_out_hours,
            "hub_below_cut_in_hours" => s.below_cut_in_hours,
            "hub_valid_hours" => s.valid_hours,
            "hub_capacity_factor" => s.capacity_factor,
            "hub_energy" => s.energy_mwh,
            "hub_ramp_up_events" if s.valid_hours > 0. => s.ramp_up_events as f32,
            "hub_ramp_down_events" if s.valid_hours > 0. => s.ramp_down_events as f32,
            "hub_power_ramp_up_events" if s.valid_hours > 0. => s.power_ramp_up_events as f32,
            "hub_power_ramp_down_events" if s.valid_hours > 0. => s.power_ramp_down_events as f32,
            "hub_max_abs_ramp" => s.max_abs_ramp_ms,
            _ => f32::NAN,
        })
        .collect()
}
fn scale(p: &Product, upper: f64) -> ColorScale {
    if p.name == "hub_shear" {
        return ColorScale::Discrete(DiscreteColorScale {
            levels: vec![-1., -0.5, -0.2, 0., 0.1, 0.2, 0.3, 0.5, 1.],
            colors: [
                [45, 70, 150],
                [75, 130, 200],
                [140, 195, 230],
                [220, 235, 245],
                [255, 240, 210],
                [255, 190, 110],
                [225, 110, 65],
                [160, 40, 45],
            ]
            .iter()
            .map(|c| Color::rgba(c[0], c[1], c[2], 255))
            .collect(),
            extend: ExtendMode::Both,
            mask_below: None,
        });
    }
    if p.name == "hub_direction" {
        return ColorScale::Discrete(DiscreteColorScale {
            levels: (0..=12).map(|i| i as f64 * 30.).collect(),
            colors: [
                [220, 55, 65],
                [240, 130, 45],
                [230, 195, 40],
                [135, 200, 65],
                [45, 180, 105],
                [40, 180, 175],
                [50, 140, 220],
                [75, 80, 205],
                [140, 65, 200],
                [195, 60, 175],
                [215, 55, 120],
                [220, 55, 65],
            ]
            .iter()
            .map(|c| Color::rgba(c[0], c[1], c[2], 255))
            .collect(),
            extend: ExtendMode::Neither,
            mask_below: None,
        });
    }
    crate::scales::spread_scale(upper)
}
fn draw(
    r: &Request,
    g: &Grid,
    p: &Product,
    values: Vec<f32>,
    suffix: &str,
    time: i64,
    upper: f64,
    detail: &str,
) -> Result<(PathBuf, Option<String>), String> {
    let slug = format!("{}_{}", p.name, suffix);
    let stamp = stamp(time);
    let out = r
        .out
        .join(&r.domain)
        .join(&slug)
        .join(&stamp[..8])
        .join(format!("{stamp}.png"));
    std::fs::create_dir_all(out.parent().unwrap()).map_err(err)?;
    // A picture with no finite value is drawn (the reader sees why) but is
    // never counted as drawn: the caller reports it as undrawn.
    let blank = if values.iter().any(|v| v.is_finite()) {
        None
    } else if p.name == "hub_ti" {
        Some("NO VALID TI: PBL TKE absent, invalid, or calm wind".to_owned())
    } else {
        Some("NO VALID DATA at this AGL height and time".to_owned())
    };
    let detail = blank.clone().unwrap_or_else(|| detail.to_owned());
    let path = render_panel(PanelRequest {
        lat_deg: &g.lat,
        lon_deg: &g.lon,
        projection: g.projection.as_ref(),
        ny: g.ny,
        nx: g.nx,
        values,
        product_slug: slug,
        title: format!("{} | {}", p.title, suffix.replace('p', ".")),
        display_units: p.units.clone(),
        scale: scale(p, upper),
        cbar_tick_step: None,
        legend: LegendControls::default(),
        render_density: RenderDensity::default(),
        subtitle_left: detail.into(),
        subtitle_center: None,
        subtitle_right: utc(time),
        width: r.width,
        height: r.height,
        contours: vec![],
        colorbar: true,
        overlays: None,
        annotations: None,
        out_path: out,
    })?;
    Ok((path, blank))
}
/// A picture with data counts as drawn; one with no finite value is kept
/// apart as undrawn with the reason it states.
fn record(maps: &mut Vec<PathBuf>, undrawn: &mut Vec<(PathBuf, String)>, drawn: (PathBuf, Option<String>)) {
    match drawn {
        (path, None) => maps.push(path),
        (path, Some(reason)) => undrawn.push((path, reason)),
    }
}
fn summary(v: &[f32]) -> Value {
    let (n, sum, max) = v
        .par_iter()
        .filter(|x| x.is_finite())
        .fold(
            || (0u64, 0f64, f32::NEG_INFINITY),
            |(n, s, m), &x| (n + 1, s + x as f64, m.max(x)),
        )
        .reduce(
            || (0, 0., f32::NEG_INFINITY),
            |(a, b, c), (d, e, f)| (a + d, b + e, c.max(f)),
        );
    json!({"valid_cells":n,"mean":if n>0{Some(sum/n as f64)}else{None},"max":if n>0{Some(max)}else{None}})
}
fn write_frame(
    path: &Path,
    g: &Grid,
    planes: &[Plane],
    time: i64,
    curve: &Turbine,
    o: &Options,
) -> Result<(), String> {
    let mut schema = Schema::new(NcFormat::Cdf5);
    let h = schema.def_dim("height", planes.len(), false).map_err(err)?;
    let y = schema.def_dim("y", g.ny, false).map_err(err)?;
    let x = schema.def_dim("x", g.nx, false).map_err(err)?;
    schema
        .put_global_attr("Conventions", AttrValue::Text("CF-1.10".into()))
        .map_err(err)?;
    schema
        .put_global_attr(
            "turbine",
            AttrValue::Text(serde_json::to_string(curve).map_err(err)?),
        )
        .map_err(err)?;
    let hv = schema
        .def_var("height", NcType::Double, &[h])
        .map_err(err)?;
    schema
        .put_var_attr(hv, "units", AttrValue::Text("m".into()))
        .map_err(err)?;
    schema
        .put_var_attr(
            hv,
            "long_name",
            AttrValue::Text("height above ground".into()),
        )
        .map_err(err)?;
    let tv = schema.def_var("time", NcType::Double, &[]).map_err(err)?;
    schema
        .put_var_attr(
            tv,
            "units",
            AttrValue::Text("seconds since 1970-01-01 00:00:00 UTC".into()),
        )
        .map_err(err)?;
    schema
        .put_var_attr(tv, "standard_name", AttrValue::Text("time".into()))
        .map_err(err)?;
    let lat = schema
        .def_var("latitude", NcType::Float, &[y, x])
        .map_err(err)?;
    let lon = schema
        .def_var("longitude", NcType::Float, &[y, x])
        .map_err(err)?;
    schema
        .put_var_attr(lat, "units", AttrValue::Text("degrees_north".into()))
        .map_err(err)?;
    schema
        .put_var_attr(lon, "units", AttrValue::Text("degrees_east".into()))
        .map_err(err)?;
    let mut ids = vec![];
    for (name, units) in [
        ("hub_speed", "m s-1"),
        ("hub_direction", "degree"),
        ("hub_ti", "1"),
        ("hub_power", "MW"),
    ] {
        let v = schema
            .def_var(name, NcType::Float, &[h, y, x])
            .map_err(err)?;
        schema
            .put_var_attr(v, "units", AttrValue::Text(units.into()))
            .map_err(err)?;
        schema
            .put_var_attr(
                v,
                "coordinates",
                AttrValue::Text("height latitude longitude time".into()),
            )
            .map_err(err)?;
        schema
            .put_var_attr(v, "_FillValue", AttrValue::Floats(vec![f32::NAN]))
            .map_err(err)?;
        ids.push(v);
    }
    let mut writer = NcWriter::create(path, schema).map_err(err)?;
    writer
        .write_var(hv, VarData::F64(&o.heights_m))
        .map_err(err)?;
    writer
        .write_var(tv, VarData::F64(&[time as f64]))
        .map_err(err)?;
    writer.write_var(lat, VarData::F32(&g.lat)).map_err(err)?;
    writer.write_var(lon, VarData::F32(&g.lon)).map_err(err)?;
    for (kind, id) in ids.into_iter().enumerate() {
        let values: Vec<_> = planes
            .iter()
            .flat_map(|p| match kind {
                0 => p.speed.clone(),
                1 => p.direction.clone(),
                2 => p.ti.clone(),
                _ => p
                    .speed
                    .par_iter()
                    .map(|&s| (curve.fraction(s as f64) * o.rated_mw) as f32)
                    .collect(),
            })
            .collect();
        writer.write_var(id, VarData::F32(&values)).map_err(err)?;
    }
    writer.finish().map_err(err)?;
    Ok(())
}
fn write_stats(
    path: &Path,
    g: &Grid,
    stats: &[Vec<Stats>],
    products: &[Product],
    o: &Options,
    curve: &Turbine,
) -> Result<(), String> {
    let mut schema = Schema::new(NcFormat::Cdf5);
    let h = schema
        .def_dim("height", o.heights_m.len(), false)
        .map_err(err)?;
    let y = schema.def_dim("y", g.ny, false).map_err(err)?;
    let x = schema.def_dim("x", g.nx, false).map_err(err)?;
    schema
        .put_global_attr("Conventions", AttrValue::Text("CF-1.10".into()))
        .map_err(err)?;
    schema
        .put_global_attr(
            "hubwind_options",
            AttrValue::Text(serde_json::to_string(o).map_err(err)?),
        )
        .map_err(err)?;
    schema
        .put_global_attr(
            "turbine",
            AttrValue::Text(serde_json::to_string(curve).map_err(err)?),
        )
        .map_err(err)?;
    let hv = schema
        .def_var("height", NcType::Double, &[h])
        .map_err(err)?;
    schema
        .put_var_attr(hv, "units", AttrValue::Text("m".into()))
        .map_err(err)?;
    schema
        .put_var_attr(
            hv,
            "long_name",
            AttrValue::Text("height above ground".into()),
        )
        .map_err(err)?;
    let lat = schema
        .def_var("latitude", NcType::Float, &[y, x])
        .map_err(err)?;
    let lon = schema
        .def_var("longitude", NcType::Float, &[y, x])
        .map_err(err)?;
    schema
        .put_var_attr(lat, "units", AttrValue::Text("degrees_north".into()))
        .map_err(err)?;
    schema
        .put_var_attr(lon, "units", AttrValue::Text("degrees_east".into()))
        .map_err(err)?;
    let mut ids = vec![];
    for p in products.iter().filter(|p| p.kind == "stat") {
        let v = schema
            .def_var(&p.name, NcType::Float, &[h, y, x])
            .map_err(err)?;
        schema
            .put_var_attr(v, "units", AttrValue::Text(p.units.clone()))
            .map_err(err)?;
        schema
            .put_var_attr(v, "long_name", AttrValue::Text(p.title.clone()))
            .map_err(err)?;
        schema
            .put_var_attr(
                v,
                "coordinates",
                AttrValue::Text("height latitude longitude".into()),
            )
            .map_err(err)?;
        ids.push((v, &p.name));
    }
    let mut writer = NcWriter::create(path, schema).map_err(err)?;
    writer
        .write_var(hv, VarData::F64(&o.heights_m))
        .map_err(err)?;
    writer.write_var(lat, VarData::F32(&g.lat)).map_err(err)?;
    writer.write_var(lon, VarData::F32(&g.lon)).map_err(err)?;
    for (id, name) in ids {
        let values: Vec<_> = stats.iter().flat_map(|s| stat_values(s, name)).collect();
        writer.write_var(id, VarData::F32(&values)).map_err(err)?;
    }
    writer.finish().map_err(err)?;
    Ok(())
}
fn run(r: Request) -> Result<(), String> {
    let whole = Instant::now();
    r.options.validate()?;
    if (!r.receipt_prefix.is_empty() && !token(&r.receipt_prefix))
        || !token(&r.domain)
        || !(r.render_frames == "all" || r.render_frames == "last")
        || !["auto", "cpu", "cuda"].contains(&r.compute_backend.as_str())
        || r.width < 320
        || r.height < 240
        || r.inputs.is_empty()
        || !r.observation_max_distance_km.is_finite()
        || r.observation_max_distance_km <= 0.
    {
        return Err("invalid hubwind domain, inputs, size or observation distance".into());
    }
    let products: Vec<Product> = serde_json::from_str(PRODUCTS).map_err(err)?;
    let known: HashSet<_> = products.iter().map(|p| p.name.as_str()).collect();
    if r.products.iter().any(|p| !known.contains(p.as_str())) {
        return Err("unknown hubwind product; use --hubwind-catalog".into());
    }
    let mut curves: Vec<Turbine> = serde_json::from_str(CURVES).map_err(err)?;
    curves.extend(r.turbines.iter().cloned());
    let mut ids = HashSet::new();
    for c in &curves {
        c.validate()?;
        if !ids.insert(&c.id) {
            return Err(format!("duplicate turbine row {}", c.id));
        }
    }
    let curve = curves
        .iter()
        .find(|c| c.id == r.options.turbine)
        .ok_or("turbine ID is absent from the table")?;
    let selected = |p: &Product| r.products.is_empty() || r.products.contains(&p.name);
    std::fs::create_dir_all(&r.out).map_err(err)?;
    let stage = Instant::now();
    let mut records = vec![];
    let mut inputs = vec![];
    for path in &r.inputs {
        let f = WrfFile::open(path).map_err(err)?;
        let axis = wrf_source_times(&f, path)?;
        inputs.push(json!({"path":path,"bytes":std::fs::metadata(path).map_err(err)?.len(),"records":axis.records.len(),"grid":[f.ny,f.nx,f.nz],"reference_unix":axis.reference_unix}));
        for t in axis.records {
            records.push((t.valid_unix, path.clone(), t.time_index));
        }
    }
    records.sort_by_key(|r| r.0);
    if records.windows(2).any(|w| w[0].0 == w[1].0) {
        return Err(
            "hubwind timeline has duplicate valid times; separate domains and members".into(),
        );
    }
    if r.record_selection != "all" {
        let index: usize = r
            .record_selection
            .parse()
            .map_err(|_| "timeidx must be all or a nonnegative integer")?;
        if index >= records.len() {
            return Err("timeidx exceeds hubwind timeline".into());
        }
        records = vec![records[index].clone()];
    }
    let inventory_seconds = stage.elapsed().as_secs_f64();
    let mut frames = vec![];
    let mut grid: Option<Grid> = None;
    let (mut read_seconds, mut render_seconds) = (0., 0.);
    let mut maps = vec![];
    let mut undrawn: Vec<(PathBuf, String)> = vec![];
    let mut tke_sources = vec![];
    let mut fields = vec![];
    let mut field_output_seconds = 0.;
    for (time, path, t) in &records {
        let stage = Instant::now();
        let file = WrfFile::open(path).map_err(err)?;
        let (lat, lon) = rayon::join(|| file.xlat(*t), || file.xlong(*t));
        let (lat, lon) = (lat.map_err(err)?.to_vec(), lon.map_err(err)?.to_vec());
        let lat: Vec<f32> = lat.into_par_iter().map(|x| x as f32).collect();
        let lon: Vec<f32> = lon.into_par_iter().map(|x| x as f32).collect();
        if let Some(g) = &grid {
            if g.nx != file.nx
                || g.ny != file.ny
                || g.lat.iter().zip(&lat).any(|(a, b)| (a - b).abs() > 1e-5)
                || g.lon.iter().zip(&lon).any(|(a, b)| (a - b).abs() > 1e-5)
            {
                return Err("hubwind statistics require one stationary grid; split moving nests before accumulation".into());
            }
        } else {
            grid = Some(Grid {
                lat,
                lon,
                nx: file.nx,
                ny: file.ny,
                projection: wrf_projection(&file),
            });
        }
        let (planes, tke, compute) =
            planes(&file, *t, &r.options, &r.compute_backend, r.verify_cuda)?;
        tke_sources.push(json!({"valid":utc(*time),"source":&tke,"compute":compute}));
        read_seconds += stage.elapsed().as_secs_f64();
        let g = grid.as_ref().unwrap();
        let detail = format!("{} | ideal {:.3} MW | AGL", curve.id, r.options.rated_mw);
        let field_stage = Instant::now();
        let stamp = stamp(*time);
        let field_path = r
            .out
            .join(&r.domain)
            .join("hub_fields")
            .join(&stamp[..8])
            .join(format!("{stamp}.nc"));
        std::fs::create_dir_all(field_path.parent().unwrap()).map_err(err)?;
        write_frame(&field_path, g, &planes, *time, curve, &r.options)?;
        fields.push(field_path);
        field_output_seconds += field_stage.elapsed().as_secs_f64();
        let stage = Instant::now();
        if r.render_frames == "all" || *time == records.last().unwrap().0 {
            // Every map of a frame is independent (its own plane, file and
            // canvas), so they are drawn across the Rayon pool. Drawn one
            // after another, the 19 default maps of each frame held the
            // end-of-run render of a 2.25 km / 750 m nest on about one core
            // of eight (2.9.5, RTX PRO 4500 box, 2026-10-23). The order of
            // `maps` and `undrawn` is the serial order.
            let mut jobs: Vec<(&Product, Vec<f32>, String, f64, String)> = vec![];
            for (i, plane) in planes.iter().enumerate() {
                let suffix = height_token(plane.height_m);
                for p in products.iter().filter(|p| selected(p) && p.kind == "frame") {
                    let values = match p.name.as_str() {
                        "hub_speed" => plane.speed.clone(),
                        "hub_direction" => plane.direction.clone(),
                        "hub_ti" => plane.ti.clone(),
                        "hub_power" => plane
                            .speed
                            .par_iter()
                            .map(|&x| (curve.fraction(x as f64) * r.options.rated_mw) as f32)
                            .collect(),
                        _ => unreachable!(),
                    };
                    let upper = if p.name == "hub_power" {
                        r.options.rated_mw
                    } else {
                        p.upper
                    };
                    let text = if p.name == "hub_ti" && tke.starts_with("diagnosed") {
                        format!("{detail} | TI diagnosed: surface-layer similarity, no PBL TKE")
                    } else {
                        detail.clone()
                    };
                    jobs.push((p, values, suffix.clone(), upper, text));
                }
                if let Some(p) = products.iter().find(|p| p.kind == "shear" && selected(p)) {
                    let next = planes
                        .iter()
                        .enumerate()
                        .filter(|(_, p)| p.height_m > plane.height_m)
                        .min_by(|a, b| a.1.height_m.total_cmp(&b.1.height_m));
                    if let Some((j, other)) = next {
                        let values = calc::shear(&planes[i], &planes[j])?;
                        jobs.push((
                            p,
                            values,
                            format!("{}-{}", suffix, height_token(other.height_m)),
                            1.,
                            "log speed ratio / log AGL height ratio".to_owned(),
                        ));
                    }
                }
            }
            let drawn = jobs
                .into_par_iter()
                .map(|(p, values, suffix, upper, text)| {
                    draw(&r, g, p, values, &suffix, *time, upper, &text)
                })
                .collect::<Result<Vec<_>, String>>()?;
            for one in drawn {
                record(&mut maps, &mut undrawn, one);
            }
        }
        render_seconds += stage.elapsed().as_secs_f64();
        frames.push(Frame {
            time: *time,
            planes,
        });
        eprintln!(
            "HUBWIND\t{}\t{}x{}\t{}",
            utc(*time),
            g.ny,
            g.nx,
            path.display()
        );
    }
    let g = grid.as_ref().unwrap();
    let stage = Instant::now();
    let mut summaries = vec![];
    let mut statistics = vec![];
    let times: Vec<_> = frames.iter().map(|f| f.time as f64).collect();
    if frames.len() >= 2 {
        for (h, &height) in r.options.heights_m.iter().enumerate() {
            let speed: Vec<_> = frames.iter().map(|f| f.planes[h].speed.clone()).collect();
            let s = calc::statistics(&times, &speed, curve, &r.options)?;
            let all: Vec<_> = products
                .iter()
                .filter(|p| p.kind == "stat")
                .map(|p| json!({"product":p.name,"values":summary(&stat_values(&s,&p.name))}))
                .collect();
            let strongest = s
                .iter()
                .enumerate()
                .filter(|(_, s)| s.max_abs_ramp_ms.is_finite())
                .max_by(|a, b| a.1.max_abs_ramp_ms.total_cmp(&b.1.max_abs_ramp_ms));
            let event=strongest.map(|(i,s)|json!({"index":i,"lat":g.lat[i],"lon":g.lon[i],"statistics":s,"trace":frames.iter().map(|f|json!({"valid":utc(f.time),"speed_ms":f.planes[h].speed[i],"direction_deg":f.planes[h].direction[i]})).collect::<Vec<_>>()}));
            summaries.push(json!({"height_m":height,"products":all,"strongest_ramp":event}));
            statistics.push(s);
        }
        write_stats(
            &r.out.join(format!("{}hub-statistics.nc", r.receipt_prefix)),
            g,
            &statistics,
            &products,
            &r.options,
            curve,
        )?;
    }
    let statistics_seconds = stage.elapsed().as_secs_f64();
    let stage = Instant::now();
    // The run statistics' maps (15 per height by default), across the pool
    // like the frame maps, in the serial order.
    let stat_jobs: Vec<(usize, &Vec<Stats>, &Product)> = statistics
        .iter()
        .enumerate()
        .flat_map(|(h, s)| {
            products
                .iter()
                .filter(|p| p.kind == "stat" && selected(p))
                .map(move |p| (h, s, p))
        })
        .collect();
    let drawn = stat_jobs
        .into_par_iter()
        .map(|(h, s, p)| {
            let duration = (times.last().unwrap() - times[0]) / 3600.;
            let upper = if p.upper > 0. {
                p.upper
            } else if p.name == "hub_energy" {
                duration * r.options.rated_mw
            } else {
                duration
            };
            draw(
                &r,
                g,
                p,
                stat_values(s, &p.name),
                &height_token(r.options.heights_m[h]),
                frames.last().unwrap().time,
                upper,
                &format!(
                    "{} to {} | {} s ramp | {} m/s threshold",
                    utc(frames[0].time),
                    utc(frames.last().unwrap().time),
                    r.options.ramp_window_seconds,
                    r.options.ramp_ms
                ),
            )
        })
        .collect::<Result<Vec<_>, String>>()?;
    for one in drawn {
        record(&mut maps, &mut undrawn, one);
    }
    render_seconds += stage.elapsed().as_secs_f64();
    let stage = Instant::now();
    let verification = verify(&r, g, &frames)?;
    let verification_seconds = stage.elapsed().as_secs_f64();
    let receipt = json!({"schema":"gpuwm.hubwind.v1","status":"complete","grid":[g.ny,g.nx],"cells":g.ny*g.nx,"frames":frames.len(),"valid_times":frames.iter().map(|f|utc(f.time)).collect::<Vec<_>>(),"inputs":inputs,"options":r.options,"turbine":curve,
 "semantics":{"interpolation":"linear earth vector versus geometric AGL; surface 10 m anchor when available; no upper extrapolation","shear":"adjacent requested heights sorted by AGL","ti":"sqrt(2*TKE/3)/speed; isotropic unresolved PBL estimate, not measured IEC turbulence; negative/absent/calm missing; where the PBL scheme carries no TKE, TKE is the surface-layer similarity form named in tke_sources","power":"ideal table at reference air density; nameplate scaling; no density correction, wakes or availability losses; normalized industry composite, not certified manufacturer curve","regime_hours":"below cut in; cut in <= speed < rated; rated <= speed < cut out; speed >= cut out; linear speed crossing durations","time_integration":"piecewise linear speed; exact piecewise table energy integral; gaps beyond max_gap excluded","ramps":"signed trailing window speed/power changes evaluated at output timestamps; contiguous threshold exceedances count once; temporal sampling can miss shorter ramps"},
 "tke_sources":tke_sources,"spatial_statistics":summaries,"verification":verification,
 "stages":{"inventory_seconds":inventory_seconds,"read_and_interpolate_seconds":read_seconds,"statistics_and_netcdf_seconds":statistics_seconds,"render_seconds":render_seconds,"field_output_seconds":field_output_seconds,"verification_seconds":verification_seconds,"total_seconds":whole.elapsed().as_secs_f64()},
 "parallelism":{"backend":"CUDA when available (or requested), Rust/Rayon otherwise; per-frame backend above","threads":rayon::current_num_threads(),"column_interpolation":"parallel across heights and cells","statistics":"parallel across cells","field_read":"parallel independent variables","render":"native rw_wrfbatch production panel"},"maps":maps,"undrawn_maps":undrawn.iter().map(|(p,reason)|json!({"path":p,"reason":reason})).collect::<Vec<_>>(),"fields":fields,"statistics_file":if statistics.is_empty(){None}else{Some(r.out.join(format!("{}hub-statistics.nc",r.receipt_prefix)))}});
    let receipt_path = r
        .out
        .join(format!("{}hubwind-receipt.json", r.receipt_prefix));
    std::fs::write(
        &receipt_path,
        serde_json::to_vec_pretty(&receipt).map_err(err)?,
    )
    .map_err(err)?;
    for path in maps {
        println!("RENDERED\t{}", path.display());
    }
    // Drawn on disk so a reader sees why, never counted as drawn.
    for (path, reason) in &undrawn {
        println!("UNDRAWN\t{}\t{}", path.display(), reason);
    }
    println!("RECEIPT\t{}", receipt_path.display());
    Ok(())
}
// Observation formats are metadata selected; no tower specific calculation.
fn csv(line: &str) -> Vec<String> {
    let mut out = vec![];
    let (mut s, mut quoted) = (String::new(), false);
    for c in line.chars() {
        if c == '"' {
            quoted = !quoted;
        } else if c == ',' && !quoted {
            out.push(s.trim().into());
            s.clear();
        } else {
            s.push(c);
        }
    }
    out.push(s.trim().into());
    out
}
fn obs_time(cols: &[String], format: &str) -> Option<i64> {
    match format {
        "year_doy_hhmm_csv" => {
            let year = cols.get(1)?.parse().ok()?;
            let day = cols.get(2)?.parse().ok()?;
            let hhmm: u32 = cols.get(3)?.parse().ok()?;
            Some(
                NaiveDate::from_yo_opt(year, day)?
                    .and_hms_opt(hhmm / 100, hhmm % 100, 0)?
                    .and_utc()
                    .timestamp(),
            )
        }
        "timestamp_csv" => Some(
            crate::verification::parse_time(cols.first()?)
                .ok()?
                .timestamp(),
        ),
        _ => None,
    }
}
// Exact valid timestamps do not depend on either adjacent sample. Calm wind
// has a valid zero vector even though its meteorological direction is missing.
fn model_vector(
    frames: &[Frame],
    height: usize,
    cell: usize,
    time: i64,
    max_gap: f64,
) -> Option<(f64, f64)> {
    if frames.is_empty() || time < frames[0].time || time > frames.last()?.time {
        return None;
    }
    let vector = |frame: &Frame| {
        let p = &frame.planes[height];
        let speed = p.speed[cell] as f64;
        let direction = p.direction[cell] as f64;
        if !speed.is_finite() || speed < 0. {
            return None;
        }
        if speed == 0. {
            return Some((0., 0.));
        }
        if !direction.is_finite() {
            return None;
        }
        Some((
            -speed * direction.to_radians().sin(),
            -speed * direction.to_radians().cos(),
        ))
    };
    let k = frames.partition_point(|f| f.time <= time) - 1;
    let a = vector(&frames[k])?;
    if frames[k].time == time {
        return Some(a);
    }
    let next = frames.get(k + 1)?;
    let dt = next.time - frames[k].time;
    if dt <= 0 || dt as f64 > max_gap {
        return None;
    }
    let b = vector(next)?;
    let fraction = (time - frames[k].time) as f64 / dt as f64;
    Some((a.0 + (b.0 - a.0) * fraction, a.1 + (b.1 - a.1) * fraction))
}
fn verify(r: &Request, g: &Grid, frames: &[Frame]) -> Result<Vec<Value>, String> {
    let rows: Vec<ObsRow> = serde_json::from_str(OBS).map_err(err)?;
    let mut results = vec![];
    for input in &r.observations {
        let row = rows
            .iter()
            .find(|s| s.id == input.network)
            .ok_or("observation row absent")?;
        let h = r
            .options
            .heights_m
            .iter()
            .position(|&h| h == row.height_m)
            .ok_or("observation height must be included in heights_m")?;
        let distance = |i: usize| {
            let a = (g.lat[i] as f64 - row.lat).to_radians();
            let b = (g.lon[i] as f64 - row.lon).to_radians();
            let x = (a * 0.5).sin().powi(2)
                + (row.lat.to_radians().cos() * (g.lat[i] as f64).to_radians().cos())
                    * (b * 0.5).sin().powi(2);
            12742. * x.sqrt().clamp(0., 1.).asin()
        };
        let (cell, km) = (0..g.lat.len())
            .into_par_iter()
            .map(|i| (i, distance(i)))
            .filter(|x| x.1.is_finite())
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .ok_or("no finite model coordinates")?;
        if km > r.observation_max_distance_km {
            return Err(format!(
                "{} is {km:.2} km outside model support; max {}",
                row.id, r.observation_max_distance_km
            ));
        }
        let text = std::fs::read_to_string(&input.path).map_err(err)?;
        let out = r.out.join(format!("{}-verification.csv", row.id));
        let mut writer = std::io::BufWriter::new(std::fs::File::create(&out).map_err(err)?);
        writeln!(writer,"valid_utc,height_m,observed_speed_ms,modeled_speed_ms,observed_direction_deg,modeled_direction_deg,speed_error_ms,direction_error_deg").map_err(err)?;
        let (mut n, mut nd, mut bias, mut absolute, mut square, mut angles) =
            (0u64, 0u64, 0., 0., 0., 0.);
        for line in text.lines() {
            let c = csv(line);
            let Some(t) = obs_time(&c, &row.format) else {
                continue;
            };
            let Some(os) = c
                .get(row.speed_column)
                .and_then(|s| s.parse::<f64>().ok())
                .filter(|s| s.is_finite() && *s >= 0.)
            else {
                continue;
            };
            let di = if row.direction_column < 0 {
                c.len().saturating_sub(row.direction_column.unsigned_abs())
            } else {
                row.direction_column as usize
            };
            let od = c
                .get(di)
                .and_then(|s| s.parse::<f64>().ok())
                .filter(|s| s.is_finite() && *s >= 0. && *s <= 360.);
            if t < frames[0].time || t > frames.last().unwrap().time {
                continue;
            }
            let Some((u, v)) = model_vector(frames, h, cell, t, r.options.max_gap_seconds) else {
                continue;
            };
            let ms = u.hypot(v);
            if !ms.is_finite() {
                continue;
            }
            let md = if ms > 1e-6 {
                (270. - v.atan2(u).to_degrees()).rem_euclid(360.)
            } else {
                f64::NAN
            };
            let e = ms - os;
            let de = od
                .filter(|_| md.is_finite())
                .map(|od| (md - od + 180.).rem_euclid(360.) - 180.);
            n += 1;
            bias += e;
            absolute += e.abs();
            square += e * e;
            if let Some(d) = de {
                nd += 1;
                angles += d.abs();
            }
            writeln!(
                writer,
                "{},{},{os},{ms},{},{md},{e},{}",
                utc(t),
                row.height_m,
                od.map_or(String::new(), |d| d.to_string()),
                de.map_or(String::new(), |d| d.to_string())
            )
            .map_err(err)?;
        }
        writer.flush().map_err(err)?;
        results.push(json!({"network":row.id,"path":input.path,"height_m":row.height_m,"distance_km":km,"grid_cell":cell,"pairs":n,"speed_bias_ms":if n>0{Some(bias/n as f64)}else{None},"speed_mae_ms":if n>0{Some(absolute/n as f64)}else{None},"speed_rmse_ms":if n>0{Some((square/n as f64).sqrt())}else{None},"direction_pairs":nd,"direction_mae_deg":if nd>0{Some(angles/nd as f64)}else{None},"csv":out}));
    }
    Ok(results)
}
pub fn cli(args: &[String]) -> Option<Result<(), String>> {
    if args.first().map(String::as_str) == Some("--hubwind-catalog") {
        return Some((|| {
            let p: Value = serde_json::from_str(PRODUCTS).map_err(err)?;
            let c: Value = serde_json::from_str(CURVES).map_err(err)?;
            let o: Value = serde_json::from_str(OBS).map_err(err)?;
            println!(
                "{}",
                json!({"schema":"gpuwm.hubwind-catalog.v1","products":p,"turbines":c,"observations":o,"default_options":Options::default()})
            );
            Ok(())
        })());
    }
    if args.first().map(String::as_str) != Some("--hubwind-request") {
        return None;
    }
    Some((|| {
        if args.len() != 2 {
            return Err("--hubwind-request requires one JSON request path".into());
        }
        let bytes = std::fs::read(&args[1]).map_err(err)?;
        let request: Request = serde_json::from_slice(&bytes).map_err(err)?;
        run(request)
    })())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn shipped_rows_validate() {
        let curves: Vec<Turbine> = serde_json::from_str(CURVES).unwrap();
        for c in curves {
            c.validate().unwrap();
            assert_eq!(c.fraction(c.cut_out_ms), 0.);
        }
    }
    #[test]
    fn verification_retains_calm_and_exact_samples_beside_missing_data() {
        let frame = |time, speed, direction| Frame {
            time,
            planes: vec![Plane {
                height_m: 100.,
                speed: vec![speed],
                direction: vec![direction],
                ti: vec![f32::NAN],
            }],
        };
        let frames = vec![
            frame(0, f32::NAN, f32::NAN),
            frame(3600, 4., 270.),
            frame(7200, 0., f32::NAN),
            frame(10800, f32::NAN, f32::NAN),
        ];
        let exact = model_vector(&frames, 0, 0, 3600, 5400.).unwrap();
        assert!((exact.0 - 4.).abs() < 1e-6);
        assert_eq!(model_vector(&frames, 0, 0, 7200, 5400.), Some((0., 0.)));
        let middle = model_vector(&frames, 0, 0, 5400, 5400.).unwrap();
        assert!((middle.0.hypot(middle.1) - 2.).abs() < 1e-6);
        assert!(model_vector(&frames, 0, 0, 1800, 5400.).is_none());
        assert!(model_vector(&frames, 0, 0, 5400, 1800.).is_none());
    }
    #[test]
    fn similarity_sigma_u_follows_the_named_forms() {
        // Neutral limit of both arms: 12^(1/3) u* at the surface.
        let neutral = super::similarity_sigma_u(0.5, 1000., 0., 0.);
        assert!((neutral - 0.5 * 12f64.cbrt()).abs() < 1e-12);
        // Unstable: Panofsky et al. (1977), height independent below zi.
        let unstable = super::similarity_sigma_u(0.4, 1500., -0.01, 100.);
        assert!((unstable - 0.4 * (12f64 + 0.5 * 1500. * 0.01).cbrt()).abs() < 1e-12);
        assert!(super::similarity_sigma_u(0.4, 1500., -0.01, 1600.).is_nan());
        // Stable: decays to zero at h, missing above it.
        let stable = super::similarity_sigma_u(0.2, 200., 0.02, 100.);
        assert!((stable - 12f64.cbrt() * 0.2 * 0.5f64.powf(0.75)).abs() < 1e-12);
        assert!(super::similarity_sigma_u(0.2, 200., 0.02, 250.).is_nan());
        // TI from TKE = 1.5 sigma^2 is sigma / speed.
        let tke = 1.5 * unstable * unstable;
        assert!(((2. * tke / 3.).sqrt() - unstable).abs() < 1e-12);
        assert!(super::similarity_sigma_u(f64::NAN, 1500., -0.01, 100.).is_nan());
    }

    #[test]
    fn domain_tokens_take_fractional_spacings_and_stay_one_component() {
        for ok in ["d01-2.25km", "d02-750m", "d01-12km", "d01-2.25km-", "d03-1.5km"] {
            assert!(token(ok), "{ok}");
        }
        for bad in ["", ".", "..", "../d01", "d01/x", ".hidden", "d01..km", "d01 km"] {
            assert!(!token(bad), "{bad}");
        }
    }

    #[test]
    fn archive_time_is_utc() {
        let c = csv("100,2015,305,1305,0,0,7,6,0,0,0,350");
        assert_eq!(
            utc(obs_time(&c, "year_doy_hhmm_csv").unwrap()),
            "2015-11-01T13:05:00+00:00"
        );
        assert_eq!(csv("\"2015-11-01 01:02:00\",1,2")[0], "2015-11-01 01:02:00");
    }
}

pub fn product_catalog() -> Vec<String> {
    let p: Vec<Product> = serde_json::from_str(PRODUCTS).expect("shipped hubwind rows");
    p.into_iter()
        .map(|p| p.name)
        .chain(std::iter::once("hub_all".into()))
        .collect()
}
