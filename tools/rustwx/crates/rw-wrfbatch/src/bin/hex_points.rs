//! `rw_hex_points`: soundings (skew-T log-p) and meteograms at named points,
//! straight off MPAS history on any mesh.
//!
//! The columns come from `rw_mpas::points` -- the same linear (barycentric)
//! interpolant the product remap draws maps with, so a sounding agrees with
//! the map under it.  Drawing uses the map renderer's own fonts, strokes and
//! wind barbs (`rustwx-render`); thermodynamics (parcel, CAPE/CIN) are
//! `wx-math`'s, through `wx-core`.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use chrono::{Datelike, NaiveDate};
use rustwx_render::{
    Rgba, RgbaImage, draw_circle_fill_aa, draw_circle_stroke_aa, draw_line_aa_width,
    draw_polyline_aa, draw_text, draw_text_bold, draw_wind_barb, fill_polygon, text_width,
    text_width_bold,
};
use rayon::prelude::*;
use rw_mpas::history::{Timestamp, read_history_cells_on_mesh};
use rw_mpas::points::{Column, PointSampler, PointSpec, columns, parse_points};

/// Every WRF name [`columns`] reads (the last four are optional there).
const POINT_FIELDS: [&str; 15] = [
    "T", "PB", "QVAPOR", "U", "V", "PHB", "T2", "Q2", "PSFC", "U10", "V10", "RAINC", "RAINNC", "REFL_10CM", "W",
];
use rw_mpas::remap::read_mesh_topology;
use rw_mpas::weights::{read_mesh_cell_spacing, read_mesh_coordinates};
use wx_core::metfuncs as thermo;

const USAGE: &str = "\
usage: rw_hex_points --mesh FILE --init FILE --history FILE... --out-dir DIR
                     --points 'Name@LAT,LON;Name@LAT,LON'
                     [--simulation-start YYYY-MM-DD_HH:MM:SS] [--model-label NAME]
                     [--grid-label TEXT] [--brand TEXT] [--sounding-every-hours H]
                     [--json FILE] [--columns-json FILE] [--soundings-only]

Writes, per point:
  <slug>/sounding_<slug>_<YYYYMMDD_HHMM>_f<LLL>.png   skew-T log-p at each
        frame whose lead is a multiple of --sounding-every-hours (default 6)
  <slug>/meteogram_<slug>_<YYYYMMDD_HH>.png            every frame, one chart
and a JSON receipt with every value drawn.  --columns-json writes the one
frame's columns as an rw_soundings JSON column frame (one --history file only),
so the site's own sounding window can draw this point; --soundings-only skips
the meteogram, and the skew-T when --columns-json stands in for it.  --brand puts the model's name and
start in bold at the top right of every picture (e.g. 'WOOF Hex · Global
120 km · Init GFS 22 Oct 2026 00Z'); without it the top right names the
model and its start ('WOOF Hex · Init 22 Oct 2026 00Z').  A
point off a limited-area mesh is refused by name rather than drawn from the
boundary ring.";

const ABI: &str = "gpuwm-rw-hex-points-v1\tPOINT\tSOUNDING\tMETEOGRAM\tFINISHED\tCOLUMNS-JSON";

/// `GPUWM_BRIDGE_SOURCE_REV=<40-hex commit>`: the source revision this
/// binary was built from (the crate's `build.rs` injects it), so the release
/// cut can prove a staged copy matches the commit being released by reading
/// bytes alone.  `main` references it so the linker cannot discard it.
pub static GPUWM_BRIDGE_SOURCE_REV_STAMP: &str =
    concat!("GPUWM_BRIDGE_SOURCE_REV=", env!("GPUWM_BRIDGE_SOURCE_REV"));

const KNOTS: f64 = 1.943_844_5;

struct Args {
    mesh: PathBuf,
    init: PathBuf,
    history: Vec<PathBuf>,
    out_dir: PathBuf,
    points: Vec<PointSpec>,
    simulation_start: Option<Timestamp>,
    model_label: String,
    grid_label: Option<String>,
    brand: Option<String>,
    sounding_every_hours: f64,
    json: Option<PathBuf>,
    columns_json: Option<PathBuf>,
    soundings_only: bool,
}

fn parse() -> Result<Args, String> {
    let mut mesh = None;
    let mut init = None;
    let mut history = Vec::new();
    let mut out_dir = None;
    let mut points = None;
    let mut simulation_start = None;
    let mut model_label = "hex".to_string();
    let mut grid_label = None;
    let mut brand: Option<String> = None;
    let mut sounding_every_hours = 6.0;
    let mut json = None;
    let mut columns_json = None;
    let mut soundings_only = false;
    let mut raw = std::env::args().skip(1).peekable();
    while let Some(flag) = raw.next() {
        let mut value = || raw.next().ok_or_else(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "--mesh" => mesh = Some(PathBuf::from(value()?)),
            "--init" => init = Some(PathBuf::from(value()?)),
            "--out-dir" => out_dir = Some(PathBuf::from(value()?)),
            "--points" => points = Some(parse_points(&value()?).map_err(|e| e.to_string())?),
            "--simulation-start" => {
                simulation_start = Some(Timestamp::parse(&value()?).map_err(|e| e.to_string())?)
            }
            "--model-label" => model_label = value()?,
            "--grid-label" => grid_label = Some(value()?),
            "--brand" => {
                let text = value()?.split_whitespace().collect::<Vec<_>>().join(" ");
                if text.chars().count() > 120 || text.chars().any(|ch| ch.is_control()) {
                    return Err("--brand takes at most 120 printable characters".to_string());
                }
                brand = Some(text).filter(|t| !t.is_empty());
            }
            "--sounding-every-hours" => {
                sounding_every_hours = value()?
                    .parse()
                    .map_err(|_| "--sounding-every-hours takes a number".to_string())?
            }
            "--json" => json = Some(PathBuf::from(value()?)),
            "--columns-json" => columns_json = Some(PathBuf::from(value()?)),
            "--soundings-only" => soundings_only = true,
            "--history" => {
                while let Some(next) = raw.peek() {
                    if next.starts_with("--") {
                        break;
                    }
                    history.push(PathBuf::from(raw.next().unwrap()));
                }
            }
            other => return Err(format!("unknown flag {other}")),
        }
    }
    let history_empty = history.is_empty();
    let args = Args {
        mesh: mesh.ok_or("--mesh is required")?,
        init: init.ok_or("--init is required (zgrid, the column heights, comes from it)")?,
        history,
        out_dir: out_dir.ok_or("--out-dir is required")?,
        points: points.ok_or("--points is required")?,
        simulation_start,
        model_label,
        grid_label,
        brand,
        sounding_every_hours,
        json,
        columns_json,
        soundings_only,
    };
    if history_empty {
        return Err("--history needs at least one file".to_string());
    }
    if args.columns_json.is_some() && args.history.len() != 1 {
        return Err("--columns-json writes one frame: give exactly one --history file".to_string());
    }
    if !(args.sounding_every_hours > 0.0) {
        return Err("--sounding-every-hours must be positive".to_string());
    }
    Ok(args)
}

// ---------------------------------------------------------------------------
// Small helpers.
// ---------------------------------------------------------------------------

fn white(w: u32, h: u32) -> RgbaImage {
    RgbaImage::from_pixel(w, h, image::Rgba([255, 255, 255, 255]))
}

const INK: Rgba = Rgba::new(28, 32, 38);
const SUBTLE: Rgba = Rgba::new(96, 104, 114);
const GRID: Rgba = Rgba::new(214, 218, 224);
const RED: Rgba = Rgba::new(200, 30, 36);
const GREEN: Rgba = Rgba::new(18, 140, 54);
const BLUE: Rgba = Rgba::new(30, 90, 190);
const PURPLE: Rgba = Rgba::new(130, 70, 170);

fn dewpoint_c(p_pa: f64, qv: f64) -> f64 {
    let w = qv.max(1.0e-10);
    let e_hpa = w * (p_pa / 100.0) / (0.621_97 + w);
    thermo::dewpoint_from_vapor_pressure(e_hpa)
}

fn c_to_f(c: f64) -> f64 {
    c * 9.0 / 5.0 + 32.0
}

fn weekday(t: &Timestamp) -> &'static str {
    match NaiveDate::from_ymd_opt(t.year, t.month, t.day).map(|d| d.weekday()) {
        Some(chrono::Weekday::Mon) => "Mon",
        Some(chrono::Weekday::Tue) => "Tue",
        Some(chrono::Weekday::Wed) => "Wed",
        Some(chrono::Weekday::Thu) => "Thu",
        Some(chrono::Weekday::Fri) => "Fri",
        Some(chrono::Weekday::Sat) => "Sat",
        Some(chrono::Weekday::Sun) => "Sun",
        None => "",
    }
}

fn stamp_text(t: &Timestamp) -> String {
    format!(
        "{} {:04}-{:02}-{:02} {:02}:{:02} UTC",
        weekday(t),
        t.year,
        t.month,
        t.day,
        t.hour,
        t.minute
    )
}

fn latlon_text(p: &PointSpec) -> String {
    format!(
        "{:.2}{} {:.2}{}",
        p.lat.abs(),
        if p.lat >= 0.0 { "N" } else { "S" },
        p.lon.abs(),
        if p.lon >= 0.0 { "E" } else { "W" }
    )
}

/// A dashed polyline: `on` pixels drawn, `off` skipped, along the path.
fn draw_dashed(img: &mut RgbaImage, pts: &[(f64, f64)], color: Rgba, width: u32, on: f64, off: f64) {
    let mut phase = 0.0;
    for seg in pts.windows(2) {
        let (x0, y0) = seg[0];
        let (x1, y1) = seg[1];
        let len = ((x1 - x0).powi(2) + (y1 - y0).powi(2)).sqrt();
        if len == 0.0 {
            continue;
        }
        let mut s = 0.0;
        while s < len {
            let period = on + off;
            let in_on = phase % period < on;
            let step = if in_on { on - phase % period } else { period - phase % period };
            let e = (s + step).min(len);
            if in_on {
                let a = s / len;
                let b = e / len;
                draw_line_aa_width(
                    img,
                    x0 + (x1 - x0) * a,
                    y0 + (y1 - y0) * a,
                    x0 + (x1 - x0) * b,
                    y0 + (y1 - y0) * b,
                    color,
                    width,
                );
            }
            phase += e - s;
            s = e;
        }
    }
}

/// Clip a polyline to a rectangle by dropping points outside and splitting.
fn clip_runs(pts: &[(f64, f64)], rect: (f64, f64, f64, f64)) -> Vec<Vec<(f64, f64)>> {
    let inside = |p: &(f64, f64)| p.0 >= rect.0 && p.0 <= rect.2 && p.1 >= rect.1 && p.1 <= rect.3;
    let mut runs = Vec::new();
    let mut cur: Vec<(f64, f64)> = Vec::new();
    for w in pts.windows(2) {
        let (a, b) = (w[0], w[1]);
        let (ia, ib) = (inside(&a), inside(&b));
        if ia && cur.is_empty() {
            cur.push(a);
        }
        if ia && ib {
            cur.push(b);
        } else if ia != ib {
            // bisect for the crossing
            let (mut lo, mut hi) = if ia { (a, b) } else { (b, a) };
            for _ in 0..30 {
                let m = ((lo.0 + hi.0) * 0.5, (lo.1 + hi.1) * 0.5);
                if inside(&m) {
                    lo = m;
                } else {
                    hi = m;
                }
            }
            if ia {
                cur.push(lo);
                runs.push(std::mem::take(&mut cur));
            } else {
                cur.push(lo);
                cur.push(b);
            }
        }
    }
    if cur.len() > 1 {
        runs.push(cur);
    }
    runs
}

fn nice_step(range: f64, target_ticks: f64) -> f64 {
    let raw = (range / target_ticks).max(1e-9);
    let mag = 10f64.powf(raw.log10().floor());
    let n = raw / mag;
    let nice = if n < 1.5 {
        1.0
    } else if n < 3.0 {
        2.0
    } else if n < 7.0 {
        5.0
    } else {
        10.0
    };
    nice * mag
}

fn tick_label(v: f64, step: f64) -> String {
    if step >= 1.0 {
        format!("{:.0}", v)
    } else if step >= 0.1 {
        format!("{:.1}", v)
    } else {
        format!("{:.2}", v)
    }
}

// ---------------------------------------------------------------------------
// The skew-T.
// ---------------------------------------------------------------------------

/// Horizontal pixels per vertical pixel of isotherm lean: about 4.5 K per
/// km on this canvas, so a standard-atmosphere troposphere stands near
/// upright and the tropopause sits mid-diagram.
const SKEW: f64 = 0.8;

struct SkewT {
    x0: f64,
    y0: f64,
    x1: f64,
    y1: f64,
    p_bot: f64,
    p_top: f64,
    t_left: f64,
    t_right: f64,
}

impl SkewT {
    fn y(&self, p_hpa: f64) -> f64 {
        self.y1 - (self.p_bot.ln() - p_hpa.ln()) / (self.p_bot.ln() - self.p_top.ln()) * (self.y1 - self.y0)
    }
    fn x(&self, t_c: f64, p_hpa: f64) -> f64 {
        let w = self.x1 - self.x0;
        self.x0 + (t_c - self.t_left) / (self.t_right - self.t_left) * w
            + SKEW * (self.y1 - self.y(p_hpa))
    }
    fn rect(&self) -> (f64, f64, f64, f64) {
        (self.x0, self.y0, self.x1, self.y1)
    }
}

struct SoundingIndices {
    sbcape: f64,
    sbcin: f64,
    lcl_m: f64,
    pw_mm: f64,
    shear06_kt: f64,
    t2_f: f64,
    td2_f: f64,
    mslp_hpa: f64,
    mlcape: f64,
    mlcin: f64,
    mucape: f64,
    shear01_kt: f64,
    srh01: f64,
    srh03: f64,
    /// Bunkers right- and left-mover motion, m/s.
    storm_rm: (f64, f64),
    storm_lm: (f64, f64),
    /// Which of the two the helicities are relative to: the cyclonic one,
    /// the right mover north of the equator and the left mover south of it.
    storm_is_right: bool,
}

/// The wind profile a hodograph is drawn from: the 10 m wind, then every
/// model level above it.  Heights are metres above ground, winds m/s,
/// pressures hPa; surface first.
struct WindProfile {
    z: Vec<f64>,
    u: Vec<f64>,
    v: Vec<f64>,
    p: Vec<f64>,
}

fn wind_profile(c: &Column) -> WindProfile {
    let mut w = WindProfile {
        z: vec![10.0],
        u: vec![c.u10],
        v: vec![c.v10],
        p: vec![c.psfc_pa / 100.0],
    };
    for k in 0..c.height_m.len() {
        let agl = c.height_m[k] - c.terrain_m;
        if agl > *w.z.last().unwrap() && c.u[k].is_finite() && c.v[k].is_finite() {
            w.z.push(agl);
            w.u.push(c.u[k]);
            w.v.push(c.v[k]);
            w.p.push(c.pressure_pa[k] / 100.0);
        }
    }
    w
}

/// The wind at a height above ground, linear between levels; `None` outside
/// the profile.
fn wind_at(w: &WindProfile, z: f64) -> Option<(f64, f64)> {
    if w.z.len() < 2 || z < w.z[0] || z > *w.z.last().unwrap() {
        return None;
    }
    for k in 0..w.z.len() - 1 {
        if w.z[k] <= z && z <= w.z[k + 1] {
            let f = (z - w.z[k]) / (w.z[k + 1] - w.z[k]);
            return Some((w.u[k] + f * (w.u[k + 1] - w.u[k]), w.v[k] + f * (w.v[k + 1] - w.v[k])));
        }
    }
    None
}

/// `(z, u, v)` from the surface to `top_m`: every level in the layer, plus
/// an interpolated point exactly at each of `breaks` (ascending, below
/// `top_m`) and at `top_m` itself, so a layer ends where it says it does.
fn layer_points(w: &WindProfile, top_m: f64, breaks: &[f64]) -> Vec<(f64, f64, f64)> {
    let mut out = Vec::new();
    for k in 0..w.z.len() {
        if w.z[k] > top_m {
            break;
        }
        out.push((w.z[k], w.u[k], w.v[k]));
        if k + 1 < w.z.len() {
            for b in breaks.iter().copied().chain(std::iter::once(top_m)) {
                if b > w.z[k] && b < w.z[k + 1] {
                    if let Some((u, v)) = wind_at(w, b) {
                        out.push((b, u, v));
                    }
                }
            }
        }
    }
    out
}

/// Storm-relative helicity from the surface to `top_m`, m2/s2: the area the
/// storm-relative hodograph sweeps, positive for winds veering with height.
/// NaN when the profile does not reach `top_m`.
fn storm_relative_helicity(w: &WindProfile, top_m: f64, storm: (f64, f64)) -> f64 {
    let pts = layer_points(w, top_m, &[]);
    if pts.len() < 2 || pts.last().unwrap().0 < top_m - 1.0 || !storm.0.is_finite() || !storm.1.is_finite() {
        return f64::NAN;
    }
    pts.windows(2)
        .map(|s| (s[1].1 - storm.0) * (s[0].2 - storm.1) - (s[0].1 - storm.0) * (s[1].2 - storm.1))
        .sum()
}

fn bulk_shear_kt(w: &WindProfile, top_m: f64) -> f64 {
    match wind_at(w, top_m) {
        Some((u, v)) => ((u - w.u[0]).powi(2) + (v - w.v[0]).powi(2)).sqrt() * KNOTS,
        None => f64::NAN,
    }
}

/// Meteorological direction (degrees the wind blows FROM) and speed in knots.
fn direction_speed_text(u: f64, v: f64) -> String {
    if !u.is_finite() || !v.is_finite() {
        return "n/a".to_string();
    }
    let speed = (u * u + v * v).sqrt() * KNOTS;
    let mut dir = (-u).atan2(-v).to_degrees();
    if dir < 0.0 {
        dir += 360.0;
    }
    format!("{:03.0}\u{b0} / {:.0} kt", dir, speed)
}

fn value_text(v: f64, unit: &str) -> String {
    if v.is_finite() { format!("{v:.0} {unit}") } else { "n/a".to_string() }
}

fn sounding_indices(c: &Column, lat: f64) -> SoundingIndices {
    let p_hpa: Vec<f64> = c.pressure_pa.iter().map(|p| p / 100.0).collect();
    let t_c: Vec<f64> = c.temperature_k.iter().map(|t| t - 273.15).collect();
    let td_c: Vec<f64> = c
        .pressure_pa
        .iter()
        .zip(c.qv.iter())
        .map(|(p, q)| dewpoint_c(*p, *q))
        .collect();
    let agl: Vec<f64> = c.height_m.iter().map(|z| z - c.terrain_m).collect();
    let t2_c = c.t2_k - 273.15;
    let td2_c = dewpoint_c(c.psfc_pa, c.q2).min(t2_c);
    let (cape, cin, lcl, _lfc) = thermo::cape_cin_core(
        &p_hpa,
        &t_c,
        &td_c,
        &agl,
        c.psfc_pa / 100.0,
        t2_c,
        td2_c,
        "sb",
        100.0,
        300.0,
        None,
    );
    // 0-6 km bulk shear: 10 m wind to the wind interpolated at 6 km AGL.
    let mut u6 = f64::NAN;
    let mut v6 = f64::NAN;
    for k in 0..agl.len().saturating_sub(1) {
        if agl[k] <= 6000.0 && agl[k + 1] > 6000.0 {
            let f = (6000.0 - agl[k]) / (agl[k + 1] - agl[k]);
            u6 = c.u[k] + f * (c.u[k + 1] - c.u[k]);
            v6 = c.v[k] + f * (c.v[k + 1] - c.v[k]);
        }
    }
    let parcel = |kind: &str| {
        thermo::cape_cin_core(
            &p_hpa,
            &t_c,
            &td_c,
            &agl,
            c.psfc_pa / 100.0,
            t2_c,
            td2_c,
            kind,
            100.0,
            300.0,
            None,
        )
    };
    let (mlcape, mlcin, _, _) = parcel("ml");
    let (mucape, _, _, _) = parcel("mu");
    let wp = wind_profile(c);
    let nan = (f64::NAN, f64::NAN);
    let (storm_rm, storm_lm) = if wp.z.len() >= 3 && *wp.z.last().unwrap() >= 6000.0 {
        thermo::bunkers_storm_motion(&wp.p, &wp.u, &wp.v, &wp.z)
    } else {
        (nan, nan)
    };
    let storm_is_right = lat >= 0.0;
    let storm = if storm_is_right { storm_rm } else { storm_lm };
    SoundingIndices {
        sbcape: cape,
        sbcin: cin,
        lcl_m: lcl,
        pw_mm: c.precipitable_water_mm(),
        shear06_kt: ((u6 - c.u10).powi(2) + (v6 - c.v10).powi(2)).sqrt() * KNOTS,
        t2_f: c_to_f(t2_c),
        td2_f: c_to_f(td2_c),
        mslp_hpa: c.mslp_pa() / 100.0,
        mlcape,
        mlcin,
        mucape,
        shear01_kt: bulk_shear_kt(&wp, 1000.0),
        srh01: storm_relative_helicity(&wp, 1000.0, storm),
        srh03: storm_relative_helicity(&wp, 3000.0, storm),
        storm_rm,
        storm_lm,
        storm_is_right,
    }
}

/// The hodograph: the wind vector's tip from the surface to 10 km above
/// ground, coloured by layer, with the Bunkers storm motions marked.  Rings
/// are knots; east is right and north is up, so a southerly wind plots above
/// the centre.
fn draw_hodograph(img: &mut RgbaImage, rect: (f64, f64, f64, f64), w: &WindProfile, idx: &SoundingIndices) {
    let pts = layer_points(w, 10_000.0, &[1000.0, 3000.0, 6000.0]);
    let (cx, cy) = ((rect.0 + rect.2) / 2.0, (rect.1 + rect.3) / 2.0);
    let half = (rect.2 - rect.0).min(rect.3 - rect.1) / 2.0 - 8.0;
    let mut vmax: f64 = 30.0;
    for (_, u, v) in &pts {
        vmax = vmax.max((u * u + v * v).sqrt() * KNOTS);
    }
    for (u, v) in [idx.storm_rm, idx.storm_lm] {
        if u.is_finite() && v.is_finite() {
            vmax = vmax.max((u * u + v * v).sqrt() * KNOTS);
        }
    }
    let ring_max = ((vmax / 10.0).ceil() * 10.0).min(250.0);
    let scale = half / ring_max;
    let xy = |u_ms: f64, v_ms: f64| (cx + u_ms * KNOTS * scale, cy - v_ms * KNOTS * scale);
    let step = if ring_max > 120.0 { 40.0 } else if ring_max > 60.0 { 20.0 } else { 10.0 };
    draw_line_aa_width(img, rect.0, cy, rect.2, cy, GRID, 1);
    draw_line_aa_width(img, cx, rect.1, cx, rect.3, GRID, 1);
    let mut r = step;
    while r <= ring_max + 1e-6 {
        draw_circle_stroke_aa(img, cx, cy, r * scale, GRID, 1);
        draw_text(img, &format!("{r:.0}"), cx as i32 + 5, (cy - r * scale) as i32 + 3, SUBTLE, 1);
        r += step;
    }
    for s in pts.windows(2) {
        let zmid = 0.5 * (s[0].0 + s[1].0);
        let color = if zmid < 1000.0 {
            RED
        } else if zmid < 3000.0 {
            GREEN
        } else if zmid < 6000.0 {
            BLUE
        } else {
            PURPLE
        };
        let (a, b) = (xy(s[0].1, s[0].2), xy(s[1].1, s[1].2));
        draw_line_aa_width(img, a.0, a.1, b.0, b.1, color, 3);
    }
    if let Some(first) = pts.first() {
        let (x, y) = xy(first.1, first.2);
        draw_circle_fill_aa(img, x, y, 4.0, INK);
    }
    for (label, (u, v)) in [("RM", idx.storm_rm), ("LM", idx.storm_lm)] {
        if u.is_finite() && v.is_finite() {
            let (x, y) = xy(u, v);
            draw_circle_stroke_aa(img, x, y, 6.0, INK, 2);
            draw_text(img, label, x as i32 + 9, y as i32 - 8, INK, 2);
        }
    }
    for (a, b) in [
        ((rect.0, rect.1), (rect.2, rect.1)),
        ((rect.2, rect.1), (rect.2, rect.3)),
        ((rect.2, rect.3), (rect.0, rect.3)),
        ((rect.0, rect.3), (rect.0, rect.1)),
    ] {
        draw_line_aa_width(img, a.0, a.1, b.0, b.1, INK, 1);
    }
    draw_text(img, "Hodograph, surface to 10 km above ground (kt)", rect.0 as i32, rect.1 as i32 - 26, SUBTLE, 2);
    let ly = rect.3 as i32 + 12;
    for (i, (label, color)) in [("0-1 km", RED), ("1-3 km", GREEN), ("3-6 km", BLUE), ("6-10 km", PURPLE)].iter().enumerate() {
        let x = rect.0 as i32 + i as i32 * 112;
        draw_line_aa_width(img, x as f64, ly as f64 + 8.0, x as f64 + 26.0, ly as f64 + 8.0, *color, 3);
        draw_text(img, label, x + 32, ly, INK, 2);
    }
}

#[allow(clippy::too_many_arguments)]
fn draw_sounding(
    path: &Path,
    c: &Column,
    point: &PointSpec,
    headline: &str,
    brand: Option<&str>,
    init: &Timestamp,
    valid: &Timestamp,
    lead_h: f64,
    idx: &SoundingIndices,
) -> Result<(), String> {
    // The skew-T keeps its 1100-pixel column; the hodograph and the
    // kinematic table stand in a second column to its right.
    let (w, h) = (1580u32, 1250u32);
    let left_w = 1100i32;
    let mut img = white(w, h);
    let sk = SkewT {
        x0: 80.0,
        y0: 140.0,
        x1: 900.0,
        y1: 1050.0,
        p_bot: 1050.0,
        p_top: 100.0,
        t_left: -40.0,
        t_right: 50.0,
    };
    let rect = sk.rect();
    let clip = (rect.0 as i32, rect.1 as i32, rect.2 as i32, rect.3 as i32);

    // Header.
    draw_text_bold(&mut img, headline, 30, 20, INK, 4);
    draw_brand(&mut img, brand, headline);
    let sub = format!(
        "Sounding at {} ({})",
        point.name,
        latlon_text(point)
    );
    draw_text(&mut img, &sub, 30, 56, INK, 3);
    let when = format!(
        "Init {}   Valid {}   F{:03}",
        stamp_text(init),
        stamp_text(valid),
        lead_h.round() as i64
    );
    draw_text(&mut img, &when, 30, 84, SUBTLE, 2);

    // Isobars.
    for p in [1000.0, 925.0, 850.0, 700.0, 600.0, 500.0, 400.0, 300.0, 250.0, 200.0, 150.0, 100.0] {
        let y = sk.y(p);
        draw_line_aa_width(&mut img, rect.0, y, rect.2, y, GRID, 1);
        let label = format!("{p:.0}");
        let tw = text_width(&label, 2) as i32;
        draw_text(&mut img, &label, rect.0 as i32 - tw - 8, y as i32 - 9, SUBTLE, 2);
    }
    // Dry adiabats.
    let levels: Vec<f64> = (0..=190).map(|i| 1050.0 - i as f64 * 5.0).collect();
    for theta in (240..=460).step_by(10) {
        let pts: Vec<(f64, f64)> = levels
            .iter()
            .map(|p| {
                let t = theta as f64 * (p / 1000.0).powf(0.2857) - 273.15;
                (sk.x(t, *p), sk.y(*p))
            })
            .collect();
        for run in clip_runs(&pts, rect) {
            draw_polyline_aa(&mut img, &run, Rgba::new(236, 196, 160), 1);
        }
    }
    // Moist adiabats (pseudo-adiabats from 1000 hPa).
    let up: Vec<f64> = (0..=180).map(|i| 1050.0 - i as f64 * 5.0).collect();
    for t0 in (-12..=36).step_by(4) {
        // start at 1050 hPa with the temperature that gives t0 at 1000
        let start = thermo::moist_lapse(&[1000.0, 1050.0], t0 as f64)[1];
        let t = thermo::moist_lapse(&up, start);
        let pts: Vec<(f64, f64)> = up.iter().zip(t.iter()).map(|(p, t)| (sk.x(*t, *p), sk.y(*p))).collect();
        for run in clip_runs(&pts, rect) {
            draw_polyline_aa(&mut img, &run, Rgba::new(166, 214, 176), 1);
        }
    }
    // Mixing-ratio lines below 600 hPa.
    for wgkg in [1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 20.0] {
        let w = wgkg / 1000.0;
        let pts: Vec<(f64, f64)> = (0..=45)
            .map(|i| {
                let p = 1050.0 - i as f64 * 10.0;
                let e = w * p / (0.621_97 + w);
                let t = thermo::dewpoint_from_vapor_pressure(e);
                (sk.x(t, p), sk.y(p))
            })
            .collect();
        for run in clip_runs(&pts, rect) {
            draw_dashed(&mut img, &run, Rgba::new(176, 150, 210), 1, 6.0, 5.0);
        }
        let p = 1040.0;
        let e = w * p / (0.621_97 + w);
        let x = sk.x(thermo::dewpoint_from_vapor_pressure(e), p);
        if x > rect.0 && x < rect.2 - 10.0 {
            draw_text(&mut img, &format!("{wgkg:.0}"), x as i32 - 4, sk.y(p) as i32 - 16, Rgba::new(140, 110, 180), 1);
        }
    }
    // Isotherms.
    for t in (-120..=50).step_by(10) {
        let pts = vec![(sk.x(t as f64, 1050.0), sk.y(1050.0)), (sk.x(t as f64, 100.0), sk.y(100.0))];
        let color = if t == 0 { Rgba::new(120, 160, 230) } else { GRID };
        for run in clip_runs(&pts, rect) {
            draw_polyline_aa(&mut img, &run, color, if t == 0 { 2 } else { 1 });
        }
        let xb = sk.x(t as f64, 1050.0);
        if xb >= rect.0 && xb <= rect.2 {
            let label = format!("{t}");
            let tw = text_width(&label, 2) as i32;
            draw_text(&mut img, &label, xb as i32 - tw / 2, rect.3 as i32 + 8, SUBTLE, 2);
        }
    }
    draw_text(&mut img, "Temperature (\u{b0}C)", (rect.0 + rect.2) as i32 / 2 - 70, rect.3 as i32 + 32, SUBTLE, 2);
    draw_text(&mut img, "hPa", 36, rect.1 as i32 - 26, SUBTLE, 2);

    // Profiles.
    let p_hpa: Vec<f64> = c.pressure_pa.iter().map(|p| p / 100.0).collect();
    let t_c: Vec<f64> = c.temperature_k.iter().map(|t| t - 273.15).collect();
    let td_c: Vec<f64> = c.pressure_pa.iter().zip(c.qv.iter()).map(|(p, q)| dewpoint_c(*p, *q)).collect();
    let keep = |p: &f64| *p >= sk.p_top;
    let tpts: Vec<(f64, f64)> = p_hpa.iter().zip(t_c.iter()).filter(|(p, _)| keep(p)).map(|(p, t)| (sk.x(*t, *p), sk.y(*p))).collect();
    let dpts: Vec<(f64, f64)> = p_hpa.iter().zip(td_c.iter()).filter(|(p, _)| keep(p)).map(|(p, t)| (sk.x(*t, *p), sk.y(*p))).collect();
    // Surface-based parcel from the 2 m state at the surface pressure.
    let psfc = c.psfc_pa / 100.0;
    let t2 = c.t2_k - 273.15;
    let td2 = dewpoint_c(c.psfc_pa, c.q2).min(t2);
    let mut parcel_p = vec![psfc];
    parcel_p.extend(p_hpa.iter().copied().filter(|p| *p < psfc && keep(p)));
    let parcel_t = thermo::parcel_profile(&parcel_p, t2, td2);
    let ppts: Vec<(f64, f64)> = parcel_p.iter().zip(parcel_t.iter()).map(|(p, t)| (sk.x(*t, *p), sk.y(*p))).collect();
    // Positive area shading between parcel and environment.
    let env_at = |p: f64| -> f64 {
        for k in 0..p_hpa.len().saturating_sub(1) {
            if p_hpa[k] >= p && p_hpa[k + 1] <= p {
                let f = (p_hpa[k].ln() - p.ln()) / (p_hpa[k].ln() - p_hpa[k + 1].ln());
                return t_c[k] + f * (t_c[k + 1] - t_c[k]);
            }
        }
        f64::NAN
    };
    for k in 0..parcel_p.len().saturating_sub(1) {
        let (pa, pb) = (parcel_p[k], parcel_p[k + 1]);
        let (ea, eb) = (env_at(pa), env_at(pb));
        let (ta, tb) = (parcel_t[k], parcel_t[k + 1]);
        if ea.is_finite() && eb.is_finite() && ta > ea && tb > eb {
            let ring = vec![
                (sk.x(ea, pa), sk.y(pa)),
                (sk.x(ta, pa), sk.y(pa)),
                (sk.x(tb, pb), sk.y(pb)),
                (sk.x(eb, pb), sk.y(pb)),
            ];
            fill_polygon(&mut img, &[ring], Rgba::with_alpha(230, 80, 60, 60), Some(clip));
        }
    }
    for run in clip_runs(&dpts, rect) {
        draw_polyline_aa(&mut img, &run, GREEN, 3);
    }
    for run in clip_runs(&tpts, rect) {
        draw_polyline_aa(&mut img, &run, RED, 3);
    }
    for run in clip_runs(&ppts, rect) {
        draw_dashed(&mut img, &run, INK, 2, 9.0, 6.0);
    }
    // Frame.
    for (a, b) in [
        ((rect.0, rect.1), (rect.2, rect.1)),
        ((rect.2, rect.1), (rect.2, rect.3)),
        ((rect.2, rect.3), (rect.0, rect.3)),
        ((rect.0, rect.3), (rect.0, rect.1)),
    ] {
        draw_line_aa_width(&mut img, a.0, a.1, b.0, b.1, INK, 1);
    }

    // Wind barbs, every ~25 hPa, knots.
    let bx = rect.2 + 70.0;
    draw_line_aa_width(&mut img, bx, rect.1, bx, rect.3, GRID, 1);
    let mut last_y = f64::NEG_INFINITY;
    for k in 0..p_hpa.len() {
        if p_hpa[k] < sk.p_top || p_hpa[k] > sk.p_bot {
            continue;
        }
        let y = sk.y(p_hpa[k]);
        if (y - last_y).abs() < 28.0 {
            continue;
        }
        last_y = y;
        draw_wind_barb(&mut img, bx, y, c.u[k] * KNOTS, c.v[k] * KNOTS, INK, 34.0, 2);
    }
    draw_text(&mut img, "Wind (kt)", bx as i32 - 30, rect.3 as i32 + 8, SUBTLE, 2);

    // Indices.
    let y0 = 1120;
    let cols = [
        ("SBCAPE", format!("{:.0} J/kg", idx.sbcape)),
        ("SBCIN", format!("{:.0} J/kg", idx.sbcin)),
        ("LCL", format!("{:.0} m AGL", idx.lcl_m)),
        ("PW", format!("{:.2} in", idx.pw_mm / 25.4)),
        ("0-6 km shear", format!("{:.0} kt", idx.shear06_kt)),
        ("2 m T / Td", format!("{:.0} / {:.0} \u{b0}F", idx.t2_f, idx.td2_f)),
        ("MSLP", format!("{:.1} hPa", idx.mslp_hpa)),
    ];
    let col_w = (left_w - 60) / 4;
    for (i, (name, value)) in cols.iter().enumerate() {
        let x = 30 + (i as i32 % 4) * col_w;
        let y = y0 + (i as i32 / 4) * 56;
        draw_text(&mut img, name, x, y, SUBTLE, 2);
        draw_text_bold(&mut img, value, x, y + 20, INK, 3);
    }
    // Legend.
    let ly = 1050 + 70;
    let lx = left_w - 250;
    let _ = ly;
    for (i, (label, color)) in [("Temperature", RED), ("Dewpoint", GREEN), ("Surface parcel", INK)].iter().enumerate() {
        let yy = 1178 + i as i32 * 22;
        draw_line_aa_width(&mut img, lx as f64, yy as f64 + 8.0, lx as f64 + 34.0, yy as f64 + 8.0, *color, 3);
        draw_text(&mut img, label, lx + 44, yy, INK, 2);
    }
    // Hodograph and kinematics, right column.
    let hodo = (1110.0, 140.0, 1550.0, 580.0);
    let wp = wind_profile(c);
    draw_hodograph(&mut img, hodo, &wp, idx);
    let mover = if idx.storm_is_right { "right mover" } else { "left mover" };
    let storm = if idx.storm_is_right { idx.storm_rm } else { idx.storm_lm };
    let table = [
        ("MLCAPE", value_text(idx.mlcape, "J/kg")),
        ("MLCIN", value_text(idx.mlcin, "J/kg")),
        ("MUCAPE", value_text(idx.mucape, "J/kg")),
        ("0-1 km shear", value_text(idx.shear01_kt, "kt")),
        ("0-1 km helicity", value_text(idx.srh01, "m2/s2")),
        ("0-3 km helicity", value_text(idx.srh03, "m2/s2")),
        ("Storm motion", direction_speed_text(storm.0, storm.1)),
    ];
    for (i, (name, value)) in table.iter().enumerate() {
        let x = hodo.0 as i32 + (i as i32 % 2) * 230;
        let y = 660 + (i as i32 / 2) * 56;
        draw_text(&mut img, name, x, y, SUBTLE, 2);
        draw_text_bold(&mut img, value, x, y + 20, INK, 3);
    }
    let note = format!("Helicity and storm motion: Bunkers {mover}");
    draw_text(&mut img, &note, hodo.0 as i32, 660 + 4 * 56, SUBTLE, 2);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    img.save(path).map_err(|e| format!("write {}: {e}", path.display()))
}

// ---------------------------------------------------------------------------
// The meteogram.
// ---------------------------------------------------------------------------

struct Series {
    leads_h: Vec<f64>,
    valid: Vec<Timestamp>,
    t2_f: Vec<f64>,
    td2_f: Vec<f64>,
    wind_kt: Vec<f64>,
    u10_kt: Vec<f64>,
    v10_kt: Vec<f64>,
    mslp_hpa: Vec<f64>,
    precip_in: Vec<f64>,
}

struct Panel {
    x0: f64,
    y0: f64,
    x1: f64,
    y1: f64,
    lo: f64,
    hi: f64,
}

impl Panel {
    fn y(&self, v: f64) -> f64 {
        self.y1 - (v - self.lo) / (self.hi - self.lo) * (self.y1 - self.y0)
    }
}

fn panel_axes(img: &mut RgbaImage, p: &Panel, title: &str, unit: &str, xs: &[f64]) -> f64 {
    let step = nice_step(p.hi - p.lo, 4.0);
    let mut v = (p.lo / step).ceil() * step;
    while v <= p.hi + 1e-9 {
        let y = p.y(v);
        draw_line_aa_width(img, p.x0, y, p.x1, y, GRID, 1);
        let label = tick_label(v, step);
        let tw = text_width(&label, 2) as i32;
        draw_text(img, &label, p.x0 as i32 - tw - 10, y as i32 - 9, SUBTLE, 2);
        v += step;
    }
    for &x in xs {
        draw_line_aa_width(img, x, p.y0, x, p.y1, Rgba::new(236, 238, 242), 1);
    }
    draw_text_bold(img, title, p.x0 as i32, p.y0 as i32 - 26, INK, 2);
    let uw = text_width(unit, 2) as i32;
    draw_text(img, unit, p.x1 as i32 - uw, p.y0 as i32 - 24, SUBTLE, 2);
    for (a, b) in [
        ((p.x0, p.y0), (p.x1, p.y0)),
        ((p.x1, p.y0), (p.x1, p.y1)),
        ((p.x1, p.y1), (p.x0, p.y1)),
        ((p.x0, p.y1), (p.x0, p.y0)),
    ] {
        draw_line_aa_width(img, a.0, a.1, b.0, b.1, Rgba::new(150, 156, 164), 1);
    }
    step
}

fn padded(lo: f64, hi: f64, min_span: f64) -> (f64, f64) {
    // No valid samples: retain a labelled empty panel with a finite axis.
    let (lo, hi) = if lo.is_finite() && hi.is_finite() && hi >= lo {
        (lo, hi)
    } else {
        (0.0, min_span)
    };
    let span = (hi - lo).max(min_span);
    let mid = 0.5 * (hi + lo);
    let lo = mid - span * 0.6;
    let hi = mid + span * 0.6;
    let step = nice_step(hi - lo, 4.0);
    ((lo / step).floor() * step, (hi / step).ceil() * step)
}

/// The model's name and start, bold and right-aligned at the top right (the
/// picture's identity, as on the website sounding window).  It steps down in
/// size before it would run into the headline; nothing is drawn without it.
/// The top-right label when the caller passed no `--brand`: "WOOF Hex · Init 22 Oct 2026 00Z" (a model label that
/// already names WOOF is kept as it is).
fn default_brand(model_label: &str, init: &Timestamp) -> String {
    const MONTHS: [&str; 12] = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
    let label = model_label.trim();
    let model = if label.starts_with("WOOF") { label.to_string() } else { "WOOF Hex".to_string() };
    let month = MONTHS.get(init.month.saturating_sub(1) as usize).copied().unwrap_or("?");
    format!("{model} · Init {} {month} {} {:02}Z", init.day, init.year, init.hour)
}

fn draw_brand(img: &mut RgbaImage, brand: Option<&str>, headline: &str) {
    let Some(brand) = brand else { return };
    let right = img.width() as i32 - 30;
    let left_limit = 30 + text_width_bold(headline, 4) as i32 + 40;
    let mut scale = 3;
    while scale > 1 && right - (text_width_bold(brand, scale) as i32) < left_limit {
        scale -= 1;
    }
    let x = right - text_width_bold(brand, scale) as i32;
    draw_text_bold(img, brand, x, 26, INK, scale);
}

fn draw_meteogram(
    path: &Path,
    s: &Series,
    point: &PointSpec,
    headline: &str,
    brand: Option<&str>,
    init: &Timestamp,
) -> Result<(), String> {
    let (w, h) = (1400u32, 1300u32);
    let mut img = white(w, h);
    draw_text_bold(&mut img, headline, 30, 20, INK, 4);
    draw_brand(&mut img, brand, headline);
    draw_text(
        &mut img,
        &format!("Meteogram at {} ({})", point.name, latlon_text(point)),
        30,
        56,
        INK,
        3,
    );
    draw_text(&mut img, &format!("Init {}", stamp_text(init)), 30, 84, SUBTLE, 2);
    let (x0, x1) = (110.0, w as f64 - 50.0);
    let lead_max = s.leads_h.last().copied().unwrap_or(1.0).max(1e-6);
    let lead_min = s.leads_h.first().copied().unwrap_or(0.0);
    let span = (lead_max - lead_min).max(1e-6);
    // The first and last leads sit one barb length inside the panel frame, so
    // a wind barb (26 px shaft plus its feathers) drawn at the final lead stays
    // inside the panel instead of running over the frame and off the picture.
    const LEAD_INSET: f64 = 40.0;
    let xof = |lead: f64| (x0 + LEAD_INSET) + (lead - lead_min) / span * (x1 - x0 - 2.0 * LEAD_INSET);
    let xs: Vec<f64> = s.leads_h.iter().map(|l| xof(*l)).collect();
    let tops = [150.0, 430.0, 680.0, 930.0];
    let heights = [220.0, 190.0, 190.0, 220.0];

    // Temperature / dewpoint.
    let all: Vec<f64> = s.t2_f.iter().chain(s.td2_f.iter()).copied().filter(|v| v.is_finite()).collect();
    let (lo, hi) = padded(
        all.iter().copied().fold(f64::INFINITY, f64::min),
        all.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        10.0,
    );
    let p = Panel { x0, y0: tops[0], x1, y1: tops[0] + heights[0], lo, hi };
    panel_axes(&mut img, &p, "2 m temperature and dewpoint", "\u{b0}F", &xs);
    let tl: Vec<(f64, f64)> = xs.iter().zip(s.t2_f.iter()).map(|(x, v)| (*x, p.y(*v))).collect();
    let dl: Vec<(f64, f64)> = xs.iter().zip(s.td2_f.iter()).map(|(x, v)| (*x, p.y(*v))).collect();
    draw_polyline_aa(&mut img, &dl, GREEN, 3);
    draw_polyline_aa(&mut img, &tl, RED, 3);
    for (x, y) in tl.iter().chain(dl.iter()) {
        draw_circle_fill_aa(&mut img, *x, *y, 3.0, INK);
    }
    draw_text(&mut img, "temperature", x1 as i32 - 300, tops[0] as i32 - 24, RED, 2);
    draw_text(&mut img, "dewpoint", x1 as i32 - 190, tops[0] as i32 - 24, GREEN, 2);

    // Wind.
    let wmax = s.wind_kt.iter().copied().fold(0.0f64, f64::max);
    let p = Panel { x0, y0: tops[1], x1, y1: tops[1] + heights[1], lo: 0.0, hi: (wmax * 1.35).max(10.0) };
    let step = nice_step(p.hi, 4.0);
    let p = Panel { hi: (p.hi / step).ceil() * step, ..p };
    panel_axes(&mut img, &p, "10 m wind", "kt", &xs);
    let wl: Vec<(f64, f64)> = xs.iter().zip(s.wind_kt.iter()).map(|(x, v)| (*x, p.y(*v))).collect();
    draw_polyline_aa(&mut img, &wl, BLUE, 3);
    let every = (s.leads_h.len() / 24).max(1);
    for (i, x) in xs.iter().enumerate() {
        if i % every == 0 {
            draw_wind_barb(&mut img, *x, p.y0 + 32.0, s.u10_kt[i], s.v10_kt[i], INK, 26.0, 2);
        }
    }

    // Pressure.
    let (lo, hi) = padded(
        s.mslp_hpa.iter().copied().fold(f64::INFINITY, f64::min),
        s.mslp_hpa.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        8.0,
    );
    let p = Panel { x0, y0: tops[2], x1, y1: tops[2] + heights[2], lo, hi };
    panel_axes(&mut img, &p, "Sea-level pressure (reduced from the surface)", "hPa", &xs);
    let pl: Vec<(f64, f64)> = xs.iter().zip(s.mslp_hpa.iter()).map(|(x, v)| (*x, p.y(*v))).collect();
    draw_polyline_aa(&mut img, &pl, INK, 3);

    // Precipitation: interval bars and running total.
    let total_max = s.precip_in.iter().copied().fold(0.0f64, f64::max);
    let hi = if total_max < 0.1 { 0.1 } else { total_max * 1.15 };
    let step = nice_step(hi, 4.0);
    let p = Panel { x0, y0: tops[3], x1, y1: tops[3] + heights[3], lo: 0.0, hi: (hi / step).ceil() * step };
    panel_axes(&mut img, &p, "Precipitation: per interval (bars) and since init (line)", "in", &xs);
    for i in 1..xs.len() {
        let amount = (s.precip_in[i] - s.precip_in[i - 1]).max(0.0);
        if amount <= 0.0 {
            continue;
        }
        let half = ((xs[i] - xs[i - 1]) * 0.35).max(2.0);
        let ring = vec![
            (xs[i] - half, p.y(0.0)),
            (xs[i] + half, p.y(0.0)),
            (xs[i] + half, p.y(amount)),
            (xs[i] - half, p.y(amount)),
        ];
        fill_polygon(&mut img, &[ring], Rgba::new(70, 150, 90), None);
    }
    let al: Vec<(f64, f64)> = xs.iter().zip(s.precip_in.iter()).map(|(x, v)| (*x, p.y(*v))).collect();
    draw_polyline_aa(&mut img, &al, Rgba::new(20, 90, 50), 3);

    // Time axis.
    let label_every = (s.leads_h.len() / 12).max(1);
    let ybase = (tops[3] + heights[3]) as i32 + 10;
    for (i, x) in xs.iter().enumerate() {
        if i % label_every != 0 && i + 1 != xs.len() {
            continue;
        }
        let t = &s.valid[i];
        let a = format!("{:02}Z", t.hour);
        let b = format!("{} {:02}/{:02}", weekday(t), t.month, t.day);
        let c = format!("F{:03}", s.leads_h[i].round() as i64);
        for (k, line) in [a, b, c].iter().enumerate() {
            let tw = text_width(line, 2) as i32;
            draw_text(&mut img, line, *x as i32 - tw / 2, ybase + k as i32 * 20, if k == 2 { SUBTLE } else { INK }, 2);
        }
    }
    let _ = text_width_bold;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    img.save(path).map_err(|e| format!("write {}: {e}", path.display()))
}

fn json_num(v: f64) -> String {
    if v.is_finite() { format!("{v:.4}") } else { "null".to_string() }
}

/// One frame's columns as an `rw_soundings` JSON column frame (rw-soundings `input.rs`, JsonFrame): lowest level
/// first, in the bridge's units (hPa, m MSL, C, m/s, Pa/s), with the surface anchor and what the run says about itself.
/// Omega comes from the frame's W by the column store's formula; a frame with no W leaves it out and the reader fills
/// it as missing. THE BREAKAGE THIS PREVENTS (the project owner, 2026-10-30): a hex box sounding drew this tool's own skew-T, not the
/// site's sounding window the column store draws; and then (the post-switch check, 2026-10-31) that window with no OMEGA
/// panel, because this frame carried none.
fn column_frame(valid: &Timestamp, init: &Timestamp, shape: Option<&str>, cols: &[Column]) -> serde_json::Value {
    let num = |v: f64| if v.is_finite() { serde_json::json!(v) } else { serde_json::Value::Null };
    let list = |v: Vec<f64>| v.into_iter().map(num).collect::<Vec<_>>();
    let columns: Vec<serde_json::Value> = cols
        .iter()
        .map(|c| {
            let t: Vec<f64> = c.temperature_k.iter().map(|t| t - 273.15).collect();
            let td: Vec<f64> = c.pressure_pa.iter().zip(&c.qv).zip(&t)
                .map(|((p, q), t)| dewpoint_c(*p, *q).min(*t))
                .collect();
            let t2_c = c.t2_k - 273.15;
            let mut column = serde_json::json!({
                "p": list(c.pressure_pa.iter().map(|p| p / 100.0).collect()),
                "z": list(c.height_m.clone()),
                "t": list(t),
                "td": list(td),
                "u": list(c.u.clone()),
                "v": list(c.v.clone()),
                "sfc": {
                    "psfc_hpa": num(c.psfc_pa / 100.0),
                    "orog_m": num(c.terrain_m),
                    "t2_c": num(t2_c),
                    "td2_c": num(dewpoint_c(c.psfc_pa, c.q2).min(t2_c)),
                    "u10": num(c.u10),
                    "v10": num(c.v10),
                },
            });
            if let Some(omega) = &c.omega_pa_s {
                column["omega"] = serde_json::json!(list(omega.clone()));
            }
            column
        })
        .collect();
    let mut frame = serde_json::json!({ "valid": format!("{}Z", valid.iso()), "init": format!("{}Z", init.iso()), "model": "Hex", "columns": columns });
    if let Some(shape) = shape {
        frame["shape"] = serde_json::json!(shape);
    }
    frame
}

fn run() -> Result<(), String> {
    let args = parse()?;
    let started = std::time::Instant::now();
    let mesh = read_mesh_coordinates(&args.mesh, None).map_err(|e| e.to_string())?;
    let spacing = read_mesh_cell_spacing(&args.mesh).map_err(|e| e.to_string())?;
    let topology = read_mesh_topology(&args.mesh, mesh.n_cells()).map_err(|e| e.to_string())?;
    let sampler = PointSampler::build(&mesh, &topology, &spacing.spacing_metres, args.points.clone())
        .map_err(|e| e.to_string())?;
    let grid_label = args.grid_label.clone().or_else(|| {
        rw_mpas::convert::grid_label(Some(&args.model_label), &spacing.spacing_metres)
    });
    let headline = grid_label.clone().unwrap_or_else(|| args.model_label.clone());
    for (k, p) in sampler.points.iter().enumerate() {
        if sampler.off_mesh[k] {
            return Err(format!(
                "point {} ({}) is {:.1} km from the nearest cell centre, beyond that cell's {:.1} km \
                 spacing: it is off this mesh, and a column there would be the boundary ring's",
                p.name,
                latlon_text(p),
                sampler.nearest_km[k],
                sampler.local_spacing_m[k] / 1000.0
            ));
        }
        println!(
            "POINT\t{}\t{}\tnearest_km={:.2}\tlocal_dx_km={:.2}",
            p.slug(),
            latlon_text(p),
            sampler.nearest_km[k],
            sampler.local_spacing_m[k] / 1000.0
        );
    }

    // Read frames in valid-time order.
    let mut frames: Vec<(Timestamp, PathBuf)> = Vec::new();
    for path in &args.history {
        let t = Timestamp::parse(path.file_name().and_then(|n| n.to_str()).unwrap_or_default())
            .map_err(|e| e.to_string())?;
        frames.push((t, path.clone()));
    }
    frames.sort_by_key(|(t, _)| t.label());
    let init = args.simulation_start.unwrap_or(frames[0].0);
    // Top right: the caller's brand, else the model and its start. Never empty, never a tool's name.
    let brand = args.brand.clone().unwrap_or_else(|| default_brand(&args.model_label, &init));
    let brand = Some(brand);
    let n = sampler.points.len();
    let mut series: Vec<Series> = (0..n)
        .map(|_| Series {
            leads_h: vec![],
            valid: vec![],
            t2_f: vec![],
            td2_f: vec![],
            wind_kt: vec![],
            u10_kt: vec![],
            v10_kt: vec![],
            mslp_hpa: vec![],
            precip_in: vec![],
        })
        .collect();
    let mut soundings_json: Vec<String> = Vec::new();
    // Each frame is read once, at the cells the points' triangles touch only,
    // and the frames are read in parallel (read_history_cells names the
    // breakage: whole-file hashes and whole-mesh reads per frame).
    let (cells, compact) = sampler.compacted();
    let read_started = std::time::Instant::now();
    let frame_columns: Vec<Vec<Column>> = frames
        .par_iter()
        .map(|(valid, path)| {
            let frame = read_history_cells_on_mesh(path, Some(&args.init), &cells, &POINT_FIELDS, Some(*valid), Some(&mesh))
                .map_err(|e| e.to_string())?;
            columns(&compact, &frame).map_err(|e| e.to_string())
        })
        .collect::<Result<_, String>>()?;
    println!(
        "POINTS READ\tframes={}\tcells={}\tseconds={:.3}",
        frames.len(),
        cells.len(),
        read_started.elapsed().as_secs_f64()
    );
    if let Some(path) = &args.columns_json {
        let (valid, _) = &frames[0];
        let body = serde_json::to_string(&column_frame(valid, &init, args.grid_label.as_deref(), &frame_columns[0]))
            .map_err(|e| e.to_string())?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        std::fs::write(path, body).map_err(|e| e.to_string())?;
        println!("COLUMNS\t{}\tpoints={}", path.display(), n);
    }
    // The site's window draws these columns: the tool's own skew-T would be CPU spent beside the forecast for nothing.
    let draw_pictures = !(args.soundings_only && args.columns_json.is_some());
    for ((valid, _), cols) in frames.iter().zip(frame_columns.iter()) {
        let lead_h = valid.seconds_since(&init) as f64 / 3600.0;
        for (k, c) in cols.iter().enumerate() {
            let s = &mut series[k];
            let t2_c = c.t2_k - 273.15;
            s.leads_h.push(lead_h);
            s.valid.push(*valid);
            s.t2_f.push(c_to_f(t2_c));
            s.td2_f.push(c_to_f(dewpoint_c(c.psfc_pa, c.q2).min(t2_c)));
            if !c.u10.is_finite() || !c.v10.is_finite() {
                println!("POINTS MISSING\t{}\tf{:.0}\tU10/V10 unavailable; wind diagnostics show gaps or n/a", sampler.points[k].slug(), lead_h);
            }
            s.u10_kt.push(c.u10 * KNOTS);
            s.v10_kt.push(c.v10 * KNOTS);
            s.wind_kt.push((c.u10 * c.u10 + c.v10 * c.v10).sqrt() * KNOTS);
            s.mslp_hpa.push(c.mslp_pa() / 100.0);
            s.precip_in.push(c.precip_mm / 25.4);
            let whole = (lead_h / args.sounding_every_hours).round() * args.sounding_every_hours;
            if draw_pictures && (lead_h - whole).abs() < 1e-6 {
                let p = &sampler.points[k];
                let idx = sounding_indices(c, p.lat);
                let file = args.out_dir.join(p.slug()).join(format!(
                    "sounding_{}_{:04}{:02}{:02}_{:02}{:02}_f{:03}.png",
                    p.slug(),
                    valid.year,
                    valid.month,
                    valid.day,
                    valid.hour,
                    valid.minute,
                    lead_h.round() as i64
                ));
                draw_sounding(&file, c, p, &headline, brand.as_deref(), &init, valid, lead_h, &idx)?;
                println!(
                    "SOUNDING\t{}\t{}\tf{:03}\tsbcape={:.0}\tsbcin={:.0}\tpw_mm={:.1}",
                    p.slug(),
                    file.display(),
                    lead_h.round() as i64,
                    idx.sbcape,
                    idx.sbcin,
                    idx.pw_mm
                );
                soundings_json.push(format!(
                    "{{\"point\": \"{}\", \"valid\": \"{}\", \"lead_hours\": {}, \"png\": \"{}\", \"sbcape_j_kg\": {}, \"sbcin_j_kg\": {}, \"lcl_m_agl\": {}, \"pw_mm\": {}, \"shear_0_6km_kt\": {}, \"mlcape_j_kg\": {}, \"mlcin_j_kg\": {}, \"mucape_j_kg\": {}, \"shear_0_1km_kt\": {}, \"srh_0_1km_m2_s2\": {}, \"srh_0_3km_m2_s2\": {}, \"bunkers_right_u_v_m_s\": [{}, {}], \"bunkers_left_u_v_m_s\": [{}, {}], \"helicity_relative_to\": \"{}\", \"levels\": {}}}",
                    p.slug(),
                    valid.iso(),
                    json_num(lead_h),
                    file.display(),
                    json_num(idx.sbcape),
                    json_num(idx.sbcin),
                    json_num(idx.lcl_m),
                    json_num(idx.pw_mm),
                    json_num(idx.shear06_kt),
                    json_num(idx.mlcape),
                    json_num(idx.mlcin),
                    json_num(idx.mucape),
                    json_num(idx.shear01_kt),
                    json_num(idx.srh01),
                    json_num(idx.srh03),
                    json_num(idx.storm_rm.0),
                    json_num(idx.storm_rm.1),
                    json_num(idx.storm_lm.0),
                    json_num(idx.storm_lm.1),
                    if idx.storm_is_right { "bunkers right mover" } else { "bunkers left mover" },
                    c.pressure_pa.len()
                ));
            }
        }
    }
    let mut meteograms_json = Vec::new();
    for (k, p) in sampler.points.iter().enumerate() {
        let file = args.out_dir.join(p.slug()).join(format!(
            "meteogram_{}_{:04}{:02}{:02}_{:02}.png",
            p.slug(),
            init.year,
            init.month,
            init.day,
            init.hour
        ));
        if args.soundings_only {
            continue;
        }
        draw_meteogram(&file, &series[k], p, &headline, brand.as_deref(), &init)?;
        println!("METEOGRAM\t{}\t{}\tframes={}", p.slug(), file.display(), series[k].leads_h.len());
        let s = &series[k];
        let list = |v: &[f64]| v.iter().map(|x| json_num(*x)).collect::<Vec<_>>().join(", ");
        meteograms_json.push(format!(
            "{{\"point\": \"{}\", \"name\": \"{}\", \"lat\": {}, \"lon\": {}, \"nearest_km\": {}, \"local_dx_km\": {}, \"png\": \"{}\", \"lead_hours\": [{}], \"t2_f\": [{}], \"td2_f\": [{}], \"wind10_kt\": [{}], \"mslp_hpa\": [{}], \"precip_total_in\": [{}]}}",
            p.slug(),
            p.name.replace('"', "'"),
            p.lat,
            p.lon,
            json_num(sampler.nearest_km[k]),
            json_num(sampler.local_spacing_m[k] / 1000.0),
            file.display(),
            list(&s.leads_h),
            list(&s.t2_f),
            list(&s.td2_f),
            list(&s.wind_kt),
            list(&s.mslp_hpa),
            list(&s.precip_in)
        ));
    }
    if let Some(path) = &args.json {
        let body = format!(
            "{{\n  \"schema\": \"gpuwm-hex.points/v1\",\n  \"engine\": \"rw_hex_points (rust)\",\n  \"interpolation\": \"linear barycentric on the mesh Delaunay triangles (cellsOnVertex)\",\n  \"mslp\": \"NCEP Shuell reduction (as the GFS post computes PRMSL) from the lowest model level: a 6.5 K/km column below the terrain, its sea-level temperature adjusted only above 290.66 K\",\n  \"headline\": \"{}\",\n  \"brand\": {},\n  \"init\": \"{}\",\n  \"linear\": {{\"contained\": {}, \"widened\": {}, \"clamped\": {}}},\n  \"soundings\": [\n    {}\n  ],\n  \"meteograms\": [\n    {}\n  ]\n}}\n",
            headline,
            brand.as_ref().map_or("null".to_string(), |b| format!("\"{}\"", b.replace('\\', "\\\\").replace('"', "\\\""))),
            init.iso(),
            sampler.linear.contained,
            sampler.linear.widened,
            sampler.linear.clamped,
            soundings_json.join(",\n    "),
            meteograms_json.join(",\n    ")
        );
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        std::fs::write(path, body).map_err(|e| e.to_string())?;
    }
    println!(
        "FINISHED\tpoints={}\tframes={}\tsoundings={}\telapsed_s={:.3}",
        n,
        frames.len(),
        soundings_json.len(),
        started.elapsed().as_secs_f64()
    );
    Ok(())
}

fn main() -> ExitCode {
    let _ = std::hint::black_box(GPUWM_BRIDGE_SOURCE_REV_STAMP);
    let argv: Vec<String> = std::env::args().skip(1).collect();
    if argv.iter().any(|a| a == "--help" || a == "-h") {
        println!("{USAGE}");
        return ExitCode::SUCCESS;
    }
    if argv.iter().any(|a| a == "--abi") {
        println!("{ABI}");
        return ExitCode::SUCCESS;
    }
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(m) => {
            eprintln!("rw_hex_points: {m}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_column_frame_is_in_the_sounding_windows_units_and_shape() {
        let init = Timestamp::parse("2026-10-23_18:00:00").unwrap();
        let valid = Timestamp::parse("2026-10-23_21:00:00").unwrap();
        let column = Column {
            pressure_pa: vec![90000.0, 50000.0],
            temperature_k: vec![290.0, 260.0],
            qv: vec![0.008, 1e-9],
            height_m: vec![1000.0, 5500.0],
            u: vec![5.0, 20.0],
            v: vec![-1.0, f64::NAN],
            terrain_m: 900.0,
            psfc_pa: 91000.0,
            t2_k: 291.15,
            q2: 0.009,
            u10: 3.0,
            v10: -2.0,
            precip_mm: 0.0,
            refl_max_dbz: None,
            omega_pa_s: Some(vec![-0.25, f64::NAN]),
        };
        let frame = column_frame(&valid, &init, Some("Zoom globe"), std::slice::from_ref(&column));
        assert_eq!(frame["valid"], "2026-10-23T21:00:00Z");
        assert_eq!(frame["init"], "2026-10-23T18:00:00Z");
        assert_eq!(frame["model"], "Hex");
        assert_eq!(frame["shape"], "Zoom globe");
        let c = &frame["columns"][0];
        assert_eq!(c["p"][0].as_f64(), Some(900.0));
        assert!((c["t"][0].as_f64().unwrap() - 16.85).abs() < 1e-9);
        assert!(c["td"][0].as_f64().unwrap() <= c["t"][0].as_f64().unwrap());
        assert_eq!(c["z"][1].as_f64(), Some(5500.0));
        // JSON has no NaN: a missing wind is null, which the sounding reader takes back as missing.
        assert!(c["v"][1].is_null());
        assert_eq!(c["sfc"]["psfc_hpa"].as_f64(), Some(910.0));
        assert_eq!(c["sfc"]["orog_m"].as_f64(), Some(900.0));
        assert!((c["sfc"]["t2_c"].as_f64().unwrap() - 18.0).abs() < 1e-9);
        assert_eq!(c["omega"][0].as_f64(), Some(-0.25));
        assert!(c["omega"][1].is_null());
    }

    #[test]
    fn the_abi_line_keeps_the_marker_older_doors_look_for_and_names_the_column_frame() {
        assert!(ABI.starts_with("gpuwm-rw-hex-points-v1\tPOINT\tSOUNDING\tMETEOGRAM\tFINISHED"));
        assert!(ABI.split('\t').any(|w| w == "COLUMNS-JSON"));
    }

    #[test]
    fn no_brand_still_names_the_model_and_its_start() {
        let init = Timestamp::parse("2026-10-23_18:00:00").unwrap();
        assert_eq!(default_brand("hex", &init), "WOOF Hex · Init 23 Oct 2026 18Z");
        assert_eq!(default_brand("WOOF Hex", &init), "WOOF Hex · Init 23 Oct 2026 18Z");
    }

    #[test]
    fn absent_samples_have_a_finite_empty_axis() {
        let (lo, hi) = padded(f64::INFINITY, f64::NEG_INFINITY, 10.0);
        assert!(lo.is_finite() && hi.is_finite() && hi > lo);
        let (lo, hi) = padded(4.0, 4.0, 10.0);
        assert!(lo < 4.0 && hi > 4.0);
    }

    #[test]
    fn nice_steps_are_1_2_5() {
        assert_eq!(nice_step(10.0, 4.0), 2.0);
        assert_eq!(nice_step(100.0, 4.0), 20.0);
        assert_eq!(nice_step(0.35, 4.0), 0.1);
    }

    #[test]
    fn clipping_keeps_inside_runs_and_cuts_at_the_edge() {
        let runs = clip_runs(&[(-5.0, 5.0), (5.0, 5.0), (15.0, 5.0)], (0.0, 0.0, 10.0, 10.0));
        assert_eq!(runs.len(), 1);
        assert!((runs[0][0].0 - 0.0).abs() < 1e-6);
        assert!((runs[0].last().unwrap().0 - 10.0).abs() < 1e-6);
    }

    fn profile(z: &[f64], u: &[f64], v: &[f64]) -> WindProfile {
        WindProfile {
            z: z.to_vec(),
            u: u.to_vec(),
            v: v.to_vec(),
            p: z.iter().map(|z| 1000.0 - z / 10.0).collect(),
        }
    }

    #[test]
    fn helicity_is_the_area_the_storm_relative_hodograph_sweeps() {
        // Winds veering from southerly to westerly over 3 km, storm at rest.
        let w = profile(&[10.0, 1000.0, 2000.0, 3000.0], &[0.0, 5.0, 10.0, 10.0], &[10.0, 10.0, 5.0, 0.0]);
        assert!((storm_relative_helicity(&w, 3000.0, (0.0, 0.0)) - 175.0).abs() < 1e-9);
        // The layer ends where it says: to 2 km only the first two segments.
        assert!((storm_relative_helicity(&w, 2000.0, (0.0, 0.0)) - 125.0).abs() < 1e-9);
        // A straight hodograph has none about a storm moving along it ...
        let line = profile(&[10.0, 1500.0, 3000.0], &[0.0, 10.0, 20.0], &[0.0, 0.0, 0.0]);
        assert!(storm_relative_helicity(&line, 3000.0, (5.0, 0.0)).abs() < 1e-9);
        // ... and a profile that stops short of the layer top claims nothing.
        assert!(storm_relative_helicity(&line, 6000.0, (5.0, 0.0)).is_nan());
    }

    #[test]
    fn layers_end_at_interpolated_heights() {
        let w = profile(&[10.0, 2010.0], &[0.0, 20.0], &[0.0, 0.0]);
        let (u, v) = wind_at(&w, 1010.0).unwrap();
        assert!((u - 10.0).abs() < 1e-9 && v.abs() < 1e-9);
        assert!(wind_at(&w, 5.0).is_none() && wind_at(&w, 3000.0).is_none());
        let pts = layer_points(&w, 1510.0, &[510.0]);
        let z: Vec<f64> = pts.iter().map(|p| p.0).collect();
        assert_eq!(z, vec![10.0, 510.0, 1510.0]);
        assert!((pts[1].1 - 5.0).abs() < 1e-9 && (pts[2].1 - 15.0).abs() < 1e-9);
        assert!((bulk_shear_kt(&w, 1010.0) - 10.0 * KNOTS).abs() < 1e-9);
    }

    #[test]
    fn a_southerly_wind_reads_180_degrees() {
        assert_eq!(direction_speed_text(0.0, 10.0), format!("180\u{b0} / {:.0} kt", 10.0 * KNOTS));
        assert_eq!(direction_speed_text(f64::NAN, 1.0), "n/a");
    }

    #[test]
    fn dewpoint_of_saturated_air_is_its_temperature() {
        // 20 C at 1000 hPa saturated: es = 23.37 hPa, w = 0.622 e/(p-e)
        let es = 6.112 * (17.67f64 * 20.0 / (20.0 + 243.5)).exp();
        let w = 0.621_97 * es / (1000.0 - es);
        assert!((dewpoint_c(100_000.0, w) - 20.0).abs() < 0.05);
    }
}
