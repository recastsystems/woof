//! Data-assimilation sheets, drawn through the production panel path.
//!
//! `rw_compare --mode MODE` reaches this module.  Every mode reads its
//! fields from the WRF-shaped NetCDF files themselves (a raw wrfout, a slim
//! composite history file, or a small file of pre-sliced planes on the
//! model grid), draws each panel through [`crate::panel::render_panel`] on
//! the files' own grid and projection, and composes the panels under one
//! header with [`crate::compare::compose_sheet_rows`].  Nothing here names
//! a case, a model run or a domain: a mode is a table row of fields and a
//! handful of arguments.
//!
//! ```text
//! rw_compare --mode increment --out-dir DIR --field qv,theta,qr+qs+qg,w
//!            [--height-agl 1000,3000,6000] [--heights-from WRFOUT]
//!            (--prior LIST --posterior LIST | --increment LIST)
//! rw_compare --mode spread --out-dir DIR --members LIST --field refc,qv
//!            [--height-agl M,...] [--columns mean,pmm,max,spread] [--spread-max V]
//! rw_compare --mode stamp --out-dir DIR --members LIST [--members LIST]...
//!            [--row-label TEXT]... [--field refc] [--stamps 6]
//!            [--thresholds 35,45] [--extras paintball,max,pmm,mean]
//! rw_compare --mode omb --out-dir DIR --obs FILE.json|FILE.csv
//!            (--grid WRFOUT | --background WRFOUT --field NAME)
//!            [--types a,b] [--all] [--quantity-label O-B] [--half-range V]
//! common:    [--name STEM] [--title TEXT] [--source-label TEXT]
//!            [--width N] [--height N] [--range LO:HI] [--half-range V]
//! ```
//!
//! A LIST is a comma list of files whose items may carry `*` and `?` in the
//! file name, expanded in natural order (`_2` before `_10`).
//!
//! **Fields.**  `--field` takes table names (qv, qc, qr, qi, qs, qg, qh,
//! theta, w, refc, t2, q2), any variable the files carry by its own name,
//! and sums of either (`qr+qs+qg`).  A 3-D field is drawn at a height above
//! ground (linear in height between the two mass levels around it; heights
//! from the file's own PH, PHB and HGT, or from `--heights-from`), a field
//! stored on one level or as a plane is drawn as it is, and `refc` is the
//! column maximum of `REFL_10CM`.
//!
//! **Scales.**  A field with a production colour table (reflectivity) wears
//! it.  Any other field panel wears the neutral generic ramp over the 1st to
//! 99th percentile of the row's field panels together, so the panels of one
//! row are read on one bar; `--range LO:HI` fixes it.  An increment or an
//! O-B panel wears the run-difference ladder of `rustwx_render::difference`
//! (`rw_wrfbatch --diff-against`): zero-centred, its half range from the
//! difference table or the table's percentile rule, stated on the panel.
//! Spread wears the ensemble spread ladder from zero.
//!
//! **Refusals**, each by name: a 3-D field with no height and more than one
//! level; a height asked of a plane; a file with no heights and no
//! `--heights-from`; files on two grids; a horizontally staggered field;
//! terms of one sum in two units; an O-B type list naming no observation.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use rayon::prelude::*;
use rustwx_core::{CanonicalField, FieldSelector, GridProjection, ModelId};
use rustwx_ensemble::{
    MemberStack, NanPolicy, PmmTieRule, ensemble_max, ensemble_mean, ensemble_spread,
    member_color, probability_matched_mean,
};
use rustwx_products::geographic_overlays::{ValuePointLayer, ValuePointSpec};
use rustwx_products::viewer::StoreVariableStyle;
use rustwx_render::difference::{DifferenceTable, difference_scale, half_range, nice_step};
use rustwx_render::{
    ColorScale, ContourLayer, LegendControls, LegendMode, LevelDensity, PngCompressionMode,
    PngWriteOptions, RenderDensity, RgbaImage,
};
use wrf_core::WrfFile;

use crate::annotate::{MapOverlays, PointSpec, parse_color};
use crate::compare::{SheetHeader, compose_sheet_rows};
use crate::panel::{PanelRequest, render_panel, safe_component};

pub const USAGE: &str = "usage: rw_compare --mode increment --out-dir DIR --field LIST \
[--height-agl M,...] [--heights-from WRFOUT] (--prior LIST --posterior LIST | --increment LIST)\n       \
rw_compare --mode spread --out-dir DIR --members LIST --field LIST [--height-agl M,...] \
[--columns mean,pmm,max,spread] [--spread-max V]\n       \
rw_compare --mode stamp --out-dir DIR --members LIST [--members LIST]... [--row-label TEXT]... \
[--field refc] [--stamps N] [--thresholds 35,45] [--extras paintball,max,pmm,mean]\n       \
rw_compare --mode omb --out-dir DIR --obs FILE.json|FILE.csv (--grid WRFOUT | --background WRFOUT --field NAME) \
[--types a,b] [--all] [--quantity-label TEXT] [--units TEXT] [--valid TEXT]\n       \
common: [--name STEM] [--title TEXT] [--source-label TEXT] [--width N] [--height N] \
[--range LO:HI] [--half-range V] [--dot-radius PX]";

// ---------------------------------------------------------------------------
// File lists
// ---------------------------------------------------------------------------

/// A comma list of files, each item optionally a wildcard (`*`, `?`) in its
/// file name, expanded in natural order (`_2` before `_10`).  A pattern that
/// matches nothing, or an item that is not a file, is refused by name: a
/// reduction quietly taken over fewer members than the caller meant is the
/// wrong picture.
pub fn expand_file_list(list: &str) -> Result<Vec<PathBuf>, String> {
    let mut files = Vec::new();
    for item in list.split(',').map(str::trim).filter(|item| !item.is_empty()) {
        let path = PathBuf::from(item);
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or_default()
            .to_string();
        if name.contains(['*', '?']) {
            let parent = path
                .parent()
                .filter(|parent| !parent.as_os_str().is_empty())
                .unwrap_or(Path::new("."));
            let mut matched: Vec<PathBuf> = std::fs::read_dir(parent)
                .map_err(|error| format!("list {}: {error}", parent.display()))?
                .filter_map(|entry| entry.ok())
                .filter(|entry| {
                    entry
                        .file_name()
                        .to_str()
                        .is_some_and(|candidate| wildcard_match(&name, candidate))
                })
                .map(|entry| entry.path())
                .filter(|path| path.is_file())
                .collect();
            if matched.is_empty() {
                return Err(format!("{item:?} matches no file"));
            }
            matched.sort_by(|a, b| natural_order(a, b));
            files.extend(matched);
        } else {
            if !path.is_file() {
                return Err(format!("{item:?} is not a file"));
            }
            files.push(path);
        }
    }
    if files.is_empty() {
        return Err(format!("{list:?} names no file"));
    }
    Ok(files)
}

/// `*` any run, `?` one character, everything else literal.
pub fn wildcard_match(pattern: &str, text: &str) -> bool {
    let pattern: Vec<char> = pattern.chars().collect();
    let text: Vec<char> = text.chars().collect();
    let (mut p, mut t) = (0usize, 0usize);
    let (mut star, mut mark) = (None, 0usize);
    while t < text.len() {
        if p < pattern.len() && (pattern[p] == '?' || pattern[p] == text[t]) {
            p += 1;
            t += 1;
        } else if p < pattern.len() && pattern[p] == '*' {
            star = Some(p);
            mark = t;
            p += 1;
        } else if let Some(position) = star {
            p = position + 1;
            mark += 1;
            t = mark;
        } else {
            return false;
        }
    }
    while p < pattern.len() && pattern[p] == '*' {
        p += 1;
    }
    p == pattern.len()
}

/// Digit runs compare as numbers, everything else as text.
pub fn natural_order(a: &Path, b: &Path) -> std::cmp::Ordering {
    fn chunks(text: &str) -> Vec<(bool, String)> {
        let mut out: Vec<(bool, String)> = Vec::new();
        for character in text.chars() {
            let digit = character.is_ascii_digit();
            match out.last_mut() {
                Some((was_digit, chunk)) if *was_digit == digit => chunk.push(character),
                _ => out.push((digit, character.to_string())),
            }
        }
        out
    }
    let a = chunks(&a.to_string_lossy());
    let b = chunks(&b.to_string_lossy());
    for (x, y) in a.iter().zip(&b) {
        let order = match (x.0, y.0) {
            (true, true) => {
                let (u, v) = (x.1.trim_start_matches('0'), y.1.trim_start_matches('0'));
                u.len()
                    .cmp(&v.len())
                    .then_with(|| u.cmp(v))
                    .then_with(|| x.1.cmp(&y.1))
            }
            _ => x.1.cmp(&y.1),
        };
        if order != std::cmp::Ordering::Equal {
            return order;
        }
    }
    a.len().cmp(&b.len())
}

/// A member's NUMBER: the digits that end its file stem (`..._17.nc` is
/// member 17), or its position in the list when the stem ends otherwise.
/// Paintball colours key on the number, so member 7 wears one colour in
/// every sheet whatever else is drawn beside it.
pub fn member_number(path: &Path, position: usize) -> u32 {
    let stem = path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or_default();
    let digits: String = stem
        .chars()
        .rev()
        .take_while(|character| character.is_ascii_digit())
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    digits.parse().unwrap_or(position as u32)
}

// ---------------------------------------------------------------------------
// Fields
// ---------------------------------------------------------------------------

/// One named field: the variable, how it becomes the drawn quantity, and
/// what the panel calls it.
struct FieldRow {
    name: &'static str,
    var: &'static str,
    factor: f64,
    offset: f64,
    units: &'static str,
    title: &'static str,
    /// The noun a sum's title is built from (`rain + snow + graupel`).
    noun: Option<&'static str>,
    column_max: bool,
    production: Option<fn() -> Option<StoreVariableStyle>>,
}

fn reflectivity_style() -> Option<StoreVariableStyle> {
    let selector = FieldSelector::entire_atmosphere(CanonicalField::CompositeReflectivity);
    rustwx_products::viewer::operational_style_for_store_variable(
        "refl",
        &serde_json::to_value(selector).ok()?,
        "dBZ",
        ModelId::WrfGdex,
    )
}

fn field_rows() -> Vec<FieldRow> {
    let ratio = |name, var, title, noun| FieldRow {
        name,
        var,
        factor: 1_000.0,
        offset: 0.0,
        units: "g/kg",
        title,
        noun: Some(noun),
        column_max: false,
        production: None,
    };
    vec![
        ratio("qv", "QVAPOR", "Water vapour mixing ratio", "water vapour"),
        ratio("qc", "QCLOUD", "Cloud water mixing ratio", "cloud water"),
        ratio("qr", "QRAIN", "Rain mixing ratio", "rain"),
        ratio("qi", "QICE", "Cloud ice mixing ratio", "cloud ice"),
        ratio("qs", "QSNOW", "Snow mixing ratio", "snow"),
        ratio("qg", "QGRAUP", "Graupel mixing ratio", "graupel"),
        ratio("qh", "QHAIL", "Hail mixing ratio", "hail"),
        ratio("q2", "Q2", "2 m water vapour mixing ratio", "2 m water vapour"),
        FieldRow {
            name: "theta",
            var: "T",
            factor: 1.0,
            offset: 300.0,
            units: "K",
            title: "Potential temperature",
            noun: None,
            column_max: false,
            production: None,
        },
        FieldRow {
            name: "w",
            var: "W",
            factor: 1.0,
            offset: 0.0,
            units: "m/s",
            title: "Vertical velocity",
            noun: None,
            column_max: false,
            production: None,
        },
        FieldRow {
            name: "t2",
            var: "T2",
            factor: 1.0,
            offset: 0.0,
            units: "K",
            title: "2 m temperature",
            noun: None,
            column_max: false,
            production: None,
        },
        FieldRow {
            name: "refc",
            var: "REFL_10CM",
            factor: 1.0,
            offset: 0.0,
            units: "dBZ",
            title: "Composite reflectivity",
            noun: None,
            column_max: true,
            production: Some(reflectivity_style),
        },
    ]
}

/// One term of a field expression, resolved.
#[derive(Clone, Debug)]
struct Term {
    var: String,
    factor: f64,
    offset: f64,
    column_max: bool,
}

/// A `--field` expression, resolved against a probe file.
#[derive(Clone)]
pub struct FieldExpr {
    pub key: String,
    pub title: String,
    pub units: String,
    terms: Vec<Term>,
    production: Option<StoreVariableStyle>,
}

fn variable_units(path: &Path, var: &str) -> Option<String> {
    let file = netcrust::open(path).ok()?;
    let variable = file.variable(var)?;
    variable
        .attributes()
        .iter()
        .find(|attribute| attribute.name() == "units")
        .and_then(|attribute| attribute.value().as_string().map(|units| units.trim().to_string()))
}

/// Resolve `qr+qs+qg`, `refc`, `QVAPOR` ... against the probe file.
pub fn resolve_field(expression: &str, probe: &WrfFile) -> Result<FieldExpr, String> {
    let rows = field_rows();
    let mut terms = Vec::new();
    let mut units: Option<String> = None;
    let mut titles = Vec::new();
    let mut nouns = Vec::new();
    let mut production = None;
    let names: Vec<&str> = expression
        .split('+')
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .collect();
    if names.is_empty() {
        return Err(format!("--field {expression:?} names nothing"));
    }
    for name in &names {
        let (term, term_units, title, noun) = match rows.iter().find(|row| row.name == *name) {
            Some(row) => {
                if !probe.has_var(row.var) {
                    return Err(format!(
                        "{}: field {name} reads {} and the file has no such variable",
                        probe.path.display(),
                        row.var
                    ));
                }
                if names.len() == 1 {
                    production = row.production.and_then(|style| style());
                }
                (
                    Term {
                        var: row.var.to_string(),
                        factor: row.factor,
                        offset: row.offset,
                        column_max: row.column_max,
                    },
                    row.units.to_string(),
                    row.title.to_string(),
                    row.noun,
                )
            }
            None if probe.has_var(name) => (
                Term {
                    var: (*name).to_string(),
                    factor: 1.0,
                    offset: 0.0,
                    column_max: false,
                },
                variable_units(&probe.path, name).unwrap_or_default(),
                (*name).to_string(),
                None,
            ),
            None => {
                return Err(format!(
                    "--field {name:?} is neither a table field ({}) nor a variable of {}",
                    rows.iter().map(|row| row.name).collect::<Vec<_>>().join(", "),
                    probe.path.display()
                ));
            }
        };
        match &units {
            None => units = Some(term_units),
            Some(have) if *have != term_units => {
                return Err(format!(
                    "--field {expression:?} adds {have} to {term_units}; a sum is taken in one unit"
                ));
            }
            Some(_) => {}
        }
        terms.push(term);
        titles.push(title);
        nouns.push(noun);
    }
    let title = if titles.len() == 1 {
        titles.remove(0)
    } else if nouns.iter().all(Option::is_some) {
        let joined = nouns.iter().flatten().copied().collect::<Vec<_>>().join(" + ");
        let mut characters = joined.chars();
        let first = characters.next().map(|c| c.to_uppercase().collect::<String>()).unwrap_or_default();
        format!("{first}{} mixing ratio", characters.as_str())
    } else {
        titles.join(" + ")
    };
    Ok(FieldExpr {
        key: names.join("+"),
        title,
        units: units.unwrap_or_default(),
        terms,
        production,
    })
}

/// Linear interpolation of a `[nz, ny, nx]` column set to one height,
/// in the units the heights are in.  NaN below the lowest level, above the
/// highest, and wherever a bounding value is not finite.
pub fn interpolate_to_height(
    values: &[f64],
    heights: &[f64],
    nz: usize,
    points: usize,
    target: f64,
) -> Vec<f64> {
    (0..points)
        .map(|point| {
            for level in 0..nz.saturating_sub(1) {
                let (z0, z1) = (heights[level * points + point], heights[(level + 1) * points + point]);
                if z0 <= target && target <= z1 && z1 > z0 {
                    let (v0, v1) = (values[level * points + point], values[(level + 1) * points + point]);
                    if !v0.is_finite() || !v1.is_finite() {
                        return f64::NAN;
                    }
                    let weight = (target - z0) / (z1 - z0);
                    return v0 + weight * (v1 - v0);
                }
            }
            f64::NAN
        })
        .collect()
}

fn destagger_z(values: &[f64], nz_stag: usize, points: usize) -> Vec<f64> {
    (0..nz_stag - 1)
        .flat_map(|level| {
            (0..points).map(move |point| {
                0.5 * (values[level * points + point] + values[(level + 1) * points + point])
            })
        })
        .collect()
}

/// Heights above ground of a file's mass levels, `[nz, ny, nx]`, when the
/// file carries PH, PHB and HGT.
fn own_heights(file: &WrfFile) -> Option<Result<Arc<[f64]>, String>> {
    if !(file.has_var("PH") && file.has_var("PHB") && file.has_var("HGT")) {
        return None;
    }
    Some(
        file.height_agl(0)
            .map_err(|error| format!("{}: heights above ground: {error}", file.path.display())),
    )
}

fn open(path: &Path) -> Result<WrfFile, String> {
    WrfFile::open(path).map_err(|error| format!("open {}: {error}", path.display()))
}

/// Which slice of a field a row draws.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Level {
    /// A plane, a one-level field, or a column maximum.
    AsStored,
    HeightAgl(f64),
}

impl Level {
    fn text(self, expr: &FieldExpr) -> String {
        match self {
            Level::HeightAgl(metres) => format!("{metres:.0} m AGL"),
            Level::AsStored if expr.terms.iter().any(|term| term.column_max) => {
                "column maximum".to_string()
            }
            Level::AsStored => "as stored".to_string(),
        }
    }

    /// How a panel's subtitle names the slice after the field's title:
    /// `... at 3000 m AGL`, `..., column maximum`, or nothing for a plane.
    fn phrase(self, expr: &FieldExpr) -> String {
        match self {
            Level::HeightAgl(metres) => format!(" at {metres:.0} m AGL"),
            Level::AsStored if expr.terms.iter().any(|term| term.column_max) => {
                ", column maximum".to_string()
            }
            Level::AsStored => String::new(),
        }
    }

    fn token(self) -> String {
        match self {
            Level::HeightAgl(metres) => format!("{metres:.0}m"),
            Level::AsStored => "plane".to_string(),
        }
    }
}

/// One file's plane of a field at a level, in drawn units.  `offsets`
/// false reads an increment file, where a state offset (theta's 300 K) has
/// already cancelled.
pub fn read_plane(
    path: &Path,
    expr: &FieldExpr,
    level: Level,
    fallback_heights: Option<&[f64]>,
    offsets: bool,
    grid: (usize, usize),
) -> Result<Vec<f64>, String> {
    let file = open(path)?;
    let (ny, nx) = grid;
    if file.ny != ny || file.nx != nx {
        return Err(format!(
            "{} is {}x{} and the sheet's grid is {ny}x{nx}",
            path.display(),
            file.ny,
            file.nx
        ));
    }
    let points = ny * nx;
    let mut heights: Option<Arc<[f64]>> = None;
    let mut total: Option<Vec<f64>> = None;
    for term in &expr.terms {
        let shape = file
            .var_shape_no_time(&term.var)
            .map_err(|error| format!("{}: {}: {error}", path.display(), term.var))?;
        let data = file
            .read_var(&term.var, 0)
            .map_err(|error| format!("{}: read {}: {error}", path.display(), term.var))?;
        let plane = match shape.as_slice() {
            [y, x] if *y == ny && *x == nx => {
                if let Level::HeightAgl(metres) = level {
                    return Err(format!(
                        "{}: {} is a plane; it has no {metres:.0} m above ground to draw (drop \
                         --height-agl for it)",
                        path.display(),
                        term.var
                    ));
                }
                data
            }
            [levels, y, x] if *y == ny && *x == nx => {
                let levels = *levels;
                if term.column_max {
                    (0..points)
                        .map(|point| {
                            (0..levels)
                                .map(|level| data[level * points + point])
                                .filter(|value| value.is_finite())
                                .fold(f64::NAN, |best, value| {
                                    if best.is_nan() || value > best { value } else { best }
                                })
                        })
                        .collect()
                } else {
                    match level {
                        Level::AsStored if levels == 1 => data,
                        Level::AsStored => {
                            return Err(format!(
                                "{}: {} has {levels} levels; name a height with --height-agl",
                                path.display(),
                                term.var
                            ));
                        }
                        Level::HeightAgl(metres) => {
                            let column: Arc<[f64]> = match &heights {
                                Some(have) => have.clone(),
                                None => {
                                    let found: Arc<[f64]> = match own_heights(&file) {
                                        Some(result) => result?,
                                        None => Arc::from(fallback_heights.ok_or_else(|| {
                                            format!(
                                                "{} carries no PH, PHB and HGT, so the height \
                                                 above ground of its levels is unknown; pass \
                                                 --heights-from WRFOUT",
                                                path.display()
                                            )
                                        })?),
                                    };
                                    heights = Some(found.clone());
                                    found
                                }
                            };
                            let mass_levels = column.len() / points;
                            let values = if levels == mass_levels {
                                data
                            } else if levels == mass_levels + 1 {
                                destagger_z(&data, levels, points)
                            } else {
                                return Err(format!(
                                    "{}: {} has {levels} levels and the heights {mass_levels}",
                                    path.display(),
                                    term.var
                                ));
                            };
                            interpolate_to_height(&values, &column, mass_levels, points, metres)
                        }
                    }
                }
            }
            other => {
                return Err(format!(
                    "{}: {} has shape {other:?}; only planes and columns on the mass grid \
                     ({ny}x{nx}) are drawn here (a horizontally staggered field is not)",
                    path.display(),
                    term.var
                ));
            }
        };
        let offset = if offsets { term.offset } else { 0.0 };
        let plane: Vec<f64> = plane.iter().map(|value| value * term.factor + offset).collect();
        total = Some(match total {
            None => plane,
            Some(sum) => sum.iter().zip(&plane).map(|(a, b)| a + b).collect(),
        });
    }
    total.ok_or_else(|| "a field with no terms".to_string())
}

/// Every file's plane of one field at one level, read in parallel.
fn read_planes(
    files: &[PathBuf],
    expr: &FieldExpr,
    level: Level,
    fallback_heights: Option<&[f64]>,
    offsets: bool,
    grid: (usize, usize),
) -> Result<Vec<Vec<f64>>, String> {
    files
        .par_iter()
        .map(|file| read_plane(file, expr, level, fallback_heights, offsets, grid))
        .collect()
}

/// The rows a field draws: one per height, or one as stored when the field
/// is a plane, one level, or a column maximum.
fn field_levels(expr: &FieldExpr, probe: &WrfFile, heights: &[f64]) -> Result<Vec<Level>, String> {
    let layered = expr.terms.iter().any(|term| {
        !term.column_max
            && probe
                .var_shape_no_time(&term.var)
                .map(|shape| shape.len() == 3 && shape[0] > 1)
                .unwrap_or(false)
    });
    if !layered {
        return Ok(vec![Level::AsStored]);
    }
    if heights.is_empty() {
        return Err(format!(
            "{} is a 3-D field; name the heights to draw with --height-agl",
            expr.key
        ));
    }
    Ok(heights.iter().map(|metres| Level::HeightAgl(*metres)).collect())
}

// ---------------------------------------------------------------------------
// The grid every panel is drawn on
// ---------------------------------------------------------------------------

/// The sheet's grid, projection and time stamps, from its first file.
pub struct SheetGrid {
    pub lat: Vec<f32>,
    pub lon: Vec<f32>,
    pub projection: Option<GridProjection>,
    pub ny: usize,
    pub nx: usize,
    pub valid: String,
    pub init: Option<String>,
}

fn time_label(stamp: &str) -> String {
    // `2026-10-01_19:00:00` -> `2026-10-01 19Z`, `19:30Z` off the hour.
    let stamp = stamp.trim();
    let (day, clock) = stamp.split_once(['_', 'T', ' ']).unwrap_or((stamp, ""));
    let parts: Vec<&str> = clock.split(':').collect();
    match parts.as_slice() {
        [hour, "00", ..] => format!("{day} {hour}Z"),
        [hour, minute, ..] => format!("{day} {hour}:{minute}Z"),
        _ => stamp.to_string(),
    }
}

pub fn sheet_grid(path: &Path) -> Result<SheetGrid, String> {
    let file = open(path)?;
    let read = |name: &str| -> Result<Vec<f32>, String> {
        file.read_var(name, 0)
            .map(|values| values.iter().map(|value| *value as f32).collect())
            .map_err(|error| format!("{}: read {name}: {error}", path.display()))
    };
    let lat = read("XLAT")?;
    let lon = read("XLONG")?;
    let valid = file
        .times()
        .ok()
        .and_then(|times| times.first().cloned())
        .map(|stamp| time_label(&stamp))
        .unwrap_or_else(|| "unknown".to_string());
    let init = ["SIMULATION_START_DATE", "START_DATE"]
        .iter()
        .find_map(|name| file.global_attr_str(name).ok())
        .map(|stamp| time_label(&stamp));
    Ok(SheetGrid {
        projection: crate::wrf_process::wrf_projection(&file),
        ny: file.ny,
        nx: file.nx,
        lat,
        lon,
        valid,
        init,
    })
}

fn check_same_grid(grid: &SheetGrid, path: &Path) -> Result<(), String> {
    let other = sheet_grid(path)?;
    let same = other.lat.len() == grid.lat.len()
        && other.lat.iter().zip(&grid.lat).all(|(a, b)| (a - b).abs() <= 1.0e-4)
        && other.lon.iter().zip(&grid.lon).all(|(a, b)| (a - b).abs() <= 1.0e-4);
    if same {
        Ok(())
    } else {
        Err(format!(
            "{} is not on the sheet's grid; every panel of a sheet shares one grid and nothing \
             is regridded here",
            path.display()
        ))
    }
}

// ---------------------------------------------------------------------------
// Styles
// ---------------------------------------------------------------------------

fn stepped() -> LegendControls {
    LegendControls {
        density: LevelDensity::default(),
        mode: LegendMode::Stepped,
    }
}

/// How one row's field panels become colour.
struct FieldStyle {
    scale: ColorScale,
    tick: Option<f64>,
    legend: LegendControls,
    density: RenderDensity,
    units: String,
    production: Option<StoreVariableStyle>,
    rule: String,
}

impl FieldStyle {
    fn convert(&self, value: f64) -> f32 {
        match &self.production {
            Some(style) => style.convert.apply(value as f32),
            None => value as f32,
        }
    }
}

fn percentile(values: &mut [f64], fraction: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let index = ((fraction * values.len() as f64).ceil() as usize).clamp(1, values.len()) - 1;
    let (_, value, _) = values.select_nth_unstable_by(index, |a, b| a.total_cmp(b));
    Some(*value)
}

fn field_style(expr: &FieldExpr, planes: &[&[f64]], range: Option<(f64, f64)>) -> FieldStyle {
    if let Some(style) = &expr.production {
        return FieldStyle {
            scale: style.scale.clone(),
            tick: style.cbar_tick_step,
            legend: style.colormap_options.legend,
            density: style.colormap_options.render_density,
            units: style.display_units.clone(),
            production: Some(style.clone()),
            rule: "production colour table".to_string(),
        };
    }
    let (lo, hi, rule) = match range {
        Some((lo, hi)) => (lo, hi, format!("range {lo} to {hi} (--range)")),
        None => {
            let mut finite: Vec<f64> = planes
                .iter()
                .flat_map(|plane| plane.iter().copied())
                .filter(|value| value.is_finite())
                .collect();
            let lo = percentile(&mut finite, 0.01).unwrap_or(0.0);
            let hi = percentile(&mut finite, 0.99).unwrap_or(1.0);
            (lo, hi, format!("range {} to {} (p1 to p99 of the row)", short(lo), short(hi)))
        }
    };
    let style = rustwx_products::viewer::generic_style_for_prescaled_store_variable(
        &safe_component(&expr.key, "field"),
        &expr.units,
        Some((lo as f32, hi as f32)),
    );
    FieldStyle {
        scale: style.scale.clone(),
        tick: style.cbar_tick_step,
        legend: style.colormap_options.legend,
        density: style.colormap_options.render_density,
        units: expr.units.clone(),
        production: None,
        rule,
    }
}

fn short(value: f64) -> String {
    if value == 0.0 || !value.is_finite() {
        return format!("{value}");
    }
    let magnitude = value.abs().log10().floor() as i32;
    let decimals = (2 - magnitude).clamp(0, 6) as usize;
    format!("{value:.decimals$}")
}

/// The zero-centred difference ladder of `rw_wrfbatch --diff-against`.
fn diverging(key: &str, units: &str, values: &[f32], fixed: Option<f64>) -> (ColorScale, f64, String) {
    let table = DifferenceTable::builtin();
    let (half, rule) = match fixed {
        Some(half) => (half, "--half-range".to_string()),
        None => {
            let (half, rule) = half_range(table, key, units, values);
            (half, rule.describe())
        }
    };
    (ColorScale::Discrete(difference_scale(table, half)), half, rule)
}

/// The spread ladder's top: the given value, or the 99th percentile of the
/// spread rounded up so that each of its nine bands is a 1-2-5 step.
fn spread_top(values: &[f64], fixed: Option<f64>) -> (f64, String) {
    if let Some(top) = fixed {
        return (top, "--spread-max".to_string());
    }
    let mut finite: Vec<f64> = values.iter().copied().filter(|value| value.is_finite()).collect();
    match percentile(&mut finite, 0.99) {
        Some(p99) if p99 > 0.0 => (nice_step(p99 / 9.0) * 9.0, format!("p99={}", short(p99))),
        _ => (1.0, "no spread".to_string()),
    }
}

#[derive(Default)]
struct PlaneStats {
    points: usize,
    mean: f64,
    rms: f64,
    min: f64,
    max: f64,
}

fn plane_stats(values: &[f64]) -> PlaneStats {
    let finite: Vec<f64> = values.iter().copied().filter(|value| value.is_finite()).collect();
    if finite.is_empty() {
        return PlaneStats::default();
    }
    let n = finite.len() as f64;
    PlaneStats {
        points: finite.len(),
        mean: finite.iter().sum::<f64>() / n,
        rms: (finite.iter().map(|value| value * value).sum::<f64>() / n).sqrt(),
        min: finite.iter().copied().fold(f64::INFINITY, f64::min),
        max: finite.iter().copied().fold(f64::NEG_INFINITY, f64::max),
    }
}

fn stats_json(stats: &PlaneStats) -> serde_json::Value {
    serde_json::json!({"points": stats.points, "mean": stats.mean, "rms": stats.rms,
        "min": stats.min, "max": stats.max})
}

// ---------------------------------------------------------------------------
// Drawing
// ---------------------------------------------------------------------------

struct Canvas<'a> {
    grid: &'a SheetGrid,
    width: u32,
    height: u32,
    scratch: &'a Path,
    right: &'a str,
}

struct Panel {
    slug: String,
    title: String,
    units: String,
    values: Vec<f32>,
    scale: ColorScale,
    tick: Option<f64>,
    legend: LegendControls,
    density: RenderDensity,
    left: String,
    contours: Vec<ContourLayer>,
    colorbar: bool,
    overlays: Option<MapOverlays>,
}

fn draw(canvas: &Canvas<'_>, index: usize, panel: Panel) -> Result<RgbaImage, String> {
    let grid = canvas.grid;
    let path = render_panel(PanelRequest {
        lat_deg: &grid.lat,
        lon_deg: &grid.lon,
        projection: grid.projection.as_ref(),
        ny: grid.ny,
        nx: grid.nx,
        values: panel.values,
        product_slug: panel.slug,
        title: panel.title,
        display_units: panel.units,
        scale: panel.scale,
        cbar_tick_step: panel.tick,
        legend: panel.legend,
        render_density: panel.density,
        subtitle_left: panel.left,
        subtitle_center: None,
        subtitle_right: canvas.right.to_string(),
        width: canvas.width,
        height: canvas.height,
        contours: panel.contours,
        colorbar: panel.colorbar,
        overlays: panel.overlays.as_ref(),
        annotations: None,
        out_path: canvas.scratch.join(format!("panel{index:03}.png")),
    })?;
    Ok(image::open(&path)
        .map_err(|error| format!("read back {}: {error}", path.display()))?
        .to_rgba8())
}

fn to_f32(values: &[f64], style: &FieldStyle) -> Vec<f32> {
    values.iter().map(|value| style.convert(*value)).collect()
}

fn field_panel(slug: String, title: String, left: String, values: &[f64], style: &FieldStyle) -> Panel {
    Panel {
        slug,
        title,
        units: style.units.clone(),
        values: to_f32(values, style),
        scale: style.scale.clone(),
        tick: style.tick,
        legend: style.legend,
        density: style.density,
        left,
        contours: Vec::new(),
        colorbar: true,
        overlays: None,
    }
}

fn diverging_panel(
    slug: String,
    title: String,
    left: String,
    values: Vec<f32>,
    units: &str,
    scale: ColorScale,
    half: f64,
) -> Panel {
    let bands = DifferenceTable::builtin().bands.max(2) as f64;
    Panel {
        slug,
        title,
        units: units.to_string(),
        values,
        scale,
        tick: Some(2.0 * half / bands),
        legend: stepped(),
        density: RenderDensity::default(),
        left,
        contours: Vec::new(),
        colorbar: true,
        overlays: None,
    }
}

/// A colour key for a scale the panels do not carry a bar for (O-B dots on
/// a shaded background), appended under the sheet without touching a map
/// pixel.  The drawing is `station_overlay::append_error_key`'s.
fn append_scale_key(sheet: &RgbaImage, label: &str, scale: &ColorScale) -> Result<RgbaImage, String> {
    let discrete = scale.resolved_discrete();
    let options = rustwx_render::ColormapBuildOptions {
        render_density: rustwx_render::StaticPlotStyle::from_env()
            .render_density(RenderDensity::default()),
        legend: stepped(),
    };
    let cmap = rustwx_render::build_colormap(scale, options);
    let background = *sheet.get_pixel(0, 0);
    let mut result = RgbaImage::from_pixel(sheet.width(), sheet.height() + 78, background);
    image::imageops::overlay(&mut result, sheet, 0, 0);
    let luminance = 0.2126 * f64::from(background[0])
        + 0.7152 * f64::from(background[1])
        + 0.0722 * f64::from(background[2]);
    let ink = if luminance >= 128.0 { rustwx_render::Rgba::BLACK } else { rustwx_render::Rgba::WHITE };
    let top = sheet.height() + 10;
    rustwx_render::draw_text(&mut result, label, 18, top as i32, ink, 1);
    let left = 18_u32;
    let usable = sheet.width().saturating_sub(36);
    let bins = discrete.colors.len() as u32;
    for bin in 0..bins {
        let x0 = left + usable * bin / bins;
        let x1 = left + usable * (bin + 1) / bins;
        let value = (discrete.levels[bin as usize] + discrete.levels[bin as usize + 1]) / 2.0;
        let color = cmap.map(value).to_image_rgba();
        for y in top + 22..top + 39 {
            for x in x0..x1.saturating_sub(1) {
                result.put_pixel(x, y, color);
            }
        }
        let text = if bin == 0 {
            format!("< {}", short(discrete.levels[1]))
        } else if bin + 1 == bins {
            format!(">= {}", short(discrete.levels[bin as usize]))
        } else {
            short(value)
        };
        rustwx_render::draw_text(&mut result, &text, x0 as i32, (top + 45) as i32, ink, 1);
    }
    Ok(result)
}

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Increment,
    Spread,
    Stamp,
    Omb,
}

impl Mode {
    fn parse(text: &str) -> Result<Self, String> {
        match text {
            "increment" => Ok(Self::Increment),
            "spread" => Ok(Self::Spread),
            "stamp" => Ok(Self::Stamp),
            "omb" => Ok(Self::Omb),
            other => Err(format!("--mode {other:?} is not compare, increment, spread, stamp or omb")),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Increment => "increment",
            Self::Spread => "spread",
            Self::Stamp => "stamp",
            Self::Omb => "omb",
        }
    }
}

struct Options {
    mode: Mode,
    out_dir: PathBuf,
    name: Option<String>,
    title: Option<String>,
    source_label: String,
    width: u32,
    height: u32,
    fields: Vec<String>,
    heights: Vec<f64>,
    heights_from: Option<PathBuf>,
    prior: Vec<PathBuf>,
    posterior: Vec<PathBuf>,
    increment: Vec<PathBuf>,
    members: Vec<Vec<PathBuf>>,
    row_labels: Vec<String>,
    stamps: usize,
    thresholds: Vec<f64>,
    extras: Vec<String>,
    columns: Vec<String>,
    spread_max: Option<f64>,
    half_range: Option<f64>,
    range: Option<(f64, f64)>,
    obs: Option<PathBuf>,
    background: Option<PathBuf>,
    grid: Option<PathBuf>,
    types: Vec<String>,
    all_panel: bool,
    quantity_label: String,
    units: Option<String>,
    valid: Option<String>,
    dot_radius: u32,
}

fn numbers(text: &str, flag: &str) -> Result<Vec<f64>, String> {
    text.split(',')
        .map(str::trim)
        .filter(|token| !token.is_empty())
        .map(|token| {
            token
                .parse::<f64>()
                .ok()
                .filter(|value| value.is_finite())
                .ok_or_else(|| format!("{flag} {token:?} is not a number"))
        })
        .collect()
}

fn words(text: &str) -> Vec<String> {
    text.split(',')
        .map(str::trim)
        .filter(|token| !token.is_empty())
        .map(str::to_string)
        .collect()
}

fn parse_options(args: &[String]) -> Result<Option<Options>, String> {
    let mut mode = None;
    let mut out_dir = None;
    let mut options = Options {
        mode: Mode::Increment,
        out_dir: PathBuf::new(),
        name: None,
        title: None,
        source_label: "WOOF".to_string(),
        width: 0,
        height: 0,
        fields: Vec::new(),
        heights: Vec::new(),
        heights_from: None,
        prior: Vec::new(),
        posterior: Vec::new(),
        increment: Vec::new(),
        members: Vec::new(),
        row_labels: Vec::new(),
        stamps: 6,
        thresholds: Vec::new(),
        extras: vec!["paintball".into(), "max".into(), "pmm".into()],
        columns: vec!["mean".into(), "spread".into()],
        spread_max: None,
        half_range: None,
        range: None,
        obs: None,
        background: None,
        grid: None,
        types: Vec::new(),
        all_panel: false,
        quantity_label: "O-B".to_string(),
        units: None,
        valid: None,
        dot_radius: 4,
    };
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        let mut value = |flag: &str| {
            iter.next()
                .cloned()
                .ok_or_else(|| format!("{flag} needs a value"))
        };
        match arg.as_str() {
            "--help" | "-h" => return Ok(None),
            "--mode" => mode = Some(Mode::parse(&value("--mode")?)?),
            "--out-dir" => out_dir = Some(PathBuf::from(value("--out-dir")?)),
            "--name" => options.name = Some(value("--name")?),
            "--title" => options.title = Some(value("--title")?),
            "--source-label" => options.source_label = value("--source-label")?,
            "--width" => {
                options.width = value("--width")?
                    .parse()
                    .map_err(|_| "--width needs a whole number".to_string())?
            }
            "--height" => {
                options.height = value("--height")?
                    .parse()
                    .map_err(|_| "--height needs a whole number".to_string())?
            }
            "--field" => options.fields.extend(words(&value("--field")?)),
            "--height-agl" => options.heights.extend(numbers(&value("--height-agl")?, "--height-agl")?),
            "--heights-from" => options.heights_from = Some(PathBuf::from(value("--heights-from")?)),
            "--prior" => options.prior.extend(expand_file_list(&value("--prior")?)?),
            "--posterior" => options.posterior.extend(expand_file_list(&value("--posterior")?)?),
            "--increment" => options.increment.extend(expand_file_list(&value("--increment")?)?),
            "--members" => options.members.push(expand_file_list(&value("--members")?)?),
            "--row-label" => options.row_labels.push(value("--row-label")?),
            "--stamps" => {
                options.stamps = value("--stamps")?
                    .parse()
                    .map_err(|_| "--stamps needs a whole number".to_string())?
            }
            "--thresholds" => options.thresholds = numbers(&value("--thresholds")?, "--thresholds")?,
            "--extras" => options.extras = words(&value("--extras")?),
            "--columns" => options.columns = words(&value("--columns")?),
            "--spread-max" => options.spread_max = numbers(&value("--spread-max")?, "--spread-max")?.first().copied(),
            "--half-range" => options.half_range = numbers(&value("--half-range")?, "--half-range")?.first().copied(),
            "--range" => {
                let text = value("--range")?;
                let (lo, hi) = text
                    .split_once(':')
                    .ok_or_else(|| format!("--range {text:?} is not LO:HI"))?;
                let parse = |token: &str| token.trim().parse::<f64>().map_err(|_| format!("--range {text:?} is not LO:HI"));
                let (lo, hi) = (parse(lo)?, parse(hi)?);
                if !(lo < hi) {
                    return Err(format!("--range {text:?}: LO must be below HI"));
                }
                options.range = Some((lo, hi));
            }
            "--obs" => options.obs = Some(PathBuf::from(value("--obs")?)),
            "--background" => options.background = Some(PathBuf::from(value("--background")?)),
            "--grid" => options.grid = Some(PathBuf::from(value("--grid")?)),
            "--types" => options.types = words(&value("--types")?),
            "--all" => options.all_panel = true,
            "--quantity-label" => options.quantity_label = value("--quantity-label")?,
            "--units" => options.units = Some(value("--units")?),
            "--valid" => options.valid = Some(value("--valid")?),
            "--dot-radius" => {
                options.dot_radius = value("--dot-radius")?
                    .parse()
                    .map_err(|_| "--dot-radius needs a whole number".to_string())?
            }
            other => return Err(format!("unknown option {other} for a DA sheet")),
        }
    }
    options.mode = mode.ok_or("--mode is required")?;
    options.out_dir = out_dir.ok_or("--out-dir is required")?;
    let (width, height) = if options.mode == Mode::Stamp { (600, 450) } else { (1_200, 900) };
    if options.width == 0 {
        options.width = width;
    }
    if options.height == 0 {
        options.height = height;
    }
    if !(256..=4096).contains(&options.width) || !(256..=4096).contains(&options.height) {
        return Err(format!(
            "panel size {}x{} is outside 256..4096 on a side",
            options.width, options.height
        ));
    }
    Ok(Some(options))
}

/// `rw_compare --mode ...`: parse, draw, write the sheet and its receipt.
pub fn cli(args: &[String]) -> Result<(), String> {
    let Some(options) = parse_options(args)? else {
        println!("{USAGE}");
        return Ok(());
    };
    std::fs::create_dir_all(&options.out_dir)
        .map_err(|error| format!("create {}: {error}", options.out_dir.display()))?;
    let scratch = options
        .out_dir
        .join(format!(".da-sheet-scratch-{}-{}", options.mode.name(), std::process::id()));
    std::fs::create_dir_all(&scratch)
        .map_err(|error| format!("create {}: {error}", scratch.display()))?;
    let drawn = match options.mode {
        Mode::Increment => increment_sheet(&options, &scratch),
        Mode::Spread => spread_sheet(&options, &scratch),
        Mode::Stamp => stamp_sheet(&options, &scratch),
        Mode::Omb => omb_sheet(&options, &scratch),
    };
    let _ = std::fs::remove_dir_all(&scratch);
    let drawn = drawn?;
    let stem = options
        .name
        .as_deref()
        .map(|name| safe_component(name, "sheet"))
        .unwrap_or_else(|| safe_component(&drawn.stem, "sheet"));
    let path = options.out_dir.join(format!("{stem}.png"));
    rustwx_render::save_rgba_png_profile_with_options(
        &drawn.image,
        &path,
        &PngWriteOptions {
            compression: PngCompressionMode::default(),
        },
    )
    .map_err(|error| format!("write {}: {error}", path.display()))?;
    let mut receipt = drawn.receipt;
    receipt["schema"] = serde_json::json!("gpuwm.da-sheet.v1");
    receipt["mode"] = serde_json::json!(options.mode.name());
    receipt["sheet"] = serde_json::json!(path);
    receipt["panel_size"] = serde_json::json!([options.width, options.height]);
    receipt["source_label"] = serde_json::json!(options.source_label);
    std::fs::write(
        path.with_extension("json"),
        serde_json::to_vec_pretty(&receipt).map_err(|error| error.to_string())?,
    )
    .map_err(|error| format!("write the receipt beside {}: {error}", path.display()))?;
    println!("RENDERED\t{}\t{}", options.mode.name(), path.display());
    Ok(())
}

/// What a mode hands back: the composed sheet, a default file stem and the
/// receipt body.
struct Drawn {
    image: RgbaImage,
    stem: String,
    receipt: serde_json::Value,
}

fn header_text(options: &Options, default_title: &str, grid: &SheetGrid, facts: Vec<String>) -> SheetHeader {
    let mut subtitle = vec![format!("Valid {}", grid.valid)];
    if let Some(init) = &grid.init {
        subtitle.push(format!("Init {init}"));
    }
    subtitle.extend(facts);
    subtitle.push(format!("source: {}", options.source_label));
    SheetHeader {
        title: options.title.clone().unwrap_or_else(|| default_title.to_string()),
        subtitle: subtitle.join(" | "),
    }
}

fn fallback_heights(options: &Options, grid: &SheetGrid) -> Result<Option<Vec<f64>>, String> {
    let Some(path) = &options.heights_from else {
        return Ok(None);
    };
    let file = open(path)?;
    if file.ny != grid.ny || file.nx != grid.nx {
        return Err(format!(
            "--heights-from {} is {}x{} and the sheet's grid is {}x{}",
            path.display(),
            file.ny,
            file.nx,
            grid.ny,
            grid.nx
        ));
    }
    let heights = own_heights(&file).ok_or_else(|| {
        format!("--heights-from {} carries no PH, PHB and HGT", path.display())
    })??;
    Ok(Some(heights.to_vec()))
}

fn stack_of(planes: &[Vec<f64>], files: &[PathBuf], grid: &SheetGrid) -> Result<MemberStack, String> {
    MemberStack::new(
        grid.ny,
        grid.nx,
        planes
            .iter()
            .zip(files)
            .enumerate()
            .map(|(position, (plane, file))| (member_number(file, position), plane.clone()))
            .collect(),
    )
    .map_err(|error| error.to_string())
}

fn mean_of(planes: &[Vec<f64>], files: &[PathBuf], grid: &SheetGrid) -> Result<Vec<f64>, String> {
    if planes.len() == 1 {
        return Ok(planes[0].clone());
    }
    ensemble_mean(&stack_of(planes, files, grid)?, NanPolicy::Mask).map_err(|error| error.to_string())
}

fn member_word(count: usize) -> String {
    if count == 1 { "1 member".to_string() } else { format!("{count} members") }
}

// ---------------------------------------------------------------------------
// Mode: increment
// ---------------------------------------------------------------------------

fn increment_sheet(options: &Options, scratch: &Path) -> Result<Drawn, String> {
    if options.fields.is_empty() {
        return Err("--mode increment needs --field".into());
    }
    let paired = !options.prior.is_empty() || !options.posterior.is_empty();
    if paired && (options.prior.is_empty() || options.posterior.is_empty()) {
        return Err("--prior and --posterior come together; give both, or --increment".into());
    }
    if paired == !options.increment.is_empty() {
        return Err("give --prior with --posterior, or --increment, not both".into());
    }
    let first = if paired { &options.prior[0] } else { &options.increment[0] };
    let grid = sheet_grid(first)?;
    for file in options.prior.iter().chain(&options.posterior).chain(&options.increment).skip(1) {
        check_same_grid(&grid, file)?;
    }
    let fallback = fallback_heights(options, &grid)?;
    let probe = open(first)?;
    let shape = (grid.ny, grid.nx);
    let right = format!("source: {}", options.source_label);
    let canvas = Canvas { grid: &grid, width: options.width, height: options.height, scratch, right: &right };
    let mut rows = Vec::new();
    let mut receipts = Vec::new();
    let mut index = 0usize;
    let mut resolved = Vec::new();
    for field in &options.fields {
        let expr = resolve_field(field, &probe)?;
        let levels = field_levels(&expr, &probe, &options.heights)?;
        resolved.push((expr, levels));
    }
    // An increment input drawn at several heights reads best as a grid:
    // one row per field, one column per height, one ladder per row.
    let columns = resolved[0].1.len();
    if !paired && columns > 1 && resolved.iter().all(|(_, levels)| levels.len() == columns) {
        for (expr, levels) in &resolved {
            let key = format!("increment_{}", safe_component(&expr.key, "field"));
            let units = expr.units.clone();
            let mut planes = Vec::with_capacity(levels.len());
            for level in levels {
                let read = read_planes(&options.increment, expr, *level, fallback.as_deref(), false, shape)?;
                let mean = mean_of(&read, &options.increment, &grid)?;
                planes.push(mean.iter().map(|value| *value as f32).collect::<Vec<f32>>());
            }
            let pooled: Vec<f32> = planes.iter().flatten().copied().collect();
            let (scale, half, rule) = diverging(&key, &units, &pooled, options.half_range);
            let mut row = Vec::new();
            for (level, plane) in levels.iter().zip(planes) {
                let stats = plane_stats(&plane.iter().map(|value| f64::from(*value)).collect::<Vec<_>>());
                println!(
                    "STATS	increment	{}	{}	points={}	mean={:.4e}	rms={:.4e}	min={:.4e}	max={:.4e}",
                    expr.key, level.token(), stats.points, stats.mean, stats.rms, stats.min, stats.max
                );
                row.push(draw(&canvas, index, diverging_panel(
                    key.clone(),
                    format!("{} increment{}", expr.title, level.phrase(expr)),
                    format!("+-{} {units}, one ladder per row ({rule})", short(half)),
                    plane,
                    &units,
                    scale.clone(),
                    half,
                ))?);
                index += 1;
                receipts.push(serde_json::json!({
                    "field": expr.key, "title": expr.title, "level": level.text(expr), "units": units,
                    "increment": stats_json(&stats), "half_range": half, "half_range_rule": rule,
                    "increment_files": options.increment,
                }));
            }
            rows.push(row);
        }
    }
    let as_grid = !rows.is_empty();
    for (expr, levels) in resolved.iter().filter(|_| !as_grid) {
        let expr = expr.clone();
        for level in levels.iter().copied() {
            let where_ = level.text(&expr);
            let phrase = level.phrase(&expr);
            let mut row = Vec::new();
            let key = format!("increment_{}", safe_component(&expr.key, "field"));
            let (increment, receipt_extra) = if paired {
                let prior = mean_of(
                    &read_planes(&options.prior, &expr, level, fallback.as_deref(), true, shape)?,
                    &options.prior,
                    &grid,
                )?;
                let posterior = mean_of(
                    &read_planes(&options.posterior, &expr, level, fallback.as_deref(), true, shape)?,
                    &options.posterior,
                    &grid,
                )?;
                let style = field_style(&expr, &[&prior, &posterior], options.range);
                let floor = expr
                    .production
                    .as_ref()
                    .and_then(|style| rustwx_render::difference::undrawn_floor(&style.scale));
                let prior_drawn = to_f32(&prior, &style);
                let posterior_drawn = to_f32(&posterior, &style);
                let increment = crate::compare::difference_above_floor(&posterior_drawn, &prior_drawn, floor);
                for (label, files, values) in [("Prior", &options.prior, &prior), ("Posterior", &options.posterior, &posterior)] {
                    let title = if files.len() > 1 {
                        format!("{label} mean ({})", member_word(files.len()))
                    } else {
                        label.to_string()
                    };
                    row.push(draw(&canvas, index, field_panel(
                        format!("{}_{}", label.to_ascii_lowercase(), key),
                        title,
                        format!("{}{phrase} | {}", expr.title, style.rule),
                        values,
                        &style,
                    ))?);
                    index += 1;
                }
                let extra = serde_json::json!({
                    "prior": stats_json(&plane_stats(&prior)),
                    "posterior": stats_json(&plane_stats(&posterior)),
                    "prior_members": options.prior, "posterior_members": options.posterior,
                    "field_scale": style.rule, "units": style.units,
                });
                (increment, extra)
            } else {
                let planes = read_planes(&options.increment, &expr, level, fallback.as_deref(), false, shape)?;
                let mean = mean_of(&planes, &options.increment, &grid)?;
                let extra = serde_json::json!({"increment_files": options.increment});
                (mean.iter().map(|value| *value as f32).collect(), extra)
            };
            let units = expr
                .production
                .as_ref()
                .map(|style| style.display_units.clone())
                .unwrap_or_else(|| expr.units.clone());
            let (scale, half, rule) = diverging(&key, &units, &increment, options.half_range);
            let as_f64: Vec<f64> = increment.iter().map(|value| f64::from(*value)).collect();
            let stats = plane_stats(&as_f64);
            println!(
                "STATS\tincrement\t{}\t{}\tpoints={}\tmean={:.4e}\trms={:.4e}\tmin={:.4e}\tmax={:.4e}",
                expr.key, level.token(), stats.points, stats.mean, stats.rms, stats.min, stats.max
            );
            let title = if paired {
                "Increment: posterior minus prior".to_string()
            } else if options.increment.len() > 1 {
                format!("Mean analysis increment ({})", member_word(options.increment.len()))
            } else {
                "Analysis increment".to_string()
            };
            row.push(draw(&canvas, index, diverging_panel(
                key.clone(),
                title,
                format!("{}{phrase} | +-{} {units} ({rule})", expr.title, short(half)),
                increment,
                &units,
                scale,
                half,
            ))?);
            index += 1;
            let mut receipt = serde_json::json!({
                "field": expr.key, "title": expr.title, "level": where_, "units": units,
                "increment": stats_json(&stats), "half_range": half, "half_range_rule": rule,
            });
            if let (Some(target), serde_json::Value::Object(extra)) = (receipt.as_object_mut(), receipt_extra) {
                target.extend(extra);
            }
            receipts.push(receipt);
            rows.push(row);
        }
    }
    let facts = vec![if paired {
        format!(
            "prior {} and posterior {}; increment = posterior mean minus prior mean",
            member_word(options.prior.len()),
            member_word(options.posterior.len())
        )
    } else {
        format!("increment input: {}", member_word(options.increment.len()))
    }];
    let header = header_text(options, "Analysis increments", &grid, facts);
    let image = compose_sheet_rows(&rows, &header)?;
    Ok(Drawn {
        image,
        stem: format!(
            "increment_{}_{}",
            options.fields.iter().map(|f| safe_component(f, "field")).collect::<Vec<_>>().join("-"),
            safe_component(&grid.valid, "valid")
        ),
        receipt: serde_json::json!({"rows": receipts, "valid": grid.valid, "init": grid.init}),
    })
}

// ---------------------------------------------------------------------------
// Mode: spread
// ---------------------------------------------------------------------------

fn spread_sheet(options: &Options, scratch: &Path) -> Result<Drawn, String> {
    let [members] = options.members.as_slice() else {
        return Err("--mode spread takes one --members list".into());
    };
    if members.len() < 2 {
        return Err("ensemble spread needs at least two members".into());
    }
    for column in &options.columns {
        if !["mean", "spread", "pmm", "max"].contains(&column.as_str()) {
            return Err(format!("--columns {column:?} is not mean, spread, pmm or max"));
        }
    }
    let fields = if options.fields.is_empty() { vec!["refc".to_string()] } else { options.fields.clone() };
    let grid = sheet_grid(&members[0])?;
    for file in members.iter().skip(1) {
        check_same_grid(&grid, file)?;
    }
    let fallback = fallback_heights(options, &grid)?;
    let probe = open(&members[0])?;
    let right = format!("source: {}", options.source_label);
    let canvas = Canvas { grid: &grid, width: options.width, height: options.height, scratch, right: &right };
    let shape = (grid.ny, grid.nx);
    let n = members.len();
    let mut rows = Vec::new();
    let mut receipts = Vec::new();
    let mut index = 0usize;
    for field in &fields {
        let expr = resolve_field(field, &probe)?;
        for level in field_levels(&expr, &probe, &options.heights)? {
            let where_ = level.text(&expr);
    let phrase = level.phrase(&expr);
            let planes = read_planes(members, &expr, level, fallback.as_deref(), true, shape)?;
            let stack = stack_of(&planes, members, &grid)?;
            let mean = ensemble_mean(&stack, NanPolicy::Mask).map_err(|e| e.to_string())?;
            // A field whose production ladder draws nothing below a floor
            // (reflectivity) spreads its members' "nothing here" sentinels
            // over clear air.  Its spread is taken with every value below
            // the floor counted as the floor, the rule the difference
            // panels use, and left blank where every member is below it.
            let floor = expr
                .production
                .as_ref()
                .and_then(|style| rustwx_render::difference::undrawn_floor(&style.scale).map(|floor| (style, floor)));
            let spread = match floor {
                None => ensemble_spread(&stack, 1, NanPolicy::Mask).map_err(|e| e.to_string())?,
                Some((style, floor)) => {
                    let clamped: Vec<Vec<f64>> = planes
                        .iter()
                        .map(|plane| plane.iter().map(|value| f64::from(style.convert.apply(*value as f32)).max(floor)).collect())
                        .collect();
                    let spread = ensemble_spread(&stack_of(&clamped, members, &grid)?, 1, NanPolicy::Mask)
                        .map_err(|e| e.to_string())?;
                    spread
                        .into_iter()
                        .enumerate()
                        .map(|(point, value)| {
                            if clamped.iter().all(|plane| plane[point] <= floor) { f64::NAN } else { value }
                        })
                        .collect()
                }
            };
            let spread_note = floor
                .map(|(_, floor)| format!(" | below {} counted as {}", short(floor), short(floor)))
                .unwrap_or_default();
            let pmm = probability_matched_mean(&stack, NanPolicy::Mask, PmmTieRule::FlatIndex)
                .map_err(|e| e.to_string())?;
            let max = ensemble_max(&stack, NanPolicy::Mask).map_err(|e| e.to_string())?;
            let field_planes: Vec<&[f64]> = options
                .columns
                .iter()
                .filter_map(|column| match column.as_str() {
                    "mean" => Some(mean.as_slice()),
                    "pmm" => Some(pmm.as_slice()),
                    "max" => Some(max.as_slice()),
                    _ => None,
                })
                .collect();
            let style = field_style(&expr, &field_planes, options.range);
            let key = safe_component(&expr.key, "field");
            let mut row = Vec::new();
            let (top, top_rule) = spread_top(&spread, options.spread_max);
            for column in &options.columns {
                let panel = match column.as_str() {
                    "mean" => field_panel(format!("ens_mean_{key}"), format!("Ensemble mean ({n} members)"),
                        format!("{}{phrase} | {}", expr.title, style.rule), &mean, &style),
                    "pmm" => field_panel(format!("ens_pmm_{key}"), format!("Probability-matched mean ({n} members)"),
                        format!("{}{phrase} | {}", expr.title, style.rule), &pmm, &style),
                    "max" => field_panel(format!("ens_max_{key}"), format!("Ensemble maximum ({n} members)"),
                        format!("{}{phrase} | {}", expr.title, style.rule), &max, &style),
                    _ => Panel {
                        slug: format!("ens_spread_{key}"),
                        title: format!("Ensemble spread, 1 sigma ({n} members)"),
                        units: expr.units.clone(),
                        values: spread.iter().map(|value| *value as f32).collect(),
                        scale: crate::scales::spread_scale(top),
                        tick: None,
                        legend: stepped(),
                        density: RenderDensity::default(),
                        left: format!("{}{phrase} | 0 to {} {} ({top_rule}){spread_note}", expr.title, short(top), expr.units),
                        contours: Vec::new(),
                        colorbar: true,
                        overlays: None,
                    },
                };
                row.push(draw(&canvas, index, panel)?);
                index += 1;
            }
            let spread_stats = plane_stats(&spread);
            println!(
                "STATS\tspread\t{}\t{}\tpoints={}\tmean={:.4e}\tmax={:.4e}",
                expr.key, level.token(), spread_stats.points, spread_stats.mean, spread_stats.max
            );
            receipts.push(serde_json::json!({
                "field": expr.key, "title": expr.title, "level": where_, "units": expr.units,
                "columns": options.columns, "field_scale": style.rule,
                "spread": stats_json(&spread_stats), "spread_top": top, "spread_rule": top_rule,
                "mean": stats_json(&plane_stats(&mean)), "pmm": stats_json(&plane_stats(&pmm)),
            }));
            rows.push(row);
        }
    }
    let header = header_text(options, "Ensemble mean and spread", &grid, vec![format!("{n} members, spread with ddof=1")]);
    let image = compose_sheet_rows(&rows, &header)?;
    Ok(Drawn {
        image,
        stem: format!(
            "spread_{}_{}",
            fields.iter().map(|f| safe_component(f, "field")).collect::<Vec<_>>().join("-"),
            safe_component(&grid.valid, "valid")
        ),
        receipt: serde_json::json!({"rows": receipts, "members": members, "valid": grid.valid, "init": grid.init}),
    })
}

// ---------------------------------------------------------------------------
// Mode: stamp
// ---------------------------------------------------------------------------

fn stamp_sheet(options: &Options, scratch: &Path) -> Result<Drawn, String> {
    if options.members.is_empty() {
        return Err("--mode stamp needs --members (one list per row)".into());
    }
    for extra in &options.extras {
        if !["paintball", "max", "pmm", "mean"].contains(&extra.as_str()) {
            return Err(format!("--extras {extra:?} is not paintball, max, pmm or mean"));
        }
    }
    let field = match options.fields.as_slice() {
        [] => "refc".to_string(),
        [one] => one.clone(),
        _ => return Err("--mode stamp draws one --field".into()),
    };
    let level = match options.heights.as_slice() {
        [] => Level::AsStored,
        [metres] => Level::HeightAgl(*metres),
        _ => return Err("--mode stamp draws one --height-agl".into()),
    };
    let grid = sheet_grid(&options.members[0][0])?;
    for file in options.members.iter().flatten().skip(1) {
        check_same_grid(&grid, file)?;
    }
    let fallback = fallback_heights(options, &grid)?;
    let probe = open(&options.members[0][0])?;
    let expr = resolve_field(&field, &probe)?;
    let shape = (grid.ny, grid.nx);
    let groups: Vec<Vec<Vec<f64>>> = options
        .members
        .iter()
        .map(|files| read_planes(files, &expr, level, fallback.as_deref(), true, shape))
        .collect::<Result<_, _>>()?;
    let all: Vec<&[f64]> = groups.iter().flatten().map(Vec::as_slice).collect();
    let style = field_style(&expr, &all, options.range);
    let thresholds = if options.thresholds.is_empty() {
        if style.units.eq_ignore_ascii_case("dbz") {
            vec![35.0, 45.0]
        } else if options.extras.iter().any(|extra| extra == "paintball") {
            return Err(format!("--thresholds is needed for a paintball of {} ({})", expr.key, style.units));
        } else {
            Vec::new()
        }
    } else {
        options.thresholds.clone()
    };
    let right = format!("source: {}", options.source_label);
    let canvas = Canvas { grid: &grid, width: options.width, height: options.height, scratch, right: &right };
    let where_ = level.text(&expr);
    let phrase = level.phrase(&expr);
    let key = safe_component(&expr.key, "field");
    let mut rows = Vec::new();
    let mut receipts = Vec::new();
    let mut index = 0usize;
    let stamps = options.stamps.min(options.members.iter().map(Vec::len).min().unwrap_or(0));
    for (group, (files, planes)) in options.members.iter().zip(&groups).enumerate() {
        let prefix = options.row_labels.get(group).map(|label| format!("{label}: ")).unwrap_or_default();
        let stack = stack_of(planes, files, &grid)?;
        let n = files.len();
        let mut row = Vec::new();
        for (position, (file, plane)) in files.iter().zip(planes).take(stamps).enumerate() {
            let number = member_number(file, position);
            row.push(draw(&canvas, index, field_panel(
                format!("stamp_{key}"),
                format!("{prefix}member {number}"),
                format!("{}{phrase}", expr.title),
                plane,
                &style,
            ))?);
            index += 1;
        }
        for extra in &options.extras {
            match extra.as_str() {
                "paintball" => {
                    for threshold in &thresholds {
                        let contours = stack
                            .members()
                            .map(|(number, plane)| ContourLayer {
                                data: plane.iter().map(|value| style.convert(*value)).collect(),
                                levels: vec![*threshold],
                                color: parse_color(member_color(number)),
                                width: 2,
                                labels: false,
                                show_extrema: false,
                                pattern: Default::default(),
                                major_every: None,
                                major_width: None,
                            })
                            .collect();
                        row.push(draw(&canvas, index, Panel {
                            slug: format!("paintball_{key}"),
                            title: format!("{prefix}paintball {} {} ({n} members)", short(*threshold), style.units),
                            units: style.units.clone(),
                            values: vec![f32::NAN; grid.ny * grid.nx],
                            scale: crate::scales::spread_scale(1.0),
                            tick: None,
                            legend: stepped(),
                            density: RenderDensity::default(),
                            left: format!("{}{phrase} | one colour per member number", expr.title),
                            contours,
                            colorbar: false,
                            overlays: None,
                        })?);
                        index += 1;
                    }
                }
                "max" => {
                    let max: Vec<f64> = (0..stack.points())
                        .map(|point| {
                            (0..stack.len())
                                .map(|member| stack.member(member)[point])
                                .filter(|value| value.is_finite())
                                .fold(f64::NAN, |best, value| if best.is_nan() || value > best { value } else { best })
                        })
                        .collect();
                    row.push(draw(&canvas, index, field_panel(
                        format!("ens_max_{key}"),
                        format!("{prefix}ensemble maximum ({n} members)"),
                        format!("{}{phrase}", expr.title),
                        &max,
                        &style,
                    ))?);
                    index += 1;
                }
                "pmm" => {
                    let pmm = probability_matched_mean(&stack, NanPolicy::Mask, PmmTieRule::FlatIndex)
                        .map_err(|e| e.to_string())?;
                    row.push(draw(&canvas, index, field_panel(
                        format!("ens_pmm_{key}"),
                        format!("{prefix}probability-matched mean ({n} members)"),
                        format!("{}{phrase}", expr.title),
                        &pmm,
                        &style,
                    ))?);
                    index += 1;
                }
                _ => {
                    let mean = ensemble_mean(&stack, NanPolicy::Mask).map_err(|e| e.to_string())?;
                    row.push(draw(&canvas, index, field_panel(
                        format!("ens_mean_{key}"),
                        format!("{prefix}ensemble mean ({n} members)"),
                        format!("{}{phrase}", expr.title),
                        &mean,
                        &style,
                    ))?);
                    index += 1;
                }
            }
        }
        receipts.push(serde_json::json!({
            "row": group, "label": options.row_labels.get(group), "members": files,
            "member_numbers": stack.numbers(), "stamps": stamps,
        }));
        rows.push(row);
    }
    let roster = options.members[0]
        .iter()
        .enumerate()
        .map(|(position, file)| {
            let number = member_number(file, position);
            format!("{number}={}", member_color(number))
        })
        .collect::<Vec<_>>()
        .join(" ");
    println!("PAINTBALL_LEGEND\t{roster}");
    let header = header_text(options, &format!("{} ensemble: stamps, paintball, maximum, PMM", expr.title), &grid,
        vec![format!("{}{phrase}", expr.title), format!("paintball at {} {}", thresholds.iter().map(|t| short(*t)).collect::<Vec<_>>().join(", "), style.units)]);
    let image = compose_sheet_rows(&rows, &header)?;
    Ok(Drawn {
        image,
        stem: format!("stamp_{key}_{}", safe_component(&grid.valid, "valid")),
        receipt: serde_json::json!({
            "field": expr.key, "level": where_, "thresholds": thresholds, "extras": options.extras,
            "rows": receipts, "paintball_colours": roster, "valid": grid.valid, "init": grid.init,
        }),
    })
}

// ---------------------------------------------------------------------------
// Mode: omb
// ---------------------------------------------------------------------------

/// One observation, its departure and whether quality control kept it.
#[derive(Clone, Debug, PartialEq)]
pub struct Departure {
    pub lat: f64,
    pub lon: f64,
    pub value: f64,
    pub kind: String,
    pub rejected: bool,
    pub units: Option<String>,
}

fn truthy(value: &serde_json::Value) -> bool {
    match value {
        serde_json::Value::Bool(flag) => *flag,
        serde_json::Value::Number(number) => number.as_f64().is_some_and(|n| n != 0.0),
        serde_json::Value::String(text) => truthy_text(text),
        _ => false,
    }
}

fn truthy_text(text: &str) -> bool {
    matches!(text.trim().to_ascii_lowercase().as_str(), "true" | "1" | "yes" | "y" | "rejected" | "t")
}

/// Departures from a JSON array of rows (or `{"valid":..., "observations":
/// [...]}`) or a CSV with a header row.  Column names: `lat`/`latitude`,
/// `lon`/`longitude`, `value` (or `omb`, `oma`, `departure`), `type` (or
/// `kind`), `rejected` (or `qc_rejected`), optional `units`.
pub fn read_departures(path: &Path) -> Result<(Vec<Departure>, Option<String>), String> {
    let text = std::fs::read_to_string(path).map_err(|error| format!("read {}: {error}", path.display()))?;
    let trimmed = text.trim_start();
    if trimmed.starts_with('[') || trimmed.starts_with('{') {
        let document: serde_json::Value = serde_json::from_str(&text)
            .map_err(|error| format!("parse {}: {error}", path.display()))?;
        let (rows, valid) = match &document {
            serde_json::Value::Array(rows) => (rows.clone(), None),
            serde_json::Value::Object(object) => (
                object
                    .get("observations")
                    .and_then(|rows| rows.as_array())
                    .cloned()
                    .ok_or_else(|| format!("{}: an object needs an \"observations\" array", path.display()))?,
                object.get("valid").and_then(|valid| valid.as_str()).map(str::to_string),
            ),
            _ => unreachable!("checked by the first character"),
        };
        let field = |row: &serde_json::Value, names: &[&str]| -> Option<serde_json::Value> {
            names.iter().find_map(|name| row.get(*name).cloned())
        };
        let mut out = Vec::with_capacity(rows.len());
        for (index, row) in rows.iter().enumerate() {
            let number = |names: &[&str]| -> Result<f64, String> {
                field(row, names)
                    .and_then(|value| value.as_f64())
                    .ok_or_else(|| format!("{} row {index}: no numeric {}", path.display(), names[0]))
            };
            out.push(Departure {
                lat: number(&["lat", "latitude"])?,
                lon: number(&["lon", "longitude"])?,
                value: number(&["value", "omb", "oma", "departure"])?,
                kind: field(row, &["type", "kind"])
                    .and_then(|value| value.as_str().map(str::to_string))
                    .unwrap_or_else(|| "all".to_string()),
                rejected: field(row, &["rejected", "qc_rejected"]).is_some_and(|value| truthy(&value)),
                units: field(row, &["units"]).and_then(|value| value.as_str().map(str::to_string)),
            });
        }
        return Ok((out, valid));
    }
    let mut lines = text.lines().filter(|line| !line.trim().is_empty() && !line.starts_with('#'));
    let header: Vec<String> = lines
        .next()
        .ok_or_else(|| format!("{} is empty", path.display()))?
        .split(',')
        .map(|name| name.trim().to_ascii_lowercase())
        .collect();
    let column = |names: &[&str]| header.iter().position(|name| names.contains(&name.as_str()));
    let lat = column(&["lat", "latitude"]).ok_or_else(|| format!("{}: no lat column", path.display()))?;
    let lon = column(&["lon", "longitude"]).ok_or_else(|| format!("{}: no lon column", path.display()))?;
    let value = column(&["value", "omb", "oma", "departure"])
        .ok_or_else(|| format!("{}: no value column", path.display()))?;
    let kind = column(&["type", "kind"]);
    let rejected = column(&["rejected", "qc_rejected"]);
    let units = column(&["units"]);
    let mut out = Vec::new();
    for (index, line) in lines.enumerate() {
        let cells: Vec<&str> = line.split(',').map(str::trim).collect();
        let number = |at: usize| -> Result<f64, String> {
            cells
                .get(at)
                .and_then(|cell| cell.parse::<f64>().ok())
                .ok_or_else(|| format!("{} data row {index}: column {} is not a number", path.display(), header[at]))
        };
        out.push(Departure {
            lat: number(lat)?,
            lon: number(lon)?,
            value: number(value)?,
            kind: kind.and_then(|at| cells.get(at)).map(|cell| cell.to_string()).unwrap_or_else(|| "all".into()),
            rejected: rejected.and_then(|at| cells.get(at)).is_some_and(|cell| truthy_text(cell)),
            units: units.and_then(|at| cells.get(at)).map(|cell| cell.to_string()).filter(|cell| !cell.is_empty()),
        });
    }
    Ok((out, None))
}

fn omb_sheet(options: &Options, scratch: &Path) -> Result<Drawn, String> {
    let obs = options.obs.as_ref().ok_or("--mode omb needs --obs FILE")?;
    let map_file = options
        .background
        .as_ref()
        .or(options.grid.as_ref())
        .ok_or("--mode omb needs --grid WRFOUT (the map) or --background WRFOUT with --field")?;
    let mut grid = sheet_grid(map_file)?;
    let (departures, file_valid) = read_departures(obs)?;
    if let Some(valid) = options.valid.clone().or(file_valid) {
        grid.valid = time_label(&valid);
    }
    let mut kinds: Vec<String> = Vec::new();
    for departure in &departures {
        if !kinds.contains(&departure.kind) {
            kinds.push(departure.kind.clone());
        }
    }
    let mut panels: Vec<(String, Vec<&Departure>)> = Vec::new();
    let chosen = if options.types.is_empty() { kinds.clone() } else { options.types.clone() };
    if options.all_panel {
        // Every chosen type on one map: only meaningful in one unit.
        let pooled: Vec<&Departure> =
            departures.iter().filter(|departure| chosen.contains(&departure.kind)).collect();
        let mut units: Vec<&str> = pooled.iter().filter_map(|row| row.units.as_deref()).collect();
        units.sort_unstable();
        units.dedup();
        if units.len() > 1 {
            return Err(format!(
                "--all pools {} observations in {}; one map of departures is one unit (choose --types)",
                pooled.len(),
                units.join(" and ")
            ));
        }
        let label = if options.types.is_empty() { "all types".to_string() } else { chosen.join(" + ") };
        panels.push((label, pooled));
    }
    for kind in &chosen {
        let rows: Vec<&Departure> = departures.iter().filter(|departure| &departure.kind == kind).collect();
        if rows.is_empty() {
            return Err(format!("--types names {kind:?} and {} holds no such observation (types: {})", obs.display(), kinds.join(", ")));
        }
        panels.push((kind.clone(), rows));
    }
    if panels.is_empty() {
        return Err(format!("{} holds no observation", obs.display()));
    }
    let background = match &options.background {
        Some(path) => {
            let probe = open(path)?;
            let field = options.fields.first().ok_or("--background needs --field NAME")?;
            let expr = resolve_field(field, &probe)?;
            let level = match options.heights.as_slice() {
                [] => Level::AsStored,
                [metres] => Level::HeightAgl(*metres),
                _ => return Err("--background draws one --height-agl".into()),
            };
            let fallback = fallback_heights(options, &grid)?;
            let plane = read_plane(path, &expr, level, fallback.as_deref(), true, (grid.ny, grid.nx))?;
            let style = field_style(&expr, &[&plane], options.range);
            Some((expr, level, plane, style))
        }
        None => None,
    };
    let right = format!("source: {}", options.source_label);
    let canvas = Canvas { grid: &grid, width: options.width, height: options.height, scratch, right: &right };
    let mut row = Vec::new();
    let mut receipts = Vec::new();
    let mut keys: Vec<(String, ColorScale)> = Vec::new();
    for (index, (kind, rows)) in panels.iter().enumerate() {
        let units = options
            .units
            .clone()
            .or_else(|| rows.iter().find_map(|row| row.units.clone()))
            .unwrap_or_default();
        let kept: Vec<&&Departure> = rows.iter().filter(|row| !row.rejected).collect();
        let rejected: Vec<&&Departure> = rows.iter().filter(|row| row.rejected).collect();
        let values: Vec<f32> = kept.iter().map(|row| row.value as f32).collect();
        let key = format!("omb_{}", safe_component(kind, "type"));
        let (scale, half, rule) = diverging(&key, &units, &values, options.half_range);
        let overlays = MapOverlays {
            value_layers: vec![ValuePointLayer {
                products: Vec::new(),
                units: units.clone(),
                points: kept
                    .iter()
                    .map(|row| ValuePointSpec { lat: row.lat, lon: row.lon, value: row.value })
                    .collect(),
                scale: Some(scale.clone()),
                radius_px: options.dot_radius,
            }],
            points: rejected
                .iter()
                .map(|row| PointSpec {
                    lat: row.lat,
                    lon: row.lon,
                    color: "#000000".to_string(),
                    radius_px: options.dot_radius + 1,
                    width_px: 1,
                    shape: "cross".to_string(),
                })
                .collect(),
            ..MapOverlays::default()
        };
        let as_f64: Vec<f64> = kept.iter().map(|row| row.value).collect();
        let stats = plane_stats(&as_f64);
        let left = format!(
            "{} used, {} rejected (x) | mean {} rms {} {units} | +-{} ({rule})",
            kept.len(),
            rejected.len(),
            short(stats.mean),
            short(stats.rms),
            short(half)
        );
        let title = format!("{} {kind}", options.quantity_label);
        let panel = match &background {
            Some((expr, _, plane, style)) => {
                let label = format!("{} {kind} ({units})", options.quantity_label);
                if !keys.iter().any(|(have, _)| *have == label) {
                    keys.push((label, scale.clone()));
                }
                Panel {
                    overlays: Some(overlays),
                    ..field_panel(key.clone(), format!("{title} over {}", expr.title), left, plane, style)
                }
            }
            None => Panel {
                slug: key.clone(),
                title,
                units: units.clone(),
                values: vec![f32::NAN; grid.ny * grid.nx],
                scale,
                tick: Some(2.0 * half / DifferenceTable::builtin().bands.max(2) as f64),
                legend: stepped(),
                density: RenderDensity::default(),
                left,
                contours: Vec::new(),
                colorbar: true,
                overlays: Some(overlays),
            },
        };
        row.push(draw(&canvas, index, panel)?);
        println!(
            "STATS\tomb\t{kind}\tused={}\trejected={}\tmean={:.4}\trms={:.4}",
            kept.len(), rejected.len(), stats.mean, stats.rms
        );
        receipts.push(serde_json::json!({
            "type": kind, "units": units, "used": kept.len(), "rejected": rejected.len(),
            "departure": stats_json(&stats), "half_range": half, "half_range_rule": rule,
        }));
    }
    let mut facts = vec![format!("{} dots, x = rejected by quality control", options.quantity_label)];
    if let Some((expr, level, _, style)) = &background {
        facts.push(format!("shading: {} {} ({})", expr.title, level.text(expr), style.rule));
    }
    let header = header_text(options, &format!("{} by observation type", options.quantity_label), &grid, facts);
    let mut image = compose_sheet_rows(&[row], &header)?;
    for (label, scale) in &keys {
        image = append_scale_key(&image, label, scale)?;
    }
    Ok(Drawn {
        image,
        stem: format!(
            "{}_{}",
            safe_component(&options.quantity_label.to_ascii_lowercase(), "omb"),
            safe_component(&grid.valid, "valid")
        ),
        receipt: serde_json::json!({
            "observations": obs, "map": map_file, "panels": receipts, "valid": grid.valid,
            "types": kinds,
        }),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wildcards_match_runs_and_single_characters() {
        assert!(wildcard_match("wrfout_leg01_*.nc", "wrfout_leg01_12.nc"));
        assert!(!wildcard_match("wrfout_leg01_*.nc", "wrfout_leg01_12.npz"));
        assert!(wildcard_match("a?c", "abc"));
        assert!(!wildcard_match("a?c", "abbc"));
        assert!(wildcard_match("*", ""));
    }

    #[test]
    fn member_numbers_come_from_the_stem_or_the_position() {
        assert_eq!(member_number(Path::new("/x/wrfout_leg01_17.nc"), 3), 17);
        assert_eq!(member_number(Path::new("/x/wrfout_leg01_control.nc"), 3), 3);
        let mut files = vec![PathBuf::from("m_10.nc"), PathBuf::from("m_2.nc"), PathBuf::from("m_1.nc")];
        files.sort_by(|a, b| natural_order(a, b));
        assert_eq!(files, vec![PathBuf::from("m_1.nc"), PathBuf::from("m_2.nc"), PathBuf::from("m_10.nc")]);
    }

    #[test]
    fn heights_interpolate_linearly_and_refuse_outside_the_column() {
        // Two points, three levels at 500, 1500, 2500 m and 0, 1000, 3000 m.
        let heights = [500.0, 0.0, 1_500.0, 1_000.0, 2_500.0, 3_000.0];
        let values = [10.0, 0.0, 20.0, 10.0, 30.0, 30.0];
        let at = interpolate_to_height(&values, &heights, 3, 2, 1_000.0);
        assert_eq!(at, vec![15.0, 10.0]);
        let low = interpolate_to_height(&values, &heights, 3, 2, 100.0);
        assert!(low[0].is_nan(), "below the lowest level is not extrapolated");
        assert_eq!(low[1], 1.0);
        assert!(interpolate_to_height(&values, &heights, 3, 2, 9_000.0).iter().all(|v| v.is_nan()));
        let staggered = destagger_z(&[0.0, 2.0, 4.0, 6.0, 8.0, 10.0], 3, 2);
        assert_eq!(staggered, vec![2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn departures_read_from_json_and_csv_alike() {
        let dir = std::env::temp_dir().join(format!("rw-da-obs-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let json = dir.join("obs.json");
        std::fs::write(&json, r#"{"valid":"2026-10-01T19:00:00","observations":[
            {"lat":35.0,"lon":-97.0,"value":-2.5,"type":"radar_z","rejected":false,"units":"dBZ"},
            {"lat":35.5,"lon":-97.5,"value":7.0,"type":"metar_t","rejected":1}]}"#).unwrap();
        let (rows, valid) = read_departures(&json).unwrap();
        assert_eq!(valid.as_deref(), Some("2026-10-01T19:00:00"));
        assert_eq!(rows.len(), 2);
        assert!(!rows[0].rejected && rows[1].rejected);
        assert_eq!(rows[0].units.as_deref(), Some("dBZ"));
        let csv = dir.join("obs.csv");
        std::fs::write(&csv, "latitude,longitude,omb,type,qc_rejected\n35,-97,-2.5,radar_z,no\n35.5,-97.5,7,metar_t,yes\n").unwrap();
        let (csv_rows, _) = read_departures(&csv).unwrap();
        assert_eq!(csv_rows.len(), 2);
        assert_eq!(csv_rows[0].value, rows[0].value);
        assert_eq!(csv_rows[1].kind, "metar_t");
        assert!(csv_rows[1].rejected);
        std::fs::write(&csv, "lat,lon\n1,2\n").unwrap();
        assert!(read_departures(&csv).is_err(), "a file with no value column is refused");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn increments_and_departures_wear_a_zero_centred_ladder() {
        let values: Vec<f32> = (-50..=50).map(|v| v as f32 * 0.01).collect();
        let (scale, half, rule) = diverging("increment_qv", "g/kg", &values, None);
        let discrete = scale.resolved_discrete();
        assert!((discrete.levels.first().unwrap() + half).abs() < 1e-9);
        assert!((discrete.levels.last().unwrap() - half).abs() < 1e-9);
        assert!(discrete.levels.contains(&0.0), "zero is a band edge");
        assert!(half >= 0.5 && rule.starts_with('p'), "{half} {rule}");
        let (_, fixed, rule) = diverging("increment_qv", "g/kg", &values, Some(2.0));
        assert_eq!((fixed, rule.as_str()), (2.0, "--half-range"));
    }

    #[test]
    fn the_spread_top_is_a_whole_number_of_nice_bands() {
        let spread: Vec<f64> = (0..100).map(|v| v as f64 * 0.1).collect();
        let (top, rule) = spread_top(&spread, None);
        assert!(top >= 9.8 && rule.starts_with("p99"), "{top} {rule}");
        let step = top / 9.0;
        assert!((nice_step(step) - step).abs() < 1e-9);
        assert_eq!(spread_top(&spread, Some(4.0)).0, 4.0);
        assert_eq!(spread_top(&[0.0, 0.0], None).0, 1.0);
    }
}
