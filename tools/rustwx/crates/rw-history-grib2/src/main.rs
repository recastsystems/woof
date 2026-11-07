//
// Copyright 2026 the WOOF authors.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License.  You may obtain a copy
// of the License at https://www.apache.org/licenses/LICENSE-2.0.  Unless
// required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied.

//! `rw_history_grib2 --table TABLE.json IN.nc OUT.grib2 [--window-seconds N] [--threads N]`
//!
//! One WOOF forecast file as GRIB2, losslessly. IN is a forecast file as WOOF
//! stores it (ArWen's wrfout on its Lambert, Mercator or polar grid, a hex door's
//! Lambert or latitude-longitude export; NetCDF classic or NetCDF-4, whole or a picked
//! copy). Every float variable the table maps becomes one GRIB2 message per
//! level, with its 32-bit values as they are:
//!
//! * data representation template 5.4 (IEEE binary32, precision 1), or, only
//!   where it is exact, simple packing 5.0 of integer-valued planes (a flag, a
//!   category, an all-zero level) with binary and decimal scale 0. Every
//!   message is decoded again after it is written and compared with its source
//!   bit for bit; a plane that does not come back identical is written 5.4, and
//!   a 5.4 plane that does not is an error. Nothing is quantized.
//! * grids: every map projection a forecast file comes on. Lambert conformal
//!   (WRF MAP_PROJ 1) as template 3.30, polar stereographic (2) as 3.20, Mercator
//!   (3) as 3.10, a regular latitude-longitude grid (6, unrotated) as 3.0;
//!   staggered U and V on their own grids (XLAT_U/XLONG_U, XLAT_V/XLONG_V, or the
//!   half-cell shift of a lat-lon grid). Any other grid is refused with its
//!   reason.
//! * the table (schema woof.grib2-parameters/v1) is data: each row names a file
//!   variable, its discipline/category/number, its level and, for an
//!   accumulation or a window maximum, its statistical process. A variable the
//!   table does not map, or maps to a message another variable of this file
//!   already wrote, is listed in the receipt with the reason.
//!
//! Prints one JSON line (schema woof.history-grib2.receipt/v1). Exit 0 written,
//! 2 refused (the message says why), 1 failed.

use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use chrono::NaiveDateTime;
use rayon::prelude::*;
use serde::Deserialize;
use serde_json::{json, Value};
use wx_core::grib2::{
    unpack_message, Grib2File, Grib2Writer, GridDefinition, MessageBuilder, PackingMethod,
    ProductDefinition, StatisticalInterval,
};

const TABLE_SCHEMA: &str = "woof.grib2-parameters/v1";
const RECEIPT_SCHEMA: &str = "woof.history-grib2.receipt/v1";
const ABI: &str = "rw_history_grib2 --table TABLE.json IN OUT [--window-seconds N] [--threads N] \
table=woof.grib2-parameters/v1 receipt=woof.history-grib2.receipt/v1 packing=ieee32,exact-integer \
verify=every-message grids=3.30,3.20,3.10,3.0";
/// Largest integer every binary32 decoder reproduces exactly.
const EXACT_INT: f64 = 16_777_216.0;

#[derive(Deserialize)]
struct Table {
    schema: String,
    message: MessageMeta,
    #[serde(default)]
    coordinates: Vec<String>,
    parameters: Vec<Param>,
    #[serde(default, rename = "netcdfOnly")]
    netcdf_only: HashMap<String, String>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct MessageMeta {
    centre: u16,
    subcentre: u16,
    master_tables: u8,
    local_tables: u8,
    generating_process_id: u8,
    earth_radius_metres: f64,
}

#[derive(Deserialize, Clone)]
struct Param {
    var: String,
    param: [u8; 3],
    level: String,
    #[serde(default)]
    stat: Option<String>,
    #[serde(default)]
    window: Option<String>,
    #[serde(default)]
    wind: bool,
}

#[derive(Debug)]
enum Fail {
    Refused(String),
    Failed(String),
}

impl Fail {
    fn message(&self) -> &str {
        match self {
            Fail::Refused(m) | Fail::Failed(m) => m,
        }
    }
}

fn refuse<T>(m: impl Into<String>) -> Result<T, Fail> {
    Err(Fail::Refused(m.into()))
}

fn failed(m: impl std::fmt::Display) -> Fail {
    Fail::Failed(m.to_string())
}

/// The horizontal geometry of the file's grid.
enum Geometry {
    /// `grid_winds`: the file's U/V components are relative to the grid's x and y (WRF's own), not east/north.
    Lambert { truelat1: f64, truelat2: f64, stand_lon: f64, dx: f64, dy: f64, grid_winds: bool },
    /// Scale true at TRUELAT1, the pole of TRUELAT1's hemisphere at the centre; winds as Lambert's.
    Polar { truelat1: f64, stand_lon: f64, dx: f64, dy: f64, grid_winds: bool },
    /// Scale true at TRUELAT1. The grid's x and y run east and north, so its winds are earth-relative as they stand.
    Mercator { truelat1: f64, dx: f64, dy: f64 },
    LatLon { lat: Vec<f64>, lon: Vec<f64>, dlat: f64, dlon: f64 },
}

/// Level of one message: (type, value), optional second surface.
#[derive(Clone, Copy, PartialEq)]
struct Level {
    kind: u8,
    value: f64,
    second: Option<(u8, f64)>,
}

/// One variable's work: everything but its values.
struct Plan {
    var: String,
    param: [u8; 3],
    grid: GridDefinition,
    flags: u8,
    levels: Vec<Level>,
    /// (forecast start, length) in `unit`, and the process, for template 4.8.
    interval: Option<(u32, u32, u8)>,
    lead: u32,
    unit: u8,
    fill: Option<u32>,
    level_label: String,
}

struct Encoded {
    bytes: Vec<u8>,
    messages: usize,
    ieee: usize,
    exact: usize,
    missing: usize,
}

fn attr_f64(file: &netcrust::File, name: &str) -> Option<f64> {
    file.attribute(name).and_then(|a| a.as_f64())
}

fn attr_str(file: &netcrust::File, name: &str) -> Option<String> {
    file.attribute(name).and_then(|a| a.as_string().map(|s| s.trim().trim_end_matches('\0').to_string()))
}

fn wrf_time(s: &str) -> Option<NaiveDateTime> {
    let t = s.trim().trim_end_matches('\0');
    NaiveDateTime::parse_from_str(t, "%Y-%m-%d_%H:%M:%S")
        .or_else(|_| NaiveDateTime::parse_from_str(t, "%Y-%m-%dT%H:%M:%S"))
        .ok()
}

fn read_f32(file: &netcrust::File, name: &str) -> Result<Vec<f32>, Fail> {
    match file.read_array::<f32>(name) {
        Ok(array) => Ok(match array.as_slice() {
            Some(s) => s.to_vec(),
            None => array.iter().copied().collect(),
        }),
        // A NetCDF-4 copy whose dimensions netcrust resolves itself (a picked file) refuses the typed read; its
        // promoted read widens each binary32 word to binary64, which narrows back to the same 32 bits (every
        // message is still compared bit for bit after it is written).
        Err(_) => {
            let wide = file.read_array_f64(name).map_err(|e| failed(format!("{name}: {e}")))?;
            Ok(wide.values().iter().map(|&v| v as f32).collect())
        }
    }
}

fn read_coordinate(file: &netcrust::File, name: &str) -> Option<Vec<f64>> {
    file.read_f64(name).ok()
}

/// The file's grid, or the reason it cannot be a GRIB2 grid.
fn geometry(file: &netcrust::File) -> Result<Geometry, Fail> {
    let proj = attr_f64(file, "MAP_PROJ").map(|v| v.round() as i64);
    match proj {
        Some(1) => {
            let need = |n: &str| attr_f64(file, n).ok_or_else(|| Fail::Refused(format!("the Lambert conformal file has no {n} attribute")));
            Ok(Geometry::Lambert {
                truelat1: need("TRUELAT1")?,
                truelat2: need("TRUELAT2")?,
                stand_lon: need("STAND_LON")?,
                dx: need("DX")?,
                dy: need("DY")?,
                grid_winds: grid_relative_winds(file),
            })
        }
        Some(2) => {
            let need = |n: &str| attr_f64(file, n).ok_or_else(|| Fail::Refused(format!("the polar stereographic file has no {n} attribute")));
            Ok(Geometry::Polar {
                truelat1: need("TRUELAT1")?,
                stand_lon: need("STAND_LON")?,
                dx: need("DX")?,
                dy: need("DY")?,
                grid_winds: grid_relative_winds(file),
            })
        }
        Some(3) => {
            let need = |n: &str| attr_f64(file, n).ok_or_else(|| Fail::Refused(format!("the Mercator file has no {n} attribute")));
            Ok(Geometry::Mercator { truelat1: need("TRUELAT1")?, dx: need("DX")?, dy: need("DY")? })
        }
        Some(6) => {
            let pole_lat = attr_f64(file, "POLE_LAT").unwrap_or(90.0);
            let pole_lon = attr_f64(file, "POLE_LON").unwrap_or(0.0);
            if (pole_lat - 90.0).abs() > 1e-6 || pole_lon.abs() > 1e-6 {
                return refuse("a rotated latitude-longitude grid (POLE_LAT/POLE_LON off the pole) is not written as GRIB2 here; its forecast files stay NetCDF");
            }
            let lat = read_coordinate(file, "lat")
                .or_else(|| {
                    let ny = file.dimension("south_north")?.len();
                    let nx = file.dimension("west_east")?.len();
                    let xlat = file.read_f64("XLAT").ok()?;
                    (xlat.len() >= ny * nx).then(|| (0..ny).map(|j| xlat[j * nx]).collect())
                })
                .ok_or_else(|| Fail::Refused("the latitude-longitude file has neither lat nor XLAT".into()))?;
            let lon = read_coordinate(file, "lon")
                .or_else(|| {
                    let nx = file.dimension("west_east")?.len();
                    let xlon = file.read_f64("XLONG").ok()?;
                    (xlon.len() >= nx).then(|| xlon[..nx].to_vec())
                })
                .ok_or_else(|| Fail::Refused("the latitude-longitude file has neither lon nor XLONG".into()))?;
            let dlat = regular_step(&lat, "latitude")?;
            // Longitudes may wrap (a ring crossing the date line): unwrap before checking the spacing.
            let mut unwrapped = lon.clone();
            for i in 1..unwrapped.len() {
                while unwrapped[i] - unwrapped[i - 1] > 180.0 { unwrapped[i] -= 360.0; }
                while unwrapped[i] - unwrapped[i - 1] < -180.0 { unwrapped[i] += 360.0; }
            }
            let dlon = regular_step(&unwrapped, "longitude")?;
            if lat.iter().any(|v| !v.is_finite() || v.abs() > 90.0 + 1e-9) {
                return refuse("a latitude beyond the poles");
            }
            Ok(Geometry::LatLon { lat, lon: unwrapped, dlat, dlon })
        }
        Some(p) => refuse(format!("map projection {p} has no GRIB2 grid here; its forecast files stay NetCDF")),
        None => refuse("the file names no map projection (MAP_PROJ), so no GRIB2 grid can place its values"),
    }
}

/// Whether a projected file's U/V are relative to its grid. WRF's are, with the map's rotation in SINALPHA/COSALPHA. A
/// file that declares no rotation (SINALPHA all 0, COSALPHA all 1: a hex door's Lambert core, whose MPAS_WIND_FRAME says
/// its winds are earth-relative) holds east/north components, and is labelled so.
fn grid_relative_winds(file: &netcrust::File) -> bool {
    let declared_earth = attr_str(file, "MPAS_WIND_FRAME").map(|t| t.contains("earth-relative")).unwrap_or(false);
    let no_rotation = match (file.read_f64("SINALPHA"), file.read_f64("COSALPHA")) {
        (Ok(s), Ok(c)) => !s.is_empty() && s.iter().all(|v| *v == 0.0) && c.iter().all(|v| *v == 1.0),
        _ => false,
    };
    !(declared_earth || no_rotation)
}

fn regular_step(v: &[f64], what: &str) -> Result<f64, Fail> {
    if v.len() < 2 {
        return refuse(format!("a {what} axis with fewer than two points"));
    }
    let step = (v[v.len() - 1] - v[0]) / (v.len() - 1) as f64;
    if step == 0.0 || !step.is_finite() {
        return refuse(format!("a degenerate {what} axis"));
    }
    let worst = v.iter().enumerate().map(|(i, x)| (x - (v[0] + i as f64 * step)).abs()).fold(0.0, f64::max);
    // GRIB2 template 3.0 stores positions and increments in micro-degrees.
    if worst > 1e-5 {
        return refuse(format!("the {what} axis is not evenly spaced (off by up to {worst:.6} degrees), so template 3.0 cannot place it"));
    }
    Ok(step)
}

/// The names of the coordinates of a projected grid's mass, U or V points.
fn projected_coordinates(xs: bool, ys: bool) -> Result<(&'static str, &'static str), Fail> {
    match (xs, ys) {
        (false, false) => Ok(("XLAT", "XLONG")),
        (true, false) => Ok(("XLAT_U", "XLONG_U")),
        (false, true) => Ok(("XLAT_V", "XLONG_V")),
        (true, true) => refuse("a variable staggered in both x and y has no WRF grid"),
    }
}

/// A coordinate's first and last values: a projected grid's first and last points, as the file itself places them.
fn coordinate_ends(file: &netcrust::File, name: &str) -> Result<(f64, f64), Fail> {
    let v = file.read_f64(name).map_err(|_| Fail::Refused(format!("the file carries no {name} to place this grid's first point")))?;
    match (v.first().copied(), v.last().copied()) {
        (Some(a), Some(b)) if a.is_finite() && b.is_finite() => Ok((a, b)),
        _ => refuse(format!("{name} is empty")),
    }
}

/// The GRIB2 grid of a variable on (x dimension, y dimension), and its section 3 flags.
fn grid_for(file: &netcrust::File, geo: &Geometry, xdim: &str, ydim: &str, nx: usize, ny: usize, wind: bool)
    -> Result<(GridDefinition, u8), Fail> {
    let xs = xdim.ends_with("_stag");
    let ys = ydim.ends_with("_stag");
    match geo {
        Geometry::Lambert { truelat1, truelat2, stand_lon, dx, dy, grid_winds } => {
            let (latn, lonn) = projected_coordinates(xs, ys)?;
            let grid = GridDefinition {
                template: 30,
                nx: nx as u32,
                ny: ny as u32,
                lat1: coordinate_ends(file, latn)?.0,
                lon1: coordinate_ends(file, lonn)?.0,
                lad: *truelat1,
                lov: *stand_lon,
                dx: *dx,
                dy: *dy,
                latin1: *truelat1,
                latin2: *truelat2,
                projection_center_flag: if *truelat1 < 0.0 { 0x80 } else { 0 },
                scan_mode: 0x40,
                ..Default::default()
            };
            // Flag table 3.3 bit 5: WRF's winds are relative to its grid's x and y.
            Ok((grid, if wind && *grid_winds { 0x38 } else { 0x30 }))
        }
        Geometry::Polar { truelat1, stand_lon, dx, dy, grid_winds } => {
            let (latn, lonn) = projected_coordinates(xs, ys)?;
            let grid = GridDefinition {
                template: 20,
                nx: nx as u32,
                ny: ny as u32,
                lat1: coordinate_ends(file, latn)?.0,
                lon1: coordinate_ends(file, lonn)?.0,
                lad: *truelat1,
                lov: *stand_lon,
                dx: *dx,
                dy: *dy,
                projection_center_flag: if *truelat1 < 0.0 { 0x80 } else { 0 },
                scan_mode: 0x40,
                ..Default::default()
            };
            Ok((grid, if wind && *grid_winds { 0x38 } else { 0x30 }))
        }
        Geometry::Mercator { truelat1, dx, dy } => {
            let (latn, lonn) = projected_coordinates(xs, ys)?;
            let (lat1, lat2) = coordinate_ends(file, latn)?;
            let (lon1, lon2) = coordinate_ends(file, lonn)?;
            let grid = GridDefinition {
                template: 10,
                nx: nx as u32,
                ny: ny as u32,
                lat1,
                lon1,
                lat2,
                lon2,
                lad: *truelat1,
                dx: *dx,
                dy: *dy,
                scan_mode: 0x40,
                ..Default::default()
            };
            Ok((grid, 0x30))
        }
        Geometry::LatLon { lat, lon, dlat, dlon } => {
            let lat0 = if ys { lat[0] - dlat / 2.0 } else { lat[0] };
            let lon0 = if xs { lon[0] - dlon / 2.0 } else { lon[0] };
            let lat_last = lat0 + (ny as f64 - 1.0) * dlat;
            let lon_last = lon0 + (nx as f64 - 1.0) * dlon;
            if lat0.abs() > 90.0 + 1e-9 || lat_last.abs() > 90.0 + 1e-9 {
                return refuse("its staggered rows reach beyond a pole, where no latitude-longitude grid can place them");
            }
            let mut scan = 0u8;
            if *dlon < 0.0 { scan |= 0x80; }
            if *dlat > 0.0 { scan |= 0x40; }
            let grid = GridDefinition {
                template: 0,
                nx: nx as u32,
                ny: ny as u32,
                lat1: lat0,
                lon1: lon0,
                lat2: lat_last,
                lon2: lon_last,
                dx: dlon.abs(),
                dy: dlat.abs(),
                scan_mode: scan,
                ..Default::default()
            };
            Ok((grid, 0x30))
        }
    }
}

/// (unit code, seconds per unit) that counts every value in `secs` exactly.
fn unit_for(secs: &[i64]) -> (u8, i64) {
    if secs.iter().all(|s| s % 3600 == 0) {
        (1, 3600)
    } else if secs.iter().all(|s| s % 60 == 0) {
        (0, 60)
    } else {
        (13, 1)
    }
}

fn statistical_process(stat: &str) -> Option<u8> {
    match stat {
        "avg" => Some(0),
        "accum" => Some(1),
        "max" => Some(2),
        "min" => Some(3),
        _ => None,
    }
}

/// Parse a 2-D level spec ("surface", "height:2", "top", "column", "agl-layer:5000:2000", "model:1").
fn surface_level(spec: &str) -> Option<Level> {
    let parts: Vec<&str> = spec.split(':').collect();
    let num = |i: usize| parts.get(i).and_then(|p| p.parse::<f64>().ok());
    Some(match parts[0] {
        "surface" => Level { kind: 1, value: 0.0, second: None },
        "height" => Level { kind: 103, value: num(1)?, second: None },
        "top" => Level { kind: 8, value: 0.0, second: None },
        "column" => Level { kind: 10, value: 0.0, second: None },
        "agl-layer" => Level { kind: 103, value: num(1)?, second: Some((103, num(2)?)) },
        "model" if parts.len() == 2 => Level { kind: 105, value: num(1)?, second: None },
        _ => return None,
    })
}

struct Context {
    lead_seconds: i64,
    window_seconds: Option<i64>,
    pressure_levels: Option<Vec<f64>>,
}

fn plan_for(file: &netcrust::File, geo: &Geometry, row: &Param, var: &netcrust::Variable, ctx: &Context) -> Result<Plan, Fail> {
    if *var.dtype() != netcrust::DataType::F32 {
        return refuse(format!("stored as {:?}, and only 32-bit float variables are written losslessly as GRIB2 here", var.dtype()));
    }
    let mut dims: Vec<(String, usize)> = var.dimensions().iter().map(|d| (d.name().to_string(), d.len())).collect();
    if dims.first().map(|d| d.0.as_str()) == Some("Time") {
        if dims[0].1 != 1 {
            return refuse(format!("the file holds {} times; one GRIB2 file is written per forecast time", dims[0].1));
        }
        dims.remove(0);
    }
    if dims.len() < 2 {
        return refuse("not a horizontal grid field");
    }
    let (xdim, nx) = dims[dims.len() - 1].clone();
    let (ydim, ny) = dims[dims.len() - 2].clone();
    if !xdim.starts_with("west_east") || !ydim.starts_with("south_north") {
        return refuse(format!("its last dimensions ({ydim}, {xdim}) are not the forecast grid's"));
    }
    let levels: Vec<Level> = match dims.len() {
        2 => {
            if row.level == "model" {
                return refuse("the table places it on model levels, and this file stores it as one plane");
            }
            vec![surface_level(&row.level).ok_or_else(|| failed(format!("{}: unknown level spec {:?}", row.var, row.level)))?]
        }
        3 => {
            if row.level != "model" {
                return refuse(format!("the table places it at {:?}, and this file stores it on {} levels", row.level, dims[0].1));
            }
            let (zdim, nz) = &dims[0];
            match zdim.as_str() {
                "bottom_top" | "bottom_top_stag" => (0..*nz).map(|k| Level { kind: 105, value: (k + 1) as f64, second: None }).collect(),
                "soil_layers_stag" | "soil_layers" => (0..*nz).map(|k| Level { kind: 151, value: (k + 1) as f64, second: None }).collect(),
                "num_press_levels_stag" => {
                    let p = ctx.pressure_levels.as_ref().ok_or_else(|| Fail::Refused("a pressure-level variable in a file without P_PL (its levels)".into()))?;
                    if p.len() != *nz {
                        return refuse("P_PL does not match its pressure-level dimension");
                    }
                    p.iter().map(|&v| Level { kind: 100, value: v, second: None }).collect()
                }
                other => return refuse(format!("its vertical dimension {other} has no GRIB2 level here")),
            }
        }
        _ => return refuse("more than one vertical dimension"),
    };
    let (grid, flags) = grid_for(file, geo, &xdim, &ydim, nx, ny, row.wind)?;
    let lead = ctx.lead_seconds;
    let (interval, unit, div) = match row.stat.as_deref() {
        None => {
            let (unit, div) = unit_for(&[lead]);
            (None, unit, div)
        }
        Some(stat) => {
            let process = statistical_process(stat).ok_or_else(|| failed(format!("{}: unknown stat {stat:?}", row.var)))?;
            let start = match row.window.as_deref().unwrap_or("run") {
                "run" => 0,
                "output" => match ctx.window_seconds {
                    Some(w) => (lead - w).max(0),
                    None => return refuse("a window maximum, and the window length (the run's output interval) was not given"),
                },
                other => return Err(failed(format!("{}: unknown window {other:?}", row.var))),
            };
            let (unit, div) = unit_for(&[start, lead - start]);
            (Some(((start / div) as u32, ((lead - start) / div) as u32, process)), unit, div)
        }
    };
    let fill = var
        .attribute("_FillValue")
        .or_else(|| var.attribute("missing_value"))
        .and_then(|a| a.as_f64())
        .map(|v| (v as f32).to_bits());
    let level_label = match levels.first() {
        Some(l) if levels.len() > 1 => format!("{}:{}..{}", l.kind, l.value, levels[levels.len() - 1].value),
        Some(l) => match l.second {
            Some((k, v)) => format!("{}:{}-{}:{}", l.kind, l.value, k, v),
            None => format!("{}:{}", l.kind, l.value),
        },
        None => String::new(),
    };
    Ok(Plan {
        var: row.var.clone(),
        param: row.param,
        grid,
        flags,
        levels,
        interval,
        lead: (lead / div) as u32,
        unit,
        fill,
        level_label,
    })
}

/// Encode one plane, decode it again and compare every value's bits.
fn encode_plane(meta: &MessageMeta, plan: &Plan, level: &Level, reference: NaiveDateTime, plane: &[f32])
    -> Result<(Vec<u8>, bool, usize), Fail> {
    let present: Vec<bool> = plane.iter().map(|v| v.is_finite() && Some(v.to_bits()) != plan.fill).collect();
    let missing = present.iter().filter(|p| !**p).count();
    let values: Vec<f64> = plane.iter().map(|&v| v as f64).collect();
    // Exact simple packing: integer-valued, below 2^24, no negative zero (a decoder adds 0 and loses its sign).
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    let mut integral = true;
    for (v, p) in plane.iter().zip(present.iter()) {
        if !*p { continue; }
        let x = *v as f64;
        if x.fract() != 0.0 || x.abs() >= EXACT_INT || v.to_bits() == 0x8000_0000 { integral = false; break; }
        lo = lo.min(x);
        hi = hi.max(x);
    }
    let exact_bits = if integral && lo.is_finite() {
        let range = hi - lo;
        let bits = if range == 0.0 { 1 } else { (range + 1.0).log2().ceil() as u8 };
        (bits <= 24).then_some(bits.max(1))
    } else {
        None
    };
    let build = |packing: PackingMethod| -> Result<Vec<u8>, Fail> {
        let product = ProductDefinition {
            template: if plan.interval.is_some() { 8 } else { 0 },
            parameter_category: plan.param[1],
            parameter_number: plan.param[2],
            generating_process: 2,
            forecast_time: plan.interval.map(|(start, _, _)| start).unwrap_or(plan.lead),
            time_range_unit: plan.unit,
            level_type: level.kind,
            level_value: level.value,
        };
        let mut msg = MessageBuilder::new(plan.param[0], values.clone())
            .center(meta.centre, meta.subcentre)
            .master_table_version(meta.master_tables)
            .local_table_version(meta.local_tables)
            .generating_process_identifier(meta.generating_process_id)
            .reference_time(reference)
            .grid(plan.grid.clone())
            .product(product)
            .earth_radius(meta.earth_radius_metres)
            .earth_shape(1)
            .resolution_flags(plan.flags)
            .packing(packing);
        if missing > 0 {
            msg = msg.bitmap(present.clone());
        }
        if let Some((k, v)) = level.second {
            msg = msg.second_surface(k, v);
        }
        if let Some((start, length, process)) = plan.interval {
            let secs = |n: u32| -> i64 {
                n as i64 * match plan.unit { 1 => 3600, 0 => 60, _ => 1 }
            };
            let end = reference + chrono::Duration::seconds(secs(start) + secs(length));
            msg = msg.statistical_interval(StatisticalInterval { end_time: end, statistical_process: process, time_unit: plan.unit, length });
        }
        Grib2Writer::new().add_message(msg).to_bytes().map_err(|e| failed(format!("{}: {e}", plan.var)))
    };
    let identical = |bytes: &[u8]| -> bool {
        let Ok(file) = Grib2File::from_bytes(bytes) else { return false };
        let Some(m) = file.messages.first() else { return false };
        let Ok(back) = unpack_message(m) else { return false };
        back.len() == plane.len()
            && back.iter().zip(plane.iter()).zip(present.iter()).all(|((b, v), p)| {
                if *p { (*b as f32).to_bits() == v.to_bits() } else { b.is_nan() }
            })
    };
    if let Some(bits) = exact_bits {
        let bytes = build(PackingMethod::Simple { bits_per_value: bits })?;
        if identical(&bytes) {
            return Ok((bytes, true, missing));
        }
    }
    let bytes = build(PackingMethod::Ieee32)?;
    if !identical(&bytes) {
        return Err(failed(format!("{} level {}: the IEEE message did not decode bit for bit", plan.var, level.value)));
    }
    Ok((bytes, false, missing))
}

fn encode_variable(meta: &MessageMeta, plan: &Plan, reference: NaiveDateTime, file: &netcrust::File) -> Result<Encoded, Fail> {
    let plane = plan.grid.nx as usize * plan.grid.ny as usize;
    // Whole-variable reads: WRF's NetCDF-4 chunks span a variable's levels, so a per-level read would inflate each
    // chunk once per level (measured 2026-10-31: 3.5 s -> 6.1 s on a 3 km hex core frame). Variables are read in
    // parallel instead (run), and each variable's levels are encoded and verified in parallel here.
    let values = read_f32(file, &plan.var)?;
    if values.len() != plane * plan.levels.len() {
        return Err(failed(format!("{}: {} values, expected {}", plan.var, values.len(), plane * plan.levels.len())));
    }
    let parts: Vec<(Vec<u8>, bool, usize)> = plan
        .levels
        .par_iter()
        .enumerate()
        .map(|(k, level)| encode_plane(meta, plan, level, reference, &values[k * plane..(k + 1) * plane]))
        .collect::<Result<_, _>>()?;
    let mut out = Encoded { bytes: Vec::new(), messages: parts.len(), ieee: 0, exact: 0, missing: 0 };
    for (bytes, exact, missing) in parts {
        out.bytes.extend_from_slice(&bytes);
        if exact { out.exact += 1 } else { out.ieee += 1 }
        out.missing += missing;
    }
    Ok(out)
}

fn run(table_path: &Path, input: &Path, output: &Path, window_seconds: Option<i64>) -> Result<Value, Fail> {
    let started = Instant::now();
    let table: Table = serde_json::from_slice(&std::fs::read(table_path).map_err(|e| failed(format!("table: {e}")))?)
        .map_err(|e| failed(format!("table: {e}")))?;
    if table.schema != TABLE_SCHEMA {
        return Err(failed(format!("table schema {:?}, expected {TABLE_SCHEMA}", table.schema)));
    }
    let file = netcrust::File::open(input).map_err(|e| failed(format!("{}: {e}", input.display())))?;
    let geo = geometry(&file)?;
    let reference = attr_str(&file, "SIMULATION_START_DATE")
        .or_else(|| attr_str(&file, "START_DATE"))
        .and_then(|s| wrf_time(&s))
        .ok_or_else(|| Fail::Refused("the file names no start time (SIMULATION_START_DATE)".into()))?;
    let times = file.read_strings("Times").map_err(|_| Fail::Refused("the file has no Times".into()))?;
    if times.len() != 1 {
        return refuse(format!("the file holds {} times; one GRIB2 file is written per forecast time", times.len()));
    }
    let valid = wrf_time(&times[0]).ok_or_else(|| Fail::Refused(format!("Times {:?} is not a WRF time", times[0])))?;
    let lead_seconds = (valid - reference).num_seconds();
    if lead_seconds < 0 {
        return refuse("the frame's time is before its start time");
    }
    let pressure_levels = file.read_f64("P_PL").ok();
    let ctx = Context { lead_seconds, window_seconds, pressure_levels };
    let variables = file.variables().map_err(failed)?;
    let by_name: HashMap<String, &netcrust::Variable> = variables.iter().map(|v| (v.name().to_string(), v)).collect();
    let coords: HashSet<&str> = table.coordinates.iter().map(String::as_str).collect();

    let mut plans: Vec<Plan> = Vec::new();
    let mut omitted: Vec<Value> = Vec::new();
    let mut claimed: HashMap<String, String> = HashMap::new();
    let mut mapped: HashSet<&str> = HashSet::new();
    for row in &table.parameters {
        let Some(var) = by_name.get(&row.var) else { continue };
        if !mapped.insert(row.var.as_str()) {
            continue;
        }
        match plan_for(&file, &geo, row, var, &ctx) {
            Ok(plan) => {
                let key = format!("{:?}|{}|{:?}|{}x{}", plan.param, plan.level_label, plan.interval, plan.grid.nx, plan.grid.ny);
                if let Some(first) = claimed.get(&key) {
                    omitted.push(json!({"name": row.var, "reason": format!("the same GRIB2 parameter, level and time as {first}, which this file also carries")}));
                } else {
                    claimed.insert(key, row.var.clone());
                    plans.push(plan);
                }
            }
            Err(Fail::Refused(why)) => omitted.push(json!({"name": row.var, "reason": why})),
            Err(e) => return Err(e),
        }
    }
    let mut coordinates = Vec::new();
    for v in &variables {
        let name = v.name();
        if mapped.contains(name) {
            continue;
        }
        if coords.contains(name) {
            coordinates.push(name.to_string());
        } else {
            let reason = table.netcdf_only.get(name).cloned().unwrap_or_else(|| "not in the GRIB2 parameter table".into());
            omitted.push(json!({"name": name, "reason": reason}));
        }
    }
    drop(file);

    let tmp = output.with_extension("grib2.part");
    let mut sink = std::io::BufWriter::with_capacity(8 << 20, std::fs::File::create(&tmp).map_err(|e| failed(format!("{}: {e}", tmp.display())))?);
    let threads = rayon::current_num_threads().max(1);
    let mut written = Vec::new();
    let (mut messages, mut ieee, mut exact, mut bytes, mut missing) = (0usize, 0usize, 0usize, 0u64, 0usize);
    for batch in plans.chunks(threads) {
        let encoded: Vec<Encoded> = batch
            .par_iter()
            .map_init(|| netcrust::File::open(input), |f, plan| match f {
                Ok(f) => encode_variable(&table.message, plan, reference, f),
                Err(e) => Err(failed(format!("{}: {e}", input.display()))),
            })
            .collect::<Result<_, _>>()?;
        for (plan, enc) in batch.iter().zip(encoded) {
            sink.write_all(&enc.bytes).map_err(|e| failed(format!("{}: {e}", tmp.display())))?;
            messages += enc.messages;
            ieee += enc.ieee;
            exact += enc.exact;
            missing += enc.missing;
            bytes += enc.bytes.len() as u64;
            written.push(json!({
                "name": plan.var,
                "param": format!("{}.{}.{}", plan.param[0], plan.param[1], plan.param[2]),
                "level": plan.level_label,
                "grid": format!("3.{} {}x{}", plan.grid.template, plan.grid.nx, plan.grid.ny),
                "messages": enc.messages,
                "bytes": enc.bytes.len(),
                "exactInteger": enc.exact,
                "stat": plan.interval.map(|(s, l, p)| json!({"process": p, "start": s, "length": l, "unit": plan.unit})),
            }));
        }
    }
    sink.flush().map_err(|e| failed(format!("{}: {e}", tmp.display())))?;
    drop(sink);
    if messages == 0 {
        let _ = std::fs::remove_file(&tmp);
        return refuse("none of this file's variables has a GRIB2 parameter");
    }
    std::fs::rename(&tmp, output).map_err(|e| failed(format!("{}: {e}", output.display())))?;
    Ok(json!({
        "schema": RECEIPT_SCHEMA,
        "input": input.file_name().map(|n| n.to_string_lossy().to_string()),
        "output": output.file_name().map(|n| n.to_string_lossy().to_string()),
        "bytes": bytes,
        "messages": messages,
        "packing": {"ieee32": ieee, "exactInteger": exact},
        "missingValues": missing,
        "verified": "every message decoded again and compared bit for bit",
        "grid": match geo {
            Geometry::Lambert { .. } => "3.30 Lambert conformal",
            Geometry::Polar { .. } => "3.20 polar stereographic",
            Geometry::Mercator { .. } => "3.10 Mercator",
            Geometry::LatLon { .. } => "3.0 latitude-longitude",
        },
        "winds": match geo {
            Geometry::Lambert { grid_winds: true, .. } | Geometry::Polar { grid_winds: true, .. } => "grid-relative",
            _ => "earth-relative",
        },
        "reference": reference.format("%Y-%m-%dT%H:%M:%SZ").to_string(),
        "valid": valid.format("%Y-%m-%dT%H:%M:%SZ").to_string(),
        "written": written,
        "omitted": omitted,
        "coordinates": coordinates,
        "seconds": (started.elapsed().as_secs_f64() * 1000.0).round() / 1000.0,
        "threads": threads,
    }))
}

const USAGE: &str = "usage: rw_history_grib2 --table TABLE.json IN OUT.grib2 [--window-seconds N] [--threads N]\n       rw_history_grib2 --abi | --version";

pub static GPUWM_BRIDGE_SOURCE_REV_STAMP: &str =
    concat!("GPUWM_BRIDGE_SOURCE_REV=", env!("GPUWM_BRIDGE_SOURCE_REV"));

fn main() -> ExitCode {
    let _ = std::hint::black_box(GPUWM_BRIDGE_SOURCE_REV_STAMP);
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut table: Option<PathBuf> = None;
    let mut window: Option<i64> = None;
    let mut threads: Option<usize> = None;
    let mut positional: Vec<PathBuf> = Vec::new();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--abi" => { println!("{ABI}"); return ExitCode::SUCCESS; }
            "--version" => { println!("rw_history_grib2 {}", env!("CARGO_PKG_VERSION")); return ExitCode::SUCCESS; }
            "--help" | "-h" => { println!("{USAGE}"); return ExitCode::SUCCESS; }
            "--table" => { i += 1; table = args.get(i).map(PathBuf::from); }
            "--window-seconds" => { i += 1; window = args.get(i).and_then(|v| v.parse().ok()); }
            "--threads" => { i += 1; threads = args.get(i).and_then(|v| v.parse().ok()); }
            other if other.starts_with("--") => { eprintln!("rw_history_grib2: unknown argument {other}\n{USAGE}"); return ExitCode::from(2); }
            other => positional.push(PathBuf::from(other)),
        }
        i += 1;
    }
    let (Some(table), [input, output]) = (table, positional.as_slice()) else {
        eprintln!("{USAGE}");
        return ExitCode::from(2);
    };
    if let Some(n) = threads {
        let _ = rayon::ThreadPoolBuilder::new().num_threads(n.max(1)).build_global();
    }
    match run(&table, input, output, window) {
        Ok(receipt) => {
            println!("{receipt}");
            ExitCode::SUCCESS
        }
        Err(e) => {
            let kind = if matches!(e, Fail::Refused(_)) { "refused" } else { "failed" };
            println!("{}", json!({"schema": RECEIPT_SCHEMA, "error": kind, "message": e.message()}));
            eprintln!("rw_history_grib2: {}", e.message());
            ExitCode::from(if kind == "refused" { 2 } else { 1 })
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lossless_planes_preserve_words_and_missing_bitmap() {
        let meta = MessageMeta {
            centre: 7, subcentre: 0, master_tables: 2, local_tables: 0,
            generating_process_id: 1, earth_radius_metres: 6_371_229.0,
        };
        let level = Level { kind: 103, value: 2.0, second: None };
        let fill = -9999.0_f32;
        let plan = Plan {
            var: "T2".into(), param: [0, 0, 0], flags: 0x30,
            grid: GridDefinition {
                nx: 3, ny: 2, lat1: 35.0, lon1: 250.0,
                lat2: 36.0, lon2: 252.0, dx: 1.0, dy: 1.0,
                scan_mode: 0x40, ..Default::default()
            },
            levels: vec![level], interval: None, lead: 1, unit: 1,
            fill: Some(fill.to_bits()), level_label: "height2".into(),
        };
        let reference = NaiveDateTime::parse_from_str(
            "2026-01-01_00:00:00", "%Y-%m-%d_%H:%M:%S").unwrap();
        let cases = [
            ([0.0, 1.0, -2.0, 255.0, fill, f32::NAN], true),
            ([0.0, -0.0, f32::from_bits(0x3f80_0001), 0.125, fill, f32::NAN], false),
        ];
        for (plane, expected_simple) in cases {
            let (bytes, simple, missing) = encode_plane(
                &meta, &plan, &level, reference, &plane).unwrap();
            assert_eq!(simple, expected_simple);
            assert_eq!(missing, 2);
            let file = Grib2File::from_bytes(&bytes).unwrap();
            assert_eq!(file.messages.len(), 1);
            let values = unpack_message(&file.messages[0]).unwrap();
            assert_eq!(values.len(), plane.len());
            for (index, (&value, &original)) in values.iter().zip(&plane).enumerate() {
                if index < 4 {
                    assert_eq!((value as f32).to_bits(), original.to_bits(),
                               "stored word changed at {index}");
                } else {
                    assert!(value.is_nan(), "missing value lost at {index}");
                }
            }
        }
    }
}
