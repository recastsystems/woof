//! Source-neutral export of a composed canonical frame to WPS intermediate v5.
//! This is a consumer of the shared frameset, with no model or hex adapter.
use crate::{
    digest,
    refusal::{Result, frame_invalid, usage},
};
use rayon::prelude::*;
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::{BufWriter, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
};

const SURFACE: f32 = 200100.;
pub(crate) const MISSING: f32 = -1.0e30;

#[derive(Deserialize)]
struct Binding {
    #[serde(default)]
    compositions: Vec<String>,
    field: String,
    wps: String,
    units: String,
    axis: String,
    #[serde(default)]
    transform: String,
    #[serde(default)]
    missing: String,
    #[serde(default)]
    optional: bool,
    /// `"skip"`: a frame that carries no such canonical field writes no
    /// record for it (PMSL and SST: the init reads neither as required;
    /// the projected HRRR door writes neither).  Empty: the field is
    /// required, as before.
    #[serde(default)]
    absent: String,
    /// A frame without this canonical field writes it from another, scaled:
    /// RDPS publishes snow depth and no water equivalent, and SNOW is the
    /// field the init reads (ArWen's own reconciliation of the two, 200 kg
    /// m-3, gpuwm/ingest/soil.py).
    #[serde(default)]
    from: Option<BindingFrom>,
}
#[derive(Deserialize)]
struct BindingFrom {
    field: String,
    units: String,
    scale: f64,
}
#[derive(Deserialize)]
struct Contract {
    schema: String,
    bindings: Vec<Binding>,
    #[serde(default)]
    soil_layers_from_nodes_cm: Vec<[usize; 2]>,
    #[serde(default)]
    map_sources: BTreeMap<String, String>,
}
#[derive(Deserialize)]
struct Field {
    name: String,
    units: String,
    axes: Vec<String>,
    shape: Vec<usize>,
    dtype: String,
    offset: u64,
    length: u64,
    sha256: String,
}
#[derive(Deserialize)]
struct Frame {
    valid_time: String,
    member: Option<String>,
    source_cycle: String,
    #[serde(deserialize_with = "coordinate")]
    latitude: Vec<f64>,
    #[serde(deserialize_with = "coordinate")]
    longitude: Vec<f64>,
    vertical_kind: String,
    vertical_units: String,
    #[serde(deserialize_with = "coordinate")]
    vertical_values: Vec<f64>,
    header: Value,
    /// A windowed frameset's granted atmospheric window
    /// (`window::Window::document`): the listed fields are cropped to
    /// `rows` x `columns` of the source grid; every other field is whole.
    #[serde(default)]
    atmospheric_window: Option<Value>,
    fields: Vec<Field>,
}

// Canonical coordinates bind their JSON values to the original f64 bytes.
// Consume that same descriptor rather than assuming a bare JSON array.
fn coordinate<'de, D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Vec<f64>, D::Error> {
    #[derive(Deserialize)]
    struct Coordinate {
        values: Vec<f64>,
        count: usize,
        sha256: String,
    }
    let c = Coordinate::deserialize(d)?;
    let bytes: Vec<u8> = c.values.iter().flat_map(|v| v.to_le_bytes()).collect();
    if c.count != c.values.len() || digest::bytes_sha256(&bytes) != c.sha256 {
        return Err(serde::de::Error::custom("coordinate count/digest differs"));
    }
    Ok(c.values)
}
#[derive(Deserialize)]
struct Stream {
    path: String,
    dtype: String,
    bytes: u64,
    sha256: String,
}
#[derive(Deserialize)]
struct Frameset {
    schema: String,
    stream: Stream,
    frames: Vec<Frame>,
}

fn io_error(e: impl std::fmt::Display) -> crate::refusal::Refusal {
    frame_invalid(e.to_string())
}
fn read_json(path: &Path) -> Result<Vec<u8>> {
    fs::read(path).map_err(io_error)
}
fn axis(values: &[f64], name: &str) -> Result<f64> {
    if values.len() < 3 || values.iter().any(|v| !v.is_finite()) {
        return Err(frame_invalid(format!(
            "{name} needs >=3 finite coordinates"
        )));
    }
    let d = values[1] - values[0];
    if d == 0. || values.windows(2).any(|w| (w[1] - w[0] - d).abs() > 1.0e-7) {
        return Err(frame_invalid(format!(
            "{name} must be regular and monotone"
        )));
    }
    Ok(d)
}
fn fixed(out: &mut Vec<u8>, value: &str, width: usize) {
    out.extend(value.as_bytes().iter().copied().take(width));
    out.resize(out.len() + width.saturating_sub(value.len()), b' ');
}
fn record(w: &mut impl Write, bytes: &[u8]) -> Result<()> {
    let n = u32::try_from(bytes.len()).map_err(io_error)?.to_be_bytes();
    w.write_all(&n)
        .and_then(|_| w.write_all(bytes))
        .and_then(|_| w.write_all(&n))
        .map_err(io_error)
}
pub(crate) struct Slab {
    pub field: String,
    pub(crate) units: String,
    pub(crate) level: f32,
    pub values: Vec<f32>,
    pub uncovered: Vec<bool>,
}
/// The regular latitude/longitude lattice a WPS record is written on:
/// first point, increments in degrees, and its size.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct LatLonLattice {
    pub south: f64,
    pub west: f64,
    pub dlat: f64,
    pub dlon: f64,
    pub nx: usize,
    pub ny: usize,
}

impl LatLonLattice {
    pub(crate) fn locate_f32(self, lat: f32, lon: f32) -> (f32, f32) {
        let place = |longitude: f32| {
            (
                (longitude - self.west as f32) / self.dlon as f32 + 1.,
                (lat - self.south as f32) / self.dlat as f32 + 1.,
            )
        };
        let (mut x, y) = place(lon);
        if x < 0.5 {
            x = place(lon + 360.).0;
        } else if x >= self.nx as f32 + 0.5 {
            x = place(lon - 360.).0;
        }
        (x, y)
    }
    /// `--target-latlon SOUTH,WEST,DLAT,DLON,NX,NY`.
    pub(crate) fn parse(text: &str) -> Result<Self> {
        let parts: Vec<&str> = text.split(',').map(str::trim).collect();
        if parts.len() != 6 {
            return Err(usage("--target-latlon takes SOUTH,WEST,DLAT,DLON,NX,NY"));
        }
        let number = |index: usize| -> Result<f64> {
            parts[index]
                .parse::<f64>()
                .ok()
                .filter(|v| v.is_finite())
                .ok_or_else(|| {
                    usage(format!(
                        "--target-latlon field {} is not a finite number",
                        index + 1
                    ))
                })
        };
        let count = |index: usize| -> Result<usize> {
            parts[index]
                .parse::<usize>()
                .ok()
                .filter(|v| *v >= 2)
                .ok_or_else(|| {
                    usage(format!(
                        "--target-latlon field {} must be an integer >= 2",
                        index + 1
                    ))
                })
        };
        let lattice = Self {
            south: number(0)?,
            west: number(1)?,
            dlat: number(2)?,
            dlon: number(3)?,
            nx: count(4)?,
            ny: count(5)?,
        };
        if lattice.dlat <= 0. || lattice.dlon <= 0. {
            return Err(usage("--target-latlon increments must be positive"));
        }
        if lattice.south < -90. || lattice.south + lattice.dlat * (lattice.ny - 1) as f64 > 90. {
            return Err(usage("--target-latlon must lie between the poles"));
        }
        Ok(lattice)
    }
}

/// Where a geographic point falls on the frame's WHOLE source grid, as a
/// zero-based fractional `(row, column)`.  The frame header's grid
/// document is the only geometry read, so a frame and its placement
/// cannot disagree.
enum SourceGeometry {
    Regular {
        lat0: f64,
        dlat: f64,
        lon0: f64,
        dlon: f64,
    },
    Rotated {
        frame: crate::rotated::RotatedFrame,
        lat1: f64,
        lon1: f64,
        dlat: f64,
        dlon: f64,
    },
    Lambert(crate::lambert::LambertGrid),
}

impl SourceGeometry {
    fn from_frame(f: &Frame) -> Result<Self> {
        let grid = &f.header["grid"];
        let parameters = &grid["parameters"];
        let number = |key: &str| -> Result<f64> {
            parameters[key]
                .as_f64()
                .filter(|v| v.is_finite())
                .ok_or_else(|| {
                    frame_invalid(format!(
                        "frame grid parameter {key} must be a finite number"
                    ))
                })
        };
        let integer = |key: &str| -> Result<i64> {
            parameters[key].as_i64().ok_or_else(|| {
                frame_invalid(format!("frame grid parameter {key} must be an integer"))
            })
        };
        match grid["projection"].as_str() {
            Some("regular_latitude_longitude") => Ok(Self::Regular {
                lat0: f.latitude[0],
                dlat: axis(&f.latitude, "latitude")?,
                lon0: f.longitude[0],
                dlon: axis(&f.longitude, "longitude")?,
            }),
            Some(crate::model::GRID_FAMILY_ROTATED) => {
                let declared = crate::model::RotatedParameters {
                    lat1: number("lat1")?,
                    lon1: number("lon1")?,
                    dlat_deg: number("dlat_deg")?,
                    dlon_deg: number("dlon_deg")?,
                    nx: integer("nx")?,
                    ny: integer("ny")?,
                    south_pole_lat: number("south_pole_lat")?,
                    south_pole_lon: number("south_pole_lon")?,
                    rotation_angle: number("rotation_angle")?,
                    earth_radius_m: number("earth_radius_m")?,
                    shape_of_earth: integer("shape_of_earth")?,
                };
                Ok(Self::Rotated {
                    frame: crate::rotated::RotatedFrame::from_declaration(&declared),
                    lat1: declared.lat1,
                    lon1: declared.lon1,
                    dlat: declared.dlat_deg,
                    dlon: declared.dlon_deg,
                })
            }
            Some(crate::model::GRID_FAMILY_LAMBERT) => {
                let declared = crate::model::LambertParameters {
                    latin1: number("latin1")?,
                    latin2: number("latin2")?,
                    lov: number("lov")?,
                    lat1: number("lat1")?,
                    lon1: number("lon1")?,
                    dx_m: number("dx_m")?,
                    dy_m: number("dy_m")?,
                    nx: integer("nx")?,
                    ny: integer("ny")?,
                    earth_radius_m: number("earth_radius_m")?,
                    shape_of_earth: integer("shape_of_earth")?,
                };
                Ok(Self::Lambert(
                    crate::lambert::LambertGrid::from_declaration(&declared),
                ))
            }
            other => Err(frame_invalid(format!(
                "wps --target-latlon cannot place a frame whose grid is {other:?}"
            ))),
        }
    }

    /// Zero-based fractional `(row, column)` of a geographic point.
    fn place(&self, latitude: f64, longitude: f64) -> (f64, f64) {
        match self {
            Self::Regular {
                lat0,
                dlat,
                lon0,
                dlon,
            } => {
                // Longitude onto the half-open revolution the axis starts.
                let span = 360.0 * dlon.signum();
                let mut lon = longitude;
                while (lon - lon0) * dlon.signum() < 0.0 {
                    lon += span;
                }
                while (lon - lon0) * dlon.signum() >= 360.0 {
                    lon -= span;
                }
                ((latitude - lat0) / dlat, (lon - lon0) / dlon)
            }
            Self::Rotated {
                frame,
                lat1,
                lon1,
                dlat,
                dlon,
            } => {
                let (rlat, rlon) = frame.from_geographic(latitude, longitude);
                let offset = (rlon - lon1 + 180.0).rem_euclid(360.0) - 180.0;
                ((rlat - lat1) / dlat, offset / dlon)
            }
            Self::Lambert(grid) => {
                let (x, y) = grid.latlon_to_ij(latitude, longitude);
                (y - 1.0, x - 1.0)
            }
        }
    }
}

/// One source plane's donors for every target point: the bilinear cell
/// and weights in the plane's own (possibly windowed) index space.
struct Donors {
    width: usize,
    cells: Vec<(usize, usize, f64, f64)>,
}

impl Donors {
    fn new(
        placed: &[(f64, f64)],
        origin: (usize, usize),
        shape: (usize, usize),
        field: &str,
    ) -> Result<Self> {
        Self::with_support(placed, origin, shape, field, None)
    }

    fn with_support(
        placed: &[(f64, f64)],
        origin: (usize, usize),
        shape: (usize, usize),
        field: &str,
        required: Option<&[bool]>,
    ) -> Result<Self> {
        let (height, width) = shape;
        let mut cells = Vec::with_capacity(placed.len());
        for (node, &(row, column)) in placed.iter().enumerate() {
            let y = row - origin.0 as f64;
            let x = column - origin.1 as f64;
            // The last row/column is a valid donor at weight 1.
            let j = (y.floor().max(0.0) as usize).min(height.saturating_sub(2));
            let i = (x.floor().max(0.0) as usize).min(width.saturating_sub(2));
            if !(y >= -1e-9
                && x >= -1e-9
                && y <= (height - 1) as f64 + 1e-9
                && x <= (width - 1) as f64 + 1e-9)
            {
                if required.is_some_and(|mask| !mask[node]) {
                    cells.push((usize::MAX, 0, 0., 0.));
                    continue;
                }
                return Err(frame_invalid(format!(
                    "the target lattice leaves {field}'s source coverage (a point falls at row {row:.2}, \
                     column {column:.2}; the field holds rows {}..{} and columns {}..{}): choose a region \
                     inside the source's domain",
                    origin.0,
                    origin.0 + height - 1,
                    origin.1,
                    origin.1 + width - 1
                )));
            }
            cells.push((
                j,
                i,
                (y - j as f64).clamp(0.0, 1.0),
                (x - i as f64).clamp(0.0, 1.0),
            ));
        }
        Ok(Self { width, cells })
    }

    /// Bilinear over the finite corners, renormalized; NaN where none is
    /// finite.  With every corner finite it is plain bilinear.
    fn bilinear(&self, plane: &[f64]) -> Vec<f64> {
        let w = self.width;
        self.cells
            .par_iter()
            .map(|&(j, i, fy, fx)| {
                if j == usize::MAX {
                    return f64::NAN;
                }
                let corners = [
                    (plane[j * w + i], (1.0 - fy) * (1.0 - fx)),
                    (plane[j * w + i + 1], (1.0 - fy) * fx),
                    (plane[(j + 1) * w + i], fy * (1.0 - fx)),
                    (plane[(j + 1) * w + i + 1], fy * fx),
                ];
                let (mut sum, mut weight) = (0.0, 0.0);
                for (value, share) in corners {
                    if value.is_finite() && share > 0.0 {
                        sum += value * share;
                        weight += share;
                    }
                }
                if weight > 0.0 {
                    sum / weight
                } else {
                    // A point exactly on a cell edge whose weighted corners
                    // are missing: the nearest finite corner, else missing.
                    corners
                        .iter()
                        .filter(|(value, _)| value.is_finite())
                        .max_by(|a, b| a.1.total_cmp(&b.1))
                        .map_or(f64::NAN, |(value, _)| *value)
                }
            })
            .collect()
    }

    /// The nearest donor's value (masks, fractions that are categories).
    fn nearest(&self, plane: &[f64]) -> Vec<f64> {
        let w = self.width;
        self.cells
            .par_iter()
            .map(|&(j, i, fy, fx)| {
                if j == usize::MAX {
                    return f64::NAN;
                }
                let jj = j + usize::from(fy >= 0.5);
                let ii = i + usize::from(fx >= 0.5);
                plane[jj * w + ii]
            })
            .collect()
    }

    /// Bilinear, then, where every corner is missing, the nearest finite
    /// value within `reach` cells (soil state on a coast whose source
    /// cells are all water): the land/sea record decides which target
    /// points read it, so a value is offered wherever the source has one
    /// near.
    fn bilinear_reaching(&self, plane: &[f64], shape: (usize, usize), reach: usize) -> Vec<f64> {
        let mut values = self.bilinear(plane);
        let (height, width) = shape;
        values
            .par_iter_mut()
            .zip(self.cells.par_iter())
            .filter(|(value, _)| value.is_nan())
            .for_each(|(value, &(j, i, fy, fx))| {
                if j == usize::MAX {
                    return;
                }
                let (cy, cx) = (j as f64 + fy, i as f64 + fx);
                let mut best: Option<(f64, f64)> = None;
                let (j0, j1) = (j.saturating_sub(reach), (j + 1 + reach).min(height - 1));
                let (i0, i1) = (i.saturating_sub(reach), (i + 1 + reach).min(width - 1));
                for jj in j0..=j1 {
                    for ii in i0..=i1 {
                        let candidate = plane[jj * width + ii];
                        if candidate.is_finite() {
                            let distance = (jj as f64 - cy).powi(2) + (ii as f64 - cx).powi(2);
                            if best.map_or(true, |(d, _)| distance < d) {
                                best = Some((distance, candidate));
                            }
                        }
                    }
                }
                if let Some((_, candidate)) = best {
                    *value = candidate;
                }
            });
        values
    }
}

/// How many source cells a coastal soil value may be carried to a
/// target point whose bilinear corners are all water.
const SOIL_REACH_CELLS: usize = 8;

fn write_slab(w: &mut impl Write, s: &Slab, f: &Frame, g: &LatLonLattice, map_source: &str) -> Result<()> {
    record(w, &5i32.to_be_bytes())?;
    let mut h = Vec::new();
    fixed(&mut h, &f.valid_time.replace('T', "_"), 24);
    h.extend(0f32.to_be_bytes());
    fixed(&mut h, map_source, 32);
    fixed(&mut h, &s.field, 9);
    fixed(&mut h, &s.units, 25);
    fixed(&mut h, &format!("canonical {}", s.field), 46);
    h.extend(s.level.to_be_bytes());
    h.extend((g.nx as i32).to_be_bytes());
    h.extend((g.ny as i32).to_be_bytes());
    h.extend(0i32.to_be_bytes());
    record(w, &h)?;
    let mut p = Vec::new();
    fixed(&mut p, "SWCORNER", 8);
    for v in [g.south, g.west, g.dlat, g.dlon, 6371.229] {
        p.extend((v as f32).to_be_bytes());
    }
    record(w, &p)?;
    record(w, &0i32.to_be_bytes())?;
    let mut data = Vec::with_capacity(s.values.len() * 4);
    for v in &s.values {
        data.extend(v.to_be_bytes());
    }
    record(w, &data)
}
fn read_field(stream: &Path, f: &Field) -> Result<Vec<f64>> {
    let count = f
        .shape
        .iter()
        .try_fold(1usize, |a, b| a.checked_mul(*b))
        .ok_or_else(|| frame_invalid("field shape overflow"))?;
    if f.dtype != "<f8" || count.checked_mul(8).map(|v| v as u64) != Some(f.length) {
        return Err(frame_invalid(format!(
            "{} has invalid dtype/length",
            f.name
        )));
    }
    let mut file = File::open(stream).map_err(io_error)?;
    file.seek(SeekFrom::Start(f.offset)).map_err(io_error)?;
    let mut bytes = vec![0; f.length as usize];
    file.read_exact(&mut bytes).map_err(io_error)?;
    let values: Vec<f64> = bytes
        .as_chunks::<8>()
        .0
        .iter()
        .map(|b| f64::from_le_bytes(*b))
        .collect();
    if digest::array_sha256(&f.shape, &values) != f.sha256 {
        return Err(frame_invalid(format!("{} digest differs", f.name)));
    }
    Ok(values)
}

/// `wps --frameset DIR --composition JSON --contract JSON --valid-time ISO --out MET
/// [--target-latlon SOUTH,WEST,DLAT,DLON,NX,NY]`.
///
/// Without a target, global coverage is mandatory: the target reader wraps
/// longitude periodically, and the frame's own lattice is written.  With
/// one, a regional run's box: the frame (regular, rotated or Lambert, and
/// windowed or whole) is resampled onto that regular lattice -- bilinear
/// for continuous fields, nearest for the land mask and sea ice, soil
/// carried a few cells to coasts -- and the box is written.  The frame's
/// winds are already in the earth basis, so none is rotated here.
pub fn run(arguments: &[String]) -> Result<Value> {
    if !arguments.len().is_multiple_of(2) {
        return Err(usage(
            "wps takes --frameset DIR --composition JSON --contract JSON --valid-time ISO --out MET [--target-latlon SOUTH,WEST,DLAT,DLON,NX,NY]",
        ));
    }
    let mut args = BTreeMap::new();
    for pair in arguments.as_chunks::<2>().0 {
        if ![
            "--frameset",
            "--composition",
            "--contract",
            "--valid-time",
            "--out",
            "--member",
            "--coverage",
            "--target-area",
            "--longitude-origin",
            "--target-latlon",
            "--target-support",
            "--target-support-sha256",
            "--soil-set",
        ]
        .contains(&pair[0].as_str())
            || args.insert(pair[0].as_str(), pair[1].as_str()).is_some()
        {
            return Err(usage(format!("unknown/duplicate wps argument {}", pair[0])));
        }
    }
    let need = |key: &str| {
        args.get(key)
            .copied()
            .ok_or_else(|| usage(format!("wps requires {key}")))
    };
    let target = args
        .get("--target-latlon")
        .map(|text| LatLonLattice::parse(text))
        .transpose()?;
    let support = match (
        args.get("--target-support"),
        args.get("--target-support-sha256"),
    ) {
        (None, None) => None,
        (Some(path), Some(hash)) => Some(crate::wps_support::MeshSupport::load(
            path,
            hash,
            target.ok_or_else(|| usage("target support requires --target-latlon"))?,
        )?),
        _ => return Err(usage("target support and its sha256 are an atomic pair")),
    };
    let dir = Path::new(need("--frameset")?);
    let out = PathBuf::from(need("--out")?);
    let manifest = read_json(&dir.join("frames.json"))?;
    let set: Frameset = serde_json::from_slice(&manifest).map_err(io_error)?;
    let windowed_admitted = target.is_some() && set.schema == crate::window::FRAMESET_SCHEMA;
    if (set.schema != crate::FRAMESET_SCHEMA && !windowed_admitted)
        || set.stream.dtype != "<f8"
        || set.stream.path != "frames.f64"
    {
        return Err(frame_invalid(
            "wps requires a canonical non-windowed frameset (a windowed one only with --target-latlon)",
        ));
    }
    let stream = dir.join(&set.stream.path);
    if fs::metadata(&stream).map_err(io_error)?.len() != set.stream.bytes
        || digest::file_sha256(
            stream
                .to_str()
                .ok_or_else(|| frame_invalid("non-UTF8 frameset path"))?,
        )
        .map_err(io_error)?
            != set.stream.sha256
    {
        return Err(frame_invalid("frameset stream size/digest differs"));
    }
    let selected: Vec<_> = set
        .frames
        .iter()
        .filter(|f| {
            f.valid_time == need("--valid-time").unwrap_or_default()
                && f.member.as_deref() == args.get("--member").copied()
        })
        .collect();
    if selected.len() != 1 {
        return Err(frame_invalid(format!(
            "wps requires one matching valid-time/member, found {}",
            selected.len()
        )));
    }
    let f = selected[0];
    chrono::NaiveDateTime::parse_from_str(&f.valid_time, "%Y-%m-%dT%H:%M:%S").map_err(io_error)?;
    let dy = axis(&f.latitude, "latitude")?;
    let dx = axis(&f.longitude, "longitude")?;
    let coverage = args.get("--coverage").copied().unwrap_or("global");
    if !["global", "regional"].contains(&coverage) {
        return Err(usage("coverage must be global or regional"));
    }
    if target.is_some() && (args.contains_key("--target-area") || args.contains_key("--longitude-origin")) {
        return Err(usage("source-cell crop/permutation and target-latlon are separate exports"));
    }
    let is_global = dx > 0. && (dx * f.longitude.len() as f64 - 360.).abs() <= 1.0e-6
        && f.latitude.iter().copied().fold(f64::INFINITY,f64::min) <= -90.+dy.abs()
        && f.latitude.iter().copied().fold(f64::NEG_INFINITY,f64::max) >= 90.-dy.abs();
    if target.is_none() && coverage == "global"
        && (dx <= 0.
            || (dx * f.longitude.len() as f64 - 360.).abs() > 1.0e-6
            || f.latitude.iter().copied().fold(f64::INFINITY, f64::min) > -90. + dy.abs()
            || f.latitude.iter().copied().fold(f64::NEG_INFINITY, f64::max) < 90. - dy.abs())
    {
        return Err(frame_invalid(
            "wps global init requires uncropped longitude (one 360-degree period, no duplicate seam) and pole-to-pole latitude",
        ));
    }
    if (target.is_none() && f.header["grid"]["projection"] != "regular_latitude_longitude")
        || f.header["grid"]["wind_basis"] != "earth_relative"
        || f.vertical_kind != "pressure"
        || !["hPa", "Pa"].contains(&f.vertical_units.as_str())
    {
        return Err(frame_invalid(
            "wps requires regular lat-lon (or --target-latlon), earth-relative winds and pressure levels",
        ));
    }
    // The lattice the records are written on: the frame's own (global), or
    // the target box with every field resampled onto it.
    // Select a rectangular target plus two source cells of interpolation halo.
    // Global source longitudes may cross their stored seam; unwrap and order
    // them around the requested west edge. The output is explicitly regional.
    let mut rows: Vec<usize> = (0..f.latitude.len()).collect();
    let mut columns: Vec<(usize,f64)> = f.longitude.iter().copied().enumerate().collect();
    // --longitude-origin zero (TC exports, 7ddbe2660): an exact cyclic column permutation of a complete global period
    // so the first column is 0 E. Breakage it prevents: a canonical frame stored from -180 E left every western-
    // hemisphere cell outside the native surface-wind locator (hex Milton init: 7,437 of 7,437 core cells without a
    // finite 10 m wind). No value is interpolated or changed.
    match args.get("--longitude-origin").copied().unwrap_or("source") {
        "source" => {}
        "zero" => {
            if args.contains_key("--target-area") {
                return Err(usage("--longitude-origin zero applies to a complete global export, not --target-area"));
            }
            let dxs = f.longitude[1] - f.longitude[0];
            if !(dxs > 0.) || (dxs * f.longitude.len() as f64 - 360.).abs() > 1.0e-5 {
                return Err(frame_invalid("zero longitude origin requires one regular complete global period"));
            }
            let column = (-f.longitude[0]).rem_euclid(360.) / dxs;
            if (column - column.round()).abs() > 1.0e-5 {
                return Err(frame_invalid("zero longitude is not on the source lattice; interpolation is forbidden"));
            }
            for (_, lon) in &mut columns { *lon = lon.rem_euclid(360.); }
            columns.sort_by(|a, b| a.1.total_cmp(&b.1));
        }
        _ => return Err(usage("--longitude-origin must be source or zero")),
    }
    if let Some(raw) = args.get("--target-area") {
        if coverage != "regional" {return Err(usage("target-area requires regional coverage"));}
        let a: Vec<f64> = raw.split(',').map(str::parse).collect::<std::result::Result<_,_>>().map_err(io_error)?;
        if a.len()!=4 || a.iter().any(|v|!v.is_finite()) || a[0]>=a[2]
            || a[3]-a[1]<=0. || a[3]-a[1]>=360. {return Err(usage("invalid target-area SOUTH,WEST,NORTH,EAST"));}
        if a[0]<f.latitude.iter().copied().fold(f64::INFINITY,f64::min)+2.*dy.abs()
            || a[2]>f.latitude.iter().copied().fold(f64::NEG_INFINITY,f64::max)-2.*dy.abs()
            {return Err(frame_invalid("target latitude halo exceeds source footprint"));}
        rows.retain(|&j| f.latitude[j]>=a[0]-2.*dy.abs() && f.latitude[j]<=a[2]+2.*dy.abs());
        let center=(a[1]+a[3])*0.5;
        for (_,lon) in &mut columns { *lon=center+(*lon-center+180.).rem_euclid(360.)-180.; }
        columns.retain(|&(_,lon)|lon>=a[1]-2.*dx.abs() && lon<=a[3]+2.*dx.abs());
        columns.sort_by(|a,b| a.1.total_cmp(&b.1));
        if rows.len()<5 || columns.len()<5 {return Err(frame_invalid("target crop is too small"));}
        if columns[0].1>a[1]-dx.abs() || columns.last().unwrap().1<a[3]+dx.abs()
            {return Err(frame_invalid("target longitude halo exceeds source footprint"));}
    }
    let latitude: Vec<_> = rows.iter().map(|&j|f.latitude[j]).collect();
    let longitude: Vec<_> = columns.iter().map(|&(_,lon)|lon).collect();
    let out_dy=axis(&latitude,"output latitude")?;
    let out_dx=axis(&longitude,"output longitude")?;
    let lattice = target.unwrap_or(LatLonLattice {
        south: latitude[0],
        west: longitude[0],
        dlat: out_dy,
        dlon: out_dx,
        nx: longitude.len(),
        ny: latitude.len(),
    });
    let resample = match target {
        None => None,
        Some(box_) => {
            let geometry = SourceGeometry::from_frame(f)?;
            let placed: Vec<(f64, f64)> = (0..box_.ny)
                .into_par_iter()
                .flat_map_iter(|j| {
                    let geometry = &geometry;
                    (0..box_.nx).map(move |i| {
                        geometry.place(
                            box_.south + box_.dlat * j as f64,
                            box_.west + box_.dlon * i as f64,
                        )
                    })
                })
                .collect();
            let window = match &f.atmospheric_window {
                None => None,
                Some(document) => {
                    let pair = |key: &str| -> Result<(usize, usize)> {
                        let values = document[key]
                            .as_array()
                            .filter(|v| v.len() == 2)
                            .ok_or_else(|| {
                                frame_invalid(format!(
                                    "frame atmospheric_window.{key} needs two integers"
                                ))
                            })?;
                        match (values[0].as_u64(), values[1].as_u64()) {
                            (Some(a), Some(b)) if a < b => Ok((a as usize, b as usize)),
                            _ => Err(frame_invalid(format!(
                                "frame atmospheric_window.{key} is not an interval"
                            ))),
                        }
                    };
                    let fields: std::collections::BTreeSet<String> = document["fields"]
                        .as_array()
                        .ok_or_else(|| {
                            frame_invalid("frame atmospheric_window.fields must be a list")
                        })?
                        .iter()
                        .filter_map(|v| v.as_str().map(str::to_owned))
                        .collect();
                    Some((pair("rows")?, pair("columns")?, fields))
                }
            };
            Some((placed, window))
        }
    };
    // A field's horizontal origin and shape in the source index space.
    let horizontal = |name: &str| -> ((usize, usize), (usize, usize)) {
        match resample.as_ref().and_then(|(_, window)| window.as_ref()) {
            Some((rows, columns, fields)) if fields.contains(name) => (
                (rows.0, columns.0),
                (rows.1 - rows.0, columns.1 - columns.0),
            ),
            _ => ((0, 0), (f.latitude.len(), f.longitude.len())),
        }
    };
    let levels: Vec<f32> = f
        .vertical_values
        .iter()
        .map(|v| (*v * if f.vertical_units == "hPa" { 100. } else { 1. }) as f32)
        .collect();
    if levels.is_empty()
        || levels
            .iter()
            .any(|v| !v.is_finite() || *v <= 0. || *v >= SURFACE)
        || levels.windows(2).any(|w| w[0] >= w[1])
    {
        return Err(frame_invalid(
            "pressure levels must be unique, positive and increasing",
        ));
    }
    let contract_bytes = read_json(Path::new(need("--contract")?))?;
    let contract: Contract = serde_json::from_slice(&contract_bytes).map_err(io_error)?;
    if contract.schema != "rw-wps.intermediate-fields.v1" {
        return Err(frame_invalid("unsupported WPS field contract"));
    }
    let composition_bytes = read_json(Path::new(need("--composition")?))?;
    let composition: Value = serde_json::from_slice(&composition_bytes).map_err(io_error)?;
    if composition["soil_layers"]["depth_units"] != "m" {
        return Err(frame_invalid("soil depth units must be m"));
    }
    // Point soil nodes (RUC) are written on the contract's node layers; layer sources as they are.
    let nodes: Option<Vec<f64>> = match composition["soil_layers"]["source_nodes"].as_array() {
        Some(rows) if composition["soil_layers"]["source_layers"].is_null() => Some(
            rows.iter()
                .map(|n| n["depth"].as_f64().filter(|d| d.is_finite() && *d >= 0.)
                    .ok_or_else(|| frame_invalid("invalid soil node depth")))
                .collect::<Result<_>>()?,
        ),
        _ => None,
    };
    if let Some(depths) = nodes.as_ref() {
        if depths.len() < 2 || depths.windows(2).any(|w| w[0] >= w[1]) {
            return Err(frame_invalid("soil nodes must be at least two, deepening"));
        }
        if contract.soil_layers_from_nodes_cm.is_empty() {
            return Err(frame_invalid("the WPS contract declares no soil_layers_from_nodes_cm for a node source"));
        }
    }
    let node_layers: Vec<Value> = contract
        .soil_layers_from_nodes_cm
        .iter()
        .map(|[top, bottom]| serde_json::json!({"top": *top as f64 / 100., "bottom": *bottom as f64 / 100.}))
        .collect();
    let layers = match nodes.as_ref() {
        Some(_) => &node_layers,
        None => composition["soil_layers"]["source_layers"]
            .as_array()
            .ok_or_else(|| frame_invalid("composition must declare soil source_layers or source_nodes"))?,
    };
    let soil: Vec<(usize, usize)> = layers
        .iter()
        .map(|l| {
            let depth = |key: &str| -> Result<usize> {
                let v = l[key]
                    .as_f64()
                    .ok_or_else(|| frame_invalid("invalid soil bound"))?
                    * 100.;
                if !v.is_finite() || !(0. ..=999.).contains(&v) || (v - v.round()).abs() > 1.0e-6 {
                    return Err(frame_invalid(
                        "WPS soil bounds require whole centimetres in 0..999",
                    ));
                }
                Ok(v.round() as usize)
            };
            let top = depth("top")?;
            let bottom = depth("bottom")?;
            if bottom <= top {
                return Err(frame_invalid("soil bounds must increase"));
            }
            Ok((top, bottom))
        })
        .collect::<Result<_>>()?;
    if soil.is_empty() || soil[0].0 != 0 || soil.windows(2).any(|w| w[0].1 != w[1].0) {
        return Err(frame_invalid(
            "soil layers must be contiguous from the surface",
        ));
    }
    let soil_set = args.get("--soil-set").copied().unwrap_or("layers");
    if !["nodes", "layers"].contains(&soil_set) {
        return Err(usage("--soil-set must be nodes or layers"));
    }
    if soil_set == "nodes" && nodes.is_none() {
        return Err(frame_invalid("--soil-set nodes requires composition source_nodes"));
    }
    let node_cm: Vec<usize> = nodes.as_ref().map(|d| d.iter().map(|x| x * 100.).map(|x| {
        if x > 999. || (x - x.round()).abs() > 1.0e-6 {
            return Err(frame_invalid("WPS soil nodes require whole centimetres in 0..999"));
        }
        Ok(x.round() as usize)
    }).collect::<Result<Vec<_>>>()).transpose()?.unwrap_or_default();
    let soil_count = if soil_set == "nodes" { node_cm.len() } else { soil.len() };
    let map_source = contract.map_sources.get(composition["name"].as_str().unwrap_or(""))
        .map(String::as_str).unwrap_or("rw-wps composed canonical state");
    // Field readers, hashes and float conversion run across the shared Rayon pool.
    let slabs: Vec<Vec<Slab>> = contract
        .bindings
        .par_iter()
        .filter(|b| b.compositions.is_empty() || b.compositions.iter().any(|n| Some(n.as_str()) == composition["name"].as_str()))
        .map(|b| -> Result<Vec<Slab>> {
            let mut matches: Vec<_> = f.fields.iter().filter(|v| v.name == b.field).collect();
            let mut scale = None;
            let mut units = b.units.as_str();
            if matches.is_empty() {
                if let Some(from) = &b.from {
                    matches = f.fields.iter().filter(|v| v.name == from.field).collect();
                    scale = Some(from.scale);
                    units = from.units.as_str();
                }
            }
            if matches.is_empty() && (b.absent == "skip" || b.optional) {
                return Ok(Vec::new());
            }
            if matches.len() != 1 {
                return Err(frame_invalid(format!(
                    "WPS binding {} requires exactly one canonical field",
                    b.field
                )));
            }
            if !["", "skip"].contains(&b.absent.as_str()) {
                return Err(frame_invalid("unknown WPS absent policy"));
            }
            let field = matches[0];
            let mut values = read_field(&stream, field)?;
            if let Some(scale) = scale {
                values.iter_mut().for_each(|value| *value *= scale);
            }
            let (origin, (height, width)) = horizontal(&field.name);
            let (axes, shape) = match b.axis.as_str() {
                "pressure" => (
                    vec!["vertical", "y", "x"],
                    vec![levels.len(), height, width],
                ),
                "surface" => (vec!["y", "x"], vec![height, width]),
                "soil" => (vec!["soil", "y", "x"], vec![nodes.as_ref().map_or(soil.len(), Vec::len), height, width]),
                _ => return Err(frame_invalid("unknown WPS axis")),
            };
            if field.units != units || field.axes != axes || field.shape != shape {
                return Err(frame_invalid(format!(
                    "{} units/axes/shape differ from WPS contract",
                    b.field
                )));
            }
            if !["", "binary_land_fraction"].contains(&b.transform.as_str()) {
                return Err(frame_invalid("unknown WPS transform"));
            }
            let mut uncovered = Vec::new();
            let planes: Vec<Vec<f64>> = match resample.as_ref() {
                None => values.chunks(height * width).map(<[f64]>::to_vec).collect(),
                Some((placed, _)) => {
                    let donors = match &support {
                        None => Donors::new(placed, origin, (height, width), &field.name)?,
                        Some(s) => Donors::with_support(
                            placed,
                            origin,
                            (height, width),
                            &field.name,
                            Some(&s.required),
                        )?,
                    };
                    if support.is_some() {
                        uncovered = donors.cells.iter().map(|c| c.0 == usize::MAX).collect();
                    }
                    let categorical =
                        b.transform == "binary_land_fraction" || b.field == "sea_ice_fraction";
                    values
                        .chunks(height * width)
                        .map(|plane| {
                            if categorical {
                                donors.nearest(plane)
                            } else if b.axis == "soil" {
                                donors.bilinear_reaching(plane, (height, width), SOIL_REACH_CELLS)
                            } else {
                                donors.bilinear(plane)
                            }
                        })
                        .collect()
                }
            };
            let planes: Vec<Vec<f64>> = match (b.axis.as_str(), nodes.as_ref()) {
                ("soil", Some(depths)) if soil_set == "layers" => soil
                    .iter()
                    .map(|&(top, bottom)| {
                        let mid = (top + bottom) as f64 / 200.;
                        let k = depths.windows(2).position(|w| mid <= w[1]).unwrap_or(depths.len() - 2);
                        let w = ((mid - depths[k]) / (depths[k + 1] - depths[k])).clamp(0., 1.);
                        planes[k].iter().zip(&planes[k + 1]).map(|(a, c)| if a.is_finite() && c.is_finite() { a + w * (c - a) } else { f64::NAN }).collect()
                    })
                    .collect(),
                _ => planes,
            };
            planes
                .iter()
                .enumerate()
                .map(|(k, values)| {
                    let name = if b.axis == "soil" {
                        if soil_set == "nodes" {
                            format!("{}{:03}", if b.wps == "ST" { "SOILT" } else { "SOILM" }, node_cm[k])
                        } else {
                            format!("{}{:03}{:03}", b.wps, soil[k].0, soil[k].1)
                        }
                    } else {
                        b.wps.clone()
                    };
                    let values = values
                        .iter()
                        .enumerate()
                        .map(|(node, v)| -> Result<f32> {
                            if v.is_nan() {
                                if b.missing != "preserve_mask"
                                    && !uncovered.get(node).copied().unwrap_or(false)
                                {
                                    return Err(frame_invalid(format!(
                                        "{} has missing values forbidden by the WPS contract",
                                        b.field
                                    )));
                                }
                                return Ok(MISSING);
                            }
                            if !v.is_finite() || !(*v as f32).is_finite() {
                                return Err(frame_invalid(format!(
                                    "{} has unrepresentable values",
                                    b.field
                                )));
                            }
                            if b.transform == "binary_land_fraction" {
                                if !(0. ..=1.).contains(v) {
                                    return Err(frame_invalid("land_fraction outside [0,1]"));
                                }
                                Ok(if *v >= 0.5 { 1. } else { 0. })
                            } else {
                                Ok(*v as f32)
                            }
                        })
                        .collect::<Result<Vec<_>>>()?;
                    Ok(Slab {
                        field: name,
                        units: b.units.clone(),
                        level: if b.axis == "pressure" {
                            levels[k]
                        } else {
                            SURFACE
                        },
                        values,
                        uncovered: uncovered.clone(),
                    })
                })
                .collect()
        })
        .collect::<Result<_>>()?;
    if let Some(support) = &support {
        support.check_search(&slabs, lattice)?;
    }
    if out.exists() {
        return Err(frame_invalid(format!(
            "refusing to overwrite {}",
            out.display()
        )));
    }
    let mut file = BufWriter::new(
        File::options()
            .write(true)
            .create_new(true)
            .open(&out)
            .map_err(io_error)?,
    );
    let result = (|| {
        for slab in slabs.iter().flatten() {
            if target.is_none() && (args.contains_key("--target-area") || args.get("--longitude-origin").copied() == Some("zero")) {
                let values = rows.iter().flat_map(|&j| columns.iter().map(move |&(i, _)| slab.values[j * f.longitude.len() + i])).collect();
                // This path crops source cells without target interpolation, so
                // there are no uncovered target nodes to carry into the slab.
                let selected = Slab { field: slab.field.clone(), units: slab.units.clone(), level: slab.level, values, uncovered: Vec::new() };
                write_slab(&mut file, &selected, f, &lattice, map_source)?;
            } else {
                write_slab(&mut file, slab, f, &lattice, map_source)?;
            }
        }
        file.flush().map_err(io_error)
    })();
    if result.is_err() {
        drop(file);
        let _ = fs::remove_file(&out);
        result?;
    }
    let mut receipt = json!({"schema":"rw-wps.intermediate-export.v1","format_version":5,"output":{"path":out,"bytes":fs::metadata(&out).map_err(io_error)?.len(),"sha256":digest::file_sha256(out.to_str().unwrap()).map_err(io_error)?},"frameset_sha256":digest::bytes_sha256(&manifest),"stream_sha256":set.stream.sha256,"composition_sha256":digest::bytes_sha256(&composition_bytes),"field_contract_sha256":digest::bytes_sha256(&contract_bytes),"valid_time":f.valid_time,"source_cycle":f.source_cycle,"soil_set":soil_set,"member":f.member,"global":target.is_none() && is_global && !args.contains_key("--target-area"),"coverage":coverage,"target_area":args.get("--target-area"),"longitude_origin_deg":lattice.west,"target_latlon":target.map(|t|json!({"south":t.south,"west":t.west,"dlat":t.dlat,"dlon":t.dlon,"nx":t.nx,"ny":t.ny})),"source_grid":f.header["grid"]["projection"],"windowed":f.atmospheric_window.is_some(),"nx":lattice.nx,"ny":lattice.ny,"skipped_absent":contract.bindings.iter().filter(|b|b.absent=="skip"&&!f.fields.iter().any(|v|v.name==b.field)).map(|b|b.wps.clone()).collect::<Vec<_>>(),"nfglevels":levels.len()+1,"nfgsoillevels":soil_count,"soil_layers_cm":soil,"fields":slabs.iter().flatten().map(|s|json!({"field":s.field,"level":s.level,"missing":s.values.iter().filter(|v|**v==MISSING).count()})).collect::<Vec<_>>()});
    if let Some(support) = &support {
        receipt["target_support"] = support.receipt.clone();
    }
    Ok(receipt)
}

#[cfg(test)]
mod support_tests {
    use super::*;

    #[test]
    fn unused_uncovered_nodes_are_missing_and_covered_words_stay_exact() {
        let placed = [(0.5, 0.75), (1.25, 1.5), (-100., 1623.88)];
        let plane = [1., 2., 3., 4., 5., 6., 7., 8., 9.];
        let old = Donors::new(&placed[..2], (0, 0), (3, 3), "field").unwrap();
        let new =
            Donors::with_support(&placed, (0, 0), (3, 3), "field", Some(&[true, true, false]))
                .unwrap();
        for (expected, actual) in [
            (old.bilinear(&plane), new.bilinear(&plane)),
            (old.nearest(&plane), new.nearest(&plane)),
            (
                old.bilinear_reaching(&plane, (3, 3), 8),
                new.bilinear_reaching(&plane, (3, 3), 8),
            ),
        ] {
            assert_eq!(
                expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                actual[..2].iter().map(|x| x.to_bits()).collect::<Vec<_>>()
            );
            assert!(actual[2].is_nan());
        }
        assert!(
            Donors::with_support(&placed, (0, 0), (3, 3), "field", Some(&[true, true, true]))
                .is_err()
        );
        assert!(Donors::new(&placed, (0, 0), (3, 3), "field").is_err());
    }
}
