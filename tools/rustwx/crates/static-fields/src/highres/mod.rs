//! LANE 3.  High-resolution overlay compute
//! (`gpuwm/static/highres.py` minus the raster substrate, which lives
//! in [`crate::raster`]).
//!
//! What moves here: bound-raster hash verification, the model-grid
//! raster geometry (`_grid_crs` / `_raster_geometry` /
//! `_extended_grid`), the continuous and mapped-category resamples,
//! the depth-weighted SoilGrids means, the USDA texture triangle, the
//! Manhattan-nearest donor BFS, BOTH merges (terrain-only and full)
//! with the deep-soil land gates and the water-sentinel discipline,
//! and the 1x1-degree tile-id enumeration the fetch driver consumes.
//!
//! What stays Python (and why): the network fetch driver
//! (`highres_fetch.py` URL/cache/retry/sidecar loops: the bytes it
//! moves are opaque payloads written verbatim to disk and EVERY byte
//! is decoded, mosaicked and warped here), and the policy shell
//! (`highres_production.py` TOML parsing, refusal routing, receipts,
//! console lines).
//!
//! Parity: triangle/crosswalk/donor-fill/merge are byte-parity against
//! the Python; the warped planes are tolerance-parity (see
//! [`crate::raster`]).  Refusal DECISIONS (which gate fires) must be
//! identical; on inputs with several simultaneous defects the field
//! NAMED may differ (this side reports in deterministic alphabetical
//! field order, the Python in dict insertion order): same decision,
//! same gate, documented divergence.

use std::collections::BTreeMap;
use std::io::Read;
use std::path::Path;

use crate::error::{Result, StaticError};
use crate::projection::GridSpec;
use crate::raster::warp::{self, Resampling};
use crate::raster::{Crs, Raster, geotiff};
use crate::types::{Field, FieldSet, Grid2, Stack3};

pub mod urban;
pub mod soil;

fn invalid(message: impl Into<String>) -> StaticError {
    StaticError::Invalid(message.into())
}

/// Render a `["A", "B"]`-style list exactly as Python prints a list of
/// strings, so refusal messages match the reference byte for byte.
fn python_list(names: &[&str]) -> String {
    let mut out = String::from("[");
    for (index, name) in names.iter().enumerate() {
        if index > 0 {
            out.push_str(", ");
        }
        out.push('\'');
        out.push_str(name);
        out.push('\'');
    }
    out.push(']');
    out
}

/// A field's shape as Python prints the numpy array's `shape` tuple.
fn python_shape(field: &Field) -> String {
    match field {
        Field::Plane(grid) => format!("({}, {})", grid.ny, grid.nx),
        Field::Stack(stack) => format!("({}, {}, {})", stack.planes, stack.ny, stack.nx),
    }
}

/// Python `format(x, '.6g')` for the deep-soil gate message.
pub(crate) fn format_g6(x: f64) -> String {
    if x.is_nan() {
        return "nan".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf".into() } else { "-inf".into() };
    }
    if x == 0.0 {
        return if x.is_sign_negative() {
            "-0".into()
        } else {
            "0".into()
        };
    }
    // Round to 6 significant digits via exponential formatting.
    let exp_text = format!("{:.5e}", x);
    let (mantissa, exponent) = exp_text.split_once('e').unwrap();
    let exponent: i32 = exponent.parse().unwrap();
    if (-4..6).contains(&exponent) {
        let decimals = (5 - exponent).max(0) as usize;
        let mut fixed = format!("{:.*}", decimals, x);
        if fixed.contains('.') {
            while fixed.ends_with('0') {
                fixed.pop();
            }
            if fixed.ends_with('.') {
                fixed.pop();
            }
        }
        fixed
    } else {
        let mut m = mantissa.to_string();
        if m.contains('.') {
            while m.ends_with('0') {
                m.pop();
            }
            if m.ends_with('.') {
                m.pop();
            }
        }
        format!(
            "{m}e{}{:02}",
            if exponent < 0 { "-" } else { "+" },
            exponent.abs()
        )
    }
}

// ---------------------------------------------------------------------------
// Bound rasters
// ---------------------------------------------------------------------------

/// The provenance-bound source description crossing the seam
/// (`BoundRaster` minus the receipt strings, which stay Python).
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct BoundRasterSpec {
    pub path: std::path::PathBuf,
    pub sha256: String,
    #[serde(default)]
    pub expected_bytes: Option<u64>,
    #[serde(default)]
    pub crs_override: Option<String>,
    #[serde(default)]
    pub nodata_override: Option<f64>,
    #[serde(default = "default_scale")]
    pub scale_factor: f64,
}

fn default_scale() -> f64 {
    1.0
}

/// Streaming SHA-256 of one file (`sha256_file`).
pub fn sha256_file(path: &Path) -> Result<String> {
    use sha2::{Digest, Sha256};
    let mut digest = Sha256::new();
    let mut file = std::fs::File::open(path).map_err(|err| {
        StaticError::Missing(format!("high-resolution raster missing: {path:?} ({err})"))
    })?;
    let mut block = vec![0u8; 8 * 1024 * 1024];
    loop {
        let n = file.read(&mut block)?;
        if n == 0 {
            break;
        }
        digest.update(&block[..n]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

impl BoundRasterSpec {
    /// `BoundRaster.verify`: existence, size, then the streaming hash.
    pub fn verify(&self) -> Result<()> {
        let meta = std::fs::metadata(&self.path).map_err(|_| {
            StaticError::Missing(format!(
                "high-resolution raster missing: {}",
                self.path.display()
            ))
        })?;
        if let Some(expected) = self.expected_bytes {
            if meta.len() != expected {
                return Err(invalid(format!(
                    "high-resolution raster size mismatch for {}: \
                     expected {expected}, observed {}",
                    self.path.display(),
                    meta.len()
                )));
            }
        }
        let observed = sha256_file(&self.path)?;
        if observed != self.sha256 {
            return Err(invalid(format!(
                "high-resolution raster hash mismatch for {}: expected \
                 {}, observed {observed}",
                self.path.display(),
                self.sha256
            )));
        }
        Ok(())
    }

    fn crs_override_parsed(&self) -> Result<Option<Crs>> {
        self.crs_override
            .as_deref()
            .map(Crs::parse_override)
            .transpose()
    }

    /// Verify, then read band 1 masked and scaled.
    pub fn open(&self) -> Result<Raster> {
        self.verify()?;
        geotiff::read_band1(
            &self.path,
            self.crs_override_parsed()?,
            self.nodata_override,
            self.scale_factor,
        )
    }
}

// ---------------------------------------------------------------------------
// Model-grid raster geometry (`_grid_crs` / `_raster_geometry` /
// `_extended_grid`)
// ---------------------------------------------------------------------------

/// Halo-extended grid spec (`_extended_grid`).
pub fn extended_spec(spec: &GridSpec, halo: usize) -> GridSpec {
    if halo == 0 {
        return spec.clone();
    }
    let mut out = spec.clone();
    out.e_we += 2 * halo as i64;
    out.e_sn += 2 * halo as i64;
    out.known_x += halo as f64;
    out.known_y += halo as f64;
    out
}

/// North-first raster geometry for WRF mass points
/// (`_raster_geometry`): `(crs, transform, (ny, nx))`.
pub fn raster_geometry(spec: &GridSpec) -> Result<(Crs, [f64; 6], (usize, usize))> {
    let crs = Crs::ModelSphere(spec.clone());
    let projection = crs.point_projection()?;
    let (ref_x, ref_y) = projection.forward(spec.ref_lon, spec.ref_lat);
    let nx = (spec.e_we - 1) as usize;
    let ny = (spec.e_sn - 1) as usize;
    let west_center = ref_x - (spec.known_x - 1.0) * spec.dx;
    let south_center = ref_y - (spec.known_y - 1.0) * spec.dy;
    let west_edge = west_center - 0.5 * spec.dx;
    let north_edge = south_center + (ny as f64 - 0.5) * spec.dy;
    Ok((
        crs,
        [spec.dx, 0.0, west_edge, 0.0, -spec.dy, north_edge],
        (ny, nx),
    ))
}

/// Open metadata after provenance validation, without decoding the full band.
pub fn open_bound_reader(source: &BoundRasterSpec) -> Result<(geotiff::TiffReader, Raster)> {
    source.verify()?;
    let mut reader = geotiff::TiffReader::open(&source.path)?;
    reader.set_cache_budget(8 * 1024 * 1024);
    let crs = match (&reader.crs, source.crs_override_parsed()?) {
        (Some(own), _) => own.clone(),
        (None, Some(given)) => given,
        (None, None) => {
            return Err(invalid(format!(
                "raster {:?} has no CRS and declares no crs_override",
                source.path
            )));
        }
    };
    let carrier = Raster {
        ny: reader.height,
        nx: reader.width,
        values: Vec::new(),
        transform: reader.transform,
        crs,
    };
    Ok((reader, carrier))
}

/// Production continuous raster path, with bounded source and corner windows.
pub fn resample_continuous_bound(
    source: &BoundRasterSpec,
    spec: &GridSpec,
    method: Resampling,
) -> Result<Grid2> {
    let (reader, carrier) = open_bound_reader(source)?;
    let nodata = source.nodata_override.or(reader.nodata);
    // Verification above hashes the immutable source once. Worker readers
    // open its metadata directly and keep private, fixed-size block caches.
    drop(reader);
    let (dst_crs, transform, (ny, nx)) = raster_geometry(spec)?;
    warp::reproject_continuous_windowed_parallel(
        &carrier,
        &dst_crs,
        transform,
        ny,
        nx,
        method,
        warp::continuous_worker_limit(),
        || {
            let mut reader = geotiff::TiffReader::open(&source.path)?;
            reader.set_cache_budget(warp::CONTINUOUS_READER_CACHE_BYTES);
            Ok(reader)
        },
        |reader, c, r, w, h| {
            let mut values = reader.read_window_raw(c, r, w, h)?;
            for value in &mut values {
                if nodata == Some(*value) || value.is_nan() {
                    *value = f64::NAN;
                } else {
                    *value *= source.scale_factor;
                }
            }
            Ok(values)
        },
    )
}

/// Production categorical raster path. The full-source validation is streamed
/// separately so categories outside the destination retain their refusal rule.
pub fn resample_mapped_categories_bound(
    source: &BoundRasterSpec,
    spec: &GridSpec,
    mapping: &BTreeMap<i64, i64>,
    category_count: usize,
) -> Result<Stack3> {
    let (mut reader, carrier) = open_bound_reader(source)?;
    let nodata = source.nodata_override.or(reader.nodata);
    let mut seen = std::collections::BTreeSet::new();
    for r in (0..carrier.ny).step_by(256) {
        for c in (0..carrier.nx).step_by(256) {
            let raw =
                reader.read_window_raw(c, r, 256.min(carrier.nx - c), 256.min(carrier.ny - r))?;
            for value in raw {
                if value.is_finite() && nodata != Some(value) {
                    seen.insert(value as i64);
                }
            }
        }
    }
    let unknown: Vec<i64> = seen
        .iter()
        .copied()
        .filter(|v| !mapping.contains_key(v))
        .collect();
    if !unknown.is_empty() {
        return Err(invalid(format!(
            "raster {} contains unmapped categories {unknown:?}",
            source.path.display()
        )));
    }
    for raw in seen {
        let v = mapping[&raw] as i16;
        if v > 0 && v as usize > category_count {
            return Err(invalid(format!(
                "mapped category {v} is outside 1..{category_count}"
            )));
        }
    }
    let (dst_crs, transform, (ny, nx)) = raster_geometry(spec)?;
    warp::reproject_categories_windowed(
        &carrier,
        &dst_crs,
        transform,
        ny,
        nx,
        category_count,
        |c, r, w, h| {
            let raw = reader.read_window_raw(c, r, w, h)?;
            Ok(raw
                .into_iter()
                .map(|v| {
                    if !v.is_finite() || nodata == Some(v) {
                        0
                    } else {
                        mapping.get(&(v as i64)).copied().unwrap_or(0) as i16
                    }
                })
                .collect())
        },
    )
}

/// Reproject one continuous raster to mass points in south-north order
/// (`resample_continuous`).
pub fn resample_continuous(source: &Raster, spec: &GridSpec, method: Resampling) -> Result<Grid2> {
    let (dst_crs, dst_transform, (ny, nx)) = raster_geometry(spec)?;
    warp::reproject_continuous(source, &dst_crs, dst_transform, ny, nx, method)
}

/// Map raw categories then compute target-cell area fractions
/// (`resample_mapped_categories`).  `raw` is the UNMASKED band with
/// `nodata` still in place, exactly as the Python reads it.
pub fn resample_mapped_categories(
    raw: &Raster,
    label: &str,
    spec: &GridSpec,
    mapping: &BTreeMap<i64, i64>,
    category_count: usize,
    nodata: Option<f64>,
) -> Result<Stack3> {
    let mut valid: Vec<bool> = raw
        .values
        .iter()
        .map(|value| {
            value.is_finite()
                && match nodata {
                    Some(sentinel) => *value != sentinel,
                    None => true,
                }
        })
        .collect();
    // Observed raw categories must all be mapped, by name.
    let mut observed: Vec<i64> = raw
        .values
        .iter()
        .zip(&valid)
        .filter(|(_, ok)| **ok)
        .map(|(value, _)| *value as i64)
        .collect();
    observed.sort_unstable();
    observed.dedup();
    let unknown: Vec<i64> = observed
        .iter()
        .copied()
        .filter(|category| !mapping.contains_key(category))
        .collect();
    if !unknown.is_empty() {
        return Err(invalid(format!(
            "raster {label} contains unmapped categories {unknown:?}"
        )));
    }
    let mut mapped = vec![0i16; raw.values.len()];
    for (index, value) in raw.values.iter().enumerate() {
        if valid[index] {
            if let Some(target) = mapping.get(&(*value as i64)) {
                mapped[index] = *target as i16;
            }
        }
    }
    for (index, ok) in valid.iter_mut().enumerate() {
        *ok = *ok && mapped[index] > 0;
    }
    let (dst_crs, dst_transform, (ny, nx)) = raster_geometry(spec)?;
    warp::reproject_category_fractions(
        &mapped,
        &valid,
        raw,
        &dst_crs,
        dst_transform,
        ny,
        nx,
        category_count,
    )
}

/// [`resample_mapped_categories`] for an 8-bit band held as bytes: the
/// same validity, the same unmapped-category refusal and the same
/// fractions, at four bytes per source pixel instead of eleven.
/// `carrier` supplies only the georeference (its `values` are unread).
pub fn resample_mapped_categories_u8(
    raw: &[u8],
    carrier: &Raster,
    label: &str,
    spec: &GridSpec,
    mapping: &BTreeMap<i64, i64>,
    category_count: usize,
    nodata: Option<f64>,
) -> Result<Stack3> {
    if raw.len() != carrier.ny * carrier.nx {
        return Err(invalid(
            "category values and validity mask shapes differ".to_string(),
        ));
    }
    // An 8-bit sample equals the sentinel only when the sentinel is a
    // whole number in 0..=255; any other sentinel masks nothing, exactly
    // as the f64 comparison does.
    let sentinel: Option<u8> = nodata
        .filter(|value| value.is_finite() && value.fract() == 0.0 && (0.0..=255.0).contains(value))
        .map(|value| value as u8);
    let mut seen = [false; 256];
    for value in raw {
        seen[*value as usize] = true;
    }
    let unknown: Vec<i64> = (0..256usize)
        .filter(|value| {
            seen[*value]
                && Some(*value as u8) != sentinel
                && !mapping.contains_key(&(*value as i64))
        })
        .map(|value| value as i64)
        .collect();
    if !unknown.is_empty() {
        return Err(invalid(format!(
            "raster {label} contains unmapped categories {unknown:?}"
        )));
    }
    let mut table = [0i16; 256];
    for (source, target) in mapping {
        if (0..=255).contains(source) {
            table[*source as usize] = *target as i16;
        }
    }
    let mapped: Vec<i16> = raw
        .iter()
        .map(|value| {
            if Some(*value) == sentinel {
                0
            } else {
                table[*value as usize]
            }
        })
        .collect();
    let valid: Vec<bool> = mapped.iter().map(|value| *value > 0).collect();
    let (dst_crs, dst_transform, (ny, nx)) = raster_geometry(spec)?;
    warp::reproject_category_fractions(
        &mapped,
        &valid,
        carrier,
        &dst_crs,
        dst_transform,
        ny,
        nx,
        category_count,
    )
}

/// Metres per degree of latitude on the mean sphere.
const METRES_PER_DEGREE: f64 = 111_320.0;

/// `(longitude, latitude)` window margin in degrees for a margin stated
/// in metres (`margin_degrees` in `gpuwm/static/highres_fetch.py`).  On a
/// geographic raster a metre margin read as degrees widened a 2 km margin
/// to 2000 degrees, so the land-cover window was the whole raster.  The
/// longitude margin is widened by the footprint's most poleward latitude,
/// floored at cos 87 degrees.
pub fn margin_degrees(lat_min: f64, lat_max: f64, margin_m: f64) -> (f64, f64) {
    let extreme = lat_min.abs().max(lat_max.abs()).min(90.0);
    let shrink = extreme.to_radians().cos().max(87.0f64.to_radians().cos());
    (
        margin_m / (METRES_PER_DEGREE * shrink),
        margin_m / METRES_PER_DEGREE,
    )
}

/// `_require_coverage`: every target value must be finite.
pub fn require_coverage(name: &str, values: &[f64]) -> Result<()> {
    let count = values.iter().filter(|value| !value.is_finite()).count();
    if count > 0 {
        return Err(invalid(format!(
            "high-resolution {name} lacks {count} target values"
        )));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// USDA texture triangle
// ---------------------------------------------------------------------------

/// USDA texture categories 1..12 from sand/silt/clay percentages
/// (`usda_texture_category`); refuses unclassified points by value,
/// exactly as the Python.
pub fn usda_texture_category(sand: &[f64], silt: &[f64], clay: &[f64]) -> Result<Vec<i16>> {
    if sand.len() != silt.len() || sand.len() != clay.len() {
        return Err(invalid("sand, silt, and clay shapes differ"));
    }
    let n = sand.len();
    let mut s = vec![0.0f64; n];
    let mut si = vec![0.0f64; n];
    let mut c = vec![0.0f64; n];
    for index in 0..n {
        let total = sand[index] + silt[index] + clay[index];
        if !total.is_finite() || total <= 0.0 {
            return Err(invalid("soil texture contains invalid totals"));
        }
        s[index] = sand[index] / total * 100.0;
        si[index] = silt[index] / total * 100.0;
        c[index] = clay[index] / total * 100.0;
    }
    let mut out = vec![0i16; n];
    for index in 0..n {
        let (sand, silt, clay) = (s[index], si[index], c[index]);
        let rules: [(i16, bool); 12] = [
            (1, silt + 1.5 * clay < 15.0),
            (2, silt + 1.5 * clay >= 15.0 && silt + 2.0 * clay < 30.0),
            (
                3,
                ((7.0..20.0).contains(&clay) && sand > 52.0 && silt + 2.0 * clay >= 30.0)
                    || (clay < 7.0 && silt < 50.0 && silt + 2.0 * clay >= 30.0),
            ),
            (
                4,
                (silt >= 50.0 && (12.0..27.0).contains(&clay))
                    || ((50.0..80.0).contains(&silt) && clay < 12.0),
            ),
            (5, silt >= 80.0 && clay < 12.0),
            (
                6,
                (7.0..27.0).contains(&clay) && (28.0..50.0).contains(&silt) && sand <= 52.0,
            ),
            (
                7,
                (20.0..35.0).contains(&clay) && silt < 28.0 && sand > 45.0,
            ),
            (8, (27.0..40.0).contains(&clay) && sand <= 20.0),
            (
                9,
                (27.0..40.0).contains(&clay) && sand > 20.0 && sand <= 45.0,
            ),
            (10, clay >= 35.0 && sand > 45.0),
            (11, clay >= 40.0 && silt >= 40.0),
            (12, clay >= 40.0 && sand <= 45.0 && silt < 40.0),
        ];
        for (category, condition) in rules {
            if out[index] == 0 && condition {
                out[index] = category;
            }
        }
        if out[index] == 0 {
            return Err(invalid(format!(
                "USDA texture rules left an unclassified point: \
                 sand={sand:.3}, silt={silt:.3}, clay={clay:.3}"
            )));
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// SoilGrids depth-weighted means (`_soilgrids_categories` minus IO)
// ---------------------------------------------------------------------------

/// Depth-weighted mean of co-registered component planes; the caller
/// (capi layer) has already read, masked, scaled and co-registration-
/// checked them.  `planes[component]` lists depth planes in the depth
/// order of `weights`, whose accumulation order defines the result.
/// Returns `(category, valid, raw_total)` over the source grid.
pub fn soilgrids_categories(
    planes: &BTreeMap<String, Vec<Vec<f64>>>,
    weights: &[f64],
    len: usize,
) -> Result<(Vec<i16>, Vec<bool>, Vec<f64>)> {
    let components = ["sand", "silt", "clay"];
    for component in components {
        let stack = planes.get(component).ok_or_else(|| {
            StaticError::Missing(format!("missing SoilGrids sources: {component:?}"))
        })?;
        if stack.len() != weights.len() {
            return Err(invalid(format!(
                "SoilGrids component {component:?} carries {} depth \
                 planes, weights expect {}",
                stack.len(),
                weights.len()
            )));
        }
        for plane in stack {
            if plane.len() != len {
                return Err(invalid("SoilGrids source rasters are not co-registered"));
            }
        }
    }
    let weight_sum: f64 = weights.iter().sum();
    let mut means: BTreeMap<&str, Vec<f64>> = BTreeMap::new();
    for component in components {
        let stack = &planes[component];
        let mut mean = vec![f64::NAN; len];
        for index in 0..len {
            let mut all_finite = true;
            let mut accumulated = 0.0f64;
            for (plane, weight) in stack.iter().zip(weights) {
                let value = plane[index];
                if !value.is_finite() {
                    all_finite = false;
                    break;
                }
                accumulated += value * weight;
            }
            if all_finite {
                mean[index] = accumulated / weight_sum;
            }
        }
        means.insert(component, mean);
    }
    let valid: Vec<bool> = (0..len)
        .map(|index| {
            components
                .iter()
                .all(|component| means[*component][index].is_finite())
        })
        .collect();
    // The triangle runs on the VALID subset, exactly as the Python
    // indexes it, so an invalid total on a valid cell refuses there.
    let subset: Vec<usize> = (0..len).filter(|index| valid[*index]).collect();
    let sand: Vec<f64> = subset.iter().map(|i| means["sand"][*i]).collect();
    let silt: Vec<f64> = subset.iter().map(|i| means["silt"][*i]).collect();
    let clay: Vec<f64> = subset.iter().map(|i| means["clay"][*i]).collect();
    let categories = usda_texture_category(&sand, &silt, &clay)?;
    let mut category = vec![0i16; len];
    for (subset_index, index) in subset.iter().enumerate() {
        category[*index] = categories[subset_index];
    }
    let raw_total: Vec<f64> = (0..len)
        .map(|index| means["sand"][index] + means["silt"][index] + means["clay"][index])
        .collect();
    Ok((category, valid, raw_total))
}

// ---------------------------------------------------------------------------
// Manhattan-nearest donors
// ---------------------------------------------------------------------------

/// Deterministic Manhattan-nearest donor indices (`_nearest_donors`):
/// multi-source BFS seeded in row-major order over the valid cells,
/// 4-neighbour expansion in the Python's (up, left, right, down) push
/// order, the queue order IS the tie-break and is part of the byte
/// contract.  Returns `(donor_y, donor_x)`.
pub fn nearest_donors(valid: &[bool], ny: usize, nx: usize) -> Result<(Vec<i32>, Vec<i32>)> {
    if valid.len() != ny * nx || !valid.iter().any(|ok| *ok) {
        return Err(invalid("nearest-donor mask must be 2-D with a valid cell"));
    }
    let mut donor_y = vec![-1i32; ny * nx];
    let mut donor_x = vec![-1i32; ny * nx];
    let mut queue: std::collections::VecDeque<(usize, usize)> = std::collections::VecDeque::new();
    for y in 0..ny {
        for x in 0..nx {
            if valid[y * nx + x] {
                donor_y[y * nx + x] = y as i32;
                donor_x[y * nx + x] = x as i32;
                queue.push_back((y, x));
            }
        }
    }
    while let Some((y, x)) = queue.pop_front() {
        let neighbours = [
            (y.wrapping_sub(1), x),
            (y, x.wrapping_sub(1)),
            (y, x + 1),
            (y + 1, x),
        ];
        for (yy, xx) in neighbours {
            if yy < ny && xx < nx && donor_y[yy * nx + xx] < 0 {
                donor_y[yy * nx + xx] = donor_y[y * nx + x];
                donor_x[yy * nx + xx] = donor_x[y * nx + x];
                queue.push_back((yy, xx));
            }
        }
    }
    Ok((donor_y, donor_x))
}

// ---------------------------------------------------------------------------
// FieldSet access helpers
// ---------------------------------------------------------------------------

fn plane<'a>(set: &'a FieldSet, name: &str) -> Result<&'a Grid2> {
    match set.fields.get(name) {
        Some(Field::Plane(grid)) => Ok(grid),
        Some(Field::Stack(_)) => Err(invalid(format!(
            "field {name} is 3-D where a 2-D plane is required"
        ))),
        None => Err(StaticError::Missing(format!(
            "field set has no field {name:?}"
        ))),
    }
}

fn missing_from(set: &FieldSet, required: &[&'static str]) -> Vec<&'static str> {
    let mut missing: Vec<&'static str> = required
        .iter()
        .copied()
        .filter(|name| !set.fields.contains_key(*name))
        .collect();
    missing.sort_unstable();
    missing
}

// ---------------------------------------------------------------------------
// Terrain-only merge (`merge_terrain_override`)
// ---------------------------------------------------------------------------

/// The audit counters of a merge, rendered to JSON by the seam.
#[derive(Debug, Clone, serde::Serialize)]
pub struct MergeAudit {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub terrain_cells_changed: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub land_water_cells_unchanged: Option<u64>,
    pub newly_land_nearest_climatology_fallback_cells: u64,
    pub newly_water_masked_cells: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub unchanged_land_water_cells: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub deep_soil_water_masked_cells: Option<u64>,
    /// New land the overlay found on a grid whose baseline has no land
    /// cell to donate climatology: kept as the baseline's water.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub newly_land_without_donor_kept_water_cells: Option<u64>,
}

/// Terrain-only merge (`merge_terrain_override`): swap HGT_M, recompute
/// TMN by the 6.5 K/km lapse on land, leave the mask alone.
pub fn merge_terrain_override(
    baseline: &FieldSet,
    hgt_override: &Grid2,
) -> Result<(FieldSet, MergeAudit)> {
    let missing = missing_from(baseline, &["HGT_M", "LANDMASK", "SOILTEMP"]);
    if !missing.is_empty() {
        return Err(StaticError::Missing(format!(
            "baseline static fields missing {}",
            python_list(&missing)
        )));
    }
    let baseline_hgt = plane(baseline, "HGT_M")?;
    if (hgt_override.ny, hgt_override.nx) != (baseline_hgt.ny, baseline_hgt.nx) {
        return Err(invalid(format!(
            "terrain override shape ({}, {}) differs from baseline \
             ({}, {})",
            hgt_override.ny, hgt_override.nx, baseline_hgt.ny, baseline_hgt.nx
        )));
    }
    let mut out = FieldSet {
        fields: baseline
            .fields
            .iter()
            .filter(|(name, _)| !matches!(name.as_str(), "HGT_M" | "TMN"))
            .map(|(name, field)| (name.clone(), field.clone()))
            .collect(),
        coverage_reports: baseline.coverage_reports.clone(),
    };
    let changed = baseline_hgt
        .data
        .iter()
        .zip(&hgt_override.data)
        .filter(|(a, b)| a != b)
        .count() as u64;
    out.fields
        .insert("HGT_M".into(), Field::Plane(hgt_override.clone()));

    let landmask = plane(baseline, "LANDMASK")?;
    let soiltemp = plane(baseline, "SOILTEMP")?;
    if landmask.data.len() != hgt_override.data.len()
        || soiltemp.data.len() != hgt_override.data.len()
    {
        return Err(invalid(
            "baseline LANDMASK/SOILTEMP shapes differ from HGT_M",
        ));
    }
    let mut tmn = Grid2 {
        ny: landmask.ny,
        nx: landmask.nx,
        data: vec![0.0; landmask.data.len()],
    };
    for index in 0..landmask.data.len() {
        let land = landmask.data[index] > 0.5;
        tmn.data[index] = if land {
            soiltemp.data[index] - 0.0065 * hgt_override.data[index]
        } else {
            soiltemp.data[index]
        };
    }
    let cell_count = landmask.data.len() as u64;
    out.fields.insert("TMN".into(), Field::Plane(tmn));

    for (name, field) in &out.fields {
        if field.data().iter().any(|value| !value.is_finite()) {
            return Err(invalid(format!(
                "merged terrain-only field {name} is non-finite"
            )));
        }
    }
    Ok((
        out,
        MergeAudit {
            terrain_cells_changed: Some(changed),
            land_water_cells_unchanged: Some(cell_count),
            newly_land_nearest_climatology_fallback_cells: 0,
            newly_water_masked_cells: 0,
            unchanged_land_water_cells: None,
            deep_soil_water_masked_cells: None,
            newly_land_without_donor_kept_water_cells: None,
        },
    ))
}

// ---------------------------------------------------------------------------
// Full merge (`merge_highres_overrides`)
// ---------------------------------------------------------------------------

/// The physical envelope a deep-soil temperature must sit in
/// (`_DEEP_SOIL_KELVIN_RANGE`).
const DEEP_SOIL_KELVIN_RANGE: (f64, f64) = (170.0, 400.0);

/// Deep-soil fields whose water cells carry a mask, not a measurement.
const LAND_ONLY_DEEP_SOIL: [&str; 2] = ["SOILTEMP", "TMN"];

/// `_refuse_unusable_merged_statics`, verbatim gate logic.
fn refuse_unusable_merged_statics(out: &FieldSet, new_land: &[bool]) -> Result<()> {
    for (name, field) in &out.fields {
        let (planes, ny, nx) = field.dims();
        let data = field.data();
        if data.is_empty() || new_land.len() != ny * nx {
            continue;
        }
        let land_cells = new_land.iter().filter(|ok| **ok).count() * planes;
        let mut holed = 0usize;
        for plane_index in 0..planes {
            for cell in 0..ny * nx {
                if new_land[cell] && !data[plane_index * ny * nx + cell].is_finite() {
                    holed += 1;
                }
            }
        }
        if holed > 0 {
            return Err(invalid(format!(
                "merged high-resolution field {name} is non-finite on \
                 {holed} land cell(s) of {land_cells} -- the \
                 high-resolution mask resolves land the source field \
                 does not cover"
            )));
        }
        if !LAND_ONLY_DEEP_SOIL.contains(&name.as_str()) {
            if data.iter().any(|value| !value.is_finite()) {
                return Err(invalid(format!(
                    "merged high-resolution field {name} is non-finite"
                )));
            }
            continue;
        }
        let (low, high) = DEEP_SOIL_KELVIN_RANGE;
        let mut count = 0usize;
        let mut sample_min = f64::INFINITY;
        let mut sample_max = f64::NEG_INFINITY;
        for plane_index in 0..planes {
            for cell in 0..ny * nx {
                if !new_land[cell] {
                    continue;
                }
                let value = data[plane_index * ny * nx + cell];
                if !(value >= low && value <= high) {
                    count += 1;
                    if value < sample_min {
                        sample_min = value;
                    }
                    if value > sample_max {
                        sample_max = value;
                    }
                }
            }
        }
        if count > 0 {
            return Err(invalid(format!(
                "merged high-resolution {name} is not a temperature on \
                 {count} land cell(s) of {land_cells}: range \
                 [{}, {}] K outside {}..{}.  0 K is the geog \
                 soil_temperature fill, so this is a deep-soil source \
                 that did not decode over the land this domain resolves \
                 -- check the geog soil_temperature tile coverage for \
                 this footprint",
                format_g6(sample_min),
                format_g6(sample_max),
                format_g6(low),
                format_g6(high)
            )));
        }
    }
    Ok(())
}

/// Full merge (`merge_highres_overrides`): override fields in, donor
/// climatology fill on newly-land, water fills (ALBEDO12M -> 8, rest ->
/// 0, deep-soil NaN sentinel inside the gate then 0.0 on return), TMN
/// recompute, land-usability gates.
pub fn merge_highres_overrides(
    baseline: &FieldSet,
    overrides: &FieldSet,
) -> Result<(FieldSet, MergeAudit)> {
    let missing = missing_from(
        baseline,
        &[
            "HGT_M",
            "LANDUSEF",
            "LANDMASK",
            "LU_INDEX",
            "SOILCTOP",
            "SCT_DOM",
            "SOILCBOT",
            "SCB_DOM",
            "GREENFRAC",
            "LAI12M",
            "ALBEDO12M",
            "SNOALB",
            "SOILTEMP",
        ],
    );
    if !missing.is_empty() {
        return Err(StaticError::Missing(format!(
            "baseline static fields missing {}",
            python_list(&missing)
        )));
    }
    let missing = missing_from(
        overrides,
        &[
            "HGT_M", "LANDUSEF", "LANDMASK", "LU_INDEX", "SOILCTOP", "SCT_DOM", "SOILCBOT",
            "SCB_DOM",
        ],
    );
    if !missing.is_empty() {
        return Err(StaticError::Missing(format!(
            "high-resolution overrides missing {}",
            python_list(&missing)
        )));
    }

    let rebuilt_names = [
        "GREENFRAC",
        "LAI12M",
        "ALBEDO12M",
        "SNOALB",
        "SOILTEMP",
        "TMN",
    ];
    let mut out = FieldSet {
        fields: baseline
            .fields
            .iter()
            .filter(|(name, _)| {
                !overrides.fields.contains_key(*name) && !rebuilt_names.contains(&name.as_str())
            })
            .map(|(name, field)| (name.clone(), field.clone()))
            .collect(),
        coverage_reports: baseline.coverage_reports.clone(),
    };
    for (name, field) in &overrides.fields {
        out.fields.insert(name.clone(), field.clone());
    }

    let old_landmask = plane(baseline, "LANDMASK")?;
    let new_landmask = plane(&out, "LANDMASK")?;
    let (ny, nx) = (old_landmask.ny, old_landmask.nx);
    if (new_landmask.ny, new_landmask.nx) != (ny, nx) {
        return Err(invalid(format!(
            "override LANDMASK shape ({}, {}) differs from baseline \
             ({ny}, {nx})",
            new_landmask.ny, new_landmask.nx
        )));
    }
    let old_land: Vec<bool> = old_landmask.data.iter().map(|value| *value > 0.5).collect();
    let mut new_land: Vec<bool> = new_landmask.data.iter().map(|value| *value > 0.5).collect();
    // A baseline with no land cell has no cell to donate climatology to
    // the overlay's new land (see the Python `_water_where_nothing_donates`):
    // those cells keep the baseline's water in every overlay field but
    // the terrain.
    let mut kept_water = 0u64;
    if !old_land.iter().any(|land| *land) && new_land.iter().any(|land| *land) {
        for (name, field) in overrides.fields.iter() {
            if name == "HGT_M" {
                continue;
            }
            let Some(water) = baseline.fields.get(name) else {
                continue;
            };
            let (planes, fny, fnx) = field.dims();
            let same_rank = matches!(
                (field, water),
                (Field::Plane(_), Field::Plane(_)) | (Field::Stack(_), Field::Stack(_))
            );
            if !same_rank || water.dims() != (planes, fny, fnx) || (fny, fnx) != (ny, nx) {
                return Err(invalid(format!(
                    "high-resolution {name} is {} and the baseline's is {}; \
                     the cells no land can donate to cannot keep the \
                     baseline's water",
                    python_shape(field),
                    python_shape(water)
                )));
            }
            let mut data = field.data().to_vec();
            for cell in 0..ny * nx {
                if new_land[cell] {
                    for plane_index in 0..planes {
                        data[plane_index * ny * nx + cell] =
                            water.data()[plane_index * ny * nx + cell];
                    }
                }
            }
            let kept = match field {
                Field::Plane(_) => Field::Plane(Grid2 { ny, nx, data }),
                Field::Stack(_) => Field::Stack(Stack3 {
                    planes,
                    ny,
                    nx,
                    data,
                }),
            };
            out.fields.insert(name.clone(), kept);
        }
        kept_water = new_land.iter().filter(|land| **land).count() as u64;
        new_land = vec![false; ny * nx];
    }
    let newly_land: Vec<bool> = old_land
        .iter()
        .zip(&new_land)
        .map(|(old, new)| *new && !*old)
        .collect();
    let newly_water_count = old_land
        .iter()
        .zip(&new_land)
        .filter(|(old, new)| !**new && **old)
        .count() as u64;
    // Donors are looked for only when a cell needs one.
    let (donor_y, donor_x) = if newly_land.iter().any(|land| *land) {
        nearest_donors(&old_land, ny, nx)?
    } else {
        (Vec::new(), Vec::new())
    };

    let fills: [(&str, f64); 5] = [
        ("GREENFRAC", 0.0),
        ("LAI12M", 0.0),
        ("ALBEDO12M", 8.0),
        ("SNOALB", 0.0),
        // A MASKED SENTINEL, not a temperature (see the Python).
        ("SOILTEMP", f64::NAN),
    ];
    for (name, water_fill) in fills {
        let field = baseline.fields.get(name).unwrap();
        let (planes, fny, fnx) = field.dims();
        if (fny, fnx) != (ny, nx) {
            return Err(invalid(format!(
                "baseline {name} shape ({fny}, {fnx}) differs from \
                 LANDMASK ({ny}, {nx})"
            )));
        }
        let mut data = field.data().to_vec();
        for cell in 0..ny * nx {
            if newly_land[cell] {
                let donor = donor_y[cell] as usize * nx + donor_x[cell] as usize;
                for plane_index in 0..planes {
                    data[plane_index * ny * nx + cell] = data[plane_index * ny * nx + donor];
                }
            }
        }
        for cell in 0..ny * nx {
            if !new_land[cell] {
                for plane_index in 0..planes {
                    data[plane_index * ny * nx + cell] = water_fill;
                }
            }
        }
        let rebuilt = if planes == 1 {
            Field::Plane(Grid2 { ny, nx, data })
        } else {
            Field::Stack(Stack3 {
                planes,
                ny,
                nx,
                data,
            })
        };
        out.fields.insert(name.to_string(), rebuilt);
    }

    let soiltemp = plane(&out, "SOILTEMP")?;
    let hgt = plane(&out, "HGT_M")?;
    if (hgt.ny, hgt.nx) != (ny, nx) {
        return Err(invalid(format!(
            "override HGT_M shape ({}, {}) differs from LANDMASK \
             ({ny}, {nx})",
            hgt.ny, hgt.nx
        )));
    }
    let mut tmn = Grid2 {
        ny,
        nx,
        data: vec![0.0; ny * nx],
    };
    for cell in 0..ny * nx {
        tmn.data[cell] = if new_land[cell] {
            soiltemp.data[cell] - 0.0065 * hgt.data[cell]
        } else {
            soiltemp.data[cell]
        };
    }
    out.fields.insert("TMN".into(), Field::Plane(tmn));

    refuse_unusable_merged_statics(&out, &new_land)?;

    // The sentinel does NOT cross this return (see the Python: geo_em's
    // on-disk convention for a land-masked field is 0.0 over water).
    let mut water_masked = 0u64;
    for name in LAND_ONLY_DEEP_SOIL {
        if let Some(Field::Plane(grid)) = out.fields.get_mut(name) {
            let count = grid
                .data
                .iter_mut()
                .filter(|value| !value.is_finite())
                .map(|value| *value = 0.0)
                .count() as u64;
            water_masked = water_masked.max(count);
        }
    }

    let newly_land_count = newly_land.iter().filter(|ok| **ok).count() as u64;
    let unchanged = old_land
        .iter()
        .zip(&new_land)
        .filter(|(old, new)| old == new)
        .count() as u64;
    Ok((
        out,
        MergeAudit {
            terrain_cells_changed: None,
            land_water_cells_unchanged: None,
            newly_land_nearest_climatology_fallback_cells: newly_land_count,
            newly_water_masked_cells: newly_water_count,
            unchanged_land_water_cells: Some(unchanged),
            deep_soil_water_masked_cells: Some(water_masked),
            newly_land_without_donor_kept_water_cells: Some(kept_water),
        },
    ))
}

// ---------------------------------------------------------------------------
// Terrain override build (`build_terrain_override` compute)
// ---------------------------------------------------------------------------

/// Production terrain build with bounded source decode windows.
pub fn build_terrain_bound(
    spec: &GridSpec,
    terrain: &BoundRasterSpec,
    halo: usize,
    smooth_passes: usize,
) -> Result<Grid2> {
    let extended = extended_spec(spec, halo);
    let warped = resample_continuous_bound(terrain, &extended, Resampling::Average)?;
    finish_terrain_grid(spec, warped, halo, smooth_passes)
}

fn finish_terrain_grid(
    spec: &GridSpec,
    warped: Grid2,
    halo: usize,
    smooth_passes: usize,
) -> Result<Grid2> {
    require_coverage("terrain", &warped.data)?;
    let smoothed = if smooth_passes > 0 {
        crate::smooth::smth_desmth_special(&warped, smooth_passes)?
    } else {
        warped
    };
    let ny = (spec.e_sn - 1) as usize;
    let nx = (spec.e_we - 1) as usize;
    let mut cropped = Grid2 {
        ny,
        nx,
        data: vec![0.0; ny * nx],
    };
    for row in 0..ny {
        let src = (row + halo) * smoothed.nx + halo;
        cropped.data[row * nx..(row + 1) * nx].copy_from_slice(&smoothed.data[src..src + nx]);
    }
    Ok(cropped)
}

// ---------------------------------------------------------------------------
// 1x1-degree tile-id enumeration (from `highres_fetch.py`, so the
// Python fetch driver computes no geography)
// ---------------------------------------------------------------------------

/// `[lat_min, lat_max, lon_min, lon_max]`, the fetch footprint.
pub type BBox = [f64; 4];

/// `three_dep_tile_ids`: `n40w084`-style staged-tile ids.
pub fn three_dep_tile_ids(bbox: BBox) -> Result<Vec<String>> {
    let [lat_min, lat_max, lon_min, lon_max] = bbox;
    if lat_min < 0.0 || lon_max > 0.0 {
        return Err(invalid(format!(
            "3DEP staged 1/3 arc-second tiles are enumerated for the \
             northern/western quadrant only; footprint [{lat_min}, \
             {lat_max}] x [{lon_min}, {lon_max}] leaves it"
        )));
    }
    let mut tiles = Vec::new();
    for north in (lat_min.floor() as i64 + 1)..=(lat_max.ceil() as i64) {
        for west_edge in (lon_min.floor() as i64)..(lon_max.ceil() as i64) {
            tiles.push(format!("n{:02}w{:03}", north, -west_edge));
        }
    }
    Ok(tiles)
}

/// `copernicus_dem_tile_ids`: `N39_00_W105_00`-style south-west ids.
pub fn copernicus_dem_tile_ids(bbox: BBox) -> Result<Vec<String>> {
    let [lat_min, lat_max, lon_min, lon_max] = bbox;
    if lon_max - lon_min > 180.0 {
        return Err(invalid(format!(
            "footprint spans {:.1} degrees of longitude; a footprint \
             that wide is an antimeridian wrap, not a domain, and \
             1x1-degree tile enumeration cannot express it",
            lon_max - lon_min
        )));
    }
    let lat_lo = (lat_min.floor() as i64).max(-90);
    let lat_hi = (lat_max.ceil() as i64).min(90);
    let lon_lo = lon_min.floor() as i64;
    let lon_hi = lon_max.ceil() as i64;
    let mut tiles = Vec::new();
    for lat_sw in lat_lo..lat_hi {
        let ns = if lat_sw >= 0 { 'N' } else { 'S' };
        for lon_sw in lon_lo..lon_hi {
            let wrapped = ((lon_sw + 180).rem_euclid(360)) - 180;
            let ew = if wrapped >= 0 { 'E' } else { 'W' };
            tiles.push(format!(
                "{ns}{:02}_00_{ew}{:03}_00",
                lat_sw.abs(),
                wrapped.abs()
            ));
        }
    }
    if tiles.is_empty() {
        return Err(invalid("footprint enumerates no Copernicus DEM tile"));
    }
    Ok(tiles)
}

/// `srtm_tile_ids`: `N39W105`-style ids off the Copernicus enumerator.
pub fn srtm_tile_ids(bbox: BBox) -> Result<Vec<String>> {
    Ok(copernicus_dem_tile_ids(bbox)?
        .into_iter()
        .map(|tile| tile.replace("_00_", "").trim_end_matches("_00").to_string())
        .collect())
}

/// `one_degree_tile_bbox`: geographic box of one tile id, either
/// naming style.
pub fn one_degree_tile_bbox(tile: &str) -> Result<BBox> {
    let bad = || {
        invalid(format!(
            "tile id {tile:?} is not a 1x1-degree south-west-corner id \
             (expected N39_00_W105_00 or N39W105)"
        ))
    };
    // Full match of `([NS])(\d{2})(?:_00)?_?([EW])(\d{3})(?:_00)?`.
    let mut rest = tile;
    let ns = match rest.chars().next() {
        Some(c @ ('N' | 'S')) => c,
        _ => return Err(bad()),
    };
    rest = &rest[1..];
    if rest.len() < 2 || !rest[..2].bytes().all(|b| b.is_ascii_digit()) {
        return Err(bad());
    }
    let lat: i64 = rest[..2].parse().map_err(|_| bad())?;
    rest = &rest[2..];
    rest = rest.strip_prefix("_00").unwrap_or(rest);
    rest = rest.strip_prefix('_').unwrap_or(rest);
    let ew = match rest.chars().next() {
        Some(c @ ('E' | 'W')) => c,
        _ => return Err(bad()),
    };
    rest = &rest[1..];
    if rest.len() < 3 || !rest[..3].bytes().all(|b| b.is_ascii_digit()) {
        return Err(bad());
    }
    let lon: i64 = rest[..3].parse().map_err(|_| bad())?;
    rest = &rest[3..];
    rest = rest.strip_prefix("_00").unwrap_or(rest);
    if !rest.is_empty() {
        return Err(bad());
    }
    let lat = if ns == 'N' { lat } else { -lat };
    let lon = if ew == 'E' { lon } else { -lon };
    Ok([lat as f64, (lat + 1) as f64, lon as f64, (lon + 1) as f64])
}
