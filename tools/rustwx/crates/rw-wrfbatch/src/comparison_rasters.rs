//! Table-bound public raster analyses. No observation or model names in the reader.
use super::{ObservationField, ObservationProduct, ObservationSpec};
use chrono::{DateTime, TimeZone, Utc};
use rayon::prelude::*;
use serde::Deserialize;
use std::{collections::BTreeMap, io::Read, path::Path};

#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RasterBinding {
    Netcdf {
        variable: String,
        latitude: String,
        longitude: String,
        start_attribute: String,
        stop_attribute: String,
        units: String,
        standard_name: String,
        factor: f64,
    },
    I16HeaderArchive {
        data_member: String,
        header_member: String,
        units: String,
        description: String,
        factor: f64,
        big_endian: bool,
    },
}
fn moment(s: &str) -> Result<DateTime<Utc>, String> {
    chrono::NaiveDateTime::parse_from_str(s.trim().trim_end_matches(" UTC"), "%Y-%m-%d %H:%M:%S")
        .map(|t| t.and_utc())
        .map_err(|e| e.to_string())
}
fn period(
    start: DateTime<Utc>,
    stop: DateTime<Utc>,
    p: &ObservationProduct,
    target: DateTime<Utc>,
) -> Result<(), String> {
    let expected = i64::from(p.accumulation_seconds.unwrap_or(0));
    if !p.time.accepts(stop, target) || (stop - start).num_seconds() != expected {
        return Err(format!(
            "analysis interval {start}..{stop} does not match target {target}, period {expected} s"
        ));
    }
    Ok(())
}
fn gunzip(raw: &[u8]) -> Result<Vec<u8>, String> {
    // Bounded even when a corrupt archive advertises a tiny compressed member.
    let mut out = Vec::new();
    flate2::read::GzDecoder::new(raw)
        .take(256 * 1024 * 1024 + 1)
        .read_to_end(&mut out)
        .map_err(|e| e.to_string())?;
    if out.len() > 256 * 1024 * 1024 {
        return Err("raster gzip exceeds 256 MiB".into());
    }
    Ok(out)
}
fn tar_text(s: &[u8]) -> Result<&str, String> {
    std::str::from_utf8(s.split(|b| *b == 0).next().unwrap_or_default())
        .map(str::trim)
        .map_err(|e| e.to_string())
}
/// Read a regular POSIX tar member without extracting paths or following links.
fn member<'a>(raw: &'a [u8], name: &str) -> Result<&'a [u8], String> {
    let mut offset = 0usize;
    let mut found = None;
    while offset.checked_add(512).is_some_and(|end| end <= raw.len()) {
        let h = &raw[offset..offset + 512];
        if h.iter().all(|b| *b == 0) {
            break;
        }
        let size = usize::from_str_radix(tar_text(&h[124..136])?, 8).map_err(|e| e.to_string())?;
        let checksum =
            usize::from_str_radix(tar_text(&h[148..156])?, 8).map_err(|e| e.to_string())?;
        let actual: usize = h
            .iter()
            .enumerate()
            .map(|(i, &b)| {
                if (148..156).contains(&i) {
                    32
                } else {
                    b as usize
                }
            })
            .sum();
        if checksum != actual {
            return Err("tar header checksum mismatch".into());
        }
        let end = offset
            .checked_add(512)
            .and_then(|n| n.checked_add(size))
            .ok_or("tar size overflow")?;
        if end > raw.len() {
            return Err("truncated raster tar member".into());
        }
        let prefix = tar_text(&h[345..500])?;
        let leaf = tar_text(&h[..100])?;
        let full = if prefix.is_empty() {
            leaf.into()
        } else {
            format!("{prefix}/{leaf}")
        };
        if full == name {
            if ![0, b'0'].contains(&h[156]) || found.is_some() {
                return Err("raster member is duplicated or not regular".into());
            }
            found = Some(&raw[offset + 512..end]);
        }
        offset = offset
            .checked_add(512)
            .and_then(|n| n.checked_add(size.div_ceil(512) * 512))
            .ok_or("tar size overflow")?;
    }
    found.ok_or_else(|| format!("archive has no {name}"))
}
struct Plane {
    values: Vec<f64>,
    lat: Vec<f32>,
    lon: Vec<f32>,
    nx: usize,
    ny: usize,
    valid: DateTime<Utc>,
    factor: f64,
}
fn netcdf(
    path: &Path,
    b: &RasterBinding,
    p: &ObservationProduct,
    target: DateTime<Utc>,
) -> Result<Plane, String> {
    let RasterBinding::Netcdf {
        variable,
        latitude,
        longitude,
        start_attribute,
        stop_attribute,
        units,
        standard_name,
        factor,
    } = b
    else {
        unreachable!()
    };
    let f = netcrust::File::open(path).map_err(|e| e.to_string())?;
    let v = f.variable(variable).ok_or("analysis variable missing")?;
    let attr = |key: &str| {
        v.attribute(key)
            .and_then(|a| a.as_string())
            .map(str::to_owned)
            .ok_or_else(|| format!("missing raster attribute {key}"))
    };
    if attr("units")? != *units || attr("standard_name")? != *standard_name {
        return Err("analysis quantity/units do not match table".into());
    }
    let (start, stop) = (
        moment(&attr(start_attribute)?)?,
        moment(&attr(stop_attribute)?)?,
    );
    period(start, stop, p, target)?;
    let a = f.read_array_f64(variable).map_err(|e| e.to_string())?;
    let [ny, nx] = a.shape() else {
        return Err("analysis must be a 2D plane".into());
    };
    let (nx, ny) = (*nx as usize, *ny as usize);
    let lat = f.read_array_f64(latitude).map_err(|e| e.to_string())?;
    let lon = f.read_array_f64(longitude).map_err(|e| e.to_string())?;
    if lat.shape() != a.shape() || lon.shape() != a.shape() {
        return Err("analysis coordinate dimensions differ".into());
    }
    Ok(Plane {
        values: a.values().to_vec(),
        lat: lat.values().par_iter().map(|x| *x as f32).collect(),
        lon: lon
            .values()
            .par_iter()
            .map(|x| ((*x + 180.).rem_euclid(360.) - 180.) as f32)
            .collect(),
        nx,
        ny,
        valid: stop,
        factor: *factor,
    })
}
fn header_archive(
    raw: &[u8],
    b: &RasterBinding,
    p: &ObservationProduct,
    target: DateTime<Utc>,
) -> Result<Plane, String> {
    let RasterBinding::I16HeaderArchive {
        data_member,
        header_member,
        units,
        description,
        factor,
        big_endian,
    } = b
    else {
        unreachable!()
    };
    let header = gunzip(member(raw, &target.format(header_member).to_string())?)?;
    let text = std::str::from_utf8(&header).map_err(|e| e.to_string())?;
    let rows: BTreeMap<_, _> = text
        .lines()
        .filter_map(|l| l.split_once(':'))
        .map(|(a, b)| (a.trim(), b.trim()))
        .collect();
    let get = |key: &str| {
        rows.get(key)
            .copied()
            .ok_or_else(|| format!("missing raster header {key}"))
    };
    let num = |key: &str| get(key)?.parse::<f64>().map_err(|e| format!("{key}: {e}"));
    if get("Data units")? != units
        || get("Description")? != description
        || get("Data type")? != "integer"
        || num("Data bytes per pixel")? != 2.
        || get("Projected")? != "no"
        || num("Benchmark column")? != 0.
        || num("Benchmark row")? != 0.
    {
        return Err("raster quantity/type/grid differs from table contract".into());
    }
    let time = |prefix: &str| -> Result<DateTime<Utc>, String> {
        let n = |suffix: &str| num(&format!("{prefix} {suffix}"));
        Utc.with_ymd_and_hms(
            n("year")? as i32,
            n("month")? as u32,
            n("day")? as u32,
            n("hour")? as u32,
            n("minute")? as u32,
            n("second")? as u32,
        )
        .single()
        .ok_or("invalid raster header time".into())
    };
    let (start, stop) = (time("Start")?, time("Stop")?);
    period(start, stop, p, target)?;
    let (x, y) = (num("Number of columns")?, num("Number of rows")?);
    if x < 2. || y < 2. || x.fract() != 0. || y.fract() != 0. || x * y > 64_000_000. {
        return Err("invalid raster dimensions".into());
    }
    let (nx, ny) = (x as usize, y as usize);
    let (west, north) = (
        num("Benchmark x-axis coordinate")?,
        num("Benchmark y-axis coordinate")?,
    );
    let (dx, dy) = (num("X-axis resolution")?, num("Y-axis resolution")?);
    if dx <= 0. || dy <= 0. {
        return Err("invalid raster spacing".into());
    }
    let bytes = gunzip(member(raw, &target.format(data_member).to_string())?)?;
    if bytes.len() != nx * ny * 2 {
        return Err("raster byte count differs from header".into());
    }
    let (slope, intercept, missing) = (
        num("Data slope")?,
        num("Data intercept")?,
        num("No data value")?,
    );
    let values = bytes
        .par_chunks_exact(2)
        .map(|b| {
            let v = if *big_endian {
                i16::from_be_bytes([b[0], b[1]])
            } else {
                i16::from_le_bytes([b[0], b[1]])
            } as f64;
            if v == missing {
                f64::NAN
            } else {
                v * slope + intercept
            }
        })
        .collect();
    let lat = (0..nx * ny)
        .into_par_iter()
        .map(|i| (north - (i / nx) as f64 * dy) as f32)
        .collect();
    let lon = (0..nx * ny)
        .into_par_iter()
        .map(|i| (west + (i % nx) as f64 * dx) as f32)
        .collect();
    Ok(Plane {
        values,
        lat,
        lon,
        nx,
        ny,
        valid: stop,
        factor: *factor,
    })
}
pub(super) fn load(
    spec: &ObservationSpec,
    p: &ObservationProduct,
    target: DateTime<Utc>,
    directory: Option<&Path>,
    cache: &Path,
    explicit: Option<&Path>,
    offline: bool,
) -> Result<ObservationField, String> {
    let binding = p.raster.as_ref().ok_or("missing raster binding")?;
    let template = p
        .url_template
        .as_deref()
        .ok_or("raster has no public URL template")?;
    let url = target.format(template).to_string();
    if !url.starts_with("https://") {
        return Err("public raster requires HTTPS".into());
    }
    let name = url.rsplit('/').next().ok_or("invalid raster URL")?;
    let cached = cache.join(&spec.name).join(name);
    let path = explicit
        .map(Path::to_path_buf)
        .or_else(|| directory.map(|d| d.join(name)).filter(|p| p.is_file()))
        .unwrap_or(cached.clone());
    let decode = |path: &Path| -> Result<(Vec<u8>, Plane), String> {
        let raw = std::fs::read(path).map_err(|e| e.to_string())?;
        let plane = match binding {
            RasterBinding::Netcdf { .. } => netcdf(path, binding, p, target)?,
            RasterBinding::I16HeaderArchive { .. } => header_archive(&raw, binding, p, target)?,
        };
        Ok((raw, plane))
    };
    let fetch = || -> Result<(Vec<u8>, Plane), String> {
        if offline || explicit.is_some() {
            return Err(format!(
                "analysis unavailable or invalid: {} (offline={offline})",
                path.display()
            ));
        }
        let agent = rw_nexrad::s3::build_agent();
        let mut response = rw_http::get(&agent, &url).map_err(|e| e.to_string())?;
        let bytes = response
            .body_mut()
            .with_config()
            .limit(512 * 1024 * 1024)
            .read_to_vec()
            .map_err(|e| e.to_string())?;
        std::fs::create_dir_all(path.parent().ok_or("raster path has no parent")?)
            .map_err(|e| e.to_string())?;
        // Validate before publication. A corrupt owned cache is replaced once;
        // explicit user files are never overwritten.
        let tmp = path.with_extension(format!("part-{}", std::process::id()));
        std::fs::write(&tmp, &bytes).map_err(|e| e.to_string())?;
        match decode(&tmp) {
            Ok(result) => {
                std::fs::rename(&tmp, &path).map_err(|e| e.to_string())?;
                Ok(result)
            }
            Err(error) => {
                let _ = std::fs::remove_file(&tmp);
                Err(error)
            }
        }
    };
    let (origin, (raw, plane)) = if path.is_file() {
        match decode(&path) {
            Ok(result) => (path.display().to_string(), result),
            Err(_) if path == cached && !offline && explicit.is_none() => (url.clone(), fetch()?),
            Err(error) => return Err(error),
        }
    } else {
        (url.clone(), fetch()?)
    };
    if !plane.factor.is_finite()
        || plane.factor <= 0.
        || plane
            .lat
            .par_iter()
            .any(|v| !v.is_finite() || v.abs() > 90.)
        || plane
            .lon
            .par_iter()
            .any(|v| !v.is_finite() || v.abs() > 180.)
    {
        return Err("invalid raster coordinates or unit factor".into());
    }
    let values: Vec<f32> = plane
        .values
        .into_par_iter()
        .map(|v| {
            if !v.is_finite() || p.missing_values.contains(&v) {
                f32::NAN
            } else {
                (v * plane.factor) as f32
            }
        })
        .collect();
    let missing_cells = values.par_iter().filter(|v| !v.is_finite()).count();
    let unobserved = values.par_iter().map(|v| !v.is_finite()).collect();
    Ok(ObservationField {
        values,
        lat: plane.lat,
        lon: plane.lon,
        nx: plane.nx,
        ny: plane.ny,
        valid: plane.valid,
        file_name: name.into(),
        origin,
        candidate: p.candidates[0].clone(),
        sha256: rw_nexrad::s3::hex_sha256(&raw),
        missing_cells,
        unobserved,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn tar_octal_checksum_accepts_nul_then_space_terminator() {
        assert_eq!(tar_text(b"007521\0 ").unwrap(), "007521");
        assert_eq!(tar_text(b"00000012345\0").unwrap(), "00000012345");
    }
    #[test]
    fn decoded_period_is_authoritative() {
        let spec = super::super::specifications()
            .unwrap()
            .into_iter()
            .find(|s| s.name == "nohrsc")
            .unwrap();
        let p = &spec.products[0];
        let end = moment("2022-12-24 12:00:00 UTC").unwrap();
        assert!(period(end - chrono::Duration::hours(24), end, p, end).is_ok());
        assert!(period(end - chrono::Duration::hours(12), end, p, end).is_err());
        assert!(
            period(
                end - chrono::Duration::hours(24),
                end,
                p,
                end + chrono::Duration::seconds(1)
            )
            .is_err()
        );
    }
    #[test]
    fn corrupt_archive_is_refused_without_extraction() {
        assert!(member(&[1; 512], "a.dat.gz").is_err());
        assert!(member(&[0; 512], "a.dat.gz").is_err());
    }
}
