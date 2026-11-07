//! Table-driven completed-output field reader. All atmospheric I/O and interpolation
//! are native Rust; Python receives only small metadata and bounded float slabs.
use netcrust::{NcSliceInfo, NcSliceInfoElem};
use rayon::prelude::*;
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    fs,
    io::Write,
    path::{Path, PathBuf},
};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

/// Embed the build revision for release artifact verification (the cut reads it as bytes).
pub static GPUWM_BRIDGE_SOURCE_REV_STAMP: &str =
    concat!("GPUWM_BRIDGE_SOURCE_REV=", env!("GPUWM_BRIDGE_SOURCE_REV"));
fn load(p: &Path) -> Result<Value> {
    Ok(serde_json::from_slice(&fs::read(p)?)?)
}
struct Array {
    dims: Vec<String>,
    shape: Vec<usize>,
    attrs: Value,
}
struct Source {
    path: PathBuf,
    nc: Option<netcrust::File>,
}
impl Source {
    fn open(path: &Path) -> Result<Self> {
        Ok(Self {
            path: path.into(),
            nc: if path.is_dir() {
                None
            } else {
                Some(netcrust::open(path)?)
            },
        })
    }
    fn names(&self) -> Result<Vec<String>> {
        if let Some(n) = &self.nc {
            Ok(n.variables()?.iter().map(|v| v.name().into()).collect())
        } else {
            Ok(fs::read_dir(&self.path)?
                .filter_map(|e| e.ok())
                .filter(|e| e.path().join(".zarray").exists())
                .map(|e| e.file_name().to_string_lossy().into())
                .collect())
        }
    }
    fn meta(&self, name: &str) -> Result<Array> {
        if let Some(n) = &self.nc {
            let v = n.variable(name).ok_or(format!("missing variable {name}"))?;
            Ok(Array {
                dims: v.dimensions().iter().map(|d| d.name().into()).collect(),
                shape: v.shape(),
                attrs: Value::Object(
                    v.attributes()
                        .iter()
                        .map(|a| {
                            (
                                a.name().into(),
                                a.as_string()
                                    .map(|s| json!(s))
                                    .or_else(|| a.as_f64().map(|f| json!(f)))
                                    .unwrap_or(Value::Null),
                            )
                        })
                        .collect(),
                ),
            })
        } else {
            let m = load(&self.path.join(name).join(".zarray"))?;
            let attrs = load(&self.path.join(name).join(".zattrs"))?;
            Ok(Array {
                dims: serde_json::from_value(attrs["_ARRAY_DIMENSIONS"].clone())?,
                shape: serde_json::from_value(m["shape"].clone())?,
                attrs,
            })
        }
    }
    fn read(&self, name: &str, select: &BTreeMap<String, usize>) -> Result<Vec<f64>> {
        let m = self.meta(name)?;
        if let Some(n) = &self.nc {
            let selections = m
                .dims
                .iter()
                .zip(&m.shape)
                .map(|(d, &s)| {
                    if let Some(&i) = select.get(d) {
                        NcSliceInfoElem::Index(i as u64)
                    } else {
                        NcSliceInfoElem::Slice {
                            start: 0,
                            end: s as u64,
                            step: 1,
                        }
                    }
                })
                .collect();
            return Ok(n
                .read_array_f64_slice(name, &NcSliceInfo { selections })?
                .into_values());
        }
        let root = self.path.join(name);
        let zm = load(&root.join(".zarray"))?;
        if zm["order"] != "C" || !zm["filters"].is_null() {
            return Err("Zarr requires C order and no external filters".into());
        }
        let chunks: Vec<usize> = serde_json::from_value(zm["chunks"].clone())?;
        let shape: Vec<usize> = m
            .dims
            .iter()
            .zip(&m.shape)
            .map(|(d, &n)| if select.contains_key(d) { 1 } else { n })
            .collect();
        let count = shape.iter().product::<usize>();
        let dtype = zm["dtype"].as_str().ok_or("missing dtype")?;
        let size = match dtype {
            "<f4" | "<i4" => 4,
            "<f8" | "<i8" => 8,
            "|u1" => 1,
            _ => return Err(format!("unsupported Zarr dtype {dtype}").into()),
        };
        // Read each overlapping chunk once, in parallel, then scatter its selected slab.
        let grids: Vec<usize> = m
            .shape
            .iter()
            .zip(&chunks)
            .zip(&m.dims)
            .map(|((&n, &c), d)| {
                if select.contains_key(d) {
                    1
                } else {
                    n.div_ceil(c)
                }
            })
            .collect();
        let tiles = (0..grids.iter().product())
            .into_par_iter()
            .map(|mut k| -> Result<Vec<(usize, f64)>> {
                let mut ci = vec![0; grids.len()];
                for d in (0..grids.len()).rev() {
                    ci[d] = if let Some(&i) = select.get(&m.dims[d]) {
                        i / chunks[d]
                    } else {
                        let q = k % grids[d];
                        k /= grids[d];
                        q
                    };
                }
                let key = ci
                    .iter()
                    .map(usize::to_string)
                    .collect::<Vec<_>>()
                    .join(".");
                let p = root.join(if key.is_empty() { "0" } else { &key });
                if !p.exists() {
                    return Ok(vec![]);
                }
                let packed = fs::read(p)?;
                let raw = if zm["compressor"].is_null() {
                    packed
                } else if zm["compressor"]["id"] == "blosc" {
                    rw_mlexport::blosc::decompress(&packed)?
                } else {
                    return Err("unsupported Zarr compressor".into());
                };
                if raw.len() != chunks.iter().product::<usize>() * size {
                    return Err("Zarr chunk byte size mismatch".into());
                }
                let mut pairs = vec![];
                for (idx, b) in raw.chunks_exact(size).enumerate() {
                    let mut rem = idx;
                    let mut dest = 0;
                    let mut stride = 1;
                    let mut keep = true;
                    for d in (0..chunks.len()).rev() {
                        let j = ci[d] * chunks[d] + rem % chunks[d];
                        rem /= chunks[d];
                        if let Some(&i) = select.get(&m.dims[d]) {
                            if j != i {
                                keep = false;
                                break;
                            }
                        } else {
                            if j >= m.shape[d] {
                                keep = false;
                                break;
                            }
                            dest += j * stride;
                            stride *= shape[d];
                        }
                    }
                    if keep {
                        let v = match dtype {
                            "<f4" => f32::from_le_bytes(b.try_into()?) as f64,
                            "<f8" => f64::from_le_bytes(b.try_into()?),
                            "<i4" => i32::from_le_bytes(b.try_into()?) as f64,
                            "<i8" => i64::from_le_bytes(b.try_into()?) as f64,
                            _ => b[0] as f64,
                        };
                        pairs.push((dest, v));
                    }
                }
                Ok(pairs)
            })
            .collect::<Result<Vec<_>>>()?;
        let mut out = vec![f64::NAN; count];
        for tile in tiles {
            for (i, v) in tile {
                out[i] = v
            }
        }
        Ok(out)
    }
}
fn inspect(path: &Path) -> Result<Value> {
    if path.is_file() {
        use std::io::Read;
        let mut magic = [0u8; 4];
        fs::File::open(path)?.read_exact(&mut magic)?;
        if &magic == b"GRIB" {
            let grib = grib_core::grib2::Grib2File::from_bytes(&fs::read(path)?)?;
            let f = grib.messages.first().ok_or("empty GRIB2 file")?;
            let scale = match f.product.time_range_unit {
                0 => 60,
                1 => 3600,
                2 => 86400,
                10 => 10800,
                11 => 21600,
                12 => 43200,
                13 => 1,
                _ => return Err("unsupported GRIB2 forecast time unit".into()),
            };
            let seconds = i64::from(f.product.forecast_time) * scale;
            let valid = f.reference_time + chrono::Duration::seconds(seconds);
            if grib.messages.iter().any(|m| {
                m.reference_time != f.reference_time
                    || i64::from(m.product.forecast_time) != i64::from(f.product.forecast_time)
                    || m.product.time_range_unit != f.product.time_range_unit
            }) {
                return Err("single-file GRIB shortcut needs one completed valid time; use a manifest for a series".into());
            }
            return Ok(
                json!({"format":"grib2","init":f.reference_time.and_utc().to_rfc3339_opts(chrono::SecondsFormat::Secs,true),"valid":valid.and_utc().to_rfc3339_opts(chrono::SecondsFormat::Secs,true),"variables":{}}),
            );
        }
    }
    let s = Source::open(path)?;
    let mut vars = serde_json::Map::new();
    for name in s.names()? {
        let m = s.meta(&name)?;
        let mut v = json!({"dims":m.dims,"shape":m.shape,"attrs":m.attrs});
        if m.shape.iter().product::<usize>() <= 4096 && m.dims.len() <= 1 {
            v["values"] = json!(s.read(&name, &BTreeMap::new())?)
        }
        vars.insert(name, v);
    }
    let attrs = if let Some(n) = &s.nc {
        Value::Object(
            n.attributes()?
                .iter()
                .map(|a| {
                    (
                        a.name().into(),
                        a.as_string()
                            .map(|v| json!(v))
                            .or_else(|| a.as_f64().map(|v| json!(v)))
                            .unwrap_or(Value::Null),
                    )
                })
                .collect(),
        )
    } else {
        let p = path.join(".zattrs");
        if p.exists() {
            load(&p)?
        } else {
            json!({})
        }
    };
    Ok(json!({"variables":vars,"attributes":attrs}))
}
fn write_f32(path: &Path, v: &[f32]) -> Result<()> {
    let mut out = std::io::BufWriter::new(fs::File::create(path)?);
    for chunk in v.chunks(65536) {
        let bytes: Vec<_> = chunk.iter().flat_map(|x| x.to_le_bytes()).collect();
        out.write_all(&bytes)?
    }
    Ok(())
}
fn east(v: f64) -> f64 {
    (v + 180.).rem_euclid(360.) - 180.
}
// Invert geographic cells using the same bilinear Newton walk as cyclone regrid.
fn stencil(
    la: f64,
    lo: f64,
    lat: &[f64],
    lon: &[f64],
    ny: usize,
    nx: usize,
    regular: bool,
) -> Option<[(usize, f64); 4]> {
    let periodic =
        regular && ((lon[nx - 1] - lon[0]).abs() + (lon[1] - lon[0]).abs() - 360.).abs() < 1e-4;
    let (mut x, mut y) = if regular {
        let dx = east(lon[1] - lon[0]);
        let mut d = east(lo - lon[0]);
        if periodic && d / dx < 0. {
            d += 360. * dx.signum()
        }
        (d / dx, (la - lat[0]) / (lat[nx] - lat[0]))
    } else {
        (nx as f64 / 2., ny as f64 / 2.)
    };
    for _ in 0..15 {
        if !x.is_finite() || !y.is_finite() {
            return None;
        }
        let i = x.floor().clamp(0., (nx - 2) as f64) as usize;
        let j = y.floor().clamp(0., (ny - 2) as f64) as usize;
        if regular {
            break;
        }
        let (a, b) = (x - i as f64, y - j as f64);
        let k = j * nx + i;
        let sample = |q: &[f64], wrap: bool| {
            let base = q[k];
            let val = |n: usize| if wrap { east(q[n] - base) } else { q[n] - base };
            let (q1, q2, q3) = (val(k + 1), val(k + nx), val(k + nx + 1));
            (
                base + a * (1. - b) * q1 + (1. - a) * b * q2 + a * b * q3,
                (1. - b) * q1 + b * (q3 - q2),
                (1. - a) * q2 + a * (q3 - q1),
            )
        };
        let (l, li, lj) = sample(lat, false);
        let (o, oi, oj) = sample(lon, true);
        let (dl, doo) = (la - l, east(lo - o));
        if dl.abs() < 1e-6 && doo.abs() < 1e-6 {
            break;
        }
        let det = li * oj - lj * oi;
        if det.abs() < 1e-12 {
            return None;
        }
        x += (dl * oj - lj * doo) / det;
        y += (li * doo - dl * oi) / det;
    }
    if y < -1e-5
        || y > (ny - 1) as f64 + 1e-5
        || (!periodic && (x < -1e-5 || x > (nx - 1) as f64 + 1e-5))
    {
        return None;
    }
    let j = (y.floor().max(0.) as usize).min(ny - 2);
    let i = if periodic {
        x.floor().max(0.) as usize % nx
    } else {
        (x.floor().max(0.) as usize).min(nx - 2)
    };
    let i1 = (i + 1) % nx;
    let a = (x - x.floor()).clamp(0., 1.);
    let a = if !periodic { x - i as f64 } else { a };
    let b = y - j as f64;
    let weights = [
        (j * nx + i, (1. - a) * (1. - b)),
        (j * nx + i1, a * (1. - b)),
        ((j + 1) * nx + i, (1. - a) * b),
        ((j + 1) * nx + i1, a * b),
    ];
    if !regular {
        let la_check: f64 = weights.iter().map(|&(k, w)| lat[k] * w).sum();
        let base = lon[weights[0].0];
        let lo_check = base
            + weights
                .iter()
                .map(|&(k, w)| east(lon[k] - base) * w)
                .sum::<f64>();
        if (la_check - la).abs() > 1e-4 || east(lo_check - lo).abs() > 1e-4 {
            return None;
        }
    }
    Some(weights)
}
fn extract(r: &Value) -> Result<()> {
    let path = Path::new(r["path"].as_str().ok_or("path")?);
    let out = Path::new(r["out"].as_str().ok_or("out")?);
    fs::create_dir_all(out)?;
    let vocab = &r["vocabulary"];
    let fmt = vocab["format"].as_str().ok_or("format")?;
    let select: BTreeMap<String, usize> = serde_json::from_value(r["select"].clone())?;
    let names: Vec<String> = serde_json::from_value(r["fields"].clone())?;
    let (lat, lon, ny, nx, regular, planes) = if fmt == "grib2" {
        let bytes = fs::read(path)?;
        let grib = grib_core::grib2::Grib2File::from_bytes(&bytes)?;
        if grib
            .messages
            .iter()
            .any(|m| m.grid.resolution_flags & 8 != 0)
        {
            return Err("GRIB2 has grid-relative winds; export with --winds earth".into());
        }
        let fields = names
            .par_iter()
            .map(|name| -> Result<_> {
                let row = &vocab["fields"][name];
                let sel = serde_json::from_value(row["selector"].clone())?;
                let f = rustwx_io::extract_field_from_bytes(&bytes, sel)?;
                Ok(f)
            })
            .collect::<Result<Vec<_>>>()?;
        let f = &fields[0];
        let ny = f.grid.shape.ny;
        let nx = f.grid.shape.nx;
        let la: Vec<f64> = f.grid.lat_deg.iter().map(|&x| x as f64).collect();
        let lo: Vec<f64> = f.grid.lon_deg.iter().map(|&x| x as f64).collect();
        let regular = (0..ny * nx).all(|i| {
            (la[i] - la[i / nx * nx]).abs() < 1e-5 && (east(lo[i] - lo[i % nx])).abs() < 1e-5
        });
        (
            la,
            lo,
            ny,
            nx,
            regular,
            fields
                .into_iter()
                .map(|f| f.values.into_iter().map(|x| x as f64).collect::<Vec<_>>())
                .collect::<Vec<_>>(),
        )
    } else {
        let s = Source::open(path)?;
        let ln = vocab["lat"].as_str().ok_or("lat name")?;
        let on = vocab["lon"].as_str().ok_or("lon name")?;
        let mut lat = s.read(ln, &select)?;
        let mut lon = s.read(on, &select)?;
        let lm = s.meta(ln)?;
        let regular = lm.dims.len() == 1;
        let (ny, nx) = if regular {
            (lat.len(), lon.len())
        } else {
            let sh: Vec<_> = lm
                .dims
                .iter()
                .zip(lm.shape)
                .filter(|(d, _)| !select.contains_key(*d))
                .map(|(_, n)| n)
                .collect();
            (sh[sh.len() - 2], sh[sh.len() - 1])
        };
        if regular {
            let l = lat.clone();
            lat = (0..ny * nx).map(|i| l[i / nx]).collect();
            let l = lon.clone();
            lon = (0..ny * nx).map(|i| l[i % nx]).collect();
        }
        let planes = names
            .par_iter()
            .map(|name| -> Result<Vec<f64>> {
                let row = &vocab["fields"][name];
                let mut sel = select.clone();
                if let Some(obj) = row["select"].as_object() {
                    for (dim, val) in obj {
                        let coord = s.read(dim, &BTreeMap::new())?;
                        let target = val.as_f64().ok_or("selection value")?;
                        let idx = coord
                            .iter()
                            .position(|x| (x - target).abs() < 1e-6)
                            .ok_or(format!("missing {dim}={target}"))?;
                        sel.insert(dim.clone(), idx);
                    }
                }
                let key = row["name"].as_str().ok_or("field name")?;
                let m = s.meta(key)?;
                if let Some(actual) = m.attrs["units"].as_str() {
                    if actual != row["units"].as_str().unwrap_or("") {
                        return Err(format!(
                            "{key}: units {actual:?} differ from authority {:?}",
                            row["units"]
                        )
                        .into());
                    }
                }
                let v = s.read(key, &sel)?;
                if v.len() != ny * nx {
                    return Err(
                        format!("{key}: expected one {ny}x{nx} plane, got {}", v.len()).into(),
                    );
                }
                let scale = m.attrs["scale_factor"].as_f64().unwrap_or(1.);
                let offset = m.attrs["add_offset"].as_f64().unwrap_or(0.);
                let fill = m.attrs["_FillValue"].as_f64();
                Ok(v.into_iter()
                    .map(|x| {
                        if fill == Some(x) {
                            f64::NAN
                        } else {
                            x * scale + offset
                        }
                    })
                    .collect())
            })
            .collect::<Result<Vec<_>>>()?;
        (lat, lon, ny, nx, regular, planes)
    };
    if nx < 2 || ny < 2 {
        return Err("tracking requires at least two horizontal cells per axis".into());
    }
    let base = lon[0];
    let unwrap = |x: f64| base + east(x - base);
    let south = lat.iter().copied().fold(90., f64::min);
    let north = lat.iter().copied().fold(-90., f64::max);
    let periodic =
        regular && ((lon[nx - 1] - lon[0]).abs() + (lon[1] - lon[0]).abs() - 360.).abs() < 1e-4;
    let west = if periodic {
        0.
    } else {
        lon.iter()
            .copied()
            .map(unwrap)
            .fold(f64::INFINITY, f64::min)
    };
    let east_ = if periodic {
        359.75
    } else {
        lon.iter()
            .copied()
            .map(unwrap)
            .fold(f64::NEG_INFINITY, f64::max)
    };
    let ys: Vec<_> = ((south * 4.).ceil() as i32..=(north * 4.).floor() as i32)
        .rev()
        .map(|x| x as f64 / 4.)
        .collect();
    let xs: Vec<_> = ((west * 4.).ceil() as i32..=(east_ * 4.).floor() as i32)
        .map(|x| x as f64 / 4.)
        .collect();
    if ys.is_empty() || xs.is_empty() {
        return Err("input contains no 0.25-degree lattice points".into());
    }
    let stencils: Vec<_> = (0..ys.len() * xs.len())
        .into_par_iter()
        .map(|k| {
            stencil(
                ys[k / xs.len()],
                xs[k % xs.len()],
                &lat,
                &lon,
                ny,
                nx,
                regular,
            )
        })
        .collect();
    let data = names
        .par_iter()
        .zip(planes)
        .map(|(name, p)| -> Result<Vec<f32>> {
            let row = &vocab["fields"][name];
            let scale = row["scale"].as_f64().ok_or("scale")?;
            let offset = row["offset"].as_f64().ok_or("offset")?;
            Ok(stencils
                .par_iter()
                .map(|s| {
                    s.map(|s| {
                        let mut sum = 0.;
                        for (i, w) in s {
                            if w > 1e-12 {
                                sum += p[i] * w
                            }
                        }
                        (sum * scale + offset) as f32
                    })
                    .unwrap_or(f32::NAN)
                })
                .collect())
        })
        .collect::<Result<Vec<_>>>()?;
    let mask: Vec<_> = (0..stencils.len())
        .into_par_iter()
        .map(|i| {
            if data.iter().all(|p| p[i].is_finite()) {
                1u8
            } else {
                0u8
            }
        })
        .collect();
    for (name, p) in names.iter().zip(&data) {
        write_f32(&out.join(format!("{name}.f32")), p)?
    }
    fs::write(out.join("mask.u8"), mask)?;
    fs::write(
        out.join("grid.json"),
        serde_json::to_vec(
            &json!({"lat":ys,"lon":xs,"shape":[ys.len(),xs.len()],"route":if ys.len()==721&&xs.len()==1440{"global_exact_lattice"}else{"regional_masked_global_lattice"}}),
        )?,
    )?;
    Ok(())
}
fn te(r: &Value) -> Result<()> {
    use netcdf_writer::{AttrValue, NcFormat, NcType, NcWriter, Schema, VarData};
    let frames = r["frames"].as_array().ok_or("frames")?;
    let grid =
        load(&Path::new(frames[0]["directory"].as_str().ok_or("directory")?).join("grid.json"))?;
    let ys: Vec<f64> = serde_json::from_value(grid["lat"].clone())?;
    let xs: Vec<f64> = serde_json::from_value(grid["lon"].clone())?;
    let fields: Vec<String> = serde_json::from_value(r["fields"].clone())?;
    let mut s = Schema::new(NcFormat::Offset64);
    let td = s.def_dim("time", frames.len(), false)?;
    let yd = s.def_dim("lat", ys.len(), false)?;
    let xd = s.def_dim("lon", xs.len(), false)?;
    let t = s.def_var("time", NcType::Double, &[td])?;
    s.put_var_attr(
        t,
        "units",
        AttrValue::Text("hours since 1970-01-01 00:00:00".into()),
    )?;
    let y = s.def_var("lat", NcType::Double, &[yd])?;
    let x = s.def_var("lon", NcType::Double, &[xd])?;
    s.put_var_attr(y, "units", AttrValue::Text("degrees_north".into()))?;
    s.put_var_attr(x, "units", AttrValue::Text("degrees_east".into()))?;
    let mut ids = vec![];
    for f in &fields {
        ids.push(s.def_var(f, NcType::Float, &[td, yd, xd])?)
    }
    let mut w = NcWriter::create(r["out"].as_str().ok_or("out")?, s)?;
    w.write_var(y, VarData::F64(&ys))?;
    w.write_var(x, VarData::F64(&xs))?;
    let times: Vec<f64> = frames
        .iter()
        .map(|f| f["unix_hours"].as_f64().unwrap())
        .collect();
    w.write_var(t, VarData::F64(&times))?;
    for (f, id) in fields.iter().zip(ids) {
        let mut values = vec![];
        for frame in frames {
            let d = Path::new(frame["directory"].as_str().ok_or("directory")?);
            if load(&d.join("grid.json"))? != grid {
                return Err(
                    "TempestExtremes needs a fixed footprint across completed frames".into(),
                );
            }
            let bytes = fs::read(d.join(format!("{f}.f32")))?;
            values.extend(
                bytes
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap())),
            );
        }
        w.write_var(id, VarData::F32(&values))?;
    }
    w.finish()?;
    Ok(())
}
/// The inscribed coordinate box used by storm-centred loop crops. Reads and
/// perimeter reductions stay native, including when several frames render at once.
fn extent(path: &Path) -> Result<Value> {
    let source = Source::open(path)?;
    let meta = source.meta("XLAT")?;
    if meta.shape.len() < 2 { return Err("XLAT needs two horizontal axes".into()); }
    let ny = meta.shape[meta.shape.len()-2];
    let nx = meta.shape[meta.shape.len()-1];
    if nx == 0 || ny == 0 { return Err("empty coordinate grid".into()); }
    let select: BTreeMap<String, usize> = meta.dims[..meta.dims.len()-2]
        .iter().map(|d| (d.clone(), 0)).collect();
    let lat = source.read("XLAT", &select)?;
    let lon = source.read("XLONG", &select)?;
    if lat.len() != nx*ny || lon.len() != nx*ny {
        return Err("coordinate shapes differ".into());
    }
    let west = (0..ny).map(|j| lon[j*nx]).fold(f64::NEG_INFINITY, f64::max);
    let east = (0..ny).map(|j| lon[j*nx+nx-1]).fold(f64::INFINITY, f64::min);
    let south = lat[..nx].iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let north = lat[(ny-1)*nx..].iter().copied().fold(f64::INFINITY, f64::min);
    if ![west,east,south,north].iter().all(|x| x.is_finite()) || west >= east || south >= north {
        return Err("grid has no finite non-wrapping inscribed crop".into());
    }
    Ok(json!([west,east,south,north]))
}

fn main() {
    let _ = std::hint::black_box(GPUWM_BRIDGE_SOURCE_REV_STAMP);
    let args: Vec<_> = std::env::args().collect();
    let result = (|| -> Result<()> {
        match args.get(1).map(String::as_str) {
            Some("--abi") | Some("--help") => println!(
                "gpuwm.tc-fields.v1 inspect extract tempest table-units masked-quarter-degree"
            ),
            Some("inspect") => println!("{}", inspect(Path::new(args.get(2).ok_or("path")?))?),
            Some("extent") => println!("{}", extent(Path::new(args.get(2).ok_or("path")?))?),
            Some("extract") => extract(&load(Path::new(args.get(2).ok_or("request")?))?)?,
            Some("tempest") => te(&load(Path::new(args.get(2).ok_or("request")?))?)?,
            _ => {
                return Err(
                    "usage: rw_tc_fields inspect FILE | extract REQUEST | tempest REQUEST".into(),
                )
            }
        }
        Ok(())
    })();
    if let Err(e) = result {
        eprintln!("rw_tc_fields: {e}");
        std::process::exit(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn curvilinear_interpolation_rejects_uncovered_cells() {
        let (ny, nx) = (4, 5);
        let lat: Vec<f64> = (0..ny * nx)
            .map(|k| (k / nx) as f64 + 0.05 * ((k % nx) as f64).powi(2))
            .collect();
        let lon: Vec<f64> = (0..ny * nx)
            .map(|k| (k % nx) as f64 + 0.03 * ((k / nx) as f64).powi(2))
            .collect();
        let known = [(1, 0.1875), (2, 0.0625), (6, 0.5625), (7, 0.1875)];
        let la: f64 = known.iter().map(|&(k, w)| lat[k] * w).sum();
        let lo: f64 = known.iter().map(|&(k, w)| lon[k] * w).sum();
        let got = stencil(la, lo, &lat, &lon, ny, nx, false).unwrap();
        assert!((got.iter().map(|&(k, w)| lat[k] * w).sum::<f64>() - la).abs() < 1e-5);
        assert!((got.iter().map(|&(k, w)| lon[k] * w).sum::<f64>() - lo).abs() < 1e-5);
        // Inside the rectangular bounds but outside the curved footprint.
        assert!(stencil(0., 4., &lat, &lon, ny, nx, false).is_none());
    }
}
