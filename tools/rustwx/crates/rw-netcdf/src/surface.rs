//! Metadata-driven prescribed surface analyses. Decode and remap stay in Rust.
use obs_regrid::{apply_plan, build_plan, Method};
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Spec {
    grid: PathBuf,
    table: PathBuf,
    inputs: Vec<Input>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Input {
    source: String,
    path: PathBuf,
    #[serde(default)]
    interval_seconds: Option<i64>,
}
#[derive(Deserialize)]
struct Table {
    schema: String,
    sources: BTreeMap<String, Source>,
}
#[derive(Deserialize)]
struct Source {
    latitude: String,
    longitude: String,
    time: String,
    max_distance_m: f64,
    interval_seconds: i64,
    fields: Vec<Field>,
    #[serde(default)]
    classifies_lakes: bool,
    #[serde(default)]
    coordinates_from_grid: bool,
}
#[derive(Deserialize)]
struct Field {
    variable: String,
    target: String,
    mask: String,
    units: Vec<String>,
    scale: f64,
    offset: f64,
    min: f64,
    max: f64,
    #[serde(default)]
    optional: bool,
}

fn hash(path: &Path) -> Result<String, String> {
    // Streaming hash keeps a public global analysis out of a second RAM copy.
    use std::io::Read;
    let mut file = fs::File::open(path).map_err(|e| e.to_string())?;
    let mut h = Sha256::new();
    let mut buf = [0u8; 65536];
    loop {
        let n = file.read(&mut buf).map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(format!("{:x}", h.finalize()))
}
fn path(base: &Path, value: &Path) -> PathBuf {
    if value.is_absolute() {
        value.to_owned()
    } else {
        base.join(value)
    }
}
fn decode(input: &Path, names: &[String], scratch: &Path) -> Result<Value, String> {
    super::dump_transformed(input, scratch, names, true, true, [1., 0.])?;
    serde_json::from_slice(&fs::read(scratch.join("metadata.json")).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())
}
fn record<'a>(meta: &'a Value, name: &str) -> Result<&'a Value, String> {
    meta["variables"]
        .as_array()
        .unwrap()
        .iter()
        .find(|r| r["name"] == name)
        .ok_or_else(|| format!("missing decoded variable {name}"))
}
fn numbers(meta: &Value, name: &str, scratch: &Path) -> Result<Vec<f64>, String> {
    let r = record(meta, name)?;
    let bytes =
        fs::read(scratch.join(r["filename"].as_str().unwrap())).map_err(|e| e.to_string())?;
    if bytes.len() % 8 != 0 {
        return Err(format!("{name}: nonnumeric coordinate/field"));
    }
    Ok(bytes
        .chunks_exact(8)
        .map(|b| f64::from_le_bytes(b.try_into().unwrap()))
        .collect())
}
fn shape(meta: &Value, name: &str) -> Result<Vec<usize>, String> {
    Ok(record(meta, name)?["shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|n| n.as_u64().unwrap() as usize)
        .collect())
}
fn times(meta: &Value, name: &str, scratch: &Path) -> Result<Vec<String>, String> {
    let r = record(meta, name)?;
    if let Some(values) = r["times"].as_array() {
        return Ok(values
            .iter()
            .map(|t| t.as_str().unwrap().to_owned())
            .collect());
    }
    if r["dtype"] == "|S1" {
        let s = shape(meta, name)?;
        if s.len() != 2 || s[1] != 19 {
            return Err("WRF Times must have shape (Time,19)".into());
        }
        let bytes =
            fs::read(scratch.join(r["filename"].as_str().unwrap())).map_err(|e| e.to_string())?;
        return bytes
            .chunks_exact(19)
            .map(|b| {
                let text = std::str::from_utf8(b).map_err(|e| e.to_string())?;
                chrono::NaiveDateTime::parse_from_str(text, "%Y-%m-%d_%H:%M:%S")
                    .map(|t| t.format("%Y-%m-%dT%H:%M:%SZ").to_string())
                    .map_err(|e| e.to_string())
            })
            .collect();
    }
    Err(format!(
        "{name}: missing CF reference time; guessing a valid time would cycle the wrong analysis"
    ))
}
fn coordinates(
    meta: &Value,
    latname: &str,
    lonname: &str,
    ny: usize,
    nx: usize,
    scratch: &Path,
) -> Result<(Vec<f64>, Vec<f64>), String> {
    let lat = numbers(meta, latname, scratch)?;
    let lon = numbers(meta, lonname, scratch)?;
    let ls = shape(meta, latname)?;
    let xs = shape(meta, lonname)?;
    if ls == [ny] && xs == [nx] {
        return Ok((
            (0..ny)
                .flat_map(|j| std::iter::repeat_n(lat[j], nx))
                .collect(),
            (0..ny).flat_map(|_| lon.iter().copied()).collect(),
        ));
    }
    let n = ny * nx;
    let matches = |s: &[usize]| s.len() >= 2 && s[s.len() - 2..] == [ny, nx];
    if !matches(&ls)
        || !matches(&xs)
        || lat.len() < n
        || lon.len() < n
        || lat.chunks_exact(n).any(|a| a != &lat[..n])
        || lon.chunks_exact(n).any(|a| a != &lon[..n])
    {
        return Err("surface coordinates must be lat(y)/lon(x) or fixed collocated 2-D arrays; drifting grids would reuse the wrong remap".into());
    }
    Ok((lat[..n].to_vec(), lon[..n].to_vec()))
}
fn clear_decode(meta: &Value, scratch: &Path) -> Result<(), String> {
    for r in meta["variables"].as_array().unwrap() {
        fs::remove_file(scratch.join(r["filename"].as_str().unwrap()))
            .map_err(|e| e.to_string())?;
    }
    fs::remove_file(scratch.join("metadata.json")).map_err(|e| e.to_string())?;
    fs::remove_dir(scratch).map_err(|e| e.to_string())
}
fn write_f32(out: &Path, name: &str, values: impl Iterator<Item = f32>) -> Result<String, String> {
    let bytes: Vec<u8> = values.flat_map(f32::to_le_bytes).collect();
    fs::write(out.join(name), bytes).map_err(|e| e.to_string())?;
    hash(&out.join(name))
}

pub fn run(specpath: &Path, out: &Path) -> Result<(), String> {
    let spec: Spec = serde_json::from_slice(&fs::read(specpath).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    let base = specpath.parent().unwrap_or(Path::new("."));
    let grid = path(base, &spec.grid);
    let tablepath = path(base, &spec.table);
    let table: Table = serde_json::from_slice(&fs::read(&tablepath).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    if table.schema != "gpuwm-surface-source-table-v1" || spec.inputs.is_empty() {
        return Err("surface spec needs a v1 source table and at least one input".into());
    }
    fs::create_dir(out)
        .map_err(|e| format!("surface output must be a new owned directory: {e}"))?;
    let scratch = out.join("decode");
    let (file, _) = super::open(&grid)?;
    let identity_keys = [
        "GRID_ID",
        "DX",
        "DY",
        "MAP_PROJ",
        "TRUELAT1",
        "TRUELAT2",
        "STAND_LON",
        "CEN_LAT",
        "CEN_LON",
    ];
    let grid_identity: Vec<_> = identity_keys
        .iter()
        .map(|name| (*name, file.attribute(name).and_then(|a| a.as_f64())))
        .collect();
    let mut names: Vec<String> = ["XLAT", "XLONG", "LANDMASK", "LU_INDEX"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    if file.variable("LAKEMASK").is_some() {
        names.push("LAKEMASK".into());
    }
    let ice_name = ["SEAICE", "XICE"]
        .into_iter()
        .find(|name| file.variable(name).is_some());
    if let Some(name) = ice_name {
        names.push(name.into());
    }
    let islake = file
        .attribute("ISLAKE")
        .and_then(|a| a.as_f64())
        .unwrap_or(-999.);
    let isice = file
        .attribute("ISICE")
        .and_then(|a| a.as_f64())
        .unwrap_or(-999.);
    let iswater = file.attribute("ISWATER").and_then(|a| a.as_f64());
    let meta = decode(&grid, &names, &scratch)?;
    let dims = shape(&meta, "LANDMASK")?;
    if dims.len() < 2 || dims[..dims.len() - 2].iter().any(|d| *d != 1) {
        return Err("destination grid must have one static horizontal plane".into());
    }
    let ny = dims[dims.len() - 2];
    let nx = dims[dims.len() - 1];
    let n = ny * nx;
    let (dlat, dlon) = coordinates(&meta, "XLAT", "XLONG", ny, nx, &scratch)?;
    let land = numbers(&meta, "LANDMASK", &scratch)?;
    let lu = numbers(&meta, "LU_INDEX", &scratch)?;
    let mut lake = if names.iter().any(|s| s == "LAKEMASK") {
        numbers(&meta, "LAKEMASK", &scratch)?
            .iter()
            .map(|v| *v > 0.5)
            .collect::<Vec<_>>()
    } else {
        lu.iter().map(|v| *v == islake).collect::<Vec<_>>()
    };
    if lu.len() != n
        || land.len() != n
        || lake.len() != n
        || land.iter().chain(&lu).any(|v| !v.is_finite())
    {
        return Err("invalid static masks; missing land categories must not become water".into());
    }
    let initial_ice = ice_name
        .map(|name| numbers(&meta, name, &scratch))
        .transpose()?;
    if initial_ice
        .as_ref()
        .is_some_and(|v| v.len() < n || v.len() % n != 0)
    {
        return Err("initial sea-ice mask must have one destination plane".into());
    }
    // ISICE is also the permanent snow/glacier land category. Its index
    // alone must not make glacier land eligible for ocean analysis writes.
    let water: Vec<bool> = (0..n)
        .map(|i| {
            land[i] < 0.5
                || lake[i]
                || (lu[i] == isice
                    && initial_ice
                        .as_ref()
                        .is_some_and(|v| v[i] > 0. && v[i] <= 1.))
        })
        .collect();
    let water_hash = write_f32(
        out,
        "water.f32",
        water.iter().map(|v| if *v { 1. } else { 0. }),
    )?;
    let coordinate_bytes: Vec<u8> = dlat
        .iter()
        .chain(&dlon)
        .flat_map(|v| (*v as f32).to_le_bytes())
        .collect();
    let coordinate_hash = format!("{:x}", Sha256::digest(&coordinate_bytes));
    clear_decode(&meta, &scratch)?;
    let mut records = Vec::new();
    let mut provenance = Vec::new();
    let mut previous = BTreeMap::<(String, String, String), Vec<f32>>::new();
    // A published lake-only product can identify freshwater coverage on an
    // input whose categorical water row merges oceans and the Great Lakes.
    // Resolve that static ownership before filtering any ocean sources.
    let mut inputs: Vec<_> = spec.inputs.iter().collect();
    inputs.sort_by_key(|input| {
        !table
            .sources
            .get(&input.source)
            .is_some_and(|s| s.classifies_lakes)
    });
    for input in inputs {
        let source = table
            .sources
            .get(&input.source)
            .ok_or_else(|| format!("unknown surface table row {}", input.source))?;
        let interval_seconds = input.interval_seconds.unwrap_or(source.interval_seconds);
        if interval_seconds <= 0
            || !source.max_distance_m.is_finite()
            || source.max_distance_m <= 0.
        {
            return Err("surface row needs a positive interval and distance bound".into());
        }
        let src = path(base, &input.path);
        let before_hash = hash(&src)?;
        let (f, _) = super::open(&src)?;
        let latitude_present = f.variable(&source.latitude).is_some();
        let longitude_present = f.variable(&source.longitude).is_some();
        let supplied_coordinates = latitude_present && longitude_present;
        if !supplied_coordinates {
            if latitude_present != longitude_present || !source.coordinates_from_grid {
                return Err("surface input lacks paired coordinates and no aligned-grid policy is declared; remapping would select the wrong cells".into());
            }
            for (name, expected) in &grid_identity {
                let actual = f.attribute(name).and_then(|a| a.as_f64());
                if expected.is_none()
                    || actual.is_none()
                    || actual != *expected
                    || !actual.unwrap().is_finite()
                {
                    return Err(format!(
                        "aligned surface grid attribute {name} differs or is missing; using destination coordinates would apply forcing to the wrong grid"
                    ));
                }
            }
        }
        let mut names = vec![source.time.clone()];
        if supplied_coordinates {
            names.extend([source.latitude.clone(), source.longitude.clone()]);
        }
        let fields: Vec<_> = source
            .fields
            .iter()
            .filter(|r| !r.optional || f.variable(&r.variable).is_some())
            .collect();
        if fields.is_empty() {
            return Err("no declared surface field is present".into());
        }
        names.extend(fields.iter().map(|r| r.variable.clone()));
        names.sort();
        names.dedup();
        let meta = decode(&src, &names, &scratch)?;
        let valid_times = times(&meta, &source.time, &scratch)?;
        if valid_times.is_empty() {
            return Err("surface input has no times".into());
        }
        for r in fields {
            if !["SST", "SEAICE", "VEGFRA", "ALBBCK"].contains(&r.target.as_str())
                || !["water", "ocean", "lake", "all"].contains(&r.mask.as_str())
                || !r.scale.is_finite()
                || !r.offset.is_finite()
                || !r.min.is_finite()
                || !r.max.is_finite()
                || r.min >= r.max
            {
                return Err("invalid surface field target, mask or physical conversion".into());
            }
            let rr = record(&meta, &r.variable)?;
            let units = rr["units"].as_str().unwrap_or("")
                .trim_matches(|c: char| c == '\0' || c.is_ascii_whitespace());
            if !r.units.iter().any(|u| u == units) {
                return Err(format!(
                    "{} units {units:?} do not match table {:?}; refusing a wrong temperature or ice scale",
                    r.variable, r.units
                ));
            }
            let dims = shape(&meta, &r.variable)?;
            if dims.len() < 2 {
                return Err("surface field must have two horizontal dimensions".into());
            }
            let sy = dims[dims.len() - 2];
            let sx = dims[dims.len() - 1];
            let sn = sy * sx;
            let (slat, slon) = if supplied_coordinates {
                coordinates(&meta, &source.latitude, &source.longitude, sy, sx, &scratch)?
            } else {
                if (sy, sx) != (ny, nx) {
                    return Err("aligned surface horizontal shape differs from destination; forcing would reach the wrong cells".into());
                }
                (dlat.clone(), dlon.clone())
            };
            let data = numbers(&meta, &r.variable, &scratch)?;
            if data.len() != sn * valid_times.len() {
                return Err(format!(
                    "{}: field planes and source times disagree; refusing to choose a depth or time silently",
                    r.variable
                ));
            }
            // Do not search past masked source cells: missing remains missing.
            let plan = build_plan(
                Method::Nearest,
                &slat,
                &slon,
                (sy, sx),
                &dlat,
                &dlon,
                (ny, nx),
                source.max_distance_m,
            )
            .map_err(|e| e.to_string())?;
            for (t, values) in data.chunks_exact(sn).enumerate() {
                let values: Vec<f64> = values.iter().map(|v| v * r.scale + r.offset).collect();
                let valid: Vec<bool> = values
                    .iter()
                    .map(|v| v.is_finite() && *v >= r.min && *v <= r.max)
                    .collect();
                let mut remapped = vec![f64::NAN; n];
                let mut usable = vec![false; n];
                apply_plan(
                    Method::Nearest,
                    &plan.source_index,
                    &plan.reachable,
                    (sy, sx),
                    (ny, nx),
                    &values,
                    &valid,
                    &mut remapped,
                    &mut usable,
                )
                .map_err(|e| e.to_string())?;
                if source.classifies_lakes && r.mask != "lake" {
                    return Err("a lake coverage declaration requires lake-owned fields".into());
                }
                for i in 0..n {
                    if source.classifies_lakes && usable[i] && water[i] {
                        lake[i] = true;
                    }
                    usable[i] &= r.mask == "all"
                        || (water[i]
                            && match r.mask.as_str() {
                                "lake" => lake[i],
                                "ocean" => !lake[i],
                                _ => true,
                            });
                    if !usable[i] {
                        remapped[i] = f64::NAN;
                    }
                }
                let filename = format!("{:04}.f32", records.len());
                let stored: Vec<f32> = remapped.iter().map(|v| *v as f32).collect();
                let finite: Vec<f32> = stored.iter().copied().filter(|v| v.is_finite()).collect();
                let mean = if finite.is_empty() {
                    None
                } else {
                    Some(finite.iter().map(|v| f64::from(*v)).sum::<f64>() / finite.len() as f64)
                };
                let key = (input.source.clone(), r.target.clone(), r.mask.clone());
                let changed = previous.get(&key).map(|old| {
                    old.iter()
                        .zip(&stored)
                        .filter(|(a, b)| {
                            a.is_finite() && b.is_finite() && a.to_bits() != b.to_bits()
                        })
                        .count()
                });
                let sha = write_f32(out, &filename, stored.iter().copied())?;
                records.push(json!({"target":r.target,"mask":r.mask,"source":input.source,"valid_time":valid_times[t],"interval_seconds":interval_seconds,"filename":filename,"sha256":sha,"valid_cells":usable.iter().filter(|v|**v).count(),"missing_source_cells":valid.iter().filter(|v|!**v).count(),"max_used_distance_m":plan.max_used_distance_m,"mean":mean,"min":finite.iter().copied().reduce(f32::min),"max":finite.iter().copied().reduce(f32::max),"changed_cells_from_previous_input":changed}));
                previous.insert(key, stored);
            }
        }
        if hash(&src)? != before_hash {
            return Err("surface input changed during decode".into());
        }
        provenance.push(json!({"source":input.source,"path":src,"sha256":before_hash,"coordinate_policy":if supplied_coordinates {"source-latlon"} else {"grid-aligned-validated"}}));
        clear_decode(&meta, &scratch)?;
    }
    let lake_hash = write_f32(
        out,
        "lake.f32",
        lake.iter().map(|v| if *v { 1. } else { 0. }),
    )?;
    let manifest = json!({"schema":"gpuwm-surface-analysis-v1","shape":[ny,nx],"surface_categories":{"water":iswater,"ice":isice},"grid_sha256":hash(&grid)?,"coordinates_sha256":coordinate_hash,"table_sha256":hash(&tablepath)?,"water":{"filename":"water.f32","sha256":water_hash},"lake":{"filename":"lake.f32","sha256":lake_hash},"records":records,"inputs":provenance,"time_policy":"hold-latest-no-lookahead"});
    fs::write(
        out.join("manifest.json"),
        serde_json::to_vec_pretty(&manifest).unwrap(),
    )
    .map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use netcdf_writer::{AttrValue, NcFormat, NcType, NcWriter, Schema, VarData};
    fn fixture(root: &Path, source: &Source, wrong_units: bool) -> PathBuf {
        fixture_with_padding(root, source, wrong_units, "")
    }
    fn fixture_with_padding(root: &Path, source: &Source, wrong_units: bool, padding: &str) -> PathBuf {
        let mut s = Schema::new(NcFormat::Classic);
        let t = s.def_dim("time", 2, false).unwrap();
        let y = s.def_dim("y", 1, false).unwrap();
        let x = s.def_dim("x", 4, false).unwrap();
        let c = s.def_dim("DateStrLen", 19, false).unwrap();
        let a = s
            .def_var(&source.latitude, NcType::Double, &[y, x])
            .unwrap();
        let b = s
            .def_var(&source.longitude, NcType::Double, &[y, x])
            .unwrap();
        let wrf = source.time == "Times";
        let tdims = if wrf { vec![t, c] } else { vec![t] };
        let tv = s
            .def_var(
                &source.time,
                if wrf { NcType::Char } else { NcType::Double },
                &tdims,
            )
            .unwrap();
        if !wrf {
            s.put_var_attr(
                tv,
                "units",
                AttrValue::Text("hours since 2024-01-01 00:00:00".into()),
            )
            .unwrap();
        }
        let mut fields = Vec::new();
        for f in &source.fields {
            let v = s.def_var(&f.variable, NcType::Short, &[t, y, x]).unwrap();
            s.put_var_attr(v, "scale_factor", AttrValue::Doubles(vec![0.1]))
                .unwrap();
            s.put_var_attr(v, "_FillValue", AttrValue::Shorts(vec![-9999]))
                .unwrap();
            s.put_var_attr(
                v,
                "units",
                AttrValue::Text(if wrong_units {
                    "invalid".into()
                } else {
                    format!("{}{}", f.units[0], padding)
                }),
            )
            .unwrap();
            fields.push((v, f));
        }
        let p = root.join("source.nc");
        let mut w = NcWriter::create(&p, s).unwrap();
        w.write_var(a, VarData::F64(&[42.; 4])).unwrap();
        w.write_var(b, VarData::F64(&[0., 1., 2., 3.])).unwrap();
        if wrf {
            w.write_var(tv, VarData::Char(b"2024-01-01_00:00:002024-01-01_01:00:00"))
                .unwrap();
        } else {
            w.write_var(tv, VarData::F64(&[0., 1.])).unwrap();
        }
        for (v, f) in fields {
            let mut values = [0i16; 8];
            for i in 0..8 {
                let converted = if f.target == "SST" {
                    280. + i as f64
                } else {
                    0.25
                };
                values[i] = ((converted - f.offset) / f.scale / 0.1).round() as i16;
            }
            values[3] = -9999;
            w.write_var(v, VarData::I16(&values)).unwrap();
        }
        w.finish().unwrap();
        p
    }
    fn grid(root: &Path) -> PathBuf {
        let mut s = Schema::new(NcFormat::Classic);
        let y = s.def_dim("y", 1, false).unwrap();
        let x = s.def_dim("x", 4, false).unwrap();
        identity_attrs(&mut s, 1.);
        s.put_global_attr("ISLAKE", AttrValue::Ints(vec![21]))
            .unwrap();
        s.put_global_attr("ISICE", AttrValue::Ints(vec![24]))
            .unwrap();
        s.put_global_attr("ISWATER", AttrValue::Ints(vec![16]))
            .unwrap();
        let mut ids = Vec::new();
        for name in ["XLAT", "XLONG", "LANDMASK", "LU_INDEX", "SEAICE"] {
            ids.push(s.def_var(name, NcType::Double, &[y, x]).unwrap());
        }
        let p = root.join("grid.nc");
        let mut w = NcWriter::create(&p, s).unwrap();
        // The last two cells share ISICE, but only the last is iced water.
        for (v, values) in ids.into_iter().zip([
            [42.; 4],
            [0., 1., 2., 3.],
            [0., 0., 1., 1.],
            [16., 21., 24., 24.],
            [0., 0., 0., 0.8],
        ]) {
            w.write_var(v, VarData::F64(&values)).unwrap();
        }
        w.finish().unwrap();
        p
    }
    fn identity_attrs(s: &mut Schema, id: f64) {
        for (name, value) in [
            ("GRID_ID", id),
            ("DX", 3000.),
            ("DY", 3000.),
            ("MAP_PROJ", 1.),
            ("TRUELAT1", 30.),
            ("TRUELAT2", 60.),
            ("STAND_LON", 0.),
            ("CEN_LAT", 42.),
            ("CEN_LON", 1.5),
        ] {
            s.put_global_attr(name, AttrValue::Doubles(vec![value]))
                .unwrap();
        }
    }
    fn auxiliary_without_coordinates(root: &Path, id: f64, nx: usize) -> PathBuf {
        let mut s = Schema::new(NcFormat::Classic);
        if id != 0. {
            identity_attrs(&mut s, id);
        }
        let t = s.def_dim("Time", 2, false).unwrap();
        let y = s.def_dim("south_north", 1, false).unwrap();
        let x = s.def_dim("west_east", nx, false).unwrap();
        let c = s.def_dim("DateStrLen", 19, false).unwrap();
        let tv = s.def_var("Times", NcType::Char, &[t, c]).unwrap();
        let sv = s.def_var("SST", NcType::Float, &[t, y, x]).unwrap();
        s.put_var_attr(sv, "units", AttrValue::Text("K".into()))
            .unwrap();
        let iv = s.def_var("SEAICE", NcType::Float, &[t, y, x]).unwrap();
        let vv = s.def_var("VEGFRA", NcType::Float, &[t, y, x]).unwrap();
        let av = s.def_var("ALBBCK", NcType::Float, &[t, y, x]).unwrap();
        let path = root.join("auxiliary.nc");
        let mut w = NcWriter::create(&path, s).unwrap();
        w.write_var(tv, VarData::Char(b"2024-01-01_00:00:002024-01-01_01:00:00"))
            .unwrap();
        let mut sst = vec![280.; 2 * nx];
        sst[0] = 200.; // native surface_driver owns the 250..350 K decision
        w.write_var(sv, VarData::F32(&sst)).unwrap();
        w.write_var(iv, VarData::F32(&vec![0.; 2 * nx])).unwrap();
        w.write_var(vv, VarData::F32(&vec![50.; 2 * nx])).unwrap();
        w.write_var(av, VarData::F32(&vec![0.2; 2 * nx])).unwrap();
        w.finish().unwrap();
        path
    }
    #[test]
    fn aligned_auxiliary_without_coordinates_verifies_grid_identity() {
        let root = std::env::temp_dir().join(format!("rw-aligned-{}", std::process::id()));
        fs::create_dir_all(&root).unwrap();
        let destination = grid(&root);
        let source = auxiliary_without_coordinates(&root, 1., 4);
        let table = root.join("table.json");
        fs::write(
            &table,
            include_str!("../../../../../woof/data/surface_analyses.json"),
        )
        .unwrap();
        let spec = root.join("spec.json");
        fs::write(&spec,serde_json::to_vec(&json!({
            "grid":destination,"table":table,"inputs":[{"source":"wrf-lower-boundary","path":source}]})).unwrap()).unwrap();
        let out = root.join("valid");
        run(&spec, &out).unwrap();
        let m: Value =
            serde_json::from_slice(&fs::read(out.join("manifest.json")).unwrap()).unwrap();
        assert_eq!(m["records"].as_array().unwrap().len(), 8);
        assert_eq!(
            m["inputs"][0]["coordinate_policy"],
            "grid-aligned-validated"
        );
        for r in m["records"].as_array().unwrap() {
            let bytes = fs::read(out.join(r["filename"].as_str().unwrap())).unwrap();
            let v: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            if r["mask"] == "all" {
                assert!(
                    v[2].is_finite(),
                    "vegetation/albedo also update glacier land"
                );
            } else {
                assert!(v[2].is_nan());
            }
            if r["target"] == "SST" {
                let want = if r["valid_time"] == "2024-01-01T00:00:00Z" {
                    200f32
                } else {
                    280f32
                };
                assert_eq!(v[0].to_bits(), want.to_bits());
            }
        }
        auxiliary_without_coordinates(&root, 2., 4);
        let error = run(&spec, &root.join("wrong-grid")).unwrap_err();
        assert!(error.contains("GRID_ID"), "{error}");
        auxiliary_without_coordinates(&root, 0., 4);
        assert!(run(&spec, &root.join("missing-identity"))
            .unwrap_err()
            .contains("GRID_ID"));
        auxiliary_without_coordinates(&root, 1., 3);
        assert!(run(&spec, &root.join("wrong-shape"))
            .unwrap_err()
            .contains("horizontal shape"));
        fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn every_source_row_decodes_masks_packing_time_and_remaps_without_adapters() {
        let table_text = include_str!("../../../../../woof/data/surface_analyses.json");
        let mut table_value: Value = serde_json::from_str(table_text).unwrap();
        // The arbitrary acceptance test: a new identifier uses an ordinary row.
        table_value["sources"]["new-analysis"] = table_value["sources"]["oisst"].clone();
        table_value["sources"]["new-lake-analysis"] = table_value["sources"]["glsea"].clone();
        let table: Table = serde_json::from_value(table_value.clone()).unwrap();
        for (id, source) in &table.sources {
            let root = std::env::temp_dir().join(format!("rw-surface-{}-{id}", std::process::id()));
            fs::create_dir_all(&root).unwrap();
            let grid = grid(&root);
            let input = fixture(&root, source, false);
            fs::write(
                root.join("table.json"),
                serde_json::to_vec(&table_value).unwrap(),
            )
            .unwrap();
            let spec = json!({"grid":grid,"table":root.join("table.json"),"inputs":[{"source":id,"path":input}]});
            fs::write(root.join("spec.json"), serde_json::to_vec(&spec).unwrap()).unwrap();
            let out = root.join("mapped");
            run(&root.join("spec.json"), &out).unwrap();
            let m: Value =
                serde_json::from_slice(&fs::read(out.join("manifest.json")).unwrap()).unwrap();
            assert_eq!(
                m["records"].as_array().unwrap().len(),
                source.fields.len() * 2
            );
            for r in m["records"].as_array().unwrap() {
                let bytes = fs::read(out.join(r["filename"].as_str().unwrap())).unwrap();
                let v: Vec<f32> = bytes
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                    .collect();
                if r["mask"] == "all" {
                    assert!(v[2].is_finite(), "land field lost glacier coverage: {id}");
                } else {
                    assert!(v[2].is_nan(), "glacier land must stay missing: {id}");
                }
                if r["mask"] == "water" && r["valid_time"] == "2024-01-01T01:00:00Z" {
                    assert!(
                        v[3].is_finite(),
                        "initialized iced water must remain eligible: {id}"
                    );
                }
                if r["mask"] == "lake" {
                    if !source.classifies_lakes {
                        assert!(v[0].is_nan() && v[3].is_nan());
                    } else {
                        assert!(v[0].is_finite());
                    }
                    assert!(v[1].is_finite());
                }
                if r["mask"] == "ocean" {
                    assert!(v[1].is_nan());
                    assert!(v[0].is_finite());
                }
                if r["valid_time"] == "2024-01-01T00:00:00Z" {
                    assert!(
                        v[3].is_nan(),
                        "fill must not become zero or a neighbour's value"
                    );
                }
                if r["target"] == "SEAICE" && v[0].is_finite() {
                    assert!((v[0] - 0.25).abs() < 0.051);
                }
                if r["target"] == "SST" {
                    let i = if r["mask"] == "lake" { 1 } else { 0 };
                    assert!(
                        (v[i]
                            - (280.
                                + i as f32
                                + if r["valid_time"] == "2024-01-01T01:00:00Z" {
                                    4.
                                } else {
                                    0.
                                }))
                        .abs()
                            < 0.051
                    );
                }
            }
            assert!(!out.join("decode").exists());
            println!(
                "row={id}: packing, units, two source times, masks and remapped f32 words checked"
            );
            fixture(&root, source, true);
            assert!(run(&root.join("spec.json"), &root.join("wrong-units"))
                .unwrap_err()
                .contains("wrong temperature or ice scale"));
            fs::remove_dir_all(root).unwrap();
        }
    }

    #[test]
    fn padded_native_surface_units_keep_decoded_words() {
        let root = std::env::temp_dir().join(format!("rw-surface-padding-{}", std::process::id()));
        fs::create_dir_all(&root).unwrap();
        let destination = grid(&root);
        let table_text = include_str!("../../../../../woof/data/surface_analyses.json");
        let table: Table = serde_json::from_str(table_text).unwrap();
        let source = &table.sources["wrf-lower-boundary"];
        let table_path = root.join("table.json");
        fs::write(&table_path, table_text).unwrap();
        let mut reference: Option<Vec<Vec<u8>>> = None;
        for (index, padding) in ["", "\0", " \0\t\r\n"].iter().enumerate() {
            let input_root = root.join(format!("input-{index}"));
            fs::create_dir_all(&input_root).unwrap();
            let input = fixture_with_padding(&input_root, source, false, padding);
            let spec = root.join(format!("spec-{index}.json"));
            fs::write(&spec, serde_json::to_vec(&json!({
                "grid": destination, "table": table_path,
                "inputs": [{"source": "wrf-lower-boundary", "path": input}]
            })).unwrap()).unwrap();
            let out = root.join(format!("mapped-{index}"));
            run(&spec, &out).unwrap();
            let manifest: Value = serde_json::from_slice(&fs::read(out.join("manifest.json")).unwrap()).unwrap();
            let words: Vec<Vec<u8>> = manifest["records"].as_array().unwrap().iter()
                .map(|row| fs::read(out.join(row["filename"].as_str().unwrap())).unwrap()).collect();
            assert!(!words.is_empty());
            if let Some(expected) = &reference { assert_eq!(&words, expected); }
            else { reference = Some(words); }
        }
    }
}
