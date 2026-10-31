//! Generic regular-grid Zarr extraction. Provider names belong in request metadata.
//! zarrs owns codecs/array decoding; the project's NetCDF writer owns serialization and
//! mapped-engine owns the existing WRF humidity derivations.
mod block_ranges;
mod http_store;
mod reach_archive;
mod soiltemp;

use chrono::{DateTime, NaiveDate, NaiveDateTime};
use netcdf_writer::{AttrValue, NcFormat, NcType, NcWriter, Schema, VarData};
use serde::Deserialize;
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fs,
    io::{BufWriter, Write},
    ops::Range,
    path::Path,
    sync::{atomic::{AtomicUsize, Ordering}, Arc, Mutex},
};
use zarrs::{
    array::Array,
    group::Group,
    storage::{ReadableStorage, ReadableStorageTraits},
};

type Result<T> = std::result::Result<T, Box<dyn Error>>;
type SourceArray = Array<dyn ReadableStorageTraits>;
const SCHEMA: &str = "arwen.regular-forcing.v1";
/// How many fields of one record are read from the store at once.
///
/// THE DEFECT THIS EXISTS TO END.  The reads were serial: one field of
/// one hour at a time, each a HEAD and a GET to the store, so a 12 MB
/// request took 146 s at 3 % CPU on a link that was mostly idle
/// (measured 2026-10-17 on the ArWen box).  The fields of a record are
/// independent chunks, so they are read concurrently and written in the
/// request's order, which leaves the output byte for byte what it was.
/// Bounded so a wide request does not open a hundred transfers at once;
/// `GPUWM_RW_ZARR_WORKERS` names another bound.
const DEFAULT_READ_WORKERS: usize = 8;

fn read_workers(fields: usize) -> usize {
    let bound = std::env::var("GPUWM_RW_ZARR_WORKERS")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_READ_WORKERS);
    bound.min(fields).max(1)
}

#[used]
pub static GPUWM_BRIDGE_SOURCE_REV_STAMP: &str =
    concat!("GPUWM_BRIDGE_SOURCE_REV=", env!("GPUWM_BRIDGE_SOURCE_REV"));

#[derive(Deserialize)]
struct Request {
    store: String,
    times: Vec<String>,
    area: [f64; 4], // south, west, north, east
    fields: Vec<Field>,
    expected_levels_hpa: Vec<f64>,
    #[serde(default)]
    derived: Vec<Derived>,
    #[serde(default)]
    specific_humidity_authority: bool,
}
#[derive(Deserialize)]
struct Field {
    source: String,
    output: String,
    units: String,
    pressure: bool,
}
#[derive(Deserialize)]
struct Derived {
    kind: String,
    output: String,
    units: String,
    #[serde(default)]
    inputs: Vec<String>,
}
fn fail<T>(message: impl Into<String>) -> Result<T> {
    Err(message.into().into())
}
fn moment(text: &str, end_of_day: bool) -> Result<NaiveDateTime> {
    if text.len() == 10 {
        let day = NaiveDate::parse_from_str(text, "%Y-%m-%d")?;
        return Ok(day
            .and_hms_opt(
                if end_of_day { 23 } else { 0 },
                if end_of_day { 59 } else { 0 },
                if end_of_day { 59 } else { 0 },
            )
            .unwrap());
    }
    let iso = text.trim().replace(' ', "T");
    if let Ok(t) = DateTime::parse_from_rfc3339(&iso) {
        return Ok(t.naive_utc());
    }
    Ok(NaiveDateTime::parse_from_str(
        iso.trim_end_matches('Z'),
        "%Y-%m-%dT%H:%M:%S%.f",
    )?)
}
fn open(store: &ReadableStorage, name: &str) -> Result<SourceArray> {
    Ok(Array::open(store.clone(), &format!("/{name}"))?)
}
fn dtype(array: &SourceArray) -> Result<String> {
    let metadata = serde_json::to_value(array.metadata())?;
    Ok(metadata
        .get("dtype")
        .or_else(|| metadata.get("data_type"))
        .and_then(Value::as_str)
        .ok_or("array has no supported numeric dtype")?
        .to_owned())
}
fn numbers(array: &SourceArray, ranges: &[Range<u64>]) -> Result<Vec<f64>> {
    let values: Vec<f64> = match dtype(array)?.as_str() {
        "<f4" | ">f4" | "float32" => array
            .retrieve_array_subset::<Vec<f32>>(&ranges)?
            .into_iter()
            .map(f64::from)
            .collect(),
        "<f8" | ">f8" | "float64" => array.retrieve_array_subset::<Vec<f64>>(&ranges)?,
        "<i8" | ">i8" | "int64" => array
            .retrieve_array_subset::<Vec<i64>>(&ranges)?
            .into_iter()
            .map(|x| x as f64)
            .collect(),
        "<i4" | ">i4" | "int32" => array
            .retrieve_array_subset::<Vec<i32>>(&ranges)?
            .into_iter()
            .map(f64::from)
            .collect(),
        other => return fail(format!("unsupported source dtype {other}")),
    };
    unpack(array, values)
}
/// The window `block_ranges` read, as the numbers `numbers` gives for it:
/// its little-endian elements read the way zarrs reads them, then unpacked.
fn numbers_from_le(array: &SourceArray, bytes: &[u8]) -> Result<Vec<f64>> {
    let values: Vec<f64> = match dtype(array)?.as_str() {
        "<f4" | "float32" => bytes.chunks_exact(4).map(|b| f64::from(f32::from_le_bytes(b.try_into().unwrap()))).collect(),
        "<f8" | "float64" => bytes.chunks_exact(8).map(|b| f64::from_le_bytes(b.try_into().unwrap())).collect(),
        "<i8" | "int64" => bytes.chunks_exact(8).map(|b| i64::from_le_bytes(b.try_into().unwrap()) as f64).collect(),
        "<i4" | "int32" => bytes.chunks_exact(4).map(|b| f64::from(i32::from_le_bytes(b.try_into().unwrap()))).collect(),
        other => return fail(format!("unsupported source dtype {other}")),
    };
    unpack(array, values)
}
/// Apply the source's CF packing and missing-value declarations.
fn unpack(array: &SourceArray, mut values: Vec<f64>) -> Result<Vec<f64>> {
    let attrs = array.attributes();
    let number = |name: &str| attrs.get(name).and_then(Value::as_f64);
    let scale = number("scale_factor").unwrap_or(1.0);
    let offset = number("add_offset").unwrap_or(0.0);
    if !scale.is_finite() || scale == 0.0 || !offset.is_finite() {
        return fail("invalid source CF packing transform");
    }
    for value in &mut values {
        if !value.is_finite()
            || number("_FillValue").is_some_and(|f| f == *value)
            || number("missing_value").is_some_and(|f| f == *value)
        {
            *value = f64::NAN;
        } else {
            *value = *value * scale + offset;
        }
    }
    Ok(values)
}
/// The symbol inside a `Name(symbol)` unit spelling, already normalised.
///
/// One string carrying BOTH spellings of one unit is a real upstream
/// convention, not a typo to be matched literally: Google's ARCO ERA5
/// publishes its `level` coordinate as `Hectopascal(hPa)` -- the
/// quantity's written-out name with its symbol in brackets after it.
/// Reduced to `hpa`, it is the same alias every other source of that
/// coordinate already matches, so the alias rows below stay one row per
/// unit instead of gaining a parenthesised twin for every spelling an
/// archive happens to publish.
///
/// A leading name is REQUIRED, so a unit that is itself bracketed --
/// `(0-1)`, which ERA5 uses for fractional fields and which the `1` row
/// carries verbatim -- keeps being read as the whole string it is rather
/// than being silently reduced to its interior.
fn parenthesised_symbol(normal: &str) -> Option<&str> {
    let (name, symbol) = normal.strip_suffix(')')?.split_once('(')?;
    (!name.is_empty() && !symbol.is_empty() && !symbol.contains('(')).then_some(symbol)
}
fn units_match(array: &SourceArray, expected: &str, name: &str) -> Result<()> {
    // Early archive arrays omit some units; the provider's versioned mapping
    // supplies those. When present, source units must agree before relabeling.
    let Some(actual) = array.attributes().get("units") else {
        return Ok(());
    };
    let actual = actual
        .as_str()
        .ok_or("source units attribute must be text")?;
    let normal = actual
        .trim()
        .to_lowercase()
        .replace("**", "")
        .replace(['^', ' '], "");
    let aliases: &[&str] = match expected {
        "K" => &["k", "kelvin"],
        "Pa" => &["pa", "pascal", "pascals"],
        "hPa" => &["hpa", "millibar", "millibars", "mbar", "mb"],
        "J m-2" => &["jm-2", "j/m2"],
        "W m-2" => &["wm-2", "w/m2"],
        "m s-1" => &["ms-1", "m/s"],
        "m2 s-2" => &["m2s-2", "m2/s2"],
        "kg kg-1" => &["kgkg-1", "kg/kg", "1"],
        "m3 m-3" => &["m3m-3", "m3/m3"],
        "kg m-3" => &["kgm-3", "kg/m3"],
        "kg m-1 s-1" => &["kgm-1s-1", "kg/(ms)", "kg/m/s"],
        "kg m-2" => &["kgm-2", "kg/m2"],
        // ECMWF writes "~" for integer class codes (ERA5 soil_type).
        "code" => &["~", "code"],
        "1" => &["1", "dimensionless", "(0-1)", "0-1", "proportion"],
        "m" => &[
            "m",
            "mofwaterequivalent",
            "mwaterequivalent",
            "metres",
            "meters",
        ],
        "degrees_north" => &["degrees_north", "degree_north", "degreesnorth"],
        "degrees_east" => &["degrees_east", "degree_east", "degreeseast"],
        _ => return fail(format!("unsupported declared quantity unit {expected}")),
    };
    let symbol = parenthesised_symbol(&normal);
    if !aliases.contains(&normal.as_str()) && !symbol.is_some_and(|s| aliases.contains(&s)) {
        return fail(format!(
            "{name}: source units {actual:?} disagree with declared units {expected:?}. This quantity reads {aliases:?}, after case folding, removing spaces, ^ and **, and reducing a Name(symbol) spelling to its symbol. Either the request declares the wrong quantity for this array, or the source publishes a spelling the {expected:?} row does not carry yet -- add it to that row rather than relabeling the values."
        ));
    }
    Ok(())
}
/// The one time window a store declares, and the ONE place its boundaries
/// are named.
///
/// A store can publish two stops, and they mean different things.  Google's
/// ARCO ERA5 root attributes carry `valid_time_stop` -- the end of the
/// FINALIZED reanalysis -- and `valid_time_stop_era5t`, the end of the
/// preliminary ERA5T stream, which runs months ahead of it.  Both are true,
/// and a reader told only one of them cannot act on either.
///
/// THE DEFECT THIS TYPE EXISTS TO PREVENT.  The notice and the refusal used
/// to resolve those attributes separately: the notice announced a boundary
/// at the finalized stop while the refusal announced coverage running to the
/// preliminary one, so one store answered "how far does this go" with two
/// different dates and neither sentence said which kind of boundary it had
/// named.  Both sentences are composed here, from one resolved record, so
/// they cannot disagree again and each names the attribute it came from.
struct Coverage {
    start: NaiveDateTime,
    finalized_stop: NaiveDateTime,
    accepted_stop: NaiveDateTime,
    start_text: String,
    finalized_text: String,
    accepted_text: String,
    accepted_attribute: &'static str,
}
impl Coverage {
    fn declared_by(group: &Group<dyn ReadableStorageTraits>) -> Result<Self> {
        let text = |name: &str| {
            group
                .attributes()
                .get(name)
                .and_then(Value::as_str)
                .map(str::to_owned)
        };
        let start_text = text("valid_time_start").ok_or("store lacks valid_time_start authority")?;
        let finalized_text =
            text("valid_time_stop").ok_or("store lacks valid_time_stop authority")?;
        let start = moment(&start_text, false)?;
        let finalized_stop = moment(&finalized_text, true)?;
        let preliminary = text("valid_time_stop_era5t")
            .map(|value| moment(&value, true).map(|stop| (value, stop)))
            .transpose()?;
        // The later of the two is what the store actually serves.  WHICH
        // attribute that was is carried, not recomputed by each reader.
        let (accepted_text, accepted_stop, accepted_attribute) = match preliminary {
            Some((value, stop)) if stop > finalized_stop => (value, stop, "valid_time_stop_era5t"),
            _ => (finalized_text.clone(), finalized_stop, "valid_time_stop"),
        };
        Ok(Self {
            start,
            finalized_stop,
            accepted_stop,
            start_text,
            finalized_text,
            accepted_text,
            accepted_attribute,
        })
    }
    fn admits(&self, moment: &NaiveDateTime) -> bool {
        *moment >= self.start && *moment <= self.accepted_stop
    }
    fn is_preliminary(&self, moment: &NaiveDateTime) -> bool {
        *moment > self.finalized_stop && self.admits(moment)
    }
    /// What the boundary that ADMITTED a time is, for the notice.
    fn preliminary_note(&self) -> String {
        format!(
            "fall after the finalized reanalysis boundary {} (valid_time_stop) and are served as preliminary ERA5T, which this store declares through {} ({}); ERA5T values are replaced when the finalized reanalysis reaches them",
            self.finalized_text, self.accepted_text, self.accepted_attribute
        )
    }
    /// What the boundary that REFUSED a time is, and the way past it.
    fn refusal(&self, moment: &NaiveDateTime) -> String {
        let edge = if *moment < self.start {
            format!("{} is the first hour of the record", self.start_text)
        } else if self.accepted_attribute == "valid_time_stop" {
            format!(
                "{} is the last hour of the finalized reanalysis, and this store declares no preliminary ERA5T stream past it",
                self.accepted_text
            )
        } else {
            format!(
                "{} is the last hour of the preliminary ERA5T stream, which runs ahead of the finalized reanalysis ending {} (valid_time_stop)",
                self.accepted_text, self.finalized_text
            )
        };
        format!(
            "requested time {moment} is outside the coverage this store declares, {} (valid_time_start) through {} ({}). {edge}. Request a time inside that window, or wait for the archive to publish the hour and fetch again.",
            self.start_text, self.accepted_text, self.accepted_attribute
        )
    }
}
fn coordinate(store: &ReadableStorage, name: &str) -> Result<(SourceArray, Vec<f64>)> {
    let array = open(store, name)?;
    if array.shape().len() != 1 {
        return fail(format!("{name} coordinate must be one dimensional"));
    }
    let values = numbers(&array, std::slice::from_ref(&(0..array.shape()[0])))?;
    if values.is_empty() || values.iter().any(|x| !x.is_finite()) {
        return fail(format!("{name} coordinate is empty or nonfinite"));
    }
    Ok((array, values))
}
fn dimensions(array: &SourceArray) -> Result<Vec<String>> {
    let metadata = serde_json::to_value(array.metadata())?;
    let names = array
        .attributes()
        .get("_ARRAY_DIMENSIONS")
        .or_else(|| metadata.get("dimension_names"))
        .and_then(Value::as_array)
        .ok_or("source array lacks explicit dimension names")?;
    names
        .iter()
        .map(|v| {
            v.as_str()
                .map(str::to_owned)
                .ok_or_else(|| "source dimension name is not a string".into())
        })
        .collect()
}
fn regular(values: &[f64], name: &str) -> Result<f64> {
    if values.len() < 2 {
        return fail(format!("{name} needs at least two coordinates"));
    }
    let step = values[1] - values[0];
    if step == 0.0
        || values
            .windows(2)
            .any(|x| ((x[1] - x[0]) - step).abs() > step.abs() * 1e-5)
    {
        return fail(format!("{name} is not a strictly regular coordinate"));
    }
    Ok(step)
}
fn write_field(
    writer: &mut NcWriter,
    record: u64,
    ids: (usize, usize),
    values: &[f64],
) -> Result<()> {
    writer.write_record(record, ids.0, VarData::F64(values))?;
    let missing: Vec<i8> = values.iter().map(|x| i8::from(!x.is_finite())).collect();
    writer.write_record(record, ids.1, VarData::I8(&missing))?;
    Ok(())
}
fn define_field(
    schema: &mut Schema,
    name: &str,
    units: &str,
    source: &str,
    dims: &[usize],
) -> Result<(usize, usize)> {
    let id = schema.def_var(name, NcType::Double, dims)?;
    schema.put_var_attr(id, "units", AttrValue::Text(units.into()))?;
    schema.put_var_attr(id, "source_variable", AttrValue::Text(source.into()))?;
    schema.put_var_attr(id, "arwen_field", AttrValue::Text(name.into()))?;
    let mask = schema.def_var(&format!("missing__{name}"), NcType::Byte, dims)?;
    schema.put_var_attr(mask, "units", AttrValue::Text("1".into()))?;
    schema.put_var_attr(
        id,
        "arwen_missing_mask",
        AttrValue::Text(format!("missing__{name}")),
    )?;
    Ok((id, mask))
}
/// `rw_zarr coverage STORE`: the time window the store declares, as JSON.
/// The hydro initial state asks it before planning a spin-up, so a forecast
/// past the reanalysis's last hour spins up to that hour and carries the
/// state forward (gpuwm.hydro.initial_state) instead of failing the fetch.
fn coverage_main(args: &[String]) -> Result<()> {
    if args.len() != 3 {
        return fail("usage: rw_zarr coverage STORE");
    }
    let store: ReadableStorage = if args[2].starts_with("https://") {
        Arc::new(http_store::HttpStore::new(&args[2])?)
    } else {
        Arc::new(zarrs::filesystem::FilesystemStore::new(&args[2])?)
    };
    let group = Group::open(store, "/")?;
    let c = Coverage::declared_by(&group)?;
    let iso = |m: &NaiveDateTime| m.format("%Y-%m-%dT%H:%M:%SZ").to_string();
    println!("{}", serde_json::to_string(&json!({
        "schema": "gpuwm.zarr-coverage.v1", "store": args[2],
        "start": iso(&c.start), "finalized_stop": iso(&c.finalized_stop),
        "accepted_stop": iso(&c.accepted_stop), "accepted_attribute": c.accepted_attribute,
        "start_text": c.start_text, "finalized_text": c.finalized_text, "accepted_text": c.accepted_text,
    }))?);
    Ok(())
}
fn extract(request: Request, output: &Path) -> Result<Value> {
    if output.exists() {
        return fail(format!("output already exists: {}", output.display()));
    }
    if request.times.is_empty() || request.fields.is_empty() {
        return fail("request needs times and fields");
    }
    let times: Vec<_> = request
        .times
        .iter()
        .map(|t| moment(t, false))
        .collect::<Result<_>>()?;
    if times.windows(2).any(|t| t[0] >= t[1]) {
        return fail("requested times must be unique and increasing");
    }
    let http = if request.store.starts_with("https://") {
        Some(Arc::new(http_store::HttpStore::new(&request.store)?))
    } else {
        None
    };
    let store: ReadableStorage = match &http {
        Some(http) => http.clone(),
        None => Arc::new(zarrs::filesystem::FilesystemStore::new(&request.store)?),
    };
    let group = Group::open(store.clone(), "/")?;
    let coverage = Coverage::declared_by(&group)?;
    let provisional_times: Vec<_> = times
        .iter()
        .filter(|t| coverage.is_preliminary(t))
        .map(ToString::to_string)
        .collect();
    if !provisional_times.is_empty() {
        eprintln!(
            "Zarr: {} requested time(s) {}",
            provisional_times.len(),
            coverage.preliminary_note()
        );
    }
    if let Some(outside) = times.iter().find(|t| !coverage.admits(t)) {
        return fail(coverage.refusal(outside));
    }
    let (time_array, source_time) = coordinate(&store, "time")?;
    let units = time_array
        .attributes()
        .get("units")
        .and_then(Value::as_str)
        .ok_or("time lacks CF units")?;
    if let Some(calendar) = time_array
        .attributes()
        .get("calendar")
        .and_then(Value::as_str)
        && !["standard", "gregorian", "proleptic_gregorian"].contains(&calendar) {
            return fail("unsupported CF calendar");
        }
    let (unit, epoch) = units
        .split_once(" since ")
        .ok_or("unsupported CF time units")?;
    let factor = match unit {
        "hours" | "hour" => 3600.0,
        "seconds" | "second" => 1.0,
        "days" | "day" => 86400.0,
        _ => return fail("unsupported CF time unit"),
    };
    let epoch = moment(epoch, false)?.and_utc().timestamp();
    let mut time_indices = Vec::new();
    for t in &times {
        let expected = (t.and_utc().timestamp() - epoch) as f64 / factor;
        let matches: Vec<_> = source_time
            .iter()
            .enumerate()
            .filter(|(_, x)| (**x - expected).abs() < 1e-8)
            .map(|(i, _)| i as u64)
            .collect();
        if matches.len() != 1 {
            return fail(format!(
                "requested time {t} does not resolve to one source coordinate"
            ));
        }
        time_indices.push(matches[0]);
    }
    let (lat_array, latitude) = coordinate(&store, "latitude")?;
    let (lon_array, longitude) = coordinate(&store, "longitude")?;
    let (level_array, levels) = coordinate(&store, "level")?;
    units_match(&lat_array, "degrees_north", "latitude")?;
    units_match(&lon_array, "degrees_east", "longitude")?;
    units_match(&level_array, "hPa", "level")?;
    let dy = regular(&latitude, "latitude")?.abs();
    let dx = regular(&longitude, "longitude")?;
    if dx <= 0.0 || (dx * longitude.len() as f64 - 360.0).abs() > 1e-4 {
        return fail("longitude must be an increasing global cyclic axis");
    }
    let mut sorted_levels = levels.clone();
    sorted_levels.sort_by(f64::total_cmp);
    let mut expected = request.expected_levels_hpa.clone();
    expected.sort_by(f64::total_cmp);
    if sorted_levels != expected || sorted_levels.windows(2).any(|p| p[0] == p[1]) {
        return fail("source pressure levels differ from the declared complete inventory");
    }
    let mut level_indices: Vec<_> = (0..levels.len()).collect();
    level_indices.sort_by(|a, b| levels[*b].total_cmp(&levels[*a]));
    let out_levels: Vec<_> = level_indices.iter().map(|i| levels[*i]).collect();
    let [south, west, north, east] = request.area;
    if request.area.iter().any(|x| !x.is_finite())
        || south < -90.0
        || north > 90.0
        || south >= north
    {
        return fail("invalid requested area");
    }
    let ys: Vec<_> = latitude
        .iter()
        .enumerate()
        .filter(|(_, y)| **y >= south - dy && **y <= north + dy)
        .map(|(i, _)| i)
        .collect();
    if ys.len() < 2 {
        return fail("requested latitude window has fewer than two source rows");
    }
    let y0 = *ys.first().unwrap();
    let y1 = *ys.last().unwrap() + 1;
    let out_lat = latitude[y0..y1].to_vec();
    if out_lat.iter().copied().fold(f64::INFINITY, f64::min) > south
        || out_lat.iter().copied().fold(f64::NEG_INFINITY, f64::max) < north
    {
        return fail("source latitude does not cover requested area");
    }
    let mut width = (east - west).rem_euclid(360.0);
    if width == 0.0 && east != west {
        width = 360.0;
    }
    if width <= 0.0 {
        return fail("requested longitude span is empty");
    }
    let x_start = ((west - longitude[0]) / dx).floor() as i64 - 1;
    let nx = (((west + width - longitude[0]) / dx).ceil() as i64 + 1 - x_start + 1)
        .min(longitude.len() as i64) as usize;
    let xs: Vec<_> = (0..nx)
        .map(|i| (x_start + i as i64).rem_euclid(longitude.len() as i64) as usize)
        .collect();
    let out_lon: Vec<_> = (0..nx)
        .map(|i| longitude[0] + (x_start + i as i64) as f64 * dx)
        .collect();
    let ny = out_lat.len();
    let plane = ny * nx;
    let mut schema = Schema::new(NcFormat::Offset64);
    schema.put_global_attr("Conventions", AttrValue::Text("CF-1.8".into()))?;
    schema.put_global_attr(
        "arwen_regular_forcing_schema",
        AttrValue::Text(SCHEMA.into()),
    )?;
    schema.put_global_attr(
        "specific_humidity_authority",
        AttrValue::Text(
            if request.specific_humidity_authority {
                "direct"
            } else {
                "unspecified"
            }
            .into(),
        ),
    )?;
    schema.put_global_attr("source_store", AttrValue::Text(request.store.clone()))?;
    schema.put_global_attr(
        "source_valid_time_start",
        AttrValue::Text(coverage.start_text.clone()),
    )?;
    // The FINALIZED boundary, as this attribute has always meant, beside the
    // hour the store actually served this request through.  Two stops with
    // two names, never one stop a reader has to guess the kind of.
    schema.put_global_attr(
        "source_valid_time_stop",
        AttrValue::Text(coverage.finalized_text.clone()),
    )?;
    schema.put_global_attr(
        "source_valid_time_accepted_through",
        AttrValue::Text(coverage.accepted_text.clone()),
    )?;
    schema.put_global_attr(
        "source_valid_time_accepted_attribute",
        AttrValue::Text(coverage.accepted_attribute.into()),
    )?;
    schema.put_global_attr(
        "source_provisional_times",
        AttrValue::Text(serde_json::to_string(&provisional_times)?),
    )?;
    let td = schema.def_dim("time", 0, true)?;
    let pd = schema.def_dim("level", levels.len(), false)?;
    let yd = schema.def_dim("latitude", ny, false)?;
    let xd = schema.def_dim("longitude", nx, false)?;
    let mut coordinates = Vec::new();
    for (name, dim, units) in [
        ("time", td, "seconds since 1970-01-01 00:00:00"),
        ("level", pd, "hPa"),
        ("latitude", yd, "degrees_north"),
        ("longitude", xd, "degrees_east"),
    ] {
        let id = schema.def_var(name, NcType::Double, &[dim])?;
        schema.put_var_attr(id, "units", AttrValue::Text(units.into()))?;
        coordinates.push(id);
    }
    let mut arrays = Vec::new();
    let mut field_ids = BTreeMap::new();
    for field in &request.fields {
        let array = open(&store, &field.source)?;
        units_match(&array, &field.units, &field.source)?;
        let dims = dimensions(&array)?;
        let expected: Vec<&str> = if field.pressure {
            vec!["time", "level", "latitude", "longitude"]
        } else if dims.len() == 2 {
            vec!["latitude", "longitude"]
        } else {
            vec!["time", "latitude", "longitude"]
        };
        if dims.iter().map(String::as_str).collect::<Vec<_>>() != expected {
            return fail(format!(
                "{} has unexpected dimension order {:?}",
                field.source, dims
            ));
        }
        let expected_shape: Vec<u64> = expected
            .iter()
            .map(|d| match *d {
                "time" => source_time.len() as u64,
                "level" => levels.len() as u64,
                "latitude" => latitude.len() as u64,
                _ => longitude.len() as u64,
            })
            .collect();
        if array.shape() != expected_shape {
            return fail(format!(
                "{} shape does not match its coordinate axes",
                field.source
            ));
        }
        let dims = if field.pressure {
            vec![td, pd, yd, xd]
        } else {
            vec![td, yd, xd]
        };
        let ids = define_field(
            &mut schema,
            &field.output,
            &field.units,
            &field.source,
            &dims,
        )?;
        if field_ids.insert(field.output.clone(), ids).is_some() {
            return fail("duplicate output field");
        }
        arrays.push(array);
    }
    for item in &request.derived {
        let dims = match item.kind.as_str() {
            "pressure_from_levels" => vec![td, pd, yd, xd],
            "specific_humidity_from_dewpoint" => vec![td, yd, xd],
            _ => return fail(format!("unknown native derivation {}", item.kind)),
        };
        let ids = define_field(&mut schema, &item.output, &item.units, &item.kind, &dims)?;
        if field_ids.insert(item.output.clone(), ids).is_some() {
            return fail("duplicate derived output field");
        }
    }
    // Over HTTP a window is read by the blosc blocks that hold it
    // (block_ranges.rs); an array that cannot be is read whole, and says so.
    // GPUWM_RW_ZARR_WHOLE_CHUNKS=1 reads every chunk whole, as before.
    let whole_chunks = std::env::var("GPUWM_RW_ZARR_WHOLE_CHUNKS").is_ok_and(|v| v.trim() == "1");
    if http.is_some() && whole_chunks {
        eprintln!("Zarr: GPUWM_RW_ZARR_WHOLE_CHUNKS=1, so every chunk is read whole");
    }
    let layouts: Vec<Option<block_ranges::Layout>> = arrays
        .iter()
        .zip(&request.fields)
        .map(|(array, field)| {
            if http.is_none() || whole_chunks {
                return None;
            }
            block_ranges::layout(array)
                .map_err(|reason| eprintln!("Zarr: {}: {reason}, so its chunks are read whole", field.source))
                .ok()
        })
        .collect();
    let windowed = AtomicUsize::new(0);
    let read_whole = AtomicUsize::new(0);
    let told_whole: Mutex<BTreeSet<String>> = Mutex::new(BTreeSet::new());
    let dependencies: BTreeSet<_> = request
        .derived
        .iter()
        .flat_map(|d| d.inputs.iter().cloned())
        .collect();
    let mut writer = NcWriter::create(output, schema)?;
    writer.write_var(coordinates[1], VarData::F64(&out_levels))?;
    writer.write_var(coordinates[2], VarData::F64(&out_lat))?;
    writer.write_var(coordinates[3], VarData::F64(&out_lon))?;
    for (record, ti) in time_indices.iter().enumerate() {
        writer.write_record(
            record as u64,
            coordinates[0],
            VarData::F64(&[times[record].and_utc().timestamp() as f64]),
        )?;
        let mut retained = BTreeMap::new();
        // One field's chunk, read and cropped to the requested window.
        let read_field = |index: usize| -> std::result::Result<Vec<f64>, String> {
            let (field, array) = (&request.fields[index], &arrays[index]);
            eprintln!(
                "Zarr: {} {}/{} {}",
                times[record],
                record + 1,
                times.len(),
                field.source
            );
            let mut ranges = Vec::new();
            if array.shape().len() > 2 {
                ranges.push(*ti..*ti + 1);
            }
            if field.pressure {
                ranges.push(0..levels.len() as u64);
            }
            ranges.extend([y0 as u64..y1 as u64, 0..longitude.len() as u64]);
            let source = match (&http, &layouts[index]) {
                (Some(http), Some(layout)) => match block_ranges::read(http, array, layout, &ranges)? {
                    block_ranges::Read::Window(bytes) => {
                        windowed.fetch_add(1, Ordering::Relaxed);
                        numbers_from_le(array, &bytes)
                    }
                    block_ranges::Read::Whole(reason) => {
                        let mut told = told_whole.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
                        if told.insert(field.source.clone()) {
                            eprintln!("Zarr: {}: {reason}, so it is read whole", field.source);
                        }
                        drop(told);
                        read_whole.fetch_add(1, Ordering::Relaxed);
                        numbers(array, &ranges)
                    }
                },
                _ => {
                    read_whole.fetch_add(1, Ordering::Relaxed);
                    numbers(array, &ranges)
                }
            }
            .map_err(|e| e.to_string())?;
            let mut values =
                Vec::with_capacity(plane * if field.pressure { levels.len() } else { 1 });
            for zi in if field.pressure {
                level_indices.clone()
            } else {
                vec![0]
            } {
                for yi in 0..ny {
                    for xi in &xs {
                        values.push(source[(zi * ny + yi) * longitude.len() + xi]);
                    }
                }
            }
            Ok(values)
        };
        // The record's fields are read concurrently (DEFAULT_READ_WORKERS)
        // and written below in the request's order.
        type FieldRead = std::result::Result<Vec<f64>, String>;
        let slots: Vec<Mutex<Option<FieldRead>>> =
            (0..request.fields.len()).map(|_| Mutex::new(None)).collect();
        let next = AtomicUsize::new(0);
        std::thread::scope(|scope| {
            for _ in 0..read_workers(request.fields.len()) {
                scope.spawn(|| loop {
                    let index = next.fetch_add(1, Ordering::SeqCst);
                    if index >= request.fields.len() {
                        break;
                    }
                    let result = read_field(index);
                    *slots[index].lock().unwrap_or_else(|poisoned| poisoned.into_inner()) =
                        Some(result);
                });
            }
        });
        for (field, slot) in request.fields.iter().zip(slots) {
            let values = slot
                .into_inner()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .ok_or_else(|| format!("{} was not read", field.source))?
                .map_err(|message| format!("{}: {message}", field.source))?;
            write_field(
                &mut writer,
                record as u64,
                field_ids[&field.output],
                &values,
            )?;
            if dependencies.contains(&field.output) {
                retained.insert(field.output.clone(), values);
            }
        }
        for item in &request.derived {
            let values: Vec<f64> = match item.kind.as_str() {
                "pressure_from_levels" => out_levels
                    .iter()
                    .flat_map(|p| std::iter::repeat_n(*p * 100.0, plane))
                    .collect(),
                "specific_humidity_from_dewpoint" => {
                    if item.inputs.len() != 3 {
                        return fail("dewpoint humidity needs ordered dewpoint, temperature, pressure inputs");
                    }
                    let input: Vec<_> = item
                        .inputs
                        .iter()
                        .map(|n| {
                            retained
                                .get(n)
                                .ok_or_else(|| format!("missing derivation input {n}"))
                        })
                        .collect::<std::result::Result<_, _>>()?;
                    if input
                        .iter()
                        .any(|x| x.len() != plane || x.iter().any(|v| !v.is_finite()))
                    {
                        return fail(
                            "surface humidity inputs must be complete finite surface fields",
                        );
                    }
                    let rh = mapped_engine::derive::surface_relative_humidity(input[0], input[1]);
                    mapped_engine::derive::saturation_mixing_ratio(input[1], input[2], &rh)
                        .into_iter()
                        .map(|r| r / (1.0 + r))
                        .collect()
                }
                _ => unreachable!(),
            };
            write_field(&mut writer, record as u64, field_ids[&item.output], &values)?;
        }
    }
    writer.finish()?;
    let transfer = http.as_ref().map(|http| {
        let (bytes, requests) = http.transfer();
        let (windowed, read_whole) = (windowed.load(Ordering::Relaxed), read_whole.load(Ordering::Relaxed));
        eprintln!(
            "Zarr: received {:.1} MB in {requests} requests: {windowed} field window(s) by blosc block ranges, {read_whole} read whole",
            bytes as f64 / 1e6
        );
        json!({"bytes_received": bytes, "requests": requests,
               "windows_by_block_ranges": windowed, "windows_read_whole": read_whole})
    });
    Ok(
        json!({"schema": SCHEMA, "transfer": transfer, "times": request.times, "levels_hpa":out_levels,
        "latitude_bounds":[out_lat.iter().copied().fold(f64::INFINITY,f64::min),out_lat.iter().copied().fold(f64::NEG_INFINITY,f64::max)],
        "longitude_bounds":[out_lon[0],out_lon[nx-1]], "shape":[times.len(),levels.len(),ny,nx],
        "fields":field_ids.keys().collect::<Vec<_>>(), "provisional_times":provisional_times, "source_metadata":group.attributes()}),
    )
}
fn dump_record(args: &[String]) -> Result<()> {
    if args.len() < 6 { return fail("usage: rw_zarr dump-record FILE INDEX OUTDIR VARIABLE..."); }
    let file = netcrust::File::open(&args[2])?;
    if file.attribute("arwen_regular_forcing_schema").and_then(|a| a.as_string().map(str::to_owned)).as_deref() != Some(SCHEMA) {
        return fail("native record read requires the regular forcing schema");
    }
    let index: u64 = args[3].parse()?;
    let out = Path::new(&args[4]);
    fs::create_dir_all(out)?;
    let mut variables = Vec::new();
    for (slot, name) in args[5..].iter().enumerate() {
        let values = file.read_array_f64_record_or_all(name,index)?;
        let leaf = format!("variable-{slot:04}.bin");
        let mut writer = BufWriter::new(fs::OpenOptions::new().create_new(true).write(true).open(out.join(&leaf))?);
        for value in values.values() { writer.write_all(&value.to_le_bytes())?; }
        writer.flush()?;
        variables.push(json!({"name":name,"file":leaf,"shape":values.shape(),"dtype":"<f8"}));
    }
    println!("{}",json!({"schema":"arwen.regular-forcing-record.v1","index":index,"variables":variables}));
    Ok(())
}
fn run() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    if args.get(1).map(String::as_str) == Some("reach-archive") {
        return reach_archive::run(&args[2..]);
    }
    if args.get(1).map(String::as_str)==Some("dump-record") { return dump_record(&args); }
    if args.get(1).map(String::as_str) == Some("soiltemp-climatology") {
        return soiltemp::run(&args[2..]);
    }
    if args.get(1).map(String::as_str) == Some("coverage") {
        return coverage_main(&args);
    }
    if args.len() != 4 || args[1] != "extract" {
        return fail("usage: rw_zarr extract REQUEST.json OUTPUT.nc | soiltemp-climatology REQUEST.json OUTPUT_DIR");
    }
    let request: Request = serde_json::from_slice(&fs::read(&args[2])?)?;
    let result = extract(request, Path::new(&args[3]))?;
    println!("{}", serde_json::to_string(&result)?);
    Ok(())
}
fn main() {
    let _ = std::hint::black_box(GPUWM_BRIDGE_SOURCE_REV_STAMP);
    if let Err(error) = run() {
        eprintln!("rw_zarr: {error}");
        std::process::exit(1);
    }
}
