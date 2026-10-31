//! Native history arrays in atomic, append-only Zarr v3 snapshots.
use anyhow::{Context, Result, bail, ensure};
use bytes::Bytes;
use futures::{StreamExt, TryStreamExt, stream};
use icechunk::{
    Repository, Store,
    format::ByteRange,
    storage::{S3Options, new_local_filesystem_storage, new_s3_storage},
};
use netcdf_reader::{NcAttribute, NcFile, NcSliceInfo, NcSliceInfoElem, NcType};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    io::Write,
    path::{Path, PathBuf},
    sync::Arc,
    time::Instant,
};
use tokio::sync::RwLock;

const ABI: &str = "rw_icechunk --request REQUEST.json schema=icechunk-output.request/v1 modes=run,append,finalize progress=jsonl native-arrays=v1";
const SOURCE_REV: &str = concat!("GPUWM_BRIDGE_SOURCE_REV=", env!("GPUWM_BRIDGE_SOURCE_REV"));
#[derive(Deserialize)]
struct Request {
    schema: String,
    mode: String,
    store: String,
    #[serde(default)]
    inputs: Vec<PathBuf>,
    #[serde(default = "default_chunks")]
    chunks: BTreeMap<String, usize>,
}
fn default_chunks() -> BTreeMap<String, usize> {
    [
        ("time", 1),
        ("level", 4),
        ("y", 256),
        ("x", 256),
        ("cell", 16384),
        ("other", 64),
    ]
    .into_iter()
    .map(|(k, v)| (k.into(), v))
    .collect()
}
#[derive(Default, Serialize, Deserialize)]
struct State {
    times: Vec<i64>,
    digests: Vec<String>,
    arrays: BTreeMap<String, Value>,
    #[serde(default)]
    frames: Vec<Value>,
}
struct Array {
    name: String,
    dims: Vec<String>,
    shape: Vec<usize>,
    dtype: String,
    width: usize,
    attrs: Value,
    bytes: Vec<u8>,
}

fn cpu_seconds() -> Option<f64> {
    #[cfg(unix)]
    {
        let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
        // getrusage initializes the structure only on success.
        if unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) } == 0 {
            let usage = unsafe { usage.assume_init() };
            return Some(
                (usage.ru_utime.tv_sec + usage.ru_stime.tv_sec) as f64
                    + (usage.ru_utime.tv_usec + usage.ru_stime.tv_usec) as f64 / 1e6,
            );
        }
    }
    None
}
fn attributes(attrs: &[NcAttribute]) -> Value {
    let mut out = serde_json::Map::new();
    for a in attrs {
        if let Some(v) = a.value.as_string() {
            out.insert(a.name.clone(), json!(v.trim_end_matches('\0')));
        } else if let Some(v) = a.value.as_f64_vec() {
            out.insert(
                a.name.clone(),
                if v.len() == 1 { json!(v[0]) } else { json!(v) },
            );
        }
    }
    Value::Object(out)
}
fn times(file: &NcFile) -> Result<Vec<i64>> {
    for name in ["Times", "xtime"] {
        if let Ok(v) = file.variable(name) {
            let raw = file.read_variable_char_bytes(name)?;
            let width = v
                .dimensions
                .last()
                .context("time strings have no width")?
                .size as usize;
            ensure!(width > 0, "empty time strings");
            return raw
                .chunks(width)
                .map(|bytes| {
                    let s = std::str::from_utf8(bytes)?.trim_matches(['\0', ' ']);
                    let s = s.replace('_', "T");
                    Ok(
                        chrono::NaiveDateTime::parse_from_str(&s, "%Y-%m-%dT%H:%M:%S")?
                            .and_utc()
                            .timestamp(),
                    )
                })
                .collect();
        }
    }
    bail!("history has neither Times nor xtime; no valid time can be assigned")
}
fn is_time(s: &str) -> bool {
    s.eq_ignore_ascii_case("time")
}
fn read_array(file: &NcFile, v: &netcdf_reader::NcVariable, frame: usize) -> Result<Array> {
    let mut start = vec![0; v.dimensions.len()];
    let mut counts: Vec<usize> = v.dimensions.iter().map(|d| d.size as usize).collect();
    let mut dims = Vec::new();
    let mut shape = Vec::new();
    for (i, d) in v.dimensions.iter().enumerate() {
        if is_time(&d.name) {
            ensure!(i == 0, "{} has non-leading time dimension", v.name);
            start[i] = frame;
            counts[i] = 1;
        } else {
            dims.push(d.name.clone());
            shape.push(d.size as usize);
        }
    }
    macro_rules! numeric {
        ($t:ty,$name:expr) => {{
            let selection = NcSliceInfo {
                selections: start
                    .iter()
                    .zip(&counts)
                    .map(|(s, n)| NcSliceInfoElem::Slice {
                        start: *s as u64,
                        end: (s + n) as u64,
                        step: 1,
                    })
                    .collect(),
            };
            let data = file.read_variable_slice::<$t>(&v.name, &selection)?;
            let bytes = data.iter().flat_map(|x| x.to_le_bytes()).collect();
            ($name.to_string(), std::mem::size_of::<$t>(), bytes)
        }};
    }
    let (dtype, width, bytes) = match v.dtype {
        NcType::Float => numeric!(f32, "float32"),
        NcType::Double => numeric!(f64, "float64"),
        NcType::Int => numeric!(i32, "int32"),
        NcType::UInt => numeric!(u32, "uint32"),
        NcType::Short => numeric!(i16, "int16"),
        NcType::UShort => numeric!(u16, "uint16"),
        NcType::Int64 => numeric!(i64, "int64"),
        NcType::UInt64 => numeric!(u64, "uint64"),
        NcType::Byte => numeric!(i8, "int8"),
        NcType::UByte => numeric!(u8, "uint8"),
        NcType::Char => {
            let all = file.read_variable_char_bytes(&v.name)?;
            let n: usize = shape.iter().product();
            let offset = if v.dimensions.first().is_some_and(|d| is_time(&d.name)) {
                frame * n
            } else {
                0
            };
            ("uint8".into(), 1, all[offset..offset + n].to_vec())
        }
        _ => bail!(
            "{} has unsupported native type {:?}; refusing to omit a history field",
            v.name,
            v.dtype
        ),
    };
    let mut attrs = attributes(&v.attributes);
    if let Some(desc) = attrs.get("description").cloned() {
        attrs
            .as_object_mut()
            .unwrap()
            .entry("long_name")
            .or_insert(desc);
    }
    let standard = match v.name.as_str() {
        "T2" | "temperature" => Some("air_temperature"),
        "theta" => Some("air_potential_temperature"),
        "PSFC" => Some("surface_air_pressure"),
        "pressure" => Some("air_pressure"),
        "HGT" | "ter" => Some("surface_altitude"),
        "TSK" => Some("surface_temperature"),
        "SST" => Some("sea_surface_temperature"),
        _ => None,
    };
    if let Some(standard) = standard {
        attrs
            .as_object_mut()
            .unwrap()
            .entry("standard_name")
            .or_insert(json!(standard));
    }
    // Native WRF metadata uses these coordinate spellings; preserve staggered coordinates.
    let (lat, lon) = if dims.iter().any(|d| d == "west_east_stag") {
        ("XLAT_U", "XLONG_U")
    } else if dims.iter().any(|d| d == "south_north_stag") {
        ("XLAT_V", "XLONG_V")
    } else {
        ("XLAT", "XLONG")
    };
    if dims.iter().any(|d| d.starts_with("south_north"))
        && v.name != lat
        && v.name != lon
        && file.variable(lat).is_ok()
        && file.variable(lon).is_ok()
    {
        attrs["coordinates"] = json!(format!("{lat} {lon}"));
        if projection(file).is_some() {
            attrs["grid_mapping"] = json!("crs");
        }
    }
    if v.name.starts_with("XLAT") {
        attrs["standard_name"] = json!("latitude");
        attrs["units"] = json!("degrees_north");
    }
    if v.name.starts_with("XLONG") {
        attrs["standard_name"] = json!("longitude");
        attrs["units"] = json!("degrees_east");
    }
    if dims.iter().any(|d| d == "nCells")
        && !["latCell", "lonCell"].contains(&v.name.as_str())
        && file.variable("latCell").is_ok()
        && file.variable("lonCell").is_ok()
    {
        attrs["coordinates"] = json!("latCell lonCell");
    }
    Ok(Array {
        name: v.name.clone(),
        dims,
        shape,
        dtype,
        width,
        attrs,
        bytes,
    })
}
fn projection(file: &NcFile) -> Option<Value> {
    let a = attributes(file.global_attributes().ok()?);
    let val = |key: &str| a.get(key).cloned().unwrap_or(json!(0));
    let p = a.get("MAP_PROJ")?.as_f64()? as i32;
    let mut c = match p {
        1 => {
            json!({"grid_mapping_name":"lambert_conformal_conic","standard_parallel":[val("TRUELAT1"),val("TRUELAT2")],"longitude_of_central_meridian":val("STAND_LON"),"latitude_of_projection_origin":val("MOAD_CEN_LAT")})
        }
        2 => {
            json!({"grid_mapping_name":"polar_stereographic","straight_vertical_longitude_from_pole":val("STAND_LON"),"latitude_of_projection_origin":if a.get("TRUELAT1")?.as_f64()? < 0. {-90.}else{90.},"standard_parallel":val("TRUELAT1")})
        }
        3 => {
            json!({"grid_mapping_name":"mercator","longitude_of_projection_origin":val("STAND_LON"),"standard_parallel":val("TRUELAT1")})
        }
        6 => json!({"grid_mapping_name":"latitude_longitude"}),
        _ => return None,
    };
    c["earth_radius"] = json!(6370000.0);
    Some(c)
}
fn chunk_shape(a: &Array, policy: &BTreeMap<String, usize>) -> Vec<usize> {
    let mut c = vec![1];
    for (d, n) in a.dims.iter().zip(&a.shape) {
        let k = if d.starts_with("west_east") {
            "x"
        } else if d.starts_with("south_north") {
            "y"
        } else if ["nCells", "nEdges", "nVertices"].contains(&d.as_str()) {
            "cell"
        } else if d.starts_with("bottom_top") || d.starts_with("nVertLevels") {
            "level"
        } else {
            "other"
        };
        c.push((*n).min(*policy.get(k).unwrap_or(&64)).max(1));
    }
    c
}
fn metadata(a: &Array, n: usize, chunks: &[usize]) -> Value {
    let mut shape = vec![n];
    shape.extend(&a.shape);
    let mut dims = vec!["time".to_string()];
    dims.extend(a.dims.clone());
    json!({"zarr_format":3,"node_type":"array","shape":shape,"data_type":a.dtype,
        "chunk_grid":{"name":"regular","configuration":{"chunk_shape":chunks}},
        "chunk_key_encoding":{"name":"default","configuration":{"separator":"/"}},
        "fill_value":if a.dtype.starts_with("float") {json!("NaN")}else{json!(0)},
        "codecs":[{"name":"bytes","configuration":{"endian":"little"}},{"name":"zstd","configuration":{"level":3,"checksum":true}}],
        "dimension_names":dims,"attributes":a.attrs})
}
fn chunks(a: &Array, time: usize, shape: &[usize]) -> Result<Vec<(String, Bytes)>> {
    let c = &shape[1..];
    let grid: Vec<usize> = a.shape.iter().zip(c).map(|(n, c)| n.div_ceil(*c)).collect();
    let total: usize = grid.iter().product();
    (0..total)
        .into_par_iter()
        .map(|index| {
            let mut at = vec![0; c.len()];
            let mut rem = index;
            for i in (0..c.len()).rev() {
                at[i] = rem % grid[i];
                rem /= grid[i];
            }
            let n: usize = c.iter().product();
            let mut buf = vec![0; n * a.width];
            if a.dtype == "float32" {
                for b in buf.chunks_exact_mut(4) {
                    b.copy_from_slice(&f32::NAN.to_le_bytes());
                }
            }
            if a.dtype == "float64" {
                for b in buf.chunks_exact_mut(8) {
                    b.copy_from_slice(&f64::NAN.to_le_bytes());
                }
            }
            for dst in 0..n {
                let mut rem = dst;
                let mut src = 0;
                let mut stride = 1;
                let mut inside = true;
                for i in (0..c.len()).rev() {
                    let coord = at[i] * c[i] + rem % c[i];
                    rem /= c[i];
                    inside &= coord < a.shape[i];
                    src += coord * stride;
                    stride *= a.shape[i];
                }
                if inside {
                    buf[dst * a.width..(dst + 1) * a.width]
                        .copy_from_slice(&a.bytes[src * a.width..(src + 1) * a.width]);
                }
            }
            let mut key = format!("{}/c/{time}", a.name);
            for x in at {
                key.push_str(&format!("/{x}"));
            }
            let mut encoder = zstd::stream::Encoder::new(Vec::new(), 3)?;
            encoder.include_checksum(true)?;
            encoder.write_all(&buf)?;
            Ok((key, Bytes::from(encoder.finish()?)))
        })
        .collect()
}
async fn repository(url: &str) -> Result<Repository> {
    let storage = if let Some(s) = url.strip_prefix("s3://") {
        let (bucket, prefix) = s
            .split_once('/')
            .context("S3 store needs s3://bucket/prefix")?;
        ensure!(
            !bucket.is_empty() && !prefix.trim_matches('/').is_empty(),
            "S3 bucket and prefix must be nonempty"
        );
        let endpoint = std::env::var("AWS_ENDPOINT_URL_S3")
            .or_else(|_| std::env::var("AWS_ENDPOINT_URL"))
            .ok();
        let mut opts = S3Options::default()
            .with_allow_http(endpoint.as_ref().is_some_and(|s| s.starts_with("http://")))
            .with_force_path_style(endpoint.is_some());
        if let Some(endpoint) = endpoint {
            opts = opts.with_endpoint_url(endpoint);
        }
        new_s3_storage(
            opts,
            bucket.into(),
            Some(prefix.trim_matches('/').into()),
            None,
            Vec::new(),
            Vec::new(),
            None,
        )?
    } else {
        ensure!(
            !url.contains("://") || url.starts_with("file://"),
            "store must be a local path, file:// path, or s3://bucket/prefix"
        );
        new_local_filesystem_storage(Path::new(url.strip_prefix("file://").unwrap_or(url))).await?
    };
    Ok(Repository::open_or_create(None, storage, Default::default(), None, true).await?)
}
async fn run(req: Request) -> Result<()> {
    ensure!(
        req.schema == "icechunk-output.request/v1",
        "unsupported request schema"
    );
    ensure!(
        ["run", "append", "finalize"].contains(&req.mode.as_str()),
        "unknown writer mode"
    );
    ensure!(
        req.chunks.get("time") == Some(&1) && req.chunks.values().all(|n| *n > 0),
        "chunks require time=1 and positive lengths"
    );
    // Filesystem stores do not implement conditional object updates. Serialize
    // their writers across processes, including repository creation and tags.
    let _lock = if !req.store.starts_with("s3://") {
        let root = Path::new(req.store.strip_prefix("file://").unwrap_or(&req.store));
        let parent = root
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        std::fs::create_dir_all(parent)?;
        let name = root
            .file_name()
            .context("store must name a repository directory")?
            .to_string_lossy();
        let lock = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(parent.join(format!(".{name}.writer.lock")))?;
        lock.lock()?;
        Some(lock)
    } else {
        None
    };
    let repo = repository(&req.store).await?;
    let initial = Arc::new(RwLock::new(repo.writable_session("main").await?));
    let store = Store::from_session(initial).await;
    let mut state = if store.exists("zarr.json").await? {
        let root: Value = serde_json::from_slice(&store.get("zarr.json", &ByteRange::ALL).await?)?;
        serde_json::from_value(root["attributes"]["icechunk_output"].clone())
            .context("store is not a native history output repository")?
    } else {
        State::default()
    };
    for path in &req.inputs {
        let file = NcFile::open(path).with_context(|| format!("read {}", path.display()))?;
        let valid = times(&file)?;
        for (frame, t) in valid.into_iter().enumerate() {
            let started = Instant::now();
            let cpu_started = cpu_seconds();
            let mut arrays: Vec<Array> = file
                .variables()?
                .par_iter()
                .map(|v| read_array(&file, v, frame))
                .collect::<Result<_>>()?;
            let native_fields = arrays.len();
            ensure!(
                !arrays.iter().any(|a| a.name == "time" || a.name == "crs"),
                "native variable collides with derived time or crs coordinate"
            );
            arrays.push(Array{name:"time".into(),dims:vec![],shape:vec![],dtype:"int64".into(),width:8,attrs:json!({"standard_name":"time","units":"seconds since 1970-01-01 00:00:00","calendar":"proleptic_gregorian"}),bytes:t.to_le_bytes().to_vec()});
            if let Some(attrs) = projection(&file) {
                arrays.push(Array {
                    name: "crs".into(),
                    dims: vec![],
                    shape: vec![],
                    dtype: "int32".into(),
                    width: 4,
                    attrs,
                    bytes: 0i32.to_le_bytes().to_vec(),
                });
            }
            arrays.sort_by(|a, b| a.name.cmp(&b.name));
            let native_attrs = attributes(file.global_attributes()?);
            let mut hash = Sha256::new();
            hash.update(serde_json::to_vec(&native_attrs)?);
            for a in &arrays {
                hash.update(serde_json::to_vec(&json!([
                    a.name, a.dims, a.shape, a.dtype, a.attrs
                ]))?);
                hash.update(&a.bytes);
            }
            let digest = format!("{:x}", hash.finalize());
            if let Some(i) = state.times.iter().position(|v| *v == t) {
                ensure!(
                    state.digests[i] == digest,
                    "time {t} is already committed with different values or metadata; refusing replacement"
                );
                println!("{}", json!({"event":"kept","time":t,"path":path}));
                continue;
            }
            ensure!(
                state.times.last().is_none_or(|last| t > *last),
                "time {t} precedes the committed tail; refusing out-of-order append"
            );
            let session = Arc::new(RwLock::new(repo.writable_session("main").await?));
            let store = Store::from_session(session.clone()).await;
            let index = state.times.len();
            // Schemas may add or omit fields at subhourly cadence. Every array grows,
            // with absent floating fields filled by NaN in the unwritten time chunk.
            for meta in state.arrays.values_mut() {
                meta["shape"][0] = json!(index + 1);
            }
            let mut prepared = Vec::new();
            for a in &arrays {
                let c = chunk_shape(a, &req.chunks);
                let meta = metadata(a, index + 1, &c);
                if let Some(old) = state.arrays.get(&a.name) {
                    for key in ["shape", "data_type", "dimension_names", "chunk_grid"] {
                        ensure!(
                            old[key] == meta[key],
                            "{} changed {key}; cannot append incompatible grid to this repository",
                            a.name
                        );
                    }
                }
                state.arrays.insert(a.name.clone(), meta);
                prepared.extend(chunks(a, index, &c)?);
            }
            for (name, meta) in &state.arrays {
                store
                    .set(
                        &format!("{name}/zarr.json"),
                        Bytes::from(serde_json::to_vec(meta)?),
                    )
                    .await?;
            }
            stream::iter(prepared.into_iter().map(|(key, bytes)| {
                let store = &store;
                async move { store.set(&key, bytes).await }
            }))
            .buffer_unordered(16)
            .try_collect::<Vec<_>>()
            .await?;
            state.times.push(t);
            state.digests.push(digest);
            state
                .frames
                .push(json!({"time":t,"global_attributes":native_attrs,
                "variables":arrays.iter().map(|a|(&a.name,&a.attrs)).collect::<BTreeMap<_,_>>() }));
            let mut attrs = native_attrs;
            attrs["Conventions"] = json!("CF-1.11");
            attrs["icechunk_output"] = serde_json::to_value(&state)?;
            store
                .set(
                    "zarr.json",
                    Bytes::from(serde_json::to_vec(
                        &json!({"zarr_format":3,"node_type":"group","attributes":attrs}),
                    )?),
                )
                .await?;
            let label = chrono::DateTime::from_timestamp(t, 0)
                .context("invalid timestamp")?
                .to_rfc3339();
            let snapshot = session
                .write()
                .await
                .commit(format!("frame {label}"))
                .execute()
                .await?;
            println!(
                "{}",
                json!({"event":"committed","valid_time":label,"time":t,"snapshot":snapshot.to_string(),"fields":native_fields,"wall_seconds":started.elapsed().as_secs_f64(),"cpu_seconds":cpu_seconds().zip(cpu_started).map(|(end,start)|end-start),"source":path,"store":req.store})
            );
        }
    }
    if req.mode != "append" {
        ensure!(
            !state.times.is_empty(),
            "cannot finalize a store with no frames"
        );
        let snapshot = repo.lookup_branch("main").await?;
        let tag = format!("final-{snapshot}");
        if repo.lookup_tag(&tag).await.is_err() {
            repo.create_tag(&tag, &snapshot).await?;
        }
        println!(
            "{}",
            json!({"event":"finalized","snapshot":snapshot.to_string(),"tag":tag,"frames":state.times.len(),"store":req.store})
        );
    }
    Ok(())
}
#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() == 2 && args[1] == "--abi" {
        println!("{ABI}\n{SOURCE_REV}");
        return;
    }
    let result = async {
        ensure!(args.len() == 3 && args[1] == "--request", "usage: {ABI}");
        let req = serde_json::from_slice(&std::fs::read(&args[2])?)?;
        run(req).await
    }
    .await;
    if let Err(e) = result {
        eprintln!("rw_icechunk: {e:#}");
        std::process::exit(1);
    }
}
