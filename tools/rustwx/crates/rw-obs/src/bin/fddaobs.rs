//! Setup-only native observation policy to an endian-fixed GPU carrier pack.
use rw_obs::{
    fddaobs,
    fddaobs_policy::{Projection, admit},
};
use serde::Deserialize;
use std::{
    error::Error,
    fs,
    io::{BufWriter, Write},
    path::PathBuf,
};
#[derive(Deserialize)]
struct Request {
    input: PathBuf,
    output: PathBuf,
    projection: Projection,
    original_start_unix_seconds: i64,
    rovg: f32,
    #[serde(default)]
    window: Option<Window>,
    #[serde(default = "workers")]
    workers: usize,
}
#[derive(Deserialize)]
struct Window {
    capacity: usize,
    begin: f32,
    end: f32,
    hours: f32,
    dt_minutes: f32,
    interval: usize,
    #[serde(default)]
    adaptive: bool,
}
fn workers() -> usize {
    std::thread::available_parallelism().map_or(1, usize::from)
}
pub static GPUWM_BRIDGE_SOURCE_REV_STAMP: &str =
    concat!("GPUWM_BRIDGE_SOURCE_REV=", env!("GPUWM_BRIDGE_SOURCE_REV"));
fn main() -> Result<(), Box<dyn Error>> {
    let _ = std::hint::black_box(GPUWM_BRIDGE_SOURCE_REV_STAMP);
    let args: Vec<_> = std::env::args_os().collect();
    if args.len() == 2 && args[1] == "--abi" {
        println!("gpuwm-fddaobs-pack-v1");
        return Ok(());
    }
    if args.len() == 3 && args[1] == "--count-rows" {
        let input = PathBuf::from(&args[2]);
        let name = input
            .file_name()
            .and_then(|s| s.to_str())
            .ok_or("invalid OBS filename")?;
        let multi = name.starts_with("OBS_DOMAIN") && name.ends_with("01");
        let prefix = &name[..name.len().saturating_sub(2)];
        let mut rows = 0usize;
        for volume in 1..=99 {
            let path = if volume == 1 {
                input.clone()
            } else {
                input.with_file_name(format!("{prefix}{volume:02}"))
            };
            if volume > 1 && (!multi || !path.is_file()) {
                break;
            }
            rows += fddaobs::decode(&fs::read(path)?, usize::MAX, workers())?
                .iter()
                .map(|r| r.levels())
                .sum::<usize>();
        }
        println!("{rows}");
        return Ok(());
    }
    if args.len() != 2 {
        return Err("usage: rw_fddaobs REQUEST.json".into());
    }
    let req: Request = serde_json::from_slice(&fs::read(&args[1])?)?;
    let mut bytes = Vec::new();
    let mut reports = Vec::new();
    let mut sources = Vec::new();
    // Native in4dob increments the two-digit volume number at EOF.
    let name = req
        .input
        .file_name()
        .and_then(|s| s.to_str())
        .ok_or("OBS_DOMAIN filename must be UTF-8")?;
    let multi = name.starts_with("OBS_DOMAIN") && name.ends_with("01");
    let prefix = &name[..name.len().saturating_sub(2)];
    for volume in 1..=99 {
        let path = if volume == 1 {
            req.input.clone()
        } else {
            req.input.with_file_name(format!("{prefix}{volume:02}"))
        };
        if volume > 1 && (!multi || !path.is_file()) {
            break;
        }
        let data = fs::read(&path)?;
        reports.extend(fddaobs::decode(&data, usize::MAX, req.workers)?);
        sources.push(serde_json::json!({"volume":volume,"sha256":rw_obs::hex_sha256(&data),"bytes":data.len()}));
        bytes.extend_from_slice(&data);
    }
    let rows = admit(
        &reports,
        &req.projection,
        req.original_start_unix_seconds,
        req.rovg,
        req.workers,
    )?;
    if let Some(w) = &req.window {
        if w.interval == 0 || w.dt_minutes <= 0. || w.hours <= 0. || !w.end.is_finite() {
            return Err("invalid observation window clock".into());
        }
        let first = (w.begin.max(0.) / (w.dt_minutes * w.interval as f32)).ceil() as usize;
        let last = (w.end.max(0.) / (w.dt_minutes * w.interval as f32)).floor() as usize;
        if w.adaptive {
            for time in std::iter::once(w.begin / 60.).chain(rows.iter().map(|r| r.time - w.hours))
            {
                if time < w.begin / 60. || time > w.end / 60. {
                    continue;
                }
                let lo = rows.partition_point(|r| r.time < time - w.hours);
                let mut hi = rows.partition_point(|r| r.time <= time + w.hours);
                if hi < rows.len() {
                    hi = (hi + (rows[hi].count - rows[hi].level) as usize + 1).min(rows.len());
                }
                if hi - lo > w.capacity {
                    return Err(format!("WRF in4dob STOP 122: adaptive observation window needs {} rows, max_obs is {}",hi-lo,w.capacity).into());
                }
            }
        }
        for tick in first..=last {
            let time = (tick * w.interval) as f32 * w.dt_minutes;
            let back = time / 60. - w.hours;
            let forward = time / 60. + w.hours;
            let lo = rows.partition_point(|r| r.time < back);
            let mut hi = rows.partition_point(|r| r.time <= forward);
            if hi < rows.len() {
                hi = (hi + (rows[hi].count - rows[hi].level) as usize + 1).min(rows.len());
            }
            if hi - lo > w.capacity {
                return Err(format!("WRF in4dob STOP 122: observation window at {time} minutes needs {} rows, max_obs is {}",hi-lo,w.capacity).into());
            }
        }
    }
    let metadata = serde_json::to_vec(
        &serde_json::json!({"schema":"gpuwm-fddaobs-pack-v1","source_sha256":rw_obs::hex_sha256(&bytes),"rows":rows.len(),"reports":reports.len(),"original_start_unix_seconds":req.original_start_unix_seconds,"projection":req.projection,"workers":req.workers,"sources":sources}),
    )?;
    let mut out = BufWriter::new(fs::File::create(&req.output)?);
    out.write_all(b"GPWMOBS1")?;
    out.write_all(&(metadata.len() as u32).to_le_bytes())?;
    out.write_all(&metadata)?;
    // Structure of arrays: rio,rjo,rko,time,nlev,lev,platform,elevation,varobs.
    for field in 0..8 {
        for r in &rows {
            let v = match field {
                0 => r.i,
                1 => r.j,
                2 => r.k,
                3 => r.time,
                4 => r.count,
                5 => r.level,
                6 => r.platform,
                _ => r.elevation,
            };
            out.write_all(&v.to_le_bytes())?;
        }
    }
    for r in &rows {
        for v in r.vars {
            out.write_all(&v.to_le_bytes())?;
        }
    }
    out.flush()?;
    println!("{}", String::from_utf8(metadata)?);
    Ok(())
}
