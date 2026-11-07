//! Durable publication using the boundary consumer's existing series contract.
use super::{BoundaryInterval, IntervalReceipt, LbcConfig};
use crate::error::MpasResult;
use std::{io::Write, path::Path};

fn atomic(path: &Path, value: &serde_json::Value) -> MpasResult<()> {
    let temporary = path.with_extension("json.partial");
    let mut file = std::fs::File::create(&temporary)?;
    serde_json::to_writer(&mut file, value)
        .map_err(|e| crate::error::MpasError::Refusal(e.to_string()))?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    std::fs::rename(temporary, path)?;
    Ok(())
}
pub fn begin(cfg: &LbcConfig) -> MpasResult<()> {
    std::fs::create_dir_all(&cfg.out_dir)?;
    let _ = std::fs::remove_file(cfg.out_dir.join("lbc-failed.json"));
    for interval in &cfg.intervals {
        let path = cfg.out_dir.join(format!(
            "lbc.{}.nc.ready.json",
            interval.valid_time.replace(':', ".")
        ));
        if path.exists() {
            std::fs::remove_file(path)?;
        }
    }
    if cfg.intervals.len() >= 2 {
        atomic(
            &cfg.out_dir.join("lbc-series.json"),
            &serde_json::json!({
                "schema":"gpuwm-hex.lbc-series/v1","interval_seconds":cfg.fg_interval_seconds,
                "valid_times":cfg.intervals.iter().map(|i|&i.valid_time).collect::<Vec<_>>()
            }),
        )?;
    }
    Ok(())
}
pub fn ready(one: &IntervalReceipt) -> MpasResult<()> {
    let path = Path::new(&one.out_path);
    atomic(
        &path.with_file_name(format!(
            "{}.ready.json",
            path.file_name().unwrap().to_string_lossy()
        )),
        &serde_json::json!({"schema":"gpuwm-hex.lbc.hour/v1","valid_time":one.valid_time,
            "path":path.file_name().unwrap().to_string_lossy(),"bytes":path.metadata()?.len(),
            "sha256":one.out_sha256}),
    )
}
pub fn failed(cfg: &LbcConfig, error: &str) {
    let _ = atomic(
        &cfg.out_dir.join("lbc-failed.json"),
        &serde_json::json!({"error":error}),
    );
}

/// A pool can start from cold zeros after the first time only if every wind
/// slot is overwritten and the same fields retain the same level slots.
/// Explicit pressure donors keep the native sequential pool behavior.
pub struct Independent {
    pub levels: Vec<f32>,
    pub scalar_species: usize,
}
pub fn independent_levels(intervals: &[BoundaryInterval]) -> MpasResult<Option<Independent>> {
    use crate::rrfs_init::{metfile::MetReader, scalars};
    let mut first = None;
    let mut levels = Vec::<f32>::new();
    for interval in intervals {
        let mut reader = MetReader::open_headers(&interval.met_path)?;
        let mut signature = Vec::new();
        while let Some((slab, _)) = reader.next_header()? {
            if super::LBC_MET_FIELDS.contains(&slab.field.as_str())
                || scalars::input_index(&slab.field).is_some()
            {
                if matches!(slab.field.as_str(), "PRES" | "PRESSURE") {
                    return Ok(None);
                }
                signature.push((slab.field, slab.xlvl.to_bits()));
            }
        }
        if let Some(ref expected) = first {
            if expected != &signature {
                return Ok(None);
            }
        } else {
            for (field, bits) in &signature {
                let level = f32::from_bits(*bits);
                if field != "SOILHGT" && !levels.contains(&level) {
                    levels.push(level);
                }
            }
            for wind in ["UU", "VV"] {
                if levels
                    .iter()
                    .any(|v| !signature.contains(&(wind.into(), v.to_bits())))
                {
                    return Ok(None);
                }
            }
            first = Some(signature);
        }
    }
    let scalar_species = first
        .unwrap_or_default()
        .iter()
        .filter_map(|(f, _)| scalars::input_index(f))
        .collect::<std::collections::BTreeSet<_>>()
        .len();
    Ok(Some(Independent {
        levels,
        scalar_species,
    }))
}

pub fn workers(cfg: &LbcConfig, nc: usize, ne: usize, nz: usize, scalar_species: usize) -> usize {
    // Include columns, work arrays, output fields and vector headers. Shared
    // mesh/metrics and two contributing aerosol months are counted once.
    let shared = (ne as u64 * 2 * (nz + 1) as u64 + nc as u64 * (nz + 1) as u64) * 4
        + nc as u64 * cfg.gocart_levels as u64 * 3 * 2 * 4
        + 1_000_000_000;
    let per_job = nc as u64
        * (cfg.n_fg_levels as u64 * 24
            + nz as u64 * 40
            + 192
            + scalar_species as u64 * (cfg.n_fg_levels + nz) as u64 * 4)
        + ne as u64 * (cfg.n_fg_levels as u64 * 8 + nz as u64 * 12 + 64);
    let available = std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|s| {
            s.lines().find_map(|line| {
                line.strip_prefix("MemAvailable:")
                    .and_then(|v| v.split_whitespace().next())
                    .and_then(|v| v.parse::<u64>().ok())
                    .map(|v| v * 1024)
            })
        })
        .unwrap_or(32_000_000_000);
    let budget = available.min(56_000_000_000);
    let requested = std::env::var("RW_MPAS_LBC_JOBS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(usize::MAX);
    let count = ((budget.saturating_sub(shared)) / per_job.max(1)) as usize;
    count
        .max(1)
        .min(requested.max(1))
        .min((rayon::current_num_threads() / 4).max(1))
        .min(2)
}
