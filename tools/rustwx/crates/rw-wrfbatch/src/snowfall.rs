//! Snowfall depths in mm of snow, integrated over the available WRF history.
//! The ratio and f32 interval arithmetic match the supplied snowfall patch.
use std::path::{Path, PathBuf};

use crate::wrf_process::{
    WrfHourFields, WrfProcessOptions, WrfProductGroup, compute_var, push_canonical_values,
};
use rustwx_core::{CanonicalField, FieldSelector, GridProjection, LatLonGrid};
use wrf_core::WrfFile;

/// Catalog metadata shared by the import and its field/selector preview.
pub const PRODUCTS: &[(&str, CanonicalField)] = &[
    ("snowfall_window", CanonicalField::ModelSnowfall),
    ("snow_10to1_window", CanonicalField::Snowfall10to1),
    ("snow_kuchera_window", CanonicalField::SnowfallKuchera),
    ("snowfall_liquid_equivalent", CanonicalField::SnowfallLiquidEquivalent),
    ("snow_depth_total", CanonicalField::SnowDepth),
    ("snow_depth_change", CanonicalField::SnowDepthChange),
];

/// Owned by one import, never shared between jobs or source snapshots.
/// Retains only the latest cumulative plane and liquid counter.
#[derive(Default)]
pub(crate) struct WindowCache {
    kuchera: Option<WindowState>,
    fixed: Option<WindowState>,
    liquid: Option<WindowState>,
}

struct WindowState {
    directory: PathBuf,
    prefix: String,
    reference: Option<i64>,
    start: i64,
    until: i64,
    previous: Vec<f32>,
    accumulated: Vec<f32>,
}

pub use rustwx_calc::snow::{column_max, kuchera_slr};
use rustwx_calc::snow::SnowMethod;

pub fn column_tmax_500(file: &WrfFile, timeidx: usize, cells: usize) -> Result<Vec<f32>, String> {
    let t = compute_var(file, "temp", timeidx, Some("K"))?;
    let p = compute_var(file, "pres", timeidx, Some("Pa"))?;
    if t.shape != [file.nz, file.ny, file.nx] || p.shape != t.shape || cells != file.ny * file.nx {
        return Err("temperature/pressure dimension order differs from [nz,ny,nx]".into());
    }
    column_max(&t.data, &p.data, file.nz, cells)
}

fn plane(file: &WrfFile, name: &str, ti: usize) -> Result<Vec<f32>, String> {
    let v = compute_var(file, name, ti, None)?;
    if v.shape != [file.ny, file.nx] || v.data.len() != file.ny * file.nx {
        return Err(format!("{name}: expected [ny,nx], got {:?}", v.shape));
    }
    Ok(v.data
        .iter()
        .map(|&v| {
            if v.is_finite() && (0.0..1.0e30).contains(&v) {
                v as f32
            } else {
                f32::NAN
            }
        })
        .collect())
}

fn frozen(file: &WrfFile, ti: usize) -> Result<Vec<f32>, String> {
    let mut snow = plane(file, "SNOWNC", ti)?;
    // Absence is allowed. A present but unreadable graupel accumulator is
    // an error, to prevent a silently understated snowfall total.
    if file.has_var("GRAUPELNC") {
        let graupel = plane(file, "GRAUPELNC", ti)?;
        for (s, g) in snow.iter_mut().zip(graupel) {
            *s += g;
        }
    }
    Ok(snow)
}

/// End-of-interval weighting, in source-patch order and precision.
/// A reset or missing accumulator poisons that cell instead of inventing
/// a zero snowfall interval after a restart.
pub fn accumulate(
    acc: &mut [f32],
    previous: &[f32],
    current: &[f32],
    tmax: &[f32],
) -> Result<(), String> {
    rustwx_calc::snow::accumulate(acc, previous, current, tmax, SnowMethod::Kuchera)
}

/// Neighboring files of the same WRF domain. Other domains and scratch
/// files cannot enter a snowfall integration or its cache identity.
pub(crate) fn history_paths(path: &Path) -> Result<Vec<PathBuf>, String> {
    let mut paths = vec![path.to_path_buf()];
    let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
    if name.starts_with("wrfout_d") && name.as_bytes().get(10) == Some(&b'_') {
        let prefix = &name[..11];
        for entry in
            std::fs::read_dir(path.parent().unwrap_or(Path::new("."))).map_err(|e| e.to_string())?
        {
            let entry = entry.map_err(|e| e.to_string())?;
            let n = entry.file_name();
            let n = n.to_string_lossy();
            if n.starts_with(prefix) && n.len() == prefix.len() + 19 {
                paths.push(entry.path());
            }
        }
    }
    paths.sort();
    paths.dedup();
    Ok(paths)
}

// A future frame may still be being written. Its filename gives its
// earliest time under the ordinary WRF history naming contract.
fn starts_after(path: &Path, until: i64) -> bool {
    let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
        return false;
    };
    let Some(stamp) = name.get(11..) else {
        return false;
    };
    chrono::NaiveDateTime::parse_from_str(&stamp.replace(':', "_"), "%Y-%m-%d_%H_%M_%S")
        .ok()
        .is_some_and(|t| t.and_utc().timestamp() > until)
}

/// The ledger of the history frames a run wrote, beside them, one file name
/// per line (`HISTORY_LEDGER` in gpuwm's `gpuwm/io/wrfout.py` writes it).
pub const HISTORY_LEDGER: &str = ".history-written";

/// The frames of this history the run wrote (its ledger names them) that
/// are gone from the folder, valid after `after` and at or before `now`.
///
/// WHAT BREAKAGE THIS PREVENTS: a host removed earlier history frames to
/// save disk (an 8x5090 host, 2026-10-21).  A Kuchera window takes each interval's
/// snow-to-liquid ratio from the column temperature at the interval's end,
/// so summed across a removed frame it takes one end temperature for the
/// whole span and is a different total from the run's own.  A cadence that
/// never wrote those times (a resumed run at a new interval, a domain whose
/// output interval changes) writes no ledger line for them, so it keeps its
/// window.  With no ledger (frames from elsewhere) nothing is called missing.
fn removed_history_frames(directory: &Path, prefix: &str, after: i64, now: i64) -> Vec<String> {
    let Ok(text) = std::fs::read_to_string(directory.join(HISTORY_LEDGER)) else {
        return Vec::new();
    };
    let mut removed: Vec<String> = text
        .lines()
        .map(str::trim)
        .filter(|name| name.starts_with(prefix) && name.len() == prefix.len() + 19)
        .filter(|name| {
            frame_valid_unix(name).is_some_and(|valid| valid > after && valid <= now)
        })
        .filter(|name| !directory.join(name).exists())
        .map(str::to_string)
        .collect();
    removed.sort();
    removed.dedup();
    removed
}

/// The valid time a `wrfout_dNN_YYYY-MM-DD_HH_MM_SS` name carries.
fn frame_valid_unix(name: &str) -> Option<i64> {
    let stamp = name.get(11..)?;
    chrono::NaiveDateTime::parse_from_str(&stamp.replace(':', "_"), "%Y-%m-%d_%H_%M_%S")
        .ok()
        .map(|time| time.and_utc().timestamp())
}

/// A history valid time as WRF names it (`YYYY-MM-DD_HH:MM:SS`, UTC).
fn valid_label(unix: i64) -> String {
    chrono::DateTime::from_timestamp(unix, 0)
        .map(|time| time.format("%Y-%m-%d_%H:%M:%S").to_string())
        .unwrap_or_else(|| unix.to_string())
}

/// Integrate all records in (since,current], including records in a
/// multi-time file. The exact baseline must exist, so a truncated history
/// cannot be advertised as a total since initialization.
pub fn window(
    file: &WrfFile,
    path: &Path,
    timeidx: usize,
    since: Option<&str>,
    kuchera: bool,
) -> Result<Vec<f32>, String> {
    window_cached(file, path, timeidx, since, if kuchera { SnowMethod::Kuchera } else { SnowMethod::TenToOne }, &mut None)
}

fn window_cached(
    file: &WrfFile,
    path: &Path,
    timeidx: usize,
    since: Option<&str>,
    method: SnowMethod,
    cache: &mut Option<WindowState>,
) -> Result<Vec<f32>, String> {
    let axis = crate::local_import::wrf_source_times(file, path)?;
    let now = axis
        .records
        .get(timeidx)
        .ok_or("current record missing")?
        .valid_unix;
    let start = match since {
        Some(s) => chrono::NaiveDateTime::parse_from_str(&s.replace(':', "_"), "%Y-%m-%d_%H_%M_%S")
            .map_err(|e| format!("snowfall baseline must be YYYY-MM-DD_HH:MM:SS: {e}"))?
            .and_utc()
            .timestamp(),
        None => axis
            .reference_unix
            .ok_or("snowfall requires a run initialization time")?,
    };
    if start > now {
        return Err("snowfall baseline is later than the current frame".into());
    }
    // Every history file is opened and checked at once; the records and
    // the first failure come out in path order, as one after another.
    let reference_xlat = file.xlat(timeidx).map_err(|e| e.to_string())?;
    let reference_xlong = file.xlong(timeidx).map_err(|e| e.to_string())?;
    let scanned = {
        use rayon::prelude::*;
        history_paths(path)?
            .into_par_iter()
            .filter(|p| !(p != path && starts_after(p, now)))
            .map(|p| -> Result<Vec<(i64, PathBuf, usize)>, String> {
                let f = match WrfFile::open(&p) {
                    Ok(f) => f,
                    // Removed since it was listed: the history no longer
                    // holds it, exactly as though it had not been listed.
                    Err(_) if !p.exists() => return Ok(Vec::new()),
                    Err(e) => return Err(format!("{}: {e}", p.display())),
                };
                let a = crate::local_import::wrf_source_times(&f, &p)?;
                let mut found = Vec::new();
                for r in a.records {
                    if r.valid_unix >= start && r.valid_unix <= now {
                        if a.reference_unix != axis.reference_unix
                            || (f.nz, f.ny, f.nx) != (file.nz, file.ny, file.nx)
                            || f.xlat(r.time_index).map_err(|e| e.to_string())? != reference_xlat
                            || f.xlong(r.time_index).map_err(|e| e.to_string())? != reference_xlong
                        {
                            return Err(
                                "snowfall history run origin or grid differs from current frame".into(),
                            );
                        }
                        found.push((r.valid_unix, p.clone(), r.time_index));
                    }
                }
                Ok(found)
            })
            .collect::<Vec<_>>()
    };
    let mut records = Vec::new();
    for found in scanned {
        records.extend(found?);
    }
    records.sort();
    if records.first().map(|r| r.0) != Some(start) {
        return Err(format!(
            "exact snowfall baseline frame missing (no history record valid at {}); refusing an understated window total",
            valid_label(start)
        ));
    }
    if records.windows(2).any(|r| r[0].0 == r[1].0) {
        return Err("duplicate snowfall valid time; refusing double counting".into());
    }
    let cells = file.ny * file.nx;
    let directory = path.parent().unwrap_or(Path::new(".")).to_path_buf();
    let name = path.file_name().and_then(|v| v.to_str()).unwrap_or("");
    let prefix = if name.starts_with("wrfout_d") && name.as_bytes().get(10) == Some(&b'_') {
        &name[..11]
    } else {
        name
    };
    let reusable = cache.as_ref().filter(|state| {
        state.directory == directory
            && state.prefix == prefix
            && state.reference == axis.reference_unix
            && state.start == start
            && state.until <= now
            && records.iter().any(|r| r.0 == state.until)
            && state.previous.len() == cells
    });
    // The integration an earlier process saved through an earlier frame of
    // this same, unchanged history (`crate::source_memo`): drawing frame N
    // folds one interval instead of re-reading 3-D temperature and pressure
    // from every frame since the baseline.
    let window_key = format!(
        "snow-v2\u{0}{}\u{0}{prefix}\u{0}{:?}\u{0}{start}\u{0}{method:?}",
        directory.display(),
        axis.reference_unix
    );
    let stamps_through = |until: i64| -> Option<Vec<String>> {
        let mut seen: Vec<&PathBuf> = Vec::new();
        for (valid, source, _) in &records {
            if *valid <= until && !seen.contains(&source) {
                seen.push(source);
            }
        }
        seen.into_iter().map(|source| crate::source_memo::file_stamp(source)).collect()
    };
    let recalled = if reusable.is_none() {
        crate::source_memo::recall_snowfall(&window_key, cells).filter(|memo| {
            memo.until <= now
                && records.iter().any(|r| r.0 == memo.until)
                && stamps_through(memo.until).is_some_and(|stamps| stamps == memo.sources)
        })
    } else {
        None
    };
    let (mut prev, mut acc, until) = if let Some(state) = reusable {
        (
            state.previous.clone(),
            state.accumulated.clone(),
            state.until,
        )
    } else if let Some(memo) = recalled {
        (memo.previous, memo.accumulated, memo.until)
    } else {
        let base = WrfFile::open(&records[0].1)
            .map_err(|e| format!("{}: {e}", records[0].1.display()))?;
        (frozen(&base, records[0].2)?, vec![0.0; cells], start)
    };
    // Kuchera across a frame this run wrote and someone removed would take
    // one end temperature for every interval the hole spans: refused by
    // name.  Intervals already folded into a memo were folded whole.
    if method == SnowMethod::Kuchera {
        let removed = removed_history_frames(&directory, prefix, until, now);
        if !removed.is_empty() {
            let named = removed
                .iter()
                .map(|name| {
                    format!(
                        "{name} (valid {})",
                        frame_valid_unix(name).map(valid_label).unwrap_or_default()
                    )
                })
                .collect::<Vec<_>>()
                .join(", ");
            return Err(format!(
                "history frame(s) this run wrote were removed: {named}; a Kuchera total across them would take one end temperature for every interval they span"
            ));
        }
    }
    // A missing baseline value stays unknown even for a zero-length window.
    for (a, p) in acc.iter_mut().zip(&prev) {
        if !p.is_finite() {
            *a = f32::NAN;
        }
    }
    // Each interval's inputs (its frozen accumulator with resets poisoned,
    // and for Kuchera the 500 hPa column temperature maximum) depend on two
    // frames alone, so they are read at once, a memory-bounded group at a
    // time; the accumulation itself folds them in record order, exactly as
    // one after another, so the window holds the same bytes.
    let pending: Vec<usize> = records
        .iter()
        .enumerate()
        .filter(|(_, r)| r.0 > until)
        .map(|(index, _)| index)
        .collect();
    let group = replay_group_width(file, method);
    for chunk in pending.chunks(group) {
        let inputs = {
            use rayon::prelude::*;
            chunk
                .par_iter()
                .map(|&index| interval_inputs(file, path, &records, index, method, cells))
                .collect::<Vec<_>>()
        };
        for input in inputs {
            let (current, tmax) = input?;
            rustwx_calc::snow::accumulate(&mut acc, &prev, &current, &tmax, method)?;
            prev = current;
        }
    }
    if let Some(sources) = stamps_through(now) {
        crate::source_memo::remember_snowfall(
            &window_key,
            &crate::source_memo::SnowfallMemo {
                until: now,
                sources,
                previous: prev.clone(),
                accumulated: acc.clone(),
            },
        );
    }
    *cache = Some(WindowState {
        directory,
        prefix: prefix.into(),
        reference: axis.reference_unix,
        start,
        until: now,
        previous: prev,
        accumulated: acc.clone(),
    });
    Ok(acc)
}

/// What computing one Kuchera interval's column maximum holds at once:
/// temperature, potential temperature, perturbation and base pressure and
/// the full pressure, each a double-precision volume, with the
/// [nz, ny, nx] copies `compute_var` hands back.
const TMAX_VOLUMES: u64 = 8;

/// How many replayed intervals read their inputs at once: every one for
/// the two-dimensional windows, and for Kuchera as many as a quarter of
/// the free memory holds (each reads two 3-D fields), at least one.
fn replay_group_width(file: &WrfFile, method: SnowMethod) -> usize {
    let threads = rayon::current_num_threads().max(1);
    if method != SnowMethod::Kuchera {
        return threads;
    }
    let per = (file.nz.max(1) * file.ny * file.nx) as u64 * 8 * TMAX_VOLUMES;
    let free = rusty_weather::host_memory::available_bytes().unwrap_or(4 << 30);
    ((free / 4 / per.max(1)) as usize).clamp(1, threads)
}

const TMAX_PLANE: &str = "column-tmax-500hpa-v1";

/// The 500 hPa column temperature maximum of record `ti` of `f` (the file
/// at `p`), from its memo when this unchanged file's own render saved it.
fn column_tmax_remembered(f: &WrfFile, p: &Path, ti: usize, cells: usize) -> Result<Vec<f32>, String> {
    if let Some(plane) = crate::source_memo::recall_frame_plane(p, ti, TMAX_PLANE, cells) {
        return Ok(plane);
    }
    let stamp = crate::source_memo::file_stamp(p);
    let plane = column_tmax_500(f, ti, cells)?;
    if let Some(stamp) = &stamp {
        crate::source_memo::remember_frame_plane(p, ti, TMAX_PLANE, stamp, &plane);
    }
    Ok(plane)
}

/// One replayed interval's inputs: the frozen accumulator of `records[index]`
/// with every counter reset since `records[index - 1]` poisoned, and for
/// Kuchera its column maximum.  The frame being drawn is `file` itself, so
/// its 3-D fields come from the volumes the import already read.
fn interval_inputs(
    file: &WrfFile,
    path: &Path,
    records: &[(i64, PathBuf, usize)],
    index: usize,
    method: SnowMethod,
    cells: usize,
) -> Result<(Vec<f32>, Vec<f32>), String> {
    let (_, p, ti) = &records[index];
    let opened;
    let f: &WrfFile = if p == path {
        file
    } else {
        opened = WrfFile::open(p).map_err(|e| format!("{}: {e}", p.display()))?;
        &opened
    };
    let mut current = frozen(f, *ti)?;
    // Check each counter independently. Graupel growth must not hide a
    // snow-counter reset (or the reverse) in their sum.
    let previous_record = &records[index - 1];
    let opened_before;
    let before: &WrfFile = if previous_record.1 == *path {
        file
    } else if previous_record.1 == *p {
        f
    } else {
        opened_before = WrfFile::open(&previous_record.1)
            .map_err(|e| format!("{}: {e}", previous_record.1.display()))?;
        &opened_before
    };
    for name in ["SNOWNC", "GRAUPELNC"] {
        if f.has_var(name) != before.has_var(name) {
            return Err(format!("{name} presence changes within snowfall history"));
        }
        if f.has_var(name) {
            let a = plane(before, name, previous_record.2)?;
            let b = plane(f, name, *ti)?;
            for ((out, old), new) in current.iter_mut().zip(a).zip(b) {
                if !old.is_finite() || !new.is_finite() || new < old { *out = f32::NAN; }
            }
        }
    }
    let tmax = if method == SnowMethod::Kuchera {
        column_tmax_remembered(f, p, *ti, cells)?
    } else {
        Vec::new()
    };
    Ok((current, tmax))
}

/// The WRF history field a snowfall product is built from.
///
/// WHAT BREAKAGE THIS PREVENTS: `SNOWFALLAC` is a RUC land-model field
/// (WRF Registry package `ruclsmscheme`, `sf_surface_physics==3`), so a
/// default Noah forecast never writes it.  `snowfall_window` still pushed
/// an "unavailable" note on every frame of such a run, the render bridge
/// read that note as a skip of a requested product (it is meant for a
/// history that lost a frame it needs), and every default `woof go` closed
/// with "snowfall_window drew no picture".  A history without the source
/// field does not carry the product at all; only a history that carries it
/// and still cannot build the window is a named gap.
pub(crate) fn history_source(field: CanonicalField) -> &'static str {
    match field {
        CanonicalField::ModelSnowfall => "SNOWFALLAC",
        CanonicalField::SnowDepth | CanonicalField::SnowDepthChange => "SNOWH",
        _ => "SNOWNC",
    }
}

pub(crate) fn push(
    fields: &mut WrfHourFields,
    file: &WrfFile,
    path: &Path,
    ti: usize,
    options: &WrfProcessOptions,
    grid: &LatLonGrid,
    projection: Option<GridProjection>,
    cache: &mut WindowCache,
) {
    for &(slug, field) in PRODUCTS {
        if !options.should_process(
            &FieldSelector::surface(field).key(),
            Some(slug),
            WrfProductGroup::Diagnostic,
        ) {
            continue;
        }
        if !file.has_var(history_source(field)) {
            // Not a gap: this history never stores the field, so the
            // product is not stored and the catalog lists it as
            // missing-fields, like every other field-gated product.
            continue;
        }
        let result = match field {
            CanonicalField::ModelSnowfall => {
                model_window(file, path, ti, options.snow_since.as_deref())
            }
            CanonicalField::SnowDepth => plane(file, "SNOWH", ti).map(|v| v.into_iter().map(|x| x*1000.0).collect()),
            CanonicalField::SnowDepthChange => depth_window(file, path, ti, options.snow_since.as_deref()),
            _ => window_cached(
                file,
                path,
                ti,
                options.snow_since.as_deref(),
                match field { CanonicalField::SnowfallKuchera => SnowMethod::Kuchera,
                    CanonicalField::SnowfallLiquidEquivalent => SnowMethod::LiquidEquivalent,
                    _ => SnowMethod::TenToOne },
                if field == CanonicalField::SnowfallKuchera {
                    &mut cache.kuchera
                } else if field == CanonicalField::SnowfallLiquidEquivalent {
                    &mut cache.liquid
                } else {
                    &mut cache.fixed
                },
            ),
        };
        match result {
            Ok(v) => {
                let missing = v.iter().filter(|v| !v.is_finite()).count();
                push_canonical_values(
                    fields,
                    grid,
                    projection.clone(),
                    slug,
                    FieldSelector::surface(field),
                    "mm",
                    v,
                );
                fields.notes.push(format!(
                    "{slug}: stored mm; baseline {}; snowfall uses saved-history intervals, not every-step SLR; SNOWH is standing depth, change retains settling/melt",
                    options
                        .snow_since
                        .as_deref()
                        .unwrap_or("model initialization")
                ));
                if missing > 0 {
                    fields.notes.push(format!("{slug}: {missing} cells remain missing; unknown column temperatures, non-finite accumulators and counter resets are not zero snowfall"));
                }
            }
            Err(e) => fields.notes.push(format!("{slug} unavailable: {e}")),
        }
    }
}

fn model_window(
    file: &WrfFile,
    path: &Path,
    ti: usize,
    since: Option<&str>,
) -> Result<Vec<f32>, String> {
    let current = plane(file, "SNOWFALLAC", ti)?;
    // Explicit model snowfall is already a depth in mm, not snow water.
    if since.is_none() {
        return Ok(current);
    }
    let wanted = chrono::NaiveDateTime::parse_from_str(
        &since.unwrap().replace(':', "_"),
        "%Y-%m-%d_%H_%M_%S",
    )
    .map_err(|e| format!("invalid snowfall baseline: {e}"))?
    .and_utc()
    .timestamp();
    let axis = crate::local_import::wrf_source_times(file, path)?;
    let now = axis.records[ti].valid_unix;
    if wanted > now {
        return Err("snowfall baseline is later than current frame".into());
    }
    let mut base = None;
    for p in history_paths(path)? {
        if p != path && starts_after(&p, wanted) {
            continue;
        }
        let f = match WrfFile::open(&p) {
            Ok(f) => f,
            Err(_) if !p.exists() => continue,
            Err(e) => return Err(format!("{}: {e}", p.display())),
        };
        let a = crate::local_import::wrf_source_times(&f, &p)?;
        for r in a.records {
            if r.valid_unix == wanted {
                if base.is_some() {
                    return Err("duplicate explicit snowfall baseline".into());
                }
                if a.reference_unix != axis.reference_unix
                    || (f.ny, f.nx) != (file.ny, file.nx)
                    || f.xlat(r.time_index).map_err(|e| e.to_string())?
                        != file.xlat(ti).map_err(|e| e.to_string())?
                    || f.xlong(r.time_index).map_err(|e| e.to_string())?
                        != file.xlong(ti).map_err(|e| e.to_string())?
                {
                    return Err("explicit snowfall baseline grid differs".into());
                }
                base = Some(plane(&f, "SNOWFALLAC", r.time_index)?);
            }
        }
    }
    let base = base.ok_or_else(|| format!(
        "exact explicit snowfall baseline missing (no history record valid at {})",
        valid_label(wanted)
    ))?;
    Ok(current
        .iter()
        .zip(base)
        .map(|(&c, b)| {
            if c.is_finite() && b.is_finite() && c >= b {
                c - b
            } else {
                f32::NAN
            }
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn hand_computed_columns_and_ratio_branches() {
        let p = [90000., 80000., 50000., 49999.];
        let t = [265.16, 270.16, 269.16, 310.];
        let m = column_max(&t, &p, 4, 1).unwrap()[0];
        assert!((kuchera_slr(m) - 13.).abs() < 0.0001);
        for (t, r) in [
            (271.16, 12.),
            (272.16, 10.),
            (273.16, 8.),
            (269.16, 14.),
            (277.16, 0.),
            (280., 0.),
        ] {
            assert!((kuchera_slr(t) - r).abs() < 0.0001);
        }
        assert!(kuchera_slr(f32::NAN).is_nan());
        assert!(kuchera_slr(f32::INFINITY).is_nan());
    }
    #[test]
    fn included_500hpa_and_invalid_columns() {
        assert_eq!(
            column_max(&[260., 275., 310.], &[90000., 50000., 49000.], 3, 1).unwrap(),
            [275.]
        );
        assert!(column_max(&[260., f64::NAN], &[90000., 70000.], 2, 1).unwrap()[0].is_nan());
        assert!(column_max(&[260.], &[49000.], 1, 1).unwrap()[0].is_nan());
        assert!(column_max(&[260.], &[f64::NAN], 1, 1).unwrap()[0].is_nan());
        assert!(column_max(&[260.], &[90000.], 2, 1).is_err());
    }
    #[test]
    fn intervals_use_end_temperature_and_propagate_missing_or_resets() {
        let mut v = [0., 0., 0.];
        accumulate(&mut v, &[0., 2., f32::NAN], &[2., 1., 3.], &[272.16; 3]).unwrap();
        assert!((v[0] - 20.).abs() < 0.0002);
        assert!(v[1].is_nan());
        assert!(v[2].is_nan());
        accumulate(&mut v, &[2., 1., 3.], &[3., 2., 4.], &[269.16; 3]).unwrap();
        assert!((v[0] - 34.).abs() < 0.0003);
        assert!(accumulate(&mut v, &[], &[], &[]).is_err());
    }
}

fn depth_window(file: &WrfFile, path: &Path, ti: usize, since: Option<&str>) -> Result<Vec<f32>, String> {
    let axis = crate::local_import::wrf_source_times(file, path)?;
    let start = match since {
        Some(s) => chrono::NaiveDateTime::parse_from_str(&s.replace(':', "_"), "%Y-%m-%d_%H_%M_%S")
            .map_err(|e| e.to_string())?.and_utc().timestamp(),
        None => axis.reference_unix.ok_or("snow depth change requires initialization time")?,
    };
    if start > axis.records[ti].valid_unix { return Err("snow depth baseline later than current frame".into()); }
    let mut base = None;
    for p in history_paths(path)? {
        if p != path && starts_after(&p, start) { continue; }
        let f = match WrfFile::open(&p) {
            Ok(f) => f,
            Err(_) if !p.exists() => continue,
            Err(e) => return Err(format!("{}: {e}", p.display())),
        };
        let a = crate::local_import::wrf_source_times(&f, &p)?;
        for r in a.records {
            if r.valid_unix != start { continue; }
            if base.is_some() { return Err("duplicate snow depth baseline".into()); }
            if a.reference_unix != axis.reference_unix || (f.ny,f.nx)!=(file.ny,file.nx)
                || f.xlat(r.time_index).map_err(|e|e.to_string())? != file.xlat(ti).map_err(|e|e.to_string())?
                || f.xlong(r.time_index).map_err(|e|e.to_string())? != file.xlong(ti).map_err(|e|e.to_string())? {
                return Err("snow depth baseline grid or run origin differs".into());
            }
            base=Some(plane(&f,"SNOWH",r.time_index)?);
        }
    }
    rustwx_calc::snow::depth_change_mm(&plane(file,"SNOWH",ti)?, &base.ok_or_else(|| format!("exact snow depth baseline missing (no history record valid at {})", valid_label(start)))?)
}
