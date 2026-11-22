//! Light-absorbing-particle (LAP) deposition onto snow, regridded to model
//! columns from GOCART-style aerosol deposition files (GEOS-FP or MERRA-2
//! `*_2d_adg_Nx`). Re-implemented from published descriptions (Lin et al.
//! 2025 JHM; Flanner et al. 2007), not bitwise to upstream Noah-MP.
//!
//! No dataset or species dispatch lives here. The manifest (written by
//! gpuwm/snow_lap_deposition.py from the table gpuwm/data/snow_papers/
//! lap_sources.json) names every channel as a linear combination of source
//! variables, so a new source is a table row, not a code path.
//!
//! Files decode in parallel, one per worker (all cores by default); each
//! worker samples every target column. Output is one little-endian f32
//! series [time][channel][column] plus a receipt with its SHA-256.
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, fs, io::Write, path::{Path, PathBuf}};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    schema: String,
    source: String,
    files: Vec<PathBuf>,
    latitude: String,
    longitude: String,
    time: String,
    /// Averaging window of each stored time, centred on the stored time.
    window_seconds: f64,
    units: String,
    channels: Vec<Channel>,
    /// Raw little-endian f64 target latitudes/longitudes, one per column.
    target_latitude: PathBuf,
    target_longitude: PathBuf,
}
#[derive(Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub(crate) struct Channel { pub key: String, pub terms: Vec<Term> }
#[derive(Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub(crate) struct Term {
    pub name: String,
    pub coef: f64,
    /// When present, `name` is a total shared among these variables in
    /// proportion to their (non-negative) values; this term takes `share`'s part.
    #[serde(default)]
    pub split_by: Vec<String>,
    #[serde(default)]
    pub share: Option<String>,
}
#[derive(Serialize)]
struct Receipt {
    schema: &'static str,
    source: String,
    units: String,
    window_seconds: f64,
    channels: Vec<String>,
    columns: usize,
    times: Vec<String>,
    files: Vec<String>,
    series: String,
    series_sha256: String,
    negative_values_zeroed: usize,
    workers: usize,
}

/// Bracketing index and fraction on a monotonic axis; clamped at the ends.
pub(crate) fn bracket(axis: &[f64], q: f64) -> (usize, f64) {
    let n = axis.len();
    if n < 2 { return (0, 0.); }
    let ascending = axis[1] > axis[0];
    let before = |a: f64| if ascending { a <= q } else { a >= q };
    if !before(axis[0]) { return (0, 0.); }
    if before(axis[n - 1]) { return (n - 2, 1.); }
    let (mut lo, mut hi) = (0usize, n - 1);
    while hi - lo > 1 { let mid = (lo + hi) / 2; if before(axis[mid]) { lo = mid } else { hi = mid } }
    let f = (q - axis[lo]) / (axis[lo + 1] - axis[lo]);
    (lo, f.clamp(0., 1.))
}
/// Four (index, weight) pairs of bilinear sampling on a global lat/lon grid
/// with a uniform ascending longitude axis, wrapping across the seam.
pub(crate) fn weights(lat: &[f64], lon: &[f64], qlat: f64, qlon: f64) -> [(usize, f64); 4] {
    let nx = lon.len();
    let (j, fy) = bracket(lat, qlat);
    let step = lon[1] - lon[0];
    let x = (qlon - lon[0]).rem_euclid(360.);
    let fi = x / step;
    let i0 = (fi.floor() as usize) % nx;
    let fx = fi - fi.floor();
    let i1 = (i0 + 1) % nx;
    let j1 = (j + 1).min(lat.len() - 1);
    [(j * nx + i0, (1. - fy) * (1. - fx)), (j * nx + i1, (1. - fy) * fx),
     (j1 * nx + i0, fy * (1. - fx)), (j1 * nx + i1, fy * fx)]
}
/// Channel values on the source grid for one stored time.
pub(crate) fn combine(channels: &[Channel], get: &dyn Fn(&str) -> Result<Vec<f64>, String>, cells: usize)
    -> Result<Vec<Vec<f64>>, String> {
    let mut cache: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    let mut fetch = |name: &str| -> Result<Vec<f64>, String> {
        if !cache.contains_key(name) {
            let v = get(name)?;
            if v.len() != cells { return Err(format!("{name}: {} values, expected {cells}", v.len())); }
            cache.insert(name.to_owned(), v);
        }
        Ok(cache[name].clone())
    };
    let mut out = Vec::with_capacity(channels.len());
    for ch in channels {
        let mut acc = vec![0.; cells];
        for t in &ch.terms {
            let total = fetch(&t.name)?;
            if t.split_by.is_empty() {
                for (a, v) in acc.iter_mut().zip(&total) { *a += t.coef * v; }
                continue;
            }
            let share = t.share.as_deref().ok_or_else(|| format!("{}: split term needs share", ch.key))?;
            let me = t.split_by.iter().position(|s| s == share)
                .ok_or_else(|| format!("{}: share {share} not in split_by", ch.key))?;
            let parts: Vec<Vec<f64>> = t.split_by.iter().map(|s| fetch(s)).collect::<Result<_, _>>()?;
            let last = t.split_by.len() - 1;
            for c in 0..cells {
                let sum: f64 = parts.iter().map(|p| p[c].max(0.)).sum();
                // A total with no partner to apportion by goes to the last
                // listed member (by table convention the hydrophilic bin).
                let w = if sum > 0. { parts[me][c].max(0.) / sum } else if me == last { 1. } else { 0. };
                acc[c] += t.coef * total[c] * w;
            }
        }
        out.push(acc);
    }
    Ok(out)
}
fn read_f64(path: &Path) -> Result<Vec<f64>, String> {
    let b = fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if b.len() % 8 != 0 { return Err(format!("{}: partial f64 payload", path.display())); }
    Ok(b.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect())
}
fn axis(file: &netcrust::File, name: &str) -> Result<Vec<f64>, String> {
    let (v, shape) = super::compose_source::values(file, name)?;
    if shape.len() != 1 || v.len() < 2 { return Err(format!("{name}: need a 1-D axis")); }
    Ok(v)
}
struct FileOut { times: Vec<String>, values: Vec<f32>, negatives: usize }

fn one_file(m: &Manifest, path: &Path, tlat: &[f64], tlon: &[f64]) -> Result<FileOut, String> {
    let f = netcrust::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let lat = axis(&f, &m.latitude)?;
    let lon = axis(&f, &m.longitude)?;
    let step = lon[1] - lon[0];
    if step <= 0. || lon.windows(2).any(|p| ((p[1] - p[0]) - step).abs() > 1e-6) {
        return Err(format!("{}: longitude must be uniform and ascending", path.display()));
    }
    let tvar = f.variable(&m.time).ok_or("missing time")?;
    let units = tvar.attribute("units").and_then(|a| a.as_string()).ok_or("time has no CF units")?;
    let reference = super::parse_cf_units(units)?.ok_or("time has no CF reference")?;
    let (tv, _) = super::compose_source::values(&f, &m.time)?;
    let times = super::decode_times(&reference, &tv)?;
    let cells = lat.len() * lon.len();
    // Decode every needed variable once per file.
    let mut full: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for ch in &m.channels { for t in &ch.terms {
        for name in std::iter::once(&t.name).chain(t.split_by.iter()) {
            if full.contains_key(name) { continue; }
            let (v, shape) = super::compose_source::values(&f, name)?;
            if shape.iter().product::<usize>() != times.len() * cells {
                return Err(format!("{}: {name} shape {shape:?} is not [time,lat,lon]", path.display()));
            }
            full.insert(name.clone(), v);
        }
    }}
    let w: Vec<[(usize, f64); 4]> = tlat.iter().zip(tlon).map(|(a, b)| weights(&lat, &lon, *a, *b)).collect();
    let mut values = Vec::with_capacity(times.len() * m.channels.len() * tlat.len());
    let mut negatives = 0usize;
    for it in 0..times.len() {
        let get = |name: &str| -> Result<Vec<f64>, String> {
            Ok(full[name][it * cells..(it + 1) * cells].to_vec())
        };
        let grid = combine(&m.channels, &get, cells)?;
        for ch in &grid {
            for wc in &w {
                let mut v: f64 = wc.iter().map(|(k, a)| ch[*k] * a).sum();
                if v < 0. { negatives += 1; v = 0.; }
                values.push(v as f32);
            }
        }
    }
    Ok(FileOut { times, values, negatives })
}

pub fn run(manifest: &Path, out: &Path, jobs: usize) -> Result<(), String> {
    let m: Manifest = serde_json::from_slice(&fs::read(manifest).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    if m.schema != "gpuwm-snow-lap-deposition-v1" { return Err(format!("unknown schema {}", m.schema)); }
    if m.units != "kg m-2 s-1" { return Err("channels must be kg m-2 s-1".into()); }
    if !(m.window_seconds > 0.) { return Err("window_seconds must be positive".into()); }
    if m.files.is_empty() || m.channels.is_empty() { return Err("manifest needs files and channels".into()); }
    let tlat = read_f64(&m.target_latitude)?;
    let tlon = read_f64(&m.target_longitude)?;
    if tlat.len() != tlon.len() || tlat.is_empty() { return Err("target latitude/longitude counts differ".into()); }
    fs::create_dir_all(out).map_err(|e| e.to_string())?;
    let workers = if jobs == 0 { std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1) } else { jobs };
    let pool = rayon::ThreadPoolBuilder::new().num_threads(workers).build().map_err(|e| e.to_string())?;
    let results: Vec<Result<FileOut, String>> =
        pool.install(|| m.files.par_iter().map(|p| one_file(&m, p, &tlat, &tlon)).collect());
    let per = m.channels.len() * tlat.len();
    let mut rows: Vec<(String, Vec<f32>)> = Vec::new();
    let mut negatives = 0;
    for r in results {
        let r = r?;
        negatives += r.negatives;
        for (k, t) in r.times.iter().enumerate() { rows.push((t.clone(), r.values[k * per..(k + 1) * per].to_vec())); }
    }
    rows.sort_by(|a, b| a.0.cmp(&b.0));
    if rows.windows(2).any(|p| p[0].0 == p[1].0) { return Err("duplicate stored time across files".into()); }
    let mut bytes = Vec::with_capacity(rows.len() * per * 4);
    for (_, v) in &rows { for x in v { bytes.extend_from_slice(&x.to_le_bytes()); } }
    fs::File::create(out.join("deposition.f32")).and_then(|mut f| f.write_all(&bytes)).map_err(|e| e.to_string())?;
    let receipt = Receipt {
        schema: "gpuwm-snow-lap-deposition-series-v1", source: m.source.clone(), units: m.units.clone(),
        window_seconds: m.window_seconds, channels: m.channels.iter().map(|c| c.key.clone()).collect(),
        columns: tlat.len(), times: rows.iter().map(|r| r.0.clone()).collect(),
        files: m.files.iter().map(|p| p.display().to_string()).collect(),
        series: "deposition.f32".into(), series_sha256: format!("{:x}", Sha256::digest(&bytes)),
        negative_values_zeroed: negatives, workers,
    };
    fs::write(out.join("receipt.json"), serde_json::to_vec_pretty(&receipt).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    println!("snow-lap-deposition: {} times x {} channels x {} columns, {} workers",
             rows.len(), m.channels.len(), tlat.len(), workers);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bilinear_wraps_and_interpolates() {
        let lat = vec![-90., 0., 90.];
        let lon: Vec<f64> = (0..4).map(|i| -180. + 90. * i as f64).collect();
        let w = weights(&lat, &lon, 45., 135.);
        let sum: f64 = w.iter().map(|x| x.1).sum();
        assert!((sum - 1.).abs() < 1e-12);
        // 135E is halfway between 90E (index 3) and 180E (wraps to index 0).
        assert!(w.iter().any(|x| x.0 == 4 + 3 && (x.1 - 0.25).abs() < 1e-12));
        assert!(w.iter().any(|x| x.0 == 4 && (x.1 - 0.25).abs() < 1e-12));
    }
    #[test]
    fn split_terms_conserve_the_total() {
        let ch = |share: &str| Channel { key: share.into(), terms: vec![Term {
            name: "SV".into(), coef: 1., split_by: vec!["A".into(), "B".into()], share: Some(share.into()) }] };
        let chans = vec![ch("A"), ch("B")];
        let get = |n: &str| -> Result<Vec<f64>, String> {
            Ok(match n { "SV" => vec![4., 2.], "A" => vec![1., 0.], "B" => vec![3., 0.], _ => unreachable!() }) };
        let g = combine(&chans, &get, 2).unwrap();
        assert_eq!(g[0], vec![1., 0.]);
        assert_eq!(g[1], vec![3., 2.]); // no partner: whole total to the last member
        for c in 0..2 { assert!((g[0][c] + g[1][c] - [4., 2.][c]).abs() < 1e-12); }
    }
}
