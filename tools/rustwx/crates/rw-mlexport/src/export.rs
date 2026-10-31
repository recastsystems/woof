//! The export: frames in time order, domain by domain, into one Zarr store
//! per domain; `finalize.rs` closes them.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use rayon::prelude::*;
use rw_isobaric::InterfaceStencil;
use serde_json::{json, Value};
use wrf_core::WrfFile;

use crate::error::{fail, refuse, Result};
use crate::frame::{self, FileMeta};
use crate::grid;
use crate::inputs::{self, Candidate};
use crate::ops::{self, Op, Period};
use crate::regrid::{self, Regrid};
use crate::request::{GridKind, Layout, LevelKind, Mode, Request, VariableKind, VariableRow};
use crate::state::{self, DomainState, InputRecord, RegridInfo, State, VarState, STATE_DIR, STATE_SCHEMA};
use crate::vertical::{self, Columns, Lid, Plan, Rule};
use crate::zarr::{self, Dtype};

/// Largest uncompressed chunk before a level variable's levels are split.
pub const CHUNK_LIMIT_BYTES: usize = 64 * 1024 * 1024;

/// Planes with more cells than this are split into horizontal tiles by
/// default (1024 x 1024 cells, a 4 MiB float32 plane).
pub const AUTO_TILE_MIN_CELLS: usize = 1024 * 1024;

/// The default tile edge, in cells, for planes above `AUTO_TILE_MIN_CELLS`.
pub const AUTO_TILE_EDGE: usize = 512;

/// The horizontal chunk shape `(y, x)` for an `ny` x `nx` output plane.
///
/// Breakage the default prevents: a 1 km CONUS plane (about 5,000 x 3,000
/// cells, 60 MB per float32 level) was one chunk, so a training loader
/// reading a 256 x 256 patch decompressed the whole plane for every level
/// of every sample.  Planes up to `AUTO_TILE_MIN_CELLS` keep one chunk, so
/// small domains are written exactly as before.
pub fn spatial_chunks(requested: Option<[usize; 2]>, ny: usize, nx: usize) -> Result<(usize, usize)> {
    match requested {
        None if ny * nx > AUTO_TILE_MIN_CELLS => Ok((AUTO_TILE_EDGE.min(ny), AUTO_TILE_EDGE.min(nx))),
        None | Some([0, 0]) => Ok((ny, nx)),
        Some([y, x]) if y == 0 || x == 0 => Err(refuse(format!(
            "chunk shape {y} x {x} has a zero edge; give two positive cell counts, or 0,0 for whole planes"
        ))),
        Some([y, x]) => Ok((y.min(ny), x.min(nx))),
    }
}

/// Copy tile (`iy`, `ix`) of a `ny` x `nx` plane into a `ty` x `tx` chunk,
/// padding the part past the grid's edge with `fill` (Zarr chunks are
/// always full-size; readers drop the padding).
#[allow(clippy::too_many_arguments)]
fn tile_of<T: Copy>(plane: &[T], ny: usize, nx: usize, ty: usize, tx: usize, iy: usize, ix: usize, fill: T, out: &mut Vec<T>) {
    let (y0, x0) = (iy * ty, ix * tx);
    let w = tx.min(nx - x0);
    for row in 0..ty {
        let y = y0 + row;
        if y < ny {
            let start = y * nx + x0;
            out.extend_from_slice(&plane[start..start + w]);
            out.extend(std::iter::repeat(fill).take(tx - w));
        } else {
            out.extend(std::iter::repeat(fill).take(tx));
        }
    }
}

/// Accumulators may fall by this much between frames (float rounding of
/// the stored values) before the drop is a refusal.
const ACCUMULATION_TOLERANCE_MM: f64 = 0.01;

/// The name of the `below_ground` mask array.
pub const MASK_NAME: &str = "below_ground";

pub type Progress<'a> = &'a mut dyn FnMut(Value);

#[derive(Debug, Default)]
pub struct Outcome {
    pub domains: Vec<String>,
    pub frames: usize,
    pub bytes: u64,
    pub zip: Option<(PathBuf, u64)>,
}

/// The options every append must repeat: two frames written under
/// different options would sit in one array meaning different things.
fn options_digest(request: &Request) -> String {
    let fixed = json!({
        "variables": request.variables,
        "levels": request.levels,
        "grid": request.grid,
        "names": request.names,
        "layout": request.layout,
        "skip_unavailable": request.skip_unavailable,
        "spacings_deg": request.spacings_deg,
        "chunk_yx": request.chunk_yx,
        "engine": request.provenance.engine,
        "exporter_version": request.provenance.exporter_version,
        "config_sha256": request.provenance.config_sha256,
        "history_attributes": request.provenance.history_attributes,
        "history_engines": request.provenance.history_engines,
        "history_engine_titles": request.provenance.history_engine_titles,
    });
    frame::sha256_bytes(fixed.to_string().as_bytes())
}

/// The export folder's own artifacts; anything else in it is somebody
/// else's and is never deleted.
fn is_ours(name: &str) -> bool {
    matches!(name, "README.txt" | "ml-export-receipt.json" | "zarr.json" | ".zgroup" | ".zattrs" | ".zmetadata")
        || name == STATE_DIR
        || name.ends_with(".zarr")
}

pub fn zip_path(out: &Path) -> PathBuf {
    let slug = slug(out);
    out.parent().unwrap_or(Path::new(".")).join(format!("{slug}-ml.zip"))
}

pub fn slug(out: &Path) -> String {
    out.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .filter(|n| !n.is_empty() && n != "." && n != "..")
        .unwrap_or_else(|| "ml-export".to_string())
}

/// Make `out` ready for a new export.
fn prepare_out(out: &Path, overwrite: bool, zip: bool) -> Result<()> {
    if out.is_file() {
        return Err(refuse(format!(
            "{} is a file, so the export folder cannot be created there",
            out.display()
        )));
    }
    if out.is_dir() {
        let mut ours = Vec::new();
        for entry in std::fs::read_dir(out)? {
            let entry = entry?;
            let name = entry.file_name().to_string_lossy().into_owned();
            if !is_ours(&name) {
                return Err(refuse(format!(
                    "{} holds {name}, which this exporter did not write; an export there would mix its files with someone else's (choose an empty or new --out)",
                    out.display()
                )));
            }
            ours.push(entry.path());
        }
        if !ours.is_empty() && !overwrite {
            return Err(refuse(format!(
                "{} already holds an export; writing another there would overwrite it (pass --overwrite to replace it)",
                out.display()
            )));
        }
        for path in ours {
            if path.is_dir() {
                std::fs::remove_dir_all(&path)?;
            } else {
                std::fs::remove_file(&path)?;
            }
        }
    }
    let archive = zip_path(out);
    if zip && archive.exists() {
        if !overwrite {
            return Err(refuse(format!(
                "{} already exists; writing the archive would overwrite it (pass --overwrite to replace it)",
                archive.display()
            )));
        }
        std::fs::remove_file(&archive)?;
    }
    std::fs::create_dir_all(out)?;
    Ok(())
}

/// Carry out a request.
pub fn execute(request: Request, progress: Progress<'_>) -> Result<Outcome> {
    if let Some(threads) = request.threads.filter(|&n| n > 0) {
        let _ = rayon::ThreadPoolBuilder::new().num_threads(threads).build_global();
    }
    request.validate()?;
    let out = request.out.clone();
    match request.mode {
        Mode::Run => {
            // Inputs first: a request whose inputs are refused must not
            // leave an empty export folder behind.
            inputs::discover(&request.inputs)?;
            let created = !out.exists();
            prepare_out(&out, request.overwrite, request.zip)?;
            let mut state = new_state(&request);
            let mut outcome = match append(&request, &mut state, progress) {
                Ok(outcome) => outcome,
                Err(error) => {
                    // The folder this call made holds nothing but this
                    // call's partial work; a refused export leaves none.
                    if created {
                        let _ = std::fs::remove_dir_all(&out);
                    }
                    return Err(error);
                }
            };
            let closed = crate::finalize::finalize(&out, &state, request.zip, progress)?;
            outcome.bytes = closed.bytes;
            outcome.zip = closed.zip;
            Ok(outcome)
        }
        Mode::Append => {
            let mut state = match State::load(&out)? {
                Some(state) => {
                    if state.options_digest != options_digest(&request) {
                        return Err(refuse(format!(
                            "{} was started with different options; appending under these would put frames that mean different things into one array",
                            out.display()
                        )));
                    }
                    state
                }
                None => {
                    inputs::discover(&request.inputs)?;
                    prepare_out(&out, request.overwrite, false)?;
                    new_state(&request)
                }
            };
            append(&request, &mut state, progress)
        }
        Mode::Finalize => {
            let state = State::load(&out)?.ok_or_else(|| {
                refuse(format!(
                    "{} holds no unfinished export, so there is nothing to finalize",
                    out.display()
                ))
            })?;
            let archive = zip_path(&out);
            if request.zip && archive.exists() {
                if !request.overwrite {
                    return Err(refuse(format!(
                        "{} already exists; writing the archive would overwrite it (pass --overwrite to replace it)",
                        archive.display()
                    )));
                }
                std::fs::remove_file(&archive)?;
            }
            let closed = crate::finalize::finalize(&out, &state, request.zip, progress)?;
            Ok(Outcome {
                domains: state.domains.keys().cloned().collect(),
                frames: state.domains.values().map(|d| d.times.len()).sum(),
                bytes: closed.bytes,
                zip: closed.zip,
            })
        }
    }
}

fn new_state(request: &Request) -> State {
    let mut stored = request.clone();
    stored.inputs.clear();
    State {
        schema: STATE_SCHEMA.to_string(),
        options_digest: options_digest(request),
        request: stored,
        domains: BTreeMap::new(),
    }
}

/// One frame to read: a candidate input and a time record in it.
struct FrameRef {
    candidate: usize,
    domain: String,
    valid: i64,
    time_index: usize,
    /// The run's start, when the file was opened to list its times.
    start: Option<i64>,
}

fn frame_list(request: &Request, candidates: &[Candidate]) -> Result<Vec<FrameRef>> {
    let wanted: Option<BTreeSet<&str>> = request
        .domains
        .as_ref()
        .map(|list| list.iter().map(String::as_str).collect());
    let mut frames = Vec::new();
    for (index, candidate) in candidates.iter().enumerate() {
        // A history file can carry many records even inside gzip or ZIP.
        // Inspect Times in every source form; the filename describes only
        // its first record and cannot stand for the records that follow.
        // The temporary copy drops here, keeping scratch to one input file.
        let scratch = state::state_dir(&request.out).join("inputs");
        let on_disk = inputs::materialize(candidate, &scratch)?;
        let opened = frame::open(&on_disk.path, &candidate.name, candidate.hint.as_ref().map(|h| h.0.as_str()), &request.provenance)?;
        for (time_index, &valid) in opened.times.iter().enumerate() {
            frames.push(FrameRef {
                candidate: index,
                domain: opened.meta.domain.clone(),
                valid,
                time_index,
                start: opened.meta.simulation_start,
            });
        }
    }
    if let Some(wanted) = &wanted {
        frames.retain(|f| wanted.contains(f.domain.as_str()));
        if frames.is_empty() {
            return Err(refuse(format!(
                "no input frame belongs to the domains asked for ({}), so there is nothing to export",
                wanted.iter().copied().collect::<Vec<_>>().join(", ")
            )));
        }
    }
    let start = request.start.as_deref().map(|s| crate::times::parse_named(s, "--start")).transpose()?;
    let end = request.end.as_deref().map(|s| crate::times::parse_named(s, "--end")).transpose()?;
    frames.retain(|f| start.is_none_or(|s| f.valid >= s) && end.is_none_or(|e| f.valid <= e));
    frames.sort_by(|a, b| (a.domain.as_str(), a.valid).cmp(&(b.domain.as_str(), b.valid)));
    for pair in frames.windows(2) {
        if pair[0].domain == pair[1].domain && pair[0].valid == pair[1].valid {
            return Err(refuse(format!(
                "two inputs hold domain {} at {}, so two frames would be written into one time slot",
                pair[0].domain,
                crate::times::iso(pair[0].valid)
            )));
        }
    }
    if frames.is_empty() {
        return Err(refuse("no input frame falls inside --start and --end, so there is nothing to export"));
    }
    Ok(frames)
}

fn append(request: &Request, state: &mut State, progress: Progress<'_>) -> Result<Outcome> {
    let out = request.out.clone();
    let candidates = inputs::discover(&request.inputs)?;
    let mut frames = frame_list(request, &candidates)?;
    if let Some(every) = request.every_hours {
        let step = (every * 3600.0).round() as i64;
        let mut starts: BTreeMap<String, i64> = BTreeMap::new();
        for f in &frames {
            let known = state.domains.get(&f.domain).map(|d| d.init_time).or(f.start);
            let entry = starts.entry(f.domain.clone()).or_insert(known.unwrap_or(f.valid));
            if known.is_none() {
                *entry = (*entry).min(f.valid);
            }
        }
        frames.retain(|f| step > 0 && (f.valid - starts[&f.domain]).rem_euclid(step) == 0);
        if frames.is_empty() {
            return Err(refuse(format!(
                "no frame lies on a multiple of {every} h from the start, so --every selects nothing"
            )));
        }
    }
    let total = frames.len();
    let scratch = state::state_dir(&out).join("inputs");
    let mut hashes: BTreeMap<usize, String> = BTreeMap::new();
    let mut done = 0usize;
    let mut written_total = 0u64;
    let mut i = 0;
    while i < frames.len() {
        // All frames of one candidate are read from one materialized copy.
        let candidate_index = frames[i].candidate;
        let mut group = vec![i];
        let mut j = i + 1;
        while j < frames.len() && frames[j].candidate == candidate_index && frames[j].domain == frames[i].domain {
            group.push(j);
            j += 1;
        }
        let candidate = &candidates[candidate_index];
        let read_clock = Instant::now();
        let on_disk = inputs::materialize(candidate, &scratch)?;
        let hash_path = on_disk.path.clone();
        let hasher = if hashes.contains_key(&candidate_index) {
            None
        } else {
            Some(std::thread::spawn(move || frame::sha256_file(&hash_path)))
        };
        let opened = frame::open(&on_disk.path, &candidate.name, candidate.hint.as_ref().map(|h| h.0.as_str()), &request.provenance)?;
        let mut read_seconds = read_clock.elapsed().as_secs_f64();
        let file = &opened.file;
        let mut results = Vec::new();
        for &g in &group {
            let f = &frames[g];
            let time_index = f.time_index;
            if opened.times.get(time_index) != Some(&f.valid) {
                return Err(refuse(format!(
                    "{} changed its Times records while it was being exported, so a time slot would hold the wrong frame",
                    candidate.describe()
                )));
            }
            if opened.meta.domain != f.domain {
                return Err(refuse(format!(
                    "{} is named for domain {} but its GRID_ID says {}, so it cannot be placed",
                    candidate.describe(),
                    f.domain,
                    opened.meta.domain
                )));
            }
            let clock = Instant::now();
            let new_domain = !state.domains.contains_key(&f.domain);
            let bytes = export_frame(request, state, &out, file, &opened.meta, time_index, f.valid)?;
            file.clear_cache();
            if new_domain {
                let d = &state.domains[&f.domain];
                progress(json!({
                    "event": "domain",
                    "domain": d.id,
                    "grid": [d.out_ny, d.out_nx],
                    "horizontal_grid": d.regrid.as_ref().map(|r| r.description.clone()).unwrap_or_else(|| d.native.description.clone()),
                    "levels": d.levels_kept,
                    "levels_dropped_above_model_top": d.levels_dropped,
                    "model_top_hpa": (d.lid_pa / 10.0).round() / 10.0,
                    "omitted": d.variables.iter().filter(|v| v.omitted.is_some()).map(|v| json!({"variable": v.id, "reason": v.omitted})).collect::<Vec<_>>(),
                }));
            }
            results.push((g, time_index, clock.elapsed().as_secs_f64(), bytes));
        }
        let wait = Instant::now();
        let sha = match hasher {
            Some(handle) => {
                let sha = handle.join().map_err(|_| fail("the hashing thread stopped"))??;
                hashes.insert(candidate_index, sha.clone());
                sha
            }
            None => hashes[&candidate_index].clone(),
        };
        read_seconds += wait.elapsed().as_secs_f64();
        let read_share = read_seconds / results.len().max(1) as f64;
        drop(opened);
        drop(on_disk);
        for (g, time_index, seconds, bytes) in results {
            let f = &frames[g];
            let domain = state.domains.get_mut(&f.domain).expect("domain initialised by its first frame");
            domain.inputs.push(InputRecord {
                name: candidate.name.clone(),
                sha256: sha.clone(),
                time_index,
                valid: f.valid,
                seconds,
                read_seconds: read_share,
                bytes_written: bytes,
            });
            done += 1;
            written_total += bytes;
            progress(json!({
                "event": "frame",
                "frame": done,
                "of": total,
                "domain": f.domain,
                "valid": crate::times::iso(f.valid),
                "seconds": (seconds * 1000.0).round() / 1000.0,
                "read_seconds": (read_share * 1000.0).round() / 1000.0,
                "bytes": bytes,
            }));
        }
        // Saved and read back after every input, in every mode: the state's
        // numbers are then the same parsed values whether the frames came
        // in one call or one call each, which is what makes the two
        // exports the same bytes.
        state.save(&out)?;
        *state = State::load(&out)?.ok_or_else(|| fail("the export state vanished while it was being written"))?;
        i = j;
    }
    let _ = std::fs::remove_dir_all(&scratch);
    Ok(Outcome {
        domains: state.domains.keys().cloned().collect(),
        frames: done,
        bytes: written_total,
        zip: None,
    })
}

/// Everything one frame of one domain needs from its file before any
/// variable is made.
struct Base {
    p: Arc<[f64]>,
    psfc: Vec<f64>,
    phi_stag: Arc<[f64]>,
    temperature: Arc<[f64]>,
    p_top: Option<f64>,
    /// Where each layer interface sits between the mass levels, when the
    /// file states its eta levels (ZNW): geopotential is then read between
    /// interfaces ([`Plan::apply_geopotential_between_interfaces`]).
    stencil: Option<InterfaceStencil>,
}

fn require_fields(file: &WrfFile, fields: &[&str], why: &str) -> Result<()> {
    let missing: Vec<&str> = fields.iter().copied().filter(|f| !file.has_var(f)).collect();
    if missing.is_empty() {
        Ok(())
    } else {
        Err(refuse(format!(
            "the history file lacks {}, which {why}",
            missing.join(", ")
        )))
    }
}

fn read_base(file: &WrfFile, t: usize) -> Result<Base> {
    require_fields(
        file,
        &["P", "PB", "T", "PH", "PHB", "PSFC"],
        "putting fields on pressure levels needs (pressure, temperature and geopotential below ground are made from them)",
    )?;
    let p = file.full_pressure(t).map_err(|e| fail(format!("pressure: {e}")))?;
    let phi_stag = file.geopotential_stag(t).map_err(|e| fail(format!("geopotential: {e}")))?;
    let temperature = file.temperature(t).map_err(|e| fail(format!("temperature: {e}")))?;
    let psfc = ops::read(file, t, "PSFC")?;
    let p_top = read_p_top(file, t)?;
    let stencil = read_interface_stencil(file, t)?;
    Ok(Base { p, psfc, phi_stag, temperature, p_top, stencil })
}

/// The file's interface stencil from ZNW (and ZNU when it carries it,
/// otherwise the middle of each layer in ZNW, as WRF defines it), or `None`
/// for a file that states no eta levels: a height-coordinate frame, whose
/// mass level is the middle of its layer in height, so its layer-mean
/// geopotential already belongs to the mass-level pressure.
fn read_interface_stencil(file: &WrfFile, t: usize) -> Result<Option<InterfaceStencil>> {
    if !file.has_var("ZNW") {
        return Ok(None);
    }
    let eta_interface = ops::read(file, t, "ZNW")?;
    let stencil = if file.has_var("ZNU") {
        InterfaceStencil::new(&ops::read(file, t, "ZNU")?, &eta_interface)
    } else {
        InterfaceStencil::from_interfaces(&eta_interface)
    }
    .map_err(|e| refuse(format!("the history file's eta levels (ZNU, ZNW) cannot place its layer interfaces: {e}")))?;
    if stencil.levels() != file.nz {
        return Err(refuse(format!(
            "the history file's eta levels describe {} layers and its fields {}, so geopotential cannot be read between its layer interfaces",
            stencil.levels(),
            file.nz
        )));
    }
    Ok(Some(stencil))
}

fn read_p_top(file: &WrfFile, t: usize) -> Result<Option<f64>> {
    Ok(if file.has_var("P_TOP") {
        ops::read(file, t, "P_TOP")?.first().copied().filter(|v| v.is_finite() && *v > 0.0)
    } else {
        None
    })
}

/// The output grid's transform of a native plane.
enum Transform<'a> {
    Identity,
    Reorder(&'a grid::Regular, usize),
    Regrid(&'a Regrid),
}

impl Transform<'_> {
    fn plane(&self, plane: &[f32]) -> Vec<f32> {
        match self {
            Transform::Identity => plane.to_vec(),
            Transform::Reorder(regular, nx) => grid::reorder(plane, *nx, regular),
            Transform::Regrid(regrid) => regrid.apply(plane),
        }
    }

    fn stack(&self, stack: &[f32], cells: usize) -> Vec<f32> {
        match self {
            Transform::Identity => stack.to_vec(),
            _ => stack.chunks(cells).flat_map(|plane| self.plane(plane)).collect(),
        }
    }
}

fn level_chunking(levels: usize, plane_bytes: usize) -> usize {
    if levels == 0 {
        return 1;
    }
    if plane_bytes >= CHUNK_LIMIT_BYTES {
        return 1;
    }
    let chunks = (levels * plane_bytes).div_ceil(CHUNK_LIMIT_BYTES).max(1);
    levels.div_ceil(chunks)
}

/// Time chunk index prefix for time position `t`.
fn time_index(layout: Layout, t: usize) -> Vec<usize> {
    match layout {
        Layout::Analysis => vec![t],
        Layout::Forecast | Layout::Earth2 => vec![0, t],
    }
}

/// The output plane and its chunk shape.
#[derive(Clone, Copy)]
struct Tiling {
    ny: usize,
    nx: usize,
    ty: usize,
    tx: usize,
    face_size: Option<usize>,
}

impl Tiling {
    fn of(domain: &DomainState) -> Tiling {
        let (ty, tx) = domain.chunk_yx();
        Tiling { ny: domain.out_ny, nx: domain.out_nx, ty, tx, face_size: None }
    }

    fn whole(&self) -> bool {
        self.ty >= self.ny && self.tx >= self.nx
    }

    fn tiles(&self) -> impl Iterator<Item = (usize, usize)> {
        let (cy, cx) = (self.ny.div_ceil(self.ty), self.nx.div_ceil(self.tx));
        (0..cy).flat_map(move |iy| (0..cx).map(move |ix| (iy, ix)))
    }
}

#[allow(clippy::too_many_arguments)]
fn write_level_chunks<T: Copy>(
    store: &Path,
    name: &str,
    data: &[T],
    fill: T,
    to_bytes: impl Fn(&[T]) -> Vec<u8>,
    dtype: Dtype,
    layout: Layout,
    t: usize,
    tiling: Tiling,
    per_chunk: usize,
) -> Result<u64> {
    let out_cells = tiling.ny * tiling.nx;
    let levels = data.len() / out_cells;
    let mut written = 0;
    let mut chunk_index = 0;
    let mut start = 0;
    while start < levels {
        let end = (start + per_chunk).min(levels);
        if tiling.whole() {
            let mut values = data[start * out_cells..end * out_cells].to_vec();
            values.resize(per_chunk * out_cells, fill);
            let mut index = time_index(layout, t);
            index.extend([chunk_index, 0, 0]);
            written += zarr::write_chunk(store, name, &index, &to_bytes(&values), dtype)?;
        } else {
            let tile_cells = tiling.ty * tiling.tx;
            for (iy, ix) in tiling.tiles() {
                let mut values = Vec::with_capacity(per_chunk * tile_cells);
                for level in start..end {
                    let plane = &data[level * out_cells..(level + 1) * out_cells];
                    tile_of(plane, tiling.ny, tiling.nx, tiling.ty, tiling.tx, iy, ix, fill, &mut values);
                }
                values.resize(per_chunk * tile_cells, fill);
                let mut index = time_index(layout, t);
                index.extend([chunk_index, iy, ix]);
                written += zarr::write_chunk(store, name, &index, &to_bytes(&values), dtype)?;
            }
        }
        chunk_index += 1;
        start = end;
    }
    Ok(written)
}

fn write_surface_chunk(store: &Path, name: &str, plane: &[f32], layout: Layout, t: usize, tiling: Tiling) -> Result<u64> {
    if tiling.whole() {
        let mut index = time_index(layout, t);
        index.extend([0, 0]);
        return zarr::write_chunk(store, name, &index, &zarr::f32_bytes(plane), Dtype::F32);
    }
    let mut written = 0;
    for (iy, ix) in tiling.tiles() {
        let mut values = Vec::with_capacity(tiling.ty * tiling.tx);
        tile_of(plane, tiling.ny, tiling.nx, tiling.ty, tiling.tx, iy, ix, f32::NAN, &mut values);
        let mut index = time_index(layout, t);
        if tiling.face_size.is_some() { index.extend([iy, 0, ix]); } else { index.extend([iy, ix]); }
        written += zarr::write_chunk(store, name, &index, &zarr::f32_bytes(&values), Dtype::F32)?;
    }
    Ok(written)
}

/// Check each selected row against the first frame's file; returns the
/// variables' states (omitted ones carry their reason).
fn check_rows(request: &Request, file: &WrfFile) -> Result<Vec<VarState>> {
    let mut states = Vec::new();
    for row in &request.variables {
        let op = Op::parse(&row.op)?;
        let name = row.out_name(&request.names)?;
        let level_op = matches!(
            op,
            Op::Geopotential | Op::Temperature | Op::EarthWind(_) | Op::SpecificHumidity | Op::Omega | Op::RhIfs | Op::Pressure
        );
        match row.kind {
            VariableKind::Level if !(level_op || matches!(op, Op::Raw(_))) => {
                return Err(refuse(format!(
                    "level variable '{}' names operator '{}', which makes a surface field",
                    row.id, row.op
                )))
            }
            VariableKind::Surface | VariableKind::Static if level_op => {
                return Err(refuse(format!(
                    "{} variable '{}' names operator '{}', which makes a field on levels",
                    if row.kind == VariableKind::Static { "static" } else { "surface" },
                    row.id,
                    row.op
                )))
            }
            _ => {}
        }
        if op == Op::Pressure && request.levels.kind == LevelKind::Pressure {
            return Err(refuse(format!(
                "'{}' is pressure on pressure levels, which is the level coordinate itself; it is a model-level variable",
                row.id
            )));
        }
        Rule::parse(row.below_ground.as_deref())?;
        let missing: Vec<&String> = row.fields.iter().filter(|f| !file.has_var(f)).collect();
        if missing.is_empty() {
            states.push(VarState { id: row.id.clone(), name, omitted: None });
        } else {
            let reason = format!(
                "the history files lack {}",
                missing.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", ")
            );
            if !request.skip_unavailable {
                return Err(refuse(format!(
                    "'{}' cannot be made because {reason}; a dataset without a variable that was asked for would be found out only deep in a training job (pass --skip-unavailable to write the rest and record the omission)",
                    row.id
                )));
            }
            states.push(VarState { id: row.id.clone(), name, omitted: Some(reason) });
        }
    }
    Ok(states)
}

/// The history file attributes that describe a run's configuration (grid,
/// projection, physics, vertical coordinate), digested when the caller has
/// no configuration file to digest.  A fixed list rather than every
/// attribute: the reader cannot enumerate attributes, and a list makes the
/// digest's meaning stable.
pub const CONFIG_ATTRIBUTES: [&str; 36] = [
    "BL_PBL_PHYSICS", "BOTTOM-TOP_GRID_DIMENSION", "CEN_LAT", "CEN_LON", "CU_PHYSICS",
    "DAMP_OPT", "DIFF_OPT", "DT", "DX", "DY", "ETAC", "GRID_ID", "HYBRID_OPT",
    "I_PARENT_START", "J_PARENT_START", "KM_OPT", "MAP_PROJ", "MOAD_CEN_LAT", "MP_PHYSICS",
    "PARENT_GRID_RATIO", "PARENT_ID", "POLE_LAT", "POLE_LON", "RA_LW_PHYSICS", "RA_SW_PHYSICS",
    "SF_LAKE_PHYSICS", "SF_SFCLAY_PHYSICS", "SF_SURFACE_PHYSICS", "SHCU_PHYSICS",
    "SIMULATION_START_DATE", "SOUTH-NORTH_GRID_DIMENSION", "STAND_LON", "TRUELAT1", "TRUELAT2",
    "W_DAMPING", "WEST-EAST_GRID_DIMENSION",
];
pub const CONFIG_ATTRIBUTE_KIND: &str = "wrfout-config-attributes-v2";

fn config_attribute_digest(file: &WrfFile, meta: &FileMeta) -> String {
    let mut lines: Vec<String> = Vec::new();
    for name in CONFIG_ATTRIBUTES {
        // Numbers first, through f64: the classic reader spells a float
        // attribute's string as its f64 widening, the NetCDF-4 reader as
        // its f32, so reading text first gave a compressed (NetCDF-4) frame
        // a different digest from the classic frame of the same run (V6000
        // A/B, 2026-10-28). Through f64 both containers spell it the way
        // the classic frame always did, so no existing digest moves.
        let text = file
            .global_attr_f64(name)
            .ok()
            .map(|v| format!("{v}"))
            .or_else(|| file.global_attr_str(name).ok());
        if let Some(text) = text {
            lines.push(format!("{name}={}", text.trim_end_matches(char::from(0)).trim()));
        }
    }
    if let Some(version) = &meta.gpuwm_version {
        lines.push(format!("MODEL_VERSION={version}"));
    }
    lines.sort();
    frame::sha256_bytes(lines.join("\n").as_bytes())
}

fn read_latlon(file: &WrfFile, t: usize) -> Result<(Vec<f64>, Vec<f64>)> {
    require_fields(file, &["XLAT", "XLONG"], "places the grid on the earth")?;
    let lat = ops::read(file, t, "XLAT")?;
    let lon = ops::read(file, t, "XLONG")?;
    Ok((lat, lon))
}

fn init_domain(request: &Request, file: &WrfFile, meta: &FileMeta, t: usize, valid: i64, out: &Path) -> Result<(DomainState, Option<Regrid>)> {
    let (xlat, xlong) = read_latlon(file, t)?;
    let native = grid::describe(meta, &xlat, &xlong);
    let regrid = match request.grid.kind {
        GridKind::Native => None,
        GridKind::Latlon => Some(regrid::plan(
            meta,
            &native,
            &xlat,
            &xlong,
            request.grid.deg,
            request.grid.method,
            &request.spacings_deg,
            request.grid.target.as_ref(),
        )?),
    };
    let (out_ny, out_nx) = match (&regrid, &native.regular) {
        (Some(r), _) => (r.ny(), r.nx()),
        (None, Some(regular)) => (regular.lat.len(), regular.lon.len()),
        (None, None) => (meta.ny, meta.nx),
    };
    let variables = check_rows(request, file)?;
    let has_level_fields = variables.iter().any(|v| v.omitted.is_none()
        && row_of(request, &v.id).kind == VariableKind::Level);
    let (mut kept, mut dropped, mut model_levels, mut eta) = (Vec::new(), Vec::new(), Vec::new(), None);
    let (mut lid_pa, mut lid_stated) = (0.0, false);
    let mut geopotential_between_interfaces = false;
    match request.levels.kind {
        LevelKind::Pressure if has_level_fields => {
            let base = read_base(file, t)?;
            geopotential_between_interfaces = base.stencil.is_some();
            let lid = match base.p_top {
                Some(top) => Lid::PTop(top),
                None => Lid::TopMassLevel,
            };
            let (k, d) = vertical::kept_levels(&request.levels.hpa, lid, &base.p, meta.nz, meta.nx * meta.ny);
            kept = k;
            dropped = d;
            lid_stated = base.p_top.is_some();
            lid_pa = match lid {
                Lid::PTop(top) => top,
                Lid::TopMassLevel => {
                    let cells = meta.nx * meta.ny;
                    base.p[(meta.nz - 1) * cells..meta.nz * cells]
                        .iter()
                        .copied()
                        .filter(|v| v.is_finite())
                        .fold(f64::MIN, f64::max)
                }
            };
            if kept.is_empty() && variables.iter().any(|v| v.omitted.is_none()) {
                return Err(refuse(format!(
                    "every requested level lies above the model top ({:.1} hPa), so no level would hold model data",
                    lid_pa / 100.0
                )));
            }
        }
        LevelKind::Model if has_level_fields => {
            model_levels = if request.levels.model_levels.is_empty() {
                (1..=meta.nz).collect()
            } else {
                request.levels.model_levels.clone()
            };
            if let Some(&bad) = model_levels.iter().find(|&&k| k == 0 || k > meta.nz) {
                return Err(refuse(format!(
                    "model level {bad} does not exist in a {}-level history file, so it cannot be selected",
                    meta.nz
                )));
            }
            if file.has_var("ZNU") {
                let znu = ops::read(file, t, "ZNU")?;
                if znu.len() == meta.nz {
                    eta = Some(model_levels.iter().map(|&k| znu[k - 1]).collect());
                }
            }
        }
        _ => {}
    }
    let levels = match request.levels.kind {
        LevelKind::Pressure => kept.len(),
        LevelKind::Model => model_levels.len(),
    };
    let (chunk_ny, chunk_nx) = spatial_chunks(request.chunk_yx, out_ny, out_nx)?;
    let levels_per_chunk = level_chunking(levels, chunk_ny * chunk_nx * 4);
    state::save_plane(out, &format!("{}-xlat-0.f64", meta.domain), &xlat)?;
    state::save_plane(out, &format!("{}-xlong-0.f64", meta.domain), &xlong)?;
    let init_time = meta.simulation_start.unwrap_or(valid);
    let (config_digest, config_digest_kind) = match &request.provenance.config_sha256 {
        Some(sha) => (sha.clone(), "experiment.toml".to_string()),
        None => (config_attribute_digest(file, meta), CONFIG_ATTRIBUTE_KIND.to_string()),
    };
    let domain = DomainState {
        id: meta.domain.clone(),
        meta: meta.clone(),
        native,
        regrid: regrid.as_ref().map(|r| RegridInfo {
            lat: r.lat.clone(),
            lon: r.lon.clone(),
            deg: r.deg,
            description: r.description.clone(),
            boundary_rows: r.boundary_rows,
        }),
        out_ny,
        out_nx,
        init_time,
        times: Vec::new(),
        inputs: Vec::new(),
        levels_kept: kept,
        levels_dropped: dropped,
        model_levels,
        eta,
        lid_pa,
        lid_stated,
        geopotential_between_interfaces,
        levels_per_chunk,
        chunk_ny,
        chunk_nx,
        variables,
        latlon_version: Vec::new(),
        latlon_versions: 1,
        period_partners: BTreeMap::new(),
        config_digest,
        config_digest_kind,
    };
    Ok((domain, regrid))
}

/// Memory one frame needs, refused before reading rather than killed
/// part-way with half a dataset written.
fn memory_preflight(meta: &FileMeta, levels: usize) -> Result<()> {
    let cells = (meta.nx * meta.ny) as u128;
    let volume = cells * (meta.nz as u128 + 1) * 8;
    // wrf-core's cached intermediates (pressure, theta, temperature, both
    // geopotentials, the winds, moisture) plus one field being made and its
    // pressure-level planes and plan.
    let need = volume * 12 + cells * levels as u128 * (8 + 2 + 1 + 4 * 2);
    if let Some(available) = rw_host_memory::available_bytes() {
        if need > u128::from(available) {
            return Err(refuse(format!(
                "one frame of this grid ({} x {} x {}) needs about {:.1} GiB and this host has {:.1} GiB available; the host would kill the export part-way",
                meta.nx,
                meta.ny,
                meta.nz,
                need as f64 / f64::from(1u32 << 30),
                available as f64 / f64::from(1u32 << 30)
            )));
        }
    }
    Ok(())
}

fn row_of<'a>(request: &'a Request, id: &str) -> &'a VariableRow {
    request.variables.iter().find(|r| r.id == id).expect("state rows come from the request")
}

fn export_frame(
    request: &Request,
    state: &mut State,
    out: &Path,
    file: &WrfFile,
    meta: &FileMeta,
    t: usize,
    valid: i64,
) -> Result<u64> {
    let mut fresh_regrid = None;
    if !state.domains.contains_key(&meta.domain) {
        let levels = match request.levels.kind {
            LevelKind::Pressure => request.levels.hpa.len(),
            LevelKind::Model => meta.nz,
        };
        memory_preflight(meta, levels)?;
        let (domain, regrid) = init_domain(request, file, meta, t, valid, out)?;
        fresh_regrid = regrid;
        state.domains.insert(meta.domain.clone(), domain);
    }
    let domain = state.domains.get_mut(&meta.domain).expect("inserted above");
    if !domain.meta.same_grid(meta) {
        return Err(refuse(format!(
            "frames of domain {} come from two different grids ({}x{}x{} against {}x{}x{}, or a different projection); stacking them would put two grids in one array",
            meta.domain, domain.meta.nx, domain.meta.ny, domain.meta.nz, meta.nx, meta.ny, meta.nz
        )));
    }
    if domain.meta.simulation_start != meta.simulation_start
        || domain.meta.gpuwm_version != meta.gpuwm_version
        || domain.meta.source_engine != meta.source_engine
        || domain.meta.ic_source != meta.ic_source
        || domain.meta.ic_cycle != meta.ic_cycle {
        return Err(refuse(format!(
            "frames of domain {} have different initialization, engine version or analysis provenance; stacking them would label unrelated forecasts as one run",
            meta.domain
        )));
    }
    let last_time = domain.times.last().copied();
    if let Some(last) = last_time {
        if valid <= last {
            return Err(refuse(format!(
                "domain {} already holds {} and this frame is {}; a duplicate or backwards time would put two frames into one time slot",
                meta.domain,
                crate::times::iso(last),
                crate::times::iso(valid)
            )));
        }
    }
    let regrid = match fresh_regrid {
        Some(r) => Some(r),
        None if request.grid.kind == GridKind::Latlon => {
            let xlat = state::load_plane(out, &format!("{}-xlat-0.f64", meta.domain))?
                .ok_or_else(|| fail("the export state lost the first frame's latitudes"))?;
            let xlong = state::load_plane(out, &format!("{}-xlong-0.f64", meta.domain))?
                .ok_or_else(|| fail("the export state lost the first frame's longitudes"))?;
            Some(regrid::plan(
                &domain.meta,
                &domain.native,
                &xlat,
                &xlong,
                request.grid.deg,
                request.grid.method,
                &request.spacings_deg,
            request.grid.target.as_ref(),
            )?)
        }
        None => None,
    };
    let (nx, nz) = (meta.nx, meta.nz);
    let cells = nx * meta.ny;
    let out_cells = domain.out_ny * domain.out_nx;
    let t_pos = domain.times.len();
    let store = out.join(format!("{}.zarr", meta.domain));
    let layout = request.layout;
    let regular = domain.native.regular.clone();
    let levels_per_chunk = domain.levels_per_chunk;
    let mut tiling = Tiling::of(domain);
    tiling.face_size = crate::earth2::nside(request);
    let model_levels = domain.model_levels.clone();
    let levels_kept = domain.levels_kept.clone();
    let lid = if domain.lid_stated { Lid::PTop(domain.lid_pa) } else { Lid::TopMassLevel };
    // Column integrals use the source's lid regardless of the selected
    // output coordinate. Choosing model levels must not change tcwv.
    let p_top_stated = if domain.variables.iter().any(|v| v.omitted.is_none()
        && row_of(request, &v.id).op == "column-vapour") {
        read_p_top(file, t)?
    } else {
        None
    };
    let variables = domain.variables.clone();

    // Moving nests: the grid's place on the earth, per frame.
    let (xlat, xlong) = read_latlon(file, t)?;
    let latest = domain.latlon_versions - 1;
    let previous_lat = state::load_plane(out, &format!("{}-xlat-{latest}.f64", meta.domain))?.unwrap_or_default();
    let previous_lon = state::load_plane(out, &format!("{}-xlong-{latest}.f64", meta.domain))?.unwrap_or_default();
    if previous_lat != xlat || previous_lon != xlong {
        if regrid.is_some() {
            return Err(refuse(format!(
                "domain {} moves between frames (a moving nest); one fixed latitude-longitude box would hold different ground at each time (use --grid native, which carries time-varying coordinates)",
                meta.domain
            )));
        }
        let version = domain.latlon_versions;
        state::save_plane(out, &format!("{}-xlat-{version}.f64", meta.domain), &xlat)?;
        state::save_plane(out, &format!("{}-xlong-{version}.f64", meta.domain), &xlong)?;
        domain.latlon_versions += 1;
    }
    let latlon_version = domain.latlon_versions - 1;

    let transform = match (&regrid, &regular) {
        (Some(r), _) => Transform::Regrid(r),
        (None, Some(reg)) => Transform::Reorder(reg, nx),
        (None, None) => Transform::Identity,
    };

    // Base fields and the vertical plan, when anything goes on pressure
    // levels.
    let pressure_mode = request.levels.kind == LevelKind::Pressure;
    let any_level = variables
        .iter()
        .any(|v| v.omitted.is_none() && row_of(request, &v.id).kind == VariableKind::Level);
    let base = if pressure_mode && any_level { Some(read_base(file, t)?) } else { None };
    if let Some(b) = &base {
        if b.stencil.is_some() != domain.geopotential_between_interfaces {
            return Err(refuse(format!(
                "domain {}'s first frame {} its eta levels (ZNW) and this frame {}; geopotential would be read between layer interfaces in some frames and at mass levels in others, two methods in one array",
                meta.domain,
                if domain.geopotential_between_interfaces { "states" } else { "does not state" },
                if b.stencil.is_some() { "does" } else { "does not" }
            )));
        }
    }
    let derived = base.as_ref().map(|b| {
        let t_bot: Vec<f64> = b.temperature[..cells].to_vec();
        let t_top: Vec<f64> = b.temperature[(nz - 1) * cells..nz * cells].to_vec();
        let phi_sfc: Vec<f64> = b.phi_stag[..cells].to_vec();
        let phi_lid: Vec<f64> = b.phi_stag[nz * cells..(nz + 1) * cells].to_vec();
        let phi_top: Vec<f64> = (0..cells)
            .map(|c| 0.5 * (b.phi_stag[(nz - 1) * cells + c] + b.phi_stag[nz * cells + c]))
            .collect();
        (t_bot, t_top, phi_sfc, phi_lid, phi_top)
    });
    let columns = match (&base, &derived) {
        (Some(b), Some((t_bot, t_top, phi_sfc, phi_lid, phi_top))) => Some(Columns {
            nz,
            cells,
            p: &b.p,
            psfc: &b.psfc,
            phi_sfc,
            t_bot,
            t_top,
            phi_top,
            phi_lid,
            lid,
        }),
        _ => None,
    };
    let plan = columns.as_ref().map(|cols| Plan::new(&levels_kept, cols));

    // Accumulated precipitation, once per frame, when any row needs it.
    let needs_accumulation = variables
        .iter()
        .any(|v| v.omitted.is_none() && is_accumulation(row_of(request, &v.id)));
    let accumulation = if needs_accumulation {
        let a = ops::accumulated_precipitation(file, t, meta.bucket_mm)?;
        state::save_plane(out, &format!("{}-acc-{valid}.f64", meta.domain), &a)?;
        Some(a)
    } else {
        None
    };

    // Channel work is independent after the frame's common fields and
    // accumulation are available. Keep single-level consumers parallel,
    // including reads, diagnostics, regridding, compression and writes.
    let outputs = variables.par_iter().map(|var| -> Result<(u64, Vec<(String, bool)>)> {
        let mut written = 0u64;
        let mut partners = Vec::new();
        if var.omitted.is_some() {
            return Ok((written, partners));
        }
        let row = row_of(request, &var.id);
        let op = Op::parse(&row.op)?;
        let scale = row.scale.unwrap_or(1.0);
        let missing: Vec<&String> = row.fields.iter().filter(|f| !file.has_var(f)).collect();
        if !missing.is_empty() {
            return Err(refuse(format!(
                "'{}' was in the first frame but this frame lacks {}; the array would have a hole in time",
                row.id,
                missing.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", ")
            )));
        }
        match row.kind {
            VariableKind::Level => {
                let single_plan = row.pressure_hpa.and_then(|p| columns.as_ref().map(|cols| Plan::new(&[p],cols)));
                let plan = single_plan.as_ref().or(plan.as_ref());
                let field = ops::level_field(file, t, &op, scale)?;
                let stencil = base.as_ref().and_then(|b| b.stencil.as_ref().map(|s| (b, s)));
                let stack: Vec<f32> = match (plan, &columns, &op, stencil) {
                    (Some(plan), Some(cols), Op::Geopotential, Some((b, stencil))) => plan
                        .apply_geopotential_between_interfaces(
                            &b.phi_stag,
                            &field,
                            stencil,
                            Rule::parse(row.below_ground.as_deref())?,
                            cols,
                        ),
                    (Some(plan), Some(cols), _, _) => plan.apply(&field, Rule::parse(row.below_ground.as_deref())?, cols),
                    _ => vertical::select_model_levels(&field, cells, &model_levels),
                };
                drop(field);
                if let Some(pressure) = row.pressure_hpa {
                    if !levels_kept.contains(&pressure) { return Err(refuse(format!("requested channel {} is above the model lid",var.name))); }
                    let k = 0;
                    let plane = transform.plane(&stack[k * cells..(k + 1) * cells]);
                    written += write_surface_chunk(&store, &var.name, &plane, layout, t_pos, tiling)?;
                    return Ok((written, partners));
                }
                let stack = transform.stack(&stack, cells);
                written += write_level_chunks(
                    &store,
                    &var.name,
                    &stack,
                    f32::NAN,
                    zarr::f32_bytes,
                    Dtype::F32,
                    layout,
                    t_pos,
                    tiling,
                    levels_per_chunk,
                )?;
            }
            VariableKind::Surface => {
                let plane: Vec<f64> = match &op {
                    Op::Accumulation(period) => {
                        let now = accumulation.as_ref().expect("computed above");
                        let back = match period {
                            Period::Interval => last_time,
                            Period::Hours(h) => Some(valid - h * 3600),
                        };
                        let mut earlier = match back {
                            Some(b) => state::load_plane(out, &format!("{}-acc-{b}.f64", meta.domain))?,
                            None => None,
                        };
                        if let (Some(previous), Some(back)) = (&earlier, back) {
                            if let Some(position) = domain.times.iter().position(|&time| time == back) {
                                let version = domain.latlon_version[position];
                                if version != latlon_version {
                                    let old_lat = state::load_plane(out, &format!("{}-xlat-{version}.f64", meta.domain))?
                                        .ok_or_else(|| fail("the export state lost an earlier moving grid's latitudes"))?;
                                    let old_lon = state::load_plane(out, &format!("{}-xlong-{version}.f64", meta.domain))?
                                        .ok_or_else(|| fail("the export state lost an earlier moving grid's longitudes"))?;
                                    earlier = Some(crate::accumulation::align(&domain.meta, &old_lat, &old_lon, &xlat, &xlong, previous)?);
                                }
                            }
                        }
                        let key = match period {
                            Period::Interval => "interval".to_string(),
                            Period::Hours(h) => format!("{h}h"),
                        };
                        partners.push((key, earlier.is_some()));
                        match earlier {
                            None => vec![f64::NAN; cells],
                            Some(earlier) => {
                                let worst = now.par_iter().zip(&earlier)
                                    .map(|(a,b)| a-b).reduce(|| 0.0, f64::min);
                                let diff: Vec<f64> = now.par_iter().zip(&earlier)
                                    .map(|(a,b)| { let d=a-b;
                                        if d.is_finite() { d.max(0.0) / 1000.0 * scale } else { f64::NAN }
                                    }).collect();
                                if worst < -ACCUMULATION_TOLERANCE_MM {
                                    return Err(refuse(format!(
                                        "accumulated precipitation in domain {} falls by {:.3} mm between {} and {}; a bucket reset or restart the file does not describe would publish negative rain",
                                        meta.domain,
                                        -worst,
                                        crate::times::iso(back.unwrap_or(valid)),
                                        crate::times::iso(valid)
                                    )));
                                }
                                diff
                            }
                        }
                    }
                    _ => ops::surface_field(file, t, &op, scale, p_top_stated)?,
                };
                let plane32: Vec<f32> = plane.par_iter().map(|&v| v as f32).collect();
                written += write_surface_chunk(&store, &var.name, &transform.plane(&plane32), layout, t_pos, tiling)?;
            }
            VariableKind::Static => {
                // A moving nest samples different terrain at every time.
                // Keep every frame until finalize knows whether the grid
                // moved; stationary exports collapse to the first plane.
                let plane = ops::surface_field(file, t, &op, scale, None)?;
                let plane32: Vec<f32> = plane.par_iter().map(|&v| v as f32).collect();
                written += write_surface_chunk(&store, &var.name, &transform.plane(&plane32), layout, t_pos, tiling)?;
            }
        }
        Ok((written, partners))
    }).collect::<Result<Vec<_>>>()?;
    let mut written = outputs.iter().map(|(bytes, _)| bytes).sum::<u64>();
    let partners: Vec<(String, bool)> = outputs.into_iter().flat_map(|(_, partners)| partners).collect();

    // The below-ground mask.
    if let (Some(plan), Some(base)) = (&plan, &base) {
      if layout != Layout::Earth2 {
        let mask = match &regrid {
            Some(r) => {
                let psfc32: Vec<f32> = base.psfc.iter().map(|&v| v as f32).collect();
                let sp = r.apply(&psfc32);
                let mut mask = Vec::with_capacity(plan.levels_pa.len() * out_cells);
                for &target in &plan.levels_pa {
                    mask.extend(sp.iter().map(|&s| u8::from(target > f64::from(s))));
                }
                mask
            }
            None => {
                let mask = plan.below_ground(&base.psfc);
                match &regular {
                    Some(reg) => mask.chunks(cells).flat_map(|p| grid::reorder(p, nx, reg)).collect(),
                    None => mask,
                }
            }
        };
        written += write_level_chunks(
            &store,
            MASK_NAME,
            &mask,
            0u8,
            |v: &[u8]| v.to_vec(),
            Dtype::U8,
            layout,
            t_pos,
            tiling,
            levels_per_chunk,
        )?;
    }

    }

    // Keep only the accumulations the longest period still needs.
    if accumulation.is_some() {
        let longest = request
            .variables
            .iter()
            .filter_map(|r| match Op::parse(&r.op) {
                Ok(Op::Accumulation(Period::Hours(h))) => Some(h * 3600),
                _ => None,
            })
            .max()
            .unwrap_or(0);
        for &old in &domain.times {
            if old < valid - longest {
                state::remove_plane(out, &format!("{}-acc-{old}.f64", meta.domain));
            }
        }
    }
    for (key, found) in partners {
        domain.period_partners.entry(key).or_default().push(found);
    }
    domain.latlon_version.push(latlon_version);
    domain.times.push(valid);
    Ok(written)
}

/// `true` when `row` is one of the accumulation rows.
pub fn is_accumulation(row: &VariableRow) -> bool {
    matches!(Op::parse(&row.op), Ok(Op::Accumulation(_)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn levels_split_evenly_under_the_chunk_limit() {
        // A 1132 x 906 plane is 4.1 MB: 13 levels fit in one chunk, 29 split
        // into chunks of 15 and 14.
        let plane = 1132 * 906 * 4;
        assert_eq!(level_chunking(13, plane), 13);
        assert_eq!(level_chunking(29, plane), 15);
        // A plane over the limit is never split.
        assert_eq!(level_chunking(5, CHUNK_LIMIT_BYTES + 1), 1);
    }

    #[test]
    fn large_planes_tile_by_default_and_small_planes_do_not() {
        // 1 km CONUS-sized plane: 512 x 512 tiles.
        assert_eq!(spatial_chunks(None, 3200, 5000).unwrap(), (512, 512));
        // A 3 km HRRR-sized plane (1059 x 1799) is above 1024 x 1024 cells.
        assert_eq!(spatial_chunks(None, 1059, 1799).unwrap(), (512, 512));
        // Small domains keep one chunk per plane, exactly as before.
        assert_eq!(spatial_chunks(None, 300, 400).unwrap(), (300, 400));
        assert_eq!(spatial_chunks(None, 1024, 1024).unwrap(), (1024, 1024));
        // Explicit shapes, the whole-plane opt-out, and a zero edge.
        assert_eq!(spatial_chunks(Some([256, 128]), 3200, 5000).unwrap(), (256, 128));
        assert_eq!(spatial_chunks(Some([4096, 4096]), 300, 400).unwrap(), (300, 400));
        assert_eq!(spatial_chunks(Some([0, 0]), 3200, 5000).unwrap(), (3200, 5000));
        assert!(spatial_chunks(Some([0, 64]), 3200, 5000).is_err());
    }

    #[test]
    fn tiles_cover_the_plane_and_pad_only_past_the_edge() {
        // A 5 x 7 plane in 2 x 3 tiles: 3 x 3 tiles, the last row and column padded.
        let (ny, nx, ty, tx) = (5usize, 7usize, 2usize, 3usize);
        let plane: Vec<i32> = (0..(ny * nx) as i32).collect();
        let tiling = Tiling { ny, nx, ty, tx, face_size: None };
        let mut rebuilt = vec![-1i32; ny * nx];
        let mut tiles = 0;
        for (iy, ix) in tiling.tiles() {
            let mut out = Vec::new();
            tile_of(&plane, ny, nx, ty, tx, iy, ix, -9, &mut out);
            assert_eq!(out.len(), ty * tx);
            for r in 0..ty {
                for c in 0..tx {
                    let (y, x) = (iy * ty + r, ix * tx + c);
                    let v = out[r * tx + c];
                    if y < ny && x < nx {
                        rebuilt[y * nx + x] = v;
                    } else {
                        assert_eq!(v, -9);
                    }
                }
            }
            tiles += 1;
        }
        assert_eq!(tiles, 9);
        assert_eq!(rebuilt, plane);
    }

    #[test]
    fn only_the_exporters_own_files_count_as_ours() {
        assert!(is_ours("d01.zarr"));
        assert!(is_ours(".ml-export-state"));
        assert!(!is_ours("notes.txt"));
    }

    #[test]
    fn the_zip_sits_beside_the_folder() {
        let out = Path::new("exports").join("my-run");
        assert_eq!(zip_path(&out), Path::new("exports").join("my-run-ml.zip"));
        assert_eq!(slug(&out), "my-run");
    }
}
