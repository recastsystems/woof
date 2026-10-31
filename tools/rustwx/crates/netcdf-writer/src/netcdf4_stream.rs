//! Lossless, compressed NetCDF-4 history frames, compressed on every core.
//!
//! # Why
//!
//! A 1 km history frame is 4.69 GB of floats in the classic container and a
//! quick-look frame 539 MB. Shuffle + deflate stores the same words in about
//! a third to a quarter of the bytes (measured on real frames, 2026-10-28:
//! 4.3x on a quick-look frame), and every reader the stack ships already
//! decodes deflate (netCDF-C, `rw_netcdf`, `rw_wrfbatch`, `rw_soundings`).
//! The HDF5 library compresses a dataset on ONE thread, so this writer does
//! not use it: chunks are gathered, shuffled and deflated on a pool of
//! worker threads while the caller goes on handing over the next variable,
//! and only the finished bytes are appended to the file.
//!
//! # What it writes
//!
//! The same file the classic tape would hold, value for value: every
//! dimension (the record dimension stays unlimited), every attribute, every
//! variable in definition order and every word bit for bit. Only the
//! container differs: HDF5 superblock v0, v2 object headers, chunked
//! datasets with the HDF5 shuffle (id 2) and deflate (id 1) filters indexed
//! by version 1 B-trees, the layout netCDF-C writes for a `NETCDF4_CLASSIC`
//! file. `_nc3_strict` marks the classic data model, so netCDF-C reports the
//! format as `NETCDF4_CLASSIC`, exactly like the netCDF4 engine's files.
//!
//! # Chunks
//!
//! Record dimension 1. Every leading dimension (levels, soil layers) whole.
//! The last two (south_north, west_east) in square tiles, as large as keeps
//! one chunk near 4 MiB. A column (a sounding at one point) therefore
//! decompresses one tile per variable, and a whole-variable read (maps, the
//! column store build) decompresses tiles in parallel.
//!
//! # Layout on disk
//!
//! `[superblock][chunks, in submission order][B-tree nodes][object headers]
//! [global heap]`. The superblock is written last, at offset 0, so a file
//! that dies early has no root and no reader can mistake it for a frame.
//!
//! Chunks are compressed in any order but appended in the order they were
//! handed over (a chunk that finishes early waits for the ones before it),
//! so the same frame written the same way is the same bytes on every run and
//! every thread count. Breakage this prevents (2.9.10 pass 1 R6, 2026-10-29):
//! appending in completion order gave equal-size frames with different bytes
//! run to run, and every byte-identity gate on history files (batch vs scalar
//! moist dycore, prepared tree vs ordinary runner, the simulated-radar
//! source keys) went red.
use crate::netcdf4::{
    cstring_attr, datatype, descriptor, fill_message, int_attr, object, scalar_int_attr, space,
    u16v, u64v, user_attrs, vlen_type, Attribute, Heap, Values, Variable,
};
use crate::schema::Schema;
use crate::types::{AttrValue, NcType};
use crate::{NcWriteError, Result};
use std::collections::{BTreeMap, VecDeque};
use std::fs::File;
use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex};

const UNDEF: u64 = u64::MAX;
/// One chunk's raw bytes, at most (the tile size is chosen to stay under it).
const CHUNK_TARGET_BYTES: u64 = 4 << 20;
/// `K` of chunk B-tree nodes: the HDF5 default every reader assumes when a
/// version 0 superblock stores none.
const CHUNK_BTREE_K: usize = 32;

/// How a frame is compressed.
#[derive(Clone, Debug)]
pub struct Nc4Options {
    /// zlib level 1-9; `None` stores chunks unfiltered.
    pub deflate: Option<u32>,
    /// Byte shuffle before deflate.
    pub shuffle: bool,
    /// Worker threads of the process-wide pool (fixed by the first writer;
    /// 0 = every core). Later writers share the pool.
    pub threads: usize,
    /// Raw bytes a writer may hold in flight before `write_*` waits.
    pub inflight_bytes: usize,
}

impl Default for Nc4Options {
    fn default() -> Self {
        Self {
            deflate: Some(4),
            shuffle: true,
            threads: 0,
            inflight_bytes: 1 << 30,
        }
    }
}

// ------------------------------------------------------------------ pool

type Job = Box<dyn FnOnce() + Send + 'static>;

struct Pool {
    queue: Mutex<VecDeque<Job>>,
    ready: Condvar,
    threads: usize,
}

/// The pool and the process that started it. A forked child inherits the
/// static but none of the worker threads, so a child that writes a frame
/// would queue chunks nobody runs; the pid check gives it its own pool.
static POOL: Mutex<Option<(u32, Arc<Pool>)>> = Mutex::new(None);

#[cfg(unix)]
unsafe extern "C" {
    fn nice(inc: std::os::raw::c_int) -> std::os::raw::c_int;
}

fn pool(threads: usize) -> Arc<Pool> {
    let mut slot = POOL.lock().unwrap_or_else(|e| e.into_inner());
    let pid = std::process::id();
    if let Some((owner, pool)) = slot.as_ref()
        && *owner == pid
    {
        return pool.clone();
    }
    let pool = {
        let cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        let threads = if threads == 0 { cores } else { threads.min(cores.max(1)) }.max(1);
        let pool = Arc::new(Pool {
            queue: Mutex::new(VecDeque::new()),
            ready: Condvar::new(),
            threads,
        });
        for index in 0..threads {
            let pool = pool.clone();
            std::thread::Builder::new()
                .name(format!("nc4-deflate-{index}"))
                .spawn(move || {
                    // Below the forecast's own threads: compression fills idle
                    // cores and yields to the step loop. On Linux nice() moves
                    // the calling thread only.
                    #[cfg(unix)]
                    unsafe {
                        nice(10);
                    }
                    loop {
                        let job = {
                            let mut queue = pool.queue.lock().unwrap_or_else(|e| e.into_inner());
                            loop {
                                if let Some(job) = queue.pop_front() {
                                    break job;
                                }
                                queue = pool.ready.wait(queue).unwrap_or_else(|e| e.into_inner());
                            }
                        };
                        job();
                    }
                })
                .expect("spawn nc4-deflate worker");
        }
        pool
    };
    *slot = Some((pid, pool.clone()));
    pool
}

impl Pool {
    fn submit(&self, job: Job) {
        self.queue
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push_back(job);
        self.ready.notify_one();
    }
}

/// Worker threads of the process-wide compression pool (0 before first use).
pub fn pool_threads() -> usize {
    POOL.lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .map(|(_, p)| p.threads)
        .unwrap_or(0)
}

// ------------------------------------------------------------------ writer

struct Stored {
    /// Chunk origin, dataset coordinates (record axis included).
    offsets: Vec<u64>,
    address: u64,
    size: u32,
}

/// A compressed chunk waiting for the chunks handed over before it.
struct Finished {
    varid: usize,
    offsets: Vec<u64>,
    raw_len: u64,
    bytes: Vec<u8>,
}

struct FileState {
    file: File,
    end: u64,
    stored: Vec<Vec<Stored>>,
    /// Sequence number of the next chunk to append.
    next_append: u64,
    /// Chunks done compressing out of turn (None: a chunk that failed; its
    /// error is already recorded and it only advances the sequence).
    waiting: BTreeMap<u64, Option<Finished>>,
    error: Option<String>,
    pending_jobs: usize,
    inflight: usize,
    /// Raw and stored bytes, for the receipt.
    raw_bytes: u64,
    stored_bytes: u64,
}

struct Shared {
    state: Mutex<FileState>,
    done: Condvar,
}

impl Shared {
    fn lock(&self) -> std::sync::MutexGuard<'_, FileState> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }
}

struct VarPlan {
    name: String,
    ty: NcType,
    elem: usize,
    dimids: Vec<usize>,
    record: bool,
    /// Non-record extents.
    inner: Vec<u64>,
    /// Non-record chunk extents.
    chunk: Vec<u64>,
    filtered: bool,
    written: Vec<bool>,
}

/// A NetCDF-4 frame being written. `write_record`/`write_var` copy the data
/// and return; compression runs on the pool; `finish` waits, indexes and
/// writes the metadata.
pub struct Nc4Writer {
    path: PathBuf,
    schema: Schema,
    plans: Vec<VarPlan>,
    options: Nc4Options,
    shared: Arc<Shared>,
    pool: Arc<Pool>,
    numrecs: u64,
    /// Sequence number of the next chunk handed to the pool.
    next_chunk: u64,
    finished: bool,
}

/// Bytes counted against a writer's in-flight budget until the last job
/// holding the slab is done with it.
struct Slab {
    bytes: Vec<u8>,
    shared: Arc<Shared>,
}

impl Drop for Slab {
    fn drop(&mut self) {
        let mut state = self.shared.lock();
        state.inflight = state.inflight.saturating_sub(self.bytes.len());
        drop(state);
        self.shared.done.notify_all();
    }
}

fn chunk_extents(inner: &[u64], elem: usize) -> Vec<u64> {
    let rank = inner.len();
    let mut chunk: Vec<u64> = inner.iter().map(|&d| d.max(1)).collect();
    if rank < 2 {
        return chunk;
    }
    let lead: u64 = chunk[..rank - 2].iter().product();
    let (ny, nx) = (chunk[rank - 2], chunk[rank - 1]);
    let mut tile = 1024u64;
    while tile > 32 && lead * tile.min(ny) * tile.min(nx) * elem as u64 > CHUNK_TARGET_BYTES {
        tile /= 2;
    }
    chunk[rank - 2] = tile.min(ny);
    chunk[rank - 1] = tile.min(nx);
    chunk
}

/// Gather the chunk at `origin` (non-record coordinates) out of a row-major
/// slab, padding the part past the edge with zeros.
fn gather(slab: &[u8], inner: &[u64], chunk: &[u64], origin: &[u64], elem: usize) -> Vec<u8> {
    let rank = inner.len();
    let total: usize = chunk.iter().product::<u64>() as usize * elem;
    if rank == 0 {
        return slab.to_vec();
    }
    if chunk == inner {
        return slab.to_vec();
    }
    let mut out = vec![0u8; total];
    let last = rank - 1;
    let mut strides = vec![1u64; rank];
    for axis in (0..last).rev() {
        strides[axis] = strides[axis + 1] * inner[axis + 1];
    }
    let run = (inner[last] - origin[last]).min(chunk[last]) as usize * elem;
    let row_bytes = chunk[last] as usize * elem;
    let rows: u64 = chunk[..last].iter().product();
    let mut row = vec![0u64; last];
    for r in 0..rows as usize {
        let inside = (0..last).all(|axis| origin[axis] + row[axis] < inner[axis]);
        if inside && run > 0 {
            let start: u64 = (0..last)
                .map(|axis| (origin[axis] + row[axis]) * strides[axis])
                .sum::<u64>()
                + origin[last];
            let start = start as usize * elem;
            out[r * row_bytes..r * row_bytes + run].copy_from_slice(&slab[start..start + run]);
        }
        for axis in (0..last).rev() {
            row[axis] += 1;
            if row[axis] < chunk[axis] {
                break;
            }
            row[axis] = 0;
        }
    }
    out
}

fn shuffle(bytes: &[u8], elem: usize) -> Vec<u8> {
    let n = bytes.len() / elem;
    let mut out = vec![0u8; bytes.len()];
    for b in 0..elem {
        let plane = &mut out[b * n..(b + 1) * n];
        for (i, v) in plane.iter_mut().enumerate() {
            *v = bytes[i * elem + b];
        }
    }
    out
}

fn deflate(bytes: &[u8], level: u32) -> std::io::Result<Vec<u8>> {
    let mut encoder = flate2::write::ZlibEncoder::new(
        Vec::with_capacity(bytes.len() / 2 + 64),
        flate2::Compression::new(level.min(9)),
    );
    encoder.write_all(bytes)?;
    encoder.finish()
}

fn write_at(file: &mut File, offset: u64, bytes: &[u8]) -> std::io::Result<()> {
    file.seek(SeekFrom::Start(offset))?;
    file.write_all(bytes)
}

/// Append every waiting chunk whose turn has come, in handing-over order.
fn append_in_order(state: &mut FileState) {
    while let Some(entry) = state.waiting.remove(&state.next_append) {
        state.next_append += 1;
        let Some(chunk) = entry else { continue };
        if state.error.is_some() {
            continue;
        }
        let address = state.end;
        match write_at(&mut state.file, address, &chunk.bytes) {
            Ok(()) => {
                state.end += chunk.bytes.len() as u64;
                state.raw_bytes += chunk.raw_len;
                state.stored_bytes += chunk.bytes.len() as u64;
                state.stored[chunk.varid].push(Stored {
                    offsets: chunk.offsets,
                    address,
                    size: chunk.bytes.len() as u32,
                });
            }
            Err(err) => {
                state.error.get_or_insert_with(|| err.to_string());
            }
        }
    }
}

impl Nc4Writer {
    /// Create the file and plan every variable's chunks.
    pub fn create(path: impl AsRef<Path>, schema: Schema, options: Nc4Options) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let mut plans = Vec::with_capacity(schema.vars.len());
        for var in &schema.vars {
            let record = matches!((var.dimids.first(), schema.record_dim), (Some(&d), Some(r)) if d == r);
            let inner: Vec<u64> = var.dimids[usize::from(record)..]
                .iter()
                .map(|&d| schema.dims[d].len as u64)
                .collect();
            let elem = var.ty.size();
            plans.push(VarPlan {
                name: var.name.clone(),
                ty: var.ty,
                elem,
                dimids: var.dimids.clone(),
                record,
                chunk: chunk_extents(&inner, elem),
                inner,
                filtered: var.ty != NcType::Char && !var.dimids.is_empty(),
                written: Vec::new(),
            });
        }
        let mut file = File::create(&path)?;
        // The superblock is written at finish; until then the file has no root.
        file.write_all(&[0u8; 96])?;
        let shared = Arc::new(Shared {
            state: Mutex::new(FileState {
                file,
                end: 96,
                stored: (0..plans.len()).map(|_| Vec::new()).collect(),
                next_append: 0,
                waiting: BTreeMap::new(),
                error: None,
                pending_jobs: 0,
                inflight: 0,
                raw_bytes: 0,
                stored_bytes: 0,
            }),
            done: Condvar::new(),
        });
        let pool = pool(options.threads);
        Ok(Self {
            path,
            schema,
            plans,
            options,
            shared,
            pool,
            numrecs: 0,
            next_chunk: 0,
            finished: false,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn num_records(&self) -> u64 {
        self.numrecs
    }

    /// Write a whole fixed variable (`raw`: native-endian elements).
    pub fn write_var(&mut self, varid: usize, nc_type: NcType, raw: &[u8]) -> Result<()> {
        let plan = self.plan(varid, nc_type)?;
        if plan.record {
            return Err(NcWriteError::Usage(format!(
                "variable '{}' is a record variable; write it with write_record",
                plan.name
            )));
        }
        self.submit(varid, None, raw)
    }

    /// Write record `recno` of a record variable.
    pub fn write_record(&mut self, recno: u64, varid: usize, nc_type: NcType, raw: &[u8]) -> Result<()> {
        let plan = self.plan(varid, nc_type)?;
        if !plan.record {
            return Err(NcWriteError::Usage(format!(
                "variable '{}' is not a record variable; write it with write_var",
                plan.name
            )));
        }
        self.submit(varid, Some(recno), raw)
    }

    fn plan(&self, varid: usize, nc_type: NcType) -> Result<&VarPlan> {
        let plan = self.plans.get(varid).ok_or_else(|| {
            NcWriteError::Usage(format!("varid {varid} is not defined"))
        })?;
        if plan.ty != nc_type {
            return Err(NcWriteError::Usage(format!(
                "variable '{}' is {} but the payload is {}",
                plan.name,
                plan.ty.name(),
                nc_type.name()
            )));
        }
        Ok(plan)
    }

    fn submit(&mut self, varid: usize, recno: Option<u64>, raw: &[u8]) -> Result<()> {
        if self.finished {
            return Err(NcWriteError::Usage("writer already finished".into()));
        }
        let (elem, inner, chunk, filtered, name) = {
            let plan = &self.plans[varid];
            (plan.elem, plan.inner.clone(), plan.chunk.clone(), plan.filtered, plan.name.clone())
        };
        let elems: u64 = inner.iter().product();
        if raw.len() as u64 != elems * elem as u64 {
            return Err(NcWriteError::Usage(format!(
                "variable '{name}' needs {} bytes per slab, got {}",
                elems * elem as u64,
                raw.len()
            )));
        }
        let index = recno.unwrap_or(0) as usize;
        {
            let plan = &mut self.plans[varid];
            if plan.written.len() <= index {
                plan.written.resize(index + 1, false);
            }
            if plan.written[index] {
                return Err(NcWriteError::Usage(format!(
                    "variable '{name}' slab {index} was already written"
                )));
            }
            plan.written[index] = true;
        }
        if let Some(r) = recno {
            self.numrecs = self.numrecs.max(r + 1);
        }
        if elems == 0 {
            return Ok(());
        }
        // Wait for room, then take one copy of the caller's bytes.
        {
            let mut state = self.shared.lock();
            while state.inflight > 0 && state.inflight + raw.len() > self.options.inflight_bytes {
                if state.error.is_some() {
                    break;
                }
                state = self.shared.done.wait(state).unwrap_or_else(|e| e.into_inner());
            }
            if let Some(error) = &state.error {
                return Err(NcWriteError::Usage(format!("compression failed: {error}")));
            }
            state.inflight += raw.len();
        }
        let slab = Arc::new(Slab {
            bytes: raw.to_vec(),
            shared: self.shared.clone(),
        });
        let rank = inner.len();
        let counts: Vec<u64> = inner
            .iter()
            .zip(&chunk)
            .map(|(d, c)| d.max(&1).div_ceil(*c))
            .collect();
        let total: u64 = counts.iter().product();
        let mut grid = vec![0u64; rank];
        for _ in 0..total {
            let origin: Vec<u64> = grid.iter().zip(&chunk).map(|(g, c)| g * c).collect();
            let seq = self.next_chunk;
            self.next_chunk += 1;
            let shared = self.shared.clone();
            let slab = slab.clone();
            let inner = inner.clone();
            let chunk = chunk.clone();
            let level = self.options.deflate;
            let shuffle_on = self.options.shuffle;
            {
                self.shared.lock().pending_jobs += 1;
            }
            self.pool.submit(Box::new(move || {
                let raw_chunk = gather(&slab.bytes, &inner, &chunk, &origin, elem);
                drop(slab);
                let raw_len = raw_chunk.len() as u64;
                let result = if filtered {
                    let shuffled = if shuffle_on && elem > 1 {
                        shuffle(&raw_chunk, elem)
                    } else {
                        raw_chunk
                    };
                    match level {
                        Some(level) => deflate(&shuffled, level),
                        None => Ok(shuffled),
                    }
                } else {
                    Ok(raw_chunk)
                };
                let mut state = shared.lock();
                let finished = match result {
                    Ok(bytes) if u32::try_from(bytes.len()).is_ok() => {
                        let mut offsets = Vec::with_capacity(origin.len() + 1);
                        if let Some(r) = recno {
                            offsets.push(r);
                        }
                        offsets.extend_from_slice(&origin);
                        Some(Finished {
                            varid,
                            offsets,
                            raw_len,
                            bytes,
                        })
                    }
                    Ok(_) => {
                        state
                            .error
                            .get_or_insert_with(|| "a stored chunk exceeds 4 GiB".to_string());
                        None
                    }
                    Err(err) => {
                        state.error.get_or_insert_with(|| err.to_string());
                        None
                    }
                };
                state.waiting.insert(seq, finished);
                append_in_order(&mut state);
                state.pending_jobs -= 1;
                drop(state);
                shared.done.notify_all();
            }));
            for axis in (0..rank).rev() {
                grid[axis] += 1;
                if grid[axis] < counts[axis] {
                    break;
                }
                grid[axis] = 0;
            }
        }
        Ok(())
    }

    fn wait_idle(&self) -> Result<()> {
        let mut state = self.shared.lock();
        while state.pending_jobs > 0 {
            state = self.shared.done.wait(state).unwrap_or_else(|e| e.into_inner());
        }
        if let Some(error) = &state.error {
            return Err(NcWriteError::Usage(format!("compression failed: {error}")));
        }
        Ok(())
    }

    /// Raw and stored bytes of every chunk written so far.
    pub fn bytes(&self) -> (u64, u64) {
        let state = self.shared.lock();
        (state.raw_bytes, state.stored_bytes)
    }

    /// Abandon: wait for queued jobs (they hold the file), leave no root.
    pub fn abort(mut self) {
        self.finished = true;
        let _ = self.wait_idle();
    }

    /// Wait for every chunk, refuse a hole, write the index, the metadata
    /// and finally the superblock; flush and fsync.
    pub fn finish(mut self) -> Result<()> {
        self.finished = true;
        self.wait_idle()?;
        for plan in &self.plans {
            let need = if plan.record { self.numrecs as usize } else { 1 };
            let elems: u64 = plan.inner.iter().product();
            if plan.record && elems == 0 {
                continue;
            }
            if !plan.record && elems == 0 {
                continue;
            }
            for index in 0..need {
                if !plan.written.get(index).copied().unwrap_or(false) {
                    return Err(NcWriteError::Usage(format!(
                        "variable '{}' {} was never written; a frame with a hole is refused",
                        plan.name,
                        if plan.record {
                            format!("record {index}")
                        } else {
                            "(fixed)".to_string()
                        }
                    )));
                }
            }
        }
        let numrecs = self.numrecs;
        let mut state = self.shared.lock();
        let stored = std::mem::take(&mut state.stored);
        let mut end = state.end;

        // B-trees, one per variable with data, appended after the chunks.
        let mut btree_roots = vec![UNDEF; self.plans.len()];
        let mut index_bytes: Vec<u8> = Vec::new();
        let index_start = end;
        let mut contiguous = vec![None; self.plans.len()];
        for (varid, mut chunks) in stored.into_iter().enumerate() {
            if chunks.is_empty() {
                continue;
            }
            let plan = &self.plans[varid];
            if plan.inner.is_empty() && !plan.record {
                // A scalar: HDF5 has no chunked scalar, so its one word is contiguous.
                contiguous[varid] = Some((chunks[0].address, chunks[0].size as u64));
                continue;
            }
            chunks.sort_by(|a, b| a.offsets.cmp(&b.offsets));
            let mut chunk_dims: Vec<u64> = Vec::with_capacity(plan.chunk.len() + 1);
            if plan.record {
                chunk_dims.push(1);
            }
            chunk_dims.extend_from_slice(&plan.chunk);
            let (root, bytes) = encode_btree(
                &chunks,
                &chunk_dims,
                plan.elem as u64,
                index_start + index_bytes.len() as u64,
            );
            btree_roots[varid] = root;
            index_bytes.extend(bytes);
        }
        end += index_bytes.len() as u64;
        end = (end + 7) & !7;
        let meta_start = end;
        let (meta, root_address, eof) = self.metadata(meta_start, numrecs, &btree_roots, &contiguous)?;
        let mut file = state.file.try_clone()?;
        drop(state);
        write_at(&mut file, index_start, &index_bytes)?;
        let pad = (meta_start - index_start - index_bytes.len() as u64) as usize;
        if pad > 0 {
            file.write_all(&vec![0u8; pad])?;
        }
        write_at(&mut file, meta_start, &meta)?;
        file.set_len(eof)?;
        // Superblock v0 last: base 0, no free space, EOF, no driver, root.
        let mut sb = b"\x89HDF\r\n\x1a\n".to_vec();
        sb.extend_from_slice(&[0, 0, 0, 0, 0, 8, 8, 0]);
        sb.extend_from_slice(&4u16.to_le_bytes());
        sb.extend_from_slice(&16u16.to_le_bytes());
        sb.extend_from_slice(&0u32.to_le_bytes());
        for n in [0, UNDEF, eof, UNDEF, 0, root_address] {
            sb.extend_from_slice(&n.to_le_bytes());
        }
        sb.extend_from_slice(&[0; 24]);
        debug_assert_eq!(sb.len(), 96);
        write_at(&mut file, 0, &sb)?;
        file.flush()?;
        file.sync_all()?;
        Ok(())
    }

    /// Object headers, the root group and the global heap, laid out from
    /// `start`. Returns the bytes, the root's address and the file's end.
    fn metadata(
        &self,
        start: u64,
        numrecs: u64,
        btree_roots: &[u64],
        contiguous: &[Option<(u64, u64)>],
    ) -> Result<(Vec<u8>, u64, u64)> {
        let schema = &self.schema;
        let dims = &schema.dims;
        let dim_len = |d: usize| -> u64 {
            if Some(d) == schema.record_dim {
                numrecs
            } else {
                dims[d].len as u64
            }
        };
        let global_owned: Vec<OwnedAttr> = schema.gattrs.iter().map(owned_attr).collect();
        let var_owned: Vec<Vec<OwnedAttr>> = schema
            .vars
            .iter()
            .map(|v| v.attrs.iter().map(owned_attr).collect())
            .collect();
        let globals = borrow_attrs(&global_owned);

        // Objects: one per variable, plus one hidden scale per dimension that
        // no coordinate variable represents.
        struct Obj<'a> {
            name: &'a str,
            dimids: Vec<usize>,
            attrs: Vec<Attribute<'a>>,
            ty: NcType,
            hidden: bool,
            varid: Option<usize>,
        }
        let mut objects: Vec<Obj<'_>> = Vec::new();
        for (j, v) in schema.vars.iter().enumerate() {
            objects.push(Obj {
                name: v.name.as_str(),
                dimids: v.dimids.clone(),
                attrs: borrow_attrs(&var_owned[j]),
                ty: v.ty,
                hidden: false,
                varid: Some(j),
            });
        }
        let mut scales = Vec::with_capacity(dims.len());
        for (i, d) in dims.iter().enumerate() {
            if let Some(j) = schema
                .vars
                .iter()
                .position(|v| v.name == d.name && v.dimids == [i])
            {
                scales.push(j);
            } else {
                if schema.vars.iter().any(|v| v.name == d.name) {
                    return Err(NcWriteError::Schema(format!(
                        "variable '{}' shares its name with a dimension it does not coordinate",
                        d.name
                    )));
                }
                scales.push(objects.len());
                objects.push(Obj {
                    name: d.name.as_str(),
                    dimids: vec![i],
                    attrs: Vec::new(),
                    ty: NcType::Float,
                    hidden: true,
                    varid: None,
                });
            }
        }

        let mut heap = Heap::default();
        let mut dimrefs = Vec::with_capacity(dims.len());
        for _ in dims {
            let id = heap.objects.len();
            heap.objects.push(vec![0; 8]);
            dimrefs.push(id);
        }

        let header_for = |k: usize, refs: &[(u64, u32)]| -> Result<Vec<u8>> {
            let o = &objects[k];
            let shape: Vec<usize> = o.dimids.iter().map(|&d| dim_len(d) as usize).collect();
            let unlimited: Vec<bool> = o.dimids.iter().map(|&d| Some(d) == schema.record_dim).collect();
            let mut sp = space(&shape)?;
            if unlimited.iter().any(|&u| u) {
                sp[2] = 1; // max dimensions follow
                for (len, unl) in shape.iter().zip(&unlimited) {
                    u64v(&mut sp, if *unl { UNDEF } else { *len as u64 });
                }
            }
            let probe = Variable {
                name: o.name,
                dimensions: o.dimids.clone(),
                attributes: o.attrs.clone(),
                values: Values::Primitive(o.ty, &[]),
            };
            let mut fill = fill_message(&probe, o.hidden)?;
            let chunked = (!o.hidden && !o.dimids.is_empty()) || unlimited.iter().any(|&u| u);
            let mut m: Vec<(u8, Vec<u8>)> = vec![(1, sp), (3, datatype(o.ty))];
            if o.dimids.is_empty() {
                let (address, size) = o
                    .varid
                    .and_then(|j| contiguous[j])
                    .unwrap_or((UNDEF, o.ty.size() as u64));
                let mut layout = vec![3, 1];
                u64v(&mut layout, address);
                u64v(&mut layout, size);
                m.push((5, fill));
                m.push((8, layout));
            } else if chunked {
                fill[1] = 3; // incremental allocation, as HDF5 does for chunked data
                let (chunk_dims, elem, root): (Vec<u64>, u32, u64) = match o.varid {
                    Some(j) => {
                        let plan = &self.plans[j];
                        let mut c = Vec::with_capacity(plan.chunk.len() + 1);
                        if plan.record {
                            c.push(1);
                        }
                        c.extend_from_slice(&plan.chunk);
                        (c, plan.elem as u32, btree_roots[j])
                    }
                    None => (vec![1], 4, UNDEF),
                };
                let mut layout = vec![3, 2, (chunk_dims.len() + 1) as u8];
                u64v(&mut layout, root);
                for c in &chunk_dims {
                    layout.extend_from_slice(&(*c as u32).to_le_bytes());
                }
                layout.extend_from_slice(&elem.to_le_bytes());
                m.push((5, fill));
                m.push((8, layout));
                let filtered = o.varid.is_some_and(|j| self.plans[j].filtered);
                if filtered {
                    let elem = o.ty.size() as u32;
                    let shuffle_on = self.options.shuffle && elem > 1;
                    let mut filters: Vec<(u16, u32)> = Vec::new();
                    if shuffle_on {
                        filters.push((2, elem));
                    }
                    if let Some(level) = self.options.deflate {
                        filters.push((1, level.min(9)));
                    }
                    if !filters.is_empty() {
                        let mut f = vec![2, filters.len() as u8];
                        for (id, value) in filters {
                            f.extend_from_slice(&id.to_le_bytes());
                            f.extend_from_slice(&1u16.to_le_bytes()); // optional
                            f.extend_from_slice(&1u16.to_le_bytes()); // one client value
                            f.extend_from_slice(&value.to_le_bytes());
                        }
                        m.push((11, f));
                    }
                }
            } else {
                // A fixed hidden scale: contiguous, never allocated.
                let mut layout = vec![3, 1];
                u64v(&mut layout, UNDEF);
                u64v(&mut layout, shape.iter().product::<usize>() as u64 * 4);
                m.push((5, fill));
                m.push((8, layout));
            }
            // A scalar variable carries no `_Netcdf4Coordinates`, as netCDF-C writes it: netCDF-C 4.9
            // refuses to open a file whose scalar variable has an empty one ("NetCDF: HDF error"; MPAS
            // history's isice_lu and iswater_lu through rw_mpas_stream, gpuwm 2.9.10 pass 4).
            if !o.hidden && !o.dimids.is_empty() {
                m.push((
                    12,
                    int_attr(
                        "_Netcdf4Coordinates",
                        &o.dimids.iter().map(|&d| d as i32).collect::<Vec<_>>(),
                    )?,
                ));
            }
            m.extend(attr_messages(&o.attrs)?);
            if let Some(d) = scales.iter().position(|&s| s == k) {
                m.push((12, cstring_attr("CLASS", "DIMENSION_SCALE")?));
                let label = if o.hidden {
                    format!(
                        "This is a netCDF dimension but not a netCDF variable.{:>10}",
                        dim_len(d)
                    )
                } else {
                    o.name.to_string()
                };
                m.push((12, cstring_attr("NAME", &label)?));
                m.push((12, scalar_int_attr("_Netcdf4Dimid", d as i32)?));
            } else if !o.dimids.is_empty() {
                let mut b = vec![];
                for &d in &o.dimids {
                    b.extend(descriptor(1, dimrefs[d], refs)?);
                }
                m.push((
                    12,
                    crate::netcdf4::attr("DIMENSION_LIST", vlen_type(false), &[o.dimids.len()], &b)?,
                ));
            }
            object(m)
        };
        let root_for = |addresses: &[u64]| -> Result<Vec<u8>> {
            // Link info: creation order tracked (definition order), compact links.
            let mut li = vec![0, 1];
            u64v(&mut li, objects.len() as u64);
            u64v(&mut li, UNDEF);
            u64v(&mut li, UNDEF);
            let mut m = vec![(2, li), (10, vec![0, 0])];
            m.extend(attr_messages(&globals)?);
            m.push((12, scalar_int_attr("_nc3_strict", 1)?));
            for (j, o) in objects.iter().enumerate() {
                let mut b = vec![1, 5];
                u64v(&mut b, j as u64);
                u16v(&mut b, o.name.len())?;
                b.extend_from_slice(o.name.as_bytes());
                u64v(&mut b, addresses[j]);
                m.push((6, b));
            }
            object(m)
        };
        // Sizes do not depend on addresses: lay out with placeholders, then
        // again with the real ones.
        let mut addresses = vec![0u64; objects.len()];
        let mut refs = vec![(0u64, 1u32); heap.objects.len()];
        let root_len = root_for(&addresses)?.len() as u64;
        let mut offset = start + root_len;
        for (k, address) in addresses.iter_mut().enumerate() {
            *address = offset;
            offset += header_for(k, &refs)?.len() as u64;
        }
        offset = (offset + 7) & !7;
        for (i, &id) in dimrefs.iter().enumerate() {
            heap.objects[id] = addresses[scales[i]].to_le_bytes().to_vec();
        }
        let (resolved, blocks, end) = heap.layout(offset);
        refs = resolved;
        let mut out = Vec::with_capacity((end - start) as usize);
        out.extend(root_for(&addresses)?);
        for k in 0..objects.len() {
            out.extend(header_for(k, &refs)?);
        }
        out.resize((offset - start) as usize, 0);
        for (_, b) in blocks {
            out.extend(b);
        }
        debug_assert_eq!(out.len() as u64, end - start);
        Ok((out, start, end))
    }
}

/// Attribute messages, as `user_attrs` writes them except an EMPTY text
/// attribute: netCDF-C stores that as a null dataspace (and reads it back as
/// the empty string, as a classic file's zero-length NC_CHAR reads), where a
/// zero-length simple dataspace reads back as an empty array. WRF writes
/// many (`stagger = ""`), and a reader comparing `stagger == ""` must see
/// the same value from both containers.
fn attr_messages(attrs: &[Attribute<'_>]) -> Result<Vec<(u8, Vec<u8>)>> {
    let mut out = Vec::with_capacity(attrs.len());
    for a in attrs {
        if a.dtype == NcType::Char && a.bytes.is_empty() {
            let dt = datatype(NcType::Char);
            let ds: [u8; 4] = [2, 0, 0, 2];
            let mut b = vec![1, 0];
            u16v(&mut b, a.name.len() + 1)?;
            u16v(&mut b, dt.len())?;
            u16v(&mut b, ds.len())?;
            b.extend_from_slice(a.name.as_bytes());
            b.push(0);
            b.resize((b.len() + 7) & !7, 0);
            b.extend_from_slice(&dt);
            b.resize((b.len() + 7) & !7, 0);
            b.extend_from_slice(&ds);
            b.resize((b.len() + 7) & !7, 0);
            out.push((12, b));
        } else {
            out.extend(user_attrs(std::slice::from_ref(a))?);
        }
    }
    Ok(out)
}

type OwnedAttr = (String, NcType, Vec<u8>);

fn owned_attr(a: &crate::schema::Attr) -> OwnedAttr {
    let bytes: Vec<u8> = match &a.value {
        AttrValue::Text(t) => t.as_bytes().to_vec(),
        AttrValue::Bytes(v) => v.iter().map(|&x| x as u8).collect(),
        AttrValue::UBytes(v) => v.clone(),
        AttrValue::Shorts(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        AttrValue::UShorts(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        AttrValue::Ints(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        AttrValue::UInts(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        AttrValue::Floats(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        AttrValue::Doubles(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        AttrValue::Int64s(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        AttrValue::UInt64s(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
    };
    (a.name.clone(), a.value.nc_type(), bytes)
}

fn borrow_attrs(owned: &[OwnedAttr]) -> Vec<Attribute<'_>> {
    owned
        .iter()
        .map(|(n, t, b)| Attribute {
            name: n.as_str(),
            dtype: *t,
            bytes: b.as_slice(),
        })
        .collect()
}

/// Version 1 B-tree (node type 1) over `chunks` (sorted by offsets), laid out
/// from `base`. Returns the root's address and the node bytes. Keys follow
/// the HDF5 library: a child's key is its first chunk's offsets, the last key
/// one chunk past the last chunk.
fn encode_btree(chunks: &[Stored], chunk: &[u64], elem: u64, base: u64) -> (u64, Vec<u8>) {
    let rank = chunk.len();
    let key_size = 8 + 8 * (rank + 1);
    let node_size = 24 + 2 * CHUNK_BTREE_K * 8 + (2 * CHUNK_BTREE_K + 1) * key_size;
    let fan = 2 * CHUNK_BTREE_K;
    // Each node: level, children (chunk indices at level 0, node indices
    // above), first and last chunk index under it.
    struct Node {
        level: u8,
        children: Vec<usize>,
        first: usize,
        last: usize,
    }
    let mut nodes: Vec<Node> = Vec::new();
    let mut level_nodes: Vec<usize> = Vec::new();
    for start in (0..chunks.len()).step_by(fan) {
        let end = (start + fan).min(chunks.len());
        level_nodes.push(nodes.len());
        nodes.push(Node {
            level: 0,
            children: (start..end).collect(),
            first: start,
            last: end - 1,
        });
    }
    let mut level = 0u8;
    while level_nodes.len() > 1 {
        level += 1;
        let mut parents = Vec::new();
        for group in level_nodes.chunks(fan) {
            parents.push(nodes.len());
            let first = nodes[group[0]].first;
            let last = nodes[*group.last().unwrap()].last;
            nodes.push(Node {
                level,
                children: group.to_vec(),
                first,
                last,
            });
        }
        level_nodes = parents;
    }
    let address = |i: usize| base + (i * node_size) as u64;
    let push_key = |out: &mut Vec<u8>, size: u32, offsets: &[u64], tail: u64| {
        out.extend_from_slice(&size.to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes());
        for o in offsets {
            out.extend_from_slice(&o.to_le_bytes());
        }
        out.extend_from_slice(&tail.to_le_bytes());
    };
    let mut out = Vec::with_capacity(nodes.len() * node_size);
    for (i, node) in nodes.iter().enumerate() {
        let start = out.len();
        out.extend_from_slice(b"TREE");
        out.push(1);
        out.push(node.level);
        out.extend_from_slice(&(node.children.len() as u16).to_le_bytes());
        let left = i
            .checked_sub(1)
            .filter(|&l| nodes[l].level == node.level)
            .map_or(UNDEF, address);
        let right = Some(i + 1)
            .filter(|&r| nodes.get(r).is_some_and(|n| n.level == node.level))
            .map_or(UNDEF, address);
        out.extend_from_slice(&left.to_le_bytes());
        out.extend_from_slice(&right.to_le_bytes());
        for &child in &node.children {
            let (first, child_address) = if node.level == 0 {
                (child, chunks[child].address)
            } else {
                (nodes[child].first, address(child))
            };
            push_key(&mut out, chunks[first].size, &chunks[first].offsets, 0);
            out.extend_from_slice(&child_address.to_le_bytes());
        }
        let past: Vec<u64> = chunks[node.last]
            .offsets
            .iter()
            .zip(chunk)
            .map(|(o, c)| o + c)
            .collect();
        push_key(&mut out, 0, &past, elem);
        out.resize(start + node_size, 0);
    }
    let root = address(nodes.len() - 1);
    (root, out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::NcFormat;

    #[test]
    fn tiles_stay_near_four_mib() {
        assert_eq!(chunk_extents(&[50, 714, 1164], 4), vec![50, 128, 128]);
        assert_eq!(chunk_extents(&[714, 1164], 4), vec![714, 1024]);
        assert_eq!(chunk_extents(&[19], 1), vec![19]);
        assert_eq!(chunk_extents(&[], 4), Vec::<u64>::new());
    }

    #[test]
    fn gather_pads_edges() {
        // 3 x 5 slab of u8, chunk 2 x 4 at origin (2, 4).
        let slab: Vec<u8> = (0..15).collect();
        let c = gather(&slab, &[3, 5], &[2, 4], &[2, 4], 1);
        assert_eq!(c, vec![14, 0, 0, 0, 0, 0, 0, 0]);
        let c = gather(&slab, &[3, 5], &[2, 4], &[0, 0], 1);
        assert_eq!(c, vec![0, 1, 2, 3, 5, 6, 7, 8]);
    }

    #[test]
    fn a_frame_round_trips_its_size() {
        let mut schema = Schema::new(NcFormat::Cdf5);
        let time = schema.def_dim("Time", 0, true).unwrap();
        let strlen = schema.def_dim("DateStrLen", 19, false).unwrap();
        let sn = schema.def_dim("south_north", 300, false).unwrap();
        let we = schema.def_dim("west_east", 400, false).unwrap();
        schema
            .put_global_attr("TITLE", AttrValue::Text(" OUTPUT FROM GPUWM".into()))
            .unwrap();
        let times = schema.def_var("Times", NcType::Char, &[time, strlen]).unwrap();
        let t2 = schema.def_var("T2", NcType::Float, &[time, sn, we]).unwrap();
        schema.put_var_attr(t2, "units", AttrValue::Text("K".into())).unwrap();
        let mut path = std::env::temp_dir();
        path.push(format!("gpuwm-nc4-stream-{}.nc", std::process::id()));
        let mut writer = Nc4Writer::create(&path, schema, Nc4Options::default()).unwrap();
        writer
            .write_record(0, times, NcType::Char, b"2026-08-16_00:00:00")
            .unwrap();
        let field: Vec<u8> = (0..120_000)
            .flat_map(|i| (280.0f32 + (i % 400) as f32 * 0.01).to_ne_bytes())
            .collect();
        writer.write_record(0, t2, NcType::Float, &field).unwrap();
        writer.finish().unwrap();
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(&bytes[..8], b"\x89HDF\r\n\x1a\n");
        assert!(bytes.len() < field.len());
        std::fs::remove_file(&path).ok();
    }

    /// A scalar variable (MPAS history's isice_lu) gets no `_Netcdf4Coordinates`: netCDF-C 4.9 refuses
    /// a file whose scalar variable carries an empty one ("NetCDF: HDF error"). Where netCDF-C's ncdump
    /// is installed, the frame is also opened with it.
    #[test]
    fn a_scalar_variable_has_no_coordinates_attribute() {
        let mut schema = Schema::new(NcFormat::Cdf5);
        let time = schema.def_dim("Time", 0, true).unwrap();
        let cells = schema.def_dim("nCells", 64, false).unwrap();
        let t2 = schema.def_var("t2m", NcType::Float, &[time, cells]).unwrap();
        let ice = schema.def_var("isice_lu", NcType::Int, &[]).unwrap();
        let mut path = std::env::temp_dir();
        path.push(format!("gpuwm-nc4-scalar-{}.nc", std::process::id()));
        let mut writer = Nc4Writer::create(&path, schema, Nc4Options::default()).unwrap();
        let field: Vec<u8> = (0..64).flat_map(|i| (280.0f32 + i as f32).to_ne_bytes()).collect();
        writer.write_record(0, t2, NcType::Float, &field).unwrap();
        writer.write_var(ice, NcType::Int, &15i32.to_ne_bytes()).unwrap();
        writer.finish().unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let name = b"_Netcdf4Coordinates";
        let count = bytes.windows(name.len()).filter(|w| *w == name).count();
        assert_eq!(count, 1, "only t2m carries _Netcdf4Coordinates");
        if let Ok(out) = std::process::Command::new("ncdump").arg("-h").arg(&path).output() {
            assert!(out.status.success(), "ncdump -h refused the frame: {}", String::from_utf8_lossy(&out.stderr));
        }
        std::fs::remove_file(&path).ok();
    }

    /// The same frame written again is the same bytes, however the pool's
    /// threads happen to finish (chunks are appended in handing-over order).
    #[test]
    fn the_same_frame_is_the_same_bytes() {
        let write = |tag: usize| -> Vec<u8> {
            let mut schema = Schema::new(NcFormat::Cdf5);
            let time = schema.def_dim("Time", 0, true).unwrap();
            let lev = schema.def_dim("bottom_top", 50, false).unwrap();
            let sn = schema.def_dim("south_north", 300, false).unwrap();
            let we = schema.def_dim("west_east", 400, false).unwrap();
            let t = schema.def_var("T", NcType::Float, &[time, lev, sn, we]).unwrap();
            let q = schema.def_var("QVAPOR", NcType::Float, &[time, lev, sn, we]).unwrap();
            let t2 = schema.def_var("T2", NcType::Float, &[time, sn, we]).unwrap();
            let mut path = std::env::temp_dir();
            path.push(format!("gpuwm-nc4-same-{}-{tag}.nc", std::process::id()));
            let mut writer = Nc4Writer::create(&path, schema, Nc4Options::default()).unwrap();
            for recno in 0..2u64 {
                for (varid, n, scale) in [(t, 50 * 300 * 400, 0.37f32), (q, 50 * 300 * 400, 1.0e-5), (t2, 300 * 400, 0.01)] {
                    // Chunks of very different compressibility finish out of order.
                    let field: Vec<u8> = (0..n)
                        .flat_map(|i: usize| {
                            let v = if (i / 64_000) % 3 == 0 {
                                ((i as u64).wrapping_mul(2_654_435_761) % 1_000_003) as f32 * scale
                            } else {
                                280.0 + (i % 400) as f32 * scale + recno as f32
                            };
                            v.to_ne_bytes()
                        })
                        .collect();
                    writer.write_record(recno, varid, NcType::Float, &field).unwrap();
                }
            }
            writer.finish().unwrap();
            let bytes = std::fs::read(&path).unwrap();
            std::fs::remove_file(&path).ok();
            bytes
        };
        let first = write(0);
        for tag in 1..4 {
            assert!(write(tag) == first, "write {tag} differs from write 0");
        }
    }
}
