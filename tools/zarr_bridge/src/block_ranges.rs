//! Read a window of blosc-compressed Zarr chunks over HTTP byte ranges.
//!
//! THE WASTE THIS ENDS.  Google's ARCO ERA5 store keeps one hour of one
//! pressure-level field as ONE chunk, (1, 37, 721, 1440) float32 in blosc-lz4,
//! about 70-85 MB.  zarrs reads every chunk a subset touches whole before it
//! decodes the part the subset asks for, so a 12 by 15 degree area moved
//! 835.5 MB for a 14.2 MB file (two times, 28 fields; bytes audit,
//! 2026-10-29), about 551 MB per time whatever the area.
//!
//! A blosc chunk (format 2, what numcodecs and c-blosc 1.x write) carries its
//! own index: a 16-byte header and the offset of every compressed block.  Each
//! block decodes on its own to `blocksize` bytes of the C-order chunk (512 KiB
//! in ARCO ERA5, about an eighth of one level).  So this reader takes the header
//! and the offset table with one small range GET, works out which blocks hold
//! the rows the window needs, GETs only those blocks, and decodes them with
//! `blosc_decompress_bytes_partial`, the same `blosc_getitem` call zarrs' own
//! partial decoder makes on the whole chunk.  The values are the bytes the
//! whole-chunk path decodes, bit for bit; only fewer bytes cross the network.
//!
//! An array this cannot serve (another codec, filters, F order, a big-endian
//! or unsupported dtype, an irregular chunk grid) or a chunk with no block table
//! is read whole by zarrs, as before, and the caller says so on stderr.
use std::{
    collections::BTreeSet,
    ops::Range,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Mutex,
    },
};

use serde_json::Value;
use zarrs::{
    array::{codec::bytes_to_bytes::blosc::blosc_decompress_bytes_partial, Array},
    storage::ReadableStorageTraits,
};

use crate::http_store::HttpStore;

type SourceArray = Array<dyn ReadableStorageTraits>;

/// The blosc header: version, codec version, flags, typesize, then nbytes,
/// blocksize and cbytes as little-endian int32.
const HEADER: usize = 16;
/// The first GET of a chunk takes the header and the offset table of a chunk
/// cut into blocks of PROBE_BLOCK bytes (c-blosc picks 16 KiB or more unless
/// told otherwise), at most PROBE bytes; a longer table takes a second GET.
const PROBE_BLOCK: u64 = 16 * 1024;
const PROBE: u64 = 64 * 1024;
/// c-blosc 1.x writes format version 2; a blosc2 frame is another layout.
const BLOSC1_FORMAT: u8 = 2;
/// Header flag: the chunk is stored uncompressed after the header.
const MEMCPYED: u8 = 0x02;
/// Two wanted extents closer than this are taken in one GET: a request costs a
/// round trip, and 64 KiB is about 13 ms even at 5 MB/s.
const MERGE_GAP: u64 = 64 * 1024;
/// When the wanted blocks are this share of the chunk or more, the chunk is
/// taken in one GET (the window needs nearly all of it anyway).
const WHOLE_SHARE: f64 = 0.8;
/// Range GETs one chunk window keeps in flight.
const RANGE_WORKERS: usize = 6;

/// What the array's metadata says about reading its chunks block by block.
pub struct Layout {
    /// Bytes per element (little-endian on disk).
    element: u64,
    /// The regular chunk shape.
    chunk: Vec<u64>,
}

/// How a window was served.
pub enum Read {
    /// The window's elements, little-endian, in C order.
    Window(Vec<u8>),
    /// A chunk it touches has no block table to read it by (the reason), or
    /// does not exist (zarrs supplies the fill value); read it with zarrs.
    Whole(String),
}

fn text<'a>(value: &'a Value, key: &str) -> Option<&'a str> {
    value.get(key).and_then(Value::as_str)
}

fn shape(value: Option<&Value>) -> Option<Vec<u64>> {
    value?.as_array()?.iter().map(Value::as_u64).collect()
}

/// Whether `array` can be read by blocks, or the reason it cannot.
pub fn layout(array: &SourceArray) -> Result<Layout, String> {
    let metadata = serde_json::to_value(array.metadata()).map_err(|e| e.to_string())?;
    let (dtype, chunk) = match metadata.get("zarr_format").and_then(Value::as_u64) {
        Some(2) => {
            let compressor = metadata.get("compressor").filter(|c| !c.is_null());
            if compressor.and_then(|c| text(c, "id")) != Some("blosc") {
                return Err(format!(
                    "its compressor is {}, not blosc",
                    compressor.and_then(|c| text(c, "id")).unwrap_or("none")
                ));
            }
            if metadata
                .get("filters")
                .and_then(Value::as_array)
                .is_some_and(|filters| !filters.is_empty())
            {
                return Err("it declares filters before blosc".into());
            }
            if text(&metadata, "order") != Some("C") {
                return Err("its chunks are not in C order".into());
            }
            (text(&metadata, "dtype"), shape(metadata.get("chunks")))
        }
        Some(3) => {
            let grid = metadata.get("chunk_grid");
            if grid.and_then(|g| text(g, "name")) != Some("regular") {
                return Err("its chunk grid is not regular".into());
            }
            let codecs = metadata.get("codecs").and_then(Value::as_array);
            let names: Vec<&str> = codecs
                .map(|list| list.iter().filter_map(|c| text(c, "name")).collect())
                .unwrap_or_default();
            if names != ["bytes", "blosc"] {
                return Err(format!("its codecs are {names:?}, not bytes then blosc"));
            }
            let endian = codecs
                .and_then(|list| list[0].get("configuration"))
                .and_then(|c| text(c, "endian"));
            if endian.is_some_and(|e| e != "little") {
                return Err("its bytes codec is not little-endian".into());
            }
            (
                text(&metadata, "data_type"),
                shape(
                    grid.and_then(|g| g.get("configuration"))
                        .and_then(|c| c.get("chunk_shape")),
                ),
            )
        }
        _ => return Err("its Zarr format is neither 2 nor 3".into()),
    };
    let element = match dtype {
        Some("<f4" | "<i4" | "float32" | "int32") => 4,
        Some("<f8" | "<i8" | "float64" | "int64") => 8,
        other => return Err(format!("its dtype {} is not a little-endian 4- or 8-byte number", other.unwrap_or("?"))),
    };
    let chunk = chunk.ok_or("it declares no chunk shape")?;
    if chunk.is_empty() || chunk.len() != array.shape().len() || chunk.contains(&0) {
        return Err("its chunk shape does not fit its shape".into());
    }
    Ok(Layout { element, chunk })
}

fn word(bytes: &[u8], at: usize) -> u64 {
    u64::from(u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap()))
}

/// The C-order strides, in elements, of `shape`.
fn strides(shape: &[u64]) -> Vec<u64> {
    let mut stride = vec![1u64; shape.len()];
    for d in (0..shape.len().saturating_sub(1)).rev() {
        stride[d] = stride[d + 1] * shape[d + 1];
    }
    stride
}

/// Step `index` through the box `ranges` in C order; false after the last.
fn advance(index: &mut [u64], ranges: &[Range<u64>]) -> bool {
    for d in (0..index.len()).rev() {
        index[d] += 1;
        if index[d] < ranges[d].end {
            return true;
        }
        index[d] = ranges[d].start;
    }
    false
}

/// Read `window` of `array` through `store`.
pub fn read(store: &HttpStore, array: &SourceArray, layout: &Layout, window: &[Range<u64>]) -> Result<Read, String> {
    let n = layout.chunk.len();
    if window.len() != n
        || window.iter().zip(array.shape()).any(|(r, size)| r.start >= r.end || r.end > *size)
    {
        return Ok(Read::Whole("the window is empty, outside the array or of another rank".into()));
    }
    let out_shape: Vec<u64> = window.iter().map(|r| r.end - r.start).collect();
    let mut out = vec![0u8; (out_shape.iter().product::<u64>() * layout.element) as usize];
    let out_stride = strides(&out_shape);
    let grid: Vec<Range<u64>> = window
        .iter()
        .zip(&layout.chunk)
        .map(|(r, c)| r.start / c..(r.end - 1) / c + 1)
        .collect();
    let mut chunk_index: Vec<u64> = grid.iter().map(|r| r.start).collect();
    loop {
        if let Some(reason) = read_chunk(store, array, layout, window, &chunk_index, &out_stride, &mut out)? {
            return Ok(Read::Whole(reason));
        }
        if !advance(&mut chunk_index, &grid) {
            return Ok(Read::Window(out));
        }
    }
}

/// Read one chunk's share of the window into `out`; Some(reason) when this
/// chunk cannot be read by blocks.
fn read_chunk(
    store: &HttpStore,
    array: &SourceArray,
    layout: &Layout,
    window: &[Range<u64>],
    chunk_index: &[u64],
    out_stride: &[u64],
    out: &mut [u8],
) -> Result<Option<String>, String> {
    let (chunk, element) = (&layout.chunk, layout.element);
    let n = chunk.len();
    let origin: Vec<u64> = chunk_index.iter().zip(chunk).map(|(i, c)| i * c).collect();
    let local: Vec<Range<u64>> = (0..n)
        .map(|d| window[d].start.max(origin[d]) - origin[d]..window[d].end.min(origin[d] + chunk[d]) - origin[d])
        .collect();
    let stride = strides(chunk);
    // The decoded byte spans the window needs: dimensions after `k` are whole,
    // so each index of the dimensions before it is one contiguous run.
    let mut k = n - 1;
    while k > 0 && local[k] == (0..chunk[k]) {
        k -= 1;
    }
    let run = (local[k].end - local[k].start) * stride[k];
    let mut spans: Vec<Range<u64>> = Vec::new();
    let outer = &local[..k];
    let mut index: Vec<u64> = outer.iter().map(|r| r.start).collect();
    loop {
        let base = index.iter().zip(&stride).map(|(i, s)| i * s).sum::<u64>() + local[k].start * stride[k];
        let span = base * element..(base + run) * element;
        match spans.last_mut() {
            Some(last) if last.end == span.start => last.end = span.end,
            _ => spans.push(span),
        }
        if !advance(&mut index, outer) {
            break;
        }
    }
    let decoded_bytes = chunk.iter().product::<u64>() * element;
    let key = array.chunk_key(chunk_index);
    let probe_end = (HEADER as u64 + 4 * decoded_bytes.div_ceil(PROBE_BLOCK)).min(PROBE);
    let Some(probe) = store.get_range(&key, 0, probe_end).map_err(|e| e.to_string())? else {
        return Ok(Some("a chunk it needs does not exist".into()));
    };
    let head = &probe.bytes;
    if head.len() < HEADER || head[0] != BLOSC1_FORMAT {
        return Ok(Some("a chunk is not a blosc format-2 buffer".into()));
    }
    let (flags, typesize) = (head[2], u64::from(head[3]));
    let (nbytes, blocksize, cbytes) = (word(head, 4), word(head, 8), word(head, 12));
    if nbytes != decoded_bytes || typesize == 0 || element % typesize != 0 || blocksize == 0 || blocksize > nbytes {
        return Ok(Some("a chunk's blosc header disagrees with the array metadata".into()));
    }
    let decoded: Vec<Vec<u8>> = if probe.whole || head.len() as u64 >= cbytes {
        // The probe already holds the whole chunk (a small chunk, or a server
        // that answers a range with the entire object): decode from it.
        decode(head, &spans, typesize)?
    } else if flags & MEMCPYED != 0 {
        // Stored raw after the header: the spans are byte ranges of the object.
        let wanted: Vec<Range<u64>> = spans.iter().map(|s| s.start + HEADER as u64..s.end + HEADER as u64).collect();
        let fetched = fetch_all(store, &key, &wanted)?;
        fetched.into_iter().map(|bytes| bytes.to_vec()).collect()
    } else {
        let blocks = nbytes.div_ceil(blocksize);
        let table_end = HEADER as u64 + 4 * blocks;
        if cbytes <= table_end {
            return Ok(Some("a chunk's blosc offset table is truncated".into()));
        }
        let mut table = head.to_vec();
        if (table.len() as u64) < table_end {
            let rest = fetch_all(store, &key, &[table.len() as u64..table_end])?;
            table.extend_from_slice(&rest[0]);
        }
        table.truncate(table_end as usize);
        let starts: Vec<u64> = (0..blocks as usize).map(|j| word(&table, HEADER + 4 * j)).collect();
        // A block runs to the next block's start in the buffer: c-blosc's
        // threads place blocks in the order they finish, not by index.
        let mut ordered: Vec<u64> = starts.clone();
        ordered.push(cbytes);
        ordered.sort_unstable();
        ordered.dedup();
        let needed: BTreeSet<u64> = spans
            .iter()
            .flat_map(|s| s.start / blocksize..=(s.end - 1) / blocksize)
            .collect();
        let mut extents: Vec<(u64, Range<u64>)> = Vec::with_capacity(needed.len());
        for &j in &needed {
            let start = starts[j as usize];
            let end = ordered[ordered.partition_point(|&o| o <= start).min(ordered.len() - 1)];
            if start < table_end || end <= start || end > cbytes {
                return Ok(Some("a chunk's blosc offset table is inconsistent".into()));
            }
            extents.push((j, start..end));
        }
        let wanted_bytes: u64 = extents.iter().map(|(_, e)| e.end - e.start).sum();
        if wanted_bytes as f64 >= WHOLE_SHARE * cbytes as f64 {
            let whole = fetch_all(store, &key, &[0..cbytes])?;
            decode(&whole[0], &spans, typesize)?
        } else {
            extents.sort_by_key(|(_, e)| e.start);
            let mut merged: Vec<Range<u64>> = Vec::new();
            for (_, e) in &extents {
                match merged.last_mut() {
                    Some(last) if e.start <= last.end + MERGE_GAP => last.end = last.end.max(e.end),
                    _ => merged.push(e.clone()),
                }
            }
            let fetched = fetch_all(store, &key, &merged)?;
            // A compact blosc buffer: the original header and offset table,
            // the fetched block bytes after them, the wanted blocks' offsets
            // pointed at their new places and the buffer size set to its own.
            let mut compact = table;
            for (range, bytes) in merged.iter().zip(&fetched) {
                let base = compact.len() as u64;
                for (j, e) in &extents {
                    if range.start <= e.start && e.end <= range.end {
                        let at = HEADER + 4 * *j as usize;
                        let offset = u32::try_from(base + e.start - range.start).map_err(|e| e.to_string())?;
                        compact[at..at + 4].copy_from_slice(&offset.to_le_bytes());
                    }
                }
                compact.extend_from_slice(bytes);
            }
            let size = u32::try_from(compact.len()).map_err(|e| e.to_string())?;
            compact[12..16].copy_from_slice(&size.to_le_bytes());
            decode(&compact, &spans, typesize)?
        }
    };
    // Copy each row of the last dimension from its decoded span into `out`.
    let row = ((local[n - 1].end - local[n - 1].start) * element) as usize;
    let rows = &local[..n - 1];
    let mut index: Vec<u64> = rows.iter().map(|r| r.start).collect();
    loop {
        let at = (index.iter().zip(&stride).map(|(i, s)| i * s).sum::<u64>() + local[n - 1].start * stride[n - 1]) * element;
        let to = (0..n)
            .map(|d| {
                let i = if d < n - 1 { index[d] } else { local[d].start };
                (origin[d] + i - window[d].start) * out_stride[d]
            })
            .sum::<u64>()
            * element;
        let s = spans.partition_point(|span| span.start <= at) - 1;
        let from = (at - spans[s].start) as usize;
        out[to as usize..to as usize + row].copy_from_slice(&decoded[s][from..from + row]);
        if !advance(&mut index, rows) {
            break;
        }
    }
    Ok(None)
}

/// Decode the byte `spans` of the chunk from a blosc buffer.
fn decode(buffer: &[u8], spans: &[Range<u64>], typesize: u64) -> Result<Vec<Vec<u8>>, String> {
    spans
        .iter()
        .map(|span| {
            let length = (span.end - span.start) as usize;
            let bytes = blosc_decompress_bytes_partial(buffer, span.start as usize, length, typesize as usize)
                .map_err(|e| format!("blosc block decode: {e}"))?;
            if bytes.len() != length {
                return Err(format!("blosc block decode returned {} of {length} bytes", bytes.len()));
            }
            Ok(bytes)
        })
        .collect()
}

/// GET each byte range of `key`, RANGE_WORKERS at a time, in order.
fn fetch_all(store: &HttpStore, key: &zarrs::storage::StoreKey, ranges: &[Range<u64>]) -> Result<Vec<bytes::Bytes>, String> {
    let slots: Vec<Mutex<Option<Result<bytes::Bytes, String>>>> = ranges.iter().map(|_| Mutex::new(None)).collect();
    let next = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for _ in 0..RANGE_WORKERS.min(ranges.len()) {
            scope.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::SeqCst);
                if i >= ranges.len() {
                    break;
                }
                let range = &ranges[i];
                let result = match store.get_range(key, range.start, range.end) {
                    Ok(Some(body)) if !body.whole && body.bytes.len() as u64 == range.end - range.start => Ok(body.bytes),
                    Ok(Some(body)) if body.whole && body.bytes.len() as u64 >= range.end => {
                        Ok(body.bytes.slice(range.start as usize..range.end as usize))
                    }
                    Ok(Some(body)) => Err(format!(
                        "{} bytes for range {}-{} of {key}",
                        body.bytes.len(),
                        range.start,
                        range.end - 1
                    )),
                    Ok(None) => Err(format!("{key} disappeared between its range reads")),
                    Err(e) => Err(e.to_string()),
                };
                *slots[i].lock().unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(result);
            });
        }
    });
    slots
        .into_iter()
        .map(|slot| {
            slot.into_inner()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .unwrap_or_else(|| Err("a range was not read".into()))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read as _, Write as _};
    use std::net::TcpListener;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::AtomicU64;
    use std::sync::Arc;
    use zarrs::array::codec::bytes_to_bytes::blosc::{
        blosc_compress_bytes, BloscCompressionLevel, BloscCompressor, BloscShuffleMode,
    };

    /// A file server that answers GET (whole or one byte range) and HEAD,
    /// counting the body bytes it sends.
    fn serve(root: PathBuf) -> (String, Arc<AtomicU64>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let sent = Arc::new(AtomicU64::new(0));
        let counter = sent.clone();
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let (root, counter) = (root.clone(), counter.clone());
                std::thread::spawn(move || {
                    let mut stream = stream.unwrap();
                    let mut request = Vec::new();
                    let mut byte = [0u8; 1];
                    while !request.ends_with(b"\r\n\r\n") && stream.read(&mut byte).unwrap_or(0) == 1 {
                        request.push(byte[0]);
                    }
                    let request = String::from_utf8_lossy(&request).to_string();
                    let mut words = request.split_whitespace();
                    let (method, path) = (words.next().unwrap_or(""), words.next().unwrap_or("/"));
                    let range = request.lines().find_map(|line| {
                        let (name, value) = line.split_once(':')?;
                        name.eq_ignore_ascii_case("range")
                            .then(|| value.trim().trim_start_matches("bytes=").to_owned())
                    });
                    let reply = match std::fs::read(root.join(path.trim_start_matches('/'))) {
                        Err(_) => b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".to_vec(),
                        Ok(body) => {
                            let (status, part) = match range {
                                Some(range) => {
                                    let (a, b) = range.split_once('-').unwrap();
                                    let a: usize = a.parse().unwrap();
                                    let b = b.parse::<usize>().unwrap().min(body.len() - 1);
                                    (format!("206 Partial Content\r\nContent-Range: bytes {a}-{b}/{}", body.len()),
                                     body[a..=b].to_vec())
                                }
                                None => ("200 OK".to_owned(), body),
                            };
                            let mut reply = format!(
                                "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                                part.len()
                            )
                            .into_bytes();
                            if method != "HEAD" {
                                counter.fetch_add(part.len() as u64, Ordering::SeqCst);
                                reply.extend(part);
                            }
                            reply
                        }
                    };
                    let _ = stream.write_all(&reply);
                });
            }
        });
        (url, sent)
    }

    /// A Zarr v2 float32 array of `shape` in `chunks`, blosc-lz4 with byte
    /// shuffle (ARCO ERA5's codec) and `blocksize`-byte blocks.
    fn blosc_array(store: &Path, name: &str, shape: &[u64], chunks: &[u64], blocksize: usize) {
        let directory = store.join(name);
        std::fs::create_dir_all(&directory).unwrap();
        let metadata = serde_json::json!({
            "zarr_format": 2, "shape": shape, "chunks": chunks, "dtype": "<f4", "fill_value": "NaN",
            "order": "C", "filters": null,
            "compressor": {"id": "blosc", "cname": "lz4", "clevel": 5, "shuffle": 1, "blocksize": blocksize}});
        std::fs::write(directory.join(".zarray"), serde_json::to_vec(&metadata).unwrap()).unwrap();
        let total: u64 = shape.iter().product();
        let values: Vec<f32> =
            (0..total).map(|i| 250.0 + (i as f32 * 0.37).sin() * 30.0 + (i % 97) as f32).collect();
        let shape_stride = strides(shape);
        let grid: Vec<Range<u64>> = shape.iter().zip(chunks).map(|(s, c)| 0..s.div_ceil(*c)).collect();
        let whole_chunk: Vec<Range<u64>> = chunks.iter().map(|c| 0..*c).collect();
        let mut index: Vec<u64> = vec![0; shape.len()];
        loop {
            // The chunk's C-order elements, padded with NaN past the array edge.
            let mut at: Vec<u64> = vec![0; shape.len()];
            let mut raw = Vec::new();
            loop {
                let global: Vec<u64> = (0..shape.len()).map(|d| index[d] * chunks[d] + at[d]).collect();
                let value = if global.iter().zip(shape).all(|(g, s)| g < s) {
                    values[global.iter().zip(&shape_stride).map(|(g, s)| g * s).sum::<u64>() as usize]
                } else {
                    f32::NAN
                };
                raw.extend(value.to_le_bytes());
                if !advance(&mut at, &whole_chunk) {
                    break;
                }
            }
            let encoded = blosc_compress_bytes(&raw, BloscCompressionLevel::try_from(5).unwrap(),
                BloscShuffleMode::Shuffle, 4, BloscCompressor::LZ4, blocksize, 1).unwrap();
            let key: Vec<String> = index.iter().map(u64::to_string).collect();
            std::fs::write(directory.join(key.join(".")), encoded).unwrap();
            if !advance(&mut index, &grid) {
                break;
            }
        }
    }

    fn whole_window(array: &SourceArray, window: &[Range<u64>]) -> Vec<u8> {
        array.retrieve_array_subset::<Vec<f32>>(&window.to_vec()).unwrap().iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    /// The defect: a cropped read moved every byte of every chunk it touched.
    /// The window read by blocks must equal the whole-chunk read exactly and
    /// move a fraction of its bytes.
    #[test]
    fn a_window_read_by_blocks_is_the_whole_chunk_read_for_a_fraction_of_the_bytes() {
        let root = std::env::temp_dir().join(format!("rw-zarr-blocks-{}", std::process::id()));
        let store = root.join("store");
        // ERA5's shape in miniature: one time per chunk, every level, the whole globe.
        blosc_array(&store, "t", &[2, 4, 181, 360], &[1, 4, 181, 360], 16384);
        // Several chunks per time, edge chunks padded, for windows that cross them.
        blosc_array(&store, "s", &[2, 45, 80], &[1, 20, 30], 1024);
        let (url, sent) = serve(root.clone());
        let http = Arc::new(HttpStore::new(&format!("{url}/store")).unwrap());
        let storage: zarrs::storage::ReadableStorage = http.clone();

        let t = Array::open(storage.clone(), "/t").unwrap();
        let layout_t = layout(&t).unwrap();
        let chunk_bytes = std::fs::metadata(store.join("t/1.0.0.0")).unwrap().len();
        let window = [1..2, 0..4, 60..75, 0..360];
        let before = sent.load(Ordering::SeqCst);
        let Read::Window(bytes) = read(&http, &t, &layout_t, &window).unwrap() else {
            panic!("not read by blocks")
        };
        let moved = sent.load(Ordering::SeqCst) - before;
        assert_eq!(bytes, whole_window(&t, &window));
        assert!(moved * 3 < chunk_bytes, "moved {moved} of a {chunk_bytes}-byte chunk");

        let s = Array::open(storage.clone(), "/s").unwrap();
        let layout_s = layout(&s).unwrap();
        for window in [[0..1, 5..44, 13..71], [1..2, 0..45, 0..80], [1..2, 44..45, 79..80]] {
            let Read::Window(bytes) = read(&http, &s, &layout_s, &window).unwrap() else {
                panic!("not read by blocks")
            };
            assert_eq!(bytes, whole_window(&s, &window), "{window:?}");
        }
        let _ = std::fs::remove_dir_all(&root);
    }

    /// An array with no blosc block table is refused here, and read whole by zarrs.
    #[test]
    fn an_array_without_a_blosc_block_table_is_left_to_zarrs() {
        let root = std::env::temp_dir().join(format!("rw-zarr-noblosc-{}", std::process::id()));
        let directory = root.join("store/raw");
        std::fs::create_dir_all(&directory).unwrap();
        let metadata = serde_json::json!({"zarr_format": 2, "shape": [4], "chunks": [4], "dtype": "<f4",
            "fill_value": null, "order": "C", "filters": null, "compressor": {"id": "zstd", "level": 1}});
        std::fs::write(directory.join(".zarray"), serde_json::to_vec(&metadata).unwrap()).unwrap();
        let (url, _) = serve(root.clone());
        let http: zarrs::storage::ReadableStorage = Arc::new(HttpStore::new(&format!("{url}/store")).unwrap());
        let array = Array::open(http, "/raw").unwrap();
        match layout(&array) {
            Err(reason) => assert!(reason.contains("zstd"), "{reason}"),
            Ok(_) => panic!("a zstd array was taken for block reads"),
        }
        let _ = std::fs::remove_dir_all(&root);
    }
}
