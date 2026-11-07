//! The point-forecast time series file (`rw-store.series.v1`): every surface plane of one fixed grid, for every
//! stored time and every member, tiled time-major, so a whole meteogram at a point (an ensemble's included) is about
//! three small reads: the header, one 12-byte index entry, one block.
//!
//! It is written once per fixed grid when the run finishes, from the grid's hourly column files' surface sections
//! (`columns/<domain>/*.wcol` and `columns/members/<member>/<domain>/*.wcol`): input and output only, no model
//! arithmetic, so every value is the very f32 the hourly file holds. A grid that moves (its files' `grid_hash`
//! differs) gets none; a reader then uses the hourly files.
//!
//! Layout, little-endian:
//! ```text
//! 0..8      b"RWSERIE1"
//! 8..12     u32 header length H
//! 12..12+H  JSON header, space padded: schema, domain, members, nx, ny, tile, tiles_x, tiles_y, times (ISO),
//!           times_unix, fields, units, grid_hash, locator (the hourly files' own), wrap_x, codec, index_offset,
//!           notes (the hourly files' surface notes), sources (per member, the hourly files read)
//! index     tiles_y * tiles_x entries of (u64 offset, u32 length), row-major
//! blocks    each block zlib(payload); payload = for each member, for each field: f32 [time][y][x] over the
//!           block's span, byte-shuffled into 4 planes
//! ```
//! Block (tx, ty) holds columns x in [tx*T, min(tx*T+T, nx-1)] and rows likewise (T = `tile`, 4 by default): blocks
//! overlap by one column, as the column file's tiles do, so the four corners of any cell are in one block. A time a
//! member's file is missing for is NaN in that member's series.

use crate::columns::{self, ColumnFile, Locator, span, shuffle_into, unshuffle};
use crate::error::{RwResult, RwStoreError};
use flate2::{Compression, read::ZlibDecoder, write::ZlibEncoder};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

pub const MAGIC: &[u8; 8] = b"RWSERIE1";
pub const SCHEMA: &str = "rw-store.series.v1";
/// The file name a grid's series is written under: `columns/<domain>/series.wser`.
pub const FILE_NAME: &str = "series.wser";

fn fmt(e: impl std::fmt::Display) -> RwStoreError {
    RwStoreError::Format(e.to_string())
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SeriesHeader {
    pub schema: String,
    pub domain: String,
    pub members: Vec<String>,
    pub nx: usize,
    pub ny: usize,
    pub tile: usize,
    pub tiles_x: usize,
    pub tiles_y: usize,
    pub times: Vec<String>,
    pub times_unix: Vec<i64>,
    pub fields: Vec<String>,
    pub units: Vec<String>,
    #[serde(default)]
    pub grid_hash: Option<String>,
    pub locator: Locator,
    #[serde(default)]
    pub wrap_x: Option<Value>,
    pub codec: String,
    pub index_offset: u64,
    #[serde(default)]
    pub notes: Value,
}

impl SeriesHeader {
    pub fn field(&self, name: &str) -> Option<usize> {
        self.fields.iter().position(|f| f == name)
    }
    pub fn member(&self, name: &str) -> Option<usize> {
        self.members.iter().position(|m| m == name)
    }
    pub fn block_span(&self, tx: usize, ty: usize) -> (usize, usize, usize, usize) {
        let (x0, x1) = span(tx, self.tile, self.nx);
        let (y0, y1) = span(ty, self.tile, self.ny);
        (x0, x1, y0, y1)
    }
    /// The block that holds grid cell (ix, iy) and the cell to its right and above.
    pub fn block_of(&self, ix: usize, iy: usize) -> (usize, usize) {
        ((ix / self.tile).min(self.tiles_x - 1), (iy / self.tile).min(self.tiles_y - 1))
    }
}

/// One decoded block: [member][field][time][y][x] over its span.
pub struct SeriesBlock {
    pub x0: usize,
    pub y0: usize,
    pub w: usize,
    pub h: usize,
    pub times: usize,
    pub fields: usize,
    pub data: Vec<f32>,
}

impl SeriesBlock {
    pub fn value(&self, member: usize, field: usize, time: usize, ix: usize, iy: usize) -> f32 {
        let plane = self.w * self.h;
        let at = ((member * self.fields + field) * self.times + time) * plane + (iy - self.y0) * self.w + (ix - self.x0);
        self.data[at]
    }
    /// One member's series of one field at fractional grid position (fx, fy), bilinear in f64 from the four stored
    /// corners (the column file's [`columns::SurfaceTile::sample`] arithmetic), NaN where a corner is missing.
    pub fn sample(&self, member: usize, field: usize, fx: f64, fy: f64, nx: usize, ny: usize) -> Vec<f64> {
        let fx = fx.clamp(0.0, (nx - 1) as f64);
        let fy = fy.clamp(0.0, (ny - 1) as f64);
        let (x0, y0) = (fx.floor() as usize, fy.floor() as usize);
        let (x1, y1) = ((x0 + 1).min(nx - 1), (y0 + 1).min(ny - 1));
        let (ax, ay) = (fx - x0 as f64, fy - y0 as f64);
        (0..self.times)
            .map(|t| {
                let g = |ix, iy| f64::from(self.value(member, field, t, ix, iy));
                (g(x0, y0) * (1.0 - ax) + g(x1, y0) * ax) * (1.0 - ay) + (g(x0, y1) * (1.0 - ax) + g(x1, y1) * ax) * ay
            })
            .collect()
    }
}

/// A series file opened for range reads.
pub struct SeriesFile<S: columns::RangeSource = columns::FileSource> {
    src: S,
    pub header: SeriesHeader,
}

impl SeriesFile<columns::FileSource> {
    pub fn open(path: &Path) -> RwResult<Self> {
        Self::from_source(columns::FileSource::open(path)?).map_err(|e| fmt(format!("{}: {e}", path.display())))
    }
}

impl<S: columns::RangeSource> SeriesFile<S> {
    pub fn from_source(src: S) -> RwResult<Self> {
        let mut head = src.read_range(0, 64 * 1024)?;
        if head.len() < 12 || &head[..8] != MAGIC.as_slice() {
            return Err(fmt("not a series file"));
        }
        let hlen = u32::from_le_bytes(head[8..12].try_into().unwrap()) as usize;
        if head.len() < 12 + hlen {
            head.extend(src.read_range(head.len() as u64, 12 + hlen - head.len())?);
        }
        let header: SeriesHeader = serde_json::from_slice(head[12..12 + hlen].trim_ascii_end()).map_err(fmt)?;
        if header.schema != SCHEMA || header.codec != "shuffle-zlib-f32" {
            return Err(fmt(format!("unknown series schema {} / codec {}", header.schema, header.codec)));
        }
        if header.fields.len() != header.units.len() || header.times.len() != header.times_unix.len() {
            return Err(fmt("series fields/units or times/times_unix differ in length"));
        }
        Ok(SeriesFile { src, header })
    }

    pub fn block_range(&self, tx: usize, ty: usize) -> RwResult<(u64, u32)> {
        let h = &self.header;
        if tx >= h.tiles_x || ty >= h.tiles_y {
            return Err(fmt(format!("block ({tx},{ty}) outside {}x{}", h.tiles_x, h.tiles_y)));
        }
        let e = self.src.read_range(h.index_offset + 12 * (ty * h.tiles_x + tx) as u64, 12)?;
        if e.len() != 12 {
            return Err(fmt("series index is truncated"));
        }
        Ok((u64::from_le_bytes(e[..8].try_into().unwrap()), u32::from_le_bytes(e[8..].try_into().unwrap())))
    }

    pub fn read_block(&self, tx: usize, ty: usize) -> RwResult<SeriesBlock> {
        let (off, len) = self.block_range(tx, ty)?;
        let blob = self.src.read_range(off, len as usize)?;
        if blob.len() != len as usize {
            return Err(fmt(format!("block ({tx},{ty}) is truncated")));
        }
        decode_block(&self.header, tx, ty, &blob)
    }

    /// The block holding the bilinear cell of (fx, fy).
    pub fn block_at(&self, fx: f64, fy: f64) -> RwResult<SeriesBlock> {
        let h = &self.header;
        let fx = fx.clamp(0.0, (h.nx - 1) as f64);
        let fy = fy.clamp(0.0, (h.ny - 1) as f64);
        let (tx, ty) = h.block_of(fx.floor() as usize, fy.floor() as usize);
        self.read_block(tx, ty)
    }
}

pub fn decode_block(h: &SeriesHeader, tx: usize, ty: usize, blob: &[u8]) -> RwResult<SeriesBlock> {
    let (x0, x1, y0, y1) = h.block_span(tx, ty);
    let (w, hh) = (x1 - x0 + 1, y1 - y0 + 1);
    let chunk = h.times.len() * w * hh;
    let count = h.members.len() * h.fields.len() * chunk;
    let mut payload = Vec::new();
    ZlibDecoder::new(blob).read_to_end(&mut payload).map_err(|e| fmt(format!("block ({tx},{ty}): {e}")))?;
    if payload.len() != count * 4 {
        return Err(fmt(format!("block ({tx},{ty}): payload {} bytes, expected {}", payload.len(), count * 4)));
    }
    let mut data = Vec::with_capacity(count);
    for c in 0..h.members.len() * h.fields.len() {
        let part = &payload[c * chunk * 4..(c + 1) * chunk * 4];
        data.extend(unshuffle(part, chunk, 4).into_iter().map(f32::from_bits));
    }
    Ok(SeriesBlock { x0, y0, w, h: hh, times: h.times.len(), fields: h.fields.len(), data })
}

/// One hourly file's surface section, held for the series build: its header and its surface index, read once; its
/// tiles read by offset as each band needs them (no file handle kept open between bands).
struct Source {
    path: PathBuf,
    header: columns::ColumnHeader,
    index: Vec<(u64, u32)>,
    member: usize,
    time: usize,
    /// For each series field, its position in this file's surface fields (None: the file lacks it, NaN).
    field_map: Vec<Option<usize>>,
}

/// What [`write_series`] wrote, or why it wrote nothing.
#[derive(Debug, Serialize)]
pub struct SeriesWritten {
    pub path: Option<PathBuf>,
    pub domain: String,
    pub members: usize,
    pub times: usize,
    pub fields: usize,
    pub bytes: u64,
    pub ms: f64,
    pub skipped: Option<String>,
}

/// The hourly column files of one grid under a store directory: `(member, path)` for `<store>/<domain>/*.wcol`
/// (member "det") and `<store>/members/<member>/<domain>/*.wcol`.
pub fn grid_files(store: &Path, domain: &str) -> Vec<(String, PathBuf)> {
    let mut out = vec![];
    let list = |dir: &Path| -> Vec<PathBuf> {
        let mut v: Vec<PathBuf> = std::fs::read_dir(dir)
            .map(|it| {
                it.filter_map(|e| e.ok().map(|e| e.path()))
                    .filter(|p| p.extension().is_some_and(|x| x == "wcol") && p.is_file())
                    .collect()
            })
            .unwrap_or_default();
        v.sort();
        v
    };
    for p in list(&store.join(domain)) {
        out.push(("det".to_string(), p));
    }
    let mut members: Vec<PathBuf> = std::fs::read_dir(store.join("members"))
        .map(|it| it.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.is_dir()).collect())
        .unwrap_or_default();
    members.sort();
    for m in members {
        let name = m.file_name().unwrap().to_string_lossy().to_string();
        for p in list(&m.join(domain)) {
            out.push((name.clone(), p));
        }
    }
    out
}

/// The grids (domain folders) a store directory holds column files for, det and members together.
pub fn grids(store: &Path) -> Vec<String> {
    let mut set = std::collections::BTreeSet::new();
    let scan = |dir: &Path, set: &mut std::collections::BTreeSet<String>| {
        if let Ok(it) = std::fs::read_dir(dir) {
            for e in it.flatten() {
                let p = e.path();
                let name = e.file_name().to_string_lossy().to_string();
                if p.is_dir() && name != "members" && !name.starts_with('.') {
                    let has = std::fs::read_dir(&p)
                        .map(|it| it.flatten().any(|f| f.path().extension().is_some_and(|x| x == "wcol")))
                        .unwrap_or(false);
                    if has {
                        set.insert(name);
                    }
                }
            }
        }
    };
    scan(store, &mut set);
    if let Ok(it) = std::fs::read_dir(store.join("members")) {
        for e in it.flatten() {
            scan(&e.path(), &mut set);
        }
    }
    set.into_iter().collect()
}

/// Write `<store>/<domain>/series.wser` from the grid's hourly files' surface sections. Nothing is written (and the
/// reason is returned) when the grid has no file with a surface section or when its files are not one fixed grid.
/// The file is written beside and renamed in. Bands of block rows are built one at a time, so memory is a band's
/// planes for every file, never the whole series.
pub fn write_series(store: &Path, domain: &str, tile: usize) -> RwResult<SeriesWritten> {
    let started = std::time::Instant::now();
    let skip = |why: String| SeriesWritten {
        path: None,
        domain: domain.to_string(),
        members: 0,
        times: 0,
        fields: 0,
        bytes: 0,
        ms: started.elapsed().as_secs_f64() * 1e3,
        skipped: Some(why),
    };
    if !(2..=64).contains(&tile) {
        return Err(fmt(format!("series tile must be 2..64 columns, got {tile}")));
    }
    let files = grid_files(store, domain);
    if files.is_empty() {
        return Ok(skip("no column files".into()));
    }
    // Headers and surface indexes, read side by side.
    let opened: Vec<(String, PathBuf, columns::ColumnHeader, Option<Vec<(u64, u32)>>)> = files
        .par_iter()
        .map(|(m, p)| {
            let f = ColumnFile::open(p)?;
            let index = f.surface_index()?;
            Ok((m.clone(), p.clone(), f.header, index))
        })
        .collect::<RwResult<_>>()?;
    let with_surface: Vec<_> = opened.into_iter().filter(|o| o.3.is_some()).collect();
    if with_surface.is_empty() {
        return Ok(skip("no column file carries a surface section (written before columns-surface-v1)".into()));
    }
    let first = with_surface[0].2.clone();
    let (nx, ny, mtile) = (first.nx, first.ny, first.tile);
    if mtile % tile != 0 {
        return Err(fmt(format!("the column files' tile {mtile} is not a multiple of the series tile {tile}")));
    }
    for o in &with_surface {
        let h = &o.2;
        if h.grid_hash != first.grid_hash || h.nx != nx || h.ny != ny || h.tile != mtile {
            return Ok(skip(format!(
                "the grid moves or differs between files ({} vs {}): a moving grid is read from its hourly files",
                o.1.display(),
                with_surface[0].1.display()
            )));
        }
    }
    let mut members: Vec<String> = vec![];
    for o in &with_surface {
        if !members.contains(&o.0) {
            members.push(o.0.clone());
        }
    }
    let valid = |h: &columns::ColumnHeader| h.valid_unix.unwrap_or(0);
    let mut times_unix: Vec<i64> = with_surface.iter().map(|o| valid(&o.2)).collect();
    times_unix.sort();
    times_unix.dedup();
    let mut fields: Vec<String> = vec![];
    let mut units: Vec<String> = vec![];
    let mut notes = serde_json::Map::new();
    for o in &with_surface {
        let s = o.2.surface.as_ref().unwrap();
        for (f, u) in s.fields.iter().zip(&s.units) {
            if !fields.contains(f) {
                fields.push(f.clone());
                units.push(u.clone());
            }
        }
        if let Some(n) = s.notes.as_object() {
            for (k, v) in n {
                notes.entry(k.clone()).or_insert_with(|| v.clone());
            }
        }
    }
    let sources: Vec<Source> = with_surface
        .into_iter()
        .map(|(m, path, header, index)| {
            let s = header.surface.as_ref().unwrap();
            let field_map = fields.iter().map(|f| s.field(f)).collect();
            Source {
                member: members.iter().position(|x| *x == m).unwrap(),
                time: times_unix.binary_search(&valid(&header)).unwrap(),
                path,
                index: index.unwrap(),
                header,
                field_map,
            }
        })
        .collect();
    let (nm, nf, nt) = (members.len(), fields.len(), times_unix.len());
    let tiles_x = (nx - 1).div_ceil(tile);
    let tiles_y = (ny - 1).div_ceil(tile);
    let iso = chrono_like;
    let mut header = json!({
        "schema": SCHEMA, "domain": domain, "members": members, "nx": nx, "ny": ny, "tile": tile,
        "tiles_x": tiles_x, "tiles_y": tiles_y,
        "times": times_unix.iter().map(|t| iso(*t)).collect::<Vec<_>>(), "times_unix": times_unix,
        "fields": fields, "units": units, "grid_hash": first.grid_hash, "locator": first.locator,
        "wrap_x": first.wrap_x, "codec": "shuffle-zlib-f32",
        "layout": "blocks of tile x tile cells (one-column overlap), each zlib of, for each member, for each field, \
                   f32 [time][y][x] byte-shuffled into 4 planes; index of (u64 offset, u32 length) per block",
        "index_offset": 0, "notes": notes,
        "sources": members.iter().enumerate().map(|(mi, m)| json!({
            "member": m,
            "files": sources.iter().filter(|s| s.member == mi).count(),
        })).collect::<Vec<_>>(),
    });
    let probe = serde_json::to_vec(&header).map_err(fmt)?;
    let index_offset = 12 + probe.len() + 24;
    header["index_offset"] = json!(index_offset);
    let mut head = serde_json::to_vec(&header).map_err(fmt)?;
    head.resize(index_offset - 12, b' ');
    let out = store.join(domain).join(FILE_NAME);
    // An ensemble that keeps no deterministic store has no <store>/<domain>/ yet: the series still goes there.
    std::fs::create_dir_all(store.join(domain)).map_err(RwStoreError::Io)?;
    let tmp = out.with_extension("wser.tmp");
    let mut file = std::fs::File::create(&tmp).map_err(RwStoreError::Io)?;
    file.write_all(MAGIC).map_err(RwStoreError::Io)?;
    file.write_all(&(head.len() as u32).to_le_bytes()).map_err(RwStoreError::Io)?;
    file.write_all(&head).map_err(RwStoreError::Io)?;
    let nblocks = tiles_x * tiles_y;
    file.write_all(&vec![0u8; nblocks * 12]).map_err(RwStoreError::Io)?;
    let mut index = vec![(0u64, 0u32); nblocks];
    let mut offset = (index_offset + nblocks * 12) as u64;
    let per_band = mtile / tile;
    let mtiles_x = first.tiles_x;
    // One band = one row of the hourly files' tiles = `per_band` rows of series blocks.
    for band in 0..first.tiles_y {
        let (by0, by1) = span(band, mtile, ny);
        let rows = by1 - by0 + 1;
        // [source][field][row][x], NaN where a source lacks a field.
        let planes: Vec<Vec<f32>> = sources
            .par_iter()
            .map(|s| -> RwResult<Vec<f32>> {
                let mut plane = vec![f32::NAN; nf * rows * nx];
                let src = columns::FileSource::open(&s.path)?;
                for tx in 0..mtiles_x {
                    let (off, len) = s.index[band * mtiles_x + tx];
                    let blob = columns::RangeSource::read_range(&src, off, len as usize)?;
                    let t = columns::decode_surface_tile(&s.header, tx, band, &blob)?;
                    for (fi, map) in s.field_map.iter().enumerate() {
                        let Some(si) = map else { continue };
                        for y in 0..t.h {
                            for x in 0..t.w {
                                plane[(fi * rows + (t.y0 + y - by0)) * nx + t.x0 + x] = t.f[*si][y * t.w + x];
                            }
                        }
                    }
                }
                Ok(plane)
            })
            .collect::<RwResult<_>>()?;
        // Which source holds (member, time).
        let mut at = vec![usize::MAX; nm * nt];
        for (i, s) in sources.iter().enumerate() {
            at[s.member * nt + s.time] = i;
        }
        let sy0 = band * per_band;
        let sy1 = ((band + 1) * per_band).min(tiles_y);
        let blocks: Vec<(usize, Vec<u8>)> = (sy0..sy1)
            .flat_map(|ty| (0..tiles_x).map(move |tx| (tx, ty)))
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|(tx, ty)| -> RwResult<(usize, Vec<u8>)> {
                let (x0, x1) = span(tx, tile, nx);
                let (y0, y1) = span(ty, tile, ny);
                let (w, h) = (x1 - x0 + 1, y1 - y0 + 1);
                let chunk = nt * w * h;
                let mut payload = vec![0u8; nm * nf * chunk * 4];
                let mut bits = vec![0u32; chunk];
                for m in 0..nm {
                    for f in 0..nf {
                        for t in 0..nt {
                            let si = at[m * nt + t];
                            for y in y0..=y1 {
                                for x in x0..=x1 {
                                    let v = if si == usize::MAX {
                                        f32::NAN
                                    } else {
                                        planes[si][(f * rows + (y - by0)) * nx + x]
                                    };
                                    bits[(t * h + (y - y0)) * w + (x - x0)] = v.to_bits();
                                }
                            }
                        }
                        let c = m * nf + f;
                        shuffle_into(&mut payload[c * chunk * 4..(c + 1) * chunk * 4], &bits, 4);
                    }
                }
                let mut enc = ZlibEncoder::new(Vec::with_capacity(payload.len() / 3), Compression::new(5));
                enc.write_all(&payload).map_err(RwStoreError::Io)?;
                Ok((ty * tiles_x + tx, enc.finish().map_err(RwStoreError::Io)?))
            })
            .collect::<RwResult<_>>()?;
        drop(planes);
        for (k, blob) in blocks {
            file.write_all(&blob).map_err(RwStoreError::Io)?;
            index[k] = (offset, blob.len() as u32);
            offset += blob.len() as u64;
        }
    }
    let mut raw = Vec::with_capacity(nblocks * 12);
    for (o, l) in &index {
        raw.extend_from_slice(&o.to_le_bytes());
        raw.extend_from_slice(&l.to_le_bytes());
    }
    file.seek(SeekFrom::Start(index_offset as u64)).map_err(RwStoreError::Io)?;
    file.write_all(&raw).map_err(RwStoreError::Io)?;
    file.sync_all().ok();
    drop(file);
    std::fs::rename(&tmp, &out).map_err(RwStoreError::Io)?;
    Ok(SeriesWritten {
        path: Some(out),
        domain: domain.to_string(),
        members: nm,
        times: nt,
        fields: nf,
        bytes: offset,
        ms: started.elapsed().as_secs_f64() * 1e3,
        skipped: None,
    })
}

/// Unix seconds as ISO UTC (2024-07-14T16:00:00Z), the column files' own form.
fn chrono_like(t: i64) -> String {
    columns::iso_time(t)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::columns::{ColumnsWrite, Field2, Field3, SurfaceWrite, Values, FIELDS_2D, FIELDS_3D, QUANTA_3D, UNITS_2D, UNITS_3D};

    fn store_file(dir: &Path, member: &str, domain: &str, valid: i64, nx: usize, ny: usize, k: f64, with_surface: bool) {
        let n = nx * ny;
        let nz = 3;
        let lat: Vec<f64> = (0..n).map(|i| 35.0 + (i / nx) as f64 * 0.03).collect();
        let lon: Vec<f64> = (0..n).map(|i| -98.0 + (i % nx) as f64 * 0.03).collect();
        let v3: Vec<Vec<f64>> = (0..7).map(|f| (0..nz * n).map(|i| f as f64 + i as f64 * 1e-3).collect()).collect();
        let v2: Vec<Vec<f64>> = (0..6).map(|f| (0..n).map(|i| 900.0 + f as f64 + i as f64 * 1e-3).collect()).collect();
        let precip: Vec<f64> = (0..n).map(|i| k + i as f64 * 0.01).collect();
        let gust: Vec<f64> = (0..n).map(|i| if i % 5 == 0 { f64::NAN } else { k * 2.0 + i as f64 }).collect();
        let w = ColumnsWrite {
            nx,
            ny,
            nz,
            lat: Values::F64(&lat),
            lon: Values::F64(&lon),
            fields3: (0..7)
                .map(|i| Field3 { name: FIELDS_3D[i], units: UNITS_3D[i], quantum: Some(QUANTA_3D[i]), data: Values::F64(&v3[i]), scale: 1.0, offset: 0.0, affine: None })
                .collect(),
            fields2: (0..6).map(|i| Field2 { name: FIELDS_2D[i + 2], units: UNITS_2D[i + 2], data: Values::F64(&v2[i]), scale: 1.0, offset: 0.0 }).collect(),
            domain,
            member,
            valid_unix: valid,
            tile: 16,
            meta: json!({}),
        };
        let s = SurfaceWrite {
            lat: w.lat,
            lon: w.lon,
            fields: vec![
                Field2 { name: "precip_mm", units: "mm", data: Values::F64(&precip), scale: 1.0, offset: 0.0 },
                Field2 { name: "gust_ms", units: "m/s", data: Values::F64(&gust), scale: 1.0, offset: 0.0 },
            ],
            notes: json!({"precip_mm": "run total"}),
        };
        let raw = columns::finish(columns::encode_main(&w).unwrap(), with_surface.then_some(&s)).unwrap();
        let d = if member == "det" { dir.join(domain) } else { dir.join("members").join(member).join(domain) };
        let stamp = columns::iso_time(valid).replace(['-', ':'], "");
        columns::write_bytes(&d.join(format!("{stamp}.wcol")), &raw).unwrap();
    }

    /// Two members, three hours (one member missing an hour): every value in the series is the hourly file's f32,
    /// a missing hour is NaN, and the bilinear sample is the column file's surface sample.
    #[test]
    fn series_holds_every_hourly_value() {
        let dir = std::env::temp_dir().join(format!("rw-series-test-{}", std::process::id()));
        std::fs::remove_dir_all(&dir).ok();
        let (nx, ny) = (37, 21);
        for (m, hours) in [("m01", vec![0, 1, 2]), ("m02", vec![0, 2])] {
            for h in hours {
                store_file(&dir, m, "d02", 1_720_972_800 + h * 3600, nx, ny, h as f64 + if m == "m02" { 100.0 } else { 0.0 }, true);
            }
        }
        let w = write_series(&dir, "d02", 4).unwrap();
        assert!(w.skipped.is_none(), "{:?}", w.skipped);
        let f = SeriesFile::open(w.path.as_ref().unwrap()).unwrap();
        let h = &f.header;
        assert_eq!(h.members, vec!["m01", "m02"]);
        assert_eq!(h.times.len(), 3);
        assert_eq!(h.fields, vec!["lat", "lon", "precip_mm", "gust_ms"]);
        let ip = h.field("precip_mm").unwrap();
        let ig = h.field("gust_ms").unwrap();
        for (fx, fy) in [(0.0, 0.0), (5.3, 7.9), (36.0, 20.0), (15.99, 16.01), (17.5, 3.25)] {
            let b = f.block_at(fx, fy).unwrap();
            for (mi, m) in ["m01", "m02"].iter().enumerate() {
                for (ti, t) in h.times_unix.iter().enumerate() {
                    let stamp = columns::iso_time(*t).replace(['-', ':'], "");
                    let p = dir.join("members").join(m).join("d02").join(format!("{stamp}.wcol"));
                    let series_p = b.sample(mi, ip, fx, fy, nx, ny)[ti];
                    let series_g = b.sample(mi, ig, fx, fy, nx, ny)[ti];
                    if !p.is_file() {
                        assert!(series_p.is_nan());
                        continue;
                    }
                    let c = ColumnFile::open(&p).unwrap();
                    let s = c.header.surface.as_ref().unwrap();
                    let (tx, ty) = c.header.tile_of(fx.floor() as usize, fy.floor() as usize);
                    let t = c.read_surface_tile(tx, ty).unwrap().unwrap();
                    let want_p = t.sample(s.field("precip_mm").unwrap(), fx, fy, nx, ny);
                    let want_g = t.sample(s.field("gust_ms").unwrap(), fx, fy, nx, ny);
                    assert_eq!(series_p.to_bits(), want_p.to_bits(), "{m} {ti} ({fx},{fy})");
                    assert_eq!(series_g.is_nan(), want_g.is_nan());
                    if !want_g.is_nan() {
                        assert_eq!(series_g.to_bits(), want_g.to_bits());
                    }
                }
            }
        }
        // A grid whose files carry no surface section gets no series; nor does a grid that moves.
        store_file(&dir, "det", "d03", 1_720_972_800, nx, ny, 0.0, false);
        assert!(write_series(&dir, "d03", 4).unwrap().skipped.is_some());
        store_file(&dir, "det", "d04", 1_720_972_800, nx, ny, 0.0, true);
        store_file(&dir, "det", "d04", 1_720_976_400, nx + 1, ny, 0.0, true);
        assert!(write_series(&dir, "d04", 4).unwrap().skipped.unwrap().contains("moves"));
        assert_eq!(grids(&dir), vec!["d02", "d03", "d04"]);
        std::fs::remove_dir_all(&dir).ok();
    }
}
