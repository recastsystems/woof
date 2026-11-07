//! rw-store version 2: the tiled column store (`rw-store.columns.v2`).
//!
//! One file holds every column of one grid at one valid time, for one member, chunked so that the sounding at any
//! point is one small read: the reader reads the header once, then the one tile that holds the point (a local
//! `pread` or an HTTP range read), and samples the column bilinearly. Nobody has to pick sounding points before a
//! run, and a server never downloads a whole hour to answer one click.
//!
//! Version 1 (`RWSTORE1`, the `.rws` hour file of [`crate::reader::HourReader`]) stays readable: [`open_any`] sniffs
//! the magic and hands back either reader, so stores written before version 2 keep working. Files written by the
//! pre-fold column writer (`GPWMCOL1`, schema `gpuwm-columns.v1`, gpuwm 2.9.6) have the same layout with every 3-D
//! field quantized and one code width for the whole tile, and read here unchanged.
//!
//! Layout, little-endian:
//! ```text
//! 0..8      b"RWSTORE2"   (b"GPWMCOL1" for pre-fold files)
//! 8..12     u32 header length H
//! 12..12+H  JSON header, space padded: schema, domain, member, valid, valid_unix, nx, ny, nz, tile, tiles_x,
//!           tiles_y, fields3d, units3d, modes3d (quantized | exact | affine), quanta3d (the quantum of a quantized
//!           field, else null), affine_block (columns per affine parameter block, when a field is affine),
//!           fields2d (lat and lon first), units2d, dtype2d (f32), codec (shuffle-zlib), filled, index_offset,
//!           grid_hash, locator {step, ix, iy, lat, lon}, meta (the writer's own provenance)
//! index     tiles_y * tiles_x entries of (u64 offset from file start, u32 length), row-major (ty outer)
//! tiles     each tile zlib(payload)
//! surface   (optional, `surface` in the header) the SURFACE SECTION: every surface plane of the point forecast,
//!           exact f32, on the same tiles: each tile zlib(payload) of f32 [field][y][x] byte-shuffled into 4 planes,
//!           fields in `surface.fields` order (lat and lon first, then the six sounding planes, then the rest);
//!           then the surface index, tiles_y * tiles_x entries of (u64 offset, u32 length) row-major, at
//!           `surface.index_offset`; then the 8-byte trailer `RWSURF01`, the last bytes of the file
//! ```
//! The surface section is additive inside schema `rw-store.columns.v2`: the main tiles keep exactly their 2-D planes
//! (the eight of `fields2d`), so every reader of the main tiles (soundings) reads a file with a surface section
//! unchanged, at the same sizes, and ignores the bytes after the last main tile. A point-forecast reader takes the
//! surface entry for tile k with one suffix range read (`bytes=-(12 * (ntiles - k) + 8)`), then the tile.
//! Tile (tx, ty) holds columns x in [tx*T, min(tx*T+T, nx-1)] and y likewise: tiles overlap by one column, so the
//! four corners of any grid cell lie in one tile and a bilinear sample is one read.
//!
//! Tile payload (the whole payload zlib-compressed):
//! 1. code widths: one byte (2 or 4) per INTEGER-coded 3-D field (quantized or affine), in header order
//!    (`GPWMCOL1`: one byte for the whole tile);
//! 2. every integer-coded 3-D field, in header order, [field][k][y][x], differenced up the column and zigzagged, its
//!    width in bytes each, byte-shuffled into width planes. A quantized field's integer is the value over its
//!    quantum, rounded (a missing value repeats the level below and is counted in `filled`); an affine field's is
//!    its 16-bit code (-32768 missing);
//! 3. every EXACT 3-D field, in header order, [field][k][y][x], as the f32 bit pattern differenced up the column
//!    (i32 wrapping difference, zigzagged), 4 bytes each, byte-shuffled into 4 planes: lossless, NaN included;
//! 4. every 2-D field as f32 [y][x], byte-shuffled into 4 planes;
//! 5. every AFFINE 3-D field's parameters, in header order: for each `affine_block` square block the tile touches
//!    (block rows y0/B..=y1/B outer, block columns x0/B..=x1/B inner), flags u8, center f32, scale f32.
//!
//! Decoding: a quantized field is `(q * quantum) as f32`; an exact field is the very f32 it was written from; an
//! affine field is v1's chunk arithmetic in f32: NaN when the block's flags say empty (1) or the code is -32768,
//! `center` when they say constant (2), else `center + scale * code`. A v1 hour transcoded with affine fields
//! ([`transcode_v1`]) therefore decodes to the bits the v1 reader decodes, at about v1's size.
//!
//! Global grids: a latitude-longitude grid that is periodic in x (its last column's east neighbour is its first, as on
//! a hex global lat-lon frame) is stored with one wrap column: column `nx` repeats column 0, its longitude carried on
//! past the last column (the first longitude plus 360). The header's `nx` counts it and `wrap_x` names it
//! (`{"source_nx": n, "columns": 1}`), so the strip between the last and the first longitude is a cell like any
//! other and every reader finds a click there with no change.
//!
//! Ensembles: one file per member (the header names it), the ensemble mean written as its own member; an ensemble
//! sounding reads the same tile of each member's file.

use crate::error::{RwResult, RwStoreError};
use crate::grid::GridFile;
use crate::reader::HourReader;
use flate2::{Compression, read::ZlibDecoder, write::ZlibEncoder};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::io::{Read, Write};
use std::path::Path;

pub const MAGIC: &[u8; 8] = b"RWSTORE2";
/// The pre-fold column writer's magic; same layout, every 3-D field quantized.
pub const MAGIC_PREFOLD: &[u8; 8] = b"GPWMCOL1";
/// Version 1 hour file magic (see [`crate::format::MAGIC`]).
pub const MAGIC_V1: &[u8; 8] = b"RWSTORE1";
pub const SCHEMA: &str = "rw-store.columns.v2";
/// The last 8 bytes of a file that carries a surface section.
pub const SURFACE_TRAILER: &[u8; 8] = b"RWSURF01";
pub const SCHEMA_PREFOLD: &str = "gpuwm-columns.v1";

/// The canonical sounding fields: names, units and quanta every column writer uses (the WOOF box, the DA loop).
pub const FIELDS_3D: [&str; 7] = ["p_hpa", "z_m", "t_c", "td_c", "u_ms", "v_ms", "omega_pas"];
pub const UNITS_3D: [&str; 7] = ["hPa", "m", "C", "C", "m/s", "m/s", "Pa/s"];
/// Quantum of each canonical 3-D field in its unit: 0.05 hPa, 0.5 m, 0.02 C, 0.05 C, 0.05 m/s, 0.05 m/s, 0.01 Pa/s.
pub const QUANTA_3D: [f64; 7] = [0.05, 0.5, 0.02, 0.05, 0.05, 0.05, 0.01];
pub const FIELDS_2D: [&str; 8] = ["lat", "lon", "psfc_hpa", "orog_m", "t2_c", "td2_c", "u10_ms", "v10_ms"];
pub const UNITS_2D: [&str; 8] = ["degrees_north", "degrees_east", "hPa", "m", "C", "C", "m/s", "m/s"];

/// Affine block flags, v1's on-disk values: the block holds no finite value / every finite value is `center`.
pub const AFFINE_EMPTY: u8 = 1;
pub const AFFINE_CONSTANT: u8 = 2;
/// The affine code of a missing value (v1's `MISSING_Q`).
pub const AFFINE_MISSING: i16 = i16::MIN;

fn fmt(e: impl std::fmt::Display) -> RwStoreError {
    RwStoreError::Format(e.to_string())
}

/// Values handed to the writer, either precision; read as f64.
#[derive(Clone, Copy)]
pub enum Values<'a> {
    F64(&'a [f64]),
    F32(&'a [f32]),
}

impl Values<'_> {
    #[inline]
    fn get(&self, i: usize) -> f64 {
        match self {
            Values::F64(a) => a[i],
            Values::F32(a) => a[i] as f64,
        }
    }
    fn len(&self) -> usize {
        match self {
            Values::F64(a) => a.len(),
            Values::F32(a) => a.len(),
        }
    }
}

/// One affine parameter block: v1's chunk flags, center and scale.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct AffineParam {
    pub flags: u8,
    pub center: f32,
    pub scale: f32,
}

impl AffineParam {
    /// v1's chunk arithmetic for one code.
    #[inline]
    pub fn decode(&self, q: i64) -> f32 {
        if self.flags & AFFINE_EMPTY != 0 || q == AFFINE_MISSING as i64 {
            f32::NAN
        } else if self.flags & AFFINE_CONSTANT != 0 {
            self.center
        } else {
            self.center + self.scale * (q as f32)
        }
    }
}

/// A 3-D field already held as 16-bit affine codes (a v1 hour's column chunks): `codes` [k][y][x] and one parameter
/// block per `block` x `block` columns, `params` [by][bx] with `nx.div_ceil(block)` blocks a row.
#[derive(Clone, Copy)]
pub struct Affine<'a> {
    pub block: usize,
    pub codes: &'a [i16],
    pub params: &'a [AffineParam],
}

/// One 3-D field, [k][y][x] with k = 0 the lowest level. The stored value is `data * scale + offset` (the offset is
/// skipped when it is 0). `quantum: None` stores that value's f32 exactly. `affine: Some` stores the codes and
/// parameters it carries instead (and `data`, `quantum`, `scale` and `offset` are not read).
pub struct Field3<'a> {
    pub name: &'a str,
    pub units: &'a str,
    pub quantum: Option<f64>,
    pub data: Values<'a>,
    pub scale: f64,
    pub offset: f64,
    pub affine: Option<Affine<'a>>,
}

impl Field3<'_> {
    fn mode(&self) -> Mode {
        match (self.affine, self.quantum) {
            (Some(_), _) => Mode::Affine,
            (None, Some(q)) => Mode::Quantized(q),
            (None, None) => Mode::Exact,
        }
    }
}

/// One 2-D field, [y][x], stored as `(data * scale + offset) as f32`.
pub struct Field2<'a> {
    pub name: &'a str,
    pub units: &'a str,
    pub data: Values<'a>,
    pub scale: f64,
    pub offset: f64,
}

/// Everything one file holds.
pub struct ColumnsWrite<'a> {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub lat: Values<'a>,
    pub lon: Values<'a>,
    pub fields3: Vec<Field3<'a>>,
    pub fields2: Vec<Field2<'a>>,
    pub domain: &'a str,
    pub member: &'a str,
    pub valid_unix: i64,
    /// Columns per tile side (4..=128; 16 is the store's default).
    pub tile: usize,
    /// The writer's own provenance (name, version, model, cycle, lead), copied into the header.
    pub meta: Value,
}

/// How one 3-D field is stored.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Mode {
    Quantized(f64),
    Exact,
    Affine,
}

impl Mode {
    fn name(&self) -> &'static str {
        match self {
            Mode::Quantized(_) => "quantized",
            Mode::Exact => "exact",
            Mode::Affine => "affine",
        }
    }
    fn integer(&self) -> bool {
        !matches!(self, Mode::Exact)
    }
}

pub(crate) fn span(t: usize, tile: usize, n: usize) -> (usize, usize) {
    let a = t * tile;
    (a, (a + tile).min(n - 1))
}

#[inline]
fn zigzag(d: i32) -> u32 {
    ((d << 1) ^ (d >> 31)) as u32
}
#[inline]
fn unzigzag(z: u32) -> i32 {
    ((z >> 1) as i32) ^ -((z & 1) as i32)
}

pub(crate) fn shuffle_into(out: &mut [u8], codes: &[u32], width: usize) {
    let n = codes.len();
    for (i, c) in codes.iter().enumerate() {
        for (b, byte) in c.to_le_bytes()[..width].iter().enumerate() {
            out[b * n + i] = *byte;
        }
    }
}

/// Blocks a tile's span touches along one axis: block indices b0..=b1.
fn blocks_of(a0: usize, a1: usize, block: usize) -> (usize, usize) {
    (a0 / block, a1 / block)
}

/// The source column of stored column `x`: the wrap column (x == nx of the source) repeats column 0.
#[inline]
fn src_x(w: &ColumnsWrite, x: usize) -> usize {
    if x == w.nx { 0 } else { x }
}

/// Longitude of stored column `x` on row `y`: the wrap column carries on past the last column.
fn lon_at(w: &ColumnsWrite, x: usize, y: usize) -> f64 {
    lon_wrap(w.lon, w.nx, x, y)
}

/// [`lon_at`] for any longitude plane of a source grid `src_nx` wide.
fn lon_wrap(lon: Values, src_nx: usize, x: usize, y: usize) -> f64 {
    let row = y * src_nx;
    if x == src_nx {
        let (first, last) = (lon.get(row), lon.get(row + src_nx - 1));
        last + (first - last).rem_euclid(360.0)
    } else {
        lon.get(row + x)
    }
}

/// The surface section of one file: every 2-D plane of the point forecast, each stored as `(data * scale + offset)
/// as f32` (exact: no quantum, nothing rounded), on the main tiles. `lat` and `lon` are the source grid's (the same
/// as the main part's) and are stored first; `fields` follow in order. A field whose inputs a frame does not carry is
/// left out by the caller, never filled.
pub struct SurfaceWrite<'a> {
    pub lat: Values<'a>,
    pub lon: Values<'a>,
    pub fields: Vec<Field2<'a>>,
    /// What each plane is, in words (accumulations, at-the-hour values), copied into the header's `surface.notes`.
    pub notes: Value,
}

/// One surface tile: every surface plane over the tile's span, f32.
fn encode_surface_tile(s: &SurfaceWrite, src_nx: usize, x0: usize, x1: usize, y0: usize, y1: usize) -> RwResult<Vec<u8>> {
    let sx = |x: usize| if x == src_nx { 0 } else { x };
    let plane = (x1 - x0 + 1) * (y1 - y0 + 1);
    let mut floats: Vec<u32> = Vec::with_capacity((s.fields.len() + 2) * plane);
    for y in y0..=y1 {
        for x in x0..=x1 {
            floats.push((s.lat.get(y * src_nx + sx(x)) as f32).to_bits());
        }
    }
    for y in y0..=y1 {
        for x in x0..=x1 {
            floats.push((lon_wrap(s.lon, src_nx, x, y) as f32).to_bits());
        }
    }
    for f in &s.fields {
        for y in y0..=y1 {
            for x in x0..=x1 {
                let a = f.data.get(y * src_nx + sx(x)) * f.scale;
                floats.push(((if f.offset == 0.0 { a } else { a + f.offset }) as f32).to_bits());
            }
        }
    }
    let mut payload = vec![0u8; floats.len() * 4];
    shuffle_into(&mut payload, &floats, 4);
    let mut enc = ZlibEncoder::new(Vec::with_capacity(payload.len() / 2), Compression::new(5));
    enc.write_all(&payload).map_err(RwStoreError::Io)?;
    enc.finish().map_err(RwStoreError::Io)
}

/// Whether the grid is periodic in x: every row is one latitude, steps east evenly, and the step from its last
/// column back to its first is that same step, so the row goes once round the globe. Such a grid gets a wrap column.
pub fn periodic_x(nx: usize, ny: usize, lat: Values, lon: Values) -> bool {
    if nx < 8 || lat.len() != nx * ny || lon.len() != nx * ny {
        return false;
    }
    let east = |a: f64, b: f64| (b - a).rem_euclid(360.0);
    (0..ny).all(|y| {
        let r = y * nx;
        let dx = east(lon.get(r), lon.get(r + 1));
        let tol = 0.01 * dx;
        dx > 0.0
            && (nx as f64 * dx - 360.0).abs() < 0.5 * dx
            && (east(lon.get(r + nx - 1), lon.get(r)) - dx).abs() < tol
            && (east(lon.get(r + nx - 2), lon.get(r + nx - 1)) - dx).abs() < tol
            && (lat.get(r) - lat.get(r + nx - 1)).abs() < 1e-3
    })
}

fn encode_tile(w: &ColumnsWrite, x0: usize, x1: usize, y0: usize, y1: usize) -> RwResult<(Vec<u8>, usize)> {
    let (tw, th, nz) = (x1 - x0 + 1, y1 - y0 + 1, w.nz);
    let n = w.nx * w.ny;
    let plane = tw * th;
    let mut ints: Vec<Vec<u32>> = Vec::new();
    let mut ecodes: Vec<u32> = Vec::new();
    let mut params: Vec<u8> = Vec::new();
    let mut filled = 0usize;
    let mut last = vec![0i64; plane];
    let mut lastbits = vec![0u32; plane];
    for f in &w.fields3 {
        let stored = |i: usize| {
            let a = f.data.get(i) * f.scale;
            if f.offset == 0.0 { a } else { a + f.offset }
        };
        match f.mode() {
            Mode::Quantized(quantum) => {
                let mut codes = Vec::with_capacity(nz * plane);
                last.iter_mut().for_each(|v| *v = 0);
                for k in 0..nz {
                    for y in y0..=y1 {
                        let row = k * n + y * w.nx;
                        for x in x0..=x1 {
                            let c = (y - y0) * tw + (x - x0);
                            let value = stored(row + src_x(w, x)) / quantum;
                            // A missing value repeats the level below (a zero difference); counted, never silent.
                            let q = if value.is_finite() {
                                value.round() as i64
                            } else {
                                filled += 1;
                                last[c]
                            };
                            let d = q - last[c];
                            last[c] = q;
                            codes.push(zigzag(d.clamp(i32::MIN as i64, i32::MAX as i64) as i32));
                        }
                    }
                }
                ints.push(codes);
            }
            Mode::Affine => {
                let a = f.affine.unwrap();
                let mut codes = Vec::with_capacity(nz * plane);
                last.iter_mut().for_each(|v| *v = 0);
                for k in 0..nz {
                    for y in y0..=y1 {
                        let row = k * n + y * w.nx;
                        for x in x0..=x1 {
                            let c = (y - y0) * tw + (x - x0);
                            let q = a.codes[row + x] as i64;
                            codes.push(zigzag((q - last[c]) as i32));
                            last[c] = q;
                        }
                    }
                }
                ints.push(codes);
                let bx_n = w.nx.div_ceil(a.block);
                let (by0, by1) = blocks_of(y0, y1, a.block);
                let (bx0, bx1) = blocks_of(x0, x1, a.block);
                for by in by0..=by1 {
                    for bx in bx0..=bx1 {
                        let p = a.params[by * bx_n + bx];
                        params.push(p.flags);
                        params.extend_from_slice(&p.center.to_le_bytes());
                        params.extend_from_slice(&p.scale.to_le_bytes());
                    }
                }
            }
            Mode::Exact => {
                lastbits.iter_mut().for_each(|v| *v = 0);
                for k in 0..nz {
                    for y in y0..=y1 {
                        let row = k * n + y * w.nx;
                        for x in x0..=x1 {
                            let c = (y - y0) * tw + (x - x0);
                            let bits = (stored(row + src_x(w, x)) as f32).to_bits();
                            ecodes.push(zigzag((bits as i32).wrapping_sub(lastbits[c] as i32)));
                            lastbits[c] = bits;
                        }
                    }
                }
            }
        }
    }
    let mut floats: Vec<u32> = Vec::with_capacity((w.fields2.len() + 2) * plane);
    let lat = Field2 { name: "lat", units: "", data: w.lat, scale: 1.0, offset: 0.0 };
    for f in [&lat].into_iter().chain(w.fields2.iter()) {
        for y in y0..=y1 {
            for x in x0..=x1 {
                floats.push(((f.data.get(y * w.nx + src_x(w, x)) * f.scale + f.offset) as f32).to_bits());
            }
        }
    }
    // lon second, as the header lists it: the wrap column's longitude carried past the last column.
    let lon: Vec<u32> = (y0..=y1).flat_map(|y| (x0..=x1).map(move |x| (lon_at(w, x, y) as f32).to_bits())).collect();
    let at = plane;
    floats.splice(at..at, lon);
    // Each integer field's codes are 2 bytes wide when every one fits, else 4.
    let widths: Vec<usize> = ints.iter().map(|c| if c.iter().all(|v| *v < 65536) { 2 } else { 4 }).collect();
    let total = widths.len()
        + ints.iter().zip(&widths).map(|(c, w)| c.len() * w).sum::<usize>()
        + ecodes.len() * 4
        + floats.len() * 4
        + params.len();
    let mut payload = vec![0u8; total];
    let mut at = 0;
    for w in &widths {
        payload[at] = *w as u8;
        at += 1;
    }
    for (c, w) in ints.iter().zip(&widths) {
        shuffle_into(&mut payload[at..at + c.len() * w], c, *w);
        at += c.len() * w;
    }
    shuffle_into(&mut payload[at..at + ecodes.len() * 4], &ecodes, 4);
    at += ecodes.len() * 4;
    shuffle_into(&mut payload[at..at + floats.len() * 4], &floats, 4);
    at += floats.len() * 4;
    payload[at..].copy_from_slice(&params);
    let mut enc = ZlibEncoder::new(Vec::with_capacity(payload.len() / 3), Compression::new(5));
    enc.write_all(&payload).map_err(RwStoreError::Io)?;
    Ok((enc.finish().map_err(RwStoreError::Io)?, filled))
}

/// SHA-256 of the grid (nx, ny, then lat and lon as f32): the same grid hashes the same across members and times,
/// so a reader can key its locator by it.
pub fn grid_hash(nx: usize, ny: usize, lat: &[f32], lon: &[f32]) -> String {
    let mut h = Sha256::new();
    h.update((nx as u64).to_le_bytes());
    h.update((ny as u64).to_le_bytes());
    for v in lat.iter().chain(lon) {
        h.update(v.to_le_bytes());
    }
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

fn iso(t: i64) -> String {
    let days = t.div_euclid(86_400);
    let secs = t.rem_euclid(86_400);
    // civil from days (Howard Hinnant)
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z", secs / 3600, secs / 60 % 60, secs % 60)
}

/// The main part of one file, encoded (header and tiles): [`finish`] lays it out, with a surface section or without.
/// Holding it lets a caller free the 3-D fields before it builds the surface planes.
pub struct Encoded {
    header: Value,
    blobs: Vec<Vec<u8>>,
    src_nx: usize,
    nx: usize,
    ny: usize,
    tile: usize,
    tiles_x: usize,
    tiles_y: usize,
}

impl Encoded {
    /// The source grid's width (the stored grid is one wider when it wraps).
    pub fn source_nx(&self) -> usize {
        self.src_nx
    }
}

/// Unix seconds as the ISO UTC form a column file's header holds (2024-07-14T16:00:00Z).
pub fn iso_time(t: i64) -> String {
    iso(t)
}

/// Encode one file in memory.
pub fn encode(w: &ColumnsWrite) -> RwResult<Vec<u8>> {
    finish(encode_main(w)?, None)
}

/// Encode the main part of one file (header fields and tiles), the layout left to [`finish`].
pub fn encode_main(w: &ColumnsWrite) -> RwResult<Encoded> {
    let (nx, ny, nz) = (w.nx, w.ny, w.nz);
    if nx < 2 || ny < 2 || nz < 1 {
        return Err(fmt(format!("{}: a {nx}x{ny}x{nz} grid has no columns to store", w.domain)));
    }
    if !(4..=128).contains(&w.tile) {
        return Err(fmt(format!("tile must be 4..128 columns, got {}", w.tile)));
    }
    let n = nx * ny;
    if w.lat.len() != n || w.lon.len() != n {
        return Err(fmt(format!("lat/lon hold {} and {} values for a {nx}x{ny} grid", w.lat.len(), w.lon.len())));
    }
    let mut affine_block = None;
    for f in &w.fields3 {
        match f.mode() {
            Mode::Affine => {
                let a = f.affine.unwrap();
                if a.block == 0 || affine_block.is_some_and(|b| b != a.block) {
                    return Err(fmt(format!("{}: affine block {} (every affine field shares one, > 0)", f.name, a.block)));
                }
                affine_block = Some(a.block);
                if a.codes.len() < nz * n {
                    return Err(fmt(format!("{}: {} codes, expected {}", f.name, a.codes.len(), nz * n)));
                }
                let blocks = nx.div_ceil(a.block) * ny.div_ceil(a.block);
                if a.params.len() != blocks {
                    return Err(fmt(format!("{}: {} parameter blocks, expected {blocks}", f.name, a.params.len())));
                }
            }
            mode => {
                if f.data.len() < nz * n {
                    return Err(fmt(format!("{}: {} values, expected {}", f.name, f.data.len(), nz * n)));
                }
                if let Mode::Quantized(q) = mode
                    && !(q.is_finite() && q > 0.0)
                {
                    return Err(fmt(format!("{}: quantum {q} must be finite and positive", f.name)));
                }
            }
        }
    }
    for f in &w.fields2 {
        if f.data.len() != n {
            return Err(fmt(format!("{}: {} values, expected {n}", f.name, f.data.len())));
        }
    }
    // A global grid periodic in x gets one wrap column (never with affine fields: their blocks are the source's).
    let wrap = affine_block.is_none() && periodic_x(nx, ny, w.lat, w.lon);
    let src_nx = nx;
    let nx = nx + wrap as usize;
    let n = nx * ny;
    let lat_at = |x: usize, y: usize| w.lat.get(y * src_nx + src_x(w, x));
    let tile = w.tile;
    let tiles_x = (nx - 1).div_ceil(tile);
    let tiles_y = (ny - 1).div_ceil(tile);
    let encoded: Vec<(Vec<u8>, usize)> = (0..tiles_x * tiles_y)
        .into_par_iter()
        .map(|t| {
            let (x0, x1) = span(t % tiles_x, tile, nx);
            let (y0, y1) = span(t / tiles_x, tile, ny);
            encode_tile(w, x0, x1, y0, y1)
        })
        .collect::<RwResult<_>>()?;
    let filled: usize = encoded.iter().map(|(_, f)| f).sum();
    let blobs: Vec<Vec<u8>> = encoded.into_iter().map(|(b, _)| b).collect();
    // The locator: lat/lon every `tile` grid points (and the last row and column), so a reader finds the tile for
    // a latitude and longitude from the header alone.
    let pick = |n: usize| -> Vec<usize> {
        let mut v: Vec<usize> = (0..n).step_by(tile).collect();
        if *v.last().unwrap() != n - 1 {
            v.push(n - 1);
        }
        v
    };
    let (ix, iy) = (pick(nx), pick(ny));
    let round = |x: f64| (x * 1e5).round() / 1e5;
    let mut llat = Vec::with_capacity(ix.len() * iy.len());
    let mut llon = Vec::with_capacity(ix.len() * iy.len());
    for &y in &iy {
        for &x in &ix {
            llat.push(round(lat_at(x, y)));
            llon.push(round(lon_at(w, x, y)));
        }
    }
    let lat32: Vec<f32> = (0..n).map(|i| lat_at(i % nx, i / nx) as f32).collect();
    let lon32: Vec<f32> = (0..n).map(|i| lon_at(w, i % nx, i / nx) as f32).collect();
    let mut fields2d = vec!["lat", "lon"];
    let mut units2d = vec!["degrees_north", "degrees_east"];
    fields2d.extend(w.fields2.iter().map(|f| f.name));
    units2d.extend(w.fields2.iter().map(|f| f.units));
    let modes: Vec<Mode> = w.fields3.iter().map(|f| f.mode()).collect();
    let mut header = json!({
        "schema": SCHEMA,
        "domain": w.domain,
        "member": w.member,
        "valid": iso(w.valid_unix),
        "valid_unix": w.valid_unix,
        "nx": nx, "ny": ny, "nz": nz,
        "tile": tile, "tiles_x": tiles_x, "tiles_y": tiles_y,
        "fields3d": w.fields3.iter().map(|f| f.name).collect::<Vec<_>>(),
        "units3d": w.fields3.iter().map(|f| f.units).collect::<Vec<_>>(),
        "modes3d": modes.iter().map(Mode::name).collect::<Vec<_>>(),
        "quanta3d": modes.iter().map(|m| if let Mode::Quantized(q) = m { Some(*q) } else { None }).collect::<Vec<_>>(),
        "codec3d": "per tile: a code width byte per quantized or affine field; quantized: multiples of the quantum, \
                    affine: 16-bit codes, both differenced up the column, zigzag, u16 or u32; exact: f32 bits \
                    differenced up the column, zigzag, u32; affine parameters (flags, center, scale) per block last",
        "fields2d": fields2d, "units2d": units2d,
        "dtype2d": "f32", "codec": "shuffle-zlib", "filled": filled,
        "grid_hash": grid_hash(nx, ny, &lat32, &lon32),
        "index_offset": 0,
        "locator": {"step": tile, "ix": ix, "iy": iy, "lat": llat, "lon": llon},
        "meta": w.meta,
    });
    if let Some(b) = affine_block {
        header["affine_block"] = json!(b);
    }
    if wrap {
        header["wrap_x"] = json!({"source_nx": src_nx, "columns": 1});
    }
    Ok(Encoded { header, blobs, src_nx, nx, ny, tile, tiles_x, tiles_y })
}

/// Lay out one file: header, index, main tiles and, when given, the surface section (its tiles on the same spans,
/// its index and the `RWSURF01` trailer).
pub fn finish(e: Encoded, surface: Option<&SurfaceWrite>) -> RwResult<Vec<u8>> {
    let Encoded { mut header, blobs, src_nx, nx, ny, tile, tiles_x, tiles_y } = e;
    let ntiles = tiles_x * tiles_y;
    let surface_blobs: Vec<Vec<u8>> = match surface {
        None => Vec::new(),
        Some(s) => {
            let n = src_nx * ny;
            if s.lat.len() != n || s.lon.len() != n {
                return Err(fmt(format!("surface lat/lon hold {} and {} values for a {src_nx}x{ny} grid", s.lat.len(), s.lon.len())));
            }
            for f in &s.fields {
                if f.data.len() != n {
                    return Err(fmt(format!("surface {}: {} values, expected {n}", f.name, f.data.len())));
                }
            }
            let mut fields = vec!["lat", "lon"];
            let mut units = vec!["degrees_north", "degrees_east"];
            fields.extend(s.fields.iter().map(|f| f.name));
            units.extend(s.fields.iter().map(|f| f.units));
            header["surface"] = json!({
                "fields": fields, "units": units, "dtype": "f32", "codec": "shuffle-zlib",
                "layout": "per tile (the main tiles' spans): f32 [field][y][x] byte-shuffled into 4 planes, zlib;                            index of (u64 offset, u32 length) per tile at index_offset; trailer RWSURF01",
                "index_offset": 0, "notes": s.notes,
            });
            (0..ntiles)
                .into_par_iter()
                .map(|t| {
                    let (x0, x1) = span(t % tiles_x, tile, nx);
                    let (y0, y1) = span(t / tiles_x, tile, ny);
                    encode_surface_tile(s, src_nx, x0, x1, y0, y1)
                })
                .collect::<RwResult<_>>()?
        }
    };
    // The offsets depend on the header's own length: leave room for the numbers, then pad to them.
    let probe = serde_json::to_vec(&header).map_err(fmt)?;
    let index_offset = 12 + probe.len() + 48;
    let main_end = index_offset + blobs.len() * 12 + blobs.iter().map(|b| b.len()).sum::<usize>();
    let surface_index = main_end + surface_blobs.iter().map(|b| b.len()).sum::<usize>();
    header["index_offset"] = json!(index_offset);
    if surface.is_some() {
        header["surface"]["index_offset"] = json!(surface_index);
    }
    let mut head = serde_json::to_vec(&header).map_err(fmt)?;
    if head.len() > index_offset - 12 {
        return Err(fmt("column file header outgrew its room"));
    }
    head.resize(index_offset - 12, b' ');
    let total = if surface.is_some() { surface_index + ntiles * 12 + 8 } else { main_end };
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&(head.len() as u32).to_le_bytes());
    out.extend_from_slice(&head);
    let mut offset = (index_offset + blobs.len() * 12) as u64;
    for b in &blobs {
        out.extend_from_slice(&offset.to_le_bytes());
        out.extend_from_slice(&(b.len() as u32).to_le_bytes());
        offset += b.len() as u64;
    }
    for b in &blobs {
        out.extend_from_slice(b);
    }
    if surface.is_some() {
        for b in &surface_blobs {
            out.extend_from_slice(b);
        }
        let mut offset = main_end as u64;
        for b in &surface_blobs {
            out.extend_from_slice(&offset.to_le_bytes());
            out.extend_from_slice(&(b.len() as u32).to_le_bytes());
            offset += b.len() as u64;
        }
        out.extend_from_slice(SURFACE_TRAILER);
    }
    debug_assert_eq!(out.len(), total);
    Ok(out)
}

/// Encode and write one file: written beside and renamed in, so an uploader or a reader never sees half a file.
pub fn write(path: &Path, w: &ColumnsWrite) -> RwResult<u64> {
    write_bytes(path, &encode(w)?)
}

/// Write an already laid out file ([`finish`]) the same way: beside, then renamed in.
pub fn write_bytes(path: &Path, out: &[u8]) -> RwResult<u64> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(RwStoreError::Io)?;
    }
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    let tmp = std::path::PathBuf::from(tmp);
    std::fs::write(&tmp, out).map_err(RwStoreError::Io)?;
    std::fs::rename(&tmp, path).map_err(RwStoreError::Io)?;
    Ok(out.len() as u64)
}

// ---------------------------------------------------------------------------------------------------------------
// Reading

/// Byte ranges of a stored file: a local file, a buffer, or (in a caller) an HTTP range request.
pub trait RangeSource: Send + Sync {
    fn read_range(&self, offset: u64, len: usize) -> RwResult<Vec<u8>>;
}

/// A local file read by offset (pread), never mapped or read whole.
pub struct FileSource {
    file: std::fs::File,
}

impl FileSource {
    pub fn open(path: &Path) -> RwResult<Self> {
        Ok(FileSource { file: std::fs::File::open(path).map_err(RwStoreError::Io)? })
    }
}

impl RangeSource for FileSource {
    fn read_range(&self, offset: u64, len: usize) -> RwResult<Vec<u8>> {
        let mut buf = vec![0u8; len];
        let mut got = 0usize;
        while got < len {
            #[cfg(unix)]
            let n = std::os::unix::fs::FileExt::read_at(&self.file, &mut buf[got..], offset + got as u64);
            #[cfg(windows)]
            let n = std::os::windows::fs::FileExt::seek_read(&self.file, &mut buf[got..], offset + got as u64);
            let n = n.map_err(RwStoreError::Io)?;
            if n == 0 {
                break;
            }
            got += n;
        }
        buf.truncate(got);
        Ok(buf)
    }
}

impl RangeSource for Vec<u8> {
    fn read_range(&self, offset: u64, len: usize) -> RwResult<Vec<u8>> {
        let a = (offset as usize).min(self.len());
        Ok(self[a..(a + len).min(self.len())].to_vec())
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Locator {
    pub step: usize,
    pub ix: Vec<usize>,
    pub iy: Vec<usize>,
    pub lat: Vec<f64>,
    pub lon: Vec<f64>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ColumnHeader {
    pub schema: String,
    pub domain: String,
    pub member: String,
    pub valid: String,
    #[serde(default)]
    pub valid_unix: Option<i64>,
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub tile: usize,
    pub tiles_x: usize,
    pub tiles_y: usize,
    pub fields3d: Vec<String>,
    #[serde(default)]
    pub units3d: Vec<String>,
    /// quantized | exact | affine per 3-D field (absent in pre-fold files: every field quantized).
    #[serde(default)]
    pub modes3d: Vec<String>,
    pub quanta3d: Vec<Option<f64>>,
    #[serde(default)]
    pub affine_block: Option<usize>,
    pub fields2d: Vec<String>,
    #[serde(default)]
    pub units2d: Vec<String>,
    pub dtype2d: String,
    pub codec: String,
    pub filled: u64,
    pub index_offset: u64,
    #[serde(default)]
    pub grid_hash: Option<String>,
    /// A global grid's wrap column: `{"source_nx": n, "columns": 1}`, column n repeating column 0.
    #[serde(default)]
    pub wrap_x: Option<Value>,
    pub locator: Locator,
    #[serde(default)]
    pub meta: Value,
    /// The surface section, when the file carries one.
    #[serde(default)]
    pub surface: Option<SurfaceHeader>,
}

/// The header's description of a file's surface section.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SurfaceHeader {
    pub fields: Vec<String>,
    pub units: Vec<String>,
    pub dtype: String,
    pub codec: String,
    pub index_offset: u64,
    #[serde(default)]
    pub notes: Value,
}

impl SurfaceHeader {
    pub fn field(&self, name: &str) -> Option<usize> {
        self.fields.iter().position(|f| f == name)
    }
}

impl ColumnHeader {
    pub fn field3(&self, name: &str) -> Option<usize> {
        self.fields3d.iter().position(|f| f == name)
    }
    pub fn field2(&self, name: &str) -> Option<usize> {
        self.fields2d.iter().position(|f| f == name)
    }
    /// How 3-D field `i` is stored.
    pub fn mode3(&self, i: usize) -> Mode {
        match self.modes3d.get(i).map(String::as_str) {
            Some("affine") => Mode::Affine,
            Some("exact") => Mode::Exact,
            _ => match self.quanta3d[i] {
                Some(q) => Mode::Quantized(q),
                None => Mode::Exact,
            },
        }
    }
    /// Whether this is a pre-fold (`gpuwm-columns.v1`) file: one code width for the whole tile.
    pub fn prefold(&self) -> bool {
        self.schema == SCHEMA_PREFOLD
    }
    /// Unit of a 3-D field (pre-fold files carry none: theirs are the canonical units).
    pub fn units3(&self, i: usize) -> &str {
        self.units3d.get(i).map(String::as_str).unwrap_or_else(|| {
            FIELDS_3D.iter().position(|f| *f == self.fields3d[i]).map(|j| UNITS_3D[j]).unwrap_or("")
        })
    }
    pub fn units2(&self, i: usize) -> &str {
        self.units2d.get(i).map(String::as_str).unwrap_or_else(|| {
            FIELDS_2D.iter().position(|f| *f == self.fields2d[i]).map(|j| UNITS_2D[j]).unwrap_or("")
        })
    }
    /// Tile spans: columns [x0, x1] and rows [y0, y1], inclusive.
    pub fn tile_span(&self, tx: usize, ty: usize) -> (usize, usize, usize, usize) {
        let (x0, x1) = span(tx, self.tile, self.nx);
        let (y0, y1) = span(ty, self.tile, self.ny);
        (x0, x1, y0, y1)
    }
    /// The tile that holds grid cell (ix, iy) and the cell to its right and above (when those exist).
    pub fn tile_of(&self, ix: usize, iy: usize) -> (usize, usize) {
        ((ix / self.tile).min(self.tiles_x - 1), (iy / self.tile).min(self.tiles_y - 1))
    }
}

/// One decoded tile: every 3-D field [k][y][x] and every 2-D field [y][x] over the tile's span, as f32.
pub struct Tile {
    pub x0: usize,
    pub y0: usize,
    pub w: usize,
    pub h: usize,
    pub nz: usize,
    pub f3: Vec<Vec<f32>>,
    pub f2: Vec<Vec<f32>>,
}

impl Tile {
    fn contains(&self, ix: usize, iy: usize) -> bool {
        ix >= self.x0 && iy >= self.y0 && ix < self.x0 + self.w && iy < self.y0 + self.h
    }
    /// One 3-D field's column at a grid point of the tile, lowest level first.
    pub fn column3(&self, field: usize, ix: usize, iy: usize) -> Vec<f32> {
        let c = (iy - self.y0) * self.w + (ix - self.x0);
        let plane = self.w * self.h;
        (0..self.nz).map(|k| self.f3[field][k * plane + c]).collect()
    }
    pub fn point2(&self, field: usize, ix: usize, iy: usize) -> f32 {
        self.f2[field][(iy - self.y0) * self.w + (ix - self.x0)]
    }
    /// The bilinear profile at fractional grid position (fx, fy): f32 corner weights, a corner that is not finite
    /// drops out and the rest are renormalised, NaN where no corner is finite. This is the arithmetic of the v1
    /// reader's `read_profile_3d`, so a column stored exactly in either version samples to the same bits.
    pub fn profile3(&self, field: usize, fx: f64, fy: f64, nx: usize, ny: usize) -> Vec<f32> {
        let fx = fx.clamp(0.0, (nx - 1) as f64);
        let fy = fy.clamp(0.0, (ny - 1) as f64);
        let (x0, x1) = (fx.floor() as usize, fx.ceil() as usize);
        let (y0, y1) = (fy.floor() as usize, fy.ceil() as usize);
        let wx = (fx - x0 as f64) as f32;
        let wy = (fy - y0 as f64) as f32;
        let corners = [
            (x0, y0, (1.0 - wx) * (1.0 - wy)),
            (x1, y0, wx * (1.0 - wy)),
            (x0, y1, (1.0 - wx) * wy),
            (x1, y1, wx * wy),
        ];
        let plane = self.w * self.h;
        let cols: Vec<(usize, f32)> =
            corners.iter().map(|&(ix, iy, w)| ((iy - self.y0) * self.w + (ix - self.x0), w)).collect();
        let a = &self.f3[field];
        (0..self.nz)
            .map(|k| {
                let mut weight_sum = 0.0f32;
                let mut value_sum = 0.0f32;
                for (c, weight) in &cols {
                    let value = a[k * plane + c];
                    if value.is_finite() {
                        weight_sum += weight;
                        value_sum += weight * value;
                    }
                }
                if weight_sum > 0.0 { value_sum / weight_sum } else { f32::NAN }
            })
            .collect()
    }
}

/// A version 2 (or pre-fold) column file opened for range reads.
pub struct ColumnFile<S: RangeSource = FileSource> {
    src: S,
    pub header: ColumnHeader,
    index: Vec<(u64, u32)>,
}

/// First read: magic, header length and, usually, the whole header and index.
const FIRST_READ: usize = 256 * 1024;

impl ColumnFile<FileSource> {
    pub fn open(path: &Path) -> RwResult<Self> {
        Self::from_source(FileSource::open(path)?).map_err(|e| fmt(format!("{}: {e}", path.display())))
    }
}

impl<S: RangeSource> ColumnFile<S> {
    pub fn from_source(src: S) -> RwResult<Self> {
        let mut head = src.read_range(0, FIRST_READ)?;
        if head.len() < 12 {
            return Err(fmt("column file is shorter than its 12-byte preamble"));
        }
        let magic: [u8; 8] = head[..8].try_into().unwrap();
        let magic: &[u8] = &magic;
        if magic != MAGIC.as_slice() && magic != MAGIC_PREFOLD.as_slice() {
            return Err(fmt(format!("not a column file (magic {:?})", String::from_utf8_lossy(magic))));
        }
        let hlen = u32::from_le_bytes(head[8..12].try_into().unwrap()) as usize;
        if head.len() < 12 + hlen {
            head.extend(src.read_range(head.len() as u64, 12 + hlen - head.len())?);
        }
        let header: ColumnHeader = serde_json::from_slice(head[12..12 + hlen].trim_ascii_end()).map_err(fmt)?;
        let schema_ok = if magic == MAGIC.as_slice() { header.schema == SCHEMA } else { header.schema == SCHEMA_PREFOLD };
        if !schema_ok {
            return Err(fmt(format!("magic {} with schema '{}'", String::from_utf8_lossy(magic), header.schema)));
        }
        if header.quanta3d.len() != header.fields3d.len() {
            return Err(fmt("quanta3d and fields3d differ in length"));
        }
        if !header.modes3d.is_empty() {
            if header.modes3d.len() != header.fields3d.len() {
                return Err(fmt("modes3d and fields3d differ in length"));
            }
            if let Some(m) = header.modes3d.iter().find(|m| !["quantized", "exact", "affine"].contains(&m.as_str())) {
                return Err(fmt(format!("unknown 3-D field mode '{m}'")));
            }
            if header.modes3d.iter().any(|m| m == "affine") && !header.affine_block.is_some_and(|b| b > 0) {
                return Err(fmt("an affine field without affine_block"));
            }
            if header.prefold() {
                return Err(fmt("a pre-fold file names field modes"));
            }
        }
        if header.codec != "shuffle-zlib" || header.dtype2d != "f32" {
            return Err(fmt(format!("unknown codec {} / dtype2d {}", header.codec, header.dtype2d)));
        }
        if let Some(s) = &header.surface {
            if s.codec != "shuffle-zlib" || s.dtype != "f32" || s.fields.len() != s.units.len() || s.fields.len() < 2
                || s.fields[0] != "lat" || s.fields[1] != "lon"
            {
                return Err(fmt("a surface section that is not shuffle-zlib f32 planes after lat and lon"));
            }
        }
        let ntiles = header.tiles_x * header.tiles_y;
        let (a, b) = (header.index_offset as usize, header.index_offset as usize + ntiles * 12);
        let raw = if head.len() >= b { head[a..b].to_vec() } else { src.read_range(a as u64, b - a)? };
        if raw.len() != b - a {
            return Err(fmt("index is truncated"));
        }
        let index = raw
            .chunks_exact(12)
            .map(|e| (u64::from_le_bytes(e[..8].try_into().unwrap()), u32::from_le_bytes(e[8..].try_into().unwrap())))
            .collect();
        Ok(ColumnFile { src, header, index })
    }

    pub fn source(&self) -> &S {
        &self.src
    }

    /// Byte range (offset, length) of one tile, for a caller that fetches it itself.
    pub fn tile_range(&self, tx: usize, ty: usize) -> (u64, u32) {
        self.index[ty * self.header.tiles_x + tx]
    }

    pub fn read_tile(&self, tx: usize, ty: usize) -> RwResult<Tile> {
        if tx >= self.header.tiles_x || ty >= self.header.tiles_y {
            return Err(fmt(format!("tile ({tx},{ty}) outside {}x{}", self.header.tiles_x, self.header.tiles_y)));
        }
        let (off, len) = self.tile_range(tx, ty);
        let blob = self.src.read_range(off, len as usize)?;
        if blob.len() != len as usize {
            return Err(fmt(format!("tile ({tx},{ty}) is truncated")));
        }
        decode_tile(&self.header, tx, ty, &blob)
    }

    /// Byte range (offset, length) of one SURFACE tile (one 12-byte read of the surface index), or None when the file
    /// carries no surface section.
    pub fn surface_tile_range(&self, tx: usize, ty: usize) -> RwResult<Option<(u64, u32)>> {
        let Some(s) = &self.header.surface else { return Ok(None) };
        if tx >= self.header.tiles_x || ty >= self.header.tiles_y {
            return Err(fmt(format!("tile ({tx},{ty}) outside {}x{}", self.header.tiles_x, self.header.tiles_y)));
        }
        let e = self.src.read_range(s.index_offset + 12 * (ty * self.header.tiles_x + tx) as u64, 12)?;
        if e.len() != 12 {
            return Err(fmt("surface index is truncated"));
        }
        Ok(Some((u64::from_le_bytes(e[..8].try_into().unwrap()), u32::from_le_bytes(e[8..].try_into().unwrap()))))
    }

    /// The whole surface index (every tile's (offset, length)), one read, or None without a surface section.
    pub fn surface_index(&self) -> RwResult<Option<Vec<(u64, u32)>>> {
        let Some(s) = &self.header.surface else { return Ok(None) };
        let n = self.header.tiles_x * self.header.tiles_y;
        let raw = self.src.read_range(s.index_offset, n * 12)?;
        if raw.len() != n * 12 {
            return Err(fmt("surface index is truncated"));
        }
        Ok(Some(
            raw.chunks_exact(12)
                .map(|e| (u64::from_le_bytes(e[..8].try_into().unwrap()), u32::from_le_bytes(e[8..].try_into().unwrap())))
                .collect(),
        ))
    }

    /// One surface tile, or None when the file carries no surface section.
    pub fn read_surface_tile(&self, tx: usize, ty: usize) -> RwResult<Option<SurfaceTile>> {
        let Some((off, len)) = self.surface_tile_range(tx, ty)? else { return Ok(None) };
        let blob = self.src.read_range(off, len as usize)?;
        if blob.len() != len as usize {
            return Err(fmt(format!("surface tile ({tx},{ty}) is truncated")));
        }
        decode_surface_tile(&self.header, tx, ty, &blob).map(Some)
    }

    /// Every surface plane over the whole grid (each tile decoded once, in parallel), in `surface.fields` order: for
    /// checks, never for a click.
    pub fn read_full_surface(&self) -> RwResult<Option<Vec<Vec<f32>>>> {
        let h = &self.header;
        let Some(s) = &h.surface else { return Ok(None) };
        let n = h.nx * h.ny;
        let tiles: Vec<SurfaceTile> = (0..h.tiles_x * h.tiles_y)
            .into_par_iter()
            .map(|t| self.read_surface_tile(t % h.tiles_x, t / h.tiles_x).map(Option::unwrap))
            .collect::<RwResult<_>>()?;
        let mut out = vec![vec![f32::NAN; n]; s.fields.len()];
        for t in &tiles {
            for y in 0..t.h {
                for x in 0..t.w {
                    let g = (t.y0 + y) * h.nx + t.x0 + x;
                    for (i, f) in out.iter_mut().enumerate() {
                        f[g] = t.f[i][y * t.w + x];
                    }
                }
            }
        }
        Ok(Some(out))
    }

    /// The tile holding the bilinear cell of (fx, fy), read once.
    pub fn tile_at(&self, fx: f64, fy: f64) -> RwResult<Tile> {
        let fx = fx.clamp(0.0, (self.header.nx - 1) as f64);
        let fy = fy.clamp(0.0, (self.header.ny - 1) as f64);
        let (tx, ty) = self.header.tile_of(fx.floor() as usize, fy.floor() as usize);
        let t = self.read_tile(tx, ty)?;
        debug_assert!(t.contains(fx.ceil() as usize, fy.ceil() as usize));
        Ok(t)
    }

    /// The whole grid's latitude and longitude (every tile decoded, in parallel): for a reader that builds its own
    /// locator. Key it by `header.grid_hash`; every member and time on the grid shares it.
    pub fn lat_lon(&self) -> RwResult<(Vec<f32>, Vec<f32>)> {
        let h = &self.header;
        let (ilat, ilon) = (h.field2("lat").ok_or_else(|| fmt("no lat"))?, h.field2("lon").ok_or_else(|| fmt("no lon"))?);
        let tiles: Vec<Tile> = (0..h.tiles_x * h.tiles_y)
            .into_par_iter()
            .map(|t| self.read_tile(t % h.tiles_x, t / h.tiles_x))
            .collect::<RwResult<_>>()?;
        let mut lat = vec![f32::NAN; h.nx * h.ny];
        let mut lon = vec![f32::NAN; h.nx * h.ny];
        for t in &tiles {
            for y in 0..t.h {
                for x in 0..t.w {
                    let g = (t.y0 + y) * h.nx + t.x0 + x;
                    lat[g] = t.f2[ilat][y * t.w + x];
                    lon[g] = t.f2[ilon][y * t.w + x];
                }
            }
        }
        Ok((lat, lon))
    }

    /// Every 3-D field over the whole grid, [k][y][x] (each tile decoded once, in parallel): for checks and
    /// conversions, never for a click.
    pub fn read_full(&self) -> RwResult<(Vec<Vec<f32>>, Vec<Vec<f32>>)> {
        let h = &self.header;
        let n = h.nx * h.ny;
        let tiles: Vec<Tile> = (0..h.tiles_x * h.tiles_y)
            .into_par_iter()
            .map(|t| self.read_tile(t % h.tiles_x, t / h.tiles_x))
            .collect::<RwResult<_>>()?;
        let mut f3 = vec![vec![f32::NAN; h.nz * n]; h.fields3d.len()];
        let mut f2 = vec![vec![f32::NAN; n]; h.fields2d.len()];
        for t in &tiles {
            let plane = t.w * t.h;
            for y in 0..t.h {
                for x in 0..t.w {
                    let g = (t.y0 + y) * h.nx + t.x0 + x;
                    let c = y * t.w + x;
                    for (i, f) in f3.iter_mut().enumerate() {
                        for k in 0..h.nz {
                            f[k * n + g] = t.f3[i][k * plane + c];
                        }
                    }
                    for (i, f) in f2.iter_mut().enumerate() {
                        f[g] = t.f2[i][c];
                    }
                }
            }
        }
        Ok((f3, f2))
    }
}

pub(crate) fn unshuffle(data: &[u8], n: usize, width: usize) -> Vec<u32> {
    (0..n)
        .map(|i| {
            let mut b = [0u8; 4];
            for (j, byte) in b[..width].iter_mut().enumerate() {
                *byte = data[j * n + i];
            }
            u32::from_le_bytes(b)
        })
        .collect()
}

/// Decode one tile's compressed bytes (from [`ColumnFile::tile_range`]) under its header.
pub fn decode_tile(h: &ColumnHeader, tx: usize, ty: usize, blob: &[u8]) -> RwResult<Tile> {
    let (x0, x1, y0, y1) = h.tile_span(tx, ty);
    let (w, hh, nz) = (x1 - x0 + 1, y1 - y0 + 1, h.nz);
    let plane = w * hh;
    let bad = |what: String| fmt(format!("tile ({tx},{ty}): {what}"));
    let mut payload = Vec::new();
    ZlibDecoder::new(blob).read_to_end(&mut payload).map_err(|e| bad(e.to_string()))?;
    let modes: Vec<Mode> = (0..h.fields3d.len()).map(|i| h.mode3(i)).collect();
    let nint = modes.iter().filter(|m| m.integer()).count();
    let nexact = modes.len() - nint;
    let naffine = modes.iter().filter(|m| **m == Mode::Affine).count();
    let block = h.affine_block.unwrap_or(1);
    let (by0, by1) = blocks_of(y0, y1, block);
    let (bx0, bx1) = blocks_of(x0, x1, block);
    let nbx = bx1 - bx0 + 1;
    let nblocks = if naffine > 0 { (by1 - by0 + 1) * nbx } else { 0 };
    // Code widths: one per integer field, or one for the whole tile in a pre-fold file.
    let (widths, mut at): (Vec<usize>, usize) = if h.prefold() {
        let wd = *payload.first().ok_or_else(|| bad("empty".into()))? as usize;
        (vec![wd; nint], 1)
    } else {
        if payload.len() < nint {
            return Err(bad("shorter than its code widths".into()));
        }
        (payload[..nint].iter().map(|b| *b as usize).collect(), nint)
    };
    if let Some(wd) = widths.iter().find(|wd| **wd != 2 && **wd != 4) {
        return Err(bad(format!("code width {wd}")));
    }
    let count = nz * plane;
    let nf = h.fields2d.len() * plane;
    let want = at + widths.iter().map(|wd| count * wd).sum::<usize>() + nexact * count * 4 + nf * 4 + naffine * nblocks * 9;
    if payload.len() != want {
        return Err(bad(format!("payload {} bytes, expected {want}", payload.len())));
    }
    let mut ints: Vec<Vec<u32>> = Vec::with_capacity(nint);
    if h.prefold() {
        // A pre-fold tile shuffles every quantized field's codes as one stream of one width.
        let wd = widths.first().copied().unwrap_or(2);
        let all = unshuffle(&payload[at..at + nint * count * wd], nint * count, wd);
        at += nint * count * wd;
        ints.extend(all.chunks_exact(count.max(1)).map(<[u32]>::to_vec));
    } else {
        for wd in &widths {
            ints.push(unshuffle(&payload[at..at + count * wd], count, *wd));
            at += count * wd;
        }
    }
    let ecodes = unshuffle(&payload[at..at + nexact * count * 4], nexact * count, 4);
    at += nexact * count * 4;
    let floats = unshuffle(&payload[at..at + nf * 4], nf, 4);
    at += nf * 4;
    let mut params: Vec<AffineParam> = Vec::with_capacity(naffine * nblocks);
    for p in payload[at..].chunks_exact(9) {
        params.push(AffineParam {
            flags: p[0],
            center: f32::from_le_bytes(p[1..5].try_into().unwrap()),
            scale: f32::from_le_bytes(p[5..9].try_into().unwrap()),
        });
    }
    // Which parameter block each column of the tile reads.
    let block_of: Vec<usize> = if naffine > 0 {
        (0..plane).map(|c| ((y0 + c / w) / block - by0) * nbx + ((x0 + c % w) / block - bx0)).collect()
    } else {
        Vec::new()
    };
    let (mut ii, mut ei, mut ai) = (0usize, 0usize, 0usize);
    let mut f3 = Vec::with_capacity(modes.len());
    for mode in &modes {
        let mut out = vec![0f32; count];
        match mode {
            Mode::Quantized(quantum) => {
                let codes = &ints[ii];
                ii += 1;
                let mut last = vec![0i64; plane];
                for k in 0..nz {
                    for c in 0..plane {
                        let q = last[c] + unzigzag(codes[k * plane + c]) as i64;
                        last[c] = q;
                        out[k * plane + c] = (q as f64 * quantum) as f32;
                    }
                }
            }
            Mode::Affine => {
                let codes = &ints[ii];
                ii += 1;
                let ps = &params[ai * nblocks..(ai + 1) * nblocks];
                ai += 1;
                let mut last = vec![0i64; plane];
                for k in 0..nz {
                    for c in 0..plane {
                        let q = last[c] + unzigzag(codes[k * plane + c]) as i64;
                        last[c] = q;
                        out[k * plane + c] = ps[block_of[c]].decode(q);
                    }
                }
            }
            Mode::Exact => {
                let codes = &ecodes[ei * count..(ei + 1) * count];
                ei += 1;
                let mut last = vec![0u32; plane];
                for k in 0..nz {
                    for c in 0..plane {
                        let bits = (last[c] as i32).wrapping_add(unzigzag(codes[k * plane + c])) as u32;
                        last[c] = bits;
                        out[k * plane + c] = f32::from_bits(bits);
                    }
                }
            }
        }
        f3.push(out);
    }
    let f2 = (0..h.fields2d.len())
        .map(|i| floats[i * plane..(i + 1) * plane].iter().map(|b| f32::from_bits(*b)).collect())
        .collect();
    Ok(Tile { x0, y0, w, h: hh, nz, f3, f2 })
}

/// One decoded surface tile: every surface plane [y][x] over the tile's span, as f32, in `surface.fields` order.
pub struct SurfaceTile {
    pub x0: usize,
    pub y0: usize,
    pub w: usize,
    pub h: usize,
    pub f: Vec<Vec<f32>>,
}

impl SurfaceTile {
    pub fn point(&self, field: usize, ix: usize, iy: usize) -> f32 {
        self.f[field][(iy - self.y0) * self.w + (ix - self.x0)]
    }
    /// The bilinear value at fractional grid position (fx, fy) (the arithmetic of the sounding's surface sample, in
    /// f64 from the four stored corners). NaN when a corner is not finite: a plane that is missing at a corner is
    /// missing at the point.
    pub fn sample(&self, field: usize, fx: f64, fy: f64, nx: usize, ny: usize) -> f64 {
        let fx = fx.clamp(0.0, (nx - 1) as f64);
        let fy = fy.clamp(0.0, (ny - 1) as f64);
        let (x0, y0) = (fx.floor() as usize, fy.floor() as usize);
        let (x1, y1) = ((x0 + 1).min(nx - 1), (y0 + 1).min(ny - 1));
        let (ax, ay) = (fx - x0 as f64, fy - y0 as f64);
        let g = |ix, iy| f64::from(self.point(field, ix, iy));
        (g(x0, y0) * (1.0 - ax) + g(x1, y0) * ax) * (1.0 - ay) + (g(x0, y1) * (1.0 - ax) + g(x1, y1) * ax) * ay
    }
}

/// Decode one surface tile's compressed bytes under its header.
pub fn decode_surface_tile(h: &ColumnHeader, tx: usize, ty: usize, blob: &[u8]) -> RwResult<SurfaceTile> {
    let s = h.surface.as_ref().ok_or_else(|| fmt("the file carries no surface section"))?;
    let (x0, x1, y0, y1) = h.tile_span(tx, ty);
    let (w, hh) = (x1 - x0 + 1, y1 - y0 + 1);
    let plane = w * hh;
    let mut payload = Vec::new();
    ZlibDecoder::new(blob).read_to_end(&mut payload).map_err(|e| fmt(format!("surface tile ({tx},{ty}): {e}")))?;
    let nf = s.fields.len() * plane;
    if payload.len() != nf * 4 {
        return Err(fmt(format!("surface tile ({tx},{ty}): payload {} bytes, expected {}", payload.len(), nf * 4)));
    }
    let floats = unshuffle(&payload, nf, 4);
    let f = (0..s.fields.len())
        .map(|i| floats[i * plane..(i + 1) * plane].iter().map(|b| f32::from_bits(*b)).collect())
        .collect();
    Ok(SurfaceTile { x0, y0, w, h: hh, f })
}

// ---------------------------------------------------------------------------------------------------------------
// Either version

/// Which store version a file is, from its first 8 bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Version {
    /// `RWSTORE1`: the v1 hour file ([`HourReader`]).
    V1,
    /// `RWSTORE2`: the tiled column store.
    V2,
    /// `GPWMCOL1`: the pre-fold column store, read as v2.
    Prefold,
}

pub fn sniff(path: &Path) -> RwResult<Version> {
    let b = FileSource::open(path)?.read_range(0, 8)?;
    match b.as_slice() {
        x if x == MAGIC_V1.as_slice() => Ok(Version::V1),
        x if x == MAGIC.as_slice() => Ok(Version::V2),
        x if x == MAGIC_PREFOLD.as_slice() => Ok(Version::Prefold),
        x => Err(fmt(format!("{}: not an rw-store file (magic {:?})", path.display(), String::from_utf8_lossy(x)))),
    }
}

/// An rw-store file of either version.
pub enum StoreFile {
    V1(HourReader),
    V2(ColumnFile),
}

/// Open an rw-store file of either version: v1 `.rws` hour files keep working beside v2 column files.
pub fn open_any(path: &Path) -> RwResult<StoreFile> {
    match sniff(path)? {
        Version::V1 => Ok(StoreFile::V1(HourReader::open(path)?)),
        Version::V2 | Version::Prefold => Ok(StoreFile::V2(ColumnFile::open(path)?)),
    }
}

/// One v1 pressure3d variable as affine codes: its column chunks' flags, center and scale and their i16 codes, read
/// from the hour file's own index (v1's frozen layout: a 64-byte header with meta_len at 12, index_count at 16 and
/// index_offset at 24; 64-byte index records; zstd-compressed i16 payloads in [y][x][level] order).
fn v1_affine(raw: &[u8], var_id: u16, nx: usize, ny: usize, levels: usize, block: usize) -> RwResult<(Vec<i16>, Vec<AffineParam>)> {
    let u32_at = |o: usize| u32::from_le_bytes(raw[o..o + 4].try_into().unwrap());
    let u64_at = |o: usize| u64::from_le_bytes(raw[o..o + 8].try_into().unwrap());
    if raw.len() < 64 || &raw[..8] != MAGIC_V1.as_slice() {
        return Err(fmt("not a v1 hour file"));
    }
    let (count, offset) = (u64_at(16) as usize, u64_at(24) as usize);
    if offset + count * 64 > raw.len() {
        return Err(fmt("v1 index runs past the file"));
    }
    let n = nx * ny;
    let bx_n = nx.div_ceil(block);
    let mut codes = vec![0i16; levels * n];
    let mut params = vec![AffineParam { flags: AFFINE_EMPTY, center: 0.0, scale: 0.0 }; bx_n * ny.div_ceil(block)];
    let mut seen = vec![false; params.len()];
    for i in 0..count {
        let r = offset + i * 64;
        if u16::from_le_bytes(raw[r..r + 2].try_into().unwrap()) != var_id || raw[r + 2] != 1 {
            continue; // another variable, or a 2-D tile
        }
        let flags = raw[r + 3];
        let (cy, cx) = (u32_at(r + 4) as usize, u32_at(r + 8) as usize);
        let (off, len, raw_len) = (u64_at(r + 12) as usize, u32_at(r + 20) as usize, u32_at(r + 24) as usize);
        let center = f32::from_le_bytes(raw[r + 28..r + 32].try_into().unwrap());
        let scale = f32::from_le_bytes(raw[r + 32..r + 36].try_into().unwrap());
        let (y0, x0) = (cy * block, cx * block);
        if y0 >= ny || x0 >= nx {
            return Err(fmt(format!("v1 chunk ({cy},{cx}) outside the {nx}x{ny} grid")));
        }
        let (rows, cols) = ((ny - y0).min(block), (nx - x0).min(block));
        params[cy * bx_n + cx] = AffineParam { flags, center, scale };
        seen[cy * bx_n + cx] = true;
        if flags & AFFINE_EMPTY != 0 || (flags & AFFINE_CONSTANT != 0 && len == 0) {
            continue; // decoded from the flags alone; the codes stay 0
        }
        let comp = raw.get(off..off + len).ok_or_else(|| fmt(format!("v1 chunk ({cy},{cx}) runs past the file")))?;
        let dense = zstd::bulk::decompress(comp, raw_len).map_err(|e| fmt(format!("v1 chunk ({cy},{cx}): {e}")))?;
        if dense.len() != rows * cols * levels * 2 {
            return Err(fmt(format!("v1 chunk ({cy},{cx}): {} bytes, expected {}", dense.len(), rows * cols * levels * 2)));
        }
        for yl in 0..rows {
            for xl in 0..cols {
                let base = (yl * cols + xl) * levels;
                let g = (y0 + yl) * nx + x0 + xl;
                for k in 0..levels {
                    let b = 2 * (base + k);
                    codes[k * n + g] = i16::from_le_bytes([dense[b], dense[b + 1]]);
                }
            }
        }
    }
    if let Some(i) = seen.iter().position(|s| !s) {
        return Err(fmt(format!("v1 variable {var_id} has no chunk ({},{})", i / bx_n, i % bx_n)));
    }
    Ok((codes, params))
}

/// Rewrite a v1 hour file (and its run's `grid.rwg`) as a v2 column file holding every value EXACTLY: each
/// pressure3d volume becomes an affine 3-D field (v1's own codes and chunk parameters, its levels in the file's
/// order, its own name and units), each surface2d field a 2-D field. Every value of the v2 file is the bit pattern
/// the v1 reader decodes, at about the v1 file's size.
pub fn transcode_v1(hour: &Path, grid: &Path, out: &Path, domain: &str, member: &str, tile: usize) -> RwResult<u64> {
    let r = HourReader::open(hour)?;
    let g = GridFile::open(grid)?;
    let m = r.meta().clone();
    if g.nx != m.nx || g.ny != m.ny {
        return Err(fmt(format!("grid {}x{} vs hour {}x{}", g.nx, g.ny, m.nx, m.ny)));
    }
    if m.chunking.col_x != m.chunking.col_y {
        return Err(fmt(format!("v1 column chunks {}x{} are not square", m.chunking.col_x, m.chunking.col_y)));
    }
    let block = m.chunking.col_x;
    let vols: Vec<_> = m.variables.iter().filter(|v| v.kind == "pressure3d").collect();
    let nz = vols.first().map(|v| v.levels_hpa.len()).ok_or_else(|| fmt("hour file holds no 3-D volume"))?;
    if let Some(v) = vols.iter().find(|v| v.levels_hpa.len() != nz) {
        return Err(fmt(format!("{} has {} levels, the others {nz}", v.name, v.levels_hpa.len())));
    }
    let raw = std::fs::read(hour).map_err(RwStoreError::Io)?;
    let data3: Vec<(Vec<i16>, Vec<AffineParam>)> =
        vols.par_iter().map(|v| v1_affine(&raw, v.id, m.nx, m.ny, nz, block)).collect::<RwResult<_>>()?;
    let surf: Vec<_> = m.variables.iter().filter(|v| v.kind == "surface2d").collect();
    let data2: Vec<Vec<f32>> = surf.par_iter().map(|v| r.read_full_2d(&v.name)).collect::<RwResult<_>>()?;
    let w = ColumnsWrite {
        nx: m.nx,
        ny: m.ny,
        nz,
        lat: Values::F32(&g.lat),
        lon: Values::F32(&g.lon),
        fields3: vols
            .iter()
            .zip(&data3)
            .map(|(v, (codes, params))| Field3 {
                name: &v.name,
                units: &v.units,
                quantum: None,
                data: Values::F32(&[]),
                scale: 1.0,
                offset: 0.0,
                affine: Some(Affine { block, codes, params }),
            })
            .collect(),
        fields2: surf
            .iter()
            .zip(&data2)
            .map(|(v, d)| Field2 { name: &v.name, units: &v.units, data: Values::F32(d), scale: 1.0, offset: 0.0 })
            .collect(),
        domain,
        member,
        valid_unix: m.valid_unix.unwrap_or(0),
        tile,
        meta: json!({
            "transcoded_from": m.schema, "model": m.model, "run": m.run, "forecast_hour": m.forecast_hour,
            "lead_seconds": m.lead_seconds, "levels": vols[0].levels_hpa, "v1_writer": m.writer.name,
            "writer": concat!("rw-store ", env!("CARGO_PKG_VERSION"), " transcode_v1"),
        }),
    };
    write(out, &w)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Grid {
        nx: usize,
        ny: usize,
        nz: usize,
        lat: Vec<f64>,
        lon: Vec<f64>,
        v3: Vec<Vec<f64>>,
        v2: Vec<Vec<f64>>,
    }

    fn grid(nx: usize, ny: usize, nz: usize) -> Grid {
        let n = nx * ny;
        let f3 = |f: &dyn Fn(usize, usize, usize) -> f64| -> Vec<f64> {
            (0..nz * n).map(|i| f(i / n, (i % n) % nx, (i % n) / nx)).collect()
        };
        Grid {
            nx,
            ny,
            nz,
            lat: (0..n).map(|i| 35.0 + (i / nx) as f64 * 0.03).collect(),
            lon: (0..n).map(|i| -98.0 + (i % nx) as f64 * 0.03).collect(),
            v3: vec![
                f3(&|k, x, y| 1000.0 - k as f64 * 80.0 - 0.013 * x as f64 + 0.007 * y as f64),
                f3(&|k, _, _| 300.0 + k as f64 * 800.0),
                f3(&|k, x, _| 25.0 - k as f64 * 6.5 + x as f64 * 0.01),
                f3(&|k, _, y| 18.0 - k as f64 * 8.0 + y as f64 * 0.01),
                f3(&|k, _, _| 2.0 + k as f64 * 3.0),
                f3(&|k, _, _| 5.0 - k as f64),
                f3(&|k, x, y| if (x + y + k) % 7 == 0 { f64::NAN } else { 0.1 * (x as f64 - y as f64) }),
            ],
            v2: (0..6).map(|j| (0..n).map(|i| 900.0 + j as f64 + i as f64 * 1e-3).collect()).collect(),
        }
    }

    fn write_of<'a>(g: &'a Grid, exact: bool) -> ColumnsWrite<'a> {
        ColumnsWrite {
            nx: g.nx,
            ny: g.ny,
            nz: g.nz,
            lat: Values::F64(&g.lat),
            lon: Values::F64(&g.lon),
            fields3: (0..7)
                .map(|i| Field3 {
                    name: FIELDS_3D[i],
                    units: UNITS_3D[i],
                    quantum: if exact { None } else { Some(QUANTA_3D[i]) },
                    data: Values::F64(&g.v3[i]),
                    scale: 1.0,
                    offset: 0.0,
                    affine: None,
                })
                .collect(),
            fields2: (0..6)
                .map(|i| Field2 { name: FIELDS_2D[i + 2], units: UNITS_2D[i + 2], data: Values::F64(&g.v2[i]), scale: 1.0, offset: 0.0 })
                .collect(),
            domain: "d01",
            member: "mem001",
            valid_unix: 1_720_972_800,
            tile: 16,
            meta: json!({"writer": "test"}),
        }
    }

    /// The layout a reader relies on: magic, header, an index whose tiles run back to back to the end of the file,
    /// one more tile than columns per tile row when the grid does not divide.
    #[test]
    fn layout_is_what_readers_read() {
        let g = grid(37, 20, 6);
        let raw = encode(&write_of(&g, false)).unwrap();
        assert_eq!(&raw[..8], MAGIC);
        let f = ColumnFile::from_source(raw.clone()).unwrap();
        assert_eq!((f.header.tiles_x, f.header.tiles_y), (3, 2));
        assert_eq!(f.header.valid, "2024-07-14T16:00:00Z");
        assert_eq!(f.header.schema, SCHEMA);
        assert_eq!(f.header.modes3d, vec!["quantized"; 7]);
        let mut end = f.header.index_offset as usize + 6 * 12;
        for t in 0..6 {
            let (off, len) = f.tile_range(t % 3, t / 3);
            assert_eq!(off as usize, end);
            end += len as usize;
        }
        assert_eq!(end, raw.len());
        // omega NaNs counted per stored value, tiles overlapping by one column
        assert!(f.header.filled > 0);
    }

    /// A file with a surface section: the main tiles, the header's main fields and every sounding read are what
    /// they are without it (an old reader sizes the 2-D block from `fields2d`, unchanged); the surface planes come
    /// back bit for bit, NaN included, through the index and through the suffix read a remote reader makes; the
    /// trailer ends the file. A global grid's surface planes carry the wrap column.
    #[test]
    fn surface_section_is_additive_and_exact() {
        for global in [false, true] {
            let mut g = grid(37, 20, 6);
            if global {
                g = grid(36, 9, 4);
                g.lon = (0..g.nx * g.ny).map(|i| -180.0 + (i % g.nx) as f64 * 10.0).collect();
                g.lat = (0..g.nx * g.ny).map(|i| -40.0 + (i / g.nx) as f64 * 10.0).collect();
            }
            let n = g.nx * g.ny;
            let precip: Vec<f64> = (0..n).map(|i| if i % 13 == 0 { f64::NAN } else { i as f64 * 0.017 }).collect();
            let mslp: Vec<f32> = (0..n).map(|i| 101_000.0 + i as f32).collect();
            let w = write_of(&g, false);
            let plain = encode(&w).unwrap();
            let s = SurfaceWrite {
                lat: w.lat,
                lon: w.lon,
                fields: vec![
                    Field2 { name: "precip_mm", units: "mm", data: Values::F64(&precip), scale: 1.0, offset: 0.0 },
                    Field2 { name: "mslp_hpa", units: "hPa", data: Values::F32(&mslp), scale: 0.01, offset: 0.0 },
                ],
                notes: json!({"precip_mm": "run total"}),
            };
            let raw = finish(encode_main(&w).unwrap(), Some(&s)).unwrap();
            assert_eq!(&raw[raw.len() - 8..], SURFACE_TRAILER);
            let a = ColumnFile::from_source(plain.clone()).unwrap();
            let b = ColumnFile::from_source(raw.clone()).unwrap();
            assert_eq!(a.header.fields2d, b.header.fields2d);
            let (a3, a2) = a.read_full().unwrap();
            let (b3, b2) = b.read_full().unwrap();
            let bits = |v: &Vec<Vec<f32>>| v.iter().flatten().map(|x| x.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&a3), bits(&b3));
            assert_eq!(bits(&a2), bits(&b2));
            let ntiles = a.header.tiles_x * a.header.tiles_y;
            for t in 0..ntiles {
                let (tx, ty) = (t % a.header.tiles_x, t / a.header.tiles_x);
                // the same main tile bytes, so a sounding read is the same size
                let (ra, rb) = (a.tile_range(tx, ty), b.tile_range(tx, ty));
                assert_eq!(ra.1, rb.1);
                assert_eq!(plain[ra.0 as usize..ra.0 as usize + ra.1 as usize], raw[rb.0 as usize..rb.0 as usize + rb.1 as usize]);
                // the suffix read a remote reader makes finds the same entry as the index
                let k = ty * a.header.tiles_x + tx;
                let tail = &raw[raw.len() - (12 * (ntiles - k) + 8)..];
                let off = u64::from_le_bytes(tail[..8].try_into().unwrap());
                let len = u32::from_le_bytes(tail[8..12].try_into().unwrap());
                assert_eq!(Some((off, len)), b.surface_tile_range(tx, ty).unwrap());
            }
            assert!(a.read_full_surface().unwrap().is_none());
            let sf = b.read_full_surface().unwrap().unwrap();
            let sh = b.header.surface.as_ref().unwrap();
            assert_eq!(sh.fields, vec!["lat", "lon", "precip_mm", "mslp_hpa"]);
            let nxo = b.header.nx;
            for y in 0..g.ny {
                for x in 0..nxo {
                    let src = if x == g.nx { 0 } else { x };
                    let want = precip[y * g.nx + src] as f32;
                    assert_eq!(sf[2][y * nxo + x].to_bits(), want.to_bits());
                    assert_eq!(sf[3][y * nxo + x].to_bits(), ((mslp[y * g.nx + src] as f64 * 0.01) as f32).to_bits());
                    assert_eq!(sf[0][y * nxo + x].to_bits(), a2[0][y * nxo + x].to_bits());
                    assert_eq!(sf[1][y * nxo + x].to_bits(), a2[1][y * nxo + x].to_bits());
                }
            }
            assert_eq!(global, b.header.wrap_x.is_some());
        }
    }

    /// Exact fields come back bit for bit (NaN included); quantized fields come back as round(x / q) * q.
    #[test]
    fn exact_is_lossless_and_quantized_is_the_rounding() {
        let g = grid(37, 20, 6);
        for exact in [true, false] {
            let f = ColumnFile::from_source(encode(&write_of(&g, exact)).unwrap()).unwrap();
            let (f3, f2) = f.read_full().unwrap();
            for i in 0..7 {
                for (j, (got, want)) in f3[i].iter().zip(&g.v3[i]).enumerate() {
                    if exact {
                        assert_eq!(got.to_bits(), (*want as f32).to_bits(), "field {i} value {j}");
                    } else if want.is_finite() {
                        let q = ((want / QUANTA_3D[i]).round() * QUANTA_3D[i]) as f32;
                        assert_eq!(got.to_bits(), q.to_bits(), "field {i} value {j}");
                    }
                }
            }
            for i in 0..6 {
                for (got, want) in f2[i + 2].iter().zip(&g.v2[i]) {
                    assert_eq!(got.to_bits(), (*want as f32).to_bits());
                }
            }
            let (lat, _) = f.lat_lon().unwrap();
            assert_eq!(lat[g.nx * 3 + 5], g.lat[g.nx * 3 + 5] as f32);
        }
    }

    /// A point on a tile edge and on the grid's last row and column reads from one tile.
    #[test]
    fn edges_read_from_one_tile() {
        let g = grid(33, 17, 3);
        let f = ColumnFile::from_source(encode(&write_of(&g, true)).unwrap()).unwrap();
        for (fx, fy) in [(16.0, 16.0), (15.5, 3.2), (32.0, 16.0), (31.9, 15.9), (0.0, 0.0)] {
            let t = f.tile_at(fx, fy).unwrap();
            let p = t.profile3(0, fx, fy, g.nx, g.ny);
            assert!(p.iter().all(|v| v.is_finite()), "({fx},{fy})");
        }
    }

    /// A global lat-lon grid (periodic in x) gets one wrap column: column nx repeats column 0 in every field, its
    /// longitude carried past the last column, so the cell between the last and the first longitude exists. A
    /// regional grid gets none.
    #[test]
    fn global_grids_wrap() {
        let mut g = grid(36, 9, 4);
        let dx = 10.0;
        g.lon = (0..g.nx * g.ny).map(|i| -180.0 + (i % g.nx) as f64 * dx).collect();
        g.lat = (0..g.nx * g.ny).map(|i| -40.0 + (i / g.nx) as f64 * dx).collect();
        assert!(periodic_x(g.nx, g.ny, Values::F64(&g.lat), Values::F64(&g.lon)));
        for exact in [true, false] {
            let f = ColumnFile::from_source(encode(&write_of(&g, exact)).unwrap()).unwrap();
            assert_eq!(f.header.nx, g.nx + 1);
            assert_eq!(f.header.wrap_x.as_ref().unwrap()["source_nx"], json!(g.nx));
            let (f3, f2) = f.read_full().unwrap();
            let (lat, lon) = f.lat_lon().unwrap();
            let nxo = g.nx + 1;
            for y in 0..g.ny {
                assert_eq!(lon[y * nxo + g.nx], 180.0);
                assert_eq!(lon[y * nxo + g.nx - 1], 170.0);
                assert_eq!(lat[y * nxo + g.nx], lat[y * nxo]);
                for i in 0..7 {
                    for k in 0..g.nz {
                        let (a, b) = (f3[i][k * nxo * g.ny + y * nxo + g.nx], f3[i][k * nxo * g.ny + y * nxo]);
                        assert!(a.to_bits() == b.to_bits(), "field {i} k {k} y {y}: wrap {a} vs first {b}");
                    }
                }
                for f in &f2[2..] {
                    assert_eq!(f[y * nxo + g.nx].to_bits(), f[y * nxo].to_bits());
                }
                // every source column where it was
                for x in 0..g.nx {
                    assert_eq!(f2[2][y * nxo + x], g.v2[0][y * g.nx + x] as f32);
                }
            }
        }
        // A regional grid, and a global grid that already repeats its first longitude, get no wrap column.
        let r = grid(36, 9, 4);
        assert!(!periodic_x(r.nx, r.ny, Values::F64(&r.lat), Values::F64(&r.lon)));
        let closed: Vec<f64> = (0..37 * 9).map(|i| -180.0 + (i % 37) as f64 * 10.0).collect();
        let lat: Vec<f64> = (0..37 * 9).map(|i| (i / 37) as f64).collect();
        assert!(!periodic_x(37, 9, Values::F64(&lat), Values::F64(&closed)));
    }

    /// A `GPWMCOL1` file laid out as gpuwm 2.9.6's rw_soundings wrote it (one code width for the tile, every
    /// quantized field's codes one shuffled stream, no modes3d or units) reads to round(x / q) * q.
    #[test]
    fn prefold_files_read() {
        let g = grid(37, 20, 6);
        let (nx, ny, nz, tile) = (g.nx, g.ny, g.nz, 16usize);
        let n = nx * ny;
        let (tiles_x, tiles_y) = ((nx - 1).div_ceil(tile), (ny - 1).div_ceil(tile));
        let mut blobs = vec![];
        for t in 0..tiles_x * tiles_y {
            let (x0, x1) = span(t % tiles_x, tile, nx);
            let (y0, y1) = span(t / tiles_x, tile, ny);
            let plane = (x1 - x0 + 1) * (y1 - y0 + 1);
            let mut codes: Vec<u32> = vec![];
            for (i, q) in QUANTA_3D.iter().enumerate() {
                let mut last = vec![0i64; plane];
                for k in 0..nz {
                    for y in y0..=y1 {
                        for x in x0..=x1 {
                            let c = (y - y0) * (x1 - x0 + 1) + (x - x0);
                            let v = g.v3[i][k * n + y * nx + x] / q;
                            let qv = if v.is_finite() { v.round() as i64 } else { last[c] };
                            codes.push(zigzag((qv - last[c]) as i32));
                            last[c] = qv;
                        }
                    }
                }
            }
            let mut floats: Vec<u32> = vec![];
            for f in [&g.lat, &g.lon].into_iter().chain(g.v2.iter()) {
                for y in y0..=y1 {
                    for x in x0..=x1 {
                        floats.push((f[y * nx + x] as f32).to_bits());
                    }
                }
            }
            let width = if codes.iter().all(|c| *c < 65536) { 2 } else { 4 };
            let mut payload = vec![0u8; 1 + codes.len() * width + floats.len() * 4];
            payload[0] = width as u8;
            let nc = codes.len();
            shuffle_into(&mut payload[1..1 + nc * width], &codes, width);
            shuffle_into(&mut payload[1 + nc * width..], &floats, 4);
            let mut enc = ZlibEncoder::new(vec![], Compression::new(5));
            enc.write_all(&payload).unwrap();
            blobs.push(enc.finish().unwrap());
        }
        let mut header = json!({
            "schema": SCHEMA_PREFOLD, "domain": "d01", "member": "det", "valid": "2024-07-14T16:00:00Z",
            "nx": nx, "ny": ny, "nz": nz, "tile": tile, "tiles_x": tiles_x, "tiles_y": tiles_y,
            "fields3d": FIELDS_3D, "fields2d": FIELDS_2D, "quanta3d": QUANTA_3D, "dtype2d": "f32",
            "codec": "shuffle-zlib", "filled": 0, "index_offset": 0,
            "locator": {"step": tile, "ix": [0, 16, 32, 36], "iy": [0, 16, 19], "lat": vec![0.0; 12], "lon": vec![0.0; 12]},
        });
        let probe = serde_json::to_vec(&header).unwrap();
        header["index_offset"] = json!(12 + probe.len() + 24);
        let mut head = serde_json::to_vec(&header).unwrap();
        head.resize(probe.len() + 24, b' ');
        let mut raw = MAGIC_PREFOLD.to_vec();
        raw.extend_from_slice(&(head.len() as u32).to_le_bytes());
        raw.extend_from_slice(&head);
        let mut off = (raw.len() + blobs.len() * 12) as u64;
        for b in &blobs {
            raw.extend_from_slice(&off.to_le_bytes());
            raw.extend_from_slice(&(b.len() as u32).to_le_bytes());
            off += b.len() as u64;
        }
        for b in &blobs {
            raw.extend_from_slice(b);
        }
        let f = ColumnFile::from_source(raw).unwrap();
        assert!(f.header.prefold());
        let (f3, f2) = f.read_full().unwrap();
        for i in 0..7 {
            for (got, want) in f3[i].iter().zip(&g.v3[i]) {
                if want.is_finite() {
                    let q = ((want / QUANTA_3D[i]).round() * QUANTA_3D[i]) as f32;
                    assert_eq!(got.to_bits(), q.to_bits(), "field {i}");
                }
            }
        }
        assert_eq!(f2[2][7].to_bits(), (g.v2[0][7] as f32).to_bits());
    }

    /// A v1 hour written by the v1 writer and transcoded: every 3-D and 2-D value of the v2 file is the v1 reader's
    /// bit pattern (missing values, a constant chunk and an all-missing chunk included), and the v2 file is not
    /// larger than the v1 file.
    #[test]
    fn transcode_v1_is_bitwise_and_no_larger() {
        use crate::writer::HourWriter;
        let (nx, ny, nz) = (45usize, 38usize, 9usize);
        let n = nx * ny;
        let dir = std::env::temp_dir().join(format!("rw-columns-transcode-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let lat: Vec<f32> = (0..n).map(|i| 35.0 + (i / nx) as f32 * 0.03).collect();
        let lon: Vec<f32> = (0..n).map(|i| -98.0 + (i % nx) as f32 * 0.03 + (i / nx) as f32 * 0.001).collect();
        let grid =
            rustwx_core::LatLonGrid::new(rustwx_core::GridShape::new(nx, ny).unwrap(), lat.clone(), lon.clone()).unwrap();
        let gpath = dir.join("grid.rwg");
        let hash = crate::grid::write_grid(&gpath, &grid, None).unwrap();
        let mut w = HourWriter::new("wrf", "run", 3, nx, ny, &hash, "test");
        let levels: Vec<u16> = (0..nz).map(|k| 1000 - k as u16 * 50).collect();
        let vol = |f: &dyn Fn(usize, usize, usize) -> f32| -> Vec<Vec<f32>> {
            (0..nz).map(|k| (0..n).map(|i| f(k, i % nx, i / nx)).collect()).collect()
        };
        let vols = [
            ("p", vol(&|k, x, y| 1000.0 - k as f32 * 50.0 + (x as f32 * 0.37).sin() + y as f32 * 0.011)),
            ("t", vol(&|k, x, y| 290.0 - k as f32 * 6.0 + (x * y) as f32 * 1e-3)),
            // NaNs scattered, a constant chunk (x, y < 16) and an all-missing chunk (x >= 32, y >= 32)
            ("w", vol(&|k, x, y| {
                if x >= 32 && y >= 32 {
                    f32::NAN
                } else if x < 16 && y < 16 {
                    4.25
                } else if (x + 2 * y + k) % 11 == 0 {
                    f32::NAN
                } else {
                    0.3 * x as f32 - 0.2 * y as f32 + k as f32
                }
            })),
        ];
        for (name, planes) in &vols {
            let refs: Vec<&[f32]> = planes.iter().map(|p| p.as_slice()).collect();
            w.add_pressure3d(name, "u", json!({}), &levels, &refs).unwrap();
        }
        let sfc: Vec<f32> = (0..n).map(|i| 1013.25 - i as f32 * 1e-3).collect();
        w.add_surface2d("psfc", "hPa", json!({}), &sfc).unwrap();
        let hpath = dir.join("f003.rws");
        w.finish(&hpath).unwrap();
        let out = dir.join("f003.v2.rws");
        let bytes = transcode_v1(&hpath, &gpath, &out, "d01", "mem001", 16).unwrap();
        let r = HourReader::open(&hpath).unwrap();
        let f = ColumnFile::open(&out).unwrap();
        assert_eq!(f.header.modes3d, vec!["affine"; 3]);
        let (f3, f2) = f.read_full().unwrap();
        for (i, (name, _)) in vols.iter().enumerate() {
            let want = r.read_full_3d(name).unwrap();
            let differ = want.iter().zip(&f3[i]).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
            assert_eq!(differ, 0, "{name}");
        }
        let want = r.read_full_2d("psfc").unwrap();
        assert!(want.iter().zip(&f2[f.header.field2("psfc").unwrap()]).all(|(a, b)| a.to_bits() == b.to_bits()));
        let (la, lo) = f.lat_lon().unwrap();
        assert!(la.iter().zip(&lat).chain(lo.iter().zip(&lon)).all(|(a, b)| a.to_bits() == b.to_bits()));
        let v1 = std::fs::metadata(&hpath).unwrap().len();
        assert!(bytes <= v1, "v2 {bytes} bytes vs v1 {v1}");
        std::fs::remove_dir_all(&dir).ok();
    }
}
