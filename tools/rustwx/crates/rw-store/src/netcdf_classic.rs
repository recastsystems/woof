//! Dependency-free NetCDF classic writer covering CDF-1, CDF-2 and CDF-5.
//!
//! [`crate::netcdf3`] emits the narrow slice the hour exporter needs: CDF-2,
//! fixed dimensions, `NC_FLOAT` only, no record dimension.  A wrfout frame
//! needs more than that: an unlimited `Time` dimension, an `NC_CHAR`
//! `Times(Time, DateStrLen)` stamp, `NC_INT` bookkeeping variables such as
//! `ITIMESTEP`, and (for a whole-mesh initial condition) 64-bit data offsets.
//! This module is that writer and shares its header serializer with
//! [`crate::netcdf3`].
//!
//! Still pure Rust and still no HDF5: the classic formats are big-endian byte
//! containers written straight against the Unidata "classic format"
//! specification.  Every width below was confirmed byte-for-byte against
//! files the netCDF-C library wrote (see the golden tests at the bottom of
//! this file), because the published grammar leaves room to misread which
//! fields widen in CDF-5 and which do not.
//!
//! ## Byte layout (everything BIG-ENDIAN)
//! ```text
//! header   := magic numrecs dim_list gatt_list var_list
//! magic    := 'C' 'D' 'F' VERSION      // 0x01 CDF-1, 0x02 CDF-2, 0x05 CDF-5
//! numrecs  := NON_NEG                  // count of records actually written
//! dim_list := ABSENT | NC_DIMENSION(0x0A) nelems [dim ...]
//!   dim    := name dim_length          // dim_length 0 marks the record dim
//! att_list := ABSENT | NC_ATTRIBUTE(0x0C) nelems [attr ...]
//!   attr   := name nc_type(u32) nelems [values, zero-padded to 4]
//! var_list := ABSENT | NC_VARIABLE(0x0B) nelems [var ...]
//!   var    := name ndims [dimid ...] vatt_list nc_type(u32) vsize begin
//! ABSENT   := 0x0000_0000 nelems(=0)
//! name     := nelems(byte length, NOT counting padding) bytes pad-to-4
//! ```
//!
//! ### What widens in CDF-5
//! `NON_NEG`: that is `numrecs`, every list `nelems`, every name length,
//! `dim_length`, every `dimid`, each attribute's value count, and `vsize`:
//! is a 4-byte field in CDF-1/CDF-2 and an 8-byte field in CDF-5.  `begin`
//! (`OFFSET`) is 4 bytes in CDF-1 and 8 bytes in CDF-2 and CDF-5.  The
//! component tags (`NC_DIMENSION`/`NC_ATTRIBUTE`/`NC_VARIABLE`, the leading
//! word of `ABSENT`) and `nc_type` stay 4 bytes in all three.  A CDF-5
//! `ABSENT` is therefore 12 bytes, not 8: the tag stays narrow while its
//! `nelems` widens.
//!
//! ### Data section
//! Fixed-size variables come first, in definition order, each at its own
//! `begin`.  The record section follows: record `r` of every record variable
//! sits contiguously, in definition order, and the whole group repeats.
//! `vsize` for a record variable is one record's slab: the product of its
//! non-record dimension lengths times the type size, rounded up to a multiple
//! of 4.  The one exception the specification carves out is honoured here:
//! when a file has exactly one record variable and that variable is
//! `NC_CHAR`, `NC_BYTE` or `NC_SHORT`, its slabs are *not* padded.
//!
//! ## Write discipline
//! The record and fixed sections interleave, so this writer seeks rather than
//! streams: [`NcClassicWriter::create`] lays out the whole file from the
//! schema and the declared record count, then each
//! [`put`](NcClassicWriter::put) /
//! [`put_record`](NcClassicWriter::put_record) call writes one slab at its
//! computed offset in any order the caller finds convenient.
//! [`finish`](NcClassicWriter::finish) refuses unless every declared slab was
//! written (a half-filled frame is a caller bug, never a file to ship) then
//! flushes and `fsync`s.  As with [`crate::netcdf3`] the destination is
//! written in place; on failure the partial file is the caller's to discard.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{File, OpenOptions};
use std::io::{Seek, SeekFrom, Write};
use std::path::Path;

use crate::error::{RwResult, RwStoreError};

// Component tags from the classic format grammar.
const NC_DIMENSION: u32 = 0x0000_000A;
const NC_VARIABLE: u32 = 0x0000_000B;
const NC_ATTRIBUTE: u32 = 0x0000_000C;
const ABSENT_TAG: u32 = 0x0000_0000;

/// Which of the three classic on-disk formats to emit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NcFormat {
    /// CDF-1: 32-bit offsets. Whole file must stay under ~2 GiB.
    Classic,
    /// CDF-2: 64-bit offsets, 32-bit sizes. What [`crate::netcdf3`] writes.
    Offset64,
    /// CDF-5: 64-bit offsets *and* 64-bit sizes. What the MPAS initial
    /// conditions use, and the only classic format that carries a variable
    /// larger than 4 GiB.
    Data64,
}

impl NcFormat {
    fn version_byte(self) -> u8 {
        match self {
            NcFormat::Classic => 1,
            NcFormat::Offset64 => 2,
            NcFormat::Data64 => 5,
        }
    }

    /// Byte width of a `NON_NEG` field (counts, lengths, dimids, `vsize`).
    fn nonneg_width(self) -> usize {
        match self {
            NcFormat::Classic | NcFormat::Offset64 => 4,
            NcFormat::Data64 => 8,
        }
    }

    /// Byte width of an `OFFSET` field (a variable's `begin`).
    fn offset_width(self) -> usize {
        match self {
            NcFormat::Classic => 4,
            NcFormat::Offset64 | NcFormat::Data64 => 8,
        }
    }

    /// Largest byte offset this format's `begin` field can name.
    fn max_offset(self) -> u64 {
        match self {
            NcFormat::Classic => u32::MAX as u64,
            NcFormat::Offset64 | NcFormat::Data64 => u64::MAX,
        }
    }

    /// Human name used in refusal messages.
    fn label(self) -> &'static str {
        match self {
            NcFormat::Classic => "CDF-1",
            NcFormat::Offset64 => "CDF-2",
            NcFormat::Data64 => "CDF-5",
        }
    }
}

/// External data type of a variable or attribute value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NcType {
    /// `NC_CHAR`: 8-bit characters, 1 byte each.
    Char,
    /// `NC_SHORT`: 16-bit signed.
    Short,
    /// `NC_INT`: 32-bit signed.
    Int,
    /// `NC_FLOAT`: IEEE single precision.
    Float,
    /// `NC_DOUBLE`: IEEE double precision.
    Double,
}

impl NcType {
    fn tag(self) -> u32 {
        match self {
            NcType::Char => 2,
            NcType::Short => 3,
            NcType::Int => 4,
            NcType::Float => 5,
            NcType::Double => 6,
        }
    }

    fn size(self) -> u64 {
        match self {
            NcType::Char => 1,
            NcType::Short => 2,
            NcType::Int => 4,
            NcType::Float => 4,
            NcType::Double => 8,
        }
    }

    fn name(self) -> &'static str {
        match self {
            NcType::Char => "NC_CHAR",
            NcType::Short => "NC_SHORT",
            NcType::Int => "NC_INT",
            NcType::Float => "NC_FLOAT",
            NcType::Double => "NC_DOUBLE",
        }
    }

    /// The specification's "single record variable" no-padding exception
    /// applies only to these types.
    fn unpadded_when_sole_record_var(self) -> bool {
        matches!(self, NcType::Char | NcType::Short)
    }
}

/// An attribute value. The variant chooses the emitted `nc_type`.
#[derive(Debug, Clone, PartialEq)]
pub enum NcAttrValue {
    /// `NC_CHAR`. Emitted as raw UTF-8 bytes with no trailing NUL.
    Text(String),
    /// `NC_INT`.
    Ints(Vec<i32>),
    /// `NC_FLOAT`.
    Floats(Vec<f32>),
    /// `NC_DOUBLE`.
    Doubles(Vec<f64>),
}

impl NcAttrValue {
    fn nc_type(&self) -> NcType {
        match self {
            NcAttrValue::Text(_) => NcType::Char,
            NcAttrValue::Ints(_) => NcType::Int,
            NcAttrValue::Floats(_) => NcType::Float,
            NcAttrValue::Doubles(_) => NcType::Double,
        }
    }

    /// Element count as the `nelems` field reports it: bytes for text,
    /// values for everything else.
    fn nelems(&self) -> u64 {
        match self {
            NcAttrValue::Text(s) => s.len() as u64,
            NcAttrValue::Ints(v) => v.len() as u64,
            NcAttrValue::Floats(v) => v.len() as u64,
            NcAttrValue::Doubles(v) => v.len() as u64,
        }
    }
}

/// A single named attribute (global, or attached to a variable).
#[derive(Debug, Clone, PartialEq)]
pub struct NcAttr {
    pub name: String,
    pub value: NcAttrValue,
}

impl NcAttr {
    pub fn text(name: impl Into<String>, value: impl Into<String>) -> Self {
        NcAttr {
            name: name.into(),
            value: NcAttrValue::Text(value.into()),
        }
    }

    pub fn int(name: impl Into<String>, value: i32) -> Self {
        NcAttr {
            name: name.into(),
            value: NcAttrValue::Ints(vec![value]),
        }
    }

    pub fn float(name: impl Into<String>, value: f32) -> Self {
        NcAttr {
            name: name.into(),
            value: NcAttrValue::Floats(vec![value]),
        }
    }

    pub fn floats(name: impl Into<String>, value: Vec<f32>) -> Self {
        NcAttr {
            name: name.into(),
            value: NcAttrValue::Floats(value),
        }
    }

    pub fn doubles(name: impl Into<String>, value: Vec<f64>) -> Self {
        NcAttr {
            name: name.into(),
            value: NcAttrValue::Doubles(value),
        }
    }
}

/// A dimension. Its index in the `dims` vector passed to
/// [`NcClassicWriter::create`] is its dimid. At most one may be `unlimited`,
/// and the format requires it to be the first dimension of any variable that
/// uses it.
#[derive(Debug, Clone)]
pub struct NcDim {
    pub name: String,
    /// Length for a fixed dimension; ignored when `unlimited` is set.
    pub len: usize,
    pub unlimited: bool,
}

impl NcDim {
    pub fn fixed(name: impl Into<String>, len: usize) -> Self {
        NcDim {
            name: name.into(),
            len,
            unlimited: false,
        }
    }

    pub fn record(name: impl Into<String>) -> Self {
        NcDim {
            name: name.into(),
            len: 0,
            unlimited: true,
        }
    }
}

/// A variable definition. `dimids` indexes into the `dims` vector, row-major
/// (the last dimid varies fastest in the data).
#[derive(Debug, Clone)]
pub struct NcVarDef {
    pub name: String,
    pub ty: NcType,
    pub dimids: Vec<usize>,
    pub attrs: Vec<NcAttr>,
}

impl NcVarDef {
    pub fn new(name: impl Into<String>, ty: NcType, dimids: Vec<usize>) -> Self {
        NcVarDef {
            name: name.into(),
            ty,
            dimids,
            attrs: Vec::new(),
        }
    }

    pub fn with_attrs(mut self, attrs: Vec<NcAttr>) -> Self {
        self.attrs = attrs;
        self
    }
}

/// One slab of variable data handed to the writer. The variant must match the
/// variable's declared [`NcType`]; a mismatch is refused, never coerced.
#[derive(Debug, Clone, Copy)]
pub enum NcData<'a> {
    /// `NC_CHAR` payload as raw bytes. Short slabs are NUL-filled to the
    /// declared length, which is how a 19-byte timestamp lands in a
    /// `Times(Time, DateStrLen=19)` slot.
    Chars(&'a [u8]),
    Shorts(&'a [i16]),
    Ints(&'a [i32]),
    Floats(&'a [f32]),
    Doubles(&'a [f64]),
}

impl NcData<'_> {
    fn nc_type(&self) -> NcType {
        match self {
            NcData::Chars(_) => NcType::Char,
            NcData::Shorts(_) => NcType::Short,
            NcData::Ints(_) => NcType::Int,
            NcData::Floats(_) => NcType::Float,
            NcData::Doubles(_) => NcType::Double,
        }
    }

    fn len(&self) -> usize {
        match self {
            NcData::Chars(v) => v.len(),
            NcData::Shorts(v) => v.len(),
            NcData::Ints(v) => v.len(),
            NcData::Floats(v) => v.len(),
            NcData::Doubles(v) => v.len(),
        }
    }

    /// Big-endian encode into `buf`, which the caller sized and zeroed.
    fn encode_into(&self, buf: &mut Vec<u8>) {
        match self {
            NcData::Chars(v) => buf.extend_from_slice(v),
            NcData::Shorts(v) => {
                for &x in *v {
                    buf.extend_from_slice(&x.to_be_bytes());
                }
            }
            NcData::Ints(v) => {
                for &x in *v {
                    buf.extend_from_slice(&x.to_be_bytes());
                }
            }
            NcData::Floats(v) => {
                for &x in *v {
                    buf.extend_from_slice(&x.to_be_bytes());
                }
            }
            NcData::Doubles(v) => {
                for &x in *v {
                    buf.extend_from_slice(&x.to_be_bytes());
                }
            }
        }
    }
}

impl NcData<'_> {
    /// The same big-endian bytes as `encode_into`, written by rayon workers in
    /// fixed-size chunks into `out`, which the caller sized and zeroed.
    fn encode_par(&self, out: &mut [u8]) {
        use rayon::prelude::*;
        const CHUNK: usize = 1 << 16;
        macro_rules! be {
            ($v:expr, $w:expr) => {
                out[..$v.len() * $w]
                    .par_chunks_mut(CHUNK * $w)
                    .zip($v.par_chunks(CHUNK))
                    .for_each(|(dst, src)| {
                        for (d, x) in dst.chunks_exact_mut($w).zip(src) {
                            d.copy_from_slice(&x.to_be_bytes());
                        }
                    })
            };
        }
        match self {
            NcData::Chars(v) => out[..v.len()].copy_from_slice(v),
            NcData::Shorts(v) => be!(v, 2),
            NcData::Ints(v) => be!(v, 4),
            NcData::Floats(v) => be!(v, 4),
            NcData::Doubles(v) => be!(v, 8),
        }
    }
}

/// An owned slab, for producers that build their values on a worker.
#[derive(Debug, Clone)]
pub enum NcOwned {
    Chars(Vec<u8>),
    Shorts(Vec<i16>),
    Ints(Vec<i32>),
    Floats(Vec<f32>),
    Doubles(Vec<f64>),
}

impl NcOwned {
    pub fn as_data(&self) -> NcData<'_> {
        match self {
            NcOwned::Chars(v) => NcData::Chars(v),
            NcOwned::Shorts(v) => NcData::Shorts(v),
            NcOwned::Ints(v) => NcData::Ints(v),
            NcOwned::Floats(v) => NcData::Floats(v),
            NcOwned::Doubles(v) => NcData::Doubles(v),
        }
    }
}

/// What a [`NcClassicWriter::put_parallel`] producer hands back: data the
/// caller already holds, or data the worker built.
pub enum NcSlab<'a> {
    Borrowed(NcData<'a>),
    Owned(NcOwned),
}

impl NcSlab<'_> {
    fn data(&self) -> NcData<'_> {
        match self {
            NcSlab::Borrowed(d) => *d,
            NcSlab::Owned(o) => o.as_data(),
        }
    }
}

/// Positional write: no shared cursor, so workers can write disjoint slabs at once.
#[cfg(unix)]
fn write_all_at(file: &File, buf: &[u8], offset: u64) -> std::io::Result<()> {
    std::os::unix::fs::FileExt::write_all_at(file, buf, offset)
}

#[cfg(windows)]
fn write_all_at(file: &File, mut buf: &[u8], mut offset: u64) -> std::io::Result<()> {
    use std::os::windows::fs::FileExt;
    while !buf.is_empty() {
        let n = file.seek_write(buf, offset)?;
        if n == 0 {
            return Err(std::io::ErrorKind::WriteZero.into());
        }
        buf = &buf[n..];
        offset += n as u64;
    }
    Ok(())
}

/// The type and length checks every slab passes before it is written.
fn check_slab(layout: &VarLayout, data: &NcData<'_>) -> RwResult<()> {
    if data.nc_type() != layout.ty {
        return Err(RwStoreError::Format(format!(
            "netcdf_classic: variable '{}' is {} but was handed {} data",
            layout.name,
            layout.ty.name(),
            data.nc_type().name()
        )));
    }
    if data.len() as u64 != layout.slab_elems {
        return Err(RwStoreError::Format(format!(
            "netcdf_classic: variable '{}' expects {} value(s) per slab but got {}",
            layout.name,
            layout.slab_elems,
            data.len()
        )));
    }
    Ok(())
}

/// Round `n` up to the next multiple of 4.
#[inline]
fn pad4(n: u64) -> u64 {
    (n + 3) & !3
}

/// Per-variable layout the writer resolves once at `create` time.
#[derive(Debug, Clone)]
struct VarLayout {
    name: String,
    ty: NcType,
    /// True when the variable's first dimension is the record dimension.
    is_record: bool,
    /// Elements in one slab: the whole array for a fixed variable, one
    /// record's worth for a record variable.
    slab_elems: u64,
    /// Unpadded byte length of one slab.
    slab_bytes: u64,
    /// Byte length one slab occupies in the file, padding included.
    slab_stride: u64,
    /// File offset of the variable's data (record 0 for a record variable).
    begin: u64,
}

/// Seeking classic-format writer. Create it with the full schema and the
/// record count, push each slab, then [`finish`](NcClassicWriter::finish).
#[derive(Debug)]
pub struct NcClassicWriter {
    file: File,
    format: NcFormat,
    layouts: Vec<VarLayout>,
    by_name: BTreeMap<String, usize>,
    /// Bytes between record `r` and record `r + 1` of the same variable.
    record_stride: u64,
    num_records: u64,
    /// One flag per slab: `written[i]` has 1 entry for a fixed variable and
    /// `num_records` entries for a record variable.
    written: Vec<Vec<bool>>,
    /// Total file length, so `finish` can size the file even when the last
    /// slab written is not the last slab in the layout.
    file_len: u64,
}

impl NcClassicWriter {
    /// Lay out and open a classic-format file.
    ///
    /// `num_records` is the number of records the file will carry; it is
    /// fixed at creation because the header records it and the data section
    /// is laid out around it. A file with no record dimension takes 0.
    pub fn create(
        path: &Path,
        format: NcFormat,
        dims: Vec<NcDim>,
        gattrs: Vec<NcAttr>,
        vars: Vec<NcVarDef>,
        num_records: u64,
    ) -> RwResult<Self> {
        Self::create_with_min_header(path, format, dims, gattrs, vars, num_records, 0)
    }

    /// [`create`](NcClassicWriter::create), with the data section starting no
    /// earlier than `min_header_bytes`.
    ///
    /// A header shorter than the floor is zero-padded up to it, which is the
    /// classic format's own reserved-header-space convention (`h_minfree`);
    /// readers never parse the pad because every variable's `begin` points
    /// past it.  This exists so a file can be laid out with the same data
    /// offsets as one written by a producer that reserves header space
    /// (PIO/SMIOL pads MPAS headers this way) making the data sections of the
    /// two files comparable byte-for-byte at equal offsets.  A floor of 0 is
    /// exactly [`create`](NcClassicWriter::create).
    #[allow(clippy::too_many_arguments)]
    pub fn create_with_min_header(
        path: &Path,
        format: NcFormat,
        dims: Vec<NcDim>,
        gattrs: Vec<NcAttr>,
        vars: Vec<NcVarDef>,
        num_records: u64,
        min_header_bytes: u64,
    ) -> RwResult<Self> {
        let record_dimid = validate_defs(&dims, &gattrs, &vars, num_records)?;

        // Slab geometry, before offsets are known.
        let mut layouts: Vec<VarLayout> = Vec::with_capacity(vars.len());
        for var in &vars {
            let is_record = record_dimid.is_some_and(|rd| var.dimids.first() == Some(&rd));
            let mut elems: u64 = 1;
            for (axis, &dimid) in var.dimids.iter().enumerate() {
                if is_record && axis == 0 {
                    continue;
                }
                let len = dims[dimid].len as u64;
                elems = elems.checked_mul(len).ok_or_else(|| {
                    RwStoreError::Format(format!(
                        "netcdf_classic: variable '{}' slab element count overflows u64",
                        var.name
                    ))
                })?;
            }
            let slab_bytes = elems.checked_mul(var.ty.size()).ok_or_else(|| {
                RwStoreError::Format(format!(
                    "netcdf_classic: variable '{}' slab byte size overflows u64",
                    var.name
                ))
            })?;
            layouts.push(VarLayout {
                name: var.name.clone(),
                ty: var.ty,
                is_record,
                slab_elems: elems,
                slab_bytes,
                slab_stride: pad4(slab_bytes),
                begin: 0,
            });
        }

        // The specification's single-record-variable exception: one record
        // variable of a sub-word type means its slabs are not padded.
        let record_var_count = layouts.iter().filter(|l| l.is_record).count();
        if record_var_count == 1 {
            if let Some(l) = layouts.iter_mut().find(|l| l.is_record) {
                if l.ty.unpadded_when_sole_record_var() {
                    l.slab_stride = l.slab_bytes;
                }
            }
        }

        // CDF-1 and CDF-2 report vsize in a 32-bit field. A slab that does not
        // fit is a real limit of the chosen format, not a field to clamp:
        // refuse and name the format that would carry it.
        if format.nonneg_width() == 4 {
            if let Some(l) = layouts.iter().find(|l| l.slab_stride > u32::MAX as u64) {
                return Err(RwStoreError::Format(format!(
                    "netcdf_classic: variable '{}' needs {} bytes per slab, past what {}'s 32-bit vsize carries; write CDF-5 instead",
                    l.name,
                    l.slab_stride,
                    format.label()
                )));
            }
        }

        // Two-pass header serialization. Pass 1 measures with placeholder
        // begins; every width is fixed, so pass 2 has the same length.
        let slab_strides: Vec<u64> = layouts.iter().map(|layout| layout.slab_stride).collect();
        let placeholder: Vec<u64> = vec![0; layouts.len()];
        let header_pass1 =
            serialize_header(format, &dims, &gattrs, &vars, &slab_strides, &placeholder, 0);
        let header_len = (header_pass1.len() as u64).max(min_header_bytes);

        // Fixed variables first, in definition order, then the record section.
        let mut cursor = header_len;
        for layout in layouts.iter_mut().filter(|l| !l.is_record) {
            layout.begin = cursor;
            cursor = cursor.checked_add(layout.slab_stride).ok_or_else(|| {
                RwStoreError::Format("netcdf_classic: data section size overflows u64".to_string())
            })?;
        }
        let record_origin = cursor;
        let mut record_stride: u64 = 0;
        for layout in layouts.iter_mut().filter(|l| l.is_record) {
            layout.begin = record_origin + record_stride;
            record_stride = record_stride.checked_add(layout.slab_stride).ok_or_else(|| {
                RwStoreError::Format("netcdf_classic: record size overflows u64".to_string())
            })?;
        }
        let file_len = record_origin
            .checked_add(record_stride.saturating_mul(num_records))
            .ok_or_else(|| {
                RwStoreError::Format("netcdf_classic: file length overflows u64".to_string())
            })?;

        // Every `begin` must be nameable in this format's offset field. The
        // last byte of the file is the binding constraint, so check that.
        let last_byte = file_len.saturating_sub(1);
        if last_byte > format.max_offset() {
            return Err(RwStoreError::Format(format!(
                "netcdf_classic: file would reach offset {last_byte}, past what {}'s offset field carries; write CDF-2 or CDF-5 instead",
                format.label()
            )));
        }

        let begins: Vec<u64> = layouts.iter().map(|l| l.begin).collect();
        let header_pass2 = serialize_header(
            format,
            &dims,
            &gattrs,
            &vars,
            &slab_strides,
            &begins,
            num_records,
        );
        debug_assert_eq!(
            header_pass2.len(),
            header_pass1.len(),
            "netcdf_classic: header length changed between passes"
        );
        debug_assert!(
            header_pass2.len() as u64 <= header_len,
            "netcdf_classic: serialized header overran its padded floor"
        );

        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(path)?;
        file.write_all(&header_pass2)?;
        // Size the file up front so unwritten padding is real zero bytes and
        // every seek lands inside the file.
        file.set_len(file_len)?;

        let mut by_name = BTreeMap::new();
        for (i, layout) in layouts.iter().enumerate() {
            by_name.insert(layout.name.clone(), i);
        }
        let written = layouts
            .iter()
            .map(|l| {
                if l.is_record {
                    vec![false; num_records as usize]
                } else {
                    vec![false]
                }
            })
            .collect();

        Ok(NcClassicWriter {
            file,
            format,
            layouts,
            by_name,
            record_stride,
            num_records,
            written,
            file_len,
        })
    }

    /// The on-disk format this writer is emitting.
    pub fn format(&self) -> NcFormat {
        self.format
    }

    /// Total byte length of the file being written.
    pub fn file_len(&self) -> u64 {
        self.file_len
    }

    /// Write a fixed-size variable's whole array.
    pub fn put(&mut self, name: &str, data: NcData<'_>) -> RwResult<()> {
        let idx = self.index_of(name)?;
        if self.layouts[idx].is_record {
            return Err(RwStoreError::Format(format!(
                "netcdf_classic: '{name}' is a record variable; use put_record with a record index"
            )));
        }
        self.write_slab(idx, 0, data)
    }

    /// Write one record's slab of a record variable.
    pub fn put_record(&mut self, name: &str, record: u64, data: NcData<'_>) -> RwResult<()> {
        let idx = self.index_of(name)?;
        if !self.layouts[idx].is_record {
            return Err(RwStoreError::Format(format!(
                "netcdf_classic: '{name}' is a fixed-size variable; use put"
            )));
        }
        if record >= self.num_records {
            return Err(RwStoreError::Format(format!(
                "netcdf_classic: '{name}' record {record} is past the {} record(s) this file declares",
                self.num_records
            )));
        }
        self.write_slab(idx, record, data)
    }

    /// Convenience for the common `Times(Time, DateStrLen)` stamp: writes the
    /// text NUL-filled to the variable's declared width, refusing a stamp
    /// that does not fit rather than truncating it.
    pub fn put_record_text(&mut self, name: &str, record: u64, text: &str) -> RwResult<()> {
        let idx = self.index_of(name)?;
        let width = self.layouts[idx].slab_elems as usize;
        let bytes = text.as_bytes();
        if bytes.len() > width {
            return Err(RwStoreError::Format(format!(
                "netcdf_classic: '{name}' holds {width} character(s) per record but '{text}' is {} byte(s)",
                bytes.len()
            )));
        }
        let mut padded = vec![0u8; width];
        padded[..bytes.len()].copy_from_slice(bytes);
        self.put_record(name, record, NcData::Chars(&padded))
    }

    fn index_of(&self, name: &str) -> RwResult<usize> {
        self.by_name.get(name).copied().ok_or_else(|| {
            RwStoreError::Format(format!("netcdf_classic: no variable named '{name}'"))
        })
    }

    /// Write many slabs at once on rayon workers.
    ///
    /// `jobs[i]` names a variable and, for a record variable, its record;
    /// `produce(i)` builds that slab on a worker, which encodes it and writes
    /// it at the slab's own offset.  Every slab has a fixed offset and the
    /// file was sized at create, so the file holds the same bytes as calling
    /// `put`/`put_record` for each job in turn.  Names, records and repeats
    /// are checked before any worker starts.
    pub fn put_parallel<'a, F>(&mut self, jobs: &[(&str, Option<u64>)], produce: F) -> RwResult<()>
    where
        F: Fn(usize) -> RwResult<NcSlab<'a>> + Sync,
    {
        use rayon::prelude::*;
        let mut targets = Vec::with_capacity(jobs.len());
        let mut seen = BTreeSet::new();
        for &(name, record) in jobs {
            let idx = self.index_of(name)?;
            let record = match (self.layouts[idx].is_record, record) {
                (true, Some(r)) if r < self.num_records => r,
                (true, Some(r)) => {
                    return Err(RwStoreError::Format(format!(
                        "netcdf_classic: '{name}' record {r} is past the {} record(s) this file declares",
                        self.num_records
                    )))
                }
                (true, None) => {
                    return Err(RwStoreError::Format(format!(
                        "netcdf_classic: '{name}' is a record variable; use put_record with a record index"
                    )))
                }
                (false, None) => 0,
                (false, Some(_)) => {
                    return Err(RwStoreError::Format(format!(
                        "netcdf_classic: '{name}' is a fixed-size variable; use put"
                    )))
                }
            };
            if self.written[idx][record as usize] || !seen.insert((idx, record)) {
                return Err(RwStoreError::Format(format!(
                    "netcdf_classic: variable '{name}' record {record} was already written"
                )));
            }
            targets.push((idx, record));
        }
        let file = &self.file;
        let layouts = &self.layouts;
        let record_stride = self.record_stride;
        targets
            .par_iter()
            .enumerate()
            .try_for_each(|(i, &(idx, record))| -> RwResult<()> {
                let slab = produce(i)?;
                let data = slab.data();
                let layout = &layouts[idx];
                check_slab(layout, &data)?;
                let mut buf = vec![0u8; layout.slab_stride as usize];
                data.encode_par(&mut buf);
                write_all_at(file, &buf, layout.begin + record * record_stride)?;
                Ok(())
            })?;
        for (idx, record) in targets {
            self.written[idx][record as usize] = true;
        }
        Ok(())
    }

    fn write_slab(&mut self, idx: usize, record: u64, data: NcData<'_>) -> RwResult<()> {
        let layout = &self.layouts[idx];
        check_slab(layout, &data)?;
        if self.written[idx][record as usize] {
            return Err(RwStoreError::Format(format!(
                "netcdf_classic: variable '{}' record {record} was already written",
                layout.name
            )));
        }

        let offset = layout.begin + record * self.record_stride;
        let mut buf = Vec::with_capacity(layout.slab_stride as usize);
        data.encode_into(&mut buf);
        debug_assert_eq!(buf.len() as u64, layout.slab_bytes);
        buf.resize(layout.slab_stride as usize, 0);

        self.file.seek(SeekFrom::Start(offset))?;
        self.file.write_all(&buf)?;
        self.written[idx][record as usize] = true;
        Ok(())
    }

    /// Verify every declared slab was written, then flush and `fsync`.
    pub fn finish(self) -> RwResult<()> {
        let file = self.finish_unsynced()?;
        file.sync_all()?;
        Ok(())
    }

    /// [`finish`](Self::finish) without the `fsync`: verify every declared
    /// slab was written, flush, and hand the file back so the caller can
    /// `sync_all` it off its critical path (and must, before it reports the
    /// file as durable).
    pub fn finish_unsynced(mut self) -> RwResult<File> {
        let mut missing: Vec<String> = Vec::new();
        for (idx, flags) in self.written.iter().enumerate() {
            let name = &self.layouts[idx].name;
            if self.layouts[idx].is_record {
                for (record, done) in flags.iter().enumerate() {
                    if !done {
                        missing.push(format!("{name}[record {record}]"));
                    }
                }
            } else if !flags[0] {
                missing.push(name.clone());
            }
        }
        if !missing.is_empty() {
            let shown = missing
                .iter()
                .take(8)
                .cloned()
                .collect::<Vec<_>>()
                .join(", ");
            let tail = if missing.len() > 8 {
                format!(" (and {} more)", missing.len() - 8)
            } else {
                String::new()
            };
            return Err(RwStoreError::Format(format!(
                "netcdf_classic: finish called with {} slab(s) never written: {shown}{tail}",
                missing.len()
            )));
        }
        self.file.flush()?;
        Ok(self.file)
    }
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

/// First char must be a letter or underscore; the rest are restricted to the
/// NC-safe subset `[A-Za-z0-9_+.@-]`. Rejects (never renames) on violation.
pub(crate) fn name_is_valid(name: &str) -> bool {
    let mut chars = name.chars();
    match chars.next() {
        Some(c) if c.is_ascii_alphabetic() || c == '_' => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '+' | '.' | '@' | '-'))
}

fn check_name(kind: &str, name: &str) -> RwResult<()> {
    if name.is_empty() {
        return Err(RwStoreError::Format(format!(
            "netcdf_classic: {kind} name is empty"
        )));
    }
    if !name_is_valid(name) {
        return Err(RwStoreError::Format(format!(
            "netcdf_classic: {kind} name '{name}' is not NC-safe (must start [A-Za-z_], rest [A-Za-z0-9_+.@-])"
        )));
    }
    Ok(())
}

fn check_attrs(scope: &str, attrs: &[NcAttr]) -> RwResult<()> {
    let mut seen: Vec<&str> = Vec::with_capacity(attrs.len());
    for attr in attrs {
        check_name("attribute", &attr.name)?;
        if seen.contains(&attr.name.as_str()) {
            return Err(RwStoreError::Format(format!(
                "netcdf_classic: duplicate attribute name '{}' in {scope}",
                attr.name
            )));
        }
        seen.push(&attr.name);
    }
    Ok(())
}

/// Returns the record dimension's dimid, when the schema declares one.
fn validate_defs(
    dims: &[NcDim],
    gattrs: &[NcAttr],
    vars: &[NcVarDef],
    num_records: u64,
) -> RwResult<Option<usize>> {
    let mut dim_names: Vec<&str> = Vec::with_capacity(dims.len());
    let mut record_dimid: Option<usize> = None;
    for (dimid, dim) in dims.iter().enumerate() {
        check_name("dimension", &dim.name)?;
        if dim.unlimited {
            if let Some(prior) = record_dimid {
                return Err(RwStoreError::Format(format!(
                    "netcdf_classic: dimensions '{}' and '{}' are both unlimited; the classic format allows one",
                    dims[prior].name, dim.name
                )));
            }
            record_dimid = Some(dimid);
        } else if dim.len == 0 {
            return Err(RwStoreError::Format(format!(
                "netcdf_classic: fixed dimension '{}' has length 0 (set unlimited to make it the record dimension)",
                dim.name
            )));
        }
        if dim_names.contains(&dim.name.as_str()) {
            return Err(RwStoreError::Format(format!(
                "netcdf_classic: duplicate dimension name '{}'",
                dim.name
            )));
        }
        dim_names.push(&dim.name);
    }

    if record_dimid.is_none() && num_records > 0 {
        return Err(RwStoreError::Format(format!(
            "netcdf_classic: {num_records} record(s) declared but no dimension is unlimited"
        )));
    }

    check_attrs("global attributes", gattrs)?;

    let mut var_names: Vec<&str> = Vec::with_capacity(vars.len());
    for var in vars {
        check_name("variable", &var.name)?;
        if var_names.contains(&var.name.as_str()) {
            return Err(RwStoreError::Format(format!(
                "netcdf_classic: duplicate variable name '{}'",
                var.name
            )));
        }
        var_names.push(&var.name);
        let mut seen_dims: Vec<usize> = Vec::with_capacity(var.dimids.len());
        for (axis, &dimid) in var.dimids.iter().enumerate() {
            if dimid >= dims.len() {
                return Err(RwStoreError::Format(format!(
                    "netcdf_classic: variable '{}' references dimid {} but only {} dimension(s) exist",
                    var.name,
                    dimid,
                    dims.len()
                )));
            }
            if seen_dims.contains(&dimid) {
                return Err(RwStoreError::Format(format!(
                    "netcdf_classic: variable '{}' uses dimension '{}' twice",
                    var.name, dims[dimid].name
                )));
            }
            seen_dims.push(dimid);
            if Some(dimid) == record_dimid && axis != 0 {
                return Err(RwStoreError::Format(format!(
                    "netcdf_classic: variable '{}' puts the record dimension '{}' at axis {axis}; the classic format requires it first",
                    var.name, dims[dimid].name
                )));
            }
        }
        check_attrs(&format!("variable '{}'", var.name), &var.attrs)?;
    }

    Ok(record_dimid)
}

// ---------------------------------------------------------------------------
// Header serialization
// ---------------------------------------------------------------------------

fn put_u32(buf: &mut Vec<u8>, v: u32) {
    buf.extend_from_slice(&v.to_be_bytes());
}

/// Emit a `NON_NEG` field at this format's width.
fn put_nonneg(buf: &mut Vec<u8>, format: NcFormat, v: u64) {
    if format.nonneg_width() == 4 {
        buf.extend_from_slice(&(v as u32).to_be_bytes());
    } else {
        buf.extend_from_slice(&v.to_be_bytes());
    }
}

/// Emit an `OFFSET` field at this format's width.
fn put_offset(buf: &mut Vec<u8>, format: NcFormat, v: u64) {
    if format.offset_width() == 4 {
        buf.extend_from_slice(&(v as u32).to_be_bytes());
    } else {
        buf.extend_from_slice(&v.to_be_bytes());
    }
}

fn pad_to_4(buf: &mut Vec<u8>) {
    while buf.len() % 4 != 0 {
        buf.push(0);
    }
}

/// Emit `name := nelems bytes pad-to-4`. `nelems` is the byte length and does
/// NOT include the padding.
fn put_name(buf: &mut Vec<u8>, format: NcFormat, name: &str) {
    let bytes = name.as_bytes();
    put_nonneg(buf, format, bytes.len() as u64);
    buf.extend_from_slice(bytes);
    pad_to_4(buf);
}

fn put_attr(buf: &mut Vec<u8>, format: NcFormat, attr: &NcAttr) {
    put_name(buf, format, &attr.name);
    put_u32(buf, attr.value.nc_type().tag());
    put_nonneg(buf, format, attr.value.nelems());
    match &attr.value {
        NcAttrValue::Text(s) => buf.extend_from_slice(s.as_bytes()),
        NcAttrValue::Ints(v) => {
            for &x in v {
                buf.extend_from_slice(&x.to_be_bytes());
            }
        }
        NcAttrValue::Floats(v) => {
            for &x in v {
                buf.extend_from_slice(&x.to_be_bytes());
            }
        }
        NcAttrValue::Doubles(v) => {
            for &x in v {
                buf.extend_from_slice(&x.to_be_bytes());
            }
        }
    }
    pad_to_4(buf);
}

/// Emit an att_list (global or per-var). Empty ⇒ ABSENT: a 4-byte zero tag
/// followed by a zero `nelems` at this format's width.
fn put_att_list(buf: &mut Vec<u8>, format: NcFormat, attrs: &[NcAttr]) {
    if attrs.is_empty() {
        put_u32(buf, ABSENT_TAG);
        put_nonneg(buf, format, 0);
        return;
    }
    put_u32(buf, NC_ATTRIBUTE);
    put_nonneg(buf, format, attrs.len() as u64);
    for attr in attrs {
        put_attr(buf, format, attr);
    }
}

pub(crate) fn serialize_header(
    format: NcFormat,
    dims: &[NcDim],
    gattrs: &[NcAttr],
    vars: &[NcVarDef],
    slab_strides: &[u64],
    begins: &[u64],
    num_records: u64,
) -> Vec<u8> {
    let mut buf = Vec::with_capacity(512);

    buf.extend_from_slice(b"CDF");
    buf.push(format.version_byte());
    put_nonneg(&mut buf, format, num_records);

    if dims.is_empty() {
        put_u32(&mut buf, ABSENT_TAG);
        put_nonneg(&mut buf, format, 0);
    } else {
        put_u32(&mut buf, NC_DIMENSION);
        put_nonneg(&mut buf, format, dims.len() as u64);
        for dim in dims {
            put_name(&mut buf, format, &dim.name);
            // The record dimension is written with length 0; readers take the
            // real count from numrecs.
            let len = if dim.unlimited { 0 } else { dim.len as u64 };
            put_nonneg(&mut buf, format, len);
        }
    }

    put_att_list(&mut buf, format, gattrs);

    if vars.is_empty() {
        put_u32(&mut buf, ABSENT_TAG);
        put_nonneg(&mut buf, format, 0);
    } else {
        put_u32(&mut buf, NC_VARIABLE);
        put_nonneg(&mut buf, format, vars.len() as u64);
        for (i, var) in vars.iter().enumerate() {
            put_name(&mut buf, format, &var.name);
            put_nonneg(&mut buf, format, var.dimids.len() as u64);
            for &dimid in &var.dimids {
                put_nonneg(&mut buf, format, dimid as u64);
            }
            put_att_list(&mut buf, format, &var.attrs);
            put_u32(&mut buf, var.ty.tag());
            put_nonneg(&mut buf, format, slab_strides[i]);
            put_offset(&mut buf, format, begins[i]);
        }
    }

    buf
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Golden files written by the netCDF-C library (via python-netCDF4) with
    /// `tests/goldens/make_goldens.py`. The schema below reproduces them
    /// exactly; a byte difference means this writer's grammar drifted from
    /// the reference implementation.
    const GOLDEN_CDF1: &[u8] = include_bytes!("../tests/goldens/golden_cdf1.nc");
    const GOLDEN_CDF2: &[u8] = include_bytes!("../tests/goldens/golden_cdf2.nc");
    const GOLDEN_CDF5: &[u8] = include_bytes!("../tests/goldens/golden_cdf5.nc");

    fn tmp_path(name: &str) -> std::path::PathBuf {
        // Parallel tests read one clock tick on the 2.8.6 windows-2025 runner and collided on this name; the counter keeps each call distinct.
        static NEXT_SCRATCH: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let mut p = std::env::temp_dir();
        p.push(format!(
            "rw-store-nc-classic-{}-{}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            name,
            NEXT_SCRATCH.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
        ));
        p
    }

    fn golden_schema() -> (Vec<NcDim>, Vec<NcAttr>, Vec<NcVarDef>) {
        let dims = vec![
            NcDim::record("Time"),
            NcDim::fixed("DateStrLen", 19),
            NcDim::fixed("west_east", 3),
            NcDim::fixed("south_north", 2),
        ];
        let gattrs = vec![
            NcAttr::text("TITLE", " OUTPUT FROM GOLDEN"),
            NcAttr::float("DX", 22000.0),
            NcAttr::int("MAP_PROJ", 1),
        ];
        let vars = vec![
            NcVarDef::new("Times", NcType::Char, vec![0, 1]),
            NcVarDef::new("ITIMESTEP", NcType::Int, vec![0]),
            NcVarDef::new("XLAT", NcType::Float, vec![0, 3, 2]).with_attrs(vec![
                NcAttr::text("units", "degree_north"),
                NcAttr::int("FieldType", 104),
            ]),
            NcVarDef::new("HGT_M", NcType::Float, vec![3, 2]),
        ];
        (dims, gattrs, vars)
    }

    fn write_golden(path: &Path, format: NcFormat) {
        let (dims, gattrs, vars) = golden_schema();
        let mut w = NcClassicWriter::create(path, format, dims, gattrs, vars, 2).unwrap();
        let hgt: Vec<f32> = (0..6).map(|i| i as f32 * 0.5).collect();
        w.put("HGT_M", NcData::Floats(&hgt)).unwrap();
        for (r, stamp) in ["2026-08-10_13:00:00", "2026-08-10_14:00:00"]
            .iter()
            .enumerate()
        {
            let r = r as u64;
            w.put_record_text("Times", r, stamp).unwrap();
            w.put_record("ITIMESTEP", r, NcData::Ints(&[100 + r as i32]))
                .unwrap();
            let xlat: Vec<f32> = (0..6).map(|i| i as f32 + r as f32 * 10.0).collect();
            w.put_record("XLAT", r, NcData::Floats(&xlat)).unwrap();
        }
        w.finish().unwrap();
    }

    fn assert_bytes_eq(got: &[u8], want: &[u8], label: &str) {
        if got == want {
            return;
        }
        assert_eq!(
            got.len(),
            want.len(),
            "{label}: length {} != golden {}",
            got.len(),
            want.len()
        );
        for (i, (a, b)) in got.iter().zip(want).enumerate() {
            assert_eq!(a, b, "{label}: first byte difference at offset 0x{i:04x}");
        }
    }

    #[test]
    fn cdf1_matches_the_netcdf_c_golden_byte_for_byte() {
        let p = tmp_path("cdf1.nc");
        write_golden(&p, NcFormat::Classic);
        let got = std::fs::read(&p).unwrap();
        assert_bytes_eq(&got, GOLDEN_CDF1, "cdf1");
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn put_parallel_writes_the_cdf2_golden_byte_for_byte() {
        let p = tmp_path("cdf2-par.nc");
        let (dims, gattrs, vars) = golden_schema();
        let mut w = NcClassicWriter::create(&p, NcFormat::Offset64, dims, gattrs, vars, 2).unwrap();
        let hgt: Vec<f32> = (0..6).map(|i| i as f32 * 0.5).collect();
        let stamps = ["2026-08-10_13:00:00", "2026-08-10_14:00:00"];
        let jobs: Vec<(&str, Option<u64>)> = vec![
            ("XLAT", Some(1)), ("HGT_M", None), ("Times", Some(0)), ("ITIMESTEP", Some(1)),
            ("XLAT", Some(0)), ("Times", Some(1)), ("ITIMESTEP", Some(0)),
        ];
        w.put_parallel(&jobs, |i| {
            let (name, r) = jobs[i];
            let r = r.unwrap_or(0);
            Ok(match name {
                "HGT_M" => NcSlab::Borrowed(NcData::Floats(&hgt)),
                "Times" => NcSlab::Owned(NcOwned::Chars(stamps[r as usize].as_bytes().to_vec())),
                "ITIMESTEP" => NcSlab::Owned(NcOwned::Ints(vec![100 + r as i32])),
                _ => NcSlab::Owned(NcOwned::Floats((0..6).map(|i| i as f32 + r as f32 * 10.0).collect())),
            })
        })
        .unwrap();
        w.finish().unwrap();
        let got = std::fs::read(&p).unwrap();
        assert_bytes_eq(&got, GOLDEN_CDF2, "cdf2 put_parallel");
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn cdf2_matches_the_netcdf_c_golden_byte_for_byte() {
        let p = tmp_path("cdf2.nc");
        write_golden(&p, NcFormat::Offset64);
        let got = std::fs::read(&p).unwrap();
        assert_bytes_eq(&got, GOLDEN_CDF2, "cdf2");
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn cdf5_matches_the_netcdf_c_golden_byte_for_byte() {
        let p = tmp_path("cdf5.nc");
        write_golden(&p, NcFormat::Data64);
        let got = std::fs::read(&p).unwrap();
        assert_bytes_eq(&got, GOLDEN_CDF5, "cdf5");
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn magic_and_numrecs_carry_the_declared_format_and_record_count() {
        for (format, version) in [
            (NcFormat::Classic, 1u8),
            (NcFormat::Offset64, 2),
            (NcFormat::Data64, 5),
        ] {
            let p = tmp_path("magic.nc");
            write_golden(&p, format);
            let bytes = std::fs::read(&p).unwrap();
            assert_eq!(&bytes[0..3], b"CDF");
            assert_eq!(bytes[3], version, "{:?} version byte", format);
            let n = if format.nonneg_width() == 4 {
                u32::from_be_bytes(bytes[4..8].try_into().unwrap()) as u64
            } else {
                u64::from_be_bytes(bytes[4..12].try_into().unwrap())
            };
            assert_eq!(n, 2, "{:?} numrecs", format);
            let _ = std::fs::remove_file(&p);
        }
    }

    #[test]
    fn a_record_free_schema_writes_a_fixed_only_file() {
        let p = tmp_path("norec.nc");
        let dims = vec![NcDim::fixed("y", 2), NcDim::fixed("x", 3)];
        let vars = vec![NcVarDef::new("field", NcType::Float, vec![0, 1])];
        let mut w =
            NcClassicWriter::create(&p, NcFormat::Offset64, dims, Vec::new(), vars, 0).unwrap();
        w.put("field", NcData::Floats(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
            .unwrap();
        let len = w.file_len();
        w.finish().unwrap();
        let bytes = std::fs::read(&p).unwrap();
        assert_eq!(bytes.len() as u64, len);
        assert_eq!(u32::from_be_bytes(bytes[4..8].try_into().unwrap()), 0);
        assert_eq!(&bytes[bytes.len() - 4..], &6.0f32.to_be_bytes());
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn char_slabs_are_nul_filled_to_the_declared_width() {
        let p = tmp_path("shortstamp.nc");
        let dims = vec![NcDim::record("Time"), NcDim::fixed("DateStrLen", 19)];
        let vars = vec![
            NcVarDef::new("Times", NcType::Char, vec![0, 1]),
            NcVarDef::new("step", NcType::Int, vec![0]),
        ];
        let mut w =
            NcClassicWriter::create(&p, NcFormat::Offset64, dims, Vec::new(), vars, 1).unwrap();
        w.put_record_text("Times", 0, "2026").unwrap();
        w.put_record("step", 0, NcData::Ints(&[7])).unwrap();
        w.finish().unwrap();
        let bytes = std::fs::read(&p).unwrap();
        let tail = &bytes[bytes.len() - 24..];
        assert_eq!(&tail[0..4], b"2026");
        assert!(tail[4..20].iter().all(|&b| b == 0), "stamp not NUL-filled");
        assert_eq!(&tail[20..24], &7i32.to_be_bytes());
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn a_stamp_wider_than_the_variable_is_refused_not_truncated() {
        let p = tmp_path("longstamp.nc");
        let dims = vec![NcDim::record("Time"), NcDim::fixed("DateStrLen", 4)];
        let vars = vec![NcVarDef::new("Times", NcType::Char, vec![0, 1])];
        let mut w =
            NcClassicWriter::create(&p, NcFormat::Offset64, dims, Vec::new(), vars, 1).unwrap();
        let err = w
            .put_record_text("Times", 0, "2026-08-10_13:00:00")
            .unwrap_err()
            .to_string();
        assert!(err.contains("holds 4 character(s)"), "{err}");
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn the_sole_record_variable_exception_drops_slab_padding() {
        // One NC_CHAR record variable of odd width: the spec says its slabs
        // are not padded, so two records occupy exactly 2 x 19 bytes.
        let p = tmp_path("sole.nc");
        let dims = vec![NcDim::record("Time"), NcDim::fixed("DateStrLen", 19)];
        let vars = vec![NcVarDef::new("Times", NcType::Char, vec![0, 1])];
        let mut w =
            NcClassicWriter::create(&p, NcFormat::Offset64, dims, Vec::new(), vars, 2).unwrap();
        let len = w.file_len();
        w.put_record_text("Times", 0, "2026-08-10_13:00:00").unwrap();
        w.put_record_text("Times", 1, "2026-08-10_14:00:00").unwrap();
        w.finish().unwrap();
        let bytes = std::fs::read(&p).unwrap();
        assert_eq!(bytes.len() as u64, len);
        assert_eq!(
            &bytes[bytes.len() - 38..bytes.len() - 19],
            b"2026-08-10_13:00:00"
        );
        assert_eq!(&bytes[bytes.len() - 19..], b"2026-08-10_14:00:00");
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn a_float_record_variable_alone_still_pads_by_the_word_rule() {
        // NC_FLOAT is not one of the exception types, so the padding rule
        // stays in force even when it is the only record variable.
        let p = tmp_path("solefloat.nc");
        let dims = vec![NcDim::record("Time"), NcDim::fixed("n", 3)];
        let vars = vec![NcVarDef::new("v", NcType::Float, vec![0, 1])];
        let mut w =
            NcClassicWriter::create(&p, NcFormat::Offset64, dims, Vec::new(), vars, 2).unwrap();
        w.put_record("v", 0, NcData::Floats(&[1.0, 2.0, 3.0]))
            .unwrap();
        w.put_record("v", 1, NcData::Floats(&[4.0, 5.0, 6.0]))
            .unwrap();
        let len = w.file_len();
        w.finish().unwrap();
        assert_eq!(std::fs::read(&p).unwrap().len() as u64, len);
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn slabs_may_be_written_in_any_order() {
        let p_fwd = tmp_path("fwd.nc");
        let p_rev = tmp_path("rev.nc");
        write_golden(&p_fwd, NcFormat::Data64);

        let (dims, gattrs, vars) = golden_schema();
        let mut w =
            NcClassicWriter::create(&p_rev, NcFormat::Data64, dims, gattrs, vars, 2).unwrap();
        for r in (0..2u64).rev() {
            let xlat: Vec<f32> = (0..6).map(|i| i as f32 + r as f32 * 10.0).collect();
            w.put_record("XLAT", r, NcData::Floats(&xlat)).unwrap();
            w.put_record("ITIMESTEP", r, NcData::Ints(&[100 + r as i32]))
                .unwrap();
            let stamp = if r == 0 {
                "2026-08-10_13:00:00"
            } else {
                "2026-08-10_14:00:00"
            };
            w.put_record_text("Times", r, stamp).unwrap();
        }
        let hgt: Vec<f32> = (0..6).map(|i| i as f32 * 0.5).collect();
        w.put("HGT_M", NcData::Floats(&hgt)).unwrap();
        w.finish().unwrap();

        assert_eq!(
            std::fs::read(&p_fwd).unwrap(),
            std::fs::read(&p_rev).unwrap()
        );
        let _ = std::fs::remove_file(&p_fwd);
        let _ = std::fs::remove_file(&p_rev);
    }

    #[test]
    fn finish_refuses_when_a_slab_was_never_written() {
        let p = tmp_path("partial.nc");
        let dims = vec![NcDim::record("Time"), NcDim::fixed("n", 2)];
        let vars = vec![NcVarDef::new("v", NcType::Float, vec![0, 1])];
        let mut w =
            NcClassicWriter::create(&p, NcFormat::Offset64, dims, Vec::new(), vars, 2).unwrap();
        w.put_record("v", 0, NcData::Floats(&[1.0, 2.0])).unwrap();
        let err = w.finish().unwrap_err().to_string();
        assert!(err.contains("v[record 1]"), "{err}");
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn a_type_mismatch_is_refused_rather_than_coerced() {
        let p = tmp_path("typemix.nc");
        let dims = vec![NcDim::fixed("n", 2)];
        let vars = vec![NcVarDef::new("v", NcType::Float, vec![0])];
        let mut w =
            NcClassicWriter::create(&p, NcFormat::Offset64, dims, Vec::new(), vars, 0).unwrap();
        let err = w.put("v", NcData::Ints(&[1, 2])).unwrap_err().to_string();
        assert!(err.contains("is NC_FLOAT but was handed NC_INT"), "{err}");
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn a_wrong_length_slab_is_refused() {
        let p = tmp_path("badlen.nc");
        let dims = vec![NcDim::fixed("n", 3)];
        let vars = vec![NcVarDef::new("v", NcType::Float, vec![0])];
        let mut w =
            NcClassicWriter::create(&p, NcFormat::Offset64, dims, Vec::new(), vars, 0).unwrap();
        let err = w
            .put("v", NcData::Floats(&[1.0, 2.0]))
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("expects 3 value(s) per slab but got 2"),
            "{err}"
        );
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn writing_the_same_slab_twice_is_refused() {
        let p = tmp_path("twice.nc");
        let dims = vec![NcDim::fixed("n", 1)];
        let vars = vec![NcVarDef::new("v", NcType::Float, vec![0])];
        let mut w =
            NcClassicWriter::create(&p, NcFormat::Offset64, dims, Vec::new(), vars, 0).unwrap();
        w.put("v", NcData::Floats(&[1.0])).unwrap();
        let err = w.put("v", NcData::Floats(&[2.0])).unwrap_err().to_string();
        assert!(err.contains("already written"), "{err}");
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn the_record_dimension_must_come_first() {
        let p = tmp_path("recaxis.nc");
        let dims = vec![NcDim::fixed("n", 2), NcDim::record("Time")];
        let vars = vec![NcVarDef::new("v", NcType::Float, vec![0, 1])];
        let err = NcClassicWriter::create(&p, NcFormat::Offset64, dims, Vec::new(), vars, 1)
            .unwrap_err()
            .to_string();
        assert!(err.contains("requires it first"), "{err}");
    }

    #[test]
    fn two_unlimited_dimensions_are_refused() {
        let p = tmp_path("tworec.nc");
        let dims = vec![NcDim::record("Time"), NcDim::record("Also")];
        let err = NcClassicWriter::create(&p, NcFormat::Offset64, dims, Vec::new(), Vec::new(), 0)
            .unwrap_err()
            .to_string();
        assert!(err.contains("both unlimited"), "{err}");
    }

    #[test]
    fn records_without_a_record_dimension_are_refused() {
        let p = tmp_path("norecdim.nc");
        let dims = vec![NcDim::fixed("n", 2)];
        let err = NcClassicWriter::create(&p, NcFormat::Offset64, dims, Vec::new(), Vec::new(), 3)
            .unwrap_err()
            .to_string();
        assert!(err.contains("no dimension is unlimited"), "{err}");
    }

    #[test]
    fn an_unsafe_name_is_refused_never_renamed() {
        let p = tmp_path("badname.nc");
        let dims = vec![NcDim::fixed("n", 2)];
        let vars = vec![NcVarDef::new("2bad", NcType::Float, vec![0])];
        let err = NcClassicWriter::create(&p, NcFormat::Offset64, dims, Vec::new(), vars, 0)
            .unwrap_err()
            .to_string();
        assert!(err.contains("not NC-safe"), "{err}");
    }

    #[test]
    fn a_slab_past_the_32_bit_vsize_field_names_cdf5() {
        // 2^30 floats = 4 GiB in one slab: fits CDF-5's 64-bit vsize, not
        // CDF-2's 32-bit one. No file is created; the refusal is at layout.
        let p = tmp_path("toobig.nc");
        let dims = vec![NcDim::fixed("n", 1 << 30)];
        let vars = vec![NcVarDef::new("v", NcType::Float, vec![0])];
        let err = NcClassicWriter::create(&p, NcFormat::Offset64, dims, Vec::new(), vars, 0)
            .unwrap_err()
            .to_string();
        assert!(err.contains("write CDF-5 instead"), "{err}");
    }

    #[test]
    fn a_file_past_the_32_bit_offset_field_names_a_wider_format() {
        let p = tmp_path("toofar.nc");
        let dims = vec![NcDim::fixed("n", 1 << 29)];
        let vars = vec![
            NcVarDef::new("a", NcType::Float, vec![0]),
            NcVarDef::new("b", NcType::Float, vec![0]),
        ];
        let err = NcClassicWriter::create(&p, NcFormat::Classic, dims, Vec::new(), vars, 0)
            .unwrap_err()
            .to_string();
        assert!(err.contains("write CDF-2 or CDF-5 instead"), "{err}");
    }

    #[test]
    fn double_and_short_variables_round_trip_their_bytes() {
        let p = tmp_path("widetypes.nc");
        let dims = vec![NcDim::fixed("n", 2)];
        let vars = vec![
            NcVarDef::new("d", NcType::Double, vec![0]),
            NcVarDef::new("s", NcType::Short, vec![0]),
        ];
        let mut w =
            NcClassicWriter::create(&p, NcFormat::Data64, dims, Vec::new(), vars, 0).unwrap();
        w.put("d", NcData::Doubles(&[1.5, -2.25])).unwrap();
        w.put("s", NcData::Shorts(&[7, -9])).unwrap();
        w.finish().unwrap();
        let bytes = std::fs::read(&p).unwrap();
        let n = bytes.len();
        assert_eq!(&bytes[n - 20..n - 12], &1.5f64.to_be_bytes());
        assert_eq!(&bytes[n - 12..n - 4], &(-2.25f64).to_be_bytes());
        assert_eq!(&bytes[n - 4..n - 2], &7i16.to_be_bytes());
        assert_eq!(&bytes[n - 2..], &(-9i16).to_be_bytes());
        let _ = std::fs::remove_file(&p);
    }
}
