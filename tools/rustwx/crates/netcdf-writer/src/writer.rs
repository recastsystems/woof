//! The writer itself: create with a schema, push data, finish.

use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use crate::error::{NcWriteError, Result};
use crate::header::{numrecs_bytes, pad4, serialize_header, NUMRECS_OFFSET};
use crate::layout::Layout;
use crate::schema::Schema;
use crate::types::{NcType, VarData};

/// Elements converted to big-endian per pass. Bounds peak scratch memory
/// at a few hundred kilobytes no matter how large the variable is -- a
/// wrfout slab is tens of megabytes and there are hundreds of them.
const CHUNK_ELEMS: usize = 1 << 15;

/// A classic-format NetCDF file being written.
///
/// Create it with the complete schema, push each fixed variable once with
/// [`write_var`](NcWriter::write_var) and each record slab once with
/// [`write_record`](NcWriter::write_record), then call
/// [`finish`](NcWriter::finish). Order is free; completeness is not --
/// `finish` refuses a file with a hole in it, because an unwritten data
/// region reads back as whatever was on the disk and nothing in the
/// format signals that.
#[derive(Debug)]
pub struct NcWriter {
    path: PathBuf,
    schema: Schema,
    layout: Layout,
    out: BufWriter<File>,
    /// Where the file cursor is, so contiguous writes skip the seek (and
    /// therefore skip flushing the buffer).
    pos: u64,
    numrecs: u64,
    fixed_written: Vec<bool>,
    /// `record_written[rec][record_var_index]`.
    record_written: Vec<Vec<bool>>,
    finished: bool,
}

impl NcWriter {
    /// Reopen a complete prefix with the identical schema and record count.
    /// Validation precedes every write; incomplete or foreign files are refused.
    pub fn resume(path: impl AsRef<Path>, schema: Schema, records: u64) -> Result<Self> {
        let layout = Layout::compute(&schema)?;
        let header = serialize_header(&schema, &layout.vars, records);
        let path = path.as_ref().to_path_buf();
        let mut file = OpenOptions::new().read(true).write(true).open(&path)?;
        let size = layout.recsize.checked_mul(records)
            .and_then(|n| layout.rec_base.checked_add(n))
            .ok_or_else(|| NcWriteError::Capacity("resume size overflow".into()))?;
        if file.metadata()?.len() != size {
            return Err(NcWriteError::Usage("resume prefix has an unexpected byte length".into()));
        }
        let mut actual = vec![0; header.len()];
        file.read_exact(&mut actual)?;
        if actual != header {
            return Err(NcWriteError::Usage("resume prefix schema or record count differs".into()));
        }
        let count = usize::try_from(records)
            .map_err(|_| NcWriteError::Capacity("resume record count exceeds usize".into()))?;
        file.seek(SeekFrom::Start(size))?;
        Ok(Self {
            path, fixed_written: vec![true; schema.vars.len()], schema,
            out: BufWriter::with_capacity(1 << 20, file), pos: size,
            numrecs: records,
            record_written: vec![vec![true; layout.num_record_vars]; count],
            layout, finished: false,
        })
    }

    /// Compute the layout, refuse anything that will not fit, then create
    /// the file and write its header.
    ///
    /// The layout is validated BEFORE the file is created, so a schema
    /// that cannot be represented leaves no truncated `.nc` behind.
    pub fn create(path: impl AsRef<Path>, schema: Schema) -> Result<Self> {
        let layout = Layout::compute(&schema)?;

        let header = serialize_header(&schema, &layout.vars, 0);
        debug_assert_eq!(
            header.len() as u64,
            layout.header_len,
            "header length moved between the measuring pass and the emitting pass"
        );

        let path = path.as_ref().to_path_buf();
        let file = File::create(&path)?;
        let mut out = BufWriter::with_capacity(1 << 20, file);
        out.write_all(&header)?;

        let fixed_written = vec![false; schema.vars.len()];
        Ok(NcWriter {
            path,
            schema,
            out,
            pos: layout.header_len,
            numrecs: 0,
            fixed_written,
            record_written: Vec::new(),
            layout,
            finished: false,
        })
    }

    /// The path being written.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// How many records exist so far.
    pub fn num_records(&self) -> u64 {
        self.numrecs
    }

    /// Write a whole fixed (non-record) variable.
    pub fn write_var(&mut self, varid: usize, data: VarData<'_>) -> Result<()> {
        let var = self.var_def(varid)?;
        if self.layout.vars[varid].is_record {
            return Err(NcWriteError::Usage(format!(
                "variable '{}' rides the record dimension; use write_record",
                var.name
            )));
        }
        self.check_payload(varid, &data)?;
        if self.fixed_written[varid] {
            return Err(NcWriteError::Usage(format!(
                "variable '{}' was already written",
                self.schema.vars[varid].name
            )));
        }
        let begin = self.layout.vars[varid].begin;
        let slab_bytes = self.layout.vars[varid].slab_bytes;
        self.write_payload(begin, &data)?;
        self.write_pad(begin + slab_bytes, pad4(slab_bytes) - slab_bytes)?;
        self.fixed_written[varid] = true;
        Ok(())
    }

    /// Write one record's slab of a record variable.
    pub fn write_record(&mut self, recno: u64, varid: usize, data: VarData<'_>) -> Result<()> {
        let var = self.var_def(varid)?;
        let record_index = match self.layout.vars[varid].record_index {
            Some(index) => index,
            None => {
                return Err(NcWriteError::Usage(format!(
                    "variable '{}' does not ride the record dimension; use write_var",
                    var.name
                )))
            }
        };
        self.check_payload(varid, &data)?;

        let recno_usize = usize::try_from(recno).map_err(|_| {
            NcWriteError::Usage(format!("record index {recno} exceeds platform usize"))
        })?;
        while self.record_written.len() <= recno_usize {
            self.record_written
                .push(vec![false; self.layout.num_record_vars]);
        }
        if self.record_written[recno_usize][record_index] {
            return Err(NcWriteError::Usage(format!(
                "record {recno} of variable '{}' was already written",
                self.schema.vars[varid].name
            )));
        }

        let slab_bytes = self.layout.vars[varid].slab_bytes;
        let offset = self.layout.vars[varid]
            .begin
            .checked_add(recno.checked_mul(self.layout.recsize).ok_or_else(|| {
                NcWriteError::Capacity(format!("record {recno} offset overflows 64 bits"))
            })?)
            .ok_or_else(|| {
                NcWriteError::Capacity(format!("record {recno} offset overflows 64 bits"))
            })?;
        self.write_payload(offset, &data)?;
        if self.layout.pad_record_slabs {
            self.write_pad(offset + slab_bytes, pad4(slab_bytes) - slab_bytes)?;
        }

        self.record_written[recno_usize][record_index] = true;
        self.numrecs = self.numrecs.max(recno + 1);
        Ok(())
    }

    /// Patch `numrecs`, flush, fsync.
    ///
    /// Refuses a file with an unwritten data region: the classic format
    /// has no in-band "this variable was never filled" signal, so a hole
    /// reads back as whatever bytes the filesystem happened to leave.
    pub fn finish(mut self) -> Result<()> {
        self.finish_inner()?;
        self.finished = true;
        Ok(())
    }

    fn finish_inner(&mut self) -> Result<()> {
        let missing_fixed: Vec<&str> = self
            .fixed_written
            .iter()
            .enumerate()
            .filter(|(varid, written)| !**written && !self.layout.vars[*varid].is_record)
            .map(|(varid, _)| self.schema.vars[varid].name.as_str())
            .collect();
        if !missing_fixed.is_empty() {
            return Err(NcWriteError::Usage(format!(
                "finish called with {} variable(s) never written: {}",
                missing_fixed.len(),
                missing_fixed.join(", ")
            )));
        }

        for rec in 0..self.numrecs {
            let rec_usize = rec as usize;
            let flags = &self.record_written[rec_usize];
            let missing: Vec<&str> = self
                .schema
                .vars
                .iter()
                .enumerate()
                .filter_map(|(varid, var)| {
                    let index = self.layout.vars[varid].record_index?;
                    if flags[index] {
                        None
                    } else {
                        Some(var.name.as_str())
                    }
                })
                .collect();
            if !missing.is_empty() {
                return Err(NcWriteError::Usage(format!(
                    "finish called with record {rec} incomplete; never written: {}",
                    missing.join(", ")
                )));
            }
        }

        let patch = numrecs_bytes(self.schema.format, self.numrecs);
        self.out.seek(SeekFrom::Start(NUMRECS_OFFSET))?;
        self.out.write_all(&patch)?;
        self.pos = NUMRECS_OFFSET + patch.len() as u64;
        self.out.flush()?;
        self.out.get_ref().sync_all()?;
        Ok(())
    }

    fn var_def(&self, varid: usize) -> Result<&crate::schema::VarDef> {
        self.schema.vars.get(varid).ok_or_else(|| {
            NcWriteError::Usage(format!(
                "varid {varid} does not exist; the file defines {} variable(s)",
                self.schema.vars.len()
            ))
        })
    }

    fn check_payload(&self, varid: usize, data: &VarData<'_>) -> Result<()> {
        let var = &self.schema.vars[varid];
        let want: NcType = var.ty;
        let got = data.nc_type();
        if want != got {
            return Err(NcWriteError::Usage(format!(
                "variable '{}' is {} but the payload is {}; writing it would change \
                 every value's byte width and shift the whole data section",
                var.name,
                want.name(),
                got.name()
            )));
        }
        let expected = self.layout.vars[varid].elems;
        if data.len() as u64 != expected {
            let scope = if self.layout.vars[varid].is_record {
                "one record of variable"
            } else {
                "variable"
            };
            return Err(NcWriteError::Usage(format!(
                "{scope} '{}' expects {expected} value(s) but got {}",
                var.name,
                data.len()
            )));
        }
        Ok(())
    }

    /// Big-endian conversion, chunked, straight into the file.
    fn write_payload(&mut self, offset: u64, data: &VarData<'_>) -> Result<()> {
        macro_rules! stream {
            ($vals:expr, $width:expr) => {{
                let mut buf: Vec<u8> = Vec::with_capacity(CHUNK_ELEMS * $width);
                let mut cursor = offset;
                for chunk in $vals.chunks(CHUNK_ELEMS) {
                    buf.clear();
                    for value in chunk {
                        buf.extend_from_slice(&value.to_be_bytes());
                    }
                    self.write_at(cursor, &buf)?;
                    cursor += buf.len() as u64;
                }
            }};
        }
        match data {
            // One-byte types need no swap, so hand the slice over as is.
            VarData::Char(v) | VarData::U8(v) => self.write_at(offset, v)?,
            VarData::I8(v) => stream!(v, 1),
            VarData::I16(v) => stream!(v, 2),
            VarData::U16(v) => stream!(v, 2),
            VarData::I32(v) => stream!(v, 4),
            VarData::U32(v) => stream!(v, 4),
            VarData::F32(v) => stream!(v, 4),
            VarData::I64(v) => stream!(v, 8),
            VarData::U64(v) => stream!(v, 8),
            VarData::F64(v) => stream!(v, 8),
        }
        Ok(())
    }

    fn write_pad(&mut self, offset: u64, count: u64) -> Result<()> {
        if count == 0 {
            return Ok(());
        }
        let zeros = [0u8; 8];
        self.write_at(offset, &zeros[..count as usize])
    }

    /// Positioned write that avoids seeking when the target is already
    /// under the cursor -- `BufWriter::seek` flushes, and the in-order
    /// case is every-frame-in-order.
    fn write_at(&mut self, offset: u64, buf: &[u8]) -> Result<()> {
        if offset != self.pos {
            self.out.seek(SeekFrom::Start(offset))?;
            self.pos = offset;
        }
        self.out.write_all(buf)?;
        self.pos += buf.len() as u64;
        Ok(())
    }
}

impl Drop for NcWriter {
    fn drop(&mut self) {
        // A writer dropped without `finish` leaves a file whose numrecs
        // is 0 and whose data section may be short. Say so once, on
        // stderr, rather than leaving a silently wrong tape: the caller
        // that meant to abort loses nothing, and the caller that forgot
        // gets told.
        if !self.finished {
            let _ = writeln!(
                std::io::stderr(),
                "netcdf-writer: {} was dropped without finish(); numrecs stayed 0 \
                 and the file is incomplete",
                self.path.display()
            );
        }
    }
}
