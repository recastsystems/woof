//! Lossless storage for the render-frame schema: the DEFAULT container of
//! every frame `rw_mpas_convert` writes.
//!
//! Project ruling, 2026-10-28: "Lossless compression is the default", and 2026-10-27:
//! Regional and hex get features together. ArWen's history frames became
//! lossless NetCDF-4 in gpuwm 2.9.10 (`netcdf_writer::Nc4Writer`); hex's map
//! frames are written here by the same writer. Only the container changes:
//! every dimension (Time stays the unlimited record dimension), attribute,
//! variable and word is the classic frame's. Chunks are byte-shuffled and
//! deflated on the writer's pool of threads, never by HDF5 on one thread
//! (THE COST THIS REMOVES: the previous opt-in path here called netCDF-C,
//! whose HDF5 deflates a dataset on one core, and it then decoded the whole
//! file again to check it).
//!
//! The ArWen history knobs choose it, so one setting covers both forms:
//! `GPUWM_HISTORY_COMPRESSION` `deflate` (default) or `none` (the classic
//! CDF-2 frame as before), `GPUWM_HISTORY_COMPRESSION_LEVEL` (zlib 1-9,
//! default 4) and `GPUWM_HISTORY_COMPRESSION_THREADS` (0 = every core). The
//! older opt-in `GPUWM_RENDER_NETCDF4_DEFLATE` still wins when it is set:
//! `0` classic, `1` compressed.
use crate::{MpasError, MpasResult};
use netcdf_writer::{
    AttrValue, NcFormat as Nc4Format, NcType as Nc4Type, Nc4Options, Nc4Writer, Schema,
};
use rw_store::netcdf_classic::{
    NcAttr, NcAttrValue, NcClassicWriter, NcData, NcDim, NcFormat, NcType, NcVarDef,
};
use std::{collections::BTreeMap, path::Path};

/// How render frames are stored.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RenderCompression {
    /// The classic frame (`options.format`), as before 2.9.10.
    None,
    /// Lossless NetCDF-4: shuffle + deflate at this zlib level, on `threads`
    /// compression threads (0 = every core).
    Deflate { level: u32, threads: usize },
}

/// The storage the environment asks for (see the module header).
pub fn render_frame_compression() -> MpasResult<RenderCompression> {
    let env = |name: &str| std::env::var(name).unwrap_or_default().trim().to_ascii_lowercase();
    let legacy = env("GPUWM_RENDER_NETCDF4_DEFLATE");
    let chosen = match legacy.as_str() {
        "0" => "none".to_string(),
        "1" => "deflate".to_string(),
        "" => env("GPUWM_HISTORY_COMPRESSION"),
        other => {
            return Err(MpasError::Refusal(format!(
                "GPUWM_RENDER_NETCDF4_DEFLATE must be 0 or 1 (got {other:?})"
            )));
        }
    };
    match chosen.as_str() {
        "none" => Ok(RenderCompression::None),
        "" | "deflate" => {
            let level = match env("GPUWM_HISTORY_COMPRESSION_LEVEL").as_str() {
                "" => 4,
                text => text.parse::<u32>().map_err(|_| {
                    MpasError::Refusal(format!(
                        "GPUWM_HISTORY_COMPRESSION_LEVEL must be a zlib level 1-9 (got {text:?})"
                    ))
                })?,
            }
            .clamp(1, 9);
            let threads = match env("GPUWM_HISTORY_COMPRESSION_THREADS").as_str() {
                "" => 0,
                text => text.parse::<usize>().map_err(|_| {
                    MpasError::Refusal(format!(
                        "GPUWM_HISTORY_COMPRESSION_THREADS must be a count of threads (got {text:?})"
                    ))
                })?,
            };
            Ok(RenderCompression::Deflate { level, threads })
        }
        other => Err(MpasError::Refusal(format!(
            "unknown history compression {other:?} (GPUWM_HISTORY_COMPRESSION): expected 'deflate' \
             (default, lossless NetCDF-4) or 'none' (classic frames)"
        ))),
    }
}

fn nc4_error(error: netcdf_writer::NcWriteError) -> MpasError {
    MpasError::Refusal(format!("lossless render writer: {error}"))
}

fn nc4_type(ty: NcType) -> Nc4Type {
    match ty {
        NcType::Char => Nc4Type::Char,
        NcType::Short => Nc4Type::Short,
        NcType::Int => Nc4Type::Int,
        NcType::Float => Nc4Type::Float,
        NcType::Double => Nc4Type::Double,
    }
}

fn nc4_attr(value: &NcAttrValue) -> AttrValue {
    match value {
        NcAttrValue::Text(v) => AttrValue::Text(v.clone()),
        NcAttrValue::Ints(v) => AttrValue::Ints(v.clone()),
        NcAttrValue::Floats(v) => AttrValue::Floats(v.clone()),
        NcAttrValue::Doubles(v) => AttrValue::Doubles(v.clone()),
    }
}

/// A slab's type and its native-endian bytes (the writer copies them).
fn data_bytes<'a>(data: NcData<'a>) -> (NcType, usize, &'a [u8]) {
    fn raw<T>(values: &[T]) -> &[u8] {
        // SAFETY: plain numeric slices (no padding, every byte initialised),
        // viewed for as long as the borrowed slice lives.
        unsafe {
            std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
        }
    }
    match data {
        NcData::Chars(v) => (NcType::Char, v.len(), v),
        NcData::Shorts(v) => (NcType::Short, v.len(), raw(v)),
        NcData::Ints(v) => (NcType::Int, v.len(), raw(v)),
        NcData::Floats(v) => (NcType::Float, v.len(), raw(v)),
        NcData::Doubles(v) => (NcType::Double, v.len(), raw(v)),
    }
}

struct Variable {
    id: usize,
    ty: NcType,
    /// A fixed-size variable (a CF coordinate axis): one slab, no record.
    fixed: bool,
    /// Values in one slab (one record of a record variable).
    count: usize,
    written: Vec<bool>,
}

pub(super) struct Deflated {
    writer: Nc4Writer,
    vars: BTreeMap<String, Variable>,
    records: u64,
}

pub(super) enum RenderFrameWriter {
    Classic(NcClassicWriter),
    Deflated(Deflated),
}

impl RenderFrameWriter {
    /// The frame in the storage the environment chooses
    /// ([`render_frame_compression`]); `format` is the classic container.
    pub fn create(
        path: &Path,
        format: NcFormat,
        dims: Vec<NcDim>,
        attrs: Vec<NcAttr>,
        vars: Vec<NcVarDef>,
        records: u64,
    ) -> MpasResult<Self> {
        match render_frame_compression()? {
            RenderCompression::None => Ok(Self::Classic(NcClassicWriter::create(
                path, format, dims, attrs, vars, records,
            )?)),
            RenderCompression::Deflate { level, threads } => {
                Self::create_deflated(path, dims, attrs, vars, records, level, threads)
            }
        }
    }

    fn create_deflated(
        path: &Path,
        dims: Vec<NcDim>,
        attrs: Vec<NcAttr>,
        vars: Vec<NcVarDef>,
        records: u64,
        level: u32,
        threads: usize,
    ) -> MpasResult<Self> {
        if records == 0 {
            return Err(MpasError::Refusal(
                "a render frame needs at least one record".into(),
            ));
        }
        let mut schema = Schema::new(Nc4Format::Offset64);
        for d in &dims {
            schema
                .def_dim(&d.name, if d.unlimited { 0 } else { d.len }, d.unlimited)
                .map_err(nc4_error)?;
        }
        for a in &attrs {
            schema.put_global_attr(&a.name, nc4_attr(&a.value)).map_err(nc4_error)?;
        }
        let mut out = BTreeMap::new();
        for v in vars {
            let id = schema.def_var(&v.name, nc4_type(v.ty), &v.dimids).map_err(nc4_error)?;
            for a in &v.attrs {
                schema.put_var_attr(id, &a.name, nc4_attr(&a.value)).map_err(nc4_error)?;
            }
            let fixed = !v.dimids.first().is_some_and(|&slot| dims[slot].unlimited);
            let count = v
                .dimids
                .iter()
                .filter(|&&slot| !dims[slot].unlimited)
                .map(|&slot| dims[slot].len)
                .product();
            out.insert(
                v.name,
                Variable {
                    id,
                    ty: v.ty,
                    fixed,
                    count,
                    written: vec![false; if fixed { 1 } else { records as usize }],
                },
            );
        }
        let options = Nc4Options {
            deflate: Some(level),
            shuffle: true,
            threads,
            ..Nc4Options::default()
        };
        let writer = Nc4Writer::create(path, schema, options).map_err(nc4_error)?;
        Ok(Self::Deflated(Deflated {
            writer,
            vars: out,
            records,
        }))
    }

    /// A fixed-size variable's whole array (a CF coordinate axis).
    pub fn put(&mut self, key: &str, values: NcData<'_>) -> MpasResult<()> {
        match self {
            Self::Classic(w) => {
                w.put(key, values)?;
                Ok(())
            }
            Self::Deflated(w) => w.slab(key, None, values),
        }
    }

    pub fn put_record(&mut self, key: &str, record: u64, values: NcData<'_>) -> MpasResult<()> {
        match self {
            Self::Classic(w) => {
                w.put_record(key, record, values)?;
                Ok(())
            }
            Self::Deflated(w) => w.slab(key, Some(record), values),
        }
    }

    pub fn put_record_text(&mut self, key: &str, record: u64, value: &str) -> MpasResult<()> {
        match self {
            Self::Classic(w) => {
                w.put_record_text(key, record, value)?;
                Ok(())
            }
            Self::Deflated(w) => {
                let width = w
                    .vars
                    .get(key)
                    .ok_or_else(|| MpasError::Refusal(format!("render variable absent: {key}")))?
                    .count;
                if value.len() > width {
                    return Err(MpasError::Refusal(
                        "render text exceeds declared slab width".into(),
                    ));
                }
                let mut data = vec![0u8; width];
                data[..value.len()].copy_from_slice(value.as_bytes());
                w.slab(key, Some(record), NcData::Chars(&data))
            }
        }
    }

    pub fn finish(self) -> MpasResult<()> {
        match self {
            Self::Classic(w) => {
                w.finish()?;
                Ok(())
            }
            Self::Deflated(w) => {
                let missing: Vec<_> = w
                    .vars
                    .iter()
                    .filter(|(_, v)| v.written.contains(&false))
                    .map(|(n, _)| n.clone())
                    .collect();
                if !missing.is_empty() {
                    w.writer.abort();
                    return Err(MpasError::Refusal(format!(
                        "unwritten render slabs: {missing:?}"
                    )));
                }
                w.writer.finish().map_err(nc4_error)
            }
        }
    }
}

impl Deflated {
    /// One whole slab: a fixed variable (`record` None) or one record.
    fn slab(&mut self, key: &str, record: Option<u64>, values: NcData<'_>) -> MpasResult<()> {
        let records = self.records;
        let v = self
            .vars
            .get_mut(key)
            .ok_or_else(|| MpasError::Refusal(format!("render variable absent: {key}")))?;
        let index = match (record, v.fixed) {
            (None, true) => 0,
            (Some(r), false) if r < records => r as usize,
            (None, false) => {
                return Err(MpasError::Refusal(format!(
                    "render variable {key} is not a fixed-size variable"
                )));
            }
            (Some(_), true) => {
                return Err(MpasError::Refusal(format!(
                    "render variable {key} is a fixed-size variable, not a record"
                )));
            }
            (Some(_), false) => {
                return Err(MpasError::Refusal(
                    "render record index outside declared extent".into(),
                ));
            }
        };
        let (ty, len, raw) = data_bytes(values);
        if ty != v.ty || len != v.count {
            return Err(MpasError::Refusal(format!(
                "render slab type/extent mismatch: {key}"
            )));
        }
        if v.written[index] {
            return Err(MpasError::Refusal(format!(
                "render slab already written: {key}"
            )));
        }
        match record {
            None => self.writer.write_var(v.id, nc4_type(ty), raw),
            Some(r) => self.writer.write_record(r, v.id, nc4_type(ty), raw),
        }
        .map_err(nc4_error)?;
        v.written[index] = true;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn schema() -> (Vec<NcDim>, Vec<NcAttr>, Vec<NcVarDef>) {
        let dims = vec![NcDim::record("Time"), NcDim::fixed("x", 8)];
        let attrs = vec![
            NcAttr::text("TITLE", "lossless ☁ test"),
            NcAttr::int("n", 8),
            NcAttr::floats("f", vec![0.0, -0.0, 1.25]),
            NcAttr::doubles("d", vec![1.0f64 / 3.0]),
        ];
        let mut vars: Vec<NcVarDef> = [
            ("C", NcType::Char),
            ("S", NcType::Short),
            ("I", NcType::Int),
            ("F", NcType::Float),
            ("D", NcType::Double),
        ]
        .into_iter()
        .map(|(n, t)| {
            NcVarDef::new(n, t, vec![0, 1]).with_attrs(vec![NcAttr::text("units", "native")])
        })
        .collect();
        vars.push(NcVarDef::new("lat", NcType::Double, vec![1]));
        (dims, attrs, vars)
    }
    fn values(w: &mut RenderFrameWriter) {
        w.put("lat", NcData::Doubles(&[0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]))
            .unwrap();
        for r in 0..2 {
            w.put_record_text("C", r, "stamp").unwrap();
            w.put_record(
                "S",
                r,
                NcData::Shorts(&[i16::MIN, -1, 0, 1, 2, 3, 4, i16::MAX]),
            )
            .unwrap();
            w.put_record(
                "I",
                r,
                NcData::Ints(&[i32::MIN, -1, 0, 1, 2, 3, 4, i32::MAX]),
            )
            .unwrap();
            let floats = [
                0x00000000, 0x80000000, 0x00000001, 0x807fffff, 0x3f800000, 0x7f800000, 0xff800000,
                0x7fc12345,
            ]
            .map(f32::from_bits);
            w.put_record("F", r, NcData::Floats(&floats)).unwrap();
            let doubles = [
                0,
                0x8000000000000000,
                1,
                0x800fffffffffffff,
                0x3ff0000000000000,
                0x7ff0000000000000,
                0xfff0000000000000,
                0x7ff812345678abcd,
            ]
            .map(f64::from_bits);
            w.put_record("D", r, NcData::Doubles(&doubles)).unwrap();
        }
    }
    fn scratch(name: &str) -> std::path::PathBuf {
        let dir = std::env::var_os("TMPDIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(std::env::temp_dir);
        dir.join(format!("{name}-{}.nc", std::process::id()))
    }
    #[test]
    fn deflated_frame_holds_the_classic_frames_words_dimensions_and_attributes() {
        let classic = scratch("lossless-classic");
        let compressed = scratch("lossless-deflated");
        let (d, a, v) = schema();
        let mut w = RenderFrameWriter::Classic(
            NcClassicWriter::create(&classic, NcFormat::Offset64, d.clone(), a.clone(), v.clone(), 2)
                .unwrap(),
        );
        values(&mut w);
        w.finish().unwrap();
        let mut w = RenderFrameWriter::create_deflated(&compressed, d, a, v, 2, 4, 2).unwrap();
        values(&mut w);
        w.finish().unwrap();
        let mut head = [0u8; 4];
        std::io::Read::read_exact(&mut std::fs::File::open(&compressed).unwrap(), &mut head).unwrap();
        assert_eq!(&head, b"\x89HDF");
        let c = netcrust::File::open(&classic).unwrap();
        let z = netcrust::File::open(&compressed).unwrap();
        assert_eq!(c.dimension("Time").unwrap().len(), z.dimension("Time").unwrap().len());
        let bits32 = |f: &netcrust::File| {
            f.read_array::<f32>("F").unwrap().iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        };
        let bits64 = |f: &netcrust::File, k: &str| {
            f.read_array::<f64>(k).unwrap().iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        };
        assert_eq!(bits32(&c), bits32(&z));
        assert_eq!(bits64(&c, "D"), bits64(&z, "D"));
        assert_eq!(bits64(&c, "lat"), bits64(&z, "lat"));
        assert_eq!(c.read_array::<i32>("I").unwrap(), z.read_array::<i32>("I").unwrap());
        assert_eq!(c.read_array::<i16>("S").unwrap(), z.read_array::<i16>("S").unwrap());
        // Times-style text reads back per record, as from the classic frame (rw_wrfbatch reads Times this way).
        assert_eq!(c.read_strings("C").unwrap(), z.read_strings("C").unwrap());
        for key in ["TITLE", "n", "f", "d"] {
            assert_eq!(format!("{:?}", c.attribute(key)), format!("{:?}", z.attribute(key)));
        }
        for key in ["C", "S", "I", "F", "D"] {
            assert_eq!(
                format!("{:?}", c.variable(key).unwrap().attribute("units")),
                format!("{:?}", z.variable(key).unwrap().attribute("units"))
            );
        }
        drop(c);
        drop(z);
        std::fs::remove_file(classic).unwrap();
        std::fs::remove_file(compressed).unwrap();
    }
    #[test]
    fn deflated_frame_rejects_incomplete_mistyped_duplicate_and_overrun_slabs() {
        let path = scratch("lossless-negative");
        let (d, a, v) = schema();
        let mut w = RenderFrameWriter::create_deflated(&path, d, a, v, 2, 4, 1).unwrap();
        assert!(w.put_record("F", 0, NcData::Ints(&[0; 8])).is_err());
        assert!(w.put_record("F", 0, NcData::Floats(&[0.0; 7])).is_err());
        assert!(w.put_record("F", 2, NcData::Floats(&[0.0; 8])).is_err());
        assert!(w.put_record("absent", 0, NcData::Floats(&[0.0; 8])).is_err());
        assert!(w.put("F", NcData::Floats(&[0.0; 8])).is_err());
        assert!(w.put_record("lat", 0, NcData::Doubles(&[0.0; 8])).is_err());
        assert!(w.put_record_text("C", 0, "123456789").is_err());
        w.put_record("F", 0, NcData::Floats(&[0.0; 8])).unwrap();
        assert!(w.put_record("F", 0, NcData::Floats(&[0.0; 8])).is_err());
        assert!(w.finish().is_err());
        let _ = std::fs::remove_file(path);
    }
}
