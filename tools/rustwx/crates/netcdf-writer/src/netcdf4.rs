//! Native, uncompressed NetCDF-4 emission for complete fixed-size snapshots.
//!
//! HDF5 superblock v0, compact v2 object headers, contiguous datasets and
//! global heaps follow the HDF Group file-format specification. Dimension
//! scales follow Unidata's NetCDF-4 mapping. No HDF5 or netCDF-C is linked.
//! This interface writes complete arrays; it does not implement append mode.
use crate::{NcType, NcWriteError, Result};
use std::{
    collections::{HashMap, HashSet},
    fs::File,
    io::{BufWriter, Write},
    path::Path,
};

#[derive(Clone)]
pub struct Dimension<'a> {
    pub name: &'a str,
    pub len: usize,
}
#[derive(Clone)]
pub struct Attribute<'a> {
    pub name: &'a str,
    pub dtype: NcType,
    pub bytes: &'a [u8],
}
#[derive(Clone)]
pub enum Values<'a> {
    Primitive(NcType, &'a [u8]),
    Strings(&'a [String]),
}
#[derive(Clone)]
pub struct Variable<'a> {
    pub name: &'a str,
    pub dimensions: Vec<usize>,
    pub attributes: Vec<Attribute<'a>>,
    pub values: Values<'a>,
}
pub(crate) fn bad(s: &str) -> NcWriteError {
    NcWriteError::Schema(s.into())
}
pub(crate) fn u16v(b: &mut Vec<u8>, v: usize) -> Result<()> {
    b.extend_from_slice(
        &u16::try_from(v)
            .map_err(|_| bad("HDF5 compact message exceeds 65535 bytes"))?
            .to_le_bytes(),
    );
    Ok(())
}
pub(crate) fn u32v(b: &mut Vec<u8>, v: u32) {
    b.extend_from_slice(&v.to_le_bytes());
}
pub(crate) fn u64v(b: &mut Vec<u8>, v: u64) {
    b.extend_from_slice(&v.to_le_bytes());
}
pub(crate) fn pad(b: &mut Vec<u8>) {
    b.resize((b.len() + 7) & !7, 0);
}
pub(crate) fn string_type(n: usize) -> Result<Vec<u8>> {
    let mut b = vec![0x13, 0x01, 0, 0];
    u32v(
        &mut b,
        u32::try_from(n.max(1)).map_err(|_| bad("string type exceeds u32"))?,
    );
    Ok(b)
}
pub(crate) fn datatype(t: NcType) -> Vec<u8> {
    let size = t.size() as u32;
    if t == NcType::Char {
        return vec![0x13, 1, 0, 0, 1, 0, 0, 0];
    }
    let mut b = vec![];
    if matches!(t, NcType::Float | NcType::Double) {
        b.extend_from_slice(&[0x11, 0x20, if size == 4 { 0x1f } else { 0x3f }, 0]);
        u32v(&mut b, size);
        b.extend_from_slice(&[0, 0]);
        b.extend_from_slice(&((size * 8) as u16).to_le_bytes());
        if size == 4 {
            b.extend_from_slice(&[23, 8, 0, 23]);
            u32v(&mut b, 127);
        } else {
            b.extend_from_slice(&[52, 11, 0, 52]);
            u32v(&mut b, 1023);
        }
    } else {
        b.extend_from_slice(&[
            0x10,
            if matches!(
                t,
                NcType::Byte | NcType::Short | NcType::Int | NcType::Int64
            ) {
                8
            } else {
                0
            },
            0,
            0,
        ]);
        u32v(&mut b, size);
        b.extend_from_slice(&[0, 0]);
        b.extend_from_slice(&((size * 8) as u16).to_le_bytes());
    }
    b
}
// Match netCDF-C's user/default fill words as well as its _FillValue attribute.
// Complete snapshots contain no unwritten data, but readers expose this property.
pub(crate) fn fill_message(v: &Variable<'_>, hidden: bool) -> Result<Vec<u8>> {
    let bytes = if hidden {
        vec![] // HDF5 default fill, not netCDF's generated dimension variable.
    } else {
        match &v.values {
            Values::Strings(_) => vec![0; 16], // null variable-length descriptor
            Values::Primitive(t, _) => {
                if let Some(a) = v.attributes.iter().find(|a| a.name == "_FillValue") {
                    if a.dtype != *t || a.bytes.len() != t.size() {
                        return Err(bad("_FillValue must be one element of the variable type"));
                    }
                    a.bytes.to_vec()
                } else {
                    match t {
                        NcType::Byte => vec![(-127i8) as u8],
                        NcType::Char => vec![0],
                        NcType::Short => (-32767i16).to_le_bytes().to_vec(),
                        NcType::Int => (-2147483647i32).to_le_bytes().to_vec(),
                        NcType::Float => 0x7cf00000u32.to_le_bytes().to_vec(),
                        NcType::Double => 0x479e000000000000u64.to_le_bytes().to_vec(),
                        NcType::UByte => vec![255],
                        NcType::UShort => u16::MAX.to_le_bytes().to_vec(),
                        NcType::UInt => u32::MAX.to_le_bytes().to_vec(),
                        NcType::Int64 => (i64::MIN + 2).to_le_bytes().to_vec(),
                        NcType::UInt64 => (u64::MAX - 1).to_le_bytes().to_vec(),
                    }
                }
            }
        }
    };
    // v2, late allocation, fill if set, defined; a zero size means HDF5 default.
    let mut b = vec![2, 2, 2, 1];
    u32v(&mut b, bytes.len() as u32);
    b.extend(bytes);
    Ok(b)
}
pub(crate) fn reference_type() -> Vec<u8> {
    vec![0x17, 0, 0, 0, 8, 0, 0, 0]
}
pub(crate) fn vlen_type(strings: bool) -> Vec<u8> {
    let mut b = if strings {
        vec![0x19, 1, 1, 0, 8, 0, 0, 0]
    } else {
        vec![0x19, 0, 0, 0, 16, 0, 0, 0]
    };
    b.extend(if strings {
        datatype(NcType::UByte)
    } else {
        reference_type()
    });
    b
}
pub(crate) fn space(dims: &[usize]) -> Result<Vec<u8>> {
    let rank = u8::try_from(dims.len()).map_err(|_| bad("HDF5 rank exceeds 255"))?;
    let mut b = vec![1, rank, 0, 0, 0, 0, 0, 0];
    for &d in dims {
        u64v(&mut b, d as u64);
    }
    Ok(b)
}
pub(crate) fn attr(name: &str, dt: Vec<u8>, dims: &[usize], bytes: &[u8]) -> Result<Vec<u8>> {
    let ds = space(dims)?;
    let mut b = vec![1, 0];
    u16v(&mut b, name.len() + 1)?;
    u16v(&mut b, dt.len())?;
    u16v(&mut b, ds.len())?;
    b.extend_from_slice(name.as_bytes());
    b.push(0);
    pad(&mut b);
    b.extend(dt);
    pad(&mut b);
    b.extend(ds);
    pad(&mut b);
    b.extend_from_slice(bytes);
    Ok(b)
}
pub(crate) fn text_attr(name: &str, text: &[u8]) -> Result<Vec<u8>> {
    attr(
        name,
        string_type(text.len())?,
        if text.is_empty() { &[0] } else { &[] },
        text,
    )
}
pub(crate) fn cstring_attr(name: &str, text: &str) -> Result<Vec<u8>> {
    let mut b = text.as_bytes().to_vec();
    b.push(0);
    let mut dt = string_type(b.len())?;
    dt[1] = 0;
    attr(name, dt, &[], &b)
}
pub(crate) fn scalar_int_attr(name: &str, value: i32) -> Result<Vec<u8>> {
    attr(name, datatype(NcType::Int), &[], &value.to_le_bytes())
}
pub(crate) fn int_attr(name: &str, values: &[i32]) -> Result<Vec<u8>> {
    let b: Vec<_> = values.iter().flat_map(|x| x.to_le_bytes()).collect();
    attr(name, datatype(NcType::Int), &[values.len()], &b)
}
pub(crate) fn user_attrs(attrs: &[Attribute<'_>]) -> Result<Vec<(u8, Vec<u8>)>> {
    attrs
        .iter()
        .map(|a| {
            let b = if a.dtype == NcType::Char {
                text_attr(a.name, a.bytes)?
            } else {
                attr(
                    a.name,
                    datatype(a.dtype),
                    &[a.bytes.len() / a.dtype.size()],
                    a.bytes,
                )?
            };
            Ok((12, b))
        })
        .collect()
}
pub(crate) fn object(messages: Vec<(u8, Vec<u8>)>) -> Result<Vec<u8>> {
    let mut body = vec![];
    let mut order = 0usize;
    for (kind, data) in messages {
        body.push(kind);
        u16v(&mut body, data.len())?;
        body.push(0);
        u16v(
            &mut body,
            if kind == 12 {
                let n = order;
                order += 1;
                n
            } else {
                0
            },
        )?;
        body.extend(data);
    }
    let mut b = b"OHDR".to_vec();
    b.extend_from_slice(&[2, 0x06]); // tracked attributes, u32 chunk size
    u32v(
        &mut b,
        u32::try_from(body.len()).map_err(|_| bad("object header exceeds u32"))?,
    );
    b.extend(body);
    let checksum = super::hdf_checksum::jenkins_lookup3(&b);
    u32v(&mut b, checksum);
    Ok(b)
}
#[derive(Default)]
pub(crate) struct Heap {
    pub(crate) objects: Vec<Vec<u8>>,
    indices: HashMap<Vec<u8>, usize>,
}
impl Heap {
    pub(crate) fn insert(&mut self, bytes: Vec<u8>) -> usize {
        if let Some(&id) = self.indices.get(&bytes) {
            return id;
        }
        let id = self.objects.len();
        self.indices.insert(bytes.clone(), id);
        self.objects.push(bytes);
        id
    }
    pub(crate) fn layout(&self, start: u64) -> (Vec<(u64, u32)>, Vec<(u64, Vec<u8>)>, u64) {
        // A collection has u16 object indices. Start a new collection before overflow.
        let mut refs = vec![];
        let mut blocks = vec![];
        let mut offset = start;
        for chunk in self.objects.chunks(65534) {
            let mut b = b"GCOL".to_vec();
            b.extend_from_slice(&[1, 0, 0, 0]);
            u64v(&mut b, 0);
            for (i, data) in chunk.iter().enumerate() {
                refs.push((offset, (i + 1) as u32));
                b.extend_from_slice(&((i + 1) as u16).to_le_bytes());
                b.extend_from_slice(&[0; 6]);
                u64v(&mut b, data.len() as u64);
                b.extend(data);
                pad(&mut b);
            }
            let total = (b.len() + 16).max(4096);
            let free = total - b.len();
            b.extend_from_slice(&[0; 8]);
            u64v(&mut b, free as u64);
            b.resize(total, 0);
            b[8..16].copy_from_slice(&(total as u64).to_le_bytes());
            offset += total as u64;
            blocks.push((offset - total as u64, b));
        }
        (refs, blocks, offset)
    }
}
pub(crate) fn descriptor(count: usize, id: usize, refs: &[(u64, u32)]) -> Result<Vec<u8>> {
    let mut b = vec![];
    u32v(
        &mut b,
        u32::try_from(count).map_err(|_| bad("variable length element exceeds u32"))?,
    );
    u64v(&mut b, refs[id].0);
    u32v(&mut b, refs[id].1);
    Ok(b)
}
pub(crate) fn valid_name(s: &str) -> bool {
    !s.is_empty() && !s.bytes().any(|b| b == 0 || b == b'/')
}

/// Write a flat, fixed-size NetCDF-4 dataset, preserving native little-endian
/// bytes and NC_STRING values. The caller owns transactional publication.
pub fn write(
    path: &Path,
    dimensions: &[Dimension<'_>],
    variables: &[Variable<'_>],
    attributes: &[Attribute<'_>],
) -> Result<()> {
    let mut names = HashSet::new();
    for d in dimensions {
        if !valid_name(d.name) || !names.insert(d.name) || d.len == 0 {
            return Err(bad("invalid fixed dimension"));
        }
    }
    names.clear();
    for v in variables {
        if !valid_name(v.name) || !names.insert(v.name) {
            return Err(bad("invalid or duplicate variable"));
        }
        let mut n = 1usize;
        for &i in &v.dimensions {
            n = n
                .checked_mul(
                    dimensions
                        .get(i)
                        .ok_or_else(|| bad("undefined dimension"))?
                        .len,
                )
                .ok_or_else(|| bad("array size overflow"))?;
        }
        match &v.values {
            Values::Primitive(t, b) if n.checked_mul(t.size()) != Some(b.len()) => {
                return Err(bad("variable byte count differs"));
            }
            Values::Strings(s) if s.len() != n || s.iter().any(|s| s.contains('\0')) => {
                return Err(bad("invalid string count or embedded NUL"));
            }
            _ => {}
        }
    }
    for attrs in
        std::iter::once(attributes).chain(variables.iter().map(|v| v.attributes.as_slice()))
    {
        let mut seen = HashSet::new();
        for a in attrs {
            if !valid_name(a.name) || !seen.insert(a.name) || a.bytes.len() % a.dtype.size() != 0 {
                return Err(bad("invalid attribute name or byte count"));
            }
        }
    }
    let mut vars = variables.to_vec();
    let mut scales = vec![];
    let mut hidden = HashSet::new();
    for (i, d) in dimensions.iter().enumerate() {
        if let Some(v) = vars
            .iter()
            .position(|v| v.name == d.name && v.dimensions == [i])
        {
            scales.push(v);
        } else {
            if vars.iter().any(|v| v.name == d.name) {
                return Err(bad("non-coordinate variable shares a dimension name"));
            }
            scales.push(vars.len());
            hidden.insert(vars.len());
            vars.push(Variable {
                name: d.name,
                dimensions: vec![i],
                attributes: vec![],
                values: Values::Primitive(NcType::Float, &[]),
            });
        }
    }
    // Metadata sizes do not depend on addresses. Two passes resolve forward object references.
    let mut addresses = vec![0u64; vars.len()];
    let mut data_addresses = vec![0u64; vars.len()];
    let mut heap = Heap::default();
    let mut refs = vec![];
    let mut strings = vec![vec![]; vars.len()];
    let mut dimrefs = vec![];
    for (j, v) in vars.iter().enumerate() {
        if let Values::Strings(values) = &v.values {
            for s in *values {
                strings[j].push(heap.insert(s.as_bytes().to_vec()));
            }
        }
    }
    // Reserve heap entries for each dimension reference (unique even at placeholder addresses).
    for _ in dimensions {
        let id = heap.objects.len();
        heap.objects.push(vec![0; 8]);
        dimrefs.push(id);
    }
    let header_for = |j: usize, data_addresses: &[u64], refs: &[(u64, u32)]| -> Result<Vec<u8>> {
        let v = &vars[j];
        let is_hidden = hidden.contains(&j);
        let dt = match &v.values {
            Values::Primitive(t, _) => datatype(*t),
            Values::Strings(_) => vlen_type(true),
        };
        let shape: Vec<_> = v.dimensions.iter().map(|&d| dimensions[d].len).collect();
        let bytes = if is_hidden {
            shape.iter().product::<usize>() * 4
        } else {
            match &v.values {
                Values::Primitive(_, b) => b.len(),
                Values::Strings(s) => s
                    .len()
                    .checked_mul(16)
                    .ok_or_else(|| bad("string descriptor size overflow"))?,
            }
        };
        let mut layout = vec![3, 1];
        u64v(
            &mut layout,
            if is_hidden {
                u64::MAX
            } else {
                data_addresses[j]
            },
        );
        u64v(&mut layout, bytes as u64);
        let mut m = vec![
            (1, space(&shape)?),
            (3, dt),
            (5, fill_message(v, is_hidden)?),
            (8, layout),
        ];
        // A scalar variable carries no `_Netcdf4Coordinates`, as netCDF-C writes it: netCDF-C 4.9 refuses
        // to open a file whose scalar variable has an empty one ("NetCDF: HDF error"; MPAS history's
        // isice_lu and iswater_lu, gpuwm 2.9.10 pass 4).
        if !is_hidden && !v.dimensions.is_empty() {
            m.push((
                12,
                int_attr(
                    "_Netcdf4Coordinates",
                    &v.dimensions.iter().map(|&d| d as i32).collect::<Vec<_>>(),
                )?,
            ));
        }
        m.extend(user_attrs(&v.attributes)?);
        if let Some(d) = scales.iter().position(|&s| s == j) {
            m.push((12, cstring_attr("CLASS", "DIMENSION_SCALE")?));
            let label = if is_hidden {
                format!(
                    "This is a netCDF dimension but not a netCDF variable.{:>10}",
                    dimensions[d].len
                )
            } else {
                v.name.into()
            };
            m.push((12, cstring_attr("NAME", &label)?));
            m.push((12, scalar_int_attr("_Netcdf4Dimid", d as i32)?));
        } else if !v.dimensions.is_empty() {
            let mut b = vec![];
            for &d in &v.dimensions {
                b.extend(descriptor(1, dimrefs[d], refs)?);
            }
            m.push((
                12,
                attr(
                    "DIMENSION_LIST",
                    vlen_type(false),
                    &[v.dimensions.len()],
                    &b,
                )?,
            ));
        }
        object(m)
    };
    let root_for = |addresses: &[u64]| -> Result<Vec<u8>> {
        let mut li = vec![0, 1];
        u64v(&mut li, vars.len() as u64);
        u64v(&mut li, u64::MAX);
        u64v(&mut li, u64::MAX);
        let mut m = vec![(2, li), (10, vec![0, 0])];
        m.extend(user_attrs(attributes)?);
        for (j, v) in vars.iter().enumerate() {
            let mut b = vec![1, 5];
            u64v(&mut b, j as u64);
            u16v(&mut b, v.name.len())?;
            b.extend_from_slice(v.name.as_bytes());
            u64v(&mut b, addresses[j]);
            m.push((6, b));
        }
        object(m)
    };
    refs.resize(heap.objects.len(), (0, 1));
    let root = root_for(&addresses)?;
    let mut offset = 96 + root.len() as u64;
    for j in 0..vars.len() {
        addresses[j] = offset;
        offset += header_for(j, &data_addresses, &refs)?.len() as u64;
    }
    offset = (offset + 7) & !7;
    for (i, &id) in dimrefs.iter().enumerate() {
        heap.objects[id] = addresses[scales[i]].to_le_bytes().to_vec();
    }
    let (resolved, blocks, end) = heap.layout(offset);
    refs = resolved;
    offset = end;
    for (j, v) in vars.iter().enumerate() {
        data_addresses[j] = offset;
        let n = if hidden.contains(&j) {
            0
        } else {
            match &v.values {
                Values::Primitive(_, b) => b.len(),
                Values::Strings(s) => s.len() * 16,
            }
        };
        offset = offset
            .checked_add(n as u64)
            .ok_or_else(|| bad("file size overflow"))?;
    }
    let mut sb = b"\x89HDF\r\n\x1a\n".to_vec();
    sb.extend_from_slice(&[0, 0, 0, 0, 0, 8, 8, 0]);
    sb.extend_from_slice(&4u16.to_le_bytes());
    sb.extend_from_slice(&16u16.to_le_bytes());
    u32v(&mut sb, 0);
    for n in [0, u64::MAX, offset, u64::MAX, 0, 96] {
        u64v(&mut sb, n);
    }
    sb.extend_from_slice(&[0; 24]); // root symbol entry cache type, reserved, scratch
    let mut out = BufWriter::new(File::create(path)?);
    out.write_all(&sb)?;
    let root = root_for(&addresses)?;
    out.write_all(&root)?;
    let mut written = 96 + root.len() as u64;
    for j in 0..vars.len() {
        let h = header_for(j, &data_addresses, &refs)?;
        out.write_all(&h)?;
        written += h.len() as u64;
    }
    let aligned = (written + 7) & !7;
    out.write_all(&vec![0; (aligned - written) as usize])?;
    for (_, b) in blocks {
        out.write_all(&b)?;
    }
    for (j, v) in vars.iter().enumerate() {
        if hidden.contains(&j) {
            continue;
        }
        match &v.values {
            Values::Primitive(_, b) => out.write_all(b)?,
            Values::Strings(s) => {
                for (k, s) in s.iter().enumerate() {
                    out.write_all(&descriptor(s.len(), strings[j][k], &refs)?)?;
                }
            }
        }
    }
    out.flush()?;
    Ok(())
}
