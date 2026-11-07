//! `nc_rewrite IN.nc OUT.nc [--format cdf2|cdf5|auto]`
//!
//! Read a NetCDF file with the project's Rust readers and write it back out
//! through `netcdf-writer`. This is the PARITY instrument for the writer:
//! a real wrfout goes in, a classic-format file comes out, and the two
//! are compared variable by variable, attribute by attribute, and
//! render by render (`tools/nc_rewrite_parity.py`,
//! `tools/nc_rewrite_render_parity.py`).
//!
//! Read side, and why it is two crates:
//!
//! * **structure** comes from `netcrust`, the sanctioned read facade --
//!   its NetCDF-4 metadata reconstruction (dimension overrides, the raw
//!   HDF5 index fallback) is the hardened part;
//! * **values** come from `netcdf-reader` directly, because netcrust's
//!   public read surface is `f64` only (gap D2 in the reset map). A
//!   parity harness may not promote an `f32` field through `f64`: the
//!   value survives, but NaN payloads and the exact bit pattern need not,
//!   and byte-comparing renders is the whole point.
//!
//! Memory: variables are read whole. A wrfout history frame is one record
//! and a few tens of megabytes per variable, so this is bounded by the
//! largest single variable, not by the file.
//!
//! `--fields-from TEMPLATE.nc NAME[,NAME]` adds named records that the
//! input lacks, with the template values and attributes unchanged. Every
//! dimension must match. It cannot replace an existing input field. This
//! lets a history file retain the static records required by an external
//! analysis while keeping its forecast fields and valid time.

use std::collections::HashMap;
use std::path::PathBuf;
use std::process::ExitCode;

use netcdf_writer::{AttrValue, NcFormat, NcType, NcWriter, Schema, VarData};

const USAGE: &str = "usage: nc_rewrite IN.nc OUT.nc [--format cdf2|cdf5|auto] [--fields-from TEMPLATE.nc NAME[,NAME]] [--select-dimension NAME=INDEX,INDEX] [--global-int NAME=VALUE]";

fn main() -> ExitCode {
    match run() {
        Ok(summary) => {
            println!("{summary}");
            ExitCode::SUCCESS
        }
        Err(message) => {
            eprintln!("nc_rewrite: {message}");
            ExitCode::FAILURE
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum FormatChoice {
    Cdf2,
    Cdf5,
    Auto,
}

fn run() -> Result<String, String> {
    let mut args = std::env::args().skip(1);
    let input = PathBuf::from(args.next().ok_or(USAGE)?);
    let output = PathBuf::from(args.next().ok_or(USAGE)?);
    let mut choice = FormatChoice::Auto;
    let mut extra = None;
    let mut selections: HashMap<String, Vec<usize>> = HashMap::new();
    let mut global_ints: HashMap<String, i32> = HashMap::new();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--select-dimension" => {
                let spec = args.next().ok_or(USAGE)?;
                let (dim, words) = spec.split_once('=').ok_or("dimension selection must be NAME=INDEX,INDEX")?;
                let indices = words.split(',').map(|v| v.parse::<usize>())
                    .collect::<Result<Vec<_>, _>>().map_err(|e| e.to_string())?;
                if dim.is_empty() || indices.is_empty() || selections.insert(dim.into(), indices).is_some() {
                    return Err("dimension names must be nonempty and unique".into());
                }
            }
            "--global-int" => {
                let spec = args.next().ok_or(USAGE)?;
                let (name, word) = spec.split_once('=').ok_or("global integer must be NAME=VALUE")?;
                let value = word.parse::<i32>().map_err(|e| e.to_string())?;
                if name.is_empty() || global_ints.insert(name.into(), value).is_some() {
                    return Err("global attribute names must be nonempty and unique".into());
                }
            }
            "--fields-from" => {
                let template = PathBuf::from(args.next().ok_or(USAGE)?);
                let names = args.next().ok_or(USAGE)?;
                if extra.is_some() || names.split(',').any(str::is_empty) {
                    return Err(format!(
                        "one nonempty --fields-from list is required; {USAGE}"
                    ));
                }
                extra = Some((template, names));
            }
            "--format" => {
                choice = match args.next().as_deref() {
                    Some("cdf2") => FormatChoice::Cdf2,
                    Some("cdf5") => FormatChoice::Cdf5,
                    Some("auto") => FormatChoice::Auto,
                    other => {
                        return Err(format!("unknown --format {other:?}; {USAGE}"));
                    }
                }
            }
            other => return Err(format!("unknown argument {other:?}; {USAGE}")),
        }
    }

    let structure = netcrust::open(&input).map_err(|err| format!("open {input:?}: {err}"))?;
    let values = netcdf_reader::NcFile::open(&input)
        .map_err(|err| format!("open {input:?} for typed reads: {err}"))?;

    let dims = structure
        .dimensions()
        .map_err(|err| format!("dimensions: {err}"))?;
    for (name, indices) in &selections {
        let dim = dims.iter().find(|d| d.name() == name).ok_or_else(|| format!("no dimension '{name}'"))?;
        if dim.is_unlimited() || indices.iter().any(|&i| i >= dim.len()) {
            return Err(format!("'{name}': selection leaves its fixed dimension"));
        }
        let mut ordered = indices.clone(); ordered.sort_unstable(); ordered.dedup();
        if ordered.len() != indices.len() {
            return Err(format!("'{name}': duplicate index"));
        }
    }
    let vars = structure
        .variables()
        .map_err(|err| format!("variables: {err}"))?;
    let gattrs = structure
        .attributes()
        .map_err(|err| format!("global attributes: {err}"))?;
    let extra_structure = extra
        .as_ref()
        .map(|(path, _)| netcrust::open(path))
        .transpose()
        .map_err(|err| format!("open field template: {err}"))?;
    let extra_values = extra
        .as_ref()
        .map(|(path, _)| netcdf_reader::NcFile::open(path))
        .transpose()
        .map_err(|err| format!("open field template for typed reads: {err}"))?;
    let mut extra_vars = Vec::new();
    if let (Some(template), Some((_, names))) = (&extra_structure, &extra) {
        let candidates = template
            .variables()
            .map_err(|err| format!("template variables: {err}"))?;
        for name in names.split(',') {
            if vars.iter().any(|var| var.name() == name) {
                return Err(format!("'{name}' already exists; copying template fields must not overwrite history values"));
            }
            if extra_vars
                .iter()
                .any(|var: &netcrust::Variable| var.name() == name)
            {
                return Err(format!("duplicate template field '{name}'"));
            }
            let var = candidates
                .iter()
                .find(|var| var.name() == name)
                .ok_or_else(|| format!("template has no field '{name}'"))?;
            for dim in var.dimensions() {
                if !dims.iter().any(|own| {
                    own.name() == dim.name()
                        && own.len() == dim.len()
                        && own.is_unlimited() == dim.is_unlimited()
                }) {
                    return Err(format!(
                        "template field '{name}' has incompatible dimension '{}'",
                        dim.name()
                    ));
                }
            }
            extra_vars.push(var.clone());
        }
    }

    // How many records the source carries: netcrust reports the record
    // dimension's length as the record count.
    let numrecs = dims
        .iter()
        .find(|dim| dim.is_unlimited())
        .map(|dim| dim.len() as u64)
        .unwrap_or(0);

    let needs_cdf5 = vars
        .iter()
        .chain(extra_vars.iter())
        .any(|var| map_dtype(var.dtype()).is_some_and(is_cdf5_only));
    let formats: Vec<NcFormat> = match choice {
        FormatChoice::Cdf2 => vec![NcFormat::Offset64],
        FormatChoice::Cdf5 => vec![NcFormat::Cdf5],
        FormatChoice::Auto if needs_cdf5 => vec![NcFormat::Cdf5],
        // CDF-2 first, CDF-5 only if the sizes demand it. A file the
        // whole world can read beats a file only new readers can.
        FormatChoice::Auto => vec![NcFormat::Offset64, NcFormat::Cdf5],
    };

    let mut last_error = String::new();
    for (attempt, &format) in formats.iter().enumerate() {
        let mut schema = Schema::new(format);
        let mut dimid_of: HashMap<String, usize> = HashMap::new();
        for dim in &dims {
            let dimid = schema
                .def_dim(dim.name(), selections.get(dim.name()).map_or(dim.len(), Vec::len), dim.is_unlimited())
                .map_err(|err| format!("dimension '{}': {err}", dim.name()))?;
            dimid_of.insert(dim.name().to_string(), dimid);
        }
        for attr in &gattrs {
            if global_ints.contains_key(attr.name()) { continue; }
            let value = map_attr(attr.value())
                .map_err(|err| format!("global attribute '{}': {err}", attr.name()))?;
            schema
                .put_global_attr(attr.name(), value)
                .map_err(|err| format!("global attribute '{}': {err}", attr.name()))?;
        }
        for (name, value) in &global_ints {
            schema.put_global_attr(name, AttrValue::Ints(vec![*value])).map_err(|e| e.to_string())?;
        }
        let mut varids = Vec::with_capacity(vars.len());
        for (var, from_template) in vars
            .iter()
            .map(|var| (var, false))
            .chain(extra_vars.iter().map(|var| (var, true)))
        {
            let ty = map_dtype(var.dtype()).ok_or_else(|| {
                format!(
                    "variable '{}': {:?} has no classic external type; \
                                        classic NetCDF cannot represent it",
                    var.name(),
                    var.dtype()
                )
            })?;
            let ids: Vec<usize> = var
                .dimensions()
                .iter()
                .map(|dim| {
                    dimid_of.get(dim.name()).copied().ok_or_else(|| {
                        format!(
                            "variable '{}' uses unknown dimension '{}'",
                            var.name(),
                            dim.name()
                        )
                    })
                })
                .collect::<Result<_, _>>()?;
            let varid = schema
                .def_var(var.name(), ty, &ids)
                .map_err(|err| format!("variable '{}': {err}", var.name()))?;
            for attr in var.attributes() {
                let value = map_attr(attr.value()).map_err(|err| {
                    format!("attribute '{}' of '{}': {err}", attr.name(), var.name())
                })?;
                schema
                    .put_var_attr(varid, attr.name(), value)
                    .map_err(|err| {
                        format!("attribute '{}' of '{}': {err}", attr.name(), var.name())
                    })?;
            }
            varids.push((varid, var, from_template));
        }

        let mut writer = match NcWriter::create(&output, schema) {
            Ok(writer) => writer,
            Err(err) if attempt + 1 < formats.len() => {
                last_error = err.to_string();
                continue;
            }
            Err(err) => return Err(format!("create {output:?}: {err}")),
        };

        for (varid, var, from_template) in &varids {
            let reader = if *from_template {
                extra_values.as_ref().unwrap()
            } else {
                &values
            };
            copy_variable(reader, &mut writer, *varid, var, numrecs, &selections)?;
        }
        writer
            .finish()
            .map_err(|err| format!("finish {output:?}: {err}"))?;

        let label = match format {
            NcFormat::Classic => "CDF-1",
            NcFormat::Offset64 => "CDF-2",
            NcFormat::Cdf5 => "CDF-5",
        };
        let size = std::fs::metadata(&output).map(|m| m.len()).unwrap_or(0);
        let retried = if last_error.is_empty() {
            String::new()
        } else {
            format!(" (retried after: {last_error})")
        };
        return Ok(format!(
            "REWROTE {} -> {} format={label} dims={} vars={} gattrs={} records={numrecs} \
             bytes={size}{retried}",
            input.display(),
            output.display(),
            dims.len(),
            vars.len() + extra_vars.len(),
            gattrs.len(),
        ));
    }
    Err(format!("no classic container fit this file: {last_error}"))
}

/// A copy of `NcType`'s CDF-5-only predicate, kept here because the crate
/// does not export it (it is an internal container rule, not an API).
fn is_cdf5_only(ty: NcType) -> bool {
    matches!(
        ty,
        NcType::UByte | NcType::UShort | NcType::UInt | NcType::Int64 | NcType::UInt64
    )
}

/// NetCDF-4 stores `NC_CHAR` as an HDF5 fixed-length string of size 1, and
/// the reader reports that as `DataType::String` -- the size is gone by
/// the time it reaches us. So a dimensioned `String` variable is treated
/// as `NC_CHAR`, which is what a wrfout's `Times` is, and
/// [`char_bytes`] refuses at read time if the shape says otherwise.
fn map_dtype(dtype: &netcrust::DataType) -> Option<NcType> {
    Some(match dtype {
        netcrust::DataType::I8 => NcType::Byte,
        netcrust::DataType::Char | netcrust::DataType::String => NcType::Char,
        netcrust::DataType::I16 => NcType::Short,
        netcrust::DataType::I32 => NcType::Int,
        netcrust::DataType::F32 => NcType::Float,
        netcrust::DataType::F64 => NcType::Double,
        netcrust::DataType::U8 => NcType::UByte,
        netcrust::DataType::U16 => NcType::UShort,
        netcrust::DataType::U32 => NcType::UInt,
        netcrust::DataType::I64 => NcType::Int64,
        netcrust::DataType::U64 => NcType::UInt64,
        _ => return None,
    })
}

fn map_attr(value: &netcrust::AttributeValue) -> Result<AttrValue, String> {
    use netcrust::AttributeValue as V;
    Ok(match value {
        V::Chars(text) => AttrValue::Text(text.clone()),
        // A NetCDF-4 file can carry a variable-length string attribute.
        // Classic has no string type, so a single string becomes NC_CHAR
        // text and an array of them is refused rather than joined -- a
        // join would silently change the attribute's meaning.
        V::Strings(items) if items.len() == 1 => AttrValue::Text(items[0].clone()),
        V::Strings(items) => {
            return Err(format!(
                "is an array of {} strings; classic NetCDF has no string type and \
                 joining them would change the attribute",
                items.len()
            ))
        }
        V::Bytes(v) => AttrValue::Bytes(v.clone()),
        V::Shorts(v) => AttrValue::Shorts(v.clone()),
        V::Ints(v) => AttrValue::Ints(v.clone()),
        V::Floats(v) => AttrValue::Floats(v.clone()),
        V::Doubles(v) => AttrValue::Doubles(v.clone()),
        V::UBytes(v) => AttrValue::UBytes(v.clone()),
        V::UShorts(v) => AttrValue::UShorts(v.clone()),
        V::UInts(v) => AttrValue::UInts(v.clone()),
        V::Int64s(v) => AttrValue::Int64s(v.clone()),
        V::UInt64s(v) => AttrValue::UInt64s(v.clone()),
    })
}

/// Read one variable in its native type and push it through the writer,
/// record by record when it rides the record dimension.
fn copy_variable(
    values: &netcdf_reader::NcFile,
    writer: &mut NcWriter,
    varid: usize,
    var: &netcrust::Variable,
    numrecs: u64,
    selections: &HashMap<String, Vec<usize>>,
) -> Result<(), String> {
    let name = var.name();
    let is_record = var
        .dimensions()
        .first()
        .is_some_and(|dim| dim.is_unlimited());
    let slab_elems: usize = if is_record {
        var.dimensions().iter().skip(1).map(|d| selections.get(d.name()).map_or(d.len(), Vec::len)).product()
    } else {
        var.dimensions().iter().map(|d| selections.get(d.name()).map_or(d.len(), Vec::len)).product()
    };

    macro_rules! copy {
        ($t:ty, $variant:path) => {{
            let array = values
                .read_variable::<$t>(name)
                .map_err(|err| format!("read '{name}': {err}"))?;
            let flat = array.into_raw_vec_and_offset().0;
            let flat = select_variable(flat, var, selections)?;
            push(
                writer,
                varid,
                name,
                is_record,
                numrecs,
                slab_elems,
                &flat,
                |slice| $variant(slice),
            )?;
        }};
    }

    match var.dtype() {
        netcrust::DataType::F32 => copy!(f32, VarData::F32),
        netcrust::DataType::F64 => copy!(f64, VarData::F64),
        netcrust::DataType::I32 => copy!(i32, VarData::I32),
        netcrust::DataType::I16 => copy!(i16, VarData::I16),
        netcrust::DataType::I8 => copy!(i8, VarData::I8),
        netcrust::DataType::U8 => copy!(u8, VarData::U8),
        netcrust::DataType::U16 => copy!(u16, VarData::U16),
        netcrust::DataType::U32 => copy!(u32, VarData::U32),
        netcrust::DataType::I64 => copy!(i64, VarData::I64),
        netcrust::DataType::U64 => copy!(u64, VarData::U64),
        netcrust::DataType::Char | netcrust::DataType::String => {
            let flat = select_variable(char_bytes(values, var)?, var, selections)?;
            push(
                writer,
                varid,
                name,
                is_record,
                numrecs,
                slab_elems,
                &flat,
                VarData::Char,
            )?;
        }
        other => {
            return Err(format!(
                "variable '{name}': {other:?} has no classic external type"
            ))
        }
    }
    Ok(())
}

/// Rebuild a char variable's exact byte block.
///
/// Neither reader hands back raw `NC_CHAR` bytes -- that is gap D1 in the
/// reset map, and it bites here too. Both offer the variable decoded to
/// strings with trailing NULs trimmed, and the two containers decode it
/// to DIFFERENT shapes:
///
/// * **classic** treats the last dimension as the string length, so a
///   `Times(Time, DateStrLen)` comes back as one string per record;
/// * **NetCDF-4** stores `NC_CHAR` as an HDF5 fixed string of size 1, so
///   the same variable comes back as one string per CHARACTER.
///
/// Both are accepted, told apart by the count, and anything that is
/// neither is refused by name rather than guessed at.
fn char_bytes(values: &netcdf_reader::NcFile, var: &netcrust::Variable) -> Result<Vec<u8>, String> {
    let name = var.name();
    let width = var
        .dimensions()
        .last()
        .map(|dim| dim.len())
        .filter(|len| *len > 0)
        .ok_or_else(|| format!("char variable '{name}' has no trailing length dimension"))?;
    let total: usize = var.dimensions().iter().map(|dim| dim.len()).product();
    let strings = values
        .read_variable_as_strings(name)
        .map_err(|err| format!("read '{name}' as strings: {err}"))?;

    let chunk = if strings.len() == total.div_ceil(width) {
        width // classic: one string per row
    } else if strings.len() == total {
        1 // NetCDF-4: one string per character
    } else {
        return Err(format!(
            "char variable '{name}' decoded to {} string(s); expected {} \
             (one per row of {width}) or {total} (one per character)",
            strings.len(),
            total.div_ceil(width)
        ));
    };

    let mut flat = Vec::with_capacity(total);
    for (index, text) in strings.iter().enumerate() {
        let bytes = text.as_bytes();
        if bytes.len() > chunk {
            return Err(format!(
                "char variable '{name}' element {index} is {} bytes but its slot is {chunk}",
                bytes.len()
            ));
        }
        flat.extend_from_slice(bytes);
        flat.resize(flat.len() + (chunk - bytes.len()), 0);
    }
    if flat.len() != total {
        return Err(format!(
            "char variable '{name}' rebuilt {} byte(s) but the shape wants {total}",
            flat.len()
        ));
    }
    Ok(flat)
}

#[allow(clippy::too_many_arguments)]
fn push<'a, T, F>(
    writer: &mut NcWriter,
    varid: usize,
    name: &str,
    is_record: bool,
    numrecs: u64,
    slab_elems: usize,
    flat: &'a [T],
    wrap: F,
) -> Result<(), String>
where
    F: Fn(&'a [T]) -> VarData<'a>,
{
    if !is_record {
        return writer
            .write_var(varid, wrap(flat))
            .map_err(|err| format!("write '{name}': {err}"));
    }
    let want = slab_elems
        .checked_mul(numrecs as usize)
        .ok_or_else(|| format!("variable '{name}' element count overflows usize"))?;
    if flat.len() != want {
        return Err(format!(
            "variable '{name}' read back {} value(s) but {numrecs} record(s) of \
             {slab_elems} were expected",
            flat.len()
        ));
    }
    for record in 0..numrecs {
        let start = record as usize * slab_elems;
        writer
            .write_record(record, varid, wrap(&flat[start..start + slab_elems]))
            .map_err(|err| format!("write record {record} of '{name}': {err}"))?;
    }
    Ok(())
}


/// Select stored words, without converting their numeric type or NaN payload.
fn select_words<T: Copy>(flat: Vec<T>, shape: &[usize], indices: &[Vec<usize>]) -> Result<Vec<T>, String> {
    if shape.len() != indices.len() || shape.iter().product::<usize>() != flat.len() {
        return Err("dimension selection shape does not match stored words".into());
    }
    if shape.iter().zip(indices).any(|(&n, ix)| ix.iter().any(|&i| i >= n)) {
        return Err("dimension index leaves stored words".into());
    }
    if indices.iter().zip(shape).all(|(ix, &n)| ix.iter().copied().eq(0..n)) { return Ok(flat); }
    let nout = indices.iter().map(Vec::len).product::<usize>();
    let mut out = Vec::with_capacity(nout);
    for k in 0..nout {
        let (mut word, mut offset, mut stride) = (k, 0usize, 1usize);
        for axis in (0..shape.len()).rev() {
            offset += indices[axis][word % indices[axis].len()] * stride;
            word /= indices[axis].len();
            stride *= shape[axis];
        }
        out.push(flat[offset]);
    }
    Ok(out)
}

fn select_variable<T: Copy>(flat: Vec<T>, var: &netcrust::Variable, selections: &HashMap<String, Vec<usize>>) -> Result<Vec<T>, String> {
    let shape = var.dimensions().iter().map(|d| d.len()).collect::<Vec<_>>();
    let indices = var.dimensions().iter().map(|d| selections.get(d.name()).cloned()
        .unwrap_or_else(|| (0..d.len()).collect())).collect::<Vec<_>>();
    select_words(flat, &shape, &indices)
}

#[cfg(test)]
mod dimension_selection_tests {
    use super::select_words;
    #[test]
    fn interior_axis_reorders_whole_planes_and_retains_payloads() {
        let bits: Vec<u32> = (0..24).map(|i| 0x7fc00000 + i).collect();
        let words: Vec<f32> = bits.iter().map(|&b| f32::from_bits(b)).collect();
        let got = select_words(words, &[2,3,4], &[vec![0,1],vec![2,0],vec![0,1,2,3]]).unwrap();
        let expected = [bits[8..12].to_vec(),bits[0..4].to_vec(),bits[20..24].to_vec(),bits[12..16].to_vec()].concat();
        assert_eq!(got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), expected);
    }
    #[test]
    fn no_selection_preserves_every_word_and_scalars() {
        assert_eq!(select_words(vec![17i32], &[], &[]).unwrap(), vec![17]);
        assert_eq!(select_words(vec![1,2,3,4], &[2,2], &[vec![0,1],vec![0,1]]).unwrap(), vec![1,2,3,4]);
    }
    #[test]
    fn invalid_extent_and_indices_fail_before_copy() {
        assert!(select_words(vec![1u8,2], &[3], &[vec![0]]).is_err());
        assert!(select_words(vec![1u8,2], &[2], &[vec![2]]).is_err());
    }
}
