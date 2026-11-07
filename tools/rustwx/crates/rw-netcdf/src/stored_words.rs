//! Exact stored words for resident history frames. No CF decoding or promotion.
use netcrust::{DataType as T, NcSliceInfo, NcSliceInfoElem};
use serde::Serialize;
use std::{
    fs,
    io::{BufWriter, Write},
    path::Path,
};

#[derive(Serialize)]
struct Record {
    name: String,
    filename: String,
    shape: Vec<usize>,
    dtype: &'static str,
}

pub fn run(path: &Path, output: &Path, names: &[String]) -> Result<(), String> {
    run_selected(path, output, names, None)
}

/// Read only one leading Time record of numeric fields. The tiny character
/// clock remains metadata: netcrust's text API reads its complete byte tape.
/// Workers own separate reader handles and output files; no global pool or
/// nested all-core fanout is introduced at a forcing seam.
pub fn run_record(
    path: &Path,
    output: &Path,
    names: &[String],
    record: usize,
) -> Result<(), String> {
    run_selected(path, output, names, Some(record))
}

fn run_selected(
    path: &Path,
    output: &Path,
    names: &[String],
    selected: Option<usize>,
) -> Result<(), String> {
    fs::create_dir_all(output).map_err(|e| e.to_string())?;
    let workers = std::thread::available_parallelism()
        .map_or(1, usize::from)
        .min(8)
        .min(names.len().max(1));
    let records = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..workers)
            .map(|worker| {
                scope.spawn(move || {
                    let mut rows = Vec::new();
                    for index in (worker..names.len()).step_by(workers) {
                        rows.push((
                            index,
                            read_one(path, output, &names[index], index, selected)?,
                        ));
                    }
                    Ok::<_, String>(rows)
                })
            })
            .collect();
        let mut rows = Vec::new();
        for handle in handles {
            rows.extend(
                handle
                    .join()
                    .map_err(|_| "stored-word reader worker panicked".to_string())??,
            );
        }
        rows.sort_by_key(|(index, _)| *index);
        Ok::<_, String>(rows.into_iter().map(|(_, row)| row).collect::<Vec<_>>())
    })?;
    let metadata = serde_json::json!({"schema": "gpuwm-rw-netcdf-stored-words-v1", "record": selected, "variables": records});
    fs::write(
        output.join("metadata.json"),
        serde_json::to_vec(&metadata).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())
}

fn read_one(
    path: &Path,
    output: &Path,
    name: &str,
    index: usize,
    selected: Option<usize>,
) -> Result<Record, String> {
    let (file, _) = super::open(path)?;
    let variable = file
        .variable(name)
        .ok_or_else(|| format!("unknown variable {name}"))?;
    let mut shape = variable.shape();
    let selection = if let Some(record) = selected {
        if variable.dimensions().first().map(|d| d.name()) != Some("Time")
            || shape.is_empty()
            || record >= shape[0]
        {
            return Err(format!(
                "{name}: record {record} requires an in-range leading Time axis"
            ));
        }
        shape[0] = 1;
        Some(NcSliceInfo {
            selections: (0..shape.len())
                .map(|axis| NcSliceInfoElem::Slice {
                    start: if axis == 0 { record as u64 } else { 0 },
                    end: if axis == 0 {
                        record as u64 + 1
                    } else {
                        u64::MAX
                    },
                    step: 1,
                })
                .collect(),
        })
    } else {
        None
    };
    let filename = format!("{index}.bin");
    let mut writer =
        BufWriter::new(fs::File::create(output.join(&filename)).map_err(|e| e.to_string())?);
    macro_rules! numeric {
        ($ty:ty, $dtype:literal) => {{
            let array = match &selection {
                Some(selection) => file.read_array_slice::<$ty>(name, selection),
                None => file.read_array::<$ty>(name),
            }
            .map_err(|e| format!("{name}: {e}"))?;
            for value in &array {
                writer
                    .write_all(&value.to_le_bytes())
                    .map_err(|e| e.to_string())?;
            }
            $dtype
        }};
    }
    let dtype = match variable.dtype() {
        T::I8 => numeric!(i8, "|i1"),
        T::U8 => numeric!(u8, "|u1"),
        T::I16 => numeric!(i16, "<i2"),
        T::U16 => numeric!(u16, "<u2"),
        T::I32 => numeric!(i32, "<i4"),
        T::U32 => numeric!(u32, "<u4"),
        T::I64 => numeric!(i64, "<i8"),
        T::U64 => numeric!(u64, "<u8"),
        T::F32 => numeric!(f32, "<f4"),
        T::F64 => numeric!(f64, "<f8"),
        T::Char => {
            let (bytes, _) = super::character_bytes(&file, &variable, false)?;
            write_chars(&mut writer, &bytes, &shape, selected)?;
            "|S1"
        }
        T::String => {
            // NetCDF-4 fixed S1 arrays are exposed as String by the
            // pinned HDF5 metadata reader. Admit only one-byte slots;
            // variable-width strings have no fixed stored-word ABI.
            let (bytes, width) = super::character_bytes(&file, &variable, true)?;
            if width.is_some() {
                return Err(format!(
                    "{name}: stored-words requires single-byte character slots"
                ));
            }
            write_chars(&mut writer, &bytes, &shape, selected)?;
            "|S1"
        }
        other => return Err(format!("{name}: stored-words does not support {other:?}")),
    };
    writer.flush().map_err(|e| e.to_string())?;
    Ok(Record {
        name: name.to_string(),
        filename,
        shape,
        dtype,
    })
}

fn write_chars(
    writer: &mut impl Write,
    bytes: &[u8],
    shape: &[usize],
    record: Option<usize>,
) -> Result<(), String> {
    let bytes = if let Some(record) = record {
        let size = shape
            .iter()
            .try_fold(1usize, |a, b| a.checked_mul(*b))
            .ok_or("character record size overflow")?;
        let start = record
            .checked_mul(size)
            .ok_or("character record offset overflow")?;
        bytes
            .get(
                start
                    ..start
                        .checked_add(size)
                        .ok_or("character record end overflow")?,
            )
            .ok_or("character record extent changed")?
    } else {
        bytes
    };
    writer.write_all(bytes).map_err(|e| e.to_string())
}
