//! Scalar-order terrain metrics and initialized vertical momentum.
//!
//! Independent cells or levels run in parallel. Each cell's edge sum retains
//! the reference order, with a rounding point at every floating operation.

use rayon::prelude::*;
use std::io::{Read, Write};
use std::ops::{Add, Mul, Sub};

pub const ABI_MARKER: &str = "rw_mpas_hostprep --protocol hex-hostprep-v1";

#[derive(Clone, Copy)]
pub(crate) enum Word {
    F32(f32),
    F64(f64),
}
impl Word {
    fn wide(self) -> f64 {
        match self {
            Self::F32(value) => value as f64,
            Self::F64(value) => value,
        }
    }
    fn add(self, rhs: Self) -> Self {
        match (self, rhs) {
            (Self::F32(a), Self::F32(b)) => Self::F32(a + b),
            (a, b) => Self::F64(a.wide() + b.wide()),
        }
    }
    fn sub(self, rhs: Self) -> Self {
        match (self, rhs) {
            (Self::F32(a), Self::F32(b)) => Self::F32(a - b),
            (a, b) => Self::F64(a.wide() - b.wide()),
        }
    }
    fn mul(self, rhs: Self) -> Self {
        match (self, rhs) {
            (Self::F32(a), Self::F32(b)) => Self::F32(a * b),
            (a, b) => Self::F64(a.wide() * b.wide()),
        }
    }
    fn signed_one(width: usize, sign: Self) -> Self {
        match sign {
            Self::F32(value) if width == 4 => Self::F32(1.0f32.copysign(value)),
            value => Self::F64(1.0f64.copysign(value.wide())),
        }
    }
}

enum Numbers {
    F32(Vec<f32>),
    F64(Vec<f64>),
}
impl Numbers {
    fn read(input: &mut impl Read, count: usize, width: u8) -> Result<Self, String> {
        match width {
            4 => Ok(Self::F32(read_values(input, count)?)),
            8 => Ok(Self::F64(read_values(input, count)?)),
            _ => Err("mixed host preparation dtype must be float32 or float64".into()),
        }
    }
    fn at(&self, index: usize) -> Word {
        match self {
            Self::F32(values) => Word::F32(values[index]),
            Self::F64(values) => Word::F64(values[index]),
        }
    }
}

pub(crate) trait Real:
    Copy + Send + Sync + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + PartialOrd
{
    const WIDTH: usize;
    fn zero() -> Self;
    fn one() -> Self;
    fn minus_one() -> Self;
    fn from_word(value: Word) -> Self;
    fn word(self) -> Word;
    fn copysign_one(value: Self) -> Self;
    fn decode(bytes: &[u8]) -> Self;
    fn encode(self, bytes: &mut Vec<u8>);
}
macro_rules! real {
    ($ty:ty, $width:expr, $variant:ident) => {
        impl Real for $ty {
            const WIDTH: usize = $width;
            fn zero() -> Self {
                0.0
            }
            fn one() -> Self {
                1.0
            }
            fn minus_one() -> Self {
                -1.0
            }
            fn from_word(value: Word) -> Self {
                value.wide() as Self
            }
            fn word(self) -> Word {
                Word::$variant(self)
            }
            fn copysign_one(value: Self) -> Self {
                (1.0 as Self).copysign(value)
            }
            fn decode(bytes: &[u8]) -> Self {
                Self::from_le_bytes(bytes.try_into().unwrap())
            }
            fn encode(self, bytes: &mut Vec<u8>) {
                bytes.extend_from_slice(&self.to_le_bytes());
            }
        }
    };
}
real!(f32, 4, F32);
real!(f64, 8, F64);

pub(crate) fn product(values: &[usize]) -> Result<usize, String> {
    values
        .iter()
        .try_fold(1usize, |size, &value| size.checked_mul(value))
        .ok_or_else(|| "host preparation array size overflow".into())
}
pub(crate) fn filled<T: Clone>(count: usize, value: T) -> Result<Vec<T>, String> {
    let mut output = Vec::new();
    output
        .try_reserve_exact(count)
        .map_err(|error| format!("host preparation allocation: {error}"))?;
    output.resize(count, value);
    Ok(output)
}
pub(crate) fn zeros<T: Real>(count: usize) -> Result<Vec<T>, String> {
    filled(count, T::zero())
}
pub(crate) fn read_values<T: Real>(input: &mut impl Read, count: usize) -> Result<Vec<T>, String> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|error| format!("host preparation allocation: {error}"))?;
    let mut block = [0u8; 65536];
    while values.len() < count {
        let elements = (count - values.len()).min(block.len() / T::WIDTH);
        let bytes = &mut block[..elements * T::WIDTH];
        input
            .read_exact(bytes)
            .map_err(|error| format!("truncated host preparation input: {error}"))?;
        values.extend(bytes.chunks_exact(T::WIDTH).map(T::decode));
    }
    Ok(values)
}
pub(crate) fn read_indices(input: &mut impl Read, count: usize) -> Result<Vec<i64>, String> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|error| format!("topology allocation: {error}"))?;
    let mut block = [0u8; 65536];
    while values.len() < count {
        let elements = (count - values.len()).min(block.len() / 8);
        let bytes = &mut block[..elements * 8];
        input
            .read_exact(bytes)
            .map_err(|error| format!("truncated topology input: {error}"))?;
        values.extend(
            bytes
                .chunks_exact(8)
                .map(|bytes| i64::from_le_bytes(bytes.try_into().unwrap())),
        );
    }
    Ok(values)
}
pub(crate) fn write_values<T: Real>(output: &mut impl Write, values: &[T]) -> Result<(), String> {
    let mut block = Vec::with_capacity(65536);
    for values in values.chunks(65536 / T::WIDTH) {
        block.clear();
        for &value in values {
            value.encode(&mut block);
        }
        output
            .write_all(&block)
            .map_err(|error| error.to_string())?;
    }
    Ok(())
}
pub(crate) fn write_indices(output: &mut impl Write, values: &[i64]) -> Result<(), String> {
    // Encode a complete block before writing to the bridge's buffered pipe.
    let mut block = vec![0u8; 65536];
    for values in values.chunks(block.len() / 8) {
        for (bytes, value) in block.chunks_exact_mut(8).zip(values) {
            bytes.copy_from_slice(&value.to_le_bytes());
        }
        output
            .write_all(&block[..values.len() * 8])
            .map_err(|error| error.to_string())?;
    }
    Ok(())
}

fn duplicate(values: &[i64], present_only: bool) -> bool {
    values
        .iter()
        .enumerate()
        .any(|(index, &value)| (!present_only || value >= 0) && values[..index].contains(&value))
}
fn error_indices(label: &str, indices: &[usize], errors: &mut Vec<String>) {
    if !indices.is_empty() {
        errors.push(format!(
            "{label}{}",
            indices
                .iter()
                .take(5)
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
}
fn bad_indices(count: usize, test: impl Fn(usize) -> bool + Sync) -> Result<Vec<usize>, String> {
    let mut flags = filled(count, false)?;
    flags
        .par_iter_mut()
        .enumerate()
        .for_each(|(index, flag)| *flag = test(index));
    let mut bad = Vec::new();
    bad.try_reserve_exact(5)
        .map_err(|error| error.to_string())?;
    for (index, invalid) in flags.into_iter().enumerate() {
        if invalid {
            bad.push(index);
        }
        if bad.len() == 5 {
            break;
        }
    }
    Ok(bad)
}
fn topology(
    input: &mut impl Read,
    output: &mut impl Write,
    nc: usize,
    ne: usize,
    me: usize,
    nv: usize,
    counts: &[i64],
    eoc: &[i64],
) -> Result<(), String> {
    let metadata = read_indices(input, 3)?;
    let me2 = usize::try_from(metadata[0]).map_err(|_| "invalid maxEdges2")?;
    let vd = usize::try_from(metadata[1]).map_err(|_| "invalid vertexDegree")?;
    let regional = metadata[2] != 0;
    if me2 == 0 || vd == 0 {
        return Err("topology extents must be positive".into());
    }
    let coc = read_indices(input, product(&[nc, me])?)?;
    let voc = read_indices(input, product(&[nc, me])?)?;
    let coe = read_indices(input, product(&[ne, 2])?)?;
    let voe = read_indices(input, product(&[ne, 2])?)?;
    let cov = read_indices(input, product(&[nv, vd])?)?;
    let eov = read_indices(input, product(&[nv, vd])?)?;
    let edge_counts = read_indices(input, ne)?;
    let eoe = read_indices(input, product(&[ne, me2])?)?;
    if edge_counts.iter().any(|&n| n < 0 || n as u64 > me2 as u64)
        || coe
            .iter()
            .any(|&v| v < -1 || v as u64 >= nc as u64 && v != -1)
        || voe.iter().any(|&v| v < 0 || v as u64 >= nv as u64)
        || cov
            .iter()
            .any(|&v| v < -1 || v as u64 >= nc as u64 && v != -1)
        || eov
            .iter()
            .any(|&v| v < -1 || v as u64 >= ne as u64 && v != -1)
    {
        return Err("topology array has an out-of-range entry".into());
    }
    for cell in 0..nc {
        let count = counts[cell] as usize;
        if count == 0
            || coc[cell * me..cell * me + count]
                .iter()
                .any(|&v| v < -1 || v as u64 >= nc as u64 && v != -1)
            || voc[cell * me..cell * me + count]
                .iter()
                .any(|&v| v < 0 || v as u64 >= nv as u64)
        {
            return Err("topology cell has an out-of-range entry".into());
        }
    }
    for edge in 0..ne {
        if eoe[edge * me2..edge * me2 + edge_counts[edge] as usize]
            .iter()
            .any(|&v| v < -1 || v as u64 >= ne as u64 && v != -1)
        {
            return Err("topology stencil has an out-of-range entry".into());
        }
    }
    let bad_cells = bad_indices(nc, |cell| {
        let count = counts[cell] as usize;
        let row_edges = &eoc[cell * me..cell * me + count];
        let row_cells = &coc[cell * me..cell * me + count];
        let row_vertices = &voc[cell * me..cell * me + count];
        if duplicate(row_edges, false)
            || duplicate(row_cells, true)
            || duplicate(row_vertices, false)
        {
            return true;
        }
        row_edges.iter().enumerate().any(|(slot, &edge)| {
            let edge = edge as usize;
            let cells = &coe[edge * 2..edge * 2 + 2];
            if !cells.contains(&(cell as i64)) {
                return true;
            }
            let other = if cells[0] == cell as i64 {
                cells[1]
            } else {
                cells[0]
            };
            let endpoints = &voe[edge * 2..edge * 2 + 2];
            row_cells[slot] != other
                || !endpoints.contains(&row_vertices[slot])
                || !endpoints.contains(&row_vertices[(slot + 1) % count])
        })
    })?;
    let bad_edges = bad_indices(ne, |edge| {
        let cells = &coe[edge * 2..edge * 2 + 2];
        let vertices = &voe[edge * 2..edge * 2 + 2];
        cells[0] == cells[1]
            || vertices[0] == vertices[1]
            || cells.iter().filter(|&&cell| cell >= 0).any(|&cell| {
                let cell = cell as usize;
                !eoc[cell * me..cell * me + counts[cell] as usize].contains(&(edge as i64))
            })
            || vertices.iter().any(|&vertex| {
                let vertex = vertex as usize;
                !eov[vertex * vd..vertex * vd + vd].contains(&(edge as i64))
            })
    })?;
    let bad_vertices = bad_indices(nv, |vertex| {
        let cells = &cov[vertex * vd..vertex * vd + vd];
        let edges = &eov[vertex * vd..vertex * vd + vd];
        duplicate(cells, true)
            || duplicate(edges, true)
            || cells.iter().filter(|&&cell| cell >= 0).any(|&cell| {
                let cell = cell as usize;
                !voc[cell * me..cell * me + counts[cell] as usize].contains(&(vertex as i64))
            })
            || edges.iter().filter(|&&edge| edge >= 0).any(|&edge| {
                let edge = edge as usize;
                !voe[edge * 2..edge * 2 + 2].contains(&(vertex as i64))
            })
    })?;
    let bad_stencils = bad_indices(ne, |edge| {
        let actual = &eoe[edge * me2..edge * me2 + edge_counts[edge] as usize];
        let cells = &coe[edge * 2..edge * 2 + 2];
        if regional && (actual.contains(&-1) || cells.contains(&-1)) {
            return actual.contains(&(edge as i64)) || duplicate(actual, true);
        }
        if actual.contains(&(edge as i64)) || duplicate(actual, false) {
            return true;
        }
        let mut expected = Vec::with_capacity(me * 2);
        for &cell in cells {
            if cell < 0 {
                return true;
            }
            let cell = cell as usize;
            for &value in &eoc[cell * me..cell * me + counts[cell] as usize] {
                if value != edge as i64 && !expected.contains(&value) {
                    expected.push(value);
                }
            }
        }
        actual.len() != expected.len() || actual.iter().any(|value| !expected.contains(value))
    })?;
    let mut visited = filled(nc, false)?;
    let mut stack = Vec::new();
    stack
        .try_reserve_exact(nc)
        .map_err(|error| error.to_string())?;
    stack.push(0usize);
    visited[0] = true;
    while let Some(cell) = stack.pop() {
        for &neighbor in &coc[cell * me..cell * me + counts[cell] as usize] {
            if neighbor >= 0 && !visited[neighbor as usize] {
                visited[neighbor as usize] = true;
                stack.push(neighbor as usize);
            }
        }
    }
    let mut errors = Vec::new();
    error_indices(
        "cell edge/neighbor/vertex slot reciprocity fails at cells ",
        &bad_cells,
        &mut errors,
    );
    error_indices(
        "edge-to-cell/vertex reciprocity fails at edges ",
        &bad_edges,
        &mut errors,
    );
    error_indices(
        "vertex-to-cell/edge reciprocity fails at vertices ",
        &bad_vertices,
        &mut errors,
    );
    error_indices(
        "edgesOnEdge tangential stencil is invalid at edges ",
        &bad_stencils,
        &mut errors,
    );
    let unvisited = visited.iter().filter(|&&seen| !seen).count();
    if unvisited != 0 {
        errors.push(format!(
            "cell graph is disconnected ({unvisited} unvisited cells)"
        ));
    }
    serde_json::to_writer(output, &errors).map_err(|error| error.to_string())
}

fn process<T: crate::hosttables::Tab>(
    input: &mut impl Read,
    output: &mut impl Write,
    nc: usize,
    ne: usize,
    me: usize,
    nl: usize,
    mode: u8,
) -> Result<(), String> {
    let counts = read_indices(input, nc)?;
    let eoc = read_indices(input, product(&[nc, me])?)?;
    if counts
        .iter()
        .any(|&count| count < 0 || count as u64 > me as u64)
    {
        return Err("n_edges_on_cell entries must be in [0, maxEdges]".into());
    }
    match mode {
        12 => return crate::hosttables::vertical_velocity_damping::<T>(input, output, nc, nl),
        14 => return crate::hosttables::pow_elementwise::<T>(input, output, nc),
        10 | 15 => {
            return crate::hosttables::deformation_weights::<T>(input, output, nc, ne, me, &counts, &eoc, mode == 15)
        }
        11 => return crate::hosttables::advection_coefficients::<T>(input, output, nc, ne, me, &counts),
        13 => return crate::hosttables::host_mesh_laws(input, output, nc, ne, me, &counts, &eoc),
        _ => {}
    }
    if mode != 0 {
        for cell in 0..nc {
            for slot in 0..counts[cell] as usize {
                let edge = eoc[cell * me + slot];
                if edge < 0 || edge as u64 >= ne as u64 {
                    return Err(format!(
                        "active edges_on_cell[{cell}, {slot}]={edge} is out of range"
                    ));
                }
            }
        }
    }
    if mode == 2 {
        return topology(input, output, nc, ne, me, nl, &counts, &eoc);
    }
    let interfaces = nl.checked_add(1).ok_or("vertical extent overflow")?;
    if mode == 0 {
        let coe = read_indices(input, product(&[ne, 2])?)?;
        let coefficient = read_values::<T>(input, 1)?[0];
        let zb = read_values::<T>(input, product(&[interfaces, 2, ne])?)?;
        let zb3 = read_values::<T>(input, product(&[interfaces, 2, ne])?)?;
        let mut canonical = filled(product(&[nc, me])?, -1i64)?;
        let mut signs = zeros::<T>(product(&[nc, me])?)?;
        let mut sides = filled(product(&[nc, me])?, 0usize)?;
        for cell in 0..nc {
            for slot in 0..counts[cell] as usize {
                let index = cell * me + slot;
                let raw_edge = eoc[index];
                if raw_edge < 0 || raw_edge as u64 >= ne as u64 {
                    return Err(format!(
                        "active edges_on_cell[{cell}, {slot}]={raw_edge} is out of range"
                    ));
                }
                let edge = raw_edge as usize;
                if eoc[cell * me..index].contains(&(edge as i64)) {
                    return Err(format!("cell {cell} lists edge {edge} more than once"));
                }
                canonical[index] = edge as i64;
                if coe[edge * 2] == cell as i64 {
                    sides[index] = 0;
                    signs[index] = T::one();
                } else if coe[edge * 2 + 1] == cell as i64 {
                    sides[index] = 1;
                    signs[index] = T::minus_one();
                } else {
                    return Err(format!(
                        "cell {cell} is not present in cells_on_edge for edge {edge}"
                    ));
                }
            }
        }
        let extent = product(&[nc, me])?;
        let mut zb_cell = zeros::<T>(product(&[interfaces, extent])?)?;
        let mut zb3_cell = zeros::<T>(product(&[interfaces, extent])?)?;
        zb_cell
            .par_chunks_mut(extent)
            .zip(zb3_cell.par_chunks_mut(extent))
            .enumerate()
            .for_each(|(level, (first, third))| {
                for cell in 0..nc {
                    for slot in 0..counts[cell] as usize {
                        let index = cell * me + slot;
                        let source = (level * 2 + sides[index]) * ne + canonical[index] as usize;
                        first[index] = zb[source];
                        third[index] = coefficient * zb3[source];
                    }
                }
            });
        write_indices(output, &canonical)?;
        write_values(output, &signs)?;
        write_values(output, &zb_cell)?;
        write_values(output, &zb3_cell)?;
    } else if mode == 3 {
        let mut widths = [0u8; 8];
        input
            .read_exact(&mut widths)
            .map_err(|error| error.to_string())?;
        if widths[7] as usize != T::WIDTH {
            return Err("mixed momentum output dtype differs from rw storage".into());
        }
        let fzm = Numbers::read(input, nl, widths[0])?;
        let fzp = Numbers::read(input, nl, widths[1])?;
        let ru = Numbers::read(input, product(&[nl, ne])?, widths[2])?;
        let interface_zz = Numbers::read(input, product(&[interfaces, nc])?, widths[3])?;
        let signs = Numbers::read(input, product(&[nc, me])?, widths[4])?;
        let zb = Numbers::read(input, product(&[interfaces, nc, me])?, widths[5])?;
        let zb3 = Numbers::read(input, product(&[interfaces, nc, me])?, widths[6])?;
        let mut rw = read_values::<T>(input, product(&[interfaces, nc])?)?;
        rw.par_chunks_mut(nc)
            .enumerate()
            .for_each(|(level, values)| {
                if level == 0 || level >= nl {
                    return;
                }
                for cell in 0..nc {
                    let mut value = values[cell];
                    for slot in 0..counts[cell] as usize {
                        let edge = eoc[cell * me + slot] as usize;
                        let flux = fzm
                            .at(level)
                            .mul(ru.at(level * ne + edge))
                            .add(fzp.at(level).mul(ru.at((level - 1) * ne + edge)));
                        let index = (level * nc + cell) * me + slot;
                        let terrain = zb
                            .at(index)
                            .add(Word::signed_one(T::WIDTH, flux).mul(zb3.at(index)));
                        let correction = signs
                            .at(cell * me + slot)
                            .mul(terrain)
                            .mul(flux)
                            .mul(interface_zz.at(level * nc + cell));
                        value = T::from_word(value.word().sub(correction));
                    }
                    values[cell] = value;
                }
            });
        write_values(output, &rw)?;
    } else if mode == 1 {
        let fzm = read_values::<T>(input, nl)?;
        let fzp = read_values::<T>(input, nl)?;
        let ru = read_values::<T>(input, product(&[nl, ne])?)?;
        let interface_zz = read_values::<T>(input, product(&[interfaces, nc])?)?;
        let signs = read_values::<T>(input, product(&[nc, me])?)?;
        let zb = read_values::<T>(input, product(&[interfaces, nc, me])?)?;
        let zb3 = read_values::<T>(input, product(&[interfaces, nc, me])?)?;
        let mut rw = read_values::<T>(input, product(&[interfaces, nc])?)?;
        rw.par_chunks_mut(nc)
            .enumerate()
            .for_each(|(level, values)| {
                if level == 0 || level >= nl {
                    return;
                }
                for cell in 0..nc {
                    let mut value = values[cell];
                    for slot in 0..counts[cell] as usize {
                        let edge = eoc[cell * me + slot] as usize;
                        let flux = fzm[level] * ru[level * ne + edge]
                            + fzp[level] * ru[(level - 1) * ne + edge];
                        let index = (level * nc + cell) * me + slot;
                        value = value
                            - signs[cell * me + slot]
                                * (zb[index] + T::copysign_one(flux) * zb3[index])
                                * flux
                                * interface_zz[level * nc + cell];
                    }
                    values[cell] = value;
                }
            });
        write_values(output, &rw)?;
    } else {
        return Err(format!("unsupported host preparation mode {mode}"));
    }
    Ok(())
}

pub fn run(mut input: impl Read, mut output: impl Write) -> Result<(), String> {
    let mut header = [0u8; 42];
    input
        .read_exact(&mut header)
        .map_err(|error| error.to_string())?;
    if &header[..8] != b"HEXHOST1" {
        return Err("invalid host preparation protocol".into());
    }
    let mut sizes = [0usize; 4];
    for (index, size) in sizes.iter_mut().enumerate() {
        let start = 8 + index * 8;
        *size = usize::try_from(u64::from_le_bytes(
            header[start..start + 8].try_into().unwrap(),
        ))
        .map_err(|_| "host preparation dimension does not fit this platform")?;
    }
    let [nc, ne, me, nl] = sizes;
    if header[41] == 15 && header[40] != 4 {
        return Err("GNU deformation mode requires float32".into());
    }
    if header[41] == 15 && !cfg!(all(target_os = "linux", target_env = "gnu")) {
        return Err("GNU deformation mode requires a GNU Linux build".into());
    }
    if nc == 0 || ne == 0 || me == 0 || (nl == 0 && header[41] != 0) {
        return Err("host preparation dimensions must be positive".into());
    }
    match header[40] {
        4 => process::<f32>(&mut input, &mut output, nc, ne, me, nl, header[41]),
        8 => process::<f64>(&mut input, &mut output, nc, ne, me, nl, header[41]),
        _ => Err("host preparation dtype must be float32 or float64".into()),
    }?;
    let mut trailing = [0u8; 1];
    if input
        .read(&mut trailing)
        .map_err(|error| error.to_string())?
        != 0
    {
        return Err("trailing host preparation input bytes".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn buffered_indices_keep_all_protocol_words() {
        let values: Vec<i64> = (0..20001).map(|i| if i % 2 == 0 { i64::MIN + i } else { i64::MAX - i }).collect();
        let expected: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut actual = Vec::new();
        write_indices(&mut actual, &values).unwrap();
        assert_eq!(actual, expected);
    }
    #[test]
    fn dimension_products_are_checked() {
        assert!(product(&[usize::MAX, 2]).is_err());
    }
    #[test]
    fn signed_zero_selects_the_reference_sign() {
        assert_eq!(f32::copysign_one(-0.0).to_bits(), (-1.0f32).to_bits());
        assert_eq!(f64::copysign_one(0.0).to_bits(), 1.0f64.to_bits());
    }
    #[test]
    fn oversized_protocol_allocation_is_rejected() {
        assert!(filled::<u64>(usize::MAX, 0).is_err());
    }
    #[test]
    fn trailing_topology_bytes_are_rejected() {
        let mut input = b"HEXHOST1".to_vec();
        for size in [1u64, 1, 1, 1] {
            input.extend_from_slice(&size.to_le_bytes());
        }
        input.extend_from_slice(&[8, 2]);
        for value in [1i64, 0, 1, 1, 1, -1, 0, 0, -1, 0, 0, 0, -1, 0, -1] {
            input.extend_from_slice(&value.to_le_bytes());
        }
        input.push(99);
        assert_eq!(
            run(&input[..], Vec::new()).unwrap_err(),
            "trailing host preparation input bytes"
        );
    }
}
