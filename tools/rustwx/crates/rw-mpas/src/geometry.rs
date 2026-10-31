//! Exact scalar-order MPAS vector geometry and RBF reconstruction.
//!
//! The reference operation order is mpas_vector_operations.F:652-771,
//! mpas_vector_reconstruction.F:51-181 and
//! mpas_rbf_interpolation.F:1079-1145,1527-1559,1670-1846.
//! Cells may run in parallel; arithmetic within each cell is never reordered.

use rayon::prelude::*;
use std::io::{Read, Write};
use std::ops::{Add, Div, Mul, Neg, Sub};

pub const ABI_MARKER: &str = "rw_mpas_geometry --protocol hex-geometry-v1";

pub(crate) trait Real: Copy + Send + Sync + PartialOrd + Add<Output=Self> + Sub<Output=Self>
    + Mul<Output=Self> + Div<Output=Self> + Neg<Output=Self> {
    const WIDTH: usize;
    fn zero() -> Self;
    fn one() -> Self;
    fn half() -> Self;
    fn count(n: usize) -> Self;
    fn number(n: f64) -> Self;
    fn cos(self) -> Self;
    fn sin(self) -> Self;
    fn asin(self) -> Self;
    fn atan2(self, x: Self) -> Self;
    fn pow(self, exponent: Self) -> Self;
    fn sqrt(self) -> Self;
    fn abs(self) -> Self;
    fn finite(self) -> bool;
    fn decode(data: &[u8]) -> Self;
    fn encode(self, out: &mut impl Write) -> Result<(), String>;
}
macro_rules! real {
    ($ty:ty, $width:expr) => {
        impl Real for $ty {
            const WIDTH: usize = $width;
            fn zero() -> Self { 0.0 }
            fn one() -> Self { 1.0 }
            fn half() -> Self { 0.5 }
            fn count(n: usize) -> Self { n as Self }
            fn number(n: f64) -> Self { n as Self }
            fn cos(self) -> Self { self.cos() }
            fn sin(self) -> Self { self.sin() }
            fn asin(self) -> Self { self.asin() }
            fn atan2(self, x: Self) -> Self { self.atan2(x) }
            fn pow(self, exponent: Self) -> Self { self.powf(exponent) }
            fn sqrt(self) -> Self { self.sqrt() }
            fn abs(self) -> Self { self.abs() }
            fn finite(self) -> bool { self.is_finite() }
            fn decode(data: &[u8]) -> Self { Self::from_le_bytes(data.try_into().unwrap()) }
            fn encode(self, out: &mut impl Write) -> Result<(), String> {
                out.write_all(&self.to_le_bytes()).map_err(|e| e.to_string())
            }
        }
    }
}
real!(f32, 4);
real!(f64, 8);

struct Input<'a> { bytes: &'a [u8], cursor: usize }
impl<'a> Input<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        let end = self.cursor.checked_add(n).ok_or("input size overflow")?;
        let result = self.bytes.get(self.cursor..end).ok_or("truncated geometry input")?;
        self.cursor = end;
        Ok(result)
    }
    fn count(&mut self) -> Result<usize, String> {
        usize::try_from(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
            .map_err(|_| "geometry dimension does not fit this platform".into())
    }
    fn real<T: Real>(&mut self) -> Result<T, String> { Ok(T::decode(self.take(T::WIDTH)?)) }
    fn reals<T: Real>(&mut self, count: usize) -> Result<Vec<T>, String> {
        let size = count.checked_mul(T::WIDTH).ok_or("array size overflow")?;
        Ok(self.take(size)?.par_chunks_exact(T::WIDTH).map(T::decode).collect())
    }
    fn indices(&mut self, count: usize) -> Result<Vec<i64>, String> {
        let size = count.checked_mul(8).ok_or("array size overflow")?;
        Ok(self.take(size)?.par_chunks_exact(8)
           .map(|b| i64::from_le_bytes(b.try_into().unwrap())).collect())
    }
}

fn product(a: usize, b: usize) -> Result<usize, String> {
    a.checked_mul(b).ok_or_else(|| "geometry dimension product overflow".into())
}
fn dot<T: Real>(a: &[T], b: &[T]) -> T { (a[0]*b[0] + a[1]*b[1]) + a[2]*b[2] }
fn normalize<T: Real>(a: [T;3], name: &str) -> Result<[T;3], String> {
    let magnitude = dot(&a, &a).sqrt();
    if magnitude == T::zero() || !magnitude.finite() {
        return Err(format!("cannot normalize degenerate {name}"));
    }
    Ok([a[0]/magnitude, a[1]/magnitude, a[2]/magnitude])
}
fn near<T: Real>(point: T, center: T, period: T) -> T {
    let distance = point-center;
    if distance.abs() > period*T::half() {
        point - (distance/distance.abs())*period
    } else { point }
}

/// The three vector-geometry passes on every core (each cell or edge depends
/// only on its own inputs). A degenerate input reruns the serial passes, so the
/// refusal returned is the first in index order, as before.
fn vector_geometry<T: Real>(cell: &[T], edge: &[T], coe: &[i64], eoc: &[i64],
    nc: usize, ne: usize, me: usize, sphere: bool, periodic: bool, xp: T, yp: T)
    -> Result<(Vec<T>, Vec<T>, Vec<T>), String> {
    let vertical_of = |i: usize| -> Result<[T;3], String> {
        if sphere { normalize([cell[i*3],cell[i*3+1],cell[i*3+2]], "cell position") }
        else { Ok([T::zero(),T::zero(),T::one()]) }
    };
    let normal_of = |i: usize| -> Result<[T;3], String> {
        let first = coe[i*2];
        let second = coe[i*2+1];
        if first < -1 || second < -1 || first >= nc as i64 || second >= nc as i64 {
            return Err("cellsOnEdge contains an out-of-range index".into());
        }
        if first == -1 && second == -1 { return Err("an edge cannot have two missing cells".into()); }
        let (a,b) = if first == -1 {
            (&edge[i*3..i*3+3], &cell[second as usize*3..second as usize*3+3])
        } else if second == -1 {
            (&cell[first as usize*3..first as usize*3+3], &edge[i*3..i*3+3])
        } else {
            (&cell[first as usize*3..first as usize*3+3], &cell[second as usize*3..second as usize*3+3])
        };
        let raw = if periodic && first == -1 {
            [b[0]-near(a[0],b[0],xp), b[1]-near(a[1],b[1],yp), b[2]-a[2]]
        } else if periodic {
            [near(b[0],a[0],xp)-a[0], near(b[1],a[1],yp)-a[1], b[2]-a[2]]
        } else { [b[0]-a[0], b[1]-a[1], b[2]-a[2]] };
        normalize(raw, "edge normal")
    };
    let mut vertical = vec![T::zero(); nc*3];
    let mut normal = vec![T::zero(); ne*3];
    let parallel = vertical.par_chunks_exact_mut(3).enumerate()
        .try_for_each(|(i, out)| { out.copy_from_slice(&vertical_of(i)?); Ok::<(), String>(()) })
        .and_then(|()| normal.par_chunks_exact_mut(3).enumerate()
            .try_for_each(|(i, out)| { out.copy_from_slice(&normal_of(i)?); Ok(()) }));
    if parallel.is_err() {
        return vector_geometry_serial(cell, edge, coe, eoc, nc, ne, me, sphere, periodic, xp, yp);
    }
    let plane_of = |i: usize| -> Result<[T;6], String> {
        let first = eoc[i*me];
        if first < 0 || first >= ne as i64 { return Err("the first edge on every cell must be valid".into()); }
        let norm = &normal[first as usize*3..first as usize*3+3];
        let vert = &vertical[i*3..i*3+3];
        let radial = dot(norm,vert);
        let x = normalize([norm[0]-radial*vert[0],norm[1]-radial*vert[1],norm[2]-radial*vert[2]], "cell tangent x vector")?;
        let y = normalize([vert[1]*x[2]-vert[2]*x[1],vert[2]*x[0]-vert[0]*x[2],vert[0]*x[1]-vert[1]*x[0]], "cell tangent y vector")?;
        Ok([x[0],x[1],x[2],y[0],y[1],y[2]])
    };
    let mut plane = vec![T::zero(); nc*6];
    if plane.par_chunks_exact_mut(6).enumerate()
        .try_for_each(|(i, out)| { out.copy_from_slice(&plane_of(i)?); Ok::<(), String>(()) }).is_err() {
        return vector_geometry_serial(cell, edge, coe, eoc, nc, ne, me, sphere, periodic, xp, yp);
    }
    Ok((normal,vertical,plane))
}

fn vector_geometry_serial<T: Real>(cell: &[T], edge: &[T], coe: &[i64], eoc: &[i64],
    nc: usize, ne: usize, me: usize, sphere: bool, periodic: bool, xp: T, yp: T)
    -> Result<(Vec<T>, Vec<T>, Vec<T>), String> {
    let mut vertical = vec![T::zero(); nc*3];
    for (i, out) in vertical.chunks_exact_mut(3).enumerate() {
        let value = if sphere {
            normalize([cell[i*3],cell[i*3+1],cell[i*3+2]], "cell position")?
        } else { [T::zero(),T::zero(),T::one()] };
        out.copy_from_slice(&value);
    }
    let mut normal = vec![T::zero(); ne*3];
    for (i, out) in normal.chunks_exact_mut(3).enumerate() {
        let first = coe[i*2];
        let second = coe[i*2+1];
        if first < -1 || second < -1 || first >= nc as i64 || second >= nc as i64 {
            return Err("cellsOnEdge contains an out-of-range index".into());
        }
        if first == -1 && second == -1 { return Err("an edge cannot have two missing cells".into()); }
        let (a,b) = if first == -1 {
            (&edge[i*3..i*3+3], &cell[second as usize*3..second as usize*3+3])
        } else if second == -1 {
            (&cell[first as usize*3..first as usize*3+3], &edge[i*3..i*3+3])
        } else {
            (&cell[first as usize*3..first as usize*3+3], &cell[second as usize*3..second as usize*3+3])
        };
        let raw = if periodic && first == -1 {
            [b[0]-near(a[0],b[0],xp), b[1]-near(a[1],b[1],yp), b[2]-a[2]]
        } else if periodic {
            [near(b[0],a[0],xp)-a[0], near(b[1],a[1],yp)-a[1], b[2]-a[2]]
        } else { [b[0]-a[0], b[1]-a[1], b[2]-a[2]] };
        out.copy_from_slice(&normalize(raw, "edge normal")?);
    }
    let mut plane = vec![T::zero(); nc*6];
    for (i,out) in plane.chunks_exact_mut(6).enumerate() {
        let first = eoc[i*me];
        if first < 0 || first >= ne as i64 { return Err("the first edge on every cell must be valid".into()); }
        let norm = &normal[first as usize*3..first as usize*3+3];
        let vert = &vertical[i*3..i*3+3];
        let radial = dot(norm,vert);
        let x = normalize([norm[0]-radial*vert[0],norm[1]-radial*vert[1],norm[2]-radial*vert[2]], "cell tangent x vector")?;
        let y = normalize([vert[1]*x[2]-vert[2]*x[1],vert[2]*x[0]-vert[0]*x[2],vert[0]*x[1]-vert[1]*x[0]], "cell tangent y vector")?;
        out[..3].copy_from_slice(&x);
        out[3..].copy_from_slice(&y);
    }
    Ok((normal,vertical,plane))
}

fn solve<T: Real>(matrix: &[T], rhs: &[T], n: usize) -> Result<Vec<T>, String> {
    let mut a = matrix.to_vec();
    let mut b = rhs.to_vec();
    let mut indices: Vec<usize> = (0..n).collect();
    let mut scales = vec![T::zero();n];
    for i in 0..n {
        let mut maximum = T::zero();
        for j in 0..n { if a[i*n+j].abs() > maximum { maximum = a[i*n+j].abs(); } }
        if maximum == T::zero() { return Err("singular RBF reconstruction matrix".into()); }
        scales[i] = maximum;
    }
    for j in 0..n-1 {
        let mut value = T::zero();
        let mut slot = j;
        for i in j..n {
            let candidate = a[indices[i]*n+j].abs()/scales[indices[i]];
            if candidate > value { value = candidate; slot = i; }
        }
        indices.swap(j,slot);
        let pivot = indices[j];
        if a[pivot*n+j] == T::zero() { return Err("singular RBF reconstruction matrix".into()); }
        for i in j+1..n {
            let row = indices[i];
            let ratio = a[row*n+j]/a[pivot*n+j];
            a[row*n+j] = ratio;
            for k in j+1..n { a[row*n+k] = a[row*n+k]-ratio*a[pivot*n+k]; }
        }
    }
    for i in 0..n-1 {
        for j in i+1..n { b[indices[j]] = b[indices[j]]-a[indices[j]*n+i]*b[indices[i]]; }
    }
    let mut result = vec![T::zero();n];
    if a[indices[n-1]*n+n-1] == T::zero() { return Err("singular RBF reconstruction matrix".into()); }
    result[n-1] = b[indices[n-1]]/a[indices[n-1]*n+n-1];
    for i in (0..n-1).rev() {
        result[i] = b[indices[i]];
        for j in i+1..n { result[i] = result[i]-a[indices[i]*n+j]*result[j]; }
        result[i] = result[i]/a[indices[i]*n+i];
    }
    Ok(result)
}

fn reconstruct_cell<T: Real>(source: &[T], normals: &[T], destination: &[T],
    basis: &[T], alpha: T, count: usize, output: &mut [T]) -> Result<(),String> {
    let mut points = vec![T::zero();count*2];
    let mut unit = vec![T::zero();count*2];
    for i in 0..count {
        for c in 0..2 {
            points[i*2+c] = dot(&source[i*3..i*3+3],&basis[c*3..c*3+3]);
            unit[i*2+c] = dot(&normals[i*3..i*3+3],&basis[c*3..c*3+3]);
        }
    }
    let target = [dot(destination,&basis[..3]),dot(destination,&basis[3..])];
    let n = count+2;
    let mut matrix = vec![T::zero();n*n];
    let mut rhs = vec![T::zero();n*2];
    let alpha_squared = alpha*alpha;
    for j in 0..count {
        for i in j..count {
            let dx = points[i*2]-points[j*2];
            let dy = points[i*2+1]-points[j*2+1];
            let r_squared = (dx*dx+dy*dy)/alpha_squared;
            let rbf = T::one()/(T::one()+r_squared).sqrt();
            let normal_dot = unit[i*2]*unit[j*2]+unit[i*2+1]*unit[j*2+1];
            matrix[i*n+j] = rbf*normal_dot;
            matrix[j*n+i] = matrix[i*n+j];
        }
    }
    for j in 0..count {
        let dx = target[0]-points[j*2];
        let dy = target[1]-points[j*2+1];
        let r_squared = (dx*dx+dy*dy)/alpha_squared;
        let rbf = T::one()/(T::one()+r_squared).sqrt();
        for c in 0..2 {
            rhs[c*n+j] = rbf*unit[j*2+c];
            matrix[j*n+count+c] = unit[j*2+c];
            matrix[(count+c)*n+j] = unit[j*2+c];
        }
    }
    rhs[count] = T::one();
    rhs[n+count+1] = T::one();
    let first = solve(&matrix,&rhs[..n],n)?;
    let second = solve(&matrix,&rhs[n..],n)?;
    for c in 0..3 { for i in 0..count {
        output[i*3+c] = basis[c]*first[i]+basis[3+c]*second[i];
    } }
    Ok(())
}

fn process<T: Real>(input: &mut Input<'_>, output: &mut impl Write,
    nc: usize, ne: usize, me: usize, solved: usize, sphere: bool, periodic: bool, mode: u8)
    -> Result<(), String> {
    let xp = input.real::<T>()?;
    let yp = input.real::<T>()?;
    let cell = input.reals::<T>(product(nc,3)?)?;
    let edge = input.reals::<T>(product(ne,3)?)?;
    let coe = input.indices(product(ne,2)?)?;
    let eoc = input.indices(product(nc,me)?)?;
    let counts = input.indices(nc)?;
    if counts.iter().any(|&n| n < 1 || n > me as i64) {
        return Err("nEdgesOnCell is inconsistent with edgesOnCell".into());
    }
    let (normal,vertical,plane) = if mode == 2 {
        (input.reals::<T>(product(ne,3)?)?,Vec::new(),input.reals::<T>(product(nc,6)?)?)
    } else { vector_geometry(&cell,&edge,&coe,&eoc,nc,ne,me,sphere,periodic,xp,yp)? };
    if input.cursor != input.bytes.len() { return Err("trailing geometry input bytes".into()); }
    let mut coefficients = if mode != 1 { vec![T::zero();product(product(nc,me)?,3)?] } else { Vec::new() };
    if mode != 1 {
        coefficients.par_chunks_mut(me*3).take(solved).enumerate().try_for_each(|(i,out)| {
            let count = counts[i] as usize;
            let mut source = vec![T::zero();count*3];
            let mut vectors = vec![T::zero();count*3];
            let destination = &cell[i*3..i*3+3];
            let mut alpha = T::zero();
            for j in 0..count {
                let index = eoc[i*me+j];
                if index < 0 || index >= ne as i64 { return Err("active edgesOnCell entry is out of range".into()); }
                let index = index as usize;
                source[j*3..j*3+3].copy_from_slice(&edge[index*3..index*3+3]);
                vectors[j*3..j*3+3].copy_from_slice(&normal[index*3..index*3+3]);
                if periodic {
                    source[j*3] = near(source[j*3],destination[0],xp);
                    source[j*3+1] = near(source[j*3+1],destination[1],yp);
                }
                let dx = destination[0]-source[j*3];
                let dy = destination[1]-source[j*3+1];
                let dz = destination[2]-source[j*3+2];
                alpha = alpha+((dx*dx+dy*dy)+dz*dz).sqrt();
            }
            alpha = alpha/T::count(count);
            if alpha == T::zero() || !alpha.finite() { return Err("invalid RBF length scale".into()); }
            reconstruct_cell(&source,&vectors,destination,&plane[i*6..i*6+6],alpha,count,out)
        })?;
    }
    // Encoded on every core into one buffer, then written once: the same bytes
    // in the same order as one encode call per value.
    for values in if mode == 2 { vec![&coefficients] } else { vec![&normal,&vertical,&plane,&coefficients] } {
        let mut bytes = vec![0u8; values.len() * T::WIDTH];
        bytes.par_chunks_mut(T::WIDTH).zip(values.par_iter())
            .try_for_each(|(slot, &value)| value.encode(&mut &mut slot[..]))?;
        output.write_all(&bytes).map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// Read the versioned little-endian array protocol and write exact raw arrays.
pub fn run(mut reader: impl Read, mut writer: impl Write) -> Result<(), String> {
    let mut bytes = Vec::new();
    reader.read_to_end(&mut bytes).map_err(|e|e.to_string())?;
    let mut input = Input { bytes: &bytes, cursor: 0 };
    if input.take(8)? != b"HEXGEO1\0" { return Err("expected HEXGEO1 geometry protocol".into()); }
    let nc = input.count()?;
    let ne = input.count()?;
    let me = input.count()?;
    let solved = input.count()?;
    if nc == 0 || ne == 0 || me == 0 || solved > nc { return Err("invalid geometry dimensions or solve count".into()); }
    let flags = input.take(4)?;
    let (width,sphere,periodic,mode) = (flags[0],flags[1],flags[2],flags[3]);
    if sphere > 1 || periodic > 1 || mode > 2 { return Err("invalid geometry protocol flags".into()); }
    match width {
        4 => process::<f32>(&mut input,&mut writer,nc,ne,me,solved,sphere==1,periodic==1,mode),
        8 => process::<f64>(&mut input,&mut writer,nc,ne,me,solved,sphere==1,periodic==1,mode),
        _ => Err("geometry protocol requires float32 or float64".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn pivot_ties_and_singular_matrix() {
        assert_eq!(solve(&[2.0_f64,1.0,1.0,3.0],&[1.0,2.0],2).unwrap(),vec![0.2,0.6]);
        assert!(solve(&[0.0_f32;4],&[0.0;2],2).unwrap_err().contains("singular"));
    }
    #[test]
    fn truncated_protocol_is_refused() {
        assert!(run(&b"HEXGEO1\0"[..],Vec::new()).unwrap_err().contains("truncated"));
    }
}
