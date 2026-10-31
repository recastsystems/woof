//! Diagnostic inventory only: no field is normalized, excluded or rewritten.
//! Uses f0 word_compare's native typed NetCDF FFI/slab convention. Python only
//! orchestrates this binary; IEEE classification and field decoding stay Rust.
use std::{ffi::{CString, CStr, c_char, c_int, c_void}, ptr};

#[link(name = "netcdf")]
unsafe extern "C" {
    fn nc_open(path: *const c_char, mode: c_int, id: *mut c_int) -> c_int;
    fn nc_close(id: c_int) -> c_int;
    fn nc_inq_nvars(id: c_int, n: *mut c_int) -> c_int;
    fn nc_inq_var(id: c_int, v: c_int, name: *mut c_char, t: *mut c_int,
        nd: *mut c_int, dims: *mut c_int, attrs: *mut c_int) -> c_int;
    fn nc_inq_dim(id: c_int, d: c_int, name: *mut c_char, len: *mut usize) -> c_int;
    fn nc_get_vara(id: c_int, v: c_int, start: *const usize, count: *const usize, data: *mut c_void) -> c_int;
    fn nc_strerror(code: c_int) -> *const c_char;
}
fn check(code: c_int) -> Result<(), String> {
    if code == 0 { Ok(()) } else {
        Err(unsafe { CStr::from_ptr(nc_strerror(code)) }.to_string_lossy().into_owned())
    }
}
struct File(c_int);
impl Drop for File { fn drop(&mut self) { unsafe { nc_close(self.0); } } }
fn quote(s: &str) -> String {
    let mut out = String::from("\"");
    for c in s.chars() { match c {
        '"' => out.push_str("\\\""), '\\' => out.push_str("\\\\"), '\n' => out.push_str("\\n"),
        '\r' => out.push_str("\\r"), '\t' => out.push_str("\\t"),
        c if c < ' ' => out.push_str(&format!("\\u{:04x}", c as u32)), c => out.push(c),
    }}
    out.push('"'); out
}
fn nonfinite(bits: u64, ty: c_int) -> Option<&'static str> {
    let (exponent, mantissa, sign) = if ty == 5 {
        (0x7f800000, 0x007fffff, 0x80000000)
    } else { (0x7ff0000000000000, 0x000fffffffffffff, 0x8000000000000000) };
    if bits & exponent != exponent { None }
    else if bits & mantissa != 0 { Some("nan") }
    else if bits & sign != 0 { Some("negative_infinity") }
    else { Some("positive_infinity") }
}
fn inventory(path: &str) -> Result<String, String> {
    if cfg!(target_endian = "big") { return Err("raw word inventory requires little-endian host".into()); }
    let p = CString::new(path).map_err(|e| e.to_string())?; let mut id = 0;
    check(unsafe { nc_open(p.as_ptr(), 0, &mut id) })?; let file = File(id);
    let mut nvars = 0; check(unsafe { nc_inq_nvars(file.0, &mut nvars) })?;
    let mut fields = Vec::new();
    for var in 0..nvars {
        let mut name = [0_i8; 257]; let mut ty = 0; let mut nd = 0; let mut dims = [0; 1024];
        check(unsafe { nc_inq_var(file.0, var, name.as_mut_ptr(), &mut ty, &mut nd, dims.as_mut_ptr(), ptr::null_mut()) })?;
        if ty != 5 && ty != 6 { continue; }
        if nd < 0 || nd > 1024 { return Err("unsupported variable rank".into()); }
        let name = unsafe { CStr::from_ptr(name.as_ptr()) }.to_string_lossy(); let mut shape = Vec::new();
        for &dim in &dims[..nd as usize] {
            let mut len = 0; check(unsafe { nc_inq_dim(file.0, dim, ptr::null_mut(), &mut len) })?; shape.push(len);
        }
        let total = shape.iter().try_fold(1_usize, |n, &d| n.checked_mul(d)).ok_or("shape overflow")?;
        let axis = shape.iter().enumerate().max_by_key(|&(_, d)| d).map(|(i, _)| i);
        let rows = axis.map(|i| shape[i]).unwrap_or(1);
        let mut nan = 0_usize; let mut positive = 0_usize; let mut negative = 0_usize; let mut first = None;
        if total > 0 {
            let width = total / rows; let batch = (262144 / width.max(1)).max(1).min(rows);
            let mut start = vec![0; shape.len()]; let mut count = shape.clone(); let mut offset = 0;
            while offset < rows {
                if let Some(i) = axis { start[i] = offset; count[i] = batch.min(rows - offset); }
                let n = count.iter().product::<usize>(); let size = if ty == 5 { 4 } else { 8 };
                let mut data = vec![0_u64; (n * size).div_ceil(8)];
                check(unsafe { nc_get_vara(file.0, var, start.as_ptr(), count.as_ptr(), data.as_mut_ptr().cast()) })?;
                for index in 0..n {
                    let bits = if ty == 5 { (unsafe { *data.as_ptr().cast::<u32>().add(index) }) as u64 } else { data[index] };
                    if let Some(kind) = nonfinite(bits, ty) {
                        match kind { "nan" => nan += 1, "positive_infinity" => positive += 1, _ => negative += 1 }
                        if first.is_none() {
                            let mut coordinates = start.clone(); let mut linear = index;
                            for d in (0..count.len()).rev() { coordinates[d] += linear % count[d]; linear /= count[d]; }
                            first = Some(format!("{{\"index\":{:?},\"bits\":\"{bits:016x}\",\"kind\":{}}}", coordinates, quote(kind)));
                        }
                    }
                }
                offset += batch;
            }
        }
        fields.push(format!("{{\"field\":{},\"netcdf_type\":{ty},\"shape\":{:?},\"words\":{total},\"nan\":{nan},\"positive_infinity\":{positive},\"negative_infinity\":{negative},\"first_nonfinite\":{}}}", quote(&name), shape, first.unwrap_or_else(|| "null".into())));
    }
    Ok(format!("{{\"schema\":\"mpas.rrfs-native-field-finiteness/v1\",\"diagnostic_only\":true,\"path\":{},\"fields\":[{}]}}", quote(path), fields.join(",")))
}
fn main() {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 2 { eprintln!("usage: mpas-rrfs-field-inventory INPUT.nc"); std::process::exit(2); }
    match inventory(&args[1]) { Ok(result) => println!("{result}"), Err(error) => { eprintln!("{error}"); std::process::exit(2); } }
}
#[cfg(test)]
mod tests {
    use super::nonfinite;
    #[test]
    fn classifies_raw_ieee_patterns_without_changing_nan_payloads() {
        for ty in [5, 6] {
            let (nan, infinity, negative, subnormal, negative_zero) = if ty == 5 {
                (0x7fa00001, 0x7f800000, 0xff800000, 1, 0x80000000)
            } else { (0x7ff0000000000001, 0x7ff0000000000000, 0xfff0000000000000, 1, 0x8000000000000000) };
            assert_eq!(nonfinite(nan, ty), Some("nan"));
            assert_eq!(nonfinite(infinity, ty), Some("positive_infinity"));
            assert_eq!(nonfinite(negative, ty), Some("negative_infinity"));
            assert_eq!(nonfinite(subnormal, ty), None); assert_eq!(nonfinite(negative_zero, ty), None);
        }
    }
}
