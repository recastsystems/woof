//! Read-only logical NetCDF word comparison; arithmetic runs on Rust workers.
use std::{
    collections::BTreeMap,
    ffi::{CStr, CString},
    os::raw::{c_char, c_int, c_void},
};
#[link(name = "netcdf")]
unsafe extern "C" {
    fn nc_open(p: *const c_char, mode: c_int, id: *mut c_int) -> c_int;
    fn nc_close(id: c_int) -> c_int;
    fn nc_inq_nvars(id: c_int, n: *mut c_int) -> c_int;
    fn nc_inq_var(
        id: c_int,
        var: c_int,
        name: *mut c_char,
        kind: *mut c_int,
        ndim: *mut c_int,
        dims: *mut c_int,
        atts: *mut c_int,
    ) -> c_int;
    fn nc_inq_dimlen(id: c_int, dim: c_int, len: *mut usize) -> c_int;
    fn nc_inq_type(id: c_int, kind: c_int, name: *mut c_char, size: *mut usize) -> c_int;
    fn nc_get_var(id: c_int, var: c_int, data: *mut c_void) -> c_int;
}
fn check(v: c_int) {
    assert_eq!(v, 0, "netCDF status {v}");
}
struct Var {
    id: c_int,
    kind: c_int,
    words: usize,
    size: usize,
}
fn variables(id: c_int) -> BTreeMap<String, Var> {
    unsafe {
        let mut n = 0;
        check(nc_inq_nvars(id, &mut n));
        let mut vars = BTreeMap::new();
        for var in 0..n {
            let mut name = [0 as c_char; 257];
            let (mut kind, mut ndim, mut atts) = (0, 0, 0);
            let mut dims = [0; 32];
            check(nc_inq_var(
                id,
                var,
                name.as_mut_ptr(),
                &mut kind,
                &mut ndim,
                dims.as_mut_ptr(),
                &mut atts,
            ));
            let mut words = 1;
            for d in &dims[..ndim as usize] {
                let mut len = 0;
                check(nc_inq_dimlen(id, *d, &mut len));
                words *= len;
            }
            assert!(kind != 12, "NC_STRING requires a separate string reader");
            let mut size = 0;
            check(nc_inq_type(id, kind, std::ptr::null_mut(), &mut size));
            vars.insert(
                CStr::from_ptr(name.as_ptr()).to_str().unwrap().to_owned(),
                Var {
                    id: var,
                    kind,
                    words,
                    size,
                },
            );
        }
        vars
    }
}
fn read(id: c_int, v: &Var) -> Vec<u8> {
    let mut b = vec![0; v.words * v.size];
    check(unsafe { nc_get_var(id, v.id, b.as_mut_ptr().cast()) });
    b
}
fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert_eq!(args.len(), 3, "left.nc right.nc");
    unsafe {
        let (mut left, mut right) = (0, 0);
        check(nc_open(
            CString::new(args[1].as_str()).unwrap().as_ptr(),
            0,
            &mut left,
        ));
        check(nc_open(
            CString::new(args[2].as_str()).unwrap().as_ptr(),
            0,
            &mut right,
        ));
        let a = variables(left);
        let b = variables(right);
        let mut rows = Vec::new();
        let (mut total, mut differing) = (0usize, 0usize);
        for (name, v) in &a {
            if let Some(w) = b.get(name) {
                assert_eq!(
                    (v.kind, v.words, v.size),
                    (w.kind, w.words, w.size),
                    "shape/type mismatch {name}"
                );
                let av = read(left, v);
                let bv = read(right, w);
                let parts = std::thread::scope(|scope| {
                    let mut jobs = Vec::new();
                    let chunk = v.words.div_ceil(2).max(1);
                    for first in (0..v.words).step_by(chunk) {
                        let end = (first + chunk).min(v.words);
                        let aa = &av[first * v.size..end * v.size];
                        let bb = &bv[first * v.size..end * v.size];
                        let size = v.size;
                        let kind = v.kind;
                        jobs.push(scope.spawn(move || {
                            let (mut diff, mut finite_a, mut finite_b) = (0usize, 0usize, 0usize);
                            let (mut max_abs, mut sum_abs, mut max_a, mut max_b) =
                                (0f64, 0f64, f64::NEG_INFINITY, f64::NEG_INFINITY);
                            for (x, y) in aa.chunks_exact(size).zip(bb.chunks_exact(size)) {
                                diff += usize::from(x != y);
                                if kind == 5 {
                                    let x = f32::from_ne_bytes(x.try_into().unwrap()) as f64;
                                    let y = f32::from_ne_bytes(y.try_into().unwrap()) as f64;
                                    finite_a += usize::from(x.is_finite());
                                    finite_b += usize::from(y.is_finite());
                                    if x.is_finite() {
                                        max_a = max_a.max(x);
                                    }
                                    if y.is_finite() {
                                        max_b = max_b.max(y);
                                    }
                                    if x.is_finite() && y.is_finite() {
                                        let d = (x - y).abs();
                                        max_abs = max_abs.max(d);
                                        sum_abs += d;
                                    }
                                }
                            }
                            (diff, finite_a, finite_b, max_abs, sum_abs, max_a, max_b)
                        }));
                    }
                    jobs.into_iter()
                        .map(|j| j.join().unwrap())
                        .collect::<Vec<_>>()
                });
                let diff: usize = parts.iter().map(|p| p.0).sum();
                total += v.words;
                differing += diff;
                let extra = if v.kind == 5 {
                    format!(
                        ",\"finite_left\":{},\"finite_right\":{},\"max_abs\":{},\"mean_abs\":{},\"max_left\":{},\"max_right\":{}",
                        parts.iter().map(|p| p.1).sum::<usize>(),
                        parts.iter().map(|p| p.2).sum::<usize>(),
                        parts.iter().map(|p| p.3).fold(0.0, f64::max),
                        parts.iter().map(|p| p.4).sum::<f64>() / v.words.max(1) as f64,
                        parts.iter().map(|p| p.5).fold(f64::NEG_INFINITY, f64::max),
                        parts.iter().map(|p| p.6).fold(f64::NEG_INFINITY, f64::max)
                    )
                } else {
                    String::new()
                };
                rows.push(format!(
                    "\"{name}\":{{\"words\":{},\"differing\":{diff}{extra}}}",
                    v.words
                ));
            }
        }
        check(nc_close(left));
        check(nc_close(right));
        let only_a: Vec<_> = a.keys().filter(|k| !b.contains_key(*k)).collect();
        let only_b: Vec<_> = b.keys().filter(|k| !a.contains_key(*k)).collect();
        println!(
            "{{\"words\":{total},\"differing\":{differing},\"only_left\":{only_a:?},\"only_right\":{only_b:?},\"fields\":{{{}}}}}",
            rows.join(",")
        );
    }
}
