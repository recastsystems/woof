//! Preserve native NetCDF attribute types after the Rust-reader fixture copy.
//! This proof-preparation tool copies metadata only, never weather fields.
use std::ffi::{CStr, CString, c_char, c_int};
#[link(name = "netcdf")]
unsafe extern "C" {
    fn nc_open(path: *const c_char, mode: c_int, id: *mut c_int) -> c_int;
    fn nc_close(id: c_int) -> c_int;
    fn nc_redef(id: c_int) -> c_int;
    fn nc_enddef(id: c_int) -> c_int;
    fn nc_inq(id: c_int, nd: *mut c_int, nv: *mut c_int, na: *mut c_int, unlim: *mut c_int) -> c_int;
    fn nc_inq_varname(id: c_int, var: c_int, name: *mut c_char) -> c_int;
    fn nc_inq_varid(id: c_int, name: *const c_char, var: *mut c_int) -> c_int;
    fn nc_inq_varnatts(id: c_int, var: c_int, na: *mut c_int) -> c_int;
    fn nc_inq_attname(id: c_int, var: c_int, at: c_int, name: *mut c_char) -> c_int;
    fn nc_copy_att(src: c_int, sv: c_int, name: *const c_char, dst: c_int, dv: c_int) -> c_int;
    fn nc_put_att_int(id: c_int, var: c_int, name: *const c_char, ty: c_int, len: usize, val: *const c_int) -> c_int;
    fn nc_strerror(error: c_int) -> *const c_char;
}
fn check(code: c_int) {
    assert_eq!(code, 0, "{}", unsafe { CStr::from_ptr(nc_strerror(code)) }.to_string_lossy());
}
unsafe fn copy_attrs(src: c_int, sv: c_int, dst: c_int, dv: c_int, n: c_int) {
    for at in 0..n {
        let mut name = [0 as c_char; 257];
        unsafe {
            check(nc_inq_attname(src, sv, at, name.as_mut_ptr()));
            check(nc_copy_att(src, sv, name.as_ptr(), dst, dv));
        }
    }
}
fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert!(args.len() >= 4, "usage: restore_attributes SOURCE TARGET MP_PHYSICS [OMITTED_VARIABLE ...]");
    let source = CString::new(args[1].as_str()).unwrap();
    let target = CString::new(args[2].as_str()).unwrap();
    let mp: c_int = args[3].parse().unwrap();
    let (mut src, mut dst, mut vars, mut attrs) = (0, 0, 0, 0);
    unsafe {
        check(nc_open(source.as_ptr(), 0, &mut src));
        check(nc_open(target.as_ptr(), 1, &mut dst));
        check(nc_redef(dst));
        check(nc_inq(src, std::ptr::null_mut(), &mut vars, &mut attrs, std::ptr::null_mut()));
        copy_attrs(src, -1, dst, -1, attrs);
        for sv in 0..vars {
            let mut name = [0 as c_char; 257];
            let (mut dv, mut n) = (0, 0);
            check(nc_inq_varname(src, sv, name.as_mut_ptr()));
            let variable_name = CStr::from_ptr(name.as_ptr()).to_str().unwrap();
            if args[4..].iter().any(|omitted| omitted == variable_name) {
                continue;
            }
            check(nc_inq_varid(dst, name.as_ptr(), &mut dv));
            check(nc_inq_varnatts(src, sv, &mut n));
            copy_attrs(src, sv, dst, dv, n);
        }
        check(nc_put_att_int(dst, -1, c"MP_PHYSICS".as_ptr(), 4, 1, &mp));
        check(nc_enddef(dst));
        check(nc_close(dst));
        check(nc_close(src));
    }
    println!("restored source attribute types for {vars} variables; MP_PHYSICS={mp}");
}
