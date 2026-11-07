//! Explicit cold-start conversion for the mp55 CLI receipt.
//! Copy active donor variables word for word, then add absent native moments as zeros.
//! The donor contains warm-rain fields and no ice; this is not a restart.
use std::{
    ffi::CString,
    os::raw::{c_char, c_int},
    path::Path,
};
#[link(name = "netcdf")]
unsafe extern "C" {
    fn nc_create(path: *const c_char, mode: c_int, id: *mut c_int) -> c_int;
    fn nc_inq(
        id: c_int,
        nd: *mut c_int,
        nv: *mut c_int,
        na: *mut c_int,
        unlim: *mut c_int,
    ) -> c_int;
    fn nc_inq_dim(id: c_int, dim: c_int, name: *mut c_char, len: *mut usize) -> c_int;
    fn nc_def_dim(id: c_int, name: *const c_char, len: usize, dim: *mut c_int) -> c_int;
    fn nc_inq_attname(id: c_int, var: c_int, num: c_int, name: *mut c_char) -> c_int;
    fn nc_copy_att(
        input: c_int,
        var: c_int,
        name: *const c_char,
        output: c_int,
        target: c_int,
    ) -> c_int;
    fn nc_inq_var(
        id: c_int,
        var: c_int,
        name: *mut c_char,
        kind: *mut c_int,
        rank: *mut c_int,
        dims: *mut c_int,
        attrs: *mut c_int,
    ) -> c_int;
    fn nc_inq_type(id: c_int, kind: c_int, name: *mut c_char, size: *mut usize) -> c_int;
    fn nc_get_var(id: c_int, var: c_int, data: *mut std::ffi::c_void) -> c_int;
    fn nc_put_vara(
        id: c_int,
        var: c_int,
        start: *const usize,
        count: *const usize,
        data: *const std::ffi::c_void,
    ) -> c_int;
    fn nc_open(path: *const c_char, mode: c_int, id: *mut c_int) -> c_int;
    fn nc_close(id: c_int) -> c_int;
    fn nc_redef(id: c_int) -> c_int;
    fn nc_enddef(id: c_int) -> c_int;
    fn nc_inq_dimid(id: c_int, name: *const c_char, dim: *mut c_int) -> c_int;
    fn nc_inq_dimlen(id: c_int, dim: c_int, len: *mut usize) -> c_int;
    fn nc_inq_varid(id: c_int, name: *const c_char, var: *mut c_int) -> c_int;
    fn nc_get_vara_float(
        id: c_int,
        var: c_int,
        start: *const usize,
        count: *const usize,
        data: *mut f32,
    ) -> c_int;
    fn nc_def_var(
        id: c_int,
        name: *const c_char,
        kind: c_int,
        ndim: c_int,
        dims: *const c_int,
        var: *mut c_int,
    ) -> c_int;
    fn nc_put_att_text(
        id: c_int,
        var: c_int,
        name: *const c_char,
        len: usize,
        text: *const c_char,
    ) -> c_int;
    fn nc_put_att_int(
        id: c_int,
        var: c_int,
        name: *const c_char,
        kind: c_int,
        len: usize,
        val: *const c_int,
    ) -> c_int;
    fn nc_put_vara_float(
        id: c_int,
        var: c_int,
        start: *const usize,
        count: *const usize,
        data: *const f32,
    ) -> c_int;
}
fn c(s: &str) -> CString {
    CString::new(s).unwrap()
}
fn check(rc: c_int) {
    assert_eq!(rc, 0, "netCDF status {rc}");
}
fn filtered_copy(source: &str, destination: &str) {
    unsafe {
        let mut input = 0;
        check(nc_open(c(source).as_ptr(), 0, &mut input));
        let (mut nd, mut nv, mut na, mut unlim) = (0, 0, 0, -1);
        check(nc_inq(input, &mut nd, &mut nv, &mut na, &mut unlim));
        let mut output = 0;
        check(nc_create(c(destination).as_ptr(), 512, &mut output));
        let mut lengths = Vec::new();
        for dim in 0..nd {
            let mut name = [0i8; 257];
            let mut len = 0;
            check(nc_inq_dim(input, dim, name.as_mut_ptr(), &mut len));
            lengths.push(len);
            let mut target = 0;
            check(nc_def_dim(
                output,
                name.as_ptr(),
                if dim == unlim { 0 } else { len },
                &mut target,
            ));
            assert_eq!(target, dim);
        }
        for att in 0..na {
            let mut name = [0i8; 257];
            check(nc_inq_attname(input, -1, att, name.as_mut_ptr()));
            check(nc_copy_att(input, -1, name.as_ptr(), output, -1));
        }
        let mut copies = Vec::new();
        for var in 0..nv {
            let (mut kind, mut rank, mut attrs) = (0, 0, 0);
            let mut name = [0i8; 257];
            let mut dims = [0i32; 32];
            check(nc_inq_var(
                input,
                var,
                name.as_mut_ptr(),
                &mut kind,
                &mut rank,
                dims.as_mut_ptr(),
                &mut attrs,
            ));
            let text = std::ffi::CStr::from_ptr(name.as_ptr()).to_str().unwrap();
            let counts: Vec<usize> = dims[..rank as usize]
                .iter()
                .map(|d| lengths[*d as usize])
                .collect();
            let count = counts.iter().product::<usize>();
            if ["QSNOW", "QGRAUP", "QNSNOW", "QNGRAUPEL"].contains(&text) {
                assert_eq!(kind, 5);
                let mut words = vec![0f32; count];
                check(nc_get_vara_float(
                    input,
                    var,
                    [0usize; 32].as_ptr(),
                    counts.as_ptr(),
                    words.as_mut_ptr(),
                ));
                assert!(
                    words.iter().all(|v| *v == 0.0),
                    "nonempty inactive donor category {text}"
                );
                println!("excluded verified zero inactive donor category {text}: {count} words");
                continue;
            }
            let mut target = 0;
            check(nc_def_var(
                output,
                name.as_ptr(),
                kind,
                rank,
                dims.as_ptr(),
                &mut target,
            ));
            for att in 0..attrs {
                let mut name = [0i8; 257];
                check(nc_inq_attname(input, var, att, name.as_mut_ptr()));
                check(nc_copy_att(input, var, name.as_ptr(), output, target));
            }
            let mut width = 0;
            check(nc_inq_type(input, kind, std::ptr::null_mut(), &mut width));
            copies.push((var, target, counts, count * width));
        }
        check(nc_enddef(output));
        for (var, target, counts, bytes) in copies {
            let mut words = vec![0u64; bytes.div_ceil(8)];
            check(nc_get_var(input, var, words.as_mut_ptr().cast()));
            check(nc_put_vara(
                output,
                target,
                [0usize; 32].as_ptr(),
                counts.as_ptr(),
                words.as_ptr().cast(),
            ));
        }
        check(nc_close(output));
        check(nc_close(input));
    }
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert_eq!(args.len(), 3, "donor.nc native.nc");
    assert_ne!(Path::new(&args[1]), Path::new(&args[2]));
    filtered_copy(&args[1], &args[2]);
    unsafe {
        let mut id = 0;
        check(nc_open(c(&args[2]).as_ptr(), 1, &mut id));
        let mut dims = [0; 4];
        let mut count = [0; 4];
        for (i, name) in ["Time", "bottom_top", "south_north", "west_east"]
            .iter()
            .enumerate()
        {
            check(nc_inq_dimid(id, c(name).as_ptr(), &mut dims[i]));
            check(nc_inq_dimlen(id, dims[i], &mut count[i]));
        }
        assert!(count.iter().all(|n| *n > 0));
        assert!(count.iter().product::<usize>() < 10_000_000);
        check(nc_redef(id));
        let mut added = Vec::new();
        for name in [
            "QICE", "QICE2", "QICE3", "QNRAIN", "QNICE", "QNICE2", "QNICE3", "QVOLI", "QAOLI",
            "QVOLI2", "QAOLI2", "QVOLI3", "QAOLI3",
        ] {
            let mut var = 0;
            if nc_inq_varid(id, c(name).as_ptr(), &mut var) == 0 {
                continue;
            }
            check(nc_def_var(
                id,
                c(name).as_ptr(),
                5,
                4,
                dims.as_ptr(),
                &mut var,
            ));
            let units = if name.starts_with("QICE") {
                "kg kg-1"
            } else if name.starts_with("QN") {
                "# kg-1"
            } else {
                "m3 kg-1"
            };
            check(nc_put_att_text(
                id,
                var,
                c("units").as_ptr(),
                units.len(),
                c(units).as_ptr(),
            ));
            added.push((name, var));
        }
        check(nc_put_att_int(id, -1, c("MP_PHYSICS").as_ptr(), 4, 1, &55));
        check(nc_enddef(id));
        let zeros = vec![0f32; count.iter().product()];
        for (name, var) in added {
            check(nc_put_vara_float(
                id,
                var,
                [0; 4].as_ptr(),
                count.as_ptr(),
                zeros.as_ptr(),
            ));
            println!("added {name}: {} zero words", zeros.len());
        }
        check(nc_close(id));
    }
}
