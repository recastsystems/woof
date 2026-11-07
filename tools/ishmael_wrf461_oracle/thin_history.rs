//! Small CF evidence planes from original WRF history; source files stay untouched.
//! netCDF reads/writes and weather transformations remain in Rust. Two workers
//! share the cell arithmetic; only the three requested ice masses are read.
use std::{
    ffi::CString,
    os::raw::{c_char, c_int},
};
#[link(name = "netcdf")]
unsafe extern "C" {
    fn nc_open(p: *const c_char, m: c_int, id: *mut c_int) -> c_int;
    fn nc_close(id: c_int) -> c_int;
    fn nc_create(p: *const c_char, m: c_int, id: *mut c_int) -> c_int;
    fn nc_inq_varid(id: c_int, n: *const c_char, v: *mut c_int) -> c_int;
    fn nc_inq_varndims(id: c_int, v: c_int, n: *mut c_int) -> c_int;
    fn nc_inq_vardimid(id: c_int, v: c_int, d: *mut c_int) -> c_int;
    fn nc_inq_dimlen(id: c_int, d: c_int, n: *mut usize) -> c_int;
    fn nc_get_var_float(id: c_int, v: c_int, p: *mut f32) -> c_int;
    fn nc_def_dim(id: c_int, n: *const c_char, len: usize, d: *mut c_int) -> c_int;
    fn nc_def_var(
        id: c_int,
        n: *const c_char,
        t: c_int,
        nd: c_int,
        d: *const c_int,
        v: *mut c_int,
    ) -> c_int;
    fn nc_put_att_text(
        id: c_int,
        v: c_int,
        n: *const c_char,
        len: usize,
        p: *const c_char,
    ) -> c_int;
    fn nc_put_var_float(id: c_int, v: c_int, p: *const f32) -> c_int;
    fn nc_put_var_double(id: c_int, v: c_int, p: *const f64) -> c_int;
    fn nc_enddef(id: c_int) -> c_int;
}
fn c(s: &str) -> CString {
    CString::new(s).unwrap()
}
fn ck(r: c_int) {
    assert_eq!(r, 0, "netCDF {r}")
}
unsafe fn read(id: c_int, name: &str) -> Option<(Vec<f32>, Vec<usize>)> {
    unsafe {
        let mut v = 0;
        if nc_inq_varid(id, c(name).as_ptr(), &mut v) != 0 {
            return None;
        }
        let mut n = 0;
        ck(nc_inq_varndims(id, v, &mut n));
        let mut d = vec![0; n as usize];
        ck(nc_inq_vardimid(id, v, d.as_mut_ptr()));
        let mut sizes = vec![];
        for dim in d {
            let mut z = 0;
            ck(nc_inq_dimlen(id, dim, &mut z));
            sizes.push(z)
        }
        let count: usize = sizes.iter().product();
        assert!(count < 100_000_000);
        let mut a = vec![0.; count];
        ck(nc_get_var_float(id, v, a.as_mut_ptr()));
        Some((a, sizes))
    }
}
unsafe fn att(id: c_int, v: c_int, n: &str, s: &str) {
    ck(unsafe { nc_put_att_text(id, v, c(n).as_ptr(), s.len(), c(s).as_ptr()) })
}
fn main() {
    let a: Vec<_> = std::env::args().collect();
    assert_eq!(a.len(), 4, "history.nc evidence.nc epoch_seconds");
    assert_ne!(a[1], a[2]);
    let epoch: f64 = a[3].parse().unwrap();
    unsafe {
        let mut id = 0;
        ck(nc_open(c(&a[1]).as_ptr(), 0, &mut id));
        let (t, shape) = read(id, "T2").unwrap();
        assert_eq!(shape.len(), 3);
        assert_eq!(shape[0], 1, "one frame per file");
        let (ny, nx) = (shape[1], shape[2]);
        let cells = nx * ny;
        let lat = read(id, "XLAT").unwrap().0;
        let lon = read(id, "XLONG").unwrap().0;
        let rain = read(id, "RAINNC").unwrap().0;
        let rainc = read(id, "RAINC").map(|x| x.0).unwrap_or(vec![0.; cells]);
        let mut temp = vec![0.; cells];
        let mut precip = vec![0.; cells];
        std::thread::scope(|s| {
            let chunk = cells.div_ceil(2);
            for (i, (to, po)) in temp
                .chunks_mut(chunk)
                .zip(precip.chunks_mut(chunk))
                .enumerate()
            {
                let t = &t;
                let rain = &rain;
                let rainc = &rainc;
                s.spawn(move || {
                    for j in 0..to.len() {
                        let k = i * chunk + j;
                        to[j] = (t[k] - 273.15) * 1.8 + 32.;
                        po[j] = rain[k] + rainc[k];
                    }
                });
            }
        });
        let iwp = if let Some((iwp, _)) = read(id, "IWP") {
            iwp
        } else {
            let ice: Vec<_> = ["QICE", "QICE2", "QICE3"]
                .iter()
                .map(|n| read(id, n).expect("all three native ice masses").0)
                .collect();
            let nz = ice[0].len() / cells;
            assert!(ice.iter().all(|v| v.len() == nz * cells));
            let mu = read(id, "MU").unwrap().0;
            let mub = read(id, "MUB").unwrap().0;
            let c1 = read(id, "C1H").unwrap().0;
            let c2 = read(id, "C2H").unwrap().0;
            let eta = read(id, "ZNW").unwrap().0;
            assert_eq!(eta.len(), nz + 1);
            let mut result = vec![0.; cells];
            std::thread::scope(|s| {
                let chunk = cells.div_ceil(2);
                for (part, out) in result.chunks_mut(chunk).enumerate() {
                    let (ice, mu, mub, c1, c2, eta) = (&ice, &mu, &mub, &c1, &c2, &eta);
                    s.spawn(move || {
                        for (j, v) in out.iter_mut().enumerate() {
                            let cell = part * chunk + j;
                            let mut sum = 0f64;
                            for k in 0..nz {
                                let mass = (f64::from(c1[k])
                                    * (f64::from(mu[cell]) + f64::from(mub[cell]))
                                    + f64::from(c2[k]))
                                    * (f64::from(eta[k]) - f64::from(eta[k + 1]))
                                    / 9.81;
                                let q: f64 = ice
                                    .iter()
                                    .map(|x| f64::from(x[k * cells + cell].max(0.)))
                                    .sum();
                                sum += q * mass * 1000.;
                            }
                            *v = sum as f32;
                        }
                    });
                }
            });
            result
        };
        ck(nc_close(id));
        let mut out = 0;
        ck(nc_create(c(&a[2]).as_ptr(), 512, &mut out));
        let mut dims = [0; 3];
        for (i, (n, l)) in [("time", 1), ("y", ny), ("x", nx)].iter().enumerate() {
            ck(nc_def_dim(out, c(n).as_ptr(), *l, &mut dims[i]));
        }
        att(out, -1, "Conventions", "CF-1.8");
        att(out, -1, "source", &a[1]);
        let mut time = 0;
        ck(nc_def_var(
            out,
            c("time").as_ptr(),
            6,
            1,
            dims.as_ptr(),
            &mut time,
        ));
        att(out, time, "units", "seconds since 1970-01-01 00:00:00 UTC");
        att(out, time, "standard_name", "time");
        let mut vars = [].to_vec();
        for (n, units, std, data, is2d) in [
            ("lat", "degrees_north", "latitude", &lat, true),
            ("lon", "degrees_east", "longitude", &lon, true),
            ("T2_F", "F", "air_temperature", &temp, false),
            ("RAIN_TOTAL", "mm", "", &precip, false),
            ("IWP", "g m-2", "", &iwp, false),
        ] {
            let mut v = 0;
            let d = if is2d { &dims[1..] } else { &dims[..] };
            ck(nc_def_var(
                out,
                c(n).as_ptr(),
                5,
                d.len() as c_int,
                d.as_ptr(),
                &mut v,
            ));
            att(out, v, "units", units);
            if !std.is_empty() {
                att(out, v, "standard_name", std)
            }
            if !is2d {
                att(out, v, "coordinates", "lat lon")
            }
            vars.push((v, data));
        }
        ck(nc_enddef(out));
        ck(nc_put_var_double(out, time, &epoch));
        for (v, data) in vars {
            ck(nc_put_var_float(out, v, data.as_ptr()));
        }
        ck(nc_close(out));
        println!(
            "{{\"cells\":{cells},\"temperature_finite\":{},\"rain_max_mm\":{},\"iwp_max_g_m2\":{}}}",
            temp.iter().filter(|v| v.is_finite()).count(),
            precip.iter().copied().fold(0f32, f32::max),
            iwp.iter().copied().fold(0f32, f32::max)
        );
    }
}
