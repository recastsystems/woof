//! Original Rust NetCDF boundary. Only the netCDF-C container ABI is linked.
//! Numeric transforms use scoped worker threads. Python receives device-ready bytes.
mod spectral_io;
use std::{
    ffi::{c_char, c_int, CStr, CString},
    fs,
    io::Write,
    path::Path,
};
#[link(name = "netcdf")]
unsafe extern "C" {
    fn nc_open(p: *const c_char, mode: c_int, id: *mut c_int) -> c_int;
    fn nc_create(p: *const c_char, mode: c_int, id: *mut c_int) -> c_int;
    fn nc_close(id: c_int) -> c_int;
    fn nc_strerror(code: c_int) -> *const c_char;
    fn nc_inq_varid(id: c_int, name: *const c_char, v: *mut c_int) -> c_int;
    fn nc_inq_varndims(id: c_int, v: c_int, n: *mut c_int) -> c_int;
    fn nc_inq_vardimid(id: c_int, v: c_int, d: *mut c_int) -> c_int;
    fn nc_inq_dimlen(id: c_int, d: c_int, n: *mut usize) -> c_int;
    fn nc_get_var_double(id: c_int, v: c_int, p: *mut f64) -> c_int;
    fn nc_get_att_double(id: c_int, v: c_int, n: *const c_char, p: *mut f64) -> c_int;
    fn nc_inq_attlen(id: c_int, v: c_int, n: *const c_char, len: *mut usize) -> c_int;
    fn nc_get_att_text(id: c_int, v: c_int, n: *const c_char, p: *mut c_char) -> c_int;
    fn nc_def_dim(id: c_int, n: *const c_char, len: usize, d: *mut c_int) -> c_int;
    fn nc_def_var(
        id: c_int,
        n: *const c_char,
        t: c_int,
        nd: c_int,
        ds: *const c_int,
        v: *mut c_int,
    ) -> c_int;
    fn nc_put_att_text(
        id: c_int,
        v: c_int,
        n: *const c_char,
        len: usize,
        p: *const c_char,
    ) -> c_int;
    fn nc_enddef(id: c_int) -> c_int;
    fn nc_put_var_float(id: c_int, v: c_int, p: *const f32) -> c_int;
    fn nc_put_vara_float(
        id: c_int,
        v: c_int,
        start: *const usize,
        count: *const usize,
        p: *const f32,
    ) -> c_int;
    fn nc_put_vara_double(
        id: c_int,
        v: c_int,
        start: *const usize,
        count: *const usize,
        p: *const f64,
    ) -> c_int;
}
type Res<T> = Result<T, String>;
fn cs(s: &str) -> CString {
    CString::new(s).unwrap()
}
fn check(n: i32) -> Res<()> {
    if n == 0 {
        Ok(())
    } else {
        Err(unsafe { CStr::from_ptr(nc_strerror(n)) }
            .to_string_lossy()
            .into_owned())
    }
}
struct Nc(i32);
impl Drop for Nc {
    fn drop(&mut self) {
        unsafe {
            nc_close(self.0);
        }
    }
}
impl Nc {
    fn open(p: &str, write: bool) -> Res<Self> {
        let mut id = 0;
        check(unsafe { nc_open(cs(p).as_ptr(), if write { 1 } else { 0 }, &mut id) })?;
        Ok(Nc(id))
    }
    fn var(&self, n: &str) -> Res<i32> {
        let mut v = 0;
        check(unsafe { nc_inq_varid(self.0, cs(n).as_ptr(), &mut v) })?;
        Ok(v)
    }
    fn read(&self, n: &str) -> Res<(Vec<usize>, Vec<f64>)> {
        let v = self.var(n)?;
        let mut nd = 0;
        check(unsafe { nc_inq_varndims(self.0, v, &mut nd) })?;
        let mut ds = vec![0; nd as usize];
        check(unsafe { nc_inq_vardimid(self.0, v, ds.as_mut_ptr()) })?;
        let mut shape = Vec::new();
        for d in ds {
            let mut len = 0;
            check(unsafe { nc_inq_dimlen(self.0, d, &mut len) })?;
            shape.push(len);
        }
        let mut a = vec![0.; shape.iter().product()];
        check(unsafe { nc_get_var_double(self.0, v, a.as_mut_ptr()) })?;
        let scale = self.number(v, "scale_factor").unwrap_or(1.);
        let offset = self.number(v, "add_offset").unwrap_or(0.);
        let fill = self.number(v, "_FillValue");
        let missing = self.number(v, "missing_value");
        parallel(&mut a, |_, x| {
            *x = if !x.is_finite() || fill == Some(*x) || missing == Some(*x) {
                f64::NAN
            } else {
                *x * scale + offset
            }
        });
        Ok((shape, a))
    }
    fn number(&self, v: i32, name: &str) -> Option<f64> {
        let mut n = 0.0;
        let mut len = 0;
        let key = cs(name);
        if unsafe { nc_inq_attlen(self.0, v, key.as_ptr(), &mut len) } != 0 || len != 1 {
            return None;
        }
        if unsafe { nc_get_att_double(self.0, v, key.as_ptr(), &mut n) } == 0 {
            Some(n)
        } else {
            None
        }
    }
    fn text(&self, v: i32, name: &str) -> Res<String> {
        let mut n = 0;
        check(unsafe { nc_inq_attlen(self.0, v, cs(name).as_ptr(), &mut n) })?;
        let mut b = vec![0u8; n];
        check(unsafe { nc_get_att_text(self.0, v, cs(name).as_ptr(), b.as_mut_ptr().cast()) })?;
        String::from_utf8(b).map_err(|e| e.to_string())
    }
}
fn workers() -> usize {
    std::env::var("WAVES_IO_THREADS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8)
        .clamp(1, 64)
}
fn parallel<T: Send, F: Fn(usize, &mut T) + Sync>(a: &mut [T], f: F) {
    let size = a.len().div_ceil(workers()).max(1);
    std::thread::scope(|scope| {
        for (i, chunk) in a.chunks_mut(size).enumerate() {
            let ff = &f;
            scope.spawn(move || {
                for (j, x) in chunk.iter_mut().enumerate() {
                    ff(i * size + j, x);
                }
            });
        }
    });
}
fn bytes_f32(path: &Path, a: &[f32]) -> Res<()> {
    let mut f = fs::File::create(path).map_err(|e| e.to_string())?;
    let p = unsafe { std::slice::from_raw_parts(a.as_ptr().cast::<u8>(), a.len() * 4) };
    f.write_all(p).map_err(|e| e.to_string())
}
fn bytes_f64(path: &Path, a: &[f64]) -> Res<()> {
    let mut f = fs::File::create(path).map_err(|e| e.to_string())?;
    let p = unsafe { std::slice::from_raw_parts(a.as_ptr().cast::<u8>(), a.len() * 8) };
    f.write_all(p).map_err(|e| e.to_string())
}
fn regular(a: &[f64]) -> Res<()> {
    if a.len() < 2
        || a.iter().any(|x| !x.is_finite())
        || a.windows(2)
            .any(|w| ((w[1] - w[0]) - (a[1] - a[0])).abs() > 1e-5)
        || a[1] == a[0]
    {
        Err("coordinate must be finite, regular, with at least two cells".into())
    } else {
        Ok(())
    }
}
fn axis(nc: &Nc, name: &str) -> Res<Vec<f64>> {
    let (sh, a) = nc.read(name)?;
    if sh.len() != 1 {
        return Err(format!("{name}: one dimensional axis required"));
    }
    regular(&a)?;
    Ok(a)
}
fn units(nc: &Nc, name: &str, wanted: &[&str]) -> Res<()> {
    let unit = nc.text(nc.var(name)?, "units")?;
    if wanted.contains(&unit.as_str()) {
        Ok(())
    } else {
        Err(format!("{name}: unsupported units {unit:?}"))
    }
}
fn pack(forcing: &str, bath: &str, directory: &str) -> Res<()> {
    let nc = Nc::open(forcing, false)?;
    let lat0 = axis(&nc, "latitude")?;
    let lon0 = axis(&nc, "longitude")?;
    let (_, times) = nc.read("time")?;
    units(&nc, "time", &["seconds since 1970-01-01 00:00:00"])?;
    if times.len() < 2
        || times.iter().any(|x| !x.is_finite())
        || times.windows(2).any(|w| w[1] <= w[0])
    {
        return Err("forcing needs increasing finite times".into());
    }
    let ny = lat0.len();
    let nx = lon0.len();
    let nt = times.len();
    let n = ny * nx;
    let yl = lat0[1] < lat0[0];
    let xl = lon0[1] < lon0[0];
    let lat: Vec<_> = if yl {
        lat0.iter().rev().copied().collect()
    } else {
        lat0.clone()
    };
    let lon: Vec<_> = if xl {
        lon0.iter().rev().copied().collect()
    } else {
        lon0.clone()
    };
    let out = Path::new(directory);
    fs::create_dir_all(out).map_err(|e| e.to_string())?;
    let reorder = |i: usize| {
        let c = i % n;
        let y = c / nx;
        let x = c % nx;
        (i / n) * n + (if yl { ny - 1 - y } else { y }) * nx + if xl { nx - 1 - x } else { x }
    };
    for name in ["U10", "V10"] {
        units(&nc, name, &["m s-1", "m/s"])?;
        let (shape, data) = nc.read(name)?;
        if shape != [nt, ny, nx] || data.iter().any(|x| !x.is_finite()) {
            return Err(format!(
                "{name}: complete [time,latitude,longitude] finite data required"
            ));
        }
        let mut dst = vec![0f32; data.len()];
        parallel(&mut dst, |i, x| *x = data[reorder(i)] as f32);
        bytes_f32(&out.join(format!("{name}.f32")), &dst)?;
    }
    let (_, land) = nc.read("LANDSEA")?;
    if land.iter().any(|x| !x.is_finite() || *x < 0.0 || *x > 1.0) {
        return Err("LANDSEA must contain finite land fractions in [0,1]".into());
    }
    if land.len() != n && land.len() != nt * n {
        return Err("LANDSEA grid differs from winds".into());
    }
    let bath = Nc::open(bath, false)?;
    let blat = axis(&bath, "latitude")?;
    let blon = axis(&bath, "longitude")?;
    let (bs, alt) = bath.read("altitude")?;
    units(&bath, "altitude", &["m"])?;
    if bs != [blat.len(), blon.len()] {
        return Err("bathymetry axes differ from altitude".into());
    }
    let by = blat.len();
    let bx = blon.len();
    let mut depth = vec![f32::NAN; n];
    parallel(&mut depth, |i, x| {
        let yy = (lat[i / nx] - blat[0]) / (blat[1] - blat[0]);
        let xx = (lon[i % nx] - blon[0]) / (blon[1] - blon[0]);
        let ls = land[reorder(i)];
        if ls >= 0.5 {
            *x = 0.;
            return;
        }
        if !ls.is_finite()
            || yy < 0.
            || xx < 0.
            || yy > (by - 1) as f64 + 1e-5
            || xx > (bx - 1) as f64 + 1e-5
        {
            return;
        }
        let y = (yy.floor() as usize).min(by - 2);
        let j = (xx.floor() as usize).min(bx - 2);
        let a = yy - y as f64;
        let b = xx - j as f64;
        let z = (1. - a) * ((1. - b) * alt[y * bx + j] + b * alt[y * bx + j + 1])
            + a * ((1. - b) * alt[(y + 1) * bx + j] + b * alt[(y + 1) * bx + j + 1]);
        // A resolved ocean point must have actual bathymetry. Land is absorbing.
        *x = if !z.is_finite() {
            f32::NAN
        } else if z < 0. {
            (-z).max(1.) as f32
        } else {
            0.
        };
    });
    if depth.iter().any(|x| !x.is_finite()) {
        return Err(
            "bathymetry does not cover all wet forcing cells; acquire the forcing halo too".into(),
        );
    }
    let wet = depth.iter().filter(|&&x| x > 0.).count();
    if wet == 0 {
        return Err("domain has no wet cells".into());
    }
    bytes_f32(
        &out.join("latitude.f32"),
        &lat.iter().map(|&x| x as f32).collect::<Vec<_>>(),
    )?;
    bytes_f32(
        &out.join("longitude.f32"),
        &lon.iter().map(|&x| x as f32).collect::<Vec<_>>(),
    )?;
    bytes_f32(&out.join("depth.f32"), &depth)?;
    bytes_f64(&out.join("time.f64"), &times)?;
    fs::write(out.join("manifest.json"),format!("{{\"schema\":\"arwen.waves.forcing.v1\",\"nx\":{nx},\"ny\":{ny},\"nt\":{nt},\"wet_cells\":{wet},\"io_threads\":{},\"dx_degrees\":{},\"dy_degrees\":{}}}\n",workers(),lon[1]-lon[0],lat[1]-lat[0])).map_err(|e|e.to_string())
}
fn att(id: i32, v: i32, k: &str, val: &str) -> Res<()> {
    check(unsafe { nc_put_att_text(id, v, cs(k).as_ptr(), val.len(), val.as_ptr().cast()) })
}
const NAMES: [&str; 7] = [
    "HS",
    "TP",
    "MEAN_DIRECTION",
    "HS_WINDSEA",
    "HS_SWELL",
    "Z0",
    "USTAR",
];
const STANDARD: [&str; 7] = [
    "sea_surface_wave_significant_height",
    "sea_surface_wave_period_at_variance_spectral_density_maximum",
    "sea_surface_wave_from_direction",
    "sea_surface_wind_wave_significant_height",
    "sea_surface_swell_wave_significant_height",
    "surface_roughness_length",
    "friction_velocity",
];
unsafe fn write_frame(
    path: &str,
    lat: &[f32],
    lon: &[f32],
    epoch: f64,
    data: &[f32],
    index: usize,
) -> Res<()> {
    let ny = lat.len();
    let nx = lon.len();
    let nc;
    if index == 0 {
        let mut id = 0;
        check(nc_create(cs(path).as_ptr(), 0x200, &mut id))?;
        nc = Nc(id);
        let mut ds = [0; 3];
        for (j, (name, len)) in [("time", 0), ("latitude", ny), ("longitude", nx)]
            .iter()
            .enumerate()
        {
            check(nc_def_dim(id, cs(name).as_ptr(), *len, &mut ds[j]))?;
        }
        for (j, (name, unit)) in [
            ("time", "seconds since 1970-01-01 00:00:00"),
            ("latitude", "degrees_north"),
            ("longitude", "degrees_east"),
        ]
        .iter()
        .enumerate()
        {
            let mut v = 0;
            check(nc_def_var(
                id,
                cs(name).as_ptr(),
                if j == 0 { 6 } else { 5 },
                1,
                &ds[j],
                &mut v,
            ))?;
            att(id, v, "standard_name", name)?;
            att(id, v, "units", unit)?;
        }
        for j in 0..7 {
            let mut v = 0;
            check(nc_def_var(
                id,
                cs(NAMES[j]).as_ptr(),
                5,
                3,
                ds.as_ptr(),
                &mut v,
            ))?;
            att(id, v, "standard_name", STANDARD[j])?;
            att(
                id,
                v,
                "units",
                if j == 1 {
                    "s"
                } else if j == 2 {
                    "degree"
                } else if j == 6 {
                    "m s-1"
                } else {
                    "m"
                },
            )?;
        }
        att(id, -1, "Conventions", "CF-1.10")?;
        att(
            id,
            -1,
            "source",
            "ArWen CUDA spectral waves; WAM cycle 4 sources and DIA",
        )?;
        check(nc_enddef(id))?;
        check(nc_put_var_float(id, nc.var("latitude")?, lat.as_ptr()))?;
        check(nc_put_var_float(id, nc.var("longitude")?, lon.as_ptr()))?;
    } else {
        nc = Nc::open(path, true)?;
    }
    check(nc_put_vara_double(
        nc.0,
        nc.var("time")?,
        [index].as_ptr(),
        [1usize].as_ptr(),
        &epoch,
    ))?;
    for j in 0..7 {
        check(nc_put_vara_float(
            nc.0,
            nc.var(NAMES[j])?,
            [index, 0, 0].as_ptr(),
            [1, ny, nx].as_ptr(),
            data[j * ny * nx..].as_ptr(),
        ))?;
    }
    Ok(())
}
#[no_mangle]
pub unsafe extern "C" fn waves_write(
    path: *const c_char,
    lat: *const f32,
    lon: *const f32,
    ny: usize,
    nx: usize,
    epoch: f64,
    data: *const f32,
    index: usize,
) -> i32 {
    let result = std::panic::catch_unwind(|| {
        write_frame(
            CStr::from_ptr(path).to_str().unwrap(),
            std::slice::from_raw_parts(lat, ny),
            std::slice::from_raw_parts(lon, nx),
            epoch,
            std::slice::from_raw_parts(data, 7 * ny * nx),
            index,
        )
    });
    match result {
        Ok(Ok(())) => 0,
        Ok(Err(e)) => {
            eprintln!("waves I/O: {e}");
            1
        }
        Err(_) => 2,
    }
}
// Compact spectral diagnostics are selected/gathered on CUDA. The native
// boundary only serializes their words; it does not integrate wave physics.
#[no_mangle]
pub unsafe extern "C" fn waves_capture(
    path: *const c_char, data: *const f32, count: usize,
) -> i32 {
    let result = std::panic::catch_unwind(|| -> Res<()> {
        let name = CStr::from_ptr(path).to_str().map_err(|e| e.to_string())?;
        let values = std::slice::from_raw_parts(data, count);
        let mut file = std::io::BufWriter::new(fs::File::create(name).map_err(|e| e.to_string())?);
        for value in values {
            file.write_all(&value.to_le_bytes()).map_err(|e| e.to_string())?;
        }
        file.flush().map_err(|e| e.to_string())
    });
    match result {
        Ok(Ok(())) => 0,
        Ok(Err(e)) => { eprintln!("waves capture: {e}"); 1 },
        Err(_) => 2,
    }
}
// UTC conversion for public buoy records, Gregorian calendar.
fn epoch(y: i64, m: i64, d: i64, h: i64, min: i64) -> i64 {
    let y = y - if m <= 2 { 1 } else { 0 };
    let era = y.div_euclid(400);
    let yy = y - era * 400;
    let mm = m + if m > 2 { -3 } else { 9 };
    let day = (153 * mm + 2) / 5 + d - 1;
    (era * 146097 + yy * 365 + yy / 4 - yy / 100 + day - 719468) * 86400 + h * 3600 + min * 60
}
fn verify(model: &str, obs: &str, station: &str, lat: f64, lon: f64, out: &str) -> Res<()> {
    let nc = Nc::open(model, false)?;
    let ys = axis(&nc, "latitude")?;
    let xs = axis(&nc, "longitude")?;
    let (_, ts) = nc.read("time")?;
    let (_, hs) = nc.read("HS")?;
    let (_, tp) = nc.read("TP")?;
    let (_, dirs) = nc.read("MEAN_DIRECTION")?;
    let ny = ys.len();
    let nx = xs.len();
    let mut choices: Vec<(f64, usize)> = (0..nx * ny)
        .filter(|&i| hs[i].is_finite())
        .map(|i| {
            (
                (ys[i / nx] - lat).powi(2) + ((xs[i % nx] - lon) * lat.to_radians().cos()).powi(2),
                i,
            )
        })
        .collect();
    choices.sort_by(|a, b| a.0.total_cmp(&b.0));
    let &(dist, cell) = choices.first().ok_or("no wet match")?;
    if dist.sqrt() > 0.5 {
        return Err("buoy farther than 0.5 degree from a wet cell".into());
    }
    let mut rows: Vec<(usize, f64, f64, f64, f64)> = Vec::new();
    for line in fs::read_to_string(obs).map_err(|e| e.to_string())?.lines() {
        if line.starts_with('#') {
            continue;
        }
        let w: Vec<&str> = line.split_whitespace().collect();
        if w.len() < 15 {
            continue;
        }
        let v: Vec<f64> = match w.iter().map(|s| s.parse::<f64>()).collect() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let stamp = epoch(
            v[0] as i64,
            v[1] as i64,
            v[2] as i64,
            v[3] as i64,
            v[4] as i64,
        ) as f64;
        if let Some((ti, &t)) = ts
            .iter()
            .enumerate()
            .min_by(|a, b| (a.1 - stamp).abs().total_cmp(&(b.1 - stamp).abs()))
        {
            if (t - stamp).abs() <= 1800. && v[8] < 90. && v[8] >= 0. {
                rows.push((ti, stamp, v[8], v[9], v[11]));
            }
        }
    }
    let mut file = fs::File::create(out).map_err(|e| e.to_string())?;
    writeln!(file,"station,time_epoch,model_time_epoch,latitude,longitude,model_latitude,model_longitude,hs_obs_m,hs_model_m,tp_obs_s,tp_model_s,mwd_obs_deg,mwd_model_deg").map_err(|e|e.to_string())?;
    let mut dh = Vec::new();
    let mut dt = Vec::new();
    let mut dd = Vec::new();
    let mut peak_obs: f64 = 0.;
    let mut peak_mod: f64 = 0.;
    for (ti, stamp, oh, ot, od) in rows {
        let i = ti * nx * ny + cell;
        let mh = hs[i];
        dh.push(mh - oh);
        peak_obs = peak_obs.max(oh);
        peak_mod = peak_mod.max(mh);
        let ot = if ot < 90. { ot } else { f64::NAN };
        let od = if od < 360. { od } else { f64::NAN };
        if ot.is_finite() && tp[i].is_finite() {
            dt.push(tp[i] - ot)
        }
        if od.is_finite() && dirs[i].is_finite() {
            dd.push((dirs[i] - od + 180.).rem_euclid(360.) - 180.)
        }
        writeln!(
            file,
            "{station},{stamp},{},{lat},{lon},{},{},{oh},{mh},{ot},{},{od},{}",
            ts[ti],
            ys[cell / nx],
            xs[cell % nx],
            tp[i],
            dirs[i]
        )
        .map_err(|e| e.to_string())?;
    }
    if dh.is_empty() {
        return Err("no observed Hs pairs within the model time window".into());
    }
    let stats = |v: &[f64]| {
        format!(
            "{{\"n\":{},\"bias\":{},\"rmse\":{}}}",
            v.len(),
            v.iter().sum::<f64>() / v.len().max(1) as f64,
            (v.iter().map(|x| x * x).sum::<f64>() / v.len().max(1) as f64).sqrt()
        )
    };
    println!("{{\"station\":\"{station}\",\"sampling\":\"nearest wet cell within 0.5 degree; nearest output within 30 minutes\",\"distance_degrees\":{},\"hs\":{},\"tp\":{},\"direction\":{},\"peak_hs_observed_m\":{peak_obs},\"peak_hs_model_m\":{peak_mod}}}",dist.sqrt(),stats(&dh),stats(&dt),stats(&dd));
    Ok(())
}
// Compare the prescribed wind with observations as well as the resulting waves.
// Buoy anemometers need not be at 10 m: retain raw speeds and label this limit.
fn verify_wind(forcing: &str, obs: &str, station: &str, lat: f64, lon: f64, out: &str) -> Res<()> {
    let nc = Nc::open(forcing, false)?;
    let ys = axis(&nc, "latitude")?;
    let xs = axis(&nc, "longitude")?;
    units(&nc, "time", &["seconds since 1970-01-01 00:00:00"])?;
    units(&nc, "U10", &["m s-1", "m/s"])?;
    units(&nc, "V10", &["m s-1", "m/s"])?;
    let (_, ts) = nc.read("time")?;
    let (shape, u) = nc.read("U10")?;
    let (vshape, v) = nc.read("V10")?;
    let n = ys.len() * xs.len();
    if shape != [ts.len(), ys.len(), xs.len()] || vshape != shape {
        return Err("wind axes differ from forcing grid".into());
    }
    let cell = (0..n)
        .min_by(|&a, &b| {
            let distance = |i: usize| {
                (ys[i / xs.len()] - lat).powi(2)
                    + ((xs[i % xs.len()] - lon) * lat.to_radians().cos()).powi(2)
            };
            distance(a).total_cmp(&distance(b))
        })
        .ok_or("no forcing grid cells")?;
    let distance = ((ys[cell / xs.len()] - lat).powi(2)
        + ((xs[cell % xs.len()] - lon) * lat.to_radians().cos()).powi(2))
    .sqrt();
    if distance > 0.5 {
        return Err("buoy farther than 0.5 degree from forcing cell".into());
    }
    let mut pairs = Vec::new();
    for line in fs::read_to_string(obs).map_err(|e| e.to_string())?.lines() {
        if line.starts_with('#') {
            continue;
        }
        let w: Vec<_> = line.split_whitespace().collect();
        if w.len() < 7 {
            continue;
        }
        let values: Vec<f64> = match w[..7].iter().map(|x| x.parse()).collect() {
            Ok(x) => x,
            Err(_) => continue,
        };
        let speed = values[6];
        if !valid_wind_speed(speed) {
            continue;
        }
        let stamp = epoch(
            values[0] as i64,
            values[1] as i64,
            values[2] as i64,
            values[3] as i64,
            values[4] as i64,
        ) as f64;
        if let Some((ti, &t)) = ts
            .iter()
            .enumerate()
            .min_by(|a, b| (a.1 - stamp).abs().total_cmp(&(b.1 - stamp).abs()))
        {
            if (t - stamp).abs() <= 1800. {
                pairs.push((ti, stamp, speed));
            }
        }
    }
    // Per-pair transformations are independent and use the native worker pool.
    let mut model = vec![0.; pairs.len()];
    parallel(&mut model, |j, x| {
        let i = pairs[j].0 * n + cell;
        *x = u[i].hypot(v[i]);
    });
    if pairs.is_empty() || model.iter().any(|x| !x.is_finite()) {
        return Err("no finite observed/forcing wind pairs".into());
    }
    let mut file = fs::File::create(out).map_err(|e| e.to_string())?;
    writeln!(file,"station,time_epoch,forcing_time_epoch,latitude,longitude,forcing_latitude,forcing_longitude,observed_speed_ms,forcing_10m_speed_ms").map_err(|e|e.to_string())?;
    let mut bias = 0.;
    let mut squared = 0.;
    let mut peak_obs: f64 = 0.;
    let mut peak_model: f64 = 0.;
    for ((ti, stamp, speed), value) in pairs.iter().zip(&model) {
        let d = value - speed;
        bias += d;
        squared += d * d;
        peak_obs = peak_obs.max(*speed);
        peak_model = peak_model.max(*value);
        writeln!(
            file,
            "{station},{stamp},{},{lat},{lon},{},{},{speed},{value}",
            ts[*ti],
            ys[cell / xs.len()],
            xs[cell % xs.len()]
        )
        .map_err(|e| e.to_string())?;
    }
    let count = pairs.len();
    println!("{{\"station\":\"{station}\",\"n\":{count},\"bias_ms\":{},\"rmse_ms\":{},\"peak_observed_ms\":{peak_obs},\"peak_forcing_10m_ms\":{peak_model},\"distance_degrees\":{distance},\"height_correction\":\"none; buoy anemometer height differs from 10 m\",\"workers\":{}}}",bias/count as f64,(squared/count as f64).sqrt(),workers());
    Ok(())
}
fn valid_wind_speed(value: f64) -> bool {
    value.is_finite() && (0. ..90.).contains(&value)
}
fn cli() -> Res<()> {
    let a: Vec<String> = std::env::args().collect();
    match a.get(1).map(String::as_str){
 Some("pack")if a.len()==5=>pack(&a[2],&a[3],&a[4]),
 Some("verify")if a.len()==8=>verify(&a[2],&a[3],&a[4],a[5].parse().map_err(|_|"invalid latitude")?,a[6].parse().map_err(|_|"invalid longitude")?,&a[7]),
 Some("verify-wind")if a.len()==8=>verify_wind(&a[2],&a[3],&a[4],a[5].parse().map_err(|_|"invalid latitude")?,a[6].parse().map_err(|_|"invalid longitude")?,&a[7]),
 _=>Err("usage: rw_waves_io pack FORCING.nc BATHYMETRY.nc DIRECTORY | verify MODEL.nc OBS.txt STATION LAT LON OUT.csv | verify-wind FORCING.nc OBS.txt STATION LAT LON OUT.csv".into())}
}
fn main() {
    if let Err(e) = cli() {
        eprintln!("rw_waves_io: {e}");
        std::process::exit(1)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn observed_wind_missing_sentinels() {
        assert!(valid_wind_speed(0.));
        assert!(valid_wind_speed(35.2));
        for value in [-1., 90., 99., 999., f64::NAN, f64::INFINITY] {
            assert!(!valid_wind_speed(value));
        }
    }
    #[test]
    fn utc() {
        assert_eq!(epoch(1970, 1, 1, 0, 0), 0);
        assert_eq!(epoch(2000, 3, 1, 0, 0) - epoch(2000, 2, 28, 0, 0), 172800);
        assert_eq!(epoch(2022, 9, 28, 12, 0), 1664366400);
    }
    #[test]
    fn regular_axes() {
        assert!(regular(&[2., 1., 0.]).is_ok());
        assert!(regular(&[0., 1., 3.]).is_err());
        assert!(regular(&[0., f64::NAN]).is_err());
    }
    #[test]
    fn parallel_indices() {
        let mut v = vec![0usize; 100003];
        parallel(&mut v, |i, x| *x = i * 3);
        assert!(v.iter().enumerate().all(|(i, x)| *x == i * 3));
    }
}
