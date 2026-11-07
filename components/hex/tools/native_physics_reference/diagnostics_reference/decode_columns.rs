//! Decode the published native init's rho, zz, theta and qv (and qc, qr) as raw float32 words, in the
//! file's own (nCells, nVertLevels) order, for the diagnostics referee (build.py). No value is converted:
//! the variables must be stored float32 and are read with nc_get_var_float.
use std::{env, ffi::{CString, c_char, c_int}, fs, path::Path};

#[link(name = "netcdf")]
extern "C" {
    fn nc_open(path: *const c_char, mode: c_int, id: *mut c_int) -> c_int;
    fn nc_close(id: c_int) -> c_int;
    fn nc_inq_varid(id: c_int, name: *const c_char, var: *mut c_int) -> c_int;
    fn nc_inq_vartype(id: c_int, var: c_int, typ: *mut c_int) -> c_int;
    fn nc_inq_varndims(id: c_int, var: c_int, n: *mut c_int) -> c_int;
    fn nc_inq_vardimid(id: c_int, var: c_int, dims: *mut c_int) -> c_int;
    fn nc_inq_dimlen(id: c_int, dim: c_int, len: *mut usize) -> c_int;
    fn nc_get_var_float(id: c_int, var: c_int, data: *mut f32) -> c_int;
}

fn checked(code: c_int) {
    assert_eq!(code, 0, "NetCDF error {code}");
}

fn read(id: c_int, name: &str) -> (Vec<f32>, Vec<usize>) {
    let (mut var, mut typ, mut rank) = (0, 0, 0);
    unsafe {
        checked(nc_inq_varid(id, CString::new(name).unwrap().as_ptr(), &mut var));
        checked(nc_inq_vartype(id, var, &mut typ));
        assert_eq!(typ, 5, "{name}: the referee needs the stored float32 words");
        checked(nc_inq_varndims(id, var, &mut rank));
    }
    let mut dims = vec![0; rank as usize];
    let mut shape = Vec::new();
    unsafe {
        checked(nc_inq_vardimid(id, var, dims.as_mut_ptr()));
        for dim in dims {
            let mut size = 0;
            checked(nc_inq_dimlen(id, dim, &mut size));
            shape.push(size);
        }
    }
    let mut data = vec![0.0f32; shape.iter().product()];
    unsafe { checked(nc_get_var_float(id, var, data.as_mut_ptr())); }
    (data, shape)
}

fn main() {
    let args: Vec<String> = env::args().collect();
    assert_eq!(args.len(), 3, "usage: decode_columns INIT.nc OUT_DIR");
    let out = Path::new(&args[2]);
    let mut id = 0;
    unsafe { checked(nc_open(CString::new(args[1].as_str()).unwrap().as_ptr(), 0, &mut id)); }
    let mut cells_levels = None;
    for name in ["rho", "zz", "theta", "qv", "qc", "qr"] {
        let (data, shape) = read(id, name);
        let tail = &shape[shape.len() - 2..];
        assert!(shape.len() == 2 || (shape.len() == 3 && shape[0] == 1), "{name}: unexpected shape {shape:?}");
        let pair = (tail[0], tail[1]);
        assert!(cells_levels.map_or(true, |p| p == pair), "{name}: shape {shape:?} differs");
        cells_levels = Some(pair);
        let bytes: Vec<u8> = data.iter().flat_map(|x| x.to_le_bytes()).collect();
        fs::write(out.join(format!("{name}.bin")), bytes).unwrap();
    }
    unsafe { checked(nc_close(id)); }
    let (cells, levels) = cells_levels.unwrap();
    fs::write(out.join("shape.json"), format!("{{\"n_cells\": {cells}, \"n_levels\": {levels}}}\n")).unwrap();
}
