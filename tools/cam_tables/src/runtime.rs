use super::{gases::GasTable, saturation_water, table_fields, verified_bytes};
use rayon::prelude::*;
use serde_json::Value;
use std::{
    collections::HashMap,
    ffi::{c_char, CStr},
    path::Path,
    sync::{Arc, Mutex, OnceLock},
};
pub struct CamTables {
    pub state: Vec<u8>,
    pub gas: GasTable,
    pub pin: Vec<f32>,
    pub lat: Vec<f32>,
    pub ozone: Vec<f32>,
    pub hybi: Vec<f32>,
}
static CACHE: OnceLock<Mutex<HashMap<String, Arc<CamTables>>>> = OnceLock::new();
static ERROR: OnceLock<Mutex<String>> = OnceLock::new();
fn field<'a>(layout: &'a Value, name: &str) -> &'a Value {
    layout
        .as_array()
        .unwrap()
        .iter()
        .find(|v| v["name"] == name)
        .unwrap()
}
fn put64(state: &mut [u8], layout: &Value, name: &str, value: f64) {
    let p = field(layout, name)["offset"].as_u64().unwrap() as usize;
    state[p..p + 8].copy_from_slice(&value.to_le_bytes());
}
fn get64(state: &[u8], layout: &Value, name: &str, index: usize) -> f64 {
    let p = field(layout, name)["offset"].as_u64().unwrap() as usize + index * 8;
    f64::from_le_bytes(state[p..p + 8].try_into().unwrap())
}
fn put32(state: &mut [u8], layout: &Value, name: &str, value: i32) {
    let p = field(layout, name)["offset"].as_u64().unwrap() as usize;
    state[p..p + 4].copy_from_slice(&value.to_le_bytes());
}
fn source_real(v: f32) -> f64 {
    v as f64
}
fn saturation(t: f64, melt: f64) -> f64 {
    let tr = 20.;
    let mut water = 0.;
    if t >= melt - tr {
        water = saturation_water(t);
        if t >= melt {
            return water;
        }
    }
    let ratio = melt / t;
    let a = source_real(2.01889049_f32) / ratio;
    let b = source_real(3.56654_f32) * ratio.ln();
    let c = source_real(20.947031_f32) * ratio;
    let ice = source_real(575.185606e10_f32) * (-(a + b + c)).exp();
    if t < melt - tr {
        ice
    } else {
        let weight = ((melt - t) / tr).min(1.);
        weight * ice + (1. - weight) * water
    }
}
// radini/radaeini/esinti initialization in WRF assignment order. Defaults here
// match the independent column harness; physical constants can be supplied by
// the caller before building a driver-specific state buffer.
fn initialize(state: &mut [u8], layout: &Value, physical: &[f64; 12], mxaerl: i32) {
    let grav = physical[0] as f32;
    let cpair = physical[1] as f32;
    let eps = physical[2] as f32;
    let stebol = physical[3] as f32;
    let rd = physical[4] as f32;
    let rv = physical[5] as f32;
    for (n, v) in [
        ("mwdry", 28.966_f32),
        ("mwco2", 44.),
        ("mwh2o", 18.016),
        ("mwch4", 16.),
        ("mwn2o", 44.),
        ("mwf11", 136.),
        ("mwf12", 120.),
        ("rair", rd),
        ("latvap", 2.501e6),
        ("latice", 3.336e5),
    ] {
        put64(state, layout, n, v as f64);
    }
    put64(state, layout, "cappa", (rd / cpair) as f64);
    put64(state, layout, "zvir", (rv / rd - 1.) as f64);
    put64(
        state,
        layout,
        "r_universal",
        (6.02214e26_f32 * stebol) as f64,
    );
    for (n, v) in [
        ("idxsul", 2),
        ("idxsslt", 3),
        ("idxcarbonfirst", 8),
        ("idxocpho", 8),
        ("idxbcpho", 9),
        ("idxocphi", 10),
        ("idxbcphi", 11),
        ("idxdustfirst", 4),
        ("idxbg", 12),
        ("idxvolc", 13),
        ("levsiz_ozone_save", 59),
        ("mxaerl", mxaerl),
        ("ixcldliq", 2),
        ("ixcldice", 3),
        ("ozncyc", 1),
        ("indirect", 1),
    ] {
        put32(state, layout, n, v);
    }
    let co2 = get64(state, layout, "co2vmr", 0) * get64(state, layout, "mwco2", 0)
        / get64(state, layout, "mwdry", 0);
    put64(state, layout, "co2mmr", co2);
    let g = (100.0_f32 * grav) as f64;
    let sslp = 1.013250e6_f32 as f64;
    for (n, v) in [
        ("gravit", g),
        ("rga", 1. / g),
        ("gravmks", grav as f64),
        ("cpair", (1.0e4_f32 * cpair) as f64),
        ("epsilo", eps as f64),
        ("sslp", sslp),
        ("stebol", (1.0e3_f32 * stebol) as f64),
        ("rgsslp", 0.5 / (g * sslp)),
        ("dpfo3", 2.5e-3_f32 as f64),
        ("dpfco2", 5.0e-3_f32 as f64),
        ("dayspy", 365.),
        ("pie", (4.0_f32 * 1.0_f32.atan()) as f64),
    ] {
        put64(state, layout, n, v);
    }
    let v0 = 22.4136_f32 as f64;
    let amd = 28.9644_f32 as f64;
    let goz = grav as f64;
    let p0 = 0.1_f32 as f64 * sslp;
    put64(state, layout, "cplos", v0 / (amd * goz) * 100.);
    put64(state, layout, "cplol", v0 / (amd * goz * p0) * 0.5 * 100.);
    put32(state, layout, "ntoplw", 1);
    let mw = get64(state, layout, "mwdry", 0);
    put64(state, layout, "p0", 1013250.);
    put64(state, layout, "amd", mw);
    put64(state, layout, "amco2", 44.);
    for (n, table, a, b) in [
        ("c16", "coefj", 2, 1),
        ("c17", "coefk", 2, 1),
        ("c26", "coefj", 5, 4),
        ("c27", "coefk", 5, 4),
    ] {
        let v = get64(state, layout, table, a) / get64(state, layout, table, b);
        put64(state, layout, n, v);
    }
    for (n, v) in [
        ("c28", 0.5_f32),
        ("c29", 0.002053_f32),
        ("c30", 0.1_f32),
        ("c31", 3.0e-5_f32),
        ("fwcoef", 0.1_f32),
        ("fwc1", 0.30_f32),
        ("fwc2", 4.5_f32),
        ("fc1", 2.6_f32),
    ] {
        put64(state, layout, n, v as f64);
    }
    let tmin = 173.16_f32 as f64;
    let tmax = 375.16_f32 as f64;
    for (n, v) in [
        ("tmin", tmin),
        ("tmax", tmax),
        ("epsqs", physical[6]),
        ("tmelt", physical[11]),
    ] {
        put64(state, layout, n, v);
    }

    let lentbl = (tmax - tmin + 2.000001_f32 as f64) as usize;
    let p = field(layout, "estbl")["offset"].as_u64().unwrap() as usize;
    let mut t = tmin - 1.;
    for i in 0..250 {
        let value = if i < lentbl {
            t += 1.;
            saturation(t, physical[11])
        } else {
            -99999.
        };
        state[p + i * 8..p + (i + 1) * 8].copy_from_slice(&value.to_le_bytes());
    }
    let p = field(layout, "pcf")["offset"].as_u64().unwrap() as usize;
    for (i, v) in [
        5.04469588506e-1_f32,
        -5.47288442819_f32,
        -3.67471858735e-1_f32,
        -8.95963532403e-3_f32,
        -7.78053686625e-5_f32,
    ]
    .into_iter()
    .enumerate()
    {
        state[p + i * 8..p + (i + 1) * 8].copy_from_slice(&(v as f64).to_le_bytes());
    }
}
fn read_f32(root: &Path, name: &str, count: usize) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let bytes = verified_bytes(root, name)?;
    let text = std::str::from_utf8(&bytes)?;
    let values: Result<Vec<f32>, _> = text
        .par_lines()
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.trim().parse())
        .collect();
    let values = values?;
    if values.len() != count || values.par_iter().any(|x| !x.is_finite()) {
        return Err(format!("{name}: wrong record count or nonfinite value").into());
    }
    Ok(values)
}
impl CamTables {
    fn load(root: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let layout: Value =
            serde_json::from_str(include_str!("../../../woof/data/cam3/state-layout.json"))?;
        let defaults: Value =
            serde_json::from_str(include_str!("../../../woof/data/cam3/state-defaults.json"))?;
        let size = layout
            .as_array()
            .unwrap()
            .iter()
            .map(|f| f["offset"].as_u64().unwrap() + f["bytes"].as_u64().unwrap())
            .max()
            .unwrap() as usize;
        let mut state = vec![0; size];
        for f in defaults["fields"].as_array().unwrap() {
            let p = f["offset"].as_u64().unwrap() as usize;
            let hex = f["hex"].as_str().unwrap();
            for (i, b) in hex.as_bytes().chunks_exact(2).enumerate() {
                state[p + i] = u8::from_str_radix(std::str::from_utf8(b)?, 16)?;
            }
        }
        for (name, values) in table_fields(root)? {
            let f = field(&layout, name);
            let p = f["offset"].as_u64().unwrap() as usize;
            if f["bytes"].as_u64().unwrap() != values.len() as u64 * 8 {
                return Err(format!("{name}: schema size mismatch").into());
            }
            state[p..p + values.len() * 8]
                .par_chunks_mut(8 * 4096)
                .zip(values.par_chunks(4096))
                .for_each(|(bytes, values)| {
                    for (word, value) in bytes.chunks_exact_mut(8).zip(values) {
                        word.copy_from_slice(&value.to_le_bytes());
                    }
                });
        }
        initialize(
            &mut state,
            &layout,
            &[
                9.81,
                1004.5,
                0.622,
                5.67e-8,
                287.,
                461.6,
                0.622_f32 as f64,
                2.501e6,
                3.336e5,
                461.6_f32 as f64,
                1004.5,
                273.16_f32 as f64,
            ],
            3,
        );
        verified_bytes(root, "CAMtr_volume_mixing_ratio")?;
        let gas = GasTable::read(&root.join("CAMtr_volume_mixing_ratio"))?;
        let mut pin = read_f32(root, "ozone_plev.formatted", 59)?;
        for x in &mut pin {
            *x *= 100.;
        }
        let lat = read_f32(root, "ozone_lat.formatted", 64)?;
        let raw = read_f32(root, "ozone.formatted", 59 * 64 * 12)?;
        let mut ozone = vec![0.; raw.len()];
        ozone.par_chunks_mut(64 * 12).enumerate().for_each(|(k, layer)| {
            for j in 0..64 {
                for m in 0..12 {
                    layer[j * 12 + m] = raw[(m * 64 + j) * 59 + k];
                }
            }
        });
        if pin.windows(2).any(|w| w[0] >= w[1])
            || lat.windows(2).any(|w| w[0] >= w[1])
            || ozone.iter().any(|x| *x <= 0.)
        {
            return Err("CAM ozone pressure/latitude order or concentrations are invalid".into());
        }
        let grid: Value =
            serde_json::from_str(include_str!("../../../woof/data/cam3/aerosol-grid.json"))?;
        let hybi = grid["float32_bits"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| f32::from_bits(v.as_u64().unwrap() as u32))
            .collect();
        Ok(Self {
            state,
            gas,
            pin,
            lat,
            ozone,
            hybi,
        })
    }
}
fn error(message: String) {
    *ERROR
        .get_or_init(|| Mutex::new(String::new()))
        .lock()
        .unwrap() = message;
}
#[no_mangle]
pub unsafe extern "C" fn cam_tables_open(root: *const c_char) -> *const CamTables {
    if root.is_null() {
        error("null CAM table root".into());
        return std::ptr::null();
    }
    let result = (|| -> Result<Arc<CamTables>, Box<dyn std::error::Error>> {
        let path = Path::new(CStr::from_ptr(root).to_str()?).canonicalize()?;
        let key = path.to_string_lossy().into_owned();
        let mut cache = CACHE
            .get_or_init(|| Mutex::new(HashMap::new()))
            .lock()
            .unwrap();
        if let Some(table) = cache.get(&key) {
            return Ok(table.clone());
        }
        let value = Arc::new(CamTables::load(&path)?);
        cache.insert(key, value.clone());
        Ok(value)
    })();
    match result {
        Ok(table) => Arc::into_raw(table),
        Err(e) => {
            error(e.to_string());
            std::ptr::null()
        }
    }
}
#[no_mangle]
pub unsafe extern "C" fn cam_tables_close(table: *const CamTables) {
    if !table.is_null() {
        drop(Arc::from_raw(table));
    }
}
#[no_mangle]
pub unsafe extern "C" fn cam_tables_state(table: *const CamTables, size: *mut usize) -> *const u8 {
    if table.is_null() || size.is_null() {
        return std::ptr::null();
    }
    *size = (*table).state.len();
    (*table).state.as_ptr()
}
#[no_mangle]
pub unsafe extern "C" fn cam_tables_ozone(
    table: *const CamTables,
    which: i32,
    size: *mut usize,
) -> *const f32 {
    if table.is_null() || size.is_null() {
        return std::ptr::null();
    }
    let v = match which {
        0 => &(*table).pin,
        1 => &(*table).lat,
        2 => &(*table).ozone,
        3 => &(*table).hybi,
        _ => return std::ptr::null(),
    };
    *size = v.len();
    v.as_ptr()
}
#[no_mangle]
pub unsafe extern "C" fn cam_tables_gases(
    table: *const CamTables,
    year: i32,
    julian: f32,
    out: *mut f64,
) -> i32 {
    if table.is_null() || out.is_null() || !julian.is_finite() {
        return -1;
    }
    let values = (*table).gas.interpolate(year, julian);
    std::ptr::copy_nonoverlapping(values.as_ptr(), out, 5);
    0
}
#[no_mangle]
pub unsafe extern "C" fn cam_tables_error(out: *mut u8, size: usize) -> usize {
    let e = ERROR
        .get_or_init(|| Mutex::new(String::new()))
        .lock()
        .unwrap();
    let n = e.len().min(size.saturating_sub(1));
    if !out.is_null() && size > 0 {
        std::ptr::copy_nonoverlapping(e.as_ptr(), out, n);
        *out.add(n) = 0;
    }
    e.len()
}

#[no_mangle]
pub unsafe extern "C" fn cam_tables_copy_state(
    table: *const CamTables,
    physical: *const f64,
    mxaerl: i32,
    out: *mut u8,
    size: usize,
) -> i32 {
    if table.is_null()
        || physical.is_null()
        || out.is_null()
        || size != (*table).state.len()
        || mxaerl < 1
    {
        return -1;
    }
    let phys = std::slice::from_raw_parts(physical, 12);
    if phys.iter().any(|x| !x.is_finite() || *x <= 0.) {
        return -2;
    }
    let mut copy = (*table).state.clone();
    let layout: Value =
        serde_json::from_str(include_str!("../../../woof/data/cam3/state-layout.json")).unwrap();
    initialize(&mut copy, &layout, phys.try_into().unwrap(), mxaerl);
    std::ptr::copy_nonoverlapping(copy.as_ptr(), out, size);
    0
}

// Keep the native record offsets and integer words. Only fast forecasts read
// each REAL(R8) field as a packed REAL32 array at that field's original offset.
fn pack_fast_state(source: &[u8], layout: &Value, output: &mut [u8]) {
    for field in layout.as_array().unwrap() {
        if field["dtype"] != "double precision" { continue; }
        let offset = field["offset"].as_u64().unwrap() as usize;
        let count = field["count"].as_u64().unwrap() as usize;
        output[offset..offset + count * 4]
            .par_chunks_mut(4096 * 4)
            .zip(source[offset..offset + count * 8].par_chunks(4096 * 8))
            .for_each(|(out, input)| {
                for (word, bytes) in out.chunks_exact_mut(4).zip(input.chunks_exact(8)) {
                    let value = f64::from_le_bytes(bytes.try_into().unwrap()) as f32;
                    word.copy_from_slice(&value.to_le_bytes());
                }
            });
    }
}

#[no_mangle]
pub unsafe extern "C" fn cam_tables_copy_state_fast(
    table: *const CamTables,
    physical: *const f64,
    mxaerl: i32,
    out: *mut u8,
    size: usize,
) -> i32 {
    let status = cam_tables_copy_state(table, physical, mxaerl, out, size);
    if status != 0 { return status; }
    let output = std::slice::from_raw_parts_mut(out, size);
    let source = output.to_vec();
    let layout: Value =
        serde_json::from_str(include_str!("../../../woof/data/cam3/state-layout.json")).unwrap();
    pack_fast_state(&source, &layout, output);
    0
}

#[cfg(test)]
mod fast_tests {
    use super::*;
    #[test]
    fn fast_packing_keeps_offsets_integer_words_and_source() {
        let layout = serde_json::json!([
            {"dtype":"integer","offset":0,"count":1},
            {"dtype":"double precision","offset":8,"count":2}
        ]);
        let mut source = vec![0x5a; 24];
        source[..4].copy_from_slice(&73i32.to_le_bytes());
        source[8..16].copy_from_slice(&0.7f64.to_le_bytes());
        source[16..24].copy_from_slice(&(-3.25f64).to_le_bytes());
        let before = source.clone();
        let mut output = source.clone();
        pack_fast_state(&source, &layout, &mut output);
        assert_eq!(&output[..8], &source[..8]);
        assert_eq!(&output[8..12], &0.7f32.to_le_bytes());
        assert_eq!(&output[12..16], &(-3.25f32).to_le_bytes());
        assert_eq!(source, before);
    }
}
