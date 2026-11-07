//! Point-forecast fixtures written by the engine's own writers: hourly column files with the surface section
//! (`columns::finish`) and the per-grid series file (`series::write_series`), for the WOOF reader's tests
//! (their rwstore-engine test data). Every value is linear in the grid index, so a bilinear
//! sample at any point is exact and the reader's answer can be checked by formula.
//!
//! `cargo run -p rw-store --example point_fixture -- OUT_DIR` writes:
//!   OUT_DIR/det/columns/d01/<t>.wcol      a single forecast, 3 hours; hour 2 carries one more plane (gust_max_ms)
//!   OUT_DIR/det/columns/d01/series.wser   its series file (member "det")
//!   OUT_DIR/ens/columns/members/m0K/d01/<t>.wcol  a 3-member ensemble, 3 hours (member k adds k to t2_c and precip_mm)
//!   OUT_DIR/ens/columns/d01/series.wser   its series file (members m01..m03)
//!
//! Formulas (i = x index, j = y index, h = hour 0..2, k = member 0 for det, 1..3 for m01..m03):
//!   lat = 38 + 0.027 j + 0.004 i            lon = -99 + 0.034 i - 0.006 j
//!   psfc_hpa = 985 + 0.1 i   orog_m = 250 + j   t2_c = 20 + 0.05 i + h + k   td2_c = 12 + 0.02 j   u10 = 2   v10 = 4
//!   precip_mm = 0.5 i + 0.25 j + 10 h + k   mslp_hpa = 1012 + 0.01 j - h   gust_ms = 9 + 0.1 i
//!   wspd80_ms = 10, wdir80_deg = 350 at even i and 10 at odd i (a north wind: averaging the degrees would say 180)
//!   freezing_level_agl_m = 1500 + 2 i       gust_max_ms (hour 2 of det only) = 15 + 0.1 i
use rw_store::columns::{self, ColumnsWrite, Field2, Field3, SurfaceWrite, Values, FIELDS_2D, FIELDS_3D, QUANTA_3D, UNITS_2D, UNITS_3D};
use rw_store::series;
use serde_json::json;
use std::path::Path;

const NX: usize = 21;
const NY: usize = 17;
const NZ: usize = 3;
const T0: i64 = 1_793_577_600; // 2026-11-02T00:00:00Z

fn write(store: &Path, member: &str, h: i64, k: f64, extra: bool) {
    let n = NX * NY;
    let ij = |c: usize| ((c % NX) as f64, (c / NX) as f64);
    let plane = |f: &dyn Fn(f64, f64) -> f64| -> Vec<f64> { (0..n).map(|c| { let (i, j) = ij(c); f(i, j) }).collect() };
    let hf = h as f64;
    let lat = plane(&|i, j| 38.0 + 0.027 * j + 0.004 * i);
    let lon = plane(&|i, j| -99.0 + 0.034 * i - 0.006 * j);
    let s2: Vec<Vec<f64>> = vec![
        plane(&|i, _| 985.0 + 0.1 * i),
        plane(&|_, j| 250.0 + j),
        plane(&|i, _| 20.0 + 0.05 * i + hf + k),
        plane(&|_, j| 12.0 + 0.02 * j),
        plane(&|_, _| 2.0),
        plane(&|_, _| 4.0),
    ];
    let v3: Vec<Vec<f64>> = (0..7)
        .map(|f| (0..NZ * n).map(|c| match f { 0 => 980.0 - 110.0 * (c / n) as f64, 1 => 300.0 + 1100.0 * (c / n) as f64, 2 => 25.0 - 7.0 * (c / n) as f64,
            3 => 18.0 - 9.0 * (c / n) as f64, 4 => 3.0, 5 => 6.0, _ => 0.0 }).collect())
        .collect();
    let precip = plane(&|i, j| 0.5 * i + 0.25 * j + 10.0 * hf + k);
    let mslp = plane(&|_, j| 1012.0 + 0.01 * j - hf);
    let gust = plane(&|i, _| 9.0 + 0.1 * i);
    let wspd = plane(&|_, _| 10.0);
    let wdir = plane(&|i, _| if (i as usize) % 2 == 0 { 350.0 } else { 10.0 });
    let fzl = plane(&|i, _| 1500.0 + 2.0 * i);
    let gmax = plane(&|i, _| 15.0 + 0.1 * i);
    let domain = "d01";
    let w = ColumnsWrite {
        nx: NX, ny: NY, nz: NZ, lat: Values::F64(&lat), lon: Values::F64(&lon),
        fields3: (0..7).map(|i| Field3 { name: FIELDS_3D[i], units: UNITS_3D[i], quantum: Some(QUANTA_3D[i]), data: Values::F64(&v3[i]), scale: 1.0, offset: 0.0, affine: None }).collect(),
        fields2: (0..6).map(|i| Field2 { name: FIELDS_2D[i + 2], units: UNITS_2D[i + 2], data: Values::F64(&s2[i]), scale: 1.0, offset: 0.0 }).collect(),
        domain, member, valid_unix: T0 + h * 3600, tile: 16, meta: json!({"writer": "point_fixture"}),
    };
    let mut fields: Vec<Field2> = (0..6).map(|i| Field2 { name: FIELDS_2D[i + 2], units: UNITS_2D[i + 2], data: Values::F64(&s2[i]), scale: 1.0, offset: 0.0 }).collect();
    for (name, units, data) in [("mslp_hpa", "hPa", &mslp), ("precip_mm", "mm", &precip), ("gust_ms", "m/s", &gust), ("wspd80_ms", "m/s", &wspd),
        ("wdir80_deg", "degrees", &wdir), ("freezing_level_agl_m", "m", &fzl)] {
        fields.push(Field2 { name, units, data: Values::F64(data), scale: 1.0, offset: 0.0 });
    }
    if extra {
        fields.push(Field2 { name: "gust_max_ms", units: "m/s", data: Values::F64(&gmax), scale: 1.0, offset: 0.0 });
    }
    let s = SurfaceWrite { lat: w.lat, lon: w.lon, fields, notes: json!({"absent": {}}) };
    let raw = columns::finish(columns::encode_main(&w).unwrap(), Some(&s)).unwrap();
    let dir = if member.is_empty() { store.join(domain) } else { store.join("members").join(member).join(domain) };
    let stamp = columns::iso_time(T0 + h * 3600).replace(['-', ':'], "");
    columns::write_bytes(&dir.join(format!("{stamp}.wcol")), &raw).unwrap();
}

fn main() {
    let out = std::path::PathBuf::from(std::env::args().nth(1).expect("OUT_DIR"));
    let det = out.join("det").join("columns");
    for h in 0..3 {
        write(&det, "", h, 0.0, h == 2);
    }
    let ens = out.join("ens").join("columns");
    for (k, m) in ["m01", "m02", "m03"].iter().enumerate() {
        for h in 0..3 {
            write(&ens, m, h, (k + 1) as f64, false);
        }
    }
    for store in [&det, &ens] {
        let w = series::write_series(store, "d01", 4).unwrap();
        assert!(w.skipped.is_none(), "{:?}", w.skipped);
        println!("{}", serde_json::to_string(&w).unwrap());
    }
}
