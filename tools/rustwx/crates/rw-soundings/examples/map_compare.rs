//! Compare a column file's surface planes with the map renderer's own planes for the same frame.
//!
//! usage: map_compare STORE.wcol MAP_HOUR.rws [PAIR...]
//!
//! `MAP_HOUR.rws` is the rw-store v1 hour `rw_wrfbatch --store-root` writes while it draws the maps (every 2-D field
//! it drew, as stored). Each PAIR is `map_name=plane[:scale]` (the map's value times `scale` is the plane's unit, e.g.
//! `mslp=mslp_hpa:0.01`); with no pair the default pairs are tried. Prints, per pair, how many grid points differ
//! (bit for bit after the unit scale, rounded to f32 once as the store does) and the largest difference. With no
//! arguments after the files it also lists the map hour's 2-D variables.
use rw_store::columns::ColumnFile;
use rw_store::reader::HourReader;
use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 2 {
        eprintln!("usage: map_compare STORE.wcol MAP_HOUR.rws [map_name=plane[:scale]]...");
        std::process::exit(2);
    }
    let c = ColumnFile::open(Path::new(&args[0])).expect("column file");
    let sf = c.read_full_surface().expect("surface").expect("the column file has no surface section");
    let names = c.header.surface.as_ref().unwrap().fields.clone();
    let r = HourReader::open(Path::new(&args[1])).expect("map hour");
    let vars: Vec<String> = r.meta().variables.iter().filter(|v| v.kind == "surface2d").map(|v| v.name.clone()).collect();
    println!("MAP-VARIABLES {}", vars.join(" "));
    let wrap = c.header.wrap_x.is_some();
    let (nx, ny) = (c.header.nx, c.header.ny);
    let src_nx = if wrap { nx - 1 } else { nx };
    let defaults = [
        "mslp=mslp_hpa:0.01",
        "apcp=precip_mm",
        "wind_gust_10m=gust_ms",
        "sbcape=sbcape_jkg",
        "sbcin=sbcin_jkg",
        "mlcape=mlcape_jkg",
        "mucape=mucape_jkg",
        "wrf_cloudfrac_low=cloud_low_pct",
        "wrf_cloudfrac_mid=cloud_mid_pct",
        "wrf_cloudfrac_high=cloud_high_pct",
        "composite_reflectivity=refl_comp_dbz",
        "visibility_surface=visibility_m",
    ];
    let pairs: Vec<String> = if args.len() > 2 { args[2..].to_vec() } else { defaults.iter().map(|s| s.to_string()).collect() };
    let mut failed = 0usize;
    for pair in pairs {
        let (map, rest) = pair.split_once('=').expect("map_name=plane[:scale]");
        let (plane, scale) = match rest.split_once(':') {
            Some((p, s)) => (p, s.parse::<f64>().expect("scale")),
            None => (rest, 1.0),
        };
        // A map name is matched exactly, else by the first stored variable that contains it.
        let Some(var) = vars.iter().find(|v| *v == map).or_else(|| vars.iter().find(|v| v.contains(map))) else {
            println!("PAIR {map} -> {plane}: the map hour has no such variable");
            continue;
        };
        let Some(pi) = names.iter().position(|n| n == plane) else {
            println!("PAIR {var} -> {plane}: the column file has no such plane");
            continue;
        };
        let m = r.read_full_2d(var).expect("map plane");
        let (mut differ, mut nan_mismatch, mut maxd) = (0usize, 0usize, 0f64);
        for y in 0..ny {
            for x in 0..src_nx {
                let want = (m[y * src_nx + x] as f64 * scale) as f32;
                let want = if scale == 1.0 { m[y * src_nx + x] } else { want };
                let got = sf[pi][y * nx + x];
                if want.is_nan() || got.is_nan() {
                    if want.is_nan() != got.is_nan() {
                        nan_mismatch += 1;
                    }
                    continue;
                }
                if want.to_bits() != got.to_bits() {
                    differ += 1;
                    maxd = maxd.max((want as f64 - got as f64).abs());
                }
            }
        }
        println!(
            "PAIR {var} -> {plane} (x{scale}): points={} differing={differ} nan_mismatch={nan_mismatch} max_abs_diff={maxd:.6}",
            src_nx * ny
        );
        failed += differ + nan_mismatch;
    }
    println!("TOTAL differing={failed}");
}
