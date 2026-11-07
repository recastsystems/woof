//! WRF auxiliary stream 4 producer. Public-domain WRF v4.7.1
//! dyn_em/module_initialize_real.F:1328-1389,8029-8095 owns monthly interpolation.
//! Files stay collocated. No real.exe process or Python numerical path is used.
use chrono::{Datelike, NaiveDate, NaiveDateTime};
use netcdf_writer::{AttrValue, NcFormat, NcType, NcWriter, Schema, VarData};
use netcrust::File;
use rayon::prelude::*;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    fs,
    path::{Path, PathBuf},
};

type Result<T> = std::result::Result<T, String>;
const FIELDS: [(&str, &str); 4] = [
    ("SST", "K"),
    ("SEAICE", "1"),
    ("VEGFRA", "%"),
    ("ALBBCK", "1"),
];
fn fail(e: impl std::fmt::Display) -> String {
    e.to_string()
}
fn numeric(file: &File, name: &str) -> Result<Vec<f32>> {
    let var = file
        .variable(name)
        .ok_or_else(|| format!("missing {name}"))?;
    let scale = var
        .attribute("scale_factor")
        .and_then(|a| a.as_f64())
        .unwrap_or(1.);
    let offset = var
        .attribute("add_offset")
        .and_then(|a| a.as_f64())
        .unwrap_or(0.);
    let missing: Vec<f64> = ["_FillValue", "missing_value"]
        .iter()
        .filter_map(|n| var.attribute(n))
        .filter_map(|a| a.value().as_f64_vec())
        .flatten()
        .collect();
    Ok(var
        .values_f64()
        .map_err(fail)?
        .par_iter()
        .map(|v| {
            if !v.is_finite() || missing.contains(v) {
                f32::NAN
            } else {
                (v * scale + offset) as f32
            }
        })
        .collect())
}
fn monthly_weights(date: NaiveDate) -> (usize, usize, f32, f32, f32) {
    let y = date.year();
    let target = date.ordinal() as i32;
    let mut middle = vec![0];
    for m in 1..=12 {
        middle.push(NaiveDate::from_ymd_opt(y, m, 15).unwrap().ordinal() as i32);
    }
    middle[0] = middle[1] - 31;
    middle.push(middle[12] + 31);
    let l = (0..13)
        .find(|&l| middle[l] < target && middle[l + 1] >= target)
        .unwrap();
    let (a, b) = if l == 0 || l == 12 {
        (11, 0)
    } else {
        (l - 1, l)
    };
    (
        a,
        b,
        (target - middle[l]) as f32,
        (middle[l + 1] - target) as f32,
        (middle[l + 1] - middle[l]) as f32,
    )
}
fn monthly(file: &File, name: &str, date: NaiveDate, n: usize) -> Result<Vec<f32>> {
    let v = numeric(file, name)?;
    if v.len() != 12 * n {
        return Err(format!(
            "{name} must contain twelve collocated monthly planes"
        ));
    }
    let (a, b, wb, wa, d) = monthly_weights(date);
    // Keep WRF's float32 multiplication, addition and division order.
    Ok((0..n)
        .into_par_iter()
        .map(|i| (v[b * n + i] * wb + v[a * n + i] * wa) / d)
        .collect())
}
#[derive(Debug)]
struct Frame {
    time: String,
    shape: (usize, usize),
    lat: Vec<f32>,
    lon: Vec<f32>,
    fields: [Vec<f32>; 4],
}
fn lapse_skin(tsk: f32, terrain: f32, soil_height: f32, land: f32) -> f32 {
    let delta = terrain - soil_height;
    if land > 0.5
        && soil_height >= -1000.
        && soil_height <= 10000.
        && delta >= -3000.
        && delta <= 3000.
    {
        tsk - 0.0065 * delta
    } else {
        tsk
    }
}
fn frame(path: &Path, surface_physics: i32) -> Result<Frame> {
    let f = File::open(path).map_err(fail)?;
    let times = f.read_strings("Times").map_err(fail)?;
    if times.len() != 1 {
        return Err("each producer input must contain exactly one valid time".into());
    }
    let time = times[0].trim_end_matches('\0').to_string();
    let date = NaiveDateTime::parse_from_str(&time, "%Y-%m-%d_%H:%M:%S")
        .map_err(fail)?
        .date();
    let ln = if f.variable("XLAT_M").is_some() {
        "XLAT_M"
    } else {
        "XLAT"
    };
    let xn = if f.variable("XLONG_M").is_some() {
        "XLONG_M"
    } else {
        "XLONG"
    };
    let dims = f.variable(ln).ok_or("missing mass-grid latitude")?.shape();
    if dims.len() != 3 || dims[0] != 1 {
        return Err("input must have one fixed mass-grid horizontal plane".into());
    }
    let shape = (dims[1], dims[2]);
    let n = shape.0 * shape.1;
    let lat = numeric(&f, ln)?;
    let lon = numeric(&f, xn)?;
    let land = numeric(&f, "LANDMASK")?;
    if lat.len() != n || lon.len() != n || land.len() != n {
        return Err("coordinate/mask shape mismatch".into());
    }
    if lat
        .par_iter()
        .zip(&lon)
        .any(|(a, b)| !a.is_finite() || !b.is_finite())
        || land.par_iter().any(|v| *v != 0. && *v != 1.)
    {
        return Err("producer needs finite coordinates and a binary LANDMASK".into());
    }
    // WRF fills a missing SST field from initialized TSK, not zero.
    // Metgrid names that field SKINTEMP.
    let have_sst = f.variable("SST").is_some()
        && f.attribute("FLAG_SST")
            .and_then(|a| a.as_f64())
            .unwrap_or(1.)
            == 1.;
    let mut sst = if have_sst {
        numeric(&f, "SST")?
    } else {
        numeric(
            &f,
            if f.variable("TSK").is_some() {
                "TSK"
            } else {
                "SKINTEMP"
            },
        )?
    };
    let ice = numeric(
        &f,
        if f.variable("SEAICE").is_some() {
            "SEAICE"
        } else {
            "XICE"
        },
    )?;
    if !have_sst && f.variable("TSK").is_none() && ice.par_iter().any(|ice| *ice > 0.) {
        return Err("SST is absent on iced water: supply initialized TSK; raw SKINTEMP has not undergone WRF's sea-ice surface initialization".into());
    }
    if !have_sst
        && f.variable("TSK").is_none()
        && matches!(surface_physics, 1 | 2 | 3 | 4 | 5 | 7 | 8)
        && f.attribute("FLAG_SOILHGT").and_then(|a| a.as_f64()) == Some(1.)
    {
        let terrain = numeric(&f, "HGT_M")?;
        let soil_height = numeric(&f, "SOILHGT")?;
        if terrain.len() != n || soil_height.len() != n {
            return Err("terrain and soil height must match the mass grid".into());
        }
        // WRF share/module_soil_pre.F:1005-1057. Missing SST is filled
        // after process_soil_real, including its land-only lapse adjustment.
        sst.par_iter_mut().enumerate().for_each(|(i, t)| {
            *t = lapse_skin(*t, terrain[i], soil_height[i], land[i]);
        });
    }
    let veg = if f.variable("VEGFRA").is_some() {
        numeric(&f, "VEGFRA")?
    } else {
        monthly(&f, "GREENFRAC", date, n)?
            .par_iter()
            .map(|v| v * 100.)
            .collect()
    };
    let alb = if f.variable("ALBBCK").is_some() {
        numeric(&f, "ALBBCK")?
    } else {
        monthly(&f, "ALBEDO12M", date, n)?
            .par_iter()
            .zip(&land)
            .map(|(v, l)| if *l < 0.5 { 0.08 } else { v / 100. })
            .collect()
    };
    let fields = [sst, ice, veg, alb];
    for (k, v) in fields.iter().enumerate() {
        if v.len() != n {
            return Err(format!("{} shape differs from mass grid", FIELDS[k].0));
        }
        let (lo, hi) = match k {
            0 => (150., 400.),
            1 => (0., 1.),
            2 => (0., 100.),
            _ => (0., 1.),
        };
        if v.par_iter()
            .zip(&land)
            .any(|(v, l)| (k >= 2 || *l < 0.5) && (!v.is_finite() || *v < lo || *v > hi))
        {
            return Err(format!(
                "{} contains invalid values on its surface ownership; supply initialized surface fields rather than guessing a repair",
                FIELDS[k].0
            ));
        }
    }
    Ok(Frame {
        time,
        shape,
        lat,
        lon,
        fields,
    })
}
fn produce(inputs: &[PathBuf], output: &Path, surface_physics: i32) -> Result<serde_json::Value> {
    if inputs.is_empty() || output.exists() {
        return Err("producer requires inputs and a new output path".into());
    }
    let mut frames = inputs
        .par_iter()
        .map(|p| frame(p, surface_physics))
        .collect::<Result<Vec<_>>>()?;
    frames.sort_by(|a, b| a.time.cmp(&b.time));
    let first = &frames[0];
    for (i, f) in frames.iter().enumerate() {
        if f.shape != first.shape || f.lat != first.lat || f.lon != first.lon {
            return Err("surface input grid moved; remap before producing stream 4".into());
        }
        if i > 0 && f.time == frames[i - 1].time {
            return Err("duplicate lower-boundary valid time".into());
        }
    }
    let source = File::open(&inputs[0]).map_err(fail)?;
    let mut static_fields = Vec::new();
    let mut s = Schema::new(NcFormat::Offset64);
    s.put_global_attr(
        "TITLE",
        AttrValue::Text("OUTPUT FROM RW-WPS WRF V4.7.1 AUXINPUT4".into()),
    )
    .map_err(fail)?;
    s.put_global_attr("SIMULATION_START_DATE", AttrValue::Text(first.time.clone()))
        .map_err(fail)?;
    let t = s.def_dim("Time", 0, true).map_err(fail)?;
    let c = s.def_dim("DateStrLen", 19, false).map_err(fail)?;
    let y = s
        .def_dim("south_north", first.shape.0, false)
        .map_err(fail)?;
    let x = s.def_dim("west_east", first.shape.1, false).map_err(fail)?;
    for key in [
        "GRID_ID",
        "DX",
        "DY",
        "MAP_PROJ",
        "TRUELAT1",
        "TRUELAT2",
        "STAND_LON",
        "CEN_LAT",
        "CEN_LON",
        "ISWATER",
        "ISICE",
        "ISLAKE",
    ] {
        if let Some(v) = source.attribute(key).and_then(|a| a.as_f64()) {
            s.put_global_attr(key, AttrValue::Floats(vec![v as f32]))
                .map_err(fail)?;
        }
    }
    if let Some(v) = source
        .attribute("MMINLU")
        .and_then(|a| a.as_string().map(str::to_owned))
    {
        s.put_global_attr("MMINLU", AttrValue::Text(v))
            .map_err(fail)?;
    }
    for name in ["LANDMASK", "LU_INDEX"] {
        let values = numeric(&source, name)?;
        if values.len() != first.shape.0 * first.shape.1 {
            return Err(format!("{name} must be one static plane"));
        }
        let id = s.def_var(name, NcType::Float, &[y, x]).map_err(fail)?;
        // Static values are written after the schema is sealed below.
        static_fields.push((id, values));
    }
    let tv = s.def_var("Times", NcType::Char, &[t, c]).map_err(fail)?;
    let mut ids = Vec::new();
    for (name, units) in [("XLAT", "degree_north"), ("XLONG", "degree_east")]
        .into_iter()
        .chain(FIELDS)
    {
        let id = s.def_var(name, NcType::Float, &[t, y, x]).map_err(fail)?;
        s.put_var_attr(id, "units", AttrValue::Text(units.into()))
            .map_err(fail)?;
        s.put_var_attr(id, "FieldType", AttrValue::Ints(vec![104]))
            .map_err(fail)?;
        s.put_var_attr(id, "MemoryOrder", AttrValue::Text("XY ".into()))
            .map_err(fail)?;
        s.put_var_attr(id, "stagger", AttrValue::Text("".into()))
            .map_err(fail)?;
        ids.push(id);
    }
    // create_new protects pre-existing output even if another producer races.
    let reservation = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)
        .map_err(fail)?;
    drop(reservation);
    let mut w = NcWriter::create(output, s).map_err(fail)?;
    for (id, values) in &static_fields {
        w.write_var(*id, VarData::F32(values)).map_err(fail)?;
    }
    for (i, f) in frames.iter().enumerate() {
        w.write_record(i as u64, tv, VarData::Char(f.time.as_bytes()))
            .map_err(fail)?;
        for (id, values) in ids
            .iter()
            .zip([&f.lat, &f.lon].into_iter().chain(f.fields.iter()))
        {
            w.write_record(i as u64, *id, VarData::F32(values))
                .map_err(fail)?;
        }
    }
    w.finish().map_err(fail)?;
    let digest = format!("{:x}", Sha256::digest(fs::read(output).map_err(fail)?));
    Ok(
        json!({"schema":"rw-wps.wrflowinp.v1","output":output,"sha256":digest,
        "records":frames.iter().map(|f|&f.time).collect::<Vec<_>>(),"shape":first.shape,
        "sf_surface_physics":surface_physics,"fields":FIELDS.iter().map(|f|f.0).collect::<Vec<_>>(),
        "source":"WRF-v4.7.1/dyn_em/module_initialize_real.F:1328-1389,8029-8095"}),
    )
}
pub fn run(inputs: &[PathBuf], output: &Path) -> Result<serde_json::Value> {
    run_with_surface_physics(inputs, output, 2)
}
pub fn run_with_surface_physics(
    inputs: &[PathBuf],
    output: &Path,
    surface_physics: i32,
) -> Result<serde_json::Value> {
    produce(inputs, output, surface_physics)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn metgrid_flag_selects_sst_or_initialized_skin() {
        // WRF module_initialize_real.F:3671 tests FLAG_SST, even if an
        // allocated SST plane is present. Exercise the real CDF reader.
        for flag in [0, 1] {
            let p =
                std::env::temp_dir().join(format!("wrf-surface-{}-{flag}.nc", std::process::id()));
            let mut s = Schema::new(NcFormat::Offset64);
            s.put_global_attr("FLAG_SST", AttrValue::Ints(vec![flag]))
                .unwrap();
            s.put_global_attr("FLAG_SOILHGT", AttrValue::Ints(vec![1]))
                .unwrap();
            let t = s.def_dim("Time", 1, false).unwrap();
            let c = s.def_dim("DateStrLen", 19, false).unwrap();
            let y = s.def_dim("south_north", 1, false).unwrap();
            let x = s.def_dim("west_east", 2, false).unwrap();
            let tv = s.def_var("Times", NcType::Char, &[t, c]).unwrap();
            let values = [
                ("XLAT_M", [30., 30.]),
                ("XLONG_M", [-80., -79.]),
                ("LANDMASK", [1., 0.]),
                ("SST", [295., 296.]),
                ("SKINTEMP", [300., 299.]),
                ("SEAICE", [0., 0.]),
                ("VEGFRA", [30., 0.]),
                ("ALBBCK", [0.2, 0.08]),
                ("HGT_M", [150., 0.]),
                ("SOILHGT", [50., 0.]),
            ];
            let ids: Vec<_> = values
                .iter()
                .map(|(n, _)| s.def_var(n, NcType::Float, &[t, y, x]).unwrap())
                .collect();
            let mut w = NcWriter::create(&p, s).unwrap();
            w.write_var(tv, VarData::Char(b"2019-09-04_21:00:00"))
                .unwrap();
            for (id, (_, v)) in ids.into_iter().zip(&values) {
                w.write_var(id, VarData::F32(v)).unwrap();
            }
            w.finish().unwrap();
            let got = frame(&p, 2).unwrap();
            let want = if flag == 1 {
                [295_f32, 296.]
            } else {
                [300_f32 - 0.0065_f32 * 100., 299.]
            };
            assert_eq!(
                got.fields[0]
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                want.map(f32::to_bits)
            );
            // Fixtures remain in TMPDIR for the test runner's cleanup.
        }
    }
    #[test]
    fn native_soil_lapse_and_skip_guards() {
        assert_eq!(
            lapse_skin(300., 150., 50., 1.).to_bits(),
            (300_f32 - 0.0065_f32 * 100.).to_bits()
        );
        for (h, sh, land) in [
            (150., 50., 0.),
            (150., -1001., 1.),
            (150., 10001., 1.),
            (3501., 500., 1.),
        ] {
            assert_eq!(lapse_skin(300., h, sh, land).to_bits(), 300_f32.to_bits());
        }
    }
    #[test]
    fn native_month_boundaries_and_leap_day() {
        assert_eq!(
            monthly_weights(NaiveDate::from_ymd_opt(2024, 2, 15).unwrap()),
            (0, 1, 31., 0., 31.)
        );
        assert_eq!(
            monthly_weights(NaiveDate::from_ymd_opt(2024, 3, 1).unwrap()),
            (1, 2, 15., 14., 29.)
        );
        assert_eq!(
            monthly_weights(NaiveDate::from_ymd_opt(2023, 1, 1).unwrap()),
            (11, 0, 17., 14., 31.)
        );
    }
}
