//! A minimal, wrfout-shaped classic NetCDF file written by
//! `netcdf-writer`, used by the stored-plane tests.
//!
//! It is deliberately NOT a physics-plausible forecast: it carries the
//! smallest set of variables `wrf-core` needs to open a raw wrfout (`T`
//! for the dimension probe, `Times` for the time axis, `XLAT`/`XLONG`
//! for the grid) plus a handful of ordinary surface planes and ONE plane
//! no catalog in the tree knows about.  That last plane is the point:
//! it stands for a variable a user added to their own WRF Registry, and
//! the `var:` product has to reach it without a line of new product
//! code.

use std::path::{Path, PathBuf};

use netcdf_writer::{AttrValue, NcFormat, NcType, NcWriter, Schema, VarData};

/// The wrfout variable no product catalog in this tree knows about.
pub const USER_PLANE: &str = "MSLP_ANOM";

/// The store name the generic `var:` route must be able to name.
pub const USER_PLANE_STORE_NAME: &str = "wrf_mslp_anom";

/// Units carried on [`USER_PLANE`], so the panel legend can be checked.
pub const USER_PLANE_UNITS: &str = "Pa";

pub const NX: usize = 24;
pub const NY: usize = 18;
pub const NZ: usize = 4;

/// The column-plane inputs: a dry column mass on full eta levels with no
/// `DNW` (this tree's history stream writes `ZNW` and not the
/// thicknesses), one cloud water value on every level, no cloud ice and
/// one skin temperature.  The pressures below put only the top level
/// below freezing, and the heights put that level in the 3 to 6 km layer
/// above ground, so the water path each row integrates is known.
pub const DRY_COLUMN_MASS_PA: f32 = 96_000.0;
pub const ETA_FULL_LEVELS: [f32; NZ + 1] = [1.0, 0.75, 0.5, 0.25, 0.0];
pub const CLOUD_WATER_KG_PER_KG: f32 = 1.0e-4;
pub const SKIN_TEMPERATURE_K: f32 = 290.0;
/// Potential temperature on every level (the stored `T` is theta - 300).
pub const THETA_K: f64 = 310.0;
/// The terrain height every cell carries, m.
pub const TERRAIN_M: f64 = 320.0;

/// Full pressure on mass level `k`, Pa: `97_000 - 12_000 k`.
#[allow(dead_code)]
pub fn pressure_pa(level: usize) -> f64 {
    97_000.0 - 12_000.0 * level as f64
}

/// Height above sea level of mass level `k`, m: the full levels sit at
/// `TERRAIN_M + 1000 k`, so a mass level is 500 m above its lower one.
#[allow(dead_code)]
pub fn mass_level_height_msl_m(level: usize) -> f64 {
    TERRAIN_M + 1_000.0 * level as f64 + 500.0
}

/// Value at cell `(y, x)` of the user plane; a deterministic ramp, so a
/// reader can prove it read THIS plane and not some neighbour's.
pub fn user_plane_value(y: usize, x: usize) -> f32 {
    (y * NX + x) as f32 - 120.0
}

/// Write the fixture as `wrfout_d01_<stamp>` inside `dir`, returning its path.
pub fn write(dir: &Path) -> PathBuf {
    write_frame(dir, "2026-08-19_00:00:00", None)
}

/// A timestamped cumulative-rain frame for the native CLI's series tests.
#[allow(dead_code)]
pub fn write_rain_frame(dir: &Path, lead_seconds: i64, rain_total: f32) -> PathBuf {
    let init = chrono::NaiveDate::from_ymd_opt(2026, 8, 19)
        .unwrap()
        .and_hms_opt(0, 0, 0)
        .unwrap();
    let valid = init + chrono::Duration::seconds(lead_seconds);
    write_frame(
        dir,
        &valid.format("%Y-%m-%d_%H:%M:%S").to_string(),
        Some(rain_total),
    )
}

/// The fixture plus extra surface planes, each `(name, units, value)`
/// written as one constant over the grid. An empty `units` writes the
/// attribute empty, the way a file that states no unit carries it.
#[allow(dead_code)]
pub fn write_with_surface_planes(dir: &Path, extras: &[(&str, &str, f32)]) -> PathBuf {
    write_frame_with(dir, "2026-08-19_00:00:00", None, extras)
}

fn write_frame(dir: &Path, valid_time: &str, rain_total: Option<f32>) -> PathBuf {
    write_frame_with(dir, valid_time, rain_total, &[])
}

fn write_frame_with(
    dir: &Path,
    valid_time: &str,
    rain_total: Option<f32>,
    extras: &[(&str, &str, f32)],
) -> PathBuf {
    write_frame_with_grid(dir, valid_time, rain_total, extras, false)
}

/// The same analytic surface fixture on its regular geographic lattice.
#[allow(dead_code)]
pub fn write_regular_rain_frame(dir: &Path, lead_seconds: i64, rain_total: f32) -> PathBuf {
    let init = chrono::NaiveDate::from_ymd_opt(2026, 8, 19)
        .unwrap()
        .and_hms_opt(0, 0, 0)
        .unwrap();
    let valid = init + chrono::Duration::seconds(lead_seconds);
    write_frame_with_grid(
        dir,
        &valid.format("%Y-%m-%d_%H:%M:%S").to_string(),
        Some(rain_total),
        &[],
        true,
    )
}

#[allow(dead_code)]
pub fn write_regular_no_rain_frame(dir: &Path) -> PathBuf {
    write_frame_with_grid(dir, "2026-08-19_00:00:00", None, &[], true)
}

#[allow(dead_code)]
pub fn write_regular_shortwave_frame(dir: &Path, lead_seconds: i64, flux: f32) -> PathBuf {
    let init = chrono::NaiveDate::from_ymd_opt(2026, 8, 19)
        .unwrap().and_hms_opt(0, 0, 0).unwrap();
    let valid = init + chrono::Duration::seconds(lead_seconds);
    write_frame_with_grid(dir, &valid.format("%Y-%m-%d_%H:%M:%S").to_string(),
        None, &[("SWDOWN", "W m-2", flux)], true)
}

#[allow(dead_code)]
pub fn write_regular_fill_wind_frame(dir: &Path) -> PathBuf {
    write_frame_with_fill(dir, "2026-08-19_00:00:00", None, &[], true, true)
}

#[allow(dead_code)]
pub fn write_regular_extrema_frame(dir:&Path,lead_seconds:i64,rain_total:f32,
    maximum:f32,minimum:f32,attrs:&[(&str,AttrValue)])->PathBuf {
    let init=chrono::NaiveDate::from_ymd_opt(2026,8,19).unwrap().and_hms_opt(0,0,0).unwrap();
    let valid=init+chrono::Duration::seconds(lead_seconds);
    write_frame_with_attrs(dir,&valid.format("%Y-%m-%d_%H:%M:%S").to_string(),Some(rain_total),
        &[("UP_HELI_MAX","m2 s-2",maximum),("UP_HELI_MIN","m2 s-2",minimum)],true,false,attrs)
}

#[allow(dead_code)]
pub fn write_regular_extrema_fill_frame(dir:&Path,lead_seconds:i64,rain_total:f32,
    maximum:f32,minimum:f32)->PathBuf {
    let init=chrono::NaiveDate::from_ymd_opt(2026,8,19).unwrap().and_hms_opt(0,0,0).unwrap();
    let valid=init+chrono::Duration::seconds(lead_seconds);
    write_frame_with_attrs(dir,&valid.format("%Y-%m-%d_%H:%M:%S").to_string(),Some(rain_total),
        &[("UP_HELI_MAX","m2 s-2",maximum),("UP_HELI_MIN","m2 s-2",minimum)],true,true,
        &[("GPUWM_EXTREME_INTERVAL_SECONDS",AttrValue::Ints(vec![3600]))])
}

#[allow(dead_code)]
pub fn write_regular_downward_frame(dir:&Path,lead_seconds:i64,rain_total:f32,downward:f32)->PathBuf {
    let init=chrono::NaiveDate::from_ymd_opt(2026,8,19).unwrap().and_hms_opt(0,0,0).unwrap();
    let valid=init+chrono::Duration::seconds(lead_seconds);
    write_frame_with_attrs(dir,&valid.format("%Y-%m-%d_%H:%M:%S").to_string(),Some(rain_total),
        &[("W_DN_MAX","m s-1",downward)],true,false,
        &[("GPUWM_EXTREME_INTERVAL_SECONDS",AttrValue::Ints(vec![1800]))])
}

fn write_frame_with_grid(
    dir: &Path,
    valid_time: &str,
    rain_total: Option<f32>,
    extras: &[(&str, &str, f32)],
    regular: bool,
) -> PathBuf {
    write_frame_with_fill(dir, valid_time, rain_total, extras, regular, false)
}

fn write_frame_with_fill(
    dir: &Path,
    valid_time: &str,
    rain_total: Option<f32>,
    extras: &[(&str, &str, f32)],
    regular: bool,
    finite_fill: bool,
) -> PathBuf {
    write_frame_with_attrs(dir,valid_time,rain_total,extras,regular,finite_fill,&[])
}

fn write_frame_with_attrs(
    dir:&Path,valid_time:&str,rain_total:Option<f32>,extras:&[(&str,&str,f32)],
    regular:bool,finite_fill:bool,attrs:&[(&str,AttrValue)],
)->PathBuf {
    let path = dir.join(format!("wrfout_d01_{}", valid_time.replace(':', "_")));
    let cells = NX * NY;
    let volume = cells * NZ;

    let mut schema = Schema::new(NcFormat::Offset64);
    for (name,value) in attrs {schema.put_global_attr(*name,value.clone()).unwrap();}
    let time = schema.def_dim("Time", 0, true).unwrap();
    let strlen = schema.def_dim("DateStrLen", 19, false).unwrap();
    let bottom_top = schema.def_dim("bottom_top", NZ, false).unwrap();
    let bottom_top_stag = schema.def_dim("bottom_top_stag", NZ + 1, false).unwrap();
    let south_north = schema.def_dim("south_north", NY, false).unwrap();
    let south_north_stag = schema.def_dim("south_north_stag", NY + 1, false).unwrap();
    let west_east = schema.def_dim("west_east", NX, false).unwrap();
    let west_east_stag = schema.def_dim("west_east_stag", NX + 1, false).unwrap();

    for (name, value) in [
        ("TITLE", " OUTPUT FROM WRF V4.6.1 MODEL"),
        ("START_DATE", "2026-08-19_00:00:00"),
        ("SIMULATION_START_DATE", "2026-08-19_00:00:00"),
        ("GRIDTYPE", "C"),
    ] {
        schema
            .put_global_attr(name, AttrValue::Text(value.into()))
            .unwrap();
    }
    for (name, value) in [
        ("MAP_PROJ", if regular { 6i32 } else { 1i32 }),
        ("GRID_ID", 1),
        ("PARENT_ID", 0),
    ] {
        schema
            .put_global_attr(name, AttrValue::Ints(vec![value]))
            .unwrap();
    }
    for (name, value) in [
        ("DX", 3000.0f32),
        ("DY", 3000.0),
        ("TRUELAT1", 38.0),
        ("TRUELAT2", 38.0),
        ("STAND_LON", -95.0),
        ("CEN_LAT", 38.0),
        ("CEN_LON", -95.0),
        ("POLE_LAT", 90.0),
        ("POLE_LON", 0.0),
    ] {
        schema
            .put_global_attr(name, AttrValue::Floats(vec![value]))
            .unwrap();
    }

    let times = schema
        .def_var("Times", NcType::Char, &[time, strlen])
        .unwrap();

    let surface = |schema: &mut Schema, name: &str, units: &str| {
        let id = schema
            .def_var(name, NcType::Float, &[time, south_north, west_east])
            .unwrap();
        schema
            .put_var_attr(id, "units", AttrValue::Text(units.into()))
            .unwrap();
        schema
            .put_var_attr(id, "description", AttrValue::Text(name.into()))
            .unwrap();
        id
    };
    let xlat = surface(&mut schema, "XLAT", "degree_north");
    let xlong = surface(&mut schema, "XLONG", "degree_east");
    let t2 = surface(&mut schema, "T2", "K");
    let th2 = surface(&mut schema, "TH2", "K");
    let q2 = surface(&mut schema, "Q2", "kg kg-1");
    let psfc = surface(&mut schema, "PSFC", "Pa");
    let hgt = surface(&mut schema, "HGT", "m");
    let u10 = surface(&mut schema, "U10", "m s-1");
    let v10 = surface(&mut schema, "V10", "m s-1");
    // Identity grid rotation.  Without these two, every earth-relative
    // wind diagnostic (`uvmet10` and everything downstream of it) fails
    // and the file can draw no wind layer at all -- which is exactly the
    // layer the streamline front door has to be provable against.
    let sinalpha = surface(&mut schema, "SINALPHA", "1");
    let cosalpha = surface(&mut schema, "COSALPHA", "1");
    let mu = surface(&mut schema, "MU", "Pa");
    let mub = surface(&mut schema, "MUB", "Pa");
    let tsk = surface(&mut schema, "TSK", "K");
    let user = surface(&mut schema, USER_PLANE, USER_PLANE_UNITS);
    let extra_planes: Vec<_> = extras
        .iter()
        .map(|(name, units, value)| (surface(&mut schema, name, units), *value))
        .collect();
    let rain = rain_total.map(|total| {
        (
            surface(&mut schema, "RAINC", "mm"),
            surface(&mut schema, "RAINNC", "mm"),
            total,
        )
    });

    let volume_var = |schema: &mut Schema, name: &str, dims: &[usize], units: &str| {
        let id = schema.def_var(name, NcType::Float, dims).unwrap();
        schema
            .put_var_attr(id, "units", AttrValue::Text(units.into()))
            .unwrap();
        id
    };
    let p_top = volume_var(&mut schema, "P_TOP", &[time], "Pa");
    let p_hyd = volume_var(
        &mut schema,
        "P_HYD",
        &[time, bottom_top, south_north, west_east],
        "Pa",
    );
    let t = volume_var(
        &mut schema,
        "T",
        &[time, bottom_top, south_north, west_east],
        "K",
    );
    let p = volume_var(
        &mut schema,
        "P",
        &[time, bottom_top, south_north, west_east],
        "Pa",
    );
    let pb = volume_var(
        &mut schema,
        "PB",
        &[time, bottom_top, south_north, west_east],
        "Pa",
    );
    let qvapor = volume_var(
        &mut schema,
        "QVAPOR",
        &[time, bottom_top, south_north, west_east],
        "kg kg-1",
    );
    let qcloud = volume_var(
        &mut schema,
        "QCLOUD",
        &[time, bottom_top, south_north, west_east],
        "kg kg-1",
    );
    let qice = volume_var(
        &mut schema,
        "QICE",
        &[time, bottom_top, south_north, west_east],
        "kg kg-1",
    );
    // Rain and snow, as every microphysics wrfout carries them: visibility
    // requires both since 2c49173f80b (a frame without them would read clear
    // air through precipitation), so a fixture without them had no visibility.
    let qrain = volume_var(
        &mut schema,
        "QRAIN",
        &[time, bottom_top, south_north, west_east],
        "kg kg-1",
    );
    let qsnow = volume_var(
        &mut schema,
        "QSNOW",
        &[time, bottom_top, south_north, west_east],
        "kg kg-1",
    );
    let znw = volume_var(&mut schema, "ZNW", &[time, bottom_top_stag], "");
    let ph = volume_var(
        &mut schema,
        "PH",
        &[time, bottom_top_stag, south_north, west_east],
        "m2 s-2",
    );
    let phb = volume_var(
        &mut schema,
        "PHB",
        &[time, bottom_top_stag, south_north, west_east],
        "m2 s-2",
    );
    let u = volume_var(
        &mut schema,
        "U",
        &[time, bottom_top, south_north, west_east_stag],
        "m s-1",
    );
    let v = volume_var(
        &mut schema,
        "V",
        &[time, bottom_top, south_north_stag, west_east],
        "m s-1",
    );

    let mut lat = Vec::with_capacity(cells);
    let mut lon = Vec::with_capacity(cells);
    let mut t2_values = Vec::with_capacity(cells);
    let mut user_values = Vec::with_capacity(cells);
    for y in 0..NY {
        for x in 0..NX {
            lat.push(36.0 + 0.05 * y as f32);
            lon.push(-98.0 + 0.05 * x as f32);
            t2_values.push(295.0 + 0.1 * (x + y) as f32);
            user_values.push(user_plane_value(y, x));
        }
    }
    let q2_values = vec![0.010f32; cells];
    let psfc_values = vec![97_000.0f32; cells];
    let hgt_values = vec![TERRAIN_M as f32; cells];
    let mu_values = vec![0.0f32; cells];
    let mub_values = vec![DRY_COLUMN_MASS_PA; cells];
    let tsk_values = vec![SKIN_TEMPERATURE_K; cells];
    let u10_values = vec![6.0f32; cells];
    let v10_values = vec![-4.0f32; cells];
    let sinalpha_values = vec![0.0f32; cells];
    let cosalpha_values = vec![1.0f32; cells];

    // Perturbation potential temperature is stored as theta - 300 K.
    let theta_perturbation = vec![(THETA_K - 300.0) as f32; volume];
    let qvapor_values = vec![0.008f32; volume];
    let qcloud_values = vec![CLOUD_WATER_KG_PER_KG; volume];
    let qice_values = vec![0.0f32; volume];
    let qrain_values = vec![0.0f32; volume];
    let qsnow_values = vec![0.0f32; volume];
    let mut base_pressure = Vec::with_capacity(volume);
    let mut pressure_perturbation = Vec::with_capacity(volume);
    for level in 0..NZ {
        let base = pressure_pa(level) as f32;
        for _ in 0..cells {
            base_pressure.push(base);
            pressure_perturbation.push(0.0);
        }
    }
    // The hydrostatic carrier follows the dry eta column and total water,
    // independently of the fixture's nonhydrostatic P + PB planes.
    let model_top_pressure = 4_000.0f32;
    let total_water = 0.008f32 + CLOUD_WATER_KG_PER_KG;
    let interface_pressure: Vec<f32> = ETA_FULL_LEVELS.iter()
        .map(|eta| model_top_pressure + (1.0 + total_water) * DRY_COLUMN_MASS_PA * eta)
        .collect();
    let hydrostatic_levels: Vec<f32> = interface_pressure.windows(2)
        .map(|p| 0.5 * (p[0] + p[1])).collect();
    let hydrostatic_pressure: Vec<f32> = hydrostatic_levels.iter()
        .flat_map(|p| std::iter::repeat_n(*p, cells)).collect();
    let specific_humidity = 0.008f32 / 1.008f32;
    let layer_temperatures: Vec<f32> = hydrostatic_levels.iter()
        .map(|p| THETA_K as f32 * (p * 1.0e-5).powf(0.28589641)).collect();
    let mut vapor_pressure = 0.0f32;
    for k in (0..NZ).rev() {
        vapor_pressure += 9.81 * (hydrostatic_levels[k] / (287.04 * layer_temperatures[k]))
            * 1_000.0 * specific_humidity;
    }
    let shelter_pressure = (DRY_COLUMN_MASS_PA + model_top_pressure + vapor_pressure)
        * (-0.068283 / layer_temperatures[0]).exp();
    let theta2_values: Vec<f32> = t2_values.iter()
        .map(|t| t / (shelter_pressure * 1.0e-5).powf(0.28589641)).collect();
    let mut base_geopotential = Vec::with_capacity(cells * (NZ + 1));
    for level in 0..=NZ {
        let value = 9.81 * (TERRAIN_M as f32 + 1_000.0 * level as f32);
        for _ in 0..cells {
            base_geopotential.push(value);
        }
    }
    let geopotential_perturbation = vec![0.0f32; cells * (NZ + 1)];
    let mut u_values = vec![7.0f32; (NX + 1) * NY * NZ];
    if finite_fill {
        u_values[0] = 9.96921e36;
    }
    let v_values = vec![-3.0f32; NX * (NY + 1) * NZ];

    let mut writer = NcWriter::create(&path, schema).unwrap();
    for (id, value) in extra_planes {
        let mut values=vec![value;cells];
        if finite_fill {values[0]=9.96921e36;}
        writer
            .write_record(0, id, VarData::F32(&values))
            .unwrap();
    }
    writer
        .write_record(0, times, VarData::Char(valid_time.as_bytes()))
        .unwrap();
    if let Some((rainc, rainnc, total)) = rain {
        writer
            .write_record(0, rainc, VarData::F32(&vec![0.0; cells]))
            .unwrap();
        let mut values=vec![total;cells];
        if finite_fill {values[0]=9.96921e36;}
        writer
            .write_record(0, rainnc, VarData::F32(&values))
            .unwrap();
    }
    for (id, values) in [
        (xlat, &lat),
        (xlong, &lon),
        (t2, &t2_values),
        (th2, &theta2_values),
        (q2, &q2_values),
        (psfc, &psfc_values),
        (hgt, &hgt_values),
        (u10, &u10_values),
        (v10, &v10_values),
        (sinalpha, &sinalpha_values),
        (cosalpha, &cosalpha_values),
        (mu, &mu_values),
        (mub, &mub_values),
        (tsk, &tsk_values),
        (user, &user_values),
    ] {
        writer
            .write_record(0, id, VarData::F32(values.as_slice()))
            .unwrap();
    }
    writer
        .write_record(0, znw, VarData::F32(&ETA_FULL_LEVELS))
        .unwrap();
    writer.write_record(0, p_top, VarData::F32(&[model_top_pressure])).unwrap();
    writer.write_record(0, p_hyd, VarData::F32(&hydrostatic_pressure)).unwrap();
    for (id, values) in [
        (t, &theta_perturbation),
        (p, &pressure_perturbation),
        (pb, &base_pressure),
        (qvapor, &qvapor_values),
        (qcloud, &qcloud_values),
        (qice, &qice_values),
        (qrain, &qrain_values),
        (qsnow, &qsnow_values),
        (ph, &geopotential_perturbation),
        (phb, &base_geopotential),
        (u, &u_values),
        (v, &v_values),
    ] {
        writer
            .write_record(0, id, VarData::F32(values.as_slice()))
            .unwrap();
    }
    writer.finish().unwrap();
    path
}
