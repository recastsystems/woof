//! WRF V3.5 windturbines_spec file. Public-domain source pins are in
//! tools/wind_legacy_wrf35_oracle/native-pin.json. No sampled curve replaces
//! the native analytic dragcof function: its six parameters remain explicit.
use rayon::prelude::*;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{error::Error, fs, path::Path};

pub fn read(path: &Path) -> Result<Value, Box<dyn Error>> {
    let bytes = fs::read(path)?;
    let text = std::str::from_utf8(&bytes)?;
    // Native READ(A256) truncates records before its whole-record '!' check.
    let lines: Vec<_> = text.lines().map(|s| s.chars().take(256).collect::<String>())
        .enumerate().filter(|(_, s)| !s.contains('!')).collect();
    let rows: Vec<Result<Value, String>> = lines.par_iter().map(|(i, line)| {
        let fields: Vec<_> = line.split(|c:char| c.is_whitespace() || c==',').filter(|s| !s.is_empty()).collect();
        if fields.len() < 9 { return Err(format!("windspec line {} needs nine fields", i+1)); }
        let id: i32 = fields[0].parse().map_err(|_| format!("windspec line {}: grid id must be an integer", i+1))?;
        let v: Vec<f32> = fields[1..9].iter().map(|s| s.replace(['d','D'],"e").parse::<f32>()
            .map_err(|_| format!("windspec line {}: invalid real", i+1)))
            .collect::<Result<_,_>>()?;
        if v.iter().any(|x| !x.is_finite()) || !(-90.0..=90.0).contains(&v[0])
            || !(-180.0..=180.0).contains(&v[1]) || v[3] <= 0.0 || v[2] <= v[3]/2.0
            || !(0.0..=1.0).contains(&v[4]) || v[5] <= 0.0 || v[6] < 0.0 || v[7] <= v[6] {
            return Err(format!("windspec line {}: invalid grid, location, rotor or coefficients", i+1));
        }
        Ok(json!({"id":format!("windspec:{}",i+1), "grid_id":id,
            "latitude":v[0],"longitude":v[1],"hub_height_m":v[2],"rotor_diameter_m":v[3],
            "standing_ct":v[4],"rated_power_mw":v[5],"cut_in_m_s":v[6],"cut_out_m_s":v[7]}))
    }).collect();
    let rows: Vec<_> = rows.into_iter().collect::<Result<_,_>>()?;
    Ok(json!({"schema":"gpuwm-wind-legacy.v1","turbines":rows,
        "provenance":{"format":"WRF V3.5 windturbines_spec","input_sha256":format!("{:x}",Sha256::digest(&bytes)),
            "source_ref":"98953e87c71ff8be232d7f56a1f9f8fa007efd84","workers":2}}))
}

pub fn ideal(p: &Value) -> Result<Value, Box<dyn Error>> {
    let integer=|k:&str| -> Result<i64,Box<dyn Error>> {p[k].as_i64().ok_or_else(||format!("missing integer {k}").into())};
    let nx=integer("nx")?; let ny=integer("ny")?;
    let x0=integer("td_pwfx")?.max(1);let y0=integer("td_pwfy")?.max(1);
    let x1=(integer("td_pwfx")?+integer("td_ewfx")?).min(nx+1);
    let y1=(integer("td_pwfy")?+integer("td_ewfy")?).min(ny+1);
    let coordinates:Vec<_>=if integer("grid_id")?==integer("td_turbgridid")? {
        (y0..y1).flat_map(|y|(x0..x1).map(move |x|(x,y))).collect()
    } else {Vec::new()};
    let turbines:Vec<_>=coordinates.par_iter().map(|(x,y)|json!({"id":format!("ideal:{x}:{y}"),
        "grid_id":p["grid_id"],"grid_x":x-1,"grid_y":y-1,
        "hub_height_m":p["td_hubheight"],"rotor_diameter_m":p["td_diameter"],
        "standing_ct":p["td_stdthrcoef"],"rated_power_mw":p["td_power"],
        "cut_in_m_s":p["td_cutinspeed"],"cut_out_m_s":p["td_cutoutspeed"]})).collect();
    Ok(json!({"schema":"gpuwm-wind-legacy.v1","turbines":turbines,
        "provenance":{"format":"WRF V3.5 windturbines_spec=ideal","parameters":p,"workers":2}}))
}
