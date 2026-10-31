//! WRF windturbines.txt and wind-turbine-N.tbl input, in SI units.
//! Layout and coefficient conversion: WRF module_wind_fitch.F, public domain.
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, error::Error, fs, path::Path};

fn lines(path: &Path) -> Result<Vec<Vec<String>>, Box<dyn Error>> {
    Ok(fs::read_to_string(path)?.lines().filter_map(|line| {
        let clean = line.split(['!', '#']).next().unwrap_or("").trim();
        if clean.is_empty() { None } else {
            Some(clean.split_whitespace().map(str::to_owned).collect())
        }
    }).collect())
}

fn number(text: &str) -> Result<f64, Box<dyn Error>> {
    let value: f64 = text.replace(['d', 'D'], "e").parse()?;
    if !value.is_finite() { return Err("nonfinite turbine value".into()); }
    Ok(value)
}

pub fn read(path: &Path) -> Result<Value, Box<dyn Error>> {
    let rows = lines(path)?;
    let mut types = BTreeMap::new();
    let mut turbines = Vec::new();
    let mut inputs = BTreeMap::new();
    inputs.insert(path.file_name().ok_or("missing turbine filename")?.to_string_lossy().to_string(),
                  format!("{:x}", Sha256::digest(fs::read(path)?)));
    for (i, row) in rows.iter().enumerate() {
        if row.len() < 3 || row.len() > 4 {
            return Err(format!("turbine row {} needs latitude longitude type [farm_id]", i+1).into());
        }
        let lat = number(&row[0])?;
        let lon = number(&row[1])?;
        if !(-90.0..=90.0).contains(&lat) || !(-180.0..=180.0).contains(&lon) {
            return Err(format!("invalid geographic turbine row {}", i+1).into());
        }
        let kind: u32 = row[2].parse()?;
        if kind == 0 { return Err("turbine type must be positive".into()); }
        let key = kind.to_string();
        if !types.contains_key(&key) {
            let name = format!("wind-turbine-{kind}.tbl");
            let table_path = path.parent().unwrap_or(Path::new(".")).join(&name);
            let table = lines(&table_path)?;
            if table.len() < 4 || table[0].len() != 1 || table[1].len() != 4 {
                return Err(format!("{name}: expected row count, hub diameter standing_CT rated_kW, then curve rows").into());
            }
            let n: usize = table[0][0].parse()?;
            if n < 2 || table.len() != n + 2 { return Err(format!("{name}: curve row count mismatch").into()); }
            let header: Vec<f64> = table[1].iter().map(|s| number(s)).collect::<Result<_,_>>()?;
            let (hub, diameter, standing, rated) = (header[0], header[1], header[2], header[3]);
            if diameter <= 0.0 || hub <= diameter/2.0 || standing < 0.0 || rated <= 0.0 {
                return Err(format!("{name}: invalid rotor geometry, standing CT or capacity").into());
            }
            let area = std::f64::consts::PI * diameter * diameter / 4.0;
            let mut curve = Vec::new();
            let mut previous = -1.0;
            for fields in &table[2..] {
                if fields.len() != 3 { return Err(format!("{name}: curve needs speed CT power_kW").into()); }
                let v: Vec<f64> = fields.iter().map(|s| number(s)).collect::<Result<_,_>>()?;
                if v[0] <= previous || v[1] < 0.0 || v[2] < 0.0 || (v[0] == 0.0 && v[2] != 0.0) {
                    return Err(format!("{name}: increasing nonnegative speed, CT and power required").into());
                }
                previous = v[0];
                let cp = if v[0] == 0.0 { 0.0 } else { v[2]*1000.0/(0.5*1.23*area*v[0].powi(3)) };
                curve.push(vec![v[0], v[1], cp]);
            }
            let spec = json!({"curve":curve,"hub_height_m":hub,"rotor_diameter_m":diameter,
                "rated_capacity_kw":rated,"curve_source":"WRF turbine table",
                "cut_in_m_s":curve[0][0],"cut_out_m_s":curve[n-1][0],
                "standing_ct":standing,"cutout_ct":curve[n-1][1],"reference_density_kg_m3":1.23,
                "wrf_curve":true});
            inputs.insert(name, format!("{:x}",Sha256::digest(fs::read(table_path)?)));
            types.insert(key.clone(), spec);
        }
        let spec = &types[&key];
        turbines.push(json!({"id":format!("row-{:06}",i+1),"type":key,"latitude":lat,"longitude":lon,
            "farm_id":row.get(3).map(String::as_str).unwrap_or("farm"),
            "hub_height_m":spec["hub_height_m"],"rotor_diameter_m":spec["rotor_diameter_m"],
            "rated_capacity_kw":spec["rated_capacity_kw"],"operating":true}));
    }
    Ok(json!({"schema":"gpuwm-wind-farm.v1","types":types,"turbines":turbines,
        "options":{"tke_factor":0.25},"provenance":{"format":"WRF Fitch","input_sha256":inputs}}))
}
