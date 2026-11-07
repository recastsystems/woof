//! WRF windturbines.txt and wind-turbine-N.tbl input, in SI units.
//! Layout and coefficient conversion: WRF module_wind_fitch.F, public domain.
use serde_json::{json, Value};
use rayon::prelude::*;
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

pub fn read(path: &Path, indexed: bool) -> Result<Value, Box<dyn Error>> {
    read_impl(path, indexed, 0)
}

pub fn read_mav(path: &Path, mode: u32) -> Result<Value, Box<dyn Error>> {
    if mode != 1 && mode != 2 { return Err("MAV windfarm_ij must be 1 (xy) or 2 (lat/lon)".into()); }
    read_impl(path, false, mode)
}

fn read_impl(path: &Path, indexed: bool, mav: u32) -> Result<Value, Box<dyn Error>> {
    let rows = lines(path)?;
    let mut types = BTreeMap::new();
    let mut turbines = Vec::new();
    let mut inputs = BTreeMap::new();
    inputs.insert(path.file_name().ok_or("missing turbine filename")?.to_string_lossy().to_string(),
                  format!("{:x}", Sha256::digest(fs::read(path)?)));
    for (i, row) in rows.iter().enumerate() {
        if (mav == 0 && (row.len() < 3 || row.len() > 4)) || (mav != 0 && row.len() != 4) {
            let coordinates = if indexed { "i j" } else { "latitude longitude" };
            return Err(format!("turbine row {} needs {coordinates} type [farm_id]", i+1).into());
        }
        let coordinates = if mav == 1 {
            json!({"grid_x":0,"grid_y":0,"mav_x":number(&row[0])?,"mav_y":number(&row[1])?})
        } else if indexed {
            // WRF reads signed INTEGER indices independently for each domain.
            // Out-of-domain rows are ignored by its tile bounds, not remapped
            // through the parent. Keep them until the domain shape is known.
            let index = |text: &str| -> Result<i64, Box<dyn Error>> {
                Ok(i64::from(text.parse::<i32>().map_err(|_| {
                    format!("turbine row {}: grid indices must be signed 32-bit integers", i+1)
                })?) - 1)
            };
            json!({"grid_x":index(&row[0])?,"grid_y":index(&row[1])?})
        } else {
            let lat = number(&row[0])?;
            let lon = number(&row[1])?;
            if !(-90.0..=90.0).contains(&lat) || !(-180.0..=180.0).contains(&lon) {
                return Err(format!("invalid geographic turbine row {}", i+1).into());
            }
            json!({"latitude":lat,"longitude":lon})
        };
        let kind: u32 = row[if mav == 0 {2} else {3}].parse()?;
        if !(1..=100).contains(&kind) { return Err("WRF turbine type must be in 1..100 (native table bound)".into()); }
        let key = kind.to_string();
        if !types.contains_key(&key) {
            let name = format!("wind-turbine-{kind}.tbl");
            let table_path = path.parent().unwrap_or(Path::new(".")).join(&name);
            let table = lines(&table_path).map_err(|error| format!("{name}: {error}"))?;
            if table.len() < 4 || table[0].len() != 1 || table[1].len() != 4 {
                return Err(format!("{name}: expected row count, hub diameter standing_CT rated_MW, then curve rows").into());
            }
            let n: usize = table[0][0].parse()?;
            if n < 2 || n > 100 || table.len() != n + 2 { return Err(format!("{name}: curve row count mismatch").into()); }
            let header: Vec<f64> = table[1].iter().map(|s| number(s)).collect::<Result<_,_>>()?;
            let (hub, diameter, standing, rated) = (header[0], header[1], header[2], header[3]);
            if diameter <= 0.0 || hub <= diameter/2.0 || standing < 0.0 || rated <= 0.0 {
                return Err(format!("{name}: invalid rotor geometry, standing CT or capacity").into());
            }
            // WRF doc/README.windturbine: nominal capacity is MW in the
            // header; the subsequent power-curve samples are already kW.
            let rated_kw = rated * 1000.0;
            if !rated_kw.is_finite() { return Err(format!("{name}: capacity overflows kW").into()); }
            let area = std::f64::consts::PI * diameter * diameter / 4.0;
            let mut curve = Vec::new();
            let mut previous = -1.0;
            let mut mav_curve = Vec::new();
            let mut native_curve = Vec::new();
            for fields in &table[2..] {
                if fields.len() != 3 { return Err(format!("{name}: curve needs speed CT power_kW").into()); }
                let v: Vec<f64> = fields.iter().map(|s| number(s)).collect::<Result<_,_>>()?;
                if (mav != 0 && v[0] <= 0.0) || v[0] <= previous || v[1] < 0.0 || v[2] < 0.0 || (v[0] == 0.0 && v[2] != 0.0) {
                    return Err(format!("{name}: increasing nonnegative speed, CT and power required").into());
                }
                previous = v[0];
                native_curve.push(vec![v[0] as f32,v[1] as f32,v[2] as f32]);
                let cp = if v[0] == 0.0 { 0.0 } else { v[2]*1000.0/(0.5*1.23*area*v[0].powi(3)) };
                curve.push(vec![v[0], v[1], cp]);
                // Native REAL(4) table initialization, including every rounding point.
                let speed=v[0] as f32;
                let rotor=diameter as f32;
                let ar=(std::f32::consts::PI/4.0)*(rotor*rotor);
                let coeff=(v[2] as f32)*1000.0/(((0.5f32*1.23)*((speed*speed)*speed))*ar);
                mav_curve.push(vec![speed,v[1] as f32,coeff]);
            }
            let mut spec = json!({"curve":curve,"hub_height_m":hub,"rotor_diameter_m":diameter,
                "rated_capacity_kw":rated_kw,"curve_source":"WRF turbine table",
                "cut_in_m_s":curve[0][0],"cut_out_m_s":curve[n-1][0],
                "standing_ct":standing,"cutout_ct":curve[n-1][1],"reference_density_kg_m3":1.23,
                "wrf_curve":true,"native_curve":native_curve});
            if mav != 0 { spec["mav_curve"] = json!(mav_curve); }
            inputs.insert(name, format!("{:x}",Sha256::digest(fs::read(table_path)?)));
            types.insert(key.clone(), spec);
        }
        let spec = &types[&key];
        let mut turbine = json!({"id":format!("row-{:06}",i+1),"type":key,
            "farm_id":row.get(if mav == 0 {3} else {2}).map(String::as_str).unwrap_or("farm"),
            "hub_height_m":spec["hub_height_m"],"rotor_diameter_m":spec["rotor_diameter_m"],
            "rated_capacity_kw":spec["rated_capacity_kw"],"operating":true});
        turbine.as_object_mut().unwrap().extend(coordinates.as_object().unwrap().clone());
        turbines.push(turbine);
    }
    let mut result = json!({"schema":"gpuwm-wind-farm.v1","types":types,"turbines":turbines,
        "options":{"tke_factor":0.25},"provenance":{"format":"WRF Fitch","input_sha256":inputs}});
    if mav != 0 {
        result["provenance"]["format"] = json!("WRF MAV");
        result["provenance"]["coordinate_mode"] = json!(mav);
        if mav == 2 {
            // WRF separates consecutive farm groups by 600 km. Each group is
            // prepared independently; its sums retain the native file order.
            let rows = result["turbines"].as_array_mut().unwrap();
            let mut ranges = Vec::new(); let mut start=0;
            while start<rows.len() { let mut end=start+1;
                while end<rows.len() && rows[end]["farm_id"]==rows[start]["farm_id"] { end+=1; }
                ranges.push((start,end)); start=end;
            }
            let groups: Vec<_> = ranges.par_iter().enumerate().map(|(index,&(start,end))| {
                let mut lons: Vec<f64> = rows[start..end].iter().map(|r|r["longitude"].as_f64().unwrap()).collect();
                lons.sort_by(|a,b|a.total_cmp(b)); let center=lons[(lons.len()+1)/2-1];
                let xy:Vec<_>=rows[start..end].iter().map(|r|latlon_xy(r["latitude"].as_f64().unwrap(),r["longitude"].as_f64().unwrap(),center)).collect();
                let (mut sx,mut sy)=(0.0,0.0); for &(x,y) in &xy { sx+=x;sy+=y; }
                sx/=xy.len() as f64; sy/=xy.len() as f64;
                (start,xy.into_iter().map(|(x,y)|((x-sx) as f32,(y-sy+(index+1) as f64*600000.0) as f32)).collect::<Vec<_>>())
            }).collect();
            for (start,xy) in groups { for (i,(x,y)) in xy.into_iter().enumerate() { rows[start+i]["mav_x"]=json!(x);rows[start+i]["mav_y"]=json!(y); } }
        }
    }
    if indexed {
        result["provenance"]["coordinates"] = json!("WRF one-based domain-local mass-grid indices");
    }
    Ok(result)
}

fn latlon_xy(latitude:f64,longitude:f64,central:f64)->(f64,f64) {
    let pi=std::f32::consts::PI as f64; let k0=0.9996f32 as f64; let e=0.00669438f32 as f64; let r=6378137.0;
    let lat=latitude*pi/180.0; let ls=lat.sin();let lc=lat.cos();let lt=ls/lc;let lt2=lt*lt;let lt4=lt2*lt2;
    let lon=longitude*pi/180.0;let center=central*pi/180.0;let dlon=(lon-center+pi)%(2.0*pi)-pi;
    let e2=e*e;let e3=e2*e;let ep=e/(1.0-e);
    let m1=1.0-e/4.0-3.0*e2/64.0-5.0*e3/256.0;
    let m2=3.0*e/8.0+3.0*e2/32.0+45.0*e3/1024.0;
    let m3=15.0*e2/256.0+45.0*e3/1024.0;let m4=35.0*e3/3072.0;
    let n=r/(1.0-e*(ls*ls)).sqrt();let c=ep*(lc*lc);
    let a=lc*dlon;let a2=a*a;let a3=a2*a;let a4=a3*a;let a5=a4*a;let a6=a5*a;
    let m=r*(m1*lat-m2*(2.0*lat).sin()+m3*(4.0*lat).sin()-m4*(6.0*lat).sin());
    let east=k0*n*(a+a3/6.0*(1.0-lt2+c)+a5/120.0*(5.0-18.0*lt2+lt4+72.0*c-58.0*ep))+500000.0;
    let north=k0*(m+n*lt*(a2/2.0+a4/24.0*(5.0-lt2+9.0*c+4.0*(c*c))+a6/720.0*(61.0-58.0*lt2+lt4+600.0*c-330.0*ep)));
    (east,north)
}
