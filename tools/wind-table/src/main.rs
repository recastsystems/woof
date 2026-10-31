//! Original generic curves, not manufacturer measurements. Drag equations use
//! Fitch et al. 2012, doi:10.1175/MWR-D-11-00352.1 and corrected TKE from
//! Archer et al. 2020, doi:10.1175/MWR-D-20-0097.1, in wind_farm.cu.
//! Source mappings and class assumptions are external table rows.
use rayon::prelude::*;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, env, fs, error::Error, time::Instant};
mod wrf;
const SOURCE_REVISION: &str = concat!("GPUWM_BRIDGE_SOURCE_REV=", env!("GPUWM_BRIDGE_SOURCE_REV"));
fn main() -> Result<(), Box<dyn Error>> {
 let args:Vec<String>=env::args().collect();
 if args.len()==2 && matches!(args[1].as_str(), "--version" | "--abi") {
  println!("rw-wind-table input-v1 {SOURCE_REVISION}");return Ok(());
 }
 if args.len()==3 && args[1]=="--wrf" {
  println!("{}",wrf::read(std::path::Path::new(&args[2]))?);return Ok(());
 }
 if args.len()!=12 {return Err("usage: rw-wind-table DATABASES.json SOURCE CLASSES.json INPUT.csv OUTPUT.json WEST SOUTH EAST NORTH YEAR WORKERS (year 0=all years)".into());}
 let begin=Instant::now();
 let sources:Value=serde_json::from_slice(&fs::read(&args[1])?)?;
 let source=&sources[&args[2]];if source.is_null(){return Err("unknown database row".into());}
 let classes:Vec<Value>=serde_json::from_slice(&fs::read(&args[3])?)?;
 let bounds:Vec<f64>=args[6..10].iter().map(|v|v.parse()).collect::<Result<_,_>>()?;
 let year:i64=args[10].parse()?;
 // Last argument is worker count, explicit to respect shared-host quotas.
 let workers:usize=args[11].parse()?;
 rayon::ThreadPoolBuilder::new().num_threads(workers).build_global()?;
 let bytes=fs::read(&args[4])?;let hash=format!("{:x}",Sha256::digest(&bytes));
 let mut reader=csv::Reader::from_reader(bytes.as_slice());let header=reader.headers()?.clone();
 let mut ix=BTreeMap::new();
 for (key,column) in source["columns"].as_object().ok_or("columns required")? {
  ix.insert(key.clone(),header.iter().position(|h|h==column.as_str().unwrap_or("")).ok_or(format!("missing column {column}"))?);
 }
 let records:Vec<_>=reader.records().collect::<Result<_,_>>()?;
 let n_input=records.len();
 let rows:Vec<_>=records.par_iter().enumerate().map(|(index,row)| {
  let text=|k:&str|row.get(ix[k]).unwrap_or("");
  let number=|k:&str|text(k).parse::<f64>().ok().filter(|v|v.is_finite());
  let lat=number("latitude");let lon=number("longitude");
  let (Some(lat),Some(lon))=(lat,lon) else {return (index,None,"invalid_location")};
  if !(-90.0..=90.0).contains(&lat)||!(-180.0..=180.0).contains(&lon){return(index,None,"invalid_location");}
  let inside_lon=if bounds[0]<=bounds[2]{lon>=bounds[0]&&lon<=bounds[2]}else{lon>=bounds[0]||lon<=bounds[2]};
  if !inside_lon||lat<bounds[1]||lat>bounds[3]{return(index,None,"outside_bounds");}
  let commissioned=number("year").map(|v|v as i64).filter(|v|*v>0);
  if year>0 && commissioned.map_or(true,|y|y>year){return(index,None,"date_or_unknown_year");}
  let Some(cap)=number("rated_capacity_kw").map(|v|v*source["capacity_scale_to_kw"].as_f64().unwrap_or(1.0)).filter(|v|*v>0.0) else{return(index,None,"missing_capacity");};
  let Some(class)=classes.iter().find(|c|cap<=c["max_capacity_kw"].as_f64().unwrap()) else{return(index,None,"no_class");};
  let hh=number("hub_height_m").filter(|v|*v>0.0).unwrap_or(class["hub_height_m"].as_f64().unwrap());
  let rd=number("rotor_diameter_m").filter(|v|*v>0.0).unwrap_or(class["rotor_diameter_m"].as_f64().unwrap());
  if hh<=rd/2.0{return(index,None,"rotor_intersects_ground");}
  let geometry_imputed=number("hub_height_m").map_or(true,|v|v<=0.0)||number("rotor_diameter_m").map_or(true,|v|v<=0.0);
  // Each capacity/diameter class has its own rated speed. Power never exceeds
  // the row's rated capacity at reference density 1.225 kg/m3.
  let area=std::f64::consts::PI*rd*rd/4.0;
  let ci=class["cut_in"].as_f64().unwrap();let co=class["cut_out"].as_f64().unwrap();
  let cpmax=class["cp_max"].as_f64().unwrap();let ct0=class["ct_below_rated"].as_f64().unwrap();
  let rated=(cap*1000.0/(0.5*1.225*area*cpmax)+ci.powi(3)).cbrt();
  let mut curve=Vec::new();
  for n in 0..=((co*4.0) as usize) {
   let u=n as f64/4.0;
   let cp=if u<=ci||u>=co{0.0}else{(cpmax*(1.0-(ci/u).powi(3))).min(cap*1000.0/(0.5*1.225*area*u.powi(3)))};
   let ct=if u<=ci||u>=co{0.0}else{(ct0*(rated/u).powi(2).min(1.0)).max(cp)};
   curve.push(json!([u,ct,cp]));
  }
  let kind=format!("{}-{}-{}",class["name"].as_str().unwrap(),cap,rd);
  let spec=json!({"curve":curve,"curve_source":"generic analytic class; not manufacturer data","reference_density_kg_m3":1.225,"rated_capacity_kw":cap,"class":class["name"]});
  let farm=ix.get("farm").and_then(|i|row.get(*i)).filter(|s|!s.is_empty()).unwrap_or("unassigned");
  let turbine=json!({"id":format!("{}:{}",args[2],text("id")),"farm_id":farm,"latitude":lat,"longitude":lon,"hub_height_m":hh,"rotor_diameter_m":rd,"rated_capacity_kw":cap,"model":text("model"),"commissioning_year":commissioned,"geometry_imputed":geometry_imputed,"type":kind,"operating":true});
  (index,Some((kind,spec,turbine)),"included")
 }).collect();
 let mut types=BTreeMap::new();let mut turbines=Vec::new();let mut counts=BTreeMap::new();
 for (_,value,reason) in rows { *counts.entry(reason).or_insert(0usize)+=1; if let Some((kind,spec,row))=value{types.insert(kind,spec);turbines.push(row);} }
 let output=json!({"schema":"gpuwm-wind-farm.v1","types":types,"turbines":turbines,"options":{"tke_factor":0.25},"provenance":{"database":source,"input_sha256":hash,"input_rows":n_input,"counts":counts,"workers":workers,"builder_wall_s":begin.elapsed().as_secs_f64(),"bounds_wsen":bounds,"latest_commissioning_year":year,"historical_caveat":"current inventories can omit retired turbines; commissioning year has no within-year date","curve_assumption":"generic class, no calibration to case observations"}});
 fs::write(&args[5],serde_json::to_vec(&output)?)?;
 println!("{}",output["provenance"]);Ok(())
}
