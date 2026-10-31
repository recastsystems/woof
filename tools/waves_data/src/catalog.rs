use crate::{Result,text,arrays,observations,spectra};
use chrono::{DateTime,Datelike};
use rayon::prelude::*;
use serde_json::{json,Value};
use std::{collections::BTreeMap,fs,path::Path};
pub fn run(r:&Value,out:&Path)->Result<Value>{
    let case=Path::new(text(r,"case")?);let acquired:Value=serde_json::from_slice(&fs::read(case.join("ACQUIRED.json"))?)?;let recipe=&acquired["request"];
    let start=DateTime::parse_from_rfc3339(text(recipe,"start")?)?;let end=DateTime::parse_from_rfc3339(text(recipe,"end")?)?;let(begin,finish)=(start.timestamp(),end.timestamp());
    let files=acquired["files"].as_array().ok_or("acquisition files missing")?;let mut bytime:BTreeMap<String,Value>=BTreeMap::new();
    for f in files.iter().filter(|f|f["error"].is_null()){let p=text(f,"path")?;if p.starts_with("gfs-wave/")||p.starts_with("gfs-wind/"){let valid=text(f,"valid")?;let row=bytime.entry(valid.into()).or_insert_with(||json!({"valid":valid}));row[if p.starts_with("gfs-wave/"){"wave"}else{"wind"}]=case.join(p).to_string_lossy().as_ref().into();}}
    if bytime.len()!=(finish-begin) as usize/3600+1||bytime.values().any(|v|v["wave"].is_null()||v["wind"].is_null()){return Err("hourly wave/wind acquisition coverage is incomplete".into());}
    let metadata=observations::contents(Path::new(text(r,"station_metadata")?))?;
    let stations=recipe["stations"].as_array().ok_or("stations missing")?;
    let rows:Vec<Result<Value>>=stations.par_iter().map(|id|{
        let id=id.as_str().ok_or("invalid station id")?;let p=case.join(format!("ndbc/{id}h{}.txt.gz",start.year()));
        if !p.exists(){return Ok(json!({"id":id,"status":"no_stdmet_file"}));}
        let obs=observations::stdmet(&p,begin,finish)?;let hs=obs.iter().filter(|o|o.values.contains_key("WVHT")).count();
        if hs==0{return Ok(json!({"id":id,"status":"no_wave_observations_in_window","records":obs.len()}));}
        let specfile=files.iter().find_map(|f|{let path=f["path"].as_str()?;if path.starts_with("station-spectra/")&&path.ends_with(&format!("-{id}.spec"))&&f["error"].is_null(){Some(case.join(path))}else{None}});
        let coordinate=if let Some(p)=specfile {let s=spectra::station(&p,begin,finish)?;s.first().map(|s|(s.lat,s.lon,format!("GFS-Wave station output {}; model output location",p.display())))}else{None};
        let coordinate=coordinate.or_else(||metadata.lines().find(|l|l.split('|').next()==Some(id)).and_then(|l|{let c:Vec<_>=l.split('|').nth(6)?.split_whitespace().collect();let lat=c.first()?.parse::<f64>().ok()?*if *c.get(1)?=="S"{-1.}else{1.};let lon=c.get(2)?.parse::<f64>().ok()?*if *c.get(3)?=="W"{-1.}else{1.};Some((lat,lon,"NDBC station registry at acquisition date; historical position not independently checked".to_string()))}));
        let Some((lat,lon,origin))=coordinate else{return Ok(json!({"id":id,"status":"missing_coordinates","wave_records":hs}));};
        let mut spectral_files=serde_json::Map::new();for(kind,code)in[("swden","w"),("swdir","d"),("swdir2","i"),("swr1","j"),("swr2","k")]{let p=case.join(format!("ndbc/{id}{code}{}.txt.gz",start.year()));if p.exists(){spectral_files.insert(kind.into(),p.to_string_lossy().as_ref().into());}}
        let peak=obs.iter().filter_map(|o|o.values.get("WVHT").map(|v|(o.epoch,*v))).max_by(|a,b|a.1.total_cmp(&b.1));let long=obs.iter().filter(|o|o.values.get("DPD").is_some_and(|v|*v>=16.)).count();
        Ok(json!({"id":id,"status":"available","lat":lat,"lon":lon,"coordinate_source":origin,"stdmet":p,"spectral_files":spectral_files,"wave_records":hs,"records":obs.len(),"hs_peak":peak,"records_tp_ge_16s":long}))
    }).collect();
    let rows:Vec<Value>=rows.into_iter().collect::<Result<_>>()?;
    if rows.iter().any(|v|v["status"]=="missing_coordinates"){return Err("observed buoy lacks a coordinate; see registry/source data".into());}
    let active:Vec<Value>=rows.iter().filter(|v|v["status"]=="available").cloned().collect();if active.is_empty(){return Err("no observed buoys in case window".into());}
    let bathy=Path::new(text(r,"bathymetry")?);
    let req=json!({"operation":"prepare","output":r["prepared_output"],"threads":r["threads"],"field_table":r["field_table"],"grid":{"kind":"latlon","bounds":recipe["bounds"],"resolution_degrees":recipe["resolution_degrees"]},"bathymetry":{"path":bathy,"sha256":arrays::sha(bathy)?,"licence":"CC0-1.0"},"frames":bytime.values().collect::<Vec<_>>(),"stations":active,"coast_fill_cells":2,"station_max_distance_km":75.});
    fs::write(out.join("prepare-request.json"),serde_json::to_vec_pretty(&req)?)?;
    let b=recipe["bounds"].as_array().ok_or("bounds missing")?;
    let era5=json!({"store":"https://storage.googleapis.com/gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3","times":bytime.keys().collect::<Vec<_>>(),"area":[b[1],b[0],b[3],b[2]],"expected_levels_hpa":[1,2,3,5,7,10,20,30,50,70,100,125,150,175,200,225,250,300,350,400,450,500,550,600,650,700,750,775,800,825,850,875,900,925,950,975,1000],"fields":[{"source":"10m_u_component_of_wind","output":"U10","units":"m s-1","pressure":false},{"source":"10m_v_component_of_wind","output":"V10","units":"m s-1","pressure":false}]});
    fs::write(out.join("era5-request.json"),serde_json::to_vec_pretty(&era5)?)?;
    let coverage=json!({"status":"complete","start":start.to_rfc3339(),"end":end.to_rfc3339(),"hourly_frames":bytime.len(),"stations":rows,"active_stations":active.len(),"era5_status":"request written, extraction pending"});fs::write(out.join("coverage.json"),serde_json::to_vec_pretty(&coverage)?)?;Ok(coverage)
}
