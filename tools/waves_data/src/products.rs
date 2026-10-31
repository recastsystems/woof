use crate::{Result,text,arrays};
use rayon::prelude::*;
use serde_json::{json,Value};
use std::{collections::BTreeMap,fs,path::Path};
pub fn share_scales(r:&Value,out:&Path)->Result<Value>{
    let roots=r["workdirs"].as_array().ok_or("workdirs required")?;
    if roots.is_empty(){return Err("no product preparations to compare".into())}
    let mut bounds:BTreeMap<String,(f64,f64)>=BTreeMap::new();let mut documents=Vec::new();
    for root in roots{let root=Path::new(root.as_str().ok_or("invalid product workdir")?);let receipt:Value=serde_json::from_slice(&fs::read(root.join("receipt.json"))?)?;
        if receipt["status"]!="complete"{return Err("incomplete product preparation".into())}
        for(key,row)in receipt["scales"].as_object().ok_or("product scales missing")?{let a=row["display_range"][0].as_f64().ok_or("lower scale missing")?;let b=row["display_range"][1].as_f64().ok_or("upper scale missing")?;bounds.entry(key.clone()).and_modify(|(lo,hi)|{*lo=lo.min(a);*hi=hi.max(b);}).or_insert((a,b));}
        documents.push((root.to_path_buf(),receipt));
    }
    for(root,mut receipt)in documents{let rowpath=root.join("product-rows.json");let mut rows:Value=serde_json::from_slice(&fs::read(&rowpath)?)?;
        for(key,row)in receipt["scales"].as_object_mut().ok_or("product scales missing")?{let(a,b)=bounds[key];row["display_range"]=json!([a,b]);row["selection"]="shared across every compared source and valid time; no clipping".into();rows["products"][key]["range"]=json!([a,b]);}
        receipt["shared_scale_workdirs"]=r["workdirs"].clone();fs::write(rowpath,serde_json::to_vec_pretty(&rows)?)?;fs::write(root.join("receipt.json"),serde_json::to_vec_pretty(&receipt)?)?;
    }
    let result=json!({"status":"complete","scales":bounds,"workdirs":roots,"selection":"union of native sequence-wide display ranges"});fs::write(out.join("shared-scales.json"),serde_json::to_vec_pretty(&result)?)?;Ok(result)
}
pub fn run(r:&Value,out:&Path)->Result<Value>{
    let mut rows:Value=serde_json::from_slice(&fs::read(text(r,"rows")?)?)?;let frames=r["frames"].as_array().ok_or("frames required")?;if frames.is_empty(){return Err("no frames to render".into());}
    let selected:Vec<&str>=r["products"].as_array().ok_or("product list required")?.iter().map(|v|v.as_str().ok_or("invalid product name")).collect::<std::result::Result<_,_>>()?;
    let first=Path::new(text(&frames[0],"path")?);let file=netcrust::open(first).map_err(|e|e.to_string())?;let latname=r["latitude"].as_str().unwrap_or("latitude");let lonname=r["longitude"].as_str().unwrap_or("longitude");
    let latshape=file.variable(latname).ok_or("latitude missing")?.shape();if latshape.len()!=2{return Err("wave map snapshots require [y,x] coordinates; point/mesh products use a registered renderer grid backend".into());}
    let(ny,nx)=(latshape[0],latshape[1]);let n=nx*ny;let lat:Vec<f32>=arrays::read_nc(first,latname)?.into_par_iter().map(|v|v as f32).collect();let lon:Vec<f32>=arrays::read_nc(first,lonname)?.into_par_iter().map(|v|v as f32).collect();if lon.len()!=n{return Err("coordinate shapes differ".into());}
    arrays::f32(out,"latitude",&lat,&[ny,nx])?;arrays::f32(out,"longitude",&lon,&[ny,nx])?;
    let wet:Vec<f32>=arrays::read_nc(first,r["wet"].as_str().unwrap_or("wet"))?.into_par_iter().map(|v|v as f32).collect();if wet.len()!=n{return Err("wet mask shape differs".into());}arrays::f32(out,"wet",&wet,&[ny,nx])?;
    let mut extrema:BTreeMap<String,(f32,f32)>=BTreeMap::new();let mut renders=Vec::new();let mut digests=Vec::new();
    for (fi,frame) in frames.iter().enumerate(){let input=Path::new(text(frame,"path")?);let f=netcrust::open(input).map_err(|e|e.to_string())?;let mut fields=BTreeMap::new();let mut arrows=BTreeMap::new();let mut masks=BTreeMap::new();
        let read=|key:&str|->Result<Vec<f32>>{let var=r["variables"][key].as_str().unwrap_or(key);let v=arrays::read_nc(input,var)?;if v.len()!=n{return Err(format!("{var} needs one [y,x] plane per snapshot").into());}if v.par_iter().enumerate().any(|(i,v)|wet[i]>0.5&&!v.is_finite()){return Err(format!("nonfinite wet wave product {var}").into());}Ok(v.into_par_iter().enumerate().map(|(i,v)|if wet[i]>0.5{v as f32}else{0.}).collect())};
        let actual_time=arrays::read_nc(input,"time")?;let expected=chrono::DateTime::parse_from_rfc3339(text(frame,"valid")?)?.timestamp();if actual_time.len()!=1||(actual_time[0]-expected as f64).abs()>0.5{return Err("render label differs from snapshot valid time".into());}
        for &key in &selected{if rows["products"][key].is_null(){return Err(format!("unknown product {key}").into());}let values=read(key)?;
            if let Some(height)=rows["products"][key]["valid_when_positive"].as_str(){let support=read(height)?;let mask:Vec<f32>=support.par_iter().enumerate().map(|(i,h)|if wet[i]>0.5&&*h>0.{1.}else{0.}).collect();let a=arrays::f32(out,&format!("frame-{fi:04}-{key}-mask"),&mask,&[ny,nx])?;masks.insert(key,out.join(a["path"].as_str().unwrap()));}
            let lo=values.par_iter().enumerate().filter(|(i,_)|wet[*i]>0.5).map(|(_,v)|*v).reduce(||f32::INFINITY,f32::min);let hi=values.par_iter().enumerate().filter(|(i,_)|wet[*i]>0.5).map(|(_,v)|*v).reduce(||f32::NEG_INFINITY,f32::max);
            extrema.entry(key.into()).and_modify(|(l,h)|{*l=l.min(lo);*h=h.max(hi);}).or_insert((lo,hi));
            let name=format!("frame-{fi:04}-{key}");let a=arrays::f32(out,&name,&values,&[ny,nx])?;fields.insert(key,out.join(a["path"].as_str().unwrap()));
            if let Some(direction)=rows["products"][key]["direction_product"].as_str(){let var=r["variables"][direction].as_str().unwrap_or(direction);if f.variable(var).is_none(){return Err(format!("{key} needs direction field {var}").into());}let values=read(direction)?;let a=arrays::f32(out,&format!("frame-{fi:04}-{key}-arrows"),&values,&[ny,nx])?;arrows.insert(key,out.join(a["path"].as_str().unwrap()));}
        }
        renders.push(json!({"valid_time":chrono::DateTime::from_timestamp(expected,0).ok_or("bad valid time")?.format("%Y-%m-%dT%H:%M:%SZ").to_string(),"fields":fields,"arrows":arrows,"masks":masks}));digests.push(json!({"path":input,"sha256":arrays::sha(input)?}));
    }
    let mut scales=BTreeMap::new();for(key,(low,high))in &extrema{
        let chosen=if let Some(explicit)=r["scales"][key].as_array(){if explicit.len()!=2{return Err("scale needs two bounds".into());}[explicit[0].as_f64().ok_or("bad scale")?,explicit[1].as_f64().ok_or("bad scale")?]}else{let step=if key.ends_with("hs"){if *high<=2.{0.25}else{1.}}else if key=="dspr"{10.}else{2.};[0.,((*high as f64/step).ceil()*step).max(step)]};
        if chosen[0]>*low as f64||chosen[1]<*high as f64{return Err(format!("display range for {key} hides data extrema {low}..{high}").into());}
        rows["products"][key]["range"]=json!(chosen);rows["products"][key]["mask"]="wet".into();scales.insert(key,json!({"data_min":low,"data_max":high,"display_range":chosen,"selection":"fixed across selected sequence, full extrema retained"}));
    }
    let rowpath=out.join("product-rows.json");fs::write(&rowpath,serde_json::to_vec_pretty(&rows)?)?;
    let render=json!({"schema":"gpuwm.ocean.products/v1","family":"waves","out_dir":r["render_output"],"domain":r["domain"],"source_label":r["source_label"],"nx":nx,"ny":ny,"latitude":out.join("latitude.bin"),"longitude":out.join("longitude.bin"),"wet":out.join("wet.bin"),"rows":rowpath,"frames":renders,"join_existing_output":r["join_existing_output"].as_bool().unwrap_or(false)});fs::write(out.join("render-request.json"),serde_json::to_vec_pretty(&render)?)?;
    Ok(json!({"status":"complete","frames":frames.len(),"products":selected,"inputs":digests,"scales":scales,"render_request":out.join("render-request.json"),"render_status":"pending rw_wrfbatch through gpuwm.rustwx"}))
}
