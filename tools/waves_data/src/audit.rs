//! Read the delivered bytes, not preparation status labels.
use crate::{arrays,text,Result};
use rayon::prelude::*;
use serde_json::{json,Value};
use std::{collections::{BTreeMap,BTreeSet},fs,path::Path};
pub fn run(r:&Value,out:&Path)->Result<Value>{
    let root=Path::new(text(r,"prepared")?).canonicalize()?;let path=root.join("manifest.json");let m:Value=serde_json::from_slice(&fs::read(&path)?)?;
    if m["schema"]!="gpuwm.waves.prepared/v1"||m["status"]!="complete"{return Err("incomplete prepared schema".into())}
    let n=m["grid"]["ncell"].as_u64().ok_or("ncell missing")? as usize;let times=m["times"].as_array().ok_or("times missing")?;
    if times.len()<2||times.windows(2).any(|w|w[0].as_i64()>=w[1].as_i64())||m["times"]!=m["forcing_times"]||m["times"]!=m["boundary_times"]{return Err("prepared times do not strictly increase and agree".into())}
    let rows=m["arrays"].as_object().ok_or("arrays missing")?;
    let checked:Vec<Result<(String,Value,Vec<u8>)>>=rows.iter().collect::<Vec<_>>().par_iter().map(|(name,row)|{
        let path=root.join(text(row,"path")?).canonicalize()?;if !path.starts_with(&root){return Err("prepared array escaped folder".into())}
        let digest=arrays::sha(&path)?;if digest!=text(row,"sha256")?{return Err(format!("hash mismatch {name}").into())}
        let bytes=fs::read(path)?;let shape=row["shape"].as_array().ok_or("shape missing")?.iter().map(|v|v.as_u64().ok_or("invalid dimension")).collect::<std::result::Result<Vec<_>,_>>()?;
        if shape.iter().product::<u64>()*4!=bytes.len() as u64{return Err(format!("byte/shape mismatch {name}").into())}
        let values:Vec<f64>=match row["dtype"].as_str(){Some("<f4")=>bytes.par_chunks_exact(4).map(|b|f32::from_le_bytes(b.try_into().unwrap()) as f64).collect(),Some("<i4")=>bytes.par_chunks_exact(4).map(|b|i32::from_le_bytes(b.try_into().unwrap()) as f64).collect(),_=>return Err("unsupported array dtype".into())};
        if values.par_iter().any(|v|!v.is_finite()){return Err(format!("nonfinite array {name}").into())}
        let min=values.par_iter().copied().reduce(||f64::INFINITY,f64::min);let max=values.par_iter().copied().reduce(||f64::NEG_INFINITY,f64::max);
        Ok((name.to_string(),json!({"sha256":digest,"bytes":bytes.len(),"shape":shape,"minimum":if values.is_empty(){None}else{Some(min)},"maximum":if values.is_empty(){None}else{Some(max)}}),bytes))
    }).collect();
    let mut summaries=BTreeMap::new();let mut data=BTreeMap::new();for item in checked{let(k,s,b)=item?;summaries.insert(k.clone(),s);data.insert(k,b);}
    let floats=|key:&str|->Result<Vec<f32>>{Ok(data.get(key).ok_or_else(||format!("array missing: {key}"))?.par_chunks_exact(4).map(|b|f32::from_le_bytes(b.try_into().unwrap())).collect())};
    let integers=|key:&str|->Result<Vec<i32>>{Ok(data.get(key).ok_or_else(||format!("array missing: {key}"))?.par_chunks_exact(4).map(|b|i32::from_le_bytes(b.try_into().unwrap())).collect())};
    let wet=integers("wet")?;let depth=floats("depth")?;let area=floats("area")?;let boundary=integers("boundary_cells")?;let initial=floats("initial_partitions")?;let series=floats("boundary_partitions")?;
    if wet.len()!=n||depth.len()!=n||area.len()!=n||initial.len()!=n*16||series.len()!=times.len()*boundary.len()*16{return Err("prepared geometry/partition shape mismatch".into())}
    if wet.iter().enumerate().any(|(i,&w)|!matches!(w,0|1)||area[i]<=0.||depth[i]<0.||(w==0&&depth[i]!=0.)||(w==1&&depth[i]<=0.)){return Err("wet/depth/area contract mismatch".into())}
    if boundary.iter().any(|&b|b<0||b as usize>=n||wet[b as usize]!=1)||boundary.iter().collect::<BTreeSet<_>>().len()!=boundary.len(){return Err("boundary indices invalid".into())}
    for (name,values) in [("initial_partitions",&initial),("boundary_partitions",&series)]{if let Some((i,p))=values.par_chunks_exact(4).enumerate().find_first(|(_,p)|p[0]<0.||!matches!(p[3],0.|1.)||(p[0]>0.&&(p[1]<=0.||p[2]<0.||p[2]>=360.))){return Err(format!("invalid partition height/period/bearing/kind: {name}[{i}]={p:?}").into())}}
    let peak=initial.par_chunks_exact(16).map(|p|4.*(p.chunks_exact(4).map(|v|(v[0]/4.).powi(2)).sum::<f32>()).sqrt()).reduce(||0.,f32::max);
    if peak<=0.{return Err("initial wave partitions are calm".into())}
    for key in ["u10_east","v10_north"]{if rows[key]["shape"]!=json!([times.len(),n]){return Err("forcing shape mismatch".into())}}
    for station in m["stations"].as_array().ok_or("stations missing")?{let i=station["cell"].as_u64().ok_or("station cell missing")? as usize;if i>=n||wet[i]!=1{return Err("station does not sample wet cell".into())}}
    let mut comparison=Value::Null;
    if let Some(other)=r["compare"].as_str(){let other=Path::new(other);let om:Value=serde_json::from_slice(&fs::read(other.join("manifest.json"))?)?;let mut differences=BTreeMap::new();
        for name in ["lat","lon","area","depth","initial_partitions","u10_east","v10_north"]{let row=&om["arrays"][name];let p=other.join(text(row,"path")?);if arrays::sha(&p)?!=text(row,"sha256")?{return Err("comparison input hash differs".into())}let b=fs::read(p)?;let a=floats(name)?;if b.len()!=a.len()*4{return Err("comparison shape differs".into())}let error=a.par_iter().zip(b.par_chunks_exact(4)).map(|(a,b)|(*a-f32::from_le_bytes(b.try_into().unwrap())).abs()).reduce(||0.,f32::max);differences.insert(name,error);}
        comparison=json!({"manifest":other.join("manifest.json"),"max_absolute_difference":differences});
    }
    let result=json!({"status":"complete","prepared_manifest":path,"manifest_sha256":arrays::sha(&path)?,"arrays":summaries,"cells":n,"wet_cells":wet.iter().filter(|&&v|v==1).count(),"times":times.len(),"boundary_cells":boundary.len(),"initial_reconstructed_hs_max_m":peak,"comparison":comparison});fs::write(out.join("audit.json"),serde_json::to_vec_pretty(&result)?)?;Ok(result)
}
