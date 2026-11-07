//! Reduce coupled history on Rayon workers. No forecast physics is changed.
//! Cd is diagnosed from the surface layer's actual stress velocity and U10.
use rayon::prelude::*;
use serde::Deserialize;
use serde_json::{json,Value};
use std::{fs,path::{Path,PathBuf},time::Instant};
use wrf_core::{WrfFile,getvar,ComputeOpts};
type R<T>=Result<T,Box<dyn std::error::Error+Send+Sync>>;
#[derive(Deserialize)]
struct Frame {path:PathBuf,valid:String,#[serde(default)] reduce:bool}
#[derive(Deserialize)]
struct Request {output:PathBuf,frames:Vec<Frame>,threads:usize,initial_center:[f64;2],search_radius_km:f64,wind_radius_km:f64,edge_cells:usize}
struct Reduced {row:Value,lat:Vec<f64>,lon:Vec<f64>,slp:Vec<f64>,wind:Vec<f64>,nx:usize,ny:usize}
fn distance(a:f64,b:f64,c:f64,d:f64)->f64 {
 let h=((c-a).to_radians()/2.).sin().powi(2)+a.to_radians().cos()*c.to_radians().cos()*((d-b).to_radians()/2.).sin().powi(2);
 12742.*h.sqrt().min(1.).asin()
}
fn write_plane(path:&Path,data:&[f64])->R<()> {
 let bytes:Vec<[u8;4]>=data.par_iter().map(|v|(*v as f32).to_le_bytes()).collect();
 let flat:Vec<u8>=bytes.into_iter().flatten().collect();fs::write(path,flat)?;Ok(())
}
fn read_frame(f:&Frame,r:&Request,index:usize)->R<Reduced> {
 let started=Instant::now();let file=WrfFile::open(&f.path)?;let n=file.nxy();
 let lat=file.read_var("XLAT",0)?;let lon=file.read_var("XLONG",0)?;
 let u=file.u10(0)?;let v=file.v10(0)?;let ust=file.read_var("UST",0)?;let land=file.read_var("XLAND",0)?;
 let slp=getvar(&file,"slp",Some(0),&ComputeOpts{units:Some("hPa".into()),..Default::default()})?.data;
 let wind:Vec<f64>=u.par_iter().zip(v.par_iter()).map(|(a,b)|a.hypot(*b)).collect();
 for (name,data) in [("lat",&lat),("lon",&lon),("slp",&slp),("wind",&wind),("ust",&ust),("land",&land)] {
  if data.len()!=n||data.par_iter().any(|v|!v.is_finite()){return Err(format!("{} invalid {name}",f.path.display()).into());}
 }
 let bins:Vec<Value>=(0..14).into_par_iter().map(|bin|{
  let lo=bin as f64*5.;let hi=lo+5.;
  let mut cd:Vec<f64>=(0..n).filter(|&i|land[i]>=1.5&&wind[i]>=lo&&wind[i]<hi&&wind[i]>=1.).map(|i|(ust[i]/wind[i]).powi(2)).collect();
  cd.sort_by(f64::total_cmp);let count=cd.len();
  if count==0 {json!({"wind_low":lo,"wind_high":hi,"count":0})} else {json!({"wind_low":lo,"wind_high":hi,"count":count,"mean_cd":cd.iter().sum::<f64>()/count as f64,"min_cd":cd[0],"max_cd":cd[count-1],"p10_cd":cd[(count-1)/10],"median_cd":cd[(count-1)/2],"p90_cd":cd[9*(count-1)/10]})}
 }).collect();
 let mut row=json!({"path":f.path,"valid":f.valid,"drag_bins":bins,"read_reduce_seconds":started.elapsed().as_secs_f64()});
 if f.reduce {
  let out=r.output.join(format!("frame-{index:03}"));fs::create_dir_all(&out)?;
  for(name,data) in [("latitude",&lat),("longitude",&lon),("mslp",&slp),("wind10",&wind)] {write_plane(&out.join(format!("{name}.f32")),data)?;}
  row["reduced"]=json!({"directory":out,"shape":[file.ny,file.nx],"pressure_units":"hPa","wind_units":"m/s"});
 }
 Ok(Reduced{row,lat,lon,slp,wind,nx:file.nx,ny:file.ny})
}
fn main()->R<()> {
 let started=Instant::now();let path=std::env::args().nth(1).ok_or("request JSON required")?;
 let r:Request=serde_json::from_slice(&fs::read(path)?)?;
 if r.threads==0||r.threads>24||r.frames.is_empty()||r.search_radius_km<=0.||r.wind_radius_km<=0. {return Err("invalid request".into());}
 for pair in r.frames.windows(2){if pair[0].valid>=pair[1].valid{return Err("frames must be ordered unique UTC times".into());}}
 fs::create_dir_all(&r.output)?;rayon::ThreadPoolBuilder::new().num_threads(r.threads).build_global()?;
 let frames:Vec<R<Reduced>>=r.frames.par_iter().enumerate().map(|(i,f)|read_frame(f,&r,i)).collect();
 let mut center=r.initial_center;let mut rows=Vec::new();
 for frame in frames {
  let mut f=frame?;let edge=r.edge_cells;
  let i=(0..f.slp.len()).filter(|&i|i%f.nx>=edge&&i%f.nx+edge<f.nx&&i/f.nx>=edge&&i/f.nx+edge<f.ny&&distance(center[0],center[1],f.lat[i],f.lon[i])<=r.search_radius_km).min_by(|&a,&b|f.slp[a].total_cmp(&f.slp[b])).ok_or("no center inside search radius")?;
  center=[f.lat[i],f.lon[i]];
  let vmax=(0..f.wind.len()).filter(|&j|distance(center[0],center[1],f.lat[j],f.lon[j])<=r.wind_radius_km).map(|j|f.wind[j]).fold(f64::NEG_INFINITY,f64::max);
  f.row["fix"]=json!({"lat":center[0],"lon":center[1],"mslp_pa":f.slp[i]*100.,"max_wind10_ms":vmax});rows.push(f.row);
 }
 let receipt=json!({"schema":"gpuwm.waves.coupled-evidence/v1","status":"complete","frames":rows,"threads":r.threads,"wall_seconds":started.elapsed().as_secs_f64(),"track_method":"grid MSLP minimum within declared radius of preceding center; initialization supplied in request; no observed locations after initialization","search_radius_km":r.search_radius_km,"wind_radius_km":r.wind_radius_km,"edge_cells":r.edge_cells,"drag_definition":"(UST/hypot(U10,V10))^2 over XLAND >= 1.5; effective actual surface drag, includes stability; initial frame is cold-start and excluded from interpretation"});
 fs::write(r.output.join("receipt.json"),serde_json::to_vec_pretty(&receipt)?)?;println!("{}",json!({"status":"complete","frames":r.frames.len(),"wall_seconds":started.elapsed().as_secs_f64()}));Ok(())
}
