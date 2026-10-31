//! Original Apache-2.0 ocean product rows. Numeric input and rendering stay native.
use crate::panel::{layout_path, render_panel_cells, render_panel_cells_with_missing, PanelRequest};
use crate::annotate::{MapOverlays,LineSpec};
use rayon::prelude::*;
use rustwx_render::{Color, ColorScale, DiscreteColorScale, ExtendMode, LegendControls, LegendMode};
use serde::Deserialize;
use serde_json::{json,Value};
use std::{collections::BTreeMap,fs,path::{Path,PathBuf},time::Instant};
type R<T> = Result<T,String>;
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {schema:String,out_dir:PathBuf,domain:String,source_label:String,nx:usize,ny:usize,latitude:PathBuf,longitude:PathBuf,#[serde(default)] wet:Option<PathBuf>,rows:PathBuf,frames:Vec<Frame>,#[serde(default)] family:Option<String>,#[serde(default)] join_existing_output:bool}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Frame {valid_time:String,fields:BTreeMap<String,PathBuf>,#[serde(default)] arrows:BTreeMap<String,PathBuf>,#[serde(default)] masks:BTreeMap<String,PathBuf>}
fn arrows(lat:&[f32],lon:&[f32],direction:&[f32],wet:Option<&[f32]>,values:&[f32],nx:usize,ny:usize)->MapOverlays{
 let mut out=MapOverlays::default();let stride_x=(nx/25).max(1);let stride_y=(ny/18).max(1);
 for y in (stride_y/2..ny).step_by(stride_y){for x in (stride_x/2..nx).step_by(stride_x){let i=y*nx+x;
  if !values[i].is_finite()||values[i]<=0.||wet.is_some_and(|w|w[i]<=0.5)||lat[i].abs()>85.{continue}
  let a=(direction[i] as f64).to_radians();let latitude=lat[i] as f64;let longitude=lon[i] as f64;
  let adjacent=(y+stride_y).min(ny-1)*nx+x;let length=((lat[adjacent]-lat[i]) as f64).abs().max(0.03)*0.65;
  let north=-a.cos()*length;let east=-a.sin()*length;let cos=latitude.to_radians().cos();
  let tail=(latitude-north/2.,longitude-east/(2.*cos));let head=(latitude+north/2.,longitude+east/(2.*cos));
  let left=(head.0-north*0.32+east*0.2,head.1+(-east*0.32-north*0.2)/cos);let right=(head.0-north*0.32-east*0.2,head.1+(-east*0.32+north*0.2)/cos);
  for color in ["#ffffff","#15202b"]{let width=if color=="#ffffff"{4}else{2};out.lines.push(LineSpec{points:vec![tail,head],color:color.into(),width,closed:false});out.lines.push(LineSpec{points:vec![left,head,right],color:color.into(),width,closed:false});}
 }}out
}
fn read(path:&Path,n:usize)->R<Vec<f32>>{
 let b=fs::read(path).map_err(|e|e.to_string())?;
 if b.len()!=n*4{return Err(format!("{} has {} bytes; expected {}",path.display(),b.len(),n*4))}
 let v:Vec<f32>=b.par_chunks_exact(4).map(|x|f32::from_le_bytes(x.try_into().unwrap())).collect();
 if v.par_iter().any(|v|!v.is_finite()){return Err(format!("nonfinite ocean plane {}",path.display()))}Ok(v)
}
fn token(s:&str)->bool{!s.is_empty()&&s.bytes().all(|b|b.is_ascii_alphanumeric()||b==b'-'||b==b'_')}
fn run(path:&Path)->R<()> {
 let start=Instant::now();let r:Request=serde_json::from_slice(&fs::read(path).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
 if r.schema!="gpuwm.ocean.products/v1"||!token(&r.domain)||r.nx<2||r.ny<2{return Err("invalid ocean product schema, domain or grid".into())}
 if r.family.as_ref().is_some_and(|v|!token(v)){return Err("invalid product family".into())}
 let rows:Value=serde_json::from_slice(&fs::read(&r.rows).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
 let lat=read(&r.latitude,r.nx*r.ny)?;let lon=read(&r.longitude,r.nx*r.ny)?;
 let wet=r.wet.as_ref().map(|p|read(p,r.nx*r.ny)).transpose()?;
 // Preserve existing products on an ordinary repeat. Claim a fresh sibling.
 let mut out=r.out_dir.clone();
 if !r.join_existing_output&&out.exists()&&fs::read_dir(&out).map_err(|e|e.to_string())?.next().is_some(){let stamp=std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_err(|e|e.to_string())?.as_nanos();out=out.with_file_name(format!("{}-{stamp}",out.file_name().unwrap_or_default().to_string_lossy()));}
 fs::create_dir_all(&out).map_err(|e|e.to_string())?;
 // Each frame writes a distinct product/time path. Keep the expensive panel
 // rasterization and PNG encoding across the same Rayon pool as field work.
 let mut times=std::collections::BTreeSet::new();
 for f in &r.frames {if !times.insert(&f.valid_time){return Err("duplicate ocean frame valid_time".into())}}
 let receipts:Vec<R<Vec<Value>>>=r.frames.par_iter().map(|f| {
  let mut receipt=Vec::new();
  if f.valid_time.len()!=20||!f.valid_time.is_ascii()||f.valid_time.as_bytes()[10]!=b'T'||!f.valid_time.bytes().all(|b|b.is_ascii_digit()||b"-:TZ".contains(&b)){return Err("valid_time must be ISO UTC YYYY-MM-DDTHH:MM:SSZ".into())}
  for (slug,path) in &f.fields {
   if !token(slug){return Err("invalid ocean product name".into())}
   let row=&rows["products"][slug];let title=row["title"].as_str().ok_or_else(||format!("unknown ocean product row {slug}"))?;
   let units=row["units"].as_str().ok_or("row units missing")?;
   let low=row["range"][0].as_f64().ok_or("row lower range missing")?;let high=row["range"][1].as_f64().ok_or("row upper range missing")?;
   if !low.is_finite()||!high.is_finite()||high<=low{return Err("invalid product range".into())}
   let factor=row["scale"].as_f64().unwrap_or(1.) as f32;
   let mut values=read(path,r.nx*r.ny)?;values.par_iter_mut().for_each(|v|*v*=factor);
   if let Some(mask)=row["mask"].as_str(){
    let wet=wet.as_ref().ok_or("ocean product mask requires prepared wet plane")?;
    let flooded=match mask{"wet"=>None,"wet-or-inundated"=>Some(read(f.fields.get("inundation").ok_or("water-level mask requires inundation plane")?,r.nx*r.ny)?),_=>return Err(format!("unknown ocean product mask {mask}"))};
    values.par_iter_mut().enumerate().for_each(|(i,v)|{if wet[i]<=0.5&&!flooded.as_ref().is_some_and(|a|a[i]>0.){*v=f32::NAN;}});
   }
   let missing=if let Some(path)=f.masks.get(slug){let mask=read(path,r.nx*r.ny)?;let missing:Vec<bool>=mask.par_iter().enumerate().map(|(i,m)|*m<=0.5&&wet.as_ref().is_some_and(|w|w[i]>0.5)).collect();values.par_iter_mut().enumerate().for_each(|(i,v)|if mask[i]<=0.5{*v=f32::NAN;});Some(missing)}else{None};
   let min=values.par_iter().copied().filter(|v|v.is_finite()).reduce(||f32::INFINITY,f32::min);let max=values.par_iter().copied().filter(|v|v.is_finite()).reduce(||f32::NEG_INFINITY,f32::max);
   let bins=20;let levels=(0..=bins).map(|i|low+(high-low)*i as f64/bins as f64).collect();
   let colors=(0..bins).map(|i|{let q=i as f64/(bins-1) as f64;Color::rgba((35.+215.*q) as u8,(90.+120.*(1.-(2.*q-1.).abs())) as u8,(235.-205.*q) as u8,255)}).collect();
   let scale=ColorScale::Discrete(DiscreteColorScale{levels,colors,extend:ExtendMode::Both,mask_below:None});
   let p=if let Some(family)=&r.family{out.join(&r.domain).join(family).join(slug).join(&f.valid_time[..10]).join(format!("{slug}_{}.png",f.valid_time.replace(':',"")))}else{layout_path(&out,&r.domain,slug,&f.valid_time[..10],&format!("{slug}_{}",f.valid_time.replace(':',"")))};
   if r.join_existing_output&&p.exists(){return Err(format!("ocean product already exists: {}; use a new run folder to preserve its provenance",p.display()));}
   let overlay=f.arrows.get(slug).map(|path|read(path,r.nx*r.ny).map(|d|arrows(&lat,&lon,&d,wet.as_deref(),&values,r.nx,r.ny))).transpose()?;
   let panel=PanelRequest {lat_deg:&lat,lon_deg:&lon,projection:None,ny:r.ny,nx:r.nx,values,product_slug:slug.clone(),title:format!("{title}; scale {low} to {high}"),display_units:units.into(),scale,cbar_tick_step:None,legend:LegendControls{mode:LegendMode::Stepped,..Default::default()},render_density:Default::default(),subtitle_left:f.valid_time.clone(),subtitle_center:if missing.is_some(){Some("Grey: no active partition; blank: dry".into())}else{overlay.as_ref().map(|_|"Arrows: travel toward".into())},subtitle_right:r.source_label.clone(),width:1200,height:900,contours:vec![],colorbar:true,overlays:overlay.as_ref(),annotations:None,out_path:p.clone()};
   if let Some(missing)=missing{render_panel_cells_with_missing(panel,&missing)?;}else{render_panel_cells(panel)?;}
   receipt.push(json!({"product":slug,"valid_time":f.valid_time,"path":p,"min":min,"max":max,"units":units,"display_range":[low,high],"direction_input":"FROM clockwise from north","arrows":"TOWARD","arrow_segments":overlay.as_ref().map(|o|o.lines.len())}));
  }
  Ok(receipt)
 }).collect();
 let receipt:Vec<Value>=receipts.into_iter().collect::<R<Vec<_>>>()?.into_iter().flatten().collect();
 let receipt=json!({"schema":"gpuwm.ocean.render-receipt/v1","renderer":"rw_wrfbatch","threads":rayon::current_num_threads(),"wall_seconds":start.elapsed().as_secs_f64(),"products":receipt});
 let receipt_path=if r.join_existing_output{let folder=out.join(&r.domain).join(r.family.as_deref().unwrap_or("ocean"));fs::create_dir_all(&folder).map_err(|e|e.to_string())?;folder.join("render-receipt.json")}else{out.join("receipt.json")};
 fs::write(receipt_path,serde_json::to_vec_pretty(&receipt).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;println!("{}",receipt);Ok(())
}
pub fn cli(args:&[String])->Option<R<()>>{match args.first().map(String::as_str){Some("--ocean-products")=>Some(args.get(1).ok_or("--ocean-products needs request JSON".into()).and_then(|p|run(Path::new(p)))),Some("--ocean-products-abi")=>Some({println!("gpuwm.ocean.products/v1");Ok(())}),_=>None}}
