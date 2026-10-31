use crate::{Result,text,grid::Grid};
use grib_core::grib2::{Grib2File,unpack_message};
use rayon::prelude::*;
use serde_json::Value;
use std::{collections::BTreeMap,path::Path};
pub struct Plane{pub values:Vec<f64>,nx:usize,ny:usize,x0:f64,y0:f64,dx:f64,dy:f64,cyclic:bool}
impl Plane {
    fn neighbours(&self,lat:f64,lon:f64)->Option<Vec<(usize,f64)>>{
        let mut x=(lon-self.x0).rem_euclid(360.)/self.dx;
        if self.dx<0.{x=-((self.x0-lon).rem_euclid(360.))/self.dx;}
        let y=(lat-self.y0)/self.dy;if y<0.||y>self.ny as f64-1.||(!self.cyclic&&(x<0.||x>self.nx as f64-1.)){return None}
        let ix=(x.floor() as usize).min(self.nx-1);let iy=(y.floor() as usize).min(self.ny-2);let fx=x-ix as f64;let fy=y-iy as f64;
        let x1=if self.cyclic{(ix+1)%self.nx}else{(ix+1).min(self.nx-1)};
        Some(vec![(iy*self.nx+ix,(1.-fx)*(1.-fy)),(iy*self.nx+x1,fx*(1.-fy)),((iy+1)*self.nx+ix,(1.-fx)*fy),((iy+1)*self.nx+x1,fx*fy)])
    }
    pub fn sample(&self,lat:f64,lon:f64,circular:bool,coast_radius:usize,energy_weight:Option<&[f64]>)->Option<f32>{
        let ns=self.neighbours(lat,lon)?;let mut total=0.;let mut sum=0.;let mut sy=0.;
        for &(j,w) in &ns {let v=self.values[j];let w=w*energy_weight.map_or(1.,|hs|if hs[j].is_finite(){hs[j].powi(2)}else{0.});if v.is_finite()&&v.abs()<1e20&&w>0.{total+=w;if circular {sum+=w*v.to_radians().cos();sy+=w*v.to_radians().sin();}else{sum+=w*v;}}}
        if total>1e-10 {return Some(if circular{sy.atan2(sum).to_degrees().rem_euclid(360.) as f32}else{(sum/total) as f32});}
        // Missing source land cells are never averaged as zero. Bounded nearest-wet
        // coast fill is recorded in the request and cannot fill a distant ocean gap.
        let center=ns[0].0;let(cx,cy)=(center%self.nx,center/self.nx);let mut best=None;let mut dist=usize::MAX;
        for dy in -(coast_radius as isize)..=coast_radius as isize {for dx in -(coast_radius as isize)..=coast_radius as isize {let y=cy as isize+dy;let mut x=cx as isize+dx;if y<0||y>=self.ny as isize{continue}if self.cyclic{x=x.rem_euclid(self.nx as isize);}if x<0||x>=self.nx as isize{continue}let v=self.values[y as usize*self.nx+x as usize];let d=(dx*dx+dy*dy) as usize;if d<dist&&v.is_finite()&&v.abs()<1e20 {dist=d;best=Some(v as f32);}}}best
    }
}
pub fn grib(path:&Path,fields:&Value,grid:&Grid,coast_radius:usize)->Result<BTreeMap<String,Vec<f32>>>{
    let file=Grib2File::open(path.to_str().ok_or("GRIB path must be UTF-8")?).map_err(|e|e.to_string())?;
    let rows=fields.as_object().ok_or("fields must be object")?;
    let mut planes:BTreeMap<String,Plane>=rows.iter().collect::<Vec<_>>().into_par_iter().map(|(name,row)|->Result<_>{
        let code=row["grib"].as_array().ok_or("field grib tuple required")?;
        if code.len()!=3{return Err("grib tuple must be discipline,category,parameter".into());}
        let found:Vec<_>=file.messages.iter().filter(|m|m.discipline as u64==code[0].as_u64().unwrap_or(999)&&m.product.parameter_category as u64==code[1].as_u64().unwrap_or(999)&&m.product.parameter_number as u64==code[2].as_u64().unwrap_or(999)&&row["level"].as_f64().is_none_or(|v|(m.product.level_value-v).abs()<1e-6)&&row["level_type"].as_u64().is_none_or(|v|m.product.level_type as u64==v)).collect();
        if found.len()!=1{return Err(format!("field {name} matched {} GRIB records in {}",found.len(),path.display()).into());}
        let m=found[0];let g=&m.grid;
        if g.template!=0||g.nx<2||g.ny<2||g.scan_mode&0x3f!=0{return Err(format!("GRIB grid {} scan {} needs explicit coordinate mapping",g.template,g.scan_mode).into());}
        if row["vector"]==true && g.resolution_flags&8!=0{return Err("source grid-relative wind needs a supplied source rotation".into());}
        let dx=if g.scan_mode&0x80!=0{-g.dx}else{g.dx};let dy=(g.lat2-g.lat1)/(g.ny-1) as f64;
        let plane=Plane{values:unpack_message(m).map_err(|e|e.to_string())?,nx:g.nx as usize,ny:g.ny as usize,x0:g.lon1,y0:g.lat1,dx,dy,cyclic:((g.nx as f64)*dx.abs()-360.).abs()<0.01};
        Ok((name.clone(),plane))
    }).collect::<Result<_>>()?;
    for(name,row)in rows{if let Some(wet_name)=row["zero_when_wet_missing"].as_str(){let wet=planes.get(wet_name).ok_or("wet mask field missing")?.values.clone();let plane=planes.get_mut(name).ok_or("field missing")?;if plane.values.len()!=wet.len(){return Err("source mask and partition grids differ".into());}plane.values.par_iter_mut().zip(&wet).for_each(|(v,h)|if !v.is_finite()&&h.is_finite(){*v=0.});}}
    rows.iter().collect::<Vec<_>>().into_par_iter().map(|(name,row)|->Result<_>{
        let plane=&planes[name];let circular=row["direction"].is_string();let toward=row["direction"]=="toward_nautical";
        let weights=row["energy_weight"].as_str().map(|s|planes.get(s).map(|p|p.values.as_slice()).ok_or("partition height field missing")).transpose()?;
        // A float64 angle just below 360 can round to 360 in float32. Wrap
        // after the cast too, including the nearest-wet coast-fill path.
        let values=(0..grid.len()).into_par_iter().map(|i|plane.sample(grid.lat[i] as f64,grid.lon[i] as f64,circular,coast_radius,weights).map(|v|if toward{(v+180.).rem_euclid(360.)}else if circular{v.rem_euclid(360.)}else{v}).unwrap_or(f32::NAN)).collect();Ok((name.clone(),values))
    }).collect()
}
pub fn fields(r:&Value)->Result<Value>{let p=Path::new(text(r,"field_table")?);Ok(serde_json::from_slice(&std::fs::read(p)?)?)}
pub fn inspect(r:&Value,out:&Path)->Result<Value>{
    let s=crate::spectra::station(Path::new(text(r,"spectrum")?),i64::MIN,i64::MAX)?;let first=s.first().ok_or("empty station spectrum")?;
    let grid=Grid::read(&serde_json::json!({"kind":"latlon","bounds":[first.lon-0.25,first.lat-0.25,first.lon+0.25,first.lat+0.25],"resolution_degrees":0.25}))?;
    let table=fields(r)?;let values=grib(Path::new(text(r,"wave")?),&table["wave"],&grid,2)?;
    let center:BTreeMap<_,_>=values.iter().map(|(k,v)|(k,v[4])).collect();
    let receipt=serde_json::json!({"status":"complete","station":first.id,"time":first.epoch,"lat":first.lat,"lon":first.lon,"grib_interpolated":center,"station_spectrum_moments":crate::spectra::moments(first),"field_table":table});
    std::fs::write(out.join("source-check.json"),serde_json::to_vec_pretty(&receipt)?)?;Ok(receipt)
}
