use crate::{Result,text,arrays};
use rayon::prelude::*;
use serde_json::Value;
use std::path::Path;
const EARTH:f64=6_371_008.8;
pub struct Grid {pub lat:Vec<f32>,pub lon:Vec<f32>,pub area:Vec<f32>,pub cos:Vec<f32>,pub sin:Vec<f32>,pub mapfac:Vec<f32>,pub nx:usize,pub ny:usize,pub kind:String,pub dx:f64,pub dy:f64}
impl Grid {
    pub fn len(&self)->usize{self.lat.len()}
    pub fn read(r:&Value)->Result<Self>{
        let kind=text(r,"kind")?.to_string();
        let (lat,lon,nx,ny,dx,dy):(Vec<f32>,Vec<f32>,usize,usize,f64,f64)=if kind=="latlon" {
            let b=r["bounds"].as_array().ok_or("grid bounds required")?;if b.len()!=4{return Err("bounds must be west,south,east,north".into());}
            let b:Vec<f64>=b.iter().map(|v|v.as_f64().ok_or("numeric bounds required")).collect::<std::result::Result<_,_>>()?;
            let d=r["resolution_degrees"].as_f64().ok_or("resolution required")?;
            if !(d>0.&&d<=10.&&b[2]>b[0]&&b[2]-b[0]<=360.&&b[3]>b[1]&&b[1]>=-90.&&b[3]<=90.){return Err("invalid geographic bounds or spacing".into());}
            let nx=((b[2]-b[0])/d).round() as usize+1;let ny=((b[3]-b[1])/d).round() as usize+1;
            if nx*ny>20_000_000{return Err("target grid exceeds 20 million cells".into());}
            ((0..nx*ny).into_par_iter().map(|i|(b[1]+(i/nx) as f64*d) as f32).collect(),(0..nx*ny).into_par_iter().map(|i|(b[0]+(i%nx) as f64*d) as f32).collect(),nx,ny,d.to_radians()*EARTH,d.to_radians()*EARTH)
        }else if kind=="curvilinear"||kind=="points" {
            let file=Path::new(text(r,"file")?);let latname=r["latitude"].as_str().unwrap_or("XLAT");let lonname=r["longitude"].as_str().unwrap_or("XLONG");
            if let Some(expected)=r["sha256"].as_str(){if arrays::sha(file)?!=expected{return Err("target grid hash differs from preparation request".into())}}
            let mut lat=arrays::read_nc(file,latname)?;let mut lon=arrays::read_nc(file,lonname)?;
            let f=netcrust::open(file).map_err(|e|e.to_string())?;let sh=f.variable(latname).ok_or("latitude missing")?.shape();
            let(nx,ny)=if kind=="points"{(lat.len(),1)}else{if sh.len()<2{return Err("structured coordinates need [y,x] or [time,y,x]".into());}(sh[sh.len()-1],sh[sh.len()-2])};
            if lat.len()<nx*ny||lon.len()<nx*ny{return Err("coordinate shapes differ".into());}lat.truncate(nx*ny);lon.truncate(nx*ny);
            if r["coordinate_units"]=="radian" {lat.par_iter_mut().for_each(|v|*v=v.to_degrees());lon.par_iter_mut().for_each(|v|*v=v.to_degrees());}
            let dx=r["dx"].as_f64().unwrap_or(0.);let dy=r["dy"].as_f64().unwrap_or(0.);
            (lat.into_par_iter().map(|v|v as f32).collect(),lon.into_par_iter().map(|v|v as f32).collect(),nx,ny,dx,dy)
        }else{return Err("grid kind must be latlon, curvilinear or points".into());};
        let field=|key:&str,default:f32|->Result<Vec<f32>>{if let Some(name)=r[key].as_str(){let v=arrays::read_nc(Path::new(text(r,"file")?),name)?;if v.len()<nx*ny{return Err(format!("{key} field shape mismatch").into());}Ok(v[..nx*ny].par_iter().map(|v|*v as f32).collect())}else{Ok(vec![default;nx*ny])}};
        if kind=="curvilinear" && (r["cosalpha"].is_null()||r["sinalpha"].is_null()||r["map_factor"].is_null()||dx<=0.||dy<=0.){return Err("curvilinear grid needs map_factor, cosalpha, sinalpha and positive dx/dy".into());}
        if kind=="points"&&r["area"].is_null(){return Err("point grids need area variable in m2".into());}
        let cos=field("cosalpha",1.)?;let sin=field("sinalpha",0.)?;let mapfac=field("map_factor",1.)?;
        let area=if kind=="points"{field("area",0.)?}else{(0..nx*ny).into_par_iter().map(|i|if kind=="latlon" {let half=dy/EARTH/2.;(EARTH*dx*((lat[i] as f64).to_radians()+half).min(std::f64::consts::FRAC_PI_2).sin()-EARTH*dx*((lat[i] as f64).to_radians()-half).max(-std::f64::consts::FRAC_PI_2).sin()) as f32}else{(dx*dy/(mapfac[i] as f64).powi(2)) as f32}).collect()};
        if (0..nx*ny).into_par_iter().any(|i|!lat[i].is_finite()||lat[i].abs()>90.||!lon[i].is_finite()||!area[i].is_finite()||area[i]<=0.||!mapfac[i].is_finite()||mapfac[i]<=0.||!cos[i].is_finite()||!sin[i].is_finite()||(cos[i]*cos[i]+sin[i]*sin[i]-1.).abs()>2e-4){return Err("invalid geometry, area, map factor or rotation".into());}
        Ok(Self{lat,lon,area,cos,sin,mapfac,nx,ny,kind,dx,dy})
    }
    pub fn depth(&self,r:&Value)->Result<Vec<f32>>{
        let path=Path::new(text(r,"path")?);let digest=arrays::sha(path)?;
        if text(r,"sha256")?!=digest{return Err("bathymetry hash differs".into());}
        let lat=arrays::read_nc(path,r["latitude"].as_str().unwrap_or("lat"))?;let lon=arrays::read_nc(path,r["longitude"].as_str().unwrap_or("lon"))?;
        let positions:Vec<(usize,usize,f64,f64)>=(0..self.len()).into_par_iter().map(|i|->Result<_>{
            let mut x=(self.lon[i] as f64-lon[0]).rem_euclid(360.)+lon[0];let mut y=self.lat[i] as f64;
            // Pixel-centred global DEM has a half-pixel seam gap. Nearest edge
            // value there is a bounded <= half-pixel shift, never a coverage hole.
            if lon[lon.len()-1]-lon[0]>359. {if x>lon[lon.len()-1]{x=if x-lon[lon.len()-1]<lon[0]+360.-x{lon[lon.len()-1]}else{lon[0]};}}
            // Global pixel centres leave half a source pixel at either pole,
            // just as they do at the longitude seam. Extend only that edge
            // footprint; a cropped regional DEM still fails outside coverage.
            let last=lat.len()-1;
            if y<lat[0]&&y>=-90.&&lat[0]+90.<=(lat[1]-lat[0])*0.51{y=lat[0];}
            if y>lat[last]&&y<=90.&&90.-lat[last]<=(lat[last]-lat[last-1])*0.51{y=lat[last];}
            let(ix,fx)=arrays::bracket(&lon,x).ok_or("bathymetry longitude coverage gap")?;let(iy,fy)=arrays::bracket(&lat,y).ok_or("bathymetry latitude coverage gap")?;
            Ok((ix,iy,fx,fy))}).collect::<Result<_>>()?;
        let y0=positions.iter().map(|p|p.1).min().ok_or("empty target")?;let y1=positions.iter().map(|p|p.1+2).max().unwrap();
        let mut xs:Vec<usize>=positions.iter().map(|p|p.0).collect();xs.sort_unstable();xs.dedup();
        let split=xs.windows(2).enumerate().max_by_key(|(_,w)|w[1]-w[0]).filter(|(_,w)|w[1]-w[0]>lon.len()/2).map(|(i,_)|i+1);
        let groups:Vec<&[usize]>=if let Some(i)=split{vec![&xs[..i],&xs[i..]]}else{vec![&xs]};
        let variable=r["elevation"].as_str().unwrap_or("z");let f=netcrust::open(path).map_err(|e|e.to_string())?;
        if f.variable(variable).ok_or("elevation missing")?.shape()!=[lat.len(),lon.len()]{return Err("bathymetry must have [lat,lon] elevation".into());}
        let mut tiles=Vec::new();for group in groups{let x0=group[0];let x1=group[group.len()-1]+2;let a=f.read_array_f64_slice(variable,&netcrust::NcSliceInfo{selections:vec![netcrust::NcSliceInfoElem::Slice{start:y0 as u64,end:y1 as u64,step:1},netcrust::NcSliceInfoElem::Slice{start:x0 as u64,end:x1 as u64,step:1}]}).map_err(|e|e.to_string())?;tiles.push((x0,x1,a.values().to_vec()));}
        positions.into_par_iter().map(|(ix,iy,fx,fy)|{
            let tile=tiles.iter().find(|t|ix>=t.0&&ix+1<t.1).ok_or("bathymetry tile gap")?;let nx=tile.1-tile.0;let j=(iy-y0)*nx+ix-tile.0;let z=&tile.2;
            let v=[z[j],z[j+1],z[j+nx],z[j+nx+1]];
            if v.iter().any(|v|!v.is_finite()||v.abs()>20000.){return Err("invalid bathymetry elevation".into());}
            let bed=(1.-fy)*((1.-fx)*v[0]+fx*v[1])+fy*((1.-fx)*v[2]+fx*v[3]);Ok((-bed).max(0.) as f32)
        }).collect()
    }
}
pub fn distance_km(a:f32,b:f32,c:f32,d:f32)->f64{
    let dy=((a-c) as f64).to_radians();let dx=((b-d) as f64).to_radians();let q=(dy/2.).sin().powi(2)+(a as f64).to_radians().cos()*(c as f64).to_radians().cos()*(dx/2.).sin().powi(2);2.*EARTH/1000.*q.clamp(0.,1.).sqrt().asin()
}
pub fn export_points(r:&Value,out:&Path)->Result<Value>{
    use netcdf_writer::{AttrValue,NcFormat,NcType,NcWriter,Schema,VarData};
    let root=Path::new(text(r,"prepared")?);let m:Value=serde_json::from_slice(&std::fs::read(root.join("manifest.json"))?)?;
    let read=|name:&str|->Result<Vec<f32>>{let row=&m["arrays"][name];let path=root.join(text(row,"path")?);if arrays::sha(&path)?!=text(row,"sha256")?{return Err("prepared geometry hash differs".into());}let b=std::fs::read(path)?;Ok(b.par_chunks_exact(4).map(|v|f32::from_le_bytes(v.try_into().unwrap())).collect())};
    let lat=read("lat")?;let lon=read("lon")?;let area=read("area")?;let n=lat.len();
    let b=std::fs::read(root.join(text(&m["arrays"]["boundary_cells"],"path")?))?;let boundary:Vec<i32>=b.chunks_exact(4).map(|v|i32::from_le_bytes(v.try_into().unwrap())).collect();
    let mut s=Schema::new(NcFormat::Offset64);let cell=s.def_dim("nCells",n,false)?;let edge=s.def_dim("nBoundary",boundary.len(),false)?;
    let la=s.def_var("latCell",NcType::Double,&[cell])?;let lo=s.def_var("lonCell",NcType::Double,&[cell])?;let ar=s.def_var("areaCell",NcType::Float,&[cell])?;let bc=s.def_var("boundary_cells",NcType::Int,&[edge])?;
    s.put_var_attr(la,"units",AttrValue::Text("radian".into()))?;s.put_var_attr(lo,"units",AttrValue::Text("radian".into()))?;s.put_var_attr(ar,"units",AttrValue::Text("m2".into()))?;
    s.put_global_attr("purpose",AttrValue::Text("Ordered point-list view of source grid, not a Voronoi connectivity mesh".into()))?;
    let path=out.join("points-grid.nc");let mut w=NcWriter::create(&path,s)?;w.write_var(la,VarData::F64(&lat.par_iter().map(|v|(*v as f64).to_radians()).collect::<Vec<_>>()))?;w.write_var(lo,VarData::F64(&lon.par_iter().map(|v|(*v as f64).to_radians()).collect::<Vec<_>>()))?;w.write_var(ar,VarData::F32(&area))?;w.write_var(bc,VarData::I32(&boundary))?;w.finish()?;
    Ok(serde_json::json!({"status":"complete","path":path,"sha256":arrays::sha(&path)?,"cells":n,"boundary_cells":boundary.len(),"source_manifest":root.join("manifest.json")}))
}
