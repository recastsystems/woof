//! Metadata-driven NetCDF surface vectors, in geographic or native grid axes.
//! Coordinate interpolation is bilinear in each source quadrilateral. Rotate
//! native vectors at source points before interpolating east/north components.
use crate::{arrays, grid::Grid, source, text, Result};
use rayon::prelude::*;
use serde_json::{json, Value};
use std::{collections::{BTreeMap, HashMap}, fs, path::Path};

type Stencil = Vec<(usize, f64)>;
fn wrap(x:f64, origin:f64)->f64{origin+(x-origin+180.).rem_euclid(360.)-180.}
fn plane(file:&netcrust::File, name:&str, ti:usize, ny:usize, nx:usize)->Result<Vec<f64>>{
    let shape=file.variable(name).ok_or_else(||format!("missing wind variable {name}"))?.shape();
    let selections=match shape.as_slice(){
        [y,x] if *y==ny&&*x==nx=>vec![],
        [t,y,x] if *y==ny&&*x==nx&&(*t==1||ti<*t)=>vec![netcrust::NcSliceInfoElem::Slice{start:if *t==1{0}else{ti as u64},end:if *t==1{1}else{ti as u64+1},step:1},netcrust::NcSliceInfoElem::Slice{start:0,end:ny as u64,step:1},netcrust::NcSliceInfoElem::Slice{start:0,end:nx as u64,step:1}],
        _=>return Err(format!("{name} must be [y,x] or [time,y,x], got {shape:?}").into())
    };
    let a=if selections.is_empty(){file.read_array_f64(name)}else{file.read_array_f64_slice(name,&netcrust::NcSliceInfo{selections})}.map_err(|e|e.to_string())?;
    Ok(a.values().to_vec())
}

fn structured_stencils(lat:&[f64],lon:&[f64],ny:usize,nx:usize,target:&Grid)->Vec<Option<Stencil>>{
    let origin=lon[0];let lon:Vec<_>=lon.par_iter().map(|&x|wrap(x,origin)).collect();
    // One-degree bins only accelerate candidate lookup; all accepted weights
    // come from the actual four source coordinates, including rotated grids.
    let cells:Vec<_>=(0..(ny-1)*(nx-1)).into_par_iter().map(|k|{
        let j=k/(nx-1)*nx+k%(nx-1);let ids=[j,j+1,j+nx,j+nx+1];
        let xmin=ids.iter().map(|&i|lon[i]).fold(f64::INFINITY,f64::min).floor() as i32;
        let xmax=ids.iter().map(|&i|lon[i]).fold(f64::NEG_INFINITY,f64::max).floor() as i32;
        let ymin=ids.iter().map(|&i|lat[i]).fold(f64::INFINITY,f64::min).floor() as i32;
        let ymax=ids.iter().map(|&i|lat[i]).fold(f64::NEG_INFINITY,f64::max).floor() as i32;
        (ids,xmin,xmax,ymin,ymax)
    }).collect();
    let mut bins:HashMap<(i32,i32),Vec<usize>>=HashMap::new();
    for(k,(_,x0,x1,y0,y1))in cells.iter().enumerate(){for x in *x0..=*x1{for y in *y0..=*y1{bins.entry((x,y)).or_default().push(k);}}}
    (0..target.len()).into_par_iter().map(|i|{
        let x=wrap(target.lon[i] as f64,origin);let y=target.lat[i] as f64;
        let candidates=bins.get(&(x.floor() as i32,y.floor() as i32))?;
        for &k in candidates{let ids=cells[k].0;let(mut a,mut b)=(0.5,0.5);
            for _ in 0..12{
                let w=[(1.-a)*(1.-b),a*(1.-b),(1.-a)*b,a*b];
                let px=(0..4).map(|j|w[j]*lon[ids[j]]).sum::<f64>();let py=(0..4).map(|j|w[j]*lat[ids[j]]).sum::<f64>();
                let xa=(1.-b)*(lon[ids[1]]-lon[ids[0]])+b*(lon[ids[3]]-lon[ids[2]]);
                let xb=(1.-a)*(lon[ids[2]]-lon[ids[0]])+a*(lon[ids[3]]-lon[ids[1]]);
                let ya=(1.-b)*(lat[ids[1]]-lat[ids[0]])+b*(lat[ids[3]]-lat[ids[2]]);
                let yb=(1.-a)*(lat[ids[2]]-lat[ids[0]])+a*(lat[ids[3]]-lat[ids[1]]);
                let det=xa*yb-xb*ya;if det.abs()<1e-12{break}
                let da=((x-px)*yb-(y-py)*xb)/det;let db=((y-py)*xa-(x-px)*ya)/det;a+=da;b+=db;
                if da.abs()+db.abs()<1e-8&&a>=-1e-6&&a<=1.+1e-6&&b>=-1e-6&&b<=1.+1e-6{
                    a=a.clamp(0.,1.);b=b.clamp(0.,1.);let w=[(1.-a)*(1.-b),a*(1.-b),(1.-a)*b,a*b];return Some(ids.into_iter().zip(w).collect());
                }
            }
        }None
    }).collect()
}

pub fn read(frame:&Value,fields:&Value,target:&Grid)->Result<BTreeMap<String,Vec<f32>>>{
    let path=Path::new(text(frame,"wind")?);let options=&frame["wind_source"];
    if options.is_null()||options["format"]=="grib"{return source::grib(path,fields,target,0)}
    if options["format"]!="netcdf"{return Err("wind format must be grib or netcdf".into())}
    let file=netcrust::open(path).map_err(|e|e.to_string())?;
    let valid=chrono::DateTime::parse_from_rfc3339(text(frame,"valid")?)?.timestamp();
    let times=if options["time_encoding"]=="text"{
        file.read_strings(text(options,"time")?).map_err(|e|e.to_string())?.iter().map(|s|chrono::NaiveDateTime::parse_from_str(s.trim(),text(options,"time_format")?).map(|t|t.and_utc().timestamp()).map_err(|e|e.to_string().into())).collect::<Result<Vec<_>>>()?
    }else{if options["time_encoding"]!="unix_seconds"{return Err("declare wind time_encoding as unix_seconds or text".into())}arrays::read_nc(path,text(options,"time")?)?.iter().map(|v|*v as i64).collect()};
    let ti=times.iter().position(|&t|t==valid).ok_or("wind file lacks exact valid time")?;
    let latname=text(options,"latitude")?;let lonname=text(options,"longitude")?;
    let shape=file.variable(latname).ok_or("wind latitude missing")?.shape();
    let(lat,lon,ny,nx,stencils)=if shape.len()==1{
        let lat=arrays::read_nc(path,latname)?;let lon=arrays::read_nc(path,lonname)?;let(ny,nx)=(lat.len(),lon.len());
        if nx<2||ny<2{return Err("source wind axes need at least two points".into())}
        for axis in [&lat,&lon]{if axis.iter().any(|v|!v.is_finite())||!axis.windows(2).all(|w|(w[1]-w[0])*(axis[1]-axis[0])>0.){return Err("source wind axes must be finite and strictly monotonic".into())}}
        let lo=lon[0].min(lon[nx-1]);let hi=lon[0].max(lon[nx-1]);
        let stencil=(0..target.len()).into_par_iter().map(|i|{let mut x=wrap(target.lon[i] as f64,0.5*(lo+hi));if x<lo&&x+360.<=hi{x+=360.}let(ix,a)=arrays::bracket(&lon,x)?;let(iy,b)=arrays::bracket(&lat,target.lat[i] as f64)?;let j=iy*nx+ix;Some(vec![(j,(1.-a)*(1.-b)),(j+1,a*(1.-b)),(j+nx,(1.-a)*b),(j+nx+1,a*b)])}).collect::<Vec<_>>();
        (lat,lon,ny,nx,stencil)
    }else{
        if shape.len()<2{return Err("invalid source wind coordinates".into())}let(ny,nx)=(shape[shape.len()-2],shape[shape.len()-1]);if nx<2||ny<2{return Err("source wind grid too small".into())}
        let lat=plane(&file,latname,ti,ny,nx)?;let lon=plane(&file,lonname,ti,ny,nx)?;
        if lat.iter().chain(&lon).any(|v|!v.is_finite()){return Err("source wind coordinates nonfinite".into())}
        let stencils=structured_stencils(&lat,&lon,ny,nx,target);(lat,lon,ny,nx,stencils)
    };drop(lat);drop(lon);
    let u=plane(&file,text(&options["variables"],"u10_east")?,ti,ny,nx)?;let v=plane(&file,text(&options["variables"],"v10_north")?,ti,ny,nx)?;
    let rotation=if options["vector_frame"]=="grid"{Some((plane(&file,text(options,"cosalpha")?,ti,ny,nx)?,plane(&file,text(options,"sinalpha")?,ti,ny,nx)?))}else if options["vector_frame"]=="earth"{None}else{return Err("declare source vector_frame as earth or grid".into())};
    let vectors:Vec<_>=(0..nx*ny).into_par_iter().map(|i|{let(u,v)=(u[i],v[i]);if !u.is_finite()||!v.is_finite()||u.abs()>150.||v.abs()>150.{return (f64::NAN,f64::NAN)}if let Some((c,s))=&rotation{if !c[i].is_finite()||!s[i].is_finite()||(c[i]*c[i]+s[i]*s[i]-1.).abs()>2e-4{return (f64::NAN,f64::NAN)}(u*c[i]-v*s[i],u*s[i]+v*c[i])}else{(u,v)}}).collect();
    let sampled:Vec<_>=stencils.par_iter().map(|st|{let(mut a,mut b,mut sum)=(0.,0.,0.);if let Some(st)=st{for &(i,w)in st{let(u,v)=vectors[i];if u.is_finite()&&v.is_finite(){a+=w*u;b+=w*v;sum+=w;}}}if sum>1e-12{((a/sum) as f32,(b/sum) as f32)}else{(f32::NAN,f32::NAN)}}).collect();
    Ok(BTreeMap::from([("u10_east".into(),sampled.par_iter().map(|p|p.0).collect()),("v10_north".into(),sampled.par_iter().map(|p|p.1).collect())]))
}

pub fn forcing(r:&Value,out:&Path)->Result<Value>{
    let source=Path::new(text(r,"prepared")?);let manifest:Value=serde_json::from_slice(&fs::read(source.join("manifest.json"))?)?;
    let grid=Grid::read(&manifest["request"]["grid"])?;let wetrow=&manifest["arrays"]["wet"];let wetpath=source.join(text(wetrow,"path")?);
    if arrays::sha(&wetpath)?!=text(wetrow,"sha256")?{return Err("prepared wet hash differs".into())}let bytes=fs::read(wetpath)?;let wet:Vec<_>=bytes.chunks_exact(4).map(|b|i32::from_le_bytes(b.try_into().unwrap())).collect();
    if wet.len()!=grid.len(){return Err("prepared grid/wet shape differs".into())}
    let frames=r["frames"].as_array().ok_or("forcing frames required")?;let times:Vec<_>=frames.iter().map(|f|chrono::DateTime::parse_from_rfc3339(text(f,"valid")?).map(|t|t.timestamp()).map_err(|e|e.to_string().into())).collect::<Result<Vec<_>>>()?;
    if json!(times)!=manifest["forcing_times"]{return Err("alternate forcing must cover every prepared valid time".into())}
    let table=source::fields(&manifest["request"])?;let results=frames.par_iter().map(|f|read(f,&table["wind"],&grid)).collect::<Result<Vec<_>>>()?;let mut arrays=serde_json::Map::new();
    for key in ["u10_east","v10_north"]{let mut values=Vec::new();for result in &results{if result[key].iter().enumerate().any(|(i,v)|wet[i]==1&&!v.is_finite()){return Err(format!("alternate forcing has uncovered wet cells: {key}").into())}values.extend(result[key].iter().enumerate().map(|(i,v)|if wet[i]==1{*v}else{0.}));}arrays.insert(key.into(),arrays::f32(out,key,&values,&[times.len(),grid.len()])?);}
    let result=json!({"schema":"gpuwm.waves.forcing/v1","status":"complete","prepared":source,"prepared_manifest_sha256":arrays::sha(&source.join("manifest.json"))?,"forcing_times":times,"arrays":arrays,"request":r});fs::write(out.join("manifest.json"),serde_json::to_vec_pretty(&result)?)?;Ok(result)
}
