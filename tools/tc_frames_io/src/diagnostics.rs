//! Field-derived TC intensity, quadrant radii and an Earth-fixed running swath.
//! No tracker smoothing or subcell fit enters an intensity measurement.
use super::{err, Field, ERROR};
use rayon::prelude::*;
use serde::Deserialize;
use serde_json::{json, Value};
use std::{collections::BTreeMap, ffi::{c_char,CStr,CString}, fs, path::PathBuf, slice,
          sync::atomic::{AtomicU32, Ordering}};
type Result<T> = std::result::Result<T,String>;
const EARTH_KM:f64=6371.0;
const KT:f64=1852.0/3600.0;

#[derive(Deserialize)]
struct Swath { path:PathBuf, south:f64, north:f64, west:f64, east:f64, spacing_deg:f64 }
#[derive(Deserialize)]
struct Request { centre_deg:[f64;2], radii_centre_deg:Option<[f64;2]>, nx:usize, ny:usize, swath:Option<Swath>,
                 #[serde(default)] unstructured:bool }
fn distance(a:[f64;2],b:[f64;2])->f64 {
    let (la,lb)=(a[0].to_radians(),b[0].to_radians());
    let d=((b[1]-a[1]+180.0).rem_euclid(360.0)-180.0).to_radians();
    let h=((lb-la)*0.5).sin().powi(2)+la.cos()*lb.cos()*(d*0.5).sin().powi(2);
    2.0*EARTH_KM*h.clamp(0.0,1.0).sqrt().asin()
}
fn calculate(r:Request,f:BTreeMap<String,&[f64]>)->Result<Value> {
    let n=r.nx.checked_mul(r.ny).ok_or("grid size overflow")?;
    if n==0 || !r.centre_deg.iter().all(|v|v.is_finite()) {return Err("invalid TC grid or centre".into());}
    let get=|key:&str|->Result<&[f64]> {
        let a=f.get(key).copied().ok_or_else(||format!("TC diagnostics require {key}"))?;
        if a.len()!=n {return Err(format!("TC {key} differs from grid shape"));} Ok(a)
    };
    let (lat,lon)=(get("lat")?,get("lon")?);
    let direct=f.contains_key("mslp_pa");
    let ps=get(if direct {"mslp_pa"} else {"psfc"})?;
    let (ht,t2,q2)=if direct { (None,None,None) } else {(Some(get("ht")?),Some(get("t2")?),Some(get("q2")?))};
    let landmask=f.contains_key("landmask");
    let land=get(if landmask {"landmask"} else {"xland"})?;
    let direct_wind=f.contains_key("wind10");
    let u=get(if direct_wind {"wind10"} else {"u10"})?;
    let v=if direct_wind {None} else {Some(get("v10")?)};
    let pressure:Vec<f64>=(0..n).into_par_iter().map(|i| {
        if !lat[i].is_finite() || !lon[i].is_finite() || !land[i].is_finite()
            || (if landmask {land[i]>=0.5} else {land[i]<=1.5}) || !ps[i].is_finite() || ps[i]<=0.0
            || distance(r.centre_deg,[lat[i],lon[i]])>150.0 {return f64::INFINITY;}
        if direct {return ps[i]*0.01;}
        let (ht,t2,q2)=(ht.unwrap()[i],t2.unwrap()[i],q2.unwrap()[i]);
        if !ht.is_finite() || !t2.is_finite() || t2<=0.0 || !q2.is_finite() {return f64::INFINITY;}
        let tv=t2*(1.0+0.608*q2.max(0.0));
        ps[i]*0.01*(9.80665*ht/(287.0*(tv+0.00325*ht))).exp()
    }).collect();
    let index=(0..n).filter(|&i|pressure[i].is_finite())
        .min_by(|&a,&b|pressure[a].total_cmp(&pressure[b]).then(a.cmp(&b)))
        .ok_or("no finite water-cell pressure within 150 km of the tracked centre")?;
    let centre=[lat[index],lon[index]];
    let speed:Vec<f64>=(0..n).into_par_iter().map(|i|if let Some(v)=v {u[i].hypot(v[i])}
        else if u[i]>=0.0 {u[i]} else {f64::NAN}).collect();
    let radii_centre=r.radii_centre_deg.unwrap_or(centre);
    let radii_distance:Vec<f64>=(0..n).into_par_iter().map(|i|distance(radii_centre,[lat[i],lon[i]])).collect();
    let vmax=(0..n).into_par_iter().filter(|&i|radii_distance[i]<=150.0 && speed[i].is_finite())
        .map(|i|speed[i]).reduce(||f64::NEG_INFINITY,f64::max);
    if !vmax.is_finite() {return Err("no finite 10 m wind within 150 km of the field centre".into());}
    let mut radii=[[0.0_f64;4];3];let mut edge=[[false;4];3];
    for i in 0..n {
        let d=radii_distance[i];if !d.is_finite() || d>1000.0 || !speed[i].is_finite(){continue;}
        let east=(lon[i]-radii_centre[1]+180.0).rem_euclid(360.0)-180.0>=0.0;
        let north=lat[i]>=radii_centre[0];
        let quadrant=match(north,east){(true,true)=>0,(false,true)=>1,(false,false)=>2,(true,false)=>3};
        for (j,threshold) in [34.0,50.0,64.0].iter().enumerate() {
            if speed[i]>=threshold*KT {
                radii[j][quadrant]=radii[j][quadrant].max(d/1.852);
                if !r.unstructured && (i/r.nx==0 || i/r.nx+1==r.ny || i%r.nx==0 || i%r.nx+1==r.nx) {edge[j][quadrant]=true;}
            }
        }
    }
    let swath=if let Some(s)=r.swath {
        if ![s.south,s.north,s.west,s.east,s.spacing_deg].iter().all(|v|v.is_finite())
            || s.spacing_deg<=0.0 || s.south>=s.north || s.west>=s.east {return Err("invalid fixed swath grid".into());}
        let nx=((s.east-s.west)/s.spacing_deg).ceil() as usize+1;
        let ny=((s.north-s.south)/s.spacing_deg).ceil() as usize+1;
        let count=nx.checked_mul(ny).filter(|&c|c<=100_000_000).ok_or("swath grid exceeds 100 million cells")?;
        let metadata=json!({"schema":"gpuwm.tc-wind-swath.v1","dtype":"<f4","units":"m s-1",
            "shape":[ny,nx],"south":s.south,"west":s.west,"spacing_deg":s.spacing_deg,
            "missing":"NaN; no sampled source cell","sampling":"maximum of source-cell 10 m wind at tracker consultation times; nearest geographic bin; no interpolation"});
        let meta=s.path.with_extension("json");
        let existing=s.path.exists();
        let values:Vec<AtomicU32>=if existing {
            let previous:Value=serde_json::from_slice(&fs::read(&meta).map_err(err)?).map_err(err)?;
            if previous!=metadata {return Err("running swath grid or units changed on resume".into());}
            let bytes=fs::read(&s.path).map_err(err)?;
            if bytes.len()!=count*4 {return Err("running swath payload is incomplete".into());}
            bytes.chunks_exact(4).map(|b| {let v=f32::from_le_bytes(b.try_into().unwrap());
                AtomicU32::new(if v.is_finite() && v>=0.0 {v.to_bits()+1}else{0})}).collect()
        }else{(0..count).map(|_|AtomicU32::new(0)).collect()};
        (0..n).into_par_iter().for_each(|i| {
            let lo=s.west+(lon[i]-s.west).rem_euclid(360.0);
            if !speed[i].is_finite() || !lat[i].is_finite() || !lo.is_finite()
                || lat[i]<s.south || lat[i]>s.north || lo>s.east {return;}
            let x=((lo-s.west)/s.spacing_deg).round() as usize;
            let y=((lat[i]-s.south)/s.spacing_deg).round() as usize;
            if x<nx&&y<ny {values[y*nx+x].fetch_max((speed[i] as f32).to_bits()+1,Ordering::Relaxed);}
        });
        let bytes:Vec<u8>=values.iter().flat_map(|v| {let b=v.load(Ordering::Relaxed);
            (if b==0 {f32::NAN}else{f32::from_bits(b-1)}).to_le_bytes()}).collect();
        if let Some(p)=s.path.parent(){fs::create_dir_all(p).map_err(err)?;}
        // Publish complete geometry before the first payload, then leave it
        // immutable. A killed update must not truncate resume metadata while
        // the previous maximum field is still valid.
        if !existing {
            let pending=meta.with_extension(format!("json.tmp-{}",std::process::id()));
            fs::write(&pending,serde_json::to_vec_pretty(&metadata).map_err(err)?).map_err(err)?;
            fs::rename(&pending,&meta).map_err(err)?;
        }
        let tmp=s.path.with_extension(format!("tmp-{}",std::process::id()));
        fs::write(&tmp,bytes).map_err(err)?;fs::rename(&tmp,&s.path).map_err(err)?;
        Some(json!({"field":s.path,"metadata":meta}))
    }else{None};
    Ok(json!({"schema":"gpuwm.tc-field-fix.v1","latitude_deg":centre[0],"longitude_deg":centre[1],
        "cell_index":index,"mslp_hpa":pressure[index],"vmax10_m_s":vmax,"vmax10_kt":vmax/KT,
        "radii_nm":{"34":radii[0],"50":radii[1],"64":radii[2]},"quadrants":["NE","SE","SW","NW"],
        "radii_reach_grid_edge":{"34":edge[0],"50":edge[1],"64":edge[2]},
        "intensity_definition":"raw water-cell minimum of sea-level surface pressure; 10 m wind maximum within 150 km of that cell; no smoothing",
        "swath":swath}))
}

#[no_mangle]
pub unsafe extern "C" fn tc_fields_diagnose(request:*const c_char,fields:*const Field,count:usize)->*mut c_char {
    let result=std::panic::catch_unwind(||->Result<String>{
        if request.is_null()||fields.is_null(){return Err("null TC diagnostic request".into());}
        let r:Request=serde_json::from_slice(CStr::from_ptr(request).to_bytes()).map_err(err)?;
        let mut f=BTreeMap::new();
        for item in slice::from_raw_parts(fields,count) {
            if item.name.is_null()||item.values.is_null(){return Err("null TC field".into());}
            let name=CStr::from_ptr(item.name).to_str().map_err(err)?.to_owned();
            if f.insert(name,slice::from_raw_parts(item.values,item.count)).is_some(){return Err("duplicate TC field".into());}
        }
        serde_json::to_string(&calculate(r,f)?).map_err(err)
    }).unwrap_or_else(|_|Err("Rust TC diagnostic panic".into()));
    match result {Ok(s)=>CString::new(s).unwrap().into_raw(),Err(e)=>{
        ERROR.with(|v|*v.borrow_mut()=CString::new(e.replace('\0'," ")).unwrap());std::ptr::null_mut()}}
}
#[no_mangle]
pub unsafe extern "C" fn tc_fields_free(value:*mut c_char) {if !value.is_null(){drop(CString::from_raw(value));}}
