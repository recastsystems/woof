//! Rust container/byte boundary for the compact spectral solver.
//! Field numerics are CUDA. This module retains the legacy NetCDF-C notice.
use super::*;
use std::io::Read;

const PRODUCTS: [&str; 23] = ["hs","tp","tm01","tm02","dp","dir","dspr",
    "ws_hs","ws_tp","ws_dp","sw1_hs","sw1_tp","sw1_dp","sw2_hs","sw2_tp",
    "sw2_dp","sw3_hs","sw3_tp","sw3_dp","ustar","charnock","z0","direction"];
unsafe extern "C" { fn nc_put_var_text(id:c_int,v:c_int,p:*const c_char)->c_int; }

fn finish(result:Res<()>) -> c_int {
    match result { Ok(())=>0, Err(e)=>{eprintln!("waves spectral I/O: {e}");2} }
}
fn def_dim(id:i32,name:&str,len:usize)->Res<i32>{
    let mut d=0;check(unsafe{nc_def_dim(id,cs(name).as_ptr(),len,&mut d)})?;Ok(d)
}
fn def_var(id:i32,name:&str,kind:i32,dims:&[i32])->Res<i32>{
    let mut v=0;check(unsafe{nc_def_var(id,cs(name).as_ptr(),kind,dims.len() as i32,dims.as_ptr(),&mut v)})?;Ok(v)
}
fn product_attrs(id:i32,v:i32,key:&str,coordinates:&str)->Res<()>{
    let unit=if key=="charnock"{"1"}else if key=="ustar"{"m s-1"}
        else if key=="z0"||key.ends_with("hs"){"m"}
        else if key.ends_with("tp")||key.starts_with("tm"){"s"}else{"degree"};
    att(id,v,"units",unit)?;att(id,v,"coordinates",coordinates)?;
    let standard=match key {
        "hs"=>Some("sea_surface_wave_significant_height"),
        "tp"=>Some("sea_surface_wave_period_at_variance_spectral_density_maximum"),
        "tm01"=>Some("sea_surface_wave_mean_period_from_variance_spectral_density_first_frequency_moment"),
        "tm02"=>Some("sea_surface_wave_mean_period_from_variance_spectral_density_second_frequency_moment"),
        "dir"=>Some("sea_surface_wave_from_direction"),
        "z0"=>Some("surface_roughness_length"), _=>None};
    if let Some(s)=standard{att(id,v,"standard_name",s)?;}
    if key=="dp"||key=="dir"||key.ends_with("_dp"){
        att(id,v,"direction_convention","FROM clockwise from true north")?;
    }else if key=="direction"{att(id,v,"direction_convention","TOWARD clockwise from true north")?;}
    att(id,v,"long_name",key)
}

#[no_mangle]
pub unsafe extern "C" fn waves_load_bytes(path:*const c_char,dest:*mut u8,count:usize)->c_int{
    finish((||{
        if path.is_null()||dest.is_null(){return Err("null byte input".into())}
        let p=CStr::from_ptr(path).to_str().map_err(|e|e.to_string())?;
        let mut file=fs::File::open(p).map_err(|e|e.to_string())?;
        if file.metadata().map_err(|e|e.to_string())?.len()!=count as u64{return Err("raw byte count mismatch".into())}
        file.read_exact(std::slice::from_raw_parts_mut(dest,count)).map_err(|e|e.to_string())
    })())
}

#[no_mangle]
pub unsafe extern "C" fn waves_v1_snapshot(path:*const c_char,lat:*const f32,lon:*const f32,
    wet:*const f32,ny:usize,nx:usize,epoch:f64,data:*const f32)->c_int{
    finish((||{
        if path.is_null()||lat.is_null()||lon.is_null()||wet.is_null()||data.is_null()||ny==0||nx==0||!epoch.is_finite(){return Err("invalid snapshot arguments".into())}
        let n=ny.checked_mul(nx).ok_or("grid size overflow")?;
        let p=CStr::from_ptr(path).to_str().map_err(|e|e.to_string())?;
        if Path::new(p).exists(){return Err(format!("snapshot already exists: {p}"))}
        let mut id=0;check(nc_create(cs(p).as_ptr(),0x200,&mut id))?;let nc=Nc(id);
        let t=def_dim(id,"time",1)?;let y=def_dim(id,"y",ny)?;let x=def_dim(id,"x",nx)?;
        let tv=def_var(id,"time",6,&[t])?;let la=def_var(id,"latitude",5,&[y,x])?;
        let lo=def_var(id,"longitude",5,&[y,x])?;let mask=def_var(id,"wet",5,&[y,x])?;
        att(id,-1,"Conventions","CF-1.10")?;att(id,-1,"source","gpuwm compact spectral action solver")?;
        att(id,tv,"units","seconds since 1970-01-01 00:00:00")?;
        att(id,la,"units","degrees_north")?;att(id,lo,"units","degrees_east")?;
        att(id,la,"standard_name","latitude")?;att(id,lo,"standard_name","longitude")?;
        let mut variables=Vec::new();
        for name in PRODUCTS{let v=def_var(id,name,5,&[t,y,x])?;product_attrs(id,v,name,"time latitude longitude")?;variables.push(v);}
        check(nc_enddef(id))?;
        check(nc_put_vara_double(id,tv,[0].as_ptr(),[1].as_ptr(),&epoch))?;
        check(nc_put_var_float(id,la,lat))?;check(nc_put_var_float(id,lo,lon))?;check(nc_put_var_float(id,mask,wet))?;
        for(j,v)in variables.into_iter().enumerate(){check(nc_put_var_float(id,v,data.add(j*n)))?;}
        drop(nc);Ok(())
    })())
}

#[no_mangle]
pub unsafe extern "C" fn waves_v1_points(path:*const c_char,ids:*const c_char,
    lat:*const f32,lon:*const f32,freq:*const f32,direction:*const f32,
    ns:usize,nt:usize,nf:usize,nd:usize,index:usize,epoch:f64,bulk:*const f32,efth:*const f32)->c_int{
    finish((||{
        if path.is_null()||ids.is_null()||lat.is_null()||lon.is_null()||freq.is_null()||direction.is_null()||bulk.is_null()||efth.is_null()
            ||ns==0||nt==0||nf==0||nd==0||index>=nt||!epoch.is_finite(){return Err("invalid station arguments".into())}
        let p=CStr::from_ptr(path).to_str().map_err(|e|e.to_string())?;
        let nc=if index==0{
            if Path::new(p).exists(){return Err(format!("station output already exists: {p}"))}
            let mut id=0;check(nc_create(cs(p).as_ptr(),0x200,&mut id))?;let nc=Nc(id);
            let st=def_dim(id,"station",ns)?;let tm=def_dim(id,"time",nt)?;
            let f=def_dim(id,"frequency",nf)?;let d=def_dim(id,"direction_bin",nd)?;let chars=def_dim(id,"id_strlen",16)?;
            att(id,-1,"Conventions","CF-1.10")?;att(id,-1,"featureType","timeSeries")?;
            let station=def_var(id,"station_id",2,&[st,chars])?;att(id,station,"cf_role","timeseries_id")?;
            let time=def_var(id,"time",6,&[tm])?;att(id,time,"units","seconds since 1970-01-01 00:00:00")?;
            let latitude=def_var(id,"latitude",5,&[st])?;att(id,latitude,"units","degrees_north")?;
            let longitude=def_var(id,"longitude",5,&[st])?;att(id,longitude,"units","degrees_east")?;
            let frequency=def_var(id,"frequency",5,&[f])?;att(id,frequency,"units","Hz")?;
            // The station coordinate name is distinct from the legacy bulk
            // product direction, which is retained as a [station,time] variable.
            let angle=def_var(id,"wave_direction",5,&[d])?;att(id,angle,"units","degree")?;
            att(id,angle,"standard_name","sea_surface_wave_from_direction")?;
            att(id,angle,"direction_convention","FROM clockwise from true north, unsorted spectral-bin order")?;
            let spectrum=def_var(id,"efth",5,&[st,tm,f,d])?;att(id,spectrum,"units","m2 Hz-1 rad-1")?;
            att(id,spectrum,"coordinates","station_id time frequency wave_direction latitude longitude")?;
            for name in PRODUCTS{let v=def_var(id,name,5,&[st,tm])?;product_attrs(id,v,name,"station_id time latitude longitude")?;}
            check(nc_enddef(id))?;
            check(nc_put_var_text(id,station,ids))?;check(nc_put_var_float(id,latitude,lat))?;
            check(nc_put_var_float(id,longitude,lon))?;check(nc_put_var_float(id,frequency,freq))?;
            check(nc_put_var_float(id,angle,direction))?;nc
        }else{Nc::open(p,true)?};
        check(nc_put_vara_double(nc.0,nc.var("time")?,[index].as_ptr(),[1].as_ptr(),&epoch))?;
        let spec=nc.var("efth")?;
        for s in 0..ns{
            check(nc_put_vara_float(nc.0,spec,[s,index,0,0].as_ptr(),[1,1,nf,nd].as_ptr(),efth.add(s*nf*nd)))?;
        }
        for(j,name)in PRODUCTS.iter().enumerate(){let v=nc.var(name)?;
            for s in 0..ns{check(nc_put_vara_float(nc.0,v,[s,index].as_ptr(),[1,1].as_ptr(),bulk.add(j*ns+s)))?;}
        }
        drop(nc);Ok(())
    })())
}
