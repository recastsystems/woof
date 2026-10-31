//! Shared public preparation for structured and point grids.
//! Extends the sibling coupled preparation using the same metadata and native stages.
use crate::{Result,text,stage,grid,arrays,prepare};
use chrono::{DateTime,Utc,Duration,Timelike};
use rayon::prelude::*;
use serde_json::{Value,json};
use std::{fs,path::{Path,PathBuf},time::Instant};
use netcdf_writer::{Schema,NcFormat,NcType,AttrValue,NcWriter,VarData};
fn coarsen(g:&grid::Grid,f:usize,path:&Path)->Result<Value>{
    if f==0{return Err("coarsening must be positive".into());}
    let(nx,ny)=(g.nx.div_ceil(f),g.ny.div_ceil(f));
    if nx<2||ny<2{return Err("coarsening leaves fewer than two wave cells on an axis".into());}
    let values:Vec<[f32;5]>=(0..nx*ny).into_par_iter().map(|k|{
        let(x0,y0)=((k%nx)*f,(k/nx)*f);let reference=g.lon[y0*g.nx+x0] as f64;
        let(mut w,mut lat,mut lon,mut ca,mut sa)=(0.,0.,0.,0.,0.);
        for y in y0..(y0+f).min(g.ny){for x in x0..(x0+f).min(g.nx){let i=y*g.nx+x;let a=g.area[i] as f64;
            w+=a;lat+=a*g.lat[i] as f64;lon+=a*(reference+(g.lon[i] as f64-reference+180.).rem_euclid(360.)-180.);
            ca+=a*g.cos[i] as f64;sa+=a*g.sin[i] as f64;
        }}let norm=ca.hypot(sa);
        [(lat/w) as f32,(lon/w) as f32,(ca/norm) as f32,(sa/norm) as f32,((g.dx*g.dy*(f*f) as f64)/w).sqrt() as f32]
    }).collect();
    let mut schema=Schema::new(NcFormat::Offset64);let y=schema.def_dim("y",ny,false)?;let x=schema.def_dim("x",nx,false)?;
    schema.put_global_attr("Conventions",AttrValue::Text("CF-1.10".into()))?;
    let mut vars=Vec::new();for name in ["latitude","longitude","cosalpha","sinalpha","map_factor"]{vars.push(schema.def_var(name,NcType::Float,&[y,x])?);}
    let mut writer=NcWriter::create(path,schema)?;
    for (j,id) in vars.into_iter().enumerate(){let plane:Vec<f32>=values.par_iter().map(|v|v[j]).collect();writer.write_var(id,VarData::F32(&plane))?;}
    writer.finish()?;
    Ok(json!({"kind":"curvilinear","file":path,"latitude":"latitude","longitude":"longitude","cosalpha":"cosalpha","sinalpha":"sinalpha","map_factor":"map_factor","dx":g.dx*f as f64,"dy":g.dy*f as f64}))
}

fn static_input(r:&Value,catalog:&Value,role:&str,start:&str,inputs:&Path)->Result<PathBuf>{
    if let Some(path)=r["shared_inputs"][role].as_str(){
        let path=PathBuf::from(path);if !path.is_file(){return Err(format!("shared {role} input is missing: {}",path.display()).into());}return Ok(path);
    }
    let row=&catalog[role];let folder=inputs.join(role);fs::create_dir_all(&folder)?;
    let receipt=stage::run(&json!({"start":start,"end":start,"stations":[],"sources":[row]}),&folder)?;
    if receipt["status"]!="complete"{return Err(format!("{role} acquisition incomplete").into());}
    Ok(folder.join(text(row,"path")?))
}

fn station_rows(path:&Path,catalog:&Value,selection:&Value,g:&grid::Grid,depth:&[f32])->Result<Vec<Value>>{
    let table=&catalog["station_table"];
    let column=|key:&str|->Result<usize>{Ok(table[key].as_u64().ok_or("station table column missing")? as usize)};
    let(id_col,coord_col)=(column("id_column")?,column("coordinate_column")?);
    let(lat_token,lat_sign,lon_token,lon_sign)=(column("latitude_token")?,column("latitude_hemisphere_token")?,column("longitude_token")?,column("longitude_hemisphere_token")?);
    let delimiter=text(table,"delimiter")?;let raw=fs::read_to_string(path)?;
    let wanted=selection.as_array();
    let limit=catalog["station_max_distance_km"].as_f64().ok_or("station distance limit missing")?;
    let lat_min=g.lat.par_iter().copied().reduce(||f32::INFINITY,f32::min);let lat_max=g.lat.par_iter().copied().reduce(||f32::NEG_INFINITY,f32::max);
    let mut longitudes:Vec<f32>=g.lon.par_iter().map(|v|v.rem_euclid(360.)).collect();longitudes.par_sort_unstable_by(f32::total_cmp);
    let n=longitudes.len();let gap=(0..n).max_by(|&a,&b|{let da=(if a+1<n{longitudes[a+1]}else{longitudes[0]+360.})-longitudes[a];let db=(if b+1<n{longitudes[b+1]}else{longitudes[0]+360.})-longitudes[b];da.total_cmp(&db)}).ok_or("empty grid")?;
    let west=longitudes[(gap+1)%n];let span=(longitudes[gap]-west).rem_euclid(360.);
    let rows:Vec<Value>=raw.par_lines().filter_map(|line|{
        let cols:Vec<_>=line.split(delimiter).collect();let id=cols.get(id_col)?.trim();
        if id.is_empty()||id.starts_with('#')||wanted.is_some_and(|w|!w.iter().any(|v|v==id)){return None;}
        let p:Vec<_>=cols.get(coord_col)?.split_whitespace().collect();
        let lat=p.get(lat_token)?.parse::<f32>().ok()?*if *p.get(lat_sign)?=="S"{-1.}else{1.};
        let lon=p.get(lon_token)?.parse::<f32>().ok()?*if *p.get(lon_sign)?=="W"{-1.}else{1.};
        if lat<lat_min||lat>lat_max||(lon-west).rem_euclid(360.)>span+1e-4{return None;}
        let closest=(0..g.len()).filter(|&i|depth[i]>0.).map(|i|grid::distance_km(lat,lon,g.lat[i],g.lon[i])).fold(f64::INFINITY,f64::min);
        if closest>limit{return None;}
        Some(json!({"id":id,"lat":lat,"lon":lon,"coordinate_source":"NDBC station registry at acquisition date; historical location requires verification"}))
    }).collect();
    if let Some(wanted)=wanted{for id in wanted{if !rows.iter().any(|r|r["id"]==*id){return Err(format!("requested station {id} has no ocean point in the wave domain").into());}}}
    Ok(rows)
}

pub fn run(r:&Value)->Result<Value>{
    let wall=Instant::now();let out=Path::new(text(r,"output")?);fs::create_dir_all(out)?;
    let mut catalog:Value=if let Some(p)=r["source_catalog"].as_str(){serde_json::from_slice(&fs::read(p)?)?}else{serde_json::from_str(include_str!("../../../woof/ocean/waves/data/public_sources.v1.json"))?};
    if let Some(limit)=r["station_max_distance_km"].as_f64(){if !limit.is_finite()||limit<=0.{return Err("station_max_distance_km must be positive".into());}catalog["station_max_distance_km"]=limit.into();}
    let start=DateTime::parse_from_rfc3339(text(r,"start")?)?.with_timezone(&Utc);
    let end=DateTime::parse_from_rfc3339(text(r,"end")?)?.with_timezone(&Utc);
    if start.minute()!=0||start.second()!=0||end<=start{return Err("wave initial time must be on the hourly source grid and end must follow start".into());}
    let cycle_hours=catalog["cycle_hours"].as_u64().ok_or("cycle_hours missing")? as u32;
    if cycle_hours==0||24%cycle_hours!=0{return Err("invalid source cycle hours".into());}
    let cycle=start.with_hour(start.hour()/cycle_hours*cycle_hours).ok_or("invalid cycle")?;
    let boundary_hours=catalog["boundary_hours"].as_i64().ok_or("boundary_hours missing")?;
    if boundary_hours!=1{return Err("source staging currently requires hourly boundary rows".into());}
    let finish=start+Duration::hours(((end-start).num_seconds()+3599)/3600);
    let source_grid=grid::Grid::read(&r["grid"])?;
    let factor=r["coarsening"].as_u64().unwrap_or(1) as usize;
    let coarse=if factor==1{r["grid"].clone()}else{if source_grid.kind!="curvilinear"{return Err("coarsening applies to atmospheric curvilinear grids only".into());}coarsen(&source_grid,factor,&out.join("wave-grid.nc"))?};
    let default_inputs=out.join("inputs");let inputs=r["input_cache"].as_str().map(Path::new).unwrap_or(&default_inputs);fs::create_dir_all(inputs)?;
    let acquisition_start=Instant::now();
    let bathy=if let Some(p)=r["bathymetry"]["path"].as_str(){PathBuf::from(p)}else{static_input(r,&catalog,"bathymetry",&start.to_rfc3339(),inputs)?};
    let stations=static_input(r,&catalog,"stations",&start.to_rfc3339(),inputs)?;
    let roles=catalog["forecast"].as_object().ok_or("forecast role table missing")?;
    if !roles.contains_key("wave")||!roles.contains_key("wind"){return Err("forecast table requires wave and wind roles".into());}
    let acquisition_request=json!({"start":start.to_rfc3339(),"end":finish.to_rfc3339(),"forecast_cycle":cycle.to_rfc3339(),"stations":[],"sources":roles.values().collect::<Vec<_>>()});
    let acquired=stage::run(&acquisition_request,&inputs.join("forecast"))?;
    fs::write(out.join("acquisition.json"),serde_json::to_vec_pretty(&acquired)?)?;
    if acquired["status"]!="complete"{return Err("wave forecast input acquisition incomplete".into());}
    let mut frames=Vec::new();let mut valid=start;
    while valid<=finish{let lead=u32::try_from((valid-cycle).num_hours())?;let mut row=json!({"valid":valid.to_rfc3339()});
        for (role,source) in roles{row[role]=inputs.join("forecast").join(stage::replace(text(source,"path")?,cycle,"",lead)).to_string_lossy().as_ref().into();}
        frames.push(row);valid+=Duration::hours(boundary_hours);
    }
    let acquisition_seconds=acquisition_start.elapsed().as_secs_f64();
    let selection_start=Instant::now();
    let bathymetry=if r["bathymetry"].is_object(){r["bathymetry"].clone()}else{json!({"path":bathy,"sha256":arrays::sha(&bathy)?,"licence":catalog["bathymetry"]["licence"]})};
    let grid=grid::Grid::read(&coarse)?;let depth=grid.depth(&bathymetry)?;
    let station_rows=station_rows(&stations,&catalog,&r["stations"],&grid,&depth)?;
    let fields=if let Some(p)=r["field_table"].as_str(){PathBuf::from(p)}else{let p=out.join("field-table.json");fs::write(&p,include_bytes!("../examples/fields.json"))?;p};
    let prepared=out.to_path_buf();
    let selection_seconds=selection_start.elapsed().as_secs_f64();
    let mut request=json!({"operation":"prepare","output":prepared,"grid":coarse,"bathymetry":bathymetry,
        "frames":frames,"stations":station_rows,"field_table":fields,"coast_fill_cells":r["coast_fill_cells"].as_u64().unwrap_or(2),"station_max_distance_km":catalog["station_max_distance_km"],
        "forecast_cycle":cycle.to_rfc3339(),"source_grid_sha256":if let Some(p)=r["grid"]["file"].as_str(){Some(arrays::sha(Path::new(p))?)}else{None},"coarsening":factor});
    request["threads"]=r["threads"].clone();request["automatic_source_request"]=r.clone();
    fs::write(out.join("prepare-request.json"),serde_json::to_vec_pretty(&request)?)?;
    let manifest=prepare::run(&request,&prepared)?;
    Ok(json!({"schema":"gpuwm.waves.coupled-preparation/v1","status":"complete","prepared":prepared,
        "forecast_cycle":cycle.to_rfc3339(),"initial_valid_time":manifest["initial_valid_time"],"wet_cells":manifest["wet_cells"],
        "stages":{"acquisition_seconds":acquisition_seconds,"station_selection_seconds":selection_seconds,"preparation":manifest["stages"]},"acquisition":out.join("acquisition.json"),"request":r,"coarsening":factor,"source_grid_sha256":request["source_grid_sha256"],"rayon_threads":rayon::current_num_threads(),"wall_seconds":wall.elapsed().as_secs_f64()}))
}
