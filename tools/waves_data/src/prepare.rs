use crate::{Result,text,arrays,grid::{Grid,distance_km},source};
use chrono::DateTime;
use rayon::prelude::*;
use serde_json::{Value,json};
use std::{collections::BTreeMap,fs,path::Path,time::Instant};
use netcdf_writer::{AttrValue,NcFormat,NcType,NcWriter,Schema,VarData};

pub fn snapshot(path:&Path,grid:&Grid,epoch:f64,fields:&BTreeMap<String,Vec<f32>>,wet:&[i32])->Result<()> {
    let mut s=Schema::new(NcFormat::Offset64);let y=s.def_dim("y",grid.ny,false)?;let x=s.def_dim("x",grid.nx,false)?;let t=s.def_dim("time",1,false)?;
    s.put_global_attr("Conventions",AttrValue::Text("CF-1.10".into()))?;
    let lat=s.def_var("latitude",NcType::Float,&[y,x])?;let lon=s.def_var("longitude",NcType::Float,&[y,x])?;let time=s.def_var("time",NcType::Double,&[t])?;let mask=s.def_var("wet",NcType::Int,&[y,x])?;
    s.put_var_attr(lat,"units",AttrValue::Text("degrees_north".into()))?;s.put_var_attr(lon,"units",AttrValue::Text("degrees_east".into()))?;s.put_var_attr(time,"units",AttrValue::Text("seconds since 1970-01-01 00:00:00".into()))?;
    let mut ids=BTreeMap::new();for name in fields.keys(){let id=s.def_var(name,NcType::Float,&[t,y,x])?;s.put_var_attr(id,"coordinates",AttrValue::Text("time latitude longitude".into()))?;
        let units=if name.ends_with("hs"){"m"}else if name.ends_with("tp")||name.starts_with("tm"){"s"}else if name=="dp"||name=="dir"||name.ends_with("dp")||name=="dspr"||name=="direction"{"degree"}else{"1"};
        s.put_var_attr(id,"units",AttrValue::Text(units.into()))?;if name=="dp"||name.ends_with("dp"){s.put_var_attr(id,"direction_convention",AttrValue::Text("FROM clockwise from true north".into()))?;}
        ids.insert(name.clone(),id);
    }
    let mut w=NcWriter::create(path,s)?;w.write_var(lat,VarData::F32(&grid.lat))?;w.write_var(lon,VarData::F32(&grid.lon))?;w.write_var(time,VarData::F64(&[epoch]))?;w.write_var(mask,VarData::I32(wet))?;
    for(name,id)in ids{w.write_var(id,VarData::F32(&fields[&name]))?;}w.finish()?;Ok(())
}
pub fn points(path:&Path,stations:&[Value],times:&[i64],fields:&BTreeMap<String,Vec<f32>>)->Result<()> {
    let mut s=Schema::new(NcFormat::Offset64);let st=s.def_dim("station",stations.len(),false)?;let tm=s.def_dim("time",times.len(),false)?;let chars=s.def_dim("id_strlen",16,false)?;
    s.put_global_attr("Conventions",AttrValue::Text("CF-1.10".into()))?;s.put_global_attr("featureType",AttrValue::Text("timeSeries".into()))?;
    let id=s.def_var("station_id",NcType::Char,&[st,chars])?;let t=s.def_var("time",NcType::Double,&[tm])?;let lat=s.def_var("latitude",NcType::Float,&[st])?;let lon=s.def_var("longitude",NcType::Float,&[st])?;
    s.put_var_attr(id,"cf_role",AttrValue::Text("timeseries_id".into()))?;s.put_var_attr(t,"units",AttrValue::Text("seconds since 1970-01-01 00:00:00".into()))?;
    let mut ids=BTreeMap::new();for name in fields.keys(){ids.insert(name,s.def_var(name,NcType::Float,&[st,tm])?);}
    let mut idbytes=vec![b' ';stations.len()*16];for(i,row)in stations.iter().enumerate(){let name=text(row,"id")?;if name.len()>16{return Err("station id exceeds 16 characters".into());}idbytes[i*16..i*16+name.len()].copy_from_slice(name.as_bytes());}
    let mut w=NcWriter::create(path,s)?;w.write_var(id,VarData::Char(&idbytes))?;w.write_var(t,VarData::F64(&times.iter().map(|t|*t as f64).collect::<Vec<_>>()))?;
    w.write_var(lat,VarData::F32(&stations.iter().map(|s|s["lat"].as_f64().unwrap_or(f64::NAN) as f32).collect::<Vec<_>>()))?;w.write_var(lon,VarData::F32(&stations.iter().map(|s|s["lon"].as_f64().unwrap_or(f64::NAN) as f32).collect::<Vec<_>>()))?;
    for(name,id)in ids{w.write_var(id,VarData::F32(&fields[name]))?;}w.finish()?;Ok(())
}
fn partitions(fields:&BTreeMap<String,Vec<f32>>,wet:&[i32])->Result<Vec<f32>> {
    let rows:Vec<Result<[f32;16]>>=(0..wet.len()).into_par_iter().map(|i|{
        let mut p=[0.;16];if wet[i]==0{return Ok(p)}
        for(k,prefix)in ["ws","sw1","sw2","sw3"].iter().enumerate(){let hs=fields[&format!("{prefix}_hs")][i];let tp=fields[&format!("{prefix}_tp")][i];let dp=fields[&format!("{prefix}_dp")][i];
            if !hs.is_finite() && !tp.is_finite() && !dp.is_finite(){p[k*4+3]=if k==0{0.}else{1.};continue;}
            if !hs.is_finite(){return Err(format!("inconsistent missing partition {prefix} at wet cell {i}").into());}
            if hs<0.||hs>50. || (hs>0. && (!tp.is_finite()||tp<=0.||tp>60.||!dp.is_finite())) {return Err(format!("invalid active partition {prefix} at wet cell {i}: {hs},{tp},{dp}").into());}
            p[k*4]=hs;p[k*4+1]=if hs>0.{tp}else{0.};p[k*4+2]=if hs>0.{dp}else{0.};p[k*4+3]=if k==0{0.}else{1.};
        }Ok(p)
    }).collect();
    let mut result=Vec::with_capacity(wet.len()*16);for row in rows{result.extend_from_slice(&row?);}Ok(result)
}
/// Smallest inland lake (km2) and lake depth (m) that preparation makes water.
const LAKE_MIN_AREA_KM2:f64=5000.;const LAKE_MIN_DEPTH_M:f64=1.;
/// Inland lakes on a structured grid that carries the atmosphere's surface
/// rows. A lake is a connected body of model water (fraction >= 0.5) that the
/// ocean wave source does not cover and that touches no ocean wet cell. Its
/// depth is the model's lake surface height minus the bed elevation (ETOPO
/// 2022 holds lake beds, for example the Great Lakes). A lake starts from rest;
/// where it meets the domain edge that edge is open and calm (the source has no
/// lake waves), so no outside wave model is needed. Every lake
/// found is logged with where it is and whether it was made water and why.
fn lakes(grid:&Grid,bed:&[f64],hs:&[f32],ocean:&[i32])->(Vec<f32>,Vec<Value>){
    let n=grid.len();let mut depth=vec![0f32;n];let mut rows=Vec::new();
    let(Some(water),Some(height))=(&grid.water,&grid.height)else{return(depth,rows)};
    let(nx,ny)=(grid.nx,grid.ny);
    let candidate:Vec<bool>=(0..n).map(|i|water[i]>=0.5&&ocean[i]==0&&!hs[i].is_finite()).collect();
    let neighbours=|i:usize|{let(x,y)=((i%nx) as i64,(i/nx) as i64);(-1i64..=1).flat_map(move|dy|(-1i64..=1).map(move|dx|(x+dx,y+dy))).filter(move|&(a,b)|(a,b)!=(x,y)&&a>=0&&b>=0&&a<nx as i64&&b<ny as i64).map(move|(a,b)|b as usize*nx+a as usize)};
    let mut seen=vec![false;n];
    for seed in 0..n{if !candidate[seed]||seen[seed]{continue;}
        let mut cells=vec![seed];seen[seed]=true;let mut k=0;let mut touches_ocean=false;
        while k<cells.len(){let i=cells[k];k+=1;for j in neighbours(i){if ocean[j]==1{touches_ocean=true;}if candidate[j]&&!seen[j]{seen[j]=true;cells.push(j);}}}
        let area:f64=cells.iter().map(|&i|grid.area[i] as f64).sum::<f64>()/1e6;
        let centre=|v:&[f32]|cells.iter().map(|&i|grid.area[i] as f64*v[i] as f64).sum::<f64>()/(area*1e6);
        let lake_depth:Vec<f64>=cells.iter().map(|&i|height[i] as f64-bed[i]).collect();
        let with_bed=lake_depth.iter().filter(|&&d|d>=LAKE_MIN_DEPTH_M).count();
        let iced=cells.iter().filter(|&&i|grid.ice.as_ref().is_some_and(|v|v[i]>=0.5)).count();
        let surface=cells.iter().map(|&i|height[i] as f64).sum::<f64>()/cells.len() as f64;
        let mut row=json!({"centre_lat":centre(&grid.lat),"centre_lon":centre(&grid.lon),"cells":cells.len(),"area_km2":area,"surface_height_m":surface,"cells_with_bed_below_surface":with_bed});
        let reason=if touches_ocean{Some("joined to water the ocean wave source covers; prepared as before")}
            else if area<LAKE_MIN_AREA_KM2{Some("smaller than the 5000 km2 lake floor; left out")}
            else if with_bed*2<cells.len(){Some("the depth source holds no lake bed below the model's lake surface here; left out, not water")}else{None};
        if let Some(reason)=reason{row["water"]=false.into();row["reason"]=reason.into();}else{
            let mut wet=0;let mut deepest:f64=0.;let mut total=0.;
            for(c,&i)in cells.iter().enumerate(){let d=lake_depth[c];let frozen=grid.ice.as_ref().is_some_and(|v|v[i]>=0.5);
                if d>=LAKE_MIN_DEPTH_M&&!frozen{depth[i]=d as f32;wet+=1;deepest=deepest.max(d);total+=d;}}
            row["water"]=(wet>0).into();row["wet_cells"]=wet.into();row["iced_cells_left_dry"]=iced.into();
            row["shallow_cells_left_dry"]=(cells.len()-with_bed).into();row["max_depth_m"]=deepest.into();row["mean_depth_m"]=(if wet>0{total/wet as f64}else{0.}).into();
            row["start"]="from rest; a domain edge across it is open and calm".into();
        }
        println!("{}",json!({"lake":row}));rows.push(row);
    }
    (depth,rows)
}
pub fn run(r:&Value,out:&Path)->Result<Value>{
    if out.join("manifest.json").exists(){return Err("prepared manifest exists; use a fresh output or load it".into());}
    let start=Instant::now();let grid=Grid::read(&r["grid"])?;let bed=grid.bed(&r["bathymetry"])?;let depth:Vec<f32>=bed.par_iter().map(|b|(-b).max(0.) as f32).collect();let geometry_wall=start.elapsed().as_secs_f64();let table=source::fields(r)?;
    let frames=r["frames"].as_array().ok_or("frames required")?;if frames.len()<2{return Err("prepared forcing/boundary need at least two valid times".into());}
    let times:Vec<i64>=frames.iter().map(|f|DateTime::parse_from_rfc3339(text(f,"valid")?).map(|v|v.timestamp()).map_err(|e|e.to_string().into())).collect::<Result<_>>()?;
    if times.windows(2).any(|w|w[1]<=w[0]){return Err("forcing times must strictly increase".into());}
    let radius=r["coast_fill_cells"].as_u64().unwrap_or(2) as usize;if radius>8{return Err("coast fill radius must be <=8 source cells".into());}
    let first=source::grib(Path::new(text(&frames[0],"wave")?),&table["wave"],&grid,radius)?;
    let ocean:Vec<i32>=(0..grid.len()).into_par_iter().map(|i|i32::from(depth[i]>0.&&first["hs"][i].is_finite())).collect();
    let(lake_depth,lake_rows)=lakes(&grid,&bed,&first["hs"],&ocean);
    let lake:Vec<bool>=lake_depth.par_iter().map(|d|*d>0.).collect();
    let wet:Vec<i32>=(0..grid.len()).into_par_iter().map(|i|i32::from(ocean[i]==1||lake[i])).collect();
    let depth:Vec<f32>=depth.into_par_iter().enumerate().map(|(i,d)|if ocean[i]==1{d}else if lake[i]{lake_depth[i]}else{0.}).collect();
    if !wet.contains(&1){return Err("no target ocean or lake cells".into());}
    let boundary:Vec<i32>=if let Some(v)=r["grid"]["boundary_cells"].as_array(){v.iter().map(|v|v.as_i64().ok_or("bad boundary index").and_then(|v|i32::try_from(v).map_err(|_|"boundary outside int32"))).collect::<std::result::Result<_,_>>()?}else if grid.kind=="points"{
        if let Some(name)=r["grid"]["boundary_index_variable"].as_str(){arrays::read_nc(Path::new(text(&r["grid"],"file")?),name)?.iter().map(|v|*v as i32).collect()}
        else if let Some(name)=r["grid"]["boundary_connectivity"].as_str(){let v=arrays::read_nc(Path::new(text(&r["grid"],"file")?),name)?;if v.len()%2!=0{return Err("boundary connectivity must have [edge,2] shape".into());}
            let mut cells:Vec<i32>=v.par_chunks_exact(2).filter_map(|p|if p[0]==0.&&p[1]>0.{Some(p[1] as i32-1)}else if p[1]==0.&&p[0]>0.{Some(p[0] as i32-1)}else{None}).collect();
            if cells.iter().any(|&i|i<0||i as usize>=grid.len()){return Err("mesh connectivity index outside cells".into());}cells.retain(|&i|wet[i as usize]==1);cells.par_sort_unstable();cells.dedup();cells
        }else{Vec::new()}
    }else{(0..grid.len()).filter(|&i|wet[i]==1&&(i%grid.nx==0||i%grid.nx==grid.nx-1||i/grid.nx==0||i/grid.nx==grid.ny-1)).map(|i|i as i32).collect()};
    if boundary.iter().any(|&i|i<0||i as usize>=wet.len()||wet[i as usize]!=1){return Err("boundary cell must be a wet full-grid index".into());}
    if boundary.iter().collect::<std::collections::BTreeSet<_>>().len()!=boundary.len(){return Err("boundary indices must be unique".into());}
    let initial=partitions(&first,&wet)?;
    let mut excluded_stations=Vec::new();
    let mut stations=Vec::new();for s in r["stations"].as_array().ok_or("station table required")?{
        let lat=s["lat"].as_f64().ok_or("station lat missing")? as f32;let lon=s["lon"].as_f64().ok_or("station lon missing")? as f32;
        let (cell,distance)=(0..grid.len()).into_par_iter().filter(|&i|wet[i]==1).map(|i|(i,distance_km(lat,lon,grid.lat[i],grid.lon[i]))).min_by(|a,b|a.1.total_cmp(&b.1)).ok_or("no wet station cell")?;
        let max=r["station_max_distance_km"].as_f64().unwrap_or(75.);if distance>max{
            // Registry discovery uses bathymetry before the source wet mask is
            // known. Inland water can be deep but outside the wave source.
            // Skip only automatically discovered, unrequested stations. An
            // explicit station remains a strict input-coverage requirement.
            let automatic=r["automatic_source_request"].is_object();
            let requested=r["automatic_source_request"]["stations"].as_array().is_some_and(|rows|rows.iter().any(|id|*id==s["id"]));
            if automatic && !requested {
                excluded_stations.push(json!({"id":s["id"],"nearest_source_wet_cell_distance_km":distance,"limit_km":max,"reason":"outside prepared source wet coverage"}));
                continue;
            }
            return Err(format!("station {} is {distance:.1} km from wet grid, limit {max}",text(s,"id")?).into());
        }
        stations.push(json!({"id":text(s,"id")?,"lat":lat,"lon":lon,"cell":cell,"distance_km":distance,"coordinate_source":s["coordinate_source"]}));
    }
    fs::create_dir_all(out.join("reference"))?;
    let start=Instant::now();
    // Parse independent GRIB containers across all workers too. Field-only
    // parallelism left the container parser occupying one core across a sequence.
    let mut winds_u=Vec::new();let mut winds_v=Vec::new();let mut boundaries=Vec::new();let mut reference=Vec::new();let mut stage_times=Vec::new();let mut station_values:BTreeMap<String,Vec<f32>>=BTreeMap::new();
    type FrameResult=(Vec<f32>,Vec<f32>,Vec<f32>,Value,Value,BTreeMap<String,Vec<f32>>);
    let completed:Vec<FrameResult>=frames.par_iter().enumerate().map(|(ti,f)|->Result<FrameResult>{
        let frame_start=Instant::now();let mut fields=if ti==0{first.clone()}else{source::grib(Path::new(text(f,"wave")?),&table["wave"],&grid,radius)?};
        let pp=partitions(&fields,&wet)?;let mut bp=Vec::with_capacity(boundary.len()*16);for &b in &boundary{bp.extend_from_slice(&pp[b as usize*16..b as usize*16+16]);}
        let winds=crate::wind::read(f,&table["wind"],&grid)?;
        for name in ["u10_east","v10_north"]{let src=&winds[name];if src.par_iter().enumerate().any(|(i,v)|wet[i]==1&&(!v.is_finite()||v.abs()>150.)){return Err(format!("invalid wind {name} at {}",times[ti]).into());}}
        let u=winds["u10_east"].par_iter().enumerate().map(|(i,&v)|if wet[i]==1{v}else{0.}).collect::<Vec<_>>();
        let v=winds["v10_north"].par_iter().enumerate().map(|(i,&v)|if wet[i]==1{v}else{0.}).collect::<Vec<_>>();
        for values in fields.values_mut(){values.par_iter_mut().enumerate().for_each(|(i,v)|if wet[i]==0||!v.is_finite(){*v=0.});}
        let mut values_at_stations=BTreeMap::new();for(name,values)in &fields{let mut dst=Vec::new();for s in &stations{dst.push(values[s["cell"].as_u64().ok_or("station cell missing")? as usize]);}values_at_stations.insert(name.clone(),dst);}
        let leaf=format!("reference/{:04}.nc",ti);snapshot(&out.join(&leaf),&grid,times[ti] as f64,&fields,&wet)?;
        let reference=json!({"valid":f["valid"],"path":leaf,"sha256":arrays::sha(&out.join(&leaf))?});
        let timing=json!({"valid":f["valid"],"wall_seconds":frame_start.elapsed().as_secs_f64()});
        println!("{}",json!({"prepared_frame":ti,"total":frames.len(),"valid":f["valid"],"wall_seconds":frame_start.elapsed().as_secs_f64()}));
        Ok((u,v,bp,reference,timing,values_at_stations))
    }).collect::<Result<_>>()?;
    for(u,v,bp,frame,timing,values)in completed{winds_u.extend(u);winds_v.extend(v);boundaries.extend(bp);reference.push(frame);stage_times.push(timing);for(name,v)in values{station_values.entry(name).or_default().extend(v);}}
    let fields_wall=start.elapsed().as_secs_f64();let n=grid.len();let mut a=serde_json::Map::new();
    let ns=stations.len();let nt=frames.len();for v in station_values.values_mut(){*v=(0..ns*nt).into_par_iter().map(|i|v[(i%nt)*ns+i/nt]).collect();}if ns>0{points(&out.join("reference/points.nc"),&stations,&times,&station_values)?;}
    for(name,v)in [("lat",&grid.lat),("lon",&grid.lon),("depth",&depth),("area",&grid.area),("cosalpha",&grid.cos),("sinalpha",&grid.sin),("map_factor",&grid.mapfac)]{a.insert(name.into(),arrays::f32(out,name,v,&[n])?);}
    a.insert("wet".into(),arrays::i32(out,"wet",&wet)?);a.insert("boundary_cells".into(),arrays::i32(out,"boundary_cells",&boundary)?);
    a.insert("initial_partitions".into(),arrays::f32(out,"initial_partitions",&initial,&[n,4,4])?);
    a.insert("boundary_partitions".into(),arrays::f32(out,"boundary_partitions",&boundaries,&[frames.len(),boundary.len(),4,4])?);
    a.insert("u10_east".into(),arrays::f32(out,"u10_east",&winds_u,&[frames.len(),n])?);a.insert("v10_north".into(),arrays::f32(out,"v10_north",&winds_v,&[frames.len(),n])?);
    let manifest=json!({"schema":"gpuwm.waves.prepared/v1","status":"complete","grid":{"kind":grid.kind,"ncell":n,"nx":grid.nx,"ny":grid.ny,"dx":grid.dx,"dy":grid.dy,"row_order":"same as target grid; latlon south to north, west to east","boundary_index_space":"full"},"arrays":a,"initial_valid_time":times[0],"times":times,"boundary_times":times,"forcing_times":times,"stations":stations,"excluded_stations":excluded_stations,"lakes":lake_rows,"wet_cells":wet.iter().filter(|&&v|v==1).count(),"reference_frames":reference,"request":r,"parallelism":{"threads":rayon::current_num_threads(),"decode":"Rust parallel fields","regrid":"Rayon target cells"},"stages":{"geometry_seconds":geometry_wall,"fields_seconds":fields_wall,"frames":stage_times},"direction_convention":"FROM clockwise from true north","partition_kind_codes":{"windsea":0,"swell":1},"bathymetry_datum":"EGM2008","coast_fill_max_source_cells":radius});
    fs::write(out.join("manifest.json"),serde_json::to_vec_pretty(&manifest)?)?;Ok(manifest)
}
#[cfg(test)]
mod lake_tests {
    use super::*;
    // 12 x 6 cells of 30 km: a 4 x 4 lake (14,400 km2) at 183 m with a 100 m bed,
    // a 2 x 2 lake (3,600 km2), and water joined to an ocean wet cell.
    fn grid(water:Vec<f32>,ice:Option<Vec<f32>>)->Grid{let(nx,ny)=(12,6);let n=nx*ny;
        Grid{lat:(0..n).map(|i|45.+(i/nx) as f32*0.27).collect(),lon:(0..n).map(|i|-88.+(i%nx) as f32*0.38).collect(),area:vec![9.0e8;n],cos:vec![1.;n],sin:vec![0.;n],mapfac:vec![1.;n],
            nx,ny,kind:"curvilinear".into(),dx:30000.,dy:30000.,water:Some(water),height:Some(vec![183.;n]),ice}}
    fn layout()->(Vec<f32>,Vec<f64>,Vec<i32>){let(nx,n)=(12,72);let mut water=vec![0f32;n];let mut bed=vec![300f64;n];let mut ocean=vec![0;n];
        for y in 1..5{for x in 1..5{water[y*nx+x]=1.;bed[y*nx+x]=100.;}}
        for y in 1..3{for x in 6..8{water[y*nx+x]=1.;bed[y*nx+x]=100.;}}
        for y in 3..6{water[y*nx+10]=1.;bed[y*nx+10]=-50.;}ocean[5*nx+11]=1;(water,bed,ocean)}
    #[test]
    fn large_inland_lake_is_water_with_surface_minus_bed_depth(){
        let(water,bed,ocean)=layout();let g=grid(water,None);let hs=vec![f32::NAN;72];
        let(depth,rows)=lakes(&g,&bed,&hs,&ocean);
        assert_eq!(rows.len(),3);
        for y in 1..5{for x in 1..5{assert_eq!(depth[y*12+x],83.);}}
        assert!(depth.iter().enumerate().all(|(i,d)|*d==0.||(i%12>=1&&i%12<5&&i/12>=1&&i/12<5)));
        assert_eq!(rows.iter().filter(|r|r["water"]==true).count(),1);
        assert!(rows.iter().any(|r|r["reason"].as_str().is_some_and(|s|s.contains("5000 km2"))));
        assert!(rows.iter().any(|r|r["reason"].as_str().is_some_and(|s|s.contains("ocean wave source"))));
    }
    #[test]
    fn lake_without_bed_below_surface_is_named_and_left_out(){
        let(water,mut bed,ocean)=layout();for y in 1..5{for x in 1..5{bed[y*12+x]=190.;}}
        let(depth,rows)=lakes(&grid(water,None),&bed,&vec![f32::NAN;72],&ocean);
        assert!(depth.iter().all(|d|*d==0.));
        assert!(rows.iter().any(|r|r["reason"].as_str().is_some_and(|s|s.contains("no lake bed"))&&r["area_km2"].as_f64()==Some(14400.)));
    }
    #[test]
    fn iced_lake_cells_stay_dry_and_grids_without_surface_rows_are_unchanged(){
        let(water,bed,ocean)=layout();let mut ice=vec![0f32;72];ice[12+1]=1.;
        let(depth,_)=lakes(&grid(water,Some(ice)),&bed,&vec![f32::NAN;72],&ocean);
        assert_eq!(depth[12+1],0.);assert_eq!(depth[12+2],83.);
        let mut plain=grid(vec![0.;72],None);plain.water=None;plain.height=None;
        let(depth,rows)=lakes(&plain,&bed,&vec![f32::NAN;72],&ocean);assert!(rows.is_empty()&&depth.iter().all(|d|*d==0.));
    }
}
