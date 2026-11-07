//! Completed native histories: model-only pressure-minimum trajectory.
//! Uses the same pressure reduction as native hex maps and point products.
use rayon::prelude::*;
use rw_mpas::{history::{read_history_cells_on_mesh, Timestamp}, weights::read_mesh_coordinates};
use serde::Deserialize;
use serde_json::{json, Value};
use std::{fs, path::PathBuf, time::Instant};
type R<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;
#[derive(Deserialize)]
struct Frame { path: PathBuf, valid: String }
#[derive(Deserialize)]
struct Request {
    mesh: PathBuf, init: PathBuf, output: PathBuf, frames: Vec<Frame>,
    initial_center: [f64; 2], search_radius_km: f64, wind_radius_km: f64,
    #[serde(default = "threads")] threads: usize,
}
fn threads() -> usize { 24 }
fn distance(a: [f64;2], b: [f64;2]) -> f64 {
    let h=((a[0]-b[0]).to_radians()/2.).sin().powi(2)
        +a[0].to_radians().cos()*b[0].to_radians().cos()*((a[1]-b[1]).to_radians()/2.).sin().powi(2);
    12742.*h.clamp(0.,1.).sqrt().asin()
}
fn main() -> R<()> {
    let started=Instant::now();
    let r:Request=serde_json::from_slice(&fs::read(std::env::args().nth(1).ok_or("request JSON required")?)?)?;
    if r.frames.is_empty() || r.threads==0 || r.threads>24 || !r.search_radius_km.is_finite()
        || r.search_radius_km<=0. || !r.wind_radius_km.is_finite() || r.wind_radius_km<=0.
        || !r.initial_center.iter().all(|x|x.is_finite()) || r.initial_center[0].abs()>90. {
        return Err("invalid trajectory request".into());
    }
    for pair in r.frames.windows(2) { if pair[0].valid>=pair[1].valid { return Err("frames must have unique ordered UTC times".into()); } }
    rayon::ThreadPoolBuilder::new().num_threads(r.threads).build_global()?;
    let mesh=read_mesh_coordinates(&r.mesh,None)?;
    let cells:Vec<u32>=(0..mesh.n_cells() as u32).collect();
    let mut center=r.initial_center;let mut rows=Vec::new();
    for f in &r.frames {
        let read_start=Instant::now();
        let frame=read_history_cells_on_mesh(&f.path,Some(&r.init),&cells,
            &["PSFC","T","PB","QVAPOR","PHB","U10","V10"],Some(Timestamp::parse(&f.valid)?),Some(&mesh))?;
        let need=|key:&str| -> R<&(Vec<f32>,usize)> {
            frame.fields.get(key).ok_or_else(||format!("{} lacks {key}",f.path.display()).into())
        };
        let ps=need("PSFC")?;let theta=need("T")?;let p=need("PB")?;let q=need("QVAPOR")?;
        let z=need("PHB")?;let u=need("U10")?;let v=need("V10")?;
        for field in [ps,theta,p,q,z,u,v] {
            if field.1==0 || field.0.len()!=cells.len()*field.1 {return Err("native field extent mismatch".into());}
        }
        if z.1<2 {return Err("native interfaces require at least two levels".into());}
        let reduced:Vec<(f64,f64)>=(0..cells.len()).into_par_iter().map(|i|{
            let temperature=theta.0[i*theta.1] as f64*(p.0[i*p.1] as f64/1e5).powf(2./7.);
            let terrain=z.0[i*z.1] as f64;
            let height=(terrain+z.0[i*z.1+1] as f64)*0.5;
            let slp=rw_mpas::mslp::shuell_mslp_pa(ps.0[i*ps.1] as f64,terrain,temperature,q.0[i*q.1] as f64,height);
            let wind=(u.0[i*u.1] as f64).hypot(v.0[i*v.1] as f64);
            (slp,wind)
        }).collect();
        let position=|i:usize|[frame.latitude_degrees[i],frame.longitude_degrees[i]];
        let candidates:Vec<usize>=(0..cells.len()).into_par_iter().filter(|&i|
            reduced[i].0.is_finite() && (20000.0..120000.0).contains(&reduced[i].0)
            && distance(center,position(i))<=r.search_radius_km).collect();
        let i=*candidates.iter().min_by(|&&a,&&b|reduced[a].0.total_cmp(&reduced[b].0)).ok_or("no valid pressure within search radius")?;
        let previous=center;center=position(i);
        let wind=(0..cells.len()).into_par_iter().filter_map(|j|{
            let w=reduced[j].1;
            (w.is_finite() && w<200. && distance(center,position(j))<=r.wind_radius_km).then_some(w)
        }).reduce(||f64::NEG_INFINITY,f64::max);
        let fix=json!({"valid":f.valid,"lat":center[0],"lon":center[1],"mslp_pa":reduced[i].0,
            "max_wind10_ms":if wind.is_finite(){Some(wind)}else{None},"flags":[]});
        rows.push(json!({"path":f.path,"fix":fix,"cell":i,"search_cells":candidates.len(),
            "previous_center":previous,"displacement_km":distance(previous,center),
            "read_reduce_seconds":read_start.elapsed().as_secs_f64()}));
    }
    let fixes:Vec<Value>=rows.iter().map(|x|x["fix"].clone()).collect();
    let result=json!({"status":"complete","schema":"gpuwm.hex.wave-track/v1","request":{
        "mesh":r.mesh,"init":r.init,"initial_center":r.initial_center,"search_radius_km":r.search_radius_km,
        "wind_radius_km":r.wind_radius_km,"threads":r.threads},"frames":rows,"fixes":fixes,
        "method":"native cell pressure minimum within declared radius of preceding center; map-consistent Shuell pressure; analysis location seeds only the first search; no future observations",
        "wind_definition":"instantaneous model 10 m wind within declared center radius, not sustained best-track wind",
        "wall_seconds":started.elapsed().as_secs_f64()});
    fs::write(r.output,serde_json::to_vec_pretty(&result)?)?;
    println!("{}",json!({"status":"complete","frames":r.frames.len(),"wall_seconds":started.elapsed().as_secs_f64()}));
    Ok(())
}
