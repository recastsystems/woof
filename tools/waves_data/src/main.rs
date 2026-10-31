// Original Apache-2.0. Public-data transport and numerical work live in Rust.
mod stage;
mod automatic;
mod grid;
mod prepare;
mod source;
mod arrays;
mod observations;
mod spectra;
mod catalog;
mod verify;
mod products;
mod wind;
mod audit;
use serde_json::Value;
use std::{error::Error, fs, path::Path, time::Instant};
pub type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;
pub const REV:&str=env!("GPUWM_WAVES_DATA_REV");
pub fn text<'a>(v: &'a Value, key: &str) -> Result<&'a str> {
    v[key].as_str().ok_or_else(|| format!("missing string {key}").into())
}
fn main() -> Result<()> {
    let start = Instant::now();
    let started=chrono::Utc::now().to_rfc3339();
    if std::env::args().nth(1).as_deref()==Some("--version"){println!("rw_waves_data gpuwm.waves.prepared/v1 GPUWM_BRIDGE_SOURCE_REV={REV}");return Ok(());}
    let request = std::env::args().nth(1).ok_or("rw_waves_data REQUEST.json")?;
    let mut r: Value = serde_json::from_slice(&fs::read(&request)?)?;
    if let Some(source_file) = r["sources_file"].as_str() {
        let path = Path::new(&request).parent().unwrap_or(Path::new(".")).join(source_file);
        r["sources"] = serde_json::from_slice(&fs::read(path)?)?;
    }
    if r["operation"]=="prepare-from" {
        let source=Path::new(text(&r,"source_prepared")?);
        let manifest:Value=serde_json::from_slice(&fs::read(source.join("manifest.json"))?)?;
        if manifest["schema"]!="gpuwm.waves.prepared/v1"||manifest["status"]!="complete"{return Err("source prepared manifest is incomplete".into());}
        let mut next=manifest["request"].clone();
        for key in ["output","grid","threads","station_max_distance_km","coast_fill_cells","field_table","bathymetry"]{if !r[key].is_null(){next[key]=r[key].clone();}}
        if let Some(wanted)=r["stations"].as_array(){
            if wanted.iter().all(Value::is_string){
                let source_rows=next["stations"].as_array().ok_or("source station recipes missing")?;
                let chosen:Vec<Value>=source_rows.iter().filter(|row|wanted.contains(&row["id"])).cloned().collect();
                if wanted.iter().any(|id|!chosen.iter().any(|row|row["id"]==*id)){return Err("requested station absent from staged recipe".into());}
                next["stations"]=chosen.into();
            }else{return Err("station selection must contain IDs".into());}
        }else if !r["stations"].is_null()&&r["stations"]!="auto"{return Err("stations must be auto or IDs".into());}
        if r["coarsening"].as_u64().is_some_and(|n|n!=1){return Err("coarsening requires automatic preparation, not a staged recipe".into());}
        if let (Some(start),Some(end))=(r["start"].as_str(),r["end"].as_str()) {
            let a=chrono::DateTime::parse_from_rfc3339(start)?.timestamp();let b=chrono::DateTime::parse_from_rfc3339(end)?.timestamp();
            let frames=next["frames"].as_array().ok_or("source frame recipes missing")?;
            let chosen:Vec<Value>=frames.iter().filter(|f|f["valid"].as_str().and_then(|t|chrono::DateTime::parse_from_rfc3339(t).ok()).is_some_and(|t|t.timestamp()>=a&&t.timestamp()<=b)).cloned().collect();
            if chosen.len()<2||chrono::DateTime::parse_from_rfc3339(text(&chosen[0],"valid")?)?.timestamp()!=a||chrono::DateTime::parse_from_rfc3339(text(chosen.last().unwrap(),"valid")?)?.timestamp()!=b{return Err("source frames do not bracket exact requested start/end".into());}next["frames"]=chosen.into();
        }
        next["operation"]="prepare".into();next["source_prepared"]=r["source_prepared"].clone();r=next;
    }
    let threads = r["threads"].as_u64().unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| n.get() as u64));
    if threads == 0 || threads > 256 { return Err("threads must be 1..256".into()); }
    rayon::ThreadPoolBuilder::new().num_threads(threads as usize).build_global()?;
    let out = Path::new(text(&r,"output")?);
    fs::create_dir_all(out)?;
    let mut receipt = match text(&r,"operation")? {
        "stage" => stage::run(&r,out)?,
        "prepare" => if r["frames"].is_null(){automatic::run(&r)?}else{prepare::run(&r,out)?},
        "catalog" => catalog::run(&r,out)?,
        "inspect-spectrum" => spectra::inspect(&r,out)?,
        "inspect-source" => source::inspect(&r,out)?,
        "verify" => verify::run(&r,out)?,
        "observations" => observations::export(&r,out)?,
        "products" => products::run(&r,out)?,
        "product-scales" => products::share_scales(&r,out)?,
        "export-point-grid" => grid::export_points(&r,out)?,
        "forcing" => wind::forcing(&r,out)?,
        "audit" => audit::run(&r,out)?,
        _ => return Err("unknown operation".into()),
    };
    receipt["wall_seconds"] = start.elapsed().as_secs_f64().into();
    receipt["started_utc"]=started.into();
    receipt["finished_utc"]=chrono::Utc::now().to_rfc3339().into();
    receipt["tool_source_revision"]=REV.into();
    receipt["rayon_threads"] = threads.into();
    fs::write(out.join("receipt.json"),serde_json::to_vec_pretty(&receipt)?)?;
    println!("{}", serde_json::json!({"status":receipt["status"],"wall_seconds":receipt["wall_seconds"],"output":out}));
    if receipt["status"] != "complete" { std::process::exit(2); }
    Ok(())
}
