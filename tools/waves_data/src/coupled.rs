//! Compatibility entry point for the atmospheric door, using shared preparation.
mod stage; mod grid; mod prepare; mod source; mod arrays; mod spectra;
mod observations; mod wind; mod automatic;
use serde_json::Value;
use std::{error::Error,fs,path::Path};
pub type Result<T> = std::result::Result<T,Box<dyn Error+Send+Sync>>;
pub const REV:&str=env!("GPUWM_WAVES_DATA_REV");
pub fn text<'a>(v:&'a Value,k:&str)->Result<&'a str>{v[k].as_str().ok_or_else(||format!("missing string {k}").into())}
fn main()->Result<()>{
    let path=std::env::args().nth(1).ok_or("rw_waves_coupled_prepare REQUEST.json")?;
    let request:Value=serde_json::from_slice(&fs::read(path)?)?;
    let threads=request["threads"].as_u64().unwrap_or(std::thread::available_parallelism()?.get() as u64);
    if threads==0||threads>256{return Err("threads must be 1..256".into());}
    rayon::ThreadPoolBuilder::new().num_threads(threads as usize).build_global()?;
    let started=chrono::Utc::now().to_rfc3339();
    let mut receipt=automatic::run(&request)?;
    receipt["started_utc"]=started.into();receipt["finished_utc"]=chrono::Utc::now().to_rfc3339().into();
    receipt["tool_source_revision"]=REV.into();
    fs::write(Path::new(text(&request,"output")?).join("receipt.json"),serde_json::to_vec_pretty(&receipt)?)?;
    println!("{receipt}");Ok(())
}
