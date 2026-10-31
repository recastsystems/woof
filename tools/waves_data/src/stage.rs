use crate::{Result,text};
use chrono::{DateTime,Utc,Duration,Timelike,Datelike};
use rayon::prelude::*;
use serde_json::{Value,json};
use sha2::{Digest,Sha256};
use std::{fs,io::{Read,Write},path::{Path,PathBuf},process::Command,time::Instant};

fn hash(path:&Path)->Result<String>{
    let mut f=fs::File::open(path)?; let mut h=Sha256::new(); let mut b=vec![0u8;1024*1024];
    loop { let n=f.read(&mut b)?; if n==0 {break} h.update(&b[..n]); }
    Ok(format!("{:x}",h.finalize()))
}
fn curl(url:&str,path:&Path,range:Option<&str>)->Result<()> {
    if !url.starts_with("https://") { return Err("only public HTTPS sources are accepted".into()); }
    let mut cmd=Command::new("curl");
    cmd.args(["--fail","--silent","--show-error","--location","--retry","2","--connect-timeout","15","--max-time","240","--output"])
        .arg(path);
    if let Some(r)=range {cmd.args(["--range",r]);}
    let result=cmd.arg(url).output()?;
    if !result.status.success() {return Err(format!("curl {}: {}",result.status,String::from_utf8_lossy(&result.stderr)).into());}
    Ok(())
}
fn remove_owned(path:&Path,out:&Path)->Result<()> {
    if !path.starts_with(out){return Err("temporary deletion escapes acquisition directory".into());}
    let bytes=fs::metadata(path)?.len();let mut log=fs::OpenOptions::new().create(true).append(true).open(out.join("deleted-files.log"))?;
    writeln!(log,"{}\t{}\t{}",Utc::now().to_rfc3339(),bytes,path.display())?;fs::remove_file(path)?;Ok(())
}
fn ranged(url:&str,path:&Path,out:&Path,size:u64,chunk:u64)->Result<()> {
    if size==0||chunk<1024||chunk>64*1024*1024{return Err("invalid range download geometry".into());}
    let ranges:Vec<_>=(0..size.div_ceil(chunk)).map(|i|(i*chunk,((i+1)*chunk).min(size)-1)).collect();
    let parts:Vec<PathBuf>=ranges.par_iter().map(|&(a,b)|->Result<PathBuf>{
        let part=path.with_extension(format!("part-{a}-{b}"));
        let range=format!("{a}-{b}");let query=if url.contains('?'){"&"}else{"?"};
        curl(&format!("{url}{query}arwen_range={range}"),&part,Some(&range))?;
        if fs::metadata(&part)?.len()!=b-a+1{return Err(format!("range {range} returned wrong byte count").into());}Ok(part)
    }).collect::<Result<_>>()?;
    let mut target=fs::File::create(path)?;for p in parts{std::io::copy(&mut fs::File::open(&p)?,&mut target)?;remove_owned(&p,out)?;}target.flush()?;Ok(())
}
#[derive(Clone)]
struct Job { url:String, relative:PathBuf, row:Value, valid:String }
pub fn replace(template:&str,t:DateTime<Utc>,station:&str,lead:u32)->String {
    template.replace("{date}",&t.format("%Y%m%d").to_string())
        .replace("{hour}",&format!("{:02}",t.hour())).replace("{year}",&t.year().to_string())
        .replace("{lead}",&format!("{lead:03}")).replace("{station}",station)
}
fn fetch(job:&Job,out:&Path)->Value {
    let start=Instant::now();
    let run=||->Result<Value>{
        if job.relative.is_absolute() || job.relative.components().any(|c|matches!(c,std::path::Component::ParentDir)) {return Err("source path escapes output".into());}
        let p=out.join(&job.relative);fs::create_dir_all(p.parent().ok_or("no parent")?)?;
        let receipt=p.with_extension(format!("{}.receipt.json",p.extension().and_then(|s|s.to_str()).unwrap_or("data")));
        if receipt.exists() && p.exists() {
            let mut previous:Value=serde_json::from_slice(&fs::read(&receipt)?)?;
            if previous["url"]==job.url && previous["sha256"].as_str()==Some(&hash(&p)?) {previous["reused"]=true.into();return Ok(previous);}
            return Err("existing artifact differs from its receipt; refusing overwrite".into());
        }
        let partial=p.with_extension("download-partial");
        let mut ranges=Vec::new();
        if let Some(selectors)=job.row["grib_select"].as_array() {
            let idx=p.with_extension("idx");curl(&format!("{}.idx",job.url),&idx,None)?;
            let lines=fs::read_to_string(&idx)?;let entries:Vec<_>=lines.lines().collect();
            let mut selected=Vec::new();
            for selector in selectors {let s=selector.as_str().ok_or("selector must be text")?;
                let found:Vec<_>=entries.iter().enumerate().filter(|(_,l)|l.contains(s)).collect();
                if found.len()!=1 {return Err(format!("selector {s} matched {} records",found.len()).into());}
                selected.push(found[0].0);
            }
            selected.sort_unstable(); selected.dedup(); let mut dest=fs::File::create(&partial)?;
            for i in selected {
                let offset=|j:usize|->Result<u64>{Ok(entries[j].split(':').nth(1).ok_or("bad GRIB index")?.parse()?)};
                let begin=offset(i)?;let end=if i+1<entries.len(){Some(offset(i+1)?-1)}else{None};
                let range=format!("{begin}-{}",end.map(|n|n.to_string()).unwrap_or_default());
                let chunk=p.with_extension("range-partial");curl(&job.url,&chunk,Some(&range))?;
                let b=fs::read(&chunk)?;
                if !b.starts_with(b"GRIB") || !b.ends_with(b"7777") || end.is_some_and(|e|b.len() as u64!=e-begin+1) {return Err("range response is not the selected complete GRIB message".into());}
                dest.write_all(&b)?;remove_owned(&chunk,out)?;ranges.push(range);
            }
        }else if let Some(chunk)=job.row["range_chunk_bytes"].as_u64(){
            ranged(&job.url,&partial,out,job.row["expected_bytes"].as_u64().ok_or("range acquisition needs authoritative expected_bytes")?,chunk)?;
        }else{curl(&job.url,&partial,None)?;}
        let bytes=fs::metadata(&partial)?.len();if bytes==0{return Err("empty download".into());}
        if job.row["expected_bytes"].as_u64().is_some_and(|n|n!=bytes){return Err("download differs from authoritative source byte length".into());}
        if job.row["format"]=="gzip" { let mut decoder=flate2::read::GzDecoder::new(fs::File::open(&partial)?);std::io::copy(&mut decoder,&mut std::io::sink())?; }
        let digest=hash(&partial)?;fs::rename(&partial,&p)?;
        let row=json!({"url":job.url,"path":job.relative,"valid":job.valid,"bytes":bytes,"sha256":digest,"licence":job.row["licence"],"attribution":job.row["attribution"],"ranges":ranges,"wall_seconds":start.elapsed().as_secs_f64(),"reused":false});
        fs::write(&receipt,serde_json::to_vec_pretty(&row)?)?;Ok(row)
    };
    match run(){Ok(v)=>v,Err(e)=>json!({"url":job.url,"path":job.relative,"error":e.to_string(),"optional":job.row["optional"].as_bool().unwrap_or(false),"wall_seconds":start.elapsed().as_secs_f64()})}
}
pub fn run(r:&Value,out:&Path)->Result<Value>{
    let begin=DateTime::parse_from_rfc3339(text(r,"start")?)?.with_timezone(&Utc);
    let end=DateTime::parse_from_rfc3339(text(r,"end")?)?.with_timezone(&Utc);
    if begin>end || (end-begin).num_days()>31 {return Err("staging window must be ordered and <=31 days".into());}
    let stations=r["stations"].as_array().ok_or("stations required")?;
    let fixed_cycle=r["forecast_cycle"].as_str().map(|s|DateTime::parse_from_rfc3339(s).map(|t|t.with_timezone(&Utc))).transpose()?;
    let mut jobs=Vec::new();
    for row in r["sources"].as_array().ok_or("sources required")? {
        let kind=text(row,"cadence")?;
        let times=match kind {"hourly"|"six_hourly_station"=> {let step=if kind=="hourly"{1}else{6}; let mut t=begin;let mut ts=Vec::new();while t<=end {ts.push(t);t+=Duration::hours(step);}ts},"station_year"|"static"=>vec![begin],_=>return Err("unknown cadence".into())};
        for valid in times {let cycle=if kind=="hourly"{fixed_cycle.unwrap_or(valid.with_hour(valid.hour()/6*6).ok_or("bad hour")?)}else{valid.with_hour(valid.hour()/6*6).ok_or("bad hour")?};
            let lead_seconds=(valid-cycle).num_seconds();
            if lead_seconds<0 || lead_seconds%3600!=0{return Err("forecast valid time must be a nonnegative whole-hour lead from its cycle".into());}
            let lead=u32::try_from(lead_seconds/3600)?;
            let ids:Vec<&str>=if kind=="station_year"||kind=="six_hourly_station" {stations.iter().map(|s|s.as_str().ok_or("bad station")).collect::<std::result::Result<_,_>>()?}else{vec![""]};
            for id in ids {jobs.push(Job{url:replace(text(row,"url")?,cycle,id,lead),relative:PathBuf::from(replace(text(row,"path")?,cycle,id,lead)),row:row.clone(),valid:valid.to_rfc3339()});}
        }
    }
    let results:Vec<Value>=jobs.par_iter().map(|job|{let v=fetch(job,out);println!("{}",json!({"path":v["path"],"bytes":v["bytes"],"error":v["error"]}));v}).collect();
    let failed=results.iter().filter(|v|!v["error"].is_null() && v["optional"]!=true).count();
    let unavailable=results.iter().filter(|v|!v["error"].is_null()).count();
    let successful=results.len()-unavailable;
    let receipt=json!({"schema":"gpuwm.waves.acquisition/v1","status":if failed==0{"complete"}else{"incomplete"},"request":r,"successful_files":successful,"unavailable_files":unavailable,"required_failures":failed,"files":results});
    // Acquisition completion is not scientific staging: coverage and preparation must pass first.
    fs::write(out.join("ACQUIRED.json"),serde_json::to_vec_pretty(&receipt)?)?;
    Ok(receipt)
}
