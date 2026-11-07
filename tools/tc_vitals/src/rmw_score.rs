// SPDX-License-Identifier: Apache-2.0
// ATCF MRD, column 20, nautical miles. Independently implemented from:
// https://science.nrlmry.navy.mil/atcf/docs/database/new/abrdeck.html
// This supplements the shared track/intensity/radii verifier with RMW only.
use chrono::{Datelike, Duration, NaiveDateTime};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, env, fs};

type Result<T> = std::result::Result<T, String>;
fn timestamp(s: &str) -> Result<NaiveDateTime> {
    NaiveDateTime::parse_from_str(&format!("{s}00"), "%Y%m%d%H%M").map_err(|e| e.to_string())
}
fn truth(text: &str, storm: &str) -> Result<BTreeMap<NaiveDateTime, Option<f64>>> {
    if storm.len()!=8 || !storm.is_ascii() || storm[2..].parse::<u32>().is_err() {
        return Err("storm must be a basin, two-digit number and four-digit year".into());
    }
    let mut rows: BTreeMap<NaiveDateTime, Option<f64>> = BTreeMap::new();
    for (i,line) in text.lines().enumerate() {
        let f: Vec<_> = line.split(',').map(str::trim).collect();
        if f.len()<6 || f[0]!=&storm[..2] || f[1]!=&storm[2..4] || f[4]!="BEST" {continue;}
        if f.len()<20 || f[5]!="0" {return Err(format!("invalid BEST record at line {}",i+1));}
        let minute = if f[3].is_empty() {0} else {f[3].parse::<i64>().map_err(|_|"invalid BEST minutes")?};
        if !(0..60).contains(&minute) {return Err("BEST minutes outside 0..59".into());}
        let time=timestamp(f[2])?+Duration::minutes(minute);
        let rmw = if f[19].is_empty() {None} else {
            let x=f[19].parse::<f64>().map_err(|_|format!("invalid MRD on line {}",i+1))?;
            if !x.is_finite() || !(0.0..=999.0).contains(&x) {return Err("MRD outside documented range".into());}
            // Zero is physically unresolved: retain as missing, never a zero-radius truth.
            if x==0.0 {None} else {Some(x*1852.0)}
        };
        if let Some(prior)=rows.get(&time) {
            if prior.is_some() && rmw.is_some() && *prior!=rmw {return Err(format!("conflicting MRD records at {time}"));}
        }
        let old=rows.entry(time).or_insert(None);
        if rmw.is_some() {*old=rmw;}
    }
    if rows.is_empty() {return Err("no matching BEST records".into());}
    let year=storm[4..].parse::<i32>().map_err(|_|"invalid storm year")?;
    if rows.first_key_value().unwrap().0.year()!=year || rows.last_key_value().unwrap().0.year()>year+1 {
        return Err("BEST storm dates do not match the declared annual storm file".into());
    }
    Ok(rows)
}
fn run() -> Result<()> {
    let args: Vec<_>=env::args().collect();
    if args.len()!=6 {return Err("usage: rw_tc_rmw_score DIAGNOSTICS.json BDECK.txt STORM YYYYMMDDHH OUTPUT.json".into());}
    let raw=fs::read(&args[1]).map_err(|e|e.to_string())?;
    let obs=fs::read(&args[2]).map_err(|e|e.to_string())?;
    let best=truth(std::str::from_utf8(&obs).map_err(|e|e.to_string())?,&args[3])?;
    let initial=timestamp(&args[4])?;
    let doc:Value=serde_json::from_slice(&raw).map_err(|e|e.to_string())?;
    let samples=doc["samples"].as_array().ok_or("samples must be an array")?;
    let mut rows=BTreeMap::new();let mut n=0;let mut sum=0.0;let mut absolute=0.0;
    for sample in samples {
        let elapsed=sample["elapsed_seconds"].as_f64().ok_or("missing model elapsed time")?;
        if !elapsed.is_finite() || elapsed<0.0 || elapsed>366.0*86400.0 || (elapsed-elapsed.round()).abs()>1e-6 {return Err("invalid or subsecond model time".into());}
        let seconds=elapsed.round() as i64;
        let valid=initial+Duration::seconds(seconds);
        let value = if let Some(vars)=sample["variables"].as_array() {
            let index=vars.iter().position(|v|v.as_str()==Some("radius_maximum_wind_m")).ok_or("H(x) lacks RMW")?;
            let members=sample["hofx"][index].as_array().ok_or("H(x) RMW must be a member array")?;
            if members.len()!=1 {return Err("score one declared deterministic forecast at a time".into());}
            members[0].as_f64().filter(|x|x.is_finite() && *x>0.0)
        } else {None};
        let observed=best.get(&valid).copied().flatten();
        let error=match (value,observed) {(Some(x),Some(y))=>Some((x-y)/1000.0),_=>None};
        if let Some(e)=error {n+=1;sum+=e;absolute+=e.abs();}
        let row=json!({"lead_hours":elapsed/3600.0,"valid_time":format!("{}Z",valid.format("%Y-%m-%dT%H:%M:%S")),"model_rmw_km":value.map(|x|x/1000.0),"best_rmw_km":observed.map(|x|x/1000.0),"error_km":error,"model_refusal":sample.get("diagnostic_refusal"),"truth_status":if best.contains_key(&valid) {if observed.is_some(){"available"}else{"blank or zero MRD"}}else{"no exact-time BEST record"}});
        if rows.insert(seconds,row).is_some() {return Err("duplicate forecast valid time".into());}
    }
    let report=json!({"schema":"gpuwm-tc-rmw-score/1","storm":args[3],"initial_time":args[4],"diagnostics_sha256":format!("{:x}",Sha256::digest(&raw)),"bdeck_sha256":format!("{:x}",Sha256::digest(&obs)),"pairs":n,"bias_km":if n>0 {Some(sum/n as f64)}else{None},"mae_km":if n>0{Some(absolute/n as f64)}else{None},"rows":rows.into_values().collect::<Vec<_>>(),"method":"Exact-time post-season BEST MRD; duplicate isotachs collapsed, conflicts refused; no temporal interpolation or replacement of missing RMW","limitation":"Point-maximum grid RMW and operational best-track MRD have finite-resolution and sampling uncertainty; this is not an azimuthal-mean RMW comparison"});
    fs::write(&args[5],serde_json::to_vec_pretty(&report).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
    Ok(())
}
fn main(){std::hint::black_box(concat!("GPUWM_BRIDGE_SOURCE_REV=",env!("GPUWM_BRIDGE_SOURCE_REV")));if let Err(e)=run(){eprintln!("{e}");std::process::exit(2);}}

#[cfg(test)]
mod tests {
    use super::*;
    fn row(mrd:&str)->String {format!("AL, 09, 2022092600, , BEST, 0, 168N, 809W, 50, 991, TS, 34, NEQ, 60, 60, 0, 30, 1007, 120, {mrd}")}
    #[test] fn duplicate_isotachs_do_not_change_truth(){let x=truth(&format!("{}\n{}",row("30"),row("30")),"AL092022").unwrap();assert_eq!(x.len(),1);assert_eq!(*x.values().next().unwrap(),Some(55560.0));}
    #[test] fn conflicting_isotachs_refuse(){assert!(truth(&format!("{}\n{}",row("30"),row("15")),"AL092022").is_err());}
}
