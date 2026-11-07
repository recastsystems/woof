// SPDX-License-Identifier: Apache-2.0
// Independently implemented from NCEP's TCVITALS record specification:
// https://www.emc.ncep.noaa.gov/emc/pages/infrastructure/obs-data/tcvitals_description.php
// Radii are km (the superscript 2 in the HTML is a footnote, not km squared).
use chrono::{Datelike, NaiveDateTime};
use rayon::prelude::*;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, env, fs, io::{self, Write}, time::Instant};

type Result<T> = std::result::Result<T, String>;

#[derive(Clone, Debug, Serialize, PartialEq)]
struct Vitals {
    organization: String,
    storm_id: String,
    basin: String,
    atcf_id: Option<String>,
    name: String,
    valid_time: String,
    latitude_deg: f64,
    longitude_deg: f64,
    motion_direction_deg: Option<f64>,
    motion_speed_m_s: Option<f64>,
    minimum_pressure_pa: Option<f64>,
    environmental_pressure_pa: Option<f64>,
    outer_closed_isobar_radius_m: Option<f64>,
    maximum_wind_m_s: Option<f64>,
    radius_maximum_wind_m: Option<f64>,
    r34_m: [Option<f64>; 4],
    r50_m: [Option<f64>; 4],
    r64_m: [Option<f64>; 4],
    depth: Option<String>,
    storm_type: Option<String>,
    first_occurrence: bool,
    source_flags: BTreeMap<String, String>,
}

// One-based inclusive columns, as in the published format.
fn field(s: &str, a: usize, b: usize) -> &str {
    s.get(a - 1..b).unwrap_or("").trim()
}
fn optional(s: &str, a: usize, b: usize, scale: f64) -> Result<Option<f64>> {
    let x = field(s, a, b);
    if x.is_empty() { return Ok(None); }
    let n: i32 = x.parse().map_err(|_| format!("invalid number at columns {a}-{b}: {x:?}"))?;
    // NCEP uses -9, -99 and -999 according to field width. Never turn them into zero.
    Ok(if n < 0 { None } else { Some(f64::from(n) * scale) })
}
fn position(s: &str, a: usize, b: usize, latitude: bool) -> Result<f64> {
    let x = field(s, a, b);
    if x.len() < 2 { return Err(format!("missing coordinate at {a}-{b}")); }
    let (n, hemi) = x.split_at(x.len() - 1);
    let n: f64 = n.trim().parse().map_err(|_| "invalid coordinate".to_string())?;
    let sign = match (latitude, hemi) {
        (true, "N") | (false, "E") => 1.,
        (true, "S") | (false, "W") => -1.,
        _ => return Err("invalid coordinate hemisphere".into()),
    };
    if n < 0. || n > if latitude { 900. } else { 1800. } {
        return Err("coordinate outside Earth".into());
    }
    Ok(n * 0.1 * sign)
}
fn radii(s: &str, start: usize) -> Result<[Option<f64>; 4]> {
    let mut out = [None; 4];
    if s.len() < start { return Ok(out); }
    if s.len() < start + 18 { return Err(format!("truncated quadrant group at {start}")); }
    for (q, v) in out.iter_mut().enumerate() { *v = optional(s, start + q*5, start + q*5+3, 1000.)?; }
    Ok(out)
}
fn parse(line: &str, century: Option<i32>) -> Result<Vitals> {
    if !line.is_ascii() { return Err("TCVITALS must be ASCII".into()); }
    // Pre-1999 records have YYMMDD; never guess the century of a retrospective record.
    let expanded;
    let s = if field(line, 20, 25).chars().all(|c| c.is_ascii_digit())
        && line.as_bytes().get(25) == Some(&b' ') {
        let c = century.ok_or("two-digit date requires --century")?;
        expanded = format!("{}{:02}{}", &line[..19], c, &line[19..]);
        expanded.as_str()
    } else { line };
    if s.len() < 93 { return Err("record truncated before R34".into()); }
    let date = NaiveDateTime::parse_from_str(
        &format!("{}{}", field(s,20,27), field(s,29,32)), "%Y%m%d%H%M")
        .map_err(|e| format!("invalid report time: {e}"))?;
    let number: u8 = field(s,6,7).parse().map_err(|_| "invalid storm number")?;
    if number == 0 { return Err("storm number is zero".into()); }
    let basin = field(s,8,8);
    let atcf_basin = match basin {
        "L"=>Some("AL"), "E"=>Some("EP"), "C"=>Some("CP"), "W"=>Some("WP"),
        "A"|"B"=>Some("IO"), "S"|"P"|"U"=>Some("SH"),
        "O"|"T"=>None, _=>return Err("unknown NCEP basin".into()),
    };
    let mut flags = BTreeMap::new();
    for (key,col) in [("motion_direction",44),("motion_speed",48),("minimum_pressure",52),
        ("environmental_pressure",57),("outer_closed_isobar_radius",62),("maximum_wind",67),("depth",94)] {
        let flag = field(s,col,col);
        if !flag.is_empty() {
            if !["C","P","Z"].contains(&flag) { return Err(format!("unknown flag {flag} at {col}")); }
            flags.insert(key.to_string(), flag.to_string());
        }
    }
    let pmin = optional(s,53,56,100.)?;
    let penv = optional(s,58,61,100.)?;
    for p in [pmin,penv].into_iter().flatten() {
        if !(80000. ..=110000.).contains(&p) { return Err("pressure outside 800-1100 hPa".into()); }
    }
    if matches!((pmin,penv), (Some(a),Some(b)) if a>b) { return Err("central pressure exceeds environment".into()); }
    let wind = optional(s,68,69,1.)?;
    let direction = optional(s,45,47,1.)?;
    if direction.is_some_and(|v| v > 360.) { return Err("invalid motion direction".into()); }
    let depth = field(s,95,95);
    if !["", "S", "M", "D", "X"].contains(&depth) { return Err("invalid depth".into()); }
    let organization = field(s,1,4).to_string();
    if organization.is_empty() { return Err("missing organization".into()); }
    Ok(Vitals {
        organization, storm_id: format!("{number:02}{basin}"), basin: basin.into(),
        atcf_id: atcf_basin.map(|b|format!("{b}{number:02}{}",date.year())),
        name: field(s,10,18).into(), valid_time: date.format("%Y-%m-%dT%H:%M:%SZ").to_string(),
        latitude_deg: position(s,34,37,true)?, longitude_deg: position(s,39,43,false)?,
        motion_direction_deg: direction, motion_speed_m_s: optional(s,49,51,0.1)?,
        minimum_pressure_pa: pmin, environmental_pressure_pa: penv,
        outer_closed_isobar_radius_m: optional(s,63,66,1000.)?, maximum_wind_m_s: wind,
        radius_maximum_wind_m: optional(s,71,73,1000.)?,
        r34_m: radii(s,75)?, r50_m: radii(s,97)?, r64_m: radii(s,131)?,
        depth: if depth.is_empty()||depth=="X" {None} else {Some(depth.into())},
        storm_type: s.get(150..).map(str::trim).filter(|x|!x.is_empty()).map(str::to_string),
        first_occurrence: field(s,19,19)==":", source_flags: flags,
    })
}

#[derive(Serialize)]
struct Record { source_line: usize, source_record_sha256: String, vitals: Vitals }
fn digest(data: &[u8]) -> String { format!("{:x}", Sha256::digest(data)) }
fn run() -> Result<()> {
    let t = Instant::now();
    let args: Vec<String> = env::args().skip(1).collect();
    if args.iter().any(|x|x=="--help") || args.is_empty() {
        println!("rw_tc_vitals --input FILE [--output FILE] [--storm 09L|AL092022] [--cycle YYYYMMDDHH] [--revision error|first|last] [--century 19] [--allow-rejected]\nSI JSON; quadrant order NE,SE,SW,NW; missing=null, zero=reported zero. Exact cycle only. Conflicting revisions fail unless explicitly selected. Archive report times do not prove operational availability.");
        return Ok(());
    }
    let mut options = BTreeMap::new(); let mut allow_rejected=false; let mut i=0;
    while i<args.len() {
        if args[i]=="--allow-rejected" {allow_rejected=true; i+=1; continue;}
        if !["--input","--output","--storm","--cycle","--revision","--century"].contains(&args[i].as_str()) {
            return Err(format!("unknown argument {}",args[i]));
        }
        let value=args.get(i+1).ok_or("missing argument value")?;
        if options.insert(args[i].as_str(),value.as_str()).is_some() {return Err("repeated option".into());}
        i+=2;
    }
    let path=options.get("--input").ok_or("--input required")?;
    let bytes=fs::read(path).map_err(|e|e.to_string())?;
    let input=std::str::from_utf8(&bytes).map_err(|e|e.to_string())?;
    let century=options.get("--century").map(|s|s.parse::<i32>().map_err(|_|"invalid century")).transpose()?;
    if century.is_some_and(|c| !(1..=99).contains(&c)) {return Err("invalid century".into());}
    let cycle=options.get("--cycle").map(|s| NaiveDateTime::parse_from_str(&format!("{s}00"),"%Y%m%d%H%M")
        .map(|d|d.format("%Y-%m-%dT%H:%M:%SZ").to_string()).map_err(|e|e.to_string())).transpose()?;
    let revision=options.get("--revision").copied().unwrap_or("error");
    if !["error","first","last"].contains(&revision) {return Err("invalid revision policy".into());}
    let lines: Vec<_>=input.lines().enumerate().filter(|(_,s)|!s.trim().is_empty()).collect();
    let parsed:Vec<_>=lines.par_iter().map(|(i,s)|(*i+1,*s,parse(s,century))).collect();
    let mut rejected=Vec::new(); let mut records:Vec<Record>=Vec::new();
    let mut keys=BTreeMap::new(); let mut duplicates=Vec::new();
    for (line, raw, result) in parsed {
        let v=match result { Ok(v)=>v, Err(e)=>{rejected.push(serde_json::json!({"source_line":line,"reason":e}));continue;} };
        if cycle.as_ref().is_some_and(|c|v.valid_time!=*c) {continue;}
        if let Some(s)=options.get("--storm") {
            if !v.storm_id.eq_ignore_ascii_case(s) && !v.atcf_id.as_ref().is_some_and(|id|id.eq_ignore_ascii_case(s)) {continue;}
        }
        let key=(v.organization.clone(),v.storm_id.clone(),v.valid_time.clone());
        let record=Record {source_line:line, source_record_sha256:digest(raw.as_bytes()),vitals:v};
        if let Some(&index)=keys.get(&key) {
            let previous:&Record=&records[index];
            let equal=previous.vitals==record.vitals;
            duplicates.push(serde_json::json!({"previous_line":previous.source_line,"line":line,"identical":equal}));
            if !equal && revision=="error" {return Err(format!("conflicting revisions at lines {} and {line}; choose --revision first|last explicitly",previous.source_line));}
            if revision=="last" {records[index]=record;}
        } else {keys.insert(key,records.len());records.push(record);}
    }
    if !rejected.is_empty() && !allow_rejected {return Err(format!("{} malformed records: {}; use --allow-rejected to retain rejection receipt",rejected.len(),serde_json::to_string(&rejected).unwrap()));}
    if records.is_empty() {return Err("no records match storm and exact cycle; no nearest/future fallback".into());}
    let result=serde_json::json!({"schema":"tc-vitals/1","source":{"path":path,"sha256":digest(&bytes),"bytes":bytes.len(),"format":"ncep-tcvitals","availability":"report-time-only; archive revisions may postdate report"},"units":"SI except degrees","quadrant_order":["NE","SE","SW","NW"],"revision_policy":revision,"records":records,"duplicates":duplicates,"rejected":rejected,"timing":{"wall_seconds":t.elapsed().as_secs_f64(),"parallelism":"rayon","threads":rayon::current_num_threads()}});
    let json=serde_json::to_vec_pretty(&result).map_err(|e|e.to_string())?;
    if let Some(path)=options.get("--output") {
        // create_new prevents concurrent writers from corrupting a partial receipt.
        let part=format!("{path}.part");
        let mut f=fs::OpenOptions::new().write(true).create_new(true).open(&part).map_err(|e|e.to_string())?;
        f.write_all(&json).and_then(|_|f.write_all(b"\n")).map_err(|e|e.to_string())?;
        fs::rename(part,path).map_err(|e|e.to_string())?;
    } else {io::stdout().write_all(&json).map_err(|e|e.to_string())?;}
    Ok(())
}
fn main() { std::hint::black_box(concat!("GPUWM_BRIDGE_SOURCE_REV=",env!("GPUWM_BRIDGE_SOURCE_REV"))); if let Err(e)=run() {eprintln!("rw_tc_vitals: {e}");std::process::exit(2);} }
