use crate::Result;
use chrono::NaiveDate;
use std::{collections::BTreeMap,fs,io::Read,path::Path};
pub fn contents(path:&Path)->Result<String>{
    let bytes=fs::read(path)?;if bytes.starts_with(&[31,139]){let mut text=String::new();flate2::read::GzDecoder::new(&bytes[..]).read_to_string(&mut text)?;Ok(text)}else{Ok(String::from_utf8(bytes)?)}
}
pub fn time(words:&[&str],minute:bool)->Option<i64>{
    let mut year:i32=words.first()?.parse().ok()?;if year<100{year+=if year>=70{1900}else{2000};}
    NaiveDate::from_ymd_opt(year,words.get(1)?.parse().ok()?,words.get(2)?.parse().ok()?)?.and_hms_opt(words.get(3)?.parse().ok()?,if minute{words.get(4)?.parse().ok()?}else{0},0).map(|v|v.and_utc().timestamp())
}
#[derive(Clone)]
pub struct Obs{pub epoch:i64,pub values:BTreeMap<String,f64>}
pub fn stdmet(path:&Path,begin:i64,end:i64)->Result<Vec<Obs>>{
    let text=contents(path)?;let header=text.lines().find(|l|l.trim_start_matches('#').split_whitespace().next().is_some_and(|s|s=="YY"||s=="YYYY")).ok_or("NDBC header missing")?;
    let columns:Vec<_>=header.trim_start_matches('#').split_whitespace().collect();let minute=columns.get(4)==Some(&"mm");
    let rows=text.lines().filter(|l|!l.starts_with('#')).filter_map(|line|{
        let w:Vec<_>=line.split_whitespace().collect();let epoch=time(&w,minute)?;if epoch<begin||epoch>end{return None}
        let mut values=BTreeMap::new();for key in ["WVHT","DPD","APD","MWD"] {if let Some(k)=columns.iter().position(|s|*s==key){if let Some(v)=w.get(k).and_then(|v|v.parse::<f64>().ok()){let limit=if key=="MWD"{360.}else{90.};if v.is_finite()&&v>=0.&&(v<limit||(key=="MWD"&&v==360.))&&(!(key=="DPD"||key=="APD")||v>0.){values.insert(key.to_string(),if key=="MWD"{v.rem_euclid(360.)}else{v});}}}}
        Some(Obs{epoch,values})
    }).collect();Ok(rows)
}
pub struct SpectralObs{pub frequencies:Vec<f64>,pub rows:BTreeMap<i64,Vec<f64>>}
pub fn spectral(path:&Path,begin:i64,end:i64,scale:f64)->Result<SpectralObs>{
    let text=contents(path)?;let header=text.lines().find(|l|l.starts_with('#')&&l.contains("YY")).ok_or("spectral header missing")?;
    let c:Vec<_>=header.trim_start_matches('#').split_whitespace().collect();let minute=c.get(4)==Some(&"mm");let skip=if minute{5}else{4};
    let frequencies=c[skip..].iter().map(|v|v.parse()).collect::<std::result::Result<Vec<f64>,_>>()?;
    if frequencies.len()<2||frequencies.windows(2).any(|w|w[1]<=w[0]){return Err("spectral frequencies must increase".into());}
    let mut rows=BTreeMap::new();for line in text.lines().filter(|l|!l.starts_with('#')){let w:Vec<_>=line.split_whitespace().collect();if let Some(t)=time(&w,minute){if t<begin||t>end{continue}if w.len()!=skip+frequencies.len(){return Err("spectral row length differs from frequency header".into());}let values=w[skip..].iter().map(|v|v.parse::<f64>().ok().filter(|v|v.is_finite()&&*v>=0.&&*v<999.).map(|v|v*scale).unwrap_or(f64::NAN)).collect();rows.insert(t,values);}}
    Ok(SpectralObs{frequencies,rows})
}

/// Public sidecar for native station readers. Values retain their observation
/// times and missing flags; the forecast pairing does not resample this series.
pub fn export(r:&serde_json::Value,out:&Path)->Result<serde_json::Value>{
    use crate::{arrays,text};
    use rayon::prelude::*;
    use serde_json::{Value,json};
    let prepared=Path::new(text(r,"prepared")?);
    let manifest:Value=serde_json::from_slice(&fs::read(prepared.join("manifest.json"))?)?;
    let receipt_path=Path::new(text(r,"acquisition_receipt")?);
    let receipt:Value=serde_json::from_slice(&fs::read(receipt_path)?)?;
    if receipt["status"]!="complete"{return Err("observation acquisition incomplete".into());}
    let retrieved=text(&receipt,"finished_utc")?;
    chrono::DateTime::parse_from_rfc3339(retrieved)?;
    let files=receipt["files"].as_array().ok_or("observation source files missing")?;
    let times=manifest["times"].as_array().ok_or("prepared times missing")?;
    let begin=times.first().and_then(Value::as_i64).ok_or("initial time missing")?;
    let end=times.last().and_then(Value::as_i64).ok_or("final time missing")?;
    let rows=manifest["request"]["stations"].as_array().ok_or("station recipes missing")?;
    let stations:Vec<Result<(String,Value)>>=rows.par_iter().map(|station|{
        let path=Path::new(text(station,"stdmet")?);let digest=arrays::sha(path)?;
        let source=files.iter().find(|f|f["sha256"].as_str()==Some(&digest)&&f["error"].is_null()).ok_or("observation bytes are absent from acquisition receipt")?;
        let obs=stdmet(path,begin,end)?;
        let obs:Vec<_>=obs.iter().filter(|o|["WVHT","DPD","MWD"].iter().any(|key|o.values.contains_key(*key))).collect();
        if obs.windows(2).any(|w|w[1].epoch<=w[0].epoch){return Err("observation times must strictly increase".into());}
        let times:Vec<String>=obs.iter().map(|o|chrono::DateTime::from_timestamp(o.epoch,0).unwrap().to_rfc3339_opts(chrono::SecondsFormat::Secs,true)).collect();
        let values=|key:&str|obs.iter().map(|o|o.values.get(key).copied()).collect::<Vec<_>>();
        Ok((text(station,"id")?.to_string(),json!({"times":times,"hs":values("WVHT"),"tp":values("DPD"),"dp":values("MWD"),"source_url":source["url"],"sha256":digest})))
    }).collect();
    let stations:BTreeMap<String,Value>=stations.into_iter().collect::<Result<_>>()?;
    let document=json!({"schema":"gpuwm.waves.observations/v1","source":"NOAA NDBC historical standard meteorological observations (public domain)","retrievedAt":retrieved,"stations":stations,"acquisition_receipt":receipt_path,"prepared_manifest_sha256":arrays::sha(&prepared.join("manifest.json"))?});
    fs::write(out.join("observations.json"),serde_json::to_vec_pretty(&document)?)?;
    Ok(json!({"status":"complete","stations":stations.len(),"observations":out.join("observations.json"),"retrievedAt":retrieved}))
}
