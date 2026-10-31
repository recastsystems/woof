use crate::{Result,text,arrays,observations,spectra::{self,Spectrum}};
use rayon::prelude::*;
use serde_json::{json,Value};
use std::{collections::{BTreeMap,BTreeSet},fs,path::Path};
const KEYS:[(&str,&str);6]=[("hs","WVHT"),("tp","DPD"),("tm01","APD"),("tm02","APD"),("dp","MWD"),("dir","MWD")];
pub struct Series{label:String,times:Vec<i64>,fields:BTreeMap<String,Vec<f64>>,spectra:BTreeMap<(usize,i64),Spectrum>}
fn nearest(times:&[i64],t:i64,tolerance:i64)->Option<usize>{let at=times.partition_point(|v|*v<t);[at.checked_sub(1),if at<times.len(){Some(at)}else{None}].into_iter().flatten().filter(|&i|(times[i]-t).abs()<=tolerance).min_by_key(|&i|((times[i]-t).abs(),times[i]))}
fn load(row:&Value,stations:&[Value],target_times:&[i64])->Result<Series>{
    let label=text(row,"label")?.to_string();let ns=stations.len();let mut spectral=BTreeMap::new();
    if row["kind"]=="station_ascii" {
        let root=Path::new(text(row,"case")?);let acq:Value=serde_json::from_slice(&fs::read(root.join("ACQUIRED.json"))?)?;
        let jobs:Vec<_>=acq["files"].as_array().ok_or("missing acquisition files")?.iter().filter(|f|f["error"].is_null()&&f["path"].as_str().is_some_and(|p|p.starts_with("station-spectra/"))).collect();
        let parsed:Vec<Result<Vec<Spectrum>>>=jobs.par_iter().map(|f|{let cycle=chrono::DateTime::parse_from_rfc3339(text(f,"valid")?)?.timestamp();spectra::station(&root.join(text(f,"path")?),cycle.max(target_times[0]),(cycle+5*3600).min(*target_times.last().ok_or("empty times")?))}).collect();
        for records in parsed{for s in records?{if let Some(i)=stations.iter().position(|v|v["id"]==s.id){spectral.insert((i,s.epoch),s);}}}
        let nt=target_times.len();let mut fields:BTreeMap<String,Vec<f64>>=KEYS.iter().map(|(k,_)|(k.to_string(),vec![f64::NAN;ns*nt])).collect();
        for(&(i,t),s)in &spectral{if let Ok(ti)=target_times.binary_search(&t){let m=spectra::moments(s);for(k,values)in &mut fields{values[i*nt+ti]=m[k].as_f64().unwrap_or(f64::NAN);}}}
        if let Some(p)=row["fallback_points"].as_str(){let fallback=load(&json!({"label":label,"points":p}),stations,target_times)?;for(k,v)in &mut fields{if let Some(f)=fallback.fields.get(k){for i in 0..ns{for(ti,&t)in target_times.iter().enumerate(){if !v[i*nt+ti].is_finite(){if let Some(j)=nearest(&fallback.times,t,0){v[i*nt+ti]=f[i*fallback.times.len()+j];}}}}}}}
        return Ok(Series{label,times:target_times.to_vec(),fields,spectra:spectral});
    }
    let path=Path::new(text(row,"points")?);let f=netcrust::open(path).map_err(|e|e.to_string())?;
    let times:Vec<i64>=arrays::read_nc(path,"time")?.into_iter().map(|v|v as i64).collect();if times.is_empty()||times.windows(2).any(|w|w[1]<=w[0]){return Err("point times must strictly increase".into());}
    let nt=times.len();let ids=f.read_strings("station_id").map_err(|e|format!("station_id is required to prevent pairing another buoy's data: {e}"))?;
    if ids.len()!=ns||ids.iter().zip(stations).any(|(a,b)|Some(a.trim())!=b["id"].as_str()){return Err("point station order differs from prepared station table".into());}
    let mut fields=BTreeMap::new();for(k,_)in KEYS{if let Some(v)=f.variable(k){if v.shape()!=[ns,nt]{return Err(format!("{k} must have [station,time] shape").into());}fields.insert(k.into(),arrays::read_nc(path,k)?);}}
    if let Some(spectral_var)=f.variable("efth"){let shape=spectral_var.shape();if shape.len()!=4||shape[0]!=ns||shape[1]!=nt||shape[2]<2||shape[3]<4{return Err("efth must have [station,time,frequency,direction] shape".into());}
        let frequency_name=row["frequency_coordinate"].as_str().unwrap_or("frequency");
        let direction_name=if let Some(name)=row["direction_coordinate"].as_str(){name.to_string()}else if f.variable("direction").is_some_and(|v|v.shape()==[shape[3]]){"direction".into()}else{
            spectral_var.attribute("coordinates").and_then(|a|a.as_string()).unwrap_or("").split_whitespace().find(|name|f.variable(name).is_some_and(|v|v.shape()==[shape[3]]&&v.attribute("standard_name").and_then(|a|a.as_string())==Some("sea_surface_wave_from_direction"))).ok_or("efth needs a one-dimensional FROM direction coordinate in CF coordinates or direction_coordinate metadata")?.to_string()
        };
        let freq=arrays::read_nc(path,frequency_name)?;let from=arrays::read_nc(path,&direction_name)?;let efth=arrays::read_nc(path,"efth")?;let nf=freq.len();let nd=from.len();if shape!=[ns,nt,nf,nd]{return Err("spectral coordinate lengths differ from efth dimensions".into());}
        if freq.iter().any(|v|!v.is_finite()||*v<=0.)||freq.windows(2).any(|w|w[1]<=w[0])||from.iter().any(|v|!v.is_finite())||efth.par_iter().any(|v|!v.is_finite()||*v<0.){return Err("invalid spectral coordinates or negative/nonfinite variance".into());}
        for i in 0..ns{for(ti,&t)in times.iter().enumerate(){let offset=(i*nt+ti)*nf*nd;spectral.insert((i,t),Spectrum{epoch:t,id:text(&stations[i],"id")?.into(),lat:stations[i]["lat"].as_f64().unwrap_or(f64::NAN),lon:stations[i]["lon"].as_f64().unwrap_or(f64::NAN),frequency:freq.clone(),from:from.clone(),efth:efth[offset..offset+nf*nd].to_vec()});}}
    }
    Ok(Series{label,times,fields,spectra:spectral})
}
pub fn scores(pairs:&[(f64,f64)],circular:bool)->Value{
    let n=pairs.len();if n==0{return json!({"n":0,"bias":null,"rmse":null,"scatter_index":null,"correlation":null});}
    let count=n as f64;let mo=pairs.iter().map(|p|p.0).sum::<f64>()/count;let mm=pairs.iter().map(|p|p.1).sum::<f64>()/count;
    let errors:Vec<f64>=pairs.iter().map(|(o,m)|if circular{(m-o+180.).rem_euclid(360.)-180.}else{m-o}).collect();let mean=errors.iter().sum::<f64>()/count;let rmse=(errors.iter().map(|e|e*e).sum::<f64>()/count).sqrt();
    let bias=if circular{errors.iter().map(|e|e.to_radians().sin()).sum::<f64>().atan2(errors.iter().map(|e|e.to_radians().cos()).sum::<f64>()).to_degrees()}else{mean};
    let covariance=pairs.iter().map(|(o,m)|(o-mo)*(m-mm)).sum::<f64>();let spread=(pairs.iter().map(|(o,_)|(o-mo).powi(2)).sum::<f64>()*pairs.iter().map(|(_,m)|(m-mm).powi(2)).sum::<f64>()).sqrt();
    json!({"n":n,"bias":bias,"rmse":rmse,"scatter_index":if !circular&&mo.abs()>1e-12{Some((errors.iter().map(|e|(e-mean).powi(2)).sum::<f64>()/count).sqrt()/mo.abs())}else{None},"rmse_over_observed_mean":if !circular&&mo.abs()>1e-12{Some(rmse/mo.abs())}else{None},"correlation":if !circular&&spread>1e-12{Some(covariance/spread)}else{None},"observed_mean":if circular{None}else{Some(mo)},"model_mean":if circular{None}else{Some(mm)}})
}
fn spectral_panel(station:&Value,index:usize,obs:&[observations::Obs],models:&[Series],begin:i64,end:i64)->Result<Option<Value>>{
    let paths=&station["spectral_files"];if paths["swden"].is_null(){return Ok(None)}
    let density=observations::spectral(Path::new(text(paths,"swden")?),begin,end,1.)?;
    let Some(peak)=obs.iter().filter_map(|o|o.values.get("WVHT").map(|v|(o.epoch,*v))).max_by(|a,b|a.1.total_cmp(&b.1))else{return Ok(None)};
    let times:Vec<i64>=density.rows.keys().copied().collect();
    let providers:Vec<_>=models.iter().filter(|m|m.spectra.keys().any(|(i,_)|*i==index)).collect();
    let common=times.iter().copied().filter(|t|providers.iter().all(|m|m.spectra.keys().any(|(i,mt)|*i==index&&(*mt-*t).abs()<=1800))).min_by_key(|t|((*t-peak.0).abs(),*t));
    let Some(t)=common.or_else(||nearest(&times,peak.0,3600).map(|k|times[k]))else{return Ok(None)};let freq=&density.frequencies;let energy=&density.rows[&t];let from:Vec<f64>=(0..72).map(|i|i as f64*5.).collect();
    let mut obs_mem=None;let mut obs_mean=None;let mut obs_spread=None;
    if ["swdir","swdir2","swr1","swr2"].iter().all(|k|paths[*k].is_string()){
        let mut coefficients=Vec::new();for(key,scale)in[("swdir",1.),("swdir2",1.),("swr1",0.01),("swr2",0.01)]{let data=observations::spectral(Path::new(text(paths,key)?),begin,end,scale)?;if data.frequencies!=*freq{return Err("NDBC coefficient and density frequency axes differ".into());}coefficients.push(data.rows.get(&t).cloned());}
        if coefficients.iter().all(Option::is_some){let c:Vec<_>=coefficients.into_iter().flatten().collect();let mut efth=Vec::new();for i in 0..freq.len(){let row=spectra::mem(c[2][i],c[3][i],c[0][i],c[1][i],&from);efth.extend(row.map(|r|r.into_iter().map(|v|v*energy[i]).collect()).unwrap_or_else(||vec![f64::NAN;from.len()]));}obs_mem=Some(efth);obs_mean=Some(c[0].clone());obs_spread=Some(c[2].iter().map(|r|if (0.0..=1.0).contains(r){(2.*(1.-r)).sqrt().to_degrees()}else{f64::NAN}).collect::<Vec<_>>());}
    }
    let mut model_spectra=Vec::new();for m in models{let chosen=m.spectra.iter().filter(|((i,time),_)|*i==index&&(*time-t).abs()<=1800).min_by_key(|((_,time),_)|(*time-t).abs());if let Some((_,s))=chosen{let moments=spectra::moments(s);let nd=s.from.len();let means:Vec<_>=s.efth.chunks_exact(nd).map(|row|{let c=row.iter().zip(&s.from).map(|(e,a)|e*a.to_radians().cos()).sum::<f64>();let si=row.iter().zip(&s.from).map(|(e,a)|e*a.to_radians().sin()).sum::<f64>();let energy=row.iter().sum::<f64>();(si.atan2(c).to_degrees().rem_euclid(360.),if energy>0.{(2.*(1.-c.hypot(si)/energy).max(0.)).sqrt().to_degrees()}else{f64::NAN})}).collect();model_spectra.push(json!({"label":m.label,"time":s.epoch,"frequency":s.frequency,"direction_from":s.from,"efth":s.efth,"energy":moments["energy"],"mean_direction":means.iter().map(|v|v.0).collect::<Vec<_>>(),"spread":means.iter().map(|v|v.1).collect::<Vec<_>>()}));}}
    let color_max=obs_mem.as_ref().into_iter().flatten().copied().filter(|v|v.is_finite()).chain(model_spectra.iter().flat_map(|m|m["efth"].as_array().into_iter().flatten().filter_map(Value::as_f64))).fold(0.,f64::max).max(1e-12);
    Ok(Some(json!({"station":station["id"],"time":t,"observed_hs_peak_time":peak.0,"peak_offset_seconds":t-peak.0,"selection":if common.is_some(){"common model/observation spectral time nearest observed Hs maximum; model cadence may offset the panel from that maximum"}else{"no common model spectrum; observed peak-near record only"},"frequency":freq,"energy":energy,"direction_from":from,"efth":obs_mem,"mean_direction":obs_mean,"spread":obs_spread,"models":model_spectra,"color_max":color_max})))
}
// Conservative overlap of frequency bins. First directional moments are
// averaged as energy-weighted vectors, never as scalar bearings.
fn spectral_bin(s:&Spectrum, low:f64, high:f64)->Option<[f64;3]>{
    let nd=s.from.len();let da=2.*std::f64::consts::PI/nd as f64;
    let(mut energy,mut c,mut q,mut covered)=(0.,0.,0.,0.);
    for(i,&f)in s.frequency.iter().enumerate(){
        let left=if i==0{0.5*(f*f/s.frequency[1]+f)}else{0.5*(s.frequency[i-1]+f)};
        let right=if i+1==s.frequency.len(){0.5*(f*f/s.frequency[i-1]+f)}else{0.5*(s.frequency[i+1]+f)};
        let overlap=(right.min(high)-left.max(low)).max(0.);if overlap==0.{continue}covered+=overlap;
        for(j,&a)in s.from.iter().enumerate(){let e=s.efth[i*nd+j];if !e.is_finite()||e<0.{return None}let v=e*overlap*da;energy+=v;c+=v*a.to_radians().cos();q+=v*a.to_radians().sin();}
    }
    if covered<(high-low)*(1.-1e-6){return None}
    Some([energy/(high-low),if energy>0.{q.atan2(c).to_degrees().rem_euclid(360.)}else{f64::NAN},if energy>0.{(2.*(1.-c.hypot(q)/energy).max(0.)).sqrt().to_degrees()}else{f64::NAN}])
}
fn spectral_scores(station:&Value,index:usize,models:&[Series],target:&[i64],tolerance:i64)->Result<Value>{
    let paths=&station["spectral_files"];if !paths["swden"].is_string(){return Ok(json!({"status":"no observed spectrum"}))}
    let(begin,end)=(target[0],*target.last().unwrap());let density=observations::spectral(Path::new(text(paths,"swden")?),begin,end,1.)?;
    let times:Vec<i64>=density.rows.keys().copied().collect();let freq=&density.frequencies;let nf=freq.len();
    let direction=if let Some(p)=paths["swdir"].as_str(){Some(observations::spectral(Path::new(p),begin,end,1.)?)}else{None};
    let r1=if let Some(p)=paths["swr1"].as_str(){Some(observations::spectral(Path::new(p),begin,end,0.01)?)}else{None};
    if direction.as_ref().is_some_and(|v|v.frequencies!=*freq)||r1.as_ref().is_some_and(|v|v.frequencies!=*freq){return Err("spectral score coefficient axes differ".into())}
    let providers:Vec<_>=models.iter().filter(|m|m.spectra.keys().any(|(i,_)|*i==index)).collect();
    let mut pairs=vec![vec![vec![Vec::<(f64,f64)>::new();3];nf];providers.len()];let mut used=BTreeSet::new();let mut count=0;
    for &t in target{
        let Some(oi)=nearest(&times,t,tolerance)else{continue};if used.contains(&oi){continue}let ot=times[oi];
        let spectra:Vec<_>=providers.iter().map(|m|m.spectra.iter().filter(|((i,time),_)|*i==index&&(*time-t).abs()<=tolerance).min_by_key(|((_,time),_)|((*time-t).abs(),*time)).map(|(_,s)|s)).collect();
        if spectra.is_empty()||spectra.iter().any(Option::is_none){continue}used.insert(oi);count+=1;
        let od=direction.as_ref().and_then(|v|v.rows.get(&ot));let or=r1.as_ref().and_then(|v|v.rows.get(&ot));
        for fi in 0..nf{
            let low=if fi==0{freq[fi]-(freq[1]-freq[0])/2.}else{(freq[fi-1]+freq[fi])/2.};
            let high=if fi+1==nf{freq[fi]+(freq[fi]-freq[fi-1])/2.}else{(freq[fi+1]+freq[fi])/2.};
            let pred:Vec<_>=spectra.iter().map(|s|spectral_bin(s.unwrap(),low,high)).collect();if pred.iter().any(Option::is_none){continue}
            let oe=density.rows[&ot][fi];let observation=[oe,od.map(|v|v[fi]).unwrap_or(f64::NAN),or.map(|v|if (0.0..=1.0).contains(&v[fi]){(2.*(1.-v[fi])).sqrt().to_degrees()}else{f64::NAN}).unwrap_or(f64::NAN)];
            for metric in 0..3{let o=observation[metric];if !o.is_finite()||oe<0.||(metric>0&&oe<=0.)||pred.iter().any(|p|!p.unwrap()[metric].is_finite()){continue}
                for(mi,p)in pred.iter().enumerate(){pairs[mi][fi][metric].push((o,p.unwrap()[metric]));}
            }
        }
    }
    let mut rows=Vec::new();for(mi,m)in providers.iter().enumerate(){for(fi,&f)in freq.iter().enumerate(){for(metric,key)in ["energy","mean_direction","spread"].iter().enumerate(){rows.push(json!({"model":m.label,"frequency_hz":f,"metric":key,"scores":scores(&pairs[mi][fi][metric],metric==1)}));}}}
    Ok(json!({"status":"complete","matched_times":count,"frequency_matching":"piecewise constant model-bin overlap onto observed bins; energy-weighted first directional moments; no extrapolation","time_matching":"one unused observed spectrum per prepared valid time, common model spectra within match_seconds","scores":rows}))
}
// Sampled period thresholds describe the records, not a unique swell system.
// Retain both bracket times: an hourly buoy and six-hour model have different
// timing precision, and missing records must not imply an exact arrival time.
fn event_summary(times:&[i64],hs:&[Option<f64>],tp:&[Option<f64>])->Value{
    let peak=|values:&[Option<f64>]|values.iter().enumerate().filter_map(|(i,v)|v.filter(|v|v.is_finite()).map(|v|(i,v))).max_by(|a,b|a.1.total_cmp(&b.1).then_with(||b.0.cmp(&a.0))).map(|(i,v)|json!({"time":times[i],"value":v}));
    let mut thresholds=Vec::new();
    for level in [16.,20.]{
        let mut prior:Option<(i64,f64)>=None;let mut crossings=Vec::new();let mut first_above=None;let mut last_above=None;let mut count=0;let mut valid_count=0;
        for(i,period)in tp.iter().enumerate(){let Some(v)=period.filter(|v|v.is_finite()&&*v>0.)else{continue};valid_count+=1;
            if v>=level{count+=1;first_above.get_or_insert(times[i]);last_above=Some(times[i]);
                if prior.is_none_or(|(_,p)|p<level){crossings.push(json!({"first_above_time":times[i],"previous_below_time":prior.map(|(t,_)|t),"bracket_seconds":prior.map(|(t,_)|times[i]-t),"left_censored":prior.is_none(),"period_seconds":v,"hs_m":hs[i]}));}
            }prior=Some((times[i],v));
        }
        thresholds.push(json!({"threshold_seconds":level,"valid_period_samples":valid_count,"samples_at_or_above":count,"first_at_or_above_time":first_above,"last_at_or_above_time":last_above,"upward_crossings":crossings}));
    }
    json!({"hs_peak":peak(hs),"tp_peak":peak(tp),"period_thresholds":thresholds,"interpretation":"sampled bulk peak-period threshold crossings; no interpolation or unique swell identification; a left-censored crossing was already present at first valid sample; bracket_seconds includes gaps"})
}
pub fn run(r:&Value,out:&Path)->Result<Value>{
    let prepared=Path::new(text(r,"prepared")?);let manifest:Value=serde_json::from_slice(&fs::read(prepared.join("manifest.json"))?)?;
    if manifest["schema"]!="gpuwm.waves.prepared/v1"{return Err("prepared schema differs".into());}
    let stations=manifest["request"]["stations"].as_array().ok_or("prepared station recipes missing")?;let target:Vec<i64>=manifest["times"].as_array().ok_or("prepared times missing")?.iter().map(|v|v.as_i64().ok_or("bad time")).collect::<std::result::Result<_,_>>()?;
    let models:Vec<Series>=r["models"].as_array().ok_or("models required")?.iter().map(|row|load(row,stations,&target)).collect::<Result<_>>()?;if models.is_empty(){return Err("at least one series required".into());}
    let tolerance=r["match_seconds"].as_i64().unwrap_or(1800);if !(0..=3600).contains(&tolerance){return Err("matching tolerance must be 0..3600 seconds".into());}
    let begin=target[0];let end=*target.last().ok_or("empty target times")?;
    let results:Vec<Result<Value>>=stations.par_iter().enumerate().map(|(si,station)|{
        let obs=observations::stdmet(Path::new(text(station,"stdmet")?),begin,end)?;
        let mut pairs=Vec::new();let mut stats=Vec::new();let mut used=BTreeSet::new();
        for &t in &target{let matched:Vec<Option<usize>>=models.iter().map(|m|nearest(&m.times,t,tolerance)).collect();
            for(key,obskey)in KEYS{
                // Standard-met rows often have 10-minute winds but waves only
                // at :10 and :40. Match the nearest valid observation of THIS
                // quantity, never a closer row whose wave value is missing.
                let oi=obs.iter().enumerate().filter(|(i,o)|o.values.contains_key(obskey)&&(o.epoch-t).abs()<=tolerance&&!used.contains(&(key,*i))).min_by_key(|(_,o)|((o.epoch-t).abs(),o.epoch)).map(|(i,_)|i);
                let Some(oi)=oi else{continue};let observed=obs[oi].values[obskey];if (key=="dp"||key=="dir")&&obs[oi].values.get("WVHT").is_none_or(|v|*v<0.05){continue}
                let available:Vec<Option<f64>>=models.iter().zip(&matched).map(|(m,&ti)|ti.and_then(|ti|m.fields.get(key).map(|v|v[si*m.times.len()+ti])).filter(|v|v.is_finite())).collect();
                // The same observed values score every series that provides this metric.
                let providers:Vec<_>=models.iter().enumerate().filter(|(_,m)|m.fields.contains_key(key)).map(|(i,_)|i).collect();
                if providers.is_empty()||providers.iter().any(|&i|available[i].is_none()){continue}
                used.insert((key,oi));
                for mi in providers{pairs.push(json!({"station":station["id"],"valid_time":t,"obs_time":obs[oi].epoch,"model_time":models[mi].times[matched[mi].unwrap()],"model":models[mi].label,"metric":key,"obs":observed,"prediction":available[mi].unwrap()}));}
            }
        }
        for m in &models{for(key,obskey)in KEYS{let p:Vec<_>=pairs.iter().filter(|p|p["model"]==m.label&&p["metric"]==key).map(|p|(p["obs"].as_f64().unwrap(),p["prediction"].as_f64().unwrap())).collect();stats.push(json!({"station":station["id"],"model":m.label,"metric":key,"observation":obskey,"scores":scores(&p,key=="dp"||key=="dir")}));}}
        let observed:Vec<Value>=obs.iter().map(|o|json!({"time":o.epoch,"hs":o.values.get("WVHT"),"tp":o.values.get("DPD"),"tm02":o.values.get("APD"),"dp":o.values.get("MWD")})).collect();
        let series:Vec<_>=models.iter().map(|m|{let v:BTreeMap<_,_>=m.fields.iter().map(|(k,v)|(k,v[si*m.times.len()..(si+1)*m.times.len()].to_vec())).collect();json!({"label":m.label,"times":m.times,"fields":v})}).collect();
        let obs_times:Vec<_>=obs.iter().map(|o|o.epoch).collect();
        let obs_hs:Vec<_>=obs.iter().map(|o|o.values.get("WVHT").copied()).collect();let obs_tp:Vec<_>=obs.iter().map(|o|o.values.get("DPD").copied()).collect();
        let model_events:Vec<_>=models.iter().map(|m|{let values=|key:&str|->Vec<Option<f64>>{(0..m.times.len()).map(|ti|m.fields.get(key).map(|v|v[si*m.times.len()+ti]).filter(|v|v.is_finite())).collect()};json!({"label":m.label,"summary":event_summary(&m.times,&values("hs"),&values("tp"))})}).collect();
        let events=json!({"observations":event_summary(&obs_times,&obs_hs,&obs_tp),"models":model_events});
        Ok(json!({"station":station["id"],"pairs":pairs,"scores":stats,"observations":observed,"series":series,"events":events,"spectra":spectral_panel(station,si,&obs,&models,begin,end)?,"spectral_scores":spectral_scores(station,si,&models,&target,tolerance)?}))
    }).collect();
    let results:Vec<Value>=results.into_iter().collect::<Result<_>>()?;let mut pooled=Vec::new();let mut csv=String::from("station,valid_time,obs_time,model_time,model,metric,obs,prediction\n");
    for row in &results{for p in row["pairs"].as_array().unwrap(){csv.push_str(&format!("{},{},{},{},{},{},{},{}\n",text(p,"station")?,p["valid_time"],p["obs_time"],p["model_time"],text(p,"model")?,text(p,"metric")?,p["obs"],p["prediction"]));}}
    for m in &models{for(key,obskey)in KEYS{let pairs:Vec<_>=results.iter().flat_map(|v|v["pairs"].as_array().unwrap()).filter(|p|p["model"]==m.label&&p["metric"]==key).map(|p|(p["obs"].as_f64().unwrap(),p["prediction"].as_f64().unwrap())).collect();pooled.push(json!({"model":m.label,"metric":key,"observation":obskey,"scores":scores(&pairs,key=="dp"||key=="dir")}));}}
    fs::write(out.join("pairs.csv"),csv)?;
    let mut csv=String::from("station,model,frequency_hz,metric,n,bias,rmse,scatter_index,correlation\n");
    for station in &results{for row in station["spectral_scores"]["scores"].as_array().into_iter().flatten(){let s=&row["scores"];csv.push_str(&format!("{},{},{},{},{},{},{},{},{}\n",text(station,"station")?,text(row,"model")?,row["frequency_hz"],text(row,"metric")?,s["n"],s["bias"],s["rmse"],s["scatter_index"],s["correlation"]));}}
    fs::write(out.join("spectral-scores.csv"),csv)?;
    let result=json!({"schema":"gpuwm.waves.verification/v1","status":"complete","prepared_manifest_sha256":arrays::sha(&prepared.join("manifest.json"))?,"request":r,"match_seconds":tolerance,"pairing":"one nearest observation per target valid time, no reused observation; common finite pairs across providers of each metric","scatter_index_definition":"standard deviation of prediction minus observation divided by observed mean; bias removed","direction_error":"wrapped to [-180,180), bias is circular mean, RMSE is wrapped-error RMS","period_note":"Tm02 vs NDBC APD is the matching spectral moment; Tm01 vs APD is a labelled diagnostic comparison","direction_note":"dp matches NDBC mean direction at dominant period; integrated dir vs MWD is diagnostic only","pooled":pooled,"stations":results});
    fs::write(out.join("verification.json"),serde_json::to_vec_pretty(&result)?)?;Ok(json!({"status":"complete","stations":stations.len(),"series":models.len(),"pooled":pooled}))
}

#[cfg(test)]
mod tests{use super::*;
    #[test]fn wrap_and_bias_removed_scatter(){let m=scores(&[(359.,1.),(1.,359.)],true);assert!(m["bias"].as_f64().unwrap().abs()<1e-12);assert_eq!(m["rmse"],2.);let m=scores(&[(1.,2.),(2.,3.)],false);assert_eq!(m["bias"],1.);assert_eq!(m["scatter_index"],0.);assert_eq!(m["correlation"],1.);}
    #[test]fn isotropic_mem_normalizes(){let angles:Vec<_>=(0..72).map(|i|i as f64*5.).collect();let d=spectra::mem(0.,0.,0.,0.,&angles).unwrap();assert!((d.iter().sum::<f64>()*2.*std::f64::consts::PI/72.-1.).abs()<1e-12);assert!(spectra::mem(1.,0.,0.,0.,&angles).is_none());}
}
