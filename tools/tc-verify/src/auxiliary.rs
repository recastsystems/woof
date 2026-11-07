//! Paired observation contracts. Grids must already share projection, cells and accumulation bounds.
//! FSS: DOI 10.1175/2007MWR2123.1; implemented from fractions definition, no external code.
use anyhow::{ensure, Result};
use rayon::prelude::*;
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
#[derive(Deserialize)]
pub struct Rain {
    pub nx: usize,
    pub ny: usize,
    pub dx_km: f64,
    pub cell_area_km2: f64,
    pub grid_id: String,
    pub start_utc: String,
    pub end_utc: String,
    pub observation_source: String,
    pub observed: Vec<Option<f64>>,
    pub forecasts: BTreeMap<String, Vec<Option<f64>>>,
    pub thresholds_mm: Vec<f64>,
    pub neighborhoods_km: Vec<f64>,
    #[serde(default)]
    pub track_distance_km: Option<Vec<Option<f64>>>,
    #[serde(default)]
    pub max_track_distance_km: Option<f64>,
}
fn integral(a: &[f64], nx: usize, ny: usize) -> Vec<f64> {
    let mut s = vec![0.; (nx + 1) * (ny + 1)];
    for y in 0..ny {
        let mut row = 0.;
        for x in 0..nx {
            row += a[y * nx + x];
            s[(y + 1) * (nx + 1) + x + 1] = s[y * (nx + 1) + x + 1] + row;
        }
    }
    s
}
fn area(s: &[f64], nx: usize, x0: usize, y0: usize, x1: usize, y1: usize) -> f64 {
    let w = nx + 1;
    s[y1 * w + x1] - s[y0 * w + x1] - s[y1 * w + x0] + s[y0 * w + x0]
}
pub fn rain(r: Rain, models: &[String]) -> Result<Value> {
    ensure!(
        !models.is_empty() && models.iter().collect::<BTreeSet<_>>().len() == models.len(),
        "explicit unique nonempty rain cohort required"
    );
    ensure!(
        r.nx > 0 && r.ny > 0 && r.dx_km > 0. && r.cell_area_km2 > 0.,
        "invalid rain grid"
    );
    let n =
        r.nx.checked_mul(r.ny)
            .ok_or_else(|| anyhow::anyhow!("grid overflow"))?;
    ensure!(
        n <= 10_000_000 && r.observed.len() == n && r.forecasts.values().all(|v| v.len() == n),
        "rain shape mismatch or >10M cell cap"
    );
    ensure!(
        models.iter().all(|m| r.forecasts.contains_key(m)),
        "rain homogeneous cohort missing model"
    );
    ensure!(
        !r.grid_id.is_empty()
            && !r.observation_source.is_empty()
            && crate::deck::time(&r.end_utc)? > crate::deck::time(&r.start_utc)?,
        "rain provenance and common positive accumulation window required"
    );
    if let Some(d) = &r.track_distance_km {
        ensure!(d.len() == n, "track distance shape mismatch");
    }
    ensure!(
        r.max_track_distance_km.is_none() || r.track_distance_km.is_some(),
        "track radius requires distance grid"
    );
    let valid: Vec<_> = (0..n)
        .map(|i| {
            r.observed[i].is_some_and(|v| v.is_finite() && v >= 0.)
                && models
                    .iter()
                    .all(|m| r.forecasts[m][i].is_some_and(|v| v.is_finite() && v >= 0.))
                && r.max_track_distance_km.is_none_or(|max| {
                    r.track_distance_km.as_ref().unwrap()[i].is_some_and(|d| d >= 0. && d <= max)
                })
        })
        .collect();
    let count = valid.iter().filter(|x| **x).count();
    let mask: Vec<_> = valid.iter().map(|v| if *v { 1. } else { 0. }).collect();
    let ms = integral(&mask, r.nx, r.ny);
    let mut tasks = vec![];
    for model in models {
        for th in &r.thresholds_mm {
            for width in &r.neighborhoods_km {
                tasks.push((model, *th, *width));
            }
        }
    }
    let rows:Vec<_>=tasks.into_par_iter().map(|(model,th,width)|->Result<Value>{
 ensure!(th>0.&&width>=r.dx_km,"rain thresholds positive and neighborhoods >= grid spacing required");
 let f=&r.forecasts[model];let o=&r.observed;let binary=|v:&Vec<Option<f64>>|(0..n).map(|i|if valid[i]&&v[i].unwrap()>=th{1.}else{0.}).collect::<Vec<_>>();
 let fb=binary(f);let ob=binary(o);let fs=integral(&fb,r.nx,r.ny);let os=integral(&ob,r.nx,r.ny);
 let radius=((width/r.dx_km-1.)/2.).round().max(0.)as usize;let(mut numerator,mut denominator,mut hits,mut misses,mut false_alarms)=(0.,0.,0usize,0usize,0usize);
 for y in 0..r.ny{for x in 0..r.nx{let i=y*r.nx+x;if !valid[i]{continue}let(x0,y0,x1,y1)=(x.saturating_sub(radius),y.saturating_sub(radius),(x+radius+1).min(r.nx),(y+radius+1).min(r.ny));let m=area(&ms,r.nx,x0,y0,x1,y1);let a=area(&fs,r.nx,x0,y0,x1,y1)/m;let b=area(&os,r.nx,x0,y0,x1,y1)/m;numerator+=(a-b).powi(2);denominator+=a*a+b*b;
 match(fb[i]>0.,ob[i]>0.){(true,true)=>hits+=1,(false,true)=>misses+=1,(true,false)=>false_alarms+=1,_=>{}}
 }}
 let expected=if count>0{(hits+misses)as f64*(hits+false_alarms)as f64/count as f64}else{0.};let ets_den=(hits+misses+false_alarms)as f64-expected;
 let total=|v:&Vec<Option<f64>>|(0..n).filter(|i|valid[*i]).map(|i|v[i].unwrap()).sum::<f64>();let ft=total(f);let ot=total(o);
 Ok(json!({"model":model,"threshold_mm":th,"requested_width_km":width,"actual_width_km":(2*radius+1)as f64*r.dx_km,"n":count,"fss":if denominator>0.{Some(1.-numerator/denominator)}else{None},"ets":if ets_den>0.{Some((hits as f64-expected)/ets_den)}else{None},"forecast_mean_mm":if count>0{Some(ft/count as f64)}else{None},"observed_mean_mm":if count>0{Some(ot/count as f64)}else{None},"volume_bias_mm_km2":(ft-ot)*r.cell_area_km2,"frequency_bias":if hits+misses>0{Some((hits+false_alarms)as f64/(hits+misses)as f64)}else{None}}))
 }).collect::<Result<_>>()?;
    Ok(
        json!({"rows":rows,"grid_id":r.grid_id,"observation_source":r.observation_source,"start_utc":r.start_utc,"end_utc":r.end_utc,"common_valid_cells":count,"max_track_distance_km":r.max_track_distance_km,"edge_policy":"clipped windows, common valid mask; all-dry FSS undefined"}),
    )
}
#[derive(Deserialize)]
pub struct Station {
    pub station: String,
    pub variable: String,
    pub units: String,
    pub datum: String,
    pub observation_source: String,
    pub times: Vec<String>,
    pub observed: Vec<Option<f64>>,
    pub forecasts: BTreeMap<String, Vec<Option<f64>>>,
}
pub fn stations(stations: Vec<Station>, models: &[String]) -> Result<Value> {
    ensure!(
        !models.is_empty() && models.iter().collect::<BTreeSet<_>>().len() == models.len(),
        "explicit unique nonempty station cohort required"
    );
    let mut rows = vec![];
    for s in stations {
        ensure!(
            matches!(
                (s.variable.as_str(), s.units.as_str()),
                ("water_level", "m")
                    | ("surge", "m")
                    | ("significant_wave_height", "m")
                    | ("peak_wave_period", "s")
            ),
            "station variable or units invalid"
        );
        ensure!(
            !s.datum.is_empty() && !s.observation_source.is_empty(),
            "datum/reference and source required, use not-applicable for waves"
        );
        let n = s.times.len();
        ensure!(
            n == s.observed.len() && s.forecasts.values().all(|v| v.len() == n),
            "station shape mismatch"
        );
        ensure!(
            models.iter().all(|m| s.forecasts.contains_key(m)),
            "station cohort missing model"
        );
        let ts: Vec<_> = s
            .times
            .iter()
            .map(|t| crate::deck::time(t))
            .collect::<Result<_>>()?;
        ensure!(
            ts.iter().collect::<BTreeSet<_>>().len() == n && ts.windows(2).all(|w| w[0] < w[1]),
            "station times must be unique and increasing"
        );
        let valid: Vec<_> = (0..n)
            .filter(|i| {
                s.observed[*i].is_some_and(f64::is_finite)
                    && models
                        .iter()
                        .all(|m| s.forecasts[m][*i].is_some_and(f64::is_finite))
            })
            .collect();
        for model in models {
            let f = &s.forecasts[model];
            let count = valid.len();
            let errors: Vec<_> = valid
                .iter()
                .map(|i| f[*i].unwrap() - s.observed[*i].unwrap())
                .collect();
            let avg = |fun: fn(f64) -> f64| {
                if count > 0 {
                    Some(errors.iter().map(|e| fun(*e)).sum::<f64>() / count as f64)
                } else {
                    None
                }
            };
            let peak = |v: &Vec<Option<f64>>| {
                valid.iter().copied().max_by(|i, j| {
                    v[*i]
                        .unwrap()
                        .total_cmp(&v[*j].unwrap())
                        .then_with(|| j.cmp(i))
                })
            };
            let peaks = peak(f).zip(peak(&s.observed));
            rows.push(json!({"station":s.station,"model":model,"variable":s.variable,"units":s.units,"datum":s.datum,"source":s.observation_source,"n":count,"bias":avg(|x|x),"mae":avg(f64::abs),"rmse":avg(|x|x*x).map(f64::sqrt),"peak_error":peaks.map(|(a,b)|f[a].unwrap()-s.observed[b].unwrap()),"peak_timing_error_hours":peaks.map(|(a,b)|(ts[a]-ts[b])as f64/3600.),"peak_window":"common observed timestamps only"}));
        }
    }
    Ok(json!({"rows":rows}))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dry_is_undefined() {
        let r = Rain {
            nx: 2,
            ny: 2,
            dx_km: 1.,
            cell_area_km2: 1.,
            grid_id: "test".into(),
            start_utc: "2020010100".into(),
            end_utc: "2020010200".into(),
            observation_source: "synthetic test".into(),
            observed: vec![Some(0.); 4],
            forecasts: BTreeMap::from([("A".into(), vec![Some(0.); 4])]),
            thresholds_mm: vec![50.],
            neighborhoods_km: vec![1.],
            track_distance_km: None,
            max_track_distance_km: None,
        };
        let v = rain(r, &["A".into()]).unwrap();
        assert!(v["rows"][0]["fss"].is_null());
    }
}

#[cfg(test)]
mod paired_tests {
    use super::*;
    #[test]
    fn station_common_times_and_peak() {
        let s:Station=serde_json::from_value(json!({"station":"synthetic","variable":"surge","units":"m","datum":"test","observation_source":"synthetic","times":["2020010100","2020010101","2020010102"],"observed":[0.,2.,1.],"forecasts":{"A":[0.,1.,3.],"B":[null,2.,1.]}})).unwrap();
        let r = stations(vec![s], &["A".into(), "B".into()]).unwrap();
        assert_eq!(r["rows"][0]["n"], 2);
        assert_eq!(r["rows"][0]["peak_timing_error_hours"], 1.);
        assert_eq!(r["rows"][0]["bias"], 0.5);
    }
    #[test]
    fn perfect_wet_fss_and_shift() {
        let v = json!({"nx":2,"ny":2,"dx_km":1.,"cell_area_km2":1.,"grid_id":"test","start_utc":"2020010100","end_utc":"2020010200","observation_source":"synthetic","observed":[100.,0.,0.,0.],"forecasts":{"A":[100.,0.,0.,0.],"B":[0.,100.,0.,0.]},"thresholds_mm":[50.],"neighborhoods_km":[1.]});
        let r = rain(
            serde_json::from_value(v).unwrap(),
            &["A".into(), "B".into()],
        )
        .unwrap();
        assert_eq!(r["rows"][0]["fss"], 1.);
        assert_eq!(r["rows"][1]["fss"], 0.);
    }
}
