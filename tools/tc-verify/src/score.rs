use crate::deck::{tropical, Fix};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
// Great-circle diagnostic uses a 6371.2 km sphere; MET-TC TK_ERR is a separate planar metric.
pub const EARTH_KM: f64 = 6371.2;
pub fn distance(a: f64, b: f64, c: f64, d: f64) -> f64 {
    let (a, c, dl) = (a.to_radians(), c.to_radians(), (d - b).to_radians());
    let h = ((c - a) / 2.).sin().powi(2) + a.cos() * c.cos() * (dl / 2.).sin().powi(2);
    2. * EARTH_KM * h.clamp(0., 1.).sqrt().asin()
}
fn dlon(a: f64, b: f64) -> f64 {
    (b - a + 180.).rem_euclid(360.) - 180.
}
// Independently implemented MET 12.2.2 ACERR conventions, including its historical
// constants. See dtcenter/MET src/libcode/vx_tc_util/acerr.f (Apache-2.0).
fn displacement(a: f64, b: f64, c: f64, d: f64) -> (f64, f64) {
    (
        111.1 * dlon(b, d) * (0.5 * 0.0174533 * (a + c)).cos(),
        111.1 * (c - a),
    )
}
fn headings(forecasts: &[Fix], best: &[Fix]) -> BTreeMap<(String, String, i64, i64), f64> {
    let mut groups: BTreeMap<_, Vec<&Fix>> = BTreeMap::new();
    for f in forecasts {
        if f.lat.is_some()
            && f.lon.is_some()
            && best.iter().any(|b| {
                b.storm == f.storm && b.valid == f.valid && b.lat.is_some() && b.lon.is_some()
            })
        {
            groups
                .entry((f.storm.clone(), f.model.clone(), f.init))
                .or_default()
                .push(f);
        }
    }
    let mut out = BTreeMap::new();
    for ((storm, model, init), fs) in groups {
        if fs.len() < 2 {
            continue;
        }
        let lo = fs.iter().map(|f| f.valid).min().unwrap();
        let hi = fs.iter().map(|f| f.valid).max().unwrap();
        // MET's regular six-hour best-track grid. Special landfall fixes do not change its tangent.
        let bs: Vec<_> = best
            .iter()
            .filter(|b| {
                b.storm == storm
                    && b.valid >= lo
                    && b.valid <= hi
                    && (b.valid - lo) % 21600 == 0
                    && b.lat.is_some()
                    && b.lon.is_some()
            })
            .collect();
        if bs.len() < 2 {
            continue;
        }
        let mut hs = vec![None; bs.len()];
        for i in 1..bs.len() {
            let (a, b) = (bs[i - 1], bs[i]);
            let (x, y) = displacement(
                a.lat.unwrap(),
                a.lon.unwrap(),
                b.lat.unwrap(),
                b.lon.unwrap(),
            );
            let rad = x.hypot(y);
            if rad > 0. {
                let angle = 57.296 * (x / rad).clamp(-1., 1.).acos();
                hs[i] = Some(if y < 0. { 360. - angle } else { angle });
            }
        }
        hs[0] = hs[1];
        for i in 1..hs.len() {
            if hs[i].is_none() {
                hs[i] = hs[i - 1];
            }
        }
        for i in (0..hs.len() - 1).rev() {
            if hs[i].is_none() {
                hs[i] = hs[i + 1];
            }
        }
        for (b, h) in bs.into_iter().zip(hs) {
            out.insert(
                (storm.clone(), model.clone(), init, b.valid),
                h.unwrap_or(90.).to_radians(),
            );
        }
    }
    out
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pair {
    pub storm: String,
    pub model: String,
    pub init: i64,
    pub lead: i64,
    pub valid: i64,
    pub errors: BTreeMap<String, f64>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Stat {
    pub cohort: Vec<String>,
    pub model: String,
    pub lead: i64,
    pub metric: String,
    pub n: usize,
    pub mean: f64,
    pub mae: f64,
    pub rmse: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ri {
    pub model: String,
    pub hours: i64,
    pub threshold_kt: f64,
    pub hits: usize,
    pub misses: usize,
    pub false_alarms: usize,
    pub correct_negatives: usize,
    pub pod: Option<f64>,
    pub far: Option<f64>,
    pub csi: Option<f64>,
    pub peirce: Option<f64>,
}
pub fn pairs(forecasts: &[Fix], best: &[Fix], filter: bool) -> Vec<Pair> {
    let b: BTreeMap<_, _> = best
        .iter()
        .map(|x| ((x.storm.clone(), x.valid), x))
        .collect();
    let heading = headings(forecasts, best);
    forecasts
        .iter()
        .filter_map(|f| {
            let o = *b.get(&(f.storm.clone(), f.valid))?;
            if filter
                && (!tropical(o)
                    || !b
                        .get(&(f.storm.clone(), f.init))
                        .is_some_and(|x| tropical(x)))
            {
                return None;
            }
            let mut e = BTreeMap::new();
            if let (Some(a), Some(c), Some(x), Some(y)) = (o.lat, o.lon, f.lat, f.lon) {
                let dist = distance(a, c, x, y);
                e.insert("track_km".into(), dist);
                e.insert("track_nmi".into(), dist / 1.852);
                let planar =
                    (60. * dlon(c, y) * ((a + x) * 0.5).to_radians().cos()).hypot(60. * (x - a));
                e.insert("met_track_nmi".into(), planar);
                e.insert("met_track_km".into(), planar * 1.852);
                if let Some(h) = heading.get(&(f.storm.clone(), f.model.clone(), f.init, f.valid)) {
                    let (dx, dy) = displacement(a, c, x, y);
                    e.insert("along_km".into(), dy * h.sin() + dx * h.cos());
                    e.insert("cross_km".into(), dx * h.sin() - dy * h.cos());
                }
            }
            for (key, a, b) in [
                ("vmax_kt", f.wind, o.wind),
                ("mslp_hpa", f.pressure, o.pressure),
            ] {
                if let (Some(a), Some(b)) = (a, b) {
                    e.insert(key.into(), a - b);
                }
            }
            for rad in [34, 50, 64] {
                if let (Some(a), Some(b)) = (f.radii.get(&rad), o.radii.get(&rad)) {
                    for (k, q) in ["NE", "SE", "SW", "NW"].iter().enumerate() {
                        if let (Some(a), Some(b)) = (a[k], b[k]) {
                            e.insert(format!("r{rad}_{q}_nmi"), a - b);
                        }
                    }
                }
            }
            Some(Pair {
                storm: f.storm.clone(),
                model: f.model.clone(),
                init: f.init,
                lead: f.lead,
                valid: f.valid,
                errors: e,
            })
        })
        .collect()
}
pub fn aggregate(pairs: &[Pair], models: &[String], homogeneous: bool) -> Vec<Stat> {
    let mut membership: BTreeMap<_, BTreeSet<String>> = BTreeMap::new();
    for p in pairs {
        for m in p.errors.keys() {
            membership
                .entry((p.storm.clone(), p.init, p.lead, m.clone()))
                .or_default()
                .insert(p.model.clone());
        }
    }
    let mut groups: BTreeMap<_, Vec<f64>> = BTreeMap::new();
    for p in pairs {
        if !models.contains(&p.model) {
            continue;
        }
        for (metric, x) in &p.errors {
            if homogeneous
                && !models.iter().all(|m| {
                    membership[&(p.storm.clone(), p.init, p.lead, metric.clone())].contains(m)
                })
            {
                continue;
            }
            groups
                .entry((p.model.clone(), p.lead, metric.clone()))
                .or_default()
                .push(*x);
        }
    }
    groups
        .into_iter()
        .map(|((model, lead, metric), v)| {
            let n = v.len();
            Stat {
                cohort: if homogeneous {
                    models.to_vec()
                } else {
                    vec![model.clone()]
                },
                model,
                lead,
                metric,
                n,
                mean: v.iter().sum::<f64>() / n as f64,
                mae: v.iter().map(|x| x.abs()).sum::<f64>() / n as f64,
                rmse: (v.iter().map(|x| x * x).sum::<f64>() / n as f64).sqrt(),
            }
        })
        .collect()
}
fn ratio(a: usize, b: usize) -> Option<f64> {
    if b == 0 {
        None
    } else {
        Some(a as f64 / b as f64)
    }
}
pub fn ri(
    forecasts: &[Fix],
    best: &[Fix],
    models: &[String],
    homogeneous: bool,
    filter: bool,
) -> Vec<Ri> {
    let b: BTreeMap<_, _> = best
        .iter()
        .map(|x| ((x.storm.clone(), x.valid), x))
        .collect();
    let a: BTreeMap<_, _> = forecasts
        .iter()
        .map(|x| ((x.storm.clone(), x.model.clone(), x.init, x.lead), x))
        .collect();
    let mut out = Vec::new();
    for (hours, threshold) in [(12, 20.), (24, 30.), (36, 45.), (48, 55.)] {
        let mut events: BTreeMap<_, BTreeMap<String, (bool, bool)>> = BTreeMap::new();
        for f in forecasts.iter().filter(|f| f.lead <= 96) {
            let Some(end) = a.get(&(f.storm.clone(), f.model.clone(), f.init, f.lead + hours))
            else {
                continue;
            };
            let (Some(o), Some(oe)) = (
                b.get(&(f.storm.clone(), f.valid)),
                b.get(&(f.storm.clone(), end.valid)),
            ) else {
                continue;
            };
            if filter
                && (!tropical(o)
                    || !tropical(oe)
                    || !b
                        .get(&(f.storm.clone(), f.init))
                        .is_some_and(|x| tropical(x)))
            {
                continue;
            }
            if let (Some(w), Some(we), Some(v), Some(ve)) = (f.wind, end.wind, o.wind, oe.wind) {
                events
                    .entry((f.storm.clone(), f.init, f.lead))
                    .or_default()
                    .insert(f.model.clone(), (we - w >= threshold, ve - v >= threshold));
            }
        }
        for model in models {
            let (mut h, mut m, mut f, mut c) = (0, 0, 0, 0);
            for ev in events.values() {
                if homogeneous && !models.iter().all(|x| ev.contains_key(x)) {
                    continue;
                }
                if let Some((p, o)) = ev.get(model) {
                    match (p, o) {
                        (true, true) => h += 1,
                        (false, true) => m += 1,
                        (true, false) => f += 1,
                        _ => c += 1,
                    }
                }
            }
            let pod = ratio(h, h + m);
            let pofd = ratio(f, f + c);
            out.push(Ri {
                model: model.clone(),
                hours,
                threshold_kt: threshold,
                hits: h,
                misses: m,
                false_alarms: f,
                correct_negatives: c,
                pod,
                far: ratio(f, h + f),
                csi: ratio(h, h + f + m),
                peirce: pod.zip(pofd).map(|(a, b)| a - b),
            });
        }
    }
    out
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn geometry() {
        assert!((distance(0., 0., 0., 1.) - 111.1984).abs() < 0.001);
        assert!(distance(1., 179.9, 1., -179.9) < 23.);
        assert_eq!(distance(30., 20., 30., 20.), 0.);
    }
    #[test]
    fn missing_cohort_is_empty() {
        let p = Pair {
            storm: "AL012024".into(),
            model: "AAAA".into(),
            init: 0,
            valid: 0,
            lead: 0,
            errors: BTreeMap::from([("vmax_kt".into(), 5.)]),
        };
        assert!(aggregate(&[p], &["AAAA".into(), "BBBB".into()], true).is_empty());
    }
}
#[cfg(test)]
mod contract_tests {
    use super::*;
    #[test]
    fn homogeneous_each_metric_separately() {
        let make = |m: &str, e: BTreeMap<String, f64>| Pair {
            storm: "AL012024".into(),
            model: m.into(),
            init: 0,
            lead: 12,
            valid: 43200,
            errors: e,
        };
        let ps = vec![
            make(
                "A",
                BTreeMap::from([("vmax_kt".into(), -10.), ("track_nmi".into(), 5.)]),
            ),
            make("B", BTreeMap::from([("track_nmi".into(), 8.)])),
        ];
        let s = aggregate(&ps, &["A".into(), "B".into()], true);
        assert_eq!(s.len(), 2);
        assert!(s.iter().all(|s| s.metric == "track_nmi" && s.n == 1));
    }
    #[test]
    fn ri_threshold_inclusive_and_missing_cohort() {
        let raw="AL, 01, 2024060100, 03, TEST, 0, 120N, 1795E, 30, 1000, TS\nAL, 01, 2024060100, 03, TEST, 24, 130N, 1795E, 60, 990, TS";
        let fs = crate::deck::parse(raw, "atcf").unwrap();
        let bs = fs
            .iter()
            .map(|f| {
                let mut b = f.clone();
                b.model = "BEST".into();
                b.init = b.valid;
                b.lead = 0;
                b
            })
            .collect::<Vec<_>>();
        let r = ri(&fs, &bs, &["TEST".into()], true, true);
        let e = r.iter().find(|r| r.hours == 24).unwrap();
        assert_eq!(e.hits, 1);
        assert_eq!(e.pod, Some(1.));
        let missing = ri(&fs, &bs, &["TEST".into(), "NONE".into()], true, true);
        assert!(missing.iter().all(|r| r.pod.is_none()));
    }
    #[test]
    fn tangent_right_and_along_sign() {
        let raw="AL, 01, 2024060100, 03, TEST, 0, 100N, 010E, 30, 1000, TS\nAL, 01, 2024060100, 03, TEST, 6, 110N, 010E, 30, 1000, TS";
        let fs = crate::deck::parse(raw, "atcf").unwrap();
        let bs = fs
            .iter()
            .map(|f| {
                let mut b = f.clone();
                b.model = "BEST".into();
                b.init = b.valid;
                b.lead = 0;
                b.lon = Some(0.);
                b
            })
            .collect::<Vec<_>>();
        let ps = pairs(&fs, &bs, true);
        assert!(ps
            .iter()
            .all(|p| p.errors["cross_km"] > 100. && p.errors["along_km"].abs() < 0.01));
    }
}
