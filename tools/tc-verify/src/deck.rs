//! ATCF format: https://www.nrlmry.navy.mil/atcf_web/docs/database/new/abdeck.txt
//! NOAA HURDAT2 and IBTrACS USA columns retain one-minute winds in knots.
use anyhow::{bail, ensure, Context, Result};
use chrono::{Duration, NaiveDateTime};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, io::Read, path::Path};
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Fix {
    pub storm: String,
    pub model: String,
    pub init: i64,
    pub valid: i64,
    pub lead: i64,
    pub lat: Option<f64>,
    pub lon: Option<f64>,
    pub wind: Option<f64>,
    pub pressure: Option<f64>,
    pub status: String,
    pub radii: BTreeMap<u32, [Option<f64>; 4]>,
}
pub fn time(s: &str) -> Result<i64> {
    for fmt in [
        "%Y%m%d%H%M",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
    ] {
        if let Ok(t) = NaiveDateTime::parse_from_str(s, fmt) {
            return Ok(t.and_utc().timestamp());
        }
    }
    if s.len() == 10 {
        return time(&format!("{s}00"));
    }
    bail!("invalid UTC time {s}")
}
pub fn stamp(t: i64) -> String {
    chrono::DateTime::from_timestamp(t, 0)
        .unwrap()
        .format("%Y%m%d%H")
        .to_string()
}
pub fn text(path: &Path) -> Result<String> {
    let bytes = std::fs::read(path).with_context(|| path.display().to_string())?;
    let mut out = String::new();
    if bytes.starts_with(&[31, 139]) {
        flate2::read::GzDecoder::new(&bytes[..])
            .take(512 * 1024 * 1024 + 1)
            .read_to_string(&mut out)?;
    } else {
        out = String::from_utf8(bytes)?;
    }
    ensure!(
        out.len() <= 512 * 1024 * 1024,
        "deck exceeds 512 MiB decoded cap"
    );
    Ok(out)
}
fn val(s: &str, zero: bool) -> Option<f64> {
    s.trim()
        .parse::<f64>()
        .ok()
        .filter(|v| v.is_finite() && if zero { *v >= 0. } else { *v > 0. })
}
fn coord(s: &str, scale: f64, latitude: bool) -> Result<Option<f64>> {
    let s = s.trim();
    if s.is_empty() || s == "0" || s == "-999" {
        return Ok(None);
    }
    ensure!(s.is_ascii(), "coordinates must be ASCII");
    let (n, h) = s.split_at(s.len() - 1);
    if n == "-999" || n == "-9999" {
        return Ok(None);
    }
    ensure!(
        if latitude {
            matches!(h, "N" | "S")
        } else {
            matches!(h, "E" | "W")
        },
        "bad hemisphere {s}"
    );
    let v = n.parse::<f64>()? * scale * if h == "S" || h == "W" { -1. } else { 1. };
    ensure!(
        v.is_finite() && v.abs() <= if latitude { 90. } else { 360. },
        "bad coordinate {s}"
    );
    Ok(Some(if latitude {
        v
    } else {
        (v + 180.).rem_euclid(360.) - 180.
    }))
}
pub fn parse(raw: &str, format: &str) -> Result<Vec<Fix>> {
    parse_selected(raw, format, None)
}
pub(crate) fn parse_selected(raw: &str, format: &str, storm: Option<&str>) -> Result<Vec<Fix>> {
    let mut out = Vec::new();
    match format {
        "atcf" => {
            for (line, s) in raw.lines().enumerate() {
                if s.trim().is_empty() {
                    continue;
                }
                let r: Vec<_> = s.split(',').map(str::trim).collect();
                ensure!(
                    r.len() >= 10,
                    "ATCF line {} has fewer than 10 fields",
                    line + 1
                );
                ensure!(
                    r[2].len() == 10 && r[2].bytes().all(|x| x.is_ascii_digit()),
                    "invalid ATCF timestamp on line {}",
                    line + 1
                );
                let best = r[4] == "BEST";
                let init = time(r[2])?;
                let lead = r[5]
                    .parse::<i64>()
                    .with_context(|| format!("bad lead on line {}", line + 1))?;
                ensure!(lead.abs() <= 10000, "lead outside supported range");
                let minutes = if best && !r[3].is_empty() {
                    r[3].parse::<i64>()?
                } else {
                    0
                };
                ensure!(!best || (0..60).contains(&minutes), "bad BEST minutes");
                if best && lead != 0 {
                    continue;
                }
                if !best && lead < 0 {
                    continue;
                }
                let valid = init + Duration::hours(lead).num_seconds() + 60 * minutes;
                let mut f = Fix {
                    storm: format!("{}{:0>2}{}", r[0], r[1], &r[2][..4]).to_uppercase(),
                    model: r[4].into(),
                    init: if best { valid } else { init },
                    valid,
                    lead,
                    lat: coord(r[6], 0.1, true)?,
                    lon: coord(r[7], 0.1, false)?,
                    wind: val(r[8], false),
                    pressure: val(r[9], false),
                    status: r.get(10).unwrap_or(&"").to_string(),
                    radii: BTreeMap::new(),
                };
                if r.len() >= 17 {
                    if let Ok(rad) = r[11].parse::<u32>() {
                        if [34, 50, 64].contains(&rad) {
                            let mut q = [None; 4];
                            if r[12] == "AAA" {
                                q.fill(val(r[13], true));
                            } else if let Some(start) = ["NEQ", "SEQ", "SWQ", "NWQ"]
                                .iter()
                                .position(|x| *x == r[12])
                            {
                                for k in 0..4 {
                                    q[(start + k) % 4] = val(r[13 + k], true);
                                }
                            }
                            f.radii.insert(rad, q);
                        }
                    }
                }
                out.push(f);
            }
        }
        "hurdat2" => {
            let mut id = String::new();
            let mut remaining = 0usize;
            for s in raw.lines().filter(|s| !s.trim().is_empty()) {
                let r: Vec<_> = s.split(',').map(str::trim).collect();
                if r.len() >= 3
                    && r[0].len() == 8
                    && r[0].is_ascii()
                    && r[0][..2].bytes().all(|c| c.is_ascii_alphabetic())
                {
                    ensure!(remaining == 0, "truncated HURDAT2 storm");
                    id = r[0].into();
                    remaining = r[2].parse()?;
                    continue;
                }
                ensure!(remaining > 0 && r.len() >= 8, "bad HURDAT2 row");
                remaining -= 1;
                let t = time(&format!("{}{}", r[0], r[1]))?;
                let mut radii = BTreeMap::new();
                for (j, rad) in [34, 50, 64].iter().enumerate() {
                    if r.len() >= 12 + 4 * j {
                        radii.insert(*rad, std::array::from_fn(|k| val(r[8 + 4 * j + k], true)));
                    }
                }
                out.push(Fix {
                    storm: id.clone(),
                    model: "BEST".into(),
                    init: t,
                    valid: t,
                    lead: 0,
                    lat: coord(r[4], 1., true)?,
                    lon: coord(r[5], 1., false)?,
                    wind: val(r[6], false),
                    pressure: val(r[7], false),
                    status: r[3].into(),
                    radii,
                });
            }
            ensure!(remaining == 0, "truncated HURDAT2 storm");
        }
        "ibtracs-usa" => {
            let mut reader = csv::ReaderBuilder::new()
                .flexible(true)
                .from_reader(raw.as_bytes());
            let headers = reader.headers()?.clone();
            let ix = |key: &str| {
                headers
                    .iter()
                    .position(|h| h == key)
                    .with_context(|| format!("IBTrACS missing {key}"))
            };
            let keys = [
                "USA_ATCF_ID",
                "ISO_TIME",
                "USA_LAT",
                "USA_LON",
                "USA_WIND",
                "USA_PRES",
                "USA_STATUS",
            ];
            let inds: Vec<_> = keys.iter().map(|k| ix(k)).collect::<Result<_>>()?;
            for row in reader.records() {
                let r = row?;
                let get = |i: usize| r.get(inds[i]).unwrap_or("").trim();
                if get(0).len() != 8 || get(1).starts_with(" ") {
                    continue;
                }
                let Ok(t) = time(get(1)) else { continue };
                let mut radii = BTreeMap::new();
                for rad in [34, 50, 64] {
                    let mut q = [None; 4];
                    for (k, quad) in ["NE", "SE", "SW", "NW"].iter().enumerate() {
                        if let Ok(i) = ix(&format!("USA_R{rad}_{quad}")) {
                            q[k] = val(r.get(i).unwrap_or(""), true);
                        }
                    }
                    radii.insert(rad, q);
                }
                out.push(Fix {
                    storm: get(0).into(),
                    model: "BEST".into(),
                    init: t,
                    valid: t,
                    lead: 0,
                    lat: get(2)
                        .parse::<f64>()
                        .ok()
                        .filter(|v| v.is_finite() && v.abs() <= 90.),
                    lon: get(3)
                        .parse::<f64>()
                        .ok()
                        .filter(|v| v.is_finite() && v.abs() <= 180.),
                    wind: val(get(4), false),
                    pressure: val(get(5), false),
                    status: get(6).into(),
                    radii,
                });
            }
        }
        _ => bail!("unknown best format {format}; use atcf, hurdat2, ibtracs-usa"),
    }
    // One fix per storm/model/cycle/lead. Radius records must never multiply samples.
    let mut merged: BTreeMap<(String, String, i64, i64), Fix> = BTreeMap::new();
    for f in out {
        if storm.is_some_and(|s| !s.eq_ignore_ascii_case(&f.storm)) {
            continue;
        }
        let key = (f.storm.clone(), f.model.clone(), f.init, f.valid);
        if let Some(old) = merged.get_mut(&key) {
            for (a, b) in [
                (&mut old.lat, f.lat),
                (&mut old.lon, f.lon),
                (&mut old.wind, f.wind),
                (&mut old.pressure, f.pressure),
            ] {
                if a.is_none() {
                    *a = b;
                } else if b.is_some() {
                    ensure!(*a == b, "conflicting ATCF core fix {key:?}");
                }
            }
            for (rad, qs) in f.radii {
                let q = old.radii.entry(rad).or_insert([None; 4]);
                for k in 0..4 {
                    if q[k].is_none() {
                        q[k] = qs[k];
                    } else if qs[k].is_some() {
                        ensure!(q[k] == qs[k], "conflicting radius {key:?} {rad}");
                    }
                }
            }
        } else {
            merged.insert(key, f);
        }
    }
    Ok(merged.into_values().collect())
}
pub fn tropical(f: &Fix) -> bool {
    matches!(
        f.status.as_str(),
        "TD" | "TS" | "HU" | "SD" | "SS" | "TY" | "ST" | "TC"
    )
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn radius_rows_merge_and_rotate() {
        let raw="AL, 01, 2024060100, 03, TEST, 12, 120N, 1795E, 70, 980, HU, 34, SEQ, 10, 20, 30, 0\nAL, 01, 2024060100, 03, TEST, 12, 120N, 1795E, 70, 980, HU, 50, AAA, 5, 0, 0, 0";
        let fs = parse(raw, "atcf").unwrap();
        assert_eq!(fs.len(), 1);
        assert_eq!(
            fs[0].radii[&34],
            [Some(0.), Some(10.), Some(20.), Some(30.)]
        );
        assert_eq!(fs[0].radii[&50], [Some(5.); 4]);
    }
    #[test]
    fn missing_is_not_zero() {
        let f = parse(
            "AL, 01, 2024060100, 03, TEST, 0, , , 0, -999, , 34, NEQ, 0, -999, , 5",
            "atcf",
        )
        .unwrap();
        assert_eq!(f[0].wind, None);
        assert_eq!(f[0].lat, None);
        assert_eq!(f[0].radii[&34], [Some(0.), None, None, Some(5.)]);
    }
    #[test]
    fn invalid_lead_and_conflicting_duplicates_fail() {
        assert!(parse(
            "AL, 01, 2024060100, 03, TEST, x, 120N, 1795E, 70, 980",
            "atcf"
        )
        .is_err());
        let row = "AL, 01, 2024060100, 03, TEST, 0, 120N, 1795E, 70, 980";
        assert!(parse(
            &format!("{row}\n{}", row.replace("70, 980", "80, 980")),
            "atcf"
        )
        .is_err());
    }
    #[test]
    fn best_special_minutes_and_hurdat() {
        let fs = parse(
            "AL, 01, 2024060100, 30, BEST, 0, 120N, 1795E, 70, 980, HU",
            "atcf",
        )
        .unwrap();
        assert_eq!(fs[0].valid, time("202406010030").unwrap());
        let h = parse(
            "AL012024, TEST, 1,\n20240601, 0030, L, HU, 12.0N, 179.5E, 70, 980, 10, 20, 30, 40,",
            "hurdat2",
        )
        .unwrap();
        assert_eq!(h[0].valid, fs[0].valid);
        assert_eq!(h[0].wind, fs[0].wind);
    }
}

pub fn read_storm(path: &Path, format: &str, storm: &str) -> Result<Vec<Fix>> {
    parse_selected(&text(path)?, format, Some(storm))
}
