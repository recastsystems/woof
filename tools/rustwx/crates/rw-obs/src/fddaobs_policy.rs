//! WRF v4.7.1 observation admission, projection and unit conversion.
//! Public domain source: wrf_fddaobs_in.F and module_llxy.F, pinned and
//! noticed in tools/fddaobs_wrf471_oracle. No competitor source is used.
use super::fddaobs::{Datum, Measurements, Report};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, thread};
const MISS: f32 = -888888.;
const RAD: f32 = std::f32::consts::PI / 180.;
const RE: f32 = 6370000.;
fn wrap(mut x: f32) -> f32 {
    if x > 180. {
        x -= 360.;
    }
    if x < -180. {
        x += 360.;
    }
    x
}
fn good(q: f32) -> bool {
    q >= 0. && q < 30000.
}
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Projection {
    pub map_proj: i32,
    pub latitude_origin: f32,
    pub longitude_origin: f32,
    pub truelat1: f32,
    pub truelat2: f32,
    pub stand_lon: f32,
    pub dx: f32,
    pub dy: f32,
    pub e_we: usize,
    pub e_sn: usize,
}
#[derive(Clone)]
struct Map {
    p: Projection,
    hemi: f32,
    cone: f32,
    pi: f32,
    pj: f32,
    dlon: f32,
    rsw: f32,
}
impl Map {
    fn new(p: &Projection) -> Result<Self, String> {
        if ![1, 2, 3, 6].contains(&p.map_proj) {
            return Err(format!(
                "WRF observation projection {} is not an ARW projection",
                p.map_proj
            ));
        }
        if p.dx <= 0. || p.dy <= 0. {
            return Err("projection grid spacing must be positive".into());
        }
        let hemi = if p.truelat1 < 0. { -1. } else { 1. };
        let cone = if p.map_proj == 1 {
            if (p.truelat1 - p.truelat2).abs() > 0.1 {
                ((p.truelat1 * RAD).cos().log10() - (p.truelat2 * RAD).cos().log10())
                    / (((45. - p.truelat1.abs() / 2.) * RAD).tan().log10()
                        - ((45. - p.truelat2.abs() / 2.) * RAD).tan().log10())
            } else {
                (p.truelat1.abs() * RAD).sin()
            }
        } else {
            1.
        };
        let mut m = Self {
            p: p.clone(),
            hemi,
            cone,
            pi: 0.,
            pj: 0.,
            dlon: 0.,
            rsw: 0.,
        };
        if p.map_proj == 1 {
            let rsw = m.lc_radius(p.latitude_origin);
            let arg = cone * (wrap(p.longitude_origin - p.stand_lon) * RAD);
            m.pi = hemi - hemi * rsw * arg.sin();
            m.pj = hemi + rsw * arg.cos();
        } else if p.map_proj == 2 {
            let rsw = m.ps_radius(p.latitude_origin);
            let a = (p.longitude_origin - (p.stand_lon + 90.)) * RAD;
            m.pi = 1. - rsw * a.cos();
            m.pj = 1. - hemi * rsw * a.sin();
        } else if p.map_proj == 3 {
            m.dlon = p.dx / (RE * (p.truelat1 * RAD).cos());
            m.rsw = if p.latitude_origin != 0. {
                (0.5 * ((p.latitude_origin + 90.) * RAD)).tan().ln() / m.dlon
            } else {
                0.
            };
        }
        Ok(m)
    }
    fn lc_radius(&self, lat: f32) -> f32 {
        RE / self.p.dx * (self.p.truelat1 * RAD).cos() / self.cone
            * (((90. * self.hemi - lat) * RAD / 2.).tan()
                / ((90. * self.hemi - self.p.truelat1) * RAD / 2.).tan())
            .powf(self.cone)
    }
    fn ps_radius(&self, lat: f32) -> f32 {
        RE / self.p.dx * (lat * RAD).cos() * (1. + self.hemi * (self.p.truelat1 * RAD).sin())
            / (1. + self.hemi * (lat * RAD).sin())
    }
    fn ij(&self, lat: f32, lon: f32) -> (f32, f32) {
        let (i, j) = match self.p.map_proj {
            1 => {
                let rm = self.lc_radius(lat);
                let a = self.cone * (wrap(lon - self.p.stand_lon) * RAD);
                (
                    self.hemi * (self.pi + self.hemi * rm * a.sin()),
                    self.hemi * (self.pj - rm * a.cos()),
                )
            }
            2 => {
                let rm = self.ps_radius(lat);
                let a = (lon - (self.p.stand_lon + 90.)) * RAD;
                (self.pi + rm * a.cos(), self.pj + self.hemi * rm * a.sin())
            }
            3 => (
                1. + wrap(lon - self.p.longitude_origin)
                    / (self.dlon * (180. / std::f32::consts::PI)),
                1. + (0.5 * ((lat + 90.) * RAD)).tan().ln() / self.dlon - self.rsw,
            ),
            _ => {
                let mut dl = lon - self.p.longitude_origin;
                if dl < 0. {
                    dl += 360.;
                }
                if dl > 360. {
                    dl -= 360.;
                }
                let li = self.p.dx * 360. / (2. * RE * std::f32::consts::PI);
                let la = self.p.dy * 360. / (2. * RE * std::f32::consts::PI);
                let mut i = dl / li;
                if i <= 0. {
                    i += 360. / li;
                }
                if i > 360. / li {
                    i -= 360. / li;
                }
                (i + 1., (lat - self.p.latitude_origin) / la + 1.)
            }
        };
        (i + 0.5, j + 0.5)
    }
    fn wind(&self, lon: f32, u: Datum, v: Datum) -> (f32, f32) {
        if !good(u.qc) || !good(v.qc) || (u.value == 0. && v.value == 0.) {
            return (MISS, MISS);
        }
        if u.qc != 129. || v.qc != 129. || ![1, 2].contains(&self.p.map_proj) {
            return (u.value, v.value);
        }
        let diff = wrap(self.p.stand_lon - lon);
        let alpha = if self.p.map_proj == 1 {
            diff * self.cone * RAD * self.hemi
        } else {
            diff * RAD * self.hemi
        };
        (
            v.value * alpha.sin() + u.value * alpha.cos(),
            v.value * alpha.cos() - u.value * alpha.sin(),
        )
    }
}
fn rh2r(rh: f32, t: f32, p: f32) -> f32 {
    let esat = 6.1078 * (17.2693882 * (t - 273.16) / (t - 35.86)).exp();
    let rsat = 0.62197 * esat / (p - esat);
    rh * 0.01 * rsat
}
#[derive(Debug, Clone)]
pub struct Level {
    pub time: f32,
    pub i: f32,
    pub j: f32,
    pub k: f32,
    pub vars: [f32; 6],
    pub count: f32,
    pub level: f32,
    pub platform: f32,
    pub elevation: f32,
}
fn platform(r: &Report) -> (i32, String) {
    let mut p = r.platform.as_bytes().to_vec();
    p.resize(16, b' ');
    let n = r.name.as_bytes();
    if n.get(1..9) == Some(b"PROFILER") {
        p[6..14].copy_from_slice(b"PROFILER");
    }
    if n.get(1..6) == Some(b"ACARS") {
        p[6..11].copy_from_slice(b"ACARS");
    }
    if n.get(0..7) == Some(b"SATWNDS") {
        p[0..11].copy_from_slice(b"SATWNDS    ");
    }
    if n.get(0..8) == Some(b"CLASS DA") {
        p[6..10].copy_from_slice(b"TEMP");
    }
    let code = if &p[6..11] == b"METAR" {
        1
    } else if &p[6..11] == b"SPECI" {
        2
    } else if &p[6..10] == b"SHIP" {
        3
    } else if &p[6..11] == b"SYNOP" {
        4
    } else if &p[6..10] == b"TEMP" {
        5
    } else if &p[6..11] == b"PILOT" {
        6
    } else if &p[..7] == b"SATWNDS" || &p[6..11] == b"SATWI" || &p[6..11] == b"ACARS" {
        7
    } else if &p[..4] == b"SAMS" {
        8
    } else if &p[6..14] == b"PROFILER" {
        9
    } else {
        99
    };
    (code, String::from_utf8(p).unwrap())
}
fn prepare(r: &Report, m: &Map, origin: i64, rovg: f32) -> Vec<Level> {
    let (i, j) = m.ij(r.latitude, r.longitude);
    let (pf, pl) = platform(r);
    if i < 2. || i > (m.p.e_we - 1) as f32 || j < 2. || j > (m.p.e_sn - 1) as f32 {
        return vec![];
    }
    if pf == 4 && pl.get(6..16) == Some("SYNOP PRET") && r.id.get(6..15) == Some("METNET= 3") {
        return vec![];
    }
    let base = Level {
        time: r.hours_from(origin),
        i,
        j,
        k: 1.,
        vars: [99999.; 6],
        count: r.levels() as f32,
        level: 1.,
        platform: pf as f32,
        elevation: r.elevation,
    };
    match &r.measurements {
        Measurements::Surface(v) => {
            let mut o = base; // Native surface records do not write VAROBS(6); initial carrier is zero.
            o.vars[5] = 0.;
            let t = v[3];
            let mut ps = v[7];
            let sl = v[0];
            let mut rh = v[6];
            if ps.qc < 0. && good(sl.qc) && good(t.qc) && sl.value > 90000. {
                let tbar = t.value + 0.5 * r.elevation * 0.0065;
                ps.value = sl.value * (-r.elevation / (rovg * tbar)).exp();
                ps.qc = 0.;
            }
            if ps.qc < 0. && good(t.qc) {
                let tbar = t.value + 0.5 * r.elevation * 0.0065;
                ps.value = 100000. * (-r.elevation / (rovg * tbar)).exp();
                ps.qc = 0.;
            }
            let pok = good(ps.qc) && ps.value > 70000. && ps.value < 105000.;
            o.vars[4] = if pok { ps.value * 0.001 } else { MISS };
            o.vars[2] = if pok && good(t.qc) { t.value } else { MISS };
            (o.vars[0], o.vars[1]) = m.wind(r.longitude, v[4], v[5]);
            if pf == 3 && rh.qc >= 0. && rh.value < 70. {
                rh.qc = MISS;
            }
            o.vars[3] = if good(rh.qc) && good(ps.qc) && good(t.qc) {
                rh2r(rh.value, t.value, ps.value * 0.01)
            } else {
                MISS
            };
            vec![o]
        }
        Measurements::Sounding(rows) => rows
            .iter()
            .enumerate()
            .map(|(n, values)| {
                let mut v = *values;
                for x in &mut v {
                    if x.qc == -777777. {
                        x.qc = 0.;
                    }
                    if x.value == MISS {
                        x.qc = MISS;
                    }
                }
                if pf == 5 && v[3].qc == 256. {
                    v[3] = Datum {
                        value: MISS,
                        qc: MISS,
                    };
                    v[4] = v[3];
                }
                if pf == 6 {
                    v[2] = Datum {
                        value: MISS,
                        qc: MISS,
                    };
                    v[5] = v[2];
                }
                let mut o = base.clone();
                o.k = -99.;
                o.level = (n + 1) as f32;
                o.vars[2] = if good(v[2].qc) && (good(v[0].qc) || good(v[1].qc)) {
                    v[2].value
                } else {
                    MISS
                };
                o.vars[4] = if good(v[0].qc) { v[0].value } else { MISS };
                if o.vars[4] >= 0. {
                    o.vars[4] *= 0.001;
                }
                o.vars[5] = if good(v[1].qc) { v[1].value } else { MISS };
                if o.vars[4] > 0. && o.vars[4] <= 8. {
                    for x in [2, 3, 4, 5] {
                        v[x] = Datum {
                            value: MISS,
                            qc: MISS,
                        };
                    }
                }
                (o.vars[0], o.vars[1]) = m.wind(r.longitude, v[3], v[4]);
                o.vars[3] = if good(v[5].qc) && good(v[0].qc) && good(v[2].qc) {
                    rh2r(v[5].value, v[2].value, v[0].value * 0.01)
                } else {
                    MISS
                };
                o
            })
            .collect(),
    }
}
/// Preserve native report order and duplicate replacement, while all projection,
/// thermodynamic conversion and wind work is shared across bounded workers.
pub fn admit(
    reports: &[Report],
    p: &Projection,
    origin: i64,
    rovg: f32,
    workers: usize,
) -> Result<Vec<Level>, String> {
    let m = Map::new(p)?;
    let workers = workers
        .max(1)
        .min(thread::available_parallelism().map_or(1, usize::from))
        .min(reports.len().max(1));
    let chunk = reports.len().div_ceil(workers).max(1);
    let groups = thread::scope(|scope| {
        let tasks: Vec<_> = reports
            .chunks(chunk)
            .map(|rs| {
                let map = &m;
                scope.spawn(move || {
                    rs.iter()
                        .map(|r| prepare(r, map, origin, rovg))
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        let mut all = Vec::new();
        for task in tasks {
            all.extend(
                task.join()
                    .map_err(|_| "observation admission worker panicked")?,
            );
        }
        Ok::<_, &str>(all)
    })
    .map_err(str::to_owned)?;
    let mut out: Vec<Level> = Vec::new();
    let mut last = -999.;
    let mut index: HashMap<(u32, u32, u32, u32), Vec<usize>> = HashMap::new();
    let bits = |x: f32| if x == 0. { 0 } else { x.to_bits() };
    for mut new in groups {
        if new.is_empty() {
            continue;
        }
        let q = &new[0];
        let mut discard = false;
        let key = (
            bits(q.time),
            bits(q.i),
            bits(q.j),
            bits(if q.platform <= 4. { 1. } else { q.platform }),
        );
        for n in index.get(&key).into_iter().flatten().copied() {
            let old = &out[n];
            if old.time != q.time || old.i != q.i || old.j != q.j || old.platform == 99. {
                continue;
            }
            if !((old.platform <= 4. && q.platform <= 4.) || old.platform == q.platform) {
                continue;
            }
            if q.k == 1. && old.k == q.k
                || q.platform == 7. && q.count == 1. && old.count == 1. && old.vars[4] == q.vars[4]
            {
                out[n].vars = q.vars;
                discard = true;
                break;
            }
            if q.k == -99. && old.platform == q.platform {
                if q.platform == 5. || q.platform == 9. || q.platform == 6. && q.count >= old.count
                {
                    let count = old.count as usize;
                    for x in &mut out[n..n + count] {
                        x.vars = [MISS; 6];
                        x.platform = 99.;
                    }
                    break;
                }
                if q.platform == 6. && q.count < old.count {
                    discard = true;
                    break;
                }
            }
        }
        if discard {
            continue;
        }
        if q.time < last {
            return Err(
                "OBS_DOMAIN observations are not chronological after native admission".into(),
            );
        }
        last = q.time;
        index
            .entry(key)
            .or_default()
            .extend(out.len()..out.len() + new.len());
        out.append(&mut new);
    }
    Ok(out)
}
