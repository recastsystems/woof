use crate::{
    analysis, api,
    input::{self, Frame, Point, Source},
    options,
    render::{self, Job, MemberTrace, Overlay, RenderRequest, SoundingScene},
    store::{self, Res},
};
use rayon::prelude::*;
use rustwx_sounding::SoundingMetadata;
use serde::Deserialize;
use serde_json::json;
use std::{collections::BTreeMap, path::PathBuf, sync::Arc};
#[derive(Deserialize)]
pub struct Request {
    pub sources: Vec<Source>,
    pub points: Vec<Point>,
    pub out: PathBuf,
    #[serde(default = "default_modes")]
    pub modes: Vec<String>,
    #[serde(default = "default_interval")]
    pub interval_minutes: u32,
    #[serde(default)]
    pub settings: options::Q,
    #[serde(default = "default_threads")]
    pub threads: usize,
}
fn default_modes() -> Vec<String> {
    vec!["mean".into(), "all".into()]
}
fn default_interval() -> u32 {
    60
}
fn default_threads() -> usize {
    4
}
fn stamp(t: i64, fmt: &str) -> String {
    chrono::DateTime::from_timestamp(t, 0)
        .unwrap()
        .format(fmt)
        .to_string()
}
fn slug(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}
fn escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}
/// The run's start for the lead: the `init` setting (2026-10-22T00:00:00Z, 2026-10-22T00:00Z or 2026-10-22T00),
/// else the domain's first valid time.
fn parse_init(text: &str) -> Res<i64> {
    let t = text.trim().trim_end_matches('Z');
    for fmt in ["%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"] {
        if let Ok(v) = chrono::NaiveDateTime::parse_from_str(t, fmt) {
            return Ok(v.and_utc().timestamp());
        }
    }
    if let Ok(v) = chrono::NaiveDateTime::parse_from_str(&format!("{t}:00"), "%Y-%m-%dT%H:%M") {
        return Ok(v.and_utc().timestamp());
    }
    Err(format!("init: expected 2026-10-22T00:00Z, got '{text}'"))
}
/// "F026", or "F026:30" for a valid time off the hour.
fn lead_label(valid: i64, init: i64) -> String {
    let minutes = (valid - init).div_euclid(60);
    let (h, m) = (minutes.div_euclid(60), minutes.rem_euclid(60));
    if m == 0 { format!("F{h:03}") } else { format!("F{h:03}:{m:02}") }
}
fn latlon_label(lat: f64, lon: f64) -> String {
    format!(
        "{:.2}{} {:.2}{}",
        lat.abs(),
        if lat < 0.0 { 'S' } else { 'N' },
        lon.abs(),
        if lon < 0.0 { 'W' } else { 'E' }
    )
}
/// What a picture shows, in words: nothing for a single run, "Ensemble mean", "All N members, P10-P90", "Member 3".
fn mode_label(mode: &str, members: usize) -> String {
    match mode {
        "mean" if members > 1 => format!("Ensemble mean ({members} members)"),
        "mean" | "det" => String::new(),
        "all" if members > 1 => format!("All {members} members, P10-P90"),
        "all" => String::new(),
        m => {
            let digits: String = m.chars().rev().take_while(|c| c.is_ascii_digit()).collect::<Vec<_>>().into_iter().rev().collect();
            match digits.parse::<u64>() {
                Ok(n) if !digits.is_empty() => format!("Member {n}"),
                _ => format!("Member {m}"),
            }
        }
    }
}
/// What the sounding says top right when the caller set no `brand`: the model, its shape and its start, from the
/// run's own description with the `model`, `shape`, `source` and `init` settings taking precedence:
/// "WOOF Regional · Init HRRR 23 Oct 2026 12Z", "WOOF Hex · Global 120 km · Init GFS 22 Oct 2026 00Z". A part the run
/// does not state is left out; with nothing stated it is just "WOOF". Never a tool's name.
pub fn brand_from(run: &input::RunInfo, init: Option<i64>) -> String {
    let words = |s: &Option<String>| s.as_deref().map(str::trim).filter(|s| !s.is_empty()).map(str::to_string);
    let mut parts = vec![match words(&run.model) {
        Some(m) if m.starts_with("WOOF") => m,
        Some(m) => format!("WOOF {m}"),
        None => "WOOF".to_string(),
    }];
    if let Some(shape) = words(&run.shape) {
        parts.push(shape);
    }
    let source = words(&run.source).map(|s| s.to_uppercase());
    match (source, init) {
        (Some(src), Some(t)) => parts.push(format!("Init {src} {}", stamp(t, "%-d %b %Y %HZ"))),
        (None, Some(t)) => parts.push(format!("Init {}", stamp(t, "%-d %b %Y %HZ"))),
        (Some(src), None) => parts.push(format!("Init {src}")),
        (None, None) => {}
    }
    parts.join(" · ")
}
/// The `title` setting's {point} {latlon} {valid} {lead} {init} {mode} {domain} filled in.
fn fill(template: &str, fields: &[(&str, &str)]) -> String {
    let mut out = template.to_string();
    for (k, v) in fields {
        out = out.replace(&format!("{{{k}}}"), v);
    }
    out
}
pub fn run(req: Request) -> Res<()> {
    if req.sources.is_empty() || req.points.is_empty() {
        return Err("a sounding needs input files and at least one point".into());
    }
    if req.interval_minutes == 0 {
        return Err("interval_minutes must be positive".into());
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(req.threads.max(1))
        .build()
        .map_err(|e| e.to_string())?;
    let started = std::time::Instant::now();
    let frames: Vec<Frame> = pool
        .install(|| {
            req.sources
                .par_iter()
                .map(|s| input::read(s, &req.points))
                .collect::<Res<Vec<_>>>()
        })?
        .into_iter()
        .flatten()
        .collect();
    let mut groups: BTreeMap<(String, i64), BTreeMap<String, Frame>> = BTreeMap::new();
    let mut origins: BTreeMap<String, i64> = BTreeMap::new();
    let mut roster: BTreeMap<String, std::collections::BTreeSet<String>> = BTreeMap::new();
    // What each domain's files say about the run (the first file that states a part wins).
    let mut runs: BTreeMap<String, input::RunInfo> = BTreeMap::new();
    for f in frames {
        let known = runs.remove(&f.domain).unwrap_or_default().or(&f.run);
        runs.insert(f.domain.clone(), known);
        origins
            .entry(f.domain.clone())
            .and_modify(|t| *t = (*t).min(f.valid))
            .or_insert(f.valid);
        roster
            .entry(f.domain.clone())
            .or_default()
            .insert(f.member.clone());
        let key = (f.domain.clone(), f.valid);
        let member = f.member.clone();
        if groups
            .entry(key)
            .or_default()
            .insert(member.clone(), f)
            .is_some()
        {
            return Err(format!(
                "duplicate input for member {member} at the same valid time; select one history/store inventory"
            ));
        }
    }
    eprintln!(
        "soundings: read {} time groups in {:.2}s using {} CPU workers",
        groups.len(),
        started.elapsed().as_secs_f64(),
        req.threads
    );
    // One picture per (domain, valid time on the interval) x point x mode.
    let pictures = groups
        .keys()
        .filter(|(domain, valid)| (valid - origins[domain]) % (i64::from(req.interval_minutes) * 60) == 0)
        .count()
        * req.points.len()
        * req.modes.len();
    // A GPU pool keeps two threads (one device each); the CPU rasteriser uses the request's CPU workers.
    let cpu_threads = req.threads.max(2);
    let device = render::batch_device(pictures, cpu_threads)?;
    let workers = match device {
        render::RenderDevice::Gpu { .. } => 2,
        render::RenderDevice::Cpu { .. } => cpu_threads.min(pictures.max(1)),
    };
    let tx = render::spawn_pool(device, workers)?;
    let ao = options::parse_analysis(&req.settings)?;
    let display = options::parse_display(&req.settings)?;
    let init_setting = options::qs(&req.settings, "init").map(parse_init).transpose()?;
    // The settings describe the run first; the files fill what they leave out.
    let said = input::RunInfo {
        init: None,
        model: options::qs(&req.settings, "model").map(str::to_string),
        shape: options::qs(&req.settings, "shape").map(str::to_string),
        source: options::qs(&req.settings, "source").map(str::to_string),
    };
    let many_domains = roster.len() > 1;
    let mut pages: BTreeMap<PathBuf, Vec<(String, String)>> = BTreeMap::new();
    let mut receipt = vec![];
    let mut pending = std::collections::VecDeque::new();
    for ((domain, valid), frames) in groups {
        if (valid - origins[&domain]) % (i64::from(req.interval_minutes) * 60) != 0 {
            continue;
        }
        if frames
            .keys()
            .cloned()
            .collect::<std::collections::BTreeSet<_>>()
            != roster[&domain]
        {
            return Err(format!(
                "{domain} {} has missing members; an ensemble band requires matching valid times",
                stamp(valid, "%FT%TZ")
            ));
        }
        let folder = req
            .out
            .join(slug(&domain))
            .join("soundings")
            .join(stamp(valid, "%Y-%m-%d"));
        std::fs::create_dir_all(&folder).map_err(|e| e.to_string())?;
        let names: Vec<String> = frames.keys().cloned().collect();
        for (pi, point) in req.points.iter().enumerate() {
            let raw: Vec<_> = frames.values().map(|f| f.columns[pi].clone()).collect();
            let common = input::common_columns(&raw);
            let meta = SoundingMetadata {
                station_id: point.label.clone(),
                valid_time: stamp(valid, "%Y-%m-%d %H:%MZ"),
                latitude_deg: Some(point.lat),
                longitude_deg: Some(point.lon),
                sample_method: Some(
                    "bilinear; common log-pressure levels for ensemble statistics".into(),
                ),
                ..Default::default()
            };
            let analyze = |r: &store::RawColumn| {
                analysis::analyze(analysis::to_column(r, meta.clone())?, &ao)
            };
            let members: Vec<_> =
                pool.install(|| raw.par_iter().map(analyze).collect::<Res<Vec<_>>>())?;
            for mode in &req.modes {
                // Top left says what the picture is: the point, its valid time and lead, and which member(s).
                let run = said.clone().or(&runs[&domain]);
                // The run's start: the `init` setting, else the start its files state, else the domain's first time.
                let stated = init_setting.or_else(|| run.init.as_deref().and_then(|t| parse_init(t).ok()));
                let init = stated.unwrap_or(origins[&domain]);
                let latlon = latlon_label(point.lat, point.lon);
                let valid_text = stamp(valid, "%a %d %b %Y %H:%MZ");
                let lead = lead_label(valid, init);
                let init_text = stamp(init, "%d %b %Y %HZ");
                let what = mode_label(mode, names.len());
                let fields = [("point", point.label.as_str()), ("latlon", latlon.as_str()), ("valid", valid_text.as_str()),
                    ("lead", lead.as_str()), ("init", init_text.as_str()), ("mode", what.as_str()), ("domain", domain.as_str())];
                // Top right is the model's name and start: the `brand` setting, else the run's own description.
                let brand = match &display.brand {
                    Some(b) => fill(b, &fields),
                    None => brand_from(&run, stated),
                };
                let title = match &display.title {
                    Some(t) => fill(t, &fields),
                    None => {
                        let mut parts = vec![format!("{}  {}", point.label, latlon)];
                        if many_domains {
                            parts.push(domain.clone());
                        }
                        parts.push(format!("Valid {valid_text}  {lead}"));
                        if !what.is_empty() {
                            parts.push(what.clone());
                        }
                        parts.join("  ·  ")
                    }
                };
                let main = if mode == "mean" || mode == "all" {
                    analyze(&store::mean_columns(&common)?)?
                } else {
                    let i = names.iter().position(|n| n == mode).ok_or_else(|| {
                        format!("unknown member {mode}; available: {}", names.join(", "))
                    })?;
                    analyze(&raw[i])?
                };
                let overlay = if mode == "all" {
                    let refs = members.iter().collect::<Vec<_>>();
                    Some(Overlay {
                        members: members
                            .iter()
                            .zip(&names)
                            .map(|(a, n)| MemberTrace {
                                label: n.clone(),
                                p: a.column.pressure_hpa.clone(),
                                t: a.column.temperature_c.clone(),
                                td: a.column.dewpoint_c.clone(),
                                hodo: api::hodo_trace(a),
                            })
                            .collect(),
                        bands: Some(api::bands_from(&common, &refs)),
                        inner_band: false,
                        show_t: display.spag_t,
                        show_td: display.spag_td,
                        alpha: display.spag_alpha,
                        legend: format!(
                            "{} members (thin) · P10-P90 band · thick = ensemble mean",
                            members.len()
                        ),
                    })
                } else {
                    None
                };
                let stem = format!(
                    "{}_{}_{}",
                    slug(&point.label),
                    stamp(valid, "%Y%m%dT%H%M%SZ"),
                    slug(mode)
                );
                let blank: Vec<_> = display.blank.iter().map(|(_, id)| id.clone()).collect();
                let mut data = json!({"schema":"gpuwm-sounding.v1","title":title,"domain":domain,"valid":stamp(valid,"%FT%TZ"),"point":{"label":point.label,"lat":point.lat,"lon":point.lon},"mode":mode,"members_used":names,"column":main.column,"analysis":analysis::analysis_json(&main),"tables":crate::tables::tables_json(&main,&blank)});
                if mode == "all" {
                    data["percentiles"] = store::percentiles5(&common);
                    data["members"]=json!(members.iter().zip(&names).map(|(a,n)|json!({"member":n,"column":a.column,"indices":analysis::summary_json(a),"hodograph_kt":api::hodo_trace(a)})).collect::<Vec<_>>());
                }
                let scene = SoundingScene {
                    patches: api::patches(&display, &main),
                    analysis: Arc::new(main),
                    view: display.view.clone(),
                    layout: display.layout.clone(),
                    corner: display.corner,
                    parcel: display.parcel,
                    title: title.clone(),
                    brand: brand.clone(),
                    style: display.style.clone(),
                    overlay,
                };
                let (reply, rx) = tokio::sync::oneshot::channel();
                tx.send(RenderRequest {
                    job: Arc::new(Job::Sounding(scene)),
                    size: egui::Vec2::new(display.width, display.height),
                    ppp: display.ppp,
                    reply,
                })
                .map_err(|e| e.to_string())?;
                pending.push_back((folder.clone(), stem, title, data, rx));
                // As many pictures in flight as the pool has threads, so every render thread draws.
                if pending.len() >= workers {
                    finish(pending.pop_front().unwrap(), &mut pages, &mut receipt)?;
                }
            }
        }
    }
    while let Some(job) = pending.pop_front() {
        finish(job, &mut pages, &mut receipt)?;
    }
    if receipt.is_empty() {
        return Err("no requested sounding times were available".into());
    }
    for (folder, rows) in pages {
        let mut html = String::from(
            "<!doctype html><meta charset=utf-8><title>Soundings</title><style>body{background:#101018;color:#eee;font:16px sans-serif}img{max-width:100%;}a{color:#8cf}figure{margin:12px 0}</style><h1>Soundings</h1><p>Hourly profiles; mean, members and ensemble P10–P90 spread.</p>",
        );
        for (stem, title) in rows {
            html.push_str(&format!("<figure><figcaption>{} · <a href=\"{}.json\">profile and indices JSON</a></figcaption><a href=\"{}.png\"><img loading=lazy src=\"{}.png\" alt=\"{}\"></a></figure>",escape(&title),stem,stem,stem,escape(&title)));
        }
        std::fs::write(folder.join("index.html"), html).map_err(|e| e.to_string())?;
    }
    std::fs::write(req.out.join("soundings-manifest.json"),serde_json::to_vec_pretty(&json!({"schema":"gpuwm-soundings.v1","pictures":receipt,"wall_seconds":started.elapsed().as_secs_f64()})).unwrap()).map_err(|e|e.to_string())?;
    Ok(())
}

// Two queued windows overlap the independent egui frame/PNG stages without retaining a time series in RAM.
type Pending = (
    PathBuf,
    String,
    String,
    serde_json::Value,
    tokio::sync::oneshot::Receiver<Res<render::Rendered>>,
);
fn finish(
    (folder, stem, title, data, rx): Pending,
    pages: &mut BTreeMap<PathBuf, Vec<(String, String)>>,
    receipt: &mut Vec<serde_json::Value>,
) -> Res<()> {
    let rendered = rx.blocking_recv().map_err(|e| e.to_string())??;
    std::fs::write(folder.join(format!("{stem}.png")), rendered.png).map_err(|e| e.to_string())?;
    std::fs::write(
        folder.join(format!("{stem}.json")),
        serde_json::to_vec_pretty(&data).unwrap(),
    )
    .map_err(|e| e.to_string())?;
    pages
        .entry(folder.clone())
        .or_default()
        .push((stem.clone(), title));
    receipt.push(json!({"png":folder.join(format!("{stem}.png")),"frame_ms":rendered.frame_ms,"encode_ms":rendered.encode_ms}));
    println!("RENDERED {}", folder.join(format!("{stem}.png")).display());
    Ok(())
}

#[cfg(test)]
mod label_tests {
    use super::*;

    #[test]
    fn modes_read_as_words() {
        assert_eq!(mode_label("mean", 1), "");
        assert_eq!(mode_label("det", 1), "");
        assert_eq!(mode_label("mean", 36), "Ensemble mean (36 members)");
        assert_eq!(mode_label("all", 3), "All 3 members, P10-P90");
        assert_eq!(mode_label("member-0003", 3), "Member 3");
        assert_eq!(mode_label("mem012", 3), "Member 12");
    }

    #[test]
    fn lead_counts_from_the_init_setting() {
        let init = parse_init("2026-10-22T00:00Z").unwrap();
        assert_eq!(parse_init("2026-10-22T00").unwrap(), init);
        assert_eq!(lead_label(init + 22 * 3600, init), "F022");
        assert_eq!(lead_label(init + 90 * 60, init), "F001:30");
        assert!(parse_init("tomorrow").is_err());
    }

    #[test]
    fn points_say_hemisphere() {
        assert_eq!(latlon_label(35.39, -97.6), "35.39N 97.60W");
        assert_eq!(latlon_label(-33.87, 151.21), "33.87S 151.21E");
    }

    #[test]
    fn the_label_is_the_run_never_a_tool() {
        let init = parse_init("2026-10-23T12:00Z").ok();
        let run = |m: Option<&str>, sh: Option<&str>, src: Option<&str>| input::RunInfo {
            init: None,
            model: m.map(str::to_string),
            shape: sh.map(str::to_string),
            source: src.map(str::to_string),
        };
        assert_eq!(brand_from(&run(None, None, None), None), "WOOF");
        assert_eq!(brand_from(&run(Some("Regional"), None, Some("hrrr")), init), "WOOF Regional · Init HRRR 23 Oct 2026 12Z");
        assert_eq!(
            brand_from(&run(Some("Hex"), Some("Global 120 km"), Some("GFS")), parse_init("2026-10-22T00").ok()),
            "WOOF Hex · Global 120 km · Init GFS 22 Oct 2026 00Z"
        );
        assert_eq!(brand_from(&run(Some("WOOF Live ensemble"), None, None), None), "WOOF Live ensemble");
        assert_eq!(brand_from(&run(None, None, None), init), "WOOF · Init 23 Oct 2026 12Z");
        assert!(!brand_from(&run(None, None, None), None).contains("sharppy"));
    }

    #[test]
    fn templates_fill_their_fields() {
        assert_eq!(fill("WOOF Hex · {point} {lead}", &[("point", "OKC"), ("lead", "F006")]), "WOOF Hex · OKC F006");
    }
}
