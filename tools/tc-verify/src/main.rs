#[path = "auxiliary.rs"]
mod aux;
mod deck;
mod page;
mod score;
use anyhow::{ensure, Context, Result};
use clap::{Parser, Subcommand};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    io::Read,
    path::{Path, PathBuf},
    time::Instant,
};
#[derive(Parser)]
#[command(
    version,
    about = "Offline tropical cyclone verification; no model state is modified"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}
#[derive(Subcommand)]
enum Command {
    Score {
        #[arg(long)]
        manifest: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value_t = 6)]
        threads: usize,
    },
    Fetch {
        #[arg(long)]
        catalog: PathBuf,
        #[arg(long)]
        cache: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value_t = 6)]
        threads: usize,
    },
    Export {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        storm: String,
        #[arg(long)]
        model: String,
        #[arg(long)]
        init: String,
        #[arg(long)]
        out: PathBuf,
    },
}
#[derive(Clone, Deserialize, Serialize)]
struct Case {
    storm: String,
    best: PathBuf,
    #[serde(default = "atcf")]
    best_format: String,
    #[serde(default)]
    adecks: Vec<PathBuf>,
    #[serde(default)]
    cycles: Vec<String>,
    models: Vec<String>,
    #[serde(default)]
    notes: String,
    #[serde(default)]
    all_statuses: bool,
    #[serde(default = "yes")]
    plots: bool,
    #[serde(default = "horizon")]
    plot_hours: i64,
    #[serde(default)]
    aid_priority: BTreeMap<String, Vec<String>>,
    #[serde(default)]
    metric_models: BTreeMap<String, Vec<String>>,
    #[serde(default)]
    rain: Option<PathBuf>,
    #[serde(default)]
    stations: Option<PathBuf>,
}
fn horizon() -> i64 {
    126
}
fn yes() -> bool {
    true
}
fn atcf() -> String {
    "atcf".into()
}
#[derive(Deserialize)]
struct Manifest {
    cases: Vec<Case>,
}
#[derive(Serialize)]
struct Report {
    schema: String,
    case: Case,
    available_models: Vec<String>,
    missing_models: Vec<String>,
    forecasts: Vec<deck::Fix>,
    best: Vec<deck::Fix>,
    pairs: Vec<score::Pair>,
    individual: Vec<score::Stat>,
    homogeneous: Vec<score::Stat>,
    ri_individual: Vec<score::Ri>,
    ri_homogeneous: Vec<score::Ri>,
    aid_selection: Vec<Value>,
    rain: Option<Value>,
    stations: Option<Value>,
    timings: Value,
    inputs: Vec<Value>,
}
fn hash(path: &Path) -> Result<Value> {
    let b = std::fs::read(path)?;
    Ok(json!({"path":path,"bytes":b.len(),"sha256":format!("{:x}",Sha256::digest(&b))}))
}
fn write(path: &Path, v: &impl Serialize) -> Result<()> {
    std::fs::write(path, serde_json::to_vec_pretty(v)?)?;
    Ok(())
}
fn run_case(case: &Case, base: &Path, out: &Path) -> Result<Report> {
    ensure!(
        case.storm.len() == 8
            && case.storm.is_ascii()
            && case.storm[..2].bytes().all(|b| b.is_ascii_alphabetic())
            && case.storm[2..].bytes().all(|b| b.is_ascii_digit()),
        "storm must be basin, number, year"
    );
    ensure!(
        case.models.iter().collect::<BTreeSet<_>>().len() == case.models.len()
            && case.models.iter().all(|m| !m.is_empty() && m != "BEST"),
        "comparison models must be unique nonempty forecast IDs"
    );
    ensure!(
        case.plot_hours > 0 && case.plot_hours <= 10000,
        "plot_hours must be 1..10000"
    );
    let start = Instant::now();
    let best_path = base.join(&case.best);
    let mut inputs = vec![hash(&best_path)?];
    let best: Vec<_> = deck::read_storm(&best_path, &case.best_format, &case.storm)?
        .into_iter()
        .filter(|x| x.storm.eq_ignore_ascii_case(&case.storm) && x.model == "BEST")
        .collect();
    ensure!(
        !best.is_empty(),
        "{} has no best-track observations",
        case.storm
    );
    ensure!(
        !case.models.is_empty(),
        "explicit comparison models are required"
    );
    let cycles: Vec<_> = case
        .cycles
        .iter()
        .map(|s| deck::time(s))
        .collect::<Result<_>>()?;
    let mut raws = String::new();
    for path in &case.adecks {
        let path = base.join(path);
        inputs.push(hash(&path)?);
        let raw = deck::text(&path)?;
        for line in raw.lines() {
            let r: Vec<_> = line.split(',').map(str::trim).collect();
            if r.len() >= 5
                && (case.models.iter().any(|m| m == r[4])
                    || case.aid_priority.values().flatten().any(|m| m == r[4]))
                && (cycles.is_empty() || cycles.contains(&deck::time(r[2])?))
            {
                raws.push_str(line);
                raws.push('\n');
            }
        }
    }
    let raw_forecasts: Vec<_> = deck::parse_selected(&raws, "atcf", Some(&case.storm))?
        .into_iter()
        .filter(|x| {
            x.storm.eq_ignore_ascii_case(&case.storm)
                && (cycles.is_empty() || cycles.contains(&x.init))
        })
        .collect();
    let mut chosen: BTreeMap<_, (usize, deck::Fix)> = BTreeMap::new();
    let mut aid_selection = vec![];
    for f in raw_forecasts {
        for model in &case.models {
            let priority = case
                .aid_priority
                .get(model)
                .cloned()
                .unwrap_or_else(|| vec![model.clone()]);
            if let Some(rank) = priority.iter().position(|m| m == &f.model) {
                let key = (f.storm.clone(), model.clone(), f.init, f.valid);
                if chosen.get(&key).is_none_or(|(old, _)| rank < *old) {
                    chosen.insert(key, (rank, f.clone()));
                }
            }
        }
    }
    let forecasts:Vec<_>=chosen.into_iter().map(|((_,model,_,_),(_,mut f))|{if case.aid_priority.contains_key(&model){aid_selection.push(json!({"storm":f.storm,"model":model,"source_aid":f.model,"init":f.init,"lead":f.lead}));}f.model=model;f}).collect();
    let parse_s = start.elapsed().as_secs_f64();
    let score_start = Instant::now();
    let available_models: Vec<_> = forecasts
        .iter()
        .map(|f| f.model.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let missing_models = case
        .models
        .iter()
        .filter(|m| !available_models.contains(m))
        .cloned()
        .collect();
    let pairs = score::pairs(&forecasts, &best, !case.all_statuses);
    let individual = score::aggregate(&pairs, &case.models, false);
    let mut homogeneous = score::aggregate(&pairs, &case.models, true);
    for (metric, models) in &case.metric_models {
        if metric == "rain" || metric == "stations" {
            continue;
        }
        ensure!(
            !models.is_empty()
                && models.iter().collect::<BTreeSet<_>>().len() == models.len()
                && models.iter().all(|m| case.models.contains(m)),
            "metric_models must be an explicit unique subset of models"
        );
        homogeneous.retain(|s| &s.metric != metric);
        homogeneous.extend(
            score::aggregate(&pairs, models, true)
                .into_iter()
                .filter(|s| &s.metric == metric),
        );
    }
    let ri_individual = score::ri(&forecasts, &best, &case.models, false, !case.all_statuses);
    let ri_homogeneous = score::ri(&forecasts, &best, &case.models, true, !case.all_statuses);
    let rain = case
        .rain
        .as_ref()
        .map(|p| -> Result<Value> {
            let p = base.join(p);
            inputs.push(hash(&p)?);
            aux::rain(
                serde_json::from_slice(&std::fs::read(p)?)?,
                case.metric_models.get("rain").unwrap_or(&case.models),
            )
        })
        .transpose()?;
    let stations = case
        .stations
        .as_ref()
        .map(|p| -> Result<Value> {
            let p = base.join(p);
            inputs.push(hash(&p)?);
            aux::stations(
                serde_json::from_slice(&std::fs::read(p)?)?,
                case.metric_models.get("stations").unwrap_or(&case.models),
            )
        })
        .transpose()?;
    let mut r = Report {
        aid_selection,
        rain,
        stations,
        schema: "gpuwm.tc-verify.v1".into(),
        case: case.clone(),
        forecasts,
        best,
        pairs,
        individual,
        homogeneous,
        ri_individual,
        ri_homogeneous,
        available_models,
        missing_models,
        inputs,
        timings: json!({"read_parse_s":parse_s,"score_s":score_start.elapsed().as_secs_f64(),"forecast_step_added_ms":0,"added_cost_basis":"structural: separate offline process; no per-step hook, not GPU ON/OFF timing","gpu_used":false,"run_cap_bytes":21474836480_u64}),
    };
    let output = Instant::now();
    page::storm(out, &r)?;
    r.timings["page_write_s"] = json!(output.elapsed().as_secs_f64());
    r.timings["case_wall_s"] = json!(start.elapsed().as_secs_f64());
    write(&out.join(format!("{}.json", case.storm)), &r)?;
    Ok(r)
}
#[derive(Deserialize)]
struct Catalog {
    cases: Vec<Source>,
}
#[derive(Deserialize)]
struct Source {
    storm: String,
    best_urls: Vec<String>,
    #[serde(default = "atcf")]
    best_format: String,
    #[serde(default)]
    adeck_urls: Vec<String>,
    #[serde(default)]
    cycles: Vec<String>,
    models: Vec<String>,
    #[serde(default)]
    metric_models: BTreeMap<String, Vec<String>>,
    #[serde(default)]
    notes: String,
}
fn fetch(url: &str, cache: &Path) -> Result<PathBuf> {
    ensure!(url.starts_with("https://"), "anonymous HTTPS sources only");
    let name = url.rsplit('/').next().unwrap_or("data");
    let file = cache.join(format!("{:x}-{name}", Sha256::digest(url.as_bytes())));
    if file.exists() {
        return Ok(file);
    }
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(120))
        .build()?;
    let mut b = Vec::new();
    client
        .get(url)
        .send()?
        .error_for_status()?
        .take(256 * 1024 * 1024 + 1)
        .read_to_end(&mut b)?;
    ensure!(b.len() <= 256 * 1024 * 1024, "download exceeds 256 MiB cap");
    let part = file.with_extension(format!(
        "part-{}-{}",
        std::process::id(),
        rayon::current_thread_index().unwrap_or(0)
    ));
    std::fs::write(&part, &b)?;
    std::fs::rename(part, &file)?;
    write(
        &file.with_extension("receipt.json"),
        &json!({"url":url,"bytes":b.len(),"sha256":format!("{:x}",Sha256::digest(&b)),"fetched_utc":chrono::Utc::now().to_rfc3339()}),
    )?;
    Ok(file)
}
fn export(input: &Path, storm: &str, model: &str, init: &str, out: &Path) -> Result<()> {
    ensure!(
        storm.len() == 8 && storm.is_ascii(),
        "storm must be ATCF basin/number/year"
    );
    ensure!(
        !model.is_empty() && model.len() <= 4 && model.chars().all(|c| c.is_ascii_alphanumeric()),
        "model must be 1-4 alphanumeric characters"
    );
    let v: Value = serde_json::from_slice(&std::fs::read(input)?)?;
    let rows = v
        .as_array()
        .or_else(|| v.get("fixes").and_then(Value::as_array))
        .or_else(|| v.get("rows").and_then(Value::as_array))
        .context(
            "input must be an array or {fixes: [...]} with valid, lat, lon, vmax_kt, mslp_hpa",
        )?;
    let t0 = deck::time(init)?;
    let mut s = String::new();
    for r in rows {
        let t = deck::time(r["valid"].as_str().context("valid UTC time required")?)?;
        ensure!(
            (t - t0) % 3600 == 0 && t >= t0,
            "ATCF requires nonnegative integer lead hours"
        );
        let f = r.get("tracker_match").unwrap_or(r);
        let lat = f["lat"]
            .as_f64()
            .or_else(|| f["latitude_deg"].as_f64())
            .context("latitude missing")?;
        let lon = f["lon"]
            .as_f64()
            .or_else(|| f["longitude_deg"].as_f64())
            .context("longitude missing")?;
        ensure!(lat.abs() <= 90. && lon.abs() <= 180., "invalid coordinate");
        let mut base = vec![
            storm[..2].to_uppercase(),
            storm[2..4].into(),
            deck::stamp(t0),
            "03".into(),
            model.to_uppercase(),
            ((t - t0) / 3600).to_string(),
            format!(
                "{:03}{}",
                (lat.abs() * 10.).round() as i64,
                if lat >= 0. { "N" } else { "S" }
            ),
            format!(
                "{:04}{}",
                (lon.abs() * 10.).round() as i64,
                if lon >= 0. { "E" } else { "W" }
            ),
            format!(
                "{:.0}",
                f["vmax_kt"]
                    .as_f64()
                    .or_else(|| f["wind10_ms"].as_f64().map(|x| x * 3600. / 1852.))
                    .unwrap_or(0.)
            ),
            format!(
                "{:.0}",
                f["mslp_hpa"]
                    .as_f64()
                    .or_else(|| f["mslp_pa"].as_f64().map(|x| x / 100.))
                    .unwrap_or(0.)
            ),
            r["status"].as_str().unwrap_or("").into(),
        ];
        if let Some(radii) = r["radii_nmi"].as_object().filter(|r| !r.is_empty()) {
            for (rad, q) in radii {
                let mut b = base.clone();
                b.extend([rad.clone(), "NEQ".into()]);
                ensure!(
                    q.as_array().is_some_and(|x| x.len() == 4),
                    "radius needs NE SE SW NW"
                );
                for x in q.as_array().unwrap() {
                    b.push(
                        x.as_f64()
                            .map(|x| format!("{x:.0}"))
                            .unwrap_or("-999".into()),
                    );
                }
                s += &format!("{}\n", b.join(", "));
            }
        } else {
            base.extend(["0".into(), "".into()]);
            s += &format!("{}\n", base.join(", "));
        }
    }
    std::fs::write(out, s)?;
    Ok(())
}
fn main() -> Result<()> {
    std::hint::black_box(concat!("GPUWM_BRIDGE_SOURCE_REV=",env!("GPUWM_BRIDGE_SOURCE_REV")));
    let cli = Cli::parse();
    match cli.command {
        Command::Score {
            manifest,
            out,
            threads,
        } => {
            ensure!(threads > 0, "threads must be positive");
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build_global()?;
            std::fs::create_dir_all(&out)?;
            let m: Manifest = serde_json::from_slice(&std::fs::read(&manifest)?)?;
            let start = Instant::now();
            ensure!(
                m.cases
                    .iter()
                    .map(|c| &c.storm)
                    .collect::<BTreeSet<_>>()
                    .len()
                    == m.cases.len(),
                "one case per storm per report directory required"
            );
            let reports: Vec<_> = m
                .cases
                .par_iter()
                .map(|c| run_case(c, manifest.parent().unwrap_or(Path::new(".")), &out))
                .collect::<Result<_>>()?;
            let all: Vec<_> = reports.iter().flat_map(|r| r.pairs.clone()).collect();
            // Cross-storm table retains each case's declared cohort; no accidental all-model intersection.
            let stats: Vec<_> = reports
                .iter()
                .flat_map(|r| {
                    r.homogeneous
                        .iter()
                        .map(|s| json!({"storm":r.case.storm,"cohort":s.cohort,"score":s}))
                })
                .collect();
            let mut sums: BTreeMap<_, (usize, f64, f64, f64)> = BTreeMap::new();
            for r in &reports {
                for s in &r.homogeneous {
                    let mut cohort = s.cohort.clone();
                    cohort.sort();
                    let e = sums
                        .entry((cohort, s.model.clone(), s.lead, s.metric.clone()))
                        .or_default();
                    e.0 += s.n;
                    e.1 += s.mean * s.n as f64;
                    e.2 += s.mae * s.n as f64;
                    e.3 += s.rmse.powi(2) * s.n as f64;
                }
            }
            let pooled: Vec<_> = sums
                .into_iter()
                .map(
                    |((cohort, model, lead, metric), (n, sum, abs, sq))| score::Stat {
                        cohort,
                        model,
                        lead,
                        metric,
                        n,
                        mean: sum / n as f64,
                        mae: abs / n as f64,
                        rmse: (sq / n as f64).sqrt(),
                    },
                )
                .collect();
            write(
                &out.join("summary.json"),
                &json!({"schema":"gpuwm.tc-verify.summary.v1","cases":stats,"pooled_by_cohort":pooled,"pair_count":all.len(),"wall_s":start.elapsed().as_secs_f64(),"threads":threads}),
            )?;
            page::index(&out, &reports)?;
            println!(
                "{} storms, {} pairs, {:.3} s; {}",
                reports.len(),
                all.len(),
                start.elapsed().as_secs_f64(),
                out.join("index.html").display()
            );
        }
        Command::Fetch {
            catalog,
            cache,
            out,
            threads,
        } => {
            ensure!(threads > 0, "threads must be positive");
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build_global()?;
            std::fs::create_dir_all(&cache)?;
            let cache = std::fs::canonicalize(cache)?;
            let c: Catalog = serde_json::from_slice(&std::fs::read(catalog)?)?;
            let entries: Vec<_> = c
                .cases
                .par_iter()
                .map(|s| -> Result<(Case, Value)> {
                    let mut errors = vec![];
                    let mut best = None;
                    for url in &s.best_urls {
                        match fetch(url, &cache) {
                            Ok(p) => {
                                best = Some(p);
                                break;
                            }
                            Err(e) => errors.push(format!("{url}: {e}")),
                        }
                    }
                    let best = best.with_context(|| format!("{} no truth: {errors:?}", s.storm))?;
                    let mut adecks = vec![];
                    for url in &s.adeck_urls {
                        match fetch(url, &cache) {
                            Ok(p) => adecks.push(p),
                            Err(e) => errors.push(format!("{url}: {e}")),
                        }
                    }
                    Ok((
                        Case {
                            storm: s.storm.clone(),
                            best,
                            best_format: s.best_format.clone(),
                            adecks,
                            cycles: s.cycles.clone(),
                            models: s.models.clone(),
                            notes: s.notes.clone(),
                            all_statuses: false,
                            plots: true,
                            plot_hours: 126,
                            aid_priority: BTreeMap::new(),
                            metric_models: s.metric_models.clone(),
                            rain: None,
                            stations: None,
                        },
                        json!({"storm":s.storm,"unavailable":errors}),
                    ))
                })
                .collect::<Result<_>>()?;
            write(
                &out,
                &json!({"cases":entries.iter().map(|e|&e.0).collect::<Vec<_>>()}),
            )?;
            write(
                &out.with_extension("fetch.json"),
                &entries.iter().map(|e| &e.1).collect::<Vec<_>>(),
            )?;
        }
        Command::Export {
            input,
            storm,
            model,
            init,
            out,
        } => export(&input, &storm, &model, &init, &out)?,
    }
    Ok(())
}
