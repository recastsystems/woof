use crate::{deck, Report};
use anyhow::Result;
use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
};
fn esc(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}
fn shell(title: &str, body: &str) -> String {
    format!("<!doctype html><html lang=en><meta charset=utf-8><meta name=viewport content='width=device-width'><title>{}</title><style>body{{font:15px system-ui;margin:2rem auto;max-width:1280px;padding:0 1rem;color:#172b40;background:#f5f8fa}}table{{border-collapse:collapse;background:white;font-size:13px}}td,th{{border:1px solid #cbd5df;padding:.4rem;text-align:right}}th{{background:#dce8f0}}.scroll{{overflow:auto}}img{{width:100%;background:white}}a{{color:#005a95}}details{{margin:1rem 0}}.note{{padding:1rem;background:#fff2cc}}h1,h2{{color:#123b56}}</style><h1>{}</h1>{}</html>",esc(title),esc(title),body)
}
fn table<T: serde::Serialize>(rows: &[T]) -> String {
    if rows.is_empty() {
        return "<p>No matched samples. Missing data are not scored as zero.</p>".into();
    }
    let rows: Vec<_> = rows
        .iter()
        .map(|r| serde_json::to_value(r).unwrap())
        .collect();
    let keys: Vec<_> = rows[0].as_object().unwrap().keys().cloned().collect();
    let mut s = "<div class=scroll><table><tr>".to_string();
    for k in &keys {
        s += &format!("<th>{}</th>", esc(k));
    }
    s += "</tr>";
    for r in rows {
        s += "<tr>";
        for k in &keys {
            let v = &r[k];
            let value = if v.is_null() {
                "NA".into()
            } else if let Some(x) = v.as_f64() {
                format!("{x:.2}")
            } else {
                v.as_str().map(str::to_string).unwrap_or(v.to_string())
            };
            s += &format!("<td>{}</td>", esc(&value));
        }
        s += "</tr>";
    }
    s += "</table></div>";
    s
}
const COLORS: [&str; 12] = [
    "#111111", "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#8c510a", "#5e3c99", "#e7298a",
    "#66a61e", "#a6761d", "#666666", "#17becf",
];
fn chart(r: &Report, init: i64) -> String {
    let fs: Vec<_> = r
        .forecasts
        .iter()
        .filter(|f| f.init == init && f.lead <= r.case.plot_hours)
        .collect();
    let max = fs.iter().map(|f| f.lead).max().unwrap_or(r.case.plot_hours);
    let mut series: BTreeMap<String, Vec<&deck::Fix>> = BTreeMap::new();
    series.insert(
        "BEST".into(),
        r.best
            .iter()
            .filter(|f| f.valid >= init && f.valid <= init + max * 3600)
            .collect(),
    );
    for f in fs {
        series.entry(f.model.clone()).or_default().push(f);
    }
    let mut ordered: Vec<_> = series.iter().collect();
    ordered.sort_by_key(|(name, _)| (name.as_str() != "BEST", name.as_str()));
    let mut s=format!("<svg xmlns='http://www.w3.org/2000/svg' width='1260' height='480' viewBox='0 0 1260 480'><rect width='1260' height='480' fill='white'/><style>text{{font:12px sans-serif;fill:#172b40}}</style><text x='25' y='22'>{} cycle {} UTC, chart through +{} h</text>",esc(&r.case.storm),deck::stamp(init),max);
    let origin = r.best.iter().find_map(|f| f.lon).unwrap_or(0.);
    for panel in 0..3 {
        let xy = |f: &deck::Fix| -> Option<(f64, f64)> {
            match panel {
                0 => Some((
                    origin + (f.lon? - origin + 180.).rem_euclid(360.) - 180.,
                    f.lat?,
                )),
                1 => Some(((f.valid - init) as f64 / 3600., f.wind?)),
                _ => Some(((f.valid - init) as f64 / 3600., f.pressure?)),
            }
        };
        let pts: Vec<_> = series.values().flatten().filter_map(|f| xy(f)).collect();
        if pts.is_empty() {
            continue;
        }
        let (mut xmin, mut xmax, mut ymin, mut ymax) = (
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
        );
        for (x, y) in pts {
            xmin = xmin.min(x);
            xmax = xmax.max(x);
            ymin = ymin.min(y);
            ymax = ymax.max(y)
        }
        let dx = (xmax - xmin).max(1.);
        let dy = (ymax - ymin).max(1.);
        xmin -= dx * 0.07;
        xmax += dx * 0.07;
        if panel > 0 {
            xmin = 0.;
            xmax = (max as f64).max(1.);
        }
        ymin -= dy * 0.07;
        ymax += dy * 0.07;
        let left = 50. + panel as f64 * 420.;
        let top = 65.;
        let (w, h) = (330., 295.);
        let p = |x: f64, y: f64| {
            (
                left + (x - xmin) / (xmax - xmin) * w,
                top + h - (y - ymin) / (ymax - ymin) * h,
            )
        };
        s += &format!(
            "<text x='{left}' y='48'>{}</text>",
            [
                "Track coordinates (degrees)",
                "Maximum wind (kt, 1 minute)",
                "Minimum pressure (hPa)"
            ][panel]
        );
        for i in 0..5 {
            let k = i as f64 / 4.;
            let x = xmin + k * (xmax - xmin);
            let y = ymin + k * (ymax - ymin);
            let (px, py) = p(x, y);
            s+=&format!("<path d='M {px} {top} v {h} M {left} {py} h {w}' stroke='#dde4eb'/><text x='{px}' y='380'>{x:.1}</text><text x='{}' y='{py}' text-anchor='end'>{y:.0}</text>",left-5.);
        }
        for (j, (name, fixes)) in ordered.iter().copied().enumerate() {
            let color = COLORS[j % COLORS.len()];
            let mut path = String::new();
            let mut last = None;
            for f in fixes {
                if let Some((x, y)) = xy(f) {
                    let (x, y) = p(x, y);
                    let cmd = if last.is_some_and(|v| {
                        f.valid - v
                            <= if (f.valid - init) / 3600 > 72 {
                                24 * 3600
                            } else {
                                12 * 3600
                            }
                    }) {
                        "L"
                    } else {
                        "M"
                    };
                    path += &format!("{cmd}{x:.2},{y:.2} ");
                    last = Some(f.valid);
                    s+=&format!("<circle cx='{x:.2}' cy='{y:.2}' r='2.5' fill='{color}'><title>{} {}</title></circle>",esc(name),deck::stamp(f.valid));
                } else {
                    last = None
                }
            }
            s += &format!(
                "<path d='{path}' fill='none' stroke='{color}' stroke-width='{}'/>",
                if name == "BEST" { 3. } else { 1.7 }
            );
        }
    }
    s+="<text x='155' y='397'>Longitude (degrees east)</text><text x='560' y='397'>Forecast lead (hours)</text><text x='980' y='397'>Forecast lead (hours)</text>";
    for (j, (name, _)) in ordered.iter().copied().enumerate() {
        let x = 25 + (j % 8) * 155;
        let y = 410 + (j / 8) * 23;
        s += &format!(
            "<path d='M{x} {y} h20' stroke='{}' stroke-width='3'/><text x='{}' y='{}'>{}</text>",
            COLORS[j % 12],
            x + 25,
            y + 4,
            esc(name)
        );
    }
    s+="<text x='25' y='468'>Track is a coordinate chart, not a weather-field map. Gaps over 12 h (24 h beyond lead 72) are unconnected. All panels use shared scales across models.</text></svg>";
    s
}
pub fn storm(out: &Path, r: &Report) -> Result<()> {
    let mut b=format!("<p><a href='index.html'>All storms</a> | <a href='{}.json'>Machine-readable evidence</a></p><p class=note>{}</p><p>Requested cohort: {}. Unavailable: {}. NHC status filter: {}. Raw and interpolated aids are separate model IDs.</p>",esc(&r.case.storm),esc(&r.case.notes),esc(&r.case.models.join(", ")),esc(&r.missing_models.join(", ")),!r.case.all_statuses);
    b += &format!(
        "<p>Explicit metric cohorts: {}</p>",
        esc(&serde_json::to_string(&r.case.metric_models)?)
    );
    let mut cycles = r.forecasts.iter().map(|f| f.init).collect::<BTreeSet<_>>();
    for cycle in &r.case.cycles {
        cycles.insert(deck::time(cycle)?);
    }
    for init in cycles.into_iter().filter(|_| r.case.plots) {
        let name = format!("{}-{}.svg", r.case.storm, deck::stamp(init));
        std::fs::write(out.join(&name), chart(r, init))?;
        b+=&format!("<details open><summary>Cycle {} UTC</summary><img src='{name}' alt='Track, wind and pressure across forecast time'></details>",deck::stamp(init));
    }
    b+="<h2>Homogeneous errors by lead</h2><p>Each metric uses the intersection of storm, initialization and lead across every requested model. Mean is signed bias for intensity, pressure, along/cross and radii; track is unsigned. MAE and RMSE are reported separately. Counts may differ between metrics. track_km/nmi use great-circle distance on a 6371.2 km sphere; met_track_km/nmi use the MET-TC planar 60 nmi/degree convention. Along/cross use the MET-TC backward best-track motion tangent and 111.1 km/degree.</p>";
    b += &table(&r.homogeneous);
    b += "<details><summary>Individual samples (not a fair model ranking)</summary>";
    b += &table(&r.individual);
    b+="</details><h2>Rapid intensification</h2><p>Paired endpoint windows, starts 0 to 96 hours. Rates with zero denominator are NA. Overlapping windows are not independent.</p>";
    b += &table(&r.ri_homogeneous);
    b+="<h2>Wind radii</h2><p>NE, SE, SW, NW quadrant metrics at 34/50/64 kt appear in the tables when both decks supply radii. Zero is a physical radius; missing is excluded.</p><h2>Rain, surge and waves</h2><p>Paired observation panels follow when supplied. Absent inputs remain unscored. No field skill is inferred from the track.</p><p>Sources: input hashes and timings in the JSON receipt. Verification does not demonstrate inner-core structure or cold wake.</p>";
    for (name, data) in [("Rain", &r.rain), ("Surge and waves", &r.stations)] {
        b += &format!("<h3>{name}</h3>");
        if let Some(data) = data {
            if let Some(rows) = data["rows"].as_array() {
                b += &table(rows);
            }
        } else {
            b += "<p>Not supplied.</p>";
        }
    }
    std::fs::write(
        out.join(format!("{}.html", r.case.storm)),
        shell(&r.case.storm, &b),
    )?;
    Ok(())
}
pub fn index(out: &Path, reports: &[Report]) -> Result<()> {
    let mut b="<p>Observations are the referee. Homogeneous samples are explicit; absent forecasts never become perfect forecasts.</p><table><tr><th>Storm</th><th>Pairs</th><th>Models present</th><th>Models absent</th></tr>".to_string();
    for r in reports {
        b += &format!(
            "<tr><td><a href='{}.html'>{}</a></td><td>{}</td><td>{}</td><td>{}</td></tr>",
            esc(&r.case.storm),
            esc(&r.case.storm),
            r.pairs.len(),
            esc(&r.available_models.join(", ")),
            esc(&r.missing_models.join(", "))
        );
    }
    b+="</table><p><a href='summary.json'>Summary data, including pooled statistics by identical cohort</a></p>";
    let rows:Vec<_>=reports.iter().flat_map(|r|r.homogeneous.iter().filter(|s|[24,48,72,120].contains(&s.lead)&&["track_nmi","vmax_kt","mslp_hpa"].contains(&s.metric.as_str())).map(|s|serde_json::json!({"storm":r.case.storm,"cohort":s.cohort.join(","),"model":s.model,"lead":s.lead,"metric":s.metric,"n":s.n,"mae":s.mae,"bias":s.mean}))).collect();
    b += "<h2>Common-sample scores across storms</h2>";
    b += &table(&rows);
    std::fs::write(
        out.join("index.html"),
        shell("Tropical cyclone verification", &b),
    )?;
    Ok(())
}
