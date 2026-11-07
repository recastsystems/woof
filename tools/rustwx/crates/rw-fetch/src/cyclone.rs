//! Government best-track acquisition and normalization. No atmospheric data enters here.
use chrono::{Duration, NaiveDateTime, Utc};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs,
    io::Read,
    path::{Path, PathBuf},
    time::Instant,
};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;
const LIMIT: u64 = 512 * 1024 * 1024;
fn hash(b: &[u8]) -> String {
    format!("{:x}", Sha256::digest(b))
}
fn time(s: &str) -> Result<NaiveDateTime> {
    for f in ["%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S", "%Y%m%d%H%M"] {
        if let Ok(t) = NaiveDateTime::parse_from_str(s, f) {
            return Ok(t);
        }
    }
    Err(format!("invalid UTC best-track time: {s}").into())
}
fn stamp(t: NaiveDateTime) -> String {
    t.format("%Y-%m-%dT%H:%M:%SZ").to_string()
}
/// RFC 4180 fields including quoted commas, escaped quotes and embedded newlines.
fn csv(text: &str) -> Result<Vec<Vec<String>>> {
    let mut rows = Vec::new();
    let mut row = Vec::new();
    let mut field = String::new();
    let mut quoted = false;
    let mut chars = text.trim_start_matches('\u{feff}').chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '"' if quoted && chars.peek() == Some(&'"') => {
                field.push('"');
                chars.next();
            }
            '"' => quoted = !quoted,
            ',' if !quoted => {
                row.push(field.trim().to_string());
                field.clear();
            }
            '\n' if !quoted => {
                row.push(field.trim().to_string());
                field.clear();
                if row.iter().any(|f| !f.is_empty()) {
                    rows.push(std::mem::take(&mut row));
                } else {
                    row.clear();
                }
            }
            '\r' if !quoted => {}
            _ => field.push(c),
        }
    }
    if quoted {
        return Err("unterminated CSV quote".into());
    }
    if !field.is_empty() || !row.is_empty() {
        row.push(field.trim().to_string());
        rows.push(row);
    }
    Ok(rows)
}
// Split only outside RFC 4180 quotes, so archives are parsed with bounded
// record memory rather than retaining millions of allocated CSV fields.
fn records(text: &str) -> impl Iterator<Item = Result<Vec<String>>> + '_ {
    let mut offset = 0;
    std::iter::from_fn(move || {
        if offset >= text.len() {
            return None;
        }
        let begin = offset;
        let mut quoted = false;
        while offset < text.len() {
            let c = text.as_bytes()[offset];
            offset += 1;
            if c == b'"' {
                quoted = !quoted;
            }
            if c == b'\n' && !quoted {
                break;
            }
        }
        Some(csv(&text[begin..offset]).and_then(|mut rows| {
            if rows.len() > 1 {
                Err("multiple CSV records".into())
            } else {
                Ok(rows.pop().unwrap_or_default())
            }
        }))
    })
    .filter(|r| r.as_ref().map_or(true, |v| !v.is_empty()))
}
fn coordinate(s: &str, scale: f64, latitude: bool) -> Result<f64> {
    let (s, sign) = match s.as_bytes().last() {
        Some(b'S' | b'W') => (&s[..s.len() - 1], -1.),
        Some(b'N' | b'E') => (&s[..s.len() - 1], 1.),
        _ => (s, 1.),
    };
    let v = s.parse::<f64>()? * scale * sign;
    if !v.is_finite() || v.abs() > if latitude { 90. } else { 360. } {
        return Err("best-track coordinate out of range".into());
    }
    Ok(if latitude {
        v
    } else {
        (v + 180.).rem_euclid(360.) - 180.
    })
}
fn optional(s: &str, scale: f64, zero_valid: bool) -> Option<f64> {
    s.parse::<f64>()
        .ok()
        .filter(|v| v.is_finite() && if zero_valid { *v >= 0. } else { *v > 0. })
        .map(|v| v * scale)
}
fn track(id: &str, name: &str) -> Value {
    json!({"member_id":"best-track","track_id":id,"storm_id":id,"name":name,"aliases":[id,name],"weight":1.0,"fixes":[]})
}
fn fix(
    t: NaiveDateTime,
    lat: f64,
    lon: f64,
    p: Option<f64>,
    w: Option<f64>,
    flags: Vec<&str>,
) -> Value {
    json!({"valid":stamp(t),"lead_hours":0.,"lat":lat,"lon":lon,"mslp_pa":p,"max_wind10_ms":w,"flags":flags})
}
fn insert(tracks: &mut BTreeMap<String, Value>, id: &str, name: &str, f: Value) -> Result<()> {
    let t = tracks
        .entry(id.to_string())
        .or_insert_with(|| track(id, name));
    if !name.is_empty() {
        let aliases = t["aliases"].as_array_mut().unwrap();
        if !aliases.contains(&json!(name)) {
            aliases.push(json!(name));
        }
        if !["UNNAMED", "NOT_NAMED"].contains(&name) {
            t["name"] = json!(name);
        }
    }
    let fixes = t["fixes"].as_array_mut().unwrap();
    if let Some(previous) = fixes.iter().find(|x| x["valid"] == f["valid"]) {
        // ATCF repeats the same center for every wind radius. Conflicting centers are corrupt input.
        for key in ["lat", "lon", "mslp_pa", "max_wind10_ms"] {
            if previous[key] != f[key] {
                return Err(
                    format!("conflicting best-track fixes: {id} {} {key}", f["valid"]).into(),
                );
            }
        }
    } else {
        fixes.push(f);
    }
    Ok(())
}
fn parse(raw: &[u8], fmt: &str) -> Result<Vec<Value>> {
    parse_window(raw, fmt, None)
}
fn parse_window(raw: &[u8], fmt: &str, window: Option<NaiveDateTime>) -> Result<Vec<Value>> {
    let text = std::str::from_utf8(raw)?;
    let mut rows = records(text);
    let mut tracks = BTreeMap::new();
    if fmt == "ibtracs" || fmt == "jtwc" {
        let header = rows.next().ok_or("empty IBTrACS CSV")??;
        let col = |key: &str| {
            header
                .iter()
                .position(|s| s == key)
                .ok_or_else(|| format!("IBTrACS missing {key}"))
        };
        for key in [
            "SID",
            "NAME",
            "ISO_TIME",
            "LAT",
            "LON",
            "USA_ATCF_ID",
            "USA_WIND",
            "USA_PRES",
            "USA_AGENCY",
        ] {
            col(key)?;
        }
        let columns: BTreeMap<&str, usize> = header
            .iter()
            .enumerate()
            .map(|(i, k)| (k.as_str(), i))
            .collect();
        for row in rows {
            let row = row?;
            let get = |key: &str| -> &str {
                columns
                    .get(key)
                    .and_then(|&i| row.get(i))
                    .map(String::as_str)
                    .unwrap_or("")
            };
            if get("SID").is_empty() || get("SID") == " " || !get("ISO_TIME").contains('-') {
                continue;
            }
            if let Some(start) = window {
                // One year either side covers cross-year and Southern Hemisphere seasons.
                let year: i32 = get("ISO_TIME")[..4].parse()?;
                let target: i32 = start.format("%Y").to_string().parse()?;
                if (year - target).abs() > 1 {
                    continue;
                }
            }
            let agency = get("USA_AGENCY").to_ascii_lowercase();
            if fmt == "jtwc" && !agency.contains("jtwc") {
                continue;
            }
            let sid = get("SID");
            let id = if get("USA_ATCF_ID").is_empty() {
                sid
            } else {
                get("USA_ATCF_ID")
            };
            let t = time(get("ISO_TIME"))?;
            let (lat_key, lon_key) = if fmt == "jtwc" {
                // A JTWC selection must retain that agency's positions, not
                // the cross-agency IBTrACS position composite.
                col("USA_LAT")?;
                col("USA_LON")?;
                if get("USA_LAT").is_empty() || get("USA_LON").is_empty() {
                    continue;
                }
                ("USA_LAT", "USA_LON")
            } else {
                ("LAT", "LON")
            };
            let lat = coordinate(get(lat_key), 1., true)?;
            let lon = coordinate(get(lon_key), 1., false)?;
            let f = fix(
                t,
                lat,
                lon,
                optional(get("USA_PRES"), 100., false),
                optional(get("USA_WIND"), 1852. / 3600., true),
                vec![],
            );
            // SID is stable even when USA_ATCF_ID appears only after genesis.
            insert(&mut tracks, sid, get("NAME"), f)?;
            let t = tracks.get_mut(sid).unwrap();
            let a = t["aliases"].as_array_mut().unwrap();
            for alias in [id, get("NAME")] {
                if !alias.is_empty() && !a.contains(&json!(alias)) {
                    a.push(json!(alias));
                }
            }
            if !["", "UNNAMED", "NOT_NAMED"].contains(&get("NAME")) {
                t["name"] = json!(get("NAME"));
            }
            if id != sid {
                t["atcf_id"] = json!(id);
            }
            t["wind_definition"] =
                json!("USA agency 1-minute sustained wind, not model instantaneous wind");
        }
    } else if fmt == "hurdat2" {
        let mut id = String::new();
        let mut name = String::new();
        let mut remaining = 0usize;
        for row in rows {
            let row = row?;
            if row.len() >= 3
                && row[0].len() == 8
                && row[0][..2].chars().all(|c| c.is_ascii_alphabetic())
            {
                if remaining != 0 {
                    return Err("HURDAT2 header count exceeds available fixes".into());
                }
                id = row[0].to_uppercase();
                name = row[1].clone();
                remaining = row[2].parse()?;
                continue;
            }
            if row.len() < 8 || remaining == 0 {
                return Err("invalid HURDAT2 record or count".into());
            }
            remaining -= 1;
            let t = time(&format!("{}{}", row[0], row[1]))?;
            let flags = if row[2] == "L" {
                vec!["landfall"]
            } else {
                vec![]
            };
            insert(
                &mut tracks,
                &id,
                &name,
                fix(
                    t,
                    coordinate(&row[4], 1., true)?,
                    coordinate(&row[5], 1., false)?,
                    optional(&row[7], 100., false),
                    optional(&row[6], 1852. / 3600., true),
                    flags,
                ),
            )?;
        }
        if remaining != 0 {
            return Err("truncated HURDAT2 storm".into());
        }
    } else if fmt == "atcf" {
        for row in rows {
            let row = row?;
            if row.len() < 10 || row[4] != "BEST" {
                continue;
            }
            if row[2].len() != 10 || !row[2].bytes().all(|b| b.is_ascii_digit()) {
                return Err("invalid ATCF BEST timestamp".into());
            }
            let id = format!("{}{:0>2}{}", row[0], row[1], &row[2][..4]).to_uppercase();
            let minute: i64 = if row[3].is_empty() {
                0
            } else {
                row[3].parse()?
            };
            if !(0..60).contains(&minute) {
                return Err("ATCF BEST minutes must be 0..59".into());
            }
            if row.get(5).is_some_and(|tau| !tau.is_empty() && tau != "0") {
                continue;
            }
            let t = time(&format!("{}00", row[2]))? + Duration::minutes(minute);
            let name = row.get(27).map(String::as_str).unwrap_or("");
            insert(
                &mut tracks,
                &id,
                name,
                fix(
                    t,
                    coordinate(&row[6], 0.1, true)?,
                    coordinate(&row[7], 0.1, false)?,
                    optional(&row[9], 100., false),
                    optional(&row[8], 1852. / 3600., true),
                    vec![],
                ),
            )?;
        }
    } else {
        return Err(format!("unknown best-track format: {fmt}").into());
    }
    let mut out: Vec<Value> = tracks.into_values().collect();
    for t in &mut out {
        if let Some(id) = t.get("atcf_id").cloned() {
            t["track_id"] = id.clone();
            t["storm_id"] = id;
        }
        t["fixes"]
            .as_array_mut()
            .unwrap()
            .sort_by_key(|f| f["valid"].as_str().unwrap().to_string());
    }
    if out.is_empty() {
        return Err("best-track source contains no fixes".into());
    }
    Ok(out)
}
struct Fetch {
    client: reqwest::blocking::Client,
    cache: PathBuf,
    receipts: Vec<Value>,
}
impl Fetch {
    fn new(cache: &Path) -> Result<Self> {
        fs::create_dir_all(cache)?;
        Ok(Self {
            client: reqwest::blocking::Client::builder()
                .timeout(std::time::Duration::from_secs(180))
                .connect_timeout(std::time::Duration::from_secs(20))
                .user_agent("ArWen cyclone-track acquisition")
                .build()?,
            cache: cache.into(),
            receipts: vec![],
        })
    }
    fn get(&mut self, url: &str) -> Result<Vec<u8>> {
        let p = self.cache.join(hash(url.as_bytes()));
        let fresh = p
            .metadata()
            .and_then(|m| m.modified())
            .ok()
            .and_then(|t| t.elapsed().ok())
            .is_some_and(|d| d.as_secs() < 3600);
        let raw = if fresh {
            fs::read(&p)?
        } else {
            let mut result = None;
            let mut error = String::new();
            for attempt in 0..3 {
                match self
                    .client
                    .get(url)
                    .send()
                    .and_then(|r| r.error_for_status())
                {
                    Ok(r) => {
                        let mut b = Vec::new();
                        if let Err(e) = r.take(LIMIT + 1).read_to_end(&mut b) {
                            error = e.to_string();
                            if attempt < 2 {
                                std::thread::sleep(std::time::Duration::from_secs(1 << attempt));
                            }
                            continue;
                        }
                        if b.len() as u64 > LIMIT {
                            return Err(format!(
                                "best-track download exceeds {LIMIT} bytes: {url}"
                            )
                            .into());
                        }
                        result = Some(b);
                        break;
                    }
                    Err(e) => {
                        error = e.to_string();
                        if attempt < 2 {
                            std::thread::sleep(std::time::Duration::from_secs(1 << attempt));
                        }
                    }
                }
            }
            let b = result.ok_or_else(|| format!("best-track download failed: {url}: {error}"))?;
            let tmp = p.with_extension(format!("{}.tmp", std::process::id()));
            fs::write(&tmp, &b)?;
            fs::rename(tmp, &p)?;
            b
        };
        self.receipts
            .push(json!({"url":url,"sha256":hash(&raw),"bytes":raw.len(),"cache_hit":fresh}));
        if raw.starts_with(&[0x1f, 0x8b]) {
            let mut b = Vec::new();
            flate2::read::GzDecoder::new(&raw[..])
                .take(LIMIT + 1)
                .read_to_end(&mut b)?;
            if b.len() as u64 > LIMIT {
                return Err("decompressed best-track archive exceeds limit".into());
            }
            Ok(b)
        } else {
            Ok(raw)
        }
    }
}
fn active(t: &Value, start: NaiveDateTime) -> bool {
    let f = t["fixes"].as_array().unwrap();
    f.first()
        .and_then(|v| v["valid"].as_str())
        .is_some_and(|s| time(s).is_ok_and(|x| x <= start))
        && f.last()
            .and_then(|v| v["valid"].as_str())
            .is_some_and(|s| time(s).is_ok_and(|x| x >= start))
}
fn matches(t: &Value, s: &str) -> bool {
    s == "auto"
        || t["aliases"]
            .as_array()
            .unwrap()
            .iter()
            .any(|a| a.as_str().is_some_and(|x| x.eq_ignore_ascii_case(s)))
}
fn acquire(
    req: &Value,
    fetch: &mut Fetch,
    table: &Value,
    start: NaiveDateTime,
    storm: &str,
) -> Result<(Vec<Value>, String)> {
    let source = req["source"].as_str().unwrap_or("auto");
    if let Some(input) = req["input"].as_str() {
        let raw = fs::read(input)?;
        fetch
            .receipts
            .push(json!({"path":input,"sha256":hash(&raw),"bytes":raw.len()}));
        let decoded = if raw.starts_with(&[0x1f, 0x8b]) {
            let mut b = Vec::new();
            flate2::read::GzDecoder::new(&raw[..])
                .take(LIMIT + 1)
                .read_to_end(&mut b)?;
            if b.len() as u64 > LIMIT {
                return Err("decompressed best-track input exceeds limit".into());
            }
            b
        } else {
            raw
        };
        let format = req["format"].as_str().unwrap_or_else(|| {
            if decoded.starts_with(b"SID,") {
                "ibtracs"
            } else if String::from_utf8_lossy(&decoded)
                .lines()
                .take(10)
                .any(|l| l.split(',').nth(4).is_some_and(|x| x.trim() == "BEST"))
            {
                "atcf"
            } else {
                "hurdat2"
            }
        });
        return Ok((
            parse_window(
                &decoded,
                format,
                req["parse_window"]
                    .as_bool()
                    .unwrap_or(false)
                    .then_some(start),
            )?,
            format.into(),
        ));
    }
    if source == "auto" || source == "nhc" {
        let row = &table["nhc"];
        let index = fetch.get(row["index_url"].as_str().ok_or("NHC index_url")?)?;
        let mut ids = BTreeMap::new();
        for r in csv(std::str::from_utf8(&index)?)? {
            if r.len() < 21 {
                continue;
            }
            let name = &r[0];
            let id = &r[20];
            if storm != "auto"
                && !storm.eq_ignore_ascii_case(name)
                && !storm.eq_ignore_ascii_case(id)
            {
                continue;
            }
            let began = time(&format!("{}00", r[11]))?;
            let ended = time(&format!("{}00", r[12]))?;
            if began <= start && start <= ended {
                ids.insert(id.clone(), name.clone());
            }
        }
        if !ids.is_empty() {
            let mut out = vec![];
            for (id, name) in ids {
                let url = row["archive_url"]
                    .as_str()
                    .ok_or("NHC archive_url")?
                    .replace("{year}", &id[4..])
                    .replace("{id}", &id.to_lowercase());
                let raw = match fetch.get(&url) {
                    Ok(b) => b,
                    Err(e) => {
                        if id[4..].parse::<i32>()?
                            != Utc::now()
                                .date_naive()
                                .format("%Y")
                                .to_string()
                                .parse::<i32>()?
                        {
                            return Err(e);
                        }
                        fetch.get(
                            &row["active_url"]
                                .as_str()
                                .ok_or("NHC active_url")?
                                .replace("{id}", &id.to_lowercase()),
                        )?
                    }
                };
                for mut t in parse(&raw, "atcf")? {
                    t["name"] = json!(name);
                    t["aliases"] = json!([id, name]);
                    out.push(t);
                }
            }
            return Ok((out, "nhc".into()));
        }
        if source == "nhc" {
            return Err(format!("no NHC storm matching {storm} active at {}", stamp(start)).into());
        }
        // An explicit non-NHC selection and area searches use the global archive.
        // Auto needs all basins, so fall through when NHC supplies no candidates.
    }
    if source == "hurdat2" {
        let row = &table["hurdat2"];
        let base = row["index_url"].as_str().ok_or("HURDAT index_url")?;
        let html = String::from_utf8(fetch.get(base)?)?;
        let mut urls = vec![];
        for token in html.split('"') {
            if token.starts_with("/data/hurdat/hurdat2-") && token.ends_with(".txt") {
                let u = format!("https://www.nhc.noaa.gov{token}");
                if !urls.contains(&u) {
                    urls.push(u);
                }
            }
        }
        if urls.is_empty() {
            return Err("NHC data page has no HURDAT2 text links".into());
        }
        let mut out = vec![];
        for u in urls {
            out.extend(parse(&fetch.get(&u)?, "hurdat2")?);
        }
        return Ok((out, "hurdat2".into()));
    }
    if !["auto", "ibtracs", "jtwc"].contains(&source) {
        return Err(format!("unknown track source {source}").into());
    }
    let raw = fetch.get(table["ibtracs"]["url"].as_str().ok_or("IBTrACS url")?)?;
    Ok((
        parse_window(
            &raw,
            if source == "jtwc" { "jtwc" } else { "ibtracs" },
            Some(start),
        )?,
        if source == "jtwc" { "jtwc" } else { "ibtracs" }.into(),
    ))
}
pub fn run(args: &[String]) -> Result<()> {
    if args.len() != 3 || args[1] != "--request" {
        return Err("usage: rw_fetch cyclone-tracks --request JSON".into());
    }
    let req: Value = serde_json::from_slice(&fs::read(&args[2])?)?;
    let start = time(
        req["start"]
            .as_str()
            .ok_or("start UTC timestamp required")?,
    )?;
    let storm = req["storm"].as_str().unwrap_or("auto");
    let cache = Path::new(req["cache"].as_str().ok_or("cache required")?);
    let mut fetch = Fetch::new(cache)?;
    let began = Instant::now();
    let table = &req["sources"];
    let (mut tracks, format) = acquire(&req, &mut fetch, table, start, storm)?;
    tracks.retain(|t| matches(t, storm) && (req["input"].is_string() || active(t, start)));
    if tracks.is_empty() {
        return Err(format!("no best track matching {storm} at {}", stamp(start)).into());
    }
    for t in &mut tracks {
        for f in t["fixes"].as_array_mut().unwrap() {
            f["lead_hours"] =
                json!((time(f["valid"].as_str().unwrap())? - start).num_seconds() as f64 / 3600.);
        }
    }
    let doc = json!({"schema":"gpuwm.tc-tracks.v1","tracker":{"backend":"observed-best-track","source":format,
        "role":"observed setup guidance only; not live nest steering or forecast forcing",
        "wind_definition":"USA/NHC/JTWC 1-minute sustained wind; differs from model instantaneous 10m wind",
        "sources":table,"acquisition":fetch.receipts,
        "parallelism":"bounded HTTP, Rust parsing; one source stream at a time"},
        "members":[{"member_id":"best-track","source":"observed-best-track","init":stamp(start)}],"tracks":tracks});
    let output = Path::new(req["output"].as_str().ok_or("output required")?);
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    let tmp = output.with_extension(format!("{}.tmp", std::process::id()));
    fs::write(&tmp, serde_json::to_vec_pretty(&doc)?)?;
    fs::rename(tmp, output)?;
    println!(
        "{}",
        json!({"path":output,"sha256":hash(&fs::read(output)?),"tracks":doc["tracks"].as_array().unwrap().len(),"seconds":began.elapsed().as_secs_f64()})
    );
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn atcf_units_duplicates_minutes() {
        let row = "AL, 01, 2020010100, 30, BEST, 0, 123N, 1799E, 100, 950, HU, 34\n";
        let t = parse(format!("{row}{row}").as_bytes(), "atcf").unwrap();
        let f = &t[0]["fixes"][0];
        assert_eq!(t[0]["fixes"].as_array().unwrap().len(), 1);
        assert_eq!(f["valid"], "2020-01-01T00:30:00Z");
        assert_eq!(f["lat"], 12.3);
        assert_eq!(f["mslp_pa"], 95000.);
        assert!((f["max_wind10_ms"].as_f64().unwrap() - 51.4444444444).abs() < 1e-7);
    }
    #[test]
    fn hurdat_missing_and_landfall() {
        let t = parse(
            b"AL012020, TEST, 1,\n20200101, 0000, L, HU, 12.3S, 179.9W, -999, -999,\n",
            "hurdat2",
        )
        .unwrap();
        assert_eq!(t[0]["fixes"][0]["lat"], -12.3);
        assert!(t[0]["fixes"][0]["mslp_pa"].is_null());
        assert_eq!(t[0]["fixes"][0]["flags"], json!(["landfall"]));
    }
    #[test]
    fn ibtracs_identity_quoted_name_and_missing_usa() {
        let b=b"SID,NAME,ISO_TIME,LAT,LON,USA_ATCF_ID,USA_WIND,USA_PRES,USA_AGENCY,USA_LAT,USA_LON\n , , , , , , , , , , \n2020001S,\"TEST, STORM\",2020-01-01 00:00:00,-11,180,SH012020,50,980,jtwc_sh,-12,181\n";
        let t = parse(b, "jtwc").unwrap();
        assert_eq!(t[0]["name"], "TEST, STORM");
        assert_eq!(t[0]["fixes"][0]["lon"], -179.);
        assert!(t[0]["aliases"]
            .as_array()
            .unwrap()
            .contains(&json!("2020001S")));
    }
    #[test]
    fn conflicts_refused() {
        assert!(parse(b"AL, 01, 2020010100, , BEST, 0, 123N, 200W, 50, 990\nAL, 01, 2020010100, , BEST, 0, 124N, 200W, 50, 990\n","atcf").is_err());
    }
}

fn distance(a: &Value, b: &Value) -> Option<f64> {
    let (la, lo, lb, ln) = (
        a["lat"].as_f64()?,
        a["lon"].as_f64()?,
        b["lat"].as_f64()?,
        b["lon"].as_f64()?,
    );
    let (la, lb) = (la.to_radians(), lb.to_radians());
    let h = ((lb - la) / 2.).sin().powi(2)
        + la.cos() * lb.cos() * ((ln - lo).to_radians() / 2.).sin().powi(2);
    Some(12742. * h.clamp(0., 1.).sqrt().asin())
}
fn observed_at(fixes: &[Value], valid: &str) -> Result<Option<Value>> {
    if let Some(f) = fixes.iter().find(|f| f["valid"] == valid) {
        return Ok(Some(f.clone()));
    }
    let target = time(valid)?;
    for pair in fixes.windows(2) {
        let (a, b) = (&pair[0], &pair[1]);
        let (ta, tb) = (
            time(a["valid"].as_str().ok_or("best valid")?)?,
            time(b["valid"].as_str().ok_or("best valid")?)?,
        );
        if !(ta < target && target < tb) {
            continue;
        }
        if tb - ta > Duration::hours(6) {
            return Ok(None);
        }
        let fraction = (target - ta).num_seconds() as f64 / (tb - ta).num_seconds() as f64;
        let vector = |v: &Value| -> Option<[f64; 3]> {
            let lat = v["lat"].as_f64()?.to_radians();
            let lon = v["lon"].as_f64()?.to_radians();
            Some([lat.cos() * lon.cos(), lat.cos() * lon.sin(), lat.sin()])
        };
        let (Some(u), Some(v)) = (vector(a), vector(b)) else {
            return Ok(None);
        };
        let angle = u
            .iter()
            .zip(v)
            .map(|(x, y)| x * y)
            .sum::<f64>()
            .clamp(-1., 1.)
            .acos();
        if std::f64::consts::PI - angle < 1e-8 {
            return Err("antipodal best-track interpolation".into());
        }
        let (wa, wb) = if angle < 1e-8 {
            (1. - fraction, fraction)
        } else {
            (
                ((1. - fraction) * angle).sin() / angle.sin(),
                (fraction * angle).sin() / angle.sin(),
            )
        };
        let xyz: [f64; 3] = std::array::from_fn(|i| wa * u[i] + wb * v[i]);
        let mut f = json!({"valid":valid,"lat":xyz[2].atan2(xyz[0].hypot(xyz[1])).to_degrees(),
            "lon":xyz[1].atan2(xyz[0]).to_degrees(),"interpolation":"great-circle; at most 6h bracket"});
        for k in ["mslp_pa", "max_wind10_ms"] {
            f[k] = match (a[k].as_f64(), b[k].as_f64()) {
                (Some(x), Some(y)) => json!(x + fraction * (y - x)),
                _ => Value::Null,
            };
        }
        return Ok(Some(f));
    }
    Ok(None)
}
pub fn score(args: &[String]) -> Result<()> {
    if args.len() != 3 || args[1] != "--request" {
        return Err("usage: rw_fetch cyclone-score --request JSON".into());
    }
    let r: Value = serde_json::from_slice(&fs::read(&args[2])?)?;
    let members = r["members"].as_array().ok_or("members")?;
    let tracks = r["tracks"].as_array().ok_or("tracks")?;
    let best = r["best"]["fixes"].as_array().ok_or("best fixes")?;
    let limit = r["association_max_distance_km"]
        .as_f64()
        .ok_or("association limit")?;
    let mut results = serde_json::Map::new();
    for m in members {
        let id = m["member_id"].as_str().ok_or("member_id")?;
        let candidates: Vec<&Value> = tracks.iter().filter(|t| t["member_id"] == id).collect();
        let mut valids = std::collections::BTreeSet::new();
        for t in &candidates {
            for f in t["fixes"].as_array().ok_or("track fixes")? {
                if let Some(v) = f["valid"].as_str() {
                    valids.insert(v.to_string());
                }
            }
        }
        if let Some(frames) = m["frames"].as_array() {
            for f in frames {
                if let Some(v) = f["valid"].as_str() {
                    valids.insert(v.to_string());
                }
            }
        }
        let mut rows = vec![];
        let mut selected: Option<String> = None;
        let mut unobserved = 0;
        for valid in valids {
            let Some(b) = observed_at(best, &valid)? else {
                unobserved += 1;
                continue;
            };
            // Bind identity once. A missed or displaced cyclone cannot be replaced
            // by a different candidate that later passes near the observed storm.
            if selected.is_none() {
                let mut near: Vec<(f64, String)> = candidates
                    .iter()
                    .flat_map(|t| {
                        t["fixes"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .filter(|f| f["valid"] == valid)
                            .filter_map(|f| {
                                distance(f, &b)
                                    .filter(|d| *d <= limit)
                                    .map(|d| (d, t["track_id"].as_str().unwrap().to_string()))
                            })
                    })
                    .collect();
                near.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
                selected = near.first().map(|(_, id)| id.clone());
            }
            let f = selected
                .as_ref()
                .and_then(|sid| candidates.iter().find(|t| t["track_id"] == sid.as_str()))
                .and_then(|t| {
                    t["fixes"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .find(|f| f["valid"] == valid)
                });
            let km = f.and_then(|f| distance(f, &b));
            let pressure =
                f.and_then(|f| Some((f["mslp_pa"].as_f64()? - b["mslp_pa"].as_f64()?) / 100.));
            let wind =
                f.and_then(|f| Some(f["max_wind10_ms"].as_f64()? - b["max_wind10_ms"].as_f64()?));
            rows.push(json!({"valid":valid,"centre_distance_km":km,"pressure_error_hpa":pressure,
                "wind_error_ms":wind,"track_id":if km.is_some(){selected.clone()}else{None},
                "best_fix":b,"lead_hours":(time(&valid)?-time(m["init"].as_str().ok_or("member init")?)?).num_seconds() as f64/3600.}));
        }
        let errors: Vec<f64> = rows
            .iter()
            .filter_map(|r| r["centre_distance_km"].as_f64())
            .collect();
        let mean = if errors.is_empty() {
            None
        } else {
            Some(errors.iter().sum::<f64>() / errors.len() as f64)
        };
        let rmse = if errors.is_empty() {
            None
        } else {
            Some((errors.iter().map(|x| x * x).sum::<f64>() / errors.len() as f64).sqrt())
        };
        let intensity_errors = |key: &str| {
            let values: Vec<f64> = rows.iter().filter_map(|row| row[key].as_f64()).collect();
            if values.is_empty() {
                json!({"count":0,"bias":null,"rmse":null})
            } else {
                json!({"count":values.len(),
                    "bias":values.iter().sum::<f64>() / values.len() as f64,
                    "rmse":(values.iter().map(|x| x*x).sum::<f64>() / values.len() as f64).sqrt()})
            }
        };
        let pressure_errors = intensity_errors("pressure_error_hpa");
        let wind_errors = intensity_errors("wind_error_ms");
        results.insert(id.into(),json!({"weight":m["weight"],"association":"nearest candidate at first detected common valid time within radius, then fixed track identity; no reassociation",
            "association_max_distance_km":limit,"selected_track_id":selected,"detections":errors.len(),
            "misses":rows.len()-errors.len(),"unobserved_valid_times":unobserved,
            "mean_track_error_km":mean,"rmse_track_error_km":rmse,
            "pressure_errors_hpa":pressure_errors,"wind_errors_ms":wind_errors,
            "maximum_track_error_km":errors.iter().copied().reduce(f64::max),"common_valid_times":rows}));
    }
    println!("{}", Value::Object(results));
    Ok(())
}
