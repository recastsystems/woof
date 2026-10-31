use rustwx_core::ModelId;

use crate::shared_context::static_title_with_suffix;

use super::types::DirectBatchRequest;

fn dataset_token_from_product(product: &str) -> Option<&str> {
    let token = product.split(['-', '_']).next().unwrap_or(product);
    if is_gdex_dataset_token(token) {
        Some(token)
    } else {
        None
    }
}

pub(super) fn is_gdex_dataset_token(token: &str) -> bool {
    token.len() > 1 && token.starts_with('d') && token[1..].chars().all(|ch| ch.is_ascii_digit())
}

fn native_stat_label_from_product(product: &str) -> Option<String> {
    let token = product
        .rsplit('/')
        .next()
        .unwrap_or(product)
        .trim()
        .to_ascii_lowercase();
    let mut token = token
        .strip_suffix("_3hrly")
        .or_else(|| token.strip_suffix("_hourly"))
        .or_else(|| token.strip_suffix("_1hrly"))
        .unwrap_or(token.as_str())
        .to_string();
    token = token.replace('-', "_");
    for suffix in ["_conus", "_ak", "_hi", "_pr"] {
        if let Some(stripped) = token.strip_suffix(suffix) {
            token = stripped.to_string();
            break;
        }
    }
    let label = match token.as_str() {
        "mean" => "Mean".to_string(),
        "avg" | "avrg" => "Average".to_string(),
        "spread" | "sprd" => "Spread".to_string(),
        "std" | "stddev" | "stdev" => "Std Dev".to_string(),
        "min" | "minimum" => "Min".to_string(),
        "max" | "maximum" => "Max".to_string(),
        "prob" | "probability" => "Probability".to_string(),
        "eas" => "EAS".to_string(),
        "lpmm" => "Localized PMM".to_string(),
        "pmmn" | "pmm" => "Probability-Matched Mean".to_string(),
        "ffri" => "FFRI".to_string(),
        value if value.len() >= 2 && value.starts_with('p') => {
            let digits = &value[1..];
            if digits.chars().all(|ch| ch.is_ascii_digit()) {
                value.to_ascii_uppercase()
            } else {
                return None;
            }
        }
        _ => return None,
    };
    Some(label)
}

pub(super) fn native_stat_label_for_request(
    request: &DirectBatchRequest,
    planned_product: Option<&str>,
) -> Option<String> {
    planned_product
        .and_then(|planned| {
            request
                .product_overrides
                .get(planned)
                .and_then(|product| native_stat_label_from_product(product))
                .or_else(|| native_stat_label_from_product(planned))
        })
        .or_else(|| {
            request
                .product_overrides
                .values()
                .find_map(|product| native_stat_label_from_product(product))
        })
}

fn model_title_prefix(model: ModelId) -> String {
    // A theme may name the model (`text.model_label`).
    rustwx_render::theme::active_theme()
        .model_name(&model.as_str().replace('-', " ").to_ascii_uppercase())
}

pub(super) fn apply_native_stat_title_prefix(
    model: ModelId,
    stat_label: &str,
    base_title: &str,
) -> String {
    let model_label = model_title_prefix(model);
    let stat_prefix = format!("{model_label} {stat_label} ");
    if base_title.starts_with(&stat_prefix) {
        return base_title.to_string();
    }
    let model_prefix = format!("{model_label} ");
    if let Some(without_model) = base_title.strip_prefix(&model_prefix) {
        return format!("{model_label} {stat_label} {without_model}");
    }
    format!("{model_label} {stat_label} {base_title}")
}

fn direct_title_for_request(
    request: &DirectBatchRequest,
    planned_product: Option<&str>,
    base_title: &str,
) -> String {
    let mut title = accumulation_title(base_title, request.accumulation_window);
    if let Some(stat_label) = native_stat_label_for_request(request, planned_product) {
        title = apply_native_stat_title_prefix(request.model, &stat_label, &title);
    }
    // Locally imported output first, and for every model: these frames were
    // never fetched from a source catalog, so the only accurate parenthetical
    // is the grid the importer read out of the file itself.  Deciding this
    // before the model check keeps the claim tied to the batch's declared
    // provenance rather than to which model identity the store happens to
    // carry.
    if request.title_provenance.is_local_import() {
        return static_title_with_suffix(request.title_provenance.apply_to_title(title));
    }
    if request.model != ModelId::WrfGdex || is_local_wrf_netcdf_request(request) {
        return static_title_with_suffix(title);
    }

    let dataset = planned_product
        .and_then(|planned| {
            request
                .product_overrides
                .get(planned)
                .and_then(|product| dataset_token_from_product(product))
                .or_else(|| dataset_token_from_product(planned))
        })
        .or_else(|| {
            request
                .product_overrides
                .values()
                .find_map(|product| dataset_token_from_product(product))
        })
        .unwrap_or("d612005");
    static_title_with_suffix(format!("{title} ({dataset})"))
}

fn accumulation_title(base_title: &str, window: Option<super::types::AccumulationWindow>) -> String {
    let Some(name) = base_title.strip_suffix(", window total") else { return base_title.into(); };
    let Some(window) = window else { return base_title.into(); };
    let (Some(start), Some(end)) = (
        chrono::DateTime::from_timestamp(window.start_unix, 0),
        chrono::DateTime::from_timestamp(window.end_unix, 0),
    ) else { return base_title.into(); };
    let seconds = window.end_unix - window.start_unix;
    if seconds < 0 { return base_title.into(); }
    let stamp = |time: chrono::DateTime<chrono::Utc>| {
        if time.timestamp().rem_euclid(3600) == 0 {
            time.format("%a %-d %b %HZ").to_string()
        } else {
            time.format("%a %-d %b %H:%M:%SZ").to_string()
        }
    };
    if !window.since_initialization && seconds % 3600 == 0 {
        format!("{} h {name} ending {}", seconds / 3600, stamp(end))
    } else {
        format!("{name} since {}", stamp(start))
    }
}

/// Retire the old caller workaround when it repeats the resolved interval.
/// Other annotations and explicit caller titles keep their exact meaning.
pub(super) fn product_annotations(request: &DirectBatchRequest, base_title: &str)
    -> Option<crate::geographic_overlays::PanelAnnotations>
{
    let mut annotations = request.panel_annotations.clone()?;
    if annotations.title.is_none() && base_title.ends_with(", window total") {
        if let (Some(window), Some(suffix)) = (request.accumulation_window, annotations.title_suffix.as_deref()) {
            let caption = accumulation_title(", window total", Some(window))
                .split_whitespace().collect::<Vec<_>>().join(" ");
            if suffix.trim().trim_start_matches(',').trim() == caption {
                annotations.title_suffix = None;
            }
        }
    }
    Some(annotations)
}

/// [`product_annotations`] for one recipe: its own title note
/// (`PanelAnnotations::title_notes`, keyed by recipe slug) goes into the
/// title suffix ahead of any caller suffix; other recipes' notes are dropped.
pub(super) fn recipe_annotations(request: &DirectBatchRequest, slug: &str, base_title: &str)
    -> Option<crate::geographic_overlays::PanelAnnotations>
{
    let mut annotations = product_annotations(request, base_title)?;
    let notes = std::mem::take(&mut annotations.title_notes);
    if let Some(note) = notes.get(slug).map(|note| note.trim()).filter(|note| !note.is_empty()) {
        annotations.title_suffix = Some(match annotations.title_suffix.take() {
            Some(existing) => format!("{note} {}", existing.trim_start()),
            None => note.to_string(),
        });
    }
    Some(annotations)
}

/// Resolve the same exact timestamp spelling accepted by the local importer.
pub fn local_accumulation_window(origin_unix: i64, end_unix: i64, since: Option<&str>)
    -> Result<super::types::AccumulationWindow, String>
{
    let start_unix = match since {
        Some(value) => chrono::NaiveDateTime::parse_from_str(&value.replace(':', "_"), "%Y-%m-%d_%H_%M_%S")
            .map_err(|err| format!("invalid snowfall baseline: {err}"))?.and_utc().timestamp(),
        None => origin_unix,
    };
    Ok(super::types::AccumulationWindow { start_unix, end_unix, since_initialization: since.is_none() })
}

/// Whole-hour stores carry their physical origin in the run's cycle.
pub fn local_accumulation_window_from_cycle(date: &str, cycle_utc: u8, hour: u16, since: Option<&str>)
    -> Result<super::types::AccumulationWindow, String>
{
    let origin = chrono::NaiveDate::parse_from_str(date, "%Y%m%d")
        .map_err(|err| format!("invalid accumulation cycle date: {err}"))?
        .and_hms_opt(u32::from(cycle_utc), 0, 0).ok_or("invalid accumulation cycle hour")?
        .and_utc().timestamp();
    local_accumulation_window(origin, origin + i64::from(hour) * 3600, since)
}

fn is_local_wrf_netcdf_request(request: &DirectBatchRequest) -> bool {
    request
        .subtitle_right_override
        .as_deref()
        .is_some_and(|subtitle| subtitle.to_ascii_lowercase().contains("local wrf netcdf"))
}

pub(super) fn direct_title_for_planned_product(
    request: &DirectBatchRequest,
    planned_product: &str,
    base_title: &str,
) -> String {
    direct_title_for_request(request, Some(planned_product), base_title)
}

pub(super) fn direct_panel_title_for_request(
    request: &DirectBatchRequest,
    base_title: &str,
) -> String {
    direct_title_for_request(request, None, base_title)
}
