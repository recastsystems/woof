//! Header and colour bar for a planned canvas (`layout_plan`).
//!
//! The header spans the canvas's content width, not the map's, so a tall
//! map cannot cut its metadata line.  Its rows carry the same facts the
//! lanes already compose, arranged for reading:
//!
//! * row 1: the title with its units, and the valid time with the lead;
//! * row 2: init time, domain and model, and the source label.
//!
//! A frame too narrow for the valid time beside the title (the table's
//! `two_row_min_content_w`) takes a third row.  The colour bar is exactly
//! the map's height beside it, or exactly its width under it, so no bar
//! runs past the map and no unit label floats over it: the units are in
//! the title.

use std::collections::HashMap;
use std::sync::OnceLock;

use image::RgbaImage;
use serde::Deserialize;

use crate::color::Rgba;
use crate::colorbar;
use crate::colormap::{LegendMode, LeveledColormap};
use crate::layout_plan::{BarSide, CanvasPlan, PlanRect};
use crate::presentation::ColorbarPresentation;
use crate::text;

const UNITS_JSON: &str = include_str!("../data/units.json");

#[derive(Deserialize)]
struct UnitsTable {
    display: HashMap<String, String>,
}

fn units_table() -> &'static HashMap<String, String> {
    static TABLE: OnceLock<HashMap<String, String>> = OnceLock::new();
    TABLE.get_or_init(|| {
        serde_json::from_str::<UnitsTable>(UNITS_JSON)
            .expect("data/units.json is compiled in and parsed by a unit test")
            .display
    })
}

/// How a unit spelling reaches the reader (`degF` prints `°F`).  A
/// spelling the table does not list prints as it is, and a display string
/// the chrome font cannot draw falls back to the lane's own spelling.
pub fn display_units(raw: &str) -> String {
    let raw = raw.trim();
    match units_table().get(raw) {
        Some(display) if text::chrome_font_covers(display) => display.clone(),
        _ => raw.to_string(),
    }
}

/// U+2212 for a negative tick when the chrome font has it: a hyphen is
/// shorter than a plus and reads as a dash beside a number.
pub fn signed_label(label: &str) -> String {
    static HAS_MINUS: OnceLock<bool> = OnceLock::new();
    let has_minus = *HAS_MINUS.get_or_init(|| text::chrome_font_covers("\u{2212}"));
    match label.strip_prefix('-') {
        Some(rest) if has_minus => format!("\u{2212}{rest}"),
        _ => label.to_string(),
    }
}

/// The header's text, arranged from the strings the lanes compose.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlanHeaderText {
    /// Title with units, e.g. `2m AGL Temperature (°F)`.
    pub title: String,
    /// `Valid 05/26 17Z | F002`, when the lane gave a valid time.
    pub when: Option<String>,
    /// `Init 05/26 15Z | d01 3 km | WRF` and anything else the lane said.
    pub meta: String,
    /// The provenance label (`source: ArWen`), right-aligned.
    pub right: Option<String>,
}

fn present(text: Option<&str>) -> Option<&str> {
    text.map(str::trim).filter(|text| !text.is_empty())
}

/// A title without its trailing `(d01 3 km)` grid label: a sheet's panel
/// labels name the member, and the domain is already in the sheet header.
pub fn without_domain_label(title: &str) -> String {
    split_domain_label(title).0
}

/// A trailing `(d01 3 km)` grid label: it names the domain, which is
/// metadata, so it moves from the title to the metadata row.
pub(crate) fn split_domain_label(title: &str) -> (String, Option<String>) {
    let trimmed = title.trim();
    if let Some(body) = trimmed.strip_suffix(')') {
        if let Some(open) = body.rfind(" (") {
            let inner = &body[open + 2..];
            let mut chars = inner.chars();
            if chars.next() == Some('d') && chars.next().is_some_and(|ch| ch.is_ascii_digit()) {
                return (body[..open].trim_end().to_string(), Some(inner.to_string()));
            }
        }
    }
    (trimmed.to_string(), None)
}

pub(crate) fn is_lead_token(token: &str) -> bool {
    let mut chars = token.chars();
    chars.next() == Some('F') && chars.next().is_some_and(|ch| ch.is_ascii_digit())
}

impl PlanHeaderText {
    pub fn compose(
        title: Option<&str>,
        units: Option<&str>,
        subtitle_left: Option<&str>,
        subtitle_center: Option<&str>,
        subtitle_right: Option<&str>,
    ) -> Self {
        let (base, domain) = split_domain_label(present(title).unwrap_or(""));
        let units = present(units).map(display_units).filter(|units| !units.is_empty());
        let title = match units {
            Some(units) if !base.contains(&format!("({units})")) && !base.is_empty() => {
                format!("{base} ({units})")
            }
            _ => base,
        };
        let tokens: Vec<&str> = present(subtitle_left)
            .map(|left| left.split(" | ").map(str::trim).filter(|t| !t.is_empty()).collect())
            .unwrap_or_default();
        let valid = tokens.iter().copied().find(|token| token.starts_with("Valid "));
        let lead = tokens.iter().copied().find(|token| is_lead_token(token));
        let when = valid.map(|valid| match lead {
            Some(lead) => format!("{valid} | {lead}"),
            None => valid.to_string(),
        });
        let mut meta: Vec<String> = Vec::new();
        let mut rest: Vec<&str> = Vec::new();
        for token in &tokens {
            let used_in_when = when.is_some() && (Some(*token) == valid || Some(*token) == lead);
            if used_in_when {
                continue;
            }
            if token.starts_with("Init") && meta.is_empty() {
                meta.push(token.to_string());
            } else {
                rest.push(token);
            }
        }
        if let Some(domain) = domain.as_deref() {
            meta.push(domain.to_string());
        }
        for token in rest {
            // `Δx 3 km` (or a resampled mesh's `render grid 15 km`) repeats
            // the spacing the domain label already carries.
            let repeats_domain = token
                .strip_prefix("Δx ")
                .or_else(|| token.strip_prefix("render grid "))
                .is_some_and(|spacing| domain.as_deref().is_some_and(|d| d.ends_with(spacing)));
            if !repeats_domain {
                meta.push(token.to_string());
            }
        }
        if let Some(center) = present(subtitle_center) {
            meta.push(center.to_string());
        }
        Self {
            title,
            when,
            meta: meta.join(" | "),
            right: present(subtitle_right).map(str::to_string),
        }
    }
}

fn ellipsized_px(text: &str, width: u32, size: f32, bold: bool) -> String {
    if text::text_width_px(text, size, bold) <= width {
        return text.to_string();
    }
    let mut chars: Vec<char> = text.chars().collect();
    while !chars.is_empty() {
        chars.pop();
        let candidate: String = chars.iter().collect::<String>().trim_end().to_string() + "...";
        if text::text_width_px(&candidate, size, bold) <= width {
            return candidate;
        }
    }
    String::new()
}

fn draw_on_baseline(
    img: &mut RgbaImage,
    text_value: &str,
    x: i32,
    baseline: f32,
    color: Rgba,
    size: f32,
    bold: bool,
) {
    let top = (baseline - text::ascent_px(size, bold)).round() as i32;
    text::draw_text_px(img, text_value, x, top, color, size, bold);
}

fn draw_right_on_baseline(
    img: &mut RgbaImage,
    text_value: &str,
    right: i32,
    baseline: f32,
    color: Rgba,
    size: f32,
    bold: bool,
) {
    let width = text::text_width_px(text_value, size, bold) as i32;
    draw_on_baseline(img, text_value, right - width, baseline, color, size, bold);
}

/// The title and metadata inks of the active theme's map chrome, for a
/// frame drawn outside the map renderer (a section) that carries the same
/// header.
pub fn header_inks() -> (Rgba, Rgba) {
    let presentation = crate::presentation::RenderPresentation::for_mode_from_env(
        crate::presentation::ProductVisualMode::FilledMeteorology,
    );
    (presentation.chrome.title_color, presentation.chrome.subtitle_color)
}

/// A label strip's text (a sheet member's or a pair leg's name), semibold
/// at `size` px, left-aligned in the strip at `(x, y)` of height `h`.
pub fn draw_strip_label(img: &mut RgbaImage, label: &str, x: u32, y: u32, h: u32, color: Rgba, size: f32) {
    let baseline = y as f32 + h as f32 * 0.72;
    draw_on_baseline(img, label, x as i32, baseline, color, size, true);
}

/// Draw the planned header.  Returns true when any text had to be cut
/// (the layout law says it never should; the caller reports it).
pub fn draw_plan_header(
    img: &mut RgbaImage,
    plan: &CanvasPlan,
    header: &PlanHeaderText,
    title_color: Rgba,
    meta_color: Rgba,
) -> bool {
    if plan.header.h == 0 || plan.header_rows == 0 {
        return false;
    }
    let left = plan.header.x as i32;
    let right = plan.header.right() as i32;
    let width = plan.header.w;
    let gap = (24.0 * plan.scale).round() as u32;
    let title_baseline = plan.pad_top as f32 + plan.title_row_h as f32 * 0.72;
    let row_baseline = |row: u32| {
        plan.pad_top as f32
            + plan.title_row_h as f32
            + plan.meta_row_h as f32 * (row as f32 - 1.0)
            + plan.meta_row_h as f32 * 0.70
    };
    let mut cut = false;

    // Row 1: the title, and the valid time beside it when there is room.
    let when = header.when.as_deref();
    let when_w = when.map(|when| text::text_width_px(when, plan.meta_px, true)).unwrap_or(0);
    let mut when_on_title_row = plan.header_rows == 2 && when.is_some();
    let title_room = |beside: bool| if beside { width.saturating_sub(when_w + gap) } else { width };
    let fits = |size: f32, room: u32| text::text_width_px(&header.title, size, true) <= room;
    let mut title_size = plan.title_px;
    if !fits(title_size, title_room(when_on_title_row)) {
        if fits(plan.title_min_px, title_room(when_on_title_row)) {
            title_size = plan.title_min_px;
        } else if when_on_title_row && fits(plan.title_px, width) {
            when_on_title_row = false;
        } else if when_on_title_row && fits(plan.title_min_px, width) {
            when_on_title_row = false;
            title_size = plan.title_min_px;
        } else {
            title_size = plan.title_min_px;
            when_on_title_row = false;
        }
    }
    let title = ellipsized_px(&header.title, title_room(when_on_title_row), title_size, true);
    cut |= title != header.title;
    draw_on_baseline(img, &title, left, title_baseline, title_color, title_size, true);
    if when_on_title_row {
        if let Some(when) = when {
            draw_right_on_baseline(img, when, right, title_baseline, title_color, plan.meta_px, true);
        }
    }

    // The metadata rows.  A valid time that did not fit on the title row
    // opens the next row; on a three-row header it has that row to itself.
    let right_label = header.right.as_deref();
    let right_w = right_label
        .map(|label| text::text_width_px(label, plan.meta_px, false))
        .unwrap_or(0);
    let meta_rows: Vec<(Option<&str>, String)> = if plan.header_rows >= 3 {
        vec![
            (when.filter(|_| !when_on_title_row), String::new()),
            (None, header.meta.clone()),
        ]
    } else {
        vec![(when.filter(|_| !when_on_title_row), header.meta.clone())]
    };
    for (index, (bold_part, regular_part)) in meta_rows.iter().enumerate() {
        let baseline = row_baseline(index as u32 + 1);
        let reserve = if index == 0 && right_w > 0 { right_w + gap } else { 0 };
        let room = width.saturating_sub(reserve);
        let mut x = left;
        let mut used = 0u32;
        if let Some(bold_part) = bold_part {
            let fitted = ellipsized_px(bold_part, room, plan.meta_px, true);
            cut |= fitted != *bold_part;
            draw_on_baseline(img, &fitted, x, baseline, title_color, plan.meta_px, true);
            used = text::text_width_px(&fitted, plan.meta_px, true) + gap / 2;
            x += used as i32;
        }
        if !regular_part.is_empty() {
            let fitted = ellipsized_px(regular_part, room.saturating_sub(used), plan.meta_px, false);
            cut |= fitted != *regular_part;
            draw_on_baseline(img, &fitted, x, baseline, meta_color, plan.meta_px, false);
        }
        if index == 0 {
            if let Some(label) = right_label {
                draw_right_on_baseline(img, label, right, baseline, meta_color, plan.meta_px, false);
            }
        }
    }
    cut
}

/// A tick label shortened to fit `span` pixels: thousands as `k`.
fn compact_label(label: &str, value: f64, span: u32, size: f32) -> String {
    if text::text_width_px(label, size, false) <= span || value.abs() < 1000.0 {
        return label.to_string();
    }
    let thousands = value / 1000.0;
    let compact = if (thousands - thousands.round()).abs() < 1e-9 {
        format!("{}k", thousands.round() as i64)
    } else {
        format!("{thousands:.1}k")
    };
    signed_label(&compact)
}

/// The colour bar for a planned canvas: the swatches, a tick mark on the
/// label side, and the labels, thinned so neighbours keep the table's
/// spacing.
#[allow(clippy::too_many_arguments)]
pub(crate) fn draw_plan_colorbar(
    img: &mut RgbaImage,
    plan: &CanvasPlan,
    bar: PlanRect,
    side: BarSide,
    cmap: &LeveledColormap,
    mode: LegendMode,
    presentation: ColorbarPresentation,
    ticks: &[f64],
    lo: f64,
    hi: f64,
) {
    match side {
        BarSide::Right => colorbar::draw_vertical_colorbar(
            img, cmap, bar.x, bar.y, bar.w, bar.h, mode, presentation,
        ),
        BarSide::Bottom => {
            colorbar::draw_colorbar(img, cmap, bar.x, bar.y, bar.w, bar.h, mode, presentation)
        }
    }
    let range = hi - lo;
    if !(range > 0.0) || ticks.is_empty() {
        return;
    }
    let tick_color = if presentation.tick_color == Rgba::TRANSPARENT {
        presentation.frame_color
    } else {
        presentation.tick_color
    };
    let size = plan.tick_px;
    let line_h = text::line_height_px(size, false) as i32;
    let tick_len = (4.0 * plan.scale).round().max(2.0) as u32;
    let labels = text::format_tick_labels(ticks);
    let min_spacing = (plan.tick_spacing_px as f64 * 0.6).max(line_h as f64 + 4.0);
    let mut last: Option<f64> = None;
    let mut last_right = i32::MIN / 4;
    for (value, label) in ticks.iter().zip(labels.iter()) {
        let frac = (value - lo) / range;
        if !frac.is_finite() || !(-1e-9..=1.0 + 1e-9).contains(&frac) {
            continue;
        }
        let frac = crate::render::legend_tick_position(cmap, *value, lo, range).clamp(0.0, 1.0);
        let label = compact_label(&signed_label(label), *value, plan.bar_label_span, size);
        let label_w = text::text_width_px(&label, size, false) as i32;
        match side {
            BarSide::Right => {
                let y = bar.y as f64 + (1.0 - frac) * (bar.h.saturating_sub(1)) as f64;
                let py = y.round() as u32;
                for dx in 0..tick_len {
                    put(img, bar.right() + dx, py, tick_color);
                }
                if last.is_some_and(|previous| (previous - y).abs() < min_spacing) {
                    continue;
                }
                last = Some(y);
                let top = (y.round() as i32 - line_h / 2)
                    .clamp(0, img.height() as i32 - line_h);
                text::draw_text_px(
                    img,
                    &label,
                    (bar.right() + plan.bar_tick_gap) as i32,
                    top,
                    presentation.label_color,
                    size,
                    false,
                );
            }
            BarSide::Bottom => {
                let x = bar.x as f64 + frac * (bar.w.saturating_sub(1)) as f64;
                let px = x.round() as u32;
                for dy in 0..tick_len {
                    put(img, px, bar.bottom() + dy, tick_color);
                }
                let left = (x.round() as i32 - label_w / 2)
                    .clamp(bar.x as i32 - 8, bar.right() as i32 + 8 - label_w)
                    .clamp(0, img.width() as i32 - label_w);
                let spaced = last.is_none_or(|previous| (x - previous) >= min_spacing);
                if !spaced || left < last_right + (8.0 * plan.scale) as i32 {
                    continue;
                }
                last = Some(x);
                last_right = left + label_w;
                text::draw_text_px(
                    img,
                    &label,
                    left,
                    (bar.bottom() + plan.bar_tick_gap) as i32,
                    presentation.label_color,
                    size,
                    false,
                );
            }
        }
    }
}

fn put(img: &mut RgbaImage, x: u32, y: u32, color: Rgba) {
    if x < img.width() && y < img.height() {
        img.put_pixel(x, y, color.to_image_rgba());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_units_table_parses_and_every_display_string_is_drawable() {
        let table = units_table();
        assert!(table.len() > 10);
        for (raw, display) in table {
            assert!(
                text::chrome_font_covers(display),
                "{raw} -> {display:?} has a glyph the chrome font cannot draw"
            );
        }
        assert_eq!(display_units("degF"), "°F");
        assert_eq!(display_units("not-a-unit"), "not-a-unit");
    }

    #[test]
    fn a_negative_tick_uses_the_minus_sign_and_zero_is_never_signed() {
        assert_eq!(signed_label("-20"), "\u{2212}20");
        assert_eq!(signed_label("20"), "20");
        let labels = text::format_tick_labels(&[-0.2 + 0.2, 0.2]);
        assert_eq!(signed_label(&labels[0]), "0");
    }

    #[test]
    fn the_header_moves_the_domain_to_metadata_and_puts_units_in_the_title() {
        let header = PlanHeaderText::compose(
            Some("2m AGL Temperature (d01 3 km)"),
            Some("degF"),
            Some("Init 05/26 15Z | F002 | Valid 05/26 17Z | WRF | Δx 3 km"),
            None,
            Some("source: ArWen"),
        );
        assert_eq!(header.title, "2m AGL Temperature (°F)");
        assert_eq!(header.when.as_deref(), Some("Valid 05/26 17Z | F002"));
        assert_eq!(header.meta, "Init 05/26 15Z | d01 3 km | WRF");
        assert_eq!(header.right.as_deref(), Some("source: ArWen"));
    }

    #[test]
    fn a_header_without_a_valid_time_keeps_every_token() {
        let header = PlanHeaderText::compose(Some("Orography"), None, Some("F000 | WRF"), None, None);
        assert_eq!(header.title, "Orography");
        assert_eq!(header.when, None);
        assert_eq!(header.meta, "F000 | WRF");
    }

    #[test]
    fn digits_are_set_in_one_width_so_a_lead_time_does_not_shift() {
        let a = text::text_width_px("F009 | Valid 05/26 17Z", 15.0, true);
        let b = text::text_width_px("F010 | Valid 05/26 18Z", 15.0, true);
        let c = text::text_width_px("F111 | Valid 11/11 11Z", 15.0, true);
        assert_eq!(a, b);
        assert_eq!(b, c);
    }

    #[test]
    fn a_large_tick_label_is_compacted_to_fit_its_gutter() {
        assert_eq!(compact_label("12000", 12000.0, 20, 14.0), "12k");
        assert_eq!(compact_label("1500", 1500.0, 20, 14.0), "1.5k");
        assert_eq!(compact_label("120", 120.0, 20, 14.0), "120");
    }
}
