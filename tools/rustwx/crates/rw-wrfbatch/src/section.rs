//! `xsec:` products: vertical cross-sections through NATIVE-LEVEL wrfout
//! fields along a great-circle line, drawn by `rustwx-cross-section`.
//!
//! The store keeps 2-D planes and five isobaric sounding volumes; it has no
//! hydrometeors, no vertical velocity and no scalar the model carries only
//! on its own eta levels.  So this family does not go through the store at
//! all: it opens the wrfout files a second time with `wrf-core`, samples
//! the requested 3-D fields at the section's columns (bilinear in grid
//! space), puts every column onto one height ladder (linear in z) and
//! hands the crate a `ScalarSection` with a terrain profile.
//!
//! ## Product grammar
//!
//! ```text
//! xsec:[<alias>:]<fill>[/<overlay>[/<overlay>...]]
//! <term> := <name>['+'<name>...]['@cold']['~log']['='<l1>,<l2>,...['@'<highlight>]]
//! ```
//!
//! * `<name>` is any `wrf-core` `getvar` name (`tk`, `wa`, `height`, `rh`,
//!   ...) or a raw 3-D wrfout variable (`QCLOUD`, `QICE`, `TR17_1`, ...);
//!   `+` sums the named fields, which is how a family of tracers becomes
//!   one fill without a per-case code path.
//! * `@cold` keeps the term only where the temperature is below 0 C
//!   (supercooled liquid is `QCLOUD@cold`); `~log` draws the fill as
//!   `log10` with zeros transparent, for quantities that span decades.
//! * `=l1,l2,...` fixes an overlay's contour levels, `@h` after them
//!   highlights one of them; absent, the levels are chosen from the
//!   field's own range (and by name for the common ones).  On the fill
//!   term the list names the colour bar's range, lowest level to highest
//!   (`QCLOUD=0.01,0.1` draws every frame on 0.01 to 0.1 g kg-1), which
//!   is how a series of cuts is put on one bar.
//! * `<alias>` names the output slug (`xsec_<alias>`), so a long tracer
//!   sum still files under a readable product folder.
//!
//! Mixing ratios (`Q*`, not `QN*`) are shown in g kg-1; everything else in
//! the units `getvar` reports.  Isotherms come from `tk` and are drawn on
//! every section (`--isotherms`), with the highlighted one in the theme's
//! attention ink.
//!
//! The section line comes from `--section lat,lon,lat,lon` or
//! `--section FILE.json` (`{"start": [lat, lon], "end": [lat, lon]}` or a
//! polyline `{"points": [[lat, lon], ...], "extend_km": 20}` whose first
//! and last points are the line, extended by `extend_km` past each end).
//! A radius-height scalar mean uses a JSON section with `center: [lat, lon]`,
//! `radius_km` and optional `azimuths` (default 72). Every bearing must be
//! inside the grid and vertically defined. Vector components are not rotated.
//! Every PNG is named the way the batch lane names its panels, so the
//! Python placement files it under `<domain>/xsec_<slug>/<valid-day>/`.

use std::collections::BTreeMap;
use rayon::prelude::*;
use std::path::{Path, PathBuf};

use rustwx_cross_section as xs;
use rustwx_render::{RenderTheme, Rgba};
use wrf_core::{ComputeOpts, WrfFile, getvar};

use crate::local_import::{parse_utc_timestamp, wrf_run_origin};

/// One term of a section product: a sum of named 3-D fields with modifiers.
#[derive(Debug, Clone, PartialEq)]
pub struct Term {
    pub names: Vec<String>,
    pub cold_only: bool,
    pub log: bool,
    pub levels: Option<Vec<f32>>,
    pub highlight: Option<f32>,
}

impl Term {
    /// `QCLOUD+QICE` -- the term's human label.
    pub fn label(&self) -> String {
        self.names.join("+")
    }
}

/// A parsed `xsec:` product.
#[derive(Debug, Clone, PartialEq)]
pub struct SectionProduct {
    /// The token as typed, for event lines.
    pub token: String,
    pub alias: Option<String>,
    pub fill: Term,
    pub overlays: Vec<Term>,
}

impl SectionProduct {
    /// `xsec_<alias>` or `xsec_<fill>[_<hash>]`: the product half of the
    /// output filename, request-safe and short enough for the placement
    /// grammar.
    pub fn slug(&self) -> String {
        if let Some(alias) = &self.alias {
            return format!("xsec_{}", safe_component(alias));
        }
        let mut safe = safe_component(&self.fill.label());
        safe.truncate(40);
        if self.overlays.is_empty() && safe == safe_component(&self.fill.label()) {
            format!("xsec_{safe}")
        } else {
            format!("xsec_{safe}_{:08x}", fnv(&self.token) as u32)
        }
    }
}

fn safe_component(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    let mut last_underscore = false;
    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
            last_underscore = false;
        } else if !last_underscore {
            out.push('_');
            last_underscore = true;
        }
    }
    out.trim_matches('_').to_string()
}

fn fnv(value: &str) -> u64 {
    value
        .as_bytes()
        .iter()
        .fold(0xcbf2_9ce4_8422_2325u64, |hash, byte| {
            (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
        })
}

/// The product-family prefix.
pub const PREFIX: &str = "xsec:";

/// How many decades below its maximum a `~log` fill stays visible.
pub const LOG_FILL_DECADES: f32 = 6.0;

/// The vertical step every section is put onto, in metres.
///
/// WHAT BREAKAGE THIS PREVENTS (gate law): the family used to
/// interpolate onto a 250 m ladder, and a 1 km field drawn on it came out
/// of the renderer as visible horizontal steps -- the delivered winter
/// sections read as blocks even though the data under them is 1 km.
pub const LADDER_STEP_M: f64 = 100.0;

/// The path is sampled at this fraction of the grid spacing.  Half a cell
/// is the coarsest sampling that cannot alias a feature the grid resolves.
pub const SAMPLES_PER_CELL: f64 = 2.0;

/// Sampling bounds: never fewer than a line can be drawn from, and a cap
/// that bounds the section's own memory on a continental cut.
pub const MIN_SECTION_SAMPLES: usize = 40;
pub const MAX_SECTION_SAMPLES: usize = 4_000;

/// The number of path samples for a line of `length_km` on a grid of
/// `dx_m`: at least two per grid cell, within the bounds above.
pub fn section_sample_count(length_km: f64, dx_m: f64) -> usize {
    let spacing_km = (dx_m / 1000.0) / SAMPLES_PER_CELL;
    if !(spacing_km.is_finite() && spacing_km > 0.0) {
        return MIN_SECTION_SAMPLES;
    }
    ((length_km / spacing_km).ceil() as usize + 1)
        .clamp(MIN_SECTION_SAMPLES, MAX_SECTION_SAMPLES)
}

/// What counts as the field being THERE, rather than merely defined.
///
/// WHAT BREAKAGE THIS PREVENTS (gate law): taking DEFINED for
/// "there" holds every frame open to the ceiling, because a hydrometeor
/// field is defined and zero all the way to the model top and a `~log`
/// fill sits flat on its own floor everywhere the plume is not.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Presence {
    /// A value must be above this to count (a `~log` fill's floor).
    pub above: f32,
    /// ...and at least this far from zero (a linear field's dust, an
    /// overlay's own lowest contour level).
    pub at_least: f32,
}

impl Presence {
    pub fn holds(self, value: f32) -> bool {
        value.is_finite() && value > self.above && value.abs() >= self.at_least
    }

    /// The test for a FILL: everything above a `~log` fill's floor, and for
    /// a linear one everything past a ten-thousandth of its own reach.
    pub fn for_fill(values: &[f32], log_floor: Option<f32>) -> Self {
        if let Some(floor) = log_floor {
            return Self {
                above: floor + 1e-3,
                at_least: 0.0,
            };
        }
        let reach = values
            .iter()
            .filter(|v| v.is_finite())
            .fold(0.0f32, |acc, v| acc.max(v.abs()));
        Self {
            above: f32::NEG_INFINITY,
            at_least: if reach > 0.0 { reach * 1e-4 } else { f32::INFINITY },
        }
    }

    /// The test for an OVERLAY: its own lowest contour level.
    pub fn for_overlay(levels: &[f32], highlight: Option<f32>) -> Self {
        Self {
            above: f32::NEG_INFINITY,
            at_least: levels
                .iter()
                .chain(highlight.iter())
                .map(|level| level.abs())
                .fold(f32::INFINITY, f32::min),
        }
    }
}

/// The highest rung `values` carries the field on.
pub fn signal_rung(values: &[f32], n_points: usize, presence: Presence) -> Option<usize> {
    if n_points == 0 || values.is_empty() {
        return None;
    }
    let rungs = values.len() / n_points;
    (0..rungs).rev().find(|k| {
        values[k * n_points..(k + 1) * n_points]
            .iter()
            .any(|v| presence.holds(*v))
    })
}

/// The ramp with its BOTTOM faded out.
///
/// The bottom of a sequential fill's ramp is the field's own absence, and
/// painting absence in the ramp's first colour floods the panel.  Fading
/// it also removes the last staircase on a section: the plume's edge was a
/// hard NaN boundary that jumped a whole rung between sample columns, and
/// an alpha that falls off through the ramp's first band draws the same
/// edge as a gradient (gpuwm addition).
pub fn fade_ramp_bottom(mut ramp: Vec<xs::Color>) -> Vec<xs::Color> {
    if ramp.len() < 3 {
        return ramp;
    }
    ramp[0] = xs::Color::rgba(ramp[0].r, ramp[0].g, ramp[0].b, 0);
    ramp[1] = xs::Color::rgba(ramp[1].r, ramp[1].g, ramp[1].b, ramp[1].a / 3);
    ramp[2] = xs::Color::rgba(ramp[2].r, ramp[2].g, ramp[2].b, (ramp[2].a / 3) * 2);
    ramp
}

/// How much of its own colour bar a section fill's values must occupy
/// before that bar is allowed to start at zero instead of at the lowest
/// value on the cut.
pub const ZERO_ANCHOR_MIN_BAR_FRACTION: f32 = 0.5;

/// Which rule set the range a fill was drawn on.
///
/// A section's bar is fitted per frame -- to the air the cut holds, and
/// to the rung its own signal reaches -- so a series of cuts of one line
/// is a series of pictures on different bars.  That is the family's
/// practice and it is what makes a shallow cut readable, but it means
/// the picture alone cannot say which bar it was drawn on.  The rule and
/// the two numbers beside it are what the receipt carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FillRangeRule {
    /// A diverging fill, symmetric about zero because zero is its middle.
    Symmetric,
    /// A log fill, whose own minimum IS its floor.
    LogFloor,
    /// A sequential fill carrying values below zero, so zero is not its
    /// floor and the bar starts at the lowest value on the cut.
    CrossesZero,
    /// The zero anchor, kept because the fill's own values occupy at
    /// least [`ZERO_ANCHOR_MIN_BAR_FRACTION`] of the bar it produces.
    ZeroAnchor,
    /// The zero anchor given up: the bar starts at the lowest value on
    /// the cut, so this frame is drawn on this frame's own range.
    OwnMinimum,
    /// The range the product spelled (`QCLOUD=0.01,0.1`): the bar runs
    /// from the lowest named level to the highest on every frame, so a
    /// series of cuts is drawn on one bar, and a frame with no signal
    /// keeps that bar instead of a placeholder.
    Named,
}

impl FillRangeRule {
    /// One token, for the event line and the receipt.
    pub fn label(self) -> &'static str {
        match self {
            Self::Symmetric => "symmetric",
            Self::LogFloor => "log-floor",
            Self::CrossesZero => "crosses-zero",
            Self::ZeroAnchor => "zero-anchor",
            Self::OwnMinimum => "own-minimum",
            Self::Named => "named",
        }
    }
}

/// The range one section fill was drawn over, and how it was chosen.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DrawnFill {
    /// The bottom of the colour bar.
    pub lo: f32,
    /// The top of it.
    pub hi: f32,
    /// Whether the bottom band is the field's ABSENCE rather than the
    /// lowest thing on the cut.  Only absence is faded away.
    pub absence: bool,
    /// The rule that chose the two numbers above.
    pub rule: FillRangeRule,
}

/// The range a section's fill is drawn over, and whether the BOTTOM of
/// that range is the field's absence rather than the lowest thing on the
/// cut.
///
/// Zero is a real floor for most of what a section fills with -- mixing
/// ratio, reflectivity, wind speed -- and starting the bar there is what
/// makes an empty column read as the bottom of the ramp instead of as
/// the middle of it.  It is nothing at all for a field measured from
/// absolute zero: a cut of air temperature in kelvin spans a few tens of
/// degrees somewhere above 240, so a bar that starts at zero spends
/// every band below that on temperatures the atmosphere never reaches
/// and paints the whole cut in one colour.
///
/// THE RULE: keep the zero anchor while the fill's own values still
/// occupy at least [`ZERO_ANCHOR_MIN_BAR_FRACTION`] of the bar that
/// anchor produces, and take the fill's own minimum otherwise -- so a
/// fill is never drawn on a bar it uses less than half of.
///
/// [`DrawnFill::absence`] says whether the ramp's first band is
/// absence, and that is what decides the fade: [`fade_ramp_bottom`]
/// exists to stop absence flooding the panel, so fading a band that
/// holds the coldest or thinnest air on the cut erases real air instead.
///
/// [`DrawnFill::rule`] says which branch below answered, because the
/// bar a cut is drawn on now moves with the frame at BOTH ends and a
/// reader holding two pictures of one line needs the receipt to say so.
pub fn fill_value_range(diverging: bool, log: bool, min: f32, max: f32) -> DrawnFill {
    let drawn = |lo: f32, hi: f32, absence: bool, rule: FillRangeRule| DrawnFill {
        lo,
        hi,
        absence,
        rule,
    };
    if diverging {
        // A diverging ramp's middle IS zero, so the range is symmetric
        // about it or the ramp says the wrong thing about which way the
        // air goes.  Nothing is absent at the bottom of it: that end is
        // the strongest descent on the cut.
        let reach = min.abs().max(max.abs()).max(1e-6);
        return drawn(-reach, reach, false, FillRangeRule::Symmetric);
    }
    if log {
        // The low end is the fill's OWN minimum, which for a log fill is
        // its floor -- the value its absence sits at.  Rounding that
        // down to the decade below put absence a band or two up the
        // ramp, past the faded end, and the panel flooded with the
        // ramp's first colour.
        return drawn(min, max.max(min + 1.0), true, FillRangeRule::LogFloor);
    }
    if min < 0.0 {
        // The fill crosses zero, so zero is not its floor and the bottom
        // of the bar is the lowest value on the cut.
        return drawn(
            min,
            if max > min { max } else { min + 1.0 },
            false,
            FillRangeRule::CrossesZero,
        );
    }
    let top = if max > 0.0 { max } else { 1.0 };
    if max > min && max - min < ZERO_ANCHOR_MIN_BAR_FRACTION * top {
        drawn(min, max, false, FillRangeRule::OwnMinimum)
    } else {
        drawn(0.0, top, true, FillRangeRule::ZeroAnchor)
    }
}

/// Colour bar ticks for a fill drawn in log10 over `lo..hi` (both in
/// log10): one at every whole decade inside the range, labelled with the
/// field's own number (`0.1`, `1`, `10`), and the 2x and 5x steps of each
/// decade as well when fewer than three whole decades fall inside, so a
/// bar under a decade tall still carries numbers.
pub fn log_decade_ticks(lo: f32, hi: f32) -> Vec<(f32, String)> {
    if !(lo.is_finite() && hi.is_finite() && hi > lo) {
        return Vec::new();
    }
    let first = lo.ceil() as i32;
    let last = hi.floor() as i32;
    let whole: Vec<i32> = (first..=last).collect();
    let steps: &[f64] = if whole.len() >= 3 { &[1.0] } else { &[1.0, 2.0, 5.0] };
    let mut ticks = Vec::new();
    for decade in (lo.floor() as i32)..=(hi.ceil() as i32) {
        for step in steps {
            let value = step * 10f64.powi(decade);
            let position = value.log10() as f32;
            if position >= lo - 1e-4 && position <= hi + 1e-4 {
                ticks.push((position.clamp(lo, hi), decade_label(value)));
            }
        }
    }
    ticks
}

/// `0.001`, `0.02`, `5`, `100`, `1000`, and `1e-4` or `2e4` past the
/// decades a reader counts in their head.
fn decade_label(value: f64) -> String {
    if !(1e-3..1e4).contains(&value) {
        let exponent = value.log10().floor() as i32;
        let mantissa = value / 10f64.powi(exponent);
        return format!("{}e{exponent}", mantissa.round() as i64);
    }
    let text = format!("{value:.3}");
    let trimmed = text.trim_end_matches('0').trim_end_matches('.');
    if trimmed.is_empty() {
        "0".to_string()
    } else {
        trimmed.to_string()
    }
}

/// The bar a fill is drawn on when the product names its levels
/// (`QCLOUD=0.01,0.1`): from the lowest named level to the highest, in
/// the units the section draws the field in.
///
/// The grammar admitted a level list on the fill term and then read it
/// only on overlays, so `xsec:QCLOUD=0.01,0.1` drew the same bar as
/// `xsec:QCLOUD`.  A named range is the one thing that makes two cuts
/// of one line comparable without the receipt, so it is honoured: a
/// `~log` fill takes the decades of the named levels (a level at or
/// below zero has no decade and is dropped), a diverging fill keeps zero
/// as its middle and spans the larger reach, and a sequential fill fades
/// its bottom band because below the named floor is, by the caller's
/// own statement, nothing.  A list with no usable level, or one whose
/// levels are all equal, names no range and the frame is fitted as if
/// none were given.
pub fn named_fill_range(levels: &[f32], diverging: bool, log: bool) -> Option<DrawnFill> {
    let usable: Vec<f32> = levels
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .map(|v| if log { v.log10() } else { v })
        .filter(|v| v.is_finite())
        .collect();
    let lo = usable.iter().copied().fold(f32::MAX, f32::min);
    let hi = usable.iter().copied().fold(f32::MIN, f32::max);
    if usable.is_empty() || !(hi > lo) {
        return None;
    }
    if diverging {
        let reach = lo.abs().max(hi.abs());
        return Some(DrawnFill {
            lo: -reach,
            hi: reach,
            absence: false,
            rule: FillRangeRule::Named,
        });
    }
    Some(DrawnFill {
        lo,
        hi,
        absence: true,
        rule: FillRangeRule::Named,
    })
}

/// The ladder top a frame is drawn to: the highest rung carrying signal
/// plus a kilometre of air above it, never less than two kilometres above
/// the reference altitude, never above the caller's `--section-top-km`.
///
/// A section cut to a fixed 10 or 14 km ceiling spends a third of the
/// frame on empty stratosphere and the rest squeezing terrain and plume
/// into what is left.
pub fn fitted_top_m(
    signal_top_m: Option<f64>,
    reference_m: f64,
    ceiling_m: f64,
    step_m: f64,
) -> f64 {
    let wanted = signal_top_m
        .map(|top| top + 1_000.0)
        .unwrap_or(ceiling_m)
        .max(reference_m + 2_000.0);
    let rungs = (wanted / step_m).ceil().max(2.0);
    (rungs * step_m).min(ceiling_m)
}

/// Split a `--products` value into the store lane's spec and the section
/// products.  `all` and the group keywords carry no sections; a
/// comma-separated list may mix both families.
pub fn split_product_spec(spec: &str) -> Result<(String, Vec<SectionProduct>), String> {
    let trimmed = spec.trim();
    if trimmed.eq_ignore_ascii_case("all")
        || ["direct", "derived", "heavy", "windowed"]
            .iter()
            .any(|group| trimmed.eq_ignore_ascii_case(group))
    {
        return Ok((trimmed.to_string(), Vec::new()));
    }
    let mut store = Vec::new();
    let mut sections = Vec::new();
    // A level list inside a section term is comma-separated too
    // (`wa=1,2,5,10@5`), so a token that follows an `xsec:` token whose
    // last term opened a level list and that starts with a level is the
    // list's continuation, not a product: no product slug is only digits,
    // a sign, a point and an `@`.  The continuation may carry the term
    // that closes the list (`0.1/wa` in `xsec:QCLOUD=0.01,0.1/wa`): the
    // comma before the last number is the list's own separator, and a
    // splitter that cut there made `0.1/wa` a store product, which the
    // catalog then refused, so the whole command line was refused for a
    // spelling the grammar admits.
    let mut tokens: Vec<String> = Vec::new();
    for token in trimmed.split(',').map(str::trim).filter(|t| !t.is_empty()) {
        let continues = tokens
            .last()
            .map(|prior| {
                prior.starts_with(PREFIX) && level_list_open(prior) && continues_level_list(token)
            })
            .unwrap_or(false);
        if continues {
            let prior = tokens.last_mut().expect("checked above");
            prior.push(',');
            prior.push_str(token);
        } else {
            tokens.push(token.to_string());
        }
    }
    for token in &tokens {
        if token.starts_with(PREFIX) {
            sections.push(parse_section_product(token)?);
        } else {
            store.push(token.clone());
        }
    }
    Ok((store.join(","), sections))
}

/// True when the token's last term carries an `=` level list that a
/// following numeric token may continue.
fn level_list_open(token: &str) -> bool {
    let last_term = token.rsplit('/').next().unwrap_or(token);
    last_term.contains('=') && !last_term.rsplit('=').next().unwrap_or("").contains('@')
}

/// A token that continues an open level list: a level (`-10`, `0.5`,
/// `10@5`), or the last level of the list followed by the `/` term that
/// closes it (`0.1/wa`, `10@5/tk=-20`).  What follows the first `/` is
/// parsed as terms later; here it only has to exist.
fn continues_level_list(token: &str) -> bool {
    match token.split_once('/') {
        Some((level, rest)) => is_level_token(level) && !rest.trim().is_empty(),
        None => is_level_token(token),
    }
}

/// `-10`, `0.5`, `10@5`: a level, optionally the highlight after it.
fn is_level_token(token: &str) -> bool {
    let (level, highlight) = match token.split_once('@') {
        Some((level, highlight)) => (level, Some(highlight)),
        None => (token, None),
    };
    let numeric = |text: &str| !text.is_empty() && text.trim().parse::<f32>().is_ok();
    numeric(level) && highlight.map_or(true, numeric)
}

/// Parse one `xsec:` token (see the module grammar).
pub fn parse_section_product(token: &str) -> Result<SectionProduct, String> {
    let body = token
        .strip_prefix(PREFIX)
        .ok_or_else(|| format!("'{token}' is not an xsec: product"))?;
    if body.is_empty() {
        return Err(format!(
            "'{token}': a section product names a fill after 'xsec:' \
             (xsec:[alias:]<fill>[/<overlay>...])"
        ));
    }
    let (alias, body) = match body.find(':') {
        Some(index) => {
            let alias = &body[..index];
            if alias.is_empty() || !alias.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-') {
                return Err(format!(
                    "'{token}': alias '{alias}' must be letters, digits, '_' or '-'"
                ));
            }
            (Some(alias.to_string()), &body[index + 1..])
        }
        None => (None, body),
    };
    let mut terms = body.split('/').map(|text| parse_term(token, text));
    let fill = terms
        .next()
        .ok_or_else(|| format!("'{token}': no fill term"))??;
    let overlays = terms.collect::<Result<Vec<_>, _>>()?;
    Ok(SectionProduct {
        token: token.to_string(),
        alias,
        fill,
        overlays,
    })
}

fn parse_term(token: &str, text: &str) -> Result<Term, String> {
    let text = text.trim();
    if text.is_empty() {
        return Err(format!("'{token}': empty term (two '/' in a row?)"));
    }
    let (head, levels_text) = match text.find('=') {
        Some(index) => (&text[..index], Some(&text[index + 1..])),
        None => (text, None),
    };
    let mut head = head.to_string();
    let mut cold_only = false;
    let mut log = false;
    loop {
        if let Some(rest) = head.strip_suffix("@cold") {
            cold_only = true;
            head = rest.to_string();
        } else if let Some(rest) = head.strip_suffix("~log") {
            log = true;
            head = rest.to_string();
        } else {
            break;
        }
    }
    let names: Vec<String> = head
        .split('+')
        .map(str::trim)
        .map(str::to_string)
        .collect();
    for name in &names {
        if name.is_empty() || !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
            return Err(format!(
                "'{token}': field name '{name}' must be letters, digits or '_' \
                 (a getvar name such as tk, wa, rh, or a raw wrfout variable)"
            ));
        }
    }
    let (levels, highlight) = match levels_text {
        None => (None, None),
        Some(text) => {
            let (list, highlight) = match text.find('@') {
                Some(index) => (&text[..index], Some(&text[index + 1..])),
                None => (text, None),
            };
            let mut levels: Vec<f32> = list
                .split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(|s| {
                    s.parse::<f32>()
                        .map_err(|_| format!("'{token}': level '{s}' is not a number"))
                })
                .collect::<Result<_, _>>()?;
            if levels.is_empty() {
                return Err(format!("'{token}': '=' needs at least one level"));
            }
            levels.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            levels.dedup();
            let highlight = highlight
                .map(|h| {
                    h.trim()
                        .parse::<f32>()
                        .map_err(|_| format!("'{token}': highlight '{h}' is not a number"))
                })
                .transpose()?;
            (Some(levels), highlight)
        }
    };
    Ok(Term {
        names,
        cold_only,
        log,
        levels,
        highlight,
    })
}

/// The isotherm set drawn on every section.
#[derive(Debug, Clone, PartialEq)]
pub struct Isotherms {
    pub levels_c: Vec<f32>,
    pub highlight_c: Option<f32>,
}

impl Default for Isotherms {
    fn default() -> Self {
        Self {
            levels_c: vec![-40.0, -30.0, -20.0, -10.0, 0.0],
            // -10 C is the line a reader of a cold-cloud section looks for
            // by name, so it is the one drawn heavy and labelled.
            highlight_c: Some(-10.0),
        }
    }
}

impl Isotherms {
    /// `0,-5,-10,-15,-20@-10` -- levels in C, an optional highlighted one;
    /// `none` draws no isotherms.
    pub fn parse(text: &str) -> Result<Self, String> {
        let text = text.trim();
        if text.eq_ignore_ascii_case("none") {
            return Ok(Self {
                levels_c: Vec::new(),
                highlight_c: None,
            });
        }
        let (list, highlight) = match text.find('@') {
            Some(index) => (&text[..index], Some(&text[index + 1..])),
            None => (text, None),
        };
        let mut levels: Vec<f32> = list
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(|s| {
                s.parse::<f32>()
                    .map_err(|_| format!("--isotherms: '{s}' is not a temperature in C"))
            })
            .collect::<Result<_, _>>()?;
        if levels.is_empty() {
            return Err("--isotherms needs at least one level in C, or 'none'".to_string());
        }
        levels.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        levels.dedup();
        let highlight_c = highlight
            .map(|h| {
                h.trim()
                    .parse::<f32>()
                    .map_err(|_| format!("--isotherms: highlight '{h}' is not a number"))
            })
            .transpose()?;
        if let Some(h) = highlight_c {
            if !levels.iter().any(|level| (level - h).abs() < 1e-3) {
                return Err(format!(
                    "--isotherms: highlight {h} is not one of the levels {levels:?}"
                ));
            }
        }
        Ok(Self {
            levels_c: levels,
            highlight_c,
        })
    }
}

/// The section line: two endpoints and a label for the headline.
#[derive(Debug, Clone, PartialEq)]
pub struct SectionLine {
    /// Equal-angle scalar mean at each radius; None retains ordinary line cuts.
    pub azimuths: Option<usize>,
    pub start: xs::GeoPoint,
    pub end: xs::GeoPoint,
    pub label: Option<String>,
}

#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct SectionFile {
    #[serde(default)]
    center: Option<[f64; 2]>,
    #[serde(default)]
    radius_km: Option<f64>,
    #[serde(default)]
    azimuths: Option<usize>,
    #[serde(default)]
    start: Option<[f64; 2]>,
    #[serde(default)]
    end: Option<[f64; 2]>,
    #[serde(default)]
    points: Vec<[f64; 2]>,
    #[serde(default)]
    extend_km: Option<f64>,
    #[serde(default)]
    label: Option<String>,
}

impl SectionLine {
    /// `lat,lon,lat,lon` or a JSON file (see the module doc).
    pub fn parse(spec: &str) -> Result<Self, String> {
        let spec = spec.trim();
        let numbers: Vec<&str> = spec.split(',').map(str::trim).collect();
        if numbers.len() == 4 && numbers.iter().all(|n| n.parse::<f64>().is_ok()) {
            let v: Vec<f64> = numbers.iter().map(|n| n.parse::<f64>().unwrap()).collect();
            let start = point(v[0], v[1])?;
            let end = point(v[2], v[3])?;
            return Self::from_endpoints(start, end, None);
        }
        let path = Path::new(spec);
        if !path.is_file() {
            return Err(format!(
                "--section '{spec}' is neither 'lat,lon,lat,lon' nor a readable JSON file"
            ));
        }
        let text = std::fs::read_to_string(path)
            .map_err(|err| format!("read section file {}: {err}", path.display()))?;
        let file: SectionFile = serde_json::from_str(&text)
            .map_err(|err| format!("section file {}: {err}", path.display()))?;
        if file.center.is_some() || file.radius_km.is_some() || file.azimuths.is_some() {
            let center = file.center.ok_or("radial section needs center")?;
            let radius = file.radius_km.ok_or("radial section needs radius_km")?;
            let azimuths = file.azimuths.unwrap_or(72);
            if !(radius.is_finite() && radius >= 1.0 && radius <= 2000.0)
                || !(12..=720).contains(&azimuths)
                || file.start.is_some() || file.end.is_some() || !file.points.is_empty()
                || file.extend_km.is_some() {
                return Err("radial section needs radius 1..2000 km, 12..720 azimuths and no line endpoints".into());
            }
            let start = point(center[0], center[1])?;
            let end = destination_point(start, 0.0, radius)?;
            let mut line = Self::from_endpoints(start, end,
                Some(file.label.unwrap_or_else(|| format!("Azimuthal mean, {azimuths} bearings"))))?;
            line.azimuths = Some(azimuths);
            return Ok(line);
        }
        let (start, end) = match (file.start, file.end, file.points.as_slice()) {
            (Some(a), Some(b), _) => (point(a[0], a[1])?, point(b[0], b[1])?),
            (None, None, points) if points.len() >= 2 => (
                point(points[0][0], points[0][1])?,
                point(points[points.len() - 1][0], points[points.len() - 1][1])?,
            ),
            _ => {
                return Err(format!(
                    "section file {}: give both start and end, or a points \
                     polyline with at least two points",
                    path.display()
                ))
            }
        };
        let (start, end) = match file.extend_km {
            Some(extend) if extend > 0.0 => (
                destination_point(start, forward_bearing_at_end(end, start), extend)?,
                destination_point(end, forward_bearing_at_end(start, end), extend)?,
            ),
            _ => (start, end),
        };
        Self::from_endpoints(start, end, file.label)
    }

    fn from_endpoints(
        start: xs::GeoPoint,
        end: xs::GeoPoint,
        label: Option<String>,
    ) -> Result<Self, String> {
        if xs::haversine_distance_km(start, end) < 1.0 {
            return Err("--section endpoints are less than 1 km apart".to_string());
        }
        Ok(Self { start, end, label, azimuths: None })
    }

    pub fn length_km(&self) -> f64 {
        xs::haversine_distance_km(self.start, self.end)
    }

    /// A line perpendicular to this one through `through`, `half_km` each way.
    pub fn perpendicular_through(&self, through: xs::GeoPoint, half_km: f64) -> Result<Self, String> {
        let bearing = xs::initial_bearing_deg(self.start, self.end);
        let left = (bearing + 270.0).rem_euclid(360.0);
        let right = (bearing + 90.0).rem_euclid(360.0);
        let a = destination_point(through, left, half_km)?;
        let b = destination_point(through, right, half_km)?;
        Self::from_endpoints(a, b, self.label.as_ref().map(|l| format!("{l} (across)")))
    }
}

fn point(lat: f64, lon: f64) -> Result<xs::GeoPoint, String> {
    xs::GeoPoint::new(lat, lon).map_err(|err| format!("section point {lat},{lon}: {err}"))
}

/// The bearing the line leaves its end point on: the back-bearing at the
/// end, reversed.
fn forward_bearing_at_end(start: xs::GeoPoint, end: xs::GeoPoint) -> f64 {
    (xs::initial_bearing_deg(end, start) + 180.0).rem_euclid(360.0)
}

/// Spherical destination point.
fn destination_point(from: xs::GeoPoint, bearing_deg: f64, distance_km: f64) -> Result<xs::GeoPoint, String> {
    // Match rustwx_cross_section::haversine_distance_km and intermediate_point,
    // which construct the section's distance ladder on this same sphere.
    const EARTH_RADIUS_KM: f64 = 6_371.0;
    let delta = distance_km / EARTH_RADIUS_KM;
    let theta = bearing_deg.to_radians();
    let lat1 = from.lat_deg.to_radians();
    let lon1 = from.lon_deg.to_radians();
    let lat2 = (lat1.sin() * delta.cos() + lat1.cos() * delta.sin() * theta.cos()).asin();
    let lon2 = lon1
        + (theta.sin() * delta.sin() * lat1.cos()).atan2(delta.cos() - lat1.sin() * lat2.sin());
    point(lat2.to_degrees(), lon2.to_degrees())
}

/// Everything one `xsec:` render pass needs beyond the products.
pub struct SectionRenderConfig<'a> {
    pub inputs: &'a [PathBuf],
    pub out_dir: &'a Path,
    pub line: SectionLine,
    /// Draw a second frame per product, perpendicular through the column
    /// of the fill's maximum, this many km long.
    pub across_km: Option<f64>,
    pub isotherms: Isotherms,
    /// Ordinal frame index across the inputs' valid times, or all.
    pub frame: Option<usize>,
    pub width: u32,
    pub height: u32,
    /// The ceiling the fitted height range may not pass.
    pub top_km: f64,
    /// The altitude the fitted range keeps at least two kilometres of air
    /// above.  `None` takes the line's own highest terrain.
    pub reference_km: Option<f64>,
    pub domain_slug: Option<String>,
    pub source_label: String,
    pub theme: &'a RenderTheme,
}

/// One product on one frame: the slug and where it went, or why not.
pub struct SectionOutcome {
    pub slug: String,
    pub result: Result<PathBuf, String>,
    /// The range this cut's fill was drawn on, and the rule that set it.
    /// `None` when nothing was drawn or the fill had no finite value on
    /// the line.  The caller reports it so that two pictures of one line
    /// drawn on two different bars are told apart by their receipt.
    pub drawn: Option<DrawnFill>,
}

struct FrameRef {
    path: PathBuf,
    time_index: usize,
    valid_unix: i64,
}

/// The frames of the inputs, the run's origin, and the model token of the
/// sections' time line: `WOOF` for the engine's own history files, a
/// model label the files agree on, otherwise the store identity `WRF`
/// (the maps' rule, `rustwx_products::shared_context::model_token`).
fn enumerate_frames(inputs: &[PathBuf]) -> Result<(Vec<FrameRef>, i64, String), String> {
    let mut frames = Vec::new();
    let mut origin: Option<i64> = None;
    let mut model_labels = Vec::with_capacity(inputs.len());
    for path in inputs {
        let file = WrfFile::open(path).map_err(|err| format!("{}: {err}", path.display()))?;
        // A label the metadata row cannot take keeps the store identity,
        // as it does on the maps; it never stops the sections.
        model_labels.push(crate::local_import::model_label(&file).unwrap_or(None));
        let times = file
            .times()
            .map_err(|err| format!("{}: Times: {err}", path.display()))?;
        for (time_index, label) in times.iter().enumerate() {
            let valid_unix = parse_utc_timestamp(label)
                .ok_or_else(|| format!("{}: unreadable timestamp {label:?}", path.display()))?;
            frames.push(FrameRef {
                path: path.clone(),
                time_index,
                valid_unix,
            });
        }
        if origin.is_none() {
            // The run's origin, not the domain's own start: a delayed nest's
            // START_DATE is later by design, and its sections are labelled on
            // the same lead clock as every other domain's (A137).
            let stamp = |name: &str| {
                file.global_attr_str(name)
                    .ok()
                    .and_then(|value| parse_utc_timestamp(&value))
            };
            origin = wrf_run_origin(stamp("START_DATE"), stamp("SIMULATION_START_DATE"))
                .map_err(|err| format!("{}: {err}", path.display()))?;
        }
    }
    if frames.is_empty() {
        return Err("no time records in the wrfout inputs".to_string());
    }
    frames.sort_by_key(|frame| frame.valid_unix);
    let origin = origin.unwrap_or(frames[0].valid_unix);
    let model = rustwx_products::shared_context::model_token_with(
        rustwx_core::ModelId::WrfGdex,
        rustwx_products::shared_context::agreed_model_label(&model_labels).as_deref(),
    );
    Ok((frames, origin, model))
}

/// Bilinear column weights in grid space for a lat/lon point.
struct GridLocator {
    ny: usize,
    nx: usize,
    lat: Vec<f64>,
    lon: Vec<f64>,
}

impl GridLocator {
    fn new(file: &WrfFile, time_index: usize) -> Result<Self, String> {
        let read = |name: &str| -> Result<Vec<f64>, String> {
            getvar(file, name, Some(time_index), &ComputeOpts::default())
                .map(|out| out.data)
                .map_err(|err| format!("read {name}: {err}"))
        };
        let lat = read("XLAT")?;
        let lon = read("XLONG")?;
        let (ny, nx) = (file.ny, file.nx);
        if lat.len() != ny * nx || lon.len() != ny * nx {
            return Err(format!(
                "XLAT/XLONG hold {}/{} values for a {ny}x{nx} grid",
                lat.len(),
                lon.len()
            ));
        }
        Ok(Self { ny, nx, lat, lon })
    }

    fn at(&self, j: usize, i: usize) -> (f64, f64) {
        let index = j * self.nx + i;
        (self.lat[index], self.lon[index])
    }

    fn nearest(&self, lat: f64, lon: f64) -> (usize, usize) {
        let cos_lat = lat.to_radians().cos();
        let mut best = (0usize, 0usize);
        let mut best_d2 = f64::INFINITY;
        for j in 0..self.ny {
            for i in 0..self.nx {
                let (glat, glon) = self.at(j, i);
                let dlat = glat - lat;
                let dlon = wrap_lon(glon - lon) * cos_lat;
                let d2 = dlat * dlat + dlon * dlon;
                if d2 < best_d2 {
                    best_d2 = d2;
                    best = (j, i);
                }
            }
        }
        best
    }

    /// Four (cell index, weight) pairs, or `None` when the point falls
    /// outside the grid.
    fn weights(&self, lat: f64, lon: f64) -> Option<[(usize, f64); 4]> {
        let (j0, i0) = self.nearest(lat, lon);
        let (lat0, lon0) = self.at(j0, i0);
        let ip = (i0 + 1).min(self.nx - 1);
        let im = i0.saturating_sub(1);
        let jp = (j0 + 1).min(self.ny - 1);
        let jm = j0.saturating_sub(1);
        let di_span = (ip - im).max(1) as f64;
        let dj_span = (jp - jm).max(1) as f64;
        let (lat_ip, lon_ip) = self.at(j0, ip);
        let (lat_im, lon_im) = self.at(j0, im);
        let (lat_jp, lon_jp) = self.at(jp, i0);
        let (lat_jm, lon_jm) = self.at(jm, i0);
        let a = (lat_ip - lat_im) / di_span; // dlat/di
        let b = (lat_jp - lat_jm) / dj_span; // dlat/dj
        let c = wrap_lon(lon_ip - lon_im) / di_span; // dlon/di
        let d = wrap_lon(lon_jp - lon_jm) / dj_span; // dlon/dj
        let det = a * d - b * c;
        if det.abs() < 1e-12 {
            return None;
        }
        let dlat = lat - lat0;
        let dlon = wrap_lon(lon - lon0);
        let di = (dlat * d - b * dlon) / det;
        let dj = (a * dlon - c * dlat) / det;
        if di.abs() > 1.5 || dj.abs() > 1.5 {
            return None;
        }
        let fi = (i0 as f64 + di).clamp(0.0, (self.nx - 1) as f64);
        let fj = (j0 as f64 + dj).clamp(0.0, (self.ny - 1) as f64);
        if (i0 == 0 && di < -0.5)
            || (i0 == self.nx - 1 && di > 0.5)
            || (j0 == 0 && dj < -0.5)
            || (j0 == self.ny - 1 && dj > 0.5)
        {
            return None;
        }
        let i_lo = fi.floor() as usize;
        let j_lo = fj.floor() as usize;
        let i_hi = (i_lo + 1).min(self.nx - 1);
        let j_hi = (j_lo + 1).min(self.ny - 1);
        let tx = fi - i_lo as f64;
        let ty = fj - j_lo as f64;
        Some([
            (j_lo * self.nx + i_lo, (1.0 - tx) * (1.0 - ty)),
            (j_lo * self.nx + i_hi, tx * (1.0 - ty)),
            (j_hi * self.nx + i_lo, (1.0 - tx) * ty),
            (j_hi * self.nx + i_hi, tx * ty),
        ])
    }
}

fn wrap_lon(delta: f64) -> f64 {
    let mut d = delta;
    while d > 180.0 {
        d -= 360.0;
    }
    while d < -180.0 {
        d += 360.0;
    }
    d
}

/// A 3-D field on native levels, `[nz, ny, nx]`.
struct Native3D {
    data: Vec<f64>,
    nz: usize,
    cells: usize,
    units: String,
}

impl Native3D {
    fn read(file: &WrfFile, name: &str, time_index: usize) -> Result<Self, String> {
        let out = getvar(file, name, Some(time_index), &ComputeOpts::default())
            .map_err(|err| format!("read {name}: {err}"))?;
        let cells = file.ny * file.nx;
        let nz = if out.data.len() == file.nz * cells {
            file.nz
        } else if out.data.len() == file.nz_stag * cells {
            // A staggered field (raw W): drop the top interface, which is
            // close enough for a section drawn at 250 m steps; `wa` is
            // the destaggered spelling when a caller wants it exact.
            file.nz_stag
        } else {
            return Err(format!(
                "{name} holds {} values, not a {}x{}x{} 3-D field",
                out.data.len(),
                file.nz,
                file.ny,
                file.nx
            ));
        };
        Ok(Self {
            data: out.data,
            nz,
            cells,
            units: out.units,
        })
    }

    fn column(&self, cell: usize) -> impl Iterator<Item = f64> + '_ {
        (0..self.nz).map(move |k| self.data[k * self.cells + cell])
    }
}

/// One column's values on the height ladder, linear in z; NaN outside the
/// column's own span.
fn column_on_ladder(z: &[f64], v: &[f64], ladder: &[f64], out: &mut [f32]) {
    for (target, slot) in ladder.iter().zip(out.iter_mut()) {
        let n = z.len().min(v.len());
        if n < 2 || *target < z[0] || *target > z[n - 1] {
            *slot = f32::NAN;
            continue;
        }
        // z increases with k on a WRF column.
        let mut lo = 0usize;
        let mut hi = n - 1;
        while hi - lo > 1 {
            let mid = (lo + hi) / 2;
            if z[mid] <= *target {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let span = z[hi] - z[lo];
        let t = if span.abs() < 1e-9 { 0.0 } else { (*target - z[lo]) / span };
        let value = v[lo] + (v[hi] - v[lo]) * t;
        *slot = if value.is_finite() { value as f32 } else { f32::NAN };
    }
}

/// The sampled columns of one frame: weights per section point, the
/// heights of each contributing column, the terrain height per point.
struct SampledColumns {
    weights: Vec<Option<Vec<(usize, f64)>>>,
    full_ring: bool,
    heights: Native3D,
    terrain_m: Vec<f64>,
}

impl SampledColumns {
    /// Level-major `[n_levels * n_points]` section of `field` on `ladder`.
    fn section(&self, field: &Native3D, ladder: &[f64]) -> Vec<f32> {
        let n_points = self.weights.len();
        let n_levels = ladder.len();
        let mut values = vec![f32::NAN; n_levels * n_points];
        let mut z = Vec::with_capacity(self.heights.nz);
        let mut v = Vec::with_capacity(field.nz);
        let mut column = vec![f32::NAN; n_levels];
        for (p, weights) in self.weights.iter().enumerate() {
            let Some(weights) = weights else { continue };
            let mut accum = vec![0.0f64; n_levels];
            let mut weight_sum = vec![0.0f64; n_levels];
            for (cell, weight) in weights {
                if *weight <= 0.0 {
                    continue;
                }
                z.clear();
                z.extend(self.heights.column(*cell));
                v.clear();
                v.extend(field.column(*cell));
                column_on_ladder(&z, &v, ladder, &mut column);
                for (k, value) in column.iter().enumerate() {
                    if value.is_finite() {
                        accum[k] += f64::from(*value) * weight;
                        weight_sum[k] += weight;
                    }
                }
            }
            for k in 0..n_levels {
                if weight_sum[k] > 0.0 && (!self.full_ring || weight_sum[k] >= 1.0 - 1.0e-9) {
                    values[k * n_points + p] = (accum[k] / weight_sum[k]) as f32;
                }
            }
        }
        values
    }
}

fn sample_columns(
    file: &WrfFile,
    time_index: usize,
    points: &[xs::GeoPoint],
    radial: Option<(&SectionLine, &[f64])>,
) -> Result<SampledColumns, String> {
    let locator = GridLocator::new(file, time_index)?;
    let weights: Vec<Option<Vec<(usize, f64)>>> = if let Some((line, radii)) = radial {
        let angles = line.azimuths.unwrap();
        // Equal angular sampling, then native-column interpolation to a common
        // MSL height before averaging. An incomplete circle remains missing.
        radii.par_iter().map(|radius| {
            let mut weights = Vec::with_capacity(angles * 4);
            for angle in 0..angles {
                let p = destination_point(line.start, 360.0 * angle as f64 / angles as f64, *radius).ok()?;
                for (cell, weight) in locator.weights(p.lat_deg, p.lon_deg)? {
                    weights.push((cell, weight / angles as f64));
                }
            }
            Some(weights)
        }).collect()
    } else {
        points.par_iter().map(|p| locator.weights(p.lat_deg, p.lon_deg).map(|w| w.to_vec())).collect()
    };
    let inside = weights.iter().filter(|w| w.is_some()).count();
    if inside < 2 {
        return Err(format!(
            "the section line lies outside the grid ({inside} of {} points inside)",
            points.len()
        ));
    }
    let heights = Native3D::read(file, "height", time_index)?;
    let terrain = getvar(file, "ter", Some(time_index), &ComputeOpts::default())
        .map_err(|err| format!("read ter: {err}"))?;
    let terrain_m = weights
        .iter()
        .map(|w| match w {
            Some(cells) => cells
                .iter()
                .map(|(cell, weight)| terrain.data[*cell] * weight)
                .sum::<f64>(),
            None => f64::NAN,
        })
        .collect();
    Ok(SampledColumns {
        full_ring: radial.is_some(),
        weights,
        heights,
        terrain_m,
    })
}

/// Units and scale for a named field, by `getvar`'s answer first and by
/// the WRF naming convention when it has none.
fn units_and_scale(name: &str, reported: &str) -> (String, f64) {
    let upper = name.to_ascii_uppercase();
    if upper.starts_with('Q') && !upper.starts_with("QN") {
        return ("g kg-1".to_string(), 1000.0);
    }
    if upper.starts_with("QN") {
        return ("kg-1".to_string(), 1.0);
    }
    if !reported.is_empty() {
        return (reported.to_string(), 1.0);
    }
    match upper.as_str() {
        "W" | "WA" | "U" | "V" | "UA" | "VA" => ("m s-1".to_string(), 1.0),
        "TK" | "TEMP" | "THETA" | "T" => ("K".to_string(), 1.0),
        _ => (String::new(), 1.0),
    }
}

/// Default contour levels for an overlay, by name where the convention is
/// settled and from the field's positive range otherwise.
fn default_levels(term: &Term, units: &str, values: &[f32]) -> (Vec<f32>, Option<f32>) {
    let first = term.names[0].to_ascii_uppercase();
    if first == "WA" || first == "W" {
        return (vec![1.0, 2.0, 5.0, 10.0], Some(5.0));
    }
    if units == "g kg-1" {
        return (vec![0.01, 0.1, 0.5, 1.0, 2.0], None);
    }
    let finite: Vec<f32> = values.iter().copied().filter(|v| v.is_finite()).collect();
    if finite.is_empty() {
        return (Vec::new(), None);
    }
    let max = finite.iter().copied().fold(f32::MIN, f32::max);
    let min = finite.iter().copied().fold(f32::MAX, f32::min);
    if max <= min {
        return (Vec::new(), None);
    }
    (nice_levels(min, max, 5), None)
}

fn nice_levels(min: f32, max: f32, count: usize) -> Vec<f32> {
    let span = (max - min) as f64;
    let raw_step = span / count as f64;
    let magnitude = 10f64.powf(raw_step.log10().floor());
    let norm = raw_step / magnitude;
    let step = if norm < 1.5 {
        1.0
    } else if norm < 3.5 {
        2.0
    } else if norm < 7.5 {
        5.0
    } else {
        10.0
    } * magnitude;
    let mut levels = Vec::new();
    let mut level = (f64::from(min) / step).ceil() * step;
    while level <= f64::from(max) && levels.len() < 20 {
        levels.push(level as f32);
        level += step;
    }
    levels
}

fn color(rgba: Rgba) -> xs::Color {
    xs::Color::rgba(rgba.r, rgba.g, rgba.b, rgba.a)
}

/// The weather family a term belongs to, read off the WRF naming
/// convention alone.  It decides which of the tree's palettes paints the
/// fill and which saturated ink draws the overlay -- so a new species is
/// a name in a table, never a code path (the project's arbitrary
/// acceptance test).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldFamily {
    /// A carried number concentration or passive tracer.
    Tracer,
    /// Liquid cloud and rain water.
    LiquidWater,
    /// Frozen hydrometeors.
    Ice,
    /// Vertical velocity: signed, and therefore diverging.
    VerticalVelocity,
    Temperature,
    Moisture,
    /// A sum that crosses families, or a hydrometeor total.
    Condensate,
    Other,
}

impl FieldFamily {
    /// True when the family's zero is its middle, so its range is drawn
    /// symmetric about it and its ramp is the diverging one.
    pub fn is_diverging(self) -> bool {
        matches!(self, Self::VerticalVelocity)
    }

    /// True when an OVERLAY of this family may raise the frame's ceiling.
    ///
    /// What is IN the air sets the top of a section: the plume, the cloud
    /// water, the ice.  The air's own motion and its temperature exist
    /// through the whole column -- a gravity wave over a range crosses
    /// 1 m s-1 at the model top -- so an overlay of either would hold every
    /// frame open to the ceiling and there would be no fit at all.
    pub fn sets_the_ceiling(self) -> bool {
        matches!(
            self,
            Self::Tracer | Self::LiquidWater | Self::Ice | Self::Condensate | Self::Moisture
        )
    }

    /// The saturated ink an OVERLAY of this family is contoured in.  One
    /// hue per family, far enough apart to be told apart on a printed
    /// sheet; the four the sections actually carry -- cloud water, ice,
    /// updraft, tracer -- are green, cyan, magenta and orange.
    pub fn overlay_ink(self) -> Option<xs::Color> {
        Some(match self {
            Self::LiquidWater => xs::Color::rgba(0, 148, 68, 235),
            Self::Ice => xs::Color::rgba(0, 168, 222, 235),
            Self::VerticalVelocity => xs::Color::rgba(214, 0, 150, 235),
            Self::Tracer => xs::Color::rgba(240, 118, 0, 235),
            Self::Condensate => xs::Color::rgba(112, 56, 200, 235),
            Self::Moisture => xs::Color::rgba(0, 110, 140, 235),
            Self::Temperature | Self::Other => return None,
        })
    }
}

/// The inks an overlay that names no family falls back to, in order.
const NEUTRAL_OVERLAY_INKS: [xs::Color; 3] = [
    xs::Color::rgba(90, 90, 96, 235),
    xs::Color::rgba(150, 90, 20, 235),
    xs::Color::rgba(190, 40, 140, 235),
];

/// One field name's family.
fn name_family(name: &str) -> FieldFamily {
    let upper = name.trim().to_ascii_uppercase();
    match upper.as_str() {
        "QICE" | "QI" | "QSNOW" | "QS" | "QGRAUP" | "QGRAUPEL" | "QG" | "QHAIL" | "QH"
        | "QNICE" | "QNSNOW" | "QNGRAUPEL" => return FieldFamily::Ice,
        "QCLOUD" | "QC" | "QRAIN" | "QR" | "QNRAIN" | "QNDROP" | "QNCLOUD" => {
            return FieldFamily::LiquidWater;
        }
        "W" | "WA" | "WW" => return FieldFamily::VerticalVelocity,
        "TK" | "TC" | "T" | "TEMP" | "THETA" | "TH" | "THETAE" | "TD" | "TWB" => {
            return FieldFamily::Temperature;
        }
        "QVAPOR" | "QV" | "RH" | "Q" => return FieldFamily::Moisture,
        _ => {}
    }
    // A number concentration (`QN...`) or a WRF passive tracer (`tr17_1`)
    // is a carried scalar: the map families paint those with the tracer
    // ramp and so does a section.
    if upper.starts_with("QN") {
        return FieldFamily::Tracer;
    }
    if upper.starts_with("TR") && upper.chars().any(|c| c.is_ascii_digit()) {
        return FieldFamily::Tracer;
    }
    if upper.starts_with('Q') {
        return FieldFamily::Condensate;
    }
    FieldFamily::Other
}

/// A term's family: the one its names agree on, a hydrometeor sum's
/// `Condensate`, or `Other` when the sum crosses families entirely.
pub fn term_family(term: &Term) -> FieldFamily {
    let mut families = term.names.iter().map(|name| name_family(name));
    let Some(first) = families.next() else {
        return FieldFamily::Other;
    };
    let mut all_same = true;
    let mut all_water = matches!(
        first,
        FieldFamily::LiquidWater | FieldFamily::Ice | FieldFamily::Condensate
    );
    let mut all_tracer = first == FieldFamily::Tracer;
    for family in families {
        all_same &= family == first;
        all_water &= matches!(
            family,
            FieldFamily::LiquidWater | FieldFamily::Ice | FieldFamily::Condensate
        );
        all_tracer &= family == FieldFamily::Tracer;
    }
    if all_same {
        return first;
    }
    if all_tracer {
        return FieldFamily::Tracer;
    }
    if all_water {
        return FieldFamily::Condensate;
    }
    FieldFamily::Other
}

/// The tree's own weather ramp for a family, at `bands` steps.
///
/// The tracer ramp is the one the map families paint a carried scalar
/// with, taken at full saturation because a section's fill has nothing
/// underneath it to show through.
pub fn family_palette(family: FieldFamily, bands: usize) -> Vec<xs::Color> {
    let bands = bands.max(2);
    let named = match family {
        FieldFamily::Tracer => {
            let anchors: Vec<Rgba> = rustwx_products::plot_design::tracer_scale_colors_saturated()
                .into_iter()
                .map(Rgba::from)
                .collect();
            return rustwx_render::theme::resample(&anchors, bands)
                .into_iter()
                .map(|c| xs::Color::rgba(c.r, c.g, c.b, c.a))
                .collect();
        }
        FieldFamily::LiquidWater => xs::CrossSectionPalette::CloudWater,
        FieldFamily::Ice => xs::CrossSectionPalette::CloudIce,
        FieldFamily::VerticalVelocity => xs::CrossSectionPalette::Omega,
        FieldFamily::Temperature => xs::CrossSectionPalette::TemperatureStandard,
        FieldFamily::Moisture => xs::CrossSectionPalette::SpecificHumidity,
        FieldFamily::Condensate | FieldFamily::Other => xs::CrossSectionPalette::TotalCondensate,
    };
    named.sampled_colors(bands)
}

/// Theme-driven request: canvas and inks from the theme, the renderer's
/// own light look when the theme is the default.
fn build_request(theme: &RenderTheme, width: u32, height: u32) -> xs::CrossSectionRenderRequest {
    let mut request = xs::CrossSectionRenderRequest::default().with_dimensions(width, height);
    if theme.is_default() {
        return request;
    }
    if let Some(canvas) = theme.canvas {
        request.page_background_top = color(canvas);
        request.page_background_bottom = color(canvas);
    }
    if let Some(map) = theme.map.or(theme.canvas) {
        request.plot_background_top = color(map);
        request.plot_background_bottom = color(map);
    }
    if let Some(ink) = theme.title_ink {
        request.text_color = color(ink);
    }
    if let Some(ink) = theme.subtitle_ink {
        request.axis_color = color(ink);
        request.isotherm_color = color(Rgba::with_alpha(ink.r, ink.g, ink.b, 205));
    }
    if let Some(frame) = theme.presentation.frame.or(theme.grid) {
        request.frame_color = color(frame);
    }
    if let Some(grid) = theme.grid {
        request.grid_major_color = color(grid);
    }
    if let Some(hairline) = theme.hairline.or(theme.grid) {
        request.grid_minor_color = color(hairline);
        // The ground reads as ground under a theme too: the theme's own
        // hairline at the surface, deepening into the frame's floor.
        request.terrain_fill_top = color(hairline);
        request.terrain_fill_bottom = color(Rgba::with_alpha(
            (u16::from(hairline.r) * 3 / 5) as u8,
            (u16::from(hairline.g) * 3 / 5) as u8,
            (u16::from(hairline.b) * 3 / 5) as u8,
            hairline.a,
        ));
    }
    if let Some(stroke) = theme.presentation.coast.or(theme.subtitle_ink) {
        request.terrain_stroke = color(stroke);
        request.terrain_highlight = color(stroke);
    }
    if let Some(attention) = theme.attention {
        request.highlight_isotherm_color = color(attention);
    }
    // The producer's mark on the title row, from `text.source_label`.  A
    // theme that names none leaves it unset and the header is drawn the
    // way it always was.
    if let Some(mark) = theme.source_subtitle(theme.source_label.clone()) {
        request.source_label = Some(mark);
    }
    request
}

/// Under auto layout a section carries the map products' header: the
/// renderer draws no title row of its own and leaves the header band
/// empty, and the caller draws the header into it after the render.
/// `None` under fixed layout, where the section keeps its own header.
fn planned_section_header(request: &mut xs::CrossSectionRenderRequest) -> Option<rustwx_render::CanvasPlan> {
    let rustwx_render::LayoutMode::Auto { class, scale } = rustwx_render::layout_mode() else {
        return None;
    };
    let plan = rustwx_render::LayoutTable::builtin().plan_header(request.width, request.height, class, scale);
    let defaults = xs::Insets::default();
    // Room under the header for the axis title and the A and B labels.
    request.margins = xs::Insets {
        left: defaults.left,
        right: defaults.right,
        top: plan.header.h + 34,
        bottom: 58,
    };
    request.external_header = true;
    Some(plan)
}

/// The smallest span of a section fill, in its display units, that is
/// drawn as a field rather than as no signal.
const SECTION_MIN_SIGNAL_SPAN: f32 = 1e-6;

/// The section's time line in the map header's grammar:
/// `Init 05/26 15Z | F002 | Valid 05/26 17Z | WOOF`.  The domain rides in
/// the title as the maps' `(d01 3 km)`, so the header composes it into the
/// same place and spelling as every map of the domain.  `model` is the
/// inputs' model token (`WOOF`, a named model, or `WRF` for stock WRF),
/// which a theme's `text.model_label` still replaces.
fn section_time_line(
    theme: &RenderTheme,
    model: &str,
    init_label: &str,
    lead_seconds: u64,
    valid_label: &str,
) -> String {
    let hours = lead_seconds / 3_600;
    let minutes = (lead_seconds % 3_600) / 60;
    let lead = if minutes == 0 {
        format!("F{hours:03}")
    } else {
        format!("F{hours:03}:{minutes:02}")
    };
    let valid = valid_label.replace(":00Z", "Z");
    format!("Init {init_label} | {lead} | Valid {valid} | {}", theme.model_name(model))
}

/// A domain key as the map headers spell it: `d01-3km` is `d01 3 km`,
/// `d02-750m` is `d02 750 m`.  Anything else is kept as it is.
fn map_style_domain(domain: &str) -> String {
    let Some((grid, spacing)) = domain.split_once('-') else {
        return domain.to_string();
    };
    let digits = spacing
        .find(|ch: char| !(ch.is_ascii_digit() || ch == '.'))
        .unwrap_or(spacing.len());
    if digits == 0 || digits == spacing.len() {
        return format!("{grid} {spacing}");
    }
    format!("{grid} {} {}", &spacing[..digits], &spacing[digits..])
}

/// The provenance a section's metadata carries: the theme's
/// `text.source_label` when it names one, else the run's own label -- the
/// rule the map products take at the renderer's seam.
fn section_source_label(theme: &RenderTheme, derived: &str) -> String {
    theme
        .source_subtitle(Some(derived.to_string()))
        .unwrap_or_else(|| derived.to_string())
}

/// The ink one overlay is contoured in: its family's own saturated hue,
/// and for a family that names none (a temperature overlay is the
/// isotherm set, drawn in the theme's isotherm ink) the next neutral in
/// the rotation.
fn overlay_ink(term: &Term, position: usize) -> xs::Color {
    term_family(term)
        .overlay_ink()
        .unwrap_or(NEUTRAL_OVERLAY_INKS[position % NEUTRAL_OVERLAY_INKS.len()])
}

/// The names a theme may use to name a colour override for a section
/// product: the output slug, the alias, and the fill's own label.
fn product_override_keys(product: &SectionProduct) -> Vec<String> {
    let mut keys = vec![product.slug()];
    if let Some(alias) = &product.alias {
        keys.push(alias.clone());
    }
    keys.push(product.fill.label().to_ascii_lowercase());
    keys
}

/// The fill's ramp.
///
/// The FIELD's own weather palette, always -- a theme moves the canvas,
/// the inks and the chrome, and a brand ramp painted over every scientific
/// field is what made a delivered gallery monotone.  The one way a theme
/// reaches a fill is by NAMING the product in its `products` table, which
/// is the seam a difference product's brand ramp goes through.
fn fill_palette(
    theme: &RenderTheme,
    product: &SectionProduct,
    family: FieldFamily,
    bands: usize,
) -> Vec<xs::Color> {
    for key in product_override_keys(product) {
        if let Some(anchors) = theme.products.get(&key) {
            if !anchors.is_empty() {
                return rustwx_render::theme::resample(anchors, bands)
                    .into_iter()
                    .map(|c| xs::Color::rgba(c.r, c.g, c.b, c.a))
                    .collect();
            }
        }
    }
    family_palette(family, bands)
}

struct TermSection {
    values: Vec<f32>,
    units: String,
    label: String,
    /// For a `~log` term, the value its absence sits at.
    log_floor: Option<f32>,
}

/// Sum the term's fields on the ladder, apply its modifiers.
fn term_section(
    term: &Term,
    file: &WrfFile,
    time_index: usize,
    columns: &SampledColumns,
    ladder: &[f64],
    temperature_c: &[f32],
    cache: &mut BTreeMap<String, Native3D>,
) -> Result<TermSection, String> {
    let n = columns.weights.len() * ladder.len();
    let mut sum = vec![0.0f32; n];
    let mut seen = vec![false; n];
    let mut units = String::new();
    for (index, name) in term.names.iter().enumerate() {
        if !cache.contains_key(name) {
            cache.insert(name.clone(), Native3D::read(file, name, time_index)?);
        }
        let field = &cache[name];
        let (field_units, scale) = units_and_scale(name, &field.units);
        if index == 0 {
            units = field_units;
        }
        let section = columns.section(field, ladder);
        for (k, value) in section.iter().enumerate() {
            if value.is_finite() {
                sum[k] += value * scale as f32;
                seen[k] = true;
            }
        }
    }
    let mut values: Vec<f32> = sum
        .iter()
        .zip(seen.iter())
        .map(|(v, s)| if *s { *v } else { f32::NAN })
        .collect();
    if term.cold_only {
        for (value, t) in values.iter_mut().zip(temperature_c.iter()) {
            if !(t.is_finite() && *t < 0.0) {
                *value = f32::NAN;
            }
        }
    }
    let mut label = term.label();
    let mut log_floor = None;
    if term.cold_only {
        label.push_str(" (T<0C)");
    }
    if term.log {
        // Decades below the field's maximum; anything fainter than six of
        // them is numerical dust (a tracer at 1e-25 kg-1 is not a plume)
        // and stays transparent, so the fill shows structure.
        let max_log = values
            .iter()
            .copied()
            .filter(|v| v.is_finite() && *v > 0.0)
            .map(f32::log10)
            .fold(f32::MIN, f32::max);
        // Nothing positive anywhere: the term is ABSENT on this frame, not
        // present at an enormous negative log.  Leaving it absent is what
        // routes the panel through the blank-frame path that says "no
        // signal" in its headline; calling it present handed the colourbar
        // a range of f32::MIN to f32::MIN.
        let floor = max_log - LOG_FILL_DECADES;
        let present = max_log > f32::MIN && floor.is_finite();
        if !present {
            for value in values.iter_mut() {
                *value = f32::NAN;
            }
            return Ok(TermSection {
                values,
                units,
                label,
                log_floor: None,
            });
        }
        log_floor = Some(floor);
        for value in values.iter_mut() {
            // Below the floor is the field's ABSENCE, and absence sits ON
            // the floor rather than punching a NaN hole: a hole makes the
            // plume's edge a hard boundary that jumps a whole rung between
            // sample columns, which is the staircase the delivered sections
            // were rejected for.  Only a column that does not reach this
            // height stays NaN.
            *value = if value.is_finite() && *value > 0.0 {
                value.log10().max(floor)
            } else if value.is_finite() {
                floor
            } else {
                f32::NAN
            };
        }
        // The label stays the field's own name: the fill is drawn in
        // log10, but its colour bar prints the field's numbers at the
        // decades (`log_decade_ticks`), so "log10" in front of the name
        // would tell a reader the ticks are exponents when they are not.
    }
    Ok(TermSection {
        values,
        units,
        label,
        log_floor,
    })
}

fn valid_parts(unix: i64) -> (i64, u32, u32, u32, u32, u32) {
    let days = unix.div_euclid(86_400);
    let second_of_day = unix.rem_euclid(86_400);
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let day_of_era = z - era * 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let mut year = year_of_era + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_prime = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_prime + 2) / 5 + 1;
    let month = month_prime + if month_prime < 10 { 3 } else { -9 };
    if month <= 2 {
        year += 1;
    }
    (
        year,
        month as u32,
        day as u32,
        (second_of_day / 3_600) as u32,
        ((second_of_day % 3_600) / 60) as u32,
        (second_of_day % 60) as u32,
    )
}

/// Render every section product on every selected frame.  Each outcome is
/// reported through `emit` as it lands so the caller's event grammar
/// (RENDERED / FAILED) stays the one the Python side reads.
pub fn render_sections(
    products: &[SectionProduct],
    config: &SectionRenderConfig<'_>,
    mut emit: impl FnMut(SectionOutcome),
) -> Result<(usize, usize), String> {
    if config.line.azimuths.is_some() {
        if config.across_km.is_some() || products.iter().any(|p|
            std::iter::once(&p.fill).chain(&p.overlays).any(|term| term.cold_only)) {
            return Err("azimuthal sections do not support across cuts or cold-only masks".into());
        }
    }
    let (frames, origin_unix, model) = enumerate_frames(config.inputs)?;
    let frames: Vec<&FrameRef> = match config.frame {
        None => frames.iter().collect(),
        Some(index) => vec![frames.get(index).ok_or_else(|| {
            format!(
                "--frames {index} out of range; the inputs hold {} frame(s)",
                frames.len()
            )
        })?],
    };
    std::fs::create_dir_all(config.out_dir)
        .map_err(|err| format!("create {}: {err}", config.out_dir.display()))?;
    let (oy, om, od, oh, _, _) = valid_parts(origin_unix);
    let date = format!("{oy:04}{om:02}{od:02}");
    let domain = config.domain_slug.clone().unwrap_or_else(|| "native_grid".to_string());
    // The full ladder runs to the caller's ceiling; each frame is drawn to
    // the rung its own data reaches (see `fitted_top_m`).
    let ladder: Vec<f64> = (0..)
        .map(|k| f64::from(k) * LADDER_STEP_M)
        .take_while(|z| *z <= config.top_km * 1000.0 + 1e-6)
        .collect();
    let mut rendered = 0usize;
    let mut failed = 0usize;

    for frame in frames {
        let file = WrfFile::open(&frame.path)
            .map_err(|err| format!("{}: {err}", frame.path.display()))?;
        let lead = (frame.valid_unix - origin_unix).max(0) as u64;
        let (vy, vm, vd, vh, vmin, vs) = valid_parts(frame.valid_unix);
        let suffix = format!(
            "valid_{vy:04}{vm:02}{vd:02}_{vh:02}{vmin:02}{vs:02}z_lead_{:03}h{:02}m{:02}s",
            lead / 3_600,
            (lead % 3_600) / 60,
            lead % 60
        );
        // The crate's header spells "Init: A   +HHH:MM   Valid: B" from
        // these three attributes.
        let init_label = format!("{om:02}/{od:02} {oh:02}Z");
        let lead_label = format!("+{:03}:{:02}", lead / 3_600, (lead % 3_600) / 60);
        let valid_label = format!("{vm:02}/{vd:02} {vh:02}:{vmin:02}Z");
        let mut lines: Vec<(SectionLine, &'static str)> = vec![(config.line.clone(), "")];
        let mut across_pending = config.across_km;
        let mut line_index = 0usize;
        while line_index < lines.len() {
            let (line, tag) = lines[line_index].clone();
            line_index += 1;
            let path = xs::SectionPath::endpoints(line.start, line.end)
                .map_err(|err| format!("section path: {err}"))?;
            let count = section_sample_count(line.length_km(), file.dx);
            let sampled = path
                .sample_count(count)
                .map_err(|err| format!("section sampling: {err}"))?;
            let points: Vec<xs::GeoPoint> = sampled.samples.iter().map(|s| s.point).collect();
            let distances: Vec<f64> = sampled.samples.iter().map(|s| s.distance_km).collect();
            let columns = match sample_columns(&file, frame.time_index, &points, line.azimuths.map(|_| (&line, distances.as_slice()))) {
                Ok(columns) => columns,
                Err(err) => {
                    for product in products {
                        failed += 1;
                        emit(SectionOutcome {
                            slug: format!("{}{}", product.slug(), tag),
                            result: Err(format!("{}: {err}", frame.path.display())),
                            drawn: None,
                        });
                    }
                    continue;
                }
            };
            let mut cache: BTreeMap<String, Native3D> = BTreeMap::new();
            let temperature_c: Vec<f32> = match Native3D::read(&file, "tk", frame.time_index) {
                Ok(tk) => {
                    let values = columns.section(&tk, &ladder);
                    cache.insert("tk".to_string(), tk);
                    values.iter().map(|t| t - 273.15).collect()
                }
                Err(err) => {
                    for product in products {
                        failed += 1;
                        emit(SectionOutcome {
                            slug: format!("{}{}", product.slug(), tag),
                            result: Err(format!("{}: {err}", frame.path.display())),
                            drawn: None,
                        });
                    }
                    continue;
                }
            };
            let terrain = xs::TerrainProfile::from_surface_height_m(
                distances.clone(),
                columns
                    .terrain_m
                    .iter()
                    .map(|t| if t.is_finite() { *t } else { 0.0 })
                    .collect(),
            )
            .map_err(|err| format!("terrain profile: {err}"))?;

            for product in products {
                let slug = format!("{}{}", product.slug(), tag);
                let outcome = (|| -> Result<(PathBuf, Option<DrawnFill>), String> {
                    let fill = term_section(
                        &product.fill,
                        &file,
                        frame.time_index,
                        &columns,
                        &ladder,
                        &temperature_c,
                        &mut cache,
                    )?;
                    // The across frame goes through the fill's maximum
                    // column of the ALONG frame: queue it once, from the
                    // first product's fill.
                    if let Some(half) = across_pending.take() {
                        let n_points = distances.len();
                        let mut best: Option<(usize, f32)> = None;
                        for p in 0..n_points {
                            let column_max = (0..ladder.len())
                                .map(|k| fill.values[k * n_points + p])
                                .filter(|v| v.is_finite())
                                .fold(f32::MIN, f32::max);
                            if column_max > f32::MIN && best.map_or(true, |(_, b)| column_max > b) {
                                best = Some((p, column_max));
                            }
                        }
                        let through = best.map(|(p, _)| points[p]).unwrap_or(points[n_points / 2]);
                        lines.push((line.perpendicular_through(through, half / 2.0)?, "_across"));
                    }

                    // Every overlay is cut on the FULL ladder first: the
                    // height range the frame is drawn to is fitted to what
                    // the fill AND the overlays actually reach.
                    let mut overlays: Vec<(usize, TermSection, Vec<f32>, Option<f32>, &Term)> =
                        Vec::new();
                    for (index, term) in product.overlays.iter().enumerate() {
                        let overlay = term_section(
                            term,
                            &file,
                            frame.time_index,
                            &columns,
                            &ladder,
                            &temperature_c,
                            &mut cache,
                        )?;
                        let (levels, highlight) = match &term.levels {
                            Some(levels) => (levels.clone(), term.highlight),
                            None => default_levels(term, &overlay.units, &overlay.values),
                        };
                        if levels.is_empty() {
                            continue;
                        }
                        overlays.push((index, overlay, levels, highlight, term));
                    }

                    // The rung the frame is cut at, from the data: the
                    // highest one the fill reaches, and the highest one any
                    // overlay reaches its own lowest contour level on.  A
                    // section held at a fixed ceiling spends a third of its
                    // frame on empty air above the weather.
                    let n_points = distances.len();
                    let mut top_rung = signal_rung(
                        &fill.values,
                        n_points,
                        Presence::for_fill(&fill.values, fill.log_floor),
                    );
                    for (_, overlay, levels, highlight, term) in &overlays {
                        if !term_family(term).sets_the_ceiling() {
                            continue;
                        }
                        let rung = signal_rung(
                            &overlay.values,
                            n_points,
                            Presence::for_overlay(levels, *highlight),
                        );
                        top_rung = match (top_rung, rung) {
                            (Some(a), Some(b)) => Some(a.max(b)),
                            (a, b) => a.or(b),
                        };
                    }
                    let reference_m = config
                        .reference_km
                        .map(|km| km * 1000.0)
                        .unwrap_or_else(|| {
                            columns
                                .terrain_m
                                .iter()
                                .copied()
                                .filter(|t| t.is_finite())
                                .fold(0.0f64, f64::max)
                        });
                    let fitted_top = fitted_top_m(
                        top_rung.map(|k| ladder[k]),
                        reference_m,
                        config.top_km * 1000.0,
                        LADDER_STEP_M,
                    );
                    let levels_kept = ladder
                        .iter()
                        .take_while(|z| **z <= fitted_top + 1e-6)
                        .count()
                        .max(2)
                        .min(ladder.len());
                    let kept = levels_kept * n_points;
                    let ladder_fitted: Vec<f64> = ladder[..levels_kept].to_vec();
                    let axis = xs::VerticalAxis::height_meters(ladder_fitted)
                        .map_err(|err| format!("fitted height ladder: {err}"))?;
                    let temperature_fitted: Vec<f32> = temperature_c[..kept].to_vec();

                    // The headline is short on purpose: the alias (or the
                    // fill's label) and where the cut is; the colorbar
                    // label carries the full field expression and units.
                    let title_span = match &line.label {
                        Some(label) => label.clone(),
                        None => format!(
                            "{:.2},{:.2} to {:.2},{:.2}",
                            line.start.lat_deg, line.start.lon_deg, line.end.lat_deg, line.end.lon_deg
                        ),
                    };
                    let headline = product.alias.clone().unwrap_or_else(|| fill.label.clone());
                    let headline = if line.azimuths.is_some() { format!("{headline} azimuthal mean") } else { headline };
                    // A fill with no finite value anywhere (a tracer before
                    // its release, a field that is zero on this line) still
                    // draws: the panel goes out blank under its overlays and
                    // says so in the headline, because a missing frame in a
                    // gallery reads as a failure and a blank one reads as
                    // "nothing here yet".
                    let mut fill = fill;
                    fill.values.truncate(kept);
                    // A fill whose values span next to nothing (a cut
                    // through clear air for a condensate, trace values
                    // near 1e-8 g/kg) is no signal too: the range collapsed
                    // onto one value and the ramp's middle flooded the whole
                    // plot as if the cut were full of cloud.  Display units
                    // put every real section fill's span far above 1e-6.
                    let fill_has_signal = {
                        let (lo, hi) = fill.values.iter().filter(|v| v.is_finite()).fold(
                            (f32::INFINITY, f32::NEG_INFINITY),
                            |(lo, hi), v| (lo.min(*v), hi.max(*v)),
                        );
                        lo.is_finite() && hi - lo > SECTION_MIN_SIGNAL_SPAN
                    };
                    if !fill_has_signal {
                        for value in fill.values.iter_mut() {
                            *value = 0.0;
                        }
                    }
                    let title = if fill_has_signal {
                        format!("{headline} section | {title_span}")
                    } else {
                        format!("{headline} section (no signal) | {title_span}")
                    };
                    let metadata = xs::SectionMetadata::new()
                        .titled(title.clone())
                        .field(fill.label.clone(), fill.units.clone())
                        .sourced_from(section_source_label(config.theme, &config.source_label))
                        .with_attribute(
                            "start_label",
                            if line.azimuths.is_some() { "Storm centre".into() }
                            else { format!("{:.2}, {:.2}", line.start.lat_deg, line.start.lon_deg) },
                        )
                        .with_attribute(
                            "end_label",
                            if line.azimuths.is_some() { format!("360-degree mean, {} bearings", line.azimuths.unwrap()) }
                            else { format!("{:.2}, {:.2}", line.end.lat_deg, line.end.lon_deg) },
                        )
                        .with_attribute("distance_label", if line.azimuths.is_some() { "Radius (km)" } else { "Distance (km)" })
                        .with_attribute("domain", domain.clone())
                        .with_attribute("init_label", init_label.clone())
                        .with_attribute("forecast_hour", lead_label.clone())
                        .with_attribute("valid_time", valid_label.clone());
                    let section = xs::ScalarSection::new(distances.clone(), axis.clone(), fill.values.clone())
                        .map_err(|err| format!("fill section: {err}"))?
                        .with_metadata(metadata)
                        .with_terrain(terrain.clone())
                        .map_err(|err| format!("terrain: {err}"))?;
                    // Enough bands that the ramp reads as shading rather
                    // than as a staircase of eight steps.
                    let bands = 24usize;
                    let family = term_family(&product.fill);
                    // The range is settled first, because it is what says
                    // whether the bottom of the ramp is the field's
                    // absence or the lowest thing on the cut, and only
                    // absence may be faded away.
                    let named_range = product
                        .fill
                        .levels
                        .as_deref()
                        .and_then(|levels| {
                            named_fill_range(levels, family.is_diverging(), product.fill.log)
                        });
                    let drawn_range = match (named_range, fill_has_signal) {
                        (Some(named), _) => Some(named),
                        (None, true) => section.finite_range().map(|(min, max)| {
                            fill_value_range(
                                family.is_diverging(),
                                product.fill.log,
                                min,
                                max,
                            )
                        }),
                        (None, false) => None,
                    };
                    let ramp = fill_palette(config.theme, product, family, bands);
                    let ramp = if drawn_range
                        .map_or(!family.is_diverging(), |drawn| drawn.absence)
                    {
                        fade_ramp_bottom(ramp)
                    } else {
                        // A diverging ramp's absence is its MIDDLE, which
                        // already reads as nothing, and a bar that starts
                        // at the fill's own minimum has real air in its
                        // first band.  Fading either erases signal.
                        ramp
                    };
                    let mut request = build_request(config.theme, config.width, config.height)
                        .with_palette(ramp);
                    let header_plan = planned_section_header(&mut request);
                    // The built-in overlay contours the FILL; the isotherms
                    // come from tk as an explicit overlay below.
                    request.isotherms_c = Vec::new();
                    request.highlight_isotherm_c = None;
                    if !fill_has_signal {
                        // Nothing to paint: a ramp made of the plot's own
                        // background over the zeroed fill leaves the plot
                        // bare under its overlays instead of flooding it
                        // with the ramp's first colour (a transparent ramp
                        // would composite onto white, not onto the page).
                        // The bar keeps a range the product named, so an
                        // empty frame of a ranged series states the same
                        // bar as its neighbours; only an unranged empty
                        // frame takes the 0..1 placeholder.
                        let bare = request.plot_background_top;
                        let (lo, hi) = drawn_range.map_or((0.0, 1.0), |drawn| (drawn.lo, drawn.hi));
                        request = request
                            .with_value_range(lo, hi)
                            .with_palette(vec![bare; 2]);
                    } else if let Some(drawn) = drawn_range {
                        request = request.with_value_range(drawn.lo, drawn.hi);
                    }
                    let mut unit_label = if fill.units.is_empty() {
                        fill.label.clone()
                    } else {
                        format!("{} [{}]", fill.label, fill.units)
                    };
                    if product.fill.log {
                        // The fill is drawn in log10 and the bar is in
                        // that space, but the ticks print the field's own
                        // numbers at the decades: 0.1, 1, 10 g kg-1, not
                        // -1, 0, 1.  Without them the bar read exponents
                        // beside linear units.
                        unit_label.push_str(" (log scale)");
                        if let Some(drawn) = drawn_range {
                            request = request.with_colorbar_ticks(log_decade_ticks(drawn.lo, drawn.hi));
                        }
                    }
                    request = request.with_colorbar_label(unit_label);
                    let mut bundles = Vec::new();
                    for (index, mut overlay, levels, highlight, term) in overlays {
                        overlay.values.truncate(kept);
                        let overlay_section = xs::ScalarSection::new(
                            distances.clone(),
                            axis.clone(),
                            overlay.values,
                        )
                        .map_err(|err| format!("overlay {}: {err}", overlay.label))?
                        .with_metadata(
                            xs::SectionMetadata::new()
                                .field(overlay.label.clone(), overlay.units.clone()),
                        )
                        .with_terrain(terrain.clone())
                        .map_err(|err| format!("overlay terrain: {err}"))?;
                        let ink = overlay_ink(term, index);
                        let mut bundle = xs::ScalarContourOverlayBundle::new(overlay_section, levels)
                            .with_label(overlay.label.clone());
                        bundle.units = (!overlay.units.is_empty()).then(|| overlay.units.clone());
                        bundle.color = ink;
                        bundle.highlight_color = ink;
                        bundle.highlight_level = highlight;
                        bundles.push(bundle);
                    }
                    if !config.isotherms.levels_c.is_empty() {
                        let iso_section = xs::ScalarSection::new(
                            distances.clone(),
                            axis.clone(),
                            temperature_fitted,
                        )
                        .map_err(|err| format!("isotherm section: {err}"))?
                        .with_metadata(xs::SectionMetadata::new().field("T", "C"))
                        .with_terrain(terrain.clone())
                        .map_err(|err| format!("isotherm terrain: {err}"))?;
                        let mut bundle = xs::ScalarContourOverlayBundle::new(
                            iso_section,
                            config.isotherms.levels_c.clone(),
                        )
                        .with_label("T");
                        bundle.units = Some("C".to_string());
                        bundle.color = request.isotherm_color;
                        bundle.highlight_color = request.highlight_isotherm_color;
                        bundle.highlight_level = config.isotherms.highlight_c;
                        bundles.push(bundle);
                    }
                    request = request.with_contour_overlays(bundles);
                    let image = xs::render_scalar_section(&section, &request)
                        .map_err(|err| format!("render: {err}"))?;
                    let mut png = image::RgbaImage::from_raw(image.width(), image.height(), image.rgba().to_vec())
                        .ok_or_else(|| "rendered buffer does not match its dimensions".to_string())?;
                    if let Some(plan) = header_plan.as_ref() {
                        let header = rustwx_render::chrome_plan::PlanHeaderText::compose(
                            Some(&if fill_has_signal {
                                format!("{headline} ({})", map_style_domain(&domain))
                            } else {
                                format!("{headline} (no signal) ({})", map_style_domain(&domain))
                            }),
                            Some(fill.units.as_str()),
                            Some(&section_time_line(config.theme, &model, &init_label, lead, &valid_label)),
                            None,
                            config.theme.source_subtitle(Some(format!("source: {}", config.source_label))).as_deref(),
                        );
                        let (title_ink, meta_ink) = rustwx_render::chrome_plan::header_inks();
                        rustwx_render::chrome_plan::draw_plan_header(&mut png, plan, &header, title_ink, meta_ink);
                    }
                    // The same strip the map families compose, on the same
                    // two conditions: a theme with a `footer` section and
                    // caption fields the caller installed.  Neither built-in
                    // theme names one, so a default section is the bytes the
                    // renderer produced.
                    let png = match (
                        rustwx_render::theme::active_theme().footer.as_ref(),
                        rustwx_render::footer_fields(),
                    ) {
                        (Some(footer_theme), Some(fields)) => {
                            let fields = fields.with_panel_defaults(Some(title.as_str()), None);
                            rustwx_render::footer::compose(&png, footer_theme, &fields)
                        }
                        _ => png,
                    };
                    let name = format!(
                        "rustwx_wrf_{date}_{oh}z_f{:03}_{domain}_{slug}_{suffix}.png",
                        lead / 3_600
                    );
                    let output = config.out_dir.join(name);
                    png.save(&output)
                        .map_err(|err| format!("write {}: {err}", output.display()))?;
                    Ok((output, drawn_range))
                })();
                match &outcome {
                    Ok(_) => rendered += 1,
                    Err(_) => failed += 1,
                }
                let (result, drawn) = match outcome {
                    Ok((output, drawn)) => (Ok(output), drawn),
                    Err(err) => (Err(err), None),
                };
                emit(SectionOutcome {
                    slug,
                    result,
                    drawn,
                });
            }
        }
    }
    Ok((rendered, failed))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_section_names_its_domain_as_the_maps_do() {
        assert_eq!(map_style_domain("d01-3km"), "d01 3 km");
        assert_eq!(map_style_domain("d01-2.25km"), "d01 2.25 km");
        assert_eq!(map_style_domain("d02-750m"), "d02 750 m");
        assert_eq!(map_style_domain("d03"), "d03");
        assert!(!section_time_line(&RenderTheme::default_theme(), "WOOF", "05/26 15Z", 7_200, "05/26 17:00Z").contains("d01"));
    }

    #[test]
    fn the_path_is_sampled_at_no_coarser_than_half_the_grid_spacing() {
        // 1 km grid, 120 km line: two samples a cell, both ends included.
        let count = section_sample_count(120.0, 1000.0);
        assert!(count >= 241, "{count} samples over 120 km at 1 km");
        let spacing_km = 120.0 / (count - 1) as f64;
        assert!(spacing_km <= 0.5 + 1e-9, "{spacing_km} km between samples");
        // The old clamp stopped at 400 and made a long cut coarser than the
        // grid it was cut from.
        let long = section_sample_count(900.0, 1000.0);
        assert!(long > 400, "{long}");
        assert!(900.0 / (long - 1) as f64 <= 0.5 + 1e-9);
        // A 3.75 km mesh gets the same rule, and the bounds hold.
        assert!(section_sample_count(30.0, 3_750.0) >= MIN_SECTION_SAMPLES);
        assert_eq!(section_sample_count(1.0, 1000.0), MIN_SECTION_SAMPLES);
        assert_eq!(section_sample_count(40_000.0, 1000.0), MAX_SECTION_SAMPLES);
        assert_eq!(section_sample_count(100.0, 0.0), MIN_SECTION_SAMPLES);
    }

    #[test]
    fn the_vertical_ladder_is_no_coarser_than_a_hundred_metres() {
        assert!(LADDER_STEP_M <= 100.0, "{LADDER_STEP_M} m rungs");
        let ladder: Vec<f64> = (0..)
            .map(|k| f64::from(k) * LADDER_STEP_M)
            .take_while(|z| *z <= 10_000.0 + 1e-6)
            .collect();
        assert_eq!(ladder.len(), 101);
        assert!((ladder[1] - ladder[0] - 100.0).abs() < 1e-9);
    }

    #[test]
    fn the_height_range_fits_the_data_and_respects_both_its_floors() {
        let step = LADDER_STEP_M;
        // Signal to 4 km over 1.2 km of terrain: a kilometre of air above
        // the signal, and the ceiling is nowhere near.
        assert_eq!(fitted_top_m(Some(4_000.0), 1_200.0, 14_000.0, step), 5_000.0);
        // Signal barely above a high crest: two kilometres above the
        // reference wins.
        assert_eq!(fitted_top_m(Some(3_100.0), 3_800.0, 14_000.0, step), 5_800.0);
        // The caller's ceiling caps it.
        assert_eq!(fitted_top_m(Some(9_500.0), 1_000.0, 8_000.0, step), 8_000.0);
        // No signal at all: the ceiling, which is what a blank frame drew
        // before any of this.
        assert_eq!(fitted_top_m(None, 500.0, 10_000.0, step), 10_000.0);
        // The fitted top always lands on a rung.
        let top = fitted_top_m(Some(3_950.0), 0.0, 14_000.0, step);
        assert!((top % step).abs() < 1e-6, "{top}");
    }

    #[test]
    fn a_field_that_is_finite_and_zero_does_not_hold_the_ceiling_open() {
        // Four points, five rungs.  The field is present on the bottom two
        // and exactly zero above: the frame is cut at rung 1, not rung 4.
        let n_points = 4usize;
        let mut values = vec![0.0f32; n_points * 5];
        values[..n_points].fill(3.0);
        values[n_points..2 * n_points].fill(1.0);
        let linear = Presence::for_fill(&values, None);
        assert!(linear.at_least > 0.0 && linear.at_least < 1.0, "{linear:?}");
        assert_eq!(signal_rung(&values, n_points, linear), Some(1));
        // A `~log` fill sits ON its floor where it is absent, so the floor
        // itself is not signal and one decade above it is.
        let mut logged = vec![-10.0f32; n_points * 5];
        logged[2 * n_points..3 * n_points].fill(-4.0);
        let log_test = Presence::for_fill(&logged, Some(-10.0));
        assert!(!log_test.holds(-10.0));
        assert!(log_test.holds(-4.0));
        assert_eq!(signal_rung(&logged, n_points, log_test), Some(2));
        // An overlay counts only where it reaches its own lowest level.
        let mut updraft = vec![0.05f32; n_points * 5];
        updraft[n_points..2 * n_points].fill(4.0);
        let overlay = Presence::for_overlay(&[1.0, 2.0, 5.0], Some(5.0));
        assert_eq!(signal_rung(&updraft, n_points, overlay), Some(1));
        // Nothing anywhere is None, and the frame falls back to the ceiling.
        assert_eq!(
            signal_rung(&vec![0.0f32; n_points * 5], n_points, overlay),
            None
        );
    }

    #[test]
    fn a_fill_is_never_drawn_on_a_bar_it_uses_less_than_half_of() {
        // Absolute temperature is the field the zero anchor ruins: a cut
        // three kilometres deep spans under thirty kelvin somewhere above
        // 290, and on a bar that starts at zero that is under a tenth of
        // the ramp, so the whole cut comes out one colour.
        let drawn = fill_value_range(false, false, 293.1, 321.1);
        assert_eq!(
            (drawn.lo, drawn.hi),
            (293.1, 321.1),
            "the bar starts at the coldest air"
        );
        assert!(!drawn.absence, "the first band is air, not absence");
        assert_eq!(drawn.rule, FillRangeRule::OwnMinimum);
        // The same field through the whole column is no better off: 240 to
        // 321 is a quarter of a bar that starts at zero.
        let deep = fill_value_range(false, false, 240.0, 321.0);
        assert_eq!((deep.lo, deep.hi, deep.absence), (240.0, 321.0, false));
        assert_eq!(deep.rule, FillRangeRule::OwnMinimum);
    }

    #[test]
    fn a_fill_that_reaches_down_toward_zero_keeps_the_zero_anchor() {
        // Mixing ratio, a thin trace of it, wind speed, and a field that is
        // flat: zero is the floor of all of them and an empty column has to
        // read as the bottom of the ramp.
        for (min, max) in [(0.0f32, 3e-3f32), (1e-6, 2e-3), (2.0, 60.0), (7.0, 7.0)] {
            let drawn = fill_value_range(false, false, min, max);
            assert_eq!(drawn.lo, 0.0, "fill {min}..{max} lost its zero anchor");
            assert!(drawn.hi > 0.0, "fill {min}..{max} has no top");
            assert!(
                drawn.absence,
                "fill {min}..{max} stopped calling its floor absence"
            );
            assert_eq!(drawn.rule, FillRangeRule::ZeroAnchor);
        }
        // A fill with nothing above zero still gets a bar to draw on.
        let flat = fill_value_range(false, false, 0.0, 0.0);
        assert_eq!((flat.lo, flat.hi, flat.absence), (0.0, 1.0, true));
    }

    #[test]
    fn the_zero_anchor_is_given_up_exactly_where_the_rule_says() {
        // The rule is a fraction of the bar the zero anchor produces, so
        // the boundary is the only place worth pinning: a span of exactly
        // half the top keeps zero, a hair under it does not.
        let top = 100.0f32;
        let keeps = top * (1.0 - ZERO_ANCHOR_MIN_BAR_FRACTION);
        let kept = fill_value_range(false, false, keeps, top);
        assert_eq!(kept.lo, 0.0);
        assert_eq!(kept.rule, FillRangeRule::ZeroAnchor);
        let gives_up = keeps + 1.0;
        let given_up = fill_value_range(false, false, gives_up, top);
        assert_eq!(given_up.lo, gives_up);
        assert_eq!(given_up.rule, FillRangeRule::OwnMinimum);
    }

    #[test]
    fn a_fill_that_crosses_zero_or_diverges_is_unchanged_by_the_rule() {
        // A Celsius cut crosses zero, so zero was never its floor.
        let celsius = fill_value_range(false, false, -52.0, 31.0);
        assert_eq!((celsius.lo, celsius.hi, celsius.absence), (-52.0, 31.0, false));
        assert_eq!(celsius.rule, FillRangeRule::CrossesZero);
        // A diverging fill stays symmetric about its middle.
        let diverging = fill_value_range(true, false, -3.0, 11.0);
        assert_eq!((diverging.lo, diverging.hi), (-11.0, 11.0));
        assert!(!diverging.absence);
        assert_eq!(diverging.rule, FillRangeRule::Symmetric);
        // A log fill's own minimum IS its floor and stays absence.
        let log = fill_value_range(false, true, 1e-7, 4.0);
        assert_eq!((log.lo, log.hi, log.absence), (1e-7, 4.0, true));
        assert_eq!(log.rule, FillRangeRule::LogFloor);
    }

    #[test]
    fn a_log_fills_bar_prints_the_fields_own_numbers_at_the_decades() {
        // A log fill spanning 0.1 to 10 g kg-1 is drawn on -1..1; the bar
        // reads 0.1, 1, 10, never -1, 0, 1.
        let ticks = log_decade_ticks(-1.0, 1.0);
        assert_eq!(
            ticks,
            vec![
                (-1.0, "0.1".to_string()),
                (0.0, "1".to_string()),
                (1.0, "10".to_string())
            ]
        );
        // Six decades below the maximum, the way the log floor sets them.
        let deep: Vec<String> = log_decade_ticks(-6.4, 0.6).into_iter().map(|(_, l)| l).collect();
        assert_eq!(deep, vec!["1e-6", "1e-5", "1e-4", "0.001", "0.01", "0.1", "1"]);
        // Under a decade of range: the 2x and 5x steps carry the numbers.
        let narrow: Vec<String> = log_decade_ticks(-1.9, -1.2).into_iter().map(|(_, l)| l).collect();
        assert_eq!(narrow, vec!["0.02", "0.05"]);
        assert!(log_decade_ticks(1.0, 1.0).is_empty());
        assert!(log_decade_ticks(f32::NAN, 1.0).is_empty());
        assert_eq!(decade_label(2e4), "2e4");
        assert_eq!(decade_label(1000.0), "1000");
    }

    #[test]
    fn a_named_fill_range_is_the_bar_the_product_spelled() {
        // The list on the fill term names the bar: lowest to highest, in
        // the units the section draws (g kg-1 for a mixing ratio).
        let named = named_fill_range(&[0.01, 0.1], false, false).expect("a range");
        assert_eq!((named.lo, named.hi, named.absence), (0.01, 0.1, true));
        assert_eq!(named.rule, FillRangeRule::Named);
        assert_eq!(named.rule.label(), "named");
        // Order in the list does not matter, and a log fill takes the
        // decades of the named levels.
        let log = named_fill_range(&[10.0, 0.1, 1.0], false, true).expect("a range");
        assert_eq!((log.lo, log.hi), (-1.0, 1.0));
        // A diverging fill keeps zero as its middle.
        let diverging = named_fill_range(&[-2.0, 8.0], true, false).expect("a range");
        assert_eq!((diverging.lo, diverging.hi, diverging.absence), (-8.0, 8.0, false));
        // No usable level, or no span, names no range: a log fill of a
        // list at zero, a single level, an empty list.
        assert!(named_fill_range(&[0.0], false, true).is_none());
        assert!(named_fill_range(&[5.0, 5.0], false, false).is_none());
        assert!(named_fill_range(&[], false, false).is_none());
    }

    #[test]
    fn the_bottom_band_is_faded_only_where_the_range_calls_it_absence() {
        // The fade exists to stop absence flooding the panel.  Every range
        // that starts at the fill's own minimum has real air in that band,
        // so the two decisions are one decision and this is the pin on it.
        for (diverging, log, min, max) in [
            (false, false, 0.0f32, 3e-3f32),
            (false, false, 2.0, 60.0),
            (false, true, 1e-7, 4.0),
            (false, false, 293.1, 321.1),
            (false, false, -52.0, 31.0),
            (true, false, -3.0, 11.0),
        ] {
            let drawn = fill_value_range(diverging, log, min, max);
            let starts_at_the_fills_own_floor = drawn.lo == 0.0 || (log && drawn.lo == min);
            assert_eq!(
                drawn.absence, starts_at_the_fills_own_floor,
                "fill {min}..{max} (diverging {diverging}, log {log}) fades the wrong band"
            );
        }
    }

    #[test]
    fn a_sequential_fills_ramp_fades_out_at_its_absent_end() {
        let ramp = fade_ramp_bottom(family_palette(FieldFamily::Tracer, 12));
        assert_eq!(ramp[0].a, 0, "absence is not painted");
        assert!(ramp[1].a < ramp[2].a && ramp[2].a < ramp[3].a, "the fade is graded");
        assert_eq!(ramp[11].a, 255, "the loud end stays loud");
        // Hues are untouched: only the alpha moves.
        let bare = family_palette(FieldFamily::Tracer, 12);
        assert_eq!((ramp[0].r, ramp[0].g, ramp[0].b), (bare[0].r, bare[0].g, bare[0].b));
    }

    #[test]
    fn only_what_is_in_the_air_raises_the_frames_ceiling() {
        // The plume, the cloud water and the ice set the top; the air's own
        // motion and its temperature do not, because both are present
        // through the whole column and would pin every frame to the
        // ceiling.
        for family in [
            FieldFamily::Tracer,
            FieldFamily::LiquidWater,
            FieldFamily::Ice,
            FieldFamily::Condensate,
        ] {
            assert!(family.sets_the_ceiling(), "{family:?}");
        }
        assert!(!FieldFamily::VerticalVelocity.sets_the_ceiling());
        assert!(!FieldFamily::Temperature.sets_the_ceiling());
        assert!(!FieldFamily::Other.sets_the_ceiling());
    }

    #[test]
    fn every_field_family_takes_the_trees_own_weather_ramp() {
        let family_of = |token: &str| {
            term_family(&parse_section_product(&format!("xsec:{token}")).unwrap().fill)
        };
        assert_eq!(family_of("TR17_1+TR17_2~log"), FieldFamily::Tracer);
        assert_eq!(family_of("QCLOUD@cold"), FieldFamily::LiquidWater);
        assert_eq!(family_of("QICE+QSNOW"), FieldFamily::Ice);
        assert_eq!(family_of("wa"), FieldFamily::VerticalVelocity);
        assert_eq!(family_of("tk"), FieldFamily::Temperature);
        assert_eq!(family_of("QCLOUD+QICE"), FieldFamily::Condensate);
        assert_eq!(family_of("rh"), FieldFamily::Moisture);

        // Vertical velocity is the only diverging one, and its ramp really
        // does cross a light middle.
        assert!(FieldFamily::VerticalVelocity.is_diverging());
        assert!(!FieldFamily::Tracer.is_diverging());
        let updraft = family_palette(FieldFamily::VerticalVelocity, 9);
        let middle = updraft[4];
        assert!(
            middle.r > 200 && middle.g > 200 && middle.b > 200,
            "the diverging ramp's middle is its light one: {middle:?}"
        );
        assert!(updraft[0].b > updraft[0].r, "descent is the cold end");
        assert!(updraft[8].r > updraft[8].b, "ascent is the warm end");

        // The tracer ramp is the map families' own, at full saturation.
        let tracer = family_palette(FieldFamily::Tracer, 10);
        assert!(
            tracer.iter().all(|c| c.a == 255),
            "a section's tracer fill is opaque"
        );
        let map_ramp = rustwx_products::plot_design::tracer_scale_colors();
        assert_eq!(
            (tracer[0].r, tracer[0].g, tracer[0].b),
            (map_ramp[0].r, map_ramp[0].g, map_ramp[0].b),
            "the same hues the map paints a carried scalar with"
        );

        // Water and ice are told apart at a glance.
        let water = family_palette(FieldFamily::LiquidWater, 8);
        let ice = family_palette(FieldFamily::Ice, 8);
        let apart: i32 = water
            .iter()
            .zip(ice.iter())
            .map(|(w, i)| {
                (i32::from(w.r) - i32::from(i.r)).abs()
                    + (i32::from(w.g) - i32::from(i.g)).abs()
                    + (i32::from(w.b) - i32::from(i.b)).abs()
            })
            .max()
            .unwrap();
        assert!(apart > 60, "cloud water and ice read as one ladder ({apart})");
    }

    #[test]
    fn a_theme_moves_the_canvas_but_never_a_field_palette() {
        let product =
            parse_section_product("xsec:agent:TR17_1~log/QCLOUD@cold/QICE/wa").unwrap();
        let family = term_family(&product.fill);
        let bare = family_palette(family, 12);

        // A theme with its own sequential brand ramp changes NOTHING about
        // the fill: a brand ramp painted over every scientific field is
        // what made the delivered gallery monotone.
        let mut brand = RenderTheme::dark_theme();
        brand.sequential = vec![
            rustwx_render::Rgba::new(0x0b, 0x3d, 0x3f),
            rustwx_render::Rgba::new(0x7f, 0xd8, 0xd2),
        ];
        assert_eq!(fill_palette(&brand, &product, family, 12), bare);
        assert_eq!(fill_palette(&RenderTheme::default_theme(), &product, family, 12), bare);

        // Naming the product IS the seam: that, and only that, overrides.
        let mut named = brand.clone();
        named.products.insert(
            "xsec_agent".to_string(),
            vec![
                rustwx_render::Rgba::new(0x0b, 0x3d, 0x3f),
                rustwx_render::Rgba::new(0x7f, 0xd8, 0xd2),
            ],
        );
        let overridden = fill_palette(&named, &product, family, 12);
        assert_ne!(overridden, bare);
        assert_eq!(overridden.len(), 12);
        assert_eq!(overridden[0], xs::Color::rgba(0x0b, 0x3d, 0x3f, 255));

        // And a theme that names some OTHER product leaves this one alone.
        let mut elsewhere = brand.clone();
        elsewhere
            .products
            .insert("xsec_difference".to_string(), vec![rustwx_render::Rgba::new(1, 2, 3)]);
        assert_eq!(fill_palette(&elsewhere, &product, family, 12), bare);
    }

    #[test]
    fn each_overlay_family_gets_its_own_saturated_ink() {
        let product =
            parse_section_product("xsec:agent:TR17_1~log/QCLOUD@cold/QICE/wa/tk").unwrap();
        let inks: Vec<xs::Color> = product
            .overlays
            .iter()
            .enumerate()
            .map(|(index, term)| overlay_ink(term, index))
            .collect();
        assert_eq!(inks.len(), 4);
        for (a, left) in inks.iter().enumerate() {
            for right in inks.iter().skip(a + 1) {
                let apart = (i32::from(left.r) - i32::from(right.r)).abs()
                    + (i32::from(left.g) - i32::from(right.g)).abs()
                    + (i32::from(left.b) - i32::from(right.b)).abs();
                assert!(apart > 120, "two overlay inks are the same colour: {apart}");
            }
        }
        // Cloud water is green, ice is cyan, updraft is magenta.
        assert!(inks[0].g > inks[0].r && inks[0].g > inks[0].b);
        assert!(inks[1].b > inks[1].r && inks[1].g > inks[1].r);
        assert!(inks[2].r > inks[2].g && inks[2].b > inks[2].g);
    }

    #[test]
    fn the_minus_ten_isotherm_is_highlighted_by_default() {
        assert_eq!(Isotherms::default().highlight_c, Some(-10.0));
        assert!(Isotherms::default().levels_c.contains(&-10.0));
        // A caller still names its own set, highlight and all.
        let named = Isotherms::parse("0,-5,-10@-5").unwrap();
        assert_eq!(named.highlight_c, Some(-5.0));
        assert!(Isotherms::parse("none").unwrap().highlight_c.is_none());
    }

    #[test]
    fn a_theme_that_names_no_source_label_leaves_the_section_header_alone() {
        let plain = build_request(&RenderTheme::dark_theme(), 1200, 900);
        assert!(plain.source_label.is_none());
        let mut theme = RenderTheme::dark_theme();
        theme.source_label = Some("OTHER 0.2.3".to_string());
        let marked = build_request(&theme, 1200, 900);
        assert_eq!(marked.source_label.as_deref(), Some("OTHER 0.2.3"));
    }

    #[test]
    fn the_section_s_provenance_takes_the_theme_s_source_label_when_it_names_one() {
        assert_eq!(section_source_label(&RenderTheme::default_theme(), "ArWen"), "ArWen");
        assert_eq!(section_source_label(&RenderTheme::dark_theme(), "ArWen"), "ArWen");
        let mut theme = RenderTheme::dark_theme();
        theme.source_label = Some("my model 1.2.3".to_string());
        assert_eq!(section_source_label(&theme, "ArWen"), "my model 1.2.3");
    }

    #[test]
    fn woof_section_headers_use_public_labels_and_generic_headers_keep_their_bytes() {
        let generic = RenderTheme::default_theme();
        let expected = "Init 05/26 15Z | F002 | Valid 05/26 17Z | WRF";
        assert_eq!(section_time_line(&generic, "WRF", "05/26 15Z", 7_200, "05/26 17:00Z"), expected);
        // The engine's own history files carry WOOF into the line with no theme.
        assert_eq!(
            section_time_line(&generic, "WOOF", "05/26 15Z", 7_200, "05/26 17:00Z"),
            "Init 05/26 15Z | F002 | Valid 05/26 17Z | WOOF"
        );
        assert!(build_request(&generic, 640, 480).source_label.is_none());
        for name in ["woof-light", "woof-dark"] {
            let theme = RenderTheme::builtin(name).unwrap();
            let line = section_time_line(&theme, "WRF", "05/26 15Z", 7_200, "05/26 17:00Z");
            assert_eq!(line, "Init 05/26 15Z | F002 | Valid 05/26 17Z | WOOF");
            assert_eq!(build_request(&theme, 640, 480).source_label.as_deref(), Some("Recast WOOF"));
            assert_eq!(section_source_label(&theme, "ArWen"), "Recast WOOF");
        }
        let mut templated = generic;
        templated.source_label = Some("Recast WOOF {version}".into());
        let mark = build_request(&templated, 640, 480).source_label.unwrap();
        assert!(mark.starts_with("Recast WOOF") && !mark.contains('{'), "{mark}");
        assert_eq!(section_source_label(&templated, "ArWen"), mark);
    }

    #[test]
    fn the_product_grammar_parses_sums_modifiers_levels_and_aliases() {
        let product = parse_section_product(
            "xsec:agent:TR17_1+TR17_2~log/QCLOUD@cold/QICE+QSNOW=0.01,0.1/wa=1,2,5,10@5",
        )
        .expect("parses");
        assert_eq!(product.alias.as_deref(), Some("agent"));
        assert_eq!(product.fill.names, vec!["TR17_1", "TR17_2"]);
        assert!(product.fill.log);
        assert!(!product.fill.cold_only);
        assert_eq!(product.overlays.len(), 3);
        assert!(product.overlays[0].cold_only);
        assert_eq!(product.overlays[1].levels.as_deref(), Some(&[0.01, 0.1][..]));
        assert_eq!(product.overlays[2].levels.as_deref(), Some(&[1.0, 2.0, 5.0, 10.0][..]));
        assert_eq!(product.overlays[2].highlight, Some(5.0));
        assert_eq!(product.slug(), "xsec_agent");

        let plain = parse_section_product("xsec:QCLOUD").expect("parses");
        assert_eq!(plain.slug(), "xsec_qcloud");
        let hashed = parse_section_product("xsec:QCLOUD/wa").expect("parses");
        assert!(hashed.slug().starts_with("xsec_qcloud_"));
        assert_ne!(hashed.slug(), plain.slug());
    }

    #[test]
    fn malformed_tokens_are_refused_by_name() {
        for bad in ["xsec:", "xsec:Q CLOUD", "xsec:QCLOUD//wa", "xsec:wa=1,x", "xsec:a b:QCLOUD"] {
            let err = parse_section_product(bad).expect_err(bad);
            assert!(err.contains("xsec:"), "{err}");
        }
    }

    #[test]
    fn the_spec_splits_into_store_and_section_families() {
        let (store, sections) =
            split_product_spec("composite_reflectivity, xsec:QCLOUD/wa ,var:wrf_qcloud_colmax")
                .expect("splits");
        assert_eq!(store, "composite_reflectivity,var:wrf_qcloud_colmax");
        assert_eq!(sections.len(), 1);
        // Comma-separated levels inside a term survive the product split,
        // and the product after them is still a product.
        let (store, sections) = split_product_spec(
            "xsec:agent:TR17_1~log/QCLOUD@cold/wa=1,2,5,10@5,2m_temperature,xsec:tk=-20,-10,0,10m_wind_speed_and_direction",
        )
        .expect("splits");
        assert_eq!(store, "2m_temperature,10m_wind_speed_and_direction");
        assert_eq!(sections.len(), 2);
        assert_eq!(sections[0].overlays[1].levels.as_deref(), Some(&[1.0, 2.0, 5.0, 10.0][..]));
        assert_eq!(sections[0].overlays[1].highlight, Some(5.0));
        assert_eq!(sections[1].fill.levels.as_deref(), Some(&[-20.0, -10.0, 0.0][..]));
        assert!(is_level_token("-10") && is_level_token("0.5") && is_level_token("10@5"));
        assert!(!is_level_token("2m_temperature") && !is_level_token("10m_wind_speed_and_direction"));
        // A level list whose last number is followed by the next '/' term
        // is still one term: the continuation token carries the overlay,
        // and the store product after it is still a product.
        let (store, sections) = split_product_spec("xsec:QCLOUD=0.01,0.1/wa").expect("splits");
        assert!(store.is_empty(), "store products invented from a level list: {store}");
        assert_eq!(sections.len(), 1);
        assert_eq!(sections[0].fill.levels.as_deref(), Some(&[0.01, 0.1][..]));
        assert_eq!(sections[0].overlays.len(), 1);
        let (store, sections) =
            split_product_spec("xsec:QCLOUD=0.01,0.1/wa,composite_reflectivity").expect("splits");
        assert_eq!(store, "composite_reflectivity");
        assert_eq!(sections.len(), 1);
        assert_eq!(sections[0].fill.levels.as_deref(), Some(&[0.01, 0.1][..]));
        assert_eq!(sections[0].overlays.len(), 1);
        let (store, sections) =
            split_product_spec("xsec:wa=1,2,5,10@5/tk=-20,-10,0/QCLOUD,2m_temperature").expect("splits");
        assert_eq!(store, "2m_temperature");
        assert_eq!(sections[0].fill.levels.as_deref(), Some(&[1.0, 2.0, 5.0, 10.0][..]));
        assert_eq!(sections[0].fill.highlight, Some(5.0));
        assert_eq!(sections[0].overlays[0].levels.as_deref(), Some(&[-20.0, -10.0, 0.0][..]));
        assert_eq!(sections[0].overlays.len(), 2);
        assert!(continues_level_list("0.1/wa") && continues_level_list("10@5/tk=-20"));
        assert!(!continues_level_list("wa/tk") && !continues_level_list("0.1/"));
        let (store, sections) = split_product_spec("all").expect("splits");
        assert_eq!(store, "all");
        assert!(sections.is_empty());
        let (store, sections) = split_product_spec("xsec:tk").expect("splits");
        assert!(store.is_empty());
        assert_eq!(sections.len(), 1);
    }

    #[test]
    fn isotherms_parse_with_a_highlight_that_must_be_a_level() {
        let iso = Isotherms::parse("0,-5,-10,-15,-20@-10").expect("parses");
        assert_eq!(iso.levels_c, vec![-20.0, -15.0, -10.0, -5.0, 0.0]);
        assert_eq!(iso.highlight_c, Some(-10.0));
        assert!(Isotherms::parse("0,-10@-7").is_err());
        assert!(Isotherms::parse("none").expect("none").levels_c.is_empty());
        assert_eq!(Isotherms::default().levels_c.len(), 5);
    }

    #[test]
    fn section_lines_come_from_four_numbers_or_a_json_file() {
        let line = SectionLine::parse("38.32,-99.0,38.32,-98.4").expect("parses");
        assert!((line.length_km() - 52.4).abs() < 1.0, "{}", line.length_km());
        assert!(SectionLine::parse("38.32,-99.0,38.32,-99.0").is_err());
        assert!(SectionLine::parse("nowhere.json").is_err());

        let dir = std::env::temp_dir().join(format!("rw_wrfbatch_section_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("line.json");
        std::fs::write(
            &file,
            r#"{"points": [[38.32, -99.0], [38.32, -98.8], [38.32, -98.4]], "extend_km": 20, "label": "release line"}"#,
        )
        .unwrap();
        let extended = SectionLine::parse(file.to_str().unwrap()).expect("parses");
        assert!((extended.length_km() - 92.4).abs() < 1.0, "{}", extended.length_km());
        assert_eq!(extended.label.as_deref(), Some("release line"));
        assert!(extended.start.lon_deg < -99.0 && extended.end.lon_deg > -98.4);
        std::fs::write(&file, r#"{"start": [38.0, -99.0], "finish": [38.0, -98.0]}"#).unwrap();
        let err = SectionLine::parse(file.to_str().unwrap()).expect_err("unknown key");
        assert!(err.contains("finish"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_perpendicular_line_crosses_the_original_at_its_midpoint() {
        let line = SectionLine::parse("38.32,-99.0,38.32,-98.4").expect("parses");
        let through = xs::GeoPoint::new(38.32, -98.7).unwrap();
        let across = line.perpendicular_through(through, 30.0).expect("across");
        assert!((across.length_km() - 60.0).abs() < 0.5, "{}", across.length_km());
        assert!((across.start.lon_deg - across.end.lon_deg).abs() < 0.05);
        // The across line straddles the original: one end north of it, the
        // other south (left of an eastbound line is north).
        assert!((across.start.lat_deg - 38.32) * (across.end.lat_deg - 38.32) < 0.0);
        assert!(across.start.lat_deg > 38.32, "left of an eastbound line is north");
    }

    #[test]
    fn columns_land_on_the_height_ladder_linearly_and_nan_outside() {
        let z = [100.0, 600.0, 1100.0, 2100.0];
        let v = [1.0, 2.0, 3.0, 5.0];
        let ladder = [0.0, 100.0, 350.0, 1600.0, 2100.0, 2500.0];
        let mut out = [0.0f32; 6];
        column_on_ladder(&z, &v, &ladder, &mut out);
        assert!(out[0].is_nan());
        assert!((out[1] - 1.0).abs() < 1e-6);
        assert!((out[2] - 1.5).abs() < 1e-6);
        assert!((out[3] - 4.0).abs() < 1e-6);
        assert!((out[4] - 5.0).abs() < 1e-6);
        assert!(out[5].is_nan());
    }

    #[test]
    fn a_regular_grid_locates_points_with_bilinear_weights() {
        let (ny, nx) = (5usize, 7usize);
        let mut lat = Vec::new();
        let mut lon = Vec::new();
        for j in 0..ny {
            for i in 0..nx {
                lat.push(38.0 + j as f64 * 0.1);
                lon.push(-99.0 + i as f64 * 0.1);
            }
        }
        let locator = GridLocator { ny, nx, lat, lon };
        let weights = locator.weights(38.25, -98.75).expect("inside");
        let total: f64 = weights.iter().map(|(_, w)| w).sum();
        assert!((total - 1.0).abs() < 1e-9);
        for (cell, w) in weights {
            assert!((w - 0.25).abs() < 1e-6, "cell {cell} weight {w}");
        }
        assert!(locator.weights(39.5, -98.75).is_none(), "north of the grid");
        assert!(locator.weights(38.2, -97.0).is_none(), "east of the grid");
        let exact = locator.weights(38.2, -98.8).expect("on a node");
        assert!((exact[0].1 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn units_follow_the_wrf_naming_convention() {
        assert_eq!(units_and_scale("QCLOUD", ""), ("g kg-1".to_string(), 1000.0));
        assert_eq!(units_and_scale("QNRAIN", ""), ("kg-1".to_string(), 1.0));
        assert_eq!(units_and_scale("wa", "m s-1"), ("m s-1".to_string(), 1.0));
        assert_eq!(units_and_scale("tk", "K"), ("K".to_string(), 1.0));
        let (levels, highlight) = default_levels(
            &parse_section_product("xsec:wa").unwrap().fill,
            "m s-1",
            &[],
        );
        assert_eq!(levels, vec![1.0, 2.0, 5.0, 10.0]);
        assert_eq!(highlight, Some(5.0));
        assert_eq!(nice_levels(0.0, 10.0, 5), vec![0.0, 2.0, 4.0, 6.0, 8.0, 10.0]);
    }

    #[test]
    fn the_dark_theme_moves_the_page_and_the_default_leaves_it() {
        let plain = build_request(&RenderTheme::default_theme(), 640, 480);
        let stock = xs::CrossSectionRenderRequest::default();
        assert_eq!(plain.page_background_top, stock.page_background_top);
        let dark = build_request(&RenderTheme::dark_theme(), 640, 480);
        assert_eq!(dark.page_background_top, xs::Color::rgba(0, 0, 0, 255));
        assert_ne!(dark.text_color, stock.text_color);
    }
    #[test]
    fn radial_mean_interpolates_each_column_before_averaging() {
        let heights = Native3D { data: vec![0., 0., 1000., 2000.], nz: 2, cells: 2, units: "m".into() };
        let mut columns = SampledColumns { weights: vec![Some(vec![(0, 0.5), (1, 0.5)])],
            full_ring: true, heights, terrain_m: vec![0.] };
        let field = Native3D { data: vec![0., 10., 1000., 2010.], nz: 2, cells: 2, units: "K".into() };
        let mean = columns.section(&field, &[500., 1500.]);
        assert!((mean[0] - 505.).abs() < 1.0e-5);
        assert!(mean[1].is_nan(), "a missing bearing must not become a partial-ring mean");
        columns.full_ring = false;
        assert!((columns.section(&field, &[1500.])[0] - 1510.).abs() < 1.0e-5);
    }

    #[test]
    fn radial_json_preserves_geodesic_radius_and_rejects_mixed_modes() {
        let path = std::env::temp_dir().join(format!("radial-section-{}.json", std::process::id()));
        std::fs::write(&path, r#"{"center":[22,-93],"radius_km":200,"azimuths":72}"#).unwrap();
        let line = SectionLine::parse(path.to_str().unwrap()).unwrap();
        assert_eq!(line.azimuths, Some(72));
        assert!((line.length_km() - 200.).abs() < 1.0e-6);
        std::fs::write(&path, r#"{"center":[22,-93],"radius_km":200,"start":[20,-90]}"#).unwrap();
        assert!(SectionLine::parse(path.to_str().unwrap()).is_err());
        std::fs::remove_file(path).unwrap();
    }

}
