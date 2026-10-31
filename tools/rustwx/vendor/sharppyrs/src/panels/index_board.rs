//! SHARPpy-Reimagined bottom index board: port of
//! `sharpmod/viz/index_board.py` laid out across THREE columns:
//!
//! 1. **Convective**: parcel table (PCL/CAPE/CINH/LCL/LI/LFC/EL/MPL for
//!    SFC/ML/FCST/MU), the thermo stats block (3 sub-columns), and the
//!    lapse-rate box paired with the colored Severe Weather Composite box
//!    (Supercell Comp / STP(cin) / STP(fix) / SHIP / Derecho Comp).
//! 2. **Kinematics**: SRH/Shear/MnWind/SRW table, BRN Shear / 4-6km SR
//!    wind, the Storm-Motion vectors, and the 1 km / 6 km AGL wind-barb pair.
//! 3. **Composite Indices**: the SHIP box-whisker chart (drawn by
//!    [`crate::panels::ship_inset`]), EHI 0-1/0-3 km, VGP, Peskov, MCS,
//!    SWEAT, MOSHE, LRGHAIL, and HGZ CAPE / NSTP / NCAPE / ECAPE / LSCP /
//!    WBZ Height.
//!
//! Value + smaller-unit text rendering ports `sharpmod/viz/unit_text.py`;
//! the per-value tier tables port `sharpmod/colors.py`. Nothing is
//! recomputed: parcels come from [`Profile`], every scalar from
//! [`DerivedParams`] (NaN renders as `--`).

use egui::{Align2, Color32, FontId, Painter, Rect, Stroke, StrokeKind, Vec2, pos2, vec2};

use sharprs::params::cape::ParcelResult;

use crate::Profile;
use crate::derived::{Comp, DerivedParams, Vect};
use crate::diagnostic_table::{
    DiagnosticTablePanelKind, DiagnosticTableRow, NativeDiagnosticPatchBoard,
    NativeDiagnosticSlotPatch,
};
use crate::skewt::SkewTStyle;
use crate::utils::{float2str, int2str, qc};

/// The string drawn in place of an unavailable value.
const MISS: &str = "--";
const PARCEL_MATRIX_HEADERS: [&str; 8] = ["PCL", "CAPE", "CINH", "LCL", "LI", "LFC", "EL", "MPL"];

// Board palette (fixed hues from index_board.py / colors.py).
const RULE: Color32 = Color32::from_rgb(0x8A, 0x8A, 0x8A);
const CYAN_TXT: Color32 = Color32::from_rgb(0x00, 0xB0, 0xB0);
const RED_TXT: Color32 = Color32::from_rgb(0xFF, 0x40, 0x40);
// Shared white -> yellow -> red -> pink intensity gradient (colors.py).
const GRADIENT_YELLOW: Color32 = Color32::from_rgb(0xFF, 0xFF, 0x00);
const GRADIENT_RED: Color32 = Color32::from_rgb(0xFF, 0x00, 0x00);
const GRADIENT_PINK: Color32 = Color32::from_rgb(0xFF, 0x00, 0xFF);
const GRADIENT_CYAN: Color32 = Color32::from_rgb(0x00, 0xFF, 0xFF);
/// Modernized L1 amber used by severe composites in `[0, 1)`.
const ALERT_L1: Color32 = Color32::from_rgb(0xC8, 0x91, 0x1F);
const GREEN: Color32 = Color32::from_rgb(0x00, 0xFF, 0x00);
const ORANGE: Color32 = Color32::from_rgb(0xFF, 0xA5, 0x00);
const SWEAT_BLUE: Color32 = Color32::from_rgb(0x33, 0x99, 0xFF);
// AGL barb pair: 6 km blue over 1 km red.
const BARB_BLUE: Color32 = Color32::from_rgb(0x0A, 0x74, 0xC6);
const BARB_RED: Color32 = Color32::from_rgb(0xAA, 0x00, 0x00);

/// Draw the legacy three-column index board into `rect`.
///
/// New layouts normally place the logical sections through
/// [`draw_convective`], [`draw_kinematics`], and [`draw_indices`] so each is a
/// first-class movable panel. This combined renderer remains available for
/// restored layouts which explicitly selected the historical `Index board`
/// panel.
pub fn draw(painter: &Painter, rect: Rect, prof: &Profile, dv: &DerivedParams, style: &SkewTStyle) {
    let (w, h) = (rect.width(), rect.height());
    if w <= 6.0 || h <= 6.0 {
        return;
    }
    let p = painter.with_clip_rect(rect);
    p.rect_filled(rect, 0.0, style.bg_color);

    // Row height / fonts from the board height (the Qt original used fixed
    // 13-px regular+bold and 10-px small-bold Helvetica over a ~2-row-slack
    // layout; scale those proportions to the given rect).
    let rh = (h / 18.0).clamp(10.0, 64.0);
    let (rf, hf, hfs) = board_fonts(rh, style);
    let b = Board {
        prof,
        dv,
        st: style,
        rh,
        rf,
        hf,
        hfs,
        patches: None,
    };

    // Column dividers: convective ends at 38% of the width, kinematics at
    // 71.8% (the composite column reaches the board frame).
    let x1 = rect.left() + w * 0.38;
    let x2 = rect.left() + w * 0.718;
    let rule = Stroke::new(1.0, RULE);
    p.line_segment(
        [pos2(x1, rect.top() + 2.0), pos2(x1, rect.bottom() - 2.0)],
        rule,
    );
    p.line_segment(
        [pos2(x2, rect.top() + 2.0), pos2(x2, rect.bottom() - 2.0)],
        rule,
    );

    let (top, bot) = (rect.top() + 2.0, rect.bottom() - 2.0);
    b.col_conv(
        &p,
        Rect::from_min_max(pos2(rect.left() + 4.0, top), pos2(x1 - 4.0, bot)),
    );
    b.col_kin(
        &p,
        Rect::from_min_max(pos2(x1 + 6.0, top), pos2(x2 - 4.0, bot)),
    );
    b.col_comp(
        &p,
        Rect::from_min_max(pos2(x2 + 6.0, top), pos2(rect.right() - 1.0, bot)),
        true,
    );
}

/// Draw the combined legacy board while applying sparse native-cell patches.
pub(crate) fn draw_patched(
    painter: &Painter,
    rect: Rect,
    prof: &Profile,
    dv: &DerivedParams,
    style: &SkewTStyle,
    patches: &NativeDiagnosticPatchBoard,
) {
    let (w, h) = (rect.width(), rect.height());
    if w <= 6.0 || h <= 6.0 {
        return;
    }
    let p = painter.with_clip_rect(rect);
    p.rect_filled(rect, 0.0, style.bg_color);
    let rh = (h / 18.0).clamp(10.0, 64.0);
    let (rf, hf, hfs) = board_fonts(rh, style);
    let b = Board {
        prof,
        dv,
        st: style,
        rh,
        rf,
        hf,
        hfs,
        patches: Some(patches),
    };
    let x1 = rect.left() + w * 0.38;
    let x2 = rect.left() + w * 0.718;
    let rule = Stroke::new(1.0, RULE);
    p.line_segment(
        [pos2(x1, rect.top() + 2.0), pos2(x1, rect.bottom() - 2.0)],
        rule,
    );
    p.line_segment(
        [pos2(x2, rect.top() + 2.0), pos2(x2, rect.bottom() - 2.0)],
        rule,
    );
    let (top, bot) = (rect.top() + 2.0, rect.bottom() - 2.0);
    b.col_conv(
        &p,
        Rect::from_min_max(pos2(rect.left() + 4.0, top), pos2(x1 - 4.0, bot)),
    );
    b.col_kin(
        &p,
        Rect::from_min_max(pos2(x1 + 6.0, top), pos2(x2 - 4.0, bot)),
    );
    b.col_comp(
        &p,
        Rect::from_min_max(pos2(x2 + 6.0, top), pos2(rect.right() - 1.0, bot)),
        true,
    );
}

/// Draw the parcel, thermodynamic, lapse-rate, and severe-composite section
/// as a standalone panel.
pub fn draw_convective(
    painter: &Painter,
    rect: Rect,
    prof: &Profile,
    dv: &DerivedParams,
    style: &SkewTStyle,
) {
    let Some((p, b, content)) = standalone_board(painter, rect, prof, dv, style, 18.0) else {
        return;
    };
    b.col_conv(&p, content);
}

pub(crate) fn draw_convective_patched(
    painter: &Painter,
    rect: Rect,
    prof: &Profile,
    dv: &DerivedParams,
    style: &SkewTStyle,
    patches: &NativeDiagnosticPatchBoard,
) {
    let Some((p, b, content)) =
        standalone_board_patched(painter, rect, prof, dv, style, 18.0, Some(patches))
    else {
        return;
    };
    b.col_conv(&p, content);
}

/// Draw the layer kinematics, storm-motion, and AGL-wind section as a
/// standalone panel.
pub fn draw_kinematics(
    painter: &Painter,
    rect: Rect,
    prof: &Profile,
    dv: &DerivedParams,
    style: &SkewTStyle,
) {
    let Some((p, b, content)) = standalone_board(painter, rect, prof, dv, style, 18.0) else {
        return;
    };
    b.col_kin(&p, content);
}

pub(crate) fn draw_kinematics_patched(
    painter: &Painter,
    rect: Rect,
    prof: &Profile,
    dv: &DerivedParams,
    style: &SkewTStyle,
    patches: &NativeDiagnosticPatchBoard,
) {
    let Some((p, b, content)) =
        standalone_board_patched(painter, rect, prof, dv, style, 18.0, Some(patches))
    else {
        return;
    };
    b.col_kin(&p, content);
}

/// Draw the environmental and severe-weather index readouts without the SHIP
/// distribution chart. SHIP is a separate first-class panel in the split
/// layout.
pub fn draw_indices(
    painter: &Painter,
    rect: Rect,
    prof: &Profile,
    dv: &DerivedParams,
    style: &SkewTStyle,
) {
    let Some((p, b, content)) = standalone_board(painter, rect, prof, dv, style, 9.0) else {
        return;
    };
    b.col_comp(&p, content, false);
}

pub(crate) fn draw_indices_patched(
    painter: &Painter,
    rect: Rect,
    prof: &Profile,
    dv: &DerivedParams,
    style: &SkewTStyle,
    patches: &NativeDiagnosticPatchBoard,
) {
    let Some((p, b, content)) =
        standalone_board_patched(painter, rect, prof, dv, style, 9.0, Some(patches))
    else {
        return;
    };
    b.col_comp(&p, content, false);
}

// ---------------------------------------------------------------------------
// Value formatting (ports of the index_board.py helpers; missing -> "--")
// ---------------------------------------------------------------------------

fn fin(v: f64) -> Option<f64> {
    if qc(v) { Some(v) } else { None }
}

fn i0o(v: Option<f64>) -> String {
    v.map_or_else(|| MISS.to_string(), |x| format!("{}", x.round() as i64))
}

fn f1o(v: Option<f64>) -> String {
    v.map_or_else(|| MISS.to_string(), |x| format!("{x:.1}"))
}

fn f2o(v: Option<f64>) -> String {
    v.map_or_else(|| MISS.to_string(), |x| format!("{x:.2}"))
}

/// Append a unit suffix, but keep the missing placeholder untouched.
fn suf(v: String, s: &str) -> String {
    if v == MISS { v } else { format!("{v}{s}") }
}

/// Temperature readout (the board's default Fahrenheit units).
fn temp_f(v: f64) -> String {
    if qc(v) {
        format!("{}\u{b0}F", v.round() as i64)
    } else {
        MISS.to_string()
    }
}

/// Precipitable water in the default inches units.
fn pwat_in(v: f64) -> String {
    if qc(v) {
        format!("{v:.2} in")
    } else {
        MISS.to_string()
    }
}

/// Magnitude of a `(u, v)` component pair (NaN when unavailable).
fn mag(c: Comp) -> f64 {
    if c.0.is_finite() && c.1.is_finite() {
        c.0.hypot(c.1)
    } else {
        f64::NAN
    }
}

/// `DDD/SS` from a `(wdir, wspd)` vector.
fn dirspd(v: Vect) -> String {
    let (d, s) = v;
    if !qc(d) || !qc(s) {
        return MISS.to_string();
    }
    format!(
        "{:03}/{:02}",
        (d.round() as i64).rem_euclid(360),
        s.round() as i64
    )
}

/// `DDD/SS` from a `(u, v)` component pair.
fn uv_dirspd(c: Comp) -> String {
    let (u, v) = c;
    if !qc(u) || !qc(v) {
        return MISS.to_string();
    }
    let spd = u.hypot(v);
    let d = (270.0 - v.atan2(u).to_degrees()).rem_euclid(360.0);
    format!(
        "{:03}/{:02}",
        (d.round() as i64).rem_euclid(360),
        spd.round() as i64
    )
}

// ---------------------------------------------------------------------------
// Tier color tables (ports of sharpmod/colors.py + the board's local scales)
// ---------------------------------------------------------------------------

/// White/yellow/red/pink gradient used by composite and index values
/// (`colors.common_gradient_color`). `higher=false` is for inverse scales.
/// Zero is intentionally neutral.
fn gradient(
    v: Option<f64>,
    yellow: f64,
    red: f64,
    pink: f64,
    higher: bool,
    fg: Color32,
) -> Color32 {
    let Some(v) = v else { return fg };
    if v == 0.0 {
        return fg;
    }
    if higher {
        if v >= pink {
            GRADIENT_PINK
        } else if v >= red {
            GRADIENT_RED
        } else if v >= yellow {
            GRADIENT_YELLOW
        } else {
            fg
        }
    } else if v <= pink {
        GRADIENT_PINK
    } else if v <= red {
        GRADIENT_RED
    } else if v <= yellow {
        GRADIENT_YELLOW
    } else {
        fg
    }
}

/// Readable amber tier for severe composites in `[0, 1)`.
fn low_severe(v: Option<f64>) -> Option<Color32> {
    match v {
        Some(x) if (0.0..1.0).contains(&x) => Some(ALERT_L1),
        _ => None,
    }
}

/// Supercell Composite Parameter (left-movers read cyan).
fn scp_color(v: Option<f64>, fg: Color32) -> Color32 {
    let Some(x) = v else { return fg };
    if x < 0.0 {
        return GRADIENT_CYAN;
    }
    low_severe(v).unwrap_or_else(|| gradient(v, 0.5, 2.0, 5.0, true, fg))
}

/// Effective-layer STP (cin), symmetric scale.
fn stp_effective_color(v: Option<f64>, fg: Color32) -> Color32 {
    low_severe(v).unwrap_or_else(|| gradient(v, 0.5, 2.0, 5.0, true, fg))
}

/// Fixed-layer Significant Tornado Parameter.
fn stp_fixed_color(v: Option<f64>, fg: Color32) -> Color32 {
    low_severe(v).unwrap_or_else(|| gradient(v, 1.0, 2.0, 5.0, true, fg))
}

/// Significant Hail Parameter.
fn ship_color(v: Option<f64>, fg: Color32) -> Color32 {
    low_severe(v).unwrap_or_else(|| gradient(v, 1.0, 2.0, 3.0, true, fg))
}

/// Derecho Composite Parameter.
fn dcp_color(v: Option<f64>, fg: Color32) -> Color32 {
    low_severe(v).unwrap_or_else(|| gradient(v, 1.0, 4.0, 6.0, true, fg))
}

fn ehi_color(v: Option<f64>, fg: Color32) -> Color32 {
    gradient(v, 1.0, 2.0, 3.0, true, fg)
}

fn peskov_color(v: Option<f64>, fg: Color32) -> Color32 {
    gradient(v, 1.0, 4.0, 7.0, true, fg)
}

/// MCS Index is the Coniglio et al. (2007) regression's linear predictor `a`
/// (sharpmod's number, shown as is). The MCS maintenance probability is
/// `MMP = 1 / (1 + e^a)`, so a LOW index is the dangerous end: coloured by
/// that probability (yellow >= 50 %, red >= 75 %, pink >= 90 %), and neutral
/// when MU CAPE < 100 J/kg, where SHARPpy's MMP is 0 by definition.
fn mcs_color(v: Option<f64>, mucape: Option<f64>, fg: Color32) -> Color32 {
    let (Some(a), Some(cape)) = (v, mucape) else { return fg };
    if cape < 100.0 {
        return fg;
    }
    gradient(Some(1.0 / (1.0 + a.exp())), 0.5, 0.75, 0.9, true, fg)
}

fn lrghail_color(v: Option<f64>, fg: Color32) -> Color32 {
    gradient(v, 4.0, 7.0, 10.0, true, fg)
}

/// SWEAT index: < 250 blue, 250-350 white, 350-500 yellow, 500-650 red,
/// >= 650 pink.
fn sweat_color(v: Option<f64>, fg: Color32) -> Color32 {
    let Some(x) = v else { return fg };
    if x == 0.0 {
        fg
    } else if x < 250.0 {
        SWEAT_BLUE
    } else if x < 350.0 {
        Color32::WHITE
    } else if x < 500.0 {
        GRADIENT_YELLOW
    } else if x < 650.0 {
        GRADIENT_RED
    } else {
        GRADIENT_PINK
    }
}

/// Lapse rate (C/km): green <= 6, yellow <= 7, orange <= 8, red <= 9, else
/// magenta.
fn lapse_rate_color(v: Option<f64>, fg: Color32) -> Color32 {
    let Some(x) = v else { return fg };
    if x == 0.0 {
        fg
    } else if x <= 6.0 {
        GREEN
    } else if x <= 7.0 {
        GRADIENT_YELLOW
    } else if x <= 8.0 {
        ORANGE
    } else if x <= 9.0 {
        GRADIENT_RED
    } else {
        GRADIENT_PINK
    }
}

/// 3CAPE/6CAPE color table: magenta > 125, red > 100, orange > 75,
/// yellow > 50, green > 25, else neutral.
fn cape3_color(v: Option<f64>, fg: Color32) -> Color32 {
    let Some(x) = v else { return fg };
    if x > 125.0 {
        GRADIENT_PINK
    } else if x > 100.0 {
        GRADIENT_RED
    } else if x > 75.0 {
        ORANGE
    } else if x > 50.0 {
        GRADIENT_YELLOW
    } else if x > 25.0 {
        GREEN
    } else {
        fg
    }
}

/// Legacy SHARPpy parcel CINH coloring: >= -50 green, -100..-50 orange,
/// < -100 red.
fn cinh_legacy(v: Option<f64>, fg: Color32) -> Color32 {
    let Some(x) = v else { return fg };
    if x >= -50.0 {
        GREEN
    } else if x >= -100.0 {
        ORANGE
    } else {
        GRADIENT_RED
    }
}

// ---------------------------------------------------------------------------
// Value + smaller-unit text (port of sharpmod/viz/unit_text.py)
// ---------------------------------------------------------------------------

/// Recognized trailing unit suffixes, longest first.
const UNIT_SUFFIXES: [&str; 14] = [
    " degrees C/km",
    " degrees C",
    " m\u{b3}/s\u{b3}",
    " J/kg/m",
    " m2/s2",
    " m AGL",
    " C/km",
    " g/kg",
    " J/kg",
    " m/s",
    " kt",
    " cm",
    " in",
    " m",
];
const DEGREE_SUFFIXES: [&str; 2] = ["\u{b0}F", "\u{b0}C"];
const UNIT_FONT_SCALE: f32 = 0.78;

/// Split a sounding value from a recognized trailing unit suffix.
fn split_value_unit(text: &str) -> Option<(&str, &str)> {
    for sfx in UNIT_SUFFIXES.iter().chain(DEGREE_SUFFIXES.iter()) {
        if let Some(value) = text.strip_suffix(sfx)
            && !value.trim().is_empty()
        {
            return Some((value, sfx));
        }
    }
    None
}

/// A legible smaller variant of `font` for a value's unit.
fn small_unit_font(font: &FontId) -> FontId {
    FontId::new(
        (font.size * UNIT_FONT_SCALE).round().max(8.0),
        font.family.clone(),
    )
}

/// Horizontal alignment for a table cell (vertical is always centered).
#[derive(Clone, Copy, PartialEq)]
enum HA {
    Left,
    Center,
}

struct NativeRowDisplay {
    label: String,
    value: String,
    color: Color32,
}

// ---------------------------------------------------------------------------
// The board
// ---------------------------------------------------------------------------

struct Board<'a> {
    prof: &'a Profile,
    dv: &'a DerivedParams,
    st: &'a SkewTStyle,
    rh: f32,
    /// Regular row font.
    rf: FontId,
    /// Bold header font (same size as `rf`).
    hf: FontId,
    /// Smaller bold font for tight column headers / the barb label.
    hfs: FontId,
    /// Sparse edits to native cells. `None` is the exact legacy renderer.
    patches: Option<&'a NativeDiagnosticPatchBoard>,
}

fn board_fonts(rh: f32, style: &SkewTStyle) -> (FontId, FontId, FontId) {
    (
        style.regular_font(rh * 0.78),
        style.bold_font(rh * 0.78),
        style.bold_font((rh * 0.60).max(8.0)),
    )
}

fn standalone_board<'a>(
    painter: &Painter,
    rect: Rect,
    prof: &'a Profile,
    dv: &'a DerivedParams,
    style: &'a SkewTStyle,
    nominal_rows: f32,
) -> Option<(Painter, Board<'a>, Rect)> {
    standalone_board_patched(painter, rect, prof, dv, style, nominal_rows, None)
}

fn standalone_board_patched<'a>(
    painter: &Painter,
    rect: Rect,
    prof: &'a Profile,
    dv: &'a DerivedParams,
    style: &'a SkewTStyle,
    nominal_rows: f32,
    patches: Option<&'a NativeDiagnosticPatchBoard>,
) -> Option<(Painter, Board<'a>, Rect)> {
    if rect.width() <= 6.0 || rect.height() <= 6.0 {
        return None;
    }
    let p = painter.with_clip_rect(rect);
    p.rect_filled(rect, 0.0, style.bg_color);
    p.rect_stroke(rect, 0.0, Stroke::new(1.0, RULE), StrokeKind::Inside);
    let content = rect.shrink2(vec2(4.0, 2.0));
    let rh = (content.height() / nominal_rows).clamp(10.0, 64.0);
    let (rf, hf, hfs) = board_fonts(rh, style);
    let board = Board {
        prof,
        dv,
        st: style,
        rh,
        rf,
        hf,
        hfs,
        patches,
    };
    Some((p, board, content))
}

impl Board<'_> {
    fn width(&self, p: &Painter, font: &FontId, s: &str) -> f32 {
        p.layout_no_wrap(s.to_owned(), font.clone(), Color32::WHITE)
            .size()
            .x
    }

    /// Width of `text` with any recognized unit compacted.
    fn vuw(&self, p: &Painter, font: &FontId, text: &str) -> f32 {
        match split_value_unit(text) {
            Some((v, u)) => self.width(p, font, v) + self.width(p, &small_unit_font(font), u),
            None => self.width(p, font, text),
        }
    }

    /// Draw a cell, rendering a recognized unit suffix smaller than its
    /// value when it fits (falls back to a plain single-line draw).
    fn text(&self, p: &Painter, r: Rect, s: &str, font: &FontId, color: Color32, ha: HA) {
        if let Some((val, unit)) = split_value_unit(s) {
            let uf = small_unit_font(font);
            let vw = self.width(p, font, val);
            let uw = self.width(p, &uf, unit);
            let gw = vw + uw;
            if gw <= r.width() {
                let left = match ha {
                    HA::Left => r.left(),
                    HA::Center => r.left() + (r.width() - gw) / 2.0,
                };
                let cy = r.center().y;
                p.text(
                    pos2(left, cy),
                    Align2::LEFT_CENTER,
                    val,
                    font.clone(),
                    color,
                );
                p.text(pos2(left + vw, cy), Align2::LEFT_CENTER, unit, uf, color);
                return;
            }
        }
        let (pos, anchor) = match ha {
            HA::Left => (pos2(r.left(), r.center().y), Align2::LEFT_CENTER),
            HA::Center => (pos2(r.center().x, r.center().y), Align2::CENTER_CENTER),
        };
        p.text(pos, anchor, s, font.clone(), color);
    }

    fn cell(&self, x: f32, y: f32, w: f32) -> Rect {
        Rect::from_min_size(pos2(x, y), vec2(w, self.rh))
    }

    /// `"label = value"` row with the value left-aligned right after the
    /// label (composite-column layout).
    #[allow(clippy::too_many_arguments)]
    fn row_at(&self, p: &Painter, cx: f32, cw: f32, cy: f32, lbl: &str, val: &str, color: Color32) {
        let ltext = format!("{lbl} = ");
        self.text(p, self.cell(cx, cy, cw), &ltext, &self.rf, color, HA::Left);
        let lw = self.width(p, &self.rf, &ltext);
        self.text(
            p,
            self.cell(cx + lw, cy, cw - lw - 2.0),
            val,
            &self.rf,
            color,
            HA::Left,
        );
    }

    fn patch(
        &self,
        panel: DiagnosticTablePanelKind,
        slot_id: &str,
    ) -> Option<&NativeDiagnosticSlotPatch> {
        self.patches
            .and_then(|patches| patches.patch(panel, slot_id))
    }

    fn inline_display(
        &self,
        panel: DiagnosticTablePanelKind,
        slot_id: &str,
        canonical_label: &str,
        canonical_value: &str,
        canonical_color: Color32,
    ) -> Option<NativeRowDisplay> {
        match self.patch(panel, slot_id) {
            None => Some(NativeRowDisplay {
                label: canonical_label.to_owned(),
                value: canonical_value.to_owned(),
                color: canonical_color,
            }),
            Some(NativeDiagnosticSlotPatch::Blank) => None,
            Some(NativeDiagnosticSlotPatch::Replace(row)) => Some(NativeRowDisplay {
                label: if row.label.trim().is_empty() {
                    canonical_label.to_owned()
                } else {
                    row.label.trim().to_owned()
                },
                value: Self::replacement_value(row),
                color: row.color.unwrap_or(canonical_color),
            }),
        }
    }

    fn replacement_value(row: &DiagnosticTableRow) -> String {
        if row.value == MISS || row.value.is_empty() || row.unit.trim().is_empty() {
            row.value.clone()
        } else {
            format!("{} {}", row.value, row.unit.trim())
        }
    }

    /// Draw a native inline `label = value` slot, changing only its contents.
    #[allow(clippy::too_many_arguments)]
    fn inline_slot(
        &self,
        p: &Painter,
        panel: DiagnosticTablePanelKind,
        slot_id: &str,
        cx: f32,
        cw: f32,
        cy: f32,
        canonical_label: &str,
        canonical_value: &str,
        canonical_color: Color32,
    ) {
        match self.patch(panel, slot_id) {
            None => self.row_at(
                p,
                cx,
                cw,
                cy,
                canonical_label,
                canonical_value,
                canonical_color,
            ),
            Some(NativeDiagnosticSlotPatch::Blank) => {}
            Some(NativeDiagnosticSlotPatch::Replace(row)) => {
                let label = if row.label.trim().is_empty() {
                    canonical_label
                } else {
                    row.label.trim()
                };
                let value = Self::replacement_value(row);
                self.row_at(
                    p,
                    cx,
                    cw,
                    cy,
                    label,
                    &value,
                    row.color.unwrap_or(canonical_color),
                );
            }
        }
    }

    fn fitted_cell_font(&self, p: &Painter, text: &str, width: f32) -> FontId {
        let mut font = self.rf.clone();
        let min_size = (self.rf.size * 0.54).max(6.0);
        while font.size > min_size && self.vuw(p, &font, text) > width {
            font.size = (font.size - 0.5).max(min_size);
        }
        font
    }

    /// Draw a value in one native matrix cell. A replacement that changes
    /// the cell's meaning includes its explicit label, fitted inside the same
    /// geometry; a same-semantic replacement remains value-only.
    #[allow(clippy::too_many_arguments)]
    fn matrix_slot(
        &self,
        p: &Painter,
        panel: DiagnosticTablePanelKind,
        slot_id: &str,
        rect: Rect,
        canonical_label: &str,
        canonical_value: &str,
        canonical_color: Color32,
    ) {
        match self.patch(panel, slot_id) {
            None => self.text(
                p,
                rect,
                canonical_value,
                &self.rf,
                canonical_color,
                HA::Center,
            ),
            Some(NativeDiagnosticSlotPatch::Blank) => {}
            Some(NativeDiagnosticSlotPatch::Replace(row)) => {
                let value = Self::replacement_value(row);
                let label = row.label.trim();
                let text = if label.is_empty() || label.eq_ignore_ascii_case(canonical_label) {
                    value
                } else if value.is_empty() {
                    label.to_owned()
                } else {
                    format!("{label} {value}")
                };
                let font = self.fitted_cell_font(p, &text, rect.width());
                self.text(
                    p,
                    rect,
                    &text,
                    &font,
                    row.color.unwrap_or(canonical_color),
                    HA::Center,
                );
            }
        }
    }

    // ---- column 1: convective -----------------------------------------
    fn col_conv(&self, p: &Painter, r: Rect) {
        let fg = self.st.fg_color;
        let rh = self.rh;
        let (x, w) = (r.left(), r.width());
        let mut y = r.top();
        let cw = w / 8.0;
        for (i, c) in PARCEL_MATRIX_HEADERS.iter().enumerate() {
            self.text(
                p,
                self.cell(x + i as f32 * cw, y, cw),
                c,
                &self.hf,
                fg,
                HA::Center,
            );
        }
        y += rh + 1.0;
        // Distribute leftover vertical space across the two section dividers.
        // Content below the header = 4 parcel + 6 stats + 5 lapse = 15 rows.
        let per_div = ((r.height() - 16.0 * rh - 1.0) / 2.0).max(6.0);

        let parcels: [(&str, &str, &ParcelResult); 4] = [
            ("SFC", "sfc", &self.prof.sfcpcl),
            ("ML", "ml", &self.prof.mlpcl),
            ("FCST", "fcst", &self.prof.fcstpcl),
            ("MU", "mu", &self.prof.mupcl),
        ];
        for (name, parcel_id, pcl) in parcels {
            let cape = fin(pcl.bplus);
            let has_cape = cape.is_some_and(|c| c > 0.0);
            // CAPE/CINH/LI escalate only when the parcel has positive CAPE;
            // LCL, LFC, EL, MPL and the parcel name stay neutral.
            let (cape_c, cinh_c, li_c) = if has_cape {
                (
                    gradient(cape, 1000.0, 2500.0, 4000.0, true, fg),
                    cinh_legacy(fin(pcl.bminus), fg),
                    gradient(fin(pcl.li5), -4.0, -7.0, -10.0, false, fg),
                )
            } else {
                (fg, fg, fg)
            };
            self.text(p, self.cell(x, y, cw), name, &self.rf, fg, HA::Center);
            let cells: [(&str, &str, String, Color32); 7] = [
                ("cape", "CAPE", int2str(pcl.bplus), cape_c),
                ("cinh", "CINH", int2str(pcl.bminus), cinh_c),
                ("lcl", "LCL", int2str(pcl.lclhght), fg),
                ("li", "LI", int2str(pcl.li5), li_c),
                ("lfc", "LFC", int2str(pcl.lfchght), fg),
                ("el", "EL", int2str(pcl.elhght), fg),
                ("mpl", "MPL", int2str(pcl.mplhght), fg),
            ];
            for (i, (field, heading, value, color)) in cells.iter().enumerate() {
                let slot_id = format!("parcel.{parcel_id}.{field}");
                let canonical_label = format!("{name} {heading}");
                self.matrix_slot(
                    p,
                    DiagnosticTablePanelKind::Convective,
                    &slot_id,
                    self.cell(x + (i + 1) as f32 * cw, y, cw),
                    &canonical_label,
                    value,
                    *color,
                );
            }
            y += rh;
        }
        y += per_div / 2.0;
        p.line_segment([pos2(x, y), pos2(x + w, y)], Stroke::new(1.0, RULE));
        y += per_div / 2.0;

        // Thermo stats block, three sub-columns of six rows.
        let dv = self.dv;
        let col1: [(&str, &str, String, Color32); 6] = [
            ("thermo.pwat", "PWAT", pwat_in(dv.pwat), fg),
            (
                "thermo.mean_mixr",
                "MeanW",
                suf(float2str(dv.mean_mixr, 2), " g/kg"),
                fg,
            ),
            ("thermo.low_rh", "LowRH", suf(int2str(dv.low_rh), "%"), fg),
            ("thermo.mid_rh", "MidRH", suf(int2str(dv.mid_rh), "%"), fg),
            ("thermo.dcape", "DCAPE", int2str(dv.dcape), fg),
            ("thermo.downrush_temp", "DownT", temp_f(dv.drush_f), fg),
        ];
        let col2: [(&str, &str, String, Color32); 6] = [
            ("thermo.k_index", "K", int2str(dv.k_idx), fg),
            ("thermo.total_totals", "TT", int2str(dv.totals_totals), fg),
            ("thermo.convective_temp", "ConvT", temp_f(dv.conv_t_f), fg),
            ("thermo.max_temp", "MaxT", temp_f(dv.max_t_f), fg),
            ("thermo.esp", "ESP", float2str(dv.esp, 1), fg),
            ("thermo.mmp", "MMP", float2str(dv.mmp, 2), fg),
        ];
        let col3: [(&str, &str, String, Color32); 6] = [
            ("thermo.wndg", "WNDG", float2str(dv.wndg, 1), fg),
            ("thermo.tei", "TEI", int2str(dv.tei), fg),
            (
                "thermo.cape_0_3km",
                "3CAPE",
                int2str(dv.cape_0_3km),
                cape3_color(fin(dv.cape_0_3km), fg),
            ),
            (
                "thermo.cape_0_6km",
                "6CAPE",
                int2str(dv.cape_0_6km),
                cape3_color(fin(dv.cape_0_6km), fg),
            ),
            ("thermo.mburst", "MBURST", int2str(dv.mburst), fg),
            (
                "thermo.sig_severe",
                "SigSvr",
                suf(int2str(dv.sig_severe), " m\u{b3}/s\u{b3}"),
                fg,
            ),
        ];
        let stat_cols = [&col1, &col2, &col3];
        let displayed_cols = stat_cols
            .iter()
            .map(|col| {
                col.iter()
                    .map(|(id, label, value, color)| {
                        self.inline_display(
                            DiagnosticTablePanelKind::Convective,
                            id,
                            label,
                            value,
                            *color,
                        )
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let mut gutter = 4.0f32;
        let mut min_widths = [0.0f32; 3];
        for (ci, col) in displayed_cols.iter().enumerate() {
            for display in col.iter().flatten() {
                let mw = self.width(p, &self.rf, &format!("{} = ", display.label))
                    + self.vuw(p, &self.rf, &display.value)
                    + 2.0;
                min_widths[ci] = min_widths[ci].max(mw);
            }
        }
        let min_total: f32 = min_widths.iter().sum::<f32>() + gutter * 2.0;
        let stat_widths = if min_total <= w {
            let extra = (w - min_total) / 3.0;
            [
                min_widths[0] + extra,
                min_widths[1] + extra,
                min_widths[2] + extra,
            ]
        } else {
            gutter = 0.0;
            [w / 3.0; 3]
        };
        let mut stat_xs = [0.0f32; 3];
        let mut cursor_x = x;
        for i in 0..3 {
            stat_xs[i] = cursor_x;
            cursor_x += stat_widths[i] + gutter;
        }
        for (ci, col) in displayed_cols.iter().enumerate() {
            let cx = stat_xs[ci];
            let col_width = stat_widths[ci];
            let val_right = cx + col_width;
            for (ri, display) in col.iter().enumerate() {
                let Some(display) = display else { continue };
                let ry = y + ri as f32 * rh;
                // "label = " left-aligned, then the value right after it so
                // long labels are never clipped on the left.
                let ltext = format!("{} = ", display.label);
                self.text(
                    p,
                    self.cell(cx, ry, col_width),
                    &ltext,
                    &self.rf,
                    display.color,
                    HA::Left,
                );
                let lw = self.width(p, &self.rf, &ltext);
                let vw = (val_right - (cx + lw) - 2.0).max(0.0);
                // Shrink the value font just enough to fit its slot so unit
                // suffixes are never clipped in the narrow sub-columns.
                let mut vfont = self.rf.clone();
                if vw > 0.0 && self.vuw(p, &vfont, &display.value) > vw {
                    let min_px = (self.rf.size * (9.0 / 13.0)).max(8.0);
                    let mut px = self.rf.size;
                    while px > min_px {
                        px -= 1.0;
                        vfont = FontId::new(px, self.rf.family.clone());
                        if self.vuw(p, &vfont, &display.value) <= vw {
                            break;
                        }
                    }
                }
                self.text(
                    p,
                    self.cell(cx + lw, ry, vw),
                    &display.value,
                    &vfont,
                    display.color,
                    HA::Left,
                );
            }
        }
        y += 6.0 * rh;
        y += per_div / 2.0;
        p.line_segment([pos2(x, y), pos2(x + w, y)], Stroke::new(1.0, RULE));
        y += per_div / 2.0;

        // Lapse-rate box (left) beside the Severe Weather Composite (right),
        // each value colored by its own threshold tier.
        let lapse: [(&str, &str, f64); 5] = [
            ("lapse.sfc_500m", "SFC-500m LR", dv.lapserate_sfc_500m),
            ("lapse.sfc_1km", "SFC-1km LR", dv.lapserate_sfc_1km),
            ("lapse.sfc_3km", "SFC-3km LR", dv.lapserate_3km),
            ("lapse.850_500", "850-500 LR", dv.lapserate_850_500),
            ("lapse.700_500", "700-500 LR", dv.lapserate_700_500),
        ];
        let severe: [(&str, &str, String, Color32); 5] = [
            (
                "composite.scp_right",
                "Supercell Comp",
                float2str(dv.right_scp, 1),
                scp_color(fin(dv.right_scp), fg),
            ),
            (
                "composite.stp_effective",
                "STP(cin)",
                float2str(dv.stp_cin, 1),
                stp_effective_color(fin(dv.stp_cin), fg),
            ),
            (
                "composite.stp_fixed",
                "STP(fix)",
                float2str(dv.stp_fixed, 1),
                stp_fixed_color(fin(dv.stp_fixed), fg),
            ),
            (
                "composite.ship",
                "SHIP",
                float2str(dv.ship, 1),
                ship_color(fin(dv.ship), fg),
            ),
            (
                "composite.dcp",
                "Derecho Comp",
                float2str(dv.dcp, 1),
                dcp_color(fin(dv.dcp), fg),
            ),
        ];
        let lwid = w * 0.49;
        let bx = x + w * 0.50; // severe box left edge
        let sx = bx + 7.0; // severe text, small inset off the separator
        let swid = x + w - sx - 2.0;
        y -= 3.0; // nudge the paired block up a touch
        let sec_top = y;
        let section_bottom = r.bottom();

        // Shared baseline sequence so paired entries stay aligned.
        let usable = (section_bottom - sec_top - rh).max(0.0);
        let compact_step = rh.max((rh * 1.12).round());
        let row_step = compact_step.min(usable / 4.0);
        let block_h = 4.0 * row_step + rh;
        let avail_h = (section_bottom - sec_top).max(rh);
        let mut start = sec_top + ((avail_h - block_h) / 2.0).max(0.0);
        let max_start = section_bottom - block_h;
        if max_start >= sec_top {
            start = start.min(max_start);
        }

        for (i, (slot_id, llbl, lval)) in lapse.iter().enumerate() {
            let row_y = start + i as f32 * row_step;
            let c = lapse_rate_color(fin(*lval), fg);
            let lvt = if fin(*lval).is_some() {
                format!("{} C/km", float2str(*lval, 1))
            } else {
                MISS.to_string()
            };
            self.inline_slot(
                p,
                DiagnosticTablePanelKind::Convective,
                slot_id,
                x,
                lwid,
                row_y,
                llbl,
                &lvt,
                c,
            );
        }
        for (i, (slot_id, slbl, sval, sc)) in severe.iter().enumerate() {
            let row_y = start + i as f32 * row_step;
            self.inline_slot(
                p,
                DiagnosticTablePanelKind::Convective,
                slot_id,
                sx,
                swid,
                row_y,
                slbl,
                sval,
                *sc,
            );
        }
        // Separator between the lapse rates and the severe composite.
        let line_top = sec_top + (rh / 4.0).max(4.0);
        let line_bottom = r.bottom() - 2.0;
        if line_bottom > line_top {
            p.line_segment(
                [pos2(bx, line_top), pos2(bx, line_bottom)],
                Stroke::new(1.0, RULE),
            );
        }
    }

    // ---- column 2: kinematics -----------------------------------------
    fn col_kin(&self, p: &Painter, r: Rect) {
        let dv = self.dv;
        let fg = self.st.fg_color;
        let rh = self.rh;
        let (x, w) = (r.left(), r.width());
        let mut y = r.top();
        // SRH/Shear are compact readouts; MnWind/SRW carry DDD/SS vectors and
        // get wider tracks.
        let lw = w * 0.28;
        let srh_w = w * 0.13;
        let shear_w = w * 0.13;
        let vector_w = ((w - lw - srh_w - shear_w) / 2.0).max(1.0);
        let srw_w = (w - lw - srh_w - shear_w - vector_w).max(1.0);
        // Pull only the SRH readout toward the layer labels.
        let srh_left_shift = (w * 0.04).max(4.0);
        let value_xs = [
            x + lw - srh_left_shift,
            x + lw + srh_w,
            x + lw + srh_w + shear_w,
            x + lw + srh_w + shear_w + vector_w,
        ];
        let value_ws = [srh_w, shear_w, vector_w, srw_w];
        // Two-line headers: short label on top, unit on a small second line.
        let units = ["m2/s2", "kt", "\u{b0}/kt", "\u{b0}/kt"];
        for (i, hh) in ["SRH", "Shear", "MnWind", "SRW"].iter().enumerate() {
            self.text(
                p,
                self.cell(value_xs[i], y, value_ws[i]),
                hh,
                &self.hfs,
                fg,
                HA::Center,
            );
            self.text(
                p,
                self.cell(value_xs[i], y + rh * 0.74, value_ws[i]),
                &format!("({})", units[i]),
                &self.hfs,
                RULE,
                HA::Center,
            );
        }
        y += 2.0 * rh - 4.0;

        let rows: [(&str, String, String, String, String); 8] = [
            (
                "SFC-500m",
                int2str(dv.srh500),
                int2str(mag(dv.sfc_500m_shear)),
                uv_dirspd(dv.mean_wind_sfc_500m),
                uv_dirspd(dv.srw_sfc_500m),
            ),
            (
                "SFC-1km",
                int2str(dv.srh1km),
                int2str(mag(dv.sfc_1km_shear)),
                dirspd(dv.mean_1km),
                dirspd(dv.srw_1km),
            ),
            (
                "SFC-3km",
                int2str(dv.srh3km),
                int2str(mag(dv.sfc_3km_shear)),
                dirspd(dv.mean_3km),
                dirspd(dv.srw_3km),
            ),
            (
                "Eff Inflow",
                int2str(dv.right_esrh),
                int2str(mag(dv.eff_shear)),
                uv_dirspd(dv.mean_eff),
                uv_dirspd(dv.srw_eff),
            ),
            (
                "SFC-6km",
                MISS.to_string(),
                int2str(mag(dv.sfc_6km_shear)),
                dirspd(dv.mean_6km),
                dirspd(dv.srw_6km),
            ),
            (
                "SFC-8km",
                MISS.to_string(),
                int2str(mag(dv.sfc_8km_shear)),
                dirspd(dv.mean_8km),
                dirspd(dv.srw_8km),
            ),
            (
                "LCL-EL",
                MISS.to_string(),
                int2str(mag(dv.lcl_el_shear)),
                dirspd(dv.mean_lcl_el),
                dirspd(dv.srw_lcl_el),
            ),
            (
                "Eff Shear",
                MISS.to_string(),
                int2str(mag(dv.ebwd)),
                uv_dirspd(dv.mean_ebw),
                uv_dirspd(dv.srw_ebw),
            ),
        ];
        let slot_ids: [[Option<&str>; 4]; 8] = [
            [
                Some("kin.sfc_500m.srh"),
                Some("kin.sfc_500m.shear"),
                Some("kin.sfc_500m.mean_wind"),
                Some("kin.sfc_500m.srw"),
            ],
            [
                Some("kin.sfc_1km.srh"),
                Some("kin.sfc_1km.shear"),
                Some("kin.sfc_1km.mean_wind"),
                Some("kin.sfc_1km.srw"),
            ],
            [
                Some("kin.sfc_3km.srh"),
                Some("kin.sfc_3km.shear"),
                Some("kin.sfc_3km.mean_wind"),
                Some("kin.sfc_3km.srw"),
            ],
            [
                Some("kin.effective.srh"),
                Some("kin.effective.shear"),
                Some("kin.effective.mean_wind"),
                Some("kin.effective.srw"),
            ],
            [
                None,
                Some("kin.sfc_6km.shear"),
                Some("kin.sfc_6km.mean_wind"),
                Some("kin.sfc_6km.srw"),
            ],
            [
                None,
                Some("kin.sfc_8km.shear"),
                Some("kin.sfc_8km.mean_wind"),
                Some("kin.sfc_8km.srw"),
            ],
            [
                None,
                Some("kin.lcl_el.shear"),
                Some("kin.lcl_el.mean_wind"),
                Some("kin.lcl_el.srw"),
            ],
            [
                None,
                Some("kin.ebwd.shear"),
                Some("kin.ebwd.mean_wind"),
                Some("kin.ebwd.srw"),
            ],
        ];
        let canonical_labels: [[Option<&str>; 4]; 8] = [
            [
                Some("SFC-500m SRH"),
                Some("SFC-500m Shear"),
                Some("SFC-500m Mean Wind"),
                Some("SFC-500m SR Wind"),
            ],
            [
                Some("SFC-1km SRH"),
                Some("SFC-1km Shear"),
                Some("SFC-1km Mean Wind"),
                Some("SFC-1km SR Wind"),
            ],
            [
                Some("SFC-3km SRH"),
                Some("SFC-3km Shear"),
                Some("SFC-3km Mean Wind"),
                Some("SFC-3km SR Wind"),
            ],
            [
                Some("Effective SRH"),
                Some("Effective Inflow Shear"),
                Some("Effective Mean Wind"),
                Some("Effective SR Wind"),
            ],
            [
                None,
                Some("SFC-6km Shear"),
                Some("SFC-6km Mean Wind"),
                Some("SFC-6km SR Wind"),
            ],
            [
                None,
                Some("SFC-8km Shear"),
                Some("SFC-8km Mean Wind"),
                Some("SFC-8km SR Wind"),
            ],
            [
                None,
                Some("LCL-EL Shear"),
                Some("LCL-EL Mean Wind"),
                Some("LCL-EL SR Wind"),
            ],
            [
                None,
                Some("Effective Bulk Shear"),
                Some("Effective Bulk Mean Wind"),
                Some("Effective Bulk SR Wind"),
            ],
        ];
        for (row_index, (lbl, srh, shr, mnw, srw)) in rows.into_iter().enumerate() {
            self.text(p, self.cell(x, y, lw), lbl, &self.rf, fg, HA::Left);
            for (i, v) in [&srh, &shr, &mnw, &srw].into_iter().enumerate() {
                let Some(slot_id) = slot_ids[row_index][i] else {
                    continue;
                };
                let canonical_label = canonical_labels[row_index][i].expect("native slot label");
                if v == MISS
                    && self
                        .patch(DiagnosticTablePanelKind::Kinematics, slot_id)
                        .is_none()
                {
                    continue; // preserve native empty unavailable cells
                }
                self.matrix_slot(
                    p,
                    DiagnosticTablePanelKind::Kinematics,
                    slot_id,
                    self.cell(value_xs[i], y, value_ws[i]),
                    canonical_label,
                    v,
                    fg,
                );
            }
            y += rh;
        }
        // Distribute the leftover vertical space across the two gaps below:
        // BRN(2) + storm header(1) + storm(4) rows remain.
        let kg = ((r.bottom() - y - (7.0 * rh + 4.0)) / 2.0).max(4.0);
        y += 4.0;
        p.line_segment([pos2(x, y), pos2(x + w, y)], Stroke::new(1.0, RULE));
        y += kg;

        // The AGL wind barbs anchor at the top of the right-hand whitespace
        // beside the BRN/SR-wind + storm-motion rows.
        let barb_top = y;
        for (slot_id, lbl, val, unit) in [
            ("kin.brn_shear", "BRN Shear", int2str(dv.brnshear), " m2/s2"),
            (
                "kin.srw_4_6km",
                "4-6km SR Wind",
                dirspd(dv.srw_4_5km),
                " kt",
            ),
        ] {
            let vt = if val != MISS {
                format!("{val}{unit}")
            } else {
                MISS.to_string()
            };
            self.inline_slot(
                p,
                DiagnosticTablePanelKind::Kinematics,
                slot_id,
                x,
                w,
                y,
                lbl,
                &vt,
                fg,
            );
            y += rh;
        }
        y += kg;

        let (ru, rv, blu, blv) = self.prof.srwind;
        let br = uv_dirspd((ru, rv));
        let bl = uv_dirspd((blu, blv));
        let cor_up = uv_dirspd(dv.corfidi_up);
        let cor_dn = uv_dirspd(dv.corfidi_dn);
        // Reserve a right-hand region for the 1 km / 6 km AGL wind barbs +
        // label; the storm-motion vectors take the rest.
        let barb_region = self.width(p, &self.hfs, "1km & 6km AGL") * 1.75 + 28.0;
        let text_w = (w * 0.54).max(w - barb_region);
        self.text(
            p,
            self.cell(x, y, text_w),
            "...Storm Motion Vectors...",
            &self.rf,
            fg,
            HA::Left,
        );
        y += rh;
        // Bunkers Right (cyan) / Left (red) follow legacy SHARPpy; Corfidi
        // vectors stay neutral. Labels stay white; only the value is colored.
        for (slot_id, lbl, val, vcol) in [
            ("kin.bunkers_right", "Bunkers Right", br, CYAN_TXT),
            ("kin.bunkers_left", "Bunkers Left", bl, RED_TXT),
            ("kin.corfidi_down", "Corfidi Dshr", cor_dn, fg),
            ("kin.corfidi_up", "Corfidi Ushr", cor_up, fg),
        ] {
            let vt = if val != MISS {
                format!("{val} kt")
            } else {
                MISS.to_string()
            };
            match self.patch(DiagnosticTablePanelKind::Kinematics, slot_id) {
                None => {
                    let lt = format!("{lbl} = ");
                    self.text(p, self.cell(x, y, text_w), &lt, &self.rf, fg, HA::Left);
                    let lw3 = self.width(p, &self.rf, &lt);
                    self.text(
                        p,
                        self.cell(x + lw3, y, text_w - lw3 - 2.0),
                        &vt,
                        &self.rf,
                        vcol,
                        HA::Left,
                    );
                }
                Some(NativeDiagnosticSlotPatch::Blank) => {}
                Some(NativeDiagnosticSlotPatch::Replace(row)) => {
                    let label = if row.label.trim().is_empty() {
                        lbl
                    } else {
                        row.label.trim()
                    };
                    let value = Self::replacement_value(row);
                    self.row_at(p, x, text_w, y, label, &value, row.color.unwrap_or(vcol));
                }
            }
            y += rh;
        }
        let agl_h = (rh * 5.0).max(r.bottom() - barb_top);
        self.draw_agl_barbs(p, x + text_w, barb_top, w - text_w, agl_h);
    }

    /// 1 km (red) & 6 km (blue) AGL wind barbs from a common origin, the
    /// combined bounding box centered in `[rx, rx+rw]` above a two-line
    /// label (port of `_draw_agl_barbs`).
    fn draw_agl_barbs(&self, p: &Painter, rx: f32, top: f32, rw: f32, h: f32) {
        let (d1, s1) = self.dv.wind1km;
        let (d6, s6) = self.dv.wind6km;
        let patch1 = self.patch(DiagnosticTablePanelKind::Kinematics, "kin.wind_1km");
        let patch6 = self.patch(DiagnosticTablePanelKind::Kinematics, "kin.wind_6km");
        let patched = patch1.is_some() || patch6.is_some();
        let have1 = qc(d1) && qc(s1) && patch1.is_none();
        let have6 = qc(d6) && qc(s6) && patch6.is_none();
        let replacement_visible = matches!(patch1, Some(NativeDiagnosticSlotPatch::Replace(_)))
            || matches!(patch6, Some(NativeDiagnosticSlotPatch::Replace(_)));
        if !have1 && !have6 && !replacement_visible {
            return;
        }
        let shemis = self.prof.latitude() < 0.0;
        // Scale the barbs to fill the reserved region. The Qt original sized
        // a ~35-px barb span against its 13-px fonts (row height 19); `s0`
        // carries that reference to the current board scale.
        let s0 = self.rh / 19.0;
        let barb_span = 35.0;
        let avail = (rw - 12.0 * s0).max(1.0).min((h * 0.54).max(1.0));
        let scale = (avail / barb_span).clamp(0.95 * s0, 1.25 * s0);
        // Enlarge the label proportionally so it stays balanced.
        let rel = (scale / s0).min(1.48);
        let base = self.hfs.size;
        let lbl_font = FontId::new(
            (base + 1.0).max((base * rel).round()),
            self.hfs.family.clone(),
        );
        let line_h = p
            .layout_no_wrap("1km & 6km AGL".to_owned(), lbl_font.clone(), BARB_BLUE)
            .size()
            .y;
        let label_h = 2.0 * line_h + 2.0;
        let label_top = top + (h - label_h - 2.0).max(0.0);
        let barb_bottom = (label_top - 4.0).max(top + 1.0);

        // Build both barbs (shared origin) and center their combined bounds.
        let mut barbs: Vec<(BarbPath, Color32)> = Vec::new();
        if have6 {
            barbs.push((build_barb(d6, s6, shemis, scale), BARB_BLUE)); // 6 km : blue
        }
        if have1 {
            barbs.push((build_barb(d1, s1, shemis, scale), BARB_RED)); // 1 km : red
        }
        let center_x = rx + rw * 0.48;
        if !barbs.is_empty() {
            let mut bmin = barbs[0].0.min;
            let mut bmax = barbs[0].0.max;
            for (b, _) in &barbs {
                bmin = bmin.min(b.min);
                bmax = bmax.max(b.max);
            }
            let cy = top + (barb_bottom - top) * 0.5;
            let d = vec2(
                center_x - (bmin.x + bmax.x) / 2.0,
                cy - (bmin.y + bmax.y) / 2.0,
            );
            let sw = 1.4 * scale.sqrt().max(1.0);
            for (b, color) in &barbs {
                let stroke = Stroke::new(sw, *color);
                if let Some(radius) = b.circle {
                    p.circle_stroke(pos2(d.x, d.y), radius, stroke);
                }
                for (a0, a1) in &b.segs {
                    p.line_segment(
                        [pos2(a0.x + d.x, a0.y + d.y), pos2(a1.x + d.x, a1.y + d.y)],
                        stroke,
                    );
                }
            }
        }

        if !patched {
            p.text(
                pos2(center_x, label_top),
                Align2::CENTER_TOP,
                "1km & 6km AGL",
                lbl_font.clone(),
                BARB_BLUE,
            );
            p.text(
                pos2(center_x, label_top + line_h),
                Align2::CENTER_TOP,
                "Wind Barbs",
                lbl_font,
                BARB_BLUE,
            );
        } else {
            for (index, (slot_id, canonical_label, color)) in [
                ("kin.wind_1km", "1km Wind", BARB_RED),
                ("kin.wind_6km", "6km Wind", BARB_BLUE),
            ]
            .into_iter()
            .enumerate()
            {
                let y = label_top + index as f32 * line_h;
                match self.patch(DiagnosticTablePanelKind::Kinematics, slot_id) {
                    Some(NativeDiagnosticSlotPatch::Blank) => {}
                    Some(NativeDiagnosticSlotPatch::Replace(row)) => {
                        let label = if row.label.trim().is_empty() {
                            canonical_label
                        } else {
                            row.label.trim()
                        };
                        let text = format!("{label} = {}", Self::replacement_value(row));
                        let font = self.fitted_cell_font(p, &text, rw - 2.0);
                        p.text(
                            pos2(center_x, y),
                            Align2::CENTER_TOP,
                            text,
                            font,
                            row.color.unwrap_or(color),
                        );
                    }
                    None => {
                        p.text(
                            pos2(center_x, y),
                            Align2::CENTER_TOP,
                            canonical_label,
                            lbl_font.clone(),
                            color,
                        );
                    }
                }
            }
        }
    }

    // ---- column 3: composite indices ----------------------------------
    fn col_comp(&self, p: &Painter, r: Rect, include_ship_chart: bool) {
        let dv = self.dv;
        let fg = self.st.fg_color;
        let rh = self.rh;
        let (x, w) = (r.left(), r.width());
        let mut y = r.top();

        let pesk = fin(dv.peskov);
        let mcs = fin(dv.mcs_index);
        let ehi1 = fin(dv.ehi_0_1km);
        let ehi3 = fin(dv.ehi_0_3km);
        let lscp = fin(dv.lscp).or(fin(dv.left_scp));
        let nstp = fin(dv.nstp);
        let mshe = fin(dv.modified_sherbe);
        let lrgh = fin(dv.lrghail);
        let swt = fin(dv.sweat);
        let top_list: [(&str, &str, String, Color32); 8] = [
            (
                "severe.ehi_0_1km",
                "EHI 0-1km",
                f1o(ehi1),
                ehi_color(ehi1, fg),
            ),
            (
                "severe.ehi_0_3km",
                "EHI 0-3km",
                f1o(ehi3),
                ehi_color(ehi3, fg),
            ),
            ("severe.vgp", "VGP", float2str(dv.vgp, 2), fg),
            (
                "severe.peskov",
                "Peskov Index",
                f1o(pesk),
                peskov_color(pesk, fg),
            ),
            (
                "severe.mcs",
                "MCS Index",
                f1o(mcs),
                mcs_color(mcs, fin(self.prof.mupcl.bplus), fg),
            ),
            ("severe.sweat", "SWEAT", i0o(swt), sweat_color(swt, fg)),
            (
                "severe.moshe",
                "MOSHE",
                f1o(mshe),
                gradient(mshe, 1.0, 2.0, 3.0, true, fg),
            ),
            (
                "severe.lrghail",
                "LRGHAIL",
                f1o(lrgh),
                lrghail_color(lrgh, fg),
            ),
        ];
        let hgz = fin(dv.hgz_cape);
        let ncape = fin(dv.ncape);
        let wbz = fin(dv.wbz_height);
        let ecape = fin(dv.ecape);
        // (label, value, unit suffix, color); the right entry is optional.
        type BotEntry = (&'static str, &'static str, String, &'static str, Color32);
        let bot: [(BotEntry, Option<BotEntry>); 4] = [
            (
                (
                    "severe.hgz_cape",
                    "HGZ CAPE",
                    i0o(hgz),
                    " J/kg",
                    gradient(hgz, 1000.0, 2500.0, 4000.0, true, fg),
                ),
                Some((
                    "severe.nstp",
                    "NSTP",
                    f1o(nstp),
                    "",
                    gradient(nstp, 1.0, 2.0, 4.0, true, fg),
                )),
            ),
            (
                (
                    "severe.ncape",
                    "NCAPE",
                    f2o(ncape),
                    " J/kg/m",
                    gradient(ncape, 0.1, 0.2, 0.3, true, fg),
                ),
                Some((
                    "severe.ecape",
                    "ECAPE",
                    i0o(ecape),
                    " J/kg",
                    gradient(ecape, 1000.0, 2500.0, 4000.0, true, fg),
                )),
            ),
            (
                (
                    "severe.lscp",
                    "LSCP",
                    f1o(lscp),
                    "",
                    gradient(lscp, -1.0, -4.0, -8.0, false, fg),
                ),
                None,
            ),
            (
                ("severe.wbz_height", "WBZ Height", i0o(wbz), " m AGL", fg),
                None,
            ),
        ];

        // SHIP box-and-whisker chart at the TOP: the two-column indices only
        // take ceil(8/2) = 4 rows, freeing vertical slack for the chart.
        let top_rows = 4.0f32;
        let n_rows = top_rows + bot.len() as f32;
        let slack = (r.height() - n_rows * rh).max(0.0);
        const MID_GAP: f32 = 12.0; // gap around the indices -> CAPE divider
        const CHART_DIV: f32 = 8.0; // gap the SHIP chart's divider consumes
        let (chart_h, mid_gap) = if include_ship_chart && slack > 70.0 {
            (slack - MID_GAP - CHART_DIV, MID_GAP)
        } else {
            (0.0, slack.max(6.0))
        };
        if chart_h >= 50.0 {
            crate::panels::ship_inset::draw(
                p,
                Rect::from_min_size(pos2(x, y), vec2(w, chart_h)),
                self.prof,
                dv,
                self.st,
            );
            y += chart_h + 2.0;
            p.line_segment([pos2(x, y), pos2(x + w, y)], Stroke::new(1.0, RULE));
            y += CHART_DIV - 2.0;
        }

        // Two-column index layout with a compact right readout column.
        let col_gutter = 6.0;
        let min_right_w = (w * 0.32).max(60.0);
        let right_x = (x + w * 0.54).min(x + w - min_right_w);
        let left_w = (right_x - x - col_gutter).max(1.0);
        let right_w = (x + w - right_x - 2.0).max(1.0);
        for (idx, (slot_id, lbl, val, c)) in top_list.iter().enumerate() {
            let (cx, cw, ri) = if idx < 4 {
                (x, left_w, idx)
            } else {
                (right_x, right_w, idx - 4)
            };
            self.inline_slot(
                p,
                DiagnosticTablePanelKind::Severe,
                slot_id,
                cx,
                cw,
                y + ri as f32 * rh,
                lbl,
                val,
                *c,
            );
        }
        y += top_rows * rh;
        y += mid_gap / 2.0;
        p.line_segment([pos2(x, y), pos2(x + w, y)], Stroke::new(1.0, RULE));
        y += mid_gap / 2.0;
        for (left, right) in bot {
            let (slot_id, lbl, val, sfx, c) = left;
            let text = if val != MISS {
                format!("{val}{sfx}")
            } else {
                MISS.to_string()
            };
            if let Some((right_id, rl, rv, rs, rc)) = right {
                self.inline_slot(
                    p,
                    DiagnosticTablePanelKind::Severe,
                    slot_id,
                    x,
                    left_w,
                    y,
                    lbl,
                    &text,
                    c,
                );
                let rtext = if rv != MISS {
                    format!("{rv}{rs}")
                } else {
                    MISS.to_string()
                };
                self.inline_slot(
                    p,
                    DiagnosticTablePanelKind::Severe,
                    right_id,
                    right_x,
                    right_w,
                    y,
                    rl,
                    &rtext,
                    rc,
                );
            } else {
                self.inline_slot(
                    p,
                    DiagnosticTablePanelKind::Severe,
                    slot_id,
                    x,
                    w,
                    y,
                    lbl,
                    &text,
                    c,
                );
            }
            y += rh;
        }
    }
}

// ---------------------------------------------------------------------------
// AGL barb pair path building (port of `_barb_path`: needs per-barb color
// and scale, which `crate::barbs::draw_barb` does not expose)
// ---------------------------------------------------------------------------

/// A barb path in local coordinates (plotted point at the origin), already
/// rotated to its direction and scaled, with its bounding box.
struct BarbPath {
    segs: Vec<(Vec2, Vec2)>,
    /// Calm marker radius (drawn instead of a staff for < 3 kt).
    circle: Option<f32>,
    min: Vec2,
    max: Vec2,
}

fn build_barb(wdir: f64, wspd: f64, shemis: bool, scale: f32) -> BarbPath {
    let mut spd = ((wspd / 5.0).round() * 5.0) as i64;
    if spd <= 0 {
        let r = 3.0 * scale;
        return BarbPath {
            segs: Vec::new(),
            circle: Some(r),
            min: Vec2::new(-r, -r),
            max: Vec2::new(r, r),
        };
    }
    // Staff + barbs/flags in barb-local coordinates (staff along +x), same
    // line work as `crate::barbs` (25-px staff, 10-px full barbs, 50-kt
    // flags), then scale and rotate by (wdir - 90) in y-down space.
    let side = if shemis { -1.0 } else { 1.0 };
    let mut raw: Vec<(Vec2, Vec2)> = Vec::new();
    let mut pen = Vec2::ZERO;
    fn line_to(raw: &mut Vec<(Vec2, Vec2)>, pen: &mut Vec2, to: Vec2) {
        raw.push((*pen, to));
        *pen = to;
    }
    line_to(&mut raw, &mut pen, Vec2::new(25.0, 0.0));
    while spd >= 50 {
        let p0 = pen;
        line_to(&mut raw, &mut pen, Vec2::new(p0.x, p0.y + side * 10.0));
        let q = pen;
        line_to(&mut raw, &mut pen, Vec2::new(q.x - 4.0, p0.y));
        pen = Vec2::new(q.x - 6.0, p0.y);
        spd -= 50;
    }
    while spd >= 10 {
        let p0 = pen;
        line_to(&mut raw, &mut pen, Vec2::new(p0.x, p0.y + side * 10.0));
        pen = Vec2::new(p0.x - 4.0, p0.y);
        spd -= 10;
    }
    while spd >= 5 {
        let p0 = pen;
        line_to(&mut raw, &mut pen, Vec2::new(p0.x, p0.y + side * 5.0));
        pen = Vec2::new(p0.x - 4.0, p0.y);
        spd -= 5;
    }

    let ang = ((wdir - 90.0).to_radians()) as f32;
    let (sin, cos) = ang.sin_cos();
    let xf = |v: Vec2| -> Vec2 {
        let v = v * scale;
        Vec2::new(v.x * cos - v.y * sin, v.x * sin + v.y * cos)
    };
    let segs: Vec<(Vec2, Vec2)> = raw.iter().map(|(a, b)| (xf(*a), xf(*b))).collect();
    let mut min = Vec2::ZERO; // the origin is part of the path
    let mut max = Vec2::ZERO;
    for (a, b) in &segs {
        min = min.min(*a).min(*b);
        max = max.max(*a).max(*b);
    }
    BarbPath {
        segs,
        circle: None,
        min,
        max,
    }
}

#[cfg(test)]
mod typography_tests {
    use super::*;
    use crate::skewt::SoundingFontPreset;

    #[test]
    fn parcel_matrix_places_maximum_parcel_level_after_equilibrium_level() {
        assert_eq!(PARCEL_MATRIX_HEADERS.len(), 8);
        assert_eq!(&PARCEL_MATRIX_HEADERS[6..], &["EL", "MPL"]);
    }

    #[test]
    fn shared_typography_reaches_index_board_regular_and_bold_cells() {
        let (base_regular, _, _) = board_fonts(20.0, &SkewTStyle::default());
        let style = SkewTStyle::default()
            .with_font_preset(SoundingFontPreset::TechnicalMonospace)
            .with_text_scale(1.3);
        let (regular, bold, small_bold) = board_fonts(20.0, &style);

        assert!((regular.size - base_regular.size * 1.3).abs() < 0.001);
        assert_eq!(regular.family, egui::FontFamily::Monospace);
        assert_eq!(bold.family, egui::FontFamily::Monospace);
        assert_eq!(small_bold.family, egui::FontFamily::Monospace);
    }
}

#[cfg(test)]
mod mcs_index_colour_tests {
    use super::*;

    const FG: Color32 = Color32::WHITE;

    /// Coniglio et al. (2007) linear predictor, as derived.rs computes it.
    fn index(shear_ms: f64, lr38: f64, mucape: f64, mean_wind_ms: f64) -> f64 {
        13.0 + (-4.59e-2 * shear_ms) + (-1.16 * lr38) + (-6.17e-4 * mucape) + (-0.17 * mean_wind_ms)
    }

    #[test]
    fn quiet_sounding_is_neutral() {
        // WOOF Mexico City 2026-10-29 12Z: MUCAPE 1, index 3.1 (MMP ~4 %).
        assert_eq!(mcs_color(Some(3.1), Some(1.0), FG), FG);
        // Same index with CAPE: still a low probability, still neutral.
        assert_eq!(mcs_color(Some(3.1), Some(1500.0), FG), FG);
    }

    #[test]
    fn mcs_environment_is_alarm() {
        // 25 m/s shear, 7.5 C/km, 3000 J/kg, 20 m/s mean wind: index ~ -2.1, MMP ~ 89 %.
        let a = index(25.0, 7.5, 3000.0, 20.0);
        assert!(a < -2.0, "index {a}");
        assert_eq!(mcs_color(Some(a), Some(3000.0), FG), GRADIENT_RED);
        // Stronger still: MMP >= 90 % reads pink.
        assert_eq!(mcs_color(Some(index(30.0, 8.0, 3500.0, 22.0)), Some(3500.0), FG), GRADIENT_PINK);
        // MMP just over 50 % reads yellow.
        assert_eq!(mcs_color(Some(-0.1), Some(800.0), FG), GRADIENT_YELLOW);
    }

    #[test]
    fn no_instability_or_missing_is_neutral() {
        // Strong winter jet, no CAPE: SHARPpy's MMP is 0 below 100 J/kg.
        assert_eq!(mcs_color(Some(-3.0), Some(20.0), FG), FG);
        assert_eq!(mcs_color(None, Some(3000.0), FG), FG);
        assert_eq!(mcs_color(Some(-3.0), None, FG), FG);
    }
}
