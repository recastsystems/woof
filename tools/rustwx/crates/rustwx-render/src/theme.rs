//! Render themes: the surface, the inks, the basemap linework, the colorbar
//! chrome, the fonts, the text sizes and the colormap overrides, as one named
//! built-in or one JSON file.
//!
//! `RenderPresentation` fixes every chrome colour per visual mode and plot
//! style, the fonts are embedded, and the one request-level `background`
//! override moves the surface without moving the inks -- so a black surface
//! produced black titles on black.  A theme is the missing seam: it is
//! applied ONCE, at the end of `RenderPresentation::for_mode_with_style`, so
//! every product family (direct, derived, windowed, generic planes) takes it
//! through the same door, and it is a no-op when no theme is active, which
//! is what keeps the standing pixel-identity claim for every existing render.
//!
//! ## File format (JSON; every key optional; unknown keys refused)
//!
//! ```json
//! {
//!   "name": "dark",
//!   "surface": {"canvas": "#000000", "map": "#000000",
//!               "ocean": "#12191c", "land": "#000000", "lake": "#12191c"},
//!   "ink": {"primary": "#e2edf0", "secondary": "#94afb8", "muted": "#64818b",
//!           "subtle": "#476069", "halo": "#000000",
//!           "highlight": "#7dcaca", "attention": "#f97b46"},
//!   "grid": {"color": "#354b53", "alpha": 1.0, "frame": "#476069",
//!            "hairline": "#223137"},
//!   "linework": {"coast": "#64818b", "state": "#64818b",
//!                "international": "#64818b", "county": "#476069",
//!                "lake": "#64818b", "county_visible": true},
//!   "colorbar": {"frame": "#476069", "divider": "#223137",
//!                "tick": "#94afb8", "label": "#94afb8"},
//!   "fonts": {"regular": "fonts/Regular.ttf", "bold": "fonts/Bold.ttf",
//!             "stack": ["Inter", "Source Sans 3", "sans-serif"],
//!             "mono_stack": ["Berkeley Mono", "Menlo", "Consolas", "monospace"]},
//!   "text": {"title_size": 1.0, "label_size": 1.0,
//!            "source_label": "my model 1.2.3", "model_label": "MINE"},
//!   "colormaps": {"sequential": ["#005f60", "#7dcaca"],
//!                 "diverging": ["#eb6e39", "#223137", "#5aa7a7"],
//!                 "products": {"composite_reflectivity": ["#000000", "#ffffff"]}},
//!   "mesh": {"edge": "#223137", "edge_width": 1.0, "edges": true,
//!            "empty": "#12191c"},
//!   "footer": {"height": 64, "background": "#000000", "rule": "#223137",
//!              "logo": "assets/wordmark-dark.png", "logo_height": 22,
//!              "title": "{product_title}",
//!              "caption": "valid {valid_time} | {mesh_or_grid} | {leg} | {note}",
//!              "title_ink": "#e2edf0", "ink": "#94afb8"}
//! }
//! ```
//!
//! `mesh` styles the polygon-mesh product family (`mesh:` / `meshdiff:` in
//! `rw_wrfbatch`): the ink and width of the per-cell hairline, whether the
//! hairline is drawn at all, and the fill a cell with no value takes (a
//! regional mesh's outer relaxation ring, a masked column).  It touches no
//! other family, so both built-ins carry one.
//!
//! `footer` is a neutral strip drawn UNDER the finished frame: a fixed-height
//! band with a background, an optional 1 px rule along its top, an optional
//! logo PNG at a fixed pixel height on the left, and two text lines from the
//! `title` and `caption` templates.  The five template fields are
//! `{product_title}`, `{valid_time}`, `{mesh_or_grid}`, `{leg}` and `{note}`;
//! any other field name is refused by name.  A caption segment whose fields
//! are all empty is dropped along with its `|` separator, so one template
//! serves a leg that has a note and a leg that does not.  Both built-in
//! themes leave `footer` unset and draw no strip, which is what keeps every
//! existing render byte-identical.
//!
//! Colours are `#rrggbb` or `#rrggbbaa`.  A malformed colour or an unknown
//! key is a refusal that names the key: a theme is read once per process
//! and a typo that silently kept the default would restyle nothing while
//! reporting success.  Font paths are relative to the theme file.  The two
//! font stacks are informational for the renderer (rusttype draws one file)
//! and are what a companion chart style reads, so one file names the type
//! for every surface.
//!
//! `colormaps.levels` maps product names to explicit, strictly increasing
//! physical boundaries in displayed units. It preserves the palette and leaves
//! unnamed products unchanged. Use it for a common range across a sequence.
//!
//! `colormaps.products` keys are the field product names the renderer sees
//! (canonical field keys for direct products, recipe slugs for derived ones);
//! `sequential` replaces
//! the auto-ranged ramp the generic `var:` plane lane invents; `diverging`
//! is reserved for zero-centred difference products and is read by the lane
//! that draws them.  The operational weather ladders (reflectivity, QPF,
//! temperature, ...) are never touched by a theme unless a product override
//! names them.
//!
//! `text.source_label` is the right-hand subtitle drawn in place of the
//! derived provenance label (`source: <model>`) on every product that
//! carries one; a product that draws no provenance (a panel inside a
//! composite) stays bare.  Unset, the derived label is drawn, which is
//! what the `default` and `dark` built-ins do.
//!
//! `text.model_label` is the model name drawn in the metadata row (`Init
//! 05/26 15Z | d01 3 km | WRF`) and in a model-prefixed title, in place of
//! the name the product derived from its store identity (`WRF`, `HRRR`).
//! Unset keeps the derived name.  Both labels take the `{version}` token.
//!
//! The two WOOF built-ins are tables, not code: `themes/woof-light.json`
//! and `themes/woof-dark.json` beside this crate, compiled in.  They carry
//! WOOF's own labels, so a frame drawn in either names WOOF as its model
//! and its source; a caller who wants another name extends one of them in
//! its own file (`{"extends": "woof-light", "text": {...}}`) or edits the
//! table.
//!
//! `woof-light-tc` is woof-light with the `tc` sheet's tables (its wind and
//! IR ladders and its MSLP isobars) laid over it: the look WOOF draws a
//! tropical cyclone run's maps with.  It is composed from the two tables,
//! not written out a third time.

use crate::color::Rgba;
use crate::request::{Color, ColorScale, DiscreteColorScale};
use crate::request::ProductKey;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// The tokens a theme's text may carry, filled in when a frame is drawn.
/// Anything else in braces refuses the theme when it loads: a theme file
/// with an unsupported or misspelled token would otherwise print literal
/// braces on every frame.
pub const TEMPLATE_TOKENS: &[&str] = &["{version}"];

static TEMPLATE_VERSION: std::sync::RwLock<Option<String>> = std::sync::RwLock::new(None);

/// The version `{version}` stands for (the executing engine's, from the
/// caller's source label).  `None` fills it with nothing.
pub fn set_template_version(version: Option<String>) {
    if let Ok(mut slot) = TEMPLATE_VERSION.write() {
        *slot = version.map(|v| v.trim().to_string()).filter(|v| !v.is_empty());
    }
}

fn check_template_tokens(text: &str) -> Result<(), String> {
    let mut rest = text;
    while let Some(open) = rest.find('{') {
        let after = &rest[open..];
        let Some(close) = after.find('}') else {
            return Err("an opening brace is never closed".into());
        };
        let token = &after[..=close];
        if !TEMPLATE_TOKENS.contains(&token) {
            return Err(format!(
                "{token} is not a template token; the only one is {}",
                TEMPLATE_TOKENS.join(", ")
            ));
        }
        rest = &after[close + 1..];
    }
    Ok(())
}

fn fill_template_tokens(text: &str) -> String {
    let version = TEMPLATE_VERSION.read().ok().and_then(|slot| slot.clone()).unwrap_or_default();
    text.replace("{version}", &version).split_whitespace().collect::<Vec<_>>().join(" ")
}

fn merge_json(base: &mut serde_json::Value, overlay: serde_json::Value) {
    match (base, overlay) {
        (serde_json::Value::Object(base), serde_json::Value::Object(overlay)) => {
            for (key, value) in overlay {
                match base.get_mut(&key) {
                    Some(existing) if existing.is_object() && value.is_object() => {
                        merge_json(existing, value)
                    }
                    _ => {
                        base.insert(key, value);
                    }
                }
            }
        }
        (base, overlay) => *base = overlay,
    }
}

/// Environment variable naming the active theme (a built-in name or a JSON
/// path) for processes that are not started through `rw_wrfbatch --theme`.
pub const THEME_ENV: &str = "RUSTWX_THEME";

/// The WOOF built-ins, as tables (see the module docs).
const WOOF_LIGHT_JSON: &str = include_str!("../themes/woof-light.json");
const WOOF_DARK_JSON: &str = include_str!("../themes/woof-dark.json");
/// The tropical-cyclone sheet: a fixed 10 m wind ladder to 180 kt with its
/// band edges on the Saffir-Simpson thresholds (34, 50, 64, 83, 96, 113,
/// 137 kt) and MSLP isobars every 4 hPa from 880 hPa, labelled every 8 hPa.
const TC_JSON: &str = include_str!("../themes/tc.json");

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RenderThemeFile {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// A built-in theme name or a theme file (relative to this one) whose
    /// every value this file starts from; this file's own values win.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extends: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub surface: Option<SurfaceSpec>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ink: Option<InkSpec>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub grid: Option<GridSpec>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub linework: Option<LineworkSpec>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub colorbar: Option<ColorbarSpec>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fonts: Option<FontSpec>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<TextSpec>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub colormaps: Option<ColormapSpec>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub contours: Option<ContourSpec>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mesh: Option<MeshSpec>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub footer: Option<FooterSpec>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SurfaceSpec {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub canvas: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub map: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ocean: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub land: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lake: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InkSpec {
    /// Title ink.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub primary: Option<String>,
    /// Subtitle, colorbar labels and ticks.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub secondary: Option<String>,
    /// Coast, state and international linework when `linework` is silent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub muted: Option<String>,
    /// County linework when `linework` is silent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subtle: Option<String>,
    /// Label halo: the surface colour on a dark theme, never white.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub halo: Option<String>,
    /// The one highlight colour (a marked leg, a release line).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub highlight: Option<String>,
    /// The one attention colour (a threshold crossing, a highlighted isotherm).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attention: Option<String>,
    /// Contour and barb ink substituted for dark requested inks.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub contour: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GridSpec {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub color: Option<String>,
    /// 0.0-1.0, multiplied into the grid colour's alpha.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub alpha: Option<f32>,
    /// Map frame and domain outline.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frame: Option<String>,
    /// Colorbar dividers and other 1 px separators.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hairline: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LineworkSpec {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub coast: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub state: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub international: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub county: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lake: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub county_visible: Option<bool>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ColorbarSpec {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frame: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub divider: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tick: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FontSpec {
    /// TrueType/OpenType file for regular text, relative to the theme file.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub regular: Option<String>,
    /// TrueType/OpenType file for bold text, relative to the theme file.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bold: Option<String>,
    /// Family names in preference order, for companion chart styles.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub stack: Vec<String>,
    /// Monospace family names in preference order, for companion chart styles.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub mono_stack: Vec<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TextSpec {
    /// Title size factor; 1.0 is the renderer's own size.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title_size: Option<f32>,
    /// Subtitle and colorbar label size factor; 1.0 is the renderer's own size.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label_size: Option<f32>,
    /// Right-hand subtitle text drawn in place of the derived provenance
    /// label (`source: <model>`); unset keeps the derived label.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_label: Option<String>,
    /// Model name drawn in the metadata row and in a model-prefixed title
    /// in place of the one the product derived (`WRF`, `HRRR`); unset
    /// keeps the derived name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_label: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ColormapSpec {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub sequential: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub diverging: Vec<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub products: BTreeMap<String, Vec<String>>,
    /// Explicit physical colour boundaries, in the product's displayed units.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub levels: BTreeMap<String, Vec<f64>>,
}

/// Contour overlays a theme sets, keyed by contoured field (`mslp`).
///
/// THE BREAKAGE THIS PREVENTS: the operational MSLP isobars are fixed at
/// 960-1045 hPa every 2 hPa, so a hurricane core (a 938 hPa eye) was blank
/// inside the 960 ring while the 2 hPa rings piled up and their labels
/// overlapped around it.  A tropical-cyclone sheet names its own set here
/// (`{"levels":{"mslp":[880,884,...]},"label_every":{"mslp":2}}`).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContourSpec {
    /// Explicit, strictly increasing contour levels in displayed units.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub levels: BTreeMap<String, Vec<f64>>,
    /// Every Nth level is a major (thicker, labelled) contour.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub label_every: BTreeMap<String, usize>,
    /// 5x5 box-smoothing passes on the contoured field (0-12).
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub smooth_passes: BTreeMap<String, usize>,
}

/// The polygon-mesh family's own linework and empty-cell fill.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeshSpec {
    /// The per-cell hairline ink.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub edge: Option<String>,
    /// Hairline width in final image pixels; scaled with the supersample.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub edge_width: Option<f32>,
    /// Whether the hairline is drawn at all.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub edges: Option<bool>,
    /// The fill for a cell carrying no value: a regional mesh's outer
    /// relaxation ring, or a masked column.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub empty: Option<String>,
}

/// The strip drawn under the finished frame.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FooterSpec {
    /// Strip height in final image pixels.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub height: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub background: Option<String>,
    /// A 1 px rule along the strip's top edge; unset draws none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rule: Option<String>,
    /// A PNG placed at the strip's left, path relative to the theme file.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logo: Option<String>,
    /// The logo's drawn height in final image pixels; the width follows the
    /// file's own aspect ratio, so the artwork is never stretched.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logo_height: Option<u32>,
    /// The bold first line's template.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// The second line's template.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub caption: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title_ink: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ink: Option<String>,
}

impl RenderThemeFile {
    pub fn from_json(text: &str) -> Result<Self, String> {
        serde_json::from_str(text).map_err(|err| format!("theme JSON: {err}"))
    }

    /// This file laid over the theme its `extends` names: every value the
    /// parent sets that this file does not.  `base` resolves a relative
    /// parent path; a chain deeper than eight is refused as a loop.
    pub fn with_parent(self, base: Option<&Path>, depth: usize) -> Result<Self, String> {
        let Some(parent_spec) = self.extends.clone() else {
            return Ok(self);
        };
        if depth >= 8 {
            return Err(format!("extends {parent_spec:?}: the chain is deeper than eight themes"));
        }
        let (parent, parent_base) = match RenderTheme::builtin_file(&parent_spec) {
            Some(file) => (file, None),
            None => {
                let path = match base {
                    Some(base) => base.join(&parent_spec),
                    None => PathBuf::from(&parent_spec),
                };
                let file = Self::load(&path)?;
                (file, path.parent().map(Path::to_path_buf))
            }
        };
        let mut parent = parent.with_parent(parent_base.as_deref().or(base), depth + 1)?;
        // Inherited assets belong to their declaring file, even when the
        // child theme lives in another directory.
        if let Some(parent_base) = parent_base.as_deref() {
            let parent_base = std::fs::canonicalize(parent_base)
                .map_err(|err| format!("theme parent directory {}: {err}", parent_base.display()))?;
            let resolve = |value: &mut Option<String>| {
                if let Some(text) = value {
                    let path = PathBuf::from(&*text);
                    if !path.is_absolute() {
                        *text = parent_base.join(path).to_string_lossy().into_owned();
                    }
                }
            };
            if let Some(fonts) = parent.fonts.as_mut() {
                resolve(&mut fonts.regular);
                resolve(&mut fonts.bold);
            }
            if let Some(footer) = parent.footer.as_mut() {
                resolve(&mut footer.logo);
            }
        }
        let mut merged = serde_json::to_value(&parent).map_err(|err| err.to_string())?;
        let mut child = serde_json::to_value(&self).map_err(|err| err.to_string())?;
        if let Some(object) = child.as_object_mut() {
            object.remove("extends");
        }
        merge_json(&mut merged, child);
        serde_json::from_value(merged).map_err(|err| format!("extends {parent_spec:?}: {err}"))
    }

    pub fn load(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path)
            .map_err(|err| format!("read theme {}: {err}", path.display()))?;
        Self::from_json(&text).map_err(|err| format!("{}: {err}", path.display()))
    }
}

/// The per-presentation part of a theme: every field is an override, `None`
/// is "the presentation's own value".  `Copy`, because `RenderPresentation`
/// is `Copy` and carries one of these.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PresentationTheme {
    pub ocean: Option<Rgba>,
    pub land: Option<Rgba>,
    pub lake: Option<Rgba>,
    pub coast: Option<Rgba>,
    pub state: Option<Rgba>,
    pub international: Option<Rgba>,
    pub county: Option<Rgba>,
    pub lake_line: Option<Rgba>,
    pub county_visible: Option<bool>,
    pub frame: Option<Rgba>,
    pub contour_ink: Option<Rgba>,
    /// Label halo.  `None` keeps the renderer's white halos.
    pub halo: Option<Rgba>,
    /// Title size factor in permille (1000 = 1.0), so the struct stays `Eq`.
    pub title_size_permille: u16,
    /// Subtitle and colorbar label size factor in permille.
    pub label_size_permille: u16,
    /// Whether any override at all is set; a fast path for the hot loops.
    pub active: bool,
}

impl PresentationTheme {
    pub fn title_factor(self) -> f32 {
        permille_factor(self.title_size_permille)
    }

    pub fn label_factor(self) -> f32 {
        permille_factor(self.label_size_permille)
    }

    /// The halo colour for a label, keeping the caller's alpha.
    pub fn halo_with_alpha(self, alpha: u8) -> Rgba {
        match self.halo {
            Some(halo) => Rgba::with_alpha(halo.r, halo.g, halo.b, alpha),
            None => Rgba::with_alpha(255, 255, 255, alpha),
        }
    }

    /// A requested white halo becomes the theme halo; any other colour is
    /// the caller's deliberate choice and stays.
    pub fn substitute_white_halo(self, requested: Rgba) -> Rgba {
        match self.halo {
            Some(halo) if requested.r == 255 && requested.g == 255 && requested.b == 255 => {
                Rgba::with_alpha(halo.r, halo.g, halo.b, requested.a)
            }
            _ => requested,
        }
    }

    /// Dark contour and barb inks (the recipes' near-black defaults) take the
    /// theme's contour ink; a coloured request stays as drawn.
    pub fn substitute_dark_ink(self, requested: Rgba) -> Rgba {
        match self.contour_ink {
            Some(ink) if requested.r.max(requested.g).max(requested.b) < 110 => {
                Rgba::with_alpha(ink.r, ink.g, ink.b, requested.a)
            }
            _ => requested,
        }
    }
}

fn permille_factor(permille: u16) -> f32 {
    if permille == 0 {
        1.0
    } else {
        f32::from(permille) / 1000.0
    }
}

fn factor_permille(factor: Option<f32>) -> u16 {
    match factor {
        Some(value) if value.is_finite() && value > 0.0 => {
            (value.clamp(0.5, 3.0) * 1000.0).round() as u16
        }
        _ => 1000,
    }
}

/// The resolved polygon-mesh style.  `Copy` and `Eq` so the render options
/// carry it beside the other per-presentation values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MeshTheme {
    pub edge: Rgba,
    /// Hairline width in permille of a final image pixel (1000 = 1.0 px), so
    /// the struct stays `Eq`.
    pub edge_width_permille: u16,
    pub draw_edges: bool,
    pub empty: Rgba,
}

impl Default for MeshTheme {
    /// The renderer's own look for a mesh: a dark hairline and a pale
    /// empty-cell fill, which is what reads on the default light canvas.
    fn default() -> Self {
        Self {
            edge: Rgba::with_alpha(0x3c, 0x3c, 0x3c, 255),
            edge_width_permille: 1000,
            draw_edges: true,
            empty: Rgba::with_alpha(0xe7, 0xe7, 0xe7, 255),
        }
    }
}

impl MeshTheme {
    pub fn edge_width_px(self) -> f32 {
        permille_factor(self.edge_width_permille)
    }
}

/// The five fields a footer template may name.  Stated once, so the
/// validator, the substitution and the documentation cannot drift apart.
pub const FOOTER_FIELDS: [&str; 5] = [
    "product_title",
    "valid_time",
    "mesh_or_grid",
    "leg",
    "note",
];

/// The resolved footer strip.
#[derive(Debug, Clone, PartialEq)]
pub struct FooterTheme {
    pub height: u32,
    pub background: Rgba,
    pub rule: Option<Rgba>,
    pub logo: Option<PathBuf>,
    pub logo_height: u32,
    pub title_template: String,
    pub caption_template: String,
    pub title_ink: Rgba,
    pub ink: Rgba,
}

/// Refuse a template that names a field the substitution cannot fill.
///
/// A `{valid}` typo would otherwise draw the brace text on every sheet in a
/// delivered gallery, and the strip is the one line of a plot a reader
/// quotes back.  The refusal names the field AND the five that exist.
fn validate_footer_template(key: &str, template: &str) -> Result<(), String> {
    let bytes: Vec<char> = template.chars().collect();
    let mut index = 0usize;
    while index < bytes.len() {
        if bytes[index] != '{' {
            index += 1;
            continue;
        }
        let start = index + 1;
        let mut end = start;
        while end < bytes.len() && bytes[end] != '}' {
            end += 1;
        }
        if end >= bytes.len() {
            return Err(format!(
                "{key}: '{template}' opens a {{ that is never closed"
            ));
        }
        let name: String = bytes[start..end].iter().collect();
        if !FOOTER_FIELDS.contains(&name.as_str()) {
            return Err(format!(
                "{key}: '{{{name}}}' is not a footer field; the fields are {}",
                FOOTER_FIELDS.join(", ")
            ));
        }
        index = end + 1;
    }
    Ok(())
}

/// A resolved theme: colours parsed, paths made absolute.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct RenderTheme {
    pub name: String,
    pub canvas: Option<Rgba>,
    pub map: Option<Rgba>,
    pub title_ink: Option<Rgba>,
    pub subtitle_ink: Option<Rgba>,
    pub highlight: Option<Rgba>,
    pub attention: Option<Rgba>,
    pub grid: Option<Rgba>,
    pub hairline: Option<Rgba>,
    pub colorbar_frame: Option<Rgba>,
    pub colorbar_divider: Option<Rgba>,
    pub colorbar_tick: Option<Rgba>,
    pub colorbar_label: Option<Rgba>,
    pub presentation: PresentationTheme,
    pub font_regular: Option<PathBuf>,
    pub font_bold: Option<PathBuf>,
    pub font_stack: Vec<String>,
    pub mono_stack: Vec<String>,
    pub sequential: Vec<Rgba>,
    pub diverging: Vec<Rgba>,
    pub products: BTreeMap<String, Vec<Rgba>>,
    pub levels: BTreeMap<String, Vec<f64>>,
    /// `contours.*`: per-field contour levels, major interval and smoothing.
    pub contours: ContourSpec,
    /// The right-hand subtitle drawn in place of the derived provenance
    /// label; `None` draws the derived label.
    pub source_label: Option<String>,
    /// The model name drawn in place of the derived one; `None` draws the
    /// derived name.
    pub model_label: Option<String>,
    /// The file this theme was read from, when it was a file.
    pub source: Option<PathBuf>,
    /// The polygon-mesh family's linework and empty-cell fill.  Always
    /// present: the family has to draw something, and no other family reads
    /// it, so a theme that says nothing about a mesh still renders one.
    pub mesh: MeshTheme,
    /// The strip under the frame.  `None` -- both built-ins -- draws none.
    pub footer: Option<FooterTheme>,
}

impl RenderTheme {
    /// The renderer's own look: no override anywhere.
    pub fn default_theme() -> Self {
        Self {
            name: "default".to_string(),
            ..Self::default()
        }
    }

    /// Neutral greys on black.  Weather ladders untouched; the generic plane
    /// ramp becomes a grey-to-white ramp so a black surface does not swallow
    /// its low end.
    pub fn dark_theme() -> Self {
        Self::from_file_spec(&Self::dark_theme_file(), None).expect("the built-in dark theme parses")
    }

    /// The WOOF light look: the renderer's own inks (its weather ladders,
    /// land and water fills, linework) named as the WOOF theme a caller
    /// can extend.
    pub fn woof_light_theme() -> Self {
        Self::from_file_spec(&Self::woof_light_theme_file(), None)
            .expect("the built-in woof-light theme parses")
    }

    /// The WOOF dark look: the built-in dark surface and inks, for the
    /// viewer.
    pub fn woof_dark_theme() -> Self {
        Self::from_file_spec(&Self::woof_dark_theme_file(), None)
            .expect("the built-in woof-dark theme parses")
    }

    fn woof_light_theme_file() -> RenderThemeFile {
        Self::table_theme_file(WOOF_LIGHT_JSON, "woof-light")
    }

    fn woof_dark_theme_file() -> RenderThemeFile {
        Self::table_theme_file(WOOF_DARK_JSON, "woof-dark")
    }

    fn tc_theme_file() -> RenderThemeFile {
        Self::table_theme_file(TC_JSON, "tc")
    }

    /// The tropical-cyclone look (see [`TC_JSON`]).
    pub fn tc_theme() -> Self {
        Self::from_file_spec(&Self::tc_theme_file(), None).expect("the built-in tc theme parses")
    }

    /// WOOF's light look with the `tc` sheet laid over it: every woof-light
    /// value, then tc's colour ladders, band edges and isobars on top.
    ///
    /// THE BREAKAGE THIS PREVENTS: WOOF draws every map with woof-light, whose
    /// 10 m wind scale stops at 60 kt and whose isobars stop at 960 hPa, so a
    /// typhoon core (WOOF owner canary run_303354e41cb59dc8, Typhoon In-fa,
    /// 942 hPa, 1 km core grid, 2026-10-27) was one colour inside a blank
    /// 960 hPa ring.  WOOF's worker names this theme for a tropical cyclone run.
    fn woof_light_tc_theme_file() -> RenderThemeFile {
        let mut merged = serde_json::to_value(Self::woof_light_theme_file())
            .expect("the built-in woof-light theme serialises");
        let mut tc = serde_json::to_value(Self::tc_theme_file())
            .expect("the built-in tc theme serialises");
        if let Some(object) = tc.as_object_mut() {
            object.remove("name");
            object.remove("extends");
        }
        merge_json(&mut merged, tc);
        let mut file: RenderThemeFile = serde_json::from_value(merged)
            .unwrap_or_else(|err| panic!("the built-in woof-light-tc theme composes: {err}"));
        file.name = Some("woof-light-tc".to_string());
        file
    }

    /// The WOOF tropical-cyclone look (see [`Self::woof_light_tc_theme_file`]).
    pub fn woof_light_tc_theme() -> Self {
        Self::from_file_spec(&Self::woof_light_tc_theme_file(), None)
            .expect("the built-in woof-light-tc theme parses")
    }

    /// A built-in written as a compiled-in JSON table, with its `extends`
    /// (a built-in name) already laid under it.
    fn table_theme_file(text: &str, name: &str) -> RenderThemeFile {
        RenderThemeFile::from_json(text)
            .and_then(|file| file.with_parent(None, 0))
            .unwrap_or_else(|err| panic!("the built-in {name} theme table parses: {err}"))
    }

    /// A built-in theme as the file it is written as, the starting point a
    /// file's `extends` names.
    pub fn builtin_file(name: &str) -> Option<RenderThemeFile> {
        match name.trim().to_ascii_lowercase().as_str() {
            "" | "default" | "light" | "none" | "classic" => Some(RenderThemeFile::default()),
            "dark" => Some(Self::dark_theme_file()),
            "woof-light" => Some(Self::woof_light_theme_file()),
            "woof-dark" => Some(Self::woof_dark_theme_file()),
            "tc" => Some(Self::tc_theme_file()),
            "woof-light-tc" => Some(Self::woof_light_tc_theme_file()),
            _ => None,
        }
    }

    fn dark_theme_file() -> RenderThemeFile {
        RenderThemeFile {
            name: Some("dark".to_string()),
            extends: None,
            surface: Some(SurfaceSpec {
                canvas: Some("#000000".into()),
                map: Some("#000000".into()),
                ocean: Some("#14181c".into()),
                land: Some("#000000".into()),
                lake: Some("#14181c".into()),
            }),
            ink: Some(InkSpec {
                primary: Some("#e6e9ec".into()),
                secondary: Some("#a3abb3".into()),
                muted: Some("#7a848c".into()),
                subtle: Some("#4f5a62".into()),
                halo: Some("#000000".into()),
                highlight: Some("#9ad0e6".into()),
                attention: Some("#f2a35a".into()),
                contour: Some("#d8dde2".into()),
            }),
            grid: Some(GridSpec {
                color: Some("#3a444c".into()),
                alpha: Some(1.0),
                frame: Some("#5a656e".into()),
                hairline: Some("#262d33".into()),
            }),
            linework: None,
            colorbar: Some(ColorbarSpec {
                frame: Some("#5a656e".into()),
                divider: Some("#262d33".into()),
                tick: Some("#a3abb3".into()),
                label: Some("#a3abb3".into()),
            }),
            fonts: None,
            text: None,
            colormaps: Some(ColormapSpec {
                sequential: vec![
                    "#2b333a".into(),
                    "#4a565f".into(),
                    "#6b7a85".into(),
                    "#8fa0ac".into(),
                    "#b5c6d2".into(),
                    "#dbeaf3".into(),
                ],
                diverging: vec![
                    "#f2a35a".into(),
                    "#c97a3c".into(),
                    "#8f5528".into(),
                    "#262d33".into(),
                    "#2f6d80".into(),
                    "#4d9bb4".into(),
                    "#9ad0e6".into(),
                ],
                products: BTreeMap::new(),
                levels: BTreeMap::new(),
            }),
            contours: None,
            // The hairline and the empty-cell fill from the dark surface's
            // own ladder.  No footer: the strip is a deliberate act by the
            // caller who has a logo and a caption to put in it, and a
            // built-in that drew one would change every dark render.
            mesh: Some(MeshSpec {
                edge: Some("#2a333a".into()),
                edge_width: Some(1.0),
                edges: Some(true),
                empty: Some("#12191c".into()),
            }),
            footer: None,
        }
    }

    /// `default` (aliases `light`, `none`) or `dark`.
    pub fn builtin(name: &str) -> Option<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "" | "default" | "light" | "none" | "classic" => Some(Self::default_theme()),
            "dark" => Some(Self::dark_theme()),
            "woof-light" => Some(Self::woof_light_theme()),
            "woof-dark" => Some(Self::woof_dark_theme()),
            "tc" => Some(Self::tc_theme()),
            "woof-light-tc" => Some(Self::woof_light_tc_theme()),
            _ => None,
        }
    }

    pub fn builtin_names() -> &'static [&'static str] {
        &["default", "light", "dark", "woof-light", "woof-dark", "tc", "woof-light-tc"]
    }

    /// A built-in name or a path to a JSON theme file.
    pub fn resolve(spec: &str) -> Result<Self, String> {
        let spec = spec.trim();
        if let Some(theme) = Self::builtin(spec) {
            return Ok(theme);
        }
        let path = Path::new(spec);
        if path.is_file() {
            return Self::from_path(path);
        }
        Err(format!(
            "unknown theme '{spec}': not one of {} and not a readable JSON file",
            Self::builtin_names().join(", ")
        ))
    }

    pub fn from_path(path: &Path) -> Result<Self, String> {
        let file = RenderThemeFile::load(path)?;
        let base = path.parent().map(Path::to_path_buf);
        let file = file
            .with_parent(base.as_deref(), 0)
            .map_err(|err| format!("{}: {err}", path.display()))?;
        let mut theme = Self::from_file_spec(&file, base.as_deref())
            .map_err(|err| format!("{}: {err}", path.display()))?;
        if theme.name == "default" && file.name.is_none() {
            theme.name = path
                .file_stem()
                .and_then(|stem| stem.to_str())
                .unwrap_or("theme")
                .to_string();
        }
        theme.source = Some(path.to_path_buf());
        Ok(theme)
    }

    pub fn from_file_spec(file: &RenderThemeFile, base: Option<&Path>) -> Result<Self, String> {
        if file.extends.is_some() {
            let expanded = file.clone().with_parent(base, 0)
                .map_err(|err| format!("extends: {err}"))?;
            return Self::from_file_spec(&expanded, base);
        }
        let color = |key: &str, value: &Option<String>| -> Result<Option<Rgba>, String> {
            value
                .as_deref()
                .map(|text| parse_color(text).map_err(|err| format!("{key}: {err}")))
                .transpose()
        };
        let surface = file.surface.clone().unwrap_or_default();
        let ink = file.ink.clone().unwrap_or_default();
        let grid = file.grid.clone().unwrap_or_default();
        let linework = file.linework.clone().unwrap_or_default();
        let colorbar = file.colorbar.clone().unwrap_or_default();
        let fonts = file.fonts.clone().unwrap_or_default();
        let text = file.text.clone().unwrap_or_default();
        let colormaps = file.colormaps.clone().unwrap_or_default();
        let mesh_spec = file.mesh.clone();
        let footer_spec = file.footer.clone();
        let source_label = match text.source_label.as_deref().map(str::trim) {
            Some("") => return Err("text.source_label must not be blank".into()),
            Some(label) => {
                check_template_tokens(label)
                    .map_err(|err| format!("text.source_label {label:?}: {err}"))?;
                Some(label.to_string())
            }
            None => None,
        };
        let model_label = match text.model_label.as_deref().map(str::trim) {
            Some("") => return Err("text.model_label must not be blank".into()),
            Some(label) => {
                if label.chars().count() > 32
                    || text.model_label.as_deref().unwrap().chars().any(|ch| ch.is_control() || ch == '|') {
                    return Err("text.model_label must contain 1-32 characters without controls or '|'".into());
                }
                check_template_tokens(label)
                    .map_err(|err| format!("text.model_label {label:?}: {err}"))?;
                Some(label.to_string())
            }
            None => None,
        };

        let canvas = color("surface.canvas", &surface.canvas)?;
        let map = color("surface.map", &surface.map)?.or(canvas);
        let primary = color("ink.primary", &ink.primary)?;
        let secondary = color("ink.secondary", &ink.secondary)?.or(primary);
        let muted = color("ink.muted", &ink.muted)?.or(secondary);
        let subtle = color("ink.subtle", &ink.subtle)?.or(muted);
        let halo = color("ink.halo", &ink.halo)?.or(canvas);
        let highlight = color("ink.highlight", &ink.highlight)?;
        let attention = color("ink.attention", &ink.attention)?;
        let contour = color("ink.contour", &ink.contour)?.or(primary);

        let grid_alpha = match grid.alpha {
            Some(alpha) if alpha.is_finite() && (0.0..=1.0).contains(&alpha) => alpha,
            Some(alpha) => return Err(format!("grid.alpha {alpha} is not within 0.0-1.0")),
            None => 1.0,
        };
        let grid_color = color("grid.color", &grid.color)?.map(|value| Rgba {
            a: ((f32::from(value.a) * grid_alpha).round().clamp(0.0, 255.0)) as u8,
            ..value
        });
        let frame = color("grid.frame", &grid.frame)?.or(grid_color).or(muted);
        let hairline = color("grid.hairline", &grid.hairline)?.or(grid_color);

        let presentation = PresentationTheme {
            ocean: color("surface.ocean", &surface.ocean)?,
            land: color("surface.land", &surface.land)?,
            lake: color("surface.lake", &surface.lake)?,
            coast: color("linework.coast", &linework.coast)?.or(muted),
            state: color("linework.state", &linework.state)?.or(muted),
            international: color("linework.international", &linework.international)?.or(muted),
            county: color("linework.county", &linework.county)?.or(subtle),
            lake_line: color("linework.lake", &linework.lake)?.or(muted),
            county_visible: linework.county_visible,
            frame,
            contour_ink: contour,
            halo,
            title_size_permille: factor_permille(text.title_size),
            label_size_permille: factor_permille(text.label_size),
            active: false,
        };
        let mut presentation = presentation;
        presentation.active = presentation != PresentationTheme::default()
            || canvas.is_some()
            || primary.is_some();

        let resolve_font = |value: &Option<String>| -> Option<PathBuf> {
            value.as_deref().map(|text| {
                let path = PathBuf::from(text);
                match (path.is_absolute(), base) {
                    (false, Some(base)) => base.join(path),
                    _ => path,
                }
            })
        };
        let parse_list = |key: &str, values: &[String]| -> Result<Vec<Rgba>, String> {
            values
                .iter()
                .enumerate()
                .map(|(index, text)| {
                    parse_color(text).map_err(|err| format!("{key}[{index}]: {err}"))
                })
                .collect()
        };
        let mut products = BTreeMap::new();
        for (slug, anchors) in &colormaps.products {
            let parsed = parse_list(&format!("colormaps.products.{slug}"), anchors)?;
            if parsed.len() < 2 {
                return Err(format!(
                    "colormaps.products.{slug}: a colormap needs at least two colours"
                ));
            }
            products.insert(slug.clone(), parsed);
        }
        for (slug, levels) in &colormaps.levels {
            if levels.len() < 2 || levels.len() > 4096
                || levels.iter().any(|v| !v.is_finite())
                || levels.windows(2).any(|v| v[0] >= v[1]) {
                return Err(format!("colormaps.levels.{slug}: need 2..4096 finite, strictly increasing boundaries"));
            }
        }
        let contours = file.contours.clone().unwrap_or_default();
        for (field, levels) in &contours.levels {
            if levels.is_empty() || levels.len() > 4096
                || levels.iter().any(|v| !v.is_finite())
                || levels.windows(2).any(|v| v[0] >= v[1]) {
                return Err(format!("contours.levels.{field}: need 1..4096 finite, strictly increasing levels"));
            }
        }
        for (field, every) in &contours.label_every {
            if *every == 0 {
                return Err(format!("contours.label_every.{field}: must be at least 1"));
            }
        }
        for (field, passes) in &contours.smooth_passes {
            if *passes > 12 {
                return Err(format!("contours.smooth_passes.{field}: {passes} is above 12"));
            }
        }
        let sequential = parse_list("colormaps.sequential", &colormaps.sequential)?;
        if sequential.len() == 1 {
            return Err("colormaps.sequential: a colormap needs at least two colours".into());
        }
        let diverging = parse_list("colormaps.diverging", &colormaps.diverging)?;
        if diverging.len() == 1 {
            return Err("colormaps.diverging: a colormap needs at least two colours".into());
        }

        let mut mesh = MeshTheme::default();
        if let Some(spec) = &mesh_spec {
            if let Some(edge) = color("mesh.edge", &spec.edge)? {
                mesh.edge = edge;
            }
            if let Some(empty) = color("mesh.empty", &spec.empty)? {
                mesh.empty = empty;
            }
            if let Some(width) = spec.edge_width {
                if !width.is_finite() || !(0.0..=8.0).contains(&width) {
                    return Err(format!(
                        "mesh.edge_width {width} is not within 0.0-8.0 pixels"
                    ));
                }
                mesh.edge_width_permille = (width * 1000.0).round() as u16;
            }
            if let Some(edges) = spec.edges {
                mesh.draw_edges = edges;
            }
        }

        let footer = match &footer_spec {
            None => None,
            Some(spec) => {
                let height = spec.height.unwrap_or(64);
                if !(16..=512).contains(&height) {
                    return Err(format!(
                        "footer.height {height} is not within 16-512 pixels"
                    ));
                }
                let logo_height = spec.logo_height.unwrap_or(height.saturating_sub(28).max(12));
                if logo_height == 0 || logo_height > height {
                    return Err(format!(
                        "footer.logo_height {logo_height} does not fit a {height}-pixel strip"
                    ));
                }
                let title_template = spec
                    .title
                    .clone()
                    .unwrap_or_else(|| "{product_title}".to_string());
                let caption_template = spec.caption.clone().unwrap_or_else(|| {
                    "valid {valid_time} | {mesh_or_grid} | {leg} | {note}".to_string()
                });
                validate_footer_template("footer.title", &title_template)?;
                validate_footer_template("footer.caption", &caption_template)?;
                Some(FooterTheme {
                    height,
                    background: color("footer.background", &spec.background)?
                        .or(canvas)
                        .unwrap_or(Rgba::WHITE),
                    rule: color("footer.rule", &spec.rule)?.or(hairline),
                    logo: resolve_font(&spec.logo),
                    logo_height,
                    title_template,
                    caption_template,
                    title_ink: color("footer.title_ink", &spec.title_ink)?
                        .or(primary)
                        .unwrap_or(Rgba::BLACK),
                    ink: color("footer.ink", &spec.ink)?
                        .or(secondary)
                        .unwrap_or(Rgba::with_alpha(0x55, 0x55, 0x55, 255)),
                })
            }
        };

        Ok(Self {
            name: file.name.clone().unwrap_or_else(|| "default".to_string()),
            canvas,
            map,
            title_ink: primary,
            subtitle_ink: secondary,
            highlight,
            attention,
            grid: grid_color,
            hairline,
            colorbar_frame: color("colorbar.frame", &colorbar.frame)?.or(frame),
            colorbar_divider: color("colorbar.divider", &colorbar.divider)?.or(hairline),
            colorbar_tick: color("colorbar.tick", &colorbar.tick)?.or(secondary),
            colorbar_label: color("colorbar.label", &colorbar.label)?.or(secondary),
            presentation,
            font_regular: resolve_font(&fonts.regular),
            font_bold: resolve_font(&fonts.bold),
            font_stack: fonts.stack.clone(),
            mono_stack: fonts.mono_stack.clone(),
            sequential,
            diverging,
            products,
            levels: colormaps.levels.clone(),
            contours,
            source_label,
            model_label,
            source: None,
            mesh,
            footer,
        })
    }

    /// True when this theme changes nothing.
    pub fn is_default(&self) -> bool {
        !self.presentation.active
            && self.canvas.is_none()
            && self.font_regular.is_none()
            && self.font_bold.is_none()
            && self.sequential.is_empty()
            && self.diverging.is_empty()
            && self.products.is_empty()
            && self.levels.is_empty()
            && self.contours == ContourSpec::default()
            && self.source_label.is_none()
            && self.model_label.is_none()
            && self.footer.is_none()
    }

    /// The theme's contour levels for `field` (`mslp`), when it names them.
    pub fn contour_levels(&self, field: &str) -> Option<&[f64]> {
        self.contours.levels.get(field).map(Vec::as_slice)
    }

    /// The theme's major-contour interval for `field`, when it names one.
    pub fn contour_label_every(&self, field: &str) -> Option<usize> {
        self.contours.label_every.get(field).copied()
    }

    /// The theme's smoothing passes for `field`, when it names them.
    pub fn contour_smooth_passes(&self, field: &str) -> Option<usize> {
        self.contours.smooth_passes.get(field).copied()
    }

    /// The right-hand subtitle for a product whose derived provenance
    /// label is `derived`: the theme's `text.source_label` when it names
    /// one and the product carries a label at all, else `derived`
    /// untouched.  A product that draws no provenance (a panel inside a
    /// composite hands `None`) stays bare under every theme.
    pub fn source_subtitle(&self, derived: Option<String>) -> Option<String> {
        match (&self.source_label, derived) {
            (Some(label), Some(_)) => Some(fill_template_tokens(label)),
            (_, derived) => derived,
        }
    }

    /// The model name a product draws whose store identity names it
    /// `derived` (`WRF`): the theme's `text.model_label` when it names one,
    /// else `derived` untouched.
    pub fn model_name(&self, derived: &str) -> String {
        match &self.model_label {
            Some(label) => fill_template_tokens(label),
            None => derived.to_string(),
        }
    }

    /// `n` colours stepped through the sequential ramp, or `None` when the
    /// theme has no sequential ramp.
    pub fn sequential_colors(&self, n: usize) -> Option<Vec<Color>> {
        (!self.sequential.is_empty()).then(|| resample(&self.sequential, n))
    }

    /// `n` colours stepped through the diverging ramp, or `None`.
    pub fn diverging_colors(&self, n: usize) -> Option<Vec<Color>> {
        (!self.diverging.is_empty()).then(|| resample(&self.diverging, n))
    }

    /// The product override for `product`, resampled onto `scale`'s own
    /// levels, or `None` when the theme names no override for it.
    pub fn product_scale_override(
        &self,
        product: &ProductKey,
        scale: &ColorScale,
    ) -> Option<ColorScale> {
        let ProductKey::Named(name) = product;
        let anchors = self.products.get(name);
        let levels = self.levels.get(name);
        if anchors.is_none() && levels.is_none() {
            return None;
        }
        let mut discrete = scale.resolved_discrete();
        if let Some(levels) = levels {
            discrete.levels = levels.clone();
        }
        match (anchors, levels) {
            // Both named: one colour per band of the theme's own levels, so a
            // table that lists exactly levels-1 colours draws them verbatim
            // (a fixed tropical-cyclone wind ladder, the same on every frame).
            (Some(anchors), Some(levels)) => {
                discrete.colors = resample(anchors, levels.len().saturating_sub(1).max(1));
            }
            (Some(anchors), None) => {
                discrete.colors = resample(anchors, discrete.colors.len().max(2));
            }
            _ => {}
        }
        Some(ColorScale::Discrete(discrete))
    }

    /// True when the theme names both band edges and band colours for
    /// `product`: its scale is drawn as listed bands, not a smooth ramp.
    pub fn product_has_banded_scale(&self, product: &ProductKey) -> bool {
        let ProductKey::Named(name) = product;
        self.products.contains_key(name) && self.levels.contains_key(name)
    }

    /// Install the theme's fonts into the text engine (before the first
    /// glyph is drawn), reporting a missing file on stderr and keeping the
    /// embedded face for that weight.
    pub fn install_fonts(&self) {
        if self.font_regular.is_none() && self.font_bold.is_none() {
            return;
        }
        crate::text::install_font_override(self.font_regular.clone(), self.font_bold.clone());
    }
}

/// `n` colours linearly interpolated through `anchors`.
pub fn resample(anchors: &[Rgba], n: usize) -> Vec<Color> {
    let n = n.max(1);
    if anchors.is_empty() {
        return vec![Color::BLACK; n];
    }
    if anchors.len() == 1 || n == 1 {
        return vec![anchors[0].into(); n];
    }
    (0..n)
        .map(|index| {
            let t = index as f64 / (n - 1) as f64;
            let position = t * (anchors.len() - 1) as f64;
            let lower = (position.floor() as usize).min(anchors.len() - 2);
            let frac = position - lower as f64;
            let a = anchors[lower];
            let b = anchors[lower + 1];
            let mix = |x: u8, y: u8| -> u8 {
                (f64::from(x) + (f64::from(y) - f64::from(x)) * frac).round() as u8
            };
            Color::rgba(mix(a.r, b.r), mix(a.g, b.g), mix(a.b, b.b), mix(a.a, b.a))
        })
        .collect()
}

/// `#rrggbb` or `#rrggbbaa`; anything else is a refusal that repeats the text.
pub fn parse_color(value: &str) -> Result<Rgba, String> {
    let text = value.trim();
    let hex = text.strip_prefix('#').unwrap_or(text);
    let byte = |index: usize| {
        hex.get(index..index + 2)
            .and_then(|pair| u8::from_str_radix(pair, 16).ok())
    };
    match hex.len() {
        6 => match (byte(0), byte(2), byte(4)) {
            (Some(r), Some(g), Some(b)) => Ok(Rgba::new(r, g, b)),
            _ => Err(format!("'{text}' is not a #rrggbb colour")),
        },
        8 => match (byte(0), byte(2), byte(4), byte(6)) {
            (Some(r), Some(g), Some(b), Some(a)) => Ok(Rgba::with_alpha(r, g, b, a)),
            _ => Err(format!("'{text}' is not a #rrggbbaa colour")),
        },
        _ => Err(format!("'{text}' is not a #rrggbb or #rrggbbaa colour")),
    }
}

static ACTIVE_THEME: OnceLock<RenderTheme> = OnceLock::new();

/// Install the process-wide theme.  The first call wins; a second call with
/// a different theme is reported and refused, because half a run drawn in
/// one theme and half in another is worse than either.
pub fn install_theme(theme: RenderTheme) -> Result<(), String> {
    theme.install_fonts();
    let name = theme.name.clone();
    match ACTIVE_THEME.set(theme) {
        Ok(()) => Ok(()),
        Err(_) => {
            let current = ACTIVE_THEME.get().map(|t| t.name.as_str()).unwrap_or("?");
            if current == name {
                Ok(())
            } else {
                Err(format!(
                    "theme '{name}' requested after theme '{current}' was already installed"
                ))
            }
        }
    }
}

/// The active theme: what `install_theme` installed, else what `RUSTWX_THEME`
/// names, else the renderer's own look.  An unreadable environment theme is
/// reported once on stderr and the default is used, because this seam cannot
/// return an error; `rw_wrfbatch --theme` resolves the same value up front
/// and refuses loudly instead.
pub fn active_theme() -> &'static RenderTheme {
    ACTIVE_THEME.get_or_init(|| {
        let theme = match std::env::var(THEME_ENV) {
            Ok(spec) if !spec.trim().is_empty() => match RenderTheme::resolve(&spec) {
                Ok(theme) => theme,
                Err(err) => {
                    eprintln!("THEME_ERROR\t{THEME_ENV}: {err}; rendering with the default theme");
                    RenderTheme::default_theme()
                }
            },
            _ => RenderTheme::default_theme(),
        };
        theme.install_fonts();
        theme
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_worker_theme_file_inherits_the_woof_palette_and_both_labels() {
        let file = RenderThemeFile::from_json(
            r##"{"extends":"woof-light","text":{"source_label":"Recast WOOF"}}"##,
        ).expect("worker JSON parses");
        let theme = RenderTheme::from_file_spec(&file, None).expect("worker theme resolves");
        assert_eq!(theme, RenderTheme::builtin("woof-light").expect("WOOF palette"));
        assert_eq!(theme.model_name("WRF"), "WOOF");
        assert_eq!(theme.source_subtitle(Some("source: model".into())), Some("Recast WOOF".into()));
        assert_eq!(theme.source_subtitle(None), None);
        assert_eq!(theme.canvas, None);
        assert_eq!(theme.mesh, RenderTheme::default_theme().mesh);
        for name in ["default", "light", "dark"] {
            let bare = RenderTheme::builtin(name).expect("generic palette");
            assert_eq!(bare.model_name("WRF"), "WRF");
            assert_eq!(bare.source_subtitle(Some("source: model".into())), Some("source: model".into()));
        }
        assert!(RenderTheme::builtin("default").unwrap().is_default());
        assert!(RenderTheme::builtin("light").unwrap().is_default());
    }

    #[test]
    fn direct_theme_spec_inheritance_keeps_unnamed_nested_dark_values() {
        let file = RenderThemeFile::from_json(
            r##"{"extends":"dark","ink":{"primary":"#112233"},
                "colormaps":{"sequential":["#123456","#abcdef"]},
                "text":{"label_size":1.25},"fonts":{"regular":"fonts/regular.ttf"}}"##,
        ).expect("parses");
        let theme = RenderTheme::from_file_spec(&file, Some(Path::new("/themes"))).expect("resolves");
        let mut expected = RenderTheme::dark_theme();
        expected.title_ink = Some(Rgba::new(0x11, 0x22, 0x33));
        expected.sequential = vec![Rgba::new(0x12, 0x34, 0x56), Rgba::new(0xab, 0xcd, 0xef)];
        expected.presentation.label_size_permille = 1250;
        expected.font_regular = Some(PathBuf::from("/themes/fonts/regular.ttf"));
        assert_eq!(theme, expected);
    }

    #[test]
    fn the_theme_model_label_respects_the_metadata_row_bounds() {
        for label in ["", "  ", "split | row", "control\nrow", "abcdefghijklmnopqrstuvwxyz0123456"] {
            let file = RenderThemeFile::from_json(&serde_json::json!({
                "extends": "light", "text": {"model_label": label}
            }).to_string()).expect("parses");
            assert!(RenderTheme::from_file_spec(&file, None).expect_err("invalid model label").contains("text.model_label"));
        }
        assert!(RenderThemeFile::from_json(r##"{"extends":"light","text":{"sorce_label":"probe"}}"##)
            .expect_err("unknown override").contains("sorce_label"));
    }

    static VERSION_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn a_theme_file_round_trips_and_parses_colours() {
        let text = r##"{
            "name": "probe",
            "surface": {"canvas": "#000000", "ocean": "#12191c"},
            "ink": {"primary": "#e2edf0", "secondary": "#94afb8", "halo": "#000000"},
            "grid": {"color": "#354b53", "alpha": 0.5, "frame": "#476069"},
            "linework": {"county": "#476069", "county_visible": true},
            "text": {"title_size": 1.2},
            "colormaps": {"sequential": ["#005f60", "#7dcaca"],
                          "products": {"composite_reflectivity": ["#000000", "#ffffff"]}}
        }"##;
        let file = RenderThemeFile::from_json(text).expect("parses");
        let theme = RenderTheme::from_file_spec(&file, None).expect("resolves");
        assert_eq!(theme.name, "probe");
        assert_eq!(theme.canvas, Some(Rgba::new(0, 0, 0)));
        assert_eq!(theme.map, Some(Rgba::new(0, 0, 0)), "map follows canvas");
        assert_eq!(theme.title_ink, Some(Rgba::new(0xe2, 0xed, 0xf0)));
        assert_eq!(theme.grid, Some(Rgba::with_alpha(0x35, 0x4b, 0x53, 128)));
        assert_eq!(theme.presentation.county, Some(Rgba::new(0x47, 0x60, 0x69)));
        assert_eq!(theme.presentation.county_visible, Some(true));
        assert_eq!(theme.presentation.title_size_permille, 1200);
        assert_eq!(theme.presentation.label_size_permille, 1000);
        assert!(theme.presentation.active);
        assert_eq!(theme.sequential.len(), 2);
        assert!(theme.products.contains_key("composite_reflectivity"));
        let json = serde_json::to_string(&file).expect("serialises");
        let again = RenderThemeFile::from_json(&json).expect("re-parses");
        assert_eq!(again, file);
    }

    #[test]
    fn an_unknown_key_is_refused_by_name() {
        let err = RenderThemeFile::from_json(r##"{"surface": {"canvas": "#000", "paper": "#fff"}}"##)
            .expect_err("unknown key");
        assert!(err.contains("paper"), "{err}");
        let err = RenderThemeFile::from_json(r##"{"background": "#000000"}"##).expect_err("unknown key");
        assert!(err.contains("background"), "{err}");
    }

    #[test]
    fn a_malformed_colour_is_refused_with_its_key() {
        let file = RenderThemeFile::from_json(r##"{"ink": {"primary": "#12"}}"##).expect("parses");
        let err = RenderTheme::from_file_spec(&file, None).expect_err("bad colour");
        assert!(err.contains("ink.primary"), "{err}");
        assert!(err.contains("#12"), "{err}");
    }

    #[test]
    fn the_default_theme_changes_nothing_and_dark_changes_the_surface() {
        let default = RenderTheme::builtin("default").expect("built in");
        assert!(default.is_default());
        assert!(RenderTheme::builtin("light").expect("alias").is_default());
        let dark = RenderTheme::builtin("dark").expect("built in");
        assert!(!dark.is_default());
        assert_eq!(dark.canvas, Some(Rgba::new(0, 0, 0)));
        assert_eq!(dark.presentation.halo, Some(Rgba::new(0, 0, 0)));
        assert!(dark.title_ink.is_some());
        assert!(RenderTheme::builtin("neon").is_none());
        assert!(RenderTheme::resolve("neon").is_err());
    }

    #[test]
    fn a_source_label_replaces_the_derived_provenance_and_a_blank_one_is_refused() {
        let file = RenderThemeFile::from_json(
            r##"{"text": {"source_label": " my model 1.2.3 "}}"##,
        )
        .expect("parses");
        let theme = RenderTheme::from_file_spec(&file, None).expect("resolves");
        assert_eq!(theme.source_label.as_deref(), Some("my model 1.2.3"));
        assert!(!theme.is_default(), "a label is a change to every plot");
        assert_eq!(
            theme.source_subtitle(Some("source: ArWen".to_string())).as_deref(),
            Some("my model 1.2.3")
        );
        assert_eq!(theme.source_subtitle(None), None, "a bare panel stays bare");
        let blank = RenderThemeFile::from_json(r##"{"text": {"source_label": "  "}}"##)
            .expect("parses");
        let err = RenderTheme::from_file_spec(&blank, None).expect_err("blank label");
        assert!(err.contains("text.source_label"), "{err}");
    }

    #[test]
    fn a_version_token_is_filled_and_an_unknown_token_refuses_the_theme() {
        let _version_guard = VERSION_TEST_LOCK.lock().expect("version test lock");
        let file = RenderThemeFile::from_json(r##"{"text": {"source_label": "Model {version}"}}"##)
            .expect("parses");
        let theme = RenderTheme::from_file_spec(&file, None).expect("resolves");
        set_template_version(None);
        assert_eq!(
            theme.source_subtitle(Some("source: ArWen".into())).as_deref(),
            Some("Model"),
            "no version known: the token fills with nothing, never braces"
        );
        set_template_version(Some("2.8.0".into()));
        assert_eq!(
            theme.source_subtitle(Some("source: ArWen".into())).as_deref(),
            Some("Model 2.8.0")
        );
        set_template_version(None);
        for label in ["Model {verison}", "Model {version", "{name} x"] {
            let text = format!(r##"{{"text": {{"source_label": "{label}"}}}}"##);
            let file = RenderThemeFile::from_json(&text).expect("parses");
            let err = RenderTheme::from_file_spec(&file, None).expect_err(label);
            assert!(err.contains("text.source_label"), "{err}");
        }
    }

    #[test]
    fn a_theme_that_extends_a_built_in_keeps_the_parents_values_under_its_own() {
        let child = RenderThemeFile::from_json(
            r##"{"extends": "dark", "name": "mine", "surface": {"ocean": "#010203"}}"##,
        )
        .expect("parses")
        .with_parent(None, 0)
        .expect("extends dark");
        let dark = RenderTheme::builtin_file("dark").unwrap();
        assert_eq!(child.name.as_deref(), Some("mine"));
        let surface = child.surface.clone().unwrap();
        assert_eq!(surface.ocean.as_deref(), Some("#010203"));
        assert_eq!(surface.canvas, dark.surface.unwrap().canvas, "the parent's canvas carries over");
        assert_eq!(child.ink, dark.ink);
        assert!(RenderThemeFile::from_json(r##"{"extends": "nowhere-theme.json"}"##)
            .unwrap()
            .with_parent(None, 0)
            .is_err());
    }

    #[test]
    fn the_woof_themes_are_built_in() {
        let light = RenderTheme::builtin("woof-light").expect("built in");
        assert_eq!(light.name, "woof-light");
        let plain = RenderTheme::from_file_spec(&RenderThemeFile::default(), None).unwrap();
        assert_eq!(light.presentation, plain.presentation, "the labels change no ink");
        let dark = RenderTheme::builtin("woof-dark").expect("built in");
        assert_eq!(dark.name, "woof-dark");
        assert_eq!(dark.presentation, RenderTheme::builtin("dark").unwrap().presentation);
    }

    #[test]
    fn woof_light_tc_is_the_woof_look_with_the_tc_ladders_and_isobars() {
        // WOOF owner canary run_303354e41cb59dc8 (Typhoon In-fa, 942 hPa): woof-light's
        // 60 kt wind scale and 960 hPa isobar floor drew the core as one colour.
        let woof = RenderTheme::builtin("woof-light").expect("built in");
        let tc = RenderTheme::builtin("tc").expect("built in");
        let both = RenderTheme::builtin("woof-light-tc").expect("built in");
        assert_eq!(both.name, "woof-light-tc");
        assert_eq!(both.source_label, woof.source_label, "WOOF's labels stay");
        assert_eq!(both.model_label, woof.model_label);
        let wind = "mslp_10m_winds_wind_speed";
        assert_eq!(both.levels.get(wind), tc.levels.get(wind), "tc's wind band edges");
        assert_eq!(both.products.get(wind), tc.products.get(wind), "tc's wind colours");
        let edges = both.levels.get(wind).expect("wind edges");
        assert!(edges.last().copied().unwrap_or(0.0) >= 180.0, "the ladder reaches 180 kt");
        for kt in [34.0, 50.0, 64.0, 83.0, 96.0, 113.0, 137.0] {
            assert!(edges.contains(&kt), "a band edge on {kt} kt");
        }
        assert!(both.product_has_banded_scale(&ProductKey::Named(wind.to_string())));
        assert_eq!(both.contour_levels("mslp").and_then(|l| l.first()).copied(), Some(880.0));
        assert_eq!(both.contour_label_every("mslp"), Some(2), "labelled every 8 hPa");
        assert!(RenderTheme::builtin_names().contains(&"woof-light-tc"));
        let file = RenderThemeFile::from_json(r##"{"extends":"woof-light-tc","text":{"source_label":"Recast WOOF · from GFS"}}"##)
            .unwrap();
        let child = RenderTheme::from_file_spec(&file, None).expect("a WOOF run's theme file extends it");
        assert_eq!(child.levels.get(wind), tc.levels.get(wind));
        assert_eq!(child.source_label.as_deref(), Some("Recast WOOF · from GFS"));
    }

    #[test]
    fn the_generic_built_ins_draw_the_derived_labels() {
        for name in ["default", "light", "dark"] {
            let theme = RenderTheme::builtin(name).expect("built in");
            assert_eq!(theme.source_label, None, "{name}");
            assert_eq!(theme.model_label, None, "{name}");
            assert_eq!(
                theme.source_subtitle(Some("source: ArWen".to_string())).as_deref(),
                Some("source: ArWen"),
                "{name}"
            );
            assert_eq!(theme.model_name("WRF"), "WRF", "{name}");
        }
    }

    #[test]
    fn the_woof_built_ins_name_woof_and_never_the_engine_or_wrf() {
        for name in ["woof-light", "woof-dark"] {
            let theme = RenderTheme::builtin(name).expect("built in");
            let source = theme
                .source_subtitle(Some("source: ArWen 2.8.0".to_string()))
                .expect("a labelled product keeps a label");
            let model = theme.model_name("WRF");
            for text in [&source, &model] {
                assert!(text.contains("WOOF"), "{name}: {text}");
                assert!(!text.contains("ArWen") && !text.contains("WRF"), "{name}: {text}");
            }
            assert_eq!(theme.source_subtitle(None), None, "{name}: a bare panel stays bare");
        }
    }

    #[test]
    fn a_model_label_replaces_the_derived_model_name_and_a_blank_one_is_refused() {
        let _version_guard = VERSION_TEST_LOCK.lock().expect("version test lock");
        let file = RenderThemeFile::from_json(r##"{"text": {"model_label": " MINE {version} "}}"##)
            .expect("parses");
        let theme = RenderTheme::from_file_spec(&file, None).expect("resolves");
        assert!(!theme.is_default(), "a model label is a change to every plot");
        set_template_version(None);
        assert_eq!(theme.model_name("WRF"), "MINE");
        let blank = RenderThemeFile::from_json(r##"{"text": {"model_label": " "}}"##)
            .expect("parses");
        let err = RenderTheme::from_file_spec(&blank, None).expect_err("blank label");
        assert!(err.contains("text.model_label"), "{err}");
        let typo = RenderThemeFile::from_json(r##"{"text": {"model_label": "M {verison}"}}"##)
            .expect("parses");
        let err = RenderTheme::from_file_spec(&typo, None).expect_err("unknown token");
        assert!(err.contains("text.model_label"), "{err}");
        let child = RenderThemeFile::from_json(
            r##"{"extends": "woof-light", "text": {"model_label": "OTHER"}}"##,
        )
        .expect("parses")
        .with_parent(None, 0)
        .expect("extends woof-light");
        let child = RenderTheme::from_file_spec(&child, None).expect("resolves");
        assert_eq!(child.model_name("WRF"), "OTHER");
        assert_eq!(
            child.source_label,
            RenderTheme::builtin("woof-light").unwrap().source_label,
            "the parent's source label carries over"
        );
    }


    #[test]
    fn a_site_theme_file_inherits_woof_labels_through_relative_parents() {
        let root = std::env::temp_dir().join(format!("rustwx-theme-inheritance-{}", std::process::id()));
        let nested = root.join("nested");
        std::fs::create_dir_all(&nested).expect("own theme scratch");
        let parent = root.join("parent.json");
        let middle = nested.join("middle.json");
        let child = nested.join("site.json");
        std::fs::write(&parent, r##"{"extends":"woof-light","surface":{"ocean":"#010203"},"fonts":{"regular":"assets/regular.ttf","bold":"assets/bold.ttf"},"footer":{"logo":"assets/logo.png"}}"##)
            .expect("parent theme");
        std::fs::write(&middle, r##"{"extends":"../parent.json","text":{"title_size":1.1}}"##)
            .expect("middle theme");
        std::fs::write(&child, r##"{"extends":"middle.json","name":"site"}"##)
            .expect("site theme");
        let theme = RenderTheme::from_path(&child).expect("site inheritance is accepted");
        assert_eq!(theme.name, "site");
        assert_eq!(theme.presentation.ocean, Some(Rgba::new(1, 2, 3)));
        assert_eq!(theme.presentation.title_size_permille, 1100);
        assert_eq!(theme.presentation.label_size_permille, 1000);
        let asset_root = std::fs::canonicalize(&root).expect("own scratch directory");
        assert_eq!(theme.font_regular, Some(asset_root.join("assets/regular.ttf")));
        assert_eq!(theme.font_bold, Some(asset_root.join("assets/bold.ttf")));
        assert_eq!(theme.footer.as_ref().unwrap().logo, Some(asset_root.join("assets/logo.png")));
        assert_eq!(theme.model_name("WRF"), "WOOF");
        assert_eq!(theme.source_subtitle(Some("source: ArWen 2.8.6".into())).as_deref(), Some("Recast WOOF"));
        assert_eq!(theme.source_subtitle(None), None);
        for path in [child, middle, parent] {
            std::fs::remove_file(path).expect("remove own theme file");
        }
        std::fs::remove_dir(nested).expect("remove own nested scratch");
        std::fs::remove_dir(root).expect("remove own theme scratch");
    }

    #[test]
    fn a_theme_file_parent_cycle_is_refused_with_a_bounded_chain() {
        let root = std::env::temp_dir().join(format!("rustwx-theme-cycle-{}", std::process::id()));
        std::fs::create_dir_all(&root).expect("own theme scratch");
        let first = root.join("first.json");
        let second = root.join("second.json");
        std::fs::write(&first, r##"{"extends":"second.json"}"##).expect("first theme");
        std::fs::write(&second, r##"{"extends":"first.json"}"##).expect("second theme");
        let error = RenderTheme::from_path(&first).expect_err("a parent cycle cannot render");
        assert!(error.contains("deeper than eight themes"), "{error}");
        for path in [first, second] {
            std::fs::remove_file(path).expect("remove own theme file");
        }
        std::fs::remove_dir(root).expect("remove own theme scratch");
    }

    #[test]
    fn both_built_ins_name_no_footer_and_carry_a_mesh_style() {
        for name in RenderTheme::builtin_names() {
            let theme = RenderTheme::builtin(name).expect("built in");
            assert!(
                theme.footer.is_none(),
                "{name} draws a footer strip, which would move every existing render"
            );
            assert!(theme.mesh.draw_edges, "{name} draws no mesh hairline");
            assert!(theme.mesh.edge_width_px() > 0.0, "{name}");
        }
        assert!(
            RenderTheme::default_theme().is_default(),
            "the default theme still changes nothing"
        );
        let dark = RenderTheme::dark_theme();
        assert_eq!(dark.mesh.empty, Rgba::new(0x12, 0x19, 0x1c));
        assert_ne!(
            dark.mesh.edge,
            RenderTheme::default_theme().mesh.edge,
            "the dark surface has its own hairline"
        );
    }

    #[test]
    fn a_footer_section_resolves_its_defaults_and_refuses_an_unknown_field() {
        let file = RenderThemeFile::from_json(
            r##"{"surface": {"canvas": "#000000"},
                 "ink": {"primary": "#e2edf0", "secondary": "#94afb8"},
                 "grid": {"hairline": "#223137"},
                 "footer": {"height": 72, "logo": "assets/mark.png", "logo_height": 24}}"##,
        )
        .expect("parses");
        let theme = RenderTheme::from_file_spec(&file, Some(Path::new("/themes"))).expect("resolves");
        let footer = theme.footer.as_ref().expect("a footer");
        assert_eq!(footer.height, 72);
        assert_eq!(footer.background, Rgba::new(0, 0, 0), "background follows the canvas");
        assert_eq!(footer.rule, Some(Rgba::new(0x22, 0x31, 0x37)));
        assert_eq!(footer.title_ink, Rgba::new(0xe2, 0xed, 0xf0));
        assert_eq!(footer.ink, Rgba::new(0x94, 0xaf, 0xb8));
        assert_eq!(footer.logo, Some(PathBuf::from("/themes").join("assets/mark.png")));
        assert_eq!(footer.title_template, "{product_title}");
        assert!(footer.caption_template.contains("{valid_time}"));
        assert!(!theme.is_default(), "a footer is a change to every plot");

        let bad = RenderThemeFile::from_json(
            r##"{"footer": {"caption": "valid {valid} | {mesh_or_grid}"}}"##,
        )
        .expect("parses");
        let err = RenderTheme::from_file_spec(&bad, None).expect_err("unknown field");
        assert!(err.contains("{valid}"), "{err}");
        assert!(err.contains("valid_time"), "{err}");

        let tall = RenderThemeFile::from_json(r##"{"footer": {"height": 4}}"##).expect("parses");
        let err = RenderTheme::from_file_spec(&tall, None).expect_err("too short");
        assert!(err.contains("footer.height"), "{err}");
    }

    #[test]
    fn a_mesh_section_overrides_only_what_it_names() {
        let file = RenderThemeFile::from_json(
            r##"{"mesh": {"edge": "#223137", "edge_width": 0.5, "empty": "#12191c"}}"##,
        )
        .expect("parses");
        let theme = RenderTheme::from_file_spec(&file, None).expect("resolves");
        assert_eq!(theme.mesh.edge, Rgba::new(0x22, 0x31, 0x37));
        assert_eq!(theme.mesh.empty, Rgba::new(0x12, 0x19, 0x1c));
        assert!((theme.mesh.edge_width_px() - 0.5).abs() < 1.0e-6);
        assert!(theme.mesh.draw_edges, "edges stay on when unnamed");
        let wide = RenderThemeFile::from_json(r##"{"mesh": {"edge_width": 40.0}}"##).expect("parses");
        let err = RenderTheme::from_file_spec(&wide, None).expect_err("too wide");
        assert!(err.contains("mesh.edge_width"), "{err}");
    }

    #[test]
    fn colours_resample_through_anchors_and_products_override_on_the_scale_s_levels() {
        let colours = resample(&[Rgba::new(0, 0, 0), Rgba::new(255, 255, 255)], 3);
        assert_eq!(colours[1], Color::rgba(128, 128, 128, 255));
        let mut theme = RenderTheme::default_theme();
        theme.products.insert(
            "probe".to_string(),
            vec![Rgba::new(0, 0, 0), Rgba::new(255, 255, 255)],
        );
        let scale = ColorScale::Discrete(DiscreteColorScale {
            levels: vec![0.0, 1.0, 2.0, 3.0, 4.0],
            colors: vec![Color::BLACK; 4],
            extend: crate::request::ExtendMode::Neither,
            mask_below: None,
        });
        let overridden = theme
            .product_scale_override(&ProductKey::named("probe"), &scale)
            .expect("named product is overridden");
        let discrete = overridden.resolved_discrete();
        assert_eq!(discrete.levels.len(), 5);
        assert_eq!(discrete.colors.len(), 4);
        assert_eq!(discrete.colors[3], Color::rgba(255, 255, 255, 255));
        assert!(
            theme
                .product_scale_override(&ProductKey::named("other"), &scale)
                .is_none()
        );
    }

    #[test]
    fn halo_and_ink_substitution_only_touch_white_and_dark_requests() {
        let dark = RenderTheme::builtin("dark").expect("built in").presentation;
        assert_eq!(
            dark.substitute_white_halo(Rgba::with_alpha(255, 255, 255, 235)),
            Rgba::with_alpha(0, 0, 0, 235)
        );
        assert_eq!(
            dark.substitute_white_halo(Rgba::with_alpha(200, 20, 20, 235)),
            Rgba::with_alpha(200, 20, 20, 235)
        );
        assert_ne!(dark.substitute_dark_ink(Rgba::with_alpha(0, 0, 0, 220)), Rgba::with_alpha(0, 0, 0, 220));
        assert_eq!(
            dark.substitute_dark_ink(Rgba::new(24, 84, 168)),
            Rgba::new(24, 84, 168)
        );
        let plain = PresentationTheme::default();
        assert_eq!(plain.substitute_white_halo(Rgba::WHITE), Rgba::WHITE);
        assert_eq!(plain.substitute_dark_ink(Rgba::BLACK), Rgba::BLACK);
    }
}

/// Concentration colour scales, not computed AQI. EPA Table 5 uses daily
/// PM2.5 and 8-hour ozone averages; these products show instantaneous fields.
/// https://document.airnow.gov/technical-assistance-document-for-the-reporting-of-daily-air-quailty.pdf
/// (May 2026, Table 5; PM2.5 breakpoints revised in 2024).
/// Ozone ends its 8-hour scale at 200 ppb. No 1-hour band is mixed into it.
pub fn air_quality_scale(ozone: bool) -> DiscreteColorScale {
    DiscreteColorScale {
        levels: if ozone { vec![0.0, 54.0, 70.0, 85.0, 105.0, 200.0] }
                else { vec![0.0, 9.0, 35.4, 55.4, 125.4, 225.4, 325.4] },
        colors: { let mut colors = vec![Color::rgba(0, 228, 0, 255), Color::rgba(255, 255, 0, 255),
                     Color::rgba(255, 126, 0, 255), Color::rgba(255, 0, 0, 255),
                     Color::rgba(143, 63, 151, 255), Color::rgba(126, 0, 35, 255)];
                  if ozone { colors.pop(); }
                  colors },
        extend: crate::request::ExtendMode::Max,
        mask_below: None,
    }
}

/// AOD rungs separate clear air, haze and thick plumes; the extension retains
/// values above 2. Dust uses a broad mass ladder for weak through dense plumes.
/// Both use the renderer's existing sequential teal ramp.
pub fn aerosol_sequential_scale(aod: bool) -> DiscreteColorScale {
    DiscreteColorScale {
        levels: if aod { vec![0.0, 0.1, 0.2, 0.5, 1.0, 2.0] }
                else { vec![0.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0] },
        colors: vec![Color::rgba(235, 247, 247, 255), Color::rgba(195, 226, 225, 255),
                     Color::rgba(125, 202, 202, 255), Color::rgba(70, 159, 160, 255),
                     Color::rgba(0, 95, 96, 255), Color::rgba(0, 58, 60, 255)],
        extend: crate::request::ExtendMode::Max,
        mask_below: None,
    }
}

#[cfg(test)]
mod explicit_level_tests {
    use super::*;
    #[test]
    fn levels_are_opt_in_and_leave_other_products_untouched() {
        let spec = RenderThemeFile::from_json(r#"{"colormaps":{"levels":{"wind":[0,40,80,120,160]}}}"#).unwrap();
        let theme = RenderTheme::from_file_spec(&spec, None).unwrap();
        let original = ColorScale::Discrete(DiscreteColorScale {
            levels: vec![0., 30., 60.], colors: vec![Color::BLACK, Color::WHITE],
            extend: crate::request::ExtendMode::Max, mask_below: None,
        });
        let changed = theme.product_scale_override(&ProductKey::named("wind"), &original).unwrap().resolved_discrete();
        assert_eq!(changed.levels, vec![0.,40.,80.,120.,160.]);
        assert_eq!(changed.colors, original.resolved_discrete().colors);
        assert!(theme.product_scale_override(&ProductKey::named("rain"), &original).is_none());
        assert!(RenderTheme::default_theme().product_scale_override(&ProductKey::named("wind"), &original).is_none());
        assert!(!theme.is_default());
    }
    #[test]
    fn unordered_or_empty_boundaries_refuse() {
        for levels in ["[]", "[1]", "[0,0]", "[2,1]"] {
            let spec = RenderThemeFile::from_json(&format!("{{\"colormaps\":{{\"levels\":{{\"wind\":{levels}}}}}}}")).unwrap();
            assert!(RenderTheme::from_file_spec(&spec, None).is_err());
        }
    }
}
