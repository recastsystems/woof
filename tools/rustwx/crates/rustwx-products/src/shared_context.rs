use chrono::{Duration, NaiveDate};
use image::DynamicImage;
use rustwx_core::{Field2D, LatLonGrid, ModelId, ProductKey, SourceId};
pub use rustwx_render::ProjectedMap;
use rustwx_render::{
    ChromeScale, Color, DomainFrame, MapRenderRequest, PanelGridLayout, PanelPadding,
    ProductVisualMode, ProjectedDomain, WeatherProduct, draw_centered_text_line,
    map_frame_aspect_ratio_for_mode, render_panel_grid,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DomainSpec {
    pub slug: String,
    pub bounds: (f64, f64, f64, f64),
}

impl DomainSpec {
    pub fn new<S: Into<String>>(slug: S, bounds: (f64, f64, f64, f64)) -> Self {
        Self {
            slug: slug.into(),
            bounds,
        }
    }
}

/// Where a batch's frames came from, as far as the plot headline is
/// concerned.
///
/// A batch fetched from a model's registered source catalog may stamp that
/// catalog's dataset token into the title, because for those frames the
/// token is the only thing that names which archive the numbers came from.
/// A batch imported from local model output has no such archive: stamping
/// one is a lie, and it is the lie a viewer screenshots.  What that viewer
/// needs instead is which grid the frame is on -- something only the
/// importer, which read the file's own grid attributes, can supply.
///
/// The variants are therefore not a style choice; they are two different
/// factual claims about the same parenthetical.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TitleProvenance {
    /// Fetched from the model's registered source catalog.  Title
    /// composition is unchanged: whichever dataset token the lane already
    /// derived still applies.
    #[default]
    SourceCatalog,
    /// Imported from local model output.  No catalog dataset token is ever
    /// stamped.  `grid_label` is the headline parenthetical when the
    /// importer could read a grid identity (`d02 750 m`); when it could
    /// not, the headline simply carries no parenthetical, which is accurate
    /// where a borrowed dataset token would not be.
    LocalImport {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        grid_label: Option<String>,
    },
}

impl TitleProvenance {
    /// A locally imported batch, whose titles must never carry a source
    /// catalog's dataset token.
    pub fn is_local_import(&self) -> bool {
        matches!(self, Self::LocalImport { .. })
    }

    /// The headline parenthetical this provenance contributes, if any.
    /// Blank labels are treated as absent so a caller that plumbs an empty
    /// string cannot produce `Title ()`.
    pub fn title_parenthetical(&self) -> Option<&str> {
        match self {
            Self::SourceCatalog => None,
            Self::LocalImport { grid_label } => grid_label
                .as_deref()
                .map(str::trim)
                .filter(|label| !label.is_empty()),
        }
    }

    /// `base` with this provenance's parenthetical appended, when it has
    /// one.  The shared entry point for every lane, so one grid identity
    /// cannot be spelled three ways.
    pub(crate) fn apply_to_title(&self, base: impl Into<String>) -> String {
        let base = base.into();
        match self.title_parenthetical() {
            Some(label) => format!("{base} ({label})"),
            None => base,
        }
    }
}

/// The run's initial-condition disclosure, when its initial state was
/// itself a forecast rather than an analysis.
///
/// A process-wide value on purpose.  `rw_wrfbatch` renders exactly one
/// forecast run per invocation -- the timeline planner refuses sources
/// that disagree on their reference time, so "which run" is a constant
/// for the lifetime of the process -- and every one of the nine product
/// lanes that builds a subtitle must disclose it identically.  Threading
/// a parameter through all nine would let one lane forget, and "one
/// product did not disclose" is precisely the finding (C-03) this
/// closes: a 174-hour-lead run and a genuine analysis run were
/// indistinguishable in all 159 rendered PNGs.  Set once, honoured at
/// the single place the `Init` token is spelled.
static INITIAL_CONDITION_DISCLOSURE: std::sync::RwLock<Option<String>> =
    std::sync::RwLock::new(None);

/// Record what this run was initialized from, or clear it with `None`.
///
/// The caller supplies the already-composed parenthetical body (e.g.
/// `GFS 08/01 00Z f174`); this module owns only where it appears.  A
/// blank string clears rather than rendering `Init 08/08 06Z ()`.
pub fn set_initial_condition_disclosure(disclosure: Option<String>) {
    let value = disclosure
        .map(|text| text.trim().to_string())
        .filter(|text| !text.is_empty());
    if let Ok(mut slot) = INITIAL_CONDITION_DISCLOSURE.write() {
        *slot = value;
    }
}

/// The disclosure in force, if any.
pub fn initial_condition_disclosure() -> Option<String> {
    INITIAL_CONDITION_DISCLOSURE
        .read()
        .ok()
        .and_then(|slot| slot.clone())
}

/// The global attribute in which a wrfout-shaped file names the model
/// that produced it, for the model token of a plot's metadata row.
///
/// A wrfout-shaped file imports under the store identity `wrf`, so every
/// one of them was labelled `WRF`, including files no WRF model wrote: a
/// variable-resolution mesh forecast resampled onto a regular grid
/// (`rw_mpas_convert`) and a global spectral model's export tape.  Their
/// maps said `WRF` beside a source label naming the real model.  The
/// producer knows what it is, so it says so in the file, and the renderer
/// prints that name instead.  A file without the attribute (stock WRF, a
/// regional run, every file written before it existed) renders exactly as
/// before.
pub const MODEL_LABEL_ATTRIBUTE: &str = "GPUWM_MODEL_LABEL";

/// The global attribute every history file the WOOF engine writes carries
/// (`gpuwm.io.wrfout.WrfoutWriter`, both of its writers, since 2.7): the
/// release that wrote it.  Stock WRF never writes it, so its presence is
/// the marker that a wrfout-shaped file is the engine's own output.
pub const ENGINE_VERSION_ATTRIBUTE: &str = "GPUWM_VERSION";

/// The model token of a file the engine wrote and that names no model of
/// its own: the engine's public name.  Before this, such a file carried
/// no [`MODEL_LABEL_ATTRIBUTE`] and every WOOF map said `WRF` unless a
/// WOOF theme renamed it.
pub const ENGINE_MODEL_LABEL: &str = "WOOF";

/// The model label of one wrfout-shaped file: the label it names
/// ([`MODEL_LABEL_ATTRIBUTE`]) when it names one, otherwise the engine's
/// public name when the engine wrote it ([`ENGINE_VERSION_ATTRIBUTE`]),
/// otherwise `None`, which keeps the store identity (`WRF`) for stock WRF
/// output.  `Err` names a label the row cannot take.
pub fn file_model_label(
    named: Option<&str>,
    engine_written: bool,
) -> Result<Option<String>, String> {
    if let Some(label) = named.map(checked_model_label).transpose()?.flatten() {
        return Ok(Some(label));
    }
    Ok(engine_written.then(|| ENGINE_MODEL_LABEL.to_string()))
}

/// The longest model label the metadata row takes, in characters.  The
/// row already carries the init time, lead, valid time and grid spacing,
/// and a longer name would push those out of a narrow frame.
pub const MODEL_LABEL_MAX_CHARS: usize = 32;

static MODEL_LABEL: std::sync::RwLock<Option<String>> = std::sync::RwLock::new(None);

/// A file's model label as the metadata row may print it, or the reason
/// it may not.  Runs of whitespace collapse to one space; a label that is
/// blank, too long, holds a control character or holds `|` (the row's own
/// separator, which would split one name into two tokens) is refused.
pub fn checked_model_label(raw: &str) -> Result<Option<String>, String> {
    let label = raw.split_whitespace().collect::<Vec<_>>().join(" ");
    if label.is_empty() {
        return Ok(None);
    }
    if label.chars().count() > MODEL_LABEL_MAX_CHARS {
        return Err(format!(
            "{MODEL_LABEL_ATTRIBUTE} {label:?} is longer than {MODEL_LABEL_MAX_CHARS} characters"
        ));
    }
    if label.chars().any(|ch| ch.is_control() || ch == '|') {
        return Err(format!(
            "{MODEL_LABEL_ATTRIBUTE} {label:?} holds a control character or '|', the metadata row's separator"
        ));
    }
    Ok(Some(label))
}

/// The one label a set of inputs agrees on.  Several inputs render as one
/// run, so a name only some of them carry, or two different names, would
/// be a claim about the whole run that is false for part of it: such a
/// set keeps the store identity, the same answer a mixed set gets for its
/// domain token.
pub fn agreed_model_label(labels: &[Option<String>]) -> Option<String> {
    let first = labels.first()?.clone()?;
    labels
        .iter()
        .all(|label| label.as_deref() == Some(first.as_str()))
        .then_some(first)
}

/// Record the model label of the run being drawn, or clear it with `None`.
/// Process-wide for the reason the disclosure above is: one invocation
/// renders one run, and every product lane spells the metadata row at the
/// one place the model token is spelled ([`model_token`]).
pub fn set_model_label(label: Option<String>) {
    let value = label.and_then(|text| checked_model_label(&text).ok().flatten());
    if let Ok(mut slot) = MODEL_LABEL.write() {
        *slot = value;
    }
}

/// The model label in force, if any.
pub fn model_label() -> Option<String> {
    MODEL_LABEL.read().ok().and_then(|slot| slot.clone())
}

/// The model token of the metadata row: the run's own model label for a
/// wrfout-shaped import that carries one, otherwise the store identity in
/// capitals (`WRF`, `HRRR`).
pub fn model_token(model: ModelId) -> String {
    model_token_with(model, model_label().as_deref())
}

/// [`model_token`] over an explicit label.  Only the generic wrfout
/// identity is replaced: a label can only come from a wrfout-shaped file,
/// and a GRIB model drawn later in the same process keeps its own name.
pub fn model_token_with(model: ModelId, label: Option<&str>) -> String {
    match label {
        Some(label) if model == ModelId::WrfGdex => label.to_string(),
        _ => model.to_string().to_ascii_uppercase(),
    }
}

/// A command line's answer on wind streamlines, when it gave one.
///
/// `None` means "nobody asked", which leaves `RUSTWX_WIND_STREAMLINES`
/// and then the automatic per-grid choice in charge. Process-wide for the
/// same reason the disclosure above is: one invocation renders one run,
/// and the wind layer is drawn from several product lanes that must all
/// answer the request identically.
///
/// The concrete breakage this closes: streamlines had NO front door at
/// all. Drawing them was reachable only by setting an environment
/// variable whose name appears in no help text and no document, which
/// under the ship-only-what-users-can-reach rule means the capability was
/// not shipped.
static WIND_STREAMLINE_REQUEST: std::sync::RwLock<Option<bool>> = std::sync::RwLock::new(None);

/// Ask for wind streamlines (`Some(true)`), ask for barbs instead
/// (`Some(false)`), or leave the choice automatic (`None`).
pub fn set_wind_streamline_request(request: Option<bool>) {
    if let Ok(mut slot) = WIND_STREAMLINE_REQUEST.write() {
        *slot = request;
    }
}

/// The streamline request in force, if a caller made one.
pub fn wind_streamline_request() -> Option<bool> {
    WIND_STREAMLINE_REQUEST.read().ok().and_then(|slot| *slot)
}

/// The environment variable that names a run's frame interval in minutes.
///
/// A caller drawing the frames of a sub-hourly run one call at a time (a
/// live pass draws each frame as it lands) sets it, so a whole-hour frame
/// knows it belongs to a run whose other frames carry minutes.
pub const FRAME_MINUTES_ENV: &str = "RUSTWX_FRAME_MINUTES";

/// A caller's explicit minute-clock choice; `None` leaves it to
/// [`FRAME_MINUTES_ENV`].
static MINUTE_CLOCK_OVERRIDE: std::sync::RwLock<Option<bool>> = std::sync::RwLock::new(None);

/// Force the minute clock on or off, or hand it back to the environment.
pub fn set_minute_clock(clock: Option<bool>) {
    if let Ok(mut slot) = MINUTE_CLOCK_OVERRIDE.write() {
        *slot = clock;
    }
}

/// Whether a whole-hour frame writes its time to the minute (`F001:00 |
/// Valid 12/18 13:00Z`) instead of to the hour (`F001 | Valid 12/18 13Z`).
///
/// THE BREAKAGE IT PREVENTS (WOOF, 2026-11-10: on the animated loops the
/// title still jittered). Frames between hours are on the exact-time axis
/// and read `F001:15 | Valid 12/18 13:15Z`; a whole-hour frame of the same
/// run, drawn in a call of its own, read `F001 | Valid 12/18 13Z`. The
/// header of a 15-minute loop changed length four times an hour, so every
/// word after the time, and on a narrow frame the whole header, slid
/// sideways. With the clock on, every frame of the run has a header of the
/// same length and only the time's digits change.
///
/// On when a caller says so ([`set_minute_clock`]), else when
/// [`FRAME_MINUTES_ENV`] names an interval that is not a whole number of
/// hours. Hourly runs are unchanged.
pub fn minute_clock() -> bool {
    if let Some(clock) = MINUTE_CLOCK_OVERRIDE.read().ok().and_then(|slot| *slot) {
        return clock;
    }
    static FROM_ENV: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FROM_ENV.get_or_init(|| {
        frame_minutes_need_minute_clock(std::env::var(FRAME_MINUTES_ENV).ok().as_deref())
    })
}

/// Whether a frame interval (minutes, as text) is sub-hourly or off the hour.
fn frame_minutes_need_minute_clock(minutes: Option<&str>) -> bool {
    minutes
        .and_then(|text| text.trim().parse::<u32>().ok())
        .is_some_and(|minutes| minutes > 0 && minutes % 60 != 0)
}

/// The whole-hour lead token: `F001`, or `F001:00` on the minute clock.
fn whole_hour_lead_label(forecast_hour: u16, minute_clock: bool) -> String {
    if minute_clock {
        format!("F{forecast_hour:03}:00")
    } else {
        format!("F{forecast_hour:03}")
    }
}

pub fn model_time_subtitle(
    model: ModelId,
    date_yyyymmdd: &str,
    cycle_utc: u8,
    forecast_hour: u16,
) -> String {
    model_time_subtitle_clocked(model, date_yyyymmdd, cycle_utc, forecast_hour, minute_clock())
}

/// [`model_time_subtitle`] with the minute clock given rather than read.
pub fn model_time_subtitle_clocked(
    model: ModelId,
    date_yyyymmdd: &str,
    cycle_utc: u8,
    forecast_hour: u16,
    minute_clock: bool,
) -> String {
    model_time_subtitle_with_lead_label_clocked(
        model,
        date_yyyymmdd,
        cycle_utc,
        forecast_hour,
        whole_hour_lead_label(forecast_hour, minute_clock),
        minute_clock,
    )
}

pub fn model_time_subtitle_with_lead_label<S: AsRef<str>>(
    model: ModelId,
    date_yyyymmdd: &str,
    cycle_utc: u8,
    forecast_hour: u16,
    lead_label: S,
) -> String {
    model_time_subtitle_with_lead_label_clocked(
        model,
        date_yyyymmdd,
        cycle_utc,
        forecast_hour,
        lead_label,
        minute_clock(),
    )
}

/// [`model_time_subtitle_with_lead_label`] with the minute clock given: on,
/// the valid time carries its minutes (`Valid 12/18 13:00Z`); the lead label
/// is the caller's, a window (`F000-F001`) included.
pub fn model_time_subtitle_with_lead_label_clocked<S: AsRef<str>>(
    model: ModelId,
    date_yyyymmdd: &str,
    cycle_utc: u8,
    forecast_hour: u16,
    lead_label: S,
    minute_clock: bool,
) -> String {
    let valid = valid_time_label(date_yyyymmdd, cycle_utc, forecast_hour, minute_clock)
        .unwrap_or_else(|| "unknown".to_string());
    let init = init_date_label(date_yyyymmdd).unwrap_or_else(|| date_yyyymmdd.to_string());
    // The `Init` token names when THIS model run started; `F` counts hours
    // into it.  Neither changes here.  What is added is the second, different
    // time a viewer cannot otherwise recover: what that initial state itself
    // was.  Absent a disclosure the subtitle is byte-identical to before, so
    // an analysis run's plots do not move.
    let disclosure = match initial_condition_disclosure() {
        Some(text) => format!(" ({text})"),
        None => String::new(),
    };
    format!(
        "Init {} {:02}Z{} | {} | Valid {} | {}",
        init,
        cycle_utc,
        disclosure,
        lead_label.as_ref(),
        valid,
        // A theme may name the model (`text.model_label`); unset keeps the
        // run's model token, including a wrfout import's own label.
        rustwx_render::theme::active_theme().model_name(&model_token(model))
    )
}

pub fn source_subtitle(source: SourceId) -> String {
    format!("source: {}", source.as_str())
}

fn valid_time_label(
    date_yyyymmdd: &str,
    cycle_utc: u8,
    forecast_hour: u16,
    minute_clock: bool,
) -> Option<String> {
    let date = NaiveDate::parse_from_str(date_yyyymmdd, "%Y%m%d").ok()?;
    let cycle_time = date.and_hms_opt(u32::from(cycle_utc), 0, 0)?;
    let valid_time = cycle_time + Duration::hours(i64::from(forecast_hour));
    let format = if minute_clock { "%m/%d %H:%MZ" } else { "%m/%d %HZ" };
    Some(valid_time.format(format).to_string())
}

fn init_date_label(date_yyyymmdd: &str) -> Option<String> {
    let date = NaiveDate::parse_from_str(date_yyyymmdd, "%Y%m%d").ok()?;
    Some(date.format("%m/%d").to_string())
}

#[derive(Debug, Clone, Default)]
pub struct PreparedProjectedContext {
    projected_maps: HashMap<(u32, u32), ProjectedMap>,
}

pub trait ProjectedMapProvider: Sync {
    fn projected_map(&self, width: u32, height: u32) -> Option<&ProjectedMap>;
}

impl PreparedProjectedContext {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn projected_map(&self, width: u32, height: u32) -> Option<&ProjectedMap> {
        self.projected_maps.get(&(width, height))
    }

    pub fn insert(&mut self, width: u32, height: u32, projected: ProjectedMap) {
        self.projected_maps.insert((width, height), projected);
    }

    pub fn contains_size(&self, width: u32, height: u32) -> bool {
        self.projected_maps.contains_key(&(width, height))
    }
}

impl ProjectedMapProvider for PreparedProjectedContext {
    fn projected_map(&self, width: u32, height: u32) -> Option<&ProjectedMap> {
        self.projected_map(width, height)
    }
}

#[derive(Debug, Clone)]
pub struct WeatherPanelField {
    pub product: WeatherProduct,
    pub artifact_slug: Option<String>,
    pub title_override: Option<String>,
    pub units: String,
    pub values: Vec<f64>,
}

impl WeatherPanelField {
    pub fn new<S: Into<String>>(product: WeatherProduct, units: S, values: Vec<f64>) -> Self {
        Self {
            product,
            artifact_slug: None,
            title_override: None,
            units: units.into(),
            values,
        }
    }

    pub fn with_artifact_slug<S: Into<String>>(mut self, slug: S) -> Self {
        self.artifact_slug = Some(slug.into());
        self
    }

    pub fn with_title_override<S: Into<String>>(mut self, title: S) -> Self {
        self.title_override = Some(title.into());
        self
    }

    pub fn artifact_slug(&self) -> &str {
        self.artifact_slug
            .as_deref()
            .unwrap_or_else(|| self.product.slug())
    }

    pub fn display_title(&self) -> &str {
        self.title_override
            .as_deref()
            .unwrap_or_else(|| self.product.display_title())
    }
}

#[derive(Debug, Clone, Default)]
pub struct WeatherPanelHeader {
    pub title: String,
    pub subtitle_lines: Vec<String>,
}

impl WeatherPanelHeader {
    pub fn new<S: Into<String>>(title: S) -> Self {
        Self {
            title: title.into(),
            subtitle_lines: Vec::new(),
        }
    }

    pub fn with_subtitle_line<S: Into<String>>(mut self, line: S) -> Self {
        self.subtitle_lines.push(line.into());
        self
    }
}

#[derive(Debug, Clone, Copy)]
pub struct WeatherPanelLayout {
    pub panel_width: u32,
    pub panel_height: u32,
    pub top_padding: u32,
}

impl Default for WeatherPanelLayout {
    fn default() -> Self {
        Self {
            panel_width: 700,
            panel_height: 520,
            top_padding: 70,
        }
    }
}

impl WeatherPanelLayout {
    pub fn target_aspect_ratio(self) -> f64 {
        map_frame_aspect_ratio_for_mode(
            ProductVisualMode::PanelMember,
            self.panel_width,
            self.panel_height,
            true,
            true,
        )
    }
}

pub fn layout_key(layout: WeatherPanelLayout) -> (u32, u32, u32) {
    (layout.panel_width, layout.panel_height, layout.top_padding)
}

pub(crate) fn static_supersample_factor() -> u32 {
    std::env::var("RUSTWX_SUPERSAMPLE_FACTOR")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(1)
}

pub(crate) fn static_supersample_sharpen() -> bool {
    std::env::var("RUSTWX_SUPERSAMPLE_SHARPEN")
        .ok()
        .and_then(|value| match value.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => Some(true),
            "0" | "false" | "no" | "off" => Some(false),
            _ => None,
        })
        .unwrap_or(false)
}

pub fn static_chrome_scale() -> ChromeScale {
    let scale = std::env::var("RUSTWX_CHROME_SCALE")
        .ok()
        .and_then(|value| value.parse::<f32>().ok())
        .unwrap_or(0.9)
        .clamp(0.75, 2.0);
    ChromeScale::Fixed(scale)
}

pub(crate) fn static_title_with_suffix(title: impl Into<String>) -> String {
    let mut title = title.into();
    let Some(suffix) = std::env::var("RUSTWX_TITLE_SUFFIX")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
    else {
        return title;
    };
    title.push_str(" (");
    title.push_str(&suffix);
    title.push(')');
    title
}

pub fn build_weather_map_request(
    grid: &LatLonGrid,
    projected: &ProjectedMap,
    field_spec: &WeatherPanelField,
    width: u32,
    height: u32,
    subtitle_left: Option<String>,
    subtitle_right: Option<String>,
) -> Result<MapRenderRequest, Box<dyn std::error::Error>> {
    let field = Field2D::new(
        ProductKey::named(field_spec.product.slug()),
        field_spec.units.clone(),
        grid.clone(),
        field_spec.values.iter().map(|&v| v as f32).collect(),
    )?;
    let mut request = MapRenderRequest::for_core_weather_product(field, field_spec.product);
    request.width = width;
    request.height = height;
    request.supersample_factor = static_supersample_factor();
    request.supersample_sharpen = static_supersample_sharpen();
    request.domain_frame = Some(DomainFrame::map_viewport_default());
    request.visual_mode = ProductVisualMode::SevereDiagnostic;
    request.title = Some(field_spec.display_title().to_string());
    request.subtitle_left = subtitle_left;
    request.subtitle_right = subtitle_right;
    request.projected_domain = Some(ProjectedDomain {
        x: projected.projected_x.clone(),
        y: projected.projected_y.clone(),
        extent: projected.extent.clone(),
    });
    request.projected_lines = projected.lines.clone();
    request.projected_polygons = projected.polygons.clone();
    Ok(request)
}

pub fn render_two_by_four_weather_panel(
    output_path: &Path,
    grid: &LatLonGrid,
    projected: &ProjectedMap,
    fields: &[WeatherPanelField],
    header: &WeatherPanelHeader,
    layout: WeatherPanelLayout,
) -> Result<(), Box<dyn std::error::Error>> {
    let grid_layout = PanelGridLayout::two_by_four(layout.panel_width, layout.panel_height)?
        .with_padding(PanelPadding {
            top: layout.top_padding,
            ..Default::default()
        });
    let mut requests = Vec::with_capacity(fields.len());

    for field_spec in fields {
        let field = Field2D::new(
            ProductKey::named(field_spec.product.slug()),
            field_spec.units.clone(),
            grid.clone(),
            field_spec.values.iter().map(|&v| v as f32).collect(),
        )?;
        let mut request = MapRenderRequest::for_core_weather_product(field, field_spec.product);
        request.width = layout.panel_width;
        request.height = layout.panel_height;
        request.visual_mode = ProductVisualMode::PanelMember;
        if let Some(title) = &field_spec.title_override {
            request.title = Some(title.clone());
        }
        request.projected_domain = Some(ProjectedDomain {
            x: projected.projected_x.clone(),
            y: projected.projected_y.clone(),
            extent: projected.extent.clone(),
        });
        request.projected_lines = projected.lines.clone();
        request.projected_polygons = projected.polygons.clone();
        requests.push(request);
    }

    let mut canvas = render_panel_grid(&grid_layout, &requests)?;
    draw_centered_text_line(&mut canvas, &header.title, 10, Color::BLACK, 2);
    for (idx, line) in header.subtitle_lines.iter().enumerate() {
        draw_centered_text_line(&mut canvas, line, 35 + (idx as i32 * 20), Color::BLACK, 1);
    }

    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)?;
    }
    DynamicImage::ImageRgba8(canvas).save(output_path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustwx_render::{ProjectedExtent, ProjectedLineOverlay, ProjectedPolygonFill};

    /// Serializes the tests that drive the initial-condition disclosure.
    ///
    /// The disclosure is process-global, and `cargo test` runs a binary's
    /// tests on a thread pool.  Three tests below set it, assert against
    /// it and set it back, so without this lock one test's reset lands
    /// between another's set and its assertion.  That is a real race and
    /// it fired: `a_forecast_initialized_run_discloses_its_source_cycle_and_lead`
    /// failed roughly one run in five, reading the subtitle its
    /// neighbour's `set_initial_condition_disclosure(None)` had just
    /// produced.  Nothing about the product is wrong; the tests were
    /// sharing a global without saying so.
    ///
    /// Poisoning is ignored deliberately: if one of these tests panics
    /// while holding the lock, the others should still report their own
    /// verdicts rather than a cascade of `PoisonError`s that hides which
    /// assertion actually failed.
    static DISCLOSURE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn disclosure_guard() -> std::sync::MutexGuard<'static, ()> {
        DISCLOSURE_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    #[test]
    fn projected_context_tracks_sizes() {
        let mut context = PreparedProjectedContext::new();
        assert!(!context.contains_size(700, 520));
        context.insert(
            700,
            520,
            ProjectedMap {
                projected_x: vec![0.0],
                projected_y: vec![0.0],
                extent: ProjectedExtent {
                    x_min: 0.0,
                    x_max: 1.0,
                    y_min: 0.0,
                    y_max: 1.0,
                },
                lines: Vec::<ProjectedLineOverlay>::new(),
                polygons: Vec::<ProjectedPolygonFill>::new(),
                inverse_raster_projection: None,
            },
        );
        assert!(context.contains_size(700, 520));
        assert!(context.projected_map(700, 520).is_some());
    }

    #[test]
    fn panel_field_keeps_title_override() {
        let field = WeatherPanelField::new(WeatherProduct::StpFixed, "dimensionless", vec![1.0])
            .with_title_override("STP (FIXED)");
        assert_eq!(field.title_override.as_deref(), Some("STP (FIXED)"));
    }

    #[test]
    fn panel_field_keeps_artifact_slug_override() {
        let field = WeatherPanelField::new(WeatherProduct::Scp, "dimensionless", vec![1.0])
            .with_artifact_slug("scp_mu_0_3km_0_6km_proxy");
        assert_eq!(field.artifact_slug(), "scp_mu_0_3km_0_6km_proxy");
    }

    #[test]
    fn model_time_subtitle_includes_init_lead_and_valid_time() {
        let _guard = disclosure_guard();
        set_initial_condition_disclosure(None);
        assert_eq!(
            model_time_subtitle(ModelId::Gfs, "20260424", 22, 4),
            "Init 04/24 22Z | F004 | Valid 04/25 02Z | GFS"
        );
    }

    /// A whole-hour frame of a sub-hourly run reads like its neighbours
    /// between hours (`F001:15 | Valid 12/18 13:15Z`): same tokens, same
    /// length, only the digits differ.
    #[test]
    fn the_minute_clock_writes_whole_hours_to_the_minute() {
        let _guard = disclosure_guard();
        set_initial_condition_disclosure(None);
        assert_eq!(
            model_time_subtitle_clocked(ModelId::Gfs, "19571218", 12, 1, true),
            "Init 12/18 12Z | F001:00 | Valid 12/18 13:00Z | GFS"
        );
        assert_eq!(
            model_time_subtitle_clocked(ModelId::Gfs, "19571218", 12, 1, false),
            "Init 12/18 12Z | F001 | Valid 12/18 13Z | GFS"
        );
        // A window label is the caller's; only the valid time gains minutes.
        assert_eq!(
            model_time_subtitle_with_lead_label_clocked(ModelId::Gfs, "19571218", 12, 1, "F000-F001", true),
            "Init 12/18 12Z | F000-F001 | Valid 12/18 13:00Z | GFS"
        );
    }

    #[test]
    fn only_a_sub_hourly_frame_interval_turns_the_minute_clock_on() {
        for (minutes, on) in [
            (Some("15"), true),
            (Some(" 5 "), true),
            (Some("90"), true),
            (Some("60"), false),
            (Some("120"), false),
            (Some("0"), false),
            (Some("quarter"), false),
            (None, false),
        ] {
            assert_eq!(frame_minutes_need_minute_clock(minutes), on, "{minutes:?}");
        }
    }

    /// C-03: a run started from a 174-hour forecast printed exactly what a
    /// genuine analysis run prints, in all 159 of its PNGs.  The lead
    /// belongs in the picture, not only in `run/report.json`.
    #[test]
    fn a_forecast_initialized_run_discloses_its_source_cycle_and_lead() {
        let _guard = disclosure_guard();
        set_initial_condition_disclosure(Some("GFS 08/01 00Z f174".to_string()));
        assert_eq!(
            model_time_subtitle(ModelId::WrfGdex, "20260808", 6, 0),
            "Init 08/08 06Z (GFS 08/01 00Z f174) | F000 | Valid 08/08 06Z | WRF"
        );
        // F still counts hours into THIS run, and Valid still tracks it.
        assert_eq!(
            model_time_subtitle(ModelId::WrfGdex, "20260808", 6, 3),
            "Init 08/08 06Z (GFS 08/01 00Z f174) | F003 | Valid 08/08 09Z | WRF"
        );
        set_initial_condition_disclosure(None);
    }

    /// An analysis run, and every file written before the provenance block
    /// existed, must be byte-identical to 1.4.0.  Absence is UNKNOWN, and
    /// UNKNOWN is never stamped as anything.
    #[test]
    fn an_undisclosed_run_is_byte_identical_to_before() {
        let _guard = disclosure_guard();
        set_initial_condition_disclosure(None);
        let plain = model_time_subtitle(ModelId::WrfGdex, "20260808", 6, 0);
        assert_eq!(plain, "Init 08/08 06Z | F000 | Valid 08/08 06Z | WRF");
        // A blank disclosure clears rather than rendering "Init ... ()".
        set_initial_condition_disclosure(Some("   ".to_string()));
        assert_eq!(model_time_subtitle(ModelId::WrfGdex, "20260808", 6, 0), plain);
        set_initial_condition_disclosure(None);
    }

    /// A hex frame and a global tape import as `wrf`; the name their file
    /// carries replaces `WRF` in the metadata row, and nothing else moves.
    #[test]
    fn a_model_label_replaces_the_generic_wrfout_identity_only() {
        assert_eq!(model_token_with(ModelId::WrfGdex, None), "WRF");
        assert_eq!(model_token_with(ModelId::WrfGdex, Some("WOOF Hex")), "WOOF Hex");
        // A GRIB model keeps its own name even if a wrfout label is in force.
        assert_eq!(model_token_with(ModelId::Hrrr, Some("WOOF Hex")), "HRRR");
        assert_eq!(model_token_with(ModelId::Gfs, None), "GFS");
    }

    #[test]
    fn the_worker_theme_keeps_source_branding_and_generic_model_identity() {
        use rustwx_render::theme::{RenderTheme, RenderThemeFile};
        let file = RenderThemeFile::from_json(
            r#"{"extends":"woof-light","text":{"source_label":"Recast WOOF"}}"#,
        ).expect("worker JSON parses");
        let theme = RenderTheme::from_file_spec(&file, None).expect("worker theme resolves");
        assert_eq!(theme.model_name(&model_token_with(ModelId::WrfGdex, Some("ArWen"))), "WOOF");
        assert_eq!(theme.source_subtitle(Some("source: ArWen".into())), Some("Recast WOOF".into()));
        for name in ["default", "light", "dark"] {
            let bare = RenderTheme::builtin(name).expect("generic palette");
            assert_eq!(bare.model_name(&model_token_with(ModelId::WrfGdex, Some("ArWen"))), "ArWen");
            assert_eq!(bare.model_name(&model_token_with(ModelId::Hrrr, None)), "HRRR");
        }
    }

    /// `rw_mpas_convert` writes this exact name (rw-mpas pins the same
    /// literal), and so do the producers outside this workspace.
    #[test]
    fn the_model_label_attribute_name_is_pinned() {
        assert_eq!(MODEL_LABEL_ATTRIBUTE, "GPUWM_MODEL_LABEL");
        // gpuwm.io.wrfout.WrfoutWriter stamps this on every history file.
        assert_eq!(ENGINE_VERSION_ATTRIBUTE, "GPUWM_VERSION");
        assert_eq!(ENGINE_MODEL_LABEL, "WOOF");
    }

    /// An engine history file says WOOF, a named model keeps its own name,
    /// and stock WRF output (no engine marker) keeps the store identity.
    #[test]
    fn an_engine_written_file_is_woof_and_stock_wrf_stays_wrf() {
        let token = |named: Option<&str>, engine: bool| {
            model_token_with(
                ModelId::WrfGdex,
                file_model_label(named, engine).unwrap().as_deref(),
            )
        };
        assert_eq!(token(None, true), "WOOF");
        assert_eq!(token(None, false), "WRF");
        assert_eq!(token(Some("WOOF Hex"), true), "WOOF Hex");
        assert_eq!(token(Some("WOOF Hex"), false), "WOOF Hex");
        assert_eq!(token(Some("   "), true), "WOOF");
        assert_eq!(token(Some("   "), false), "WRF");
        assert!(file_model_label(Some("a | b"), true).is_err());
        // HRRR and other GRIB models never take a wrfout label.
        assert_eq!(
            model_token_with(ModelId::Hrrr, file_model_label(None, true).unwrap().as_deref()),
            "HRRR"
        );
    }

    #[test]
    fn a_model_label_is_checked_before_it_reaches_the_row() {
        assert_eq!(
            checked_model_label("  ArWen   Global ").unwrap().as_deref(),
            Some("ArWen Global")
        );
        assert_eq!(checked_model_label("   ").unwrap(), None);
        let split = checked_model_label("Hex | F001").unwrap_err();
        assert!(split.contains("separator"), "{split}");
        let long = checked_model_label(&"x".repeat(MODEL_LABEL_MAX_CHARS + 1)).unwrap_err();
        assert!(long.contains("longer than"), "{long}");
        assert!(checked_model_label("Hex\u{7}").is_err());
    }

    #[test]
    fn inputs_that_disagree_on_a_model_label_keep_the_store_identity() {
        let hex = Some("WOOF Hex".to_string());
        assert_eq!(agreed_model_label(&[hex.clone(), hex.clone()]), hex);
        assert_eq!(agreed_model_label(&[hex.clone(), None]), None);
        assert_eq!(agreed_model_label(&[None, hex.clone()]), None);
        assert_eq!(
            agreed_model_label(&[hex.clone(), Some("WOOF Global".to_string())]),
            None
        );
        assert_eq!(agreed_model_label(&[]), None);
    }

    #[test]
    fn panel_field_default_artifact_slug_stays_on_product_slug() {
        let field = WeatherPanelField::new(WeatherProduct::StpFixed, "dimensionless", vec![1.0])
            .with_title_override("STP (fixed layer)");
        assert_eq!(field.artifact_slug(), WeatherProduct::StpFixed.slug());
    }
}
