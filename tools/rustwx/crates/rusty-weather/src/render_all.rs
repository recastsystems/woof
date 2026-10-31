#![allow(dead_code)]

//! Shared "render every stored product" flow, used by `rw_render` (one
//! hour per invocation) and `rw_batch` (per pipelined hour) via `#[path]`
//! inclusion. This module owns the product-request partitioning (catalog
//! keywords vs strict slug lists), the per-hour direct + derived/heavy
//! render pass over [`store_render`], and the windowed compute + render
//! pass over [`windowed_store`] + the products crate's windowed render
//! seam. No render logic lives here: everything feeds the EXACT render
//! paths the GRIB-lane smoke bins use (pixel-parity proven in Task 4).

use std::path::{Path, PathBuf};
use std::time::Instant;

use rustwx_core::{CycleSpec, ModelId, SourceId};
use rustwx_models::{LatestRun, plot_recipe};
use rustwx_products::derived::{
    DerivedBatchRequest, NativeContourRenderMode, is_heavy_derived_recipe_slug,
    store_derived_recipe_slugs, store_heavy_recipe_slugs,
};
use rustwx_products::direct::{DirectBatchRequest, store_direct_recipe_slugs};
use rustwx_products::places::PlaceLabelOverlay;
use rustwx_products::shared_context::{DomainSpec, TitleProvenance};
use rustwx_products::source::ProductSourceMode;
use rustwx_products::windowed::{
    HrrrWindowedBatchRequest, HrrrWindowedProduct, StoreWindowedGrid,
    render_windowed_products_from_store_grids,
};
use rustwx_render::PngCompressionMode;
use rw_store::RwsExactTime;

#[path = "store_render.rs"]
pub mod store_render;
#[path = "windowed_store.rs"]
pub mod windowed_store;

pub use store_render::{StoreFieldSource, StoreRenderSkip};

/// Which products were asked for, and whether unresolvable ones fail the
/// run (only explicit slug lists are strict; the catalog keywords render
/// what exists and report the rest).
pub struct ProductRequest {
    pub direct: Vec<String>,
    pub derived: Vec<String>,
    /// Explicit arbitrary 2-D store variable names (`var:<name>`).  The
    /// catalog KEYWORDS leave this empty -- the keyword vocabulary is
    /// store-independent -- but the store-aware catalog expansion
    /// (`inspect_renderable_products`) enumerates every stored variable
    /// as a `var:` slug, so a catalog-expanded "all" does include them.
    pub generic: Vec<String>,
    pub windowed: Vec<String>,
    /// The windowed list came from the "all" keyword: render it only when
    /// the run has more than one stored hour (a single hour realizes only
    /// the degenerate 1 h windows, which the per-hour lanes already cover).
    pub windowed_auto: bool,
    pub strict: bool,
}

impl ProductRequest {
    /// Drop the heavy recipe slugs from a non-strict request, for runs
    /// whose ingest skipped the heavy stage, where the 16 heavy grids are
    /// EXPECTED absent rather than blocked. Returns how many were dropped.
    /// Strict (explicit slug list) requests are left alone: asking for a
    /// heavy product by name against a no-heavy store should fail loudly.
    pub fn drop_heavy_unless_strict(&mut self) -> usize {
        if self.strict {
            return 0;
        }
        let before = self.derived.len();
        self.derived
            .retain(|slug| !is_heavy_derived_recipe_slug(slug));
        before - self.derived.len()
    }
}

// Model-identity-free (gpuwm vendor divergence): the store lanes list
// every non-opt-in catalog candidate and let per-store field
// availability decide -- filtering "all"/"direct" through a per-model
// fetch plan here was a source-identity gate on the render path.
/// Every product slug this build can be asked for by name, deduplicated.
///
/// The four families overlap in principle (a slug is classified by the
/// first family that claims it), so this counts distinct spellings rather
/// than summing the catalogs.
///
/// The direct half is the WHOLE direct catalog, opt-in ensemble and
/// probabilistic families included, because this answers "what may I ask
/// for by name?" and those families are exactly the ones a user MUST ask
/// for by name.  It is deliberately not `store_direct_recipe_slugs`,
/// which answers the different question "what does `all` expand to" and
/// must keep excluding them.  A slug the unknown-product refusal counts
/// but the listing hides is a vocabulary nobody can read.
pub fn known_product_slugs() -> Vec<String> {
    let mut slugs: Vec<String> = rustwx_products::spec::direct_product_specs()
        .into_iter()
        .map(|spec| spec.slug)
        .collect();
    slugs.extend(store_derived_recipe_slugs().into_iter().map(str::to_string));
    slugs.extend(store_heavy_recipe_slugs().into_iter().map(str::to_string));
    slugs.extend(
        HrrrWindowedProduct::supported_products()
            .iter()
            .map(|product| product.slug().to_string()),
    );
    slugs.sort();
    slugs.dedup();
    slugs
}

/// How many distinct product slugs `--list-products` will print.
pub fn known_product_slug_count() -> usize {
    known_product_slugs().len()
}

pub fn partition_products(spec: &str) -> Result<ProductRequest, Box<dyn std::error::Error>> {
    let derived_catalog = || {
        store_derived_recipe_slugs()
            .into_iter()
            .map(str::to_string)
            .collect::<Vec<_>>()
    };
    let heavy_catalog = || {
        store_heavy_recipe_slugs()
            .into_iter()
            .map(str::to_string)
            .collect::<Vec<_>>()
    };
    let windowed_catalog = || {
        HrrrWindowedProduct::supported_products()
            .iter()
            .map(|product| product.slug().to_string())
            .collect::<Vec<_>>()
    };
    match spec.trim() {
        "all" => Ok(ProductRequest {
            direct: store_direct_recipe_slugs(),
            derived: derived_catalog()
                .into_iter()
                .chain(heavy_catalog())
                .collect(),
            generic: Vec::new(),
            windowed: windowed_catalog(),
            windowed_auto: true,
            strict: false,
        }),
        "direct" => Ok(ProductRequest {
            direct: store_direct_recipe_slugs(),
            derived: Vec::new(),
            generic: Vec::new(),
            windowed: Vec::new(),
            windowed_auto: false,
            strict: false,
        }),
        "derived" => Ok(ProductRequest {
            direct: Vec::new(),
            derived: derived_catalog(),
            generic: Vec::new(),
            windowed: Vec::new(),
            windowed_auto: false,
            strict: false,
        }),
        "heavy" => Ok(ProductRequest {
            direct: Vec::new(),
            derived: heavy_catalog(),
            generic: Vec::new(),
            windowed: Vec::new(),
            windowed_auto: false,
            strict: false,
        }),
        "windowed" => Ok(ProductRequest {
            direct: Vec::new(),
            derived: Vec::new(),
            generic: Vec::new(),
            windowed: windowed_catalog(),
            windowed_auto: false,
            strict: false,
        }),
        list => {
            let mut direct = Vec::new();
            let mut derived = Vec::new();
            let mut generic = Vec::new();
            let mut windowed = Vec::new();
            for slug in list.split(',').map(str::trim).filter(|s| !s.is_empty()) {
                if let Some(name) = slug.strip_prefix("var:") {
                    validate_generic_variable_name(name)?;
                    generic.push(name.to_string());
                    continue;
                }
                let is_derived = store_derived_recipe_slugs().contains(&slug)
                    || store_heavy_recipe_slugs().contains(&slug)
                    || is_heavy_derived_recipe_slug(slug);
                if HrrrWindowedProduct::from_slug(slug).is_some() {
                    windowed.push(slug.to_string());
                } else if is_derived {
                    derived.push(slug.to_string());
                } else if plot_recipe(slug).is_some() {
                    direct.push(slug.to_string());
                } else {
                    // Name the problem AND the choices.  The bare "neither a
                    // direct plot recipe, ..." told a user what the token was
                    // not, which is unactionable when the vocabulary is a few
                    // hundred slugs long; the group keywords are the choices
                    // at this level and --list-products prints the rest.
                    return Err(format!(
                        "unknown product '{slug}'; choose from 'all', 'direct', \
                         'derived', 'heavy', 'windowed', 'variables' (every \
                         stored 2-D variable; these keywords stand alone), \
                         'var:<stored 2-D variable>', or a comma-separated \
                         list of the {} product slugs that --list-products \
                         prints",
                        known_product_slug_count()
                    )
                    .into());
                }
            }
            if direct.is_empty() && derived.is_empty() && generic.is_empty() && windowed.is_empty()
            {
                return Err("pass at least one product slug via --products".into());
            }
            Ok(ProductRequest {
                direct,
                derived,
                generic,
                windowed,
                windowed_auto: false,
                strict: true,
            })
        }
    }
}

/// Request-safety validation for a `var:<name>` generic product token.
/// The vocabulary is the store's, not this build's, so only the properties
/// that keep the name safe inside comma-joined product specs and log lines
/// are enforced here; existence is checked against the opened store.
pub(crate) fn validate_generic_variable_name(name: &str) -> Result<(), Box<dyn std::error::Error>> {
    if name.is_empty() {
        return Err("generic store product must name a variable after 'var:'".into());
    }
    if name.len() > 512 {
        return Err("generic store variable name exceeds 512 bytes".into());
    }
    if name.trim() != name || name.contains(',') || name.chars().any(char::is_control) {
        return Err(format!("generic store variable name is not request-safe: {name:?}").into());
    }
    Ok(())
}

/// Everything the render passes need to know, independent of any bin's CLI.
#[derive(Clone)]
pub struct StoreRenderConfig {
    pub model: ModelId,
    pub date_yyyymmdd: String,
    pub cycle_utc: u8,
    /// Source stamped into provenance subtitles (the store does not record
    /// the fetch source).
    pub source: SourceId,
    /// Grid spacing appended to the time subtitle (`Δx 3 km`).  The store
    /// carries no spacing metadata, so this is the importer's reading of
    /// the source file's own `DX`.
    pub subtitle_spacing: Option<String>,
    /// Provenance label displacing [`Self::source`] in the subtitle, for
    /// runs that were produced locally rather than fetched.
    pub source_label: Option<String>,
    /// Where these frames came from, for the plot headline.  A locally
    /// imported run names its own grid there; a fetched run leaves the
    /// lane's dataset token alone.
    pub title_provenance: TitleProvenance,
    pub domain: DomainSpec,
    pub out_dir: PathBuf,
    pub contour_mode: NativeContourRenderMode,
    pub native_fill_level_multiplier: usize,
    pub output_width: u32,
    pub output_height: u32,
    pub png_compression: PngCompressionMode,
    pub place_label_overlay: Option<PlaceLabelOverlay>,
    /// gpuwm addition (VENDOR.md): caller-supplied map overlays in
    /// geographic degrees (`rw_wrfbatch --overlays FILE.json`).  `None`
    /// executes no overlay code, so the default render is byte-unchanged.
    pub geographic_overlays: Option<rustwx_products::geographic_overlays::MapOverlays>,
    /// gpuwm addition (VENDOR.md): title/subtitle overrides
    /// (`rw_wrfbatch --annotate FILE.json`).
    pub panel_annotations: Option<rustwx_products::geographic_overlays::PanelAnnotations>,
}

impl StoreRenderConfig {
    fn latest_run(&self) -> Result<LatestRun, Box<dyn std::error::Error>> {
        Ok(LatestRun {
            model: self.model,
            cycle: CycleSpec::new(self.date_yyyymmdd.clone(), self.cycle_utc)?,
            source: self.source,
        })
    }
}

#[derive(Debug)]
pub(crate) struct HourPresentation {
    pub(crate) forecast_hour: u16,
    pub(crate) output_suffix: Option<String>,
    pub(crate) subtitle_left: Option<String>,
}

/// `source: ArWen` -- the provenance line for a run nothing fetched.
fn source_subtitle_override(config: &StoreRenderConfig) -> Option<String> {
    config
        .source_label
        .as_deref()
        .map(|label| format!("source: {label}"))
}

fn hour_presentation(
    config: &StoreRenderConfig,
    storage_slot: u16,
    exact_time: Option<RwsExactTime>,
) -> Result<HourPresentation, Box<dyn std::error::Error>> {
    let Some(exact) = exact_time else {
        // The whole-hour axis has no bespoke subtitle; it only needs one
        // built here when a spacing segment has to be appended to it.
        let subtitle_left = config.subtitle_spacing.as_deref().map(|spacing| {
            format!(
                "{} | {spacing}",
                rustwx_products::shared_context::model_time_subtitle(
                    config.model,
                    &config.date_yyyymmdd,
                    config.cycle_utc,
                    storage_slot,
                )
            )
        });
        return Ok(HourPresentation {
            forecast_hour: storage_slot,
            output_suffix: None,
            subtitle_left,
        });
    };

    // Exact-time stores already carry ordinal u16 slots separately from their
    // physical u64 lead. The legacy products request uses this number only for
    // its artifact token here: fields come from the opened slot, and BOTH the
    // subtitle and suffix below carry the authoritative exact timestamp/lead.
    // Long integrations must use their slot token rather than narrowing their
    // physical lead or refusing otherwise valid reanalysis frames.
    let forecast_hour = u16::try_from(exact.lead_seconds / 3_600).unwrap_or(storage_slot);
    let valid = exact_valid_parts(exact.valid_unix);
    let lead_hours = exact.lead_seconds / 3_600;
    let lead_minutes = (exact.lead_seconds % 3_600) / 60;
    let lead_seconds = exact.lead_seconds % 60;
    let init = if config.date_yyyymmdd.len() == 8 {
        format!(
            "{}/{}",
            &config.date_yyyymmdd[4..6],
            &config.date_yyyymmdd[6..8]
        )
    } else {
        config.date_yyyymmdd.clone()
    };
    // Sub-minute leads are rare; keep the subtitle compact enough to
    // survive the renderer's width limit by dropping zero seconds.
    let lead_tail = if lead_seconds == 0 {
        String::new()
    } else {
        format!(":{lead_seconds:02}")
    };
    let valid_tail = if valid.5 == 0 {
        String::new()
    } else {
        format!(":{:02}", valid.5)
    };
    let spacing_tail = match config.subtitle_spacing.as_deref() {
        Some(spacing) => format!(" | {spacing}"),
        None => String::new(),
    };
    // The init clock is the frame's own (valid - lead), not the run's cycle
    // HOUR: a forecast from a 10:15Z analysis is "Init 07/10 10:15Z | +003:00 |
    // Valid 07/10 13:15Z", never "Init 07/10 10Z | +003:00 | Valid ... 13:15Z".
    let init_clock = i64::try_from(exact.lead_seconds)
        .ok()
        .and_then(|lead| exact.valid_unix.checked_sub(lead))
        .map(exact_valid_parts)
        .filter(|init| init.4 != 0 || init.5 != 0)
        .map(|init| match init.5 {
            0 => format!("{:02}:{:02}Z", init.3, init.4),
            second => format!("{:02}:{:02}:{second:02}Z", init.3, init.4),
        })
        .unwrap_or_else(|| format!("{:02}Z", config.cycle_utc));
    Ok(HourPresentation {
        forecast_hour,
        output_suffix: Some(format!(
            "valid_{:04}{:02}{:02}_{:02}{:02}{:02}z_lead_{lead_hours:03}h{lead_minutes:02}m{lead_seconds:02}s",
            valid.0, valid.1, valid.2, valid.3, valid.4, valid.5
        )),
        subtitle_left: Some(format!(
            "Init {init} {init_clock} | +{lead_hours:03}:{lead_minutes:02}{lead_tail} | Valid {:02}/{:02} {:02}:{:02}{valid_tail}Z | {}{spacing_tail}",
            valid.1,
            valid.2,
            valid.3,
            valid.4,
            // A theme may name the model (`text.model_label`).
            rustwx_render::theme::active_theme()
                .model_name(&rustwx_products::shared_context::model_token(config.model))
        )),
    })
}

/// Proleptic-Gregorian UTC components from an i64 Unix timestamp.
fn exact_valid_parts(valid_unix: i64) -> (i128, u32, u32, u32, u32, u32) {
    let days = valid_unix.div_euclid(86_400);
    let second_of_day = valid_unix.rem_euclid(86_400);
    let z = i128::from(days) + 719_468;
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

/// One rendered product (any lane), with its render wall and output path.
pub struct RenderedProduct {
    pub slug: String,
    pub total_ms: u128,
    pub output_path: PathBuf,
    /// What the finished PNG maps to on the Earth, when the render lane
    /// could publish it (gpuwm addition, VENDOR.md).
    pub georeference: Option<rustwx_render::PanelGeoReference>,
    /// Why `georeference` is `None`, from the lane that rendered it.
    pub georeference_absent_reason: Option<String>,
}

/// Outcome of one hour's direct + derived/heavy render pass.
pub struct HourRenderOutcome {
    pub rendered: Vec<RenderedProduct>,
    pub skipped: Vec<StoreRenderSkip>,
}

/// Render the requested direct and derived/heavy products from one stored
/// hour through the existing render paths. Products whose inputs are not
/// in the store come back in `skipped` with the missing selector/grid:
/// the caller decides whether that fails the run (strict requests).
pub fn render_hour_products(
    config: &StoreRenderConfig,
    store: &StoreFieldSource,
    hour: u16,
    direct_slugs: &[String],
    derived_slugs: &[String],
    generic_variables: &[String],
    // Optional pacing hook for the direct lane's chunked render: called
    // before each chunk loads its fields. `rw_batch` passes its memory
    // gate (defer chunks inside high-memory ingest windows); `rw_render`
    // passes None. Timing-only: pixels are gate-independent.
    direct_chunk_gate: Option<&dyn Fn()>,
) -> Result<HourRenderOutcome, Box<dyn std::error::Error>> {
    let mut rendered = Vec::new();
    let mut skipped = Vec::new();
    let presentation = hour_presentation(config, hour, store.exact_time())?;

    if !direct_slugs.is_empty() {
        let mut direct_request = DirectBatchRequest {
            accumulation_window: None,
            geographic_overlays: config.geographic_overlays.clone(),
            panel_annotations: config.panel_annotations.clone(),
            model: config.model,
            date_yyyymmdd: config.date_yyyymmdd.clone(),
            cycle_override_utc: Some(config.cycle_utc),
            forecast_hour: presentation.forecast_hour,
            source: config.source,
            domain: config.domain.clone(),
            out_dir: config.out_dir.clone(),
            cache_root: config.out_dir.join("cache"),
            use_cache: false,
            recipe_slugs: direct_slugs.to_vec(),
            product_overrides: std::collections::HashMap::new(),
            contour_mode: config.contour_mode,
            native_fill_level_multiplier: config.native_fill_level_multiplier.max(1),
            output_width: config.output_width,
            output_height: config.output_height,
            png_compression: config.png_compression,
            place_label_overlay: config.place_label_overlay.clone(),
            output_suffix: presentation.output_suffix.clone(),
            subtitle_left_override: presentation.subtitle_left.clone(),
            subtitle_right_override: source_subtitle_override(config),
            title_provenance: config.title_provenance.clone(),
        };
        if config.title_provenance.is_local_import() && direct_slugs.iter().any(|slug| {
            matches!(slug.as_str(), "snowfall_window" | "snow_10to1_window" | "snow_kuchera_window" | "snowfall_liquid_equivalent" | "snow_depth_change")
        }) {
            let since = std::env::var("GPUWM_SNOW_SINCE").ok()
                .filter(|value| !value.trim().is_empty());
            if let Some(exact) = store.exact_time() {
                if let Some(origin) = exact.origin_unix() {
                    direct_request.accumulation_window = Some(
                        rustwx_products::direct::local_accumulation_window(
                            origin, exact.valid_unix, since.as_deref(),
                        )?,
                    );
                }
            } else {
                direct_request.accumulation_window = Some(
                    rustwx_products::direct::local_accumulation_window_from_cycle(
                        &config.date_yyyymmdd, config.cycle_utc, hour, since.as_deref(),
                    )?,
                );
            }
        }
        let outcome = store_render::render_direct_recipes_from_store(
            store,
            &direct_request,
            &config.latest_run()?,
            direct_slugs,
            direct_chunk_gate,
        )?;
        rendered.extend(outcome.rendered.into_iter().map(|recipe| RenderedProduct {
            slug: recipe.recipe_slug,
            total_ms: recipe.timing.total_ms,
            output_path: recipe.output_path,
            georeference: recipe.georeference,
            georeference_absent_reason: recipe.georeference_absent_reason,
        }));
        skipped.extend(outcome.skipped);
    }

    if !derived_slugs.is_empty() {
        // The derived/heavy store-render pass loads every requested grid
        // as f64 up front (~0.5-0.7 GB at HRRR size); defer its START out
        // of high-memory ingest windows the same way direct chunks defer.
        if let Some(gate) = direct_chunk_gate {
            gate();
        }
        let derived_request = DerivedBatchRequest {
            model: config.model,
            date_yyyymmdd: config.date_yyyymmdd.clone(),
            cycle_override_utc: Some(config.cycle_utc),
            forecast_hour: presentation.forecast_hour,
            source: config.source,
            domain: config.domain.clone(),
            out_dir: config.out_dir.clone(),
            cache_root: config.out_dir.join("cache"),
            use_cache: false,
            recipe_slugs: derived_slugs.to_vec(),
            surface_product_override: None,
            pressure_product_override: None,
            source_mode: ProductSourceMode::Canonical,
            allow_large_heavy_domain: false,
            contour_mode: config.contour_mode,
            native_fill_level_multiplier: config.native_fill_level_multiplier.max(1),
            output_width: config.output_width,
            output_height: config.output_height,
            png_compression: config.png_compression,
            place_label_overlay: config.place_label_overlay.clone(),
            output_suffix: presentation.output_suffix.clone(),
            subtitle_left_override: presentation.subtitle_left.clone(),
            subtitle_right_override: source_subtitle_override(config),
            title_provenance: config.title_provenance.clone(),
        };
        let outcome = store_render::render_derived_recipes_from_store(
            store,
            &derived_request,
            config.cycle_utc,
            derived_slugs,
        )?;
        rendered.extend(outcome.rendered.into_iter().map(|recipe| RenderedProduct {
            slug: recipe.recipe_slug,
            total_ms: recipe.timing.total_ms,
            output_path: recipe.output_path,
            georeference: recipe.georeference,
            georeference_absent_reason: recipe.georeference_absent_reason,
        }));
        skipped.extend(outcome.skipped);
    }

    for variable in generic_variables {
        // Same memory pacing as the other lanes: a generic render decodes
        // one full plane and builds one projected map.
        if let Some(gate) = direct_chunk_gate {
            gate();
        }
        match store_render::render_generic_store_variable(store, config, hour, variable) {
            Ok(product) => rendered.push(RenderedProduct {
                slug: format!("var:{}", product.variable),
                total_ms: product.total_ms,
                output_path: product.output_path,
                georeference: product.georeference,
                georeference_absent_reason: product.georeference_absent_reason,
            }),
            Err(error) => skipped.push(StoreRenderSkip {
                slug: format!("var:{variable}"),
                reason: error.to_string(),
            }),
        }
    }

    Ok(HourRenderOutcome { rendered, skipped })
}

/// Outcome of the windowed compute + render pass over the run's stored
/// hours, anchored at the requested stored hour.
pub struct WindowedRenderOutcome {
    pub rendered: Vec<RenderedProduct>,
    pub blocked: Vec<StoreRenderSkip>,
    pub anchor_hour: u16,
    pub stored_hours: usize,
    pub compute_ms: u128,
}

/// Compute and render the requested windowed products across the run's
/// stored hours through `anchor_hour`, a storage slot: on an exact-time
/// store the pictures carry that frame's whole lead hour, which the
/// window compute returns as its own `anchor_hour`. Earlier frames remain
/// available as accumulation baselines even when only the anchor frame was
/// selected for rendering. Future frames never change the requested window or its time.
/// `auto` is the "all"-keyword gate: with it set, a run with at most one
/// stored hour skips the lane entirely (returns `None`).
/// `store` only carries the run grid + projection for the render half.
pub fn render_windowed_products(
    config: &StoreRenderConfig,
    store: &StoreFieldSource,
    store_root: &Path,
    model_slug: &str,
    run_slug: &str,
    requested: &[String],
    anchor_hour: u16,
    auto: bool,
) -> Result<Option<WindowedRenderOutcome>, Box<dyn std::error::Error>> {
    let stored_hours = windowed_store::stored_run_hours(store_root, model_slug, run_slug)?;
    if auto && stored_hours.len() <= 1 {
        return Ok(None);
    }
    if !stored_hours.contains(&anchor_hour) {
        return Err(format!(
            "window anchor F{anchor_hour:03} is not stored in {model_slug}/{run_slug}"
        )
        .into());
    }
    let available_hours: Vec<u16> = stored_hours
        .into_iter()
        .filter(|hour| *hour <= anchor_hour)
        .collect();
    let compute_started = Instant::now();
    let outcome = windowed_store::compute_windowed_products(
        store_root,
        model_slug,
        run_slug,
        &available_hours,
        requested,
    )?;
    let compute_ms = compute_started.elapsed().as_millis();
    let windowed_request = HrrrWindowedBatchRequest {
        model: config.model,
        date_yyyymmdd: config.date_yyyymmdd.clone(),
        cycle_override_utc: Some(config.cycle_utc),
        forecast_hour: outcome.anchor_hour,
        source: config.source,
        domain: config.domain.clone(),
        out_dir: config.out_dir.clone(),
        cache_root: config.out_dir.join("cache"),
        use_cache: false,
        products: Vec::new(),
        output_width: config.output_width,
        output_height: config.output_height,
        png_compression: config.png_compression,
        place_label_overlay: config.place_label_overlay.clone(),
        // The windowed lane composes its own lead label (the window, not
        // a forecast hour), so the spacing rides as a suffix rather than
        // replacing the whole line.
        subtitle_left_suffix: config.subtitle_spacing.clone(),
        subtitle_right_override: source_subtitle_override(config),
        title_provenance: config.title_provenance.clone(),
    };
    let grids: Vec<StoreWindowedGrid> = outcome
        .grids
        .into_iter()
        .map(|grid| StoreWindowedGrid {
            slug: grid.slug,
            title: grid.title,
            units: grid.units,
            values: grid.values,
            hours_used: grid.hours_used,
            window_hours: grid.window_hours,
            strategy: grid.strategy,
        })
        .collect();
    let rendered = render_windowed_products_from_store_grids(
        &windowed_request,
        config.cycle_utc,
        &store.full_grid(),
        store.projection(),
        &grids,
    )?;
    Ok(Some(WindowedRenderOutcome {
        rendered: rendered
            .into_iter()
            .map(|product| RenderedProduct {
                slug: product.product.slug().to_string(),
                total_ms: product.timing.total_ms,
                output_path: product.output_path,
                georeference: product.georeference,
                georeference_absent_reason: product.georeference_absent_reason,
            })
            .collect(),
        blocked: outcome
            .blockers
            .into_iter()
            .map(|(slug, reason)| StoreRenderSkip { slug, reason })
            .collect(),
        anchor_hour: outcome.anchor_hour,
        stored_hours: available_hours.len(),
        compute_ms,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn presentation_config(
        subtitle_spacing: Option<&str>,
        source_label: Option<&str>,
    ) -> StoreRenderConfig {
        StoreRenderConfig {
            model: ModelId::WrfGdex,
            date_yyyymmdd: "19740403".to_string(),
            cycle_utc: 18,
            source: SourceId::Gdex,
            subtitle_spacing: subtitle_spacing.map(str::to_string),
            source_label: source_label.map(str::to_string),
            title_provenance: TitleProvenance::default(),
            domain: DomainSpec::new("d02-1km", (-98.0, -95.0, 38.0, 40.0)),
            out_dir: PathBuf::from("out"),
            contour_mode: Default::default(),
            native_fill_level_multiplier: 1,
            output_width: 1_200,
            output_height: 900,
            png_compression: PngCompressionMode::Fast,
            place_label_overlay: None,
            geographic_overlays: None,
            panel_annotations: None,
        }
    }

    #[test]
    fn whole_hour_subtitles_gain_the_spacing_segment() {
        // Without a spacing there is nothing to say, and the products
        // crate keeps composing the line itself (override stays None).
        let plain = presentation_config(None, None);
        assert!(
            hour_presentation(&plain, 3, None)
                .unwrap()
                .subtitle_left
                .is_none()
        );

        let spaced = presentation_config(Some("\u{0394}x 1 km"), None);
        assert_eq!(
            hour_presentation(&spaced, 3, None).unwrap().subtitle_left,
            Some("Init 04/03 18Z | F003 | Valid 04/03 21Z | WRF | \u{0394}x 1 km".to_string())
        );
    }

    #[test]
    fn exact_time_subtitles_gain_the_same_segment() {
        let spaced = presentation_config(Some("\u{0394}x 333 m"), None);
        // 1974-04-03 18:30:00Z, half an hour after an 18Z initialization.
        let exact = RwsExactTime::new(1_800, 134_245_800);
        let presentation = hour_presentation(&spaced, 0, Some(exact)).unwrap();
        let subtitle = presentation.subtitle_left.unwrap();
        assert!(
            subtitle.ends_with(" | \u{0394}x 333 m"),
            "spacing must ride at the end of the exact-time line: {subtitle}"
        );
        assert!(
            subtitle.starts_with("Init 04/03 18Z | +000:30 | Valid "),
            "{subtitle}"
        );
        // The exact-time suffix is untouched by the spacing segment.
        assert_eq!(
            presentation.output_suffix.as_deref(),
            Some("valid_19740403_183000z_lead_000h30m00s")
        );
    }

    #[test]
    fn exact_time_init_off_the_hour_names_its_minutes() {
        let mut config = presentation_config(None, None);
        config.date_yyyymmdd = "20230710".to_string();
        config.cycle_utc = 10;
        // 2023-07-10 13:15:00Z, 3 h after a 10:15Z analysis.
        let exact = RwsExactTime::new(10_800, 1_688_994_900);
        let subtitle = hour_presentation(&config, 1, Some(exact)).unwrap().subtitle_left.unwrap();
        assert!(subtitle.starts_with("Init 07/10 10:15Z | +003:00 | Valid 07/10 13:15Z |"), "{subtitle}");
    }

    #[test]
    fn exact_time_long_integration_retains_physical_lead_and_valid_stamp() {
        let mut config = presentation_config(Some("Δx 4 km"), None);
        config.date_yyyymmdd = "19791001".to_string();
        config.cycle_utc = 0;
        let exact = RwsExactTime::new(1_317_330_000, 1_624_914_000);
        let presentation = hour_presentation(&config, 7, Some(exact)).unwrap();
        assert_eq!(presentation.forecast_hour, 7, "artifact token is the storage slot");
        assert_eq!(presentation.output_suffix.as_deref(),
            Some("valid_20210628_210000z_lead_365925h00m00s"));
        let subtitle = presentation.subtitle_left.unwrap();
        assert!(subtitle.contains("+365925:00"), "{subtitle}");
        assert!(subtitle.contains("Valid 06/28 21:00Z"), "{subtitle}");
        // Keep the existing whole-hour token when it is representable.
        let boundary = hour_presentation(&config, 7,
            Some(RwsExactTime::new(u64::from(u16::MAX) * 3_600, exact.valid_unix))).unwrap();
        assert_eq!(boundary.forecast_hour, u16::MAX);
    }

    #[test]
    fn generic_subtitle_row_never_sacrifices_the_stamp_or_valid_time() {
        // Exact-time frame with a spacing segment: the LONGEST left line
        // this lane composes.  The right half must be EXACTLY the
        // provenance stamp named products carry -- the renderer measures
        // the right subtitle and end-ellipsizes anything longer inside a
        // half-row cap, which is how "rw-store variable el [m] |
        // source: ArWen" lost its stamp and squeezed the valid time to
        // "Valid 1..." in the field.
        let config = presentation_config(Some("\u{0394}x 100 m"), Some("ArWen"));
        // 18Z init + 4 h 50 min lead -> valid 1974-04-03 22:50:00Z.
        let exact = RwsExactTime::new(17_400, 134_261_400);
        let presentation = hour_presentation(&config, 0, Some(exact)).unwrap();
        let (left, right) = store_render::generic_subtitle_row(&config, &presentation);
        assert_eq!(
            right, "source: ArWen",
            "the stamp stands alone; no variable/units segment may share \
             (and therefore squeeze) the right half"
        );
        assert!(
            left.contains("Valid 04/03 22:50Z"),
            "the full valid time must survive: {left}"
        );
        assert!(left.starts_with("Init 04/03 18Z | +004:50 |"), "{left}");
        assert!(left.ends_with("| \u{0394}x 100 m"), "{left}");

        // Whole-hour frame without overrides: the shared model time
        // subtitle carries the full valid segment; the stamp is the
        // registered source.
        let plain = presentation_config(None, None);
        let presentation = hour_presentation(&plain, 3, None).unwrap();
        let (left, right) = store_render::generic_subtitle_row(&plain, &presentation);
        assert!(left.contains("F003"), "{left}");
        assert!(left.contains("Valid"), "{left}");
        assert!(right.starts_with("source: "), "{right}");
    }

    #[test]
    fn the_source_label_displaces_the_inherited_fetch_source() {
        // A locally imported run was never fetched from GDEX; saying so
        // is the whole point of the label.
        assert_eq!(
            source_subtitle_override(&presentation_config(None, Some("ArWen"))),
            Some("source: ArWen".to_string())
        );
        // Absent a label the lane keeps the model's registered source.
        assert_eq!(
            source_subtitle_override(&presentation_config(None, None)),
            None
        );
    }

    #[test]
    fn products_keywords_pull_the_catalogs() {
        let all = partition_products("all").unwrap();
        assert!(!all.strict);
        assert_eq!(all.direct, store_direct_recipe_slugs());
        assert!(
            all.generic.is_empty(),
            "the store-independent keyword vocabulary cannot name store variables; \
             the catalog expansion adds them"
        );
        assert_eq!(
            all.derived.len(),
            store_derived_recipe_slugs().len() + store_heavy_recipe_slugs().len()
        );
        assert_eq!(
            all.windowed.len(),
            HrrrWindowedProduct::supported_products().len()
        );
        assert!(
            all.windowed_auto,
            "'all' must gate windowed on multi-hour stores"
        );

        let heavy = partition_products("heavy").unwrap();
        assert!(heavy.direct.is_empty());
        assert_eq!(heavy.derived.len(), store_heavy_recipe_slugs().len());
        assert!(heavy.windowed.is_empty());

        let windowed = partition_products("windowed").unwrap();
        assert!(windowed.direct.is_empty() && windowed.derived.is_empty());
        assert_eq!(
            windowed.windowed.len(),
            HrrrWindowedProduct::supported_products().len()
        );
        assert!(
            !windowed.windowed_auto,
            "explicit 'windowed' keyword must render even single-hour stores"
        );
        assert!(!windowed.strict);
    }

    #[test]
    fn product_lists_classify_into_lanes_and_are_strict() {
        let picked = partition_products(
            "2m_temperature,sbcape,ecape_stp,var:custom_plane,qpf_6h,uh_2to5km_run_max",
        )
        .unwrap();
        assert!(picked.strict);
        assert_eq!(picked.direct, vec!["2m_temperature".to_string()]);
        assert_eq!(
            picked.derived,
            vec!["sbcape".to_string(), "ecape_stp".to_string()]
        );
        assert_eq!(picked.generic, vec!["custom_plane".to_string()]);
        assert_eq!(
            picked.windowed,
            vec!["qpf_6h".to_string(), "uh_2to5km_run_max".to_string()]
        );
        assert!(!picked.windowed_auto);
        assert!(partition_products("definitely_not_a_product").is_err());
        assert!(partition_products("var:").is_err());
        assert!(partition_products("var:unsafe\nname").is_err());
        assert!(partition_products("var: padded ").is_err());
    }

    #[test]
    fn drop_heavy_strips_only_heavy_slugs_and_respects_strict() {
        let mut all = partition_products("all").unwrap();
        let dropped = all.drop_heavy_unless_strict();
        assert_eq!(dropped, store_heavy_recipe_slugs().len());
        assert_eq!(all.derived.len(), store_derived_recipe_slugs().len());
        assert!(
            all.derived
                .iter()
                .all(|slug| !is_heavy_derived_recipe_slug(slug))
        );

        let mut strict = partition_products("sbcape,ecape_stp").unwrap();
        assert_eq!(strict.drop_heavy_unless_strict(), 0);
        assert_eq!(
            strict.derived,
            vec!["sbcape".to_string(), "ecape_stp".to_string()],
            "strict requests must keep explicitly named heavy slugs"
        );
    }
}
