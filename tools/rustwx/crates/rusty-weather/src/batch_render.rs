//! Bounded, callback-driven batch rendering over an `rw-store` run.
//!
//! This is orchestration only.  Every image goes through [`crate::render_all`],
//! which in turn uses the production direct, derived, heavy, and windowed
//! render paths.  Callers own the worker thread and receive plain-data events,
//! making the API suitable for egui without ever blocking its frame thread.

use std::collections::{HashMap, HashSet};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;
use std::time::Instant;

use rayon::iter::{IntoParallelIterator, ParallelIterator};

use rustwx_core::{CycleSpec, ModelId, SourceId};
use rustwx_models::{model_summary, plot_recipe_store_requirements};
use rustwx_products::derived::{
    is_heavy_derived_recipe_slug, store_derived_recipe_slugs, store_heavy_recipe_slugs,
};
use rustwx_products::direct::{direct_recipe_is_time_invariant, store_direct_recipe_slugs};
use rustwx_products::shared_context::{DomainSpec, TitleProvenance};
use rustwx_products::windowed::HrrrWindowedProduct;
use rustwx_render::PngCompressionMode;
use rustwx_render::advisory::{self, Advisory};

use crate::host_memory;
use crate::render_all::{
    StoreFieldSource, StoreRenderConfig, partition_products, render_hour_products,
    render_windowed_products,
};

/// Hard defaults used by the GUI lane.  The default job is still one product
/// for one hour; these ceilings merely prevent an accidental unbounded click.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatchRenderLimits {
    pub max_hours: usize,
    pub max_products_per_hour: usize,
    pub max_work_items: usize,
    pub max_output_width: u32,
    pub max_output_height: u32,
    pub max_output_pixels: u64,
}

impl Default for BatchRenderLimits {
    fn default() -> Self {
        Self {
            // Covers long global-model runs while the product-hour ceiling
            // still prevents a giant cross product.
            max_hours: 256,
            max_products_per_hour: 32,
            max_work_items: 512,
            max_output_width: 1_920,
            max_output_height: 1_440,
            max_output_pixels: 2_764_800,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchHourScope {
    Current(u16),
    AllStored,
}

#[derive(Debug, Clone, PartialEq)]
pub enum BatchRenderDomain {
    /// Tight finite lat/lon extent of the run's `grid.rwg`.
    NativeGrid,
    Bounds {
        slug: String,
        west: f64,
        east: f64,
        south: f64,
        north: f64,
    },
}

#[derive(Debug, Clone)]
pub struct BatchRenderRequest {
    pub store_root: PathBuf,
    /// Exact rw-store directory component (for example `hrrr` or `wrf`).
    pub model_slug: String,
    pub run_slug: String,
    pub hours: BatchHourScope,
    /// Comma-separated production slugs, or a `render_all` catalog keyword.
    pub product_spec: String,
    pub out_dir: PathBuf,
    pub domain: BatchRenderDomain,
    /// Slug written into every output filename in place of the generic
    /// `native_grid`, for [`BatchRenderDomain::NativeGrid`] runs whose
    /// caller knows the grid's identity (a WRF nest's `d02-3km`).  One
    /// run's nests are otherwise indistinguishable in the filename --
    /// same model, same cycle, same forecast hour -- and the second
    /// domain rendered at a lead overwrites the first.
    pub native_domain_slug: Option<String>,
    /// Grid spacing appended to the plot's time subtitle (`Δx 3 km`).
    pub subtitle_spacing: Option<String>,
    /// Provenance label for the subtitle, replacing the store model's
    /// registered fetch source.  A locally-imported run was never
    /// fetched from anywhere.
    pub source_label: Option<String>,
    /// Where these frames came from, for the plot headline.  Defaults to
    /// the source catalog, so a fetched run's titles are untouched; a
    /// locally-imported run declares itself here and names its own grid.
    pub title_provenance: TitleProvenance,
    /// Optional label overrides for nonstandard/local run names.
    pub date_yyyymmdd: Option<String>,
    pub cycle_utc: Option<u8>,
    /// Provenance subtitle override.  The model's primary source is used when
    /// omitted because rw-store v1 does not retain fetch-source identity.
    pub source: Option<SourceId>,
    pub output_width: u32,
    pub output_height: u32,
    pub limits: BatchRenderLimits,
    /// gpuwm addition (VENDOR.md): map overlays in geographic degrees,
    /// projected into each product's own frame at render time
    /// (`rw_wrfbatch --overlays FILE.json`).  `None` -- what every caller
    /// that does not pass the flag produces -- runs no overlay code, so
    /// the default render stays byte-identical.
    pub geographic_overlays: Option<rustwx_products::geographic_overlays::MapOverlays>,
    /// gpuwm addition (VENDOR.md): title/subtitle overrides
    /// (`rw_wrfbatch --annotate FILE.json`).
    pub panel_annotations: Option<rustwx_products::geographic_overlays::PanelAnnotations>,
}

impl BatchRenderRequest {
    pub fn conservative(
        store_root: impl Into<PathBuf>,
        model_slug: impl Into<String>,
        run_slug: impl Into<String>,
        hour: u16,
        product_slug: impl Into<String>,
        out_dir: impl Into<PathBuf>,
    ) -> Self {
        Self {
            store_root: store_root.into(),
            model_slug: model_slug.into(),
            run_slug: run_slug.into(),
            hours: BatchHourScope::Current(hour),
            product_spec: product_slug.into(),
            out_dir: out_dir.into(),
            domain: BatchRenderDomain::NativeGrid,
            native_domain_slug: None,
            subtitle_spacing: None,
            source_label: None,
            title_provenance: TitleProvenance::default(),
            date_yyyymmdd: None,
            cycle_utc: None,
            source: None,
            geographic_overlays: None,
            panel_annotations: None,
            output_width: 1_200,
            output_height: 900,
            limits: BatchRenderLimits::default(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum BatchProductKind {
    Direct,
    Derived,
    Heavy,
    Generic,
    Windowed,
}

impl BatchProductKind {
    pub fn label(self) -> &'static str {
        match self {
            Self::Direct => "direct",
            Self::Derived => "derived",
            Self::Heavy => "heavy",
            Self::Generic => "generic",
            Self::Windowed => "windowed",
        }
    }
}

/// One recipe available to the inspected run. Direct recipes are proven from
/// selector metadata and derived recipes from stored slug grids. Windowed
/// candidates are listed for multi-hour HRRR runs and may still report an
/// accurate blocker when their exact contiguous window is incomplete.
/// `source_fields` maps a store-browser selection to recipes that consume it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchProductOption {
    pub slug: String,
    pub kind: BatchProductKind,
    pub source_fields: Vec<String>,
    /// Stored units of the backing variable, when the product IS one
    /// stored variable (derived/heavy grids and generic `var:` rows).
    pub units: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchRenderCatalog {
    pub products: Vec<BatchProductOption>,
    pub stored_hours: Vec<u16>,
}

/// Inspect one hour without decoding any field payloads.  Selector metadata is
/// enough to prove direct-recipe coverage; derived grids are slug-addressed.
/// Model-identity-free (gpuwm vendor divergence): every non-opt-in direct
/// recipe is a candidate and the STORED FIELDS decide, never a per-model
/// fetch plan.
pub fn inspect_renderable_products(
    store_root: &Path,
    model_slug: &str,
    run_slug: &str,
    hour: u16,
) -> Result<BatchRenderCatalog, String> {
    inspect_renderable_products_over(store_root, model_slug, run_slug, &[hour])
}

/// [`inspect_renderable_products`] over several stored hours: every product
/// at least one of `hours` can draw.
///
/// WHAT BREAKAGE THIS PREVENTS (gate law): a series store was cataloged
/// from its FIRST stored frame alone.  A forecast's analysis frame stores
/// no `REFL_10CM`, so a whole-hour frame drawn beside its hour's earlier
/// frames (the baselines its windows fold) lost composite reflectivity
/// although it stored the field itself: on a 3 km CONUS run the lead-1 h
/// frame was published without it, and a render of a whole run from its
/// analysis frame drew it on no frame at all.  An instantaneous product is
/// drawn frame by frame and skipped by name on a frame that lacks its
/// fields, so the catalog of several frames is the union of theirs.
/// Windowed products are decided by the run's time axis, once, as before.
pub fn inspect_renderable_products_over(
    store_root: &Path,
    model_slug: &str,
    run_slug: &str,
    hours: &[u16],
) -> Result<BatchRenderCatalog, String> {
    validate_store_component("model", model_slug)?;
    validate_store_component("run", run_slug)?;
    if hours.is_empty() {
        return Err(format!(
            "run {model_slug}/{run_slug}: a catalog needs at least one stored frame"
        ));
    }
    let mut products = Vec::new();
    let mut named: HashSet<String> = HashSet::new();
    // Every stored 2-D variable of any of the hours, first seen first:
    // (name, units).
    let mut variables: Vec<(String, String)> = Vec::new();
    let mut variable_names: HashSet<String> = HashSet::new();
    let direct_slugs = store_direct_recipe_slugs();
    let known_derived: HashSet<&str> = store_derived_recipe_slugs()
        .into_iter()
        .chain(store_heavy_recipe_slugs())
        .collect();

    for &hour in hours {
        let store = StoreFieldSource::open(store_root, model_slug, run_slug, hour)
            .map_err(|err| err.to_string())?;
        for slug in &direct_slugs {
            if named.contains(slug) {
                continue;
            }
            let Ok(requirements) = plot_recipe_store_requirements(slug) else {
                continue;
            };
            let mut source_fields = Vec::new();
            let mut renderable = true;
            for requirement in requirements {
                let resolved = requirement
                    .selector
                    .and_then(|selector| store.resolve(&selector));
                let Some(name) = resolved else {
                    renderable = false;
                    break;
                };
                if !source_fields.iter().any(|existing| existing == name) {
                    source_fields.push(name.to_string());
                }
            }
            if renderable {
                named.insert(slug.clone());
                products.push(BatchProductOption {
                    slug: slug.clone(),
                    kind: BatchProductKind::Direct,
                    source_fields,
                    units: None,
                });
            }
        }

        for slug in store.derived_slugs() {
            if !known_derived.contains(slug.as_str()) || named.contains(slug) {
                continue;
            }
            named.insert(slug.clone());
            products.push(BatchProductOption {
                slug: slug.clone(),
                kind: if is_heavy_derived_recipe_slug(slug) {
                    BatchProductKind::Heavy
                } else {
                    BatchProductKind::Derived
                },
                source_fields: vec![slug.clone()],
                units: store.surface_variable(slug).map(|var| var.units.clone()),
            });
        }

        for variable in store.surface_variables() {
            if variable_names.insert(variable.name.clone()) {
                variables.push((variable.name.clone(), variable.units.clone()));
            }
        }
    }

    // Every remaining stored 2-D variable is renderable through the generic
    // lane as `var:<name>`.  A variable that is already the SOLE source
    // field of a named product above is excluded -- listing it would render
    // the same grid twice under two slugs -- and every exclusion says so on
    // stderr rather than silently thinning the catalog.
    let sole_source_of_named: HashSet<String> = products
        .iter()
        .filter_map(|product| match product.source_fields.as_slice() {
            [only] => Some(only.clone()),
            _ => None,
        })
        .collect();
    for (name, units) in variables {
        if let Err(reason) = crate::render_all::validate_generic_variable_name(&name) {
            eprintln!(
                "GENERIC_EXCLUDED\t{}\tname is not request-safe: {reason}",
                name.escape_debug()
            );
            continue;
        }
        if sole_source_of_named.contains(&name) {
            eprintln!(
                "GENERIC_EXCLUDED\t{name}\talready rendered by a named product using exactly \
                 this stored grid"
            );
            continue;
        }
        products.push(BatchProductOption {
            slug: format!("var:{name}"),
            kind: BatchProductKind::Generic,
            source_fields: vec![name],
            units: Some(units),
        });
    }

    let stored_hours =
        crate::render_all::windowed_store::stored_run_hours(store_root, model_slug, run_slug)
            .map_err(|err| err.to_string())?;
    // Windowed products are axis-gated, not model-gated: any run with more
    // than one stored frame lists them, and each product's per-plane
    // availability is proven (or blocked with a reason) when the windows
    // compute.  An exact-time run qualifies too: its windows are served
    // from its frames' leads and close on its whole-hour frames.
    let windowed_ready =
        crate::render_all::windowed_store::windowed_axis_ready(store_root, model_slug, run_slug)
            .map_err(|err| err.to_string())?;
    //
    // And only the windows this run can close.  A window whose last hour
    // lies past the run's last stored frame blocks on EVERY frame, so it
    // is not a product this run can draw and it is not a candidate: a
    // catalog keyword that expanded to it asked an 18 h run for 41
    // windowed families, each skipped on all 19 frames (805 skip lines).
    // Named explicitly it is still refused per frame, with a reason that
    // says the run is too short (`windowed_store::compute_windowed_products`).
    let last_stored_hour = crate::render_all::windowed_store::last_window_hour(
        &crate::render_all::windowed_store::stored_window_frames(store_root, model_slug, run_slug)
            .map_err(|err| err.to_string())?,
    );
    if windowed_ready {
        products.extend(
            HrrrWindowedProduct::supported_products()
                .iter()
                .filter(|product| {
                    last_stored_hour.is_some_and(|last| {
                        crate::render_all::windowed_store::window_fits_run(**product, last)
                    })
                })
                .map(|product| BatchProductOption {
                    slug: product.slug().to_string(),
                    kind: BatchProductKind::Windowed,
                    source_fields: Vec::new(),
                    units: None,
                }),
        );
    }

    let mut seen = HashSet::new();
    products.retain(|product| seen.insert(product.slug.clone()));
    products.sort_by(|left, right| {
        left.kind
            .cmp(&right.kind)
            .then_with(|| left.slug.cmp(&right.slug))
    });
    Ok(BatchRenderCatalog {
        products,
        stored_hours,
    })
}

#[derive(Debug, Clone)]
pub enum BatchRenderEvent {
    Started {
        planned_items: usize,
        hours: Vec<u16>,
        products: Vec<String>,
        output_dir: PathBuf,
    },
    HourStarted {
        hour: u16,
        index: usize,
        total: usize,
    },
    ItemStarted {
        hour: Option<u16>,
        slug: String,
        kind: BatchProductKind,
        completed: usize,
        total: usize,
    },
    ItemRendered {
        hour: Option<u16>,
        slug: String,
        output_path: PathBuf,
        render_ms: u128,
        /// What the finished PNG maps to on the Earth, when the render
        /// lane could publish it (gpuwm addition, VENDOR.md).  Carried on
        /// the event so `rw_wrfbatch` can write the run's georeference
        /// manifest beside the PNGs.
        georeference: Option<rustwx_render::PanelGeoReference>,
        /// Why `georeference` is `None`, from the lane that rendered it.
        georeference_absent_reason: Option<String>,
        completed: usize,
        total: usize,
    },
    ItemSkipped {
        hour: Option<u16>,
        slug: String,
        reason: String,
        completed: usize,
        total: usize,
    },
    ItemFailed {
        hour: Option<u16>,
        slug: String,
        error: String,
        completed: usize,
        total: usize,
    },
    Finished(BatchRenderSummary),
}

#[derive(Debug, Clone, Default)]
pub struct BatchRenderSummary {
    pub planned: usize,
    pub rendered: usize,
    pub skipped: usize,
    pub failed: usize,
    pub cancelled: bool,
    pub elapsed_ms: u128,
    pub outputs: Vec<PathBuf>,
}

enum ProductOutcome {
    Rendered {
        output_path: PathBuf,
        render_ms: u128,
        georeference: Option<rustwx_render::PanelGeoReference>,
        georeference_absent_reason: Option<String>,
    },
    Skipped(String),
}

/// Run a validated job synchronously on the caller's worker thread.
/// Cancellation is observed between products; an in-flight native render is
/// allowed to finish so renderer state is never abandoned halfway through a
/// file write.
pub fn run_batch_render(
    request: BatchRenderRequest,
    cancel: &AtomicBool,
    mut emit: impl FnMut(BatchRenderEvent),
) -> Result<BatchRenderSummary, String> {
    let started = Instant::now();
    validate_store_component("model", &request.model_slug)?;
    validate_store_component("run", &request.run_slug)?;
    if request.product_spec.len() > 16 * 1024 {
        return Err("product selection exceeds the 16 KiB GUI request limit".to_string());
    }
    let model = parse_model(&request.model_slug)?;
    validate_dimensions(&request)?;
    let cycle = resolve_cycle(&request)?;
    let source = match request.source {
        Some(source) => source,
        None => model_summary(model)
            .sources
            .first()
            .map(|descriptor| descriptor.id)
            .ok_or_else(|| format!("model {model} has no configured provenance source"))?,
    };

    let mut product_request =
        partition_products(&request.product_spec).map_err(|err| err.to_string())?;
    dedup(&mut product_request.direct);
    dedup(&mut product_request.derived);
    dedup(&mut product_request.generic);
    dedup(&mut product_request.windowed);

    let stored_hours = crate::render_all::windowed_store::stored_run_hours(
        &request.store_root,
        &request.model_slug,
        &request.run_slug,
    )
    .map_err(|err| err.to_string())?;
    // The "all" keyword only pulls the windowed lane onto runs whose axis
    // can serve it (more than one stored frame); explicit windowed slug
    // requests are left to fail loudly with the axis reason.
    if product_request.windowed_auto
        && !crate::render_all::windowed_store::windowed_axis_ready(
            &request.store_root,
            &request.model_slug,
            &request.run_slug,
        )
        .map_err(|err| err.to_string())?
    {
        product_request.windowed.clear();
    }
    // A keyword (`all`, `windowed`) stands for what the run can draw, so
    // it keeps only the windows the run's last stored frame closes; the
    // same cut `inspect_renderable_products` makes for the catalog.  An
    // explicit slug list is left alone and each window it names that the
    // run cannot close is refused per frame with the run's length.
    let window_frames = crate::render_all::windowed_store::stored_window_frames(
        &request.store_root,
        &request.model_slug,
        &request.run_slug,
    )
    .map_err(|err| err.to_string())?;
    if !product_request.strict {
        if let Some(last) = crate::render_all::windowed_store::last_window_hour(&window_frames) {
            product_request.windowed.retain(|slug| {
                HrrrWindowedProduct::from_slug(slug).is_some_and(|product| {
                    crate::render_all::windowed_store::window_fits_run(product, last)
                })
            });
        }
    }
    let hours = match request.hours {
        BatchHourScope::Current(hour) => vec![hour],
        BatchHourScope::AllStored => stored_hours.clone(),
    };
    if hours.is_empty() {
        return Err(format!(
            "run {}/{} has no stored hours",
            request.model_slug, request.run_slug
        ));
    }

    let mut per_hour = Vec::new();
    per_hour.extend(
        product_request
            .direct
            .iter()
            .cloned()
            .map(|slug| (BatchProductKind::Direct, slug)),
    );
    per_hour.extend(product_request.derived.iter().cloned().map(|slug| {
        let kind = if is_heavy_derived_recipe_slug(&slug) {
            BatchProductKind::Heavy
        } else {
            BatchProductKind::Derived
        };
        (kind, slug)
    }));
    per_hour.extend(
        product_request
            .generic
            .iter()
            .map(|name| (BatchProductKind::Generic, format!("var:{name}"))),
    );
    validate_work(
        &request,
        &hours,
        per_hour.len(),
        product_request.windowed.len(),
    )?;
    std::fs::create_dir_all(&request.out_dir).map_err(|err| {
        format!(
            "create output directory {}: {err}",
            request.out_dir.display()
        )
    })?;

    // Where the windowed products are drawn: every selected frame whose
    // lead is a whole hour.  A frame between hours ends no window, so a
    // whole run, or a keyword, plans none there: a 15-minute history
    // would otherwise add three skip lines per windowed product per hour.
    // A frame picked by itself for a NAMED window keeps its anchor, and
    // its skip line says why it has none.
    let window_frame = |slot: u16| {
        window_frames
            .iter()
            .find(|frame| frame.slot == slot)
            .copied()
    };
    let frame_label = |slot: u16| {
        window_frame(slot)
            .map(|frame| frame.label())
            .unwrap_or_else(|| format!("F{slot:03}"))
    };
    let picked_by_name =
        product_request.strict && matches!(request.hours, BatchHourScope::Current(_));
    let window_anchors: Vec<u16> = hours
        .iter()
        .copied()
        .filter(|&slot| {
            picked_by_name || window_frame(slot).is_none_or(|frame| frame.closes_windows())
        })
        .collect();

    let planned = hours
        .len()
        .checked_mul(per_hour.len())
        .and_then(|per_frame| {
            window_anchors
                .len()
                .checked_mul(product_request.windowed.len())
                .and_then(|windowed| per_frame.checked_add(windowed))
        })
        .ok_or_else(|| "batch work-item count overflowed usize".to_string())?;
    let all_products = per_hour
        .iter()
        .map(|(_, slug)| slug.clone())
        .chain(product_request.windowed.iter().cloned())
        .collect::<Vec<_>>();
    emit(BatchRenderEvent::Started {
        planned_items: planned,
        hours: hours.clone(),
        products: all_products,
        output_dir: request.out_dir.clone(),
    });

    let mut summary = BatchRenderSummary {
        planned,
        ..BatchRenderSummary::default()
    };
    let mut completed = 0usize;
    let mut native_domain = None;
    let mut planned_canvas = None;
    // Say-once advisory keys, for the whole job rather than per hour: a
    // warning about a subtitle that does not fit is one fact about this
    // gallery, not one fact per stored hour.
    let mut advised: HashSet<String> = HashSet::new();

    // One pool for the whole job, built before the first hour so worker
    // threads are created once rather than per stored hour.  It is a
    // PRIVATE pool and the product fan-out `install`s into it, which is
    // what keeps the total thread count bounded: the met kernels inside
    // `rustwx-calc` and `rustwx-products` are themselves rayon-parallel,
    // and left on the process-wide pool they would stack a second
    // full-width fan-out on top of this one.
    // Asked once, so the width and the line explaining the width cannot
    // disagree about how much memory the box had.
    let available_bytes = host_memory::available_bytes();
    let worker_count = product_worker_count_within(per_hour.len(), available_bytes);
    advise_if_memory_bound(worker_count, per_hour.len(), available_bytes);
    let processors = render_processor_budget();
    let pool = if processors > 1 {
        Some(
            rayon::ThreadPoolBuilder::new()
                .num_threads(processors)
                .thread_name(|index| format!("rustwx-batch-render-{index}"))
                .build()
                .map_err(|err| format!("build the batch render worker pool: {err}"))?,
        )
    } else {
        None
    };

    'hours: for (hour_index, &hour) in hours.iter().enumerate() {
        if cancel.load(Ordering::Relaxed) {
            break;
        }
        emit(BatchRenderEvent::HourStarted {
            hour,
            index: hour_index + 1,
            total: hours.len(),
        });
        let store = match protected(|| {
            StoreFieldSource::open(
                &request.store_root,
                &request.model_slug,
                &request.run_slug,
                hour,
            )
            .map_err(|err| err.to_string())
        }) {
            Ok(store) => store,
            Err(error) => {
                for (kind, slug) in &per_hour {
                    if cancel.load(Ordering::Relaxed) {
                        break 'hours;
                    }
                    emit(BatchRenderEvent::ItemStarted {
                        hour: Some(hour),
                        slug: slug.clone(),
                        kind: *kind,
                        completed,
                        total: planned,
                    });
                    completed += 1;
                    summary.failed += 1;
                    emit(BatchRenderEvent::ItemFailed {
                        hour: Some(hour),
                        slug: slug.clone(),
                        error: format!("open stored hour: {error}"),
                        completed,
                        total: planned,
                    });
                }
                continue;
            }
        };

        let domain = resolve_render_domain(
            &request.domain,
            request.native_domain_slug.as_deref(),
            &store,
            &mut native_domain,
        )?;
        let mut config = render_config(&request, model, &cycle, source, domain);
        apply_planned_canvas(&mut config, &store, &mut planned_canvas)?;

        // Full-domain requests carry multiple mesh-sized fields and geometry.
        // Bound concurrent products by grid size, independently of the pool
        // whose threads also execute disjoint rows within each product.
        let product_bytes = (store.grid_coordinates().0.len() as u64)
            .saturating_mul(320)
            .max(PRODUCT_WORKER_BYTES);
        let product_workers = product_worker_count_within_bytes(
            per_hour.len(),
            host_memory::available_bytes(),
            product_bytes,
        );
        render_hour_items(
            pool.as_ref(),
            product_workers,
            &config,
            &store,
            hour,
            hour_index,
            hours[0],
            &per_hour,
            cancel,
            planned,
            &mut completed,
            &mut summary,
            &mut advised,
            &mut emit,
        );
        if cancel.load(Ordering::Relaxed) {
            break 'hours;
        }
    }

    // Windowed products follow the same selected anchors as ordinary
    // products, less the frames between hours (`window_anchors`). The
    // complete earlier store remains baseline context; a selected hour
    // must never silently become the latest stored hour.
    for &anchor_hour in &window_anchors {
        if cancel.load(Ordering::Relaxed) || product_request.windowed.is_empty() {
            break;
        }
        let store = protected(|| {
            StoreFieldSource::open(
                &request.store_root,
                &request.model_slug,
                &request.run_slug,
                anchor_hour,
            )
            .map_err(|err| err.to_string())
        });
        match store {
            Ok(store) => {
                let domain = resolve_render_domain(
                    &request.domain,
                    request.native_domain_slug.as_deref(),
                    &store,
                    &mut native_domain,
                )?;
                let mut config = render_config(&request, model, &cycle, source, domain);
                apply_planned_canvas(&mut config, &store, &mut planned_canvas)?;
                for slug in &product_request.windowed {
                    emit(BatchRenderEvent::ItemStarted {
                        hour: Some(anchor_hour),
                        slug: slug.clone(),
                        kind: BatchProductKind::Windowed,
                        completed,
                        total: planned,
                    });
                }
                // One multi-product call is essential: windowed_store can
                // read each (hour, source plane) once and fold it into every
                // selected accumulator. Calling once per slug would turn a
                // 40-product export into 40 full passes over the run.
                let outcomes = protected(|| {
                    render_windowed_items(
                        &config,
                        &store,
                        &request.store_root,
                        &request.model_slug,
                        &request.run_slug,
                        &product_request.windowed,
                        anchor_hour,
                    )
                });
                match outcomes {
                    Ok(outcomes) => {
                        for (slug, outcome) in outcomes {
                            completed += 1;
                            match outcome {
                                ProductOutcome::Rendered {
                                    output_path,
                                    render_ms,
                                    georeference,
                                    georeference_absent_reason,
                                } => {
                                    summary.rendered += 1;
                                    summary.outputs.push(output_path.clone());
                                    emit(BatchRenderEvent::ItemRendered {
                                        hour: Some(anchor_hour),
                                        slug,
                                        output_path,
                                        render_ms,
                                        georeference,
                                        georeference_absent_reason,
                                        completed,
                                        total: planned,
                                    });
                                }
                                ProductOutcome::Skipped(reason) => {
                                    summary.skipped += 1;
                                    emit(BatchRenderEvent::ItemSkipped {
                                        hour: Some(anchor_hour),
                                        slug,
                                        reason: format!("{}: {reason}", frame_label(anchor_hour)),
                                        completed,
                                        total: planned,
                                    });
                                }
                            }
                        }
                    }
                    Err(error) => {
                        for slug in &product_request.windowed {
                            completed += 1;
                            summary.failed += 1;
                            emit(BatchRenderEvent::ItemFailed {
                                hour: Some(anchor_hour),
                                slug: slug.clone(),
                                error: format!("{}: {error}", frame_label(anchor_hour)),
                                completed,
                                total: planned,
                            });
                        }
                    }
                }
            }
            Err(error) => {
                for slug in &product_request.windowed {
                    if cancel.load(Ordering::Relaxed) {
                        break;
                    }
                    emit(BatchRenderEvent::ItemStarted {
                        hour: Some(anchor_hour),
                        slug: slug.clone(),
                        kind: BatchProductKind::Windowed,
                        completed,
                        total: planned,
                    });
                    completed += 1;
                    summary.failed += 1;
                    emit(BatchRenderEvent::ItemFailed {
                        hour: Some(anchor_hour),
                        slug: slug.clone(),
                        error: format!("open window anchor {}: {error}", frame_label(anchor_hour)),
                        completed,
                        total: planned,
                    });
                }
            }
        }
    }

    summary.cancelled = cancel.load(Ordering::Relaxed);
    summary.elapsed_ms = started.elapsed().as_millis();
    emit(BatchRenderEvent::Finished(summary.clone()));
    Ok(summary)
}

/// How many of an hour's products render at the same time.
///
/// The CPU pool uses the available processor budget. Concurrent products are
/// limited separately because decoded planes and projected geometry grow with
/// the input grid. The per-product allowance is the larger of the small-grid
/// floor and 320 bytes per grid point, including temporary field/mesh copies.
/// Half the currently available memory is shared between live render slots.
/// Explicit product limits retain their existing override semantics.
/// `RUSTWX_BATCH_RENDER_THREADS` limits concurrent products only; nested
/// projection and raster work can still use the CPU pool.
fn product_worker_count_within(item_count: usize, available_bytes: Option<u64>) -> usize {
    product_worker_count_within_bytes(item_count, available_bytes, PRODUCT_WORKER_BYTES)
}

fn product_worker_count_within_bytes(
    item_count: usize,
    available_bytes: Option<u64>,
    product_bytes: u64,
) -> usize {
    if item_count <= 1 {
        return 1;
    }
    let override_threads = std::env::var("RUSTWX_BATCH_RENDER_THREADS")
        .ok()
        .or_else(|| std::env::var("RUSTWX_RENDER_THREADS").ok())
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|&value| value > 0);
    if let Some(width) = override_threads {
        return width.min(item_count);
    }
    let live_slots = live_render_slots();
    let detected = render_processor_budget();
    let affordable = match available_bytes {
        Some(bytes) => {
            let budget = (bytes as f64 * BUDGET_FRACTION) as u64
                / live_slots.unwrap_or(1) as u64;
            let for_workers = budget.saturating_sub(PRODUCT_LOOP_BASE_BYTES);
            usize::try_from(for_workers / product_bytes)
                .unwrap_or(usize::MAX)
                .max(1)
        }
        None => usize::MAX,
    };
    detected.min(item_count).min(affordable)
}

/// Simultaneous live frame imports share the existing CPU and memory budget.
fn live_render_slots() -> Option<usize> {
    std::env::var("RUSTWX_LIVE_RENDER_SLOTS")
        .ok()
        .and_then(|value| value.parse().ok())
        .filter(|value| *value > 0)
}

/// How many live frames share the processors now: the frames drawing as this
/// one starts (`RUSTWX_LIVE_RENDER_DRAWING`, never more than the slots), else
/// the slots. THE BREAKAGE: divided by the planned three slots, a lone live
/// frame on an eight-processor box drew its products two at a time while
/// five processors sat idle (2.25 km / 750 m HRRR nest, 2026-10-24:
/// 1.4 s of a 4.6 s frame render at two cores).
fn live_processor_share() -> Option<usize> {
    let drawing = std::env::var("RUSTWX_LIVE_RENDER_DRAWING")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|value| *value > 0);
    match (drawing, live_render_slots()) {
        (Some(drawing), Some(slots)) => Some(drawing.min(slots)),
        (drawing, slots) => drawing.or(slots),
    }
}

/// One CPU budget for nested image work and the product scheduler.
fn render_processor_budget() -> usize {
    let processors = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let requested = ["RUSTWX_RENDER_THREADS", "RAYON_NUM_THREADS"]
        .iter()
        .find_map(|key| {
            std::env::var(key)
                .ok()
                .and_then(|v| v.trim().parse::<usize>().ok())
                .filter(|&n| n > 0)
        });
    default_product_worker_count(
        requested.unwrap_or(processors).min(processors),
        live_processor_share(),
    )
}

fn default_product_worker_count(processors: usize, live_slots: Option<usize>) -> usize {
    match live_slots {
        Some(slots) => processors.saturating_sub(1) / slots.max(1),
        None => processors,
    }
    .max(1)
}

/// Fixed cost of the loop itself: the store handle, the basemap, the
/// process.  Measured as the peak resident set of the width-1 render
/// phase, where the ladder starts.
const PRODUCT_LOOP_BASE_BYTES: u64 = 470 * 1024 * 1024 + rustwx_render::PROJECTED_MAP_CACHE_BYTES;

/// What one more concurrent product costs at peak: the slope of that
/// ladder, rounded UP past the steeper of the two grids measured,
/// because a constant that under-estimates is the failure it exists to
/// prevent.
const PRODUCT_WORKER_BYTES: u64 = 96 * 1024 * 1024;

/// The share of free memory this loop will plan to occupy.
const BUDGET_FRACTION: f64 = 0.5;

/// Say so, once, when free memory rather than the box's cores is what
/// decided the width.
///
/// Silence here would be the defect: a run that is four times slower than
/// the same box managed yesterday, with nothing on the console to say
/// that the memory it had at the time is the reason.
fn advise_if_memory_bound(width: usize, item_count: usize, available_bytes: Option<u64>) {
    if item_count <= 1 {
        return;
    }
    if std::env::var_os("RUSTWX_BATCH_RENDER_THREADS").is_some()
        || std::env::var_os("RUSTWX_RENDER_THREADS").is_some()
    {
        return;
    }
    let uncapped = render_processor_budget().min(item_count);
    if width >= uncapped {
        return;
    }
    let free_mib = available_bytes.unwrap_or(0) / (1024 * 1024);
    advisory::advise(format!(
        "RENDER_WIDTH\t{width}\tfree memory ({free_mib} MiB) rather than core count \
         set the width; {uncapped} products would have rendered at once on a box with \
         room for them. Set RUSTWX_BATCH_RENDER_THREADS to override."
    ));
}

/// What one worker reports back about one product: `None` when the job
/// was never attempted because cancellation was already set when the
/// worker picked it up, plus everything the render advised on stderr
/// while it ran, held so the caller can print it in catalog order.
type ProductReport = (usize, Option<Result<ProductOutcome, String>>, Vec<Advisory>);

/// Render every product of one stored hour and fold the results into
/// `summary`, in parallel across the job's worker pool.
///
/// Products are independent renders.  Each one reads the hour through
/// `&StoreFieldSource` -- an mmap or a RAM buffer plus lookup maps built
/// once at open, with no interior mutability anywhere on the type -- and
/// writes its own output path, so there is no state to share and none is
/// shared.  The one process-wide value the render lanes read, the
/// initial-condition disclosure, is written by the import stage before
/// any of this runs and is read-only here.
///
/// ORDER, not timing, is what the event stream promises.  Results are
/// held back and reported in catalog order on the CALLER's thread, so a
/// parallel run emits exactly the event sequence -- and `rw_wrfbatch`
/// exactly the stdout -- a serial run emits, and `summary.outputs` keeps
/// its deterministic order.  It also keeps `emit` single-threaded, so
/// callers may still pass a non-`Send` closure (the egui lane does).
///
/// STDERR rides the same hold.  A render advises about itself while it
/// runs (a generic colortable fallback, a clipped subtitle), and written
/// straight out of a worker those lines arrive in completion order --
/// which would leave one of the two console streams reproducible and the
/// other not.  Each product renders inside an [`advisory::hold`] and its
/// lines are printed where the product sits in the catalog, between its
/// own start and outcome events, exactly where the serial loop put them.
/// `advised` carries the say-once keys across the WHOLE job so a line
/// that should be said once is still said once, by the first product in
/// catalog order rather than by whichever worker won a race.
#[allow(clippy::too_many_arguments)]
fn render_hour_items(
    pool: Option<&rayon::ThreadPool>,
    product_workers: usize,
    config: &StoreRenderConfig,
    store: &StoreFieldSource,
    hour: u16,
    hour_index: usize,
    first_hour: u16,
    per_hour: &[(BatchProductKind, String)],
    cancel: &AtomicBool,
    planned: usize,
    completed: &mut usize,
    summary: &mut BatchRenderSummary,
    advised: &mut HashSet<String>,
    emit: &mut dyn FnMut(BatchRenderEvent),
) {
    // A grid property, not a forecast: render it on the first selected
    // hour and say so for the rest, rather than writing one identical
    // image per lead.  Announced, because a silent drop and a deliberate
    // single render look the same in a directory listing.  Decided HERE,
    // not inside a worker: it is a pure function of kind, slug and hour
    // index, and resolving it up front means no thread can ever be the
    // reason a static field was or was not skipped.
    let mut work: Vec<(usize, BatchProductKind, &str)> = Vec::with_capacity(per_hour.len());
    let mut static_skip = vec![false; per_hour.len()];
    for (index, (kind, slug)) in per_hour.iter().enumerate() {
        if matches!(kind, BatchProductKind::Direct)
            && hour_index > 0
            && direct_recipe_is_time_invariant(slug)
        {
            static_skip[index] = true;
        } else {
            work.push((index, *kind, slug.as_str()));
        }
    }

    let work = &work;
    std::thread::scope(|scope| {
        // The fan-out feeds results back over a channel; the caller's
        // thread does every emit.  With no pool there is no channel and
        // no extra thread at all: the products render RIGHT HERE, which
        // is where `rustwx-render` keeps its per-thread projection and
        // static-base caches -- handing the work to a scratch thread
        // would cool them once per stored hour.
        let receiver = pool.map(|pool| {
            let (sender, receiver) = std::sync::mpsc::channel::<ProductReport>();
            scope.spawn(move || {
                pool.install(move || {
                    let next = std::sync::atomic::AtomicUsize::new(0);
                    (0..product_workers.min(work.len()))
                        .into_par_iter()
                        .for_each_with(sender, |sender: &mut Sender<ProductReport>, _| {
                            while let Some(&(index, kind, slug)) =
                                work.get(next.fetch_add(1, Ordering::Relaxed))
                            {
                                // Cancellation is observed when a product is
                                // PICKED UP.  The renders already in flight
                                // (at most one per worker) are allowed to
                                // finish so renderer state is never abandoned
                                // halfway through a file write.
                                let report = if cancel.load(Ordering::Relaxed) {
                                    (index, None, Vec::new())
                                } else {
                                    let (outcome, advice) = advisory::hold(|| {
                                        protected(|| {
                                            render_hour_item(config, store, hour, kind, slug)
                                        })
                                    });
                                    (index, Some(outcome), advice)
                                };
                                let _ = sender.send(report);
                            }
                        });
                });
            });
            receiver
        });

        // The in-order report, on the thread that owns `emit`.
        let mut pending: HashMap<usize, (Option<Result<ProductOutcome, String>>, Vec<Advisory>)> =
            HashMap::new();
        for (index, (kind, slug)) in per_hour.iter().enumerate() {
            if static_skip[index] {
                emit(BatchRenderEvent::ItemStarted {
                    hour: Some(hour),
                    slug: slug.clone(),
                    kind: *kind,
                    completed: *completed,
                    total: planned,
                });
                *completed += 1;
                summary.skipped += 1;
                emit(BatchRenderEvent::ItemSkipped {
                    hour: Some(hour),
                    slug: slug.clone(),
                    reason: format!(
                        "static field: does not vary with forecast time, rendered once for this domain at F{first_hour:03}"
                    ),
                    completed: *completed,
                    total: planned,
                });
                continue;
            }
            let (outcome, advice) = match &receiver {
                Some(receiver) => {
                    let report = loop {
                        if let Some(report) = pending.remove(&index) {
                            break report;
                        }
                        match receiver.recv() {
                            Ok((reported, outcome, advice)) => {
                                pending.insert(reported, (outcome, advice));
                            }
                            // Every worker is gone and this product never
                            // reported: cancelled before it was picked up.
                            Err(_) => break (None, Vec::new()),
                        }
                    };
                    let (Some(outcome), advice) = report else {
                        continue;
                    };
                    (outcome, advice)
                }
                None => {
                    if cancel.load(Ordering::Relaxed) {
                        break;
                    }
                    // Held on this path too, so the no-pool loop and the
                    // pool one put the same lines in the same places.
                    advisory::hold(|| {
                        protected(|| render_hour_item(config, store, hour, *kind, slug))
                    })
                }
            };
            emit(BatchRenderEvent::ItemStarted {
                hour: Some(hour),
                slug: slug.clone(),
                kind: *kind,
                completed: *completed,
                total: planned,
            });
            for held in advice {
                advisory::drain_one(held, advised);
            }
            *completed += 1;
            match outcome {
                Ok(ProductOutcome::Rendered {
                    output_path,
                    render_ms,
                    georeference,
                    georeference_absent_reason,
                }) => {
                    summary.rendered += 1;
                    summary.outputs.push(output_path.clone());
                    emit(BatchRenderEvent::ItemRendered {
                        hour: Some(hour),
                        slug: slug.clone(),
                        output_path,
                        render_ms,
                        georeference,
                        georeference_absent_reason,
                        completed: *completed,
                        total: planned,
                    });
                }
                Ok(ProductOutcome::Skipped(reason)) => {
                    summary.skipped += 1;
                    emit(BatchRenderEvent::ItemSkipped {
                        hour: Some(hour),
                        slug: slug.clone(),
                        reason,
                        completed: *completed,
                        total: planned,
                    });
                }
                Err(error) => {
                    summary.failed += 1;
                    emit(BatchRenderEvent::ItemFailed {
                        hour: Some(hour),
                        slug: slug.clone(),
                        error,
                        completed: *completed,
                        total: planned,
                    });
                }
            }
        }
    });
}

fn render_hour_item(
    config: &StoreRenderConfig,
    store: &StoreFieldSource,
    hour: u16,
    kind: BatchProductKind,
    slug: &str,
) -> Result<ProductOutcome, String> {
    if kind == BatchProductKind::Generic {
        let variable = slug.strip_prefix("var:").ok_or_else(|| {
            format!("internal generic product identity {slug:?} is missing the 'var:' prefix")
        })?;
        // Skipped by name on a frame that does not store it, as the direct
        // and derived lanes skip theirs: a catalog of several frames lists
        // a variable any of them stores
        // ([`inspect_renderable_products_over`]), and one frame without it
        // failed the whole invocation, every other frame's pictures with it.
        if store.surface_variable(variable).is_none() {
            return Ok(ProductOutcome::Skipped(format!(
                "not stored in this frame: 2-D variable {variable:?}"
            )));
        }
        let rendered = crate::render_all::store_render::render_generic_store_variable(
            store, config, hour, variable,
        )
        .map_err(|err| err.to_string())?;
        return Ok(ProductOutcome::Rendered {
            output_path: rendered.output_path,
            render_ms: rendered.total_ms,
            georeference: rendered.georeference,
            georeference_absent_reason: rendered.georeference_absent_reason,
        });
    }
    let single = vec![slug.to_string()];
    let empty: &[String] = &[];
    let (direct, derived) = if kind == BatchProductKind::Direct {
        (single.as_slice(), empty)
    } else {
        (empty, single.as_slice())
    };
    let mut outcome = render_hour_products(config, store, hour, direct, derived, empty, None)
        .map_err(|err| err.to_string())?;
    if let Some(rendered) = outcome.rendered.pop() {
        return Ok(ProductOutcome::Rendered {
            output_path: rendered.output_path,
            render_ms: rendered.total_ms,
            georeference: rendered.georeference,
            georeference_absent_reason: rendered.georeference_absent_reason,
        });
    }
    if let Some(skipped) = outcome.skipped.pop() {
        return Ok(ProductOutcome::Skipped(skipped.reason));
    }
    Err(format!(
        "renderer returned no image and no blocker for '{slug}'"
    ))
}

fn render_windowed_items(
    config: &StoreRenderConfig,
    store: &StoreFieldSource,
    store_root: &Path,
    model_slug: &str,
    run_slug: &str,
    requested: &[String],
    anchor_hour: u16,
) -> Result<Vec<(String, ProductOutcome)>, String> {
    let Some(mut outcome) = render_windowed_products(
        config,
        store,
        store_root,
        model_slug,
        run_slug,
        requested,
        anchor_hour,
        false,
    )
    .map_err(|err| err.to_string())?
    else {
        return Ok(requested
            .iter()
            .cloned()
            .map(|slug| {
                (
                    slug,
                    ProductOutcome::Skipped(
                        "windowed renderer found no usable multi-hour window".to_string(),
                    ),
                )
            })
            .collect());
    };
    let mut by_slug = HashMap::new();
    for rendered in outcome.rendered.drain(..) {
        by_slug.insert(
            rendered.slug,
            ProductOutcome::Rendered {
                output_path: rendered.output_path,
                render_ms: rendered.total_ms,
                georeference: rendered.georeference,
                georeference_absent_reason: rendered.georeference_absent_reason,
            },
        );
    }
    for blocked in outcome.blocked.drain(..) {
        by_slug.insert(blocked.slug, ProductOutcome::Skipped(blocked.reason));
    }
    Ok(requested
        .iter()
        .cloned()
        .map(|slug| {
            let outcome = by_slug.remove(&slug).unwrap_or_else(|| {
                ProductOutcome::Skipped(
                    "windowed renderer returned no image and no blocker".to_string(),
                )
            });
            (slug, outcome)
        })
        .collect())
}

fn render_config(
    request: &BatchRenderRequest,
    model: ModelId,
    cycle: &CycleSpec,
    source: SourceId,
    domain: DomainSpec,
) -> StoreRenderConfig {
    StoreRenderConfig {
        model,
        date_yyyymmdd: cycle.date_yyyymmdd.clone(),
        cycle_utc: cycle.hour_utc,
        source,
        subtitle_spacing: request.subtitle_spacing.clone(),
        source_label: request.source_label.clone(),
        title_provenance: request.title_provenance.clone(),
        domain,
        out_dir: request.out_dir.clone(),
        contour_mode: Default::default(),
        native_fill_level_multiplier: 1,
        output_width: request.output_width,
        output_height: request.output_height,
        png_compression: PngCompressionMode::Fast,
        place_label_overlay: None,
        geographic_overlays: request.geographic_overlays.clone(),
        panel_annotations: request.panel_annotations.clone(),
    }
}

/// Under auto layout, the canvas every product of this run's domain is
/// drawn on: planned once from the stored grid's own projected shape and
/// kept for every hour, so the whole run shares one frame size.  Fixed
/// layout leaves the request's pixel size alone.
fn apply_planned_canvas(
    config: &mut StoreRenderConfig,
    store: &StoreFieldSource,
    planned: &mut Option<(u32, u32)>,
) -> Result<(), String> {
    if !rustwx_render::auto_layout_active() {
        return Ok(());
    }
    if planned.is_none() {
        let (latitudes, longitudes) = store.grid_coordinates();
        let plan = rustwx_products::direct::plan_canvas_for_grid(
            latitudes,
            longitudes,
            store.projection(),
            config.domain.bounds,
        )
        .map_err(|err| format!("plan the canvas for the {} grid: {err}", config.domain.slug))?;
        *planned = plan.map(|plan| (plan.canvas_w, plan.canvas_h));
    }
    if let Some((width, height)) = *planned {
        config.output_width = width;
        config.output_height = height;
    }
    if config.place_label_overlay.is_none() {
        config.place_label_overlay =
            rustwx_products::direct::planned_place_labels(config.domain.bounds);
    }
    Ok(())
}

fn parse_model(model_slug: &str) -> Result<ModelId, String> {
    model_slug.parse::<ModelId>().map_err(|err| {
        format!("store model '{model_slug}' has no production renderer identity: {err}")
    })
}

fn validate_store_component(label: &str, value: &str) -> Result<(), String> {
    let mut components = Path::new(value).components();
    let valid = matches!(components.next(), Some(std::path::Component::Normal(_)))
        && components.next().is_none();
    if valid {
        Ok(())
    } else {
        Err(format!(
            "store {label} must be one non-empty path component, got '{value}'"
        ))
    }
}

fn validate_dimensions(request: &BatchRenderRequest) -> Result<(), String> {
    if request.output_width < 320 || request.output_height < 240 {
        return Err("output dimensions must be at least 320 x 240".to_string());
    }
    if request.output_width > request.limits.max_output_width
        || request.output_height > request.limits.max_output_height
    {
        return Err(format!(
            "output {}x{} exceeds the GUI ceiling {}x{}",
            request.output_width,
            request.output_height,
            request.limits.max_output_width,
            request.limits.max_output_height
        ));
    }
    let pixels = u64::from(request.output_width) * u64::from(request.output_height);
    if pixels > request.limits.max_output_pixels {
        return Err(format!(
            "output has {pixels} pixels; GUI ceiling is {}",
            request.limits.max_output_pixels
        ));
    }
    Ok(())
}

fn validate_work(
    request: &BatchRenderRequest,
    hours: &[u16],
    per_hour_products: usize,
    windowed_products: usize,
) -> Result<(), String> {
    if hours.len() > request.limits.max_hours {
        return Err(format!(
            "{} hours selected; GUI ceiling is {} (split the run into smaller jobs)",
            hours.len(),
            request.limits.max_hours
        ));
    }
    let products = per_hour_products
        .checked_add(windowed_products)
        .ok_or_else(|| "batch product count overflowed usize".to_string())?;
    if products > request.limits.max_products_per_hour {
        return Err(format!(
            "{products} per-hour products selected; GUI ceiling is {}",
            request.limits.max_products_per_hour
        ));
    }
    let work = hours
        .len()
        .checked_mul(products)
        .ok_or_else(|| "batch work-item count overflowed usize".to_string())?;
    if work == 0 {
        return Err("select at least one product".to_string());
    }
    if work > request.limits.max_work_items {
        return Err(format!(
            "{work} product-hours selected; GUI ceiling is {} (split the job)",
            request.limits.max_work_items
        ));
    }
    Ok(())
}

fn resolve_cycle(request: &BatchRenderRequest) -> Result<CycleSpec, String> {
    let inferred = infer_run_cycle(&request.run_slug);
    let date = request
        .date_yyyymmdd
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .or_else(|| inferred.as_ref().map(|(date, _)| date.clone()))
        .ok_or_else(|| {
            format!(
                "cannot infer an init date from run '{}'; provide YYYYMMDD",
                request.run_slug
            )
        })?;
    let hour = request
        .cycle_utc
        .or_else(|| inferred.map(|(_, hour)| hour))
        .ok_or_else(|| {
            format!(
                "cannot infer a cycle hour from run '{}'; provide 0-23",
                request.run_slug
            )
        })?;
    CycleSpec::new(date, hour).map_err(|err| err.to_string())
}

/// Infer standard `YYYYMMDD_CCz` and local-WRF `...YYYYMMDD_HHMMSS` names.
pub fn infer_run_cycle(run_slug: &str) -> Option<(String, u8)> {
    let bytes = run_slug.as_bytes();
    if bytes.len() < 10 {
        return None;
    }
    for start in 0..=bytes.len().saturating_sub(8) {
        let date_bytes = &bytes[start..start + 8];
        if !date_bytes.iter().all(|byte| byte.is_ascii_digit()) {
            continue;
        }
        let Ok(date) = std::str::from_utf8(date_bytes) else {
            continue;
        };
        if CycleSpec::new(date.to_string(), 0).is_err() {
            continue;
        }
        let search_end = (start + 16).min(bytes.len());
        for hour_start in start + 8..search_end.saturating_sub(1) {
            let pair = &bytes[hour_start..hour_start + 2];
            if !pair.iter().all(|byte| byte.is_ascii_digit()) {
                continue;
            }
            let hour = (pair[0] - b'0') * 10 + (pair[1] - b'0');
            if hour < 24 {
                return Some((date.to_string(), hour));
            }
        }
    }
    None
}

fn explicit_domain(domain: &BatchRenderDomain) -> Result<DomainSpec, String> {
    let BatchRenderDomain::Bounds {
        slug,
        west,
        east,
        south,
        north,
    } = domain
    else {
        return Err("internal error: native domain was not resolved".to_string());
    };
    let (west, east, south, north) = (*west, *east, *south, *north);
    if ![west, east, south, north]
        .iter()
        .all(|value| value.is_finite())
    {
        return Err("custom domain bounds must be finite".to_string());
    }
    let raw_lon_span = (east - west).abs();
    if raw_lon_span == 0.0 || raw_lon_span > 360.0 || south >= north {
        return Err(
            "custom domain needs distinct west/east bounds no more than 360 degrees apart and south < north"
                .to_string(),
        );
    }
    if south < -90.0 || north > 90.0 {
        return Err("custom domain latitude must stay inside -90..90".to_string());
    }
    let slug = safe_slug(slug, "custom");
    Ok(DomainSpec::new(slug, (west, east, south, north)))
}

fn resolve_render_domain(
    requested: &BatchRenderDomain,
    native_slug: Option<&str>,
    store: &StoreFieldSource,
    native_cache: &mut Option<DomainSpec>,
) -> Result<DomainSpec, String> {
    match requested {
        BatchRenderDomain::NativeGrid => {
            if native_cache.is_none() {
                let mut domain = native_grid_domain(store)?;
                if let Some(slug) = native_slug {
                    domain.slug = safe_slug(slug, "native_grid");
                }
                *native_cache = Some(domain);
            }
            native_cache
                .clone()
                .ok_or_else(|| "native render domain was not cached".to_string())
        }
        custom => explicit_domain(custom),
    }
}

fn native_grid_domain(store: &StoreFieldSource) -> Result<DomainSpec, String> {
    let (latitudes, longitudes) = store.grid_coordinates();
    native_grid_domain_from_coordinates(latitudes, longitudes)
}

/// The native-grid render domain of a bare lat/lon mesh.
///
/// gpuwm divergence (VENDOR.md): a pure extraction of
/// [`native_grid_domain`]'s body so the sibling binaries that render
/// ensemble reductions and observation grids frame their panels on the
/// EXACT same bounds arithmetic -- the antimeridian-aware longitude
/// choice, the degenerate-extent padding, the pole clamp -- rather than a
/// second, subtly different copy.  Two framings of one grid is how a
/// member-mean panel stops overlaying the deterministic panel it is meant
/// to be compared against.
pub fn native_grid_domain_from_coordinates(
    latitudes: &[f32],
    longitudes: &[f32],
) -> Result<DomainSpec, String> {
    if latitudes.is_empty() || latitudes.len() != longitudes.len() {
        return Err(format!(
            "grid.rwg coordinate lengths are invalid (lat {}, lon {})",
            latitudes.len(),
            longitudes.len()
        ));
    }
    let mut south = f64::INFINITY;
    let mut north = f64::NEG_INFINITY;
    let mut west_180 = f64::INFINITY;
    let mut east_180 = f64::NEG_INFINITY;
    let mut west_360 = f64::INFINITY;
    let mut east_360 = f64::NEG_INFINITY;
    for (&lat, &lon) in latitudes.iter().zip(longitudes.iter()) {
        if !lat.is_finite() || !lon.is_finite() {
            continue;
        }
        let lat = f64::from(lat);
        let lon = f64::from(lon);
        south = south.min(lat);
        north = north.max(lat);
        let lon_180 = (lon + 180.0).rem_euclid(360.0) - 180.0;
        let lon_360 = lon.rem_euclid(360.0);
        west_180 = west_180.min(lon_180);
        east_180 = east_180.max(lon_180);
        west_360 = west_360.min(lon_360);
        east_360 = east_360.max(lon_360);
    }
    if ![south, north, west_180, east_180, west_360, east_360]
        .iter()
        .all(|value| value.is_finite())
    {
        return Err("grid.rwg has no finite latitude/longitude coordinates".to_string());
    }
    let (mut west, mut east) = if east_180 - west_180 <= east_360 - west_360 {
        (west_180, east_180)
    } else {
        (west_360, east_360)
    };
    if (east - west).abs() < 0.01 {
        west -= 0.5;
        east += 0.5;
    } else {
        west -= 0.05;
        east += 0.05;
    }
    if (north - south).abs() < 0.01 {
        south -= 0.5;
        north += 0.5;
    } else {
        south -= 0.05;
        north += 0.05;
    }
    south = south.max(-90.0);
    north = north.min(90.0);
    Ok(DomainSpec::new("native_grid", (west, east, south, north)))
}

fn dedup(values: &mut Vec<String>) {
    let mut seen = HashSet::new();
    values.retain(|value| seen.insert(value.clone()));
}

/// A domain slug is only ever a filename *component* (never a path
/// element), so `.` is admissible and essential: a 3:1 nest of a 12 km
/// parent is 1.333 km, and `d03-1_333km` would read as a typo.  Every
/// separator character still collapses to `_`.
fn safe_slug(value: &str, fallback: &str) -> String {
    let slug = value
        .trim()
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '-' | '_' | '.') {
                character
            } else {
                '_'
            }
        })
        .collect::<String>();
    if slug.is_empty() {
        fallback.to_string()
    } else {
        slug
    }
}

fn protected<T>(run: impl FnOnce() -> Result<T, String>) -> Result<T, String> {
    match catch_unwind(AssertUnwindSafe(run)) {
        Ok(result) => result,
        Err(payload) => Err(format!("renderer panicked: {}", panic_message(payload))),
    }
}

fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "unknown panic payload".to_string()
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::atomic::AtomicBool;

    use rustwx_core::{CanonicalField, FieldSelector, GridShape, LatLonGrid, SelectedField2D};
    use rw_store::ingest::{DerivedFieldInput, write_hour_from_fields_with_derived};
    use rw_store::run::RwsRunManifest;
    use rw_store::writer::HourWriter;

    use super::*;

    #[test]
    fn infers_operational_and_local_wrf_run_names() {
        assert_eq!(
            infer_run_cycle("20260608_00z"),
            Some(("20260608".to_string(), 0))
        );
        assert_eq!(
            infer_run_cycle("local_wrf_20110524_180000"),
            Some(("20110524".to_string(), 18))
        );
        assert_eq!(infer_run_cycle("local_wrf_no_time"), None);
    }

    fn test_dir(label: &str) -> PathBuf {
        // Parallel tests read one clock tick on the 2.8.6 windows-2025 runner and collided on this name; the counter keeps each call distinct.
        static NEXT_SCRATCH: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "rusty-weather-batch-{label}-{}-{unique}-{}",
            std::process::id(),
            NEXT_SCRATCH.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn batch_grid() -> LatLonGrid {
        LatLonGrid::new(
            GridShape::new(4, 3).unwrap(),
            (0..12).map(|cell| 35.0 + (cell / 4) as f32).collect(),
            (0..12).map(|cell| -100.0 + (cell % 4) as f32).collect(),
        )
        .unwrap()
    }

    fn write_windowed_store(root: &std::path::Path, run: &str, hours: &[u16]) {
        let grid = batch_grid();
        for &hour in hours {
            let temperature = SelectedField2D::new(
                FieldSelector::height_agl(CanonicalField::Temperature, 2),
                "K",
                grid.clone(),
                vec![280.0; 12],
            )
            .unwrap();
            // A cumulative WRF-style plane whose trailing increment changes
            // every hour. Reusing the final window cannot pass as an earlier
            // window even if its filename is relabelled.
            let rain = SelectedField2D::new(
                FieldSelector::surface(CanonicalField::TotalPrecipitation),
                "kg/m^2",
                grid.clone(),
                (0..12)
                    .map(|cell| f32::from(hour * (hour + 1)) * (cell + 1) as f32)
                    .collect(),
            )
            .unwrap();
            write_hour_from_fields_with_derived(
                root,
                "wrf",
                run,
                hour,
                &[("temperature_2m", &temperature), ("apcp", &rain)],
                &[],
                &[],
                "windowed-anchor-test",
                1_578_000_000 + u64::from(hour) * 3600,
            )
            .unwrap();
        }
    }

    fn windowed_request(root: &std::path::Path, run: &str, out: &str) -> BatchRenderRequest {
        let mut request =
            BatchRenderRequest::conservative(root, "wrf", run, 0, "qpf_1h", root.join(out));
        request.output_width = 480;
        request.output_height = 360;
        request
    }

    #[test]
    fn all_stored_windows_render_each_anchor_and_current_retains_earlier_context() {
        let root = test_dir("windowed-anchors");
        let run = "local_wrf_20200102_210000";
        write_windowed_store(&root, run, &[0, 1, 2, 3, 4, 5, 6]);
        let mut all = windowed_request(&root, run, "all");
        all.hours = BatchHourScope::AllStored;
        let mut rendered = Vec::new();
        let mut skipped = Vec::new();
        let summary = run_batch_render(all, &AtomicBool::new(false), |event| match event {
            BatchRenderEvent::ItemRendered {
                hour, output_path, ..
            } => {
                rendered.push((hour, output_path));
            }
            BatchRenderEvent::ItemSkipped { hour, reason, .. } => skipped.push((hour, reason)),
            _ => {}
        })
        .unwrap();
        assert_eq!(
            (
                summary.planned,
                summary.rendered,
                summary.skipped,
                summary.failed
            ),
            (7, 6, 1, 0)
        );
        assert_eq!(
            rendered.iter().map(|(hour, _)| *hour).collect::<Vec<_>>(),
            (1..=6).map(Some).collect::<Vec<_>>()
        );
        assert_eq!(skipped.len(), 1);
        assert_eq!(skipped[0].0, Some(0));
        assert!(skipped[0].1.starts_with("F000:"), "{skipped:?}");

        for (hour, path) in &rendered {
            let hour = hour.unwrap();
            assert!(
                path.file_name()
                    .unwrap()
                    .to_string_lossy()
                    .contains(&format!("_f{hour:03}_"))
            );
            // A selected non-final anchor sees the same baseline and values
            // as the full gallery; the stored F006 must not replace it.
            let mut current = windowed_request(&root, run, &format!("current-{hour}"));
            current.hours = BatchHourScope::Current(hour);
            let one = run_batch_render(current, &AtomicBool::new(false), |_| {}).unwrap();
            assert_eq!(
                (one.planned, one.rendered, one.skipped, one.failed),
                (1, 1, 0, 0)
            );
            assert_eq!(
                std::fs::read(path).unwrap(),
                std::fs::read(&one.outputs[0]).unwrap()
            );
        }

        // Independent native compute control: the fixture's F001->F002
        // increment is 4*(cell+1) mm, with later F003..F006 still stored.
        let computed = crate::render_all::windowed_store::compute_windowed_products(
            &root,
            "wrf",
            run,
            &[0, 1, 2],
            &["qpf_1h".to_string()],
        )
        .unwrap();
        assert_eq!(computed.anchor_hour, 2);
        assert_eq!(computed.grids[0].hours_used, vec![1, 2]);
        for (cell, value) in computed.grids[0].values.iter().enumerate() {
            assert!((value - 4.0 * (cell + 1) as f64 / 25.4).abs() < 1e-6);
        }
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn windowed_missing_predecessor_is_a_skip_for_its_actual_anchor() {
        let root = test_dir("windowed-gap");
        let run = "local_wrf_20200102_210000";
        write_windowed_store(&root, run, &[0, 1, 3]);
        let mut request = windowed_request(&root, run, "out");
        request.hours = BatchHourScope::AllStored;
        let mut skipped = Vec::new();
        let summary = run_batch_render(request, &AtomicBool::new(false), |event| {
            if let BatchRenderEvent::ItemSkipped { hour, reason, .. } = event {
                skipped.push((hour, reason));
            }
        })
        .unwrap();
        assert_eq!(
            (
                summary.planned,
                summary.rendered,
                summary.skipped,
                summary.failed
            ),
            (3, 1, 2, 0)
        );
        assert_eq!(
            skipped.iter().map(|(hour, _)| *hour).collect::<Vec<_>>(),
            vec![Some(0), Some(3)]
        );
        assert!(skipped[1].1.starts_with("F003:"), "{skipped:?}");
        assert!(skipped[1].1.contains("F002"), "{skipped:?}");
        assert!(
            summary.outputs[0]
                .file_name()
                .unwrap()
                .to_string_lossy()
                .contains("_f001_")
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn windowed_work_limit_counts_every_selected_anchor() {
        let root = test_dir("windowed-limit");
        let run = "local_wrf_20200102_210000";
        write_windowed_store(&root, run, &[0, 1, 2, 3, 4, 5, 6]);
        let mut request = windowed_request(&root, run, "out");
        request.hours = BatchHourScope::AllStored;
        request.limits.max_work_items = 6;
        let error = run_batch_render(request, &AtomicBool::new(false), |_| {}).unwrap_err();
        assert!(error.contains("7 product-hours"), "{error}");
        assert!(!root.join("out").exists());
        let _ = std::fs::remove_dir_all(root);
    }

    /// A store whose one hour carries: a canonical plane that is the sole
    /// source of a named direct product, a known derived grid, an unknown
    /// derived-marker diagnostic, and a plane whose selector is neither a
    /// FieldSelector nor a derived marker.  The last one used to abort
    /// `StoreFieldSource::open` for the WHOLE hour.
    fn write_mixed_store(root: &std::path::Path, run: &str) -> (String, Vec<String>) {
        let grid = batch_grid();
        let temperature_selector = FieldSelector::height_agl(CanonicalField::Temperature, 2);
        let temperature_values: Vec<f32> = (0..12).map(|value| 270.0 + value as f32).collect();
        let temperature =
            SelectedField2D::new(temperature_selector, "K", grid, temperature_values.clone())
                .unwrap();
        let sbcape_values: Vec<f32> = (0..12).map(|value| 500.0 + value as f32 * 100.0).collect();
        let custom_values: Vec<f32> = (0..12).map(|value| value as f32).collect();
        write_hour_from_fields_with_derived(
            root,
            "wrf",
            run,
            0,
            &[("temperature_2m", &temperature)],
            &[
                DerivedFieldInput {
                    name: "sbcape",
                    units: "J/kg",
                    values: &sbcape_values,
                },
                DerivedFieldInput {
                    name: "custom_diagnostic_plane",
                    units: "widgets",
                    values: &custom_values,
                },
            ],
            &[],
            "batch-render-test",
            1_800_000_000,
        )
        .unwrap();

        // Rewrite the hour file with one extra variable whose selector is
        // deliberately opaque.  The writer API cannot spell it, which is
        // the point: only files, not this build's vocabulary, decide what
        // a store contains.
        let manifest = RwsRunManifest::load(&root.join("wrf").join(run).join("run.json")).unwrap();
        let mystery_values: Vec<f32> = (0..12).map(|value| -2.0 + value as f32 * 0.5).collect();
        let mut writer = HourWriter::new(
            "wrf",
            run,
            0,
            manifest.nx,
            manifest.ny,
            &manifest.grid_hash,
            "batch-render-test",
        );
        writer
            .add_surface2d(
                "temperature_2m",
                "K",
                serde_json::to_value(temperature_selector).unwrap(),
                &temperature_values,
            )
            .unwrap();
        writer
            .add_surface2d(
                "sbcape",
                "J/kg",
                rw_store::ingest::derived_selector("sbcape"),
                &sbcape_values,
            )
            .unwrap();
        writer
            .add_surface2d(
                "custom_diagnostic_plane",
                "widgets",
                rw_store::ingest::derived_selector("custom_diagnostic_plane"),
                &custom_values,
            )
            .unwrap();
        writer
            .add_surface2d(
                "mystery_plane",
                "widgets",
                serde_json::json!({ "bespoke": "wrfout attribute soup" }),
                &mystery_values,
            )
            .unwrap();
        let entry_file = manifest.hours.get(&0).unwrap().file.clone();
        writer
            .finish(&root.join("wrf").join(run).join(&entry_file))
            .unwrap();
        (
            "mystery_plane".to_string(),
            vec![
                "temperature_2m".to_string(),
                "sbcape".to_string(),
                "custom_diagnostic_plane".to_string(),
                "mystery_plane".to_string(),
            ],
        )
    }

    #[test]
    fn unknown_selectors_are_nonfatal_and_render_through_the_generic_lane() {
        let root = test_dir("generic-lane");
        let out = root.join("out");
        let run = "local_wrf_20200102_030000";
        let (mystery, all_variables) = write_mixed_store(&root, run);

        // (a) The opaque selector no longer aborts the hour.
        let store = StoreFieldSource::open(&root, "wrf", run, 0)
            .expect("one unknown diagnostic must not abort the whole render hour");
        assert_eq!(
            store
                .surface_variables()
                .iter()
                .map(|var| var.name.clone())
                .collect::<Vec<_>>(),
            all_variables,
            "the surface inventory must retain every stored 2-D variable in file order"
        );
        assert_eq!(store.surface_variable(&mystery).unwrap().units, "widgets");

        // (b) The catalog lists the opaque plane as a generic product.
        let catalog = inspect_renderable_products(&root, "wrf", run, 0).unwrap();
        let generic_row = catalog
            .products
            .iter()
            .find(|product| product.slug == format!("var:{mystery}"))
            .expect("var:mystery_plane must be in the catalog");
        assert_eq!(generic_row.kind, BatchProductKind::Generic);
        assert_eq!(generic_row.units.as_deref(), Some("widgets"));

        // (c) Rendering it through the production batch path yields a PNG.
        let request =
            BatchRenderRequest::conservative(&root, "wrf", run, 0, format!("var:{mystery}"), &out);
        let summary = run_batch_render(request, &AtomicBool::new(false), |_event| {}).unwrap();
        assert_eq!(summary.rendered, 1, "{summary:?}");
        assert_eq!(summary.failed, 0);
        let output = &summary.outputs[0];
        assert!(output.exists(), "{}", output.display());
        assert!(
            std::fs::metadata(output).unwrap().len() > 1_000,
            "suspiciously small PNG at {}",
            output.display()
        );
        let name = output.file_name().unwrap().to_string_lossy().into_owned();
        // The variable's own name, with no hash beside it.
        assert!(name.ends_with("_var_mystery_plane.png"), "{name}");
        let _ = std::fs::remove_dir_all(root);
    }

    /// One hour whose two bulk-shear planes hold `values` in `units`.
    fn write_shear_store(root: &std::path::Path, run: &str, units: &str, values: &[f32]) {
        let grid = batch_grid();
        let temperature = SelectedField2D::new(
            FieldSelector::height_agl(CanonicalField::Temperature, 2),
            "K",
            grid,
            vec![290.0; 12],
        )
        .unwrap();
        write_hour_from_fields_with_derived(
            root,
            "wrf",
            run,
            0,
            &[("temperature_2m", &temperature)],
            &[
                DerivedFieldInput {
                    name: "bulk_shear_0_1km",
                    units,
                    values,
                },
                DerivedFieldInput {
                    name: "bulk_shear_0_6km",
                    units,
                    values,
                },
            ],
            &[],
            "shear-units-test",
            1_800_000_000,
        )
        .unwrap();
    }

    /// Every picture of one store's shear products, by slug, plus the
    /// skipped slugs with their reasons.
    fn render_shear_store(
        label: &str,
        units: &str,
        values: &[f32],
    ) -> (
        std::collections::BTreeMap<String, Vec<u8>>,
        std::collections::BTreeMap<String, String>,
    ) {
        let root = test_dir(label);
        let run = "local_wrf_20200102_030000";
        write_shear_store(&root, run, units, values);
        let products = "bulk_shear_0_1km,bulk_shear_0_6km,var:bulk_shear_0_1km,var:bulk_shear_0_6km";
        let mut request =
            BatchRenderRequest::conservative(&root, "wrf", run, 0, products, root.join("out"));
        request.output_width = 480;
        request.output_height = 360;
        let mut drawn = std::collections::BTreeMap::new();
        let mut skipped = std::collections::BTreeMap::new();
        let summary = run_batch_render(request, &AtomicBool::new(false), |event| match event {
            BatchRenderEvent::ItemRendered {
                slug, output_path, ..
            } => {
                drawn.insert(slug, std::fs::read(output_path).unwrap());
            }
            BatchRenderEvent::ItemSkipped { slug, reason, .. } => {
                skipped.insert(slug, reason);
            }
            _ => {}
        })
        .unwrap();
        assert_eq!(summary.failed, 0, "{summary:?}");
        let _ = std::fs::remove_dir_all(root);
        (drawn, skipped)
    }

    #[test]
    fn stored_bulk_shear_is_drawn_in_knots_whatever_unit_the_store_holds() {
        // A raw-wrfout import stores bulk shear in m/s, the GRIB ingest in
        // kt; the named products and their `var:` pictures are drawn in kt.
        let metres_per_second: Vec<f32> = (0..12).map(|cell| 2.0 + 2.0 * cell as f32).collect();
        // The named products convert in f64 with the compute lane's factor,
        // the `var:` pictures in f32 with the viewer's; each is compared
        // against a knot store holding exactly the numbers it should draw.
        let knots_f64: Vec<f32> = metres_per_second
            .iter()
            .map(|value| (f64::from(*value) * 1.943_844_5) as f32)
            .collect();
        let knots_f32: Vec<f32> = metres_per_second
            .iter()
            .map(|value| *value * 1.943_844_5)
            .collect();

        let (from_ms, ms_skips) = render_shear_store("shear-ms", "m/s", &metres_per_second);
        let (from_kt, _) = render_shear_store("shear-kt", "kt", &knots_f64);
        let (from_kt_f32, _) = render_shear_store("shear-kt-f32", "kt", &knots_f32);
        let (as_if_knots, _) = render_shear_store("shear-raw", "kt", &metres_per_second);
        assert!(ms_skips.is_empty(), "{ms_skips:?}");
        for slug in ["bulk_shear_0_1km", "bulk_shear_0_6km"] {
            let var = format!("var:{slug}");
            assert!(
                from_ms[slug] == from_kt[slug],
                "{slug}: shear stored in m/s must be drawn in knots"
            );
            assert!(
                from_ms[var.as_str()] == from_kt_f32[var.as_str()],
                "{var}: shear stored in m/s must be drawn in knots"
            );
            assert!(
                from_ms[slug] != as_if_knots[slug],
                "{slug}: m/s numbers must not be read off the knot colour bar"
            );
            assert!(
                from_ms[var.as_str()] != as_if_knots[var.as_str()],
                "{var}: m/s numbers must not be read off the knot colour bar"
            );
        }

        // A unit with no way to knots draws no named picture and says why.
        let (from_kelvin, kelvin_skips) =
            render_shear_store("shear-kelvin", "K", &metres_per_second);
        for slug in ["bulk_shear_0_1km", "bulk_shear_0_6km"] {
            assert!(!from_kelvin.contains_key(slug), "{slug} drawn from a K plane");
            let reason = &kelvin_skips[slug];
            assert!(reason.contains("\"K\"") && reason.contains("\"kt\""), "{reason}");
        }
    }

    /// The parallel product loop's two determinism claims, on the
    /// SHIPPED default width (no environment override, so what runs here
    /// is what a campaign runs): the same request rendered twice
    /// produces the same event sequence in the same order, and the same
    /// PNG bytes.  Bit-for-bit, because a product loop that is only
    /// statistically reproducible is not a render engine.
    #[test]
    fn parallel_product_loop_is_byte_deterministic_and_reports_in_catalog_order() {
        let root = test_dir("parallel-determinism");
        let run = "local_wrf_20200102_030000";
        write_mixed_store(&root, run);
        let products = "2m_temperature,sbcape,var:custom_diagnostic_plane,var:mystery_plane";
        assert!(
            product_worker_count_within(4, host_memory::available_bytes()) > 1
                || std::thread::available_parallelism().unwrap().get() < 4,
            "a multi-core box must actually fan this request out, or the test proves nothing"
        );

        let render_pass = |label: &str| {
            let out = root.join(label);
            let request = BatchRenderRequest::conservative(&root, "wrf", run, 0, products, &out);
            let mut events = Vec::new();
            let summary = run_batch_render(request, &AtomicBool::new(false), |event| {
                // Timings are wall clock and never identical; identity is
                // claimed over the sequence, the slugs and the counters.
                match event {
                    BatchRenderEvent::ItemStarted {
                        slug, completed, ..
                    } => events.push(format!("started {slug} {completed}")),
                    BatchRenderEvent::ItemRendered {
                        slug,
                        output_path,
                        completed,
                        ..
                    } => events.push(format!(
                        "rendered {slug} {completed} {}",
                        output_path.file_name().unwrap().to_string_lossy()
                    )),
                    BatchRenderEvent::ItemSkipped {
                        slug,
                        reason,
                        completed,
                        ..
                    } => events.push(format!("skipped {slug} {completed} {reason}")),
                    BatchRenderEvent::ItemFailed { slug, error, .. } => {
                        events.push(format!("failed {slug} {error}"))
                    }
                    _ => {}
                }
            })
            .unwrap();
            let bytes = summary
                .outputs
                .iter()
                .map(|path| {
                    (
                        path.file_name().unwrap().to_string_lossy().into_owned(),
                        std::fs::read(path).unwrap(),
                    )
                })
                .collect::<Vec<_>>();
            (summary, events, bytes)
        };

        let (first_summary, first_events, first_bytes) = render_pass("pass-a");
        let (second_summary, second_events, second_bytes) = render_pass("pass-b");

        assert_eq!(first_summary.failed, 0, "{first_summary:?}");
        assert!(first_summary.rendered >= 3, "{first_summary:?}");
        assert_eq!(first_events, second_events, "event stream is not stable");
        assert_eq!(
            first_summary.rendered, second_summary.rendered,
            "{first_summary:?} vs {second_summary:?}"
        );
        assert_eq!(first_summary.skipped, second_summary.skipped);
        assert_eq!(
            first_bytes.iter().map(|(name, _)| name).collect::<Vec<_>>(),
            second_bytes
                .iter()
                .map(|(name, _)| name)
                .collect::<Vec<_>>(),
            "summary.outputs must stay in catalog order"
        );
        for ((name, left), (_, right)) in first_bytes.iter().zip(second_bytes.iter()) {
            assert_eq!(
                left, right,
                "{name} differs between two runs of the same request"
            );
        }
        let _ = std::fs::remove_dir_all(root);
    }

    /// Cancelling mid-hour under the fan-out has to terminate, report a
    /// consistent tally, and leave every PNG it announced on disk.  The
    /// in-order drain waits on a channel, so a worker that returned
    /// without reporting would hang the whole batch rather than fail it.
    #[test]
    fn cancelling_mid_hour_terminates_with_a_consistent_tally() {
        let root = test_dir("parallel-cancel");
        let out = root.join("out");
        let run = "local_wrf_20200102_030000";
        write_mixed_store(&root, run);
        let request = BatchRenderRequest::conservative(
            &root,
            "wrf",
            run,
            0,
            "2m_temperature,sbcape,var:custom_diagnostic_plane,var:mystery_plane",
            &out,
        );
        let cancel = AtomicBool::new(false);
        let summary = run_batch_render(request, &cancel, |event| {
            if matches!(event, BatchRenderEvent::ItemRendered { .. }) {
                cancel.store(true, std::sync::atomic::Ordering::Relaxed);
            }
        })
        .unwrap();
        assert!(summary.cancelled, "{summary:?}");
        assert_eq!(summary.failed, 0, "{summary:?}");
        assert!(
            summary.rendered + summary.skipped <= summary.planned,
            "{summary:?}"
        );
        assert_eq!(summary.rendered, summary.outputs.len(), "{summary:?}");
        for output in &summary.outputs {
            assert!(
                output.exists(),
                "announced but absent: {}",
                output.display()
            );
        }
        let _ = std::fs::remove_dir_all(root);
    }

    /// The width policy itself: never wider than the work, never zero,
    /// and a one-product hour never builds a pool at all.
    #[test]
    fn product_worker_count_is_bounded_by_the_work() {
        let free = host_memory::available_bytes();
        assert_eq!(product_worker_count_within(0, free), 1);
        assert_eq!(product_worker_count_within(1, free), 1);
        for items in [2usize, 3, 8, 64, 4096] {
            let workers = product_worker_count_within(items, free);
            assert!(workers >= 1, "{items} items produced {workers} workers");
            assert!(workers <= items, "{items} items produced {workers} workers");
        }
    }

    #[test]
    fn live_frames_alone_share_the_forecast_cpu_budget() {
        for (processors, batch, single_live, three_live) in [
            (1, 1, 1, 1),
            (2, 2, 1, 1),
            (8, 8, 7, 2),
            (16, 16, 15, 5),
            (64, 64, 63, 21),
        ] {
            assert_eq!(default_product_worker_count(processors, None), batch);
            assert_eq!(default_product_worker_count(processors, Some(1)), single_live);
            assert_eq!(default_product_worker_count(processors, Some(3)), three_live);
        }
    }

    #[test]
    fn ordinary_batch_uses_the_processor_budget() {
        assert_eq!(
            product_worker_count_within(usize::MAX, None),
            render_processor_budget()
        );
    }

    /// The width a box can afford is the width it gets.  Driven through
    /// `product_worker_count_within` with the host query supplied, so the
    /// rule is tested at memory sizes this machine does not have.
    #[test]
    fn the_default_width_is_capped_by_free_memory() {
        let cores = render_processor_budget();
        let plenty = PRODUCT_LOOP_BASE_BYTES * 8 + PRODUCT_WORKER_BYTES * 4096;

        // Room for everything: cores and the work are what bind.
        assert_eq!(
            product_worker_count_within(4096, Some(plenty * 4)),
            cores,
            "an unloaded box must still fan out to its cores"
        );
        assert_eq!(
            product_worker_count_within(3, Some(plenty * 4)),
            cores.min(3)
        );

        // A platform that will not answer imposes no cap, which is what
        // this loop did before the cap existed.
        assert_eq!(product_worker_count_within(4096, None), cores);

        // Room for the loop and about four workers.  Halved first: the
        // budget is BUDGET_FRACTION of free memory, not all of it.
        let four = (PRODUCT_LOOP_BASE_BYTES + PRODUCT_WORKER_BYTES * 4) * 2;
        assert_eq!(product_worker_count_within(4096, Some(four)), cores.min(4));

        // Not even room for the loop's own footprint: still one worker,
        // never zero, because refusing to render is not this rule's call.
        assert_eq!(product_worker_count_within(4096, Some(1024)), 1);
    }

    /// A memory-capped width is never silent, and a width the box chose
    /// freely never says anything.
    #[test]
    fn a_memory_bound_width_advises_and_an_unbound_one_does_not() {
        let cores = render_processor_budget();
        if cores < 2 {
            return; // Nothing to be capped below.
        }
        let (_, said) = advisory::hold(|| advise_if_memory_bound(cores, 4096, Some(1 << 40)));
        assert!(
            said.is_empty(),
            "an uncapped width must not editorialise: {said:?}"
        );

        let (_, said) = advisory::hold(|| advise_if_memory_bound(1, 4096, Some(1 << 30)));
        assert_eq!(said.len(), 1, "a capped width must say so: {said:?}");
        assert!(
            said[0].line.starts_with("RENDER_WIDTH\t1\t"),
            "{}",
            said[0].line
        );
        assert!(
            said[0].line.contains("RUSTWX_BATCH_RENDER_THREADS"),
            "the advisory must name the override: {}",
            said[0].line
        );
    }

    #[test]
    fn generic_rows_excluded_are_exactly_the_sole_sources_of_named_products() {
        let root = test_dir("generic-dedup");
        let run = "local_wrf_20200102_030000";
        let (_, all_variables) = write_mixed_store(&root, run);
        let catalog = inspect_renderable_products(&root, "wrf", run, 0).unwrap();

        let sole_sources: std::collections::HashSet<&str> = catalog
            .products
            .iter()
            .filter(|product| product.kind != BatchProductKind::Generic)
            .filter_map(|product| match product.source_fields.as_slice() {
                [only] => Some(only.as_str()),
                _ => None,
            })
            .collect();
        assert!(
            sole_sources.contains("temperature_2m"),
            "2m_temperature must claim its stored plane: {catalog:?}"
        );
        assert!(sole_sources.contains("sbcape"), "{catalog:?}");

        for variable in &all_variables {
            let listed = catalog
                .products
                .iter()
                .any(|product| product.slug == format!("var:{variable}"));
            let excluded = sole_sources.contains(variable.as_str());
            assert!(
                listed != excluded,
                "variable '{variable}': listed={listed} excluded-by-dedup={excluded}; \
                 a stored grid must render under exactly one slug"
            );
        }
        let _ = std::fs::remove_dir_all(root);
    }
}
