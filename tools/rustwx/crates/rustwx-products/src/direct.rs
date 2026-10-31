use crate::derived::NativeContourRenderMode;
#[cfg(test)]
use rustwx_core::CanonicalField;
use rustwx_core::{FieldSelector, ModelId, SelectedField2D, SourceId};
use rustwx_models::{LatestRun, PlotRecipe, plot_recipe};
#[cfg(test)]
use rustwx_models::{PlotRecipeFetchMode, plot_recipe_fetch_plan};
use rustwx_render::{
    Color, ContourLayer, PanelGridLayout, PanelPadding, PngCompressionMode, PngWriteOptions,
    ProductVisualMode, ProjectedMap, RenderImageTiming, RenderStateTiming, WindBarbLayer,
    WindStreamlineLayer, draw_centered_text_line, render_panel_grid, save_png_profile_with_options,
    save_rgba_png_profile_with_options,
};
#[cfg(test)]
use rustwx_render::{ColorScale, ExtendMode, MapRenderRequest, RasterSampleMode};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::PathBuf;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};
use std::thread;
use std::time::Instant;

use crate::publication::artifact_identity_from_path;
use crate::shared_context::{
    TitleProvenance,
    DomainSpec, ProjectedMapProvider, model_time_subtitle, source_subtitle,
};
use crate::source::direct_route_for_recipe_slug;

mod batch;
mod composite;
mod domain;
mod fetch;
mod planning;
mod projection;
mod query;
mod rendering;
mod titles;
mod types;
pub use titles::{local_accumulation_window, local_accumulation_window_from_cycle};
pub use types::AccumulationWindow;
pub(crate) use batch::{
    prepare_direct_batch_from_loaded, run_direct_batch_from_loaded, run_direct_batch_from_prepared,
};
pub use batch::{run_direct_batch, run_hrrr_direct_batch};
use composite::{CompositePanelSpec, composite_panel_spec};
#[cfg(test)]
use domain::{
    DirectGridCrop, crop_for_direct_grid, crop_latlon_grid_for_direct,
    crop_selected_field_for_domain, is_global_scale_domain, longitude_bounds_span_deg,
    point_in_geographic_bounds,
};
use domain::{
    crop_bounds_for_direct_request, crop_direct_fields_for_domain, render_bounds_for_direct_field,
};
use fetch::{extract_direct_fetch_group_from_loaded, find_loaded_bytes_for_group};
pub use planning::{
    FetchGroup, direct_recipe_is_time_invariant, direct_recipe_requires_explicit_opt_in,
    store_direct_recipe_slugs, supported_direct_recipe_slugs,
};
use planning::{
    PlannedDirectRecipe, canonical_fetch_product_for_selectors, group_direct_fetches,
    plan_direct_recipes, plan_selected_recipes,
};
#[cfg(test)]
use planning::{canonical_fetch_product, should_attach_direct_idx_patterns};
pub(crate) use projection::inverse_raster_projection_for_grid;
#[cfg(test)]
use projection::{
    PIVOTAL_CONUS_CENTRAL_MERIDIAN_DEG, PIVOTAL_CONUS_REFERENCE_LATITUDE_DEG,
    PIVOTAL_CONUS_STANDARD_PARALLEL_1_DEG, PIVOTAL_CONUS_STANDARD_PARALLEL_2_DEG,
    ProjectionPresentationVariant, center_longitude_for_bounds,
    full_domain_projected_frame_default, inverse_raster_clip_bounds,
    presentation_frame_bounds_for_grid, presentation_projection_for_bounds,
    reference_latitude_for_projection_variant,
};
pub use projection::{
    build_full_domain_projected_map_with_projection, full_domain_panel_resolved_projection,
    build_natural_projected_map_with_projection,
    build_natural_projected_map_with_projection_and_basemap_padding, build_projected_map,
    build_projected_map_with_projection, build_requested_projected_map_with_projection,
    direct_map_frame_aspect_ratio, model_data_domain_frame_for_projection,
    frame_short_side_km, natural_grid_aspect, panel_resolved_projection, plan_canvas_for_grid,
    planned_place_labels,
    project_points_with_projection, requested_panel_resolved_projection,
};
pub(crate) use query::{load_direct_sampled_fields_from_latest, required_direct_fetch_products};
#[cfg(test)]
use rendering::{
    StreamlineSetting, barb_target_columns_rows, convert_filled_field, render_filled_field,
    scale_for_filled_selector, scale_for_recipe, static_streamline_setting,
    streamlines_enabled_for_grid,
};
use rendering::{
    apply_source_raster_policy, build_render_request, sanitize_output_suffix,
    should_render_overlay_only, visual_mode_for_direct_recipe,
};
pub(crate) use rendering::{direct_fill_unit_conversion, direct_recipe_render_controls};
#[cfg(test)]
use titles::{apply_native_stat_title_prefix, native_stat_label_for_request};
use titles::{direct_panel_title_for_request, direct_title_for_planned_product};
pub(crate) use types::PreparedDirectBatch;
pub use types::{
    DirectBatchReport, DirectBatchRequest, DirectFetchRuntimeInfo, DirectFetchTiming,
    DirectRecipeBlocker, DirectRecipeTiming, DirectRenderedRecipe, HrrrDirectBatchReport,
    HrrrDirectBatchRequest, HrrrDirectFetchRuntimeInfo, HrrrDirectFetchTiming,
    HrrrDirectRecipeBlocker, HrrrDirectRecipeTiming, HrrrDirectRenderedRecipe,
};
use types::{DirectRequestBuildTiming, OUTPUT_HEIGHT, OUTPUT_WIDTH};

fn direct_data_layer_draw_ms(image_timing: &RenderImageTiming) -> u128 {
    image_timing.polygon_fill_ms
        + image_timing.projected_pixel_ms
        + image_timing.rasterize_ms
        + image_timing.raster_blit_ms
}

fn direct_overlay_draw_ms(image_timing: &RenderImageTiming) -> u128 {
    image_timing.linework_ms + image_timing.contour_ms + image_timing.barb_ms
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct BarbStrideCacheKey {
    u_selector: FieldSelector,
    v_selector: FieldSelector,
    bounds_bits: [u64; 4],
}

type SharedContourLayerCache =
    Arc<Mutex<HashMap<(FieldSelector, usize, usize), Option<ContourLayer>>>>;
type SharedBarbStrideCache = Arc<Mutex<HashMap<BarbStrideCacheKey, (usize, usize)>>>;
type SharedBarbLayerCache = Arc<Mutex<HashMap<BarbStrideCacheKey, Vec<WindBarbLayer>>>>;
type SharedStreamlineLayerCache = Arc<Mutex<HashMap<BarbStrideCacheKey, Vec<WindStreamlineLayer>>>>;
type ProjectedMapCacheKey = (u32, u32, u8, usize, usize, String);
type SharedProjectedMapCache = Arc<Mutex<HashMap<ProjectedMapCacheKey, ProjectedMap>>>;
type PreparedProjectedMaps = Arc<HashMap<ProjectedMapCacheKey, ProjectedMap>>;

impl DirectBatchRequest {
    fn from_hrrr(request: &HrrrDirectBatchRequest) -> Self {
        Self {
            accumulation_window: None,
            model: ModelId::Hrrr,
            date_yyyymmdd: request.date_yyyymmdd.clone(),
            cycle_override_utc: request.cycle_override_utc,
            forecast_hour: request.forecast_hour,
            source: request.source,
            domain: request.domain.clone(),
            out_dir: request.out_dir.clone(),
            cache_root: request.cache_root.clone(),
            use_cache: request.use_cache,
            recipe_slugs: request.recipe_slugs.clone(),
            product_overrides: HashMap::new(),
            contour_mode: request.contour_mode,
            native_fill_level_multiplier: request.native_fill_level_multiplier,
            output_width: request.output_width,
            output_height: request.output_height,
            png_compression: request.png_compression,
            place_label_overlay: request.place_label_overlay.clone(),
            output_suffix: None,
            subtitle_left_override: None,
            subtitle_right_override: None,
            title_provenance: TitleProvenance::default(),
            geographic_overlays: None,
            panel_annotations: None,
        }
    }

    /// Public planner-side conversion: lets the unified non-ECAPE-hour
    /// runner build a `DirectBatchRequest` from the HRRR-pinned variant
    /// so it can ask the direct lane to plan its fetch groups before
    /// loading bundles.
    pub fn from_hrrr_for_planner(request: &HrrrDirectBatchRequest) -> Self {
        Self::from_hrrr(request)
    }
}

impl DirectBatchRequest {
    fn png_write_options(&self) -> PngWriteOptions {
        PngWriteOptions {
            compression: self.png_compression,
        }
    }
}

fn sampling_direct_request(
    model: ModelId,
    source: SourceId,
    forecast_hour: u16,
    cache_root: &std::path::Path,
    use_cache: bool,
) -> DirectBatchRequest {
    DirectBatchRequest {
        accumulation_window: None,
        model,
        date_yyyymmdd: String::new(),
        cycle_override_utc: None,
        forecast_hour,
        source,
        domain: DomainSpec::new("sampling", (-180.0, 180.0, -90.0, 90.0)),
        out_dir: PathBuf::new(),
        cache_root: cache_root.to_path_buf(),
        use_cache,
        recipe_slugs: Vec::new(),
        product_overrides: HashMap::new(),
        contour_mode: NativeContourRenderMode::Automatic,
        native_fill_level_multiplier: 1,
        output_width: OUTPUT_WIDTH,
        output_height: OUTPUT_HEIGHT,
        png_compression: PngCompressionMode::Default,
        place_label_overlay: None,
        output_suffix: None,
        subtitle_left_override: None,
        subtitle_right_override: None,
        title_provenance: TitleProvenance::default(),
        geographic_overlays: None,
        panel_annotations: None,
    }
}

/// Plan the direct lane's fetch groups without running the loader. The
/// unified non-ECAPE-hour runner uses this to build a single execution
/// plan that covers direct + derived (+ severe/ECAPE if requested).
pub fn plan_direct_fetch_groups(
    request: &DirectBatchRequest,
) -> Result<Vec<FetchGroup>, Box<dyn std::error::Error>> {
    let planned = plan_direct_recipes(request.model, &request.recipe_slugs)?;
    Ok(group_direct_fetches(request, &planned))
}

pub fn render_direct_recipe_from_selected_fields(
    request: &DirectBatchRequest,
    latest: &LatestRun,
    recipe_slug: &str,
    extracted: &HashMap<FieldSelector, SelectedField2D>,
    fetched_product: impl Into<String>,
    resolved_url: impl Into<String>,
    fetch_key: impl Into<String>,
) -> Result<DirectRenderedRecipe, Box<dyn std::error::Error>> {
    let mut rendered = render_direct_recipes_from_selected_fields(
        request,
        latest,
        &[recipe_slug.to_string()],
        extracted,
        fetched_product,
        resolved_url,
        fetch_key,
    )?;
    rendered
        .pop()
        .ok_or_else(|| "direct recipe rendered no outputs".into())
}

pub fn render_direct_recipes_from_selected_fields(
    request: &DirectBatchRequest,
    latest: &LatestRun,
    recipe_slugs: &[String],
    extracted: &HashMap<FieldSelector, SelectedField2D>,
    fetched_product: impl Into<String>,
    resolved_url: impl Into<String>,
    fetch_key: impl Into<String>,
) -> Result<Vec<DirectRenderedRecipe>, Box<dyn std::error::Error>> {
    fs::create_dir_all(&request.out_dir)?;
    let planned = plan_selected_recipes(request.model, recipe_slugs)?;
    let fetch_truth_by_actual_product = direct_fetch_truth_for_planned(
        request,
        latest,
        &planned,
        fetched_product.into(),
        resolved_url.into(),
        fetch_key.into(),
    );

    let missing = planned
        .iter()
        .flat_map(|item| item.plan.selectors())
        .filter(|selector| !extracted.contains_key(selector))
        .collect::<HashSet<_>>();
    if !missing.is_empty() {
        return Err(format!("missing selected fields for direct render: {:?}", missing).into());
    }

    render_direct_recipes(
        request,
        latest,
        &planned,
        extracted,
        &fetch_truth_by_actual_product,
        None,
    )
}

/// Default recipe-chunk size for [`render_direct_recipes_chunked_from_loader`],
/// overridable via `RUSTWX_DIRECT_RENDER_CHUNK`. Sized so one chunk's
/// full-grid input fields (~23 MB per selector at HRRR size, transient
/// until the domain crop lands) plus its cropped fields and worker scratch
/// stay well under ~1.5 GB, while chunks stay large enough that the
/// shared-selector reloads across chunks cost only a few store reads.
pub fn direct_render_chunk_size() -> usize {
    std::env::var("RUSTWX_DIRECT_RENDER_CHUNK")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(16)
}

/// Chunked sibling of [`render_direct_recipes_from_selected_fields`] for
/// callers that can load fields on demand (the `.rws` store render lane):
/// instead of holding every needed full-grid field across the whole pass
/// (~2 GB at HRRR size for the all-products list), recipes render in
/// chunks of `chunk_recipes`; each chunk loads only its own selectors,
/// crops them to the render domain, and FREES the full-grid copies before
/// rendering. Pixel identity with the unchunked pass holds because:
///
/// * the crop bounds are derived once, by the exact selection rule of
///   `crop_bounds_for_direct_request` over the full planned list, so every
///   chunk crops with the same bounds the unchunked pass used;
/// * each recipe sees the same cropped fields, the same fetch truth (built
///   from the full planned list), and the same render path;
/// * the shared layer/projected-map caches persist across chunks, and
///   every cached entry is a pure function of its key inputs, so chunk
///   grouping changes only WHEN an entry is built, never its content.
#[allow(clippy::too_many_arguments)]
pub fn render_direct_recipes_chunked_from_loader(
    request: &DirectBatchRequest,
    latest: &LatestRun,
    recipe_slugs: &[String],
    load_field: &mut dyn FnMut(
        &FieldSelector,
    ) -> Result<SelectedField2D, Box<dyn std::error::Error>>,
    chunk_recipes: usize,
    // Called before each chunk loads its fields. Hosts that pipeline this
    // render against other memory-hungry work (rw_batch) pass a gate that
    // blocks while the process is inside a declared high-memory window;
    // pixel output is independent of WHEN a chunk runs, so the gate can
    // only trade wall time for peak working set.
    chunk_gate: Option<&dyn Fn()>,
    fetched_product: impl Into<String>,
    resolved_url: impl Into<String>,
    fetch_key: impl Into<String>,
) -> Result<Vec<DirectRenderedRecipe>, Box<dyn std::error::Error>> {
    fs::create_dir_all(&request.out_dir)?;
    let planned = plan_selected_recipes(request.model, recipe_slugs)?;
    if planned.is_empty() {
        return Ok(Vec::new());
    }
    let fetch_truth_by_actual_product = direct_fetch_truth_for_planned(
        request,
        latest,
        &planned,
        fetched_product.into(),
        resolved_url.into(),
        fetch_key.into(),
    );

    // Crop bounds exactly as `crop_bounds_for_direct_request` would pick
    // them from the full extracted map: the first planned recipe whose
    // filled selector the (full) pass would have loaded.
    let needed: HashSet<FieldSelector> = planned
        .iter()
        .flat_map(|item| item.plan.selectors())
        .collect();
    let crop_bounds = match planned.iter().find_map(|item| {
        let selector = item.recipe.filled.selector?;
        needed
            .contains(&selector)
            .then_some((item.recipe, selector))
    }) {
        Some((recipe, selector)) => {
            let field = load_field(&selector)?;
            let overlay_only =
                should_render_overlay_only(field.selector, recipe.contours.is_some());
            let visual_mode = visual_mode_for_direct_recipe(recipe, field.selector, overlay_only);
            render_bounds_for_direct_field(
                request.domain.bounds,
                &field,
                visual_mode,
                request.output_width,
                request.output_height,
            )
        }
        None => request.domain.bounds,
    };

    let shared = DirectSharedRenderCaches::default();
    let mut prepared_accum: HashMap<ProjectedMapCacheKey, ProjectedMap> = HashMap::new();
    let mut completed = Vec::with_capacity(planned.len());
    for chunk in planned.chunks(chunk_recipes.max(1)) {
        if let Some(gate) = chunk_gate {
            gate();
        }
        let mut extracted = HashMap::<FieldSelector, SelectedField2D>::new();
        for item in chunk {
            for selector in item.plan.selectors() {
                if !extracted.contains_key(&selector) {
                    extracted.insert(selector, load_field(&selector)?);
                }
            }
        }
        let domain_extracted = crop_direct_fields_for_domain(&extracted, crop_bounds)?;
        // The full-grid inputs leave RAM before any rendering starts; the
        // render workers only ever see the domain-cropped fields.
        drop(extracted);
        extend_prepared_projected_maps(request, chunk, &domain_extracted, &mut prepared_accum)?;
        let prepared: PreparedProjectedMaps = Arc::new(prepared_accum.clone());
        if prepared.is_empty() {
            // Mirrors the unchunked pass: no projected map means no
            // renderable recipe in this chunk (and, in practice, none at
            // all, a chunk only lacks maps when its fields are absent).
            continue;
        }
        completed.extend(render_cropped_direct_recipes(
            request,
            latest,
            chunk,
            &domain_extracted,
            &fetch_truth_by_actual_product,
            None,
            &shared,
            &prepared,
        )?);
    }
    Ok(completed)
}

/// Fetch-provenance truth per canonical product family, shared verbatim by
/// the whole-pass and chunked direct render entries (report metadata only,
/// never pixels).
fn direct_fetch_truth_for_planned(
    request: &DirectBatchRequest,
    latest: &LatestRun,
    planned: &[PlannedDirectRecipe],
    fetched_product: String,
    resolved_url: String,
    fetch_key: String,
) -> HashMap<String, DirectFetchRuntimeInfo> {
    let groups = group_direct_fetches(request, planned);
    let mut fetch_truth_by_actual_product = HashMap::<String, DirectFetchRuntimeInfo>::new();
    for group in &groups {
        fetch_truth_by_actual_product.insert(
            group.product.clone(),
            DirectFetchRuntimeInfo {
                fetch_key: fetch_key.clone(),
                planned_product: group.product.clone(),
                fetched_product: fetched_product.clone(),
                planned_family_aliases: group.planned_family_aliases.iter().cloned().collect(),
                requested_source: request.source,
                resolved_source: latest.source,
                resolved_url: resolved_url.clone(),
            },
        );
    }
    fetch_truth_by_actual_product
}

/// The shared mutable caches one direct render pass carries across its
/// recipes (and, in the chunked entry, across its chunks). Every entry is
/// a pure function of its cache key inputs, so sharing changes only when
/// an entry is built, never its content.
#[derive(Default)]
struct DirectSharedRenderCaches {
    contour_layer_cache: SharedContourLayerCache,
    barb_layer_cache: SharedBarbLayerCache,
    streamline_layer_cache: SharedStreamlineLayerCache,
    barb_stride_cache: SharedBarbStrideCache,
    projected_map_cache: SharedProjectedMapCache,
}

fn render_direct_recipes(
    request: &DirectBatchRequest,
    latest: &LatestRun,
    planned: &[PlannedDirectRecipe],
    extracted: &HashMap<FieldSelector, SelectedField2D>,
    fetch_truth_by_actual_product: &HashMap<String, DirectFetchRuntimeInfo>,
    shared_context: Option<&dyn ProjectedMapProvider>,
) -> Result<Vec<DirectRenderedRecipe>, Box<dyn std::error::Error>> {
    if planned.is_empty() {
        return Ok(Vec::new());
    }

    let crop_bounds = crop_bounds_for_direct_request(request, planned, extracted);
    let domain_extracted = crop_direct_fields_for_domain(extracted, crop_bounds)?;
    let extracted = &domain_extracted;
    let shared = DirectSharedRenderCaches::default();
    let prepared_projected_maps = build_prepared_projected_maps(request, planned, extracted)?;
    if prepared_projected_maps.is_empty() {
        return Ok(Vec::new());
    }
    render_cropped_direct_recipes(
        request,
        latest,
        planned,
        extracted,
        fetch_truth_by_actual_product,
        shared_context,
        &shared,
        &prepared_projected_maps,
    )
}

/// The worker-loop core of one direct render pass over ALREADY
/// domain-cropped fields: the exact per-recipe render path, parallelized
/// over `render_worker_count` self-scheduling workers. Factored out so the
/// chunked store-render entry drives the identical code with bounded
/// per-chunk inputs.
#[allow(clippy::too_many_arguments)]
fn render_cropped_direct_recipes(
    request: &DirectBatchRequest,
    latest: &LatestRun,
    planned: &[PlannedDirectRecipe],
    extracted: &HashMap<FieldSelector, SelectedField2D>,
    fetch_truth_by_actual_product: &HashMap<String, DirectFetchRuntimeInfo>,
    shared_context: Option<&dyn ProjectedMapProvider>,
    shared: &DirectSharedRenderCaches,
    prepared_projected_maps: &PreparedProjectedMaps,
) -> Result<Vec<DirectRenderedRecipe>, Box<dyn std::error::Error>> {
    let DirectSharedRenderCaches {
        contour_layer_cache,
        barb_layer_cache,
        streamline_layer_cache,
        barb_stride_cache,
        projected_map_cache,
    } = shared;
    let worker_count = render_worker_count(planned.len());
    if worker_count <= 1 {
        return planned
            .iter()
            .map(|item| {
                render_direct_recipe(
                    request,
                    latest,
                    item,
                    extracted,
                    fetch_truth_by_actual_product,
                    shared_context,
                    contour_layer_cache,
                    barb_layer_cache,
                    streamline_layer_cache,
                    barb_stride_cache,
                    projected_map_cache,
                    prepared_projected_maps,
                )
            })
            .collect();
    }

    let next_index = AtomicUsize::new(0);
    let mut rendered = vec![None; planned.len()];

    thread::scope(|scope| -> Result<(), std::io::Error> {
        let mut handles = Vec::new();
        for _ in 0..worker_count {
            let barb_stride_cache = Arc::clone(barb_stride_cache);
            let contour_layer_cache = Arc::clone(contour_layer_cache);
            let barb_layer_cache = Arc::clone(barb_layer_cache);
            let streamline_layer_cache = Arc::clone(streamline_layer_cache);
            let projected_map_cache = Arc::clone(projected_map_cache);
            let prepared_projected_maps = Arc::clone(prepared_projected_maps);
            let next_index = &next_index;
            handles.push(scope.spawn(
                move || -> Result<Vec<(usize, DirectRenderedRecipe)>, std::io::Error> {
                    let mut worker_rendered = Vec::new();
                    loop {
                        let index = next_index.fetch_add(1, Ordering::Relaxed);
                        let Some(item) = planned.get(index) else {
                            break;
                        };
                        let rendered = render_direct_recipe(
                            request,
                            latest,
                            item,
                            extracted,
                            fetch_truth_by_actual_product,
                            shared_context,
                            &contour_layer_cache,
                            &barb_layer_cache,
                            &streamline_layer_cache,
                            &barb_stride_cache,
                            &projected_map_cache,
                            &prepared_projected_maps,
                        )
                        .map_err(|err| {
                            std::io::Error::other(format!(
                                "failed rendering recipe '{}': {err}",
                                item.recipe.slug
                            ))
                        })?;
                        worker_rendered.push((index, rendered));
                    }
                    Ok(worker_rendered)
                },
            ));
        }

        for handle in handles {
            let chunk_rendered = handle
                .join()
                .map_err(|_| std::io::Error::other("parallel direct render worker panicked"))??;
            for (index, recipe) in chunk_rendered {
                rendered[index] = Some(recipe);
            }
        }
        Ok(())
    })?;

    let mut completed = Vec::with_capacity(planned.len());
    for recipe in rendered {
        completed.push(recipe.ok_or_else(|| {
            std::io::Error::other("parallel direct render worker dropped a recipe result")
        })?);
    }
    Ok(completed)
}

fn render_worker_count(recipe_count: usize) -> usize {
    if recipe_count <= 1 {
        return 1;
    }

    let override_threads = std::env::var("RUSTWX_RENDER_THREADS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0);

    thread::available_parallelism()
        .map(|count| override_threads.unwrap_or((count.get() / 2).max(1)))
        .unwrap_or(1)
        .min(recipe_count)
}

fn visual_mode_cache_key(mode: ProductVisualMode) -> u8 {
    match mode {
        ProductVisualMode::FilledMeteorology => 0,
        ProductVisualMode::UpperAirAnalysis => 1,
        ProductVisualMode::OverlayAnalysis => 2,
        ProductVisualMode::SevereDiagnostic => 3,
        ProductVisualMode::PanelMember => 4,
        ProductVisualMode::ComparisonPanel => 5,
    }
}

fn standard_projected_key(
    request: &DirectBatchRequest,
    recipe: &PlotRecipe,
) -> Option<(u32, u32, u8)> {
    let filled_selector = recipe.filled.selector?;
    let overlay_only = should_render_overlay_only(filled_selector, recipe.contours.is_some());
    let visual_mode = visual_mode_for_direct_recipe(recipe, filled_selector, overlay_only);
    Some((
        request.output_width,
        request.output_height,
        visual_mode_cache_key(visual_mode),
    ))
}

fn projected_map_cache_key(
    width: u32,
    height: u32,
    mode_key: u8,
    field: &SelectedField2D,
) -> ProjectedMapCacheKey {
    (
        width,
        height,
        mode_key,
        field.grid.shape.nx,
        field.grid.shape.ny,
        format!("{:?}", field.projection),
    )
}

fn visual_mode_for_key(mode_key: u8) -> ProductVisualMode {
    match mode_key {
        0 => ProductVisualMode::FilledMeteorology,
        1 => ProductVisualMode::UpperAirAnalysis,
        2 => ProductVisualMode::OverlayAnalysis,
        3 => ProductVisualMode::SevereDiagnostic,
        4 => ProductVisualMode::PanelMember,
        5 => ProductVisualMode::ComparisonPanel,
        _ => ProductVisualMode::FilledMeteorology,
    }
}

fn projected_sample_selector(item: &PlannedDirectRecipe) -> Option<FieldSelector> {
    if let Some(selector) = item.recipe.filled.selector {
        return Some(selector);
    }
    composite_panel_spec(item.recipe.slug).and_then(|spec| {
        spec.component_slugs.iter().find_map(|component_slug| {
            plot_recipe(component_slug).and_then(|component| component.filled.selector)
        })
    })
}

fn build_prepared_projected_maps(
    request: &DirectBatchRequest,
    planned: &[PlannedDirectRecipe],
    extracted: &HashMap<FieldSelector, SelectedField2D>,
) -> Result<PreparedProjectedMaps, Box<dyn std::error::Error>> {
    let mut prepared = HashMap::new();
    extend_prepared_projected_maps(request, planned, extracted, &mut prepared)?;
    Ok(Arc::new(prepared))
}

/// Build the projected maps `planned` needs into `prepared`, skipping keys
/// already present. The chunked render entry calls this once per chunk so
/// maps persist across chunks; each map is a pure function of its cache
/// key (output size, visual mode, grid shape/projection) and the request
/// bounds, so build order cannot change its content.
fn extend_prepared_projected_maps(
    request: &DirectBatchRequest,
    planned: &[PlannedDirectRecipe],
    extracted: &HashMap<FieldSelector, SelectedField2D>,
    prepared: &mut HashMap<ProjectedMapCacheKey, ProjectedMap>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut requested = Vec::<(ProjectedMapCacheKey, ProductVisualMode, &SelectedField2D)>::new();
    for item in planned {
        if let Some(spec) = composite_panel_spec(item.recipe.slug) {
            let spec = spec.scaled_for_request(request);
            let Some(first_field) =
                projected_sample_selector(item).and_then(|selector| extracted.get(&selector))
            else {
                continue;
            };
            requested.push((
                projected_map_cache_key(
                    spec.panel_width,
                    spec.panel_height,
                    visual_mode_cache_key(ProductVisualMode::PanelMember),
                    first_field,
                ),
                ProductVisualMode::PanelMember,
                first_field,
            ));
        } else if let Some((width, height, mode_key)) = standard_projected_key(request, item.recipe)
        {
            let Some(filled) = item
                .recipe
                .filled
                .selector
                .and_then(|selector| extracted.get(&selector))
            else {
                continue;
            };
            requested.push((
                projected_map_cache_key(width, height, mode_key, filled),
                visual_mode_for_key(mode_key),
                filled,
            ));
        }
    }

    for (cache_key, visual_mode, sample_field) in requested {
        if prepared.contains_key(&cache_key) {
            continue;
        }
        let (width, height, _, _, _, _) = cache_key.clone();
        let target_ratio = direct_map_frame_aspect_ratio(
            visual_mode,
            width,
            height,
            sample_field.projection.as_ref(),
        );
        let projected = build_projected_map_with_projection(
            &sample_field.grid.lat_deg,
            &sample_field.grid.lon_deg,
            sample_field.projection.as_ref(),
            request.domain.bounds,
            target_ratio,
        )?;
        prepared.insert(cache_key, projected);
    }
    Ok(())
}

fn render_direct_recipe(
    request: &DirectBatchRequest,
    latest: &LatestRun,
    item: &PlannedDirectRecipe,
    extracted: &HashMap<FieldSelector, SelectedField2D>,
    fetch_truth_by_actual_product: &HashMap<String, DirectFetchRuntimeInfo>,
    shared_context: Option<&dyn ProjectedMapProvider>,
    contour_layer_cache: &SharedContourLayerCache,
    barb_layer_cache: &SharedBarbLayerCache,
    streamline_layer_cache: &SharedStreamlineLayerCache,
    barb_stride_cache: &SharedBarbStrideCache,
    projected_map_cache: &SharedProjectedMapCache,
    prepared_projected_maps: &PreparedProjectedMaps,
) -> Result<DirectRenderedRecipe, Box<dyn std::error::Error>> {
    let render_start = Instant::now();
    let suffix = request
        .output_suffix
        .as_deref()
        .filter(|value| !value.trim().is_empty())
        .map(|value| format!("_{}", sanitize_output_suffix(value)))
        .unwrap_or_default();
    let output_path = request.out_dir.join(format!(
        "rustwx_{}_{}_{}z_f{:03}_{}_{}{}.png",
        request.model.as_str().replace('-', "_"),
        request.date_yyyymmdd,
        latest.cycle.hour_utc,
        request.forecast_hour,
        request.domain.slug,
        item.recipe.slug,
        suffix
    ));
    let canonical_product = canonical_fetch_product_for_selectors(
        request,
        item.plan.product.as_ref(),
        &item.plan.selectors(),
    );
    let runtime_fetch = fetch_truth_by_actual_product
        .get::<str>(canonical_product.as_str())
        .ok_or_else(|| {
            format!(
                "missing direct fetch runtime truth for canonical family '{}'",
                canonical_product
            )
        })?;
    let (
        project_ms,
        field_prepare_ms,
        contour_prepare_ms,
        barb_prepare_ms,
        request_build_ms,
        render_state_prep_ms,
        png_encode_ms,
        file_write_ms,
        state_timing,
        image_timing,
        georeference,
        georeference_absent_reason,
    ) = if let Some(spec) = composite_panel_spec(item.recipe.slug) {
        render_direct_composite_panel(
            item.recipe,
            spec.scaled_for_request(request),
            request,
            latest,
            extracted,
            &output_path,
            shared_context,
            contour_layer_cache,
            barb_layer_cache,
            streamline_layer_cache,
            barb_stride_cache,
            projected_map_cache,
            prepared_projected_maps,
        )?
    } else {
        let filled_selector = item
            .recipe
            .filled
            .selector
            .ok_or("recipe filled field missing selector binding")?;
        let filled = extracted
            .get(&filled_selector)
            .ok_or_else(|| format!("missing filled selector {:?}", filled_selector))?;

        let project_start = Instant::now();
        let overlay_only =
            should_render_overlay_only(filled_selector, item.recipe.contours.is_some());
        let visual_mode = visual_mode_for_direct_recipe(item.recipe, filled_selector, overlay_only);
        let target_ratio = direct_map_frame_aspect_ratio(
            visual_mode,
            request.output_width,
            request.output_height,
            filled.projection.as_ref(),
        );
        let render_bounds = render_bounds_for_direct_field(
            request.domain.bounds,
            filled,
            visual_mode,
            request.output_width,
            request.output_height,
        );
        let cache_key = projected_map_cache_key(
            request.output_width,
            request.output_height,
            visual_mode_cache_key(visual_mode),
            filled,
        );
        let projected = if let Some(projected) = shared_context.and_then(|ctx| {
            ctx.projected_map(request.output_width, request.output_height)
                .cloned()
        }) {
            projected
        } else if let Some(projected) = prepared_projected_maps.get(&cache_key).cloned() {
            projected
        } else if let Some(projected) = projected_map_cache
            .lock()
            .expect("projected map cache poisoned")
            .get(&cache_key)
            .cloned()
        {
            projected
        } else {
            let projected = build_projected_map_with_projection(
                &filled.grid.lat_deg,
                &filled.grid.lon_deg,
                filled.projection.as_ref(),
                request.domain.bounds,
                target_ratio,
            )?;
            projected_map_cache
                .lock()
                .expect("projected map cache poisoned")
                .insert(cache_key, projected.clone());
            projected
        };
        let project_ms = project_start.elapsed().as_millis();

        let request_build_start = Instant::now();
        let (mut render_request, build_timing) = build_render_request(
            item.recipe,
            filled,
            extracted,
            projected,
            render_bounds,
            request.output_width,
            request.output_height,
            contour_layer_cache,
            barb_layer_cache,
            streamline_layer_cache,
            barb_stride_cache,
            request.contour_mode,
            request.native_fill_level_multiplier,
        )?;
        let request_build_ms = request_build_start.elapsed().as_millis();
        // gpuwm addition (VENDOR.md): publish this panel's transform,
        // resolved from EXACTLY the arguments the
        // `build_projected_map_with_projection` call above took
        // (`request.domain.bounds` + `target_ratio` + the filled mesh) --
        // the same pairing rule the overlay pass below already relies on.
        // Set at THIS call site, not inside `build_render_request`,
        // because that function receives `render_bounds` (a possibly
        // reshaped frame) and never sees the ratio; resolving from either
        // would publish a transform for a map that was not drawn.
        render_request.resolved_projection = Some(panel_resolved_projection(
            &filled.grid.lat_deg,
            &filled.grid.lon_deg,
            filled.projection.as_ref(),
            request.domain.bounds,
            target_ratio,
        )?);
        render_request.geographic_bounds = Some(request.domain.bounds);
        apply_source_raster_policy(latest.source, &mut render_request);
        render_request.title = Some(direct_title_for_planned_product(
            request,
            item.plan.product.as_ref(),
            item.recipe.title,
        ));
        render_request.subtitle_left =
            Some(request.subtitle_left_override.clone().unwrap_or_else(|| {
                model_time_subtitle(
                    request.model,
                    &request.date_yyyymmdd,
                    latest.cycle.hour_utc,
                    request.forecast_hour,
                )
            }));
        render_request.subtitle_right = Some(
            request
                .subtitle_right_override
                .clone()
                .unwrap_or_else(|| source_subtitle(latest.source)),
        );
        if let Some(overlay) = request.place_label_overlay.as_ref() {
            crate::apply_place_label_overlay_with_density_styling(
                &mut render_request,
                overlay,
                &request.domain,
                &filled.grid.lat_deg,
                &filled.grid.lon_deg,
                filled.projection.as_ref(),
            )?;
        }
        // gpuwm addition (VENDOR.md): caller-supplied geographic overlays
        // and panel annotations, projected into the SAME frame the fill
        // was projected into (`request.domain.bounds` + `target_ratio`,
        // the two arguments the `build_projected_map_with_projection`
        // above took).  Both are `None` for every caller that does not
        // pass them, so the default path is byte-unchanged.
        if let Some(overlays) = request.geographic_overlays.as_ref() {
            overlays.apply(
                &mut render_request,
                &filled.grid.lat_deg,
                &filled.grid.lon_deg,
                filled.projection.as_ref(),
                request.domain.bounds,
                target_ratio,
            )?;
        }
        if let Some(annotations) = titles::recipe_annotations(request, item.recipe.slug, item.recipe.title) {
            annotations.apply(&mut render_request);
        }
        let save_timing = save_png_profile_with_options(
            &render_request,
            &output_path,
            &request.png_write_options(),
        )?;
        (
            project_ms,
            build_timing.field_prepare_ms,
            build_timing.contour_prepare_ms,
            build_timing.barb_prepare_ms,
            request_build_ms,
            save_timing.state_timing.state_prep_ms,
            save_timing.png_timing.png_encode_ms,
            save_timing.file_write_ms,
            save_timing.state_timing,
            save_timing.png_timing.image_timing,
            save_timing.georeference,
            save_timing.georeference_absent_reason,
        )
    };
    let content_identity = artifact_identity_from_path(&output_path)?;
    let total_ms = render_start.elapsed().as_millis();

    let panel_compose_ms = if composite_panel_spec(item.recipe.slug).is_some() {
        image_timing.total_ms
    } else {
        0
    };

    Ok(DirectRenderedRecipe {
        recipe_slug: item.recipe.slug.to_string(),
        title: direct_title_for_planned_product(
            request,
            item.plan.product.as_ref(),
            item.recipe.title,
        ),
        source_route: direct_route_for_recipe_slug(item.recipe.slug),
        grib_product: item.plan.product.to_string(),
        fetched_grib_product: runtime_fetch.fetched_product.clone(),
        resolved_source: runtime_fetch.resolved_source,
        resolved_url: runtime_fetch.resolved_url.clone(),
        output_path,
        content_identity,
        input_fetch_keys: vec![runtime_fetch.fetch_key.clone()],
        georeference,
        georeference_absent_reason,
        timing: DirectRecipeTiming {
            render_to_image_ms: image_timing.total_ms,
            data_layer_draw_ms: direct_data_layer_draw_ms(&image_timing),
            overlay_draw_ms: direct_overlay_draw_ms(&image_timing),
            panel_compose_ms,
            project_ms,
            field_prepare_ms,
            contour_prepare_ms,
            barb_prepare_ms,
            request_build_ms,
            render_state_prep_ms,
            png_encode_ms,
            file_write_ms,
            render_ms: total_ms.saturating_sub(project_ms),
            total_ms,
            state_timing,
            image_timing,
        },
    })
}

/// The sheet a composite product is laid out on under auto layout: the
/// domain's planned canvas says the grid's shape, the table says how many
/// rows and columns suit it.  `None` under fixed layout.
fn planned_composite_sheet(
    request: &DirectBatchRequest,
    members: usize,
) -> Option<rustwx_render::SheetPlan> {
    let rustwx_render::LayoutMode::Auto { class, scale } = rustwx_render::layout_mode() else {
        return None;
    };
    let plan = rustwx_render::canvas_plan_for(request.output_width, request.output_height)?;
    Some(rustwx_render::LayoutTable::builtin().plan_sheet(members, plan.grid_aspect, class, scale))
}

fn render_direct_composite_panel(
    recipe: &PlotRecipe,
    spec: CompositePanelSpec,
    request: &DirectBatchRequest,
    latest: &LatestRun,
    extracted: &HashMap<FieldSelector, SelectedField2D>,
    output_path: &std::path::Path,
    shared_context: Option<&dyn ProjectedMapProvider>,
    contour_layer_cache: &SharedContourLayerCache,
    barb_layer_cache: &SharedBarbLayerCache,
    streamline_layer_cache: &SharedStreamlineLayerCache,
    barb_stride_cache: &SharedBarbStrideCache,
    projected_map_cache: &SharedProjectedMapCache,
    prepared_projected_maps: &PreparedProjectedMaps,
) -> Result<
    (
        u128,
        u128,
        u128,
        u128,
        u128,
        u128,
        u128,
        u128,
        RenderStateTiming,
        RenderImageTiming,
        Option<rustwx_render::PanelGeoReference>,
        Option<String>,
    ),
    Box<dyn std::error::Error>,
> {
    let first_component = plot_recipe(spec.component_slugs[0])
        .ok_or_else(|| format!("missing component recipe '{}'", spec.component_slugs[0]))?;
    let first_selector = first_component
        .filled
        .selector
        .ok_or("component recipe filled field missing selector binding")?;
    let first_field = extracted
        .get(&first_selector)
        .ok_or_else(|| format!("missing component selector {:?}", first_selector))?;

    let project_start = Instant::now();
    // Under auto layout the members are laid out as a sheet of the
    // domain's own shape: one header, one shared colour bar, and each
    // member map at one cell's size.  The fixed-pixel spec stays the
    // fixed-layout path.
    let sheet = planned_composite_sheet(request, spec.component_slugs.len());
    let spec = match sheet.as_ref() {
        Some(sheet) => {
            rustwx_render::register_canvas_plan(sheet.member);
            CompositePanelSpec {
                panel_width: sheet.member.canvas_w,
                panel_height: sheet.member.canvas_h,
                ..spec
            }
        }
        None => spec,
    };
    let cache_key = projected_map_cache_key(
        spec.panel_width,
        spec.panel_height,
        visual_mode_cache_key(ProductVisualMode::PanelMember),
        first_field,
    );
    let panel_target_ratio = direct_map_frame_aspect_ratio(
        ProductVisualMode::PanelMember,
        spec.panel_width,
        spec.panel_height,
        first_field.projection.as_ref(),
    );
    let projected = if let Some(projected) = shared_context.and_then(|ctx| {
        ctx.projected_map(spec.panel_width, spec.panel_height)
            .cloned()
    }) {
        projected
    } else if let Some(projected) = prepared_projected_maps.get(&cache_key).cloned() {
        projected
    } else if let Some(projected) = projected_map_cache
        .lock()
        .expect("projected map cache poisoned")
        .get(&cache_key)
        .cloned()
    {
        projected
    } else {
        let projected = build_projected_map_with_projection(
            &first_field.grid.lat_deg,
            &first_field.grid.lon_deg,
            first_field.projection.as_ref(),
            request.domain.bounds,
            panel_target_ratio,
        )?;
        projected_map_cache
            .lock()
            .expect("projected map cache poisoned")
            .insert(cache_key, projected.clone());
        projected
    };
    let project_ms = project_start.elapsed().as_millis();

    let request_build_start = Instant::now();
    let mut build_timing = DirectRequestBuildTiming::default();
    let mut panel_requests = Vec::with_capacity(spec.component_slugs.len());
    for component_slug in spec.component_slugs {
        let component_recipe = plot_recipe(component_slug)
            .ok_or_else(|| format!("missing component recipe '{component_slug}'"))?;
        let selector = component_recipe
            .filled
            .selector
            .ok_or("component recipe filled field missing selector binding")?;
        let filled = extracted
            .get(&selector)
            .ok_or_else(|| format!("missing component selector {:?}", selector))?;
        let panel_render_bounds = render_bounds_for_direct_field(
            request.domain.bounds,
            filled,
            ProductVisualMode::PanelMember,
            spec.panel_width,
            spec.panel_height,
        );
        let (mut panel_request, panel_timing) = build_render_request(
            component_recipe,
            filled,
            extracted,
            projected.clone(),
            panel_render_bounds,
            spec.panel_width,
            spec.panel_height,
            contour_layer_cache,
            barb_layer_cache,
            streamline_layer_cache,
            barb_stride_cache,
            request.contour_mode,
            request.native_fill_level_multiplier,
        )?;
        build_timing.field_prepare_ms += panel_timing.field_prepare_ms;
        build_timing.contour_prepare_ms += panel_timing.contour_prepare_ms;
        build_timing.barb_prepare_ms += panel_timing.barb_prepare_ms;
        apply_source_raster_policy(latest.source, &mut panel_request);
        panel_request.width = spec.panel_width;
        panel_request.height = spec.panel_height;
        panel_request.visual_mode = ProductVisualMode::PanelMember;
        panel_request.subtitle_left = None;
        panel_request.subtitle_right = None;
        if let Some(overlay) = request.place_label_overlay.as_ref() {
            crate::apply_place_label_overlay_with_density_styling(
                &mut panel_request,
                overlay,
                &request.domain,
                &filled.grid.lat_deg,
                &filled.grid.lon_deg,
                filled.projection.as_ref(),
            )?;
        }
        panel_requests.push(panel_request);
    }
    let request_build_ms = request_build_start.elapsed().as_millis();

    let title = direct_panel_title_for_request(request, recipe.title);
    let subtitle_left = request.subtitle_left_override.clone().unwrap_or_else(|| {
        model_time_subtitle(
            request.model,
            &request.date_yyyymmdd,
            latest.cycle.hour_utc,
            request.forecast_hour,
        )
    });
    // Resolve once for both the planned sheet and the fallback grid, just
    // as single maps resolve their provenance through the active theme.
    let subtitle_right = rustwx_render::theme::active_theme().source_subtitle(Some(
        request.subtitle_right_override.clone()
            .unwrap_or_else(|| source_subtitle(latest.source)),
    )).unwrap_or_default();
    let render_start = Instant::now();
    let canvas = if let Some(sheet) = sheet.as_ref() {
        let labels: Vec<String> = panel_requests
            .iter()
            .map(|member| {
                rustwx_render::chrome_plan::without_domain_label(
                    member.title.as_deref().unwrap_or_default(),
                )
            })
            .collect();
        rustwx_render::render_planned_sheet(
            sheet,
            &panel_requests,
            &labels,
            rustwx_render::SheetHeader {
                title: Some(&title),
                units: None,
                subtitle_left: Some(&subtitle_left),
                subtitle_center: None,
                subtitle_right: Some(&subtitle_right),
            },
        )?
    } else {
        let layout =
            PanelGridLayout::new(spec.rows, spec.columns, spec.panel_width, spec.panel_height)?
                .with_padding(PanelPadding {
                    top: spec.top_padding,
                    ..Default::default()
                });
        let mut canvas = render_panel_grid(&layout, &panel_requests)?;
        draw_centered_text_line(&mut canvas, &title, 10, Color::BLACK, 2);
        draw_centered_text_line(
            &mut canvas,
            &format!("{subtitle_left} | {subtitle_right}"),
            35,
            Color::BLACK,
            1,
        );
        canvas
    };
    let render_ms = render_start.elapsed().as_millis();
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let save_timing =
        save_rgba_png_profile_with_options(&canvas, output_path, &request.png_write_options())?;
    Ok((
        project_ms,
        build_timing.field_prepare_ms,
        build_timing.contour_prepare_ms,
        build_timing.barb_prepare_ms,
        request_build_ms,
        save_timing.state_timing.state_prep_ms,
        save_timing.png_timing.png_encode_ms,
        save_timing.file_write_ms,
        save_timing.state_timing,
        RenderImageTiming {
            total_ms: render_ms,
            ..RenderImageTiming::default()
        },
        // A composite panel composes SEVERAL member maps into one PNG; no
        // single plot rectangle or projection describes the file, so no
        // georeference exists to publish (gpuwm addition, VENDOR.md).
        None,
        Some(
            "composite panel: one PNG composes multiple member maps; no single plot \
             rectangle or projection describes the file"
                .to_string(),
        ),
    ))
}

#[cfg(test)]
mod tests;
