//! Shared per-hour live-GRIB -> `.rws` ingest flow, used by `rw_ingest`
//! (serial hours), `rw_batch` (pipelined hours), and the UI shell's
//! ingest worker. Also the parent of the `ingest_profile` (what one run
//! fetches/extracts/computes/stores: the default `full` profile is the
//! behavior described below, unchanged) and `size_estimate` (exact +
//! predictive store/download sizing) child modules.
//!
//! Progress is reported through the [`IngestEvent`] sink on
//! [`IngestConfig`] (stage start/done walls plus the historical
//! stdout/stderr note lines: see [`crate::print_event`]); cancellation is
//! observed at stage boundaries via the config's cancel flag and surfaces
//! as [`IngestError::Cancelled`]. The store write is atomic (temp + rename
//! inside rw-store), so a cancel mid-hour never leaves partial files, at
//! worst the in-flight stage finishes and the hour is dropped before its
//! write.
//!
//! Per forecast hour: fetch the `prs` product file (cache-aware), extract
//! the full 3D isobaric superset plus the sparse per-level vorticity planes
//! AND every isobaric plane a supported direct plot recipe consumes (stored
//! as bit-exact 2D variables named by selector key: the 3D volume codec is
//! quantized, so render-grade isobaric planes must ride the lossless f32
//! 2D codec) in ONE decode pass, fetch the `sfc` product file, extract the
//! 2D surface set in ONE decode pass (plus one re-select at `hour - 1` for
//! the trailing 1 h window fields: the APCP increment, the native
//! sub-hourly MXUPHL max, and the native sub-hourly WIND 10 m max: the two
//! max fields the GRIB windowed lane consumed), then ENCODE the extracted
//! 2D planes and volumes into the spill-backed
//! [`rw_store::ingest::HourIngestWriter`] immediately (freeing the raw f32
//! planes; volumes ride deferred variable ids so the file keeps the
//! historical fields/derived/heavy/volumes order), decode the surface +
//! pressure thermo pair through the render lanes' own products decoder,
//! compute every non-heavy derived recipe grid AND (when `heavy` is set)
//! every heavy ECAPE-class recipe grid from that pair (see `ingest_compute`;
//! stored as ordinary 2D variables named by recipe slug, selector = the
//! `{"derived": ...}` marker), and finish the hour into
//! `<store-root>/<model>/<run>/f{hour:03}.rws` plus `grid.rwg` and
//! `run.json`. Output bytes are identical to the historical
//! assemble-everything-then-write flow; only the peak memory changed.
//!
//! The flow is split at the network/CPU boundary: [`fetch_hour`] (network
//! or cache disk read only) and [`process_fetched_hour`] (extract, derived,
//! heavy, encode, all CPU on the ONE global rayon pool), so `rw_batch`
//! can run hour N's compute while hour N+1 downloads. [`ingest_hour`]
//! composes the two for serial callers.
//!
//! No `nat` (wrfnat) fetch: every formerly "native-only" 2D field this plan
//! needs (composite reflectivity, 1 km AGL reflectivity, 2-5 km updraft
//! helicity, 8 m AGL smoke, column-integrated smoke, simulated IR) is also
//! carried by the HRRR `sfc` file: verified against the live
//! `hrrr.t00z.wrfsfcf06.grib2.idx` listing on AWS for the 2026-06-08 00z run,
//! so they ride the existing sfc fetch + decode pass for free instead of
//! pulling any of the ~770 MB wrfnat file.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use rustwx_core::{
    CanonicalField, CycleSpec, FieldSelector, ModelId, ModelRunRequest, SourceId, VerticalSelector,
};
use rustwx_io::{
    CachedFetchResult, FetchRequest, SharedExtractionGrid,
    extract_field_values_partial_from_model_bytes_at_forecast_hour, fetch_bytes_with_cache,
};
use rustwx_models::plot_recipe_fetch_plan;
use rustwx_products::derived::{store_derived_recipe_slugs, store_heavy_recipe_slugs};
use rustwx_products::direct::supported_direct_recipe_slugs;
use rw_store::grid::GridFile;
use rw_store::ingest::{HourIngestWriter, derived_selector, read_field_2d, read_grid_2d};
use rw_store::reader::HourReader;

#[path = "ingest_compute.rs"]
pub mod ingest_compute;
#[path = "ingest_profile.rs"]
pub mod ingest_profile;
#[path = "size_estimate.rs"]
pub mod size_estimate;
use crate::events::{IngestError, IngestEvent, IngestStage, other};
use crate::{fetch_plan, profile_scope, validate_ingest_profile_for_model};
use ingest_compute::DerivedGrid2D;
use ingest_profile::{FieldSet, IngestProfile, VolumeChoice, surface_plan};

fn ingest_selector(model: ModelId, selector: FieldSelector) -> FieldSelector {
    if model == ModelId::Hgefs && selector.product.is_default() {
        selector.with_product(rustwx_core::FieldProduct::EnsembleMean)
    } else { selector }
}

fn selector_store_metadata(
    model: ModelId,
    selector: FieldSelector,
    member: Option<u8>,
) -> Result<serde_json::Value, serde_json::Error> {
    let mut metadata = serde_json::to_value(selector)?;
    if let Some(member) = selector.product.is_default().then(|| rustwx_models::declared_member(model, member.unwrap_or(0)).expect("validated member")).flatten() {
        metadata
            .as_object_mut()
            .expect("FieldSelector serializes as an object")
            .insert(
                "ensemble_member".to_string(),
                serde_json::Value::from(member.ordinal),
            );
    }
    Ok(metadata)
}

fn volume_store_metadata(model: ModelId, field: CanonicalField, member: Option<u8>) -> serde_json::Value {
    let selector = ingest_selector(model, FieldSelector::isobaric(field, 500));
    let mut metadata = serde_json::json!({
        "field": field.as_str(),
        "vertical": "isobaric",
    });
    let object = metadata
        .as_object_mut()
        .expect("volume selector metadata is an object");
    if !selector.product.is_default() {
        object.insert(
            "product".to_string(),
            serde_json::to_value(selector.product).expect("FieldProduct is serializable"),
        );
    }
    if let Some(member) = selector.product.is_default().then(|| rustwx_models::declared_member(model, member.unwrap_or(0)).expect("validated member")).flatten() {
        object.insert(
            "ensemble_member".to_string(),
            serde_json::Value::from(member.ordinal),
        );
    }
    metadata
}

// Store schema v1 keeps source identity within each selector object. No URL
// is persisted here: only the public source enum and canonical product tokens.
fn add_source_metadata(metadata: &mut serde_json::Value, config: &IngestConfig<'_>,
    hour: u16, source: SourceId, selector: Option<FieldSelector>, pressure: bool) -> Result<(), IngestError> {
    let products: Vec<String> = if let Some(selector) = selector.filter(|_| matches!(config.profile.surface_fields, FieldSet::Selectors(_))) {
        let plan = rustwx_models::selector_fetch_plan(config.model, selector).map_err(other)?;
        vec![if selector.product.is_default() {
            rustwx_models::selected_member_product(config.model, plan.product, config.member.unwrap_or(0)).map_err(other)?
        } else { plan.product.to_string() }]
    } else {
        fetch_plan(config.model)?.iter().filter(|product| if pressure { product.pressure_source } else { product.surface_source })
            .map(|product| rustwx_models::selected_member_product(config.model, product.product, config.member.unwrap_or(0)).map_err(other))
            .collect::<Result<_, _>>()?
    };
    let object = metadata.as_object_mut().expect("selector object");
    object.insert("source_provider".into(), serde_json::to_value(source).map_err(other)?);
    object.insert("source_products".into(), serde_json::to_value(products).map_err(other)?);
    // Only member/selected boundaries above verified the received GRIB clock.
    if matches!(config.model, ModelId::Gefs | ModelId::Aigefs) || matches!(config.profile.surface_fields, FieldSet::Selectors(_)) {
        object.insert("source_cycle".into(), serde_json::json!(format!("{}{:02}", config.cycle.date_yyyymmdd, config.cycle.hour_utc)));
    }
    object.insert("source_forecast_hour".into(), serde_json::json!(hour));
    Ok(())
}

/// The volume plan under one profile: `(field, store name)` pairs in the
/// stable full-ingest order. Dewpoint falls back to RelativeHumidity
/// ("rh_iso") when the file realizes fewer than two dewpoint levels.
///
/// Moisture note: the derived/CAPE compute path (`rustwx_calc::ecape::
/// {SurfaceInputs, EcapeVolumeInputs}`) consumes mixing ratio, which
/// rustwx-products derives from dewpoint (preferred fallback after SPFH,
/// which has no CanonicalField) plus pressure, `dewpoint_iso` here and
/// `dewpoint_2m` + `surface_pressure` on the 2D side cover it, so no
/// dedicated moisture volume is needed.
fn volume_plan(profile: &IngestProfile) -> Vec<(CanonicalField, &'static str)> {
    VolumeChoice::ALL
        .iter()
        .filter(|choice| profile.volumes.contains(choice))
        .map(|choice| (choice.field(), choice.store_name()))
        .collect()
}

/// The trailing (h-1)->h window field names (see the re-select pass in
/// `process_fetched_hour`); stored only under a full-2D profile for models
/// that carry them (see [`model_has_trailing_1h_window`]).
const TRAILING_2D_NAMES: [&str; 3] = ["apcp_1h", "uh_2to5km_max_1h", "wind_speed_10m_max_1h"];

/// Whether `model` carries the HRRR-style trailing 1 h window fields
/// (`apcp_1h`, `uh_2to5km_max_1h`, `wind_speed_10m_max_1h`) the re-select
/// pass produces.
///
/// These rest on HRRR-class native sub-hourly statistical messages
/// (`MXUPHL`/`WIND ... hour max fcst`) and an HOURLY APCP accumulation. GFS
/// `pgrb2` APCP is a BUCKETED accumulation that resets every 6 h (0-6, 6-12,
/// ...), so the trailing-window re-select at `hour - 1` would NOT yield an
/// accurate 1 h precip increment -- and GFS carries no native sub-hourly UH/
/// wind-max messages at all. So GFS (and any future non-HRRR model) excludes
/// the trailing set rather than claim a 1 h field it can't produce. Accurate
/// GFS windowed QPF (bucket-difference logic) is a separate, deferred feature.
///
/// RRFS-A's `natlev.na` is genuinely HRRR-grade here: recon-verified messages
/// from `rrfs.t00z.natlev.3km.f001.na.grib2.idx` (2026-06-11 00z):
/// ```text
/// APCP:surface:0-1 hour acc fcst              (HOURLY, not GFS's 6 h reset)
/// MXUPHL:5000-2000 m above ground:0-1 hour max fcst   (UH 2-5 km)
/// WIND:10 m above ground:0-1 hour max fcst
/// ```
/// so `apcp_1h`/`uh_2to5km_max_1h`/`wind_speed_10m_max_1h` are accurate for
/// RRFS-A. The RRFS fetch plan subsets exactly these messages from `nat-na`.
fn model_has_trailing_1h_window(model: ModelId) -> bool {
    matches!(model, ModelId::Hrrr | ModelId::HrrrAk | ModelId::RrfsA)
}

/// Compute the crop-at-ingest spec for this hour, or `None` for models with no
/// crop box (HRRR/GFS: no-op) or whose grid already fits the box. The spec is
/// the contiguous index block whose cells' geographic coordinates intersect the
/// model's CONUS box, computed ONCE from the reference (grid carrier) grid so
/// every stored plane and the compute domain share the identical block.
///
/// Errors only when a crop box is configured but no cell of the grid lies in it
/// (a misconfigured box or a wrong-domain file): failing loud beats silently
/// storing an off-target or full-domain grid.
fn compute_ingest_crop(
    model: ModelId,
    reference_plane: &FieldPlane2D,
    extraction_grids: &[Vec<SharedExtractionGrid>; 3],
    hour: u16,
    config: &IngestConfig<'_>,
) -> Result<Option<rustwx_products::gridded::GridCrop>, IngestError> {
    let Some(bounds) = crate::model_crop_box(model) else {
        return Ok(None);
    };
    let (ref_pass, ref_slot) = reference_plane.grid_ref;
    let reference = &extraction_grids[ref_pass][ref_slot].grid;
    let Some(crop) = rustwx_products::gridded::grid_crop_for_bounds(reference, bounds) else {
        return Err(other(format!(
            "f{hour:03}: crop box {bounds:?} for {model} selected no grid cells \
             (wrong domain file?)"
        )));
    };
    // A crop covering the whole grid is a no-op; skip it so the bytes match a
    // crop-less ingest of the same already-CONUS grid.
    if crop.width() == reference.shape.nx && crop.height() == reference.shape.ny {
        return Ok(None);
    }
    config.emit(IngestEvent::Info {
        hour,
        message: format!(
            "f{hour:03}: crop-at-ingest {}x{} -> {}x{} (CONUS box {bounds:?})",
            reference.shape.nx,
            reference.shape.ny,
            crop.width(),
            crop.height()
        ),
    });
    Ok(Some(crop))
}

/// Slice every extraction grid, every 2D field plane, and every realized volume
/// level plane to the one crop block, in lock-step. Each distinct extraction
/// grid is cropped once; each plane's values are cropped using that plane's
/// source grid width (the pre-crop `nx`), so coords and values stay aligned.
fn apply_crop_to_extraction(
    crop: rustwx_products::gridded::GridCrop,
    extraction_grids: &mut [Vec<SharedExtractionGrid>; 3],
    fields_2d: &mut [(String, FieldPlane2D)],
    volumes_data: &mut [VolumeData],
) -> Result<(), String> {
    use rustwx_products::gridded::{crop_latlon_grid, crop_values_f32};
    // Source widths per (pass, slot), captured before any grid is replaced.
    let mut source_nx: [Vec<usize>; 3] = Default::default();
    for (pass, grids) in extraction_grids.iter().enumerate() {
        source_nx[pass] = grids.iter().map(|g| g.grid.shape.nx).collect();
    }
    // Crop each distinct extraction grid's coordinates.
    for grids in extraction_grids.iter_mut() {
        for shared in grids.iter_mut() {
            shared.grid = crop_latlon_grid(&shared.grid, crop).map_err(|err| err.to_string())?;
        }
    }
    // Crop each 2D field plane's values by its source grid width. Every plane's
    // values are exactly source nx*ny (pre-crop), so the row slice is in bounds.
    for (_, plane) in fields_2d.iter_mut() {
        let (pass, slot) = plane.grid_ref;
        let nx = source_nx[pass][slot];
        plane.values = crop_values_f32(&plane.values, nx, crop);
    }
    // Crop each realized volume level plane (all on the prs-pass grid width).
    let prs_nx = source_nx[PASS_PRS].first().copied();
    if let Some(nx) = prs_nx {
        for volume in volumes_data.iter_mut() {
            for (_, plane) in volume.levels.iter_mut() {
                *plane = crop_values_f32(plane, nx, crop);
            }
        }
    }
    Ok(())
}

/// Sparse per-level absolute vorticity planes pulled from the prs file in
/// the same decode pass as the volumes, stored as five 2D variables (the
/// direct plot recipes consume vorticity per-level, and five levels do not
/// justify a 37-level volume slot).
const VORTICITY_PLAN: &[(u16, &str)] = &[
    (200, "absolute_vorticity_200"),
    (300, "absolute_vorticity_300"),
    (500, "absolute_vorticity_500"),
    (700, "absolute_vorticity_700"),
    (850, "absolute_vorticity_850"),
];

/// Every isobaric plane a supported direct plot recipe consumes for this
/// model, derived from the recipe catalog's own fetch plans so coverage is
/// provable rather than hand-maintained. These are stored as bit-exact 2D
/// variables named by selector key (e.g. `geopotential_height_500hpa`):
/// the 3D volume codec quantizes to i16 per chunk, so a render-grade plane
/// must ride the lossless f32 2D codec to keep store-rendered plots
/// pixel-identical to the GRIB lane. Absolute vorticity is excluded: it
/// already ships through `VORTICITY_PLAN` under its stable store names.
fn direct_isobaric_plane_selectors(model: ModelId) -> Vec<FieldSelector> {
    let mut selectors = Vec::new();
    for slug in supported_direct_recipe_slugs(model) {
        let Ok(plan) = plot_recipe_fetch_plan(&slug, model) else {
            continue;
        };
        for selector in plan.selectors() {
            if matches!(selector.vertical, VerticalSelector::IsobaricHpa(_))
                && selector.field != CanonicalField::AbsoluteVorticity
                && !selectors.contains(&selector)
            {
                selectors.push(selector);
            }
        }
    }
    selectors
}

/// Everything one ingest pass needs to know, independent of any bin's CLI.
pub struct IngestConfig<'a> {
    pub model: ModelId,
    /// Individual member ordinal; None selects the model's declared default.
    pub member: Option<u8>,
    pub cycle: &'a CycleSpec,
    /// `Some` pins one source; `None` tries every configured source in
    /// catalog order, which also lets a warm raw-byte cache hit no matter
    /// which source originally served the file.
    pub source_override: Option<SourceId>,
    pub cache_root: &'a Path,
    pub use_cache: bool,
    pub store_root: &'a Path,
    pub model_slug: &'a str,
    pub run_slug: &'a str,
    /// What to fetch/extract/compute/store, validated up front (see
    /// `ingest_profile`); `IngestProfile::full()` is today's behavior.
    pub profile: &'a IngestProfile,
    /// After the write, re-open the hour and verify a bit-exact 2D
    /// round-trip of every field plus one profile per 3D variable.
    pub verify: bool,
    /// Progress sink: stage start/done walls plus the historical
    /// stdout/stderr note lines. Bins pass [`crate::print_event`] (which
    /// reproduces the old inline prints byte-for-byte); UI hosts forward
    /// events over a channel. `Sync` because `rw_batch` shares one config
    /// across its fetch/process threads.
    pub progress: &'a (dyn Fn(IngestEvent) + Sync),
    /// Checked at stage boundaries (between the prs and sfc fetches, and
    /// before each CPU stage); when set, the flow returns
    /// [`IngestError::Cancelled`]. Callers without cancellation pass
    /// [`crate::NEVER_CANCEL`].
    pub cancel: &'a AtomicBool,
}

impl IngestConfig<'_> {
    fn emit(&self, event: IngestEvent) {
        (self.progress)(event);
    }

    /// `Err(Cancelled)` once the cancel flag is set; called at every stage
    /// boundary.
    fn check_cancel(&self) -> Result<(), IngestError> {
        if self.cancel.load(Ordering::Relaxed) {
            Err(IngestError::Cancelled)
        } else {
            Ok(())
        }
    }
}

/// The network half of one hour: both family files fetched (cache-aware),
/// with per-file fetch walls. Everything here is `Send`, so a fetch thread
/// can hand it across a channel to the CPU half.
#[derive(Debug)]
pub struct FetchedHour {
    pub hour: u16,
    pub prs: CachedFetchResult,
    pub sfc: CachedFetchResult,
    pub prs_fetch_ms: u128,
    pub sfc_fetch_ms: u128,
}

/// One family file of a [`SpilledFetchedHour`]: the fetch provenance plus
/// where the raw bytes live ON DISK instead of in RAM.
#[derive(Debug)]
struct SpilledFetch {
    source: SourceId,
    url: String,
    cache_hit: bool,
    bytes_path: PathBuf,
    metadata_path: PathBuf,
    bytes_len: usize,
    /// Set when the bytes had to be written to a dedicated spill file
    /// (no readable cache copy); deleted right after rehydration.
    temp_spill: Option<PathBuf>,
}

/// A [`FetchedHour`] whose raw GRIB bytes have left RAM: queued pipeline
/// hours carry ~555 MB of raw bytes per hour, and `rw_batch`'s measured
/// peak coincides with TWO queued raw-byte sets (the channel slot plus the
/// fetch thread blocked in `send`) riding across the next hour's compute
/// peaks (~1.1 GB of the 8.2 GB baseline peak). Spilling (not gating)
/// keeps the fetch overlap fully intact, warm and cold: on a cache hit the
/// bytes are already on disk (the fetch cache file), so the spill is a
/// metadata check; without a usable cache copy the bytes are written to a
/// spill file once. Rehydration is a single sequential file read (~0.4 s
/// at HRRR size on NVMe), hidden inside the pipeline.
#[derive(Debug)]
pub struct SpilledFetchedHour {
    pub hour: u16,
    prs: SpilledFetch,
    sfc: SpilledFetch,
    prs_fetch_ms: u128,
    sfc_fetch_ms: u128,
}

fn spill_fetch(
    fetched: CachedFetchResult,
    spill_dir: &Path,
    label: &str,
) -> std::io::Result<SpilledFetch> {
    let CachedFetchResult {
        result,
        cache_hit,
        bytes_path,
        metadata_path,
    } = fetched;
    let bytes_len = result.bytes.len();
    // The fetch cache file carries exactly these bytes whenever it exists
    // with the matching length: on a cache hit the bytes were just read
    // from it, and on a cache-mediated miss the fetch wrote them to it.
    // Any doubt (missing file, length mismatch, --no-cache) falls through
    // to writing a dedicated spill file, never a wrong-bytes risk.
    let cache_copy_usable = std::fs::metadata(&bytes_path)
        .map(|meta| meta.is_file() && meta.len() == bytes_len as u64)
        .unwrap_or(false);
    let temp_spill = if cache_copy_usable {
        None
    } else {
        std::fs::create_dir_all(spill_dir)?;
        let temp_path = spill_dir.join(format!("{label}.grib2.spill"));
        std::fs::write(&temp_path, &result.bytes)?;
        Some(temp_path)
    };
    Ok(SpilledFetch {
        source: result.source,
        url: result.url,
        cache_hit,
        bytes_path,
        metadata_path,
        bytes_len,
        temp_spill,
    })
}

fn rehydrate_fetch(spilled: SpilledFetch) -> std::io::Result<CachedFetchResult> {
    let read_path = spilled.temp_spill.as_deref().unwrap_or(&spilled.bytes_path);
    let bytes = std::fs::read(read_path)?;
    if bytes.len() != spilled.bytes_len {
        return Err(std::io::Error::other(format!(
            "rehydrate {}: expected {} bytes, read {}",
            read_path.display(),
            spilled.bytes_len,
            bytes.len()
        )));
    }
    if let Some(temp_path) = &spilled.temp_spill {
        // Best effort: a leftover spill file costs disk, never correctness
        // (the next spill of the same label overwrites it).
        let _ = std::fs::remove_file(temp_path);
    }
    Ok(CachedFetchResult {
        result: rustwx_io::FetchResult {
            source: spilled.source,
            url: spilled.url,
            bytes,
        },
        cache_hit: spilled.cache_hit,
        bytes_path: spilled.bytes_path,
        metadata_path: spilled.metadata_path,
    })
}

impl FetchedHour {
    /// Move this hour's raw bytes out of RAM (see [`SpilledFetchedHour`]).
    /// `spill_dir` is only written when a family file has no usable fetch
    /// cache copy (e.g. `--no-cache`).
    pub fn spill(self, spill_dir: &Path) -> std::io::Result<SpilledFetchedHour> {
        let hour = self.hour;
        Ok(SpilledFetchedHour {
            hour,
            prs: spill_fetch(self.prs, spill_dir, &format!("f{hour:03}_prs"))?,
            sfc: spill_fetch(self.sfc, spill_dir, &format!("f{hour:03}_sfc"))?,
            prs_fetch_ms: self.prs_fetch_ms,
            sfc_fetch_ms: self.sfc_fetch_ms,
        })
    }
}

impl SpilledFetchedHour {
    /// Read the raw bytes back from disk, reproducing the exact
    /// [`FetchedHour`] that was spilled (same bytes, same provenance, same
    /// fetch walls): `process_fetched_hour` sees no difference.
    pub fn rehydrate(self) -> std::io::Result<FetchedHour> {
        Ok(FetchedHour {
            hour: self.hour,
            prs: rehydrate_fetch(self.prs)?,
            sfc: rehydrate_fetch(self.sfc)?,
            prs_fetch_ms: self.prs_fetch_ms,
            sfc_fetch_ms: self.sfc_fetch_ms,
        })
    }
}

/// One realized 3D variable summary for reporting.
#[derive(Debug, Clone)]
pub struct VolumeSummary {
    pub name: String,
    pub levels: Vec<u16>,
}

/// One ingested hour: per-stage walls, cache provenance, store stats, and
/// realized/planned counts for accurate reporting.
#[derive(Debug)]
pub struct IngestedHour {
    pub hour: u16,
    pub prs_fetch_ms: u128,
    pub sfc_fetch_ms: u128,
    pub prs_cache_hit: bool,
    pub sfc_cache_hit: bool,
    /// True when the model's fetch plan has exactly one file serving both the
    /// pressure and surface roles (GFS `pgrb2.0p25`). In that case `prs_mb`
    /// holds the one download size and `sfc_mb` is `0.0`: the download is
    /// priced once, not twice. Consumers must check this flag before summing
    /// `prs_mb + sfc_mb` to avoid double-counting.
    pub single_file_model: bool,
    pub prs_mb: f64,
    /// Zero for single-file models (`single_file_model == true`).
    pub sfc_mb: f64,
    pub prs_extract_ms: u128,
    pub sfc_extract_ms: u128,
    pub thermo_decode_ms: u128,
    pub derived_ms: u128,
    pub heavy_ms: u128,
    /// Full store-write stage wall (volume assembly + encode + fs write).
    pub write_ms: u128,
    /// The codec's own encode wall, a subset of `write_ms`.
    pub encode_ms: u128,
    /// CPU-half wall: extract through write (and verify when enabled).
    pub process_ms: u128,
    pub store_path: PathBuf,
    pub store_mb: f64,
    pub fields_2d: usize,
    pub planned_2d: usize,
    pub derived: usize,
    pub planned_derived: usize,
    pub heavy: usize,
    pub planned_heavy: usize,
    pub volumes: Vec<VolumeSummary>,
}

impl IngestedHour {
    /// Fetch + process wall (the serial cost of this hour).
    pub fn total_ms(&self) -> u128 {
        self.prs_fetch_ms + self.sfc_fetch_ms + self.process_ms
    }
}

pub fn cache_state(hit: bool) -> &'static str {
    if hit { "cache hit" } else { "cache miss" }
}

/// Download one product file (network or cache disk read; no decode, no
/// rayon), bracketed by the given stage's start/done events.
///
/// `idx_patterns` selects which GRIB messages to fetch: empty = the whole file
/// (HRRR/GFS, byte-identical to history); non-empty = the `.idx` message-subset
/// path (RRFS-A's multi-GB NA files), keyed in the fetch cache by the pattern
/// set so subsetted fetches stay cache-coherent.
fn fetch_product(
    config: &IngestConfig<'_>,
    hour: u16,
    product: &str,
    idx_patterns: &[&str],
    stage: IngestStage,
) -> Result<(CachedFetchResult, u128), IngestError> {
    let product = rustwx_models::selected_member_product(config.model, product, config.member.unwrap_or(0)).map_err(other)?;
    let fetch = FetchRequest {
        request: ModelRunRequest::new(config.model, config.cycle.clone(), hour, &product)
            .map_err(other)?,
        source_override: config.source_override,
        variable_patterns: idx_patterns.iter().map(|p| p.to_string()).collect(),
    };
    config.emit(IngestEvent::StageStarted { hour, stage });
    let fetch_started = Instant::now();
    let fetched =
        fetch_bytes_with_cache(&fetch, config.cache_root, config.use_cache).map_err(other)?;
    if let Some(member) = rustwx_models::product_member(config.model, &product) {
        rustwx_io::verify_model_member_bytes(config.model, &fetched.result.bytes, &member, config.cycle).map_err(other)?;
    }
    let fetch_ms = fetch_started.elapsed().as_millis();
    config.emit(IngestEvent::StageDone {
        hour,
        stage,
        ms: fetch_ms,
    });
    Ok((fetched, fetch_ms))
}

/// Fetch the per-model product file(s) for one hour (network or cache disk
/// read; no decode, no rayon) and bind them to the two extraction roles:
/// the pressure-source slot ([`FetchedHour::prs`]) and the surface-source
/// slot ([`FetchedHour::sfc`]). The cancel flag is observed before each
/// download; a cancel mid-download takes effect once the in-flight file
/// completes (the byte fetch has no abort hook today).
///
/// HRRR keeps its historical two-file sequence exactly: the `prs` file
/// (pressure role, `FetchPrs` stage) downloads first, then the `sfc` file
/// (surface role, `FetchSfc` stage), same URLs, same order, same events.
/// GFS's single `pgrb2.0p25` file is fetched once under the `FetchPrs`
/// stage and bound to BOTH slots (the surface slot clones it; only one
/// HTTP fetch happens), so `process_fetched_hour` reads the same bytes for
/// the surface and pressure passes.
pub fn fetch_hour(config: &IngestConfig<'_>, hour: u16) -> Result<FetchedHour, IngestError> {
    validate_ingest_profile_for_model(config.model, config.profile)?;
    if matches!(config.profile.surface_fields, FieldSet::Selectors(_)) {
        return fetch_selected_hour(config, hour);
    }
    let plan = fetch_plan(config.model)?;
    let mut pressure: Option<(CachedFetchResult, u128)> = None;
    let mut surface: Option<(CachedFetchResult, u128)> = None;
    // First entry downloads under the FetchPrs stage, the second (HRRR's
    // sfc) under FetchSfc, preserving the historical two-stage sequence;
    // GFS has a single entry and only emits FetchPrs.
    for product in &plan {
        if product.pressure_source && !product.surface_source && !config.profile.needs_prs() { continue; }
        config.check_cancel()?;
        let stage = if product.pressure_source {
            IngestStage::FetchPrs
        } else {
            IngestStage::FetchSfc
        };
        let (fetched, fetch_ms) =
            fetch_product(config, hour, product.product, product.idx_patterns, stage)?;
        // Same file serving both roles (GFS): clone into the second slot so
        // each extraction pass reads its own bytes without a second fetch.
        if product.pressure_source && product.surface_source {
            surface = Some((fetched.clone(), fetch_ms));
            pressure = Some((fetched, fetch_ms));
        } else if product.pressure_source {
            pressure = Some((fetched, fetch_ms));
        } else if product.surface_source {
            surface = Some((fetched, fetch_ms));
        }
    }
    let (sfc, sfc_fetch_ms) = surface.ok_or_else(|| {
        other(format!(
            "fetch plan for {} has no surface-source product",
            config.model
        ))
    })?;
    // The existing engine pair ABI retains an empty unused pressure slot.
    let (prs, prs_fetch_ms) = pressure.unwrap_or_else(|| (empty_pressure_slot(&sfc), 0));
    Ok(FetchedHour {
        hour,
        prs,
        sfc,
        prs_fetch_ms,
        sfc_fetch_ms,
    })
}

pub fn selected_field_fetch_requests(
    config: &IngestConfig<'_>, hour: u16,
) -> Result<Vec<FetchRequest>, IngestError> {
    let FieldSet::Selectors(selectors) = &config.profile.surface_fields else {
        return Err(other("exact field requests require a selected-plane profile"));
    };
    let mut requests: Vec<FetchRequest> = Vec::new();
    for selector in selectors {
        let plan = rustwx_models::selector_fetch_plan(config.model, *selector).map_err(other)?;
        let product = if selector.product.is_default() {
            rustwx_models::selected_member_product(config.model, plan.product, config.member.unwrap_or(0)).map_err(other)?
        } else {
            plan.product.to_string()
        };
        let patterns = plan.idx_patterns().into_iter().map(str::to_string).collect::<Vec<_>>();
        if let Some(request) = requests.iter_mut().find(|request| request.request.product == product) {
            // An empty list means structured whole-file extraction. Never turn
            // that product into an incomplete subset when another field joins.
            if request.variable_patterns.is_empty() || patterns.is_empty() {
                request.variable_patterns.clear();
            } else {
                for pattern in patterns {
                    if !request.variable_patterns.contains(&pattern) { request.variable_patterns.push(pattern); }
                }
            }
        } else {
            requests.push(FetchRequest {
                request: ModelRunRequest::new(config.model, config.cycle.clone(), hour, product).map_err(other)?,
                source_override: config.source_override, variable_patterns: patterns,
            });
        }
    }
    Ok(requests)
}

fn fetch_selected_hour(config: &IngestConfig<'_>, hour: u16) -> Result<FetchedHour, IngestError> {
    config.check_cancel()?;
    config.emit(IngestEvent::StageStarted { hour, stage: IngestStage::FetchSfc });
    let started = Instant::now();
    let mut combined: Option<CachedFetchResult> = None;
    for request in selected_field_fetch_requests(config, hour)? {
        config.check_cancel()?;
        let fetched = fetch_bytes_with_cache(&request, config.cache_root, config.use_cache).map_err(other)?;
        if let Some(member) = rustwx_models::product_member(config.model, &request.request.product) {
            rustwx_io::verify_model_member_bytes(config.model, &fetched.result.bytes, &member, config.cycle).map_err(other)?;
        }
        if let Some(prior) = &mut combined {
            if prior.result.source != fetched.result.source {
                return Err(other("selected native fields resolved through different provider sources"));
            }
            prior.result.bytes.extend_from_slice(&fetched.result.bytes);
            prior.result.url.push_str(" | ");
            prior.result.url.push_str(&fetched.result.url);
            prior.cache_hit &= fetched.cache_hit;
            // Every component has its own immutable fetch cache entry. The
            // combined stream is not the first component's cache payload.
            prior.bytes_path = PathBuf::new();
            prior.metadata_path = PathBuf::new();
        } else {
            combined = Some(fetched);
        }
    }
    let sfc = combined.ok_or_else(|| other("selected native field plan is empty"))?;
    let sfc_fetch_ms = started.elapsed().as_millis();
    config.emit(IngestEvent::StageDone { hour, stage: IngestStage::FetchSfc, ms: sfc_fetch_ms });
    let prs = empty_pressure_slot(&sfc);
    Ok(FetchedHour { hour, prs, sfc, prs_fetch_ms: 0, sfc_fetch_ms })
}

fn empty_pressure_slot(surface: &CachedFetchResult) -> CachedFetchResult {
    CachedFetchResult {
        result: rustwx_io::FetchResult { source: surface.result.source, url: String::new(), bytes: Vec::new() },
        cache_hit: surface.cache_hit, bytes_path: PathBuf::new(), metadata_path: PathBuf::new(),
    }
}

/// One owned 3D variable assembled from extraction. The level planes are
/// encoded into the hour writer right after extraction and freed (kept
/// only under `--verify`, which re-reads them for the round-trip checks);
/// the `(level, ...)` list itself survives for the realized-level summary.
struct VolumeData {
    name: &'static str,
    field: CanonicalField,
    units: &'static str,
    levels: Vec<(u16, Vec<f32>)>,
}

/// One extracted 2D plane staged for the hour write: bare values plus the
/// selector/units meta. The coordinate grid is NOT carried per field (the
/// historical `SelectedField2D` shape cost ~15 MB of identical grid clone
/// per plane, ~950 MB per hour); `grid_ref` points at the shared grid of
/// its extraction pass for the one-time identity check.
struct FieldPlane2D {
    selector: FieldSelector,
    units: String,
    values: Vec<f32>,
    /// (extraction pass, grid slot) into the per-pass shared grids.
    grid_ref: (usize, usize),
}

/// Extraction-pass indices for [`FieldPlane2D::grid_ref`].
const PASS_PRS: usize = 0;
const PASS_SFC: usize = 1;
const PASS_TRAILING: usize = 2;

/// The CPU half of one hour: extract both files, compute derived/heavy
/// grids, write the store hour, and (optionally) verify the round-trip.
/// All parallelism rides the rayon pool of the CALLING thread: the global
/// pool for the bins, or a dedicated below-normal pool when an interactive
/// host runs this inside `ThreadPool::install` (see [`crate::throttle`]).
pub fn process_fetched_hour(
    config: &IngestConfig<'_>,
    fetched: FetchedHour,
) -> Result<IngestedHour, IngestError> {
    // The CLIs validate via resolve_profile; guard programmatic callers too
    // (e.g. derived=false + heavy=true would silently compute derived).
    debug_assert!(
        config.profile.validate().is_ok(),
        "process_fetched_hour called with an unvalidated profile: {:?}",
        config.profile.validate()
    );
    validate_ingest_profile_for_model(config.model, config.profile)?;
    config.check_cancel()?;
    let process_started = Instant::now();
    let FetchedHour {
        hour,
        prs,
        sfc,
        prs_fetch_ms,
        sfc_fetch_ms,
    } = fetched;
    let member = rustwx_models::declared_member(config.model, config.member.unwrap_or(0)).map_err(other)?;
    if let FieldSet::Selectors(selectors) = &config.profile.surface_fields {
        rustwx_io::verify_model_selected_bytes(config.model, &sfc.result.bytes, member.as_ref(), config.cycle, selectors, hour).map_err(other)?;
    } else if let Some(member) = &member {
        rustwx_io::verify_model_member_bytes(config.model, &sfc.result.bytes, member, config.cycle).map_err(other)?;
        if !prs.result.bytes.is_empty() {
            rustwx_io::verify_model_member_bytes(config.model, &prs.result.bytes, member, config.cycle).map_err(other)?;
        }
    }
    let prs_cache_hit = prs.cache_hit;
    let prs_mb = prs.result.bytes.len() as f64 / (1024.0 * 1024.0);
    let sfc_cache_hit = sfc.cache_hit;
    // Single-file models (GFS): the sfc slot is a clone of the prs file; the
    // download was one fetch, not two. Price the download once by zeroing sfc_mb
    // so callers that sum prs_mb + sfc_mb get the accurate single-file total.
    let single_file_model = fetch_plan(config.model)
        .map(|plan| plan.len() == 1)
        .unwrap_or(false);
    let sfc_mb = if single_file_model && !prs.result.bytes.is_empty() {
        0.0
    } else {
        sfc.result.bytes.len() as f64 / (1024.0 * 1024.0)
    };

    // --- prs product file: 3D isobaric superset, one decode pass ---
    // The profile picks the volumes and level step; the render-grade 2D
    // planes (vorticity + direct-recipe) ride only with a full-2D profile.
    let profile = config.profile;
    let levels = profile.candidate_levels();
    let volume_plan = volume_plan(profile);
    let include_full_2d = profile.includes_full_2d();
    let direct_planes = if include_full_2d {
        direct_isobaric_plane_selectors(config.model).into_iter().map(|selector| ingest_selector(config.model, selector)).collect()
    } else {
        Vec::new()
    };
    let mut prs_selectors =
        Vec::with_capacity(volume_plan.len() * levels.len() + VORTICITY_PLAN.len());
    for (field, _) in &volume_plan {
        for &level in &levels {
            prs_selectors.push(FieldSelector::isobaric(*field, level));
        }
    }
    if include_full_2d {
        for (level, _) in VORTICITY_PLAN {
            prs_selectors.push(FieldSelector::isobaric(
                CanonicalField::AbsoluteVorticity,
                *level,
            ));
        }
    }
    // Direct-recipe isobaric planes ride the same decode pass; most are
    // already in the volume superset (T/Td/U/V/Z), but e.g. isobaric RH is
    // plane-only and joins the request here.
    for selector in &direct_planes {
        if !prs_selectors.contains(selector) {
            prs_selectors.push(*selector);
        }
    }
    let prs_selectors: Vec<_> = prs_selectors.into_iter().map(|selector| ingest_selector(config.model, selector)).collect();
    config.emit(IngestEvent::StageStarted {
        hour,
        stage: IngestStage::ExtractPrs,
    });
    let extract_started = Instant::now();
    let mut prs_extract_ms = 0u128;
    let mut volumes_data: Vec<VolumeData> = volume_plan
        .iter()
        .map(|(field, name)| VolumeData {
            name,
            field: *field,
            units: FieldSelector::isobaric(*field, 500).native_units(),
            levels: Vec::new(),
        })
        .collect();
    let mut prs_fields_2d: Vec<(String, FieldPlane2D)> = Vec::new();
    let mut prs_grids: Vec<SharedExtractionGrid> = Vec::new();
    if !prs_selectors.is_empty() {
        profile_scope!("ingest_extract_prs");
        let prs_extraction = extract_field_values_partial_from_model_bytes_at_forecast_hour(
            config.model,
            &prs.result.bytes,
            Some(&prs.bytes_path),
            &prs_selectors,
            Some(hour),
        )
        .map_err(other)?;
        prs_extract_ms = extract_started.elapsed().as_millis();
        prs_grids = prs_extraction.grids;
        for extracted in prs_extraction.extracted {
            let VerticalSelector::IsobaricHpa(level) = extracted.selector.vertical else {
                continue;
            };
            if extracted.selector.field == CanonicalField::AbsoluteVorticity {
                if let Some((_, name)) = VORTICITY_PLAN.iter().find(|(have, _)| *have == level) {
                    prs_fields_2d.push((
                        name.to_string(),
                        FieldPlane2D {
                            selector: extracted.selector,
                            units: extracted.units,
                            values: extracted.values,
                            grid_ref: (PASS_PRS, extracted.grid_index),
                        },
                    ));
                }
                continue;
            }
            // Direct-recipe planes are stored as bit-exact 2D variables under
            // their selector key, in addition to (not instead of) the volume,
            // only the values clone, the grid is shared.
            if direct_planes.contains(&extracted.selector) {
                prs_fields_2d.push((
                    extracted.selector.key(),
                    FieldPlane2D {
                        selector: extracted.selector,
                        units: extracted.units.clone(),
                        values: extracted.values.clone(),
                        grid_ref: (PASS_PRS, extracted.grid_index),
                    },
                ));
            }
            if let Some(volume) = volumes_data
                .iter_mut()
                .find(|volume| volume.field == extracted.selector.field)
            {
                // Move the plane out (bare values; nothing else to drop).
                volume.levels.push((level, extracted.values));
            }
        }
    }
    if include_full_2d {
        for (level, name) in VORTICITY_PLAN {
            if !prs_fields_2d.iter().any(|(have, _)| have == name) {
                config.emit(IngestEvent::Warning {
                    hour,
                    message: format!(
                        "f{hour:03}: 2D field '{name}' (absolute vorticity {level} hPa) \
                         missing from the prs file; skipped"
                    ),
                });
            }
        }
    }
    for selector in &direct_planes {
        let key = selector.key();
        if !prs_fields_2d.iter().any(|(have, _)| *have == key) {
            config.emit(IngestEvent::Warning {
                hour,
                message: format!(
                    "f{hour:03}: direct-recipe plane '{key}' missing from the prs file; skipped"
                ),
            });
        }
    }

    // Dewpoint fallback: when the profile stores a dewpoint volume but the
    // prs file realizes < 2 dewpoint levels, re-select RelativeHumidity
    // from the already-fetched bytes (the GRIB index re-parses, but only
    // the RH messages decode).
    let dewpoint_planned = volumes_data
        .iter()
        .any(|volume| volume.field == CanonicalField::Dewpoint);
    let dewpoint_realized = volumes_data
        .iter()
        .find(|volume| volume.field == CanonicalField::Dewpoint)
        .map(|volume| volume.levels.len())
        .unwrap_or(0);
    if dewpoint_planned && dewpoint_realized < 2 {
        config.emit(IngestEvent::Warning {
            hour,
            message: format!(
                "f{hour:03}: dewpoint_iso realized only {dewpoint_realized} level(s); \
                 falling back to relative humidity (rh_iso)"
            ),
        });
        let rh_selectors: Vec<FieldSelector> = levels
            .iter()
            .map(|&level| ingest_selector(config.model, FieldSelector::isobaric(CanonicalField::RelativeHumidity, level)))
            .collect();
        let rh_started = Instant::now();
        let rh_extraction = extract_field_values_partial_from_model_bytes_at_forecast_hour(
            config.model,
            &prs.result.bytes,
            Some(&prs.bytes_path),
            &rh_selectors,
            Some(hour),
        )
        .map_err(other)?;
        prs_extract_ms += rh_started.elapsed().as_millis();
        let dewpoint = volumes_data
            .iter_mut()
            .find(|volume| volume.field == CanonicalField::Dewpoint)
            .expect("dewpoint volume slot exists");
        dewpoint.name = "rh_iso";
        dewpoint.field = CanonicalField::RelativeHumidity;
        dewpoint.units =
            FieldSelector::isobaric(CanonicalField::RelativeHumidity, 500).native_units();
        dewpoint.levels.clear();
        for extracted in rh_extraction.extracted {
            if let VerticalSelector::IsobaricHpa(level) = extracted.selector.vertical {
                dewpoint.levels.push((level, extracted.values));
            }
        }
    }
    // `prs` stays alive: the derived/heavy compute stage decodes the
    // thermo pair from the raw prs + sfc bytes below.
    config.emit(IngestEvent::StageDone {
        hour,
        stage: IngestStage::ExtractPrs,
        ms: prs_extract_ms,
    });
    config.check_cancel()?;

    // --- sfc product file: 2D surface set (profile-filtered, plan order
    //     so the grid carrier stays the first surface-plan field), one
    //     decode pass ---
    config.emit(IngestEvent::StageStarted {
        hour,
        stage: IngestStage::ExtractSfc,
    });
    let surface_plan: Vec<(String, FieldSelector)> = if let FieldSet::Selectors(selectors) = &profile.surface_fields {
        selectors.iter().map(|selector| (selector.key(), *selector)).collect()
    } else {
        surface_plan().into_iter().filter(|(name, _)| profile.includes_surface_field(name))
            .map(|(name, selector)| (name.to_string(), ingest_selector(config.model, selector))).collect()
    };
    let sfc_selectors: Vec<FieldSelector> =
        surface_plan.iter().map(|(_, selector)| *selector).collect();
    let extract_started = Instant::now();
    let mut sfc_extraction = {
        profile_scope!("ingest_extract_sfc");
        extract_field_values_partial_from_model_bytes_at_forecast_hour(
            config.model,
            &sfc.result.bytes,
            Some(&sfc.bytes_path),
            &sfc_selectors,
            Some(hour),
        )
        .map_err(other)?
    };
    let mut sfc_extract_ms = extract_started.elapsed().as_millis();
    let sfc_grids = std::mem::take(&mut sfc_extraction.grids);

    let mut fields_2d_owned: Vec<(String, FieldPlane2D)> = Vec::new();
    for (name, selector) in &surface_plan {
        match sfc_extraction
            .extracted
            .iter()
            .position(|field| field.selector == *selector)
        {
            Some(index) => {
                let extracted = sfc_extraction.extracted.swap_remove(index);
                fields_2d_owned.push((
                    name.to_string(),
                    FieldPlane2D {
                        selector: extracted.selector,
                        units: extracted.units,
                        values: extracted.values,
                        grid_ref: (PASS_SFC, extracted.grid_index),
                    },
                ));
            }
            None if matches!(profile.surface_fields, FieldSet::Selectors(_)) => {
                return Err(other(format!("requested native field '{}' is absent at f{hour:03}", selector.key())));
            }
            None => config.emit(IngestEvent::Warning {
                hour,
                message: format!(
                    "f{hour:03}: 2D field '{name}' ({}) missing from the sfc file; skipped",
                    selector.key()
                ),
            }),
        }
    }

    // Trailing (h-1)->h window fields, all selected at `hour - 1` in ONE
    // re-select pass (the GRIB index re-parses, but only these three
    // messages decode):
    //
    // * apcp_1h, both sfc APCP accumulations end at hour h and tie on
    //   match score at hour h, where the run total wins as first in file
    //   order (stored above as `apcp_run_total`). At `hour - 1` only the
    //   trailing window matches: its start hour scores an exact 0 while
    //   the run total's start and end both miss.
    // * uh_2to5km_max_1h: the native sub-hourly MXUPHL max
    //   (`MXUPHL:5000-2000 m above ground:(h-1)-h hour max fcst`), the
    //   exact message the GRIB windowed lane reduced. The start-hour match
    //   pins the statistical message even if a future HRRR build adds an
    //   instantaneous UPHL message (which would win the plain `uh_2to5km`
    //   selection at hour h; in current sfc files MXUPHL is the only 2-5 km
    //   UH message, so `uh_2to5km` is the same plane selected by its
    //   end-hour score).
    // * wind_speed_10m_max_1h: the native sub-hourly
    //   `WIND:10 m above ground:(h-1)-h hour max fcst` field the GRIB
    //   windowed lane consumed; the sfc file carries no instantaneous wind
    //   speed message (only UGRD/VGRD), so this is the only WIND match.
    //
    // LATENT FRAGILITY (review 2026-06-09): apcp_run_total's identity rests
    // on NOAA keeping the 0->h message ahead of the (h-1)->h message in the
    // sfc file. If a future HRRR build reorders them, run_total silently
    // becomes the 1 h window. A windowed-product sanity check (run_total >=
    // 1h sum for h > 1) is the cheap detector if this ever bites.
    // GFS (and any non-HRRR model) excludes the trailing 1 h window set: its
    // bucketed APCP can't accurately produce a 1 h increment and it has no
    // native sub-hourly UH/wind-max messages (see
    // `model_has_trailing_1h_window`).
    let include_trailing = include_full_2d && model_has_trailing_1h_window(config.model);
    let mut trailing_grids: Vec<SharedExtractionGrid> = Vec::new();
    if hour >= 1 && include_trailing {
        let trailing_plan = [
            (
                TRAILING_2D_NAMES[0], // apcp_1h
                FieldSelector::surface(CanonicalField::TotalPrecipitation),
            ),
            (
                TRAILING_2D_NAMES[1], // uh_2to5km_max_1h
                FieldSelector::height_layer_agl(CanonicalField::UpdraftHelicity, 2000, 5000),
            ),
            (
                TRAILING_2D_NAMES[2], // wind_speed_10m_max_1h
                FieldSelector::height_agl(CanonicalField::WindSpeed, 10),
            ),
        ];
        let trailing_selectors: Vec<FieldSelector> = trailing_plan
            .iter()
            .map(|(_, selector)| *selector)
            .collect();
        let trailing_started = Instant::now();
        let mut trailing_extraction =
            extract_field_values_partial_from_model_bytes_at_forecast_hour(
                config.model,
                &sfc.result.bytes,
                Some(&sfc.bytes_path),
                &trailing_selectors,
                Some(hour - 1),
            )
            .map_err(other)?;
        sfc_extract_ms += trailing_started.elapsed().as_millis();
        trailing_grids = std::mem::take(&mut trailing_extraction.grids);
        for (name, selector) in &trailing_plan {
            match trailing_extraction
                .extracted
                .iter()
                .position(|field| field.selector == *selector)
            {
                Some(index) => {
                    let extracted = trailing_extraction.extracted.swap_remove(index);
                    fields_2d_owned.push((
                        name.to_string(),
                        FieldPlane2D {
                            selector: extracted.selector,
                            units: extracted.units,
                            values: extracted.values,
                            grid_ref: (PASS_TRAILING, extracted.grid_index),
                        },
                    ));
                }
                None => config.emit(IngestEvent::Warning {
                    hour,
                    message: format!(
                        "f{hour:03}: 2D field '{name}' (trailing 1 h window of {}) \
                         missing from the sfc file; skipped",
                        selector.key()
                    ),
                }),
            }
        }
    } else if include_trailing {
        config.emit(IngestEvent::Warning {
            hour,
            message: format!(
                "f{hour:03}: trailing 1 h window fields (apcp_1h, uh_2to5km_max_1h, \
                 wind_speed_10m_max_1h) have no window at analysis; skipped"
            ),
        });
    }

    config.emit(IngestEvent::StageDone {
        hour,
        stage: IngestStage::ExtractSfc,
        ms: sfc_extract_ms,
    });
    config.check_cancel()?;

    // Sparse prs-sourced planes ride behind the sfc set so the grid carrier
    // stays the first surface-plan field (both files share the model grid;
    // the store write bit-verifies that). The vorticity + direct-recipe
    // planes ride with any full-2D profile; the trailing 1 h window set
    // (apcp_1h, uh_2to5km_max_1h, wind_speed_10m_max_1h) rides only for
    // models that carry it (HRRR-class): GFS omits it (bucketed APCP).
    let planned_2d = surface_plan.len()
        + if include_full_2d {
            VORTICITY_PLAN.len() + direct_planes.len()
        } else {
            0
        }
        + if include_trailing {
            TRAILING_2D_NAMES.len()
        } else {
            0
        };
    fields_2d_owned.extend(prs_fields_2d);
    if fields_2d_owned.is_empty() {
        return Err(other(format!(
            "f{hour:03}: no 2D fields realized; cannot write an hour without a grid carrier"
        )));
    }

    let mut extraction_grids: [Vec<SharedExtractionGrid>; 3] =
        [prs_grids, sfc_grids, trailing_grids];

    // --- crop-at-ingest: for models whose native domain is larger than the
    //     store target (RRFS-A's North America rotated-pole grid), slice every
    //     extraction grid + field plane + volume level plane to one CONUS index
    //     block, computed ONCE from the reference grid so coords and values stay
    //     provably in lock-step. The same `GridCrop` drives the compute stage
    //     (`compute_product_grids`) so the derived/heavy grids land on the
    //     identical cropped grid. HRRR/GFS have no crop box → this is a no-op
    //     and their bytes stay identical. ---
    let crop_spec = compute_ingest_crop(
        config.model,
        &fields_2d_owned[0].1,
        &extraction_grids,
        hour,
        config,
    )?;
    if let Some(crop) = crop_spec {
        apply_crop_to_extraction(
            crop,
            &mut extraction_grids,
            &mut fields_2d_owned,
            &mut volumes_data,
        )
        .map_err(other)?;
    }

    // --- early encode: the 2D fields and the extracted volumes go into the
    //     (spill-backed) hour writer NOW, before the compute stages, so
    //     their raw f32 planes (~1.8 GB at HRRR size) never sit across the
    //     derived/heavy window. Volumes ride deferred variable ids, so the
    //     file bytes keep the historical fields/derived/heavy/volumes
    //     order. Under --verify the planes are kept for the round-trip
    //     checks (the historical memory shape). ---
    let mut hour_writer = {
        let (ref_pass, ref_slot) = fields_2d_owned[0].1.grid_ref;
        let first_name = fields_2d_owned[0].0.clone();
        let reference = &extraction_grids[ref_pass][ref_slot];
        // One-time grid identity check, once per distinct (pass, slot)
        // pair: every stored 2D plane must sit on the grid carrier's exact
        // coordinates, the same bit-compare the store writer used to run
        // per field (65x on identical arrays).
        let mut checked: HashSet<(usize, usize)> = HashSet::new();
        checked.insert((ref_pass, ref_slot));
        for (name, plane) in &fields_2d_owned {
            if !checked.insert(plane.grid_ref) {
                continue;
            }
            let candidate = &extraction_grids[plane.grid_ref.0][plane.grid_ref.1];
            if candidate.grid.shape != reference.grid.shape {
                return Err(other(format!(
                    "2D field '{name}': grid {}x{} does not match the hour grid {}x{}",
                    candidate.grid.shape.nx,
                    candidate.grid.shape.ny,
                    reference.grid.shape.nx,
                    reference.grid.shape.ny
                )));
            }
            let coords_match = candidate
                .grid
                .lat_deg
                .iter()
                .zip(&reference.grid.lat_deg)
                .all(|(a, b)| a.to_bits() == b.to_bits())
                && candidate
                    .grid
                    .lon_deg
                    .iter()
                    .zip(&reference.grid.lon_deg)
                    .all(|(a, b)| a.to_bits() == b.to_bits());
            if !coords_match {
                return Err(other(format!(
                    "2D field '{name}': same {}x{} dims as the first field '{first_name}' \
                     but different coordinates",
                    reference.grid.shape.nx, reference.grid.shape.ny
                )));
            }
        }
        HourIngestWriter::begin(
            config.store_root,
            config.model_slug,
            config.run_slug,
            hour,
            &reference.grid,
            reference.projection.as_ref(),
            env!("RW_BUILD_SHA"),
        )
        .map_err(other)?
    };
    let keep_planes = config.verify;
    {
        profile_scope!("ingest_encode_extracted");
        for (name, plane) in &mut fields_2d_owned {
            let mut selector = selector_store_metadata(config.model, plane.selector, config.member)
                .map_err(|err| other(format!("2D field '{name}': selector JSON: {err}")))?;
            let fetched = if plane.grid_ref.0 == PASS_PRS { &prs } else { &sfc };
            add_source_metadata(&mut selector, config, hour, fetched.result.source, Some(plane.selector), plane.grid_ref.0 == PASS_PRS)?;
            hour_writer
                .add_field_2d(name, &plane.units, selector, &plane.values)
                .map_err(other)?;
            if !keep_planes {
                plane.values = Vec::new();
            }
        }
        for volume in &mut volumes_data {
            if volume.levels.len() < 2 {
                config.emit(IngestEvent::Warning {
                    hour,
                    message: format!(
                        "f{hour:03}: 3D variable '{}' realized {} level(s) (< 2); skipped",
                        volume.name,
                        volume.levels.len()
                    ),
                });
                continue;
            }
            let levels: Vec<(u16, &[f32])> = volume
                .levels
                .iter()
                .map(|(level, plane)| (*level, plane.as_slice()))
                .collect();
            hour_writer
                .add_volume(
                    volume.name,
                    volume.units,
                    {
                        let mut metadata = volume_store_metadata(config.model, volume.field, config.member);
                        add_source_metadata(&mut metadata, config, hour, prs.result.source, None, true)?;
                        metadata
                    },
                    &levels,
                )
                .map_err(other)?;
            drop(levels);
            if !keep_planes {
                for (_, plane) in &mut volume.levels {
                    *plane = Vec::new();
                }
            }
        }
    }
    drop(extraction_grids);
    config.check_cancel()?;

    // --- derived + heavy precompute: decode the surface + pressure thermo
    //     pair from the still-resident raw bytes through the render lanes'
    //     own products decoder (same messages, same moisture preference,
    //     same f64 precision, the stored grids are bit-identical to a
    //     render-lane compute over the same files), then run every
    //     non-heavy recipe grid and every heavy ECAPE-class recipe grid ---
    let planned_derived = if profile.derived {
        store_derived_recipe_slugs().len()
    } else {
        0
    };
    let planned_heavy = if profile.heavy {
        store_heavy_recipe_slugs().len()
    } else {
        0
    };
    // Hand the raw GRIB buffers to the compute stage by value: every
    // earlier consumer (the dewpoint-fallback re-extract and the trailing
    // (h-1) re-select) has already run, and the thermo decode frees each
    // buffer at its last use instead of after both compute stages.
    let stages = compute_product_grids(
        config,
        sfc.result.bytes,
        prs.result.bytes,
        hour,
        profile.derived,
        profile.heavy,
        crop_spec,
    )?;
    config.check_cancel()?;
    let thermo_decode_ms = stages.decode_ms;
    let derived_ms = stages.derived_ms;
    let heavy_ms = stages.heavy_ms;
    let derived_grids = stages.derived;
    let heavy_grids = stages.heavy;

    // --- write: the derived/heavy grids join the already-encoded fields
    //     and volumes (ids continue in add order; the deferred volumes are
    //     numbered last at finish), then the hour streams to disk ---
    config.emit(IngestEvent::StageStarted {
        hour,
        stage: IngestStage::Write,
    });
    let write_started = Instant::now();
    for grid in derived_grids.iter().chain(heavy_grids.iter()) {
        hour_writer
            .add_derived_2d(grid.name, &grid.units, &grid.values)
            .map_err(other)?;
    }

    let unix_now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|err| other(format!("system clock before unix epoch: {err}")))?
        .as_secs();
    let written = {
        profile_scope!("ingest_write_hour");
        hour_writer.finish(unix_now).map_err(other)?
    };
    let write_ms = write_started.elapsed().as_millis();
    config.emit(IngestEvent::StageDone {
        hour,
        stage: IngestStage::Write,
        ms: write_ms,
    });

    if config.verify {
        config.emit(IngestEvent::StageStarted {
            hour,
            stage: IngestStage::Verify,
        });
        let verify_started = Instant::now();
        verify_hour(
            config,
            hour,
            &written.path,
            &fields_2d_owned,
            &derived_grids,
            &heavy_grids,
            &volumes_data,
        )
        .map_err(other)?;
        config.emit(IngestEvent::StageDone {
            hour,
            stage: IngestStage::Verify,
            ms: verify_started.elapsed().as_millis(),
        });
    }

    let volumes_summary = volumes_data
        .iter()
        .filter(|volume| volume.levels.len() >= 2)
        .map(|volume| {
            let mut realized: Vec<u16> = volume.levels.iter().map(|(level, _)| *level).collect();
            realized.sort_unstable();
            VolumeSummary {
                name: volume.name.to_string(),
                levels: realized,
            }
        })
        .collect();

    Ok(IngestedHour {
        hour,
        prs_fetch_ms,
        sfc_fetch_ms,
        prs_cache_hit,
        sfc_cache_hit,
        single_file_model,
        prs_mb,
        sfc_mb,
        prs_extract_ms,
        sfc_extract_ms,
        thermo_decode_ms,
        derived_ms,
        heavy_ms,
        write_ms,
        encode_ms: u128::from(written.encode_ms),
        process_ms: process_started.elapsed().as_millis(),
        store_path: written.path,
        store_mb: written.bytes as f64 / (1024.0 * 1024.0),
        fields_2d: fields_2d_owned.len(),
        planned_2d,
        derived: derived_grids.len(),
        planned_derived,
        heavy: heavy_grids.len(),
        planned_heavy,
        volumes: volumes_summary,
    })
}

/// Serial fetch + process for one hour (rw_ingest's flow).
pub fn ingest_hour(config: &IngestConfig<'_>, hour: u16) -> Result<IngestedHour, IngestError> {
    let fetched = fetch_hour(config, hour)?;
    process_fetched_hour(config, fetched)
}

/// Both compute stages' outputs for one hour, with per-stage wall times.
/// `decode_ms` is the products-lane thermo pair decode (the stage's input
/// assembly); `derived_ms` is the non-heavy pass; `heavy_ms` is the heavy
/// ECAPE stage alone (height-AGL prep + ECAPE triplet + composites).
#[derive(Default)]
struct ComputedProductGrids {
    derived: Vec<DerivedGrid2D>,
    heavy: Vec<DerivedGrid2D>,
    decode_ms: u128,
    derived_ms: u128,
    heavy_ms: u128,
}

/// Decode the thermo pair from the raw sfc + prs bytes through the render
/// lanes' own products decoder, then run the non-heavy derived pass and
/// (when enabled) the heavy ECAPE pass. Each stage degrades independently
/// to "no variables from this stage", never a failed ingest: the
/// extracted fields still carry the hour. Profiles with both stages off
/// (e.g. `sounding`) skip even the thermo decode. The only error is
/// `Cancelled`: the cancel flag is checked before each compute stage so a
/// cancel never waits out the (long) heavy stage.
fn compute_product_grids(
    config: &IngestConfig<'_>,
    surface_bytes: Vec<u8>,
    pressure_bytes: Vec<u8>,
    hour: u16,
    derived_enabled: bool,
    heavy_enabled: bool,
    crop: Option<rustwx_products::gridded::GridCrop>,
) -> Result<ComputedProductGrids, IngestError> {
    if !derived_enabled && !heavy_enabled {
        config.emit(IngestEvent::Info {
            hour,
            message: format!("f{hour:03}: derived/heavy compute stages skipped (profile)"),
        });
        return Ok(ComputedProductGrids::default());
    }
    config.check_cancel()?;
    config.emit(IngestEvent::StageStarted {
        hour,
        stage: IngestStage::ThermoDecode,
    });
    let decode_started = Instant::now();
    let mut inputs = {
        profile_scope!("ingest_thermo_decode");
        match ingest_compute::decode_products_inputs_cropped(surface_bytes, pressure_bytes, crop) {
            Ok(inputs) => inputs,
            Err(err) => {
                config.emit(IngestEvent::Warning {
                    hour,
                    message: format!(
                        "f{hour:03}: derived/heavy precompute skipped: thermo decode failed: {err}"
                    ),
                });
                return Ok(ComputedProductGrids::default());
            }
        }
    };
    let decode_ms = decode_started.elapsed().as_millis();
    config.emit(IngestEvent::StageDone {
        hour,
        stage: IngestStage::ThermoDecode,
        ms: decode_ms,
    });

    config.check_cancel()?;
    config.emit(IngestEvent::StageStarted {
        hour,
        stage: IngestStage::Derived,
    });
    let derived_started = Instant::now();
    let derived = {
        profile_scope!("ingest_derived");
        // `heavy_enabled` keeps the wind volumes resident for the heavy
        // ECAPE stage; without it they leave RAM as soon as the derived
        // lane's wind-consuming kernels are done (~1.13 GB off the long
        // parcel window).
        match ingest_compute::compute_derived_2d_from_inputs(&mut inputs, heavy_enabled) {
            Ok(grids) => grids,
            Err(err) => {
                config.emit(IngestEvent::Warning {
                    hour,
                    message: format!("f{hour:03}: derived precompute skipped: {err}"),
                });
                Vec::new()
            }
        }
    };
    let derived_ms = derived_started.elapsed().as_millis();
    config.emit(IngestEvent::StageDone {
        hour,
        stage: IngestStage::Derived,
        ms: derived_ms,
    });

    if !heavy_enabled {
        config.emit(IngestEvent::Info {
            hour,
            message: format!("f{hour:03}: heavy ingest stage skipped (profile/--no-heavy)"),
        });
        return Ok(ComputedProductGrids {
            derived,
            heavy: Vec::new(),
            decode_ms,
            derived_ms,
            heavy_ms: 0,
        });
    }

    config.check_cancel()?;
    config.emit(IngestEvent::StageStarted {
        hour,
        stage: IngestStage::Heavy,
    });
    let heavy_started = Instant::now();
    let heavy = {
        profile_scope!("ingest_heavy");
        // By value: heavy is the last compute stage, and the products lane
        // frees the f64 thermo volumes the moment the kernels finish.
        match ingest_compute::compute_heavy_2d_from_inputs(inputs) {
            Ok(heavy) => {
                for (slug, reason) in &heavy.skipped {
                    config.emit(IngestEvent::Warning {
                        hour,
                        message: format!("f{hour:03}: heavy recipe '{slug}' skipped: {reason}"),
                    });
                }
                if heavy.ecape_failure_count > 0 {
                    config.emit(IngestEvent::Warning {
                        hour,
                        message: format!(
                            "f{hour:03}: ECAPE triplet failed on {} column(s) (NaN in grids, \
                             same as the render lane)",
                            heavy.ecape_failure_count
                        ),
                    });
                }
                config.emit(IngestEvent::Info {
                    hour,
                    message: format!(
                        "f{hour:03}: heavy breakdown: height-AGL prep {} ms | ECAPE triplet {} ms | \
                         wind diagnostics {} ms | ML classic (STP LCL) {} ms | composites {} ms",
                        heavy.timing.prepare_height_agl_ms,
                        heavy.timing.kernels.ecape_triplet_ms,
                        heavy.timing.kernels.wind_diagnostics_ms,
                        heavy.timing.kernels.ml_classic_ms,
                        heavy.timing.kernels.composites_ms,
                    ),
                });
                heavy.grids
            }
            Err(err) => {
                config.emit(IngestEvent::Warning {
                    hour,
                    message: format!("f{hour:03}: heavy precompute skipped: {err}"),
                });
                Vec::new()
            }
        }
    };
    let heavy_ms = heavy_started.elapsed().as_millis();
    config.emit(IngestEvent::StageDone {
        hour,
        stage: IngestStage::Heavy,
        ms: heavy_ms,
    });

    Ok(ComputedProductGrids {
        derived,
        heavy,
        decode_ms,
        derived_ms,
        heavy_ms,
    })
}

/// Re-open the just-written hour: bit-exact round-trip of every 2D field
/// plus the first derived and first heavy variable (via `read_grid_2d`,
/// marker selector checked), plus one center-of-grid profile per 3D
/// variable, each profile value checked against the source plane's center
/// value within the quantization bound. Only runs under `--verify`, which
/// is also the mode that keeps the source planes resident after encode.
#[allow(clippy::too_many_arguments)]
fn verify_hour(
    config: &IngestConfig<'_>,
    hour: u16,
    hour_path: &Path,
    fields_2d: &[(String, FieldPlane2D)],
    derived_grids: &[DerivedGrid2D],
    heavy_grids: &[DerivedGrid2D],
    volumes_data: &[VolumeData],
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let reader = HourReader::open(hour_path)?;
    let grid_path = hour_path
        .parent()
        .ok_or("hour path has no parent directory")?
        .join("grid.rwg");
    let grid = GridFile::open(&grid_path)?;

    // Every extracted 2D field must round-trip bit-exactly, not just the
    // first. (A sub-epsilon constant-tile shortcut in the f32 codec once
    // flattened the 8 m smoke plane while temperature_2m verified clean.)
    for (name, original) in fields_2d {
        let round_trip = read_field_2d(&reader, &grid, name)?;
        let exact = round_trip.values.len() == original.values.len()
            && round_trip
                .values
                .iter()
                .zip(&original.values)
                .all(|(a, b)| a.to_bits() == b.to_bits());
        if !exact {
            return Err(format!("verify: 2D round-trip of '{name}' is not bit-exact").into());
        }
    }
    let name = &fields_2d[0].0;

    let mut derived_note = String::from("no derived vars");
    if let Some(derived) = derived_grids.first() {
        verify_marked_grid(&reader, &grid, derived)?;
        derived_note = format!("derived '{}' bit-exact", derived.name);
    }
    let mut heavy_note = String::from("no heavy vars");
    if let Some(heavy) = heavy_grids.first() {
        verify_marked_grid(&reader, &grid, heavy)?;
        heavy_note = format!("heavy '{}' bit-exact", heavy.name);
    }

    // Integer grid center: at exact integer coordinates the bilinear profile
    // degenerates to the stored column at that point, so each profile value
    // compares directly against the source plane's center value.
    let (cx, cy) = (grid.nx / 2, grid.ny / 2);
    let center = cy * grid.nx + cx;
    let mut profiles = Vec::new();
    for var in &reader.meta().variables {
        if var.kind != "pressure3d" {
            continue;
        }
        let profile = reader.read_profile_3d(&var.name, cx as f64, cy as f64)?;
        if profile.len() != var.levels_hpa.len() {
            return Err(format!(
                "verify: profile of '{}' returned {} values for {} levels",
                var.name,
                profile.len(),
                var.levels_hpa.len()
            )
            .into());
        }
        let source = volumes_data
            .iter()
            .find(|volume| volume.name == var.name)
            .ok_or_else(|| {
                format!(
                    "verify: no source volume for stored variable '{}'",
                    var.name
                )
            })?;
        // Conservative quantization bound: the 3D codec quantizes whole
        // [y][x][z] chunks with one scale per chunk, so every chunk's scale
        // is <= (global variable range) / 65534; half of that is the max
        // rounding error, leaving generous headroom for f32 decode noise.
        let mut vmin = f32::INFINITY;
        let mut vmax = f32::NEG_INFINITY;
        for (_, plane) in &source.levels {
            for value in plane {
                if value.is_finite() {
                    vmin = vmin.min(*value);
                    vmax = vmax.max(*value);
                }
            }
        }
        let bound = if vmax >= vmin {
            (vmax - vmin) / 65534.0 + 1e-3
        } else {
            1e-3 // no finite source values: only NaN-vs-NaN checks remain
        };
        for (k, &level) in var.levels_hpa.iter().enumerate() {
            let (_, plane) = source
                .levels
                .iter()
                .find(|(have, _)| *have == level)
                .ok_or_else(|| {
                    format!(
                        "verify: stored level {level} hPa of '{}' missing from the source volume",
                        var.name
                    )
                })?;
            let expected = plane[center];
            let got = profile[k];
            if expected.is_nan() {
                if !got.is_nan() {
                    return Err(format!(
                        "verify: '{}' {level} hPa at grid center: source is NaN \
                         but profile is {got}",
                        var.name
                    )
                    .into());
                }
                continue;
            }
            let diff = (got - expected).abs();
            if diff.is_nan() || diff > bound {
                return Err(format!(
                    "verify: '{}' {level} hPa at grid center: profile {got} vs source \
                     {expected} exceeds quantization bound {bound}",
                    var.name
                )
                .into());
            }
        }
        profiles.push(format!("{}:{}", var.name, profile.len()));
    }
    config.emit(IngestEvent::Info {
        hour,
        message: format!(
            "  verify ok: all {} 2D fields bit-exact (first '{name}'), {derived_note}, \
             {heavy_note}, profiles at grid center [{}], values within quantization bound",
            fields_2d.len(),
            profiles.join(" ")
        ),
    });
    Ok(())
}

/// Bit-exact round-trip of one derived/heavy grid via `read_grid_2d`,
/// including the `{"derived": slug}` marker selector.
fn verify_marked_grid(
    reader: &HourReader,
    grid: &GridFile,
    expected: &DerivedGrid2D,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let stored = read_grid_2d(reader, grid, expected.name)?;
    if stored.selector != derived_selector(expected.name) {
        return Err(format!(
            "verify: derived '{}' stored selector {} is not the derived marker",
            expected.name, stored.selector
        )
        .into());
    }
    let exact = stored.values.len() == expected.values.len()
        && stored
            .values
            .iter()
            .zip(&expected.values)
            .all(|(a, b)| a.to_bits() == b.to_bits());
    if !exact {
        return Err(format!(
            "verify: derived round-trip of '{}' is not bit-exact",
            expected.name
        )
        .into());
    }
    Ok(())
}

/// The store variables one profile plans to write per hour, by stable
/// store name: the single source of truth the size estimator prices.
/// `volumes` carries the planned (candidate) level count; 2D names ride in
/// stored order (surface plan, trailing windows, vorticity planes,
/// direct-recipe planes), then the derived and heavy recipe slugs.
pub struct PlannedStoreVariables {
    pub volumes: Vec<(&'static str, usize)>,
    pub fields_2d: Vec<String>,
    pub derived: Vec<&'static str>,
    pub heavy: Vec<&'static str>,
}

/// Resolve one profile into the variables it plans to store for `model`.
/// Predictive by construction: candidate levels are assumed realized (true
/// for HRRR's 25 hPa prs files) and the dewpoint volume keeps its planned
/// `dewpoint_iso` name (the rh_iso fallback is a per-file degradation).
pub fn planned_store_variables(profile: &IngestProfile, model: ModelId) -> PlannedStoreVariables {
    if let FieldSet::Selectors(selectors) = &profile.surface_fields {
        return PlannedStoreVariables { volumes: Vec::new(), fields_2d: selectors.iter().map(|selector| selector.key()).collect(), derived: Vec::new(), heavy: Vec::new() };
    }
    let level_count = profile.candidate_levels().len();
    let volumes = volume_plan(profile)
        .iter()
        .map(|(_, name)| (*name, level_count))
        .collect();
    let mut fields_2d: Vec<String> = surface_plan()
        .iter()
        .filter(|(name, _)| profile.includes_surface_field(name))
        .map(|(name, _)| (*name).to_string())
        .collect();
    if profile.includes_full_2d() {
        // Trailing 1 h window fields ride only for models that carry them
        // (HRRR-class); GFS omits them (bucketed APCP: see
        // `model_has_trailing_1h_window`).
        if model_has_trailing_1h_window(model) {
            fields_2d.extend(TRAILING_2D_NAMES.iter().map(|name| (*name).to_string()));
        }
        fields_2d.extend(VORTICITY_PLAN.iter().map(|(_, name)| (*name).to_string()));
        fields_2d.extend(
            direct_isobaric_plane_selectors(model)
                .iter()
                .map(|selector| selector.key()),
        );
    }
    PlannedStoreVariables {
        volumes,
        fields_2d,
        derived: if profile.derived {
            store_derived_recipe_slugs()
        } else {
            Vec::new()
        },
        heavy: if profile.heavy {
            store_heavy_recipe_slugs()
        } else {
            Vec::new()
        },
    }
}

/// Parse the `--hours` spec: a single hour ("6"), an inclusive range
/// ("0-6"), or a comma list of either ("12,0,4-6"). The output is sorted
/// ascending and deduplicated.
pub fn parse_hours(spec: &str) -> Result<Vec<u16>, Box<dyn std::error::Error>> {
    fn parse_token(token: &str) -> Result<u16, Box<dyn std::error::Error>> {
        token.parse().map_err(|_| {
            format!("--hours: invalid token '{token}' (expected N, N-M, or comma list)").into()
        })
    }
    let mut hours = Vec::new();
    for part in spec.split(',') {
        let part = part.trim();
        if part.is_empty() {
            return Err(format!(
                "--hours: empty entry in '{spec}' (expected N, N-M, or comma list)"
            )
            .into());
        }
        if let Some((start, end)) = part.split_once('-') {
            let start = parse_token(start.trim())?;
            let end = parse_token(end.trim())?;
            if start > end {
                return Err(format!("--hours: invalid range '{part}': start > end").into());
            }
            hours.extend(start..=end);
        } else {
            hours.push(parse_token(part)?);
        }
    }
    hours.sort_unstable();
    hours.dedup();
    if hours.is_empty() {
        return Err("pass at least one forecast hour via --hours".into());
    }
    Ok(hours)
}

/// Validate that the run cycle and every requested forecast hour sit on the
/// model's published grids, delegating to rustwx-models' authoritative cycle
/// table (`model_summary(model).cycle_hours_utc`) and
/// [`rustwx_models::supported_forecast_hours`] cadence table (so each rule has
/// exactly one home). This is the spec-resolution gate the bins run after
/// `parse_hours`:
///
/// * **Cycle:** GFS publishes only `[0, 6, 12, 18]z`, so `--cycle 3` is
///   rejected here. Without this gate the cycle flows straight into
///   `build_gfs_url`, which does NOT validate it, building a non-existent
///   `gfs.t03z...` URL that only fails as a download 404. HRRR's all-24-hour
///   cycle table accepts every cycle, so HRRR is unaffected.
/// * **Forecast hour:** GFS is hourly to f120 then 3-hourly to f384, so f121
///   is rejected while f123/f384 pass; HRRR's 0..=18/0..=48 range is
///   unchanged. (`ModelRunRequest::new`/`build_gfs_url` do NOT gate the GFS
///   hour either, so without this an off-cadence hour would silently build a
///   non-existent URL.)
pub fn validate_forecast_hours(
    model: ModelId,
    cycle_hour_utc: u8,
    hours: &[u16],
) -> Result<(), Box<dyn std::error::Error>> {
    let cycles = rustwx_models::model_summary(model).cycle_hours_utc;
    if !cycles.contains(&cycle_hour_utc) {
        let valid = cycles
            .iter()
            .map(|cycle| format!("{cycle:02}z"))
            .collect::<Vec<_>>()
            .join(", ");
        return Err(format!(
            "--cycle: {cycle_hour_utc:02}z is not a published {model} cycle \
             (valid cycles: {valid})"
        )
        .into());
    }
    let supported = rustwx_models::supported_forecast_hours(model, cycle_hour_utc);
    if let Some(&bad) = hours.iter().find(|hour| !supported.contains(hour)) {
        let max = supported.last().copied().unwrap_or(0);
        let cadence = forecast_hour_cadence_note(model, cycle_hour_utc, max);
        return Err(format!(
            "--hours: f{bad:03} is not a valid {model} forecast hour for the {cycle_hour_utc:02}z \
             cycle (valid hours run 0..={max}; {cadence})"
        )
        .into());
    }
    Ok(())
}

fn forecast_hour_cadence_note(model: ModelId, cycle_hour_utc: u8, max: u16) -> String {
    match model {
        ModelId::Hrrr | ModelId::HrrrAk => {
            if cycle_hour_utc % 6 == 0 {
                "hourly to f048 on 00/06/12/18z cycles".to_string()
            } else {
                "hourly to f018 on off-synoptic cycles".to_string()
            }
        }
        ModelId::Gfs => "GFS is hourly to f120, then 3-hourly to f384".to_string(),
        ModelId::Gdas => "GDAS is hourly to f009".to_string(),
        ModelId::Gefs => "GEFS is 3-hourly to f240, then 6-hourly to f384".to_string(),
        ModelId::Aigfs | ModelId::Aigefs => "AI global feeds are 6-hourly".to_string(),
        ModelId::Hgefs => "HGEFS is 6-hourly to f240".to_string(),
        ModelId::EcmwfOpenData => {
            "ECMWF Open Data is 3-hourly to f144, then 6-hourly on 00/12z cycles".to_string()
        }
        ModelId::Rap => "RAP is hourly; 03/09/15/21z cycles extend to f051".to_string(),
        ModelId::Nam => "NAM is hourly to f036, then 3-hourly to f084".to_string(),
        ModelId::RrfsA => "RRFS-A is hourly to f060".to_string(),
        _ => format!("see the model cadence table for supported hours through f{max:03}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires retained public AIGEFS files; no network"]
    fn retained_provider_individual_member_roundtrip_and_wrong_member_rejection() {
        let fixtures = PathBuf::from(std::env::var_os("ARWEN_AIGEFS_FIXTURES").expect("fixtures"));
        let output = PathBuf::from(std::env::var_os("ARWEN_AIGEFS_PROOF_DIR").expect("proof output")).join("normalized-individual");
        let cycle = CycleSpec::new("20260817", 0).unwrap();
        let selector = FieldSelector::height_agl(CanonicalField::Temperature, 2);
        let profile = IngestProfile { volumes:Vec::new(),level_step_hpa:25,
            surface_fields:FieldSet::Named(vec!["temperature_2m".into()]),derived:false,heavy:false };
        let cache = output.join("cache"); let stores=output.join("stores");
        let sink=|_event:IngestEvent|{}; let cancel=AtomicBool::new(false);
        let mut values = Vec::new(); let mut cache_paths=Vec::new();
        for member in [0u8,1] {
            let run = format!("member-{member}");
            let config = IngestConfig { model:ModelId::Aigefs,member:Some(member),cycle:&cycle,
                source_override:Some(SourceId::Aws),cache_root:&cache,use_cache:true,store_root:&stores,
                model_slug:"aigefs",run_slug:&run,profile:&profile,verify:true,progress:&sink,cancel:&cancel };
            let product = rustwx_models::selected_member_product(ModelId::Aigefs,"sfc/mem000",member).unwrap();
            let request=FetchRequest { request:ModelRunRequest::new(ModelId::Aigefs,cycle.clone(),0,&product).unwrap(),
                source_override:Some(SourceId::Aws),variable_patterns:Vec::new() };
            let bytes=std::fs::read(fixtures.join(format!("mem{member:03}.aigefs.t00z.sfc.f000.grib2"))).unwrap();
            let expected=extract_field_values_partial_from_model_bytes_at_forecast_hour(ModelId::Aigefs,&bytes,None,&[selector],Some(0)).unwrap().extracted.remove(0).values;
            let url=rustwx_models::resolve_urls(&request.request).unwrap().into_iter().find(|url|url.source==SourceId::Aws).unwrap().grib_url;
            rustwx_io::store_cached_fetch(&cache,&request,&rustwx_io::FetchResult {source:SourceId::Aws,url,bytes}).unwrap();
            cache_paths.push(rustwx_io::fetch_cache_paths(&cache,&request));
            let fetched=fetch_hour(&config,0).unwrap();
            assert!(fetched.prs.result.bytes.is_empty() && fetched.sfc.cache_hit);
            let result=process_fetched_hour(&config,fetched).unwrap();
            let reader=HourReader::open(&result.store_path).unwrap();
            let grid=GridFile::open(&result.store_path.parent().unwrap().join("grid.rwg")).unwrap();
            let actual=read_field_2d(&reader,&grid,"temperature_2m").unwrap();
            assert_eq!(actual.values.iter().map(|v|v.to_bits()).collect::<Vec<_>>(), expected.iter().map(|v|v.to_bits()).collect::<Vec<_>>());
            let metadata=&reader.variable("temperature_2m").unwrap().selector;
            assert_eq!(metadata["ensemble_member"],member);
            assert_eq!(metadata["source_products"],serde_json::json!([product]));
            assert_eq!(metadata["source_cycle"],"2026081700");
            values.push(expected);
        }
        assert_ne!(cache_paths[0],cache_paths[1]);
        assert!(values[0].iter().zip(&values[1]).any(|(a,b)|a.is_finite()&&b.is_finite()&&a!=b));
        let wrong_stores=output.join("wrong-member-stores");
        let config=IngestConfig {model:ModelId::Aigefs,member:Some(0),cycle:&cycle,source_override:Some(SourceId::Aws),
            cache_root:&cache,use_cache:true,store_root:&wrong_stores,model_slug:"aigefs",run_slug:"wrong-member",profile:&profile,verify:true,progress:&sink,cancel:&cancel};
        let received=CachedFetchResult {result:rustwx_io::FetchResult {source:SourceId::Aws,url:"retained-wrong-member".into(),
            bytes:std::fs::read(fixtures.join("mem001.aigefs.t00z.sfc.f000.grib2")).unwrap()},cache_hit:false,bytes_path:PathBuf::new(),metadata_path:PathBuf::new()};
        let err=process_fetched_hour(&config,FetchedHour {hour:0,prs:empty_pressure_slot(&received),sfc:received,prs_fetch_ms:0,sfc_fetch_ms:0}).unwrap_err().to_string();
        assert!(err.contains("member"),"{err}"); assert!(!wrong_stores.exists());
        std::fs::write(output.join("receipt.json"),serde_json::to_vec_pretty(&serde_json::json!({"status":"PASS","members":[0,1],"fields_2d_bit_exact":true,
            "cache_keys_distinct":true,"wrong_member":err,"rws_schema_preserved":true})).unwrap()).unwrap();
    }


    #[test]
    fn selected_statistics_resolve_products_without_member_substitution() {
        let temperature = FieldSelector::height_agl(CanonicalField::Temperature, 2);
        let selectors = vec![temperature.with_ensemble_mean(), temperature.with_ensemble_standard_deviation()];
        let profile = IngestProfile::selected(selectors.clone());
        let cycle = CycleSpec::new("20260817", 0).unwrap();
        let cancel = AtomicBool::new(false);
        let sink = |_event: IngestEvent| {};
        for (model, expected) in [(ModelId::Gefs, ["pgrb2ap5/geavg", "pgrb2ap5/gespr"]),
            (ModelId::Aigefs, ["sfc/avg", "sfc/spr"]), (ModelId::Hgefs, ["sfc/avg", "sfc/spr"])] {
            let (source, hour) = if model == ModelId::Gefs { (SourceId::Aws, 0) } else { (SourceId::Nomads, 6) };
            let config = IngestConfig { member: Some(17), model, cycle: &cycle,
                source_override: Some(source), cache_root: Path::new("unused-cache"), use_cache: true,
                store_root: Path::new("unused-store"), model_slug: model.as_str(), run_slug: "test",
                profile: &profile, verify: true, progress: &sink, cancel: &cancel };
            validate_ingest_profile_for_model(model, &profile).unwrap();
            let requests = selected_field_fetch_requests(&config, hour).unwrap();
            assert_eq!(requests.iter().map(|request| request.request.product.as_str()).collect::<Vec<_>>(), expected);
            assert!(requests.iter().all(|request| !request.variable_patterns.is_empty()));
            assert!(requests.iter().all(|request| rustwx_models::resolve_urls(&request.request).unwrap()
                .iter().any(|url| url.source == source && !url.grib_url.starts_with("unsupported:"))));
            for selector in &selectors {
                let metadata = selector_store_metadata(model, *selector, config.member).unwrap();
                assert!(metadata.get("ensemble_member").is_none());
                assert_eq!(metadata["product"], serde_json::to_value(selector.product).unwrap());
            }
            assert_eq!(planned_store_variables(&profile, model).fields_2d,
                selectors.iter().map(|selector| selector.key()).collect::<Vec<_>>());
        }
        let pressure = IngestProfile::selected(vec![FieldSelector::isobaric(CanonicalField::Temperature, 500).with_ensemble_mean()]);
        assert!(!pressure.needs_prs()); // Native 2-D plane; no broad volume fetch.
        assert!(validate_ingest_profile_for_model(ModelId::Gefs, &pressure).is_ok());
    }

    #[test]
    fn global_ensemble_store_metadata_identifies_statistics_and_individual_members() {
        let base = FieldSelector::isobaric(CanonicalField::Temperature, 500);
        for model in [ModelId::Hgefs] {
            let selector = ingest_selector(model, base);
            assert_eq!(selector.product, rustwx_core::FieldProduct::EnsembleMean);
            let metadata = selector_store_metadata(model, selector, None).unwrap();
            assert_eq!(metadata["product"], "ensemble_mean");
            let volume = volume_store_metadata(model, CanonicalField::Temperature, None);
            assert_eq!(volume["product"], "ensemble_mean");
            assert!(volume.get("ensemble_member").is_none());
        }

        let selector = ingest_selector(ModelId::Gefs, base);
        assert!(selector.product.is_default());
        let metadata = selector_store_metadata(ModelId::Gefs, selector, None).unwrap();
        assert_eq!(metadata["ensemble_member"], 0);
        assert!(metadata.get("product").is_none());
        let volume = volume_store_metadata(ModelId::Gefs, CanonicalField::Temperature, None);
        assert_eq!(volume["ensemble_member"], 0);
        assert!(volume.get("product").is_none());
        let selected = selector_store_metadata(ModelId::Aigefs, base, Some(17)).unwrap();
        assert_eq!(selected["ensemble_member"],17);
        assert!(selected.get("product").is_none());

        assert!(
            ingest_selector(ModelId::Aigfs, base).product.is_default(),
            "deterministic AIGFS must remain a default product"
        );
    }

    #[test]
    #[ignore = "requires retained public provider fixtures; no network"]
    fn retained_provider_rrfs_ingest_rotates_native_winds_before_storage() {
        let fixtures = PathBuf::from(std::env::var_os("ARWEN_AIGEFS_FIXTURES").expect("fixtures"));
        let output = PathBuf::from(std::env::var_os("ARWEN_AIGEFS_PROOF_DIR").expect("proof output")).join("normalized-rrfs");
        std::fs::create_dir_all(&output).unwrap();
        let bytes = std::fs::read(fixtures.join("rrfs-ops-f000-mslp-winds.grib2")).unwrap();
        let selectors = [FieldSelector::mean_sea_level(CanonicalField::PressureReducedToMeanSeaLevel),
            FieldSelector::height_agl(CanonicalField::UWind, 10), FieldSelector::height_agl(CanonicalField::VWind, 10)];
        let mut direct = rustwx_io::extract_fields_partial_from_model_bytes_at_forecast_hour(
            ModelId::Rrfs, &bytes, None, &selectors, Some(0)).unwrap().extracted;
        assert_eq!(direct.len(), 3);
        let original_u = direct.iter().find(|field| field.selector == selectors[1]).unwrap().values.clone();
        let original_v = direct.iter().find(|field| field.selector == selectors[2]).unwrap().values.clone();
        rustwx_io::rotate_normalized_grid_relative_wind_fields_to_earth(ModelId::Rrfs, &mut direct).unwrap();
        let earth_u = &direct.iter().find(|field| field.selector == selectors[1]).unwrap().values;
        let earth_v = &direct.iter().find(|field| field.selector == selectors[2]).unwrap().values;
        assert!(original_u.iter().zip(earth_u).any(|(raw, earth)| raw.is_finite() && (raw-earth).abs() > 0.1));
        for ((u,v),(east,north)) in original_u.iter().zip(&original_v).zip(earth_u.iter().zip(earth_v)) {
            if [u,v,east,north].iter().all(|value|value.is_finite()) {
                assert!((u.hypot(*v)-east.hypot(*north)).abs() < 1e-4);
            }
        }
        let cycle = CycleSpec::new("20260817", 0).unwrap();
        let profile = IngestProfile { volumes: Vec::new(), level_step_hpa: 25,
            surface_fields: FieldSet::Named(vec!["mslp".into(), "u_10m".into(), "v_10m".into()]),
            derived: false, heavy: false };
        let cache = output.join("cache"); let stores = output.join("stores");
        let cancel = AtomicBool::new(false); let sink = |_event: IngestEvent| {};
        let config = IngestConfig { member: None, model: ModelId::Rrfs, cycle: &cycle,
            source_override: Some(SourceId::Aws), cache_root: &cache, use_cache: true,
            store_root: &stores, model_slug: "rrfs", run_slug: "operational-native",
            profile: &profile, verify: true, progress: &sink, cancel: &cancel };
        let plan = fetch_plan(ModelId::Rrfs).unwrap();
        assert_eq!(plan.len(), 1); assert!(!plan[0].pressure_source);
        let request = FetchRequest { request: ModelRunRequest::new(ModelId::Rrfs, cycle.clone(), 0, plan[0].product).unwrap(),
            source_override: Some(SourceId::Aws), variable_patterns: plan[0].idx_patterns.iter().map(|pattern|pattern.to_string()).collect() };
        let url = rustwx_models::resolve_urls(&request.request).unwrap().into_iter().find(|url|url.source == SourceId::Aws).unwrap().grib_url;
        assert!(url.contains("noaa-rrfs-ops-pds"));
        rustwx_io::store_cached_fetch(&cache,&request,&rustwx_io::FetchResult {source:SourceId::Aws,url,bytes}).unwrap();
        let fetched = fetch_hour(&config, 0).unwrap(); assert!(fetched.prs.result.bytes.is_empty() && fetched.sfc.cache_hit);
        let result = process_fetched_hour(&config, fetched).unwrap();
        let reader = HourReader::open(&result.store_path).unwrap();
        let grid = GridFile::open(&result.store_path.parent().unwrap().join("grid.rwg")).unwrap();
        for (name, selector) in [("mslp",selectors[0]),("u_10m",selectors[1]),("v_10m",selectors[2])] {
            let expected = direct.iter().find(|field|field.selector==selector).unwrap();
            let actual = read_field_2d(&reader,&grid,name).unwrap();
            assert_eq!(actual.values.iter().map(|value|value.to_bits()).collect::<Vec<_>>(),expected.values.iter().map(|value|value.to_bits()).collect::<Vec<_>>());
        }
        std::fs::write(output.join("receipt.json"),serde_json::to_vec_pretty(&serde_json::json!({"status":"PASS",
            "store":result.store_path,"subset_fields":3,"member":null,"native_winds_rotated":true,
            "wind_magnitude_preserved":true,"direct_map_parity_bit_exact":true})).unwrap()).unwrap();
    }

    #[test]
    #[ignore = "requires retained public provider fixtures; no network"]
    fn retained_provider_statistics_ingest_preserves_actual_mean_spread_and_rejects_wrong_product() {
        let fixtures = PathBuf::from(std::env::var_os("ARWEN_AIGEFS_FIXTURES").expect("fixture directory"));
        let output = PathBuf::from(std::env::var_os("ARWEN_AIGEFS_PROOF_DIR").expect("proof directory")).join("normalized-statistics");
        std::fs::create_dir_all(&output).unwrap();
        let temperature = FieldSelector::height_agl(CanonicalField::Temperature, 2);
        let selectors = vec![temperature.with_ensemble_mean(), temperature.with_ensemble_standard_deviation()];
        let profile = IngestProfile::selected(selectors.clone());
        let cycle = CycleSpec::new("20260817", 0).unwrap();
        let cancel = AtomicBool::new(false);
        let sink = |_event: IngestEvent| {};
        let cache = output.join("cache");
        let stores = output.join("stores");
        let config = IngestConfig { member: None, model: ModelId::Gefs, cycle: &cycle,
            source_override: Some(SourceId::Aws), cache_root: &cache, use_cache: true,
            store_root: &stores, model_slug: "gefs", run_slug: "mean-and-spread",
            profile: &profile, verify: true, progress: &sink, cancel: &cancel };
        let requests = selected_field_fetch_requests(&config, 0).unwrap();
        let mut expected = Vec::new();
        for (request, selector) in requests.iter().zip(&selectors) {
            let token = request.request.product.split('/').next_back().unwrap();
            let path = fixtures.join(format!("{token}.t00z.pgrb2a.0p50.f000"));
            let bytes = std::fs::read(path).unwrap();
            let extracted = rustwx_io::extract_field_values_partial_from_model_bytes_at_forecast_hour(ModelId::Gefs, &bytes, None, &[*selector], Some(0)).unwrap();
            assert_eq!(extracted.extracted.len(), 1);
            expected.push(extracted.extracted[0].values.clone());
            let url = rustwx_models::resolve_urls(&request.request).unwrap().into_iter()
                .find(|url| url.source == SourceId::Aws).unwrap().grib_url;
            rustwx_io::store_cached_fetch(&cache, request, &rustwx_io::FetchResult { source: SourceId::Aws, url, bytes }).unwrap();
        }
        let fetched = fetch_hour(&config, 0).unwrap();
        assert!(fetched.sfc.cache_hit && fetched.prs.result.bytes.is_empty());
        // Combined bytes must not masquerade as either component cache file.
        assert!(fetched.sfc.bytes_path.as_os_str().is_empty());
        let result = process_fetched_hour(&config, fetched).unwrap();
        let reader = HourReader::open(&result.store_path).unwrap();
        let grid = GridFile::open(&result.store_path.parent().unwrap().join("grid.rwg")).unwrap();
        assert_eq!(result.fields_2d, 2);
        for (selector, values) in selectors.iter().zip(&expected) {
            let stored = read_field_2d(&reader, &grid, &selector.key()).unwrap();
            assert_eq!(stored.selector, *selector);
            assert_eq!(stored.values.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), values.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
            assert!(reader.variable(&selector.key()).unwrap().selector.get("ensemble_member").is_none());
        }
        assert!(expected[0].iter().zip(&expected[1]).any(|(mean, spread)| mean.is_finite() && spread.is_finite() && mean != spread));
        let products = reader.variable(&selectors[0].key()).unwrap().selector["source_products"].clone();
        assert_eq!(products, serde_json::json!(["pgrb2ap5/geavg"]));
        // Actual mean bytes under a requested spread contract must fail before
        // writing a store, even though the meteorological variable exists.
        let wrong_cache = output.join("wrong-cache");
        let wrong_stores = output.join("wrong-stores");
        let wrong_profile = IngestProfile::selected(vec![selectors[1]]);
        let wrong = IngestConfig { profile: &wrong_profile, cache_root: &wrong_cache, store_root: &wrong_stores, ..config };
        let request = selected_field_fetch_requests(&wrong, 0).unwrap().remove(0);
        let bytes = std::fs::read(fixtures.join("geavg.t00z.pgrb2a.0p50.f000")).unwrap();
        rustwx_io::store_cached_fetch(&wrong_cache, &request, &rustwx_io::FetchResult { source: SourceId::Aws, url: "retained-wrong-statistic-control".into(), bytes }).unwrap();
        let err = process_fetched_hour(&wrong, fetch_hour(&wrong, 0).unwrap()).unwrap_err().to_string();
        assert!(err.contains("absent") || err.contains("missing"), "{err}");
        assert!(!wrong_stores.exists());
        let later_cycle = CycleSpec::new("20260817", 6).unwrap();
        let cycle_stores = output.join("wrong-cycle-stores");
        let cycle_config = IngestConfig { cycle: &later_cycle, store_root: &cycle_stores, ..wrong };
        let received = rustwx_io::load_cached_fetch(&cache, &requests[1]).unwrap().unwrap();
        let cycle_error = process_fetched_hour(&cycle_config, FetchedHour {hour:0,prs:empty_pressure_slot(&received),sfc:received,
            prs_fetch_ms:0,sfc_fetch_ms:0}).unwrap_err().to_string();
        assert!(cycle_error.contains("cycle") || cycle_error.contains("reference"), "{cycle_error}");
        assert!(!cycle_stores.exists());
        let mixed_profile = IngestProfile::selected(vec![selectors[0], temperature]);
        let mixed_stores = output.join("wrong-mixed-member-stores");
        let mixed_config = IngestConfig { cycle: &cycle, member:Some(0), profile:&mixed_profile,
            store_root:&mixed_stores, ..cycle_config };
        let mut mixed_bytes = std::fs::read(fixtures.join("geavg.t00z.pgrb2a.0p50.f000")).unwrap();
        mixed_bytes.extend(std::fs::read(fixtures.join("gep01.t00z.pgrb2a.0p50.f000")).unwrap());
        let mixed_error = process_fetched_hour(&mixed_config,FetchedHour {hour:0,prs:empty_pressure_slot(&rustwx_io::load_cached_fetch(&cache, &requests[0]).unwrap().unwrap()),
            sfc:CachedFetchResult {result:rustwx_io::FetchResult {source:SourceId::Aws,
                url:"retained-mean-plus-wrong-member".into(),bytes:mixed_bytes},cache_hit:false,
                bytes_path:PathBuf::new(),metadata_path:PathBuf::new()},prs_fetch_ms:0,sfc_fetch_ms:0}).unwrap_err().to_string();
        assert!(mixed_error.contains("member"), "{mixed_error}");
        assert!(!mixed_stores.exists());
        std::fs::write(output.join("receipt.json"), serde_json::to_vec_pretty(&serde_json::json!({
            "status":"PASS", "store":result.store_path, "selectors":selectors,
            "source_products":products, "fields":2, "wrong_product":err,
            "wrong_cycle":cycle_error,"wrong_mixed_member":mixed_error,
            "values_bit_exact":true, "member":null
        })).unwrap()).unwrap();
    }


    #[test]
    fn individual_member_requests_and_cache_identity_remain_separate() {
        let temperature = FieldSelector::height_agl(CanonicalField::Temperature, 2);
        let profile = IngestProfile::selected(vec![temperature]);
        let cycle = CycleSpec::new("20260817", 0).unwrap();
        let sink = |_event: IngestEvent| {}; let cancel = AtomicBool::new(false);
        for model in [ModelId::Gefs, ModelId::Aigefs] {
            let config = IngestConfig { model, member:Some(0), cycle:&cycle, source_override:Some(SourceId::Aws),
                cache_root:Path::new("unused-cache"), use_cache:true, store_root:Path::new("unused-store"),
                model_slug:model.as_str(), run_slug:"test", profile:&profile, verify:false, progress:&sink,cancel:&cancel };
            let a = selected_field_fetch_requests(&config, 0).unwrap().remove(0);
            let b = selected_field_fetch_requests(&IngestConfig { member:Some(17), ..config }, 0).unwrap().remove(0);
            assert_ne!(a.request.product, b.request.product);
            assert_ne!(rustwx_io::fetch_cache_paths(config.cache_root, &a), rustwx_io::fetch_cache_paths(config.cache_root, &b));
            assert_eq!(rustwx_models::product_member(model, &b.request.product).unwrap().ordinal, 17);
            let metadata = volume_store_metadata(model, CanonicalField::Temperature, Some(17));
            assert_eq!(metadata["ensemble_member"], 17);
        }
    }


    fn fetched_hour_fixture(
        dir: &Path,
        write_cache_copies: bool,
    ) -> (FetchedHour, Vec<u8>, Vec<u8>) {
        let prs_bytes: Vec<u8> = (0u32..40_000).flat_map(|v| v.to_le_bytes()).collect();
        let sfc_bytes: Vec<u8> = prs_bytes.iter().rev().copied().collect();
        let make = |label: &str, bytes: &[u8]| {
            let bytes_path = dir.join(format!("{label}.grib2"));
            if write_cache_copies {
                std::fs::write(&bytes_path, bytes).unwrap();
            }
            CachedFetchResult {
                result: rustwx_io::FetchResult {
                    source: SourceId::Aws,
                    url: format!("https://example.invalid/{label}"),
                    bytes: bytes.to_vec(),
                },
                cache_hit: write_cache_copies,
                bytes_path,
                metadata_path: dir.join(format!("{label}.json")),
            }
        };
        let fetched = FetchedHour {
            hour: 4,
            prs: make("prs", &prs_bytes),
            sfc: make("sfc", &sfc_bytes),
            prs_fetch_ms: 123,
            sfc_fetch_ms: 456,
        };
        (fetched, prs_bytes, sfc_bytes)
    }

    fn assert_round_trip(rehydrated: &FetchedHour, prs_bytes: &[u8], sfc_bytes: &[u8]) {
        assert_eq!(rehydrated.hour, 4);
        assert_eq!(rehydrated.prs_fetch_ms, 123);
        assert_eq!(rehydrated.sfc_fetch_ms, 456);
        assert_eq!(rehydrated.prs.result.bytes, prs_bytes);
        assert_eq!(rehydrated.sfc.result.bytes, sfc_bytes);
        assert_eq!(rehydrated.prs.result.url, "https://example.invalid/prs");
        assert_eq!(rehydrated.sfc.result.url, "https://example.invalid/sfc");
    }

    /// Cache-hit shape: the fetch cache file holds the exact bytes, so the
    /// spill writes nothing and rehydration reads the cache copy back.
    #[test]
    fn spill_reuses_cache_copy_and_rehydrates_identically() {
        let dir = std::env::temp_dir().join("rw_ingest_spill_test_cache_hit");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let spill_dir = dir.join("spill");

        let (fetched, prs_bytes, sfc_bytes) = fetched_hour_fixture(&dir, true);
        let spilled = fetched.spill(&spill_dir).unwrap();
        assert!(
            spilled.prs.temp_spill.is_none() && spilled.sfc.temp_spill.is_none(),
            "a usable cache copy must spill without writing"
        );
        assert!(
            !spill_dir.exists(),
            "the spill dir must not even be created on the cache-copy path"
        );
        let rehydrated = spilled.rehydrate().unwrap();
        assert_round_trip(&rehydrated, &prs_bytes, &sfc_bytes);
        assert!(rehydrated.prs.cache_hit && rehydrated.sfc.cache_hit);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// No-cache shape: no file at `bytes_path`, so the bytes go to a spill
    /// file that is deleted again after rehydration.
    #[test]
    fn spill_writes_temp_file_without_cache_copy_and_cleans_up() {
        let dir = std::env::temp_dir().join("rw_ingest_spill_test_no_cache");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let spill_dir = dir.join("spill");

        let (fetched, prs_bytes, sfc_bytes) = fetched_hour_fixture(&dir, false);
        let spilled = fetched.spill(&spill_dir).unwrap();
        let prs_temp = spilled.prs.temp_spill.clone().expect("prs spill file");
        let sfc_temp = spilled.sfc.temp_spill.clone().expect("sfc spill file");
        assert_eq!(std::fs::read(&prs_temp).unwrap(), prs_bytes);
        assert_eq!(std::fs::read(&sfc_temp).unwrap(), sfc_bytes);
        let rehydrated = spilled.rehydrate().unwrap();
        assert_round_trip(&rehydrated, &prs_bytes, &sfc_bytes);
        assert!(!rehydrated.prs.cache_hit && !rehydrated.sfc.cache_hit);
        assert!(
            !prs_temp.exists() && !sfc_temp.exists(),
            "spill files must be deleted after rehydration"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A cache file whose length disagrees with the fetched bytes must NOT
    /// be trusted as the spill copy.
    #[test]
    fn spill_distrusts_cache_copy_with_wrong_length() {
        let dir = std::env::temp_dir().join("rw_ingest_spill_test_bad_len");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let spill_dir = dir.join("spill");

        let (fetched, prs_bytes, sfc_bytes) = fetched_hour_fixture(&dir, true);
        // Truncate the prs cache copy: its length no longer matches.
        std::fs::write(&fetched.prs.bytes_path, &prs_bytes[..100]).unwrap();
        let spilled = fetched.spill(&spill_dir).unwrap();
        assert!(
            spilled.prs.temp_spill.is_some(),
            "mismatched cache copy must force a spill file"
        );
        assert!(
            spilled.sfc.temp_spill.is_none(),
            "the intact sfc cache copy must still be reused"
        );
        let rehydrated = spilled.rehydrate().unwrap();
        assert_round_trip(&rehydrated, &prs_bytes, &sfc_bytes);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Every stored 2D variable the FULL ingest plan realizes either
    /// resolves to its production plot styling through the viewer resolver
    /// (`rustwx_products::viewer`) or is a known fallback with no production
    /// fill counterpart (barb inputs, compute inputs, contour-only height
    /// planes, contour-only mslp). The fallback set is pinned exactly: a new
    /// plan entry that silently falls back fails this test.
    #[test]
    fn viewer_style_resolver_covers_the_full_ingest_plan() {
        use rustwx_products::viewer::operational_style_for_store_variable;

        let model = ModelId::Hrrr;
        // (name, selector) for every planned 2D variable, mirroring
        // `process_fetched_hour`'s stored selectors exactly.
        let mut planned: Vec<(String, FieldSelector)> = surface_plan()
            .into_iter()
            .map(|(name, selector)| (name.to_string(), selector))
            .collect();
        planned.extend([
            (
                TRAILING_2D_NAMES[0].to_string(),
                FieldSelector::surface(CanonicalField::TotalPrecipitation),
            ),
            (
                TRAILING_2D_NAMES[1].to_string(),
                FieldSelector::height_layer_agl(CanonicalField::UpdraftHelicity, 2000, 5000),
            ),
            (
                TRAILING_2D_NAMES[2].to_string(),
                FieldSelector::height_agl(CanonicalField::WindSpeed, 10),
            ),
        ]);
        for (level, name) in VORTICITY_PLAN {
            planned.push((
                name.to_string(),
                FieldSelector::isobaric(CanonicalField::AbsoluteVorticity, *level),
            ));
        }
        for selector in direct_isobaric_plane_selectors(model) {
            planned.push((selector.key(), selector));
        }

        let mut mapped = Vec::new();
        let mut fallback = Vec::new();
        for (name, selector) in &planned {
            let selector_json = serde_json::to_value(selector).expect("selector json");
            match operational_style_for_store_variable(name, &selector_json, "native", model) {
                Some(style) => {
                    assert!(!style.title.is_empty(), "'{name}' must carry a title");
                    assert!(
                        !style.display_units.is_empty(),
                        "'{name}' must carry display units"
                    );
                    mapped.push(name.clone());
                }
                None => fallback.push(name.clone()),
            }
        }

        // The complete fallback set: u/v barb inputs (surface + isobaric),
        // compute-only inputs, the contour-only ISOBARIC geopotential
        // heights, and mslp (production contours mslp; its plot fills the
        // companion 10 m wind speed, so no production colorbar exists for
        // the plane).  `orography` left this set when the terrain product
        // gave the surface plane a production fill of its own.
        let mut expected_fallback: Vec<String> = vec![
            "u_10m".to_string(),
            "v_10m".to_string(),
            "surface_pressure".to_string(),
            "mslp".to_string(),
        ];
        for selector in direct_isobaric_plane_selectors(model) {
            if matches!(
                selector.field,
                CanonicalField::GeopotentialHeight | CanonicalField::UWind | CanonicalField::VWind
            ) {
                expected_fallback.push(selector.key());
            }
        }
        let mut actual_fallback = fallback.clone();
        actual_fallback.sort();
        expected_fallback.sort();
        assert_eq!(
            actual_fallback, expected_fallback,
            "every fallback variable must be a known no-counterpart input"
        );
        assert_eq!(mapped.len() + fallback.len(), planned.len());
        // Pinned inventory counts (2026-06 full HRRR plan): 42 of the 65
        // planned 2D planes carry production styling; the 22 fallbacks are
        // the exact set asserted above.  (43/22, not 42/23: `orography`
        // moved across when the terrain product was added; 43/25 since
        // `925mb_height_winds` added the 925 hPa height and u/v planes,
        // which are contour/barb inputs like the other isobaric levels.)
        assert_eq!(mapped.len(), 43, "mapped 2D plane count");
        assert_eq!(fallback.len(), 25, "fallback 2D plane count");

        // Derived + heavy store grids resolve through their slug markers.
        for slug in store_derived_recipe_slugs()
            .into_iter()
            .chain(store_heavy_recipe_slugs())
        {
            let marker = serde_json::json!({ "derived": slug });
            // The units this ingest stores each derived grid in.
            let units = rustwx_products::derived::derived_product_units(slug).unwrap_or("units");
            assert!(
                operational_style_for_store_variable(slug, &marker, units, model).is_some(),
                "derived/heavy slug '{slug}' must resolve to production styling"
            );
        }

        eprintln!(
            "viewer coverage: {} mapped 2D planes, {} fallback 2D planes, {} derived, {} heavy",
            mapped.len(),
            fallback.len(),
            store_derived_recipe_slugs().len(),
            store_heavy_recipe_slugs().len(),
        );
    }

    #[test]
    fn planned_store_variables_full_is_the_complete_plan() {
        let plan = planned_store_variables(&IngestProfile::full(), ModelId::Hrrr);
        assert_eq!(
            plan.volumes
                .iter()
                .map(|(name, _)| *name)
                .collect::<Vec<_>>(),
            vec![
                "temperature_iso",
                "dewpoint_iso",
                "u_iso",
                "v_iso",
                "height_iso"
            ],
        );
        assert!(plan.volumes.iter().all(|(_, levels)| *levels == 37));
        // Surface plan + trailing windows + vorticity + direct planes, in
        // stored order with the grid carrier first.
        assert_eq!(plan.fields_2d[0], "temperature_2m");
        assert!(plan.fields_2d.contains(&"apcp_1h".to_string()));
        assert!(
            plan.fields_2d
                .contains(&"absolute_vorticity_500".to_string())
        );
        assert!(
            plan.fields_2d
                .contains(&"geopotential_height_500hpa".to_string()),
            "direct-recipe planes must be planned under their selector keys"
        );
        assert!(plan.derived.contains(&"sbcape"));
        assert!(plan.heavy.contains(&"sbecape"));
    }

    /// APCP accuracy: the GFS full 2D plan must NOT claim `apcp_1h` (nor the
    /// other HRRR-trick trailing fields), because GFS `pgrb2` APCP is a
    /// bucketed accumulation (0-6h resets) the re-select can't accurately turn
    /// into a 1 h increment. HRRR's plan keeps the trailing set unchanged.
    #[test]
    fn gfs_full_plan_excludes_apcp_1h_but_hrrr_keeps_it() {
        let hrrr = planned_store_variables(&IngestProfile::full(), ModelId::Hrrr);
        assert!(
            hrrr.fields_2d.contains(&"apcp_1h".to_string()),
            "HRRR must still carry the trailing 1 h apcp window"
        );
        assert!(hrrr.fields_2d.contains(&"uh_2to5km_max_1h".to_string()));
        assert!(
            hrrr.fields_2d
                .contains(&"wind_speed_10m_max_1h".to_string())
        );

        let gfs = planned_store_variables(&IngestProfile::full(), ModelId::Gfs);
        for trailing in TRAILING_2D_NAMES {
            assert!(
                !gfs.fields_2d.contains(&trailing.to_string()),
                "GFS plan must not claim the trailing 1 h field '{trailing}' (bucketed APCP)"
            );
        }
        // The accurate GFS run total is still planned (the plain accumulation).
        assert!(
            gfs.fields_2d.contains(&"apcp_run_total".to_string()),
            "GFS still stores the plain run-total accumulation"
        );
    }

    #[test]
    fn planned_store_variables_respect_sounding_and_view() {
        let mut profile = IngestProfile::sounding();
        profile.level_step_hpa = 50;
        let sounding = planned_store_variables(&profile, ModelId::Hrrr);
        assert_eq!(sounding.volumes.len(), 5);
        assert!(sounding.volumes.iter().all(|(_, levels)| *levels == 19));
        assert_eq!(sounding.fields_2d.len(), 7);
        assert_eq!(
            sounding.fields_2d[0], "temperature_2m",
            "grid carrier first"
        );
        assert!(!sounding.fields_2d.contains(&"apcp_1h".to_string()));
        assert!(sounding.derived.is_empty() && sounding.heavy.is_empty());

        let view = planned_store_variables(&IngestProfile::view(), ModelId::Hrrr);
        assert!(view.volumes.is_empty());
        assert!(
            view.fields_2d
                .contains(&"absolute_vorticity_500".to_string())
        );
        assert!(!view.derived.is_empty());
        assert!(view.heavy.is_empty());
    }

    #[test]
    fn parse_hours_single() {
        assert_eq!(parse_hours("6").unwrap(), vec![6]);
    }

    #[test]
    fn parse_hours_degenerate_range() {
        assert_eq!(parse_hours("0-0").unwrap(), vec![0]);
    }

    #[test]
    fn parse_hours_list_sorts_ascending() {
        assert_eq!(parse_hours("12,0,6").unwrap(), vec![0, 6, 12]);
    }

    #[test]
    fn parse_hours_range_is_inclusive() {
        assert_eq!(parse_hours("0-6").unwrap(), (0..=6).collect::<Vec<u16>>());
    }

    /// A pre-set cancel flag stops both halves at their first boundary
    /// (before any network or extraction work) with the distinguishable
    /// `Cancelled` error and no events emitted.
    #[test]
    fn cancel_flag_stops_fetch_and_process_before_any_work() {
        use std::sync::Mutex;

        let cycle = CycleSpec::new("20260608", 0).expect("valid cycle");
        let profile = IngestProfile::sounding();
        let cancel = AtomicBool::new(true);
        let events: Mutex<Vec<IngestEvent>> = Mutex::new(Vec::new());
        let sink = |event: IngestEvent| events.lock().unwrap().push(event);
        let config = IngestConfig {
            model: ModelId::Hrrr,
            member: None,
            cycle: &cycle,
            source_override: None,
            cache_root: Path::new("nonexistent-cache"),
            use_cache: false,
            store_root: Path::new("nonexistent-store"),
            model_slug: "hrrr",
            run_slug: "20260608_00z",
            profile: &profile,
            verify: false,
            progress: &sink,
            cancel: &cancel,
        };

        let err = fetch_hour(&config, 6).expect_err("cancelled fetch must error");
        assert!(err.is_cancelled(), "got: {err}");

        let fetched = FetchedHour {
            hour: 6,
            prs: rustwx_io::CachedFetchResult {
                result: rustwx_io::FetchResult {
                    source: SourceId::Aws,
                    url: String::new(),
                    bytes: Vec::new(),
                },
                cache_hit: true,
                bytes_path: PathBuf::new(),
                metadata_path: PathBuf::new(),
            },
            sfc: rustwx_io::CachedFetchResult {
                result: rustwx_io::FetchResult {
                    source: SourceId::Aws,
                    url: String::new(),
                    bytes: Vec::new(),
                },
                cache_hit: true,
                bytes_path: PathBuf::new(),
                metadata_path: PathBuf::new(),
            },
            prs_fetch_ms: 0,
            sfc_fetch_ms: 0,
        };
        let err = process_fetched_hour(&config, fetched).expect_err("cancelled process");
        assert!(err.is_cancelled(), "got: {err}");
        assert!(
            events.lock().unwrap().is_empty(),
            "a pre-set cancel must stop the flow before any stage event"
        );
    }

    /// GFS hour-cadence validation: hourly to f120, then 3-hourly to f384.
    /// f121 (off the 3-hourly grid past 120) is rejected; f123 and f384 are
    /// accepted; f385 (past the horizon) is rejected. The error names the
    /// flag and the offending hour.
    #[test]
    fn validate_forecast_hours_enforces_gfs_cadence() {
        let cycle = 0;
        validate_forecast_hours(ModelId::Gfs, cycle, &[0, 1, 120, 123, 384])
            .expect("on-cadence GFS hours pass");

        let err = validate_forecast_hours(ModelId::Gfs, cycle, &[121])
            .expect_err("f121 is off the GFS cadence");
        let message = err.to_string();
        assert!(message.contains("--hours"), "must name the flag: {message}");
        assert!(message.contains("f121"), "must name the hour: {message}");

        validate_forecast_hours(ModelId::Gfs, cycle, &[123]).expect("f123 is on the 3-hourly grid");
        validate_forecast_hours(ModelId::Gfs, cycle, &[384]).expect("f384 is the horizon");
        assert!(
            validate_forecast_hours(ModelId::Gfs, cycle, &[385]).is_err(),
            "f385 is past the GFS horizon"
        );
    }

    #[test]
    fn validate_forecast_hours_uses_model_specific_cadence_note() {
        let err = validate_forecast_hours(ModelId::Aigefs, 0, &[1])
            .expect_err("AI-GEFS is 6-hourly, so f001 is invalid");
        let message = err.to_string();
        assert!(message.contains("AI global feeds"), "got: {message}");
        assert!(
            !message.contains("GFS is hourly"),
            "AI-GEFS errors must not carry the GFS cadence note: {message}"
        );
    }

    /// The cycle must sit on the model's published cycle table: GFS runs only
    /// at 00/06/12/18z, so `--cycle 3` is rejected (with a message listing the
    /// valid cycles) BEFORE it can build a dead `gfs.t03z` URL; 06z passes.
    /// HRRR's all-24-hour table accepts every cycle (e.g. 17z).
    #[test]
    fn validate_forecast_hours_enforces_the_cycle_table() {
        let err = validate_forecast_hours(ModelId::Gfs, 3, &[0])
            .expect_err("03z is not a published GFS cycle");
        let message = err.to_string();
        assert!(message.contains("--cycle"), "must name the flag: {message}");
        assert!(
            message.contains("03z"),
            "must name the bad cycle: {message}"
        );
        for cycle in ["00z", "06z", "12z", "18z"] {
            assert!(
                message.contains(cycle),
                "must list valid cycle {cycle}: {message}"
            );
        }

        validate_forecast_hours(ModelId::Gfs, 6, &[0, 3]).expect("06z is a published GFS cycle");
        validate_forecast_hours(ModelId::Hrrr, 17, &[0, 3])
            .expect("HRRR runs hourly, so 17z is valid");
    }

    /// RRFS-A cadence comes straight from the registry and is domain-neutral
    /// (it does not depend on the unsettled CONUS-vs-NA fetch-plan question:
    /// see `docs/internal/plans/2026-06-11-rrfs-support.md`). Cycles run
    /// HOURLY 00z-23z and the forecast horizon is f000..=f060, so every cycle
    /// is valid, f060 passes, and f061 is rejected with the flag named.
    #[test]
    fn validate_forecast_hours_enforces_rrfs_a_horizon() {
        // Hourly cycle table: an off-synoptic cycle like 13z is valid.
        validate_forecast_hours(ModelId::RrfsA, 13, &[0, 1, 30, 60])
            .expect("RRFS-A runs hourly to f060 on every cycle");
        // f060 is the horizon; f061 is past it.
        validate_forecast_hours(ModelId::RrfsA, 0, &[60]).expect("f060 is the RRFS-A horizon");
        let err = validate_forecast_hours(ModelId::RrfsA, 0, &[61])
            .expect_err("f061 is past the RRFS-A horizon");
        let message = err.to_string();
        assert!(message.contains("--hours"), "must name the flag: {message}");
        assert!(message.contains("f061"), "must name the hour: {message}");
    }

    /// HRRR cadence is unchanged: f018 is valid on every cycle, the 6-hourly
    /// cycles reach f048, and f019 is rejected on an off-synoptic cycle.
    #[test]
    fn validate_forecast_hours_leaves_hrrr_unchanged() {
        validate_forecast_hours(ModelId::Hrrr, 0, &[0, 18, 48]).expect("00z HRRR reaches f048");
        validate_forecast_hours(ModelId::Hrrr, 1, &[0, 18]).expect("01z HRRR reaches f018");
        assert!(
            validate_forecast_hours(ModelId::Hrrr, 1, &[19]).is_err(),
            "01z HRRR stops at f018"
        );
    }

    /// `single_file_model` in IngestedHour is driven by the fetch plan: GFS
    /// has a one-entry plan (single file), HRRR has two. Verify the flag
    /// derivation used inside process_fetched_hour matches the canonical plan.
    #[test]
    fn single_file_flag_matches_fetch_plan_entry_count() {
        let gfs_is_single = fetch_plan(ModelId::Gfs)
            .map(|plan| plan.len() == 1)
            .unwrap_or(false);
        assert!(
            gfs_is_single,
            "GFS fetch plan has one entry → single_file_model"
        );

        let hrrr_is_single = fetch_plan(ModelId::Hrrr)
            .map(|plan| plan.len() == 1)
            .unwrap_or(false);
        assert!(
            !hrrr_is_single,
            "HRRR fetch plan has two entries → not single_file"
        );
    }

    #[test]
    fn parse_hours_rejects_junk_naming_the_flag() {
        for spec in ["1x", "6-2", "", ","] {
            let err = parse_hours(spec).unwrap_err();
            let message = err.to_string();
            assert!(
                message.contains("--hours"),
                "error for '{spec}' must name the flag, got: {message}"
            );
        }
        let message = parse_hours("1x").unwrap_err().to_string();
        assert!(
            message.contains("invalid token '1x'"),
            "error must name the offending token, got: {message}"
        );
    }
}
