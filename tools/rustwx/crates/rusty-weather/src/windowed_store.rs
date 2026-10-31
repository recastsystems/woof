#![allow(dead_code)]

//! Windowed products (multi-hour accumulations and extrema) computed FROM
//! THE STORE across per-hour `.rws` files, mirroring the GRIB windowed
//! lane's semantics (`rustwx_products::windowed` + `windowed_decoder`)
//! product for product:
//!
//! * QPF: `qpf_1h` and `qpf_total` read the trailing 1 h / run-total APCP
//!   accumulations the ingest stored from the anchor hour's sfc file
//!   (`apcp_1h`, `apcp_run_total`): the GRIB lane's "direct" strategy. The
//!   fixed trailing windows (`qpf_6h`/`12h`/`24h`) sum stored hourly
//!   `apcp_1h` increments, exactly the GRIB lane's HRRR path (HRRR never
//!   carries 6/12/24 h APCP messages, so that lane always summed hourly
//!   increments too). Millimeters fold first, inches out: the GRIB lane's
//!   conversion order. A store written by the wrfout import lane carries
//!   no hourly increment at all, only the run accumulation since
//!   simulation start, so the hourly windows fall back to differencing
//!   two stored run totals, naming both hours in the strategy note.
//! * 2-5 km UH: pointwise maxima of the stored sub-hourly 1 h max planes
//!   (`uh_2to5km_max_1h`, the native MXUPHL message selected at its window
//!   start hour), the exact field the GRIB windowed lane reduced. Hours
//!   ingested before the max field existed fall back to the stored hourly
//!   `uh_2to5km` plane, with the fallback hours named in the strategy
//!   note. (In current HRRR sfc files that plane is itself the MXUPHL
//!   message: the file carries no instantaneous UPHL, so plain selection
//!   matched MXUPHL by its end-hour score, but the note stays
//!   conservative: a store written from a file that DOES carry
//!   instantaneous UPHL holds top-of-hour snapshots, a lower bound on the
//!   sub-hourly max.)
//! * 10 m wind: pointwise maxima of `wind_speed_10m_max_1h` (the native
//!   sub-hourly `WIND:10 m above ground` max field the GRIB lane
//!   consumed); m/s folds first, knots out. Hours without it read WRF
//!   WSPD10MAX as the wrfout import stores it (`wrf_wspd10max`), the
//!   exact max over the history interval ending at its frame, like
//!   UP_HELI_MAX ([`HISTORY_INTERVAL_MAX_ROWS`]). Hours with neither fall
//!   back to top-of-hour hypot(`u_10m`, `v_10m`) speeds: a genuine lower
//!   bound on the sub-hourly max (the sfc file carries no instantaneous
//!   wind-speed message), named in the strategy note.
//! * 2 m temp/RH/dewpoint/VPD: pointwise max/min/range over the fixed
//!   F001-F024 / F025-F048 / F001-F048 snapshot windows. Temperature and
//!   dewpoint convert K -> degC per hour before the fold and RH clamps to
//!   0..100, mirroring `surface_snapshot_values_for_hour`; VPD reads the
//!   ingest-computed `vpd_2m` derived grid (hPa) instead of recomputing
//!   from temp + RH.
//!
//! Gap handling mirrors the GRIB lane's blocker pattern exactly: a window
//! realizes only when EVERY contributing hour is present, in the store
//! AND carrying the source variable(s) in the expected units. A missing
//! middle hour blocks the product with a reason naming the gap; it is
//! never silently skipped. Window minimums (e.g. 24 h products need F024)
//! reuse the lane's planning blockers verbatim, with the anchor hour = the
//! run's max stored hour.
//!
//! Exact-time stores.  A history written more often than hourly puts the
//! run on the exact-time axis: slots are ordinals and each frame carries
//! its own lead in seconds ([`rw_store::RwsExactTime`]).  Windows are still
//! the whole-hour windows above, served from those leads:
//!
//! * a window (t-W, t] ends only on a frame whose lead is a whole hour,
//!   and needs the stored frames at BOTH of its bounding whole-hour
//!   leads; a frame between hours closes no window and says so;
//! * accumulations and snapshot windows read the frames at their whole-
//!   hour leads, exactly as on the whole-hour axis (the run totals are
//!   cumulative, so the frames between add nothing);
//! * the maxima (UH, 10 m wind) fold EVERY stored frame inside the
//!   window.  WRF's UP_HELI_MAX and WSPD10MAX are reset at each history
//!   write (`gpuwm/core/uh_diag.py`), so on a 15-minute history the
//!   whole-hour frame alone holds only the last quarter hour; a plane read
//!   from an instant (the wrfout 10 m wind, from U10 and V10) is one of the
//!   window's instants and stays a labelled lower bound.  Either way the
//!   frames inside a window must be evenly spaced from its start and the
//!   frame at its start must be stored (unless the window starts with
//!   the run), since a frame that was never stored cannot be folded and
//!   the fold without it reads low ([`WindowGaps`]).  An instant fold
//!   also needs a stored frame at every whole hour of the window, as the
//!   whole-hour axis does.
//!
//! Memory: accumulations stream hour by hour, each hour file is opened
//! once, each needed source plane is read once (`read_full_2d`, ~3.6 ms)
//! and folded into every per-product accumulator that wants it; no
//! per-hour plane outlives its hour iteration.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use rustwx_products::windowed::HrrrWindowedProduct;
use rw_store::error::RwStoreError;
use rw_store::grid::GridFile;
use rw_store::ingest::read_grid_2d;
use rw_store::reader::HourReader;
use rw_store::run::{RwsRunManifest, validate_store_component};

pub(crate) const MM_PER_INCH: f64 = 25.4;
pub(crate) const MS_TO_KT: f64 = 1.943_844_5;

/// Stored APCP variable names. Spelled once so the plan-time probe and
/// the per-hour read agree on what "this store carries it" means: a
/// probe that tested a different spelling than the read would hand a
/// plan to a reader that cannot serve it.
const APCP_1H_VAR: &str = "apcp_1h";
/// The GRIB ingest lane's run accumulation.
const APCP_RUN_TOTAL_VAR: &str = "apcp_run_total";
/// The wrfout import lane's name for the same physical plane (RAINC +
/// RAINNC [+ RAINSH] since simulation start, kg/m^2): WRF's Registry
/// spelling, stored verbatim.
const APCP_WRFOUT_RUN_TOTAL_VAR: &str = "apcp";

/// One realized windowed product grid: display values (already in display
/// units) on the full run grid, plus the metadata the windowed render path
/// stamps into subtitles and reports.
#[derive(Debug, Clone)]
pub struct WindowedGrid {
    pub slug: String,
    pub units: String,
    pub title: String,
    pub values: Vec<f64>,
    pub hours_used: Vec<u16>,
    pub window_hours: Option<u16>,
    pub strategy: String,
}

/// Outcome of one windowed compute pass: realized grids in request order,
/// blocked products as `(slug, reason)` (window minimum not met, an hour
/// missing from the store, a source variable missing from an hour file, or
/// unexpected stored units), and the anchor hour trailing windows ended at.
#[derive(Debug)]
pub struct WindowedStoreOutcome {
    pub grids: Vec<WindowedGrid>,
    pub blockers: Vec<(String, String)>,
    pub anchor_hour: u16,
}

/// Forecast hours registered in the run's `run.json` manifest, ascending.
pub fn stored_run_hours(
    store_root: &Path,
    model_slug: &str,
    run_slug: &str,
) -> Result<Vec<u16>, Box<dyn std::error::Error>> {
    let (_, manifest) = load_run_manifest(store_root, model_slug, run_slug)?;
    Ok(manifest.hours.keys().copied().collect())
}

/// Can this run's stored axis serve the fixed-hour windowed lane at all?
/// True exactly when more than one frame is stored: a window needs a frame
/// at each of its two bounds.  Both axes qualify -- an exact-time run's
/// windows are served from its frames' leads ([`stored_window_frames`]).
/// Model identity is deliberately NOT part of this answer -- per-plane
/// availability is checked against the store when the windows compute,
/// so a WRF (or any) run with the needed stored planes participates.
pub fn windowed_axis_ready(
    store_root: &Path,
    model_slug: &str,
    run_slug: &str,
) -> Result<bool, Box<dyn std::error::Error>> {
    let (_, manifest) = load_run_manifest(store_root, model_slug, run_slug)?;
    Ok(manifest.hours.len() > 1)
}

const SECONDS_PER_HOUR: u64 = 3_600;

/// One stored frame as the windowed lane sees it: its storage slot and its
/// lead from the run's start.  On the whole-hour axis the slot IS the
/// forecast hour; on the exact-time axis it is an ordinal and the lead
/// comes from the frame's own [`rw_store::RwsExactTime`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WindowFrame {
    pub slot: u16,
    pub lead_seconds: u64,
}

impl WindowFrame {
    /// Whether a window can end on this frame: its lead is a whole hour.
    pub fn closes_windows(&self) -> bool {
        self.lead_seconds % SECONDS_PER_HOUR == 0
    }

    /// The whole forecast hours this frame's lead covers.
    pub fn whole_hours(&self) -> u64 {
        self.lead_seconds / SECONDS_PER_HOUR
    }

    /// `F001` on a whole-hour lead, `+000:15` between hours.
    pub fn label(&self) -> String {
        if self.closes_windows() {
            format!("F{:03}", self.whole_hours())
        } else {
            lead_label(self.lead_seconds)
        }
    }
}

/// `+HHH:MM`, with `:SS` only when the seconds are not zero.
fn lead_label(lead_seconds: u64) -> String {
    let hours = lead_seconds / SECONDS_PER_HOUR;
    let minutes = (lead_seconds % SECONDS_PER_HOUR) / 60;
    let seconds = lead_seconds % 60;
    if seconds == 0 {
        format!("+{hours:03}:{minutes:02}")
    } else {
        format!("+{hours:03}:{minutes:02}:{seconds:02}")
    }
}

/// Every stored frame of the run with its lead, in slot order (which is
/// lead order on both axes).
pub fn stored_window_frames(
    store_root: &Path,
    model_slug: &str,
    run_slug: &str,
) -> Result<Vec<WindowFrame>, Box<dyn std::error::Error>> {
    let (_, manifest) = load_run_manifest(store_root, model_slug, run_slug)?;
    manifest_window_frames(&manifest).map_err(Into::into)
}

fn manifest_window_frames(manifest: &RwsRunManifest) -> Result<Vec<WindowFrame>, String> {
    manifest
        .hours
        .iter()
        .map(|(&slot, entry)| {
            let lead_seconds = if manifest.is_exact_time_axis() {
                entry
                    .exact_time()
                    .ok_or_else(|| {
                        format!("exact-time slot {slot} carries no lead in the run manifest")
                    })?
                    .lead_seconds
            } else {
                u64::from(slot) * SECONDS_PER_HOUR
            };
            Ok(WindowFrame { slot, lead_seconds })
        })
        .collect()
}

/// The forecast hour the run's stored frames reach: the last stored hour on
/// the whole-hour axis, the whole hours of the last lead on the exact-time
/// axis.  What [`window_fits_run`] is asked against; `None` for an empty run.
pub fn last_window_hour(frames: &[WindowFrame]) -> Option<u16> {
    frames
        .iter()
        .map(|frame| u16::try_from(frame.whole_hours()).unwrap_or(u16::MAX))
        .max()
}

/// Can ANY frame of a run whose stored frames end at `last_hour` close
/// `product`'s window?
///
/// Asked of [`plan_product`] itself, the planner every windowed render of
/// a store goes through, so the answer cannot drift from what a render
/// would do.  The planner's refusals depend only on the anchor hour, never
/// on which APCP plane the store carries, so the plane is not consulted.
///
/// WHAT BREAKAGE THIS PREVENTS (gate law): an 18 h run asked for `all`
/// requested 41 windowed families whose windows end at F024 or F048,
/// every one of them skipped on every one of its 19 frames -- 805 skip
/// lines for pictures the run could never contain.  A catalog keyword
/// expands to what the run can draw; a window longer than the run is not
/// in that set.
pub fn window_fits_run(product: HrrrWindowedProduct, last_hour: u16) -> bool {
    plan_product(product, last_hour, QpfSource::NativeHourly).is_ok()
}

/// The first forecast hour at which `product`'s window closes, or `None`
/// when no hour up to [`WINDOW_SEARCH_LIMIT_HOURS`] does.  What a plan
/// review compares a run's length against before a single frame exists.
pub fn minimum_window_hour(product: HrrrWindowedProduct) -> Option<u16> {
    (0..=WINDOW_SEARCH_LIMIT_HOURS).find(|&hour| window_fits_run(product, hour))
}

/// How far [`minimum_window_hour`] looks.  The longest window in the
/// catalog closes at F048; this is headroom, not a policy.
pub const WINDOW_SEARCH_LIMIT_HOURS: u16 = 240;

/// A blocker reason that also says when the RUN, not just this frame, is
/// too short for the window.
///
/// The planner's own sentence ("0-24 h 10 m wind max requires forecast
/// hour >= 24") is true of the frame it was asked about.  On a run whose
/// last stored frame is F018 it is true of every frame, and a reader
/// needs to be told that no frame of this run will draw the product --
/// which is a fact about the run's length and is said in those words.
fn beyond_run_reason(
    product: HrrrWindowedProduct,
    reason: String,
    last_hour: Option<u16>,
) -> String {
    let Some(last) = last_hour else {
        return reason;
    };
    if window_fits_run(product, last) {
        return reason;
    }
    format!(
        "{reason}; this run's stored frames end at F{last:03}, so none of them closes this window"
    )
}

fn load_run_manifest(
    store_root: &Path,
    model_slug: &str,
    run_slug: &str,
) -> Result<(PathBuf, RwsRunManifest), RwStoreError> {
    validate_store_component("model", model_slug)?;
    validate_store_component("run", run_slug)?;
    let root = std::fs::canonicalize(store_root).map_err(|err| {
        RwStoreError::Meta(format!(
            "cannot resolve store root {}: {err}",
            store_root.display()
        ))
    })?;
    let requested = store_root.join(model_slug).join(run_slug);
    let run_dir = std::fs::canonicalize(&requested).map_err(|err| {
        RwStoreError::Meta(format!(
            "cannot resolve run directory {}: {err}",
            requested.display()
        ))
    })?;
    if !run_dir.starts_with(&root) {
        return Err(RwStoreError::Meta(format!(
            "run directory {} resolves outside store root {}",
            requested.display(),
            root.display()
        )));
    }
    let manifest_path =
        canonical_contained_path(&run_dir, &run_dir.join("run.json"), "run manifest")?;
    let manifest = RwsRunManifest::load_for_run(&manifest_path, model_slug, run_slug)?;
    Ok((run_dir, manifest))
}

fn canonical_contained_path(
    run_dir: &Path,
    path: &Path,
    label: &str,
) -> Result<PathBuf, RwStoreError> {
    let canonical = std::fs::canonicalize(path).map_err(|err| {
        RwStoreError::Meta(format!("cannot resolve {label} {}: {err}", path.display()))
    })?;
    if !canonical.starts_with(run_dir) {
        return Err(RwStoreError::Meta(format!(
            "{label} {} resolves outside run directory {}",
            path.display(),
            run_dir.display()
        )));
    }
    Ok(canonical)
}

/// Compute the requested windowed products from the stored hour files of
/// `<store_root>/<model_slug>/<run_slug>/`, anchored at the max slot in
/// `available_hours`. Unknown slugs are an error (the caller validates
/// requests against `HrrrWindowedProduct::supported_products()`); windows
/// that do not fit the available frames come back as blockers, never as
/// silently shortened windows.  On an exact-time store the slots are
/// ordinals and the windows are served from each frame's lead (see the
/// module notes); the outcome's `anchor_hour` is then the anchor's whole
/// lead hour, not its slot.
pub fn compute_windowed_products(
    store_root: &Path,
    model_slug: &str,
    run_slug: &str,
    available_hours: &[u16],
    requested: &[String],
) -> Result<WindowedStoreOutcome, Box<dyn std::error::Error>> {
    let (run_dir, manifest) = load_run_manifest(store_root, model_slug, run_slug)?;
    let available: BTreeSet<u16> = available_hours.iter().copied().collect();
    let Some(&anchor_slot) = available.iter().next_back() else {
        return Err("windowed compute needs at least one stored hour".into());
    };
    if let Some(hour) = available
        .iter()
        .find(|&&hour| !manifest.hours.contains_key(&hour))
    {
        return Err(format!(
            "available hour F{hour:03} is not registered in {model_slug}/{run_slug}/run.json"
        )
        .into());
    }
    let grid_path = canonical_contained_path(&run_dir, &run_dir.join("grid.rwg"), "grid file")?;
    let grid =
        GridFile::open(&grid_path).map_err(|err| format!("open {}: {err}", grid_path.display()))?;
    manifest.validate_grid(&grid.hash, grid.nx, grid.ny)?;

    let exact_axis = manifest.is_exact_time_axis();
    let run_frames = manifest_window_frames(&manifest)?;
    let last_hour = last_window_hour(&run_frames);
    let cadence = history_cadence(&run_frames);
    // The frames this pass may read, in slot (= lead) order; the anchor is
    // the last of them.
    let frames: Vec<WindowFrame> = run_frames
        .into_iter()
        .filter(|frame| available.contains(&frame.slot))
        .collect();
    let anchor = frames
        .last()
        .copied()
        .ok_or("windowed compute needs at least one stored hour")?;
    let anchor_hour = u16::try_from(anchor.whole_hours()).map_err(|_| {
        format!(
            "lead {} s of the window anchor exceeds the forecast-hour range",
            anchor.lead_seconds
        )
    })?;
    let name_of = |slot: u16| -> FrameName {
        let frame = frames
            .iter()
            .find(|frame| frame.slot == slot)
            .copied()
            .unwrap_or(WindowFrame {
                slot,
                lead_seconds: u64::from(slot) * SECONDS_PER_HOUR,
            });
        FrameName::of(frame, exact_axis)
    };

    // Plan: dedupe slugs (mirroring the GRIB lane), block products whose
    // window minimum exceeds the anchor or whose window has store gaps.
    let mut blockers: Vec<(String, String)> = Vec::new();
    let mut accums: Vec<Accum> = Vec::new();
    let mut seen = BTreeSet::new();
    // Which APCP plane the hourly QPF windows reduce is a property of
    // the store, not of the request, so it is probed at most once per
    // pass and shared by every QPF window (memoized rather than
    // unconditional: a request that touches no QPF product must not pay
    // an anchor-hour file open).
    let mut qpf_source: Option<QpfSource> = None;
    for slug in requested {
        if !seen.insert(slug.as_str()) {
            continue;
        }
        let product = HrrrWindowedProduct::from_slug(slug)
            .ok_or_else(|| format!("'{slug}' is not a windowed product slug"))?;
        if !anchor.closes_windows() {
            // Only an exact-time frame can sit between hours.  Every window
            // of the catalog is a whole number of hours ending on a whole
            // hour, so this frame ends none of them.
            blockers.push((
                slug.clone(),
                format!(
                    "windows close at whole forecast hours, and this frame is at {} \
                     (between F{:03} and F{:03})",
                    anchor.label(),
                    anchor_hour,
                    u32::from(anchor_hour) + 1
                ),
            ));
            continue;
        }
        let source = if reduces_hourly_apcp(product) {
            *qpf_source.get_or_insert_with(|| probe_qpf_source(&run_dir, &manifest, anchor_slot))
        } else {
            // Not consulted by plans that read no hourly APCP.
            QpfSource::NativeHourly
        };
        let mut spec = match plan_product(product, anchor_hour, source) {
            Ok(spec) => spec,
            Err(reason) => {
                blockers.push((slug.clone(), beyond_run_reason(product, reason, last_hour)));
                continue;
            }
        };
        let footprint_swath = matches!(product, HrrrWindowedProduct::Wind10mRunMax | HrrrWindowedProduct::QpfTotal)
            && manifest.hours.get(&anchor_slot).and_then(|entry| HourReader::open(&run_dir.join(&entry.file)).ok())
                .is_some_and(|reader| reader.variable("nest_sample_coverage").is_some());
        if footprint_swath && product == HrrrWindowedProduct::QpfTotal {
            // Nest totals include only integrated exposure. Keep departed
            // ground, including return visits, without borrowing parent rain.
            spec.reduce = Reduce::Max;
            spec.hours = (1..=anchor_hour).collect();
        }
        if let Some(reason) = off_cadence_reason(&spec, cadence, exact_axis) {
            blockers.push((slug.clone(), reason));
            continue;
        }
        let slots = if exact_axis {
            exact_window_slots(&spec, &frames)
        } else {
            whole_hour_window_slots(&spec, &available).map(|slots| (slots, WindowGaps::default()))
        };
        match slots {
            Ok((slots, gaps)) => {
                // A 1 h interval maximum reads one plane on the whole-hour
                // axis and every frame of its hour on the exact-time axis.
                if slots.len() > 1 && spec.reduce == Reduce::Direct {
                    spec.reduce = Reduce::Max;
                }
                let mut accum = Accum::new(spec, slots).with_gaps(gaps);
                accum.footprint_swath = footprint_swath;
                accums.push(accum);
            }
            Err(reason) => blockers.push((slug.clone(), reason)),
        }
    }

    // Which source planes each frame must serve, across live products.
    let mut slots_needed: BTreeMap<u16, BTreeSet<SourceKind>> = BTreeMap::new();
    for accum in &accums {
        for &slot in &accum.slots {
            slots_needed
                .entry(slot)
                .or_default()
                .insert(accum.spec.source);
        }
    }

    // Stream: one HourReader per frame, one read per (frame, source plane),
    // folded into every accumulator that wants it. Ascending slot order is
    // the BTreeMap iteration order, which is lead order on both axes and
    // mirrors the GRIB lane's hour order.
    for (&slot, kinds) in &slots_needed {
        let needs = |accum: &Accum, kind: SourceKind| {
            accum.failed.is_none() && accum.spec.source == kind && accum.slots.contains(&slot)
        };
        if !accums
            .iter()
            .any(|accum| kinds.iter().any(|&kind| needs(accum, kind)))
        {
            continue;
        }
        let name = name_of(slot);
        let fail_all = |accums: &mut Vec<Accum>, reason: String| {
            for accum in accums.iter_mut() {
                if accum.failed.is_none() && accum.slots.contains(&slot) {
                    accum.failed = Some(reason.clone());
                }
            }
        };
        let entry = match manifest.hours.get(&slot) {
            Some(entry) => entry,
            None => {
                fail_all(
                    &mut accums,
                    format!(
                        "{} is not registered in {model_slug}/{run_slug}/run.json",
                        name.noun
                    ),
                );
                continue;
            }
        };
        let hour_path = match canonical_contained_path(
            &run_dir,
            &run_dir.join(&entry.file),
            &format!("{} file", name.noun),
        ) {
            Ok(path) => path,
            Err(err) => {
                fail_all(&mut accums, err.to_string());
                continue;
            }
        };
        let reader = match HourReader::open(&hour_path) {
            Ok(reader) => reader,
            Err(err) => {
                fail_all(&mut accums, format!("open {}: {err}", hour_path.display()));
                continue;
            }
        };
        let meta = reader.meta();
        let metadata_result = if exact_axis {
            // Slot, identity, grid AND the frame's exact lead and valid
            // time, which is what every window on this axis is built on.
            manifest.validate_hour_meta(slot, meta).map(|_| ())
        } else {
            manifest
                .validate_identity(&meta.model, &meta.run)
                .and_then(|()| manifest.validate_grid(&meta.grid_hash, meta.nx, meta.ny))
                .and_then(|()| {
                    if meta.forecast_hour == slot {
                        Ok(())
                    } else {
                        Err(RwStoreError::Meta(format!(
                            "manifest hour F{slot:03} resolves to {}, whose metadata says F{:03}",
                            hour_path.display(),
                            meta.forecast_hour
                        )))
                    }
                })
        };
        if let Err(err) = metadata_result {
            fail_all(&mut accums, err.to_string());
            continue;
        }
        let between_hours = !name.frame.closes_windows();
        for &kind in kinds {
            if !accums.iter().any(|accum| needs(accum, kind)) {
                continue;
            }
            match read_source_plane(&reader, &grid, kind, &name) {
                Ok(plane) => {
                    for accum in accums.iter_mut() {
                        if !needs(accum, kind) {
                            continue;
                        }
                        if between_hours && plane.fidelity == PlaneFidelity::Exact {
                            // A native trailing 1 h max plane covers the hour
                            // ENDING at its own frame; stored between hours it
                            // reaches back past the window's start.  Its
                            // window is read at the whole-hour frames alone.
                            continue;
                        }
                        if accum.footprint_swath {
                            match read_grid_2d(&reader, &grid, "nest_sample_coverage") {
                                Ok(mask) if mask.units == "1" => accum.fold_footprint(&plane.values, &mask.values),
                                _ => {accum.failed = Some("moving-nest swath lacks its explicit footprint mask".into()); continue;}
                            }
                        } else {
                            accum.fold(&plane.values);
                        }
                        match plane.fidelity {
                            PlaneFidelity::Exact => {
                                accum.exact_planes += 1;
                            }
                            PlaneFidelity::InstantaneousLowerBound => {
                                accum.fallback_frames.push(name.at.clone());
                            }
                            PlaneFidelity::HistoryIntervalMax(row) => {
                                accum.interval_max_frames.push(name.at.clone());
                                accum.interval_max_field = Some(row.wrf_name);
                            }
                        }
                    }
                }
                Err(reason) => {
                    for accum in accums.iter_mut() {
                        if needs(accum, kind) {
                            accum.failed = Some(reason.clone());
                        }
                    }
                }
            }
        }
    }

    let mut grids = Vec::with_capacity(accums.len());
    for accum in accums {
        let slug = accum.spec.product.slug().to_string();
        match accum.finish(exact_axis) {
            Ok(grid) => grids.push(grid),
            Err(reason) => blockers.push((slug, reason)),
        }
    }
    Ok(WindowedStoreOutcome {
        grids,
        blockers,
        anchor_hour,
    })
}

/// How a message names one stored frame: `hour F001` on the whole-hour
/// axis (the wording every reason used before exact-time windows), and
/// `frame +000:15 (slot 1)` on the exact-time axis.  `at` is the short
/// form a strategy note lists.
struct FrameName {
    frame: WindowFrame,
    noun: String,
    at: String,
}

impl FrameName {
    fn of(frame: WindowFrame, exact_axis: bool) -> Self {
        if exact_axis {
            let at = lead_label(frame.lead_seconds);
            Self {
                frame,
                noun: format!("frame {at} (slot {})", frame.slot),
                at,
            }
        } else {
            Self {
                frame,
                noun: format!("hour F{:03}", frame.slot),
                at: format!("F{:03}", frame.slot),
            }
        }
    }
}

/// The maxima folded from every stored frame inside a window on the
/// exact-time axis: a wrfout UP_HELI_MAX or WSPD10MAX plane holds the max
/// over the interval since the history write before it, and a 10 m wind
/// read from U10 and V10 is one instant of the window.
fn folds_interval_maxima(source: SourceKind) -> bool {
    matches!(source, SourceKind::Uh2to5km | SourceKind::WindSpeed10m)
}

/// A whole-hour store's frames for one window: its hours, every one of
/// them stored.
fn whole_hour_window_slots(
    spec: &ProductSpec,
    available: &BTreeSet<u16>,
) -> Result<Vec<u16>, String> {
    let missing: Vec<u16> = spec
        .hours
        .iter()
        .copied()
        .filter(|hour| !available.contains(hour))
        .collect();
    if missing.is_empty() {
        Ok(spec.hours.clone())
    } else {
        Err(missing_frames_reason(spec, &missing, "hour(s)"))
    }
}

/// What a fold over an exact-time window's frames would lack, found when
/// the window is planned and applied by [`Accum::finish`] once it knows
/// what the folded planes measure ([`PlaneFidelity`]).
///
/// WHAT BREAKAGE THIS PREVENTS (gate law): a window folded from part of
/// its frames reads low, and a picture drawn so is kept.  The live pass
/// of a sub-hourly grid draws each whole hour beside that hour's frames
/// only; the 10 m wind run maximum at F002 of a 15-minute grid was then
/// folded from +1:00 to +2:00 alone, and a named request that drew it
/// there never drew it again (1,439 of 5,624 cells low, by up to
/// 2.72 kt, against a render of the same run's whole series).  Being
/// labelled a lower bound does not license a fold of fewer frames than
/// the run stored.  Only a native trailing 1 h max plane, read at the
/// whole hours alone, asks nothing of the frames between them.
#[derive(Debug, Clone, Default)]
struct WindowGaps {
    /// The frame at the window's start is not stored (a window starting
    /// with the run needs none), or the frames inside it are unevenly
    /// spaced: a frame inside the window is missing.  Refused for every
    /// plane folded from the frames between the hours, per-history-
    /// interval maxima and instants alike.
    intervals: Option<String>,
    /// One of the window's whole hours has no stored frame.  Refused for
    /// a native trailing 1 h max plane, read at each whole hour, and for
    /// an instant, which the whole-hour axis refuses the same gap for;
    /// per-history-interval maxima evenly spaced from the window's start
    /// cover it without a frame on each hour.
    hours: Option<String>,
}

/// An exact-time store's frames for one window (t-W, t], by lead, and
/// what a fold over them would lack.
///
/// Accumulations and snapshots read the frames at the window's whole
/// hours, as on the whole-hour axis, and every one of them is required.
/// The interval maxima read EVERY stored frame inside the window, which
/// must end on a stored frame.  Whether the window also needs the frame
/// at its start, and frames evenly spaced from it, depends on what the
/// stored planes measure, so those gaps are returned rather than refused
/// here ([`WindowGaps`]).  A window starting with the run needs no frame
/// at its start: no interval begins before the run does.
fn exact_window_slots(
    spec: &ProductSpec,
    frames: &[WindowFrame],
) -> Result<(Vec<u16>, WindowGaps), String> {
    let slot_at = |hour: u16| {
        frames
            .iter()
            .find(|frame| frame.lead_seconds == u64::from(hour) * SECONDS_PER_HOUR)
            .map(|frame| frame.slot)
    };
    let (Some(&first), Some(&last)) = (spec.hours.first(), spec.hours.last()) else {
        return Err("the window names no forecast hour".to_string());
    };
    let missing_hours: Vec<u16> = spec
        .hours
        .iter()
        .copied()
        .filter(|&hour| slot_at(hour).is_none())
        .collect();
    if !folds_interval_maxima(spec.source) {
        return if missing_hours.is_empty() {
            Ok((
                spec.hours.iter().filter_map(|&hour| slot_at(hour)).collect(),
                WindowGaps::default(),
            ))
        } else {
            Err(missing_frames_reason(spec, &missing_hours, "frame(s) at"))
        };
    }

    // Planning never hands an interval window a first hour of zero: every
    // one of them starts at F001 or later.
    let start_hour = first.saturating_sub(1);
    if slot_at(last).is_none() {
        return Err(format!(
            "missing stored frame(s) at F{last:03} (window F{start_hour:03}-F{last:03} ends on \
             it; gaps are never skipped)"
        ));
    }
    let mut gaps = WindowGaps::default();
    if !missing_hours.is_empty() {
        gaps.hours = Some(missing_frames_reason(spec, &missing_hours, "frame(s) at"));
    }
    if start_hour > 0 && slot_at(start_hour).is_none() {
        gaps.intervals = Some(format!(
            "missing stored frame(s) at F{start_hour:03} (window F{start_hour:03}-F{last:03} \
             folds every stored frame inside it and needs the frame at its start to place its \
             first interval and show that no frame after it is missing; gaps are never skipped)"
        ));
    }
    let start = u64::from(start_hour) * SECONDS_PER_HOUR;
    let end = u64::from(last) * SECONDS_PER_HOUR;
    let inside: Vec<WindowFrame> = frames
        .iter()
        .copied()
        .filter(|frame| frame.lead_seconds > start && frame.lead_seconds <= end)
        .collect();
    let mut previous = start;
    let mut cadence: Option<(u64, u64, u64)> = None;
    for frame in &inside {
        let step = frame.lead_seconds - previous;
        match cadence {
            None => cadence = Some((step, previous, frame.lead_seconds)),
            Some((expected, from, to)) if step != expected => {
                if gaps.intervals.is_none() {
                    gaps.intervals = Some(format!(
                        "stored frames inside window F{start_hour:03}-F{last:03} are unevenly \
                         spaced: {} to {} is {} where {} to {} is {}; a frame that was never \
                         stored cannot be folded, and a fold without it would read low",
                        lead_label(previous),
                        lead_label(frame.lead_seconds),
                        duration_label(step),
                        lead_label(from),
                        lead_label(to),
                        duration_label(expected)
                    ));
                }
                break;
            }
            Some(_) => {}
        }
        previous = frame.lead_seconds;
    }
    Ok((inside.iter().map(|frame| frame.slot).collect(), gaps))
}

/// The opening of a blocker that is not a gap but a product this history
/// can never draw.  The render reads a pass whose only skips carry it as a
/// successful skip (`gpuwm.render.NOT_APPLICABLE_SKIP`).
pub const NOT_APPLICABLE_PREFIX: &str = "not-applicable: ";

/// A history written on a regular interval of two or more whole hours, as
/// `(interval seconds, phase seconds)`, or `None` (an hourly or finer grid,
/// an interval that is not whole hours, a single frame, or frames on no
/// common interval).
///
/// The interval is the smallest spacing between stored frames; every other
/// spacing must be a whole number of it (a missing frame on that interval is
/// a gap, and stays one).
fn history_cadence(frames: &[WindowFrame]) -> Option<(u64, u64)> {
    let mut leads: Vec<u64> = frames.iter().map(|frame| frame.lead_seconds).collect();
    leads.sort_unstable();
    leads.dedup();
    let steps: Vec<u64> = leads.windows(2).map(|pair| pair[1] - pair[0]).collect();
    let interval = *steps.iter().min()?;
    if interval <= SECONDS_PER_HOUR
        || interval % SECONDS_PER_HOUR != 0
        || steps.iter().any(|step| step % interval != 0)
    {
        return None;
    }
    Some((interval, leads[0] % interval))
}

/// The blocker for a window that needs a whole hour this history never
/// writes.
///
/// WHAT BREAKAGE THIS PREVENTS (gate law): a 3 km nest that keeps 3-hourly
/// history (d02 history 10800 s under a 1-hourly 12 km parent) was asked for
/// `qpf_1h` by the finalize stage's windowed pass; every window F(h-1)-Fh
/// needs a frame the history never writes, so the pass drew nothing, exited
/// 1 and `woof go` printed `FAILED render` on a run whose every picture was
/// in place.  Such a window is not a gap in the history: it does not apply
/// to it, and says so with [`NOT_APPLICABLE_PREFIX`].  A window whose hours
/// all fall on the history's interval (`qpf_3h`, `qpf_6h` on that nest) is
/// planned as before, and a frame missing on the interval stays a gap.
fn off_cadence_reason(
    spec: &ProductSpec,
    cadence: Option<(u64, u64)>,
    exact_axis: bool,
) -> Option<String> {
    let (interval, phase) = cadence?;
    let needed: Vec<u16> = if exact_axis && folds_interval_maxima(spec.source) {
        // On the exact-time axis an interval maximum needs the frame its
        // window ends on and the one at its start (none for a window that
        // starts with the run), not one on every whole hour between
        // (`exact_window_slots`).
        let start = spec.hours.first().and_then(|first| first.checked_sub(1));
        start
            .filter(|&start| start > 0)
            .into_iter()
            .chain(spec.hours.last().copied())
            .collect()
    } else {
        spec.hours.clone()
    };
    let never: Vec<u16> = needed
        .iter()
        .copied()
        .filter(|&hour| (u64::from(hour) * SECONDS_PER_HOUR) % interval != phase)
        .collect();
    if never.is_empty() {
        return None;
    }
    let first = needed.first().copied().unwrap_or_default();
    let last = needed.last().copied().unwrap_or_default();
    Some(format!(
        "{NOT_APPLICABLE_PREFIX}this grid's history is written every {}, so F{} that window \
         F{first:03}-F{last:03} needs is never stored; the product does not apply to this history",
        if interval % SECONDS_PER_HOUR == 0 {
            format!("{} h", interval / SECONDS_PER_HOUR)
        } else {
            duration_label(interval)
        },
        never
            .iter()
            .map(|hour| format!("{hour:03}"))
            .collect::<Vec<_>>()
            .join(", F"),
    ))
}

fn duration_label(seconds: u64) -> String {
    if seconds % 60 == 0 {
        format!("{} min", seconds / 60)
    } else {
        format!("{seconds} s")
    }
}

/// The blocker for a window whose frames are not all stored, naming each
/// missing whole hour and what the window needs of it.
fn missing_frames_reason(spec: &ProductSpec, missing: &[u16], noun: &str) -> String {
    let first = spec.hours.first().copied().unwrap_or_default();
    let last = spec.hours.last().copied().unwrap_or_default();
    let requirement = match spec.reduce {
        // A run-total difference requires exactly its two endpoints and
        // nothing between them; claiming it needs every hour of the span
        // would misreport which stored hours the product actually
        // depends on.
        Reduce::Difference => format!(
            "window F{first:03}-F{last:03} is differenced from the stored run \
             totals at F{first:03} and F{last:03}, both required"
        ),
        _ => format!("window F{first:03}-F{last:03} needs every hour"),
    };
    format!(
        "missing stored {noun} {} ({requirement}; gaps are never skipped)",
        missing
            .iter()
            .map(|hour| format!("F{hour:03}"))
            .collect::<Vec<_>>()
            .join(", "),
    )
}

/// The stored source plane a windowed product reduces.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum SourceKind {
    /// `apcp_1h` (kg/m^2 == mm), the trailing (h-1)->h accumulation.
    Apcp1h,
    /// `apcp_run_total` (kg/m^2 == mm), the 0->h run accumulation.
    ApcpRunTotal,
    /// `uh_2to5km_max_1h` (m^2/s^2), the native sub-hourly MXUPHL max;
    /// falls back to the stored hourly `uh_2to5km` plane when the max
    /// field is absent (stores ingested before it existed).
    Uh2to5km,
    /// `wind_speed_10m_max_1h` (m/s), the native sub-hourly WIND max;
    /// then the wrfout import's WRF WSPD10MAX (`wrf_wspd10max`), and
    /// top-of-hour hypot(`u_10m`, `v_10m`) when both are absent.
    WindSpeed10m,
    /// `temperature_2m` converted K -> degC per hour.
    Temp2mC,
    /// `rh_2m` clamped to 0..100 %.
    Rh2mPct,
    /// `dewpoint_2m` converted K -> degC per hour.
    Dewpoint2mC,
    /// `vpd_2m` (hPa), the ingest-computed derived grid.
    Vpd2mHpa,
}

/// How the per-hour planes reduce into the product grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Reduce {
    /// Single stored plane (1 h / run-total accumulations, 1 h UH/wind).
    Direct,
    Sum,
    Max,
    Min,
    /// Pointwise max - min over the window.
    Range,
    /// Pointwise later - earlier over exactly two stored run totals: the
    /// accumulation window between them. Two hours, never one: see
    /// [`AccumState::DifferencePending`].
    Difference,
}

/// Which stored APCP plane a hourly QPF window can reduce in THIS store.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum QpfSource {
    /// The store carries the native trailing 1 h increment (`apcp_1h`),
    /// the measured quantity: reduce it directly.
    NativeHourly,
    /// The store carries only a run accumulation (the wrfout import lane
    /// stores nothing else): the window is the difference of the run
    /// totals at its two endpoints.
    DifferencedRunTotal,
}

/// The windowed products that reduce hourly APCP increments: the only
/// plans whose source plane depends on what the store actually carries.
fn reduces_hourly_apcp(product: HrrrWindowedProduct) -> bool {
    use HrrrWindowedProduct::*;
    matches!(product, Qpf1h | Qpf6h | Qpf12h | Qpf24h)
}

/// Decide [`QpfSource`] from the anchor hour's variable list: a header
/// read, no plane decode.
///
/// `apcp_1h` present wins unconditionally: it is the measured hourly
/// increment, and a derived difference must never preempt it. The
/// fallback is claimed only when a run-total plane is actually there, so
/// a store carrying neither still blocks naming `apcp_1h` (the plane a
/// QPF window primarily expects) instead of naming only the substitutes.
/// Any failure to inspect the anchor hour also keeps the native plan: the
/// per-hour read reports the real cause (unreadable file, bad metadata)
/// far better than this probe could, and inventing a blocker here would
/// hide it.
fn probe_qpf_source(run_dir: &Path, manifest: &RwsRunManifest, anchor_hour: u16) -> QpfSource {
    let Some(entry) = manifest.hours.get(&anchor_hour) else {
        return QpfSource::NativeHourly;
    };
    let Ok(path) = canonical_contained_path(
        run_dir,
        &run_dir.join(&entry.file),
        &format!("hour F{anchor_hour:03} file"),
    ) else {
        return QpfSource::NativeHourly;
    };
    let Ok(reader) = HourReader::open(&path) else {
        return QpfSource::NativeHourly;
    };
    let has_run_total = reader.variable(APCP_RUN_TOTAL_VAR).is_some()
        || reader.variable(APCP_WRFOUT_RUN_TOTAL_VAR).is_some();
    if reader.variable(APCP_1H_VAR).is_none() && has_run_total {
        QpfSource::DifferencedRunTotal
    } else {
        QpfSource::NativeHourly
    }
}

/// Display-unit conversion applied AFTER the fold (the GRIB lane's order:
/// QPF sums millimeters then divides; wind maxes m/s then multiplies).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Finish {
    None,
    MmToInches,
    MsToKnots,
}

#[derive(Debug, Clone)]
struct ProductSpec {
    product: HrrrWindowedProduct,
    source: SourceKind,
    reduce: Reduce,
    /// Contributing hours, ascending; every one of them is required.
    hours: Vec<u16>,
    window_hours: Option<u16>,
    units: &'static str,
    finish: Finish,
    strategy: String,
}

/// Mirror of the GRIB lane's `plan_windowed_products` + per-kernel window
/// definitions for one product, anchored at the max stored hour. `Err` is
/// the planning blocker reason (same wording as the GRIB lane where the
/// constraint is identical). `qpf_source` is ignored by every product
/// [`reduces_hourly_apcp`] rejects.
fn plan_product(
    product: HrrrWindowedProduct,
    end: u16,
    qpf_source: QpfSource,
) -> Result<ProductSpec, String> {
    use HrrrWindowedProduct::*;
    if let Some(plan) = snapshot_plan(product) {
        if end < plan.window_end {
            return Err(format!(
                "{} requires forecast hour >= {}",
                plan.blocker_label, plan.window_end
            ));
        }
        return Ok(ProductSpec {
            product,
            source: plan.source,
            reduce: plan.reduce,
            hours: (plan.window_start..=plan.window_end).collect(),
            window_hours: Some(plan.window_hours),
            units: plan.units,
            finish: Finish::None,
            strategy: format!(
                "pointwise {} of stored hourly {} snapshots across {}",
                plan.op_label, plan.field_label, plan.window_label
            ),
        });
    }

    let spec = |source, reduce, hours: Vec<u16>, window_hours, units, finish, strategy| {
        Ok(ProductSpec {
            product,
            source,
            reduce,
            hours,
            window_hours,
            units,
            finish,
            strategy,
        })
    };
    // Run-total fallback: ONE difference of the window's two endpoints,
    // not a sum of `window` hourly differences. The intermediate run
    // totals cancel algebraically, so summing them would read `window`
    // extra planes to reach the same quantity through more roundings.
    // Callers guarantee `window <= end`; the window minimum is checked
    // before this runs, so F000 never gets a predecessor invented for it.
    //
    // What makes the "{window} h" label accurate is that `end` and `start`
    // are forecast HOURS on both axes: on the whole-hour axis they are the
    // slots themselves, and on the exact-time axis `exact_window_slots`
    // maps each to the frame whose lead is exactly that many hours, so the
    // gap between them IS that many hours of accumulation. No sub-hourly
    // frame can slip a shorter span in under an hourly name.
    let qpf_difference = |window: u16| {
        let start = end - window;
        spec(
            SourceKind::ApcpRunTotal,
            Reduce::Difference,
            vec![start, end],
            Some(window),
            "in",
            Finish::MmToInches,
            format!(
                "{window} h APCP differenced from stored run-total accumulations \
                 (F{end:03} minus F{start:03}); this store carries no native \
                 {APCP_1H_VAR} plane"
            ),
        )
    };
    let qpf_window = |window: u16| {
        if end < window {
            return Err(format!("{window}-h QPF requires forecast hour >= {window}"));
        }
        match qpf_source {
            QpfSource::NativeHourly => spec(
                SourceKind::Apcp1h,
                Reduce::Sum,
                (end + 1 - window..=end).collect(),
                Some(window),
                "in",
                Finish::MmToInches,
                format!("sum of {window} stored hourly APCP increments ({APCP_1H_VAR})"),
            ),
            QpfSource::DifferencedRunTotal => qpf_difference(window),
        }
    };
    match product {
        Qpf1h => {
            if end < 1 {
                return Err(
                    "1-h QPF requires forecast hour >= 1 because the first 1 h accumulation window ends at F001"
                        .to_string(),
                );
            }
            match qpf_source {
                QpfSource::NativeHourly => spec(
                    SourceKind::Apcp1h,
                    Reduce::Direct,
                    vec![end],
                    Some(1),
                    "in",
                    Finish::MmToInches,
                    format!("stored trailing 1 h APCP accumulation ({APCP_1H_VAR}) at F{end:03}"),
                ),
                QpfSource::DifferencedRunTotal => qpf_difference(1),
            }
        }
        Qpf6h => qpf_window(6),
        Qpf12h => qpf_window(12),
        Qpf24h => qpf_window(24),
        QpfTotal => {
            if end < 1 {
                return Err("total QPF requires forecast hour >= 1".to_string());
            }
            spec(
                SourceKind::ApcpRunTotal,
                Reduce::Direct,
                vec![end],
                None,
                "in",
                Finish::MmToInches,
                format!(
                    "stored run-total APCP accumulation (apcp_run_total, 0-{end} h) at F{end:03}"
                ),
            )
        }
        Uh25km1h => {
            if end < 1 {
                return Err(
                    "1-h UH max requires forecast hour >= 1 because native UH windows start at 0-1 h"
                        .to_string(),
                );
            }
            spec(
                SourceKind::Uh2to5km,
                Reduce::Direct,
                vec![end],
                Some(1),
                "m^2/s^2",
                Finish::None,
                format!(
                    "stored sub-hourly 1 h max 2-5 km UH plane (uh_2to5km_max_1h) at F{end:03}"
                ),
            )
        }
        Uh25km3h => {
            if end < 3 {
                return Err("3-h UH max requires forecast hour >= 3".to_string());
            }
            spec(
                SourceKind::Uh2to5km,
                Reduce::Max,
                (end - 2..=end).collect(),
                Some(3),
                "m^2/s^2",
                Finish::None,
                "pointwise max of stored sub-hourly 1 h max 2-5 km UH planes across \
                 trailing 3 hours"
                    .to_string(),
            )
        }
        Uh25kmRunMax => {
            if end < 1 {
                return Err("run-max UH requires forecast hour >= 1".to_string());
            }
            spec(
                SourceKind::Uh2to5km,
                Reduce::Max,
                (1..=end).collect(),
                None,
                "m^2/s^2",
                Finish::None,
                "run max of stored sub-hourly 1 h max 2-5 km UH planes".to_string(),
            )
        }
        Wind10m1hMax => {
            if end < 1 {
                return Err(
                    "1-h 10 m wind max requires forecast hour >= 1 because native wind max windows start at 0-1 h"
                        .to_string(),
                );
            }
            spec(
                SourceKind::WindSpeed10m,
                Reduce::Direct,
                vec![end],
                Some(1),
                "kt",
                Finish::MsToKnots,
                format!(
                    "stored sub-hourly 1 h max 10 m wind speed (wind_speed_10m_max_1h) at F{end:03}"
                ),
            )
        }
        Wind10mRunMax => {
            if end < 1 {
                return Err("run-max 10 m wind requires forecast hour >= 1".to_string());
            }
            spec(
                SourceKind::WindSpeed10m,
                Reduce::Max,
                (1..=end).collect(),
                None,
                "kt",
                Finish::MsToKnots,
                "run max of stored sub-hourly 1 h max 10 m wind speeds".to_string(),
            )
        }
        Wind10m0to24hMax => {
            if end < 24 {
                return Err("0-24 h 10 m wind max requires forecast hour >= 24".to_string());
            }
            spec(
                SourceKind::WindSpeed10m,
                Reduce::Max,
                (1..=24).collect(),
                Some(24),
                "kt",
                Finish::MsToKnots,
                "max of stored sub-hourly 1 h max 10 m wind speeds across F001-F024".to_string(),
            )
        }
        Wind10m24to48hMax => {
            if end < 48 {
                return Err("24-48 h 10 m wind max requires forecast hour >= 48".to_string());
            }
            spec(
                SourceKind::WindSpeed10m,
                Reduce::Max,
                (25..=48).collect(),
                Some(24),
                "kt",
                Finish::MsToKnots,
                "max of stored sub-hourly 1 h max 10 m wind speeds across F025-F048".to_string(),
            )
        }
        Wind10m0to48hMax => {
            if end < 48 {
                return Err("0-48 h 10 m wind max requires forecast hour >= 48".to_string());
            }
            spec(
                SourceKind::WindSpeed10m,
                Reduce::Max,
                (1..=48).collect(),
                Some(48),
                "kt",
                Finish::MsToKnots,
                "max of stored sub-hourly 1 h max 10 m wind speeds across F001-F048".to_string(),
            )
        }
        _ => unreachable!("surface snapshot window products are handled before the match"),
    }
}

struct SnapshotPlan {
    source: SourceKind,
    reduce: Reduce,
    window_start: u16,
    window_end: u16,
    window_hours: u16,
    /// e.g. "F001-F024" (strategy text).
    window_label: &'static str,
    /// e.g. "0-24 h 2 m surface snapshot window" (planning blocker text,
    /// mirroring the GRIB lane verbatim).
    blocker_label: &'static str,
    field_label: &'static str,
    op_label: &'static str,
    units: &'static str,
}

/// Decompose a 2 m snapshot-window product into its field, window, and
/// reduction, `None` for QPF/UH/wind products.
fn snapshot_plan(product: HrrrWindowedProduct) -> Option<SnapshotPlan> {
    use HrrrWindowedProduct::*;
    let (source, field_label, units) = match product {
        Temp2m0to24hMax | Temp2m24to48hMax | Temp2m0to48hMax | Temp2m0to24hMin
        | Temp2m24to48hMin | Temp2m0to48hMin | Temp2m0to24hRange | Temp2m24to48hRange
        | Temp2m0to48hRange => (SourceKind::Temp2mC, "2 m temperature", "degC"),
        Rh2m0to24hMax | Rh2m24to48hMax | Rh2m0to48hMax | Rh2m0to24hMin | Rh2m24to48hMin
        | Rh2m0to48hMin | Rh2m0to24hRange | Rh2m24to48hRange | Rh2m0to48hRange => {
            (SourceKind::Rh2mPct, "2 m relative humidity", "%")
        }
        Dewpoint2m0to24hMax
        | Dewpoint2m24to48hMax
        | Dewpoint2m0to48hMax
        | Dewpoint2m0to24hMin
        | Dewpoint2m24to48hMin
        | Dewpoint2m0to48hMin
        | Dewpoint2m0to24hRange
        | Dewpoint2m24to48hRange
        | Dewpoint2m0to48hRange => (SourceKind::Dewpoint2mC, "2 m dewpoint", "degC"),
        Vpd2m0to24hMax | Vpd2m24to48hMax | Vpd2m0to48hMax | Vpd2m0to24hMin | Vpd2m24to48hMin
        | Vpd2m0to48hMin | Vpd2m0to24hRange | Vpd2m24to48hRange | Vpd2m0to48hRange => {
            (SourceKind::Vpd2mHpa, "2 m vapor pressure deficit", "hPa")
        }
        _ => return None,
    };
    let (window_start, window_end, window_hours, window_label, blocker_label) = match product {
        Temp2m0to24hMax
        | Temp2m0to24hMin
        | Temp2m0to24hRange
        | Rh2m0to24hMax
        | Rh2m0to24hMin
        | Rh2m0to24hRange
        | Dewpoint2m0to24hMax
        | Dewpoint2m0to24hMin
        | Dewpoint2m0to24hRange
        | Vpd2m0to24hMax
        | Vpd2m0to24hMin
        | Vpd2m0to24hRange => (1, 24, 24, "F001-F024", "0-24 h 2 m surface snapshot window"),
        Temp2m24to48hMax
        | Temp2m24to48hMin
        | Temp2m24to48hRange
        | Rh2m24to48hMax
        | Rh2m24to48hMin
        | Rh2m24to48hRange
        | Dewpoint2m24to48hMax
        | Dewpoint2m24to48hMin
        | Dewpoint2m24to48hRange
        | Vpd2m24to48hMax
        | Vpd2m24to48hMin
        | Vpd2m24to48hRange => (
            25,
            48,
            24,
            "F025-F048",
            "24-48 h 2 m surface snapshot window",
        ),
        _ => (1, 48, 48, "F001-F048", "0-48 h 2 m surface snapshot window"),
    };
    let (reduce, op_label) = match product {
        Temp2m0to24hMax | Temp2m24to48hMax | Temp2m0to48hMax | Rh2m0to24hMax | Rh2m24to48hMax
        | Rh2m0to48hMax | Dewpoint2m0to24hMax | Dewpoint2m24to48hMax | Dewpoint2m0to48hMax
        | Vpd2m0to24hMax | Vpd2m24to48hMax | Vpd2m0to48hMax => (Reduce::Max, "max"),
        Temp2m0to24hMin | Temp2m24to48hMin | Temp2m0to48hMin | Rh2m0to24hMin | Rh2m24to48hMin
        | Rh2m0to48hMin | Dewpoint2m0to24hMin | Dewpoint2m24to48hMin | Dewpoint2m0to48hMin
        | Vpd2m0to24hMin | Vpd2m24to48hMin | Vpd2m0to48hMin => (Reduce::Min, "min"),
        _ => (Reduce::Range, "max-min range"),
    };
    Some(SnapshotPlan {
        source,
        reduce,
        window_start,
        window_end,
        window_hours,
        window_label,
        blocker_label,
        field_label,
        op_label,
        units,
    })
}

/// A WRF history field that holds the maximum over the history interval
/// ending at its frame, reset at every history write, as the wrfout import
/// stores it: one row per windowed source that has one.  The row is read
/// by [`read_source_plane`] and names the field in the strategy note
/// [`Accum::finish`] writes.
///
/// WHAT BREAKAGE THIS PREVENTS (gate law): the wrfout import stores WRF
/// WSPD10MAX as the raw extra `wrf_wspd10max`, and the 10 m wind windows
/// read only the GRIB lane's `wind_speed_10m_max_1h` before falling back
/// to top-of-hour U10/V10 speeds.  A WRF history with nwp_diagnostics = 1
/// holds the true wind maximum, and its 1 h and run maxima were drawn as
/// hourly snapshots, titled "no stored max" and described as a history
/// that stores no WSPD10MAX.
#[derive(Debug, PartialEq, Eq)]
struct HistoryIntervalMaxRow {
    source: SourceKind,
    /// The store variable the wrfout import writes it under.
    store_name: &'static str,
    /// The unit spellings the import may stamp on it: the lane's own and
    /// WRF's Registry spelling.  Any other unit blocks the window.
    units: &'static [&'static str],
    /// The WRF history field it is, for the strategy note.
    wrf_name: &'static str,
}

static HISTORY_INTERVAL_MAX_ROWS: &[HistoryIntervalMaxRow] = &[
    HistoryIntervalMaxRow {
        source: SourceKind::Uh2to5km,
        store_name: "updraft_helicity_2to5km",
        units: &["m^2/s^2", "m2/s2"],
        wrf_name: "UP_HELI_MAX",
    },
    HistoryIntervalMaxRow {
        source: SourceKind::WindSpeed10m,
        store_name: "wrf_wspd10max",
        units: &["m/s", "m s-1"],
        wrf_name: "WSPD10MAX",
    },
];

/// The history-interval maximum `source` reads, if it has one.
fn history_interval_max_row(source: SourceKind) -> Option<&'static HistoryIntervalMaxRow> {
    HISTORY_INTERVAL_MAX_ROWS.iter().find(|row| row.source == source)
}

/// Every store variable [`read_source_plane`] may read for `source`, in the
/// order it tries them: the native plane, the wrfout import's
/// history-interval maximum ([`HISTORY_INTERVAL_MAX_ROWS`]), then the
/// instantaneous fallbacks.
fn source_store_names(source: SourceKind) -> Vec<&'static str> {
    let mut names: Vec<&'static str> = match source {
        SourceKind::Apcp1h => vec![APCP_1H_VAR],
        SourceKind::ApcpRunTotal => vec![APCP_RUN_TOTAL_VAR, APCP_WRFOUT_RUN_TOTAL_VAR],
        SourceKind::Uh2to5km => vec!["uh_2to5km_max_1h", "uh_2to5km"],
        SourceKind::WindSpeed10m => vec!["wind_speed_10m_max_1h"],
        SourceKind::Temp2mC => vec!["temperature_2m"],
        SourceKind::Rh2mPct => vec!["rh_2m", "relative_humidity_2m"],
        SourceKind::Dewpoint2mC => vec!["dewpoint_2m"],
        SourceKind::Vpd2mHpa => vec!["vpd_2m"],
    };
    if let Some(row) = history_interval_max_row(source) {
        names.push(row.store_name);
    }
    if source == SourceKind::WindSpeed10m {
        names.extend(["u_10m", "v_10m"]);
    }
    names
}

/// Every store variable `product`'s window may read, over every plan it can
/// take (a QPF window reads `apcp_1h` or differences the run totals,
/// whichever the store carries).  What a scoped wrfout import keeps for a
/// window (`rw-wrfbatch`'s `required_fields_for_products`).
///
/// WHAT BREAKAGE THIS PREVENTS (gate law): the scoped import kept only the
/// window's recipe selector, so `10m_wind_1h_max` and the other 10 m wind
/// windows found neither `wrf_wspd10max` nor `u_10m`/`v_10m` in the store
/// and drew no picture, while the whole import they fell back to without a
/// scope spent about 5 minutes importing 25 frames at 3 km for the
/// end-of-run windowed pass.
pub fn window_store_reads(product: HrrrWindowedProduct) -> Vec<&'static str> {
    let Some(end) = minimum_window_hour(product) else {
        return Vec::new();
    };
    let mut names: Vec<&'static str> = Vec::new();
    for source in [QpfSource::NativeHourly, QpfSource::DifferencedRunTotal] {
        if let Ok(spec) = plan_product(product, end, source) {
            for name in source_store_names(spec.source) {
                if !names.contains(&name) {
                    names.push(name);
                }
            }
        }
    }
    names
}

/// What one hour's source plane actually measures: recorded so the
/// product's strategy note can label the fold accurately.  The
/// distinction matters scientifically: an instantaneous snapshot makes
/// the fold a lower bound on the sub-hourly max, while WRF's
/// UP_HELI_MAX and WSPD10MAX planes are themselves exact per-interval
/// maxima.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PlaneFidelity {
    /// The stored sub-hourly max field (or an exact-equivalent plane):
    /// the fold is exact, no note needed.
    Exact,
    /// A [`HistoryIntervalMaxRow`] field (WRF UP_HELI_MAX, WSPD10MAX)
    /// pulled verbatim from the wrfout import lane:
    /// the exact running max over the source run's history interval
    /// ending at this frame (reset at every history write).  NOT an
    /// instantaneous snapshot and NOT a lower bound, as long as every
    /// history frame of the window is folded.  A sub-hourly history
    /// puts the store on the exact-time axis, where every stored frame
    /// inside the window is folded (`exact_window_slots`).  On the
    /// whole-hour axis each plane is the exact trailing 1 h max only if
    /// the source wrote history hourly; a run with sub-hourly history
    /// imported at its whole hours alone holds just the last interval,
    /// and the strategy note says so rather than claiming the hour.
    HistoryIntervalMax(&'static HistoryIntervalMaxRow),
    /// Top-of-hour instantaneous plane (no stored max field at all):
    /// a genuine lower bound on the native sub-hourly max.
    InstantaneousLowerBound,
}

/// One source plane read for one hour: the fold-ready values plus what
/// they measure ([`PlaneFidelity`]).
struct SourcePlane {
    values: Vec<f64>,
    fidelity: PlaneFidelity,
}

impl SourcePlane {
    fn exact(values: Vec<f64>) -> Self {
        Self {
            values,
            fidelity: PlaneFidelity::Exact,
        }
    }
}

/// Why one stored-variable read failed: the variable is absent from the
/// hour file (eligible for the documented instantaneous fallback) vs any
/// other failure (unit drift, codec error, always a blocker, never a
/// silent fallback).
enum ReadFailure {
    MissingVariable(String),
    Failed(String),
}

impl ReadFailure {
    fn into_reason(self) -> String {
        match self {
            Self::MissingVariable(reason) | Self::Failed(reason) => reason,
        }
    }
}

/// Read one source plane for one hour, unit-checked and transformed to the
/// per-hour values the fold consumes (the GRIB lane's per-hour transforms:
/// K -> degC, RH clamp; accumulation/UH/wind planes stay raw, their
/// display conversion happens after the fold). UH/wind prefer the stored
/// sub-hourly max fields and fall back to the instantaneous planes ONLY
/// when the max variable is absent (older stores); a max field present
/// with wrong units blocks instead of falling back.
fn read_source_plane(
    reader: &HourReader,
    grid: &GridFile,
    kind: SourceKind,
    frame: &FrameName,
) -> Result<SourcePlane, String> {
    let (at, noun) = (frame.at.as_str(), frame.noun.as_str());
    let read_any_units = |name: &str,
                          expected_units: &[&str]|
     -> Result<Vec<f32>, ReadFailure> {
        match read_grid_2d(reader, grid, name) {
            Ok(stored) => {
                if !expected_units.contains(&stored.units.as_str()) {
                    return Err(ReadFailure::Failed(format!(
                        "stored '{name}' at {at} has units '{}', expected '{}'",
                        stored.units,
                        expected_units.join("' or '")
                    )));
                }
                Ok(stored.values)
            }
            Err(RwStoreError::UnknownVariable(_)) => Err(ReadFailure::MissingVariable(format!(
                "stored {noun} has no '{name}' variable"
            ))),
            Err(err) => Err(ReadFailure::Failed(format!(
                "read '{name}' from stored {noun}: {err}"
            ))),
        }
    };
    let read =
        |name: &str, expected_units: &str| -> Result<Vec<f32>, ReadFailure> {
            read_any_units(name, &[expected_units])
        };
    let plain = |result: Result<Vec<f32>, ReadFailure>| -> Result<Vec<f32>, String> {
        result.map_err(ReadFailure::into_reason)
    };
    // The wrfout import's per-history-interval maximum of this source
    // ([`HISTORY_INTERVAL_MAX_ROWS`]), read only when the native max field
    // is absent.
    let read_interval_max = || -> Result<SourcePlane, ReadFailure> {
        let row = history_interval_max_row(kind).ok_or_else(|| {
            ReadFailure::MissingVariable(format!(
                "no history-interval maximum is read for {kind:?}"
            ))
        })?;
        read_any_units(row.store_name, row.units).map(|values| SourcePlane {
            values: to_f64(values),
            fidelity: PlaneFidelity::HistoryIntervalMax(row),
        })
    };
    match kind {
        SourceKind::Apcp1h => Ok(SourcePlane::exact(to_f64(plain(read(
            APCP_1H_VAR,
            "kg/m^2",
        ))?))),
        // The GRIB ingest lane stores the run accumulation as
        // `apcp_run_total`; the wrfout import lane stores the same
        // physical plane (RAINC + RAINNC since simulation start, kg/m^2)
        // under its WRF Registry name `apcp`.  Same quantity, second
        // name, exact-plane semantics (no fallback flag).
        SourceKind::ApcpRunTotal => match read(APCP_RUN_TOTAL_VAR, "kg/m^2") {
            Ok(values) => Ok(SourcePlane::exact(to_f64(values))),
            Err(ReadFailure::Failed(reason)) => Err(reason),
            Err(ReadFailure::MissingVariable(missing)) => {
                match read(APCP_WRFOUT_RUN_TOTAL_VAR, "kg/m^2") {
                    Ok(values) => Ok(SourcePlane::exact(to_f64(values))),
                    Err(err) => Err(format!(
                        "{missing}; wrfout-lane '{APCP_WRFOUT_RUN_TOTAL_VAR}' fallback also \
                         unavailable: {}",
                        err.into_reason()
                    )),
                }
            }
        },
        SourceKind::Uh2to5km => match read("uh_2to5km_max_1h", "m^2/s^2") {
            Ok(values) => Ok(SourcePlane::exact(to_f64(values))),
            Err(ReadFailure::Failed(reason)) => Err(reason),
            Err(ReadFailure::MissingVariable(missing)) => {
                // Two fallbacks with DIFFERENT semantics: the GRIB
                // lane's `uh_2to5km` is a top-of-hour instantaneous
                // snapshot (a lower bound on the sub-hourly max); the
                // wrfout import's `updraft_helicity_2to5km` is WRF
                // UP_HELI_MAX pulled verbatim -- the exact max over the
                // history interval ending at its frame, reset at each
                // history write.  That is the trailing 1 h max only on
                // an hourly history; a sub-hourly history puts the store
                // on the exact-time axis, where every frame of the
                // window is folded ([`PlaneFidelity::HistoryIntervalMax`]).
                match read("uh_2to5km", "m^2/s^2") {
                    Ok(values) => Ok(SourcePlane {
                        values: to_f64(values),
                        fidelity: PlaneFidelity::InstantaneousLowerBound,
                    }),
                    Err(ReadFailure::Failed(reason)) => Err(reason),
                    // The wrfout import stamps WRF's Registry spelling
                    // "m2/s2"; same quantity as the GRIB lane's "m^2/s^2".
                    Err(_) => read_interval_max().map_err(|err| {
                        format!(
                            "{missing}; hourly 'uh_2to5km' and wrfout-lane \
                             'updraft_helicity_2to5km' fallbacks also unavailable: {}",
                            err.into_reason()
                        )
                    }),
                }
            }
        },
        SourceKind::WindSpeed10m => match read("wind_speed_10m_max_1h", "m/s") {
            Ok(values) => Ok(SourcePlane::exact(to_f64(values))),
            Err(ReadFailure::Failed(reason)) => Err(reason),
            Err(ReadFailure::MissingVariable(missing)) => {
                // The wrfout import's `wrf_wspd10max` is WRF WSPD10MAX
                // pulled verbatim, the same meaning as UP_HELI_MAX above:
                // the exact max over the history interval ending at its
                // frame, reset at each history write.  Present with other
                // units it blocks; only a history without it falls back to
                // the top-of-hour speed, a lower bound.
                match read_interval_max() {
                    Ok(plane) => Ok(plane),
                    Err(ReadFailure::Failed(reason)) => Err(reason),
                    Err(ReadFailure::MissingVariable(no_interval_max)) => {
                        let speeds = (|| -> Result<Vec<f64>, ReadFailure> {
                            let u = read("u_10m", "m/s")?;
                            let v = read("v_10m", "m/s")?;
                            Ok(u.iter()
                                .zip(&v)
                                .map(|(&u, &v)| f64::from(u).hypot(f64::from(v)))
                                .collect())
                        })();
                        match speeds {
                            Ok(values) => Ok(SourcePlane {
                                values,
                                fidelity: PlaneFidelity::InstantaneousLowerBound,
                            }),
                            Err(err) => Err(format!(
                                "{missing}; {no_interval_max}; hypot(u_10m, v_10m) fallback \
                                 also unavailable: {}",
                                err.into_reason()
                            )),
                        }
                    }
                }
            }
        },
        SourceKind::Temp2mC | SourceKind::Dewpoint2mC => {
            let name = if kind == SourceKind::Temp2mC {
                "temperature_2m"
            } else {
                "dewpoint_2m"
            };
            Ok(SourcePlane::exact(
                plain(read(name, "K"))?
                    .iter()
                    .map(|&value| f64::from(value) - 273.15)
                    .collect(),
            ))
        }
        SourceKind::Rh2mPct => {
            // GRIB lane: `rh_2m`; wrfout import lane: the same 2 m RH
            // plane under its canonical name `relative_humidity_2m`.
            let values = match read("rh_2m", "%") {
                Ok(values) => values,
                Err(ReadFailure::Failed(reason)) => return Err(reason),
                Err(ReadFailure::MissingVariable(missing)) => {
                    match read("relative_humidity_2m", "%") {
                        Ok(values) => values,
                        Err(err) => {
                            return Err(format!(
                                "{missing}; wrfout-lane 'relative_humidity_2m' fallback \
                                 also unavailable: {}",
                                err.into_reason()
                            ));
                        }
                    }
                }
            };
            Ok(SourcePlane::exact(
                values
                    .iter()
                    .map(|&value| f64::from(value).clamp(0.0, 100.0))
                    .collect(),
            ))
        }
        SourceKind::Vpd2mHpa => Ok(SourcePlane::exact(to_f64(plain(read("vpd_2m", "hPa"))?))),
    }
}

fn to_f64(values: Vec<f32>) -> Vec<f64> {
    values.into_iter().map(f64::from).collect()
}

/// `fold(a, b)`, or NaN (the store's missing value) when either is missing.
fn missing_or(a: f64, b: f64, fold: impl Fn(f64, f64) -> f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        fold(a, b)
    }
}

/// Per-product streaming accumulator: per-frame planes fold in ascending
/// slot order; `failed` records the first per-frame read failure (the
/// product's blocker reason: once failed, later frames stop folding).
/// `slots` are the stored frames the window reads: its hours on the
/// whole-hour axis, and on the exact-time axis the frames at its whole-
/// hour leads, or every frame inside it for an interval maximum.
/// `fallback_frames` collects the frames whose plane was a genuine
/// instantaneous snapshot (lower-bound note); `interval_max_frames` the
/// frames served by a wrfout-lane per-history-interval maximum
/// ([`HistoryIntervalMaxRow`], exact-semantics note), and
/// `interval_max_field` the WRF field it was; `exact_planes` counts the
/// native max planes.  `gaps` is what the window's stored frames lack,
/// refused in `finish` for the planes it would make read low.
struct Accum {
    spec: ProductSpec,
    slots: Vec<u16>,
    state: Option<AccumState>,
    failed: Option<String>,
    fallback_frames: Vec<String>,
    interval_max_frames: Vec<String>,
    interval_max_field: Option<&'static str>,
    exact_planes: usize,
    gaps: WindowGaps,
    footprint_swath: bool,
}

enum AccumState {
    Sum(Vec<f64>),
    Max(Vec<f64>),
    Min(Vec<f64>),
    Range {
        max: Vec<f64>,
        min: Vec<f64>,
    },
    Direct(Vec<f64>),
    /// [`Reduce::Difference`] after ONE fold: the window's earlier run
    /// total, which is not the product and carries no value out of
    /// `finish`.  The variant exists precisely so a difference that saw
    /// only one hour blocks instead of publishing a run accumulation
    /// relabeled as a window increment.
    DifferencePending(Vec<f64>),
    /// [`Reduce::Difference`] after both folds: the window increment.
    DifferenceReady(Vec<f64>),
}

impl Accum {
    fn new(spec: ProductSpec, slots: Vec<u16>) -> Self {
        Self {
            spec,
            slots,
            state: None,
            failed: None,
            fallback_frames: Vec::new(),
            interval_max_frames: Vec::new(),
            interval_max_field: None,
            exact_planes: 0,
            gaps: WindowGaps::default(),
            footprint_swath: false,
        }
    }

    fn with_gaps(mut self, gaps: WindowGaps) -> Self {
        self.gaps = gaps;
        self
    }

    fn fold_footprint(&mut self, values: &[f64], mask: &[f32]) {
        if mask.len() != values.len() || mask.iter().any(|m| *m != 0. && *m != 1.) {
            self.failed = Some("invalid moving-nest footprint mask".into());
            return;
        }
        let sampled: Vec<_> = values.iter().zip(mask).map(|(v,m)|
            if *m == 0. {f64::NEG_INFINITY} else {*v}).collect();
        self.fold(&sampled);
    }

    fn fold(&mut self, values: &[f64]) {
        match &mut self.state {
            None => {
                self.state = Some(match self.spec.reduce {
                    Reduce::Direct => AccumState::Direct(values.to_vec()),
                    Reduce::Sum => AccumState::Sum(values.to_vec()),
                    Reduce::Max => AccumState::Max(values.to_vec()),
                    Reduce::Min => AccumState::Min(values.to_vec()),
                    Reduce::Range => AccumState::Range {
                        max: values.to_vec(),
                        min: values.to_vec(),
                    },
                    Reduce::Difference => AccumState::DifferencePending(values.to_vec()),
                });
            }
            Some(AccumState::Direct(_)) => {
                unreachable!("direct windowed products fold exactly one hour")
            }
            Some(AccumState::Sum(acc)) => {
                for (target, value) in acc.iter_mut().zip(values) {
                    *target += *value;
                }
            }
            // A missing value (NaN) is missing for the whole window: `f64::max`
            // and `min` return the other operand, which would fold a cell
            // from fewer frames than its window holds and draw it low (a
            // moving nest's new ground holds no value from before the nest
            // arrived, rw_wrfbatch's `nest_move`).  Gaps are never skipped.
            Some(AccumState::Max(acc)) => {
                for (target, value) in acc.iter_mut().zip(values) {
                    *target = missing_or(*target, *value, f64::max);
                }
            }
            Some(AccumState::Min(acc)) => {
                for (target, value) in acc.iter_mut().zip(values) {
                    *target = missing_or(*target, *value, f64::min);
                }
            }
            Some(AccumState::Range { max, min }) => {
                for ((max, min), value) in max.iter_mut().zip(min.iter_mut()).zip(values) {
                    *max = missing_or(*max, *value, f64::max);
                    *min = missing_or(*min, *value, f64::min);
                }
            }
            Some(AccumState::DifferencePending(start)) => {
                // Hour order is not an assumption here, it is enforced
                // upstream: `slots_needed` is a BTreeMap streamed in
                // ascending hour order, so the fold already held is the
                // window's EARLIER endpoint and `values` is the later
                // one.  Anything else would be a planning bug, not a
                // data condition.
                let mut increment = std::mem::take(start);
                for (target, value) in increment.iter_mut().zip(values) {
                    // Round-off guard, not cosmetics: this producer's
                    // RAINC/RAINNC are monotonic (no bucket reset), so a
                    // true negative increment cannot occur, but the
                    // stored run totals are f32, and differencing two
                    // large near-equal ones can land a hair below zero.
                    // Clamping publishes the physical floor instead of a
                    // negative rainfall pixel.  A missing endpoint stays
                    // missing: `NaN.max(0.0)` is 0.0, which would draw a
                    // cell the earlier frame never covered as a dry hour.
                    *target = missing_or(*value, *target, |later, earlier| {
                        (later - earlier).max(0.0)
                    });
                }
                self.state = Some(AccumState::DifferenceReady(increment));
            }
            Some(AccumState::DifferenceReady(_)) => {
                unreachable!("differenced windows fold exactly two hours")
            }
        }
    }

    /// The product grid, or the reason it cannot be one.  `exact_axis`
    /// picks how the strategy note names frames and what it can claim.
    fn finish(self, exact_axis: bool) -> Result<WindowedGrid, String> {
        if let Some(reason) = self.failed {
            return Err(reason);
        }
        // A gap makes every fold read low, a labelled lower bound
        // included: the picture is kept, and a render of the whole series
        // would have drawn it higher ([`WindowGaps`]).  Only a native 1 h
        // max plane reads no frame between the hours, and only evenly
        // spaced interval maxima cover a whole hour without its frame.
        let instants = !self.fallback_frames.is_empty();
        if instants || !self.interval_max_frames.is_empty() {
            if let Some(reason) = self.gaps.intervals {
                return Err(reason);
            }
        }
        if instants || self.exact_planes > 0 {
            if let Some(reason) = self.gaps.hours {
                return Err(reason);
            }
        }
        let mut values = match self.state {
            None => {
                return Err("no stored hours folded into this window".to_string());
            }
            // A run-total difference that folded one endpoint holds a
            // run accumulation, NOT a window increment.  Publishing it
            // would silently relabel "rain since simulation start" as
            // "rain this window", so the half-folded state is a blocker.
            Some(AccumState::DifferencePending(_)) => {
                return Err(format!(
                    "run-total differencing folded only one of the two window endpoints \
                     ({}); a window increment is undefined without both",
                    self.spec
                        .hours
                        .iter()
                        .map(|hour| format!("F{hour:03}"))
                        .collect::<Vec<_>>()
                        .join(", ")
                ));
            }
            Some(AccumState::Direct(values))
            | Some(AccumState::Sum(values))
            | Some(AccumState::Max(values))
            | Some(AccumState::Min(values))
            | Some(AccumState::DifferenceReady(values)) => values,
            Some(AccumState::Range { max, min }) => max
                .into_iter()
                .zip(min)
                .map(|(max, min)| max - min)
                .collect(),
        };
        if self.footprint_swath {
            for value in &mut values {if *value == f64::NEG_INFINITY {*value = f64::NAN;}}
        }
        match self.spec.finish {
            Finish::None => {}
            Finish::MmToInches => {
                for value in values.iter_mut() {
                    *value /= MM_PER_INCH;
                }
            }
            Finish::MsToKnots => {
                for value in values.iter_mut() {
                    *value *= MS_TO_KT;
                }
            }
        }
        // Top-of-hour snapshots on the whole-hour axis fold to the largest
        // hourly snapshot, not the window's maximum.  A product whose
        // catalog row names that fold is titled and described by the row
        // ([`rustwx_products::windowed::SnapshotFoldRow`]); every other
        // fold keeps its product title and its note below.  A window whose
        // other hours read a stored maximum, native or per history
        // interval, is titled as partly snapshots.
        let snapshot_fold = if exact_axis || self.fallback_frames.is_empty() {
            None
        } else {
            self.spec.product.snapshot_fold()
        };
        let read_a_maximum = self.exact_planes > 0 || !self.interval_max_frames.is_empty();
        let title = match snapshot_fold {
            None => self.spec.product.title(),
            Some(row) if read_a_maximum => row.partial_title.unwrap_or(row.title),
            Some(row) => row.title,
        };
        let mut strategy = self.spec.strategy;
        if let Some(row) = snapshot_fold {
            strategy = format!(
                "{} at {} ({})",
                row.fold,
                self.fallback_frames.join(", "),
                row.why
            );
            if read_a_maximum {
                strategy.push_str("; the other hours read the stored sub-hourly maximum");
            }
        } else if !self.fallback_frames.is_empty() {
            let frames = self.fallback_frames.join(", ");
            if exact_axis {
                strategy.push_str(&format!(
                    " (instantaneous fallback at {frames}: no stored max field, so this is \
                     the max of the stored instants, a lower bound on the true max)"
                ));
            } else {
                strategy.push_str(&format!(
                    " (top-of-hour instantaneous fallback at {frames}: no stored sub-hourly max \
                     field, a lower bound on the native sub-hourly max)"
                ));
            }
        }
        if let Some(field) = self.interval_max_field {
            // The wrfout lane's per-history-interval maxima (WRF
            // UP_HELI_MAX, WSPD10MAX: [`HISTORY_INTERVAL_MAX_ROWS`]) are
            // reset at every history write, so each folded plane is the
            // exact max over the history interval ending at its frame, not
            // a lower bound (that wording is reserved for genuinely
            // instantaneous planes above).  On the exact-time axis every
            // stored frame of the window was folded, so the fold is the
            // window's max when every history frame was stored; a series
            // thinned to every other file is evenly spaced too, and the
            // store cannot tell it from a whole one.  On the whole-hour
            // axis each plane is the whole hour only if the run wrote
            // history hourly, which the store cannot prove either.
            let frames = self.interval_max_frames.join(", ");
            if exact_axis {
                strategy.push_str(&format!(
                    " (WRF {field} per-history-interval max at {frames}: reset at each \
                     history write and folded over every stored frame inside the window, so \
                     this is the exact max over the window when every history frame of the \
                     run was rendered; a series thinned to fewer frames reads low)"
                ));
            } else {
                strategy.push_str(&format!(
                    " (WRF {field} per-history-interval max at {frames}: reset at each \
                     history write, so this is the exact trailing 1 h max when history is \
                     written hourly; a run with sub-hourly history is exact only when \
                     rendered with its frames between the hours)"
                ));
            }
        }
        if self.footprint_swath {
            strategy.push_str("; union of explicitly sampled nest footprints; periods outside the nest are unobserved, never parent-filled");
            if self.spec.product == HrrrWindowedProduct::QpfTotal {
                strategy.push_str("; only history-sampled totals are available: rain between the last saved frame and departure can be absent from a departed strip; use committed-step cyclone rain for the full sampled exposure");
            }
        }
        Ok(WindowedGrid {
            slug: self.spec.product.slug().to_string(),
            units: self.spec.units.to_string(),
            title: if self.footprint_swath {
                if title == "10 m Wind Speed (largest hourly snapshot, no stored max)" {
                    "10 m wind max (hourly samples; sampled nest exposure)".to_string()
                } else if self.spec.product == HrrrWindowedProduct::QpfTotal {
                    "Nest rain total (sampled nest exposure)".to_string()
                } else { format!("{title} (sampled nest exposure)") }
            } else { title.to_string() },
            values,
            hours_used: self.spec.hours,
            window_hours: self.spec.window_hours,
            strategy,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;

    use rustwx_core::{CanonicalField, FieldSelector, GridShape, LatLonGrid, SelectedField2D};
    use rw_store::RwsExactTime;
    use rw_store::ingest::{
        DerivedFieldInput, write_hour_from_fields_with_derived,
        write_hour_from_fields_with_derived_exact,
    };

    #[test]
    fn moving_nest_max_retains_both_strips_but_not_missing_interior() {
        let spec = plan_product(HrrrWindowedProduct::Wind10mRunMax, 2, QpfSource::NativeHourly).unwrap();
        let mut a = Accum::new(spec, vec![1, 2]);
        a.footprint_swath = true;
        a.fold_footprint(&[25., 20., f64::NAN, f64::NAN, f64::NAN], &[1., 1., 0., 1., 0.]);
        a.fold_footprint(&[f64::NAN, 30., 35., 40., f64::NAN], &[0., 1., 1., 1., 0.]);
        let result = a.finish(false).unwrap();
        assert_eq!(&result.values[..3], &[25.*MS_TO_KT, 30.*MS_TO_KT, 35.*MS_TO_KT]);
        assert!(result.values[3].is_nan()); // Missing inside a footprint stays missing.
        assert!(result.values[4].is_nan()); // Never visited is not a zero wind.
        assert!(result.title.contains("sampled nest exposure"));
    }

    const NX: usize = 2;
    const NY: usize = 2;
    const CELLS: usize = NX * NY;

    fn test_dir(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("rw-windowed-store-{}-{name}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn grid() -> LatLonGrid {
        LatLonGrid::new(
            GridShape::new(NX, NY).unwrap(),
            vec![40.0, 40.0, 41.0, 41.0],
            vec![-100.0, -99.0, -100.0, -99.0],
        )
        .unwrap()
    }

    fn field(selector: FieldSelector, units: &str, values: Vec<f32>) -> SelectedField2D {
        SelectedField2D {
            selector,
            units: units.to_string(),
            grid: grid(),
            values,
            projection: None,
        }
    }

    // --- deterministic per-(variable, hour, cell) synthetic planes ---

    fn apcp_1h_plane(hour: u16) -> Vec<f32> {
        (0..CELLS)
            .map(|cell| 0.25 * hour as f32 + 0.05 * cell as f32)
            .collect()
    }

    fn apcp_total_plane(hour: u16) -> Vec<f32> {
        (0..CELLS)
            .map(|cell| 10.0 + hour as f32 + 0.5 * cell as f32)
            .collect()
    }

    /// Run accumulation as the wrfout import lane stores it (`apcp`,
    /// kg/m^2 since simulation start): monotonic in hour, a DIFFERENT
    /// increment per cell, and already nonzero at F000, so a window
    /// that published the run total, or differenced the wrong pair of
    /// hours, cannot coincide with the expected increment. Every value
    /// and every difference is an exact binary fraction, so the
    /// expectations below are bit-exact.
    fn apcp_run_accum_plane(hour: u16) -> Vec<f32> {
        (0..CELLS)
            .map(|cell| 2.0 + (0.75 + 0.25 * cell as f32) * hour as f32)
            .collect()
    }

    /// Non-monotonic in hour AND cell so pointwise maxima differ per cell.
    fn uh_plane(hour: u16) -> Vec<f32> {
        let by_hour: &[[f32; 4]] = &[
            [5.0, 50.0, 1.0, 0.0],  // F001
            [60.0, 10.0, 2.0, 0.0], // F002
            [20.0, 30.0, 3.0, 0.0], // F003
            [25.0, 5.0, 4.0, 0.0],  // F004
            [10.0, 45.0, 5.0, 0.0], // F005
            [30.0, 20.0, 6.0, 0.0], // F006
        ];
        by_hour[(hour as usize - 1) % by_hour.len()].to_vec()
    }

    /// Sub-hourly 1 h max UH planes (`uh_2to5km_max_1h`): the hourly plane
    /// plus a positive sub-hourly excess, so a fold that wrongly read the
    /// instantaneous fallback would miss every expectation.
    fn uh_max_plane(hour: u16) -> Vec<f32> {
        uh_plane(hour).iter().map(|value| value + 6.25).collect()
    }

    /// Exact Pythagorean (u, v) pairs so hypot folds bit-exactly.
    fn wind_uv_planes(hour: u16) -> (Vec<f32>, Vec<f32>) {
        let by_hour: &[([f32; 4], [f32; 4])] = &[
            ([3.0, 0.0, 8.0, 20.0], [4.0, 5.0, 15.0, 21.0]), // speeds 5 5 17 29
            ([6.0, 5.0, 0.0, 3.0], [8.0, 12.0, 2.0, 4.0]),   // speeds 10 13 2 5
            ([0.0, 3.0, 6.0, 5.0], [5.0, 4.0, 8.0, 12.0]),   // speeds 5 5 10 13
            ([8.0, 0.0, 3.0, 0.0], [15.0, 1.0, 4.0, 2.0]),   // speeds 17 1 5 2
            ([20.0, 6.0, 0.0, 8.0], [21.0, 8.0, 5.0, 15.0]), // speeds 29 10 5 17
            ([5.0, 20.0, 3.0, 6.0], [12.0, 21.0, 4.0, 8.0]), // speeds 13 29 5 10
        ];
        let (u, v) = &by_hour[(hour as usize - 1) % by_hour.len()];
        (u.to_vec(), v.to_vec())
    }

    /// Sub-hourly 1 h max wind speed plane (`wind_speed_10m_max_1h`, m/s):
    /// strictly above the hourly hypot(u, v) snapshot, so a fold that
    /// wrongly used the fallback would miss every expectation.
    fn wind_max_plane(hour: u16) -> Vec<f32> {
        let (u, v) = wind_uv_planes(hour);
        u.iter().zip(&v).map(|(&u, &v)| u.hypot(v) + 1.5).collect()
    }

    /// Quadratic in hour (peak at F012) so max/min land mid-window.
    fn temp_k_plane(hour: u16) -> Vec<f32> {
        (0..CELLS)
            .map(|cell| 285.0 + cell as f32 - 0.1 * (hour as f32 - 12.0) * (hour as f32 - 12.0))
            .collect()
    }

    /// Crosses 100 % at later hours to exercise the clamp.
    fn rh_plane(hour: u16) -> Vec<f32> {
        (0..CELLS)
            .map(|cell| -5.0 + 5.0 * hour as f32 + cell as f32)
            .collect()
    }

    fn dewpoint_k_plane(hour: u16) -> Vec<f32> {
        (0..CELLS)
            .map(|cell| 278.0 + 0.5 * cell as f32 + 0.2 * hour as f32)
            .collect()
    }

    fn vpd_plane(hour: u16) -> Vec<f32> {
        (0..CELLS)
            .map(|cell| 0.3 * hour as f32 + 0.1 * cell as f32)
            .collect()
    }

    /// Write one synthetic hour carrying every windowed source variable
    /// except `skip_vars`, mirroring the ingest's store names and native
    /// units (`temperature_2m` always present as the grid carrier).
    fn write_test_hour(store_root: &Path, run: &str, hour: u16, skip_vars: &[&str]) {
        let temp = field(
            FieldSelector::height_agl(CanonicalField::Temperature, 2),
            "K",
            temp_k_plane(hour),
        );
        let dewpoint = field(
            FieldSelector::height_agl(CanonicalField::Dewpoint, 2),
            "K",
            dewpoint_k_plane(hour),
        );
        let rh = field(
            FieldSelector::height_agl(CanonicalField::RelativeHumidity, 2),
            "%",
            rh_plane(hour),
        );
        let (u_values, v_values) = wind_uv_planes(hour);
        let u10 = field(
            FieldSelector::height_agl(CanonicalField::UWind, 10),
            "m/s",
            u_values,
        );
        let v10 = field(
            FieldSelector::height_agl(CanonicalField::VWind, 10),
            "m/s",
            v_values,
        );
        let apcp_1h = field(
            FieldSelector::surface(CanonicalField::TotalPrecipitation),
            "kg/m^2",
            apcp_1h_plane(hour),
        );
        let apcp_total = field(
            FieldSelector::surface(CanonicalField::TotalPrecipitation),
            "kg/m^2",
            apcp_total_plane(hour),
        );
        let uh = field(
            FieldSelector::height_layer_agl(CanonicalField::UpdraftHelicity, 2000, 5000),
            "m^2/s^2",
            uh_plane(hour),
        );
        let uh_max = field(
            FieldSelector::height_layer_agl(CanonicalField::UpdraftHelicity, 2000, 5000),
            "m^2/s^2",
            uh_max_plane(hour),
        );
        let wind_max = field(
            FieldSelector::height_agl(CanonicalField::WindSpeed, 10),
            "m/s",
            wind_max_plane(hour),
        );
        let mut fields: Vec<(&str, &SelectedField2D)> = vec![
            ("temperature_2m", &temp),
            ("dewpoint_2m", &dewpoint),
            ("rh_2m", &rh),
            ("u_10m", &u10),
            ("v_10m", &v10),
            ("apcp_run_total", &apcp_total),
            ("apcp_1h", &apcp_1h),
            ("uh_2to5km", &uh),
            ("uh_2to5km_max_1h", &uh_max),
            ("wind_speed_10m_max_1h", &wind_max),
        ];
        fields.retain(|(name, _)| !skip_vars.contains(name));
        let vpd_values = vpd_plane(hour);
        let mut derived = Vec::new();
        if !skip_vars.contains(&"vpd_2m") {
            derived.push(DerivedFieldInput {
                name: "vpd_2m",
                units: "hPa",
                values: &vpd_values,
            });
        }
        write_hour_from_fields_with_derived(
            store_root,
            "hrrr",
            run,
            hour,
            &fields,
            &derived,
            &[],
            "windowed-store-test",
            1_780_000_000 + hour as u64,
        )
        .unwrap();
    }

    fn write_test_run(store_root: &Path, run: &str, hours: &[u16]) {
        for &hour in hours {
            write_test_hour(store_root, run, hour, &[]);
        }
    }

    /// Write one synthetic hour the way the wrfout import lane does: the
    /// run accumulation under WRF's Registry name `apcp` and NO hourly
    /// increment plane at all (`temperature_2m` carries the grid).
    fn write_wrfout_apcp_hour(store_root: &Path, run: &str, hour: u16, run_total: Vec<f32>) {
        let temp = field(
            FieldSelector::height_agl(CanonicalField::Temperature, 2),
            "K",
            temp_k_plane(hour),
        );
        let apcp = field(
            FieldSelector::surface(CanonicalField::TotalPrecipitation),
            "kg/m^2",
            run_total,
        );
        write_hour_from_fields_with_derived(
            store_root,
            "hrrr",
            run,
            hour,
            &[("temperature_2m", &temp), ("apcp", &apcp)],
            &[],
            &[],
            "windowed-store-test",
            1_780_000_000 + hour as u64,
        )
        .unwrap();
    }

    fn compute(
        store_root: &Path,
        run: &str,
        hours: &[u16],
        slugs: &[&str],
    ) -> WindowedStoreOutcome {
        compute_windowed_products(
            store_root,
            "hrrr",
            run,
            hours,
            &slugs.iter().map(|s| s.to_string()).collect::<Vec<_>>(),
        )
        .unwrap()
    }

    fn grid_named<'a>(outcome: &'a WindowedStoreOutcome, slug: &str) -> &'a WindowedGrid {
        outcome
            .grids
            .iter()
            .find(|grid| grid.slug == slug)
            .unwrap_or_else(|| panic!("'{slug}' must realize; blockers: {:?}", outcome.blockers))
    }

    fn blocker_reason<'a>(outcome: &'a WindowedStoreOutcome, slug: &str) -> &'a str {
        outcome
            .blockers
            .iter()
            .find(|(have, _)| have == slug)
            .map(|(_, reason)| reason.as_str())
            .unwrap_or_else(|| {
                panic!(
                    "'{slug}' must be blocked; realized: {:?}",
                    outcome.grids.iter().map(|g| &g.slug).collect::<Vec<_>>()
                )
            })
    }

    fn assert_values(grid: &WindowedGrid, expected: &[f64]) {
        assert_eq!(grid.values.len(), expected.len(), "{}: length", grid.slug);
        for (cell, (got, want)) in grid.values.iter().zip(expected).enumerate() {
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "{} cell {cell}: got {got}, want {want}",
                grid.slug
            );
        }
    }

    #[test]
    fn six_hour_store_realizes_direct_trailing_and_run_windows_exactly() {
        let dir = test_dir("six-hour");
        let hours: Vec<u16> = (1..=6).collect();
        write_test_run(&dir, "20260608_00z", &hours);
        let outcome = compute(
            &dir,
            "20260608_00z",
            &hours,
            &[
                "qpf_1h",
                "qpf_6h",
                "qpf_total",
                "uh_2to5km_1h_max",
                "uh_2to5km_3h_max",
                "uh_2to5km_run_max",
                "10m_wind_1h_max",
                "10m_wind_run_max",
                "qpf_12h",
                "2m_temp_0_24h_max",
            ],
        );
        assert_eq!(outcome.anchor_hour, 6);
        assert_eq!(outcome.grids.len(), 8);
        assert_eq!(outcome.blockers.len(), 2);

        // qpf_1h: the stored trailing 1 h accumulation at F006, mm -> in.
        let qpf_1h = grid_named(&outcome, "qpf_1h");
        let expected: Vec<f64> = apcp_1h_plane(6)
            .iter()
            .map(|&mm| f64::from(mm) / MM_PER_INCH)
            .collect();
        assert_values(qpf_1h, &expected);
        assert_eq!(qpf_1h.units, "in");
        assert_eq!(qpf_1h.hours_used, vec![6]);
        assert_eq!(qpf_1h.window_hours, Some(1));

        // qpf_6h: sum of the six stored hourly increments, THEN mm -> in.
        let qpf_6h = grid_named(&outcome, "qpf_6h");
        let expected: Vec<f64> = (0..CELLS)
            .map(|cell| {
                let mm: f64 = (1..=6)
                    .map(|hour| f64::from(apcp_1h_plane(hour)[cell]))
                    .sum();
                mm / MM_PER_INCH
            })
            .collect();
        assert_values(qpf_6h, &expected);
        assert_eq!(qpf_6h.hours_used, (1..=6).collect::<Vec<u16>>());
        assert_eq!(qpf_6h.title, "6-h QPF");

        // qpf_total: the stored run-total accumulation at F006 (direct).
        let qpf_total = grid_named(&outcome, "qpf_total");
        let expected: Vec<f64> = apcp_total_plane(6)
            .iter()
            .map(|&mm| f64::from(mm) / MM_PER_INCH)
            .collect();
        assert_values(qpf_total, &expected);
        assert_eq!(qpf_total.hours_used, vec![6]);
        assert_eq!(qpf_total.window_hours, None);

        // UH: direct F006 sub-hourly max plane; trailing-3 and run maxima
        // fold the stored uh_2to5km_max_1h planes (NOT the instantaneous
        // uh_2to5km fallback, whose values sit strictly below).
        let uh_1h = grid_named(&outcome, "uh_2to5km_1h_max");
        assert_values(
            uh_1h,
            &uh_max_plane(6)
                .iter()
                .map(|&v| f64::from(v))
                .collect::<Vec<_>>(),
        );
        assert_eq!(uh_1h.units, "m^2/s^2");
        assert!(
            uh_1h.strategy.contains("uh_2to5km_max_1h") && !uh_1h.strategy.contains("fallback"),
            "strategy must name the stored max field with no fallback note: {}",
            uh_1h.strategy
        );
        let uh_3h = grid_named(&outcome, "uh_2to5km_3h_max");
        let expected: Vec<f64> = (0..CELLS)
            .map(|cell| {
                (4..=6)
                    .map(|hour| f64::from(uh_max_plane(hour)[cell]))
                    .fold(f64::NEG_INFINITY, f64::max)
            })
            .collect();
        assert_values(uh_3h, &expected);
        assert_eq!(uh_3h.hours_used, vec![4, 5, 6]);
        assert_eq!(uh_3h.window_hours, Some(3));
        let uh_run = grid_named(&outcome, "uh_2to5km_run_max");
        let expected: Vec<f64> = (0..CELLS)
            .map(|cell| {
                (1..=6)
                    .map(|hour| f64::from(uh_max_plane(hour)[cell]))
                    .fold(f64::NEG_INFINITY, f64::max)
            })
            .collect();
        assert_values(uh_run, &expected);

        // Wind: the stored sub-hourly max speeds (m/s) fold, THEN -> knots.
        let wind_1h = grid_named(&outcome, "10m_wind_1h_max");
        let expected: Vec<f64> = wind_max_plane(6)
            .iter()
            .map(|&speed| f64::from(speed) * MS_TO_KT)
            .collect();
        assert_values(wind_1h, &expected);
        assert_eq!(wind_1h.units, "kt");
        assert!(
            wind_1h.strategy.contains("wind_speed_10m_max_1h")
                && !wind_1h.strategy.contains("fallback"),
            "strategy must name the stored max field with no fallback note: {}",
            wind_1h.strategy
        );
        let wind_run = grid_named(&outcome, "10m_wind_run_max");
        let expected: Vec<f64> = (0..CELLS)
            .map(|cell| {
                (1..=6)
                    .map(|hour| f64::from(wind_max_plane(hour)[cell]))
                    .fold(f64::NEG_INFINITY, f64::max)
                    * MS_TO_KT
            })
            .collect();
        assert_values(wind_run, &expected);

        // Window minimums block with the GRIB lane's reasons.
        assert!(blocker_reason(&outcome, "qpf_12h").contains(">= 12"));
        assert!(blocker_reason(&outcome, "2m_temp_0_24h_max").contains(">= 24"));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn snapshot_windows_reduce_max_min_range_exactly_over_24_hours() {
        let dir = test_dir("snapshot-24h");
        let hours: Vec<u16> = (1..=24).collect();
        write_test_run(&dir, "20260608_00z", &hours);
        let outcome = compute(
            &dir,
            "20260608_00z",
            &hours,
            &[
                "2m_temp_0_24h_max",
                "2m_temp_0_24h_min",
                "2m_temp_0_24h_range",
                "2m_rh_0_24h_max",
                "2m_vpd_0_24h_min",
                "2m_dewpoint_0_24h_range",
                "qpf_24h",
                "10m_wind_0_24h_max",
                "2m_temp_24_48h_max",
                "2m_temp_0_48h_range",
            ],
        );
        assert_eq!(outcome.anchor_hour, 24);
        assert_eq!(outcome.grids.len(), 8);
        assert_eq!(outcome.blockers.len(), 2);

        // Mirror the fold in f64: K -> degC per hour, then pointwise ops.
        let temp_c = |hour: u16, cell: usize| f64::from(temp_k_plane(hour)[cell]) - 273.15;
        let fold = |cell: usize, op: fn(f64, f64) -> f64, init: f64| {
            (1..=24).map(|hour| temp_c(hour, cell)).fold(init, op)
        };
        let max: Vec<f64> = (0..CELLS)
            .map(|cell| fold(cell, f64::max, f64::NEG_INFINITY))
            .collect();
        let min: Vec<f64> = (0..CELLS)
            .map(|cell| fold(cell, f64::min, f64::INFINITY))
            .collect();
        let range: Vec<f64> = max.iter().zip(&min).map(|(max, min)| max - min).collect();
        let temp_max = grid_named(&outcome, "2m_temp_0_24h_max");
        assert_values(temp_max, &max);
        assert_eq!(temp_max.units, "degC");
        assert_eq!(temp_max.hours_used, (1..=24).collect::<Vec<u16>>());
        assert_eq!(temp_max.window_hours, Some(24));
        assert_values(grid_named(&outcome, "2m_temp_0_24h_min"), &min);
        assert_values(grid_named(&outcome, "2m_temp_0_24h_range"), &range);

        // RH max: raw values cross 100 at late hours; the clamp must hold
        // the fold at exactly 100.
        let rh_max = grid_named(&outcome, "2m_rh_0_24h_max");
        let expected: Vec<f64> = (0..CELLS)
            .map(|cell| {
                (1..=24)
                    .map(|hour| f64::from(rh_plane(hour)[cell]).clamp(0.0, 100.0))
                    .fold(f64::NEG_INFINITY, f64::max)
            })
            .collect();
        assert_values(rh_max, &expected);
        assert!(rh_max.values.iter().all(|&v| v == 100.0));
        assert_eq!(rh_max.units, "%");

        // VPD min reads the ingest-computed derived grid (hPa, no convert).
        let vpd_min = grid_named(&outcome, "2m_vpd_0_24h_min");
        let expected: Vec<f64> = (0..CELLS)
            .map(|cell| {
                (1..=24)
                    .map(|hour| f64::from(vpd_plane(hour)[cell]))
                    .fold(f64::INFINITY, f64::min)
            })
            .collect();
        assert_values(vpd_min, &expected);
        assert_eq!(vpd_min.units, "hPa");

        // Dewpoint range: K -> degC per hour first (range is invariant to
        // the offset, but the fold path is the converted one).
        let dew_range = grid_named(&outcome, "2m_dewpoint_0_24h_range");
        let expected: Vec<f64> = (0..CELLS)
            .map(|cell| {
                let values = (1..=24).map(|hour| f64::from(dewpoint_k_plane(hour)[cell]) - 273.15);
                values.clone().fold(f64::NEG_INFINITY, f64::max)
                    - values.fold(f64::INFINITY, f64::min)
            })
            .collect();
        assert_values(dew_range, &expected);

        // qpf_24h sums all 24 stored hourly increments.
        let qpf_24h = grid_named(&outcome, "qpf_24h");
        let expected: Vec<f64> = (0..CELLS)
            .map(|cell| {
                let mm: f64 = (1..=24)
                    .map(|hour| f64::from(apcp_1h_plane(hour)[cell]))
                    .sum();
                mm / MM_PER_INCH
            })
            .collect();
        assert_values(qpf_24h, &expected);

        // 48 h windows block: only 24 hours are stored.
        assert!(blocker_reason(&outcome, "2m_temp_24_48h_max").contains(">= 48"));
        assert!(blocker_reason(&outcome, "2m_temp_0_48h_range").contains(">= 48"));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn gaps_block_windows_instead_of_silently_skipping() {
        let dir = test_dir("gaps");
        let hours: Vec<u16> = vec![1, 2, 4];
        write_test_run(&dir, "20260608_00z", &hours);
        let outcome = compute(
            &dir,
            "20260608_00z",
            &hours,
            &[
                "uh_2to5km_3h_max",
                "uh_2to5km_run_max",
                "10m_wind_run_max",
                "qpf_1h",
                "qpf_total",
                "uh_2to5km_1h_max",
            ],
        );
        assert_eq!(outcome.anchor_hour, 4);

        // The trailing 3 h window F002-F004 is missing F003: blocked, with
        // the gap named, never computed from the two present hours.
        let reason = blocker_reason(&outcome, "uh_2to5km_3h_max");
        assert!(reason.contains("F003"), "gap must be named: {reason}");
        assert!(
            reason.contains("never skipped"),
            "no-silent-gap contract must be stated: {reason}"
        );
        assert!(blocker_reason(&outcome, "uh_2to5km_run_max").contains("F003"));
        assert!(blocker_reason(&outcome, "10m_wind_run_max").contains("F003"));

        // Direct single-hour products at the anchor still realize.
        let qpf_1h = grid_named(&outcome, "qpf_1h");
        let expected: Vec<f64> = apcp_1h_plane(4)
            .iter()
            .map(|&mm| f64::from(mm) / MM_PER_INCH)
            .collect();
        assert_values(qpf_1h, &expected);
        assert_eq!(grid_named(&outcome, "uh_2to5km_1h_max").hours_used, vec![4]);
        assert!(outcome.grids.iter().any(|grid| grid.slug == "qpf_total"));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn missing_variables_block_only_the_products_that_need_them() {
        let dir = test_dir("missing-vars");
        write_test_hour(&dir, "20260608_00z", 1, &[]);
        write_test_hour(&dir, "20260608_00z", 2, &["uh_2to5km_max_1h", "uh_2to5km"]);
        write_test_hour(&dir, "20260608_00z", 3, &["wind_speed_10m_max_1h", "v_10m"]);
        let outcome = compute(
            &dir,
            "20260608_00z",
            &[1, 2, 3],
            &[
                "uh_2to5km_3h_max",
                "uh_2to5km_1h_max",
                "10m_wind_1h_max",
                "qpf_1h",
            ],
        );

        // F002 lacks both the max field AND the instantaneous fallback:
        // the 3 h window dies with both variables and the hour named; the
        // 1 h product (F003 only) still realizes.
        let reason = blocker_reason(&outcome, "uh_2to5km_3h_max");
        assert!(
            reason.contains("uh_2to5km_max_1h")
                && reason.contains("uh_2to5km")
                && reason.contains("F002"),
            "reason must name both variables and the hour: {reason}"
        );
        assert!(outcome.grids.iter().any(|g| g.slug == "uh_2to5km_1h_max"));

        // F003 lacks the wind max field and v_10m: the wind speed product
        // blocks naming the failed fallback input.
        let reason = blocker_reason(&outcome, "10m_wind_1h_max");
        assert!(
            reason.contains("wind_speed_10m_max_1h") && reason.contains("v_10m"),
            "{reason}"
        );

        // Unrelated products are untouched.
        assert!(outcome.grids.iter().any(|g| g.slug == "qpf_1h"));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn missing_max_fields_fall_back_to_instantaneous_with_lower_bound_note() {
        let dir = test_dir("fallback");
        for hour in 1..=3 {
            write_test_hour(
                &dir,
                "20260608_00z",
                hour,
                &["uh_2to5km_max_1h", "wind_speed_10m_max_1h"],
            );
        }
        let outcome = compute(
            &dir,
            "20260608_00z",
            &[1, 2, 3],
            &["uh_2to5km_3h_max", "10m_wind_run_max", "qpf_1h"],
        );

        // UH folds the instantaneous uh_2to5km planes and says so.
        let uh_3h = grid_named(&outcome, "uh_2to5km_3h_max");
        let expected: Vec<f64> = (0..CELLS)
            .map(|cell| {
                (1..=3)
                    .map(|hour| f64::from(uh_plane(hour)[cell]))
                    .fold(f64::NEG_INFINITY, f64::max)
            })
            .collect();
        assert_values(uh_3h, &expected);
        assert!(
            uh_3h.strategy.contains("F001, F002, F003") && uh_3h.strategy.contains("lower bound"),
            "strategy must name every fallback hour and the lower-bound caveat: {}",
            uh_3h.strategy
        );
        // The lower-bound caveat belongs to THIS genuinely instantaneous
        // plane only; the exact-semantics UP_HELI_MAX note must not
        // appear (that plane is not in this store).
        assert!(
            !uh_3h.strategy.contains("UP_HELI_MAX"),
            "{}",
            uh_3h.strategy
        );

        // Wind folds hypot(u_10m, v_10m) and says so.
        let wind_run = grid_named(&outcome, "10m_wind_run_max");
        let expected: Vec<f64> = (0..CELLS)
            .map(|cell| {
                (1..=3)
                    .map(|hour| {
                        let (u, v) = wind_uv_planes(hour);
                        f64::from(u[cell]).hypot(f64::from(v[cell]))
                    })
                    .fold(f64::NEG_INFINITY, f64::max)
                    * MS_TO_KT
            })
            .collect();
        assert_values(wind_run, &expected);
        assert!(
            wind_run.strategy.contains("lower bound"),
            "{}",
            wind_run.strategy
        );

        // Products that never touch the max fields carry no note.
        let qpf_1h = grid_named(&outcome, "qpf_1h");
        assert!(!qpf_1h.strategy.contains("fallback"), "{}", qpf_1h.strategy);

        let _ = fs::remove_dir_all(&dir);
    }

    /// An hourly history with no stored 10 m wind maximum holds one
    /// top-of-hour speed per hour, and a picture of their largest must not
    /// be titled a maximum.  The catalog row names the fold, the frames and
    /// why; a stored maximum and a sub-hourly history keep the product title.
    #[test]
    fn hourly_wind_snapshots_are_titled_as_snapshots_and_maxima_keep_their_title() {
        let slugs = [
            "10m_wind_1h_max",
            "10m_wind_run_max",
            "10m_wind_0_24h_max",
            "uh_2to5km_run_max",
            "2m_temp_0_24h_max",
        ];
        let snapshots = test_dir("wind-snapshots");
        let hours: Vec<u16> = (1..=24).collect();
        for &hour in &hours {
            write_test_hour(&snapshots, "20260608_00z", hour, &["wind_speed_10m_max_1h"]);
        }
        let outcome = compute(&snapshots, "20260608_00z", &hours, &slugs);
        for (slug, title) in [
            ("10m_wind_1h_max", "10 m Wind Speed (hourly snapshot, no stored 1 h max)"),
            ("10m_wind_run_max", "10 m Wind Speed (largest hourly snapshot, no stored max)"),
            (
                "10m_wind_0_24h_max",
                "10 m Wind Speed (largest hourly snapshot 0-24 h, no stored max)",
            ),
        ] {
            let grid = grid_named(&outcome, slug);
            assert_eq!(grid.title, title);
            let product = HrrrWindowedProduct::from_slug(slug).unwrap();
            assert_ne!(grid.title, product.title(), "{slug}");
            assert!(
                grid.strategy.contains("F024")
                    && grid.strategy.contains("hypot(u_10m, v_10m)")
                    && grid.strategy.contains("WSPD10MAX")
                    && grid.strategy.contains("not the maximum"),
                "{slug}: {}",
                grid.strategy
            );
            assert!(!grid.strategy.contains("stored sub-hourly 1 h max"), "{}", grid.strategy);
        }
        let frames = hours
            .iter()
            .map(|hour| format!("F{hour:03}"))
            .collect::<Vec<_>>()
            .join(", ");
        assert_eq!(
            grid_named(&outcome, "10m_wind_run_max").strategy,
            format!(
                "the largest top-of-hour 10 m wind speed, hypot(u_10m, v_10m), over the run at \
                 {frames} (the history stores no sub-hourly 10 m wind maximum there (neither \
                 wind_speed_10m_max_1h nor WRF WSPD10MAX), so the wind between its hourly \
                 writes is not seen: a lower bound on the window's maximum, not the maximum)"
            )
        );
        // Stored maxima and snapshot statistics are what their titles say.
        for slug in ["uh_2to5km_run_max", "2m_temp_0_24h_max"] {
            let grid = grid_named(&outcome, slug);
            let product = HrrrWindowedProduct::from_slug(slug).unwrap();
            assert_eq!(grid.title, product.title(), "{slug}");
        }

        // The stored sub-hourly maximum is a maximum, and keeps its title.
        let maxima = test_dir("wind-maxima");
        write_test_run(&maxima, "20260608_00z", &hours);
        let outcome = compute(&maxima, "20260608_00z", &hours, &slugs);
        for slug in ["10m_wind_1h_max", "10m_wind_run_max", "10m_wind_0_24h_max"] {
            let grid = grid_named(&outcome, slug);
            let product = HrrrWindowedProduct::from_slug(slug).unwrap();
            assert_eq!(grid.title, product.title(), "{slug}");
            assert!(!grid.strategy.contains("snapshot"), "{slug}: {}", grid.strategy);
        }

        // A sub-hourly history folds every instant inside the window and
        // keeps the product title and its lower-bound note.
        let exact = test_dir("wind-exact");
        let run = "quarter_hour_title";
        for (slot, lead) in [(0u16, 0u64), (1, 15), (2, 30), (3, 45), (4, 60)] {
            write_exact_frame(&exact, run, slot, lead);
        }
        let outcome = compute(&exact, run, &[0, 1, 2, 3, 4], &["10m_wind_1h_max"]);
        let wind = grid_named(&outcome, "10m_wind_1h_max");
        assert_eq!(wind.title, "10 m Wind Speed (1 h max)");
        assert!(wind.strategy.contains("lower bound"), "{}", wind.strategy);

        for dir in [snapshots, maxima, exact] {
            let _ = fs::remove_dir_all(&dir);
        }
    }

    /// One wrfout-lane hour: the 10 m wind components, and WRF WSPD10MAX
    /// as the import stores it (the raw extra `wrf_wspd10max`, in the
    /// units the file declares) unless `wspd10max_units` is `None`.
    fn write_wrfout_wind_hour(
        store_root: &Path,
        run: &str,
        hour: u16,
        wspd10max_units: Option<&str>,
    ) {
        let temp = field(
            FieldSelector::height_agl(CanonicalField::Temperature, 2),
            "K",
            temp_k_plane(hour),
        );
        let (u_values, v_values) = wind_uv_planes(hour);
        let u10 = field(FieldSelector::height_agl(CanonicalField::UWind, 10), "m/s", u_values);
        let v10 = field(FieldSelector::height_agl(CanonicalField::VWind, 10), "m/s", v_values);
        let wspd10max = wind_max_plane(hour);
        let derived: Vec<DerivedFieldInput> = wspd10max_units
            .map(|units| DerivedFieldInput {
                name: "wrf_wspd10max",
                units,
                values: &wspd10max,
            })
            .into_iter()
            .collect();
        write_hour_from_fields_with_derived(
            store_root,
            "hrrr",
            run,
            hour,
            &[("temperature_2m", &temp), ("u_10m", &u10), ("v_10m", &v10)],
            &derived,
            &[],
            "windowed-store-test",
            1_780_000_000 + hour as u64,
        )
        .unwrap();
    }

    fn knots(values: impl Iterator<Item = f64>) -> Vec<f64> {
        values.map(|speed| speed * MS_TO_KT).collect()
    }

    /// A WRF history with nwp_diagnostics = 1 stores WSPD10MAX, the 10 m
    /// wind maximum over each history interval, which the wrfout import
    /// keeps as `wrf_wspd10max`.  The windows read it as a maximum, the
    /// same meaning as UP_HELI_MAX, and keep their maximum titles; they
    /// were drawn from top-of-hour U10/V10 speeds as "no stored max"
    /// snapshots of a history that does store it.
    #[test]
    fn a_wrfout_wspd10max_is_read_as_the_wind_maximum_and_keeps_the_max_titles() {
        let dir = test_dir("wrfout-wspd10max");
        let run = "20260608_00z";
        let hours: Vec<u16> = (1..=24).collect();
        for &hour in &hours {
            // WRF's Registry spelling, and the lane's own.
            let units = if hour % 2 == 1 { "m s-1" } else { "m/s" };
            write_wrfout_wind_hour(&dir, run, hour, Some(units));
        }
        let slugs = ["10m_wind_1h_max", "10m_wind_run_max", "10m_wind_0_24h_max"];
        let outcome = compute(&dir, run, &hours, &slugs);
        assert!(outcome.blockers.is_empty(), "{:?}", outcome.blockers);
        for slug in slugs {
            let grid = grid_named(&outcome, slug);
            let product = HrrrWindowedProduct::from_slug(slug).unwrap();
            assert_eq!(grid.title, product.title(), "{slug}");
            assert!(
                grid.strategy.contains("WRF WSPD10MAX per-history-interval max at")
                    && grid.strategy.contains("exact trailing 1 h max"),
                "{slug}: {}",
                grid.strategy
            );
            for wrong in ["snapshot", "lower bound", "hypot", "UP_HELI_MAX", "not the maximum"] {
                assert!(!grid.strategy.contains(wrong), "{slug} says {wrong:?}: {}", grid.strategy);
            }
        }
        let run_max = grid_named(&outcome, "10m_wind_run_max");
        let expected = knots((0..CELLS).map(|cell| {
            hours
                .iter()
                .map(|&hour| f64::from(wind_max_plane(hour)[cell]))
                .fold(f64::NEG_INFINITY, f64::max)
        }));
        assert_values(run_max, &expected);
        assert!(run_max.strategy.contains("F001, F002") && run_max.strategy.contains("F024"));
        let one_hour = grid_named(&outcome, "10m_wind_1h_max");
        let expected = knots(wind_max_plane(24).into_iter().map(f64::from));
        assert_values(one_hour, &expected);

        let _ = fs::remove_dir_all(&dir);
    }

    /// A stored WSPD10MAX in units the lane does not read blocks the window
    /// with the reason, as a native max field does, and is never replaced
    /// by the top-of-hour speed without a word.
    #[test]
    fn a_wspd10max_plane_in_other_units_blocks_instead_of_falling_back() {
        let dir = test_dir("wrfout-wspd10max-units");
        let run = "20260608_00z";
        for hour in 1..=3u16 {
            let units = if hour == 2 { "kt" } else { "m s-1" };
            write_wrfout_wind_hour(&dir, run, hour, Some(units));
        }
        let outcome = compute(&dir, run, &[1, 2, 3], &["10m_wind_run_max"]);
        let reason = blocker_reason(&outcome, "10m_wind_run_max");
        assert!(
            reason.contains("'wrf_wspd10max'") && reason.contains("'kt'"),
            "{reason}"
        );

        let _ = fs::remove_dir_all(&dir);
    }

    /// A window whose hours read a stored maximum at some hours and a
    /// top-of-hour snapshot at the others is neither a maximum nor wholly
    /// snapshots, and is titled as partly snapshots; the note names the
    /// snapshot hours and the maximum the others read.
    #[test]
    fn a_window_partly_read_from_a_stored_maximum_is_titled_partly_snapshots() {
        // WSPD10MAX at F001 and F003, none at F002.
        let wrfout = test_dir("wspd10max-partial");
        let run = "20260608_00z";
        for hour in 1..=3u16 {
            write_wrfout_wind_hour(&wrfout, run, hour, (hour != 2).then_some("m s-1"));
        }
        let outcome = compute(&wrfout, run, &[1, 2, 3], &["10m_wind_run_max"]);
        let grid = grid_named(&outcome, "10m_wind_run_max");
        assert_eq!(grid.title, "10 m Wind Speed (run max, partly hourly snapshots)");
        assert!(
            grid.strategy.starts_with(
                "the largest top-of-hour 10 m wind speed, hypot(u_10m, v_10m), over the run at \
                 F002 ("
            ) && grid.strategy.contains(
                "; the other hours read the stored sub-hourly maximum (WRF WSPD10MAX \
                 per-history-interval max at F001, F003:"
            ),
            "{}",
            grid.strategy
        );
        let expected = knots((0..CELLS).map(|cell| {
            let (u, v) = wind_uv_planes(2);
            f64::from(wind_max_plane(1)[cell])
                .max(f64::from(u[cell]).hypot(f64::from(v[cell])))
                .max(f64::from(wind_max_plane(3)[cell]))
        }));
        assert_values(grid, &expected);

        // The GRIB lane's native maximum at F001 and F003, none at F002.
        let grib = test_dir("wind-max-partial");
        write_test_hour(&grib, run, 1, &[]);
        write_test_hour(&grib, run, 2, &["wind_speed_10m_max_1h"]);
        write_test_hour(&grib, run, 3, &[]);
        let outcome = compute(&grib, run, &[1, 2, 3], &["10m_wind_run_max"]);
        let grid = grid_named(&outcome, "10m_wind_run_max");
        assert_eq!(grid.title, "10 m Wind Speed (run max, partly hourly snapshots)");
        assert!(
            grid.strategy.contains("at F002 (")
                && grid.strategy.ends_with("; the other hours read the stored sub-hourly maximum"),
            "{}",
            grid.strategy
        );

        for dir in [wrfout, grib] {
            let _ = fs::remove_dir_all(&dir);
        }
    }

    #[test]
    fn mixed_stores_fall_back_only_for_hours_missing_the_max_field() {
        let dir = test_dir("mixed-fallback");
        write_test_hour(&dir, "20260608_00z", 1, &[]);
        write_test_hour(&dir, "20260608_00z", 2, &["uh_2to5km_max_1h"]);
        write_test_hour(&dir, "20260608_00z", 3, &[]);
        let outcome = compute(&dir, "20260608_00z", &[1, 2, 3], &["uh_2to5km_3h_max"]);

        // F001/F003 fold the stored max planes; F002 folds the
        // instantaneous fallback plane.
        let uh_3h = grid_named(&outcome, "uh_2to5km_3h_max");
        let expected: Vec<f64> = (0..CELLS)
            .map(|cell| {
                f64::from(uh_max_plane(1)[cell])
                    .max(f64::from(uh_plane(2)[cell]))
                    .max(f64::from(uh_max_plane(3)[cell]))
            })
            .collect();
        assert_values(uh_3h, &expected);
        assert!(
            uh_3h.strategy.contains("F002")
                && !uh_3h.strategy.contains("F001")
                && !uh_3h.strategy.contains("F003"),
            "the note must name exactly the fallback hour: {}",
            uh_3h.strategy
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn wrfout_up_heli_max_planes_are_labeled_exact_not_lower_bound() {
        let dir = test_dir("wrfout-uh");
        for hour in 1..=3u16 {
            let temp = field(
                FieldSelector::height_agl(CanonicalField::Temperature, 2),
                "K",
                temp_k_plane(hour),
            );
            // WRF Registry units spelling, as the wrfout import stamps it.
            let uh = field(
                FieldSelector::height_layer_agl(CanonicalField::UpdraftHelicity, 2000, 5000),
                "m2/s2",
                uh_plane(hour),
            );
            write_hour_from_fields_with_derived(
                &dir,
                "hrrr",
                "20260608_00z",
                hour,
                &[("temperature_2m", &temp), ("updraft_helicity_2to5km", &uh)],
                &[],
                &[],
                "windowed-store-test",
                1_780_000_000 + hour as u64,
            )
            .unwrap();
        }
        let outcome = compute(
            &dir,
            "20260608_00z",
            &[1, 2, 3],
            &["uh_2to5km_3h_max", "uh_2to5km_1h_max"],
        );

        // The fold is the exact max of the per-history-interval maxima.
        let uh_3h = grid_named(&outcome, "uh_2to5km_3h_max");
        let expected: Vec<f64> = (0..CELLS)
            .map(|cell| {
                (1..=3)
                    .map(|hour| f64::from(uh_plane(hour)[cell]))
                    .fold(f64::NEG_INFINITY, f64::max)
            })
            .collect();
        assert_values(uh_3h, &expected);

        // Corrected contract: gpuwm/WRF UP_HELI_MAX is reset at every
        // history write, so each plane is the exact max over its
        // trailing history interval -- the exact 1 h max at this
        // lane's hourly cadence.  The lower-bound wording is reserved
        // for genuinely instantaneous planes and must NOT appear here.
        for slug in ["uh_2to5km_3h_max", "uh_2to5km_1h_max"] {
            let grid = grid_named(&outcome, slug);
            assert!(
                grid.strategy.contains("UP_HELI_MAX")
                    && grid.strategy.contains("exact trailing 1 h max"),
                "{slug}: {}",
                grid.strategy
            );
            assert!(
                !grid.strategy.contains("lower bound")
                    && !grid.strategy.contains("instantaneous"),
                "{slug}: {}",
                grid.strategy
            );
        }
        assert!(
            uh_3h.strategy.contains("F001, F002, F003"),
            "every UP_HELI_MAX hour must be named: {}",
            uh_3h.strategy
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn wrfout_run_total_store_differences_the_1h_qpf_window() {
        let dir = test_dir("wrfout-qpf-1h");
        let run = "20260608_00z";
        for hour in 0..=3u16 {
            write_wrfout_apcp_hour(&dir, run, hour, apcp_run_accum_plane(hour));
        }
        // F003 is stored but not offered: the anchor is F002, and the
        // plan must reach BACKWARD to F001 rather than forward.
        let outcome = compute(&dir, run, &[0, 1, 2], &["qpf_1h", "qpf_total"]);
        assert_eq!(outcome.anchor_hour, 2);
        assert!(outcome.blockers.is_empty(), "{:?}", outcome.blockers);

        let qpf_1h = grid_named(&outcome, "qpf_1h");
        let expected: Vec<f64> = (0..CELLS)
            .map(|cell| {
                (f64::from(apcp_run_accum_plane(2)[cell])
                    - f64::from(apcp_run_accum_plane(1)[cell]))
                    / MM_PER_INCH
            })
            .collect();
        assert_values(qpf_1h, &expected);
        assert_eq!(qpf_1h.units, "in");
        assert_eq!(qpf_1h.window_hours, Some(1));
        assert_eq!(qpf_1h.hours_used, vec![1, 2]);
        // Pinned verbatim: the note is the only thing that lets a reader
        // of the plot tell a differenced window from a native one, so it
        // must name the fallback AND the exact pair of hours subtracted.
        assert_eq!(
            qpf_1h.strategy,
            "1 h APCP differenced from stored run-total accumulations (F002 minus F001); \
             this store carries no native apcp_1h plane"
        );

        // The run total itself still reads whole, through the same
        // wrfout-lane `apcp` plane.
        let qpf_total = grid_named(&outcome, "qpf_total");
        let expected: Vec<f64> = apcp_run_accum_plane(2)
            .iter()
            .map(|&mm| f64::from(mm) / MM_PER_INCH)
            .collect();
        assert_values(qpf_total, &expected);

        let _ = fs::remove_dir_all(&dir);
    }

    /// WHAT BREAKAGE THIS PREVENTS (gate law): a moving nest's earlier
    /// frame, moved onto the nest's later place, holds NaN on the ground the
    /// nest had not reached yet.  `NaN.max(0.0)` is 0.0 and `f64::max`
    /// returns its other operand, so the difference drew that ground as a
    /// dry hour and a maximum folded it from the frames after the nest
    /// arrived, drawn as the whole window.  Missing stays missing.
    #[test]
    fn a_cell_missing_from_one_frame_is_missing_from_its_window() {
        let dir = test_dir("missing-cell");
        let run = "20260930_00z";
        for hour in 0..=2u16 {
            let mut total = apcp_run_accum_plane(hour);
            if hour == 1 {
                // New ground on the nest's leading edge: no F001 value.
                total[0] = f32::NAN;
            }
            write_wrfout_apcp_hour(&dir, run, hour, total);
        }
        let outcome = compute(&dir, run, &[0, 1, 2], &["qpf_1h", "qpf_total"]);
        let qpf_1h = grid_named(&outcome, "qpf_1h");
        assert!(qpf_1h.values[0].is_nan(), "{:?}", qpf_1h.values);
        for cell in 1..CELLS {
            let want = (f64::from(apcp_run_accum_plane(2)[cell])
                - f64::from(apcp_run_accum_plane(1)[cell]))
                / MM_PER_INCH;
            assert_eq!(qpf_1h.values[cell].to_bits(), want.to_bits(), "cell {cell}");
        }
        // The run total at F002 has its own value there and keeps it.
        assert!(grid_named(&outcome, "qpf_total").values.iter().all(|v| v.is_finite()));
        let _ = fs::remove_dir_all(&dir);

        let dir = test_dir("missing-cell-max");
        for hour in 1..=3u16 {
            let temp = field(
                FieldSelector::height_agl(CanonicalField::Temperature, 2),
                "K",
                temp_k_plane(hour),
            );
            let mut values = uh_plane(hour);
            if hour == 1 {
                values[1] = f32::NAN;
            }
            let uh = field(
                FieldSelector::height_layer_agl(CanonicalField::UpdraftHelicity, 2000, 5000),
                "m2/s2",
                values,
            );
            write_hour_from_fields_with_derived(
                &dir,
                "hrrr",
                run,
                hour,
                &[("temperature_2m", &temp), ("updraft_helicity_2to5km", &uh)],
                &[],
                &[],
                "windowed-store-test",
                1_780_000_000 + hour as u64,
            )
            .unwrap();
        }
        let outcome = compute(&dir, run, &[1, 2, 3], &["uh_2to5km_3h_max"]);
        let uh = grid_named(&outcome, "uh_2to5km_3h_max");
        assert!(uh.values[1].is_nan(), "{:?}", uh.values);
        for cell in [0usize, 2, 3] {
            let want = (1..=3)
                .map(|hour| f64::from(uh_plane(hour)[cell]))
                .fold(f64::NEG_INFINITY, f64::max);
            assert_eq!(uh.values[cell].to_bits(), want.to_bits(), "cell {cell}");
        }
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn differenced_qpf_is_the_increment_never_the_run_total() {
        // Negative control for the differencing fallback: it fails if
        // the implementation ever publishes apcp(h) where the 1 h window
        // wants apcp(h) - apcp(h-1).
        let dir = test_dir("wrfout-qpf-negative-control");
        let run = "20260608_00z";
        for hour in 0..=2u16 {
            write_wrfout_apcp_hour(&dir, run, hour, apcp_run_accum_plane(hour));
        }
        let outcome = compute(&dir, run, &[0, 1, 2], &["qpf_1h", "qpf_total"]);
        let qpf_1h = grid_named(&outcome, "qpf_1h");
        let qpf_total = grid_named(&outcome, "qpf_total");

        // Guard the fixture: an increment that happened to equal the run
        // total would make every assertion below pass vacuously.
        for cell in 0..CELLS {
            let increment =
                f64::from(apcp_run_accum_plane(2)[cell]) - f64::from(apcp_run_accum_plane(1)[cell]);
            let total = f64::from(apcp_run_accum_plane(2)[cell]);
            assert!(
                increment > 0.0 && total - increment > 1.0,
                "fixture cell {cell}: increment {increment} must be positive and far \
                 below the run total {total}"
            );
        }

        // Pinned numerically, independent of the fixture function: the
        // F001->F002 increments are 0.75/1.0/1.25/1.5 mm, while the run
        // totals at F002 are 3.5/4.0/4.5/5.0 mm.
        let expected: Vec<f64> = [0.75_f64, 1.0, 1.25, 1.5]
            .iter()
            .map(|mm| mm / MM_PER_INCH)
            .collect();
        assert_values(qpf_1h, &expected);
        for (cell, (hourly, total)) in qpf_1h.values.iter().zip(&qpf_total.values).enumerate() {
            assert!(
                hourly < total,
                "cell {cell}: qpf_1h {hourly} is the F001->F002 increment and must sit \
                 strictly below the run total {total}"
            );
        }

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_three_hourly_history_does_not_apply_hourly_windows_and_keeps_its_gaps() {
        // A nest keeping 3-hourly history: qpf_1h needs F005, which this
        // history never writes, so it is not applicable rather than a gap;
        // qpf_6h falls on the interval and is drawn.
        let dir = test_dir("wrfout-qpf-3hourly");
        let run = "20260608_00z";
        for hour in [0u16, 3, 6] {
            write_wrfout_apcp_hour(&dir, run, hour, apcp_run_accum_plane(hour));
        }
        let outcome = compute(&dir, run, &[0, 3, 6], &["qpf_1h", "qpf_6h"]);
        let reason = blocker_reason(&outcome, "qpf_1h");
        assert!(reason.starts_with(NOT_APPLICABLE_PREFIX), "{reason}");
        assert!(reason.contains("every 3 h") && reason.contains("F005"), "{reason}");
        assert_eq!(grid_named(&outcome, "qpf_6h").hours_used, vec![0, 6]);
        let _ = fs::remove_dir_all(&dir);

        // A frame missing ON the interval is a gap and stays one.
        let dir = test_dir("wrfout-qpf-3hourly-gap");
        for hour in [0u16, 3, 9, 12] {
            write_wrfout_apcp_hour(&dir, run, hour, apcp_run_accum_plane(hour));
        }
        let outcome = compute(&dir, run, &[0, 3, 9, 12], &["qpf_6h"]);
        let reason = blocker_reason(&outcome, "qpf_6h");
        assert!(!reason.starts_with(NOT_APPLICABLE_PREFIX), "{reason}");
        assert!(reason.contains("F006") && reason.contains("never skipped"), "{reason}");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn wrfout_run_total_store_differences_only_the_6h_window_endpoints() {
        let dir = test_dir("wrfout-qpf-6h");
        let run = "20260608_00z";
        for hour in 0..=6u16 {
            write_wrfout_apcp_hour(&dir, run, hour, apcp_run_accum_plane(hour));
        }
        let hours: Vec<u16> = (0..=6).collect();
        let outcome = compute(&dir, run, &hours, &["qpf_6h", "qpf_12h"]);

        // Six hours of accumulation = 6 * the per-cell step, reached
        // with two plane reads instead of seven.
        let qpf_6h = grid_named(&outcome, "qpf_6h");
        let expected: Vec<f64> = [4.5_f64, 6.0, 7.5, 9.0]
            .iter()
            .map(|mm| mm / MM_PER_INCH)
            .collect();
        assert_values(qpf_6h, &expected);
        assert_eq!(
            qpf_6h.hours_used,
            vec![0, 6],
            "only the window endpoints are read"
        );
        assert_eq!(qpf_6h.window_hours, Some(6));
        assert_eq!(
            qpf_6h.strategy,
            "6 h APCP differenced from stored run-total accumulations (F006 minus F000); \
             this store carries no native apcp_1h plane"
        );

        // The window minimum still governs: no predecessor, no window.
        assert!(blocker_reason(&outcome, "qpf_12h").contains(">= 12"));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn missing_predecessor_hour_blocks_the_differenced_window() {
        let dir = test_dir("wrfout-qpf-gap");
        let run = "20260608_00z";
        for hour in [0u16, 1, 3] {
            write_wrfout_apcp_hour(&dir, run, hour, apcp_run_accum_plane(hour));
        }
        let outcome = compute(&dir, run, &[0, 1, 3], &["qpf_1h", "qpf_total"]);
        assert_eq!(outcome.anchor_hour, 3);

        // F002 is the anchor's predecessor and it is not stored: the 1 h
        // window blocks naming it, never publishing a zero or reaching
        // further back for a wider increment wearing a "1 h" label.
        let reason = blocker_reason(&outcome, "qpf_1h");
        assert!(
            reason.contains("F002"),
            "the missing predecessor hour must be named: {reason}"
        );
        assert!(
            reason.contains("never skipped"),
            "no-silent-gap contract must be stated: {reason}"
        );

        // The gap does not touch the run total at the anchor.
        assert!(outcome.grids.iter().any(|grid| grid.slug == "qpf_total"));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn native_hourly_apcp_is_never_preempted_by_the_differenced_fallback() {
        let dir = test_dir("native-not-preempted");
        let run = "20260608_00z";
        // write_test_run stores BOTH apcp_1h and apcp_run_total, the
        // GRIB ingest lane's shape.
        write_test_run(&dir, run, &[1, 2, 3]);
        let outcome = compute(&dir, run, &[1, 2, 3], &["qpf_1h"]);

        let qpf_1h = grid_named(&outcome, "qpf_1h");
        let expected: Vec<f64> = apcp_1h_plane(3)
            .iter()
            .map(|&mm| f64::from(mm) / MM_PER_INCH)
            .collect();
        assert_values(qpf_1h, &expected);
        assert_eq!(qpf_1h.hours_used, vec![3], "one hour, no predecessor read");
        assert!(
            qpf_1h.strategy.contains("apcp_1h") && !qpf_1h.strategy.contains("differenced"),
            "{}",
            qpf_1h.strategy
        );

        // Not a cosmetic assertion: in this fixture the measured hourly
        // increment and a difference of the stored run totals are
        // different numbers, so a fallback that preempted the native
        // plane would land somewhere else.
        let differenced = f64::from(apcp_total_plane(3)[0]) - f64::from(apcp_total_plane(2)[0]);
        assert!(
            (differenced - f64::from(apcp_1h_plane(3)[0])).abs() > 0.1,
            "fixture must separate the two answers"
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_half_folded_difference_blocks_instead_of_publishing_the_run_total() {
        // The store-level gate normally makes this unreachable, but the
        // accumulator must not depend on that gate to stay accurate: with
        // only the earlier endpoint folded it holds a run accumulation,
        // and it must refuse rather than hand it out wearing a 1 h label.
        let spec = plan_product(
            HrrrWindowedProduct::Qpf1h,
            2,
            QpfSource::DifferencedRunTotal,
        )
        .unwrap();
        assert_eq!(spec.reduce, Reduce::Difference);
        assert_eq!(spec.hours, vec![1, 2]);

        let slots = spec.hours.clone();
        let mut accum = Accum::new(spec, slots);
        accum.fold(&[7.0, 8.0, 9.0, 10.0]);
        let reason = accum.finish(false).unwrap_err();
        assert!(
            reason.contains("F001") && reason.contains("F002") && reason.contains("undefined"),
            "the half-folded blocker must name both endpoints: {reason}"
        );
    }

    #[test]
    fn round_off_negative_run_total_steps_clamp_to_zero() {
        let dir = test_dir("wrfout-qpf-clamp");
        let run = "20260608_00z";
        // A run total that dips by one f32 ulp between hours: physically
        // impossible for this producer's monotonic RAINC/RAINNC (no
        // bucket reset), reachable only by round-off in the stored f32.
        let later = 1234.5_f32;
        let earlier = f32::from_bits(later.to_bits() + 1);
        assert!(
            f64::from(later) - f64::from(earlier) < 0.0,
            "the fixture must actually step backward"
        );
        write_wrfout_apcp_hour(&dir, run, 0, vec![0.0; CELLS]);
        write_wrfout_apcp_hour(&dir, run, 1, vec![earlier; CELLS]);
        write_wrfout_apcp_hour(&dir, run, 2, vec![later; CELLS]);

        let outcome = compute(&dir, run, &[0, 1, 2], &["qpf_1h"]);
        assert_values(grid_named(&outcome, "qpf_1h"), &[0.0; CELLS]);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn unexpected_stored_units_block_instead_of_converting_blindly() {
        let dir = test_dir("bad-units");
        // Hand-build an hour whose apcp_1h claims inches and whose UH max
        // field claims knots (beside a perfectly good instantaneous
        // uh_2to5km): the lane must refuse rather than divide by 25.4
        // again, and unit drift on the max field must block, NOT silently
        // fall back to the instantaneous plane.
        let temp = field(
            FieldSelector::height_agl(CanonicalField::Temperature, 2),
            "K",
            temp_k_plane(1),
        );
        let apcp_bad = field(
            FieldSelector::surface(CanonicalField::TotalPrecipitation),
            "in",
            apcp_1h_plane(1),
        );
        let uh = field(
            FieldSelector::height_layer_agl(CanonicalField::UpdraftHelicity, 2000, 5000),
            "m^2/s^2",
            uh_plane(1),
        );
        let uh_max_bad = field(
            FieldSelector::height_layer_agl(CanonicalField::UpdraftHelicity, 2000, 5000),
            "kt",
            uh_max_plane(1),
        );
        write_hour_from_fields_with_derived(
            &dir,
            "hrrr",
            "20260608_00z",
            1,
            &[
                ("temperature_2m", &temp),
                ("apcp_1h", &apcp_bad),
                ("uh_2to5km", &uh),
                ("uh_2to5km_max_1h", &uh_max_bad),
            ],
            &[],
            &[],
            "windowed-store-test",
            1_780_000_001,
        )
        .unwrap();
        let outcome = compute(&dir, "20260608_00z", &[1], &["qpf_1h", "uh_2to5km_1h_max"]);
        let reason = blocker_reason(&outcome, "qpf_1h");
        assert!(
            reason.contains("units 'in'") && reason.contains("kg/m^2"),
            "reason must name actual and expected units: {reason}"
        );
        let reason = blocker_reason(&outcome, "uh_2to5km_1h_max");
        assert!(
            reason.contains("units 'kt'") && reason.contains("m^2/s^2"),
            "unit drift on the max field must block, not fall back: {reason}"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn unknown_slugs_error_and_duplicates_dedupe() {
        let dir = test_dir("slugs");
        write_test_run(&dir, "20260608_00z", &[1]);
        let err = compute_windowed_products(
            &dir,
            "hrrr",
            "20260608_00z",
            &[1],
            &["not_a_windowed_product".to_string()],
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("not_a_windowed_product"), "{err}");

        let outcome = compute(&dir, "20260608_00z", &[1], &["qpf_1h", "qpf_1h", "qpf_1h"]);
        assert_eq!(outcome.grids.len(), 1, "duplicates must dedupe");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn stored_run_hours_reads_the_manifest() {
        let dir = test_dir("manifest");
        write_test_run(&dir, "20260608_00z", &[1, 2, 5]);
        let hours = stored_run_hours(&dir, "hrrr", "20260608_00z").unwrap();
        assert_eq!(hours, vec![1, 2, 5]);
        assert!(stored_run_hours(&dir, "hrrr", "20990101_00z").is_err());
        let _ = fs::remove_dir_all(&dir);
    }

    /// Run origin of the exact-time fixtures, on a whole hour.
    const EXACT_ORIGIN_UNIX: i64 = 1_779_998_400;

    /// WRF UP_HELI_MAX as a 15-minute history writes it: each frame holds
    /// the max over the interval since the previous history write, and the
    /// analysis frame carries a large value in cell 0 that lies OUTSIDE
    /// every window ending after it.  Indexed by lead in minutes.
    fn interval_uh_plane(lead_minutes: u64) -> Vec<f32> {
        match lead_minutes {
            0 => vec![99.0, 0.0, 0.0, 0.0],
            15 => vec![10.0, 1.0, 7.0, 0.0],
            30 => vec![3.0, 40.0, 2.0, 0.0],
            45 => vec![5.0, 2.0, 30.0, 0.0],
            60 => vec![4.0, 6.0, 1.0, 0.5],
            75 => vec![2.0, 3.0, 0.0, 9.0],
            90 => vec![1.0, 8.0, 4.0, 0.0],
            105 => vec![6.0, 0.0, 2.0, 1.0],
            120 => vec![0.0, 5.0, 3.0, 7.0],
            180 => vec![2.0, 1.0, 6.0, 4.0],
            other => panic!("no interval plane at +{other} min"),
        }
    }

    /// RAINNC + RAINC since the run's start as the wrfout import stores it
    /// (`apcp`, kg/m^2): 0.5 * (cell + 1) mm every 15 minutes on top of
    /// 1 mm at the start, so every value and every difference is exact.
    fn quarter_hour_run_total(lead_minutes: u64) -> Vec<f32> {
        (0..CELLS)
            .map(|cell| 1.0 + 0.5 * (cell + 1) as f32 * (lead_minutes / 15) as f32)
            .collect()
    }

    /// U10 as the wrfout import stores it, an instant (V10 is zero, so the
    /// speed is U10).  Strongest at +0:30, so a fold that misses the first
    /// hour reads low, and uneven inside every hour, so a fold that misses
    /// one frame of an hour does too.
    fn quarter_hour_u10(lead_minutes: u64) -> Vec<f32> {
        let gust = match lead_minutes {
            0 => 1.0,
            15 => 4.0,
            30 => 9.0,
            45 => 2.0,
            60 => 3.0,
            75 => 5.0,
            90 => 1.0,
            105 => 6.0,
            120 => 2.0,
            180 => 7.0,
            other => panic!("no 10 m wind at +{other} min"),
        };
        (0..CELLS).map(|cell| cell as f32 + gust).collect()
    }

    /// One wrfout-lane frame on the exact-time axis: a sub-hourly history
    /// forces it, and each ordinal slot carries its own lead.
    fn write_exact_frame(store_root: &Path, run: &str, slot: u16, lead_minutes: u64) {
        let temp = field(
            FieldSelector::height_agl(CanonicalField::Temperature, 2),
            "K",
            temp_k_plane(0),
        );
        let apcp = field(
            FieldSelector::surface(CanonicalField::TotalPrecipitation),
            "kg/m^2",
            quarter_hour_run_total(lead_minutes),
        );
        let uh = field(
            FieldSelector::height_layer_agl(CanonicalField::UpdraftHelicity, 2000, 5000),
            "m2/s2",
            interval_uh_plane(lead_minutes),
        );
        let u10 = field(
            FieldSelector::height_agl(CanonicalField::UWind, 10),
            "m/s",
            quarter_hour_u10(lead_minutes),
        );
        let v10 = field(
            FieldSelector::height_agl(CanonicalField::VWind, 10),
            "m/s",
            vec![0.0; CELLS],
        );
        let lead = lead_minutes * 60;
        write_hour_from_fields_with_derived_exact(
            store_root,
            "hrrr",
            run,
            slot,
            RwsExactTime::new(lead, EXACT_ORIGIN_UNIX + lead as i64),
            &[
                ("temperature_2m", &temp),
                ("apcp", &apcp),
                ("updraft_helicity_2to5km", &uh),
                ("u_10m", &u10),
                ("v_10m", &v10),
            ],
            &[],
            &[],
            "windowed-store-test",
            1_780_000_000 + u64::from(slot),
        )
        .unwrap();
    }

    #[test]
    fn exact_time_runs_serve_windows_from_exact_leads() {
        // A 15-minute history puts the store on the exact-time axis: slot 4
        // is the frame at +1 h, not forecast hour 4.  The 1 h window ending
        // there is served from the frames that bound and fill it.
        let dir = test_dir("exact-time-windows");
        let run = "quarter_hour";
        let leads = [0u64, 15, 30, 45, 60];
        for (slot, &lead) in leads.iter().enumerate() {
            write_exact_frame(&dir, run, slot as u16, lead);
        }

        let slots = stored_run_hours(&dir, "hrrr", run).unwrap();
        assert_eq!(slots, vec![0, 1, 2, 3, 4], "ordinal slots must be listable");

        let outcome = compute(
            &dir,
            run,
            &slots,
            &["qpf_1h", "uh_2to5km_1h_max", "qpf_total", "qpf_6h"],
        );
        assert_eq!(outcome.anchor_hour, 1, "the anchor is +1 h, not slot 4");
        assert!(windowed_axis_ready(&dir, "hrrr", run).unwrap());

        // Accumulation: the run totals at the two bounding whole-hour
        // frames, differenced.  2, 4, 6 and 8 mm.
        let qpf_1h = grid_named(&outcome, "qpf_1h");
        let expected: Vec<f64> = (0..CELLS)
            .map(|cell| {
                (f64::from(quarter_hour_run_total(60)[cell])
                    - f64::from(quarter_hour_run_total(0)[cell]))
                    / MM_PER_INCH
            })
            .collect();
        assert_values(qpf_1h, &expected);
        assert_eq!(qpf_1h.hours_used, vec![0, 1]);
        assert_eq!(qpf_1h.window_hours, Some(1));

        // Per-history-interval maxima: every frame whose interval lies
        // inside (0, 1 h] is folded.  Not the 60-minute plane alone (the
        // last quarter hour), and not the analysis frame, whose plane lies
        // before the window.
        let uh = grid_named(&outcome, "uh_2to5km_1h_max");
        let expected: Vec<f64> = (0..CELLS)
            .map(|cell| {
                [15u64, 30, 45, 60]
                    .iter()
                    .map(|&lead| f64::from(interval_uh_plane(lead)[cell]))
                    .fold(f64::NEG_INFINITY, f64::max)
            })
            .collect();
        assert_eq!(expected, vec![10.0, 40.0, 30.0, 0.5]);
        assert_values(uh, &expected);
        let last_quarter: Vec<f64> = interval_uh_plane(60)
            .iter()
            .map(|&v| f64::from(v))
            .collect();
        assert_ne!(uh.values, last_quarter, "the last interval alone reads low");
        assert!(
            uh.strategy.contains("+000:15, +000:30, +000:45, +001:00"),
            "every folded frame is named: {}",
            uh.strategy
        );
        assert!(
            !uh.strategy.contains("lower bound"),
            "a fold of every interval max is exact: {}",
            uh.strategy
        );

        // The run total at the anchor still reads whole.
        let total = grid_named(&outcome, "qpf_total");
        let expected: Vec<f64> = quarter_hour_run_total(60)
            .iter()
            .map(|&mm| f64::from(mm) / MM_PER_INCH)
            .collect();
        assert_values(total, &expected);
        assert!(blocker_reason(&outcome, "qpf_6h").contains(">= 6"));

        // A frame between whole hours closes no window, and says so.
        let between = compute(&dir, run, &[0, 1, 2], &["qpf_1h", "uh_2to5km_1h_max"]);
        for slug in ["qpf_1h", "uh_2to5km_1h_max"] {
            let reason = blocker_reason(&between, slug);
            assert!(
                reason.contains("whole forecast hours") && reason.contains("+000:30"),
                "{slug}: {reason}"
            );
        }
        let _ = fs::remove_dir_all(&dir);
    }

    /// The largest of `planes` cell by cell, as the fold takes it.
    fn cellwise_max(planes: &[Vec<f64>]) -> Vec<f64> {
        (0..CELLS)
            .map(|cell| {
                planes
                    .iter()
                    .map(|plane| plane[cell])
                    .fold(f64::NEG_INFINITY, f64::max)
            })
            .collect()
    }

    fn uh_at(leads: &[u64]) -> Vec<f64> {
        cellwise_max(
            &leads
                .iter()
                .map(|&lead| interval_uh_plane(lead).iter().map(|&v| f64::from(v)).collect())
                .collect::<Vec<_>>(),
        )
    }

    fn wind_kt_at(leads: &[u64]) -> Vec<f64> {
        cellwise_max(
            &leads
                .iter()
                .map(|&lead| {
                    quarter_hour_u10(lead)
                        .iter()
                        .map(|&u| f64::from(u).hypot(0.0))
                        .collect()
                })
                .collect::<Vec<_>>(),
        )
        .into_iter()
        .map(|speed| speed * MS_TO_KT)
        .collect()
    }

    #[test]
    fn exact_time_interval_maxima_block_on_a_missing_frame() {
        // The 30-minute frame was never stored: its interval max and its
        // 10 m wind instant are gone, so a fold of the rest would read low,
        // and the picture drawn from it is kept.  The wind is labelled a
        // lower bound, but a lower bound of fewer frames than the run
        // stored is still lower than the run's own.  The accumulation only
        // needs the two bounding frames and still draws.
        let dir = test_dir("exact-time-gap");
        let run = "quarter_hour_gap";
        for (slot, lead) in [(0u16, 0u64), (1, 15), (2, 45), (3, 60)] {
            write_exact_frame(&dir, run, slot, lead);
        }
        let outcome = compute(
            &dir,
            run,
            &[0, 1, 2, 3],
            &["qpf_1h", "uh_2to5km_1h_max", "10m_wind_1h_max"],
        );
        for slug in ["uh_2to5km_1h_max", "10m_wind_1h_max"] {
            let reason = blocker_reason(&outcome, slug);
            assert!(
                reason.contains("+000:15")
                    && reason.contains("+000:45")
                    && reason.contains("read low"),
                "{slug}: {reason}"
            );
        }
        assert!(outcome.grids.iter().any(|grid| grid.slug == "qpf_1h"));

        // A window that starts with the run needs no frame at its start:
        // no interval begins before the run, so the first stored frame's
        // maximum is already inside it, and no instant of the run comes
        // before it.  The rainfall difference still needs the run total
        // at F000.
        let dir2 = test_dir("exact-time-no-start");
        for (slot, lead) in [(0u16, 15u64), (1, 30), (2, 45), (3, 60)] {
            write_exact_frame(&dir2, run, slot, lead);
        }
        let outcome = compute(
            &dir2,
            run,
            &[0, 1, 2, 3],
            &["qpf_1h", "uh_2to5km_1h_max", "10m_wind_1h_max", "10m_wind_run_max"],
        );
        assert!(blocker_reason(&outcome, "qpf_1h").contains("F000"));
        let uh = grid_named(&outcome, "uh_2to5km_1h_max");
        assert_values(uh, &uh_at(&[15, 30, 45, 60]));
        assert_eq!(uh.values, vec![10.0, 40.0, 30.0, 0.5]);
        for slug in ["10m_wind_1h_max", "10m_wind_run_max"] {
            let wind = grid_named(&outcome, slug);
            assert_values(wind, &wind_kt_at(&[15, 30, 45, 60]));
            assert!(wind.strategy.contains("lower bound"), "{slug}: {}", wind.strategy);
        }
        let _ = fs::remove_dir_all(&dir);
        let _ = fs::remove_dir_all(&dir2);
    }

    #[test]
    fn exact_time_maxima_drawn_beside_one_hour_close_only_that_hour() {
        // What the live pass hands the engine at +2 h on a 15-minute grid:
        // the frame on the hour and the hour it closes, +1:00 to +2:00.
        // The 1 h maxima hold all of their window.  The run maxima would
        // fold that hour alone and read low (the 10 m wind as much as the
        // UH, labelled a lower bound or not), so both are refused by name
        // and the whole series draws them.
        let dir = test_dir("exact-time-one-hour");
        let run = "quarter_hour_one_hour";
        let leads = [0u64, 15, 30, 45, 60, 75, 90, 105, 120];
        for (slot, &lead) in leads.iter().enumerate() {
            write_exact_frame(&dir, run, slot as u16, lead);
        }
        let slugs = [
            "uh_2to5km_1h_max",
            "uh_2to5km_run_max",
            "10m_wind_1h_max",
            "10m_wind_run_max",
            "qpf_1h",
        ];
        let outcome = compute(&dir, run, &[4, 5, 6, 7, 8], &slugs);
        assert_eq!(outcome.anchor_hour, 2);
        assert_values(grid_named(&outcome, "uh_2to5km_1h_max"), &uh_at(&[75, 90, 105, 120]));
        assert_values(grid_named(&outcome, "10m_wind_1h_max"), &wind_kt_at(&[75, 90, 105, 120]));
        assert!(outcome.grids.iter().any(|grid| grid.slug == "qpf_1h"));
        for slug in ["uh_2to5km_run_max", "10m_wind_run_max"] {
            let reason = blocker_reason(&outcome, slug);
            assert!(
                reason.contains("unevenly") && reason.contains("read low"),
                "{slug}: {reason}"
            );
        }

        // The whole series draws them, over every frame of the run.
        let outcome = compute(&dir, run, &(0..leads.len() as u16).collect::<Vec<_>>(), &slugs);
        assert_values(grid_named(&outcome, "uh_2to5km_run_max"), &uh_at(&leads[1..]));
        let wind = grid_named(&outcome, "10m_wind_run_max");
        assert_values(wind, &wind_kt_at(&leads[1..]));
        assert_ne!(
            wind.values,
            wind_kt_at(&[60, 75, 90, 105, 120]),
            "the fixture tells the whole run from its last hour"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_exact_time_instant_fold_needs_every_whole_hour() {
        // A 90-minute history: evenly spaced, but no frame at +1 h or +2 h.
        // Each UP_HELI_MAX plane covers its whole 90 minutes, so the UH run
        // maximum is whole without them.  The 10 m wind holds two instants
        // of three hours, where the whole-hour axis refuses a run missing
        // an hour ("gaps are never skipped"), and is refused the same way.
        let dir = test_dir("exact-time-ninety");
        let run = "ninety_minute";
        for (slot, lead) in [(0u16, 0u64), (1, 90), (2, 180)] {
            write_exact_frame(&dir, run, slot, lead);
        }
        let outcome = compute(&dir, run, &[0, 1, 2], &["uh_2to5km_run_max", "10m_wind_run_max"]);
        assert_eq!(outcome.anchor_hour, 3);
        let uh = grid_named(&outcome, "uh_2to5km_run_max");
        assert_values(uh, &uh_at(&[90, 180]));
        let reason = blocker_reason(&outcome, "10m_wind_run_max");
        assert!(
            reason.contains("F001, F002") && reason.contains("gaps are never skipped"),
            "{reason}"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn exact_time_interval_maxima_after_the_first_hour_need_the_frame_at_their_start() {
        // The +1 h frame was never stored.  The frames inside the hour to
        // +2 h are still evenly spaced from its start, so only the missing
        // start frame shows that the interval ending at +1:15 is not the
        // hour's first: it is the one that began at +1 h, but nothing
        // proves it.  The 10 m wind asks the same of its instants: with
        // the start frame missing, a store holding only the last frames
        // of the hour looks whole.
        let dir = test_dir("exact-time-no-hour-start");
        let run = "quarter_hour_no_hour_start";
        let leads = [0u64, 15, 30, 45, 75, 90, 105, 120];
        for (slot, &lead) in leads.iter().enumerate() {
            write_exact_frame(&dir, run, slot as u16, lead);
        }
        let slots: Vec<u16> = (0..leads.len() as u16).collect();
        let outcome = compute(
            &dir,
            run,
            &slots,
            &[
                "uh_2to5km_1h_max",
                "uh_2to5km_run_max",
                "10m_wind_1h_max",
                "10m_wind_run_max",
                "qpf_total",
            ],
        );
        for slug in ["uh_2to5km_1h_max", "10m_wind_1h_max"] {
            let reason = blocker_reason(&outcome, slug);
            assert!(
                reason.contains("F001") && reason.contains("start"),
                "{slug}: the missing start frame is named: {reason}"
            );
        }
        for slug in ["uh_2to5km_run_max", "10m_wind_run_max"] {
            let reason = blocker_reason(&outcome, slug);
            assert!(reason.contains("unevenly"), "{slug}: {reason}");
        }
        assert!(outcome.grids.iter().any(|grid| grid.slug == "qpf_total"));

        // The same hour's last frame alone: one frame inside the window,
        // which no spacing check can see past.
        let outcome = compute(&dir, run, &[7], &["10m_wind_1h_max", "uh_2to5km_1h_max"]);
        for slug in ["10m_wind_1h_max", "uh_2to5km_1h_max"] {
            let reason = blocker_reason(&outcome, slug);
            assert!(reason.contains("F001") && reason.contains("start"), "{slug}: {reason}");
        }
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_thinned_exact_time_series_says_its_maximum_is_exact_only_when_whole() {
        // Every other file of a 15-minute history: the spacing is even, so
        // nothing in the store shows that the 15- and 45-minute intervals
        // are missing.  The fold reads low, and its note says when it is
        // exact rather than claiming the window.
        let dir = test_dir("exact-time-thinned");
        let run = "quarter_hour_thinned";
        for (slot, lead) in [(0u16, 0u64), (1, 30), (2, 60)] {
            write_exact_frame(&dir, run, slot, lead);
        }
        let outcome = compute(&dir, run, &[0, 1, 2], &["uh_2to5km_1h_max"]);
        let uh = grid_named(&outcome, "uh_2to5km_1h_max");
        assert_values(uh, &uh_at(&[30, 60]));
        assert_ne!(uh.values, uh_at(&[15, 30, 45, 60]), "the thinned fold reads low");
        assert!(
            uh.strategy.contains("when every history frame of the run was rendered")
                && uh.strategy.contains("thinned"),
            "{}",
            uh.strategy
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn windowed_compute_uses_the_validated_manifest_hour_filename() {
        let dir = test_dir("manifest-hour-filename");
        let run = "20260608_00z";
        write_test_run(&dir, run, &[1]);
        let run_dir = dir.join("hrrr").join(run);
        let manifest_path = run_dir.join("run.json");
        let mut manifest = RwsRunManifest::load_for_run(&manifest_path, "hrrr", run).unwrap();
        fs::rename(run_dir.join("f001.rws"), run_dir.join("renamed-f001.rws")).unwrap();
        manifest.hours.get_mut(&1).unwrap().file = "renamed-f001.rws".to_string();
        manifest.save(&manifest_path).unwrap();

        let outcome = compute(&dir, run, &[1], &["qpf_1h"]);
        assert!(outcome.blockers.is_empty(), "{:?}", outcome.blockers);
        assert_eq!(grid_named(&outcome, "qpf_1h").hours_used, vec![1]);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn stored_run_hours_rejects_unsafe_persisted_hour_path() {
        let dir = test_dir("unsafe-manifest-hour");
        let run = "20260608_00z";
        write_test_run(&dir, run, &[1]);
        let manifest_path = dir.join("hrrr").join(run).join("run.json");
        let mut manifest = RwsRunManifest::load_for_run(&manifest_path, "hrrr", run).unwrap();
        manifest.hours.get_mut(&1).unwrap().file = "../outside.rws".to_string();
        // Deliberately bypass save(), which rejects this before persistence,
        // to model an externally modified/untrusted store.
        fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();

        let err = stored_run_hours(&dir, "hrrr", run).unwrap_err().to_string();
        assert!(err.contains("normal path component"), "{err}");
        let _ = fs::remove_dir_all(&dir);
    }
}
