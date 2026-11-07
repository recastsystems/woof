//! `--diff-against`: every requested map product drawn as run A minus run B,
//! at the valid time both runs share, on the grid both runs share.
//!
//! The positional wrfouts are run A and the `--diff-against` files are run
//! B: two start sources, two physics choices, two engine versions, or a
//! forecast and an analysis.  Each run is imported into its own store under
//! `--store-root` (`b/` and `a/`) and drawn by the ordinary batch renderer,
//! so every product, derived ones included, is computed from each run's own
//! inputs.  Run B's pass keeps each product's drawn field; run A's pass
//! subtracts it and draws the difference instead of run A's picture
//! (`rustwx_render::difference`).  With `--diff-sheet` both runs' own
//! pictures are drawn as well and composed beside the difference as an
//! `A | B | A minus B` sheet.
//!
//! Refusals, each by name and before any product is drawn where it can be:
//!
//! * run A's frame and run B's frames share no valid time;
//! * the two runs' stores are on grids of different sizes (a grid of the
//!   same size with different coordinates is refused per product by the
//!   renderer, which compares every cell);
//! * `xsec:` and `mesh:` products, which do not draw through the store.
//!
//! One valid time per invocation.  Pairing a whole series by valid time is
//! the caller's (`gpuwm.rustwx.pair_frames_by_valid_time`).

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::time::Instant;

use chrono::Timelike;

use rustwx_render::difference::{self, DifferenceLabels, DrawnDifference};
use rustwx_products::shared_context::TitleProvenance;
use rusty_weather::batch_render::{
    BatchHourScope, BatchRenderEvent, inspect_renderable_products_over, run_batch_render,
};

use super::{
    Args, CliError, ImportedRun, RenderedPanelGeoref, StoreBatch, domain_title_label,
    expand_catalog_keywords, frame_attributed, grid_identity, import_run, native_domain_slug,
    spacing_subtitle, store_batch_request, write_georef_manifest,
};

/// The difference half of the command line.
#[derive(Debug, Clone)]
pub struct DifferenceArgs {
    /// Run B's wrfouts.
    pub against: Vec<PathBuf>,
    pub labels: DifferenceLabels,
    /// Also draw both runs' own panels and an `A | B | A minus B` sheet.
    pub sheet: bool,
}

/// Refuse what a difference cannot draw, before any file is opened.
pub(super) fn validate(
    args: &Args,
    difference: &DifferenceArgs,
    has_file_products: bool,
) -> Result<(), CliError> {
    if has_file_products {
        return Err(CliError::Usage(
            "a difference is drawn for map products from the store; xsec: and mesh: products \
             are cut from the files directly and are not differenced here (meshdiff: is the \
             mesh family's own difference)"
                .to_string(),
        ));
    }
    if args.list_products {
        return Err(CliError::Usage(
            "--list-products describes one run; ask it of each run without --diff-against"
                .to_string(),
        ));
    }
    for path in &difference.against {
        if !path.is_file() {
            return Err(CliError::Failed(format!(
                "{}: unreadable run B wrfout (no such file)",
                path.display()
            )));
        }
        if !crate::wrf_process::is_supported_wrf_file(path) {
            return Err(CliError::Failed(format!(
                "{}: unreadable run B wrfout (not a raw WRF or post-processed NetCDF file this \
                 build recognises)",
                path.display()
            )));
        }
    }
    Ok(())
}

/// Every valid time the inputs hold, ascending and distinct, read from the
/// files themselves (WRF `Times`, or a CF time coordinate).
fn valid_times(inputs: &[PathBuf]) -> Result<Vec<i64>, String> {
    let mut times = Vec::new();
    for path in inputs {
        let raw = crate::wrf_process::isolate_panics("read WRF times", || {
            wrf_core::WrfFile::open(path).map_err(|err| err.to_string())
        });
        let axis = match raw {
            Ok(file) => crate::local_import::wrf_source_times(&file, path)?,
            Err(_) => {
                let nc = netcrust::open(path)
                    .map_err(|err| format!("open {}: {err}", path.display()))?;
                crate::local_import::netcdf_source_times(&nc, path)
                    .map_err(|err| format!("read times from {}: {err}", path.display()))?
            }
        };
        times.extend(axis.records.iter().map(|record| record.valid_unix));
    }
    times.sort_unstable();
    times.dedup();
    Ok(times)
}

fn format_valid(unix: i64) -> String {
    chrono::DateTime::from_timestamp(unix, 0)
        .map(|time| time.format("%Y-%m-%d %H:%MZ").to_string())
        .unwrap_or_else(|| unix.to_string())
}

/// Which frame of each run is drawn: run A's `--frames N` (or its only
/// frame), and run B's frame valid at the same time.
pub fn pick_frames(
    a_times: &[i64],
    b_times: &[i64],
    a_frame: Option<usize>,
) -> Result<(usize, usize, i64), String> {
    let a_index = match a_frame {
        Some(index) => {
            if index >= a_times.len() {
                return Err(format!(
                    "--frames {index} out of range; run A has {} frame(s)",
                    a_times.len()
                ));
            }
            index
        }
        None if a_times.len() == 1 => 0,
        None => {
            return Err(format!(
                "run A holds {} valid times ({} to {}); a difference draws one, so name it \
                 with --frames N",
                a_times.len(),
                format_valid(a_times[0]),
                format_valid(a_times[a_times.len() - 1])
            ));
        }
    };
    let valid = a_times[a_index];
    let b_index = b_times.iter().position(|time| *time == valid).ok_or_else(|| {
        let held: Vec<String> = b_times.iter().map(|time| format_valid(*time)).collect();
        format!(
            "run B has no frame valid at {}, the valid time of run A's frame (run B's frames \
             are valid at {}); a difference is drawn only where both runs share the valid time",
            format_valid(valid),
            if held.is_empty() {
                "no time".to_string()
            } else {
                held.join(", ")
            }
        )
    })?;
    Ok((a_index, b_index, valid))
}

/// The grid size a run's store holds, from its manifest.
fn store_grid(store_root: &Path, imported: &ImportedRun) -> Result<(usize, usize, String), String> {
    let path = store_root
        .join(&imported.model)
        .join(&imported.run)
        .join("run.json");
    let manifest = rw_store::run::RwsRunManifest::load_bounded(&path)
        .map_err(|err| format!("read {}: {err}", path.display()))?;
    Ok((manifest.nx, manifest.ny, manifest.grid_hash))
}

struct RunContext {
    store_root: PathBuf,
    imported: ImportedRun,
    slot: u16,
    domain_slug: Option<String>,
    spacing: Option<String>,
    title_provenance: TitleProvenance,
    initial_condition_disclosure: Option<String>,
    model_label: Option<String>,
}

fn import_side(
    label: &str,
    inputs: Vec<PathBuf>,
    store_root: PathBuf,
    products: &str,
    heavy: bool,
    frame_index: usize,
) -> Result<RunContext, String> {
    let identity = grid_identity(&inputs);
    let domain_slug = native_domain_slug(&identity);
    let spacing = identity.spacing_m.and_then(spacing_subtitle);
    let title_provenance = TitleProvenance::LocalImport {
        grid_label: domain_title_label(&identity),
    };
    let imported = import_run(inputs, &store_root, products, heavy, false)
        .map_err(|err| format!("run {label}: {err}"))?;
    let slot = imported
        .stored_slots
        .get(frame_index)
        .copied()
        .ok_or_else(|| {
            format!(
                "run {label}: frame {frame_index} was not stored ({} frame(s) imported)",
                imported.stored_slots.len()
            )
        })?;
    Ok(RunContext {
        store_root,
        imported,
        slot,
        domain_slug,
        spacing,
        title_provenance,
        initial_condition_disclosure:
            rustwx_products::shared_context::initial_condition_disclosure(),
        model_label: rustwx_products::shared_context::model_label(),
    })
}

/// Restore the provenance the import resolved for this entire run. Both
/// imports complete before either pass renders, so the process-wide model
/// label and initial-condition line otherwise belong to the last import.
fn install_provenance(side: &RunContext) {
    rustwx_products::shared_context::set_initial_condition_disclosure(
        side.initial_condition_disclosure.clone());
    rustwx_products::shared_context::set_model_label(side.model_label.clone());
}

struct PassCounts {
    rendered: usize,
    skipped: usize,
    failed: usize,
    elapsed_ms: u128,
}

pub(super) fn run(mut args: Args) -> Result<(), String> {
    let started = Instant::now();
    let difference = args
        .difference
        .take()
        .ok_or("internal: a difference run without --diff-against")?;
    // A difference table the caller named but that cannot be read is
    // refused before either run is imported, not after both are drawn.
    difference::DifferenceTable::active().map_err(|err| err.to_string())?;
    let a_times = valid_times(&args.inputs)?;
    let b_times = valid_times(&difference.against)?;
    let (a_index, b_index, valid) = pick_frames(&a_times, &b_times, args.frames)?;
    println!(
        "DIFFERENCE valid={} a_frame={a_index} b_frame={b_index} a={:?} b={:?}",
        format_valid(valid),
        difference.labels.a,
        difference.labels.b
    );
    let work_root = args.store_root.clone();
    let panel_dir = difference.sheet.then(|| work_root.join("panels"));
    if let Some(dir) = &panel_dir {
        std::fs::create_dir_all(dir)
            .map_err(|err| format!("create {}: {err}", dir.display()))?;
    }
    std::fs::create_dir_all(&args.out_dir)
        .map_err(|err| format!("create {}: {err}", args.out_dir.display()))?;

    // Both runs are imported before either is drawn, so a grid of another
    // size is refused before any product is computed.
    let import_b_started = Instant::now();
    let b = import_side(
        "B",
        difference.against.clone(),
        work_root.join("b"),
        &args.products,
        args.heavy,
        b_index,
    )?;
    let import_b_ms = import_b_started.elapsed().as_millis();
    let import_a_started = Instant::now();
    let a = import_side(
        "A",
        args.inputs.clone(),
        work_root.join("a"),
        &args.products,
        args.heavy,
        a_index,
    )?;
    let import_a_ms = import_a_started.elapsed().as_millis();
    let (a_nx, a_ny, a_hash) = store_grid(&a.store_root, &a.imported)?;
    let (b_nx, b_ny, b_hash) = store_grid(&b.store_root, &b.imported)?;
    if (a_nx, a_ny) != (b_nx, b_ny) {
        return Err(format!(
            "run A is on a {a_nx}x{a_ny} grid and run B on a {b_nx}x{b_ny} grid; a difference \
             is taken on one native grid and is never regridded"
        ));
    }
    if a_hash != b_hash {
        // Same size, different identity: the renderer compares every cell's
        // coordinates per product and refuses by name if they differ.
        eprintln!(
            "NOTE run A and run B grid identities differ ({a_hash} vs {b_hash}); every product \
             is checked cell by cell before it is differenced"
        );
    }
    // Run A's catalog decides the product list, and run B is asked for the
    // same list: a product run B cannot draw is named when run A's pass
    // finds nothing to subtract.
    let catalog_a = inspect_renderable_products_over(
        &a.store_root,
        &a.imported.model,
        &a.imported.run,
        &[a.slot],
    )?;
    let product_spec = expand_catalog_keywords(&args.products, &catalog_a)?;
    println!(
        "CATALOG products={} stored_hours={:?}",
        catalog_a.products.len(),
        catalog_a.stored_hours
    );
    let b_frames: HashMap<u16, PathBuf> = b.imported.frame_sources.clone();

    let request_for = |side: &RunContext, out_dir: PathBuf| {
        store_batch_request(StoreBatch {
            store_root: side.store_root.clone(),
            model_slug: side.imported.model.clone(),
            run_slug: side.imported.run.clone(),
            hours: BatchHourScope::Current(side.slot),
            stored_frames: 1,
            product_spec: product_spec.clone(),
            out_dir,
            native_domain_slug: side.domain_slug.clone(),
            subtitle_spacing: side.spacing.clone(),
            source_label: args.source_label.clone(),
            title_provenance: side.title_provenance.clone(),
            geographic_overlays: args.overlays.clone(),
            panel_annotations: args.annotations.clone(),
            width: args.width,
            height: args.height,
            render_bounds: args.render_bounds,
        })
    };
    let cancel = AtomicBool::new(false);

    // Pass B: capture.  Nothing is written to --out-dir; with a sheet, run
    // B's own pictures go to the work folder.
    install_provenance(&b);
    difference::begin_capture(difference.labels.clone(), panel_dir.clone());
    let b_out = work_root.join("b-out");
    let mut unavailable_b = HashSet::new();
    let b_summary = run_batch_render(request_for(&b, b_out.clone()), &cancel, |event| match event {
        BatchRenderEvent::ItemSkipped {
            hour, slug, reason, ..
        } => {
            unavailable_b.insert(slug.clone());
            println!("SKIPPED {slug} run B: {}",
                     frame_attributed(&b_frames, hour, &reason));
        }
        BatchRenderEvent::ItemFailed {
            hour, slug, error, ..
        } => eprintln!(
            "FAILED {slug} run B: {}",
            frame_attributed(&b_frames, hour, &error)
        ),
        _ => {}
    });
    let b_summary = match b_summary {
        Ok(summary) => summary,
        Err(err) => {
            difference::finish();
            return Err(format!("run B: {err}"));
        }
    };
    let capture = PassCounts {
        rendered: b_summary.rendered,
        skipped: b_summary.skipped,
        failed: b_summary.failed,
        elapsed_ms: b_summary.elapsed_ms,
    };
    let captured = difference::begin_subtract().map_err(|err| err.to_string())?;
    println!(
        "CAPTURED run_b_products={captured} skipped={} failed={} elapsed_ms={}",
        capture.skipped, capture.failed, capture.elapsed_ms
    );

    // Pass A: subtract.  Each product's difference is written beside the
    // name run A's own picture would have had, with `_difference`.
    let a_frames: HashMap<u16, PathBuf> = a.imported.frame_sources.clone();
    install_provenance(&a);
    let mut panel_georefs: Vec<RenderedPanelGeoref> = Vec::new();
    let mut a_request = request_for(&a, args.out_dir.clone());
    // A registered absence in run B has no quantity to subtract. Keep
    // its named skip rather than trying to read an uncaptured field and
    // turning a missing accumulation window into a hard render failure.
    a_request.product_spec = product_spec.split(',')
        .filter(|slug| !unavailable_b.contains(*slug))
        .collect::<Vec<_>>().join(",");
    let a_summary = if a_request.product_spec.is_empty() {
        Ok(rusty_weather::batch_render::BatchRenderSummary::default())
    } else { run_batch_render(
        a_request,
        &cancel,
        |event| match event {
            BatchRenderEvent::ItemRendered {
                slug,
                output_path,
                georeference,
                georeference_absent_reason,
                ..
            } => {
                // The difference was written under run A's own name
                // (every lane reads back the file it asked for); it is
                // filed as its own product here.
                let path = difference::difference_path(&output_path);
                match std::fs::rename(&output_path, &path) {
                    Ok(()) => {
                        println!("RENDERED {slug}_difference {}", path.display());
                        panel_georefs.push((path, georeference, georeference_absent_reason));
                    }
                    Err(err) => eprintln!(
                        "FAILED {slug} move {} to {}: {err}",
                        output_path.display(),
                        path.display()
                    ),
                }
            }
            BatchRenderEvent::ItemSkipped {
                hour, slug, reason, ..
            } => println!(
                "SKIPPED {slug} {}",
                frame_attributed(&a_frames, hour, &reason)
            ),
            BatchRenderEvent::ItemFailed {
                hour, slug, error, ..
            } => eprintln!(
                "FAILED {slug} {}",
                frame_attributed(&a_frames, hour, &error)
            ),
            _ => {}
        },
    ) };
    let (drawn, unmatched) = difference::finish();
    let a_summary = a_summary.map_err(|err| format!("run A: {err}"))?;
    let subtract = PassCounts {
        rendered: a_summary.rendered,
        skipped: a_summary.skipped,
        failed: a_summary.failed,
        elapsed_ms: a_summary.elapsed_ms,
    };
    for item in &drawn {
        print_drawn(item);
    }
    for key in &unmatched {
        println!("SKIPPED {key} run A drew no {key} to subtract run B's from");
    }
    write_georef_manifest(&args.out_dir, &panel_georefs)?;

    // The sheets.
    let sheet_started = Instant::now();
    let mut sheets = 0usize;
    let mut sheet_failures = 0usize;
    if difference.sheet {
        for item in &drawn {
            match compose_sheet(item, &difference.labels, valid) {
                Ok(path) => {
                    sheets += 1;
                    println!("RENDERED {}_sheet {}", item.key, path.display());
                }
                Err(err) => {
                    sheet_failures += 1;
                    eprintln!("FAILED {}_sheet {err}", item.key);
                }
            }
        }
    }
    let sheet_ms = sheet_started.elapsed().as_millis();

    // The work folder holds both stores, run B's discarded pictures and
    // the run panels the sheets were composed from; none of it is output.
    for dir in ["a", "b", "b-out", "panels"] {
        let path = work_root.join(dir);
        if path.exists() {
            if let Err(err) = std::fs::remove_dir_all(&path) {
                eprintln!("NOTE could not remove work folder {}: {err}", path.display());
            }
        }
    }
    let _ = std::fs::remove_dir(&b_out);

    println!(
        "DIFFTIMING import_b_ms={import_b_ms} import_a_ms={import_a_ms} capture_ms={} \
         subtract_ms={} sheet_ms={sheet_ms}",
        capture.elapsed_ms, subtract.elapsed_ms
    );
    let failed = capture.failed + subtract.failed + sheet_failures;
    let skipped = subtract.skipped + unavailable_b.len() + unmatched.len();
    println!(
        "FINISHED rendered={} skipped={} failed={failed} elapsed_ms={}",
        subtract.rendered + sheets,
        skipped,
        started.elapsed().as_millis()
    );
    let _ = capture.rendered;
    if failed > 0 || (drawn.is_empty() && skipped == 0) {
        return Err(format!(
            "difference incomplete: drawn={} skipped={} failed={failed}",
            drawn.len(),
            skipped
        ));
    }
    Ok(())
}

fn print_drawn(item: &DrawnDifference) {
    println!(
        "DIFFERENCE {} units={} half_range={} step={} rule={} defined_cells={} \
         differing_cells={} max_abs={}",
        item.key,
        item.units,
        item.half_range,
        item.step,
        item.rule.describe(),
        item.defined_cells,
        item.differing_cells,
        difference::magnitude_label(item.max_abs)
    );
}

/// `A | B | A minus B` for one product: the two runs' own pictures and the
/// difference, under one header naming the comparison.
fn compose_sheet(
    item: &DrawnDifference,
    labels: &DifferenceLabels,
    valid: i64,
) -> Result<PathBuf, String> {
    let (Some(a_panel), Some(b_panel)) = (&item.a_panel, &item.b_panel) else {
        return Err("the run panels were not drawn".to_string());
    };
    let legs = vec![
        (format!("A: {}", labels.a), a_panel.clone()),
        (format!("B: {}", labels.b), b_panel.clone()),
        ("A minus B".to_string(), item.output.clone()),
    ];
    let title = format!("{} minus {}", labels.a, labels.b);
    // Spelled as the panels' own headers spell a valid time.
    let subtitle = chrono::DateTime::from_timestamp(valid, 0)
        .map(|time| {
            if time.minute() == 0 {
                format!("Valid {}", time.format("%m/%d %HZ"))
            } else {
                format!("Valid {}", time.format("%m/%d %H:%MZ"))
            }
        })
        .unwrap_or_else(|| format!("Valid {}", format_valid(valid)));
    let canvas = crate::sheet::compose_legs(&title, Some(&subtitle), &legs)?;
    let stem = item
        .output
        .file_stem()
        .map(|stem| stem.to_string_lossy().into_owned())
        .unwrap_or_default();
    let path = item.output.with_file_name(format!("{stem}_sheet.png"));
    canvas
        .save(&path)
        .map_err(|err| format!("write {}: {err}", path.display()))?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frames_pair_by_valid_time_and_a_missing_time_is_refused_by_name() {
        let a = [3600 * 12];
        let b = [3600 * 11, 3600 * 12, 3600 * 13];
        assert_eq!(pick_frames(&a, &b, None).unwrap(), (0, 1, 3600 * 12));
        let err = pick_frames(&[3600 * 14], &b, None).unwrap_err();
        assert!(err.contains("no frame valid at 1970-01-01 14:00Z"), "{err}");
        assert!(err.contains("share the valid time"), "{err}");
        let err = pick_frames(&b, &b, None).unwrap_err();
        assert!(err.contains("--frames N"), "{err}");
        assert_eq!(pick_frames(&b, &b, Some(2)).unwrap(), (2, 2, 3600 * 13));
        assert!(pick_frames(&b, &b, Some(3)).is_err());
    }
}
