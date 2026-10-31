//! Native WOOF simulated-radar history consumer.
mod adapter;
mod atmosphere;
mod input_contract;
mod loops;
mod manifest;
mod ppi;
mod request;
mod resources;

use bowecho_simradar::radar_core::{MomentType, RadarVolume};
use bowecho_simradar::wrf_scene_adapter::inventory_wrf_paths;
use bowecho_simradar::wrf_scene_inventory::WrfSceneLocator;
use bowecho_simradar::wrf_temporal::{TemporalSamplingOutcome, plan_for_scene};
use bowecho_simradar::{
    WrfFile, WrfRadarFields, build_synthetic_volume_reporting_temporal,
    read_wrf_radar_fields_for_config, try_build_synthetic_volume,
};
use rayon::prelude::*;
use request::{Fields, Request};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// PPI images drawn per volume: the lowest distinct tilts, each in pixels.
/// `resources::output_bound` prices exactly these.
const PPI_TILTS: usize = 2;
const PPI_WIDTH: u32 = 960;
const PPI_HEIGHT: u32 = 900;

fn read_scene(
    scene: &WrfSceneLocator,
    config: &bowecho_simradar::SyntheticRadarConfig,
) -> Result<WrfRadarFields, String> {
    let file = WrfFile::open(&scene.path).map_err(|e| e.to_string())?;
    let mut fields =
        read_wrf_radar_fields_for_config(&file, &scene.source_identity, scene.time_index, config)?;
    atmosphere::native_winds(&file, scene.time_index, &mut fields)?;
    if fields.ref_source == bowecho_simradar::REFL_10CM_SOURCE {
        let (nx, ny, nz) = (fields.nx, fields.ny, fields.nz);
        unwritten_reflectivity_ring(&mut fields.dbz, nx, ny, nz);
    }
    Ok(fields)
}

/// Marks missing (NaN) the cells of the domain's outermost row and column
/// whose `REFL_10CM` is exactly 0, the array's initial value; returns how many.
///
/// `REFL_10CM` is history-only (WRF Registry `hdu`, initialised 0) and the
/// microphysics fills it on the columns it computes. The domain's boundary
/// ring is not one of them, so those cells read exactly 0 dBZ in every frame
/// (gpuwm writes WRF's zeros there as well: `gpuwm/core/refl.py`
/// `analysis_refl_10cm`). Microphysics floors its own values at -35 dBZ or
/// another scheme floor, so no interior cell is exactly 0 in a real frame.
/// Scanned as an echo, 0 dBZ is above the -30 dBZ floor, so every gate on the
/// ring kept its radial velocity and the PPI drew a one-cell line along the
/// domain's edges.
///
/// THE BREAKAGE THIS PREVENTS (WOOF run_5d864756da757b75, 2026-10-29: 2.25 to
/// 0.75 km, 5-min quick look with "Simulated radar every N min"): the
/// velocity PPIs of the quick-look volumes (KMQT 01:05Z and 01:10Z) showed a
/// +/-20 m/s line down d02's west edge and along its south edge. A quick-look
/// frame carries no hydrometeors, so its radar reflectivity is the frame's
/// `REFL_10CM`; a full frame's comes from the hydrometeors on every cell and
/// was clean. Winds on the ring match the cells beside them.
///
/// `sample_column` drops a stencil column whose reflectivity is not finite,
/// so a gate at the edge samples the interior columns beside it. A ring cell
/// holding any other value (an echo a model did write there) is kept.
fn unwritten_reflectivity_ring(dbz: &mut [f32], nx: usize, ny: usize, nz: usize) -> usize {
    if nx == 0 || ny == 0 || dbz.len() != nx * ny * nz {
        return 0;
    }
    let mut marked = 0;
    let mut mark = |cell: &mut f32| {
        if *cell == 0.0 {
            *cell = f32::NAN;
            marked += 1;
        }
    };
    for plane in dbz.chunks_exact_mut(nx * ny) {
        for j in [0, ny - 1] {
            plane[j * nx..(j + 1) * nx].iter_mut().for_each(&mut mark);
            if ny == 1 {
                break;
            }
        }
        for j in 1..ny.saturating_sub(1) {
            mark(&mut plane[j * nx]);
            if nx > 1 {
                mark(&mut plane[j * nx + nx - 1]);
            }
        }
    }
    marked
}

/// The atmosphere bytes one history's scene holds, from its header alone.
fn scene_atmosphere_bytes(path: &Path) -> Result<u64, String> {
    let header = WrfFile::open(path).map_err(|e| e.to_string())?;
    resources::atmosphere_bytes(header.nx, header.ny, header.nz)
}

fn field_name(moment: &MomentType) -> &str {
    match moment {
        MomentType::Reflectivity => "reflectivity",
        MomentType::Velocity => "velocity",
        MomentType::DifferentialReflectivity => "zdr",
        MomentType::CorrelationCoefficient => "rhohv",
        MomentType::DifferentialPhase => "phidp",
        MomentType::SpecificDifferentialPhase => "kdp",
        MomentType::SpectrumWidth => "spectrum_width",
        MomentType::Unknown(v) => v,
    }
}

fn select_fields(volume: &mut RadarVolume, fields: &Fields) -> Result<(), String> {
    if let Fields::List(wanted) = fields {
        for name in wanted {
            if !volume
                .cuts
                .iter()
                .any(|c| c.moments.keys().any(|m| field_name(m) == name))
            {
                return Err(format!(
                    "requested radar field {name} is unavailable for this history's microphysics"
                ));
            }
        }
        for cut in &mut volume.cuts {
            cut.moments
                .retain(|moment, _| wanted.iter().any(|f| f == field_name(moment)));
        }
    } else {
        for cut in &mut volume.cuts {
            cut.moments.retain(|moment, _| {
                matches!(
                    moment,
                    MomentType::Reflectivity
                        | MomentType::Velocity
                        | MomentType::DifferentialReflectivity
                        | MomentType::CorrelationCoefficient
                        | MomentType::DifferentialPhase
                        | MomentType::SpecificDifferentialPhase
                )
            });
        }
    }
    Ok(())
}

/// Admit a whole batch before any site allocates. Never wait for permits on
/// Rayon workers: nested beam/encoder/PPI work uses the same pool. Collect an
/// indexed iterator, then commit in input order, including successes before an
/// error. A completed batch retains metadata only, not its polar volumes.
fn site_batches<T: Sync, V: Send>(
    sites: &[T],
    limit: usize,
    admit: impl Fn(usize) -> Result<(), String>,
    scan: impl Fn(&T) -> Result<V, String> + Sync,
    mut commit: impl FnMut(V) -> Result<(), String>,
) -> Result<(), String> {
    let mut remaining = sites;
    while !remaining.is_empty() {
        let mut count = remaining.len().min(limit.max(1));
        loop {
            match admit(count) {
                Ok(()) => break,
                Err(error) if count == 1 => return Err(error),
                Err(_) => count -= 1,
            }
        }
        let results: Vec<_> = remaining[..count].par_iter().map(&scan).collect();
        for result in results {
            commit(result?)?;
        }
        remaining = &remaining[count..];
    }
    Ok(())
}

fn process(request: Request) -> Result<PathBuf, String> {
    if request.schema != "simulated-radar.request/v1" {
        return Err("unsupported simulated radar request schema".into());
    }
    request.config.validate()?;
    // One request draws one radar look, selected before the first PPI.
    rustwx_render::install_radar_color_set(request.config.color_set()?)?;
    // Reject impossible scans before even hashing the history tape.
    request.config.check_memory(0)?;
    if request.history_paths.is_empty() {
        return Err("simulated radar requires history files".into());
    }
    if !request.scene_shapes.is_empty() {
        return Err("scene_shapes prices a forecast before its history exists (--estimate only); a volume request reads the shapes of its own history files".into());
    }
    let mut histories = request.history_paths;
    histories.sort();
    histories.dedup();
    // The histories that publish volumes; the others are scan-timing
    // neighbours only. A name outside the request would publish nothing.
    let volume_paths: Option<std::collections::BTreeSet<PathBuf>> = match request.volume_paths {
        None => None,
        Some(paths) => {
            if paths.is_empty() {
                return Err("volume_paths is empty, so the request would publish no volume; omit it to publish every history".into());
            }
            if let Some(path) = paths
                .iter()
                .find(|path| histories.binary_search(*path).is_err())
            {
                return Err(format!(
                    "volume_paths names {} which is not in history_paths; a volume is published only from a history the request reads",
                    path.display()
                ));
            }
            Some(paths.into_iter().collect())
        }
    };
    for path in &histories {
        input_contract::validate_history(path)?;
    }
    // Every history is hashed at once: one file per core, not one core for all.
    let source_hashes: BTreeMap<_, _> = histories
        .par_iter()
        .map(|path| manifest::file_hash(path).map(|(hash, _)| (path.clone(), hash)))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .collect();
    let inventory = inventory_wrf_paths(&histories)?;
    if inventory
        .inventory
        .groups
        .iter()
        .all(|g| g.scenes.is_empty())
    {
        return Err("history contains no model records to scan".into());
    }
    let mut unique_frames = std::collections::BTreeSet::new();
    for group in &inventory.inventory.groups {
        for scene in &group.scenes {
            if !unique_frames.insert((scene.run_domain.domain, scene.time.valid_time().cloned())) {
                return Err("inputs repeat a domain and valid time across runs or grids; give each run its own radar output directory".into());
            }
        }
    }
    for note in inventory.notes {
        eprintln!("warning: {}: {}", note.source_name, note.message);
    }
    let _lock = manifest::RunLock::acquire(&request.outdir)?;
    let provenance: serde_json::Value =
        serde_json::from_str(include_str!("../../../vendor/bowecho/SOURCE.json"))
            .map_err(|e| e.to_string())?;
    let extraction = provenance
        .get("extraction_commit")
        .and_then(|v| v.as_str())
        .ok_or("vendored BowEcho source lacks extraction commit")?;
    let mut manifest = manifest::Manifest::load(&request.outdir, extraction)?;
    manifest.bowecho_source_commit = provenance["source_commit"]
        .as_str()
        .ok_or("BowEcho source pin is missing")?
        .into();
    manifest.bowecho_extraction_commit = extraction.into();
    #[cfg(target_os = "linux")]
    let executable = PathBuf::from("/proc/self/exe");
    #[cfg(not(target_os = "linux"))]
    let executable = std::env::current_exe().map_err(|e| e.to_string())?;
    let binary_hash = manifest::file_hash(&executable)?.0;
    let implementation = serde_json::json!({"bowecho_source_commit":provenance["source_commit"],
        "bowecho_extraction_commit":extraction,"writer_source_commit":"c206a2495c36341caa2a62ff7be3025320dbb028",
        "contract":request::ABI,"native_source_sha256":env!("SIMRADAR_IMPLEMENTATION_SHA256"),
        "native_binary_sha256":binary_hash});
    if !request.config.enabled {
        manifest.save(&request.outdir)?;
        return Ok(manifest::Manifest::path(&request.outdir));
    }
    let config_hash =
        manifest::digest(&serde_json::to_vec(&request.config).map_err(|e| e.to_string())?);
    let base = request.config.native();
    base.validate_science_contract()?;
    let mut processed = 0usize;
    let groups = inventory.inventory.groups;
    // Plan every volume before reading any atmosphere, so the next volume's
    // history can be read while the current one is scanned and written.
    let mut work = Vec::new();
    for (group_index, group) in groups.iter().enumerate() {
        if !group.diagnostics.duplicate_times.is_empty() {
            return Err("history inputs repeat a domain time; ambiguous radar atmospheres cannot share one volume path".into());
        }
        if !group.diagnostics.unavailable_times.is_empty() {
            return Err("history has no valid time; radar volumes cannot be dated".into());
        }
        let domain = group.key.run_domain.domain.label();
        for (index, scene) in group.scenes.iter().enumerate() {
            if volume_paths
                .as_ref()
                .is_some_and(|selected| !selected.contains(&scene.path))
            {
                continue;
            }
            let plan = plan_for_scene(
                group,
                index,
                chrono::Duration::milliseconds(base.planned_scan_duration_ms()),
                base.atmosphere_time_mode,
                base.missing_neighbor_policy,
            )
            .map_err(|e| e.to_string())?
            .ok_or("radar temporal plan unexpectedly omitted an output")?;
            let mut source_times = vec![plan.anchor_time.to_rfc3339()];
            let mut source_history = vec![source_description(&scene.locator(), &source_hashes)];
            if plan.outcome == TemporalSamplingOutcome::LinearAdjacent {
                if let Some(time) = plan.neighbor_time {
                    source_times.push(time.to_rfc3339());
                }
                if let Some(source) = &plan.neighbor {
                    source_history.push(source_description(source, &source_hashes));
                }
            }
            let generation=manifest::digest(&serde_json::to_vec(&serde_json::json!({
                "config":request.config,"implementation":implementation,"sources":source_history,
                "source_times":source_times,"domain":domain,"run":group.key.run_domain.run.0,
                "timing_outcome":format!("{:?}",plan.outcome),
            })).map_err(|e|e.to_string())?);
            if manifest.complete_generation(
                &request.outdir,
                &domain,
                &plan.anchor_time.to_rfc3339(),
                &generation,
            ) {
                println!(
                    "{}",
                    serde_json::json!({"event":"already_complete","domain":domain,"valid_time":plan.anchor_time.to_rfc3339()})
                );
                continue;
            }
            work.push((
                group_index,
                index,
                domain.clone(),
                plan,
                source_times,
                source_history,
                generation,
            ));
        }
    }
    // The next volume's atmosphere, read on its own thread while the current
    // volume is scanned and written: a history read is mostly one core.
    let mut ahead: Option<(
        PathBuf,
        usize,
        std::thread::JoinHandle<Result<WrfRadarFields, String>>,
    )> = None;
    for (position, (group_index, index, domain, plan, source_times, source_history, generation)) in
        work.iter().enumerate()
    {
        let (domain, generation) = (domain.clone(), generation.clone());
        let scene = &groups[*group_index].scenes[*index];
        {
            let started = Instant::now();
            let scene_bytes = scene_atmosphere_bytes(&scene.path)?.checked_mul(request.config.atmosphere_factor()).ok_or("radar band memory estimate overflow")?;
            let resident_scenes = if plan.outcome == TemporalSamplingOutcome::LinearAdjacent {2} else {1};
            let resident_bytes = scene_bytes.checked_mul(resident_scenes).ok_or("radar memory estimate overflow")?;
            request.config.check_memory(resident_bytes)?;
            let locator = scene.locator();
            let current = ahead.take();
            // The next volume's read starts before this one's is awaited, so
            // the first two histories are read side by side as well.
            if let Some((next_group, next_index, ..)) = work.get(position + 1) {
                let next = groups[*next_group].scenes[*next_index].locator();
                let admitted = scene_atmosphere_bytes(&next.path).and_then(|next_bytes| {
                    request.config.check_memory(
                        resident_bytes
                            .checked_add(next_bytes)
                            .ok_or("radar memory estimate overflow")?,
                    )
                });
                match admitted {
                    Ok(()) => {
                        let config = base.clone();
                        let (path, time_index) = (next.path.clone(), next.time_index);
                        let handle = std::thread::spawn(move || read_scene(&next, &config));
                        ahead = Some((path, time_index, handle));
                    }
                    Err(reason) => eprintln!(
                        "warning: next radar history is read after this volume, not beside it: {reason}"
                    ),
                }
            }
            let fields = match current {
                Some((path, time_index, handle))
                    if path == locator.path && time_index == locator.time_index =>
                {
                    handle
                        .join()
                        .map_err(|_| "radar history read-ahead panicked".to_string())??
                }
                other => {
                    if let Some((_, _, handle)) = other {
                        let _ = handle.join();
                    }
                    read_scene(&locator, &base)?
                }
            };
            let sites = request.config.resolve_sites(&fields)?;
            if sites.is_empty() {
                manifest.retire_where(|v| {
                    v.domain == domain && v.valid_time == plan.anchor_time.to_rfc3339()
                });
                let warning = format!(
                    "no NEXRAD coverage overlaps {domain} at {}; custom sites can scan this domain",
                    plan.anchor_time
                );
                eprintln!("warning: {warning}");
                if !manifest.warnings.contains(&warning) {
                    manifest.warnings.push(warning);
                }
                manifest.save(&request.outdir)?;
                continue;
            }
            let expected_sites: Vec<_> = sites.iter().map(|s| s.id.clone()).collect();
            // Superseded generations leave the manifest here and their files
            // are deleted once a save no longer names them.
            manifest.retire_where(|v| {
                v.domain == domain
                    && v.valid_time == plan.anchor_time.to_rfc3339()
                    && v.generation != generation
            });
            let neighbor = if plan.outcome == TemporalSamplingOutcome::LinearAdjacent {
                request.config.check_memory(scene_bytes)?;
                Some(read_scene(
                    plan.neighbor
                        .as_ref()
                        .ok_or("temporal plan lacks neighbor")?,
                    &base,
                )?)
            } else {
                None
            };
            let read_seconds = started.elapsed().as_secs_f64();
            let pending: Vec<_> = sites
                .into_iter()
                .filter(|site| {
                    !manifest.site_complete(
                        &request.outdir,
                        &domain,
                        &plan.anchor_time.to_rfc3339(),
                        &generation,
                        &site.id,
                    )
                })
                .collect();
            // Reserve the read-ahead even if its allocation has not started.
            // Atmospheres are shared; only polar/writer/PPI storage scales by site.
            let ahead_bytes = ahead
                .as_ref()
                .map(|(path, _, _)| scene_atmosphere_bytes(path))
                .transpose()?
                .unwrap_or(0);
            let reserved = resident_bytes
                .checked_add(ahead_bytes)
                .ok_or("radar memory estimate overflow")?;
            // C/X-band sites read their own band atmospheres (atmosphere_factor 6), so a request
            // that names one keeps radar-bands' one-site-at-a-time memory envelope.
            let site_limit = if request.config.atmosphere_factor() > 1 {
                1
            } else {
                rayon::current_num_threads()
            };
            site_batches(
                &pending,
                site_limit,
                |count| {
                    let extra = request
                        .config
                        .scan_memory_bytes()?
                        .checked_mul((count - 1) as u64)
                        .and_then(|n| n.checked_add(reserved))
                        .ok_or("radar memory estimate overflow")?;
                    request.config.check_memory(extra)
                },
                |site| {
                    let site_started = Instant::now();
                    let mut config = base.clone();
                    config.site_id = site.id.clone();
                    config.site_lat_deg = Some(site.lat);
                    config.site_lon_deg = Some(site.lon);
                    config.antenna_msl_m = site.height_m;
                    if let Some(width) = site.beamwidth_deg {
                        config.beam_width_deg = width as f32;
                    }
                    let band_fields;
                    let band_neighbor;
                    let (site_fields, site_neighbor) = if site.band != "S" {
                        config.scattering_band = Some(site.band.clone());
                        config.radar_frequency_mhz = match site.band.as_str() {
                            "C" => 5604,
                            "X" => 9003,
                            _ => return Err("unknown radar site band".into()),
                        };
                        config.dual_pol = true;
                        config.propagation = true;
                        band_fields = read_scene(&locator, &config)?;
                        band_neighbor = if let Some(next) =
                            plan.neighbor.as_ref().filter(|_| neighbor.is_some())
                        {
                            Some(read_scene(next, &config)?)
                        } else {
                            None
                        };
                        (&band_fields, band_neighbor.as_ref())
                    } else {
                        (&fields, neighbor.as_ref())
                    };
                    let mut volume = if let Some(site_neighbor) = site_neighbor {
                        build_synthetic_volume_reporting_temporal(
                            site_fields,
                            site_neighbor,
                            plan.anchor_time,
                            &config,
                            &|_| {},
                            &plan,
                        )?
                    } else {
                        try_build_synthetic_volume(site_fields, plan.anchor_time, &config)?
                    };
                    select_fields(&mut volume, &request.config.fields)?;
                    // Producer labels are consumer metadata; the forward operator remains unchanged.
                    volume.site.name = Some("WOOF simulated radar".into());
                    let mut fm301 = adapter::to_fm301(&volume)?;
                    if site.band != "S" {
                        let wavelength_mm = if site.band == "C" { 53.5 } else { 33.3 };
                        fm301.radar_parameters.frequency_hz = vec![299_792_458_000. / wavelength_mm];
                    }
                    let stem = plan.anchor_time.format("%Y%m%dT%H%M%SZ").to_string();
                    let directory = request
                        .outdir
                        .join("radar")
                        .join(&domain)
                        .join(&site.id)
                        .join(&generation);
                    // Every format is encoded and every PPI drawn at the same
                    // time; each writes its own file, listed in request order.
                    let (written, images) = rayon::join(
                        || {
                            request
                                .config
                                .formats
                                .par_iter()
                                .map(|format| {
                                    let suffix = match format.as_str() {
                                        "level2" => "ar2v",
                                        "cfradial1" => "cfradial1.nc",
                                        "cfradial2" => "cfradial2.nc",
                                        "odim" => "odim.h5",
                                        _ => unreachable!(),
                                    };
                                    let path = directory.join(format!("{stem}.{suffix}"));
                                    let (bytes, report) = adapter::encode(&fm301, format)?;
                                    manifest::atomic_bytes(&path, &bytes)?;
                                    Ok((
                                        manifest::artifact(&request.outdir, &path, format)?,
                                        report,
                                    ))
                                })
                                .collect::<Result<Vec<_>, String>>()
                        },
                        || {
                            ppi::render(
                                &request.outdir,
                                &domain,
                                &generation,
                                &volume,
                                PPI_TILTS,
                                PPI_WIDTH,
                                PPI_HEIGHT,
                            )
                        },
                    );
                    let mut files = Vec::new();
                    let mut writer_reports = BTreeMap::new();
                    for (format, (artifact, report)) in request.config.formats.iter().zip(written?)
                    {
                        files.push(artifact);
                        writer_reports.insert(format.clone(), report);
                    }
                    let images = images?;
                    let offsets = volume
                        .cuts
                        .iter()
                        .flat_map(|c| c.radials.iter())
                        .map(|r| r.time_offset_ms);
                    let last_ms = offsets.max().unwrap_or(0);
                    let mut fields_present: Vec<_> = volume
                        .cuts
                        .iter()
                        .flat_map(|c| c.moments.keys().map(|m| field_name(m).to_owned()))
                        .collect();
                    fields_present.sort();
                    fields_present.dedup();
                    let timing_used = match plan.outcome {
                        TemporalSamplingOutcome::Frozen => "history",
                        TemporalSamplingOutcome::LinearAdjacent => "linear_adjacent",
                        TemporalSamplingOutcome::HeldAnchor(_) => "held_anchor",
                    };
                    let elapsed = site_started.elapsed().as_secs_f64();
                    Ok(manifest::Volume {
                        domain: domain.clone(),
                        generation: generation.clone(),
                        expected_sites: expected_sites.clone(),
                        implementation: implementation.clone(),
                        site: manifest::Site {
                            id: site.id.clone(),
                            band: (site.band != "S").then(|| site.band.clone()),
                            latitude_deg: site.lat,
                            longitude_deg: site.lon,
                            antenna_height_msl_m: f64::from(
                                volume
                                    .site
                                    .elevation_m
                                    .ok_or("simulated radar lacks antenna height")?,
                            ),
                        },
                        valid_time: plan.anchor_time.to_rfc3339(),
                        scan_start: plan.anchor_time.to_rfc3339(),
                        scan_end: (plan.anchor_time
                            + chrono::Duration::milliseconds(i64::from(last_ms)))
                        .to_rfc3339(),
                        timing_requested: request.config.timing.clone(),
                        timing_used: timing_used.into(),
                        source_times: source_times.clone(),
                        source_history: source_history.clone(),
                        tilts_deg: volume.cuts.iter().map(|c| c.elevation_deg).collect(),
                        fields: fields_present,
                        config_sha256: config_hash.clone(),
                        reflectivity_source: site_fields.ref_source.into(),
                        dual_pol_status: site_fields.dual_pol_status.clone(),
                        writer_reports,
                        files,
                        images,
                        elapsed_seconds: elapsed,
                    })
                },
                |volume| {
                    let (site, elapsed) = (volume.site.id.clone(), volume.elapsed_seconds);
                    manifest.upsert(volume);
                    manifest.save(&request.outdir)?;
                    processed += 1;
                    println!(
                        "{}",
                        serde_json::json!({"event":"volume_committed","domain":domain,"site":site,
                    "valid_time":plan.anchor_time.to_rfc3339(),"read_seconds":read_seconds,"volume_seconds":elapsed})
                    );
                    Ok(())
                },
            )?;
        }
    }
    let _ = processed;
    loops::update(&request.outdir, &mut manifest)?;
    Ok(manifest::Manifest::path(&request.outdir))
}

fn source_description(scene: &WrfSceneLocator, hashes: &BTreeMap<PathBuf, String>) -> String {
    format!(
        "{}#{}:sha256={}",
        scene.path.file_name().unwrap_or_default().to_string_lossy(),
        scene.time_index,
        hashes[&scene.path]
    )
}

fn run() -> Result<(), String> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.len()==1 && args[0]=="--site-catalog" {
        let ready:serde_json::Value=serde_json::from_str(include_str!("../data/sites.json")).map_err(|e|e.to_string())?;
        let reference=bowecho_simradar::all_sites().into_iter().map(|site|serde_json::json!({
            "id":site.site.settings_key(),"label":site.label,"country":site.country,
            "band":site.band,"coordinates":site.lat_lon,"antenna_height_msl_m":null,
        })).collect::<Vec<_>>();
        println!("{}",serde_json::json!({"schema":"simulated-radar.site-catalog/v1",
            "simulation_sites":ready,"reference_sites":reference,
            "reference_usage":"Reference rows without antenna height require a custom RadarSite with antenna height MSL; unknown bands require an explicit verified band."}));
        return Ok(());
    }
    if args.len() == 1 && args[0] == "--capabilities" {
        println!("{}", serde_json::json!({
            "schema":"rw-simradar.capabilities/v1",
            "request_abi":request::ABI,
            "canonical_columns_abi":atmosphere::ABI,
            "resource_estimate_schema":resources::SCHEMA,
            "input_validation":"full-columns/v1",
            "memory_policy":"checked-estimate-within-host-and-cgroup-headroom/v1",
            "ppi_fields":ppi::PPI_FIELDS,
            "color_tables":rustwx_render::RadarColorSet::NAMES,
            "bands":["S","C","X"],
            "band_wavelengths_mm":{"S":107.068735,"C":53.5,"X":33.3},
            "attenuation_bands":["C","X"],
            "finite_particle_microphysics":[8,9,18,50,51],
            "s_band_operator":"legacy bulk path, unchanged",
            "site_count_cap":null
        }));
        return Ok(());
    }
    if args.len() == 4 && args[0] == "--canonical-atmosphere" && args[2] == "--out" {
        atmosphere::convert(Path::new(&args[1]), Path::new(&args[3]))?;
        println!(
            "{}",
            serde_json::json!({"event":"canonical_atmosphere_committed", "path":args[3]})
        );
        return Ok(());
    }
    if args.len() == 1 && args[0] == "--abi" {
        println!("{}", request::ABI);
        return Ok(());
    }
    if args.len() == 1 && args[0] == "--canonical-abi" {
        println!("{}", atmosphere::ABI);
        return Ok(());
    }
    if args.len() == 1 && (args[0] == "--help" || args[0] == "-h") {
        println!(
            "{}\nrw_simradar --capabilities\nrw_simradar --estimate REQUEST.json schema=simulated-radar.resources/v1\nrw_simradar --canonical-atmosphere SOURCE.nc --out OUTPUT.nc schema=native-atmosphere.columns/v1",
            request::ABI
        );
        return Ok(());
    }
    if args.len() != 2 || (args[0] != "--request" && args[0] != "--estimate") {
        return Err(format!("usage: {}", request::ABI));
    }
    // Rayon's global pool spans every core this process may use (or
    // RAYON_NUM_THREADS); the beam sampling, encoders and PPIs share it.
    let mut document: serde_json::Value =
        serde_json::from_slice(&std::fs::read(Path::new(&args[1])).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    if let Some(config) = document.get_mut("config").and_then(|v| v.as_object_mut()) {
        if config.contains_key("elevations_deg") && !config.contains_key("scan_strategy") {
            config.insert("scan_strategy".into(), serde_json::json!("custom"));
        }
        if config.get("scan_strategy").and_then(|v| v.as_str()) == Some("low_tilts")
            && !config.contains_key("elevations_deg")
        {
            config.insert("elevations_deg".into(), serde_json::json!([0.5, 0.9, 1.3]));
        }
    }
    let request: Request = serde_json::from_value(document).map_err(|e| e.to_string())?;
    if args[0] == "--estimate" {
        println!(
            "{}",
            serde_json::to_string_pretty(&resources::estimate(&request)?)
                .map_err(|e| e.to_string())?
        );
        return Ok(());
    }
    let manifest = process(request)?;
    println!(
        "{}",
        serde_json::json!({"event":"complete","manifest":manifest})
    );
    Ok(())
}
/// `GPUWM_BRIDGE_SOURCE_REV=<40-hex commit>`: the source revision this
/// binary was built from, embedded so the release cut can prove a staged
/// binary matches the commit being released by reading bytes alone.
/// `build.rs` injects the value; `main` references the constant so the
/// linker cannot discard it.
pub static GPUWM_BRIDGE_SOURCE_REV_STAMP: &str =
    concat!("GPUWM_BRIDGE_SOURCE_REV=", env!("GPUWM_BRIDGE_SOURCE_REV"));

fn main() -> std::process::ExitCode {
    let _ = std::hint::black_box(GPUWM_BRIDGE_SOURCE_REV_STAMP);
    match run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            std::process::ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn site_batches_preserve_order_and_encoded_bytes_with_nested_work() {
        use rayon::prelude::*;
        use crate::adapter::radar_model::{
            Field, FieldData, FieldName, FloatCoding, Sweep, SweepMode, Volume,
        };
        let sites = ["KMQT", "KAPX", "KGRB", "KDTX", "KLOT"];
        let scan = |site: &&str| {
            let time = chrono::DateTime::parse_from_rfc3339("2026-10-29T01:35:00Z")
                .unwrap()
                .with_timezone(&chrono::Utc);
            let mut volume = Volume::new(*site, time);
            volume.location.latitude_deg = Some(46.5);
            volume.location.longitude_deg = Some(-87.5);
            volume.location.altitude_m = Some(450.0);
            let mut sweep = Sweep::new(0, SweepMode::AzimuthSurveillance, 0.5);
            sweep.push_ray(0.0, 0.0, 0.5);
            sweep.push_ray(1.0, 180.0, 0.5);
            let geometry = sweep.attach_geometry(0.0, 250.0, 2).unwrap();
            sweep
                .add_field(Field::new(
                    FieldName::parse("DBZH"),
                    geometry,
                    2,
                    FieldData::F32 {
                        values: vec![10.0, 20.0, 30.0, 40.0],
                        coding: FloatCoding::default(),
                    },
                ))
                .unwrap();
            volume.sweeps.push(sweep);
            volume.seal().unwrap();
            let formats = ["level2", "cfradial2"];
            let bytes = formats
                .par_iter()
                .map(|format| crate::adapter::encode(&volume, format))
                .collect::<Result<Vec<_>, _>>()?;
            Ok((site.to_string(), bytes))
        };
        let serial = sites
            .iter()
            .map(scan)
            .collect::<Result<Vec<_>, String>>()
            .unwrap();
        for threads in [1, 2, 4] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            let mut parallel = Vec::new();
            pool.install(|| {
                crate::site_batches(
                    &sites,
                    threads,
                    |count| {
                        if count <= 2 {
                            Ok(())
                        } else {
                            Err("memory limit".into())
                        }
                    },
                    scan,
                    |result| {
                        parallel.push(result);
                        Ok(())
                    },
                )
            })
            .unwrap();
            assert_eq!(parallel, serial);
        }
    }

    #[test]
    fn site_batches_obey_admission_and_commit_successes_before_an_error() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let calls = AtomicUsize::new(0);
        let mut committed = Vec::new();
        let error = crate::site_batches(
            &[0, 1, 2, 3],
            4,
            |count| {
                if count <= 2 {
                    Ok(())
                } else {
                    Err("memory".into())
                }
            },
            |&site| {
                calls.fetch_add(1, Ordering::SeqCst);
                if site == 1 {
                    Err("bad site".into())
                } else {
                    Ok(site)
                }
            },
            |site| {
                committed.push(site);
                Ok(())
            },
        )
        .unwrap_err();
        assert_eq!(error, "bad site");
        assert_eq!(committed, [0]);
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        let refused = crate::site_batches(
            &[0],
            4,
            |_| Err("no headroom".into()),
            |_| -> Result<(), String> { panic!("unadmitted allocation") },
            |_| Ok(()),
        );
        assert_eq!(refused.unwrap_err(), "no headroom");
        crate::site_batches::<i32, ()>(&[], 4, |_| panic!("empty batch"), |_| Ok(()), |_| Ok(()))
            .unwrap();
    }

    #[test]
    fn unwritten_reflectivity_ring_is_missing_and_echoes_stay() {
        // Two levels of a 4 x 3 grid as WRF writes REFL_10CM: interior cells
        // at the scheme floor or an echo, the ring at the initial 0, except
        // one ring cell an echo reached (12.5) and one at the floor (-35).
        let (nx, ny, nz) = (4, 3, 2);
        let mut dbz = vec![0.0f32; nx * ny * nz];
        for k in 0..nz {
            dbz[k * nx * ny + nx + 1] = -35.0;
            dbz[k * nx * ny + nx + 2] = 22.0;
        }
        dbz[3] = 12.5; // level 0, south row, east corner
        dbz[nx * ny + nx] = -35.0; // level 1, west column, middle row
        let marked = crate::unwritten_reflectivity_ring(&mut dbz, nx, ny, nz);
        assert_eq!(marked, 2 * 10 - 2);
        for k in 0..nz {
            let plane = &dbz[k * nx * ny..(k + 1) * nx * ny];
            assert_eq!(plane[nx + 1], -35.0);
            assert_eq!(plane[nx + 2], 22.0);
        }
        assert_eq!(dbz[3], 12.5);
        assert_eq!(dbz[nx * ny + nx], -35.0);
        assert!(dbz[0].is_nan() && dbz[nx * ny - 1].is_nan() && dbz[nx * ny + nx + 3].is_nan());
        // A shape that does not match the array is left alone.
        let mut other = vec![0.0f32; 5];
        let marked = crate::unwritten_reflectivity_ring(&mut other, nx, ny, nz);
        assert_eq!(marked, 0);
        assert!(other.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn ppi_fields_carry_the_manifest_field_names() {
        for &field in crate::ppi::PPI_FIELDS {
            let drawn = crate::ppi::drawn_field(field);
            assert_eq!(crate::field_name(&drawn.moment), field);
        }
    }
}
