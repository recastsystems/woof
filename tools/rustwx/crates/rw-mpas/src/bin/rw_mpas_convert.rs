//! Convert MPAS history frames into wrfout-shaped frames the renderer takes.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use rw_mpas::composite::{CompositePlan, CompositeSource, COMPOSITE_CONTEXT_FACTOR};
use rw_mpas::convert::{
    convert_frame, convert_frame_composite, convert_frame_remapped, CompositeGather,
    CompositeLayer, ConvertOptions, TerrainAuthority,
};
use rw_mpas::remap::{read_mesh_topology, ProductRemap};
use rw_mpas::history::{
    assert_mesh_agrees, read_history_on_mesh_with_field_set, read_init_fields, MpasFrame,
    Timestamp,
};
use rw_mpas::sha256_file;
use rw_mpas::weights::{
    build_weights, read_boundary_rings, read_mesh_cell_spacing, read_mesh_coordinates,
    ON_MESH_SPACING_FACTOR,
};
use rw_mpas::window::{Window, GLOBAL_MESH_WINDOW, REGION_LATLON_SPEC_PREFIX, REGION_SPEC_PREFIX};
use rw_store::netcdf_classic::NcFormat;

const USAGE: &str = "\
usage: rw_mpas_convert --history FILE... --mesh FILE --out-dir DIR
                       [--init FILE] [--mesh-sha256 HEX] [--init-sha256 HEX]
                       [--window focus|global|global-mesh|mesh|composite|region:...|region-latlon:...|latlon:...|lambert:...]
                       [--terrain-authority FIXED_CF_FRAME]
                       [--compose SOURCES.json] [--compose-base-only]
                       [--field-set full|surface]
                       [--simulation-start YYYY-MM-DD_HH:MM:SS]
                       [--prefix NAME] [--format cdf2|cdf5] [--clobber]
                       [--json FILE] [--model-label NAME] [--grid-label TEXT]
                       [--remap nearest|product] [--keep-boundary-zone]
       rw_mpas_convert --help | --abi

windows:
  focus   a fixed 240x150 Lambert box at 22 km over the CONUS. Frozen
          geometry: its weights digests are already recorded in evidence.
  global  a fixed 0.25 degree lat-lon overview of the whole sphere.
  global-mesh  DERIVED from --mesh: the whole sphere on a lat-lon grid at
          the mesh's own median spacing (rounded down, never coarser), so a
          15 km mesh draws at 15 km and a 60 km mesh at 60 km.
  region:CLAT,CLON,WIDTH_KM,HEIGHT_KM[,TRUELAT1,TRUELAT2]
          DERIVED from --mesh: a Lambert box of that ground extent (lat-lon
          near the equator) whose dx is the median spacing of the mesh cells
          inside it, rounded down. The caller names the ground; the mesh
          names the resolution.
  region-latlon:CLAT,CLON,WIDTH_KM,HEIGHT_KM
          the same box at the mesh's own spacing, held on a regular lat-lon
          grid at any latitude. Lat-lon frames carry CF 1-D coordinates
          (lat, lon, Time, bottom_top, bottom_top_stag, soil_layers_stag),
          so this is the window an ArWen domain reads as its parent.
  latlon:SOUTH,NORTH,WEST,EAST,SPACING_DEG
          explicit fixed grid shared across independently meshed segments.
          Spacing must partition both extents. --terrain-authority FRAME
          binds HGT from an earlier CF frame on the identical grid; other
          atmospheric columns and absolute heights remain unchanged.
  lambert:CLAT,CLON,DX_M,NX,NY,TRUELAT1,TRUELAT2,STAND_LON
          an explicit WRF-style Lambert grid, for landing on another
          model's own grid point for point.
  mesh    DERIVED from --mesh: the cells within 2x the mesh's minimum
          spacing are its refined region, and the window is centred on
          them, at their median spacing, covering their extent. A mesh
          refined anywhere renders its own fine region with this, and no
          new box has to be added for the next placement.
  composite  DERIVED from --compose: one frame covering every fine region
          in the source list, at the finest spacing among them, pulled out
          to 3x their extent so the coarse field around them is visible.

--compose SOURCES.json composes several runs of the same hours into one
frame -- the coarse run everywhere, each fine run inside the ground it
actually resolves:

  {\"base\":     {\"label\": \"coarse\", \"mesh\": \"g96.static.nc\",
                \"history\": [\"...06.nc\", \"...09.nc\"]},
   \"overlays\": [{\"label\": \"s01\", \"mesh\": \"s01.static.nc\",
                 \"history\": [\"...06.nc\", \"...09.nc\"]}]}

A point takes the finest source whose refined region it falls inside, and
the base everywhere else. The change of source is a hard switch: nothing is
blended, and the frame carries COMPOSITE_SOURCE and COMPOSITE_DX_KM so the
seam can be drawn and audited.

--model-label NAME writes NAME into each frame as GPUWM_MODEL_LABEL, the
model the renderer names in a plot's metadata row in place of the generic
WRF identity every wrfout-shaped frame imports under (at most 32 characters,
no '|').

Every single-source frame also carries GPUWM_GRID_LABEL, the grid a plot's
headline names: the model label (or 'mesh') and the spacing the mesh is
known by, read from the mesh itself -- 'hex 15 km', 'hex 60-3 km' -- in
place of WRF's 'd01' nest token and the resample grid's spacing.

--grid-label TEXT (single-source only) names the picture instead: TEXT is
written as GPUWM_GRID_LABEL and as the receipt's grid_label, and the label
the mesh alone would have given is kept in the receipt as mesh_grid_label.
'{spacing}' in TEXT becomes the spacing of the mesh cells this window
samples, finest first ('Core zoom {spacing}' -> 'Core zoom 8-38 km'; one
value on a uniform mesh).  At most 32 characters, no '|', no control
character.  Without it every frame is written exactly as before.

--keep-boundary-zone draws a limited-area mesh's boundary rings.  By default
the window points whose nearest cell has bdyMaskCell 1-7 (rings 1-5 relaxed
toward the boundary file, 6-7 overwritten by it every step) are left blank
like points off the mesh, as operational limited-area products leave them;
stdout's BOUNDARY_ZONE line and the receipt's boundary_zone say how many
points that hid.  A mesh without bdyMaskCell (every global mesh) is
unchanged.

--remap product (single-source only) carries each field the way its quantity
needs: linear (barycentric on the mesh's own Delaunay triangles) for state,
first-order area-conservative for accumulated precipitation, nearest cell for
categories and soil.  Each frame records the method per variable
(MPAS_RESAMPLE) and the remap digest; stdout adds REMAP and per-frame BUDGET
lines comparing the window integral of RAINC+RAINNC with the mesh's own.
--remap nearest (the default) is the frozen nearest-cell map.

A split-stream history (RRFS-MPAS writes its history with no latCell/lonCell
and its 2 m, 10 m, precipitation and reflectivity fields to diag.*.nc) takes
its cell centres from --mesh, once the cell counts and the static fields both
files carry (shdmin, shdmax) agree, and reads the fields it lacks from the
diag file of the same time beside it (history.T.nc -> diag.T.nc).  The frame
names both files (MPAS_CELL_CENTRES_*, MPAS_DIAG_*).  A history that carries
its own centres, as MPAS-A writes it, keeps them; a mapped field it lacks is
read from its diag file of the same time when one is beside it (MPAS-A writes
refl10cm_max, which becomes REFC, only to its diagnostics stream), and the
frame names that file (MPAS_DIAG_*).  A history that carries every field
converts exactly as before.

--jobs N (single-source only) converts N history frames at a time.  The
default converts as many at once as there are frames, cores and host memory
for (each frame in flight holds its history in memory; the bound is half the
available memory over twice the history file's size).  A frame is read,
remapped, written and hashed on its own, so frames are independent; stdout
lines and the JSON receipt keep the order of --history whatever N is, and
every output file is the same bytes.

--compose-base-only frames the same window from the same sources and then
converts the BASE alone, so the composite has a coarse counterpart over
exactly the same ground to be compared against.";

/// The window name that is derived rather than served from the catalogue.
const MESH_WINDOW: &str = "mesh";

/// The tape contract: the output schema name and the stdout progress
/// tokens, which is the literal a stale build gets wrong. It stays `v1`
/// across the derived window because the stdout grammar is untouched --
/// `--window mesh` reports what it derived on STDERR, the way the mesh
/// reader already advises about inferred angle units. `gpuwm.bridges` and
/// `gpuwm.mpas_mesh` both carry this literal; a change here is a change
/// there.
///
/// It also names the flags a caller may pass that an older build refuses
/// (`--remap`, with its REMAP and BUDGET stdout lines, and `--jobs`).  THE
/// BREAKAGE THIS PREVENTS: the literal did not move when 2.9.0 added them,
/// so a converter staged before 2.9.0 in `~/.gpuwm/bridges` passed `gpuwm
/// doctor` and `fetch-bridges` as current, and the gpuwm-hex product stage
/// then died on `rw_mpas_convert: unknown flag --remap` with no maps drawn.
const ABI: &str = "gpuwm-rw-mpas-convert-v1\tCONVERTED\tWINDOW\tWEIGHTS\tFINISHED\tREMAP\tBUDGET\t--remap nearest|product\t--jobs N\t--terrain-authority FILE\t--window latlon:S,N,W,E,SPACING";

/// `GPUWM_BRIDGE_SOURCE_REV=<40-hex commit>`: the source revision this
/// binary was built from, embedded so the gpuwm release cut can prove a
/// staged bridge matches the commit being released by reading bytes alone
/// (`tools/build_bridge_bundle.py pin --source-rev`).  `build.rs` injects
/// the value; `main` references the constant so the linker cannot discard
/// it.
pub static GPUWM_BRIDGE_SOURCE_REV_STAMP: &str =
    concat!("GPUWM_BRIDGE_SOURCE_REV=", env!("GPUWM_BRIDGE_SOURCE_REV"));

struct Args {
    history: Vec<PathBuf>,
    mesh: Option<PathBuf>,
    mesh_sha256: Option<String>,
    init: Option<PathBuf>,
    init_sha256: Option<String>,
    out_dir: Option<PathBuf>,
    window: String,
    field_set: String,
    simulation_start: Option<String>,
    prefix: String,
    format: NcFormat,
    clobber: bool,
    json: Option<PathBuf>,
    compose: Option<PathBuf>,
    compose_base_only: bool,
    model_label: Option<String>,
    /// `--grid-label`: the caller's headline grid name, `{spacing}` not yet filled.
    grid_label: Option<String>,
    remap: String,
    jobs: usize,
    terrain_authority: Option<PathBuf>,
    keep_boundary_zone: bool,
}

/// One run named in a `--compose` source list.
#[derive(Debug, serde::Deserialize)]
struct ComposeEntry {
    label: String,
    mesh: PathBuf,
    history: Vec<PathBuf>,
}

#[derive(Debug, serde::Deserialize)]
struct ComposeSpec {
    base: ComposeEntry,
    #[serde(default)]
    overlays: Vec<ComposeEntry>,
}

/// One source, read and resampled onto the shared window.
struct LoadedSource {
    label: String,
    source: CompositeSource,
    frames: Vec<MpasFrame>,
}

fn parse() -> Result<Args, String> {
    let mut args = Args {
        history: Vec::new(),
        mesh: None,
        mesh_sha256: None,
        init: None,
        init_sha256: None,
        out_dir: None,
        window: "focus".to_string(),
        field_set: "full".to_string(),
        simulation_start: None,
        prefix: "wrfout_d01".to_string(),
        format: NcFormat::Offset64,
        clobber: false,
        json: None,
        compose: None,
        compose_base_only: false,
        model_label: None,
        grid_label: None,
        remap: "nearest".to_string(),
        // 0 = not given: sized from the frames, cores and memory in run().
        jobs: 0,
        terrain_authority: None,
        keep_boundary_zone: false,
    };
    let mut raw = std::env::args().skip(1).peekable();
    while let Some(flag) = raw.next() {
        let mut value = || -> Result<String, String> {
            raw.next()
                .ok_or_else(|| format!("{flag} needs a value"))
        };
        match flag.as_str() {
            "--history" => {
                while let Some(next) = raw.peek() {
                    if next.starts_with("--") {
                        break;
                    }
                    args.history.push(PathBuf::from(raw.next().unwrap()));
                }
                if args.history.is_empty() {
                    return Err("--history needs at least one file".to_string());
                }
            }
            "--mesh" => args.mesh = Some(PathBuf::from(value()?)),
            "--mesh-sha256" => args.mesh_sha256 = Some(value()?),
            "--init" => args.init = Some(PathBuf::from(value()?)),
            "--terrain-authority" => args.terrain_authority = Some(PathBuf::from(value()?)),
            "--init-sha256" => args.init_sha256 = Some(value()?),
            "--out-dir" => args.out_dir = Some(PathBuf::from(value()?)),
            "--window" => args.window = value()?,
            "--field-set" => args.field_set = value()?,
            "--simulation-start" => args.simulation_start = Some(value()?),
            "--prefix" => args.prefix = value()?,
            "--format" => {
                args.format = match value()?.as_str() {
                    "cdf2" => NcFormat::Offset64,
                    "cdf5" => NcFormat::Data64,
                    other => return Err(format!("--format '{other}' is not cdf2 or cdf5")),
                }
            }
            "--clobber" => args.clobber = true,
            "--keep-boundary-zone" => args.keep_boundary_zone = true,
            "--json" => args.json = Some(PathBuf::from(value()?)),
            "--compose" => args.compose = Some(PathBuf::from(value()?)),
            "--compose-base-only" => args.compose_base_only = true,
            "--model-label" => args.model_label = Some(checked_model_label(&value()?)?),
            // Checked again once `{spacing}` is filled (caller_grid_label); refused here when
            // it can never pass, before any mesh is read.
            "--grid-label" => {
                let text = value()?;
                rw_mpas::convert::caller_grid_label(&text, Some("0 km"))?;
                args.grid_label = Some(text);
            }
            "--remap" => {
                args.remap = value()?;
                if !matches!(args.remap.as_str(), "nearest" | "product") {
                    return Err(format!("--remap '{}' is not nearest or product", args.remap));
                }
            }
            "--jobs" => {
                args.jobs = value()?
                    .parse()
                    .ok()
                    .filter(|n| (1..=64).contains(n))
                    .ok_or("--jobs takes a whole number from 1 to 64")?;
            }
            other => return Err(format!("unknown flag {other}")),
        }
    }
    Ok(args)
}

/// `--model-label`, refused here rather than written into a frame the
/// renderer would then decline: the row's limits are the renderer's
/// (`rustwx_products::shared_context::checked_model_label`).
fn checked_model_label(raw: &str) -> Result<String, String> {
    let label = raw.split_whitespace().collect::<Vec<_>>().join(" ");
    if label.is_empty() {
        return Err("--model-label needs a name".to_string());
    }
    if label.chars().count() > 32 {
        return Err(format!("--model-label {label:?} is longer than 32 characters"));
    }
    if label.chars().any(|ch| ch.is_control() || ch == '|') {
        return Err(format!(
            "--model-label {label:?} holds a control character or '|', the plot metadata row's separator"
        ));
    }
    Ok(label)
}

/// `{"min": .., "median": .., "max": ..}` of a mesh's per-cell spacing, in
/// km, for the receipt: the numbers the grid label and a derived window's
/// dx were read from.
fn spacing_summary_km(spacing_m: &[f64]) -> String {
    let mut sorted: Vec<f64> = spacing_m.iter().copied().filter(|v| v.is_finite()).collect();
    if sorted.is_empty() {
        return "null".to_string();
    }
    sorted.sort_by(|a, b| a.total_cmp(b));
    format!(
        "{{\"min\": {:.4}, \"median\": {:.4}, \"max\": {:.4}}}",
        sorted[0] / 1000.0,
        sorted[sorted.len() / 2] / 1000.0,
        sorted[sorted.len() - 1] / 1000.0
    )
}

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

/// Read one source's mesh, resample it onto `window`, and read its frames.
fn load_source(
    entry: &ComposeEntry,
    window: &Window,
    window_name: &str,
    is_base: bool,
    field_set: &str,
) -> Result<LoadedSource, String> {
    if entry.history.is_empty() {
        return Err(format!(
            "composite source '{}' names no history file, so it has nothing to contribute at \
             any hour",
            entry.label
        ));
    }
    let mesh = read_mesh_coordinates(&entry.mesh, None).map_err(|e| e.to_string())?;
    let spacing = read_mesh_cell_spacing(&entry.mesh).map_err(|e| e.to_string())?;
    if spacing.spacing_metres.len() != mesh.n_cells() {
        return Err(format!(
            "composite source '{}' mesh {} carries {} cell centre(s) but {} spacing value(s)",
            entry.label,
            entry.mesh.display(),
            mesh.n_cells(),
            spacing.spacing_metres.len()
        ));
    }
    let weights = build_weights(&mesh, window_name, window).map_err(|e| e.to_string())?;
    let source = if is_base {
        CompositeSource::base(entry.label.clone(), &mesh, weights, spacing.spacing_metres)
    } else {
        CompositeSource::overlay(entry.label.clone(), &mesh, weights, spacing.spacing_metres)
    }
    .map_err(|e| e.to_string())?;
    let mut frames = Vec::with_capacity(entry.history.len());
    for history in &entry.history {
        let frame =
            read_history_on_mesh_with_field_set(history, None, None, None, Some(&mesh), field_set)
                .map_err(|e| e.to_string())?;
        assert_mesh_agrees(&frame, &mesh, 1.0e-4).map_err(|e| e.to_string())?;
        frames.push(frame);
    }
    Ok(LoadedSource {
        label: entry.label.clone(),
        source,
        frames,
    })
}

/// Read the refined cells of every overlay and size one window over all of
/// them.  It reads the meshes itself rather than taking loaded sources
/// because the weights cannot be built until the window exists, and the
/// window is what this returns.
fn composite_window(overlays: &[&ComposeEntry]) -> Result<Window, String> {
    if overlays.is_empty() {
        return Err("--window composite sizes itself to the fine regions in --compose; the \
                    source list names no overlay, so there is no fine region to frame and \
                    nothing to composite. Use --window global for the base alone"
            .to_string());
    }
    let mut lat = Vec::new();
    let mut lon = Vec::new();
    let mut spacing = Vec::new();
    for entry in overlays {
        let mesh = read_mesh_coordinates(&entry.mesh, None).map_err(|e| e.to_string())?;
        let spacings = read_mesh_cell_spacing(&entry.mesh).map_err(|e| e.to_string())?;
        let refined = Window::refined_region(&spacings.spacing_metres).map_err(|e| {
            format!(
                "composite source '{}' has no refined region to frame: {e}",
                entry.label
            )
        })?;
        for cell in refined {
            lat.push(mesh.latitude_degrees[cell]);
            lon.push(mesh.longitude_degrees[cell]);
            spacing.push(spacings.spacing_metres[cell]);
        }
    }
    Window::composite_focus(&lat, &lon, &spacing, COMPOSITE_CONTEXT_FACTOR)
        .map_err(|e| e.to_string())
}

fn run_compose(args: Args, spec_path: &Path) -> Result<(), String> {
    if args.remap != "nearest" {
        return Err("--remap product interpolates across one mesh; a composite switches \
                    source point by point, so it keeps the nearest-cell map"
            .to_string());
    }
    if !args.history.is_empty() || args.mesh.is_some() {
        return Err(
            "--compose owns the mesh and history list; passing --mesh or --history as well \
             would leave two source lists that can disagree about which run is the base"
                .to_string(),
        );
    }
    let out_dir = args.out_dir.clone().ok_or("--out-dir is required")?;
    let simulation_start = match &args.simulation_start {
        Some(text) => Some(Timestamp::parse(text).map_err(|e| e.to_string())?),
        None => None,
    };
    let bytes =
        std::fs::read(spec_path).map_err(|e| format!("read {}: {e}", spec_path.display()))?;
    let spec: ComposeSpec = serde_json::from_slice(&bytes)
        .map_err(|e| format!("parse {}: {e}", spec_path.display()))?;

    let started = std::time::Instant::now();
    let overlay_refs: Vec<&ComposeEntry> = spec.overlays.iter().collect();
    let window = match args.window.as_str() {
        "composite" => composite_window(&overlay_refs)?,
        MESH_WINDOW => {
            return Err(
                "--window mesh derives from ONE mesh's refined region; a composite has \
                 several. Use --window composite, which frames all of them at once"
                    .to_string(),
            )
        }
        other => Window::named(other).map_err(|e| e.to_string())?,
    };
    eprintln!(
        "rw_mpas_convert: advisory: composite window: {}",
        window.description()
    );

    let base = load_source(&spec.base, &window, &args.window, true, &args.field_set)?;
    let mut overlays = Vec::with_capacity(spec.overlays.len());
    for entry in &spec.overlays {
        overlays.push(load_source(entry, &window, &args.window, false, &args.field_set)?);
    }
    // --compose-base-only builds the SAME window from the same overlays and
    // then converts the base alone. It exists so the composite can be
    // compared against what the coarse run shows over the identical ground:
    // without it the only coarse counterpart is on a different frame, and a
    // comparison across two frames cannot separate what the fine data added
    // from what the reframing did.
    let overlay_sources: Vec<&CompositeSource> = if args.compose_base_only {
        Vec::new()
    } else {
        overlays.iter().map(|o| &o.source).collect()
    };
    let plan = CompositePlan::build(&base.source, &overlay_sources).map_err(|e| e.to_string())?;

    println!(
        "WINDOW\t{}\t{}x{}\tcells={}",
        args.window,
        plan.ny,
        plan.nx,
        base.source.weights.n_cells
    );
    for (index, label) in plan.labels.iter().enumerate() {
        println!(
            "COMPOSITE\t{index}\t{label}\tpoints={}\tshare={:.4}",
            plan.counts[index],
            plan.counts[index] as f64 / plan.points() as f64
        );
    }
    println!(
        "COMPOSITE\tseam\tboundary_points={}",
        plan.boundary_slots().len()
    );

    // Only hours every source carries can be composed. A frame missing
    // from one overlay is dropped by name rather than composed without it:
    // silently falling back to the base for that hour would put a coarse
    // patch in the middle of a fine sequence with nothing saying so.
    let mut shared: Vec<Timestamp> = Vec::new();
    for frame in &base.frames {
        let mut everywhere = true;
        for overlay in &overlays {
            if !overlay
                .frames
                .iter()
                .any(|f| f.valid_time.label() == frame.valid_time.label())
            {
                everywhere = false;
                eprintln!(
                    "rw_mpas_convert: advisory: dropping {} -- composite source '{}' has no \
                     frame at that time",
                    frame.valid_time.label(),
                    overlay.label
                );
            }
        }
        if everywhere {
            shared.push(frame.valid_time);
        }
    }
    if shared.is_empty() {
        return Err(format!(
            "no valid time is carried by all {} source(s); the base has {} frame(s) and none \
             of them line up",
            overlays.len() + 1,
            base.frames.len()
        ));
    }

    let mut records: Vec<String> = Vec::new();
    for valid in &shared {
        let base_frame = base
            .frames
            .iter()
            .find(|f| f.valid_time.label() == valid.label())
            .expect("shared times come from the base");
        let mut layers: Vec<CompositeLayer<'_>> = vec![CompositeLayer {
            label: &base.label,
            frame: base_frame,
            weights: &base.source.weights,
        }];
        if !args.compose_base_only {
            for overlay in &overlays {
                let frame = overlay
                    .frames
                    .iter()
                    .find(|f| f.valid_time.label() == valid.label())
                    .expect("shared times are carried by every source");
                layers.push(CompositeLayer {
                    label: &overlay.label,
                    frame,
                    weights: &overlay.source.weights,
                });
            }
        }
        let gather = CompositeGather {
            plan: &plan,
            layers: &layers,
        };
        let stamp = valid.label().replace(':', "_");
        let destination = Path::new(&out_dir).join(format!("{}_{stamp}", args.prefix));
        let options = ConvertOptions {
            simulation_start,
            field_set: args.field_set.clone(),
            format: args.format,
            clobber: args.clobber,
            model_label: args.model_label.clone(),
            ..ConvertOptions::default()
        };
        let emitted = convert_frame_composite(
            base_frame,
            &base.source.weights,
            Some(&gather),
            &destination,
            &options,
        )
        .map_err(|e| e.to_string())?;
        println!(
            "CONVERTED\t{}\t{:.3}s\t{} bytes\t{} written\t{} absent",
            emitted.path.display(),
            emitted.seconds,
            emitted.bytes,
            emitted.written.len(),
            emitted.absent.len()
        );
        for stat in &emitted.seam {
            println!(
                "SEAM\t{}\t{}\tpoints={}\tmean={:.4}\tp95={:.4}\tmax={:.4}\tframe_range={:.4}",
                emitted.valid_time.label(),
                stat.field,
                stat.points,
                stat.mean_abs,
                stat.p95_abs,
                stat.max_abs,
                stat.frame_range
            );
        }
        records.push(format!(
            "{{\"seam\": [{}], \"output\": \"{}\", \"output_bytes\": {}, \"valid_time\": \"{}\", \"convert_seconds\": {}, \"written_variables\": [{}], \"absent_wrf_fields\": [{}]}}",
            emitted
                .seam
                .iter()
                .map(|s| s.json())
                .collect::<Vec<_>>()
                .join(", "),
            json_escape(&emitted.path.display().to_string()),
            emitted.bytes,
            emitted.valid_time.iso(),
            emitted.seconds,
            emitted
                .written
                .iter()
                .map(|n| format!("\"{n}\""))
                .collect::<Vec<_>>()
                .join(", "),
            emitted
                .absent
                .iter()
                .map(|n| format!("\"{n}\""))
                .collect::<Vec<_>>()
                .join(", "),
        ));
    }

    if let Some(path) = &args.json {
        let body = format!(
            "{{\n  \"schema\": \"{}\",\n  \"engine\": \"rw-mpas {} (rust)\",\n  \"window\": \"{}\",\n  \"window_spec\": {},\n  \"composite\": {},\n  \"resample_method\": \"{}\",\n  \"field_set\": \"{}\",\n  \"frames\": [\n    {}\n  ]\n}}\n",
            rw_mpas::convert::BRIDGE_SCHEMA,
            env!("CARGO_PKG_VERSION"),
            args.window,
            window.spec_json(),
            plan.spec_json(),
            rw_mpas::weights::RESAMPLE_METHOD,
            args.field_set,
            records.join(",\n    ")
        );
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        std::fs::write(path, body).map_err(|e| e.to_string())?;
        println!("REPORT\t{}", path.display());
    }
    println!(
        "FINISHED\tframes={}\telapsed_s={:.3}",
        shared.len(),
        started.elapsed().as_secs_f64()
    );
    Ok(())
}

fn run() -> Result<(), String> {
    let args = parse()?;
    if let Some(spec) = args.compose.clone() {
        if args.terrain_authority.is_some() {
            return Err("--terrain-authority binds a single-source segment export; compose segments after exporting each source".into());
        }
        if args.grid_label.is_some() {
            // A composite writes no grid label at all (each source keeps its own mesh), so a
            // caller's would be dropped without a word.
            return Err("--grid-label names one mesh's picture; a --compose frame switches \
                        meshes point by point and carries no grid label"
                .into());
        }
        return run_compose(args, &spec);
    }
    let mesh_path = args.mesh.ok_or("--mesh is required")?;
    let out_dir = args.out_dir.ok_or("--out-dir is required")?;
    if args.history.is_empty() {
        return Err("--history is required".to_string());
    }
    let simulation_start = match &args.simulation_start {
        Some(text) => Some(Timestamp::parse(text).map_err(|e| e.to_string())?),
        None => None,
    };

    let started = std::time::Instant::now();
    let mesh = read_mesh_coordinates(&mesh_path, args.mesh_sha256.as_deref())
        .map_err(|e| e.to_string())?;
    let mesh_seconds = started.elapsed().as_secs_f64();

    // The window is built AFTER the mesh, and the order is the point: a
    // derived window is a property OF the mesh, so it cannot be named before
    // the mesh has been read. The two fixed windows do not care when they are
    // built, and their geometry is untouched by the move.
    let window = if args.window == MESH_WINDOW {
        let spacing = read_mesh_cell_spacing(&mesh_path).map_err(|e| e.to_string())?;
        if spacing.spacing_metres.len() != mesh.n_cells() {
            return Err(format!(
                "mesh {} carries {} cell centre(s) but {} spacing value(s); a window derived \
                 from mismatched arrays would pair one cell's position with another cell's \
                 resolution",
                mesh_path.display(),
                mesh.n_cells(),
                spacing.spacing_metres.len()
            ));
        }
        let source = netcrust::File::open(&mesh_path).map_err(|e| e.to_string())?;
        let is_regional = if source.variable("bdyMaskCell").is_some() {
            let mask = source.read_f64("bdyMaskCell").map_err(|e| e.to_string())?;
            if mask.len() != mesh.n_cells()
                || mask.iter().any(|v| !v.is_finite() || *v < 0.0 || *v > 7.0 || v.fract() != 0.0)
            {
                return Err("bdyMaskCell must contain one finite integer ring 0..7 per cell".to_string());
            }
            mask.iter().any(|v| *v > 0.0)
        } else {
            false
        };
        let derive = if is_regional { Window::regional_focus } else { Window::mesh_focus };
        let derived = derive(
            &mesh.latitude_degrees,
            &mesh.longitude_degrees,
            &spacing.spacing_metres,
        )
        .map_err(|e| e.to_string())?;
        eprintln!(
            "rw_mpas_convert: advisory: --window mesh derived from {} using {}: {}",
            mesh_path.display(),
            spacing.source.label(),
            derived.description()
        );
        derived
    } else if args.window == GLOBAL_MESH_WINDOW
        || args.window.starts_with(REGION_SPEC_PREFIX)
        || args.window.starts_with(REGION_LATLON_SPEC_PREFIX)
    {
        let spacing = read_mesh_cell_spacing(&mesh_path).map_err(|e| e.to_string())?;
        if spacing.spacing_metres.len() != mesh.n_cells() {
            return Err(format!(
                "mesh {} carries {} cell centre(s) but {} spacing value(s); a window sized \
                 from mismatched arrays would pair one cell's position with another cell's \
                 resolution",
                mesh_path.display(),
                mesh.n_cells(),
                spacing.spacing_metres.len()
            ));
        }
        let derived = if args.window == GLOBAL_MESH_WINDOW {
            Window::global_at_mesh_spacing(&spacing.spacing_metres)
        } else {
            Window::region_at_mesh_spacing(
                &args.window,
                &mesh.latitude_degrees,
                &mesh.longitude_degrees,
                &spacing.spacing_metres,
            )
        }
        .map_err(|e| e.to_string())?;
        eprintln!(
            "rw_mpas_convert: advisory: --window {} derived from {} using {}: {}",
            args.window,
            mesh_path.display(),
            spacing.source.label(),
            derived.description()
        );
        derived
    } else {
        Window::named(&args.window).map_err(|e| e.to_string())?
    };

    let mut weights =
        build_weights(&mesh, &args.window, &window).map_err(|e| e.to_string())?;
    // A mesh that does not cover the sphere has a FOOTPRINT, and a window is
    // a rectangle: the points outside the footprint have to be marked, or the
    // nearest-cell resample fills them with the furthest cell it can reach.
    //
    // MEASURED (2026-08-26, r4.75.11020): the first weather field ever
    // rendered from a limited-area forecast came out with radial streaks of
    // the outermost ring's reflectivity across three corners of the frame, at
    // a maximum nearest-cell distance of 195.8 km on a 4.6 km mesh.  A reader
    // cannot tell a streak from a squall line.
    //
    // A GLOBAL mesh covers every window point by construction, so its
    // footprint marks nothing and its output bytes are unchanged -- which is
    // why this is unconditional rather than a regional switch.
    let footprint_spacing = read_mesh_cell_spacing(&mesh_path).map_err(|e| e.to_string())?;
    if footprint_spacing.spacing_metres.len() != mesh.n_cells() {
        return Err(format!(
            "mesh {} carries {} cell centre(s) but {} spacing value(s); a \
             footprint derived from mismatched arrays would mask one cell's \
             window points using another cell's resolution",
            mesh_path.display(),
            mesh.n_cells(),
            footprint_spacing.spacing_metres.len()
        ));
    }
    let off_mesh = weights
        .mark_footprint(&footprint_spacing.spacing_metres, ON_MESH_SPACING_FACTOR)
        .map_err(|e| e.to_string())?;
    // A limited-area mesh's boundary rings are the parent's state forced in,
    // not this model's forecast: hidden like the off-mesh points (before the
    // product remap is built, so every remap honours it), unless asked for.
    // Measured breakage: weights.rs NearestCellWeights::mask_boundary_zone.
    let boundary_zone = match read_boundary_rings(&mesh_path).map_err(|e| e.to_string())? {
        Some(rings) => {
            let zone_cells = rings.iter().filter(|&&r| r > 0).count();
            let hidden = if args.keep_boundary_zone {
                0
            } else {
                weights.mask_boundary_zone(&rings).map_err(|e| e.to_string())?
            };
            println!(
                "BOUNDARY_ZONE\tzone_cells={zone_cells}\thidden_points={hidden}\tkept={}",
                args.keep_boundary_zone
            );
            format!(
                "{{\"rings\": \"1-7\", \"zone_cells\": {zone_cells}, \"hidden_points\": {hidden}, \"kept\": {}}}",
                args.keep_boundary_zone
            )
        }
        None => "null".to_string(),
    };
    // The headline's grid name, from the mesh this run has just read: the
    // footprint pass already holds every cell's spacing.
    let mesh_grid_label =
        rw_mpas::convert::grid_label(args.model_label.as_deref(), &footprint_spacing.spacing_metres);
    // A caller's --grid-label names THIS picture instead (Project ruling, 2026-10-30: the
    // core picture of run_713c9e1913164e02 read `(hex 133-6.9 km)` under the
    // site's `Core zoom · 8 → 38 km`); `{spacing}` is the cells the window
    // samples, read after the footprint and boundary rings are marked so a
    // hidden point names no cell.
    let grid_label = match &args.grid_label {
        None => mesh_grid_label.clone(),
        Some(text) => {
            let sampled: Vec<usize> = weights
                .cell_index
                .iter()
                .enumerate()
                .filter(|(i, c)| **c >= 0 && !weights.off_mesh.get(*i).copied().unwrap_or(false))
                .map(|(_, c)| *c as usize)
                .collect::<std::collections::BTreeSet<usize>>()
                .into_iter()
                .collect();
            let spacing =
                rw_mpas::convert::window_spacing_label(&footprint_spacing.spacing_metres, &sampled);
            let label = rw_mpas::convert::caller_grid_label(text, spacing.as_deref())?;
            println!("GRID_LABEL\t{label}\tmesh={}", mesh_grid_label.as_deref().unwrap_or("-"));
            Some(label)
        }
    };
    let spacing_stats = spacing_summary_km(&footprint_spacing.spacing_metres);
    println!(
        "WINDOW\t{}\t{}x{}\tcells={}",
        args.window,
        weights.ny,
        weights.nx,
        weights.n_cells
    );
    println!(
        "WEIGHTS\t{}\tbuild_s={:.3}\tmesh_read_s={:.3}\tmax_nn_km={:.3}\tmean_nn_km={:.3}",
        weights.weights_sha256,
        weights.build_seconds,
        mesh_seconds,
        weights.max_distance_km(),
        weights.mean_distance_km()
    );
    println!(
        "FOOTPRINT\toff_mesh_points={}\tof={}\ton_mesh_max_nn_km={:.3}\tfactor={:.1}",
        off_mesh,
        weights.ny * weights.nx,
        weights.max_on_mesh_distance_km(),
        ON_MESH_SPACING_FACTOR
    );

    let product = if args.remap == "product" {
        let topology = read_mesh_topology(&mesh_path, mesh.n_cells()).map_err(|e| e.to_string())?;
        let remap = ProductRemap::build(
            &mesh,
            &topology,
            &footprint_spacing.spacing_metres,
            &weights,
        )
        .map_err(|e| e.to_string())?;
        println!(
            "REMAP\tproduct\t{}\tbuild_s={:.3}\tlinear_contained={}\tlinear_widened={}\tlinear_clamped={}\tsamples_per_pixel={}..{}",
            remap.sha256,
            remap.build_seconds,
            remap.linear.contained,
            remap.linear.widened,
            remap.linear.clamped,
            remap.conservative.samples_min,
            remap.conservative.samples_max
        );
        Some(remap)
    } else {
        None
    };

    // One frame: what it prints, in order, and its receipt record.  Frames
    // share only read-only state (the mesh, the weights, the remap), so
    // --jobs converts several at a time; results are consumed in --history
    // order, so stdout and the receipt do not depend on which finished first.
    // The init's fields are the same for every frame: hash and read the init
    // once (read_init_fields names the 419 GB-per-view breakage this removes).
    let init_fields = match args.init.as_deref() {
        Some(path) => Some(
            read_init_fields(path, args.init_sha256.as_deref()).map_err(|e| e.to_string())?,
        ),
        None => None,
    };
    let terrain_authority = args.terrain_authority.as_deref()
        .map(|path|TerrainAuthority::read(path,&weights).map(std::sync::Arc::new))
        .transpose().map_err(|e|e.to_string())?;
    let convert_one = |history: &PathBuf| -> Result<(Vec<String>, String), String> {
        let mut lines: Vec<String> = Vec::new();
        let frame = read_history_on_mesh_with_field_set(
            history,
            init_fields.as_ref(),
            None,
            None,
            Some(&mesh),
            &args.field_set,
        )
        .map_err(|e| e.to_string())?;
        let (dlat, dlon) =
            assert_mesh_agrees(&frame, &mesh, 1.0e-4).map_err(|e| e.to_string())?;
        // A split-stream history (RRFS-MPAS) took its centres from --mesh and
        // fields from its diag stream: said on stderr and in the receipt,
        // and only for such a frame, so MPAS-A receipts are unchanged.
        let split_json = match &frame.split {
            None => String::new(),
            Some(split) => {
                eprintln!(
                    "rw_mpas_convert: advisory: history {} carries no cell centres; they are                      the mesh's {} (static fields agreeing cell for cell: {}); {} field(s)                      from the diag stream {}: {}",
                    history.display(),
                    split.coordinates_path.display(),
                    if split.agreed_static_fields.is_empty() {
                        "none".to_string()
                    } else {
                        split.agreed_static_fields.join(", ")
                    },
                    split.diag_fields.len(),
                    split
                        .diag_path
                        .as_ref()
                        .map(|p| p.display().to_string())
                        .unwrap_or_else(|| "(none beside it)".to_string()),
                    split.diag_fields.join(", ")
                );
                format!(
                    ", \"split_streams\": {{\"cell_centres_from\": \"{}\", \"cell_centres_sha256\": \"{}\", \"agreed_static_fields\": [{}], \"diag\": {}, \"diag_sha256\": {}, \"diag_fields\": [{}]}}",
                    json_escape(&split.coordinates_path.display().to_string()),
                    split.coordinates_sha256,
                    split
                        .agreed_static_fields
                        .iter()
                        .map(|n| format!("\"{n}\""))
                        .collect::<Vec<_>>()
                        .join(", "),
                    match &split.diag_path {
                        Some(p) => format!("\"{}\"", json_escape(&p.display().to_string())),
                        None => "null".to_string(),
                    },
                    match &split.diag_sha256 {
                        Some(d) => format!("\"{d}\""),
                        None => "null".to_string(),
                    },
                    split
                        .diag_fields
                        .iter()
                        .map(|n| format!("\"{n}\""))
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            }
        };
        // A history with its own centres completed from its diag stream
        // (MPAS-A's output stream carries no refl10cm_max): named in the
        // receipt, and only for such a frame.
        let split_json = match &frame.companion_diag {
            None => split_json,
            Some(diag) => format!(
                "{split_json}, \"diag_stream\": {{\"diag\": \"{}\", \"diag_sha256\": {}, \"diag_fields\": [{}]}}",
                json_escape(&diag.path.display().to_string()),
                match &diag.sha256 {
                    Some(d) => format!("\"{d}\""),
                    None => "null".to_string(),
                },
                diag.fields.iter().map(|n| format!("\"{n}\"")).collect::<Vec<_>>().join(", ")
            ),
        };
        let stamp = frame.valid_time.label().replace(':', "_");
        let destination = Path::new(&out_dir).join(format!("{}_{stamp}", args.prefix));
        let options = ConvertOptions {
            simulation_start,
            field_set: args.field_set.clone(),
            format: args.format,
            clobber: args.clobber,
            model_label: args.model_label.clone(),
            grid_label: grid_label.clone(),
            terrain_authority: terrain_authority.clone(),
            ..ConvertOptions::default()
        };
        let emitted = match &product {
            Some(remap) => convert_frame_remapped(&frame, remap, &destination, &options),
            None => convert_frame(&frame, &weights, &destination, &options),
        }
        .map_err(|e| e.to_string())?;
        // The accumulation budget: the window's integral of total
        // precipitation against the mesh's own sum(value * areaCell).
        let mut budget_json = "null".to_string();
        if let Some(remap) = &product {
            if let (Some((rc, 1)), Some((rnc, 1))) = (
                frame.fields.get("RAINC").map(|(v, l)| (v, *l)),
                frame.fields.get("RAINNC").map(|(v, l)| (v, *l)),
            ) {
                let total: Vec<f32> = rc.iter().zip(rnc.iter()).map(|(a, b)| a + b).collect();
                let remapped = remap.conservative.gather(&total, 1);
                let (mesh_integral, grid_integral, covered) =
                    remap.conservative.budget(&total, &remap.area_unit, &remapped);
                // The two integrals are the same water only when the window
                // holds the whole mesh; a regional window holds part of it,
                // and its difference from the mesh total is geography, not
                // remap error, so no relative difference is claimed there.
                let whole = (covered - 1.0).abs() <= 1.0e-3;
                let rel = if !whole {
                    None
                } else if mesh_integral != 0.0 {
                    Some(grid_integral / mesh_integral - 1.0)
                } else {
                    Some(0.0)
                };
                let rel_text = match rel {
                    Some(v) => format!("{v:.3e}"),
                    None => "n/a(partial-window)".to_string(),
                };
                lines.push(format!(
                    "BUDGET\t{}\tRAINC+RAINNC\tmesh={:.9e}\tgrid={:.9e}\trel={}\tcovered={:.9}",
                    frame.valid_time.label(),
                    mesh_integral,
                    grid_integral,
                    rel_text,
                    covered
                ));
                budget_json = format!(
                    "{{\"field\": \"RAINC+RAINNC\", \"mesh_integral_unit_sphere_mm\": {mesh_integral:e}, \"grid_integral_unit_sphere_mm\": {grid_integral:e}, \"relative_difference\": {}, \"covered_mesh_area_fraction\": {covered}}}",
                    match rel {
                        Some(v) => format!("{v:e}"),
                        None => "null".to_string(),
                    }
                );
            }
        }
        lines.push(format!(
            "CONVERTED\t{}\t{:.3}s\t{} bytes\t{} written\t{} absent",
            emitted.path.display(),
            emitted.seconds,
            emitted.bytes,
            emitted.written.len(),
            emitted.absent.len()
        ));
        // The receipt's output digest is the OUTPUT's.  It was the history's
        // digest written a second time under this name, so a consumer that
        // verified the frame it was handed against the receipt refused
        // every frame the converter had just written.
        let output_sha256 = sha256_file(&emitted.path).map_err(|e| e.to_string())?;
        let record = format!(
            "{{\"output\": \"{}\", \"output_sha256\": \"{}\", \"output_bytes\": {}, \"valid_time\": \"{}\", \"simulation_start\": \"{}\", \"lead_seconds\": {}, \"convert_seconds\": {}, \"history_sha256\": \"{}\", \"init_sha256\": \"{}\", \"max_latitude_error_degrees\": {:e}, \"max_longitude_error_degrees\": {:e}, \"written_variables\": [{}], \"absent_wrf_fields\": [{}], \"precipitation_budget\": {}{}}}",
            json_escape(&emitted.path.display().to_string()),
            output_sha256,
            emitted.bytes,
            emitted.valid_time.iso(),
            emitted.simulation_start.iso(),
            emitted.lead_seconds,
            emitted.seconds,
            frame.history_sha256,
            frame.init_sha256.clone().unwrap_or_default(),
            dlat,
            dlon,
            emitted
                .written
                .iter()
                .map(|n| format!("\"{n}\""))
                .collect::<Vec<_>>()
                .join(", "),
            emitted
                .absent
                .iter()
                .map(|n| format!("\"{n}\""))
                .collect::<Vec<_>>()
                .join(", "),
            budget_json,
            split_json,
        );
        Ok((lines, record))
    };
    let jobs = if args.jobs > 0 { args.jobs } else { auto_frame_jobs(&args.history) };
    let mut records: Vec<String> = Vec::new();
    for chunk in args.history.chunks(jobs.max(1)) {
        let done: Vec<Result<(Vec<String>, String), String>> = if chunk.len() == 1 {
            vec![convert_one(&chunk[0])]
        } else {
            use rayon::prelude::*;
            chunk.par_iter().map(&convert_one).collect()
        };
        for result in done {
            let (lines, record) = result?;
            for line in lines {
                println!("{line}");
            }
            records.push(record);
        }
    }

    if let Some(path) = &args.json {
        let body = format!(
            "{{\n  \"schema\": \"{}\",\n  \"engine\": \"rw-mpas {} (rust)\",\n  \"product_remap\": {},\n  \"boundary_zone\": {},\n  \"window\": \"{}\",\n  \"window_spec\": {},\n  \"window_description\": \"{}\",\n  \"grid_label\": {},\n  \"mesh_spacing_km\": {},\n  \"mesh_sha256\": \"{}\",\n  \"weights_sha256\": \"{}\",\n  \"resample_method\": \"{}\",\n  \"field_set\": \"{}\",\n  \"weights_build_seconds\": {},\n  \"frames\": [\n    {}\n  ]\n}}\n",
            rw_mpas::convert::BRIDGE_SCHEMA,
            env!("CARGO_PKG_VERSION"),
            product.as_ref().map(|r| r.summary_json()).unwrap_or_else(|| "null".to_string()),
            boundary_zone,
            json_escape(&args.window),
            window.spec_json(),
            json_escape(window.description()),
            {
                let quoted = |label: &Option<String>| match label {
                    Some(label) => format!("\"{}\"", json_escape(label)),
                    None => "null".to_string(),
                };
                // The mesh's own label rides beside a caller's only when one was
                // given, so a receipt without --grid-label is byte-identical.
                if args.grid_label.is_some() {
                    format!("{},\n  \"mesh_grid_label\": {}", quoted(&grid_label), quoted(&mesh_grid_label))
                } else {
                    quoted(&grid_label)
                }
            },
            spacing_stats,
            weights.mesh_sha256,
            weights.weights_sha256,
            if product.is_some() { rw_mpas::remap::PRODUCT_REMAP_METHOD } else { rw_mpas::weights::RESAMPLE_METHOD },
            args.field_set,
            weights.build_seconds,
            records.join(",\n    ")
        );
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        std::fs::write(path, body).map_err(|e| e.to_string())?;
        println!("REPORT\t{}", path.display());
    }
    println!(
        "FINISHED\tframes={}\telapsed_s={:.3}",
        args.history.len(),
        started.elapsed().as_secs_f64()
    );
    Ok(())
}

/// Frames converted at a time when --jobs is not given: every frame, up to the
/// cores, and no more than half the available host memory holds at about
/// twice each history file's size (its fields as read, plus the plane being
/// written).
fn auto_frame_jobs(history: &[PathBuf]) -> usize {
    let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let largest = history
        .iter()
        .filter_map(|path| std::fs::metadata(path).ok())
        .map(|meta| meta.len())
        .max()
        .unwrap_or(0);
    let per_frame = largest.saturating_mul(2).saturating_add(256 << 20);
    let by_memory = match rw_host_memory::available_bytes() {
        Some(available) => ((available / 2) / per_frame.max(1)).max(1) as usize,
        None => 1,
    };
    history.len().min(cores).min(by_memory).clamp(1, 64)
}

fn main() -> ExitCode {
    let _ = std::hint::black_box(GPUWM_BRIDGE_SOURCE_REV_STAMP);
    let argv: Vec<String> = std::env::args().skip(1).collect();
    if argv.iter().any(|a| a == "--help" || a == "-h") {
        println!("{USAGE}");
        return ExitCode::SUCCESS;
    }
    if argv.iter().any(|a| a == "--abi") {
        println!("{ABI}");
        return ExitCode::SUCCESS;
    }
    if argv.is_empty() {
        eprintln!("{USAGE}");
        return ExitCode::FAILURE;
    }
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("rw_mpas_convert: {message}");
            ExitCode::FAILURE
        }
    }
}
