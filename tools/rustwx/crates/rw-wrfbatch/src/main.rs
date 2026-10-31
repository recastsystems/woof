//! Headless raw-WRF import plus production Rusty Weather batch rendering.
//!
//! This intentionally reuses the UI crate's hardened WRF processing modules;
//! it only replaces the egui orchestration with a bounded command-line job.
//!
//! gpuwm adaptations over the source workspace's rw_wrf_batch bin:
//! every stored frame renders by default (`--frames all`), a single frame
//! is selectable by stored slot (`--frames N`), and the work limits are
//! sized to the request instead of the GUI's one-hour click ceilings --
//! which is what lets exact-time (sub-hourly) imports render every frame.
//!
//! It also stamps the run's own identity onto every output.  The upstream
//! filename carried model, init cycle, forecast hour, and the generic
//! `native_grid` slug -- which is identical for every nest of one run, so
//! rendering two domains at one lead into one directory silently
//! overwrote one forecast with the other.  The inputs' `GRID_ID` and `DX`
//! become a `d02-3km` domain token in the slug position, and the same
//! spacing appears as `Δx 3 km` in the plot subtitle.  Provenance is the
//! local model (`--source-label`, default `WOOF` plus engine version) rather than the GDEX
//! source inherited from the `wrf` store-model identity.

#[path = "grib_import.rs"]
mod grib_import;
#[path = "local_import.rs"]
mod local_import;
#[path = "nest_move.rs"]
mod nest_move;
#[path = "postproc_severe.rs"]
mod postproc_severe;
#[path = "wrf_process.rs"]
mod wrf_process;
#[path = "wrf_volumes.rs"]
mod wrf_volumes;
mod wrf_chart_planes;
mod wrf_column_planes;
mod ivt;
mod wrf_vertical_velocity;
mod mslp_smoothing;
mod snowfall;
mod visibility;
mod source_memo;
#[path = "mesh.rs"]
mod mesh;
#[path = "section.rs"]
mod section;
mod input_list;
mod run_difference;
mod sheet;
mod store_render;
use rw_wrfbatch::sfire;
mod viewer_profile;
mod worker;

use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::atomic::AtomicBool;

/// `GPUWM_BRIDGE_SOURCE_REV=<40-hex commit>`: the source revision this
/// binary was built from, embedded so the gpuwm release cut can prove a
/// staged bridge matches the commit being released by reading bytes
/// alone (`tools/build_bridge_bundle.py pin --source-rev`).  `build.rs`
/// injects the value; `main` references the constant so the linker
/// cannot discard it.
pub static GPUWM_BRIDGE_SOURCE_REV_STAMP: &str =
    concat!("GPUWM_BRIDGE_SOURCE_REV=", env!("GPUWM_BRIDGE_SOURCE_REV"));

use rustwx_products::shared_context::TitleProvenance;
use rusty_weather::batch_render::{
    BatchHourScope, BatchRenderDomain, BatchRenderEvent, BatchRenderLimits, BatchRenderRequest,
    inspect_renderable_products_over, run_batch_render,
};
use wrf_process::{WrfProcessMessage, WrfProcessOptions, spawn_process_paths};

/// Provenance label for a locally-imported run.  The store model slug is
/// `wrf`, whose only registered fetch source is GDEX -- a label this lane
/// never fetched from.  `--source-label` renames it for stock-WRF files,
/// which ArWen did not produce and must not claim.
const DEFAULT_SOURCE_LABEL: &str = concat!("WOOF ", env!("GPUWM_ENGINE_VERSION"));

/// The exact `--abi` line `gpuwm.rustwx.RENDERER_ABI_MARKER` pins.
///
/// The same handshake `rw_fetch` and `rw_nexrad` -- this workspace's two
/// other bundled binaries -- already answer, applied to the third.  It
/// exists because "does the binary start" is not a contract check: two
/// rw_wrfbatch builds with different md5s both printed the usage line
/// and both were reported `verified`, so a stale renderer substituted a
/// foreign engine into every rust render gate and drew the plots
/// afterwards (task #106).
///
/// What is listed is the vocabulary the PYTHON half parses, not a
/// version number -- a rebuild bumps a version whether or not anything
/// changed, and a marker that moves on every rebuild proves nothing:
///
/// * the five tab-separated `PRODUCT` fields and the `CATALOG` tally
///   line `gpuwm.rustwx.list_products` reads;
/// * the `RENDERED` / `SKIPPED` / `FAILED` event words, and the
///   `SECTIONFILL` line that says which range a vertical cut's fill was
///   drawn over and the rule that set it
///   `gpuwm.rustwx.run_renderer` reads.  Events v2: a per-frame `SKIPPED`
///   or `FAILED` reason opens with the input file of the frame it is
///   about (`frame-attributed`), because a series invocation has many
///   inputs and the reader used to put the LAST one in front of every
///   line, naming F018 against an F000 reason;
/// * the generic `var:` family and the `selectable_slugs` count of the
///   store-independent catalog -- the lane whose ABSENCE from a stale
///   build is what #106 was reported as (catalog 153 against this
///   tree's 168, zero generic rows) -- the `xsec:` family, the
///   vertical-section lane cut from the wrfout files directly, and the
///   `mesh:`/`meshdiff:` families, the polygon-mesh lane cut from an MPAS
///   history frame and its grid file with no regrid in between.
///   Vocabulary v2: `all` is every NAMED product the frames can draw and
///   no longer every stored variable; the stored variables are the
///   `variables` keyword, asked for by name;
///
/// The PRODUCT row carries a sixth field, `code`: a stable machine
/// spelling of WHY the row has the status it has.  The detail column
/// stays prose and stays byte-identical, because it is what a reader
/// sees; a consumer that has to decide something matches the code.  A
/// reader that matched on the prose was a table that stopped matching
/// the moment the sentence was reworded, at which point the excluded
/// slug was forwarded and the whole render failed.
///
/// `--list-products` with NO inputs additionally prints the fileless
/// requirement pair: one `NEEDS` row per catalog slug naming the store
/// selectors its recipe resolves, and the `PLANNED` set of store fields
/// this build's wrfout import would write.  Together they answer "can
/// this install draw that product?" before a single frame exists, which
/// is the question a plan review asks and a store-aware listing cannot
/// be asked until after the import.
///
/// It also prints one `WRFOUT` row per selectable slug: the verdict of
/// the wrfout import lane on that slug, fileless.  `drawable` rows carry
/// the first forecast hour at which the product can exist (0 for an
/// instantaneous field, the hour its window closes for a windowed one);
/// `missing` rows name what no wrfout import writes.  The verdict is
/// taken in the recipe's own vocabulary -- selectors against the
/// selectors the import plans, recipe slugs against the grids it plans --
/// which is what the NEEDS/PLANNED pair could not do on its own.
///
/// Changing any of those is changing this contract, so the literal
/// changes with it and every binary predating the change fails the
/// handshake instead of quietly answering the old grammar.
const ABI_MARKER: &str = "gpuwm-rw-wrfbatch-catalog-v1\tPRODUCT\tslug\tkind\tstatus\tdetail\tcode\tCATALOG\t\
gpuwm-rw-wrfbatch-requirements-v1\tNEEDS\tslug\tselector\tPLANNED\tstore_field\t\
gpuwm-rw-wrfbatch-wrfout-lane-v1\tWRFOUT\tslug\tkind\tverdict\tminimum_hour\tdetail\t\
gpuwm-rw-wrfbatch-events-v2\tRENDERED\tSKIPPED\tFAILED\tframe-attributed\t\
gpuwm-rw-wrfbatch-sections-v1\tSECTIONFILL\tslug\tlo\thi\tabsence\trule\t\
gpuwm-rw-wrfbatch-inputs-json-v1\t\
gpuwm-rw-wrfbatch-vocabulary-v2\tgeneric\tvar:\tvariables\txsec:\tmesh:\tmeshdiff:\tselectable_slugs\t\
gpuwm-rw-wrfbatch-layout-v1\t--layout\tauto\tfixed\t--size-class\t--scale\t--pair-sheet\t\
gpuwm-rw-wrfbatch-difference-v2\t--diff-against\t--diff-inputs-json\t--diff-label-a\t--diff-label-b\t--diff-labels\t--diff-sheet\tDIFFERENCE\tdiffering_cells\tRUSTWX_DIFFERENCE_TABLE";

#[derive(Debug, Clone)]
struct Args {
    store_root: PathBuf,
    out_dir: PathBuf,
    products: String,
    frames: Option<usize>,
    /// The fixed-canvas size, and the nominal size the batch limits and the
    /// mesh lane read.  Under auto layout each domain's canvas comes from
    /// the layout table instead.
    width: u32,
    height: u32,
    /// `--layout auto|fixed`, `--size-class`, `--scale`: how each frame's
    /// canvas is chosen.  Auto sizes it from the grid's own shape; fixed
    /// keeps `--width x --height` for callers that tile at fixed pixels.
    layout: rustwx_render::LayoutMode,
    heavy: bool,
    list_products: bool,
    /// `--streamlines` / `--barbs`: the wind layer this invocation asks
    /// for.  `None` leaves `RUSTWX_WIND_STREAMLINES`, and then the
    /// automatic per-grid choice, in charge.
    streamlines: Option<bool>,
    source_label: String,
    /// `--overlays FILE.json`: map overlays in geographic degrees.
    overlays: Option<rustwx_products::geographic_overlays::MapOverlays>,
    /// `--annotate FILE.json`: title/subtitle overrides.
    annotations: Option<rustwx_products::geographic_overlays::PanelAnnotations>,
    /// `--theme NAME|FILE.json`: the render theme, resolved before any
    /// import so a typo is refused before a file is opened.  `None` is
    /// the renderer's own look, byte-identical to every earlier build.
    theme: Option<rustwx_render::RenderTheme>,
    /// `--radar-colors standard|classic`: the tables the reflectivity and
    /// radial velocity products draw with.  The flag outranks
    /// RUSTWX_RADAR_COLORS; absent both it is `standard`.
    radar_colors: rustwx_render::RadarColorSet,
    /// `--section lat,lon,lat,lon | FILE.json`: the line every `xsec:`
    /// product is cut along; required when one is requested.
    section: Option<section::SectionLine>,
    /// `--section-across KM`: a second frame per section product,
    /// perpendicular to the line through the fill's maximum column.
    section_across_km: Option<f64>,
    /// `--isotherms L,L,...[@H]`: the isotherms drawn on every section.
    isotherms: section::Isotherms,
    /// `--section-top-km N`: the ceiling the fitted height range may not
    /// pass.
    section_top_km: f64,
    /// `--section-size WxH`: the size a SECTION is drawn at.  Absent, a
    /// section is landscape 2:1 at the map's width, because a section
    /// handed the map's own size came out portrait -- the shape a vertical
    /// cut is least readable in.
    section_size: Option<(u32, u32)>,
    /// `--section-reference-km N`: the altitude the fitted height range
    /// keeps at least two kilometres of air above.  Absent, the line's own
    /// highest terrain.
    section_reference_km: Option<f64>,
    /// `--mesh-grid FILE.nc`: the MPAS grid file whose `verticesOnCell`
    /// gives every `mesh:` product its polygons.  Required when one is
    /// requested; a history frame carries no cell boundaries.
    mesh_grid: Option<PathBuf>,
    /// `--mesh-reference DIR|FILE`: the other leg, for `meshdiff:`.
    mesh_reference: Option<PathBuf>,
    /// `--mesh-labels A,B`: the two legs' names in a difference headline.
    mesh_labels: (String, String),
    /// `--mesh-bounds W,E,S,N`: the frame a mesh product is drawn in.
    mesh_bounds: Option<(f64, f64, f64, f64)>,
    /// `--footer-*`: the caption fields the theme's footer strip fills.
    /// Absent every one of them, no strip is drawn even under a theme that
    /// names one, so an existing render is unchanged.
    footer: rustwx_render::FooterFields,
    /// `--diff-against WRFOUT` (repeatable), `--diff-labels A,B`,
    /// `--diff-sheet`: draw every requested map product as this run minus
    /// the named run, at the valid time both share (`run_difference.rs`).
    difference: Option<run_difference::DifferenceArgs>,
    /// `--save-import FILE`: after the import, record what a later draw needs from the frames (the run,
    /// its stored slots, frame attribution, the run context and the grid identity) in FILE.
    save_import: Option<PathBuf>,
    /// `--from-import FILE`: draw from the store run a `--save-import` recorded, without opening the
    /// converted frames again (they may already be deleted); no wrfout input is needed.
    from_import: Option<PathBuf>,
    inputs: Vec<PathBuf>,
}

fn usage() -> &'static str {
    "usage: rw_wrfbatch --store-root DIR --out-dir DIR [--products all|SLUGS] \
[--frames all|N] [--layout auto|fixed] [--size-class standard|phone|large] [--scale S] \
[--width N] [--height N] [--heavy] [--streamlines|--barbs] \
[--source-label TEXT] [--theme NAME|FILE.json] [--radar-colors standard|classic] [--section lat,lon,lat,lon|FILE.json] \
[--section-across KM] [--isotherms L,L,...[@H]] [--section-top-km N] \
[--section-size WxH] [--section-reference-km N] \
[--mesh-grid FILE.nc] [--mesh-reference DIR|FILE] [--mesh-labels A,B] [--mesh-bounds W,E,S,N] \
[--footer-title TEXT] [--footer-valid TEXT] [--footer-mesh TEXT] [--footer-leg TEXT] \
[--footer-note TEXT] [--diff-against WRFOUT]... [--diff-inputs-json FILE] \
[--diff-labels A,B] [--diff-label-a TEXT] [--diff-label-b TEXT] [--diff-sheet] \
[--list-products] [--save-import FILE] [--from-import FILE] [--inputs-json FILE] wrfout...\n       \
rw_wrfbatch --cf FILE.nc --variable NAME --out-dir DIR [--valid UTC] [--bounds W,E,S,N] [--overlays FILE.json] [--observation FILE]\n       rw_wrfbatch --help | --abi | --worker-abi\n       rw_wrfbatch --worker --worker-host-mib N [fixed render options]"
}

/// What went wrong, and therefore what the user should be shown.
///
/// The three arms exist because the old single `String` channel made every
/// failure look identical: the message, then the usage line, then exit 1 --
/// so "unknown product 'x'" and "the store has no hours object" were
/// reported the same way, and the usage line was the LAST stderr line, which
/// is exactly the line `gpuwm/rustwx.py` surfaces as the reason.
///
/// `Help` also replaces the previous `error == usage()` string-equality
/// test.  That test decided the exit code of `--help` by comparing the
/// error text to the usage text, so any future error that happened to equal
/// the usage string would have exited 0.  The observable contract is
/// unchanged -- `--help` still prints exactly `usage()` and still exits 0,
/// which is what `gpuwm.rustwx.probe_renderer` requires to declare the rust
/// engine usable at all.
enum CliError {
    /// `--help`/`-h`: print the usage line, succeed.
    Help,
    /// The command line is wrong.  Message, then the usage line, exit 2 --
    /// the code `gpuwm render`'s matplotlib engine uses for the same class
    /// of mistake (`gpuwm/render.py` raises on an unknown product and the
    /// CLI turns that into 2).
    Usage(String),
    /// The command line was fine and the work failed.  Message only: the
    /// usage line says nothing about a store that will not open, and
    /// printing it buries the sentence that does.
    Failed(String),
}

impl From<String> for CliError {
    fn from(message: String) -> Self {
        CliError::Usage(message)
    }
}

impl From<&str> for CliError {
    fn from(message: &str) -> Self {
        CliError::Usage(message.to_string())
    }
}

/// A parsed command line.  `--list-products` with no inputs asks a question
/// about this build, not about a store, so it does not need `--store-root`,
/// `--out-dir` or a wrfout to answer.
enum Invocation {
    Batch(Box<Args>),
    /// Print the product vocabulary and exit.
    Catalog,
    /// Print [`ABI_MARKER`] and exit: the stale-build handshake.
    Abi,
}

/// How many products one `--products` spelling asks for per frame.
///
/// Counted after the group keywords are expanded.  A keyword is ONE token
/// and many products: counted as one, `--products windowed` (49 products)
/// kept the GUI's 32-per-hour ceiling and refused its own whole-hour
/// series with "49 per-hour products selected; GUI ceiling is 32".
fn requested_product_count(spec: &str) -> usize {
    let tokens = spec
        .split(',')
        .filter(|slug| !slug.trim().is_empty())
        .count();
    rusty_weather::render_all::partition_products(spec)
        .map(|request| {
            request.direct.len()
                + request.derived.len()
                + request.generic.len()
                + request.windowed.len()
        })
        .unwrap_or_default()
        .max(tokens)
}

/// The static product vocabulary, for `--list-products` with no inputs.
///
/// The store-aware listing (which of these the imported frames can actually
/// render, and why not) still needs a store and is unchanged; this answers
/// the other question -- "what may I put in --products?" -- which is the one
/// a user has after an unknown-product refusal.
fn print_product_catalog() -> Result<(), CliError> {
    let slugs = rusty_weather::render_all::known_product_slugs();
    println!(
        "group keywords: all, direct, derived, heavy, windowed, {VARIABLES_KEYWORD}"
    );
    // The generic family's vocabulary belongs to the STORE, not this
    // build, so no slug list can be printed here; the store-aware listing
    // names each `var:` row it can serve.
    println!(
        "generic products: var:<stored 2-D variable name>; '{VARIABLES_KEYWORD}' draws every \
         stored variable no named product draws ('all' draws named products only)"
    );
    println!(
        "mesh products: mesh:<history variable>[:colmax|:colmin|:level=K][~log] and \
meshdiff:<...> (need --mesh-grid FILE.nc; meshdiff also --mesh-reference)"
    );
    for slug in &slugs {
        // Deliberately NOT the `PRODUCT\t...` record the store-aware listing
        // emits: that one is parsed by gpuwm/rustwx.py as tab-separated
        // fields, and a same-prefixed two-field line is a trap.
        println!("  {slug}");
    }
    for row in rw_wrfbatch::hubwind_products::product_catalog() {
        println!("  {}", row);
        println!("WRFOUT\t{}\tdirect\teligible\t\thubwind native columns and timeline", row);
    }
    for (slug,field,_,_) in sfire::PRODUCTS {
        println!("  {slug}");
        println!("NEEDS\t{slug}\tFXLAT\tFXLONG\tLFN\t{field}");
        println!("WRFOUT\t{slug}\tdirect\teligible\t\trefined fire fields with geographic coordinates");
    }
    // The fileless half of availability: what each slug NEEDS, in the
    // store's own selector vocabulary, and what this build's wrfout
    // import is PLANNED to write.  Neither opens a file, so both can be
    // asked at plan review -- before a wrfout exists, which is exactly
    // when a preset is chosen and when the answer is still useful.
    //
    // The selector vocabulary is deliberate.  A row spelled in WRFOUT
    // variable names would be an invented mapping: a wrf-core VarDef
    // declares no input list at all, and its compute function opens the
    // file and reads whatever it needs at run time.  Emitting guesses
    // for a few hundred slugs is the correct-looking-wrong-mapping
    // defect this tree has already paid for once.
    for slug in &slugs {
        let requirements = match rustwx_models::plot_recipe_store_requirements(slug) {
            Ok(requirements) => requirements,
            Err(_) => continue,
        };
        let keys: Vec<String> = requirements
            .iter()
            .map(|requirement| match requirement.selector {
                Some(selector) => selector.key(),
                // A requirement with no canonical selector cannot be
                // resolved from ANY store, which is a fact worth
                // printing rather than a row worth hiding.
                None => format!("{}!no-canonical-selector", requirement.field_key),
            })
            .collect();
        if keys.is_empty() {
            continue;
        }
        println!("NEEDS\t{slug}\t{}", keys.join("\t"));
    }
    for field in crate::wrf_process::WrfProcessOptions::default().planned_store_fields() {
        println!("PLANNED\t{field}");
    }
    for row in wrfout_lane_rows() {
        println!(
            "WRFOUT\t{}\t{}\t{}\t{}\t{}",
            row.slug,
            row.kind,
            row.verdict,
            row.minimum_hour
                .map(|hour| hour.to_string())
                .unwrap_or_default(),
            row.detail
        );
    }
    // Not "total=": the store-aware listing already owns that word for its
    // own count of catalog ROWS (which includes rows no --products spelling
    // selects).  This is the size of the --products vocabulary.
    // The hub-height maps (lane/cap-hubwind, 089efd92470) are printed as
    // products above and `gpuwm render --products hub_*` routes them, so
    // they are part of the --products vocabulary this count declares.
    println!("selectable_slugs={}", slugs.len()
        + rw_wrfbatch::hubwind_products::product_catalog().len()
        + sfire::PRODUCTS.len());
    println!(
        "pass --store-root DIR --out-dir DIR wrfout... with --list-products \
         for per-frame availability"
    );
    Ok(())
}

/// The keyword that draws every stored 2-D variable no named product draws.
///
/// WHAT BREAKAGE THIS PREVENTS (gate law): `all` used to expand to the
/// store's whole catalog, generic rows included, and an 18 h run drawn
/// with the TUI's default `all` published 137 of its 204 product folders
/// as raw variables (`var_wrf_t2_1222df9c491fb635`) -- most of them the
/// same grids the named products beside them already drew.  `all` is the
/// named products now; the raw variables are this keyword, asked for by
/// name.
const VARIABLES_KEYWORD: &str = "variables";

/// Named plots do not consume arbitrary diagnostic browse grids.  A render
/// of named products, and a listing of them, imports them with the named
/// profile, so the listing imports what its render will; a listing of group
/// keywords alone, generic variables and unknown terms still import in full.
fn named_product_request(products: &str) -> bool {
    let known = rusty_weather::render_all::known_product_slugs();
    let terms: Vec<_> = products.split(',').map(str::trim).collect();
    !terms.is_empty() && terms.iter().all(|term| {
        matches!(*term, "all" | "direct" | "derived" | "windowed")
            || known.iter().any(|slug| slug == term)
    })
}

/// Whether a request names at least one product rather than only the
/// group keywords [`named_product_request`] accepts.
fn names_a_product(products: &str) -> bool {
    products.split(',').map(str::trim).any(|term| {
        !term.is_empty() && !matches!(term, "all" | "direct" | "derived" | "windowed")
    })
}

fn rainfall_import_options(products: &str) -> Option<WrfProcessOptions> {
    let requested: Vec<String> = products.split(',').map(|term| term.trim().to_string()).collect();
    let profile = viewer_profile::ViewerProfile::new(&requested).ok()?;
    // The recipe dependency table proves these products use only cumulative
    // rain. They need no volume, parcel diagnostic or surface approximation.
    (profile.options.only.iter().any(|name| name == "apcp")
        && profile.options.only.iter().all(|name| matches!(name.as_str(), "apcp" | "orography")))
        .then_some(profile.options)
}

/// One slug's verdict on the wrfout import lane, fileless.
struct WrfoutLaneRow {
    slug: String,
    kind: &'static str,
    verdict: &'static str,
    minimum_hour: Option<u16>,
    detail: String,
}

/// Every selectable slug's verdict on the wrfout import lane.
///
/// What a plan review has to know before a single frame exists: which
/// products a LOCAL run can ever draw, and from which forecast hour.  The
/// catalog door used to offer every product of every model -- NBM and
/// ensemble families included -- and the default preset asked a local run
/// for three products no wrfout carries the fields of, which drew zero
/// pictures on every frame of every run and were never named.
///
/// Decided in the recipe's own vocabulary, from the import's own tables:
/// a direct recipe's requirement SELECTORS against
/// [`crate::wrf_process::WrfProcessOptions::planned_store_selectors`], a
/// derived recipe's slug against the grids the import plans, and a window
/// against the store lane's own planner for the first hour it closes.  It
/// is a property of each product and of the import lane, never of a
/// model: a source added to the registry changes nothing here.
fn wrfout_lane_rows() -> Vec<WrfoutLaneRow> {
    use std::collections::{HashMap, HashSet};

    let options = crate::wrf_process::WrfProcessOptions::default().normalized();
    let selectors: HashSet<String> = options
        .planned_store_selectors()
        .into_iter()
        .map(|selector| selector.key())
        .collect();
    let grids: HashSet<String> = options.planned_store_fields().into_iter().collect();
    let mut rows: HashMap<String, WrfoutLaneRow> = HashMap::new();
    let mut put = |slug: &str, kind, verdict, minimum_hour, detail: String| {
        rows.entry(slug.to_string()).or_insert(WrfoutLaneRow {
            slug: slug.to_string(),
            kind,
            verdict,
            minimum_hour,
            detail,
        });
    };
    // In the order the partition classifies a slug (windowed, then
    // derived, then direct), so a slug two lanes claim gets the verdict
    // of the lane that would actually draw it.
    for product in rustwx_products::windowed::HrrrWindowedProduct::supported_products() {
        match rusty_weather::render_all::windowed_store::minimum_window_hour(*product) {
            Some(hour) => put(
                product.slug(),
                "windowed",
                "drawable",
                Some(hour),
                format!("its window first closes at F{hour:03}"),
            ),
            None => put(
                product.slug(),
                "windowed",
                "missing",
                None,
                "no stored forecast hour closes this window".to_string(),
            ),
        }
    }
    for entry in rustwx_products::derived::supported_derived_recipe_inventory() {
        let kind = if entry.heavy { "heavy" } else { "derived" };
        if grids.contains(entry.slug) {
            put(entry.slug, kind, "drawable", Some(0), entry.title.to_string());
        } else if entry.heavy {
            put(
                entry.slug,
                kind,
                "missing",
                None,
                format!(
                    "no '{}' grid: only the heavy import stage computes it, and a \
                     default wrfout import does not run that stage",
                    entry.slug
                ),
            );
        } else {
            put(
                entry.slug,
                kind,
                "missing",
                None,
                format!(
                    "no '{}' grid: no wrf-core diagnostic of the wrfout import is \
                     stored under this recipe slug",
                    entry.slug
                ),
            );
        }
    }
    for entry in rustwx_products::derived::blocked_derived_recipe_inventory() {
        put(entry.slug, "derived", "missing", None, entry.reason.to_string());
    }
    for spec in rustwx_products::spec::direct_product_specs() {
        match rustwx_models::plot_recipe_store_requirements(&spec.slug) {
            Err(err) => put(
                &spec.slug,
                "direct",
                "missing",
                None,
                format!("no plot recipe: {err}"),
            ),
            Ok(requirements) => {
                let missing: Vec<String> = requirements
                    .iter()
                    .filter_map(|requirement| match requirement.selector {
                        Some(selector) if selectors.contains(&selector.key()) => None,
                        Some(selector) => Some(selector.key()),
                        None => Some(format!(
                            "{} (no canonical store selector exists for this field)",
                            requirement.field_key
                        )),
                    })
                    .collect();
                if missing.is_empty() {
                    put(&spec.slug, "direct", "drawable", Some(0), spec.title.clone());
                } else {
                    put(
                        &spec.slug,
                        "direct",
                        "missing",
                        None,
                        format!("a wrfout import writes no {}", missing.join(", ")),
                    );
                }
            }
        }
    }
    rusty_weather::render_all::known_product_slugs()
        .into_iter()
        .map(|slug| {
            rows.remove(&slug).unwrap_or_else(|| WrfoutLaneRow {
                slug,
                kind: "unclassified",
                verdict: "missing",
                minimum_hour: None,
                detail: "no lane of this build classifies this slug for a wrfout import"
                    .to_string(),
            })
        })
        .collect()
}

/// The catalog keywords this binary expands against the imported store:
/// `all`, `windowed` and [`VARIABLES_KEYWORD`], alone or together.
///
/// A keyword stands for what THESE frames can draw, read off the
/// store-aware catalog: `all` is every named product, `windowed` every
/// window the run's last stored frame closes, `variables` every stored
/// variable no named product draws.  Anything else -- a slug list, or a
/// keyword beside slugs -- is returned unchanged for the strict partition.
fn expand_catalog_keywords(
    spec: &str,
    catalog: &rusty_weather::batch_render::BatchRenderCatalog,
) -> Result<String, String> {
    use rusty_weather::batch_render::BatchProductKind;

    if !is_catalog_keyword_spec(spec) {
        return Ok(spec.to_string());
    }
    let tokens: Vec<&str> = spec
        .split(',')
        .map(str::trim)
        .filter(|token| !token.is_empty())
        .collect();
    let is = |token: &str, keyword: &str| token.eq_ignore_ascii_case(keyword);
    let wants = |word: &str| tokens.iter().any(|token| is(token, word));
    let slugs: Vec<&str> = catalog
        .products
        .iter()
        .filter(|product| match product.kind {
            BatchProductKind::Generic => wants(VARIABLES_KEYWORD),
            BatchProductKind::Windowed => wants("all") || wants("windowed"),
            _ => wants("all"),
        })
        .map(|product| product.slug.as_str())
        .collect();
    if slugs.is_empty() {
        return Err(format!(
            "'{spec}' names no product these frames can draw: windowed products need \
             more than one stored whole-hour frame and a window the last of them closes, \
             and '{VARIABLES_KEYWORD}' needs a stored variable no named product draws"
        ));
    }
    Ok(slugs.join(","))
}

/// Whether `spec` is made only of the keywords [`expand_catalog_keywords`]
/// expands: `all`, `windowed` and [`VARIABLES_KEYWORD`], alone or together.
fn is_catalog_keyword_spec(spec: &str) -> bool {
    let tokens: Vec<&str> = spec
        .split(',')
        .map(str::trim)
        .filter(|token| !token.is_empty())
        .collect();
    !tokens.is_empty()
        && tokens.iter().all(|token| {
            ["all", "windowed", VARIABLES_KEYWORD]
                .iter()
                .any(|keyword| token.eq_ignore_ascii_case(keyword))
        })
}

/// One per-frame event reason, opened with the input file of its frame.
///
/// A series invocation imports many files into one store, and a slot is
/// not a filename; the Python reader used to put the LAST input in front
/// of every line, so a skip at F000 was filed against the F018 file.  The
/// importer knows which file each slot came from, so the line says it.
/// An event with no frame, or a slot the import did not record, keeps its
/// reason unchanged.
fn frame_attributed(
    sources: &std::collections::HashMap<u16, PathBuf>,
    hour: Option<u16>,
    reason: &str,
) -> String {
    match hour.and_then(|hour| sources.get(&hour)) {
        Some(path) => format!("{}: {reason}", path.display()),
        None => reason.to_string(),
    }
}

fn parse_args() -> Result<Invocation, CliError> {
    parse_args_from(std::env::args().skip(1))
}

fn parse_args_from(raw: impl IntoIterator<Item = String>) -> Result<Invocation, CliError> {
    let mut store_root = None;
    let mut out_dir = None;
    let mut products = "all".to_string();
    let mut frames = None;
    let mut width: Option<u32> = None;
    let mut height: Option<u32> = None;
    let mut layout_spec: Option<String> = None;
    let mut size_class = rustwx_render::SizeClass::Standard;
    let mut layout_scale = 1.0f64;
    let mut heavy = false;
    let mut list_products = false;
    let mut save_import: Option<PathBuf> = None;
    let mut from_import: Option<PathBuf> = None;
    let mut streamlines: Option<bool> = None;
    let mut source_label = DEFAULT_SOURCE_LABEL.to_string();
    let mut overlays_path: Option<PathBuf> = None;
    let mut annotate_path: Option<PathBuf> = None;
    let mut theme_spec: Option<String> = None;
    let mut radar_colors_spec: Option<String> = None;
    let mut section_spec: Option<String> = None;
    let mut section_across_km: Option<f64> = None;
    let mut isotherms_spec: Option<String> = None;
    let mut section_top_km = 14.0f64;
    let mut section_size: Option<(u32, u32)> = None;
    let mut section_reference_km: Option<f64> = None;
    let mut mesh_grid: Option<PathBuf> = None;
    let mut mesh_reference: Option<PathBuf> = None;
    let mut mesh_labels = ("TREATMENT".to_string(), "CONTROL".to_string());
    let mut mesh_bounds: Option<(f64, f64, f64, f64)> = None;
    let mut footer = rustwx_render::FooterFields::default();
    let mut diff_against: Vec<PathBuf> = Vec::new();
    let mut diff_labels: Option<rustwx_render::difference::DifferenceLabels> = None;
    let mut diff_sheet = false;
    let mut inputs = Vec::new();
    let mut raw = raw.into_iter();

    while let Some(arg) = raw.next() {
        match arg.as_str() {
            "--save-import" => {
                save_import = Some(PathBuf::from(raw.next().ok_or_else(|| {
                    CliError::Usage("--save-import requires a file".to_string())
                })?));
            }
            "--from-import" => {
                from_import = Some(PathBuf::from(raw.next().ok_or_else(|| {
                    CliError::Usage("--from-import requires a file".to_string())
                })?));
            }
            "--inputs-json" => {
                let path = PathBuf::from(raw.next().ok_or_else(|| {
                    CliError::Usage("--inputs-json requires a JSON file".to_string())
                })?);
                inputs.extend(input_list::read(&path).map_err(CliError::Usage)?);
            }
            "--diff-inputs-json" => {
                let path = PathBuf::from(raw.next().ok_or_else(|| {
                    CliError::Usage("--diff-inputs-json requires a JSON file".to_string())
                })?);
                diff_against.extend(input_list::read(&path).map_err(CliError::Usage)?);
            }
            "--store-root" => {
                store_root = Some(PathBuf::from(
                    raw.next().ok_or("--store-root requires a directory")?,
                ));
            }
            "--out-dir" => {
                out_dir = Some(PathBuf::from(
                    raw.next().ok_or("--out-dir requires a directory")?,
                ));
            }
            "--products" => {
                products = raw.next().ok_or("--products requires a value")?;
            }
            "--frames" => {
                let value = raw.next().ok_or("--frames requires 'all' or an index")?;
                if !value.eq_ignore_ascii_case("all") {
                    frames = Some(
                        value
                            .parse::<usize>()
                            .map_err(|err| format!("invalid --frames: {err}"))?,
                    );
                }
            }
            "--width" => {
                width = Some(
                    raw.next()
                        .ok_or("--width requires a value")?
                        .parse()
                        .map_err(|err| format!("invalid --width: {err}"))?,
                );
            }
            "--height" => {
                height = Some(
                    raw.next()
                        .ok_or("--height requires a value")?
                        .parse()
                        .map_err(|err| format!("invalid --height: {err}"))?,
                );
            }
            "--layout" => {
                layout_spec = Some(raw.next().ok_or("--layout requires auto or fixed")?);
            }
            "--size-class" => {
                let value = raw.next().ok_or("--size-class requires a value")?;
                size_class = rustwx_render::SizeClass::parse(&value).ok_or_else(|| {
                    format!("--size-class must be standard, phone or large, got {value:?}")
                })?;
            }
            "--scale" => {
                let value = raw.next().ok_or("--scale requires a value")?;
                layout_scale = value
                    .parse::<f64>()
                    .ok()
                    .filter(|scale| scale.is_finite() && (0.25..=4.0).contains(scale))
                    .ok_or_else(|| {
                        format!("--scale must be a number from 0.25 to 4, got {value:?}")
                    })?;
            }
            "--source-label" => {
                let value = raw.next().ok_or("--source-label requires a value")?;
                if value.trim().is_empty() {
                    return Err(CliError::Usage(
                        "--source-label must not be blank".to_string(),
                    ));
                }
                source_label = value.trim().to_string();
            }
            // Absent, these two run no code at all: every product this
            // build already draws is byte-identical without them, which
            // `tools/rustwx_render_regression_gate.py` is the gate for.
            "--overlays" => {
                overlays_path = Some(PathBuf::from(
                    raw.next().ok_or("--overlays requires a JSON file")?,
                ));
            }
            "--annotate" => {
                annotate_path = Some(PathBuf::from(
                    raw.next().ok_or("--annotate requires a JSON file")?,
                ));
            }
            // The render theme: a built-in name (`default`, `light`,
            // `dark`) or a JSON file.  The flag outranks RUSTWX_THEME, and
            // absent both the renderer draws exactly as it always has.
            "--theme" => {
                let value = raw.next().ok_or("--theme requires a name or a JSON file")?;
                if value.trim().is_empty() {
                    return Err(CliError::Usage("--theme must not be blank".to_string()));
                }
                theme_spec = Some(value);
            }
            // The radar colour set: `standard` (the radar tables) or
            // `classic` (the reflectivity ladder and blue-red velocity scale
            // they replaced).  One name selects every radar-table product.
            "--radar-colors" => {
                radar_colors_spec = Some(
                    raw.next()
                        .ok_or("--radar-colors requires standard or classic")?,
                );
            }
            // The section line and its dressing, for the `xsec:` family.
            "--section" => {
                section_spec =
                    Some(raw.next().ok_or("--section requires lat,lon,lat,lon or a JSON file")?);
            }
            "--section-across" => {
                let value = raw.next().ok_or("--section-across requires a length in km")?;
                let km: f64 = value
                    .parse()
                    .ok()
                    .filter(|km: &f64| km.is_finite() && *km >= 2.0)
                    .ok_or_else(|| {
                        format!("--section-across '{value}' is not a length of at least 2 km")
                    })?;
                section_across_km = Some(km);
            }
            "--isotherms" => {
                isotherms_spec =
                    Some(raw.next().ok_or("--isotherms requires levels in C (or none)")?);
            }
            "--section-size" => {
                let value = raw.next().ok_or("--section-size requires WxH in pixels")?;
                let (w, h) = value
                    .split_once(['x', 'X'])
                    .ok_or_else(|| format!("--section-size '{value}' is not WxH"))?;
                let parse = |text: &str, name: &str| -> Result<u32, String> {
                    text.trim()
                        .parse::<u32>()
                        .ok()
                        .filter(|value| (200..=12_000).contains(value))
                        .ok_or_else(|| {
                            format!("--section-size {name} '{text}' is not 200-12000 pixels")
                        })
                };
                section_size = Some((parse(w, "width")?, parse(h, "height")?));
            }
            "--section-reference-km" => {
                let value = raw
                    .next()
                    .ok_or("--section-reference-km requires a height in km")?;
                section_reference_km = Some(
                    value
                        .parse::<f64>()
                        .ok()
                        .filter(|km| (0.0..=40.0).contains(km))
                        .ok_or_else(|| {
                            format!("--section-reference-km '{value}' is not within 0-40 km")
                        })?,
                );
            }
            "--section-top-km" => {
                let value = raw.next().ok_or("--section-top-km requires a height in km")?;
                section_top_km = value
                    .parse()
                    .ok()
                    .filter(|km: &f64| km.is_finite() && (1.0..=40.0).contains(km))
                    .ok_or_else(|| format!("--section-top-km '{value}' is not within 1-40 km"))?;
            }
            // The polygon-mesh family's three flags.  The grid file is
            // separate from the history frames because one mesh serves a
            // whole run: reading its connectivity once per frame is work
            // with no answer attached to it.
            "--mesh-grid" => {
                mesh_grid = Some(PathBuf::from(
                    raw.next().ok_or("--mesh-grid requires an MPAS grid file")?,
                ));
            }
            "--mesh-reference" => {
                mesh_reference = Some(PathBuf::from(
                    raw.next()
                        .ok_or("--mesh-reference requires the other leg's frame directory or file")?,
                ));
            }
            "--mesh-labels" => {
                let value = raw.next().ok_or("--mesh-labels requires A,B")?;
                let mut parts = value.splitn(2, ',').map(str::trim);
                let a = parts.next().unwrap_or("").to_string();
                let b = parts.next().unwrap_or("").to_string();
                if a.is_empty() || b.is_empty() {
                    return Err(CliError::Usage(format!(
                        "--mesh-labels '{value}' is not two comma-separated leg names"
                    )));
                }
                mesh_labels = (a, b);
            }
            "--mesh-bounds" => {
                let value = raw.next().ok_or("--mesh-bounds requires W,E,S,N")?;
                let parts: Vec<f64> = value
                    .split(',')
                    .map(str::trim)
                    .filter_map(|part| part.parse::<f64>().ok())
                    .collect();
                if parts.len() != 4 || parts.iter().any(|value| !value.is_finite()) {
                    return Err(CliError::Usage(format!(
                        "--mesh-bounds '{value}' is not four finite degrees W,E,S,N"
                    )));
                }
                if parts[2] >= parts[3] {
                    return Err(CliError::Usage(format!(
                        "--mesh-bounds '{value}': south {} is not below north {}",
                        parts[2], parts[3]
                    )));
                }
                mesh_bounds = Some((parts[0], parts[1], parts[2], parts[3]));
            }
            // The footer strip's caption fields.  The strip itself is the
            // theme's; these fill it.  Setting none of them draws none.
            "--footer-title" => {
                footer.product_title = Some(raw.next().ok_or("--footer-title requires text")?);
            }
            "--footer-valid" => {
                footer.valid_time = Some(raw.next().ok_or("--footer-valid requires text")?);
            }
            "--footer-mesh" => {
                footer.mesh_or_grid = Some(raw.next().ok_or("--footer-mesh requires text")?);
            }
            "--footer-leg" => {
                footer.leg = Some(raw.next().ok_or("--footer-leg requires text")?);
            }
            "--footer-note" => {
                footer.note = Some(raw.next().ok_or("--footer-note requires text")?);
            }
            // A run difference: the positional wrfouts are run A, these
            // are run B.
            "--diff-against" => {
                diff_against.push(PathBuf::from(
                    raw.next().ok_or("--diff-against requires run B's wrfout")?,
                ));
            }
            "--diff-labels" => {
                let value = raw.next().ok_or("--diff-labels requires A,B")?;
                let mut parts = value.splitn(2, ',').map(str::trim);
                let a = parts.next().unwrap_or("").to_string();
                let b = parts.next().unwrap_or("").to_string();
                if a.is_empty() || b.is_empty() {
                    return Err(CliError::Usage(format!(
                        "--diff-labels '{value}' is not two comma-separated run names"
                    )));
                }
                diff_labels = Some(rustwx_render::difference::DifferenceLabels { a, b });
            }
            "--diff-label-a" | "--diff-label-b" => {
                let value = raw.next().ok_or_else(|| {
                    CliError::Usage(format!("{arg} requires a run name"))
                })?;
                let value = value.trim();
                if value.is_empty() {
                    return Err(CliError::Usage(format!("{arg} requires a nonempty run name")));
                }
                let labels = diff_labels.get_or_insert_with(Default::default);
                if arg == "--diff-label-a" {
                    labels.a = value.to_string();
                } else {
                    labels.b = value.to_string();
                }
            }
            "--diff-sheet" => diff_sheet = true,
            "--heavy" => heavy = true,
            // The wind layer, at the front door.  Drawing streamlines was
            // reachable only through RUSTWX_WIND_STREAMLINES, a name in no
            // help text and no document, so the capability was engine-only.
            // The variable still works; these two outrank it, because a
            // flag the environment can silently overrule is a lie.
            "--streamlines" => streamlines = Some(true),
            "--barbs" => streamlines = Some(false),
            "--list-products" => list_products = true,
            "--help" | "-h" => return Err(CliError::Help),
            // Answered before anything else is validated, and on stdout:
            // the caller is asking what contract this BINARY speaks, not
            // asking it to do work, so no other argument is required and
            // none can suppress the answer.
            "--abi" => return Ok(Invocation::Abi),
            _ if arg.starts_with('-') => {
                return Err(CliError::Usage(format!("unknown option {arg}")));
            }
            _ => inputs.push(PathBuf::from(arg)),
        }
    }

    // Asking what this build can render is answerable without a store, a
    // destination or an input; demanding all three first is why the field
    // report saw --list-products answer with the usage line.
    if list_products && inputs.is_empty() && store_root.is_none() && out_dir.is_none() {
        return Ok(Invocation::Catalog);
    }

    if inputs.is_empty() && from_import.is_none() {
        return Err(CliError::Usage(
            "at least one wrfout input is required".to_string(),
        ));
    }
    let layout = resolve_layout(layout_spec.as_deref(), width, height, size_class, layout_scale)
        .map_err(CliError::Usage)?;
    let width = width.unwrap_or(1_200);
    let height = height.unwrap_or(900);
    let theme = theme_spec
        .or_else(|| {
            std::env::var(rustwx_render::THEME_ENV)
                .ok()
                .filter(|value| !value.trim().is_empty())
        })
        .map(|spec| rustwx_render::RenderTheme::resolve(&spec))
        .transpose()
        .map_err(CliError::Usage)?;
    let radar_colors = match radar_colors_spec {
        Some(name) => rustwx_render::RadarColorSet::parse(&name)
            .map_err(|error| CliError::Usage(format!("--radar-colors: {error}")))?,
        None => rustwx_render::radar_color_set_from_env().map_err(CliError::Usage)?,
    };
    let section = section_spec
        .as_deref()
        .map(section::SectionLine::parse)
        .transpose()
        .map_err(CliError::Usage)?;
    let isotherms = match isotherms_spec {
        Some(text) => section::Isotherms::parse(&text).map_err(CliError::Usage)?,
        None => section::Isotherms::default(),
    };
    let difference = if diff_against.is_empty() {
        if diff_labels.is_some() || diff_sheet {
            return Err(CliError::Usage(
                "--diff-labels and --diff-sheet belong to a difference: name run B with \
                 --diff-against WRFOUT"
                    .to_string(),
            ));
        }
        None
    } else {
        Some(run_difference::DifferenceArgs {
            against: diff_against,
            labels: diff_labels.unwrap_or_default(),
            sheet: diff_sheet,
        })
    };
    Ok(Invocation::Batch(Box::new(Args {
        store_root: store_root.ok_or("--store-root is required")?,
        out_dir: out_dir.ok_or("--out-dir is required")?,
        products,
        frames,
        width,
        height,
        layout,
        heavy,
        list_products,
        streamlines,
        source_label,
        overlays: overlays_path
            .as_deref()
            .map(rustwx_products::geographic_overlays::MapOverlays::load)
            .transpose()
            .map_err(CliError::Usage)?,
        annotations: annotate_path
            .as_deref()
            .map(rustwx_products::geographic_overlays::PanelAnnotations::load)
            .transpose()
            .map_err(CliError::Usage)?,
        theme,
        radar_colors,
        section,
        section_across_km,
        isotherms,
        section_top_km,
        section_size,
        section_reference_km,
        mesh_grid,
        mesh_reference,
        mesh_labels,
        mesh_bounds,
        footer,
        difference,
        save_import,
        from_import,
        inputs,
    })))
}

/// How this invocation's canvases are chosen.
///
/// No flag: auto, the canvas sized from each domain's own shape.  A pixel
/// size with no `--layout`: fixed at that size, so a caller that asks for
/// pixels still gets them.  `--layout auto` WITH a pixel size is refused:
/// the size would be silently dropped, and a caller tiling panels at
/// fixed pixels would get misaligned tiles with no error.
fn resolve_layout(
    spec: Option<&str>,
    width: Option<u32>,
    height: Option<u32>,
    class: rustwx_render::SizeClass,
    scale: f64,
) -> Result<rustwx_render::LayoutMode, String> {
    let sized = width.is_some() || height.is_some();
    match spec.map(|value| value.trim().to_ascii_lowercase()) {
        None if sized => Ok(rustwx_render::LayoutMode::Fixed),
        None => Ok(rustwx_render::LayoutMode::Auto { class, scale }),
        Some(value) if value == "fixed" => Ok(rustwx_render::LayoutMode::Fixed),
        Some(value) if value == "auto" && sized => Err(
            "--layout auto sizes each canvas from its domain; --width/--height would be ignored. \
             Drop them, or pass --layout fixed to draw at that size."
                .to_string(),
        ),
        Some(value) if value == "auto" => Ok(rustwx_render::LayoutMode::Auto { class, scale }),
        Some(value) => Err(format!("--layout must be auto or fixed, got {value:?}")),
    }
}

/// Refuse a command line this build cannot serve, before any file is opened.
///
/// Both checks used to happen deep in the work: `--products` was validated
/// only after a full wrfout import, so a typo cost an import and then
/// reported `No supported WRF files selected` -- the wrong sentence
/// entirely.  A path that does not exist, or that is not a WRF file, was
/// reported without ever naming the path.
fn validate_request(args: &Args) -> Result<(), CliError> {
    // The two store-free families are split off before the store's own
    // vocabulary is consulted.  `partition_products` knows the STORE's
    // slugs, so it reads `mesh:qi:colmax` and `xsec:QICE` as typos and
    // refuses a command line that is correct -- which is what a
    // mesh-only or section-only invocation is.
    let (non_fire,fire_products)=sfire::split_products(&args.products);
    let (non_mesh, mesh_products) =
        mesh::split_product_spec(&non_fire).map_err(CliError::Usage)?;
    let (store_products, section_products) =
        section::split_product_spec(&non_mesh).map_err(CliError::Usage)?;
    // A spec made only of the catalog keywords this binary expands
    // against the store (`all`, `windowed`, `variables`) names no slug
    // the partition could check; it is expanded after the import.
    if (!store_products.trim().is_empty()
        || (mesh_products.is_empty() && section_products.is_empty() && fire_products.is_empty()))
        && !is_catalog_keyword_spec(&store_products)
    {
        rusty_weather::render_all::partition_products(&store_products)
            .map_err(|err| CliError::Usage(err.to_string()))?;
    }
    if let Some(difference) = &args.difference {
        run_difference::validate(
            args,
            difference,
            !mesh_products.is_empty() || !section_products.is_empty(),
        )?;
    }
    if !mesh_products.is_empty() {
        // A mesh: input is an MPAS history frame, which is deliberately NOT
        // a wrfout and would fail the readability check below by design.
        // The mesh reader refuses a file that is not one, by name, with the
        // dimension it looked for.
        if args.mesh_grid.is_none() {
            return Err(CliError::Usage(
                "mesh: products need --mesh-grid FILE.nc: a history frame carries cell CENTRES                  and no cell boundaries, so there are no polygons to draw without it."
                    .to_string(),
            ));
        }
        for path in &args.inputs {
            if !path.is_file() {
                return Err(CliError::Failed(format!(
                    "{}: unreadable history frame (no such file)",
                    path.display()
                )));
            }
        }
        return Ok(());
    }

    // A draw from a saved import opens no frame: the store run is the input.
    if args.from_import.is_some() {
        return Ok(());
    }
    // Per-path, in the order given, so the first sentence names the first
    // problem.  The wording matches `gpuwm/render.py`'s matplotlib engine
    // ("{path}: unreadable wrfout ({exc})") so `--engine auto` cannot change
    // what a script reads.
    let mut unreadable = Vec::new();
    for path in &args.inputs {
        if !path.exists() {
            unreadable.push(format!("{}: unreadable wrfout (no such file)", path.display()));
        } else if !path.is_file() {
            unreadable.push(format!(
                "{}: unreadable wrfout (not a regular file)",
                path.display()
            ));
        } else if !wrf_process::is_supported_wrf_file(path) {
            unreadable.push(format!(
                "{}: unreadable wrfout (not a raw WRF or post-processed NetCDF \
                 file this build recognises)",
                path.display()
            ));
        }
    }
    // Existing semantics: a mixed set renders the files that work.  Only a
    // set with nothing usable in it is a refusal -- but it now says which
    // paths, and why each one, instead of "No supported WRF files selected".
    if unreadable.len() == args.inputs.len() {
        return Err(CliError::Failed(unreadable.join("\n")));
    }
    Ok(())
}

/// The run's own grid identity, read from the inputs' WRF global
/// attributes: the nest number (`GRID_ID`, else a `wrfout_dNN` filename)
/// and the horizontal spacing (`DX`, metres).
#[derive(Debug, Default, PartialEq)]
struct GridIdentity {
    domain: Option<String>,
    spacing_m: Option<f64>,
    /// The grid a producer named for the headline (`GPUWM_GRID_LABEL`,
    /// `hex 15 km`), which outranks the WRF nest token when present.
    label: Option<String>,
    /// What the headline calls a WRF grid in place of its nest token
    /// (`GPUWM_GRID_NAME`, `nest`: `nest 3 km`, not `d01 3 km`); the
    /// filename's domain slot keeps the nest token.
    name: Option<String>,
}

/// The global attribute a wrfout-shaped file names its grid in, for the
/// plot headline and the filename's domain slot (`rw_mpas_convert` writes
/// it; `rw_mpas::convert::GRID_LABEL_ATTRIBUTE` pins the same literal).
///
/// THE BREAKAGE THIS PREVENTS, MEASURED (2026-10-10): a resampled hex mesh
/// forecast carries `GRID_ID = 1` and the resample grid's `DX`, so every
/// hex map was headlined `(d01 20 km)` and filed `..._d01-20km_...` -- a
/// WRF nest the model does not have and a spacing the model does not run
/// at. A file without the attribute (every WRF file) names its grid
/// exactly as before.
const GRID_LABEL_ATTRIBUTE: &str = "GPUWM_GRID_LABEL";

/// The global attribute a WRF-shaped file names its grid in for the
/// headline ONLY: it replaces the nest token (`d01`) before the spacing,
/// which still comes from `DX`, and leaves the filename's domain slot and
/// the subtitle alone (`gpuwm.runtime` writes it from `GPUWM_GRID_NAME`).
///
/// THE BREAKAGE THIS PREVENTS (WOOF Hex with Regional, 2026-10-28,
/// hexreg-nest canary run_9a0796b2e9c8e251): a nest added to a hex run is
/// its own one-domain ArWen run, so its file says `GRID_ID = 1` and every
/// nest map was headlined `d01 3 km` while the site files it as d05.  It is
/// not `GPUWM_GRID_LABEL` because that one also renames the domain folder
/// (`d01-3km` -> `nest3km`), which the WOOF worker matches a nest's column
/// stores to, and turns the subtitle's `Δx` into a render grid.
const GRID_NAME_ATTRIBUTE: &str = "GPUWM_GRID_NAME";

/// The global attribute a converted mesh frame says what its `REFC` holds
/// in (`rw_mpas::convert::REFC_NOTE_ATTRIBUTE` writes the same literal);
/// the note is added to the composite reflectivity maps' titles.
///
/// THE BREAKAGE THIS PREVENTS (WOOF review, 2026-10-29): a coarse hex
/// row's REFC carries the cumulus scheme's Z-R reflectivity
/// (fix/hex-convective-refc), and nothing on its map said its echoes are
/// mostly parameterised rain rather than hydrometeors the model carried.
const REFC_NOTE_ATTRIBUTE: &str = "GPUWM_REFC_NOTE";

/// The recipes whose field is `REFC`, which a `GPUWM_REFC_NOTE` is about.
const REFC_RECIPES: [&str; 2] = ["composite_reflectivity", "composite_reflectivity_uh"];

/// A file's grid label as the headline may print it: whitespace collapsed,
/// at most 32 characters, no control character and no `|`; anything else
/// is ignored rather than drawn.
fn checked_grid_label(raw: &str) -> Option<String> {
    let label = raw.split_whitespace().collect::<Vec<_>>().join(" ");
    (!label.is_empty()
        && label.chars().count() <= 32
        && !label.chars().any(|ch| ch.is_control() || ch == '|'))
    .then_some(label)
}

/// `hex 15 km` -> `hex15km`, `hex 60-3 km` -> `hex60-3km`: the label as a
/// filename token, lower case, with nothing but letters, digits, `-` and
/// `.` kept.
fn grid_label_slug(label: &str) -> Option<String> {
    let slug: String = label
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric() || *ch == '-' || *ch == '.')
        .map(|ch| ch.to_ascii_lowercase())
        .collect();
    (!slug.is_empty()).then_some(slug)
}

/// One identity for the whole invocation, or nothing.
///
/// Several inputs import into ONE store and render as one run, so a
/// per-file token would be a lie on the shared output. Inputs that
/// disagree therefore yield no token at all: the generic `native_grid`
/// slug is accurate about a mixed run in a way that `d02` would not be.
fn grid_identity(inputs: &[PathBuf]) -> GridIdentity {
    let mut identity: Option<GridIdentity> = None;
    for path in inputs {
        let found = file_grid_identity(path);
        match &identity {
            None => identity = Some(found),
            Some(first) if *first == found => {}
            Some(_) => return GridIdentity::default(),
        }
    }
    identity.unwrap_or_default()
}

fn file_grid_identity(path: &std::path::Path) -> GridIdentity {
    let file = wrf_core::WrfFile::open(path).ok();
    let domain = file
        .as_ref()
        .and_then(|file| file.global_attr_i32("GRID_ID").ok())
        .and_then(domain_token_from_id)
        .or_else(|| domain_token_from_filename(path));
    let spacing_m = file
        .as_ref()
        .and_then(|file| file.global_attr_f64("DX").ok())
        .filter(|value| value.is_finite() && *value > 0.0);
    let label = file
        .as_ref()
        .and_then(|file| file.global_attr_str(GRID_LABEL_ATTRIBUTE).ok())
        .and_then(|raw| checked_grid_label(&raw));
    let name = file
        .as_ref()
        .and_then(|file| file.global_attr_str(GRID_NAME_ATTRIBUTE).ok())
        .and_then(|raw| checked_grid_label(&raw));
    GridIdentity { domain, spacing_m, label, name }
}

/// The note any input states about its `REFC` (`GPUWM_REFC_NOTE`), checked
/// as a headline label is.  Not part of [`GridIdentity`]: a frame where
/// the cumulus scheme has not rained yet is on the same grid.
fn inputs_refc_note(inputs: &[PathBuf]) -> Option<String> {
    inputs.iter().find_map(|path| {
        wrf_core::WrfFile::open(path)
            .ok()?
            .global_attr_str(REFC_NOTE_ATTRIBUTE)
            .ok()
            .and_then(|raw| checked_grid_label(&raw))
    })
}

/// The caller's annotations with the REFC note on the REFC recipes' titles.
fn with_refc_note(
    annotations: Option<rustwx_products::geographic_overlays::PanelAnnotations>,
    note: Option<String>,
) -> Option<rustwx_products::geographic_overlays::PanelAnnotations> {
    let Some(note) = note else { return annotations };
    let mut annotations = annotations.unwrap_or_default();
    for recipe in REFC_RECIPES {
        annotations.title_notes.entry(recipe.to_string()).or_insert_with(|| note.clone());
    }
    Some(annotations)
}

fn domain_token_from_id(grid_id: i32) -> Option<String> {
    (1..=99).contains(&grid_id).then(|| format!("d{grid_id:02}"))
}

/// `wrfout_d02_1974-04-03_18:00:00` -> `d02`, for files whose `GRID_ID`
/// is absent (idealized and hand-built wrfouts).
fn domain_token_from_filename(path: &std::path::Path) -> Option<String> {
    let name = path.file_name()?.to_str()?;
    let rest = name.strip_prefix("wrfout_d")?;
    let digits: String = rest.chars().take(2).collect();
    if digits.len() != 2 || !digits.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    Some(format!("d{digits}"))
}

// The resolution token and its parts live in the library, so the fire
// renderer (rw_wrfbatch::sfire) names its domain folder the same way.
use rw_wrfbatch::domain_naming::{resolution_token, spacing_parts};

/// `Δx 3 km`, `Δx 333 m` -- the same number, spelled for the subtitle.
fn spacing_subtitle(spacing_m: f64) -> Option<String> {
    let (value, unit) = spacing_parts(spacing_m)?;
    Some(format!("\u{0394}x {value} {unit}"))
}

/// The subtitle's spacing token for this run: the model's own `Δx` for a
/// file that IS the model grid (every WRF file), `render grid 22 km` for a
/// file a producer resampled from a model mesh it named in
/// `GPUWM_GRID_LABEL` -- there `DX` is the picture's grid, not the model's.
///
/// THE BREAKAGE THIS PREVENTS, MEASURED (2026-10-28): RRFS-MPAS maps of the
/// 120 km and 30 km meshes, drawn on the fixed 22 km focus window, printed
/// `Δx 22 km` beside a headline that said `30 km`, so one picture stated
/// two model spacings and the subtitle's was the resample grid's.
fn subtitle_spacing(identity: &GridIdentity) -> Option<String> {
    let spacing_m = identity.spacing_m?;
    if identity.label.is_some() {
        let (value, unit) = spacing_parts(spacing_m)?;
        Some(format!("render grid {value} {unit}"))
    } else {
        spacing_subtitle(spacing_m)
    }
}


/// The slug that replaces `native_grid` in every output filename:
/// `d02-3km`, or `d02` when the file declares no usable `DX`.
fn native_domain_slug(identity: &GridIdentity) -> Option<String> {
    if let Some(slug) = identity.label.as_deref().and_then(grid_label_slug) {
        return Some(slug);
    }
    let domain = identity.domain.as_ref()?;
    match identity.spacing_m.and_then(resolution_token) {
        Some(resolution) => Some(format!("{domain}-{resolution}")),
        None => Some(domain.clone()),
    }
}

/// `d02 750 m` -- the same identity as the filename slug, spelled for a
/// human reading the plot's headline.  The spacing is spelled the way the
/// subtitle spells it, so one grid is not two different numbers on one
/// image.
///
/// This is what the headline's parenthetical is FOR.  Without it the
/// renderer fell back to the store model's registered source catalog and
/// stamped that catalog's dataset id -- a dataset this lane never read,
/// naming an archive these numbers did not come from, on every frame.
fn domain_title_label(identity: &GridIdentity) -> Option<String> {
    if let Some(label) = &identity.label {
        return Some(label.clone());
    }
    let domain = identity.name.as_ref().or(identity.domain.as_ref())?;
    match identity.spacing_m.and_then(spacing_parts) {
        Some((value, unit)) => Some(format!("{domain} {value} {unit}")),
        None => Some(domain.clone()),
    }
}

/// One import note as its own greppable stderr record.  Tab-separated so
/// the note text (which contains spaces and colons) stays one field.
fn import_note_line(note: &str) -> String {
    format!("IMPORT_NOTE\t{note}")
}

fn run(mut args: Args) -> Result<(), String> {
    // Recorded before any product lane builds a wind layer, and cleared to
    // `None` when neither flag was given so RUSTWX_WIND_STREAMLINES and the
    // automatic per-grid choice keep their existing meaning.
    rustwx_products::shared_context::set_wind_streamline_request(args.streamlines);
    // The theme is installed before the first presentation is built and
    // before the first glyph is drawn (its fonts load with it).  Absent, the
    // renderer's own look is installed by name so a later RUSTWX_THEME read
    // cannot restyle half a run.
    let theme_name = match args.theme.take() {
        Some(theme) => {
            let name = theme.name.clone();
            rustwx_render::install_theme(theme)?;
            name
        }
        None => {
            rustwx_render::install_theme(rustwx_render::RenderTheme::default_theme())?;
            "default".to_string()
        }
    };
    rustwx_render::set_layout_mode(args.layout);
    rustwx_render::theme::set_template_version(
        args.source_label
            .split_whitespace()
            .last()
            .filter(|token| token.starts_with(|ch: char| ch.is_ascii_digit()))
            .map(str::to_string),
    );
    println!("THEME {theme_name}");
    // Selected before the first product resolves its scale, so one run
    // draws one radar look.
    rustwx_render::install_radar_color_set(args.radar_colors)?;
    // The section crate draws its own text; a theme with fonts hands it the
    // same bytes so one theme names the type on every surface.
    {
        let theme = rustwx_render::active_theme();
        if theme.font_regular.is_some() || theme.font_bold.is_some() {
            let read = |path: &Option<PathBuf>| path.as_ref().and_then(|p| std::fs::read(p).ok());
            rustwx_cross_section::install_cross_section_fonts(
                read(&theme.font_regular),
                read(&theme.font_bold),
            );
        }
    }
    // The caption fields for the theme's footer strip, if the theme names
    // one and the caller filled any.  The grid row defaults to the run's own
    // domain and spacing, read from the inputs' global attributes -- the
    // same identity the subtitle already carries, so one plot never states
    // two different grids.
    //
    // Installed HERE, before the store-free families are dispatched: a
    // section-only or mesh-only invocation returns without ever reaching
    // the import, so a strip installed after it was a strip those panels
    // could never carry.
    if args.footer != rustwx_render::FooterFields::default() {
        let mut footer = args.footer.clone();
        if footer.mesh_or_grid.is_none() {
            footer.mesh_or_grid = domain_title_label(&run_grid_identity(&args));
        }
        rustwx_render::set_footer_fields(footer);
    }
    if args.difference.is_some() {
        return run_difference::run(args);
    }
    // Two families never touch the store.  `mesh:` reads an MPAS history
    // frame and its grid file; `xsec:` cuts the wrfout files directly.
    // Both are split off here, and when they are all that was asked for the
    // import is skipped entirely.
    let (non_fire_products,fire_products)=sfire::split_products_for(&args.products,&args.inputs);
    let (non_mesh_products, mesh_products) = mesh::split_product_spec(&non_fire_products)?;
    let (store_products, section_products) = section::split_product_spec(&non_mesh_products)?;
    if !mesh_products.is_empty() && !(store_products.is_empty() && section_products.is_empty() && fire_products.is_empty()) {
        // The INPUTS differ, not just the drawing.  A mesh: product's input
        // is an MPAS history frame; every other family's is a wrfout.  One
        // invocation cannot be handed both lists, and importing a history
        // frame as a wrfout fails deep inside the importer with a message
        // about a missing Times variable -- which says nothing about the
        // real mistake.
        return Err(format!(
            "mesh: products read an MPAS history frame plus --mesh-grid; {} \
             read wrfout files. Ask for them in separate invocations.",
            if section_products.is_empty() {
                "the store products"
            } else {
                "the store and xsec: products"
            }
        ));
    }
    if !mesh_products.is_empty() {
        let grid_path = args.mesh_grid.as_deref().ok_or_else(|| {
            "mesh: products need the mesh: --mesh-grid FILE.nc. A history frame carries cell \
             CENTRES and no cell boundaries, so there are no polygons to draw without it."
                .to_string()
        })?;
        let read_started = std::time::Instant::now();
        let geometry = mesh::read_mesh_geometry(grid_path)?;
        println!(
            "MESH grid={} cells={} spacing_km={:.3}-{:.3} description={} read_ms={}",
            grid_path.display(),
            geometry.n_cells,
            geometry.spacing_km.0,
            geometry.spacing_km.1,
            geometry.description(),
            read_started.elapsed().as_millis()
        );
        let config = mesh::MeshRenderConfig {
            inputs: &args.inputs,
            out_dir: &args.out_dir,
            grid: &geometry,
            reference: args.mesh_reference.as_deref(),
            labels: args.mesh_labels.clone(),
            width: args.width,
            height: args.height,
            source_label: args.source_label.clone(),
            theme: rustwx_render::active_theme(),
            frame: args.frames,
            bounds: args.mesh_bounds,
            footer: args.footer.clone(),
        };
        let mesh_started = std::time::Instant::now();
        let (rendered, failed) =
            mesh::render_mesh_products(&mesh_products, &config, |outcome| match outcome.result {
                Ok(path) => {
                    // The timing rides its OWN line.  `RENDERED <slug>
                    // <path>` is a two-field record every reader of this
                    // stream splits on the first space, and a third field
                    // would land inside the path they read.
                    println!(
                        "MESH_RENDER {} cells={} render_ms={}",
                        outcome.slug, outcome.cells, outcome.render_ms
                    );
                    println!("RENDERED {} {}", outcome.slug, path.display());
                }
                Err(err) => eprintln!("FAILED {} {err}", outcome.slug),
            })?;
        println!(
            "FINISHED rendered={rendered} skipped=0 failed={failed} elapsed_ms={}",
            mesh_started.elapsed().as_millis()
        );
        if rendered == 0 || failed > 0 {
            return Err(format!(
                "mesh render incomplete: rendered={rendered} failed={failed}"
            ));
        }
        return Ok(());
    }
    if !section_products.is_empty() && args.section.is_none() {
        return Err(
            "xsec: products need a line: --section lat,lon,lat,lon or --section FILE.json"
                .to_string(),
        );
    }
    let section_inputs = args.inputs.clone();
    let section_out_dir = args.out_dir.clone();
    let section_source_label = args.source_label.clone();
    let section_domain_slug = native_domain_slug(&run_grid_identity(&args));
    let (section_width, section_height) = match (args.layout, args.section_size) {
        (rustwx_render::LayoutMode::Auto { class, scale }, None) => {
            let plan = rustwx_render::LayoutTable::builtin().plan_section(class, scale);
            (plan.canvas_w, plan.canvas_h)
        }
        _ => section_dimensions(args.section_size, args.width, args.height),
    };
    let section_args = SectionArgs {
        line: args.section.clone(),
        across_km: args.section_across_km,
        isotherms: args.isotherms.clone(),
        frame: args.frames,
        width: section_width,
        height: section_height,
        top_km: args.section_top_km,
        reference_km: args.section_reference_km,
    };
    let section_started = std::time::Instant::now();
    let fire_rows=sfire::catalog(&args.inputs);
    let (fire_rendered,fire_skipped,fire_failed,fire_georefs)=if args.list_products || fire_products.is_empty() {(0,0,0,Vec::new())} else {
        sfire::render(&fire_products,&sfire::RenderConfig {inputs:&args.inputs,out_dir:&args.out_dir,
            frame:args.frames,width:args.width,height:args.height,source_label:&args.source_label,
            overlays:args.overlays.as_ref(),annotations:args.annotations.as_ref()})?
    };
    if store_products.is_empty() && section_products.is_empty() && !fire_products.is_empty() {
        if args.list_products {
            for (slug,kind,status,detail,code) in &fire_rows {println!("PRODUCT\t{slug}\t{kind}\t{status}\t{detail}\t{code}");}
            println!("CATALOG total={}",fire_rows.len());
        } else {
            if !fire_georefs.is_empty() {write_georef_manifest(&args.out_dir,&fire_georefs)?;}
            println!("FINISHED rendered={fire_rendered} skipped={fire_skipped} failed={fire_failed} elapsed_ms={}",section_started.elapsed().as_millis());
            if fire_failed>0 || (fire_rendered==0 && !sfire::all_inactive(&args.inputs)) {return Err("Fire render incomplete; see frame-attributed diagnostics".into());}
        }
        return Ok(());
    }
    if store_products.is_empty() && !section_products.is_empty() && !args.list_products {
        let (rendered, failed) = render_section_products(
            &section_products,
            &section_args,
            &section_inputs,
            &section_out_dir,
            section_domain_slug,
            &section_source_label,
        )?;
        println!(
            "FINISHED rendered={rendered} skipped=0 failed={failed} elapsed_ms={}",
            section_started.elapsed().as_millis()
        );
        if rendered == 0 || failed > 0 {
            return Err(format!(
                "section render incomplete: rendered={rendered} failed={failed}"
            ));
        }
        return Ok(());
    }
    // Read before the import consumes the paths: the domain token and the
    // subtitle spacing come from the inputs' own global attributes, never
    // from the store (rw-store v1 retains no grid-spacing metadata).
    let identity = run_grid_identity(&args);
    let domain_slug = native_domain_slug(&identity);
    let spacing = subtitle_spacing(&identity);
    let title_provenance = TitleProvenance::LocalImport {
        grid_label: domain_title_label(&identity),
    };
    let refc_note = run_refc_note(&args);
    let imported = match &args.from_import {
        Some(path) => saved_import(path)?,
        None => {
            let imported = import_run(
                args.inputs,
                &args.store_root,
                &store_products,
                args.heavy,
                args.list_products,
            )?;
            if let Some(path) = &args.save_import {
                save_import(path, &imported, &identity, refc_note.as_deref())?;
            }
            imported
        }
    };
    let ImportedRun {
        model: import_model,
        run: import_run_slug,
        stored_slots,
        frame_sources,
        run_context: _,
    } = imported;
    if args.list_products {
        return list_products(
            &args.store_root,
            &import_model,
            &import_run_slug,
            &stored_slots,
            args.heavy,
            fire_rows,
        );
    }

    // --frames N is an ordinal index into the ascending stored slots (the
    // Nth stored frame), matching `gpuwm render --timeidx`: within one
    // imported file the plan orders slots by valid time, so the index is
    // the frame index whether the axis is whole-hour (slot = forecast
    // hour) or exact-time (slot = ordinal).
    let hour_scope = match args.frames {
        None => BatchHourScope::AllStored,
        Some(index) => {
            let slot = stored_slots.get(index).copied().ok_or_else(|| {
                format!(
                    "--frames {index} out of range; the store has {} frame(s)",
                    stored_slots.len()
                )
            })?;
            BatchHourScope::Current(slot)
        }
    };
    // The catalog a keyword expands from is the one of the frames this
    // launch draws, never the store's first frame alone: a series whose
    // analysis frame stores no REFL_10CM dropped composite reflectivity
    // from `all` on every frame (`inspect_renderable_products_over`).
    let drawn_slots: Vec<u16> = match hour_scope {
        BatchHourScope::AllStored => stored_slots.clone(),
        BatchHourScope::Current(slot) => vec![slot],
    };
    let catalog = inspect_renderable_products_over(
        &args.store_root,
        &import_model,
        &import_run_slug,
        &drawn_slots,
    )?;
    let product_spec = expand_catalog_keywords(&store_products, &catalog)?;
    println!(
        "CATALOG products={} stored_hours={:?}",
        catalog.products.len(),
        catalog.stored_hours
    );

    // The manifest destination survives `args.out_dir` moving into the
    // request below.
    let georef_out_dir = args.out_dir.clone();
    let request = store_batch_request(StoreBatch {
        store_root: args.store_root,
        model_slug: import_model,
        run_slug: import_run_slug,
        hours: hour_scope,
        stored_frames: stored_slots.len(),
        product_spec,
        out_dir: args.out_dir,
        native_domain_slug: domain_slug,
        subtitle_spacing: spacing,
        source_label: args.source_label,
        title_provenance,
        geographic_overlays: args.overlays,
        panel_annotations: with_refc_note(args.annotations, refc_note),
        width: args.width,
        height: args.height,
    });
    let cancel = AtomicBool::new(false);
    // Every RENDERED panel's georeference (or the lane's reason it has
    // none), collected off the event stream so the run can publish its
    // geographic transforms beside the PNGs.  The pinned stdout grammar
    // is untouched: the collection rides the same event the RENDERED
    // line already prints.
    let mut panel_georefs: Vec<RenderedPanelGeoref> = fire_georefs;
    let summary = run_batch_render(request, &cancel, |event| match event {
        BatchRenderEvent::Started {
            planned_items,
            output_dir,
            ..
        } => println!(
            "RENDER planned={planned_items} out={}",
            output_dir.display()
        ),
        BatchRenderEvent::ItemRendered {
            slug,
            output_path,
            georeference,
            georeference_absent_reason,
            ..
        } => {
            println!("RENDERED {slug} {}", output_path.display());
            panel_georefs.push((output_path, georeference, georeference_absent_reason));
        }
        BatchRenderEvent::ItemSkipped {
            hour, slug, reason, ..
        } => {
            println!(
                "SKIPPED {slug} {}",
                frame_attributed(&frame_sources, hour, &reason)
            )
        }
        BatchRenderEvent::ItemFailed {
            hour, slug, error, ..
        } => {
            eprintln!(
                "FAILED {slug} {}",
                frame_attributed(&frame_sources, hour, &error)
            )
        }
        BatchRenderEvent::Finished(summary) => println!(
            "FINISHED rendered={} skipped={} failed={} elapsed_ms={}",
            summary.rendered+fire_rendered, summary.skipped+fire_skipped, summary.failed+fire_failed, summary.elapsed_ms
        ),
        _ => {}
    })?;
    // Default-on, no flag: a bare run must stop showing the defect (a
    // consumer holding a PNG with no way to put a lat/lon on a pixel).
    // Written before the failure check below so a partially-failed run
    // still georeferences every panel it DID produce.
    write_georef_manifest(&georef_out_dir, &panel_georefs)?;
    // Sections requested beside store products ride the same run, after
    // the store lane, with their own tally folded into the verdict.
    let (section_rendered, section_failed) = if section_products.is_empty() {
        (0, 0)
    } else {
        let counts = render_section_products(
            &section_products,
            &section_args,
            &section_inputs,
            &section_out_dir,
            section_domain_slug,
            &section_source_label,
        )?;
        println!(
            "SECTIONS rendered={} failed={} elapsed_ms={}",
            counts.0,
            counts.1,
            section_started.elapsed().as_millis()
        );
        counts
    };
    if summary.rendered + section_rendered + fire_rendered == 0 || summary.failed + section_failed + fire_failed > 0 {
        return Err(format!(
            "batch render incomplete: rendered={} skipped={} failed={}",
            summary.rendered + section_rendered + fire_rendered,
            summary.skipped + fire_skipped,
            summary.failed + section_failed + fire_failed
        ));
    }
    Ok(())
}

/// A run's inputs, imported into a store: what the render needs from it.
struct ImportedRun {
    model: String,
    run: String,
    /// The stored frame slots, ascending, which is ascending valid time.
    stored_slots: Vec<u16>,
    frame_sources: std::collections::HashMap<u16, PathBuf>,
    /// The import's initial-condition disclosure and model label (what the pictures are labelled with).
    run_context: wrf_process::WrfRunContext,
}

/// The grid identity this draw labels its pictures with: the one a `--save-import` recorded when
/// drawing from it, else the inputs' own global attributes.
/// [`inputs_refc_note`] of the invocation, or the one its saved import recorded.
fn run_refc_note(args: &Args) -> Option<String> {
    if let Some(path) = &args.from_import {
        let value: serde_json::Value = serde_json::from_slice(&std::fs::read(path).ok()?).ok()?;
        return value.get("refc_note").and_then(serde_json::Value::as_str).and_then(checked_grid_label);
    }
    inputs_refc_note(&args.inputs)
}

fn run_grid_identity(args: &Args) -> GridIdentity {
    if let Some(path) = &args.from_import {
        if let Ok(bytes) = std::fs::read(path) {
            if let Ok(value) = serde_json::from_slice::<serde_json::Value>(&bytes) {
                let grid = &value["grid"];
                let text = |key: &str| grid.get(key).and_then(serde_json::Value::as_str).map(str::to_string);
                return GridIdentity {
                    domain: text("domain"),
                    spacing_m: grid.get("spacing_m").and_then(serde_json::Value::as_f64),
                    label: text("label"),
                    name: text("name"),
                };
            }
        }
        return GridIdentity::default();
    }
    grid_identity(&args.inputs)
}

const SAVED_IMPORT_SCHEMA: &str = "rw-wrfbatch.saved-import.v1";

/// THE BREAKAGE THIS SERVES (4x RTX PRO 6000 box, 2026-10-17): a draw reopened every converted frame
/// although the listing had already imported it into the store (22 MB per frame against a 4.3 GB
/// frame), so the products door had to keep every frame of a series on disk until the last map was
/// drawn, and member batches filled the disk.  The listing records here what the draw takes from the
/// frames, so the door can delete each frame as soon as it is imported.
fn save_import(
    path: &std::path::Path,
    imported: &ImportedRun,
    identity: &GridIdentity,
    refc_note: Option<&str>,
) -> Result<(), String> {
    let mut sources: Vec<(u16, String)> = imported
        .frame_sources
        .iter()
        .map(|(slot, file)| (*slot, file.display().to_string()))
        .collect();
    sources.sort();
    let document = serde_json::json!({
        "schema": SAVED_IMPORT_SCHEMA,
        "writer": env!("CARGO_PKG_VERSION"),
        "model": imported.model,
        "run": imported.run,
        "stored_slots": imported.stored_slots,
        "frame_sources": sources,
        "run_context": imported.run_context.to_json(),
        "grid": {"domain": identity.domain, "spacing_m": identity.spacing_m, "label": identity.label,
                 "name": identity.name},
        "refc_note": refc_note,
    });
    let bytes = serde_json::to_vec_pretty(&document).map_err(|err| format!("encode {}: {err}", path.display()))?;
    std::fs::write(path, bytes).map_err(|err| format!("write {}: {err}", path.display()))
}

/// The import a `--save-import` recorded; its run context is installed as the import would have.
fn saved_import(path: &std::path::Path) -> Result<ImportedRun, String> {
    let bytes = std::fs::read(path).map_err(|err| format!("read {}: {err}", path.display()))?;
    let value: serde_json::Value =
        serde_json::from_slice(&bytes).map_err(|err| format!("parse {}: {err}", path.display()))?;
    if value.get("schema").and_then(serde_json::Value::as_str) != Some(SAVED_IMPORT_SCHEMA) {
        return Err(format!("{} is not a {SAVED_IMPORT_SCHEMA} record", path.display()));
    }
    let text = |key: &str| {
        value
            .get(key)
            .and_then(serde_json::Value::as_str)
            .map(str::to_string)
            .ok_or_else(|| format!("{} carries no {key}", path.display()))
    };
    let stored_slots: Vec<u16> = value
        .get("stored_slots")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| format!("{} carries no stored_slots", path.display()))?
        .iter()
        .filter_map(|slot| slot.as_u64().and_then(|n| u16::try_from(n).ok()))
        .collect();
    if stored_slots.is_empty() {
        return Err(format!("{} names no stored slot", path.display()));
    }
    let frame_sources = value
        .get("frame_sources")
        .and_then(serde_json::Value::as_array)
        .map(|rows| {
            rows.iter()
                .filter_map(|row| {
                    let slot = row.get(0)?.as_u64().and_then(|n| u16::try_from(n).ok())?;
                    Some((slot, PathBuf::from(row.get(1)?.as_str()?)))
                })
                .collect()
        })
        .unwrap_or_default();
    let run_context = wrf_process::WrfRunContext::from_json(&value["run_context"]);
    run_context.apply();
    println!("IMPORTED from {} (no frame reopened)", path.display());
    Ok(ImportedRun {
        model: text("model")?,
        run: text("run")?,
        stored_slots,
        frame_sources,
        run_context,
    })
}

/// Import `inputs` into `store_root` for the store products asked for.
fn import_run(
    inputs: Vec<PathBuf>,
    store_root: &std::path::Path,
    store_products: &str,
    heavy: bool,
    list_products: bool,
) -> Result<ImportedRun, String> {
    // A listing asked about NAMED products imports the frames exactly as the
    // render of those products does, so the render that follows into the
    // same store finds that run there and imports nothing
    // (`wrf_process::published_import_record`).  A listing of group keywords
    // alone, the default, still imports in full: it answers what the frames
    // can draw at all.
    let named = !heavy
        && named_product_request(store_products)
        && (!list_products || names_a_product(store_products));
    let mut options = WrfProcessOptions {
        heavy_ecape: heavy,
        snow_since: std::env::var("GPUWM_SNOW_SINCE").ok().filter(|s| !s.trim().is_empty()),
        named_products_only: named,
        // A request of named products imports only what those products
        // read, in ONE import per file for all of them; a listing of the
        // same names imports the same plan, so the render that follows
        // reuses it.  A keyword in the request (`all`, `direct`, `derived`)
        // keeps the named import whole; `windowed` scopes to what its
        // windows read.
        required_fields: if named {
            wrf_process::required_fields_for_products(store_products)
        } else {
            None
        },
        ..WrfProcessOptions::default()
    };
    if !heavy {
        if let Some(rainfall) = rainfall_import_options(store_products) {
            options = rainfall;
        }
    }
    let task = spawn_process_paths(inputs, store_root.to_path_buf(), options);
    let import = loop {
        match task
            .rx
            .recv()
            .map_err(|err| format!("WRF processor exited without a result: {err}"))?
        {
            WrfProcessMessage::Progress(message) => println!("PROCESS {message}"),
            WrfProcessMessage::Done(result) => break result?,
        }
    };
    println!(
        "IMPORTED model={} run={} files={} hours={} variables={} notes={}",
        import.model,
        import.run,
        import.files_seen,
        import.hours_written,
        import.variables.len(),
        import.notes.len()
    );
    // Every note verbatim, not just the count.  A `notes=7` tally hid real
    // uvmet/uvmet10/interpolation failures -- named products degraded or
    // vanished with nothing on the transcript saying why.
    for note in &import.notes {
        eprintln!("{}", import_note_line(note));
    }
    let frame_sources: std::collections::HashMap<u16, PathBuf> =
        import.frame_sources.iter().cloned().collect();

    let run_manifest = store_root
        .join(&import.model)
        .join(&import.run)
        .join("run.json");
    let manifest: serde_json::Value = serde_json::from_slice(
        &std::fs::read(&run_manifest)
            .map_err(|err| format!("read {}: {err}", run_manifest.display()))?,
    )
    .map_err(|err| format!("parse {}: {err}", run_manifest.display()))?;
    let stored_slots: Vec<u16> = {
        let mut slots: Vec<u16> = manifest
            .get("hours")
            .and_then(serde_json::Value::as_object)
            .ok_or_else(|| format!("{} has no hours object", run_manifest.display()))?
            .keys()
            .filter_map(|key| key.parse::<u16>().ok())
            .collect();
        slots.sort_unstable();
        slots
    };
    if stored_slots.is_empty() {
        return Err(format!("{} has no stored forecast slots", run_manifest.display()));
    }
    Ok(ImportedRun {
        model: import.model,
        run: import.run,
        stored_slots,
        frame_sources,
        run_context: import.run_context.clone(),
    })
}

/// What one store render asks of the batch renderer.
struct StoreBatch {
    store_root: PathBuf,
    model_slug: String,
    run_slug: String,
    hours: BatchHourScope,
    stored_frames: usize,
    product_spec: String,
    out_dir: PathBuf,
    native_domain_slug: Option<String>,
    subtitle_spacing: Option<String>,
    source_label: String,
    title_provenance: TitleProvenance,
    geographic_overlays: Option<rustwx_products::geographic_overlays::MapOverlays>,
    panel_annotations: Option<rustwx_products::geographic_overlays::PanelAnnotations>,
    width: u32,
    height: u32,
}

/// The batch request for one store render, with its limits sized to it.
fn store_batch_request(batch: StoreBatch) -> BatchRenderRequest {
    // Size the limits to the request: this is a command-line job whose
    // work is exactly frames x products, not a GUI guarding against an
    // accidental unbounded click.
    let selected_frames = match batch.hours {
        BatchHourScope::AllStored => batch.stored_frames.max(1),
        BatchHourScope::Current(_) => 1,
    };
    let per_frame_products = requested_product_count(&batch.product_spec);
    let mut limits = BatchRenderLimits::default();
    limits.max_hours = limits.max_hours.max(selected_frames);
    limits.max_products_per_hour = limits.max_products_per_hour.max(per_frame_products);
    limits.max_work_items = limits.max_work_items.max(
        selected_frames
            .saturating_mul(per_frame_products)
            .saturating_add(per_frame_products),
    );
    limits.max_output_width = batch.width.max(limits.max_output_width);
    limits.max_output_height = batch.height.max(limits.max_output_height);
    limits.max_output_pixels = u64::from(batch.width) * u64::from(batch.height);
    BatchRenderRequest {
        store_root: batch.store_root,
        model_slug: batch.model_slug,
        run_slug: batch.run_slug,
        hours: batch.hours,
        product_spec: batch.product_spec,
        out_dir: batch.out_dir,
        domain: BatchRenderDomain::NativeGrid,
        native_domain_slug: batch.native_domain_slug,
        subtitle_spacing: batch.subtitle_spacing,
        source_label: Some(batch.source_label),
        title_provenance: batch.title_provenance,
        date_yyyymmdd: None,
        cycle_utc: None,
        source: None,
        geographic_overlays: batch.geographic_overlays,
        panel_annotations: batch.panel_annotations,
        output_width: batch.width,
        output_height: batch.height,
        limits,
    }
}

/// What the section lane needs from the invocation, captured before the
/// store lane consumes the rest of `Args`.
struct SectionArgs {
    line: Option<section::SectionLine>,
    across_km: Option<f64>,
    isotherms: section::Isotherms,
    frame: Option<usize>,
    width: u32,
    height: u32,
    top_km: f64,
    reference_km: Option<f64>,
}

/// The size a SECTION is drawn at: what the caller asked for, or landscape
/// 2:1 at the map's width.
///
/// WHAT BREAKAGE THIS PREVENTS (gate law): the pair tool handed
/// sections the MAP's size and a 1800x1464 near-square map produced a
/// 1800x1464 near-square section -- a vertical cut, which is 100 km wide
/// and 6 km tall, drawn in a portrait-ish frame.
pub fn section_dimensions(
    explicit: Option<(u32, u32)>,
    map_width: u32,
    map_height: u32,
) -> (u32, u32) {
    if let Some((width, height)) = explicit {
        return (width, height);
    }
    let width = map_width.max(map_height).max(640);
    (width, (width / 2).max(320))
}

/// The `xsec:` lane: sections cut from the wrfout files directly and
/// reported through the RENDERED / FAILED grammar the Python side reads.
/// Returns `(rendered, failed)`.
fn render_section_products(
    products: &[section::SectionProduct],
    args: &SectionArgs,
    inputs: &[PathBuf],
    out_dir: &std::path::Path,
    domain_slug: Option<String>,
    source_label: &str,
) -> Result<(usize, usize), String> {
    let line = args
        .line
        .clone()
        .ok_or_else(|| "xsec: products need --section".to_string())?;
    println!(
        "SECTIONS products={} line={:.4},{:.4}->{:.4},{:.4} length_km={:.1}",
        products.len(),
        line.start.lat_deg,
        line.start.lon_deg,
        line.end.lat_deg,
        line.end.lon_deg,
        line.length_km()
    );
    let config = section::SectionRenderConfig {
        inputs,
        out_dir,
        line,
        across_km: args.across_km,
        isotherms: args.isotherms.clone(),
        frame: args.frame,
        width: args.width,
        height: args.height,
        top_km: args.top_km,
        reference_km: args.reference_km,
        domain_slug,
        source_label: source_label.to_string(),
        theme: rustwx_render::active_theme(),
    };
    section::render_sections(products, &config, |outcome| match outcome.result {
        Ok(path) => {
            println!("RENDERED {} {}", outcome.slug, path.display());
            // The bar a cut is drawn on is fitted to the frame at both
            // ends, so the picture cannot say which bar it is.  The
            // caller records this beside the ceiling, and two cuts of
            // one line an hour apart are compared through the receipt.
            if let Some(drawn) = outcome.drawn {
                if drawn.lo.is_finite() && drawn.hi.is_finite() {
                    println!(
                        "SECTIONFILL {} lo={} hi={} absence={} rule={}",
                        outcome.slug,
                        drawn.lo,
                        drawn.hi,
                        u8::from(drawn.absence),
                        drawn.rule.label()
                    );
                }
            }
        }
        Err(err) => eprintln!("FAILED {} {err}", outcome.slug),
    })
}

/// The schema string every `render-georef.json` declares.
const GEOREF_MANIFEST_SCHEMA: &str = "rustwx.render-georef/v1";

/// One rendered panel's georeference report off the event stream: the
/// output path, the transform when the lane published one, and the lane's
/// reason when it did not.
type RenderedPanelGeoref = (
    PathBuf,
    Option<rustwx_render::PanelGeoReference>,
    Option<String>,
);

/// `<out_dir>/render-georef.json`: what each PNG of one render run maps to
/// on the Earth.
///
/// This file is the fix for a measured defect: `rw_wrfbatch` published no
/// geographic transform anywhere, so a consumer holding a rendered PNG
/// recovered one by registration -- and got 4.175 px/deg longitude against
/// 4.38 px/deg latitude on a global panel, an aspect no flat map has,
/// because the panel was Robinson and no linear fit describes Robinson.
/// Different products in ONE batch also draw in different projections, so
/// the transforms are per-panel, keyed by the PNG's path relative to the
/// output directory (forward slashes).  Every panel that got no transform
/// is listed in `without_georeference` with the reason -- a silent
/// omission is exactly the defect being fixed.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct GeorefManifest {
    schema: String,
    generated_utc: String,
    panels: std::collections::BTreeMap<String, rustwx_render::PanelGeoReference>,
    without_georeference: Vec<GeorefAbsence>,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct GeorefAbsence {
    path: String,
    reason: String,
}

/// The manifest key for one output: relative to the run's output
/// directory, forward slashes, so the file stays meaningful when the run
/// directory moves between machines.
fn georef_manifest_key(out_dir: &std::path::Path, output_path: &std::path::Path) -> String {
    output_path
        .strip_prefix(out_dir)
        .unwrap_or(output_path)
        .to_string_lossy()
        .replace('\\', "/")
}

fn build_georef_manifest(
    out_dir: &std::path::Path,
    generated_utc: String,
    panels: &[RenderedPanelGeoref],
) -> GeorefManifest {
    let mut published = std::collections::BTreeMap::new();
    let mut without_georeference = Vec::new();
    for (output_path, georeference, absent_reason) in panels {
        let key = georef_manifest_key(out_dir, output_path);
        match georeference {
            Some(georeference) => {
                published.insert(key, georeference.clone());
            }
            None => without_georeference.push(GeorefAbsence {
                path: key,
                // The lane's own reason travels with the event; a lane
                // that supplied neither still gets a non-empty entry
                // naming the gap, never a silent omission.
                reason: absent_reason.clone().unwrap_or_else(|| {
                    "the render lane threaded neither a georeference nor an absence \
                     reason through its rendered-product record"
                        .to_string()
                }),
            }),
        }
    }
    GeorefManifest {
        schema: GEOREF_MANIFEST_SCHEMA.to_string(),
        generated_utc,
        panels: published,
        without_georeference,
    }
}

/// Fold one batch's manifest into the one already on disk.
///
/// WHAT BREAKAGE THIS PREVENTS: the caller renders a run as many
/// `rw_wrfbatch` invocations into ONE output directory (one per history
/// file and per grid), and each invocation used to write the file from
/// its own batch alone.  On a real 6 h two-grid run of 646 pictures the
/// manifest listed 60, all of them the last grid's last frames, so a map
/// could place none of the rest.  A batch now only adds and replaces.
///
/// A path the batch rendered again replaces its earlier record in
/// whichever half held it, so a panel is never in both halves.  A file
/// that is unreadable or declares another schema is not trusted: the
/// batch starts a fresh record and the caller is told.
fn merge_georef_manifest(existing: Option<GeorefManifest>, batch: GeorefManifest) -> GeorefManifest {
    let Some(mut merged) = existing else {
        return batch;
    };
    for (key, georeference) in batch.panels {
        merged.without_georeference.retain(|absence| absence.path != key);
        merged.panels.insert(key, georeference);
    }
    for absence in batch.without_georeference {
        merged.panels.remove(&absence.path);
        merged
            .without_georeference
            .retain(|held| held.path != absence.path);
        merged.without_georeference.push(absence);
    }
    merged
        .without_georeference
        .sort_by(|left, right| left.path.cmp(&right.path));
    merged.schema = GEOREF_MANIFEST_SCHEMA.to_string();
    merged.generated_utc = batch.generated_utc;
    merged
}

/// Exclusive hold on `<out_dir>/render-georef.json.lock` for the
/// read-merge-write, so two batches that finish together cannot each
/// read the old file and have the later write drop the earlier batch.
/// The lock file is removed when the guard drops.  A lock older than
/// `GEOREF_LOCK_STALE` is left by a process that died holding it and is
/// taken over rather than waited on forever.
struct GeorefLock(std::path::PathBuf);

const GEOREF_LOCK_STALE: std::time::Duration = std::time::Duration::from_secs(120);
const GEOREF_LOCK_WAIT: std::time::Duration = std::time::Duration::from_secs(300);

impl GeorefLock {
    fn acquire(path: std::path::PathBuf) -> Result<Self, String> {
        let started = std::time::Instant::now();
        loop {
            match std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)
            {
                Ok(_) => return Ok(Self(path)),
                Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists
                    || err.kind() == std::io::ErrorKind::PermissionDenied =>
                {
                    let stale = std::fs::metadata(&path)
                        .and_then(|meta| meta.modified())
                        .ok()
                        .and_then(|modified| modified.elapsed().ok())
                        .is_some_and(|age| age > GEOREF_LOCK_STALE);
                    if stale {
                        let _ = std::fs::remove_file(&path);
                        continue;
                    }
                    if started.elapsed() > GEOREF_LOCK_WAIT {
                        return Err(format!(
                            "another render held {} for {} s",
                            path.display(),
                            GEOREF_LOCK_WAIT.as_secs()
                        ));
                    }
                    std::thread::sleep(std::time::Duration::from_millis(25));
                }
                Err(err) => return Err(format!("lock {}: {err}", path.display())),
            }
        }
    }
}

impl Drop for GeorefLock {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// Replace `target` with `json` through a sibling temporary file, so a
/// reader never sees a half-written manifest.  A reader holding the file
/// open can refuse the rename on Windows for a moment, so it is retried.
fn replace_georef_file(target: &std::path::Path, json: &str) -> Result<(), String> {
    let temporary = target.with_extension(format!("json.tmp-{}", std::process::id()));
    std::fs::write(&temporary, json)
        .map_err(|err| format!("write {}: {err}", temporary.display()))?;
    let mut attempt = 0;
    loop {
        match std::fs::rename(&temporary, target) {
            Ok(()) => return Ok(()),
            Err(_) if attempt < 200 => {
                attempt += 1;
                std::thread::sleep(std::time::Duration::from_millis(25));
            }
            Err(err) => {
                let _ = std::fs::remove_file(&temporary);
                return Err(format!("replace {}: {err}", target.display()));
            }
        }
    }
}

/// Write `<out_dir>/render-georef.json` and print the `GEOREF` line.
/// Default-on: an opt-in flag would leave a bare run showing the defect.
/// The batch is MERGED into the manifest already there (see
/// [`merge_georef_manifest`]) under an exclusive lock.
fn write_georef_manifest(
    out_dir: &std::path::Path,
    panels: &[RenderedPanelGeoref],
) -> Result<(), String> {
    let generated_utc = chrono::Utc::now()
        .format("%Y-%m-%dT%H:%M:%SZ")
        .to_string();
    let batch = build_georef_manifest(out_dir, generated_utc, panels);
    let batch_count = batch.panels.len() + batch.without_georeference.len();
    let manifest_path = out_dir.join("render-georef.json");
    std::fs::create_dir_all(out_dir)
        .map_err(|err| format!("create {}: {err}", out_dir.display()))?;
    let _lock = GeorefLock::acquire(out_dir.join("render-georef.json.lock"))?;
    let existing = match std::fs::read(&manifest_path) {
        Ok(bytes) => match serde_json::from_slice::<GeorefManifest>(&bytes) {
            Ok(mut held) if held.schema == GEOREF_MANIFEST_SCHEMA => {
                // An earlier record whose picture has left the folder
                // would place nothing; it is dropped rather than carried.
                held.panels.retain(|key, _| out_dir.join(key).is_file());
                held.without_georeference
                    .retain(|absence| out_dir.join(&absence.path).is_file());
                Some(held)
            }
            Ok(held) => {
                eprintln!(
                    "WARNING georef manifest {} declares schema {:?}; starting a fresh record",
                    manifest_path.display(),
                    held.schema
                );
                None
            }
            Err(err) => {
                eprintln!(
                    "WARNING georef manifest {} is unreadable ({err}); starting a fresh record",
                    manifest_path.display()
                );
                None
            }
        },
        Err(_) => None,
    };
    let manifest = merge_georef_manifest(existing, batch);
    let json = serde_json::to_string_pretty(&manifest)
        .map_err(|err| format!("serialize {}: {err}", manifest_path.display()))?;
    replace_georef_file(&manifest_path, &json)?;
    // A NEW stdout line type after FINISHED.  Safe against the pinned
    // grammar: gpuwm.rustwx matches known prefixes and ignores the rest,
    // and no existing line changed.  The tallies are the whole manifest's;
    // `batch=` is what this invocation added or replaced.
    println!(
        "GEOREF {} panels={} without={} batch={}",
        manifest_path.display(),
        manifest.panels.len(),
        manifest.without_georeference.len(),
        batch_count
    );
    Ok(())
}

/// One PRODUCT row per catalog entry, with per-store availability:
/// `PRODUCT\t<slug>\t<kind>\t<status>\t<detail>`, then one CATALOG
/// summary line.  Statuses: `renderable` (proven against the imported
/// store), `missing-fields` (direct recipe whose required fields are
/// not all stored), `blocked` (windowed compute reported an accurate
/// per-product blocker), and `excluded` (recipe cannot be realized by
/// this lane, reason given).  Every decision routes through stored
/// FIELD availability; there is no model-identity gate and no `gated`
/// status -- a product that cannot render always names the fields it
/// is missing (gpuwm architectural rule; the listing test rejects any
/// identity-gated row).
///
/// WHAT is renderable comes from `inspect_renderable_products` -- the
/// same shared catalog the render pass consumes -- so this listing can
/// never disagree with what a render would attempt (it used to rebuild
/// its own product list and silently omitted whole families, which is
/// how generic `var:` rows stayed invisible).  This function only adds
/// the WHY for rows the catalog does not carry.
/// The parcel layer one heavy (ECAPE-class) recipe slug is about, worded
/// the way the shared store lane words the same three parcels
/// (`rustwx_products::derived::store`). `None` for the composites, which
/// mix parcels.
fn heavy_recipe_parcel_layer(slug: &str) -> Option<&'static str> {
    if slug.starts_with("sb") {
        Some("surface-based")
    } else if slug.starts_with("ml") {
        Some("0-90 mb mixed layer")
    } else if slug.starts_with("mu") {
        Some("0-255 mb most-unstable layer")
    } else {
        None
    }
}

/// Why ONE heavy (ECAPE-class) recipe is not renderable from this store.
///
/// WHAT BREAKAGE THIS PREVENTS (gate law): the catalog used to print one
/// literal for every excluded heavy row, so a permanent exclusion (the
/// native-CAPE ratio pairs, which need the SOURCE model's own decoded CAPE
/// plane and can never come off a wrfout) read exactly like a per-hour
/// input gap that a re-import fixes. A reader could not tell which rows
/// were worth re-running for. Every row now names the grid the catalog
/// looked for, what computes it, and the route that does compute it.
fn heavy_recipe_exclusion_reason(slug: &str, heavy_imported: bool) -> String {
    if !heavy_imported {
        return format!(
            "no '{slug}' grid: this import ran without --heavy, so the ECAPE \
             stage that computes it never ran. Re-run the import with --heavy."
        );
    }
    let layer = heavy_recipe_parcel_layer(slug).unwrap_or("the ECAPE");
    // The shared store lane already classifies these three as legitimately
    // skippable (rustwx_products::derived::store, NATIVE_RATIO_SLUGS): they
    // divide by the source model's OWN decoded CAPE plane. A wrfout carries
    // no such message, and wrf-core's sbcape/mlcape/mucape are diagnostic,
    // not native, so this is a named permanent exclusion on this lane.
    if slug.ends_with("_ecape_native_cape_ratio") {
        return format!(
            "no '{slug}' grid: it divides {layer} ECAPE by the source model's own \
             decoded CAPE plane, which a wrfout does not carry (wrf-core's CAPE is \
             diagnostic, not native), so it stays excluded on this lane. Ingest a \
             GRIB surface file that carries that CAPE message to get this pair."
        );
    }
    let way_out = "Re-import with --heavy from a wrfout that carries the 3-D fields, \
                   or ingest the same valid time through the GRIB heavy lane.";
    if slug.ends_with("_ecape_derived_cape_ratio") {
        return format!(
            "no '{slug}' grid: it needs the {layer} ECAPE grid and the {layer} \
             derived CAPE grid of one heavy solve, and this store has neither. \
             {way_out}"
        );
    }
    match slug {
        "sbecape" | "mlecape" | "muecape" | "sbncape" | "sbecin" | "mlecin" => format!(
            "no '{slug}' grid: it is a {layer} entraining-parcel grid, which the heavy \
             stage solves from this hour's isobaric volumes plus the 2 m and 10 m \
             surface planes. {way_out}"
        ),
        "ecape_scp" | "ecape_ehi_0_1km" | "ecape_ehi_0_3km" | "ecape_stp" => format!(
            "no '{slug}' grid: it composes an ECAPE grid with 0-1 km / 0-3 km SRH and \
             0-6 km bulk shear taken from the same isobaric volumes, and this store \
             carries no such grid. {way_out}"
        ),
        _ => format!(
            "no '{slug}' grid in this store, and the heavy stage reported none under \
             that slug. {way_out}"
        ),
    }
}

/// The machine code for a family that renders only when it is named.
///
/// It is a CODE, not a status: the status vocabulary stays exactly
/// {renderable, missing-fields, blocked, excluded}, because an
/// availability row must justify itself with FIELDS and never with a
/// policy word.  The opt-in state rides beside the field truth instead
/// of replacing it.
const OPT_IN_CODE: &str = "opt-in-ensemble-family";

/// The one line of prose that goes with it.
const OPT_IN_REASON: &str =
    "ensemble/probabilistic family: never included by 'all', name the slug explicitly";

fn list_products(
    store_root: &std::path::Path,
    model_slug: &str,
    run_slug: &str,
    stored_slots: &[u16],
    heavy_imported: bool,
    fire_rows:Vec<sfire::CatalogRow>,
) -> Result<(), String> {
    use rusty_weather::batch_render::BatchProductKind;
    use rusty_weather::render_all::StoreFieldSource;
    use rusty_weather::render_all::windowed_store;

    let first_slot = stored_slots
        .first()
        .copied()
        .ok_or("catalog listing needs at least one stored frame")?;
    // Every product ANY stored frame can draw: the render draws each one
    // frame by frame and skips it by name where a frame lacks its fields.
    // Decided from the first frame alone, a series whose analysis frame
    // stores no REFL_10CM listed composite reflectivity as missing, and
    // the render door dropped it from every frame of the series.
    let catalog =
        inspect_renderable_products_over(store_root, model_slug, run_slug, stored_slots)?;
    let renderable_slugs: std::collections::HashSet<&str> = catalog
        .products
        .iter()
        .map(|product| product.slug.as_str())
        .collect();
    // Diagnostics only: the catalog decides WHAT renders; the store is
    // still consulted to name the fields a non-renderable recipe misses.
    let store = StoreFieldSource::open(store_root, model_slug, run_slug, first_slot)
        .map_err(|err| err.to_string())?;

    // (slug, kind, status, detail, code).  `detail` is the prose a
    // reader sees and is never matched on; `code` is the stable machine
    // spelling a consumer decides with, so a reworded reason cannot
    // silently turn an excluded slug back into a forwarded one.
    let mut rows: Vec<(String, &str, &str, String, &str)> = fire_rows;

    for spec in rustwx_products::spec::direct_product_specs() {
        // The ensemble/probabilistic families stay out of `all` -- that
        // part of the filter is right and is untouched at
        // planning.rs:44 and :59 -- but they are LISTED, with the status
        // their stored fields justify and the opt-in state carried in
        // the code column.  A slug a user is told to name explicitly and
        // cannot discover is an unreachable capability.
        let opt_in =
            rustwx_products::direct::direct_recipe_requires_explicit_opt_in(&spec.slug);
        match rustwx_models::plot_recipe_store_requirements(&spec.slug) {
            Err(err) => rows.push((
                spec.slug,
                "direct",
                "excluded",
                format!("catalog spec has no plot recipe: {err}"),
                if opt_in { OPT_IN_CODE } else { "no-plot-recipe" },
            )),
            Ok(requirements) => {
                let missing: Vec<String> = requirements
                    .iter()
                    .filter_map(|requirement| match requirement.selector {
                        Some(selector) => store
                            .resolve(&selector)
                            .is_none()
                            .then(|| selector.key()),
                        None => Some(format!(
                            "{} (no canonical store selector exists for this field)",
                            requirement.field_key
                        )),
                    })
                    .collect();
                if renderable_slugs.contains(spec.slug.as_str()) {
                    // The title, then the canonical selector the FILL
                    // resolves to.  A title alone cannot distinguish a
                    // chart of a quantity from a chart with that
                    // quantity drawn on top of something else -- "MSLP
                    // / 10m Winds" reads like a wind product and fills
                    // with pressure.  Naming the filled field makes the
                    // listing answer "what am I looking at?" without
                    // reading the catalog source.
                    let detail = match requirements
                        .first()
                        .and_then(|requirement| requirement.selector)
                    {
                        Some(selector) => format!("{} [fill: {}]", spec.title, selector.key()),
                        None => spec.title.clone(),
                    };
                    let detail = if opt_in {
                        format!("{detail}; {OPT_IN_REASON}")
                    } else {
                        detail
                    };
                    rows.push((
                        spec.slug,
                        "direct",
                        "renderable",
                        detail,
                        if opt_in { OPT_IN_CODE } else { "renderable" },
                    ));
                } else if !missing.is_empty() {
                    let detail = format!("not stored: {}", missing.join(", "));
                    rows.push((
                        spec.slug,
                        "direct",
                        "missing-fields",
                        if opt_in {
                            format!("{detail}; {OPT_IN_REASON}")
                        } else {
                            detail
                        },
                        if opt_in { OPT_IN_CODE } else { "missing-fields" },
                    ));
                } else {
                    let detail =
                        "not offered by the shared render catalog for this store".to_string();
                    rows.push((
                        spec.slug,
                        "direct",
                        "excluded",
                        if opt_in {
                            format!("{detail}; {OPT_IN_REASON}")
                        } else {
                            detail
                        },
                        if opt_in { OPT_IN_CODE } else { "not-offered-by-catalog" },
                    ));
                }
            }
        }
    }

    for entry in rustwx_products::derived::supported_derived_recipe_inventory() {
        let kind = if entry.heavy { "heavy" } else { "derived" };
        if renderable_slugs.contains(entry.slug) {
            rows.push((
                entry.slug.to_string(),
                kind,
                "renderable",
                entry.title.to_string(),
                "renderable",
            ));
        } else if entry.heavy {
            // The candidate's per-row reason text rides in the branch's code column:
            // the code says which class of gap this is, the reason names the grid
            // the catalog looked for, what computes it and the route that does.
            let code = if heavy_imported {
                "heavy-grid-not-produced"
            } else {
                "heavy-not-imported"
            };
            rows.push((
                entry.slug.to_string(),
                kind,
                "excluded",
                heavy_recipe_exclusion_reason(entry.slug, heavy_imported),
                code,
            ));
        } else {
            rows.push((
                entry.slug.to_string(),
                kind,
                "excluded",
                "not realized by the wrfout import lane (no matching \
                 wrf-core diagnostic is stored under this recipe slug)"
                    .to_string(),
                "not-realized-by-import",
            ));
        }
    }
    for entry in rustwx_products::derived::blocked_derived_recipe_inventory() {
        rows.push((
            entry.slug.to_string(),
            "derived",
            "excluded",
            entry.reason.to_string(),
            "recipe-blocked",
        ));
    }

    // Generic rows come straight off the shared catalog: every stored 2-D
    // variable that no named product already renders, as `var:<name>`.
    // The catalog's dedup exclusions were logged on stderr when it built.
    for product in &catalog.products {
        if product.kind != BatchProductKind::Generic {
            continue;
        }
        let field = product
            .source_fields
            .first()
            .map(String::as_str)
            .unwrap_or("");
        rows.push((
            product.slug.clone(),
            "generic",
            "renderable",
            format!(
                "stored 2-D variable '{field}' [{}]",
                product.units.as_deref().unwrap_or("unknown units")
            ),
            "renderable",
        ));
    }

    // Asked of the store's axis directly: the catalog now lists only the
    // windows the run can close, so "no windowed row in the catalog" no
    // longer means "the axis cannot serve windows".
    let windowed_ready = windowed_store::windowed_axis_ready(store_root, model_slug, run_slug)
        .map_err(|err| err.to_string())?;
    let windowed_slugs: Vec<String> =
        rustwx_products::windowed::HrrrWindowedProduct::supported_products()
            .iter()
            .map(|product| product.slug().to_string())
            .collect();
    if !windowed_ready {
        // The window-axis exclusion is the row a consumer HAS to act on --
        // it must skip those slugs rather than forward them -- so it is
        // the code with the longest reach.  The prose beside it is
        // unchanged, byte for byte, because it is what the render door
        // prints and what its tests read.  (An exact-time store is no
        // longer excluded: its windows are served from its frames' leads,
        // so `windowed-ordinal-axis` is not emitted any more.)
        let (reason, code) = (
            "windowed accumulations need more than one stored whole-hour frame",
            "windowed-needs-whole-hour-frames",
        );
        for slug in &windowed_slugs {
            rows.push((slug.clone(), "windowed", "excluded", reason.to_string(), code));
        }
    } else {
        // The verdict is the run's last window: at its last whole-hour
        // frame, which on an exact-time store may come before its last
        // frame.  Asked of a frame between hours, every window would read
        // "closes at whole forecast hours", which is true of that frame
        // and says nothing about the run.
        let frames = windowed_store::stored_window_frames(store_root, model_slug, run_slug)
            .map_err(|err| err.to_string())?;
        let through = frames
            .iter()
            .rev()
            .find(|frame| frame.closes_windows())
            .map(|frame| frame.slot);
        let verdict_slots: Vec<u16> = match through {
            Some(last) => stored_slots
                .iter()
                .copied()
                .filter(|slot| *slot <= last)
                .collect(),
            None => stored_slots.to_vec(),
        };
        match windowed_store::compute_windowed_products(
            store_root,
            model_slug,
            run_slug,
            &verdict_slots,
            &windowed_slugs,
        ) {
            Ok(outcome) => {
                let blocked: std::collections::HashMap<String, String> =
                    outcome.blockers.into_iter().collect();
                for grid in &outcome.grids {
                    rows.push((
                        grid.slug.clone(),
                        "windowed",
                        "renderable",
                        grid.strategy.clone(),
                        "renderable",
                    ));
                }
                for slug in &windowed_slugs {
                    if let Some(reason) = blocked.get(slug) {
                        rows.push((
                            slug.clone(),
                            "windowed",
                            "blocked",
                            reason.clone(),
                            "windowed-blocked",
                        ));
                    }
                }
            }
            Err(err) => {
                let reason = format!("windowed compute unavailable: {err}");
                for slug in &windowed_slugs {
                    rows.push((
                        slug.clone(),
                        "windowed",
                        "excluded",
                        reason.clone(),
                        "windowed-compute-unavailable",
                    ));
                }
            }
        }
    }

    let mut counts = std::collections::BTreeMap::<&str, usize>::new();
    for (slug, kind, status, detail, code) in &rows {
        *counts.entry(status).or_default() += 1;
        println!("PRODUCT\t{slug}\t{kind}\t{status}\t{detail}\t{code}");
    }
    let summary = counts
        .iter()
        .map(|(status, count)| format!("{status}={count}"))
        .collect::<Vec<_>>()
        .join(" ");
    println!("CATALOG total={} {summary}", rows.len());
    Ok(())
}

#[cfg(test)]
mod tests {

    #[test]
    fn a_group_keyword_is_counted_as_the_products_it_expands_to() {
        // The breakage: `--products windowed` sized the per-hour ceiling
        // as one product and refused the 49 it expands to.
        let windowed = requested_product_count("windowed");
        assert!(windowed > 32, "windowed expands to {windowed}");
        assert_eq!(requested_product_count("qpf_1h,qpf_total"), 2);
        assert_eq!(requested_product_count(" , "), 0);
    }
    use super::*;

    /// WHAT BREAKAGE THIS PREVENTS (gate law): `--list-products` printed ONE
    /// reason literal for every excluded heavy row, so a permanent exclusion
    /// (the three native-CAPE ratio pairs, which divide by the source model's
    /// own decoded CAPE plane and can never come off a wrfout) read exactly
    /// like a per-hour input gap a re-import fixes. Seventeen rows asserted
    /// one cause; the family has sixteen members and at least three causes.
    #[test]
    fn every_excluded_heavy_row_names_its_own_missing_grid_and_a_way_out() {
        let heavy: Vec<&'static str> =
            rustwx_products::derived::supported_derived_recipe_inventory()
                .iter()
                .filter(|entry| entry.heavy)
                .map(|entry| entry.slug)
                .collect();
        assert!(
            heavy.len() >= 13,
            "the heavy family shrank to {}: re-check the reasons",
            heavy.len()
        );
        for heavy_imported in [true, false] {
            let reasons: Vec<String> = heavy
                .iter()
                .map(|slug| heavy_recipe_exclusion_reason(slug, heavy_imported))
                .collect();
            for (slug, reason) in heavy.iter().zip(reasons.iter()) {
                assert!(
                    reason.contains(slug),
                    "the row must name the grid it is missing: {slug} -> {reason}"
                );
                assert!(
                    reason.contains("--heavy") || reason.contains("GRIB"),
                    "the row must name a way out: {slug} -> {reason}"
                );
                assert!(
                    !reason.contains('\t') && !reason.contains('\n'),
                    "a PRODUCT row is tab separated: {slug} -> {reason}"
                );
            }
            // Distinct as WRITTEN, and still distinct once the slug is
            // removed -- a single blanket literal with the slug pasted in
            // would pass the first check and fail this one.
            let written: std::collections::BTreeSet<&str> =
                reasons.iter().map(String::as_str).collect();
            assert!(
                written.len() >= 3,
                "heavy_imported={heavy_imported}: only {} distinct reason(s)",
                written.len()
            );
            let shapes: std::collections::BTreeSet<String> = heavy
                .iter()
                .zip(reasons.iter())
                .map(|(slug, reason)| reason.replace(slug, "<slug>"))
                .collect();
            let expected_shapes = if heavy_imported { 3 } else { 1 };
            assert!(
                shapes.len() >= expected_shapes,
                "heavy_imported={heavy_imported}: {} distinct reason shape(s), want {expected_shapes}: {shapes:?}",
                shapes.len()
            );
        }

        // The permanent exclusion must not promise a re-import fixes it, and
        // the per-hour gaps must not read as permanent.
        for slug in [
            "sb_ecape_native_cape_ratio",
            "ml_ecape_native_cape_ratio",
            "mu_ecape_native_cape_ratio",
        ] {
            let reason = heavy_recipe_exclusion_reason(slug, true);
            assert!(
                reason.contains("GRIB"),
                "the permanent exclusion must name the route that does compute it: {reason}"
            );
            assert!(
                !reason.contains("Re-import"),
                "a permanent exclusion must not promise a re-import: {reason}"
            );
        }
        for slug in ["mlecape", "muecape", "mlecin"] {
            let reason = heavy_recipe_exclusion_reason(slug, true);
            assert!(
                reason.contains("--heavy"),
                "a per-hour gap must name the import that fills it: {reason}"
            );
        }
    }

    /// WHAT BREAKAGE THIS PREVENTS (gate law): the pair tool
    /// handed a section the MAP's size, so a 1800x1464 near-square map
    /// produced a 1800x1464 near-square section -- a cut 100 km wide and
    /// 6 km tall drawn in a portrait-ish frame.
    #[test]
    fn a_section_is_landscape_by_default_whatever_size_the_map_is() {
        for (map_w, map_h) in [(1800u32, 1464u32), (1200, 1600), (2400, 1200), (960, 720)] {
            let (w, h) = section_dimensions(None, map_w, map_h);
            assert_eq!(w, h * 2, "{map_w}x{map_h} gave {w}x{h}");
            assert!(w >= map_w, "{map_w}x{map_h} lost width");
        }
        // The two shapes the rejected sheets came out at.
        assert_eq!(section_dimensions(None, 1800, 1464), (1800, 900));
        assert_eq!(section_dimensions(None, 2400, 1200), (2400, 1200));
        // A caller that names a size gets exactly it.
        assert_eq!(section_dimensions(Some((1600, 1200)), 2400, 1200), (1600, 1200));
    }

    /// An `Args` that is valid apart from what a test deliberately breaks.
    fn args_for(products: &str, inputs: Vec<PathBuf>) -> Args {
        Args {
            store_root: PathBuf::from("store"),
            out_dir: PathBuf::from("out"),
            products: products.to_string(),
            frames: None,
            width: 1_200,
            height: 900,
            layout: rustwx_render::LayoutMode::Fixed,
            heavy: false,
            list_products: false,
            streamlines: None,
            source_label: DEFAULT_SOURCE_LABEL.to_string(),
            overlays: None,
            annotations: None,
            theme: None,
            radar_colors: rustwx_render::RadarColorSet::Standard,
            section: None,
            section_across_km: None,
            isotherms: section::Isotherms::default(),
            section_top_km: 14.0,
            section_size: None,
            section_reference_km: None,
            mesh_grid: None,
            mesh_reference: None,
            mesh_labels: ("TREATMENT".to_string(), "CONTROL".to_string()),
            mesh_bounds: None,
            footer: rustwx_render::FooterFields::default(),
            difference: None,
            save_import: None,
            from_import: None,
            inputs,
        }
    }

    fn refusal(result: Result<(), CliError>) -> (u8, String) {
        match result {
            Ok(()) => panic!("expected a refusal"),
            Err(CliError::Help) => panic!("expected a refusal, got --help"),
            Err(CliError::Usage(message)) => (EXIT_USAGE, message),
            Err(CliError::Failed(message)) => (1, message),
        }
    }

    #[test]
    fn the_usage_line_keeps_the_token_the_python_probe_requires() {
        // gpuwm.rustwx.probe_renderer declares the whole rust engine
        // unusable -- silently dropping `--engine auto` to matplotlib --
        // unless `--help` exits 0 with this literal in its transcript.
        assert!(usage().starts_with("usage: rw_wrfbatch"));
    }

    #[test]
    fn an_unknown_product_names_the_token_and_the_choices() {
        let (code, message) = refusal(validate_request(&args_for(
            "definitely_not_a_product",
            vec![PathBuf::from("wrfout_d02_x.nc")],
        )));
        assert_eq!(code, EXIT_USAGE, "a bad command line is a usage failure");
        assert!(
            message.contains("definitely_not_a_product"),
            "the refusal must name the token: {message}"
        );
        for choice in ["'all'", "'direct'", "'derived'", "'heavy'", "'windowed'"] {
            assert!(
                message.contains(choice),
                "the refusal must name the choices, missing {choice}: {message}"
            );
        }
        assert!(
            message.contains("--list-products"),
            "the refusal must route to the full vocabulary: {message}"
        );
    }

    #[test]
    fn a_product_typo_is_caught_before_any_input_is_opened() {
        // The regression this pins: --products used to be validated only
        // after a full wrfout import, so a typo reported "No supported WRF
        // files selected" -- about a file, not about the typo.
        let (_code, message) = refusal(validate_request(&args_for(
            "not_a_product",
            vec![PathBuf::from("no/such/path/wrfout_d01_x.nc")],
        )));
        assert!(
            message.contains("not_a_product"),
            "the product refusal must win over the missing input: {message}"
        );
    }

    #[test]
    fn a_missing_input_is_named_in_the_matplotlib_wording() {
        let (code, message) = refusal(validate_request(&args_for(
            "all",
            vec![PathBuf::from("no/such/path/wrfout_d01_x.nc")],
        )));
        assert_eq!(code, 1, "a real run that cannot proceed is not a usage error");
        assert!(message.contains("wrfout_d01_x.nc"), "{message}");
        assert!(message.contains("unreadable wrfout"), "{message}");
        assert!(message.contains("no such file"), "{message}");
    }

    #[test]
    fn an_input_that_is_not_a_wrf_file_is_named() {
        let dir = std::env::temp_dir().join("rw_wrfbatch_not_a_wrf_file");
        std::fs::create_dir_all(&dir).expect("scratch dir");
        let path = dir.join("plain.txt");
        std::fs::write(&path, b"not a wrfout").expect("scratch file");

        let (code, message) = refusal(validate_request(&args_for("all", vec![path.clone()])));
        assert_eq!(code, 1);
        assert!(
            message.contains("plain.txt"),
            "'No supported WRF files selected' never named the file: {message}"
        );
        assert!(message.contains("unreadable wrfout"), "{message}");

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn one_usable_input_among_several_still_runs() {
        // Existing semantics preserved: a mixed set is not a refusal.  Only
        // a set with nothing usable in it is.
        let dir = std::env::temp_dir().join("rw_wrfbatch_mixed_inputs");
        std::fs::create_dir_all(&dir).expect("scratch dir");
        let good = dir.join("wrfout_d02_1974-04-03_18:00:00".replace(':', "-"));
        std::fs::write(&good, b"placeholder").expect("scratch file");
        let bad = dir.join("plain.txt");
        std::fs::write(&bad, b"not a wrfout").expect("scratch file");

        if wrf_process::is_supported_wrf_file(&good) {
            assert!(
                validate_request(&args_for("all", vec![good.clone(), bad.clone()])).is_ok(),
                "a set with one usable input must not be refused"
            );
        }

        let _ = std::fs::remove_file(&good);
        let _ = std::fs::remove_file(&bad);
    }

    /// A Lambert georeference serialized into the manifest and read back
    /// must place a point on the SAME pixel.  This is the wire-format half
    /// of the fix: a transform that survives rendering but not JSON would
    /// still leave the consumer registering coastlines.
    #[test]
    fn georef_manifest_round_trips_a_projected_point_unchanged() {
        let projection = rustwx_render::ResolvedProjection::LambertConformal {
            standard_parallel_1_deg: 33.0,
            standard_parallel_2_deg: 45.0,
            central_meridian_deg: -96.0,
            reference_latitude_deg: 39.0,
        };
        let (x, y) = projection.project(35.0, -97.5);
        let georeference = rustwx_render::PanelGeoReference::new(
            1_600,
            900,
            rustwx_render::PlotRect {
                x: 22,
                y: 51,
                width: 1_430,
                height: 810,
            },
            projection,
            rustwx_render::ProjectedExtent {
                x_min: x - 500_000.0,
                x_max: x + 500_000.0,
                y_min: y - 300_000.0,
                y_max: y + 300_000.0,
            },
            (-103.0, -92.0, 31.0, 39.0),
        );
        let before = georeference
            .lonlat_to_pixel(35.0, -97.5)
            .expect("the point is inside the frame");

        let out_dir = PathBuf::from("C:\\proof\\out");
        let manifest = build_georef_manifest(
            &out_dir,
            "2026-08-26T00:00:00Z".to_string(),
            &[(
                out_dir.join("d02-3km").join("refl").join("frame.png"),
                Some(georeference),
                None,
            )],
        );
        let json = serde_json::to_string_pretty(&manifest).expect("manifest serializes");
        let parsed: GeorefManifest = serde_json::from_str(&json).expect("manifest parses back");

        assert_eq!(parsed.schema, "rustwx.render-georef/v1");
        assert!(parsed.without_georeference.is_empty());
        // Path relative to out_dir, forward slashes: the key contract.
        let restored = parsed
            .panels
            .get("d02-3km/refl/frame.png")
            .expect("the panel is keyed by its out_dir-relative forward-slash path");
        let after = restored
            .lonlat_to_pixel(35.0, -97.5)
            .expect("the restored transform still places the point");
        assert!(
            (before.0 - after.0).abs() < 1.0e-9 && (before.1 - after.1).abs() < 1.0e-9,
            "JSON round trip moved the point: {before:?} -> {after:?}"
        );
    }

    /// A panel with no transform lands in `without_georeference` with a
    /// non-empty reason -- the lane's own when it gave one, a fallback
    /// naming the gap when it did not.  A silent omission is exactly the
    /// defect the manifest exists to fix.
    #[test]
    fn a_panel_without_a_georeference_is_listed_with_its_reason() {
        let out_dir = PathBuf::from("C:\\proof\\out");
        let manifest = build_georef_manifest(
            &out_dir,
            "2026-08-26T00:00:00Z".to_string(),
            &[
                (
                    out_dir.join("cloud_levels.png"),
                    None,
                    Some("composite panel: one PNG composes multiple member maps".to_string()),
                ),
                (out_dir.join("unthreaded.png"), None, None),
            ],
        );
        assert!(manifest.panels.is_empty());
        assert_eq!(manifest.without_georeference.len(), 2);
        let by_path: std::collections::HashMap<&str, &str> = manifest
            .without_georeference
            .iter()
            .map(|absence| (absence.path.as_str(), absence.reason.as_str()))
            .collect();
        assert_eq!(
            by_path["cloud_levels.png"],
            "composite panel: one PNG composes multiple member maps",
            "the lane's own reason must survive verbatim"
        );
        let fallback = by_path["unthreaded.png"];
        assert!(
            !fallback.is_empty() && fallback.contains("threaded neither"),
            "a lane that said nothing still gets a reason naming the gap: {fallback}"
        );
    }

    /// A later batch adds to the record instead of replacing it, and a
    /// panel rendered again moves between the halves rather than
    /// appearing in both.
    #[test]
    fn a_later_batch_merges_into_the_manifest_instead_of_replacing_it() {
        let out_dir = PathBuf::from("C:\\proof\\out");
        let first = build_georef_manifest(
            &out_dir,
            "2026-08-26T00:00:00Z".to_string(),
            &[
                (out_dir.join("d01_f001.png"), None, Some("first".to_string())),
                (out_dir.join("d01_f002.png"), None, Some("first".to_string())),
            ],
        );
        let second = build_georef_manifest(
            &out_dir,
            "2026-08-26T01:00:00Z".to_string(),
            &[
                (out_dir.join("d02_f001.png"), None, Some("second".to_string())),
                (out_dir.join("d01_f002.png"), None, Some("again".to_string())),
            ],
        );
        let merged = merge_georef_manifest(Some(first), second);
        let paths: Vec<&str> = merged
            .without_georeference
            .iter()
            .map(|absence| absence.path.as_str())
            .collect();
        assert_eq!(paths, ["d01_f001.png", "d01_f002.png", "d02_f001.png"]);
        assert_eq!(merged.without_georeference[1].reason, "again");
        assert_eq!(merged.generated_utc, "2026-08-26T01:00:00Z");
        assert!(merge_georef_manifest(None, build_georef_manifest(&out_dir, String::new(), &[]))
            .without_georeference
            .is_empty());
    }

    #[test]
    fn import_failures_surface_as_import_note_lines_not_a_count() {
        // The regression this pins: the transcript said `notes=1` and
        // nothing else, so a real import failure (wrf_process pushes notes
        // like "PSFC skipped: ...", "uvmet unavailable: ...") was invisible
        // unless the caller went digging.  Every note must become one
        // verbatim, tab-separated IMPORT_NOTE record.
        let notes = [
            "PSFC skipped: missing variable".to_string(),
            "uvmet unavailable: rotation constants absent".to_string(),
        ];
        let lines: Vec<String> = notes.iter().map(|note| import_note_line(note)).collect();
        assert_eq!(lines.len(), notes.len(), "one record per note, no tally");
        for (line, note) in lines.iter().zip(&notes) {
            assert_eq!(line, &format!("IMPORT_NOTE\t{note}"));
            let (tag, body) = line.split_once('\t').expect("tab-separated record");
            assert_eq!(tag, "IMPORT_NOTE");
            assert_eq!(body, note, "the note text must survive verbatim");
        }
    }

    #[test]
    fn an_opt_in_family_is_selectable_by_name_and_still_outside_all() {
        // The families a user is TOLD to name explicitly must be
        // discoverable, or the instruction has no vocabulary behind it.
        // They still stay out of the `all` expansion, which is the part
        // of the old behaviour that was right.
        let opt_in: Vec<String> = rustwx_products::spec::direct_product_specs()
            .into_iter()
            .map(|spec| spec.slug)
            .filter(|slug| {
                rustwx_products::direct::direct_recipe_requires_explicit_opt_in(slug)
            })
            .collect();
        assert!(
            !opt_in.is_empty(),
            "no opt-in family exists to test the listing against"
        );
        let selectable = rusty_weather::render_all::known_product_slugs();
        let expanded_by_all = rustwx_products::direct::store_direct_recipe_slugs();
        for slug in &opt_in {
            assert!(
                selectable.contains(slug),
                "{slug} cannot be discovered from the product vocabulary"
            );
            assert!(
                !expanded_by_all.contains(slug),
                "{slug} joined the 'all' expansion, which it must never do"
            );
        }
    }

    #[test]
    fn the_abi_marker_pins_the_code_column_and_the_requirement_rows() {
        // A build that answers the old grammar must fail the handshake
        // rather than silently emit five fields where six are read.
        assert!(ABI_MARKER.contains("\tdetail\tcode\t"), "{ABI_MARKER}");
        assert!(ABI_MARKER.contains("requirements-v1\tNEEDS\t"), "{ABI_MARKER}");
        assert!(ABI_MARKER.contains("\tPLANNED\tstore_field\t"), "{ABI_MARKER}");
    }

    #[test]
    fn resolution_tokens_are_trimmed_km_or_integer_metres() {
        assert_eq!(resolution_token(3_000.0).as_deref(), Some("3km"));
        assert_eq!(resolution_token(12_000.0).as_deref(), Some("12km"));
        assert_eq!(resolution_token(1_000.0).as_deref(), Some("1km"));
        // A 3:1 nest of a 12 km parent, twice over.
        assert_eq!(resolution_token(1_333.3333).as_deref(), Some("1.333km"));
        assert_eq!(resolution_token(1_500.0).as_deref(), Some("1.5km"));
        // Sub-kilometre reads as metres, not as 0.111 km.
        assert_eq!(resolution_token(111.1111).as_deref(), Some("111m"));
        assert_eq!(resolution_token(500.0).as_deref(), Some("500m"));
        assert_eq!(resolution_token(250.0).as_deref(), Some("250m"));
        // The boundary rounds before it chooses a unit.
        assert_eq!(resolution_token(999.6).as_deref(), Some("1km"));
        assert_eq!(resolution_token(999.4).as_deref(), Some("999m"));
        // Nothing usable is never guessed at.
        assert_eq!(resolution_token(0.0), None);
        assert_eq!(resolution_token(-3_000.0), None);
        assert_eq!(resolution_token(f64::NAN), None);
    }

    #[test]
    fn spacing_subtitles_spell_the_same_number() {
        assert_eq!(spacing_subtitle(3_000.0).as_deref(), Some("\u{0394}x 3 km"));
        assert_eq!(
            spacing_subtitle(1_333.3333).as_deref(),
            Some("\u{0394}x 1.333 km")
        );
        assert_eq!(spacing_subtitle(111.1111).as_deref(), Some("\u{0394}x 111 m"));
        assert_eq!(spacing_subtitle(f64::INFINITY), None);
    }

    #[test]
    fn a_resampled_mesh_names_its_dx_the_render_grid() {
        let wrf = GridIdentity { domain: Some("d01".into()), spacing_m: Some(3_000.0), label: None, name: None };
        assert_eq!(subtitle_spacing(&wrf).as_deref(), Some("\u{0394}x 3 km"));
        let hex = GridIdentity {
            domain: Some("d01".into()),
            spacing_m: Some(22_000.0),
            label: checked_grid_label("p655-8c 30 km"),
            name: None,
        };
        assert_eq!(subtitle_spacing(&hex).as_deref(), Some("render grid 22 km"));
        let none = GridIdentity { domain: None, spacing_m: None, label: checked_grid_label("hex 15 km"), name: None };
        assert_eq!(subtitle_spacing(&none), None);
    }

    #[test]
    fn domain_tokens_come_from_grid_id_then_the_filename() {
        assert_eq!(domain_token_from_id(2).as_deref(), Some("d02"));
        assert_eq!(domain_token_from_id(11).as_deref(), Some("d11"));
        assert_eq!(domain_token_from_id(0), None);
        assert_eq!(domain_token_from_id(-1), None);

        let named = PathBuf::from("/runs/wrfout_d03_1974-04-03_18:00:00");
        assert_eq!(domain_token_from_filename(&named).as_deref(), Some("d03"));
        let anonymous = PathBuf::from("/runs/model_output.nc");
        assert_eq!(domain_token_from_filename(&anonymous), None);
        let truncated = PathBuf::from("/runs/wrfout_d");
        assert_eq!(domain_token_from_filename(&truncated), None);
    }

    #[test]
    fn the_slug_pairs_domain_with_resolution_and_degrades_accurately() {
        let full = GridIdentity {
            domain: Some("d02".to_string()),
            spacing_m: Some(3_000.0),
            label: None,
            name: None,
        };
        assert_eq!(native_domain_slug(&full).as_deref(), Some("d02-3km"));

        // DX absent: the domain still separates the nests, which is the
        // whole point of the token.
        let no_spacing = GridIdentity {
            domain: Some("d02".to_string()),
            spacing_m: None,
            label: None,
            name: None,
        };
        assert_eq!(native_domain_slug(&no_spacing).as_deref(), Some("d02"));

        // Domain absent: no token at all -- `native_grid` is accurate about
        // an unidentified grid in a way that `d01` would not be.
        let no_domain = GridIdentity {
            domain: None,
            spacing_m: Some(3_000.0),
            label: None,
            name: None,
        };
        assert_eq!(native_domain_slug(&no_domain), None);
    }

    #[test]
    fn a_producer_grid_label_names_the_headline_and_the_slug() {
        // A resampled hex mesh frame: WRF's GRID_ID and the resample DX are
        // present, and the producer's label outranks both.
        let hex = GridIdentity {
            domain: Some("d01".to_string()),
            spacing_m: Some(14_900.0),
            label: checked_grid_label("hex  15 km"),
            name: None,
        };
        assert_eq!(domain_title_label(&hex).as_deref(), Some("hex 15 km"));
        assert_eq!(native_domain_slug(&hex).as_deref(), Some("hex15km"));
        let graded = GridIdentity {
            label: checked_grid_label("hex 60-3 km"),
            ..GridIdentity::default()
        };
        assert_eq!(native_domain_slug(&graded).as_deref(), Some("hex60-3km"));
        assert_eq!(checked_grid_label("a|b"), None);
        assert_eq!(checked_grid_label("   "), None);
        assert_eq!(checked_grid_label(&"x".repeat(33)), None);
        // No label: the WRF identity, exactly as before.
        let wrf = GridIdentity {
            domain: Some("d02".to_string()),
            spacing_m: Some(3_000.0),
            label: None,
            name: None,
        };
        assert_eq!(domain_title_label(&wrf).as_deref(), Some("d02 3 km"));
        assert_eq!(native_domain_slug(&wrf).as_deref(), Some("d02-3km"));
    }

    /// A Hex with Regional nest is its own one-domain run (`GRID_ID = 1`):
    /// `GPUWM_GRID_NAME` renames it in the headline only.  The domain folder
    /// stays `d01-3km` (the WOOF worker matches a nest's column stores to
    /// it) and the subtitle keeps the model's own `Δx`.
    #[test]
    fn a_grid_name_renames_the_headline_only() {
        assert_eq!(GRID_NAME_ATTRIBUTE, "GPUWM_GRID_NAME");
        let nest = GridIdentity {
            domain: Some("d01".to_string()),
            spacing_m: Some(3_000.0),
            label: None,
            name: checked_grid_label("nest"),
        };
        assert_eq!(domain_title_label(&nest).as_deref(), Some("nest 3 km"));
        assert_eq!(native_domain_slug(&nest).as_deref(), Some("d01-3km"));
        assert_eq!(subtitle_spacing(&nest).as_deref(), Some("\u{0394}x 3 km"));
        // A producer's grid label still outranks it.
        let labelled = GridIdentity { label: checked_grid_label("hex 15 km"), ..nest };
        assert_eq!(domain_title_label(&labelled).as_deref(), Some("hex 15 km"));
    }

    /// The REFC note goes on the REFC recipes' titles only, beside whatever
    /// the caller's `--annotate` file says, and is absent when no frame
    /// states it.
    #[test]
    fn the_refc_note_reaches_the_reflectivity_recipes() {
        assert_eq!(REFC_NOTE_ATTRIBUTE, "GPUWM_REFC_NOTE");
        assert_eq!(with_refc_note(None, None), None);
        let annotated = with_refc_note(None, Some("incl. parameterised convection".into())).unwrap();
        assert_eq!(annotated.title_notes.get("composite_reflectivity").map(String::as_str),
                   Some("incl. parameterised convection"));
        assert_eq!(annotated.title_notes.get("composite_reflectivity_uh").map(String::as_str),
                   Some("incl. parameterised convection"));
        assert!(!annotated.title_notes.contains_key("1km_reflectivity"));
        let caller: rustwx_products::geographic_overlays::PanelAnnotations =
            serde_json::from_str(r#"{"subtitle_center":"VERIFIED"}"#).unwrap();
        let merged = with_refc_note(Some(caller), Some("incl. parameterised convection".into())).unwrap();
        assert_eq!(merged.subtitle_center.as_deref(), Some("VERIFIED"));
        assert_eq!(merged.title_notes.len(), 2);
    }

    #[test]
    fn disagreeing_inputs_yield_no_token() {
        // Two nests imported into ONE store render as one run, so no
        // single token can be true of the output.  Filenames alone are
        // enough to reach the disagreement branch.
        let mixed = [
            PathBuf::from("wrfout_d02_1974-04-03_18:00:00"),
            PathBuf::from("wrfout_d03_1974-04-03_18:00:00"),
        ];
        assert_eq!(grid_identity(&mixed), GridIdentity::default());
        assert_eq!(native_domain_slug(&grid_identity(&mixed)), None);

        let agreeing = [
            PathBuf::from("wrfout_d02_1974-04-03_18:00:00"),
            PathBuf::from("wrfout_d02_1974-04-03_19:00:00"),
        ];
        assert_eq!(
            grid_identity(&agreeing).domain.as_deref(),
            Some("d02"),
            "one domain across several files keeps its token"
        );
    }
}

/// The command line is wrong.  Matches `gpuwm render`'s matplotlib engine,
/// so the two engines agree on what a bad `--products` costs.
const EXIT_USAGE: u8 = 2;

fn dispatch() -> Result<(), CliError> {
    match parse_args()? {
        Invocation::Abi => {
            println!("{ABI_MARKER}");
            Ok(())
        }
        Invocation::Catalog => print_product_catalog(),
        Invocation::Batch(args) => {
            validate_request(&args)?;
            run(*args).map_err(CliError::Failed)
        }
    }
}

fn render_main() -> ExitCode {
    if let Some(result) = rw_wrfbatch::hubwind_products::cli(&std::env::args().skip(1).collect::<Vec<_>>()) {
        return match result { Ok(()) => ExitCode::SUCCESS, Err(message) => { eprintln!("FAILED\t{message}"); ExitCode::FAILURE } };
    }
    if let Some(result) = rw_wrfbatch::solar_products::cli(&std::env::args().skip(1).collect::<Vec<_>>()) {
        return match result { Ok(()) => ExitCode::SUCCESS, Err(message) => { eprintln!("FAILED\t{message}"); ExitCode::FAILURE } };
    }
    if let Some(result) = rw_wrfbatch::nextgen_products::cli(&std::env::args().skip(1).collect::<Vec<_>>()) {
        return match result { Ok(()) => ExitCode::SUCCESS, Err(message) => { eprintln!("FAILED\t{message}"); ExitCode::FAILURE } };
    }
    if let Some(result) = rw_wrfbatch::ocean_products::cli(&std::env::args().skip(1).collect::<Vec<_>>()) {
        return match result { Ok(()) => ExitCode::SUCCESS, Err(message) => { eprintln!("FAILED\t{message}"); ExitCode::FAILURE } };
    }
    let _ = std::hint::black_box(GPUWM_BRIDGE_SOURCE_REV_STAMP);
    if std::env::args().nth(1).as_deref() == Some("--worker-abi") {
        println!("{}", worker::ABI_MARKER);
        return ExitCode::SUCCESS;
    }
    if std::env::args().nth(1).as_deref() == Some("--worker") {
        return worker::main();
    }
    if let Some(result) = rw_wrfbatch::cyclone_products::try_cli(&std::env::args().skip(1).collect::<Vec<_>>()) {
        return match result { Ok(()) => ExitCode::SUCCESS, Err(message) => { eprintln!("FAILED\t{message}"); ExitCode::FAILURE } };
    }
    if let Some(result) = rw_wrfbatch::ensemble_products::try_cli(&std::env::args().skip(1).collect::<Vec<_>>()) {
        return match result { Ok(()) => ExitCode::SUCCESS, Err(message) => { eprintln!("FAILED\t{message}"); ExitCode::FAILURE } };
    }
    if let Some(result) = sheet::try_cli(&std::env::args().skip(1).collect::<Vec<_>>()) {
        return match result { Ok(()) => ExitCode::SUCCESS, Err(message) => { eprintln!("{message}"); ExitCode::FAILURE } };
    }
    if let Some(result) = store_render::try_cli(&std::env::args().skip(1).collect::<Vec<_>>()) {
        return match result { Ok(()) => ExitCode::SUCCESS, Err(message) => { eprintln!("{message}"); ExitCode::FAILURE } };
    }
    if let Some(result) = rw_wrfbatch::process_request::try_cli(&std::env::args().skip(1).collect::<Vec<_>>()) {
        return match result { Ok(()) => ExitCode::SUCCESS, Err(message) => { eprintln!("{message}"); ExitCode::FAILURE } };
    }
    match dispatch() {
        Ok(()) => ExitCode::SUCCESS,
        Err(CliError::Help) => {
            // stderr and exit 0, exactly as before: gpuwm.rustwx.probe_renderer
            // reads stdout+stderr together and requires both.
            eprintln!("{}", usage());
            ExitCode::SUCCESS
        }
        Err(CliError::Usage(message)) => {
            // Usage line FIRST, then the sentence that names the problem --
            // argparse's order, which is also the order matplotlib's engine
            // prints in.  It used to be the other way round, and
            // gpuwm/rustwx.py surfaces the LAST non-empty stderr line as the
            // cause, so every argument mistake reached the user as the usage
            // line with the actionable sentence scrolled off above it.
            eprintln!("{}", usage());
            eprintln!("{message}");
            ExitCode::from(EXIT_USAGE)
        }
        Err(CliError::Failed(message)) => {
            // No usage line.  gpuwm/rustwx.py surfaces the LAST non-empty
            // stderr line as the reason, so appending usage() here is what
            // made `gpuwm render` report a usage string as the cause of a
            // store failure.
            eprintln!("{message}");
            ExitCode::FAILURE
        }
    }
}

mod cli_contracts;
#[path = "bin/ensbatch.rs"] mod ensbatch;
#[path = "bin/compare.rs"] mod compare_cli;
#[path = "bin/verify.rs"] mod verify_cli;
#[path = "bin/scorecard.rs"] mod scorecard_cli;
#[path = "bin/nowcast_frames.rs"] mod nowcast_frames;

/// Stack of each worker in this binary's global rayon pool (std's default is
/// 2 MiB).  THE BREAKAGE THIS PREVENTS (hex graded-plateau 1 km region, box J,
/// 2026-10-24): `rw_wrfbatch --list-products` on a 973x971 frame aborted with
/// "fatal runtime error: stack overflow" in 3 of 5 door runs.  The import
/// computes its twelve product groups in one `rayon::scope`, and their
/// per-column `par_iter`s (wrf-core's CAPE/CIN among them) wait in `join`;
/// a waiting worker steals and runs other halves on top of its own stack, so
/// on a busy 384-thread host the steal chain inside `compute_cape_cin` grew
/// past 2 MiB (gdb: SIGSEGV in a pool worker, every frame
/// `join_context`/`bridge_producer_consumer::helper`/`wait_until_cold`).
/// The steal depth is a scheduling outcome, not a property of the data, so
/// the pool gets room for it; the computed values do not depend on it.
/// Pages are touched only as deep as a chain reaches.
const RAYON_WORKER_STACK_BYTES: usize = 32 << 20;

/// Build the global rayon pool with [`RAYON_WORKER_STACK_BYTES`] (or a larger
/// `RUST_MIN_STACK`) before anything uses it.  The thread count stays
/// rayon's own default (`RAYON_NUM_THREADS`, else every core).
fn install_rayon_worker_stack() {
    let requested = std::env::var("RUST_MIN_STACK")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(0);
    // An error here only means a pool already exists, which cannot happen
    // before the first line of main; the default pool then stands.
    let _ = rayon::ThreadPoolBuilder::new()
        .stack_size(RAYON_WORKER_STACK_BYTES.max(requested))
        .build_global();
}

fn main() -> ExitCode {
    install_rayon_worker_stack();
    if std::env::args().any(|arg| arg == "--cf") {
        return rw_wrfbatch::cf_render::main();
    }
    // Launchers preserve argv[0] and every following argument.
    let invoked = std::env::args_os().next().unwrap_or_default();
    let name = std::path::Path::new(&invoked).file_stem().and_then(|name| name.to_str());
    // Windows resolves the old executable names without regard to case.
    #[cfg(windows)]
    let normalized = name.map(str::to_ascii_lowercase);
    #[cfg(windows)]
    let name = normalized.as_deref();
    match name {
        Some("rw_ensbatch") => ensbatch::main(),
        Some("rw_compare") => compare_cli::main(),
        Some("rw_verify") => verify_cli::main(),
        Some("rw_scorecard") => scorecard_cli::main(),
        Some("rw_nowcast_frames") => nowcast_frames::main(),
        _ => render_main(),
    }
}

mod fire_observations;
