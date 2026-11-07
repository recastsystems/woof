//! The door of `rw_mpas_static` -- make the STATIC file a generated grid
//! needs to run. The executable's other tools are dispatched in `main.rs`.
//!
//! A grid file is not runnable. It carries a mesh on a unit sphere with no
//! terrain, no land use and no soil, and every consumer that tries to run one
//! asks for a matching static and refuses without it. Until this existed, a
//! generated grid was a file nobody could use.
//!
//! It also GRADES itself: `--compare` puts a static beside a reference and
//! reports every field's agreement, so the builder is measured against a known
//! answer rather than trusted.
//!
//! ONE BUILDER, ONE DOOR. This used to be one of two static writers in the
//! crate: this one owned the schema the mesh registry pins, the other owned
//! host-memory admission and a byte-identical parallel tile decode, and one
//! edited `use` line was enough to swap which of them a user got. There is now
//! a single writer -- [`rw_mpas::static_builder`] -- and it is the streaming,
//! admission-gated one, carrying the computations this door's writer owned.
//! `rw_mpas::staticfile::schema::gate_bounded_writer_matches_pin` is what
//! holds that true.

use std::path::PathBuf;
use std::process::ExitCode;

use rw_mpas::static_builder::{
    build_static_reporting, build_terrain_reporting, StaticBuildConfig, StaticGeogPaths, DEFAULT_VALID_TIME,
    NOMINAL_DX_CROSS_CHECK_RELATIVE,
};
use rw_mpas::staticfile;

/// The literal a bridge contract handshakes on.
///
/// `[--legacy-static]` is in it because the DEFAULT changed under it: a bare
/// command writes the native MPAS-A v8.4.2 static stream with native GWD
/// statistics (2.9.0), where every earlier build wrote the legacy 82-variable
/// stream from the with-lakes land use. A pre-2.9.0 binary still answers
/// `--version` and still takes `--grid --out`, so without this literal moving
/// a stale install would keep writing legacy statics -- lakes stored as land
/// -- under a wrapper that believes it asked for native ones.
pub const ABI_MARKER: &str = "rw_mpas_static --grid GRID.nc --out STATIC.nc [--geog DIR] \
[--legacy-static] [--terrain-only] [--nominal-dx-m M] [--valid-time YYYY-MM-DD_hh:mm:ss] [--receipt JSON] [--clobber] \
| --compare REFERENCE.nc CANDIDATE.nc [--report JSON]";

/// Progress tokens this binary prints, one per stage, tab separated.
pub const PROGRESS_TOKENS: &str =
    "GRID\tNOMINALDX\tVECTORS\tOPERATORS\tGEOGGRID\tGWDBAND\tWROTE\tFINISHED";

/// The geography ladder, spelled from the code that walks it.
///
/// Printed rather than described so the help and the refusal cannot drift from
/// the search: the previous help named two of the candidates and the door
/// refused archives sitting in the third.
fn geog_ladder_text(separator: &str) -> String {
    let candidates = staticfile::geog::geog_root_candidates();
    if candidates.is_empty() {
        return "(no candidate resolves: neither $GPUWM_WPS_GEOG, \
                $GPUWM_CASE_DATA_ROOT nor a home directory is set)"
            .to_string();
    }
    candidates
        .iter()
        .map(|p| p.display().to_string())
        .collect::<Vec<_>>()
        .join(separator)
}

/// The dataset directory each field is read from, from the table itself.
fn geog_table_text(separator: &str) -> String {
    rw_mpas::static_builder::GEOG_DATASET_TABLE
        .iter()
        .map(|(slot, dataset)| format!("{slot:<17}{dataset}"))
        .collect::<Vec<_>>()
        .join(separator)
}

/// The directories `gpuwm fetch-geog` stages under a container, from the
/// table itself, so the help cannot describe a layout the builder does not
/// read.
fn geog_container_text() -> String {
    let mut containers: Vec<&str> = Vec::new();
    for (_, container) in rw_mpas::static_builder::GEOG_CONTAINER_TABLE.iter() {
        if !containers.contains(container) {
            containers.push(container);
        }
    }
    containers
        .iter()
        .map(|container| {
            let children = rw_mpas::static_builder::GEOG_CONTAINER_TABLE
                .iter()
                .filter(|(_, c)| c == container)
                .map(|(child, _)| *child)
                .collect::<Vec<_>>()
                .join(", ");
            format!("{container}/ carries {children}")
        })
        .collect::<Vec<_>>()
        .join("; ")
}

fn usage() -> String {
    format!(
        "usage: {ABI_MARKER}\n\n\
         WHAT A STATIC IS\n\
        \x20  The same mesh as the grid, carried onto the earth radius and given geography.\n\
        \x20  Terrain, land use, soil class and composition, deep-soil temperature,\n\
        \x20  green-ness and albedo climatologies, sub-grid orography statistics for the\n\
        \x20  gravity-wave drag, the operator tables, and an FP32-bit-exact nominalMinDc.\n\n\
         THE INPUT\n\
         --grid           the grid file to build against. Its topology is copied VERBATIM, so\n\
        \x20                 the pair passes a consumer's bit-identity cross-examination.\n\
         --geog           WPS_GEOG root. Defaults to the ladder `gpuwm fetch-geog` stages\n\
        \x20                 into, best first:\n\
        \x20                   {}\n\
         --nominal-dx-m   the nominal spacing to DECLARE, in metres. Defaults to the grid's own\n\
        \x20                 implied value. Refused when it disagrees with the grid by more than\n\
        \x20                 {:.1}%, because that means grid and static are not the same mesh.\n\
         --valid-time     the xtime stamp. Default {}.\n\n\
         THE DEFAULT IS THE NATIVE STATIC\n\
        \x20  With no --namelist, the build applies MPAS-A v8.4.2's own init_atmosphere\n\
        \x20  registry defaults, exactly as an empty namelist would: the native static\n\
        \x20  stream (79 variables), MODIFIED_IGBP_MODIS_NOAH land use read from\n\
        \x20  modis_landuse_20class_30s (lakes are water, category 17, as in native),\n\
        \x20  STATSGO soil, MODIS vegetation, albedo and snow albedo, supersample 3 for\n\
        \x20  albedo and snow, and the ten GWD statistics by native arithmetic over the\n\
        \x20  USGS landuse_30s water mask. Explicit flags below still override.\n\
         --legacy-static  the pre-2.9.0 stream instead: 82 variables, with-lakes land use\n\
        \x20                 (its lake class, 21, is water: category 17, water soil and land\n\
        \x20                 mask 0, and water in the GWD statistics), legacy GWD statistics.\n\
        \x20                 For a consumer that still needs that schema. It is NOT the bytes\n\
        \x20                 of a pre-2.9.0 build: 2.9.0's shared static arithmetic applies.\n\n\
         STATIC OPTIONS\n\
         --supersample N  native general sampling factor (albedo and snow)\n\
         --landuse-supersample N  native land-use sampling factor\n\
         --supersample-30s N  native terrain, soil, vegetation and texture factor\n\
         --topo-data GTOPO30|GMTED2010\n\
         --landuse-data USGS|MODIFIED_IGBP_MODIS_NOAH|MODIFIED_IGBP_MODIS_NOAH_15s\n\
         --soilcat-data SOILGRIDS|STATSGO|BNU  default SOILGRIDS; STATSGO is FAO-UNESCO\n\
        \x20                 outside CONUS, not permissive, read only when named\n\
         --soiltemp-data ERA5|ISLSCP  default ERA5; ISLSCP-I soiltemp_1deg has an\n\
        \x20                 unclear licence, read only when named\n\
         --vegfrac-data MODIS|NCEP  --albedo-data MODIS|NCEP\n\
         --maxsnowalbedo-data MODIS|NCEP\n\
         --no-gwd        build the static stage separately from GWD\n\
         --no-noahmp     omit the optional Noah-MP geography pass\n\
         --preserve-lake-mask  add tc_lakemask from the with-lakes MODIS archive\n\
         --native-init   emit the native init_atmosphere static stream\n\
         --namelist FILE native init_atmosphere static options; explicit flags override\n\
         THE OUTPUT\n\
         --out            static file to write\n\
         --receipt        write the build receipt as JSON here\n\
         --native-gwd-static  compute the ten orographic statistics using native MPAS v8.4.2\n\
        \x20                 arithmetic and USGS landuse_30s (RKIND=binary32).\n\
         --topo-data      GTOPO30 or GMTED2010; applies to --native-gwd-static.\n\
         --gwd-cell-scaling  positive effective-diameter factor; native default 1.0.\n\
         --gwd-only       write only the ten native GWD fields as an init overlay.\n\
         --terrain-only   write the exact terrain and operator overlay for the vertical\n\
        \x20                 authority; only the terrain archive is read, no runnable static\n\
         --clobber        replace an existing --out\n\n\
         GRADING\n\
         --compare A B    read two statics and report every field's agreement. Prints one line\n\
        \x20                 per field: name, count, exactly-equal, worst absolute, worst relative.\n\
         --report         write the comparison as JSON here\n\n\
         HOST MEMORY\n\
        \x20  Geography is streamed tile-plane by tile-plane and the sub-grid orography is\n\
        \x20  streamed in latitude bands, so peak host memory is bounded by the mesh plus one\n\
        \x20  band and never by the size of the archive. RW_MPAS_HOST_MEMORY_LIMIT_GIB (or\n\
        \x20  _BYTES) sets the budget the admission gate plans against; RW_MPAS_TILE_WORKERS\n\
        \x20  caps how many tiles decode at once. The static is byte-identical at every\n\
        \x20  setting of both.\n\n\
         WHERE EACH FIELD COMES FROM\n\
        \x20  {}\n\
        \x20  Each directory is read at <root>/<directory>, or at the container\n\
        \x20  `gpuwm fetch-geog` stages it under: {}.\n\n\
         WRITTEN AS A BITWISE +0 PLACEHOLDER, NOT AS VALUES\n\
        \x20  {}\n\
        \x20    the local frames and the RBF reconstruction weights belong to the vertical\n\
        \x20    grid the init stage builds; a consumer overlays the exact arrays over these\n\
        \x20    slots. The SLOT is not optional -- a reader that finds the variable absent\n\
        \x20    stops before it allocates any device memory.\n\n\
         NO LONGER ABSENT\n\
        \x20  cell_gradient_coef_x, cell_gradient_coef_y, soilcomp, soilcl1, soilcl2,\n\
        \x20  soilcl3, soilcl4 are all WRITTEN, with real values. They were absent while\n\
        \x20  two writers split the schema between them; they are named here so a reader\n\
        \x20  who learnt the old absent list does not go looking for a gap that is not there.\n\n\
         LEFT TO THE INIT FILE, AS NATIVE DOES\n\
        \x20  defc_a, defc_b, lu_index, soilcat_top are not in the native static stream\n\
        \x20  (MPAS Registry.xml); --legacy-static still writes them.\n",
        geog_ladder_text("\n\x20                   "),
        NOMINAL_DX_CROSS_CHECK_RELATIVE * 100.0,
        DEFAULT_VALID_TIME,
        geog_table_text("\n\x20  "),
        geog_container_text(),
        staticfile::ZERO_PLACEHOLDER_FIELDS.join(", "),
    ) + &tools_text()
}

/// The executable's other tools, from the dispatch table itself.
fn tools_text() -> String {
    let mut text = String::from(
        "\n TOOLS IN THIS EXECUTABLE\n\
        \x20  rw_mpas_static TOOL ARGS... A link or copy named after the former binary\n\
        \x20  (or that name plus -suffix) runs the same tool with the same ARGS.\n",
    );
    for tool in &crate::TOOLS {
        text.push_str(&format!("   {:<13}{} (was {})\n", tool.name, tool.summary, tool.legacy));
    }
    text
}

fn value(argv: &[String], flag: &str) -> Option<String> {
    argv.iter()
        .position(|a| a == flag)
        .and_then(|i| argv.get(i + 1))
        .cloned()
}

/// The door: every invocation that names no other tool.
pub fn run(argv: Vec<std::ffi::OsString>) -> ExitCode {
    let Ok(argv) = crate::utf8_arguments("rw_mpas_static", argv) else { return ExitCode::from(2) };

    if argv.iter().any(|a| a == "--version" || a == "-V") {
        println!("rw_mpas_static {}", env!("CARGO_PKG_VERSION"));
        return ExitCode::SUCCESS;
    }
    if argv.iter().any(|a| a == "--abi") {
        println!("{ABI_MARKER}\n{PROGRESS_TOKENS}\n{}", crate::tools_abi_line());
        return ExitCode::SUCCESS;
    }
    if argv.is_empty() || argv.iter().any(|a| a == "-h" || a == "--help") {
        print!("{}", usage());
        return ExitCode::SUCCESS;
    }

    if let Some(i) = argv.iter().position(|a| a == "--compare") {
        let (Some(a), Some(b)) = (argv.get(i + 1), argv.get(i + 2)) else {
            eprintln!("--compare needs two files: REFERENCE.nc CANDIDATE.nc");
            return ExitCode::from(2);
        };
        return match staticfile::compare::compare(&PathBuf::from(a), &PathBuf::from(b)) {
            Ok(c) => {
                println!(
                    "DIMENSIONS\t{}",
                    if c.dimensions_agree { "AGREE" } else { "DIFFER" }
                );
                if !c.dimensions_agree {
                    println!("  reference: {:?}", c.reference_dimensions);
                    println!("  candidate: {:?}", c.candidate_dimensions);
                }
                println!(
                    "FIELD\tN\tEXACT\tMAX_ABS\tMAX_REL\tREF_RANGE\tCAND_RANGE"
                );
                for f in &c.fields {
                    println!(
                        "{}\t{}\t{}\t{:.6e}\t{:.6e}\t[{:.6},{:.6}]\t[{:.6},{:.6}]",
                        f.name,
                        f.n,
                        f.exact,
                        f.max_abs,
                        f.max_rel,
                        f.reference_min,
                        f.reference_max,
                        f.candidate_min,
                        f.candidate_max
                    );
                }
                if !c.only_in_reference.is_empty() {
                    println!("ONLY_IN_REFERENCE\t{}", c.only_in_reference.join(","));
                }
                if !c.only_in_candidate.is_empty() {
                    println!("ONLY_IN_CANDIDATE\t{}", c.only_in_candidate.join(","));
                }
                if !c.shape_mismatch.is_empty() {
                    println!("NOT_GRADED\t{}", c.shape_mismatch.join(","));
                }
                if !c.type_mismatch.is_empty() {println!("TYPE_MISMATCH\t{}",c.type_mismatch.join(","));}
                println!("WORD_IDENTICAL\t{}",c.word_identical);
                if let Some(path)=value(&argv,"--word-witness") {
                    let result=staticfile::compare::differing_word_witness(&PathBuf::from(a),&PathBuf::from(b))
                        .map_err(|e|e.to_string())
                        .and_then(|w|serde_json::to_string_pretty(&w).map_err(|e|e.to_string()))
                        .and_then(|s|std::fs::write(&path,s).map_err(|e|e.to_string()));
                    if let Err(e)=result {eprintln!("cannot write exact word witness {path}: {e}");return ExitCode::FAILURE;}
                }
                if let Some(p) = value(&argv, "--report") {
                    match serde_json::to_string_pretty(&c)
                        .map_err(|e| e.to_string())
                        .and_then(|s| std::fs::write(&p, s).map_err(|e| e.to_string()))
                    {
                        Ok(()) => println!("REPORT\t{p}"),
                        Err(e) => {
                            eprintln!("cannot write {p}: {e}");
                            return ExitCode::FAILURE;
                        }
                    }
                }
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("{e}");
                ExitCode::FAILURE
            }
        };
    }

    let Some(grid) = value(&argv, "--grid") else {
        eprintln!("--grid is required. {}", ABI_MARKER);
        return ExitCode::from(2);
    };
    let Some(out) = value(&argv, "--out") else {
        eprintln!("--out is required. {}", ABI_MARKER);
        return ExitCode::from(2);
    };
    // 2.9.0: the native static is the default. `--legacy-static` is the one
    // way back to the pre-2.9.0 stream, and it cannot be combined with a
    // request for the native one: the two ask for different files.
    let legacy = argv.iter().any(|a| a == "--legacy-static");
    let terrain_only = argv.iter().any(|a| a == "--terrain-only");
    if legacy {
        for native_only in ["--namelist", "--native-init"] {
            if argv.iter().any(|a| a == native_only) {
                eprintln!("--legacy-static and {native_only} ask for different files: {native_only} \
                           selects the native static stream, --legacy-static the pre-2.9.0 one. Pass one.");
                return ExitCode::from(2);
            }
        }
    }
    let namelist = match value(&argv,"--namelist") {
        Some(path) => match std::fs::read_to_string(&path).map_err(rw_mpas::MpasError::from)
            .and_then(|text|rw_mpas::static_namelist::StaticNamelist::parse(&text)) {
                Ok(n)=>Some(n),Err(e)=>{eprintln!("{path}: {e}");return ExitCode::from(2);}
            },
        // MPAS-A's own registry defaults, as an empty namelist gives them.
        // `--terrain-only` writes an overlay, not a static, and keeps its
        // own contract.
        None if !legacy && !terrain_only => Some(rw_mpas::static_namelist::StaticNamelist::default()),
        None=>None,
    };
    let geog = match value(&argv, "--geog").map(PathBuf::from)
        .or_else(||namelist.as_ref().and_then(|n|n.geog_root()))
        .or_else(staticfile::geog::default_geog_root) {
        Some(p) => p,
        None => {
            eprintln!(
                "no geography root. A static carries terrain, land use and soil, so it cannot be \
                 built without one. Pass --geog DIR, run `gpuwm fetch-geog`, or put the archive \
                 at one of these -- every one was looked at and none is a directory:\n  {}",
                geog_ladder_text("\n  ")
            );
            return ExitCode::from(2);
        }
    };
    let nominal_dx_m = match value(&argv, "--nominal-dx-m").map(|s| s.parse::<f64>()) {
        Some(Ok(v)) => Some(v),
        Some(Err(e)) => {
            eprintln!("--nominal-dx-m: {e}");
            return ExitCode::from(2);
        }
        None => None,
    };

    // Native GWD is part of the native default; legacy keeps the old
    // statistics unless it asks for native ones by name.
    let native_gwd_requested = argv.iter().any(|a| a == "--native-gwd-static")
        || (namelist.is_some() && !argv.iter().any(|a| a == "--no-gwd"));
    let native_gwd = if native_gwd_requested {
        let scaling = match value(&argv, "--gwd-cell-scaling").unwrap_or_else(|| "1.0".into()).parse::<f32>() {
            Ok(v) => v,
            Err(e) => { eprintln!("--gwd-cell-scaling: {e}"); return ExitCode::from(2); }
        };
        match rw_mpas::native_gwd::NativeGwdConfig::under(&geog,
            &value(&argv, "--topo-data").unwrap_or_else(|| "GMTED2010".into()), scaling) {
            Ok(cfg) => Some(cfg),
            Err(e) => { eprintln!("{e}"); return ExitCode::from(2); }
        }
    } else { None };
    if argv.iter().any(|a| a == "--gwd-only") {
        let Some(native) = native_gwd.as_ref() else {
            eprintln!("--gwd-only writes native GWD statistics; with --legacy-static or --no-gwd it also needs \
                       --native-gwd-static, because an overlay must specify the statistics it carries");
            return ExitCode::from(2);
        };
        let receipt = match rw_mpas::native_gwd::write_overlay(&PathBuf::from(&grid), &PathBuf::from(&out), native,
            argv.iter().any(|a| a == "--clobber"), &|line| println!("{line}")) {
            Ok(r) => r,
            Err(e) => { eprintln!("{e}"); return ExitCode::FAILURE; }
        };
        if let Some(path) = value(&argv, "--receipt") {
            let result = serde_json::to_vec_pretty(&receipt).map_err(|e| e.to_string())
                .and_then(|json| std::fs::write(&path, json).map_err(|e| e.to_string()));
            if let Err(e) = result { eprintln!("cannot write {path}: {e}"); return ExitCode::FAILURE; }
            println!("RECEIPT\t{path}");
        }
        let bytes = std::fs::metadata(&out).map(|m| m.len()).unwrap_or(0);
        println!("WROTE\t{out}\t{bytes}\t10\t{}", receipt.output_sha256);
        return ExitCode::SUCCESS;
    }
    let mut geog_paths = StaticGeogPaths::under(&geog);
    if let Some(native) = &native_gwd { geog_paths.terrain = native.topo.clone(); }
    let mut cfg = StaticBuildConfig {
        native_gwd,
        stages: rw_mpas::static_builder::StaticStageOptions::default(),
        grid_path: PathBuf::from(&grid),
        out_path: PathBuf::from(&out),
        geog: geog_paths,
        supersample: 1,
        supersample_landuse: 1,
        supersample_30s: 1,
        host_memory_limit_bytes: None,
        tile_workers: None,
        receipt_path: None,
        provenance: format!("rw_mpas_static {}", env!("CARGO_PKG_VERSION")),
        valid_time: value(&argv, "--valid-time")
            .unwrap_or_else(|| DEFAULT_VALID_TIME.to_string()),
        nominal_dx_m,
        clobber: argv.iter().any(|a| a == "--clobber"),
        // The GRID says which representation its static must carry; there is
        // no flag, because the choice is a property of the mesh (does it have
        // a native MPAS-A counterpart?) and not of the command line.  A grid
        // that declares nothing gets binary32 -- every published mesh, and
        // every cull of one.
        coordinates: match rw_mpas::staticfile::coordframe::for_static_from_grid(&PathBuf::from(
            &grid,
        )) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("{e}");
                return ExitCode::from(2);
            }
        },
    };

    if let Some(nml)=namelist {
        if let Err(e)=nml.apply(&mut cfg,&geog) {eprintln!("{e}");return ExitCode::from(2);}
        // The original namelist is pinned in the output's existing provenance.
        match value(&argv,"--namelist") {
            Some(path) => match rw_mpas::sha256_file(&PathBuf::from(path)) {
                Ok(hash)=>cfg.provenance.push_str(&format!("; namelist_sha256={hash}")),
                Err(e)=>{eprintln!("{e}");return ExitCode::from(2);}
            },
            None => cfg.provenance.push_str("; namelist=native registry defaults"),
        }
    } else if legacy {
        cfg.provenance.push_str("; legacy-static");
    }
    if let Some(stamp)=value(&argv,"--valid-time") {cfg.valid_time=stamp;}

    // MPAS: mpas_init_atm_static.F:369-447,488-1250: expose the native
    // selectors and factors through the same executable users run.
    for (flag,target) in [
        ("--supersample",&mut cfg.supersample),
        ("--landuse-supersample",&mut cfg.supersample_landuse),
        ("--supersample-30s",&mut cfg.supersample_30s),
    ] {
        if let Some(raw)=value(&argv,flag) {
            match rw_mpas::static_namelist::sampling_factor(&raw) {
                Ok(n) => *target=n,
                Err(error) => { eprintln!("{flag}: {error}"); return ExitCode::from(2); }
            }
        }
    }
    if argv.iter().any(|a| a=="--native-gwd-static") {cfg.stages.gwd=true;}
    if argv.iter().any(|a| a=="--no-gwd") {cfg.stages.gwd=false;}
    if argv.iter().any(|a| a=="--no-noahmp") {cfg.stages.noahmp=false;}
    cfg.stages.native_init |= argv.iter().any(|a| a=="--native-init");
    if argv.iter().any(|a| a == "--preserve-lake-mask") {
        if !cfg.stages.native_init || terrain_only {
            eprintln!("--preserve-lake-mask requires the native static stream");
            return ExitCode::from(2);
        }
        cfg.stages.lake_mask_geog = Some(rw_mpas::static_builder::resolve_geog_dataset_dir(
            &geog, "modis_landuse_20class_30s_with_lakes"));
    }
    if cfg.stages.native_init && cfg.stages.landuse_label.is_none() {
        // MPAS: mpas_init_atm_static.F:369-393. The driver supplies this
        // label independently of the geoindex's lowercased character data.
        cfg.stages.landuse_label=Some("MODIFIED_IGBP_MODIS_NOAH".into());
    }
    for flag in ["topo-data","landuse-data","soilcat-data","soiltemp-data","vegfrac-data","albedo-data","maxsnowalbedo-data"] {
        if let Some(choice)=value(&argv,&format!("--{flag}")) {
            if let Err(e)=rw_mpas::static_builder::select_static_source(&mut cfg,&geog,flag,&choice) {
                eprintln!("{e}"); return ExitCode::from(2);
            }
        }
    }

    // MPAS: mpas_init_atm_gwd.F:102-156. The selected native static stage
    // uses the same terrain choice as static interpolation, and independent USGS.
    if cfg.stages.gwd && (cfg.stages.native_init || cfg.native_gwd.is_some()) {
        let topo_data = if cfg.geog.terrain == geog.join("topo_30s") {
            "GTOPO30"
        } else { "GMTED2010" };
        let scaling = cfg.native_gwd.as_ref().map_or(1.0, |gwd| gwd.cell_scaling);
        let scaling = match value(&argv, "--gwd-cell-scaling").map(|v| v.parse::<f32>()) {
            Some(Ok(v)) => v,
            Some(Err(e)) => {eprintln!("--gwd-cell-scaling: {e}"); return ExitCode::from(2);}
            None => scaling,
        };
        cfg.native_gwd = match rw_mpas::native_gwd::NativeGwdConfig::under(&geog, topo_data, scaling) {
            Ok(gwd) => Some(gwd),
            Err(e) => {eprintln!("{e}"); return ExitCode::from(2);}
        };
    } else if !cfg.stages.gwd { cfg.native_gwd = None; }

    let started = std::time::Instant::now();
    if argv.iter().any(|a| a == "--terrain-only") {
        let receipt = match build_terrain_reporting(&cfg, &|line| println!("{line}")) {
            Ok(r) => r,
            Err(e) => { eprintln!("{e}"); return ExitCode::FAILURE; }
        };
        println!("WROTE\t{}\t{}\t{}\t{}", receipt.output_path, receipt.output_bytes, receipt.variables.len(), receipt.sha256);
        if let Some(path) = value(&argv, "--receipt") {
            let result = serde_json::to_vec_pretty(&receipt)
                .map_err(|e| e.to_string())
                .and_then(|json| std::fs::write(&path, json).map_err(|e| e.to_string()));
            if let Err(error) = result {
                eprintln!("cannot write {path}: {error}"); return ExitCode::FAILURE;
            }
            println!("RECEIPT\t{path}");
        }
        println!("FINISHED\t{:.2}", started.elapsed().as_secs_f64());
        return ExitCode::SUCCESS;
    }
    let receipt = match build_static_reporting(&cfg, &|line| println!("{line}")) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };
    println!(
        "WROTE\t{}\t{}\t{}\t{}",
        receipt.output_path,
        receipt.output_bytes,
        receipt.variables.len(),
        receipt.sha256
    );
    if let Some(p) = value(&argv, "--receipt") {
        match serde_json::to_vec_pretty(&receipt)
            .map_err(|e| e.to_string())
            .and_then(|json| std::fs::write(&p, json).map_err(|e| e.to_string()))
        {
            Ok(()) => println!("RECEIPT\t{p}"),
            Err(e) => {
                eprintln!("cannot write {p}: {e}");
                return ExitCode::FAILURE;
            }
        }
    }
    println!("FINISHED\t{:.2}", started.elapsed().as_secs_f64());
    ExitCode::SUCCESS
}
