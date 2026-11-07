use clap::{Args, Parser, Subcommand};
use rw_wps::mapping::{
    SourceFormat, import_wps_vtable, inspect_sources, inventory_sources, mapping_template,
    read_mapping, validate_mapping,
};
use rw_wps::namelist::{NamelistSupportRequest, query_namelist_support};
use rw_wps::{
    AUTHOR_MAPPED_PLAN_SCHEMA, AuthorMappedDecoder, AuthorMappedRequest, AuthorMappingInput,
    RwWpsError, build_author_mapped_args, discover_capabilities, plan, read_config, run, template,
    validate_config,
};
use serde::Serialize;
use std::ffi::OsString;
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Parser)]
#[command(name = "rw-wps", about = "Rusty Weather Preprocessing Suite")]
struct Cli {
    /// Native gpuwm preprocessing entrypoint.
    #[arg(long, global = true)]
    engine: Option<OsString>,
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Args)]
struct AuthorMappedArgs {
    #[arg(long, value_parser = ["grib1", "grib2", "netcdf"])]
    source_format: String,
    /// Use an existing executable rw-wps.mapping.v1 document.
    #[arg(long)]
    mapping: Option<PathBuf>,
    /// Author a mapping from this explicit rw-wps.descriptor.v1 document.
    #[arg(long)]
    descriptor: Option<PathBuf>,
    /// Required with a GRIB descriptor; forbidden for NetCDF/existing mappings.
    #[arg(long)]
    vtable: Option<PathBuf>,
    /// Create-only mapping output used with --descriptor.
    #[arg(long)]
    author_mapping: Option<PathBuf>,
    #[arg(long)]
    composition: PathBuf,
    /// Primary source files in deterministic time/file order.
    #[arg(long, required = true)]
    input: Vec<PathBuf>,
    /// ROLE=PATH supplement binding; repeat to preserve role/file order.
    #[arg(long, required = true)]
    supplement: Vec<OsString>,
    /// ROLE=PATH provenance binding.
    #[arg(long, required = true)]
    provenance: Vec<OsString>,
    /// GRIB1 decoder executable.
    #[arg(long)]
    bridge: Option<PathBuf>,
    /// GRIB2 inventory executable.
    #[arg(long)]
    grib2_inventory: Option<PathBuf>,
    /// GRIB2 dump executable.
    #[arg(long)]
    grib2_dump: Option<PathBuf>,
    /// Create-only exact input manifest output.
    #[arg(long)]
    author_input_manifest: PathBuf,
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// Produce WRF auxiliary surface input in Rust.
    Wrflowinp {
        #[arg(long, required = true)]
        input: Vec<PathBuf>,
        #[arg(long)]
        output: PathBuf,
        #[arg(long, default_value_t = 2)]
        sf_surface_physics: i32,
    },
    /// Prepare refined fire-grid fuel and terrain using the Rust static builder.
    FireStatic {
        config: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        receipt: Option<PathBuf>,
        #[arg(long, default_value = "gpuwm-sfire-static")]
        static_engine: OsString,
    },
    /// Discover the engine's provenance-bound source capabilities.
    Sources,
    /// Classify paired WRF/WPS namelists and inventory required initialized state.
    NamelistSupport {
        /// Standard WPS namelist.wps; geometry/topology authority.
        wps: PathBuf,
        /// Standard WRF namelist.input; vertical/physics authority.
        input: PathBuf,
        /// Smallest pressure represented by the selected source product.
        #[arg(long)]
        source_top_pressure_pa: Option<f64>,
    },
    /// Print a fail-closed starter configuration for a preset or generic mapping.
    Template {
        #[arg(value_parser = ["hrrr", "gfs", "era5", "mapped"])]
        source: String,
    },
    /// Print a generic mapping skeleton for GRIB1, GRIB2, or NetCDF.
    MappingTemplate {
        #[arg(value_parser = ["grib1", "grib2", "netcdf"])]
        format: String,
    },
    /// Validate a generic mapping against its physics/domain contract.
    MappingValidate { mapping: PathBuf },
    /// Inventory real source metadata before authoring a mapping.
    Inventory {
        #[arg(value_parser = ["grib1", "grib2", "netcdf"])]
        format: String,
        #[arg(required = true)]
        input: Vec<PathBuf>,
    },
    /// Resolve every direct mapping selector against real source metadata.
    Inspect {
        mapping: PathBuf,
        #[arg(required = true)]
        input: Vec<PathBuf>,
    },
    /// Preserve a WPS Vtable as a hash-bound, non-executable row draft.
    ImportVtable {
        vtable: PathBuf,
        #[arg(long, default_value = "imported-wps-vtable")]
        name: String,
    },
    /// Print an exact create-only mapped authoring argv for woof-wrf-init.
    AuthorMapped(Box<AuthorMappedArgs>),
    /// Validate configuration and all declared input paths.
    Validate { config: PathBuf },
    /// Print the exact native-engine argv without launching it.
    Plan { config: PathBuf },
    /// Run the native engine, stream progress, and seal a WRF output receipt.
    Run {
        config: PathBuf,
        #[arg(long)]
        receipt: PathBuf,
        /// Emit machine-readable progress events to stderr.
        #[arg(long)]
        json_progress: bool,
    },
}

fn print_json(value: &impl Serialize) -> Result<(), RwWpsError> {
    println!("{}", serde_json::to_string_pretty(value)?);
    Ok(())
}

fn real_main() -> Result<(), RwWpsError> {
    let cli = Cli::parse();
    let engine = cli
        .engine
        .or_else(|| std::env::var_os("RW_WPS_ENGINE"))
        .unwrap_or_else(|| OsString::from("woof-wrf-init"));
    match cli.command {
        Commands::Wrflowinp { input, output, sf_surface_physics } => {
            print_json(&rw_wps::wrflowinp::run(&input, &output, sf_surface_physics)?)
        }
        Commands::FireStatic {config,output,receipt,static_engine} => {
            rw_wps::sfire::run(&config,&output,receipt.as_deref(),&static_engine)
        }
        Commands::Sources => {
            let (manifest, _) = discover_capabilities(&engine)?;
            print_json(&manifest)
        }
        Commands::NamelistSupport {
            wps,
            input,
            source_top_pressure_pa,
        } => {
            let report = query_namelist_support(
                &engine,
                &NamelistSupportRequest {
                    wps,
                    input,
                    source_top_pressure_pa,
                },
            )?;
            print_json(&report)?;
            if report.verdict == "PASS" {
                Ok(())
            } else {
                Err(RwWpsError::Config(
                    "namelist configuration is unsupported; inspect the ".to_owned()
                        + "machine-readable issues above",
                ))
            }
        }
        Commands::Template { source } => print_json(&template(&source)?),
        Commands::MappingTemplate { format } => {
            print_json(&mapping_template(parse_format(&format)?))
        }
        Commands::MappingValidate { mapping } => {
            let report = validate_mapping(&read_mapping(&mapping)?);
            print_json(&report)?;
            if report.passed() {
                Ok(())
            } else {
                Err(RwWpsError::Config(format!(
                    "mapping validation failed with {} error(s)",
                    report.errors.len()
                )))
            }
        }
        Commands::Inventory { format, input } => {
            print_json(&inventory_sources(parse_format(&format)?, &input)?)
        }
        Commands::Inspect { mapping, input } => {
            let report = inspect_sources(&read_mapping(&mapping)?, &input)?;
            print_json(&report)?;
            if report.verdict == "PASS" {
                Ok(())
            } else {
                Err(RwWpsError::Config(format!(
                    "source inspection failed: {}",
                    report.errors.join("; ")
                )))
            }
        }
        Commands::ImportVtable { vtable, name } => {
            let bytes = fs::read(&vtable).map_err(|source| RwWpsError::Io {
                path: vtable.clone(),
                source,
            })?;
            print_json(&import_wps_vtable(&bytes, name)?)
        }
        Commands::AuthorMapped(args) => {
            let AuthorMappedArgs {
                source_format,
                mapping,
                descriptor,
                vtable,
                author_mapping,
                composition,
                input,
                supplement,
                provenance,
                bridge,
                grib2_inventory,
                grib2_dump,
                author_input_manifest,
            } = *args;
            let format = parse_format(&source_format)?;
            let mapping = author_mapping_mode(mapping, descriptor, vtable, author_mapping)?;
            let decoder = author_decoder(format, bridge, grib2_inventory, grib2_dump)?;
            let args = build_author_mapped_args(&AuthorMappedRequest {
                format,
                mapping,
                composition,
                primary_files: input,
                supplements: supplement,
                provenance,
                decoder,
                input_manifest: author_input_manifest,
            })?;
            print_json(&serde_json::json!({
                "schema": AUTHOR_MAPPED_PLAN_SCHEMA,
                "dry_plan": true,
                "engine": engine.to_string_lossy(),
                "args": args
                    .iter()
                    .map(|value| value.to_string_lossy().into_owned())
                    .collect::<Vec<_>>(),
            }))
        }
        Commands::Validate { config } => {
            let (config, _) = read_config(&config)?;
            validate_config(&config, true)?;
            print_json(&serde_json::json!({
                "schema": "rw-wps.validation.v1",
                "verdict": "PASS",
                "source": config.source.id(),
                "backend": config.backend.id(),
                "output_root": config.output.root(),
            }))
        }
        Commands::Plan { config } => {
            let (config, bytes) = read_config(&config)?;
            let planned = plan(config, bytes)?;
            let args = planned
                .args
                .iter()
                .map(|value| value.to_string_lossy().into_owned())
                .collect::<Vec<_>>();
            print_json(&serde_json::json!({
                "schema": "rw-wps.plan.v1",
                "engine": engine.to_string_lossy(),
                "config_sha256": planned.config_sha256,
                "args": args,
            }))
        }
        Commands::Run {
            config,
            receipt,
            json_progress,
        } => {
            let (config, bytes) = read_config(&config)?;
            let planned = plan(config, bytes)?;
            let result = run(&engine, &planned, &receipt, |event| {
                if json_progress {
                    eprintln!(
                        "{}",
                        serde_json::to_string(event).unwrap_or_else(|_| {
                            "{\"schema\":\"rw-wps.progress.v1\",\"stage\":\"error\"}".to_owned()
                        })
                    );
                } else {
                    eprintln!("RW-WPS [{}] {}", event.stage, event.message);
                }
            })?;
            print_json(&result)
        }
    }
}

fn author_mapping_mode(
    mapping: Option<PathBuf>,
    descriptor: Option<PathBuf>,
    vtable: Option<PathBuf>,
    author_mapping: Option<PathBuf>,
) -> Result<AuthorMappingInput, RwWpsError> {
    match (mapping, descriptor, author_mapping) {
        (Some(mapping), None, None) if vtable.is_none() => {
            Ok(AuthorMappingInput::Existing { mapping })
        }
        (Some(_), None, None) => Err(RwWpsError::Config(
            "--vtable is only used with --descriptor and --author-mapping".to_owned(),
        )),
        (None, Some(descriptor), Some(output_mapping)) => Ok(AuthorMappingInput::Descriptor {
            descriptor,
            vtable,
            output_mapping,
        }),
        _ => Err(RwWpsError::Config(
            "choose exactly one of --mapping or --descriptor with --author-mapping".to_owned(),
        )),
    }
}

fn author_decoder(
    format: SourceFormat,
    bridge: Option<PathBuf>,
    grib2_inventory: Option<PathBuf>,
    grib2_dump: Option<PathBuf>,
) -> Result<AuthorMappedDecoder, RwWpsError> {
    match (format, bridge, grib2_inventory, grib2_dump) {
        (SourceFormat::Grib1, Some(bridge), None, None) => {
            Ok(AuthorMappedDecoder::Grib1 { bridge })
        }
        (SourceFormat::Grib2, None, Some(inventory), Some(dump)) => {
            Ok(AuthorMappedDecoder::Grib2 { inventory, dump })
        }
        (SourceFormat::Netcdf, None, None, None) => Ok(AuthorMappedDecoder::Netcdf),
        (SourceFormat::Grib1, ..) => Err(RwWpsError::Config(
            "GRIB1 authoring requires exactly --bridge".to_owned(),
        )),
        (SourceFormat::Grib2, ..) => Err(RwWpsError::Config(
            "GRIB2 authoring requires exactly --grib2-inventory and --grib2-dump".to_owned(),
        )),
        (SourceFormat::Netcdf, ..) => Err(RwWpsError::Config(
            "NetCDF authoring does not accept external GRIB decoder paths".to_owned(),
        )),
    }
}

fn parse_format(value: &str) -> Result<SourceFormat, RwWpsError> {
    match value {
        "grib1" => Ok(SourceFormat::Grib1),
        "grib2" => Ok(SourceFormat::Grib2),
        "netcdf" => Ok(SourceFormat::Netcdf),
        _ => Err(RwWpsError::Config(format!(
            "unsupported source format {value:?}"
        ))),
    }
}

fn main() {
    if let Err(error) = real_main() {
        eprintln!("rw-wps: {error}");
        std::process::exit(1);
    }
}
