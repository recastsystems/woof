//! `rw_mpas_init`: build an MPAS initial-conditions file from a WPS
//! intermediate file, in Rust, with no Fortran in the chain.
//!
//! Every switch that selects physics is required and closed.  There is no
//! default `--extrap-airtemp`, no default `--use-spechumd`, and no default
//! `--virtual-factor`: each of them changes the numbers in a file that will
//! look structurally perfect either way, and this project has already shipped
//! one silent default that disabled three repairs at once.

use std::path::PathBuf;
use std::process::ExitCode;

use rw_mpas::init::dynamics::VirtualFactor;
use rw_mpas::init::hinterp::Underflow;
use rw_mpas::init::soil::DeepMoisture;
use rw_mpas::init::vinterp::Extrap;
use rw_mpas::init::{build_init, InitConfig, Lineage};

/// `GPUWM_BRIDGE_SOURCE_REV=<40-hex commit>`: the source revision this
/// binary was built from, embedded so the gpuwm release cut can prove a
/// staged bridge matches the commit being released by reading bytes alone
/// (`tools/build_bridge_bundle.py pin --source-rev`).  `build.rs` injects
/// the value; `main` references the constant so the linker cannot discard
/// it.
pub static GPUWM_BRIDGE_SOURCE_REV_STAMP: &str =
    concat!("GPUWM_BRIDGE_SOURCE_REV=", env!("GPUWM_BRIDGE_SOURCE_REV"));

/// The literal the Python bridge contract handshakes on.  It spells the
/// argument vector out, so a stale binary on a user's disk fails the static
/// check before it can accept input it will mis-parse.
pub const ABI_MARKER: &str = "rw_mpas_init --met MET --static STATIC.nc --capsule CAPSULE.nc \
--reference REFERENCE.nc --out INIT.nc --start-time YYYY-MM-DD_HH:MM:SS --nfglevels N \
--nfgsoillevels N --extrap-airtemp MODE --use-spechumd yes|no --theta-adv-order N \
--coef-3rd-order X --virtual-factor reproduce-fortran|consistent \
--deep-soil-moisture reproduce-fortran|corrected --landuse-table NAME \
--frac-seaice yes|no --tsk-seaice-threshold K \n--oned-underflow preserve|reproduce-ifx-ftz [--receipt JSON] [--gocart FILE] [--gocartlevels N]";

fn usage() -> String {
    format!(
        "usage: {ABI_MARKER}\n\nSurface updates: {}\n\n\
         --sfc-met MET         optional SST and sea-ice input (config_input_sst)\n\
         --seaice-met MET      optional auxiliary SEAICE_FRACTIONAL input\n\
         --omld-met MET        optional auxiliary OMLD input\n\
         --extrap-airtemp     constant | linear | lapse-rate  (config_extrap_airtemp)\n\
         --use-spechumd       yes | no                        (config_use_spechumd)\n\
         --virtual-factor     reproduce-fortran keeps the (rvord-1)/1.61 mix the Fortran has;\n\
        \x20                    consistent uses rvord-1 throughout so the theta round trip closes\n\
         --deep-soil-moisture reproduce-fortran keeps sm_input(nFGSoilLevels), which is the\n\
        \x20                    second-deepest first-guess level; corrected uses the deepest\n\n\
         --oned-underflow     preserve keeps the subnormal 1e-20*1e-20 inside oned's guards,
\n                             which is what the source says and what ifx -O0, ifx -no-ftz and
\n                             gfortran -O3 all produce; reproduce-ifx-ftz flushes it, which is
\n                             what the reference build (mpiifx -O3, -ftz on by default) does

\n         None of these has a default.  Each changes the numbers in a file that\n\
         opens cleanly and reads plausibly either way.\n\n\
         --land-surface       noah | noahmp | ruc (optional, default noah); ruc writes the\n\
        \x20                    nine RUC soil nodes from the source's nodes or layers\n\
         --first-guess-hydrometeors yes | no (optional, default no): yes reads the first\n\
        \x20                    guess's QC QR QI QS QG and QNC QNI QNR into qc qr qi qs qg nc ni nr\n\
        \x20                    (numbers it lacks are made from the mass); no is MPAS-A v8.4.2\n\
        \x20                    case 7, which starts every hydrometeor at zero\n\n\
         Physics completion of an existing init (each mode stands alone):\n\
        \x20 --augment-init INIT.nc --scalar-names a,b [--aerosol-fields DIR]\n\
        \x20                    [--land-surface noah|noahmp|ruc]\n\
        \x20 --aerosol-status INIT.nc\n\
        \x20 --aerosol-target INIT.nc --target-dir DIR\n\
        \x20 DIR holds little-endian float32 files, cell-major.",
        rw_mpas::init::surface::SURFACE_ABI
    )
}

struct Args {
    map: std::collections::BTreeMap<String, String>,
}

impl Args {
    fn parse(argv: Vec<String>) -> Result<Args, String> {
        let mut map = std::collections::BTreeMap::new();
        let mut it = argv.into_iter();
        while let Some(token) = it.next() {
            if !token.starts_with("--") {
                return Err(format!("unexpected argument \"{token}\"\n\n{}", usage()));
            }
            let key = token.trim_start_matches("--").to_string();
            let value = it
                .next()
                .ok_or_else(|| format!("--{key} needs a value\n\n{}", usage()))?;
            map.insert(key, value);
        }
        Ok(Args { map })
    }

    fn need(&self, key: &str) -> Result<&str, String> {
        self.map.get(key).map(String::as_str).ok_or_else(|| {
            format!(
                "--{key} was not given, and it has no default\n\n{}",
                usage()
            )
        })
    }

    /// An optional switch, keyed WITHOUT the leading dashes like `need`.
    fn get_optional(&self, key: &str) -> Option<String> {
        self.map
            .get(key.trim_start_matches("--"))
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
    }

    fn path(&self, key: &str) -> Result<PathBuf, String> {
        Ok(PathBuf::from(self.need(key)?))
    }

    fn number<T: std::str::FromStr>(&self, key: &str) -> Result<T, String> {
        self.need(key)?
            .parse::<T>()
            .map_err(|_| format!("--{key} is not a number"))
    }
}

fn yes_no(key: &str, value: &str) -> Result<bool, String> {
    match value {
        "yes" | "true" | "YES" | "T" => Ok(true),
        "no" | "false" | "NO" | "F" => Ok(false),
        other => Err(format!("--{key} takes yes or no, not \"{other}\"")),
    }
}

fn run() -> Result<String, String> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    if argv.is_empty() || argv.iter().any(|a| a == "--help" || a == "-h") {
        return Err(usage());
    }
    // The release resolver probes this contract before supplying inputs.
    if argv.iter().any(|argument| argument == "--abi") {
        if argv
            .windows(2)
            .any(|pair| pair[0] == "--mode" && pair[1] == "surface-update")
        {
            return Ok(rw_mpas::init::surface::SURFACE_ABI.to_string());
        }
        return Ok(ABI_MARKER.to_string());
    }
    let args = Args::parse(argv)?;
    if let Some(path) = args.get_optional("aerosol-status") {
        let supplied = rw_mpas::init::aerosol::analyzed_pair(PathBuf::from(&path).as_path()).map_err(|e|e.to_string())?;
        let mismatch = rw_mpas::init::aerosol::soil_category_mismatches(PathBuf::from(path).as_path()).map_err(|e|e.to_string())?;
        return Ok(serde_json::json!({"analysis_pair":supplied,"soil_category_mismatches":mismatch}).to_string());
    }
    if let Some(path) = args.get_optional("aerosol-target") {
        let dir = args.path("target-dir")?;
        let target = rw_mpas::init::aerosol::target(PathBuf::from(path).as_path(), &dir).map_err(|e|e.to_string())?;
        return serde_json::to_string(&target).map_err(|e|e.to_string());
    }

    if let Some(path) = args.get_optional("augment-init") {
        let names: Vec<String> = args.need("scalar-names")?.split(',').map(str::to_string).collect();
        let fields = args.get_optional("aerosol-fields").map(PathBuf::from);
        let land = args.get_optional("land-surface").unwrap_or_else(||"noah".into());
        if !["noah", "noahmp", "ruc"].contains(&land.as_str()) { return Err("--land-surface takes noah, noahmp or ruc".into()); }
        let receipt = rw_mpas::init::aerosol::enrich(PathBuf::from(path).as_path(), &names, fields.as_deref(), land != "noah").map_err(|e| e.to_string())?;
        return serde_json::to_string_pretty(&receipt).map_err(|e| e.to_string());
    }
    if let Some(mode) = args.get_optional("mode") {
        if mode != "surface-update" {
            return Err(format!(
                "unknown init mode {mode:?}; expected surface-update"
            ));
        }
        let cfg = rw_mpas::init::surface::SurfaceConfig {
            sfc_prefix: args.path("sfc-prefix")?,
            static_path: args.path("static")?,
            out_path: args.path("out")?,
            start_time: args.need("start-time")?.to_string(),
            stop_time: args.need("stop-time")?.to_string(),
            fg_interval: args.number("fg-interval")?,
            surface_interval: args.number("surface-interval")?,
        };
        let receipt = rw_mpas::init::surface::build_surface(&cfg).map_err(|e| e.to_string())?;
        let json = serde_json::to_string_pretty(&receipt).map_err(|e| e.to_string())?;
        if let Some(path) = args.get_optional("receipt") {
            std::fs::write(&path, &json)
                .map_err(|e| format!("cannot write receipt {path}: {e}"))?;
        }
        return Ok(json);
    }

    let land_surface = args.get_optional("land-surface").unwrap_or_else(|| "noah".into());
    if !["noah", "noahmp", "ruc"].contains(&land_surface.as_str()) { return Err("--land-surface takes noah, noahmp or ruc".into()); }
    let cfg = InitConfig {
        met_path: args.path("met")?,
        gocart_path: args.get_optional("gocart").map(PathBuf::from),
        gocart_levels: args
            .get_optional("gocartlevels")
            .map(|s| {
                s.parse::<usize>()
                    .map_err(|_| "--gocartlevels is not a nonnegative integer".to_string())
            })
            .transpose()?,
        seaice_met_path:args.get_optional("seaice-met").map(PathBuf::from),
        omld_met_path:args.get_optional("omld-met").map(PathBuf::from),
        sfc_met_path:args.get_optional("sfc-met").map(PathBuf::from),
        static_path: args.path("static")?,
        capsule_path: args.path("capsule")?,
        reference_path: args.path("reference")?,
        out_path: args.path("out")?,
        start_time: args.need("start-time")?.to_string(),
        n_fg_levels: args.number("nfglevels")?,
        n_fg_soil_levels: args.number("nfgsoillevels")?,
        land_surface,
        extrap_airtemp: Extrap::parse(args.need("extrap-airtemp")?)?,
        use_spechumd: yes_no("use-spechumd", args.need("use-spechumd")?)?,
        theta_adv_order: args.number("theta-adv-order")?,
        coef_3rd_order: args.number("coef-3rd-order")?,
        virtual_factor: match args.need("virtual-factor")? {
            "reproduce-fortran" => VirtualFactor::ReproduceFortran,
            "consistent" => VirtualFactor::Consistent,
            other => {
                return Err(format!(
                    "--virtual-factor takes reproduce-fortran or consistent, not \"{other}\""
                ))
            }
        },
        deep_moisture: match args.need("deep-soil-moisture")? {
            "reproduce-fortran" => DeepMoisture::ReproduceFortran,
            "corrected" => DeepMoisture::Corrected,
            other => {
                return Err(format!(
                    "--deep-soil-moisture takes reproduce-fortran or corrected, not \"{other}\""
                ))
            }
        },
        landuse_table: args.need("landuse-table")?.to_string(),
        frac_seaice: yes_no("frac-seaice", args.need("frac-seaice")?)?,
        tsk_seaice_threshold: args.number("tsk-seaice-threshold")?,
        oned_underflow: Underflow::parse(args.need("oned-underflow")?)?,
        // Optional, and native (no) when absent, so every native-identity
        // caller keeps MPAS-A v8.4.2 case 7.  The hex init door passes yes
        // unless its numerics are strict.
        first_guess_hydrometeors: match args.get_optional("first-guess-hydrometeors") {
            None => false,
            Some(value) => yes_no("first-guess-hydrometeors", &value)?,
        },
        lineage: Lineage {
            // Optional, and defaulted to this engine's own identity, so a
            // caller that says nothing still writes an init rw_mpas_lbc will
            // accept as a --grid.  A caller that knows which model will
            // integrate the file says so and the file carries that name.
            model_name: args
                .get_optional("--model-name")
                .unwrap_or_else(|| Lineage::default().model_name),
            core_name: args
                .get_optional("--core-name")
                .unwrap_or_else(|| Lineage::default().core_name),
            version: args
                .get_optional("--model-version")
                .unwrap_or_else(|| Lineage::default().version),
            git_version: args
                .get_optional("--git-version")
                .unwrap_or_else(|| Lineage::default().git_version),
        },
        provenance: format!(
            "rw_mpas_init from {}",
            std::env::current_exe()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|_| "an unnamed tree".to_string())
        ),
    };

    let receipt = build_init(&cfg).map_err(|e| e.to_string())?;
    let json = serde_json::to_string_pretty(&receipt).map_err(|e| e.to_string())?;
    if let Some(path) = args.map.get("receipt") {
        std::fs::write(path, &json).map_err(|e| format!("cannot write {path}: {e}"))?;
    }
    Ok(json)
}

fn main() -> ExitCode {
    let _ = std::hint::black_box(GPUWM_BRIDGE_SOURCE_REV_STAMP);
    match run() {
        Ok(json) => {
            println!("{json}");
            ExitCode::SUCCESS
        }
        Err(message) => {
            eprintln!("rw_mpas_init: {message}");
            ExitCode::FAILURE
        }
    }
}
