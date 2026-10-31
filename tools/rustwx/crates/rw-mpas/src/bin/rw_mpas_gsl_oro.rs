//! GSL generator command. MPAS sources and driver boundary in gsl_oro.rs.
use rw_mpas::gsl_oro::{self, BASE_NAMES, Real, Tiles};
use rw_store::netcdf_classic::{
    NcAttr, NcClassicWriter, NcData, NcDim, NcFormat, NcType, NcVarDef,
};
use std::path::{Path, PathBuf};

const USAGE: &str = "rw_mpas_gsl_oro --grid MESH.nc --geog WPS_GEOG --out ORO.nc [--cell-scaling FACTOR] [--sphere-radius METERS] [--precision single|double] [--receipt JSON] [--clobber]\n       rw_mpas_gsl_oro --compare NATIVE.nc GENERATED.nc [--precision single|double] [--receipt JSON]";
pub static GPUWM_BRIDGE_SOURCE_REV_STAMP: &str =
    concat!("GPUWM_BRIDGE_SOURCE_REV=", env!("GPUWM_BRIDGE_SOURCE_REV"));

#[derive(Default)]
struct Args {
    grid: Option<PathBuf>,
    geog: Option<PathBuf>,
    out: Option<PathBuf>,
    receipt: Option<PathBuf>,
    scaling: Option<f64>,
    radius: Option<f64>,
    double: bool,
    clobber: bool,
    compare: Option<(PathBuf, PathBuf)>,
}

fn parse() -> Result<Option<Args>, String> {
    let mut argv = std::env::args().skip(1);
    let mut a = Args::default();
    while let Some(arg) = argv.next() {
        if arg == "--help" || arg == "-h" {
            println!("{USAGE}\n{GPUWM_BRIDGE_SOURCE_REV_STAMP}");
            return Ok(None);
        }
        if arg == "--clobber" {
            a.clobber = true;
            continue;
        }
        let value = argv
            .next()
            .ok_or_else(|| format!("{arg} requires a value\n{USAGE}"))?;
        match arg.as_str() {
            "--grid" => a.grid = Some(value.into()),
            "--geog" => a.geog = Some(value.into()),
            "--out" => a.out = Some(value.into()),
            "--receipt" => a.receipt = Some(value.into()),
            "--cell-scaling" => {
                a.scaling = Some(value.parse().map_err(|_| "invalid --cell-scaling")?)
            }
            "--sphere-radius" => {
                a.radius = Some(value.parse().map_err(|_| "invalid --sphere-radius")?)
            }
            "--precision" => {
                a.double = match value.as_str() {
                    "single" => false,
                    "double" => true,
                    _ => return Err("precision must be single or double".into()),
                }
            }
            "--compare" => {
                a.compare = Some((
                    value.into(),
                    argv.next().ok_or("--compare requires two files")?.into(),
                ))
            }
            _ => return Err(format!("unknown option {arg}\n{USAGE}")),
        }
    }
    if a.compare.is_some()
        && (a.grid.is_some()
            || a.geog.is_some()
            || a.out.is_some()
            || a.scaling.is_some()
            || a.radius.is_some()
            || a.clobber)
    {
        return Err("--compare cannot be combined with generation options".into());
    }
    Ok(Some(a))
}

fn read<T: Real>(file: &netcrust::File, name: &str) -> Result<Vec<T>, String> {
    Ok(file
        .read_f64(name)
        .map_err(|e| format!("{name}: {e}"))?
        .into_iter()
        .map(T::from_f64)
        .collect())
}
fn indices(file: &netcrust::File, name: &str) -> Result<Vec<usize>, String> {
    file.read_f64(name)
        .map_err(|e| format!("{name}: {e}"))?
        .into_iter()
        .map(|v| {
            if !v.is_finite() || v < 0. || v.fract() != 0. || v > usize::MAX as f64 {
                Err(format!("{name} contains an invalid MPAS index"))
            } else {
                Ok(v as usize)
            }
        })
        .collect()
}

fn generate<T: Real + Send + Sync>(a: &Args) -> Result<serde_json::Value, String> {
    let grid = a.grid.as_ref().ok_or("--grid is required")?;
    let geog = a.geog.as_ref().ok_or("--geog is required")?;
    let out = a.out.as_ref().ok_or("--out is required")?;
    if out.exists() && !a.clobber {
        return Err(format!(
            "{} exists; --clobber is required to replace its geography",
            out.display()
        ));
    }
    if out.exists() && std::fs::canonicalize(out).ok() == std::fs::canonicalize(grid).ok() {
        return Err("output cannot replace its input mesh".into());
    }
    if a.receipt.as_ref() == Some(out) || a.receipt.as_ref() == Some(grid) {
        return Err("receipt must not overwrite mesh or output".into());
    }
    let mesh = netcrust::File::open(grid).map_err(|e| e.to_string())?;
    let nc = mesh.dimension("nCells").ok_or("mesh has no nCells")?.len();
    let ne = mesh.dimension("nEdges").ok_or("mesh has no nEdges")?.len();
    let me = mesh
        .dimension("maxEdges")
        .ok_or("mesh has no maxEdges")?
        .len();
    for (name, shape) in [
        ("latCell", vec![nc]),
        ("lonCell", vec![nc]),
        ("dcEdge", vec![ne]),
        ("nEdgesOnCell", vec![nc]),
        ("edgesOnCell", vec![nc, me]),
    ] {
        if mesh
            .variable(name)
            .ok_or_else(|| format!("missing {name}"))?
            .shape()
            != shape
        {
            return Err(format!(
                "{name} dimensions do not match the MPAS mesh; cannot pair cell boxes with terrain"
            ));
        }
    }
    let radius = a.radius.unwrap_or(6371229.);
    let lat = read::<T>(&mesh, "latCell")?;
    let lon = read::<T>(&mesh, "lonCell")?;
    let dc = read::<T>(&mesh, "dcEdge")?;
    let areas = gsl_oro::cell_areas(
        &indices(&mesh, "nEdgesOnCell")?,
        &indices(&mesh, "edgesOnCell")?,
        me,
        &dc,
        T::from_f64(radius),
        T::from_f64(a.scaling.unwrap_or(1.)),
    )?;
    let mut small = Tiles::open(geog, true)?;
    let mut large = Tiles::open(geog, false)?;
    let native_read_errors: Vec<_> = [&small, &large].iter().filter_map(|tile| {
        use gsl_oro::Topography;
        tile.read_error().map(str::to_owned)
    }).collect();
    let fields = gsl_oro::generate_parallel(&lat, &lon, &areas,
        &|| Tiles::open(geog, true), &|| Tiles::open(geog, false))?;
    if let Some(parent) = out.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let tmp = out.with_file_name(format!(
        ".{}.{}.partial",
        out.file_name()
            .ok_or("output must name a file")?
            .to_string_lossy(),
        std::process::id()
    ));
    // Owned scratch is removed on write failure. A complete field file is
    // published only after all slabs have been flushed by the Rust writer.
    let write = || -> Result<(), String> {
        let ty = if a.double {
            NcType::Double
        } else {
            NcType::Float
        };
        let mut vars = Vec::new();
        for suffix in ["ls", "ss"] {
            for name in BASE_NAMES {
                vars.push(NcVarDef::new(format!("{name}{suffix}"), ty, vec![0]));
            }
        }
        let mut writer = NcClassicWriter::create(
            &tmp,
            NcFormat::Offset64,
            vec![NcDim::fixed("nCells", nc)],
            vec![
                NcAttr::text("rw_gsl_source", "MPAS-A v8.4.2"),
                NcAttr::text("rw_source_revision", env!("GPUWM_BRIDGE_SOURCE_REV")),
            ],
            vars,
            0,
        )
        .map_err(|e| e.to_string())?;
        for (suffix, data) in [("ls", &fields.large), ("ss", &fields.small)] {
            for (k, name) in BASE_NAMES.iter().enumerate() {
                let name = format!("{name}{suffix}");
                if a.double {
                    let values: Vec<f64> = data.iter().map(|c| c[k].to_f64()).collect();
                    writer
                        .put(&name, NcData::Doubles(&values))
                        .map_err(|e| e.to_string())?;
                } else {
                    let values: Vec<f32> = data.iter().map(|c| c[k].to_f64() as f32).collect();
                    writer
                        .put(&name, NcData::Floats(&values))
                        .map_err(|e| e.to_string())?;
                }
            }
        }
        writer.finish().map_err(|e| e.to_string())
    };
    if let Err(e) = write() {
        let _ = std::fs::remove_file(&tmp);
        return Err(e);
    }
    if a.clobber {
        std::fs::rename(&tmp, out).map_err(|e| e.to_string())?;
    } else {
        std::fs::hard_link(&tmp, out).map_err(|e| e.to_string())?;
        std::fs::remove_file(&tmp).map_err(|e| e.to_string())?;
    }
    Ok(
        serde_json::json!({"schema":"rw-mpas-gsl-oro/v1","n_cells":nc,"fields":20,
        "rkind_bytes":if a.double {8} else {4},"source_revision":env!("GPUWM_BRIDGE_SOURCE_REV"),
        "duplicate_cells":fields.duplicate.iter().filter(|x|**x).count(),"sphere_radius":radius,
        "native_read_errors":native_read_errors,
        "cell_scaling":a.scaling.unwrap_or(1.),"grid_sha256":rw_mpas::sha256_file(grid).map_err(|e|e.to_string())?,
        "output_sha256":rw_mpas::sha256_file(out).map_err(|e|e.to_string())?,"identity_tier":"not compared"}),
    )
}

fn compare(a: &Args, reference: &Path, candidate: &Path) -> Result<serde_json::Value, String> {
    let left = netcrust::File::open(reference).map_err(|e| e.to_string())?;
    let right = netcrust::File::open(candidate).map_err(|e| e.to_string())?;
    let mut fields = Vec::new();
    let mut words = 0;
    let mut unequal = 0;
    for suffix in ["ls", "ss"] {
        for base in BASE_NAMES {
            let name = format!("{base}{suffix}");
            let l = left
                .variable(&name)
                .ok_or_else(|| format!("missing {name} in reference"))?;
            let rr = right
                .variable(&name)
                .ok_or_else(|| format!("missing {name} in candidate"))?;
            if l.shape() != rr.shape() {
                return Err(format!("{name} shape mismatch"));
            }
            let expected = if a.double {
                netcrust::DataType::F64
            } else {
                netcrust::DataType::F32
            };
            if *l.dtype() != expected || *rr.dtype() != expected {
                return Err(format!(
                    "{name} type does not match --precision; conversion would hide unequal words"
                ));
            }
            // Typed reads preserve every stored word, including signaling
            // NaN bits; promotion to f64 and back can quiet a binary32 NaN.
            let (field_words, mismatch) = if a.double {
                let x = left.read_array::<f64>(&name).map_err(|e| e.to_string())?;
                let y = right.read_array::<f64>(&name).map_err(|e| e.to_string())?;
                (x.len(), x.iter().zip(y.iter()).filter(|(x, y)| x.to_bits() != y.to_bits()).count())
            } else {
                let x = left.read_array::<f32>(&name).map_err(|e| e.to_string())?;
                let y = right.read_array::<f32>(&name).map_err(|e| e.to_string())?;
                (x.len(), x.iter().zip(y.iter()).filter(|(x, y)| x.to_bits() != y.to_bits()).count())
            };
            words += field_words;
            unequal += mismatch;
            fields.push(serde_json::json!({"field":name,"words":field_words,"unequal_words":mismatch}));
        }
    }
    Ok(
        serde_json::json!({"schema":"rw-mpas-gsl-oro-compare/v1","rkind_bytes":if a.double {8} else {4},
        "words":words,"unequal_words":unequal,"fields":fields,"pass":unequal==0,
        "reference_sha256":rw_mpas::sha256_file(reference).map_err(|e|e.to_string())?,
        "candidate_sha256":rw_mpas::sha256_file(candidate).map_err(|e|e.to_string())?}),
    )
}

fn run() -> Result<(), String> {
    let Some(a) = parse()? else {
        return Ok(());
    };
    if let Some(receipt) = &a.receipt {
        let inputs: Vec<&Path> = if let Some((left, right)) = &a.compare {
            vec![left, right]
        } else {
            a.grid
                .iter()
                .chain(a.out.iter())
                .map(PathBuf::as_path)
                .collect()
        };
        for input in inputs {
            if receipt == input
                || (receipt.exists()
                    && std::fs::canonicalize(receipt).ok() == std::fs::canonicalize(input).ok())
            {
                return Err(
                    "receipt must not overwrite a mesh, geography output or comparison input"
                        .into(),
                );
            }
        }
    }
    let receipt = if let Some((left, right)) = &a.compare {
        compare(&a, left, right)?
    } else if a.double {
        generate::<f64>(&a)?
    } else {
        generate::<f32>(&a)?
    };
    let json = serde_json::to_string_pretty(&receipt).map_err(|e| e.to_string())?;
    if let Some(path) = &a.receipt {
        std::fs::write(path, format!("{json}\n")).map_err(|e| e.to_string())?;
    }
    println!("{json}");
    if receipt.get("pass") == Some(&serde_json::Value::Bool(false)) {
        return Err("GSL words differ from the reference".into());
    }
    Ok(())
}
fn main() -> std::process::ExitCode {
    match run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{e}");
            std::process::ExitCode::FAILURE
        }
    }
}
