//! Reading one MPAS history frame, plus the pinned static geometry it needs.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::error::{MpasError, MpasResult};
use crate::fieldmap::{
    FIELD_MAP, FieldMapping, FieldSource, field_set_allows, field_set_is_known, native_history_name,
};
use crate::weights::MeshCoordinates;
use rw_store::netcdf_classic::{NcAttr, NcAttrValue};

/// Physics identity is copied only when the native history actually declares it.
/// A renderer conversion cannot infer a microphysics scheme from field names.
const SCIENCE_ATTRIBUTES: &[&str] = &[
    "MP_PHYSICS",
    "WOOF_MICROPHYSICS_SCHEME",
    "WOOF_ENGINE_MICROPHYSICS_SCHEME",
    "WOOF_PHYSICS_BACKEND",
    "WOOF_SCALAR_NAMES",
    "RADAR_MIXING_RATIO_BASIS",
    "MORR_RIMED_ICE",
    "morr_rimed_ice",
];

fn read_science_attributes(file: &netcrust::File) -> MpasResult<Vec<NcAttr>> {
    use netcrust::AttributeValue as Value;
    let mut attributes = Vec::new();
    for name in SCIENCE_ATTRIBUTES {
        let Some(attribute) = file.attribute(name) else {
            continue;
        };
        let unsupported = || {
            MpasError::Refusal(format!(
                "history physics attribute {name} cannot be represented exactly in classic netCDF"
            ))
        };
        let value = match attribute.value() {
            Value::Chars(value) => NcAttrValue::Text(value.clone()),
            Value::Strings(values) if values.len() == 1 => NcAttrValue::Text(values[0].clone()),
            Value::Bytes(values) => {
                NcAttrValue::Ints(values.iter().map(|&v| i32::from(v)).collect())
            }
            Value::Shorts(values) => {
                NcAttrValue::Ints(values.iter().map(|&v| i32::from(v)).collect())
            }
            Value::Ints(values) => NcAttrValue::Ints(values.clone()),
            Value::UBytes(values) => {
                NcAttrValue::Ints(values.iter().map(|&v| i32::from(v)).collect())
            }
            Value::UShorts(values) => {
                NcAttrValue::Ints(values.iter().map(|&v| i32::from(v)).collect())
            }
            Value::UInts(values) => NcAttrValue::Ints(
                values
                    .iter()
                    .map(|&v| i32::try_from(v))
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|_| unsupported())?,
            ),
            Value::Int64s(values) => NcAttrValue::Ints(
                values
                    .iter()
                    .map(|&v| i32::try_from(v))
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|_| unsupported())?,
            ),
            Value::UInt64s(values) => NcAttrValue::Ints(
                values
                    .iter()
                    .map(|&v| i32::try_from(v))
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|_| unsupported())?,
            ),
            Value::Floats(values) => NcAttrValue::Floats(values.clone()),
            Value::Doubles(values) => NcAttrValue::Doubles(values.clone()),
            _ => return Err(unsupported()),
        };
        let canonical = if *name == "morr_rimed_ice" {
            "MORR_RIMED_ICE"
        } else {
            name
        };
        let mapped = NcAttr {
            name: canonical.to_string(),
            value,
        };
        if let Some(previous) = attributes
            .iter()
            .find(|attr: &&NcAttr| attr.name == mapped.name)
        {
            if previous != &mapped {
                return Err(MpasError::Refusal(format!(
                    "history has conflicting values for physics attribute {}",
                    mapped.name
                )));
            }
        } else {
            attributes.push(mapped);
        }
    }
    Ok(attributes)
}

/// A calendar instant, to the second. The frame carries one and stamps it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Timestamp {
    pub year: i32,
    pub month: u32,
    pub day: u32,
    pub hour: u32,
    pub minute: u32,
    pub second: u32,
}

impl Timestamp {
    /// Parse the `YYYY-MM-DD_HH.MM.SS` (or `:`, or `T`) stamp MPAS puts in a
    /// history file name.
    pub fn parse(text: &str) -> MpasResult<Timestamp> {
        let bytes: Vec<char> = text.chars().collect();
        for start in 0..bytes.len().saturating_sub(18) {
            let slice: String = bytes[start..start + 19].iter().collect();
            if let Some(stamp) = Timestamp::parse_exact(&slice) {
                return Ok(stamp);
            }
        }
        Err(MpasError::Refusal(format!(
            "no YYYY-MM-DD_HH.MM.SS timestamp in '{text}'"
        )))
    }

    fn parse_exact(s: &str) -> Option<Timestamp> {
        let b: Vec<char> = s.chars().collect();
        if b.len() != 19 {
            return None;
        }
        let digits = |from: usize, n: usize| -> Option<u32> {
            let mut value = 0u32;
            for c in &b[from..from + n] {
                value = value * 10 + c.to_digit(10)?;
            }
            Some(value)
        };
        if b[4] != '-' || b[7] != '-' {
            return None;
        }
        if !matches!(b[10], '_' | 'T') {
            return None;
        }
        if !matches!(b[13], '.' | ':') || !matches!(b[16], '.' | ':') {
            return None;
        }
        Some(Timestamp {
            year: digits(0, 4)? as i32,
            month: digits(5, 2)?,
            day: digits(8, 2)?,
            hour: digits(11, 2)?,
            minute: digits(14, 2)?,
            second: digits(17, 2)?,
        })
    }

    /// The 19-byte WRF label, which is what `Times` and the date attributes
    /// carry.
    pub fn label(&self) -> String {
        format!(
            "{:04}-{:02}-{:02}_{:02}:{:02}:{:02}",
            self.year, self.month, self.day, self.hour, self.minute, self.second
        )
    }

    pub fn iso(&self) -> String {
        format!(
            "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}",
            self.year, self.month, self.day, self.hour, self.minute, self.second
        )
    }

    /// Days since the proleptic-Gregorian epoch, for a difference in seconds.
    fn day_number(&self) -> i64 {
        let (y, m) = if self.month <= 2 {
            (self.year as i64 - 1, self.month as i64 + 12)
        } else {
            (self.year as i64, self.month as i64)
        };
        let era = if y >= 0 { y } else { y - 399 } / 400;
        let yoe = y - era * 400;
        let doy = (153 * (m - 3) + 2) / 5 + self.day as i64 - 1;
        let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
        era * 146_097 + doe - 719_468
    }

    pub fn seconds_since(&self, other: &Timestamp) -> i64 {
        let days = self.day_number() - other.day_number();
        let secs = (self.hour as i64 * 3600 + self.minute as i64 * 60 + self.second as i64)
            - (other.hour as i64 * 3600 + other.minute as i64 * 60 + other.second as i64);
        days * 86_400 + secs
    }
}

/// One MPAS history frame plus the pinned static geometry it needs.
#[derive(Debug)]
pub struct MpasFrame {
    pub history_path: PathBuf,
    pub history_sha256: String,
    pub init_path: Option<PathBuf>,
    pub init_sha256: Option<String>,
    pub valid_time: Timestamp,
    pub n_cells: usize,
    pub n_levels: usize,
    pub n_soil_levels: usize,
    /// WRF name -> (values as `(n_cells, levels)` row-major, levels).
    pub fields: BTreeMap<String, (Vec<f32>, usize)>,
    pub latitude_degrees: Vec<f64>,
    pub longitude_degrees: Vec<f64>,
    pub absent: Vec<String>,
    /// Exact declared physics values, absent when the history did not record them.
    pub science_attributes: Vec<NcAttr>,
    /// Set only for a split-stream history (one that carries no cell
    /// centres, as RRFS-MPAS writes it): where the centres and the diag
    /// fields came from.  `None` for every history that carries its own
    /// centres, which are read exactly as before.
    pub split: Option<SplitSources>,
    /// Set only for a history that carries its own centres AND lacked a
    /// mapped field that its `diag` stream file of the same time carried:
    /// that file and the WRF names read from it.  `None` whenever the
    /// history alone answered every field, so such a frame converts to the
    /// same bytes as before.
    pub companion_diag: Option<DiagCompanion>,
}

/// The `diag` stream file a history that carries its own cell centres was
/// completed from.
///
/// THE BREAKAGE THIS PREVENTS (case study 2025-05-19, config H2, 2026-10-24):
/// MPAS-A's own `output` stream, which the hex MPAS-physics route writes as
/// `history.*.nc`, carries `latCell`/`lonCell` but no reflectivity; MPAS writes
/// `refl10cm_max` (and its 2 m, 10 m and precipitation diagnostics) to the
/// `diagnostics` stream, `diag.*.nc`.  The converter read the diag file only
/// for a split-stream history, so every export, column store and map of that
/// route had no model reflectivity (the case sheets' H2 radar panels were
/// empty and its reflectivity scores read 0).
#[derive(Debug, Clone)]
pub struct DiagCompanion {
    pub path: PathBuf,
    pub sha256: Option<String>,
    /// WRF names whose values came from the diag file.
    pub fields: Vec<String>,
}

/// Where a split-stream frame's cell centres and diag fields were read from.
///
/// RRFS-MPAS (the NOAA fork's `streams.atmosphere`) writes a `history`
/// stream with no mesh variables and puts its 2 m, 10 m, precipitation and
/// reflectivity fields in a separate `diag` stream.  MPAS-A's own `output`
/// stream carries `latCell`/`lonCell` and those fields, so it never takes
/// this path.
#[derive(Debug, Clone)]
pub struct SplitSources {
    /// The mesh file the cell centres were taken from.
    pub coordinates_path: PathBuf,
    pub coordinates_sha256: String,
    /// The static fields the history repeats that were checked equal to the
    /// mesh file's, cell for cell (empty when the history repeats none).
    pub agreed_static_fields: Vec<String>,
    /// The `diag` stream file at the history's own time, when it exists.
    pub diag_path: Option<PathBuf>,
    pub diag_sha256: Option<String>,
    /// WRF names whose values came from the diag file.
    pub diag_fields: Vec<String>,
}

/// Static fields MPAS writes unchanged from the static file into a history
/// that repeats them (the RRFS fork's history does).  They tie a history
/// with no cell centres to its mesh: a different mesh with the same cell
/// count (a `grid_rotate` copy of it, for one) holds them at other cells.
/// `ter` is NOT one of them: init_atmosphere smooths it
/// (`config_smooth_surfaces`), measured 1,145 m apart on x1.2562.
const MESH_IDENTITY_FIELDS: &[&str] = &["shdmin", "shdmax"];

/// The `diag` stream file written at the same time as `history_path`: its
/// name with the last `history` token replaced by `diag`, in the same
/// directory (`history.2023-03-31_13.00.00.nc` -> `diag.2023-03-31_13.00.00.nc`,
/// the fork's and MPAS-A's own templates).
pub fn diag_companion(history_path: &Path) -> Option<PathBuf> {
    let name = history_path.file_name()?.to_str()?;
    let at = name.rfind("history")?;
    let diag = format!("{}diag{}", &name[..at], &name[at + "history".len()..]);
    Some(history_path.with_file_name(diag))
}

/// Read one cell-indexed field, dropping a leading single `Time` record.
fn read_cell_field(
    file: &netcrust::File,
    name: &str,
    n_cells: usize,
) -> MpasResult<Option<(Vec<f32>, usize)>> {
    let Some(variable) = file.variable(name) else {
        return Ok(None);
    };
    let has_time = variable
        .dimensions()
        .first()
        .map(|d| d.name() == "Time")
        .unwrap_or(false);
    let shape: Vec<usize> = variable.shape();
    if has_time && shape.first() != Some(&1) {
        return Err(MpasError::Refusal(format!(
            "{name} carries {} time records; expected exactly 1",
            shape.first().copied().unwrap_or(0)
        )));
    }
    let array = file.read_array_f64(name)?;
    let values = array.into_values();
    let trailing: usize = if has_time {
        shape[1..].iter().product()
    } else {
        shape.iter().product()
    };
    if values.len() < trailing {
        return Err(MpasError::Refusal(format!(
            "{name} read back {} value(s), fewer than the {trailing} its shape declares",
            values.len()
        )));
    }
    let slice = &values[values.len() - trailing..];
    if slice.len() % n_cells != 0 {
        return Err(MpasError::Refusal(format!(
            "{name} has {} value(s), not a whole number of levels over {n_cells} cells",
            slice.len()
        )));
    }
    let levels = slice.len() / n_cells;
    let mut out = Vec::with_capacity(slice.len());
    for &v in slice {
        let f = v as f32;
        if !f.is_finite() {
            return Err(MpasError::Refusal(format!("history {name} is not finite")));
        }
        out.push(f);
    }
    Ok(Some((out, levels)))
}

/// The init-sourced fields of a pinned init file, read (and hashed) once.
///
/// Every frame of a run shares them, so a converter that draws many frames
/// reads them here once and hands them to [`read_history_with_init`].
///
/// THE BREAKAGE THIS PREVENTS: [`read_history`] with an init hashed and read
/// the whole init for every frame.  On the 1,604,031-cell 20 -> 2.5 km global
/// case (a hex run, 2026-10-13) one product view's `rw_mpas_convert` read
/// 419 GB in 12 minutes for seven frames, 15 GB of init per frame twice over,
/// and the product stage outlasted the forecast.
#[derive(Debug, Clone)]
pub struct InitFields {
    pub path: PathBuf,
    pub sha256: String,
    pub n_cells: usize,
    /// WRF name -> (values, levels), as in [`MpasFrame::fields`].
    pub fields: BTreeMap<String, (Vec<f32>, usize)>,
    /// Init-sourced WRF names the file does not carry.
    pub absent: Vec<String>,
}

/// Invariant fields a history may omit, read from the run's init only when the
/// history does not carry them (WRF name, MPAS name).
///
/// THE BREAKAGE THIS PREVENTS (gpuwm-hex F-PG4 and F-MPAS-PRODUCTS,
/// 2026-10-24): MPAS-A's own `output` stream writes no `ter`, and the hex
/// MPAS-physics route writes exactly that stream, so every lat-lon frame from
/// it had no `HGT`.  The renderer's wrfout import then dropped everything that
/// needs the ground: MSLP ("mslp unavailable: HGT"), and with it the 2 m
/// temperature and dewpoint maps that contour it, SBCAPE/MUCAPE, the 0 C
/// height, IVT and simulated IR (column planes "read terrain"), precipitation
/// type (height above ground); `gpuwm prep --source hex-global` refused the
/// export (`terrain_height`) and no column store was written.  The init the
/// converter already reads for PHB holds the terrain that run integrated on.
/// A history that carries `ter` (the ArWen-physics route's) keeps its own
/// words, so its frames are unchanged -- unless those words were never filled
/// ([`history_invariant_unset`]).
pub const INIT_FALLBACK_FIELDS: &[(&str, &str)] = &[("HGT", "ter")];

fn is_init_fallback(wrf_name: &str) -> bool {
    INIT_FALLBACK_FIELDS.iter().any(|(wrf, _)| *wrf == wrf_name)
}

/// True when a history's copy of an init-fallback invariant was written but
/// never filled: every word zero while the run's init carries a nonzero one.
/// The init's words then stand in, as for a history that omits the field.
///
/// THE BREAKAGE THIS PREVENTS (RRFS-MPAS hrrrv5 maps, 2026-10-28): the RRFS
/// fork's history stream writes `ter`, and the run never fills it, so every
/// word is 0.0 -- native MPAS-Model-rrfs 1d6e370e (pinned strict build) and
/// hex write the same zeros (lam7k, 9,957 cells: init `ter` 134.7-344.2 m,
/// history `ter` 0.0 everywhere, both models).  The converter kept the
/// history's zeros as `HGT`, so the renderer's MSLP reduction ran from sea
/// level and drew the surface pressure (the mv3km cut: 864-1013 hPa contours
/// following the terrain, lows of 867.5 and 940.8 hPa, while the model's own
/// diag mslp read 989-1023 hPa), and every height-above-ground product read
/// heights off by the terrain.  A domain that is all sea level has an all-zero
/// init `ter` too, so its frames keep the history's words.
fn history_invariant_unset(history: &[f32], init: &[f32]) -> bool {
    history.iter().all(|v| *v == 0.0) && init.iter().any(|v| *v != 0.0)
}

/// Hash `path` (refusing a mismatch against `init_sha256`) and read every
/// init-sourced field of the field map from it, plus the
/// [`INIT_FALLBACK_FIELDS`] it carries.
pub fn read_init_fields(path: &Path, init_sha256: Option<&str>) -> MpasResult<InitFields> {
    // The digest and the field reads are independent passes over the same
    // file: run them side by side, then refuse a digest mismatch first, as
    // the serial order did.
    let (digest, body) = rayon::join(
        || crate::sha256_file(path),
        || -> MpasResult<_> {
            let init = netcrust::File::open(path)?;
            let n_cells = init
                .dimension("nCells")
                .ok_or_else(|| MpasError::Refusal("init has no nCells dimension".to_string()))?
                .len();
            let wanted: Vec<(&'static str, &'static str)> = FIELD_MAP
                .iter()
                .filter(|mapping| mapping.source == FieldSource::Init)
                .filter_map(|mapping| mapping.mpas_name.map(|name| (mapping.wrf_name, name)))
                .chain(INIT_FALLBACK_FIELDS.iter().copied())
                .collect();
            let names: Vec<&str> = wanted.iter().map(|(_, name)| *name).collect();
            let read = read_cell_fields_parallel(&init, &names, n_cells);
            let mut fields = BTreeMap::new();
            let mut absent = Vec::new();
            for ((wrf_name, _), value) in wanted.iter().zip(read) {
                match value? {
                    Some(value) => {
                        fields.insert(wrf_name.to_string(), value);
                    }
                    // A fallback the init lacks is the history's to report.
                    None if is_init_fallback(wrf_name) => {}
                    None => absent.push(wrf_name.to_string()),
                }
            }
            Ok((n_cells, fields, absent))
        },
    );
    let d = digest?;
    if let Some(want) = init_sha256 {
        if d != want {
            return Err(MpasError::Refusal(format!(
                "init {} is sha256 {d}, expected {want}",
                path.display()
            )));
        }
    }
    let (n_cells, fields, absent) = body?;
    Ok(InitFields { path: path.to_path_buf(), sha256: d, n_cells, fields, absent })
}

/// [`read_cell_field`] for several variables of one file at once, one rayon
/// task per variable; results come back in `names` order, so callers insert
/// and report exactly as a serial loop would.
fn read_cell_fields_parallel(
    file: &netcrust::File,
    names: &[&str],
    n_cells: usize,
) -> Vec<MpasResult<Option<(Vec<f32>, usize)>>> {
    use rayon::prelude::*;
    names
        .par_iter()
        .map(|name| read_cell_field(file, name, n_cells))
        .collect()
}

/// Read a history frame and, when given, the pinned init file beside it.
pub fn read_history(
    history_path: &Path,
    init_path: Option<&Path>,
    valid_time: Option<Timestamp>,
    history_sha256: Option<&str>,
    init_sha256: Option<&str>,
) -> MpasResult<MpasFrame> {
    let init = match init_path {
        Some(path) => Some(read_init_fields(path, init_sha256)?),
        None => None,
    };
    read_history_with_init(history_path, init.as_ref(), valid_time, history_sha256)
}

/// [`read_history`] with the init's fields already read by
/// [`read_init_fields`]: the frame gets a copy of them, and the init file is
/// not opened again.
pub fn read_history_with_init(
    history_path: &Path,
    init: Option<&InitFields>,
    valid_time: Option<Timestamp>,
    history_sha256: Option<&str>,
) -> MpasResult<MpasFrame> {
    read_history_on_mesh(history_path, init, valid_time, history_sha256, None)
}

/// The sources of a split-stream history (one with no `latCell`/`lonCell`,
/// as RRFS-MPAS writes it): the mesh that gives its cell centres, after the
/// cell counts and every static field of [`MESH_IDENTITY_FIELDS`] both files
/// carry agree, and its `diag` stream file when one is beside it.
/// `hash_diag` is false for the cell-subset reads of [`read_history_cells_on_mesh`],
/// which hash no file; `tool` names the program in the advisories.
#[allow(clippy::too_many_arguments)]
fn open_split_sources(
    file: &netcrust::File,
    history_path: &Path,
    name: &str,
    mesh: &MeshCoordinates,
    n_cells: usize,
    n_levels: usize,
    hash_diag: bool,
    tool: &str,
) -> MpasResult<(SplitSources, Option<netcrust::File>)> {
    let mut diag: Option<netcrust::File> = None;
    if mesh.n_cells() != n_cells {
        return Err(MpasError::Refusal(format!(
            "history {} has no {name}, so its cell centres are taken from the mesh {}; \
             the mesh has {} cells and the history {n_cells}, so it is not this run's mesh",
            history_path.display(),
            mesh.source_path.display(),
            mesh.n_cells()
        )));
    }
    let mesh_file = netcrust::File::open(&mesh.source_path)?;
    let mut agreed = Vec::new();
    for field in MESH_IDENTITY_FIELDS {
        if file.variable(field).is_none() || mesh_file.variable(field).is_none() {
            continue;
        }
        let ours = file.read_f64(field)?;
        let theirs = mesh_file.read_f64(field)?;
        let differing = if ours.len() == n_cells && theirs.len() == n_cells {
            ours.iter().zip(&theirs).filter(|(a, b)| a.to_bits() != b.to_bits()).count()
        } else {
            n_cells
        };
        if differing > 0 {
            return Err(MpasError::Refusal(format!(
                "history {} has no {name}, so its cell centres are taken from the mesh {}; \
                 its static {field} differs from the mesh's at {differing} of {n_cells} \
                 cells, so the mesh is not this run's (pass the static file the run \
                 was initialised from as --mesh)",
                history_path.display(),
                mesh.source_path.display()
            )));
        }
        agreed.push(field.to_string());
    }
    if agreed.is_empty() {
        eprintln!(
            "{tool}: advisory: history {} has no {name} and repeats none of \
             {MESH_IDENTITY_FIELDS:?}; its cell centres are taken from the mesh {} on \
             the cell count ({n_cells}) alone",
            history_path.display(),
            mesh.source_path.display()
        );
    }
    let mut diag_path = None;
    let mut diag_sha256 = None;
    match open_diag_companion(history_path, n_cells, n_levels, hash_diag)? {
        Some((path, sha256, opened)) => {
            diag_sha256 = sha256;
            diag_path = Some(path);
            diag = Some(opened);
        }
        None => eprintln!(
            "{tool}: advisory: history {} has no {name} and no diag stream \
             file beside it; fields it lacks stay absent",
            history_path.display()
        ),
    }
    Ok((
        SplitSources {
            coordinates_path: mesh.source_path.clone(),
            coordinates_sha256: mesh.source_sha256.clone(),
            agreed_static_fields: agreed,
            diag_path,
            diag_sha256,
            diag_fields: Vec::new(),
        },
        diag,
    ))
}

/// The `diag` stream file beside `history_path` ([`diag_companion`]), opened
/// and checked to be this run's (same cell count, same level count when it
/// carries levels), with its digest when `hash` is set; `None` when there is
/// no such file.  A diag file of another mesh is refused, not skipped: its
/// values would land on the wrong cells.
fn open_diag_companion(
    history_path: &Path,
    n_cells: usize,
    n_levels: usize,
    hash: bool,
) -> MpasResult<Option<(PathBuf, Option<String>, netcrust::File)>> {
    let Some(path) = diag_companion(history_path).filter(|p| p.is_file()) else {
        return Ok(None);
    };
    let opened = netcrust::File::open(&path)?;
    let diag_cells = opened.dimension("nCells").map(|d| d.len());
    if diag_cells != Some(n_cells) {
        return Err(MpasError::Refusal(format!(
            "diag {} has {diag_cells:?} cells and its history {n_cells}; it is \
             not this run's diag stream",
            path.display()
        )));
    }
    if let Some(levels) = opened.dimension("nVertLevels").map(|d| d.len()) {
        if levels != n_levels {
            return Err(MpasError::Refusal(format!(
                "diag {} has {levels} levels and its history {n_levels}; it is \
                 not this run's diag stream",
                path.display()
            )));
        }
    }
    let sha256 = if hash { Some(crate::sha256_file(&path)?) } else { None };
    Ok(Some((path, sha256, opened)))
}

/// Whether a mapped field is carried by `file` under either spelling.
fn carries_mapped(file: &netcrust::File, wrf_name: &str, mpas_name: &str) -> bool {
    file.variable(mpas_name).is_some()
        || native_history_name(wrf_name).is_some_and(|name| file.variable(name).is_some())
}

/// [`read_history_with_init`] that also reads a split-stream history: one
/// that carries no `latCell`/`lonCell`, as RRFS-MPAS writes it.
///
/// A history that carries its own cell centres is read exactly as
/// [`read_history_with_init`] reads it, whatever `mesh` is: MPAS-A output
/// converts to the same bytes.  For a history without them, and only then:
///
/// * the centres are `mesh`'s (the `--mesh` file), after the cell counts
///   agree and every static field of [`MESH_IDENTITY_FIELDS`] that both
///   files carry agrees cell for cell;
/// * a mapped field the history lacks is read from its `diag` stream file
///   ([`diag_companion`]) when that file exists.
///
/// THE BREAKAGE THIS PREVENTS (hex 0.4.0 completeness row #39, 2026-10-16):
/// `gpuwm-hex render` on an RRFS-MPAS run's own output stopped with
/// "history has no latCell" although `--mesh` was given, and the run's 2 m
/// temperature and reflectivity, which the fork writes to `diag.*.nc`, could
/// not be drawn at all.
pub fn read_history_on_mesh(
    history_path: &Path,
    init: Option<&InitFields>,
    valid_time: Option<Timestamp>,
    history_sha256: Option<&str>,
    mesh: Option<&MeshCoordinates>,
) -> MpasResult<MpasFrame> {
    read_history_on_mesh_with_field_set(history_path, init, valid_time, history_sha256, mesh, "full")
}

/// [`read_history_on_mesh`] that decodes only the fields the requested
/// output set carries ([`field_set_allows`]); `"full"` is
/// [`read_history_on_mesh`] itself.
///
/// THE BREAKAGE THIS PREVENTS: `rw_mpas_convert --field-set surface` wrote
/// only the surface set but decoded every mapped history field first, so a
/// 3 km CONUS frame inflated every hydrometeor and number-concentration
/// profile (each 2.3 M cells x 59 levels) only to drop it, and an unrequested
/// profile that was not finite refused the whole frame.
pub fn read_history_on_mesh_with_field_set(
    history_path: &Path,
    init: Option<&InitFields>,
    valid_time: Option<Timestamp>,
    history_sha256: Option<&str>,
    mesh: Option<&MeshCoordinates>,
    field_set: &str,
) -> MpasResult<MpasFrame> {
    if !field_set_is_known(field_set) {
        return Err(MpasError::Refusal(format!(
            "unknown history field set '{field_set}'; its requested fields cannot be determined"
        )));
    }
    // Hashing the history and reading its fields are independent passes over
    // the same file, so they run side by side; a digest mismatch is still
    // refused before anything read is used.
    let (digest, body) = rayon::join(
        || crate::sha256_file(history_path),
        || read_history_body(history_path, init, valid_time, mesh, field_set),
    );
    let digest = digest?;
    if let Some(want) = history_sha256 {
        if digest != want {
            return Err(MpasError::Refusal(format!(
                "history {} is sha256 {digest}, expected {want}",
                history_path.display()
            )));
        }
    }
    let mut frame = body?;
    frame.history_sha256 = digest;
    Ok(frame)
}

/// [`read_history_on_mesh`] less the history digest, which the caller
/// computes beside it.
fn read_history_body(
    history_path: &Path,
    init: Option<&InitFields>,
    valid_time: Option<Timestamp>,
    mesh: Option<&MeshCoordinates>,
    field_set: &str,
) -> MpasResult<MpasFrame> {
    // A WRF name with no FIELD_MAP row is kept: only the set's own rows decide.
    let requested = |wrf_name: &str| {
        FIELD_MAP
            .iter()
            .find(|m| m.wrf_name == wrf_name)
            .map_or(true, |m| field_set_allows(field_set, m))
    };
    let digest = String::new();
    let mut fields: BTreeMap<String, (Vec<f32>, usize)> = BTreeMap::new();
    let mut absent: Vec<String> = Vec::new();

    let file = netcrust::File::open(history_path)?;
    let science_attributes = read_science_attributes(&file)?;
    let n_cells = file
        .dimension("nCells")
        .ok_or_else(|| MpasError::Refusal("history has no nCells dimension".to_string()))?
        .len();
    let n_levels = file
        .dimension("nVertLevels")
        .ok_or_else(|| MpasError::Refusal("history has no nVertLevels dimension".to_string()))?
        .len();
    let n_soil = file.dimension("nSoilLevels").map(|d| d.len()).unwrap_or(0);
    let missing_centre = ["latCell", "lonCell"]
        .into_iter()
        .find(|name| file.variable(name).is_none());
    // `Some((latitude, longitude))` in radians when the history carries its
    // own centres; `None` for a split-stream history, whose centres are the
    // mesh's (already in degrees).
    let mut own_centres: Option<(Vec<f64>, Vec<f64>)> = None;
    let mut split: Option<SplitSources> = None;
    let mut diag: Option<netcrust::File> = None;
    match (missing_centre, mesh) {
        (None, _) => {
            own_centres = Some((file.read_f64("latCell")?, file.read_f64("lonCell")?));
        }
        (Some(name), None) => {
            return Err(MpasError::Refusal(format!("history has no {name}")));
        }
        (Some(name), Some(mesh)) => {
            let (sources, opened) = open_split_sources(
                &file, history_path, name, mesh, n_cells, n_levels, true, "rw_mpas_convert",
            )?;
            split = Some(sources);
            diag = opened;
        }
    }
    // A history with its own centres that lacks a requested mapped field is
    // completed from its diag stream file of the same time (MPAS-A writes
    // refl10cm_max there, not in `output`); see [`DiagCompanion`].  Opened only
    // when a field is missing, so a complete history reads exactly as before.
    let mut companion: Option<DiagCompanion> = None;
    if own_centres.is_some() {
        let lacks_a_field = FIELD_MAP.iter().any(|mapping| {
            mapping.source == FieldSource::History
                && field_set_allows(field_set, mapping)
                && mapping
                    .mpas_name
                    .is_some_and(|mpas| !carries_mapped(&file, mapping.wrf_name, mpas))
        });
        if lacks_a_field {
            if let Some((path, sha256, opened)) =
                open_diag_companion(history_path, n_cells, n_levels, true)?
            {
                companion = Some(DiagCompanion { path, sha256, fields: Vec::new() });
                diag = Some(opened);
            }
        }
    }
    // Which file carries each mapped field is a header question; the reads
    // themselves run one task per field, and their results are taken in
    // FIELD_MAP order, so `fields`, `absent` and the split receipt are what
    // the serial loop produced.
    enum Carrier {
        History(&'static str),
        Diag(&'static str),
    }
    let mut planned: Vec<(&'static str, Option<Carrier>)> = Vec::new();
    for mapping in FIELD_MAP.iter() {
        if mapping.source != FieldSource::History || !field_set_allows(field_set, mapping) {
            continue;
        }
        let Some(mpas_name) = mapping.mpas_name else {
            continue;
        };
        // Prefer the canonical MPAS Registry carrier when both spellings
        // exist. Native winds/t2m must not become absent evidence just because
        // older hex output used u_zonal/v_meridional/t2.
        let name = native_history_name(mapping.wrf_name)
            .filter(|name| file.variable(name).is_some())
            .unwrap_or(mpas_name);
        if file.variable(name).is_some() {
            planned.push((mapping.wrf_name, Some(Carrier::History(name))));
            continue;
        }
        // A split-stream run writes this field to its diag stream, under the
        // same Registry names; so does MPAS-A for the fields its `output`
        // stream leaves out (the companion opened above).
        if let Some(diag) = diag.as_ref() {
            let name = native_history_name(mapping.wrf_name)
                .filter(|name| diag.variable(name).is_some())
                .unwrap_or(mpas_name);
            if diag.variable(name).is_some() {
                planned.push((mapping.wrf_name, Some(Carrier::Diag(name))));
                continue;
            }
        }
        planned.push((mapping.wrf_name, None));
    }
    let read: Vec<MpasResult<Option<(Vec<f32>, usize)>>> = {
        use rayon::prelude::*;
        planned
            .par_iter()
            .map(|(_, carrier)| match carrier {
                Some(Carrier::History(name)) => read_cell_field(&file, name, n_cells),
                Some(Carrier::Diag(name)) => read_cell_field(
                    diag.as_ref().expect("a diag carrier is planned only with a diag file"),
                    name,
                    n_cells,
                ),
                None => Ok(None),
            })
            .collect()
    };
    for ((wrf_name, carrier), value) in planned.iter().zip(read) {
        match (carrier, value?) {
            (Some(Carrier::Diag(_)), Some(value)) => {
                fields.insert(wrf_name.to_string(), value);
                if let Some(split) = split.as_mut() {
                    split.diag_fields.push(wrf_name.to_string());
                } else if let Some(companion) = companion.as_mut() {
                    companion.fields.push(wrf_name.to_string());
                }
            }
            (_, Some(value)) => {
                fields.insert(wrf_name.to_string(), value);
            }
            (_, None) => absent.push(wrf_name.to_string()),
        }
    }
    drop(diag);
    drop(file);

    let mut init_resolved: Option<PathBuf> = None;
    let mut init_digest: Option<String> = None;
    match init {
        Some(init) => {
            if init.n_cells != n_cells {
                return Err(MpasError::Refusal(format!(
                    "init and history disagree on nCells ({} vs {n_cells})",
                    init.n_cells
                )));
            }
            for (name, value) in &init.fields {
                if !requested(name) {
                    continue;
                }
                // An invariant the history omitted, or carried unfilled, comes
                // from the init; one it carries filled keeps the history's words.
                if is_init_fallback(name) {
                    let take = fields
                        .get(name)
                        .map_or(true, |(history, _)| history_invariant_unset(history, &value.0));
                    if take {
                        fields.insert(name.clone(), value.clone());
                        absent.retain(|missing| missing != name);
                    }
                    continue;
                }
                fields.insert(name.clone(), value.clone());
            }
            absent.extend(init.absent.iter().filter(|name| requested(name)).cloned());
            init_resolved = Some(init.path.clone());
            init_digest = Some(init.sha256.clone());
        }
        None => {
            for mapping in FIELD_MAP.iter() {
                if mapping.source == FieldSource::Init && field_set_allows(field_set, mapping) {
                    absent.push(mapping.wrf_name.to_string());
                }
            }
        }
    }

    // The parameterised convection's reflectivity joins REFC on the mesh, before any remap
    // (fieldmap::add_convective_reflectivity says why).
    if let Some((rate, 1)) = fields.get("CUPREC").map(|(v, l)| (v.clone(), *l)) {
        if let Some((refc, 1)) = fields.get_mut("REFC").map(|(v, l)| (v, *l)) {
            crate::fieldmap::add_convective_reflectivity(refc, &rate);
        }
    }

    if let Some((_, levels)) = fields.get("PHB") {
        if *levels != n_levels + 1 {
            return Err(MpasError::Refusal(format!(
                "init zgrid carries {levels} levels; expected {}",
                n_levels + 1
            )));
        }
    }

    let resolved_time = match valid_time {
        Some(t) => t,
        None => Timestamp::parse(
            history_path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or_default(),
        )?,
    };

    let deg = 180.0 / std::f64::consts::PI;
    let (latitude_degrees, longitude_degrees) = match own_centres {
        Some((latitude, longitude)) => (
            latitude.iter().map(|v| v * deg).collect(),
            longitude
                .iter()
                .map(|v| (v * deg + 180.0).rem_euclid(360.0) - 180.0)
                .collect(),
        ),
        None => {
            // Only a split-stream history reaches here, and only with a mesh.
            let mesh = mesh.expect("a history without centres is refused unless a mesh is given");
            (mesh.latitude_degrees.clone(), mesh.longitude_degrees.clone())
        }
    };
    Ok(MpasFrame {
        history_path: history_path.to_path_buf(),
        history_sha256: digest,
        init_path: init_resolved,
        init_sha256: init_digest,
        valid_time: resolved_time,
        n_cells,
        n_levels,
        n_soil_levels: n_soil,
        fields,
        latitude_degrees,
        longitude_degrees,
        absent,
        science_attributes,
        split,
        companion_diag: companion.filter(|c| !c.fields.is_empty()),
    })
}

/// One cell-indexed variable at the listed cells only, as `(values, levels)`
/// with `values` laid out `(cells.len(), levels)` row-major -- the layout
/// [`read_cell_field`] gives the whole mesh, so a gather written for one reads
/// the other.  The cell axis must be the first non-`Time` dimension, as it is
/// in every MPAS stream.  Each cell is one hyperslab read.
fn read_cells_f64(
    file: &netcrust::File,
    name: &str,
    n_cells: usize,
    cells: &[u32],
) -> MpasResult<Option<(Vec<f64>, usize)>> {
    use netcrust::{NcSliceInfo, NcSliceInfoElem};
    let Some(variable) = file.variable(name) else {
        return Ok(None);
    };
    let has_time = variable
        .dimensions()
        .first()
        .map(|d| d.name() == "Time")
        .unwrap_or(false);
    let shape: Vec<usize> = variable.shape();
    if has_time && shape.first() != Some(&1) {
        return Err(MpasError::Refusal(format!(
            "{name} carries {} time records; expected exactly 1",
            shape.first().copied().unwrap_or(0)
        )));
    }
    let rest = if has_time { &shape[1..] } else { &shape[..] };
    if rest.first() != Some(&n_cells) {
        return Err(MpasError::Refusal(format!(
            "{name} leads with {:?} values, not nCells ({n_cells}): a cell subset cannot be read from it",
            rest.first()
        )));
    }
    let levels: usize = rest[1..].iter().product();
    let mut out = Vec::with_capacity(cells.len() * levels);
    for &cell in cells {
        if cell as usize >= n_cells {
            return Err(MpasError::Refusal(format!(
                "{name}: cell {cell} is outside the {n_cells}-cell mesh"
            )));
        }
        let mut selections = Vec::with_capacity(rest.len() + 1);
        if has_time {
            selections.push(NcSliceInfoElem::Index(0));
        }
        selections.push(NcSliceInfoElem::Index(u64::from(cell)));
        for &n in &rest[1..] {
            selections.push(NcSliceInfoElem::Slice { start: 0, end: n as u64, step: 1 });
        }
        let values = file
            .read_array_f64_slice(name, &NcSliceInfo { selections })?
            .into_values();
        if values.len() != levels {
            return Err(MpasError::Refusal(format!(
                "{name} cell {cell} read back {} value(s), not its {levels} level(s)",
                values.len()
            )));
        }
        out.extend_from_slice(&values);
    }
    Ok(Some((out, levels)))
}

/// [`read_cells_f64`] narrowed to `f32` exactly as [`read_cell_field`] narrows
/// a whole field, refusing a non-finite value at a cell it reads.
fn read_cell_field_at(
    file: &netcrust::File,
    name: &str,
    n_cells: usize,
    cells: &[u32],
) -> MpasResult<Option<(Vec<f32>, usize)>> {
    let Some((values, levels)) = read_cells_f64(file, name, n_cells, cells)? else {
        return Ok(None);
    };
    let mut out = Vec::with_capacity(values.len());
    for v in values {
        let f = v as f32;
        if !f.is_finite() {
            return Err(MpasError::Refusal(format!("history {name} is not finite")));
        }
        out.push(f);
    }
    Ok(Some((out, levels)))
}

/// A history frame at a few mesh cells only: what point columns
/// ([`crate::points::columns`]) draw from.
///
/// The returned frame's cell axis is `cells` in the order given
/// (`n_cells == cells.len()`), so it pairs with
/// [`crate::points::PointSampler::compacted`]'s sampler, whose weights index
/// that list.  Only the WRF names in `wanted` are read: history-sourced ones
/// from the history, init-sourced ones (zgrid) from `init_path`; a name the
/// file lacks is recorded in `absent` as [`read_history`] records it.
///
/// THE BREAKAGE THIS PREVENTS: `rw_hex_points` called [`read_history`] once
/// per frame, which hashes the whole history and the whole init and reads
/// every mapped field of both on one thread to keep a handful of columns.  On
/// the 1,472,532-cell 20 km global case (a hex run, 2026-10-13) that read about
/// 300 GB for six points and seven hourly frames and took 2,150 s, longer than
/// the 1,600 s forecast it drew; the 4-card run's cards sat idle behind it.
/// Neither file is hashed here (`history_sha256` says so): the point pictures
/// are drawn from the cells read, and the forecast receipt already pins both
/// files' bytes.
pub fn read_history_cells(
    history_path: &Path,
    init_path: Option<&Path>,
    cells: &[u32],
    wanted: &[&str],
    valid_time: Option<Timestamp>,
) -> MpasResult<MpasFrame> {
    read_history_cells_on_mesh(history_path, init_path, cells, wanted, valid_time, None)
}

/// [`read_history_cells`] that also reads a split-stream history (no
/// `latCell`/`lonCell`, as RRFS-MPAS writes it), as [`read_history_on_mesh`]
/// reads one for the converter: the centres at `cells` are `mesh`'s once the
/// cell counts and the repeated static fields agree, and a wanted field the
/// history lacks is read from the `diag` file of the same time, the native
/// Registry spelling first.  A history with its own centres is read exactly
/// as [`read_history_cells`] reads it, whatever `mesh` is.
///
/// THE BREAKAGE THIS PREVENTS (2.9.1): `rw_hex_points` on an RRFS-MPAS run's
/// own output stopped with "history has no latCell" although `--mesh` was
/// given, after the converter had learned to read the same output.
pub fn read_history_cells_on_mesh(
    history_path: &Path,
    init_path: Option<&Path>,
    cells: &[u32],
    wanted: &[&str],
    valid_time: Option<Timestamp>,
    mesh: Option<&MeshCoordinates>,
) -> MpasResult<MpasFrame> {
    let mut fields: BTreeMap<String, (Vec<f32>, usize)> = BTreeMap::new();
    let mut absent: Vec<String> = Vec::new();
    let wanted_from = |source: FieldSource| {
        FIELD_MAP
            .iter()
            .filter(move |m| m.source == source && wanted.contains(&m.wrf_name))
            .filter_map(|m| m.mpas_name.map(|mpas| (m.wrf_name, mpas)))
    };

    let file = netcrust::File::open(history_path)?;
    let science_attributes = read_science_attributes(&file)?;
    let mesh_cells = file
        .dimension("nCells")
        .ok_or_else(|| MpasError::Refusal("history has no nCells dimension".to_string()))?
        .len();
    let n_levels = file
        .dimension("nVertLevels")
        .ok_or_else(|| MpasError::Refusal("history has no nVertLevels dimension".to_string()))?
        .len();
    let n_soil = file.dimension("nSoilLevels").map(|d| d.len()).unwrap_or(0);
    let missing_centre = ["latCell", "lonCell"]
        .into_iter()
        .find(|name| file.variable(name).is_none());
    let deg = 180.0 / std::f64::consts::PI;
    let mut split: Option<SplitSources> = None;
    let mut companion: Option<Option<(PathBuf, netcrust::File)>> = None;
    let mut companion_fields: Vec<String> = Vec::new();
    let (latitude_degrees, longitude_degrees): (Vec<f64>, Vec<f64>) = match (missing_centre, mesh) {
        (None, _) => {
            let centre = |name: &str| -> MpasResult<Vec<f64>> {
                read_cells_f64(&file, name, mesh_cells, cells)?
                    .map(|(v, _)| v)
                    .ok_or_else(|| MpasError::Refusal(format!("history has no {name}")))
            };
            let latitude = centre("latCell")?;
            let longitude = centre("lonCell")?;
            for (wrf_name, mpas_name) in wanted_from(FieldSource::History) {
                // The legacy spelling first, so every history that carries it
                // reads as before; a history that carries only the native
                // Registry spelling (`uReconstructZonal`, `t2m`, as current hex
                // output does) used to leave the field absent, and a sounding
                // then refused ("carries no U").
                let mut value = match read_cell_field_at(&file, mpas_name, mesh_cells, cells)? {
                    Some(value) => Some(value),
                    None => match native_history_name(wrf_name) {
                        Some(native) => read_cell_field_at(&file, native, mesh_cells, cells)?,
                        None => None,
                    },
                };
                // A field MPAS-A's `output` stream leaves out comes from its
                // diag stream file of the same time ([`DiagCompanion`]), as the
                // converter reads it; opened only once a wanted field is missing.
                if value.is_none() {
                    if companion.is_none() {
                        companion = Some(
                            open_diag_companion(history_path, mesh_cells, n_levels, false)?
                                .map(|(path, _, opened)| (path, opened)),
                        );
                    }
                    if let Some(Some((_, diag))) = companion.as_ref() {
                        let name = native_history_name(wrf_name)
                            .filter(|native| diag.variable(native).is_some())
                            .unwrap_or(mpas_name);
                        value = read_cell_field_at(diag, name, mesh_cells, cells)?;
                        if value.is_some() {
                            companion_fields.push(wrf_name.to_string());
                        }
                    }
                }
                match value {
                    Some(value) => {
                        fields.insert(wrf_name.to_string(), value);
                    }
                    None => absent.push(wrf_name.to_string()),
                }
            }
            (
                latitude.iter().map(|v| v * deg).collect(),
                longitude
                    .iter()
                    .map(|v| (v * deg + 180.0).rem_euclid(360.0) - 180.0)
                    .collect(),
            )
        }
        (Some(name), None) => {
            return Err(MpasError::Refusal(format!("history has no {name}")));
        }
        (Some(name), Some(mesh)) => {
            let (mut sources, diag) = open_split_sources(
                &file, history_path, name, mesh, mesh_cells, n_levels, false, "rw_hex_points",
            )?;
            let mut gathered = (Vec::with_capacity(cells.len()), Vec::with_capacity(cells.len()));
            for &cell in cells {
                let cell = cell as usize;
                if cell >= mesh_cells {
                    return Err(MpasError::Refusal(format!(
                        "cell {cell} is outside the mesh's {mesh_cells} cells"
                    )));
                }
                gathered.0.push(mesh.latitude_degrees[cell]);
                gathered.1.push(mesh.longitude_degrees[cell]);
            }
            for (wrf_name, mpas_name) in wanted_from(FieldSource::History) {
                // The converter's rule for a split-stream history: the native
                // Registry spelling first, then the legacy one; history, then diag.
                let pick = |f: &netcrust::File| {
                    native_history_name(wrf_name)
                        .filter(|native| f.variable(native).is_some())
                        .unwrap_or(mpas_name)
                };
                if let Some(value) = read_cell_field_at(&file, pick(&file), mesh_cells, cells)? {
                    fields.insert(wrf_name.to_string(), value);
                    continue;
                }
                if let Some(diag) = diag.as_ref() {
                    if let Some(value) = read_cell_field_at(diag, pick(diag), mesh_cells, cells)? {
                        fields.insert(wrf_name.to_string(), value);
                        sources.diag_fields.push(wrf_name.to_string());
                        continue;
                    }
                }
                absent.push(wrf_name.to_string());
            }
            split = Some(sources);
            gathered
        }
    };
    drop(file);

    let mut init_resolved: Option<PathBuf> = None;
    match init_path {
        Some(path) => {
            let init = netcrust::File::open(path)?;
            let init_cells = init
                .dimension("nCells")
                .ok_or_else(|| MpasError::Refusal("init has no nCells dimension".to_string()))?
                .len();
            if init_cells != mesh_cells {
                return Err(MpasError::Refusal(format!(
                    "init and history disagree on nCells ({init_cells} vs {mesh_cells})"
                )));
            }
            for (wrf_name, mpas_name) in wanted_from(FieldSource::Init) {
                match read_cell_field_at(&init, mpas_name, mesh_cells, cells)? {
                    Some(value) => {
                        fields.insert(wrf_name.to_string(), value);
                    }
                    None => absent.push(wrf_name.to_string()),
                }
            }
            // The same invariants the frame path completes from the init
            // ([`INIT_FALLBACK_FIELDS`]), for a wanted one the history omitted
            // or carried unfilled ([`history_invariant_unset`]).
            for (wrf_name, mpas_name) in INIT_FALLBACK_FIELDS.iter().copied() {
                let carried_filled = fields
                    .get(wrf_name)
                    .is_some_and(|(history, _)| history.iter().any(|v| *v != 0.0));
                if !wanted.contains(&wrf_name) || carried_filled {
                    continue;
                }
                if let Some(value) = read_cell_field_at(&init, mpas_name, mesh_cells, cells)? {
                    let take = fields
                        .get(wrf_name)
                        .map_or(true, |(history, _)| history_invariant_unset(history, &value.0));
                    if take {
                        fields.insert(wrf_name.to_string(), value);
                        absent.retain(|missing| missing != wrf_name);
                    }
                }
            }
            init_resolved = Some(path.to_path_buf());
        }
        None => absent.extend(wanted_from(FieldSource::Init).map(|(w, _)| w.to_string())),
    }

    // The parameterised convection's reflectivity joins REFC on the mesh, before any remap
    // (fieldmap::add_convective_reflectivity says why).
    if let Some((rate, 1)) = fields.get("CUPREC").map(|(v, l)| (v.clone(), *l)) {
        if let Some((refc, 1)) = fields.get_mut("REFC").map(|(v, l)| (v, *l)) {
            crate::fieldmap::add_convective_reflectivity(refc, &rate);
        }
    }

    if let Some((_, levels)) = fields.get("PHB") {
        if *levels != n_levels + 1 {
            return Err(MpasError::Refusal(format!(
                "init zgrid carries {levels} levels; expected {}",
                n_levels + 1
            )));
        }
    }
    let resolved_time = match valid_time {
        Some(t) => t,
        None => Timestamp::parse(
            history_path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or_default(),
        )?,
    };
    Ok(MpasFrame {
        history_path: history_path.to_path_buf(),
        history_sha256: "not hashed: cell-subset read (read_history_cells)".to_string(),
        init_path: init_resolved,
        init_sha256: None,
        valid_time: resolved_time,
        n_cells: cells.len(),
        n_levels,
        n_soil_levels: n_soil,
        fields,
        latitude_degrees,
        longitude_degrees,
        absent,
        science_attributes,
        split,
        companion_diag: match (companion, companion_fields.is_empty()) {
            (Some(Some((path, _))), false) => {
                Some(DiagCompanion { path, sha256: None, fields: companion_fields })
            }
            _ => None,
        },
    })
}

/// Refuse unless the history's own cell centres are the weights' mesh.
///
/// This is the guard against the one bug that would silently ruin every map:
/// resampling one mesh's fields through another mesh's index map, or through
/// a de-rotated copy of a `grid_rotate` output. The rotated grid stores
/// post-rotation earth coordinates, so a correct pairing agrees to float32
/// round-off; anything else is a different mesh and is refused here rather
/// than discovered by eye in a flipped map.
pub fn assert_mesh_agrees(
    frame: &MpasFrame,
    mesh: &MeshCoordinates,
    tolerance_degrees: f64,
) -> MpasResult<(f64, f64)> {
    if mesh.n_cells() != frame.n_cells {
        return Err(MpasError::Refusal(format!(
            "mesh has {} cells; history has {}",
            mesh.n_cells(),
            frame.n_cells
        )));
    }
    let mut latitude_error: f64 = 0.0;
    let mut longitude_error: f64 = 0.0;
    for i in 0..frame.n_cells {
        let dlat = (mesh.latitude_degrees[i] - frame.latitude_degrees[i]).abs();
        if dlat > latitude_error {
            latitude_error = dlat;
        }
        let raw = (mesh.longitude_degrees[i] - frame.longitude_degrees[i]).abs();
        let dlon = raw.min(360.0 - raw);
        if dlon > longitude_error {
            longitude_error = dlon;
        }
    }
    if latitude_error > tolerance_degrees || longitude_error > tolerance_degrees {
        return Err(MpasError::Refusal(format!(
            "history cell centres do not match the weights mesh (max |dlat|={latitude_error:e} deg, max |dlon|={longitude_error:e} deg)"
        )));
    }
    Ok((latitude_error, longitude_error))
}

/// The mapping's declared level count for this frame, or `None` when the
/// frame cannot carry it at all.
pub fn expected_levels(mapping: &FieldMapping, frame: &MpasFrame) -> Option<usize> {
    match mapping.rank {
        crate::fieldmap::Rank::Surface => Some(1),
        crate::fieldmap::Rank::Mass3d | crate::fieldmap::Rank::U3d | crate::fieldmap::Rank::V3d => {
            Some(frame.n_levels)
        }
        crate::fieldmap::Rank::W3d => Some(frame.n_levels + 1),
        crate::fieldmap::Rank::Soil => {
            if frame.n_soil_levels == 0 {
                None
            } else {
                Some(frame.n_soil_levels)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The surface set decodes only its own rows: an unrequested profile is
    /// never read (so its non-finite value cannot refuse the frame), while
    /// the full set still reads and refuses it.
    #[test]
    fn surface_read_does_not_decode_an_unrequested_nonfinite_profile() {
        use rw_store::netcdf_classic::{NcClassicWriter, NcData, NcDim, NcFormat, NcType, NcVarDef};
        use std::sync::atomic::{AtomicU64, Ordering};

        struct CreatedFixture(PathBuf);
        impl Drop for CreatedFixture {
            fn drop(&mut self) {
                let _ = std::fs::remove_file(&self.0);
            }
        }
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let fixture = CreatedFixture(std::env::temp_dir().join(format!(
            "rw-mpas-field-set-{}-{}.nc",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        )));
        let mut writer = NcClassicWriter::create(
            &fixture.0,
            NcFormat::Offset64,
            vec![NcDim::fixed("nCells", 2), NcDim::fixed("nVertLevels", 2)],
            vec![],
            vec![
                NcVarDef::new("latCell", NcType::Double, vec![0]),
                NcVarDef::new("lonCell", NcType::Double, vec![0]),
                NcVarDef::new("t2", NcType::Float, vec![0]),
                NcVarDef::new("qc", NcType::Float, vec![0, 1]),
            ],
            0,
        )
        .unwrap();
        writer.put("latCell", NcData::Doubles(&[0.0, 0.01])).unwrap();
        writer.put("lonCell", NcData::Doubles(&[0.0, 0.01])).unwrap();
        writer.put("t2", NcData::Floats(&[280.0, 281.0])).unwrap();
        writer
            .put("qc", NcData::Floats(&[1.0e-4, 2.0e-4, 3.0e-4, f32::NAN]))
            .unwrap();
        writer.finish().unwrap();
        let time = Some(Timestamp::parse("2026-01-01_00:00:00").unwrap());
        let surface =
            read_history_on_mesh_with_field_set(&fixture.0, None, time, None, None, "surface")
                .expect("unrequested profile is not decoded");
        assert_eq!(surface.fields["T2"].0, [280.0, 281.0]);
        assert!(!surface.fields.contains_key("QCLOUD"));
        assert!(!surface.absent.iter().any(|name| name == "QCLOUD"));
        let error = read_history_on_mesh(&fixture.0, None, time, None, None)
            .unwrap_err()
            .to_string();
        assert!(error.contains("history qc is not finite"), "{error}");
        let unknown =
            read_history_on_mesh_with_field_set(&fixture.0, None, time, None, None, "nope")
                .unwrap_err()
                .to_string();
        assert!(unknown.contains("unknown history field set"), "{unknown}");
    }

    #[test]
    fn a_history_file_name_yields_its_valid_time() {
        let t = Timestamp::parse("cuda-history.2026-08-10_13.00.00.nc").unwrap();
        assert_eq!(t.label(), "2026-08-10_13:00:00");
        assert_eq!(t.iso(), "2026-08-10T13:00:00");
    }

    #[test]
    fn a_name_without_a_stamp_is_refused() {
        let err = Timestamp::parse("history.nc").unwrap_err().to_string();
        assert!(err.contains("no YYYY-MM-DD"), "{err}");
    }

    #[test]
    fn a_lead_time_is_the_difference_in_seconds() {
        let origin = Timestamp::parse("2026-08-10_12:00:00").unwrap();
        let valid = Timestamp::parse("2026-08-10_13:00:00").unwrap();
        assert_eq!(valid.seconds_since(&origin), 3600);
        let next_day = Timestamp::parse("2026-08-11_00:00:00").unwrap();
        assert_eq!(next_day.seconds_since(&origin), 43_200);
        let next_month = Timestamp::parse("2026-09-01_12:00:00").unwrap();
        assert_eq!(next_month.seconds_since(&origin), 22 * 86_400);
    }
}
