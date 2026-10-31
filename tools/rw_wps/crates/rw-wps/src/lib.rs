//! RW-WPS: a typed, fail-closed frontend to gpuwm's native WRF preprocessor.
//!
//! This crate owns configuration, capability discovery, process orchestration,
//! progress events, and output receipts.  It intentionally does not copy the
//! interpolation, WRF-real, static-geography, or export algorithms from
//! gpuwm.  The engine remains the scientific authority and must advertise a
//! certified source before RW-WPS will launch it.

pub mod mapping;
pub mod namelist;
pub mod sfire;

use mapping::{
    AxisRole, GridLocation, MissingPolicy, SourceFormat, Staggering, TargetContract,
    VariableSelector, inspect_source_fields, read_mapping, validate_mapping,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::ffi::{OsStr, OsString};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use thiserror::Error;

pub const RUN_SCHEMA: &str = "rw-wps.run.v1";
pub const RECEIPT_SCHEMA: &str = "rw-wps.receipt.v1";
pub const PROGRESS_SCHEMA: &str = "rw-wps.progress.v1";
pub const CAPABILITY_SCHEMA: &str = "gpuwm-native-source-adapters-v1";
pub const MAPPED_COMPOSITION_CONTRACT: &str = "gpuwm-mapped-composition-v2";
pub const MAPPED_INPUT_MANIFEST_SCHEMA: &str = "gpuwm-mapped-composition-inputs-v1";
pub const AUTHOR_MAPPED_PLAN_SCHEMA: &str = "rw-wps.author-mapped-plan.v1";

#[derive(Debug, Error)]
pub enum RwWpsError {
    #[error("I/O error for {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("invalid JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("invalid RW-WPS configuration: {0}")]
    Config(String),
    #[error("native engine failed: {0}")]
    Engine(String),
    #[error("WRF output verification failed: {0}")]
    Output(String),
}

pub(crate) fn io_error(path: impl Into<PathBuf>, source: std::io::Error) -> RwWpsError {
    RwWpsError::Io {
        path: path.into(),
        source,
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RunConfig {
    pub schema: String,
    pub source: SourceConfig,
    pub domain: DomainConfig,
    pub backend: BackendConfig,
    pub output: OutputTarget,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "lowercase", deny_unknown_fields)]
pub enum SourceConfig {
    Hrrr {
        #[serde(default)]
        source_root: Option<PathBuf>,
        input_manifest: PathBuf,
        input_manifest_sha256: String,
        valid_time: String,
        #[serde(default)]
        root_preparation: Option<PathBuf>,
        #[serde(default)]
        run_seconds: Option<u32>,
        #[serde(default)]
        pipeline_workers: Option<u16>,
    },
    Gfs {
        series: PathBuf,
        cycle: String,
        bridge: PathBuf,
        input_manifest: PathBuf,
        input_manifest_sha256: String,
    },
    Era5 {
        grib: PathBuf,
        vtable: PathBuf,
        bridge: PathBuf,
        source_orography: PathBuf,
        #[serde(default = "default_orography_variable")]
        source_orography_variable: String,
        input_manifest: PathBuf,
        input_manifest_sha256: String,
    },
    Mapped {
        contract: MappedContract,
        format: SourceFormat,
        mapping: PathBuf,
        composition: PathBuf,
        primary_files: Vec<PathBuf>,
        supplements: Vec<RolePathBinding>,
        provenance: Vec<RolePathBinding>,
        decoder: MappedDecoderConfig,
        input_manifest: PathBuf,
        input_manifest_sha256: String,
        #[serde(default)]
        hierarchy_workers: Option<u16>,
    },
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum MappedContract {
    #[serde(rename = "gpuwm-mapped-composition-v2")]
    CompositionV2,
}

impl MappedContract {
    pub const fn id(self) -> &'static str {
        match self {
            Self::CompositionV2 => MAPPED_COMPOSITION_CONTRACT,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RolePathBinding {
    pub role: String,
    pub path: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "format", rename_all = "lowercase", deny_unknown_fields)]
pub enum MappedDecoderConfig {
    Grib1 { bridge: PathBuf },
    Grib2 { inventory: PathBuf, dump: PathBuf },
    Netcdf,
}

/// Explicit mapped-source contract authoring request.
///
/// This frontend deliberately does not parse a Vtable into scientific
/// semantics or hash inputs itself. It only preserves the caller's exact
/// paths/order and delegates create-only authoring to `woof-wrf-init`, whose
/// Python authoring engine remains authoritative.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuthorMappedRequest {
    pub format: SourceFormat,
    pub mapping: AuthorMappingInput,
    pub composition: PathBuf,
    pub primary_files: Vec<PathBuf>,
    pub supplements: Vec<OsString>,
    pub provenance: Vec<OsString>,
    pub decoder: AuthorMappedDecoder,
    pub input_manifest: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuthorMappingInput {
    Existing {
        mapping: PathBuf,
    },
    Descriptor {
        descriptor: PathBuf,
        vtable: Option<PathBuf>,
        output_mapping: PathBuf,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuthorMappedDecoder {
    Grib1 { bridge: PathBuf },
    Grib2 { inventory: PathBuf, dump: PathBuf },
    Netcdf,
}

impl MappedDecoderConfig {
    const fn format(&self) -> SourceFormat {
        match self {
            Self::Grib1 { .. } => SourceFormat::Grib1,
            Self::Grib2 { .. } => SourceFormat::Grib2,
            Self::Netcdf => SourceFormat::Netcdf,
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct MappedCompositionV2 {
    schema: String,
    name: String,
    mapping_binding: String,
    soil_layers: MappedSoilLayers,
    supplements: MappedSupplements,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct MappedSoilLayers {
    temperature_field: String,
    moisture_field: String,
    depth_units: String,
    source_layers: Vec<MappedSoilSourceLayer>,
    target_layers: Vec<MappedSoilLayer>,
    remap: MappedSoilRemap,
    missing: MappedSoilMissing,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct MappedSoilSourceLayer {
    top: f64,
    bottom: f64,
    selectors: MappedSoilLayerSelectors,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct MappedSoilLayerSelectors {
    soil_temperature: VariableSelector,
    volumetric_soil_moisture: VariableSelector,
}

#[derive(Debug, Deserialize, Clone, Copy, PartialEq)]
#[serde(deny_unknown_fields)]
struct MappedSoilLayer {
    top: f64,
    bottom: f64,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum MappedSoilRemap {
    LinearPointSamples {
        source_value_location: String,
        target_value_location: String,
        top_anchor: MappedSoilAnchor,
        bottom_anchor: MappedSoilAnchor,
    },
    ConservativeLayerMeans {
        source_value_location: String,
        target_value_location: String,
        coverage: String,
    },
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct MappedSoilAnchor {
    depth: f64,
    temperature: String,
    moisture: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct MappedSoilMissing {
    land: String,
    ocean: MappedSoilOceanRepair,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct MappedSoilOceanRepair {
    stage: String,
    temperature: String,
    moisture: f64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct MappedSupplements {
    terrain_height: MappedTerrainSupplement,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct MappedTerrainSupplement {
    data_role: String,
    provenance_role: String,
    format: SourceFormat,
    field: String,
    selector_authority: String,
    grid_alignment: String,
    time_alignment: String,
    require_invariant_across_time: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct MappedInputManifestV1 {
    schema: String,
    mapping_sha256: String,
    composition_sha256: String,
    primary_files: Vec<ManifestFileBinding>,
    supplements: BTreeMap<String, ManifestFileInventory>,
    provenance: BTreeMap<String, ManifestFileBinding>,
    decoders: BTreeMap<String, ManifestFileBinding>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ManifestFileBinding {
    path: PathBuf,
    bytes: u64,
    sha256: String,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum ManifestFileInventory {
    One(ManifestFileBinding),
    Many(Vec<ManifestFileBinding>),
}

fn default_orography_variable() -> String {
    "SOILHGT".to_owned()
}

impl SourceConfig {
    pub fn id(&self) -> &'static str {
        match self {
            Self::Hrrr { .. } => "hrrr",
            Self::Gfs { .. } => "gfs",
            Self::Era5 { .. } => "era5",
            Self::Mapped { .. } => "mapped",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct DomainConfig {
    #[serde(default)]
    pub wps_namelist: Option<PathBuf>,
    #[serde(default)]
    pub namelist_input: Option<PathBuf>,
    #[serde(default)]
    pub stock_wrf_namelist_input: Option<PathBuf>,
    #[serde(default)]
    pub root_domain_spec: Option<PathBuf>,
    #[serde(default)]
    pub geog_root: Option<PathBuf>,
    #[serde(default)]
    pub static_cache: Option<PathBuf>,
    #[serde(default)]
    pub static_input: Option<PathBuf>,
    #[serde(default)]
    pub static_receipt: Option<PathBuf>,
    #[serde(default)]
    pub experiment_config: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "lowercase", deny_unknown_fields)]
pub enum BackendConfig {
    Cpu {
        #[serde(default)]
        workers: Option<u16>,
        #[serde(default)]
        bridge: Option<PathBuf>,
    },
    Cuda,
    Auto {
        #[serde(default)]
        workers: Option<u16>,
    },
}

impl BackendConfig {
    pub fn id(&self) -> &'static str {
        match self {
            Self::Cpu { .. } => "cpu",
            Self::Cuda => "cuda",
            Self::Auto { .. } => "auto",
        }
    }

    fn workers(&self) -> Option<u16> {
        match self {
            Self::Cpu { workers, .. } | Self::Auto { workers } => *workers,
            Self::Cuda => None,
        }
    }

    fn bridge(&self) -> Option<&Path> {
        match self {
            Self::Cpu { bridge, .. } => bridge.as_deref(),
            Self::Cuda | Self::Auto { .. } => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "format", rename_all = "lowercase", deny_unknown_fields)]
pub enum OutputTarget {
    Wrf { root: PathBuf },
}

impl OutputTarget {
    pub fn root(&self) -> &Path {
        match self {
            Self::Wrf { root } => root,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SourceCapability {
    pub source_id: String,
    pub status: String,
    pub runnable: bool,
    pub runner: Option<String>,
    #[serde(default)]
    pub notes: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CapabilityManifest {
    pub schema: String,
    pub sources: Vec<SourceCapability>,
    pub runnable_source_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProgressEvent {
    pub schema: String,
    pub sequence: u64,
    pub stage: String,
    pub stream: String,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ArtifactReceipt {
    pub path: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub declared_path: Option<String>,
    pub byte_count: u64,
    pub sha256: String,
    pub kind: String,
}

#[derive(Debug, Deserialize)]
struct NativeWrfManifest {
    schema: String,
    status: String,
    boundary_interval_seconds: u64,
    boundary_record_count: usize,
    boundary_times: Vec<String>,
    next_boundary_times: Vec<String>,
    files: BTreeMap<String, NativeWrfFileSpec>,
    #[serde(default)]
    dimensions: Option<NativeWrfDimensions>,
    #[serde(default)]
    hierarchy: Vec<NativeWrfDomain>,
}

#[derive(Debug, Deserialize)]
struct NativeWrfFileSpec {
    bytes: u64,
    sha256: String,
}

#[derive(Debug, Deserialize)]
struct NativeWrfDimensions {
    nx: usize,
    ny: usize,
    nz: usize,
}

#[derive(Debug, Deserialize)]
struct NativeWrfDomain {
    grid_id: usize,
    nx: usize,
    ny: usize,
    nz: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RunReceipt {
    pub schema: String,
    pub verdict: String,
    pub source: String,
    pub backend: String,
    pub config_sha256: String,
    pub engine: String,
    pub capability_manifest_sha256: String,
    pub command: Vec<String>,
    /// Trajectory-changing regular files captured before launch. Directory
    /// authorities remain the engine receipt's responsibility.
    #[serde(default)]
    pub input_artifacts: Vec<ArtifactReceipt>,
    pub started_unix_ms: u128,
    pub elapsed_ms: u128,
    pub exit_code: i32,
    pub output_root: String,
    pub max_dom: usize,
    pub artifacts: Vec<ArtifactReceipt>,
    #[serde(default)]
    pub error: Option<String>,
}

#[derive(Debug, Clone)]
pub struct PlannedRun {
    pub config: RunConfig,
    pub config_bytes: Vec<u8>,
    pub config_sha256: String,
    pub args: Vec<OsString>,
}

pub fn read_config(path: &Path) -> Result<(RunConfig, Vec<u8>), RwWpsError> {
    let bytes = fs::read(path).map_err(|error| io_error(path, error))?;
    let config = serde_json::from_slice::<RunConfig>(&bytes)?;
    Ok((config, bytes))
}

pub fn validate_config(config: &RunConfig, check_paths: bool) -> Result<(), RwWpsError> {
    if config.schema != RUN_SCHEMA {
        return Err(RwWpsError::Config(format!(
            "schema must be {RUN_SCHEMA:?}, got {:?}",
            config.schema
        )));
    }
    if matches!(config.source, SourceConfig::Hrrr { .. })
        && !matches!(config.backend, BackendConfig::Cpu { .. })
    {
        return Err(RwWpsError::Config(
            "HRRR currently exposes only the certified CPU preprocessing routes".to_owned(),
        ));
    }
    if config.backend.workers() == Some(0) {
        return Err(RwWpsError::Config(
            "backend workers must be positive".to_owned(),
        ));
    }
    if config.backend.workers().is_some_and(|workers| workers > 32) {
        return Err(RwWpsError::Config(
            "backend workers must not exceed the engine limit of 32".to_owned(),
        ));
    }
    validate_source_and_domain(config)?;
    if check_paths {
        for (label, path, expected) in input_paths(config) {
            let valid = match expected {
                PathKind::File => path.is_file(),
                PathKind::Directory => path.is_dir(),
            };
            if !valid {
                return Err(RwWpsError::Config(format!(
                    "{label} {} is not an existing {}",
                    path.display(),
                    expected.label()
                )));
            }
        }
        verify_source_manifest(&config.source)?;
        if let SourceConfig::Mapped {
            format,
            mapping,
            composition,
            primary_files,
            supplements,
            provenance,
            decoder,
            input_manifest,
            ..
        } = &config.source
        {
            let loaded = read_mapping(mapping)?;
            if loaded.format != *format {
                return Err(RwWpsError::Config(format!(
                    "source.format is {}, but mapping {} declares {}",
                    format.id(),
                    mapping.display(),
                    loaded.format.id()
                )));
            }
            let report = validate_mapping(&loaded);
            if !report.passed() {
                let summary = report
                    .errors
                    .iter()
                    .take(8)
                    .map(|item| format!("{}: {}", item.code, item.message))
                    .collect::<Vec<_>>()
                    .join("; ");
                return Err(RwWpsError::Config(format!(
                    "mapping {} failed with {} error(s): {summary}",
                    mapping.display(),
                    report.errors.len()
                )));
            }
            let (supplement_role, _) =
                validate_composition_bindings(composition, &loaded, supplements, provenance)?;
            let _ = mapped_inspection_paths(primary_files, supplements)?;
            verify_mapped_input_manifest(
                input_manifest,
                mapping,
                composition,
                primary_files,
                supplements,
                provenance,
                decoder,
            )?;
            let primary_fields = loaded
                .fields
                .keys()
                .filter(|field| field.as_str() != "terrain_height")
                .cloned()
                .collect::<BTreeSet<_>>();
            let primary_inspection =
                inspect_source_fields(&loaded, primary_files, &primary_fields)?;
            if primary_inspection.verdict != "PASS" {
                return Err(RwWpsError::Config(format!(
                    "mapped primary-source inventory failed: {}",
                    primary_inspection.errors.join("; ")
                )));
            }
            let supplement_paths = supplements
                .iter()
                .filter(|binding| binding.role == supplement_role)
                .map(|binding| binding.path.clone())
                .collect::<Vec<_>>();
            let terrain_fields = BTreeSet::from(["terrain_height".to_owned()]);
            let supplement_inspection =
                inspect_source_fields(&loaded, &supplement_paths, &terrain_fields)?;
            if supplement_inspection.verdict != "PASS" {
                return Err(RwWpsError::Config(format!(
                    "mapped terrain-supplement inventory failed: {}",
                    supplement_inspection.errors.join("; ")
                )));
            }
        }
    }
    Ok(())
}

fn source_manifest(source: &SourceConfig) -> (&Path, &str) {
    match source {
        SourceConfig::Hrrr {
            input_manifest,
            input_manifest_sha256,
            ..
        }
        | SourceConfig::Gfs {
            input_manifest,
            input_manifest_sha256,
            ..
        }
        | SourceConfig::Era5 {
            input_manifest,
            input_manifest_sha256,
            ..
        }
        | SourceConfig::Mapped {
            input_manifest,
            input_manifest_sha256,
            ..
        } => (input_manifest, input_manifest_sha256),
    }
}

fn verify_source_manifest(source: &SourceConfig) -> Result<(), RwWpsError> {
    let (manifest, declared_sha256) = source_manifest(source);
    let actual_sha256 = sha256_file(manifest)?;
    if !actual_sha256.eq_ignore_ascii_case(declared_sha256) {
        return Err(RwWpsError::Config(format!(
            "input manifest digest mismatch for {}: declared {}, actual {}",
            manifest.display(),
            declared_sha256,
            actual_sha256
        )));
    }
    Ok(())
}

fn mapped_inspection_paths(
    primary_files: &[PathBuf],
    supplements: &[RolePathBinding],
) -> Result<Vec<PathBuf>, RwWpsError> {
    let mut union_seen = BTreeSet::new();
    let mut primary_seen = BTreeSet::new();
    let mut supplement_seen = BTreeSet::new();
    let mut paths = Vec::new();
    for path in primary_files {
        let canonical = fs::canonicalize(path).map_err(|error| io_error(path, error))?;
        if !primary_seen.insert(canonical.clone()) {
            return Err(RwWpsError::Config(format!(
                "mapped primary inventory contains duplicate resolved path {}",
                canonical.display()
            )));
        }
        if union_seen.insert(canonical.clone()) {
            paths.push(canonical);
        }
    }
    for binding in supplements {
        let canonical =
            fs::canonicalize(&binding.path).map_err(|error| io_error(&binding.path, error))?;
        if !supplement_seen.insert((binding.role.clone(), canonical.clone())) {
            return Err(RwWpsError::Config(format!(
                "mapped supplement role {:?} contains duplicate resolved path {}",
                binding.role,
                canonical.display()
            )));
        }
        if union_seen.insert(canonical.clone()) {
            paths.push(canonical);
        }
    }
    Ok(paths)
}

const NOAH_SOIL_LAYERS: [MappedSoilLayer; 4] = [
    MappedSoilLayer {
        top: 0.0,
        bottom: 0.1,
    },
    MappedSoilLayer {
        top: 0.1,
        bottom: 0.4,
    },
    MappedSoilLayer {
        top: 0.4,
        bottom: 1.0,
    },
    MappedSoilLayer {
        top: 1.0,
        bottom: 2.0,
    },
];

fn validate_layer_geometry(layers: &[MappedSoilLayer], label: &str) -> Result<(), RwWpsError> {
    if layers.is_empty() {
        return Err(RwWpsError::Config(format!(
            "{label} must be a non-empty ordered layer list"
        )));
    }
    let mut previous_top = f64::NEG_INFINITY;
    let mut previous_bottom = None;
    for (index, layer) in layers.iter().enumerate() {
        if !layer.top.is_finite()
            || !layer.bottom.is_finite()
            || layer.top < 0.0
            || layer.bottom <= layer.top
        {
            return Err(RwWpsError::Config(format!(
                "{label}[{index}] must have non-negative, positive-thickness depths ordered top-to-bottom"
            )));
        }
        if layer.top < previous_top {
            return Err(RwWpsError::Config(format!(
                "{label} is not ordered shallow-to-deep"
            )));
        }
        if let Some(bottom) = previous_bottom {
            if layer.top < bottom {
                return Err(RwWpsError::Config(format!(
                    "{label} layers {} and {index} overlap",
                    index - 1
                )));
            }
            if layer.top > bottom {
                return Err(RwWpsError::Config(format!(
                    "{label} has a gap between layers {} and {index}",
                    index - 1
                )));
            }
        }
        previous_top = layer.top;
        previous_bottom = Some(layer.bottom);
    }
    Ok(())
}

fn validate_soil_mapping_field(
    mapping: &mapping::NativeMapping,
    field_name: &str,
    target_units: &str,
    source_layers: &[MappedSoilSourceLayer],
) -> Result<(), RwWpsError> {
    let field = mapping.fields.get(field_name).ok_or_else(|| {
        RwWpsError::Config(format!("soil contract field {field_name:?} is not mapped"))
    })?;
    if field.target_axes != [AxisRole::Soil, AxisRole::Y, AxisRole::X]
        || field.location != GridLocation::Soil
        || field.staggering != Staggering::None
        || field.units.target != target_units
    {
        return Err(RwWpsError::Config(format!(
            "soil contract field {field_name:?} must be unstaggered soil/y/x in {target_units}"
        )));
    }
    if !matches!(
        field.missing,
        MissingPolicy::Reject | MissingPolicy::PreserveMask
    ) {
        return Err(RwWpsError::Config(format!(
            "soil contract field {field_name:?} must reject missing values or preserve an ocean mask for declared repair"
        )));
    }
    let count = field.selectors.len();
    if count != source_layers.len() {
        return Err(RwWpsError::Config(format!(
            "soil contract requires exactly {} ordered direct selectors for {field_name:?}",
            source_layers.len()
        )));
    }
    let expected_stack_axis = if mapping.format == SourceFormat::Netcdf {
        Some(AxisRole::Soil)
    } else {
        None
    };
    if field.selector_stack_axis != expected_stack_axis {
        return Err(RwWpsError::Config(format!(
            "soil contract field {field_name:?} uses the wrong selector_stack_axis for {:?}",
            mapping.format
        )));
    }
    for (index, (selector, layer)) in field.selectors.iter().zip(source_layers).enumerate() {
        let declared = match field_name {
            "soil_temperature" => &layer.selectors.soil_temperature,
            "volumetric_soil_moisture" => &layer.selectors.volumetric_soil_moisture,
            _ => {
                return Err(RwWpsError::Config(format!(
                    "unsupported canonical soil field {field_name:?}"
                )));
            }
        };
        if selector != declared {
            return Err(RwWpsError::Config(format!(
                "{field_name} selector {index} differs from the selector bound to its declared soil depth"
            )));
        }
    }

    if mapping.format == SourceFormat::Grib2 {
        for (index, (selector, layer)) in field.selectors.iter().zip(source_layers).enumerate() {
            match selector {
                VariableSelector::Grib2 {
                    level_type: Some(106),
                    level_value: Some(top),
                    second_level_type: Some(106),
                    second_level_value: Some(bottom),
                    ..
                } if top.is_finite()
                    && bottom.is_finite()
                    && *top == layer.top
                    && *bottom == layer.bottom => {}
                VariableSelector::Grib2 {
                    level_type: Some(106),
                    level_value: Some(top),
                    second_level_type: Some(106),
                    second_level_value: Some(bottom),
                    ..
                } => {
                    return Err(RwWpsError::Config(format!(
                        "{field_name} selector {index} depth ({top}, {bottom}) differs from soil contract layer ({}, {})",
                        layer.top, layer.bottom
                    )));
                }
                _ => {
                    return Err(RwWpsError::Config(format!(
                        "{field_name} selector {index} must declare a bounded GRIB2 depth-below-land layer (type 106)"
                    )));
                }
            }
        }
    }
    Ok(())
}

fn validate_soil_contract(
    soil: &MappedSoilLayers,
    mapping: &mapping::NativeMapping,
) -> Result<(), RwWpsError> {
    if soil.temperature_field != "soil_temperature"
        || soil.moisture_field != "volumetric_soil_moisture"
    {
        return Err(RwWpsError::Config(
            "composition must bind the canonical soil fields".to_owned(),
        ));
    }
    if soil.depth_units != "m" {
        return Err(RwWpsError::Config(
            "composition soil depth_units must be 'm'; implicit depth-unit conversion is forbidden"
                .to_owned(),
        ));
    }
    let source_bounds = soil
        .source_layers
        .iter()
        .map(|layer| MappedSoilLayer {
            top: layer.top,
            bottom: layer.bottom,
        })
        .collect::<Vec<_>>();
    validate_layer_geometry(&source_bounds, "soil source_layers")?;
    validate_layer_geometry(&soil.target_layers, "soil target_layers")?;
    if source_bounds[0].top != 0.0 {
        return Err(RwWpsError::Config(
            "soil source_layers must begin at the 0 m surface".to_owned(),
        ));
    }
    if soil.target_layers != NOAH_SOIL_LAYERS {
        return Err(RwWpsError::Config(
            "soil target_layers differ from the selected four-layer Noah contract".to_owned(),
        ));
    }

    match &soil.remap {
        MappedSoilRemap::LinearPointSamples {
            source_value_location,
            target_value_location,
            top_anchor,
            bottom_anchor,
        } => {
            if source_value_location != "layer_bottom" || target_value_location != "layer_midpoint"
            {
                return Err(RwWpsError::Config(
                    "linear soil remap requires layer-bottom source values and layer-midpoint target values"
                        .to_owned(),
                ));
            }
            if !top_anchor.depth.is_finite()
                || !bottom_anchor.depth.is_finite()
                || top_anchor.depth != 0.0
                || top_anchor.temperature != "skin_temperature"
                || top_anchor.moisture != "repeat_shallowest"
                || bottom_anchor.depth != 3.0
                || bottom_anchor.temperature != "deep_soil_temperature"
                || bottom_anchor.moisture != "repeat_deepest"
            {
                return Err(RwWpsError::Config(
                    "soil remap anchors do not match the WRF-real/Noah surface/deep boundary contract"
                        .to_owned(),
                ));
            }
            let mut points = Vec::with_capacity(soil.source_layers.len() + 2);
            points.push(top_anchor.depth);
            points.extend(source_bounds.iter().map(|layer| layer.bottom));
            points.push(bottom_anchor.depth);
            if points.windows(2).any(|pair| pair[1] <= pair[0]) {
                return Err(RwWpsError::Config(
                    "linear soil remap sample depths are not strictly ordered".to_owned(),
                ));
            }
            let first_target = (soil.target_layers[0].top + soil.target_layers[0].bottom) / 2.0;
            let last = soil.target_layers.last().expect("non-empty target layers");
            let last_target = (last.top + last.bottom) / 2.0;
            if first_target < points[0] || last_target > *points.last().expect("anchors") {
                return Err(RwWpsError::Config(
                    "linear soil remap does not cover every target midpoint".to_owned(),
                ));
            }
        }
        MappedSoilRemap::ConservativeLayerMeans {
            source_value_location,
            target_value_location,
            coverage,
        } => {
            if source_value_location != "layer_mean"
                || target_value_location != "layer_mean"
                || coverage != "require_complete"
            {
                return Err(RwWpsError::Config(
                    "conservative soil remap requires layer means and complete coverage".to_owned(),
                ));
            }
            let source_last = source_bounds.last().expect("non-empty source layers");
            let target_last = soil.target_layers.last().expect("non-empty target layers");
            if source_bounds[0].top > soil.target_layers[0].top
                || source_last.bottom < target_last.bottom
            {
                return Err(RwWpsError::Config(
                    "conservative soil source layers do not completely cover target layers"
                        .to_owned(),
                ));
            }
        }
    }

    if soil.missing.land != "reject"
        || soil.missing.ocean.stage != "after_horizontal_interpolation"
        || soil.missing.ocean.temperature != "skin_temperature"
        || !soil.missing.ocean.moisture.is_finite()
        || soil.missing.ocean.moisture != 1.0
    {
        return Err(RwWpsError::Config(
            "soil missing policy must reject land gaps and repair ocean temperature from skin temperature with moisture 1.0 after horizontal interpolation"
                .to_owned(),
        ));
    }

    validate_soil_mapping_field(mapping, &soil.temperature_field, "K", &soil.source_layers)?;
    validate_soil_mapping_field(mapping, &soil.moisture_field, "m3 m-3", &soil.source_layers)?;
    if mapping.target.soil_layer_count.map(usize::from) != Some(soil.source_layers.len()) {
        return Err(RwWpsError::Config(
            "mapping target.soil_layer_count differs from the declarative source soil-layer count"
                .to_owned(),
        ));
    }
    Ok(())
}

fn validate_composition_bindings(
    composition: &Path,
    mapping: &mapping::NativeMapping,
    supplements: &[RolePathBinding],
    provenance: &[RolePathBinding],
) -> Result<(String, String), RwWpsError> {
    let bytes = fs::read(composition).map_err(|error| io_error(composition, error))?;
    let document = serde_json::from_slice::<MappedCompositionV2>(&bytes)?;
    if document.schema != MAPPED_COMPOSITION_CONTRACT {
        return Err(RwWpsError::Config(format!(
            "composition {} must declare schema {MAPPED_COMPOSITION_CONTRACT:?}",
            composition.display()
        )));
    }
    if document.name.trim().is_empty() {
        return Err(RwWpsError::Config(
            "composition.name must be a non-empty string".to_owned(),
        ));
    }
    if document.mapping_binding != "input_manifest_sha256" {
        return Err(RwWpsError::Config(format!(
            "composition {} must bind its mapping through input_manifest_sha256",
            composition.display()
        )));
    }
    validate_soil_contract(&document.soil_layers, mapping)?;

    let terrain = &document.supplements.terrain_height;
    validate_role(&terrain.data_role, "composition data_role")?;
    validate_role(&terrain.provenance_role, "composition provenance_role")?;
    if terrain.format != mapping.format {
        return Err(RwWpsError::Config(format!(
            "composition terrain format {:?} differs from source format {:?}",
            terrain.format.id(),
            mapping.format.id()
        )));
    }
    if terrain.field != "terrain_height"
        || terrain.selector_authority != "mapping_field_exact"
        || terrain.grid_alignment != "exact_coordinate_subset"
        || terrain.time_alignment != "valid_time_exact"
        || !terrain.require_invariant_across_time
    {
        return Err(RwWpsError::Config(
            "composition terrain supplement must declare the exact mapped-field, coordinate-subset, valid-time, invariant contract"
                .to_owned(),
        ));
    }
    let terrain_mapping = mapping.fields.get("terrain_height").ok_or_else(|| {
        RwWpsError::Config("mapping lacks the direct terrain_height field".to_owned())
    })?;
    let materialized_source_axes = terrain_mapping
        .source_axes
        .iter()
        .copied()
        .filter(|axis| !matches!(axis, AxisRole::Time | AxisRole::Member))
        .collect::<Vec<_>>();
    if terrain_mapping.derivation.is_some()
        || terrain_mapping.selectors.is_empty()
        || materialized_source_axes != [AxisRole::Y, AxisRole::X]
        || terrain_mapping.target_axes != [AxisRole::Y, AxisRole::X]
        || terrain_mapping.location != GridLocation::Surface
        || terrain_mapping.staggering != Staggering::None
        || terrain_mapping.units.target != "m"
        || !matches!(terrain_mapping.missing, MissingPolicy::Reject)
    {
        return Err(RwWpsError::Config(
            "mapping terrain_height must be direct, finite, unstaggered surface metres on y/x"
                .to_owned(),
        ));
    }
    let configured_data_roles = supplements
        .iter()
        .map(|binding| binding.role.clone())
        .collect::<BTreeSet<_>>();
    let configured_provenance_roles = provenance
        .iter()
        .map(|binding| binding.role.clone())
        .collect::<BTreeSet<_>>();
    let expected_data_roles = BTreeSet::from([terrain.data_role.clone()]);
    let expected_provenance_roles = BTreeSet::from([terrain.provenance_role.clone()]);
    if configured_data_roles != expected_data_roles {
        return Err(RwWpsError::Config(format!(
            "configured supplement roles differ from composition: expected={expected_data_roles:?} actual={configured_data_roles:?}"
        )));
    }
    if configured_provenance_roles != expected_provenance_roles {
        return Err(RwWpsError::Config(format!(
            "configured provenance roles differ from composition: expected={expected_provenance_roles:?} actual={configured_provenance_roles:?}"
        )));
    }
    Ok((terrain.data_role.clone(), terrain.provenance_role.clone()))
}

fn verify_manifest_file_binding(
    manifest_path: &Path,
    binding: &ManifestFileBinding,
    actual_path: &Path,
    label: &str,
) -> Result<(), RwWpsError> {
    if binding.path.as_os_str().is_empty() {
        return Err(RwWpsError::Config(format!(
            "{label}.path must be a non-empty path"
        )));
    }
    let declared = if binding.path.is_absolute() {
        binding.path.clone()
    } else {
        manifest_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join(&binding.path)
    };
    let declared = fs::canonicalize(&declared).map_err(|error| io_error(&declared, error))?;
    let actual = fs::canonicalize(actual_path).map_err(|error| io_error(actual_path, error))?;
    if declared != actual {
        return Err(RwWpsError::Config(format!(
            "{label}.path differs from the configured file"
        )));
    }
    let actual_bytes = actual
        .metadata()
        .map_err(|error| io_error(&actual, error))?
        .len();
    if binding.bytes != actual_bytes {
        return Err(RwWpsError::Config(format!(
            "{label}.bytes differs from the configured file"
        )));
    }
    validate_sha(&binding.sha256)?;
    let actual_sha256 = sha256_file(&actual)?;
    if !actual_sha256.eq_ignore_ascii_case(&binding.sha256) {
        return Err(RwWpsError::Config(format!(
            "{label}.sha256 differs from the configured file"
        )));
    }
    Ok(())
}

fn verify_manifest_file_inventory(
    manifest_path: &Path,
    inventory: &ManifestFileInventory,
    actual_paths: &[&Path],
    label: &str,
) -> Result<(), RwWpsError> {
    if actual_paths.is_empty() {
        return Err(RwWpsError::Config(format!(
            "{label} configured inventory is empty"
        )));
    }
    match inventory {
        ManifestFileInventory::One(binding) if actual_paths.len() == 1 => {
            verify_manifest_file_binding(manifest_path, binding, actual_paths[0], label)
        }
        ManifestFileInventory::Many(bindings) if bindings.len() == actual_paths.len() => {
            for (index, (binding, path)) in bindings.iter().zip(actual_paths).enumerate() {
                verify_manifest_file_binding(
                    manifest_path,
                    binding,
                    path,
                    &format!("{label}[{index}]"),
                )?;
            }
            Ok(())
        }
        _ => Err(RwWpsError::Config(format!(
            "{label} file inventory differs from the configured request"
        ))),
    }
}

fn mapped_decoder_paths(decoder: &MappedDecoderConfig) -> BTreeMap<&'static str, &Path> {
    match decoder {
        MappedDecoderConfig::Grib1 { bridge } => {
            BTreeMap::from([("grib1_bridge", bridge.as_path())])
        }
        MappedDecoderConfig::Grib2 { inventory, dump } => BTreeMap::from([
            ("grib2_dump", dump.as_path()),
            ("grib2_inventory", inventory.as_path()),
        ]),
        MappedDecoderConfig::Netcdf => BTreeMap::new(),
    }
}

fn verify_mapped_input_manifest(
    manifest_path: &Path,
    mapping_path: &Path,
    composition_path: &Path,
    primary_files: &[PathBuf],
    supplements: &[RolePathBinding],
    provenance: &[RolePathBinding],
    decoder: &MappedDecoderConfig,
) -> Result<(), RwWpsError> {
    let bytes = fs::read(manifest_path).map_err(|error| io_error(manifest_path, error))?;
    let manifest = serde_json::from_slice::<MappedInputManifestV1>(&bytes)?;
    if manifest.schema != MAPPED_INPUT_MANIFEST_SCHEMA {
        return Err(RwWpsError::Config(format!(
            "mapped input manifest must declare schema {MAPPED_INPUT_MANIFEST_SCHEMA:?}"
        )));
    }
    validate_sha(&manifest.mapping_sha256)?;
    validate_sha(&manifest.composition_sha256)?;
    if !sha256_file(mapping_path)?.eq_ignore_ascii_case(&manifest.mapping_sha256) {
        return Err(RwWpsError::Config(
            "mapped input manifest mapping_sha256 differs from the mapping bytes".to_owned(),
        ));
    }
    if !sha256_file(composition_path)?.eq_ignore_ascii_case(&manifest.composition_sha256) {
        return Err(RwWpsError::Config(
            "mapped input manifest composition_sha256 differs from the composition bytes"
                .to_owned(),
        ));
    }
    if manifest.primary_files.len() != primary_files.len() {
        return Err(RwWpsError::Config(
            "mapped input manifest primary file inventory differs from the configured request"
                .to_owned(),
        ));
    }
    for (index, (binding, path)) in manifest.primary_files.iter().zip(primary_files).enumerate() {
        verify_manifest_file_binding(
            manifest_path,
            binding,
            path,
            &format!("manifest.primary_files[{index}]"),
        )?;
    }

    let mut supplement_paths = BTreeMap::<String, Vec<&Path>>::new();
    for binding in supplements {
        supplement_paths
            .entry(binding.role.clone())
            .or_default()
            .push(&binding.path);
    }
    let supplement_roles = supplement_paths.keys().cloned().collect::<BTreeSet<_>>();
    let manifest_supplement_roles = manifest
        .supplements
        .keys()
        .cloned()
        .collect::<BTreeSet<_>>();
    if manifest_supplement_roles != supplement_roles {
        return Err(RwWpsError::Config(
            "mapped input manifest supplement roles differ from the configured request".to_owned(),
        ));
    }
    for (role, paths) in &supplement_paths {
        verify_manifest_file_inventory(
            manifest_path,
            &manifest.supplements[role],
            paths,
            &format!("manifest.supplements.{role}"),
        )?;
    }

    let provenance_paths = provenance
        .iter()
        .map(|binding| (binding.role.as_str(), binding.path.as_path()))
        .collect::<BTreeMap<_, _>>();
    if manifest
        .provenance
        .keys()
        .map(String::as_str)
        .collect::<BTreeSet<_>>()
        != provenance_paths.keys().copied().collect::<BTreeSet<_>>()
    {
        return Err(RwWpsError::Config(
            "mapped input manifest provenance roles differ from the configured request".to_owned(),
        ));
    }
    for (role, path) in provenance_paths {
        verify_manifest_file_binding(
            manifest_path,
            &manifest.provenance[role],
            path,
            &format!("manifest.provenance.{role}"),
        )?;
    }

    let decoder_paths = mapped_decoder_paths(decoder);
    if manifest
        .decoders
        .keys()
        .map(String::as_str)
        .collect::<BTreeSet<_>>()
        != decoder_paths.keys().copied().collect::<BTreeSet<_>>()
    {
        return Err(RwWpsError::Config(
            "mapped input manifest decoder roles differ from the configured request".to_owned(),
        ));
    }
    for (role, path) in decoder_paths {
        verify_manifest_file_binding(
            manifest_path,
            &manifest.decoders[role],
            path,
            &format!("manifest.decoders.{role}"),
        )?;
    }
    Ok(())
}

fn validate_source_and_domain(config: &RunConfig) -> Result<(), RwWpsError> {
    let domain = &config.domain;
    match &config.source {
        SourceConfig::Hrrr {
            input_manifest_sha256,
            source_root,
            valid_time,
            root_preparation,
            run_seconds,
            pipeline_workers,
            ..
        } => {
            validate_sha(input_manifest_sha256)?;
            validate_cycle(valid_time, false)?;
            if run_seconds.is_some_and(|value| value == 0 || value > 43_200) {
                return Err(RwWpsError::Config(
                    "HRRR run_seconds must be between 1 and 43200".to_owned(),
                ));
            }
            if pipeline_workers.is_some_and(|value| value == 0 || value > 13) {
                return Err(RwWpsError::Config(
                    "HRRR pipeline_workers must be between 1 and 13".to_owned(),
                ));
            }
            require(&domain.namelist_input, "domain.namelist_input")?;
            if root_preparation.is_some() {
                if source_root.is_some() {
                    return Err(RwWpsError::Config(
                        "HRRR hierarchy does not consume source.source_root".to_owned(),
                    ));
                }
                require(
                    &domain.stock_wrf_namelist_input,
                    "domain.stock_wrf_namelist_input",
                )?;
                require(&domain.root_domain_spec, "domain.root_domain_spec")?;
                require(&domain.geog_root, "domain.geog_root")?;
                require(&domain.wps_namelist, "domain.wps_namelist")?;
                if domain.static_cache.is_some() || domain.static_input.is_some() {
                    return Err(RwWpsError::Config(
                        "HRRR hierarchy cannot mix geog_root with static_cache/static_input"
                            .to_owned(),
                    ));
                }
                if domain.static_receipt.is_some() {
                    return Err(RwWpsError::Config(
                        "HRRR hierarchy owns static evidence; domain.static_receipt must be omitted"
                            .to_owned(),
                    ));
                }
                if run_seconds.is_some() || pipeline_workers.is_some() {
                    return Err(RwWpsError::Config(
                        "HRRR hierarchy does not use run_seconds or pipeline_workers".to_owned(),
                    ));
                }
            } else if domain.geog_root.is_some() {
                require(source_root, "source.source_root")?;
                require(&domain.root_domain_spec, "domain.root_domain_spec")?;
                if domain.static_cache.is_some() || domain.static_input.is_some() {
                    return Err(RwWpsError::Config(
                        "HRRR geog mode cannot mix static_cache/static_input".to_owned(),
                    ));
                }
                if domain.static_receipt.is_some() {
                    return Err(RwWpsError::Config(
                        "HRRR geog mode creates its own static evidence; domain.static_receipt must be omitted"
                            .to_owned(),
                    ));
                }
            } else {
                require(source_root, "source.source_root")?;
                require(&domain.static_cache, "domain.static_cache")?;
                require(&domain.static_receipt, "domain.static_receipt")?;
                if domain.static_input.is_some() {
                    return Err(RwWpsError::Config(
                        "HRRR static-cache mode does not consume domain.static_input".to_owned(),
                    ));
                }
            }
            if root_preparation.is_none() && domain.stock_wrf_namelist_input.is_some() {
                return Err(RwWpsError::Config(
                    "HRRR standalone routes do not consume domain.stock_wrf_namelist_input"
                        .to_owned(),
                ));
            }
            if root_preparation.is_none() && config.backend.bridge().is_some() {
                return Err(RwWpsError::Config(
                    "HRRR standalone routes do not consume backend.bridge".to_owned(),
                ));
            }
            if root_preparation.is_none() && domain.wps_namelist.is_some() {
                return Err(RwWpsError::Config(
                    "HRRR standalone routes do not consume domain.wps_namelist".to_owned(),
                ));
            }
            if domain.experiment_config.is_some() {
                return Err(RwWpsError::Config(
                    "domain.experiment_config is not used by HRRR".to_owned(),
                ));
            }
        }
        SourceConfig::Gfs {
            input_manifest_sha256,
            cycle,
            ..
        } => {
            validate_sha(input_manifest_sha256)?;
            validate_cycle(cycle, true)?;
            require(&domain.static_input, "domain.static_input")?;
            require(&domain.static_receipt, "domain.static_receipt")?;
            require(&domain.experiment_config, "domain.experiment_config")?;
            require(&domain.wps_namelist, "domain.wps_namelist")?;
            reject_hrrr_domain_options(domain, "GFS")?;
        }
        SourceConfig::Era5 {
            input_manifest_sha256,
            source_orography_variable,
            ..
        } => {
            validate_sha(input_manifest_sha256)?;
            if source_orography_variable.trim().is_empty() {
                return Err(RwWpsError::Config(
                    "ERA5 source_orography_variable is empty".to_owned(),
                ));
            }
            require(&domain.static_input, "domain.static_input")?;
            require(&domain.static_receipt, "domain.static_receipt")?;
            require(&domain.experiment_config, "domain.experiment_config")?;
            require(&domain.wps_namelist, "domain.wps_namelist")?;
            reject_hrrr_domain_options(domain, "ERA5")?;
        }
        SourceConfig::Mapped {
            contract,
            format,
            primary_files,
            supplements,
            provenance,
            decoder,
            input_manifest_sha256,
            hierarchy_workers,
            ..
        } => {
            validate_sha(input_manifest_sha256)?;
            if contract.id() != MAPPED_COMPOSITION_CONTRACT {
                return Err(RwWpsError::Config(
                    "unsupported mapped composition contract".to_owned(),
                ));
            }
            if primary_files.is_empty() {
                return Err(RwWpsError::Config(
                    "mapped source must declare at least one primary file".to_owned(),
                ));
            }
            if primary_files.iter().collect::<BTreeSet<_>>().len() != primary_files.len() {
                return Err(RwWpsError::Config(
                    "mapped primary file inventory must be unique".to_owned(),
                ));
            }
            if supplements.is_empty() || provenance.is_empty() {
                return Err(RwWpsError::Config(
                    "mapped composition requires supplement and provenance bindings".to_owned(),
                ));
            }
            let mut supplement_bindings = BTreeSet::new();
            for binding in supplements {
                validate_role(&binding.role, "mapped supplement role")?;
                if !supplement_bindings.insert((&binding.role, &binding.path)) {
                    return Err(RwWpsError::Config(format!(
                        "mapped supplement binding is duplicated for role {:?} and path {}",
                        binding.role,
                        binding.path.display()
                    )));
                }
            }
            let mut provenance_roles = BTreeSet::new();
            for binding in provenance {
                validate_role(&binding.role, "mapped provenance role")?;
                if !provenance_roles.insert(&binding.role) {
                    return Err(RwWpsError::Config(format!(
                        "mapped provenance role {:?} is bound more than once",
                        binding.role
                    )));
                }
            }
            if decoder.format() != *format {
                return Err(RwWpsError::Config(format!(
                    "mapped source format {} requires the matching decoder contract, got {}",
                    format.id(),
                    decoder.format().id()
                )));
            }
            if hierarchy_workers.is_some_and(|workers| workers == 0 || workers > 32) {
                return Err(RwWpsError::Config(
                    "mapped hierarchy_workers must be between 1 and 32".to_owned(),
                ));
            }
            require(&domain.experiment_config, "domain.experiment_config")?;
            require(&domain.wps_namelist, "domain.wps_namelist")?;
            require(&domain.geog_root, "domain.geog_root")?;
            reject_mapped_domain_options(domain)?;
        }
    }
    Ok(())
}

fn reject_hrrr_domain_options(domain: &DomainConfig, source: &str) -> Result<(), RwWpsError> {
    let unused = [
        ("namelist_input", domain.namelist_input.is_some()),
        (
            "stock_wrf_namelist_input",
            domain.stock_wrf_namelist_input.is_some(),
        ),
        ("root_domain_spec", domain.root_domain_spec.is_some()),
        ("geog_root", domain.geog_root.is_some()),
        ("static_cache", domain.static_cache.is_some()),
    ];
    if let Some((name, _)) = unused.into_iter().find(|(_, present)| *present) {
        return Err(RwWpsError::Config(format!(
            "domain.{name} is not used by {source}"
        )));
    }
    Ok(())
}

fn reject_mapped_domain_options(domain: &DomainConfig) -> Result<(), RwWpsError> {
    let unused = [
        ("namelist_input", domain.namelist_input.is_some()),
        (
            "stock_wrf_namelist_input",
            domain.stock_wrf_namelist_input.is_some(),
        ),
        ("root_domain_spec", domain.root_domain_spec.is_some()),
        ("static_cache", domain.static_cache.is_some()),
        ("static_input", domain.static_input.is_some()),
        ("static_receipt", domain.static_receipt.is_some()),
    ];
    if let Some((name, _)) = unused.into_iter().find(|(_, present)| *present) {
        return Err(RwWpsError::Config(format!(
            "domain.{name} belongs to the retired mapped static-input route; composed mapped runs require domain.geog_root"
        )));
    }
    Ok(())
}

fn require<T>(value: &Option<T>, label: &str) -> Result<(), RwWpsError> {
    if value.is_none() {
        Err(RwWpsError::Config(format!("{label} is required")))
    } else {
        Ok(())
    }
}

fn validate_sha(value: &str) -> Result<(), RwWpsError> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(RwWpsError::Config(
            "input_manifest_sha256 must be exactly 64 hexadecimal characters".to_owned(),
        ));
    }
    Ok(())
}

fn validate_role(value: &str, label: &str) -> Result<(), RwWpsError> {
    if value.is_empty()
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'.' | b'-'))
    {
        return Err(RwWpsError::Config(format!(
            "{label} {value:?} must match [A-Za-z0-9_.-]+"
        )));
    }
    Ok(())
}

fn validate_cycle(value: &str, synoptic: bool) -> Result<(), RwWpsError> {
    let bytes = value.as_bytes();
    let structural = bytes.len() == 19
        && bytes[4] == b'-'
        && bytes[7] == b'-'
        && bytes[10] == b'_'
        && bytes[13] == b':'
        && bytes[16] == b':'
        && bytes
            .iter()
            .enumerate()
            .all(|(index, byte)| matches!(index, 4 | 7 | 10 | 13 | 16) || byte.is_ascii_digit());
    if !structural {
        return Err(RwWpsError::Config(
            "cycle/time must use YYYY-MM-DD_HH:MM:SS".to_owned(),
        ));
    }
    let hour = value[11..13]
        .parse::<u8>()
        .map_err(|_| RwWpsError::Config("cycle hour is invalid".to_owned()))?;
    if &value[14..] != "00:00" || hour > 23 {
        return Err(RwWpsError::Config(
            "cycle/time must be an exact UTC hour".to_owned(),
        ));
    }
    if synoptic && !matches!(hour, 0 | 6 | 12 | 18) {
        return Err(RwWpsError::Config(
            "GFS cycle must be 00/06/12/18 UTC".to_owned(),
        ));
    }
    Ok(())
}

#[derive(Clone, Copy)]
enum PathKind {
    File,
    Directory,
}

impl PathKind {
    fn label(self) -> &'static str {
        match self {
            Self::File => "file",
            Self::Directory => "directory",
        }
    }
}

fn input_paths(config: &RunConfig) -> Vec<(&'static str, &Path, PathKind)> {
    let mut paths = Vec::new();
    macro_rules! optional_file {
        ($name:literal, $value:expr) => {
            if let Some(value) = $value.as_deref() {
                paths.push(($name, value, PathKind::File));
            }
        };
    }
    optional_file!("domain.namelist_input", config.domain.namelist_input);
    optional_file!("domain.wps_namelist", config.domain.wps_namelist);
    optional_file!(
        "domain.stock_wrf_namelist_input",
        config.domain.stock_wrf_namelist_input
    );
    optional_file!("domain.root_domain_spec", config.domain.root_domain_spec);
    optional_file!("domain.static_cache", config.domain.static_cache);
    optional_file!("domain.static_input", config.domain.static_input);
    optional_file!("domain.static_receipt", config.domain.static_receipt);
    optional_file!("domain.experiment_config", config.domain.experiment_config);
    if let Some(path) = config.domain.geog_root.as_deref() {
        paths.push(("domain.geog_root", path, PathKind::Directory));
    }
    match &config.source {
        SourceConfig::Hrrr {
            source_root,
            input_manifest,
            root_preparation,
            ..
        } => {
            if let Some(path) = source_root {
                paths.push(("source.source_root", path, PathKind::Directory));
            }
            paths.push(("source.input_manifest", input_manifest, PathKind::File));
            if let Some(path) = root_preparation {
                paths.push(("source.root_preparation", path, PathKind::Directory));
            }
        }
        SourceConfig::Gfs {
            series,
            bridge,
            input_manifest,
            ..
        } => {
            paths.push(("source.series", series, PathKind::File));
            paths.push(("source.bridge", bridge, PathKind::File));
            paths.push(("source.input_manifest", input_manifest, PathKind::File));
        }
        SourceConfig::Mapped {
            mapping,
            composition,
            primary_files,
            supplements,
            provenance,
            decoder,
            input_manifest,
            ..
        } => {
            paths.push(("source.mapping", mapping, PathKind::File));
            paths.push(("source.composition", composition, PathKind::File));
            for path in primary_files {
                paths.push(("source.primary_files", path, PathKind::File));
            }
            for binding in supplements {
                paths.push(("source.supplements", &binding.path, PathKind::File));
            }
            for binding in provenance {
                paths.push(("source.provenance", &binding.path, PathKind::File));
            }
            match decoder {
                MappedDecoderConfig::Grib1 { bridge } => {
                    paths.push(("source.decoder.grib1_bridge", bridge, PathKind::File));
                }
                MappedDecoderConfig::Grib2 { inventory, dump } => {
                    paths.push(("source.decoder.grib2_inventory", inventory, PathKind::File));
                    paths.push(("source.decoder.grib2_dump", dump, PathKind::File));
                }
                MappedDecoderConfig::Netcdf => {}
            }
            paths.push(("source.input_manifest", input_manifest, PathKind::File));
        }
        SourceConfig::Era5 {
            grib,
            vtable,
            bridge,
            source_orography,
            input_manifest,
            ..
        } => {
            paths.push(("source.grib", grib, PathKind::File));
            paths.push(("source.vtable", vtable, PathKind::File));
            paths.push(("source.bridge", bridge, PathKind::File));
            paths.push(("source.source_orography", source_orography, PathKind::File));
            paths.push(("source.input_manifest", input_manifest, PathKind::File));
        }
    }
    if let Some(path) = config.backend.bridge() {
        paths.push(("backend.bridge", path, PathKind::File));
    }
    paths
}

pub fn build_author_mapped_args(
    request: &AuthorMappedRequest,
) -> Result<Vec<OsString>, RwWpsError> {
    let decoder_format = match request.decoder {
        AuthorMappedDecoder::Grib1 { .. } => SourceFormat::Grib1,
        AuthorMappedDecoder::Grib2 { .. } => SourceFormat::Grib2,
        AuthorMappedDecoder::Netcdf => SourceFormat::Netcdf,
    };
    if decoder_format != request.format {
        return Err(RwWpsError::Config(format!(
            "author-mapped decoder format {:?} does not match source format {:?}",
            decoder_format, request.format
        )));
    }
    if request.primary_files.is_empty() {
        return Err(RwWpsError::Config(
            "author-mapped requires at least one ordered --input".to_owned(),
        ));
    }
    if request.supplements.is_empty() {
        return Err(RwWpsError::Config(
            "author-mapped requires at least one --supplement ROLE=PATH".to_owned(),
        ));
    }
    if request.provenance.is_empty() {
        return Err(RwWpsError::Config(
            "author-mapped requires at least one --provenance ROLE=PATH".to_owned(),
        ));
    }

    let mut args = vec![
        OsString::from("--source"),
        OsString::from("mapped"),
        OsString::from("--source-format"),
        OsString::from(request.format.id()),
    ];
    match &request.mapping {
        AuthorMappingInput::Existing { mapping } => {
            push_path(&mut args, "--mapping", mapping);
        }
        AuthorMappingInput::Descriptor {
            descriptor,
            vtable,
            output_mapping,
        } => {
            push_path(&mut args, "--descriptor", descriptor);
            match request.format {
                SourceFormat::Grib1 | SourceFormat::Grib2 => {
                    let vtable = vtable.as_ref().ok_or_else(|| {
                        RwWpsError::Config(
                            "--vtable is required when authoring a GRIB descriptor".to_owned(),
                        )
                    })?;
                    push_path(&mut args, "--vtable", vtable);
                }
                SourceFormat::Netcdf if vtable.is_some() => {
                    return Err(RwWpsError::Config(
                        "--vtable is forbidden when authoring a NetCDF descriptor".to_owned(),
                    ));
                }
                SourceFormat::Netcdf => {}
            }
            push_path(&mut args, "--author-mapping", output_mapping);
        }
    }
    push_path(&mut args, "--composition", &request.composition);
    for input in &request.primary_files {
        push_path(&mut args, "--input", input);
    }
    for supplement in &request.supplements {
        push_value(&mut args, "--supplement", supplement);
    }
    for provenance in &request.provenance {
        push_value(&mut args, "--provenance", provenance);
    }
    match &request.decoder {
        AuthorMappedDecoder::Grib1 { bridge } => {
            push_path(&mut args, "--bridge", bridge);
        }
        AuthorMappedDecoder::Grib2 { inventory, dump } => {
            push_path(&mut args, "--grib2-inventory", inventory);
            push_path(&mut args, "--grib2-dump", dump);
        }
        AuthorMappedDecoder::Netcdf => {}
    }
    push_path(
        &mut args,
        "--author-input-manifest",
        &request.input_manifest,
    );
    args.push(OsString::from("--author-only"));
    Ok(args)
}

pub fn plan(config: RunConfig, config_bytes: Vec<u8>) -> Result<PlannedRun, RwWpsError> {
    validate_config(&config, false)?;
    let args = build_engine_args(&config)?;
    Ok(PlannedRun {
        config_sha256: sha256_bytes(&config_bytes),
        config,
        config_bytes,
        args,
    })
}

pub fn build_engine_args(config: &RunConfig) -> Result<Vec<OsString>, RwWpsError> {
    validate_config(config, false)?;
    let mut args = vec![
        OsString::from("--source"),
        OsString::from(config.source.id()),
    ];
    let domain = &config.domain;
    let output = config.output.root();
    match &config.source {
        SourceConfig::Hrrr {
            source_root,
            input_manifest,
            input_manifest_sha256,
            valid_time,
            root_preparation,
            run_seconds,
            pipeline_workers,
        } => {
            push_path(&mut args, "--source-manifest", input_manifest);
            push_value(&mut args, "--source-manifest-sha256", input_manifest_sha256);
            push_value(&mut args, "--valid-time", valid_time);
            push_path(
                &mut args,
                "--namelist-input",
                domain.namelist_input.as_deref().expect("validated"),
            );
            if let Some(root_preparation) = root_preparation {
                push_path(&mut args, "--root-preparation", root_preparation);
                push_path(
                    &mut args,
                    "--domain-spec",
                    domain.root_domain_spec.as_deref().expect("validated"),
                );
                push_path(
                    &mut args,
                    "--wps-namelist",
                    domain.wps_namelist.as_deref().expect("validated"),
                );
                push_path(
                    &mut args,
                    "--stock-wrf-namelist-input",
                    domain
                        .stock_wrf_namelist_input
                        .as_deref()
                        .expect("validated"),
                );
                push_path(
                    &mut args,
                    "--geog-root",
                    domain.geog_root.as_deref().expect("validated"),
                );
                if let Some(workers) = config.backend.workers() {
                    push_value(&mut args, "--child-workers", workers.to_string());
                }
            } else if let Some(geog_root) = &domain.geog_root {
                push_path(
                    &mut args,
                    "--source-root",
                    source_root.as_deref().expect("validated"),
                );
                push_path(&mut args, "--geog-root", geog_root);
                push_path(
                    &mut args,
                    "--domain-spec",
                    domain.root_domain_spec.as_deref().expect("validated"),
                );
                if let Some(workers) = config.backend.workers() {
                    push_value(&mut args, "--prepare-workers", workers.to_string());
                }
            } else {
                push_path(
                    &mut args,
                    "--source-root",
                    source_root.as_deref().expect("validated"),
                );
                push_path(
                    &mut args,
                    "--static-receipt",
                    domain.static_receipt.as_deref().expect("validated"),
                );
                push_path(
                    &mut args,
                    "--static-cache",
                    domain.static_cache.as_deref().expect("validated"),
                );
                if let Some(domain_spec) = &domain.root_domain_spec {
                    push_path(&mut args, "--domain-spec", domain_spec);
                }
                if let Some(workers) = config.backend.workers() {
                    push_value(&mut args, "--prepare-workers", workers.to_string());
                }
            }
            if root_preparation.is_none() {
                if let Some(seconds) = run_seconds {
                    push_value(&mut args, "--run-seconds", seconds.to_string());
                }
                if let Some(workers) = pipeline_workers {
                    push_value(&mut args, "--pipeline-workers", workers.to_string());
                }
            }
            if let Some(bridge) = config.backend.bridge() {
                push_path(&mut args, "--cpu-preprocess-bridge", bridge);
            }
        }
        SourceConfig::Gfs {
            series,
            cycle,
            bridge,
            input_manifest,
            input_manifest_sha256,
        } => {
            push_path(&mut args, "--gfs-series", series);
            push_value(&mut args, "--cycle", cycle);
            push_path(&mut args, "--bridge", bridge);
            append_pressure_source_common(&mut args, domain, input_manifest, input_manifest_sha256);
            append_backend(&mut args, &config.backend);
        }
        SourceConfig::Era5 {
            grib,
            vtable,
            bridge,
            source_orography,
            source_orography_variable,
            input_manifest,
            input_manifest_sha256,
        } => {
            push_path(&mut args, "--grib", grib);
            push_path(&mut args, "--vtable", vtable);
            push_path(&mut args, "--bridge", bridge);
            push_path(&mut args, "--source-orography", source_orography);
            push_value(
                &mut args,
                "--source-orography-variable",
                source_orography_variable,
            );
            append_pressure_source_common(&mut args, domain, input_manifest, input_manifest_sha256);
            append_backend(&mut args, &config.backend);
        }
        SourceConfig::Mapped {
            format,
            mapping,
            composition,
            primary_files,
            supplements,
            provenance,
            decoder,
            input_manifest,
            input_manifest_sha256,
            hierarchy_workers,
            ..
        } => {
            push_value(&mut args, "--source-format", format.id());
            push_path(&mut args, "--mapping", mapping);
            push_path(&mut args, "--composition", composition);
            for path in primary_files {
                push_path(&mut args, "--input", path);
            }
            for binding in supplements {
                push_role_path(&mut args, "--supplement", binding);
            }
            for binding in provenance {
                push_role_path(&mut args, "--provenance", binding);
            }
            match decoder {
                MappedDecoderConfig::Grib1 { bridge } => {
                    push_path(&mut args, "--bridge", bridge);
                }
                MappedDecoderConfig::Grib2 { inventory, dump } => {
                    push_path(&mut args, "--grib2-inventory", inventory);
                    push_path(&mut args, "--grib2-dump", dump);
                }
                MappedDecoderConfig::Netcdf => {}
            }
            push_path(
                &mut args,
                "--wps-namelist",
                domain.wps_namelist.as_deref().expect("validated"),
            );
            push_path(
                &mut args,
                "--geog-root",
                domain.geog_root.as_deref().expect("validated"),
            );
            push_path(
                &mut args,
                "--experiment-config",
                domain.experiment_config.as_deref().expect("validated"),
            );
            push_path(&mut args, "--source-manifest", input_manifest);
            push_value(&mut args, "--source-manifest-sha256", input_manifest_sha256);
            append_backend(&mut args, &config.backend);
            if let Some(workers) = hierarchy_workers {
                push_value(&mut args, "--hierarchy-workers", workers.to_string());
            }
        }
    }
    push_path(&mut args, "--output-root", output);
    Ok(args)
}

fn append_pressure_source_common(
    args: &mut Vec<OsString>,
    domain: &DomainConfig,
    input_manifest: &Path,
    input_manifest_sha256: &str,
) {
    push_path(
        args,
        "--wps-namelist",
        domain.wps_namelist.as_deref().expect("validated"),
    );
    push_path(
        args,
        "--static-input",
        domain.static_input.as_deref().expect("validated"),
    );
    push_path(
        args,
        "--static-receipt",
        domain.static_receipt.as_deref().expect("validated"),
    );
    push_path(
        args,
        "--experiment-config",
        domain.experiment_config.as_deref().expect("validated"),
    );
    push_path(args, "--source-manifest", input_manifest);
    push_value(args, "--source-manifest-sha256", input_manifest_sha256);
}

fn append_backend(args: &mut Vec<OsString>, backend: &BackendConfig) {
    push_value(args, "--preprocess-backend", backend.id());
    if let Some(workers) = backend.workers() {
        push_value(args, "--preprocess-workers", workers.to_string());
    }
    if let Some(bridge) = backend.bridge() {
        push_path(args, "--cpu-preprocess-bridge", bridge);
    }
}

fn push_path(args: &mut Vec<OsString>, flag: &str, path: &Path) {
    args.push(OsString::from(flag));
    args.push(path.as_os_str().to_owned());
}

fn push_value(args: &mut Vec<OsString>, flag: &str, value: impl AsRef<OsStr>) {
    args.push(OsString::from(flag));
    args.push(value.as_ref().to_owned());
}

fn push_role_path(args: &mut Vec<OsString>, flag: &str, binding: &RolePathBinding) {
    args.push(OsString::from(flag));
    let mut value = OsString::from(&binding.role);
    value.push("=");
    value.push(binding.path.as_os_str());
    args.push(value);
}

pub fn discover_capabilities(engine: &OsStr) -> Result<(CapabilityManifest, Vec<u8>), RwWpsError> {
    let output = Command::new(engine)
        .arg("--list-sources")
        .output()
        .map_err(|error| io_error(PathBuf::from(engine), error))?;
    if !output.status.success() {
        return Err(RwWpsError::Engine(format!(
            "{} --list-sources exited {:?}: {}",
            Path::new(engine).display(),
            output.status.code(),
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }
    let manifest = serde_json::from_slice::<CapabilityManifest>(&output.stdout)?;
    if manifest.schema != CAPABILITY_SCHEMA {
        return Err(RwWpsError::Engine(format!(
            "capability schema {:?} is not supported; expected {CAPABILITY_SCHEMA:?}",
            manifest.schema
        )));
    }
    Ok((manifest, output.stdout))
}

pub fn validate_capability(
    manifest: &CapabilityManifest,
    source: &SourceConfig,
) -> Result<(), RwWpsError> {
    let id = source.id();
    let capability = manifest
        .sources
        .iter()
        .find(|value| value.source_id == id)
        .ok_or_else(|| RwWpsError::Engine(format!("engine does not declare source {id:?}")))?;
    if !capability.runnable || capability.status != "certified_stock_wrf" {
        return Err(RwWpsError::Engine(format!(
            "source {id:?} is not certified/runnable: status={} runnable={} {}",
            capability.status, capability.runnable, capability.notes
        )));
    }
    if matches!(source, SourceConfig::Mapped { .. })
        && capability.runner.as_deref() != Some("mapped_composition_v1")
    {
        return Err(RwWpsError::Engine(format!(
            "source {id:?} does not advertise the required mapped_composition_v1 runner: runner={:?}",
            capability.runner
        )));
    }
    Ok(())
}

fn validate_mapped_output_contract(
    target: &TargetContract,
    max_dom: usize,
    boundary_interval_seconds: u64,
) -> Result<(), RwWpsError> {
    let ceiling = usize::from(target.max_dom);
    if max_dom > ceiling {
        return Err(RwWpsError::Output(format!(
            "mapped target contract allows at most max_dom={ceiling}, but output contains {max_dom} domain(s)"
        )));
    }
    if !target.require_lateral_boundaries {
        return Err(RwWpsError::Output(
            "the current native WRF export schemas always emit lateral boundaries; a mapped target that disables them is not yet supported"
                .to_owned(),
        ));
    }
    let accepted = match target.boundary_interval_seconds.map(u64::from) {
        Some(spacing) if target.accept_boundary_interval_multiples && spacing > 0 => {
            boundary_interval_seconds > 0 && boundary_interval_seconds.is_multiple_of(spacing)
        }
        declared => declared == Some(boundary_interval_seconds),
    };
    if !accepted {
        return Err(RwWpsError::Output(format!(
            "mapped target boundary interval {:?} does not match native manifest interval {boundary_interval_seconds}",
            target.boundary_interval_seconds
        )));
    }
    Ok(())
}

pub fn run(
    engine: &OsStr,
    planned: &PlannedRun,
    receipt_path: &Path,
    mut progress: impl FnMut(&ProgressEvent),
) -> Result<RunReceipt, RwWpsError> {
    let rebound = serde_json::from_slice::<RunConfig>(&planned.config_bytes)?;
    if rebound != planned.config
        || sha256_bytes(&planned.config_bytes) != planned.config_sha256
        || build_engine_args(&planned.config)? != planned.args
    {
        return Err(RwWpsError::Config(
            "planned run was modified after configuration binding".to_owned(),
        ));
    }
    validate_config(&planned.config, true)?;
    if receipt_path.exists() {
        return Err(RwWpsError::Config(format!(
            "receipt {} already exists; choose a new path so prior evidence is preserved",
            receipt_path.display()
        )));
    }
    ensure_empty_output_root(planned.config.output.root())?;
    let resolved_engine = resolve_executable(engine)?;
    let input_artifacts = capture_run_inputs(&planned.config, &resolved_engine)?;
    let (capabilities, capability_bytes) = discover_capabilities(resolved_engine.as_os_str())?;
    validate_capability(&capabilities, &planned.config.source)?;
    let started = unix_ms();
    let timer = Instant::now();
    let mut sequence = 0_u64;
    emit(
        &mut progress,
        &mut sequence,
        "launch",
        "rw-wps",
        format!(
            "starting {} with {} backend",
            planned.config.source.id(),
            planned.config.backend.id()
        ),
    );

    let mut child = Command::new(&resolved_engine)
        .args(&planned.args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| io_error(&resolved_engine, error))?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| RwWpsError::Engine("engine stdout was not piped".to_owned()))?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| RwWpsError::Engine("engine stderr was not piped".to_owned()))?;
    let (sender, receiver) = mpsc::channel::<(&'static str, String)>();
    let stdout_sender = sender.clone();
    let stdout_thread = thread::spawn(move || {
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            if stdout_sender.send(("stdout", line)).is_err() {
                break;
            }
        }
    });
    let stderr_thread = thread::spawn(move || {
        for line in BufReader::new(stderr).lines().map_while(Result::ok) {
            if sender.send(("stderr", line)).is_err() {
                break;
            }
        }
    });
    for (stream, line) in receiver {
        let stage = classify_stage(&line);
        emit(&mut progress, &mut sequence, stage, stream, line);
    }
    let _ = stdout_thread.join();
    let _ = stderr_thread.join();
    let status = child
        .wait()
        .map_err(|error| io_error(&resolved_engine, error))?;
    let exit_code = status.code().unwrap_or(-1);
    let elapsed_ms = timer.elapsed().as_millis();
    let command = planned
        .args
        .iter()
        .map(|value| value.to_string_lossy().into_owned())
        .collect::<Vec<_>>();
    let input_error = verify_run_inputs(&input_artifacts)
        .err()
        .map(|error| error.to_string());
    if !status.success() || input_error.is_some() {
        let error = input_error.unwrap_or_else(|| format!("native engine exited {exit_code}"));
        let failure = RunReceipt {
            schema: RECEIPT_SCHEMA.to_owned(),
            verdict: "FAIL".to_owned(),
            source: planned.config.source.id().to_owned(),
            backend: planned.config.backend.id().to_owned(),
            config_sha256: planned.config_sha256.clone(),
            engine: resolved_engine.display().to_string(),
            capability_manifest_sha256: sha256_bytes(&capability_bytes),
            command,
            input_artifacts,
            started_unix_ms: started,
            elapsed_ms,
            exit_code,
            output_root: planned.config.output.root().display().to_string(),
            max_dom: 0,
            artifacts: Vec::new(),
            error: Some(error.clone()),
        };
        let failure_path = failure_receipt_path(receipt_path);
        write_json_new(&failure_path, &failure)?;
        return Err(RwWpsError::Engine(format!(
            "{error}; failure receipt {}",
            failure_path.display()
        )));
    }
    emit(
        &mut progress,
        &mut sequence,
        "verify",
        "rw-wps",
        "verifying WRF-compatible output inventory",
    );
    let output_verification: Result<(usize, u64, Vec<ArtifactReceipt>), RwWpsError> = (|| {
        let (max_dom, boundary_interval_seconds, artifacts) =
            verify_wrf_output(planned.config.output.root())?;
        if let SourceConfig::Mapped { mapping, .. } = &planned.config.source {
            let target = read_mapping(mapping)?.target;
            validate_mapped_output_contract(&target, max_dom, boundary_interval_seconds)?;
        }
        Ok((max_dom, boundary_interval_seconds, artifacts))
    })();
    let (max_dom, _boundary_interval_seconds, artifacts) = match output_verification {
        Ok(result) => result,
        Err(error) => {
            let message = error.to_string();
            let failure = RunReceipt {
                schema: RECEIPT_SCHEMA.to_owned(),
                verdict: "FAIL".to_owned(),
                source: planned.config.source.id().to_owned(),
                backend: planned.config.backend.id().to_owned(),
                config_sha256: planned.config_sha256.clone(),
                engine: resolved_engine.display().to_string(),
                capability_manifest_sha256: sha256_bytes(&capability_bytes),
                command: command.clone(),
                input_artifacts: input_artifacts.clone(),
                started_unix_ms: started,
                elapsed_ms,
                exit_code,
                output_root: planned.config.output.root().display().to_string(),
                max_dom: 0,
                artifacts: Vec::new(),
                error: Some(message.clone()),
            };
            let failure_path = failure_receipt_path(receipt_path);
            write_json_new(&failure_path, &failure)?;
            return Err(RwWpsError::Output(format!(
                "{message}; failure receipt {}",
                failure_path.display()
            )));
        }
    };
    let receipt = RunReceipt {
        schema: RECEIPT_SCHEMA.to_owned(),
        verdict: "PASS".to_owned(),
        source: planned.config.source.id().to_owned(),
        backend: planned.config.backend.id().to_owned(),
        config_sha256: planned.config_sha256.clone(),
        engine: resolved_engine.display().to_string(),
        capability_manifest_sha256: sha256_bytes(&capability_bytes),
        command,
        input_artifacts,
        started_unix_ms: started,
        elapsed_ms,
        exit_code,
        output_root: planned.config.output.root().display().to_string(),
        max_dom,
        artifacts,
        error: None,
    };
    write_json_new(receipt_path, &receipt)?;
    emit(
        &mut progress,
        &mut sequence,
        "complete",
        "rw-wps",
        format!("PASS receipt {}", receipt_path.display()),
    );
    Ok(receipt)
}

fn ensure_empty_output_root(root: &Path) -> Result<(), RwWpsError> {
    if !root.exists() {
        return Ok(());
    }
    Err(RwWpsError::Config(format!(
        "output root {} already exists; the native publishers require a new generation path",
        root.display()
    )))
}

fn emit(
    progress: &mut impl FnMut(&ProgressEvent),
    sequence: &mut u64,
    stage: impl Into<String>,
    stream: impl Into<String>,
    message: impl Into<String>,
) {
    let event = ProgressEvent {
        schema: PROGRESS_SCHEMA.to_owned(),
        sequence: *sequence,
        stage: stage.into(),
        stream: stream.into(),
        message: message.into(),
    };
    *sequence += 1;
    progress(&event);
}

fn classify_stage(line: &str) -> &'static str {
    let normalized = line.to_ascii_lowercase();
    if normalized.contains("download") || normalized.contains("fetch") {
        "fetch"
    } else if normalized.contains("static") || normalized.contains("geog") {
        "static"
    } else if normalized.contains("interpol") || normalized.contains("preprocess") {
        "preprocess"
    } else if normalized.contains("wrfinput") || normalized.contains("wrfbdy") {
        "export"
    } else if normalized.contains("error") || normalized.contains("fail") {
        "error"
    } else {
        "engine"
    }
}

pub fn verify_wrf_output(root: &Path) -> Result<(usize, u64, Vec<ArtifactReceipt>), RwWpsError> {
    if !root.is_dir() {
        return Err(RwWpsError::Output(format!(
            "output root {} is not a directory",
            root.display()
        )));
    }
    let mut files = Vec::new();
    collect_files(root, &mut files)?;
    let mut domains = BTreeMap::new();
    let mut boundary = None;
    let mut native_manifests = Vec::new();
    let mut artifacts = Vec::new();
    for path in files {
        let Some(name) = path.file_name().and_then(OsStr::to_str) else {
            continue;
        };
        let kind = if let Some(domain) = parse_domain_file(name, "wrfinput_d") {
            if let Some(previous) = domains.insert(domain, path.clone()) {
                return Err(RwWpsError::Output(format!(
                    "duplicate wrfinput_d{domain:02} files: {} and {}",
                    previous.display(),
                    path.display()
                )));
            }
            Some("wrfinput")
        } else if name == "wrfbdy_d01" {
            if let Some(previous) = boundary.replace(path.clone()) {
                return Err(RwWpsError::Output(format!(
                    "duplicate wrfbdy_d01 files: {} and {}",
                    previous.display(),
                    path.display()
                )));
            }
            Some("wrfbdy")
        } else if name == "manifest.json" {
            native_manifests.push(path.clone());
            Some("engine_manifest")
        } else if name == "receipt.json" || name.ends_with("-receipt.json") {
            Some("engine_receipt")
        } else {
            None
        };
        if let Some(kind) = kind {
            let byte_count = path
                .metadata()
                .map_err(|error| io_error(&path, error))?
                .len();
            if byte_count == 0 {
                return Err(RwWpsError::Output(format!(
                    "{kind} artifact {} is empty",
                    path.display()
                )));
            }
            if matches!(kind, "wrfinput" | "wrfbdy") {
                verify_netcdf_signature(&path)?;
            }
            artifacts.push(ArtifactReceipt {
                path: path
                    .strip_prefix(root)
                    .expect("collected under root")
                    .to_string_lossy()
                    .replace('\\', "/"),
                declared_path: None,
                byte_count,
                sha256: sha256_file(&path)?,
                kind: kind.to_owned(),
            });
        }
    }
    if boundary.is_none() {
        return Err(RwWpsError::Output("wrfbdy_d01 is absent".to_owned()));
    }
    if domains.is_empty() || !domains.contains_key(&1) {
        return Err(RwWpsError::Output("wrfinput_d01 is absent".to_owned()));
    }
    let max_dom = *domains.keys().next_back().expect("not empty");
    let expected = (1..=max_dom).collect::<BTreeSet<_>>();
    let actual = domains.keys().copied().collect::<BTreeSet<_>>();
    if actual != expected {
        return Err(RwWpsError::Output(format!(
            "wrfinput domain inventory is not contiguous: {actual:?}"
        )));
    }
    if native_manifests.len() != 1 {
        return Err(RwWpsError::Output(format!(
            "expected exactly one native WRF manifest.json, found {}",
            native_manifests.len()
        )));
    }
    let boundary_interval_seconds = verify_native_wrf_manifest(
        &native_manifests[0],
        &domains,
        boundary.as_deref().expect("checked above"),
    )?;
    artifacts.sort_by(|left, right| left.path.cmp(&right.path));
    Ok((max_dom, boundary_interval_seconds, artifacts))
}

fn verify_native_wrf_manifest(
    path: &Path,
    domains: &BTreeMap<usize, PathBuf>,
    boundary: &Path,
) -> Result<u64, RwWpsError> {
    let bytes = fs::read(path).map_err(|error| io_error(path, error))?;
    let manifest = serde_json::from_slice::<NativeWrfManifest>(&bytes).map_err(|error| {
        RwWpsError::Output(format!(
            "native WRF manifest {} is invalid: {error}",
            path.display()
        ))
    })?;
    if manifest.status != "READY" {
        return Err(RwWpsError::Output(format!(
            "native WRF manifest status must be READY, got {:?}",
            manifest.status
        )));
    }
    let max_dom = *domains.keys().next_back().expect("checked by caller");
    match manifest.schema.as_str() {
        "gpuwm-native-direct-wrf-export-v2" => {
            if max_dom != 1 || !manifest.hierarchy.is_empty() {
                return Err(RwWpsError::Output(
                    "single-domain native WRF manifest does not match output domain inventory"
                        .to_owned(),
                ));
            }
            let dimensions = manifest.dimensions.as_ref().ok_or_else(|| {
                RwWpsError::Output("single-domain native WRF manifest omits dimensions".to_owned())
            })?;
            if dimensions.nx == 0 || dimensions.ny == 0 || dimensions.nz == 0 {
                return Err(RwWpsError::Output(
                    "native WRF manifest dimensions must be positive".to_owned(),
                ));
            }
        }
        "gpuwm-native-direct-wrf-hierarchy-export-v1" => {
            let grid_ids = manifest
                .hierarchy
                .iter()
                .map(|domain| domain.grid_id)
                .collect::<Vec<_>>();
            if grid_ids != (1..=max_dom).collect::<Vec<_>>() {
                return Err(RwWpsError::Output(format!(
                    "native WRF hierarchy IDs do not match output: {grid_ids:?}"
                )));
            }
            if manifest
                .hierarchy
                .iter()
                .any(|domain| domain.nx == 0 || domain.ny == 0 || domain.nz == 0)
            {
                return Err(RwWpsError::Output(
                    "native WRF hierarchy dimensions must be positive".to_owned(),
                ));
            }
        }
        schema => {
            return Err(RwWpsError::Output(format!(
                "unsupported native WRF manifest schema {schema:?}"
            )));
        }
    }
    if manifest.boundary_interval_seconds == 0
        || manifest.boundary_record_count == 0
        || manifest.boundary_times.len() != manifest.boundary_record_count
        || manifest.next_boundary_times.len() != manifest.boundary_record_count
    {
        return Err(RwWpsError::Output(
            "native WRF manifest has inconsistent boundary metadata".to_owned(),
        ));
    }

    let mut expected_paths = domains
        .iter()
        .map(|(domain, file)| (format!("wrfinput_d{domain:02}"), file.as_path()))
        .collect::<BTreeMap<_, _>>();
    expected_paths.insert("wrfbdy_d01".to_owned(), boundary);
    let manifest_names = manifest.files.keys().cloned().collect::<BTreeSet<_>>();
    let expected_names = expected_paths.keys().cloned().collect::<BTreeSet<_>>();
    if manifest_names != expected_names {
        return Err(RwWpsError::Output(format!(
            "native WRF manifest file inventory mismatch: expected {expected_names:?}, got {manifest_names:?}"
        )));
    }
    let manifest_parent = path.parent().unwrap_or_else(|| Path::new("."));
    for (name, file) in expected_paths {
        if file.parent().unwrap_or_else(|| Path::new(".")) != manifest_parent {
            return Err(RwWpsError::Output(format!(
                "native WRF artifact {name} is not beside its manifest"
            )));
        }
        let spec = &manifest.files[&name];
        if spec.sha256.len() != 64 || !spec.sha256.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err(RwWpsError::Output(format!(
                "native WRF manifest has invalid SHA-256 for {name}"
            )));
        }
        let actual_bytes = file
            .metadata()
            .map_err(|error| io_error(file, error))?
            .len();
        let actual_sha256 = sha256_file(file)?;
        if actual_bytes != spec.bytes || !actual_sha256.eq_ignore_ascii_case(&spec.sha256) {
            return Err(RwWpsError::Output(format!(
                "native WRF manifest digest/size mismatch for {name}"
            )));
        }
    }
    Ok(manifest.boundary_interval_seconds)
}

fn verify_netcdf_signature(path: &Path) -> Result<(), RwWpsError> {
    let mut file = File::open(path).map_err(|error| io_error(path, error))?;
    let mut signature = [0_u8; 8];
    let count = file
        .read(&mut signature)
        .map_err(|error| io_error(path, error))?;
    let classic = count >= 4 && &signature[..3] == b"CDF" && matches!(signature[3], 1 | 2 | 5);
    let netcdf4 = count == signature.len() && signature == *b"\x89HDF\r\n\x1a\n";
    if !classic && !netcdf4 {
        return Err(RwWpsError::Output(format!(
            "WRF artifact {} is not a NetCDF classic/64-bit/NetCDF4 container",
            path.display()
        )));
    }
    Ok(())
}

fn parse_domain_file(name: &str, prefix: &str) -> Option<usize> {
    let suffix = name.strip_prefix(prefix)?;
    if suffix.len() != 2 || !suffix.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    suffix.parse::<usize>().ok().filter(|domain| *domain > 0)
}

fn collect_files(directory: &Path, output: &mut Vec<PathBuf>) -> Result<(), RwWpsError> {
    for entry in fs::read_dir(directory).map_err(|error| io_error(directory, error))? {
        let entry = entry.map_err(|error| io_error(directory, error))?;
        let path = entry.path();
        let kind = entry.file_type().map_err(|error| io_error(&path, error))?;
        if kind.is_symlink() {
            return Err(RwWpsError::Output(format!(
                "output tree contains symlink {}",
                path.display()
            )));
        }
        if kind.is_dir() {
            collect_files(&path, output)?;
        } else if kind.is_file() {
            output.push(path);
        }
    }
    Ok(())
}

pub fn write_json_new(path: &Path, value: &impl Serialize) -> Result<(), RwWpsError> {
    if path.exists() {
        return Err(RwWpsError::Config(format!(
            "refusing to overwrite existing evidence {}",
            path.display()
        )));
    }
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent).map_err(|error| io_error(parent, error))?;
    let temporary = parent.join(format!(
        ".{}.{}.tmp",
        path.file_name().and_then(OsStr::to_str).unwrap_or("rw-wps"),
        std::process::id()
    ));
    if temporary.exists() {
        return Err(RwWpsError::Config(format!(
            "temporary receipt path already exists: {}",
            temporary.display()
        )));
    }
    let payload = serde_json::to_vec_pretty(value)?;
    let mut file = File::create_new(&temporary).map_err(|error| io_error(&temporary, error))?;
    if let Err(error) = (|| {
        file.write_all(&payload)?;
        file.write_all(b"\n")?;
        file.sync_all()
    })() {
        let _ = fs::remove_file(&temporary);
        return Err(io_error(&temporary, error));
    }
    drop(file);
    // Publishing with a hard link is an atomic create-if-absent operation.
    // Unlike rename, it cannot replace a receipt created by another process
    // between the initial existence check and publication on Unix.
    fs::hard_link(&temporary, path).map_err(|error| {
        let _ = fs::remove_file(&temporary);
        io_error(path, error)
    })?;
    let _ = fs::remove_file(&temporary);
    #[cfg(unix)]
    File::open(parent)
        .and_then(|directory| directory.sync_all())
        .map_err(|error| io_error(parent, error))?;
    Ok(())
}

fn failure_receipt_path(receipt: &Path) -> PathBuf {
    let parent = receipt.parent().unwrap_or_else(|| Path::new("."));
    let name = receipt
        .file_name()
        .and_then(OsStr::to_str)
        .unwrap_or("receipt.json");
    parent.join(format!("{name}.failed.{}.json", std::process::id()))
}

pub fn source_templates() -> BTreeMap<&'static str, RunConfig> {
    BTreeMap::from([
        ("hrrr", template_hrrr()),
        ("gfs", template_gfs()),
        ("era5", template_era5()),
        ("mapped", template_mapped()),
    ])
}

pub fn template(source: &str) -> Result<RunConfig, RwWpsError> {
    source_templates()
        .remove(source)
        .ok_or_else(|| RwWpsError::Config(format!("unsupported template source {source:?}")))
}

fn template_domain() -> DomainConfig {
    DomainConfig {
        wps_namelist: Some(PathBuf::from("namelist.wps")),
        namelist_input: None,
        stock_wrf_namelist_input: None,
        root_domain_spec: None,
        geog_root: None,
        static_cache: None,
        static_input: Some(PathBuf::from("native-static.npz")),
        static_receipt: Some(PathBuf::from("native-static-receipt.json")),
        experiment_config: Some(PathBuf::from("experiment.toml")),
    }
}

fn template_gfs() -> RunConfig {
    RunConfig {
        schema: RUN_SCHEMA.to_owned(),
        source: SourceConfig::Gfs {
            series: PathBuf::from("gfs-series.tsv"),
            cycle: "2026-07-20_00:00:00".to_owned(),
            bridge: PathBuf::from("gfs_grib2_bridge"),
            input_manifest: PathBuf::from("input-manifest.json"),
            input_manifest_sha256: "0".repeat(64),
        },
        domain: template_domain(),
        backend: BackendConfig::Cpu {
            workers: Some(8),
            bridge: Some(PathBuf::from("libgpuwm_preprocess_cpu.so")),
        },
        output: OutputTarget::Wrf {
            root: PathBuf::from("rw-wps-output"),
        },
    }
}

fn template_era5() -> RunConfig {
    let mut config = template_gfs();
    config.source = SourceConfig::Era5 {
        grib: PathBuf::from("era5-combined.grb"),
        vtable: PathBuf::from("Vtable.ERA5_CDO"),
        bridge: PathBuf::from("grib1_bridge"),
        source_orography: PathBuf::from("era5-orography.nc"),
        source_orography_variable: default_orography_variable(),
        input_manifest: PathBuf::from("input-manifest.json"),
        input_manifest_sha256: "0".repeat(64),
    };
    config
}

fn template_hrrr() -> RunConfig {
    let mut domain = template_domain();
    domain.static_input = None;
    domain.experiment_config = None;
    domain.wps_namelist = None;
    domain.namelist_input = Some(PathBuf::from("namelist.input"));
    domain.static_cache = Some(PathBuf::from("native-static.npz"));
    RunConfig {
        schema: RUN_SCHEMA.to_owned(),
        source: SourceConfig::Hrrr {
            source_root: Some(PathBuf::from("hrrr-f00-f12")),
            input_manifest: PathBuf::from("SHA256SUMS"),
            input_manifest_sha256: "0".repeat(64),
            valid_time: "2026-07-20_00:00:00".to_owned(),
            root_preparation: None,
            run_seconds: Some(43_200),
            pipeline_workers: Some(8),
        },
        domain,
        backend: BackendConfig::Cpu {
            workers: Some(8),
            bridge: None,
        },
        output: OutputTarget::Wrf {
            root: PathBuf::from("rw-wps-output"),
        },
    }
}

fn template_mapped() -> RunConfig {
    let mut config = template_gfs();
    config.source = SourceConfig::Mapped {
        contract: MappedContract::CompositionV2,
        format: SourceFormat::Grib2,
        mapping: PathBuf::from("source-mapping.json"),
        composition: PathBuf::from("source-composition.json"),
        primary_files: vec![PathBuf::from("input.grib2")],
        supplements: vec![RolePathBinding {
            role: "terrain".to_owned(),
            path: PathBuf::from("terrain.grib2"),
        }],
        provenance: vec![RolePathBinding {
            role: "terrain_provenance".to_owned(),
            path: PathBuf::from("terrain-provenance.md"),
        }],
        decoder: MappedDecoderConfig::Grib2 {
            inventory: PathBuf::from("grib2_inventory"),
            dump: PathBuf::from("grib2_dump"),
        },
        input_manifest: PathBuf::from("input-manifest.json"),
        input_manifest_sha256: "0".repeat(64),
        hierarchy_workers: Some(8),
    };
    config.domain.geog_root = Some(PathBuf::from("WPS_GEOG"));
    config.domain.static_input = None;
    config.domain.static_receipt = None;
    config
}

fn capture_run_inputs(
    config: &RunConfig,
    resolved_engine: &Path,
) -> Result<Vec<ArtifactReceipt>, RwWpsError> {
    let mut candidates = input_paths(config)
        .into_iter()
        .filter_map(|(label, path, expected)| {
            matches!(expected, PathKind::File).then_some((label.to_owned(), path.to_owned()))
        })
        .collect::<Vec<_>>();
    candidates.push(("engine".to_owned(), resolved_engine.to_owned()));

    let mut seen = BTreeSet::new();
    let mut artifacts = Vec::new();
    for (kind, path) in candidates {
        let canonical = fs::canonicalize(&path).map_err(|error| io_error(&path, error))?;
        if !seen.insert(canonical.clone()) {
            continue;
        }
        let byte_count = canonical
            .metadata()
            .map_err(|error| io_error(&canonical, error))?
            .len();
        artifacts.push(ArtifactReceipt {
            path: canonical.to_string_lossy().into_owned(),
            declared_path: Some(path.to_string_lossy().into_owned()),
            byte_count,
            sha256: sha256_file(&canonical)?,
            kind,
        });
    }
    artifacts.sort_by(|left, right| left.path.cmp(&right.path));
    Ok(artifacts)
}

fn verify_run_inputs(artifacts: &[ArtifactReceipt]) -> Result<(), RwWpsError> {
    for artifact in artifacts {
        let path = Path::new(&artifact.path);
        let declared = artifact.declared_path.as_deref().ok_or_else(|| {
            RwWpsError::Config(format!(
                "bound input {} does not retain its declared path",
                path.display()
            ))
        })?;
        let resolved = fs::canonicalize(declared).map_err(|error| io_error(declared, error))?;
        if resolved != path {
            return Err(RwWpsError::Config(format!(
                "bound input path was retargeted during the run: {declared}"
            )));
        }
        let byte_count = path
            .metadata()
            .map_err(|error| io_error(path, error))?
            .len();
        let sha256 = sha256_file(path)?;
        if byte_count != artifact.byte_count || sha256 != artifact.sha256 {
            return Err(RwWpsError::Config(format!(
                "bound input changed during the run: {}",
                path.display()
            )));
        }
    }
    Ok(())
}

fn resolve_executable(engine: &OsStr) -> Result<PathBuf, RwWpsError> {
    let requested = PathBuf::from(engine);
    if requested.is_file() {
        return fs::canonicalize(&requested).map_err(|error| io_error(&requested, error));
    }
    if requested.components().count() > 1 {
        return Err(RwWpsError::Config(format!(
            "engine executable {} is not a regular file",
            requested.display()
        )));
    }
    let search_path = std::env::var_os("PATH").ok_or_else(|| {
        RwWpsError::Config("PATH is unset, so the engine executable cannot be bound".to_owned())
    })?;
    let names = vec![requested.clone()];
    #[cfg(windows)]
    let mut names = names;
    #[cfg(windows)]
    if requested.extension().is_none() {
        let extensions =
            std::env::var_os("PATHEXT").unwrap_or_else(|| OsString::from(".COM;.EXE;.BAT;.CMD"));
        names.extend(
            extensions
                .to_string_lossy()
                .split(';')
                .filter(|value| !value.is_empty())
                .map(|extension| {
                    let mut name = requested.as_os_str().to_os_string();
                    name.push(extension);
                    PathBuf::from(name)
                }),
        );
    }
    for directory in std::env::split_paths(&search_path) {
        for name in &names {
            let candidate = directory.join(name);
            if candidate.is_file() {
                return fs::canonicalize(&candidate).map_err(|error| io_error(&candidate, error));
            }
        }
    }
    Err(RwWpsError::Config(format!(
        "engine executable {:?} was not found on PATH",
        engine
    )))
}

fn sha256_file(path: &Path) -> Result<String, RwWpsError> {
    let mut file = File::open(path).map_err(|error| io_error(path, error))?;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let count = file
            .read(&mut buffer)
            .map_err(|error| io_error(path, error))?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(format_digest(digest.finalize()))
}

fn sha256_bytes(bytes: &[u8]) -> String {
    let mut digest = Sha256::new();
    digest.update(bytes);
    format_digest(digest.finalize())
}

fn format_digest(bytes: impl AsRef<[u8]>) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let bytes = bytes.as_ref();
    let mut value = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        value.push(HEX[(byte >> 4) as usize] as char);
        value.push(HEX[(byte & 0x0f) as usize] as char);
    }
    value
}

fn unix_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    const NETCDF_STUB: &[u8] = b"CDF\x01stub";

    fn arg_strings(config: &RunConfig) -> Vec<String> {
        build_engine_args(config)
            .unwrap()
            .into_iter()
            .map(|value| value.to_string_lossy().into_owned())
            .collect()
    }

    #[test]
    fn author_mapped_plan_preserves_explicit_paths_and_repeated_order() {
        let request = AuthorMappedRequest {
            format: SourceFormat::Grib2,
            mapping: AuthorMappingInput::Descriptor {
                descriptor: PathBuf::from("contracts/source descriptor.json"),
                vtable: Some(PathBuf::from("contracts/Vtable.HRRR")),
                output_mapping: PathBuf::from("generated/source mapping.json"),
            },
            composition: PathBuf::from("contracts/source composition.json"),
            primary_files: vec![
                PathBuf::from("inputs/f000 grib2"),
                PathBuf::from("inputs/f003 grib2"),
            ],
            supplements: vec![
                OsString::from("terrain=inputs/terrain f000.grib2"),
                OsString::from("terrain=inputs/terrain f003.grib2"),
            ],
            provenance: vec![OsString::from(
                "terrain_provenance=contracts/terrain source.md",
            )],
            decoder: AuthorMappedDecoder::Grib2 {
                inventory: PathBuf::from("bin/grib2 inventory"),
                dump: PathBuf::from("bin/grib2 dump"),
            },
            input_manifest: PathBuf::from("generated/input manifest.json"),
        };

        let args = build_author_mapped_args(&request).unwrap();
        let expected = [
            "--source",
            "mapped",
            "--source-format",
            "grib2",
            "--descriptor",
            "contracts/source descriptor.json",
            "--vtable",
            "contracts/Vtable.HRRR",
            "--author-mapping",
            "generated/source mapping.json",
            "--composition",
            "contracts/source composition.json",
            "--input",
            "inputs/f000 grib2",
            "--input",
            "inputs/f003 grib2",
            "--supplement",
            "terrain=inputs/terrain f000.grib2",
            "--supplement",
            "terrain=inputs/terrain f003.grib2",
            "--provenance",
            "terrain_provenance=contracts/terrain source.md",
            "--grib2-inventory",
            "bin/grib2 inventory",
            "--grib2-dump",
            "bin/grib2 dump",
            "--author-input-manifest",
            "generated/input manifest.json",
            "--author-only",
        ]
        .map(OsString::from)
        .to_vec();
        assert_eq!(args, expected);
    }

    #[test]
    fn author_mapped_existing_mapping_mode_stays_distinct() {
        let request = AuthorMappedRequest {
            format: SourceFormat::Netcdf,
            mapping: AuthorMappingInput::Existing {
                mapping: PathBuf::from("existing.mapping.json"),
            },
            composition: PathBuf::from("composition.json"),
            primary_files: vec![PathBuf::from("source.nc")],
            supplements: vec![OsString::from("terrain=terrain.nc")],
            provenance: vec![OsString::from("terrain_provenance=terrain.md")],
            decoder: AuthorMappedDecoder::Netcdf,
            input_manifest: PathBuf::from("inputs.json"),
        };
        let args = build_author_mapped_args(&request).unwrap();
        assert!(args.windows(2).any(|pair| {
            pair == [
                OsString::from("--mapping"),
                OsString::from("existing.mapping.json"),
            ]
        }));
        assert!(!args.iter().any(|arg| arg == "--descriptor"));
        assert!(!args.iter().any(|arg| arg == "--author-mapping"));
        assert!(!args.iter().any(|arg| arg == "--vtable"));
        assert_eq!(args.last(), Some(&OsString::from("--author-only")));
    }

    #[test]
    fn author_mapped_plan_enforces_format_aware_authorities() {
        let base = AuthorMappedRequest {
            format: SourceFormat::Grib1,
            mapping: AuthorMappingInput::Descriptor {
                descriptor: PathBuf::from("descriptor.json"),
                vtable: None,
                output_mapping: PathBuf::from("mapping.json"),
            },
            composition: PathBuf::from("composition.json"),
            primary_files: vec![PathBuf::from("source.grib1")],
            supplements: vec![OsString::from("terrain=terrain.grib1")],
            provenance: vec![OsString::from("terrain_provenance=terrain.md")],
            decoder: AuthorMappedDecoder::Grib1 {
                bridge: PathBuf::from("bridge"),
            },
            input_manifest: PathBuf::from("inputs.json"),
        };
        assert!(
            build_author_mapped_args(&base)
                .unwrap_err()
                .to_string()
                .contains("--vtable is required")
        );

        let mut netcdf = base.clone();
        netcdf.format = SourceFormat::Netcdf;
        netcdf.decoder = AuthorMappedDecoder::Netcdf;
        let AuthorMappingInput::Descriptor { vtable, .. } = &mut netcdf.mapping else {
            unreachable!()
        };
        *vtable = Some(PathBuf::from("forbidden-vtable"));
        assert!(
            build_author_mapped_args(&netcdf)
                .unwrap_err()
                .to_string()
                .contains("--vtable is forbidden")
        );

        let mut mismatched = base;
        mismatched.mapping = AuthorMappingInput::Existing {
            mapping: PathBuf::from("mapping.json"),
        };
        mismatched.decoder = AuthorMappedDecoder::Netcdf;
        assert!(
            build_author_mapped_args(&mismatched)
                .unwrap_err()
                .to_string()
                .contains("decoder format")
        );
    }

    #[test]
    fn templates_cover_initial_sources_without_claiming_more() {
        let templates = source_templates();
        assert_eq!(
            templates.keys().copied().collect::<Vec<_>>(),
            ["era5", "gfs", "hrrr", "mapped"]
        );
        for config in templates.values() {
            validate_config(config, false).unwrap();
        }
        assert!(template("rap").is_err());
    }

    #[test]
    fn gfs_cpu_plan_maps_to_certified_engine_arguments() {
        let config = template("gfs").unwrap();
        let args = arg_strings(&config);
        assert_eq!(&args[..2], ["--source", "gfs"]);
        for expected in [
            "--gfs-series",
            "--cycle",
            "--wps-namelist",
            "--static-input",
            "--static-receipt",
            "--experiment-config",
            "--preprocess-backend",
            "cpu",
            "--preprocess-workers",
            "8",
            "--cpu-preprocess-bridge",
            "--output-root",
        ] {
            assert!(
                args.iter().any(|value| value == expected),
                "missing {expected}"
            );
        }
    }

    #[test]
    fn era5_cuda_plan_omits_cpu_only_arguments() {
        let mut config = template("era5").unwrap();
        config.backend = BackendConfig::Cuda;
        let args = arg_strings(&config);
        assert!(
            args.windows(2)
                .any(|pair| pair == ["--preprocess-backend", "cuda"])
        );
        assert!(!args.iter().any(|value| value == "--preprocess-workers"));
        assert!(!args.iter().any(|value| value == "--cpu-preprocess-bridge"));
    }

    #[test]
    fn hrrr_cuda_is_fail_closed() {
        let mut config = template("hrrr").unwrap();
        config.backend = BackendConfig::Cuda;
        let error = validate_config(&config, false).unwrap_err().to_string();
        assert!(error.contains("only the certified CPU"));
    }

    #[test]
    fn hierarchy_plan_passes_topology_authorities_and_worker_bound() {
        let mut config = template("hrrr").unwrap();
        let SourceConfig::Hrrr {
            source_root,
            root_preparation,
            run_seconds,
            pipeline_workers,
            ..
        } = &mut config.source
        else {
            unreachable!()
        };
        *source_root = None;
        *root_preparation = Some(PathBuf::from("prepared-root"));
        *run_seconds = None;
        *pipeline_workers = None;
        config.domain.root_domain_spec = Some(PathBuf::from("domain.json"));
        config.domain.wps_namelist = Some(PathBuf::from("namelist.wps"));
        config.domain.geog_root = Some(PathBuf::from("WPS_GEOG"));
        config.domain.static_cache = None;
        config.domain.static_receipt = None;
        config.domain.stock_wrf_namelist_input = Some(PathBuf::from("namelist.stock.input"));
        let args = arg_strings(&config);
        for expected in [
            "--root-preparation",
            "--domain-spec",
            "--wps-namelist",
            "--namelist-input",
            "--stock-wrf-namelist-input",
            "--geog-root",
            "--child-workers",
        ] {
            assert!(
                args.iter().any(|value| value == expected),
                "missing {expected}"
            );
        }
        assert!(!args.iter().any(|value| value == "--source-root"));
        assert!(!args.iter().any(|value| value == "--static-receipt"));
        assert!(!args.iter().any(|value| value == "--run-seconds"));
        assert!(!args.iter().any(|value| value == "--pipeline-workers"));
    }

    #[test]
    fn hrrr_geog_plan_omits_standalone_static_evidence() {
        let mut config = template("hrrr").unwrap();
        config.domain.geog_root = Some(PathBuf::from("WPS_GEOG"));
        config.domain.root_domain_spec = Some(PathBuf::from("domain.json"));
        config.domain.static_cache = None;
        config.domain.static_receipt = None;
        let args = arg_strings(&config);
        assert!(args.iter().any(|value| value == "--geog-root"));
        assert!(args.iter().any(|value| value == "--domain-spec"));
        assert!(!args.iter().any(|value| value == "--static-cache"));
        assert!(!args.iter().any(|value| value == "--static-receipt"));
    }

    #[test]
    fn hrrr_static_cache_forwards_optional_domain_spec() {
        let mut config = template("hrrr").unwrap();
        config.domain.root_domain_spec = Some(PathBuf::from("domain.json"));
        let args = arg_strings(&config);
        assert!(
            args.windows(2)
                .any(|pair| pair == ["--domain-spec", "domain.json"])
        );
    }

    #[test]
    fn hrrr_rejects_route_inputs_that_would_be_ignored() {
        let mut hierarchy = template("hrrr").unwrap();
        let SourceConfig::Hrrr {
            root_preparation,
            run_seconds,
            pipeline_workers,
            ..
        } = &mut hierarchy.source
        else {
            unreachable!()
        };
        *root_preparation = Some(PathBuf::from("prepared-root"));
        *run_seconds = None;
        *pipeline_workers = None;
        hierarchy.domain.root_domain_spec = Some(PathBuf::from("domain.json"));
        hierarchy.domain.wps_namelist = Some(PathBuf::from("namelist.wps"));
        hierarchy.domain.geog_root = Some(PathBuf::from("WPS_GEOG"));
        hierarchy.domain.static_cache = None;
        hierarchy.domain.static_receipt = None;
        hierarchy.domain.stock_wrf_namelist_input = Some(PathBuf::from("namelist.stock.input"));
        assert!(validate_config(&hierarchy, false).is_err());

        let mut standalone = template("hrrr").unwrap();
        standalone.domain.stock_wrf_namelist_input = Some(PathBuf::from("namelist.stock.input"));
        assert!(validate_config(&standalone, false).is_err());
    }

    #[test]
    fn mapped_composition_plan_forwards_exact_engine_contract() {
        let config = template("mapped").unwrap();
        let args = arg_strings(&config);
        assert_eq!(
            args,
            vec![
                "--source",
                "mapped",
                "--source-format",
                "grib2",
                "--mapping",
                "source-mapping.json",
                "--composition",
                "source-composition.json",
                "--input",
                "input.grib2",
                "--supplement",
                "terrain=terrain.grib2",
                "--provenance",
                "terrain_provenance=terrain-provenance.md",
                "--grib2-inventory",
                "grib2_inventory",
                "--grib2-dump",
                "grib2_dump",
                "--wps-namelist",
                "namelist.wps",
                "--geog-root",
                "WPS_GEOG",
                "--experiment-config",
                "experiment.toml",
                "--source-manifest",
                "input-manifest.json",
                "--source-manifest-sha256",
                &"0".repeat(64),
                "--preprocess-backend",
                "cpu",
                "--preprocess-workers",
                "8",
                "--cpu-preprocess-bridge",
                "libgpuwm_preprocess_cpu.so",
                "--hierarchy-workers",
                "8",
                "--output-root",
                "rw-wps-output",
            ]
        );
        assert!(!args.iter().any(|value| value == "--static-input"));
        assert!(!args.iter().any(|value| value == "--static-receipt"));
    }

    #[test]
    fn mapped_role_binding_preserves_order_spaces_and_equals() {
        let mut config = template("mapped").unwrap();
        let SourceConfig::Mapped {
            primary_files,
            supplements,
            provenance,
            ..
        } = &mut config.source
        else {
            unreachable!()
        };
        *primary_files = vec![
            PathBuf::from("forcing files/f000.grib2"),
            PathBuf::from("forcing files/f003.grib2"),
        ];
        *supplements = vec![
            RolePathBinding {
                role: "terrain.v1".to_owned(),
                path: PathBuf::from("terrain files/analysis=a.grib2"),
            },
            RolePathBinding {
                role: "terrain.v1".to_owned(),
                path: PathBuf::from("terrain files/forecast=b.grib2"),
            },
        ];
        *provenance = vec![RolePathBinding {
            role: "terrain-v1_provenance".to_owned(),
            path: PathBuf::from("evidence files/terrain provenance.md"),
        }];
        let args = arg_strings(&config);
        let input_values = args
            .windows(2)
            .filter(|pair| pair[0] == "--input")
            .map(|pair| pair[1].clone())
            .collect::<Vec<_>>();
        let supplement_values = args
            .windows(2)
            .filter(|pair| pair[0] == "--supplement")
            .map(|pair| pair[1].clone())
            .collect::<Vec<_>>();
        assert_eq!(
            input_values,
            ["forcing files/f000.grib2", "forcing files/f003.grib2"]
        );
        assert_eq!(
            supplement_values,
            [
                "terrain.v1=terrain files/analysis=a.grib2",
                "terrain.v1=terrain files/forecast=b.grib2",
            ]
        );
        assert!(args.windows(2).any(|pair| {
            pair == [
                "--provenance",
                "terrain-v1_provenance=evidence files/terrain provenance.md",
            ]
        }));
    }

    #[test]
    fn mapped_decoder_contract_is_format_exact() {
        for (selected_format, selected_decoder, expected_flag) in [
            (
                SourceFormat::Grib1,
                MappedDecoderConfig::Grib1 {
                    bridge: PathBuf::from("grib1_bridge"),
                },
                Some("--bridge"),
            ),
            (
                SourceFormat::Grib2,
                MappedDecoderConfig::Grib2 {
                    inventory: PathBuf::from("grib2_inventory"),
                    dump: PathBuf::from("grib2_dump"),
                },
                Some("--grib2-inventory"),
            ),
            (SourceFormat::Netcdf, MappedDecoderConfig::Netcdf, None),
        ] {
            let mut config = template("mapped").unwrap();
            let SourceConfig::Mapped {
                format, decoder, ..
            } = &mut config.source
            else {
                unreachable!()
            };
            *format = selected_format;
            *decoder = selected_decoder;
            validate_config(&config, false).unwrap();
            let args = arg_strings(&config);
            if let Some(expected_flag) = expected_flag {
                assert!(args.iter().any(|value| value == expected_flag));
            } else {
                assert!(!args.iter().any(|value| {
                    matches!(
                        value.as_str(),
                        "--bridge" | "--grib2-inventory" | "--grib2-dump"
                    )
                }));
            }
        }
        let mut config = template("mapped").unwrap();
        let SourceConfig::Mapped {
            format, decoder, ..
        } = &mut config.source
        else {
            unreachable!()
        };
        *format = SourceFormat::Netcdf;
        *decoder = MappedDecoderConfig::Grib2 {
            inventory: PathBuf::from("grib2_inventory"),
            dump: PathBuf::from("grib2_dump"),
        };
        assert!(
            validate_config(&config, false)
                .unwrap_err()
                .to_string()
                .contains("matching decoder contract")
        );
    }

    #[test]
    fn mapped_domain_requires_geog_and_rejects_retired_static_authorities() {
        let mut config = template("mapped").unwrap();
        config.domain.geog_root = None;
        assert!(
            validate_config(&config, false)
                .unwrap_err()
                .to_string()
                .contains("domain.geog_root is required")
        );
        config.domain.geog_root = Some(PathBuf::from("WPS_GEOG"));
        config.domain.static_input = Some(PathBuf::from("legacy-static.npz"));
        let error = validate_config(&config, false).unwrap_err().to_string();
        assert!(error.contains("retired mapped static-input route"));
    }

    #[test]
    fn mapped_hierarchy_worker_bounds_are_fail_closed() {
        for workers in [1, 32] {
            let mut config = template("mapped").unwrap();
            if let SourceConfig::Mapped {
                hierarchy_workers, ..
            } = &mut config.source
            {
                *hierarchy_workers = Some(workers);
            }
            validate_config(&config, false).unwrap();
        }
        for workers in [0, 33] {
            let mut config = template("mapped").unwrap();
            if let SourceConfig::Mapped {
                hierarchy_workers, ..
            } = &mut config.source
            {
                *hierarchy_workers = Some(workers);
            }
            assert!(
                validate_config(&config, false)
                    .unwrap_err()
                    .to_string()
                    .contains("between 1 and 32")
            );
        }
    }

    #[test]
    fn mapped_role_and_inventory_validation_rejects_ambiguous_bindings() {
        let mut bad_role = template("mapped").unwrap();
        if let SourceConfig::Mapped { supplements, .. } = &mut bad_role.source {
            supplements[0].role = "bad role".to_owned();
        }
        assert!(validate_config(&bad_role, false).is_err());

        let mut duplicate_supplement = template("mapped").unwrap();
        if let SourceConfig::Mapped { supplements, .. } = &mut duplicate_supplement.source {
            let duplicate = supplements[0].clone();
            supplements.push(duplicate);
        }
        assert!(
            validate_config(&duplicate_supplement, false)
                .unwrap_err()
                .to_string()
                .contains("duplicated")
        );

        let mut duplicate_provenance = template("mapped").unwrap();
        if let SourceConfig::Mapped { provenance, .. } = &mut duplicate_provenance.source {
            let role = provenance[0].role.clone();
            provenance.push(RolePathBinding {
                role,
                path: PathBuf::from("other.md"),
            });
        }
        assert!(
            validate_config(&duplicate_provenance, false)
                .unwrap_err()
                .to_string()
                .contains("bound more than once")
        );

        let mut duplicate_primary = template("mapped").unwrap();
        if let SourceConfig::Mapped { primary_files, .. } = &mut duplicate_primary.source {
            primary_files.push(primary_files[0].clone());
        }
        assert!(
            validate_config(&duplicate_primary, false)
                .unwrap_err()
                .to_string()
                .contains("primary file inventory must be unique")
        );
    }

    #[test]
    fn legacy_mapped_static_input_config_fails_closed() {
        let legacy = serde_json::json!({
            "schema": RUN_SCHEMA,
            "source": {
                "kind": "mapped",
                "format": "grib2",
                "mapping": "source-mapping.json",
                "files": ["input.grib2"],
                "input_manifest": "input-manifest.json",
                "input_manifest_sha256": "0".repeat(64),
            },
            "domain": template_domain(),
            "backend": {"kind": "cpu", "workers": 8},
            "output": {"format": "wrf", "root": "output"},
        });
        let error = serde_json::from_value::<RunConfig>(legacy)
            .unwrap_err()
            .to_string();
        assert!(error.contains("unknown field `files`") || error.contains("missing field"));
    }

    #[test]
    fn mapped_template_json_round_trip_preserves_explicit_contract() {
        let template = template("mapped").unwrap();
        let bytes = serde_json::to_vec_pretty(&template).unwrap();
        let decoded = serde_json::from_slice::<RunConfig>(&bytes).unwrap();
        assert_eq!(decoded, template);
        let text = String::from_utf8(bytes).unwrap();
        assert!(text.contains(MAPPED_COMPOSITION_CONTRACT));
        assert!(text.contains("primary_files"));
        assert!(!text.contains("native-static-receipt"));
    }

    #[test]
    fn mapped_composition_roles_and_format_must_match_exactly() {
        let root = temp_root("composition-contract");
        fs::create_dir_all(&root).unwrap();
        let composition = root.join("composition.json");
        write_composition(&composition, "grib2", "terrain", "terrain_provenance");
        let (supplements, provenance) = terrain_bindings("terrain.grib2");
        let grib2_mapping = composition_mapping(SourceFormat::Grib2);
        validate_composition_bindings(&composition, &grib2_mapping, &supplements, &provenance)
            .unwrap();
        let wrong_role = vec![RolePathBinding {
            role: "orography".to_owned(),
            path: PathBuf::from("terrain.grib2"),
        }];
        assert!(
            validate_composition_bindings(&composition, &grib2_mapping, &wrong_role, &provenance,)
                .unwrap_err()
                .to_string()
                .contains("supplement roles differ")
        );
        let mut document: serde_json::Value =
            serde_json::from_slice(&fs::read(&composition).unwrap()).unwrap();
        document["supplements"]["terrain_height"]["format"] = serde_json::json!("netcdf");
        fs::write(&composition, serde_json::to_vec_pretty(&document).unwrap()).unwrap();
        assert!(
            validate_composition_bindings(&composition, &grib2_mapping, &supplements, &provenance,)
                .unwrap_err()
                .to_string()
                .contains("differs from source format")
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn mapped_composition_schema_matches_engine_strictness() {
        let root = temp_root("composition-strict-schema");
        fs::create_dir_all(&root).unwrap();
        let composition = root.join("composition.json");
        write_composition(&composition, "grib2", "terrain", "terrain_provenance");
        let (supplements, provenance) = terrain_bindings("terrain.grib2");
        let mapping = composition_mapping(SourceFormat::Grib2);
        validate_composition_bindings(&composition, &mapping, &supplements, &provenance).unwrap();

        let mut document =
            serde_json::from_slice::<serde_json::Value>(&fs::read(&composition).unwrap()).unwrap();
        document
            .as_object_mut()
            .unwrap()
            .insert("future_guess".to_owned(), serde_json::json!(true));
        fs::write(&composition, serde_json::to_vec_pretty(&document).unwrap()).unwrap();
        assert!(
            validate_composition_bindings(&composition, &mapping, &supplements, &provenance)
                .unwrap_err()
                .to_string()
                .contains("unknown field")
        );

        write_composition(&composition, "grib2", "terrain", "terrain_provenance");
        let mut document =
            serde_json::from_slice::<serde_json::Value>(&fs::read(&composition).unwrap()).unwrap();
        document["supplements"]["terrain_height"]["grid_alignment"] =
            serde_json::json!("interpolate_if_close");
        fs::write(&composition, serde_json::to_vec_pretty(&document).unwrap()).unwrap();
        assert!(
            validate_composition_bindings(&composition, &mapping, &supplements, &provenance)
                .unwrap_err()
                .to_string()
                .contains("exact mapped-field")
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn mapped_soil_contract_rejects_gap_overlap_order_units_and_ocean_policy() {
        let root = temp_root("composition-soil-adversarial");
        fs::create_dir_all(&root).unwrap();
        let composition = root.join("composition.json");
        let (supplements, provenance) = terrain_bindings("terrain.grib2");
        let mapping = composition_mapping(SourceFormat::Grib2);

        let reject = |edit: &dyn Fn(&mut serde_json::Value), expected: &str| {
            write_composition(&composition, "grib2", "terrain", "terrain_provenance");
            let mut document: serde_json::Value =
                serde_json::from_slice(&fs::read(&composition).unwrap()).unwrap();
            edit(&mut document);
            fs::write(&composition, serde_json::to_vec_pretty(&document).unwrap()).unwrap();
            let error =
                validate_composition_bindings(&composition, &mapping, &supplements, &provenance)
                    .unwrap_err()
                    .to_string();
            assert!(
                error.contains(expected),
                "expected {expected:?} in {error:?}"
            );
        };

        reject(
            &|document| {
                document["soil_layers"]["source_layers"][1]["top"] = serde_json::json!(0.11);
            },
            "gap",
        );
        reject(
            &|document| {
                document["soil_layers"]["source_layers"][1]["top"] = serde_json::json!(0.09);
            },
            "overlap",
        );
        reject(
            &|document| {
                document["soil_layers"]["source_layers"]
                    .as_array_mut()
                    .unwrap()
                    .swap(0, 1);
            },
            "ordered",
        );
        reject(
            &|document| {
                document["soil_layers"]["depth_units"] = serde_json::json!("cm");
            },
            "depth_units",
        );
        reject(
            &|document| {
                document["soil_layers"]["missing"]["ocean"]["moisture"] = serde_json::json!(0.0);
            },
            "missing policy",
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn mapped_soil_contract_binds_all_selector_identities_to_depths() {
        let root = temp_root("composition-soil-selector-order");
        fs::create_dir_all(&root).unwrap();
        let composition = root.join("composition.json");
        let (supplements, provenance) = terrain_bindings("terrain.source");
        for (format, format_name) in [
            (SourceFormat::Grib1, "grib1"),
            (SourceFormat::Grib2, "grib2"),
            (SourceFormat::Netcdf, "netcdf"),
        ] {
            write_composition(&composition, format_name, "terrain", "terrain_provenance");
            let mut mapping = composition_mapping(format);
            mapping
                .fields
                .get_mut("soil_temperature")
                .unwrap()
                .selectors
                .swap(0, 1);
            let error =
                validate_composition_bindings(&composition, &mapping, &supplements, &provenance)
                    .unwrap_err()
                    .to_string();
            assert!(
                error.contains("selector 0 differs") && error.contains("declared soil depth"),
                "unexpected {format_name} error: {error}"
            );
        }
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn mapped_soil_contract_rejects_one_selector_for_multiple_grib_layers() {
        let root = temp_root("composition-soil-selector-count");
        fs::create_dir_all(&root).unwrap();
        let composition = root.join("composition.json");
        write_composition(&composition, "grib2", "terrain", "terrain_provenance");
        let (supplements, provenance) = terrain_bindings("terrain.grib2");
        let mut mapping = composition_mapping(SourceFormat::Grib2);
        mapping
            .fields
            .get_mut("soil_temperature")
            .unwrap()
            .selectors
            .truncate(1);
        let error =
            validate_composition_bindings(&composition, &mapping, &supplements, &provenance)
                .unwrap_err()
                .to_string();
        assert!(error.contains("requires exactly 4 ordered direct selectors"));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn linear_point_soil_contract_is_source_named_independent() {
        let root = temp_root("composition-soil-linear");
        fs::create_dir_all(&root).unwrap();
        let composition = root.join("composition.json");
        write_composition(&composition, "netcdf", "terrain", "terrain_provenance");
        let mut document: serde_json::Value =
            serde_json::from_slice(&fs::read(&composition).unwrap()).unwrap();
        for (layer, (top, bottom)) in document["soil_layers"]["source_layers"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .zip([(0.0, 0.07), (0.07, 0.28), (0.28, 1.0), (1.0, 2.89)])
        {
            layer["top"] = serde_json::json!(top);
            layer["bottom"] = serde_json::json!(bottom);
        }
        document["soil_layers"]["remap"] = serde_json::json!({
            "kind": "linear_point_samples",
            "source_value_location": "layer_bottom",
            "target_value_location": "layer_midpoint",
            "top_anchor": {
                "depth": 0.0,
                "temperature": "skin_temperature",
                "moisture": "repeat_shallowest"
            },
            "bottom_anchor": {
                "depth": 3.0,
                "temperature": "deep_soil_temperature",
                "moisture": "repeat_deepest"
            }
        });
        fs::write(&composition, serde_json::to_vec_pretty(&document).unwrap()).unwrap();
        let (supplements, provenance) = terrain_bindings("terrain.nc");
        validate_composition_bindings(
            &composition,
            &composition_mapping(SourceFormat::Netcdf),
            &supplements,
            &provenance,
        )
        .unwrap();
        let text = fs::read_to_string(&composition).unwrap();
        assert!(!text.contains("era5") && !text.contains("gfs") && !text.contains("packing"));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn mapped_inspection_union_includes_supplements_and_deduplicates_reuse() {
        let root = temp_root("mapped-inspection-union");
        fs::create_dir_all(&root).unwrap();
        let primary = root.join("primary.grib2");
        let terrain = root.join("terrain.grib2");
        fs::write(&primary, b"primary").unwrap();
        fs::write(&terrain, b"terrain").unwrap();
        let mut supplements = vec![
            RolePathBinding {
                role: "terrain".to_owned(),
                path: primary.clone(),
            },
            RolePathBinding {
                role: "terrain".to_owned(),
                path: terrain.clone(),
            },
        ];
        let union = mapped_inspection_paths(std::slice::from_ref(&primary), &supplements).unwrap();
        assert_eq!(
            union,
            [
                fs::canonicalize(&primary).unwrap(),
                fs::canonicalize(&terrain).unwrap(),
            ]
        );
        supplements.push(RolePathBinding {
            role: "terrain".to_owned(),
            path: terrain.clone(),
        });
        assert!(
            mapped_inspection_paths(std::slice::from_ref(&primary), &supplements)
                .unwrap_err()
                .to_string()
                .contains("duplicate resolved path")
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn mapped_input_capture_inventory_binds_every_composed_authority() {
        let config = template("mapped").unwrap();
        let labels = input_paths(&config)
            .into_iter()
            .map(|(label, _, _)| label)
            .collect::<BTreeSet<_>>();
        for expected in [
            "domain.wps_namelist",
            "domain.experiment_config",
            "source.mapping",
            "source.composition",
            "source.primary_files",
            "source.supplements",
            "source.provenance",
            "source.decoder.grib2_inventory",
            "source.decoder.grib2_dump",
            "source.input_manifest",
            "backend.bridge",
        ] {
            assert!(labels.contains(expected), "missing {expected}");
        }
        assert!(!labels.contains("domain.static_input"));
        assert!(!labels.contains("domain.static_receipt"));
    }

    #[test]
    fn mapped_input_manifest_binds_exact_roles_order_bytes_and_digests() {
        let root = temp_root("mapped-input-manifest");
        fs::create_dir_all(&root).unwrap();
        let mapping = root.join("mapping.json");
        let composition = root.join("composition.json");
        let primary = root.join("primary.grib2");
        let terrain_a = root.join("terrain-a.grib2");
        let terrain_b = root.join("terrain-b.grib2");
        let provenance_path = root.join("terrain.md");
        let inventory = root.join("grib2_inventory");
        let dump = root.join("grib2_dump");
        for (path, bytes) in [
            (&mapping, b"mapping".as_slice()),
            (&composition, b"composition".as_slice()),
            (&primary, b"primary".as_slice()),
            (&terrain_a, b"terrain-a".as_slice()),
            (&terrain_b, b"terrain-b".as_slice()),
            (&provenance_path, b"provenance".as_slice()),
            (&inventory, b"inventory".as_slice()),
            (&dump, b"dump".as_slice()),
        ] {
            fs::write(path, bytes).unwrap();
        }
        let manifest_path = root.join("manifest.json");
        let manifest = serde_json::json!({
            "schema": MAPPED_INPUT_MANIFEST_SCHEMA,
            "mapping_sha256": sha256_file(&mapping).unwrap(),
            "composition_sha256": sha256_file(&composition).unwrap(),
            "primary_files": [manifest_file_json(&primary)],
            "supplements": {
                "terrain": [manifest_file_json(&terrain_a), manifest_file_json(&terrain_b)]
            },
            "provenance": {
                "terrain_provenance": manifest_file_json(&provenance_path)
            },
            "decoders": {
                "grib2_inventory": manifest_file_json(&inventory),
                "grib2_dump": manifest_file_json(&dump)
            }
        });
        fs::write(
            &manifest_path,
            serde_json::to_vec_pretty(&manifest).unwrap(),
        )
        .unwrap();
        let supplements = vec![
            RolePathBinding {
                role: "terrain".to_owned(),
                path: terrain_a.clone(),
            },
            RolePathBinding {
                role: "terrain".to_owned(),
                path: terrain_b.clone(),
            },
        ];
        let provenance = vec![RolePathBinding {
            role: "terrain_provenance".to_owned(),
            path: provenance_path.clone(),
        }];
        let decoder = MappedDecoderConfig::Grib2 {
            inventory: inventory.clone(),
            dump: dump.clone(),
        };
        verify_mapped_input_manifest(
            &manifest_path,
            &mapping,
            &composition,
            std::slice::from_ref(&primary),
            &supplements,
            &provenance,
            &decoder,
        )
        .unwrap();

        let mut wrong_order = manifest;
        wrong_order["supplements"]["terrain"]
            .as_array_mut()
            .unwrap()
            .swap(0, 1);
        fs::write(
            &manifest_path,
            serde_json::to_vec_pretty(&wrong_order).unwrap(),
        )
        .unwrap();
        assert!(
            verify_mapped_input_manifest(
                &manifest_path,
                &mapping,
                &composition,
                std::slice::from_ref(&primary),
                &supplements,
                &provenance,
                &decoder,
            )
            .unwrap_err()
            .to_string()
            .contains("differs from the configured file")
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn mapped_output_domain_count_uses_mapping_max_dom_as_a_ceiling() {
        let mut target = mapping::mapping_template(SourceFormat::Grib2).target;
        target.max_dom = 4;
        target.require_lateral_boundaries = true;
        target.boundary_interval_seconds = Some(3600);
        for domains in [1, 2, 4] {
            validate_mapped_output_contract(&target, domains, 3600).unwrap();
        }
        assert!(
            validate_mapped_output_contract(&target, 5, 3600)
                .unwrap_err()
                .to_string()
                .contains("at most max_dom=4")
        );
        assert!(validate_mapped_output_contract(&target, 2, 10_800).is_err());
        target.accept_boundary_interval_multiples = true;
        validate_mapped_output_contract(&target, 2, 10_800).unwrap();
        validate_mapped_output_contract(&target, 2, 3600).unwrap();
        assert!(validate_mapped_output_contract(&target, 2, 5400).is_err());
    }

    #[test]
    fn capability_gate_accepts_certified_composed_mapped_runner() {
        let manifest = CapabilityManifest {
            schema: CAPABILITY_SCHEMA.to_owned(),
            runnable_source_count: 1,
            sources: vec![SourceCapability {
                source_id: "mapped".to_owned(),
                status: "certified_stock_wrf".to_owned(),
                runnable: true,
                runner: Some("mapped_composition_v1".to_owned()),
                notes: "strict composed route".to_owned(),
            }],
        };
        validate_capability(&manifest, &template("mapped").unwrap().source).unwrap();
    }

    #[test]
    fn capability_gate_rejects_wrong_mapped_runner() {
        let manifest = CapabilityManifest {
            schema: CAPABILITY_SCHEMA.to_owned(),
            runnable_source_count: 1,
            sources: vec![SourceCapability {
                source_id: "mapped".to_owned(),
                status: "certified_stock_wrf".to_owned(),
                runnable: true,
                runner: Some("legacy_mapped_runner".to_owned()),
                notes: "wrong argv contract".to_owned(),
            }],
        };
        let error = validate_capability(&manifest, &template("mapped").unwrap().source)
            .unwrap_err()
            .to_string();
        assert!(error.contains("mapped_composition_v1"));
    }

    #[test]
    fn capability_gate_rejects_readable_but_uncertified_source() {
        let manifest = CapabilityManifest {
            schema: CAPABILITY_SCHEMA.to_owned(),
            runnable_source_count: 0,
            sources: vec![SourceCapability {
                source_id: "gfs".to_owned(),
                status: "adapter_mapping_required".to_owned(),
                runnable: false,
                runner: None,
                notes: "not stock-WRF gated".to_owned(),
            }],
        };
        let source = template("gfs").unwrap().source;
        assert!(
            validate_capability(&manifest, &source)
                .unwrap_err()
                .to_string()
                .contains("not certified/runnable")
        );
    }

    #[test]
    fn wrf_output_receipt_requires_contiguous_domains_and_boundary() {
        let root = temp_root("output-ok");
        fs::create_dir_all(root.join("generation")).unwrap();
        fs::write(root.join("generation/wrfinput_d01"), NETCDF_STUB).unwrap();
        fs::write(root.join("generation/wrfinput_d02"), NETCDF_STUB).unwrap();
        fs::write(root.join("generation/wrfbdy_d01"), NETCDF_STUB).unwrap();
        write_native_manifest(&root.join("generation"), 2);
        let (max_dom, boundary_interval_seconds, artifacts) = verify_wrf_output(&root).unwrap();
        assert_eq!(max_dom, 2);
        assert_eq!(boundary_interval_seconds, 3600);
        assert_eq!(artifacts.len(), 4);
        assert!(artifacts.windows(2).all(|pair| pair[0].path < pair[1].path));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn wrf_output_receipt_rejects_domain_gap() {
        let root = temp_root("output-gap");
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("wrfinput_d01"), NETCDF_STUB).unwrap();
        fs::write(root.join("wrfinput_d03"), NETCDF_STUB).unwrap();
        fs::write(root.join("wrfbdy_d01"), NETCDF_STUB).unwrap();
        assert!(
            verify_wrf_output(&root)
                .unwrap_err()
                .to_string()
                .contains("not contiguous")
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn wrf_output_receipt_rejects_empty_named_artifact() {
        let root = temp_root("output-empty");
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("wrfinput_d01"), []).unwrap();
        fs::write(root.join("wrfbdy_d01"), NETCDF_STUB).unwrap();
        assert!(
            verify_wrf_output(&root)
                .unwrap_err()
                .to_string()
                .contains("is empty")
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn wrf_output_receipt_rejects_non_netcdf_named_artifact() {
        let root = temp_root("output-not-netcdf");
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("wrfinput_d01"), b"not actually NetCDF").unwrap();
        fs::write(root.join("wrfbdy_d01"), NETCDF_STUB).unwrap();
        assert!(
            verify_wrf_output(&root)
                .unwrap_err()
                .to_string()
                .contains("not a NetCDF")
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn wrf_output_receipt_recomputes_native_manifest_digests() {
        let root = temp_root("output-manifest-drift");
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("wrfinput_d01"), NETCDF_STUB).unwrap();
        fs::write(root.join("wrfbdy_d01"), NETCDF_STUB).unwrap();
        write_native_manifest(&root, 1);
        fs::write(root.join("wrfbdy_d01"), b"CDF\x01changed").unwrap();
        assert!(
            verify_wrf_output(&root)
                .unwrap_err()
                .to_string()
                .contains("digest/size mismatch")
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn receipt_writer_never_replaces_existing_evidence() {
        let root = temp_root("receipt");
        fs::create_dir_all(&root).unwrap();
        let path = root.join("receipt.json");
        write_json_new(&path, &serde_json::json!({"generation": 1})).unwrap();
        let previous = fs::read(&path).unwrap();
        assert!(write_json_new(&path, &serde_json::json!({"generation": 2})).is_err());
        assert_eq!(fs::read(&path).unwrap(), previous);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn manifest_digest_is_verified_not_just_well_formed() {
        let root = temp_root("manifest-digest");
        fs::create_dir_all(&root).unwrap();
        let manifest = root.join("manifest.json");
        fs::write(&manifest, b"bound-inputs\n").unwrap();
        let mut source = template("gfs").unwrap().source;
        if let SourceConfig::Gfs { input_manifest, .. } = &mut source {
            *input_manifest = manifest.clone();
        } else {
            unreachable!();
        }
        assert!(verify_source_manifest(&source).is_err());
        if let SourceConfig::Gfs {
            input_manifest_sha256,
            ..
        } = &mut source
        {
            *input_manifest_sha256 = sha256_file(&manifest).unwrap();
        }
        verify_source_manifest(&source).unwrap();
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn bound_input_mutation_is_detected() {
        let root = temp_root("bound-input");
        fs::create_dir_all(&root).unwrap();
        let path = root.join("namelist.input");
        fs::write(&path, b"version one").unwrap();
        let canonical = fs::canonicalize(&path).unwrap();
        let artifact = ArtifactReceipt {
            path: canonical.to_string_lossy().into_owned(),
            declared_path: Some(path.to_string_lossy().into_owned()),
            byte_count: path.metadata().unwrap().len(),
            sha256: sha256_file(&path).unwrap(),
            kind: "namelist_input".to_owned(),
        };
        verify_run_inputs(std::slice::from_ref(&artifact)).unwrap();
        fs::write(&path, b"version two").unwrap();
        assert!(verify_run_inputs(&[artifact]).is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn stale_output_generation_is_rejected() {
        let root = temp_root("stale-output");
        ensure_empty_output_root(&root).unwrap();
        fs::create_dir_all(&root).unwrap();
        assert!(
            ensure_empty_output_root(&root)
                .unwrap_err()
                .to_string()
                .contains("already exists")
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn duplicate_wrf_artifacts_are_rejected() {
        let root = temp_root("duplicate-output");
        fs::create_dir_all(root.join("generation-a")).unwrap();
        fs::create_dir_all(root.join("generation-b")).unwrap();
        fs::write(root.join("generation-a/wrfinput_d01"), NETCDF_STUB).unwrap();
        fs::write(root.join("generation-b/wrfinput_d01"), NETCDF_STUB).unwrap();
        fs::write(root.join("wrfbdy_d01"), NETCDF_STUB).unwrap();
        assert!(
            verify_wrf_output(&root)
                .unwrap_err()
                .to_string()
                .contains("duplicate wrfinput")
        );
        fs::remove_dir_all(root).unwrap();
    }

    fn terrain_bindings(data_path: &str) -> (Vec<RolePathBinding>, Vec<RolePathBinding>) {
        (
            vec![RolePathBinding {
                role: "terrain".to_owned(),
                path: PathBuf::from(data_path),
            }],
            vec![RolePathBinding {
                role: "terrain_provenance".to_owned(),
                path: PathBuf::from("terrain.md"),
            }],
        )
    }

    fn composition_mapping(format: SourceFormat) -> mapping::NativeMapping {
        let mut mapping = mapping::mapping_template(format);
        let mut terrain = mapping.fields["air_temperature"].clone();
        terrain.units.source = "m".to_owned();
        terrain.units.target = "m".to_owned();
        terrain.source_axes = vec![AxisRole::Y, AxisRole::X];
        terrain.target_axes = vec![AxisRole::Y, AxisRole::X];
        terrain.location = GridLocation::Surface;
        terrain.staggering = Staggering::None;
        terrain.missing = MissingPolicy::Reject;
        mapping.fields.insert("terrain_height".to_owned(), terrain);
        let bounds = [(0.0, 0.1), (0.1, 0.4), (0.4, 1.0), (1.0, 2.0)];
        for (name, units, moisture) in [
            ("soil_temperature", "K", false),
            ("volumetric_soil_moisture", "m3 m-3", true),
        ] {
            let mut soil = mapping.fields["air_temperature"].clone();
            soil.units.source = units.to_owned();
            soil.units.target = units.to_owned();
            soil.source_axes = vec![AxisRole::Soil, AxisRole::Y, AxisRole::X];
            soil.target_axes = vec![AxisRole::Soil, AxisRole::Y, AxisRole::X];
            soil.location = GridLocation::Soil;
            soil.staggering = Staggering::None;
            soil.missing = MissingPolicy::PreserveMask;
            soil.selector_stack_axis = if format == SourceFormat::Netcdf {
                Some(AxisRole::Soil)
            } else {
                None
            };
            soil.selectors = bounds
                .iter()
                .enumerate()
                .map(|(index, (top, bottom))| match format {
                    SourceFormat::Grib1 => VariableSelector::Grib1 {
                        parameter: u8::try_from(index + if moisture { 40 } else { 140 }).unwrap(),
                        table_version: Some(128),
                        center: Some(98),
                        level_type: Some(1),
                        level_value: Some(0.0),
                    },
                    SourceFormat::Grib2 => VariableSelector::Grib2 {
                        discipline: 2,
                        category: 0,
                        parameter: if moisture { 192 } else { 2 },
                        center: None,
                        subcenter: None,
                        master_table_version: None,
                        local_table_version: None,
                        level_type: Some(106),
                        level_value: Some(*top),
                        second_level_type: Some(106),
                        second_level_value: Some(*bottom),
                        member: None,
                    },
                    SourceFormat::Netcdf => VariableSelector::Netcdf {
                        name: Some(mapping::NameSpec::One(format!(
                            "{}{}",
                            if moisture { "swvl" } else { "stl" },
                            index + 1
                        ))),
                        standard_name: None,
                    },
                })
                .collect();
            mapping.fields.insert(name.to_owned(), soil);
        }
        mapping.target.soil_layer_count = Some(4);
        mapping
    }

    fn manifest_file_json(path: &Path) -> serde_json::Value {
        serde_json::json!({
            "path": path,
            "bytes": path.metadata().unwrap().len(),
            "sha256": sha256_file(path).unwrap(),
        })
    }

    fn write_composition(path: &Path, format: &str, data_role: &str, provenance_role: &str) {
        let bounds = [(0.0, 0.1), (0.1, 0.4), (0.4, 1.0), (1.0, 2.0)];
        let source_layers = bounds
            .iter()
            .enumerate()
            .map(|(index, (top, bottom))| {
                let (temperature, moisture) = match format {
                    "grib1" => (
                        serde_json::json!({
                            "format": "grib1",
                            "parameter": index + 140,
                            "table_version": 128,
                            "center": 98,
                            "level_type": 1,
                            "level_value": 0.0
                        }),
                        serde_json::json!({
                            "format": "grib1",
                            "parameter": index + 40,
                            "table_version": 128,
                            "center": 98,
                            "level_type": 1,
                            "level_value": 0.0
                        }),
                    ),
                    "grib2" => (
                        serde_json::json!({
                            "format": "grib2",
                            "discipline": 2,
                            "category": 0,
                            "parameter": 2,
                            "level_type": 106,
                            "level_value": top,
                            "second_level_type": 106,
                            "second_level_value": bottom
                        }),
                        serde_json::json!({
                            "format": "grib2",
                            "discipline": 2,
                            "category": 0,
                            "parameter": 192,
                            "level_type": 106,
                            "level_value": top,
                            "second_level_type": 106,
                            "second_level_value": bottom
                        }),
                    ),
                    "netcdf" => (
                        serde_json::json!({
                            "format": "netcdf",
                            "name": format!("stl{}", index + 1)
                        }),
                        serde_json::json!({
                            "format": "netcdf",
                            "name": format!("swvl{}", index + 1)
                        }),
                    ),
                    _ => panic!("unsupported test composition format {format}"),
                };
                serde_json::json!({
                    "top": top,
                    "bottom": bottom,
                    "selectors": {
                        "soil_temperature": temperature,
                        "volumetric_soil_moisture": moisture
                    }
                })
            })
            .collect::<Vec<_>>();
        fs::write(
            path,
            serde_json::to_vec_pretty(&serde_json::json!({
                "schema": MAPPED_COMPOSITION_CONTRACT,
                "name": "test composition",
                "mapping_binding": "input_manifest_sha256",
                "soil_layers": {
                    "temperature_field": "soil_temperature",
                    "moisture_field": "volumetric_soil_moisture",
                    "depth_units": "m",
                    "source_layers": source_layers,
                    "target_layers": [
                        {"top": 0.0, "bottom": 0.1},
                        {"top": 0.1, "bottom": 0.4},
                        {"top": 0.4, "bottom": 1.0},
                        {"top": 1.0, "bottom": 2.0}
                    ],
                    "remap": {
                        "kind": "conservative_layer_means",
                        "source_value_location": "layer_mean",
                        "target_value_location": "layer_mean",
                        "coverage": "require_complete"
                    },
                    "missing": {
                        "land": "reject",
                        "ocean": {
                            "stage": "after_horizontal_interpolation",
                            "temperature": "skin_temperature",
                            "moisture": 1.0
                        }
                    }
                },
                "supplements": {
                    "terrain_height": {
                        "data_role": data_role,
                        "provenance_role": provenance_role,
                        "format": format,
                        "field": "terrain_height",
                        "selector_authority": "mapping_field_exact",
                        "grid_alignment": "exact_coordinate_subset",
                        "time_alignment": "valid_time_exact",
                        "require_invariant_across_time": true
                    }
                }
            }))
            .unwrap(),
        )
        .unwrap();
    }

    fn write_native_manifest(directory: &Path, max_dom: usize) {
        let mut files = serde_json::Map::new();
        for name in (1..=max_dom)
            .map(|domain| format!("wrfinput_d{domain:02}"))
            .chain(std::iter::once("wrfbdy_d01".to_owned()))
        {
            let path = directory.join(&name);
            files.insert(
                name,
                serde_json::json!({
                    "bytes": path.metadata().unwrap().len(),
                    "sha256": sha256_file(&path).unwrap(),
                }),
            );
        }
        let (schema, hierarchy, dimensions) = if max_dom == 1 {
            (
                "gpuwm-native-direct-wrf-export-v2",
                serde_json::json!([]),
                serde_json::json!({"nx": 10, "ny": 10, "nz": 49}),
            )
        } else {
            (
                "gpuwm-native-direct-wrf-hierarchy-export-v1",
                serde_json::Value::Array(
                    (1..=max_dom)
                        .map(|grid_id| {
                            serde_json::json!({
                                "grid_id": grid_id,
                                "nx": 10,
                                "ny": 10,
                                "nz": 49,
                            })
                        })
                        .collect(),
                ),
                serde_json::Value::Null,
            )
        };
        fs::write(
            directory.join("manifest.json"),
            serde_json::to_vec_pretty(&serde_json::json!({
                "schema": schema,
                "status": "READY",
                "boundary_interval_seconds": 3600,
                "boundary_record_count": 2,
                "boundary_times": ["t0", "t1"],
                "next_boundary_times": ["t1", "t2"],
                "files": files,
                "dimensions": dimensions,
                "hierarchy": hierarchy,
            }))
            .unwrap(),
        )
        .unwrap();
    }

    fn temp_root(label: &str) -> PathBuf {
        // Parallel tests read one clock tick on the 2.8.6 windows-2025 runner and collided on this name; the counter keeps each call distinct.
        static NEXT_SCRATCH: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        std::env::temp_dir().join(format!(
            "rw-wps-{label}-{}-{}-{}",
            std::process::id(),
            unix_ms(),
            NEXT_SCRATCH.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
        ))
    }
}
