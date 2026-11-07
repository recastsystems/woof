use rustwx_core::{
    CanonicalBundleDescriptor, CanonicalDataFamily, CanonicalField, CycleSpec, FieldProduct,
    FieldSelector, ModelId, ModelRunRequest, ProbabilitySelection, ProductKeyMetadata,
    ProductLineage, ProductMaturity, ProductProvenance, ProductSemanticFlag, ProductWindowSpec,
    ResolvedUrl, RustwxError, SourceId, StatisticalProcess, VerticalSelector,
};
use serde::{Deserialize, Serialize};
use std::fmt;
mod members;
mod archives;
pub use members::{DeclaredMember, DeclaredRewrite, declared_member, declared_members, selected_member_product, product_member, member_url};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProductFamily {
    Surface,
    Pressure,
    Native,
    Subhourly,
}

/// Broad runtime family used by the model-compatibility layer. The
/// renderer still consumes canonical selectors, but this tells future
/// model integrations which adapter style to implement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelRuntimeFamily {
    Grib2Forecast,
    LocalNetcdfForecast,
    WrfNetcdfArchive,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EnsembleMode {
    Deterministic,
    MemberGribFiles,
    MemberDimensionNetcdf,
}

impl ProductFamily {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Surface => "surface",
            Self::Pressure => "pressure",
            Self::Native => "native",
            Self::Subhourly => "subhourly",
        }
    }

    pub fn default_lineage(self) -> ProductLineage {
        match self {
            Self::Surface | Self::Pressure | Self::Native => ProductLineage::Direct,
            Self::Subhourly => ProductLineage::Windowed,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum GribLevelKind {
    Surface,
    MeanSeaLevel,
    HeightAboveGround,
    HeightAboveGroundLayer,
    IsobaricHpa,
    EntireAtmosphere,
    NominalTop,
    /// The surface at a named temperature (the 0 C isotherm and its
    /// colder siblings).
    Isotherm,
    /// One number per column, the largest on any model level.
    ColumnMaximum,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum RenderStyle {
    WeatherWbgt,
    WeatherHdw,
    WeatherHaines,
    WeatherLightningThreat,
    WeatherFlashRate,
    WeatherCape,
    WeatherCin,
    WeatherReflectivity,
    WeatherUh,
    WeatherVerticalVelocity,
    WeatherTemperature,
    WeatherDewpoint,
    WeatherRh,
    WeatherSpecificHumidity,
    WeatherOmega,
    WeatherProbability,
    WeatherWinds,
    WeatherHeight,
    WeatherPressure,
    WeatherWindGust,
    WeatherCloudCover,
    WeatherPrecipitableWater,
    WeatherIvt,
    WeatherQpf,
    WeatherSnowfall,
    WeatherSnowDepthChange,
    WeatherHail,
    WeatherCategorical,
    WeatherVisibility,
    WeatherRadarReflectivity,
    WeatherSatellite,
    WeatherLightning,
    WeatherVorticity,
    WeatherStp,
    WeatherScp,
    WeatherEhi,
    /// Surface elevation.  Distinct from [`RenderStyle::WeatherHeight`],
    /// which is the isobaric height-analysis style: that one draws the
    /// heights as contours and fills with the companion wind speed, which
    /// for orography would produce a plot of the wind over a field that
    /// does not vary.
    WeatherTerrain,
    /// The height of an isotherm in metres, on a sequential ramp whose
    /// cold end is a low level.
    WeatherIsothermHeight,
    /// A supercooled liquid water path in g m-2, sequential, masked
    /// below the lowest level an icing reader cares about.
    WeatherSupercooledWaterPath,
    /// A hydrometeor mixing ratio in g kg-1, sequential, masked below a
    /// hundredth of a gram.
    WeatherHydrometeorMixingRatio,
    AirQualityPm25,
    AirQualityOzone,
    AerosolOpticalDepth,
    DustConcentration,
}

fn recipe_lineage(slug: &str, family: ProductFamily) -> ProductLineage {
    match slug {
        "2m_theta_e_10m_winds" | "2m_heat_index" | "2m_wind_chill" => ProductLineage::Derived,
        "wbgt_2m" | "hot_dry_windy" | "haines_low" | "haines_mid" | "haines_high" | "lightning_threat_1" | "lightning_threat_2" | "lightning_threat_3" | "pr92_flash_rate" => ProductLineage::Derived,
        "1h_qpf" => ProductLineage::Windowed,
        _ => family.default_lineage(),
    }
}

fn recipe_maturity(slug: &str) -> ProductMaturity {
    if matches!(slug, "lightning_threat_1" | "lightning_threat_2" | "lightning_threat_3" | "pr92_flash_rate") { return ProductMaturity::Experimental; }
    if slug.starts_with("nbm_qmd_") {
        return ProductMaturity::Experimental;
    }
    if slug.starts_with("sref_prob_") {
        return ProductMaturity::Experimental;
    }
    if slug.starts_with("aigefs_spr_") {
        return ProductMaturity::Experimental;
    }
    if slug.starts_with("hgefs_spr_") {
        return ProductMaturity::Experimental;
    }
    if slug.starts_with("href_sprd_") {
        return ProductMaturity::Experimental;
    }
    if slug.starts_with("href_prob_") {
        return ProductMaturity::Experimental;
    }
    if slug.starts_with("href_mean_") {
        return ProductMaturity::Experimental;
    }
    if slug.starts_with("refs_sprd_") || slug.starts_with("refs_prob_") {
        return ProductMaturity::Experimental;
    }
    if slug.starts_with("gefs_avg_") || slug.starts_with("gefs_spr_") {
        return ProductMaturity::Experimental;
    }
    match slug {
        "simulated_ir_satellite" | "lightning_flash_density" => ProductMaturity::Experimental,
        _ => ProductMaturity::Operational,
    }
}

fn recipe_flags(slug: &str) -> Vec<ProductSemanticFlag> {
    if slug.starts_with("nbm_qmd_") {
        return vec![ProductSemanticFlag::ProofOriented];
    }
    if slug.starts_with("sref_prob_") {
        return vec![ProductSemanticFlag::ProofOriented];
    }
    if slug.starts_with("aigefs_spr_") {
        return vec![ProductSemanticFlag::ProofOriented];
    }
    if slug.starts_with("hgefs_spr_") {
        return vec![ProductSemanticFlag::ProofOriented];
    }
    if slug.starts_with("href_sprd_") {
        return vec![ProductSemanticFlag::ProofOriented];
    }
    if slug.starts_with("href_prob_") {
        return vec![ProductSemanticFlag::ProofOriented];
    }
    if slug.starts_with("href_mean_") {
        return vec![ProductSemanticFlag::ProofOriented];
    }
    if slug.starts_with("refs_sprd_") || slug.starts_with("refs_prob_") {
        return vec![ProductSemanticFlag::ProofOriented];
    }
    if slug.starts_with("gefs_avg_") || slug.starts_with("gefs_spr_") {
        return vec![ProductSemanticFlag::ProofOriented];
    }
    match slug {
        "cloud_cover_levels" | "precipitation_type" | "composite_reflectivity_uh" => {
            vec![ProductSemanticFlag::Composite]
        }
        _ if is_uh_layer_overlay_recipe(slug) => vec![ProductSemanticFlag::Composite],
        "1h_qpf" => vec![ProductSemanticFlag::Alias],
        _ => Vec::new(),
    }
}

/// The reflectivity recipes with an updraft-helicity layer contoured on
/// top (WOOF uh-layers, 2026-11-02): composite reflectivity over the 0-1,
/// 0-2 and 0-3 km layers, and 1 km AGL reflectivity over all four. The
/// 2-5 km composite overlay keeps its own name, `composite_reflectivity_uh`.
pub const UH_LAYER_OVERLAY_RECIPES: [&str; 7] = [
    "composite_reflectivity_uh0to1km",
    "composite_reflectivity_uh0to2km",
    "composite_reflectivity_uh0to3km",
    "1km_reflectivity_uh2to5km",
    "1km_reflectivity_uh0to1km",
    "1km_reflectivity_uh0to2km",
    "1km_reflectivity_uh0to3km",
];

/// Whether `slug` is one of [`UH_LAYER_OVERLAY_RECIPES`].
pub fn is_uh_layer_overlay_recipe(slug: &str) -> bool {
    UH_LAYER_OVERLAY_RECIPES.contains(&slug)
}

fn recipe_window(slug: &str, lineage: ProductLineage) -> Option<ProductWindowSpec> {
    match slug {
        "1h_qpf" => Some(ProductWindowSpec::accumulation(Some(1))),
        _ if matches!(lineage, ProductLineage::Windowed) => {
            Some(ProductWindowSpec::accumulation(None))
        }
        _ => None,
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct GribFieldSpec {
    pub key: &'static str,
    pub label: &'static str,
    pub family: ProductFamily,
    pub level_kind: GribLevelKind,
    pub level_value: Option<i32>,
    pub selector: Option<FieldSelector>,
    pub idx_fallback_patterns: &'static [&'static str],
}

impl GribFieldSpec {
    pub fn idx_patterns(&self) -> &'static [&'static str] {
        self.idx_fallback_patterns
    }

    pub fn provenance(&self) -> ProductProvenance {
        let mut provenance =
            ProductProvenance::new(self.family.default_lineage(), ProductMaturity::Operational);
        if let Some(selector) = self.selector {
            provenance = provenance.with_selector(selector);
        }
        if self.family == ProductFamily::Subhourly {
            provenance = provenance.with_window(ProductWindowSpec {
                process: StatisticalProcess::Accumulation,
                duration_hours: None,
            });
        }
        provenance
    }

    pub fn product_metadata(&self) -> ProductKeyMetadata {
        let mut metadata = ProductKeyMetadata::new(self.label).with_category(self.family.as_str());
        if let Some(selector) = self.selector {
            metadata = metadata.with_native_units(selector.native_units());
        }
        metadata.with_provenance(self.provenance())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct PlotRecipe {
    pub slug: &'static str,
    pub title: &'static str,
    pub filled: GribFieldSpec,
    pub contours: Option<GribFieldSpec>,
    pub barbs_u: Option<GribFieldSpec>,
    pub barbs_v: Option<GribFieldSpec>,
    pub style: RenderStyle,
}

impl PlotRecipe {
    pub fn provenance(&self) -> ProductProvenance {
        let lineage = recipe_lineage(self.slug, self.filled.family);
        let mut provenance = ProductProvenance::new(lineage, recipe_maturity(self.slug));
        if let Some(selector) = self.filled.selector {
            provenance = provenance.with_selector(selector);
        }
        if let Some(window) = recipe_window(self.slug, lineage) {
            provenance = provenance.with_window(window);
        }
        for flag in recipe_flags(self.slug) {
            provenance = provenance.with_flag(flag);
        }
        provenance
    }

    pub fn product_metadata(&self) -> ProductKeyMetadata {
        let mut metadata = ProductKeyMetadata::new(self.title)
            .with_category(self.filled.family.as_str())
            .with_provenance(self.provenance());
        if let Some(selector) = self.filled.selector {
            metadata = metadata.with_native_units(selector.native_units());
        }
        metadata
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PlotRecipeFetchMode {
    IndexedSubset,
    WholeFileStructuredExtract,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PlotRecipeFetchPolicy {
    PreferIndexedSubset,
    WholeFile,
}

impl PlotRecipeFetchPolicy {
    pub fn fetch_mode(self) -> PlotRecipeFetchMode {
        match self {
            Self::PreferIndexedSubset => PlotRecipeFetchMode::IndexedSubset,
            Self::WholeFile => PlotRecipeFetchMode::WholeFileStructuredExtract,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct PlotRecipeBlocker {
    pub field_key: &'static str,
    pub field_label: &'static str,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct PlotRecipeFetchPlan {
    pub recipe_slug: &'static str,
    pub model: ModelId,
    pub product: &'static str,
    pub fetch_policy: PlotRecipeFetchPolicy,
    pub fetch_mode: PlotRecipeFetchMode,
    pub fields: Vec<&'static GribFieldSpec>,
}

impl PlotRecipeFetchPlan {
    pub fn idx_patterns(&self) -> Vec<&'static str> {
        dedupe_patterns(
            self.fields
                .iter()
                .flat_map(|field| field.idx_patterns().iter().copied()),
        )
    }

    pub fn selectors(&self) -> Vec<FieldSelector> {
        self.fields
            .iter()
            .map(|field| {
                field
                    .selector
                    .expect("plot recipe fetch plan only returns selector-backed fields")
            })
            .collect()
    }

    pub fn variable_patterns(&self) -> Vec<&'static str> {
        match self.fetch_mode {
            PlotRecipeFetchMode::IndexedSubset => self.idx_patterns(),
            PlotRecipeFetchMode::WholeFileStructuredExtract => Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SourceDescriptor {
    pub id: SourceId,
    pub idx_available: bool,
    pub priority: u8,
    pub max_age_hours: Option<u32>,
    pub notes: &'static str,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ModelSummary {
    pub id: ModelId,
    pub description: &'static str,
    pub default_product: &'static str,
    pub cycle_hours_utc: &'static [u8],
    pub max_forecast_hour: u16,
    pub sources: &'static [SourceDescriptor],
    pub runtime_family: ModelRuntimeFamily,
    pub ensemble_mode: EnsembleMode,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LatestRun {
    pub model: ModelId,
    pub cycle: CycleSpec,
    pub source: SourceId,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResolvedCanonicalBundleProduct {
    pub bundle: CanonicalBundleDescriptor,
    pub family: CanonicalDataFamily,
    pub native_product: String,
}

impl ResolvedCanonicalBundleProduct {
    pub fn new<S: Into<String>>(bundle: CanonicalBundleDescriptor, native_product: S) -> Self {
        Self {
            bundle,
            family: bundle.family(),
            native_product: native_product.into(),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error(transparent)]
    Core(#[from] RustwxError),
    #[error("unsupported product '{product}' for model '{model}'")]
    UnsupportedProduct { model: ModelId, product: String },
    #[error(
        "unsupported forecast hour '{forecast_hour}' for model '{model}' cycle '{cycle_hour:02}z': {reason}"
    )]
    UnsupportedForecastHour {
        model: ModelId,
        cycle_hour: u8,
        forecast_hour: u16,
        reason: String,
    },
    #[error("unknown plot recipe '{slug}'")]
    UnknownPlotRecipe { slug: String },
    #[error("plot recipe '{recipe}' is not supported for model '{model}': {reason}")]
    UnsupportedPlotRecipeModel {
        recipe: &'static str,
        model: ModelId,
        reason: String,
    },
    #[error("no working source found for model '{model}' while probing latest availability")]
    NoAvailableRun { model: ModelId },
}

const HRRR_CYCLE_HOURS: &[u8] = &[
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
];
const GFS_CYCLE_HOURS: &[u8] = &[0, 6, 12, 18];
const GDAS_CYCLE_HOURS: &[u8] = &[0, 6, 12, 18];
const GEFS_CYCLE_HOURS: &[u8] = &[0, 6, 12, 18];
const AI_MODEL_CYCLE_HOURS: &[u8] = &[0, 6, 12, 18];
const ECMWF_CYCLE_HOURS: &[u8] = &[0, 6, 12, 18];
const AIFS_CYCLE_HOURS: &[u8] = &[0, 6, 12, 18];
const AIFS_OPEN_DATA_MAX_FORECAST_HOUR: u16 = 360;
const RAP_CYCLE_HOURS: &[u8] = &[
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
];
const NAM_CYCLE_HOURS: &[u8] = &[0, 6, 12, 18];
const HIRESW_CYCLE_HOURS: &[u8] = &[0, 12];
const HREF_CYCLE_HOURS: &[u8] = &[0, 6, 12, 18];
const SREF_CYCLE_HOURS: &[u8] = &[3, 9, 15, 21];
const HOURLY_ANALYSIS_CYCLE_HOURS: &[u8] = &[
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
];
const RRFS_CYCLE_HOURS: &[u8] = HOURLY_ANALYSIS_CYCLE_HOURS;
const RRFS_CYCLE_HORIZONS: [u16; 24] = [
    84, 18, 18, 18, 18, 18, 84, 18, 18, 18, 18, 18,
    84, 18, 18, 18, 18, 18, 84, 18, 18, 18, 18, 18,
];
const RRFS_A_CYCLE_HOURS: &[u8] = &[
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
];
const RRFS_PUBLIC_CYCLE_HOURS: &[u8] = &[0, 3, 6, 9, 12, 15, 18, 21];
const REFS_CYCLE_HOURS: &[u8] = &[0, 6, 12, 18];
const RRFS_FIREWX_CYCLE_HOURS: &[u8] = &[0, 6, 12, 18];
const WRF_GDEX_CYCLE_HOURS: &[u8] = &[
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
];
const WRF_GDEX_DEFAULT_SURFACE_PRODUCT: &str = "d612005-hist2d";
const WRF_GDEX_DEFAULT_PRESSURE_PRODUCT: &str = "d612005-hist3d";

const HRRR_SOURCES: &[SourceDescriptor] = &[
    SourceDescriptor {
        id: SourceId::Nomads,
        idx_available: true,
        priority: 1,
        max_age_hours: Some(48),
        notes: "Operational NOMADS feed",
    },
    SourceDescriptor {
        id: SourceId::Aws,
        idx_available: true,
        priority: 2,
        max_age_hours: None,
        notes: "AWS open data archive",
    },
    SourceDescriptor {
        id: SourceId::Google,
        idx_available: true,
        priority: 3,
        max_age_hours: None,
        notes: "Google mirror",
    },
    SourceDescriptor {
        id: SourceId::Azure,
        idx_available: false,
        priority: 4,
        max_age_hours: None,
        notes: "Azure mirror without .idx coverage",
    },
];

const NOMADS_ONLY_SOURCES: &[SourceDescriptor] = &[SourceDescriptor {
    id: SourceId::Nomads,
    idx_available: true,
    priority: 1,
    max_age_hours: Some(72),
    notes: "Operational NOMADS GRIB2 feed",
}];

const NOMADS_AWS_SOURCES: &[SourceDescriptor] = &[
    SourceDescriptor {
        id: SourceId::Nomads,
        idx_available: true,
        priority: 1,
        max_age_hours: Some(72),
        notes: "Operational NOMADS GRIB2 feed",
    },
    SourceDescriptor {
        id: SourceId::Aws,
        idx_available: true,
        priority: 2,
        max_age_hours: None,
        notes: "NOAA public S3 open-data archive",
    },
];

const GFS_SOURCES: &[SourceDescriptor] = &[
    SourceDescriptor {
        id: SourceId::Nomads,
        idx_available: true,
        priority: 1,
        max_age_hours: Some(48),
        notes: "Operational NOMADS feed",
    },
    SourceDescriptor {
        id: SourceId::Aws,
        idx_available: true,
        priority: 2,
        max_age_hours: None,
        notes: "AWS open data archive",
    },
    SourceDescriptor {
        id: SourceId::Google,
        idx_available: true,
        priority: 3,
        max_age_hours: None,
        notes: "Google mirror",
    },
    SourceDescriptor {
        id: SourceId::Ncei,
        idx_available: false,
        priority: 4,
        max_age_hours: None,
        notes: "Historical NCEI archive; requested resolution is explicit",
    },
    SourceDescriptor {
        id: SourceId::Gdex,
        idx_available: false,
        priority: 3,
        max_age_hours: None,
        notes: "GDEX d084001 (RDA ds084.1), CC BY 4.0",
    },
];

const GEFS_SOURCES: &[SourceDescriptor] = &[
    SourceDescriptor {
        id: SourceId::Nomads,
        idx_available: true,
        priority: 1,
        max_age_hours: Some(48),
        notes: "Operational NOAA GEFS feed",
    },
    SourceDescriptor {
        id: SourceId::Aws,
        idx_available: true,
        priority: 2,
        max_age_hours: None,
        notes: "NOAA GEFS AWS open-data archive",
    },
];

const ECMWF_SOURCES: &[SourceDescriptor] = &[SourceDescriptor {
    id: SourceId::Ecmwf,
    // ECMWF open-data doesn't publish `.grib2.idx` companion files, so
    // HEAD-probing the `.idx` URL returns 404 and `latest_available_run`
    // falsely concludes the run is unavailable. Flagging this source
    // `idx_available=false` makes `availability_probe_url` fall back to
    // the grib URL itself, which HEAD 200s correctly. The tradeoff: no
    // idx-based range extraction, but ECMWF's open-data is served as
    // single grib2 files that the fetcher already pulls whole.
    idx_available: false,
    priority: 1,
    max_age_hours: None,
    notes: "ECMWF open data",
}];

const AIFS_SOURCES: &[SourceDescriptor] = &[
    SourceDescriptor {
        id: SourceId::Ecmwf,
        idx_available: false,
        priority: 1,
        max_age_hours: None,
        notes: "ECMWF AIFS Single v2 open-data GRIB2 feed",
    },
    SourceDescriptor {
        id: SourceId::AifsInference,
        idx_available: false,
        priority: 2,
        max_age_hours: None,
        notes: "Local NetCDF archive disseminated by an active AIFS-v2 inference harness",
    },
    SourceDescriptor {
        id: SourceId::Earth2Archive,
        idx_available: false,
        priority: 3,
        max_age_hours: None,
        notes: "Legacy local NetCDF archive name for Earth2Studio or older inference harnesses",
    },
];

// Same CONUS filename contract, separate v1.0 and historical namespaces.
const RRFS_CONUS_BASES: &[(ModelId, SourceId, &str)] = &[
    (ModelId::Rrfs, SourceId::Aws, "https://noaa-rrfs-ops-pds.s3.amazonaws.com"),
    (ModelId::Rrfs, SourceId::Nomads, "https://nomads.ncep.noaa.gov/pub/data/nccf/com/rrfs/v1.0"),
    (ModelId::RrfsPublic, SourceId::Aws, "https://noaa-rrfs-pds.s3.amazonaws.com/rrfs_public"),
];

const RRFS_SOURCES: &[SourceDescriptor] = &[
    SourceDescriptor { id: SourceId::Aws, idx_available: true, priority: 1, max_age_hours: None,
        notes: "NOAA RRFS v1.0 archive (noaa-rrfs-ops-pds)" },
    SourceDescriptor { id: SourceId::Nomads, idx_available: true, priority: 2, max_age_hours: Some(48),
        notes: "NCEP RRFS v1.0 prslev and 2dfld products" },
];

const RRFS_A_SOURCES: &[SourceDescriptor] = &[SourceDescriptor {
    id: SourceId::Aws,
    idx_available: true,
    priority: 1,
    max_age_hours: None,
    notes: "NOAA RRFS AWS bucket",
}];

const RRFS_PUBLIC_SOURCES: &[SourceDescriptor] = &[SourceDescriptor {
    id: SourceId::Aws,
    idx_available: true,
    priority: 1,
    max_age_hours: None,
    notes: "NOAA RRFS public prototype AWS bucket",
}];

const REFS_SOURCES: &[SourceDescriptor] = &[SourceDescriptor {
    id: SourceId::Aws,
    idx_available: true,
    priority: 1,
    max_age_hours: None,
    notes: "NOAA REFS ensemble post-processing AWS bucket",
}];

const RRFS_FIREWX_SOURCES: &[SourceDescriptor] = &[SourceDescriptor {
    id: SourceId::Aws,
    idx_available: true,
    priority: 1,
    max_age_hours: None,
    notes: "NOAA RRFS public fire-weather nest AWS bucket",
}];

const WRF_GDEX_SOURCES: &[SourceDescriptor] = &[SourceDescriptor {
    id: SourceId::Gdex,
    idx_available: false,
    priority: 1,
    max_age_hours: None,
    notes: "UCAR GDEX THREDDS fileServer",
}];

const MODELS: &[ModelSummary] = &[
    ModelSummary {
        id: ModelId::Hrrr,
        description: "HRRR 3 km CONUS rapid-refresh forecast",
        default_product: "sfc",
        cycle_hours_utc: HRRR_CYCLE_HOURS,
        max_forecast_hour: 48,
        sources: HRRR_SOURCES,
        runtime_family: ModelRuntimeFamily::Grib2Forecast,
        ensemble_mode: EnsembleMode::Deterministic,
    },
    ModelSummary {
        id: ModelId::HrrrAk,
        description: "HRRR Alaska rapid-refresh forecast",
        default_product: "sfc",
        cycle_hours_utc: HRRR_CYCLE_HOURS,
        max_forecast_hour: 48,
        sources: HRRR_SOURCES,
        runtime_family: ModelRuntimeFamily::Grib2Forecast,
        ensemble_mode: EnsembleMode::Deterministic,
    },
    ModelSummary {
        id: ModelId::Gfs,
        description: "GFS global 0.25 degree atmospheric grid",
        default_product: "pgrb2.0p25",
        cycle_hours_utc: GFS_CYCLE_HOURS,
        max_forecast_hour: 384,
        sources: GFS_SOURCES,
        runtime_family: ModelRuntimeFamily::Grib2Forecast,
        ensemble_mode: EnsembleMode::Deterministic,
    },
    ModelSummary {
        id: ModelId::Gdas,
        description: "GDAS global data-assimilation analysis/forecast grids",
        default_product: "pgrb2.0p25",
        cycle_hours_utc: GDAS_CYCLE_HOURS,
        max_forecast_hour: 9,
        sources: NOMADS_AWS_SOURCES,
        runtime_family: ModelRuntimeFamily::Grib2Forecast,
        ensemble_mode: EnsembleMode::Deterministic,
    },
    ModelSummary {
        id: ModelId::Gefs,
        description: "GEFS global 0.5 degree ensemble forecast",
        default_product: "pgrb2ap5/gec00",
        cycle_hours_utc: GEFS_CYCLE_HOURS,
        max_forecast_hour: 384,
        sources: GEFS_SOURCES,
        runtime_family: ModelRuntimeFamily::Grib2Forecast,
        ensemble_mode: EnsembleMode::MemberGribFiles,
    },
    ModelSummary {
        id: ModelId::Aigfs,
        description: "NOAA AI-GFS global data-driven forecast",
        default_product: "sfc",
        cycle_hours_utc: AI_MODEL_CYCLE_HOURS,
        max_forecast_hour: 384,
        sources: NOMADS_ONLY_SOURCES,
        runtime_family: ModelRuntimeFamily::Grib2Forecast,
        ensemble_mode: EnsembleMode::Deterministic,
    },
    ModelSummary {
        id: ModelId::Aigefs,
        description: "NOAA AI-GEFS individual-member global data-driven forecast and published statistics",
        default_product: "sfc/mem000",
        cycle_hours_utc: AI_MODEL_CYCLE_HOURS,
        max_forecast_hour: 384,
        sources: NOMADS_AWS_SOURCES,
        runtime_family: ModelRuntimeFamily::Grib2Forecast,
        ensemble_mode: EnsembleMode::MemberGribFiles,
    },
    ModelSummary {
        id: ModelId::Hgefs,
        description: "NOAA HGEFS hybrid AI/GEFS ensemble-stat global forecast",
        default_product: "sfc/avg",
        cycle_hours_utc: AI_MODEL_CYCLE_HOURS,
        max_forecast_hour: 240,
        sources: NOMADS_ONLY_SOURCES,
        runtime_family: ModelRuntimeFamily::Grib2Forecast,
        ensemble_mode: EnsembleMode::MemberGribFiles,
    },
    ModelSummary {
        id: ModelId::EcmwfOpenData,
        description: "ECMWF IFS Cycle 50r1 open-data 0.25 degree feed",
        default_product: "oper",
        cycle_hours_utc: ECMWF_CYCLE_HOURS,
        max_forecast_hour: 360,
        sources: ECMWF_SOURCES,
        runtime_family: ModelRuntimeFamily::Grib2Forecast,
        ensemble_mode: EnsembleMode::Deterministic,
    },
    ModelSummary {
        id: ModelId::Aifs,
        description: "ECMWF AIFS Single v2 open-data 0.25 degree forecast",
        default_product: "oper",
        cycle_hours_utc: AIFS_CYCLE_HOURS,
        max_forecast_hour: AIFS_OPEN_DATA_MAX_FORECAST_HOUR,
        sources: AIFS_SOURCES,
        runtime_family: ModelRuntimeFamily::Grib2Forecast,
        ensemble_mode: EnsembleMode::Deterministic,
    },
    ModelSummary {
        id: ModelId::Rap,
        description: "RAP hourly North America forecast grids",
        default_product: "awp130pgrb",
        cycle_hours_utc: RAP_CYCLE_HOURS,
        max_forecast_hour: 51,
        sources: NOMADS_AWS_SOURCES,
        runtime_family: ModelRuntimeFamily::Grib2Forecast,
        ensemble_mode: EnsembleMode::Deterministic,
    },
    ModelSummary {
        id: ModelId::Nam,
        description: "NAM CONUS/North America/nest regional forecast grids",
        default_product: "awip12",
        cycle_hours_utc: NAM_CYCLE_HOURS,
        max_forecast_hour: 84,
        sources: NOMADS_AWS_SOURCES,
        runtime_family: ModelRuntimeFamily::Grib2Forecast,
        ensemble_mode: EnsembleMode::Deterministic,
    },
    ModelSummary {
        id: ModelId::Hiresw,
        description: "HIRESW high-resolution regional ARW/FV3 forecast grids",
        default_product: "arw_2p5km/conus",
        cycle_hours_utc: HIRESW_CYCLE_HOURS,
        max_forecast_hour: 48,
        sources: NOMADS_ONLY_SOURCES,
        runtime_family: ModelRuntimeFamily::Grib2Forecast,
        ensemble_mode: EnsembleMode::MemberGribFiles,
    },
    ModelSummary {
        id: ModelId::Href,
        description: "HREF CONUS ensemble products",
        default_product: "ensprod/conus/sprd",
        cycle_hours_utc: HREF_CYCLE_HOURS,
        max_forecast_hour: 48,
        sources: NOMADS_ONLY_SOURCES,
        runtime_family: ModelRuntimeFamily::Grib2Forecast,
        ensemble_mode: EnsembleMode::MemberGribFiles,
    },
    ModelSummary {
        id: ModelId::Sref,
        description: "SREF regional ensemble member/statistic forecast grids",
        default_product: "ensprod/pgrb212/mean_3hrly",
        cycle_hours_utc: SREF_CYCLE_HOURS,
        max_forecast_hour: 87,
        sources: NOMADS_ONLY_SOURCES,
        runtime_family: ModelRuntimeFamily::Grib2Forecast,
        ensemble_mode: EnsembleMode::MemberGribFiles,
    },
    ModelSummary {
        id: ModelId::Rtma,
        description: "RTMA 2.5 km hourly surface analysis",
        default_product: "2dvaranl_ndfd",
        cycle_hours_utc: HOURLY_ANALYSIS_CYCLE_HOURS,
        max_forecast_hour: 0,
        sources: NOMADS_ONLY_SOURCES,
        runtime_family: ModelRuntimeFamily::Grib2Forecast,
        ensemble_mode: EnsembleMode::Deterministic,
    },
    ModelSummary {
        id: ModelId::Urma,
        description: "URMA 2.5 km hourly surface analysis",
        default_product: "2dvaranl_ndfd",
        cycle_hours_utc: HOURLY_ANALYSIS_CYCLE_HOURS,
        max_forecast_hour: 0,
        sources: NOMADS_ONLY_SOURCES,
        runtime_family: ModelRuntimeFamily::Grib2Forecast,
        ensemble_mode: EnsembleMode::Deterministic,
    },
    ModelSummary {
        id: ModelId::Nbm,
        description: "National Blend of Models gridded core forecast",
        default_product: "core/co",
        cycle_hours_utc: HOURLY_ANALYSIS_CYCLE_HOURS,
        max_forecast_hour: 264,
        sources: NOMADS_AWS_SOURCES,
        runtime_family: ModelRuntimeFamily::Grib2Forecast,
        ensemble_mode: EnsembleMode::Deterministic,
    },
    ModelSummary {
        id: ModelId::Rrfs,
        description: "NOAA RRFS v1.0 3 km CONUS deterministic forecast",
        default_product: "prs-conus",
        cycle_hours_utc: RRFS_CYCLE_HOURS,
        max_forecast_hour: 84,
        sources: RRFS_SOURCES,
        runtime_family: ModelRuntimeFamily::Grib2Forecast,
        ensemble_mode: EnsembleMode::Deterministic,
    },
    ModelSummary {
        id: ModelId::RrfsA,
        description: "RRFS-A AWS open data feed with CONUS/NA/HI/PR variants",
        default_product: "prs-conus",
        cycle_hours_utc: RRFS_A_CYCLE_HOURS,
        max_forecast_hour: 60,
        sources: RRFS_A_SOURCES,
        runtime_family: ModelRuntimeFamily::Grib2Forecast,
        ensemble_mode: EnsembleMode::Deterministic,
    },
    ModelSummary {
        id: ModelId::RrfsPublic,
        description: "RRFS public prototype 3 km CONUS deterministic forecast",
        default_product: "prs-conus",
        cycle_hours_utc: RRFS_PUBLIC_CYCLE_HOURS,
        max_forecast_hour: 60,
        sources: RRFS_PUBLIC_SOURCES,
        runtime_family: ModelRuntimeFamily::Grib2Forecast,
        ensemble_mode: EnsembleMode::Deterministic,
    },
    ModelSummary {
        id: ModelId::Refs,
        description: "REFS RRFS ensemble post-processed mean/spread/probability products",
        default_product: "mean-conus",
        cycle_hours_utc: REFS_CYCLE_HOURS,
        max_forecast_hour: 60,
        sources: REFS_SOURCES,
        runtime_family: ModelRuntimeFamily::Grib2Forecast,
        ensemble_mode: EnsembleMode::MemberGribFiles,
    },
    ModelSummary {
        id: ModelId::RrfsFireWx,
        description: "RRFS public 1.5 km fire-weather nest deterministic forecast",
        default_product: "2dfld-firewx",
        cycle_hours_utc: RRFS_FIREWX_CYCLE_HOURS,
        max_forecast_hour: 36,
        sources: RRFS_FIREWX_SOURCES,
        runtime_family: ModelRuntimeFamily::Grib2Forecast,
        ensemble_mode: EnsembleMode::Deterministic,
    },
    ModelSummary {
        id: ModelId::WrfGdex,
        description: "WRF NetCDF/wrfout datasets; UCAR GDEX is one supported source",
        default_product: WRF_GDEX_DEFAULT_SURFACE_PRODUCT,
        cycle_hours_utc: WRF_GDEX_CYCLE_HOURS,
        max_forecast_hour: 0,
        sources: WRF_GDEX_SOURCES,
        runtime_family: ModelRuntimeFamily::WrfNetcdfArchive,
        ensemble_mode: EnsembleMode::Deterministic,
    },
];

const fn field_spec(
    key: &'static str,
    label: &'static str,
    family: ProductFamily,
    level_kind: GribLevelKind,
    level_value: Option<i32>,
    selector: Option<FieldSelector>,
    idx_patterns: &'static [&'static str],
) -> GribFieldSpec {
    GribFieldSpec {
        key,
        label,
        family,
        level_kind,
        level_value,
        selector,
        idx_fallback_patterns: idx_patterns,
    }
}

const FIELD_500_HEIGHT: GribFieldSpec = field_spec(
    "height_500mb",
    "500mb Height",
    ProductFamily::Pressure,
    GribLevelKind::IsobaricHpa,
    Some(500),
    Some(FieldSelector::isobaric(
        CanonicalField::GeopotentialHeight,
        500,
    )),
    &["HGT:500 mb"],
);

const FIELD_AIGEFS_SPR_500_HEIGHT_STDDEV: GribFieldSpec = field_spec(
    "aigefs_spread_height_500mb_stddev",
    "AI-GEFS 500mb Height Spread",
    ProductFamily::Pressure,
    GribLevelKind::IsobaricHpa,
    Some(500),
    Some(
        FieldSelector::isobaric(CanonicalField::GeopotentialHeight, 500)
            .with_ensemble_standard_deviation(),
    ),
    &["HGT:500 mb"],
);

const FIELD_HGEFS_SPR_500_HEIGHT_STDDEV: GribFieldSpec = field_spec(
    "hgefs_spread_height_500mb_stddev",
    "HGEFS 500mb Height Spread",
    ProductFamily::Pressure,
    GribLevelKind::IsobaricHpa,
    Some(500),
    Some(
        FieldSelector::isobaric(CanonicalField::GeopotentialHeight, 500)
            .with_ensemble_standard_deviation(),
    ),
    &["HGT:500 mb"],
);

const FIELD_HREF_SPRD_500_HEIGHT: GribFieldSpec = field_spec(
    "href_spread_height_500mb",
    "HREF 500mb Height Spread",
    ProductFamily::Pressure,
    GribLevelKind::IsobaricHpa,
    Some(500),
    Some(
        FieldSelector::isobaric(CanonicalField::GeopotentialHeight, 500)
            .with_product(FieldProduct::EnsembleSpread),
    ),
    &["HGT:500 mb"],
);

const FIELD_HREF_MEAN_500_HEIGHT: GribFieldSpec = field_spec(
    "href_mean_height_500mb",
    "HREF 500mb Height Mean",
    ProductFamily::Pressure,
    GribLevelKind::IsobaricHpa,
    Some(500),
    Some(FieldSelector::isobaric(CanonicalField::GeopotentialHeight, 500).with_ensemble_mean()),
    &["HGT:500 mb"],
);

const FIELD_REFS_SPRD_500_HEIGHT: GribFieldSpec = field_spec(
    "refs_spread_height_500mb",
    "REFS 500mb Height Spread",
    ProductFamily::Pressure,
    GribLevelKind::IsobaricHpa,
    Some(500),
    Some(
        FieldSelector::isobaric(CanonicalField::GeopotentialHeight, 500)
            .with_product(FieldProduct::EnsembleSpread),
    ),
    &["HGT:500 mb"],
);

const FIELD_GEFS_AVG_500_HEIGHT: GribFieldSpec = field_spec(
    "gefs_mean_height_500mb",
    "GEFS 500mb Height Mean",
    ProductFamily::Pressure,
    GribLevelKind::IsobaricHpa,
    Some(500),
    Some(FieldSelector::isobaric(CanonicalField::GeopotentialHeight, 500).with_ensemble_mean()),
    &["HGT:500 mb"],
);

const FIELD_GEFS_SPR_500_HEIGHT_STDDEV: GribFieldSpec = field_spec(
    "gefs_spread_height_500mb_stddev",
    "GEFS 500mb Height Spread",
    ProductFamily::Pressure,
    GribLevelKind::IsobaricHpa,
    Some(500),
    Some(
        FieldSelector::isobaric(CanonicalField::GeopotentialHeight, 500)
            .with_ensemble_standard_deviation(),
    ),
    &["HGT:500 mb"],
);

const FIELD_700_HEIGHT: GribFieldSpec = field_spec(
    "height_700mb",
    "700mb Height",
    ProductFamily::Pressure,
    GribLevelKind::IsobaricHpa,
    Some(700),
    Some(FieldSelector::isobaric(
        CanonicalField::GeopotentialHeight,
        700,
    )),
    &["HGT:700 mb"],
);

const FIELD_850_HEIGHT: GribFieldSpec = field_spec(
    "height_850mb",
    "850mb Height",
    ProductFamily::Pressure,
    GribLevelKind::IsobaricHpa,
    Some(850),
    Some(FieldSelector::isobaric(
        CanonicalField::GeopotentialHeight,
        850,
    )),
    &["HGT:850 mb"],
);

const FIELD_500_TEMP: GribFieldSpec = field_spec(
    "temperature_500mb",
    "500mb Temperature",
    ProductFamily::Pressure,
    GribLevelKind::IsobaricHpa,
    Some(500),
    Some(FieldSelector::isobaric(CanonicalField::Temperature, 500)),
    &["TMP:500 mb"],
);

const FIELD_GEFS_AVG_500_TEMP: GribFieldSpec = field_spec(
    "gefs_mean_temperature_500mb",
    "GEFS 500mb Temperature Mean",
    ProductFamily::Pressure,
    GribLevelKind::IsobaricHpa,
    Some(500),
    Some(FieldSelector::isobaric(CanonicalField::Temperature, 500).with_ensemble_mean()),
    &["TMP:500 mb"],
);

const FIELD_GEFS_SPR_500_TEMP_STDDEV: GribFieldSpec = field_spec(
    "gefs_spread_temperature_500mb_stddev",
    "GEFS 500mb Temperature Spread",
    ProductFamily::Pressure,
    GribLevelKind::IsobaricHpa,
    Some(500),
    Some(
        FieldSelector::isobaric(CanonicalField::Temperature, 500)
            .with_ensemble_standard_deviation(),
    ),
    &["TMP:500 mb"],
);

const FIELD_AIGEFS_SPR_500_TEMP_STDDEV: GribFieldSpec = field_spec(
    "aigefs_spread_temperature_500mb_stddev",
    "AI-GEFS 500mb Temperature Spread",
    ProductFamily::Pressure,
    GribLevelKind::IsobaricHpa,
    Some(500),
    Some(
        FieldSelector::isobaric(CanonicalField::Temperature, 500)
            .with_ensemble_standard_deviation(),
    ),
    &["TMP:500 mb"],
);

const FIELD_HGEFS_SPR_500_TEMP_STDDEV: GribFieldSpec = field_spec(
    "hgefs_spread_temperature_500mb_stddev",
    "HGEFS 500mb Temperature Spread",
    ProductFamily::Pressure,
    GribLevelKind::IsobaricHpa,
    Some(500),
    Some(
        FieldSelector::isobaric(CanonicalField::Temperature, 500)
            .with_ensemble_standard_deviation(),
    ),
    &["TMP:500 mb"],
);

const FIELD_HREF_SPRD_500_TEMP: GribFieldSpec = field_spec(
    "href_spread_temperature_500mb",
    "HREF 500mb Temperature Spread",
    ProductFamily::Pressure,
    GribLevelKind::IsobaricHpa,
    Some(500),
    Some(
        FieldSelector::isobaric(CanonicalField::Temperature, 500)
            .with_product(FieldProduct::EnsembleSpread),
    ),
    &["TMP:500 mb"],
);

const FIELD_HREF_MEAN_500_TEMP: GribFieldSpec = field_spec(
    "href_mean_temperature_500mb",
    "HREF 500mb Temperature Mean",
    ProductFamily::Pressure,
    GribLevelKind::IsobaricHpa,
    Some(500),
    Some(FieldSelector::isobaric(CanonicalField::Temperature, 500).with_ensemble_mean()),
    &["TMP:500 mb"],
);

const FIELD_REFS_SPRD_500_TEMP: GribFieldSpec = field_spec(
    "refs_spread_temperature_500mb",
    "REFS 500mb Temperature Spread",
    ProductFamily::Pressure,
    GribLevelKind::IsobaricHpa,
    Some(500),
    Some(
        FieldSelector::isobaric(CanonicalField::Temperature, 500)
            .with_product(FieldProduct::EnsembleSpread),
    ),
    &["TMP:500 mb"],
);

const FIELD_850_TEMP: GribFieldSpec = field_spec(
    "temperature_850mb",
    "850mb Temperature",
    ProductFamily::Pressure,
    GribLevelKind::IsobaricHpa,
    Some(850),
    Some(FieldSelector::isobaric(CanonicalField::Temperature, 850)),
    &["TMP:850 mb"],
);

const FIELD_700_TEMP: GribFieldSpec = field_spec(
    "temperature_700mb",
    "700mb Temperature",
    ProductFamily::Pressure,
    GribLevelKind::IsobaricHpa,
    Some(700),
    Some(FieldSelector::isobaric(CanonicalField::Temperature, 700)),
    &["TMP:700 mb"],
);

const FIELD_700_DEWPOINT: GribFieldSpec = field_spec(
    "dewpoint_700mb",
    "700mb Dewpoint",
    ProductFamily::Pressure,
    GribLevelKind::IsobaricHpa,
    Some(700),
    Some(FieldSelector::isobaric(CanonicalField::Dewpoint, 700)),
    &["DPT:700 mb"],
);

const FIELD_850_DEWPOINT: GribFieldSpec = field_spec(
    "dewpoint_850mb",
    "850mb Dewpoint",
    ProductFamily::Pressure,
    GribLevelKind::IsobaricHpa,
    Some(850),
    Some(FieldSelector::isobaric(CanonicalField::Dewpoint, 850)),
    &["DPT:850 mb"],
);

const FIELD_500_RH: GribFieldSpec = field_spec(
    "rh_500mb",
    "500mb Relative Humidity",
    ProductFamily::Pressure,
    GribLevelKind::IsobaricHpa,
    Some(500),
    Some(FieldSelector::isobaric(
        CanonicalField::RelativeHumidity,
        500,
    )),
    &["RH:500 mb"],
);

const FIELD_GEFS_AVG_500_RH: GribFieldSpec = field_spec(
    "gefs_mean_rh_500mb",
    "GEFS 500mb Relative Humidity Mean",
    ProductFamily::Pressure,
    GribLevelKind::IsobaricHpa,
    Some(500),
    Some(FieldSelector::isobaric(CanonicalField::RelativeHumidity, 500).with_ensemble_mean()),
    &["RH:500 mb"],
);

const FIELD_GEFS_SPR_500_RH_STDDEV: GribFieldSpec = field_spec(
    "gefs_spread_rh_500mb_stddev",
    "GEFS 500mb Relative Humidity Spread",
    ProductFamily::Pressure,
    GribLevelKind::IsobaricHpa,
    Some(500),
    Some(
        FieldSelector::isobaric(CanonicalField::RelativeHumidity, 500)
            .with_ensemble_standard_deviation(),
    ),
    &["RH:500 mb"],
);

const FIELD_700_RH: GribFieldSpec = field_spec(
    "rh_700mb",
    "700mb Relative Humidity",
    ProductFamily::Pressure,
    GribLevelKind::IsobaricHpa,
    Some(700),
    Some(FieldSelector::isobaric(
        CanonicalField::RelativeHumidity,
        700,
    )),
    &["RH:700 mb"],
);

const FIELD_850_RH: GribFieldSpec = field_spec(
    "rh_850mb",
    "850mb Relative Humidity",
    ProductFamily::Pressure,
    GribLevelKind::IsobaricHpa,
    Some(850),
    Some(FieldSelector::isobaric(
        CanonicalField::RelativeHumidity,
        850,
    )),
    &["RH:850 mb"],
);

const FIELD_500_ABSOLUTE_VORTICITY: GribFieldSpec = field_spec(
    "absolute_vorticity_500mb",
    "500mb Absolute Vorticity",
    ProductFamily::Pressure,
    GribLevelKind::IsobaricHpa,
    Some(500),
    Some(FieldSelector::isobaric(
        CanonicalField::AbsoluteVorticity,
        500,
    )),
    &["ABSV:500 mb"],
);

const FIELD_700_ABSOLUTE_VORTICITY: GribFieldSpec = field_spec(
    "absolute_vorticity_700mb",
    "700mb Absolute Vorticity",
    ProductFamily::Pressure,
    GribLevelKind::IsobaricHpa,
    Some(700),
    Some(FieldSelector::isobaric(
        CanonicalField::AbsoluteVorticity,
        700,
    )),
    &["ABSV:700 mb"],
);

const FIELD_850_ABSOLUTE_VORTICITY: GribFieldSpec = field_spec(
    "absolute_vorticity_850mb",
    "850mb Absolute Vorticity",
    ProductFamily::Pressure,
    GribLevelKind::IsobaricHpa,
    Some(850),
    Some(FieldSelector::isobaric(
        CanonicalField::AbsoluteVorticity,
        850,
    )),
    &["ABSV:850 mb"],
);

const FIELD_500_U: GribFieldSpec = field_spec(
    "u_500mb",
    "500mb U Wind",
    ProductFamily::Pressure,
    GribLevelKind::IsobaricHpa,
    Some(500),
    Some(FieldSelector::isobaric(CanonicalField::UWind, 500)),
    &["UGRD:500 mb"],
);

const FIELD_500_V: GribFieldSpec = field_spec(
    "v_500mb",
    "500mb V Wind",
    ProductFamily::Pressure,
    GribLevelKind::IsobaricHpa,
    Some(500),
    Some(FieldSelector::isobaric(CanonicalField::VWind, 500)),
    &["VGRD:500 mb"],
);

const FIELD_GEFS_AVG_500_U: GribFieldSpec = field_spec(
    "gefs_mean_u_500mb",
    "GEFS 500mb U Wind Mean",
    ProductFamily::Pressure,
    GribLevelKind::IsobaricHpa,
    Some(500),
    Some(FieldSelector::isobaric(CanonicalField::UWind, 500).with_ensemble_mean()),
    &["UGRD:500 mb"],
);

const FIELD_GEFS_AVG_500_V: GribFieldSpec = field_spec(
    "gefs_mean_v_500mb",
    "GEFS 500mb V Wind Mean",
    ProductFamily::Pressure,
    GribLevelKind::IsobaricHpa,
    Some(500),
    Some(FieldSelector::isobaric(CanonicalField::VWind, 500).with_ensemble_mean()),
    &["VGRD:500 mb"],
);

const FIELD_700_U: GribFieldSpec = field_spec(
    "u_700mb",
    "700mb U Wind",
    ProductFamily::Pressure,
    GribLevelKind::IsobaricHpa,
    Some(700),
    Some(FieldSelector::isobaric(CanonicalField::UWind, 700)),
    &["UGRD:700 mb"],
);

const FIELD_700_V: GribFieldSpec = field_spec(
    "v_700mb",
    "700mb V Wind",
    ProductFamily::Pressure,
    GribLevelKind::IsobaricHpa,
    Some(700),
    Some(FieldSelector::isobaric(CanonicalField::VWind, 700)),
    &["VGRD:700 mb"],
);

const FIELD_850_U: GribFieldSpec = field_spec(
    "u_850mb",
    "850mb U Wind",
    ProductFamily::Pressure,
    GribLevelKind::IsobaricHpa,
    Some(850),
    Some(FieldSelector::isobaric(CanonicalField::UWind, 850)),
    &["UGRD:850 mb"],
);

const FIELD_850_V: GribFieldSpec = field_spec(
    "v_850mb",
    "850mb V Wind",
    ProductFamily::Pressure,
    GribLevelKind::IsobaricHpa,
    Some(850),
    Some(FieldSelector::isobaric(CanonicalField::VWind, 850)),
    &["VGRD:850 mb"],
);

const FIELD_200_HEIGHT: GribFieldSpec = field_spec(
    "height_200mb",
    "200mb Height",
    ProductFamily::Pressure,
    GribLevelKind::IsobaricHpa,
    Some(200),
    Some(FieldSelector::isobaric(
        CanonicalField::GeopotentialHeight,
        200,
    )),
    &["HGT:200 mb"],
);

const FIELD_300_HEIGHT: GribFieldSpec = field_spec(
    "height_300mb",
    "300mb Height",
    ProductFamily::Pressure,
    GribLevelKind::IsobaricHpa,
    Some(300),
    Some(FieldSelector::isobaric(
        CanonicalField::GeopotentialHeight,
        300,
    )),
    &["HGT:300 mb"],
);

const FIELD_250_HEIGHT: GribFieldSpec = field_spec(
    "height_250mb",
    "250mb Height",
    ProductFamily::Pressure,
    GribLevelKind::IsobaricHpa,
    Some(250),
    Some(FieldSelector::isobaric(
        CanonicalField::GeopotentialHeight,
        250,
    )),
    &["HGT:250 mb"],
);

const FIELD_200_TEMP: GribFieldSpec = field_spec(
    "temperature_200mb",
    "200mb Temperature",
    ProductFamily::Pressure,
    GribLevelKind::IsobaricHpa,
    Some(200),
    Some(FieldSelector::isobaric(CanonicalField::Temperature, 200)),
    &["TMP:200 mb"],
);

const FIELD_300_TEMP: GribFieldSpec = field_spec(
    "temperature_300mb",
    "300mb Temperature",
    ProductFamily::Pressure,
    GribLevelKind::IsobaricHpa,
    Some(300),
    Some(FieldSelector::isobaric(CanonicalField::Temperature, 300)),
    &["TMP:300 mb"],
);

const FIELD_250_TEMP: GribFieldSpec = field_spec(
    "temperature_250mb",
    "250mb Temperature",
    ProductFamily::Pressure,
    GribLevelKind::IsobaricHpa,
    Some(250),
    Some(FieldSelector::isobaric(CanonicalField::Temperature, 250)),
    &["TMP:250 mb"],
);

const FIELD_200_RH: GribFieldSpec = field_spec(
    "rh_200mb",
    "200mb Relative Humidity",
    ProductFamily::Pressure,
    GribLevelKind::IsobaricHpa,
    Some(200),
    Some(FieldSelector::isobaric(
        CanonicalField::RelativeHumidity,
        200,
    )),
    &["RH:200 mb"],
);

const FIELD_300_RH: GribFieldSpec = field_spec(
    "rh_300mb",
    "300mb Relative Humidity",
    ProductFamily::Pressure,
    GribLevelKind::IsobaricHpa,
    Some(300),
    Some(FieldSelector::isobaric(
        CanonicalField::RelativeHumidity,
        300,
    )),
    &["RH:300 mb"],
);

const FIELD_200_ABSOLUTE_VORTICITY: GribFieldSpec = field_spec(
    "absolute_vorticity_200mb",
    "200mb Absolute Vorticity",
    ProductFamily::Pressure,
    GribLevelKind::IsobaricHpa,
    Some(200),
    Some(FieldSelector::isobaric(
        CanonicalField::AbsoluteVorticity,
        200,
    )),
    &["ABSV:200 mb"],
);

const FIELD_300_ABSOLUTE_VORTICITY: GribFieldSpec = field_spec(
    "absolute_vorticity_300mb",
    "300mb Absolute Vorticity",
    ProductFamily::Pressure,
    GribLevelKind::IsobaricHpa,
    Some(300),
    Some(FieldSelector::isobaric(
        CanonicalField::AbsoluteVorticity,
        300,
    )),
    &["ABSV:300 mb"],
);

const FIELD_200_U: GribFieldSpec = field_spec(
    "u_200mb",
    "200mb U Wind",
    ProductFamily::Pressure,
    GribLevelKind::IsobaricHpa,
    Some(200),
    Some(FieldSelector::isobaric(CanonicalField::UWind, 200)),
    &["UGRD:200 mb"],
);

const FIELD_200_V: GribFieldSpec = field_spec(
    "v_200mb",
    "200mb V Wind",
    ProductFamily::Pressure,
    GribLevelKind::IsobaricHpa,
    Some(200),
    Some(FieldSelector::isobaric(CanonicalField::VWind, 200)),
    &["VGRD:200 mb"],
);

const FIELD_300_U: GribFieldSpec = field_spec(
    "u_300mb",
    "300mb U Wind",
    ProductFamily::Pressure,
    GribLevelKind::IsobaricHpa,
    Some(300),
    Some(FieldSelector::isobaric(CanonicalField::UWind, 300)),
    &["UGRD:300 mb"],
);

const FIELD_300_V: GribFieldSpec = field_spec(
    "v_300mb",
    "300mb V Wind",
    ProductFamily::Pressure,
    GribLevelKind::IsobaricHpa,
    Some(300),
    Some(FieldSelector::isobaric(CanonicalField::VWind, 300)),
    &["VGRD:300 mb"],
);

const FIELD_250_U: GribFieldSpec = field_spec(
    "u_250mb",
    "250mb U Wind",
    ProductFamily::Pressure,
    GribLevelKind::IsobaricHpa,
    Some(250),
    Some(FieldSelector::isobaric(CanonicalField::UWind, 250)),
    &["UGRD:250 mb"],
);

const FIELD_250_V: GribFieldSpec = field_spec(
    "v_250mb",
    "250mb V Wind",
    ProductFamily::Pressure,
    GribLevelKind::IsobaricHpa,
    Some(250),
    Some(FieldSelector::isobaric(CanonicalField::VWind, 250)),
    &["VGRD:250 mb"],
);

const FIELD_2M_TEMP: GribFieldSpec = field_spec(
    "temperature_2m_agl",
    "2m AGL Temperature",
    ProductFamily::Surface,
    GribLevelKind::HeightAboveGround,
    Some(2),
    Some(FieldSelector::height_agl(CanonicalField::Temperature, 2)),
    &["TMP:2 m above ground"],
);

const FIELD_GEFS_AVG_2M_TEMP: GribFieldSpec = field_spec(
    "gefs_mean_temperature_2m_agl",
    "GEFS 2m AGL Temperature Mean",
    ProductFamily::Surface,
    GribLevelKind::HeightAboveGround,
    Some(2),
    Some(FieldSelector::height_agl(CanonicalField::Temperature, 2).with_ensemble_mean()),
    &["TMP:2 m above ground"],
);

const FIELD_GEFS_SPR_2M_TEMP_STDDEV: GribFieldSpec = field_spec(
    "gefs_spread_temperature_2m_agl_stddev",
    "GEFS 2m AGL Temperature Spread",
    ProductFamily::Surface,
    GribLevelKind::HeightAboveGround,
    Some(2),
    Some(
        FieldSelector::height_agl(CanonicalField::Temperature, 2)
            .with_ensemble_standard_deviation(),
    ),
    &["TMP:2 m above ground"],
);

const FIELD_GEFS_AVG_2M_RH: GribFieldSpec = field_spec(
    "gefs_mean_relative_humidity_2m_agl",
    "GEFS 2m AGL Relative Humidity Mean",
    ProductFamily::Surface,
    GribLevelKind::HeightAboveGround,
    Some(2),
    Some(FieldSelector::height_agl(CanonicalField::RelativeHumidity, 2).with_ensemble_mean()),
    &["RH:2 m above ground"],
);

const FIELD_GEFS_SPR_2M_RH_STDDEV: GribFieldSpec = field_spec(
    "gefs_spread_relative_humidity_2m_agl_stddev",
    "GEFS 2m AGL Relative Humidity Spread",
    ProductFamily::Surface,
    GribLevelKind::HeightAboveGround,
    Some(2),
    Some(
        FieldSelector::height_agl(CanonicalField::RelativeHumidity, 2)
            .with_ensemble_standard_deviation(),
    ),
    &["RH:2 m above ground"],
);

const fn qmd_height_agl_stat_field_spec(
    key: &'static str,
    label: &'static str,
    field: CanonicalField,
    height_m: u16,
    product: FieldProduct,
    idx_patterns: &'static [&'static str],
) -> GribFieldSpec {
    field_spec(
        key,
        label,
        ProductFamily::Surface,
        GribLevelKind::HeightAboveGround,
        Some(height_m as i32),
        Some(FieldSelector::height_agl(field, height_m).with_product(product)),
        idx_patterns,
    )
}

const FIELD_QMD_2M_TEMP_MEAN: GribFieldSpec = field_spec(
    "qmd_temperature_2m_agl_mean",
    "NBM QMD 2m AGL Temperature Mean",
    ProductFamily::Surface,
    GribLevelKind::HeightAboveGround,
    Some(2),
    Some(FieldSelector::height_agl(CanonicalField::Temperature, 2).with_ensemble_mean()),
    &["TMP:2 m above ground"],
);

const FIELD_QMD_2M_TEMP_STDDEV: GribFieldSpec = field_spec(
    "qmd_temperature_2m_agl_stddev",
    "NBM QMD 2m AGL Temperature Std Dev",
    ProductFamily::Surface,
    GribLevelKind::HeightAboveGround,
    Some(2),
    Some(
        FieldSelector::height_agl(CanonicalField::Temperature, 2)
            .with_ensemble_standard_deviation(),
    ),
    &["TMP:2 m above ground"],
);

const FIELD_QMD_2M_TEMP_P05: GribFieldSpec = field_spec(
    "qmd_temperature_2m_agl_p05",
    "NBM QMD 2m AGL Temperature P05",
    ProductFamily::Surface,
    GribLevelKind::HeightAboveGround,
    Some(2),
    Some(FieldSelector::height_agl(CanonicalField::Temperature, 2).with_percentile(5)),
    &["TMP:2 m above ground"],
);

const FIELD_QMD_2M_TEMP_P10: GribFieldSpec = field_spec(
    "qmd_temperature_2m_agl_p10",
    "NBM QMD 2m AGL Temperature P10",
    ProductFamily::Surface,
    GribLevelKind::HeightAboveGround,
    Some(2),
    Some(FieldSelector::height_agl(CanonicalField::Temperature, 2).with_percentile(10)),
    &["TMP:2 m above ground"],
);

const FIELD_QMD_2M_TEMP_P25: GribFieldSpec = field_spec(
    "qmd_temperature_2m_agl_p25",
    "NBM QMD 2m AGL Temperature P25",
    ProductFamily::Surface,
    GribLevelKind::HeightAboveGround,
    Some(2),
    Some(FieldSelector::height_agl(CanonicalField::Temperature, 2).with_percentile(25)),
    &["TMP:2 m above ground"],
);

const FIELD_QMD_2M_TEMP_P50: GribFieldSpec = field_spec(
    "qmd_temperature_2m_agl_p50",
    "NBM QMD 2m AGL Temperature P50",
    ProductFamily::Surface,
    GribLevelKind::HeightAboveGround,
    Some(2),
    Some(FieldSelector::height_agl(CanonicalField::Temperature, 2).with_percentile(50)),
    &["TMP:2 m above ground"],
);

const FIELD_QMD_2M_TEMP_P75: GribFieldSpec = field_spec(
    "qmd_temperature_2m_agl_p75",
    "NBM QMD 2m AGL Temperature P75",
    ProductFamily::Surface,
    GribLevelKind::HeightAboveGround,
    Some(2),
    Some(FieldSelector::height_agl(CanonicalField::Temperature, 2).with_percentile(75)),
    &["TMP:2 m above ground"],
);

const FIELD_QMD_2M_TEMP_P90: GribFieldSpec = field_spec(
    "qmd_temperature_2m_agl_p90",
    "NBM QMD 2m AGL Temperature P90",
    ProductFamily::Surface,
    GribLevelKind::HeightAboveGround,
    Some(2),
    Some(FieldSelector::height_agl(CanonicalField::Temperature, 2).with_percentile(90)),
    &["TMP:2 m above ground"],
);

const FIELD_QMD_2M_TEMP_P95: GribFieldSpec = field_spec(
    "qmd_temperature_2m_agl_p95",
    "NBM QMD 2m AGL Temperature P95",
    ProductFamily::Surface,
    GribLevelKind::HeightAboveGround,
    Some(2),
    Some(FieldSelector::height_agl(CanonicalField::Temperature, 2).with_percentile(95)),
    &["TMP:2 m above ground"],
);

const FIELD_QMD_PROB_2M_TEMP_BELOW_270P928K: GribFieldSpec = qmd_height_agl_stat_field_spec(
    "qmd_probability_temperature_2m_agl_lt_270p928k",
    "NBM QMD Probability 2m AGL Temperature < 270.928 K",
    CanonicalField::Temperature,
    2,
    FieldProduct::Probability(ProbabilitySelection::below_milli(270_928)),
    &["TMP:2 m above ground"],
);

const FIELD_QMD_PROB_2M_TEMP_BELOW_273P15K: GribFieldSpec = qmd_height_agl_stat_field_spec(
    "qmd_probability_temperature_2m_agl_lt_273p15k",
    "NBM QMD Probability 2m AGL Temperature < 273.15 K",
    CanonicalField::Temperature,
    2,
    FieldProduct::Probability(ProbabilitySelection::below_milli(273_150)),
    &["TMP:2 m above ground"],
);

const FIELD_QMD_PROB_2M_TEMP_ABOVE_299P817K: GribFieldSpec = qmd_height_agl_stat_field_spec(
    "qmd_probability_temperature_2m_agl_gt_299p817k",
    "NBM QMD Probability 2m AGL Temperature > 299.817 K",
    CanonicalField::Temperature,
    2,
    FieldProduct::Probability(ProbabilitySelection::above_milli(299_817)),
    &["TMP:2 m above ground"],
);

const FIELD_QMD_PROB_2M_TEMP_ABOVE_305P372K: GribFieldSpec = qmd_height_agl_stat_field_spec(
    "qmd_probability_temperature_2m_agl_gt_305p372k",
    "NBM QMD Probability 2m AGL Temperature > 305.372 K",
    CanonicalField::Temperature,
    2,
    FieldProduct::Probability(ProbabilitySelection::above_milli(305_372)),
    &["TMP:2 m above ground"],
);

const FIELD_QMD_PROB_2M_TEMP_ABOVE_310P928K: GribFieldSpec = qmd_height_agl_stat_field_spec(
    "qmd_probability_temperature_2m_agl_gt_310p928k",
    "NBM QMD Probability 2m AGL Temperature > 310.928 K",
    CanonicalField::Temperature,
    2,
    FieldProduct::Probability(ProbabilitySelection::above_milli(310_928)),
    &["TMP:2 m above ground"],
);

const FIELD_QMD_PROB_2M_TEMP_ABOVE_316P483K: GribFieldSpec = qmd_height_agl_stat_field_spec(
    "qmd_probability_temperature_2m_agl_gt_316p483k",
    "NBM QMD Probability 2m AGL Temperature > 316.483 K",
    CanonicalField::Temperature,
    2,
    FieldProduct::Probability(ProbabilitySelection::above_milli(316_483)),
    &["TMP:2 m above ground"],
);

const FIELD_QMD_2M_DEWPOINT_MEAN: GribFieldSpec = qmd_height_agl_stat_field_spec(
    "qmd_dewpoint_2m_agl_mean",
    "NBM QMD 2m AGL Dewpoint Mean",
    CanonicalField::Dewpoint,
    2,
    FieldProduct::EnsembleMean,
    &["DPT:2 m above ground"],
);

const FIELD_QMD_2M_DEWPOINT_STDDEV: GribFieldSpec = qmd_height_agl_stat_field_spec(
    "qmd_dewpoint_2m_agl_stddev",
    "NBM QMD 2m AGL Dewpoint Std Dev",
    CanonicalField::Dewpoint,
    2,
    FieldProduct::EnsembleStandardDeviation,
    &["DPT:2 m above ground"],
);

const FIELD_QMD_2M_DEWPOINT_P05: GribFieldSpec = qmd_height_agl_stat_field_spec(
    "qmd_dewpoint_2m_agl_p05",
    "NBM QMD 2m AGL Dewpoint P05",
    CanonicalField::Dewpoint,
    2,
    FieldProduct::Percentile(5),
    &["DPT:2 m above ground"],
);

const FIELD_QMD_2M_DEWPOINT_P10: GribFieldSpec = qmd_height_agl_stat_field_spec(
    "qmd_dewpoint_2m_agl_p10",
    "NBM QMD 2m AGL Dewpoint P10",
    CanonicalField::Dewpoint,
    2,
    FieldProduct::Percentile(10),
    &["DPT:2 m above ground"],
);

const FIELD_QMD_2M_DEWPOINT_P25: GribFieldSpec = qmd_height_agl_stat_field_spec(
    "qmd_dewpoint_2m_agl_p25",
    "NBM QMD 2m AGL Dewpoint P25",
    CanonicalField::Dewpoint,
    2,
    FieldProduct::Percentile(25),
    &["DPT:2 m above ground"],
);

const FIELD_QMD_2M_DEWPOINT_P50: GribFieldSpec = qmd_height_agl_stat_field_spec(
    "qmd_dewpoint_2m_agl_p50",
    "NBM QMD 2m AGL Dewpoint P50",
    CanonicalField::Dewpoint,
    2,
    FieldProduct::Percentile(50),
    &["DPT:2 m above ground"],
);

const FIELD_QMD_2M_DEWPOINT_P75: GribFieldSpec = qmd_height_agl_stat_field_spec(
    "qmd_dewpoint_2m_agl_p75",
    "NBM QMD 2m AGL Dewpoint P75",
    CanonicalField::Dewpoint,
    2,
    FieldProduct::Percentile(75),
    &["DPT:2 m above ground"],
);

const FIELD_QMD_2M_DEWPOINT_P90: GribFieldSpec = qmd_height_agl_stat_field_spec(
    "qmd_dewpoint_2m_agl_p90",
    "NBM QMD 2m AGL Dewpoint P90",
    CanonicalField::Dewpoint,
    2,
    FieldProduct::Percentile(90),
    &["DPT:2 m above ground"],
);

const FIELD_QMD_2M_DEWPOINT_P95: GribFieldSpec = qmd_height_agl_stat_field_spec(
    "qmd_dewpoint_2m_agl_p95",
    "NBM QMD 2m AGL Dewpoint P95",
    CanonicalField::Dewpoint,
    2,
    FieldProduct::Percentile(95),
    &["DPT:2 m above ground"],
);

const FIELD_QMD_PROB_2M_DEWPOINT_BELOW_273P15K: GribFieldSpec = qmd_height_agl_stat_field_spec(
    "qmd_probability_dewpoint_2m_agl_lt_273p15k",
    "NBM QMD Probability 2m AGL Dewpoint < 273.15 K",
    CanonicalField::Dewpoint,
    2,
    FieldProduct::Probability(ProbabilitySelection::below_milli(273_150)),
    &["DPT:2 m above ground"],
);

const FIELD_QMD_PROB_2M_DEWPOINT_ABOVE_288P706K: GribFieldSpec = qmd_height_agl_stat_field_spec(
    "qmd_probability_dewpoint_2m_agl_gt_288p706k",
    "NBM QMD Probability 2m AGL Dewpoint > 288.706 K",
    CanonicalField::Dewpoint,
    2,
    FieldProduct::Probability(ProbabilitySelection::above_milli(288_706)),
    &["DPT:2 m above ground"],
);

const FIELD_QMD_PROB_2M_DEWPOINT_ABOVE_291P483K: GribFieldSpec = qmd_height_agl_stat_field_spec(
    "qmd_probability_dewpoint_2m_agl_gt_291p483k",
    "NBM QMD Probability 2m AGL Dewpoint > 291.483 K",
    CanonicalField::Dewpoint,
    2,
    FieldProduct::Probability(ProbabilitySelection::above_milli(291_483)),
    &["DPT:2 m above ground"],
);

const FIELD_QMD_PROB_2M_DEWPOINT_ABOVE_294P261K: GribFieldSpec = qmd_height_agl_stat_field_spec(
    "qmd_probability_dewpoint_2m_agl_gt_294p261k",
    "NBM QMD Probability 2m AGL Dewpoint > 294.261 K",
    CanonicalField::Dewpoint,
    2,
    FieldProduct::Probability(ProbabilitySelection::above_milli(294_261)),
    &["DPT:2 m above ground"],
);

const FIELD_QMD_PROB_2M_DEWPOINT_ABOVE_297P039K: GribFieldSpec = qmd_height_agl_stat_field_spec(
    "qmd_probability_dewpoint_2m_agl_gt_297p039k",
    "NBM QMD Probability 2m AGL Dewpoint > 297.039 K",
    CanonicalField::Dewpoint,
    2,
    FieldProduct::Probability(ProbabilitySelection::above_milli(297_039)),
    &["DPT:2 m above ground"],
);

const FIELD_QMD_2M_RH_P05: GribFieldSpec = qmd_height_agl_stat_field_spec(
    "qmd_relative_humidity_2m_agl_p05",
    "NBM QMD 2m AGL Relative Humidity P05",
    CanonicalField::RelativeHumidity,
    2,
    FieldProduct::Percentile(5),
    &["RH:2 m above ground"],
);

const FIELD_QMD_2M_RH_P10: GribFieldSpec = qmd_height_agl_stat_field_spec(
    "qmd_relative_humidity_2m_agl_p10",
    "NBM QMD 2m AGL Relative Humidity P10",
    CanonicalField::RelativeHumidity,
    2,
    FieldProduct::Percentile(10),
    &["RH:2 m above ground"],
);

const FIELD_QMD_2M_RH_P25: GribFieldSpec = qmd_height_agl_stat_field_spec(
    "qmd_relative_humidity_2m_agl_p25",
    "NBM QMD 2m AGL Relative Humidity P25",
    CanonicalField::RelativeHumidity,
    2,
    FieldProduct::Percentile(25),
    &["RH:2 m above ground"],
);

const FIELD_QMD_2M_RH_P50: GribFieldSpec = qmd_height_agl_stat_field_spec(
    "qmd_relative_humidity_2m_agl_p50",
    "NBM QMD 2m AGL Relative Humidity P50",
    CanonicalField::RelativeHumidity,
    2,
    FieldProduct::Percentile(50),
    &["RH:2 m above ground"],
);

const FIELD_QMD_2M_RH_P75: GribFieldSpec = qmd_height_agl_stat_field_spec(
    "qmd_relative_humidity_2m_agl_p75",
    "NBM QMD 2m AGL Relative Humidity P75",
    CanonicalField::RelativeHumidity,
    2,
    FieldProduct::Percentile(75),
    &["RH:2 m above ground"],
);

const FIELD_QMD_2M_RH_P90: GribFieldSpec = qmd_height_agl_stat_field_spec(
    "qmd_relative_humidity_2m_agl_p90",
    "NBM QMD 2m AGL Relative Humidity P90",
    CanonicalField::RelativeHumidity,
    2,
    FieldProduct::Percentile(90),
    &["RH:2 m above ground"],
);

const FIELD_QMD_2M_RH_P95: GribFieldSpec = qmd_height_agl_stat_field_spec(
    "qmd_relative_humidity_2m_agl_p95",
    "NBM QMD 2m AGL Relative Humidity P95",
    CanonicalField::RelativeHumidity,
    2,
    FieldProduct::Percentile(95),
    &["RH:2 m above ground"],
);

const FIELD_AIGEFS_SPR_2M_TEMP_STDDEV: GribFieldSpec = field_spec(
    "aigefs_spread_temperature_2m_agl_stddev",
    "AI-GEFS 2m AGL Temperature Spread",
    ProductFamily::Surface,
    GribLevelKind::HeightAboveGround,
    Some(2),
    Some(
        FieldSelector::height_agl(CanonicalField::Temperature, 2)
            .with_ensemble_standard_deviation(),
    ),
    &["TMP:2 m above ground"],
);

const FIELD_HGEFS_SPR_2M_TEMP_STDDEV: GribFieldSpec = field_spec(
    "hgefs_spread_temperature_2m_agl_stddev",
    "HGEFS 2m AGL Temperature Spread",
    ProductFamily::Surface,
    GribLevelKind::HeightAboveGround,
    Some(2),
    Some(
        FieldSelector::height_agl(CanonicalField::Temperature, 2)
            .with_ensemble_standard_deviation(),
    ),
    &["TMP:2 m above ground"],
);

const FIELD_HREF_SPRD_2M_TEMP: GribFieldSpec = field_spec(
    "href_spread_temperature_2m_agl",
    "HREF 2m AGL Temperature Spread",
    ProductFamily::Surface,
    GribLevelKind::HeightAboveGround,
    Some(2),
    Some(
        FieldSelector::height_agl(CanonicalField::Temperature, 2)
            .with_product(FieldProduct::EnsembleSpread),
    ),
    &["TMP:2 m above ground"],
);

const FIELD_HREF_MEAN_2M_TEMP: GribFieldSpec = field_spec(
    "href_mean_temperature_2m_agl",
    "HREF 2m AGL Temperature Mean",
    ProductFamily::Surface,
    GribLevelKind::HeightAboveGround,
    Some(2),
    Some(FieldSelector::height_agl(CanonicalField::Temperature, 2).with_ensemble_mean()),
    &["TMP:2 m above ground"],
);

const FIELD_REFS_SPRD_2M_TEMP: GribFieldSpec = field_spec(
    "refs_spread_temperature_2m_agl",
    "REFS 2m AGL Temperature Spread",
    ProductFamily::Surface,
    GribLevelKind::HeightAboveGround,
    Some(2),
    Some(
        FieldSelector::height_agl(CanonicalField::Temperature, 2)
            .with_product(FieldProduct::EnsembleSpread),
    ),
    &["TMP:2 m above ground"],
);

const FIELD_2M_DEWPOINT: GribFieldSpec = field_spec(
    "dewpoint_2m_agl",
    "2m AGL Dewpoint",
    ProductFamily::Surface,
    GribLevelKind::HeightAboveGround,
    Some(2),
    Some(FieldSelector::height_agl(CanonicalField::Dewpoint, 2)),
    &["DPT:2 m above ground"],
);

const FIELD_HREF_SPRD_2M_DEWPOINT: GribFieldSpec = field_spec(
    "href_spread_dewpoint_2m_agl",
    "HREF 2m AGL Dewpoint Spread",
    ProductFamily::Surface,
    GribLevelKind::HeightAboveGround,
    Some(2),
    Some(
        FieldSelector::height_agl(CanonicalField::Dewpoint, 2)
            .with_product(FieldProduct::EnsembleSpread),
    ),
    &["DPT:2 m above ground"],
);

const FIELD_HREF_MEAN_2M_DEWPOINT: GribFieldSpec = field_spec(
    "href_mean_dewpoint_2m_agl",
    "HREF 2m AGL Dewpoint Mean",
    ProductFamily::Surface,
    GribLevelKind::HeightAboveGround,
    Some(2),
    Some(FieldSelector::height_agl(CanonicalField::Dewpoint, 2).with_ensemble_mean()),
    &["DPT:2 m above ground"],
);

const FIELD_REFS_SPRD_2M_DEWPOINT: GribFieldSpec = field_spec(
    "refs_spread_dewpoint_2m_agl",
    "REFS 2m AGL Dewpoint Spread",
    ProductFamily::Surface,
    GribLevelKind::HeightAboveGround,
    Some(2),
    Some(
        FieldSelector::height_agl(CanonicalField::Dewpoint, 2)
            .with_product(FieldProduct::EnsembleSpread),
    ),
    &["DPT:2 m above ground"],
);

const FIELD_2M_RH: GribFieldSpec = field_spec(
    "relative_humidity_2m_agl",
    "2m AGL Relative Humidity",
    ProductFamily::Surface,
    GribLevelKind::HeightAboveGround,
    Some(2),
    Some(FieldSelector::height_agl(
        CanonicalField::RelativeHumidity,
        2,
    )),
    &["RH:2 m above ground"],
);

const FIELD_10M_U: GribFieldSpec = field_spec(
    "u_10m_agl",
    "10m AGL U Wind",
    ProductFamily::Surface,
    GribLevelKind::HeightAboveGround,
    Some(10),
    Some(FieldSelector::height_agl(CanonicalField::UWind, 10)),
    &["UGRD:10 m above ground"],
);

const FIELD_10M_V: GribFieldSpec = field_spec(
    "v_10m_agl",
    "10m AGL V Wind",
    ProductFamily::Surface,
    GribLevelKind::HeightAboveGround,
    Some(10),
    Some(FieldSelector::height_agl(CanonicalField::VWind, 10)),
    &["VGRD:10 m above ground"],
);

/// Instantaneous 10 m wind SPEED -- the scalar the standalone surface
/// wind product fills with.  It is a first-class field, not a companion
/// of a pressure chart: every store lane already carries it (the WRF
/// import publishes wrf-core's `wspd10` under this exact selector, the
/// post-processed `wrf2d` route computes it from U10/V10, and the GRIB
/// decoder maps it to parameter 0/2/1), and nothing in the catalog
/// rendered it until `10m_wind_speed_and_direction` existed.
const FIELD_10M_WIND_SPEED: GribFieldSpec = field_spec(
    "wind_speed_10m_agl",
    "10m AGL Wind Speed",
    ProductFamily::Surface,
    GribLevelKind::HeightAboveGround,
    Some(10),
    Some(FieldSelector::height_agl(CanonicalField::WindSpeed, 10)),
    &["WIND:10 m above ground"],
);

const FIELD_GEFS_AVG_10M_U: GribFieldSpec = field_spec(
    "gefs_mean_u_10m_agl",
    "GEFS 10m AGL U Wind Mean",
    ProductFamily::Surface,
    GribLevelKind::HeightAboveGround,
    Some(10),
    Some(FieldSelector::height_agl(CanonicalField::UWind, 10).with_ensemble_mean()),
    &["UGRD:10 m above ground"],
);

const FIELD_GEFS_AVG_10M_V: GribFieldSpec = field_spec(
    "gefs_mean_v_10m_agl",
    "GEFS 10m AGL V Wind Mean",
    ProductFamily::Surface,
    GribLevelKind::HeightAboveGround,
    Some(10),
    Some(FieldSelector::height_agl(CanonicalField::VWind, 10).with_ensemble_mean()),
    &["VGRD:10 m above ground"],
);

const FIELD_10M_WIND_GUST: GribFieldSpec = field_spec(
    "wind_gust_10m_agl",
    "10m AGL Wind Gust",
    ProductFamily::Surface,
    GribLevelKind::HeightAboveGround,
    Some(10),
    Some(FieldSelector::height_agl(CanonicalField::WindGust, 10)),
    // GRIB inventory spellings ONLY.  `WSPD10MAX` used to sit in this
    // list; it is not a GRIB idx pattern at all but a WRF Registry
    // variable, and WRF 4.6.1 defines no wind gust: `module_diag_nwp`
    // titles WSPD10MAX "WIND SPD MAX 10 M", the running maximum of
    // sqrt(u10^2+v10^2) between history writes.  Matching it here made
    // a running maximum resolve a field labelled "Wind Gust", which is
    // a different physical quantity.  A gust reaches this field only
    // from a source whose own files publish one (GRIB2 discipline 0,
    // category 2, number 22).
    &["GUST:surface", "GUST:10 m above ground"],
);

const FIELD_QMD_10M_WIND_GUST_MEAN: GribFieldSpec = qmd_height_agl_stat_field_spec(
    "qmd_wind_gust_10m_agl_mean",
    "NBM QMD 10m AGL Wind Gust Mean",
    CanonicalField::WindGust,
    10,
    FieldProduct::EnsembleMean,
    &["GUST:10 m above ground"],
);

const FIELD_QMD_10M_WIND_GUST_STDDEV: GribFieldSpec = qmd_height_agl_stat_field_spec(
    "qmd_wind_gust_10m_agl_stddev",
    "NBM QMD 10m AGL Wind Gust Std Dev",
    CanonicalField::WindGust,
    10,
    FieldProduct::EnsembleStandardDeviation,
    &["GUST:10 m above ground"],
);

const FIELD_QMD_PROB_10M_WIND_GUST_ABOVE_17P4911MS: GribFieldSpec = qmd_height_agl_stat_field_spec(
    "qmd_probability_wind_gust_10m_agl_gt_17p4911ms",
    "NBM QMD Probability 10m AGL Wind Gust > 17.4911 m/s",
    CanonicalField::WindGust,
    10,
    FieldProduct::Probability(ProbabilitySelection::above_milli(17_491)),
    &["GUST:10 m above ground"],
);

const FIELD_QMD_PROB_10M_WIND_GUST_ABOVE_21P0922MS: GribFieldSpec = qmd_height_agl_stat_field_spec(
    "qmd_probability_wind_gust_10m_agl_gt_21p0922ms",
    "NBM QMD Probability 10m AGL Wind Gust > 21.0922 m/s",
    CanonicalField::WindGust,
    10,
    FieldProduct::Probability(ProbabilitySelection::above_milli(21_092)),
    &["GUST:10 m above ground"],
);

const FIELD_QMD_PROB_10M_WIND_GUST_ABOVE_24P6933MS: GribFieldSpec = qmd_height_agl_stat_field_spec(
    "qmd_probability_wind_gust_10m_agl_gt_24p6933ms",
    "NBM QMD Probability 10m AGL Wind Gust > 24.6933 m/s",
    CanonicalField::WindGust,
    10,
    FieldProduct::Probability(ProbabilitySelection::above_milli(24_693)),
    &["GUST:10 m above ground"],
);

const FIELD_QMD_PROB_10M_WIND_GUST_ABOVE_28P8089MS: GribFieldSpec = qmd_height_agl_stat_field_spec(
    "qmd_probability_wind_gust_10m_agl_gt_28p8089ms",
    "NBM QMD Probability 10m AGL Wind Gust > 28.8089 m/s",
    CanonicalField::WindGust,
    10,
    FieldProduct::Probability(ProbabilitySelection::above_milli(28_809)),
    &["GUST:10 m above ground"],
);

const FIELD_QMD_PROB_10M_WIND_GUST_ABOVE_32P9244MS: GribFieldSpec = qmd_height_agl_stat_field_spec(
    "qmd_probability_wind_gust_10m_agl_gt_32p9244ms",
    "NBM QMD Probability 10m AGL Wind Gust > 32.9244 m/s",
    CanonicalField::WindGust,
    10,
    FieldProduct::Probability(ProbabilitySelection::above_milli(32_924)),
    &["GUST:10 m above ground"],
);

const FIELD_QMD_10M_WIND_GUST_P05: GribFieldSpec = qmd_height_agl_stat_field_spec(
    "qmd_wind_gust_10m_agl_p05",
    "NBM QMD 10m AGL Wind Gust P05",
    CanonicalField::WindGust,
    10,
    FieldProduct::Percentile(5),
    &["GUST:10 m above ground"],
);

const FIELD_QMD_10M_WIND_GUST_P10: GribFieldSpec = qmd_height_agl_stat_field_spec(
    "qmd_wind_gust_10m_agl_p10",
    "NBM QMD 10m AGL Wind Gust P10",
    CanonicalField::WindGust,
    10,
    FieldProduct::Percentile(10),
    &["GUST:10 m above ground"],
);

const FIELD_QMD_10M_WIND_GUST_P25: GribFieldSpec = qmd_height_agl_stat_field_spec(
    "qmd_wind_gust_10m_agl_p25",
    "NBM QMD 10m AGL Wind Gust P25",
    CanonicalField::WindGust,
    10,
    FieldProduct::Percentile(25),
    &["GUST:10 m above ground"],
);

const FIELD_QMD_10M_WIND_GUST_P50: GribFieldSpec = qmd_height_agl_stat_field_spec(
    "qmd_wind_gust_10m_agl_p50",
    "NBM QMD 10m AGL Wind Gust P50",
    CanonicalField::WindGust,
    10,
    FieldProduct::Percentile(50),
    &["GUST:10 m above ground"],
);

const FIELD_QMD_10M_WIND_GUST_P75: GribFieldSpec = qmd_height_agl_stat_field_spec(
    "qmd_wind_gust_10m_agl_p75",
    "NBM QMD 10m AGL Wind Gust P75",
    CanonicalField::WindGust,
    10,
    FieldProduct::Percentile(75),
    &["GUST:10 m above ground"],
);

const FIELD_QMD_10M_WIND_GUST_P90: GribFieldSpec = qmd_height_agl_stat_field_spec(
    "qmd_wind_gust_10m_agl_p90",
    "NBM QMD 10m AGL Wind Gust P90",
    CanonicalField::WindGust,
    10,
    FieldProduct::Percentile(90),
    &["GUST:10 m above ground"],
);

const FIELD_QMD_10M_WIND_GUST_P95: GribFieldSpec = qmd_height_agl_stat_field_spec(
    "qmd_wind_gust_10m_agl_p95",
    "NBM QMD 10m AGL Wind Gust P95",
    CanonicalField::WindGust,
    10,
    FieldProduct::Percentile(95),
    &["GUST:10 m above ground"],
);

const FIELD_QMD_10M_WIND_SPEED_MEAN: GribFieldSpec = qmd_height_agl_stat_field_spec(
    "qmd_wind_speed_10m_agl_mean",
    "NBM QMD 10m AGL Wind Speed Mean",
    CanonicalField::WindSpeed,
    10,
    FieldProduct::EnsembleMean,
    &["WIND:10 m above ground"],
);

const FIELD_QMD_10M_WIND_SPEED_STDDEV: GribFieldSpec = qmd_height_agl_stat_field_spec(
    "qmd_wind_speed_10m_agl_stddev",
    "NBM QMD 10m AGL Wind Speed Std Dev",
    CanonicalField::WindSpeed,
    10,
    FieldProduct::EnsembleStandardDeviation,
    &["WIND:10 m above ground"],
);

const FIELD_QMD_PROB_10M_WIND_SPEED_ABOVE_8P7456MS: GribFieldSpec = qmd_height_agl_stat_field_spec(
    "qmd_probability_wind_speed_10m_agl_gt_8p7456ms",
    "NBM QMD Probability 10m AGL Wind Speed > 8.7456 m/s",
    CanonicalField::WindSpeed,
    10,
    FieldProduct::Probability(ProbabilitySelection::above_milli(8_746)),
    &["WIND:10 m above ground"],
);

const FIELD_QMD_PROB_10M_WIND_SPEED_ABOVE_11P3177MS: GribFieldSpec = qmd_height_agl_stat_field_spec(
    "qmd_probability_wind_speed_10m_agl_gt_11p3177ms",
    "NBM QMD Probability 10m AGL Wind Speed > 11.3177 m/s",
    CanonicalField::WindSpeed,
    10,
    FieldProduct::Probability(ProbabilitySelection::above_milli(11_318)),
    &["WIND:10 m above ground"],
);

const FIELD_QMD_PROB_10M_WIND_SPEED_ABOVE_15P4333MS: GribFieldSpec = qmd_height_agl_stat_field_spec(
    "qmd_probability_wind_speed_10m_agl_gt_15p4333ms",
    "NBM QMD Probability 10m AGL Wind Speed > 15.4333 m/s",
    CanonicalField::WindSpeed,
    10,
    FieldProduct::Probability(ProbabilitySelection::above_milli(15_433)),
    &["WIND:10 m above ground"],
);

const FIELD_QMD_PROB_10M_WIND_SPEED_ABOVE_17P4911MS: GribFieldSpec = qmd_height_agl_stat_field_spec(
    "qmd_probability_wind_speed_10m_agl_gt_17p4911ms",
    "NBM QMD Probability 10m AGL Wind Speed > 17.4911 m/s",
    CanonicalField::WindSpeed,
    10,
    FieldProduct::Probability(ProbabilitySelection::above_milli(17_491)),
    &["WIND:10 m above ground"],
);

const FIELD_QMD_PROB_10M_WIND_SPEED_ABOVE_24P6933MS: GribFieldSpec = qmd_height_agl_stat_field_spec(
    "qmd_probability_wind_speed_10m_agl_gt_24p6933ms",
    "NBM QMD Probability 10m AGL Wind Speed > 24.6933 m/s",
    CanonicalField::WindSpeed,
    10,
    FieldProduct::Probability(ProbabilitySelection::above_milli(24_693)),
    &["WIND:10 m above ground"],
);

const FIELD_QMD_10M_WIND_SPEED_P05: GribFieldSpec = qmd_height_agl_stat_field_spec(
    "qmd_wind_speed_10m_agl_p05",
    "NBM QMD 10m AGL Wind Speed P05",
    CanonicalField::WindSpeed,
    10,
    FieldProduct::Percentile(5),
    &["WIND:10 m above ground"],
);

const FIELD_QMD_10M_WIND_SPEED_P10: GribFieldSpec = qmd_height_agl_stat_field_spec(
    "qmd_wind_speed_10m_agl_p10",
    "NBM QMD 10m AGL Wind Speed P10",
    CanonicalField::WindSpeed,
    10,
    FieldProduct::Percentile(10),
    &["WIND:10 m above ground"],
);

const FIELD_QMD_10M_WIND_SPEED_P25: GribFieldSpec = qmd_height_agl_stat_field_spec(
    "qmd_wind_speed_10m_agl_p25",
    "NBM QMD 10m AGL Wind Speed P25",
    CanonicalField::WindSpeed,
    10,
    FieldProduct::Percentile(25),
    &["WIND:10 m above ground"],
);

const FIELD_QMD_10M_WIND_SPEED_P50: GribFieldSpec = qmd_height_agl_stat_field_spec(
    "qmd_wind_speed_10m_agl_p50",
    "NBM QMD 10m AGL Wind Speed P50",
    CanonicalField::WindSpeed,
    10,
    FieldProduct::Percentile(50),
    &["WIND:10 m above ground"],
);

const FIELD_QMD_10M_WIND_SPEED_P75: GribFieldSpec = qmd_height_agl_stat_field_spec(
    "qmd_wind_speed_10m_agl_p75",
    "NBM QMD 10m AGL Wind Speed P75",
    CanonicalField::WindSpeed,
    10,
    FieldProduct::Percentile(75),
    &["WIND:10 m above ground"],
);

const FIELD_QMD_10M_WIND_SPEED_P90: GribFieldSpec = qmd_height_agl_stat_field_spec(
    "qmd_wind_speed_10m_agl_p90",
    "NBM QMD 10m AGL Wind Speed P90",
    CanonicalField::WindSpeed,
    10,
    FieldProduct::Percentile(90),
    &["WIND:10 m above ground"],
);

const FIELD_QMD_10M_WIND_SPEED_P95: GribFieldSpec = qmd_height_agl_stat_field_spec(
    "qmd_wind_speed_10m_agl_p95",
    "NBM QMD 10m AGL Wind Speed P95",
    CanonicalField::WindSpeed,
    10,
    FieldProduct::Percentile(95),
    &["WIND:10 m above ground"],
);

const FIELD_HREF_SPRD_10M_WIND_SPEED: GribFieldSpec = field_spec(
    "href_spread_wind_speed_10m_agl",
    "HREF 10m AGL Wind Speed Spread",
    ProductFamily::Surface,
    GribLevelKind::HeightAboveGround,
    Some(10),
    Some(
        FieldSelector::height_agl(CanonicalField::WindSpeed, 10)
            .with_product(FieldProduct::EnsembleSpread),
    ),
    &["WIND:10 m above ground"],
);

const FIELD_HREF_MEAN_10M_WIND_SPEED: GribFieldSpec = field_spec(
    "href_mean_wind_speed_10m_agl",
    "HREF 10m AGL Wind Speed Mean",
    ProductFamily::Surface,
    GribLevelKind::HeightAboveGround,
    Some(10),
    Some(FieldSelector::height_agl(CanonicalField::WindSpeed, 10).with_ensemble_mean()),
    &["WIND:10 m above ground"],
);

const FIELD_REFS_SPRD_10M_WIND_SPEED: GribFieldSpec = field_spec(
    "refs_spread_wind_speed_10m_agl",
    "REFS 10m AGL Wind Speed Spread",
    ProductFamily::Surface,
    GribLevelKind::HeightAboveGround,
    Some(10),
    Some(
        FieldSelector::height_agl(CanonicalField::WindSpeed, 10)
            .with_product(FieldProduct::EnsembleSpread),
    ),
    &["WIND:10 m above ground"],
);

const FIELD_SREF_PROB_2M_TEMP_BELOW_FREEZING: GribFieldSpec = field_spec(
    "sref_probability_temperature_2m_agl_lt_273k",
    "SREF Probability 2m Temperature < 273 K",
    ProductFamily::Surface,
    GribLevelKind::HeightAboveGround,
    Some(2),
    Some(
        FieldSelector::height_agl(CanonicalField::Temperature, 2)
            .with_probability(ProbabilitySelection::below_milli(273_000)),
    ),
    &["TMP:2 m above ground"],
);

const FIELD_SREF_PROB_2M_TEMP_ABOVE_298P8K: GribFieldSpec = field_spec(
    "sref_probability_temperature_2m_agl_gt_298p8k",
    "SREF Probability 2m Temperature > 298.8 K",
    ProductFamily::Surface,
    GribLevelKind::HeightAboveGround,
    Some(2),
    Some(
        FieldSelector::height_agl(CanonicalField::Temperature, 2)
            .with_probability(ProbabilitySelection::above_milli(298_800)),
    ),
    &["TMP:2 m above ground"],
);

const FIELD_SREF_PROB_850MB_TEMP_BELOW_FREEZING: GribFieldSpec = field_spec(
    "sref_probability_temperature_850mb_lt_273k",
    "SREF Probability 850mb Temperature < 273 K",
    ProductFamily::Pressure,
    GribLevelKind::IsobaricHpa,
    Some(850),
    Some(
        FieldSelector::isobaric(CanonicalField::Temperature, 850)
            .with_probability(ProbabilitySelection::below_milli(273_000)),
    ),
    &["TMP:850 mb"],
);

const FIELD_SREF_PROB_VISIBILITY_BELOW_ONE_MILE: GribFieldSpec = field_spec(
    "sref_probability_visibility_surface_lt_1609m",
    "SREF Probability Visibility < 1609 m",
    ProductFamily::Surface,
    GribLevelKind::Surface,
    None,
    Some(
        FieldSelector::surface(CanonicalField::Visibility)
            .with_probability(ProbabilitySelection::below_milli(1_609_000)),
    ),
    &["VIS:surface"],
);

const FIELD_SREF_PROB_VISIBILITY_BELOW_402M: GribFieldSpec = field_spec(
    "sref_probability_visibility_surface_lt_402m",
    "SREF Probability Visibility < 402 m",
    ProductFamily::Surface,
    GribLevelKind::Surface,
    None,
    Some(
        FieldSelector::surface(CanonicalField::Visibility)
            .with_probability(ProbabilitySelection::below_milli(402_000)),
    ),
    &["VIS:surface"],
);

const FIELD_SREF_PROB_VISIBILITY_BELOW_804M: GribFieldSpec = field_spec(
    "sref_probability_visibility_surface_lt_804m",
    "SREF Probability Visibility < 804 m",
    ProductFamily::Surface,
    GribLevelKind::Surface,
    None,
    Some(
        FieldSelector::surface(CanonicalField::Visibility)
            .with_probability(ProbabilitySelection::below_milli(804_000)),
    ),
    &["VIS:surface"],
);

const FIELD_SREF_PROB_VISIBILITY_BELOW_3218M: GribFieldSpec = field_spec(
    "sref_probability_visibility_surface_lt_3218m",
    "SREF Probability Visibility < 3218 m",
    ProductFamily::Surface,
    GribLevelKind::Surface,
    None,
    Some(
        FieldSelector::surface(CanonicalField::Visibility)
            .with_probability(ProbabilitySelection::below_milli(3_218_000)),
    ),
    &["VIS:surface"],
);

const FIELD_SREF_PROB_VISIBILITY_BELOW_4827M: GribFieldSpec = field_spec(
    "sref_probability_visibility_surface_lt_4827m",
    "SREF Probability Visibility < 4827 m",
    ProductFamily::Surface,
    GribLevelKind::Surface,
    None,
    Some(
        FieldSelector::surface(CanonicalField::Visibility)
            .with_probability(ProbabilitySelection::below_milli(4_827_000)),
    ),
    &["VIS:surface"],
);

const FIELD_SREF_PROB_VISIBILITY_BELOW_8046M: GribFieldSpec = field_spec(
    "sref_probability_visibility_surface_lt_8046m",
    "SREF Probability Visibility < 8046 m",
    ProductFamily::Surface,
    GribLevelKind::Surface,
    None,
    Some(
        FieldSelector::surface(CanonicalField::Visibility)
            .with_probability(ProbabilitySelection::below_milli(8_046_000)),
    ),
    &["VIS:surface"],
);

const FIELD_SREF_PROB_VISIBILITY_BELOW_9654M: GribFieldSpec = field_spec(
    "sref_probability_visibility_surface_lt_9654m",
    "SREF Probability Visibility < 9654 m",
    ProductFamily::Surface,
    GribLevelKind::Surface,
    None,
    Some(
        FieldSelector::surface(CanonicalField::Visibility)
            .with_probability(ProbabilitySelection::below_milli(9_654_000)),
    ),
    &["VIS:surface"],
);

const FIELD_SREF_PROB_10M_WIND_SPEED_ABOVE_12P89MS: GribFieldSpec = field_spec(
    "sref_probability_wind_speed_10m_agl_gt_12p89ms",
    "SREF Probability 10m Wind Speed > 12.89 m/s",
    ProductFamily::Surface,
    GribLevelKind::HeightAboveGround,
    Some(10),
    Some(
        FieldSelector::height_agl(CanonicalField::WindSpeed, 10)
            .with_probability(ProbabilitySelection::above_milli(12_890)),
    ),
    &["WIND:10 m above ground"],
);

const FIELD_SREF_PROB_10M_WIND_SPEED_ABOVE_17P5MS: GribFieldSpec = field_spec(
    "sref_probability_wind_speed_10m_agl_gt_17p5ms",
    "SREF Probability 10m Wind Speed > 17.5 m/s",
    ProductFamily::Surface,
    GribLevelKind::HeightAboveGround,
    Some(10),
    Some(
        FieldSelector::height_agl(CanonicalField::WindSpeed, 10)
            .with_probability(ProbabilitySelection::above_milli(17_500)),
    ),
    &["WIND:10 m above ground"],
);

const FIELD_SREF_PROB_10M_WIND_SPEED_ABOVE_25P78MS: GribFieldSpec = field_spec(
    "sref_probability_wind_speed_10m_agl_gt_25p78ms",
    "SREF Probability 10m Wind Speed > 25.78 m/s",
    ProductFamily::Surface,
    GribLevelKind::HeightAboveGround,
    Some(10),
    Some(
        FieldSelector::height_agl(CanonicalField::WindSpeed, 10)
            .with_probability(ProbabilitySelection::above_milli(25_780)),
    ),
    &["WIND:10 m above ground"],
);

const FIELD_REFS_PROB_2M_TEMP_BELOW_273P15K: GribFieldSpec = field_spec(
    "refs_probability_temperature_2m_agl_lt_273p15k",
    "REFS Probability 2m Temperature < 273.15 K",
    ProductFamily::Surface,
    GribLevelKind::HeightAboveGround,
    Some(2),
    Some(
        FieldSelector::height_agl(CanonicalField::Temperature, 2)
            .with_probability(ProbabilitySelection::below_milli(273_150)),
    ),
    &["TMP:2 m above ground"],
);

const FIELD_HREF_PROB_2M_TEMP_BELOW_273P15K: GribFieldSpec = field_spec(
    "href_probability_temperature_2m_agl_lt_273p15k",
    "HREF Probability 2m Temperature < 273.15 K",
    ProductFamily::Surface,
    GribLevelKind::HeightAboveGround,
    Some(2),
    Some(
        FieldSelector::height_agl(CanonicalField::Temperature, 2)
            .with_probability(ProbabilitySelection::below_milli(273_150)),
    ),
    &["TMP:2 m above ground"],
);

const FIELD_REFS_PROB_2M_DEWPOINT_ABOVE_291P48K: GribFieldSpec = field_spec(
    "refs_probability_dewpoint_2m_agl_gt_291p48k",
    "REFS Probability 2m Dewpoint > 291.48 K",
    ProductFamily::Surface,
    GribLevelKind::HeightAboveGround,
    Some(2),
    Some(
        FieldSelector::height_agl(CanonicalField::Dewpoint, 2)
            .with_probability(ProbabilitySelection::above_milli(291_480)),
    ),
    &["DPT:2 m above ground"],
);

const FIELD_HREF_PROB_2M_DEWPOINT_ABOVE_291P48K: GribFieldSpec = field_spec(
    "href_probability_dewpoint_2m_agl_gt_291p48k",
    "HREF Probability 2m Dewpoint > 291.48 K",
    ProductFamily::Surface,
    GribLevelKind::HeightAboveGround,
    Some(2),
    Some(
        FieldSelector::height_agl(CanonicalField::Dewpoint, 2)
            .with_probability(ProbabilitySelection::above_milli(291_480)),
    ),
    &["DPT:2 m above ground"],
);

const FIELD_REFS_PROB_2M_DEWPOINT_ABOVE_294P26K: GribFieldSpec = field_spec(
    "refs_probability_dewpoint_2m_agl_gt_294p26k",
    "REFS Probability 2m Dewpoint > 294.26 K",
    ProductFamily::Surface,
    GribLevelKind::HeightAboveGround,
    Some(2),
    Some(
        FieldSelector::height_agl(CanonicalField::Dewpoint, 2)
            .with_probability(ProbabilitySelection::above_milli(294_260)),
    ),
    &["DPT:2 m above ground"],
);

const FIELD_HREF_PROB_2M_DEWPOINT_ABOVE_294P26K: GribFieldSpec = field_spec(
    "href_probability_dewpoint_2m_agl_gt_294p26k",
    "HREF Probability 2m Dewpoint > 294.26 K",
    ProductFamily::Surface,
    GribLevelKind::HeightAboveGround,
    Some(2),
    Some(
        FieldSelector::height_agl(CanonicalField::Dewpoint, 2)
            .with_probability(ProbabilitySelection::above_milli(294_260)),
    ),
    &["DPT:2 m above ground"],
);

const FIELD_REFS_PROB_PWAT_ABOVE_25MM: GribFieldSpec = field_spec(
    "refs_probability_precipitable_water_gt_25mm",
    "REFS Probability Precipitable Water > 25 mm",
    ProductFamily::Surface,
    GribLevelKind::EntireAtmosphere,
    None,
    Some(
        FieldSelector::entire_atmosphere(CanonicalField::PrecipitableWater)
            .with_probability(ProbabilitySelection::above_milli(25_000)),
    ),
    &["PWAT:entire atmosphere", "PWAT:"],
);

const FIELD_HREF_PROB_PWAT_ABOVE_25MM: GribFieldSpec = field_spec(
    "href_probability_precipitable_water_gt_25mm",
    "HREF Probability Precipitable Water > 25 mm",
    ProductFamily::Surface,
    GribLevelKind::EntireAtmosphere,
    None,
    Some(
        FieldSelector::entire_atmosphere(CanonicalField::PrecipitableWater)
            .with_probability(ProbabilitySelection::above_milli(25_000)),
    ),
    &["PWAT:entire atmosphere", "PWAT:"],
);

const FIELD_REFS_PROB_PWAT_ABOVE_37P5MM: GribFieldSpec = field_spec(
    "refs_probability_precipitable_water_gt_37p5mm",
    "REFS Probability Precipitable Water > 37.5 mm",
    ProductFamily::Surface,
    GribLevelKind::EntireAtmosphere,
    None,
    Some(
        FieldSelector::entire_atmosphere(CanonicalField::PrecipitableWater)
            .with_probability(ProbabilitySelection::above_milli(37_500)),
    ),
    &["PWAT:entire atmosphere", "PWAT:"],
);

const FIELD_HREF_PROB_PWAT_ABOVE_37P5MM: GribFieldSpec = field_spec(
    "href_probability_precipitable_water_gt_37p5mm",
    "HREF Probability Precipitable Water > 37.5 mm",
    ProductFamily::Surface,
    GribLevelKind::EntireAtmosphere,
    None,
    Some(
        FieldSelector::entire_atmosphere(CanonicalField::PrecipitableWater)
            .with_probability(ProbabilitySelection::above_milli(37_500)),
    ),
    &["PWAT:entire atmosphere", "PWAT:"],
);

const FIELD_REFS_PROB_PWAT_ABOVE_50MM: GribFieldSpec = field_spec(
    "refs_probability_precipitable_water_gt_50mm",
    "REFS Probability Precipitable Water > 50 mm",
    ProductFamily::Surface,
    GribLevelKind::EntireAtmosphere,
    None,
    Some(
        FieldSelector::entire_atmosphere(CanonicalField::PrecipitableWater)
            .with_probability(ProbabilitySelection::above_milli(50_000)),
    ),
    &["PWAT:entire atmosphere", "PWAT:"],
);

const FIELD_REFS_PROB_QPF_ABOVE_1MM: GribFieldSpec = field_spec(
    "refs_probability_total_qpf_gt_1mm",
    "REFS Probability Total QPF > 1 mm",
    ProductFamily::Surface,
    GribLevelKind::Surface,
    None,
    Some(
        FieldSelector::surface(CanonicalField::TotalPrecipitation)
            .with_probability(ProbabilitySelection::above_milli(1_000)),
    ),
    &["APCP:surface"],
);

const FIELD_REFS_PROB_QPF_ABOVE_2MM: GribFieldSpec = field_spec(
    "refs_probability_total_qpf_gt_2mm",
    "REFS Probability Total QPF > 2 mm",
    ProductFamily::Surface,
    GribLevelKind::Surface,
    None,
    Some(
        FieldSelector::surface(CanonicalField::TotalPrecipitation)
            .with_probability(ProbabilitySelection::above_milli(2_000)),
    ),
    &["APCP:surface"],
);

const FIELD_REFS_PROB_QPF_ABOVE_5MM: GribFieldSpec = field_spec(
    "refs_probability_total_qpf_gt_5mm",
    "REFS Probability Total QPF > 5 mm",
    ProductFamily::Surface,
    GribLevelKind::Surface,
    None,
    Some(
        FieldSelector::surface(CanonicalField::TotalPrecipitation)
            .with_probability(ProbabilitySelection::above_milli(5_000)),
    ),
    &["APCP:surface"],
);

const FIELD_REFS_PROB_QPF_ABOVE_10MM: GribFieldSpec = field_spec(
    "refs_probability_total_qpf_gt_10mm",
    "REFS Probability Total QPF > 10 mm",
    ProductFamily::Surface,
    GribLevelKind::Surface,
    None,
    Some(
        FieldSelector::surface(CanonicalField::TotalPrecipitation)
            .with_probability(ProbabilitySelection::above_milli(10_000)),
    ),
    &["APCP:surface"],
);

const FIELD_REFS_PROB_QPF_ABOVE_12P7MM: GribFieldSpec = field_spec(
    "refs_probability_total_qpf_gt_12p7mm",
    "REFS Probability Total QPF > 12.7 mm",
    ProductFamily::Surface,
    GribLevelKind::Surface,
    None,
    Some(
        FieldSelector::surface(CanonicalField::TotalPrecipitation)
            .with_probability(ProbabilitySelection::above_milli(12_700)),
    ),
    &["APCP:surface"],
);

const FIELD_REFS_PROB_QPF_ABOVE_25MM: GribFieldSpec = field_spec(
    "refs_probability_total_qpf_gt_25mm",
    "REFS Probability Total QPF > 25 mm",
    ProductFamily::Surface,
    GribLevelKind::Surface,
    None,
    Some(
        FieldSelector::surface(CanonicalField::TotalPrecipitation)
            .with_probability(ProbabilitySelection::above_milli(25_000)),
    ),
    &["APCP:surface"],
);

const FIELD_REFS_PROB_QPF_ABOVE_25P4MM: GribFieldSpec = field_spec(
    "refs_probability_total_qpf_gt_25p4mm",
    "REFS Probability Total QPF > 25.4 mm",
    ProductFamily::Surface,
    GribLevelKind::Surface,
    None,
    Some(
        FieldSelector::surface(CanonicalField::TotalPrecipitation)
            .with_probability(ProbabilitySelection::above_milli(25_400)),
    ),
    &["APCP:surface"],
);

const FIELD_REFS_PROB_QPF_ABOVE_50MM: GribFieldSpec = field_spec(
    "refs_probability_total_qpf_gt_50mm",
    "REFS Probability Total QPF > 50 mm",
    ProductFamily::Surface,
    GribLevelKind::Surface,
    None,
    Some(
        FieldSelector::surface(CanonicalField::TotalPrecipitation)
            .with_probability(ProbabilitySelection::above_milli(50_000)),
    ),
    &["APCP:surface"],
);

const FIELD_REFS_PROB_QPF_ABOVE_50P8MM: GribFieldSpec = field_spec(
    "refs_probability_total_qpf_gt_50p8mm",
    "REFS Probability Total QPF > 50.8 mm",
    ProductFamily::Surface,
    GribLevelKind::Surface,
    None,
    Some(
        FieldSelector::surface(CanonicalField::TotalPrecipitation)
            .with_probability(ProbabilitySelection::above_milli(50_800)),
    ),
    &["APCP:surface"],
);

const FIELD_REFS_PROB_QPF_ABOVE_76P2MM: GribFieldSpec = field_spec(
    "refs_probability_total_qpf_gt_76p2mm",
    "REFS Probability Total QPF > 76.2 mm",
    ProductFamily::Surface,
    GribLevelKind::Surface,
    None,
    Some(
        FieldSelector::surface(CanonicalField::TotalPrecipitation)
            .with_probability(ProbabilitySelection::above_milli(76_200)),
    ),
    &["APCP:surface"],
);

const FIELD_REFS_PROB_QPF_ABOVE_100MM: GribFieldSpec = field_spec(
    "refs_probability_total_qpf_gt_100mm",
    "REFS Probability Total QPF > 100 mm",
    ProductFamily::Surface,
    GribLevelKind::Surface,
    None,
    Some(
        FieldSelector::surface(CanonicalField::TotalPrecipitation)
            .with_probability(ProbabilitySelection::above_milli(100_000)),
    ),
    &["APCP:surface"],
);

const FIELD_REFS_PROB_QPF_ABOVE_127MM: GribFieldSpec = field_spec(
    "refs_probability_total_qpf_gt_127mm",
    "REFS Probability Total QPF > 127 mm",
    ProductFamily::Surface,
    GribLevelKind::Surface,
    None,
    Some(
        FieldSelector::surface(CanonicalField::TotalPrecipitation)
            .with_probability(ProbabilitySelection::above_milli(127_000)),
    ),
    &["APCP:surface"],
);

const FIELD_HREF_PROB_PWAT_ABOVE_50MM: GribFieldSpec = field_spec(
    "href_probability_precipitable_water_gt_50mm",
    "HREF Probability Precipitable Water > 50 mm",
    ProductFamily::Surface,
    GribLevelKind::EntireAtmosphere,
    None,
    Some(
        FieldSelector::entire_atmosphere(CanonicalField::PrecipitableWater)
            .with_probability(ProbabilitySelection::above_milli(50_000)),
    ),
    &["PWAT:entire atmosphere", "PWAT:"],
);

const FIELD_REFS_PROB_VISIBILITY_BELOW_1600M: GribFieldSpec = field_spec(
    "refs_probability_visibility_surface_lt_1600m",
    "REFS Probability Visibility < 1600 m",
    ProductFamily::Surface,
    GribLevelKind::Surface,
    None,
    Some(
        FieldSelector::surface(CanonicalField::Visibility)
            .with_probability(ProbabilitySelection::below_milli(1_600_000)),
    ),
    &["VIS:surface"],
);

const FIELD_HREF_PROB_VISIBILITY_BELOW_1600M: GribFieldSpec = field_spec(
    "href_probability_visibility_surface_lt_1600m",
    "HREF Probability Visibility < 1600 m",
    ProductFamily::Surface,
    GribLevelKind::Surface,
    None,
    Some(
        FieldSelector::surface(CanonicalField::Visibility)
            .with_probability(ProbabilitySelection::below_milli(1_600_000)),
    ),
    &["VIS:surface"],
);

const FIELD_REFS_PROB_VISIBILITY_BELOW_3200M: GribFieldSpec = field_spec(
    "refs_probability_visibility_surface_lt_3200m",
    "REFS Probability Visibility < 3200 m",
    ProductFamily::Surface,
    GribLevelKind::Surface,
    None,
    Some(
        FieldSelector::surface(CanonicalField::Visibility)
            .with_probability(ProbabilitySelection::below_milli(3_200_000)),
    ),
    &["VIS:surface"],
);

const FIELD_HREF_PROB_VISIBILITY_BELOW_3200M: GribFieldSpec = field_spec(
    "href_probability_visibility_surface_lt_3200m",
    "HREF Probability Visibility < 3200 m",
    ProductFamily::Surface,
    GribLevelKind::Surface,
    None,
    Some(
        FieldSelector::surface(CanonicalField::Visibility)
            .with_probability(ProbabilitySelection::below_milli(3_200_000)),
    ),
    &["VIS:surface"],
);

const FIELD_REFS_PROB_VISIBILITY_BELOW_8049M: GribFieldSpec = field_spec(
    "refs_probability_visibility_surface_lt_8049m",
    "REFS Probability Visibility < 8049 m",
    ProductFamily::Surface,
    GribLevelKind::Surface,
    None,
    Some(
        FieldSelector::surface(CanonicalField::Visibility)
            .with_probability(ProbabilitySelection::below_milli(8_049_000)),
    ),
    &["VIS:surface"],
);

const FIELD_HREF_PROB_VISIBILITY_BELOW_6400M: GribFieldSpec = field_spec(
    "href_probability_visibility_surface_lt_6400m",
    "HREF Probability Visibility < 6400 m",
    ProductFamily::Surface,
    GribLevelKind::Surface,
    None,
    Some(
        FieldSelector::surface(CanonicalField::Visibility)
            .with_probability(ProbabilitySelection::below_milli(6_400_000)),
    ),
    &["VIS:surface"],
);

const FIELD_REFS_PROB_10M_WIND_SPEED_ABOVE_15P4MS: GribFieldSpec = field_spec(
    "refs_probability_wind_speed_10m_agl_gt_15p4ms",
    "REFS Probability 10m Wind Speed > 15.4 m/s",
    ProductFamily::Surface,
    GribLevelKind::HeightAboveGround,
    Some(10),
    Some(
        FieldSelector::height_agl(CanonicalField::WindSpeed, 10)
            .with_probability(ProbabilitySelection::above_milli(15_400)),
    ),
    &["WIND:10 m above ground"],
);

const FIELD_HREF_PROB_10M_WIND_SPEED_ABOVE_15P4MS: GribFieldSpec = field_spec(
    "href_probability_wind_speed_10m_agl_gt_15p4ms",
    "HREF Probability 10m Wind Speed > 15.4 m/s",
    ProductFamily::Surface,
    GribLevelKind::HeightAboveGround,
    Some(10),
    Some(
        FieldSelector::height_agl(CanonicalField::WindSpeed, 10)
            .with_probability(ProbabilitySelection::above_milli(15_400)),
    ),
    &["WIND:10 m above ground"],
);

const FIELD_REFS_PROB_10M_WIND_SPEED_ABOVE_20P6MS: GribFieldSpec = field_spec(
    "refs_probability_wind_speed_10m_agl_gt_20p6ms",
    "REFS Probability 10m Wind Speed > 20.6 m/s",
    ProductFamily::Surface,
    GribLevelKind::HeightAboveGround,
    Some(10),
    Some(
        FieldSelector::height_agl(CanonicalField::WindSpeed, 10)
            .with_probability(ProbabilitySelection::above_milli(20_600)),
    ),
    &["WIND:10 m above ground"],
);

const FIELD_HREF_PROB_10M_WIND_SPEED_ABOVE_20P6MS: GribFieldSpec = field_spec(
    "href_probability_wind_speed_10m_agl_gt_20p6ms",
    "HREF Probability 10m Wind Speed > 20.6 m/s",
    ProductFamily::Surface,
    GribLevelKind::HeightAboveGround,
    Some(10),
    Some(
        FieldSelector::height_agl(CanonicalField::WindSpeed, 10)
            .with_probability(ProbabilitySelection::above_milli(20_600)),
    ),
    &["WIND:10 m above ground"],
);

const FIELD_REFS_PROB_10M_WIND_SPEED_ABOVE_25P72MS: GribFieldSpec = field_spec(
    "refs_probability_wind_speed_10m_agl_gt_25p72ms",
    "REFS Probability 10m Wind Speed > 25.72 m/s",
    ProductFamily::Surface,
    GribLevelKind::HeightAboveGround,
    Some(10),
    Some(
        FieldSelector::height_agl(CanonicalField::WindSpeed, 10)
            .with_probability(ProbabilitySelection::above_milli(25_720)),
    ),
    &["WIND:10 m above ground"],
);

const FIELD_HREF_PROB_10M_WIND_SPEED_ABOVE_25P72MS: GribFieldSpec = field_spec(
    "href_probability_wind_speed_10m_agl_gt_25p72ms",
    "HREF Probability 10m Wind Speed > 25.72 m/s",
    ProductFamily::Surface,
    GribLevelKind::HeightAboveGround,
    Some(10),
    Some(
        FieldSelector::height_agl(CanonicalField::WindSpeed, 10)
            .with_probability(ProbabilitySelection::above_milli(25_720)),
    ),
    &["WIND:10 m above ground"],
);

const FIELD_REFS_PROB_UH_ABOVE_25: GribFieldSpec = field_spec(
    "refs_probability_updraft_helicity_2to5km_gt_25",
    "REFS Probability 2-5 km Updraft Helicity > 25",
    ProductFamily::Native,
    GribLevelKind::HeightAboveGroundLayer,
    None,
    Some(
        FieldSelector::height_layer_agl(CanonicalField::UpdraftHelicity, 2000, 5000)
            .with_probability(ProbabilitySelection::above_milli(25_000)),
    ),
    &["MXUPHL:5000-2000"],
);

const FIELD_HREF_PROB_UH_ABOVE_25: GribFieldSpec = field_spec(
    "href_probability_updraft_helicity_2to5km_gt_25",
    "HREF Probability 2-5 km Updraft Helicity > 25",
    ProductFamily::Native,
    GribLevelKind::HeightAboveGroundLayer,
    None,
    Some(
        FieldSelector::height_layer_agl(CanonicalField::UpdraftHelicity, 2000, 5000)
            .with_probability(ProbabilitySelection::above_milli(25_000)),
    ),
    &["MXUPHL:5000-2000"],
);

const FIELD_REFS_PROB_UH_ABOVE_75: GribFieldSpec = field_spec(
    "refs_probability_updraft_helicity_2to5km_gt_75",
    "REFS Probability 2-5 km Updraft Helicity > 75",
    ProductFamily::Native,
    GribLevelKind::HeightAboveGroundLayer,
    None,
    Some(
        FieldSelector::height_layer_agl(CanonicalField::UpdraftHelicity, 2000, 5000)
            .with_probability(ProbabilitySelection::above_milli(75_000)),
    ),
    &["MXUPHL:5000-2000"],
);

const FIELD_HREF_PROB_UH_ABOVE_75: GribFieldSpec = field_spec(
    "href_probability_updraft_helicity_2to5km_gt_75",
    "HREF Probability 2-5 km Updraft Helicity > 75",
    ProductFamily::Native,
    GribLevelKind::HeightAboveGroundLayer,
    None,
    Some(
        FieldSelector::height_layer_agl(CanonicalField::UpdraftHelicity, 2000, 5000)
            .with_probability(ProbabilitySelection::above_milli(75_000)),
    ),
    &["MXUPHL:5000-2000"],
);

const FIELD_REFS_PROB_UH_ABOVE_150: GribFieldSpec = field_spec(
    "refs_probability_updraft_helicity_2to5km_gt_150",
    "REFS Probability 2-5 km Updraft Helicity > 150",
    ProductFamily::Native,
    GribLevelKind::HeightAboveGroundLayer,
    None,
    Some(
        FieldSelector::height_layer_agl(CanonicalField::UpdraftHelicity, 2000, 5000)
            .with_probability(ProbabilitySelection::above_milli(150_000)),
    ),
    &["MXUPHL:5000-2000"],
);

const FIELD_HREF_PROB_UH_ABOVE_150: GribFieldSpec = field_spec(
    "href_probability_updraft_helicity_2to5km_gt_150",
    "HREF Probability 2-5 km Updraft Helicity > 150",
    ProductFamily::Native,
    GribLevelKind::HeightAboveGroundLayer,
    None,
    Some(
        FieldSelector::height_layer_agl(CanonicalField::UpdraftHelicity, 2000, 5000)
            .with_probability(ProbabilitySelection::above_milli(150_000)),
    ),
    &["MXUPHL:5000-2000"],
);

/// Surface elevation.  The WRF import already publishes this plane
/// (`HGT` -> the store's `orography`, metres) under the surface
/// geopotential-height selector, so the product needs no ingest work --
/// only a recipe that asks for it.
const FIELD_TERRAIN_HEIGHT: GribFieldSpec = field_spec(
    "terrain_height",
    "Terrain Height",
    ProductFamily::Surface,
    GribLevelKind::Surface,
    None,
    Some(FieldSelector::surface(
        CanonicalField::GeopotentialHeight,
    )),
    &["HGT:surface"],
);

const FIELD_MSLP: GribFieldSpec = field_spec(
    "pressure_reduced_to_mean_sea_level",
    "MSLP",
    ProductFamily::Surface,
    GribLevelKind::MeanSeaLevel,
    None,
    Some(FieldSelector::mean_sea_level(
        CanonicalField::PressureReducedToMeanSeaLevel,
    )),
    &[
        "PRMSL:mean sea level",
        "MSLMA:mean sea level",
        "MSLET:mean sea level",
    ],
);

const FIELD_GEFS_AVG_MSLP: GribFieldSpec = field_spec(
    "gefs_mean_pressure_reduced_to_mean_sea_level",
    "GEFS MSLP Mean",
    ProductFamily::Surface,
    GribLevelKind::MeanSeaLevel,
    None,
    Some(
        FieldSelector::mean_sea_level(CanonicalField::PressureReducedToMeanSeaLevel)
            .with_ensemble_mean(),
    ),
    &["PRMSL:mean sea level"],
);

const FIELD_GEFS_SPR_MSLP_STDDEV: GribFieldSpec = field_spec(
    "gefs_spread_pressure_reduced_to_mean_sea_level_stddev",
    "GEFS MSLP Spread",
    ProductFamily::Surface,
    GribLevelKind::MeanSeaLevel,
    None,
    Some(
        FieldSelector::mean_sea_level(CanonicalField::PressureReducedToMeanSeaLevel)
            .with_ensemble_standard_deviation(),
    ),
    &["PRMSL:mean sea level"],
);

const FIELD_AIGEFS_SPR_MSLP_STDDEV: GribFieldSpec = field_spec(
    "aigefs_spread_pressure_reduced_to_mean_sea_level_stddev",
    "AI-GEFS MSLP Spread",
    ProductFamily::Surface,
    GribLevelKind::MeanSeaLevel,
    None,
    Some(
        FieldSelector::mean_sea_level(CanonicalField::PressureReducedToMeanSeaLevel)
            .with_ensemble_standard_deviation(),
    ),
    &["PRMSL:mean sea level"],
);

const FIELD_HGEFS_SPR_MSLP_STDDEV: GribFieldSpec = field_spec(
    "hgefs_spread_pressure_reduced_to_mean_sea_level_stddev",
    "HGEFS MSLP Spread",
    ProductFamily::Surface,
    GribLevelKind::MeanSeaLevel,
    None,
    Some(
        FieldSelector::mean_sea_level(CanonicalField::PressureReducedToMeanSeaLevel)
            .with_ensemble_standard_deviation(),
    ),
    &["PRMSL:mean sea level"],
);

const FIELD_HREF_SPRD_MSLP: GribFieldSpec = field_spec(
    "href_spread_pressure_reduced_to_mean_sea_level",
    "HREF MSLP Spread",
    ProductFamily::Surface,
    GribLevelKind::MeanSeaLevel,
    None,
    Some(
        FieldSelector::mean_sea_level(CanonicalField::PressureReducedToMeanSeaLevel)
            .with_product(FieldProduct::EnsembleSpread),
    ),
    &["MSLET:mean sea level"],
);

const FIELD_HREF_MEAN_MSLP: GribFieldSpec = field_spec(
    "href_mean_pressure_reduced_to_mean_sea_level",
    "HREF MSLP Mean",
    ProductFamily::Surface,
    GribLevelKind::MeanSeaLevel,
    None,
    Some(
        FieldSelector::mean_sea_level(CanonicalField::PressureReducedToMeanSeaLevel)
            .with_ensemble_mean(),
    ),
    &["MSLET:mean sea level"],
);

const FIELD_REFS_SPRD_MSLP: GribFieldSpec = field_spec(
    "refs_spread_pressure_reduced_to_mean_sea_level",
    "REFS MSLP Spread",
    ProductFamily::Surface,
    GribLevelKind::MeanSeaLevel,
    None,
    Some(
        FieldSelector::mean_sea_level(CanonicalField::PressureReducedToMeanSeaLevel)
            .with_product(FieldProduct::EnsembleSpread),
    ),
    &["MSLET:mean sea level", "PRMSL:mean sea level"],
);

const FIELD_IVT: GribFieldSpec = field_spec(
    "integrated_vapor_transport", "Integrated Vapor Transport",
    ProductFamily::Native, GribLevelKind::EntireAtmosphere, None,
    Some(FieldSelector::entire_atmosphere(CanonicalField::IntegratedVaporTransport)), &[],
);
const FIELD_IVT_U: GribFieldSpec = field_spec(
    "integrated_vapor_transport_u", "Eastward Integrated Vapor Transport",
    ProductFamily::Native, GribLevelKind::EntireAtmosphere, None,
    Some(FieldSelector::entire_atmosphere(CanonicalField::IntegratedVaporTransportU)), &[],
);
const FIELD_IVT_V: GribFieldSpec = field_spec(
    "integrated_vapor_transport_v", "Northward Integrated Vapor Transport",
    ProductFamily::Native, GribLevelKind::EntireAtmosphere, None,
    Some(FieldSelector::entire_atmosphere(CanonicalField::IntegratedVaporTransportV)), &[],
);
const FIELD_925_HEIGHT: GribFieldSpec = field_spec(
    "height_925mb", "925 hPa HEIGHT", ProductFamily::Pressure,
    GribLevelKind::IsobaricHpa, Some(925),
    Some(FieldSelector::isobaric(CanonicalField::GeopotentialHeight, 925)), &[],
);
const FIELD_925_U: GribFieldSpec = field_spec(
    "u_925mb", "925 hPa U", ProductFamily::Pressure,
    GribLevelKind::IsobaricHpa, Some(925),
    Some(FieldSelector::isobaric(CanonicalField::UWind, 925)), &[],
);
const FIELD_925_V: GribFieldSpec = field_spec(
    "v_925mb", "925 hPa V", ProductFamily::Pressure,
    GribLevelKind::IsobaricHpa, Some(925),
    Some(FieldSelector::isobaric(CanonicalField::VWind, 925)), &[],
);

const FIELD_PWAT: GribFieldSpec = field_spec(
    "precipitable_water",
    "Precipitable Water",
    ProductFamily::Surface,
    GribLevelKind::EntireAtmosphere,
    None,
    Some(FieldSelector::entire_atmosphere(
        CanonicalField::PrecipitableWater,
    )),
    &["PWAT:entire atmosphere", "PWAT:"],
);

const FIELD_GEFS_AVG_PWAT: GribFieldSpec = field_spec(
    "gefs_mean_precipitable_water",
    "GEFS Precipitable Water Mean",
    ProductFamily::Surface,
    GribLevelKind::EntireAtmosphere,
    None,
    Some(FieldSelector::entire_atmosphere(CanonicalField::PrecipitableWater).with_ensemble_mean()),
    &["PWAT:entire atmosphere", "PWAT:"],
);

const FIELD_GEFS_SPR_PWAT_STDDEV: GribFieldSpec = field_spec(
    "gefs_spread_precipitable_water_stddev",
    "GEFS Precipitable Water Spread",
    ProductFamily::Surface,
    GribLevelKind::EntireAtmosphere,
    None,
    Some(
        FieldSelector::entire_atmosphere(CanonicalField::PrecipitableWater)
            .with_ensemble_standard_deviation(),
    ),
    &["PWAT:entire atmosphere", "PWAT:"],
);

const FIELD_HREF_SPRD_PWAT: GribFieldSpec = field_spec(
    "href_spread_precipitable_water",
    "HREF Precipitable Water Spread",
    ProductFamily::Surface,
    GribLevelKind::EntireAtmosphere,
    None,
    Some(
        FieldSelector::entire_atmosphere(CanonicalField::PrecipitableWater)
            .with_product(FieldProduct::EnsembleSpread),
    ),
    &["PWAT:entire atmosphere", "PWAT:"],
);

const FIELD_HREF_MEAN_PWAT: GribFieldSpec = field_spec(
    "href_mean_precipitable_water",
    "HREF Precipitable Water Mean",
    ProductFamily::Surface,
    GribLevelKind::EntireAtmosphere,
    None,
    Some(FieldSelector::entire_atmosphere(CanonicalField::PrecipitableWater).with_ensemble_mean()),
    &["PWAT:entire atmosphere", "PWAT:"],
);

const FIELD_REFS_SPRD_PWAT: GribFieldSpec = field_spec(
    "refs_spread_precipitable_water",
    "REFS Precipitable Water Spread",
    ProductFamily::Surface,
    GribLevelKind::EntireAtmosphere,
    None,
    Some(
        FieldSelector::entire_atmosphere(CanonicalField::PrecipitableWater)
            .with_product(FieldProduct::EnsembleSpread),
    ),
    &["PWAT:entire atmosphere", "PWAT:"],
);

const FIELD_TOTAL_CLOUD_COVER: GribFieldSpec = field_spec(
    "total_cloud_cover",
    "Total Cloud Cover",
    ProductFamily::Surface,
    GribLevelKind::EntireAtmosphere,
    None,
    Some(FieldSelector::entire_atmosphere(
        CanonicalField::TotalCloudCover,
    )),
    &["TCDC:entire atmosphere", "TCDC:"],
);

const FIELD_GEFS_AVG_TOTAL_CLOUD_COVER: GribFieldSpec = field_spec(
    "gefs_mean_total_cloud_cover",
    "GEFS Total Cloud Cover Mean",
    ProductFamily::Surface,
    GribLevelKind::EntireAtmosphere,
    None,
    Some(FieldSelector::entire_atmosphere(CanonicalField::TotalCloudCover).with_ensemble_mean()),
    &["TCDC:entire atmosphere", "TCDC:"],
);

const FIELD_HREF_MEAN_TOTAL_CLOUD_COVER: GribFieldSpec = field_spec(
    "href_mean_total_cloud_cover",
    "HREF Total Cloud Cover Mean",
    ProductFamily::Surface,
    GribLevelKind::EntireAtmosphere,
    None,
    Some(FieldSelector::entire_atmosphere(CanonicalField::TotalCloudCover).with_ensemble_mean()),
    &["TCDC:entire atmosphere", "TCDC:"],
);

const FIELD_GEFS_SPR_TOTAL_CLOUD_COVER_STDDEV: GribFieldSpec = field_spec(
    "gefs_spread_total_cloud_cover_stddev",
    "GEFS Total Cloud Cover Spread",
    ProductFamily::Surface,
    GribLevelKind::EntireAtmosphere,
    None,
    Some(
        FieldSelector::entire_atmosphere(CanonicalField::TotalCloudCover)
            .with_ensemble_standard_deviation(),
    ),
    &["TCDC:entire atmosphere", "TCDC:"],
);

const FIELD_REFS_SPRD_TOTAL_CLOUD_COVER: GribFieldSpec = field_spec(
    "refs_spread_total_cloud_cover",
    "REFS Total Cloud Cover Spread",
    ProductFamily::Surface,
    GribLevelKind::EntireAtmosphere,
    None,
    Some(
        FieldSelector::entire_atmosphere(CanonicalField::TotalCloudCover)
            .with_product(FieldProduct::EnsembleSpread),
    ),
    &["TCDC:entire atmosphere", "TCDC:"],
);

const FIELD_LOW_CLOUD_COVER: GribFieldSpec = field_spec(
    "low_cloud_cover",
    "Low Cloud Cover",
    ProductFamily::Surface,
    GribLevelKind::EntireAtmosphere,
    None,
    Some(FieldSelector::entire_atmosphere(
        CanonicalField::LowCloudCover,
    )),
    &["LCDC:low cloud layer", "LCDC:"],
);

const FIELD_MIDDLE_CLOUD_COVER: GribFieldSpec = field_spec(
    "middle_cloud_cover",
    "Middle Cloud Cover",
    ProductFamily::Surface,
    GribLevelKind::EntireAtmosphere,
    None,
    Some(FieldSelector::entire_atmosphere(
        CanonicalField::MiddleCloudCover,
    )),
    &["MCDC:middle cloud layer", "MCDC:"],
);

const FIELD_HIGH_CLOUD_COVER: GribFieldSpec = field_spec(
    "high_cloud_cover",
    "High Cloud Cover",
    ProductFamily::Surface,
    GribLevelKind::EntireAtmosphere,
    None,
    Some(FieldSelector::entire_atmosphere(
        CanonicalField::HighCloudCover,
    )),
    &["HCDC:high cloud layer", "HCDC:"],
);

const FIELD_TOTAL_QPF: GribFieldSpec = field_spec(
    "total_qpf",
    "Total QPF",
    ProductFamily::Surface,
    GribLevelKind::Surface,
    None,
    Some(FieldSelector::surface(CanonicalField::TotalPrecipitation)),
    &["APCP:surface"],
);

const FIELD_GEFS_AVG_6H_QPF: GribFieldSpec = field_spec(
    "gefs_mean_6h_qpf",
    "GEFS 6h QPF Mean",
    ProductFamily::Surface,
    GribLevelKind::Surface,
    None,
    Some(FieldSelector::surface(CanonicalField::TotalPrecipitation).with_ensemble_mean()),
    &["APCP:surface"],
);

const FIELD_GEFS_SPR_6H_QPF_STDDEV: GribFieldSpec = field_spec(
    "gefs_spread_6h_qpf_stddev",
    "GEFS 6h QPF Spread",
    ProductFamily::Surface,
    GribLevelKind::Surface,
    None,
    Some(
        FieldSelector::surface(CanonicalField::TotalPrecipitation)
            .with_ensemble_standard_deviation(),
    ),
    &["APCP:surface"],
);

const FIELD_AIGEFS_SPR_6H_QPF_STDDEV: GribFieldSpec = field_spec(
    "aigefs_spread_6h_qpf_stddev",
    "AI-GEFS 6h QPF Spread",
    ProductFamily::Surface,
    GribLevelKind::Surface,
    None,
    Some(
        FieldSelector::surface(CanonicalField::TotalPrecipitation)
            .with_ensemble_standard_deviation(),
    ),
    &["APCP:surface"],
);

const FIELD_HGEFS_SPR_6H_QPF_STDDEV: GribFieldSpec = field_spec(
    "hgefs_spread_6h_qpf_stddev",
    "HGEFS 6h QPF Spread",
    ProductFamily::Surface,
    GribLevelKind::Surface,
    None,
    Some(
        FieldSelector::surface(CanonicalField::TotalPrecipitation)
            .with_ensemble_standard_deviation(),
    ),
    &["APCP:surface"],
);

const FIELD_POP: GribFieldSpec = field_spec(
    "probability_of_precipitation",
    "Probability of Precipitation",
    ProductFamily::Surface,
    GribLevelKind::Surface,
    None,
    Some(FieldSelector::surface(
        CanonicalField::ProbabilityOfPrecipitation,
    )),
    &["APCP:surface"],
);

const FIELD_CATEGORICAL_RAIN: GribFieldSpec = field_spec(
    "categorical_rain",
    "Categorical Rain",
    ProductFamily::Surface,
    GribLevelKind::Surface,
    None,
    Some(FieldSelector::surface(CanonicalField::CategoricalRain)),
    &["CRAIN:surface"],
);

const FIELD_CATEGORICAL_FREEZING_RAIN: GribFieldSpec = field_spec(
    "categorical_freezing_rain",
    "Categorical Freezing Rain",
    ProductFamily::Surface,
    GribLevelKind::Surface,
    None,
    Some(FieldSelector::surface(
        CanonicalField::CategoricalFreezingRain,
    )),
    &["CFRZR:surface", "FRZR:surface"],
);

const FIELD_CATEGORICAL_ICE_PELLETS: GribFieldSpec = field_spec(
    "categorical_ice_pellets",
    "Categorical Ice Pellets",
    ProductFamily::Surface,
    GribLevelKind::Surface,
    None,
    Some(FieldSelector::surface(
        CanonicalField::CategoricalIcePellets,
    )),
    &["CICEP:surface"],
);

const FIELD_CATEGORICAL_SNOW: GribFieldSpec = field_spec(
    "categorical_snow",
    "Categorical Snow",
    ProductFamily::Surface,
    GribLevelKind::Surface,
    None,
    Some(FieldSelector::surface(CanonicalField::CategoricalSnow)),
    &["CSNOW:surface"],
);

const FIELD_FOG_FLAG: GribFieldSpec = field_spec(
    "fog_flag_surface", "Low Visibility (< 1000 m)", ProductFamily::Surface,
    GribLevelKind::Surface, None, Some(FieldSelector::surface(CanonicalField::FogFlag)), &[],
);

const FIELD_VISIBILITY: GribFieldSpec = field_spec(
    "visibility_surface",
    "Visibility",
    ProductFamily::Surface,
    GribLevelKind::Surface,
    None,
    Some(FieldSelector::surface(CanonicalField::Visibility)),
    &["VIS:surface"],
);

const FIELD_HREF_MEAN_VISIBILITY: GribFieldSpec = field_spec(
    "href_mean_visibility_surface",
    "HREF Visibility Mean",
    ProductFamily::Surface,
    GribLevelKind::Surface,
    None,
    Some(FieldSelector::surface(CanonicalField::Visibility).with_ensemble_mean()),
    &["VIS:surface"],
);

const FIELD_REFS_SPRD_VISIBILITY: GribFieldSpec = field_spec(
    "refs_spread_visibility_surface",
    "REFS Visibility Spread",
    ProductFamily::Surface,
    GribLevelKind::Surface,
    None,
    Some(
        FieldSelector::surface(CanonicalField::Visibility)
            .with_product(FieldProduct::EnsembleSpread),
    ),
    &["VIS:surface"],
);

const FIELD_SIMULATED_IR: GribFieldSpec = field_spec(
    "simulated_infrared_brightness_temperature",
    "Simulated IR Satellite",
    ProductFamily::Native,
    GribLevelKind::NominalTop,
    None,
    Some(FieldSelector::nominal_top(
        CanonicalField::SimulatedInfraredBrightnessTemperature,
    )),
    &["SBT113:top of atmosphere"],
);

const FIELD_ISOTHERM_HEIGHT_0C: GribFieldSpec = field_spec(
    "isotherm_height_0c",
    "0 C Isotherm Height",
    ProductFamily::Native,
    GribLevelKind::Isotherm,
    Some(0),
    Some(FieldSelector::isotherm_celsius(CanonicalField::GeopotentialHeight, 0)),
    &["HGT:0C isotherm"],
);

const FIELD_ISOTHERM_HEIGHT_MINUS10C: GribFieldSpec = field_spec(
    "isotherm_height_minus10c",
    "-10 C Isotherm Height",
    ProductFamily::Native,
    GribLevelKind::Isotherm,
    Some(-10),
    Some(FieldSelector::isotherm_celsius(CanonicalField::GeopotentialHeight, -10)),
    &[],
);

const FIELD_ISOTHERM_HEIGHT_MINUS20C: GribFieldSpec = field_spec(
    "isotherm_height_minus20c",
    "-20 C Isotherm Height",
    ProductFamily::Native,
    GribLevelKind::Isotherm,
    Some(-20),
    Some(FieldSelector::isotherm_celsius(CanonicalField::GeopotentialHeight, -20)),
    &[],
);

const FIELD_ICE_WATER_PATH: GribFieldSpec = field_spec(
    "ice_water_path", "Ice Water Path", ProductFamily::Native,
    GribLevelKind::EntireAtmosphere, None,
    Some(FieldSelector::entire_atmosphere(CanonicalField::IceWaterPath)), &[],
);

const FIELD_SUPERCOOLED_WATER_PATH: GribFieldSpec = field_spec(
    "supercooled_water_path",
    "Supercooled Liquid Water Path",
    ProductFamily::Native,
    GribLevelKind::EntireAtmosphere,
    None,
    Some(FieldSelector::entire_atmosphere(CanonicalField::SupercooledLiquidWaterPath)),
    &[],
);

const FIELD_SUPERCOOLED_WATER_PATH_0_3KM: GribFieldSpec = field_spec(
    "supercooled_water_path_0_3km",
    "Supercooled Liquid Water Path, 0 to 3 km AGL",
    ProductFamily::Native,
    GribLevelKind::HeightAboveGroundLayer,
    None,
    Some(FieldSelector::height_layer_agl(CanonicalField::SupercooledLiquidWaterPath, 0, 3000)),
    &[],
);

const FIELD_SUPERCOOLED_WATER_PATH_3_6KM: GribFieldSpec = field_spec(
    "supercooled_water_path_3_6km",
    "Supercooled Liquid Water Path, 3 to 6 km AGL",
    ProductFamily::Native,
    GribLevelKind::HeightAboveGroundLayer,
    None,
    Some(FieldSelector::height_layer_agl(CanonicalField::SupercooledLiquidWaterPath, 3000, 6000)),
    &[],
);

const FIELD_CLOUD_FRACTION_COLUMN_MAX: GribFieldSpec = field_spec(
    "cloud_fraction_column_max",
    "Radiation Cloud Fraction, Column Maximum",
    ProductFamily::Native,
    GribLevelKind::ColumnMaximum,
    None,
    Some(FieldSelector::column_maximum(CanonicalField::TotalCloudCover)),
    &[],
);

const FIELD_CLOUD_WATER_COLUMN_MAX: GribFieldSpec = field_spec(
    "cloud_water_column_max",
    "Cloud Water Mixing Ratio, Column Maximum",
    ProductFamily::Native,
    GribLevelKind::ColumnMaximum,
    None,
    Some(FieldSelector::column_maximum(CanonicalField::CloudWaterMixingRatio)),
    &[],
);

const FIELD_RAIN_WATER_COLUMN_MAX: GribFieldSpec = field_spec(
    "rain_water_column_max",
    "Rain Water Mixing Ratio, Column Maximum",
    ProductFamily::Native,
    GribLevelKind::ColumnMaximum,
    None,
    Some(FieldSelector::column_maximum(CanonicalField::RainWaterMixingRatio)),
    &[],
);

const FIELD_CLOUD_ICE_COLUMN_MAX: GribFieldSpec = field_spec(
    "cloud_ice_column_max",
    "Cloud Ice Mixing Ratio, Column Maximum",
    ProductFamily::Native,
    GribLevelKind::ColumnMaximum,
    None,
    Some(FieldSelector::column_maximum(CanonicalField::CloudIceMixingRatio)),
    &[],
);

const FIELD_SNOW_COLUMN_MAX: GribFieldSpec = field_spec(
    "snow_column_max",
    "Snow Mixing Ratio, Column Maximum",
    ProductFamily::Native,
    GribLevelKind::ColumnMaximum,
    None,
    Some(FieldSelector::column_maximum(CanonicalField::SnowMixingRatio)),
    &[],
);

const FIELD_MODEL_SNOWFALL: GribFieldSpec = field_spec(
    "snowfall_window", "Model snowfall depth, window total", ProductFamily::Surface,
    GribLevelKind::Surface, None, Some(FieldSelector::surface(CanonicalField::ModelSnowfall)), &[],
);
const FIELD_SNOW_10TO1: GribFieldSpec = field_spec(
    "snow_10to1_window", "10:1 snowfall depth, window total", ProductFamily::Surface,
    GribLevelKind::Surface, None, Some(FieldSelector::surface(CanonicalField::Snowfall10to1)), &[],
);
const FIELD_SNOW_KUCHERA: GribFieldSpec = field_spec(
    "snow_kuchera_window", "Kuchera snowfall depth, window total", ProductFamily::Surface,
    GribLevelKind::Surface, None, Some(FieldSelector::surface(CanonicalField::SnowfallKuchera)), &[],
);
const FIELD_SNOWFALL_LIQUID_EQUIVALENT: GribFieldSpec = field_spec("snowfall_liquid_equivalent", "Snowfall liquid equivalent, window total", ProductFamily::Surface, GribLevelKind::Surface, None, Some(FieldSelector::surface(CanonicalField::SnowfallLiquidEquivalent)), &[]);

const FIELD_SNOW_DEPTH_TOTAL: GribFieldSpec = field_spec("snow_depth_total", "LSM snow depth", ProductFamily::Surface, GribLevelKind::Surface, None, Some(FieldSelector::surface(CanonicalField::SnowDepth)), &[]);

const FIELD_SNOW_DEPTH_CHANGE: GribFieldSpec = field_spec("snow_depth_change", "LSM snow depth change", ProductFamily::Surface, GribLevelKind::Surface, None, Some(FieldSelector::surface(CanonicalField::SnowDepthChange)), &[]);


const FIELD_HAILMAXSURFACE: GribFieldSpec = field_spec(
    "hail_max_surface", "PSD maximum hail diameter at lowest model level", ProductFamily::Surface,
    GribLevelKind::Surface, None, Some(FieldSelector::surface(CanonicalField::HailMaxSurface)), &[],
);
const FIELD_HAILMAXCOLUMN: GribFieldSpec = field_spec(
    "hail_max_column", "PSD maximum hail diameter over column", ProductFamily::Surface,
    GribLevelKind::Surface, None, Some(FieldSelector::surface(CanonicalField::HailMaxColumn)), &[],
);
const FIELD_HAILMAXSURFACERUN: GribFieldSpec = field_spec(
    "hail_max_surface_run", "Run maximum PSD surface hail diameter", ProductFamily::Surface,
    GribLevelKind::Surface, None, Some(FieldSelector::surface(CanonicalField::HailMaxSurfaceRun)), &[],
);
const FIELD_HAILMAXCOLUMNRUN: GribFieldSpec = field_spec(
    "hail_max_column_run", "Run maximum PSD column hail diameter", ProductFamily::Surface,
    GribLevelKind::Surface, None, Some(FieldSelector::surface(CanonicalField::HailMaxColumnRun)), &[],
);
const FIELD_HAILCASTMAX: GribFieldSpec = field_spec(
    "hail_max_hailcast", "HAILCAST maximum surviving ice diameter", ProductFamily::Surface,
    GribLevelKind::Surface, None, Some(FieldSelector::surface(CanonicalField::HailcastMax)), &[],
);
const FIELD_HAILCASTMEAN: GribFieldSpec = field_spec(
    "hail_max_hailcast_mean", "HAILCAST mean surviving ice diameter", ProductFamily::Surface,
    GribLevelKind::Surface, None, Some(FieldSelector::surface(CanonicalField::HailcastMean)), &[],
);
const FIELD_HAILCASTSTD: GribFieldSpec = field_spec(
    "hail_max_hailcast_std", "HAILCAST diameter standard deviation", ProductFamily::Surface,
    GribLevelKind::Surface, None, Some(FieldSelector::surface(CanonicalField::HailcastStd)), &[],
);
const FIELD_HAILCASTRUN: GribFieldSpec = field_spec(
    "hail_max_hailcast_run", "Run maximum HAILCAST diameter", ProductFamily::Surface,
    GribLevelKind::Surface, None, Some(FieldSelector::surface(CanonicalField::HailcastRun)), &[],
);

const FIELD_GRAUPEL_COLUMN_MAX: GribFieldSpec = field_spec(
    "graupel_column_max",
    "Graupel Mixing Ratio, Column Maximum",
    ProductFamily::Native,
    GribLevelKind::ColumnMaximum,
    None,
    Some(FieldSelector::column_maximum(CanonicalField::GraupelMixingRatio)),
    &[],
);

const FIELD_2M_THETA_E: GribFieldSpec = field_spec(
    "theta_e_2m_agl",
    "2m AGL Theta-e",
    ProductFamily::Surface,
    GribLevelKind::HeightAboveGround,
    Some(2),
    None,
    &[],
);

const FIELD_2M_HEAT_INDEX: GribFieldSpec = field_spec(
    "heat_index_2m_agl",
    "2m AGL Heat Index",
    ProductFamily::Surface,
    GribLevelKind::HeightAboveGround,
    Some(2),
    None,
    &[],
);

const FIELD_2M_WIND_CHILL: GribFieldSpec = field_spec(
    "wind_chill_2m_agl",
    "2m AGL Wind Chill",
    ProductFamily::Surface,
    GribLevelKind::HeightAboveGround,
    Some(2),
    None,
    &[],
);

const FIELD_LIGHTNING_FLASH_DENSITY: GribFieldSpec = field_spec(
    "lightning_flash_density",
    "Lightning Flash Density",
    ProductFamily::Surface,
    GribLevelKind::HeightAboveGround,
    Some(1),
    // A canonical selector, like every other surface spec here.  Passing
    // `None` meant the slug could not resolve from ANY store: the
    // requirement came back as "no canonical store selector exists for
    // this field", so no import on any lane could ever satisfy it, and
    // the catalog carried a product nothing could serve.  The canonical
    // field already exists with its units and label
    // (rustwx-core: LightningFlashDensity, km^-2 day^-1).
    Some(FieldSelector::height_agl(
        CanonicalField::LightningFlashDensity,
        1,
    )),
    &[
        "LTNGSD:1 m above ground",
        "LTNGSD:2 m above ground",
        "LTNG:entire atmosphere",
    ],
);

const FIELD_CLOUD_COVER_LEVELS: GribFieldSpec = field_spec(
    "cloud_cover_levels",
    "Cloud Cover Levels",
    ProductFamily::Surface,
    GribLevelKind::EntireAtmosphere,
    None,
    None,
    &[],
);

const FIELD_ONE_HOUR_QPF: GribFieldSpec = field_spec(
    "one_hour_qpf",
    "1h QPF",
    ProductFamily::Surface,
    GribLevelKind::Surface,
    None,
    None,
    &["APCP:surface"],
);

const FIELD_PRECIPITATION_TYPE: GribFieldSpec = field_spec(
    "precipitation_type",
    "Precipitation Type",
    ProductFamily::Surface,
    GribLevelKind::Surface,
    None,
    None,
    &[
        "CRAIN:surface",
        "CFRZR:surface",
        "CICEP:surface",
        "CSNOW:surface",
    ],
);

const FIELD_1KM_VERTICAL_VELOCITY: GribFieldSpec = field_spec(
    "vertical_velocity_1km_agl",
    "1 km AGL Vertical Velocity (upward positive)",
    ProductFamily::Native,
    GribLevelKind::HeightAboveGround,
    Some(1000),
    Some(FieldSelector::height_agl(CanonicalField::GeometricVerticalVelocity, 1000)),
    &["DZDT:1000 m above ground"],
);

const FIELD_3KM_VERTICAL_VELOCITY: GribFieldSpec = field_spec(
    "vertical_velocity_3km_agl",
    "3 km AGL Vertical Velocity (upward positive)",
    ProductFamily::Native,
    GribLevelKind::HeightAboveGround,
    Some(3000),
    Some(FieldSelector::height_agl(CanonicalField::GeometricVerticalVelocity, 3000)),
    &["DZDT:3000 m above ground"],
);

const FIELD_1KM_REFLECTIVITY: GribFieldSpec = field_spec(
    "radar_reflectivity_1km_agl",
    "1km AGL Reflectivity",
    ProductFamily::Native,
    GribLevelKind::HeightAboveGround,
    Some(1000),
    Some(FieldSelector::height_agl(
        CanonicalField::RadarReflectivity,
        1000,
    )),
    &["REFD:1000 m above ground", "REFD:1 km above ground"],
);

const FIELD_COMPOSITE_REFLECTIVITY: GribFieldSpec = field_spec(
    "composite_reflectivity",
    "Composite Reflectivity",
    ProductFamily::Native,
    GribLevelKind::EntireAtmosphere,
    None,
    Some(FieldSelector::entire_atmosphere(
        CanonicalField::CompositeReflectivity,
    )),
    &["REFC:entire atmosphere", "REFC:"],
);

const FIELD_UH: GribFieldSpec = field_spec(
    "updraft_helicity",
    "Updraft Helicity",
    ProductFamily::Native,
    GribLevelKind::HeightAboveGroundLayer,
    None,
    Some(FieldSelector::height_layer_agl(
        CanonicalField::UpdraftHelicity,
        2000,
        5000,
    )),
    &["MXUPHL:5000-2000", "UPHL:5000-2000", "UHEL:"],
);

/// A low-level interval-max updraft helicity layer, as HRRR/RRFS write
/// `UP_HELI_MAX01/02/03` (WOOF uh-layers, 2026-11-02). The WRF store path
/// reads it by selector; no GRIB fetch pattern is claimed for it.
const FIELD_UH_0TO1KM: GribFieldSpec = field_spec(
    "updraft_helicity_0to1km",
    "Updraft Helicity 0-1 km",
    ProductFamily::Native,
    GribLevelKind::HeightAboveGroundLayer,
    None,
    Some(FieldSelector::height_layer_agl(
        CanonicalField::UpdraftHelicity,
        0,
        1000,
    )),
    &[],
);

/// A low-level interval-max updraft helicity layer, as HRRR/RRFS write
/// `UP_HELI_MAX01/02/03` (WOOF uh-layers, 2026-11-02). The WRF store path
/// reads it by selector; no GRIB fetch pattern is claimed for it.
const FIELD_UH_0TO2KM: GribFieldSpec = field_spec(
    "updraft_helicity_0to2km",
    "Updraft Helicity 0-2 km",
    ProductFamily::Native,
    GribLevelKind::HeightAboveGroundLayer,
    None,
    Some(FieldSelector::height_layer_agl(
        CanonicalField::UpdraftHelicity,
        0,
        2000,
    )),
    &[],
);

/// A low-level interval-max updraft helicity layer, as HRRR/RRFS write
/// `UP_HELI_MAX01/02/03` (WOOF uh-layers, 2026-11-02). The WRF store path
/// reads it by selector; no GRIB fetch pattern is claimed for it.
const FIELD_UH_0TO3KM: GribFieldSpec = field_spec(
    "updraft_helicity_0to3km",
    "Updraft Helicity 0-3 km",
    ProductFamily::Native,
    GribLevelKind::HeightAboveGroundLayer,
    None,
    Some(FieldSelector::height_layer_agl(
        CanonicalField::UpdraftHelicity,
        0,
        3000,
    )),
    &[],
);

const FIELD_SMOKE_MASS_DENSITY_8M: GribFieldSpec = field_spec(
    "smoke_mass_density_8m_agl",
    "8m AGL Smoke Mass Density",
    ProductFamily::Native,
    GribLevelKind::HeightAboveGround,
    Some(8),
    Some(FieldSelector::height_agl(
        CanonicalField::SmokeMassDensity,
        8,
    )),
    &["MASSDEN:8 m above ground"],
);

const FIELD_COLUMN_INTEGRATED_SMOKE: GribFieldSpec = field_spec(
    "column_integrated_smoke",
    "Column-Integrated Smoke",
    ProductFamily::Native,
    GribLevelKind::EntireAtmosphere,
    None,
    Some(FieldSelector::entire_atmosphere(
        CanonicalField::ColumnIntegratedSmoke,
    )),
    &[
        "COLMD:entire atmosphere (considered as a single layer)",
        "COLMD:entire atmosphere",
    ],
);

const FIELD_PM25_SFC: GribFieldSpec = field_spec(
    "pm25_near_surface", "Lowest-Level Dry PM2.5", ProductFamily::Native,
    GribLevelKind::Surface, None,
    Some(FieldSelector::surface(CanonicalField::Pm25Dry)), &[],
);
const FIELD_DUST_SFC: GribFieldSpec = field_spec(
    "dust_near_surface", "Near-Surface Dust", ProductFamily::Native,
    GribLevelKind::Surface, None,
    Some(FieldSelector::surface(CanonicalField::DustMassConcentration)), &[],
);
const FIELD_OZONE_SFC: GribFieldSpec = field_spec(
    "ozone_near_surface", "Lowest-Level Ozone", ProductFamily::Native,
    GribLevelKind::Surface, None,
    Some(FieldSelector::surface(CanonicalField::OzoneConcentration)), &[],
);
const FIELD_AOD550: GribFieldSpec = field_spec(
    "aod_550", "550 nm Aerosol Optical Depth", ProductFamily::Native,
    GribLevelKind::EntireAtmosphere, None,
    Some(FieldSelector::entire_atmosphere(CanonicalField::AerosolOpticalDepth550)),
    &["AOTK:entire atmosphere"],
);

const FIELD_850_Q: GribFieldSpec = field_spec(
    "q_850mb", "850 hPa Specific Humidity", ProductFamily::Pressure,
    GribLevelKind::IsobaricHpa, Some(850),
    Some(FieldSelector::isobaric(CanonicalField::SpecificHumidity, 850)), &["SPFH:850 mb"],
);
const FIELD_500_OMEGA: GribFieldSpec = field_spec(
    "omega_500mb", "500 hPa Pressure Vertical Velocity", ProductFamily::Pressure,
    GribLevelKind::IsobaricHpa, Some(500),
    Some(FieldSelector::isobaric(CanonicalField::Omega, 500)), &["VVEL:500 mb"],
);
const FIELD_100_WIND: GribFieldSpec = field_spec(
    "wind_100m", "100 m AGL Wind Speed", ProductFamily::Native,
    GribLevelKind::HeightAboveGround, Some(100),
    Some(FieldSelector::height_agl(CanonicalField::WindSpeed, 100)), &["WIND:100 m above ground"],
);

// Native model-level moisture transport products.
const FIELD_IVTMAGNITUDE: GribFieldSpec = field_spec("ivt_1000_300", "IVT 1000-300 hPa", ProductFamily::Native, GribLevelKind::EntireAtmosphere, None, Some(FieldSelector::entire_atmosphere(CanonicalField::IvtMagnitude)), &[]);
const FIELD_IVTEASTWARD: GribFieldSpec = field_spec("ivt_eastward_1000_300", "Eastward IVT 1000-300 hPa", ProductFamily::Native, GribLevelKind::EntireAtmosphere, None, Some(FieldSelector::entire_atmosphere(CanonicalField::IvtEastward)), &[]);
const FIELD_IVTNORTHWARD: GribFieldSpec = field_spec("ivt_northward_1000_300", "Northward IVT 1000-300 hPa", ProductFamily::Native, GribLevelKind::EntireAtmosphere, None, Some(FieldSelector::entire_atmosphere(CanonicalField::IvtNorthward)), &[]);
const FIELD_INTEGRATEDWATERVAPOUR: GribFieldSpec = field_spec("integrated_water_vapour", "Integrated Water Vapour Surface to Model Top", ProductFamily::Native, GribLevelKind::EntireAtmosphere, None, Some(FieldSelector::entire_atmosphere(CanonicalField::IntegratedWaterVapour)), &[]);
const FIELD_FULLCOLUMNIVTEASTWARD: GribFieldSpec = field_spec("ivt_eastward_full_column", "Eastward Water Vapour Flux Surface to Model Top", ProductFamily::Native, GribLevelKind::EntireAtmosphere, None, Some(FieldSelector::entire_atmosphere(CanonicalField::FullColumnIvtEastward)), &[]);
const FIELD_FULLCOLUMNIVTNORTHWARD: GribFieldSpec = field_spec("ivt_northward_full_column", "Northward Water Vapour Flux Surface to Model Top", ProductFamily::Native, GribLevelKind::EntireAtmosphere, None, Some(FieldSelector::entire_atmosphere(CanonicalField::FullColumnIvtNorthward)), &[]);
const FIELD_FULLCOLUMNIVTMAGNITUDE: GribFieldSpec = field_spec("ivt_full_column", "Water Vapour Flux Surface to Model Top", ProductFamily::Native, GribLevelKind::EntireAtmosphere, None, Some(FieldSelector::entire_atmosphere(CanonicalField::FullColumnIvtMagnitude)), &[]);

const FIELD_WBGT: GribFieldSpec = field_spec(
    "wbgt_2m", "Outdoor WBGT, radiation balance", ProductFamily::Surface,
    GribLevelKind::Surface, None, Some(FieldSelector::surface(CanonicalField::Wbgt)), &[],
);
const FIELD_HOTDRYWINDY: GribFieldSpec = field_spec(
    "hot_dry_windy", "Hot Dry Windy, instantaneous", ProductFamily::Surface,
    GribLevelKind::Surface, None, Some(FieldSelector::surface(CanonicalField::HotDryWindy)), &[],
);
const FIELD_HAINESLOW: GribFieldSpec = field_spec(
    "haines_low", "Haines index, low elevation", ProductFamily::Surface,
    GribLevelKind::Surface, None, Some(FieldSelector::surface(CanonicalField::HainesLow)), &[],
);
const FIELD_HAINESMID: GribFieldSpec = field_spec(
    "haines_mid", "Haines index, middle elevation", ProductFamily::Surface,
    GribLevelKind::Surface, None, Some(FieldSelector::surface(CanonicalField::HainesMid)), &[],
);
const FIELD_HAINESHIGH: GribFieldSpec = field_spec(
    "haines_high", "Haines index, high elevation", ProductFamily::Surface,
    GribLevelKind::Surface, None, Some(FieldSelector::surface(CanonicalField::HainesHigh)), &[],
);
const FIELD_LIGHTNINGTHREAT1: GribFieldSpec = field_spec(
    "lightning_threat_1", "Lightning threat LTG1", ProductFamily::Surface,
    GribLevelKind::Surface, None, Some(FieldSelector::surface(CanonicalField::LightningThreat1)), &[],
);
const FIELD_LIGHTNINGTHREAT2: GribFieldSpec = field_spec(
    "lightning_threat_2", "Lightning threat LTG2", ProductFamily::Surface,
    GribLevelKind::Surface, None, Some(FieldSelector::surface(CanonicalField::LightningThreat2)), &[],
);
const FIELD_LIGHTNINGTHREAT3: GribFieldSpec = field_spec(
    "lightning_threat_3", "Lightning threat LTG3", ProductFamily::Surface,
    GribLevelKind::Surface, None, Some(FieldSelector::surface(CanonicalField::LightningThreat3)), &[],
);
const FIELD_PR92FLASHRATE: GribFieldSpec = field_spec(
    "pr92_flash_rate", "PR92 maximum-updraft flash rate", ProductFamily::Surface,
    GribLevelKind::Surface, None, Some(FieldSelector::surface(CanonicalField::Pr92FlashRate)), &[],
);

const PLOT_RECIPES: &[PlotRecipe] = &[
    PlotRecipe {
        slug: "wbgt_2m", title: "Outdoor WBGT, radiation balance", filled: FIELD_WBGT,
        contours: None, barbs_u: None, barbs_v: None, style: RenderStyle::WeatherWbgt,
    },
    PlotRecipe {
        slug: "hot_dry_windy", title: "Hot Dry Windy, instantaneous", filled: FIELD_HOTDRYWINDY,
        contours: None, barbs_u: None, barbs_v: None, style: RenderStyle::WeatherHdw,
    },
    PlotRecipe {
        slug: "haines_low", title: "Haines index, low elevation", filled: FIELD_HAINESLOW,
        contours: None, barbs_u: None, barbs_v: None, style: RenderStyle::WeatherHaines,
    },
    PlotRecipe {
        slug: "haines_mid", title: "Haines index, middle elevation", filled: FIELD_HAINESMID,
        contours: None, barbs_u: None, barbs_v: None, style: RenderStyle::WeatherHaines,
    },
    PlotRecipe {
        slug: "haines_high", title: "Haines index, high elevation", filled: FIELD_HAINESHIGH,
        contours: None, barbs_u: None, barbs_v: None, style: RenderStyle::WeatherHaines,
    },
    PlotRecipe {
        slug: "lightning_threat_1", title: "Lightning threat LTG1", filled: FIELD_LIGHTNINGTHREAT1,
        contours: None, barbs_u: None, barbs_v: None, style: RenderStyle::WeatherLightningThreat,
    },
    PlotRecipe {
        slug: "lightning_threat_2", title: "Lightning threat LTG2", filled: FIELD_LIGHTNINGTHREAT2,
        contours: None, barbs_u: None, barbs_v: None, style: RenderStyle::WeatherLightningThreat,
    },
    PlotRecipe {
        slug: "lightning_threat_3", title: "Lightning threat LTG3", filled: FIELD_LIGHTNINGTHREAT3,
        contours: None, barbs_u: None, barbs_v: None, style: RenderStyle::WeatherLightningThreat,
    },
    PlotRecipe {
        slug: "pr92_flash_rate", title: "PR92 maximum-updraft flash rate", filled: FIELD_PR92FLASHRATE,
        contours: None, barbs_u: None, barbs_v: None, style: RenderStyle::WeatherFlashRate,
    },

    PlotRecipe { slug: "850mb_specific_humidity", title: "850 hPa Specific Humidity", filled: FIELD_850_Q,
        contours: None, barbs_u: None, barbs_v: None, style: RenderStyle::WeatherSpecificHumidity },
    PlotRecipe { slug: "500mb_omega", title: "500 hPa Omega (negative is ascent)", filled: FIELD_500_OMEGA,
        contours: None, barbs_u: None, barbs_v: None, style: RenderStyle::WeatherOmega },
    PlotRecipe { slug: "100m_wind_speed", title: "100 m AGL Wind Speed", filled: FIELD_100_WIND,
        contours: None, barbs_u: None, barbs_v: None, style: RenderStyle::WeatherWinds },
    PlotRecipe {
        slug: "smoke_near_surface", title: "Near-Surface Smoke", filled: FIELD_SMOKE_MASS_DENSITY_8M,
        contours: None, barbs_u: None, barbs_v: None, style: RenderStyle::AirQualityPm25,
    },
    PlotRecipe {
        slug: "pm25_near_surface", title: "Lowest-Level Dry PM2.5", filled: FIELD_PM25_SFC,
        contours: None, barbs_u: None, barbs_v: None, style: RenderStyle::AirQualityPm25,
    },
    PlotRecipe {
        slug: "dust_near_surface", title: "Near-Surface Dust", filled: FIELD_DUST_SFC,
        contours: None, barbs_u: None, barbs_v: None, style: RenderStyle::DustConcentration,
    },
    PlotRecipe {
        slug: "ozone_near_surface", title: "Lowest-Level Ozone (instantaneous)", filled: FIELD_OZONE_SFC,
        contours: None, barbs_u: None, barbs_v: None, style: RenderStyle::AirQualityOzone,
    },
    PlotRecipe {
        slug: "aod_550", title: "550 nm Aerosol Optical Depth", filled: FIELD_AOD550,
        contours: None, barbs_u: None, barbs_v: None, style: RenderStyle::AerosolOpticalDepth,
    },

    PlotRecipe {
        slug: "200mb_height_winds",
        title: "200mb Height / Winds",
        filled: FIELD_200_HEIGHT,
        contours: Some(FIELD_200_HEIGHT),
        barbs_u: Some(FIELD_200_U),
        barbs_v: Some(FIELD_200_V),
        style: RenderStyle::WeatherHeight,
    },
    PlotRecipe {
        slug: "300mb_height_winds",
        title: "300mb Height / Winds",
        filled: FIELD_300_HEIGHT,
        contours: Some(FIELD_300_HEIGHT),
        barbs_u: Some(FIELD_300_U),
        barbs_v: Some(FIELD_300_V),
        style: RenderStyle::WeatherHeight,
    },
    PlotRecipe {
        slug: "250mb_height_winds",
        title: "250mb Height / Winds",
        filled: FIELD_250_HEIGHT,
        contours: Some(FIELD_250_HEIGHT),
        barbs_u: Some(FIELD_250_U),
        barbs_v: Some(FIELD_250_V),
        style: RenderStyle::WeatherHeight,
    },
    PlotRecipe {
        slug: "500mb_height_winds",
        title: "500mb Height / Winds",
        filled: FIELD_500_HEIGHT,
        contours: Some(FIELD_500_HEIGHT),
        barbs_u: Some(FIELD_500_U),
        barbs_v: Some(FIELD_500_V),
        style: RenderStyle::WeatherHeight,
    },
    PlotRecipe {
        slug: "gefs_avg_500mb_height_winds",
        title: "GEFS 500mb Height / Winds Mean",
        filled: FIELD_GEFS_AVG_500_HEIGHT,
        contours: Some(FIELD_GEFS_AVG_500_HEIGHT),
        barbs_u: Some(FIELD_GEFS_AVG_500_U),
        barbs_v: Some(FIELD_GEFS_AVG_500_V),
        style: RenderStyle::WeatherHeight,
    },
    PlotRecipe {
        slug: "gefs_spr_500mb_height_stddev",
        title: "GEFS 500mb Height Spread",
        filled: FIELD_GEFS_SPR_500_HEIGHT_STDDEV,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherHeight,
    },
    PlotRecipe {
        slug: "700mb_height_winds",
        title: "700mb Height / Winds",
        filled: FIELD_700_HEIGHT,
        contours: Some(FIELD_700_HEIGHT),
        barbs_u: Some(FIELD_700_U),
        barbs_v: Some(FIELD_700_V),
        style: RenderStyle::WeatherHeight,
    },
    PlotRecipe {
        slug: "850mb_height_winds",
        title: "850mb Height / Winds",
        filled: FIELD_850_HEIGHT,
        contours: Some(FIELD_850_HEIGHT),
        barbs_u: Some(FIELD_850_U),
        barbs_v: Some(FIELD_850_V),
        style: RenderStyle::WeatherHeight,
    },
    PlotRecipe {
        slug: "200mb_temperature_height_winds",
        title: "200mb Temperature / Height / Winds",
        filled: FIELD_200_TEMP,
        contours: Some(FIELD_200_HEIGHT),
        barbs_u: Some(FIELD_200_U),
        barbs_v: Some(FIELD_200_V),
        style: RenderStyle::WeatherTemperature,
    },
    PlotRecipe {
        slug: "300mb_temperature_height_winds",
        title: "300mb Temperature / Height / Winds",
        filled: FIELD_300_TEMP,
        contours: Some(FIELD_300_HEIGHT),
        barbs_u: Some(FIELD_300_U),
        barbs_v: Some(FIELD_300_V),
        style: RenderStyle::WeatherTemperature,
    },
    PlotRecipe {
        slug: "250mb_temperature_height_winds",
        title: "250mb Temperature / Height / Winds",
        filled: FIELD_250_TEMP,
        contours: Some(FIELD_250_HEIGHT),
        barbs_u: Some(FIELD_250_U),
        barbs_v: Some(FIELD_250_V),
        style: RenderStyle::WeatherTemperature,
    },
    PlotRecipe {
        slug: "500mb_temperature_height_winds",
        title: "500mb Temperature / Height / Winds",
        filled: FIELD_500_TEMP,
        contours: Some(FIELD_500_HEIGHT),
        barbs_u: Some(FIELD_500_U),
        barbs_v: Some(FIELD_500_V),
        style: RenderStyle::WeatherTemperature,
    },
    PlotRecipe {
        slug: "gefs_avg_500mb_temperature_height_winds",
        title: "GEFS 500mb Temperature / Height / Winds Mean",
        filled: FIELD_GEFS_AVG_500_TEMP,
        contours: Some(FIELD_GEFS_AVG_500_HEIGHT),
        barbs_u: Some(FIELD_GEFS_AVG_500_U),
        barbs_v: Some(FIELD_GEFS_AVG_500_V),
        style: RenderStyle::WeatherTemperature,
    },
    PlotRecipe {
        slug: "gefs_spr_500mb_temperature_stddev",
        title: "GEFS 500mb Temperature Spread",
        filled: FIELD_GEFS_SPR_500_TEMP_STDDEV,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherTemperature,
    },
    PlotRecipe {
        slug: "850mb_temperature_height_winds",
        title: "850mb Temperature / Height / Winds",
        filled: FIELD_850_TEMP,
        contours: Some(FIELD_850_HEIGHT),
        barbs_u: Some(FIELD_850_U),
        barbs_v: Some(FIELD_850_V),
        style: RenderStyle::WeatherTemperature,
    },
    PlotRecipe {
        slug: "700mb_temperature_height_winds",
        title: "700mb Temperature / Height / Winds",
        filled: FIELD_700_TEMP,
        contours: Some(FIELD_700_HEIGHT),
        barbs_u: Some(FIELD_700_U),
        barbs_v: Some(FIELD_700_V),
        style: RenderStyle::WeatherTemperature,
    },
    PlotRecipe {
        slug: "2m_relative_humidity",
        title: "2m AGL Relative Humidity",
        filled: FIELD_2M_RH,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherRh,
    },
    PlotRecipe {
        slug: "2m_relative_humidity_10m_winds",
        title: "2m AGL Relative Humidity / 10m Winds",
        filled: FIELD_2M_RH,
        contours: Some(FIELD_MSLP),
        barbs_u: Some(FIELD_10M_U),
        barbs_v: Some(FIELD_10M_V),
        style: RenderStyle::WeatherRh,
    },
    PlotRecipe {
        slug: "2m_temperature",
        title: "2m AGL Temperature",
        filled: FIELD_2M_TEMP,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherTemperature,
    },
    PlotRecipe {
        slug: "nbm_qmd_2m_temperature_mean",
        title: "NBM QMD 2m AGL Temperature Mean",
        filled: FIELD_QMD_2M_TEMP_MEAN,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherTemperature,
    },
    PlotRecipe {
        slug: "nbm_qmd_2m_temperature_stddev",
        title: "NBM QMD 2m AGL Temperature Std Dev",
        filled: FIELD_QMD_2M_TEMP_STDDEV,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherTemperature,
    },
    PlotRecipe {
        slug: "nbm_qmd_2m_temperature_p05",
        title: "NBM QMD 2m AGL Temperature P05",
        filled: FIELD_QMD_2M_TEMP_P05,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherTemperature,
    },
    PlotRecipe {
        slug: "nbm_qmd_2m_temperature_p10",
        title: "NBM QMD 2m AGL Temperature P10",
        filled: FIELD_QMD_2M_TEMP_P10,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherTemperature,
    },
    PlotRecipe {
        slug: "nbm_qmd_2m_temperature_p25",
        title: "NBM QMD 2m AGL Temperature P25",
        filled: FIELD_QMD_2M_TEMP_P25,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherTemperature,
    },
    PlotRecipe {
        slug: "nbm_qmd_2m_temperature_p50",
        title: "NBM QMD 2m AGL Temperature P50",
        filled: FIELD_QMD_2M_TEMP_P50,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherTemperature,
    },
    PlotRecipe {
        slug: "nbm_qmd_2m_temperature_p75",
        title: "NBM QMD 2m AGL Temperature P75",
        filled: FIELD_QMD_2M_TEMP_P75,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherTemperature,
    },
    PlotRecipe {
        slug: "nbm_qmd_2m_temperature_p90",
        title: "NBM QMD 2m AGL Temperature P90",
        filled: FIELD_QMD_2M_TEMP_P90,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherTemperature,
    },
    PlotRecipe {
        slug: "nbm_qmd_2m_temperature_p95",
        title: "NBM QMD 2m AGL Temperature P95",
        filled: FIELD_QMD_2M_TEMP_P95,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherTemperature,
    },
    PlotRecipe {
        slug: "nbm_qmd_prob_2m_temperature_below_270p928k",
        title: "NBM QMD Probability 2m AGL Temperature < 270.928 K",
        filled: FIELD_QMD_PROB_2M_TEMP_BELOW_270P928K,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherProbability,
    },
    PlotRecipe {
        slug: "nbm_qmd_prob_2m_temperature_below_273p15k",
        title: "NBM QMD Probability 2m AGL Temperature < 273.15 K",
        filled: FIELD_QMD_PROB_2M_TEMP_BELOW_273P15K,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherProbability,
    },
    PlotRecipe {
        slug: "nbm_qmd_prob_2m_temperature_above_299p817k",
        title: "NBM QMD Probability 2m AGL Temperature > 299.817 K",
        filled: FIELD_QMD_PROB_2M_TEMP_ABOVE_299P817K,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherProbability,
    },
    PlotRecipe {
        slug: "nbm_qmd_prob_2m_temperature_above_305p372k",
        title: "NBM QMD Probability 2m AGL Temperature > 305.372 K",
        filled: FIELD_QMD_PROB_2M_TEMP_ABOVE_305P372K,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherProbability,
    },
    PlotRecipe {
        slug: "nbm_qmd_prob_2m_temperature_above_310p928k",
        title: "NBM QMD Probability 2m AGL Temperature > 310.928 K",
        filled: FIELD_QMD_PROB_2M_TEMP_ABOVE_310P928K,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherProbability,
    },
    PlotRecipe {
        slug: "nbm_qmd_prob_2m_temperature_above_316p483k",
        title: "NBM QMD Probability 2m AGL Temperature > 316.483 K",
        filled: FIELD_QMD_PROB_2M_TEMP_ABOVE_316P483K,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherProbability,
    },
    PlotRecipe {
        slug: "nbm_qmd_2m_dewpoint_mean",
        title: "NBM QMD 2m AGL Dewpoint Mean",
        filled: FIELD_QMD_2M_DEWPOINT_MEAN,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherDewpoint,
    },
    PlotRecipe {
        slug: "nbm_qmd_2m_dewpoint_stddev",
        title: "NBM QMD 2m AGL Dewpoint Std Dev",
        filled: FIELD_QMD_2M_DEWPOINT_STDDEV,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherDewpoint,
    },
    PlotRecipe {
        slug: "nbm_qmd_2m_dewpoint_p05",
        title: "NBM QMD 2m AGL Dewpoint P05",
        filled: FIELD_QMD_2M_DEWPOINT_P05,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherDewpoint,
    },
    PlotRecipe {
        slug: "nbm_qmd_2m_dewpoint_p10",
        title: "NBM QMD 2m AGL Dewpoint P10",
        filled: FIELD_QMD_2M_DEWPOINT_P10,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherDewpoint,
    },
    PlotRecipe {
        slug: "nbm_qmd_2m_dewpoint_p25",
        title: "NBM QMD 2m AGL Dewpoint P25",
        filled: FIELD_QMD_2M_DEWPOINT_P25,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherDewpoint,
    },
    PlotRecipe {
        slug: "nbm_qmd_2m_dewpoint_p50",
        title: "NBM QMD 2m AGL Dewpoint P50",
        filled: FIELD_QMD_2M_DEWPOINT_P50,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherDewpoint,
    },
    PlotRecipe {
        slug: "nbm_qmd_2m_dewpoint_p75",
        title: "NBM QMD 2m AGL Dewpoint P75",
        filled: FIELD_QMD_2M_DEWPOINT_P75,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherDewpoint,
    },
    PlotRecipe {
        slug: "nbm_qmd_2m_dewpoint_p90",
        title: "NBM QMD 2m AGL Dewpoint P90",
        filled: FIELD_QMD_2M_DEWPOINT_P90,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherDewpoint,
    },
    PlotRecipe {
        slug: "nbm_qmd_2m_dewpoint_p95",
        title: "NBM QMD 2m AGL Dewpoint P95",
        filled: FIELD_QMD_2M_DEWPOINT_P95,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherDewpoint,
    },
    PlotRecipe {
        slug: "nbm_qmd_prob_2m_dewpoint_below_273p15k",
        title: "NBM QMD Probability 2m AGL Dewpoint < 273.15 K",
        filled: FIELD_QMD_PROB_2M_DEWPOINT_BELOW_273P15K,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherProbability,
    },
    PlotRecipe {
        slug: "nbm_qmd_prob_2m_dewpoint_above_288p706k",
        title: "NBM QMD Probability 2m AGL Dewpoint > 288.706 K",
        filled: FIELD_QMD_PROB_2M_DEWPOINT_ABOVE_288P706K,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherProbability,
    },
    PlotRecipe {
        slug: "nbm_qmd_prob_2m_dewpoint_above_291p483k",
        title: "NBM QMD Probability 2m AGL Dewpoint > 291.483 K",
        filled: FIELD_QMD_PROB_2M_DEWPOINT_ABOVE_291P483K,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherProbability,
    },
    PlotRecipe {
        slug: "nbm_qmd_prob_2m_dewpoint_above_294p261k",
        title: "NBM QMD Probability 2m AGL Dewpoint > 294.261 K",
        filled: FIELD_QMD_PROB_2M_DEWPOINT_ABOVE_294P261K,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherProbability,
    },
    PlotRecipe {
        slug: "nbm_qmd_prob_2m_dewpoint_above_297p039k",
        title: "NBM QMD Probability 2m AGL Dewpoint > 297.039 K",
        filled: FIELD_QMD_PROB_2M_DEWPOINT_ABOVE_297P039K,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherProbability,
    },
    PlotRecipe {
        slug: "nbm_qmd_2m_relative_humidity_p05",
        title: "NBM QMD 2m AGL Relative Humidity P05",
        filled: FIELD_QMD_2M_RH_P05,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherRh,
    },
    PlotRecipe {
        slug: "nbm_qmd_2m_relative_humidity_p10",
        title: "NBM QMD 2m AGL Relative Humidity P10",
        filled: FIELD_QMD_2M_RH_P10,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherRh,
    },
    PlotRecipe {
        slug: "nbm_qmd_2m_relative_humidity_p25",
        title: "NBM QMD 2m AGL Relative Humidity P25",
        filled: FIELD_QMD_2M_RH_P25,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherRh,
    },
    PlotRecipe {
        slug: "nbm_qmd_2m_relative_humidity_p50",
        title: "NBM QMD 2m AGL Relative Humidity P50",
        filled: FIELD_QMD_2M_RH_P50,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherRh,
    },
    PlotRecipe {
        slug: "nbm_qmd_2m_relative_humidity_p75",
        title: "NBM QMD 2m AGL Relative Humidity P75",
        filled: FIELD_QMD_2M_RH_P75,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherRh,
    },
    PlotRecipe {
        slug: "nbm_qmd_2m_relative_humidity_p90",
        title: "NBM QMD 2m AGL Relative Humidity P90",
        filled: FIELD_QMD_2M_RH_P90,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherRh,
    },
    PlotRecipe {
        slug: "nbm_qmd_2m_relative_humidity_p95",
        title: "NBM QMD 2m AGL Relative Humidity P95",
        filled: FIELD_QMD_2M_RH_P95,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherRh,
    },
    PlotRecipe {
        slug: "nbm_qmd_10m_wind_gust_mean",
        title: "NBM QMD 10m AGL Wind Gust Mean",
        filled: FIELD_QMD_10M_WIND_GUST_MEAN,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherWindGust,
    },
    PlotRecipe {
        slug: "nbm_qmd_10m_wind_gust_stddev",
        title: "NBM QMD 10m AGL Wind Gust Std Dev",
        filled: FIELD_QMD_10M_WIND_GUST_STDDEV,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherWindGust,
    },
    PlotRecipe {
        slug: "nbm_qmd_prob_10m_wind_gust_above_17p4911ms",
        title: "NBM QMD Probability 10m AGL Wind Gust > 17.4911 m/s",
        filled: FIELD_QMD_PROB_10M_WIND_GUST_ABOVE_17P4911MS,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherProbability,
    },
    PlotRecipe {
        slug: "nbm_qmd_prob_10m_wind_gust_above_21p0922ms",
        title: "NBM QMD Probability 10m AGL Wind Gust > 21.0922 m/s",
        filled: FIELD_QMD_PROB_10M_WIND_GUST_ABOVE_21P0922MS,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherProbability,
    },
    PlotRecipe {
        slug: "nbm_qmd_10m_wind_gust_p05",
        title: "NBM QMD 10m AGL Wind Gust P05",
        filled: FIELD_QMD_10M_WIND_GUST_P05,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherWindGust,
    },
    PlotRecipe {
        slug: "nbm_qmd_10m_wind_gust_p10",
        title: "NBM QMD 10m AGL Wind Gust P10",
        filled: FIELD_QMD_10M_WIND_GUST_P10,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherWindGust,
    },
    PlotRecipe {
        slug: "nbm_qmd_10m_wind_gust_p25",
        title: "NBM QMD 10m AGL Wind Gust P25",
        filled: FIELD_QMD_10M_WIND_GUST_P25,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherWindGust,
    },
    PlotRecipe {
        slug: "nbm_qmd_10m_wind_gust_p50",
        title: "NBM QMD 10m AGL Wind Gust P50",
        filled: FIELD_QMD_10M_WIND_GUST_P50,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherWindGust,
    },
    PlotRecipe {
        slug: "nbm_qmd_10m_wind_gust_p75",
        title: "NBM QMD 10m AGL Wind Gust P75",
        filled: FIELD_QMD_10M_WIND_GUST_P75,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherWindGust,
    },
    PlotRecipe {
        slug: "nbm_qmd_10m_wind_gust_p90",
        title: "NBM QMD 10m AGL Wind Gust P90",
        filled: FIELD_QMD_10M_WIND_GUST_P90,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherWindGust,
    },
    PlotRecipe {
        slug: "nbm_qmd_10m_wind_gust_p95",
        title: "NBM QMD 10m AGL Wind Gust P95",
        filled: FIELD_QMD_10M_WIND_GUST_P95,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherWindGust,
    },
    PlotRecipe {
        slug: "nbm_qmd_prob_10m_wind_gust_above_24p6933ms",
        title: "NBM QMD Probability 10m AGL Wind Gust > 24.6933 m/s",
        filled: FIELD_QMD_PROB_10M_WIND_GUST_ABOVE_24P6933MS,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherProbability,
    },
    PlotRecipe {
        slug: "nbm_qmd_prob_10m_wind_gust_above_28p8089ms",
        title: "NBM QMD Probability 10m AGL Wind Gust > 28.8089 m/s",
        filled: FIELD_QMD_PROB_10M_WIND_GUST_ABOVE_28P8089MS,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherProbability,
    },
    PlotRecipe {
        slug: "nbm_qmd_prob_10m_wind_gust_above_32p9244ms",
        title: "NBM QMD Probability 10m AGL Wind Gust > 32.9244 m/s",
        filled: FIELD_QMD_PROB_10M_WIND_GUST_ABOVE_32P9244MS,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherProbability,
    },
    PlotRecipe {
        slug: "nbm_qmd_10m_wind_speed_mean",
        title: "NBM QMD 10m AGL Wind Speed Mean",
        filled: FIELD_QMD_10M_WIND_SPEED_MEAN,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherWinds,
    },
    PlotRecipe {
        slug: "nbm_qmd_10m_wind_speed_stddev",
        title: "NBM QMD 10m AGL Wind Speed Std Dev",
        filled: FIELD_QMD_10M_WIND_SPEED_STDDEV,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherWinds,
    },
    PlotRecipe {
        slug: "nbm_qmd_prob_10m_wind_speed_above_8p7456ms",
        title: "NBM QMD Probability 10m AGL Wind Speed > 8.7456 m/s",
        filled: FIELD_QMD_PROB_10M_WIND_SPEED_ABOVE_8P7456MS,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherProbability,
    },
    PlotRecipe {
        slug: "nbm_qmd_prob_10m_wind_speed_above_11p3177ms",
        title: "NBM QMD Probability 10m AGL Wind Speed > 11.3177 m/s",
        filled: FIELD_QMD_PROB_10M_WIND_SPEED_ABOVE_11P3177MS,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherProbability,
    },
    PlotRecipe {
        slug: "nbm_qmd_10m_wind_speed_p05",
        title: "NBM QMD 10m AGL Wind Speed P05",
        filled: FIELD_QMD_10M_WIND_SPEED_P05,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherWinds,
    },
    PlotRecipe {
        slug: "nbm_qmd_10m_wind_speed_p10",
        title: "NBM QMD 10m AGL Wind Speed P10",
        filled: FIELD_QMD_10M_WIND_SPEED_P10,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherWinds,
    },
    PlotRecipe {
        slug: "nbm_qmd_10m_wind_speed_p25",
        title: "NBM QMD 10m AGL Wind Speed P25",
        filled: FIELD_QMD_10M_WIND_SPEED_P25,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherWinds,
    },
    PlotRecipe {
        slug: "nbm_qmd_10m_wind_speed_p50",
        title: "NBM QMD 10m AGL Wind Speed P50",
        filled: FIELD_QMD_10M_WIND_SPEED_P50,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherWinds,
    },
    PlotRecipe {
        slug: "nbm_qmd_10m_wind_speed_p75",
        title: "NBM QMD 10m AGL Wind Speed P75",
        filled: FIELD_QMD_10M_WIND_SPEED_P75,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherWinds,
    },
    PlotRecipe {
        slug: "nbm_qmd_10m_wind_speed_p90",
        title: "NBM QMD 10m AGL Wind Speed P90",
        filled: FIELD_QMD_10M_WIND_SPEED_P90,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherWinds,
    },
    PlotRecipe {
        slug: "nbm_qmd_10m_wind_speed_p95",
        title: "NBM QMD 10m AGL Wind Speed P95",
        filled: FIELD_QMD_10M_WIND_SPEED_P95,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherWinds,
    },
    PlotRecipe {
        slug: "nbm_qmd_prob_10m_wind_speed_above_15p4333ms",
        title: "NBM QMD Probability 10m AGL Wind Speed > 15.4333 m/s",
        filled: FIELD_QMD_PROB_10M_WIND_SPEED_ABOVE_15P4333MS,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherProbability,
    },
    PlotRecipe {
        slug: "nbm_qmd_prob_10m_wind_speed_above_17p4911ms",
        title: "NBM QMD Probability 10m AGL Wind Speed > 17.4911 m/s",
        filled: FIELD_QMD_PROB_10M_WIND_SPEED_ABOVE_17P4911MS,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherProbability,
    },
    PlotRecipe {
        slug: "nbm_qmd_prob_10m_wind_speed_above_24p6933ms",
        title: "NBM QMD Probability 10m AGL Wind Speed > 24.6933 m/s",
        filled: FIELD_QMD_PROB_10M_WIND_SPEED_ABOVE_24P6933MS,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherProbability,
    },
    PlotRecipe {
        slug: "aigefs_spr_2m_temperature_stddev",
        title: "AI-GEFS 2m AGL Temperature Spread",
        filled: FIELD_AIGEFS_SPR_2M_TEMP_STDDEV,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherTemperature,
    },
    PlotRecipe {
        slug: "aigefs_spr_500mb_height_stddev",
        title: "AI-GEFS 500mb Height Spread",
        filled: FIELD_AIGEFS_SPR_500_HEIGHT_STDDEV,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherHeight,
    },
    PlotRecipe {
        slug: "aigefs_spr_500mb_temperature_stddev",
        title: "AI-GEFS 500mb Temperature Spread",
        filled: FIELD_AIGEFS_SPR_500_TEMP_STDDEV,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherTemperature,
    },
    PlotRecipe {
        slug: "aigefs_spr_mslp_stddev",
        title: "AI-GEFS MSLP Spread",
        filled: FIELD_AIGEFS_SPR_MSLP_STDDEV,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherPressure,
    },
    PlotRecipe {
        slug: "aigefs_spr_6h_qpf_stddev",
        title: "AI-GEFS 6h QPF Spread",
        filled: FIELD_AIGEFS_SPR_6H_QPF_STDDEV,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherQpf,
    },
    PlotRecipe {
        slug: "hgefs_spr_2m_temperature_stddev",
        title: "HGEFS 2m AGL Temperature Spread",
        filled: FIELD_HGEFS_SPR_2M_TEMP_STDDEV,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherTemperature,
    },
    PlotRecipe {
        slug: "hgefs_spr_500mb_height_stddev",
        title: "HGEFS 500mb Height Spread",
        filled: FIELD_HGEFS_SPR_500_HEIGHT_STDDEV,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherHeight,
    },
    PlotRecipe {
        slug: "hgefs_spr_500mb_temperature_stddev",
        title: "HGEFS 500mb Temperature Spread",
        filled: FIELD_HGEFS_SPR_500_TEMP_STDDEV,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherTemperature,
    },
    PlotRecipe {
        slug: "hgefs_spr_mslp_stddev",
        title: "HGEFS MSLP Spread",
        filled: FIELD_HGEFS_SPR_MSLP_STDDEV,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherPressure,
    },
    PlotRecipe {
        slug: "hgefs_spr_6h_qpf_stddev",
        title: "HGEFS 6h QPF Spread",
        filled: FIELD_HGEFS_SPR_6H_QPF_STDDEV,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherQpf,
    },
    PlotRecipe {
        slug: "href_sprd_2m_temperature",
        title: "HREF 2m AGL Temperature Spread",
        filled: FIELD_HREF_SPRD_2M_TEMP,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherTemperature,
    },
    PlotRecipe {
        slug: "href_sprd_2m_dewpoint",
        title: "HREF 2m AGL Dewpoint Spread",
        filled: FIELD_HREF_SPRD_2M_DEWPOINT,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherDewpoint,
    },
    PlotRecipe {
        slug: "href_sprd_10m_wind_speed",
        title: "HREF 10m AGL Wind Speed Spread",
        filled: FIELD_HREF_SPRD_10M_WIND_SPEED,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherWinds,
    },
    PlotRecipe {
        slug: "href_sprd_mslp",
        title: "HREF MSLP Spread",
        filled: FIELD_HREF_SPRD_MSLP,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherPressure,
    },
    PlotRecipe {
        slug: "href_sprd_precipitable_water",
        title: "HREF Precipitable Water Spread",
        filled: FIELD_HREF_SPRD_PWAT,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherPrecipitableWater,
    },
    PlotRecipe {
        slug: "href_sprd_500mb_height",
        title: "HREF 500mb Height Spread",
        filled: FIELD_HREF_SPRD_500_HEIGHT,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherHeight,
    },
    PlotRecipe {
        slug: "href_sprd_500mb_temperature",
        title: "HREF 500mb Temperature Spread",
        filled: FIELD_HREF_SPRD_500_TEMP,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherTemperature,
    },
    PlotRecipe {
        slug: "href_mean_2m_temperature",
        title: "HREF 2m AGL Temperature Mean",
        filled: FIELD_HREF_MEAN_2M_TEMP,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherTemperature,
    },
    PlotRecipe {
        slug: "href_mean_2m_dewpoint",
        title: "HREF 2m AGL Dewpoint Mean",
        filled: FIELD_HREF_MEAN_2M_DEWPOINT,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherDewpoint,
    },
    PlotRecipe {
        slug: "href_mean_10m_wind_speed",
        title: "HREF 10m AGL Wind Speed Mean",
        filled: FIELD_HREF_MEAN_10M_WIND_SPEED,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherWinds,
    },
    PlotRecipe {
        slug: "href_mean_mslp",
        title: "HREF MSLP Mean",
        filled: FIELD_HREF_MEAN_MSLP,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherPressure,
    },
    PlotRecipe {
        slug: "href_mean_precipitable_water",
        title: "HREF Precipitable Water Mean",
        filled: FIELD_HREF_MEAN_PWAT,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherPrecipitableWater,
    },
    PlotRecipe {
        slug: "href_mean_visibility",
        title: "HREF Visibility Mean",
        filled: FIELD_HREF_MEAN_VISIBILITY,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherVisibility,
    },
    PlotRecipe {
        slug: "href_mean_cloud_cover",
        title: "HREF Total Cloud Cover Mean",
        filled: FIELD_HREF_MEAN_TOTAL_CLOUD_COVER,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherCloudCover,
    },
    PlotRecipe {
        slug: "href_mean_500mb_height",
        title: "HREF 500mb Height Mean",
        filled: FIELD_HREF_MEAN_500_HEIGHT,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherHeight,
    },
    PlotRecipe {
        slug: "href_mean_500mb_temperature",
        title: "HREF 500mb Temperature Mean",
        filled: FIELD_HREF_MEAN_500_TEMP,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherTemperature,
    },
    PlotRecipe {
        slug: "refs_sprd_2m_temperature",
        title: "REFS 2m AGL Temperature Spread",
        filled: FIELD_REFS_SPRD_2M_TEMP,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherTemperature,
    },
    PlotRecipe {
        slug: "refs_sprd_2m_dewpoint",
        title: "REFS 2m AGL Dewpoint Spread",
        filled: FIELD_REFS_SPRD_2M_DEWPOINT,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherDewpoint,
    },
    PlotRecipe {
        slug: "refs_sprd_10m_wind_speed",
        title: "REFS 10m AGL Wind Speed Spread",
        filled: FIELD_REFS_SPRD_10M_WIND_SPEED,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherWinds,
    },
    PlotRecipe {
        slug: "refs_sprd_mslp",
        title: "REFS MSLP Spread",
        filled: FIELD_REFS_SPRD_MSLP,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherPressure,
    },
    PlotRecipe {
        slug: "refs_sprd_precipitable_water",
        title: "REFS Precipitable Water Spread",
        filled: FIELD_REFS_SPRD_PWAT,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherPrecipitableWater,
    },
    PlotRecipe {
        slug: "refs_sprd_visibility",
        title: "REFS Visibility Spread",
        filled: FIELD_REFS_SPRD_VISIBILITY,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherVisibility,
    },
    PlotRecipe {
        slug: "refs_sprd_cloud_cover",
        title: "REFS Total Cloud Cover Spread",
        filled: FIELD_REFS_SPRD_TOTAL_CLOUD_COVER,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherCloudCover,
    },
    PlotRecipe {
        slug: "refs_sprd_500mb_height",
        title: "REFS 500mb Height Spread",
        filled: FIELD_REFS_SPRD_500_HEIGHT,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherHeight,
    },
    PlotRecipe {
        slug: "refs_sprd_500mb_temperature",
        title: "REFS 500mb Temperature Spread",
        filled: FIELD_REFS_SPRD_500_TEMP,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherTemperature,
    },
    PlotRecipe {
        slug: "refs_prob_uh_2to5km_above_25",
        title: "REFS Probability 2-5 km Updraft Helicity > 25",
        filled: FIELD_REFS_PROB_UH_ABOVE_25,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherProbability,
    },
    PlotRecipe {
        slug: "refs_prob_uh_2to5km_above_75",
        title: "REFS Probability 2-5 km Updraft Helicity > 75",
        filled: FIELD_REFS_PROB_UH_ABOVE_75,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherProbability,
    },
    PlotRecipe {
        slug: "refs_prob_uh_2to5km_above_150",
        title: "REFS Probability 2-5 km Updraft Helicity > 150",
        filled: FIELD_REFS_PROB_UH_ABOVE_150,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherProbability,
    },
    PlotRecipe {
        slug: "refs_prob_2m_temperature_below_273p15k",
        title: "REFS Probability 2m Temperature < 273.15 K",
        filled: FIELD_REFS_PROB_2M_TEMP_BELOW_273P15K,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherProbability,
    },
    PlotRecipe {
        slug: "refs_prob_2m_dewpoint_above_291p48k",
        title: "REFS Probability 2m Dewpoint > 291.48 K",
        filled: FIELD_REFS_PROB_2M_DEWPOINT_ABOVE_291P48K,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherProbability,
    },
    PlotRecipe {
        slug: "refs_prob_2m_dewpoint_above_294p26k",
        title: "REFS Probability 2m Dewpoint > 294.26 K",
        filled: FIELD_REFS_PROB_2M_DEWPOINT_ABOVE_294P26K,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherProbability,
    },
    PlotRecipe {
        slug: "refs_prob_pwat_above_25mm",
        title: "REFS Probability Precipitable Water > 25 mm",
        filled: FIELD_REFS_PROB_PWAT_ABOVE_25MM,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherProbability,
    },
    PlotRecipe {
        slug: "refs_prob_pwat_above_37p5mm",
        title: "REFS Probability Precipitable Water > 37.5 mm",
        filled: FIELD_REFS_PROB_PWAT_ABOVE_37P5MM,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherProbability,
    },
    PlotRecipe {
        slug: "refs_prob_pwat_above_50mm",
        title: "REFS Probability Precipitable Water > 50 mm",
        filled: FIELD_REFS_PROB_PWAT_ABOVE_50MM,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherProbability,
    },
    PlotRecipe {
        slug: "refs_prob_qpf_above_1mm",
        title: "REFS Probability Total QPF > 1 mm",
        filled: FIELD_REFS_PROB_QPF_ABOVE_1MM,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherProbability,
    },
    PlotRecipe {
        slug: "refs_prob_qpf_above_2mm",
        title: "REFS Probability Total QPF > 2 mm",
        filled: FIELD_REFS_PROB_QPF_ABOVE_2MM,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherProbability,
    },
    PlotRecipe {
        slug: "refs_prob_qpf_above_5mm",
        title: "REFS Probability Total QPF > 5 mm",
        filled: FIELD_REFS_PROB_QPF_ABOVE_5MM,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherProbability,
    },
    PlotRecipe {
        slug: "refs_prob_qpf_above_10mm",
        title: "REFS Probability Total QPF > 10 mm",
        filled: FIELD_REFS_PROB_QPF_ABOVE_10MM,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherProbability,
    },
    PlotRecipe {
        slug: "refs_prob_qpf_above_12p7mm",
        title: "REFS Probability Total QPF > 12.7 mm",
        filled: FIELD_REFS_PROB_QPF_ABOVE_12P7MM,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherProbability,
    },
    PlotRecipe {
        slug: "refs_prob_qpf_above_25mm",
        title: "REFS Probability Total QPF > 25 mm",
        filled: FIELD_REFS_PROB_QPF_ABOVE_25MM,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherProbability,
    },
    PlotRecipe {
        slug: "refs_prob_qpf_above_25p4mm",
        title: "REFS Probability Total QPF > 25.4 mm",
        filled: FIELD_REFS_PROB_QPF_ABOVE_25P4MM,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherProbability,
    },
    PlotRecipe {
        slug: "refs_prob_qpf_above_50mm",
        title: "REFS Probability Total QPF > 50 mm",
        filled: FIELD_REFS_PROB_QPF_ABOVE_50MM,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherProbability,
    },
    PlotRecipe {
        slug: "refs_prob_qpf_above_50p8mm",
        title: "REFS Probability Total QPF > 50.8 mm",
        filled: FIELD_REFS_PROB_QPF_ABOVE_50P8MM,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherProbability,
    },
    PlotRecipe {
        slug: "refs_prob_qpf_above_76p2mm",
        title: "REFS Probability Total QPF > 76.2 mm",
        filled: FIELD_REFS_PROB_QPF_ABOVE_76P2MM,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherProbability,
    },
    PlotRecipe {
        slug: "refs_prob_qpf_above_100mm",
        title: "REFS Probability Total QPF > 100 mm",
        filled: FIELD_REFS_PROB_QPF_ABOVE_100MM,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherProbability,
    },
    PlotRecipe {
        slug: "refs_prob_qpf_above_127mm",
        title: "REFS Probability Total QPF > 127 mm",
        filled: FIELD_REFS_PROB_QPF_ABOVE_127MM,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherProbability,
    },
    PlotRecipe {
        slug: "refs_prob_visibility_below_1600m",
        title: "REFS Probability Visibility < 1600 m",
        filled: FIELD_REFS_PROB_VISIBILITY_BELOW_1600M,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherProbability,
    },
    PlotRecipe {
        slug: "refs_prob_visibility_below_3200m",
        title: "REFS Probability Visibility < 3200 m",
        filled: FIELD_REFS_PROB_VISIBILITY_BELOW_3200M,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherProbability,
    },
    PlotRecipe {
        slug: "refs_prob_visibility_below_8049m",
        title: "REFS Probability Visibility < 8049 m",
        filled: FIELD_REFS_PROB_VISIBILITY_BELOW_8049M,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherProbability,
    },
    PlotRecipe {
        slug: "refs_prob_10m_wind_speed_above_15p4ms",
        title: "REFS Probability 10m Wind Speed > 15.4 m/s",
        filled: FIELD_REFS_PROB_10M_WIND_SPEED_ABOVE_15P4MS,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherProbability,
    },
    PlotRecipe {
        slug: "refs_prob_10m_wind_speed_above_20p6ms",
        title: "REFS Probability 10m Wind Speed > 20.6 m/s",
        filled: FIELD_REFS_PROB_10M_WIND_SPEED_ABOVE_20P6MS,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherProbability,
    },
    PlotRecipe {
        slug: "refs_prob_10m_wind_speed_above_25p72ms",
        title: "REFS Probability 10m Wind Speed > 25.72 m/s",
        filled: FIELD_REFS_PROB_10M_WIND_SPEED_ABOVE_25P72MS,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherProbability,
    },
    PlotRecipe {
        slug: "href_prob_uh_2to5km_above_25",
        title: "HREF Probability 2-5 km Updraft Helicity > 25",
        filled: FIELD_HREF_PROB_UH_ABOVE_25,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherProbability,
    },
    PlotRecipe {
        slug: "href_prob_uh_2to5km_above_75",
        title: "HREF Probability 2-5 km Updraft Helicity > 75",
        filled: FIELD_HREF_PROB_UH_ABOVE_75,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherProbability,
    },
    PlotRecipe {
        slug: "href_prob_uh_2to5km_above_150",
        title: "HREF Probability 2-5 km Updraft Helicity > 150",
        filled: FIELD_HREF_PROB_UH_ABOVE_150,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherProbability,
    },
    PlotRecipe {
        slug: "href_prob_2m_temperature_below_273p15k",
        title: "HREF Probability 2m Temperature < 273.15 K",
        filled: FIELD_HREF_PROB_2M_TEMP_BELOW_273P15K,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherProbability,
    },
    PlotRecipe {
        slug: "href_prob_2m_dewpoint_above_291p48k",
        title: "HREF Probability 2m Dewpoint > 291.48 K",
        filled: FIELD_HREF_PROB_2M_DEWPOINT_ABOVE_291P48K,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherProbability,
    },
    PlotRecipe {
        slug: "href_prob_2m_dewpoint_above_294p26k",
        title: "HREF Probability 2m Dewpoint > 294.26 K",
        filled: FIELD_HREF_PROB_2M_DEWPOINT_ABOVE_294P26K,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherProbability,
    },
    PlotRecipe {
        slug: "href_prob_pwat_above_25mm",
        title: "HREF Probability Precipitable Water > 25 mm",
        filled: FIELD_HREF_PROB_PWAT_ABOVE_25MM,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherProbability,
    },
    PlotRecipe {
        slug: "href_prob_pwat_above_37p5mm",
        title: "HREF Probability Precipitable Water > 37.5 mm",
        filled: FIELD_HREF_PROB_PWAT_ABOVE_37P5MM,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherProbability,
    },
    PlotRecipe {
        slug: "href_prob_pwat_above_50mm",
        title: "HREF Probability Precipitable Water > 50 mm",
        filled: FIELD_HREF_PROB_PWAT_ABOVE_50MM,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherProbability,
    },
    PlotRecipe {
        slug: "href_prob_visibility_below_1600m",
        title: "HREF Probability Visibility < 1600 m",
        filled: FIELD_HREF_PROB_VISIBILITY_BELOW_1600M,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherProbability,
    },
    PlotRecipe {
        slug: "href_prob_visibility_below_3200m",
        title: "HREF Probability Visibility < 3200 m",
        filled: FIELD_HREF_PROB_VISIBILITY_BELOW_3200M,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherProbability,
    },
    PlotRecipe {
        slug: "href_prob_visibility_below_6400m",
        title: "HREF Probability Visibility < 6400 m",
        filled: FIELD_HREF_PROB_VISIBILITY_BELOW_6400M,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherProbability,
    },
    PlotRecipe {
        slug: "href_prob_10m_wind_speed_above_15p4ms",
        title: "HREF Probability 10m Wind Speed > 15.4 m/s",
        filled: FIELD_HREF_PROB_10M_WIND_SPEED_ABOVE_15P4MS,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherProbability,
    },
    PlotRecipe {
        slug: "href_prob_10m_wind_speed_above_20p6ms",
        title: "HREF Probability 10m Wind Speed > 20.6 m/s",
        filled: FIELD_HREF_PROB_10M_WIND_SPEED_ABOVE_20P6MS,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherProbability,
    },
    PlotRecipe {
        slug: "href_prob_10m_wind_speed_above_25p72ms",
        title: "HREF Probability 10m Wind Speed > 25.72 m/s",
        filled: FIELD_HREF_PROB_10M_WIND_SPEED_ABOVE_25P72MS,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherProbability,
    },
    PlotRecipe {
        slug: "sref_prob_2m_temperature_below_273k",
        title: "SREF Probability 2m Temperature < 273 K",
        filled: FIELD_SREF_PROB_2M_TEMP_BELOW_FREEZING,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherProbability,
    },
    PlotRecipe {
        slug: "sref_prob_2m_temperature_above_298p8k",
        title: "SREF Probability 2m Temperature > 298.8 K",
        filled: FIELD_SREF_PROB_2M_TEMP_ABOVE_298P8K,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherProbability,
    },
    PlotRecipe {
        slug: "sref_prob_850mb_temperature_below_273k",
        title: "SREF Probability 850mb Temperature < 273 K",
        filled: FIELD_SREF_PROB_850MB_TEMP_BELOW_FREEZING,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherProbability,
    },
    PlotRecipe {
        slug: "sref_prob_visibility_below_1609m",
        title: "SREF Probability Visibility < 1609 m",
        filled: FIELD_SREF_PROB_VISIBILITY_BELOW_ONE_MILE,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherProbability,
    },
    PlotRecipe {
        slug: "sref_prob_visibility_below_402m",
        title: "SREF Probability Visibility < 402 m",
        filled: FIELD_SREF_PROB_VISIBILITY_BELOW_402M,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherProbability,
    },
    PlotRecipe {
        slug: "sref_prob_visibility_below_804m",
        title: "SREF Probability Visibility < 804 m",
        filled: FIELD_SREF_PROB_VISIBILITY_BELOW_804M,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherProbability,
    },
    PlotRecipe {
        slug: "sref_prob_visibility_below_3218m",
        title: "SREF Probability Visibility < 3218 m",
        filled: FIELD_SREF_PROB_VISIBILITY_BELOW_3218M,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherProbability,
    },
    PlotRecipe {
        slug: "sref_prob_visibility_below_4827m",
        title: "SREF Probability Visibility < 4827 m",
        filled: FIELD_SREF_PROB_VISIBILITY_BELOW_4827M,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherProbability,
    },
    PlotRecipe {
        slug: "sref_prob_visibility_below_8046m",
        title: "SREF Probability Visibility < 8046 m",
        filled: FIELD_SREF_PROB_VISIBILITY_BELOW_8046M,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherProbability,
    },
    PlotRecipe {
        slug: "sref_prob_visibility_below_9654m",
        title: "SREF Probability Visibility < 9654 m",
        filled: FIELD_SREF_PROB_VISIBILITY_BELOW_9654M,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherProbability,
    },
    PlotRecipe {
        slug: "sref_prob_10m_wind_speed_above_12p89ms",
        title: "SREF Probability 10m Wind Speed > 12.89 m/s",
        filled: FIELD_SREF_PROB_10M_WIND_SPEED_ABOVE_12P89MS,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherProbability,
    },
    PlotRecipe {
        slug: "sref_prob_10m_wind_speed_above_17p5ms",
        title: "SREF Probability 10m Wind Speed > 17.5 m/s",
        filled: FIELD_SREF_PROB_10M_WIND_SPEED_ABOVE_17P5MS,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherProbability,
    },
    PlotRecipe {
        slug: "sref_prob_10m_wind_speed_above_25p78ms",
        title: "SREF Probability 10m Wind Speed > 25.78 m/s",
        filled: FIELD_SREF_PROB_10M_WIND_SPEED_ABOVE_25P78MS,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherProbability,
    },
    PlotRecipe {
        slug: "2m_temperature_10m_winds",
        title: "2m AGL Temperature / 10m Winds",
        filled: FIELD_2M_TEMP,
        contours: Some(FIELD_MSLP),
        barbs_u: Some(FIELD_10M_U),
        barbs_v: Some(FIELD_10M_V),
        style: RenderStyle::WeatherTemperature,
    },
    PlotRecipe {
        slug: "gefs_avg_2m_temperature_10m_winds",
        title: "GEFS 2m AGL Temperature / 10m Winds Mean",
        filled: FIELD_GEFS_AVG_2M_TEMP,
        contours: Some(FIELD_GEFS_AVG_MSLP),
        barbs_u: Some(FIELD_GEFS_AVG_10M_U),
        barbs_v: Some(FIELD_GEFS_AVG_10M_V),
        style: RenderStyle::WeatherTemperature,
    },
    PlotRecipe {
        slug: "gefs_spr_2m_temperature_stddev",
        title: "GEFS 2m AGL Temperature Spread",
        filled: FIELD_GEFS_SPR_2M_TEMP_STDDEV,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherTemperature,
    },
    PlotRecipe {
        slug: "gefs_avg_2m_relative_humidity",
        title: "GEFS 2m AGL Relative Humidity Mean",
        filled: FIELD_GEFS_AVG_2M_RH,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherRh,
    },
    PlotRecipe {
        slug: "gefs_spr_2m_relative_humidity_stddev",
        title: "GEFS 2m AGL Relative Humidity Spread",
        filled: FIELD_GEFS_SPR_2M_RH_STDDEV,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherRh,
    },
    PlotRecipe {
        slug: "2m_dewpoint",
        title: "2m AGL Dewpoint",
        filled: FIELD_2M_DEWPOINT,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherDewpoint,
    },
    PlotRecipe {
        slug: "2m_dewpoint_10m_winds",
        title: "2m AGL Dewpoint / 10m Winds",
        filled: FIELD_2M_DEWPOINT,
        contours: Some(FIELD_MSLP),
        barbs_u: Some(FIELD_10M_U),
        barbs_v: Some(FIELD_10M_V),
        style: RenderStyle::WeatherDewpoint,
    },
    PlotRecipe {
        slug: "mslp_10m_winds",
        title: "MSLP / 10m Winds",
        filled: FIELD_MSLP,
        contours: Some(FIELD_MSLP),
        barbs_u: Some(FIELD_10M_U),
        barbs_v: Some(FIELD_10M_V),
        style: RenderStyle::WeatherWinds,
    },
    PlotRecipe {
        slug: "gefs_avg_mslp_10m_winds",
        title: "GEFS MSLP / 10m Winds Mean",
        filled: FIELD_GEFS_AVG_MSLP,
        contours: Some(FIELD_GEFS_AVG_MSLP),
        barbs_u: Some(FIELD_GEFS_AVG_10M_U),
        barbs_v: Some(FIELD_GEFS_AVG_10M_V),
        style: RenderStyle::WeatherWinds,
    },
    PlotRecipe {
        slug: "gefs_spr_mslp_stddev",
        title: "GEFS MSLP Spread",
        filled: FIELD_GEFS_SPR_MSLP_STDDEV,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherPressure,
    },
    PlotRecipe {
        // Static per domain: this is the ground the forecast is standing
        // on, and it is what makes a wind or precipitation frame legible
        // -- a viewer cannot see WHY the wind is where it is without it.
        // The batch runner renders it once per domain (see
        // `direct_recipe_is_time_invariant`).
        slug: "terrain_height",
        title: "Terrain Height",
        filled: FIELD_TERRAIN_HEIGHT,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherTerrain,
    },
    PlotRecipe {
        // The standalone surface wind chart: 10 m wind SPEED as the fill,
        // 10 m wind DIRECTION as the barbs, and nothing else on the map.
        //
        // Until this entry existed, wind was reachable only as a barb
        // overlay on somebody else's field (`mslp_10m_winds`,
        // `2m_temperature_10m_winds`, `2m_dewpoint_10m_winds`,
        // `2m_relative_humidity_10m_winds`, `theta_e_2m_10m_winds`) or as
        // a multi-frame window statistic (`10m_wind_1h_max`,
        // `10m_wind_run_max`), which needs neighbouring hours and
        // therefore cannot render from a single frame at all.  A person
        // who wants "the wind right now" had neither.
        //
        // It depends on ONE frame: three planes from the hour being
        // rendered, no accumulator, no window, no neighbour.
        slug: "10m_wind_speed_and_direction",
        title: "10m AGL Wind Speed and Direction",
        filled: FIELD_10M_WIND_SPEED,
        contours: None,
        barbs_u: Some(FIELD_10M_U),
        barbs_v: Some(FIELD_10M_V),
        style: RenderStyle::WeatherWinds,
    },
    PlotRecipe {
        slug: "10m_wind_gusts",
        title: "10m AGL Wind Gusts",
        filled: FIELD_10M_WIND_GUST,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherWindGust,
    },
    PlotRecipe {
        slug: "ivt_column", title: "Integrated Vapor Transport (model column)", filled: FIELD_IVT,
        contours: None, barbs_u: None, barbs_v: None, style: RenderStyle::WeatherIvt,
    },
    PlotRecipe {
        slug: "ivt_column_eastward", title: "Eastward Integrated Vapor Transport (model column)", filled: FIELD_IVT_U,
        contours: None, barbs_u: None, barbs_v: None, style: RenderStyle::WeatherRh,
    },
    PlotRecipe {
        slug: "ivt_column_northward", title: "Northward Integrated Vapor Transport (model column)", filled: FIELD_IVT_V,
        contours: None, barbs_u: None, barbs_v: None, style: RenderStyle::WeatherRh,
    },
    PlotRecipe {
        slug: "ivt_vectors", title: "Integrated Vapor Transport / Direction (model column)", filled: FIELD_IVT,
        contours: None, barbs_u: Some(FIELD_IVT_U), barbs_v: Some(FIELD_IVT_V), style: RenderStyle::WeatherIvt,
    },
    PlotRecipe {
        slug: "925mb_height_winds", title: "925mb Height / Winds", filled: FIELD_925_HEIGHT,
        contours: Some(FIELD_925_HEIGHT), barbs_u: Some(FIELD_925_U), barbs_v: Some(FIELD_925_V),
        style: RenderStyle::WeatherHeight,
    },
    PlotRecipe {
        slug: "precipitable_water",
        title: "Precipitable Water",
        filled: FIELD_PWAT,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherPrecipitableWater,
    },
    PlotRecipe {
        slug: "gefs_avg_precipitable_water",
        title: "GEFS Precipitable Water Mean",
        filled: FIELD_GEFS_AVG_PWAT,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherPrecipitableWater,
    },
    PlotRecipe {
        slug: "gefs_spr_precipitable_water_stddev",
        title: "GEFS Precipitable Water Spread",
        filled: FIELD_GEFS_SPR_PWAT_STDDEV,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherPrecipitableWater,
    },
    PlotRecipe {
        slug: "cloud_cover",
        title: "Cloud Cover",
        filled: FIELD_TOTAL_CLOUD_COVER,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherCloudCover,
    },
    PlotRecipe {
        slug: "gefs_avg_cloud_cover",
        title: "GEFS Cloud Cover Mean",
        filled: FIELD_GEFS_AVG_TOTAL_CLOUD_COVER,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherCloudCover,
    },
    PlotRecipe {
        slug: "gefs_spr_cloud_cover_stddev",
        title: "GEFS Cloud Cover Spread",
        filled: FIELD_GEFS_SPR_TOTAL_CLOUD_COVER_STDDEV,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherCloudCover,
    },
    PlotRecipe {
        slug: "low_cloud_cover",
        title: "Low Cloud Cover",
        filled: FIELD_LOW_CLOUD_COVER,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherCloudCover,
    },
    PlotRecipe {
        slug: "middle_cloud_cover",
        title: "Middle Cloud Cover",
        filled: FIELD_MIDDLE_CLOUD_COVER,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherCloudCover,
    },
    PlotRecipe {
        slug: "high_cloud_cover",
        title: "High Cloud Cover",
        filled: FIELD_HIGH_CLOUD_COVER,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherCloudCover,
    },
    PlotRecipe {
        slug: "cloud_cover_levels",
        title: "Cloud Cover, Levels",
        filled: FIELD_CLOUD_COVER_LEVELS,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherCloudCover,
    },
    PlotRecipe {
        slug: "fog_flag",
        title: "Low Visibility (< 1000 m)",
        filled: FIELD_FOG_FLAG,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherCategorical,
    },
    PlotRecipe {
        slug: "visibility",
        title: "Visibility",
        filled: FIELD_VISIBILITY,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherVisibility,
    },
    PlotRecipe {
        slug: "simulated_ir_satellite",
        title: "Simulated IR Satellite",
        filled: FIELD_SIMULATED_IR,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherSatellite,
    },
    // Column planes the wrfout import derives from its native levels
    // (rw-wrfbatch wrf_column_planes.rs); no GRIB source in the catalog
    // carries them, so a fetch plan for one is a table row still to write.
    PlotRecipe {
        slug: "isotherm_height_0c",
        title: "0 C Isotherm Height",
        filled: FIELD_ISOTHERM_HEIGHT_0C,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherIsothermHeight,
    },
    PlotRecipe {
        slug: "isotherm_height_minus10c",
        title: "-10 C Isotherm Height",
        filled: FIELD_ISOTHERM_HEIGHT_MINUS10C,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherIsothermHeight,
    },
    PlotRecipe {
        slug: "isotherm_height_minus20c",
        title: "-20 C Isotherm Height",
        filled: FIELD_ISOTHERM_HEIGHT_MINUS20C,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherIsothermHeight,
    },
    PlotRecipe {
        slug: "ice_water_path", title: "Ice Water Path", filled: FIELD_ICE_WATER_PATH,
        contours: None, barbs_u: None, barbs_v: None,
        style: RenderStyle::WeatherSupercooledWaterPath,
    },
    PlotRecipe {
        slug: "supercooled_water_path",
        title: "Supercooled Liquid Water Path",
        filled: FIELD_SUPERCOOLED_WATER_PATH,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherSupercooledWaterPath,
    },
    PlotRecipe {
        slug: "supercooled_water_path_0_3km",
        title: "Supercooled Liquid Water Path, 0 to 3 km AGL",
        filled: FIELD_SUPERCOOLED_WATER_PATH_0_3KM,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherSupercooledWaterPath,
    },
    PlotRecipe {
        slug: "supercooled_water_path_3_6km",
        title: "Supercooled Liquid Water Path, 3 to 6 km AGL",
        filled: FIELD_SUPERCOOLED_WATER_PATH_3_6KM,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherSupercooledWaterPath,
    },
    PlotRecipe {
        slug: "cloud_fraction_column_max",
        title: "Radiation Cloud Fraction, Column Maximum",
        filled: FIELD_CLOUD_FRACTION_COLUMN_MAX,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherCloudCover,
    },
    PlotRecipe {
        slug: "cloud_water_column_max",
        title: "Cloud Water, Column Maximum",
        filled: FIELD_CLOUD_WATER_COLUMN_MAX,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherHydrometeorMixingRatio,
    },
    PlotRecipe {
        slug: "rain_water_column_max",
        title: "Rain Water, Column Maximum",
        filled: FIELD_RAIN_WATER_COLUMN_MAX,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherHydrometeorMixingRatio,
    },
    PlotRecipe {
        slug: "cloud_ice_column_max",
        title: "Cloud Ice, Column Maximum",
        filled: FIELD_CLOUD_ICE_COLUMN_MAX,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherHydrometeorMixingRatio,
    },
    PlotRecipe {
        slug: "snow_column_max",
        title: "Snow, Column Maximum",
        filled: FIELD_SNOW_COLUMN_MAX,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherHydrometeorMixingRatio,
    },
    PlotRecipe {
        slug: "snowfall_window", title: "Model snowfall, window total",
        filled: FIELD_MODEL_SNOWFALL, contours: None, barbs_u: None, barbs_v: None,
        style: RenderStyle::WeatherSnowfall,
    },
    PlotRecipe {
        slug: "snow_10to1_window", title: "10:1 snowfall, window total",
        filled: FIELD_SNOW_10TO1, contours: None, barbs_u: None, barbs_v: None,
        style: RenderStyle::WeatherSnowfall,
    },
    PlotRecipe {
        slug: "snow_kuchera_window", title: "Kuchera snowfall, window total",
        filled: FIELD_SNOW_KUCHERA, contours: None, barbs_u: None, barbs_v: None,
        style: RenderStyle::WeatherSnowfall,
    },
    PlotRecipe { slug: "snowfall_liquid_equivalent", title: "Snowfall liquid equivalent, window total", filled: FIELD_SNOWFALL_LIQUID_EQUIVALENT, contours: None, barbs_u: None, barbs_v: None, style: RenderStyle::WeatherQpf },
    PlotRecipe { slug: "snow_depth_total", title: "LSM snow depth", filled: FIELD_SNOW_DEPTH_TOTAL, contours: None, barbs_u: None, barbs_v: None, style: RenderStyle::WeatherSnowfall },
    PlotRecipe { slug: "snow_depth_change", title: "LSM snow depth change", filled: FIELD_SNOW_DEPTH_CHANGE, contours: None, barbs_u: None, barbs_v: None, style: RenderStyle::WeatherSnowDepthChange },
    PlotRecipe {
        slug: "hail_max_surface", title: "PSD maximum hail diameter at lowest model level", filled: FIELD_HAILMAXSURFACE,
        contours: None, barbs_u: None, barbs_v: None, style: RenderStyle::WeatherHail,
    },
    PlotRecipe {
        slug: "hail_max_column", title: "PSD maximum hail diameter over column", filled: FIELD_HAILMAXCOLUMN,
        contours: None, barbs_u: None, barbs_v: None, style: RenderStyle::WeatherHail,
    },
    PlotRecipe {
        slug: "hail_max_surface_run", title: "Run maximum PSD surface hail diameter", filled: FIELD_HAILMAXSURFACERUN,
        contours: None, barbs_u: None, barbs_v: None, style: RenderStyle::WeatherHail,
    },
    PlotRecipe {
        slug: "hail_max_column_run", title: "Run maximum PSD column hail diameter", filled: FIELD_HAILMAXCOLUMNRUN,
        contours: None, barbs_u: None, barbs_v: None, style: RenderStyle::WeatherHail,
    },
    PlotRecipe {
        slug: "hail_max_hailcast", title: "HAILCAST maximum surviving ice diameter", filled: FIELD_HAILCASTMAX,
        contours: None, barbs_u: None, barbs_v: None, style: RenderStyle::WeatherHail,
    },
    PlotRecipe {
        slug: "hail_max_hailcast_mean", title: "HAILCAST mean surviving ice diameter", filled: FIELD_HAILCASTMEAN,
        contours: None, barbs_u: None, barbs_v: None, style: RenderStyle::WeatherHail,
    },
    PlotRecipe {
        slug: "hail_max_hailcast_std", title: "HAILCAST diameter standard deviation", filled: FIELD_HAILCASTSTD,
        contours: None, barbs_u: None, barbs_v: None, style: RenderStyle::WeatherHail,
    },
    PlotRecipe {
        slug: "hail_max_hailcast_run", title: "Run maximum HAILCAST diameter", filled: FIELD_HAILCASTRUN,
        contours: None, barbs_u: None, barbs_v: None, style: RenderStyle::WeatherHail,
    },
    PlotRecipe {
        slug: "graupel_column_max",
        title: "Graupel, Column Maximum",
        filled: FIELD_GRAUPEL_COLUMN_MAX,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherHydrometeorMixingRatio,
    },
    PlotRecipe {
        slug: "lightning_flash_density",
        title: "Lightning Flash Density",
        filled: FIELD_LIGHTNING_FLASH_DENSITY,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherLightning,
    },
    PlotRecipe {
        slug: "total_qpf",
        title: "Total QPF",
        filled: FIELD_TOTAL_QPF,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherQpf,
    },
    PlotRecipe {
        slug: "gefs_avg_6h_qpf",
        title: "GEFS 6h QPF Mean",
        filled: FIELD_GEFS_AVG_6H_QPF,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherQpf,
    },
    PlotRecipe {
        slug: "gefs_spr_6h_qpf_stddev",
        title: "GEFS 6h QPF Spread",
        filled: FIELD_GEFS_SPR_6H_QPF_STDDEV,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherQpf,
    },
    PlotRecipe {
        slug: "1h_qpf",
        title: "1h QPF",
        filled: FIELD_ONE_HOUR_QPF,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherQpf,
    },
    PlotRecipe {
        slug: "probability_of_precipitation",
        title: "Probability of Precipitation",
        filled: FIELD_POP,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherRh,
    },
    PlotRecipe {
        slug: "categorical_rain",
        title: "Categorical Rain",
        filled: FIELD_CATEGORICAL_RAIN,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherCategorical,
    },
    PlotRecipe {
        slug: "categorical_freezing_rain",
        title: "Categorical Freezing Rain",
        filled: FIELD_CATEGORICAL_FREEZING_RAIN,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherCategorical,
    },
    PlotRecipe {
        slug: "categorical_ice_pellets",
        title: "Categorical Ice Pellets",
        filled: FIELD_CATEGORICAL_ICE_PELLETS,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherCategorical,
    },
    PlotRecipe {
        slug: "categorical_snow",
        title: "Categorical Snow",
        filled: FIELD_CATEGORICAL_SNOW,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherCategorical,
    },
    PlotRecipe {
        slug: "precipitation_type",
        title: "Precipitation Type",
        filled: FIELD_PRECIPITATION_TYPE,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherCategorical,
    },
    PlotRecipe {
        slug: "2m_theta_e_10m_winds",
        title: "2m AGL Theta-e / 10m Winds",
        filled: FIELD_2M_THETA_E,
        contours: None,
        barbs_u: Some(FIELD_10M_U),
        barbs_v: Some(FIELD_10M_V),
        style: RenderStyle::WeatherTemperature,
    },
    PlotRecipe {
        slug: "2m_heat_index",
        title: "2m AGL Heat Index",
        filled: FIELD_2M_HEAT_INDEX,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherTemperature,
    },
    PlotRecipe {
        slug: "2m_wind_chill",
        title: "2m AGL Wind Chill",
        filled: FIELD_2M_WIND_CHILL,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherTemperature,
    },
    PlotRecipe {
        slug: "700mb_dewpoint_height_winds",
        title: "700mb Dewpoint / Height / Winds",
        filled: FIELD_700_DEWPOINT,
        contours: Some(FIELD_700_HEIGHT),
        barbs_u: Some(FIELD_700_U),
        barbs_v: Some(FIELD_700_V),
        style: RenderStyle::WeatherDewpoint,
    },
    PlotRecipe {
        slug: "850mb_dewpoint_height_winds",
        title: "850mb Dewpoint / Height / Winds",
        filled: FIELD_850_DEWPOINT,
        contours: Some(FIELD_850_HEIGHT),
        barbs_u: Some(FIELD_850_U),
        barbs_v: Some(FIELD_850_V),
        style: RenderStyle::WeatherDewpoint,
    },
    PlotRecipe {
        slug: "200mb_rh_height_winds",
        title: "200mb RH / Height / Winds",
        filled: FIELD_200_RH,
        contours: Some(FIELD_200_HEIGHT),
        barbs_u: Some(FIELD_200_U),
        barbs_v: Some(FIELD_200_V),
        style: RenderStyle::WeatherRh,
    },
    PlotRecipe {
        slug: "300mb_rh_height_winds",
        title: "300mb RH / Height / Winds",
        filled: FIELD_300_RH,
        contours: Some(FIELD_300_HEIGHT),
        barbs_u: Some(FIELD_300_U),
        barbs_v: Some(FIELD_300_V),
        style: RenderStyle::WeatherRh,
    },
    PlotRecipe {
        slug: "500mb_rh_height_winds",
        title: "500mb RH / Height / Winds",
        filled: FIELD_500_RH,
        contours: Some(FIELD_500_HEIGHT),
        barbs_u: Some(FIELD_500_U),
        barbs_v: Some(FIELD_500_V),
        style: RenderStyle::WeatherRh,
    },
    PlotRecipe {
        slug: "gefs_avg_500mb_rh_height_winds",
        title: "GEFS 500mb RH / Height / Winds Mean",
        filled: FIELD_GEFS_AVG_500_RH,
        contours: Some(FIELD_GEFS_AVG_500_HEIGHT),
        barbs_u: Some(FIELD_GEFS_AVG_500_U),
        barbs_v: Some(FIELD_GEFS_AVG_500_V),
        style: RenderStyle::WeatherRh,
    },
    PlotRecipe {
        slug: "gefs_spr_500mb_rh_stddev",
        title: "GEFS 500mb RH Spread",
        filled: FIELD_GEFS_SPR_500_RH_STDDEV,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherRh,
    },
    PlotRecipe {
        slug: "700mb_rh_height_winds",
        title: "700mb RH / Height / Winds",
        filled: FIELD_700_RH,
        contours: Some(FIELD_700_HEIGHT),
        barbs_u: Some(FIELD_700_U),
        barbs_v: Some(FIELD_700_V),
        style: RenderStyle::WeatherRh,
    },
    PlotRecipe {
        slug: "850mb_rh_height_winds",
        title: "850mb RH / Height / Winds",
        filled: FIELD_850_RH,
        contours: Some(FIELD_850_HEIGHT),
        barbs_u: Some(FIELD_850_U),
        barbs_v: Some(FIELD_850_V),
        style: RenderStyle::WeatherRh,
    },
    PlotRecipe {
        slug: "200mb_absolute_vorticity_height_winds",
        title: "200mb Absolute Vorticity / Height / Winds",
        filled: FIELD_200_ABSOLUTE_VORTICITY,
        contours: Some(FIELD_200_HEIGHT),
        barbs_u: Some(FIELD_200_U),
        barbs_v: Some(FIELD_200_V),
        style: RenderStyle::WeatherVorticity,
    },
    PlotRecipe {
        slug: "300mb_absolute_vorticity_height_winds",
        title: "300mb Absolute Vorticity / Height / Winds",
        filled: FIELD_300_ABSOLUTE_VORTICITY,
        contours: Some(FIELD_300_HEIGHT),
        barbs_u: Some(FIELD_300_U),
        barbs_v: Some(FIELD_300_V),
        style: RenderStyle::WeatherVorticity,
    },
    PlotRecipe {
        slug: "500mb_absolute_vorticity_height_winds",
        title: "500mb Absolute Vorticity / Height / Winds",
        filled: FIELD_500_ABSOLUTE_VORTICITY,
        contours: Some(FIELD_500_HEIGHT),
        barbs_u: Some(FIELD_500_U),
        barbs_v: Some(FIELD_500_V),
        style: RenderStyle::WeatherVorticity,
    },
    PlotRecipe {
        slug: "700mb_absolute_vorticity_height_winds",
        title: "700mb Absolute Vorticity / Height / Winds",
        filled: FIELD_700_ABSOLUTE_VORTICITY,
        contours: Some(FIELD_700_HEIGHT),
        barbs_u: Some(FIELD_700_U),
        barbs_v: Some(FIELD_700_V),
        style: RenderStyle::WeatherVorticity,
    },
    PlotRecipe {
        slug: "850mb_absolute_vorticity_height_winds",
        title: "850mb Absolute Vorticity / Height / Winds",
        filled: FIELD_850_ABSOLUTE_VORTICITY,
        contours: Some(FIELD_850_HEIGHT),
        barbs_u: Some(FIELD_850_U),
        barbs_v: Some(FIELD_850_V),
        style: RenderStyle::WeatherVorticity,
    },
    PlotRecipe {
        slug: "1km_vertical_velocity",
        title: "1 km AGL Vertical Velocity (upward positive)",
        filled: FIELD_1KM_VERTICAL_VELOCITY,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherVerticalVelocity,
    },
    PlotRecipe {
        slug: "3km_vertical_velocity",
        title: "3 km AGL Vertical Velocity (upward positive)",
        filled: FIELD_3KM_VERTICAL_VELOCITY,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherVerticalVelocity,
    },
    PlotRecipe {
        slug: "1km_reflectivity",
        title: "1km AGL Reflectivity",
        filled: FIELD_1KM_REFLECTIVITY,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherRadarReflectivity,
    },
    PlotRecipe {
        slug: "composite_reflectivity",
        title: "Composite Reflectivity",
        filled: FIELD_COMPOSITE_REFLECTIVITY,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherReflectivity,
    },
    PlotRecipe {
        slug: "composite_reflectivity_uh",
        title: "Composite Reflectivity / UH",
        filled: FIELD_COMPOSITE_REFLECTIVITY,
        contours: Some(FIELD_UH),
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherReflectivity,
    },
    PlotRecipe {
        slug: "uh_2to5km",
        title: "Updraft Helicity 2-5 km",
        filled: FIELD_UH,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherUh,
    },
    PlotRecipe {
        slug: "uh_0to1km",
        title: "Updraft Helicity 0-1 km",
        filled: FIELD_UH_0TO1KM,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherUh,
    },
    PlotRecipe {
        slug: "uh_0to2km",
        title: "Updraft Helicity 0-2 km",
        filled: FIELD_UH_0TO2KM,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherUh,
    },
    PlotRecipe {
        slug: "uh_0to3km",
        title: "Updraft Helicity 0-3 km",
        filled: FIELD_UH_0TO3KM,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherUh,
    },
    PlotRecipe {
        slug: "composite_reflectivity_uh0to1km",
        title: "Composite Reflectivity / UH 0-1 km",
        filled: FIELD_COMPOSITE_REFLECTIVITY,
        contours: Some(FIELD_UH_0TO1KM),
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherReflectivity,
    },
    PlotRecipe {
        slug: "composite_reflectivity_uh0to2km",
        title: "Composite Reflectivity / UH 0-2 km",
        filled: FIELD_COMPOSITE_REFLECTIVITY,
        contours: Some(FIELD_UH_0TO2KM),
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherReflectivity,
    },
    PlotRecipe {
        slug: "composite_reflectivity_uh0to3km",
        title: "Composite Reflectivity / UH 0-3 km",
        filled: FIELD_COMPOSITE_REFLECTIVITY,
        contours: Some(FIELD_UH_0TO3KM),
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherReflectivity,
    },
    PlotRecipe {
        slug: "1km_reflectivity_uh2to5km",
        title: "1km AGL Reflectivity / UH 2-5 km",
        filled: FIELD_1KM_REFLECTIVITY,
        contours: Some(FIELD_UH),
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherRadarReflectivity,
    },
    PlotRecipe {
        slug: "1km_reflectivity_uh0to1km",
        title: "1km AGL Reflectivity / UH 0-1 km",
        filled: FIELD_1KM_REFLECTIVITY,
        contours: Some(FIELD_UH_0TO1KM),
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherRadarReflectivity,
    },
    PlotRecipe {
        slug: "1km_reflectivity_uh0to2km",
        title: "1km AGL Reflectivity / UH 0-2 km",
        filled: FIELD_1KM_REFLECTIVITY,
        contours: Some(FIELD_UH_0TO2KM),
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherRadarReflectivity,
    },
    PlotRecipe {
        slug: "1km_reflectivity_uh0to3km",
        title: "1km AGL Reflectivity / UH 0-3 km",
        filled: FIELD_1KM_REFLECTIVITY,
        contours: Some(FIELD_UH_0TO3KM),
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherRadarReflectivity,
    },
    PlotRecipe {
        slug: "smoke_pm25_native",
        title: "PM2.5 Smoke",
        filled: FIELD_SMOKE_MASS_DENSITY_8M,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::AirQualityPm25,
    },
    PlotRecipe {
        slug: "smoke_column",
        title: "Column-Integrated Smoke",
        filled: FIELD_COLUMN_INTEGRATED_SMOKE,
        contours: None,
        barbs_u: None,
        barbs_v: None,
        style: RenderStyle::WeatherTemperature,
    },
    PlotRecipe { slug: "ivt", title: "IVT 1000-300 hPa and Transport Direction", filled: FIELD_IVTMAGNITUDE, contours: Some(FIELD_IVTMAGNITUDE), barbs_u: Some(FIELD_IVTEASTWARD), barbs_v: Some(FIELD_IVTNORTHWARD), style: RenderStyle::WeatherRh },

    PlotRecipe { slug: "ar_outline", title: "AR Threshold: IVT >= 250 kg m-1 s-1", filled: FIELD_IVTMAGNITUDE, contours: Some(FIELD_IVTMAGNITUDE), barbs_u: None, barbs_v: None, style: RenderStyle::WeatherRh },

    PlotRecipe { slug: "iwv", title: "Integrated Water Vapour Surface to Model Top", filled: FIELD_INTEGRATEDWATERVAPOUR, contours: None, barbs_u: None, barbs_v: None, style: RenderStyle::WeatherRh },

    PlotRecipe { slug: "freezing_level", title: "Lowest 0 C Isotherm Height MSL", filled: FIELD_ISOTHERM_HEIGHT_0C, contours: None, barbs_u: None, barbs_v: None, style: RenderStyle::WeatherRh },

    PlotRecipe { slug: "ivt_full_column", title: "Water Vapour Flux Surface to Model Top", filled: FIELD_FULLCOLUMNIVTMAGNITUDE, contours: None, barbs_u: None, barbs_v: None, style: RenderStyle::WeatherRh },
    PlotRecipe { slug: "ivt_eastward_1000_300", title: "Eastward IVT 1000-300 hPa", filled: FIELD_IVTEASTWARD, contours: None, barbs_u: None, barbs_v: None, style: RenderStyle::WeatherRh },
    PlotRecipe { slug: "ivt_northward_1000_300", title: "Northward IVT 1000-300 hPa", filled: FIELD_IVTNORTHWARD, contours: None, barbs_u: None, barbs_v: None, style: RenderStyle::WeatherRh },
    PlotRecipe { slug: "ivt_eastward_full_column", title: "Eastward Water Vapour Flux Surface to Model Top", filled: FIELD_FULLCOLUMNIVTEASTWARD, contours: None, barbs_u: None, barbs_v: None, style: RenderStyle::WeatherRh },
    PlotRecipe { slug: "ivt_northward_full_column", title: "Northward Water Vapour Flux Surface to Model Top", filled: FIELD_FULLCOLUMNIVTNORTHWARD, contours: None, barbs_u: None, barbs_v: None, style: RenderStyle::WeatherRh },

];

pub fn built_in_models() -> &'static [ModelSummary] {
    MODELS
}

/// The models rusty-weather exposes to users. The wider registry returned by
/// [`built_in_models`] remains linked (`ModelId` match arms thread through
/// rustwx-products), but every user-facing enumeration must go through this
/// list.
pub fn supported_models() -> [ModelId; 16] {
    [
        ModelId::Hrrr,
        ModelId::HrrrAk,
        ModelId::Rap,
        ModelId::Gfs,
        ModelId::Gdas,
        ModelId::Gefs,
        ModelId::Aigfs,
        ModelId::Aigefs,
        ModelId::Hgefs,
        ModelId::EcmwfOpenData,
        ModelId::Aifs,
        ModelId::Nam,
        ModelId::Rrfs,
        ModelId::RrfsA,
        ModelId::Refs,
        ModelId::Nbm,
    ]
}

pub fn built_in_plot_recipes() -> &'static [PlotRecipe] {
    PLOT_RECIPES
}

pub fn plot_recipe(slug: &str) -> Option<&'static PlotRecipe> {
    let wanted = canonical_recipe_token(slug);
    PLOT_RECIPES
        .iter()
        .find(|recipe| normalize_token(recipe.slug) == wanted)
}

pub fn plot_recipe_fetch_plan(
    slug: &str,
    model: ModelId,
) -> Result<PlotRecipeFetchPlan, ModelError> {
    let recipe = plot_recipe(slug).ok_or_else(|| ModelError::UnknownPlotRecipe {
        slug: slug.to_string(),
    })?;
    plot_recipe_fetch_plan_for(recipe, model)
}

/// Plan a recipe whose canonical fields are supplied by a local store or
/// caller. Download capability cannot reject an already available plane.
/// The host must resolve every store requirement before rendering.
pub fn plot_recipe_store_plan(slug: &str, model: ModelId) -> Result<PlotRecipeFetchPlan, ModelError> {
    let recipe = plot_recipe(slug).ok_or_else(|| ModelError::UnknownPlotRecipe { slug: slug.into() })?;
    let fields = collect_recipe_fields(recipe);
    let (product, fetch_policy) = plot_recipe_fetch_defaults(model, &fields);
    Ok(PlotRecipeFetchPlan { recipe_slug: recipe.slug, model, product, fetch_policy,
        fetch_mode: fetch_policy.fetch_mode(), fields })
}

pub fn selector_fetch_plan(
    model: ModelId,
    selector: FieldSelector,
) -> Result<PlotRecipeFetchPlan, ModelError> {
    let field = PLOT_RECIPES.iter().flat_map(collect_recipe_fields)
        .find(|field| field.selector == Some(selector))
        .ok_or_else(|| ModelError::UnsupportedPlotRecipeModel {
            recipe: "selected_native_fields", model,
            reason: format!("no registered native field for selector '{selector}'"),
        })?;
    // A source-prefixed recipe key is not a canonical selector identity. The
    // statistical selector carries its own product and is checked against the
    // model's typed field contract; its name cannot redirect it to that recipe's
    // original model. Default/native recipes retain their extra field guards.
    let blocker = if selector.product.is_default() {
        plot_recipe_field_blocker(field, model).map(|blocker| blocker.reason)
    } else if !selector_supported_for_model(selector, model) {
        Some(unsupported_selector_reason(selector, model))
    } else { None };
    if let Some(reason) = blocker {
        return Err(ModelError::UnsupportedPlotRecipeModel {
            recipe: "selected_native_fields", model, reason,
        });
    }
    let fields = vec![field];
    let (product, fetch_policy) = plot_recipe_fetch_defaults(model, &fields);
    Ok(PlotRecipeFetchPlan { recipe_slug: "selected_native_fields", model,
        product, fetch_policy, fetch_mode: fetch_policy.fetch_mode(), fields })
}

pub fn plot_recipe_fetch_blockers(
    slug: &str,
    model: ModelId,
) -> Result<Vec<PlotRecipeBlocker>, ModelError> {
    let recipe = plot_recipe(slug).ok_or_else(|| ModelError::UnknownPlotRecipe {
        slug: slug.to_string(),
    })?;
    Ok(plot_recipe_fetch_blockers_for(recipe, model))
}

/// One field a plot recipe needs, model-identity-free (gpuwm vendor
/// divergence). `selector` is the canonical store address when one
/// exists; a `None` means no canonical FieldSelector binding exists
/// yet, so no store can prove the field either way.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlotRecipeStoreRequirement {
    pub field_key: &'static str,
    pub field_label: &'static str,
    pub selector: Option<FieldSelector>,
}

/// The store-availability contract for local (wrfout/GRIB-import)
/// stores: what a recipe NEEDS, independent of any model identity.
/// Store-backed listing and rendering decide availability by resolving
/// these selectors against the stored fields -- never by consulting the
/// per-model fetch tables, which describe what an operational source's
/// FILES carry (an input fact for downloading, not a render gate).
/// gpuwm vendor divergence; see tools/rustwx/VENDOR.md.
pub fn plot_recipe_store_requirements(
    slug: &str,
) -> Result<Vec<PlotRecipeStoreRequirement>, ModelError> {
    let recipe = plot_recipe(slug).ok_or_else(|| ModelError::UnknownPlotRecipe {
        slug: slug.to_string(),
    })?;
    Ok(collect_recipe_fields(recipe)
        .into_iter()
        .map(|field| PlotRecipeStoreRequirement {
            field_key: field.key,
            field_label: field.label,
            selector: field.selector,
        })
        .collect())
}

/// FETCH capability table: which operational source's downloadable
/// FILES carry a field (e.g. only HRRR's GRIB2 output contains smoke
/// mass density or simulated IR brightness temperature).  This is an
/// input-availability fact about sources, not a render gate: no
/// store-backed listing or render decision consults it -- those route
/// through [`plot_recipe_store_requirements`] and the stored fields
/// (gpuwm vendor divergence 11, tools/rustwx/VENDOR.md).
pub fn selector_supported_for_model(selector: FieldSelector, model: ModelId) -> bool {
    if !selector.product.is_default() {
        match (model, selector.product) {
            (ModelId::Nbm, _) => {}
            (ModelId::Sref, FieldProduct::Probability(_)) => {}
            (
                ModelId::Gefs,
                FieldProduct::EnsembleMean
                | FieldProduct::EnsembleStandardDeviation
                | FieldProduct::EnsembleSpread,
            ) => {}
            (
                ModelId::Aigefs | ModelId::Hgefs,
                FieldProduct::EnsembleMean | FieldProduct::EnsembleStandardDeviation | FieldProduct::EnsembleSpread,
            ) => {}
            (
                ModelId::Href,
                FieldProduct::EnsembleMean
                | FieldProduct::EnsembleSpread
                | FieldProduct::Probability(_),
            ) => {}
            (
                ModelId::Refs,
                FieldProduct::EnsembleMean
                | FieldProduct::EnsembleSpread
                | FieldProduct::Probability(_),
            ) => {}
            _ => return false,
        }
    }
    if matches!(model, ModelId::Href) && selector.product.is_default() {
        return false;
    }
    if matches!(
        (model, selector.vertical),
        (
            ModelId::Rtma | ModelId::Urma | ModelId::Nbm,
            VerticalSelector::IsobaricHpa(_)
        )
    ) {
        return false;
    }
    match (selector.field, selector.vertical) {
        (
            CanonicalField::GeopotentialHeight
            | CanonicalField::Temperature
            | CanonicalField::RelativeHumidity
            | CanonicalField::AbsoluteVorticity
            | CanonicalField::UWind
            | CanonicalField::VWind,
            VerticalSelector::IsobaricHpa(level_hpa),
        ) if is_supported_upper_air_level(level_hpa) => true,
        (CanonicalField::Dewpoint, VerticalSelector::IsobaricHpa(level_hpa))
            if matches!(level_hpa, 700 | 850) =>
        {
            true
        }
        (
            CanonicalField::Temperature
            | CanonicalField::Dewpoint
            | CanonicalField::RelativeHumidity,
            VerticalSelector::HeightAboveGroundMeters(2),
        ) => true,
        (
            CanonicalField::UWind | CanonicalField::VWind,
            VerticalSelector::HeightAboveGroundMeters(10),
        ) => true,
        (CanonicalField::WindSpeed, VerticalSelector::HeightAboveGroundMeters(10)) => true,
        (
            CanonicalField::Pressure | CanonicalField::SmokeMassDensity,
            VerticalSelector::HybridLevel(level),
        ) => {
            matches!(model, ModelId::Hrrr | ModelId::HrrrAk)
                && is_supported_hrrr_smoke_hybrid_level(level)
        }
        (CanonicalField::WindGust, VerticalSelector::HeightAboveGroundMeters(10)) => true,
        (CanonicalField::SmokeMassDensity, VerticalSelector::HeightAboveGroundMeters(8)) => {
            matches!(model, ModelId::Hrrr | ModelId::HrrrAk | ModelId::WrfGdex)
        }
        (CanonicalField::PressureReducedToMeanSeaLevel, VerticalSelector::MeanSeaLevel) => true,
        (
            CanonicalField::PrecipitableWater | CanonicalField::TotalCloudCover,
            VerticalSelector::EntireAtmosphere,
        ) => true,
        (
            CanonicalField::LowCloudCover
            | CanonicalField::MiddleCloudCover
            | CanonicalField::HighCloudCover,
            VerticalSelector::EntireAtmosphere,
        ) => true,
        // HRRR-Smoke's AOTK/COLMD from its files, and ArWen's own wrfout
        // (AOD5502D, SMOKE_COLUMN) through the store: the chem products
        // draw both through one recipe.
        (CanonicalField::AerosolOpticalDepth550, VerticalSelector::EntireAtmosphere) => {
            matches!(model, ModelId::Hrrr | ModelId::HrrrAk | ModelId::WrfGdex)
        }
        (CanonicalField::ColumnIntegratedSmoke, VerticalSelector::EntireAtmosphere) => {
            matches!(model, ModelId::Hrrr | ModelId::HrrrAk | ModelId::WrfGdex)
        }
        // ArWen chem wrfout planes with no operational-source counterpart
        // (PM2_5_DRY, DUST_SFC and o3 at the lowest mass level).
        (
            CanonicalField::Pm25Dry
            | CanonicalField::DustMassConcentration
            | CanonicalField::OzoneConcentration,
            VerticalSelector::Surface,
        ) => matches!(model, ModelId::WrfGdex),
        (
            CanonicalField::TotalPrecipitation | CanonicalField::ProbabilityOfPrecipitation,
            VerticalSelector::Surface,
        ) => true,
        (CanonicalField::Visibility, VerticalSelector::Surface) => true,
        (CanonicalField::FogFlag, VerticalSelector::Surface) => matches!(model,ModelId::WrfGdex),
        // Surface geopotential height is orography.  Every NWP source in
        // the catalog carries it (it is the model's own terrain), and the
        // store lane resolves it from the WRF import's `orography` plane.
        (CanonicalField::GeopotentialHeight, VerticalSelector::Surface) => true,
        (
            CanonicalField::CategoricalRain
            | CanonicalField::CategoricalFreezingRain
            | CanonicalField::CategoricalIcePellets
            | CanonicalField::CategoricalSnow,
            VerticalSelector::Surface,
        ) => matches!(
            model,
            ModelId::Hrrr
                | ModelId::HrrrAk
                | ModelId::Gfs
                | ModelId::Gdas
                | ModelId::RrfsA
                | ModelId::Rrfs | ModelId::RrfsPublic
                | ModelId::RrfsFireWx
                // WRF-family stores: the importer derives the four flags
                // from the lowest-level hydrometeor fluxes (rw-wrfbatch
                // push_precip_type). Breakage it prevents: a WOOF winter
                // run had no precipitation-type map at all.
                | ModelId::WrfGdex
        ),
        (CanonicalField::LandSeaMask, VerticalSelector::Surface) => {
            matches!(model, ModelId::EcmwfOpenData)
        }
        (CanonicalField::RadarReflectivity, VerticalSelector::HeightAboveGroundMeters(1000)) => {
            matches!(
                model,
                ModelId::Hrrr
                    | ModelId::HrrrAk
                    | ModelId::RrfsA
                    | ModelId::Rrfs | ModelId::RrfsPublic
                    | ModelId::Refs
                    | ModelId::RrfsFireWx
                    | ModelId::WrfGdex
            )
        }
        (CanonicalField::CompositeReflectivity, VerticalSelector::EntireAtmosphere) => {
            matches!(
                model,
                ModelId::Hrrr
                    | ModelId::HrrrAk
                    | ModelId::RrfsA
                    | ModelId::Rrfs | ModelId::RrfsPublic
                    | ModelId::Refs
                    | ModelId::RrfsFireWx
                    | ModelId::WrfGdex
            )
        }
        (
            CanonicalField::UpdraftHelicity,
            VerticalSelector::HeightAboveGroundLayerMeters {
                bottom_m: 2000,
                top_m: 5000,
            },
        ) => matches!(
            model,
            ModelId::Hrrr
                | ModelId::HrrrAk
                | ModelId::Href
                | ModelId::RrfsA
                | ModelId::Rrfs | ModelId::RrfsPublic
                | ModelId::Refs
                | ModelId::RrfsFireWx
                | ModelId::WrfGdex
        ),
        (CanonicalField::SimulatedInfraredBrightnessTemperature, VerticalSelector::NominalTop) => {
            matches!(model, ModelId::Hrrr | ModelId::HrrrAk | ModelId::WrfGdex)
        }
        // The column planes the wrfout import derives itself.  A GRIB
        // source that carries one (HRRR's HGT at the 0 C isotherm) joins
        // this arm when its message mapping is written.
        (CanonicalField::GeopotentialHeight, VerticalSelector::IsothermCelsius(_)) => {
            matches!(model, ModelId::WrfGdex)
        }
        (
            CanonicalField::SupercooledLiquidWaterPath,
            VerticalSelector::EntireAtmosphere | VerticalSelector::HeightAboveGroundLayerMeters { .. },
        ) => matches!(model, ModelId::WrfGdex),
        (
            CanonicalField::TotalCloudCover
            | CanonicalField::CloudWaterMixingRatio
            | CanonicalField::RainWaterMixingRatio
            | CanonicalField::CloudIceMixingRatio
            | CanonicalField::SnowMixingRatio
            | CanonicalField::GraupelMixingRatio,
            VerticalSelector::ColumnMaximum,
        ) => matches!(model, ModelId::WrfGdex),
        _ => false,
    }
}

pub fn model_summary(model: ModelId) -> &'static ModelSummary {
    MODELS
        .iter()
        .find(|entry| entry.id == model)
        .expect("built-in model summary missing")
}

pub fn supported_forecast_hours(model: ModelId, cycle_hour_utc: u8) -> Vec<u16> {
    match model {
        ModelId::Hrrr | ModelId::HrrrAk => {
            if cycle_hour_utc % 6 == 0 {
                (0..=48).collect()
            } else {
                (0..=18).collect()
            }
        }
        ModelId::Gfs => {
            let mut hours = (0..=120).collect::<Vec<u16>>();
            hours.extend((123..=384).step_by(3));
            hours
        }
        ModelId::Gdas => (0..=9).collect(),
        ModelId::Gefs => {
            let mut hours = (0..=240).step_by(3).collect::<Vec<u16>>();
            hours.extend((246..=384).step_by(6));
            hours
        }
        ModelId::Aigfs | ModelId::Aigefs => (0..=384).step_by(6).collect(),
        ModelId::Hgefs => (0..=240).step_by(6).collect(),
        // ECMWF Open Data currently publishes four daily IFS runs. The 00/12z
        // deterministic/ensemble open-data stream carries 3-hourly steps to
        // 144h and then 6-hourly steps to 360h; 06/18z carries 3-hourly steps
        // to 144h only.
        ModelId::EcmwfOpenData => match cycle_hour_utc {
            0 | 12 => {
                let mut hours = (0..=144).step_by(3).collect::<Vec<u16>>();
                hours.extend((150..=360).step_by(6));
                hours
            }
            6 | 18 => (0..=144).step_by(3).collect(),
            _ => Vec::new(),
        },
        ModelId::Aifs => match cycle_hour_utc {
            0 | 6 | 12 | 18 => (0..=AIFS_OPEN_DATA_MAX_FORECAST_HOUR).step_by(6).collect(),
            _ => Vec::new(),
        },
        ModelId::Rap => {
            if rap_extended_forecast_cycle(cycle_hour_utc) {
                (0..=51).collect()
            } else {
                (0..=21).collect()
            }
        }
        ModelId::Nam => {
            let mut hours = (0..=36).collect::<Vec<u16>>();
            hours.extend((39..=84).step_by(3));
            hours
        }
        ModelId::Hiresw => (0..=48).collect(),
        ModelId::Href => (1..=48).collect(),
        ModelId::Sref => (0..=87).step_by(3).collect(),
        ModelId::Rtma | ModelId::Urma => vec![0],
        ModelId::Nbm => (1..=264).collect(),
        ModelId::RrfsA => (0..=60).collect(),
        ModelId::Rrfs => RRFS_CYCLE_HORIZONS
            .get(usize::from(cycle_hour_utc))
            .map(|last| (0..=*last).collect())
            .unwrap_or_default(),
        ModelId::RrfsPublic => (0..=60).collect(),
        ModelId::Refs => (1..=60).collect(),
        ModelId::RrfsFireWx => (0..=36).collect(),
        ModelId::WrfGdex => (0..=23).collect(),
    }
}

pub fn forecast_hour_supported(model: ModelId, cycle_hour_utc: u8, forecast_hour: u16) -> bool {
    supported_forecast_hours(model, cycle_hour_utc)
        .into_iter()
        .any(|candidate| candidate == forecast_hour)
}

fn rap_extended_forecast_cycle(cycle_hour_utc: u8) -> bool {
    matches!(cycle_hour_utc, 3 | 9 | 15 | 21)
}

pub fn resolve_canonical_bundle_product(
    model: ModelId,
    bundle: CanonicalBundleDescriptor,
    native_override: Option<&str>,
) -> ResolvedCanonicalBundleProduct {
    let native_product = native_override
        .unwrap_or_else(|| default_canonical_bundle_product(model, bundle))
        .to_string();
    ResolvedCanonicalBundleProduct::new(bundle, native_product)
}

pub fn default_bundle_product(model: ModelId, bundle: CanonicalBundleDescriptor) -> &'static str {
    default_canonical_bundle_product(model, bundle)
}

fn default_canonical_bundle_product(
    model: ModelId,
    bundle: CanonicalBundleDescriptor,
) -> &'static str {
    match (model, bundle) {
        (ModelId::Hrrr, CanonicalBundleDescriptor::SurfaceAnalysis) => "sfc",
        (ModelId::Hrrr, CanonicalBundleDescriptor::PressureAnalysis) => "prs",
        // HRRR native (UH/composite reflectivity / sub-hourly bundles) live
        // in the `nat` GRIB; the direct lane reroutes them onto `sfc` when
        // the surface family already covers the requested fields, but the
        // canonical native bundle still maps to `nat` here.
        (ModelId::Hrrr, CanonicalBundleDescriptor::NativeAnalysis) => "nat",
        (ModelId::HrrrAk, CanonicalBundleDescriptor::SurfaceAnalysis) => "sfc",
        (ModelId::HrrrAk, CanonicalBundleDescriptor::PressureAnalysis) => "prs",
        (ModelId::HrrrAk, CanonicalBundleDescriptor::NativeAnalysis) => "nat",
        (ModelId::Gfs, _) => "pgrb2.0p25",
        (ModelId::Gdas, _) => "pgrb2.0p25",
        (ModelId::Gefs, _) => "pgrb2ap5/gec00",
        (ModelId::Aigfs, CanonicalBundleDescriptor::SurfaceAnalysis) => "sfc",
        (ModelId::Aigfs, CanonicalBundleDescriptor::PressureAnalysis) => "pres",
        (ModelId::Aigfs, CanonicalBundleDescriptor::NativeAnalysis) => "sfc",
        (ModelId::Aigefs, CanonicalBundleDescriptor::SurfaceAnalysis) => "sfc/mem000",
        (ModelId::Aigefs, CanonicalBundleDescriptor::PressureAnalysis) => "pres/mem000",
        (ModelId::Aigefs, CanonicalBundleDescriptor::NativeAnalysis) => "sfc/mem000",
        (ModelId::Hgefs, CanonicalBundleDescriptor::SurfaceAnalysis) => "sfc/avg",
        (ModelId::Hgefs, CanonicalBundleDescriptor::PressureAnalysis) => "pres/avg",
        (ModelId::Hgefs, CanonicalBundleDescriptor::NativeAnalysis) => "sfc/avg",
        (ModelId::EcmwfOpenData, _) => "oper",
        (ModelId::Aifs, _) => "oper",
        (ModelId::Rap, _) => "awp130pgrb",
        (ModelId::Nam, CanonicalBundleDescriptor::SurfaceAnalysis) => "awip3d",
        (ModelId::Nam, CanonicalBundleDescriptor::PressureAnalysis) => "awip3d",
        (ModelId::Nam, CanonicalBundleDescriptor::NativeAnalysis) => "awip12",
        (ModelId::Hiresw, _) => "arw_2p5km/conus",
        (ModelId::Href, _) => "ensprod/conus/sprd",
        (ModelId::Sref, _) => "ensprod/pgrb212/mean_3hrly",
        (ModelId::Rtma, _) => "2dvaranl_ndfd",
        (ModelId::Urma, _) => "2dvaranl_ndfd",
        (ModelId::Nbm, _) => "core/co",
        // RRFS-A keeps the faster CONUS direct lane on `prs-conus`, but the
        // shared thermo/severe kernels need the NA-domain pair that actually
        // carries both the surface bundle and matching pressure grid.
        (ModelId::RrfsA, CanonicalBundleDescriptor::SurfaceAnalysis) => "nat-na",
        (ModelId::RrfsA, CanonicalBundleDescriptor::PressureAnalysis) => "prs-na",
        (ModelId::RrfsA, CanonicalBundleDescriptor::NativeAnalysis) => "nat-na",
        (ModelId::Rrfs | ModelId::RrfsPublic, CanonicalBundleDescriptor::SurfaceAnalysis) => "2dfld-conus",
        (ModelId::Rrfs | ModelId::RrfsPublic, CanonicalBundleDescriptor::PressureAnalysis) => "prs-conus",
        (ModelId::Rrfs | ModelId::RrfsPublic, CanonicalBundleDescriptor::NativeAnalysis) => "prs-conus",
        (ModelId::Refs, _) => "mean-conus",
        (ModelId::RrfsFireWx, CanonicalBundleDescriptor::SurfaceAnalysis) => "2dfld-firewx",
        (ModelId::RrfsFireWx, CanonicalBundleDescriptor::PressureAnalysis) => "prs-firewx",
        (ModelId::RrfsFireWx, CanonicalBundleDescriptor::NativeAnalysis) => "2dfld-firewx",
        (ModelId::WrfGdex, CanonicalBundleDescriptor::SurfaceAnalysis) => {
            WRF_GDEX_DEFAULT_SURFACE_PRODUCT
        }
        (ModelId::WrfGdex, CanonicalBundleDescriptor::PressureAnalysis) => {
            WRF_GDEX_DEFAULT_PRESSURE_PRODUCT
        }
        (ModelId::WrfGdex, CanonicalBundleDescriptor::NativeAnalysis) => {
            WRF_GDEX_DEFAULT_PRESSURE_PRODUCT
        }
    }
}

/// Build a typed `CanonicalBundleId` for `(model, cycle, fhour, source)`
/// given a typed `BundleRequirement`. This is the planner-side entry point
/// that converts requirements into deduplicated load identities.
pub fn resolve_canonical_bundle_id(
    model: ModelId,
    cycle: rustwx_core::CycleSpec,
    forecast_hour: u16,
    source: SourceId,
    requirement: &rustwx_core::BundleRequirement,
) -> rustwx_core::CanonicalBundleId {
    let resolved = resolve_canonical_bundle_product(
        model,
        requirement.bundle,
        requirement.native_override.as_deref(),
    );
    rustwx_core::CanonicalBundleId::new(
        model,
        cycle,
        forecast_hour,
        source,
        requirement.bundle,
        resolved.native_product,
    )
}

pub fn resolve_urls(request: &ModelRunRequest) -> Result<Vec<ResolvedUrl>, ModelError> {
    let mut urls = Vec::new();
    let mut errors = Vec::new();
    for source in model_summary(request.model).sources {
        match build_grib_url(source.id, request) {
            Ok(grib_url) => {
                if grib_url.starts_with("unsupported://") {
                    continue;
                }
                let idx_url = if source.idx_available {
                    Some(format!("{grib_url}.idx"))
                } else {
                    None
                };
                urls.push(ResolvedUrl {
                    source: source.id,
                    grib_url,
                    idx_url,
                });
            }
            Err(err) => errors.push(err),
        }
    }
    if urls.is_empty() {
        if let Some(err) = errors.into_iter().next() {
            return Err(err);
        }
        return Err(ModelError::UnsupportedProduct {
            model: request.model,
            product: request.product.clone(),
        });
    }
    urls.sort_by_key(|entry| {
        model_summary(request.model)
            .sources
            .iter()
            .find(|source| source.id == entry.source)
            .map(|source| source.priority)
            .unwrap_or(u8::MAX)
    });
    Ok(urls)
}

pub fn latest_available_run(
    model: ModelId,
    source: Option<SourceId>,
    date_yyyymmdd: &str,
) -> Result<LatestRun, ModelError> {
    let agent = build_agent();
    latest_available_run_with_probe(model, source, date_yyyymmdd, |resolved| {
        availability_probe_ok(&agent, resolved)
    })
}

pub fn latest_available_run_at_forecast_hour(
    model: ModelId,
    source: Option<SourceId>,
    date_yyyymmdd: &str,
    forecast_hour: u16,
) -> Result<LatestRun, ModelError> {
    let agent = build_agent();
    latest_available_run_for_products_with_probe_at_forecast_hour(
        model,
        source,
        date_yyyymmdd,
        &[model_summary(model).default_product],
        forecast_hour,
        |resolved| availability_probe_ok(&agent, resolved),
    )
}

pub fn latest_available_run_for_products_at_forecast_hour(
    model: ModelId,
    source: Option<SourceId>,
    date_yyyymmdd: &str,
    products: &[&str],
    forecast_hour: u16,
) -> Result<LatestRun, ModelError> {
    let agent = build_agent();
    latest_available_run_for_products_with_probe_at_forecast_hour(
        model,
        source,
        date_yyyymmdd,
        products,
        forecast_hour,
        |resolved| availability_probe_ok(&agent, resolved),
    )
}

fn latest_available_run_with_probe<F>(
    model: ModelId,
    source: Option<SourceId>,
    date_yyyymmdd: &str,
    probe_available: F,
) -> Result<LatestRun, ModelError>
where
    F: FnMut(&ResolvedUrl) -> bool,
{
    latest_available_run_for_products_with_probe_at_forecast_hour(
        model,
        source,
        date_yyyymmdd,
        &[model_summary(model).default_product],
        0,
        probe_available,
    )
}

#[cfg(test)]
fn latest_available_run_for_products_with_probe<F>(
    model: ModelId,
    source: Option<SourceId>,
    date_yyyymmdd: &str,
    products: &[&str],
    probe_available: F,
) -> Result<LatestRun, ModelError>
where
    F: FnMut(&ResolvedUrl) -> bool,
{
    latest_available_run_for_products_with_probe_at_forecast_hour(
        model,
        source,
        date_yyyymmdd,
        products,
        0,
        probe_available,
    )
}

fn latest_available_run_for_products_with_probe_at_forecast_hour<F>(
    model: ModelId,
    source: Option<SourceId>,
    date_yyyymmdd: &str,
    products: &[&str],
    forecast_hour: u16,
    mut probe_available: F,
) -> Result<LatestRun, ModelError>
where
    F: FnMut(&ResolvedUrl) -> bool,
{
    let summary = model_summary(model);
    let allowed_sources = summary
        .sources
        .iter()
        .filter(|candidate| source.map(|wanted| candidate.id == wanted).unwrap_or(true))
        .map(|candidate| candidate.id)
        .collect::<Vec<_>>();
    if allowed_sources.is_empty() {
        return Err(ModelError::NoAvailableRun { model });
    }
    let required_products = if products.is_empty() {
        vec![summary.default_product]
    } else {
        let mut deduped = Vec::new();
        for product in products {
            if !deduped.iter().any(|seen| seen == product) {
                deduped.push(*product);
            }
        }
        deduped
    };

    // Walk today, then (if nothing on today) yesterday. During the publishing
    // window between two cycles the newest cycle may not be available yet,
    // but the previous day's last cycle typically still is, without this
    // rollback, `latest_available_run` wrongly reports NoAvailableRun just
    // after UTC rollover or during a publication delay.
    let candidate_dates = cycle_date_rollback_candidates(date_yyyymmdd);
    for candidate_date in &candidate_dates {
        for hour_utc in summary.cycle_hours_utc.iter().rev().copied() {
            if !forecast_hour_supported(model, hour_utc, forecast_hour) {
                continue;
            }
            let cycle = match CycleSpec::new(candidate_date.clone(), hour_utc) {
                Ok(cycle) => cycle,
                Err(_) => continue,
            };
            let available_source = allowed_sources.iter().copied().find(|candidate_source| {
                required_products.iter().all(|product| {
                    let request =
                        match ModelRunRequest::new(model, cycle.clone(), forecast_hour, *product) {
                            Ok(request) => request,
                            Err(_) => return false,
                        };
                    resolve_urls(&request)
                        .map(|resolved_urls| {
                            resolved_urls.into_iter().any(|resolved| {
                                resolved.source == *candidate_source && probe_available(&resolved)
                            })
                        })
                        .unwrap_or(false)
                })
            });

            if let Some(source) = available_source {
                return Ok(LatestRun {
                    model,
                    cycle,
                    source,
                });
            }
        }
    }

    Err(ModelError::NoAvailableRun { model })
}

fn cycle_date_rollback_candidates(date_yyyymmdd: &str) -> Vec<String> {
    let mut dates = Vec::with_capacity(2);
    dates.push(date_yyyymmdd.to_string());
    if let Some(previous) = previous_day_yyyymmdd(date_yyyymmdd) {
        dates.push(previous);
    }
    dates
}

fn previous_day_yyyymmdd(date_yyyymmdd: &str) -> Option<String> {
    if date_yyyymmdd.len() != 8 || !date_yyyymmdd.chars().all(|ch| ch.is_ascii_digit()) {
        return None;
    }
    let year: i32 = date_yyyymmdd[..4].parse().ok()?;
    let month: u32 = date_yyyymmdd[4..6].parse().ok()?;
    let day: u32 = date_yyyymmdd[6..8].parse().ok()?;
    let (new_year, new_month, new_day) = if day > 1 {
        (year, month, day - 1)
    } else if month > 1 {
        (year, month - 1, days_in_month(year, month - 1))
    } else {
        (year - 1, 12, 31)
    };
    if new_year < 1 {
        return None;
    }
    Some(format!("{:04}{:02}{:02}", new_year, new_month, new_day))
}

fn next_day_yyyymmdd(date_yyyymmdd: &str) -> Option<String> {
    if date_yyyymmdd.len() != 8 || !date_yyyymmdd.chars().all(|ch| ch.is_ascii_digit()) {
        return None;
    }
    let year: i32 = date_yyyymmdd[..4].parse().ok()?;
    let month: u32 = date_yyyymmdd[4..6].parse().ok()?;
    let day: u32 = date_yyyymmdd[6..8].parse().ok()?;
    let month_days = days_in_month(year, month);
    let (new_year, new_month, new_day) = if day < month_days {
        (year, month, day + 1)
    } else if month < 12 {
        (year, month + 1, 1)
    } else {
        (year + 1, 1, 1)
    };
    Some(format!("{:04}{:02}{:02}", new_year, new_month, new_day))
}

fn advance_yyyymmddhh(
    date_yyyymmdd: &str,
    hour_utc: u8,
    forecast_hour: u16,
) -> Option<(String, u8)> {
    let mut date = date_yyyymmdd.to_string();
    let total_hours = u32::from(hour_utc) + u32::from(forecast_hour);
    let day_rollovers = total_hours / 24;
    let valid_hour = (total_hours % 24) as u8;
    for _ in 0..day_rollovers {
        date = next_day_yyyymmdd(&date)?;
    }
    Some((date, valid_hour))
}

fn days_in_month(year: i32, month: u32) -> u32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => {
            let is_leap = (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
            if is_leap { 29 } else { 28 }
        }
        _ => 30,
    }
}

fn build_agent() -> ureq::Agent {
    rustls::crypto::CryptoProvider::install_default(rustls_rustcrypto::provider()).ok();
    let crypto = std::sync::Arc::new(rustls_rustcrypto::provider());
    ureq::Agent::config_builder()
        .tls_config(
            ureq::tls::TlsConfig::builder()
                .provider(ureq::tls::TlsProvider::Rustls)
                .root_certs(ureq::tls::RootCerts::WebPki)
                .unversioned_rustls_crypto_provider(crypto)
                .build(),
        )
        .build()
        .new_agent()
}

fn availability_probe_ok(agent: &ureq::Agent, resolved: &ResolvedUrl) -> bool {
    if should_use_range_probe(resolved.source) {
        return range_probe_ok(agent, &resolved.grib_url);
    }
    head_ok(agent, resolved.availability_probe_url())
}

fn should_use_range_probe(source: SourceId) -> bool {
    matches!(source, SourceId::Nomads)
}

fn head_ok(agent: &ureq::Agent, url: &str) -> bool {
    let response = if url.contains("nomads.ncep.noaa.gov") {
        agent.get(url).header("Range", "bytes=0-0").call()
    } else {
        agent.head(url).call()
    };
    match response {
        Ok(_) => true,
        Err(ureq::Error::StatusCode(code)) if code == 403 || code == 404 => false,
        Err(_) => false,
    }
}

fn range_probe_ok(agent: &ureq::Agent, url: &str) -> bool {
    match agent.get(url).header("Range", "bytes=0-0").call() {
        Ok(_) => true,
        Err(ureq::Error::StatusCode(code)) if code == 403 || code == 404 => false,
        Err(_) => false,
    }
}

fn build_grib_url(source: SourceId, request: &ModelRunRequest) -> Result<String, ModelError> {
    if let Some(result) = archives::object_url(source, request) {
        return result;
    }
    Ok(match request.model {
        ModelId::Hrrr => build_hrrr_url(source, request),
        ModelId::HrrrAk => build_hrrr_ak_url(source, request),
        ModelId::Gfs => build_gfs_url(source, request)?,
        ModelId::Gdas => build_gdas_url(source, request)?,
        ModelId::Gefs => build_gefs_url(source, request)?,
        ModelId::Aigfs => build_aigfs_url(source, request)?,
        ModelId::Aigefs => build_aigefs_url(source, request)?,
        ModelId::Hgefs => build_hgefs_url(source, request)?,
        ModelId::EcmwfOpenData => build_ecmwf_url(source, request)?,
        ModelId::Aifs => build_aifs_url(source, request)?,
        ModelId::Rap => build_rap_url(source, request)?,
        ModelId::Nam => build_nam_url(source, request)?,
        ModelId::Hiresw => build_hiresw_url(source, request)?,
        ModelId::Href => build_href_url(source, request)?,
        ModelId::Sref => build_sref_url(source, request)?,
        ModelId::Rtma => build_rtma_url(source, request)?,
        ModelId::Urma => build_urma_url(source, request)?,
        ModelId::Nbm => build_nbm_url(source, request)?,
        ModelId::RrfsA => build_rrfs_a_url(source, request)?,
        ModelId::Rrfs | ModelId::RrfsPublic => build_rrfs_conus_url(source, request)?,
        ModelId::Refs => build_refs_url(source, request)?,
        ModelId::RrfsFireWx => build_rrfs_firewx_url(source, request)?,
        ModelId::WrfGdex => build_wrf_gdex_url(source, request)?,
    })
}

enum WrfGdexProduct {
    LegacyWrfout {
        dataset: String,
        domain: String,
    },
    ClimateArchive {
        dataset: String,
        branch: &'static str,
        filename_prefix: &'static str,
        cadence_hours: Option<u8>,
    },
}

fn parse_wrf_gdex_product(product: &str) -> Option<WrfGdexProduct> {
    let token = normalize_token(product);
    if let Some((dataset, suffix)) = token.split_once('_') {
        if dataset.starts_with('d')
            && dataset.len() == 7
            && dataset[1..].chars().all(|ch| ch.is_ascii_digit())
        {
            if suffix.starts_with('d')
                && suffix.len() == 3
                && suffix[1..].chars().all(|ch| ch.is_ascii_digit())
            {
                return Some(WrfGdexProduct::LegacyWrfout {
                    dataset: dataset.to_string(),
                    domain: suffix.to_string(),
                });
            }
            let climate = match suffix {
                "hist2d" => Some(("hist2D", "wrf2d_d01", None)),
                "hist3d" => Some(("hist3D", "wrf3d_d01", Some(3))),
                "future2d" => Some(("future2D", "wrf2d_d01", None)),
                "future3d" => Some(("future3D", "wrf3d_d01", Some(3))),
                _ => None,
            };
            if let Some((branch, filename_prefix, cadence_hours)) = climate {
                return Some(WrfGdexProduct::ClimateArchive {
                    dataset: dataset.to_string(),
                    branch,
                    filename_prefix,
                    cadence_hours,
                });
            }
        }
    }
    if token == "d612005" {
        return Some(WrfGdexProduct::ClimateArchive {
            dataset: token,
            branch: "hist2D",
            filename_prefix: "wrf2d_d01",
            cadence_hours: None,
        });
    }
    if token.starts_with('d')
        && token.len() == 7
        && token[1..].chars().all(|ch| ch.is_ascii_digit())
    {
        return Some(WrfGdexProduct::LegacyWrfout {
            dataset: token,
            domain: "d01".to_string(),
        });
    }
    None
}

fn build_wrf_gdex_url(source: SourceId, request: &ModelRunRequest) -> Result<String, ModelError> {
    if source != SourceId::Gdex {
        return Ok(unsupported_source(source, request.model));
    }
    let product =
        parse_wrf_gdex_product(&request.product).ok_or_else(|| ModelError::UnsupportedProduct {
            model: request.model,
            product: request.product.clone(),
        })?;
    let (valid_date, valid_hour) = advance_yyyymmddhh(
        &request.cycle.date_yyyymmdd,
        request.cycle.hour_utc,
        request.forecast_hour,
    )
    .ok_or_else(|| ModelError::UnsupportedForecastHour {
        model: request.model,
        cycle_hour: request.cycle.hour_utc,
        forecast_hour: request.forecast_hour,
        reason: "wrfout archive timestamp rollover exceeded the supported date arithmetic path"
            .to_string(),
    })?;
    let year = &valid_date[..4];
    let month = &valid_date[4..6];
    let day = &valid_date[6..8];
    match product {
        WrfGdexProduct::LegacyWrfout { dataset, domain } => Ok(format!(
            "https://tds.gdex.ucar.edu/thredds/fileServer/files/{dataset}/{year}{month}/wrfout_{domain}_{year}-{month}-{day}_{:02}:00:00.nc",
            valid_hour
        )),
        WrfGdexProduct::ClimateArchive {
            dataset,
            branch,
            filename_prefix,
            cadence_hours,
        } => {
            if let Some(cadence) = cadence_hours {
                if valid_hour % cadence != 0 {
                    return Err(ModelError::UnsupportedForecastHour {
                        model: request.model,
                        cycle_hour: request.cycle.hour_utc,
                        forecast_hour: request.forecast_hour,
                        reason: format!(
                            "WRF GDEX product '{branch}' is published every {cadence} hours; requested valid time {:02}Z is off cadence",
                            valid_hour
                        ),
                    });
                }
            }
            Ok(format!(
                "https://tds.gdex.ucar.edu/thredds/fileServer/files/g/{dataset}/{branch}/{year}{month}/{filename_prefix}_{year}-{month}-{day}_{:02}:00:00.nc",
                valid_hour
            ))
        }
    }
}

fn build_hrrr_url(source: SourceId, request: &ModelRunRequest) -> String {
    let product_code = match normalize_token(&request.product).as_str() {
        "sfc" | "surface" => "wrfsfc",
        "prs" | "pressure" => "wrfprs",
        "nat" | "native" => "wrfnat",
        "subh" | "subhourly" => "wrfsubh",
        _ => "wrfsfc",
    };

    match source {
        SourceId::Aws => format!(
            "https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.{}/conus/hrrr.t{:02}z.{}f{:02}.grib2",
            request.cycle.date_yyyymmdd,
            request.cycle.hour_utc,
            product_code,
            request.forecast_hour
        ),
        SourceId::Nomads => format!(
            "https://nomads.ncep.noaa.gov/pub/data/nccf/com/hrrr/prod/hrrr.{}/conus/hrrr.t{:02}z.{}f{:02}.grib2",
            request.cycle.date_yyyymmdd,
            request.cycle.hour_utc,
            product_code,
            request.forecast_hour
        ),
        SourceId::Google => format!(
            "https://storage.googleapis.com/high-resolution-rapid-refresh/hrrr.{}/conus/hrrr.t{:02}z.{}f{:02}.grib2",
            request.cycle.date_yyyymmdd,
            request.cycle.hour_utc,
            product_code,
            request.forecast_hour
        ),
        SourceId::Azure => format!(
            "https://noaahrrr.blob.core.windows.net/hrrr/hrrr.{}/conus/hrrr.t{:02}z.{}f{:02}.grib2",
            request.cycle.date_yyyymmdd,
            request.cycle.hour_utc,
            product_code,
            request.forecast_hour
        ),
        other => unsupported_source(other, request.model),
    }
}

fn build_hrrr_ak_url(source: SourceId, request: &ModelRunRequest) -> String {
    let product_code = match normalize_token(&request.product).as_str() {
        "sfc" | "surface" => "wrfsfc",
        "prs" | "pressure" => "wrfprs",
        "nat" | "native" => "wrfnat",
        "subh" | "subhourly" => "wrfsubh",
        _ => "wrfsfc",
    };

    match source {
        SourceId::Aws => format!(
            "https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.{}/alaska/hrrr.t{:02}z.{}f{:02}.ak.grib2",
            request.cycle.date_yyyymmdd,
            request.cycle.hour_utc,
            product_code,
            request.forecast_hour
        ),
        SourceId::Nomads => format!(
            "https://nomads.ncep.noaa.gov/pub/data/nccf/com/hrrr/prod/hrrr.{}/alaska/hrrr.t{:02}z.{}f{:02}.ak.grib2",
            request.cycle.date_yyyymmdd,
            request.cycle.hour_utc,
            product_code,
            request.forecast_hour
        ),
        SourceId::Google => format!(
            "https://storage.googleapis.com/high-resolution-rapid-refresh/hrrr.{}/alaska/hrrr.t{:02}z.{}f{:02}.ak.grib2",
            request.cycle.date_yyyymmdd,
            request.cycle.hour_utc,
            product_code,
            request.forecast_hour
        ),
        SourceId::Azure => format!(
            "https://noaahrrr.blob.core.windows.net/hrrr/hrrr.{}/alaska/hrrr.t{:02}z.{}f{:02}.ak.grib2",
            request.cycle.date_yyyymmdd,
            request.cycle.hour_utc,
            product_code,
            request.forecast_hour
        ),
        other => unsupported_source(other, request.model),
    }
}

fn gfs_product_filename_part(product: &str) -> Result<GfsProductPart, ModelError> {
    let token = normalize_token(product);
    let part = match token.as_str() {
        "pgrb2_0p25" | "pgrb2_0_25" | "0p25" | "0_25" | "hourly" | "pgrb2" => {
            GfsProductPart::Forecast("pgrb2.0p25")
        }
        "pgrb2_0p50" | "pgrb2_0_50" | "0p50" | "0_50" => GfsProductPart::Forecast("pgrb2.0p50"),
        "pgrb2_1p00" | "pgrb2_1_00" | "1p00" | "1_00" => GfsProductPart::Forecast("pgrb2.1p00"),
        "pgrb2b_0p25" | "pgrb2b_0_25" | "secondary" | "secondary_params" => {
            GfsProductPart::Forecast("pgrb2b.0p25")
        }
        "sflux" | "sfluxgrb" | "sfluxgrbf" => GfsProductPart::Sflux,
        other => {
            return Err(ModelError::UnsupportedProduct {
                model: ModelId::Gfs,
                product: other.to_string(),
            });
        }
    };
    Ok(part)
}

enum GfsProductPart {
    Forecast(&'static str),
    Sflux,
}

fn build_gfs_url(source: SourceId, request: &ModelRunRequest) -> Result<String, ModelError> {
    let filename = match gfs_product_filename_part(&request.product) {
        Ok(GfsProductPart::Forecast(part)) => format!(
            "gfs.t{:02}z.{}.f{:03}",
            request.cycle.hour_utc, part, request.forecast_hour
        ),
        Ok(GfsProductPart::Sflux) => format!(
            "gfs.t{:02}z.sfluxgrbf{:03}.grib2",
            request.cycle.hour_utc, request.forecast_hour
        ),
        Err(err) => return Err(err),
    };
    Ok(match source {
        SourceId::Aws => format!(
            "https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{}/{:02}/atmos/{}",
            request.cycle.date_yyyymmdd, request.cycle.hour_utc, filename
        ),
        SourceId::Nomads => format!(
            "https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/gfs.{}/{:02}/atmos/{}",
            request.cycle.date_yyyymmdd, request.cycle.hour_utc, filename
        ),
        SourceId::Google => format!(
            "https://storage.googleapis.com/global-forecast-system/gfs.{}/{:02}/atmos/{}",
            request.cycle.date_yyyymmdd, request.cycle.hour_utc, filename
        ),
        SourceId::Ncei => {
            let year = &request.cycle.date_yyyymmdd[..4];
            let month = &request.cycle.date_yyyymmdd[4..6];
            let day = &request.cycle.date_yyyymmdd[6..8];
            format!(
                "https://www.ncei.noaa.gov/data/global-forecast-system/access/grid-004-0.5-degree/analysis/{}{}/{}{}{}/gfs_4_{}{}{}_{}00_{:03}.grb2",
                year,
                month,
                year,
                month,
                day,
                year,
                month,
                day,
                format_args!("{:02}", request.cycle.hour_utc),
                request.forecast_hour
            )
        }
        other => unsupported_source(other, request.model),
    })
}

fn build_gdas_url(source: SourceId, request: &ModelRunRequest) -> Result<String, ModelError> {
    if !forecast_hour_supported(request.model, request.cycle.hour_utc, request.forecast_hour) {
        return Err(ModelError::UnsupportedForecastHour {
            model: request.model,
            cycle_hour: request.cycle.hour_utc,
            forecast_hour: request.forecast_hour,
            reason: "GDAS pgrb2 fields are wired for f000 through f009 in this v0.5 path"
                .to_string(),
        });
    }
    let product = match normalize_token(&request.product).as_str() {
        "pgrb2_0p25" | "pgrb2_0_25" | "0p25" | "0_25" | "pgrb2" => "pgrb2.0p25",
        "pgrb2_1p00" | "pgrb2_1_00" | "1p00" | "1_00" => "pgrb2.1p00",
        other => {
            return Err(ModelError::UnsupportedProduct {
                model: request.model,
                product: other.to_string(),
            });
        }
    };
    Ok(match source {
        SourceId::Aws => format!(
            "https://noaa-gfs-bdp-pds.s3.amazonaws.com/gdas.{}/{:02}/atmos/gdas.t{:02}z.{}.f{:03}",
            request.cycle.date_yyyymmdd,
            request.cycle.hour_utc,
            request.cycle.hour_utc,
            product,
            request.forecast_hour
        ),
        SourceId::Nomads => format!(
            "https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/gdas.{}/{:02}/atmos/gdas.t{:02}z.{}.f{:03}",
            request.cycle.date_yyyymmdd,
            request.cycle.hour_utc,
            request.cycle.hour_utc,
            product,
            request.forecast_hour
        ),
        other => unsupported_source(other, request.model),
    })
}

fn build_gefs_url(source: SourceId, request: &ModelRunRequest) -> Result<String, ModelError> {
    if !forecast_hour_supported(request.model, request.cycle.hour_utc, request.forecast_hour) {
        return Err(ModelError::UnsupportedForecastHour {
            model: request.model,
            cycle_hour: request.cycle.hour_utc,
            forecast_hour: request.forecast_hour,
            reason: "GEFS pgrb2ap5 fields are expected every 3 hours through f384".to_string(),
        });
    }
    let product = gefs_product_from_product(&request.product)?;
    Ok(match source {
        SourceId::Aws => format!(
            "https://noaa-gefs-pds.s3.amazonaws.com/gefs.{}/{:02}/atmos/{}/{}.t{:02}z.{}.f{:03}",
            request.cycle.date_yyyymmdd,
            request.cycle.hour_utc,
            product.directory,
            product.member_or_stat,
            request.cycle.hour_utc,
            product.file_product,
            request.forecast_hour
        ),
        SourceId::Nomads => format!(
            "https://nomads.ncep.noaa.gov/pub/data/nccf/com/gens/prod/gefs.{}/{:02}/atmos/{}/{}.t{:02}z.{}.f{:03}",
            request.cycle.date_yyyymmdd,
            request.cycle.hour_utc,
            product.directory,
            product.member_or_stat,
            request.cycle.hour_utc,
            product.file_product,
            request.forecast_hour
        ),
        other => unsupported_source(other, request.model),
    })
}

struct GefsProduct {
    directory: &'static str,
    file_product: &'static str,
    member_or_stat: String,
}

fn gefs_product_from_product(product: &str) -> Result<GefsProduct, ModelError> {
    let token = normalize_token(product);
    let (family, explicit) = token.split_once('/').map(|(family, value)| (family, Some(value)))
        .unwrap_or((token.as_str(), None));
    let (directory, file_product) = match family {
        "pgrb2ap5" | "pgrb2a" => ("pgrb2ap5", "pgrb2a.0p50"),
        "pgrb2bp5" | "pgrb2b" | "secondary" => ("pgrb2bp5", "pgrb2b.0p50"),
        "pgrb2sp25" | "pgrb2s" | "0p25" | "0_25" => ("pgrb2sp25", "pgrb2s.0p25"),
        _ => return Err(ModelError::UnsupportedProduct { model: ModelId::Gefs, product: product.into() }),
    };
    let candidate = explicit.unwrap_or(if directory == "pgrb2sp25" { "geavg" } else { "gec00" });
    let individual = product_member(ModelId::Gefs, &format!("{directory}/{candidate}")).is_some();
    let statistic = matches!(candidate, "geavg" | "gespr");
    if !individual && !statistic || statistic && directory == "pgrb2bp5" {
        return Err(ModelError::UnsupportedProduct { model: ModelId::Gefs, product: product.into() });
    }
    Ok(GefsProduct { directory, file_product, member_or_stat: candidate.to_string() })
}

fn build_ecmwf_url(source: SourceId, request: &ModelRunRequest) -> Result<String, ModelError> {
    if source != SourceId::Ecmwf {
        return Ok(unsupported_source(source, request.model));
    }
    if !forecast_hour_supported(request.model, request.cycle.hour_utc, request.forecast_hour) {
        let reason = match request.cycle.hour_utc {
            0 | 12 => {
                "00/12z open-data runs expose 3-hourly steps to 144h, then 6-hourly steps to 360h"
            }
            6 | 18 => "06/18z open-data runs expose 3-hourly steps to 144h only",
            _ => "ECMWF open-data runs are currently expected at 00/06/12/18 UTC",
        };
        return Err(ModelError::UnsupportedForecastHour {
            model: request.model,
            cycle_hour: request.cycle.hour_utc,
            forecast_hour: request.forecast_hour,
            reason: reason.to_string(),
        });
    }
    let stream = match normalize_token(&request.product).as_str() {
        "oper" | "hres" | "euro" | "ifs" | "ecmwf" => "oper",
        "ens" | "enfo" | "ensemble" => "enfo",
        "wave" | "wam" => "wave",
        other => {
            return Err(ModelError::UnsupportedProduct {
                model: request.model,
                product: other.to_string(),
            });
        }
    };
    Ok(format!(
        "https://data.ecmwf.int/forecasts/{}/{:02}z/ifs/0p25/{}/{}{:02}0000-{}h-{}-fc.grib2",
        request.cycle.date_yyyymmdd,
        request.cycle.hour_utc,
        stream,
        request.cycle.date_yyyymmdd,
        request.cycle.hour_utc,
        request.forecast_hour,
        stream
    ))
}

fn build_aifs_url(source: SourceId, request: &ModelRunRequest) -> Result<String, ModelError> {
    match source {
        SourceId::AifsInference => Ok(format!(
            "aifs-inference://aifs/{}T{:02}Z/lead{:03}.nc",
            request.cycle.date_yyyymmdd, request.cycle.hour_utc, request.forecast_hour
        )),
        SourceId::Earth2Archive => Ok(format!(
            "earth2-archive://aifs/{}T{:02}Z/lead{:03}.nc",
            request.cycle.date_yyyymmdd, request.cycle.hour_utc, request.forecast_hour
        )),
        SourceId::Ecmwf => {
            if !aifs_open_data_forecast_hour_supported(
                request.cycle.hour_utc,
                request.forecast_hour,
            ) {
                return Err(ModelError::UnsupportedForecastHour {
                    model: request.model,
                    cycle_hour: request.cycle.hour_utc,
                    forecast_hour: request.forecast_hour,
                    reason: "ECMWF AIFS Single v2 publishes 6-hourly steps from f000 through f360 on 00/06/12/18z cycles; explicit local archive URLs retain their own experimental horizons".to_string(),
                });
            }
            let stream = match normalize_token(&request.product).as_str() {
                "oper" | "hres" | "aifs" | "aifs_v2" | "aifsv2" | "aifs_single"
                | "aifs_single_v2" => "oper",
                "wave" | "wam" => "wave",
                other => {
                    return Err(ModelError::UnsupportedProduct {
                        model: request.model,
                        product: other.to_string(),
                    });
                }
            };
            Ok(format!(
                "https://data.ecmwf.int/forecasts/{}/{:02}z/aifs-single/0p25/{}/{}{:02}0000-{}h-{}-fc.grib2",
                request.cycle.date_yyyymmdd,
                request.cycle.hour_utc,
                stream,
                request.cycle.date_yyyymmdd,
                request.cycle.hour_utc,
                request.forecast_hour,
                stream
            ))
        }
        other => Ok(unsupported_source(other, request.model)),
    }
}

fn aifs_open_data_forecast_hour_supported(cycle_hour_utc: u8, forecast_hour: u16) -> bool {
    AIFS_CYCLE_HOURS.contains(&cycle_hour_utc)
        && forecast_hour <= AIFS_OPEN_DATA_MAX_FORECAST_HOUR
        && forecast_hour % 6 == 0
}

fn build_aigfs_url(source: SourceId, request: &ModelRunRequest) -> Result<String, ModelError> {
    if source != SourceId::Nomads {
        return Ok(unsupported_source(source, request.model));
    }
    let family = match normalize_token(&request.product).as_str() {
        "sfc" | "surface" => "sfc",
        "pres" | "pressure" | "pgrb" | "pgrb2" => "pres",
        other => {
            return Err(ModelError::UnsupportedProduct {
                model: request.model,
                product: other.to_string(),
            });
        }
    };
    Ok(format!(
        "https://nomads.ncep.noaa.gov/pub/data/nccf/com/aigfs/prod/aigfs.{}/{:02}/model/atmos/grib2/aigfs.t{:02}z.{}.f{:03}.grib2",
        request.cycle.date_yyyymmdd,
        request.cycle.hour_utc,
        request.cycle.hour_utc,
        family,
        request.forecast_hour
    ))
}

fn build_aigefs_url(source: SourceId, request: &ModelRunRequest) -> Result<String, ModelError> {
    let token = normalize_token(&request.product);
    let (family, selection) = token.split_once('/').or_else(|| token.split_once('_'))
        .unwrap_or((token.as_str(), "mem000"));
    let family = match family {
        "sfc" | "surface" => "sfc",
        "pres" | "pressure" => "pres",
        _ => return Err(ModelError::UnsupportedProduct { model: request.model, product: request.product.clone() }),
    };
    let selected = ModelRunRequest { product: format!("{family}/{selection}"), ..request.clone() };
    let individual = product_member(request.model, &selected.product).is_some();
    if !individual && !matches!(selection, "avg" | "spr" | "p10" | "p50" | "p90") {
        return Err(ModelError::UnsupportedProduct { model: request.model, product: request.product.clone() });
    }
    if !forecast_hour_supported(request.model, request.cycle.hour_utc, request.forecast_hour) {
        return Err(ModelError::UnsupportedForecastHour { model: request.model,
            cycle_hour: request.cycle.hour_utc, forecast_hour: request.forecast_hour,
            reason: "AI-GEFS products publish every six hours through f384".into() });
    }
    if individual {
        return Ok(member_url(source, &selected).unwrap_or_else(|| unsupported_source(source, request.model)));
    }
    if source != SourceId::Nomads {
        return Ok(unsupported_source(source, request.model));
    }
    if family == "sfc" && request.forecast_hour == 0 {
        return Err(ModelError::UnsupportedForecastHour { model: request.model,
            cycle_hour: request.cycle.hour_utc, forecast_hour: request.forecast_hour,
            reason: "surface statistics start at f006; select an individual member for f000".into() });
    }
    Ok(format!(
        "https://nomads.ncep.noaa.gov/pub/data/nccf/com/aigefs/prod/aigefs.{}/{:02}/ensstat/products/atmos/grib2/aigefs.t{:02}z.{}.{}.f{:03}.grib2",
        request.cycle.date_yyyymmdd, request.cycle.hour_utc,
        request.cycle.hour_utc, family, selection, request.forecast_hour
    ))
}

fn build_hgefs_url(source: SourceId, request: &ModelRunRequest) -> Result<String, ModelError> {
    if source != SourceId::Nomads {
        return Ok(unsupported_source(source, request.model));
    }
    let token = normalize_token(&request.product);
    let family = if token.contains("pres") || token.contains("pressure") {
        "pres"
    } else {
        "sfc"
    };
    let stat = token
        .split(['_', '/'])
        .find(|part| matches!(*part, "avg" | "spr" | "p10" | "p50" | "p90"))
        .unwrap_or("avg");
    Ok(format!(
        "https://nomads.ncep.noaa.gov/pub/data/nccf/com/hgefs/prod/hgefs.{}/{:02}/ensstat/products/atmos/grib2/hgefs.t{:02}z.{}.{}.f{:03}.grib2",
        request.cycle.date_yyyymmdd,
        request.cycle.hour_utc,
        request.cycle.hour_utc,
        family,
        stat,
        request.forecast_hour
    ))
}

fn build_rap_url(source: SourceId, request: &ModelRunRequest) -> Result<String, ModelError> {
    if !forecast_hour_supported(request.model, request.cycle.hour_utc, request.forecast_hour) {
        return Err(ModelError::UnsupportedForecastHour {
            model: request.model,
            cycle_hour: request.cycle.hour_utc,
            forecast_hour: request.forecast_hour,
            reason: "RAP publishes f000 through f021 on most cycles and f000 through f051 on 03/09/15/21z cycles"
                .to_string(),
        });
    }
    let prefix = match normalize_token(&request.product).as_str() {
        "awp130pgrb" | "pgrb" | "prs" | "pressure" | "conus" => "awp130pgrb",
        "awp130bgrb" | "bgrb" | "secondary" => "awp130bgrb",
        "awip32" | "na32" | "north_america_32km" => "awip32",
        "wrfprs" | "wrf_prs" => "wrfprs",
        "wrfsfc" | "sfc" | "surface" => "wrfsfc",
        "wrfnat" | "nat" | "native" => "wrfnat",
        other => {
            return Err(ModelError::UnsupportedProduct {
                model: request.model,
                product: other.to_string(),
            });
        }
    };
    Ok(match source {
        SourceId::Aws => format!(
            "https://noaa-rap-pds.s3.amazonaws.com/rap.{}/rap.t{:02}z.{}f{:02}.grib2",
            request.cycle.date_yyyymmdd, request.cycle.hour_utc, prefix, request.forecast_hour
        ),
        SourceId::Nomads => format!(
            "https://nomads.ncep.noaa.gov/pub/data/nccf/com/rap/prod/rap.{}/rap.t{:02}z.{}f{:02}.grib2",
            request.cycle.date_yyyymmdd, request.cycle.hour_utc, prefix, request.forecast_hour
        ),
        other => unsupported_source(other, request.model),
    })
}

fn build_nam_url(source: SourceId, request: &ModelRunRequest) -> Result<String, ModelError> {
    let filename = match normalize_token(&request.product).as_str() {
        "awip12" | "conus" | "conus12" | "conus_12km" => format!(
            "nam.t{:02}z.awip12{:02}.tm00.grib2",
            request.cycle.hour_utc, request.forecast_hour
        ),
        "awip32" | "na32" | "north_america" | "north_america_32km" => format!(
            "nam.t{:02}z.awip32{:02}.tm00.grib2",
            request.cycle.hour_utc, request.forecast_hour
        ),
        "awip3d" | "3d" | "pressure" | "prs" => format!(
            "nam.t{:02}z.awip3d{:02}.tm00.grib2",
            request.cycle.hour_utc, request.forecast_hour
        ),
        "conusnest" | "nest_conus" | "namnest_conus" => format!(
            "nam.t{:02}z.conusnest.hiresf{:02}.tm00.grib2",
            request.cycle.hour_utc, request.forecast_hour
        ),
        "alaskanest" | "nest_alaska" | "aknest" => format!(
            "nam.t{:02}z.alaskanest.hiresf{:02}.tm00.grib2",
            request.cycle.hour_utc, request.forecast_hour
        ),
        "hawaiinest" | "nest_hawaii" | "hinest" => format!(
            "nam.t{:02}z.hawaiinest.hiresf{:02}.tm00.grib2",
            request.cycle.hour_utc, request.forecast_hour
        ),
        "priconest" | "nest_pr" | "prnest" => format!(
            "nam.t{:02}z.priconest.hiresf{:02}.tm00.grib2",
            request.cycle.hour_utc, request.forecast_hour
        ),
        other => {
            return Err(ModelError::UnsupportedProduct {
                model: request.model,
                product: other.to_string(),
            });
        }
    };
    Ok(match source {
        SourceId::Aws => format!(
            "https://noaa-nam-pds.s3.amazonaws.com/nam.{}/{}",
            request.cycle.date_yyyymmdd, filename
        ),
        SourceId::Nomads => format!(
            "https://nomads.ncep.noaa.gov/pub/data/nccf/com/nam/prod/nam.{}/{}",
            request.cycle.date_yyyymmdd, filename
        ),
        other => unsupported_source(other, request.model),
    })
}

fn build_hiresw_url(source: SourceId, request: &ModelRunRequest) -> Result<String, ModelError> {
    if source != SourceId::Nomads {
        return Ok(unsupported_source(source, request.model));
    }
    let token = normalize_token(&request.product);
    let raw_core = token
        .split('/')
        .next()
        .unwrap_or("arw_2p5km")
        .replace("__", "_");
    let mem2_core = matches!(
        raw_core.as_str(),
        "arw_mem2" | "mem2arw" | "arwmem2" | "arw_mem2_2p5km" | "arwmem2_2p5km"
    );
    let core = match raw_core.as_str() {
        "arw" | "arw_2p5km" | "arw_2_5km" => "arw_2p5km",
        "fv3" | "fv3_2p5km" | "fv3_2_5km" => "fv3_2p5km",
        "arw_mem2" | "mem2arw" | "arwmem2" | "arw_mem2_2p5km" | "arwmem2_2p5km" => "arw_5km",
        other => other,
    };
    let domain = token
        .split(['_', '/'])
        .find(|part| {
            matches!(
                *part,
                "conus"
                    | "conusmem2"
                    | "ak"
                    | "akmem2"
                    | "alaska"
                    | "hi"
                    | "himem2"
                    | "hawaii"
                    | "guam"
                    | "pr"
                    | "prmem2"
            )
        })
        .unwrap_or("conus");
    let domain = match domain {
        "alaska" => "ak",
        "hawaii" => "hi",
        "conus" if mem2_core => "conusmem2",
        "ak" if mem2_core => "akmem2",
        "hi" if mem2_core => "himem2",
        "pr" if mem2_core => "prmem2",
        other => other,
    };
    Ok(format!(
        "https://nomads.ncep.noaa.gov/pub/data/nccf/com/hiresw/prod/hiresw.{}/hiresw.t{:02}z.{}.f{:02}.{}.grib2",
        request.cycle.date_yyyymmdd, request.cycle.hour_utc, core, request.forecast_hour, domain
    ))
}

fn build_href_url(source: SourceId, request: &ModelRunRequest) -> Result<String, ModelError> {
    if source != SourceId::Nomads {
        return Ok(unsupported_source(source, request.model));
    }
    if !forecast_hour_supported(request.model, request.cycle.hour_utc, request.forecast_hour) {
        return Err(ModelError::UnsupportedForecastHour {
            model: request.model,
            cycle_hour: request.cycle.hour_utc,
            forecast_hour: request.forecast_hour,
            reason: "HREF CONUS ensprod files are expected for f01 through f48".to_string(),
        });
    }
    let token = normalize_token(&request.product);
    let product = token
        .split(['_', '/'])
        .find(|part| {
            matches!(
                *part,
                "avrg" | "eas" | "ffri" | "lpmm" | "mean" | "pmmn" | "prob" | "sprd"
            )
        })
        .unwrap_or("sprd");
    let domain = token
        .split(['_', '/'])
        .find(|part| matches!(*part, "conus"))
        .unwrap_or("conus");
    Ok(format!(
        "https://nomads.ncep.noaa.gov/pub/data/nccf/com/href/prod/href.{}/ensprod/href.t{:02}z.{}.{}.f{:02}.grib2",
        request.cycle.date_yyyymmdd, request.cycle.hour_utc, domain, product, request.forecast_hour
    ))
}

fn build_sref_url(source: SourceId, request: &ModelRunRequest) -> Result<String, ModelError> {
    if source != SourceId::Nomads {
        return Ok(unsupported_source(source, request.model));
    }
    let token = normalize_token(&request.product);
    if token.starts_with("ensprod") || token.contains("mean") || token.contains("spread") {
        let grid = token
            .split(['_', '/'])
            .find(|part| part.starts_with("pgrb"))
            .unwrap_or("pgrb212");
        let stat = token
            .split(['_', '/'])
            .find(|part| {
                matches!(
                    *part,
                    "mean"
                        | "max"
                        | "min"
                        | "p10"
                        | "p25"
                        | "p50"
                        | "p75"
                        | "p90"
                        | "spread"
                        | "prob"
                )
            })
            .unwrap_or("mean");
        let cadence = if token.contains("1hrly") {
            "1hrly"
        } else {
            "3hrly"
        };
        return Ok(format!(
            "https://nomads.ncep.noaa.gov/pub/data/nccf/com/sref/prod/sref.{}/{:02}/ensprod/sref.t{:02}z.{}.{}_{}.grib2",
            request.cycle.date_yyyymmdd,
            request.cycle.hour_utc,
            request.cycle.hour_utc,
            grid,
            stat,
            cadence
        ));
    }
    let member_family = token
        .split(['_', '/'])
        .find(|part| matches!(*part, "arw" | "nmb"))
        .unwrap_or("arw");
    let member = token
        .split(['_', '/'])
        .find(|part| *part == "ctl" || part.starts_with('p') || part.starts_with('n'))
        .unwrap_or("ctl");
    let grid = token
        .split(['_', '/'])
        .find(|part| part.starts_with("pgrb"))
        .unwrap_or("pgrb132");
    Ok(format!(
        "https://nomads.ncep.noaa.gov/pub/data/nccf/com/sref/prod/sref.{}/{:02}/pgrb/sref_{}.t{:02}z.{}.{}.f{:02}.grib2",
        request.cycle.date_yyyymmdd,
        request.cycle.hour_utc,
        member_family,
        request.cycle.hour_utc,
        grid,
        member,
        request.forecast_hour
    ))
}

fn build_rtma_url(source: SourceId, request: &ModelRunRequest) -> Result<String, ModelError> {
    if source != SourceId::Nomads {
        return Ok(unsupported_source(source, request.model));
    }
    let product = match normalize_token(&request.product).as_str() {
        "2dvaranl_ndfd" | "anl" | "analysis" => "2dvaranl_ndfd",
        "2dvarges_ndfd" | "ges" | "guess" => "2dvarges_ndfd",
        "2dvarerr_ndfd" | "err" | "error" => "2dvarerr_ndfd",
        other => {
            return Err(ModelError::UnsupportedProduct {
                model: request.model,
                product: other.to_string(),
            });
        }
    };
    Ok(format!(
        "https://nomads.ncep.noaa.gov/pub/data/nccf/com/rtma/prod/rtma2p5.{}/rtma2p5.t{:02}z.{}.grb2_wexp",
        request.cycle.date_yyyymmdd, request.cycle.hour_utc, product
    ))
}

fn build_urma_url(source: SourceId, request: &ModelRunRequest) -> Result<String, ModelError> {
    if source != SourceId::Nomads {
        return Ok(unsupported_source(source, request.model));
    }
    let product = match normalize_token(&request.product).as_str() {
        "2dvaranl_ndfd" | "anl" | "analysis" => "2dvaranl_ndfd",
        "2dvarges_ndfd" | "ges" | "guess" => "2dvarges_ndfd",
        "2dvarerr_ndfd" | "err" | "error" => "2dvarerr_ndfd",
        other => {
            return Err(ModelError::UnsupportedProduct {
                model: request.model,
                product: other.to_string(),
            });
        }
    };
    Ok(format!(
        "https://nomads.ncep.noaa.gov/pub/data/nccf/com/urma/prod/urma2p5.{}/urma2p5.t{:02}z.{}.grb2_wexp",
        request.cycle.date_yyyymmdd, request.cycle.hour_utc, product
    ))
}

fn build_nbm_url(source: SourceId, request: &ModelRunRequest) -> Result<String, ModelError> {
    let token = normalize_token(&request.product);
    let stream = token
        .split(['_', '/'])
        .find(|part| {
            matches!(
                *part,
                "core" | "qmd" | "ak" | "co" | "hi" | "pr" | "gu" | "oc" | "global"
            )
        })
        .unwrap_or("core");
    let domain = token
        .split(['_', '/'])
        .find(|part| {
            matches!(
                *part,
                "ak" | "co" | "conus" | "hi" | "hawaii" | "pr" | "gu" | "guam" | "oc" | "global"
            )
        })
        .unwrap_or("co");
    let domain = match domain {
        "conus" => "co",
        "hawaii" => "hi",
        "guam" => "gu",
        other => other,
    };
    let filename = format!(
        "blend.t{:02}z.{}.f{:03}.{}.grib2",
        request.cycle.hour_utc, stream, request.forecast_hour, domain
    );
    Ok(match source {
        SourceId::Aws => format!(
            "https://noaa-nbm-grib2-pds.s3.amazonaws.com/blend.{}/{:02}/{}/{}",
            request.cycle.date_yyyymmdd, request.cycle.hour_utc, stream, filename
        ),
        SourceId::Nomads => format!(
            "https://nomads.ncep.noaa.gov/pub/data/nccf/com/blend/prod/blend.{}/{:02}/{}/{}",
            request.cycle.date_yyyymmdd, request.cycle.hour_utc, stream, filename
        ),
        other => unsupported_source(other, request.model),
    })
}

fn build_rrfs_a_url(source: SourceId, request: &ModelRunRequest) -> Result<String, ModelError> {
    if source != SourceId::Aws {
        return Ok(unsupported_source(source, request.model));
    }

    let suffix = match normalize_token(&request.product).as_str() {
        "prs_conus" | "prslev_conus" | "conus" => {
            format!("prslev.3km.f{:03}.conus.grib2", request.forecast_hour)
        }
        "prs_na" | "prslev_na" | "na" => {
            format!("prslev.3km.f{:03}.na.grib2", request.forecast_hour)
        }
        "prs_ak" | "prslev_ak" | "ak" => {
            format!("prslev.3km.f{:03}.ak.grib2", request.forecast_hour)
        }
        "prs_hi" | "prslev_hi" | "hi" => {
            format!("prslev.2p5km.f{:03}.hi.grib2", request.forecast_hour)
        }
        "prs_pr" | "prslev_pr" | "pr" => {
            format!("prslev.2p5km.f{:03}.pr.grib2", request.forecast_hour)
        }
        "subh_hi" | "prs_subh_hi" | "prslev_subh_hi" => {
            format!("prslev.2p5km.subh.f{:03}.hi.grib2", request.forecast_hour)
        }
        "subh_pr" | "prs_subh_pr" | "prslev_subh_pr" => {
            format!("prslev.2p5km.subh.f{:03}.pr.grib2", request.forecast_hour)
        }
        "nat_na" | "natlev_na" => format!("natlev.3km.f{:03}.na.grib2", request.forecast_hour),
        other => {
            return Err(ModelError::UnsupportedProduct {
                model: request.model,
                product: other.to_string(),
            });
        }
    };

    Ok(format!(
        "https://noaa-rrfs-pds.s3.amazonaws.com/rrfs_a/rrfs.{}/{:02}/rrfs.t{:02}z.{}",
        request.cycle.date_yyyymmdd, request.cycle.hour_utc, request.cycle.hour_utc, suffix
    ))
}

fn build_rrfs_conus_url(
    source: SourceId,
    request: &ModelRunRequest,
) -> Result<String, ModelError> {
    let Some((_, _, base)) = RRFS_CONUS_BASES.iter().find(|(model, origin, _)|
        *model == request.model && *origin == source) else {
        return Ok(unsupported_source(source, request.model));
    };
    if !forecast_hour_supported(request.model, request.cycle.hour_utc, request.forecast_hour) {
        return Err(ModelError::UnsupportedForecastHour {
            model: request.model,
            cycle_hour: request.cycle.hour_utc,
            forecast_hour: request.forecast_hour,
            reason: "The requested RRFS CONUS cycle and hour are outside its declared provider schedule".into(),
        });
    }

    let suffix = match normalize_token(&request.product).as_str() {
        "prs_conus" | "prslev_conus" | "conus" => {
            format!("prslev.3km.f{:03}.conus.grib2", request.forecast_hour)
        }
        "2dfld_conus" | "sfc_conus" | "surface_conus" | "surface" | "sfc" => {
            format!("2dfld.3km.f{:03}.conus.grib2", request.forecast_hour)
        }
        other => {
            return Err(ModelError::UnsupportedProduct {
                model: request.model,
                product: other.to_string(),
            });
        }
    };

    Ok(format!(
        "{base}/rrfs.{}/{:02}/rrfs.t{:02}z.{}",
        request.cycle.date_yyyymmdd, request.cycle.hour_utc, request.cycle.hour_utc, suffix
    ))
}

fn build_refs_url(source: SourceId, request: &ModelRunRequest) -> Result<String, ModelError> {
    if source != SourceId::Aws {
        return Ok(unsupported_source(source, request.model));
    }

    let token = normalize_token(&request.product);
    let mut product_kind = None;
    let mut domain = if token.contains("puerto_rico") {
        Some("pr")
    } else {
        None
    };

    for part in token.split(['/', '_']) {
        match part {
            "sprd" | "spread" => product_kind = Some("sprd"),
            "prob" | "probability" => product_kind = Some("prob"),
            "mean" => product_kind = Some("mean"),
            "avrg" | "avg" | "average" => product_kind = Some("avrg"),
            "eas" => product_kind = Some("eas"),
            "lpmm" => product_kind = Some("lpmm"),
            "pmmn" | "pmm" | "probabilitymatchedmean" => product_kind = Some("pmmn"),
            "ffri" => product_kind = Some("ffri"),
            "conus" => domain = Some("conus"),
            "ak" | "alaska" => domain = Some("ak"),
            "hi" | "hawaii" => domain = Some("hi"),
            "pr" => domain = Some("pr"),
            _ => {}
        }
    }

    let product_kind = product_kind.ok_or_else(|| ModelError::UnsupportedProduct {
        model: request.model,
        product: token.clone(),
    })?;
    let domain = domain.unwrap_or("conus");

    if product_kind == "ffri" && domain != "conus" {
        return Err(ModelError::UnsupportedProduct {
            model: request.model,
            product: token,
        });
    }

    Ok(format!(
        "https://noaa-rrfs-pds.s3.amazonaws.com/rrfs_public/refs.{}/{:02}/enspost/refs.t{:02}z.{}.f{:02}.{}.grib2",
        request.cycle.date_yyyymmdd,
        request.cycle.hour_utc,
        request.cycle.hour_utc,
        product_kind,
        request.forecast_hour,
        domain
    ))
}

fn build_rrfs_firewx_url(
    source: SourceId,
    request: &ModelRunRequest,
) -> Result<String, ModelError> {
    if source != SourceId::Aws {
        return Ok(unsupported_source(source, request.model));
    }

    let suffix = match normalize_token(&request.product).as_str() {
        "prs_firewx" | "prslev_firewx" | "firewx_prs" | "firewx_pressure" | "pressure" => {
            format!(
                "prslev.1p5km.f{:03}.firewx_lcc.grib2",
                request.forecast_hour
            )
        }
        "2dfld_firewx" | "firewx" | "firewx_sfc" | "firewx_surface" | "surface" | "sfc" => {
            format!("2dfld.1p5km.f{:03}.firewx_lcc.grib2", request.forecast_hour)
        }
        other => {
            return Err(ModelError::UnsupportedProduct {
                model: request.model,
                product: other.to_string(),
            });
        }
    };

    Ok(format!(
        "https://noaa-rrfs-pds.s3.amazonaws.com/rrfs_public/firewx.{}/{:02}/rrfs.t{:02}z.{}",
        request.cycle.date_yyyymmdd, request.cycle.hour_utc, request.cycle.hour_utc, suffix
    ))
}

fn normalize_token(value: &str) -> String {
    value
        .trim()
        .to_ascii_lowercase()
        .replace(['-', ' ', '.'], "_")
}

fn canonical_recipe_token(value: &str) -> String {
    let normalized = normalize_token(value);
    match normalized.as_str() {
        "500mb_vorticity_height_winds" => "500mb_absolute_vorticity_height_winds".to_string(),
        "700mb_vorticity_height_winds" => "700mb_absolute_vorticity_height_winds".to_string(),
        "850mb_vorticity_height_winds" => "850mb_absolute_vorticity_height_winds".to_string(),
        // The spellings a person reaching for "the wind" actually types.
        // `normalize_token` has already folded hyphens and spaces to
        // underscores, so `10m-wind-speed-and-direction` and
        // `10 m wind` arrive here in this form too.
        "10m_wind" | "10m_winds" | "wind_10m" | "10m_wind_speed" | "surface_wind"
        | "surface_winds" => "10m_wind_speed_and_direction".to_string(),
        _ => normalized,
    }
}

fn plot_recipe_fetch_plan_for(
    recipe: &'static PlotRecipe,
    model: ModelId,
) -> Result<PlotRecipeFetchPlan, ModelError> {
    let fields = collect_recipe_fields(recipe);
    let blockers = plot_recipe_fetch_blockers_for_fields(&fields, model);
    if !blockers.is_empty() {
        return Err(ModelError::UnsupportedPlotRecipeModel {
            recipe: recipe.slug,
            model,
            reason: summarize_plot_recipe_blockers(&blockers),
        });
    }

    let (product, fetch_policy) = plot_recipe_fetch_defaults(model, &fields);

    Ok(PlotRecipeFetchPlan {
        recipe_slug: recipe.slug,
        model,
        product,
        fetch_policy,
        fetch_mode: fetch_policy.fetch_mode(),
        fields,
    })
}

fn plot_recipe_fetch_blockers_for(
    recipe: &'static PlotRecipe,
    model: ModelId,
) -> Vec<PlotRecipeBlocker> {
    let fields = collect_recipe_fields(recipe);
    plot_recipe_fetch_blockers_for_fields(&fields, model)
}

fn plot_recipe_fetch_blockers_for_fields(
    fields: &[&'static GribFieldSpec],
    model: ModelId,
) -> Vec<PlotRecipeBlocker> {
    fields
        .iter()
        .copied()
        .filter_map(|field| plot_recipe_field_blocker(field, model))
        .collect()
}

fn plot_recipe_field_blocker(
    field: &'static GribFieldSpec,
    model: ModelId,
) -> Option<PlotRecipeBlocker> {
    if field.family == ProductFamily::Native {
        if let Some(reason) = native_field_gap_reason(field, model) {
            return Some(PlotRecipeBlocker {
                field_key: field.key,
                field_label: field.label,
                reason,
            });
        }

        let reason = match field.selector {
            Some(selector) if selector_supported_for_model(selector, model) => return None,
            Some(selector) => unsupported_selector_reason(selector, model),
            None => field_selector_gap_reason(field).to_string(),
        };

        return Some(PlotRecipeBlocker {
            field_key: field.key,
            field_label: field.label,
            reason,
        });
    }

    if field.family == ProductFamily::Pressure {
        if let Some(reason) = model_specific_pressure_field_gap(field, model) {
            return Some(PlotRecipeBlocker {
                field_key: field.key,
                field_label: field.label,
                reason,
            });
        }
    }

    if field.family == ProductFamily::Surface {
        if let Some(reason) = model_specific_surface_field_gap(field, model) {
            return Some(PlotRecipeBlocker {
                field_key: field.key,
                field_label: field.label,
                reason,
            });
        }
    }

    let reason = match field.selector {
        Some(selector) if selector_supported_for_model(selector, model) => return None,
        Some(selector) => unsupported_selector_reason(selector, model),
        None => field_selector_gap_reason(field).to_string(),
    };

    Some(PlotRecipeBlocker {
        field_key: field.key,
        field_label: field.label,
        reason,
    })
}

fn plot_recipe_fetch_defaults(
    model: ModelId,
    fields: &[&'static GribFieldSpec],
) -> (&'static str, PlotRecipeFetchPolicy) {
    if let Some(product) = wrf_gdex_recipe_product_override(model, fields) {
        return (product, PlotRecipeFetchPolicy::WholeFile);
    }
    let has_native = fields
        .iter()
        .any(|field| field.family == ProductFamily::Native);
    let has_surface = fields
        .iter()
        .any(|field| field.family == ProductFamily::Surface);
    let has_product_selector = fields.iter().any(|field| {
        field
            .selector
            .is_some_and(|selector| !selector.product.is_default())
    });
    let has_probability_selector = fields.iter().any(|field| {
        field
            .selector
            .is_some_and(|selector| matches!(selector.product, FieldProduct::Probability(_)))
    });
    let has_ensemble_spread_selector = fields.iter().any(|field| {
        field.selector.is_some_and(|selector| {
            matches!(
                selector.product,
                FieldProduct::EnsembleStandardDeviation | FieldProduct::EnsembleSpread
            )
        })
    });
    let has_ensemble_mean_selector = fields.iter().any(|field| {
        field
            .selector
            .is_some_and(|selector| matches!(selector.product, FieldProduct::EnsembleMean))
    });
    match (model, has_native, has_surface) {
        (ModelId::Hrrr, true, _) => ("nat", PlotRecipeFetchPolicy::PreferIndexedSubset),
        (ModelId::Hrrr, false, true) => ("sfc", PlotRecipeFetchPolicy::PreferIndexedSubset),
        (ModelId::Hrrr, false, false) => ("prs", PlotRecipeFetchPolicy::PreferIndexedSubset),
        (ModelId::HrrrAk, true, _) => ("nat", PlotRecipeFetchPolicy::PreferIndexedSubset),
        (ModelId::HrrrAk, false, true) => ("sfc", PlotRecipeFetchPolicy::PreferIndexedSubset),
        (ModelId::HrrrAk, false, false) => ("prs", PlotRecipeFetchPolicy::PreferIndexedSubset),
        (ModelId::Gfs, _, _) => ("pgrb2.0p25", PlotRecipeFetchPolicy::PreferIndexedSubset),
        (ModelId::Gdas, _, _) => ("pgrb2.0p25", PlotRecipeFetchPolicy::PreferIndexedSubset),
        (ModelId::Gefs, _, _) if has_ensemble_spread_selector => {
            ("pgrb2ap5/gespr", PlotRecipeFetchPolicy::PreferIndexedSubset)
        }
        (ModelId::Gefs, _, _) if has_ensemble_mean_selector => {
            ("pgrb2ap5/geavg", PlotRecipeFetchPolicy::PreferIndexedSubset)
        }
        (ModelId::Gefs, _, _) => ("pgrb2ap5/gec00", PlotRecipeFetchPolicy::PreferIndexedSubset),
        (ModelId::Aigfs, _, true) => ("sfc", PlotRecipeFetchPolicy::PreferIndexedSubset),
        (ModelId::Aigfs, _, false) => ("pres", PlotRecipeFetchPolicy::PreferIndexedSubset),
        (ModelId::Aigefs, _, true) if has_ensemble_spread_selector => {
            ("sfc/spr", PlotRecipeFetchPolicy::PreferIndexedSubset)
        }
        (ModelId::Aigefs, _, false) if has_ensemble_spread_selector => {
            ("pres/spr", PlotRecipeFetchPolicy::PreferIndexedSubset)
        }
        (ModelId::Aigefs, _, true) if has_ensemble_mean_selector => ("sfc/avg", PlotRecipeFetchPolicy::PreferIndexedSubset),
        (ModelId::Aigefs, _, false) if has_ensemble_mean_selector => ("pres/avg", PlotRecipeFetchPolicy::PreferIndexedSubset),
        (ModelId::Aigefs, _, true) => ("sfc/mem000", PlotRecipeFetchPolicy::PreferIndexedSubset),
        (ModelId::Aigefs, _, false) => ("pres/mem000", PlotRecipeFetchPolicy::PreferIndexedSubset),
        (ModelId::Hgefs, _, true) if has_ensemble_spread_selector => {
            ("sfc/spr", PlotRecipeFetchPolicy::PreferIndexedSubset)
        }
        (ModelId::Hgefs, _, false) if has_ensemble_spread_selector => {
            ("pres/spr", PlotRecipeFetchPolicy::PreferIndexedSubset)
        }
        (ModelId::Hgefs, _, true) => ("sfc/avg", PlotRecipeFetchPolicy::PreferIndexedSubset),
        (ModelId::Hgefs, _, false) => ("pres/avg", PlotRecipeFetchPolicy::PreferIndexedSubset),
        (ModelId::Rap, _, _) => ("awp130pgrb", PlotRecipeFetchPolicy::PreferIndexedSubset),
        (ModelId::Nam, _, _) => ("awip12", PlotRecipeFetchPolicy::PreferIndexedSubset),
        (ModelId::Hiresw, _, _) => (
            "arw_2p5km/conus",
            PlotRecipeFetchPolicy::PreferIndexedSubset,
        ),
        (ModelId::Href, _, _) if has_ensemble_mean_selector => (
            "ensprod/conus/mean",
            PlotRecipeFetchPolicy::PreferIndexedSubset,
        ),
        (ModelId::Href, _, _) if has_probability_selector => (
            "ensprod/conus/prob",
            PlotRecipeFetchPolicy::PreferIndexedSubset,
        ),
        (ModelId::Href, _, _) if has_ensemble_spread_selector => (
            "ensprod/conus/sprd",
            PlotRecipeFetchPolicy::PreferIndexedSubset,
        ),
        (ModelId::Href, _, _) => (
            "ensprod/conus/sprd",
            PlotRecipeFetchPolicy::PreferIndexedSubset,
        ),
        (ModelId::Sref, _, _) if has_probability_selector => (
            "ensprod/pgrb212/prob_3hrly",
            PlotRecipeFetchPolicy::PreferIndexedSubset,
        ),
        (ModelId::Sref, _, _) => (
            "arw/ctl/pgrb132",
            PlotRecipeFetchPolicy::PreferIndexedSubset,
        ),
        (ModelId::Refs, _, _) if has_probability_selector => {
            ("prob-conus", PlotRecipeFetchPolicy::PreferIndexedSubset)
        }
        (ModelId::Refs, _, _) if has_ensemble_spread_selector => {
            ("sprd-conus", PlotRecipeFetchPolicy::PreferIndexedSubset)
        }
        (ModelId::Refs, true, _) => ("pmmn-conus", PlotRecipeFetchPolicy::PreferIndexedSubset),
        (ModelId::Refs, _, _) => ("mean-conus", PlotRecipeFetchPolicy::PreferIndexedSubset),
        (ModelId::Rtma, _, _) => ("2dvaranl_ndfd", PlotRecipeFetchPolicy::PreferIndexedSubset),
        (ModelId::Urma, _, _) => ("2dvaranl_ndfd", PlotRecipeFetchPolicy::PreferIndexedSubset),
        (ModelId::Nbm, _, _) if has_product_selector => {
            ("qmd/co", PlotRecipeFetchPolicy::PreferIndexedSubset)
        }
        (ModelId::Nbm, _, _) => ("core/co", PlotRecipeFetchPolicy::PreferIndexedSubset),
        (ModelId::RrfsA, true, _) => ("prs-conus", PlotRecipeFetchPolicy::PreferIndexedSubset),
        (ModelId::RrfsA, false, true) => ("nat-na", PlotRecipeFetchPolicy::PreferIndexedSubset),
        (ModelId::RrfsA, false, false) => ("prs-conus", PlotRecipeFetchPolicy::PreferIndexedSubset),
        (ModelId::Rrfs | ModelId::RrfsPublic, true, _) => {
            ("2dfld-conus", PlotRecipeFetchPolicy::PreferIndexedSubset)
        }
        (ModelId::Rrfs | ModelId::RrfsPublic, false, true) => {
            ("2dfld-conus", PlotRecipeFetchPolicy::PreferIndexedSubset)
        }
        (ModelId::Rrfs | ModelId::RrfsPublic, false, false) => {
            ("prs-conus", PlotRecipeFetchPolicy::PreferIndexedSubset)
        }
        (ModelId::RrfsFireWx, true, _) => {
            ("2dfld-firewx", PlotRecipeFetchPolicy::PreferIndexedSubset)
        }
        (ModelId::RrfsFireWx, false, true) => {
            ("2dfld-firewx", PlotRecipeFetchPolicy::PreferIndexedSubset)
        }
        (ModelId::RrfsFireWx, false, false) => {
            ("prs-firewx", PlotRecipeFetchPolicy::PreferIndexedSubset)
        }
        (ModelId::EcmwfOpenData, _, _) => ("oper", PlotRecipeFetchPolicy::WholeFile),
        (ModelId::Aifs, _, _) => ("oper", PlotRecipeFetchPolicy::WholeFile),
        (ModelId::WrfGdex, true, _) => (
            WRF_GDEX_DEFAULT_PRESSURE_PRODUCT,
            PlotRecipeFetchPolicy::WholeFile,
        ),
        (ModelId::WrfGdex, false, true) => (
            WRF_GDEX_DEFAULT_SURFACE_PRODUCT,
            PlotRecipeFetchPolicy::WholeFile,
        ),
        (ModelId::WrfGdex, false, false) => (
            WRF_GDEX_DEFAULT_PRESSURE_PRODUCT,
            PlotRecipeFetchPolicy::WholeFile,
        ),
    }
}

fn wrf_gdex_recipe_product_override(
    model: ModelId,
    fields: &[&'static GribFieldSpec],
) -> Option<&'static str> {
    if model != ModelId::WrfGdex {
        return None;
    }
    let all_native_surface_diagnostics = !fields.is_empty()
        && fields.iter().all(|field| {
            matches!(
                field.key,
                "composite_reflectivity" | "radar_reflectivity_1km_agl"
            )
        });
    if all_native_surface_diagnostics {
        return Some(WRF_GDEX_DEFAULT_SURFACE_PRODUCT);
    }
    if fields.iter().any(|field| field.key == "updraft_helicity") {
        return Some(WRF_GDEX_DEFAULT_PRESSURE_PRODUCT);
    }
    let all_cloud_cover = !fields.is_empty()
        && fields.iter().all(|field| {
            matches!(
                field.key,
                "total_cloud_cover" | "low_cloud_cover" | "middle_cloud_cover" | "high_cloud_cover"
            )
        });
    if all_cloud_cover {
        return Some(WRF_GDEX_DEFAULT_PRESSURE_PRODUCT);
    }
    None
}

fn native_field_gap_reason(field: &GribFieldSpec, model: ModelId) -> Option<String> {
    match (field.key, model) {
        (key, ModelId::Href)
            if !key.starts_with("href_spread_")
                && !key.starts_with("href_probability_")
                && !key.starts_with("href_mean_") =>
        {
            Some(format!(
                "{} is not yet wired for HREF; HREF support is currently limited to explicit `href_sprd_*`, `href_prob_*`, and `href_mean_*` ensprod recipes",
                field.label
            ))
        }
        (key, model) if key.starts_with("href_mean_") && model != ModelId::Href => Some(format!(
            "{} is only verified for HREF ensprod mean fields right now; do not route this recipe through model '{model}'",
            field.label
        )),
        (key, model) if key.starts_with("href_probability_") && model != ModelId::Href => {
            Some(format!(
                "{} is only verified for HREF ensprod probability fields right now; do not route this recipe through model '{model}'",
                field.label
            ))
        }
        (key, model)
            if (key.starts_with("refs_spread_") || key.starts_with("refs_probability_"))
                && model != ModelId::Refs =>
        {
            Some(format!(
                "{} is only verified for REFS enspost fields right now; do not route this recipe through model '{model}'",
                field.label
            ))
        }
        (
            "composite_reflectivity" | "radar_reflectivity_1km_agl" | "updraft_helicity",
            ModelId::Gfs
            | ModelId::Gdas
            | ModelId::Gefs
            | ModelId::Aigfs
            | ModelId::Aigefs
            | ModelId::EcmwfOpenData
            | ModelId::Aifs
            | ModelId::Rap
            | ModelId::Nam
            | ModelId::Hiresw
            | ModelId::Href
            | ModelId::Sref
            | ModelId::Rtma
            | ModelId::Urma
            | ModelId::Nbm,
        ) => Some(format!(
            "{} is not wired for model '{model}'; rustwx-models only has native convective product fetch planning for HRRR/RRFS-A right now",
            field.label
        )),
        (
            "smoke_mass_density_8m_agl" | "column_integrated_smoke",
            ModelId::Gfs
            | ModelId::Gdas
            | ModelId::Gefs
            | ModelId::Aigfs
            | ModelId::Aigefs
            | ModelId::EcmwfOpenData
            | ModelId::Aifs
            | ModelId::Rap
            | ModelId::Nam
            | ModelId::Hiresw
            | ModelId::Href
            | ModelId::Sref
            | ModelId::Rtma
            | ModelId::Urma
            | ModelId::Nbm,
        ) => Some(format!(
            "{} is only verified and wired for HRRR wrfnat right now; the native smoke GRIB signature is not verified yet for model '{model}'",
            field.label
        )),
        (
            "simulated_infrared_brightness_temperature",
            ModelId::Gfs
            | ModelId::Gdas
            | ModelId::Gefs
            | ModelId::Aigfs
            | ModelId::Aigefs
            | ModelId::EcmwfOpenData
            | ModelId::Aifs
            | ModelId::Rap
            | ModelId::Nam
            | ModelId::Hiresw
            | ModelId::Href
            | ModelId::Sref
            | ModelId::Rtma
            | ModelId::Urma
            | ModelId::Nbm
            | ModelId::RrfsA
            | ModelId::Rrfs | ModelId::RrfsPublic
            | ModelId::RrfsFireWx,
        ) => Some(format!(
            "{} is only verified and wired for HRRR right now; the native GRIB signature is not verified yet for model '{model}'",
            field.label
        )),
        (
            "smoke_mass_density_8m_agl" | "column_integrated_smoke",
            ModelId::RrfsA | ModelId::Rrfs | ModelId::RrfsPublic | ModelId::RrfsFireWx,
        ) => Some(format!(
            "{} is only verified and wired for HRRR wrfnat right now; the native GRIB signature is not verified yet for model '{model}'",
            field.label
        )),
        _ => None,
    }
}

fn model_specific_pressure_field_gap(field: &GribFieldSpec, model: ModelId) -> Option<String> {
    match (model, field.key) {
        (model, key)
            if (key.starts_with("refs_spread_") || key.starts_with("refs_probability_"))
                && model != ModelId::Refs =>
        {
            Some(format!(
                "{} is only verified for REFS enspost fields right now; do not route this recipe through model '{model}'",
                field.label
            ))
        }
        (ModelId::Href, key)
            if !key.starts_with("href_spread_")
                && !key.starts_with("href_probability_")
                && !key.starts_with("href_mean_") =>
        {
            Some(format!(
                "{} is not yet wired for HREF; HREF support is currently limited to explicit `href_sprd_*`, `href_prob_*`, and `href_mean_*` ensprod recipes",
                field.label
            ))
        }
        (model, key) if key.starts_with("href_mean_") && model != ModelId::Href => Some(format!(
            "{} is only verified for HREF ensprod mean fields right now; do not route this recipe through model '{model}'",
            field.label
        )),
        (model, key) if key.starts_with("href_probability_") && model != ModelId::Href => {
            Some(format!(
                "{} is only verified for HREF ensprod probability fields right now; do not route this recipe through model '{model}'",
                field.label
            ))
        }
        (model, key) if key.starts_with("href_spread_") && model != ModelId::Href => Some(format!(
            "{} is only verified for HREF ensprod spread fields right now; do not route this recipe through model '{model}'",
            field.label
        )),
        (model, key)
            if (key.starts_with("gefs_mean_") || key.starts_with("gefs_spread_"))
                && model != ModelId::Gefs =>
        {
            Some(format!(
                "{} is only verified for GEFS native geavg/gespr fields right now; do not route this recipe through model '{model}'",
                field.label
            ))
        }
        (model, key) if key.starts_with("aigefs_spread_") && model != ModelId::Aigefs => {
            Some(format!(
                "{} is only verified for AI-GEFS ensstat spread fields right now; do not route this recipe through model '{model}'",
                field.label
            ))
        }
        (model, key) if key.starts_with("hgefs_spread_") && model != ModelId::Hgefs => {
            Some(format!(
                "{} is only verified for HGEFS ensstat spread fields right now; do not route this recipe through model '{model}'",
                field.label
            ))
        }
        (model, key) if key.starts_with("sref_probability_") && model != ModelId::Sref => {
            Some(format!(
                "{} is only verified for SREF ensprod probability fields right now; do not route this recipe through model '{model}'",
                field.label
            ))
        }
        (ModelId::Rtma | ModelId::Urma | ModelId::Nbm, _) => Some(format!(
            "{} requires an isobaric-pressure product; model '{model}' is currently wired only through surface/core grids in rustwx v0.5",
            field.label
        )),
        (ModelId::Gfs, "dewpoint_700mb" | "dewpoint_850mb") => Some(format!(
            "{} is not present in the GFS 0.25-degree pgrb2 file currently wired by rustwx-models; keep it blocked until a verified direct field or derived dewpoint path is implemented",
            field.label
        )),
        (
            ModelId::Gdas
            | ModelId::Aigfs
            | ModelId::Aigefs
            | ModelId::Rap
            | ModelId::Nam
            | ModelId::Hiresw
            | ModelId::Sref,
            "dewpoint_700mb" | "dewpoint_850mb",
        ) => Some(format!(
            "{} is not verified as a direct pressure-level dewpoint field for model '{model}'; use RH/TMP or add a model-specific derived dewpoint path",
            field.label
        )),
        (ModelId::Gefs, "dewpoint_700mb" | "dewpoint_850mb") => Some(format!(
            "{} is not present in the GEFS 0.5-degree pgrb2ap5 member files currently wired by rustwx-models; use RH/TMP or add derived dewpoint support for this model",
            field.label
        )),
        (ModelId::EcmwfOpenData, "dewpoint_700mb" | "dewpoint_850mb") => Some(format!(
            "{} is not present in the ECMWF open-data 'oper' pressure product currently wired by rustwx-models; use RH/TMP or add derived dewpoint support for this model",
            field.label
        )),
        (
            ModelId::EcmwfOpenData,
            "absolute_vorticity_200mb"
            | "absolute_vorticity_300mb"
            | "absolute_vorticity_500mb"
            | "absolute_vorticity_700mb"
            | "absolute_vorticity_850mb",
        ) => Some(format!(
            "{} is not present in the ECMWF open-data 'oper' pressure product currently wired by rustwx-models",
            field.label
        )),
        (
            ModelId::Aifs,
            "absolute_vorticity_200mb"
            | "absolute_vorticity_300mb"
            | "absolute_vorticity_500mb"
            | "absolute_vorticity_700mb"
            | "absolute_vorticity_850mb",
        ) => Some(format!(
            "{} is not present in the Earth2Archive AIFS NetCDF schema currently wired by rustwx-models",
            field.label
        )),
        (
            ModelId::Aigfs
            | ModelId::Aigefs
            | ModelId::Rap
            | ModelId::Nam
            | ModelId::Hiresw
            | ModelId::Sref,
            "absolute_vorticity_200mb"
            | "absolute_vorticity_300mb"
            | "absolute_vorticity_500mb"
            | "absolute_vorticity_700mb"
            | "absolute_vorticity_850mb",
        ) => Some(format!(
            "{} is not verified for model '{model}' in the v0.5 compatibility path; keep vorticity recipes conservative until the GRIB signature is checked",
            field.label
        )),
        _ => None,
    }
}

fn model_specific_surface_field_gap(field: &GribFieldSpec, model: ModelId) -> Option<String> {
    match (model, field.key) {
        (model, key)
            if (key.starts_with("refs_spread_") || key.starts_with("refs_probability_"))
                && model != ModelId::Refs =>
        {
            Some(format!(
                "{} is only verified for REFS enspost fields right now; do not route this recipe through model '{model}'",
                field.label
            ))
        }
        (ModelId::Href, key)
            if !key.starts_with("href_spread_")
                && !key.starts_with("href_probability_")
                && !key.starts_with("href_mean_") =>
        {
            Some(format!(
                "{} is not yet wired for HREF; HREF support is currently limited to explicit `href_sprd_*`, `href_prob_*`, and `href_mean_*` ensprod recipes",
                field.label
            ))
        }
        (model, key) if key.starts_with("href_mean_") && model != ModelId::Href => {
            Some(format!(
                "{} is only verified for HREF ensprod mean fields right now; do not route this recipe through model '{model}'",
                field.label
            ))
        }
        (model, key) if key.starts_with("href_probability_") && model != ModelId::Href => {
            Some(format!(
                "{} is only verified for HREF ensprod probability fields right now; do not route this recipe through model '{model}'",
                field.label
            ))
        }
        (model, key) if key.starts_with("href_spread_") && model != ModelId::Href => {
            Some(format!(
                "{} is only verified for HREF ensprod spread fields right now; do not route this recipe through model '{model}'",
                field.label
            ))
        }
        (model, key)
            if (key.starts_with("gefs_mean_") || key.starts_with("gefs_spread_"))
                && model != ModelId::Gefs =>
        {
            Some(format!(
                "{} is only verified for GEFS native geavg/gespr fields right now; do not route this recipe through model '{model}'",
                field.label
            ))
        }
        (model, key) if key.starts_with("aigefs_spread_") && model != ModelId::Aigefs => {
            Some(format!(
                "{} is only verified for AI-GEFS ensstat spread fields right now; do not route this recipe through model '{model}'",
                field.label
            ))
        }
        (model, key) if key.starts_with("hgefs_spread_") && model != ModelId::Hgefs => {
            Some(format!(
                "{} is only verified for HGEFS ensstat spread fields right now; do not route this recipe through model '{model}'",
                field.label
            ))
        }
        (model, key) if key.starts_with("sref_probability_") && model != ModelId::Sref => {
            Some(format!(
                "{} is only verified for SREF ensprod probability fields right now; do not route this recipe through model '{model}'",
                field.label
            ))
        }
        (ModelId::Hrrr, "theta_e_2m_agl") => Some(
            "2m Theta-e is surface-derived rather than native; HRRR exposes it through the derived product 'theta_e_2m_10m_winds' (legacy plot-recipe slug '2m_theta_e_10m_winds'), not as a direct/native GRIB recipe.".to_string(),
        ),
        (_, "theta_e_2m_agl") => Some(
            "2m Theta-e is surface-derived rather than native; the direct/native recipe registry does not yet wire the required PSFC/T2/SPFH/U10/V10 dependency bundle into one renderable product".to_string(),
        ),
        (ModelId::Hrrr, "heat_index_2m_agl") => Some(
            "2m Heat Index is surface-derived rather than native; HRRR exposes it through the derived product 'heat_index_2m' (legacy plot-recipe slug '2m_heat_index'), not as a direct/native GRIB recipe.".to_string(),
        ),
        (_, "heat_index_2m_agl") => Some(
            "2m Heat Index is surface-derived rather than native; the direct/native recipe registry does not yet wire the required T2/SPFH/U10/V10 dependency bundle into one renderable product".to_string(),
        ),
        (ModelId::Hrrr, "wind_chill_2m_agl") => Some(
            "2m Wind Chill is surface-derived rather than native; HRRR exposes it through the derived product 'wind_chill_2m' (legacy plot-recipe slug '2m_wind_chill'), not as a direct/native GRIB recipe.".to_string(),
        ),
        (_, "wind_chill_2m_agl") => Some(
            "2m Wind Chill is surface-derived rather than native; the direct/native recipe registry does not yet wire the required T2/U10/V10 dependency bundle into one renderable product".to_string(),
        ),

        (ModelId::Aifs, "wind_gust_10m_agl" | "visibility_surface") => Some(format!(
            "{} is not present in the current Earth2Archive AIFS NetCDF schema",
            field.label
        )),
        (
            ModelId::Aifs,
            "low_cloud_cover" | "middle_cloud_cover" | "high_cloud_cover",
        ) => Some(format!(
            "{} is not present in the current Earth2Archive AIFS NetCDF schema; total cloud cover is available as tcc",
            field.label
        )),
        (_, "cloud_cover_levels") => None,
        (ModelId::Hrrr, "one_hour_qpf") => Some(
            "1h QPF is handled accurately in the HRRR windowed lane as 'qpf_1h' (legacy plot-recipe slug '1h_qpf'); do not treat it as a native/direct APCP recipe.".to_string(),
        ),
        (_, "one_hour_qpf") => Some(
            "1h QPF is not yet exposed as a generic native recipe because APCP accumulation windows vary by model and forecast hour.".to_string(),
        ),
        (model, "probability_of_precipitation") if model != ModelId::Nbm => Some(format!(
            "{} is only verified for NBM core APCP probability fields right now; add a model-specific idx/GRIB signature test before exposing it for '{model}'",
            field.label
        )),
        (_, "precipitation_type") => None,
        (_, "lightning_flash_density") => Some(
            "Verified HRRR surface files expose LTNGSD at 1 m and 2 m AGL as discipline 0/category 17/number 0 Lightning Strike Density [m^-2 s^-1], plus LTNG as discipline 0/category 17/number 192 Lightning [non-dim]; HRRR does not expose the flash-density parameters 2/3/4, so wiring this slug would mislabel strike density or a lightning flag.".to_string(),
        ),
        (ModelId::EcmwfOpenData, "simulated_infrared_brightness_temperature") => Some(format!(
            "{} is still a placeholder in rustwx-models for model '{model}'; the GRIB signature is not verified yet",
            field.label
        )),
        _ => None,
    }
}

// NOTE: rustwx-io has a same-named fn with intentionally wider semantics
// (every 25 hPa from 100 to 1000): that one is what extraction can admit;
// this one is what recipe validation/UI exposes.
fn is_supported_upper_air_level(level_hpa: u16) -> bool {
    matches!(level_hpa, 200 | 250 | 300 | 500 | 700 | 850 | 925)
}

fn unsupported_selector_reason(selector: FieldSelector, model: ModelId) -> String {
    format!(
        "selector '{selector}' is not yet supported for model '{model}' by the rustwx registry/extractor path"
    )
}

fn field_selector_gap_reason(_field: &GribFieldSpec) -> &'static str {
    "recipe field does not yet have a rustwx-models FieldSelector binding"
}

fn summarize_plot_recipe_blockers(blockers: &[PlotRecipeBlocker]) -> String {
    let mut grouped = Vec::<(String, Vec<&'static str>)>::new();
    for blocker in blockers {
        if let Some((_, labels)) = grouped
            .iter_mut()
            .find(|(reason, _)| reason == &blocker.reason)
        {
            labels.push(blocker.field_label);
        } else {
            grouped.push((blocker.reason.clone(), vec![blocker.field_label]));
        }
    }

    grouped
        .into_iter()
        .map(|(reason, labels)| format!("{}: {}", labels.join(", "), reason))
        .collect::<Vec<_>>()
        .join("; ")
}

// Model-identity-free by construction: a recipe's field set is a fact
// about the recipe, not about any source model (gpuwm vendor divergence
// dropped the unused ModelId parameter to make that explicit).
fn collect_recipe_fields(recipe: &'static PlotRecipe) -> Vec<&'static GribFieldSpec> {
    let mut fields = match recipe.slug {
        "cloud_cover_levels" => vec![
            &FIELD_LOW_CLOUD_COVER,
            &FIELD_MIDDLE_CLOUD_COVER,
            &FIELD_HIGH_CLOUD_COVER,
        ],
        "precipitation_type" => vec![
            &FIELD_CATEGORICAL_RAIN,
            &FIELD_CATEGORICAL_FREEZING_RAIN,
            &FIELD_CATEGORICAL_ICE_PELLETS,
            &FIELD_CATEGORICAL_SNOW,
        ],
        _ => vec![&recipe.filled],
    };
    if let Some(contours) = &recipe.contours {
        fields.push(contours);
    }
    if let Some(barbs_u) = &recipe.barbs_u {
        fields.push(barbs_u);
    }
    if let Some(barbs_v) = &recipe.barbs_v {
        fields.push(barbs_v);
    }

    let mut deduped = Vec::with_capacity(fields.len());
    for field in fields {
        if !deduped
            .iter()
            .any(|existing: &&GribFieldSpec| existing.key == field.key)
        {
            deduped.push(field);
        }
    }
    deduped
}

fn dedupe_patterns<I>(patterns: I) -> Vec<&'static str>
where
    I: IntoIterator<Item = &'static str>,
{
    let mut out = Vec::new();
    for pattern in patterns {
        if !out.contains(&pattern) {
            out.push(pattern);
        }
    }
    out
}

fn unsupported_source(source: SourceId, model: ModelId) -> String {
    format!("unsupported://{source}/{model}")
}

fn is_supported_hrrr_smoke_hybrid_level(level: u16) -> bool {
    (1..=50).contains(&level)
}

impl fmt::Display for ModelSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.id, self.description)
    }
}

#[cfg(test)]
mod tests;
