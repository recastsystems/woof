use serde::{Deserialize, Serialize};
use thiserror::Error;

const AIFS_MAX_FORECAST_HOUR: u16 = 43_848;

/// Largest horizontal grid accepted by the shared desktop data model. One
/// f32 plane at this ceiling is about 95 MiB; a lat/lon grid plus one field is
/// already about 286 MiB. Rejecting larger metadata before allocation keeps a
/// corrupt local file from turning a shape header into a multi-gigabyte job.
pub const MAX_GRID_CELLS: usize = 25_000_000;

#[derive(Debug, Error)]
pub enum RustwxError {
    #[error("invalid grid shape: nx={nx}, ny={ny}")]
    InvalidGridShape { nx: usize, ny: usize },
    #[error("grid shape nx={nx}, ny={ny} has {cells} cells; supported maximum is {max_cells}")]
    GridTooLarge {
        nx: usize,
        ny: usize,
        cells: usize,
        max_cells: usize,
    },
    #[error("volume shape {levels} levels x {cells} cells exceeds the platform's f32 address space")]
    VolumeAddressSpace { levels: usize, cells: usize },
    #[error("invalid field data length: expected {expected}, got {actual}")]
    InvalidFieldDataLength { expected: usize, actual: usize },
    #[error("unknown model '{0}'")]
    UnknownModel(String),
    #[error("unknown source '{0}'")]
    UnknownSource(String),
    #[error("invalid cycle date '{0}', expected YYYYMMDD")]
    InvalidCycleDate(String),
    #[error("invalid cycle hour {0}, expected 0..23")]
    InvalidCycleHour(u8),
    #[error("invalid UTC timestamp '{0}', expected YYYY-MM-DDTHH:MM:SSZ")]
    InvalidTimeStamp(String),
    #[error("invalid forecast hour {0}")]
    InvalidForecastHour(u16),
    #[error("pressure-level volume requires at least one level")]
    EmptyPressureLevels,
    #[error("invalid pressure level at index {index}: {value}")]
    InvalidPressureLevel { index: usize, value: f32 },
    #[error("hybrid-level volume requires at least one level")]
    EmptyHybridLevels,
    #[error("invalid hybrid level at index {index}: {value}")]
    InvalidHybridLevel { index: usize, value: u16 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct GridShape {
    pub nx: usize,
    pub ny: usize,
}

impl GridShape {
    pub fn new(nx: usize, ny: usize) -> Result<Self, RustwxError> {
        let shape = Self { nx, ny };
        shape.checked_len()?;
        Ok(shape)
    }

    /// Validate this shape and return its cell count. Use this at every
    /// allocation or persistence boundary: `GridShape` keeps public fields for
    /// API compatibility, so Rust callers can still create unchecked literals.
    pub fn checked_len(self) -> Result<usize, RustwxError> {
        if self.nx == 0 || self.ny == 0 {
            return Err(RustwxError::InvalidGridShape {
                nx: self.nx,
                ny: self.ny,
            });
        }
        let cells = self
            .nx
            .checked_mul(self.ny)
            .ok_or(RustwxError::InvalidGridShape {
                nx: self.nx,
                ny: self.ny,
            })?;
        if cells > MAX_GRID_CELLS {
            return Err(RustwxError::GridTooLarge {
                nx: self.nx,
                ny: self.ny,
                cells,
                max_cells: MAX_GRID_CELLS,
            });
        }
        Ok(cells)
    }

    /// Cell count for already-validated shapes. Invalid public struct literals
    /// return zero rather than panicking or wrapping; error-bearing code should
    /// call [`Self::checked_len`] so it can report the malformed dimensions.
    pub fn len(self) -> usize {
        self.checked_len().unwrap_or(0)
    }
}

impl<'de> Deserialize<'de> for GridShape {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct GridShapeWire {
            nx: usize,
            ny: usize,
        }

        let wire = GridShapeWire::deserialize(deserializer)?;
        Self::new(wire.nx, wire.ny).map_err(serde::de::Error::custom)
    }
}

/// Check a dense level-by-cell f32 shape without imposing a data-size ceiling.
/// Actual allocations remain fallible at their owning boundary. Rust buffers
/// must fit both the platform element count and its isize-sized byte offsets.
pub fn checked_volume_elements(levels: usize, cells: usize) -> Result<usize, RustwxError> {
    let invalid = || RustwxError::VolumeAddressSpace { levels, cells };
    let elements = levels.checked_mul(cells).ok_or_else(invalid)?;
    let bytes = elements.checked_mul(std::mem::size_of::<f32>()).ok_or_else(invalid)?;
    if bytes > isize::MAX as usize {
        return Err(invalid());
    }
    Ok(elements)
}

/// Native map-projection metadata carried alongside a lat/lon grid when the
/// upstream source knows the model's actual projection family.
///
/// This is intentionally lightweight: it captures the projection parameters
/// needed to project model footprints and overlays consistently, while keeping
/// the public core model independent of any GRIB-specific parser types.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GridProjection {
    Geographic,
    LambertConformal {
        standard_parallel_1_deg: f64,
        standard_parallel_2_deg: f64,
        central_meridian_deg: f64,
    },
    PolarStereographic {
        true_latitude_deg: f64,
        central_meridian_deg: f64,
        south_pole_on_projection_plane: bool,
    },
    Mercator {
        latitude_of_true_scale_deg: f64,
        central_meridian_deg: f64,
    },
    Other {
        template: u16,
    },
}

impl GridProjection {
    pub fn is_projected(&self) -> bool {
        !matches!(self, Self::Geographic)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct LatLonGrid {
    pub shape: GridShape,
    pub lat_deg: Vec<f32>,
    pub lon_deg: Vec<f32>,
}

impl LatLonGrid {
    pub fn new(
        shape: GridShape,
        lat_deg: Vec<f32>,
        lon_deg: Vec<f32>,
    ) -> Result<Self, RustwxError> {
        // `GridShape` retains public fields for API compatibility, so a Rust
        // caller can still form a literal that bypasses `GridShape::new`.
        // Re-establish the invariant at this allocation-bearing boundary.
        let cells = shape.checked_len()?;
        if lat_deg.len() != cells || lon_deg.len() != cells {
            return Err(RustwxError::InvalidGridShape {
                nx: shape.nx,
                ny: shape.ny,
            });
        }
        Ok(Self {
            shape,
            lat_deg,
            lon_deg,
        })
    }
}

impl<'de> Deserialize<'de> for LatLonGrid {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct LatLonGridWire {
            shape: GridShape,
            lat_deg: Vec<f32>,
            lon_deg: Vec<f32>,
        }

        let wire = LatLonGridWire::deserialize(deserializer)?;
        Self::new(wire.shape, wire.lat_deg, wire.lon_deg).map_err(serde::de::Error::custom)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct GeoPoint {
    pub lat_deg: f64,
    pub lon_deg: f64,
}

impl GeoPoint {
    pub const fn new(lat_deg: f64, lon_deg: f64) -> Self {
        Self { lat_deg, lon_deg }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct GeoBounds {
    pub west_lon_deg: f64,
    pub east_lon_deg: f64,
    pub south_lat_deg: f64,
    pub north_lat_deg: f64,
}

impl GeoBounds {
    pub const fn new(
        west_lon_deg: f64,
        east_lon_deg: f64,
        south_lat_deg: f64,
        north_lat_deg: f64,
    ) -> Self {
        Self {
            west_lon_deg,
            east_lon_deg,
            south_lat_deg,
            north_lat_deg,
        }
    }

    pub fn contains(self, point: GeoPoint) -> bool {
        point.lon_deg >= self.west_lon_deg
            && point.lon_deg <= self.east_lon_deg
            && point.lat_deg >= self.south_lat_deg
            && point.lat_deg <= self.north_lat_deg
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GeoPolygon {
    pub exterior: Vec<GeoPoint>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub holes: Vec<Vec<GeoPoint>>,
}

impl GeoPolygon {
    pub fn new(exterior: Vec<GeoPoint>, holes: Vec<Vec<GeoPoint>>) -> Self {
        Self { exterior, holes }
    }

    pub fn bounds(&self) -> Option<GeoBounds> {
        let mut iter = self.exterior.iter().copied();
        let first = iter.next()?;
        let mut west = first.lon_deg;
        let mut east = first.lon_deg;
        let mut south = first.lat_deg;
        let mut north = first.lat_deg;
        for point in iter {
            west = west.min(point.lon_deg);
            east = east.max(point.lon_deg);
            south = south.min(point.lat_deg);
            north = north.max(point.lat_deg);
        }
        Some(GeoBounds::new(west, east, south, north))
    }

    pub fn contains(&self, point: GeoPoint) -> bool {
        if self.exterior.len() < 3 || !point_in_ring(point, &self.exterior) {
            return false;
        }
        !self.holes.iter().any(|ring| point_in_ring(point, ring))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FieldPointSampleMethod {
    Nearest,
    InverseDistance4,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FieldPointSampleContribution {
    pub grid_index: usize,
    pub location: GeoPoint,
    pub weight: f64,
    pub value: Option<f32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FieldPointSample {
    pub point: GeoPoint,
    pub method: FieldPointSampleMethod,
    pub value: Option<f32>,
    pub contributors: Vec<FieldPointSampleContribution>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FieldAreaSummaryMethod {
    CellCentersWithinPolygon,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FieldAreaSummary {
    pub method: FieldAreaSummaryMethod,
    pub included_cell_count: usize,
    pub valid_cell_count: usize,
    pub missing_cell_count: usize,
    pub min: Option<f32>,
    pub max: Option<f32>,
    pub mean: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TimeStamp {
    pub iso8601_utc: String,
}

impl TimeStamp {
    pub fn new<S: Into<String>>(iso8601_utc: S) -> Result<Self, RustwxError> {
        let iso8601_utc = iso8601_utc.into();
        validate_utc_timestamp(&iso8601_utc)?;
        Ok(Self { iso8601_utc })
    }

    pub fn as_str(&self) -> &str {
        self.iso8601_utc.as_str()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProductKey {
    Named(String),
}

impl ProductKey {
    pub fn named<S: Into<String>>(name: S) -> Self {
        Self::Named(name.into())
    }

    pub fn as_named(&self) -> Option<&str> {
        match self {
            Self::Named(name) => Some(name.as_str()),
        }
    }
}

impl std::fmt::Display for ProductKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Named(name) => f.write_str(name),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CanonicalField {
    Pressure,
    GeopotentialHeight,
    Temperature,
    RelativeHumidity,
    SpecificHumidity,
    Omega,
    GeometricVerticalVelocity,
    Dewpoint,
    PressureReducedToMeanSeaLevel,
    AbsoluteVorticity,
    RelativeVorticity,
    UWind,
    VWind,
    WindSpeed,
    WindGust,
    TotalCloudCover,
    LowCloudCover,
    MiddleCloudCover,
    HighCloudCover,
    PrecipitableWater,
    IntegratedVaporTransport,
    IntegratedVaporTransportU,
    IntegratedVaporTransportV,
    TotalPrecipitation,
    ModelSnowfall,
    Snowfall10to1,
    SnowfallKuchera,
    SnowfallLiquidEquivalent,
    SnowDepth,
    SnowDepthChange,
    ProbabilityOfPrecipitation,
    Visibility,
    FogFlag,
    SimulatedInfraredBrightnessTemperature,
    RadarReflectivity,
    LightningFlashDensity,
    CategoricalRain,
    CategoricalFreezingRain,
    CategoricalIcePellets,
    CategoricalSnow,
    LandSeaMask,
    CompositeReflectivity,
    UpdraftHelicity,
    SmokeMassDensity,
    ColumnIntegratedSmoke,
    /// Incoming solar flux on a horizontal surface at the valid time.
    DownwardShortwaveRadiationFlux,
    /// The five hydrometeor mixing ratios a bulk microphysics scheme
    /// carries, kg per kg of dry air.
    CloudWaterMixingRatio,
    RainWaterMixingRatio,
    CloudIceMixingRatio,
    SnowMixingRatio,
    GraupelMixingRatio,
    /// Cloud liquid water in a column, or a layer of it, where the air
    /// is colder than 0 C, g m-2.
    SupercooledLiquidWaterPath,
    /// Total column frozen-water mass, including adaptive-habit categories, g m-2.
    IceWaterPath,
    Pm25Dry,
    DustMassConcentration,
    OzoneConcentration,
    AerosolOpticalDepth550,
    IvtMagnitude,
    IvtEastward,
    IvtNorthward,
    IntegratedWaterVapour,
    FullColumnIvtEastward,
    FullColumnIvtNorthward,
    FullColumnIvtMagnitude,

    HailMaxSurface,
    HailMaxColumn,
    HailMaxSurfaceRun,
    HailMaxColumnRun,
    HailcastMax,
    HailcastMean,
    HailcastStd,
    HailcastRun,
    Wbgt,
    HotDryWindy,
    HainesLow,
    HainesMid,
    HainesHigh,
    LightningThreat1,
    LightningThreat2,
    LightningThreat3,
    Pr92FlashRate,
}

impl CanonicalField {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Pressure => "pressure",
            Self::GeopotentialHeight => "geopotential_height",
            Self::Temperature => "temperature",
            Self::RelativeHumidity => "relative_humidity",
            Self::SpecificHumidity => "specific_humidity",
            Self::GeometricVerticalVelocity => "geometric_vertical_velocity",
            Self::Omega => "omega",
            Self::Dewpoint => "dewpoint",
            Self::PressureReducedToMeanSeaLevel => "pressure_reduced_to_mean_sea_level",
            Self::AbsoluteVorticity => "absolute_vorticity",
            Self::RelativeVorticity => "relative_vorticity",
            Self::UWind => "u_wind",
            Self::VWind => "v_wind",
            Self::WindSpeed => "wind_speed",
            Self::WindGust => "wind_gust",
            Self::TotalCloudCover => "total_cloud_cover",
            Self::LowCloudCover => "low_cloud_cover",
            Self::MiddleCloudCover => "middle_cloud_cover",
            Self::HighCloudCover => "high_cloud_cover",
            Self::PrecipitableWater => "precipitable_water",
            Self::IntegratedVaporTransport => "integrated_vapor_transport",
            Self::IntegratedVaporTransportU => "integrated_vapor_transport_u",
            Self::IntegratedVaporTransportV => "integrated_vapor_transport_v",

            Self::TotalPrecipitation => "total_precipitation",
            Self::ModelSnowfall => "model_snowfall",
            Self::Snowfall10to1 => "snowfall_10to1",
            Self::SnowfallKuchera => "snowfall_kuchera",
            Self::SnowfallLiquidEquivalent => "snowfall_liquid_equivalent",
            Self::SnowDepth => "snow_depth_total",
            Self::SnowDepthChange => "snow_depth_change",
            Self::ProbabilityOfPrecipitation => "probability_of_precipitation",
            Self::Visibility => "visibility",
            Self::FogFlag => "fog_flag",
            Self::SimulatedInfraredBrightnessTemperature => {
                "simulated_infrared_brightness_temperature"
            }
            Self::RadarReflectivity => "radar_reflectivity",
            Self::LightningFlashDensity => "lightning_flash_density",
            Self::CategoricalRain => "categorical_rain",
            Self::CategoricalFreezingRain => "categorical_freezing_rain",
            Self::CategoricalIcePellets => "categorical_ice_pellets",
            Self::CategoricalSnow => "categorical_snow",
            Self::LandSeaMask => "land_sea_mask",
            Self::CompositeReflectivity => "composite_reflectivity",
            Self::UpdraftHelicity => "updraft_helicity",
            Self::SmokeMassDensity => "smoke_mass_density",
            Self::ColumnIntegratedSmoke => "column_integrated_smoke",
            Self::DownwardShortwaveRadiationFlux => "downward_shortwave_radiation_flux",
            Self::Pm25Dry => "pm25_dry",
            Self::DustMassConcentration => "dust_mass_concentration",
            Self::OzoneConcentration => "ozone_concentration",
            Self::AerosolOpticalDepth550 => "aerosol_optical_depth_550",
            Self::IvtMagnitude => "ivt_1000_300",
            Self::IvtEastward => "ivt_eastward_1000_300",
            Self::IvtNorthward => "ivt_northward_1000_300",
            Self::IntegratedWaterVapour => "integrated_water_vapour",
            Self::FullColumnIvtEastward => "ivt_eastward_full_column",
            Self::FullColumnIvtNorthward => "ivt_northward_full_column",
            Self::FullColumnIvtMagnitude => "ivt_full_column",

            Self::HailMaxSurface => "hail_max_surface",
            Self::HailMaxColumn => "hail_max_column",
            Self::HailMaxSurfaceRun => "hail_max_surface_run",
            Self::HailMaxColumnRun => "hail_max_column_run",
            Self::HailcastMax => "hail_max_hailcast",
            Self::HailcastMean => "hail_max_hailcast_mean",
            Self::HailcastStd => "hail_max_hailcast_std",
            Self::HailcastRun => "hail_max_hailcast_run",
            Self::Wbgt => "wbgt_2m",
            Self::HotDryWindy => "hot_dry_windy",
            Self::HainesLow => "haines_low",
            Self::HainesMid => "haines_mid",
            Self::HainesHigh => "haines_high",
            Self::LightningThreat1 => "lightning_threat_1",
            Self::LightningThreat2 => "lightning_threat_2",
            Self::LightningThreat3 => "lightning_threat_3",
            Self::Pr92FlashRate => "pr92_flash_rate",

            Self::CloudWaterMixingRatio => "cloud_water_mixing_ratio",
            Self::RainWaterMixingRatio => "rain_water_mixing_ratio",
            Self::CloudIceMixingRatio => "cloud_ice_mixing_ratio",
            Self::SnowMixingRatio => "snow_mixing_ratio",
            Self::GraupelMixingRatio => "graupel_mixing_ratio",
            Self::SupercooledLiquidWaterPath => "supercooled_liquid_water_path",
            Self::IceWaterPath => "ice_water_path",
        }
    }

    pub fn display_name(self) -> &'static str {
        match self {
            Self::Pressure => "Pressure",
            Self::GeopotentialHeight => "Geopotential Height",
            Self::Temperature => "Temperature",
            Self::RelativeHumidity => "Relative Humidity",
            Self::SpecificHumidity => "Specific Humidity",
            Self::GeometricVerticalVelocity => "Geometric Vertical Velocity",
            Self::Omega => "Pressure Vertical Velocity",
            Self::Dewpoint => "Dewpoint",
            Self::PressureReducedToMeanSeaLevel => "Pressure Reduced to Mean Sea Level",
            Self::AbsoluteVorticity => "Absolute Vorticity",
            Self::RelativeVorticity => "Relative Vorticity",
            Self::UWind => "U Wind",
            Self::VWind => "V Wind",
            Self::WindSpeed => "Wind Speed",
            Self::WindGust => "Wind Gust",
            Self::TotalCloudCover => "Total Cloud Cover",
            Self::LowCloudCover => "Low Cloud Cover",
            Self::MiddleCloudCover => "Middle Cloud Cover",
            Self::HighCloudCover => "High Cloud Cover",
            Self::PrecipitableWater => "Precipitable Water",
            Self::IntegratedVaporTransport => "Integrated Vapor Transport",
            Self::IntegratedVaporTransportU => "Eastward Integrated Vapor Transport",
            Self::IntegratedVaporTransportV => "Northward Integrated Vapor Transport",

            Self::TotalPrecipitation => "Total Precipitation",
            Self::ModelSnowfall => "Model Snowfall",
            Self::Snowfall10to1 => "10:1 Snowfall",
            Self::SnowfallKuchera => "Kuchera Snowfall",
            Self::SnowfallLiquidEquivalent => "Snowfall Liquid Equivalent",
            Self::SnowDepth => "LSM Snow Depth",
            Self::SnowDepthChange => "LSM Snow Depth Change",
            Self::ProbabilityOfPrecipitation => "Probability of Precipitation",
            Self::Visibility => "Visibility",
            Self::FogFlag => "Low visibility (< 1000 m)",
            Self::SimulatedInfraredBrightnessTemperature => {
                "Simulated Infrared Brightness Temperature"
            }
            Self::RadarReflectivity => "Radar Reflectivity",
            Self::LightningFlashDensity => "Lightning Flash Density",
            Self::CategoricalRain => "Categorical Rain",
            Self::CategoricalFreezingRain => "Categorical Freezing Rain",
            Self::CategoricalIcePellets => "Categorical Ice Pellets",
            Self::CategoricalSnow => "Categorical Snow",
            Self::LandSeaMask => "Land-Sea Mask",
            Self::CompositeReflectivity => "Composite Reflectivity",
            Self::UpdraftHelicity => "Updraft Helicity",
            Self::SmokeMassDensity => "Smoke Mass Density",
            Self::ColumnIntegratedSmoke => "Column-Integrated Smoke",
            Self::DownwardShortwaveRadiationFlux => "Downward Shortwave Radiation Flux",
            Self::Pm25Dry => "Dry PM2.5",
            Self::DustMassConcentration => "Dust Mass Concentration",
            Self::OzoneConcentration => "Ozone Concentration",
            Self::AerosolOpticalDepth550 => "550 nm Aerosol Optical Depth",
            Self::IvtMagnitude => "IVT 1000-300 hPa",
            Self::IvtEastward => "Eastward IVT 1000-300 hPa",
            Self::IvtNorthward => "Northward IVT 1000-300 hPa",
            Self::IntegratedWaterVapour => "Integrated Water Vapour Surface to Model Top",
            Self::FullColumnIvtEastward => "Eastward Water Vapour Flux Surface to Model Top",
            Self::FullColumnIvtNorthward => "Northward Water Vapour Flux Surface to Model Top",
            Self::FullColumnIvtMagnitude => "Water Vapour Flux Surface to Model Top",

            Self::HailMaxSurface => "PSD maximum hail diameter at lowest model level",
            Self::HailMaxColumn => "PSD maximum hail diameter over column",
            Self::HailMaxSurfaceRun => "Run maximum PSD surface hail diameter",
            Self::HailMaxColumnRun => "Run maximum PSD column hail diameter",
            Self::HailcastMax => "HAILCAST maximum surviving ice diameter",
            Self::HailcastMean => "HAILCAST mean surviving ice diameter",
            Self::HailcastStd => "HAILCAST diameter standard deviation",
            Self::HailcastRun => "Run maximum HAILCAST diameter",
            Self::Wbgt => "Outdoor WBGT, radiation balance",
            Self::HotDryWindy => "Hot Dry Windy, instantaneous",
            Self::HainesLow => "Haines index, low elevation",
            Self::HainesMid => "Haines index, middle elevation",
            Self::HainesHigh => "Haines index, high elevation",
            Self::LightningThreat1 => "Lightning threat LTG1",
            Self::LightningThreat2 => "Lightning threat LTG2",
            Self::LightningThreat3 => "Lightning threat LTG3",
            Self::Pr92FlashRate => "PR92 maximum-updraft flash rate",

            Self::CloudWaterMixingRatio => "Cloud Water Mixing Ratio",
            Self::RainWaterMixingRatio => "Rain Water Mixing Ratio",
            Self::CloudIceMixingRatio => "Cloud Ice Mixing Ratio",
            Self::SnowMixingRatio => "Snow Mixing Ratio",
            Self::GraupelMixingRatio => "Graupel Mixing Ratio",
            Self::SupercooledLiquidWaterPath => "Supercooled Liquid Water Path",
            Self::IceWaterPath => "Ice Water Path",
        }
    }

    pub fn native_units(self) -> &'static str {
        match self {
            Self::Pressure => "Pa",
            Self::GeopotentialHeight => "gpm",
            Self::Temperature => "K",
            Self::RelativeHumidity => "%",
            Self::SpecificHumidity => "kg/kg",
            Self::GeometricVerticalVelocity => "m/s",
            Self::Omega => "Pa/s",
            Self::Dewpoint => "K",
            Self::PressureReducedToMeanSeaLevel => "Pa",
            Self::AbsoluteVorticity | Self::RelativeVorticity => "s^-1",
            Self::UWind | Self::VWind | Self::WindSpeed => "m/s",
            Self::WindGust => "m/s",
            Self::TotalCloudCover => "%",
            Self::LowCloudCover | Self::MiddleCloudCover | Self::HighCloudCover => "%",
            Self::PrecipitableWater => "kg/m^2",
            Self::IntegratedVaporTransport => "kg m-1 s-1",
            Self::IntegratedVaporTransportU => "kg m-1 s-1",
            Self::IntegratedVaporTransportV => "kg m-1 s-1",

            Self::TotalPrecipitation => "kg/m^2",
            Self::ModelSnowfall | Self::Snowfall10to1 | Self::SnowfallKuchera | Self::SnowfallLiquidEquivalent | Self::SnowDepth | Self::SnowDepthChange => "mm",
            Self::ProbabilityOfPrecipitation => "%",
            Self::Visibility => "m",
            Self::FogFlag => "1",
            Self::SimulatedInfraredBrightnessTemperature => "K",
            Self::RadarReflectivity => "dBZ",
            Self::LightningFlashDensity => "km^-2 day^-1",
            Self::CategoricalRain
            | Self::CategoricalFreezingRain
            | Self::CategoricalIcePellets
            | Self::CategoricalSnow => "0/1",
            Self::LandSeaMask => "fraction",
            Self::CompositeReflectivity => "dBZ",
            Self::UpdraftHelicity => "m^2/s^2",
            Self::SmokeMassDensity => "kg/m^3",
            Self::ColumnIntegratedSmoke => "kg/m^2",
            Self::DownwardShortwaveRadiationFlux => "W/m^2",
            Self::Pm25Dry => "ug/m^3",
            Self::DustMassConcentration => "ug/m^3",
            Self::OzoneConcentration => "ppb",
            Self::AerosolOpticalDepth550 => "1",
            Self::IvtMagnitude => "kg m-1 s-1",
            Self::IvtEastward => "kg m-1 s-1",
            Self::IvtNorthward => "kg m-1 s-1",
            Self::IntegratedWaterVapour => "kg m-2",
            Self::FullColumnIvtEastward => "kg m-1 s-1",
            Self::FullColumnIvtNorthward => "kg m-1 s-1",
            Self::FullColumnIvtMagnitude => "kg m-1 s-1",

            Self::HailMaxSurface => "mm",
            Self::HailMaxColumn => "mm",
            Self::HailMaxSurfaceRun => "mm",
            Self::HailMaxColumnRun => "mm",
            Self::HailcastMax => "mm",
            Self::HailcastMean => "mm",
            Self::HailcastStd => "mm",
            Self::HailcastRun => "mm",
            Self::Wbgt => "degC",
            Self::HotDryWindy => "hPa m s^-1",
            Self::HainesLow => "1",
            Self::HainesMid => "1",
            Self::HainesHigh => "1",
            Self::LightningThreat1 => "flashes km^-2 (5 min)^-1",
            Self::LightningThreat2 => "flashes km^-2 (5 min)^-1",
            Self::LightningThreat3 => "flashes km^-2 (5 min)^-1",
            Self::Pr92FlashRate => "flashes s^-1 cell^-1",

            Self::CloudWaterMixingRatio
            | Self::RainWaterMixingRatio
            | Self::CloudIceMixingRatio
            | Self::SnowMixingRatio
            | Self::GraupelMixingRatio => "kg/kg",
            Self::SupercooledLiquidWaterPath | Self::IceWaterPath => "g/m^2",
        }
    }
}

impl std::fmt::Display for CanonicalField {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VerticalSelector {
    Surface,
    MeanSeaLevel,
    AltitudeMeters(u16),
    HeightAboveGroundMeters(u16),
    HeightAboveGroundLayerMeters { bottom_m: u16, top_m: u16 },
    HybridLevel(u16),
    IsobaricHpa(u16),
    EntireAtmosphere,
    NominalTop,
    /// The surface where the air is at this temperature: the lowest
    /// crossing above the ground, for a field such as its height.
    IsothermCelsius(i16),
    /// The largest value in the column, for a field a model carries on
    /// its own levels and a map shows as one number per column.
    ColumnMaximum,
}

impl VerticalSelector {
    pub fn as_slug(self) -> String {
        match self {
            Self::Surface => "surface".to_string(),
            Self::MeanSeaLevel => "mean_sea_level".to_string(),
            Self::AltitudeMeters(height_m) => format!("{height_m}m_msl"),
            Self::HeightAboveGroundMeters(height_m) => format!("{height_m}m_agl"),
            Self::HeightAboveGroundLayerMeters { bottom_m, top_m } => {
                format!("{bottom_m}m_to_{top_m}m_agl")
            }
            Self::HybridLevel(level) => format!("hybrid_level_{level}"),
            Self::IsobaricHpa(level_hpa) => format!("{level_hpa}hpa"),
            Self::EntireAtmosphere => "entire_atmosphere".to_string(),
            Self::NominalTop => "nominal_top".to_string(),
            Self::IsothermCelsius(celsius) => format!("isotherm_{}c", signed_slug(celsius)),
            Self::ColumnMaximum => "column_maximum".to_string(),
        }
    }
}

/// `0`, `10`, `minus10`: a signed number a slug can carry.
fn signed_slug(value: i16) -> String {
    if value < 0 {
        format!("minus{}", value.unsigned_abs())
    } else {
        value.to_string()
    }
}

impl std::fmt::Display for VerticalSelector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Surface => f.write_str("surface"),
            Self::MeanSeaLevel => f.write_str("mean_sea_level"),
            Self::AltitudeMeters(height_m) => write!(f, "{height_m}m_msl"),
            Self::HeightAboveGroundMeters(height_m) => write!(f, "{height_m}m_agl"),
            Self::HeightAboveGroundLayerMeters { bottom_m, top_m } => {
                write!(f, "{bottom_m}-{top_m}m_agl")
            }
            Self::HybridLevel(level) => write!(f, "hybrid_level_{level}"),
            Self::IsobaricHpa(level_hpa) => write!(f, "{level_hpa}hpa"),
            Self::EntireAtmosphere => f.write_str("entire_atmosphere"),
            Self::NominalTop => f.write_str("nominal_top"),
            Self::IsothermCelsius(celsius) => write!(f, "{celsius}C_isotherm"),
            Self::ColumnMaximum => f.write_str("column_maximum"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FieldProduct {
    Default,
    EnsembleMean,
    EnsembleStandardDeviation,
    EnsembleSpread,
    EnsembleMinimum,
    EnsembleMaximum,
    Percentile(u8),
    Probability(ProbabilitySelection),
}

impl Default for FieldProduct {
    fn default() -> Self {
        Self::Default
    }
}

impl FieldProduct {
    pub const fn is_default(&self) -> bool {
        matches!(self, Self::Default)
    }

    pub fn as_slug(self) -> String {
        match self {
            Self::Default => "default".to_string(),
            Self::EnsembleMean => "ensemble_mean".to_string(),
            Self::EnsembleStandardDeviation => "ensemble_stddev".to_string(),
            Self::EnsembleSpread => "ensemble_spread".to_string(),
            Self::EnsembleMinimum => "ensemble_min".to_string(),
            Self::EnsembleMaximum => "ensemble_max".to_string(),
            Self::Percentile(value) => format!("p{value}"),
            Self::Probability(selection) => selection.as_slug(),
        }
    }

    pub fn display_prefix(self) -> Option<String> {
        match self {
            Self::Default => None,
            Self::EnsembleMean => Some("Ensemble Mean".to_string()),
            Self::EnsembleStandardDeviation => Some("Ensemble Std Dev".to_string()),
            Self::EnsembleSpread => Some("Ensemble Spread".to_string()),
            Self::EnsembleMinimum => Some("Ensemble Min".to_string()),
            Self::EnsembleMaximum => Some("Ensemble Max".to_string()),
            Self::Percentile(value) => Some(format!("P{value}")),
            Self::Probability(_) => Some("Probability".to_string()),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ProbabilitySelection {
    pub probability_type: Option<u8>,
    pub lower_limit_milli: Option<i64>,
    pub upper_limit_milli: Option<i64>,
}

impl ProbabilitySelection {
    pub const fn new(
        probability_type: Option<u8>,
        lower_limit_milli: Option<i64>,
        upper_limit_milli: Option<i64>,
    ) -> Self {
        Self {
            probability_type,
            lower_limit_milli,
            upper_limit_milli,
        }
    }

    pub const fn any() -> Self {
        Self::new(None, None, None)
    }

    pub const fn above_milli(lower_limit_milli: i64) -> Self {
        Self::new(None, Some(lower_limit_milli), None)
    }

    pub const fn below_milli(upper_limit_milli: i64) -> Self {
        Self::new(None, None, Some(upper_limit_milli))
    }

    fn as_slug(self) -> String {
        let type_slug = self
            .probability_type
            .map(|value| format!("type{value}_"))
            .unwrap_or_default();
        match (self.lower_limit_milli, self.upper_limit_milli) {
            (Some(lower), Some(upper)) => format!("prob_{type_slug}{lower}m_to_{upper}m"),
            (Some(lower), None) => format!("prob_{type_slug}gt_{lower}m"),
            (None, Some(upper)) => format!("prob_{type_slug}lt_{upper}m"),
            (None, None) => format!("prob_{type_slug}any"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FieldSelector {
    pub field: CanonicalField,
    pub vertical: VerticalSelector,
    #[serde(default, skip_serializing_if = "FieldProduct::is_default")]
    pub product: FieldProduct,
}

impl FieldSelector {
    pub const fn new(field: CanonicalField, vertical: VerticalSelector) -> Self {
        Self {
            field,
            vertical,
            product: FieldProduct::Default,
        }
    }

    pub const fn with_product(self, product: FieldProduct) -> Self {
        Self {
            field: self.field,
            vertical: self.vertical,
            product,
        }
    }

    pub const fn with_ensemble_mean(self) -> Self {
        self.with_product(FieldProduct::EnsembleMean)
    }

    pub const fn with_ensemble_standard_deviation(self) -> Self {
        self.with_product(FieldProduct::EnsembleStandardDeviation)
    }

    pub const fn with_percentile(self, percentile: u8) -> Self {
        self.with_product(FieldProduct::Percentile(percentile))
    }

    pub const fn with_probability(self, selection: ProbabilitySelection) -> Self {
        self.with_product(FieldProduct::Probability(selection))
    }

    pub const fn isobaric(field: CanonicalField, level_hpa: u16) -> Self {
        Self::new(field, VerticalSelector::IsobaricHpa(level_hpa))
    }

    pub const fn hybrid_level(field: CanonicalField, level: u16) -> Self {
        Self::new(field, VerticalSelector::HybridLevel(level))
    }

    pub const fn surface(field: CanonicalField) -> Self {
        Self::new(field, VerticalSelector::Surface)
    }

    pub const fn mean_sea_level(field: CanonicalField) -> Self {
        Self::new(field, VerticalSelector::MeanSeaLevel)
    }

    pub const fn height_agl(field: CanonicalField, height_m: u16) -> Self {
        Self::new(field, VerticalSelector::HeightAboveGroundMeters(height_m))
    }

    pub const fn altitude_msl(field: CanonicalField, height_m: u16) -> Self {
        Self::new(field, VerticalSelector::AltitudeMeters(height_m))
    }

    pub const fn entire_atmosphere(field: CanonicalField) -> Self {
        Self::new(field, VerticalSelector::EntireAtmosphere)
    }

    pub const fn nominal_top(field: CanonicalField) -> Self {
        Self::new(field, VerticalSelector::NominalTop)
    }

    pub const fn height_layer_agl(field: CanonicalField, bottom_m: u16, top_m: u16) -> Self {
        Self::new(
            field,
            VerticalSelector::HeightAboveGroundLayerMeters { bottom_m, top_m },
        )
    }

    pub const fn isotherm_celsius(field: CanonicalField, celsius: i16) -> Self {
        Self::new(field, VerticalSelector::IsothermCelsius(celsius))
    }

    pub const fn column_maximum(field: CanonicalField) -> Self {
        Self::new(field, VerticalSelector::ColumnMaximum)
    }

    pub fn key(self) -> String {
        let base = format!("{}_{}", self.field.as_str(), self.vertical.as_slug());
        if self.product.is_default() {
            base
        } else {
            format!("{}_{}", base, self.product.as_slug())
        }
    }

    pub fn product_key(self) -> ProductKey {
        ProductKey::named(self.key())
    }

    pub fn display_name(self) -> String {
        if let Some(prefix) = self.product.display_prefix() {
            format!("{prefix} {} ({})", self.field.display_name(), self.vertical)
        } else {
            format!("{} ({})", self.field.display_name(), self.vertical)
        }
    }

    pub fn native_units(self) -> &'static str {
        if matches!(self.product, FieldProduct::Probability(_)) {
            return "%";
        }
        if self.field == CanonicalField::TotalPrecipitation
            && self.vertical == VerticalSelector::AltitudeMeters(0)
        {
            return "mm";
        }
        self.field.native_units()
    }
}

impl std::fmt::Display for FieldSelector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.product.is_default() {
            write!(f, "{}@{}", self.field, self.vertical)
        } else {
            write!(
                f,
                "{}@{}:{}",
                self.field,
                self.vertical,
                self.product.as_slug()
            )
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProductLineage {
    Direct,
    Derived,
    Windowed,
    Bundled,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProductKind {
    Direct,
    Derived,
    Windowed,
    Bundled,
}

impl ProductKind {
    pub const fn lineage(self) -> ProductLineage {
        match self {
            Self::Direct => ProductLineage::Direct,
            Self::Derived => ProductLineage::Derived,
            Self::Windowed => ProductLineage::Windowed,
            Self::Bundled => ProductLineage::Bundled,
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Direct => "direct",
            Self::Derived => "derived",
            Self::Windowed => "windowed",
            Self::Bundled => "bundled",
        }
    }
}

impl std::fmt::Display for ProductKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProductMaturity {
    Operational,
    Experimental,
    Proof,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProductSemanticFlag {
    Proxy,
    Composite,
    Alias,
    ProofOriented,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StatisticalProcess {
    Instantaneous,
    Accumulation,
    Average,
    Maximum,
    Minimum,
    Range,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProductWindowSpec {
    pub process: StatisticalProcess,
    pub duration_hours: Option<u16>,
}

impl ProductWindowSpec {
    pub fn instantaneous() -> Self {
        Self {
            process: StatisticalProcess::Instantaneous,
            duration_hours: None,
        }
    }

    pub fn accumulation(duration_hours: Option<u16>) -> Self {
        Self {
            process: StatisticalProcess::Accumulation,
            duration_hours,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ProductId {
    pub kind: ProductKind,
    pub slug: String,
}

impl ProductId {
    pub fn new<S: Into<String>>(kind: ProductKind, slug: S) -> Self {
        Self {
            kind,
            slug: slug.into(),
        }
    }

    pub fn as_slug(&self) -> &str {
        self.slug.as_str()
    }

    pub fn product_key(&self) -> ProductKey {
        ProductKey::named(self.slug.clone())
    }
}

impl std::fmt::Display for ProductId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}", self.kind, self.slug)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CanonicalProductIdentity {
    pub canonical: ProductId,
    pub alias_slugs: Vec<String>,
}

impl CanonicalProductIdentity {
    pub fn new(canonical: ProductId) -> Self {
        Self {
            canonical,
            alias_slugs: Vec::new(),
        }
    }

    pub fn with_alias_slug<S: Into<String>>(mut self, alias_slug: S) -> Self {
        let alias_slug = alias_slug.into();
        if alias_slug != self.canonical.slug && !self.alias_slugs.contains(&alias_slug) {
            self.alias_slugs.push(alias_slug);
        }
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProductProvenance {
    pub lineage: ProductLineage,
    pub maturity: ProductMaturity,
    pub flags: Vec<ProductSemanticFlag>,
    pub selector: Option<FieldSelector>,
    pub window: Option<ProductWindowSpec>,
}

impl ProductProvenance {
    pub fn new(lineage: ProductLineage, maturity: ProductMaturity) -> Self {
        Self {
            lineage,
            maturity,
            flags: Vec::new(),
            selector: None,
            window: None,
        }
    }

    pub fn selector_backed(selector: FieldSelector) -> Self {
        Self::new(ProductLineage::Direct, ProductMaturity::Operational).with_selector(selector)
    }

    pub fn with_flag(mut self, flag: ProductSemanticFlag) -> Self {
        if !self.flags.contains(&flag) {
            self.flags.push(flag);
        }
        self
    }

    pub fn with_selector(mut self, selector: FieldSelector) -> Self {
        self.selector = Some(selector);
        self
    }

    pub fn with_window(mut self, window: ProductWindowSpec) -> Self {
        self.window = Some(window);
        self
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SelectedField2D {
    pub selector: FieldSelector,
    pub units: String,
    pub grid: LatLonGrid,
    pub values: Vec<f32>,
    pub projection: Option<GridProjection>,
}

impl SelectedField2D {
    pub fn new<S: Into<String>>(
        selector: FieldSelector,
        units: S,
        grid: LatLonGrid,
        values: Vec<f32>,
    ) -> Result<Self, RustwxError> {
        let expected = grid.shape.checked_len()?;
        if values.len() != expected {
            return Err(RustwxError::InvalidFieldDataLength {
                expected,
                actual: values.len(),
            });
        }
        Ok(Self {
            selector,
            units: units.into(),
            grid,
            values,
            projection: None,
        })
    }

    pub fn with_projection(mut self, projection: GridProjection) -> Self {
        self.projection = Some(projection);
        self
    }

    pub fn into_field2d(self) -> Field2D {
        Field2D {
            product: self.selector.product_key(),
            units: self.units,
            grid: self.grid,
            values: self.values,
        }
    }

    pub fn sample_point(
        &self,
        point: GeoPoint,
        method: FieldPointSampleMethod,
    ) -> FieldPointSample {
        sample_field_point(&self.grid, &self.values, point, method)
    }

    pub fn summarize_polygon(&self, polygon: &GeoPolygon) -> FieldAreaSummary {
        summarize_field_polygon(&self.grid, &self.values, polygon)
    }
}

impl From<SelectedField2D> for Field2D {
    fn from(value: SelectedField2D) -> Self {
        value.into_field2d()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SelectedHybridLevelVolume {
    pub field: CanonicalField,
    pub levels_hybrid: Vec<u16>,
    pub units: String,
    pub grid: LatLonGrid,
    pub values: Vec<f32>,
    pub projection: Option<GridProjection>,
}

impl SelectedHybridLevelVolume {
    pub fn new<S: Into<String>>(
        field: CanonicalField,
        levels_hybrid: Vec<u16>,
        units: S,
        grid: LatLonGrid,
        values: Vec<f32>,
    ) -> Result<Self, RustwxError> {
        validate_hybrid_levels(&levels_hybrid)?;
        let cells = grid.shape.checked_len()?;
        let expected = checked_volume_elements(levels_hybrid.len(), cells)?;
        if values.len() != expected {
            return Err(RustwxError::InvalidFieldDataLength {
                expected,
                actual: values.len(),
            });
        }
        Ok(Self {
            field,
            levels_hybrid,
            units: units.into(),
            grid,
            values,
            projection: None,
        })
    }

    pub fn level_count(&self) -> usize {
        self.levels_hybrid.len()
    }

    pub fn level_slice(&self, level_index: usize) -> Option<&[f32]> {
        let layer_len = self.grid.shape.len();
        let start = level_index.checked_mul(layer_len)?;
        let end = start.checked_add(layer_len)?;
        self.values.get(start..end)
    }

    pub fn selector_at(&self, level_index: usize) -> Option<FieldSelector> {
        self.levels_hybrid
            .get(level_index)
            .copied()
            .map(|level| FieldSelector::hybrid_level(self.field, level))
    }

    pub fn with_projection(mut self, projection: GridProjection) -> Self {
        self.projection = Some(projection);
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProductKeyMetadata {
    pub display_name: String,
    pub description: Option<String>,
    pub native_units: Option<String>,
    pub category: Option<String>,
    pub identity: Option<CanonicalProductIdentity>,
    pub provenance: Option<ProductProvenance>,
}

impl ProductKeyMetadata {
    pub fn new<S: Into<String>>(display_name: S) -> Self {
        Self {
            display_name: display_name.into(),
            description: None,
            native_units: None,
            category: None,
            identity: None,
            provenance: None,
        }
    }

    pub fn with_description<S: Into<String>>(mut self, description: S) -> Self {
        self.description = Some(description.into());
        self
    }

    pub fn with_native_units<S: Into<String>>(mut self, native_units: S) -> Self {
        self.native_units = Some(native_units.into());
        self
    }

    pub fn with_category<S: Into<String>>(mut self, category: S) -> Self {
        self.category = Some(category.into());
        self
    }

    pub fn with_identity(mut self, identity: CanonicalProductIdentity) -> Self {
        self.identity = Some(identity);
        self
    }

    pub fn with_provenance(mut self, provenance: ProductProvenance) -> Self {
        self.provenance = Some(provenance);
        self
    }
}

impl FieldSelector {
    pub fn product_id(self) -> ProductId {
        ProductId::new(ProductKind::Direct, self.key())
    }

    pub fn product_provenance(self) -> ProductProvenance {
        ProductProvenance::selector_backed(self)
    }

    pub fn product_metadata(self) -> ProductKeyMetadata {
        ProductKeyMetadata::new(self.display_name())
            .with_native_units(self.native_units())
            .with_identity(CanonicalProductIdentity::new(self.product_id()))
            .with_provenance(self.product_provenance())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelTimestep {
    pub model: ModelId,
    pub cycle: CycleSpec,
    pub forecast_hour: u16,
    pub valid_time: TimeStamp,
    pub source: Option<SourceId>,
}

impl ModelTimestep {
    pub fn new(
        model: ModelId,
        cycle: CycleSpec,
        forecast_hour: u16,
        valid_time: TimeStamp,
    ) -> Result<Self, RustwxError> {
        Self::with_source(model, cycle, forecast_hour, valid_time, None)
    }

    pub fn with_source(
        model: ModelId,
        cycle: CycleSpec,
        forecast_hour: u16,
        valid_time: TimeStamp,
        source: Option<SourceId>,
    ) -> Result<Self, RustwxError> {
        if !forecast_hour_allowed_for_model(model, forecast_hour) {
            return Err(RustwxError::InvalidForecastHour(forecast_hour));
        }
        Ok(Self {
            model,
            cycle,
            forecast_hour,
            valid_time,
            source,
        })
    }

    pub fn request<S: Into<String>>(&self, product: S) -> Result<ModelRunRequest, RustwxError> {
        ModelRunRequest::new(self.model, self.cycle.clone(), self.forecast_hour, product)
    }

    pub fn descriptor(&self) -> ForecastDescriptor {
        ForecastDescriptor::new(
            self.model.as_str(),
            self.cycle.clone(),
            self.valid_time.clone(),
            self.forecast_hour,
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelFieldMetadata {
    pub timestep: ModelTimestep,
    pub product: ProductKey,
    pub product_metadata: Option<ProductKeyMetadata>,
    pub units: String,
}

impl ModelFieldMetadata {
    pub fn new<S: Into<String>>(timestep: ModelTimestep, product: ProductKey, units: S) -> Self {
        Self {
            timestep,
            product,
            product_metadata: None,
            units: units.into(),
        }
    }

    pub fn with_product_metadata(mut self, product_metadata: ProductKeyMetadata) -> Self {
        self.product_metadata = Some(product_metadata);
        self
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Field2D {
    pub product: ProductKey,
    pub units: String,
    pub grid: LatLonGrid,
    pub values: Vec<f32>,
}

impl Field2D {
    pub fn new<S: Into<String>>(
        product: ProductKey,
        units: S,
        grid: LatLonGrid,
        values: Vec<f32>,
    ) -> Result<Self, RustwxError> {
        let expected = grid.shape.checked_len()?;
        if values.len() != expected {
            return Err(RustwxError::InvalidGridShape {
                nx: grid.shape.nx,
                ny: grid.shape.ny,
            });
        }
        Ok(Self {
            product,
            units: units.into(),
            grid,
            values,
        })
    }

    pub fn sample_point(
        &self,
        point: GeoPoint,
        method: FieldPointSampleMethod,
    ) -> FieldPointSample {
        sample_field_point(&self.grid, &self.values, point, method)
    }

    pub fn summarize_polygon(&self, polygon: &GeoPolygon) -> FieldAreaSummary {
        summarize_field_polygon(&self.grid, &self.values, polygon)
    }
}

fn sample_field_point(
    grid: &LatLonGrid,
    values: &[f32],
    point: GeoPoint,
    method: FieldPointSampleMethod,
) -> FieldPointSample {
    let keep = match method {
        FieldPointSampleMethod::Nearest => 1usize,
        FieldPointSampleMethod::InverseDistance4 => 4usize,
    };
    let mut nearest = Vec::<(usize, f64)>::new();
    for idx in 0..grid.shape.len() {
        let distance = geographic_distance_score(grid, idx, point);
        insert_best_sample_candidate(&mut nearest, keep, idx, distance);
    }
    if nearest.is_empty() {
        return FieldPointSample {
            point,
            method,
            value: None,
            contributors: Vec::new(),
        };
    }

    let exact_match = nearest[0].1 <= 1.0e-12;
    let mut contributions = if exact_match || matches!(method, FieldPointSampleMethod::Nearest) {
        vec![point_sample_contribution(grid, values, nearest[0].0, 1.0)]
    } else {
        let mut weights = nearest
            .iter()
            .map(|(_, distance)| 1.0 / distance.max(1.0e-12))
            .collect::<Vec<_>>();
        let weight_sum = weights.iter().sum::<f64>().max(1.0e-12);
        for weight in &mut weights {
            *weight /= weight_sum;
        }
        nearest
            .iter()
            .zip(weights.iter())
            .map(|((idx, _), weight)| point_sample_contribution(grid, values, *idx, *weight))
            .collect::<Vec<_>>()
    };

    let finite_weight_sum = contributions
        .iter()
        .filter(|entry| entry.value.map(|value| value.is_finite()).unwrap_or(false))
        .map(|entry| entry.weight)
        .sum::<f64>();
    let value = if finite_weight_sum <= 0.0 {
        None
    } else {
        for contribution in &mut contributions {
            if contribution
                .value
                .map(|sample| sample.is_finite())
                .unwrap_or(false)
            {
                contribution.weight /= finite_weight_sum;
            } else {
                contribution.weight = 0.0;
            }
        }
        Some(
            contributions
                .iter()
                .filter_map(|entry| entry.value.map(|sample| sample as f64 * entry.weight))
                .sum::<f64>() as f32,
        )
    };

    FieldPointSample {
        point,
        method,
        value,
        contributors: contributions,
    }
}

fn summarize_field_polygon(
    grid: &LatLonGrid,
    values: &[f32],
    polygon: &GeoPolygon,
) -> FieldAreaSummary {
    let Some(bounds) = polygon.bounds() else {
        return FieldAreaSummary {
            method: FieldAreaSummaryMethod::CellCentersWithinPolygon,
            included_cell_count: 0,
            valid_cell_count: 0,
            missing_cell_count: 0,
            min: None,
            max: None,
            mean: None,
        };
    };

    let mut included_cell_count = 0usize;
    let mut valid_cell_count = 0usize;
    let mut missing_cell_count = 0usize;
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut sum = 0.0f64;

    for idx in 0..grid.shape.len() {
        let point = GeoPoint::new(grid.lat_deg[idx] as f64, grid.lon_deg[idx] as f64);
        if !bounds.contains(point) || !polygon.contains(point) {
            continue;
        }
        included_cell_count += 1;
        let value = values[idx];
        if value.is_finite() {
            valid_cell_count += 1;
            min = min.min(value);
            max = max.max(value);
            sum += value as f64;
        } else {
            missing_cell_count += 1;
        }
    }

    FieldAreaSummary {
        method: FieldAreaSummaryMethod::CellCentersWithinPolygon,
        included_cell_count,
        valid_cell_count,
        missing_cell_count,
        min: (valid_cell_count > 0).then_some(min),
        max: (valid_cell_count > 0).then_some(max),
        mean: (valid_cell_count > 0).then_some(sum / valid_cell_count as f64),
    }
}

fn point_sample_contribution(
    grid: &LatLonGrid,
    values: &[f32],
    idx: usize,
    weight: f64,
) -> FieldPointSampleContribution {
    FieldPointSampleContribution {
        grid_index: idx,
        location: GeoPoint::new(grid.lat_deg[idx] as f64, grid.lon_deg[idx] as f64),
        weight,
        value: values.get(idx).copied(),
    }
}

fn insert_best_sample_candidate(
    nearest: &mut Vec<(usize, f64)>,
    keep: usize,
    idx: usize,
    distance: f64,
) {
    let keep = keep.max(1);
    let insert_at = nearest
        .iter()
        .position(|&(existing_idx, existing_distance)| {
            distance < existing_distance
                || ((distance - existing_distance).abs() <= 1.0e-12 && idx < existing_idx)
        })
        .unwrap_or(nearest.len());
    if insert_at >= keep {
        return;
    }
    nearest.insert(insert_at, (idx, distance));
    if nearest.len() > keep {
        nearest.truncate(keep);
    }
}

fn geographic_distance_score(grid: &LatLonGrid, idx: usize, point: GeoPoint) -> f64 {
    let cos_lat = point.lat_deg.to_radians().cos().abs().max(0.2);
    let dlat = grid.lat_deg[idx] as f64 - point.lat_deg;
    let dlon = normalized_longitude_delta(grid.lon_deg[idx] as f64 - point.lon_deg) * cos_lat;
    dlat * dlat + dlon * dlon
}

fn point_in_ring(point: GeoPoint, ring: &[GeoPoint]) -> bool {
    if ring.len() < 3 {
        return false;
    }
    let point_x = 0.0f64;
    let point_y = point.lat_deg;
    let mut inside = false;
    let mut previous = *ring.last().expect("ring length checked");

    for current in ring {
        let current_x = normalized_longitude_delta(current.lon_deg - point.lon_deg);
        let current_y = current.lat_deg;
        let previous_x = normalized_longitude_delta(previous.lon_deg - point.lon_deg);
        let previous_y = previous.lat_deg;

        if point_on_segment(
            point_x, point_y, current_x, current_y, previous_x, previous_y,
        ) {
            return true;
        }

        let intersects = ((current_y > point_y) != (previous_y > point_y))
            && (point_x
                < (previous_x - current_x) * (point_y - current_y) / (previous_y - current_y)
                    + current_x);
        if intersects {
            inside = !inside;
        }
        previous = *current;
    }
    inside
}

fn point_on_segment(
    point_x: f64,
    point_y: f64,
    start_x: f64,
    start_y: f64,
    end_x: f64,
    end_y: f64,
) -> bool {
    let cross = (point_y - start_y) * (end_x - start_x) - (point_x - start_x) * (end_y - start_y);
    if cross.abs() > 1.0e-9 {
        return false;
    }
    let min_x = start_x.min(end_x) - 1.0e-9;
    let max_x = start_x.max(end_x) + 1.0e-9;
    let min_y = start_y.min(end_y) - 1.0e-9;
    let max_y = start_y.max(end_y) + 1.0e-9;
    point_x >= min_x && point_x <= max_x && point_y >= min_y && point_y <= max_y
}

fn normalized_longitude_delta(delta_deg: f64) -> f64 {
    let mut delta = delta_deg;
    while delta <= -180.0 {
        delta += 360.0;
    }
    while delta > 180.0 {
        delta -= 360.0;
    }
    delta
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Field3D {
    pub product: ProductKey,
    pub units: String,
    pub levels: Vec<f32>,
    pub grid: LatLonGrid,
    pub values: Vec<f32>,
}

impl Field3D {
    pub fn new<S: Into<String>>(
        product: ProductKey,
        units: S,
        levels: Vec<f32>,
        grid: LatLonGrid,
        values: Vec<f32>,
    ) -> Result<Self, RustwxError> {
        let cells = grid.shape.checked_len()?;
        let expected = checked_volume_elements(levels.len(), cells)?;
        if values.len() != expected {
            return Err(RustwxError::InvalidGridShape {
                nx: grid.shape.nx,
                ny: grid.shape.ny,
            });
        }
        Ok(Self {
            product,
            units: units.into(),
            levels,
            grid,
            values,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelField2D {
    pub metadata: ModelFieldMetadata,
    pub grid: LatLonGrid,
    pub values: Vec<f32>,
}

impl ModelField2D {
    pub fn new(
        metadata: ModelFieldMetadata,
        grid: LatLonGrid,
        values: Vec<f32>,
    ) -> Result<Self, RustwxError> {
        let expected = grid.shape.checked_len()?;
        if values.len() != expected {
            return Err(RustwxError::InvalidFieldDataLength {
                expected,
                actual: values.len(),
            });
        }
        Ok(Self {
            metadata,
            grid,
            values,
        })
    }

    pub fn into_field2d(self) -> Field2D {
        Field2D {
            product: self.metadata.product,
            units: self.metadata.units,
            grid: self.grid,
            values: self.values,
        }
    }
}

impl From<ModelField2D> for Field2D {
    fn from(value: ModelField2D) -> Self {
        value.into_field2d()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PressureLevelVolume {
    pub metadata: ModelFieldMetadata,
    pub levels_hpa: Vec<f32>,
    pub grid: LatLonGrid,
    pub values: Vec<f32>,
}

impl PressureLevelVolume {
    pub fn new(
        metadata: ModelFieldMetadata,
        levels_hpa: Vec<f32>,
        grid: LatLonGrid,
        values: Vec<f32>,
    ) -> Result<Self, RustwxError> {
        validate_pressure_levels(&levels_hpa)?;
        let cells = grid.shape.checked_len()?;
        let expected = checked_volume_elements(levels_hpa.len(), cells)?;
        if values.len() != expected {
            return Err(RustwxError::InvalidFieldDataLength {
                expected,
                actual: values.len(),
            });
        }
        Ok(Self {
            metadata,
            levels_hpa,
            grid,
            values,
        })
    }

    pub fn level_count(&self) -> usize {
        self.levels_hpa.len()
    }

    pub fn level_slice(&self, level_index: usize) -> Option<&[f32]> {
        let layer_len = self.grid.shape.len();
        let start = level_index.checked_mul(layer_len)?;
        let end = start.checked_add(layer_len)?;
        self.values.get(start..end)
    }

    pub fn into_field3d(self) -> Field3D {
        Field3D {
            product: self.metadata.product,
            units: self.metadata.units,
            levels: self.levels_hpa,
            grid: self.grid,
            values: self.values,
        }
    }
}

impl From<PressureLevelVolume> for Field3D {
    fn from(value: PressureLevelVolume) -> Self {
        value.into_field3d()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd, Serialize, Deserialize)]
pub enum ModelId {
    Hrrr,
    HrrrAk,
    Gfs,
    Gdas,
    Gefs,
    Aigfs,
    Aigefs,
    Hgefs,
    EcmwfOpenData,
    Aifs,
    Rap,
    Nam,
    Hiresw,
    Href,
    Sref,
    Rtma,
    Urma,
    Nbm,
    Rrfs,
    RrfsA,
    RrfsPublic,
    Refs,
    RrfsFireWx,
    WrfGdex,
}

impl ModelId {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Hrrr => "hrrr",
            Self::HrrrAk => "hrrr-ak",
            Self::Gfs => "gfs",
            Self::Gdas => "gdas",
            Self::Gefs => "gefs",
            Self::Aigfs => "aigfs",
            Self::Aigefs => "aigefs",
            Self::Hgefs => "hgefs",
            Self::EcmwfOpenData => "ecmwf-open-data",
            Self::Aifs => "aifs",
            Self::Rap => "rap",
            Self::Nam => "nam",
            Self::Hiresw => "hiresw",
            Self::Href => "href",
            Self::Sref => "sref",
            Self::Rtma => "rtma",
            Self::Urma => "urma",
            Self::Nbm => "nbm",
            Self::Rrfs => "rrfs",
            Self::RrfsA => "rrfs-a",
            Self::RrfsPublic => "rrfs-public",
            Self::Refs => "refs",
            Self::RrfsFireWx => "rrfs-firewx",
            Self::WrfGdex => "wrf",
        }
    }
}

impl std::fmt::Display for ModelId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::str::FromStr for ModelId {
    type Err = RustwxError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "hrrr" | "hrrr-prs" | "hrrr_prs" => Ok(Self::Hrrr),
            "hrrr-ak" | "hrrrak" | "hrrr_ak" | "hrrr-alaska" | "hrrr_alaska" => Ok(Self::HrrrAk),
            "gfs" | "gfs-0p25" | "gfs_0p25" | "gfs-0.25" | "gfs_0.25" => Ok(Self::Gfs),
            "gdas" | "gdas-0p25" | "gdas_0p25" | "gdas-0.25" | "gdas_0.25" => Ok(Self::Gdas),
            "gefs" | "gefs-ens" | "gefs_ens" | "gefs-ensemble" => Ok(Self::Gefs),
            "aigfs" | "ai-gfs" | "ai_gfs" => Ok(Self::Aigfs),
            "aigefs" | "ai-gefs" | "ai_gefs" => Ok(Self::Aigefs),
            "hgefs" | "hybrid-gefs" | "hybrid_gefs" | "hybrid-ai-gefs" | "hybrid_ai_gefs" => {
                Ok(Self::Hgefs)
            }
            "ecmwf" | "ifs" | "euro" | "european" | "ecmwf-ifs" | "ecmwf_ifs"
            | "ecmwf-open-data" | "ecmwf_open_data" => Ok(Self::EcmwfOpenData),
            "aifs" | "aifs-v2" | "aifsv2" | "aifs_single_v2" | "aifs-single" | "aifs_single"
            | "aifs-single-1.1" => Ok(Self::Aifs),
            "rap" => Ok(Self::Rap),
            "nam" => Ok(Self::Nam),
            "hiresw" | "hires" | "hires-window" | "hires_window" => Ok(Self::Hiresw),
            "href" | "hrefconus" | "href-conus" | "href_conus" => Ok(Self::Href),
            "sref" => Ok(Self::Sref),
            "rtma" | "rtma2p5" | "rtma-2p5" | "rtma_2p5" => Ok(Self::Rtma),
            "urma" | "urma2p5" | "urma-2p5" | "urma_2p5" => Ok(Self::Urma),
            "nbm" | "blend" | "national-blend" | "national_blend" => Ok(Self::Nbm),
            "rrfs" | "rrfs-ops" | "rrfs_ops" => Ok(Self::Rrfs),
            "rrfs-a" | "rrfsa" | "rrfs_a" => Ok(Self::RrfsA),
            "rrfs-public" | "rrfspublic" | "rrfs_public" | "rrfs-prototype" | "rrfs_prototype" => {
                Ok(Self::RrfsPublic)
            }
            "refs" | "rrfs-ensemble" | "rrfs_ensemble" => Ok(Self::Refs),
            "rrfs-firewx" | "rrfs_firewx" | "rrfsfirewx" | "firewx" | "fire-weather" => {
                Ok(Self::RrfsFireWx)
            }
            "wrf" | "wrf-arw" | "wrf_arw" | "wrf-gdex" | "wrf_gdex" | "wrfgdex" => {
                Ok(Self::WrfGdex)
            }
            other => Err(RustwxError::UnknownModel(other.to_string())),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CanonicalDataFamily {
    Surface,
    Pressure,
    Native,
}

impl CanonicalDataFamily {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Surface => "surface",
            Self::Pressure => "pressure",
            Self::Native => "native",
        }
    }
}

impl std::fmt::Display for CanonicalDataFamily {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum CanonicalBundleDescriptor {
    SurfaceAnalysis,
    PressureAnalysis,
    NativeAnalysis,
}

impl CanonicalBundleDescriptor {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::SurfaceAnalysis => "surface_analysis",
            Self::PressureAnalysis => "pressure_analysis",
            Self::NativeAnalysis => "native_analysis",
        }
    }

    pub const fn family(self) -> CanonicalDataFamily {
        match self {
            Self::SurfaceAnalysis => CanonicalDataFamily::Surface,
            Self::PressureAnalysis => CanonicalDataFamily::Pressure,
            Self::NativeAnalysis => CanonicalDataFamily::Native,
        }
    }
}

impl std::fmt::Display for CanonicalBundleDescriptor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Typed identity for an executable input bundle: the unique key by which
/// the planner dedupes fetch+decode work across products. A
/// `CanonicalBundleId` resolves to exactly one fetched GRIB file (one
/// `(model, cycle, forecast_hour, source, native_product)` tuple); two
/// `BundleRequirement`s with the same id share the same load.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Ord, PartialOrd, Serialize, Deserialize)]
pub struct CanonicalBundleId {
    pub model: ModelId,
    pub cycle: CycleSpec,
    pub forecast_hour: u16,
    pub source: SourceId,
    pub bundle: CanonicalBundleDescriptor,
    pub native_product: String,
}

impl CanonicalBundleId {
    pub fn new<S: Into<String>>(
        model: ModelId,
        cycle: CycleSpec,
        forecast_hour: u16,
        source: SourceId,
        bundle: CanonicalBundleDescriptor,
        native_product: S,
    ) -> Self {
        Self {
            model,
            cycle,
            forecast_hour,
            source,
            bundle,
            native_product: native_product.into(),
        }
    }

    pub fn family(&self) -> CanonicalDataFamily {
        self.bundle.family()
    }
}

impl std::fmt::Display for CanonicalBundleId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}@{}:{}{:02}z+f{:03}:{}",
            self.bundle,
            self.model,
            self.cycle.date_yyyymmdd,
            self.cycle.hour_utc,
            self.forecast_hour,
            self.native_product
        )
    }
}

/// What a product needs from the fetch layer. The planner translates each
/// requirement into a `CanonicalBundleId` for dedupe.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Ord, PartialOrd, Serialize, Deserialize)]
pub struct BundleRequirement {
    pub bundle: CanonicalBundleDescriptor,
    pub forecast_hour: u16,
    pub native_override: Option<String>,
}

impl BundleRequirement {
    pub fn new(bundle: CanonicalBundleDescriptor, forecast_hour: u16) -> Self {
        Self {
            bundle,
            forecast_hour,
            native_override: None,
        }
    }

    pub fn with_native_override<S: Into<String>>(mut self, native_product: S) -> Self {
        self.native_override = Some(native_product.into());
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd, Serialize, Deserialize)]
pub enum SourceId {
    Aws,
    Nomads,
    Google,
    Azure,
    Ecmwf,
    Ncei,
    Gdex,
    /// Local NetCDF archive populated by an active AIFS-v2 inference/dissemination harness.
    /// Layout: `$RUSTWX_AIFS_INFERENCE_ARCHIVE/{model}/{YYYYMMDD}T{HH}Z/lead{HHH}.nc`.
    AifsInference,
    /// Local NetCDF archive populated by data-driven weather-model inference.
    /// Layout: `$RUSTWX_EARTH2_ARCHIVE/{model}/{YYYYMMDD}T{HH}Z/lead{HHH}.nc`.
    Earth2Archive,
}

impl SourceId {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Aws => "aws",
            Self::Nomads => "nomads",
            Self::Google => "google",
            Self::Azure => "azure",
            Self::Ecmwf => "ecmwf",
            Self::Ncei => "ncei",
            Self::Gdex => "gdex",
            Self::AifsInference => "aifs-inference",
            Self::Earth2Archive => "earth2-archive",
        }
    }
}

impl std::fmt::Display for SourceId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::str::FromStr for SourceId {
    type Err = RustwxError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "aws" => Ok(Self::Aws),
            "nomads" => Ok(Self::Nomads),
            "google" => Ok(Self::Google),
            "azure" => Ok(Self::Azure),
            "ecmwf" => Ok(Self::Ecmwf),
            "ncei" => Ok(Self::Ncei),
            "gdex" => Ok(Self::Gdex),
            "aifs-inference" | "aifs_inference" | "aifsinference" | "inferenced-aifs"
            | "inferenced_aifs" | "aifsv2-inference" | "aifsv2_inference" => {
                Ok(Self::AifsInference)
            }
            "earth2-archive" | "earth2_archive" | "earth2archive" | "earth2" => {
                Ok(Self::Earth2Archive)
            }
            other => Err(RustwxError::UnknownSource(other.to_string())),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Ord, PartialOrd, Serialize, Deserialize)]
pub struct CycleSpec {
    pub date_yyyymmdd: String,
    pub hour_utc: u8,
}

impl CycleSpec {
    pub fn new<S: Into<String>>(date_yyyymmdd: S, hour_utc: u8) -> Result<Self, RustwxError> {
        let date_yyyymmdd = date_yyyymmdd.into();
        if date_yyyymmdd.len() != 8 || !date_yyyymmdd.chars().all(|ch| ch.is_ascii_digit()) {
            return Err(RustwxError::InvalidCycleDate(date_yyyymmdd));
        }
        validate_cycle_date(&date_yyyymmdd)?;
        if hour_utc > 23 {
            return Err(RustwxError::InvalidCycleHour(hour_utc));
        }
        Ok(Self {
            date_yyyymmdd,
            hour_utc,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelRunRequest {
    pub model: ModelId,
    pub cycle: CycleSpec,
    pub forecast_hour: u16,
    pub product: String,
}

impl ModelRunRequest {
    pub fn new<S: Into<String>>(
        model: ModelId,
        cycle: CycleSpec,
        forecast_hour: u16,
        product: S,
    ) -> Result<Self, RustwxError> {
        if !forecast_hour_allowed_for_model(model, forecast_hour) {
            return Err(RustwxError::InvalidForecastHour(forecast_hour));
        }
        Ok(Self {
            model,
            cycle,
            forecast_hour,
            product: product.into(),
        })
    }
}

fn forecast_hour_allowed_for_model(model: ModelId, forecast_hour: u16) -> bool {
    match model {
        ModelId::Aifs => forecast_hour <= AIFS_MAX_FORECAST_HOUR,
        _ => forecast_hour <= 999,
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResolvedUrl {
    pub source: SourceId,
    pub grib_url: String,
    pub idx_url: Option<String>,
}

impl ResolvedUrl {
    pub fn availability_probe_url(&self) -> &str {
        self.idx_url.as_deref().unwrap_or(&self.grib_url)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ForecastDescriptor {
    pub model: String,
    pub cycle: CycleSpec,
    pub valid_time: TimeStamp,
    pub forecast_hour: u16,
}

impl ForecastDescriptor {
    pub fn new<S: Into<String>>(
        model: S,
        cycle: CycleSpec,
        valid_time: TimeStamp,
        forecast_hour: u16,
    ) -> Self {
        Self {
            model: model.into(),
            cycle,
            valid_time,
            forecast_hour,
        }
    }
}

fn validate_cycle_date(date_yyyymmdd: &str) -> Result<(), RustwxError> {
    let year = date_yyyymmdd[..4]
        .parse::<u32>()
        .map_err(|_| RustwxError::InvalidCycleDate(date_yyyymmdd.to_string()))?;
    let month = date_yyyymmdd[4..6]
        .parse::<u32>()
        .map_err(|_| RustwxError::InvalidCycleDate(date_yyyymmdd.to_string()))?;
    let day = date_yyyymmdd[6..8]
        .parse::<u32>()
        .map_err(|_| RustwxError::InvalidCycleDate(date_yyyymmdd.to_string()))?;

    let max_day = match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => {
            if is_leap_year(year) {
                29
            } else {
                28
            }
        }
        _ => return Err(RustwxError::InvalidCycleDate(date_yyyymmdd.to_string())),
    };

    if day == 0 || day > max_day {
        return Err(RustwxError::InvalidCycleDate(date_yyyymmdd.to_string()));
    }

    Ok(())
}

fn validate_utc_timestamp(iso8601_utc: &str) -> Result<(), RustwxError> {
    let bytes = iso8601_utc.as_bytes();
    if bytes.len() != 20
        || bytes[4] != b'-'
        || bytes[7] != b'-'
        || bytes[10] != b'T'
        || bytes[13] != b':'
        || bytes[16] != b':'
        || bytes[19] != b'Z'
    {
        return Err(RustwxError::InvalidTimeStamp(iso8601_utc.to_string()));
    }

    let year = iso8601_utc[..4]
        .parse::<u32>()
        .map_err(|_| RustwxError::InvalidTimeStamp(iso8601_utc.to_string()))?;
    let month = iso8601_utc[5..7]
        .parse::<u32>()
        .map_err(|_| RustwxError::InvalidTimeStamp(iso8601_utc.to_string()))?;
    let day = iso8601_utc[8..10]
        .parse::<u32>()
        .map_err(|_| RustwxError::InvalidTimeStamp(iso8601_utc.to_string()))?;
    let hour = iso8601_utc[11..13]
        .parse::<u32>()
        .map_err(|_| RustwxError::InvalidTimeStamp(iso8601_utc.to_string()))?;
    let minute = iso8601_utc[14..16]
        .parse::<u32>()
        .map_err(|_| RustwxError::InvalidTimeStamp(iso8601_utc.to_string()))?;
    let second = iso8601_utc[17..19]
        .parse::<u32>()
        .map_err(|_| RustwxError::InvalidTimeStamp(iso8601_utc.to_string()))?;

    validate_cycle_date(&format!("{year:04}{month:02}{day:02}"))
        .map_err(|_| RustwxError::InvalidTimeStamp(iso8601_utc.to_string()))?;
    if hour > 23 || minute > 59 || second > 59 {
        return Err(RustwxError::InvalidTimeStamp(iso8601_utc.to_string()));
    }

    Ok(())
}

fn is_leap_year(year: u32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0)
}

fn validate_pressure_levels(levels_hpa: &[f32]) -> Result<(), RustwxError> {
    if levels_hpa.is_empty() {
        return Err(RustwxError::EmptyPressureLevels);
    }

    for (index, value) in levels_hpa.iter().copied().enumerate() {
        if !value.is_finite() || value <= 0.0 {
            return Err(RustwxError::InvalidPressureLevel { index, value });
        }
    }

    Ok(())
}

fn validate_hybrid_levels(levels_hybrid: &[u16]) -> Result<(), RustwxError> {
    if levels_hybrid.is_empty() {
        return Err(RustwxError::EmptyHybridLevels);
    }

    for (index, value) in levels_hybrid.iter().copied().enumerate() {
        if value == 0 {
            return Err(RustwxError::InvalidHybridLevel { index, value });
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests;
