use crate::gridded::{
    FetchRuntimeInfo, GridCrop, crop_latlon_grid, crop_values_f32, decode_cache_path,
    decode_surface_grid, fetch_family_file_with_patterns, load_surface_geometry_from_latest,
    resolve_model_run,
};
use crate::hrrr::HrrrFetchRuntimeInfo;
use crate::places::PlaceLabelOverlay;
use crate::planner::ExecutionPlanBuilder;
use crate::publication::{PublishedFetchIdentity, fetch_identity_from_cached_result};
use crate::runtime::{
    BundleLoaderConfig, FetchedBundleBytes, LoadedBundleSet, load_execution_plan,
};
use crate::shared_context::{
    DomainSpec, ProjectedMap, TitleProvenance, model_time_subtitle_with_lead_label,
    source_subtitle, static_chrome_scale, static_supersample_factor, static_supersample_sharpen,
    static_title_with_suffix,
};
use crate::windowed_decoder::{
    HrrrApcpDecode, HrrrSurfaceSnapshotDecode, HrrrUhDecode, HrrrWind10mMaxDecode,
    compute_qpf_product, compute_surface_snapshot_product, compute_uh_product,
    compute_wind10m_product, load_or_decode_apcp, load_or_decode_surface_snapshot,
    load_or_decode_uh25, load_or_decode_wind10m_max, qpf_fallback_hours_if_direct_missing,
};
use rustwx_core::{
    BundleRequirement, CanonicalBundleDescriptor, CanonicalField, Field2D, FieldSelector, ModelId,
    SourceId,
};
use rustwx_models::{LatestRun, resolve_canonical_bundle_product};
use rustwx_render::map_frame_aspect_ratio;
use rustwx_render::{
    LegendMode, MapRenderRequest, PngCompressionMode, PngWriteOptions, ProductVisualMode,
    WeatherProduct, save_png_profile_with_options,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io;
use std::path::PathBuf;
use std::thread;
use std::time::Instant;

const OUTPUT_WIDTH: u32 = 1200;
const OUTPUT_HEIGHT: u32 = 900;

fn default_output_width() -> u32 {
    OUTPUT_WIDTH
}

fn default_output_height() -> u32 {
    OUTPUT_HEIGHT
}

fn default_png_compression() -> PngCompressionMode {
    PngCompressionMode::Default
}

fn default_windowed_model() -> ModelId {
    ModelId::Hrrr
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HrrrWindowedProduct {
    Qpf1h,
    Qpf6h,
    Qpf12h,
    Qpf24h,
    QpfTotal,
    Uh25km1h,
    Uh25km3h,
    Uh25kmRunMax,
    Wind10m1hMax,
    Wind10mRunMax,
    Wind10m0to24hMax,
    Wind10m24to48hMax,
    Wind10m0to48hMax,
    Temp2m0to24hMax,
    Temp2m24to48hMax,
    Temp2m0to48hMax,
    Temp2m0to24hMin,
    Temp2m24to48hMin,
    Temp2m0to48hMin,
    Temp2m0to24hRange,
    Temp2m24to48hRange,
    Temp2m0to48hRange,
    Rh2m0to24hMax,
    Rh2m24to48hMax,
    Rh2m0to48hMax,
    Rh2m0to24hMin,
    Rh2m24to48hMin,
    Rh2m0to48hMin,
    Rh2m0to24hRange,
    Rh2m24to48hRange,
    Rh2m0to48hRange,
    Dewpoint2m0to24hMax,
    Dewpoint2m24to48hMax,
    Dewpoint2m0to48hMax,
    Dewpoint2m0to24hMin,
    Dewpoint2m24to48hMin,
    Dewpoint2m0to48hMin,
    Dewpoint2m0to24hRange,
    Dewpoint2m24to48hRange,
    Dewpoint2m0to48hRange,
    Vpd2m0to24hMax,
    Vpd2m24to48hMax,
    Vpd2m0to48hMax,
    Vpd2m0to24hMin,
    Vpd2m24to48hMin,
    Vpd2m0to48hMin,
    Vpd2m0to24hRange,
    Vpd2m24to48hRange,
    Vpd2m0to48hRange,
}

impl HrrrWindowedProduct {
    pub fn supported_products() -> &'static [Self] {
        SUPPORTED_HRRR_WINDOWED_PRODUCTS
    }

    /// Parse a windowed product slug back to its enum value (the inverse
    /// of [`Self::slug`]). Used by store-side callers that carry products
    /// as slugs.
    pub fn from_slug(slug: &str) -> Option<Self> {
        Self::supported_products()
            .iter()
            .copied()
            .find(|product| product.slug() == slug)
    }

    pub fn slug(self) -> &'static str {
        match self {
            Self::Qpf1h => "qpf_1h",
            Self::Qpf6h => "qpf_6h",
            Self::Qpf12h => "qpf_12h",
            Self::Qpf24h => "qpf_24h",
            Self::QpfTotal => "qpf_total",
            Self::Uh25km1h => "uh_2to5km_1h_max",
            Self::Uh25km3h => "uh_2to5km_3h_max",
            Self::Uh25kmRunMax => "uh_2to5km_run_max",
            Self::Wind10m1hMax => "10m_wind_1h_max",
            Self::Wind10mRunMax => "10m_wind_run_max",
            Self::Wind10m0to24hMax => "10m_wind_0_24h_max",
            Self::Wind10m24to48hMax => "10m_wind_24_48h_max",
            Self::Wind10m0to48hMax => "10m_wind_0_48h_max",
            Self::Temp2m0to24hMax => "2m_temp_0_24h_max",
            Self::Temp2m24to48hMax => "2m_temp_24_48h_max",
            Self::Temp2m0to48hMax => "2m_temp_0_48h_max",
            Self::Temp2m0to24hMin => "2m_temp_0_24h_min",
            Self::Temp2m24to48hMin => "2m_temp_24_48h_min",
            Self::Temp2m0to48hMin => "2m_temp_0_48h_min",
            Self::Temp2m0to24hRange => "2m_temp_0_24h_range",
            Self::Temp2m24to48hRange => "2m_temp_24_48h_range",
            Self::Temp2m0to48hRange => "2m_temp_0_48h_range",
            Self::Rh2m0to24hMax => "2m_rh_0_24h_max",
            Self::Rh2m24to48hMax => "2m_rh_24_48h_max",
            Self::Rh2m0to48hMax => "2m_rh_0_48h_max",
            Self::Rh2m0to24hMin => "2m_rh_0_24h_min",
            Self::Rh2m24to48hMin => "2m_rh_24_48h_min",
            Self::Rh2m0to48hMin => "2m_rh_0_48h_min",
            Self::Rh2m0to24hRange => "2m_rh_0_24h_range",
            Self::Rh2m24to48hRange => "2m_rh_24_48h_range",
            Self::Rh2m0to48hRange => "2m_rh_0_48h_range",
            Self::Dewpoint2m0to24hMax => "2m_dewpoint_0_24h_max",
            Self::Dewpoint2m24to48hMax => "2m_dewpoint_24_48h_max",
            Self::Dewpoint2m0to48hMax => "2m_dewpoint_0_48h_max",
            Self::Dewpoint2m0to24hMin => "2m_dewpoint_0_24h_min",
            Self::Dewpoint2m24to48hMin => "2m_dewpoint_24_48h_min",
            Self::Dewpoint2m0to48hMin => "2m_dewpoint_0_48h_min",
            Self::Dewpoint2m0to24hRange => "2m_dewpoint_0_24h_range",
            Self::Dewpoint2m24to48hRange => "2m_dewpoint_24_48h_range",
            Self::Dewpoint2m0to48hRange => "2m_dewpoint_0_48h_range",
            Self::Vpd2m0to24hMax => "2m_vpd_0_24h_max",
            Self::Vpd2m24to48hMax => "2m_vpd_24_48h_max",
            Self::Vpd2m0to48hMax => "2m_vpd_0_48h_max",
            Self::Vpd2m0to24hMin => "2m_vpd_0_24h_min",
            Self::Vpd2m24to48hMin => "2m_vpd_24_48h_min",
            Self::Vpd2m0to48hMin => "2m_vpd_0_48h_min",
            Self::Vpd2m0to24hRange => "2m_vpd_0_24h_range",
            Self::Vpd2m24to48hRange => "2m_vpd_24_48h_range",
            Self::Vpd2m0to48hRange => "2m_vpd_0_48h_range",
        }
    }

    pub fn title(self) -> &'static str {
        match self {
            Self::Qpf1h => "1-h QPF",
            Self::Qpf6h => "6-h QPF",
            Self::Qpf12h => "12-h QPF",
            Self::Qpf24h => "24-h QPF",
            Self::QpfTotal => "Total QPF",
            Self::Uh25km1h => "Updraft Helicity: 2-5 km AGL (1 h max)",
            Self::Uh25km3h => "Updraft Helicity: 2-5 km AGL (3 h max)",
            Self::Uh25kmRunMax => "Updraft Helicity: 2-5 km AGL (run max)",
            Self::Wind10m1hMax => "10 m Wind Speed (1 h max)",
            Self::Wind10mRunMax => "10 m Wind Speed (run max)",
            Self::Wind10m0to24hMax => "10 m Wind Speed (0-24 h max)",
            Self::Wind10m24to48hMax => "10 m Wind Speed (24-48 h max)",
            Self::Wind10m0to48hMax => "10 m Wind Speed (0-48 h max)",
            Self::Temp2m0to24hMax => "2 m Temperature (0-24 h max)",
            Self::Temp2m24to48hMax => "2 m Temperature (24-48 h max)",
            Self::Temp2m0to48hMax => "2 m Temperature (0-48 h max)",
            Self::Temp2m0to24hMin => "2 m Temperature (0-24 h min)",
            Self::Temp2m24to48hMin => "2 m Temperature (24-48 h min)",
            Self::Temp2m0to48hMin => "2 m Temperature (0-48 h min)",
            Self::Temp2m0to24hRange => "2 m Temperature Range (0-24 h)",
            Self::Temp2m24to48hRange => "2 m Temperature Range (24-48 h)",
            Self::Temp2m0to48hRange => "2 m Temperature Range (0-48 h)",
            Self::Rh2m0to24hMax => "2 m Relative Humidity (0-24 h max)",
            Self::Rh2m24to48hMax => "2 m Relative Humidity (24-48 h max)",
            Self::Rh2m0to48hMax => "2 m Relative Humidity (0-48 h max)",
            Self::Rh2m0to24hMin => "2 m Relative Humidity (0-24 h min)",
            Self::Rh2m24to48hMin => "2 m Relative Humidity (24-48 h min)",
            Self::Rh2m0to48hMin => "2 m Relative Humidity (0-48 h min)",
            Self::Rh2m0to24hRange => "2 m Relative Humidity Range (0-24 h)",
            Self::Rh2m24to48hRange => "2 m Relative Humidity Range (24-48 h)",
            Self::Rh2m0to48hRange => "2 m Relative Humidity Range (0-48 h)",
            Self::Dewpoint2m0to24hMax => "2 m Dewpoint (0-24 h max)",
            Self::Dewpoint2m24to48hMax => "2 m Dewpoint (24-48 h max)",
            Self::Dewpoint2m0to48hMax => "2 m Dewpoint (0-48 h max)",
            Self::Dewpoint2m0to24hMin => "2 m Dewpoint (0-24 h min)",
            Self::Dewpoint2m24to48hMin => "2 m Dewpoint (24-48 h min)",
            Self::Dewpoint2m0to48hMin => "2 m Dewpoint (0-48 h min)",
            Self::Dewpoint2m0to24hRange => "2 m Dewpoint Range (0-24 h)",
            Self::Dewpoint2m24to48hRange => "2 m Dewpoint Range (24-48 h)",
            Self::Dewpoint2m0to48hRange => "2 m Dewpoint Range (0-48 h)",
            Self::Vpd2m0to24hMax => "2 m Vapor Pressure Deficit (0-24 h max)",
            Self::Vpd2m24to48hMax => "2 m Vapor Pressure Deficit (24-48 h max)",
            Self::Vpd2m0to48hMax => "2 m Vapor Pressure Deficit (0-48 h max)",
            Self::Vpd2m0to24hMin => "2 m Vapor Pressure Deficit (0-24 h min)",
            Self::Vpd2m24to48hMin => "2 m Vapor Pressure Deficit (24-48 h min)",
            Self::Vpd2m0to48hMin => "2 m Vapor Pressure Deficit (0-48 h min)",
            Self::Vpd2m0to24hRange => "2 m Vapor Pressure Deficit Range (0-24 h)",
            Self::Vpd2m24to48hRange => "2 m Vapor Pressure Deficit Range (24-48 h)",
            Self::Vpd2m0to48hRange => "2 m Vapor Pressure Deficit Range (0-48 h)",
        }
    }

    /// This product's row in [`SNAPSHOT_FOLD_ROWS`], or `None` when it
    /// has none.
    pub fn snapshot_fold(self) -> Option<&'static SnapshotFoldRow> {
        SNAPSHOT_FOLD_ROWS.iter().find(|row| row.product == self)
    }

    pub(crate) fn is_qpf(self) -> bool {
        matches!(
            self,
            Self::Qpf1h | Self::Qpf6h | Self::Qpf12h | Self::Qpf24h | Self::QpfTotal
        )
    }

    /// The canonical input-field selectors this product's window folds
    /// (gpuwm vendor divergence).  These are INPUT requirements: the
    /// catalog derives per-source support for the non-QPF windowed
    /// entries from whether a source's files serve these fields, and
    /// the store-backed windowed compute proves the same planes (or
    /// their documented exact-equivalent fallbacks) against the stored
    /// fields.  Neither decision consults model identity.
    pub fn input_selectors(self) -> Vec<FieldSelector> {
        if self.is_qpf() {
            return vec![FieldSelector::surface(CanonicalField::TotalPrecipitation)];
        }
        if self.is_uh() {
            return vec![FieldSelector::height_layer_agl(
                CanonicalField::UpdraftHelicity,
                2000,
                5000,
            )];
        }
        if self.is_wind10m() {
            return vec![FieldSelector::height_agl(CanonicalField::WindSpeed, 10)];
        }
        use HrrrWindowedProduct::*;
        match self {
            Temp2m0to24hMax | Temp2m24to48hMax | Temp2m0to48hMax | Temp2m0to24hMin
            | Temp2m24to48hMin | Temp2m0to48hMin | Temp2m0to24hRange | Temp2m24to48hRange
            | Temp2m0to48hRange => {
                vec![FieldSelector::height_agl(CanonicalField::Temperature, 2)]
            }
            Rh2m0to24hMax | Rh2m24to48hMax | Rh2m0to48hMax | Rh2m0to24hMin | Rh2m24to48hMin
            | Rh2m0to48hMin | Rh2m0to24hRange | Rh2m24to48hRange | Rh2m0to48hRange => {
                vec![FieldSelector::height_agl(CanonicalField::RelativeHumidity, 2)]
            }
            Dewpoint2m0to24hMax
            | Dewpoint2m24to48hMax
            | Dewpoint2m0to48hMax
            | Dewpoint2m0to24hMin
            | Dewpoint2m24to48hMin
            | Dewpoint2m0to48hMin
            | Dewpoint2m0to24hRange
            | Dewpoint2m24to48hRange
            | Dewpoint2m0to48hRange => {
                vec![FieldSelector::height_agl(CanonicalField::Dewpoint, 2)]
            }
            // VPD derives from the 2 m temperature/humidity pair.
            Vpd2m0to24hMax | Vpd2m24to48hMax | Vpd2m0to48hMax | Vpd2m0to24hMin
            | Vpd2m24to48hMin | Vpd2m0to48hMin | Vpd2m0to24hRange | Vpd2m24to48hRange
            | Vpd2m0to48hRange => vec![
                FieldSelector::height_agl(CanonicalField::Temperature, 2),
                FieldSelector::height_agl(CanonicalField::RelativeHumidity, 2),
            ],
            _ => unreachable!("QPF/UH/wind families are handled above"),
        }
    }

    fn is_uh(self) -> bool {
        matches!(self, Self::Uh25km1h | Self::Uh25km3h | Self::Uh25kmRunMax)
    }

    pub(crate) fn is_wind10m(self) -> bool {
        matches!(
            self,
            Self::Wind10m1hMax
                | Self::Wind10mRunMax
                | Self::Wind10m0to24hMax
                | Self::Wind10m24to48hMax
                | Self::Wind10m0to48hMax
        )
    }

    pub fn is_surface_snapshot(self) -> bool {
        matches!(
            self,
            Self::Temp2m0to24hMax
                | Self::Temp2m24to48hMax
                | Self::Temp2m0to48hMax
                | Self::Temp2m0to24hMin
                | Self::Temp2m24to48hMin
                | Self::Temp2m0to48hMin
                | Self::Temp2m0to24hRange
                | Self::Temp2m24to48hRange
                | Self::Temp2m0to48hRange
                | Self::Rh2m0to24hMax
                | Self::Rh2m24to48hMax
                | Self::Rh2m0to48hMax
                | Self::Rh2m0to24hMin
                | Self::Rh2m24to48hMin
                | Self::Rh2m0to48hMin
                | Self::Rh2m0to24hRange
                | Self::Rh2m24to48hRange
                | Self::Rh2m0to48hRange
                | Self::Dewpoint2m0to24hMax
                | Self::Dewpoint2m24to48hMax
                | Self::Dewpoint2m0to48hMax
                | Self::Dewpoint2m0to24hMin
                | Self::Dewpoint2m24to48hMin
                | Self::Dewpoint2m0to48hMin
                | Self::Dewpoint2m0to24hRange
                | Self::Dewpoint2m24to48hRange
                | Self::Dewpoint2m0to48hRange
                | Self::Vpd2m0to24hMax
                | Self::Vpd2m24to48hMax
                | Self::Vpd2m0to48hMax
                | Self::Vpd2m0to24hMin
                | Self::Vpd2m24to48hMin
                | Self::Vpd2m0to48hMin
                | Self::Vpd2m0to24hRange
                | Self::Vpd2m24to48hRange
                | Self::Vpd2m0to48hRange
        )
    }
}

pub fn minimum_forecast_hour_for_windowed_product(product: HrrrWindowedProduct) -> u16 {
    use HrrrWindowedProduct::*;
    match product {
        Qpf1h | QpfTotal | Uh25km1h | Uh25kmRunMax | Wind10m1hMax | Wind10mRunMax => 1,
        Uh25km3h => 3,
        Qpf6h => 6,
        Qpf12h => 12,
        Qpf24h
        | Wind10m0to24hMax
        | Temp2m0to24hMax
        | Temp2m0to24hMin
        | Temp2m0to24hRange
        | Rh2m0to24hMax
        | Rh2m0to24hMin
        | Rh2m0to24hRange
        | Dewpoint2m0to24hMax
        | Dewpoint2m0to24hMin
        | Dewpoint2m0to24hRange
        | Vpd2m0to24hMax
        | Vpd2m0to24hMin
        | Vpd2m0to24hRange => 24,
        Wind10m24to48hMax
        | Wind10m0to48hMax
        | Temp2m24to48hMax
        | Temp2m24to48hMin
        | Temp2m24to48hRange
        | Temp2m0to48hMax
        | Temp2m0to48hMin
        | Temp2m0to48hRange
        | Rh2m24to48hMax
        | Rh2m24to48hMin
        | Rh2m24to48hRange
        | Rh2m0to48hMax
        | Rh2m0to48hMin
        | Rh2m0to48hRange
        | Dewpoint2m24to48hMax
        | Dewpoint2m24to48hMin
        | Dewpoint2m24to48hRange
        | Dewpoint2m0to48hMax
        | Dewpoint2m0to48hMin
        | Dewpoint2m0to48hRange
        | Vpd2m24to48hMax
        | Vpd2m24to48hMin
        | Vpd2m24to48hRange
        | Vpd2m0to48hMax
        | Vpd2m0to48hMin
        | Vpd2m0to48hRange => 48,
    }
}

pub fn windowed_product_available_at_forecast_hour(
    product: HrrrWindowedProduct,
    forecast_hour: u16,
) -> bool {
    forecast_hour >= minimum_forecast_hour_for_windowed_product(product)
}

/// How a maximum window is drawn when the history it is folded from stores
/// no sub-hourly maximum and holds one frame per hour: the fold is the
/// largest of the top-of-hour snapshots, which is not the window's maximum.
///
/// WHAT BREAKAGE THIS PREVENTS (gate law): an ArWen hourly history writes no
/// WSPD10MAX, so `10m_wind_1h_max`, `10m_wind_run_max` and the 24 and 48 h
/// maxima folded top-of-hour U10/V10 speeds and were drawn under "(1 h
/// max)" titles; only the catalog detail said "lower bound", and a
/// meteorologist reading the picture read a maximum.  A product with a row
/// here is drawn under the row's title, and its catalog detail is the row's
/// fold and reason.  A window read from a stored maximum field (the GRIB
/// lane's `wind_speed_10m_max_1h`, or WRF WSPD10MAX, which the wrfout import
/// stores as `wrf_wspd10max`), or from the frames between the hours of a
/// sub-hourly history, keeps its product title and its bytes.  A window
/// whose hours read a stored maximum at some hours and a snapshot at the
/// others is drawn under the row's `partial_title`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SnapshotFoldRow {
    pub product: HrrrWindowedProduct,
    /// The picture's title: what the fold is, and why it is not a maximum.
    pub title: &'static str,
    /// The title when some of the window's hours read a stored maximum and
    /// the others a snapshot.  `None` for a window that folds one hour,
    /// which cannot be partly either.
    pub partial_title: Option<&'static str>,
    /// What was folded, for the catalog detail; the frames follow it.
    pub fold: &'static str,
    /// Why no maximum could be drawn at those frames, for the catalog
    /// detail.
    pub why: &'static str,
}

const WIND10M_SNAPSHOT_WHY: &str = "the history stores no sub-hourly 10 m wind maximum there \
    (neither wind_speed_10m_max_1h nor WRF WSPD10MAX), so the wind between its hourly writes \
    is not seen: a lower bound on the window's maximum, not the maximum";

pub static SNAPSHOT_FOLD_ROWS: &[SnapshotFoldRow] = &[
    SnapshotFoldRow {
        product: HrrrWindowedProduct::Wind10m1hMax,
        title: "10 m Wind Speed (hourly snapshot, no stored 1 h max)",
        partial_title: None,
        fold: "the top-of-hour 10 m wind speed, hypot(u_10m, v_10m),",
        why: WIND10M_SNAPSHOT_WHY,
    },
    SnapshotFoldRow {
        product: HrrrWindowedProduct::Wind10mRunMax,
        title: "10 m Wind Speed (largest hourly snapshot, no stored max)",
        partial_title: Some("10 m Wind Speed (run max, partly hourly snapshots)"),
        fold: "the largest top-of-hour 10 m wind speed, hypot(u_10m, v_10m), over the run",
        why: WIND10M_SNAPSHOT_WHY,
    },
    SnapshotFoldRow {
        product: HrrrWindowedProduct::Wind10m0to24hMax,
        title: "10 m Wind Speed (largest hourly snapshot 0-24 h, no stored max)",
        partial_title: Some("10 m Wind Speed (0-24 h max, partly hourly snapshots)"),
        fold: "the largest top-of-hour 10 m wind speed, hypot(u_10m, v_10m), over 0-24 h",
        why: WIND10M_SNAPSHOT_WHY,
    },
    SnapshotFoldRow {
        product: HrrrWindowedProduct::Wind10m24to48hMax,
        title: "10 m Wind Speed (largest hourly snapshot 24-48 h, no stored max)",
        partial_title: Some("10 m Wind Speed (24-48 h max, partly hourly snapshots)"),
        fold: "the largest top-of-hour 10 m wind speed, hypot(u_10m, v_10m), over 24-48 h",
        why: WIND10M_SNAPSHOT_WHY,
    },
    SnapshotFoldRow {
        product: HrrrWindowedProduct::Wind10m0to48hMax,
        title: "10 m Wind Speed (largest hourly snapshot 0-48 h, no stored max)",
        partial_title: Some("10 m Wind Speed (0-48 h max, partly hourly snapshots)"),
        fold: "the largest top-of-hour 10 m wind speed, hypot(u_10m, v_10m), over 0-48 h",
        why: WIND10M_SNAPSHOT_WHY,
    },
];

pub static SUPPORTED_HRRR_WINDOWED_PRODUCTS: &[HrrrWindowedProduct] = &[
    HrrrWindowedProduct::Qpf1h,
    HrrrWindowedProduct::Qpf6h,
    HrrrWindowedProduct::Qpf12h,
    HrrrWindowedProduct::Qpf24h,
    HrrrWindowedProduct::QpfTotal,
    HrrrWindowedProduct::Uh25km1h,
    HrrrWindowedProduct::Uh25km3h,
    HrrrWindowedProduct::Uh25kmRunMax,
    HrrrWindowedProduct::Wind10m1hMax,
    HrrrWindowedProduct::Wind10mRunMax,
    HrrrWindowedProduct::Wind10m0to24hMax,
    HrrrWindowedProduct::Wind10m24to48hMax,
    HrrrWindowedProduct::Wind10m0to48hMax,
    HrrrWindowedProduct::Temp2m0to24hMax,
    HrrrWindowedProduct::Temp2m24to48hMax,
    HrrrWindowedProduct::Temp2m0to48hMax,
    HrrrWindowedProduct::Temp2m0to24hMin,
    HrrrWindowedProduct::Temp2m24to48hMin,
    HrrrWindowedProduct::Temp2m0to48hMin,
    HrrrWindowedProduct::Temp2m0to24hRange,
    HrrrWindowedProduct::Temp2m24to48hRange,
    HrrrWindowedProduct::Temp2m0to48hRange,
    HrrrWindowedProduct::Rh2m0to24hMax,
    HrrrWindowedProduct::Rh2m24to48hMax,
    HrrrWindowedProduct::Rh2m0to48hMax,
    HrrrWindowedProduct::Rh2m0to24hMin,
    HrrrWindowedProduct::Rh2m24to48hMin,
    HrrrWindowedProduct::Rh2m0to48hMin,
    HrrrWindowedProduct::Rh2m0to24hRange,
    HrrrWindowedProduct::Rh2m24to48hRange,
    HrrrWindowedProduct::Rh2m0to48hRange,
    HrrrWindowedProduct::Dewpoint2m0to24hMax,
    HrrrWindowedProduct::Dewpoint2m24to48hMax,
    HrrrWindowedProduct::Dewpoint2m0to48hMax,
    HrrrWindowedProduct::Dewpoint2m0to24hMin,
    HrrrWindowedProduct::Dewpoint2m24to48hMin,
    HrrrWindowedProduct::Dewpoint2m0to48hMin,
    HrrrWindowedProduct::Dewpoint2m0to24hRange,
    HrrrWindowedProduct::Dewpoint2m24to48hRange,
    HrrrWindowedProduct::Dewpoint2m0to48hRange,
    HrrrWindowedProduct::Vpd2m0to24hMax,
    HrrrWindowedProduct::Vpd2m24to48hMax,
    HrrrWindowedProduct::Vpd2m0to48hMax,
    HrrrWindowedProduct::Vpd2m0to24hMin,
    HrrrWindowedProduct::Vpd2m24to48hMin,
    HrrrWindowedProduct::Vpd2m0to48hMin,
    HrrrWindowedProduct::Vpd2m0to24hRange,
    HrrrWindowedProduct::Vpd2m24to48hRange,
    HrrrWindowedProduct::Vpd2m0to48hRange,
];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HrrrWindowedBatchRequest {
    #[serde(default = "default_windowed_model")]
    pub model: ModelId,
    pub date_yyyymmdd: String,
    pub cycle_override_utc: Option<u8>,
    pub forecast_hour: u16,
    pub source: SourceId,
    pub domain: DomainSpec,
    pub out_dir: PathBuf,
    pub cache_root: PathBuf,
    pub use_cache: bool,
    pub products: Vec<HrrrWindowedProduct>,
    #[serde(default = "default_output_width")]
    pub output_width: u32,
    #[serde(default = "default_output_height")]
    pub output_height: u32,
    #[serde(default = "default_png_compression")]
    pub png_compression: PngCompressionMode,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub place_label_overlay: Option<PlaceLabelOverlay>,
    /// Appended to the time subtitle (`| Δx 3 km`).  A suffix rather than
    /// an override because this lane's lead label names the accumulation
    /// window, which only the product itself can compose.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subtitle_left_suffix: Option<String>,
    /// Replaces the `source: <fetch source>` provenance line.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subtitle_right_override: Option<String>,
    /// Where these frames came from, for the headline parenthetical.
    #[serde(default)]
    pub title_provenance: TitleProvenance,
}

impl HrrrWindowedBatchRequest {
    pub fn png_write_options(&self) -> PngWriteOptions {
        PngWriteOptions {
            compression: self.png_compression,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HrrrWindowedHourFetchInfo {
    pub hour: u16,
    pub planned_product: String,
    pub fetched_product: String,
    pub requested_source: SourceId,
    pub resolved_source: SourceId,
    pub resolved_url: String,
    pub fetch_cache_hit: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_fetch: Option<PublishedFetchIdentity>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HrrrWindowedSharedTiming {
    pub fetch_geometry_ms: u128,
    pub decode_geometry_ms: u128,
    pub project_ms: u128,
    pub fetch_surface_ms: u128,
    pub decode_surface_ms: u128,
    pub fetch_nat_ms: u128,
    pub decode_nat_ms: u128,
    #[serde(default)]
    pub fetch_wind_ms: u128,
    #[serde(default)]
    pub decode_wind_ms: u128,
    #[serde(default)]
    pub fetch_temp_ms: u128,
    #[serde(default)]
    pub decode_temp_ms: u128,
    #[serde(default)]
    pub compute_products_ms: u128,
    pub geometry_fetch_cache_hit: bool,
    pub geometry_decode_cache_hit: bool,
    pub surface_hours_loaded: Vec<u16>,
    pub nat_hours_loaded: Vec<u16>,
    #[serde(default)]
    pub wind_hours_loaded: Vec<u16>,
    #[serde(default)]
    pub temp_hours_loaded: Vec<u16>,
    pub geometry_fetch: Option<HrrrFetchRuntimeInfo>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub geometry_input_fetch: Option<PublishedFetchIdentity>,
    pub surface_hour_fetches: Vec<HrrrWindowedHourFetchInfo>,
    pub uh_hour_fetches: Vec<HrrrWindowedHourFetchInfo>,
    #[serde(default)]
    pub wind_hour_fetches: Vec<HrrrWindowedHourFetchInfo>,
    #[serde(default)]
    pub temp_hour_fetches: Vec<HrrrWindowedHourFetchInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HrrrWindowedProductTiming {
    pub compute_ms: u128,
    pub render_ms: u128,
    pub total_ms: u128,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HrrrWindowedProductMetadata {
    pub strategy: String,
    pub contributing_forecast_hours: Vec<u16>,
    pub window_hours: Option<u16>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HrrrWindowedRenderedProduct {
    pub product: HrrrWindowedProduct,
    pub output_path: PathBuf,
    pub timing: HrrrWindowedProductTiming,
    pub metadata: HrrrWindowedProductMetadata,
    /// What the finished PNG maps to on the Earth, when the save path
    /// could publish it (gpuwm addition, VENDOR.md).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub georeference: Option<rustwx_render::PanelGeoReference>,
    /// Why `georeference` is `None`, from the save path itself.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub georeference_absent_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HrrrWindowedBlocker {
    pub product: HrrrWindowedProduct,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HrrrWindowedBatchReport {
    #[serde(default = "default_windowed_model")]
    pub model: ModelId,
    pub date_yyyymmdd: String,
    pub cycle_utc: u8,
    pub forecast_hour: u16,
    pub source: SourceId,
    pub domain: DomainSpec,
    pub shared_timing: HrrrWindowedSharedTiming,
    pub products: Vec<HrrrWindowedRenderedProduct>,
    pub blockers: Vec<HrrrWindowedBlocker>,
    pub total_ms: u128,
}

#[derive(Debug, Clone)]
pub(crate) struct WindowedSampledProductField {
    pub product: HrrrWindowedProduct,
    pub field: rustwx_core::Field2D,
    pub input_fetches: Vec<PublishedFetchIdentity>,
}

#[derive(Debug, Clone)]
pub(crate) struct WindowedSampledProductSet {
    pub fields: Vec<WindowedSampledProductField>,
    pub blockers: Vec<HrrrWindowedBlocker>,
}

#[derive(Debug, Clone)]
pub(crate) struct PreparedWindowedProduct {
    product: HrrrWindowedProduct,
    computed: crate::windowed_decoder::ComputedWindowedField,
}

#[derive(Debug, Clone)]
pub(crate) struct PreparedWindowedBatch {
    latest: LatestRun,
    shared_timing: HrrrWindowedSharedTiming,
    products: Vec<PreparedWindowedProduct>,
    blockers: Vec<HrrrWindowedBlocker>,
    grid: rustwx_core::LatLonGrid,
    projection: Option<rustwx_core::GridProjection>,
}

#[derive(Debug, Clone)]
struct PreparedWindowedGeometryContext {
    fetch_geometry_ms: u128,
    decode_geometry_ms: u128,
    geometry_fetch_cache_hit: bool,
    geometry_decode_cache_hit: bool,
    geometry_fetch: Option<HrrrFetchRuntimeInfo>,
    geometry_input_fetch: Option<PublishedFetchIdentity>,
    projected: ProjectedMap,
    grid: rustwx_core::LatLonGrid,
    projection: Option<rustwx_core::GridProjection>,
}

#[derive(Debug)]
enum WindowedProductOutcome {
    Rendered {
        index: usize,
        rendered: HrrrWindowedRenderedProduct,
    },
}

#[derive(Debug)]
enum WindowedComputeOutcome {
    Computed {
        index: usize,
        prepared: PreparedWindowedProduct,
    },
    Blocker {
        index: usize,
        blocker: HrrrWindowedBlocker,
    },
}

fn prepare_windowed_geometry_context(
    request: &HrrrWindowedBatchRequest,
    latest: &LatestRun,
) -> Result<PreparedWindowedGeometryContext, Box<dyn std::error::Error>> {
    match prepare_windowed_geometry_context_from_surface_bundle(request, latest) {
        Ok(context) => Ok(context),
        Err(surface_err) => {
            prepare_windowed_geometry_context_from_minimal_grib(request, latest).map_err(|grid_err| {
                format!(
                    "windowed geometry unavailable: surface-bundle decode failed ({surface_err}); minimal GRIB grid decode failed ({grid_err})"
                )
                .into()
            })
        }
    }
}

fn prepare_windowed_geometry_context_from_surface_bundle(
    request: &HrrrWindowedBatchRequest,
    latest: &LatestRun,
) -> Result<PreparedWindowedGeometryContext, Box<dyn std::error::Error>> {
    let geometry = load_surface_geometry_from_latest(
        latest.clone(),
        request.forecast_hour,
        None,
        &request.cache_root,
        request.use_cache,
    )?;
    let projected_maps = crate::gridded::build_projected_maps_for_sizes(
        &geometry.surface_decode.value,
        request.domain.bounds,
        &[(request.output_width, request.output_height)],
    )?;
    let projected = projected_maps
        .projected_map(request.output_width, request.output_height)
        .cloned()
        .ok_or("missing projected map for windowed batch")?;

    Ok(PreparedWindowedGeometryContext {
        fetch_geometry_ms: geometry.fetch_ms,
        decode_geometry_ms: geometry.decode_ms,
        geometry_fetch_cache_hit: geometry.surface_file.fetched.cache_hit,
        geometry_decode_cache_hit: geometry.surface_decode.cache_hit,
        geometry_fetch: Some(hrrr_fetch_runtime_info_from_bundle(
            &geometry.surface_file.runtime_info(&geometry.surface_bundle),
        )),
        geometry_input_fetch: Some(fetch_identity_from_cached_result(
            &geometry.surface_bundle.native_product,
            &geometry.surface_file.request,
            &geometry.surface_file.fetched,
        )),
        projected,
        grid: geometry.grid,
        projection: geometry.surface_decode.value.projection.clone(),
    })
}

fn prepare_windowed_geometry_context_from_minimal_grib(
    request: &HrrrWindowedBatchRequest,
    latest: &LatestRun,
) -> Result<PreparedWindowedGeometryContext, Box<dyn std::error::Error>> {
    let geometry_bundle = resolve_canonical_bundle_product(
        latest.model,
        CanonicalBundleDescriptor::NativeAnalysis,
        Some(&windowed_fetch_product(latest.model)),
    );
    let fetch_start = Instant::now();
    let geometry_file = fetch_family_file_with_patterns(
        latest.model,
        latest.cycle.clone(),
        request.forecast_hour,
        latest.source,
        &geometry_bundle,
        windowed_hour_patterns(false, false, false, true),
        &request.cache_root,
        request.use_cache,
    )?;
    let fetch_ms = fetch_start.elapsed().as_millis();

    let decode_start = Instant::now();
    let grid_layout = decode_surface_grid(&geometry_file.bytes)?;
    let decode_ms = decode_start.elapsed().as_millis();
    let grid = rustwx_core::LatLonGrid::new(
        rustwx_core::GridShape::new(grid_layout.nx, grid_layout.ny)?,
        grid_layout.lat.iter().copied().map(|v| v as f32).collect(),
        grid_layout.lon.iter().copied().map(|v| v as f32).collect(),
    )?;
    let projected = crate::direct::build_projected_map_with_projection(
        &grid.lat_deg,
        &grid.lon_deg,
        grid_layout.projection.as_ref(),
        request.domain.bounds,
        map_frame_aspect_ratio(request.output_width, request.output_height, true, true),
    )?;

    Ok(PreparedWindowedGeometryContext {
        fetch_geometry_ms: fetch_ms,
        decode_geometry_ms: decode_ms,
        geometry_fetch_cache_hit: geometry_file.fetched.cache_hit,
        geometry_decode_cache_hit: false,
        geometry_fetch: Some(hrrr_fetch_runtime_info_from_bundle(
            &geometry_file.runtime_info(&geometry_bundle),
        )),
        geometry_input_fetch: Some(fetch_identity_from_cached_result(
            &geometry_bundle.native_product,
            &geometry_file.request,
            &geometry_file.fetched,
        )),
        projected,
        grid,
        projection: grid_layout.projection,
    })
}

fn hrrr_fetch_runtime_info_from_bundle(fetch: &FetchRuntimeInfo) -> HrrrFetchRuntimeInfo {
    HrrrFetchRuntimeInfo {
        planned_product: fetch.planned_product.clone(),
        fetched_product: fetch.fetched_product.clone(),
        requested_source: fetch.requested_source,
        resolved_source: fetch.resolved_source,
        resolved_url: fetch.resolved_url.clone(),
    }
}

/// All `PublishedFetchIdentity` values that contributed to a windowed
/// batch, deduplicated by fetch key. Extracted so standalone runners
/// (`hrrr_windowed_batch`) and the unified runner (`hrrr_non_ecape_hour`)
/// publish the same input-fetch set.
pub fn collect_windowed_input_fetches(
    report: &HrrrWindowedBatchReport,
) -> Vec<PublishedFetchIdentity> {
    let mut by_key = std::collections::BTreeMap::<String, PublishedFetchIdentity>::new();
    if let Some(identity) = &report.shared_timing.geometry_input_fetch {
        by_key
            .entry(identity.fetch_key.clone())
            .or_insert_with(|| identity.clone());
    }
    for fetch in report
        .shared_timing
        .surface_hour_fetches
        .iter()
        .chain(report.shared_timing.uh_hour_fetches.iter())
        .chain(report.shared_timing.wind_hour_fetches.iter())
        .chain(report.shared_timing.temp_hour_fetches.iter())
    {
        if let Some(identity) = &fetch.input_fetch {
            by_key
                .entry(identity.fetch_key.clone())
                .or_insert_with(|| identity.clone());
        }
    }
    by_key.into_values().collect()
}

/// Fetch keys that cited this product as an input, in contributing-hour
/// order. Mirrors the runtime identity the rendered product actually
/// depended on (QPF products consume `sfc` hourly fetches; UH products
/// consume `nat` hourly fetches).
pub fn windowed_product_input_fetch_keys(
    product: &HrrrWindowedRenderedProduct,
    shared_timing: &HrrrWindowedSharedTiming,
) -> Vec<String> {
    let contributing_hours = &product.metadata.contributing_forecast_hours;
    let fetches = if product.product.is_qpf() {
        &shared_timing.surface_hour_fetches
    } else if product.product.is_wind10m() {
        &shared_timing.wind_hour_fetches
    } else if product.product.is_surface_snapshot() {
        &shared_timing.temp_hour_fetches
    } else {
        &shared_timing.uh_hour_fetches
    };
    let mut keys = Vec::new();
    for fetch in fetches
        .iter()
        .filter(|fetch| contributing_hours.contains(&fetch.hour))
    {
        if let Some(identity) = &fetch.input_fetch {
            if !keys.contains(&identity.fetch_key) {
                keys.push(identity.fetch_key.clone());
            }
        }
    }
    keys
}

pub(crate) fn required_windowed_fetch_products(products: &[HrrrWindowedProduct]) -> Vec<String> {
    (!products.is_empty())
        .then(|| vec!["sfc".to_string()])
        .unwrap_or_default()
}

fn windowed_fetch_product(model: ModelId) -> String {
    resolve_canonical_bundle_product(model, CanonicalBundleDescriptor::SurfaceAnalysis, None)
        .native_product
}

fn windowed_hour_patterns(
    needs_apcp: bool,
    needs_uh: bool,
    needs_wind_max: bool,
    needs_surface_snapshot: bool,
) -> Vec<String> {
    let mut patterns = Vec::<String>::new();
    if needs_apcp {
        push_unique(&mut patterns, "APCP:surface");
    }
    if needs_uh {
        push_unique(&mut patterns, "UPHL");
        push_unique(&mut patterns, "MXUPHL");
    }
    if needs_wind_max {
        push_unique(&mut patterns, "WIND:10 m above ground");
    }
    if needs_surface_snapshot {
        push_unique(&mut patterns, "TMP:2 m above ground");
        push_unique(&mut patterns, "RH:2 m above ground");
        push_unique(&mut patterns, "DPT:2 m above ground");
    }
    patterns
}

fn push_unique(patterns: &mut Vec<String>, value: &str) {
    let value = value.to_string();
    if !patterns.contains(&value) {
        patterns.push(value);
    }
}

fn qpf_hourly_fallback_hours_if_allowed(
    model: ModelId,
    product: HrrrWindowedProduct,
    forecast_hour: u16,
    apcp_by_hour: &BTreeMap<u16, Result<HrrrApcpDecode, String>>,
) -> Option<Vec<u16>> {
    if !qpf_hourly_fallback_supported(model, forecast_hour) {
        return None;
    }
    qpf_fallback_hours_if_direct_missing(product, forecast_hour, apcp_by_hour)
}

fn qpf_hourly_fallback_supported(model: ModelId, forecast_hour: u16) -> bool {
    match model {
        ModelId::Hrrr | ModelId::HrrrAk | ModelId::Rap => true,
        ModelId::Gfs | ModelId::Gdas | ModelId::Aigfs => forecast_hour <= 120,
        _ => false,
    }
}

fn load_apcp_fallback_hours(
    loaded: Option<&LoadedBundleSet>,
    request: &HrrrWindowedBatchRequest,
    latest: &LatestRun,
    hours: &BTreeSet<u16>,
) -> Result<
    (
        BTreeMap<u16, Result<HrrrApcpDecode, String>>,
        Vec<HrrrWindowedHourFetchInfo>,
        u128,
        u128,
    ),
    Box<dyn std::error::Error>,
> {
    let fetch_product = windowed_fetch_product(latest.model);
    let mut already_loaded_hours = BTreeSet::new();
    let mut missing_hours = BTreeSet::new();
    for &hour in hours {
        if loaded
            .and_then(|set| lookup_planner_bundle_for_hour(set, hour, fetch_product.as_str()))
            .is_some()
        {
            already_loaded_hours.insert(hour);
        } else {
            missing_hours.insert(hour);
        }
    }

    let mut out = BTreeMap::new();
    let mut fetches = Vec::new();
    let mut fetch_ms = 0u128;
    let mut decode_ms = 0u128;

    if !already_loaded_hours.is_empty() {
        let (mut loaded_apcp, mut loaded_fetches, loaded_fetch_ms, loaded_decode_ms) =
            load_apcp_hours_from_plan(loaded, request, &already_loaded_hours)?;
        out.append(&mut loaded_apcp);
        fetches.append(&mut loaded_fetches);
        fetch_ms += loaded_fetch_ms;
        decode_ms += loaded_decode_ms;
    }

    if !missing_hours.is_empty() {
        let empty_hours = BTreeSet::new();
        let fallback_loaded = load_windowed_plan_for_hours(
            request,
            latest,
            &missing_hours,
            &empty_hours,
            &empty_hours,
            &empty_hours,
        )?;
        let (mut fallback_apcp, mut fallback_fetches, fallback_fetch_ms, fallback_decode_ms) =
            load_apcp_hours_from_plan(fallback_loaded.as_ref(), request, &missing_hours)?;
        out.append(&mut fallback_apcp);
        fetches.append(&mut fallback_fetches);
        fetch_ms += fallback_fetch_ms;
        decode_ms += fallback_decode_ms;
    }

    Ok((out, fetches, fetch_ms, decode_ms))
}

fn load_windowed_plan_for_hours(
    request: &HrrrWindowedBatchRequest,
    latest: &LatestRun,
    surface_hours: &BTreeSet<u16>,
    nat_hours: &BTreeSet<u16>,
    wind_hours: &BTreeSet<u16>,
    temp_hours: &BTreeSet<u16>,
) -> Result<Option<LoadedBundleSet>, Box<dyn std::error::Error>> {
    let mut all_hours: BTreeSet<u16> = surface_hours.iter().copied().collect();
    all_hours.extend(nat_hours.iter().copied());
    all_hours.extend(wind_hours.iter().copied());
    all_hours.extend(temp_hours.iter().copied());

    let mut plan_builder = ExecutionPlanBuilder::new(latest, request.forecast_hour);
    let fetch_product = windowed_fetch_product(latest.model);
    for &hour in &all_hours {
        let requirement = BundleRequirement::new(CanonicalBundleDescriptor::NativeAnalysis, hour)
            .with_native_override(fetch_product.clone());
        let patterns = windowed_hour_patterns(
            surface_hours.contains(&hour),
            nat_hours.contains(&hour),
            wind_hours.contains(&hour),
            temp_hours.contains(&hour),
        );
        // Preserve the logical alias names manifests have always
        // surfaced for windowed: QPF hours show up as "sfc"; UH hours
        // show up as "nat" because the windowed lane historically
        // logged them as native-family fetches even though both decode
        // out of wrfsfc.
        if surface_hours.contains(&hour) || wind_hours.contains(&hour) || temp_hours.contains(&hour)
        {
            plan_builder.require_with_logical_family_and_patterns(
                &requirement,
                Some(fetch_product.as_str()),
                patterns.clone(),
            );
        }
        if nat_hours.contains(&hour) {
            plan_builder.require_with_logical_family_and_patterns(
                &requirement,
                Some(fetch_product.as_str()),
                patterns,
            );
        }
    }
    let plan = plan_builder.build();
    if plan.bundles.is_empty() {
        return Ok(None);
    }
    Ok(Some(load_execution_plan(
        plan,
        &BundleLoaderConfig::new(request.cache_root.clone(), request.use_cache),
    )?))
}

pub(crate) fn load_windowed_sampled_fields_from_latest(
    latest: &LatestRun,
    forecast_hour: u16,
    cache_root: &std::path::Path,
    use_cache: bool,
    products: &[HrrrWindowedProduct],
) -> Result<WindowedSampledProductSet, Box<dyn std::error::Error>> {
    let (planned_products, mut blockers, planned_surface_hours, nat_hours, wind_hours, temp_hours) =
        plan_windowed_products(products, forecast_hour, Some(latest.cycle.hour_utc));
    if planned_products.is_empty() {
        return Ok(WindowedSampledProductSet {
            fields: Vec::new(),
            blockers,
        });
    }

    let fetch_product = windowed_fetch_product(latest.model);
    let request = sampling_windowed_request(
        latest.model,
        forecast_hour,
        latest.source,
        cache_root,
        use_cache,
    );
    let mut surface_hours = if planned_products.iter().any(|product| product.is_qpf()) {
        let mut hours = BTreeSet::new();
        hours.insert(forecast_hour);
        hours
    } else {
        planned_surface_hours
    };
    let loaded = load_windowed_plan_for_hours(
        &request,
        latest,
        &surface_hours,
        &nat_hours,
        &wind_hours,
        &temp_hours,
    )?
    .ok_or("windowed sampling produced no input bundle plan")?;
    let geometry = lookup_planner_bundle_for_hour(&loaded, forecast_hour, fetch_product.as_str())
        .ok_or("windowed sampling missing surface bundle for query grid")?;
    let surface_grid = decode_surface_grid(&geometry.file.bytes)?;
    let grid = rustwx_core::LatLonGrid::new(
        rustwx_core::GridShape::new(surface_grid.nx, surface_grid.ny)?,
        surface_grid
            .lat
            .iter()
            .copied()
            .map(|value| value as f32)
            .collect(),
        surface_grid
            .lon
            .iter()
            .copied()
            .map(|value| value as f32)
            .collect(),
    )?;
    let (mut apcp_by_hour, mut surface_hour_fetches, _, _) =
        load_apcp_hours_from_plan(Some(&loaded), &request, &surface_hours)?;
    let fallback_surface_hours = planned_products
        .iter()
        .filter(|product| product.is_qpf())
        .flat_map(|&product| {
            qpf_hourly_fallback_hours_if_allowed(
                latest.model,
                product,
                forecast_hour,
                &apcp_by_hour,
            )
            .unwrap_or_default()
        })
        .filter(|hour| !surface_hours.contains(hour))
        .collect::<BTreeSet<_>>();
    if !fallback_surface_hours.is_empty() {
        let (fallback_apcp_by_hour, mut fallback_surface_hour_fetches, _, _) =
            load_apcp_fallback_hours(Some(&loaded), &request, latest, &fallback_surface_hours)?;
        apcp_by_hour.extend(fallback_apcp_by_hour);
        surface_hour_fetches.append(&mut fallback_surface_hour_fetches);
        surface_hours.extend(fallback_surface_hours);
    }
    let (uh_by_hour, uh_hour_fetches, _, _) =
        load_uh_hours_from_plan(Some(&loaded), &request, &nat_hours)?;
    let (wind_by_hour, wind_hour_fetches, _, _) =
        load_wind10m_hours_from_plan(Some(&loaded), &request, &wind_hours)?;
    let (snapshot_by_hour, temp_hour_fetches, _, _) =
        load_surface_snapshot_hours_from_plan(Some(&loaded), &request, &temp_hours)?;

    let mut fields = Vec::new();
    for &product in &planned_products {
        let computed = if product.is_qpf() {
            compute_qpf_product(product, forecast_hour, &grid, &apcp_by_hour)
        } else if product.is_wind10m() {
            compute_wind10m_product(product, forecast_hour, &grid, &wind_by_hour)
        } else if product.is_surface_snapshot() {
            compute_surface_snapshot_product(product, &grid, &snapshot_by_hour)
        } else {
            compute_uh_product(product, forecast_hour, &grid, &uh_by_hour)
        };
        match computed {
            Ok(computed) => fields.push(WindowedSampledProductField {
                product,
                input_fetches: input_fetches_for_windowed_product(
                    product,
                    &computed.metadata.contributing_forecast_hours,
                    &surface_hour_fetches,
                    &uh_hour_fetches,
                    &wind_hour_fetches,
                    &temp_hour_fetches,
                ),
                field: computed.field,
            }),
            Err(reason) => blockers.push(HrrrWindowedBlocker { product, reason }),
        }
    }

    Ok(WindowedSampledProductSet { fields, blockers })
}

fn sampling_windowed_request(
    model: ModelId,
    forecast_hour: u16,
    source: SourceId,
    cache_root: &std::path::Path,
    use_cache: bool,
) -> HrrrWindowedBatchRequest {
    HrrrWindowedBatchRequest {
        model,
        date_yyyymmdd: String::new(),
        cycle_override_utc: None,
        forecast_hour,
        source,
        domain: DomainSpec::new("sampling", (-180.0, 180.0, -90.0, 90.0)),
        out_dir: PathBuf::new(),
        cache_root: cache_root.to_path_buf(),
        use_cache,
        products: Vec::new(),
        output_width: OUTPUT_WIDTH,
        output_height: OUTPUT_HEIGHT,
        png_compression: PngCompressionMode::Default,
        place_label_overlay: None,
        subtitle_left_suffix: None,
        subtitle_right_override: None,
        title_provenance: TitleProvenance::default(),
    }
}

fn input_fetches_for_windowed_product(
    product: HrrrWindowedProduct,
    contributing_forecast_hours: &[u16],
    surface_hour_fetches: &[HrrrWindowedHourFetchInfo],
    uh_hour_fetches: &[HrrrWindowedHourFetchInfo],
    wind_hour_fetches: &[HrrrWindowedHourFetchInfo],
    temp_hour_fetches: &[HrrrWindowedHourFetchInfo],
) -> Vec<PublishedFetchIdentity> {
    let fetches = if product.is_qpf() {
        surface_hour_fetches
    } else if product.is_wind10m() {
        wind_hour_fetches
    } else if product.is_surface_snapshot() {
        temp_hour_fetches
    } else {
        uh_hour_fetches
    };
    let mut by_key = BTreeMap::<String, PublishedFetchIdentity>::new();
    for fetch in fetches
        .iter()
        .filter(|fetch| contributing_forecast_hours.contains(&fetch.hour))
    {
        if let Some(identity) = fetch.input_fetch.clone() {
            by_key.entry(identity.fetch_key.clone()).or_insert(identity);
        }
    }
    by_key.into_values().collect()
}

pub fn run_hrrr_windowed_batch(
    request: &HrrrWindowedBatchRequest,
) -> Result<HrrrWindowedBatchReport, Box<dyn std::error::Error>> {
    fs::create_dir_all(&request.out_dir)?;
    if request.use_cache {
        fs::create_dir_all(&request.cache_root)?;
    }

    let latest = resolve_model_run(
        request.model,
        &request.date_yyyymmdd,
        request.cycle_override_utc,
        request.forecast_hour,
        request.source,
    )?;
    run_hrrr_windowed_batch_with_context(request, &latest)
}

pub(crate) fn run_hrrr_windowed_batch_with_context(
    request: &HrrrWindowedBatchRequest,
    latest: &rustwx_models::LatestRun,
) -> Result<HrrrWindowedBatchReport, Box<dyn std::error::Error>> {
    fs::create_dir_all(&request.out_dir)?;
    if request.use_cache {
        fs::create_dir_all(&request.cache_root)?;
    }

    let total_start = Instant::now();
    let prepared = prepare_hrrr_windowed_batch_with_context(request, latest)?;
    run_hrrr_windowed_batch_from_prepared_with_total_start(request, &prepared, total_start)
}

pub(crate) fn prepare_hrrr_windowed_batch_with_context(
    request: &HrrrWindowedBatchRequest,
    latest: &rustwx_models::LatestRun,
) -> Result<PreparedWindowedBatch, Box<dyn std::error::Error>> {
    fs::create_dir_all(&request.out_dir)?;
    if request.use_cache {
        fs::create_dir_all(&request.cache_root)?;
    }

    let total_start = Instant::now();
    let geometry_context = prepare_windowed_geometry_context(request, latest)?;
    let fetch_geometry_ms = geometry_context.fetch_geometry_ms;
    let decode_geometry_ms = geometry_context.decode_geometry_ms;
    let geometry_fetch_cache_hit = geometry_context.geometry_fetch_cache_hit;
    let geometry_decode_cache_hit = geometry_context.geometry_decode_cache_hit;
    let geometry_fetch = geometry_context.geometry_fetch;
    let geometry_input_fetch = geometry_context.geometry_input_fetch;
    let _source_projected = geometry_context.projected;
    let grid = geometry_context.grid;
    let projection = geometry_context.projection;

    let (planned_products, mut blockers, planned_surface_hours, nat_hours, wind_hours, temp_hours) =
        plan_windowed_products(
            &request.products,
            request.forecast_hour,
            Some(latest.cycle.hour_utc),
        );

    let mut surface_hours = if planned_products.iter().any(|product| product.is_qpf()) {
        let mut hours = BTreeSet::new();
        hours.insert(request.forecast_hour);
        hours
    } else {
        planned_surface_hours.clone()
    };
    let loaded = load_windowed_plan_for_hours(
        request,
        latest,
        &surface_hours,
        &nat_hours,
        &wind_hours,
        &temp_hours,
    )?;

    let (mut apcp_by_hour, mut surface_hour_fetches, mut fetch_surface_ms, mut decode_surface_ms) =
        load_apcp_hours_from_plan(loaded.as_ref(), request, &surface_hours)?;
    let fallback_surface_hours = planned_products
        .iter()
        .filter(|product| product.is_qpf())
        .flat_map(|&product| {
            qpf_hourly_fallback_hours_if_allowed(
                latest.model,
                product,
                request.forecast_hour,
                &apcp_by_hour,
            )
            .unwrap_or_default()
        })
        .filter(|hour| !surface_hours.contains(hour))
        .collect::<BTreeSet<_>>();
    if !fallback_surface_hours.is_empty() {
        let (
            fallback_apcp_by_hour,
            mut fallback_surface_hour_fetches,
            fallback_fetch_surface_ms,
            fallback_decode_surface_ms,
        ) = load_apcp_fallback_hours(loaded.as_ref(), request, latest, &fallback_surface_hours)?;
        apcp_by_hour.extend(fallback_apcp_by_hour);
        surface_hour_fetches.append(&mut fallback_surface_hour_fetches);
        fetch_surface_ms += fallback_fetch_surface_ms;
        decode_surface_ms += fallback_decode_surface_ms;
        surface_hours.extend(fallback_surface_hours);
    }
    let (uh_by_hour, uh_hour_fetches, fetch_nat_ms, decode_nat_ms) =
        load_uh_hours_from_plan(loaded.as_ref(), request, &nat_hours)?;
    let (wind_by_hour, wind_hour_fetches, fetch_wind_ms, decode_wind_ms) =
        load_wind10m_hours_from_plan(loaded.as_ref(), request, &wind_hours)?;
    let (snapshot_by_hour, temp_hour_fetches, fetch_temp_ms, decode_temp_ms) =
        load_surface_snapshot_hours_from_plan(loaded.as_ref(), request, &temp_hours)?;

    let compute_start = Instant::now();
    let (products, compute_blockers) = compute_prepared_windowed_products(
        latest.source,
        &planned_products,
        request.forecast_hour,
        &grid,
        &apcp_by_hour,
        &uh_by_hour,
        &wind_by_hour,
        &snapshot_by_hour,
    )?;
    let compute_products_ms = compute_start.elapsed().as_millis();
    blockers.extend(compute_blockers);

    let shared_timing = HrrrWindowedSharedTiming {
        fetch_geometry_ms,
        decode_geometry_ms,
        project_ms: 0,
        fetch_surface_ms,
        decode_surface_ms,
        fetch_nat_ms,
        decode_nat_ms,
        fetch_wind_ms,
        decode_wind_ms,
        fetch_temp_ms,
        decode_temp_ms,
        compute_products_ms,
        geometry_fetch_cache_hit,
        geometry_decode_cache_hit,
        surface_hours_loaded: surface_hours.into_iter().collect(),
        nat_hours_loaded: nat_hours.into_iter().collect(),
        wind_hours_loaded: wind_hours.into_iter().collect(),
        temp_hours_loaded: temp_hours.into_iter().collect(),
        geometry_fetch,
        geometry_input_fetch,
        surface_hour_fetches,
        uh_hour_fetches,
        wind_hour_fetches,
        temp_hour_fetches,
    };

    let _prepare_ms = total_start.elapsed().as_millis();
    Ok(PreparedWindowedBatch {
        latest: latest.clone(),
        shared_timing,
        products,
        blockers,
        grid,
        projection,
    })
}

fn compute_prepared_windowed_products(
    source: SourceId,
    planned_products: &[HrrrWindowedProduct],
    forecast_hour: u16,
    grid: &rustwx_core::LatLonGrid,
    apcp_by_hour: &BTreeMap<u16, Result<HrrrApcpDecode, String>>,
    uh_by_hour: &BTreeMap<u16, Result<HrrrUhDecode, String>>,
    wind_by_hour: &BTreeMap<u16, Result<HrrrWind10mMaxDecode, String>>,
    snapshot_by_hour: &BTreeMap<u16, Result<HrrrSurfaceSnapshotDecode, String>>,
) -> Result<(Vec<PreparedWindowedProduct>, Vec<HrrrWindowedBlocker>), Box<dyn std::error::Error>> {
    let product_parallelism = windowed_parallelism(source, planned_products.len());
    let mut outcomes = thread::scope(|scope| -> Result<Vec<WindowedComputeOutcome>, io::Error> {
        let mut done = Vec::with_capacity(planned_products.len());
        let mut pending = std::collections::VecDeque::new();

        for (index, &product) in planned_products.iter().enumerate() {
            pending.push_back(
                scope.spawn(move || -> Result<WindowedComputeOutcome, io::Error> {
                    let computed = if product.is_qpf() {
                        compute_qpf_product(product, forecast_hour, grid, apcp_by_hour)
                    } else if product.is_wind10m() {
                        compute_wind10m_product(product, forecast_hour, grid, wind_by_hour)
                    } else if product.is_surface_snapshot() {
                        compute_surface_snapshot_product(product, grid, snapshot_by_hour)
                    } else {
                        compute_uh_product(product, forecast_hour, grid, uh_by_hour)
                    };

                    let computed = match computed {
                        Ok(value) => value,
                        Err(reason) => {
                            return Ok(WindowedComputeOutcome::Blocker {
                                index,
                                blocker: HrrrWindowedBlocker { product, reason },
                            });
                        }
                    };

                    Ok(WindowedComputeOutcome::Computed {
                        index,
                        prepared: PreparedWindowedProduct { product, computed },
                    })
                }),
            );

            if pending.len() >= product_parallelism {
                done.push(join_windowed_job(pending.pop_front().unwrap())?);
            }
        }

        while let Some(handle) = pending.pop_front() {
            done.push(join_windowed_job(handle)?);
        }

        Ok(done)
    })?;
    outcomes.sort_by_key(|outcome| match outcome {
        WindowedComputeOutcome::Computed { index, .. } => *index,
        WindowedComputeOutcome::Blocker { index, .. } => *index,
    });
    let mut products = Vec::new();
    let mut blockers = Vec::new();
    for outcome in outcomes {
        match outcome {
            WindowedComputeOutcome::Computed { prepared, .. } => products.push(prepared),
            WindowedComputeOutcome::Blocker { blocker, .. } => blockers.push(blocker),
        }
    }
    Ok((products, blockers))
}

pub(crate) fn run_hrrr_windowed_batch_from_prepared(
    request: &HrrrWindowedBatchRequest,
    prepared: &PreparedWindowedBatch,
) -> Result<HrrrWindowedBatchReport, Box<dyn std::error::Error>> {
    run_hrrr_windowed_batch_from_prepared_with_total_start(request, prepared, Instant::now())
}

/// One windowed product grid computed outside this module (by the rw-store
/// windowed lane, which mirrors this lane's window semantics over stored
/// per-hour fields) ready to render through the windowed lane's own render
/// path. `values` are full-grid row-major display values already in the
/// product's display units (`units`); `hours_used`/`window_hours`/`strategy`
/// feed the same metadata the GRIB kernels emit (the subtitle hour label and
/// the report strategy line).  `title` is what the picture is titled: the
/// product's own title, or a [`SnapshotFoldRow`]'s when the window was
/// folded from hourly snapshots.
#[derive(Debug, Clone)]
pub struct StoreWindowedGrid {
    pub slug: String,
    pub title: String,
    pub units: String,
    pub values: Vec<f64>,
    pub hours_used: Vec<u16>,
    pub window_hours: Option<u16>,
    pub strategy: String,
}

/// Render windowed product grids computed from the store through the EXACT
/// render half of the GRIB windowed batch (`run_hrrr_windowed_batch_from_prepared`:
/// same domain crop, projected map, per-product scales/titles/subtitles,
/// place-label overlay, and PNG save). No fetch, no decode, no compute:
/// the grids already exist; the prepared batch is assembled directly with
/// zeroed fetch/decode timing.
pub fn render_windowed_products_from_store_grids(
    request: &HrrrWindowedBatchRequest,
    cycle_utc: u8,
    full_grid: &rustwx_core::LatLonGrid,
    projection: Option<&rustwx_core::GridProjection>,
    grids: &[StoreWindowedGrid],
) -> Result<Vec<HrrrWindowedRenderedProduct>, Box<dyn std::error::Error>> {
    let latest = LatestRun {
        model: request.model,
        cycle: rustwx_core::CycleSpec::new(request.date_yyyymmdd.clone(), cycle_utc)?,
        source: request.source,
    };
    // An explicit ceiling keeps all frames of a cyclone sheet comparable
    // without saturating hurricane-force cores at the default 70 kt.
    let wind_ceiling = std::env::var("RUSTWX_WINDOWED_WIND_MAX_KT").ok()
        .map(|text| text.parse::<f64>().map_err(|_| "RUSTWX_WINDOWED_WIND_MAX_KT must be numeric"))
        .transpose()?;
    if wind_ceiling.is_some_and(|value| !value.is_finite() || value <= 0.) {
        return Err("RUSTWX_WINDOWED_WIND_MAX_KT must be finite and positive".into());
    }
    let cells = full_grid.shape.len();
    let mut products = Vec::with_capacity(grids.len());
    for grid in grids {
        let product = HrrrWindowedProduct::from_slug(&grid.slug)
            .ok_or_else(|| format!("'{}' is not a windowed product slug", grid.slug))?;
        if grid.values.len() != cells {
            return Err(format!(
                "windowed store grid '{}' holds {} values, expected {cells}",
                grid.slug,
                grid.values.len()
            )
            .into());
        }
        let field = Field2D::new(
            rustwx_core::ProductKey::named(product.slug()),
            grid.units.clone(),
            full_grid.clone(),
            grid.values.iter().map(|&value| value as f32).collect(),
        )?;
        products.push(PreparedWindowedProduct {
            product,
            computed: crate::windowed_decoder::ComputedWindowedField {
                field,
                title: grid.title.clone(),
                metadata: HrrrWindowedProductMetadata {
                    strategy: grid.strategy.clone(),
                    contributing_forecast_hours: grid.hours_used.clone(),
                    window_hours: grid.window_hours,
                },
                scale: {
                    let mut scale = crate::windowed_decoder::windowed_product_scale(product);
                    if let Some(ceiling) = wind_ceiling.filter(|_| product.is_wind10m()) {
                        let mut discrete = scale.resolved_discrete();
                        let intervals = discrete.levels.len() - 1;
                        discrete.levels = (0..=intervals)
                            .map(|i| ceiling * i as f64 / intervals as f64).collect();
                        discrete.mask_below = None;
                        scale = rustwx_render::ColorScale::Discrete(discrete);
                    }
                    scale
                },
            },
        });
    }
    let prepared = PreparedWindowedBatch {
        latest,
        shared_timing: empty_windowed_shared_timing(),
        products,
        blockers: Vec::new(),
        grid: full_grid.clone(),
        projection: projection.cloned(),
    };
    Ok(run_hrrr_windowed_batch_from_prepared(request, &prepared)?.products)
}

/// Shared-timing placeholder for store-rendered windowed batches: nothing
/// was fetched or decoded, so every counter is zero and every fetch list
/// empty.
fn empty_windowed_shared_timing() -> HrrrWindowedSharedTiming {
    HrrrWindowedSharedTiming {
        fetch_geometry_ms: 0,
        decode_geometry_ms: 0,
        project_ms: 0,
        fetch_surface_ms: 0,
        decode_surface_ms: 0,
        fetch_nat_ms: 0,
        decode_nat_ms: 0,
        fetch_wind_ms: 0,
        decode_wind_ms: 0,
        fetch_temp_ms: 0,
        decode_temp_ms: 0,
        compute_products_ms: 0,
        geometry_fetch_cache_hit: false,
        geometry_decode_cache_hit: false,
        surface_hours_loaded: Vec::new(),
        nat_hours_loaded: Vec::new(),
        wind_hours_loaded: Vec::new(),
        temp_hours_loaded: Vec::new(),
        geometry_fetch: None,
        geometry_input_fetch: None,
        surface_hour_fetches: Vec::new(),
        uh_hour_fetches: Vec::new(),
        wind_hour_fetches: Vec::new(),
        temp_hour_fetches: Vec::new(),
    }
}

fn run_hrrr_windowed_batch_from_prepared_with_total_start(
    request: &HrrrWindowedBatchRequest,
    prepared: &PreparedWindowedBatch,
    total_start: Instant,
) -> Result<HrrrWindowedBatchReport, Box<dyn std::error::Error>> {
    fs::create_dir_all(&request.out_dir)?;
    if request.use_cache {
        fs::create_dir_all(&request.cache_root)?;
    }

    let crop = crop_for_domain_grid(
        &prepared.grid,
        request.domain.bounds,
        windowed_domain_crop_pad_cells(),
    )?;
    let domain_grid = if let Some(crop) = crop {
        crop_latlon_grid(&prepared.grid, crop)?
    } else {
        prepared.grid.clone()
    };

    let project_start = Instant::now();
    // One ratio local feeds the map build and the published projection
    // below, so the two can never disagree.
    let target_ratio =
        map_frame_aspect_ratio(request.output_width, request.output_height, true, true);
    let projected = crate::direct::build_projected_map_with_projection(
        &domain_grid.lat_deg,
        &domain_grid.lon_deg,
        prepared.projection.as_ref(),
        request.domain.bounds,
        target_ratio,
    )?;
    // gpuwm addition (VENDOR.md): the transform every windowed panel of
    // this batch publishes, resolved once from EXACTLY the arguments the
    // map build above took.
    let resolved_projection = crate::direct::panel_resolved_projection(
        &domain_grid.lat_deg,
        &domain_grid.lon_deg,
        prepared.projection.as_ref(),
        request.domain.bounds,
        target_ratio,
    )?;
    let project_ms = project_start.elapsed().as_millis();

    let product_parallelism = windowed_parallelism(prepared.latest.source, prepared.products.len());
    let date_yyyymmdd = request.date_yyyymmdd.as_str();
    let cycle_utc = prepared.latest.cycle.hour_utc;
    let forecast_hour = request.forecast_hour;
    let domain_slug = request.domain.slug.as_str();
    let out_dir = &request.out_dir;
    let model = prepared.latest.model;
    let source = prepared.latest.source;
    let projected = &projected;
    let projection = prepared.projection.as_ref();
    let domain_grid = &domain_grid;
    let prepared_products = &prepared.products;
    let mut outcomes = thread::scope(|scope| -> Result<Vec<WindowedProductOutcome>, io::Error> {
        let mut done = Vec::with_capacity(prepared_products.len());
        let mut pending = std::collections::VecDeque::new();

        for (index, prepared_product) in prepared_products.iter().enumerate() {
            pending.push_back(
                scope.spawn(move || -> Result<WindowedProductOutcome, io::Error> {
                    let product = prepared_product.product;
                    let computed = cropped_windowed_field_for_domain(
                        &prepared_product.computed,
                        domain_grid,
                        crop,
                    )
                    .map_err(thread_windowed_error)?;
                    let output_path = out_dir.join(format!(
                        "rustwx_{}_{}_{}z_f{:03}_{}_{}.png",
                        model.as_str().replace('-', "_"),
                        date_yyyymmdd,
                        cycle_utc,
                        forecast_hour,
                        domain_slug,
                        product.slug()
                    ));
                    let render_start = Instant::now();
                    let mut render_request = build_windowed_render_request(
                        product,
                        &computed,
                        request,
                        projected,
                        date_yyyymmdd,
                        cycle_utc,
                        forecast_hour,
                        model,
                        source,
                    );
                    render_request.resolved_projection = Some(resolved_projection);
                    render_request.geographic_bounds = Some(request.domain.bounds);
                    if let Some(overlay) = request.place_label_overlay.as_ref() {
                        crate::apply_place_label_overlay_with_density_styling(
                            &mut render_request,
                            overlay,
                            &request.domain,
                            &computed.field.grid.lat_deg,
                            &computed.field.grid.lon_deg,
                            projection,
                        )
                        .map_err(thread_windowed_error)?;
                    }
                    let save_timing = save_png_profile_with_options(
                        &render_request,
                        &output_path,
                        &request.png_write_options(),
                    )
                    .map_err(thread_windowed_error)?;
                    let render_ms = render_start.elapsed().as_millis();

                    Ok(WindowedProductOutcome::Rendered {
                        index,
                        rendered: HrrrWindowedRenderedProduct {
                            product,
                            output_path,
                            timing: HrrrWindowedProductTiming {
                                compute_ms: 0,
                                render_ms,
                                total_ms: render_ms,
                            },
                            metadata: computed.metadata,
                            georeference: save_timing.georeference,
                            georeference_absent_reason: save_timing.georeference_absent_reason,
                        },
                    })
                }),
            );

            if pending.len() >= product_parallelism {
                done.push(join_windowed_job(pending.pop_front().unwrap())?);
            }
        }

        while let Some(handle) = pending.pop_front() {
            done.push(join_windowed_job(handle)?);
        }

        Ok(done)
    })?;
    outcomes.sort_by_key(|outcome| match outcome {
        WindowedProductOutcome::Rendered { index, .. } => *index,
    });
    let mut rendered = Vec::new();
    let blockers = prepared.blockers.clone();
    for outcome in outcomes {
        match outcome {
            WindowedProductOutcome::Rendered { rendered: item, .. } => rendered.push(item),
        }
    }
    let mut shared_timing = prepared.shared_timing.clone();
    shared_timing.project_ms = project_ms;

    Ok(HrrrWindowedBatchReport {
        model: prepared.latest.model,
        date_yyyymmdd: request.date_yyyymmdd.clone(),
        cycle_utc: prepared.latest.cycle.hour_utc,
        forecast_hour: request.forecast_hour,
        source: prepared.latest.source,
        domain: request.domain.clone(),
        shared_timing,
        products: rendered,
        blockers,
        total_ms: total_start.elapsed().as_millis(),
    })
}

fn windowed_domain_crop_pad_cells() -> usize {
    std::env::var("RUSTWX_DOMAIN_CROP_PAD_CELLS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(6)
}

fn crop_for_domain_grid(
    grid: &rustwx_core::LatLonGrid,
    bounds: (f64, f64, f64, f64),
    pad_cells: usize,
) -> Result<Option<GridCrop>, Box<dyn std::error::Error>> {
    let nx = grid.shape.nx;
    let ny = grid.shape.ny;
    if nx == 0 || ny == 0 {
        return Ok(None);
    }

    let mut min_x = nx;
    let mut max_x = 0usize;
    let mut min_y = ny;
    let mut max_y = 0usize;
    let mut found = false;

    for y in 0..ny {
        let row_offset = y * nx;
        for x in 0..nx {
            let idx = row_offset + x;
            let lat = grid.lat_deg[idx] as f64;
            let lon = grid.lon_deg[idx] as f64;
            if point_in_geographic_bounds(lon, lat, bounds) {
                min_x = min_x.min(x);
                max_x = max_x.max(x);
                min_y = min_y.min(y);
                max_y = max_y.max(y);
                found = true;
            }
        }
    }

    if !found {
        return Ok(None);
    }

    let crop = GridCrop {
        x_start: min_x.saturating_sub(pad_cells),
        x_end: (max_x + 1 + pad_cells).min(nx),
        y_start: min_y.saturating_sub(pad_cells),
        y_end: (max_y + 1 + pad_cells).min(ny),
    };

    if crop.x_start == 0 && crop.x_end == nx && crop.y_start == 0 && crop.y_end == ny {
        Ok(None)
    } else {
        Ok(Some(crop))
    }
}

fn cropped_windowed_field_for_domain(
    computed: &crate::windowed_decoder::ComputedWindowedField,
    domain_grid: &rustwx_core::LatLonGrid,
    crop: Option<GridCrop>,
) -> Result<crate::windowed_decoder::ComputedWindowedField, Box<dyn std::error::Error>> {
    let field = if let Some(crop) = crop {
        Field2D::new(
            computed.field.product.clone(),
            computed.field.units.clone(),
            domain_grid.clone(),
            crop_values_f32(&computed.field.values, computed.field.grid.shape.nx, crop),
        )?
    } else {
        computed.field.clone()
    };

    Ok(crate::windowed_decoder::ComputedWindowedField {
        field,
        title: computed.title.clone(),
        metadata: computed.metadata.clone(),
        scale: computed.scale.clone(),
    })
}

fn point_in_geographic_bounds(lon: f64, lat: f64, bounds: (f64, f64, f64, f64)) -> bool {
    if !lon.is_finite() || !lat.is_finite() || lat < bounds.2 || lat > bounds.3 {
        return false;
    }
    // Native projected grids can cross the date line. Their padded
    // coordinate envelope may cover a full revolution; wrapping its two
    // endpoints first would turn that envelope into a narrow date-line
    // strip and crop windowed fields into the wrong hemisphere.
    if (bounds.1 - bounds.0).abs() >= 359.0 {
        return true;
    }
    let west = normalize_longitude_for_bounds(bounds.0);
    let east = normalize_longitude_for_bounds(bounds.1);
    let lon = normalize_longitude_for_bounds(lon);
    if west <= east {
        lon >= west && lon <= east
    } else {
        lon >= west || lon <= east
    }
}

fn normalize_longitude_for_bounds(lon: f64) -> f64 {
    let mut lon = lon % 360.0;
    if lon > 180.0 {
        lon -= 360.0;
    } else if lon <= -180.0 {
        lon += 360.0;
    }
    lon
}

fn build_windowed_render_request(
    product: HrrrWindowedProduct,
    computed: &crate::windowed_decoder::ComputedWindowedField,
    request: &HrrrWindowedBatchRequest,
    projected: &ProjectedMap,
    date_yyyymmdd: &str,
    cycle_utc: u8,
    forecast_hour: u16,
    model: ModelId,
    source: SourceId,
) -> MapRenderRequest {
    let mut render_request = if product.is_uh() {
        // The UH tracks: the UH ramp without the haze below 25 m2/s2, outlined
        // at 75 (plot_design::apply_uh_track_style); RUSTWX_UH_FILL=off keeps
        // the 2.9.13 look.
        let mut tracks =
            MapRenderRequest::for_core_weather_product(computed.field.clone(), WeatherProduct::Uh);
        crate::plot_design::apply_uh_track_style(&mut tracks, &computed.field.values);
        tracks
    } else {
        MapRenderRequest::from_core_field(computed.field.clone(), computed.scale.clone())
    };
    render_request.width = request.output_width;
    render_request.height = request.output_height;
    render_request.title = Some(static_title_with_suffix(
        request
            .title_provenance
            .apply_to_title(computed.title.clone()),
    ));
    let hour_label = windowed_display_hour_label(product, &computed.metadata, forecast_hour);
    let subtitle_left = model_time_subtitle_with_lead_label(
        model,
        date_yyyymmdd,
        cycle_utc,
        forecast_hour,
        hour_label,
    );
    render_request.subtitle_left = Some(match request.subtitle_left_suffix.as_deref() {
        Some(suffix) => format!("{subtitle_left} | {suffix}"),
        None => subtitle_left,
    });
    render_request.subtitle_right = Some(
        request
            .subtitle_right_override
            .clone()
            .unwrap_or_else(|| source_subtitle(source)),
    );
    render_request.chrome_scale = static_chrome_scale();
    render_request.supersample_factor = static_supersample_factor();
    render_request.supersample_sharpen = static_supersample_sharpen();
    let visual_mode = if product.is_qpf() || product.is_wind10m() || product.is_surface_snapshot() {
        ProductVisualMode::FilledMeteorology
    } else {
        ProductVisualMode::SevereDiagnostic
    };
    crate::plot_design::StaticPlotDesign::new(request.domain.bounds, visual_mode)
        .apply_to_request(&mut render_request);
    if matches!(
        product,
        HrrrWindowedProduct::Dewpoint2m0to24hMax
            | HrrrWindowedProduct::Dewpoint2m24to48hMax
            | HrrrWindowedProduct::Dewpoint2m0to48hMax
            | HrrrWindowedProduct::Dewpoint2m0to24hMin
            | HrrrWindowedProduct::Dewpoint2m24to48hMin
            | HrrrWindowedProduct::Dewpoint2m0to48hMin
            | HrrrWindowedProduct::Dewpoint2m0to24hRange
            | HrrrWindowedProduct::Dewpoint2m24to48hRange
            | HrrrWindowedProduct::Dewpoint2m0to48hRange
    ) {
        render_request.legend.mode = LegendMode::Stepped;
    }
    if product.is_qpf() {
        // One listed colour per band, every band the same length on the bar
        // (`crate::qpf::qpf_inches_scale` names the breakage).
        render_request.legend.mode = crate::qpf::qpf_legend_mode();
    }
    render_request.projected_domain = Some(rustwx_render::ProjectedDomain {
        x: projected.projected_x.clone(),
        y: projected.projected_y.clone(),
        extent: projected.extent.clone(),
    });
    render_request.projected_lines = projected.lines.clone();
    render_request.projected_polygons = projected.polygons.clone();
    // gpuwm divergence (VENDOR.md): a regular lat/lon grid draws through
    // the map's inverse raster, as the direct lane's panel beside it does.
    // Without it a global field's contour bands closed across the whole
    // map at the date line, and a regional crop framed itself to the strip
    // inscribed in its curved footprint.  `None` for every projected grid
    // (HRRR, WRF Lambert), whose pictures this leaves byte-identical.
    render_request.inverse_raster_projection = projected.inverse_raster_projection.clone();
    render_request
}

fn windowed_display_hour_label(
    product: HrrrWindowedProduct,
    metadata: &HrrrWindowedProductMetadata,
    forecast_hour: u16,
) -> String {
    if let Some((start_hour, end_hour, _)) = surface_snapshot_window_hours(product) {
        return format!("F{start_hour:03}-F{end_hour:03}");
    }
    match metadata.contributing_forecast_hours.as_slice() {
        [] => format!("F{forecast_hour:03}"),
        [hour] => {
            if let Some(window_hours) = metadata.window_hours.filter(|window| *window > 1) {
                let start_hour = forecast_hour.saturating_add(1).saturating_sub(window_hours);
                format!("F{start_hour:03}-F{forecast_hour:03}")
            } else {
                format!("F{hour:03}")
            }
        }
        hours => {
            let start_hour = hours.first().copied().unwrap_or(forecast_hour);
            let end_hour = hours.last().copied().unwrap_or(forecast_hour);
            if start_hour == end_hour {
                format!("F{end_hour:03}")
            } else {
                format!("F{start_hour:03}-F{end_hour:03}")
            }
        }
    }
}

fn plan_windowed_products(
    products: &[HrrrWindowedProduct],
    forecast_hour: u16,
    _cycle_utc: Option<u8>,
) -> (
    Vec<HrrrWindowedProduct>,
    Vec<HrrrWindowedBlocker>,
    BTreeSet<u16>,
    BTreeSet<u16>,
    BTreeSet<u16>,
    BTreeSet<u16>,
) {
    let mut seen = BTreeSet::new();
    let mut planned = Vec::new();
    let mut blockers = Vec::new();
    let mut surface_hours = BTreeSet::new();
    let mut nat_hours = BTreeSet::new();
    let mut wind_hours = BTreeSet::new();
    let mut temp_hours = BTreeSet::new();

    for &product in products {
        if !seen.insert(product.slug().to_string()) {
            continue;
        }
        if let Some((start_hour, end_hour, label)) = surface_snapshot_window_hours(product) {
            if forecast_hour < end_hour {
                blockers.push(blocker(
                    product,
                    format!("{label} requires forecast hour >= {end_hour}"),
                ));
                continue;
            }
            temp_hours.extend(start_hour..=end_hour);
            planned.push(product);
            continue;
        }

        match product {
            HrrrWindowedProduct::Qpf1h => {
                if forecast_hour < 1 {
                    blockers.push(blocker(
                        product,
                        "1-h QPF requires forecast hour >= 1 because the first 1 h accumulation window ends at F001",
                    ));
                    continue;
                }
                surface_hours.insert(forecast_hour);
            }
            HrrrWindowedProduct::Qpf6h => {
                if forecast_hour < 6 {
                    blockers.push(blocker(product, "6-h QPF requires forecast hour >= 6"));
                    continue;
                }
                surface_hours.extend((forecast_hour - 5)..=forecast_hour);
            }
            HrrrWindowedProduct::Qpf12h => {
                if forecast_hour < 12 {
                    blockers.push(blocker(product, "12-h QPF requires forecast hour >= 12"));
                    continue;
                }
                surface_hours.extend((forecast_hour - 11)..=forecast_hour);
            }
            HrrrWindowedProduct::Qpf24h => {
                if forecast_hour < 24 {
                    blockers.push(blocker(product, "24-h QPF requires forecast hour >= 24"));
                    continue;
                }
                surface_hours.extend((forecast_hour - 23)..=forecast_hour);
            }
            HrrrWindowedProduct::QpfTotal => {
                if forecast_hour < 1 {
                    blockers.push(blocker(product, "total QPF requires forecast hour >= 1"));
                    continue;
                }
                surface_hours.extend(1..=forecast_hour);
            }
            HrrrWindowedProduct::Uh25km1h => {
                if forecast_hour < 1 {
                    blockers.push(blocker(
                        product,
                        "1-h UH max requires forecast hour >= 1 because native UH windows start at 0-1 h",
                    ));
                    continue;
                }
                nat_hours.insert(forecast_hour);
            }
            HrrrWindowedProduct::Uh25km3h => {
                if forecast_hour < 3 {
                    blockers.push(blocker(product, "3-h UH max requires forecast hour >= 3"));
                    continue;
                }
                nat_hours.extend((forecast_hour - 2)..=forecast_hour);
            }
            HrrrWindowedProduct::Uh25kmRunMax => {
                if forecast_hour < 1 {
                    blockers.push(blocker(product, "run-max UH requires forecast hour >= 1"));
                    continue;
                }
                nat_hours.extend(1..=forecast_hour);
            }
            HrrrWindowedProduct::Wind10m1hMax => {
                if forecast_hour < 1 {
                    blockers.push(blocker(
                        product,
                        "1-h 10 m wind max requires forecast hour >= 1 because native wind max windows start at 0-1 h",
                    ));
                    continue;
                }
                wind_hours.insert(forecast_hour);
            }
            HrrrWindowedProduct::Wind10mRunMax => {
                if forecast_hour < 1 {
                    blockers.push(blocker(
                        product,
                        "run-max 10 m wind requires forecast hour >= 1",
                    ));
                    continue;
                }
                wind_hours.extend(1..=forecast_hour);
            }
            HrrrWindowedProduct::Wind10m0to24hMax => {
                if forecast_hour < 24 {
                    blockers.push(blocker(
                        product,
                        "0-24 h 10 m wind max requires forecast hour >= 24",
                    ));
                    continue;
                }
                wind_hours.extend(1..=24);
            }
            HrrrWindowedProduct::Wind10m24to48hMax => {
                if forecast_hour < 48 {
                    blockers.push(blocker(
                        product,
                        "24-48 h 10 m wind max requires forecast hour >= 48",
                    ));
                    continue;
                }
                wind_hours.extend(25..=48);
            }
            HrrrWindowedProduct::Wind10m0to48hMax => {
                if forecast_hour < 48 {
                    blockers.push(blocker(
                        product,
                        "0-48 h 10 m wind max requires forecast hour >= 48",
                    ));
                    continue;
                }
                wind_hours.extend(1..=48);
            }
            _ => unreachable!("surface snapshot window products are handled before match"),
        }

        planned.push(product);
    }

    (
        planned,
        blockers,
        surface_hours,
        nat_hours,
        wind_hours,
        temp_hours,
    )
}

fn surface_snapshot_window_hours(product: HrrrWindowedProduct) -> Option<(u16, u16, &'static str)> {
    use HrrrWindowedProduct::*;
    match product {
        Temp2m0to24hMax
        | Temp2m0to24hMin
        | Temp2m0to24hRange
        | Rh2m0to24hMax
        | Rh2m0to24hMin
        | Rh2m0to24hRange
        | Dewpoint2m0to24hMax
        | Dewpoint2m0to24hMin
        | Dewpoint2m0to24hRange
        | Vpd2m0to24hMax
        | Vpd2m0to24hMin
        | Vpd2m0to24hRange => Some((1, 24, "0-24 h 2 m surface snapshot window")),
        Temp2m24to48hMax
        | Temp2m24to48hMin
        | Temp2m24to48hRange
        | Rh2m24to48hMax
        | Rh2m24to48hMin
        | Rh2m24to48hRange
        | Dewpoint2m24to48hMax
        | Dewpoint2m24to48hMin
        | Dewpoint2m24to48hRange
        | Vpd2m24to48hMax
        | Vpd2m24to48hMin
        | Vpd2m24to48hRange => Some((25, 48, "24-48 h 2 m surface snapshot window")),
        Temp2m0to48hMax
        | Temp2m0to48hMin
        | Temp2m0to48hRange
        | Rh2m0to48hMax
        | Rh2m0to48hMin
        | Rh2m0to48hRange
        | Dewpoint2m0to48hMax
        | Dewpoint2m0to48hMin
        | Dewpoint2m0to48hRange
        | Vpd2m0to48hMax
        | Vpd2m0to48hMin
        | Vpd2m0to48hRange => Some((1, 48, "0-48 h 2 m surface snapshot window")),
        _ => None,
    }
}

fn blocker(product: HrrrWindowedProduct, reason: impl Into<String>) -> HrrrWindowedBlocker {
    HrrrWindowedBlocker {
        product,
        reason: reason.into(),
    }
}

/// Planner-loaded APCP hour decode. The bytes were already fetched by
/// the runtime's `load_execution_plan`; this just wraps the decode +
/// hour-info bookkeeping.
///
/// Partial-success: an hour whose fetch failed upstream is recorded as
/// `Err(reason)` in the returned map rather than short-circuiting. The
/// windowed compute kernels (`compute_qpf_product` / `compute_uh_product`)
/// propagate per-hour `Err` into a per-product blocker, so a single 404
/// on one contributing hour collapses just the products whose window
/// included that hour - the rest still render.
fn load_apcp_hours_from_plan(
    loaded: Option<&LoadedBundleSet>,
    request: &HrrrWindowedBatchRequest,
    hours: &BTreeSet<u16>,
) -> Result<
    (
        BTreeMap<u16, Result<HrrrApcpDecode, String>>,
        Vec<HrrrWindowedHourFetchInfo>,
        u128,
        u128,
    ),
    Box<dyn std::error::Error>,
> {
    let mut out = BTreeMap::new();
    let mut fetches = Vec::new();
    let mut total_fetch_ms = 0u128;
    let mut total_decode_ms = 0u128;
    let fetch_product = windowed_fetch_product(request.model);

    for &hour in hours {
        let fetched = match loaded
            .and_then(|set| lookup_planner_bundle_for_hour(set, hour, fetch_product.as_str()))
        {
            Some(bytes) => bytes,
            None => {
                let reason = planner_hour_failure_reason(loaded, hour, fetch_product.as_str());
                out.insert(hour, Err(reason));
                continue;
            }
        };
        total_fetch_ms += fetched.fetch_ms;
        let decode_path =
            decode_cache_path(&request.cache_root, &fetched.file.request, "windowed_apcp");
        let decode_start = Instant::now();
        let decode_result = load_or_decode_apcp(
            &decode_path,
            &fetched.file.bytes,
            request.use_cache,
            Some(hour),
        )
        .map_err(|err| err.to_string());
        total_decode_ms += decode_start.elapsed().as_millis();
        fetches.push(HrrrWindowedHourFetchInfo {
            hour,
            planned_product: fetch_product.clone(),
            fetched_product: fetched.file.request.request.product.clone(),
            requested_source: fetched
                .file
                .request
                .source_override
                .unwrap_or(fetched.file.fetched.result.source),
            resolved_source: fetched.file.fetched.result.source,
            resolved_url: fetched.file.fetched.result.url.clone(),
            fetch_cache_hit: fetched.file.fetched.cache_hit,
            input_fetch: Some(fetch_identity_from_cached_result(
                fetch_product.as_str(),
                &fetched.file.request,
                &fetched.file.fetched,
            )),
        });
        out.insert(hour, decode_result);
    }
    Ok((out, fetches, total_fetch_ms, total_decode_ms))
}

/// Planner-loaded native UH hour decode. UH messages live in the same
/// wrfsfc file the QPF lane already pulled, so the planner's dedupe
/// means we only fetch each hour once even when both QPF and UH ask for
/// it.
///
/// Same partial-success contract as `load_apcp_hours_from_plan`: a
/// missing hour is an `Err` entry, not an aborted lane.
fn load_uh_hours_from_plan(
    loaded: Option<&LoadedBundleSet>,
    request: &HrrrWindowedBatchRequest,
    hours: &BTreeSet<u16>,
) -> Result<
    (
        BTreeMap<u16, Result<HrrrUhDecode, String>>,
        Vec<HrrrWindowedHourFetchInfo>,
        u128,
        u128,
    ),
    Box<dyn std::error::Error>,
> {
    let mut out = BTreeMap::new();
    let mut fetches = Vec::new();
    let mut total_fetch_ms = 0u128;
    let mut total_decode_ms = 0u128;
    let fetch_product = windowed_fetch_product(request.model);

    for &hour in hours {
        let fetched = match loaded
            .and_then(|set| lookup_planner_bundle_for_hour(set, hour, fetch_product.as_str()))
        {
            Some(bytes) => bytes,
            None => {
                let reason = planner_hour_failure_reason(loaded, hour, fetch_product.as_str());
                out.insert(hour, Err(reason));
                continue;
            }
        };
        total_fetch_ms += fetched.fetch_ms;
        let decode_path =
            decode_cache_path(&request.cache_root, &fetched.file.request, "windowed_uh25");
        let decode_start = Instant::now();
        let decode_result =
            load_or_decode_uh25(&decode_path, &fetched.file.bytes, request.use_cache)
                .map_err(|err| err.to_string());
        total_decode_ms += decode_start.elapsed().as_millis();
        fetches.push(HrrrWindowedHourFetchInfo {
            hour,
            planned_product: fetch_product.clone(),
            fetched_product: fetched.file.request.request.product.clone(),
            requested_source: fetched
                .file
                .request
                .source_override
                .unwrap_or(fetched.file.fetched.result.source),
            resolved_source: fetched.file.fetched.result.source,
            resolved_url: fetched.file.fetched.result.url.clone(),
            fetch_cache_hit: fetched.file.fetched.cache_hit,
            input_fetch: Some(fetch_identity_from_cached_result(
                fetch_product.as_str(),
                &fetched.file.request,
                &fetched.file.fetched,
            )),
        });
        out.insert(hour, decode_result);
    }
    Ok((out, fetches, total_fetch_ms, total_decode_ms))
}

/// Planner-loaded native 10 m wind-max hour decode. HRRR carries this
/// as `WIND:10 m above ground:<hourly range> max fcst` in wrfsfc.
fn load_wind10m_hours_from_plan(
    loaded: Option<&LoadedBundleSet>,
    request: &HrrrWindowedBatchRequest,
    hours: &BTreeSet<u16>,
) -> Result<
    (
        BTreeMap<u16, Result<HrrrWind10mMaxDecode, String>>,
        Vec<HrrrWindowedHourFetchInfo>,
        u128,
        u128,
    ),
    Box<dyn std::error::Error>,
> {
    let mut out = BTreeMap::new();
    let mut fetches = Vec::new();
    let mut total_fetch_ms = 0u128;
    let mut total_decode_ms = 0u128;
    let fetch_product = windowed_fetch_product(request.model);

    for &hour in hours {
        let fetched = match loaded
            .and_then(|set| lookup_planner_bundle_for_hour(set, hour, fetch_product.as_str()))
        {
            Some(bytes) => bytes,
            None => {
                let reason = planner_hour_failure_reason(loaded, hour, fetch_product.as_str());
                out.insert(hour, Err(reason));
                continue;
            }
        };
        total_fetch_ms += fetched.fetch_ms;
        let decode_path = decode_cache_path(
            &request.cache_root,
            &fetched.file.request,
            "windowed_wind10m_max",
        );
        let decode_start = Instant::now();
        let decode_result =
            load_or_decode_wind10m_max(&decode_path, &fetched.file.bytes, request.use_cache)
                .map_err(|err| err.to_string());
        total_decode_ms += decode_start.elapsed().as_millis();
        fetches.push(HrrrWindowedHourFetchInfo {
            hour,
            planned_product: fetch_product.clone(),
            fetched_product: fetched.file.request.request.product.clone(),
            requested_source: fetched
                .file
                .request
                .source_override
                .unwrap_or(fetched.file.fetched.result.source),
            resolved_source: fetched.file.fetched.result.source,
            resolved_url: fetched.file.fetched.result.url.clone(),
            fetch_cache_hit: fetched.file.fetched.cache_hit,
            input_fetch: Some(fetch_identity_from_cached_result(
                fetch_product.as_str(),
                &fetched.file.request,
                &fetched.file.fetched,
            )),
        });
        out.insert(hour, decode_result);
    }
    Ok((out, fetches, total_fetch_ms, total_decode_ms))
}

/// Planner-loaded native 2 m surface snapshot decode. HRRR does not
/// carry reliable fixed-window extrema for these fields in wrfsfc, so
/// diurnal products reduce hourly snapshots pulled by idx.
fn load_surface_snapshot_hours_from_plan(
    loaded: Option<&LoadedBundleSet>,
    request: &HrrrWindowedBatchRequest,
    hours: &BTreeSet<u16>,
) -> Result<
    (
        BTreeMap<u16, Result<HrrrSurfaceSnapshotDecode, String>>,
        Vec<HrrrWindowedHourFetchInfo>,
        u128,
        u128,
    ),
    Box<dyn std::error::Error>,
> {
    let mut out = BTreeMap::new();
    let mut fetches = Vec::new();
    let mut total_fetch_ms = 0u128;
    let mut total_decode_ms = 0u128;
    let fetch_product = windowed_fetch_product(request.model);

    for &hour in hours {
        let fetched = match loaded
            .and_then(|set| lookup_planner_bundle_for_hour(set, hour, fetch_product.as_str()))
        {
            Some(bytes) => bytes,
            None => {
                let reason = planner_hour_failure_reason(loaded, hour, fetch_product.as_str());
                out.insert(hour, Err(reason));
                continue;
            }
        };
        total_fetch_ms += fetched.fetch_ms;
        let decode_path = decode_cache_path(
            &request.cache_root,
            &fetched.file.request,
            "windowed_surface_snapshot",
        );
        let decode_start = Instant::now();
        let decode_result =
            load_or_decode_surface_snapshot(&decode_path, &fetched.file.bytes, request.use_cache)
                .map_err(|err| err.to_string());
        total_decode_ms += decode_start.elapsed().as_millis();
        fetches.push(HrrrWindowedHourFetchInfo {
            hour,
            planned_product: fetch_product.clone(),
            fetched_product: fetched.file.request.request.product.clone(),
            requested_source: fetched
                .file
                .request
                .source_override
                .unwrap_or(fetched.file.fetched.result.source),
            resolved_source: fetched.file.fetched.result.source,
            resolved_url: fetched.file.fetched.result.url.clone(),
            fetch_cache_hit: fetched.file.fetched.cache_hit,
            input_fetch: Some(fetch_identity_from_cached_result(
                fetch_product.as_str(),
                &fetched.file.request,
                &fetched.file.fetched,
            )),
        });
        out.insert(hour, decode_result);
    }
    Ok((out, fetches, total_fetch_ms, total_decode_ms))
}

fn lookup_planner_bundle_for_hour<'a>(
    loaded: &'a LoadedBundleSet,
    hour: u16,
    native_product: &str,
) -> Option<&'a FetchedBundleBytes> {
    loaded.fetched.values().find(|bundle| {
        bundle.key.forecast_hour == hour && bundle.key.native_product == native_product
    })
}

/// Resolve the best available failure reason for a missing windowed
/// hour: the upstream planner fetch error if one was captured, else a
/// generic "planner produced no bundles" fallback.
fn planner_hour_failure_reason(
    loaded: Option<&LoadedBundleSet>,
    hour: u16,
    native_product: &str,
) -> String {
    let Some(loaded) = loaded else {
        return format!("planner produced no bundles for hour {hour}");
    };
    loaded
        .fetch_failures
        .iter()
        .find(|(key, _)| key.forecast_hour == hour && key.native_product == native_product)
        .map(|(_, reason)| format!("hour {hour} fetch failed: {reason}"))
        .unwrap_or_else(|| format!("planner missed windowed hour {hour}"))
}

fn windowed_parallelism(_source: SourceId, job_count: usize) -> usize {
    let override_threads = std::env::var("RUSTWX_RENDER_THREADS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0);

    thread::available_parallelism()
        .map(|parallelism| override_threads.unwrap_or((parallelism.get() / 2).max(1)))
        .unwrap_or(1)
        .min(job_count.max(1))
}

fn thread_windowed_error(err: impl std::fmt::Display) -> io::Error {
    io::Error::other(err.to_string())
}

fn join_windowed_job<T>(
    handle: thread::ScopedJoinHandle<'_, Result<T, io::Error>>,
) -> Result<T, io::Error> {
    match handle.join() {
        Ok(result) => result,
        Err(panic) => Err(io::Error::other(format!(
            "windowed worker panicked: {}",
            panic_message(panic)
        ))),
    }
}

fn panic_message(panic: Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = panic.downcast_ref::<&'static str>() {
        (*message).to_string()
    } else if let Some(message) = panic.downcast_ref::<String>() {
        message.clone()
    } else {
        "unknown panic payload".to_string()
    }
}

#[cfg(test)]
mod tests;
