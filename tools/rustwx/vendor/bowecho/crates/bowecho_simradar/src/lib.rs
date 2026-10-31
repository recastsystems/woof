//! Headless virtual radar using the BowEcho reference forward operator.
extern crate self as app_ui;
extern crate self as bowecho_simradar;

pub use radar_core::geo;
pub mod model_layer {
    include!("model_grid.rs");
}
pub mod vcp_catalog;
pub mod wrf_p3_assets;
pub mod wrf_property_reader;
pub mod wrf_radar_estimator;
pub mod wrf_radar_physics;
pub mod wrf_band_scattering;
pub mod wrf_radar_validation;
pub mod wrf_refractivity;
pub mod wrf_scene_adapter;
pub mod wrf_scene_inventory;
pub mod wrf_temporal;
pub mod wrf_tmatrix_assets;
pub mod wrf_tmatrix_band_assets;
pub mod wrf_tmatrix_cuda;
pub mod wrf_tmatrix_legacy_pack;
pub mod wrf_tmatrix_scene;

pub mod wrf_radar {
    #![allow(dead_code)]
    include!("wrf_radar_core.rs");
    include!("model_input.rs");
    #[cfg(test)]
    mod tests {
        use super::*;
        include!("core_tests.rs");
    }
}

pub use data_source::fallback_sites as embedded_radar_sites;
pub use data_source::sites::{SiteKind, SiteRecord, SiteRef, all_sites};
pub use radar_core;
pub use wrf_core::WrfFile;
pub use wrf_radar::*;
