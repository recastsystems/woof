//! The wrfout import lane and the panel/overlay seam, as a library.
//!
//! `main.rs` (the `rw_wrfbatch` binary) keeps its own `#[path]` module
//! declarations and is not built from this target -- adding a library here
//! is purely additive, and the renderer binary's bytes are unaffected by
//! anything in this file.  What the library exists for is the two SIBLING
//! binaries in this crate:
//!
//! * `rw_ensbatch` -- ensemble products (mean, spread, NMEP, PMM,
//!   paintball) from N member wrfouts at one valid time;
//! * `rw_obsgrid` -- the `gpuwm-obs.radar-grid` observation grid (v1 and
//!   the windowed v2), read natively rather than disguised as a forecast.
//!
//! Both need the same three things `rw_wrfbatch` already has: the hardened
//! wrfout import ([`wrf_process`]), a way to put an arbitrary 2-D plane
//! through the production render path ([`panel`]), and a way to put a
//! place known in degrees onto that panel ([`annotate`]).

pub mod cyclone_products;
pub mod solar_products;
pub mod annotate;
pub mod compare;
pub mod latlon_pair;
pub mod coverage;
pub mod nowcast_frames;
pub mod da_sheet;
pub mod verification;
pub mod scorecard;
pub mod verification_io;
pub mod verification_render;
pub mod station_overlay;
pub mod panel;
pub mod nextgen_products;
pub mod ocean_products;
pub mod ensemble_products;
pub mod ensemble_reduce;
pub mod sfire;
pub mod fire_observations;
pub mod snowfall;
mod visibility;
mod source_memo;
pub mod domain_naming;

#[path = "grib_import.rs"]
pub mod grib_import;
#[path = "local_import.rs"]
pub mod local_import;
#[path = "nest_move.rs"]
mod nest_move;
#[path = "postproc_severe.rs"]
pub mod postproc_severe;
#[path = "wrf_process.rs"]
pub mod wrf_process;
#[path = "wrf_volumes.rs"]
pub mod wrf_volumes;

pub mod obs_grid;
pub mod scales;
pub mod section;
pub mod process_request;
pub mod viewer_profile;
mod wrf_chart_planes;
mod wrf_column_planes;
mod ivt;

pub mod cf_grid;
pub mod gridded_verification;
pub mod cf_render;
pub mod comparison_observations;

pub mod hubwind_products;
pub mod lightning_verification;
mod wrf_vertical_velocity;
mod mslp_smoothing;
