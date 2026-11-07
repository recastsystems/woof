//! WPS-geogrid-equivalent static fields, in Rust.
//!
//! Port of `gpuwm/static/*.py` (the Python implementation is the
//! byte-parity reference on the WPS path).  Three lanes own three
//! disjoint module families off this skeleton:
//!
//! * **lane 1, grid math**: [`projection`] (float64 transforms, the
//!   float32 WPS sampling twins, staggered coordinate arrays, nest and
//!   translation arithmetic), [`corridor`] (parent-extent corridor
//!   geometry and crop), [`npz`] (the deterministic STORED-zip NPZ the
//!   corridor seals).
//! * **lane 2, tile ingest + field building**: [`geog`] (WPS_GEOG
//!   index/tile readers, windows, coverage), [`interp`], [`smooth`],
//!   [`sampler`] (the `_DomainSampler` port: pixel binning, gcell
//!   accumulation, categorical fractions), [`fields`] (`build_static`).
//! * **lane 3, high-resolution overlays**: [`raster`] (GeoTIFF decode
//!   /encode, mosaic, reprojection/area-average warp), [`highres`]
//!   (bound-raster verification, override build, USDA soil triangle,
//!   water-mask merge with donor fill).
//!
//! [`types`], [`error`] and the [`capi`] handle plumbing are the shared
//! floor committed by the design lane; lanes extend, never reshape,
//! those signatures.  The C ABI (`gpuwm_static_*`) is loaded by
//! `gpuwm/static/rust_bridge.py` following the same ctypes discipline
//! as `netcdf-writer`.

pub mod capi;
pub mod corridor;
pub mod error;
pub mod extra;
pub mod fields;
pub mod geog;
pub mod highres;
pub mod interp;
pub mod npz;
pub mod projection;
pub mod raster;
pub mod sampler;
pub mod sfire;
pub mod landfire;
pub mod sfire_debug;
pub mod sfire_ideal;
pub mod smooth;
pub mod snow_optics;
#[cfg(test)]
pub(crate) mod testsupport;
pub mod types;

pub use error::StaticError;
pub use types::{Field, FieldSet, Grid2, Stack3};

/// Earth radius (m) -- module_llxy.F:152; WPS uses the same value.
pub const EARTH_RADIUS_M: f64 = 6_370_000.0;
/// Earth angular velocity (rad/s) -- WPS constants_module OMEGA_E.  This
/// is the value geogrid bakes into geo_em F/E; WRF's EOMEG does NOT
/// reproduce geogrid's fields.
pub const OMEGA_E: f64 = 7.292e-5;
/// geogrid's processing halo (cells beyond the domain edge).
pub const HALO: usize = 3;
/// Minimum source-to-grid resolution ratio for grid-cell averaging
/// (GEOGRID.TBL `average_gcell(4.0)`).
pub const GCELL_RATIO: f64 = 4.0;
/// Metres per degree of a great circle on the WPS sphere.
pub const M_PER_DEG: f64 = EARTH_RADIUS_M * std::f64::consts::PI / 180.0;

pub mod fire_ignition;

pub mod snow_shapes;
pub mod snow_aging;
