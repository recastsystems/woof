//! Per-hour, self-contained weather field store. Each file holds one model
//! run hour: 2D surface fields as zstd-compressed f32 tiles supporting
//! windowed (regional) reads, and 3D pressure-level fields as quantized
//! affine-i16 column chunks laid out for fast single-point sounding pulls.
//!
//! Version 2 ([`columns`]) is the tiled column store: every column of a grid at one valid time in overlapping
//! 16x16 tiles, so one sounding anywhere is one range read. [`columns::open_any`] opens either version.

pub mod atomic;
pub mod codec;
pub mod columns;
pub mod series;
pub mod diff;
pub mod error;
pub mod format;
pub mod grid;
pub mod header;
pub mod index;
pub mod ingest;
pub mod lock;
pub mod netcdf_classic;
pub mod reader;
pub mod run;
pub mod validate;
pub mod writer;

pub use diff::{Difference, build_matches, compare, read_writer_build};
pub use error::{RwResult, RwStoreError};
pub use format::RwsExactTime;
pub use ingest::{
    DerivedFieldInput, HourIngestWriter, PressureVolumeInput, StoredField2D, WrittenHour,
    derived_selector, derived_selector_slug, read_field_2d, read_grid_2d, write_hour_from_fields,
    write_hour_from_fields_exact, write_hour_from_fields_with_derived,
    write_hour_from_fields_with_derived_exact, write_hour_from_grid_with_derived,
    write_hour_from_grid_with_derived_exact,
};
pub use lock::{LOCK_FILE_NAME, RunLock};
pub use validate::{
    ValidateDepth, ValidationReport, ValidationStats, validate_hour_file, validate_run_dir,
};
