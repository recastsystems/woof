#![recursion_limit = "512"]
//! Public sounding viewer, ported from the live sounding window.
//! rw-store columns -> rustwx-sounding column contract -> sharppyrs (the
//! SHARPpy SPC window in Rust) rendered headless, plus JSON for the site.
pub mod analysis;
pub mod api;
pub mod options;
pub mod render;
pub mod store;
pub mod tables;

pub mod batch;
pub mod columns;
pub mod input;
pub mod surface;
