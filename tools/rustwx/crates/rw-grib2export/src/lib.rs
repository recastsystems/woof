//
// Copyright 2026 the WOOF authors.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License.  You may obtain a copy
// of the License at https://www.apache.org/licenses/LICENSE-2.0.  Unless
// required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied.

//! WOOF native GRIB2 exporter (engine 2.8.7, clean room).
//!
//! `gpuwm export-grib2`, `gpuwm go --grib2`, `run_options.grib2` and
//! `gpuwm render --grib2-out` write a request ([`request::Request`], schema
//! `grib2-export.request/v1`) and run `rw_grib2export --request FILE`.
//! Every array operation happens here:
//!
//! * history discovery (folders, gzip, ZIP) is `rw-mlexport`'s;
//! * every product is computed by `rw-post` from ONE shared column state,
//!   on a GPU by default (`post_device = auto`) with the CPU reference path
//!   when no card has room;
//! * messages are packed by the vendored wx-core GRIB2 writer (WMO FM 92
//!   Edition 2: sections 0 to 8, grid templates 3.0, 3.10, 3.20 and 3.30,
//!   product templates 4.0 and 4.8, simple packing 5.0 and complex packing
//!   with spatial differencing 5.3, bitmap section 6 for missing cells).
//!
//! The catalog is `woof-post/v1`, profile `woof-post-2.8.7` (spec D29).

pub mod catalog;
pub mod vertical;
pub mod export;
pub mod extrema;
pub mod grid;
pub mod group_d;
pub mod renderer;
pub mod request;
/// The RRFS v2 UPP field set (rrfs_mpas_postcntrl_L60) writer and its GPU extras.
#[cfg(any(windows, target_os = "linux"))]
pub mod rrfs_gpu;
#[cfg(any(windows, target_os = "linux"))]
pub mod rrfs_upp;

/// The contract line `rw_grib2export --abi` prints and the literal
/// `gpuwm.bridges.BRIDGE_ABI_MARKERS` looks for in the built binary.
pub const ABI: &str = "rw_grib2export --request REQUEST.json schema=grib2-export.request/v1 \
modes=run,append,finalize progress=jsonl default_definitions=woof \
grid_geometry=wrf-native-locations/v1 surface_catalog=woof-post/v1 post_device=auto,gpu,cpu \
definitions=woof,renderer extrema_window=seconds composite_level=200,10 \
gust=auto,similarity,tke";

/// Catalog identity (spec D29).
pub const CATALOG: &str = "woof-post/v1";
/// Profile identity (spec D29).
pub const PROFILE: &str = "woof-post-2.8.7";

pub use rw_mlexport::error::{fail, refuse, ExportError, Result};
