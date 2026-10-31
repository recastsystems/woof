//! Native Rust NetCDF emission: classic CDF-1/2/5 and fixed NetCDF-4 snapshots.
//!
//! # Why this crate exists
//!
//! Everything NetCDF-shaped in the Rust stack READS. `netcrust`,
//! `hdf5-reader` and `netcdf-reader` are read-only by design, and
//! `rw-netcdf` is `inventory` plus `dump`. The single write capability
//! was `rw-store/src/netcdf3.rs`, a CDF-2 emitter with **no record
//! dimension** and **NC_FLOAT as its only variable type** -- which is
//! precisely why it cannot write a wrfout, whose `Times` variable is
//! `NC_CHAR` on an unlimited dimension. This crate is seeded from that
//! module's byte grammar and extends it to the whole classic format:
//!
//! | | `rw-store/netcdf3` | this crate |
//! |---|---|---|
//! | containers | CDF-2 | CDF-1, CDF-2, CDF-5 |
//! | variable types | `NC_FLOAT` | all eleven classic types |
//! | attribute types | text, `NC_FLOAT` | text and all ten numeric types |
//! | record dimension | none (`numrecs` always 0) | full, with the netCDF-C stride rules |
//! | writes | whole variable, in order | any variable or record slab, any order |
//! | `numrecs` | fixed at 0 | patched at `finish` |
//!
//! # Shape
//!
//! ```no_run
//! use netcdf_writer::{AttrValue, NcFormat, NcType, NcWriter, Schema, VarData};
//!
//! let mut schema = Schema::new(NcFormat::Offset64);
//! let time = schema.def_dim("Time", 0, true)?;          // the record dim
//! let strlen = schema.def_dim("DateStrLen", 19, false)?;
//! let south_north = schema.def_dim("south_north", 300, false)?;
//! let west_east = schema.def_dim("west_east", 400, false)?;
//!
//! schema.put_global_attr("TITLE", AttrValue::Text(" OUTPUT FROM GPUWM".into()))?;
//! let times = schema.def_var("Times", NcType::Char, &[time, strlen])?;
//! let t2 = schema.def_var("T2", NcType::Float, &[time, south_north, west_east])?;
//! schema.put_var_attr(t2, "units", AttrValue::Text("K".into()))?;
//!
//! let mut writer = NcWriter::create("wrfout_d01", schema)?;
//! writer.write_record(0, times, VarData::Char(b"2026-08-16_00:00:00"))?;
//! writer.write_record(0, t2, VarData::F32(&vec![300.0; 300 * 400]))?;
//! writer.finish()?;
//! # Ok::<(), netcdf_writer::NcWriteError>(())
//! ```
//!
//! # What is deliberately NOT here
//!
//! Fixed NetCDF-4 snapshots, including NC_STRING, use [`netcdf4`]. This
//! independent container path does not alter classic serialization or the C ABI.
//!
//! * **Fill values on unwritten data.** netCDF-C pre-fills; this writer
//!   refuses an unwritten region at `finish` instead. A tape with a hole
//!   in it is a defect, not a default.
//! * **Reading.** That is `netcrust`'s job. The one exception is
//!   [`scan_nonfinite`], which reads a file this crate wrote BACK and
//!   names the float variables holding a non-finite value. It is a
//!   read-back verification of one question, not a reader: it decodes no
//!   value for a caller and interprets no attribute. `gpuwm.wrf_direct`
//!   has always ended its `wrfinput`/`wrfbdy` export with exactly that
//!   sweep, and routing it through `rw_netcdf` would cost one process
//!   launch and one f64 temp file per variable on the preparation path.
//!
//!   The other exception is [`NcPatcher`], which parses the header of an
//!   EXISTING classic file to find where a variable's bytes live and then
//!   rewrites those bytes in a copy. It exists for files another program
//!   edits in place and expects back unchanged in every byte it did not
//!   ask to change: a schema-driven rewrite reproduces the values but not
//!   the header bytes or the data-section alignment the producing library
//!   chose.
//!
//! # Verified against
//!
//! * `netcdf-reader` 0.3 (an independent pure-Rust classic parser), in
//!   `tests/classic_roundtrip.rs`.
//! * netCDF4-python 1.7.4, i.e. the Unidata C library, in
//!   `tests/test_nc_writer_bridge.py` and `tools/nc_rewrite_parity.py`.
//! * `rw_wrfbatch`, the production renderer, in
//!   `tools/nc_rewrite_render_parity.py`.

pub mod netcdf4;
pub mod netcdf4_stream;
mod hdf_checksum;
mod capi;
mod error;
mod header;
mod layout;
mod patch;
mod scan;
mod schema;
mod types;
mod writer;

pub use capi::NCWRITE_ABI_VERSION;
pub use error::{NcWriteError, Result};
pub use patch::{NcPatcher, PatchVar};
pub use scan::scan_nonfinite;
pub use schema::{name_is_valid, Attr, Dim, Schema, VarDef};
pub use types::{AttrValue, NcFormat, NcType, VarData};
pub use netcdf4_stream::{Nc4Options, Nc4Writer};
pub use writer::NcWriter;
