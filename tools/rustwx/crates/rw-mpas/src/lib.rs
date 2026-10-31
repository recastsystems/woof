//! MPAS mesh handling, static generation, and the history-to-wrfout render
//! bridge, in Rust.
//!
//! This is the model-output-to-pixels half of the one-stop preprocessing
//! engine: it takes an MPAS history frame on an unstructured mesh, gathers
//! every field with a WRF product equivalent onto a structured render window,
//! and writes a wrfout-shaped classic netCDF frame the pinned renderer opens
//! directly. Reading is netcrust; writing is `rw_store::netcdf_classic`. No
//! HDF5 write binding, no Python, no SciPy.
//!
//! The static-builder modules are deliberately in `rw-mpas` too: they consume
//! an MPAS mesh, not a structured WRF domain. WPS_GEOG decoding is streamed
//! tile-plane-by-tile-plane so host memory is bounded by one source plane plus
//! destination-sized accumulators, never by the size of the geography tree.
//!
//! ONE STATIC WRITER. `static_builder` is it, with `static_geog` reading the
//! archive, `static_gwd` cutting the sub-grid orography, `static_operators`
//! building the coefficient tables and `static_memory` deciding what fits.
//! Both `rw_mpas_static` and its `bounded` tool (formerly the
//! `rw_mpas_static_bounded` binary) run it and produce the identical file;
//! they differ only in how the geography directories are named.
//!
//! There were two, and the difference was invisible from outside. One wrote
//! the 69 variables the mesh registry pins and owned the `rw_mpas_static`
//! name; the other wrote 56 under host-memory admission with a parallel tile
//! decode. One edited `use` line was enough to hand a user the wrong file
//! under the right name -- a static with no `var2d`, no `oa1..oa4`, no
//! `cellsOnVertex` and no `kiteAreasOnVertex` -- and the registry pins grid
//! and static together by byte count and SHA-256, so it would have been
//! refused at registration rather than at the door.
//!
//! `staticfile::schema` is the gate that held that open, red on purpose,
//! naming the 26 variables the survivor had to gain. It is green because they
//! converged: the pin is now the UNION of both manifests, 82 names, and
//! `staticfile` is the remnant of the retired writer -- the pin itself, the
//! `--compare` grader, the archive ladder and the FP32 metric reading.

use std::io::Read;
use std::path::Path;

use sha2::{Digest, Sha256};

pub mod composite;
pub mod convert;
pub mod error;
pub mod fieldmap;
pub mod geometry;
pub mod history;
pub mod hostprep;
pub mod hosttables;
pub mod init;
pub mod rrfs_init;
pub mod lbc;
pub mod rrfs_lbc;
pub mod mesh;
pub mod mpassit;
pub mod mslp;
pub mod points;
pub mod remap;
pub mod soundings;
pub mod staticfile;
pub mod static_builder;
pub mod rrfs_static;
pub mod rrfs_static_gwd;
pub mod rrfs_static_operators;
pub mod static_geog;
pub mod static_gwd;
pub mod gsl_oro;
pub mod static_memory;
pub mod static_native;
pub mod static_namelist;
pub mod static_operators;
pub mod weights;
pub mod window;

pub use error::{MpasError, MpasResult};

/// SHA-256 of a file, streamed. Every digest this crate publishes -- and
/// every digest it refuses against -- comes from here.
pub fn sha256_file(path: &Path) -> MpasResult<String> {
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 16 * 1024 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(weights::hex(&hasher.finalize()))
}

pub mod native_gwd;
