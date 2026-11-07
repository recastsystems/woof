//! `gpuwm_mapped_engine`: the decode engine behind gpuwm's mapped prep.
//!
//! The seam is normative in `docs/dev/decode-vendor-design.md` §3, and this
//! crate and `gpuwm/mapped_engine_bridge.py` both encode it.  What moved
//! here is DATA PATH: selector→record matching, declared-grid cross-checks,
//! wind rotation, record assembly, transposition, unit transforms, the
//! closed derivation catalog, canonical-frame invariants, missing-count
//! accounting, and the NetCDF variable resolution above the reader.  What
//! did NOT move is orchestration: authority snapshotting, manifest
//! verification, the decoder ladder, receipts, member grammar, CLI parsing
//! and exception-type selection all stay in Python, which is why this
//! binary never invokes Python and never shells out to another tool.
//!
//! Outputs: `frames.json` (schema [`FRAMESET_SCHEMA`], the compiled ABI
//! marker) beside `frames.f64` (one little-endian float64 stream, fields
//! packed row-major in manifest order).  Refusals leave as one JSON object
//! on the last stderr line, schema [`REFUSAL_SCHEMA`].

pub mod array;
pub mod assemble;
pub mod codec;
pub mod compose;
pub mod derive;
pub mod digest;
pub mod engine;
pub mod frames;
pub mod grib;
pub mod grib1;
pub mod join;
pub mod lambert;
pub mod model;
pub mod ncdf;
pub mod node;
pub mod portable;
pub mod refusal;
pub mod regrid;
pub mod rotated;
pub mod space;
pub mod threads;
pub mod window;
pub mod vertical;
pub mod wps;
mod wps_support;

/// The output schema name, which changes exactly when the frameset
/// contract changes.  It rides inside :data:`ABI_CONTRACT` below.
pub const FRAMESET_SCHEMA: &str = "gpuwm-mapped-frameset-v1";

/// The ABI marker: ONE literal carrying BOTH contracts a stale staged
/// binary can break, because `gpuwm.bridges.BRIDGE_ABI_MARKERS` holds one
/// byte string per artifact and searches the binary for it.
///
/// The frameset schema is the OUTPUT contract: it changes when the shape
/// the Python side reads back changes.  The template list is the DECODE
/// contract: it changes when the set of Section-5 data representations
/// this engine can read changes.  They are not the same contract, and a
/// marker that moved only with the first let a pre-fix binary pass the
/// handshake and then refuse conformant IEEE-packed (template 5.4) bytes
/// with a message blaming the publisher of the file.
///
/// `window-fields=omega` is the atmospheric-window inventory contract
/// (window.rs CANONICAL_ATMOSPHERIC_FIELDS against
/// gpuwm.ingest.atmospheric_window): an engine built before vertical
/// motion joined it (1cbcc73e1c) passed the handshake, then refused every
/// HRRR native forecast lead (a time-lagged DA member, cycle 08Z f02) with
/// "window field pressure_vertical_velocity is outside the atmospheric
/// regular-join inventory".
///
/// `bound-clocks=window-start` is the composition clock
/// `source_cycle_window_start_broadcast` (join.rs): a composition that
/// declares it would be refused by an engine built before it as an
/// unknown clock.
pub const ABI_CONTRACT: &str = "gpuwm-mapped-engine-abi frameset=gpuwm-mapped-frameset-v1 height-interfaces=1 window-fields=omega bound-clocks=window-start grib2-drt=0,2,3,4,40,41,42,50,51,61,200";
pub const REFUSAL_SCHEMA: &str = "gpuwm-mapped-refusal-v1";
pub const PROGRESS_SCHEMA: &str = "gpuwm-mapped-engine-progress-v1";
pub const INSPECTION_SCHEMA: &str = "gpuwm-mapped-source-inspection-v1";
pub const CAPABILITIES_SCHEMA: &str = "gpuwm-mapped-engine-capabilities-v1";
/// The raw per-record product-identity surface (`inventory`): every
/// GRIB2 identity octet the subprocess `grib2_inventory` renders, in the
/// same string spellings, so a product-identity gate reads ONE spelling
/// whichever instrument measured it.
pub const RECORD_INVENTORY_SCHEMA: &str = "gpuwm-mapped-record-inventory-v1";

pub const ENGINE_NAME: &str = "gpuwm_mapped_engine";
pub const ENGINE_VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    #[test]
    fn the_marker_carries_both_contracts_it_stands_for() {
        // Stated as a test because the marker is one literal and the two
        // contracts inside it are defined elsewhere: if either drifts out
        // of the marker, the reason the handshake works -- a change to the
        // frameset shape OR to the readable template set forces a marker
        // change -- quietly stops holding.
        assert_eq!(super::FRAMESET_SCHEMA, "gpuwm-mapped-frameset-v1");
        assert!(super::ABI_CONTRACT.contains(super::FRAMESET_SCHEMA));
        assert!(super::ABI_CONTRACT
            .ends_with(grib_core::grib2::DECODE_TEMPLATES));
    }
}

pub mod lead_index;
