//! The frontend's stock-WRF admission set agrees with the engine's inventory.
//!
//! `STOCK_WRF_INVENTORIED_MP_PHYSICS` is a hand-written id set; the engine's
//! `gpuwm/wrf_physics_inventory.py` is the producer, and the physics registry
//! now publishes per scheme whether an evidenced package inventory exists
//! (`consumers.stock_wrf_export.inventoried`).  The engine exports that as
//! `gpuwm/physics_consumer_export_v1.json` (`tools/build_registry.py`), and
//! this test reads it in both directions: the frontend never admits an id the
//! engine cannot inventory, and every inventoried id the frontend still
//! refuses is named below with the breakage that keeps it out.  The Python
//! side holds the same contract as a text property of this file
//! (`tests/test_rw_wps_stock_inventory_contract.py`); this is the crate's own
//! half of it, against data rather than a regex.

use std::collections::BTreeSet;
use std::path::Path;

use rw_wps::namelist::STOCK_WRF_INVENTORIED_MP_PHYSICS;

/// Inventoried ids this frontend still refuses, each with its reason.
///
/// The earlier mp=18 and mp=28 rows were retired by audit R-054:
/// mp=18 had no obstacle at all, and mp=28's -- the field-shape check
/// admitting only the 4-D dimension tuple, while its package carries two 2-D
/// wrfinput members -- was fixed rather than worked around, so the check now
/// admits a member by its DECLARED rank.  A row here again means the engine
/// inventories a package this frontend has no evidenced Registry contract
/// for; the test below still requires that every such row name its breakage.
const CITED_ABSENCES: &[(u16, &str)] = &[(
    30,
    "FAST_SBM's 142 spectral bin members and six number moments have no evidenced \
     frontend contract; the engine fills the bins from bulk condensate at the first step",
)];

fn export() -> serde_json::Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../../woof/physics_consumer_export_v1.json");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|error| {
        panic!(
            "{} is the engine's consumer export and this test cannot run without it: {error}",
            path.display()
        )
    });
    let value: serde_json::Value = serde_json::from_str(&text).expect("consumer export parses");
    assert_eq!(
        value["schema"].as_str(),
        Some("gpuwm-physics-consumer-export-v1"),
        "unexpected consumer export schema"
    );
    value
}

fn inventoried(export: &serde_json::Value) -> BTreeSet<u16> {
    export["microphysics"]
        .as_object()
        .expect("microphysics is an object keyed by mp_physics")
        .iter()
        .filter(|(_, row)| row["stock_wrf_export_inventoried"].as_bool() == Some(true))
        .map(|(key, _)| key.parse::<u16>().expect("mp_physics key"))
        .collect()
}

#[test]
fn the_frontend_never_admits_an_uninventoried_scheme() {
    let export = export();
    let producer = inventoried(&export);
    for id in STOCK_WRF_INVENTORIED_MP_PHYSICS {
        assert!(
            producer.contains(id),
            "mp_physics={id} is admitted by the frontend but the engine inventories no package for it; \
             a widened gate would certify a member list nobody checked"
        );
    }
}

#[test]
fn every_refused_inventoried_scheme_is_cited() {
    let export = export();
    let producer = inventoried(&export);
    let consumer: BTreeSet<u16> = STOCK_WRF_INVENTORIED_MP_PHYSICS.iter().copied().collect();
    let mut problems = Vec::new();
    for id in producer.difference(&consumer) {
        if !CITED_ABSENCES.iter().any(|(cited, _)| cited == id) {
            problems.push(format!(
                "the engine inventories mp_physics={id} and this frontend refuses it with no recorded reason"
            ));
        }
    }
    for (id, reason) in CITED_ABSENCES {
        if consumer.contains(id) {
            problems.push(format!(
                "mp_physics={id} is admitted and still cited as refused ({reason}); retire the citation"
            ));
        }
        if !producer.contains(id) {
            problems.push(format!(
                "mp_physics={id} is cited as refused but the engine inventories no package for it; retire the citation"
            ));
        }
    }
    assert!(
        problems.is_empty(),
        "STOCK_WRF_INVENTORIED_MP_PHYSICS disagrees with the engine's inventory:\n  {}",
        problems.join("\n  ")
    );
}

#[test]
fn every_admitted_schemes_member_shapes_are_ones_the_check_admits() {
    // The frontend's field-shape check admits a package member declared on
    // either WRF dimension spec -- the four 3-D dimensions or the three 2-D
    // ones.  It used to admit only the 4-D tuple, which is why mp=28 was
    // refused by id: its two surface aerosol emission members are `ij`, and
    // admitting the id alone would have moved the refusal one loop later and
    // left it unnamed.  Audit R-054 taught the check the second shape, so
    // this test moved with it -- from "every admitted package is uniformly
    // 3-D" to "every admitted package declares only shapes the check
    // admits", which is the property that was ever actually needed.
    let export = export();
    let admitted = ["Time/bottom_top/south_north/west_east", "Time/south_north/west_east"];
    for id in STOCK_WRF_INVENTORIED_MP_PHYSICS {
        let row = &export["microphysics"][id.to_string()];
        for value in row["wrfinput_dimensions"]
            .as_array()
            .expect("wrfinput_dimensions")
        {
            let spec = value.as_str().expect("dimension spec");
            assert!(
                admitted.contains(&spec),
                "mp_physics={id} is admitted but its package declares {spec}, \
                 a member shape the field-shape check refuses"
            );
        }
    }
}
