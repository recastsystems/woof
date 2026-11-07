//! Typed Rust binding for the authoritative RW-WPS namelist support report.
//!
//! The gpuwm preprocessing engine owns the scientific classification.  The
//! standalone Rust frontend owns path binding, process execution, schema
//! validation, and refusal of contradictory PASS/FAIL evidence.

use crate::{RwWpsError, io_error};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::ffi::{OsStr, OsString};
use std::path::{Path, PathBuf};
use std::process::Command;

pub const NAMELIST_SUPPORT_SCHEMA: &str = "rw-wps.namelist-support.v1";

/// Microphysics ids this frontend has an EVIDENCED stock-WRF v4.6.1
/// initialization inventory for, and will therefore forward.
///
/// This is the consuming half of ONE contract whose producing half is
/// `gpuwm/wrf_physics_inventory.py::_INVENTORIES`, published there as
/// `supported_stock_wrf_mp_physics()`.  Every producer row is derived from
/// `Registry/Registry.EM_COMMON` package declarations and field I/O flags,
/// never from the schemes the gpuwm forecast runtime implements.  When the
/// two halves disagree the frontend refuses, because a report it cannot
/// check is not a report it may certify.
///
/// What this set does NOT decide, since two sets with the same values can
/// answer different questions: it is not "schemes gpuwm can forecast".  That
/// verdict travels separately in `required_state.gpuwm_runtime`, which this
/// binding deliberately does not gate on -- a stock-WRF export is valid for
/// a package the gpuwm runtime never runs, and the report says so in two
/// independent verdicts on purpose.
///
/// mp=50, P3 one-category two-moment ice, is admitted.  Its package is
/// `package p3_1category mp_physics==50 - moist:qv,qc,qr,qi;
/// scalar:qni,qnr,qir,qib; state:re_cloud,re_ice,vmi3d,rhopo3d,di3d,
/// refl_10cm,th_old,qv_old` (`Registry.EM_COMMON:3038`).  ONE ice category,
/// so the moist list carries no `qs` and no `qg`: rime mass `qir` and rime
/// volume `qib` (`module_mp_p3.F:744`, bound to `qirim`/`birim` at
/// :1081-1083) span the graupel-to-snow continuum instead of splitting it
/// into species.  WRF's own dispatch agrees -- `module_microphysics_driver.F`
/// :1557-1602 calls `mp_p3_wrapper_wrf` with `N_ICECAT=1` and passes no snow
/// or graupel argument at all, and asks for `diag_effc_3d`/`diag_effi_3d`
/// with no snow radius.  Every member of that package, eight `wrfinput`
/// fields and eight runtime-state fields, is float32 on the four 3-D
/// dimensions, so mp=50 satisfies the field-shape invariant enforced below
/// as well as the id check here.
///
/// mp=51 adds scalar qnc (QNCLOUD) to that inventory, as declared by
/// p3_1category_nc at Registry.EM_COMMON:3039. The same rank and dtype
/// rules apply.
///
/// mp=28 and mp=18 joined on audit R-054.  Both had been RECORDED here as
/// refusals rather than left as omissions, and neither reason survived
/// examination:
///
///   * mp=28, Thompson aerosol-aware, was refused because its package
///     carries two 2-D `wrfinput` members, `qnwfa2d`/`qnifa2d`
///     (`Registry.EM_COMMON:492-493`, dimension spec `ij`, I/O string
///     `i01{17}rhdu` -- an `i` list that begins with stream 0, so they are
///     initialization-file variables), and a 2-D runtime diagnostic
///     `taod5502d` (:1739), while the field-shape check below accepted only
///     `Time,bottom_top,south_north,west_east`.  That was a real obstacle
///     and it is now removed rather than worked around: the check reads
///     each member's DECLARED rank from the engine inventory and admits
///     the 2-D shape for a member declared 2-D, so the frontend certifies
///     what the package actually is instead of refusing every package that
///     is not uniformly 3-D.
///   * mp=18, NSSL 2-moment.  Every member is 3-D float32, so it had no
///     structural obstacle at all -- this frontend had simply never ruled
///     on it.  It rules now.
///
/// A refusal survives here only for an id the paired engine inventories and
/// this frontend has no evidenced Registry contract for, and one does:
///
///   * mp=30, FAST_SBM (registry.sbm:318).  Its package carries 142 spectral
///     bin members and six number moments as scalars, with mixed units
///     (the ff8 aerosol bins per kg, the hydrometeor bins kg/kg), and stock
///     real.exe writes them all as zero.  This frontend has no evidenced
///     contract for that member list, so certifying it would export a bin
///     state nobody checked.  The engine fills the bins from the analysis'
///     bulk condensate at the first step instead
///     (gpuwm/core/fast_sbm_bulk_start.py).
///
/// mp=13 (SBU-YLin), 15 (Ferrier-Aligo advected rime), 40 (Morrison
/// aerosol), 52 (P3 two-category), 53 (P3 three-moment ice), 55 (ISHMAEL),
/// 56 (NTU) and 97 (Goddard GCE) joined when their engine ports landed: the
/// paired engine derives each package from Registry.EM_COMMON like every
/// row above, and every member of all eight is float32 on the 3-D or 2-D
/// stock shape the check below enforces, so none had an obstacle -- the
/// frontend had only never ruled on them, which is an omission, not a
/// refusal.
///
/// mp=27 (UDM), 29 (RCON), 95 (Eta Ferrier) and 96 (MAD-WRF) joined the same
/// way when the 3.0 port folds inventoried them (b2817cdc11a, 9846f70556b, 74abec86b8f, 13a5a66c5bc):
/// every wrfinput member of the four is float32 on the 3-D stock shape, or
/// the 2-D `ij` shape for RCON's qnwfa2d/qnifa2d pair exactly as mp=28's,
/// so the frontend had no obstacle to cite.  mp=29's open speed proof
/// (gpuwm/config.py RCON-SPEED-01) refuses an ArWen FORECAST with it; it is
/// not a reason to refuse the stock-WRF initialization file this frontend
/// certifies, which an unchanged WRF reads.
///
/// A new engine package arrives as a new id here plus whatever its members'
/// shapes require, never as a silently narrowed export.
///
/// Public so `tests/engine_stock_inventory.rs` can hold it to the engine's
/// own inventory (gpuwm/physics_consumer_export_v1.json).
pub const STOCK_WRF_INVENTORIED_MP_PHYSICS: &[u16] = &[2, 4, 5, 6, 7, 8, 10, 11, 13, 14, 15, 18, 24, 26, 27, 28, 29, 38, 40, 50, 51, 52, 53, 55, 56, 95, 96, 97, 106];

#[derive(Debug, Clone, PartialEq)]
pub struct NamelistSupportRequest {
    pub wps: PathBuf,
    pub input: PathBuf,
    pub source_top_pressure_pa: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct NamelistSupportReport {
    pub schema: String,
    pub verdict: String,
    pub max_dom: Option<u16>,
    pub classifications: NamelistClassifications,
    pub geometry: NamelistGeometry,
    pub vertical: Option<NamelistVertical>,
    pub required_state: RequiredStateReport,
    pub issues: Vec<NamelistIssue>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct NamelistClassifications {
    pub preprocessing_relevant: Vec<NamelistSetting>,
    pub physics_state_relevant: Vec<NamelistSetting>,
    pub runtime_output_only: Vec<NamelistSetting>,
    pub legacy_stage_only: Vec<NamelistSetting>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct NamelistSetting {
    pub source: String,
    pub section: String,
    pub key: String,
    pub values: Vec<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct NamelistGeometry {
    pub domain_count: usize,
    pub domains: Vec<NamelistDomain>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct NamelistDomain {
    pub grid_id: u16,
    pub parent_id: u16,
    pub parent_grid_ratio: u16,
    pub parent_time_step_ratio: u16,
    pub i_parent_start: u32,
    pub j_parent_start: u32,
    pub e_we: u32,
    pub e_sn: u32,
    pub mass_nx: u32,
    pub mass_ny: u32,
    pub dx_m: f64,
    pub dy_m: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct NamelistVertical {
    pub e_vert: usize,
    pub mass_levels: usize,
    pub eta_levels: Vec<f64>,
    pub p_top_requested_pa: f64,
    pub hybrid_opt: u16,
    pub etac: f64,
    pub source_top_pressure_pa: Option<f64>,
    pub coverage: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RequiredStateReport {
    pub stock_wrf_export: StockWrfExportReport,
    pub gpuwm_runtime: GpuwmRuntimeReport,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct StockWrfExportReport {
    pub verdict: String,
    pub target: String,
    pub domains: Vec<StockWrfDomainState>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct GpuwmRuntimeReport {
    pub verdict: String,
    pub reasons: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct StockWrfDomainState {
    pub grid_id: u16,
    pub mp_physics: u16,
    pub microphysics: String,
    pub bl_pbl_physics: u16,
    pub sf_sfclay_physics: u16,
    pub sf_surface_physics: u16,
    pub num_soil_layers: u16,
    pub wrfinput_fields: Vec<WrfStateField>,
    pub runtime_state_not_wrfinput: Vec<WrfStateField>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct WrfStateField {
    pub registry_name: String,
    pub netcdf_name: String,
    #[serde(default)]
    pub collection: Option<String>,
    pub dtype: String,
    pub dimensions: Vec<String>,
    #[serde(default)]
    pub units: Option<String>,
    pub initialization: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct NamelistIssue {
    pub code: String,
    pub location: String,
    pub message: String,
    pub action: String,
}

pub fn build_namelist_support_args(request: &NamelistSupportRequest) -> Vec<OsString> {
    let mut args = vec![
        OsString::from("--namelist-support-report"),
        OsString::from("--wps-namelist"),
        request.wps.as_os_str().to_owned(),
        OsString::from("--namelist-input"),
        request.input.as_os_str().to_owned(),
    ];
    if let Some(value) = request.source_top_pressure_pa {
        args.push(OsString::from("--source-top-pressure-pa"));
        args.push(OsString::from(value.to_string()));
    }
    args
}

pub fn query_namelist_support(
    engine: &OsStr,
    request: &NamelistSupportRequest,
) -> Result<NamelistSupportReport, RwWpsError> {
    require_file(&request.wps, "namelist.wps")?;
    require_file(&request.input, "namelist.input")?;
    if let Some(value) = request.source_top_pressure_pa
        && (!value.is_finite() || value <= 0.0)
    {
        return Err(RwWpsError::Config(format!(
            "source top pressure must be finite and positive, got {value:?}"
        )));
    }

    let args = build_namelist_support_args(request);
    let output = Command::new(engine)
        .args(&args)
        .output()
        .map_err(|source| {
            RwWpsError::Engine(format!(
                "cannot run {} for namelist support: {source}",
                Path::new(engine).display()
            ))
        })?;
    let report: NamelistSupportReport =
        serde_json::from_slice(&output.stdout).map_err(|error| {
            RwWpsError::Engine(format!(
                "native engine emitted an invalid namelist support report: {error}; stderr: {}",
                String::from_utf8_lossy(&output.stderr).trim()
            ))
        })?;
    validate_namelist_support_report(&report)?;
    if output.status.success() != (report.verdict == "PASS") {
        return Err(RwWpsError::Engine(format!(
            "native engine exit status {} contradicts namelist verdict {}",
            output.status, report.verdict
        )));
    }
    Ok(report)
}

pub fn validate_namelist_support_report(report: &NamelistSupportReport) -> Result<(), RwWpsError> {
    if report.schema != NAMELIST_SUPPORT_SCHEMA {
        return Err(RwWpsError::Engine(format!(
            "unsupported namelist support schema {:?}",
            report.schema
        )));
    }
    if report.verdict != "PASS" && report.verdict != "FAIL" {
        return Err(RwWpsError::Engine(format!(
            "invalid namelist verdict {:?}",
            report.verdict
        )));
    }
    let passed = report.verdict == "PASS";
    if passed != (report.required_state.stock_wrf_export.verdict == "PASS") {
        return Err(RwWpsError::Engine(
            "top-level and stock-WRF namelist verdicts contradict".to_owned(),
        ));
    }
    if passed != report.issues.is_empty() {
        return Err(RwWpsError::Engine(
            "PASS must have no issues and FAIL must have at least one issue".to_owned(),
        ));
    }
    if report.geometry.domain_count != report.geometry.domains.len() {
        return Err(RwWpsError::Engine(
            "namelist geometry domain_count differs from its domain array".to_owned(),
        ));
    }
    if let Some(max_dom) = report.max_dom
        && report.geometry.domain_count != 0
        && usize::from(max_dom) != report.geometry.domain_count
    {
        return Err(RwWpsError::Engine(
            "namelist max_dom differs from parsed geometry domain count".to_owned(),
        ));
    }
    for (index, domain) in report.geometry.domains.iter().enumerate() {
        if usize::from(domain.grid_id) != index + 1
            || domain.mass_nx + 1 != domain.e_we
            || domain.mass_ny + 1 != domain.e_sn
            || !domain.dx_m.is_finite()
            || !domain.dy_m.is_finite()
            || domain.dx_m <= 0.0
            || domain.dy_m <= 0.0
        {
            return Err(RwWpsError::Engine(format!(
                "invalid parsed geometry for d{:02}",
                index + 1
            )));
        }
    }
    if let Some(vertical) = &report.vertical
        && (vertical.e_vert != vertical.mass_levels + 1
            || vertical.eta_levels.len() != vertical.e_vert
            || vertical.eta_levels.first() != Some(&1.0)
            || vertical.eta_levels.last() != Some(&0.0)
            || !vertical
                .eta_levels
                .windows(2)
                .all(|pair| pair[0].is_finite() && pair[0] > pair[1])
            || !vertical.p_top_requested_pa.is_finite()
            || vertical.p_top_requested_pa <= 0.0)
        {
            return Err(RwWpsError::Engine(
                "invalid explicit vertical grid in namelist support report".to_owned(),
            ));
        }
    if passed
        && report.required_state.stock_wrf_export.domains.len() != report.geometry.domain_count
    {
        return Err(RwWpsError::Engine(
            "PASS report does not inventory every parsed WRF domain".to_owned(),
        ));
    }
    for domain in &report.required_state.stock_wrf_export.domains {
        if !STOCK_WRF_INVENTORIED_MP_PHYSICS.contains(&domain.mp_physics) {
            return Err(RwWpsError::Engine(format!(
                concat!(
                    "stock-WRF export inventories mp_physics={} for ",
                    "d{:02}, which this frontend has no evidenced WRF ",
                    "v4.6.1 package contract for. Its exit status is the ",
                    "documented gate before wrfinput_dNN is built and an ",
                    "unchanged WRF is launched against it, so it must ",
                    "not certify a package member list it never checked ",
                    "against Registry.EM_COMMON. Evidenced here: {:?}. ",
                    "STOCK_WRF_INVENTORIED_MP_PHYSICS records which ids ",
                    "the paired engine inventories that this frontend ",
                    "still refuses, and the breakage each refusal ",
                    "prevents."
                ),
                domain.mp_physics, domain.grid_id, STOCK_WRF_INVENTORIED_MP_PHYSICS
            )));
        }
        for field in domain
            .wrfinput_fields
            .iter()
            .chain(&domain.runtime_state_not_wrfinput)
        {
            // WRF's Registry gives a package member either the four 3-D
            // dimensions or the three 2-D ones (`ij` spec:
            // Registry.EM_COMMON:492-493 for mp=28's surface aerosol
            // emission pair, :1739 for taod5502d).  Both are valid stock
            // declarations, so both are admitted BY SHAPE; anything else
            // is still a declaration this frontend cannot certify.
            const STOCK_3D: [&str; 4] = ["Time", "bottom_top", "south_north", "west_east"];
            const STOCK_2D: [&str; 3] = ["Time", "south_north", "west_east"];
            let shape_ok = field.dimensions == STOCK_3D || field.dimensions == STOCK_2D;
            if field.dtype != "float32" || !shape_ok {
                return Err(RwWpsError::Engine(format!(
                    "invalid stock-WRF state declaration for {}: dtype {} \
                     dimensions {:?}; WRF v4.6.1 declares a package member \
                     as float32 on {:?} or {:?}",
                    field.netcdf_name, field.dtype, field.dimensions, STOCK_3D, STOCK_2D
                )));
            }
        }
    }
    Ok(())
}

fn require_file(path: &Path, label: &str) -> Result<(), RwWpsError> {
    let metadata = std::fs::metadata(path).map_err(|source| io_error(path, source))?;
    if !metadata.is_file() {
        return Err(RwWpsError::Config(format!(
            "{label} path {} is not a regular file",
            path.display()
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn field(name: &str, collection: Option<&str>) -> Value {
        json!({
            "registry_name": name.to_ascii_lowercase(),
            "netcdf_name": name,
            "collection": collection,
            "dtype": "float32",
            "dimensions": ["Time", "bottom_top", "south_north", "west_east"],
            "units": if collection == Some("scalar") { "# kg-1" } else { "kg kg-1" },
            "initialization": "zero_if_source_absent"
        })
    }

    fn valid_json(mp_physics: u16) -> Value {
        let mut fields = vec![field("QVAPOR", Some("moist"))];
        if mp_physics == 8 {
            fields.push(field("QNICE", Some("scalar")));
            fields.push(field("QNRAIN", Some("scalar")));
        }
        if matches!(mp_physics, 50 | 51) {
            // Registry.EM_COMMON:3038, package p3_1category:
            // moist:qv,qc,qr,qi and scalar:qni,qnr,qir,qib.  QSNOW and
            // QGRAUP are absent on purpose -- P3 carries one ice category
            // and spans the graupel-to-snow continuum with the rime pair.
            for name in ["QCLOUD", "QRAIN", "QICE"] {
                fields.push(field(name, Some("moist")));
            }
            for name in ["QNICE", "QNRAIN", "QIR", "QIB"] {
                fields.push(field(name, Some("scalar")));
            }
        }
        if mp_physics == 51 {
            fields.push(field("QNCLOUD", Some("scalar")));
        }
        json!({
            "schema": NAMELIST_SUPPORT_SCHEMA,
            "verdict": "PASS",
            "max_dom": 1,
            "classifications": {
                "preprocessing_relevant": [],
                "physics_state_relevant": [],
                "runtime_output_only": [],
                "legacy_stage_only": []
            },
            "geometry": {"domain_count": 1, "domains": [{
                "grid_id": 1, "parent_id": 0, "parent_grid_ratio": 1,
                "parent_time_step_ratio": 1, "i_parent_start": 1,
                "j_parent_start": 1, "e_we": 101, "e_sn": 81,
                "mass_nx": 100, "mass_ny": 80, "dx_m": 12000.0,
                "dy_m": 12000.0
            }]},
            "vertical": {
                "e_vert": 4, "mass_levels": 3,
                "eta_levels": [1.0, 0.7, 0.3, 0.0],
                "p_top_requested_pa": 5000.0, "hybrid_opt": 2,
                "etac": 0.2, "source_top_pressure_pa": null,
                "coverage": "deferred_to_source_mapping"
            },
            "required_state": {
                "stock_wrf_export": {"verdict": "PASS", "target": "unchanged WRF v4.6.1", "domains": [{
                    "grid_id": 1, "mp_physics": mp_physics,
                    "microphysics": match mp_physics {
                        8 => "Thompson",
                        50 => "P3 one-category two-moment ice",
                        51 => "P3 one-category two-moment ice, predicted droplet number",
                        _ => "WSM6",
                    },
                    "bl_pbl_physics": 1, "sf_sfclay_physics": 91,
                    "sf_surface_physics": 2, "num_soil_layers": 4,
                    "wrfinput_fields": fields,
                    "runtime_state_not_wrfinput": []
                }]},
                "gpuwm_runtime": {"verdict": "PASS", "reasons": []}
            },
            "issues": []
        })
    }

    #[test]
    fn python_schema_round_trips_through_typed_rust_binding() {
        let report: NamelistSupportReport = serde_json::from_value(valid_json(8)).unwrap();
        validate_namelist_support_report(&report).unwrap();
        let round_trip = serde_json::to_value(report).unwrap();
        assert_eq!(round_trip["schema"], NAMELIST_SUPPORT_SCHEMA);
        assert_eq!(
            round_trip["required_state"]["stock_wrf_export"]["domains"][0]["wrfinput_fields"][2]["netcdf_name"],
            "QNRAIN"
        );
    }

    #[test]
    fn schema_drift_and_false_pass_fail_closed() {
        let mut drift = valid_json(6);
        drift["unexpected"] = json!(true);
        assert!(serde_json::from_value::<NamelistSupportReport>(drift).is_err());

        let mut contradiction = valid_json(6);
        contradiction["issues"] = json!([{
            "code": "X", "location": "&physics/x", "message": "bad", "action": "fix"
        }]);
        let report: NamelistSupportReport = serde_json::from_value(contradiction).unwrap();
        assert!(validate_namelist_support_report(&report).is_err());
    }

    #[test]
    fn p3_one_category_stock_export_is_forwarded_not_refused() {
        // The paired engine inventories mp=50 from Registry.EM_COMMON:3038
        // and emits it inside a PASS report
        // (gpuwm/wrf_physics_inventory.py::_INVENTORIES[50]).  Before this
        // frontend declared the id, that report was refused here, so the
        // documented step-one preflight could not be completed for a P3
        // case whose stock-WRF package contract is fully evidenced.
        let report: NamelistSupportReport = serde_json::from_value(valid_json(50)).unwrap();
        validate_namelist_support_report(&report).unwrap();
        let members: Vec<&str> = report.required_state.stock_wrf_export.domains[0]
            .wrfinput_fields
            .iter()
            .map(|entry| entry.netcdf_name.as_str())
            .collect();
        assert!(members.contains(&"QIR") && members.contains(&"QIB"));
        assert!(!members.contains(&"QSNOW") && !members.contains(&"QGRAUP"));
    }

    #[test]
    fn p3_predicted_droplet_stock_export_carries_qncloud() {
        let report: NamelistSupportReport = serde_json::from_value(valid_json(51)).unwrap();
        validate_namelist_support_report(&report).unwrap();
        let fields = &report.required_state.stock_wrf_export.domains[0].wrfinput_fields;
        assert!(fields.iter().any(|entry| entry.netcdf_name == "QNCLOUD"));
        assert!(fields.iter().any(|entry| entry.netcdf_name == "QIR"));
        assert!(!fields.iter().any(|entry| entry.netcdf_name == "QSNOW"));
    }

    #[test]
    fn undeclared_stock_inventory_refusal_names_its_breakage() {
        // A refusal stands only if it names the breakage it prevents.  This
        // holds the line that the message stays a named refusal rather than
        // decaying back to "not in the list"; mp=16 is an id no producer row
        // inventories.
        let report: NamelistSupportReport = serde_json::from_value(valid_json(16)).unwrap();
        let message = validate_namelist_support_report(&report)
            .unwrap_err()
            .to_string();
        assert!(message.contains("mp_physics=16"), "{message}");
        assert!(message.contains("Registry.EM_COMMON"), "{message}");
        assert!(
            message.contains("STOCK_WRF_INVENTORIED_MP_PHYSICS"),
            "{message}"
        );
        assert!(message.contains("50"), "{message}");
    }

    #[test]
    fn invocation_preserves_paths_and_source_top() {
        let request = NamelistSupportRequest {
            wps: PathBuf::from("a/namelist.wps"),
            input: PathBuf::from("b/namelist.input"),
            source_top_pressure_pa: Some(5000.0),
        };
        assert_eq!(
            build_namelist_support_args(&request),
            [
                "--namelist-support-report",
                "--wps-namelist",
                "a/namelist.wps",
                "--namelist-input",
                "b/namelist.input",
                "--source-top-pressure-pa",
                "5000",
            ]
            .map(OsString::from)
        );
    }
}
