"""Inspect installed radar contracts without fetching artifacts or using CUDA."""
from __future__ import annotations

from importlib import metadata
from importlib.machinery import PathFinder
import json
import subprocess

from woof.simulated_radar_config import (
    FIELDS, FORMATS, MANIFEST_SCHEMA, REQUEST_SCHEMA, SCAN_STRATEGIES,
    SimulatedRadarOptions,
)


def _version(distribution):
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def _module_present(name):
    # Importing a companion's package can import its model backend. Inspect
    # the module locations only: presence is evidence, not route admission.
    locations = None
    parts = name.split(".")
    for count in range(1, len(parts) + 1):
        spec = PathFinder.find_spec(".".join(parts[:count]), locations)
        if spec is None:
            return False
        locations = spec.submodule_search_locations
    return True


def _probe(binary, flag, expected, env):
    try:
        result = subprocess.run([str(binary), flag], capture_output=True,
                                text=True, timeout=10, env=env)
    except (OSError, subprocess.SubprocessError) as error:
        return {"available": False, "status": "probe_failed", "detail": str(error),
                "expected": expected}
    observed = result.stdout.strip()
    available = result.returncode == 0 and observed == expected
    return {"available": available,
            "status": "verified" if available else "incompatible",
            "expected": expected, "observed": observed,
            "returncode": result.returncode}


def native_capabilities():
    from woof import bridges, rustwx

    with bridges.inspection_only():
        try:
            binary = rustwx.simulated_radar_binary()
        except (OSError, RuntimeError) as error:
            return {"available": False, "status": "resolution_failed", "detail": str(error)}
        if binary is None:
            if bridges.on_demand_fetch_offered("rw_simradar"):
                # On demand since 2.9.11 (PyPI per-file limit).
                return {"available": False, "status": "on_demand",
                        "detail": "rw_simradar is staged from this release's bundle by the "
                                  "first radar run; woof fetch-bridges --partial "
                                  "--artifact rw_simradar stages it now"}
            return {"available": False, "status": "missing",
                    "detail": "rw_simradar is not installed"}
        env = rustwx.renderer_env()
        request = _probe(binary, "--abi", rustwx.SIMULATED_RADAR_ABI, env)
        columns = _probe(binary, "--canonical-abi", rustwx.CANONICAL_RADAR_ABI, env)
        features = {"status": "unavailable"}
        try:
            result = subprocess.run([str(binary), "--capabilities"], capture_output=True,
                                    text=True, timeout=10, env=env)
            value = json.loads(result.stdout)
            if (result.returncode == 0 and isinstance(value, dict)
                    and value.get("schema") == "rw-simradar.capabilities/v1"
                    and value.get("request_abi") == rustwx.SIMULATED_RADAR_ABI):
                features = {"status": "verified", **value}
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
    return {"available": request["available"], "status": request["status"],
            "binary": str(binary), "request": request, "canonical_columns": columns,
            "features": features}


def input_contracts():
    return {
        "wrf": {
            "cli_input_kind": "wrf",
            "description": "Full WRF-shaped atmospheric columns, including canonical scenes",
            "required_variables": ["XLAT", "XLONG", "HGT", "T", "PH", "PHB", "U", "V", "W"],
            "reflectivity_alternatives": [
                ["REFL_10CM"], ["P", "PB", "T", "QVAPOR", "QRAIN"]],
            "required_dimensions": ["bottom_top", "south_north", "west_east"],
            "temperature_dimensions": ["Time", "bottom_top", "south_north", "west_east"],
            "native_winds_marker": "RADAR_NATIVE_WINDS=earth-relative-mass-grid/v1",
            "native_winds_variables": ["RADAR_U_EARTH", "RADAR_V_EARTH"],
            "optional_variables": [
                "SINALPHA", "COSALPHA", "QCLOUD", "QICE", "QSNOW", "QGRAUP", "QHAIL",
                "QNCLOUD", "QNRAIN", "QNICE", "QNSNOW", "QNGRAUP", "QNHAIL",
                "QIR", "QIB", "QZI", "QICE2", "QNICE2", "QIR2", "QIB2",
                "QVOLI", "QAOLI", "QVOLI2", "QAOLI2", "QICE3", "QNICE3",
                "QVOLI3", "QAOLI3", "D_ICE", "RHO_ICE", "PHI_ICE", "V_ICE",
                "D_ICE2", "RHO_ICE2", "PHI_ICE2", "V_ICE2",
                "D_ICE3", "RHO_ICE3", "PHI_ICE3", "V_ICE3", "TKE_PBL", "TKE", "QKE",
            ],
            "time": "Internal Times is authoritative; named WRF timestamp fallback is reported",
            "missing_fields_refusal": "radar_input_missing_columns",
            "display_only_2d_accepted": False,
        },
        "native_columns": {
            "cli_input_kind": "native-columns",
            "schema": "native-atmosphere.columns/v1",
            "required_variables": [
                "latitude_deg", "longitude_deg", "pressure_pa", "eastward_wind_m_s",
                "northward_wind_m_s", "qv_kg_kg", "qc_kg_kg", "qr_kg_kg", "qi_kg_kg",
                "qs_kg_kg", "qg_kg_kg", "height_half_m", "vertical_velocity_half_m_s",
                "terrain_height_m"],
            "temperature_alternatives": [["temperature_k"], ["potential_temperature_k"]],
            "dimensions": {"mass": ["level", "latitude", "longitude"],
                           "interfaces": ["interface", "latitude", "longitude"]},
            "vertical_order": "top-to-surface",
            "wind_convention": "earth-relative mass-grid horizontal, geometric upward interface vertical",
            "moisture_convention": "kg per kg moist air, converted to dry-air by the Rust adapter",
            "required_text_attributes": ["schema", "valid_time", "simulation_start", "source_model",
                                         "source_checkpoint", "config_sha256", "microphysics_scheme",
                                         "vertical_velocity_method", "derivative_stencil"],
            "required_numeric_attributes": ["mp_physics", "derivative_interval_s", "gravity_m_s2"],
            "adapter": "rw_simradar --canonical-atmosphere SOURCE.nc --out SCENE.nc",
            "python_adapter": "woof.rustwx.canonical_radar_scene",
        },
    }


def describe():
    native = native_capabilities()
    request_available = native["available"]
    columns_available = native.get("canonical_columns", {}).get("available", False)
    features = native.get("features", {})
    routes = {
        "regional": {
            "available": request_available,
            "entrypoint": "woof go with [simulated_radar] in the experiment",
            "prepared_entrypoint": "woof sim --simulated-radar-table JSON",
            "input": "committed full WRF history",
        },
        "cyclone": {
            "available": request_available,
            "entrypoint": "the regional experiment and prepared-run routes",
            "input": "committed full WRF history for each domain",
        },
    }
    for model, distribution, module, commit, entrypoint, requirements in (
        ("hex", "woof hex", "hexcore.simulated_radar",
         "edd87c72eae24187bf860f1d763874b0873bd9c8",
         "woof hex forecast ... --simulated-radar radar.toml --radar-window WINDOW",
         ["companion forecast door forwards both radar arguments",
          "rw_mpas_convert field_set=full with the run mesh and initial vertical coordinate",
          "engine/companion compatibility admission for the installed artifacts"]),
        ("global", "woof global", "arwen_global.simulated_radar",
         "9f785b375e4d024233dd3c33233c8f3c4da2518d",
         "woof global run experiment.toml --outdir OUTPUT_ROOT",
         ["companion experiment carries [simulated_radar] into its checkpoint producer",
          "canonical-atmosphere temperature/wind ABI and Rust NetCDF transport writer",
          "UTC origin and native one-step geometric vertical-wind derivative"]),
    ):
        present = _module_present(module)
        routes[model] = {
            # A package version or a helper module does not prove its forecast
            # driver forwards options. Do not offer an ignored requested product.
            "available": None if request_available and present else False,
            "status": "requires_companion_admission" if present else "missing_companion_hook",
            "distribution": distribution, "installed_version": _version(distribution),
            "hook_module": module, "hook_module_present": present,
            "qualified_source_commit": commit,
            "version_is_capability_proof": False,
            "entrypoint": entrypoint, "requirements": requirements,
        }
    return {
        "schema": REQUEST_SCHEMA, "manifest_schema": MANIFEST_SCHEMA,
        "capabilities_schema": "simulated-radar.capabilities/v1",
        "engine_version": _version("woof"),
        "defaults": SimulatedRadarOptions(enabled=True).to_mapping(),
        "scan_strategies": SCAN_STRATEGIES, "formats": FORMATS, "fields": FIELDS,
        "native": native,
        "supports": {"history_replay": request_available,
                     "native_columns_adapter": request_available and columns_available,
                     "resource_estimate": request_available and features.get("resource_estimate_schema") == "simulated-radar.resources/v1",
                     "named_input_refusals": request_available and features.get("input_validation") == "full-columns/v1",
                     "display_only_2d": False, "live_routes": routes},
        "inputs": input_contracts(),
        "resource_estimate": {
            "entrypoint": "rw_simradar --estimate REQUEST.json",
            "description": "Native dimensions and host-memory admission; empty history_paths gives geometry only",
            "not_a_price_quote": True,
        },
    }
