"""Compile reviewed research recipes through WOOF-ArWen's native domain planner.

Creation writes a new configuration bundle only. It does not acquire forcing,
prepare a state, integrate a forecast, or turn execution qualification into
a successful scientific result. The TOML is published last as the commit marker.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
import tomllib

from woof.configuration_recovery import MemoryAdmissionError, retain_final_candidate
from woof.explain import warn
from woof.filesystem_paths import publish_new
from woof.tui_products import _unique_object


CATALOG_PATH = Path(__file__).parent / "data" / "tui" / "research-workspaces.json"
HARDWARE_PATH = CATALOG_PATH.with_name("research-hardware-profiles.json")
DIAGNOSTICS_PATH = CATALOG_PATH.with_name("research-diagnostics.json")
HARDWARE_CLASSES = ("8", "12", "16", "24", "32")
_METHODS = {"regional", "nested", "moving_nest", "controlled_scenario",
            "archived_downscale"}
_RECIPE_KEYS = {"id", "leaf_id", "title", "research_question", "method",
                "plot_preset", "geometry", "diagnostics", "input_requirements",
                "tracker", "scenario", "comparison", "limitations",
                "citation_ids", "requires_existing_state", "qualification_status",
                "validation_status", "further_analysis"}
_GEOMETRY_KEYS = {"root_dx_km", "nest_ratios", "forecast_hours",
                  "history_interval_s", "extent_intent", "preferred_finest_dx_km",
                  "minimum_domain_span_km", "minimum_root_span_km"}
_TRACKER_KEYS = {"kind", "field", "units", "level_hpa", "threshold",
                 "fallback_threshold", "cadence_seconds", "notes", "attribute",
                 "extremum", "reduction", "model_level"}
_SCENARIO_KEYS = {"operation", "amplitude_k", "center_height_m", "radius_km",
                  "depth_m", "rh_preserve", "placement"}


_QUALIFICATION_STATUS_ALIASES = {
    "unvalidated": "unqualified",
    "unvalidated-candidate-policy": "unqualified-candidate-policy",
    "catalog recommendations are not science validation": "unqualified",
}
_LEGACY_QUALIFICATION_STATUSES = {
    "unqualified": "unvalidated",
    "unqualified-candidate-policy": "unvalidated-candidate-policy",
}


def qualification_status(recipe: dict) -> str:
    """Read execution qualification, accepting historical field/value aliases."""
    statuses = {}
    for key in ("qualification_status", "validation_status"):
        if key not in recipe:
            continue
        value = recipe[key]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Research recipe {key} must be nonempty text")
        statuses[key] = _QUALIFICATION_STATUS_ALIASES.get(value, value)
    if not statuses:
        raise ValueError("Research recipe is missing qualification_status "
                         "(legacy validation_status is also accepted)")
    if len(set(statuses.values())) != 1:
        raise ValueError("Research recipe qualification_status and legacy "
                         "validation_status disagree; use one status or matching aliases")
    return next(iter(statuses.values()))


def _qualification_fields(recipe: dict) -> dict[str, str]:
    status = qualification_status(recipe)
    # Keep the old exported field and spelling for existing JSON consumers.
    legacy = recipe.get("validation_status", _LEGACY_QUALIFICATION_STATUSES.get(status, status))
    return {"qualification_status": status, "validation_status": legacy}


def _read_json(path: Path, label: str) -> tuple[dict, str]:
    try:
        payload = path.read_bytes()
        document = json.loads(payload.decode("utf-8"), object_pairs_hook=_unique_object)
    except (OSError, ValueError) as error:
        raise ValueError(f"Cannot read packaged {label} at {path}: {error}. "
                         "Restore that file from the matching WOOF-ArWen installation, then retry.") from error
    if not isinstance(document, dict):
        raise ValueError(f"Packaged {label} at {path} must be a JSON object. "
                         "Restore the matching WOOF-ArWen package data, then retry.")
    return document, hashlib.sha256(payload).hexdigest()


def _hardware_document() -> tuple[dict, str]:
    document, digest = _read_json(HARDWARE_PATH, "research hardware profiles")
    try:
        if document.get("schema") != "arwen.research.hardware-profiles.v1":
            raise ValueError("expected schema arwen.research.hardware-profiles.v1")
        profiles = document["profiles"]
        if not isinstance(profiles, dict) or set(profiles) != set(HARDWARE_CLASSES):
            raise ValueError("profiles must declare classes 8, 12, 16, 24 and 32")
        if not isinstance(document["policy"], str) or not document["policy"].strip():
            raise ValueError("policy must be nonempty text")
        for name, profile in profiles.items():
            if not isinstance(profile, dict):
                raise ValueError(f"profile {name} must be an object")
            nz = profile["nz"]
            if isinstance(nz, bool) or not isinstance(nz, int) or nz < 2:
                raise ValueError(f"profile {name} nz must be an integer >= 2")
            if not isinstance(profile["tradeoff"], str):
                raise ValueError(f"profile {name} tradeoff must be text")
            for intent in ("regional", "mesoscale", "storm", *(("storm_context",) if "storm_context" in profile else ())):
                geometry = profile[intent]
                _number(geometry["root_dx_km"], f"profile {name} {intent} root_dx_km")
                ratios = geometry["nest_ratios"]
                if not isinstance(ratios, list) or any(isinstance(r, bool) or not isinstance(r, int) or r < 2 for r in ratios):
                    raise ValueError(f"profile {name} {intent} nest_ratios must contain integer ratios >= 2")
            if "storm_context" in profile:
                _number(profile["storm_context_minimum_km"], f"profile {name} storm_context_minimum_km")
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid research hardware profiles at {HARDWARE_PATH}: {error}. "
                         "Restore the matching WOOF-ArWen package data, then retry.") from error
    return document, digest


def _keys(value: dict, allowed: set[str], label: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"Unknown {label} fields: {', '.join(unknown)}")


def _number(value, label: str, *, positive: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or not math.isfinite(value) or (value <= 0 if positive else value < 0):
        raise ValueError(f"{label} must be a finite {'positive' if positive else 'nonnegative'} number")
    return float(value)


def validate_recipe(recipe: dict, *, known_products: set[str] | None = None,
                    _capabilities: dict | None = None) -> None:
    """Reject unsupported declarations before probing hardware or writing files."""
    _keys(recipe, _RECIPE_KEYS, "recipe")
    missing = _RECIPE_KEYS - {"further_analysis", "qualification_status", "validation_status"} - set(recipe)
    if missing:
        raise ValueError(f"Research recipe is missing fields: {', '.join(sorted(missing))}")
    if not isinstance(recipe["id"], str) or not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", recipe["id"]):
        raise ValueError("Research recipe id must use lowercase letters, digits, dots, underscores or hyphens")
    qualification_status(recipe)
    for key in ("leaf_id", "title", "research_question", "comparison"):
        if not isinstance(recipe[key], str) or not recipe[key].strip():
            raise ValueError(f"Research recipe {key} must be nonempty text")
    for key in ("input_requirements", "limitations", "citation_ids", "further_analysis"):
        values = recipe.get(key, [])
        if not isinstance(values, list) or any(not isinstance(value, str) or not value.strip() for value in values):
            raise ValueError(f"Research recipe {key} must be a list of nonempty strings")
    method = recipe["method"]
    if method not in _METHODS:
        raise ValueError(f"Unsupported research method: {method}")
    geometry = recipe["geometry"]
    _keys(geometry, _GEOMETRY_KEYS, "geometry")
    if _GEOMETRY_KEYS - set(geometry):
        raise ValueError("Research geometry must declare its complete intent and minimum root span")
    for key in ("root_dx_km", "forecast_hours", "history_interval_s", "preferred_finest_dx_km"):
        _number(geometry.get(key), f"geometry.{key}")
    if int(geometry["forecast_hours"]) != geometry["forecast_hours"]:
        raise ValueError("geometry.forecast_hours must be a whole number of hours")
    ratios = geometry.get("nest_ratios")
    if not isinstance(ratios, list) or any(isinstance(ratio, bool) or not isinstance(ratio, int)
                                          or ratio < 2 for ratio in ratios):
        raise ValueError("geometry.nest_ratios must contain integer ratios >= 2")
    if method == "regional" and ratios:
        raise ValueError("A regional recipe cannot silently create nests")
    if method in {"nested", "moving_nest", "controlled_scenario"} and not ratios:
        raise ValueError(f"{method} requires at least two domains")
    _number(geometry["minimum_root_span_km"], "geometry.minimum_root_span_km", positive=False)
    if not isinstance(recipe["requires_existing_state"], bool):
        raise ValueError("requires_existing_state must be a boolean")
    if method == "archived_downscale" and not recipe["requires_existing_state"]:
        raise ValueError("Archived downscaling requires an existing parent state")
    tracker = recipe["tracker"]
    if (method == "moving_nest") != (tracker is not None):
        raise ValueError("A tracker must be present exactly for moving_nest recipes")
    if tracker is not None:
        _keys(tracker, _TRACKER_KEYS, "tracker")
        if tracker.get("kind") != "feature_follow":
            raise ValueError("Only feature_follow tracking is supported")
        if tracker.get("field") not in {"pressure", "uh", "reflectivity", "attribute"}:
            raise ValueError("Trackers support pressure, uh, reflectivity or native attribute only")
        from woof.core.attribute_tracking import ATTRIBUTE_UNITS
        units = (ATTRIBUTE_UNITS.get(tracker.get("attribute")) if tracker["field"] == "attribute" else
                 "m2/s2" if tracker["field"] == "uh" else
                 "dBZ" if tracker["field"] == "reflectivity" else
                 "hPa" if tracker.get("level_hpa") == 0 else "m")
        if tracker.get("units") != units:
            raise ValueError(f"Tracker {tracker['field']} at this surface requires units {units}")
        _number(tracker.get("cadence_seconds"), "tracker.cadence_seconds")
        # The native validator owns the field-dependent threshold/level units.
        from woof.core.storm_tracking import build_follow_config
        build_follow_config(_follow_values(tracker), recipe["id"])
    scenario = recipe["scenario"]
    if (method == "controlled_scenario") != (scenario is not None):
        raise ValueError("A scenario must be present exactly for controlled_scenario recipes")
    if scenario is not None:
        _keys(scenario, _SCENARIO_KEYS, "scenario")
        if scenario.get("operation") != "warm_bubble":
            raise ValueError("Only the native warm_bubble initial perturbation is supported")
        if scenario.get("placement") != "user-selected domain centre":
            raise ValueError("Warm bubbles require the user-selected domain centre")
        from woof.experiment import BubbleConfig
        BubbleConfig(center_lat=0, center_lon=0, **{
            key: scenario[key] for key in _SCENARIO_KEYS - {"operation", "placement"}})
    from woof.tui_products import presets
    preset = next((row for row in presets()["presets"] if row["id"] == recipe["plot_preset"]), None)
    if preset is None:
        raise ValueError(f"Unknown plot preset: {recipe['plot_preset']}")
    capabilities = diagnostic_capabilities() if _capabilities is None else _capabilities
    if not isinstance(recipe["diagnostics"], list) or not recipe["diagnostics"] \
            or any(not isinstance(product, str) or not product.strip() for product in recipe["diagnostics"]):
        raise ValueError("Recipe diagnostics must name supported native renderer products")
    # ONE authority for "can this lane draw that?", and it is not this
    # file.  The renderer's catalog is the vocabulary; the packaged JSON
    # records only two things the renderer cannot state for itself -- the
    # concrete reason a product is unserved on this lane, and the window
    # a product needs before it means anything.
    #
    # `known_products` is the renderer's catalog when a caller has one.
    # This door has none: it runs at packaged-catalog load and at recipe
    # lookup, where there is no run, no wrfout, no store root and no
    # built renderer, and creating a research TOML is promised not to
    # need one.  Absent, the vocabulary check is simply not made, which
    # is the truthful answer here rather than a smaller catalog standing
    # in for the real one.
    for product in recipe["diagnostics"]:
        reason = capabilities["unavailable"].get(product)
        if reason is not None:
            raise ValueError(f"Research diagnostic {product} is not supported by the WOOF-ArWen history renderer. {reason}")
        if known_products is not None and product not in known_products:
            raise ValueError(f"Research diagnostic {product} is not supported by the WOOF-ArWen history renderer. The renderer's catalog carries no product of that name; woof render --list-products names every product this install can draw. Recipe diagnostics must name supported native renderer products")
        recorded = capabilities["products"].get(product)
        if recorded is None:
            # No recorded row is not a breakage and not a refusal.  The
            # product runs, priced from the most conservative recorded
            # basis -- no recorded window requirement, so zero hours --
            # with that basis stated and one warning.  A window that IS
            # required is still enforced downstream, by name and with its
            # own reason, so running cannot produce a wrong picture here.
            warn(f"research diagnostic {product} has no recorded window "
                 f"requirement in {DIAGNOSTICS_PATH.name}, so {recipe['id']} "
                 "is priced with none; the renderer reports its own "
                 "availability for this product at render time.",
                 "The packaged record carries a row per product whose "
                 "window requirement was measured. A product with no row "
                 "is one nobody measured a window for, which is not the "
                 "same as one that cannot be drawn: the render catalog "
                 "answers availability per store, and this file never "
                 "narrows it.")
            continue
        minimum = recorded["minimum_hours"]
        if geometry["forecast_hours"] < minimum:
            raise ValueError(f"Research diagnostic {product} needs at least {minimum} hours of history; {recipe['id']} declares {geometry['forecast_hours']:g} hours. Select a matching window or a longer study.")


def diagnostic_capabilities() -> dict:
    """The packaged lane record, read by the one function both doors call.

    The reader lives in :mod:`woof.tui_products` beside the presets,
    because the preset picker asks this same file the same question and
    two readers of one record is how a picker and a recipe validator
    come to disagree about one configuration.  The PATH stays this
    module's, so a caller that points this door at another copy still
    points the reader at it.
    """

    from woof.tui_products import lane_capabilities

    return lane_capabilities(DIAGNOSTICS_PATH)


def _native_products() -> set[str]:
    """Packaged input capabilities; creating TOML never needs a renderer process."""
    return set(diagnostic_capabilities()["products"])


def catalog_document() -> dict:
    document, catalog_digest = _read_json(CATALOG_PATH, "research catalog")
    if document.get("schema_version") != 1 or document.get("catalog_id") != "arwen-research-workspaces-v1":
        raise ValueError(f"Unsupported research catalog schema at {CATALOG_PATH}; expected arwen-research-workspaces-v1 / schema_version 1. Restore the matching WOOF-ArWen package data, then retry.")
    capabilities = diagnostic_capabilities()
    try:
        collections = {}
        for key in ("families", "submodes", "leaves", "configurations", "citations"):
            rows = document[key]
            if not isinstance(rows, list) or not rows or any(not isinstance(row, dict) or not isinstance(row.get("id"), str) for row in rows):
                raise ValueError(f"{key} must be a nonempty list of objects with string ids")
            collections[key] = {row["id"]: row for row in rows}
            if len(collections[key]) != len(rows):
                raise ValueError(f"duplicate id in {key}")
        for family in collections["families"].values():
            if set(family["submode_ids"]) != {row["id"] for row in collections["submodes"].values() if row["family_id"] == family["id"]}:
                raise ValueError(f"family {family['id']} has inconsistent submode_ids")
        for submode in collections["submodes"].values():
            if submode["family_id"] not in collections["families"]:
                raise ValueError(f"submode {submode['id']} has unknown family_id")
            if set(submode["leaf_ids"]) != {row["id"] for row in collections["leaves"].values() if row["submode_id"] == submode["id"]}:
                raise ValueError(f"submode {submode['id']} has inconsistent leaf_ids")
        for leaf in collections["leaves"].values():
            if leaf["submode_id"] not in collections["submodes"]:
                raise ValueError(f"leaf {leaf['id']} has unknown submode_id")
            if set(leaf["configuration_ids"]) != {row["id"] for row in collections["configurations"].values() if row["leaf_id"] == leaf["id"]}:
                raise ValueError(f"leaf {leaf['id']} has inconsistent configuration_ids")
        for row in collections["configurations"].values():
            validate_recipe(row, _capabilities=capabilities)
            row.update(_qualification_fields(row))
            if row["leaf_id"] not in collections["leaves"]:
                raise ValueError(f"configuration {row['id']} has unknown leaf_id")
            if any(citation not in collections["citations"] for citation in row["citation_ids"]):
                raise ValueError(f"configuration {row['id']} has an unknown citation")
        for key in ("families", "submodes"):
            for row in collections[key].values():
                if any(config not in collections["configurations"] for config in row["recommended_config_ids"]):
                    raise ValueError(f"{row['id']} recommends an unknown configuration")
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid research catalog at {CATALOG_PATH}: {error}. Restore the matching WOOF-ArWen package data, then retry.") from error
    hardware, hardware_digest = _hardware_document()
    return {**document, "hardware_profiles": hardware["profiles"],
            "hardware_policy": hardware["policy"], "catalog_sha256": catalog_digest,
            "hardware_policy_sha256": hardware_digest,
            "diagnostic_capabilities": capabilities}


def effective_profile(recipe: dict, hardware_class: str, *, hardware: dict | None = None) -> dict:
    """The exact visible policy shared by catalog consumers and the compiler."""
    if hardware is None:
        hardware, _ = _hardware_document()
    profile = hardware["profiles"][hardware_class]
    original = recipe["geometry"]
    parent_dependent = recipe["requires_existing_state"] and recipe["method"] != "controlled_scenario"
    if parent_dependent:
        intent, nz = "existing-parent" if recipe["method"] == "archived_downscale" else "existing-scenario", None
        geometry = dict(original)
        tradeoff = "The existing state determines the actual parent grid and vertical levels; class selection cannot replace that evidence."
    elif recipe["method"] == "controlled_scenario":
        intent, nz = "controlled-comparison", 49
        geometry = dict(original)
        tradeoff = "Controlled comparisons preserve the same horizontal ladder and 49 levels across classes."
    else:
        intent = ("regional" if not original["nest_ratios"] else
                  "mesoscale" if original["preferred_finest_dx_km"] >= 3 else "storm")
        context = (intent == "storm" and "storm_context" in profile
                   and original["minimum_root_span_km"] >= profile["storm_context_minimum_km"])
        geometry = {**original, **profile["storm_context" if context else intent]}
        nz = profile["nz"]
        tradeoff = profile["tradeoff"]
        if context:
            tradeoff += (f" This recipe requires {original['minimum_root_span_km']:g} km root span, "
                         f"so it selects the three-domain context ladder at the "
                         f"{profile['storm_context_minimum_km']:g} km policy boundary.")
    finest = geometry["root_dx_km"] / math.prod(geometry["nest_ratios"])
    return {"hardware_class": hardware_class, "intent": intent, "nz": nz,
            "geometry": geometry, "requested_geometry": original,
            "effective_finest_dx_km": finest, "tradeoff": tradeoff,
            "parent_dependent": parent_dependent,
            "resolution_tradeoff": ("Existing parent/state metadata must resolve this geometry"
                                    if parent_dependent else
                                    "Profile is coarser than the recipe's preferred finest spacing"
                                    if finest > original["preferred_finest_dx_km"] else
                                    "Profile meets or exceeds the recipe's preferred finest spacing"),
            "qualification_status": "unqualified-candidate-policy",
            "validation_status": "unvalidated-candidate-policy"}


def _recipe(recipe_id: str, *, document: dict | None = None) -> dict:
    document = catalog_document() if document is None else document
    matches = [row for row in document["configurations"] if row.get("id") == recipe_id]
    if not matches:
        raise ValueError(f"Unknown research configuration {recipe_id}. Run woof research catalog to choose an existing id.")
    if len(matches) != 1:
        raise ValueError(f"Duplicate research configuration {recipe_id} in {CATALOG_PATH}; restore the matching package data.")
    validate_recipe(matches[0], _capabilities=document.get("diagnostic_capabilities"))
    return matches[0]


def _hardware_class(sizing, requested: str) -> str:
    if requested != "auto":
        return requested
    from woof.domain_wizard import card_assumed_free_gib
    # A busy large card should receive a practical small profile. This selects
    # geometry only: the original free-byte sample still prices every candidate.
    # CUDA may report 15.51 GiB on a nominal 16 GiB card after driver
    # reservations. The profile's assumed-free allowance is the comparable
    # quantity, not its marketing capacity. Never increase the measured budget.
    available_gib = min(sizing.vram_gib, sizing.free_bytes / 2**30)
    eligible = [value for value in HARDWARE_CLASSES
                if card_assumed_free_gib(float(value)) <= available_gib + 1e-9]
    return eligible[-1] if eligible else HARDWARE_CLASSES[0]


def _budget_document(sizing, requested: str, *, hardware: dict | None = None) -> dict:
    from woof.domain_wizard import fit_headroom_bytes
    from woof.core.preflight import EXTERNAL_MARGIN_BYTES
    selected = _hardware_class(sizing, requested)
    envelope_budget = sizing.free_bytes - EXTERNAL_MARGIN_BYTES
    device = sizing.device_profile
    if hardware is None:
        hardware, _ = _hardware_document()
    return {
        "requested_class": requested, "selected_class": selected,
        "selection_reason": (f"{sizing.vram_gib:g} GiB GPU, {sizing.free_bytes / 2**30:.2f} GiB free "
                             f"selects the {selected} GiB resource profile" if requested == "auto"
                             else f"Explicit {selected} GiB profile; the memory budget is independent"),
        "profile": hardware["profiles"][selected],
        "basis": "measured" if sizing.measured else "declared-capacity-assumed-free",
        "capacity_gib": sizing.vram_gib, "free_bytes": sizing.free_bytes,
        "external_margin_bytes": EXTERNAL_MARGIN_BYTES,
        "envelope_budget_bytes": envelope_budget,
        "point_fit_headroom_bytes": fit_headroom_bytes(envelope_budget),
        "device_profile": None if device is None else vars(device),
        "note": sizing.note,
    }


def hardware_document(*, hardware_class: str = "auto", vram_gib: float | None = None) -> dict:
    from woof.domain_wizard import resolve_sizing_budget
    sizing = resolve_sizing_budget(None, vram_gib, declare=("--vram-gib",))
    return {"schema": "arwen.research.hardware.v1", **_budget_document(sizing, hardware_class)}


def attributes_document() -> dict:
    """Discover the native closed registry without probing a device or state."""
    from woof.core.attribute_tracking import (ATTRIBUTE_UNITS, ATTRIBUTE_EXTREMA,
                                               ATTRIBUTE_REDUCTIONS)
    meanings = {
        "theta": "Total potential temperature: base state plus perturbation",
        "qv": "Water-vapour mixing ratio per kg of dry air",
        "qc": "Cloud-water mixing ratio per kg of dry air",
        "qr": "Rain-water mixing ratio per kg of dry air",
        "w": "Vertical velocity averaged from adjacent faces onto mass levels",
    }
    return {
        "schema": "arwen.research.attributes.v1",
        "field": "attribute",
        "attributes": [{"id": name, "units": units, "meaning": meanings[name],
                        "requires_moist": name in {"qv", "qc", "qr"}}
                       for name, units in ATTRIBUTE_UNITS.items()],
        "extrema": list(ATTRIBUTE_EXTREMA),
        "reductions": list(ATTRIBUTE_REDUCTIONS),
        "reduction_semantics": {
            "column_max": "Maximum across all source mass levels",
            "column_min": "Minimum across all source mass levels",
            "column_mean": "Unweighted arithmetic mean across all source mass levels",
            "model_level": "One explicitly selected zero-based source mass level",
        },
        "model_level": {"type": "integer", "minimum": 0, "index_basis": "zero-based mass levels",
                        "required_when": "reduction = model_level",
                        "allowed_when": "reduction = model_level",
                        "upper_bound": "strictly less than nz on every source grid"},
        "threshold": {"units": "the selected attribute's units",
                      "max": "select values at or above threshold",
                      "min": "select values at or below threshold"},
        "constraints": [
            "Declare attribute, extremum and reduction explicitly with field = attribute",
            "Omit pressure level_hpa and reflectivity fallback_threshold",
            "Nonfinite values in any sampled level make that column ineligible",
            "Only the listed native attributes and reductions are supported; expressions are not evaluated",
            "Native source-state, movement, timing and memory admission still apply",
        ],
    }


def _follow_values(tracker: dict) -> dict:
    values = {"field": tracker.get("field"), "threshold": tracker.get("threshold"),
              "search_margin_cells": 10, "min_shift_cells": 1,
              "max_shift_cells": 4, "cooldown_seconds": 300.0, "radius_km": 50.0}
    for key in ("level_hpa", "fallback_threshold", "attribute", "extremum", "reduction", "model_level"):
        if tracker.get(key) is not None:
            values[key] = tracker[key]
    return values


def _toml_values(values: dict) -> str:
    return "\n".join(f"{key} = {json.dumps(value, ensure_ascii=False)}" for key, value in values.items())


def _method_text(recipe: dict, *, lat: float, lon: float, grid_id: int) -> str:
    tracker = recipe["tracker"]
    if tracker is not None:
        return ("\n# Reviewed feature following: thresholds are detection settings, not severity.\n"
                "# Prepare the moving child's full statics corridor before forecasting.\n"
                "[relocation]\nenabled = true\n"
                f"grid_id = {grid_id}\n"
                "max_move_parent_cells = 4\nmin_overlap_fraction = 0.5\n"
                f"cadence_seconds = {tracker['cadence_seconds']}\n"
                "\n[relocation.follow]\n" + _toml_values(_follow_values(tracker)) + "\n"
                "\n# Track rows use the existing tracker at its consultation cadence.\n"
                "[relocation.track]\npath = \"storm-track.csv\"\n")
    scenario = recipe["scenario"]
    if scenario is not None:
        values = {"center_lat": lat, "center_lon": lon, **{
            key: scenario[key] for key in ("center_height_m", "radius_km", "depth_m", "amplitude_k", "rh_preserve")}}
        return ("\n# Reviewed warm-bubble initial-state sensitivity experiment.\n"
                "# This is an initiation nudge, not a balanced storm or hurricane insertion.\n"
                "[[perturbation.bubbles]]\n" + _toml_values(values) + "\n")
    return ""


def _native_args(args, recipe: dict, staged: Path, sizing, *, profile: dict | None = None):
    from woof.domain_wizard import register_cli as register_domain_cli
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    register_domain_cli(sub)
    if profile is None:
        selected = _hardware_class(sizing, args.hardware_class)
        profile = effective_profile(recipe, selected)
    geometry = profile["geometry"]
    name = args.name or recipe["id"]
    # All paths that leave the stage are bound to the final bundle directory.
    data_dir = args.out.absolute().parent / "data" / args.out.stem
    command = ["domain", f"--source={args.source}", f"--cycle={args.cycle}",
               f"--hours={args.hours if args.hours is not None else int(geometry['forecast_hours'])}",
               f"--root-dx={geometry['root_dx_km']}",
               f"--nz={profile['nz']}",
               f"--history-interval={geometry['history_interval_s']}",
               f"--nest-history-interval={geometry['history_interval_s']}",
               f"--out={staged}", f"--name={name}", f"--data-dir={data_dir}"]
    if geometry["nest_ratios"]:
        command.append("--chain=" + ",".join(map(str, geometry["nest_ratios"])))
    if args.point is not None:
        command.append(f"--point={args.point}")
    else:
        command.append(f"--polygon={args.polygon.absolute()}")
    if args.physics_profile:
        command.append(f"--physics-profile={args.physics_profile}")
    if args.tiles:
        command.append(f"--tiles={args.tiles}")
    if args.nz is not None:
        command.append(f"--nz={args.nz}")
    for acknowledgement in args.ack:
        command.append(f"--ack={acknowledgement}")
    return parser.parse_args(command), data_dir


def _final_text(text: str, recipe: dict, *, lat: float, lon: float, data_dir: Path, profile: dict) -> str:
    # The wizard deliberately prints cwd-relative fetch commands. A movable
    # staging directory must not become a saved forcing directory in our bundle.
    fetch = re.search(r"(?ms)^\[fetch\]\s*\n(?P<body>.*?)(?=^\[|\Z)", text)
    if fetch:
        body = re.sub(r"(?m)^out\s*=.*$", "out = " + json.dumps(data_dir.as_posix()), fetch["body"])
        text = text[:fetch.start("body")] + body + text[fetch.end("body"):]
    case_data = re.search(r"(?ms)^\[case_data\]\s*\n(?P<body>.*?)(?=^\[|\Z)", text)
    if case_data:
        # This compiler exposes the wizard's default ERA5 acquisition path.
        # Its relative path was authored from the private stage, so bind it to
        # the final data directory while preserving every other case-data key.
        #
        # ONLY THE DIRECTORY MOVES.  The file NAME is the emitter's answer
        # and this function has no better one: the two ERA5 providers
        # publish two different containers, so re-spelling a literal here
        # would overwrite a correct name with the other provider's the
        # moment a recipe selected one -- and the config would then declare
        # a forcing file its own [fetch] table cannot produce, which is
        # refused only after the download has run.  Rebinding the directory
        # and keeping the basename cannot drift from the emitter at all.
        declared = tomllib.loads(case_data["body"])["forcing"]
        forcing = json.dumps([(data_dir / PurePosixPath(str(item)).name).as_posix()
                              for item in declared])
        body = re.sub(r"(?m)^forcing\s*=.*$", "forcing = " + forcing, case_data["body"])
        text = text[:case_data.start("body")] + body + text[case_data.end("body"):]
    heading = (f"# Research configuration: {recipe['id']}\n"
               f"# Method: {recipe['method']}; execution qualification remains {qualification_status(recipe)}.\n")
    return heading + text.rstrip() + "\n" + _method_text(
        recipe, lat=lat, lon=lon, grid_id=len(profile["geometry"]["nest_ratios"]) + 1)


def _admission(text: str, *, recipe: dict, source: str, sizing, path: Path,
               retry_hint: str | None = None) -> tuple[object, dict]:
    from woof import domain_wizard as wizard
    experiment = wizard.experiment_from_text(text, source=str(path))
    interval = wizard.source_forcing_interval_seconds(source)
    phases = wizard._sizing_phases(experiment, free_bytes=sizing.free_bytes,
                                   source=source, forcing_interval_seconds=interval,
                                   vram_gib=sizing.vram_gib, profile=sizing.device_profile)
    budget = wizard.sizing_budget_bytes(experiment, free_bytes=sizing.free_bytes,
                                       vram_gib=sizing.vram_gib,
                                       forcing_interval_seconds=interval,
                                       profile=sizing.device_profile)
    if retry_hint is None:
        fitting_class = _hardware_class(sizing, "auto")
        retry_hint = (f"Retry this same question ({recipe['id']}) with --hardware-class auto "
                      f"or --hardware-class {fitting_class}, keeping the location, source, cycle, "
                      "duration and GPU capacity unchanged. In the TUI, reopen the saved guide "
                      f"and set Research GPU profile to auto or {fitting_class}. "
                      "The retry still checks the full study area; if it also refuses, make more "
                      "memory available or choose a question with a smaller required area.")
    if phases.peak_envelope_bytes > budget:
        raise MemoryAdmissionError(
            f"The complete {recipe['method']} configuration exceeds the immutable "
            f"memory budget: {phases.peak_envelope_bytes / 2**30:.2f} GiB needed, "
            f"{budget / 2**30:.2f} GiB available. {retry_hint}",
            peak_envelope_bytes=phases.peak_envelope_bytes, budget_bytes=budget,
            binding_phase=phases.binding_phase)
    minimum = float(recipe["geometry"].get("minimum_root_span_km", 0))
    root = experiment.domains[0].run
    span_x, span_y = root.nx * root.dx / 1000, root.ny * root.dy / 1000
    if min(span_x, span_y) < minimum:
        ladder = " -> ".join(f"{domain.run.dx / 1000:g}" for domain in experiment.domains)
        raise ValueError(f"The fitted {span_x:g} x {span_y:g} km root ({ladder} km ladder) "
                         f"is smaller than this research question's "
                         f"{minimum:g} km minimum span. {retry_hint}")
    return experiment, {"status": "passed-cpu-estimate", "binding_phase": phases.binding_phase,
                        "peak_envelope_bytes": phases.peak_envelope_bytes,
                        "envelope_budget_bytes": budget,
                        "remaining_envelope_bytes": budget - phases.peak_envelope_bytes,
                        "forecast_started": False, "prepared_inputs_validated": False,
                        "execution_qualification": qualification_status(recipe),
                        # Historical receipt key retained for existing consumers.
                        "science_validation": _qualification_fields(recipe)["validation_status"]}


def _publish_bundle(stage: Path, destination: Path, *, exp, source) -> list[Path]:
    """Create companions exclusively, then commit the TOML; roll back our files.

    `publish_new` is atomic create-only publication on the same filesystem. An
    existing companion, including a dangling symlink, is always a refusal.

    ``exp`` is the experiment the staged configuration resolves to and
    ``source`` its own ``[fetch].source``, both required rather than
    optional: what a configuration must carry beside it belongs to the
    ROUTE it will run on, and a publisher that could omit the question
    would publish a configuration its route refuses before it starts.
    Every file the route reads is rendered HERE, from the staged
    configuration's own experiment, so a bundle whose TOML was edited
    after the emission door wrote into this stage publishes namelists
    that match the TOML beside them rather than the ones the emission
    door rendered from an earlier draft.
    """
    from woof.hrrr_route_inputs import candidate_companions, route_input_paths

    roles = {role: path.name for role, path in route_input_paths(destination).items()}
    wps = stage / roles["wps_namelist"]
    if not wps.is_file():
        raise ValueError(
            f"This configuration's input route reads {roles['wps_namelist']} "
            f"beside {destination.name}, and the staged bundle has none. "
            "Create it again with a source whose emission writes one")
    for target, content in candidate_companions(
            destination, exp, wps_text=wps.read_text(encoding="utf-8"),
            source=source):
        (stage / target.name).write_text(
            content, encoding="utf-8", newline="\n")
    files = sorted(stage.iterdir(), key=lambda path: (path.name == destination.name, path.name))
    if any(not path.is_file() or path.is_symlink() for path in files):
        raise ValueError("Research staging produced an unexpected non-file companion")
    targets = [destination.parent / path.name for path in files]
    for target in targets:
        if os.path.lexists(target):
            raise FileExistsError(f"Research creation preserves existing files: {target}")
    created = []
    try:
        for staged, target in zip(files, targets):
            publish_new(staged, target)
            stat = target.stat()
            created.append((target, (stat.st_dev, stat.st_ino)))
    except BaseException:
        for target, identity in reversed(created):
            try:
                stat = target.lstat()
                if (stat.st_dev, stat.st_ino) == identity:
                    target.unlink()
            except FileNotFoundError:
                pass
        raise
    return targets


def _geometry_review(experiment) -> dict:
    """Describe fitted coverage using the runner's native placement check."""
    from dataclasses import replace
    from types import SimpleNamespace
    from woof.core.nest_relocation import _prevalidate_placement

    spans = [{"grid_id": domain.grid_id, "parent_id": domain.parent_id,
              "span_x_km": domain.run.nx * domain.run.dx / 1000,
              "span_y_km": domain.run.ny * domain.run.dy / 1000}
             for domain in experiment.domains]
    review = {"domain_spans": spans, "coverage_notice": (
        "Fixed domains have finite coverage. A moving feature can leave a fine nest; "
        "review every domain against the expected event path and duration.")}
    relocation = getattr(experiment, "relocation", None)
    if relocation is None or getattr(relocation, "follow", None) is None:
        return review
    child = next(domain for domain in experiment.domains if domain.grid_id == relocation.grid_id)
    parent = next(domain for domain in experiment.domains if domain.grid_id == child.parent_id)
    parent_node = SimpleNamespace(cfg=parent)
    moving = {"grid_id": child.grid_id, "parent_id": parent.grid_id,
              "span_x_km": child.run.nx * child.run.dx / 1000,
              "span_y_km": child.run.ny * child.run.dy / 1000,
              "parent_span_x_km": parent.run.nx * parent.run.dx / 1000,
              "parent_span_y_km": parent.run.ny * parent.run.dy / 1000,
              "centroid_radius_km": relocation.follow.radius_km,
              "centroid_notice": "Configurable centroid radius; suitability for a particular cell or vortex is unvalidated.",
              "cadence_seconds": relocation.cadence_seconds,
              "track_path": relocation.track.path if relocation.track is not None else None,
              "prepared_statics": "not yet prepared"}
    review["moving_nest"] = moving
    review["coverage_notice"] = (
        "Following is bounded by the immediate parent. The native runner clamps or holds at its "
        "admissible band; these distances do not guarantee following for the whole forecast. "
        "Prepare the moving child's complete statics corridor before running.")
    try:
        _prevalidate_placement(child, parent_node)
    except ValueError as error:
        moving["placement_status"] = "not_admissible"
        moving["placement_error"] = str(error)
        return review
    starts, travel = {}, {}
    for axis, key, extent, spacing in (
            ("x", "i_parent_start", parent.run.nx, parent.run.dx),
            ("y", "j_parent_start", parent.run.ny, parent.run.dy)):
        current = int(getattr(child, key))
        def valid(position):
            try:
                _prevalidate_placement(replace(child, **{key: position}), parent_node)
            except ValueError:
                return False
            return True
        low, high = 1, current
        while low < high:
            middle = (low + high) // 2
            if valid(middle):
                high = middle
            else:
                low = middle + 1
        minimum = low
        low, high = current, int(extent)
        while low < high:
            middle = (low + high + 1) // 2
            if valid(middle):
                low = middle
            else:
                high = middle - 1
        starts[key] = [minimum, low]
        travel[f"{axis}_km"] = [(minimum - current) * spacing / 1000,
                               (low - current) * spacing / 1000]
    moving.update(placement_status="native_stencil_checked",
                  admissible_starts_parent_cells=starts,
                  travel_from_initial_grid_axes=travel)
    return review


def create_workspace(args) -> dict:
    from woof import domain_wizard as wizard
    catalog = catalog_document()
    recipe = _recipe(args.configuration_id, document=catalog)
    hardware = {"profiles": catalog["hardware_profiles"], "policy": catalog["hardware_policy"]}
    if recipe["requires_existing_state"] and recipe["method"] != "controlled_scenario":
        if recipe["method"] == "archived_downscale":
            raise ValueError("This archived-downscale recipe requires actual parent history, restart "
                             "physics evidence and child surface state. Open the native Downscale "
                             "guide, or use woof downscale; research create cannot invent a parent.")
        raise ValueError("This research recipe requires a real existing scenario state. Open the "
                         "existing configuration for review; research create cannot substitute "
                         "an ordinary GFS analysis for the required state.")
    # An unknown source is refused here, before any probe.  No known one
    # is refused for a controlled scenario: the recipe is always a domain
    # tree (validate_recipe), and every source's tree preparation defers
    # its bubbles to the tree runner that applies them
    # (woof.experiment.deferred_initial_perturbation).
    wizard.resolve_source(args.source)
    destination = args.out.absolute()
    if destination.suffix.lower() != ".toml":
        raise ValueError("Research output must be a new .toml path")
    if os.path.lexists(destination):
        raise FileExistsError(f"Research creation preserves existing files: {destination}")
    for companion in (destination.with_suffix(".namelist.wps"),
                      destination.with_name(destination.name + ".arwen-plots.json"),
                      destination.with_name(destination.name + ".arwen-research.json")):
        if os.path.lexists(companion):
            raise FileExistsError(f"Research creation preserves existing companion: {companion}")
    sizing = wizard.resolve_sizing_budget(None, args.vram_gib, declare=("--vram-gib",))
    budget_document = _budget_document(sizing, args.hardware_class, hardware=hardware)
    profile = effective_profile(recipe, budget_document["selected_class"], hardware=hardware)
    if args.point is not None:
        lat, lon = wizard._parse_point(args.point)
    else:
        footprint = wizard.load_polygon_footprint(args.polygon)
        lat, lon = footprint.center_lat, footprint.center_lon
    destination.parent.mkdir(parents=True, exist_ok=True)
    log = io.StringIO()
    with tempfile.TemporaryDirectory(prefix=".arwen-research-", dir=destination.parent) as temporary:
        stage = Path(temporary)
        staged = stage / destination.name
        native, data_dir = _native_args(args, recipe, staged, sizing, profile=profile)
        with redirect_stdout(log), redirect_stderr(log):
            result = wizard.domain_main(native, sizing_budget=sizing)
        if result:
            raise ValueError(f"Native domain creation failed (exit {result}):\n{log.getvalue()}")
        native_text = staged.read_text(encoding="utf-8")
        if native.source == "era5":
            # Each bundle owns its Vtable: separate studies can share a folder
            # without colliding with or changing somebody else's common Vtable.
            vtable = stage / wizard._PACKAGED_VTABLE.name
            if vtable.exists():
                unique_vtable = stage / (destination.stem + "." + vtable.name)
                vtable.rename(unique_vtable)
                native_text = native_text.replace(
                    'vtable = ' + json.dumps(vtable.name),
                    'vtable = ' + json.dumps(unique_vtable.name))
        native_text = native_text.replace(str(stage), str(destination.parent)).replace(
            stage.as_posix(), destination.parent.as_posix())
        text = _final_text(native_text, recipe,
                           lat=lat, lon=lon, data_dir=data_dir, profile=profile)
        try:
            experiment, admission = _admission(text, recipe=recipe, source=native.source,
                                              sizing=sizing, path=destination)
        except MemoryAdmissionError as error:
            retain_final_candidate(error, text=text, requested_path=destination, stage=stage,
                metadata={"configuration_id": recipe["id"], "source": native.source,
                          "catalog_sha256": catalog["catalog_sha256"]})
            raise
        staged.write_text(text, encoding="utf-8", newline="\n")
        domains = [{"grid_id": domain.grid_id, "parent_id": domain.parent_id,
                    "nx": domain.run.nx, "ny": domain.run.ny, "nz": domain.run.nz,
                    "dx_km": domain.run.dx / 1000,
                    "span_x_km": domain.run.nx * domain.run.dx / 1000,
                    "span_y_km": domain.run.ny * domain.run.dy / 1000,
                    "history_interval_s": domain.history_interval_s,
                    "time_step_s": domain.run.dt} for domain in experiment.domains]
        receipt = {"schema": "arwen.research.workspace.v1",
                   "created_utc": datetime.now(timezone.utc).isoformat(),
                   "configuration": str(destination), "recipe": recipe,
                   "catalog_sha256": catalog["catalog_sha256"],
                   "diagnostic_capabilities_sha256": catalog["diagnostic_capabilities"]["sha256"],
                   "config_sha256": hashlib.sha256(staged.read_bytes()).hexdigest(),
                   "hardware": budget_document, "source": native.source,
                   "effective_profile": profile,
                   "hardware_policy_sha256": catalog["hardware_policy_sha256"],
                   "start_time": experiment.start_time.isoformat(),
                   "run_seconds": experiment.run_seconds, "domains": domains,
                   "requested_overrides": {"hours": args.hours, "nz": args.nz,
                                           "physics_profile": args.physics_profile,
                                           "tiles": args.tiles},
                   "admission": admission,
                   "geometry_review": _geometry_review(experiment),
                   "required_next_steps": ["Review actual geometry and scientific settings",
                                           "Acquire and prepare forcing and statics",
                                           "Check actual free memory again before launch"],
                   "native_creation_log": log.getvalue().replace(
                       str(stage), str(destination.parent)).replace(
                       stage.as_posix(), destination.parent.as_posix())}
        if recipe["tracker"]:
            receipt["required_next_steps"].append("Prepare the moving child's complete --statics-corridor")
        if recipe["scenario"]:
            receipt["required_next_steps"].append("Validate warm-bubble coverage during native preparation")
        receipt["files"] = [str(destination.parent / name) for name in sorted(
            [path.name for path in stage.iterdir()]
            + [destination.name + ".arwen-plots.json", destination.name + ".arwen-research.json"])]
        plots = {"schema": "gpuwm-tui-plots-v1", "label": recipe["title"],
                 "products": ",".join(recipe["diagnostics"])}
        (stage / (destination.name + ".arwen-plots.json")).write_text(
            json.dumps(plots, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        (stage / (destination.name + ".arwen-research.json")).write_text(
            json.dumps(receipt, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
        _publish_bundle(stage, destination, exp=experiment,
                        source=(tomllib.loads(text).get("fetch") or {}).get("source"))
    return receipt


def _research_main(args) -> int:
    if args.research_command == "catalog":
        document = catalog_document()
    elif args.research_command == "hardware":
        document = hardware_document(hardware_class=args.hardware_class, vram_gib=args.vram_gib)
    elif args.research_command == "attributes":
        document = attributes_document()
    else:
        document = create_workspace(args)
    if args.json:
        print(json.dumps(document, ensure_ascii=False, default=str))
    elif args.research_command == "create":
        hardware = document["hardware"]
        print(f"Created {document['configuration']}")
        print(f"Research: {document['recipe']['title']} ({document['recipe']['method']})")
        for note in document["recipe"].get("further_analysis", []):
            print(f"Additional analysis required: {note}")
        print(f"Hardware class {hardware['selected_class']}; {hardware['basis']}; "
              f"{hardware['free_bytes'] / 2**30:.2f} GiB available")
        print("Domains: " + "; ".join(f"d{domain['grid_id']:02d} {domain['nx']}x{domain['ny']}x{domain['nz']} "
                                      f"at {domain['dx_km']:g} km" for domain in document["domains"]))
        geometry = document["geometry_review"]
        print("Coverage: " + "; ".join(
            f"d{domain['grid_id']:02d} {domain['span_x_km']:g} x {domain['span_y_km']:g} km"
            for domain in geometry["domain_spans"]))
        if moving := geometry.get("moving_nest"):
            print(f"Following d{moving['grid_id']:02d} ({moving['span_x_km']:g} x {moving['span_y_km']:g} km) "
                  f"inside d{moving['parent_id']:02d} ({moving['parent_span_x_km']:g} x {moving['parent_span_y_km']:g} km)")
            if travel := moving.get("travel_from_initial_grid_axes"):
                print(f"Native placement band from initial position (grid axes): "
                      f"x {travel['x_km'][0]:g} to {travel['x_km'][1]:g} km; "
                      f"y {travel['y_km'][0]:g} to {travel['y_km'][1]:g} km")
            else:
                print(f"Movement band unavailable: {moving.get('placement_error', 'not established')}")
            print(f"Centroid radius: {moving['centroid_radius_km']:g} km. {moving['centroid_notice']}")
            print(f"Forecast track output: {moving['track_path']} under the run output directory, at tracker consultations.")
        print(geometry["coverage_notice"])
        print("Native CPU geometry/memory admission passed. Review, prepare and check before launching.")
        for step in document["required_next_steps"]:
            print(f"  {step}")
    else:
        print(json.dumps(document, indent=2, ensure_ascii=False, default=str))
    return 0


def research_main(args) -> int:
    try:
        return _research_main(args)
    except OSError as error:
        target = error.filename or getattr(args, "out", CATALOG_PATH)
        action = ("Choose a new output filename; the existing file and its companion files are preserved."
                  if isinstance(error, FileExistsError) else
                  "Check that the named path is available and writable, then retry.")
        raise ValueError(f"Research {args.research_command} could not use {target}: {error}. {action}") from error


def register_cli(subparsers) -> None:
    parser = subparsers.add_parser("research", help="browse and create reviewed research workspaces",
                                  description="Choose a research recipe, review its hardware profile, and create a new native configuration. Creation does not prepare or run a forecast.")
    commands = parser.add_subparsers(dest="research_command", required=True)
    catalog = commands.add_parser("catalog", help="show the packaged research catalog")
    catalog.add_argument("--json", action="store_true", help="emit the complete native catalog and hardware profiles as compact JSON")
    catalog.set_defaults(func=research_main)
    attributes = commands.add_parser("attributes", help="show the native attribute-following registry and scientific units")
    attributes.add_argument("--json", action="store_true", help="emit supported attributes, units, extrema, reductions and model-level semantics as compact JSON")
    attributes.set_defaults(func=research_main)
    hardware = commands.add_parser("hardware", help="measure hardware or estimate an explicit capacity")
    hardware.add_argument("--json", action="store_true", help="emit capacity, free-memory budget and selected resource profile as compact JSON")
    hardware.set_defaults(func=research_main)
    create = commands.add_parser("create", help="create a new native configuration and review receipt")
    create.add_argument("configuration_id", help="exact configuration id from research catalog; existing-state recipes must continue from their supplied archive or scenario")
    target = create.add_mutually_exclusive_group(required=True)
    target.add_argument("--point", metavar="LAT,LON", help="centre latitude and longitude; native sizing fits coverage to the actual budget and enforces the recipe's minimum span")
    target.add_argument("--polygon", type=Path, metavar="GEOJSON", help="GeoJSON study footprint; retain its full required coverage")
    create.add_argument("--source", default="gfs", help="native input source id (default gfs); source, cycle and research method must be compatible")
    create.add_argument("--cycle", required=True, help="input cycle in UTC, YYYY-MM-DDTHH; latest uses native source discovery")
    create.add_argument("--hours", type=int, help="explicit whole-hour study duration; omitted keeps the recipe's duration")
    create.add_argument("--out", type=Path, required=True, help="new experiment TOML path; existing configuration or companion files are never replaced")
    create.add_argument("--name", help="descriptive name for this study; omitted keeps the recipe title")
    create.add_argument("--physics-profile", help="native physics suite override; omitted keeps the admitted source default")
    create.add_argument("--nz", type=int, help="explicit vertical-level override")
    create.add_argument("--tiles", choices=("off", "auto", "on"), help="explicit native tile-streaming mode; host-memory and transfer costs remain subject to admission")
    create.add_argument("--ack", action="append", default=[], help="native source acknowledgement code; repeat for each required acknowledgement")
    create.add_argument("--json", action="store_true", help="emit the created workspace receipt, geometry, resource budget and required next steps as compact JSON")
    create.set_defaults(func=research_main)
    for command in (hardware, create):
        command.add_argument("--hardware-class", choices=("auto", *HARDWARE_CLASSES), default="auto",
                             help="resource profile; does not declare available memory")
        command.add_argument("--vram-gib", type=float,
                             help="explicit capacity estimate; omit to measure actual total/free memory")
    from woof.explain import add_explain_flag
    for command in (catalog, attributes, hardware, create):
        add_explain_flag(command, nested=True)
