"""The namelist contract: what ``woof import-namelist`` accepts, as a table.

    python -m woof.namelist_contract --out namelist-contract.json

A site that takes WRF namelists (WOOF) should tell a user about every
setting the engine will refuse BEFORE a GPU box boots, and it should do so
from the engine's own rules, not a hand-kept copy that drifts.  This module
writes those rules out as one JSON table, generated from the engine that
will run the namelist:

* **the vocabulary** -- every key the importer reads, per namelist group.
  It is not transcribed: the importer is run over a set of valid baseline
  namelists with every key WRF v4.7.1 declares added at its Registry
  default (woof/data/wrf_namelist), and every key the translation asks a
  group for is recorded (:data:`woof.namelist_import._TAKE_TRACE`).  A key
  outside it is refused by the importer as unmapped.
* **the value rules** -- the importer's own single-key tables
  (:data:`woof.namelist_import.PHYSICS_PINS` and its neighbours, the scheme
  maps, the Noah-MP and MYNN option identities): the values each key admits
  on every domain, and why any other is refused.  A key whose value is a
  ``+``-joined token list (``geog_data_res``) carries ``tokens`` and
  ``token_separator`` instead of ``values``: each token must be listed.
* **the groups** -- which namelist groups each file may carry, which groups
  are read whole (every key recorded and dropped), and the numbered
  auxiliary-stream key pattern.

What the table does NOT carry is a rule that depends on more than one key
(scheme pairings, nest geometry, cadence divisibility): the site cannot
evaluate those faithfully from a table, and the importer on the box stays
the authority for them.  A pair the table passes can still be refused
there; a pair the table refuses would be refused there too.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from copy import deepcopy
from pathlib import Path

SCHEMA = "arwen.namelist-contract.v1"

_WPS_GROUPS = ("share", "geogrid", "ungrib", "metgrid")

# ---------------------------------------------------------------------------
# Baselines: valid namelist pairs, one per branch of the translation that
# asks for keys the others do not.
# ---------------------------------------------------------------------------

_ETA = [1.0, 0.99685, 0.99353, 0.99003, 0.98636, 0.98249, 0.97843, 0.97415,
        0.96965, 0.96492, 0.95995, 0.95473, 0.94924, 0.94347, 0.93742,
        0.93106, 0.92438, 0.91738, 0.91003, 0.90233, 0.89425, 0.88579,
        0.87692, 0.86763, 0.85791, 0.84774, 0.83711, 0.82599, 0.81438,
        0.80151, 0.78728, 0.77154, 0.75419, 0.73508, 0.71409, 0.69108,
        0.66594, 0.63853, 0.60878, 0.5766, 0.54196, 0.50485, 0.46535,
        0.42357, 0.38139, 0.33892, 0.29644, 0.25429, 0.21281, 0.17241,
        0.1335, 0.09805, 0.06777, 0.04191, 0.01982, 0.00096, 0.0]


def _base_pair() -> tuple[dict, dict]:
    """A valid two-domain pair: 3 km root, 1 km nest, Thompson/YSU/Noah."""
    wps = {
        "share": {"wrf_core": ["ARW"], "max_dom": [2],
                  "start_date": ["2026-09-30_12:00:00"] * 2,
                  "end_date": ["2026-09-30_18:00:00"] * 2,
                  "interval_seconds": [10800]},
        "geogrid": {"parent_id": [1, 1], "parent_grid_ratio": [1, 3],
                    "i_parent_start": [1, 20], "j_parent_start": [1, 20],
                    "e_we": [61, 61], "e_sn": [61, 61],
                    "geog_data_res": ["default", "default"],
                    "dx": [3000], "dy": [3000], "map_proj": ["lambert"],
                    "ref_lat": [39.35], "ref_lon": [-106.05],
                    "truelat1": [38.0], "truelat2": [41.0],
                    "stand_lon": [-106.05]},
    }
    inp = {
        "time_control": {
            "run_days": [0], "run_hours": [6], "run_minutes": [0],
            "run_seconds": [0],
            "start_year": [2026, 2026], "start_month": [9, 9],
            "start_day": [30, 30], "start_hour": [12, 12],
            "end_year": [2026, 2026], "end_month": [9, 9],
            "end_day": [30, 30], "end_hour": [18, 18],
            "interval_seconds": [10800],
            "input_from_file": [True, True],
            "history_interval": [60, 60]},
        "domains": {
            "time_step": [15], "max_dom": [2], "e_we": [61, 61],
            "e_sn": [61, 61], "e_vert": [57, 57],
            "p_top_requested": [10000], "eta_levels": list(_ETA),
            "dx": [3000, 1000], "dy": [3000, 1000], "grid_id": [1, 2],
            "parent_id": [0, 1], "i_parent_start": [1, 20],
            "j_parent_start": [1, 20], "parent_grid_ratio": [1, 3],
            "parent_time_step_ratio": [1, 3], "feedback": [0]},
        "physics": {
            "mp_physics": [8, 8], "ra_lw_physics": [4, 4],
            "ra_sw_physics": [4, 4], "radt": [3, 3],
            "sf_sfclay_physics": [1, 1], "sf_surface_physics": [2, 2],
            "bl_pbl_physics": [1, 1], "cu_physics": [0, 0],
            "num_soil_layers": [4], "num_land_cat": [21]},
        "dynamics": {
            "w_damping": [1], "diff_opt": [2, 2],
            "mix_full_fields": [True, True], "km_opt": [4, 4],
            "diff_6th_opt": [2, 2], "diff_6th_factor": [0.12, 0.12],
            "damp_opt": [3], "zdamp": [4000.0, 4000.0],
            "dampcoef": [0.2, 0.2], "non_hydrostatic": [True, True],
            "moist_adv_opt": [1, 1], "scalar_adv_opt": [1, 1]},
        "bdy_control": {"spec_bdy_width": [5], "specified": [True]},
    }
    return wps, inp


def _variant(physics=None, dynamics=None, options=None, edits=None):
    """A baseline's edits of the base pair: ``edits`` maps (group, key) to
    a column, or to None to leave the key out."""
    return {"physics": physics or {}, "dynamics": dynamics or {},
            "options": options or {}, "edits": edits or {}}


#: (label, edits) -- each a branch the base pair does not take.  A variant
#: that does not import cleanly on this engine is skipped and named in the
#: table's ``baselines`` record, so a scheme that leaves the engine takes
#: its keys with it rather than breaking the generator.
BASELINES = (
    ("rrtmg-rte-rrtmgp", _variant()),
    ("rrtmg-legacy", _variant(options={"rrtmg_variant": "rrtmg_legacy"})),
    ("rrtm-dudhia", _variant(physics={"ra_lw_physics": [1, 1],
                                      "ra_sw_physics": [1, 1]})),
    ("kain-fritsch", _variant(physics={"cu_physics": [1, 1],
                                       "cudt": [0, 0]})),
    ("grell-freitas", _variant(physics={"cu_physics": [3, 3]})),
    ("new-tiedtke", _variant(physics={"cu_physics": [16, 16]})),
    ("new-sas", _variant(physics={"cu_physics": [96, 96]})),
    ("thompson-aerosol", _variant(physics={"mp_physics": [28, 28]})),
    ("nssl", _variant(physics={"mp_physics": [18, 18]})),
    ("morrison", _variant(physics={"mp_physics": [10, 10]})),
    ("mynn", _variant(physics={"bl_pbl_physics": [5, 5],
                               "sf_sfclay_physics": [5, 5]})),
    ("myj", _variant(physics={"bl_pbl_physics": [2, 2],
                              "sf_sfclay_physics": [2, 2]})),
    ("noah-mp", _variant(physics={"sf_surface_physics": [4, 4]})),
    ("ruc", _variant(physics={"sf_surface_physics": [3, 3],
                              "num_soil_layers": [9]})),
    ("tke-km2", _variant(physics={"bl_pbl_physics": [0, 0]},
                         dynamics={"km_opt": [2, 2]})),
    # Spellings the base pair does not use, so a key another spelling can
    # stand in for is not measured as required.
    ("mercator", _variant(edits={("geogrid", "map_proj"): ["mercator"],
                                 ("geogrid", "truelat1"): [39.0],
                                 ("geogrid", "truelat2"): None})),
    ("polar", _variant(edits={("geogrid", "map_proj"): ["polar"],
                              ("geogrid", "truelat1"): [60.0],
                              ("geogrid", "truelat2"): None})),
    ("history-in-minutes", _variant(edits={
        ("time_control", "history_interval"): None,
        ("time_control", "history_interval_m"): [60, 60]})),
    ("run-length-only", _variant(edits={
        ("time_control", key): None
        for key in ("end_year", "end_month", "end_day", "end_hour")})),
    ("dates-only", _variant(edits={
        ("time_control", key): None
        for key in ("run_days", "run_hours", "run_minutes",
                    "run_seconds")})),
)


def _apply(pair, variant):
    wps, inp = deepcopy(pair[0]), deepcopy(pair[1])
    for section in ("physics", "dynamics"):
        inp[section].update(deepcopy(variant[section]))
    for (section, key), column in variant["edits"].items():
        document = wps if section in _WPS_GROUPS else inp
        entries = document.setdefault(section, {})
        if column is None:
            entries.pop(key, None)
        else:
            entries[key] = list(column)
    return wps, inp


# ---------------------------------------------------------------------------
# WRF Registry defaults as parsed namelist values
# ---------------------------------------------------------------------------

def _registry_value(row: dict):
    raw = str(row["default"]).strip()
    kind = row["type"]
    if kind == "logical":
        return raw.lower() in (".true.", "t", ".t.", "true")
    if kind == "integer":
        try:
            return int(raw)
        except ValueError:
            return int(float(raw.replace("d", "e").replace("D", "e")))
    if kind == "real":
        return float(raw.replace("d", "e").replace("D", "e").rstrip("."))\
            if raw.rstrip(".") else 0.0
    return raw.strip("'\"")


def _registry_column(row: dict, max_dom: int) -> list:
    value = _registry_value(row)
    return [value] * (max_dom if row["nentries"] == "max_domains" else 1)


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------

def _probe_vocabulary() -> tuple[dict[str, set[str]], list[dict]]:
    """Every key each group is asked for, over every clean baseline."""
    from woof import namelist_import as ni
    from woof.wrf_namelist_registry import wrf_namelist_keys

    registry = wrf_namelist_keys()
    asked: dict[str, set[str]] = {}
    records = []
    base = _base_pair()
    for label, variant in BASELINES:
        wps, inp = _apply(base, variant)
        options = dict(variant["options"])
        if ni.namelist_refusals(wps, inp, **options):
            records.append({"baseline": label, "used": False,
                            "reason": "does not import on this engine"})
            continue
        # The baseline alone, then every WRF key of each group added at
        # its Registry default.  Keys the baseline sets keep its values.
        trace: set = set()
        token = ni._TAKE_TRACE.set(trace)
        try:
            ni.namelist_refusals(wps, inp, **options)
            for section in ni.INPUT_SECTIONS:
                probe = deepcopy(inp)
                entries = probe.setdefault(section, {})
                for (group, key), row in registry.items():
                    if group == section and key not in entries:
                        entries[key] = _registry_column(row, 2)
                ni.namelist_refusals(wps, probe, **options)
        finally:
            ni._TAKE_TRACE.reset(token)
        for section, key in trace:
            asked.setdefault(section, set()).add(key)
        records.append({"baseline": label, "used": True})
    return asked, records


def _value_rules() -> dict[tuple[str, str], dict]:
    """The importer's single-key value tables, as contract rows."""
    from woof import namelist_import as ni
    from woof.config import (MYNN_PBL_OPTION_IDENTITY,
                              NOAHMP_OPTION_IDENTITY_EVIDENCE,
                              NOAHMP_OPTIONS_WITHOUT_CONSUMER)
    from woof.core.nssl2_contract import DEPRECATED_MP_PHYSICS_FLAGS

    rules: dict[tuple[str, str], dict] = {}

    def rule(section, key, values, why, *, scalar=False):
        rules[(section, key)] = {"values": list(values), "scalar": scalar,
                                 "why": why}

    maps = (
        ("mp_physics", sorted(set(ni._MP_MAP)
                              | set(DEPRECATED_MP_PHYSICS_FLAGS))),
        ("bl_pbl_physics", sorted(ni._BL_MAP)),
        ("ra_lw_physics", sorted(ni._RA_LW_MAP)),
        ("ra_sw_physics", sorted(ni._RA_SW_MAP)),
        ("sf_sfclay_physics", sorted(ni._SFCLAY_ALLOWED)),
        ("sf_surface_physics", sorted(ni._SFSFC_ALLOWED)),
        ("cu_physics", sorted(ni._CU_ALLOWED)),
    )
    from woof.config import unported_scheme_reasons
    for key, values in maps:
        rule("physics", key, values,
             f"no ratified woof mapping (implemented: {values}).")
        # The WRF values that are refused by name, with the sentence the
        # importer prints for each, so a site can say why before a box boots.
        rules[("physics", key)]["unported"] = {
            str(value): reason for value, reason
            in sorted(unported_scheme_reasons(key).items())}
    rule("physics", "sf_ocean_physics", [0, 1, 2],
         "must be 0 (off), 1 (Pollard) or 2 (3DPWP)", scalar=True)
    rule("physics", "sf_lake_physics", [0, 1],
         "must be integer 0 (off) or 1 (CLM lake).")
    rule("physics", "use_lakedepth", [0, 1],
         "must be integer 0 (constant depth) or 1 (input bathymetry).")
    for key in ("mosaic_lu", "mosaic_soil"):
        rule("physics", key, [0, 1],
             "must be 0 or 1; nonzero requires RUC land surface.", scalar=True)
    for key, pin, why in ni.PHYSICS_PINS:
        rule("physics", key, [pin], f"woof implements {key} = {pin} only "
                                    f"({why}).")
    for key, pin, why in ni.DYNAMICS_PINS:
        rule("dynamics", key, [pin], f"woof implements {key} = {pin} only "
                                     f"({why}).")
    for key, (value, why) in ni.DYNAMICS_REQUIRED_VALUES.items():
        rule("dynamics", key, [value], why)
    rule("dynamics", "diff_opt", [1, 2],
         "1 selects model-coordinate diffusion for km_opt=2/4; "
         "2 selects the metric stress/scalar form.")
    rule("dynamics", "mix_full_fields", [False, True],
         "must contain Fortran logicals; the coordinate operator and named "
         "source requests import supplied values as declared. Generic metric-operator "
         "import retains its established full-field resolution.")
    for key, (allowed, why) in ni.PHYSICS_CHOICES.items():
        rule("physics", key, allowed, why, scalar=True)
    for key, (ok, why) in ni.NEST_GUARDS.items():
        rule("domains", key, [ok], why, scalar=True)
    for key in ("slope_rad", "topo_shading"):
        rule("physics", key, [0, 1],
             "must be 0 or 1 on every domain (WRF Registry.EM_COMMON: 1 "
             "turns it on).")
    # WRF's sub-grid terrain drag (woof.core.terrain_drag): gwd_opt is read
    # from &dynamics (WRF v4) or &physics (its v3 placement).
    from woof.config import GWD_OPT_VALUES, TOPO_WIND_VALUES
    rule("physics", "topo_wind", list(TOPO_WIND_VALUES),
         "must be 0, 1 or 2 on every domain (WRF v4.7.1: 1 Jimenez-Dudhia, "
         "2 the VAR form).")
    for section in ("dynamics", "physics"):
        rule(section, "gwd_opt", list(GWD_OPT_VALUES),
             "must be 0, 1 or 3 on every domain (WRF v4.7.1: 1 the KIM drag, "
             "3 the GSL drag suite; 2 is not ported).")
    for key, admitted in MYNN_PBL_OPTION_IDENTITY.items():
        rule("physics", key, [admitted],
             f"woof implements {key} = {admitted!r} only (MYNN option "
             "identity); no nearby branch is substituted for an unported "
             "one.")
    for key, (admitted, evidence) in NOAHMP_OPTION_IDENTITY_EVIDENCE.items():
        if key in NOAHMP_OPTIONS_WITHOUT_CONSUMER:
            continue
        rule("noah_mp", key, [admitted],
             f"woof's Noah-MP port implements {key} = {admitted!r} only "
             f"({evidence}).")
    rule("physics", "num_land_cat", [20, 21, 61],
         "the MODIFIED_IGBP_MODIS_NOAH category count the static build "
         "stamps: 21, 20 where geog_data_res selects the 15 arc-second "
         "MODIS land use without its lake class (modis_15s), "
         "or 61 where geog_data_res names a Local Climate Zone "
         "land cover (cglc_modis_lcz) and the urban canopy runs with "
         "use_wudapt_lcz = 1 (the importer checks that pairing).")
    # geog_data_res is a '+'-joined token list per domain, so its rule is
    # on tokens, not whole values: each token must be one woof builds.
    tokens = ni.geog_data_res_tokens()
    rules[("geogrid", "geog_data_res")] = {
        "values": None, "scalar": False,
        "tokens": sorted(tokens), "token_separator": "+",
        "why": ("each '+'-separated token must be one woof builds: "
                + ", ".join(sorted(t for t, how in tokens.items()
                                   if how == "geog"))
                + " from the GEOG tree, "
                + ", ".join(sorted(t for t, how in tokens.items()
                                   if how != "geog"))
                + " as the [static.highres] land cover (which also "
                  "replaces terrain and soil); any other names a dataset "
                  "woof does not build.")}
    rule("fdda", "grid_fdda", [0, 1, 2],
         "0 disables nudging; 1 selects grid analysis nudging; 2 selects spectral nudging. Active modes require a positive analysis interval and complete prepared analyses.")
    for key in ("grid_sfdda", "obs_nudge_opt"):
        rule("fdda", key, [0], ni.FDDA_OBSERVATION_INPUT)
    rule("geogrid", "map_proj", ["lambert", "mercator", "polar"],
         "implemented projections: 'lambert', 'mercator', 'polar'.",
         scalar=True)
    rule("share", "wrf_core", ["ARW"], "only ARW is supported.",
         scalar=True)
    return rules


# ---------------------------------------------------------------------------
# Every value rule measured against the importer
# ---------------------------------------------------------------------------



def _kind(values: list) -> str:
    first = values[0]
    if isinstance(first, bool):
        return "bool"
    if isinstance(first, str):
        return "str"
    if isinstance(first, float):
        return "float"
    return "int"


def _outside(values: list):
    """A value the rule does not admit, or an invalid type when both
    possible logical values are admitted."""
    kind = _kind(values)
    if kind == "bool":
        opposite = not values[0]
        return opposite if opposite not in values else 0
    if kind == "str":
        return "unsupported_" + str(values[0]).lower()
    numbers = [float(v) for v in values]
    bad = max(numbers) + 1.0
    while bad in numbers:
        bad += 1.0
    return bad if kind == "float" else int(bad)


def _refused(problems: list, section: str, key: str) -> bool:
    import re

    for problem in problems:
        keys = getattr(problem, "keys", None)
        if keys:
            if (section, key) in keys:
                return True
        elif re.search(rf"\b{re.escape(key)}\b", str(problem)):
            return True
    return False


def _with_column(pair, section, key, column):
    wps, inp = deepcopy(pair[0]), deepcopy(pair[1])
    document = wps if section in _WPS_GROUPS else inp
    document.setdefault(section, {})[key] = list(column)
    return wps, inp


def _measure_rules(rules: dict) -> None:
    """Hold every exported rule to the importer, and measure its reach.

    For each rule, on every baseline that imports: the rule's own values
    must import and a value outside them must be refused BY NAME.  A rule
    the importer does not enforce would make the site refuse a namelist
    the engine runs, so the generator stops instead of writing it.  The
    reach -- how many of a column's values the importer reads -- is
    measured, not assumed: ``first`` (a scalar), ``max_dom`` (one per
    domain) or ``all`` (every value written, past max_dom too).
    """
    from woof import namelist_import as ni

    base = _base_pair()
    pairs = []
    for label, variant in BASELINES:
        wps, inp = _apply(base, variant)
        options = dict(variant["options"])
        if not ni.namelist_refusals(wps, inp, **options):
            pairs.append((label, (wps, inp), options))
    failures = []
    for (section, key), rule in sorted(rules.items()):
        if rule.get("tokens") is not None:
            failures += _measure_token_rule(section, key, rule, pairs)
            continue
        values, bad = rule["values"], _outside(rule["values"])
        reach = None
        for label, pair, options in pairs:
            document = pair[0] if section in _WPS_GROUPS else pair[1]
            own = document.get(section, {}).get(key)
            width = 1 if rule["scalar"] else 2

            def refuses(column):
                wps, inp = _with_column(pair, section, key, column)
                return _refused(ni.namelist_refusals(wps, inp, **options),
                                section, key)

            if not refuses([bad] * width):
                failures.append(f"{label}: {section}/{key} = {bad!r} (outside "
                                f"{values}) imports")
                continue
            if reach is not None:
                continue
            # A value the rule admits may still be refused here by a rule
            # between keys (use_mp_re = 0 needs the legacy RRTMG port), which
            # is the importer's to judge on the box: the table only promises
            # that what it refuses, the importer refuses.
            candidates = ([list(own)[:width]] if own else []) + [
                [value] * width for value in values]
            good = next((column for column in candidates
                         if not refuses(column + [column[-1]]
                                        * (width - len(column)))), None)
            if good is None:
                continue
            first = good[0]
            reach = ("all" if refuses([first, first, bad])
                     else "max_dom" if refuses([first, bad])
                     else "first")
        if reach is None:
            failures.append(f"{section}/{key}: no value of {values} imports "
                            "on any baseline")
        rule["reach"] = reach
        rule["kind"] = _kind(values)
    if failures:
        raise RuntimeError(
            "the namelist contract's value rules disagree with the importer "
            "they are exported from:\n  " + "\n  ".join(failures))


def _measure_token_rule(section: str, key: str, rule: dict,
                        pairs: list) -> list[str]:
    """Hold a token rule to the importer, as :func:`_measure_rules` holds a
    value rule: every listed token imports alone on some baseline, a token
    outside the list is refused by name on every baseline, and the reach is
    measured (``max_dom`` when the second domain's value is read)."""
    from woof import namelist_import as ni

    separator = rule["token_separator"]
    bad = "unsupported_dataset"
    failures = []

    def refuses(pair, options, column):
        wps, inp = _with_column(pair, section, key, column)
        return _refused(ni.namelist_refusals(wps, inp, **options),
                        section, key)

    for label, pair, options in pairs:
        if not refuses(pair, options, [bad, bad]):
            failures.append(f"{label}: {section}/{key} token {bad!r} "
                            f"(outside {rule['tokens']}) imports")
    for token in rule["tokens"]:
        if not any(not refuses(pair, options, [token, token])
                   for _, pair, options in pairs):
            failures.append(f"{section}/{key}: token {token!r} imports on "
                            "no baseline")
    reach = None
    for _, pair, options in pairs:
        good = next((token for token in rule["tokens"]
                     if not refuses(pair, options, [token, token])), None)
        if good is None:
            continue
        reach = ("all" if refuses(pair, options, [good, good, bad])
                 else "max_dom" if refuses(pair, options,
                                           [good, separator.join(
                                               [good, bad])])
                 else "first")
        break
    rule["reach"] = reach
    rule["kind"] = "str"
    return failures


def _measure_required() -> set[tuple[str, str]]:
    """The keys the importer refuses a pair for leaving out.

    Measured: each key of the base pair is removed on every baseline, and
    it is required when every baseline is then refused naming it.  A key
    required only under some configurations (a scheme's own inputs) is left
    to the importer on the box.
    """
    from woof import namelist_import as ni

    # Removing a key from one concrete baseline can make its Registry
    # default disagree with another key (a date or WPS grid extent).
    # That is a pair-dependent refusal, not an unconditional requirement.
    defaulted = {(section, key)
                 for (_, section), defaults in ni._NAMELIST_DEFAULTS.items()
                 for key in defaults}
    defaulted.update({("geogrid", "parent_id"), ("geogrid", "truelat2")})
    base = _base_pair()
    pairs = []
    for label, variant in BASELINES:
        wps, inp = _apply(base, variant)
        options = dict(variant["options"])
        if not ni.namelist_refusals(wps, inp, **options):
            pairs.append(((wps, inp), options))
    candidates = sorted(
        {(section, key) for document in base
         for section, entries in document.items() for key in entries}
        - defaulted)
    required = set()
    for section, key in candidates:
        refused_everywhere = True
        for pair, options in pairs:
            wps, inp = deepcopy(pair[0]), deepcopy(pair[1])
            document = wps if section in _WPS_GROUPS else inp
            if key not in document.get(section, {}):
                refused_everywhere = False
                break
            del document[section][key]
            problems = ni.namelist_refusals(wps, inp, **options)
            if not _refused(problems, section, key):
                refused_everywhere = False
                break
        if refused_everywhere:
            required.add((section, key))
    return required


def _engine_identity() -> dict:
    import woof

    root = Path(woof.__file__).resolve().parent.parent
    commit = None
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=20,
            check=True).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        commit = None
    return {"package": "woof", "version": str(woof.__version__),
            "git_commit": commit}


def build_namelist_contract() -> dict:
    """The contract table, generated from the importer that is running."""
    from woof import namelist_import as ni
    from woof.wrf_namelist_registry import (wrf_namelist_keys,
                                             wrf_namelist_table)

    asked, records = _probe_vocabulary()
    rules = _value_rules()
    registry = wrf_namelist_keys()
    from woof.namelist_stochastic import SUPPORTED_SELECTORS
    for key in ni.STOCH_SELECTORS:
        row = registry[("stoch", key)]
        rules[("stoch", key)] = {
            "values": ([0, 1] if key in SUPPORTED_SELECTORS else
                       [False] if row["type"] == "logical" else [0]),
            "scalar": False, "why": ("supported stochastic selector; domain forcing, physics consumer and parameter pairings remain importer checks"
                if key in SUPPORTED_SELECTORS else "selected stochastic consumer is not implemented")}
    _measure_rules(rules)
    required = _measure_required()
    sections = {}
    for file_name, groups, dropped in (
            ("namelist.wps", ni.WPS_SECTIONS, ni.WPS_DROPPED_SECTIONS),
            ("namelist.input", ni.INPUT_SECTIONS, ni.INPUT_DROPPED_SECTIONS)):
        for group in groups:
            keys = {}
            # A group read whole (&fdda) still refuses its value rules.
            ruled = {key for (section, key) in rules if section == group}
            ignored_keys = (set(ni.TIME_CONTROL_IGNORED_KEYS)
                            if group == "time_control" else set())
            for key in sorted(asked.get(group, set()) | ruled | ignored_keys):
                row = {"values": None, "scalar": False, "why": None,
                       "required": (group, key) in required}
                row.update(rules.get((group, key), {}))
                keys[key] = row
            entry = {"file": file_name, "any_key": group in dropped,
                     "key_patterns": [], "keys": keys}
            if group == "time_control":
                entry["key_patterns"].append({
                    "regex": ni.AUX_STREAM_KEY.pattern,
                    "why": "WRF's numbered auxiliary stream keys are "
                           "recorded and ignored after forcing selector checks"})
                entry["key_patterns"].append({
                    "regex": ni.AUX_STREAM_FORMAT_KEY.pattern,
                    "why": "WRF's numbered auxiliary stream format and "
                           "frame controls are recorded and ignored"})
            if group == "stoch":
                # Registered parameters are inert when their consumer is off.
                # Active parameter pairings remain the importer's authority.
                for section, key in registry:
                    if section == group:
                        keys.setdefault(key, {"values": None, "scalar": False,
                                              "why": None, "required": False})
            sections[group] = entry
    # Where WRF declares each key the importer reads somewhere: a key found
    # in the wrong group can then be named with the group to move it to.
    read = {key for entry in sections.values() for key in entry["keys"]}
    declares: dict[str, list[str]] = {}
    for group, key in registry:
        if key in read:
            declares.setdefault(key, []).append(group)
    return {
        "schema": SCHEMA,
        "generated_by": "python -m woof.namelist_contract",
        "engine": _engine_identity(),
        "wrf_registry": dict(wrf_namelist_table()["wrf"]),
        "files": {"namelist.wps": list(ni.WPS_SECTIONS),
                  "namelist.input": list(ni.INPUT_SECTIONS)},
        "sections": sections,
        "wrf_declares": {key: sorted(groups)
                         for key, groups in sorted(declares.items())},
        "baselines": records,
        "scope": ("single-key rules only; pairings, nest geometry and "
                  "cadence are the importer's, on the box"),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m woof.namelist_contract",
        description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True,
                        help="where to write the JSON table")
    args = parser.parse_args(argv)
    contract = build_namelist_contract()
    text = json.dumps(contract, indent=1, sort_keys=False) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8", newline="\n")
    used = sum(1 for row in contract["baselines"] if row["used"])
    total = sum(len(s["keys"]) for s in contract["sections"].values())
    print(f"{total} keys in {len(contract['sections'])} groups from "
          f"{used}/{len(contract['baselines'])} baselines -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
