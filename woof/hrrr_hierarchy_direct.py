"""Sealed HRRR root preparation -> parallel nested stock-WRF inputs."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, replace
from datetime import datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import tomllib
from types import SimpleNamespace

import numpy as np

from woof import runtime_manifest
from woof.aerosol_source_receipt import (
    AEROSOL_SOURCE_KEY,
    aerosol_source_report_entry,
)
from woof.config import radiation_scheme_ids
from woof.experiment import build_experiment, deferred_perturbation_config
from woof.hrrr_forecast import hrrr_forcing_end_hour
from woof.hrrr_native_static import (
    sha256_file,
    verified_static_catalog,
    verify_hrrr_native_static,
)
from woof.ingest.cpu_backend import resolve_cpu_bridge
from woof.ingest.preprocess_backend import preprocess_math_call
from woof.ingest.hrrr import load_hrrr_native_series
from woof.ingest.hrrr_target import load_hrrr_target_domain
from woof.ingest.source_coverage import owns_source_coverage_refusal
from woof.static.terrain_smoothing import selection_carrier_kwargs
from woof.ingest.nest_init import NestedInputCatalog, ParentInitView
from woof.ingest.prepared_cache import (
    PREPARATION_INERT_RUN_FIELDS,
    PreparedCacheReader,
    PreparedHeadReader,
    compare_prepared_domain_config,
    effective_prepared_domain_config,
    prepared_domain_config_identity,
    prepared_cache_identity,
    restore_prepared_cache,
)
from woof.ingest.ruc_soil import preprocess_land_surface_soil
from woof.namelist_import import import_namelists, parse_namelist
from woof.namelist_seal import validated_namelist_extension_invariant
from woof.native_domain_artifacts import (
    _atomic_staging_sibling,
    published_path_refusal,
)
from woof.native_hierarchy import initialize_and_export_native_hierarchy
from woof.native_wrf_contract import validate_native_lambert_contracts
from woof.static.corridor import (
    STATICS_CORRIDOR_DIRNAME,
    catalog_with_corridor_births,
    corridor_birth_statics,
    emit_statics_corridor_set,
    move_statics_corridor_set,
    validated_corridor_selection,
)
from woof.static.lambert import grids_from_projection_config
from woof.core.microphysics_transition import resolve_microphysics_transition
from woof.wrf_physics_inventory import EXPORT_USE_THETA_M


SCHEMA = "gpuwm-native-hrrr-hierarchy-direct-v1"


def publish_hierarchy_forecast_companions(root, *, resolved_experiment,
                                         experiment, wps_namelist,
                                         perturbation_deferral=None):
    """Publish the actual native tree authority before its atomic handoff.

    Prepared arrays without a matching experiment and WPS file cannot reach
    the forecast door. Domain identities, vertical levels and start time are
    round-tripped by the same writer the root preparation uses.
    """
    from woof.experiment_document import publish_experiment_document
    raw = tomllib.loads(resolved_experiment)
    expected = experiment
    if perturbation_deferral is not None:
        perturbation = deferred_perturbation_config(perturbation_deferral)
        raw["perturbation"] = asdict(perturbation)
        expected = replace(experiment, perturbation=perturbation)
    root = Path(root)
    config = publish_experiment_document(root / "experiment.toml", raw, expected)
    wps = root / "namelist.wps"
    if wps.exists():
        raise FileExistsError(f"refusing to replace published WPS authority: {wps}")
    shutil.copyfile(wps_namelist, wps)
    validate_native_lambert_contracts(expected, wps, source_name="native HRRR hierarchy handoff")
    return {"experiment_config": {"path": config.name, "sha256": sha256_file(config)},
            "wps_namelist": {"path": wps.name, "sha256": sha256_file(wps)}}


_DOMAIN_PREPARATION_OVERRIDES = frozenset({
    "cu_physics", "cudt_minutes", "radt", "radt_minutes", "bldt",
    # The Grell-Freitas closure and shallow-arm selectors ride with
    # cu_physics: the importer writes them only on a domain whose
    # cu_physics is 3, so a Grell-Freitas root with clos_choice = 1 has a
    # cumulus-off child holding 0.  Preparation reads neither
    # (woof.ingest.prepared_cache.PREPARATION_INERT_RUN_FIELDS).
    "clos_choice", "ishallow", "cugd_avedx",
    "diff_6th_factor", "epssm", "spec_exp", "mp_physics", "moist",
    "moist_cq", "nest_microphysics_transition",
    # The fork's second filter factor rides with diff_6th_factor beside
    # it; preparation reads neither
    # (woof.ingest.prepared_cache.PREPARATION_INERT_RUN_FIELDS).
    "diff_6th_factor2",
    # Per-domain history cadence.  This is the ladder's whole point -- a
    # 3 km parent written hourly beside a 1 km child written every 15
    # minutes -- and the machinery for it already exists end to end: the
    # experiment loader gives every domain its own history_interval_s and
    # divisibility check (woof/experiment.py), the clock registers a
    # per-domain history alarm (woof/core/clock.py:586), and the tree
    # runner writes per-domain wrfouts on it
    # (woof/prepared_domain_tree_forecast.py, io_modes "history cadence:
    # per-domain experiment TOML").  Only this drift check disagreed, and
    # it did so AFTER the expensive preparation: a wizard-emitted
    # 3600/900 ladder passed the wizard, the root preparer and fetch, then
    # died on "output_interval_s (900.0, 3600.0)".  Preparation itself
    # writes no history frame at all.
    "output_interval_s",
    # The rest of the per-domain history cadence (the operational HRRR
    # hourly-then-15-minute pair): writes no prepared byte, and
    # woof.ingest.prepared_cache already keeps it out of the trajectory
    # identity.  A storm slot keeps its own 15-minute cadence.
    "history_interval2_s", "history_interval_change_s",
    # A storm slot (woof.storm_nests) runs without the GSL drag and the
    # lake model: a child's statics carry no sub-grid orography statistics
    # (experiment._refuse_child_terrain_drag) and a rebuilt nest does not
    # carry the lake column (experiment._refuse_rebuilt_nest_lake).  The
    # child is prepared from its OWN namelist column, which carries both
    # zeros, so preparation builds exactly the child the forecast runs.
    "gwd_opt", "sf_lake_physics",
    # The per-domain turbulence row (woof/experiment.py
    # _DOMAIN_RUN_OVERRIDES, "a PBL parent may carry a PBL-off
    # Smagorinsky child").  Same shape of finding as output_interval_s
    # above: HRRR hierarchy PREPARATION never reads one of these keys.
    # `grep -rn 'bl_pbl_physics\|km_opt' woof/ingest/ woof/native_hierarchy.py
    # tools/prepare_hrrr_wrf.py` returns a single hit, and it is a comment.
    # Preparation writes static fields and an interpolated initial state;
    # the closure selects tendencies that only the forecast computes.  The
    # one place the turbulence choice reaches preparation is
    # DomainState(cfg) allocating km_opt=2's zero-initialised tke/tke0,
    # which the prepared cache round-trips like any other state array.
    "bl_pbl_physics", "km_opt", "c_s", "c_k", "diff_6th_opt",
    "mix_isotropic", "mix_upper_bound", "isfflx",
    "tke_heat_flux", "tke_drag_coefficient", "tke_upper_bound",
    # The P3 inflow-seeding keys, for the third time the same finding:
    # preparation never reads them, and this drift check refused them
    # AFTER the expensive preparation.  The ruling that they are
    # preparation-inert is not new here and is not being made here -- it
    # is already committed one module over, as
    # woof.ingest.prepared_cache.PREPARATION_INERT_RUN_FIELDS
    # (prepared_cache.py:241-246), whose published reason applies
    # verbatim: the generator acts at runtime FORCE on the child-owned
    # rolling NEST boundary tables, which preparation never computes, so
    # a tree prepared with any value of these fields holds exactly the
    # prepared state every value of them runs from.  Provenance for the
    # ruling itself: controller decision 2026-08-03 under ArWen's standing
    # delegation, docs/superpowers/receipts/les/
    # INFLOW-GENERATOR-ACCEPTANCE-V2.md item 10.  These stay
    # trajectory-RELEVANT inside experiment_fingerprint and the restart
    # identity; only the two preparation-side comparisons drop them, and
    # until now only one of the two did.
    "inflow_perturbation", "inflow_perturbation_seed",
    "inflow_perturbation_amplitude_scale", "inflow_perturbation_faces",
}) | frozenset(
    # Every run field the prepared-cache comparison already rules
    # preparation-inert, read from that one table instead of copied into
    # this one a field at a time.  Copying is how this check fell behind:
    # the inflow keys above were inert there and a drift here until they
    # were copied, and the adaptive clock's per-domain targets and clamps
    # (a 1 km parent at max_step_increase_pct 5 beside a 500 m nest at
    # 51) were the same again, so a two-level adaptive tree was refused
    # by this stage after its root had been prepared.
    path.partition(".")[2] for path in PREPARATION_INERT_RUN_FIELDS
    if path.startswith("run."))
_MAX_PUBLIC_DOMAINS = 21


def _json(path: Path) -> dict[str, object]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _same_typed_values(observed, expected) -> bool:
    return (isinstance(observed, list)
            and len(observed) == len(expected)
            and all(type(left) is type(right) and left == right
                    for left, right in zip(observed, expected)))


def _sealed_root_rrtmg_variant(identity: dict[str, object]) -> str | None:
    """The RRTMG variant the sealed root preparation pinned, if any.

    A WRF namelist spells radiation as selector integers; which 4/4
    IMPLEMENTATION serves them (RTE+RRTMGP or the exact legacy port) is
    the sealed root's physics profile's decision, recorded in its cache
    identity.  This reads that decision so the hierarchy imports its
    namelists under the same variant.  Absent or malformed shapes return
    None (the importer's default) rather than refusing here: the
    d01-binding comparison downstream still refuses any real mismatch,
    so failing open cannot admit a wrong trajectory -- it only preserves
    the behaviour of headers written before the field existed.
    """
    domain_config = identity.get("domain_config")
    if not isinstance(domain_config, dict):
        return None
    run = domain_config.get("run")
    if not isinstance(run, dict):
        return None
    variant = run.get("ra_rrtmg_variant")
    return variant if isinstance(variant, str) else None


def _sealed_root_rrtmg_compatibility(identity: dict[str, object]) -> str | None:
    """The WRF RRTMG lineage the sealed root preparation pinned, if any.

    The companion of :func:`_sealed_root_rrtmg_variant`, read for the same
    reason and failing open the same way.  A WRF namelist has no key for
    ``wrf_rrtmg_compatibility`` and the token is not decoration -- the
    RTE+RRTMGP arm reads it to choose its snow treatment and stamps it
    into the restart algorithm identity -- so an import that derives it
    can hand this hierarchy a different lineage from the root it must
    match.  The d01-binding comparison downstream still refuses any real
    mismatch.
    """

    domain_config = identity.get("domain_config")
    if not isinstance(domain_config, dict):
        return None
    run = domain_config.get("run")
    if not isinstance(run, dict):
        return None
    compatibility = run.get("wrf_rrtmg_compatibility")
    return compatibility if isinstance(compatibility, str) else None


def with_config_lifecycle(native_exp, config: Path):
    """``native_exp`` with each domain's spawn/follow/retire/rearm from ``config``.

    The namelists this stage reads have no spelling for a nest's lifecycle,
    so a storm slot came through as a fixed t = 0 child: its statics
    corridor covered only its placeholder footprint and the forecast could
    not crop ground where its trigger fired.  The tables are copied by grid
    id from the config the chain runs (woof go's re-stated copy); geometry
    and physics stay the namelists'.
    """
    from dataclasses import fields, replace

    from woof.config_authority import read_config_authority
    from woof.experiment import build_experiment_from_config_tables

    authority = read_config_authority(config)
    declared = build_experiment_from_config_tables(
        tomllib.loads(authority.payload.decode("utf-8")),
        source=str(authority.source), base_dir=authority.base_dir)
    by_id = {int(dc.grid_id): dc for dc in declared.domains}
    names = [name for name in ("spawn", "follow", "retire", "rearm")
             if name in {f.name for f in fields(native_exp.domains[0])}]
    domains = []
    for dc in native_exp.domains:
        source = by_id.get(int(dc.grid_id))
        if source is None:
            domains.append(dc)
            continue
        domains.append(replace(dc, **{name: getattr(source, name) for name in names}))
    return replace(native_exp, domains=tuple(domains))


def _native_experiment(wps_namelist: Path, namelist_input: Path,
                       *, rrtmg_variant: str | None = None,
                       rrtmg_compatibility: str | None = None,
                       acknowledgements: tuple[str, ...] = ()):
    keywords = ({} if rrtmg_variant is None
                else {"rrtmg_variant": rrtmg_variant})
    if rrtmg_compatibility is not None:
        keywords["rrtmg_compatibility"] = rrtmg_compatibility
    # The emission's round trip reads the pair under the same request
    # source (woof.hrrr_route_inputs.verify_round_trip); one constant, so
    # the two readings of one pair cannot fill its silences differently.
    from woof.hrrr_route_inputs import ROUTE_REQUEST_SOURCE
    resolved, report = import_namelists(
        wps_namelist, namelist_input, name="native_hrrr_hierarchy",
        request_source=ROUTE_REQUEST_SOURCE,
        acknowledgements=tuple(acknowledgements), **keywords)
    # The import is a one-file case: a WUDAPT geog_data_res writes a
    # [static] companion, validated and split off here like every file
    # door does.  This route's statics come from its sealed root, and its
    # static catalog refuses the token by name unless that root builds it.
    from woof.experiment import build_experiment_from_config_tables
    exp = build_experiment_from_config_tables(
        tomllib.loads(resolved),
        source=f"native HRRR hierarchy {wps_namelist} + {namelist_input}",
        base_dir=Path(wps_namelist).parent,
    )
    validate_native_lambert_contracts(
        exp, wps_namelist, source_name="native HRRR hierarchy")
    return exp, resolved, report


def _require_raw_stock_delta(
        native_namelist: Path,
        stock_namelist: Path,
) -> dict[str, object]:
    """Require only the explicit native-to-stock runtime deltas.

    The longwave delta is a PER-DOMAIN substitution, not a constant: the
    stock arm selects RRTM (1) exactly where the native arm runs with
    longwave off (0), and carries any other admitted longwave selection
    unchanged -- so under the resolved RRTMG (4, 4) pair the delta
    collapses and both arms run 4.  ``ghg_input=0`` stays stock-only in
    both cases: on the (0, 1) suite it fixes the substituted RRTM's gas
    table, and on the (4, 4) suite it mirrors the fixed-gas configuration
    the native arm runs on EITHER engine -- ``woof/core/rrtmg_legacy.py``
    pins it in its switch table, and the RTE+RRTMGP arm selects one annual
    mean CO2 for the run's calendar year (``woof/core/rrtmgp.py``).
    Pinning it is mandatory either way: WRF's default ``1`` reads the
    time-varying CAM gas table and is not a valid implicit substitute for
    the fixed-gas configurations used by the acceptance gate.

    ``do_radar_ref=1`` is the second stock-only key and is mandatory for
    the same class of reason: it is a setting the native arm answers in
    CODE rather than from a namelist, so the native file must stay
    silent about it while the stock file has to be told.  woof
    evaluates REFL_10CM at output time unconditionally; WRF allocates
    the array only under ``package radar_refl compute_radar_ref==1``.
    At WRF's default the stock arm's history frames carry no REFL_10CM
    at all, and every reflectivity score on that arm is unanswerable.

    ``use_theta_m`` is NOT a delta: both files declare 0.  The engine
    integrates dry theta and the stock-WRF export holds dry theta
    (:data:`woof.wrf_physics_inventory.EXPORT_USE_THETA_M`), so a stock
    namelist that says 1 stops wrf.exe at its input gate.  It was a delta
    (0 -> 1) while the export declared ``USE_THETA_M = 1``; that pair
    carried dry-coupled boundary rows under a moist header, which a
    moist-theta wrf.exe read as a boundary colder than the analysis.

    The fork-only ``alb_sol`` is optional in the native file and must be
    absent from the stock file. WRF 4.6.1 has no key, so retaining it
    would stop the Fortran namelist read. Its omission is recorded as
    an explicit physical difference when the native correction is on.
    """

    native = parse_namelist(native_namelist)
    stock = parse_namelist(stock_namelist)
    from woof.hrrr_route_inputs import STOCK_ALB_SOL_REASON

    native_alb_sol = native.get("physics", {}).get("alb_sol")
    if "alb_sol" in stock.get("physics", {}):
        raise ValueError("stock-WRF namelist must omit alb_sol: "
                         + STOCK_ALB_SOL_REASON)
    if (native_alb_sol is not None
            and (not isinstance(native_alb_sol, list)
                 or len(native_alb_sol) != 1
                 or type(native_alb_sol[0]) is not int
                 or native_alb_sol[0] not in (0, 1))):
        raise ValueError(
            "native &physics/alb_sol must be omitted or one integer 0 or 1; "
            f"got {native_alb_sol}")
    try:
        native_max_dom = native["domains"]["max_dom"]
        stock_max_dom = stock["domains"]["max_dom"]
        native_lw = native["physics"]["ra_lw_physics"]
        stock_lw = stock["physics"]["ra_lw_physics"]
        native_theta = native["dynamics"]["use_theta_m"]
        stock_theta = stock["dynamics"]["use_theta_m"]
        stock_ghg = stock["physics"]["ghg_input"]
        stock_radar_ref = stock["physics"]["do_radar_ref"]
    except KeyError as exc:
        raise ValueError(
            "both native and stock namelists must explicitly declare "
            "&domains/max_dom, "
            "&physics/ra_lw_physics and &dynamics/use_theta_m; the stock "
            "namelist must also declare &physics/ghg_input and "
            "&physics/do_radar_ref") from exc
    if (len(native_max_dom) != 1 or isinstance(native_max_dom[0], bool)
            or not isinstance(native_max_dom[0], int)
            or not 1 <= native_max_dom[0] <= _MAX_PUBLIC_DOMAINS
            or not _same_typed_values(stock_max_dom, native_max_dom)):
        raise ValueError(
            "native and stock namelists must explicitly declare the same "
            f"integer max_dom in [1, {_MAX_PUBLIC_DOMAINS}]")
    max_dom = native_max_dom[0]
    # The soil column count follows the land surface the tree runs, one
    # scheme on every domain as WRF requires: Noah's four layers, RUC's
    # nine.  A literal 4 here refused a RUC tree after the download and
    # the root preparation, so the sub-km default could not reach a
    # nested --source hrrr domain at all.
    surface = native.get("physics", {}).get("sf_surface_physics")
    if (not isinstance(surface, list) or len(surface) != max_dom
            or any(type(value) is not int for value in surface)
            or len(set(surface)) != 1):
        raise ValueError(
            "native hierarchy namelist must declare one integer "
            "&physics/sf_surface_physics entry per domain, the same on "
            f"every domain; got {surface}.  num_soil_layers is pinned "
            "from that one land surface (4 for Noah and Noah-MP, 9 for "
            "RUC), and a tree that mixes or omits it has no single soil "
            "column to pin, so its prepared soil would not match the "
            "scheme some domain runs")
    from woof.config import validated_soil_layer_count

    try:
        soil_layers = validated_soil_layer_count(surface[0])
    except KeyError:
        raise ValueError(
            f"&physics/sf_surface_physics = {surface[0]} has no soil "
            "geometry woof prepares, so num_soil_layers has no count to "
            "pin and the prepared soil column could not match the land "
            "surface the tree runs") from None

    certified_pins = {
        ("time_control", "interval_seconds"): [3600],
        ("time_control", "frames_per_outfile"): [1] * max_dom,
        ("time_control", "restart"): [False],
        ("time_control", "io_form_history"): [2],
        ("time_control", "io_form_restart"): [2],
        ("time_control", "io_form_input"): [2],
        ("time_control", "io_form_boundary"): [2],
        ("domains", "num_metgrid_levels"): [51],
        ("domains", "num_metgrid_soil_levels"): [9],
        ("physics", "isfflx"): [1],
        ("physics", "ifsnow"): [1],
        ("physics", "icloud"): [1],
        ("physics", "surface_input_source"): [1],
        ("physics", "num_soil_layers"): [soil_layers],
        ("physics", "sf_urban_physics"): [0] * max_dom,
        ("physics", "sst_update"): [0],
    }
    surface_pressure = native.get("domains", {}).get("sfcp_to_sfcp")
    if (not isinstance(surface_pressure, list) or len(surface_pressure) != 1
            or type(surface_pressure[0]) is not bool
            or not _same_typed_values(
                stock.get("domains", {}).get("sfcp_to_sfcp"), surface_pressure)):
        raise ValueError("native and stock namelists must declare the same boolean sfcp_to_sfcp")
    observed_pins = {"domains.sfcp_to_sfcp": surface_pressure}
    for (section, key), expected in certified_pins.items():
        observed = native.get(section, {}).get(key)
        observed_pins[f"{section}.{key}"] = observed
        if not _same_typed_values(observed, expected):
            raise ValueError(
                "native hierarchy namelist is outside the certified raw "
                f"runtime contract: &{section}/{key} expected {expected}, "
                f"got {observed}")
    for section, key in (
            ("time_control", "nwp_diagnostics"),
            ("physics", "do_radar_ref")):
        if key in native.get(section, {}):
            raise ValueError(
                "native hierarchy namelist is outside the certified raw "
                f"runtime contract: &{section}/{key} must be omitted")

    if (not isinstance(native_lw, list) or len(native_lw) != max_dom
            or any(isinstance(value, bool) or not isinstance(value, int)
                   or value not in (0, 4) for value in native_lw)):
        raise ValueError(
            "native hierarchy namelist must declare one explicit "
            "ra_lw_physics entry per domain, each 0 (longwave off) or 4 "
            f"(RRTMG); got {native_lw}")
    if (not _same_typed_values(native_theta, [0])
            or not _same_typed_values(stock_theta, [EXPORT_USE_THETA_M])):
        raise ValueError(
            "native and stock namelists must both declare "
            f"&dynamics/use_theta_m = {EXPORT_USE_THETA_M} (dry theta); "
            f"got native {native_theta}, stock {stock_theta}. The engine "
            "integrates dry theta and the stock-WRF files it exports hold "
            f"dry theta (USE_THETA_M = {EXPORT_USE_THETA_M} on wrfinput "
            "and wrfbdy), so a stock namelist that says otherwise stops "
            "wrf.exe at its input gate with 'use_theta_m values must be "
            "consistent'. A stock namelist generated before the export "
            "wrote dry theta says use_theta_m = 1: change that line to "
            f"{EXPORT_USE_THETA_M}, or regenerate the pair")
    stock_lw_expected = [1 if value == 0 else value for value in native_lw]
    if (not _same_typed_values(stock_lw, stock_lw_expected)
            or "ghg_input" in native["physics"]
            or not _same_typed_values(stock_ghg, [0])
            # do_radar_ref stays FORBIDDEN in the native namelist (the
            # loop above) and is now MANDATORY at 1 in the stock one:
            # woof evaluates REFL_10CM at output time whatever a
            # namelist says, while WRF allocates the array only under
            # `package radar_refl compute_radar_ref==1`
            # (Registry.EM_COMMON:3059, fed by do_radar_ref via
            # module_check_a_mundo.F:3477).  Left at the Registry
            # default the stock arm writes history frames with no
            # REFL_10CM at all and every reflectivity score on that arm
            # is unanswerable -- measured, first WRF-arm scoring
            # attempt.  A scalar: Registry.EM_COMMON:2447, nentries 1.
            or not _same_typed_values(stock_radar_ref, [1])):
        raise ValueError(
            "the evidenced raw namelist deltas require ra_lw_physics "
            "0 -> 1 exactly where the native entry is 0 (a native 4 "
            "carries to the stock arm unchanged), "
            "stock-only ghg_input=0, and stock-only do_radar_ref=1")
    normalized_stock = deepcopy(stock)
    normalized_stock["physics"]["ra_lw_physics"] = list(native_lw)
    normalized_stock["physics"].pop("ghg_input")
    normalized_stock["physics"].pop("do_radar_ref")
    if native_alb_sol is not None:
        # Only the typed value verified above is normalized. Its actual
        # absence from stock and disabled stock behavior stay in the receipt.
        normalized_stock["physics"]["alb_sol"] = list(native_alb_sol)
    # A chem tree's &chem group and the chem array's &dynamics switches are
    # native-only (woof.hrrr_route_inputs._chem_namelist_rows): the stock
    # WRF arm is a non-chem build, so it runs the meteorology without them.
    native_chem = deepcopy(native.get("chem")) if "chem" in native else None
    if native_chem is not None and "chem" not in stock:
        normalized_stock["chem"] = deepcopy(native_chem)
        for key in ("chem_adv_opt", "chem_mix2_off", "chem_mix6_off"):
            if (key in native.get("dynamics", {})
                    and key not in stock.get("dynamics", {})):
                normalized_stock["dynamics"][key] = deepcopy(
                    native["dynamics"][key])
    if native != normalized_stock:
        differences = []
        for section in sorted(set(native) | set(normalized_stock)):
            left = native.get(section, {})
            right = normalized_stock.get(section, {})
            for key in sorted(set(left) | set(right)):
                if left.get(key) != right.get(key):
                    differences.append(f"&{section}/{key}")
        raise ValueError(
            "stock-WRF namelist raw settings differ beyond "
            "ra_lw_physics 0 -> 1, stock-only "
            "ghg_input=0, stock-only do_radar_ref=1, and native-only alb_sol: "
            + ", ".join(differences))
    receipt = {
        "schema": "gpuwm-native-to-stock-namelist-delta-v5",
        "status": "PASS",
        "max_dom": max_dom,
        "certified_native_runtime": observed_pins,
        "allowed_deltas": {
            "physics.ra_lw_physics": {
                "native": list(native_lw), "stock": stock_lw_expected},
            "physics.ghg_input": {"native": None, "stock": [0]},
            "physics.do_radar_ref": {"native": None, "stock": [1]},
        },
        "shared_dry_theta": {
            "dynamics.use_theta_m": [EXPORT_USE_THETA_M]},
    }
    if native_chem is not None and "chem" not in stock:
        receipt["allowed_deltas"]["chem"] = {
            "native": native_chem, "stock": None,
            "reason": ("the stock WRF arm is a non-chem build; it runs the "
                       "meteorology without the native arm's chemistry"),
        }
    if native_alb_sol is not None:
        receipt["allowed_deltas"]["physics.alb_sol"] = {
            "native": list(native_alb_sol), "stock": None,
            "stock_effective": 0,
            "physics_equivalent": native_alb_sol[0] == 0,
            "reason": STOCK_ALB_SOL_REASON,
        }
        if native_alb_sol[0] == 1:
            receipt["comparison_limitations"] = [
                "The native arm applies sun-angle land albedo and the "
                "stock WRF 4.6.1 arm does not. This comparison cannot "
                "qualify identical land shortwave forcing."]
    return receipt


def _require_raw_wps_contract(
        wps_namelist: Path, max_dom: int) -> dict[str, object]:
    wps = parse_namelist(wps_namelist)
    required = {
        ("share", "max_dom"): [max_dom],
        ("share", "interval_seconds"): [3600],
    }
    observed = {}
    for (section, key), expected in required.items():
        value = wps.get(section, {}).get(key)
        observed[f"{section}.{key}"] = value
        if not _same_typed_values(value, expected):
            raise ValueError(
                "WPS namelist is outside the certified raw hierarchy "
                f"contract: &{section}/{key} expected {expected}, got "
                f"{value}")
    return {
        "schema": "gpuwm-native-wps-hierarchy-contract-v1",
        "status": "PASS",
        "max_dom": max_dom,
        "certified": observed,
    }


def _domain_layout(domain) -> dict[str, object]:
    return {
        "grid_id": domain.grid_id,
        "parent_id": domain.parent_id,
        "i_parent_start": domain.i_parent_start,
        "j_parent_start": domain.j_parent_start,
        "parent_grid_ratio": domain.parent_grid_ratio,
        "parent_time_step_ratio": domain.parent_time_step_ratio,
        "nx": domain.run.nx, "ny": domain.run.ny,
        "nz": domain.run.nz, "dx": domain.run.dx,
        "dy": domain.run.dy, "dt": domain.run.dt,
        "specified": domain.run.specified,
        "nested": domain.run.nested,
        "history_interval_s": domain.history_interval_s,
    }


def namelist_start_refusal(*, requested: datetime, observed: datetime,
                           namelist_name: str) -> str:
    """Why the namelist's start time and model time zero must be one instant.

    A pure function so the sentence a mis-copied time meets is testable
    without a sealed root preparation on disk.
    """

    return (
        "model time zero differs from the native namelist start: this "
        f"stage was told {requested:%Y-%m-%d %H:%M:%S} and "
        f"{namelist_name} starts at {observed:%Y-%m-%d %H:%M:%S}.  The two "
        "must be the same instant.  If the root preparation began at a "
        "forecast lead, model time zero is cycle + K, not the cycle: pass "
        "--cycle CYCLE --forecast-start-hour K (the same pair "
        "tools/prepare_hrrr_wrf.py was given), and emit the namelist with "
        "`woof domain --forecast-start-hour K` so its start_* keys carry "
        "the same instant.")


def sealed_start_refusal(*, requested: datetime, sealed_start: datetime,
                         sealed_cycle: object) -> str:
    """Why the sealed cache's model time zero and the request must agree.

    Names the cycle and the lead the root was actually prepared at
    whenever the cache recorded them, because that pair is the answer:
    the same two values passed here reproduce this instant exactly.
    """

    lead = ""
    if isinstance(sealed_cycle, str):
        try:
            offset = sealed_start - datetime.fromisoformat(sealed_cycle)
        except ValueError:
            pass
        else:
            lead = (f"  That root was prepared from cycle {sealed_cycle} at "
                    f"lead f{int(offset.total_seconds() // 3600):02d}.")
    return (
        "root preparation model time zero differs from request: the sealed "
        f"cache starts at {sealed_start:%Y-%m-%d %H:%M:%S}, this stage was "
        f"told {requested:%Y-%m-%d %H:%M:%S}." + lead
        + "  Pass --cycle CYCLE --forecast-start-hour K -- the same pair "
        "the root preparation was given -- so both stages derive the same "
        "model time zero.")


def sealed_source_leads(identity: Mapping[str, object],
                        forcing_hours) -> tuple[int, ...]:
    """The ABSOLUTE HRRR leads the sealed root's decoded bridge tree holds.

    Two hour numberings meet here and are equal only at lead 0.  The
    prepared cache's ``forcing_hours`` are MODEL-relative -- 0, 1, 2 --
    because that is what a forecast clock counts.  The decoder's
    published tree is keyed by NOAA's absolute cycle-relative lead: its
    directories are ``atmosphere-f06``/``soil-f06`` and its gate declares
    ``(6, 7, 8)``.  Handing the model-relative 0 to that tree asked for
    an hour it does not carry, and a root prepared at f06 refused with
    "forecast_hour must be one of (6, 7, 8), got 0" after every check
    above it had passed.

    The sealed cache records both sequences in its own source identity,
    so the absolute leads are read from the bundle rather than
    reconstructed.  A cache sealed before that field existed carries only
    the model-relative inventory, which at the lead 0 those releases
    could express IS the absolute one -- so the fallback is exact, not a
    guess.
    """

    model_relative = tuple(int(hour) for hour in forcing_hours)
    source_identity = identity.get("source_identity")
    leads = (source_identity.get("source_forecast_hours")
             if isinstance(source_identity, Mapping) else None)
    if leads is None:
        return model_relative
    absolute = tuple(int(hour) for hour in leads)
    if len(absolute) != len(model_relative):
        raise ValueError(
            "root preparation source identity declares "
            f"{len(absolute)} absolute forecast lead(s) but its cache "
            f"carries {len(model_relative)} model forcing hour(s)")
    if absolute != tuple(range(absolute[0], absolute[0] + len(absolute))):
        raise ValueError(
            "root preparation absolute forecast leads must be contiguous "
            f"and ordered; got {list(absolute)}")
    return absolute


def sealed_forcing_horizon_seconds(forcing_hours) -> float:
    """The forecast length one sealed root preparation can force.

    The prepared root carries a contiguous hourly forcing inventory
    starting at its own hour zero; a hierarchy may run to its last hour
    and no further.  That is a property of the bundle in front of us, so
    it is read from the bundle.
    """

    hours = [int(hour) for hour in forcing_hours]
    expected = list(range(len(hours)))
    if not hours or hours != expected:
        raise ValueError(
            "root preparation forcing inventory must be contiguous hourly "
            f"leads starting at 0; got {hours}")
    return float(hours[-1]) * 3600.0


def verified_root_forcing_inventory(
        forcing_hours, *, run_seconds: float) -> tuple[int, ...]:
    """Require exactly the inventory the preparer seals for this run.

    ONE endpoint convention on both stages, and it is a ceiling.
    ``tools.prepare_hrrr_wrf`` sizes its fetch/decode window with
    :func:`woof.hrrr_forecast.hrrr_forcing_end_hour`: hourly boundary
    forcing brackets every model instant between two frames, so a 900 s
    run -- whose endpoint at 0.25 h lies BETWEEN f000 and f001 -- is
    prepared with model forcing hours (0, 1).  This check used to
    recompute the endpoint with a floor (``run_seconds // 3600``),
    expected ``(0,)`` for that same run, and refused every sub-hour root
    the preparer can build; ``(0,)`` is also a series the tree runner
    (woof/prepared_domain_tree_forecast.py) rejects outright, because a
    single frame brackets nothing.  The expectation is derived from the
    shared ceiling now, and a genuinely truncated, over-long, or gapped
    inventory still refuses with the same sentence.
    """

    observed = tuple(forcing_hours)
    expected = tuple(range(hrrr_forcing_end_hour(run_seconds) + 1))
    if observed != expected:
        raise ValueError(
            "root preparation must contain consecutive hourly HRRR forcing "
            f"through the forecast endpoint: expected {expected}, "
            f"got {observed}")
    return observed


def _supported_hierarchy_slice(exp, root_target, *, forcing_hours) -> None:
    """Bind the public adapter to its sealed root and generic valid nests.

    ``build_experiment`` is the geometry authority: it rejects cycles,
    orphans, invalid ratios, misaligned extents, and insufficient parent-row
    clearance before this gate runs.  The target-domain document and sealed
    root preparation are the d01 authority; this function verifies that the
    requested namelist describes that root without pinning one location,
    duration, or vertical grid.  The fixed native HRRR physics product remains
    explicit and children may vary only in their geometry and nest identity.

    ``forcing_hours`` is the sealed root preparation's own inventory.  The
    duration ceiling is derived from it rather than from a constant: the
    docstring above already promised not to pin a duration, and every
    other stage of this route -- ``hrrr_source_window`` in the fetch/root
    wrapper, the forcing-hour equality below, the tree runner -- handles
    f00..f24 dynamically.  A hardcoded 12 h refused an f00..f24 root
    preparation that had already been fetched and prepared, with a
    sentence ("the native f00..f12 forcing horizon") describing a
    limitation of neither the data nor the code.
    """

    if not 1 <= len(exp.domains) <= _MAX_PUBLIC_DOMAINS:
        raise ValueError(
            "the public HRRR hierarchy gate requires between 1 and "
            f"{_MAX_PUBLIC_DOMAINS} domains")
    from woof.wps_domain_ids import validated_domain_order
    validated_domain_order(exp.domains)
    # Two-way trees are admitted: feedback and its smoother are a runtime
    # coupling the tree executor runs, not a property of anything this
    # stage prepares, so a one-way refusal here only turned away every
    # two-way layout after its root had been fetched and prepared.
    horizon = sealed_forcing_horizon_seconds(forcing_hours)
    if not 0.0 < exp.run_seconds <= horizon:
        raise ValueError(
            "the public HRRR hierarchy duration must be positive and no "
            f"longer than the {horizon / 3600.0:g} h forcing horizon this "
            "root preparation was sealed with (f00.."
            f"f{int(horizon // 3600):02d}); requested "
            f"{exp.run_seconds / 3600.0:g} h.  Re-prepare the root over a "
            "longer window to run longer.")
    expected_projection = {
        "map_proj": root_target.map_proj,
        "ref_lat": root_target.ref_lat,
        "ref_lon": root_target.ref_lon,
        "truelat1": root_target.truelat1,
        "truelat2": root_target.truelat2,
        "stand_lon": root_target.stand_lon,
    }
    if exp.projection is None or asdict(exp.projection) != expected_projection:
        raise ValueError(
            "the public HRRR hierarchy projection differs from the sealed "
            "root target-domain document")
    observed_root = _domain_layout(exp.domains[0])
    expected_root = {
        "grid_id": 1,
        "parent_id": 0,
        "i_parent_start": 1,
        "j_parent_start": 1,
        "parent_grid_ratio": 1,
        "parent_time_step_ratio": 1,
        "nx": root_target.nx,
        "ny": root_target.ny,
        "nz": root_target.nz,
        "dx": root_target.dx_m,
        "dy": root_target.dy_m,
        "dt": float(root_target.time_step_exact),
        "specified": True,
        "nested": False,
        "history_interval_s": observed_root["history_interval_s"],
    }
    if observed_root != expected_root:
        raise ValueError(
            "d01 differs from the sealed HRRR root target-domain document: "
            f"{observed_root}")
    root_run = exp.domains[0].run
    if (exp.spec_bdy_width != root_target.spec_bdy_width
            or root_run.spec_zone != root_target.spec_zone
            or root_run.relax_zone != root_target.relax_zone):
        raise ValueError(
            "d01 boundary-zone controls differ from the sealed HRRR root "
            "target-domain document")

    root_domain = exp.domains[0]
    root_run = asdict(root_run)
    seen = {1}
    by_id = {domain.grid_id: domain for domain in exp.domains}
    for domain in exp.domains:
        if domain.grid_id > 1:
            if domain.parent_id not in seen:
                raise ValueError(
                    f"d{domain.grid_id:02d} names unavailable parent "
                    f"d{domain.parent_id:02d}; domains must be stored "
                    "parent-before-child")
            if domain.parent_id == domain.grid_id:
                raise ValueError(
                    f"d{domain.grid_id:02d} cannot be its own parent")
        seen.add(domain.grid_id)
        if domain.run.nz != root_target.nz:
            raise ValueError(
                f"d{domain.grid_id:02d} must use the sealed root's "
                f"{root_target.nz}-level vertical grid")
        radiation_scheme_ids(domain.run)  # validate selector coherence
        # The configured initializer and shared runtime own physics capability.
        # A registered source/preset pair is evidence, not an execution requirement.
        expected_run = dict(root_run)
        expected_run.update({
            "grid_id": domain.grid_id,
            "nx": domain.run.nx, "ny": domain.run.ny,
            "dx": domain.run.dx, "dy": domain.run.dy,
            "dt": domain.run.dt,
            "specified": domain.grid_id == 1,
            "nested": domain.grid_id != 1,
        })
        observed_run = asdict(domain.run)
        drift = {
            name: (observed_run[name], expected_run[name])
            for name in observed_run
            if observed_run[name] != expected_run[name]
            and name not in _DOMAIN_PREPARATION_OVERRIDES
        }
        if drift:
            raise ValueError(
                f"d{domain.grid_id:02d} trajectory controls differ from "
                f"the HRRR root outside supported per-domain overrides: "
                f"{drift}")
        if domain is not root_domain:
            # This is the executable authority, including the exact MP8->MP18
            # diagnosis contract.  Unsupported mixed pairs and missing CQ
            # requirements remain genuine compatibility errors; an admitted
            # but not acceptance-gated edge is recorded as unverified by the
            # forecast runner rather than blocked for consent.
            resolve_microphysics_transition(
                by_id[domain.parent_id].run, domain.run)


def _compare_stock_experiment(native_exp, stock_exp) -> None:
    for native_domain, stock_domain in zip(native_exp.domains,
                                           stock_exp.domains):
        expected = (1 if native_domain.run.ra_lw_physics == 0
                    else native_domain.run.ra_lw_physics)
        if stock_domain.run.ra_lw_physics != expected:
            raise ValueError(
                f"stock-WRF namelist d{stock_domain.grid_id:02d} must "
                f"select ra_lw_physics={expected} (RRTM substitutes the "
                "native arm's disabled longwave; any other native "
                "longwave carries unchanged)")
        native_alb_sol = getattr(native_domain.run, "alb_sol", 0)
        stock_alb_sol = getattr(stock_domain.run, "alb_sol", 0)
        if (type(native_alb_sol) is not int or native_alb_sol not in (0, 1)
                or type(stock_alb_sol) is not int or stock_alb_sol != 0):
            from woof.hrrr_route_inputs import STOCK_ALB_SOL_REASON
            raise ValueError(
                "stock-WRF imported configuration must carry alb_sol=0 "
                "beside native integer 0 or 1: " + STOCK_ALB_SOL_REASON)
    from woof.experiment import experiment_config_document
    native = experiment_config_document(native_exp)
    stock = experiment_config_document(stock_exp)
    for document in (native, stock):
        document["name"] = "normalized"
        # Who chose epssm is a label; the value each domain runs is on
        # run.epssm and is compared there, so a namelist that lists it
        # once and one that lists it per domain still agree on 0.1.
        document.pop("auto_epssm", None)
        for domain in document["domains"]:
            domain["run"]["ra_lw_physics"] = 0
            domain["run"]["alb_sol"] = 0
    if native != stock:
        raise ValueError(
            "stock-WRF namelist differs from the native hierarchy setup "
            "beyond the allowed longwave and native-only alb_sol runtime changes")


def _effective_domain_config(domain) -> dict[str, object]:
    """Canonicalize cadence fields that are inactive under selected schemes.

    One line now, because the normalization moved to
    :func:`woof.ingest.prepared_cache.effective_prepared_domain_config`
    where every consumer of a prepared-domain identity can reach it --
    this gate and the tree forecast's preflight were normalizing
    differently, which is how a wizard-emitted config could pass here and
    be refused there.
    """

    return effective_prepared_domain_config(
        prepared_domain_config_identity(domain))


def _validated_root_preparation_binding(
        identity: dict[str, object], root_domain,
) -> tuple[str, dict[str, object]]:
    """Authorize cache reuse by exact effective-d01 trajectory equality.

    A prepared root is intentionally independent of later child topology.
    Its original full-namelist digest remains part of the immutable cache
    identity, while the requested hierarchy namelist is bound separately in
    the hierarchy receipt.  Reuse is allowed only when the cache's resolved
    d01 configuration is exactly the requested effective d01 configuration.
    """

    prepared_domain = identity.get("domain_config")
    if not isinstance(prepared_domain, dict):
        raise ValueError("root preparation identity lacks domain_config")
    if not isinstance(prepared_domain.get("run"), dict):
        raise ValueError("root preparation domain_config lacks run controls")
    differing = root_binding_differences(prepared_domain, root_domain)
    if differing:
        raise ValueError(
            "native namelist d01 trajectory controls differ from the sealed "
            f"root preparation: {', '.join(sorted(differing))}")

    root_namelist_sha256 = identity.get("namelist_sha256")
    if (not isinstance(root_namelist_sha256, str)
            or len(root_namelist_sha256) != 64
            or any(character not in "0123456789abcdef"
                   for character in root_namelist_sha256)):
        raise ValueError(
            "root preparation identity has an invalid namelist_sha256")
    return root_namelist_sha256, prepared_domain


def root_binding_differences(prepared_domain: dict[str, object],
                             root_domain) -> set[str]:
    """The d01 trajectory fields ``root_domain`` holds differently.

    ``prepared_domain`` is a prepared-domain identity document (a sealed
    root's ``domain_config``).  The one rule the hierarchy binds a sealed
    root by, also asked by the route-inputs choice
    (:func:`woof.hrrr_route_inputs.run_route_inputs`) before anything is
    fetched, so a namelist set the binding would refuse is never the set
    a run prepares from.
    """

    effective_prepared_domain = effective_prepared_domain_config(
        deepcopy(prepared_domain))
    # Same rule as every other prepared-identity comparison: a field the
    # header predates, holding its not-in-use default, is schema growth
    # and not a trajectory difference.  Everything else still refuses.
    live_root = _effective_domain_config(root_domain)
    # d01 is never a delayed domain -- the delayed-start rules apply to
    # exp.domains[1:] -- so the root's own start IS the experiment's, and
    # that is precisely the not-in-use value a header written before the
    # field existed describes.
    # The same not-in-use map every other prepared-cache comparison passes
    # (prepared_cache.undelayed_identity_defaults): a field that joined the
    # identity after the sealed root was written (the 2.9.4 sea-ice
    # parameters) is tolerated at its off value instead of refused.  Passing
    # start_time alone refused a nested HRRR-recipe tree at this binding:
    # "run.seaice_initial_thickness, run.seaice_new_ice_thickness,
    # run.seaice_ocean_heat_flux" (2.9.4 rolling, a 4090 host).
    from types import SimpleNamespace as _Namespace

    from woof.ingest.prepared_cache import undelayed_identity_defaults
    not_in_use = dict(undelayed_identity_defaults(_Namespace(start_time=None)))
    not_in_use["start_time"] = live_root.get("start_time")
    _, differing = compare_prepared_domain_config(
        effective_prepared_domain, live_root, not_in_use=not_in_use)
    return set(differing)


def _expected_root_cache_identity(
        identity: dict[str, object], *, root_domain,
        bridge_manifest_sha256: str, source_manifest_sha256: str,
        static_cache_sha256: str, forcing_hours,
) -> dict[str, object]:
    """Reconstruct the exact cache identity consumed by the hierarchy."""

    root_namelist_sha256, prepared_domain = \
        _validated_root_preparation_binding(identity, root_domain)
    invariant = identity.get("namelist_extension_invariant")
    if invariant is not None:
        invariant = validated_namelist_extension_invariant(
            invariant, context="root preparation identity")
    expected = prepared_cache_identity(
        bridge_manifest_sha256=bridge_manifest_sha256,
        source_manifest_sha256=source_manifest_sha256,
        static_cache_sha256=static_cache_sha256,
        namelist_sha256=root_namelist_sha256,
        domain_config=root_domain,
        forcing_hours=forcing_hours,
        source_identity=identity["source_identity"],
        namelist_extension_invariant=invariant,
    )
    # The sealed single-domain preparer stores legacy cadence shadow fields.
    # Their effective equality was checked above; preserve the exact sealed
    # document for cache verification.
    expected["domain_config"] = prepared_domain
    return expected


def _source_identity(cpu_bridge: Path) -> dict[str, object]:
    """What this install is, resolved the one way every consumer does.

    :func:`woof.runtime_manifest.provenance` is the single ladder
    ``woof doctor`` reports: the sealed manifest validated against its
    WHOLE schema, then a genuine checkout of THIS tree, then the
    installed wheel's ``RECORD`` digests.  Both branches this replaced
    were the field report's bugs in one function -- a manifest checked a
    key at a time (a document missing ``contract`` entirely passed here
    and died three consumers later), and an unguarded ``git rev-parse``
    rooted at the package directory, which for the pip install that runs
    this nested HRRR route is ``site-packages``.
    """

    root = Path(__file__).resolve().parent.parent
    identity = dict(runtime_manifest.provenance(root))
    identity["cpu_preprocess_bridge"] = {
        "path": str(cpu_bridge),
        "bytes": cpu_bridge.stat().st_size,
        "sha256": sha256_file(cpu_bridge),
    }
    # The hierarchy's own extra demand, unchanged: a checkout publishing
    # this route may have nothing uncommitted.  It belongs to the git
    # path alone -- a sealed manifest has already asserted
    # `source.worktree_clean`, and a wheel has no worktree to dirty.
    if (identity.get("identity_source") == "git"
            and identity.get("git_status_short")):
        raise RuntimeError(
            "public HRRR hierarchy requires a completely clean source tree")
    return identity


def _surface_state(restored, static_fields, *, sf_surface_physics, num_soil_layers=None,
                   fractional_seaice=False, seaice_threshold=100.0):
    if restored.surface is None:
        return preprocess_land_surface_soil(
            restored.met.fields,
            # real.exe's seaice_threshold rule (module_soil_pre.F:341-342).
            seaice_threshold=float(seaice_threshold),
            sf_surface_physics=int(sf_surface_physics),
            # real.exe keeps the sea-ice fraction under fractional_seaice = 1
            # (module_soil_pre.F:216-219, :337-343 of the HRRR v4.1.21 fork).
            fractional_seaice=bool(fractional_seaice),
            num_soil_layers=num_soil_layers,
            soil_type=static_fields["SCT_DOM"],
            deep_soil_temperature=static_fields["TMN"],
        )
    fields = restored.surface.fields
    return SimpleNamespace(
        tsk=fields["TSK"],
        soil_temperature=fields["TSLB"],
        soil_moisture=fields["SMOIS"],
        liquid_moisture=fields["SH2O"],
        deep_soil_temperature=fields["TMN"],
        xice=fields["SEAICE"],
        xland=fields["XLAND"],
        landmask=fields["LANDMASK"],
        snow_water=fields["SNOW"],
        snow_depth=fields["SNOWH"],
    )


def _root_paths(root: Path) -> dict[str, Path]:
    return {
        "static_cache": root / "native-static.npz",
        "static_receipt": root / "native-static-receipt.json",
        "prepared_cache": root / "native" / "prepared-cache",
        "bridge": root / "native" / "native-bridge",
        "bridge_manifest": root / "native" / "native-bridge" / "SHA256SUMS",
        "preparation_report": (
            root / "native" / "preparation-report" / "report.json"),
    }


def _highres_cache_sibling(output_root: Path) -> Path:
    """The fetch folder a join uses when it cannot write the recorded one.

    It sits beside the output root, outside the bundle, so publication and
    later rebuilds keep it without adding fetched tiles to artifact reuse
    hashing.  Its name is the output folder's first ten characters, a
    six-character hash of the whole name and ``.highres``: at most 25
    characters whatever the output is called.  Repeating the whole output
    name put the receipts written inside it past Windows' 259-character
    path limit under a deep output root (295 characters for a 125-character
    parent and a 92-character name), where writing them fails.  The hash
    is of the case-folded name, so a rebuild of the same output root finds
    the same folder, and reuses its tiles, however a case-insensitive file
    system is asked for it.
    """
    output_root = Path(output_root).absolute()
    name = output_root.name
    digest = hashlib.sha256(
        name.casefold().encode("utf-8", "surrogatepass")).hexdigest()[:6]
    return output_root.with_name(f"{name[:10]}-{digest}.highres")


def _local_highres_cache(config, *, recorded_cache_root, output_root):
    """Resolve a writable fetch folder without writing into a foreign path.

    When the join fetches into the sibling beside ``output_root``
    (:func:`_highres_cache_sibling`), the receipts it writes there are
    measured with the bundle first, by the same refusal the stage makes
    at its door, and a folder too deep for them is refused before the
    sibling is created: past Windows' path limit a receipt write fails
    partway through the join.
    """
    if config is None or not config.enabled:
        return config, None

    def writable(path):
        path.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryFile(dir=path):
            pass

    cache_root = config.cache_root
    substituted = not Path(recorded_cache_root).is_absolute()
    if not substituted:
        try:
            writable(cache_root)
        except OSError:
            substituted = True
    if substituted:
        cache_root = _highres_cache_sibling(output_root)
        from woof.static.highres_production import deepest_receipt_path
        refusal = published_path_refusal(
            output_root, also=(deepest_receipt_path(
                cache_root, config.terrain_source,
                config.landcover_source),))
        if refusal is not None:
            raise ValueError(refusal)
        try:
            writable(cache_root)
        except OSError as error:
            raise ValueError(
                "cannot prepare high-resolution child and corridor statics: "
                f"the fetch cache is not writable at {cache_root}") from error
    return replace(config, cache_root=cache_root), {
        "recorded_cache_root": recorded_cache_root,
        "cache_root": str(cache_root),
        "substituted": substituted,
    }


#: The seal document of a native HRRR domain tree, one-shot or chained.
RECEIPT_NAME = "receipt.json"


def chained_root_head(root_preparation: Path):
    """The live chained head of ``root_preparation``, or ``None``.

    ``None`` for a root preparation that is sealed (its ``proof.json``
    exists), published at its seal, or will never finish; the hierarchy is
    then built one-shot from the sealed root, as before.  A head whose root
    preparation is still building its boundary hours makes this tree chain
    too (:func:`_chained_hierarchy_tail`).
    """

    from woof.ingest.boundary_stream import live_chained_head

    return live_chained_head(Path(root_preparation))


def _head_cache_header(root_head) -> dict[str, object]:
    """A chained root's cache as far as its head knows it.

    The identity and the metadata the start time wrote (every key but the
    user metadata the seal completes is the sealed header's own, which
    ``verify_seal`` holds it to); there is no content digest yet.
    """

    cache = root_head["basis"]["cache"]
    return {"schema": "gpuwm-prepared-real-cache-v1", "status": "HEAD",
            "identity": cache["identity"], "metadata": cache["metadata"]}


_POSTED_TREE_SEAL_KEYS = ("provenance", "input_manifest_sha256", "posting")


def _root_source_binding(*, paths, source_manifest, source_manifest_sha256,
                         identity, root_head, root_preparation):
    """Bind the source documents, or exactly the live root's plan placeholders."""

    posted = (root_head or {}).get("basis", {}).get("as_posted")
    if posted is not None:
        from woof.ingest.boundary_stream import as_posted_placeholder

        placeholder = as_posted_placeholder(posted["input_plan_sha256"])
        keys = posted.get("document_bound_identity_keys") or {}
        if keys != {"bridge_manifest_sha256": "bridge",
                    "source_manifest_sha256": "source_manifest"}:
            raise ValueError("native HRRR root head lacks its document bindings")
        if (source_manifest_sha256 != placeholder
                or identity.get("source_manifest_sha256") != placeholder
                or identity.get("bridge_manifest_sha256") != placeholder):
            raise ValueError("native HRRR root head differs from its input plan")
        document = posted["documents"]["source_manifest"]
        if Path(source_manifest).resolve() != (
                Path(root_preparation) / document["path"]).resolve():
            raise ValueError("native HRRR source manifest differs from its root head")
        return placeholder, placeholder
    observed_source_sha = sha256_file(Path(source_manifest))
    if (observed_source_sha != source_manifest_sha256
            or identity.get("source_manifest_sha256") != observed_source_sha):
        raise ValueError("source manifest differs from the root preparation")
    bridge_sha = sha256_file(paths["bridge_manifest"])
    if identity.get("bridge_manifest_sha256") != bridge_sha:
        raise ValueError("HRRR bridge manifest differs from root preparation")
    return observed_source_sha, bridge_sha


def _posted_tree_head_inputs(root_head):
    """The root's input plan and start-lead bindings, relayed into the tree."""

    posted = root_head["basis"]["as_posted"]
    return {
        "input_plan": deepcopy(posted["input_plan"]),
        "start_marker_sha256": dict(posted["start_marker_sha256"]),
        "forcing_leads": list(posted["forcing_leads"]),
        "seal_authored_proof_keys": _POSTED_TREE_SEAL_KEYS,
        "manifest_path": str(posted["manifest_path"]),
        "lead_role_prefix": str(posted["lead_role_prefix"]),
        "derived_roles": tuple(posted.get("derived_roles") or ()),
        "manifest_bound_identity_keys": tuple(
            posted["manifest_bound_identity_keys"]),
        "document_bound_identity_keys": dict(
            posted["document_bound_identity_keys"]),
        "documents": deepcopy(posted["documents"]),
    }


def _settled_source_manifest_sha(root_preparation, source_manifest, requested):
    """Resolve a head's document placeholder if its root sealed before entry."""

    from woof.ingest.boundary_stream import (
        is_as_posted_placeholder, read_head, verify_seal)

    if not is_as_posted_placeholder(requested):
        return requested
    head = read_head(root_preparation)
    _root_source_binding(
        paths=_root_paths(Path(root_preparation)),
        source_manifest=source_manifest, source_manifest_sha256=requested,
        identity=head["basis"]["cache"]["identity"], root_head=head,
        root_preparation=root_preparation)
    verify_seal(root_preparation, head=head)
    return sha256_file(Path(source_manifest))


def _posted_start_snapshots(root_preparation, bridge, root_head, lead):
    """Map the start lead only after its root interval binds its decoded bytes."""

    from woof.ingest.boundary_stream import (
        StreamedIntervals, decoded_lead_record_sha256)
    from woof.ingest.hrrr import load_hrrr_pipeline_ready_window
    from tools.hrrr_single_domain_benchmark import _decoded_lead_record

    stream = StreamedIntervals(root_preparation, head=root_head)
    # Its head already admitted the first two source leads.  The first
    # interval's marker binds the decoded start snapshot we build children
    # from, without reading or waiting for the sealed bridge manifest.
    stream[0]
    marker = stream.consumed_markers()[0]
    def checked_snapshot(path):
        record = _decoded_lead_record(path, lead)
        if (marker.get("decoded_leads") or {}).get(str(int(lead))) \
                != decoded_lead_record_sha256(record):
            raise ValueError("native HRRR start snapshot differs from its root interval")
        return load_hrrr_pipeline_ready_window(path, lead)

    path = _posted_start_bridge(root_preparation, bridge)
    try:
        snapshots = (checked_snapshot(path),)
    except FileNotFoundError:
        # The decoder removes its retained staging after publishing the
        # canonical bridge.  If it sealed between choosing that staging
        # and mapping the snapshot, the canonical copy binds the same
        # decoded record and can be read instead.
        if path == Path(bridge) or not Path(bridge).is_dir():
            raise
        snapshots = (checked_snapshot(Path(bridge)),)
    stream.release(0)
    return snapshots


def _posted_start_bridge(root_preparation, bridge):
    """The root decoder's own retained staging, or its published bridge."""

    from tools.hrrr_pipeline import _path_is_reparse_point, _read_tsv

    bridge = Path(bridge)
    if bridge.is_dir():
        return bridge
    receipt = _read_tsv(Path(root_preparation)
                        / "native/pipeline-signals/preflight.ready")
    staging = Path(receipt.get("staging_root", ""))
    prefix = f".{bridge.name}.partial-"
    pid = staging.name.removeprefix(prefix)
    if (receipt.get("status") != "PASS"
            or receipt.get("staging_retention") != "until_consumer_finish"
            or Path(receipt.get("canonical_output", "")).resolve()
            != bridge.resolve()
            or not staging.name.startswith(prefix) or not pid.isdigit()
            or int(pid) < 1
            or staging.resolve().parent != bridge.resolve().parent
            or _path_is_reparse_point(staging) or not staging.is_dir()):
        if bridge.is_dir() and receipt.get("status") == "PASS" \
                and Path(receipt.get("canonical_output", "")).resolve() \
                == bridge.resolve():
            return bridge
        raise ValueError("native HRRR decoder staging differs from its root bridge")
    return staging


def _copy_posted_root_inputs(root_preparation, destination, root_head,
                             sealed_root):
    """Copy the documents whose bytes the root's completed seal verified."""

    from woof.ingest.boundary_stream import document_bound_digests

    posted = root_head["basis"]["as_posted"]
    source = Path(root_preparation)
    manifest_path = str(posted["manifest_path"])
    names = [manifest_path, *(str(document["path"])
                              for document in posted["documents"].values())]
    for name in names:
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("native HRRR input document leaves its prepared bundle")
        target = Path(destination) / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / relative, target)
    manifest = _json(Path(destination) / manifest_path)
    digest = sha256_file(Path(destination) / manifest_path)
    if digest != sealed_root["as_posted"]["input_manifest_sha256"]:
        raise ValueError("native HRRR input manifest changed after its root seal")
    documents = document_bound_digests(posted, manifest)
    for key, role in posted["document_bound_identity_keys"].items():
        path = Path(destination) / posted["documents"][role]["path"]
        if sha256_file(path) != documents[key]:
            raise ValueError("native HRRR input document changed after its root seal")
    return digest, documents


def _relay_root_wait(writer, report):
    """A tree waiting on its root names the same source lead on its heartbeat."""

    cause = None if report is None else report.get("cause")
    if isinstance(cause, Mapping) and cause.get("on") == "source":
        writer.waiting_for_source(cause)
    else:
        writer.source_arrived()


def _require_root_preparation_report(path: Path, *, identity,
                                     content_sha256) -> None:
    """The root preparation's report passed and binds its sealed cache."""

    preparation_report = _json(path)
    if preparation_report.get("status") != "PASS":
        raise ValueError("root preparation report is not PASS")
    if preparation_report.get("source_identity") != identity.get("source_identity"):
        raise ValueError("root preparation source identity differs from cache")
    prepared_report = preparation_report.get("prepared_cache", {})
    if prepared_report.get("content_sha256") != content_sha256:
        raise ValueError("root preparation report does not bind its cache")


def _root_preparation_record(content_sha256: str) -> dict[str, object]:
    """The receipt's record of the root preparation's sealed cache.

    A seal-only proof key (``root_preparation`` in
    :data:`woof.ingest.boundary_stream.SEAL_ONLY_PROOF_KEYS`): a chained
    tree's head is written before its root preparation seals.
    """

    return {"prepared_content_sha256": str(content_sha256)}


def _root_preparation_metadata(content_sha256: str | None
                               ) -> dict[str, object]:
    """d01's user metadata naming the root preparation it was joined to.

    ``None`` is a chained tree's head, which holds the mapping empty and
    names it ``seal_completes``; its seal adds the digest, so the sealed
    header is the one-shot writer's.
    """

    return {"root_preparation": ({} if content_sha256 is None
                                 else _root_preparation_record(
                                     content_sha256))}


def _export_provenance(provenance, content_sha256: str) -> dict[str, object]:
    """The companion WRF export's input provenance: the receipt's, plus the root's seal."""

    return {**dict(provenance),
            "root_prepared_content_sha256": str(content_sha256)}


def _aerosol_entry(cache_header, native_exp, paths) -> dict[str, object]:
    return aerosol_source_report_entry(
        cache_header.get("metadata", {}).get(AEROSOL_SOURCE_KEY, {}),
        mp_physics=native_exp.root.run.mp_physics,
        when_unrecorded=(
            "the root prepared cache at "
            f"{paths['prepared_cache']} carries no "
            "aerosol-initialization receipt, so it was written by "
            "a preparation predating the receipt being stored; "
            "re-prepare the root to record which source filled "
            "its nwfa/nifa"))


def _interval_host_bytes(interval) -> int:
    """What one written boundary interval holds in host RAM.

    Its value and its tendency, each a float64 copy of the four sides
    (tools/hrrr_single_domain_benchmark.py ``_interval_host_pricing``).
    """

    return int(sum(
        (np.asarray(side.value).size + np.asarray(side.tendency).size)
        * np.dtype(np.float64).itemsize
        for boundary in interval.fields.values()
        for side in (boundary.west, boundary.east, boundary.south,
                     boundary.north)))


def _chained_hierarchy_tail(c) -> dict[str, object]:
    """A native HRRR domain tree, chained on its root preparation's head.

    The head is published as soon as the children exist: the root's static
    files and its start state (a streamed cache under
    ``hierarchy-head/domains/d01/prepared-cache``, header at the seal),
    every child's complete artifact set under ``hierarchy-head/domains/dNN``
    (a child needs only its start-time snapshot and the root's start state,
    :func:`woof.native_hierarchy.initialize_native_hierarchy_children`) and
    a moving nest's statics corridor under ``hierarchy-head/``.  Segment k
    of d01 is the root preparation's boundary interval k, relayed as the
    root preparation writes it.  Once the root preparation seals (held to
    the head it published, :func:`woof.ingest.boundary_stream.verify_seal`)
    the seal re-reads the children from the head and writes the one-shot
    ``hierarchy-artifacts/`` tree from the root preparation's sealed cache,
    through the same writer and the same arguments as the one-shot tree,
    then ``receipt.json`` last.  The root's sealed content digest is the
    one fact the head cannot hold: ``root_preparation`` in the receipt (a
    seal-only key) and d01's ``root_preparation`` user metadata (completed
    at the seal).
    """

    from woof.ingest.boundary_stream import (
        HIERARCHY_HEAD_DIRNAME, PreparedTreeWriter, StreamedIntervals,
        TreeStartStates, domain_tree_head_fields, verify_seal)
    from woof.ingest.water_overlay import verify_overlay_sequence
    from woof.native_domain_artifacts import (
        root_domain_artifact_binding, write_child_domain_artifacts,
        write_domain_static_files)
    from woof.native_hierarchy import (
        export_native_hierarchy, initialize_native_hierarchy_children)
    from woof.native_wrf_contract import canonical_noah_surface
    from woof.progress import prep_progress
    from woof.static.corridor import copy_statics_corridor_set

    exp = c.export_exp
    staging = c.staging
    posted = c.root_head["basis"].get("as_posted")
    writer = None
    try:
        head_started = time.perf_counter()
        # The corridor is sealed BEFORE the children initialize, so each
        # corridor child is born on its crop (plan call 5, 2.9.0); it needs
        # no boundary time, so a moving nest's forecast re-grounds over the
        # head's copy from its first move, and the seal copies it into
        # hierarchy-artifacts/ (A136 L7d).
        corridor_directory = (staging / HIERARCHY_HEAD_DIRNAME
                              / STATICS_CORRIDOR_DIRNAME)
        corridor_receipt = emit_statics_corridor_set(
            exp=c.lifecycle_exp, grids=c.grids, static_catalog=c.static_catalog,
            directory=corridor_directory,
            statics_corridor=c.statics_corridor,
            static_highres=c.static_highres)
        births = corridor_birth_statics(
            corridor_directory, corridor_receipt, c.lifecycle_exp)
        child_results, child_seconds = initialize_native_hierarchy_children(
            exp=exp, root_node=c.root,
            catalog=catalog_with_corridor_births(c.catalog, births),
            root_initial_result=c.restored.initial_result,
            workers=c.workers, preprocess_backend="cpu",
            cpu_bridge=c.cpu_bridge,
            sfcp_to_sfcp=c.case_policy["sfcp_to_sfcp"])
        head_domains = staging / HIERARCHY_HEAD_DIRNAME / "domains"
        root_directory = head_domains / "d01"
        root_directory.mkdir(parents=True)
        root_static_receipt, _geometry = write_domain_static_files(
            root_directory, domain=exp.domains[0], grid=c.grids[0],
            static_fields=c.static_fields)
        child_builds = write_child_domain_artifacts(
            head_domains, exp=exp, child_results=child_results,
            bridge_manifest_sha256=c.bridge_sha,
            source_manifest_sha256=c.observed_source_sha,
            namelist_sha256=c.namelist_sha, forcing_hours=c.forcing_hours,
            source_identity=c.hierarchy_source_identity,
            valid_time=exp.start_time)
        binding = root_domain_artifact_binding(
            exp=exp, static_cache_sha256=root_static_receipt["sha256"],
            bridge_manifest_sha256=c.bridge_sha,
            source_manifest_sha256=c.observed_source_sha,
            namelist_sha256=c.namelist_sha, forcing_hours=c.forcing_hours,
            source_identity=c.hierarchy_source_identity,
            valid_time=exp.start_time,
            root_metadata=_root_preparation_metadata(None))
        stock_copy = staging / "namelist.input"
        shutil.copyfile(c.stock_wrf_namelist_input, stock_copy)
        proof_head = {
            "schema": SCHEMA,
            "status": "PASS",
            "valid_time": c.valid_time.isoformat(),
            "workers": c.workers,
            "preprocess_backend": "cpu",
            "domain_count": len(c.native_exp.domains),
            "forcing_hours": list(c.forcing_hours),
            "provenance": c.provenance,
            "stock_wrf_namelist": {
                "path": stock_copy.name,
                "sha256": sha256_file(stock_copy),
            },
            **({"initial_perturbation": dict(c.root_deferral)}
               if c.root_deferral is not None else {}),
            **({"statics_corridor": dict(corridor_receipt)}
               if corridor_receipt is not None else {}),
            **_aerosol_entry(c.cache_header, c.native_exp, c.paths),
        }
        if posted is not None:
            proof_head.pop("provenance")
        cache_name = f"{HIERARCHY_HEAD_DIRNAME}/domains/d01/prepared-cache"
        writer = PreparedTreeWriter(
            staging=staging, output_root=c.output_root,
            identity=binding.identity, cache_name=cache_name,
            proof_name=RECEIPT_NAME)
        writer.admit(experiment=c.native_exp, backend="cpu", source="hrrr")
        root_lbc = dict(c.root_head["basis"]["cache"]["lbc"])
        writer.write_head(
            initial_result=c.restored.initial_result, met=c.restored.met,
            surface=canonical_noah_surface(c.root_soil),
            metadata=dict(binding.metadata), lbc=root_lbc,
            proof_head=proof_head,
            input_manifest_sha256=(None if posted is not None
                                   else c.observed_source_sha),
            forcing=SimpleNamespace(interval_host_bytes=_interval_host_bytes(
                c.restored.boundaries.intervals[0])),
            tree=domain_tree_head_fields(
                [f"d{int(domain.grid_id):02d}" for domain in exp.domains],
                root_cache=cache_name,
                children_receipts={
                    f"d{int(build.receipt['grid_id']):02d}": sha256_file(
                        head_domains
                        / f"d{int(build.receipt['grid_id']):02d}"
                        / "receipt.json")
                    for build in child_builds}),
            extra_head_payload_bytes=sum(
                int(build.receipt["artifacts"]["prepared_cache"][
                    "payload_bytes"]) for build in child_builds),
            seal_completes=("root_preparation",),
            as_posted=(None if posted is None
                       else _posted_tree_head_inputs(c.root_head)))
        head_seconds = time.perf_counter() - head_started
        # Every start state is in the head now; none stays resident while
        # the root preparation builds its later hours.
        start_states = TreeStartStates.release(
            root_result=c.restored.initial_result, root_met=c.restored.met,
            child_results=child_results,
            child_content_sha256={
                f"d{int(build.receipt['grid_id']):02d}": build.receipt[
                    "artifacts"]["prepared_cache"]["content_sha256"]
                for build in child_builds})
        del child_results
        c.restored = c.root = c.root_soil = None

        # d01's boundary intervals: the root preparation's, as it writes
        # them, each checked against its segment marker on the way.
        relay_started = time.perf_counter()
        root_stream = StreamedIntervals(
            c.root_preparation, head=c.root_head,
            start_time=c.native_exp.start_time,
            on_wait=(None if posted is None else
                     lambda report: _relay_root_wait(writer, report)))
        count = len(root_stream)
        for k in range(count):
            built = time.perf_counter()
            interval = root_stream[k]
            if posted is not None:
                # Keep the verified lead provenance beside a nested prefix,
                # so a fresh root producer retains its original head binding.
                from woof.ingest.stream_resume import POSTED_PREFIX_DIRNAME
                source_prefix = (Path(c.root_preparation) / "boundary-stream"
                                 / POSTED_PREFIX_DIRNAME)
                target_prefix = writer.stream_path / POSTED_PREFIX_DIRNAME
                target_prefix.mkdir(parents=True, exist_ok=True)
                for marker_path in source_prefix.glob("f*.json"):
                    shutil.copyfile(marker_path, target_prefix / marker_path.name)
            writer.write_segment(k, interval, **({} if posted is None else {
                "relay_marker": root_stream.consumed_markers()[k]}))
            root_stream.release(k)
            del interval
            writer.note_build_seconds(time.perf_counter() - built)
            prep_progress("hierarchy_boundaries",
                          label="Root boundary intervals", done=k + 1,
                          count=count)
        root_stream.wait_sealed()
        relay_seconds = time.perf_counter() - relay_started

        # The root preparation's seal, held to the head this tree was
        # built on, then the checks the one-shot tree makes up front.
        seal_started = time.perf_counter()
        sealed_root = verify_seal(c.root_preparation, head=c.root_head,
                                  consumed=root_stream.consumed_markers())
        root_content = str(sealed_root["content_sha256"])
        root_header = _json(c.paths["prepared_cache"] / "header.json")
        if (root_header.get("schema") != "gpuwm-prepared-real-cache-v1"
                or root_header.get("status") != "READY"
                or root_header.get("content_sha256") != root_content):
            raise ValueError("root preparation cache is not READY")
        _require_root_preparation_report(
            c.paths["preparation_report"], identity=root_header["identity"],
            content_sha256=root_content)
        seal_arguments = {}
        if posted is not None:
            from woof.ingest.boundary_stream import (
                POSTED_LEADS_NAME, stream_dir)

            manifest_digest, documents = _copy_posted_root_inputs(
                c.root_preparation, writer.root, c.root_head, sealed_root)
            c.bridge_sha = documents["bridge_manifest_sha256"]
            c.observed_source_sha = documents["source_manifest_sha256"]
            c.expected_identity = _expected_root_cache_identity(
                root_header["identity"], root_domain=c.native_exp.root,
                bridge_manifest_sha256=c.bridge_sha,
                source_manifest_sha256=c.observed_source_sha,
                static_cache_sha256=c.expected_identity["static_cache_sha256"],
                forcing_hours=c.forcing_hours)
            c.provenance = {
                **c.provenance, "bridge_manifest_sha256": c.bridge_sha,
                "source_manifest_sha256": c.observed_source_sha}
            lead_records = _json(stream_dir(c.root_preparation)
                                 / POSTED_LEADS_NAME)
            writer.write_posted_leads(
                {int(lead): row["marker"]
                 for lead, row in lead_records["leads"].items()},
                route_table_sha256=lead_records["route_table_sha256"],
                decoded={int(lead): row["decoded"]
                         for lead, row in lead_records["leads"].items()})
            sealed_binding = root_domain_artifact_binding(
                exp=exp, static_cache_sha256=root_static_receipt["sha256"],
                bridge_manifest_sha256=c.bridge_sha,
                source_manifest_sha256=c.observed_source_sha,
                namelist_sha256=c.namelist_sha, forcing_hours=c.forcing_hours,
                source_identity=c.hierarchy_source_identity,
                valid_time=exp.start_time,
                root_metadata=_root_preparation_metadata(None))
            seal_arguments = {
                "identity": sealed_binding.identity,
                "manifest_sha256": manifest_digest,
                "document_sha256": documents}
            proof_head = {
                **proof_head, "provenance": c.provenance,
                "input_manifest_sha256": manifest_digest,
                "posting": deepcopy(_json(
                    c.root_preparation / "proof.json")["posting"])}
        tree_root_content = str(writer.seal_cache(
            completed_metadata={
                "root_preparation": _root_preparation_record(root_content)},
            **seal_arguments)["content_sha256"])
        root_reader = PreparedCacheReader(
            c.paths["prepared_cache"], expected_identity=c.expected_identity)
        if root_reader.verify_all().get("content_sha256") != root_content:
            raise ValueError(
                "the root preparation's sealed cache is not the one its "
                "seal check read")
        restored = restore_prepared_cache(
            c.paths["prepared_cache"], expected_identity=c.expected_identity,
            cfg=c.native_exp.root.run, static=c.static_fields,
            array_module=np)
        root_soil = _surface_state(
            restored, c.static_fields,
            sf_surface_physics=c.native_exp.root.run.sf_surface_physics,
            num_soil_layers=c.native_exp.root.run.num_soil_layers,
            fractional_seaice=int(getattr(c.native_exp.root.run, "fractional_seaice", 0)) == 1,
            seaice_threshold=float(getattr(c.native_exp.root.run, "seaice_threshold", 100.0)))
        children = start_states.reread_children(writer.root, exp=exp)
        result = export_native_hierarchy(
            exp=exp,
            root_node=ParentInitView(cfg=c.native_exp.root, grid=c.grids[0],
                                     state=restored.initial_result.state),
            artifact_output=writer.root / "hierarchy-artifacts",
            wrf_output=writer.root / "wrf-native-input",
            root_initial_result=restored.initial_result,
            root_met=restored.met, root_soil=root_soil,
            root_static_fields=c.static_fields,
            root_boundaries=restored.boundaries, child_results=children,
            child_initialization_seconds=child_seconds,
            bridge_manifest_sha256=c.bridge_sha,
            source_manifest_sha256=c.observed_source_sha,
            namelist_sha256=c.namelist_sha, forcing_hours=c.forcing_hours,
            source_identity=c.hierarchy_source_identity,
            boundary_interval_seconds=3600,
            root_metadata=_root_preparation_metadata(root_content),
            input_provenance=_export_provenance(c.provenance, root_content),
            artifact_manifest_reference=(
                "../hierarchy-artifacts/domain-artifacts.json"),
            stock_wrf_export=c.stock_wrf_export)
        start_states.require_sealed_is_head(
            result.artifacts.receipt, root_content_sha256=tree_root_content,
            **({} if posted is None else {"as_posted": {
                "root": writer.root, "head": writer.head,
                "manifest_sha256": manifest_digest}}))
        verify_overlay_sequence(c.snapshots)
        if corridor_receipt is not None:
            copy_statics_corridor_set(
                writer.root / HIERARCHY_HEAD_DIRNAME
                / STATICS_CORRIDOR_DIRNAME,
                writer.root / "hierarchy-artifacts" / STATICS_CORRIDOR_DIRNAME,
                receipt=corridor_receipt)
        payload = {
            **proof_head,
            "root_preparation": _root_preparation_record(root_content),
            "timing_seconds": {
                "verified_preflight": c.preflight_seconds,
                "restore_root_prepared_head": c.restore_seconds,
                "load_initial_snapshot_and_static_catalog": (
                    c.snapshot_seconds),
                **dict(result.timings_seconds),
                "prepared_head": head_seconds,
                "head_published_after": writer.head_seconds,
                "relay_root_boundary_intervals": relay_seconds,
                "hierarchy_seal": time.perf_counter() - seal_started,
                "total": time.perf_counter() - c.started,
            },
            "artifact_receipt": dict(result.artifacts.receipt),
            "wrf_manifest": dict(result.wrf_manifest),
            "boundary_stream": writer.boundary_stream_proof(),
        }
        writer.publish(payload)
    except BaseException as error:
        if writer is not None:
            # A head already published stays, marked failed, so a waiting
            # forecast ends with this reason; the caller removes a staging
            # tree that was never published.
            writer.fail(error)
        raise
    return payload


@preprocess_math_call(fixed_backend="cpu")
def prepare_hrrr_hierarchy(
        *, root_preparation: Path, root_domain_spec: Path,
        wps_namelist: Path, namelist_input: Path,
        stock_wrf_namelist_input: Path, geog_root: Path,
        source_manifest: Path, source_manifest_sha256: str,
        valid_time: datetime, output_root: Path, workers: int = 8,
        cpu_bridge: Path | None = None, statics_corridor=None,
        acknowledgements: tuple[str, ...] = (),
        lifecycle_config: Path | None = None,
        stock_wrf_export: str = "optional",
) -> dict[str, object]:
    """Verify one root preparation and publish generic stock-WRF inputs.

    ``stock_wrf_export`` is the companion unchanged-WRF file set's mode:
    ``"optional"`` (a state the WRF format cannot represent is recorded as
    a REFUSED export manifest) or ``"off"`` (``woof prep
    --no-stock-wrf-export``: not attempted, NOT_REQUESTED in the receipt).

    ``statics_corridor`` opts this stage into sealing child-resolution
    statics over the ground each child can reach
    (:mod:`woof.static.corridor`): ``None`` emits nothing and leaves
    the bundle byte-for-byte unchanged, ``"all"`` covers every child
    domain, and a sequence of grid ids covers exactly those children.

    This is the HRRR chain's corridor door, and it is HERE rather than
    in ``tools/prepare_hrrr_wrf`` for one reason: a corridor is
    child-resolution statics, and this is the stage that knows the
    children exist.  The root preparer prepares d01 alone and never
    reads a child geometry; this stage takes ``--geog-root``, builds
    d02..dNN, and already holds the verified GEOG catalog every child's
    statics come from.  The set receipt is bound into ``receipt.json``
    beside ``artifact_receipt``, exactly as the GFS chain binds it into
    ``proof.json``, which is what makes the sealed artifact rather than
    any printed line the contract the tree runner verifies.
    """

    if isinstance(workers, bool) or workers not in range(1, 33):
        raise ValueError("workers must be an integer from 1 through 32")
    output_root = Path(output_root)
    if output_root.exists():
        raise FileExistsError(f"refusing existing output root {output_root}")
    refusal = published_path_refusal(output_root)
    if refusal is not None:
        raise ValueError(refusal)
    cpu_bridge = resolve_cpu_bridge(cpu_bridge)
    paths = _root_paths(Path(root_preparation))
    # A root preparation still building its boundary hours beside a
    # published head: this tree then chains too (_chained_hierarchy_tail).
    # Its preparation report is written at the root's seal, so it is
    # checked there.
    root_head = chained_root_head(Path(root_preparation))
    posted = (root_head or {}).get("basis", {}).get("as_posted")
    if root_head is None:
        source_manifest_sha256 = _settled_source_manifest_sha(
            root_preparation, source_manifest, source_manifest_sha256)
    required = (
        root_domain_spec, wps_namelist, namelist_input,
        stock_wrf_namelist_input, geog_root, cpu_bridge,
        *((source_manifest,) if posted is None else ()),
        *(path for key, path in paths.items()
          if (root_head is None or key != "preparation_report")
          and (posted is None or key not in {"bridge", "bridge_manifest"})),
    )
    for path in required:
        if not Path(path).exists():
            raise FileNotFoundError(path)

    started = time.perf_counter()
    stock_runtime_delta = _require_raw_stock_delta(
        Path(namelist_input), Path(stock_wrf_namelist_input))
    # The sealed root preparation is the physics and forcing authority,
    # so read its identity BEFORE importing the namelists: the namelist
    # spells radiation as selector integers, and which 4/4 implementation
    # serves them is the sealed profile's recorded decision
    # (_sealed_root_rrtmg_variant), which the import must inherit for the
    # d01 binding below to compare like with like.  The forcing inventory
    # is read here for the same reason -- the duration ceiling is a
    # property of this bundle rather than a constant.
    if root_head is None:
        cache_header = _json(paths["prepared_cache"] / "header.json")
        if (cache_header.get("schema") != "gpuwm-prepared-real-cache-v1"
                or cache_header.get("status") != "READY"):
            raise ValueError("root preparation cache is not READY")
    else:
        # The head's cache record carries the identity and the metadata
        # the start time made; the header written at the root's seal must
        # equal them (verify_seal), and is checked there.
        cache_header = _head_cache_header(root_head)
    identity = cache_header.get("identity")
    if not isinstance(identity, dict):
        raise ValueError("root preparation cache lacks an identity")
    from woof.case_data import preparation_case_policy
    case_policy = identity.get("source_identity", {}).get("preparation_case_policy")
    if case_policy is None:
        # Older sealed roots used the documented true/default-water operands.
        case_policy = preparation_case_policy(None)
    if type(case_policy.get("sfcp_to_sfcp")) is not bool:
        raise ValueError("sealed root preparation lacks a boolean sfcp_to_sfcp policy")
    if stock_runtime_delta["certified_native_runtime"]["domains.sfcp_to_sfcp"] != [case_policy["sfcp_to_sfcp"]]:
        raise ValueError("hierarchy sfcp_to_sfcp differs from the sealed root preparation")
    forcing_hours = tuple(identity.get("forcing_hours", ()))
    native_exp, native_resolved, native_report = _native_experiment(
        Path(wps_namelist), Path(namelist_input),
        rrtmg_variant=_sealed_root_rrtmg_variant(identity),
        rrtmg_compatibility=_sealed_root_rrtmg_compatibility(identity),
        acknowledgements=tuple(acknowledgements))
    if lifecycle_config is not None:
        from woof.experiment import load_experiment
        requested_nudging=load_experiment(lifecycle_config).nudging
        if requested_nudging is not None:
            from woof.toml_document import emit_experiment_toml
            native_exp=replace(native_exp,nudging=requested_nudging)
            native_tables=tomllib.loads(native_resolved)
            native_tables['nudging']={'domain':{str(gid):{k:v for k,v in control.to_dict().items() if v is not None}
                for gid,control in requested_nudging.domains}}
            native_resolved=emit_experiment_toml(native_tables)
    if getattr(native_exp, "nudging", None) is not None and root_head is not None:
        raise ValueError('NUDGING_TREE_ANALYSIS_SEAL: a start-only root head does not contain all selected nests\' analysis times; finish native source preparation before the tree forecast')
    # The lifecycle tables reach ONLY the statics corridor (a dormant slot's
    # corridor spans the whole parent).  Every document this stage publishes
    # or compares stays the namelists' own: the forecast companion
    # experiment.toml must reload to the namelist identity it was rendered
    # from, which a spawn table it cannot spell broke (2.9.4 rolling,
    # a 4090 host: ExperimentDocumentError).
    lifecycle_exp = (native_exp if lifecycle_config is None
                     else with_config_lifecycle(native_exp, Path(lifecycle_config)))
    from woof.static.highres_production import parse_sealed_static_highres
    declared_highres = identity.get("source_identity", {}).get("static_highres")
    # The echo carries the nests' own terrain smoothing and the urban
    # land-cover legend beside the [static.highres] keys; all are read, or
    # the children are rebuilt with terrain or land cover the sealed
    # configuration did not ask for.  The legend is held to the tree's.
    static_highres = parse_sealed_static_highres(
        declared_highres, source="sealed root preparation",
        base_dir=Path(root_preparation), run_config=native_exp.root.run)
    target = load_hrrr_target_domain(root_domain_spec)
    _supported_hierarchy_slice(
        native_exp, target, forcing_hours=forcing_hours)
    # Resolved the moment a domain tree exists, and long before the
    # expensive restore/decode/join below: a corridor asked of a
    # single-domain namelist, or naming a grid id this tree does not
    # have, is a typed flag rather than a failure of the preparation,
    # and refusing it after the join would throw away work that
    # succeeded.  The emission re-resolves the same pure function.
    if statics_corridor is not None and len(native_exp.domains) < 2:
        raise ValueError(
            "--statics-corridor seals child-resolution statics over a "
            "parent extent, and this namelist has no child domain; "
            "remove the flag or prepare a domain tree")
    validated_corridor_selection(native_exp, statics_corridor)
    wps_runtime_contract = _require_raw_wps_contract(
        Path(wps_namelist), len(native_exp.domains))
    if native_exp.start_time != valid_time:
        raise ValueError(namelist_start_refusal(
            requested=valid_time, observed=native_exp.start_time,
            namelist_name=Path(namelist_input).name))

    static_fields, static_receipt = verify_hrrr_native_static(
        paths["static_cache"], paths["static_receipt"], target)
    from woof.static.highres_production import require_prepared_highres
    if static_highres is not None and static_highres.enabled:
        require_prepared_highres(static_receipt, target.grid(), config=static_highres,
                                 domain_id=1, case_date=native_exp.start_time.date())
    observed_source_sha, bridge_sha = _root_source_binding(
        paths=paths, source_manifest=source_manifest,
        source_manifest_sha256=source_manifest_sha256,
        identity=identity, root_head=root_head,
        root_preparation=root_preparation)
    static_sha = sha256_file(paths["static_cache"])
    if identity.get("static_cache_sha256") != static_sha:
        raise ValueError("static cache differs from root preparation identity")
    verified_root_forcing_inventory(
        forcing_hours, run_seconds=native_exp.run_seconds)

    if root_head is None:
        _require_root_preparation_report(
            paths["preparation_report"], identity=identity,
            content_sha256=cache_header.get("content_sha256"))
    user_metadata = cache_header.get("metadata", {}).get("user", {})
    sealed_start = datetime.fromisoformat(
        user_metadata.get("initial_valid_time", ""))
    if sealed_start != valid_time:
        raise ValueError(sealed_start_refusal(
            requested=valid_time, sealed_start=sealed_start,
            sealed_cycle=user_metadata.get("source_cycle")))

    expected_identity = _expected_root_cache_identity(
        identity, root_domain=native_exp.root,
        bridge_manifest_sha256=bridge_sha,
        source_manifest_sha256=observed_source_sha,
        static_cache_sha256=static_sha,
        forcing_hours=forcing_hours,
    )
    root_namelist_sha = expected_identity["namelist_sha256"]
    namelist_sha = sha256_file(Path(namelist_input))
    if root_head is None:
        reader = PreparedCacheReader(
            paths["prepared_cache"], expected_identity=expected_identity)
    else:
        # The start-time arrays, each held to the head's row; the cache
        # content digest exists at the root's seal and is checked there.
        reader = PreparedHeadReader(
            Path(root_preparation), root_head,
            expected_identity=expected_identity)
    reader.verify_all()
    # Before the staging folder exists: a fetch folder beside the output
    # root too deep for its receipts is refused here with nothing written.
    static_highres, highres_cache = _local_highres_cache(
        static_highres,
        recorded_cache_root=(declared_highres or {}).get("cache_root"),
        output_root=output_root)
    cache_provenance = ({"static_highres_cache": highres_cache}
                        if highres_cache is not None else {})
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = _atomic_staging_sibling(output_root)
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir()
    try:
        publish_hierarchy_forecast_companions(
            staging, resolved_experiment=native_resolved,
            experiment=native_exp, wps_namelist=wps_namelist,
            perturbation_deferral=identity.get("source_identity", {}).get("initial_perturbation"))
        preflight_seconds = time.perf_counter() - started

        restore_started = time.perf_counter()
        # A host state: this stage prepares on the CPU (see
        # preprocess_backend="cpu" below) and never integrates the root,
        # so it must not need CUDA to read the root it builds from.
        if root_head is None:
            restored = restore_prepared_cache(
                paths["prepared_cache"], expected_identity=expected_identity,
                cfg=native_exp.root.run, static=static_fields,
                array_module=np)
        else:
            # The root's start state from its head; its boundary intervals
            # stream from the root preparation as it writes them, and the
            # children read none of them.
            from woof.ingest.boundary_stream import streamed_boundaries

            restored = restore_prepared_cache(
                paths["prepared_cache"], expected_identity=expected_identity,
                cfg=native_exp.root.run, static=static_fields, reader=reader,
                boundary_source=streamed_boundaries(
                    Path(root_preparation), head=root_head),
                array_module=np)
        root_soil = _surface_state(
            restored, static_fields,
            sf_surface_physics=native_exp.root.run.sf_surface_physics,
            num_soil_layers=native_exp.root.run.num_soil_layers,
            fractional_seaice=int(getattr(native_exp.root.run, "fractional_seaice", 0)) == 1,
            seaice_threshold=float(getattr(native_exp.root.run, "seaice_threshold", 100.0)))
        restore_seconds = time.perf_counter() - restore_started

        snapshots_started = time.perf_counter()
        source_leads = sealed_source_leads(identity, forcing_hours)
        nudged_children=(getattr(native_exp, "nudging", None) is not None and any(
            native_exp.nudging.for_domain(d.grid_id) is not None for d in native_exp.domains[1:]))
        snapshots = (_posted_start_snapshots(
            Path(root_preparation), paths["bridge"], root_head, source_leads[0])
            if posted is not None else load_hrrr_native_series(
                paths["bridge"], source_leads if nudged_children else source_leads[:1],
                expected_manifest_sha256=bridge_sha))
        from woof.ingest.water_overlay import (
            load_bound_water_overlay, overlay_snapshot_sequence, verify_overlay_sequence)
        water_overlay, water_binding = load_bound_water_overlay(
            case_policy.get("water_temperature_overlay"))
        sealed_water_binding = identity.get("source_identity", {}).get("water_temperature_overlay")
        if water_binding != sealed_water_binding:
            raise ValueError("water overlay differs from the sealed root preparation")
        snapshots = overlay_snapshot_sequence(
            snapshots, water_overlay, binding=water_binding, workers=workers)
        static_catalog, catalog_receipt = verified_static_catalog(
            Path(wps_namelist), Path(geog_root),
            [domain.grid_id for domain in native_exp.domains],
            **selection_carrier_kwargs(static_highres))
        if catalog_receipt["selections"]["d01"] != static_receipt.get(
                "geog_selection"):
            raise ValueError(
                "WPS d01 GEOG selection differs from the sealed root static "
                "selection")
        catalog = NestedInputCatalog(
            snapshots=snapshots, static_catalog=static_catalog,
            water_temperature_policy=case_policy["water_temperature_policy"],
            files=tuple(static_catalog.files), static_highres=static_highres,
            provenance={
                "adapter": "native-HRRR-hierarchy-direct-v1",
                "bridge_manifest_sha256": bridge_sha,
                "static_catalog_receipt": catalog_receipt,
                **cache_provenance,
                "surface_fallback_radius_cells": (
                    target.surface_fallback_radius_cells),
            })
        snapshot_seconds = time.perf_counter() - snapshots_started

        grids = tuple(grids_from_projection_config(native_exp))
        root = ParentInitView(
            cfg=native_exp.root, grid=grids[0], state=restored.initial_result.state)
        hierarchy_identity = _source_identity(cpu_bridge)
        provenance = {
            "source_manifest_sha256": observed_source_sha,
            "bridge_manifest_sha256": bridge_sha,
            "root_static_receipt_sha256": sha256_file(paths["static_receipt"]),
            # The root's sealed cache content digest is not here: a chained
            # tree's head is written before the root seals, so the receipt
            # carries it in ``root_preparation``, which only the seal writes
            # (_root_preparation_record), and the companion WRF export,
            # written at the seal too, keeps it in its input provenance.
            "root_preparation_namelist_sha256": root_namelist_sha,
            "wps_namelist_sha256": sha256_file(Path(wps_namelist)),
            "native_namelist_input_sha256": namelist_sha,
            "stock_wrf_namelist_input_sha256": sha256_file(
                Path(stock_wrf_namelist_input)),
            "native_resolved_experiment_sha256": hashlib.sha256(
                native_resolved.encode("utf-8")).hexdigest(),
            "native_translation_report": asdict(native_report),
            "stock_runtime_delta": stock_runtime_delta,
            "wps_runtime_contract": wps_runtime_contract,
            "static_catalog": catalog_receipt,
            **cache_provenance,
            "surface_fallback_radius_cells": (
                target.surface_fallback_radius_cells),
            "hierarchy_source_identity": hierarchy_identity,
        }

        hierarchy_source_identity = {
            "root_preparation": identity["source_identity"],
            "hierarchy": hierarchy_identity,
            "static_catalog": catalog_receipt,
            "wps_namelist_sha256": provenance["wps_namelist_sha256"],
            "native_namelist_input_sha256": namelist_sha,
            "stock_wrf_namelist_input_sha256": provenance[
                "stock_wrf_namelist_input_sha256"],
        }

        # The root preparation is the stage of this route that reads the
        # configuration, and it sealed the tree's [perturbation] deferral
        # into its identity.  The namelists this stage builds from spell
        # no such block, so the experiment the export sees takes it back
        # from there: the companion stock-WRF export then refuses a state
        # it cannot hold, as it does on every other source's tree, rather
        # than writing the unperturbed state beside a receipt that says
        # the bubbles are deferred.  The children are prepared exactly as
        # before; no preparation applies a bubble.
        root_deferral = identity["source_identity"].get(
            "initial_perturbation")
        # The prepared caches are bound to the experiment the forecast runs,
        # so a storm slot's lifecycle tables enter their identity (the tree
        # forecast refused d02 on "follow, retire, spawn" against a cache
        # prepared without them; box D 21:35Z).  Without a lifecycle config
        # this is native_exp itself.
        export_exp = lifecycle_exp if root_deferral is None else replace(
            lifecycle_exp,
            perturbation=deferred_perturbation_config(root_deferral))
        if root_head is not None:
            return _chained_hierarchy_tail(SimpleNamespace(
                started=started, preflight_seconds=preflight_seconds,
                restore_seconds=restore_seconds,
                snapshot_seconds=snapshot_seconds,
                root_preparation=Path(root_preparation), root_head=root_head,
                paths=paths, staging=staging, output_root=output_root,
                native_exp=native_exp, export_exp=export_exp, grids=grids,
                lifecycle_exp=lifecycle_exp,
                root=root, restored=restored, root_soil=root_soil,
                static_fields=static_fields, catalog=catalog,
                static_catalog=static_catalog, static_highres=static_highres,
                snapshots=snapshots, case_policy=case_policy,
                bridge_sha=bridge_sha, observed_source_sha=observed_source_sha,
                namelist_sha=namelist_sha, forcing_hours=forcing_hours,
                identity=identity, expected_identity=expected_identity,
                hierarchy_source_identity=hierarchy_source_identity,
                provenance=provenance, root_deferral=root_deferral,
                statics_corridor=statics_corridor, workers=workers,
                cpu_bridge=cpu_bridge,
                stock_wrf_namelist_input=Path(stock_wrf_namelist_input),
                cache_header=cache_header, valid_time=valid_time,
                stock_wrf_export=stock_wrf_export))
        hierarchy_started = time.perf_counter()
        # Sealed BEFORE the children initialize, so each corridor child is
        # born on its crop (plan call 5, 2.9.0).  The export refuses an
        # existing artifact folder, so the set waits beside it and moves in
        # after the artifact join.
        pending_corridor = staging / ".statics-corridor-pending"
        corridor_receipt = emit_statics_corridor_set(
            exp=lifecycle_exp, grids=grids, static_catalog=static_catalog,
            directory=pending_corridor,
            statics_corridor=statics_corridor, static_highres=static_highres)
        births = corridor_birth_statics(
            pending_corridor, corridor_receipt, lifecycle_exp)
        result = initialize_and_export_native_hierarchy(
            exp=export_exp, root_node=root,
            catalog=catalog_with_corridor_births(catalog, births),
            artifact_output=staging / "hierarchy-artifacts",
            wrf_output=staging / "wrf-native-input",
            root_initial_result=restored.initial_result,
            root_met=restored.met, root_soil=root_soil,
            root_static_fields=static_fields,
            root_boundaries=restored.boundaries,
            sfcp_to_sfcp=case_policy["sfcp_to_sfcp"],
            bridge_manifest_sha256=bridge_sha,
            source_manifest_sha256=observed_source_sha,
            namelist_sha256=namelist_sha,
            forcing_hours=forcing_hours,
            source_identity=hierarchy_source_identity,
            workers=workers, preprocess_backend="cpu", cpu_bridge=cpu_bridge,
            boundary_interval_seconds=3600,
            root_metadata=_root_preparation_metadata(
                restored.receipt["content_sha256"]),
            input_provenance=_export_provenance(
                provenance, restored.receipt["content_sha256"]),
            artifact_manifest_reference=(
                "../hierarchy-artifacts/domain-artifacts.json"),
            # The GPU runner consumes hierarchy-artifacts/;
            # wrf-native-input/ is the unchanged-WRF oracle set beside
            # it.  gfs_direct already asks for "optional" here, so a
            # state woof can integrate but the WRF file format cannot
            # represent is recorded as a REFUSED export manifest in the
            # receipt instead of destroying the preparation.  This route
            # was the last caller inheriting "required".  "off" is
            # `woof prep --no-stock-wrf-export` relayed (U-E6).
            stock_wrf_export=stock_wrf_export,
        )
        hierarchy_seconds = time.perf_counter() - hierarchy_started
        if getattr(native_exp, "nudging", None) is not None:
            from woof.spectral_nudging.prepared import series_path,selected,write_nested_analyses
            if selected(native_exp) is not None:
                source=series_path(root_preparation,native_exp.root.grid_id)
                destination=series_path(output_root,native_exp.root.grid_id)
                shutil.copytree(source,destination)
            # Child initialization retains only the parent's static ht/mub/phb
            # blend frame. Each child's atmosphere comes from its own source
            # snapshot, so the sealed root geometry is valid at every time.
            write_nested_analyses(output_root,native_exp,
                tuple(native_exp.start_time+timedelta(hours=h) for h in forcing_hours),
                observed_source_sha,catalog=catalog_with_corridor_births(catalog,births),
                root_grid=grids[0],root_at=lambda index:(restored.met,restored.initial_result),
                workers=workers,preprocess_backend='cpu',cpu_bridge=cpu_bridge,
                sfcp_to_sfcp=case_policy['sfcp_to_sfcp'])
        verify_overlay_sequence(snapshots)
        # AFTER the artifact join, and inside the same staging directory
        # the atomic publication renames: the hierarchy tree is already
        # sealed on its own terms, and the corridor set (sealed before the
        # children, above) moves in beside it under hierarchy-artifacts/,
        # which is where the tree runner looks for it whichever chain wrote
        # the bundle.  The GFS chain emits through the same function.
        if corridor_receipt is not None:
            move_statics_corridor_set(
                pending_corridor,
                staging / "hierarchy-artifacts" / STATICS_CORRIDOR_DIRNAME)
        stock_copy = staging / "namelist.input"
        shutil.copyfile(stock_wrf_namelist_input, stock_copy)
        payload = {
            "schema": SCHEMA,
            "status": "PASS",
            "valid_time": valid_time.isoformat(),
            "workers": workers,
            "preprocess_backend": "cpu",
            "domain_count": len(native_exp.domains),
            "forcing_hours": list(forcing_hours),
            "provenance": provenance,
            "root_preparation": _root_preparation_record(
                restored.receipt["content_sha256"]),
            "stock_wrf_namelist": {
                "path": stock_copy.name,
                "sha256": sha256_file(stock_copy),
            },
            "timing_seconds": {
                "verified_preflight": preflight_seconds,
                "restore_root_prepared_cache": restore_seconds,
                "load_initial_snapshot_and_static_catalog": snapshot_seconds,
                **dict(result.timings_seconds),
                "hierarchy_call_wall": hierarchy_seconds,
                "total": time.perf_counter() - started,
            },
            "artifact_receipt": dict(result.artifacts.receipt),
            "wrf_manifest": dict(result.wrf_manifest),
            # Present only when the configuration carries [perturbation]:
            # the bubbles the tree runner applies at start, deferred by
            # the root preparation, which is the stage of this route that
            # reads the configuration (woof.hrrr_configuration
            # root_perturbation_deferral).  Relayed from the sealed root
            # identity, which every domain's cache also carries under
            # root_preparation.  Absent, the receipt is byte-for-byte
            # what it always was.
            **({"initial_perturbation": dict(root_deferral)}
               if root_deferral is not None else {}),
            # Present only when this preparation opted in: the sealed
            # statics-corridor set, digest-bound here the way every
            # other sealed artifact of this bundle is, and read from
            # here by woof-prepared-tree-forecast's preflight.  Absent,
            # the receipt is byte-for-byte what it always was.
            **({"statics_corridor": dict(corridor_receipt)}
               if corridor_receipt is not None else {}),
            # WHICH AEROSOL SOURCE THE ROOT PREPARATION USED.  This stage
            # publishes a hierarchy built ON that root state, so the fact
            # is this bundle's as much as the root's, and every forecast
            # launched from these artifacts inherits it.  Read from the
            # root prepared cache's own metadata rather than re-resolved:
            # re-resolving here would be a second resolution path over the
            # same config field, which is how a run reads one dataset and
            # reports another.  Empty -- and the receipt byte-for-byte
            # unchanged -- for every scheme with no aerosol fields.
            **_aerosol_entry(cache_header, native_exp, paths),
        }
        receipt = staging / RECEIPT_NAME
        temporary = receipt.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
            + "\n", encoding="utf-8")
        os.replace(temporary, receipt)
        os.replace(staging, output_root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-preparation", type=Path, required=True)
    parser.add_argument("--root-domain-spec", type=Path, required=True)
    parser.add_argument("--wps-namelist", type=Path, required=True)
    parser.add_argument("--namelist-input", type=Path, required=True)
    parser.add_argument(
        "--stock-wrf-namelist-input", type=Path, required=True)
    parser.add_argument("--geog-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument(
        "--cycle",
        help="the HRRR CYCLE the sealed root preparation was built from, "
             "YYYY-MM-DD_HH:MM:SS (UTC).  Model time zero -- the instant "
             "this stage checks the namelist and the sealed cache against "
             "-- is cycle + --forecast-start-hour")
    parser.add_argument(
        "--forecast-start-hour", type=int, default=0,
        help="absolute HRRR cycle-relative lead the root preparation "
             "began at; the same value passed to tools/prepare_hrrr_wrf.py")
    parser.add_argument(
        "--valid-time",
        help="deprecated: on THIS command --valid-time meant model time "
             "zero, not the cycle (the two differ by the lead).  Accepted "
             "unchanged for v1.4.0 scripts; use --cycle with "
             "--forecast-start-hour")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--ack", action="append", default=[], metavar="ID",
        help="declared-experiment acknowledgement id, forwarded into the "
             "namelist import (a WRF namelist has no spelling for a woof "
             "governance declaration, so a profile that requires one -- "
             "e.g. the shortwave-only suites' "
             "constant-downward-longwave-v1 -- could never prepare a "
             "nested tree without this flag); repeatable, exactly as "
             "tools/prepare_hrrr_wrf.py spells it")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--lifecycle-config", type=Path, default=None, metavar="TOML",
        help="the experiment config whose [[domain]] spawn/follow/retire/"
             "rearm tables the namelists cannot carry (storm nests)")
    parser.add_argument("--cpu-preprocess-bridge", type=Path)
    # Spelled exactly as the GFS door spells it
    # (woof/gfs_direct.py), because one runner consumes both bundles
    # and its refusal names this flag as the remedy whichever chain
    # prepared the tree.
    parser.add_argument(
        "--statics-corridor", nargs="?", const="all", default=None,
        metavar="GRID_IDS",
        help="also seal child-resolution statics over the ground each "
             "child can reach (the moving-nest corridor); bare flag covers "
             "every child domain, or pass comma-separated child grid ids "
             "(e.g. 2,3).  Required before the prepared tree runner will "
             "honor a [relocation] follow source; omitted, the bundle is "
             "byte-for-byte unchanged")
    parser.add_argument(
        "--no-stock-wrf-export", dest="stock_wrf_export",
        action="store_const", const="off", default="optional",
        help="prepare the forecast only, and do not attempt the bonus "
             "unchanged-WRF wrfinput/wrfbdy export")
    return parser


@owns_source_coverage_refusal
def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    from woof import explain
    from woof.hrrr_forecast import resolve_cycle_flags

    if args.forecast_start_hour < 0:
        raise ValueError(
            "--forecast-start-hour must be a nonnegative forecast lead")
    if args.valid_time is not None and args.forecast_start_hour:
        raise ValueError(
            "--forecast-start-hour needs --cycle: on this command "
            "--valid-time is model time zero already, so adding a lead to "
            "it would move the clock twice.  Pass --cycle CYCLE "
            "--forecast-start-hour K instead.")
    # `--valid-time` on this stage always meant model time zero; `--cycle`
    # means the cycle and has the lead added to it.  Both land on the same
    # instant, which is the only thing the checks below compare.
    instant, from_valid_time = resolve_cycle_flags(
        args.cycle, args.valid_time, tool="hrrr_hierarchy_direct",
        legacy_means="model time zero (cycle + lead)", warn=explain.warn)
    valid_time = (
        instant if from_valid_time
        else instant + timedelta(hours=args.forecast_start_hour))
    statics_corridor = args.statics_corridor
    if statics_corridor is not None and statics_corridor != "all":
        try:
            statics_corridor = tuple(
                int(part) for part in statics_corridor.split(",") if part)
        except ValueError:
            raise ValueError(
                "--statics-corridor accepts 'all' or comma-separated "
                f"child grid ids, got {args.statics_corridor!r}") from None
    payload = prepare_hrrr_hierarchy(
        root_preparation=args.root_preparation,
        root_domain_spec=args.root_domain_spec,
        wps_namelist=args.wps_namelist,
        namelist_input=args.namelist_input,
        stock_wrf_namelist_input=args.stock_wrf_namelist_input,
        geog_root=args.geog_root,
        source_manifest=args.source_manifest,
        source_manifest_sha256=args.source_manifest_sha256,
        valid_time=valid_time, output_root=args.output_root,
        acknowledgements=tuple(args.ack),
        workers=args.workers, cpu_bridge=args.cpu_preprocess_bridge,
        statics_corridor=statics_corridor,
        lifecycle_config=args.lifecycle_config,
        stock_wrf_export=args.stock_wrf_export,
    )
    # The forecast runner takes --preparation-receipt-sha256 over the
    # receipt this stage just wrote, and the emitted chain tells the
    # reader the hierarchy prints it.  It did not: the one placeholder
    # in that chain no stage filled, so the printed route could not be
    # walked without hashing a file by hand.  Print it here, from the
    # published receipt rather than from the payload, so the digest is
    # of the bytes the next stage will read.
    # The export slot is a READY manifest with a `files` inventory OR the
    # NOT_REQUESTED/REFUSED document that says why there is none
    # (woof/native_hierarchy.py: STOCK_WRF_EXPORT_MODES).  Reading
    # ["files"] unconditionally made the refused shape a KeyError here,
    # which threw away the prepared hierarchy that the "optional" mode
    # exists to keep -- the same failure the refusal being uncatchable
    # caused one layer down.  Report the slot as it is.
    wrf_manifest = payload["wrf_manifest"]
    print(json.dumps({
        "status": payload["status"],
        "output_root": str(args.output_root.resolve()),
        "workers": payload["workers"],
        "timing_seconds": payload["timing_seconds"],
        "wrf_export_status": wrf_manifest.get("status"),
        "wrf_files": wrf_manifest.get("files"),
        "wrf_export_refusal": wrf_manifest.get("reason"),
        "preparation_receipt_sha256": sha256_file(
            args.output_root / "receipt.json"),
    }, indent=2, sort_keys=True, allow_nan=False))
    corridor = payload.get("statics_corridor")
    if isinstance(corridor, dict):
        # Size accuracy at the door, in the GFS door's own words (one
        # line, written once): the corridor's cost and the ground it
        # covers are stated where they are paid.
        from woof.static.corridor import corridor_summary_line
        for label, entry in sorted(corridor.get("domains", {}).items()):
            print(corridor_summary_line(label, entry), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
